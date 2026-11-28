// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632), after the measuring tool CAPA-85 wrote. Not CGBJ code, but part
// of crates/xcl-cgbj, which is CGBJ's shader translator (GPL-3.0-or-later, see ../../LICENSE and
// ../../PROVENANCE.md), so it carries that licence.
//
//! Census of what the translator does with folders of console shader containers.
//!
//! xcl-cgbj-census [--wave exact|approx] [--host-wave 32|64] [--binding victory|xcl]
//!                 [--jobs N] [--timeout S] [--dump DIR] DIR...
//!
//! Every *.bin file in the folders is translated in a child process of its own, so a panic or
//! a hang costs one row. Rows go to stdout as tab-separated values and a summary goes to
//! stderr. Nothing of a shader is printed but the file name it was given under.
//!
//! --wave exact (the default) is xsemu's policy (xcl_cgbj::xcl::set_wave_exact): only forms
//! that keep the console's results. --wave approx is CGBJ's own behaviour, without its list
//! of two title programs. VICTORY_* variables of the caller are not passed on, so a row
//! depends on the flags alone.
//!
//! --binding victory (the default) is CGBJ's own resource binding. --binding xcl translates
//! under the XCL binding (xcl_cgbj::xcl::translate_compute, and translate_pixel without links
//! for pixel programs, so their inputs take the rows xsemu's own translator gives them),
//! which takes compute and pixel programs of either wave size, a wave32 one only where
//! --host-wave is 32, and adds what the binding emitted (XclStats) as a last column.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xcl_cgbj::dxbc::{Container, FourCC};
use xcl_cgbj::emit::{self, census};
use xcl_cgbj::native::HwStage;

/// Stage, wave64 and code bytes of the program the translator picks from a container.
fn program(bytes: &[u8]) -> Option<(String, bool, usize)> {
    let c = Container::parse(bytes).ok()?;
    let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
    let p = xcl_cgbj::native_programs(part.data)
        .ok()?
        .into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))?;
    Some((format!("{:?}", p.stage), p.wave64, p.code.len()))
}

type Form = Option<(String, xcl_cgbj::xcl::FormKind)>;

/// What a translation gave: the DXIL, the wave form it took, and under the XCL binding its
/// statistics as text.
fn translate(bytes: &[u8], binding: &str) -> Result<(Vec<u8>, Form, String), xcl_cgbj::ShaderError> {
    if binding == "victory" {
        return emit::translate(bytes).map(|h| (h.dxil, xcl_cgbj::xcl::take_form(), String::new()));
    }
    let cfg = xcl_cgbj::xcl::XclConfig::default();
    let s = match program(bytes) {
        Some((stage, _, _)) if stage == "Ps" => {
            let p = xcl_cgbj::xcl::translate_pixel(bytes, &cfg, &[])?;
            xcl_cgbj::xcl::XclShader { dxil: p.dxil, samplers: p.samplers, threads: [0; 3], form: None, stats: p.stats }
        }
        _ => xcl_cgbj::xcl::translate_compute(bytes, &cfg)?,
    };

    let t = s.stats;
    let stats = format!(
        "instructions={},resolves_single={},resolves_multi={},record_loads={},scalar_loads={},buffer_accesses={},global_accesses={},image_ops={},static_samplers={},heap_samplers={},page_groups={},grouped_loads={}",
        t.instructions,
        t.resolves_single,
        t.resolves_multi,
        t.record_loads,
        t.scalar_loads,
        t.buffer_accesses,
        t.global_accesses,
        t.image_ops,
        t.static_samplers,
        t.heap_samplers,
        t.page_groups,
        t.grouped_loads
    );
    Ok((s.dxil, s.form, stats))
}

fn child(path: &str, exact: bool, host_wave: u32, dump: Option<&str>, binding: &str) {
    // An NVIDIA host as CGBJ's own host code sets it up: not RDNA, the view heap capped at
    // 1,000,000 with the last slot null and no raw twins.
    emit::set_host_max_wave(host_wave);
    emit::set_host_rdna(false);
    emit::set_view_heap_limit(999_999);
    xcl_cgbj::xcl::set_wave_exact(exact);
    let bytes = std::fs::read(path).unwrap_or_default();
    let (stage, wave64, code) = program(&bytes).unwrap_or(("?".into(), false, 0));
    let t = Instant::now();
    let r = translate(&bytes, binding);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    match r {
        Ok((dxil, form, stats)) => {
            let counts = census::take(&dxil)
                .map(|c| {
                    census::KINDS
                        .iter()
                        .zip(c.counts.iter())
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let (form, kind) = match form {
                Some((label, kind)) => (label, kind.name()),
                None => ("as it stood".to_string(), "exact"),
            };
            println!("RESULT\tok\t{stage}\t{wave64}\t{code}\t{}\t{ms:.1}\t-\t{form}\t{kind}\t{counts}\t{stats}", dxil.len());
            if let Some(dir) = dump {
                let name = Path::new(path).file_stem().unwrap_or_default().to_string_lossy().to_string();
                if let Err(e) = std::fs::write(format!("{dir}/{name}.dxil"), &dxil) {
                    eprintln!("could not write the DXIL of {name}: {e}");
                }
            }
        }
        Err(e) => {
            let why = e.to_string().replace(['\t', '\n'], " ");
            println!("RESULT\trefused\t{stage}\t{wave64}\t{code}\t0\t{ms:.1}\t{why}\t-\t-\t\t");
        }
    }
}

struct Row {
    name: String,
    result: String,
    stage: String,
    wave64: String,
    code: u64,
    dxil: u64,
    ms: String,
    reason: String,
    form: String,
    kind: String,
    census: String,
    stats: String,
}

/// A refusal without its code offset, so equal reasons count together.
fn reason_class(reason: &str) -> String {
    match (reason.find(" at 0x"), reason.find(": ")) {
        (Some(at), Some(colon)) if colon > at => format!("{}{}", &reason[..at], &reason[colon..]),
        _ => reason.to_string(),
    }
}

fn one(exe: &str, f: &Path, wave: &str, host_wave: u32, dump: Option<&str>, binding: &str, timeout: Duration) -> Row {
    let name = f.file_stem().unwrap_or_default().to_string_lossy().to_string();
    // (for a row whose child gives no result)
    let (stage, wave64, code) = std::fs::read(f)
        .ok()
        .and_then(|b| std::panic::catch_unwind(|| program(&b)).ok().flatten())
        .unwrap_or(("?".into(), false, 0));
    let mut row = Row {
        name,
        result: "crash".into(),
        stage,
        wave64: wave64.to_string(),
        code: code as u64,
        dxil: 0,
        ms: "0".into(),
        reason: String::new(),
        form: "-".into(),
        kind: "-".into(),
        census: String::new(),
        stats: String::new(),
    };
    let mut cmd = Command::new(exe);
    cmd.arg("--one").arg(f).arg(wave).arg(host_wave.to_string()).arg(dump.unwrap_or("")).arg(binding);
    for (k, _) in std::env::vars_os() {
        if k.to_string_lossy().starts_with("VICTORY_") {
            cmd.env_remove(k);
        }
    }
    cmd.env("VICTORY_SHADER_CENSUS", "1");
    let start = Instant::now();
    let Ok(mut ch) = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() else {
        row.reason = "could not start the child".into();
        return row;
    };
    let (mut so, mut se) = (ch.stdout.take().unwrap(), ch.stderr.take().unwrap());
    let ho = std::thread::spawn(move || {
        let mut s = String::new();
        so.read_to_string(&mut s).ok();
        s
    });
    let he = std::thread::spawn(move || {
        let mut s = String::new();
        se.read_to_string(&mut s).ok();
        s
    });
    let status = loop {
        if let Ok(Some(s)) = ch.try_wait() {
            break Some(s);
        }
        if start.elapsed() > timeout {
            ch.kill().ok();
            ch.wait().ok();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let (out, err) = (ho.join().unwrap_or_default(), he.join().unwrap_or_default());
    let Some(line) = out.lines().find(|l| l.starts_with("RESULT")) else {
        row.ms = start.elapsed().as_millis().to_string();
        match status {
            None => {
                row.result = "timeout".into();
                row.reason = format!("no result in {} s", timeout.as_secs());
            }
            Some(s) => {
                let last = err.lines().rev().find(|l| l.contains("panicked")).or(err.lines().rev().find(|l| !l.is_empty()));
                row.reason = format!("{s}: {}", last.unwrap_or("").replace('\t', " "));
            }
        }
        return row;
    };
    // RESULT result stage wave64 code dxil ms reason form kind census stats
    let cols: Vec<&str> = line.split('\t').collect();
    let col = |i: usize| cols.get(i).copied().unwrap_or("").to_string();
    (row.result, row.stage, row.wave64) = (col(1), col(2), col(3));
    (row.code, row.dxil) = (col(4).parse().unwrap_or(0), col(5).parse().unwrap_or(0));
    (row.ms, row.reason, row.form, row.kind, row.census) = (col(6), col(7), col(8), col(9), col(10));
    row.stats = col(11);
    row
}

fn usage() -> ! {
    eprintln!("usage: xcl-cgbj-census [--wave exact|approx] [--host-wave 32|64] [--binding victory|xcl] [--jobs N] [--timeout S] [--dump DIR] DIR...");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 5 && args[1] == "--one" {
        let dump = args.get(5).filter(|d| !d.is_empty()).map(String::as_str);
        return child(&args[2], args[3] == "exact", args[4].parse().unwrap_or(32), dump, args.get(6).map_or("victory", String::as_str));
    }
    let mut binding = "victory".to_string();
    let (mut wave, mut host_wave, mut jobs, mut timeout, mut dump) = ("exact".to_string(), 32u32, 4usize, 300u64, None::<String>);
    let mut dirs = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned().unwrap_or_else(|| usage());
        match a.as_str() {
            "--wave" => wave = value(),
            "--host-wave" => host_wave = value().parse().unwrap_or_else(|_| usage()),
            "--jobs" => jobs = value().parse().unwrap_or_else(|_| usage()),
            "--timeout" => timeout = value().parse().unwrap_or_else(|_| usage()),
            "--dump" => dump = Some(value()),
            "--binding" => binding = value(),
            flag if flag.starts_with('-') => usage(),
            _ => dirs.push(a.clone()),
        }
    }
    if dirs.is_empty()
        || !matches!(wave.as_str(), "exact" | "approx")
        || !matches!(host_wave, 32 | 64)
        || !matches!(binding.as_str(), "victory" | "xcl")
    {
        usage();
    }
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("{d}: {e}"));
    }
    let mut files: Vec<PathBuf> = dirs
        .iter()
        .flat_map(|d| std::fs::read_dir(d).unwrap_or_else(|e| panic!("{d}: {e}")).flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    let work = Arc::new(Mutex::new(files.into_iter().enumerate().collect::<Vec<_>>()));
    let rows = Arc::new(Mutex::new(Vec::<(usize, Row)>::new()));
    let threads: Vec<_> = (0..jobs.max(1))
        .map(|_| {
            let (work, rows, exe, wave, dump) = (work.clone(), rows.clone(), args[0].clone(), wave.clone(), dump.clone());
            let binding = binding.clone();
            std::thread::spawn(move || {
                loop {
                    let Some((i, f)) = work.lock().unwrap().pop() else { break };
                    let row = one(&exe, &f, &wave, host_wave, dump.as_deref(), &binding, Duration::from_secs(timeout));
                    rows.lock().unwrap().push((i, row));
                }
            })
        })
        .collect();
    for t in threads {
        t.join().ok();
    }
    let mut rows = std::mem::take(&mut *rows.lock().unwrap());
    rows.sort_by_key(|r| r.0);
    println!("name\tresult\tstage\twave64\tcode_bytes\tdxil_bytes\tms\treason\twave_form\tform_kind\tcensus\txcl_stats");
    let (mut by_stage, mut by_form, mut by_reason) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (_, r) in &rows {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.name,
            r.result,
            r.stage,
            r.wave64,
            r.code,
            r.dxil,
            r.ms,
            if r.reason.is_empty() { "-" } else { &r.reason },
            r.form,
            r.kind,
            r.census,
            if r.stats.is_empty() { "-" } else { &r.stats }
        );
        let s = by_stage.entry((r.stage.clone(), r.wave64.clone())).or_insert((0u32, 0u32, 0u64, 0u64, 0u64));
        s.0 += 1;
        if r.result == "ok" {
            s.1 += 1;
            s.2 += r.code;
            s.3 += r.dxil;
            s.4 = s.4.max(r.dxil);
            if r.form != "as it stood" {
                *by_form.entry((r.stage.clone(), r.wave64.clone(), r.kind.clone(), r.form.clone())).or_insert(0u32) += 1;
            }
        } else {
            *by_reason.entry((r.result.clone(), r.stage.clone(), r.wave64.clone(), reason_class(&r.reason))).or_insert(0u32) += 1;
        }
    }
    eprintln!("census: {} programs, --wave {wave}, host wave {host_wave}, --binding {binding}", rows.len());
    eprintln!("stage\twave64\tprograms\ttranslated\tcode_bytes_of_translated\tdxil_bytes\tlargest_dxil");
    for ((stage, w), (n, ok, code, dxil, max)) in &by_stage {
        eprintln!("{stage}\t{w}\t{n}\t{ok}\t{code}\t{dxil}\t{max}");
    }
    eprintln!("translated in another form than the program as it stood:");
    for ((stage, w, kind, form), n) in &by_form {
        eprintln!("{n}\t{stage}\twave64={w}\t{kind}\t{form}");
    }
    eprintln!("not translated:");
    for ((result, stage, w, why), n) in &by_reason {
        eprintln!("{n}\t{result}\t{stage}\twave64={w}\t{why}");
    }
}
