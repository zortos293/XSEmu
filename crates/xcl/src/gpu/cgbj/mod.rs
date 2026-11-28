//! A second translator for compute and pixel programs: crates/xcl-cgbj under its XCL binding
//! (CAPA-86, M1 compute, M2a pixel).
//!
//! Off unless `XCL_SHADER_EMITTER` asks for it. Then real.rs tries [`compute`] or [`pixel`]
//! before today's translator, and a program CGBJ refuses goes to today's path. This folder is
//! outside the translator fingerprint (crates/xcl/build.rs), so changing it keeps the HLSL caches
//! warm.
//!
//! Switches (read once):
//! - `XCL_SHADER_EMITTER`: `xcl` (default), `cgbj`, or `cs:cgbj,ps:cgbj,vs:xcl,ms:xcl`. `cs` and `ps` are honoured.
//! - `XCL_SHADER_EMITTER_ONLY` / `XCL_SHADER_EMITTER_SKIP`: blob hashes (16 hex digits, as logged).
//! - `XCL_CGBJ_REFUSED`: `xcl` (default) or `skip` (refused programs stay untranslated, A/B only).
//! - `XCL_CGBJ_WAVE`: `exact` (default) or `approx`.
//! - `XCL_CGBJ_VALIDATE`: `1` (default) or `0`.
//! - `XCL_TRANSLATED_CACHE=0`: no disk cache (the memory map stays).

pub(crate) mod cache;
mod switches;
mod validate;

use crate::gpu::recomp::pso::Compiled;
use cache::{Entry, Key};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use switches::{Emitter, Refused, Switches};
use xcl_cgbj::xcl::{XclConfig, BINDING_VERSION};

/// What [`compute`] or [`pixel`] came to.
pub enum Outcome {
    Compiled(Compiled),
    /// Not CGBJ's program: today's translator takes it.
    Fallback,
    /// Inside the cached-only probe with nothing cached: `dxc::defer` was called.
    Deferred,
    /// Refused with `XCL_CGBJ_REFUSED=skip`: it stays untranslated.
    Skipped,
}

/// The stage of a program: its switch, counters and log lines.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Compute,
    Pixel,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Compute => "compute",
            Kind::Pixel => "pixel",
        }
    }

    fn counters(self) -> &'static Counters {
        static C: [Counters; 2] = [Counters::new(), Counters::new()];
        &C[self as usize]
    }
}

fn switches() -> Option<&'static Switches> {
    static S: OnceLock<Option<Switches>> = OnceLock::new();
    S.get_or_init(|| match switches::parse(|name| std::env::var(name).ok()) {
        Ok(s) => {
            for stage in &s.unavailable {
                crate::xlog_warn!("XCL_SHADER_EMITTER asks for cgbj for {stage}: not available yet, xcl translates it");
            }
            let stages: Vec<&str> = [(s.compute, "compute"), (s.pixel, "pixel")].into_iter().filter(|(e, _)| *e == Emitter::Cgbj).map(|(_, n)| n).collect();
            (!stages.is_empty()).then(|| {
                crate::xlog_info!(
                    "cgbj: {} programs go to xcl-cgbj first (binding v{BINDING_VERSION}, wave {}, validate {}, refused -> {}{}{})",
                    stages.join(" and "),
                    if s.wave_exact { "exact" } else { "approx" },
                    s.validate as u8,
                    if s.refused == Refused::Xcl { "xcl" } else { "skip" },
                    s.only.as_ref().map_or_else(String::new, |o| format!(", only {} programs", o.len())),
                    if s.skip.is_empty() { String::new() } else { format!(", never {} programs", s.skip.len()) }
                );
                s
            })
        }
        Err(e) => {
            crate::xlog_warn!("cgbj: {e}; xcl translates every program");
            None
        }
    })
    .as_ref()
}

/// Host facts, set once before the first translation.
struct Host {
    hw_wave: u32,
    /// Hash of the translator's fingerprint, the binding and the validator; with the per-call
    /// facts it makes the cache identity.
    base: u64,
    /// The disk cache's folder; `None` with `XCL_TRANSLATED_CACHE=0`.
    dir: Option<PathBuf>,
}

fn host(s: &Switches, hw_wave: u32) -> &'static Host {
    static H: OnceLock<Host> = OnceLock::new();
    H.get_or_init(|| {
        xcl_cgbj::emit::set_host_max_wave(hw_wave);
        // Off: the emulator does not know whether the GPU is RDNA (pso.rs keeps only the PCI
        // vendor, privately), and off keeps the console's float results on every GPU.
        xcl_cgbj::emit::set_host_rdna(false);
        xcl_cgbj::xcl::set_wave_exact(s.wave_exact);
        xcl_cgbj::xcl::set_d16_half_bits(std::env::var_os("XCL_RECOMP_D16_HALF_BITS").is_some());
        let fingerprint = xcl_cgbj::emit::translation_fingerprint();
        let validator = if s.validate {
            let _ = crate::gpu::recomp::dxc::location();
            serde_json::to_string(&crate::gpu::recomp::dxc::toolchain()).unwrap_or_default()
        } else {
            "no validation".into()
        };
        let base = cache::identity(&[b"xcl-cgbj-cache-v1", &fingerprint, &BINDING_VERSION.to_le_bytes(), validator.as_bytes()]);
        let disk = std::env::var("XCL_TRANSLATED_CACHE").map_or(true, |v| v != "0");
        Host { hw_wave, base, dir: disk.then(crate::gpu::recomp::dxc::cache_dir) }
    })
}

struct Counters {
    tried: AtomicU64,
    produced: AtomicU64,
    from_cache: AtomicU64,
    invalid: AtomicU64,
    dxil_bytes: AtomicU64,
    translate_ms: AtomicU64,
    refused: Mutex<BTreeMap<String, u64>>,
}

impl Counters {
    const fn new() -> Counters {
        Counters {
            tried: AtomicU64::new(0),
            produced: AtomicU64::new(0),
            from_cache: AtomicU64::new(0),
            invalid: AtomicU64::new(0),
            dxil_bytes: AtomicU64::new(0),
            translate_ms: AtomicU64::new(0),
            refused: Mutex::new(BTreeMap::new()),
        }
    }

    /// Counts a refusal; true the first time this reason is seen.
    fn refusal(&self, reason: &str) -> bool {
        let Ok(mut m) = self.refused.lock() else { return false };
        let n = m.entry(reason.to_string()).or_default();
        *n += 1;
        *n == 1
    }
}

/// Translate the console compute program `blob` with CGBJ, or say why not. `hw_wave` is 32
/// when the device runs only 32-lane waves (real.rs `wave32_only`), else 64. With the
/// switches off this returns `Fallback` at once: no translation, no file, no log.
pub fn compute(blob: &[u8], sampler_base: u32, hw_wave: u32) -> Outcome {
    let Some(s) = switches() else { return Outcome::Fallback };
    let fnv = crate::gpu::shader::fnv64(blob);
    if !s.wants(s.compute, fnv) {
        return Outcome::Fallback;
    }
    let thread_limits = crate::gpu::recomp::dispatch_thread_dimensions();
    produce(Kind::Compute, s, blob, fnv, hw_wave, Key::new(0, fnv, blob, sampler_base, 0), thread_limits, &[])
}

/// Translate the console pixel program `blob` with CGBJ, or say why not. Its static samplers
/// start at `sampler_base` (the vertex program's count). Its inputs take the rows of the
/// outputs of `vs` (the pipeline's translated vertex program, DXIL) and of the rect-list GS
/// `rect_gs` gives, when the pipeline gets one.
///
/// `hw_wave` is 32 when the device runs only 32-lane waves (real.rs `wave32_only`), else 64.
/// A wave64 program translates either way: its cross-lane code is host-wave operations and
/// reads no lane count, as xsemu's own forms for a wave64 graphics program do (see
/// `xcl_cgbj::xcl::translate_pixel`). A wave32 program translates only on a device of
/// 32-lane waves: in a 64-lane wave the other console wave shares the host wave and joins its
/// ballots and reductions. With the switches off this returns `Fallback` at once and calls no
/// closure.
pub fn pixel(blob: &[u8], sampler_base: u32, hw_wave: u32, vs: &[u8], rect_gs: impl FnOnce() -> Option<Vec<u8>>) -> Outcome {
    let Some(s) = switches() else { return Outcome::Fallback };
    let fnv = crate::gpu::shader::fnv64(blob);
    if !s.wants_pixel(fnv) {
        return Outcome::Fallback;
    }
    let gs = rect_gs();
    let links: Vec<&[u8]> = std::iter::once(vs).chain(gs.as_deref()).collect();
    let signatures: Vec<Vec<u8>> = links.iter().map(|l| output_signature(l)).collect();
    let link = cache::identity(&signatures.iter().map(Vec::as_slice).collect::<Vec<_>>());
    produce(Kind::Pixel, s, blob, fnv, hw_wave, Key::new(0, fnv, blob, sampler_base, link), false, &links)
}

/// The output signature part of a DXIL container (the container itself when it has none).
fn output_signature(dxil: &[u8]) -> Vec<u8> {
    let part = xcl_cgbj::Container::parse(dxil).ok().and_then(|c| c.part(xcl_cgbj::FourCC::OSG1).map(|p| p.data.to_vec()));
    part.unwrap_or_else(|| dxil.to_vec())
}

/// The cache, then CGBJ's translation, for a program `compute` or `pixel` took. `key` is
/// whole but for its identity.
#[allow(clippy::too_many_arguments)]
fn produce(kind: Kind, s: &Switches, blob: &[u8], fnv: u64, hw_wave: u32, key: Key, thread_limits: bool, links: &[&[u8]]) -> Outcome {
    let counters = kind.counters();
    let host = host(s, hw_wave);
    if hw_wave != host.hw_wave {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| crate::xlog_warn!("cgbj: a device with {hw_wave}-lane waves after one with {}: xcl translates its programs", host.hw_wave));
        return Outcome::Fallback;
    }
    let refused_outcome = if s.refused == Refused::Xcl { Outcome::Fallback } else { Outcome::Skipped };
    let windows = crate::gpu::recomp::pso::mem_windows();
    let ident = cache::identity(&[&host.base.to_le_bytes(), &windows.to_le_bytes(), &[thread_limits as u8], &hw_wave.to_le_bytes()]);
    let key = Key { ident, ..key };
    let cached = cache::remembered(&key).or_else(|| {
        let e = cache::load(host.dir.as_deref()?, &key)?;
        cache::remember(key, &e);
        Some(e)
    });
    match cached {
        Some(e) => {
            counters.tried.fetch_add(1, Relaxed);
            counters.from_cache.fetch_add(1, Relaxed);
            return match e {
                Entry::Good(c) => Outcome::Compiled(c),
                Entry::Refused(reason) => {
                    counters.refusal(&reason);
                    refused_outcome
                }
            };
        }
        None if crate::gpu::recomp::dxc::in_cached_only() => {
            crate::gpu::recomp::dxc::defer();
            return Outcome::Deferred;
        }
        None => {}
    }
    counters.tried.fetch_add(1, Relaxed);
    if let Err(why) = s.validate.then(validate::ready).unwrap_or(Ok(())) {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| crate::xlog_warn!("cgbj: no validator ({why}); xcl translates programs until there is one"));
        counters.refusal("validator unavailable");
        return refused_outcome;
    }
    let cfg = XclConfig { mem_windows: windows, thread_limits, sampler_base: key.sampler_base };
    let keep = |e: &Entry| {
        cache::remember(key, e);
        if let Some(dir) = &host.dir {
            cache::store(dir, &key, e);
        }
    };
    let started = std::time::Instant::now();
    match translate(kind, blob, &cfg, links, s.validate) {
        Ok((c, form)) => {
            let ms = started.elapsed().as_millis() as u64;
            counters.produced.fetch_add(1, Relaxed);
            counters.dxil_bytes.fetch_add(c.dxil.len() as u64, Relaxed);
            counters.translate_ms.fetch_add(ms, Relaxed);
            crate::xlog_info!(
                "[SHADER] {} {fnv:016x}: cgbj, {} B DXIL, {ms} ms, {} static samplers{}",
                kind.name(),
                c.dxil.len(),
                c.samplers.len(),
                form.map_or_else(String::new, |(label, kind)| format!(", form {label} ({})", kind.name()))
            );
            keep(&Entry::Good(c.clone()));
            Outcome::Compiled(c)
        }
        Err(Refusal::Transient(why)) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| crate::xlog_warn!("cgbj: {why}; programs go to xcl until it works"));
            counters.refusal("validator unavailable");
            refused_outcome
        }
        Err(Refusal::Kept { reason, detail }) => {
            if counters.refusal(&reason) {
                crate::xlog_info!("[SHADER] {} {fnv:016x}: cgbj refused ({detail}), first of its kind; {}", kind.name(), if s.refused == Refused::Xcl { "xcl translates it" } else { "left untranslated" });
            }
            keep(&Entry::Refused(reason));
            refused_outcome
        }
    }
}

enum Refusal {
    /// Cached: the same build refuses the same program again.
    Kept { reason: String, detail: String },
    /// Not the program's fault (no validator): not cached.
    Transient(String),
}

type Made = (Compiled, Option<(String, xcl_cgbj::xcl::FormKind)>, Option<[u32; 3]>);

/// CGBJ's translation on a thread of its own with a 64 MiB stack (as dxc.rs runs its direct
/// backend), then the checks a container needs before it is used.
fn translate(kind: Kind, blob: &[u8], cfg: &XclConfig, links: &[&[u8]], validate: bool) -> Result<(Compiled, Option<(String, xcl_cgbj::xcl::FormKind)>), Refusal> {
    let kept = |reason: &str, detail: String| Refusal::Kept { reason: reason.to_string(), detail };
    let work = || -> Result<Made, xcl_cgbj::ShaderError> {
        Ok(match kind {
            Kind::Compute => {
                let s = xcl_cgbj::xcl::translate_compute(blob, cfg)?;
                (Compiled { dxil: s.dxil, samplers: s.samplers }, s.form, Some(s.threads))
            }
            Kind::Pixel => {
                let p = xcl_cgbj::xcl::translate_pixel(blob, cfg, links)?;
                (Compiled { dxil: p.dxil, samplers: p.samplers }, None, None)
            }
        })
    };
    let run = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new().name("xcl-cgbj".into()).stack_size(64 << 20).spawn_scoped(scope, work);
        worker.map(|w| w.join())
    });
    let (compiled, form, threads) = match run {
        Err(e) => return Err(Refusal::Transient(format!("no translation thread ({e})"))),
        Ok(Err(_)) => return Err(kept("panic", "the translator panicked".into())),
        Ok(Ok(Err(e))) => {
            use xcl_cgbj::ShaderError as E;
            let reason = match &e {
                E::Unsupported { what, .. } => what.to_string(),
                E::Isa { reason, .. } => reason.to_string(),
                E::Malformed(m) => m.to_string(),
                other => other.to_string(),
            };
            return Err(kept(&reason, e.to_string()));
        }
        Ok(Ok(Ok(made))) => made,
    };
    if let (Some(threads), Some(console)) = (threads, crate::gpu::recomp::pso::compute_interface(blob).map(|i| i.threads)) {
        let packed = form.as_ref().is_some_and(|(label, _)| label == xcl_cgbj::xcl::PACKED_FORM);
        if threads != if packed { xcl_cgbj::xcl::packed_threads(console) } else { console } {
            return Err(kept("numthreads differ from the console's", format!("numthreads {threads:?}, console {console:?}")));
        }
    }
    if validate {
        match validate::check(&compiled.dxil) {
            Ok(()) => {}
            Err(validate::Failure::Unavailable(why)) => return Err(Refusal::Transient(why)),
            Err(validate::Failure::Invalid(msg)) => {
                let n = kind.counters().invalid.fetch_add(1, Relaxed);
                if n < 8 {
                    let head: Vec<&str> = msg.lines().take(6).collect();
                    crate::xlog_warn!("cgbj: dxil.dll rejects a {} container ({} B): {}", kind.name(), compiled.dxil.len(), head.join(" | "));
                }
                return Err(kept("invalid", "dxil.dll rejects the container".into()));
            }
        }
    }
    Ok((compiled, form))
}

/// One part for the caches log, a stage after another; `None` while CGBJ is off.
pub fn summary() -> Option<String> {
    let s = switches()?;
    let parts: Vec<String> = [(s.compute, Kind::Compute, "cs"), (s.pixel, Kind::Pixel, "ps")]
        .into_iter()
        .filter(|(e, ..)| *e == Emitter::Cgbj)
        .map(|(_, kind, stage)| {
            let c = kind.counters();
            let refused = c.refused.lock().map(|m| m.clone()).unwrap_or_default();
            let total: u64 = refused.values().sum();
            let mut top: Vec<_> = refused.into_iter().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let top: Vec<String> = top.iter().take(4).map(|(r, n)| format!("{r} {n}")).collect();
            format!(
                "cgbj {stage} {} tried, {} made ({:.1} MB DXIL in {:.1} s), {} from cache, {total} refused ({} invalid{}{})",
                c.tried.load(Relaxed),
                c.produced.load(Relaxed),
                c.dxil_bytes.load(Relaxed) as f64 / (1u64 << 20) as f64,
                c.translate_ms.load(Relaxed) as f64 / 1000.0,
                c.from_cache.load(Relaxed),
                c.invalid.load(Relaxed),
                if top.is_empty() { "" } else { "; " },
                top.join(", ")
            )
        })
        .collect();
    Some(parts.join("; "))
}
