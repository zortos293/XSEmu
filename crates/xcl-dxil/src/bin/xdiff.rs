//! Differential runner: two DXIL containers, the same random inputs, compared outputs (owner: the interp subagent).
//!
//! ~~~text
//! xdiff a.dxil b.dxil [options]        compare one pair (a is the reference)
//! xdiff --batch A_DIR B_DIR [options]  compare every file of A_DIR with the file of the
//!                                      same name in B_DIR; with one directory, every
//!                                      NAME.a.dxil with NAME.b.dxil
//! xdiff --run a.dxil [options]         run one program and print what it did
//!
//!   --seeds N     seeds per pair (default 8)       --seed S     first seed (default 1)
//!   --groups G    thread groups per run (1)        --budget B   lane-instructions per run (2^31; 0: no limit)
//!   --jobs J      worker threads (all cores)       --wave W     wave size when a program declares none (32);
//!                                                               the vertices or pixels of a vs / ps run
//!   --explore R   input variants tried per seed to reach more code (see interp::explore)
//!   --lanes L     only the lowest L lanes of each wave run (a partial launch, see interp::Config)
//!   --natural     do not force empty ballots on odd seeds (see interp::Config)
//!   -v            per-seed lines; in a batch, a line per pair that is equal too
//! ~~~
//!
//! Outcomes of a pair: equal; equal except NaN payloads (every stored word that differs is
//! a NaN pattern in both runs, see interp::Verdict); data race (the runs differ after
//! lanes or waves raced: the program does not fix the result); different after undefined
//! behaviour; budget exceeded by both; unsupported; different.
//!
//! Exit codes: 1 a pair differs; else 2 a pair could not be compared (unsupported, budget,
//! data race, undefined behaviour); else 4 a pair is equal except NaN payloads; else 0,
//! every pair equal. 3 usage or I/O error.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use xcl_dxil::interp::{self, Config, Program, Verdict};

struct Options {
    seeds: u64,
    first_seed: u64,
    cfg: Config,
    jobs: usize,
    wave: u32,
    verbose: bool,
    /// input variants tried per seed to reach more code
    explore: u32,
}

fn usage() -> ! {
    eprintln!("usage: xdiff a.dxil b.dxil | --batch A_DIR [B_DIR] | --run a.dxil  [--seeds N] [--seed S] [--groups G] [--budget B] [--jobs J] [--wave W] [--lanes L] [-v]");
    std::process::exit(3);
}

fn load(path: &Path, o: &Options) -> Result<Program, Verdict> {
    let bytes = std::fs::read(path).map_err(|e| Verdict::Unsupported(format!("unreadable: {}: {e}", path.display())))?;
    interp::load_with(&bytes, o.wave).map_err(|e| interp::load_verdict(&e))
}

/// The class a verdict is counted under.
fn class(v: &Verdict) -> String {
    match v {
        Verdict::Equal => "equal".into(),
        Verdict::NanPayloads(_) => "equal except NaN payloads".into(),
        Verdict::Different(_) => "different".into(),
        Verdict::Race(_) => "data race".into(),
        Verdict::Undefined(_) => "different after undefined behaviour".into(),
        Verdict::Budget => "budget exceeded by both".into(),
        Verdict::Unsupported(what) => format!("unsupported: {what}"),
    }
}

fn exit_code(v: &Verdict) -> i32 {
    match v {
        Verdict::Equal => 0,
        Verdict::Different(_) => 1,
        Verdict::NanPayloads(_) => 4,
        _ => 2,
    }
}

/// Run `n` jobs on the worker threads.
fn parallel(jobs: usize, n: usize, work: &(dyn Fn(usize) + Sync)) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs.min(n).max(1) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= n {
                    break;
                }
                work(k);
            });
        }
    });
}

/// One pair over the seeds, seeds in parallel. The verdict of the lowest seed that is not
/// equal, and its seed.
fn pair(a: &Program, b: &Program, o: &Options, jobs: usize) -> (Verdict, u64, Coverage) {
    let found: Mutex<Option<(u64, Verdict)>> = Mutex::new(None);
    let covered: Mutex<[Vec<bool>; 2]> = Mutex::new([Vec::new(), Vec::new()]);
    parallel(jobs, o.seeds as usize, &|k| {
        let seed = o.first_seed + k as u64;
        if found.lock().unwrap().as_ref().is_some_and(|f| f.0 < seed) {
            return;
        }
        let (v, input, ran) = interp::diff_seed(a, b, seed, &o.cfg, o.explore);
        if o.verbose {
            eprintln!("seed {seed}: {}; {} of the input's words replaced", class(&v), input.patches.len());
        }
        for (all, ran) in covered.lock().unwrap().iter_mut().zip(&ran) {
            all.resize(ran.len(), false);
            all.iter_mut().zip(ran).for_each(|(c, r)| *c |= *r);
        }
        if v != Verdict::Equal {
            let mut f = found.lock().unwrap();
            if f.as_ref().map_or(true, |f| seed < f.0) {
                *f = Some((seed, v));
            }
        }
    });
    let covered = covered.into_inner().unwrap();
    let coverage = Coverage([a.coverage_detail(&covered[0]), b.coverage_detail(&covered[1])]);
    match found.into_inner().unwrap() {
        Some((seed, v)) => (v, seed, coverage),
        None => (Verdict::Equal, o.first_seed, coverage),
    }
}

/// What the inputs of a pair reached in its two programs: (ran, all) of the blocks, the
/// instructions, and the stores and atomics (Program::coverage_detail).
#[derive(Clone, Copy, Default)]
struct Coverage([[(usize, usize); 3]; 2]);

impl Coverage {
    fn percent(&self, program: usize, what: usize) -> f64 {
        let (ran, all) = self.0[program][what];
        100.0 * ran as f64 / all.max(1) as f64
    }
}

impl std::fmt::Display for Coverage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ran {:.1} % / {:.1} % of the blocks, {:.1} % / {:.1} % of the stores and atomics", self.percent(0, 0), self.percent(1, 0), self.percent(0, 2), self.percent(1, 2))
    }
}

fn print_verdict(v: &Verdict, seed: u64) {
    match v {
        Verdict::Equal => println!("equal"),
        Verdict::Different(text) | Verdict::Undefined(text) | Verdict::Race(text) | Verdict::NanPayloads(text) => println!("{} (seed {seed})\n{text}", class(v)),
        Verdict::Budget => println!("budget exceeded by both (seed {seed})"),
        Verdict::Unsupported(what) => println!("unsupported: {what}"),
    }
}

fn single(a: &Path, b: &Path, o: &Options) -> i32 {
    let (pa, pb) = match (load(a, o), load(b, o)) {
        (Ok(pa), Ok(pb)) => (pa, pb),
        (Err(v), _) | (_, Err(v)) => {
            print_verdict(&v, 0);
            return exit_code(&v);
        }
    };
    if pa.numthreads != pb.numthreads || pa.wave_size != pb.wave_size {
        println!("different\nnumthreads {:?} wave {} / numthreads {:?} wave {}", pa.numthreads, pa.wave_size, pb.numthreads, pb.wave_size);
        return 1;
    }
    let (v, seed, coverage) = pair(&pa, &pb, o, o.jobs);
    print_verdict(&v, seed);
    println!("{coverage}");
    exit_code(&v)
}

fn dxil_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "dxil")).collect(),
        Err(e) => {
            eprintln!("{}: {e}", dir.display());
            std::process::exit(3);
        }
    };
    files.sort();
    files
}

fn batch(dir_a: &Path, dir_b: Option<&Path>, o: &Options) -> i32 {
    let mut pairs: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    for a in dxil_files(dir_a) {
        let name = a.file_name().unwrap_or_default().to_string_lossy().to_string();
        match dir_b {
            Some(dir_b) => {
                let b = dir_b.join(&name);
                if b.exists() {
                    pairs.push((name, a, b));
                }
            }
            None => {
                if let Some(stem) = name.strip_suffix(".a.dxil") {
                    let b = dir_a.join(format!("{stem}.b.dxil"));
                    if b.exists() {
                        pairs.push((stem.to_string(), a, b));
                    }
                }
            }
        }
    }
    if pairs.is_empty() {
        eprintln!("no pairs found");
        return 3;
    }
    let results: Mutex<Vec<(usize, Verdict, Coverage)>> = Mutex::new(Vec::new());
    let started = Instant::now();
    parallel(o.jobs, pairs.len(), &|k| {
        let (name, a, b) = &pairs[k];
        let t = Instant::now();
        let (v, seed, coverage) = match (load(a, o), load(b, o)) {
            (Ok(pa), Ok(pb)) => {
                if pa.numthreads != pb.numthreads || pa.wave_size != pb.wave_size {
                    (Verdict::Different("numthreads or wave size differ".into()), 0, Coverage::default())
                } else {
                    pair(&pa, &pb, o, 1)
                }
            }
            (Err(v), _) | (_, Err(v)) => (v, 0, Coverage::default()),
        };
        if v != Verdict::Equal || o.verbose {
            let detail = match &v {
                Verdict::Different(text) | Verdict::Undefined(text) | Verdict::Race(text) | Verdict::NanPayloads(text) => format!(" (seed {seed})\n    {}", text.replace('\n', "\n    ")),
                Verdict::Budget => format!(" (seed {seed})"),
                _ => String::new(),
            };
            println!("{name}: {}{detail}; {coverage}; {:.1} s", class(&v), t.elapsed().as_secs_f64());
        }
        results.lock().unwrap().push((k, v, coverage));
    });
    let results = results.into_inner().unwrap();
    let mut classes: std::collections::BTreeMap<String, usize> = Default::default();
    let mut code = 0;
    for (_, v, _) in &results {
        *classes.entry(class(v)).or_default() += 1;
        // the worst: a difference, then a pair not compared, then NaN payloads
        let rank = |c: i32| [0, 3, 2, 0, 1][c as usize];
        code = if rank(exit_code(v)) > rank(code) { exit_code(v) } else { code };
    }
    println!("--- {} pairs, {} seeds each, {:.1} s", results.len(), o.seeds, started.elapsed().as_secs_f64());
    let mut sorted: Vec<_> = classes.into_iter().collect();
    sorted.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    for (class, n) in sorted {
        println!("{n:6}  {class}");
    }
    // how much of the reference programs the inputs reached, and how much an equal is worth
    let mut ran: Vec<(f64, usize)> = results.iter().filter(|r| !matches!(r.1, Verdict::Unsupported(_))).map(|r| (r.2.percent(0, 0), r.0)).collect();
    ran.sort_by(|x, y| x.0.total_cmp(&y.0));
    if let Some(&(least, k)) = ran.first() {
        let mean = ran.iter().map(|r| r.0).sum::<f64>() / ran.len() as f64;
        println!("blocks run in the reference programs: mean {mean:.1} %, median {:.1} %, least {least:.1} % ({})", ran[ran.len() / 2].0, pairs[k].0);
        let bands = [(100.0, "all"), (90.0, "90 % and more"), (75.0, "75 % and more"), (50.0, "50 % and more"), (25.0, "25 % and more"), (0.0, "less")];
        let mut counts = [0; 6];
        for r in results.iter().filter(|r| r.1 == Verdict::Equal) {
            counts[bands.iter().position(|b| r.2.percent(0, 0) >= b.0).unwrap_or(5)] += 1;
        }
        let text: Vec<String> = bands.iter().zip(counts).filter(|(_, n)| *n != 0).map(|(b, n)| format!("{n} {}", b.1)).collect();
        println!("equal pairs by blocks run: {}", text.join(", "));
    }
    code
}

fn run_one(path: &Path, o: &Options) -> i32 {
    let t = Instant::now();
    let p = match load(path, o) {
        Ok(p) => p,
        Err(v) => {
            print_verdict(&v, 0);
            return 2;
        }
    };
    let s = &p.stats;
    println!(
        "{}: numthreads {:?}, wave {}; {} instructions ({} in the IR), {} blocks, {} loops; rows: {} kept, {} scratch, {} constants; loaded in {:.3} s",
        p.entry,
        p.numthreads,
        p.wave_size,
        s.insts,
        s.ir_insts,
        s.blocks,
        s.loops,
        s.persistent_rows,
        s.scratch_rows,
        s.const_rows,
        t.elapsed().as_secs_f64()
    );
    let mut covered = Vec::new();
    for k in 0..o.seeds {
        let seed = o.first_seed + k;
        let t = Instant::now();
        let inputs = interp::explore(&p, seed, &o.cfg, o.explore);
        if o.explore != 0 {
            println!("seed {seed}: {} inputs from {} variants in {:.3} s", inputs.len(), o.explore, t.elapsed().as_secs_f64());
        }
        let t = Instant::now();
        let mut r = interp::run(&p, seed, &o.cfg);
        let secs = t.elapsed().as_secs_f64();
        covered.resize(r.covered.len(), false);
        for input in &inputs[1..] {
            let more = interp::run_input(&p, input, &o.cfg, false).0;
            covered.iter_mut().zip(&more.covered).for_each(|(c, r)| *c |= *r);
            r.ub |= more.ub;
        }
        covered.iter_mut().zip(&r.covered).for_each(|(c, r)| *c |= *r);
        println!(
            "seed {seed}: {}; {} lane-instructions in {:.3} s ({:.0} M/s); covered {:.0} %; written {}, atomics {}; buffer reads {}, texel reads {}, samples {}, out of bounds {}, null {}{}",
            match &r.stop {
                None => "finished".to_string(),
                Some(s) => format!("{s:?}"),
            },
            r.steps,
            secs,
            r.steps as f64 / secs / 1e6,
            100.0 * p.coverage(&r.covered).0 as f64 / p.coverage(&r.covered).1.max(1) as f64,
            r.written.len(),
            r.atomics.count,
            r.access.buffer_reads,
            r.access.texture_reads,
            r.access.samples,
            r.access.out_of_bounds,
            r.access.null,
            if r.ub != 0 { format!("; undefined: {}", interp::exec::describe_ub(r.ub)) } else { String::new() }
        );
        if o.verbose {
            for w in r.written.iter().take(64) {
                println!("  {} = {:x?}", interp::res::describe(w), &w.3[..if w.0 == 0 { 1 } else { 4 }]);
            }
            // in the order they ran
            for a in r.atomics.sample.iter().take(4096) {
                println!("  atomic {} on {:#x} +{:#x}: {:#x}, {:#x}", a.op, a.res, a.addr, a.a, a.b);
            }
        }
    }
    let (ran, all) = p.coverage(&covered);
    println!("covered by {} seeds: {ran} of {all} instructions ({:.1} %)", o.seeds, 100.0 * ran as f64 / all.max(1) as f64);
    0
}

fn main() {
    let mut o = Options { seeds: 8, first_seed: 1, cfg: Config::default(), jobs: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1), wave: interp::DEFAULT_WAVE_SIZE, verbose: false, explore: 0 };
    let mut files: Vec<PathBuf> = Vec::new();
    let mut mode = "pair";
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut number = |what: &str| -> u64 {
            match args.next().and_then(|v| v.parse().ok()) {
                Some(n) => n,
                None => {
                    eprintln!("{what} needs a number");
                    usage()
                }
            }
        };
        match a.as_str() {
            "--seeds" => o.seeds = number("--seeds"),
            "--seed" => o.first_seed = number("--seed"),
            "--groups" => o.cfg.groups = number("--groups") as u32,
            "--budget" => o.cfg.budget = number("--budget"),
            "--jobs" => o.jobs = number("--jobs").max(1) as usize,
            "--wave" => o.wave = number("--wave") as u32,
            "--explore" => o.explore = number("--explore") as u32,
            "--lanes" => o.cfg.live = Some(number("--lanes") as u32),
            "--natural" => o.cfg.force_ballots = false,
            "--batch" => mode = "batch",
            "--run" => mode = "run",
            "-v" => o.verbose = true,
            _ if a.starts_with("--") => usage(),
            _ => files.push(a.into()),
        }
    }
    let code = match (mode, files.as_slice()) {
        ("pair", [a, b]) => single(a, b, &o),
        ("batch", [a]) => batch(a, None, &o),
        ("batch", [a, b]) => batch(a, Some(b), &o),
        ("run", [a]) => run_one(a, &o),
        _ => usage(),
    };
    std::process::exit(code);
}
