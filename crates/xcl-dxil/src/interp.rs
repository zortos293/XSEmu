//! A reference interpreter of DXIL compute, vertex and pixel shaders for differential testing against DXC's output (owner: the interp subagent).
//!
//! Two DXIL containers compiled from one HLSL program (DXC's and this crate's, or DXC's at
//! two optimisation levels) must compute the same thing. [`load`] reads a container into a
//! [`Program`], [`run`] executes its entry point for some thread groups over resources
//! generated from a seed, and [`compare`] says whether two runs wrote the same values to
//! the same places and performed the same atomics. A vertex program runs as one wave of
//! vertices and a pixel program as one wave of whole 2x2 quads, with inputs generated from
//! the seed and the input signature ([`res::World::input`]) and every stored output
//! component compared; a discarded pixel runs on for the derivatives of its quad but
//! stores nothing, and counts in wave operations as any other lane (a simplification:
//! both programs get it). Derivatives are differences across the quad; a sample that takes
//! its level from the quad is a function of those differences too. No GPU is involved: the
//! semantics are D3D's where D3D defines them and one fixed choice where it does not (the
//! same choice for both programs), and what is not modelled is reported as unsupported,
//! never guessed.
//!
//! * [`cfg`], [`decode`]: the entry function pre-decoded into a compact form;
//! * [`exec`]: waves, lanes, reconvergence, barriers, every instruction and dx.op;
//! * [`res`]: buffers, textures and the descriptor heap, generated from the seed;
//! * [`math`]: float semantics (NaN rules, DXC-compatible transcendental functions).

pub mod cfg;
pub mod decode;
pub mod exec;
pub mod math;
pub mod res;

pub use decode::Program;
pub use exec::Stop;

use crate::{Error, Result};

/// The wave size of a program that declares none.
pub const DEFAULT_WAVE_SIZE: u32 = 32;

/// Read a DXIL container (or bare LLVM bitcode) into a program.
pub fn load(bytes: &[u8]) -> Result<Program> {
    load_with(bytes, DEFAULT_WAVE_SIZE)
}

pub fn load_with(bytes: &[u8], default_wave: u32) -> Result<Program> {
    let bitcode = if bytes.starts_with(b"DXBC") { crate::bcreader::container_bitcode(bytes)? } else { bytes };
    let (module, _) = crate::bcreader::read_with(bitcode, crate::bcreader::Options { keep_dead: true })?;
    decode::decode(&module, default_wave)
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// thread groups dispatched, as (groups, 1, 1); they run one after the other
    pub groups: u32,
    /// lane-instructions a run may execute; 0 for no limit
    pub budget: u64,
    /// On odd seeds, three of four ballots (WaveActiveBallot) in which no lane voted come
    /// back with lanes set. The translator's programs gate nearly all their code on such
    /// masks (the console's execution mask), and random data empties them at once; with
    /// this, the code behind the gates runs. It is the same change of semantics for both
    /// programs of a pair, so they must still agree, but a run no longer shows what the
    /// program computes on a GPU.
    pub force_ballots: bool,
    /// Lanes of each wave that start live, the lowest ones; None: all of them. A wave with
    /// fewer live lanes than it has is a partial launch (a primitive's edge in a pixel
    /// shader): the other lanes never run, and their registers keep what the run started
    /// them with. What a GPU leaves in a lane it did not launch is undefined, so a program
    /// that reads one has no defined result there; this shows what it reads.
    pub live: Option<u32>,
}

impl Default for Config {
    fn default() -> Config {
        Config { groups: 1, budget: 1 << 31, force_ballots: true, live: None }
    }
}

/// What one run did.
#[derive(Debug, Clone)]
pub struct Run {
    /// None: it ran to the end
    pub stop: Option<Stop>,
    /// the final value of every buffer word and texel written, sorted
    pub written: Vec<res::Written>,
    pub atomics: res::Atomics,
    pub access: res::Access,
    /// lane-instructions executed
    pub steps: u64,
    /// undefined behaviour met ([`exec::UB_LOCAL_BOUNDS`], ...)
    pub ub: u32,
    /// which blocks ran ([`Program::coverage`] turns it into instruction counts)
    pub covered: Vec<bool>,
}

/// The inputs of a run: the seed everything is generated from, and generated buffer words
/// replaced ((resource << 32 | word, value); [`explore`] finds them).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Input {
    pub seed: u64,
    pub patches: Vec<(u64, u32)>,
}

pub fn run(p: &Program, seed: u64, cfg: &Config) -> Run {
    run_input(p, &Input { seed, patches: Vec::new() }, cfg, false).0
}

/// Run on an input; with `trace`, also return what the program read and compared.
pub fn run_input(p: &Program, input: &Input, cfg: &Config, trace: bool) -> (Run, Option<res::Trace>) {
    let seed = input.seed;
    let mut world = res::World::new(seed);
    world.patches.extend(input.patches.iter().copied());
    world.force_ballots = cfg.force_ballots && seed & 1 != 0;
    if trace {
        world.trace = Some(res::Trace::default());
    }
    let mut steps = 0;
    let mut ub = 0;
    let mut stop = None;
    let budget = if cfg.budget == 0 { u64::MAX } else { cfg.budget };
    math::take_nan_open();
    for g in 0..cfg.groups {
        let mut group = exec::Group::new(p, &mut world, [g, 0, 0], budget.saturating_sub(steps), cfg.live);
        let result = group.run();
        steps += group.steps;
        ub |= group.ub;
        if let Err(s) = result {
            stop = Some(s);
            break;
        }
    }
    if math::take_nan_open() {
        ub |= exec::UB_NAN;
    }
    let mut trace = world.trace.take();
    if let Some(trace) = &mut trace {
        trace.finish();
    }
    (Run { stop, written: world.written(), atomics: world.atomics.clone(), access: world.access, steps, ub, covered: std::mem::take(&mut world.covered) }, trace)
}

/// Lane-instructions the variants of one seed may execute together.
const EXPLORE_STEPS: u64 = 1 << 31;

/// Inputs that reach more of `p` than its seed alone: the seed's own input first, then
/// every variant of an earlier one that ran a block none before it had. A variant replaces
/// a few of the buffer words its parent read, by small and special values, by a neighbour
/// of the old value, or by the other side of a comparison the parent failed, moved by the
/// offset between the word and the compared value when that is small (translated
/// programs compare what they load with constants and with each other: making a loaded
/// word equal to what it was compared with opens the branch). `rounds` variants are tried at most:
/// the search ends when every block has run.
///
/// Programs gate most of their code on data, so random words alone leave it unreached.
/// The inputs found for one program are then given to both programs of a pair.
pub fn explore(p: &Program, seed: u64, cfg: &Config, rounds: u32) -> Vec<Input> {
    let first = Input { seed, patches: Vec::new() };
    if rounds == 0 {
        return vec![first];
    }
    // variants that run long say little more than short ones and cost the time of many
    let cfg = Config { budget: cfg.budget.wrapping_sub(1).min((1 << 26) - 1) + 1, ..*cfg };
    let (run, trace) = run_input(p, &first, &cfg, true);
    // the variants together may cost what a few full runs cost
    let mut left = EXPLORE_STEPS.saturating_sub(run.steps);
    let mut union = run.covered;
    let mut pool = vec![(first, trace.unwrap_or_default())];
    let mut rng = res::mix(seed ^ 0x6578_706c_6f72_65);
    let mut next = || {
        rng = res::mix(rng.wrapping_add(0x9e37_79b9_7f4a_7c15));
        rng
    };
    for _ in 0..rounds {
        if left == 0 || union.iter().all(|c| *c) {
            break;
        }
        // the newest input most of the time: it is the deepest
        let parent = if next() % 4 != 0 { pool.len() - 1 } else { (next() % pool.len() as u64) as usize };
        let (input, trace) = &pool[parent];
        if trace.reads.is_empty() {
            continue;
        }
        let mut child = input.clone();
        for _ in 0..1 + next() % 3 {
            let r = next();
            let mut target = trace.reads[(r >> 8) as usize % trace.reads.len()];
            let value = if r & 1 == 0 && !trace.compares.is_empty() {
                // a word that holds one side of a failed comparison gets the other side
                // one of the last, more often than not: the run stopped going further there
                let n = trace.compares.len();
                let (x, y) = trace.compares[if r >> 60 & 3 != 0 { n - 1 - (r >> 24) as usize % n.min(96) } else { (r >> 24) as usize % n }];
                let (x, y) = if r & 2 == 0 { (x, y) } else { (y, x) };
                // the word itself, or one the compared value is a small offset from
                // (registers are compared after an add of a constant as often as before)
                let near = |read: &&(u64, u32)| ((read.1 as u64).wrapping_sub(x) as u32).wrapping_add(4096) <= 8192;
                let mut delta = 0;
                if let Some(&hit) = trace.reads.iter().find(|read| read.1 as u64 == x).or_else(|| trace.reads.iter().find(near)) {
                    target = hit;
                    delta = (hit.1 as u64).wrapping_sub(x);
                }
                let y = y.wrapping_add(delta);
                match (r >> 2) & 7 {
                    0 => y.wrapping_add(1),
                    1 => y.wrapping_sub(1),
                    _ => y,
                }
            } else {
                let old = target.1 as u64;
                match (r >> 2) % 8 {
                    0 => 0,
                    1 => 1,
                    2 => u64::MAX,
                    3 => old ^ 1 << ((r >> 40) & 31),
                    4 => old.wrapping_add(1),
                    5 => old.wrapping_sub(1),
                    6 => (r >> 40) & 0xff,
                    _ => res::biased(next()) as u64,
                }
            } as u32;
            match child.patches.iter_mut().find(|patch| patch.0 == target.0) {
                Some(patch) => patch.1 = value,
                None => child.patches.push((target.0, value)),
            }
        }
        let (run, trace) = run_input(p, &child, &cfg, true);
        left = left.saturating_sub(run.steps);
        if run.stop.is_some() || !run.covered.iter().zip(&union).any(|(c, u)| *c && !*u) {
            continue;
        }
        union.iter_mut().zip(&run.covered).for_each(|(u, c)| *u |= *c);
        pool.push((child, trace.unwrap_or_default()));
    }
    pool.into_iter().map(|(input, _)| input).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Equal,
    /// Equal but for the payloads of NaNs: every word that differs is a NaN in both runs
    /// (as a 32-bit float; nothing records whether a float computation wrote it, so any
    /// two words that are both NaN patterns count), and so is every atomic operand. The
    /// IR does not fix the payload ([math]); the text counts the words and shows the first.
    NanPayloads(String),
    /// the first differences
    Different(String),
    /// they differ, and lanes or waves raced in one of them first: the program does not
    /// fix the order of two accesses to one location, and the two shapes took two orders
    Race(String),
    /// they differ, and one of them met other undefined behaviour first (not a miscompile
    /// by itself: the compilers may treat undefined programs differently)
    Undefined(String),
    /// both ran out of budget: nothing to compare
    Budget,
    Unsupported(String),
}

/// Compare two runs of the same program on the same seed. `a` is the reference.
pub fn compare(a: &Run, b: &Run) -> Verdict {
    for r in [a, b] {
        if let Some(Stop::Unsupported(what)) = &r.stop {
            return Verdict::Unsupported(what.clone());
        }
    }
    let ub = a.ub | b.ub;
    let differ = |text: String| {
        if ub & exec::UB_RACE != 0 {
            Verdict::Race(format!("{text} (after {})", exec::describe_ub(ub)))
        } else if ub != 0 {
            Verdict::Undefined(format!("{text} (after {})", exec::describe_ub(ub)))
        } else {
            Verdict::Different(text)
        }
    };
    match (&a.stop, &b.stop) {
        (Some(Stop::Budget), Some(Stop::Budget)) => return Verdict::Budget,
        (Some(Stop::Budget), None) => return differ("the first ran out of budget, the second finished".into()),
        (None, Some(Stop::Budget)) => return differ("the second ran out of budget, the first finished".into()),
        _ => {}
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    let mut count = 0usize;
    // whether every difference so far is one NaN for another
    let mut payloads = true;
    while i < a.written.len() || j < b.written.len() {
        let (x, y) = (a.written.get(i), b.written.get(j));
        let line = match (x, y) {
            (Some(x), Some(y)) if (x.0, x.1, x.2) == (y.0, y.1, y.2) => {
                i += 1;
                j += 1;
                if x.3 == y.3 {
                    continue;
                }
                payloads &= x.3.iter().zip(&y.3).all(|(p, q)| res::any_nan(*p as u64) == res::any_nan(*q as u64));
                format!("{}: {} / {}", res::describe(x), value(x), value(y))
            }
            (Some(x), Some(y)) if (x.0, x.1, x.2) < (y.0, y.1, y.2) => {
                i += 1;
                payloads = false;
                format!("{}: {} / not written", res::describe(x), value(x))
            }
            (Some(x), None) => {
                i += 1;
                payloads = false;
                format!("{}: {} / not written", res::describe(x), value(x))
            }
            (_, Some(y)) => {
                j += 1;
                payloads = false;
                format!("{}: not written / {}", res::describe(y), value(y))
            }
            (None, None) => break,
        };
        count += 1;
        if out.len() < 8 {
            out.push(line);
        }
    }
    let words = count;
    if a.atomics.count != b.atomics.count || a.atomics.hash != b.atomics.hash {
        count += 1;
        payloads &= a.atomics.count == b.atomics.count && a.atomics.hash_any_nan == b.atomics.hash_any_nan;
        let mut line = format!("atomics: {} / {}", a.atomics.count, b.atomics.count);
        let (mut x, mut y) = (a.atomics.sample.clone(), b.atomics.sample.clone());
        x.sort_unstable();
        y.sort_unstable();
        let first = x.iter().zip(&y).find(|(p, q)| p != q).map(|(p, q)| (Some(*p), Some(*q))).or_else(|| match x.len().cmp(&y.len()) {
            std::cmp::Ordering::Less => Some((None, Some(y[x.len()]))),
            std::cmp::Ordering::Greater => Some((Some(x[y.len()]), None)),
            _ => None,
        });
        if let Some((p, q)) = first {
            line += &format!("; first unmatched: {} / {}", atomic(p), atomic(q));
        }
        out.push(line);
    }
    if count == 0 {
        return Verdict::Equal;
    }
    if count > out.len() {
        out.push(format!("... {count} differences"));
    }
    if payloads {
        return Verdict::NanPayloads(format!("{words} stored words differ, each a NaN in both{}\n{}", if count > words { ", and the operands of atomics likewise" } else { "" }, out.join("\n")));
    }
    differ(out.join("\n"))
}

fn value(w: &res::Written) -> String {
    if w.0 != 1 {
        format!("{:#010x}", w.3[0])
    } else {
        format!("({:#x}, {:#x}, {:#x}, {:#x})", w.3[0], w.3[1], w.3[2], w.3[3])
    }
}

fn atomic(a: Option<res::Atomic>) -> String {
    match a {
        None => "none".into(),
        Some(a) => {
            let place = if a.res == res::SHARED { format!("groupshared {:06x} +{}", a.addr >> 40, a.addr & ((1 << 40) - 1)) } else { res::describe(&(0, a.res, a.addr, [0; 4])) };
            format!("op {} on {place} with {:#x}, {:#x}", a.op, a.a, a.b)
        }
    }
}

/// The verdict of a pair over several seeds: the first that is not equal, else equal.
/// Returns it with the seed it was found at.
pub fn diff(a: &Program, b: &Program, seeds: std::ops::Range<u64>, cfg: &Config) -> (Verdict, u64) {
    let first = seeds.start;
    for seed in seeds {
        let v = compare(&run(a, seed, cfg), &run(b, seed, cfg));
        if v != Verdict::Equal {
            return (v, seed);
        }
    }
    (Verdict::Equal, first)
}

/// Compare a pair on one seed and the inputs [`explore`] derives from it for `a`. Returns
/// the first verdict that is not equal with its input, and the blocks of `a` and of `b`
/// that ran.
pub fn diff_seed(a: &Program, b: &Program, seed: u64, cfg: &Config, rounds: u32) -> (Verdict, Input, [Vec<bool>; 2]) {
    let mut covered = [Vec::new(), Vec::new()];
    let inputs = explore(a, seed, cfg, rounds);
    for input in &inputs {
        let (ra, rb) = (run_input(a, input, cfg, false).0, run_input(b, input, cfg, false).0);
        for (all, run) in covered.iter_mut().zip([&ra, &rb]) {
            all.resize(run.covered.len(), false);
            all.iter_mut().zip(&run.covered).for_each(|(u, c)| *u |= *c);
        }
        let v = compare(&ra, &rb);
        if v != Verdict::Equal {
            return (v, input.clone(), covered);
        }
    }
    (Verdict::Equal, Input { seed, patches: Vec::new() }, covered)
}

/// Why a container could not be loaded, as the outcome of its pair.
pub fn load_verdict(e: &Error) -> Verdict {
    match e {
        Error::Unsupported(what) => Verdict::Unsupported(what.clone()),
        other => Verdict::Unsupported(format!("unreadable: {other}")),
    }
}
