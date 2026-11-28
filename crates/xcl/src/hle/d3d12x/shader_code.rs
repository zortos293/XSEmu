//! A meter of the compiled shader code the title's lists use and, when asked for, a budget for it.
//!
//! NVIDIA's D3D12 driver keeps the machine code of the pipelines its lists use in one heap of at
//! most 512 MB, and removes the device when a submission needs more: 0x887a0005 (removed, not
//! hung), nothing in DRED, no event in the system log, as there was no GPU fault (ivall/xcl
//! #2403, #2526, #2590). Code goes into the heap when a pipeline is first used and out of it
//! with the pipeline. play-97 and play-99 lost the device with exactly those signs, at about
//! 1,000 pipelines in use; the runs that kept it ended under 800.
//!
//! So every host pipeline XCL binds for the title is counted once, with the size of its
//! `GetCachedBlob` (about 1/9 less than its code takes in the driver's heap, by upstream's
//! measurement: the 512 MB heap is about 455 MB as counted here). The blob is read on a thread of
//! the meter's own, not on the bind. A line goes to the log at every `STEP` and at the device's
//! removal.
//!
//! Not counted: a pipeline of the title's own that is bound as it is (not through a stand-in),
//! and XCL's own few pipelines. A pipeline released outside `pso_async::release` stays counted.
//!
//! `XCL_SHADER_CODE_METER=0` turns the meter off.
//!
//! The budget (`XCL_SHADER_CODE_MB`, megabytes as the meter counts; unset or 0, the default: none,
//! and nothing below happens). Over it, `pso_async` takes the pipelines no list took for
//! `idle_ms` away from their stand-ins, the longest unused first (`to_release`): the compute ones,
//! and the graphics ones too unless `XCL_SHADER_CODE_GRAPHICS=0` (`graphics`). Each is built
//! again at its next bind, as a pipeline released before its first bind is (behaviour after
//! ivall/xcl #2526). A pipeline taken away may still be on lists that are recorded or running, so
//! XCL keeps it (`retire`) until those lists are through (`through`): every list that takes a
//! pipeline is noted with its command allocator (`Uses::take`), and the title's `Reset` of an
//! allocator – its word that the GPU is past the lists recorded on it – ends its lists' hold.
//! The paths that put a pipeline on a list without the meter's count (`taken`) are noted too.
//! A bundle is recorded once and run by other lists again and again with no bind of its own, so
//! a pipeline a bundle took is never taken away (`Uses::pinned`), a compute one too: a bundle
//! may dispatch. A list handed to a queue takes what it took again (`submitted`): one recorded
//! once and run again with no `Reset` keeps its pipelines.
//! Not exact: a bind stalled for over `GRACE_MS` between reading its pipeline and setting it on
//! its list, or a list run over `HOLD_MAX_MS` after it took its pipeline with no `Reset` seen, can
//! still meet a released pipeline.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

/// a line in the log each time the code in use passes another of these
const STEP: usize = 32 << 20;
/// the driver's heap as this meter counts it (512 MB less 1/9)
const WALL: usize = 455 << 20;

#[derive(Default)]
struct Meter {
    /// host pipeline → (which sighting of that address, its code's bytes once read)
    seen: HashMap<usize, (usize, Option<usize>)>,
    sightings: usize,
    bytes: usize,
    largest: usize,
    /// the most `STEP`s a line was written for
    steps: usize,
    /// (pipelines, bytes) released since the start
    gone: (usize, usize),
}

impl Meter {
    /// `pso` was bound: the number of this sighting when it is its first.
    fn sight(&mut self, pso: usize) -> Option<usize> {
        if self.seen.contains_key(&pso) {
            return None;
        }
        self.sightings += 1;
        self.seen.insert(pso, (self.sightings, None));
        Some(self.sightings)
    }

    /// The code of sighting `seq` of `pso` is `bytes`: the line to log when the sum passed a step.
    fn measured(&mut self, pso: usize, seq: usize, bytes: usize) -> Option<String> {
        match self.seen.get_mut(&pso) {
            Some((s, size @ None)) if *s == seq => *size = Some(bytes),
            _ => return None,
        }
        self.bytes += bytes;
        self.largest = self.largest.max(bytes);
        if self.bytes / STEP <= self.steps {
            return None;
        }
        self.steps = self.bytes / STEP;
        Some(format!("D3D12.X: shader code in use: {} (XCL_SHADER_CODE_METER=0 turns this meter off)", self.summary()))
    }

    /// `pso` was released: whether it was counted.
    fn forget(&mut self, pso: usize) -> bool {
        let Some((_, size)) = self.seen.remove(&pso) else { return false };
        let bytes = size.unwrap_or(0);
        self.bytes -= bytes;
        self.gone = (self.gone.0 + 1, self.gone.1 + bytes);
        true
    }

    fn summary(&self) -> String {
        let unread = self.seen.values().filter(|(_, size)| size.is_none()).count();
        format!(
            "{} pipelines the title's lists bound hold {} MB of the driver's compiled code ({unread} of them not read yet, the largest {} KB; {} released since the start held {} MB); NVIDIA's driver removes the device when the code in use passes its heap, about {} MB as counted here",
            self.seen.len(),
            self.bytes >> 20,
            self.largest >> 10,
            self.gone.0,
            self.gone.1 >> 20,
            WALL >> 20
        )
    }
}

static METER: Mutex<Option<Meter>> = Mutex::new(None);
/// bumped when a counted pipeline goes: its address may come back for another one
static GENERATION: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// (the generation they are of, the pipelines this thread has told the meter of)
    static KNOWN: RefCell<(usize, HashSet<usize>)> = RefCell::new((0, HashSet::new()));
}

/// A pipeline the meter holds until its blob is read.
struct Held(usize, ID3D12PipelineState);
// SAFETY: a D3D12 pipeline state is free-threaded; the meter's thread only reads its cached blob
// and gives its reference back.
unsafe impl Send for Held {}

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // (the tests give `pso_async` pipelines that are none)
    *ON.get_or_init(|| !cfg!(test) && (budget() != 0 || std::env::var("XCL_SHADER_CODE_METER").map_or(true, |v| v.trim() != "0")))
}

/// How much compiled code the lists may have in use, in bytes as the meter counts
/// (`XCL_SHADER_CODE_MB`; 0, the default: no budget, and nothing is released).
pub(super) fn budget() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    // (the tests give `pso_async` pipelines that are none)
    *B.get_or_init(|| if cfg!(test) { 0 } else { std::env::var("XCL_SHADER_CODE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0) << 20 })
}

/// A pipeline no list took for this long may be released (`XCL_SHADER_CODE_IDLE_S`, 30 s).
pub(super) fn idle_ms() -> u64 {
    static I: OnceLock<u64> = OnceLock::new();
    *I.get_or_init(|| std::env::var("XCL_SHADER_CODE_IDLE_S").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(30) * 1000)
}

/// Whether the budget takes graphics pipelines away too (`XCL_SHADER_CODE_GRAPHICS=0`: no, only
/// the compute ones, and nothing is noted for graphics pipelines alone).
pub(super) fn graphics() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| std::env::var("XCL_SHADER_CODE_GRAPHICS").map_or(true, |v| v.trim() != "0"))
}

/// A pipeline taken from its stand-in is kept at least this long: a bind that read it just before
/// reaches its list, and is noted there, well within it.
const GRACE_MS: u64 = 5_000;
/// A list is taken to be through this long after it took a pipeline, whether or not a `Reset` of
/// its allocator was seen since.
const HOLD_MAX_MS: u64 = 120_000;

fn now_ms() -> u64 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Which lists took which pipelines, kept only with a budget.
#[derive(Default)]
struct Uses {
    /// host pipeline → when a list last took it (`now_ms`)
    last: HashMap<usize, u64>,
    /// command allocator (0: not known) → the pipelines its lists took since its last `Reset`, and when
    held: HashMap<usize, HashMap<usize, u64>>,
    /// the pipelines taken from their stand-ins that XCL still holds, and since when
    parked: Vec<(usize, u64)>,
    /// the pipelines a bundle took: they stay with their stand-ins, and one taken away just
    /// before stays parked
    pinned: HashSet<usize>,
}

impl Uses {
    /// A list recording on `allocator` takes `pso`.
    fn take(&mut self, allocator: usize, pso: usize, now: u64) {
        self.last.insert(pso, now);
        self.held.entry(allocator).or_default().insert(pso, now);
    }

    /// `allocator` was reset or released: the GPU is past its lists.
    fn reset(&mut self, allocator: usize) {
        if allocator != 0 {
            self.held.remove(&allocator);
        }
    }

    /// `pso` was taken from its stand-in.
    fn park(&mut self, pso: usize, now: u64) {
        self.last.remove(&pso);
        self.parked.push((pso, now));
    }

    /// Lists recorded on `allocator` are handed to a queue: what they took since its last `Reset`
    /// is taken again now.
    fn run(&mut self, allocator: usize, now: u64) {
        let Some(of) = self.held.get_mut(&allocator) else { return };
        for (pso, at) in of.iter_mut() {
            *at = now;
            if let Some(last) = self.last.get_mut(pso) {
                *last = now;
            }
        }
    }

    /// Of the pipelines `sizes` (each with the bytes of its code), (how many, their bytes) by when
    /// a list last took each: within each of `AGES_S`, longer ago, and never noted.
    fn ages(&self, sizes: &[(usize, usize)], now: u64) -> [(usize, usize); AGES_S.len() + 2] {
        let mut out = [(0, 0); AGES_S.len() + 2];
        for &(pso, bytes) in sizes {
            let at = match self.last.get(&pso) {
                Some(&last) => AGES_S.iter().position(|&s| now.saturating_sub(last) < s * 1000).unwrap_or(AGES_S.len()),
                None => AGES_S.len() + 1,
            };
            out[at] = (out[at].0 + 1, out[at].1 + bytes);
        }
        out
    }

    /// When a list last took `pso`, if it may be taken from its stand-in (no bundle took it).
    fn candidate(&self, pso: usize) -> Option<u64> {
        if self.pinned.contains(&pso) {
            return None;
        }
        self.last.get(&pso).copied()
    }

    /// The parked pipelines no list can still run with, which leave here.
    fn through(&mut self, now: u64) -> Vec<usize> {
        for of in self.held.values_mut() {
            of.retain(|_, at| now.saturating_sub(*at) < HOLD_MAX_MS);
        }
        self.held.retain(|_, of| !of.is_empty());
        let (held, pinned, mut out) = (&self.held, &self.pinned, Vec::new());
        self.parked.retain(|&(pso, at)| {
            let keep = now.saturating_sub(at) < GRACE_MS || pinned.contains(&pso) || held.values().any(|of| of.contains_key(&pso));
            if !keep {
                out.push(pso);
            }
            keep
        });
        out
    }
}

static USES: Mutex<Option<Uses>> = Mutex::new(None);

/// the ages `by_age` sorts the code in use by, in seconds
const AGES_S: [u64; 3] = [5, 30, 120];
/// (when, the frames shown by then), one a second for a little over the longest of `AGES_S`
static FRAMES: Mutex<std::collections::VecDeque<(u64, u64)>> = Mutex::new(std::collections::VecDeque::new());

/// `frames` were shown so far (told once a second by `pso_async`'s code thread).
pub(super) fn frame_mark(frames: u64) {
    let (mut f, now) = (FRAMES.lock().unwrap(), now_ms());
    f.push_back((now, frames));
    while f.front().is_some_and(|&(at, _)| now.saturating_sub(at) > (AGES_S[AGES_S.len() - 1] + 10) * 1000) {
        f.pop_front();
    }
}

/// How many frames were shown in the last `secs` seconds, of the marks `marks` (None: the marks
/// do not go back that far).
fn frames_in(marks: &std::collections::VecDeque<(u64, u64)>, now: u64, secs: u64) -> Option<u64> {
    let &(_, newest) = marks.back()?;
    let &(_, then) = marks.iter().rev().find(|&&(at, _)| now.saturating_sub(at) >= secs * 1000)?;
    Some(newest.saturating_sub(then))
}

/// A line for the log: the code in use by when a list last took its pipeline, and the frames
/// shown meanwhile. What the lists took within a frame or two cannot be taken away, whatever the
/// idle time, so this says whether a budget can hold at all (`XCL_SHADER_CODE_AGES=0`: no line;
/// None with no budget too).
pub(super) fn by_age() -> Option<String> {
    static ON: OnceLock<bool> = OnceLock::new();
    if budget() == 0 || !*ON.get_or_init(|| std::env::var("XCL_SHADER_CODE_AGES").map_or(true, |v| v.trim() != "0")) {
        return None;
    }
    let sizes: Vec<(usize, usize)> = METER.lock().unwrap().as_ref()?.seen.iter().map(|(&pso, &(_, bytes))| (pso, bytes.unwrap_or(0))).collect();
    let now = now_ms();
    let (ages, parked, pinned) = {
        let u = USES.lock().unwrap();
        let u = u.as_ref()?;
        (u.ages(&sizes, now), u.parked.len(), u.pinned.len())
    };
    let frames = {
        let f = FRAMES.lock().unwrap();
        AGES_S.map(|s| frames_in(&f, now, s).map_or("?".to_string(), |n| n.to_string()))
    };
    Some(format!(
        "D3D12.X: the compiled code in use by when a list last took its pipeline: {} MB ({}) in the last {} s, {} MB ({}) up to {} s ago, {} MB ({}) up to {} s ago, {} MB ({}) longer ago, {} MB ({}) never noted; frames shown in the last {} / {} / {} s: {} / {} / {}; {parked} taken away are still kept for their lists, {pinned} are held by bundles (XCL_SHADER_CODE_AGES=0: no such line)",
        ages[0].1 >> 20, ages[0].0, AGES_S[0],
        ages[1].1 >> 20, ages[1].0, AGES_S[1],
        ages[2].1 >> 20, ages[2].0, AGES_S[2],
        ages[3].1 >> 20, ages[3].0,
        ages[4].1 >> 20, ages[4].0,
        AGES_S[0], AGES_S[1], AGES_S[2],
        frames[0], frames[1], frames[2],
    ))
}

/// Of the pipelines that may be released, each (when a list last took it, its bytes, which), the
/// ones to release for `over` bytes to go: the longest unused first.
pub(super) fn to_release(mut idle: Vec<(u64, usize, usize)>, over: usize) -> Vec<(usize, usize)> {
    idle.sort_unstable();
    let mut freed = 0;
    idle.into_iter()
        .take_while(|&(_, bytes, _)| {
            let more = freed < over;
            freed += bytes;
            more
        })
        .map(|(_, bytes, which)| (which, bytes))
        .collect()
}

/// When a list last took `pso` and the bytes of its code, if it may be released: it is counted,
/// its code was read, no list took it for `idle_ms`, and no bundle took it.
pub(super) fn idle(pso: usize) -> Option<(u64, usize)> {
    let last = USES.lock().unwrap().as_ref()?.candidate(pso)?;
    if now_ms().saturating_sub(last) < idle_ms() {
        return None;
    }
    let bytes = METER.lock().unwrap().as_ref()?.seen.get(&pso)?.1?;
    Some((last, bytes))
}

/// (the bytes of code in use, how many of them are over the budget; 0 with no budget)
pub(super) fn over_budget() -> (usize, usize) {
    let bytes = METER.lock().unwrap().as_ref().map_or(0, |m| m.bytes);
    (bytes, if budget() == 0 { 0 } else { bytes.saturating_sub(budget()) })
}

/// `pso_async` took `pso` from its stand-in and gives XCL's reference to it here: it leaves the
/// sum now and is handed back by `through` for its release.
pub(super) fn retire(pso: usize) {
    USES.lock().unwrap().get_or_insert_with(Uses::default).park(pso, now_ms());
    uncount(pso);
}

/// (the retired pipelines no list can still run with, for `pso_async` to release; how many stay)
pub(super) fn through() -> (Vec<usize>, usize) {
    match USES.lock().unwrap().as_mut() {
        Some(u) => (u.through(now_ms()), u.parked.len()),
        None => (Vec::new(), 0),
    }
}

/// The command allocator `allocator` was reset or released.
pub(super) fn allocator_reset(allocator: usize) {
    if let Some(u) = USES.lock().unwrap().as_mut() {
        u.reset(allocator);
    }
}

fn reader() -> &'static Sender<Held> {
    static TX: OnceLock<Sender<Held>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = channel::<Held>();
        let _ = std::thread::Builder::new().name("xcl-shader-code".into()).spawn(move || {
            for Held(seq, p) in rx {
                let pso = p.as_raw() as usize;
                let bytes = unsafe { p.GetCachedBlob().map_or(0, |b| b.GetBufferSize()) };
                // (before the lock: the last reference's Release is the driver's)
                drop(p);
                let line = METER.lock().unwrap().as_mut().and_then(|m| m.measured(pso, seq, bytes));
                if let Some(line) = line {
                    crate::xlog_info!("{line}");
                }
            }
        });
        tx
    })
}

/// The host pipeline `pso` (live, not a stand-in) goes onto the title's list `list`.
pub(super) fn used(list: usize, pso: usize) {
    if pso != 0 && on() {
        bundle_takes(list, pso);
        used_on(if budget() == 0 { 0 } else { super::list_x::allocator_of(list) }, pso);
    }
}

/// `used` for a list that records on the command allocator `allocator` (0: not known).
pub(super) fn used_on(allocator: usize, pso: usize) {
    if pso == 0 || !on() {
        return;
    }
    // with `XCL_PSO_FREE_IDLE` a pipeline state may be taken away under memory pressure too, and
    // then its lists' holds are what keeps it alive until they are through (`through`)
    if budget() != 0 || super::real::pso_free_idle() {
        taken_on(allocator, pso);
        if budget() != 0 {
            super::pso_async::start_code_release();
        }
    }
    let generation = GENERATION.load(Ordering::Acquire);
    let known = KNOWN.with(|k| {
        let mut k = k.borrow_mut();
        if k.0 != generation {
            *k = (generation, HashSet::new());
        }
        !k.1.insert(pso)
    });
    if known {
        return;
    }
    let Some(seq) = METER.lock().unwrap().get_or_insert_with(Meter::default).sight(pso) else { return };
    let raw = pso as *mut c_void;
    if let Some(p) = unsafe { ID3D12PipelineState::from_raw_borrowed(&raw) } {
        let _ = reader().send(Held(seq, p.clone()));
    }
}

/// The list `list` takes `pso` by a path the meter does not count (a list's first pipeline at its
/// `Reset` or creation, a pipeline XCL sets again for the title): with a budget its hold is noted,
/// with none nothing happens. `pso` may be any pipeline, a stand-in too: only its address is kept.
pub(super) fn taken(list: usize, pso: usize) {
    if pso != 0 && budget() != 0 {
        bundle_takes(list, pso);
        taken_on(super::list_x::allocator_of(list), pso);
    }
}

/// `taken_on` for a path only a draw's pipeline takes (the `Reset` of the list a put-off draw
/// is recorded on): noted when graphics pipelines are taken away.
pub(super) fn drawn_on(allocator: usize, pso: usize) {
    if graphics() {
        taken_on(allocator, pso);
    }
}

/// The list `list` takes `pso`: a bundle's is pinned (`in_bundle`).
pub(super) fn bundle_takes(list: usize, pso: usize) {
    if pso != 0 && budget() != 0 && super::list_x::is_bundle(list) {
        in_bundle(pso);
    }
}

/// A bundle takes `pso` (any pipeline, a stand-in too: only its address is kept): with a budget
/// it is pinned, and stays with its stand-in until XCL releases it.
pub(super) fn in_bundle(pso: usize) {
    if pso != 0 && budget() != 0 {
        USES.lock().unwrap().get_or_insert_with(Uses::default).pinned.insert(pso);
    }
}

/// The lists `lists` are handed to a queue (the title's and the ones they run as): what each
/// took since its allocator's last `Reset` is in use now (`Uses::run`).
pub(super) fn submitted(lists: impl Iterator<Item = usize>) {
    if budget() == 0 {
        return;
    }
    let allocators: Vec<usize> = lists.map(super::list_x::allocator_of).filter(|&a| a != 0).collect();
    if allocators.is_empty() {
        return;
    }
    if let Some(u) = USES.lock().unwrap().as_mut() {
        let now = now_ms();
        for a in allocators {
            u.run(a, now);
        }
    }
}

/// `taken` for a list that records on the command allocator `allocator` (0: not known).
pub(super) fn taken_on(allocator: usize, pso: usize) {
    if pso != 0 && budget() != 0 {
        USES.lock().unwrap().get_or_insert_with(Uses::default).take(allocator, pso, now_ms());
    }
}

/// XCL releases the host pipeline `pso`.
pub(super) fn gone(pso: usize) {
    uncount(pso);
    if let Some(u) = USES.lock().unwrap().as_mut() {
        u.pinned.remove(&pso);
    }
}

/// `pso` leaves the sum and the pipelines that may be taken away.
fn uncount(pso: usize) {
    if METER.lock().unwrap().as_mut().is_some_and(|m| m.forget(pso)) {
        GENERATION.fetch_add(1, Ordering::AcqRel);
    }
    if let Some(u) = USES.lock().unwrap().as_mut() {
        u.last.remove(&pso);
    }
}

/// At the device's removal: what was in use.
pub(super) fn log_removal() {
    match METER.lock().unwrap().as_ref() {
        Some(m) => crate::xlog_error!("shader code in use at the removal: {}", m.summary()),
        None if on() => crate::xlog_error!("shader code in use at the removal: no pipeline was bound"),
        None => (),
    }
    if let Some(line) = by_age() {
        crate::xlog_error!("at the removal: {line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pipeline_is_counted_once_and_each_step_is_logged_once() {
        let mut m = Meter::default();
        assert_eq!(m.sight(0x10), Some(1));
        assert_eq!(m.sight(0x10), None);
        assert_eq!(m.sight(0x20), Some(2));
        assert_eq!(m.measured(0x10, 1, 20 << 20), None);
        assert_eq!(m.measured(0x10, 1, 20 << 20), None, "a pipeline is read once");
        let line = m.measured(0x20, 2, 20 << 20).expect("40 MB is past the first step");
        assert!(line.contains("2 pipelines") && line.contains("hold 40 MB") && line.contains("the largest 20480 KB") && line.contains("XCL_SHADER_CODE_METER=0"), "{line}");
        assert_eq!(m.sight(0x30), Some(3));
        assert_eq!(m.measured(0x30, 3, 1 << 20), None, "41 MB is in the step already logged");
        assert!(m.summary().contains("3 pipelines") && m.summary().contains("0 of them not read yet") && m.summary().contains("about 455 MB"), "{}", m.summary());
    }

    #[test]
    fn a_released_pipeline_leaves_the_sum_and_its_address_counts_again() {
        let mut m = Meter::default();
        assert_eq!(m.sight(0x10), Some(1));
        assert_eq!(m.measured(0x10, 1, 8 << 20), None);
        assert!(m.forget(0x10));
        assert!(!m.forget(0x10), "counted once, forgotten once");
        assert_eq!((m.bytes, m.gone), (0, (1, 8 << 20)));
        // another pipeline at the same address; the first one's late reading is not its size
        assert_eq!(m.sight(0x10), Some(2));
        assert_eq!(m.measured(0x10, 1, 99 << 20), None);
        assert_eq!(m.bytes, 0);
        assert!(m.summary().contains("1 of them not read yet"), "{}", m.summary());
        assert_eq!(m.measured(0x10, 2, 3 << 20), None);
        assert_eq!((m.bytes, m.largest), (3 << 20, 8 << 20));
        // released before it was read: nothing to take from the sum
        assert_eq!(m.sight(0x20), Some(3));
        assert!(m.forget(0x20));
        assert_eq!((m.bytes, m.gone), (3 << 20, (2, 8 << 20)));
    }

    #[test]
    fn the_longest_unused_go_first_and_no_more_than_the_budget_asks() {
        // (when last taken, bytes, which)
        let idle = vec![(900, 40, 1), (100, 30, 2), (500, 50, 3), (100, 10, 4)];
        assert_eq!(to_release(idle.clone(), 0), [(0usize, 0usize); 0]);
        assert_eq!(to_release(idle.clone(), 1), [(4, 10)]);
        assert_eq!(to_release(idle.clone(), 10), [(4, 10)]);
        assert_eq!(to_release(idle.clone(), 41), [(4, 10), (2, 30), (3, 50)]);
        assert_eq!(to_release(idle.clone(), 10_000), [(4, 10), (2, 30), (3, 50), (1, 40)]);
    }

    #[test]
    fn a_pipeline_taken_away_is_kept_until_its_lists_are_through() {
        let mut u = Uses::default();
        // two lists, on the allocators 0xA and 0xB, took the pipeline 0x10; a list whose
        // allocator is not known took 0x20
        u.take(0xA, 0x10, 1_000);
        u.take(0xB, 0x10, 2_000);
        u.take(0, 0x20, 2_000);
        assert_eq!(u.last[&0x10], 2_000);
        u.park(0x10, 40_000);
        u.park(0x20, 40_000);
        assert!(!u.last.contains_key(&0x10), "a pipeline taken away is no candidate again");
        assert_eq!(u.through(40_000 + GRACE_MS), [0usize; 0], "both allocators' lists may still run");
        u.reset(0xA);
        assert_eq!(u.through(41_000 + GRACE_MS), [0usize; 0], "0xB's lists may still run");
        u.reset(0xB);
        u.reset(0);
        assert_eq!(u.through(42_000 + GRACE_MS), [0x10], "0x20's list has no allocator to reset: it waits out the time");
        assert_eq!(u.through(42_000 + GRACE_MS), [0usize; 0], "handed back once");
        assert_eq!(u.through(2_000 + HOLD_MAX_MS - 1), [0usize; 0]);
        assert_eq!(u.through(2_000 + HOLD_MAX_MS), [0x20]);
        assert!(u.parked.is_empty() && u.held.is_empty());
    }

    #[test]
    fn a_pipeline_taken_away_is_kept_through_the_grace_and_past_a_late_take() {
        let mut u = Uses::default();
        u.take(0xA, 0x10, 1_000);
        u.reset(0xA);
        u.park(0x10, 50_000);
        assert_eq!(u.through(50_000 + GRACE_MS - 1), [0usize; 0], "a bind that read it just before may not have reached its list");
        // that bind's list takes it now
        u.take(0xC, 0x10, 50_100);
        assert_eq!(u.through(50_000 + GRACE_MS), [0usize; 0]);
        u.reset(0xC);
        assert_eq!(u.through(50_001 + GRACE_MS), [0x10]);
        // an allocator never reset holds no longer than HOLD_MAX_MS
        u.take(0xD, 0x30, 60_000);
        u.park(0x30, 100_000);
        assert_eq!(u.through(60_000 + HOLD_MAX_MS - 1), [0usize; 0]);
        assert_eq!(u.through(60_000 + HOLD_MAX_MS), [0x30]);
    }

    #[test]
    fn a_pipeline_a_bundle_took_stays_with_its_stand_in() {
        let mut u = Uses::default();
        u.take(0xA, 0x10, 1_000);
        u.take(0xA, 0x20, 1_000);
        assert_eq!((u.candidate(0x10), u.candidate(0x20), u.candidate(0x30)), (Some(1_000), Some(1_000), None));
        u.pinned.insert(0x10);
        assert_eq!((u.candidate(0x10), u.candidate(0x20)), (None, Some(1_000)), "a bundle runs it with no bind: never idle");
        // 0x20 was taken away just before a bundle's bind, which had read it, reached its list
        u.park(0x20, 50_000);
        u.take(0xB, 0x20, 50_100);
        u.pinned.insert(0x20);
        u.reset(0xA);
        u.reset(0xB);
        assert_eq!(u.through(50_100 + HOLD_MAX_MS), [0usize; 0], "the bundle may run it at any time: kept");
        assert_eq!(u.parked.len(), 1);
        // XCL releases it for good (its stand-in went): no longer pinned
        u.pinned.remove(&0x20);
        assert_eq!(u.through(50_100 + HOLD_MAX_MS), [0x20]);
    }

    #[test]
    fn the_code_in_use_is_sorted_by_when_a_list_last_took_it() {
        let mut u = Uses::default();
        let now = 500_000;
        u.take(0xA, 0x10, now - 1_000);
        u.take(0xA, 0x20, now - 4_999);
        u.take(0xA, 0x30, now - 5_000);
        u.take(0xA, 0x40, now - 100_000);
        u.take(0xA, 0x50, now - 120_000);
        let sizes = [(0x10, 1 << 20), (0x20, 2 << 20), (0x30, 4 << 20), (0x40, 8 << 20), (0x50, 16 << 20), (0x60, 32 << 20)];
        assert_eq!(u.ages(&sizes, now), [(2, 3 << 20), (1, 4 << 20), (1, 8 << 20), (1, 16 << 20), (1, 32 << 20)]);
        let marks: std::collections::VecDeque<(u64, u64)> = [(now - 40_000, 100), (now - 30_000, 101), (now - 6_000, 103), (now - 5_000, 104), (now - 1_000, 104), (now, 106)].into();
        assert_eq!((frames_in(&marks, now, 5), frames_in(&marks, now, 30), frames_in(&marks, now, 120)), (Some(2), Some(5), None));
        assert_eq!(frames_in(&Default::default(), now, 5), None);
    }

    #[test]
    fn a_list_run_again_without_a_reset_keeps_its_pipelines() {
        let mut u = Uses::default();
        u.take(0xA, 0x10, 1_000);
        u.take(0xA, 0x20, 1_000);
        u.take(0xB, 0x30, 1_000);
        // 0x20 was taken away; then the list recorded on 0xA is handed to a queue again
        u.park(0x20, 80_000);
        u.run(0xA, 90_000);
        u.run(0xC, 90_000);
        assert_eq!((u.candidate(0x10), u.candidate(0x30)), (Some(90_000), Some(1_000)), "used at its submit, not only when recorded");
        assert!(!u.last.contains_key(&0x20), "one taken away is no candidate again");
        assert_eq!(u.through(1_000 + HOLD_MAX_MS), [0usize; 0], "the list that runs again still holds it");
        assert_eq!(u.through(90_000 + HOLD_MAX_MS), [0x20]);
    }
}
