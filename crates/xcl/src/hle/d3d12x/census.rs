//! `XCL_DRAW_CENSUS=1`: every draw, indirect and dispatch recorded on the lists of one presented
//! frame in every 300, with what it targets - the depth view and its title address, render target
//! 0 and its title address, the pipeline and its pixel shader, whether the pipeline writes depth,
//! and whether the call was skipped (its shader untranslatable) or issued with no instances or
//! groups. The census says which call should have filled a target, and whether it ran at all.
//!
//! Every line starts with "draw census:". The census logs the first four frames after each present
//! that is a multiple of 300 (300, 600, ...), 4000 lines a frame.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DRAW_CENSUS").is_ok_and(|v| v.trim() == "1"))
}

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// `XCL_DRAW_CENSUS_FROM` (default 300): the first present a frame is logged from.
fn from() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| setting("XCL_DRAW_CENSUS_FROM", 300))
}

/// `XCL_DRAW_CENSUS_EVERY` (default 300): presents between the first presents of two logged frames.
fn every() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| setting("XCL_DRAW_CENSUS_EVERY", 300).max(1))
}

/// `XCL_DRAW_CENSUS_SPAN` (default 4): presents logged from each such present.
fn span() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| setting("XCL_DRAW_CENSUS_SPAN", 4).max(1))
}

/// The present being recorded.
static PRESENT: AtomicU64 = AtomicU64::new(0);

/// The first present that is not logged (set by a trigger: the world's entry moves).
static ARMED_UNTIL: AtomicU64 = AtomicU64::new(0);

/// `XCL_DRAW_CENSUS_MIN_DRAWS` (default 200): a present with at least this many draws is the
/// world's, and arms the census at once - a title's menus draw tens, its world thousands.
fn min_draws() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| setting("XCL_DRAW_CENSUS_MIN_DRAWS", 200).max(1))
}

/// Whether the present being recorded is one the census logs: `XCL_DRAW_CENSUS_FROM`'s schedule,
/// or a trigger (`trigger`, the first present drawn by volume).
fn armed() -> bool {
    let n = PRESENT.load(Ordering::Relaxed);
    if n < ARMED_UNTIL.load(Ordering::Relaxed) {
        return true;
    }
    n >= from() && (n - from()) % every() < span()
}

/// Arms the census at once, for `XCL_DRAW_CENSUS_SPAN` presents from this one (`what`: why).
pub(super) fn trigger(what: &str) {
    if !on() {
        return;
    }
    let n = PRESENT.load(Ordering::Relaxed);
    if n < ARMED_UNTIL.load(Ordering::Relaxed) {
        return;
    }
    ARMED_UNTIL.store(n + span(), Ordering::Relaxed);
    crate::xlog_info!("draw census: frame {n} armed by {what}, the next {} present(s) logged", span());
    crate::gpu::pix_capture::world_entry();
}

static LINES: AtomicU64 = AtomicU64::new(0);

const MAX_LINES: u64 = 4000;

#[derive(Default)]
struct Summary {
    frame: u64,
    draws: u64,
    skipped: u64,
    writes: u64,
    nowrites: u64,
    unknown: u64,
    indirect: u64,
    dispatch: u64,
}

static SUMMARY: Mutex<Option<Summary>> = Mutex::new(None);

/// A presented frame.
pub(super) fn present(frames: u64) {
    if on() {
        PRESENT.store(frames, Ordering::Relaxed);
    }
}

/// The summary of the last frame logged, written out before the first line of the next.
fn turn(frame: u64) {
    let mut g = SUMMARY.lock().unwrap();
    let s = g.get_or_insert_with(Summary::default);
    if s.frame == frame {
        return;
    }
    if s.frame != 0 {
        crate::xlog_info!(
            "draw census: frame {}: {} draws ({} skipped), {} with a depth-writing pipeline, {} without, {} unknown; {} indirect, {} dispatches",
            s.frame, s.draws, s.skipped, s.writes, s.nowrites, s.unknown, s.indirect, s.dispatch
        );
    }
    *s = Summary { frame, ..Summary::default() };
    LINES.store(0, Ordering::Relaxed);
    if armed() {
        crate::xlog_info!("draw census: frame {frame}");
    }
}

#[derive(Default)]
struct Lists {
    /// command list -> its depth-stencil view handle
    dsv: HashMap<usize, usize>,
    /// command list -> render target 0's handle
    rt0: HashMap<usize, usize>,
    /// view handle (DSV or RTV) -> the resource it is a view of
    views: HashMap<usize, usize>,
    /// graphics pipeline -> whether it writes depth
    psos: HashMap<usize, bool>,
}

static STATE: Mutex<Option<Lists>> = Mutex::new(None);

fn state<R>(f: impl FnOnce(&mut Lists) -> R) -> R {
    f(STATE.lock().unwrap().get_or_insert_with(Lists::default))
}

/// A depth-stencil view or render target view `handle` on `res`.
pub(super) fn view(handle: usize, res: *mut c_void) {
    if on() && handle != 0 && !res.is_null() {
        state(|s| {
            s.views.insert(handle, res as usize);
        });
    }
}

/// `pso` is made: whether it writes depth (enabled, mask ALL, a test that can pass).
pub(super) fn pso(pso: usize, writes: bool) {
    if on() && pso != 0 {
        state(|s| {
            s.psos.insert(pso, writes);
        });
    }
}

/// OMSetRenderTargets on `list`: its first render target and its depth view.
pub(super) fn render_targets(list: usize, n: u32, rtvs: *const usize, dsv: *const usize) {
    if !on() {
        return;
    }
    let rt0 = if n > 0 && !rtvs.is_null() { unsafe { *rtvs } } else { 0 };
    let dsv = unsafe { dsv.as_ref().copied() }.unwrap_or(0);
    state(|s| {
        if rt0 != 0 { s.rt0.insert(list, rt0); } else { s.rt0.remove(&list); }
        if dsv != 0 { s.dsv.insert(list, dsv); } else { s.dsv.remove(&list); }
    });
}

/// What a call targets on `list`.
struct Targets {
    pso: usize,
    depth: Option<bool>,
    dsv: Option<(usize, Option<usize>)>,
    rt0: Option<(usize, Option<usize>)>,
}

fn targets(list: usize) -> Targets {
    let pso = super::list_x::draw_gate_pso(list as *mut c_void);
    state(|s| Targets {
        pso,
        depth: s.psos.get(&pso).copied(),
        dsv: s.dsv.get(&list).and_then(|h| s.views.get(h).map(|&r| (r, super::real::title_address(r)))),
        rt0: s.rt0.get(&list).and_then(|h| s.views.get(h).map(|&r| (r, super::real::title_address(r)))),
    })
}

fn view_text(view: Option<(usize, Option<usize>)>) -> String {
    match view {
        Some((r, Some(va))) => format!("{r:#x} at {va:#x}"),
        Some((r, None)) => format!("{r:#x} at None"),
        None => "none".to_string(),
    }
}

fn depth_text(writes: Option<bool>) -> &'static str {
    match writes {
        Some(true) => "writes",
        Some(false) => "no-write",
        None => "unknown",
    }
}

fn skipped_text(list: usize) -> &'static str {
    if super::bind::draw_gate_calls() && super::bind::shader_skip(super::list_x::draw_gate_pso(list as *mut c_void)) {
        " SKIPPED (shader untranslated)"
    } else {
        ""
    }
}

/// One logged call, while the frame's room is left.
fn line(args: std::fmt::Arguments<'_>) {
    let frame = PRESENT.load(Ordering::Relaxed);
    turn(frame);
    let n = LINES.fetch_add(1, Ordering::Relaxed);
    if n < MAX_LINES {
        crate::xlog_info!("draw census: {}", args);
    } else if n == MAX_LINES {
        crate::xlog_info!("draw census: frame {frame} truncated at {MAX_LINES} lines");
    }
}

/// A call counted into its frame's summary; a present that reaches
/// `XCL_DRAW_CENSUS_MIN_DRAWS` arms the census.
fn counted(f: impl FnOnce(&mut Summary)) {
    let frame = PRESENT.load(Ordering::Relaxed);
    let draws = {
        let mut g = SUMMARY.lock().unwrap();
        let s = g.get_or_insert_with(Summary::default);
        if s.frame != frame {
            return;
        }
        f(s);
        s.draws + s.dispatch
    };
    if draws == min_draws() {
        trigger("a present drawn by volume");
    }
}

/// A draw or a mesh dispatch on `list`: `kind` and its counts.
pub(super) fn draw(list: usize, kind: &str, counts: std::fmt::Arguments<'_>) {
    if !on() {
        return;
    }
    turn(PRESENT.load(Ordering::Relaxed));
    let t = targets(list);
    let ps = super::real::pso_shaders(t.pso).map_or(0, |(_, ps)| ps);
    counted(|s| {
        s.draws += 1;
        s.skipped += u64::from(super::bind::draw_gate_calls() && super::bind::shader_skip(t.pso));
        match t.depth {
            Some(true) => s.writes += 1,
            Some(false) => s.nowrites += 1,
            None => s.unknown += 1,
        }
    });
    if !armed() {
        return;
    }
    line(format_args!(
        "{} {kind} list {list:#x} {counts}{} pso {:#x} ps {ps:016x} depth {} dsv {} rt0 {}",
        PRESENT.load(Ordering::Relaxed),
        skipped_text(list),
        t.pso,
        depth_text(t.depth),
        view_text(t.dsv),
        view_text(t.rt0)
    ));
}

/// An ExecuteIndirect on `list`: its signature kind, the command limit the title asked for and the
/// limit XCL runs, and the first record's words as they are now.
pub(super) unsafe fn indirect(list: usize, sig: usize, requested: u32, max: u32, args: *mut c_void, args_off: u64, count: *mut c_void, count_off: u64) {
    if !on() {
        return;
    }
    turn(PRESENT.load(Ordering::Relaxed));
    let t = targets(list);
    let ps = super::real::pso_shaders(t.pso).map_or(0, |(_, ps)| ps);
    let kind = super::indirect::signature(sig).map_or(-1, |(k, _, _)| k as i32);
    counted(|s| {
        s.draws += 1;
        s.indirect += 1;
        s.skipped += u64::from(super::bind::draw_gate_calls() && super::bind::shader_skip(t.pso));
        match t.depth {
            Some(true) => s.writes += 1,
            Some(false) => s.nowrites += 1,
            None => s.unknown += 1,
        }
    });
    let words = |res: *mut c_void, off: u64| {
        super::real::title_address(res as usize).map(|va| va as u64 + off).and_then(|va| crate::memmap::title_words(va as usize, 8))
    };
    if !armed() {
        return;
    }
    line(format_args!(
        "{} ExecuteIndirect list {list:#x} signature kind {kind}, {requested} command(s) asked, {max} run{} pso {:#x} ps {ps:016x} depth {} dsv {} rt0 {} arguments {:x?} count {:x?}",
        PRESENT.load(Ordering::Relaxed),
        skipped_text(list),
        t.pso,
        depth_text(t.depth),
        view_text(t.dsv),
        view_text(t.rt0),
        words(args, args_off),
        words(count, count_off).map(|w| w[0])
    ));
}

/// A compute dispatch on `list` with `groups` thread groups.
pub(super) fn dispatch(list: usize, groups: [u32; 3]) {
    if !on() {
        return;
    }
    turn(PRESENT.load(Ordering::Relaxed));
    let pso = super::list_x::draw_gate_pso(list as *mut c_void);
    counted(|s| {
        s.dispatch += 1;
        s.skipped += u64::from(super::bind::draw_gate_calls() && super::bind::shader_skip(pso));
    });
    if !armed() {
        return;
    }
    line(format_args!(
        "{} Dispatch list {list:#x} groups {} {} {}{} pso {pso:#x}",
        PRESENT.load(Ordering::Relaxed),
        groups[0],
        groups[1],
        groups[2],
        skipped_text(list)
    ));
}
