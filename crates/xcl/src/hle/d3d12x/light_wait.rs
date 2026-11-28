//! `XCL_LIGHT_WAIT` – a pass that renders into a lighting target waits for its pipeline
//! (`pso_async::resolve_wait_light`) instead of being dropped.
//!
//! A lighting target is an image a pass writes that is not one the title presents: a colour
//! target that is not a presented resource, or, for a pass that binds no colour target, the depth
//! image it writes. The presented images are the ones the title shows as they are; every other
//! image it renders into is read by a pass later in the frame, and that is where a deferred
//! renderer keeps its light – scene colour, the light and probe tables, the shadow masks. A pass
//! into one that is dropped leaves it as it was: the frame is dark where that pass would have lit
//! it, and nothing later puts the light back.
//!
//! The tree's other budgets bound a wait by the frame (`XCL_FRAME_STALL_BUDGET_MS`,
//! `XCL_SCREEN_WAIT_MS`, `pso_async::DEPTH_ONLY_BUDGETS`) and then skip every later pass with that
//! pipeline at once. This switch bounds the wait by the pipeline instead.
//!
//! `XCL_LIGHT_WAIT=<ms>`: unset or `0` (default): today. A number: such a pass waits for its
//! pipeline, at most that many milliseconds once a pipeline and that many in all in one frame (the
//! one number bounds both), and the wait is logged with the pipeline, its title name and the call
//! (`waited`). A pipeline not built by then is skipped as before and its later passes are not
//! waited for (one wait a pipeline). A pass over the screen keeps its own budget first
//! (`XCL_SCREEN_WAIT_MS`), and a dispatch keeps the wait it has always had: of the calls the
//! emulator records, only a draw, an indirect draw and a mesh dispatch are skipped for a pipeline
//! still being built.
//!
//! Nothing here changes a translated shader, so it is not in the translation cache's identity.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// `XCL_LIGHT_WAIT`'s value: milliseconds (0, unset or anything that is no number: off).
fn parse(value: &str) -> Option<Duration> {
    let ms = value.trim().parse::<u64>().ok()?;
    (ms != 0).then(|| Duration::from_millis(ms))
}

/// `XCL_LIGHT_WAIT`: how long a pass into a lighting target waits for its pipeline, once a
/// pipeline and in all in one frame (None: it is skipped as before).
pub(super) fn budget() -> Option<Duration> {
    static B: OnceLock<Option<Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let budget = parse(&std::env::var("XCL_LIGHT_WAIT").unwrap_or_default())?;
        crate::xlog_info!(
            "D3D12.X: a pass into a lighting target (a colour target that is not presented, or a depth-only pass) waits up to {} ms for its pipeline, once a pipeline and {} ms a frame in all (XCL_LIGHT_WAIT)",
            budget.as_millis(),
            budget.as_millis()
        );
        Some(budget)
    })
}

/// The lighting target a pass on `list` renders into, as the log names it: the first colour target
/// that is not a presented resource, or the depth image of a pass with no colour target. None: it
/// writes none (or the list bound none since its last reset).
fn lighting_target(list: *mut c_void) -> Option<String> {
    let (handles, depth_only) = super::list_x::targets(list)?;
    for handle in handles {
        let Some(res) = super::present::rtv_resource(handle) else { continue };
        if !super::present::is_presented(res) {
            return Some(shape(res));
        }
    }
    depth_only.then(|| "depth only".to_string())
}

/// the shape of a resource, for the log
fn shape(res: usize) -> String {
    let p = res as *mut c_void;
    let desc = unsafe { ID3D12Resource::from_raw_borrowed(&p) }.map(|r| unsafe { r.GetDesc() });
    desc.map_or_else(|| format!("{res:#x}"), |d| format!("{}x{} fmt {}", d.Width, d.Height, d.Format.0))
}

/// A pass on `list` whose pipeline `pso` is still being built renders into a lighting target: its
/// bind waits within `budget` instead of skipping it (`pso_async::allow_light_wait`).
pub(super) fn mark(list: *mut c_void, pso: usize) {
    if budget().is_none() || !super::pso_async::is_pending(pso) {
        return;
    }
    if let Some(target) = lighting_target(list) {
        super::pso_async::allow_light_wait(pso, &target);
    }
}

/// the frame (`present::presents`) the waits belong to, and what was waited of it
static FRAME_WAITED: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));

/// What is left of this frame's wait, reserved now so that two threads cannot both spend the whole
/// of it (`waited` gives back what its wait did not use). None: the switch is off, or the frame's
/// wait is used up (the pipeline's own wait stays to be taken).
pub(super) fn left() -> Option<Duration> {
    let budget = budget()?;
    let frame = super::present::presents();
    let mut waited = FRAME_WAITED.lock().unwrap();
    if waited.0 != frame {
        *waited = (frame, Duration::ZERO);
    }
    let left = budget.saturating_sub(waited.1);
    if left.is_zero() {
        return None;
    }
    waited.1 += left;
    Some(left)
}

/// What the wait of `granted` (reserved by `left`) used: charged, the rest of the reservation
/// given back.
fn charge(granted: Duration, waited: Duration) {
    let frame = super::present::presents();
    let mut charged = FRAME_WAITED.lock().unwrap();
    if charged.0 != frame {
        *charged = (frame, waited);
        return;
    }
    charged.1 = charged.1.saturating_sub(granted) + waited;
}

/// passes into a lighting target that waited
static WAITS: AtomicU64 = AtomicU64::new(0);
/// the waits above, in ms
static WAIT_MS: AtomicU64 = AtomicU64::new(0);
/// the waits that ended with the pipeline built: their pass is recorded with the real pipeline
static BUILT: AtomicU64 = AtomicU64::new(0);
/// the waits that ended with the pipeline still building: their pass is skipped as before
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// The pass `pass` (the call it was recorded with) waited `waited` of the `granted` its frame
/// reserved (`left`) for its pipeline `pso`: counted, its frame's wait charged, and logged with the
/// pipeline, its name and the target it renders into.
pub(super) fn waited(pass: &str, pso: usize, waited: Duration, granted: Duration, built: bool) {
    let ms = waited.as_millis() as u64;
    WAITS.fetch_add(1, Ordering::Relaxed);
    WAIT_MS.fetch_add(ms, Ordering::Relaxed);
    (if built { &BUILT } else { &SKIPPED }).fetch_add(1, Ordering::Relaxed);
    charge(granted, waited);
    let target = super::pso_async::light_target(pso).unwrap_or_default();
    let what = super::pso_async::what(pso).unwrap_or_default();
    let name = super::real::pso_name(pso).map(|name| format!(" '{name}'")).unwrap_or_default();
    crate::xlog_info!(
        "D3D12.X: {pass} into a lighting target ({target}) waited {ms} ms for its pipeline {pso:#x}{name} ({what}): {}",
        if built { "built, its pass is recorded with the real pipeline" } else { "not built, its pass is skipped as before and later passes with it are not waited for" }
    );
}

/// What the switch did so far, for the 30 s caches line.
pub(super) fn summary() -> String {
    let (waits, ms, built, skipped) = (WAITS.load(Ordering::Relaxed), WAIT_MS.load(Ordering::Relaxed), BUILT.load(Ordering::Relaxed), SKIPPED.load(Ordering::Relaxed));
    match budget() {
        None => "XCL_LIGHT_WAIT off: a pass into a lighting target is skipped while its pipeline is built".into(),
        Some(budget) => format!(
            "lighting passes waited {waits} ({ms} ms in all; {built} built and drawn, {skipped} skipped after the wait; at most {} ms a pipeline and a frame, XCL_LIGHT_WAIT)",
            budget.as_millis()
        ),
    }
}

#[test]
fn the_switch_reads_milliseconds_and_anything_else_is_off() {
    assert_eq!(parse("0"), None);
    assert_eq!(parse(""), None);
    assert_eq!(parse("nope"), None);
    assert_eq!(parse(" 2500 "), Some(Duration::from_millis(2500)));
}
