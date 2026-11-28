//! When memory a title has stopped using goes back, and how much of it is kept for reuse.
//!
//! XCL frees early. A page of video memory no target uses any more goes back to the pool, an
//! arena heap is released 2 s after its last resource, and – with `XCL_VRAM_BUDGET` – an
//! allocation cold for long enough is evicted from local video memory so another can have it.
//! Each of those takes memory the title may be about to use again: a title that re-makes a
//! render target at an address it released a moment ago reads what it rendered there (that is
//! the console's model, and `vram`'s `IDLE` exists for it), and a texture sampled again after
//! eviction pays a `MakeResident` of its whole heap before the draw that samples it can run.
//! The bounds here are what keeps a free late enough to be safe and early enough to be worth
//! doing:
//!
//! * `XCL_FREE_AFTER_MS` (default 2000, the tree's `IDLE` before this): the wall time since the
//!   thing was last used. A title at 60 fps re-makes a target within a frame or two of releasing
//!   it; a title at the 1–2 fps this emulator reaches in the world spends longer than a second
//!   in a frame, so a time bound alone cannot say "not this frame".
//! * `XCL_FREE_AFTER_FRAMES` (default 0: off): whole presented frames since the frame it was
//!   last used in. Off by default because the tree's frees have always been timed, and because
//!   a title that has not presented yet has no frames to count (then only the time bound
//!   applies). Both bounds that are set must pass: one alone can free memory the other still
//!   protects.
//! * `XCL_FREE_POOL_MB` (default 0: off): freed arena heaps up to this many MiB are kept by size
//!   and alignment for the next arena of their class instead of being released. A title that
//!   makes and drops a 256 MiB depth arena per level streams the same size again and again, and
//!   a kept heap is that class's block reused: no `CreateHeap` and no new video memory.
//!
//! A freed block goes back only after the submission that last used it has completed: for an
//! eviction that is the residency fence (`residency`'s `LastUse`), for a pooled heap the same
//! fence read through `residency::completed`. Neither bound here replaces that check – these
//! only say how long after the last use a free may be considered at all.
//!
//! All three are read once, on first use, and logged once when any of them is not the default.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The tree's `IDLE`, and the default `XCL_FREE_AFTER_MS`.
const DEFAULT_MS: u64 = 2000;

/// The bounds an early free waits out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Wait {
    /// presented frames without use (`XCL_FREE_AFTER_FRAMES`)
    pub frames: u64,
    /// wall time without use (`XCL_FREE_AFTER_MS`)
    pub idle: Duration,
}

impl Wait {
    /// No bound at all: what the tree freed before these switches existed.
    pub(super) const NONE: Wait = Wait { frames: 0, idle: Duration::ZERO };

    /// Whether something last used in frame `last_frame`, at `last_at`, may go now (`now`, in
    /// frame `frame`): every bound that is set has passed.
    ///
    /// A frame bound with no frame clock yet (`frame == 0`, before the title's first present)
    /// leaves the decision to the time bound: nothing has been presented, so counting frames
    /// would keep memory a load needs freed for as long as the load lasts.
    pub(super) fn due(&self, now: Instant, frame: u64, last_at: Instant, last_frame: u64) -> bool {
        let idle = self.idle.is_zero() || now.saturating_duration_since(last_at) >= self.idle;
        let frames = self.frames == 0 || frame == 0 || frame.saturating_sub(last_frame) >= self.frames;
        idle && frames
    }
}

/// A count from an environment variable: `None` when unset or not a number.
fn count(raw: Option<&str>) -> Option<u64> {
    raw?.trim().parse::<u64>().ok()
}

/// `XCL_FREE_AFTER_FRAMES` and `XCL_FREE_AFTER_MS` as given (`None`: unset): the defaults are 0
/// frames (off) and 2000 ms. A variable that is set but not a number reads as off – a typo in a
/// log-run's environment must not make frees wait forever.
fn bounds(frames: Option<&str>, ms: Option<&str>) -> Wait {
    let idle = match ms {
        None => DEFAULT_MS,
        Some(_) => count(ms).unwrap_or(DEFAULT_MS),
    };
    Wait { frames: count(frames).unwrap_or(0), idle: Duration::from_millis(idle) }
}

/// The configured bounds, read once.
pub(super) fn wait() -> Wait {
    static WAIT: OnceLock<Wait> = OnceLock::new();
    *WAIT.get_or_init(|| {
        let frames = std::env::var("XCL_FREE_AFTER_FRAMES").ok();
        let ms = std::env::var("XCL_FREE_AFTER_MS").ok();
        let wait = bounds(frames.as_deref(), ms.as_deref());
        if wait != DEFAULT_WAIT {
            crate::xlog_info!(
                "D3D12.X: an early free waits out {} frames and {} ms without use (XCL_FREE_AFTER_FRAMES, XCL_FREE_AFTER_MS)",
                wait.frames, wait.idle.as_millis(),
            );
        }
        wait
    })
}

/// `Wait { frames: 0, idle: 2000 ms }`, the defaults.
const DEFAULT_WAIT: Wait = Wait { frames: 0, idle: Duration::from_millis(DEFAULT_MS) };

/// Whole frames the title has presented (`present::presents`): 0 before its first one.
pub(super) fn frame() -> u64 {
    super::present::presents()
}

/// `XCL_FREE_POOL_MB` in bytes (0: nothing is pooled, a freed heap is released as before). A
/// value that is not a number reads as off, and one too large for the machine's `usize` is
/// clamped rather than wrapping into a small pool.
pub(super) fn pool_bytes() -> usize {
    static POOL: OnceLock<usize> = OnceLock::new();
    *POOL.get_or_init(|| {
        let bytes = pool(std::env::var("XCL_FREE_POOL_MB").ok().as_deref());
        if bytes != 0 {
            crate::xlog_info!("D3D12.X: freed arena heaps up to {} MiB are kept for reuse (XCL_FREE_POOL_MB)", bytes >> 20);
        }
        bytes
    })
}

/// The pool cap `raw` asks for, in bytes.
fn pool(raw: Option<&str>) -> usize {
    let mib = count(raw).unwrap_or(0);
    usize::try_from(mib.saturating_mul(1 << 20)).unwrap_or(usize::MAX / 2).min(usize::MAX / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn the_defaults_are_the_trees_own_two_second_idle_and_no_frame_bound() {
        assert_eq!(bounds(None, None), Wait { frames: 0, idle: Duration::from_millis(2000) });
        assert_eq!(bounds(Some("0"), Some("2000")), bounds(None, None));
        // the plan's 1-4 s window, and the frame bound on
        assert_eq!(bounds(Some("3"), Some("1000")), Wait { frames: 3, idle: Duration::from_millis(1000) });
        assert_eq!(bounds(Some("2"), Some("4000")), Wait { frames: 2, idle: Duration::from_millis(4000) });
    }

    #[test]
    fn a_variable_that_is_not_a_number_reads_as_off() {
        assert_eq!(bounds(Some("two"), Some("soon")), Wait { frames: 0, idle: Duration::from_millis(2000) });
        assert_eq!(bounds(Some("-1"), Some("")), Wait { frames: 0, idle: Duration::from_millis(2000) });
        assert_eq!(bounds(Some(" 4 "), Some(" 500 ")), Wait { frames: 4, idle: Duration::from_millis(500) });
    }

    #[test]
    fn no_bound_is_the_old_behaviour() {
        let now = t0();
        assert!(Wait::NONE.due(now, 0, now, 0));
        assert!(Wait::NONE.due(now, 9, now - Duration::from_secs(9), 8));
    }

    #[test]
    fn a_time_bound_waits_it_out() {
        let (now, wait) = (t0(), Wait { frames: 0, idle: Duration::from_millis(2000) });
        let used = now - Duration::from_millis(1999);
        assert!(!wait.due(now, 7, used, 7));
        assert!(wait.due(now, 7, now - Duration::from_millis(2000), 7));
        assert!(wait.due(now, 7, now - Duration::from_secs(30), 7));
    }

    #[test]
    fn a_frame_bound_counts_presented_frames_and_both_bounds_must_pass() {
        let (now, wait) = (t0(), Wait { frames: 3, idle: Duration::from_millis(1000) });
        let long_ago = now - Duration::from_secs(10);
        // long enough by the clock, but the last two frames used it
        assert!(!wait.due(now, 100, long_ago, 98));
        assert!(wait.due(now, 100, long_ago, 97));
        // cold for frames now, but used a moment ago by the clock
        assert!(!wait.due(now, 100, now, 1));
        assert!(wait.due(now, 100, now - Duration::from_millis(1000), 1));
    }

    #[test]
    fn before_the_first_present_only_the_time_bound_applies() {
        let (now, wait) = (t0(), Wait { frames: 4, idle: Duration::from_millis(500) });
        assert!(wait.due(now, 0, now - Duration::from_millis(500), 0), "a load has no frames to count");
        assert!(!wait.due(now, 0, now, 0));
    }

    #[test]
    fn a_free_pool_of_zero_is_off_and_a_huge_one_is_clamped_not_wrapped() {
        assert_eq!(pool(None), 0);
        assert_eq!(pool(Some("0")), 0);
        assert_eq!(pool(Some("256")), 256 << 20);
        assert_eq!(pool(Some("4096")), 4 << 30);
        assert_eq!(pool(Some("no")), 0);
        assert_eq!(pool(Some("18446744073709551615")), usize::MAX / 2);
    }
}
