//! A cap on the video memory committed for the title's placed textures (`XCL_TEXTURE_GATE`).
//!
//! Why. A sampled texture of the title's that has texels somewhere gets a 64 KiB page of video
//! memory for each 64 KiB page of the title's addresses it covers (`vram::Pages`), and the pages
//! come from pool heaps of 64 MiB (`vram::POOL_PAGES`). A pool heap is never given back: the pages
//! of a texture that goes are reused by the next one, the block is not. One retail session made
//! 101 of them (6 464 MB) by the time its world loaded, 2 050 MB of that already spilled to
//! system memory, and its local segment sat at 7 685 of the 8 526 MiB DXGI budget (401 s), while
//! the pages the title had actually mapped came to 2 927 MB (vram-accounting.md, play-56b): the
//! working set drifts over the budget with nothing giving a block back.
//!
//! What this does. Before such a block is committed (`vram::take_tile`, when a sampled texture's
//! texels need a page), the local segment's usage from DXGI `QueryVideoMemoryInfo` is compared
//! with a cap: while the block would take it past the cap the block is not committed, and the
//! texels wait in the state a page without memory already has – they read as zero, and the mip is
//! mapped when a later upload asks again (`vram::note_content`). The texture is remembered here;
//! the next title submit after the local segment is back under the cap uploads it again
//! (`title_textures::rearm_gated`), which maps the mips that waited and copies their texels. A
//! texture refused again at that upload waits again, so nothing is dropped: it is late.
//!
//! What it never holds back: a GPU-written texture's pages (a render target, depth buffer or UAV
//! texture needs its memory when its placement is made; the placement would fall back to an arena
//! or a committed resource of its own – the same bytes in a form nothing can share or give back),
//! and a page that comes from a pool heap that already exists (only new blocks are commits).
//!
//! Invariants. `admits` is called from `vram::Pages::take_tile` with the `PAGES` lock held
//! (`PAGES` → here; nothing here takes another lock, so no order back). A refusal commits nothing,
//! maps nothing and signals nothing: the tile mapping state is what it was, so no fence is owed
//! and `vram::before_execute` has nothing new to wait for; the unmapped tiles read as zero
//! (tiled resources tier 2) and writes to them go nowhere, as they already did without memory.
//! `ready` is called from `title_textures::refresh`, before the upload batch of the same submit,
//! and only marks textures for upload: the mapping that follows is ordered before that upload by
//! `vram::before_execute` on the upload queue, which is the ordering `note_content` relies on. A
//! texture released while it waits is forgotten (`vram::forget_pages` → `forget`); its COM object
//! goes with its pages, so there is nothing left to be late for.
//!
//! Provenance. This is the idea of the gate another NVIDIA fork of an Xbox title's host calls
//! POOL_GATE (its long sessions ran 0.8 to 1.0 GB over the VRAM budget); no code comes from it –
//! the shape above is this tree's own (`Pages::take_tile`, `note_content`, `refresh`), so nothing
//! here carries that fork's licence.
//!
//! Switches. `XCL_TEXTURE_GATE=1` turns it on (off: nothing is compared, nothing waits, nothing is
//! counted). `XCL_TEXTURE_GATE_MB` is the cap in MB; 0 or unset means `PCT` percent of the DXGI
//! local budget at the time of the check. `PCT` is 70: of the 8 526 MiB budget of the PC's RTX
//! 3080 (Series S profile) that is 5 968 MiB, the 6.0 GB peak this phase is for. A card whose
//! budget models a different amount of video memory should set the cap in MB instead (a 32 GB
//! card's 70 % is 21 GB, which nothing here should use for an 8 GB console).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The cap as a percentage of the DXGI local budget when `XCL_TEXTURE_GATE_MB` is 0.
const PCT: u64 = 70;

/// How often the gate asks DXGI whether the local segment is back under its cap while textures
/// wait: a query is a factory and an adapter lookup, and the title submits at frame rate (0.2 to
/// 18 a second in the runs of this phase).
const POLL: Duration = Duration::from_millis(250);

/// How long an answer from DXGI is reused while blocks are refused (see `asked`).
const CACHE: Duration = Duration::from_millis(100);

/// `XCL_TEXTURE_GATE` on: anything but unset, `0`, `off` or `false`.
fn switch(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "off" | "false"))
}

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch(std::env::var("XCL_TEXTURE_GATE").ok().as_deref()))
}

/// `XCL_TEXTURE_GATE_MB` in bytes. None: unset, not a number, or 0 (the budget percentage).
fn cap_mb(raw: Option<&str>) -> Option<u64> {
    raw?.trim().parse::<u64>().ok().filter(|&v| v != 0)?.checked_mul(1 << 20)
}

fn cap_bytes() -> Option<u64> {
    static MB: OnceLock<Option<u64>> = OnceLock::new();
    *MB.get_or_init(|| cap_mb(std::env::var("XCL_TEXTURE_GATE_MB").ok().as_deref()))
}

/// The cap on local video memory a `budget` gives: `XCL_TEXTURE_GATE_MB` if set, else `PCT` %.
fn cap(mb: Option<u64>, budget: u64) -> u64 {
    mb.unwrap_or(budget / 100 * PCT + budget % 100 * PCT / 100)
}

/// Whether a block of `bytes` fits: `usage` and the block together under `cap`.
fn fits(usage: u64, bytes: u64, cap: u64) -> bool {
    usage.saturating_add(bytes) <= cap
}

/// The local segment's usage and budget, as DXGI tells them.
fn local() -> Option<(u64, u64)> {
    let info = super::real::video_memory_info()?;
    Some((info[0].CurrentUsage, info[0].Budget))
}

/// `local()`, reused for `CACHE`: a shut gate is asked once for every page a texture wants, and a
/// page is often one mip of a dozen, while a query is a DXGI factory and an adapter lookup. The
/// local segment cannot move a whole block in 100 ms.
fn asked(state: &mut Gate, now: Instant) -> Option<(u64, u64)> {
    if let Some((at, usage, budget)) = state.checked {
        if now.duration_since(at) < CACHE {
            return Some((usage, budget));
        }
    }
    let (usage, budget) = local()?;
    state.checked = Some((now, usage, budget));
    Some((usage, budget))
}

#[derive(Default)]
struct Gate {
    /// the textures a block was refused for and when the first refusal came, oldest first
    waiting: Vec<(usize, Instant)>,
    /// block commits refused, and their bytes
    gated: u64,
    bytes: u64,
    /// textures let through to an upload again, and the longest and the last wait of those
    through: u64,
    longest: Duration,
    last: Duration,
    /// textures released while the gate held them back
    gone: u64,
    /// the size of the block the last check was for
    block: u64,
    /// the last local usage, budget and cap seen
    seen: (u64, u64, u64),
    /// the last answer from DXGI and when it came (`asked`)
    checked: Option<(Instant, u64, u64)>,
    /// when `ready` last looked
    polled: Option<Instant>,
}

impl Gate {
    /// The counts, for a log line: what was held back, what the segment is at, which textures
    /// waited and for how long.
    fn summary(&self, now: Instant) -> String {
        let (usage, budget, cap) = self.seen;
        let oldest = self.waiting.first().map_or(0.0, |e| now.saturating_duration_since(e.1).as_secs_f64());
        format!(
            "{} block commits of {} MB held back ({} MB); local video memory {} of {} MB in use, cap {} MB; {} let through after waiting (longest {:.1} s, last {:.1} s), {} released while waiting, {} waiting now (oldest {:.1} s)",
            self.gated,
            self.block >> 20,
            self.bytes >> 20,
            usage >> 20,
            budget >> 20,
            cap >> 20,
            self.through,
            self.longest.as_secs_f64(),
            self.last.as_secs_f64(),
            self.gone,
            self.waiting.len(),
            oldest
        )
    }
}

static GATE: Mutex<Option<Gate>> = Mutex::new(None);

fn gate() -> std::sync::MutexGuard<'static, Option<Gate>> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether a block of `bytes` of video memory may be committed now for the texture `key` (its
/// resource as an address): false while the local segment would pass the cap, and `key` waits
/// until `ready` (or `forget`, if it goes first). Counted and logged (the first refusal, then
/// each doubling of the count). A segment that tells no budget is gated not at all.
pub(super) fn admits(bytes: u64, key: usize) -> bool {
    if !on() {
        return true;
    }
    let now = Instant::now();
    let mut g = gate();
    let s = g.get_or_insert_with(Gate::default);
    let Some((usage, budget)) = asked(s, now) else { return true };
    let cap = cap(cap_bytes(), budget);
    s.seen = (usage, budget, cap);
    s.block = bytes;
    static OPENED: AtomicBool = AtomicBool::new(false);
    let opened = !OPENED.swap(true, Ordering::Relaxed);
    if cap == 0 || fits(usage, bytes, cap) {
        drop(g);
        if opened {
            log_opened(bytes, cap, usage, budget);
        }
        return true;
    }
    s.gated += 1;
    s.bytes = s.bytes.saturating_add(bytes);
    if !s.waiting.iter().any(|e| e.0 == key) {
        s.waiting.push((key, now));
    }
    let line = (opened || s.gated == 1 || s.gated.is_power_of_two()).then(|| s.summary(now));
    drop(g);
    if opened {
        log_opened(bytes, cap, usage, budget);
    }
    if let Some(line) = line {
        crate::xlog_info!("D3D12.X: texture gate: {line}");
    }
    false
}

/// The one line that says the gate is on, what a block is and where its cap comes from.
fn log_opened(bytes: u64, cap: u64, usage: u64, budget: u64) {
    let source = if cap_bytes().is_some() { "XCL_TEXTURE_GATE_MB".to_string() } else { format!("{PCT}% of the DXGI local budget") };
    crate::xlog_info!(
        "D3D12.X: texture gate on (XCL_TEXTURE_GATE): a block of {} MB of pages is committed only while the local video memory stays under {} MB ({}), now {} of {} MB in use (XCL_TEXTURE_GATE=0 turns it off)",
        bytes >> 20,
        cap >> 20,
        source,
        usage >> 20,
        budget >> 20
    );
}

/// The textures whose texels the gate held back that may be uploaded again now: the local segment
/// is back under its cap within one block (`POLL` at most every 250 ms). An upload of one maps the
/// mips that waited (`vram::note_content`); one the gate refuses there again waits again, so no
/// texture leaves this list for good before it has its memory.
pub(super) fn ready() -> Vec<usize> {
    if !on() {
        return Vec::new();
    }
    let mut g = gate();
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.waiting.is_empty() {
        return Vec::new();
    }
    let now = Instant::now();
    if s.polled.is_some_and(|at| now.duration_since(at) < POLL) {
        return Vec::new();
    }
    s.polled = Some(now);
    let Some((usage, budget)) = asked(s, now) else { return Vec::new() };
    let cap = cap(cap_bytes(), budget);
    s.seen = (usage, budget, cap);
    if cap != 0 && !fits(usage, s.block, cap) {
        return Vec::new();
    }
    let released = std::mem::take(&mut s.waiting);
    let wait = released.iter().map(|e| now.saturating_duration_since(e.1)).max().unwrap_or_default();
    s.through += released.len() as u64;
    s.longest = s.longest.max(wait);
    s.last = wait;
    let first = s.through == released.len() as u64;
    let keys: Vec<usize> = released.into_iter().map(|e| e.0).collect();
    if first || s.through.is_power_of_two() {
        let line = s.summary(now);
        drop(g);
        crate::xlog_info!("D3D12.X: texture gate: {line}");
    }
    keys
}

/// A texture released for good: one that was waiting is waited for no more.
pub(super) fn forget(key: usize) {
    if !on() {
        return;
    }
    let mut g = gate();
    let Some(s) = g.as_mut() else { return };
    if let Some(i) = s.waiting.iter().position(|e| e.0 == key) {
        s.waiting.remove(i);
        s.gone += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const MIB: u64 = 1 << 20;
    /// the DXGI local budget of the PC's RTX 3080 in the Series S profile (vram-accounting.md)
    const BUDGET: u64 = 8526 * MIB;
    const BLOCK: u64 = 64 * MIB;

    #[test]
    fn the_switch_is_off_unless_it_is_set() {
        assert!(!switch(None));
        assert!(!switch(Some("0")));
        assert!(!switch(Some(" off ")));
        assert!(!switch(Some("FALSE")));
        assert!(!switch(Some("")));
        assert!(switch(Some("1")));
        assert!(switch(Some("on")));
        assert!(switch(Some("true")));
    }

    #[test]
    fn the_cap_is_mb_or_a_percentage_of_the_budget() {
        assert_eq!(cap_mb(None), None);
        assert_eq!(cap_mb(Some("0")), None);
        assert_eq!(cap_mb(Some("6000")), Some(6000 * MIB));
        assert_eq!(cap_mb(Some(" 6000 ")), Some(6000 * MIB));
        assert_eq!(cap_mb(Some("six")), None);
        assert_eq!(cap_mb(Some("18446744073709551615")), None);
        assert_eq!(cap(None, BUDGET), BUDGET / 100 * 70 + BUDGET % 100 * 70 / 100);
        assert_eq!(cap(None, BUDGET) >> 20, 5968, "5 968 MB of the 8 526 MB budget: the 6.0 GB this phase is for");
        assert_eq!(cap(None, 0), 0);
        assert_eq!(cap(Some(6000 * MIB), BUDGET), 6000 * MIB);
    }

    #[test]
    fn a_block_fits_until_it_would_pass_the_cap() {
        let cap = cap(None, BUDGET);
        assert!(fits(5000 * MIB, BLOCK, cap), "a block well under the cap");
        assert!(fits(cap - BLOCK, BLOCK, cap), "a block ending exactly at the cap");
        assert!(!fits(cap - BLOCK + 1, BLOCK, cap), "a block one byte past it");
        assert!(!fits(cap, BLOCK, cap), "any block with no room at all");
        assert!(fits(u64::MAX - 1, BLOCK, u64::MAX), "no overflow at the end of the range");
    }

    /// A gate with a fixed clock: refusals, waits and releases, as `admits`, `ready` and `forget`
    /// keep them (without DXGI).
    fn gated(state: &mut Gate, key: usize, now: Instant) {
        state.gated += 1;
        state.bytes += BLOCK;
        state.block = BLOCK;
        if !state.waiting.iter().any(|e| e.0 == key) {
            state.waiting.push((key, now));
        }
    }

    #[test]
    fn the_oldest_refusal_is_measured_from_its_first_block() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut s = Gate::default();
        gated(&mut s, 7, at(0));
        gated(&mut s, 9, at(1000));
        gated(&mut s, 7, at(2000));
        assert_eq!(s.waiting, vec![(7, at(0)), (9, at(1000))], "one entry per texture, from its first refusal");
        assert_eq!((s.gated, s.bytes), (3, 3 * BLOCK));
        assert_eq!(s.waiting.first().map(|e| at(2500).saturating_duration_since(e.1)), Some(Duration::from_millis(2500)));
        assert!(s.summary(at(2500)).contains("3 block commits of 64 MB held back (192 MB)"));
        assert!(s.summary(at(2500)).contains("2 waiting now (oldest 2.5 s)"));
    }

    #[test]
    fn the_wait_of_a_texture_that_is_let_through_is_kept_as_the_longest() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut s = Gate::default();
        gated(&mut s, 7, at(0));
        gated(&mut s, 9, at(1000));
        let released = std::mem::take(&mut s.waiting);
        let wait = released.iter().map(|e| at(3000).saturating_duration_since(e.1)).max().unwrap();
        s.through += released.len() as u64;
        s.longest = s.longest.max(wait);
        s.last = wait;
        assert_eq!((s.through, s.longest, s.last), (2, Duration::from_millis(3000), Duration::from_millis(3000)), "the oldest waited 3 s");
        gated(&mut s, 7, at(4000));
        let released = std::mem::take(&mut s.waiting);
        let wait = released.iter().map(|e| at(5000).saturating_duration_since(e.1)).max().unwrap();
        s.through += released.len() as u64;
        s.longest = s.longest.max(wait);
        s.last = wait;
        assert_eq!((s.through, s.longest, s.last), (3, Duration::from_millis(3000), Duration::from_millis(1000)), "a short wait does not lower the longest");
    }

    #[test]
    fn a_texture_released_while_it_waits_is_not_let_through() {
        let t0 = Instant::now();
        let mut s = Gate::default();
        gated(&mut s, 7, t0);
        gated(&mut s, 9, t0);
        if let Some(i) = s.waiting.iter().position(|e| e.0 == 7) {
            s.waiting.remove(i);
            s.gone += 1;
        }
        assert_eq!(s.waiting, vec![(9, t0)]);
        assert_eq!((s.gone, s.through), (1, 0));
        assert!(s.summary(t0).contains("1 released while waiting, 1 waiting now"));
    }
}
