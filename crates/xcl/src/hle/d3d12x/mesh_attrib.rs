//! `XCL_MESH_ATTRIB` (default on): what happened to every mesh draw range the console layer saw.
//!
//! A range is one DispatchMesh, or one command record of a mesh ExecuteIndirect (the title's
//! indirect command buffer holds one DispatchMesh a record). Each range is counted once, by
//! outcome:
//!
//! * `drawn` – it ran: as a host mesh shader, in two passes, or as a PC pipeline's dispatch;
//! * `failed-pipeline` – its pipeline has stand-in shaders: still building, or its translation or
//!   build failed (`bind::note_failed_skip`);
//! * `view-slot` – XCL could not produce what the call binds: its ABI root signature, the memory
//!   windows' views, a unified-memory slot for io/res, or the record memory of a two-pass call;
//! * `gate` – a gate skipped it before it was bound: `XCL_SKIP_SHADERS`, the live skip file, or a
//!   pipeline held while ray tracing is off (`bind::shader_skip`);
//! * `zero-group` – it asked for no thread group (or an indirect command's groups were 0): the
//!   mesh shader has nothing to run;
//! * `never-submitted` – the title's indirect count leaves the range out (a draw range the title
//!   disabled), or a draw with a mesh pipeline asked for no index or instance;
//! * `dropped` – dropped for a reason none of the others names; the line's `why` names it.
//!
//! Each range is also kept by (console pipeline key, meshlet count), so the line names which mesh
//! pipelines run at which LOD and which are missing. Once every 300 presents the counts of the
//! interval go out as one `mesh attrib:` line, bounded to `MAX_LINE` bytes: the outcome totals,
//! then up to `PAIRS_PER_OUTCOME` pairs for each outcome (the missing ones first), then the
//! reasons. A range that a drop path saw before its records were read (an indirect call whose
//! arguments cannot be read, a call refused before its command buffer is looked at) carries the
//! meshlet count `*`.
//!
//! The counts of an indirect call are the title's memory as the CPU holds it when the call is
//! recorded: the count the GPU writes there is the one its last run left, usually the previous
//! frame's (`bind::note_two_pass_cut` has the same caveat). What a record's groups are at its own
//! execution is the GPU's; this says what the range asked for when it was recorded.
//!
//! Cost: one short map update per range – an indirect call reads at most `MAX_COMMANDS` records
//! and lumps the rest into two counts, so ten updates a call at most. The world's runs show about
//! 217,000 mesh calls over a 500 s run, about 430 a second, so the census is at most about 4,000
//! updates a second: well under a tenth of a percent of one core, beside the STATES and pending-
//! pipeline locks a mesh call already takes. The updates happen under one mutex, taken after
//! `key_only` and with nothing else locked inside it. The line is at most `MAX_LINE` bytes every
//! 300 presents.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

/// Every mesh range the emulation saw, by what became of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Outcome {
    Drawn,
    FailedPipeline,
    ViewSlot,
    Gate,
    ZeroGroup,
    NotSubmitted,
    Dropped,
}

const OUTCOMES: usize = 7;
const OUTCOME_NAMES: [&str; OUTCOMES] = ["drawn", "failed-pipeline", "view-slot", "gate", "zero-group", "never-submitted", "dropped"];

/// The order the line names the pairs in: what is missing first, what ran last.
const PREFERENCE: [Outcome; OUTCOMES] =
    [Outcome::NotSubmitted, Outcome::ZeroGroup, Outcome::FailedPipeline, Outcome::ViewSlot, Outcome::Gate, Outcome::Dropped, Outcome::Drawn];

/// A meshlet count not known when the range was recorded.
pub(super) const MESHLETS_ANY: u32 = u32::MAX;

/// Presents between two census lines.
const EVERY: u64 = 300;

/// The most (pipeline, meshlets) pairs kept per interval; the rest are counted in the totals and
/// the line's tail says how many ranges were not itemised.
const MAX_PAIRS: usize = 512;

/// Pairs the line names for each outcome.
const PAIRS_PER_OUTCOME: usize = 8;

/// The line's bound, tail included.
const MAX_LINE: usize = 4096;

/// The most command records an indirect call's census reads (the title's count, and its first
/// records' group counts).
const MAX_COMMANDS: u32 = 8;

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_MESH_ATTRIB").map_or(true, |v| v.trim() != "0");
        if on {
            crate::xlog_info!("D3D12.X: mesh attribution on (XCL_MESH_ATTRIB=0 turns it off): one census line every {EVERY} presents");
        }
        on
    })
}

/// A group count as a census key: a console mesh program runs one meshlet a group.
pub(super) fn meshlet_count(groups: u64) -> u32 {
    groups.min(u64::from(MESHLETS_ANY) - 1) as u32
}

/// A range of `groups` meshlets that ran (`how`: the path it took). No groups: nothing ran.
pub(super) fn ran(how: &'static str, pso: usize, groups: u64) {
    if groups == 0 {
        note(Outcome::ZeroGroup, pso, 0, "count-zero");
    } else {
        note(Outcome::Drawn, pso, meshlet_count(groups), how);
    }
}

/// The outcome of a call XCL's binding refused: its pipeline is a stand-in (a build still going,
/// or gone), or a view XCL's ABI or a two-pass call's record memory could not produce.
pub(super) fn bind_failure(pso: usize, why: &str) -> Outcome {
    if super::real::is_standin_pso(pso) || why.starts_with("its pipeline is not built") {
        return Outcome::FailedPipeline;
    }
    const VIEWS: [&str; 9] = [
        "no unified memory",
        "unified memory is full",
        "memory windows have no views",
        "no XCL ABI root signature",
        "root signature",
        "pipeline",
        "a shader memory window is not open",
        "shader memory is not open",
        "its arguments could not be sanitised",
    ];
    if VIEWS.iter().any(|prefix| why.starts_with(prefix)) {
        Outcome::ViewSlot
    } else {
        Outcome::Dropped
    }
}

/// One range was seen.
pub(super) fn note(outcome: Outcome, pso: usize, meshlets: u32, why: &'static str) {
    note_n(outcome, pso, meshlets, why, 1);
}

/// `n` ranges of one (pipeline, meshlets, outcome, reason) were seen.
pub(super) fn note_n(outcome: Outcome, pso: usize, meshlets: u32, why: &'static str, n: u64) {
    if n == 0 || !on() {
        return;
    }
    // (before the census lock: nothing else is locked inside it)
    note_key(outcome, super::pso_async::key_only(pso), meshlets, why, n);
}

/// `n` ranges of one (pipeline key, meshlets, outcome, reason) were seen: `note_n` with the key
/// already taken, for a caller counting many ranges of one pipeline.
fn note_key(outcome: Outcome, key: u64, meshlets: u32, why: &'static str, n: u64) {
    let frame = super::present::presents();
    let mut emit = None;
    {
        let mut g = DETAIL.lock().unwrap_or_else(|e| e.into_inner());
        let d = g.get_or_insert_with(|| Detail { last: frame, ..Detail::default() });
        count(d, outcome, key, meshlets, why, n);
        if frame >= d.last.saturating_add(EVERY) {
            let since = frame.saturating_sub(d.last);
            emit = Some(snapshot(d, frame, since));
        }
    }
    if let Some(s) = emit {
        crate::xlog_info!("{}", line(&s));
    }
}

/// An indirect mesh call that ran: the ranges its count enables, and the ones the title's count
/// leaves out. `args` and `count` are the call's console addresses, `sig` its command signature.
pub(super) fn indirect(pso: usize, args: Option<u64>, count: Option<u64>, sig: usize, max: u32, how: &'static str) {
    if max == 0 || !on() {
        return;
    }
    let key = super::pso_async::key_only(pso);
    let active = count
        .and_then(|at| crate::memmap::title_words(at as usize, 1))
        .map_or(max, |w| w.first().copied().unwrap_or(max).min(max));
    let stride = u64::from(super::indirect::stride_of(sig).unwrap_or(12).max(12));
    let mut read = 0u32;
    while read < active.min(MAX_COMMANDS) {
        let words = args
            .and_then(|a| a.checked_add(u64::from(read) * stride))
            .and_then(|at| crate::memmap::title_words(at as usize, 3));
        let Some(words) = words else { break };
        let (Some(&x), Some(&y), Some(&z)) = (words.first(), words.get(1), words.get(2)) else { break };
        let groups = (x as u64).saturating_mul(y as u64).saturating_mul(z as u64);
        let (outcome, why) = if groups == 0 { (Outcome::ZeroGroup, "count-zero") } else { (Outcome::Drawn, how) };
        note_key(outcome, key, meshlet_count(groups), why, 1);
        read += 1;
    }
    if read < active {
        let why = if read == 0 { "arguments-unread" } else { "records-past-read" };
        note_key(Outcome::Drawn, key, MESHLETS_ANY, why, u64::from(active - read));
    }
    if active < max {
        note_key(Outcome::NotSubmitted, key, MESHLETS_ANY, "count-disabled", u64::from(max - active));
    }
}

#[derive(Default)]
struct Detail {
    /// ranges by outcome, the interval's
    totals: [u64; OUTCOMES],
    /// (pipeline key, meshlets) -> the interval's counts by outcome
    pairs: HashMap<(u64, u32), [u64; OUTCOMES]>,
    /// ranges whose pair came past `MAX_PAIRS`
    overflow: [u64; OUTCOMES],
    /// why, by the short name the site gave: how a call ran or what dropped it
    reasons: Vec<(&'static str, [u64; OUTCOMES])>,
    /// the present count of the last census line
    last: u64,
}

static DETAIL: Mutex<Option<Detail>> = Mutex::new(None);

struct Pair {
    key: u64,
    meshlets: u32,
    counts: [u64; OUTCOMES],
}

struct Snapshot {
    frame: u64,
    since: u64,
    totals: [u64; OUTCOMES],
    pairs: Vec<Pair>,
    overflow: [u64; OUTCOMES],
    reasons: Vec<(&'static str, [u64; OUTCOMES])>,
}

/// One range into the interval's counts.
fn count(d: &mut Detail, outcome: Outcome, key: u64, meshlets: u32, why: &'static str, n: u64) {
    let slot = outcome as usize;
    d.totals[slot] += n;
    let pair = (key, meshlets);
    if let Some(counts) = d.pairs.get_mut(&pair) {
        counts[slot] += n;
    } else if d.pairs.len() < MAX_PAIRS {
        let mut counts = [0; OUTCOMES];
        counts[slot] = n;
        d.pairs.insert(pair, counts);
    } else {
        d.overflow[slot] += n;
    }
    if let Some(at) = d.reasons.iter().position(|(name, _)| *name == why) {
        d.reasons[at].1[slot] += n;
    } else {
        let mut counts = [0; OUTCOMES];
        counts[slot] = n;
        d.reasons.push((why, counts));
    }
}

/// The interval's counts out of `d`, and `d` emptied for the next interval.
fn snapshot(d: &mut Detail, frame: u64, since: u64) -> Snapshot {
    let taken = std::mem::take(d);
    d.last = frame;
    Snapshot {
        frame,
        since,
        totals: taken.totals,
        pairs: taken.pairs.into_iter().map(|((key, meshlets), counts)| Pair { key, meshlets, counts }).collect(),
        overflow: taken.overflow,
        reasons: taken.reasons,
    }
}

fn meshlets_text(meshlets: u32) -> String {
    if meshlets == MESHLETS_ANY {
        "*".to_string()
    } else {
        meshlets.to_string()
    }
}

/// The census line: the totals, the pairs that name what is missing, and why.
fn line(s: &Snapshot) -> String {
    let total: u64 = s.totals.iter().sum();
    let mut out = format!("mesh attrib: frame {} ({} presents): {total} mesh ranges:", s.frame, s.since);
    for (i, name) in OUTCOME_NAMES.iter().enumerate() {
        out.push_str(if i == 0 { " " } else { ", " });
        let _ = write!(out, "{name} {}", s.totals[i]);
    }
    out.push_str("; key/meshlets");
    let mut shown_ranges = 0u64;
    for &outcome in &PREFERENCE {
        let mut pairs: Vec<&Pair> = s.pairs.iter().filter(|p| p.counts[outcome as usize] != 0).collect();
        pairs.sort_unstable_by_key(|p| std::cmp::Reverse(p.counts[outcome as usize]));
        for p in pairs.into_iter().take(PAIRS_PER_OUTCOME) {
            if out.len() + 96 >= MAX_LINE {
                break;
            }
            let _ = write!(out, "; {} {:016x}:{} {}", OUTCOME_NAMES[outcome as usize], p.key, meshlets_text(p.meshlets), p.counts[outcome as usize]);
            shown_ranges += p.counts[outcome as usize];
        }
    }
    if shown_ranges < total {
        let over: u64 = s.overflow.iter().sum();
        let _ = write!(out, "; +{} ranges not itemised", total - shown_ranges);
        if over != 0 {
            let _ = write!(out, " ({over} past the {MAX_PAIRS}-pair cap)");
        }
    }
    if !s.reasons.is_empty() {
        let mut reasons: Vec<(&'static str, [u64; OUTCOMES])> = s.reasons.clone();
        reasons.sort_unstable_by_key(|(_, counts)| std::cmp::Reverse(counts.iter().sum::<u64>()));
        out.push_str("; why");
        for (name, counts) in reasons.into_iter().take(6) {
            let _ = write!(out, " {name} {}", counts.iter().sum::<u64>());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_range_lands_in_one_outcome_and_one_pair() {
        let mut d = Detail::default();
        count(&mut d, Outcome::Drawn, 0xaa, 64, "two-pass", 3);
        count(&mut d, Outcome::ZeroGroup, 0xaa, 0, "count-zero", 2);
        count(&mut d, Outcome::NotSubmitted, 0xaa, MESHLETS_ANY, "count-disabled", 5);
        assert_eq!(d.totals[Outcome::Drawn as usize], 3);
        assert_eq!(d.totals[Outcome::ZeroGroup as usize], 2);
        assert_eq!(d.totals[Outcome::NotSubmitted as usize], 5);
        assert_eq!(d.totals.iter().sum::<u64>(), 10);
        assert_eq!(d.pairs[&(0xaa, 64)], [3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(d.pairs[&(0xaa, 0)][Outcome::ZeroGroup as usize], 2);
        assert_eq!(d.pairs[&(0xaa, MESHLETS_ANY)][Outcome::NotSubmitted as usize], 5);
        assert_eq!(d.reasons.iter().find(|(name, _)| *name == "count-disabled").expect("the reason").1[Outcome::NotSubmitted as usize], 5);
    }

    #[test]
    fn a_zero_group_range_draws_nothing_and_a_count_keeps_the_rest_out() {
        assert_eq!(meshlet_count(0), 0);
        assert_eq!(meshlet_count(1024), 1024);
        assert_eq!(meshlet_count(u64::from(u32::MAX)), MESHLETS_ANY - 1, "clamped below the unknown mark");
    }

    #[test]
    fn a_refused_binding_names_the_views_or_the_failed_pipeline() {
        assert_eq!(bind_failure(0, "no unified memory"), Outcome::ViewSlot);
        assert_eq!(bind_failure(0, "a shader memory window is not open on the GPU"), Outcome::ViewSlot);
        assert_eq!(bind_failure(0, "no XCL ABI root signature"), Outcome::ViewSlot);
        assert_eq!(bind_failure(0, "7 groups in two passes are over what one prepass holds"), Outcome::Dropped);
        assert_eq!(bind_failure(0, "its pipeline is not built (taken away for the code budget, or its build failed)"), Outcome::FailedPipeline);
        assert_ne!(bind_failure(0, "no unified memory"), Outcome::FailedPipeline, "no stand-in here");
    }

    #[test]
    fn ranges_past_the_pair_cap_are_counted_in_the_totals_not_lost() {
        let mut d = Detail::default();
        for k in 0..(MAX_PAIRS as u64 + 5) {
            count(&mut d, Outcome::Drawn, k + 1, 1, "native", 1);
        }
        assert_eq!(d.totals[Outcome::Drawn as usize], MAX_PAIRS as u64 + 5);
        assert_eq!(d.overflow[Outcome::Drawn as usize], 5);
        assert_eq!(d.pairs.len(), MAX_PAIRS);
    }

    #[test]
    fn the_line_names_the_outcomes_the_missing_pairs_and_the_reasons_within_its_bound() {
        let mut d = Detail::default();
        for k in 0..MAX_PAIRS as u64 {
            for outcome in [Outcome::Drawn, Outcome::ZeroGroup, Outcome::NotSubmitted] {
                count(&mut d, outcome, 0x0123_4567_89ab_cdef ^ k, (k % 7) as u32, "two-pass", 1);
            }
        }
        let s = snapshot(&mut d, 900, 300);
        let line = line(&s);
        assert!(line.len() <= MAX_LINE, "{} bytes", line.len());
        assert!(line.starts_with("mesh attrib: frame 900 (300 presents):"));
        assert!(line.contains("never-submitted"));
        assert!(line.contains("zero-group"));
        assert!(line.contains("; why two-pass "));
        assert!(line.contains("not itemised"), "the tail names what the bound left out: {line}");
        assert!(!line.contains('\n'));
    }

    #[test]
    fn the_snapshot_empties_the_interval_and_keeps_the_frame() {
        let mut d = Detail::default();
        count(&mut d, Outcome::Drawn, 1, 1, "two-pass", 2);
        let s = snapshot(&mut d, 42, 300);
        assert_eq!(s.totals[Outcome::Drawn as usize], 2);
        assert_eq!(d.totals[Outcome::Drawn as usize], 0, "the next interval starts empty");
        assert_eq!(d.last, 42);
        assert!(d.pairs.is_empty() && d.reasons.is_empty());
    }
}
