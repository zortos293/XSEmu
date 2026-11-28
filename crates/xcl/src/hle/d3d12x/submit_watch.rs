//! The submit watch (on by default; `XCL_SUBMIT_WATCH=0` off): what XCL itself
//! knows of the GPU's work when DRED has nothing. Each ExecuteCommandLists on a queue is followed
//! by a Signal of the watch's own fence for that queue (one more GPU command a submission), and
//! the watch keeps the last 32 submissions of each queue with their lists' ops: the count of each
//! kind, and every dispatch with its pipeline and thread-group counts. A thread reads the fences
//! every 100 ms, before the device can be removed (a removed device's fences read as complete):
//! - a submission that has not finished `XCL_SUBMIT_WATCH_SLOW_MS` (1000) after it went in is
//!   logged once, with its dispatches;
//! - a finished one that took longer than any before (from when the one before it was seen to
//!   finish, to the 100 ms of the polling) is logged from 250 ms on, as the longest so far;
//! - at the device's removal (`log_removal`), each queue's submissions not seen finished, the
//!   first of them named: the work the GPU was in when it stopped.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Whether the watch is on for `XCL_SUBMIT_WATCH`'s value: on unless it is `0`.
fn wanted(value: Option<&str>) -> bool {
    !value.is_some_and(|v| v.trim() == "0")
}

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = wanted(std::env::var("XCL_SUBMIT_WATCH").ok().as_deref());
        if on {
            crate::xlog_info!("submit watch: on (XCL's own record of the work each queue has in flight, read from its fences every 100 ms; XCL_SUBMIT_WATCH=0 turns it off)");
        } else {
            crate::xlog_info!("submit watch: off (XCL_SUBMIT_WATCH=0): a device removal will not name the work that was in flight");
        }
        on
    })
}

fn slow() -> Duration {
    static MS: OnceLock<u64> = OnceLock::new();
    Duration::from_millis(*MS.get_or_init(|| std::env::var("XCL_SUBMIT_WATCH_SLOW_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1000)))
}

/// A list's ops since its reset: how many of each kind, and its dispatches (the title's pipeline,
/// thread groups; None: an indirect one), the first 256.
#[derive(Clone, Default, Debug, PartialEq)]
struct Ops {
    kinds: BTreeMap<&'static str, u32>,
    dispatches: Vec<(usize, Option<[u32; 3]>)>,
}

struct Submission {
    n: u64,
    at: Instant,
    lists: Vec<(usize, Ops)>,
}

struct Queue {
    id: usize,
    fence: ID3D12Fence,
    next: u64,
    ring: VecDeque<Submission>,
    /// the fence's value at the last poll, and when it was first seen there
    completed: u64,
    completed_at: Instant,
    polled: Option<Instant>,
    reported_slow: u64,
}

#[derive(Default)]
struct Watch {
    lists: HashMap<usize, Ops>,
    queues: HashMap<usize, Queue>,
    longest: Duration,
    longest_logged: Option<Instant>,
}

// the fences are only used under the lock
unsafe impl Send for Watch {}

static WATCH: Mutex<Option<Watch>> = Mutex::new(None);

/// An op (`kind`) with the title's pipeline `title_pso` was recorded on `list`.
pub(super) fn op(list: usize, kind: &'static str, title_pso: usize) {
    if !on() {
        return;
    }
    let mut g = WATCH.lock().unwrap();
    let ops = g.get_or_insert_with(Watch::default).lists.entry(list).or_default();
    *ops.kinds.entry(kind).or_default() += 1;
    if kind.contains("dispatch") && !kind.contains("mesh") && ops.dispatches.len() < 256 {
        ops.dispatches.push((title_pso, None));
    }
}

/// The dispatch just recorded on `list` ran `groups` thread groups.
pub(super) fn groups(list: usize, groups: [u32; 3]) {
    if !on() {
        return;
    }
    if let Some(d) = WATCH.lock().unwrap().as_mut().and_then(|w| w.lists.get_mut(&list)).and_then(|o| o.dispatches.last_mut()) {
        d.1 = Some(groups);
    }
}

pub(super) fn reset(list: usize) {
    if on() {
        if let Some(w) = WATCH.lock().unwrap().as_mut() {
            w.lists.remove(&list);
        }
    }
}

/// `lists` went to `queue` in one ExecuteCommandLists, just now.
pub(super) unsafe fn submitted(queue: *mut c_void, lists: &[*mut c_void]) {
    if !on() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let mut g = WATCH.lock().unwrap();
    let w = g.get_or_insert_with(Watch::default);
    if !w.queues.contains_key(&(queue as usize)) {
        let raw = super::real::device_raw();
        let Some(dev) = ID3D12Device::from_raw_borrowed(&raw) else { return };
        let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
        let now = Instant::now();
        w.queues.insert(queue as usize, Queue { id: super::list_x::queue_id(queue as usize), fence, next: 0, ring: VecDeque::new(), completed: 0, completed_at: now, polled: None, reported_slow: 0 });
        start_polling();
    }
    let ops: Vec<(usize, Ops)> = lists.iter().map(|&l| (l as usize, w.lists.get(&(l as usize)).cloned().unwrap_or_default())).collect();
    let entry = w.queues.get_mut(&(queue as usize)).unwrap();
    entry.next += 1;
    if q.Signal(&entry.fence, entry.next).is_err() {
        return;
    }
    if entry.ring.len() == 32 {
        entry.ring.pop_front();
    }
    entry.ring.push_back(Submission { n: entry.next, at: Instant::now(), lists: ops });
}

fn start_polling() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-submit-watch".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_millis(100));
            poll(Instant::now());
        });
    });
}

fn poll(now: Instant) {
    let mut lines = Vec::new();
    {
        let mut g = WATCH.lock().unwrap();
        let Some(w) = g.as_mut() else { return };
        let mut longest = w.longest;
        let mut new_longest = None;
        for q in w.queues.values_mut() {
            let c = unsafe { q.fence.GetCompletedValue() };
            // a removed device's fences read as complete: keep what the last poll saw
            if c == u64::MAX {
                continue;
            }
            q.polled = Some(now);
            if c > q.completed {
                let mut began = q.completed_at;
                for s in q.ring.iter().filter(|s| s.n > q.completed && s.n <= c) {
                    let took = now.saturating_duration_since(began.max(s.at));
                    if took > longest {
                        longest = took;
                        new_longest = Some((q.id, s.n, took, describe(&s.lists)));
                    }
                    began = now;
                }
                q.completed = c;
                q.completed_at = now;
            }
            if let Some(s) = q.ring.iter().find(|s| s.n > c && s.n > q.reported_slow) {
                let age = now.saturating_duration_since(s.at);
                if age >= slow() {
                    q.reported_slow = s.n;
                    lines.push((true, format!(
                        "submit watch: q{} submission #{} has not finished {:.1} s after it went in (the one before finished {:.1} s ago): {}",
                        q.id, s.n, age.as_secs_f64(), now.saturating_duration_since(q.completed_at).as_secs_f64(), describe(&s.lists)
                    )));
                }
            }
        }
        if let Some((id, n, took, what)) = new_longest {
            w.longest = longest;
            if took >= Duration::from_millis(250) && (took >= Duration::from_millis(1500) || w.longest_logged.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(10))) {
                w.longest_logged = Some(now);
                lines.push((took >= Duration::from_millis(1500), format!("submit watch: the longest submission so far: q{id} #{n}, about {:.1} s on the GPU (to 0.1 s): {what}", took.as_secs_f64())));
            }
        }
    }
    for (warn, line) in lines {
        if warn {
            crate::xlog_warn!("{line}");
        } else {
            crate::xlog_info!("{line}");
        }
    }
}

/// A submission's lists as text: each list's op counts and its dispatches, the largest first.
fn describe(lists: &[(usize, Ops)]) -> String {
    describe_with(lists, super::markers::pipeline)
}

fn describe_with(lists: &[(usize, Ops)], name: impl Fn(usize) -> String) -> String {
    let mut s = String::new();
    for (k, (list, ops)) in lists.iter().enumerate() {
        if k > 0 {
            s.push_str("; ");
        }
        let kinds: Vec<String> = ops.kinds.iter().map(|(kind, n)| format!("{n} {kind}")).collect();
        let _ = write!(s, "list {list:#x}: {}", if kinds.is_empty() { "no ops XCL recorded".into() } else { kinds.join(", ") });
        let mut d = ops.dispatches.clone();
        d.sort_by_key(|(_, g)| std::cmp::Reverse(g.map_or(u64::MAX, |g| g.iter().map(|&v| v as u64).product())));
        let mut merged: Vec<(usize, Option<[u32; 3]>, u32)> = Vec::new();
        for (pso, g) in d {
            match merged.iter_mut().find(|m| m.0 == pso && m.1 == g) {
                Some(m) => m.2 += 1,
                None => merged.push((pso, g, 1)),
            }
        }
        for (pso, g, n) in merged.iter().take(6) {
            let groups = g.map_or_else(|| "indirect".to_string(), |g| format!("{}x{}x{} groups", g[0], g[1], g[2]));
            let _ = write!(s, "; dispatch {groups}{} of {}", if *n > 1 { format!(" ×{n}") } else { String::new() }, name(*pso));
        }
        if merged.len() > 6 {
            let _ = write!(s, "; {} more dispatch kinds", merged.len() - 6);
        }
    }
    s
}

/// At the device's removal: per queue, the submissions the last poll had not seen finish.
pub(super) fn log_removal() {
    if !on() {
        return;
    }
    let now = Instant::now();
    let g = WATCH.lock().unwrap();
    let Some(w) = g.as_ref() else {
        crate::xlog_error!("submit watch: nothing submitted");
        return;
    };
    let mut queues: Vec<&Queue> = w.queues.values().collect();
    queues.sort_by_key(|q| q.id);
    for q in queues {
        let polled = q.polled.map_or_else(|| "never polled".into(), |t| format!("polled {} ms before", now.saturating_duration_since(t).as_millis()));
        let open: Vec<&Submission> = q.ring.iter().filter(|s| s.n > q.completed).collect();
        if open.is_empty() {
            crate::xlog_error!("submit watch at removal: q{}: all {} submissions had finished ({polled})", q.id, q.next);
            continue;
        }
        for (k, s) in open.iter().take(4).enumerate() {
            crate::xlog_error!(
                "submit watch at removal: q{}: {}#{} went in {} ms before and was not seen finished ({polled}; #{} finished {} ms before): {}",
                q.id, if k == 0 { "FIRST NOT FINISHED " } else { "" }, s.n, now.saturating_duration_since(s.at).as_millis(), q.completed,
                now.saturating_duration_since(q.completed_at).as_millis(), describe(&s.lists)
            );
        }
        if open.len() > 4 {
            crate::xlog_error!("submit watch at removal: q{}: {} more submissions after those", q.id, open.len() - 4);
        }
    }
}

/// At the device's removal for `reason`: hand the dispatches of the submissions no fence had seen
/// finished to the hung-pipeline guard, which records the compute pipelines among them for the next
/// start of the same translator to skip. Taken under the watch's lock, named outside it.
pub(super) fn record_hung(reason: u32) {
    if !on() {
        return;
    }
    let dispatches: Vec<(usize, Option<[u32; 3]>)> = {
        let g = WATCH.lock().unwrap();
        let Some(w) = g.as_ref() else { return };
        w.queues
            .values()
            .flat_map(|q| q.ring.iter().filter(|s| s.n > q.completed))
            .flat_map(|s| s.lists.iter().flat_map(|(_, ops)| ops.dispatches.iter().copied()))
            .collect()
    };
    super::hung_skip::record_at_removal(reason, &dispatches);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A submission is described by its lists' op counts and its dispatches, the most thread
    /// groups first, the same pipeline and groups counted once.
    #[test]
    fn a_submission_names_its_dispatches_largest_first() {
        let mut ops = Ops::default();
        ops.kinds.insert("dispatch", 4);
        ops.kinds.insert("draw", 2);
        ops.dispatches = vec![(0xA, Some([1, 1, 1])), (0xB, Some([80, 45, 1])), (0xA, Some([1, 1, 1])), (0xC, None)];
        let text = describe_with(&[(0x10, ops), (0x20, Ops::default())], |p| format!("P{p:x}"));
        assert_eq!(text, "list 0x10: 4 dispatch, 2 draw; dispatch indirect of Pc; dispatch 80x45x1 groups of Pb; dispatch 1x1x1 groups ×2 of Pa; list 0x20: no ops XCL recorded");
    }

    /// The watch is on unless it is switched off: a removal has to name the work in flight without
    /// anyone having set anything, and `XCL_SUBMIT_WATCH=0` is the way out of its cost.
    #[test]
    fn the_watch_is_on_unless_it_is_switched_off() {
        assert!(wanted(None));
        assert!(wanted(Some("1")));
        assert!(wanted(Some("")));
        assert!(!wanted(Some("0")));
        assert!(!wanted(Some(" 0 ")));
    }
}
