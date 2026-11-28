//! Why the pipelines counted failed (`pso_async`'s failed count) made no pipeline: one line per
//! failure, the first 64 and then every 64th, and a summary by reason every 30 s.
//!
//! The count and the console's "pipelines N of M built, P pending, F failed" line do not change.
//! This only names the reason behind each failure, so a run's failed count can be read.

use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};

/// How often the reasons are summed up, in game time (`log::uptime`).
const SUMMARY_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// The stage a failed pipeline was made for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stage {
    Compute,
    Graphics,
    Mesh,
}

impl Stage {
    fn text(self) -> &'static str {
        match self {
            Stage::Compute => "compute",
            Stage::Graphics => "graphics",
            Stage::Mesh => "mesh",
        }
    }
}

/// The short reasons the summary groups by.
pub(super) const TRANSLATOR_REFUSAL: &str = "translator refusal";
pub(super) const DRIVER_REJECTION: &str = "driver rejection";
pub(super) const MEMORY_REFUSAL: &str = "memory refusal";
pub(super) const TOO_BIG: &str = "too big";
pub(super) const NOT_ATTEMPTED: &str = "not attempted";
pub(super) const REFUSED: &str = "refused";

/// One pipeline's failure: the reason the summary groups by, and what the log line adds to it
/// (the HRESULT, the translator's message, the reason it was left out).
#[derive(Clone, Debug)]
pub(super) struct Failure {
    pub(super) reason: Cow<'static, str>,
    pub(super) detail: String,
}

impl Failure {
    pub(super) fn new(reason: impl Into<Cow<'static, str>>, detail: impl Into<String>) -> Self {
        Failure { reason: reason.into(), detail: detail.into() }
    }

    /// A build that gave no reason.
    pub(super) fn unexplained() -> Self {
        Failure { reason: REFUSED.into(), detail: "no reason noted".into() }
    }
}

thread_local! {
    /// why this thread's build failed: noted where the build gave up, taken at the counting site
    static LAST: RefCell<Option<Failure>> = const { RefCell::new(None) };
}

/// This thread's build failed for `reason`; `detail` is the HRESULT or error text, empty if none.
pub(super) fn note(reason: impl Into<Cow<'static, str>>, detail: impl Into<String>) {
    LAST.with(|f| *f.borrow_mut() = Some(Failure::new(reason, detail)));
}

/// `note` only when this build has no reason yet: the site closest to the failure wins over the
/// fallback at the counting site.
pub(super) fn note_if_absent(reason: impl Into<Cow<'static, str>>, detail: impl Into<String>) {
    let absent = LAST.with(|f| f.borrow().is_none());
    if absent {
        note(reason, detail);
    }
}

/// The build made its pipeline: drop the reason of a refusal it went past.
pub(super) fn clear() {
    LAST.with(|f| *f.borrow_mut() = None);
}

/// Take this thread's build reason, if any.
pub(super) fn take() -> Option<Failure> {
    LAST.with(|f| f.borrow_mut().take())
}

/// The line for the `n`th failure, when the rate limit shows it: the first 64, then every 64th.
/// None when this one is held back.
pub(super) fn failure_line(n: u64, stage: Stage, key: u64, shaders: &str, f: &Failure) -> Option<String> {
    if n > 64 && n % 64 != 0 {
        return None;
    }
    let shaders = if shaders.is_empty() { "shaders unknown" } else { shaders };
    let tail = if f.detail.is_empty() { f.reason.to_string() } else { format!("{}: {}", f.reason, f.detail) };
    Some(format!("D3D12.X: failed pipeline #{n} ({}, key {key:016x}, {shaders}): {tail}", stage.text()))
}

/// One line, the reasons by count with the biggest first. `reasons` may be in any order.
pub(super) fn summary_line(now_s: f64, total: u64, reasons: &[(Cow<'static, str>, u64)]) -> String {
    let mut sorted = reasons.to_vec();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let groups: Vec<String> = sorted.iter().map(|(reason, n)| format!("{reason} {n}")).collect();
    let groups = if groups.is_empty() { "none".to_string() } else { groups.join(", ") };
    format!("D3D12.X: failed pipelines at {now_s:.0} s: {total} in all: {groups}")
}

/// the failures counted so far, by reason (at most one entry per distinct reason)
static REASONS: Mutex<Vec<(Cow<'static, str>, u64)>> = Mutex::new(Vec::new());
/// the count at the last failure seen (`pso_async`'s failed count)
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// A pipeline was counted failed: log its line when the rate limit allows, and add it to the
/// reason summary.
pub(super) fn counted(n: u64, stage: Stage, key: u64, shaders: &str, f: &Failure) {
    if let Some(line) = failure_line(n, stage, key, shaders, f) {
        crate::xlog_info!("{line}");
    }
    {
        let mut g = REASONS.lock().unwrap_or_else(|e| e.into_inner());
        match g.iter_mut().find(|(reason, _)| reason.as_ref() == f.reason.as_ref()) {
            Some((_, count)) => *count += 1,
            None => g.push((f.reason.clone(), 1)),
        }
    }
    TOTAL.fetch_max(n, Ordering::Relaxed);
    start_summary();
}

/// The summary thread: every 30 s, when the count changed since the last summary, one line with
/// the reasons and their counts.
fn start_summary() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-fail".into()).spawn(|| {
            let mut shown = 0;
            loop {
                std::thread::sleep(SUMMARY_EVERY);
                let total = TOTAL.load(Ordering::Relaxed);
                if total == shown {
                    continue;
                }
                shown = total;
                let reasons = REASONS.lock().unwrap_or_else(|e| e.into_inner()).clone();
                crate::xlog_info!("{}", summary_line(crate::log::uptime(), total, &reasons));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 64 failures of a run are logged whole, then one in 64: a 1 085-failure run
    /// writes 79 failure lines, not 1 085.
    #[test]
    fn the_first_64_failures_are_logged_and_then_every_64th() {
        let why = Failure::new(DRIVER_REJECTION, "HRESULT 0x80070057");
        let shown: Vec<u64> = (1..=1000).filter(|&n| failure_line(n, Stage::Compute, 0x0123456789abcdef, "CS fedcba9876543210", &why).is_some()).collect();
        assert_eq!(&shown[..64], &(1..=64).collect::<Vec<u64>>()[..], "the first 64 are all logged");
        assert_eq!(&shown[64..], &[128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960][..]);
        assert_eq!(shown.len(), 64 + 14, "#64 is one of the first 64");
        assert!(failure_line(65, Stage::Compute, 1, "CS 1", &why).is_none());
    }

    /// The summary names the reasons with their counts, biggest first, on one line.
    #[test]
    fn the_summary_groups_the_reasons_biggest_first() {
        let line = summary_line(60.4, 1085, &[(DRIVER_REJECTION.into(), 5), ("not attempted: ray tracing off: a stage of the acceleration-structure builder left out unbuilt".into(), 1080)]);
        assert_eq!(
            line,
            "D3D12.X: failed pipelines at 60 s: 1085 in all: not attempted: ray tracing off: a stage of the acceleration-structure builder left out unbuilt 1080, driver rejection 5"
        );
        assert!(!line.contains('\n'));
        assert!(summary_line(0.0, 2, &[(TOO_BIG.into(), 1), (MEMORY_REFUSAL.into(), 1)]).ends_with("memory refusal 1, too big 1"), "equal counts keep a stable order");
        assert!(summary_line(0.0, 0, &[]).ends_with("0 in all: none"));
    }

    /// The counting site groups equal reasons, and the summary prints the biggest group first.
    #[test]
    fn counted_failures_are_grouped_by_reason() {
        for n in 1..=3 {
            counted(n, Stage::Compute, 0x1111, "CS 2222", &Failure::new(DRIVER_REJECTION, "HRESULT 0x80070057"));
        }
        counted(4, Stage::Compute, 0x1111, "CS 2222", &Failure::new(TOO_BIG, ""));
        let reasons = REASONS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let line = summary_line(30.0, TOTAL.load(Ordering::Relaxed), &reasons);
        assert!(line.contains("driver rejection 3"), "{line}");
        assert!(line.contains("too big 1"), "{line}");
        assert!(line.find("driver rejection").unwrap() < line.find("too big").unwrap(), "biggest first: {line}");
    }

    /// A reason noted once is taken once, and the line names the stage, the key, the shaders and
    /// the reason. The first reason noted for a build wins.
    #[test]
    fn a_noted_failure_is_taken_once_and_the_line_names_what_it_was() {
        clear();
        assert!(take().is_none(), "nothing noted");
        note(TRANSLATOR_REFUSAL, "CS fedcba9876543210: buffer_load_format_xyzw");
        let why = take().expect("the reason just noted");
        assert_eq!((why.reason.as_ref(), why.detail.as_str()), (TRANSLATOR_REFUSAL, "CS fedcba9876543210: buffer_load_format_xyzw"));
        assert!(take().is_none(), "taken once");
        clear();
        note_if_absent(MEMORY_REFUSAL, "first");
        note_if_absent(TOO_BIG, "second");
        assert_eq!(take().expect("a reason").reason.as_ref(), MEMORY_REFUSAL);
        let line = failure_line(64, Stage::Mesh, 0xfedcba9876543210, "MS 1111222233334444 PS 5555666677778888", &Failure::new(TRANSLATOR_REFUSAL, "")).expect("#64 is logged");
        assert_eq!(line, "D3D12.X: failed pipeline #64 (mesh, key fedcba9876543210, MS 1111222233334444 PS 5555666677778888): translator refusal");
    }
}
