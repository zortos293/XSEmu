//! `xcl warm-up --progress-json` (docs/launcher-protocol.md): the warm-up's JSON lines for the
//! launcher, the launcher's commands on stdin, and the command line that turns them on. Everything
//! here is plain data and text; `main` writes the lines and `real::warm_up` does the work.

use serde::Serialize;
use std::time::{Duration, Instant};

/// One line the warm-up writes on stdout.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "ev", rename_all = "snake_case")]
pub enum Event {
    /// once the pipelines the pass goes through are known
    WarmupStart { total: usize, full: bool, threads: usize },
    /// about once a second and once at the end; `done` and `total` count the current `phase`
    WarmupProgress {
        done: usize,
        total: usize,
        keys_per_s: f64,
        /// left out until the phase has a rate
        #[serde(skip_serializing_if = "Option::is_none")]
        eta_s: Option<u64>,
        ram_free_mb: u64,
        giants_compiling: usize,
        in_flight: usize,
        failed: usize,
        phase: &'static str,
        paused_for_ram: bool,
    },
    /// the last line
    WarmupDone { built: usize, failed: usize, too_big: usize, took_s: f64, complete: bool, cancelled: bool, code: i32 },
    /// refused or failed before starting
    Error { message: String },
}

/// `e` as one line of JSON, without the line end.
pub fn line(e: &Event) -> String {
    serde_json::to_string(e).expect("warm-up events always serialize")
}

/// The id of `real::warm_up`'s phase `name` in `warmup_progress`; a name it does not know goes as it is.
pub fn phase_id(name: &'static str) -> &'static str {
    match name {
        "pipelines" => "starting",
        "giants first" => "giants",
        "bound by an earlier run" => "bound",
        "compute pipelines" => "compute",
        "compute blobs" => "compute_blobs",
        "the rest" => "rest",
        other => other,
    }
}

/// Builds a phase went through per second since it started, as `warm-up --mesh-only` reckons
/// its ETA: averaged over the phase, which starts again at 0 of its own total.
pub struct Rate {
    phase: &'static str,
    since: Instant,
}

impl Rate {
    pub fn new(now: Instant) -> Rate {
        Rate { phase: "", since: now }
    }

    /// (builds per second, seconds left) for `done` of `total` of `phase` at `now`; no ETA while the rate is 0.
    pub fn at(&mut self, phase: &'static str, done: usize, total: usize, now: Instant) -> (f64, Option<u64>) {
        if phase != self.phase {
            *self = Rate { phase, since: now };
        }
        let secs = now.duration_since(self.since).max(Duration::from_millis(1)).as_secs_f64();
        let rate = done as f64 / secs;
        let eta = (rate > 0.0).then(|| (total.saturating_sub(done) as f64 / rate).round() as u64);
        ((rate * 10.0).round() / 10.0, eta)
    }
}

/// How the launcher asked the warm-up to stop (`wait_for_cancel`).
#[derive(Debug, PartialEq, Eq)]
pub enum Cancel {
    Command,
    /// stdin ended or failed: the launcher is gone, and a warm-up never runs on unattended
    Closed,
}

/// Reads the launcher's JSON lines until `{"cmd":"cancel"}` or the end of `input`; other
/// commands, other fields and lines that are not JSON are ignored.
pub fn wait_for_cancel(input: impl std::io::BufRead) -> Cancel {
    for line in input.lines() {
        let Ok(line) = line else { break };
        let cmd = serde_json::from_str::<serde_json::Value>(&line).ok().and_then(|v| v.get("cmd").and_then(|c| c.as_str()).map(str::to_owned));
        if cmd.as_deref() == Some("cancel") {
            return Cancel::Command;
        }
    }
    Cancel::Closed
}

/// `xcl warm-up`'s options besides `--mesh-only` and `--giants-only`; others are ignored, as before.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Args {
    pub full: bool,
    pub progress_json: bool,
}

/// `args`: the words after `warm-up`.
pub fn warm_up_args(args: &[String]) -> Args {
    Args { full: args.iter().any(|a| a == "--full"), progress_json: args.iter().any(|a| a == "--progress-json") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_events_have_the_protocols_shape() {
        assert_eq!(line(&Event::WarmupStart { total: 41906, full: false, threads: 16 }), r#"{"ev":"warmup_start","total":41906,"full":false,"threads":16}"#);
        let progress = Event::WarmupProgress {
            done: 27292,
            total: 41906,
            keys_per_s: 243.0,
            eta_s: Some(60),
            ram_free_mb: 12288,
            giants_compiling: 2,
            in_flight: 5,
            failed: 0,
            phase: "bound",
            paused_for_ram: false,
        };
        assert_eq!(
            line(&progress),
            r#"{"ev":"warmup_progress","done":27292,"total":41906,"keys_per_s":243.0,"eta_s":60,"ram_free_mb":12288,"giants_compiling":2,"in_flight":5,"failed":0,"phase":"bound","paused_for_ram":false}"#
        );
        let Event::WarmupProgress { done, total, keys_per_s, ram_free_mb, giants_compiling, in_flight, failed, phase, .. } = progress else { unreachable!() };
        let unknown = Event::WarmupProgress { done, total, keys_per_s, eta_s: None, ram_free_mb, giants_compiling, in_flight, failed, phase, paused_for_ram: true };
        assert!(!line(&unknown).contains("eta_s"), "no ETA is left out, not invented");
        assert!(line(&unknown).ends_with(r#""paused_for_ram":true}"#));
        assert_eq!(
            line(&Event::WarmupDone { built: 41890, failed: 16, too_big: 0, took_s: 812.4, complete: true, cancelled: false, code: 0 }),
            r#"{"ev":"warmup_done","built":41890,"failed":16,"too_big":0,"took_s":812.4,"complete":true,"cancelled":false,"code":0}"#
        );
        assert_eq!(
            line(&Event::Error { message: "another XSEmu process is running (xcl.exe 1234)".into() }),
            r#"{"ev":"error","message":"another XSEmu process is running (xcl.exe 1234)"}"#
        );
        assert_eq!(line(&Event::Error { message: "a \"quoted\"\nline".into() }), r#"{"ev":"error","message":"a \"quoted\"\nline"}"#, "one line, escaped");
    }

    #[test]
    fn every_warm_up_phase_has_a_short_id() {
        let ids: Vec<_> = ["pipelines", "giants first", "bound by an earlier run", "compute pipelines", "compute blobs", "the rest"].into_iter().map(phase_id).collect();
        assert_eq!(ids, ["starting", "giants", "bound", "compute", "compute_blobs", "rest"]);
        assert_eq!(phase_id("something new"), "something new");
    }

    #[test]
    fn the_rate_is_averaged_over_the_phase() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut r = Rate::new(t0);
        assert_eq!(r.at("bound", 0, 100, at(0)), (0.0, None), "no rate yet: no ETA");
        assert_eq!(r.at("bound", 20, 100, at(10)), (2.0, Some(40)));
        assert_eq!(r.at("bound", 100, 100, at(20)), (5.0, Some(0)));
        // a new phase starts its own clock
        assert_eq!(r.at("rest", 0, 30, at(21)), (0.0, None));
        assert_eq!(r.at("rest", 3, 30, at(24)), (1.0, Some(27)));
    }

    #[test]
    fn cancel_comes_from_the_command_or_the_end_of_stdin() {
        let input = "{\"cmd\":\"pause\"}\nnot json\n{\"cmd\":\"cancel\",\"extra\":1}\n{\"cmd\":\"later\"}\n";
        assert_eq!(wait_for_cancel(std::io::Cursor::new(input)), Cancel::Command);
        assert_eq!(wait_for_cancel(std::io::Cursor::new("{\"cmd\":\"focus\"}\n{\"ev\":\"cancel\"}\n")), Cancel::Closed);
        assert_eq!(wait_for_cancel(std::io::Cursor::new("")), Cancel::Closed);
    }

    #[test]
    fn the_options_are_read_anywhere_after_warm_up() {
        let args = |s: &[&str]| warm_up_args(&s.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(args(&[]), Args::default());
        assert_eq!(args(&["--full"]), Args { full: true, progress_json: false });
        assert_eq!(args(&["--progress-json"]), Args { full: false, progress_json: true });
        assert_eq!(args(&["--progress-json", "--full"]), Args { full: true, progress_json: true });
        assert_eq!(args(&["--unknown", "--full"]), Args { full: true, progress_json: false }, "unknown options stay ignored");
    }
}
