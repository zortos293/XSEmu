//! `xcl warm-up --blocking` with its compiles in child processes of the same exe
//! (`xcl warm-up --blocking-child`): a giant's DXC and driver memory goes back to the system when
//! its child exits, and a driver that crashes or hangs costs the keys it had in flight, not the
//! pass. This is the parent's side: the protocol, the lanes, the suspects and the bad list; the
//! children run `real`'s worker loop, and their governors see each other through `warm_peers`.
//! Rationale: docs/warm-up-children.md.
//!
//! Parts are adapted from CGBJ (VIctory, commit f4b48ad, apps/victory/src/warm.rs), each marked
//! where it is: the reply line per key, the hang kill, the bad list and the child's exit when
//! its parent is gone. Copyright (C) the VIctory/CGBJ contributors, GPL-3.0-or-later.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use super::warm_peers::Usage;

/// Console shader bytes from which a key is taken for a giant: `pso_sched::GIANT_DXIL` of DXIL at
/// `real::TRANSLATED_DXIL_PER_GUEST_BYTE` (43,691 B). It gets a child of its own.
pub(super) const GIANT_GUEST_BYTES: usize = super::pso_sched::GIANT_DXIL.div_ceil(super::real::TRANSLATED_DXIL_PER_GUEST_BYTE);
/// A key whose longest earlier build (`pso_manifest::slow_ms`, waits for memory left out) took
/// this long gets a child of its own as a giant does. A key merely slow (`pso_manifest::SLOW_MS`)
/// stays in the pool, where the target order already puts it first.
pub(super) const GIANT_BUILD_MS: u64 = 60_000;
/// Children that end in a row without a key done (before they say hello, or crashing with no key
/// in flight) after which the pass gives up.
const FRUITLESS: usize = 3;
/// Children of another translation or driver than the first after which the pass gives up.
const MISMATCHES: usize = 3;
/// How often the parent looks at its children with nothing heard.
const TICK: Duration = Duration::from_millis(250);
/// How long the next giant waits for memory, with no giant running anywhere, before the pool is
/// not fed until a giant starts; after twice this the pool children are told to end.
const GIANT_PATIENCE: Duration = Duration::from_secs(300);
/// The memory a single-key child is taken to hold outside its compiles until its hello (its
/// D3D12 device, the shader memory windows and the pipeline index): an estimate, not a
/// measurement. The parent reserves it for each giant child it started that has not said hello.
const CHILD_BASELINE: u64 = 1 << 30;
/// The prefix of a child's reply lines; any other line on its stdout goes to the parent's stderr.
const TAG: &str = "@xcl-warm";
/// The bad list, beside `blocking-progress.bin`.
pub(super) const BAD_FILE: &str = "blocking-bad.txt";

/// One key's result, as `real::KeyReport`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Done {
    pub key: u64,
    pub kind: String,
    pub ok: bool,
    pub refused: bool,
    pub hits: usize,
    pub cold: usize,
    pub rejected: usize,
    pub saved: usize,
    pub elapsed_us: u64,
    /// waiting for the translation's slot
    pub admitted_us: u64,
    /// the build less every wait for memory (`real::memory_waited`), for `pso_manifest::note_build_time`
    pub build_ms: u64,
    pub guest: usize,
    /// its pipeline library hits, misses, pipelines stored and shards refused (`pso_library`)
    pub library_hits: usize,
    pub library_misses: usize,
    pub library_stored: usize,
    pub library_refused: usize,
}

/// A child's line to the parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Msg {
    /// initialised: its translation era and driver (the progress record is the driver's)
    Hello { era: u64, driver: String },
    Start { key: u64 },
    /// key `key` waits for memory: its hang clock stops until `Go`
    Wait { key: u64 },
    Go { key: u64 },
    /// a pool child left key `key`'s giant driver compile of `dxil_len` bytes of DXIL
    /// (`--leave-giants`) for a single-key child
    Giant { key: u64, dxil_len: usize },
    Done(Done),
    /// not in the manifest when the child read it
    Missing { key: u64 },
    Mem { commit: u64, peak_working_set: u64, peak_commit: u64 },
    /// its keys are done; its `dxc::stage_summary`
    Bye { shaders: String },
}

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s, 16).ok()
}

impl Msg {
    pub(super) fn line(&self) -> String {
        match self {
            Msg::Hello { era, driver } => format!("{TAG} hello {era:016x} {driver}"),
            Msg::Start { key } => format!("{TAG} start {key:016x}"),
            Msg::Wait { key } => format!("{TAG} wait {key:016x}"),
            Msg::Go { key } => format!("{TAG} go {key:016x}"),
            Msg::Giant { key, dxil_len } => format!("{TAG} giant {key:016x} {dxil_len}"),
            // (CGBJ f4b48ad apps/victory/src/warm.rs `make`: "made <n> <ms> <there>", one line per pipeline)
            Msg::Done(d) => format!(
                "{TAG} done {:016x} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                d.key, d.kind, d.ok as u8, d.refused as u8, d.hits, d.cold, d.rejected, d.saved, d.elapsed_us, d.admitted_us, d.build_ms, d.guest,
                d.library_hits, d.library_misses, d.library_stored, d.library_refused
            ),
            Msg::Missing { key } => format!("{TAG} missing {key:016x}"),
            Msg::Mem { commit, peak_working_set, peak_commit } => format!("{TAG} mem {commit} {peak_working_set} {peak_commit}"),
            Msg::Bye { shaders } => format!("{TAG} bye {shaders}"),
        }
    }

    /// None: not a reply line (the child's own output) or a torn one.
    pub(super) fn parse(line: &str) -> Option<Msg> {
        let rest = line.trim_end_matches(['\r', '\n']).strip_prefix(TAG)?.strip_prefix(' ')?;
        let (word, rest) = rest.split_once(' ').unwrap_or((rest, ""));
        let fields: Vec<&str> = rest.split(' ').collect();
        let number = |i: usize| fields.get(i)?.parse::<u64>().ok();
        let flag = |i: usize| match fields.get(i) {
            Some(&"0") => Some(false),
            Some(&"1") => Some(true),
            _ => None,
        };
        match (word, fields.len()) {
            ("hello", _) => {
                let (era, driver) = rest.split_once(' ')?;
                if driver.trim().is_empty() {
                    return None;
                }
                Some(Msg::Hello { era: hex(era)?, driver: driver.to_string() })
            }
            ("start", 1) => Some(Msg::Start { key: hex(fields[0])? }),
            ("wait", 1) => Some(Msg::Wait { key: hex(fields[0])? }),
            ("go", 1) => Some(Msg::Go { key: hex(fields[0])? }),
            ("giant", 2) => Some(Msg::Giant { key: hex(fields[0])?, dxil_len: number(1)? as usize }),
            ("missing", 1) => Some(Msg::Missing { key: hex(fields[0])? }),
            ("done", 16) if !fields[1].is_empty() => Some(Msg::Done(Done {
                key: hex(fields[0])?,
                kind: fields[1].to_string(),
                ok: flag(2)?,
                refused: flag(3)?,
                hits: number(4)? as usize,
                cold: number(5)? as usize,
                rejected: number(6)? as usize,
                saved: number(7)? as usize,
                elapsed_us: number(8)?,
                admitted_us: number(9)?,
                build_ms: number(10)?,
                guest: number(11)? as usize,
                library_hits: number(12)? as usize,
                library_misses: number(13)? as usize,
                library_stored: number(14)? as usize,
                library_refused: number(15)? as usize,
            })),
            ("mem", 3) => Some(Msg::Mem { commit: number(0)?, peak_working_set: number(1)?, peak_commit: number(2)? }),
            ("bye", _) => Some(Msg::Bye { shaders: rest.to_string() }),
            _ => None,
        }
    }
}

/// The parent's line to a child (on its stdin).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Order {
    /// keys to build, in order
    Keys(Vec<u64>),
    /// no more keys: finish those given and exit (end of input means the same)
    End,
    /// the pass stops (Cancel, Ctrl+C, the budget): no more driver calls, then exit
    Stop,
}

impl Order {
    pub(super) fn line(&self) -> String {
        match self {
            Order::Keys(keys) => std::iter::once("keys".to_string()).chain(keys.iter().map(|k| format!("{k:016x}"))).collect::<Vec<_>>().join(" "),
            Order::End => "end".into(),
            Order::Stop => "stop".into(),
        }
    }

    pub(super) fn parse(line: &str) -> Option<Order> {
        let mut words = line.split_whitespace();
        match words.next()? {
            "keys" => {
                let keys = words.map(hex).collect::<Option<Vec<_>>>()?;
                (!keys.is_empty()).then_some(Order::Keys(keys))
            }
            "end" if words.next().is_none() => Some(Order::End),
            "stop" if words.next().is_none() => Some(Order::Stop),
            _ => None,
        }
    }
}

/// A child's replies on its stdout, one line each, flushed.
pub(super) struct Reply(Mutex<std::io::Stdout>);

static REPLY: std::sync::OnceLock<Reply> = std::sync::OnceLock::new();

thread_local! {
    /// the key a warm-up child's worker builds now (`working_on`)
    static WORKING_ON: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// A warm-up child's worker builds `key` now (None: between keys).
pub(super) fn working_on(key: Option<u64>) {
    WORKING_ON.with(|k| k.set(key));
}

/// The key this thread builds starts (true) or ends (false) a wait for memory
/// (`real::Governor::acquire_within`): the parent's hang clock for it stops meanwhile. Nothing
/// outside a warm-up child's worker.
pub(super) fn memory_wait(waiting: bool) {
    let (Some(key), Some(reply)) = (WORKING_ON.with(|k| k.get()), REPLY.get()) else { return };
    reply.send(&if waiting { Msg::Wait { key } } else { Msg::Go { key } });
}

impl Reply {
    /// This child's replies (the first call makes them).
    pub(super) fn stdout() -> &'static Reply {
        REPLY.get_or_init(|| Reply(Mutex::new(std::io::stdout())))
    }

    pub(super) fn send(&self, msg: &Msg) {
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if writeln!(out, "{}", msg.line()).and_then(|()| out.flush()).is_err() {
            // (CGBJ f4b48ad apps/victory/src/warm.rs `make`: whoever asked is gone)
            std::process::exit(1);
        }
    }
}

/// `xcl warm-up --blocking`'s children: `--pool-children`, `--giant-children`, `--recycle-keys`,
/// `--recycle-mb`, `--key-timeout-s`.
#[derive(Clone, Copy, Debug)]
pub struct ChildOptions {
    /// long-lived children sharing the cores (None: 2 from 8 logical cores, else 1)
    pub pool_children: Option<usize>,
    /// single-key children at once (giants and suspects; None: half the logical cores), each giant
    /// started only when its expected peak fits the memory (`giant_fits`)
    pub giant_children: Option<usize>,
    /// keys a pool child is given before it is replaced
    pub recycle_keys: usize,
    /// private commit a pool child reports from which it is replaced
    pub recycle_mb: u64,
    /// a key running longer is taken for a hang: its child is killed
    pub key_timeout_s: u64,
}

impl Default for ChildOptions {
    fn default() -> Self {
        ChildOptions { pool_children: None, giant_children: None, recycle_keys: 400, recycle_mb: 8 << 10, key_timeout_s: 1800 }
    }
}

/// The two kinds of child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lane {
    /// long-lived, many threads, recycled (`Plan::recycle_keys`, `Plan::recycle_commit`)
    Pool,
    /// one key (a giant or a suspect), then it exits
    Single,
}

/// `ChildOptions` for this PC's cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub pool_children: usize,
    pub pool_threads: usize,
    pub single_children: usize,
    pub single_dxc_threads: usize,
    pub recycle_keys: usize,
    pub recycle_commit: u64,
    pub key_timeout: Duration,
    /// `XCL_DXC_THREADS` as set by hand: every child keeps it (`ProcessSpawner`), and a pool child
    /// has no more workers than that (`with_env`)
    pub dxc_set: Option<usize>,
    /// a pool child is replaced only once the one it replaces has exited (`with_env`)
    pub strict_pool: bool,
    /// keys go to the pool children as the memory allows (`with_env`, `Follow`)
    pub follow: Option<Follow>,
    /// the giants get their children only once every pool key is done (`with_env`;
    /// `XCL_WARM_GIANTS_LAST=0`: beside the pool). play-102w's hour: its one giant child finished
    /// no giant (one ran past the 1,800 s timeout), took the RAM available from 22 GB to 2 to 4 GB
    /// and held the pool back – no pool compile ran for 66% of the hour, and its 8 threads were
    /// busy 25% of it – so 7,172 small keys took 3,572 s (2.0 keys/s) that its try1 checked at 14
    /// to 34 keys/s with no giant running.
    pub giants_last: bool,
    /// the giants are left out of the pass, for the game's background builds (`with_env`;
    /// `XCL_WARM_SKIP_GIANTS=1`; off by default): a pass that ends once the rest is ready
    pub skip_giants: bool,
    /// the pass stops itself when the RAM available stays under `URGENT_FLOOR` and this for
    /// `LOW_GRACE` (`with_env`; `XCL_WARM_LOW_MB`, 512 by default; 0: off): its single-key
    /// children are killed first, then, still under it, its pool children; their keys are left for
    /// the next pass, neither suspect nor bad. The 102m test at 8 pool children ended at 168 MB
    /// available in the run's own guard, which kills the whole process tree.
    pub low_memory: Option<u64>,
    /// at most this many children started and not yet ready (no hello) at once (`with_env`;
    /// `XCL_WARM_STARTS`, 2 by default; 0: no limit). A child's device creation maps NVIDIA's
    /// shader cache (58 GB of DXCache on the PC) and holds ~20 GB of it until it is trimmed:
    /// play-102o's 8 pool children starting together took the RAM available from 24.6 to 8.7 GB,
    /// and one giant child starting took it from 21.6 to 1.8 GB, where the pass stopped itself.
    pub starts: Option<usize>,
}

/// The pool children are fed as the memory allows (`XCL_WARM_FOLLOW_MEMORY=0`: as their workers
/// free up): a key is given only while the memory available, less `per_key` for each key given
/// and not started or started within `FRESH` (its memory does not show yet) and `CHILD_BASELINE`
/// for each pool child that has not said hello, holds `per_key` above `URGENT_FLOOR` and `margin`.
/// With nothing in flight anywhere one key is always given. The children's governors keep the
/// same margin (`margin`), and so does a giant's child (`giant_fits`). play-102's passes ran the
/// PC into its 1 GB stop: 25 driver compiles admitted at 14 to 15 GB available took it to 1.7 to
/// 2.3 GB within seconds, each time as a replacement pool child started beside the one it replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Follow {
    pub margin: u64,
    pub per_key: u64,
}

/// What a warm-up process's driver compiles keep available above `URGENT_FLOOR`: the pass's
/// `Follow::margin`, from the environment its children inherit (`real::DRIVER_COMPILES`).
pub(super) fn governor_margin() -> u64 {
    static MARGIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MARGIN.get_or_init(|| Plan::new(&ChildOptions::default(), 1).with_env(|name| std::env::var(name).ok()).margin())
}

/// How long a key's memory is taken not to show in the memory available yet (`Follow`).
pub(super) const FRESH: Duration = Duration::from_secs(10);

/// How long the RAM available stays low before the pass sheds children (`Plan::low_memory`).
const LOW_GRACE: Duration = Duration::from_secs(2);

impl Plan {
    pub(super) fn new(options: &ChildOptions, cores: usize) -> Plan {
        let cores = cores.max(1);
        let pool_children = options.pool_children.unwrap_or(if cores >= 8 { 2 } else { 1 }).clamp(1, cores);
        let single_children = options.giant_children.unwrap_or(cores / 2).clamp(1, cores).min(super::warm_peers::SLOTS.saturating_sub(2 * pool_children + 1)).max(1);
        Plan {
            pool_children,
            pool_threads: (cores / pool_children).max(1),
            single_children,
            single_dxc_threads: (cores / (pool_children + single_children)).max(2),
            recycle_keys: options.recycle_keys.max(1),
            recycle_commit: options.recycle_mb.max(1) << 20,
            key_timeout: Duration::from_secs(options.key_timeout_s.max(1)),
            dxc_set: None,
            strict_pool: false,
            follow: None,
            giants_last: false,
            skip_giants: false,
            low_memory: None,
            starts: None,
        }
    }

    /// The plan under the environment (`var`): `XCL_DXC_THREADS` caps a pool child's workers as
    /// well as its DXC threads, `XCL_WARM_POOL_THREADS` sets them; a pool child is replaced only
    /// once the one it replaces has exited (`XCL_WARM_STRICT_POOL=0`: beside it, as before); keys
    /// follow the memory (`Follow`; `XCL_WARM_FOLLOW_MEMORY=0`: off) with `XCL_WARM_MARGIN_MB` above
    /// the 1 GB floor (2048) and `XCL_WARM_KEY_MB` for each key whose memory does not show yet
    /// (512). play-102's pass 3 set `XCL_DXC_THREADS=4` and its pool child still ran 24 workers.
    pub(super) fn with_env(mut self, var: impl Fn(&str) -> Option<String>) -> Plan {
        let number = |name: &str| var(name).and_then(|v| v.trim().parse::<u64>().ok());
        let on = |name: &str| var(name).map_or(true, |v| v.trim() != "0");
        self.dxc_set = number("XCL_DXC_THREADS").map(|n| n.clamp(1, 64) as usize);
        let threads = number("XCL_WARM_POOL_THREADS").map(|n| n.clamp(1, 64) as usize).or(self.dxc_set);
        self.pool_threads = threads.map_or(self.pool_threads, |n| self.pool_threads.min(n));
        self.strict_pool = on("XCL_WARM_STRICT_POOL");
        self.giants_last = on("XCL_WARM_GIANTS_LAST");
        self.skip_giants = number("XCL_WARM_SKIP_GIANTS").is_some_and(|n| n != 0);
        self.low_memory = Some(number("XCL_WARM_LOW_MB").unwrap_or(512).min(64 << 10) << 20).filter(|&b| b != 0);
        self.starts = Some(number("XCL_WARM_STARTS").unwrap_or(2).min(64) as usize).filter(|&n| n != 0);
        self.follow = on("XCL_WARM_FOLLOW_MEMORY").then(|| Follow {
            margin: number("XCL_WARM_MARGIN_MB").unwrap_or(2048).min(64 << 10) << 20,
            per_key: number("XCL_WARM_KEY_MB").unwrap_or(512).clamp(64, 16 << 10) << 20,
        });
        self
    }

    /// What every admission of the pass keeps above `URGENT_FLOOR` (`Follow::margin`).
    pub(super) fn margin(&self) -> u64 {
        self.follow.map_or(0, |f| f.margin)
    }

    pub(super) fn threads(&self, lane: Lane) -> usize {
        match lane {
            Lane::Pool => self.pool_threads,
            Lane::Single => 1,
        }
    }

    /// A child's share of the cores for DXC (`XCL_DXC_THREADS`).
    pub(super) fn dxc_threads(&self, lane: Lane) -> usize {
        if let Some(n) = self.dxc_set {
            return n;
        }
        match lane {
            Lane::Pool => self.pool_threads,
            Lane::Single => self.single_dxc_threads,
        }
    }

    /// Worker threads of all children at once at most.
    pub(super) fn workers(&self) -> usize {
        self.pool_children * self.pool_threads + self.single_children
    }
}

/// A key to build, with what decides its lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Job {
    pub key: u64,
    pub guest: usize,
    /// a child of its own: `GIANT_GUEST_BYTES` or more, or built in `GIANT_BUILD_MS` or more before
    pub giant: bool,
    /// the DXIL length of its driver compile, when a child measured it (`Msg::Giant`)
    pub dxil: Option<usize>,
}

impl Job {
    /// `slow_ms`: the longest earlier build of it (`pso_manifest::slow_ms`).
    pub(super) fn new(key: u64, guest: usize, slow_ms: Option<u64>) -> Job {
        Job { key, guest, giant: guest >= GIANT_GUEST_BYTES || slow_ms.is_some_and(|ms| ms >= GIANT_BUILD_MS), dxil: None }
    }

    /// Its expected peak: of the DXIL a child measured, else of what its console bytes predict,
    /// for a giant at least that of `GIANT_DXIL`.
    fn peak(&self) -> u64 {
        let predicted = super::real::translated_dxil(self.guest);
        super::pso_sched::peak(self.dxil.unwrap_or(if self.giant { predicted.max(super::pso_sched::GIANT_DXIL) } else { predicted }))
    }
}

/// The memory as the parent sees it now: available, and every child's compiles (`warm_peers`).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Room {
    pub available: u64,
    pub usage: Usage,
}

/// Whether a giant of expected peak `peak` gets its child now: when its peak fits the memory
/// available less the children's reservations and `URGENT_FLOOR`, however many giants run (the
/// user's rule, 2026-11-26 15:22Z), or when no compile runs anywhere (its own governor then waits
/// for the memory, or refuses what never fits). `room` counts the giant children the table may
/// not show yet (`Coordinator::with_pending`). Its governor also waits for commit headroom.
pub(super) fn giant_fits(peak: u64, room: &Room) -> bool {
    room.usage.running == 0 || room.available.saturating_sub(room.usage.reserved) >= peak.saturating_add(super::real::URGENT_FLOOR)
}

/// What the parent does to its children.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Spawn { child: u32, lane: Lane },
    Send { child: u32, order: Order },
    Kill { child: u32 },
}

/// What the pass hears of its keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Hello { era: u64, driver: String },
    Done(Done),
    Missing(u64),
    /// in flight in a child that died: tried again alone
    Retry { key: u64, why: String },
    /// it crashed its child again, alone: listed in the bad file
    Bad { key: u64, why: String },
    /// it hung again, alone: left for the next pass (a timeout alone never makes a key bad)
    Hung { key: u64, why: String },
    /// a child of another translation or driver than the first (killed, its keys given again)
    Mismatch { era: u64, driver: String },
    /// a pool child handed back a key whose driver compile is a giant: queued for a single-key child
    Giant(u64),
    Shaders(String),
}

/// The children of a pass so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stats {
    pub started: usize,
    pub recycled: usize,
    pub crashed: usize,
    pub hung: usize,
    pub retried: usize,
    pub bad: usize,
    /// keys that hung again alone
    pub hung_keys: usize,
    /// children killed for another translation or driver
    pub mismatched: usize,
    /// keys pool children handed back as giants
    pub handed: usize,
    /// giants left out of the pass (`Plan::skip_giants`)
    pub giants_left: usize,
    /// times the pass killed children for the RAM available (`Plan::low_memory`), and the RAM
    /// available the last time
    pub memory_sheds: usize,
    pub shed_at: u64,
    /// steps in which the pool children were given fewer keys than their workers had room for,
    /// for the memory (`Follow`)
    pub memory_holds: usize,
    pub peak_working_set: u64,
    pub peak_commit: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Killed {
    /// a key ran past the timeout (None: the child never said hello in time)
    Hung { key: Option<u64> },
    Aborted,
    /// its hello named another translation or driver than the first child's
    Mismatch,
    /// the RAM available stayed low (`Plan::low_memory`)
    Memory,
}

/// A key's time in flight, its waits for memory left out (`Msg::Wait`, `Msg::Go`).
#[derive(Clone, Copy, Debug)]
struct Clock {
    ran: Duration,
    /// None: waiting for memory
    since: Option<Instant>,
}

impl Clock {
    fn new(now: Instant) -> Clock {
        Clock { ran: Duration::ZERO, since: Some(now) }
    }

    fn at(&self, now: Instant) -> Duration {
        self.ran + self.since.map_or(Duration::ZERO, |since| now.saturating_duration_since(since))
    }

    fn wait(&mut self, now: Instant) {
        if let Some(since) = self.since.take() {
            self.ran += now.saturating_duration_since(since);
        }
    }

    fn go(&mut self, now: Instant) {
        self.since.get_or_insert(now);
    }
}

struct Child {
    lane: Lane,
    spawned: Instant,
    hello: bool,
    /// keys given so far
    given: usize,
    /// given, not started
    assigned: VecDeque<u64>,
    /// started, not done (a single-key child's key from its start)
    running: BTreeMap<u64, Clock>,
    /// told to end (or stop): given no more keys
    ending: bool,
    stop_sent: bool,
    killed: Option<Killed>,
    kill_sent: bool,
    commit: u64,
    bye: bool,
    /// a single-key child's key is a suspect
    retry: bool,
    /// a single-key child's key is a giant
    giant: bool,
}

impl Child {
    fn new(lane: Lane, now: Instant) -> Child {
        Child { lane, spawned: now, hello: false, given: 0, assigned: VecDeque::new(), running: BTreeMap::new(), ending: false, stop_sent: false, killed: None, kill_sent: false, commit: 0, bye: false, retry: false, giant: false }
    }

    fn live(&self) -> bool {
        self.killed.is_none() && !self.ending
    }

    fn outstanding(&self) -> usize {
        self.assigned.len() + self.running.len()
    }

    fn forget(&mut self, key: u64) {
        self.assigned.retain(|&k| k != key);
        self.running.remove(&key);
    }
}

/// The parent's decisions, apart from processes and time (`run` carries them out): which child
/// gets which key, when a child is recycled or killed, and what a dead child's keys become.
pub(super) struct Coordinator {
    plan: Plan,
    jobs: HashMap<u64, Job>,
    pool: VecDeque<u64>,
    giants: VecDeque<u64>,
    /// suspects waiting for their retry alone
    retries: VecDeque<u64>,
    /// keys whose child died once
    suspects: HashSet<u64>,
    children: BTreeMap<u32, Child>,
    next_child: u32,
    /// the keys are known (after the first hello)
    known: bool,
    stopping: bool,
    fruitless: usize,
    /// the first child's translation era and driver: every child's must be the same
    identity: Option<(u64, String)>,
    aborted: Option<String>,
    /// since when the next giant has waited for memory with no giant compiling in the table
    giant_waiting: Option<Instant>,
    /// since when the RAM available has been under `Plan::low_memory`
    low_since: Option<Instant>,
    pub(super) stats: Stats,
}

impl Coordinator {
    pub(super) fn new(plan: Plan) -> Coordinator {
        Coordinator {
            plan,
            jobs: HashMap::new(),
            pool: VecDeque::new(),
            giants: VecDeque::new(),
            retries: VecDeque::new(),
            suspects: HashSet::new(),
            children: BTreeMap::new(),
            next_child: 0,
            known: false,
            stopping: false,
            fruitless: 0,
            identity: None,
            aborted: None,
            giant_waiting: None,
            low_since: None,
            stats: Stats::default(),
        }
    }

    /// The first child, a pool child: its hello gives the driver the keys depend on.
    pub(super) fn start(&mut self, now: Instant) -> Vec<Action> {
        vec![self.spawn(Lane::Pool, now)]
    }

    fn spawn(&mut self, lane: Lane, now: Instant) -> Action {
        let child = self.next_child;
        self.next_child += 1;
        self.children.insert(child, Child::new(lane, now));
        self.stats.started += 1;
        Action::Spawn { child, lane }
    }

    /// The keys of the pass, in order: giants to the single-key lane, the rest to the pool.
    pub(super) fn set_jobs(&mut self, jobs: Vec<Job>) {
        for job in jobs {
            if self.jobs.insert(job.key, job).is_none() {
                if job.giant && self.plan.skip_giants {
                    self.stats.giants_left += 1;
                } else {
                    if job.giant { &mut self.giants } else { &mut self.pool }.push_back(job.key);
                }
            }
        }
        self.known = true;
    }

    pub(super) fn stop(&mut self) {
        self.stopping = true;
    }

    pub(super) fn abort(&mut self, why: String) {
        self.aborted.get_or_insert(why);
    }

    pub(super) fn aborted(&self) -> Option<&str> {
        self.aborted.as_deref()
    }

    /// Every child has exited, and nothing is left to do or the pass stops.
    pub(super) fn finished(&self) -> bool {
        self.children.is_empty() && (self.aborted.is_some() || self.stopping || (self.known && self.not_started() == 0))
    }

    /// Keys not given to a child yet, suspects waiting for their retry among them.
    pub(super) fn not_started(&self) -> usize {
        self.pool.len() + self.giants.len() + self.retries.len()
    }

    /// (pool children, single-key children) running now.
    pub(super) fn children(&self) -> (usize, usize) {
        let pool = self.children.values().filter(|c| c.lane == Lane::Pool).count();
        (pool, self.children.len() - pool)
    }

    /// Another child may be started: fewer than `Plan::starts` are starting (not said hello).
    fn may_start(&self) -> bool {
        self.plan.starts.is_none_or(|most| self.children.values().filter(|c| c.killed.is_none() && !c.hello).count() < most)
    }

    fn requeue(&mut self, key: u64, retry: bool) {
        if retry {
            self.retries.push_front(key);
        } else if self.jobs.get(&key).is_some_and(|job| job.giant) {
            self.giants.push_front(key);
        } else {
            self.pool.push_front(key);
        }
    }

    pub(super) fn on_msg(&mut self, child: u32, msg: Msg, now: Instant) -> Vec<Outcome> {
        let Some(c) = self.children.get_mut(&child) else { return Vec::new() };
        match msg {
            Msg::Hello { era, driver } => match &self.identity {
                None => {
                    c.hello = true;
                    self.identity = Some((era, driver.clone()));
                    vec![Outcome::Hello { era, driver }]
                }
                Some((first_era, first_driver)) if *first_era == era && *first_driver == driver => {
                    c.hello = true;
                    Vec::new()
                }
                Some((first_era, first_driver)) => {
                    c.killed = Some(Killed::Mismatch);
                    self.stats.mismatched += 1;
                    if self.stats.mismatched >= MISMATCHES {
                        self.aborted.get_or_insert(format!(
                            "{} warm-up children reported another translation or driver than the first ({first_era:016x}, {first_driver}); the last: {era:016x}, {driver}",
                            self.stats.mismatched
                        ));
                    }
                    vec![Outcome::Mismatch { era, driver }]
                }
            },
            Msg::Start { key } => {
                c.assigned.retain(|&k| k != key);
                c.running.insert(key, Clock::new(now));
                Vec::new()
            }
            Msg::Wait { key } => {
                c.running.get_mut(&key).into_iter().for_each(|clock| clock.wait(now));
                Vec::new()
            }
            Msg::Go { key } => {
                c.running.get_mut(&key).into_iter().for_each(|clock| clock.go(now));
                Vec::new()
            }
            Msg::Giant { key, dxil_len } => {
                if !c.running.contains_key(&key) && !c.assigned.contains(&key) {
                    return Vec::new();
                }
                c.forget(key);
                let job = self.jobs.entry(key).or_insert(Job { key, guest: 0, giant: true, dxil: None });
                (job.giant, job.dxil) = (true, Some(dxil_len));
                self.giants.push_back(key);
                self.stats.handed += 1;
                vec![Outcome::Giant(key)]
            }
            Msg::Done(done) => {
                c.forget(done.key);
                self.fruitless = 0;
                vec![Outcome::Done(done)]
            }
            Msg::Missing { key } => {
                c.forget(key);
                self.fruitless = 0;
                vec![Outcome::Missing(key)]
            }
            Msg::Mem { commit, peak_working_set, peak_commit } => {
                c.commit = commit;
                self.stats.peak_working_set = self.stats.peak_working_set.max(peak_working_set);
                self.stats.peak_commit = self.stats.peak_commit.max(peak_commit);
                Vec::new()
            }
            Msg::Bye { shaders } => {
                c.bye = true;
                vec![Outcome::Shaders(shaders)]
            }
        }
    }

    /// Child `child` exited (`status` as the system says). Keys it was given but never started go
    /// back where they came from, as do those a child that said bye left (stopped before they
    /// started); after a crash or a hang kill, each key it had in flight is a suspect, tried again
    /// alone, and one that was a suspect already is bad.
    pub(super) fn on_exit(&mut self, child: u32, success: bool, status: &str) -> Vec<Outcome> {
        let Some(c) = self.children.remove(&child) else { return Vec::new() };
        for &key in c.assigned.iter().rev() {
            self.requeue(key, false);
        }
        let running: Vec<u64> = c.running.keys().copied().collect();
        if success && c.bye && c.killed.is_none() {
            for &key in running.iter().rev() {
                self.requeue(key, c.retry);
            }
            return Vec::new();
        }
        if matches!(c.killed, Some(Killed::Aborted | Killed::Mismatch | Killed::Memory)) || !c.hello {
            for &key in running.iter().rev() {
                self.requeue(key, c.retry);
            }
            if c.killed.is_none() || matches!(c.killed, Some(Killed::Hung { .. })) {
                self.fruitless(status);
            }
            return Vec::new();
        }
        let hung = match c.killed {
            Some(Killed::Hung { key }) => Some(key),
            _ => None,
        };
        if hung.is_some() {
            self.stats.hung += 1;
        } else {
            self.stats.crashed += 1;
        }
        if running.is_empty() {
            self.fruitless(status);
        }
        let timeout = self.plan.key_timeout.as_secs();
        let mut out = Vec::new();
        for key in running {
            let why = match hung {
                Some(Some(k)) if k == key => format!("ran past --key-timeout-s {timeout} and its child was killed"),
                Some(Some(k)) => format!("was in flight when its child was killed for the hang of {k:016x}"),
                Some(None) => "was in flight when its child was killed".to_string(),
                None => format!("was in flight when its child exited ({status})"),
            };
            if self.suspects.insert(key) {
                self.retries.push_back(key);
                out.push(Outcome::Retry { key, why });
            } else if hung.is_none() {
                self.stats.bad += 1;
                out.push(Outcome::Bad { key, why });
            } else {
                self.stats.hung_keys += 1;
                out.push(Outcome::Hung { key, why });
            }
        }
        out
    }

    fn fruitless(&mut self, status: &str) {
        self.fruitless += 1;
        if self.fruitless >= FRUITLESS {
            self.abort(format!("{} warm-up children in a row ended without a key done (last: {status})", self.fruitless));
        }
    }

    /// What to do now: kill the children that hang, stop or feed the others, start new ones.
    pub(super) fn step(&mut self, now: Instant, room: &Room) -> Vec<Action> {
        let mut actions = Vec::new();
        let timeout = self.plan.key_timeout;
        // (CGBJ f4b48ad apps/victory/src/warm.rs `child`: a child not heard of for LONGEST is killed;
        // here per key in flight, and a child that does not say hello in time)
        for c in self.children.values_mut().filter(|c| c.killed.is_none()) {
            let hung = c.running.iter().map(|(key, clock)| (*key, clock.at(now))).filter(|(_, ran)| *ran >= timeout).max_by_key(|(_, ran)| *ran).map(|(key, _)| key);
            if hung.is_some() || (!c.hello && now.saturating_duration_since(c.spawned) >= timeout) {
                c.killed = Some(Killed::Hung { key: hung });
            }
        }
        self.shed_for_memory(now, room);
        if self.aborted.is_some() {
            for c in self.children.values_mut().filter(|c| c.killed.is_none()) {
                c.killed = Some(Killed::Aborted);
            }
        }
        for (&child, c) in self.children.iter_mut().filter(|(_, c)| c.killed.is_some() && !c.kill_sent) {
            c.kill_sent = true;
            actions.push(Action::Kill { child });
        }
        if self.aborted.is_some() {
            return actions;
        }
        if self.stopping {
            for (&child, c) in self.children.iter_mut().filter(|(_, c)| c.killed.is_none() && !c.stop_sent) {
                c.stop_sent = true;
                c.ending = true;
                actions.push(Action::Send { child, order: Order::Stop });
            }
            return actions;
        }
        if !self.known {
            if self.children.is_empty() {
                actions.push(self.spawn(Lane::Pool, now));
            }
            return actions;
        }
        let gate = self.start_singles(now, room, &mut actions);
        self.feed_pool(now, gate, room, &mut actions);
        actions
    }

    /// The RAM available under `Plan::low_memory` for `LOW_GRACE`: the pass stops, its single-key
    /// children are killed, and if it stays low for another `LOW_GRACE` its pool children too.
    fn shed_for_memory(&mut self, now: Instant, room: &Room) {
        let Some(low) = self.plan.low_memory else { return };
        if room.available >= super::real::URGENT_FLOOR.saturating_add(low) {
            self.low_since = None;
            return;
        }
        if now.duration_since(*self.low_since.get_or_insert(now)) < LOW_GRACE {
            return;
        }
        let singles = self.children.values().any(|c| c.lane == Lane::Single && c.killed.is_none());
        let shed: Vec<&mut Child> = self.children.values_mut().filter(|c| c.killed.is_none() && (c.lane == Lane::Single || !singles)).collect();
        if shed.is_empty() {
            return;
        }
        for c in shed {
            c.killed = Some(Killed::Memory);
        }
        self.stopping = true;
        self.low_since = Some(now);
        self.stats.memory_sheds += 1;
        self.stats.shed_at = room.available;
    }

    /// Keys the pool children may be given now (`Follow`; unlimited without it).
    fn pool_allowance(&self, now: Instant, room: &Room) -> usize {
        let Some(follow) = self.plan.follow else { return usize::MAX };
        let pool = || self.children.values().filter(|c| c.lane == Lane::Pool && c.killed.is_none());
        let fresh: usize = pool().map(|c| c.assigned.len() + c.running.values().filter(|clock| clock.at(now) < FRESH).count()).sum();
        let starting = pool().filter(|c| !c.hello).count() as u64;
        let held = (fresh as u64).saturating_mul(follow.per_key).saturating_add(starting * CHILD_BASELINE);
        let spare = room.available.saturating_sub(super::real::URGENT_FLOOR + follow.margin).saturating_sub(held);
        let allowed = (spare / follow.per_key) as usize;
        let idle = room.usage.running == 0 && pool().all(|c| c.outstanding() == 0);
        if idle { allowed.max(1) } else { allowed }
    }

    /// `room` with the single-key giant children the shared table may not show yet, and what was
    /// added. Each giant child with its key in flight is pending; the table does not say whose its
    /// giants are, so as many pending children as it has giants (`Usage::giants`) are taken for
    /// published, and the largest peaks of the rest are added as reserved and running, with
    /// `CHILD_BASELINE` for each pending child that has not said hello. Without it every step
    /// would spend the same memory again on children started since the last one.
    fn with_pending(&self, room: &Room) -> (Room, u64) {
        let pending = self.children.values().filter(|c| c.lane == Lane::Single && c.giant && c.killed.is_none() && !c.running.is_empty());
        let (mut peaks, mut added) = (Vec::new(), 0u64);
        for c in pending {
            peaks.extend(c.running.keys().map(|key| self.jobs.get(key).map_or(0, Job::peak)));
            added += if c.hello { 0 } else { CHILD_BASELINE };
        }
        let unpublished = peaks.len() - peaks.len().min(room.usage.giants);
        peaks.sort_unstable_by(|a, b| b.cmp(a));
        added += peaks[..unpublished].iter().sum::<u64>();
        let mut room = *room;
        room.usage.reserved = room.usage.reserved.saturating_add(added);
        room.usage.running += unpublished;
        (room, added)
    }

    /// Single-key children for the suspects, then for the giants that fit (`giant_fits`, with the
    /// children not in the table yet: `with_pending`). True: the next giant waits for memory with
    /// no giant compiling in the table, and either the pool holds that memory (it would fit with
    /// the pool's reservations and its children's last reported commit gone) or it has waited
    /// `GIANT_PATIENCE`; the pool is then not fed until a giant starts (a waiting giant keeps
    /// smaller compiles from taking its memory). A giant child waiting in its own governor is not
    /// published across processes: the pool pause covers it. A giant waiting while another giant
    /// compiles never pauses it.
    fn start_singles(&mut self, now: Instant, room: &Room, actions: &mut Vec<Action>) -> bool {
        let (mut room, mut pending) = self.with_pending(room);
        loop {
            let count = self.children.values().filter(|c| c.lane == Lane::Single).count();
            if count >= self.plan.single_children {
                self.giant_waiting = None;
                return false;
            }
            let (retry, key) = match (self.retries.front(), self.giants.front()) {
                (Some(&key), _) => (true, key),
                (None, Some(&key)) => (false, key),
                (None, None) => {
                    self.giant_waiting = None;
                    return false;
                }
            };
            let job = self.jobs.get(&key).copied().unwrap_or(Job { key, guest: 0, giant: false, dxil: None });
            let pool_busy = || !self.pool.is_empty() || self.children.values().any(|c| c.lane == Lane::Pool && c.killed.is_none() && c.outstanding() > 0);
            if !retry && self.plan.giants_last && pool_busy() {
                self.giant_waiting = None;
                return false;
            }
            if job.giant && !giant_fits(job.peak().saturating_add(self.plan.margin()), &room) {
                if room.usage.giants != 0 {
                    self.giant_waiting = None;
                    return false;
                }
                let waited = now.duration_since(*self.giant_waiting.get_or_insert(now));
                let held: u64 = self.children.values().filter(|c| c.lane == Lane::Pool).map(|c| c.commit).sum();
                let published = room.usage.reserved.saturating_sub(pending);
                let drained = room.available.saturating_add(published).saturating_add(held).saturating_sub(pending);
                return waited >= GIANT_PATIENCE || drained >= job.peak().saturating_add(super::real::URGENT_FLOOR);
            }
            if !self.may_start() {
                return false;
            }
            if job.giant {
                self.giant_waiting = None;
            }
            if retry {
                self.retries.pop_front();
                self.stats.retried += 1;
            } else {
                self.giants.pop_front();
            }
            let spawn = self.spawn(Lane::Single, now);
            let Action::Spawn { child, .. } = spawn else { unreachable!() };
            let c = self.children.get_mut(&child).unwrap();
            (c.retry, c.giant, c.given, c.ending) = (retry, job.giant, 1, true);
            c.running.insert(key, Clock::new(now));
            if job.giant {
                let taken = job.peak().saturating_add(CHILD_BASELINE);
                room.usage.reserved = room.usage.reserved.saturating_add(taken);
                room.usage.running += 1;
                pending = pending.saturating_add(taken);
            }
            actions.extend([spawn, Action::Send { child, order: Order::Keys(vec![key]) }, Action::Send { child, order: Order::End }]);
        }
    }

    /// Each pool child kept at its threads plus one key given; one given `recycle_keys` or over
    /// `recycle_commit` is told to end and another takes its place. While a giant has waited for
    /// memory twice `GIANT_PATIENCE` (`gate`), every pool child is told to end, its memory freed.
    fn feed_pool(&mut self, now: Instant, gate: bool, room: &Room, actions: &mut Vec<Action>) {
        let plan = self.plan;
        let drain = gate && self.giant_waiting.is_some_and(|since| now.duration_since(since) >= 2 * GIANT_PATIENCE);
        let mut allowance = self.pool_allowance(now, room);
        let mut held = false;
        let mut feed = |c: &mut Child, pool: &mut VecDeque<u64>| {
            let room_for = (plan.pool_threads + 1).saturating_sub(c.outstanding()).min(plan.recycle_keys.saturating_sub(c.given));
            let want = room_for.min(allowance);
            held |= want < room_for && !pool.is_empty();
            allowance -= want.min(allowance);
            let keys: Vec<u64> = (0..want).map_while(|_| pool.pop_front()).collect();
            c.given += keys.len();
            c.assigned.extend(&keys);
            (!keys.is_empty()).then_some(Order::Keys(keys))
        };
        for (&child, c) in self.children.iter_mut().filter(|(_, c)| c.lane == Lane::Pool && c.live()) {
            let worn = c.given >= plan.recycle_keys || c.commit >= plan.recycle_commit;
            if worn || drain || self.pool.is_empty() {
                c.ending = true;
                self.stats.recycled += ((worn || drain) && !self.pool.is_empty()) as usize;
                actions.push(Action::Send { child, order: Order::End });
            } else if !gate {
                actions.extend(feed(c, &mut self.pool).map(|order| Action::Send { child, order }));
            }
        }
        // (strict: a pool child told to end still counts until it has exited)
        let counted = |c: &Child| c.lane == Lane::Pool && if plan.strict_pool { c.killed.is_none() } else { c.live() };
        while !gate && !self.pool.is_empty() && self.children.values().filter(|c| counted(c)).count() < plan.pool_children && self.may_start() {
            let spawn = self.spawn(Lane::Pool, now);
            let Action::Spawn { child, .. } = spawn else { unreachable!() };
            actions.push(spawn);
            let c = self.children.get_mut(&child).unwrap();
            actions.extend(feed(c, &mut self.pool).map(|order| Action::Send { child, order }));
        }
        self.stats.memory_holds += held as usize;
    }
}

/// How a child's process ended, or a line it wrote.
pub(super) enum Event {
    Line(String),
    Exited { success: bool, status: String },
}

pub(super) trait ChildHandle {
    fn send(&mut self, order: &Order);
    fn kill(&mut self);
}

/// Why a child did not start.
#[derive(Debug)]
pub(super) enum SpawnError {
    /// counted as a child that ended before its hello
    Retry(String),
    /// the pass stops (a child that could not be put in the job object)
    Fatal(String),
}

/// Starts children; their lines and their exit come back on `events`, in that order.
pub(super) trait Spawner {
    fn spawn(&mut self, child: u32, lane: Lane, events: mpsc::Sender<(u32, Event)>) -> Result<Box<dyn ChildHandle>, SpawnError>;
}

/// The pass the children work for: its keys, its lines and its memory.
pub(super) trait Pass {
    /// The first child initialised (its driver and translation era): the keys to build.
    fn hello(&mut self, era: u64, driver: &str) -> Result<Vec<Job>, String>;
    fn outcome(&mut self, outcome: &Outcome);
    fn tick(&mut self, coordinator: &Coordinator);
    fn stopping(&self) -> bool;
    fn room(&self) -> Room;
}

/// A pass in children of `spawner` (`Coordinator`), until every key is done or the pass stops.
pub(super) fn run(plan: Plan, spawner: &mut dyn Spawner, pass: &mut dyn Pass) -> Result<Stats, String> {
    let (events, heard) = mpsc::channel();
    let mut coordinator = Coordinator::new(plan);
    let mut handles: HashMap<u32, Box<dyn ChildHandle>> = HashMap::new();
    let mut hello = false;
    let mut actions = coordinator.start(Instant::now());
    loop {
        let mut outcomes = Vec::new();
        for action in actions.drain(..) {
            match action {
                Action::Spawn { child, lane } => match spawner.spawn(child, lane, events.clone()) {
                    Ok(handle) => drop(handles.insert(child, handle)),
                    Err(SpawnError::Retry(e)) => outcomes.extend(coordinator.on_exit(child, false, &format!("could not start: {e}"))),
                    Err(SpawnError::Fatal(e)) => {
                        coordinator.abort(e.clone());
                        outcomes.extend(coordinator.on_exit(child, false, &e));
                    }
                },
                Action::Send { child, order } => handles.get_mut(&child).into_iter().for_each(|h| h.send(&order)),
                Action::Kill { child } => handles.get_mut(&child).into_iter().for_each(|h| h.kill()),
            }
        }
        if coordinator.finished() {
            break;
        }
        match heard.recv_timeout(TICK) {
            Ok((child, Event::Line(line))) => match Msg::parse(&line) {
                Some(msg) => outcomes.extend(coordinator.on_msg(child, msg, Instant::now())),
                None => eprintln!("{line}"),
            },
            Ok((child, Event::Exited { success, status })) => {
                handles.remove(&child);
                outcomes.extend(coordinator.on_exit(child, success, &status));
            }
            Err(_) => {}
        }
        for outcome in outcomes {
            match outcome {
                Outcome::Hello { era, driver } if !hello => {
                    hello = true;
                    match pass.hello(era, &driver) {
                        Ok(jobs) => coordinator.set_jobs(jobs),
                        Err(e) => coordinator.abort(e),
                    }
                }
                Outcome::Hello { .. } => {}
                other => pass.outcome(&other),
            }
        }
        if pass.stopping() {
            coordinator.stop();
        }
        pass.tick(&coordinator);
        let sheds = coordinator.stats.memory_sheds;
        actions = coordinator.step(Instant::now(), &pass.room());
        if coordinator.stats.memory_sheds > sheds {
            eprintln!(
                "warm-up --blocking: stopping: {} MiB RAM available, under the floor and XCL_WARM_LOW_MB for {} s; {} children killed, their keys left for the next pass",
                coordinator.stats.shed_at >> 20,
                LOW_GRACE.as_secs(),
                actions.iter().filter(|a| matches!(a, Action::Kill { .. })).count()
            );
        }
    }
    match coordinator.aborted() {
        Some(why) => Err(why.to_string()),
        None => Ok(coordinator.stats),
    }
}

/// Children as processes of this exe (`xcl warm-up --blocking-child`): keys on stdin, replies on
/// stdout, logs on the inherited stderr, each in a slot of `peers` and in a job that ends them
/// with the parent (started suspended, put in the job, then resumed; a child that cannot be put
/// in it stops the pass).
pub(super) struct ProcessSpawner {
    exe: std::path::PathBuf,
    plan: Plan,
    deadline: Instant,
    peers: Arc<super::warm_peers::Peers>,
    slots: Arc<Mutex<Vec<bool>>>,
    job: JobObject,
    /// `XCL_DXC_THREADS` was set by hand: every child keeps it
    dxc_set: bool,
}

impl ProcessSpawner {
    pub(super) fn new(plan: Plan, deadline: Instant, peers: Arc<super::warm_peers::Peers>) -> std::io::Result<ProcessSpawner> {
        Ok(ProcessSpawner {
            exe: std::env::current_exe()?,
            plan,
            deadline,
            peers,
            slots: Arc::new(Mutex::new(vec![false; super::warm_peers::SLOTS])),
            job: JobObject::new()?,
            dxc_set: std::env::var_os("XCL_DXC_THREADS").is_some(),
        })
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Spawner for ProcessSpawner {
    fn spawn(&mut self, child: u32, lane: Lane, events: mpsc::Sender<(u32, Event)>) -> Result<Box<dyn ChildHandle>, SpawnError> {
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        let retry = |e: std::io::Error| SpawnError::Retry(e.to_string());
        let slot = {
            let mut slots = lock(&self.slots);
            let free = slots.iter().position(|used| !used).ok_or_else(|| SpawnError::Retry("every warm-up peer slot is taken".into()))?;
            slots[free] = true;
            free
        };
        let budget = self.deadline.saturating_duration_since(Instant::now()).as_secs().max(1);
        let mut command = Command::new(&self.exe);
        command
            .args(["warm-up", "--blocking-child", "--threads", &self.plan.threads(lane).to_string(), "--budget-s", &budget.to_string()])
            .args((lane == Lane::Pool).then_some("--leave-giants"))
            .env(super::warm_peers::ENV, format!("{}:{slot}", std::process::id()))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .creation_flags(windows_sys::Win32::System::Threading::CREATE_SUSPENDED);
        if !self.dxc_set {
            command.env("XCL_DXC_THREADS", self.plan.dxc_threads(lane).to_string());
        }
        // (CGBJ f4b48ad apps/victory/src/warm.rs `child`: the same exe, replies on a piped stdout)
        let mut process = match command.spawn() {
            Ok(process) => process,
            Err(e) => {
                lock(&self.slots)[slot] = false;
                return Err(retry(e));
            }
        };
        // suspended until it is in the job: it cannot outlive the parent from its first instruction
        if let Err(e) = self.job.assign(&process).and_then(|()| resume(process.id())) {
            let _ = process.kill();
            let _ = process.wait();
            lock(&self.slots)[slot] = false;
            return Err(SpawnError::Fatal(format!("could not start a warm-up child inside the warm-up's job object: {e}")));
        }
        let (stdin, stdout) = (process.stdin.take(), process.stdout.take());
        let process = Arc::new(Mutex::new(process));
        let (orders, inbox) = mpsc::channel::<String>();
        std::thread::Builder::new().name(format!("xcl-warm-child-{child}-in")).spawn(move || {
            let Some(mut stdin) = stdin else { return };
            for line in inbox {
                if writeln!(stdin, "{line}").and_then(|()| stdin.flush()).is_err() {
                    break;
                }
            }
        }).map_err(retry)?;
        let (watched, peers, slots) = (process.clone(), self.peers.clone(), self.slots.clone());
        std::thread::Builder::new().name(format!("xcl-warm-child-{child}-out")).spawn(move || {
            if let Some(out) = stdout {
                for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                    let _ = events.send((child, Event::Line(line)));
                }
            }
            let status = loop {
                match lock(&watched).try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                    Err(e) => break Err(e),
                }
            };
            peers.lock().clear(slot);
            lock(&slots)[slot] = false;
            let (success, status) = match status {
                Ok(status) => (status.success(), status.to_string()),
                Err(e) => (false, e.to_string()),
            };
            let _ = events.send((child, Event::Exited { success, status }));
        }).map_err(retry)?;
        Ok(Box::new(ProcessChild { orders, process }))
    }
}

struct ProcessChild {
    orders: mpsc::Sender<String>,
    process: Arc<Mutex<std::process::Child>>,
}

impl ChildHandle for ProcessChild {
    fn send(&mut self, order: &Order) {
        let _ = self.orders.send(order.line());
    }

    fn kill(&mut self) {
        let _ = lock(&self.process).kill();
    }
}

/// A job object that ends every child with the parent (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
struct JobObject(windows_sys::Win32::Foundation::HANDLE);

unsafe impl Send for JobObject {}

impl JobObject {
    fn new() -> std::io::Result<JobObject> {
        use windows_sys::Win32::System::JobObjects::{CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE};
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let set = SetInformationJobObject(job, JobObjectExtendedLimitInformation, (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), std::mem::size_of_val(&limits) as u32);
            if set == 0 {
                let e = std::io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(e);
            }
            Ok(JobObject(job))
        }
    }

    fn assign(&self, process: &std::process::Child) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        if unsafe { windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(self.0, process.as_raw_handle()) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Resumes process `pid`, started suspended (`CREATE_SUSPENDED`: its one thread).
pub(super) fn resume(pid: u32) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut resumed = 0;
        let mut more = Thread32First(snap, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == pid {
                let thread = OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID);
                if !thread.is_null() {
                    resumed += (ResumeThread(thread) != u32::MAX) as usize;
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snap, &mut entry) != 0;
        }
        CloseHandle(snap);
        if resumed == 0 {
            return Err(std::io::Error::other(format!("no thread of process {pid} could be resumed")));
        }
        Ok(())
    }
}

impl Drop for JobObject {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

fn driver_word(driver: &str) -> String {
    let word = driver.split_whitespace().collect::<Vec<_>>().join("_");
    if word.is_empty() { "unknown".into() } else { word }
}

// (CGBJ f4b48ad apps/victory/src/warm.rs `run`: warm-bad.txt, "<key> <why>" per line, left out of
// later passes; here with the translation era and driver it was bad under)
/// The keys `BAD_FILE` in `dir` lists for translation `era` and driver `driver`.
pub(super) fn bad_keys(dir: &Path, era: u64, driver: &str) -> HashSet<u64> {
    let text = std::fs::read_to_string(dir.join(BAD_FILE)).unwrap_or_default();
    let driver = driver_word(driver);
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split(' ');
            let key = hex(fields.next()?)?;
            (hex(fields.next()?)? == era && fields.next()? == driver).then_some(key)
        })
        .collect()
}

/// Key `key` ended its child twice: listed in `BAD_FILE` with why.
pub(super) fn note_bad(dir: &Path, key: u64, era: u64, driver: &str, why: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(BAD_FILE))?;
    writeln!(file, "{key:016x} {era:016x} {} {}", driver_word(driver), why.replace(['\r', '\n'], " "))
}

/// `--fresh`: the bad keys are tried again.
pub(super) fn clear_bad(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(BAD_FILE));
}

/// The children's `dxc::stage_summary` lines as one: the same text with each number summed (every
/// one of them is a count or a sum of seconds), or, worded differently, side by side.
pub(super) fn merge_summaries(parts: &[String]) -> String {
    fn split(s: &str) -> (Vec<&str>, Vec<&str>) {
        let (mut text, mut numbers, mut at) = (Vec::new(), Vec::new(), 0);
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i].is_ascii_digit() {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || (bytes[i] == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))) {
                    i += 1;
                }
                text.push(&s[at..start]);
                numbers.push(&s[start..i]);
                at = i;
            } else {
                i += 1;
            }
        }
        text.push(&s[at..]);
        (text, numbers)
    }
    let parts: Vec<_> = parts.iter().filter(|p| !p.trim().is_empty()).map(|p| split(p)).collect();
    let Some((shape, _)) = parts.first() else { return String::new() };
    if parts.iter().any(|(text, _)| text != shape) {
        return parts.iter().map(|(text, numbers)| {
            let mut out = String::new();
            for (i, t) in text.iter().enumerate() {
                out.push_str(t);
                out.push_str(numbers.get(i).copied().unwrap_or(""));
            }
            out
        }).collect::<Vec<_>>().join(" | ");
    }
    let mut out = String::new();
    for (i, t) in shape.iter().enumerate() {
        out.push_str(t);
        if i == shape.len() - 1 {
            break;
        }
        let column = parts.iter().map(|(_, numbers)| numbers[i]);
        let decimals = column.clone().map(|n| n.split_once('.').map_or(0, |(_, d)| d.len())).max().unwrap_or(0);
        let sum: f64 = column.map(|n| n.parse::<f64>().unwrap_or(0.0)).sum();
        out.push_str(&format!("{sum:.decimals$}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(key: u64) -> Done {
        Done { key, kind: "Graphics".into(), ok: true, refused: false, hits: 1, cold: 2, rejected: 0, saved: 2, elapsed_us: 1_500_000, admitted_us: 250_000, build_ms: 1250, guest: 4096, library_hits: 3, library_misses: 1, library_stored: 1, library_refused: 0 }
    }

    fn plan() -> Plan {
        Plan::new(&ChildOptions { pool_children: Some(2), giant_children: Some(2), recycle_keys: 4, recycle_mb: 1024, key_timeout_s: 60 }, 4)
    }

    const ROOMY: Room = Room { available: 64 << 30, usage: Usage { running: 0, giants: 0, reserved: 0 } };

    fn keys_sent(actions: &[Action], to: u32) -> Vec<u64> {
        actions.iter().filter_map(|a| match a {
            Action::Send { child, order: Order::Keys(keys) } if *child == to => Some(keys.clone()),
            _ => None,
        }).flatten().collect()
    }

    fn spawned(actions: &[Action]) -> Vec<(u32, Lane)> {
        actions.iter().filter_map(|a| match a {
            Action::Spawn { child, lane } => Some((*child, *lane)),
            _ => None,
        }).collect()
    }

    /// A coordinator whose first child said hello, with `jobs`.
    fn started(plan: Plan, jobs: Vec<Job>, now: Instant) -> Coordinator {
        let mut c = Coordinator::new(plan);
        assert_eq!(c.start(now), [Action::Spawn { child: 0, lane: Lane::Pool }]);
        assert!(matches!(c.on_msg(0, hello(), now)[..], [Outcome::Hello { era: 5, .. }]));
        c.set_jobs(jobs);
        c
    }

    fn hello() -> Msg {
        Msg::Hello { era: 5, driver: "10de-2684-a".into() }
    }

    fn small(keys: std::ops::Range<u64>) -> Vec<Job> {
        keys.map(|k| Job::new(k, 1000, None)).collect()
    }

    #[test]
    fn every_line_reads_back_as_written() {
        let msgs = [
            Msg::Hello { era: 0xc695_aa7a_3203_a941, driver: "10de-2684-0000001f00001234".into() },
            Msg::Start { key: 0xdead_beef },
            Msg::Wait { key: 0xdead_beef },
            Msg::Go { key: 1 },
            Msg::Giant { key: 0x1234, dxil_len: 1_500_000 },
            Msg::Done(done(u64::MAX)),
            Msg::Done(Done { ok: false, refused: true, ..done(7) }),
            Msg::Missing { key: 3 },
            Msg::Mem { commit: 5 << 30, peak_working_set: 6 << 30, peak_commit: 7 << 30 },
            Msg::Bye { shaders: "translation 3 from disk, 4 made; DXC 1 hits".into() },
        ];
        for msg in msgs {
            assert_eq!(Msg::parse(&msg.line()), Some(msg.clone()), "{}", msg.line());
            assert_eq!(Msg::parse(&format!("{}\r", msg.line())), Some(msg));
        }
        for order in [Order::Keys(vec![1, 0xffff_ffff_ffff_ffff]), Order::End, Order::Stop] {
            assert_eq!(Order::parse(&order.line()), Some(order));
        }
    }

    #[test]
    fn torn_and_foreign_lines_are_not_replies() {
        let line = Msg::Done(done(9)).line();
        for bad in [
            "",
            "[SHADER] CS 0123: compiled",
            "@xcl-warmdone",
            "@xcl-warm hello 12",
            "@xcl-warm hello zz driver",
            "@xcl-warm start",
            "@xcl-warm start 12 13",
            "@xcl-warm wait",
            "@xcl-warm go 1 2",
            "@xcl-warm giant",
            "@xcl-warm giant zz 1500000",
            "@xcl-warm giant 1234",
            "@xcl-warm giant 1234 x",
            "@xcl-warm mem 1 2",
            "@xcl-warm what 1",
            line.rsplit_once(' ').unwrap().0,
            &line.replace(" 1 0 ", " 2 0 "),
            &format!("{line} 4"),
        ] {
            assert_eq!(Msg::parse(bad), None, "{bad:?}");
        }
        for bad in ["keys", "keys zz", "end now", "stop 1", "go"] {
            assert_eq!(Order::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_giant_is_predicted_from_the_giant_dxil_line_or_its_measured_time() {
        assert_eq!(GIANT_GUEST_BYTES, 43_691);
        assert!(GIANT_GUEST_BYTES * super::super::real::TRANSLATED_DXIL_PER_GUEST_BYTE >= super::super::pso_sched::GIANT_DXIL);
        assert!((GIANT_GUEST_BYTES - 1) * super::super::real::TRANSLATED_DXIL_PER_GUEST_BYTE < super::super::pso_sched::GIANT_DXIL);
        assert!(!Job::new(1, GIANT_GUEST_BYTES - 1, None).giant);
        assert!(Job::new(1, GIANT_GUEST_BYTES, None).giant);
        assert!(Job::new(1, 10, Some(GIANT_BUILD_MS)).giant, "a build of a minute or more before");
        assert!(!Job::new(1, 10, Some(GIANT_BUILD_MS - 1)).giant, "merely slow keys stay in the pool");
        assert!(!Job::new(1, 10, Some(super::super::pso_manifest::SLOW_MS)).giant);
        assert_eq!(GIANT_BUILD_MS, 60_000);
    }

    #[test]
    fn the_plan_shares_the_cores() {
        let p = Plan::new(&ChildOptions::default(), 24);
        assert_eq!((p.pool_children, p.pool_threads, p.single_children, p.single_dxc_threads), (2, 12, 12, 2));
        assert_eq!((p.recycle_keys, p.recycle_commit, p.key_timeout), (400, 8 << 30, Duration::from_secs(1800)));
        assert_eq!(p.workers(), 36);
        assert_eq!((p.dxc_threads(Lane::Pool), p.dxc_threads(Lane::Single), p.threads(Lane::Single)), (12, 2, 1));
        let p = Plan::new(&ChildOptions::default(), 4);
        assert_eq!((p.pool_children, p.pool_threads, p.single_dxc_threads), (1, 4, 2));
        let p = Plan::new(&ChildOptions { pool_children: Some(64), giant_children: Some(9), ..ChildOptions::default() }, 2);
        assert_eq!((p.pool_children, p.pool_threads, p.single_children), (2, 1, 2));
        let p = Plan::new(&ChildOptions { pool_children: Some(16), giant_children: Some(200), ..ChildOptions::default() }, 256);
        assert_eq!(p.single_children, super::super::warm_peers::SLOTS - 2 * 16 - 1, "within the peer table's slots, recycled pool children overlapping");
    }

    /// play-102's pass 3: `XCL_DXC_THREADS=4` and one pool child, which still ran 24 workers.
    #[test]
    fn the_environment_caps_the_pool_workers_and_sets_the_memory_rules() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string());
        let one = ChildOptions { pool_children: Some(1), ..ChildOptions::default() };
        let p = Plan::new(&one, 24).with_env(env(&[]));
        assert_eq!((p.pool_threads, p.dxc_threads(Lane::Pool), p.dxc_set), (24, 24, None));
        assert!(p.strict_pool);
        assert_eq!(p.follow, Some(Follow { margin: 2 << 30, per_key: 512 << 20 }));
        let p = Plan::new(&one, 24).with_env(env(&[("XCL_DXC_THREADS", "4")]));
        assert_eq!((p.pool_threads, p.dxc_threads(Lane::Pool), p.dxc_threads(Lane::Single)), (4, 4, 4), "XCL_DXC_THREADS caps the workers too");
        let p = Plan::new(&one, 24).with_env(env(&[("XCL_DXC_THREADS", "4"), ("XCL_WARM_POOL_THREADS", "8")]));
        assert_eq!((p.pool_threads, p.dxc_threads(Lane::Pool)), (8, 4));
        let p = Plan::new(&ChildOptions::default(), 24).with_env(env(&[("XCL_WARM_POOL_THREADS", "64")]));
        assert_eq!(p.pool_threads, 12, "never more than the cores' share");
        let p = Plan::new(&one, 24).with_env(env(&[("XCL_WARM_STRICT_POOL", "0"), ("XCL_WARM_FOLLOW_MEMORY", "0")]));
        assert!(!p.strict_pool && p.follow.is_none() && p.margin() == 0);
        let p = Plan::new(&one, 24).with_env(env(&[("XCL_WARM_MARGIN_MB", "3072"), ("XCL_WARM_KEY_MB", "1024")]));
        assert_eq!(p.follow, Some(Follow { margin: 3 << 30, per_key: 1 << 30 }));
    }

    /// play-102o: 8 pool children starting together took the RAM available from 24.6 to 8.7 GB.
    #[test]
    fn at_most_starts_children_start_before_their_hello() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string());
        assert_eq!(Plan::new(&ChildOptions::default(), 24).with_env(env(&[])).starts, Some(2));
        assert_eq!(Plan::new(&ChildOptions::default(), 24).with_env(env(&[("XCL_WARM_STARTS", "0")])).starts, None);
        let now = Instant::now();
        let mut c = started(Plan { pool_children: 4, starts: Some(1), ..plan() }, small(0..40), now);
        assert_eq!(spawned(&c.step(now, &ROOMY)), [(1, Lane::Pool)]);
        assert_eq!(spawned(&c.step(now, &ROOMY)), [], "the next waits for its hello");
        c.on_msg(1, hello(), now);
        assert_eq!(spawned(&c.step(now, &ROOMY)), [(2, Lane::Pool)]);
        let mut c = started(Plan { pool_children: 4, ..plan() }, small(0..40), now);
        assert_eq!(spawned(&c.step(now, &ROOMY)), [(1, Lane::Pool), (2, Lane::Pool), (3, Lane::Pool)], "no limit: all at once");
    }

    /// play-102: each pass hit the RAM stop as a replacement pool child started beside the one it
    /// replaced (3 pool children with 2 asked for, 2 to 3 with 1). Strict, the replacement starts
    /// once the old one has exited.
    #[test]
    fn a_strict_pool_child_is_replaced_only_after_the_old_one_exits() {
        let now = Instant::now();
        let plan = Plan { pool_children: 1, strict_pool: true, ..plan() };
        let mut c = started(plan, small(1..20), now);
        assert_eq!(keys_sent(&c.step(now, &ROOMY), 0), [1, 2, 3]);
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Done(done(1)), now);
        assert_eq!(keys_sent(&c.step(now, &ROOMY), 0), [4], "the 4th key: recycle_keys");
        for key in [2, 3] {
            c.on_msg(0, Msg::Done(done(key)), now);
        }
        let a = c.step(now, &ROOMY);
        assert!(a.contains(&Action::Send { child: 0, order: Order::End }));
        assert!(spawned(&a).is_empty(), "the old child still runs key 4");
        c.on_msg(0, Msg::Done(done(4)), now);
        assert!(spawned(&c.step(now, &ROOMY)).is_empty(), "nor until it has exited");
        c.on_exit(0, true, "exit code 0");
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Pool)]);
        assert_eq!(keys_sent(&a, 1), [5, 6, 7]);
    }

    /// Keys follow the memory: each key given and not yet showing holds `per_key`, a child not yet
    /// started holds `CHILD_BASELINE`, and with nothing in flight one key always goes.
    #[test]
    fn pool_keys_follow_the_memory_available() {
        let now = Instant::now();
        let follow = Follow { margin: 2 << 30, per_key: 1 << 30 };
        let plan = Plan { pool_children: 1, pool_threads: 8, recycle_keys: 100, strict_pool: true, follow: Some(follow), ..plan() };
        let floor = super::super::real::URGENT_FLOOR + follow.margin;
        let room = |available: u64, running: usize| Room { available, usage: Usage { running, giants: 0, reserved: 0 } };
        let mut c = started(plan, small(1..40), now);
        assert_eq!(keys_sent(&c.step(now, &room(floor + (3 << 30), 0)), 0), [1, 2, 3], "3 GiB above the margin: 3 keys of the 9 it has room for");
        assert!(keys_sent(&c.step(now, &room(floor + (3 << 30), 3)), 0).is_empty(), "their memory does not show yet");
        for key in 1..=3 {
            c.on_msg(0, Msg::Start { key }, now);
        }
        let later = now + FRESH;
        assert_eq!(keys_sent(&c.step(later, &room(floor + (2 << 30), 3)), 0), [4, 5], "after FRESH their memory shows in the RAM available");
        assert!(keys_sent(&c.step(later, &room(floor + (2 << 30), 3)), 0).is_empty());
        assert_eq!(c.stats.memory_holds, 4);
        let mut idle = started(plan, small(1..4), now);
        assert_eq!(keys_sent(&idle.step(now, &room(1 << 30, 0)), 0), [1], "nothing in flight: one key whatever the memory");
        assert!(keys_sent(&idle.step(now, &room(1 << 30, 0)), 0).is_empty());
    }

    #[test]
    fn a_giant_starts_when_its_peak_fits_or_nothing_runs() {
        let peak = 2 << 30;
        let busy = |available: u64, reserved: u64, giants: usize| Room { available, usage: Usage { running: 4, giants, reserved } };
        assert!(giant_fits(peak, &busy(10 << 30, 6 << 30, 0)), "4 GiB spare holds 2 GiB and the 1 GiB floor");
        assert!(!giant_fits(peak, &busy(10 << 30, 8 << 30, 0)), "2 GiB spare does not");
        assert!(giant_fits(peak, &Room { available: 1 << 30, usage: Usage::default() }), "nothing runs: its governor waits");
        assert!(giant_fits(peak, &busy(64 << 30, 9 << 30, 3)), "no count of giants at once");
        assert!(!giant_fits(peak, &busy(11 << 30, 9 << 30, 3)), "memory alone holds it");
    }

    #[test]
    fn pool_children_get_their_threads_plus_one_and_are_recycled() {
        let now = Instant::now();
        let mut c = started(plan(), small(1..20), now);
        let a = c.step(now, &ROOMY);
        assert_eq!(keys_sent(&a, 0), [1, 2, 3]);
        assert_eq!(spawned(&a), [(1, Lane::Pool)]);
        assert_eq!(keys_sent(&a, 1), [4, 5, 6]);
        assert!(c.step(now, &ROOMY).is_empty(), "nothing more until a key is done");
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Done(done(1)), now);
        assert_eq!(keys_sent(&c.step(now, &ROOMY), 0), [7], "the 4th key: recycle_keys");
        for key in [2, 3, 7] {
            c.on_msg(0, Msg::Done(done(key)), now);
        }
        let a = c.step(now, &ROOMY);
        assert!(a.contains(&Action::Send { child: 0, order: Order::End }));
        assert_eq!(spawned(&a), [(2, Lane::Pool)], "another takes its place");
        assert_eq!(keys_sent(&a, 2), [8, 9, 10]);
        assert_eq!(c.stats.recycled, 1);
        c.on_msg(1, Msg::Mem { commit: 2 << 30, peak_working_set: 3 << 30, peak_commit: 2 << 30 }, now);
        let a = c.step(now, &ROOMY);
        assert!(a.contains(&Action::Send { child: 1, order: Order::End }), "over recycle_mb");
        assert_eq!(spawned(&a), [(3, Lane::Pool)]);
        assert_eq!((c.stats.recycled, c.stats.peak_working_set), (2, 3 << 30));
        c.on_msg(0, Msg::Bye { shaders: String::new() }, now);
        assert!(c.on_exit(0, true, "exit code: 0").is_empty());
        let mut c = started(plan(), small(1..2), now);
        let a = c.step(now, &ROOMY);
        assert_eq!(keys_sent(&a, 0), [1]);
        assert!(spawned(&a).is_empty(), "no second pool child for one key");
        c.on_msg(0, Msg::Done(done(1)), now);
        assert_eq!(c.step(now, &ROOMY), [Action::Send { child: 0, order: Order::End }], "no keys left: it ends");
        assert_eq!(c.stats.recycled, 0);
        c.on_msg(0, Msg::Bye { shaders: String::new() }, now);
        c.on_exit(0, true, "exit code: 0");
        assert!(c.finished());
    }

    #[test]
    fn giants_get_a_child_each_within_their_lane() {
        let now = Instant::now();
        let jobs = vec![Job::new(1, 50_000, None), Job::new(2, 1000, None), Job::new(3, 60_000, None), Job::new(4, 100, Some(GIANT_BUILD_MS)), Job::new(5, 100, Some(GIANT_BUILD_MS - 1))];
        let mut c = started(plan(), jobs, now);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single), (2, Lane::Single)], "two single-key children: --giant-children 2");
        assert_eq!(keys_sent(&a, 1), [1]);
        assert_eq!(keys_sent(&a, 2), [3]);
        assert!(a.contains(&Action::Send { child: 1, order: Order::End }), "a single-key child exits after its key");
        assert_eq!(keys_sent(&a, 0), [2, 5], "the pool child gets the small one and the merely slow one");
        c.on_msg(1, hello(), now);
        c.on_msg(1, Msg::Start { key: 1 }, now);
        c.on_msg(1, Msg::Done(done(1)), now);
        c.on_msg(1, Msg::Bye { shaders: String::new() }, now);
        c.on_exit(1, true, "exit code: 0");
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(3, Lane::Single)], "the slow one next");
        assert_eq!(keys_sent(&a, 3), [4]);
    }

    /// The giants wait until every pool key is done (`Plan::giants_last`), then get their children.
    #[test]
    fn giants_wait_for_the_pool_s_keys_when_last() {
        let now = Instant::now();
        let jobs = vec![Job::new(1, 50_000, None), Job::new(2, 1000, None), Job::new(3, 1000, None)];
        let mut c = started(Plan { giants_last: true, ..plan() }, jobs, now);
        let a = c.step(now, &ROOMY);
        assert!(spawned(&a).is_empty(), "no giant child while pool keys wait");
        assert_eq!(keys_sent(&a, 0), [2, 3]);
        c.on_msg(0, Msg::Start { key: 2 }, now);
        c.on_msg(0, Msg::Done(done(2)), now);
        assert!(spawned(&c.step(now, &ROOMY)).is_empty(), "nor while one is in flight");
        c.on_msg(0, Msg::Start { key: 3 }, now);
        c.on_msg(0, Msg::Done(done(3)), now);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single)], "then the giant");
        assert_eq!(keys_sent(&a, 1), [1]);
        let jobs = vec![Job::new(1, 50_000, None), Job::new(2, 1000, None)];
        let mut c = started(Plan { skip_giants: true, ..plan() }, jobs, now);
        assert_eq!(keys_sent(&c.step(now, &ROOMY), 0), [2]);
        assert_eq!((c.not_started(), c.stats.giants_left), (0, 1), "XCL_WARM_SKIP_GIANTS=1: the giant is left out");
    }

    /// RAM available under the floor and `Plan::low_memory` for `LOW_GRACE`: the giant child is
    /// killed and the pass stops; still low, the pool child too; the keys are left, not suspects.
    #[test]
    fn low_memory_sheds_the_giant_then_the_pool_and_stops() {
        let now = Instant::now();
        let jobs = vec![Job::new(1, 50_000, None), Job::new(2, 1000, None), Job::new(3, 1000, None)];
        let mut c = started(Plan { low_memory: Some(512 << 20), ..plan() }, jobs, now);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single)]);
        c.on_msg(1, hello(), now);
        c.on_msg(0, Msg::Start { key: 2 }, now);
        let low = Room { available: super::super::real::URGENT_FLOOR + (100 << 20), usage: Usage::default() };
        assert!(!c.step(now, &low).iter().any(|a| matches!(a, Action::Kill { .. })), "not before LOW_GRACE");
        let a = c.step(now + LOW_GRACE, &low);
        assert!(a.contains(&Action::Kill { child: 1 }) && !a.contains(&Action::Kill { child: 0 }), "the giant child first");
        assert_eq!(c.stats.memory_sheds, 1);
        assert!(c.on_exit(1, false, "killed").is_empty(), "its key is neither suspect nor bad");
        let a = c.step(now + 2 * LOW_GRACE, &low);
        assert!(a.contains(&Action::Kill { child: 0 }), "then the pool child");
        assert!(c.on_exit(0, false, "killed").is_empty());
        assert!(c.finished(), "the pass stops");
        assert_eq!((c.stats.bad, c.stats.retried), (0, 0));
    }

    /// A giant child started holds its peak and `CHILD_BASELINE` until its compile shows in the
    /// shared table, so a second step with the same table does not spend that memory again, and
    /// the "nothing runs" start is not taken while a child is pending.
    #[test]
    fn a_giant_child_not_yet_in_the_table_keeps_its_reservation() {
        let now = Instant::now();
        let plan = Plan::new(&ChildOptions { pool_children: Some(1), giant_children: Some(4), recycle_keys: 400, recycle_mb: 8192, key_timeout_s: 1800 }, 8);
        let giants = || (100..104).map(|k| Job::new(k, 90_000, None)).collect::<Vec<_>>();
        let p = giants()[0].peak();
        let floor = super::super::real::URGENT_FLOOR;
        let one = Room { available: 2 * p + CHILD_BASELINE + floor - 1, usage: Usage { running: 2, giants: 0, reserved: 0 } };
        let mut c = started(plan, giants(), now);
        assert_eq!(spawned(&c.step(now, &one)), [(1, Lane::Single)], "room for one giant and its child's baseline");
        assert!(spawned(&c.step(now, &one)).is_empty(), "the same table: the first child's peak and baseline are still held");
        c.on_msg(1, hello(), now);
        c.on_msg(1, Msg::Start { key: 100 }, now);
        let up = Room { available: one.available - CHILD_BASELINE, usage: Usage { running: 3, giants: 1, reserved: p } };
        assert!(spawned(&c.step(now, &up)).is_empty(), "its baseline landed and its compile is in the table");
        c.on_msg(1, Msg::Done(done(100)), now);
        assert_eq!(spawned(&c.step(now, &one)), [(2, Lane::Single)], "its key done: the memory is free again");
        let idle = Room { available: 1 << 30, usage: Usage::default() };
        let mut c = started(plan, giants(), now);
        assert_eq!(spawned(&c.step(now, &idle)), [(1, Lane::Single)], "nothing runs: a giant starts, its governor waits");
        assert!(spawned(&c.step(now, &idle)).is_empty(), "nothing in the table, but a giant child is pending");
    }

    #[test]
    fn a_giant_that_does_not_fit_holds_the_pool_until_it_starts() {
        let now = Instant::now();
        let mut c = started(plan(), vec![Job::new(1, 100_000, None), Job::new(2, 10, None), Job::new(3, 10, None)], now);
        let tight = Room { available: 3 << 30, usage: Usage { running: 2, giants: 0, reserved: 2 << 30 } };
        let a = c.step(now, &tight);
        assert!(spawned(&a).is_empty());
        assert!(keys_sent(&a, 0).is_empty(), "the pool waits for the giant");
        let busy = Room { usage: Usage { giants: 1, ..tight.usage }, ..tight };
        assert_eq!(keys_sent(&c.step(now, &busy), 0), [2, 3], "with a giant compiling elsewhere the pool goes on");
        let mut c = started(plan(), vec![Job::new(1, 100_000, None), Job::new(2, 10, None)], now);
        assert!(keys_sent(&c.step(now, &tight), 0).is_empty());
        let idle = Room { available: 3 << 30, usage: Usage::default() };
        let a = c.step(now, &idle);
        assert_eq!(spawned(&a), [(1, Lane::Single)], "nothing runs: the giant starts, its governor waits for memory");
        assert_eq!(keys_sent(&a, 0), [2]);
    }

    #[test]
    fn a_crash_makes_suspects_tried_alone_and_a_second_crash_bad() {
        let now = Instant::now();
        let mut c = started(plan(), small(1..4), now);
        c.step(now, &ROOMY);
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Start { key: 2 }, now);
        let out = c.on_exit(0, false, "exit code: 0xc0000005");
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], Outcome::Retry { key: 1, why } if why.contains("0xc0000005")));
        assert!(matches!(&out[1], Outcome::Retry { key: 2, .. }));
        assert_eq!(c.stats.crashed, 1);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single), (2, Lane::Single), (3, Lane::Pool)], "each suspect alone; the key never started goes back to the pool");
        assert_eq!((keys_sent(&a, 1), keys_sent(&a, 2), keys_sent(&a, 3)), (vec![1], vec![2], vec![3]));
        assert_eq!(c.stats.retried, 2);
        c.on_msg(1, hello(), now);
        let out = c.on_exit(1, false, "exit code: 0xc0000005");
        assert!(matches!(&out[..], [Outcome::Bad { key: 1, why }] if why.contains("exited")), "{out:?}");
        assert_eq!(c.stats.bad, 1);
        c.on_msg(2, hello(), now);
        assert!(matches!(c.on_msg(2, Msg::Done(done(2)), now)[..], [Outcome::Done(_)]), "the other suspect was fine alone");
        c.on_msg(2, Msg::Bye { shaders: String::new() }, now);
        assert!(c.on_exit(2, true, "exit code: 0").is_empty());
        assert_eq!(c.not_started(), 0, "nothing dropped: done, bad or with a child");
    }

    #[test]
    fn a_key_past_the_timeout_kills_its_child_as_a_hang() {
        let now = Instant::now();
        let mut c = started(plan(), small(1..3), now);
        c.step(now, &ROOMY);
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Start { key: 2 }, now + Duration::from_secs(30));
        assert!(c.step(now + Duration::from_secs(59), &ROOMY).iter().all(|a| !matches!(a, Action::Kill { .. })));
        let a = c.step(now + Duration::from_secs(60), &ROOMY);
        assert_eq!(a, [Action::Kill { child: 0 }]);
        assert!(c.step(now + Duration::from_secs(61), &ROOMY).iter().all(|a| !matches!(a, Action::Kill { .. })), "killed once");
        let out = c.on_exit(0, false, "exit code: 1");
        assert!(matches!(&out[0], Outcome::Retry { key: 1, why } if why.contains("--key-timeout-s 60")), "{out:?}");
        assert!(matches!(&out[1], Outcome::Retry { key: 2, why } if why.contains("hang of 0000000000000001")), "{out:?}");
        assert_eq!((c.stats.hung, c.stats.crashed), (1, 0));
        let later = now + Duration::from_secs(70);
        let a = c.step(later, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single), (2, Lane::Single)]);
        c.on_msg(1, hello(), later);
        c.on_msg(1, Msg::Start { key: 1 }, later);
        assert_eq!(c.step(later + Duration::from_secs(60), &ROOMY)[0], Action::Kill { child: 1 });
        let out = c.on_exit(1, false, "exit code: 1");
        assert!(matches!(&out[..], [Outcome::Hung { key: 1, why }] if why.contains("--key-timeout-s")), "a timeout alone never bad-lists: {out:?}");
        assert_eq!((c.stats.bad, c.stats.hung_keys, c.stats.hung), (0, 1, 2));
    }

    #[test]
    fn waits_for_memory_stop_the_hang_clock() {
        let now = Instant::now();
        let at = |s: u64| now + Duration::from_secs(s);
        let mut c = started(plan(), small(1..2), now);
        c.step(now, &ROOMY);
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Wait { key: 1 }, at(10));
        assert!(c.step(at(100), &ROOMY).iter().all(|a| !matches!(a, Action::Kill { .. })), "10 s run, 90 s waiting for memory");
        c.on_msg(0, Msg::Go { key: 1 }, at(100));
        assert!(c.step(at(149), &ROOMY).iter().all(|a| !matches!(a, Action::Kill { .. })), "59 s run");
        c.on_msg(0, Msg::Wait { key: 1 }, at(120));
        c.on_msg(0, Msg::Go { key: 1 }, at(130));
        assert!(c.step(at(159), &ROOMY).iter().all(|a| !matches!(a, Action::Kill { .. })), "59 s run again");
        assert_eq!(c.step(at(160), &ROOMY), [Action::Kill { child: 0 }], "60 s run");
    }

    #[test]
    fn a_child_of_another_driver_is_killed_and_its_keys_given_again() {
        let now = Instant::now();
        let mut c = started(plan(), small(1..10), now);
        let a = c.step(now, &ROOMY);
        assert_eq!(keys_sent(&a, 1), [4, 5, 6]);
        assert!(c.on_msg(1, hello(), now).is_empty(), "the same translation and driver");
        let warp = Msg::Hello { era: 5, driver: "unknown".into() };
        assert!(matches!(&c.on_msg(1, warp.clone(), now)[..], [Outcome::Mismatch { driver, .. }] if driver == "unknown"));
        let a = c.step(now, &ROOMY);
        assert!(a.contains(&Action::Kill { child: 1 }));
        assert_eq!(spawned(&a), [(2, Lane::Pool)], "another takes its place");
        assert_eq!(keys_sent(&a, 2), [7, 8, 9]);
        assert!(c.on_exit(1, false, "exit code: 1").is_empty(), "no suspects: none of its keys ran");
        assert_eq!((c.stats.mismatched, c.stats.crashed, c.stats.hung, c.stats.bad), (1, 0, 0, 0));
        assert_eq!(c.not_started(), 3, "its keys went back");
        assert!(matches!(&c.on_msg(2, Msg::Hello { era: 6, driver: "10de-2684-a".into() }, now)[..], [Outcome::Mismatch { era: 6, .. }]), "another translation era");
        assert!(c.aborted().is_none());
        let a = c.step(now, &ROOMY);
        assert!(a.contains(&Action::Kill { child: 2 }));
        assert_eq!(keys_sent(&a, 3), [4, 5, 6], "child 1's keys first");
        c.on_exit(2, false, "exit code: 1");
        c.on_msg(3, warp, now);
        assert!(c.aborted().is_some_and(|why| why.contains("3 warm-up children") && why.contains("unknown")));
        assert!(c.step(now, &ROOMY).iter().all(|a| matches!(a, Action::Kill { .. })));
    }

    #[test]
    fn children_that_never_start_end_the_pass() {
        let now = Instant::now();
        let mut c = Coordinator::new(plan());
        c.start(now);
        assert!(c.on_exit(0, false, "exit code: 2").is_empty());
        assert_eq!(spawned(&c.step(now, &ROOMY)), [(1, Lane::Pool)], "the first child is tried again");
        assert_eq!(c.step(now + Duration::from_secs(60), &ROOMY), [Action::Kill { child: 1 }], "no hello in time");
        c.on_exit(1, false, "exit code: 1");
        assert!(c.aborted().is_none());
        c.step(now, &ROOMY);
        c.on_exit(2, false, "could not start: denied");
        assert!(c.aborted().is_some_and(|why| why.contains("3 warm-up children") && why.contains("denied")));
        assert!(c.finished());
        let mut c = started(plan(), small(1..3), now);
        c.step(now, &ROOMY);
        assert!(c.on_exit(0, false, "exit code: 2").is_empty(), "before any start: its keys go back");
        assert_eq!(c.not_started(), 2);
        assert_eq!(c.stats.crashed, 1);
    }

    #[test]
    fn a_stop_tells_every_child_once_and_leaves_the_rest_not_started() {
        let now = Instant::now();
        let mut c = started(plan(), (1..10).map(|k| Job::new(k, if k == 9 { 90_000 } else { 10 }, None)).collect(), now);
        c.step(now, &ROOMY);
        c.stop();
        let a = c.step(now, &ROOMY);
        assert_eq!(a.iter().filter(|a| matches!(a, Action::Send { order: Order::Stop, .. })).count(), 3);
        assert_eq!(a.len(), 3, "nothing started, nothing given");
        assert!(c.step(now, &ROOMY).is_empty());
        for child in 0..3 {
            c.on_msg(child, hello(), now);
            c.on_msg(child, Msg::Bye { shaders: String::new() }, now);
            c.on_exit(child, true, "exit code: 0");
        }
        assert!(c.finished());
        assert_eq!(c.not_started(), 9, "every key given came back: none was started");
    }

    #[test]
    fn the_bad_list_is_kept_per_translation_and_driver() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(bad_keys(&dir, 5, "10de-2684-a").is_empty());
        note_bad(&dir, 0xab, 5, "10de-2684-a", "ran past --key-timeout-s 1800\r\nand died").unwrap();
        note_bad(&dir, 0xcd, 5, "10de 2684 b", "exited").unwrap();
        note_bad(&dir, 0xef, 6, "10de-2684-a", "exited").unwrap();
        std::fs::OpenOptions::new().append(true).open(dir.join(BAD_FILE)).unwrap().write_all(b"zz 5 10de\n12").unwrap();
        assert_eq!(bad_keys(&dir, 5, "10de-2684-a"), [0xab].into_iter().collect());
        assert_eq!(bad_keys(&dir, 5, "10de 2684 b"), [0xcd].into_iter().collect());
        assert_eq!(std::fs::read_to_string(dir.join(BAD_FILE)).unwrap().lines().next(), Some("00000000000000ab 0000000000000005 10de-2684-a ran past --key-timeout-s 1800  and died"));
        clear_bad(&dir);
        assert!(bad_keys(&dir, 5, "10de-2684-a").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_childrens_shader_lines_are_summed() {
        let a = "translation 3 from disk, 4 made; DXC 1 hits, 2 compiled in 10 thread-s; driver 5 pipelines in 7 thread-s".to_string();
        let b = "translation 10 from disk, 0 made; DXC 0 hits, 1 compiled in 2 thread-s; driver 1 pipelines in 3 thread-s".to_string();
        assert_eq!(merge_summaries(&[a.clone(), b.clone(), String::new()]), "translation 13 from disk, 4 made; DXC 1 hits, 3 compiled in 12 thread-s; driver 6 pipelines in 10 thread-s");
        assert_eq!(merge_summaries(&["took 1.5 s".into(), "took 2.25 s".into()]), "took 3.75 s");
        assert_eq!(merge_summaries(&[a.clone(), "other 1".into()]), format!("{a} | other 1"));
        assert_eq!(merge_summaries(&[]), "");
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Fate {
        Fine,
        CrashOnce,
        Crash,
        Hang,
    }

    /// Children as threads acting out `fate` for each key.
    struct FakeSpawner {
        fate: fn(u64) -> Fate,
        tries: Arc<Mutex<HashMap<u64, usize>>>,
        lanes: Arc<Mutex<Vec<Lane>>>,
    }

    struct FakeChild {
        orders: mpsc::Sender<Order>,
        killed: Arc<std::sync::atomic::AtomicBool>,
    }

    impl ChildHandle for FakeChild {
        fn send(&mut self, order: &Order) {
            let _ = self.orders.send(order.clone());
        }

        fn kill(&mut self) {
            self.killed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl Spawner for FakeSpawner {
        fn spawn(&mut self, child: u32, lane: Lane, events: mpsc::Sender<(u32, Event)>) -> Result<Box<dyn ChildHandle>, SpawnError> {
            lock(&self.lanes).push(lane);
            let (orders, inbox) = mpsc::channel::<Order>();
            let killed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (fate, tries, dead) = (self.fate, self.tries.clone(), killed.clone());
            std::thread::spawn(move || {
                let say = |msg: Msg| drop(events.send((child, Event::Line(msg.line()))));
                let exit = |success: bool, status: &str| drop(events.send((child, Event::Exited { success, status: status.into() })));
                say(Msg::Hello { era: 5, driver: "fake".into() });
                events.send((child, Event::Line("[SHADER] a child's own line".into()))).unwrap();
                for order in inbox {
                    let keys = match order {
                        Order::Keys(keys) => keys,
                        Order::End | Order::Stop => break,
                    };
                    for key in keys {
                        say(Msg::Start { key });
                        let tried = {
                            let mut tries = lock(&tries);
                            let n = tries.entry(key).or_insert(0);
                            *n += 1;
                            *n
                        };
                        match fate(key) {
                            Fate::Crash => return exit(false, "exit code: 0xc0000005"),
                            Fate::CrashOnce if tried == 1 => return exit(false, "exit code: 0xc0000005"),
                            Fate::Hang => {
                                while !dead.load(std::sync::atomic::Ordering::Relaxed) {
                                    std::thread::sleep(Duration::from_millis(5));
                                }
                                return exit(false, "exit code: 1");
                            }
                            _ => say(Msg::Done(done(key))),
                        }
                    }
                }
                say(Msg::Bye { shaders: "driver 1 pipelines".into() });
                exit(true, "exit code: 0");
            });
            Ok(Box::new(FakeChild { orders, killed }))
        }
    }

    #[derive(Default)]
    struct FakePass {
        jobs: Vec<Job>,
        heard: Vec<Outcome>,
    }

    impl Pass for FakePass {
        fn hello(&mut self, era: u64, driver: &str) -> Result<Vec<Job>, String> {
            assert_eq!((era, driver), (5, "fake"));
            Ok(std::mem::take(&mut self.jobs))
        }

        fn outcome(&mut self, outcome: &Outcome) {
            self.heard.push(outcome.clone());
        }

        fn tick(&mut self, _: &Coordinator) {}

        fn stopping(&self) -> bool {
            false
        }

        fn room(&self) -> Room {
            ROOMY
        }
    }

    #[test]
    fn a_pass_through_fake_children_accounts_for_every_key() {
        let fate = |key: u64| match key {
            3 => Fate::CrashOnce,
            5 => Fate::Crash,
            7 => Fate::Hang,
            _ => Fate::Fine,
        };
        let mut spawner = FakeSpawner { fate, tries: Default::default(), lanes: Default::default() };
        let jobs: Vec<Job> = (1..=12).map(|k| Job::new(k, if k == 11 { 50_000 } else { 10 }, None)).collect();
        let mut pass = FakePass { jobs: jobs.clone(), heard: Vec::new() };
        let plan = Plan { key_timeout: Duration::from_millis(1000), ..plan() };
        let stats = run(plan, &mut spawner, &mut pass).unwrap();
        let mut done: Vec<u64> = pass.heard.iter().filter_map(|o| match o { Outcome::Done(d) => Some(d.key), _ => None }).collect();
        let mut bad: Vec<u64> = pass.heard.iter().filter_map(|o| match o { Outcome::Bad { key, .. } => Some(*key), _ => None }).collect();
        done.sort();
        bad.sort();
        assert_eq!(done, [1, 2, 3, 4, 6, 8, 9, 10, 11, 12]);
        assert_eq!(bad, [5], "only a crash bad-lists a key");
        assert!(pass.heard.iter().any(|o| matches!(o, Outcome::Hung { key: 7, .. })), "the hang is left for the next pass");
        assert_eq!(lock(&spawner.tries).get(&3), Some(&2), "tried again alone, then fine");
        assert_eq!((stats.bad, stats.hung_keys, stats.hung >= 2, stats.crashed >= 2), (1, 1, true, true), "{stats:?}");
        assert!(stats.retried >= 3, "{stats:?}");
        assert!(lock(&spawner.lanes).contains(&Lane::Single));
        assert_eq!(pass.heard.iter().filter(|o| matches!(o, Outcome::Shaders(_))).count() + stats.crashed + stats.hung, stats.started);
    }
    #[test]
    fn a_giant_handed_back_goes_to_the_single_lane_behind_the_others_and_counts_once() {
        let now = Instant::now();
        let mut jobs = vec![Job::new(20, 50_000, None), Job::new(21, 50_000, None), Job::new(22, 50_000, None)];
        jobs.extend(small(1..4));
        let mut c = started(plan(), jobs, now);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(1, Lane::Single), (2, Lane::Single)]);
        assert_eq!(keys_sent(&a, 0), [1, 2, 3]);
        c.on_msg(0, Msg::Start { key: 1 }, now);
        c.on_msg(0, Msg::Wait { key: 1 }, now);
        assert_eq!(c.on_msg(0, Msg::Giant { key: 1, dxil_len: 3 << 20 }, now), [Outcome::Giant(1)]);
        assert_eq!(c.jobs[&1].peak(), super::super::pso_sched::peak(3 << 20), "the peak of the DXIL the child measured");
        assert_eq!((c.stats.handed, c.stats.crashed, c.stats.retried, c.stats.bad), (1, 0, 0, 0));
        assert!(c.on_msg(0, Msg::Giant { key: 1, dxil_len: 3 << 20 }, now).is_empty(), "a key no longer in flight is not handed twice");
        assert_eq!(c.not_started(), 2, "22, then the handed key");
        for child in [1, 2] {
            c.on_msg(child, hello(), now);
            c.on_msg(child, Msg::Done(done(19 + child as u64)), now);
            c.on_msg(child, Msg::Bye { shaders: String::new() }, now);
            assert!(c.on_exit(child, true, "exit code: 0").is_empty());
        }
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a), [(3, Lane::Single), (4, Lane::Single)]);
        assert_eq!((keys_sent(&a, 3), keys_sent(&a, 4)), (vec![22], vec![1]), "appended: giants in the order found");
        assert!(c.on_exit(0, false, "exit code: 0xc0000005").iter().all(|o| !matches!(o, Outcome::Retry { key: 1, .. })), "no longer in flight in the pool child");
        c.on_msg(4, hello(), now);
        c.on_msg(4, Msg::Start { key: 1 }, now);
        assert_eq!(c.on_msg(4, Msg::Done(done(1)), now), [Outcome::Done(done(1))], "done once, from its single-key child");
        assert_eq!(c.stats.handed, 1);
    }

    #[test]
    fn the_pool_keeps_flowing_while_three_giants_run_in_single_key_children() {
        let now = Instant::now();
        let plan = Plan::new(&ChildOptions { pool_children: Some(2), giant_children: Some(3), recycle_keys: 400, recycle_mb: 8192, key_timeout_s: 1800 }, 8);
        let mut jobs: Vec<Job> = (100..104).map(|k| Job::new(k, 90_000, None)).collect();
        jobs.extend(small(1..60));
        let mut c = started(plan, jobs, now);
        let a = c.step(now, &ROOMY);
        assert_eq!(spawned(&a).iter().filter(|(_, lane)| *lane == Lane::Single).count(), 3);
        let first = keys_sent(&a, 0);
        assert_eq!(first.len(), 5);
        let capped = Room { available: 6 << 30, usage: Usage { running: 3, giants: 3, reserved: 9 << 30 } };
        for key in first {
            c.on_msg(0, Msg::Done(done(key)), now);
        }
        let a = c.step(now, &capped);
        assert!(spawned(&a).is_empty(), "the fourth giant waits: three single-key children run (--giant-children 3), and no memory");
        assert_eq!(keys_sent(&a, 0).len(), 5, "the pool is fed while three giants run");
        let mut c = started(plan, vec![Job::new(100, 90_000, None), Job::new(1, 10, None), Job::new(2, 10, None)], now);
        let hopeless = Room { available: 1 << 30, usage: Usage { running: 4, giants: 0, reserved: 1 << 30 } };
        assert_eq!(keys_sent(&c.step(now, &hopeless), 0), [1, 2], "draining the pool would not make room: it is not paused");
    }

    #[test]
    fn a_giant_waiting_for_memory_pauses_the_pool_after_five_minutes_and_ends_it_after_ten() {
        let now = Instant::now();
        let minutes = |m: u64| now + Duration::from_secs(60 * m);
        let plan = Plan::new(&ChildOptions { pool_children: Some(1), giant_children: Some(3), recycle_keys: 400, recycle_mb: 64 << 10, key_timeout_s: 1800 }, 4);
        let mut jobs = vec![Job::new(100, 90_000, None)];
        jobs.extend(small(1..60));
        let tight = Room { available: 1 << 30, usage: Usage { running: 4, giants: 0, reserved: 1 << 30 } };
        let mut held = started(plan, jobs.clone(), now);
        held.on_msg(0, Msg::Mem { commit: 32 << 30, peak_working_set: 0, peak_commit: 0 }, now);
        assert!(keys_sent(&held.step(now, &tight), 0).is_empty(), "the pool children's commit would make room: paused");
        let mut c = started(plan, jobs, now);
        let fed = |c: &mut Coordinator, at: Instant| {
            let a = c.step(at, &tight);
            assert!(spawned(&a).is_empty());
            let keys = keys_sent(&a, 0);
            for &key in &keys {
                c.on_msg(0, Msg::Done(done(key)), at);
            }
            (keys.len(), a.contains(&Action::Send { child: 0, order: Order::End }))
        };
        assert_eq!(fed(&mut c, now), (5, false), "draining the pool would not make room: fed");
        assert_eq!(fed(&mut c, minutes(4)), (5, false));
        assert_eq!(fed(&mut c, minutes(5)), (0, false), "waited five minutes: paused until a giant starts");
        assert_eq!(fed(&mut c, minutes(9)), (0, false));
        assert_eq!(fed(&mut c, minutes(10)), (0, true), "ten minutes: the pool child ends, its memory freed");
        assert_eq!(c.stats.recycled, 1);
        assert_eq!(fed(&mut c, minutes(11)), (0, false), "told once");
        c.on_msg(0, Msg::Bye { shaders: String::new() }, minutes(11));
        assert!(c.on_exit(0, true, "exit code: 0").is_empty());
        let a = c.step(minutes(11), &Room { available: 1 << 30, usage: Usage::default() });
        assert_eq!(spawned(&a), [(1, Lane::Single), (2, Lane::Pool)], "nothing runs: the giant starts, and the pool again");
    }

    /// Simulated time through the real `Coordinator`: the PC's mix of 10,321 keys, about 350
    /// predicted giants (most among the first 1000, as the target order puts measured waits
    /// first) and 3% of the rest giants the console bytes do not predict, each 15 s of DXC then
    /// 45–165 s of driver compile with at most `giants` driver compiles of giants at once across
    /// processes (a count: RAM is not modelled, nor the children's memory outside compiles); every other key a driver compile of the live game's
    /// times (log-normal, 16.5 s median, 47 s p90, 1–600 s). `handoff` false: a pool worker
    /// holding a giant waits for that room (the behaviour before `--leave-giants`); true: the pool
    /// child hands it back after its DXC. `plan`: the children. Children are up 3 s after their
    /// spawn, CPU contention is not modelled, and neither is the per-process "a waiting giant holds
    /// back smaller compiles" rule (it holds only when memory is short).
    fn simulate(handoff: bool, giants: usize, plan: Plan) -> SimRun {
        #[derive(Clone, Copy)]
        enum Work {
            Normal(f64),
            Giant { dxc: f64, driver: f64 },
        }
        #[derive(Clone, Copy)]
        enum Stage {
            Run(f64),
            Dxc(f64),
            Waiting(f64),
            Driver(f64),
        }
        struct Sim {
            lane: Lane,
            threads: usize,
            ready: f64,
            hello: bool,
            queue: VecDeque<u64>,
            ended: bool,
            work: Vec<(u64, Stage)>,
        }
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut kinds, mut jobs) = (HashMap::new(), Vec::new());
        for key in 1..=10_321u64 {
            let predicted = random() < if key <= 1000 { 0.3 } else { 0.006 };
            let unpredicted = !predicted && random() < 0.03;
            let u = random();
            kinds.insert(key, if predicted || unpredicted {
                Work::Giant { dxc: 15.0, driver: 45.0 + 120.0 * u }
            } else {
                let normal = (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * random()).cos();
                Work::Normal((16.5 * (normal * (47.0f64 / 16.5).ln() / 1.2816).exp()).clamp(1.0, 600.0))
            });
            jobs.push(Job::new(key, if predicted { GIANT_GUEST_BYTES } else { 26_000 }, None));
        }
        let base = Instant::now();
        let at = |t: f64| base + Duration::from_secs_f64(t);
        let mut c = Coordinator::new(plan);
        let mut jobs = Some(jobs);
        let mut sims: BTreeMap<u32, Sim> = BTreeMap::new();
        let (mut handed, mut waiters, mut cap) = (HashSet::new(), VecDeque::<(u32, u64)>::new(), 0usize);
        let (mut t, mut finished) = (0.0f64, 0usize);
        let mut run = SimRun::default();
        let mut actions = c.start(at(0.0));
        loop {
            for action in actions.drain(..) {
                match action {
                    Action::Spawn { child, lane } => drop(sims.insert(child, Sim { lane, threads: plan.threads(lane), ready: t + 3.0, hello: false, queue: VecDeque::new(), ended: false, work: Vec::new() })),
                    Action::Send { child, order: Order::Keys(keys) } => sims.get_mut(&child).unwrap().queue.extend(keys),
                    Action::Send { child, .. } => sims.get_mut(&child).unwrap().ended = true,
                    Action::Kill { child } => panic!("child {child} killed at {t:.0} s"),
                }
            }
            for id in sims.keys().copied().collect::<Vec<_>>() {
                let s = sims.get_mut(&id).unwrap();
                if !s.hello && s.ready <= t {
                    s.hello = true;
                    c.on_msg(id, Msg::Hello { era: 5, driver: "sim".into() }, at(t));
                    if let Some(jobs) = jobs.take() {
                        c.set_jobs(jobs);
                    }
                }
                if !s.hello {
                    continue;
                }
                while s.work.len() < s.threads {
                    let Some(key) = s.queue.pop_front() else { break };
                    c.on_msg(id, Msg::Start { key }, at(t));
                    let stage = match kinds[&key] {
                        Work::Normal(d) => Stage::Run(t + d),
                        Work::Giant { driver, .. } if handed.contains(&key) => {
                            waiters.push_back((id, key));
                            c.on_msg(id, Msg::Wait { key }, at(t));
                            Stage::Waiting(driver)
                        }
                        Work::Giant { dxc, .. } => Stage::Dxc(t + dxc),
                    };
                    s.work.push((key, stage));
                }
                if s.ended && s.queue.is_empty() && s.work.is_empty() {
                    sims.remove(&id);
                    c.on_msg(id, Msg::Bye { shaders: String::new() }, at(t));
                    assert!(c.on_exit(id, true, "exit code: 0").is_empty());
                }
            }
            while cap < giants {
                let Some((id, key)) = waiters.pop_front() else { break };
                let work = sims.get_mut(&id).unwrap().work.iter_mut().find(|(k, _)| *k == key).unwrap();
                let Stage::Waiting(driver) = work.1 else { unreachable!() };
                work.1 = Stage::Driver(t + driver);
                cap += 1;
                c.on_msg(id, Msg::Go { key }, at(t));
            }
            if c.finished() {
                break;
            }
            let running = sims.values().map(|s| s.work.iter().filter(|(_, stage)| !matches!(stage, Stage::Waiting(_))).count()).sum();
            actions = c.step(at(t), &Room { available: 64 << 30, usage: Usage { running, giants: cap, reserved: 0 } });
            if !actions.is_empty() {
                continue;
            }
            let ends = |s: &Sim| s.work.iter().filter_map(|(_, stage)| match *stage {
                Stage::Run(end) | Stage::Dxc(end) | Stage::Driver(end) => Some(end),
                Stage::Waiting(_) => None,
            }).chain((!s.hello).then_some(s.ready)).collect::<Vec<_>>();
            let before = t;
            t = sims.values().flat_map(ends).fold(f64::INFINITY, f64::min);
            assert!(t.is_finite(), "the pass is stuck");
            let pool_waiting = sims.values().filter(|s| s.lane == Lane::Pool).flat_map(|s| &s.work).filter(|(_, stage)| matches!(stage, Stage::Waiting(_))).count();
            run.pool_waiting_s += (t - before) * pool_waiting as f64;
            for (&id, s) in sims.iter_mut() {
                let mut i = 0;
                while i < s.work.len() {
                    let (key, stage) = s.work[i];
                    match stage {
                        Stage::Run(end) | Stage::Driver(end) if end <= t => {
                            cap -= matches!(stage, Stage::Driver(_)) as usize;
                            s.work.swap_remove(i);
                            c.on_msg(id, Msg::Done(done(key)), at(t));
                            finished += 1;
                            run.first_hour += (t <= 3600.0) as usize;
                            if matches!(kinds[&key], Work::Normal(_)) {
                                run.others_done_s = t;
                            }
                            continue;
                        }
                        Stage::Dxc(end) if end <= t => {
                            if handoff && s.lane == Lane::Pool {
                                s.work.swap_remove(i);
                                handed.insert(key);
                                c.on_msg(id, Msg::Giant { key, dxil_len: super::super::pso_sched::GIANT_DXIL }, at(t));
                                continue;
                            }
                            let Work::Giant { driver, .. } = kinds[&key] else { unreachable!() };
                            s.work[i].1 = Stage::Waiting(driver);
                            waiters.push_back((id, key));
                            c.on_msg(id, Msg::Wait { key }, at(t));
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
        }
        SimRun { keys: finished, wall_s: t, ..run }
    }

    #[derive(Default)]
    struct SimRun {
        keys: usize,
        wall_s: f64,
        /// when the last key that is not a giant was done
        others_done_s: f64,
        /// keys done in the first hour
        first_hour: usize,
        /// pool worker-seconds spent waiting for room for a giant
        pool_waiting_s: f64,
    }

    impl std::fmt::Display for SimRun {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{} keys in {:.0} s ({:.2} keys/s); {} keys in the first hour; the other keys done at {:.0} s; pool workers waited {:.0} worker-s for room for a giant",
                self.keys, self.wall_s, self.keys as f64 / self.wall_s, self.first_hour, self.others_done_s, self.pool_waiting_s)
        }
    }

    #[test]
    fn simulated_pass_giants_handed_to_single_key_children_or_blocking_the_pool() {
        let took = Instant::now();
        let capped = Plan::new(&ChildOptions { giant_children: Some(3), ..ChildOptions::default() }, 24);
        let old = simulate(false, 3, capped);
        let new = simulate(true, 3, capped);
        // (RAM is not modelled: `giants` is a count of giant driver compiles at once, and the
        // children's memory outside their compiles is left out)
        let wide = simulate(true, 13, Plan::new(&ChildOptions::default(), 24));
        println!("simulated, giants block pool workers on the cap of 3: {old}");
        println!("simulated, pool children hand giants to 3 single-key children: {new}");
        println!("simulated, 13 giants at once, 12 single-key children; RAM not modelled: {wide}");
        println!("(the simulations took {:.2} s)", took.elapsed().as_secs_f64());
        assert_eq!((old.keys, new.keys, wide.keys), (10_321, 10_321, 10_321));
        assert_eq!(new.pool_waiting_s, 0.0, "no pool worker waits for room for a giant");
        assert!(old.pool_waiting_s > 0.0);
        assert!(new.others_done_s < old.others_done_s && new.first_hour > old.first_hour);
        assert!(wide.wall_s < new.wall_s, "13 giants at once and 12 single-key children against 3 and 3, RAM not modelled");
    }
}
