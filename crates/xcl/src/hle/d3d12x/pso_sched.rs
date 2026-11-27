//! The policy of background pipeline builds, apart from the threads and the driver that carry it
//! out (`pso_async`, `real::DRIVER_COMPILES`): what a driver compile may start within the memory
//! left, and which build a background worker takes next. Nothing here asks the system anything,
//! so its tests run on any host.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// DXIL from which a program is a giant (a warm-up builds giants in single-key children,
/// `warm_children`); giants, like every compile, start when their expected peak fits
pub const GIANT_DXIL: usize = 1 << 20;
pub const GIANTS_AT_ONCE: usize = 3;

/// How far back `Trend` looks for the fastest fall of a memory figure, in seconds: a burst the
/// title loaded is let go of soon after (play-87n: a 30 s window held its whole loading phase).
pub const TREND_WINDOW: f64 = 10.0;
/// The shortest span `Trend` measures a fall over, in seconds.
pub const TREND_SPAN: f64 = 2.0;
/// A fall of the memory available, in bytes a second, at which no new background giant starts.
pub const FAST_FALL: u64 = 128 << 20;
/// How long after live demand (a compile a skipped draw or a bind waits for asked for a driver
/// slot, or one the title waited for had to wait) no new background giant starts: a giant's slot
/// is held for minutes, and a title that left its menus streams in bursts (play-88: one started
/// in a lull held 8.3 GB of estimate for 164 s while the world load's compiles waited behind it).
pub const LIVE_QUIET: std::time::Duration = std::time::Duration::from_secs(30);
/// The fall of the memory available or of the commit room, in bytes a second, from which a
/// compile past its expected time is not taken as settled (`Load::settled`): something is still
/// growing, and a giant may be. Giants ran 0.8 to 3.2 times `compile_secs` and peaked up to 2.9
/// times their estimate (play-89a: 1,592,684 B, +3.9 GB of commit over 76 s, about 51 MB/s);
/// under this, a 5 s live compile sees at most 160 MB of growth, inside the 1 GB floor.
pub const SETTLE_FALL: u64 = 32 << 20;

/// A compile a bind waits for counts the compiles past their expected time as the memory available
/// shows them whatever the trend (`XCL_SETTLE_OVERDUE=0`: only while it falls under `SETTLE_FALL`).
/// Their memory is resident and already missing from the memory available; subtracting their
/// estimates again counted it twice exactly while the title loaded: play-95 held four waited-for
/// compiles 250 to 281 s with 5,059 MiB available falling ~70 MB/s and all 8,137 MiB of the
/// reservation overdue. The RAM check keeps the 1 GB floor; the commit check keeps the trend rule.
fn settle_overdue() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_SETTLE_OVERDUE").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: compiles past their expected time keep their estimates while the memory falls (XCL_SETTLE_OVERDUE=0)");
        }
        on
    })
}
/// The most of the memory available and of the commit room a background compile keeps free for
/// the title's growth (`growth`): a loading burst holds background work back, never for good.
pub const GROWTH_CAP: u64 = 2 << 30;

/// Seconds a driver compile of `dxil_len` bytes of DXIL is expected to run beside the title: the
/// span over which the title's growth is held back for it. Giants ran 27 s to 46 s and more beside
/// the loading title (play-86, play-86n live stalls); their peak comes late.
pub fn compile_secs(dxil_len: usize) -> u64 {
    if dxil_len < GIANT_DXIL {
        5
    } else {
        (30 + 10 * dxil_len as u64 / (1 << 20)).min(90)
    }
}

/// Recent samples of a memory figure (the memory available, the commit room), for how fast the
/// title is taking it.
pub struct Trend {
    samples: VecDeque<(f64, u64)>,
}

impl Trend {
    pub const fn new() -> Self {
        Self { samples: VecDeque::new() }
    }

    /// The figure is `value` at `at` seconds (samples under half a second apart are skipped).
    pub fn note(&mut self, at: f64, value: u64) {
        if self.samples.back().is_some_and(|&(t, _)| at - t < 0.5) {
            return;
        }
        self.samples.push_back((at, value));
        while self.samples.front().is_some_and(|&(t, _)| at - t > TREND_WINDOW) {
            self.samples.pop_front();
        }
    }

    /// The fastest fall of the figure over any span of `TREND_SPAN` or more within the last
    /// `TREND_WINDOW`, in bytes a second (0: it did not fall).
    pub fn fall_rate(&self) -> u64 {
        let mut fastest = 0;
        for (i, &(t0, v0)) in self.samples.iter().enumerate() {
            if let Some(&(t1, v1)) = self.samples.iter().skip(i + 1).find(|&&(t, _)| t - t0 >= TREND_SPAN) {
                fastest = fastest.max((v0.saturating_sub(v1) as f64 / (t1 - t0)) as u64);
            }
        }
        fastest
    }
}

/// The memory available and the commit room a background compile of `dxil_len` bytes of DXIL
/// keeps free on top of its floors for the title's growth while it runs: each figure's fall rate
/// × its expected duration, at most `GROWTH_CAP`.
pub fn growth(room: &Room, dxil_len: usize) -> (u64, u64) {
    let secs = compile_secs(dxil_len);
    (room.fall.saturating_mul(secs).min(GROWTH_CAP), room.commit_fall.saturating_mul(secs).min(GROWTH_CAP))
}

/// An estimated driver compile budget, calibrated with isolated cold compiles on the NVIDIA
/// Windows driver (GeForce RTX 3080). Growth is relative to the pre-compile sample, in MiB.
///
/// | DXIL bytes | Peak private commit | Peak working set | Available RAM drop |
/// |-----------:|--------------------:|-----------------:|-------------------:|
/// |    533,812 |                 307 |                — |                  — |
/// |    660,536 |                 425 |                — |                  — |
/// |  1,208,312 |                 947 |              953 |                  — |
/// |  1,588,808 |               1,067 |            1,059 |                  — |
/// |  1,717,176 |               1,067 |            1,067 |              1,104 |
/// |  1,727,332 |               1,775 |            1,777 |                  — |
/// |  1,945,712 |               1,217 |            1,218 |              1,351 |
/// |  3,240,760 |               3,034 |            3,021 |                  — |
/// |  3,241,524 |               2,897 |            2,874 |              3,047 |
/// |  3,515,708 |               4,468 |            4,432 |              4,282 |
///
/// The giant curve interpolates between the highest observed costs with 10% headroom below
/// 1.5 MiB of DXIL and 25% above it. Nearby lower samples do not pull the budget down.
/// Between the largest measured point and 4 MiB the curve returns to the old conservative
/// bound. At 4 MiB and up it keeps the unmeasured-size guard.
pub fn peak(dxil_len: usize) -> u64 {
    peak_at(dxil_len, SMALL_PEAK.load(Ordering::Relaxed))
}

/// `peak` with `small` bytes of memory for each byte of DXIL under the giant line.
pub fn peak_at(dxil_len: usize, small: u64) -> u64 {
    let len = dxil_len as u64;
    const MIB: u64 = 1 << 20;
    const SMALL_CAP: u64 = 2 << 30;
    const GIANT_BUDGET_MIB: &[(u64, u64)] = &[
        (MIB, 911),
        (1_208_312, 1049),
        (1_588_808, 1334),
        (1_727_332, 2222),
        (1_945_712, 2222),
        (3_240_760, 3809),
        (3_515_708, 5585),
        (4 * MIB, 9600),
    ];
    if len >= 4 * MIB {
        len * 3380
    } else if dxil_len < GIANT_DXIL {
        (len * small).clamp(128 << 20, SMALL_CAP)
    } else {
        let range = GIANT_BUDGET_MIB.windows(2).find(|p| len <= p[1].0).unwrap();
        let (from, to) = (range[0], range[1]);
        let budget = from.1 * MIB + ((len - from.0) * (to.1 - from.1) * MIB).div_ceil(to.0 - from.0);
        budget.max((MIB * small).min(SMALL_CAP))
    }
}

/// What a driver compile is admitted and reserved at: `peak`, with a cold giant's times `factor`
/// up to the unmeasured-size guard (`dxil_len` × 3,380), and one whose driver blob is kept on disk
/// (`kept`) at the small-program rate. In game, cold giants peaked at 0.8 to 4.4 times the curve
/// (play-89: 1,127,196 B 2.7×, 1,509,056 B 2.8×, 1,415,460 B 4.4×; play-89a: 1,592,684 B 2.9×;
/// play-87: 1.0 to 1.1×), each figure a process-wide commit delta that other compiles add to;
/// 106 loads of a kept blob peaked at 0.03 to 0.18 times it.
pub fn cost(dxil_len: usize, kept: bool, factor: u64) -> u64 {
    let small = SMALL_PEAK.load(Ordering::Relaxed);
    let len = dxil_len as u64;
    if kept && dxil_len >= GIANT_DXIL {
        return (len * small).clamp(128 << 20, 2 << 30);
    }
    let curve = peak_at(dxil_len, small);
    if dxil_len < GIANT_DXIL || dxil_len >= 4 << 20 {
        return curve;
    }
    curve.saturating_mul(factor).min(len * 3380).max(curve)
}

/// Memory expected for each byte of DXIL of a driver compile under the giant line.
pub const SMALL_PEAK_PER_BYTE: u64 = 800;
static SMALL_PEAK: AtomicU64 = AtomicU64::new(SMALL_PEAK_PER_BYTE);

/// The memory expected for each byte of DXIL under the giant line is `per_byte` from now on
/// (256 to 4096: a PC whose driver takes more or less than the one measured).
pub fn set_small_peak(per_byte: u64) {
    SMALL_PEAK.store(per_byte.clamp(256, 4096), Ordering::Relaxed);
}

/// The driver compiles running or waiting now.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub running: usize,
    /// giants among `running`, the ones a bind waits for too
    pub giants: usize,
    /// compiles a bind waits for among `running`
    pub urgent: usize,
    /// other compiles a bind waits for, waiting for memory
    pub starving: usize,
    /// the expected peaks of those running
    pub reserved: u64,
    /// the smallest expected peak of the giants a blocking warm-up waits to start (0: none)
    pub giant_waiting: u64,
    /// live demand within `LIVE_QUIET`
    pub live_recent: bool,
    /// the expected peaks among `reserved` of compiles running past their expected time
    /// (`compile_secs`): what they take shows in the memory available by now. The memory check of
    /// a waited-for compile and of a small background one leaves them out whatever the trend
    /// (`settle_overdue`, `settle_background`); the commit check of a waited-for compile only
    /// while the memory does not fall `SETTLE_FALL` (`Room::trend_fall`)
    pub settled: u64,
    /// the reservations among `reserved` of cold giants (`Ask::small_lane` leaves them out, and the
    /// settled ones too while the memory does not fall `SETTLE_FALL`: play-92 held every small live
    /// compile for minutes with 12,514 of 12,902 MiB of reservation settled and 2,737 MiB available)
    pub giant_reserved: u64,
}

/// A driver compile asking to start.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    pub dxil_len: usize,
    /// a bind waits for it
    pub wanted: bool,
    /// background compiles at once at most
    pub most: usize,
    /// the process's private memory to stay within, when given
    pub cap: Option<u64>,
    /// the memory it is admitted and reserved at (`cost`)
    pub cost: u64,
    /// (live lanes) a compile a bind or a skipped draw waits for that is not a cold giant: it
    /// counts the cold giants running as the memory available shows them, never by their
    /// reservations (play-89: one live giant's 7,226 MiB and the floor took all 8.2 GB available,
    /// and the title's small compiles waited 228 to 403 s behind it)
    pub small_lane: bool,
    /// (in a game) a cold giant: it starts only when its estimate, the title's growth
    /// (`Room::title_fall`) and the compiles' growth fit the memory available less the
    /// reservations held and 1 GB, and never beside another cold giant (`hard_giants`)
    pub hard_giant: bool,
    /// (a giant helper's compile, `real::giant_helper`) it runs in a child process held to `cost` by
    /// its job object and ended when the memory available falls under the helper's floor, so it
    /// keeps no memory for the title's growth or the compiles' rise (`helper_skips_growth`); the
    /// commit check, the one-giant rule and the floor stay
    pub helper: bool,
}

/// The memory a compile starts in.
#[derive(Clone, Copy, Debug)]
pub struct Room {
    /// physical memory available now
    pub available: u64,
    /// what a background compile leaves available
    pub floor: u64,
    /// what a compile a bind waits for leaves available
    pub urgent_floor: u64,
    /// the process's private memory now (read with `Ask::cap`)
    pub commit: u64,
    /// the system's commit limit less its commit charge
    pub commit_room: u64,
    /// how fast the memory available is falling now (`Trend::fall_rate`), bytes a second
    pub fall: u64,
    /// how fast the commit room is falling now, bytes a second
    pub commit_fall: u64,
    /// the faster of the two falls, measured even when no growth headroom is kept (`SETTLE_FALL`)
    pub trend_fall: u64,
    /// how fast the memory available is falling now, kept for a cold giant even when `fall` is
    /// not (`Ask::hard_giant`), bytes a second
    pub title_fall: u64,
    /// how fast the process heap less XCL's Rust heap is rising now: the driver compiles running
    /// still taking memory (`compile_rise`), bytes a second
    pub compile_rise: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// it starts as a compile a bind waits for, outside the background limits
    Urgent,
    /// it starts as a background compile
    Background,
    /// it does not start now
    Hold,
}

pub fn fits_on_machine(peak: u64, floor: u64, physical_total: u64) -> bool {
    physical_total == 0 || peak <= physical_total.saturating_sub(floor)
}

/// Whether a driver compile starts now (the user's rule, 2026-11-26 15:22Z: a compile takes the
/// memory it needs, giants too, with no count of them at once). The memory available, less the
/// full expected peaks of the compiles running (they may still be growing) and, for background
/// work, what the title is expected to take while this one runs (`growth`: its peak comes late,
/// and a loading title keeps growing), must hold its own expected peak and the floor; the commit
/// headroom, less the same peaks and the title's commit growth, its peak and the urgent floor (a
/// process out of commit fails its allocations). A compile a bind or a title thread waits for
/// (`Ask::wanted`) keeps no growth headroom: the title is stalled on it, and the compiles running
/// past their expected time count as the memory available shows them, not by their expected peaks
/// (`Load::settled`: play-88b's background giant kept 8.6 GB of estimate for 225 s, and the plain
/// rule held every compile the world load waited for while 7 to 10 GB were available). That holds
/// whatever the trend for the memory check (`settle_overdue`: the memory available already shows
/// them, so they are never taken off twice) and for a small background compile's
/// (`settle_background`); for a waited-for compile's commit check, only while neither the memory
/// available nor the commit room falls `SETTLE_FALL` or faster. No new background giant starts
/// while the memory available or the commit room falls `FAST_FALL` or faster, or within
/// `LIVE_QUIET` of live demand (`Load::live_recent`). A background one stays within
/// `most` at once and the private-memory cap, and holds back while a waited-for compile waits
/// for memory. While a blocking warm-up's giant waits, a smaller background compile starts
/// only if it leaves the giant its peak, unless the giant could not fit even with every running
/// compile gone: small compiles finishing and starting again would otherwise keep the memory
/// below a giant's peak for the whole pass.
pub fn admit(ask: &Ask, load: &Load, room: &Room) -> Verdict {
    admit_why(ask, load, room).0
}

/// `admit`, and why a compile is held (empty when it starts).
pub fn admit_why(ask: &Ask, load: &Load, room: &Room) -> (Verdict, &'static str) {
    let cost = ask.cost;
    let (growth, commit_growth) = if ask.wanted { (0, 0) } else { growth(room, ask.dxil_len) };
    let growth = if ask.helper { 0 } else { growth };
    let small_background = !ask.wanted && ask.dxil_len < GIANT_DXIL && settle_background();
    let small_lane_background = !ask.wanted && ask.dxil_len < GIANT_DXIL && background_small_lane();
    let held_back = |settle: bool| {
        if !ask.wanted {
            let giants = if small_lane_background { load.giant_reserved } else { 0 };
            let settled = if settle && small_background { load.settled } else { 0 };
            load.reserved.saturating_sub(giants.max(settled))
        } else if ask.small_lane {
            load.reserved.saturating_sub(if settle { load.giant_reserved.max(load.settled) } else { load.giant_reserved })
        } else if settle {
            load.reserved.saturating_sub(load.settled)
        } else {
            load.reserved
        }
    };
    let steady = room.trend_fall < SETTLE_FALL;
    let reserved = held_back(steady || settle_overdue() || small_background);
    let commit_reserved = if ask.wanted { held_back(steady) } else { load.reserved };
    let rise = if ask.helper { 0 } else { compile_rise(room, ask.dxil_len) };
    let spare = room.available.saturating_sub(reserved).saturating_sub(growth).saturating_sub(rise);
    if room.commit_room.saturating_sub(commit_reserved) < cost.saturating_add(room.urgent_floor).saturating_add(commit_growth) {
        return (Verdict::Hold, "commit room: the commit limit less the commit charge, the reservations and the title's commit growth is under its estimate and 1 GB");
    }
    if ask.hard_giant {
        if load.giant_reserved != 0 {
            return (Verdict::Hold, "a cold giant runs: one at a time in the game");
        }
        let title = if ask.helper { 0 } else { room.title_fall.saturating_mul(compile_secs(ask.dxil_len)).min(GROWTH_CAP) };
        if spare.saturating_sub(if ask.wanted { title } else { 0 }) < cost.saturating_add(room.urgent_floor) {
            return (Verdict::Hold, "a cold giant: the memory available less the reservations held, the title's and the compiles' growth is under its estimate and 1 GB");
        }
    }
    if ask.wanted {
        // (a live cold giant with no other cold giant running falls back to the curve when its
        // corrected estimate does not fit: play-91b's two waited 250 s with 7.5 GB available;
        // never in a game's hard rule)
        let cost = if !ask.small_lane && !ask.hard_giant && load.giant_reserved == 0 && spare < cost + room.urgent_floor { cost.min(peak(ask.dxil_len)) } else { cost };
        return if spare >= cost + room.urgent_floor {
            (Verdict::Urgent, "")
        } else {
            (Verdict::Hold, "memory: the memory available less the reservations held is under its estimate and 1 GB")
        };
    }
    let floor = room.floor.max(room.urgent_floor);
    let giant = ask.dxil_len >= GIANT_DXIL;
    let why = if load.starving > 0 {
        "a compile a bind waits for waits for memory first"
    } else if giant && (room.fall >= FAST_FALL || room.commit_fall >= FAST_FALL) {
        "a background giant while the title's memory falls fast"
    } else if giant && load.live_recent {
        "a background giant within 30 s of live demand"
    } else if load.running.saturating_sub(load.urgent) >= ask.most {
        "every background slot is taken"
    } else if spare < floor + cost {
        "memory: the memory available less the reservations held and the title's growth is under its estimate and the floor"
    } else if !giant && load.giant_waiting != 0 && room.available.saturating_add(load.reserved) >= floor + load.giant_waiting && spare < floor + cost + load.giant_waiting {
        "room kept for a warm-up's giant"
    } else if ask.cap.is_some_and(|c| room.commit + load.reserved + cost > c) {
        "the private-memory cap"
    } else {
        return (Verdict::Background, "");
    };
    (Verdict::Hold, why)
}

/// Whether `admit` holds a background compile of `dxil_len` bytes at `cost` whatever the commit room
/// and the title's growth are: its verdict on the memory sample alone, with unlimited commit room
/// and nothing falling. The commit room, the growth and a fast fall only hold more, so what is
/// held here is held there. The backlog asks this of every parked pipeline under its lock before
/// the full verdict, which reads the system's memory each time (`real::ParkedRoom`): play-95 had
/// 3,059 parked, and the title's binds queued on that lock until its watchdog ended the run.
pub fn held_on_sample(dxil_len: usize, cost: u64, most: usize, load: &Load, available: u64, floor: u64, urgent_floor: u64) -> bool {
    let room = Room { available, floor, urgent_floor, commit: 0, commit_room: u64::MAX, fall: 0, commit_fall: 0, trend_fall: 0, title_fall: 0, compile_rise: 0 };
    admit(&Ask { dxil_len, wanted: false, most, cap: None, cost, small_lane: false, hard_giant: false, helper: false }, load, &room) == Verdict::Hold
}

/// A background compile under `GIANT_DXIL` counts the compiles past their expected time as the
/// memory available shows them (`XCL_SETTLE_BACKGROUND=0`: by their whole estimates). Their
/// memory is resident already; the title's growth is still kept (`growth`). play-95 ran no
/// dispatch in 310 s: 2 KB loading passes never built while 2 giants held 8,137 MiB of overdue
/// reservation against ~5 GB available, and 3,059 builds waited for memory.
fn settle_background() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_SETTLE_BACKGROUND").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: small background compiles count compiles past their expected time by their whole estimates (XCL_SETTLE_BACKGROUND=0)");
        }
        on
    })
}

/// A giant helper's compile (`Ask::helper`) keeps no memory for the title's growth or the measured
/// rise of the compiles running (`XCL_HELPER_SKIPS_GROWTH=0`: it keeps both, as a compile in this
/// process does): its child is held to its reservation and ended when the memory available falls
/// under the helper's floor, then tried again at its measured peak. play-101's FFX_FSR4_PASS_14
/// and 16 (est 7,810 MiB in the helper) were held 500 s with ~10 GB available and ~2 GB reserved,
/// with the compile heap rising 57 to 258 MB/s against their ~minutes-long expected time.
pub fn helper_skips_growth() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_HELPER_SKIPS_GROWTH").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: giant helper compiles keep memory for the title's growth and the compiles' rise (XCL_HELPER_SKIPS_GROWTH=0)");
        }
        on
    })
}

/// A small background compile (under `GIANT_DXIL`) leaves the cold giants' reservations out of its
/// memory check, as a small live one does (`Ask::small_lane`), and keeps the title's growth, the
/// floor and its slot count (`XCL_BACKGROUND_SMALL_LANE=0`: they hold it back). A cold giant is
/// reserved at `GIANT_FACTOR` times the curve for its whole expected time: play-97 held 1,786
/// background compiles of 9 to 612 KB with 5 to 9.5 GB available behind 5.4 to 7.5 GB of giants'
/// reservation, about 850 builds pending with the oldest 257 s old.
fn background_small_lane() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_BACKGROUND_SMALL_LANE").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: small background compiles wait on the cold giants' reservations (XCL_BACKGROUND_SMALL_LANE=0)");
        }
        on
    })
}

/// A compile a bind waits for would not fit even if every running compile ended and gave back its
/// whole estimate: `cost` and the floor exceed the memory available and the reservations.
/// play-97's FFX_FSR4_PASS_14, 16 and 18 (4,961, 4,959 and 7,698 KB of the title's own DXIL, est
/// 16,375 to 25,411 MiB) held three workers 310 s with 2.6 to 5 GB available and 7.5 to 11 GB
/// reserved.
pub fn never_fits_now(cost: u64, available: u64, reserved: u64, urgent_floor: u64) -> bool {
    cost.saturating_add(urgent_floor) > available.saturating_add(reserved)
}

/// A compile a bind waits for that `never_fits_now` for this long without a break is refused: a
/// loading burst may take the memory for a moment (an idle governor's waited-for compile must
/// survive 2 s of it), but not for good.
pub const NEVER_FIT_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

/// `never_fits_now` for `NEVER_FIT_GRACE` refuses the compile (`XCL_REFUSE_NEVER_FIT=0`: it waits for memory).
pub fn refuse_never_fit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_REFUSE_NEVER_FIT").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: compiles a bind waits for wait for memory even when they cannot fit with every running compile done (XCL_REFUSE_NEVER_FIT=0)");
        }
        on
    })
}

/// The memory a compile of `dxil_len` bytes of DXIL keeps free for the compiles running to take
/// while it runs: their measured rise (`Room::compile_rise`) × its expected duration, at most
/// `GROWTH_CAP` as the title's growth. play-98's giant grew the process heap past its overdue
/// estimate: XCL's private memory went from 21.4 to 23.6 GB and the memory available from 2,623
/// to 781 MB in ~40 s, and memwatch stopped the run.
pub fn compile_rise(room: &Room, dxil_len: usize) -> u64 {
    room.compile_rise.saturating_mul(compile_secs(dxil_len)).min(GROWTH_CAP)
}

pub const IDLE_BREAK_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

pub struct IdleBreaker {
    queued: VecDeque<std::thread::ThreadId>,
    idle_since: Option<std::time::Instant>,
}

impl IdleBreaker {
    pub const fn new() -> Self {
        Self { queued: VecDeque::new(), idle_since: None }
    }

    pub fn hold(&mut self, id: std::thread::ThreadId, blocking: bool, running: usize, available: u64, floor: u64, now: std::time::Instant) -> bool {
        if !blocking {
            self.queued.retain(|&queued| queued != id);
            if self.queued.is_empty() {
                self.idle_since = None;
            }
            return false;
        }
        if !self.queued.contains(&id) {
            self.queued.push_back(id);
        }
        if running != 0 {
            return false;
        }
        let idle = now.duration_since(*self.idle_since.get_or_insert(now));
        idle >= IDLE_BREAK_GRACE && self.queued.front() == Some(&id) && available > floor
    }

    pub fn started(&mut self, id: std::thread::ThreadId) {
        self.queued.retain(|&queued| queued != id);
        self.idle_since = None;
    }
}

/// Where a pipeline made for later waits for a background worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// its big driver compile did not fit the memory when a worker came to it (`Backlog::park`)
    Parked,
    ParkedPeer,
    /// console shaders of `HEAVY_SHADER_BYTES` or more: its driver compile is likely a big one
    Heavy,
    /// shaders an earlier run used
    Known,
    Compute,
    Graphics,
    /// (`Backlog::drawn_first`) an earlier run bound it, and an earlier build took
    /// `pso_manifest::SLOW_MS` or more, or its console shaders are heavy
    Slow,
    /// (`Backlog::drawn_first`) no run bound it, and it is slow or heavy as `Slow`
    LaterSlow,
    /// a ray pass built with every ray a miss (`Backlog::rt_pass`): after every other lane
    RtPass,
}

/// Console shader bytes from which a pipeline is built ahead of the others (`Lane::Heavy`): a
/// retail game's compute programs came to ~16 B of DXIL for each byte of console code, so these
/// are the ones whose driver compile takes seconds to minutes and hundreds of MB to GB. Started
/// when the title makes them, they compile while it still loads and the memory is there; left
/// for their first bind, the bind waits that long, and a giant no longer fits beside the game.
pub const HEAVY_SHADER_BYTES: usize = 16 << 10;

/// Root signatures of the pipelines bound last that `Backlog::pop` looks at.
pub const RECENT_GROUPS: usize = 8;

struct Parked<T> {
    group: u64,
    id: u64,
    first_use: Option<usize>,
    item: T,
}

/// The pipelines made for later, in the order background workers take them – by how soon a
/// bind is likely to ask for them, not by when they were made:
/// 1. parked pipelines which fit now, preferring peers of an active bind, then
///    groups bound lately (2.), then by previous first use, or biggest first without history;
/// 2. those of a group (the title's root signature: a pass's pipelines share one) bound
///    lately, the last bound group first, the newest of it first;
/// 3. the heavy ones, the biggest first, on a share of the workers so the rest keep going
///    through the many small ones;
/// 4. those of shaders an earlier run used, then compute (a dispatch waits for its pipeline
///    however long it takes), then graphics – of each the newest first: a title makes a
///    pipeline when what draws with it comes in, and of the tens of thousands it makes while
///    it loads the first ones are no likelier than the last.
pub struct Backlog<T> {
    /// By DXIL size, smallest first
    parked: Vec<(usize, Parked<T>)>,
    /// the pipelines waiting for a worker, by the number each got when it was made; a number
    /// in the lists below that is no longer here was taken through another list
    waiting: HashMap<u64, (T, Lane)>,
    /// (console shader bytes, number), smallest first
    heavy: Vec<(usize, u64)>,
    /// (first use, number): the one an earlier run used first on top, then those with no first
    /// use (`push_as`); of equals the newest first
    known: BinaryHeap<(Reverse<u64>, u64)>,
    compute: Vec<u64>,
    graphics: Vec<u64>,
    /// those of `graphics` without colour targets (`depth_only`), taken before the others
    depth: Vec<u64>,
    /// the numbers of each group, oldest first
    groups: HashMap<u64, Vec<u64>>,
    promoted: VecDeque<u64>,
    next: u64,
    /// heavy ones among `waiting`
    heavy_waiting: usize,
    /// workers building a heavy one now
    heavy_out: usize,
    peer_out: usize,
    /// Build what was drawn first (`drawn_first`): the rest waits in the later lanes.
    drawn_first: bool,
    /// (measured time, smallest first from the end, number) of `Lane::Slow` and `Lane::LaterSlow`
    slow: Vec<(usize, u64)>,
    later_slow: Vec<(usize, u64)>,
    slow_waiting: usize,
    /// workers building a slow one now, of any lane
    slow_out: usize,
    /// workers building one of the later lanes now
    later_out: usize,
    /// ray passes built with every ray a miss (`rt_pass`), oldest first: taken when nothing else is
    rt: VecDeque<T>,
}

impl<T> Default for Backlog<T> {
    fn default() -> Self {
        Backlog {
            parked: Vec::new(),
            waiting: HashMap::new(),
            heavy: Vec::new(),
            known: BinaryHeap::new(),
            compute: Vec::new(),
            graphics: Vec::new(),
            depth: Vec::new(),
            groups: HashMap::new(),
            promoted: VecDeque::new(),
            next: 1,
            heavy_waiting: 0,
            heavy_out: 0,
            peer_out: 0,
            drawn_first: false,
            slow: Vec::new(),
            later_slow: Vec::new(),
            slow_waiting: 0,
            slow_out: 0,
            later_out: 0,
            rt: VecDeque::new(),
        }
    }
}

fn insert_by_size<T>(list: &mut Vec<(usize, T)>, size: usize, item: T) {
    // (before its equals: of two the same size the one made first is taken first, from the end)
    let at = list.partition_point(|(s, _)| *s < size);
    list.insert(at, (size, item));
}

/// The last number of `list` still in `waiting`; the ones after it, taken through another list, go.
fn last_waiting<T>(list: &mut Vec<u64>, waiting: &HashMap<u64, (T, Lane)>) -> Option<u64> {
    while let Some(n) = list.pop() {
        if waiting.contains_key(&n) {
            return Some(n);
        }
    }
    None
}

/// The known pipeline still in `waiting` that an earlier run used first (`Backlog::known`); the
/// ones before it, taken through another list, go.
fn first_known<T>(known: &mut BinaryHeap<(Reverse<u64>, u64)>, waiting: &HashMap<u64, (T, Lane)>) -> Option<u64> {
    while let Some((_, n)) = known.pop() {
        if waiting.contains_key(&n) {
            return Some(n);
        }
    }
    None
}

impl<T> Backlog<T> {
    // The slow share and the order of early, slow and requested pipelines are adapted from CGBJ
    // (a VIctory fork) at f4b48ad766ddbec1a1867022cd42f7645b0dd55e,
    // crates/victory-gpu/src/pipeline/make.rs (`slow_limit`, `Queue::push_as`, `Queue::pop`).
    // Copyright (C) the VIctory/CGBJ contributors. GPL-3.0-or-later.
    /// The order in which earlier runs' draws come first: what was drawn first, and what no run drew only
    /// behind it, on at most a quarter of the workers. A retail title creates ~84 000 pipelines
    /// while it loads and draws with ~2 000 (CGBJ `pipeline/known.rs`); building every one ended
    /// its run inside the driver's compiler, and an NVIDIA driver keeps ~93% of a pipeline's
    /// memory after it is released. A pipeline no run drew is still built – later, or at its
    /// first bind, which never waits behind this order.
    ///
    /// 1. parked, promoted and recently bound groups' ones as `Backlog` says (a later one
    ///    within the later lanes' share);
    /// 2. those an earlier run bound and that took `pso_manifest::SLOW_MS` or more (or whose
    ///    shaders are heavy), the quickest measured first, on a quarter of the workers;
    /// 3. the other ones an earlier run bound, the newest first;
    /// 4. the ones no run bound, on a quarter of the workers: slow ones within the slow share,
    ///    then compute, then depth-only, then graphics, the newest first.
    pub fn drawn_first() -> Self {
        Backlog { drawn_first: true, ..Default::default() }
    }

    fn later(&self, lane: Lane) -> bool {
        self.drawn_first && matches!(lane, Lane::Compute | Lane::Graphics | Lane::LaterSlow)
    }

    /// A pipeline the title made: `shader_bytes` of console shaders, `known` from an earlier
    /// run, of `group` (its root signature; 0: unknown); `slow_ms`: an earlier build of it took
    /// that long (`pso_manifest::slow_ms`).
    pub fn push(&mut self, shader_bytes: usize, known: bool, compute: bool, group: u64, slow_ms: Option<u64>, item: T) -> u64 {
        let n = self.next;
        self.push_as(n, shader_bytes, known, compute, group, slow_ms, None, item);
        n
    }

    /// `push` as number `n`, which the caller gave out (from 1, never twice). `first_use`: where an
    /// earlier run first used it (smaller: sooner); a known one is taken in that order.
    #[allow(clippy::too_many_arguments)]
    pub fn push_as(&mut self, n: u64, shader_bytes: usize, known: bool, compute: bool, group: u64, slow_ms: Option<u64>, first_use: Option<u64>, item: T) {
        self.next = self.next.max(n + 1);
        let slow = slow_ms.is_some() || shader_bytes >= HEAVY_SHADER_BYTES;
        let quickest_last = usize::MAX - slow_ms.map_or(usize::MAX, |ms| ms as usize);
        let lane = if self.drawn_first && slow {
            self.slow_waiting += 1;
            if known {
                insert_by_size(&mut self.slow, quickest_last, n);
                Lane::Slow
            } else {
                insert_by_size(&mut self.later_slow, quickest_last, n);
                Lane::LaterSlow
            }
        } else if shader_bytes >= HEAVY_SHADER_BYTES {
            insert_by_size(&mut self.heavy, shader_bytes, n);
            self.heavy_waiting += 1;
            Lane::Heavy
        } else if known {
            self.known.push((Reverse(first_use.unwrap_or(u64::MAX)), n));
            Lane::Known
        } else if compute {
            self.compute.push(n);
            Lane::Compute
        } else {
            self.graphics.push(n);
            Lane::Graphics
        };
        if group != 0 {
            self.groups.entry(group).or_default().push(n);
        }
        self.waiting.insert(n, (item, lane));
    }

    /// Graphics pipeline `n` has no colour targets (a shadow map's, a depth prepass's): what it
    /// draws is read by the draws after it, so it is built before the other graphics ones.
    pub fn depth_only(&mut self, n: u64) {
        if self.waiting.get(&n).is_some_and(|(_, lane)| *lane == Lane::Graphics) {
            self.depth.push(n);
        }
    }

    /// Pipeline `n` is a ray pass built with every ray a miss while ray tracing is off
    /// (`raytracing::miss_pass`): it is taken after every other pipeline waiting, the newest first.
    pub fn rt_pass(&mut self, n: u64) {
        let Some((item, lane)) = self.waiting.remove(&n) else { return };
        match lane {
            Lane::Heavy => self.heavy_waiting -= 1,
            Lane::Slow | Lane::LaterSlow => self.slow_waiting -= 1,
            _ => {}
        }
        self.rt.push_back(item);
    }

    pub fn promote(&mut self, n: u64) {
        if n == 0 || self.promoted.back() == Some(&n) {
            return;
        }
        self.promoted.retain(|&id| id != n);
        if self.promoted.len() >= 64 {
            self.promoted.pop_front();
        }
        self.promoted.push_back(n);
    }

    /// A pipeline of `group` whose driver compile of `dxil_len` bytes did not fit: it waits for
    /// memory. A bind that needs it schedules an urgent build instead.
    pub fn park(&mut self, dxil_len: usize, group: u64, id: u64, first_use: Option<usize>, item: T) {
        insert_by_size(&mut self.parked, dxil_len, Parked { group, id, first_use, item });
    }

    fn take(&mut self, n: u64) -> Option<(T, Lane)> {
        let (item, lane) = self.waiting.remove(&n)?;
        if lane == Lane::Heavy {
            self.heavy_waiting -= 1;
            self.heavy_out += 1;
        }
        if matches!(lane, Lane::Slow | Lane::LaterSlow) {
            self.slow_waiting -= 1;
            self.slow_out += 1;
        }
        if self.later(lane) {
            self.later_out += 1;
        }
        Some((item, lane))
    }

    /// The first of `list` (from its end) still waiting, if it may be taken now.
    fn take_measured(list: &mut Vec<(usize, u64)>, waiting: &HashMap<u64, (T, Lane)>) -> Option<u64> {
        while let Some(&(_, n)) = list.last() {
            if waiting.contains_key(&n) {
                return Some(n);
            }
            list.pop();
        }
        None
    }

    fn pop_drawn_first(&mut self, workers: usize) -> Option<(T, Lane)> {
        let share = heavy_share(workers);
        let only_slow = self.slow_waiting == self.waiting.len();
        if self.slow_out < share || only_slow {
            if let Some(n) = Self::take_measured(&mut self.slow, &self.waiting) {
                self.slow.pop();
                return self.take(n);
            }
        }
        if let Some(n) = first_known(&mut self.known, &self.waiting) {
            return self.take(n);
        }
        if self.later_out >= share {
            return None;
        }
        if self.slow_out < share || only_slow {
            if let Some(n) = Self::take_measured(&mut self.later_slow, &self.waiting) {
                self.later_slow.pop();
                return self.take(n);
            }
        }
        let n = last_waiting(&mut self.compute, &self.waiting)
            .or_else(|| last_waiting(&mut self.depth, &self.waiting))
            .or_else(|| last_waiting(&mut self.graphics, &self.waiting))?;
        self.take(n)
    }

    /// The next pipeline for one of `workers` background workers, and the lane it came from
    /// (`done` is told when it is built). `recent`: the groups of the pipelines bound last, the
    /// last one first; `live`: it still wants building (parked ones that do not are dropped);
    /// `fits`: a worker can start a parked compile of this DXIL size under the ordinary floor
    /// (asked once for a size in a take, like `peer_fits`).
    pub fn pop(&mut self, workers: usize, recent: &[u64], live: impl Fn(&T) -> bool, fits: impl Fn(usize) -> bool, peer_fits: impl Fn(usize) -> bool) -> Option<(T, Lane)> {
        let recent = &recent[..recent.len().min(RECENT_GROUPS)];
        // (one sample of the memory stands behind a take: a DXIL size is asked about once)
        let (fit, peer_fit) = (std::cell::RefCell::new(HashMap::new()), std::cell::RefCell::new(HashMap::new()));
        let fits = |len: usize| *fit.borrow_mut().entry(len).or_insert_with(|| fits(len));
        let peer_fits = |len: usize| *peer_fit.borrow_mut().entry(len).or_insert_with(|| peer_fits(len));
        let list = &mut self.parked;
        list.retain(|(_, p)| live(&p.item));
        {
            let peer = recent.iter().filter(|g| **g != 0).find_map(|g| {
                list.iter().enumerate().filter(|(_, (len, p))| p.group == *g && peer_fits(*len))
                    .min_by_key(|(_, (_, p))| (p.first_use.unwrap_or(usize::MAX), std::cmp::Reverse(p.id))).map(|(at, _)| at)
            });
            if let Some(at) = peer {
                let (_, Parked { id, item, .. }) = list.remove(at);
                self.promoted.retain(|&n| n != id);
                self.peer_out += 1;
                return Some((item, Lane::ParkedPeer));
            }
        }
        let promoted = self.promoted.iter().enumerate().rev().find_map(|(pos, &id)| {
            if self.waiting.contains_key(&id) {
                Some((pos, id, None))
            } else {
                list.iter().position(|(len, p)| p.id == id && fits(*len)).map(|at| (pos, id, Some(at)))
            }
        });
        if let Some((pos, id, parked)) = promoted {
            self.promoted.remove(pos);
            if let Some(at) = parked {
                return Some((list.remove(at).1.item, Lane::Parked));
            }
            return self.take(id);
        }
        let pick = |group: Option<u64>| {
            list.iter().enumerate().rev().filter(|(_, (len, p))| group.is_none_or(|g| g == p.group) && fits(*len))
                .min_by_key(|(_, (_, p))| p.first_use.unwrap_or(usize::MAX))
                .map(|(at, _)| at)
        };
        if let Some(at) = recent.iter().copied().filter(|&g| g != 0).find_map(|g| pick(Some(g))).or_else(|| pick(None)) {
            return Some((list.remove(at).1.item, Lane::Parked));
        }
        let later_full = self.later_out >= heavy_share(workers);
        for group in recent {
            let Some(list) = self.groups.get_mut(group) else { continue };
            match last_waiting(list, &self.waiting) {
                Some(n) if later_full && self.waiting.get(&n).is_some_and(|(_, lane)| self.drawn_first && matches!(lane, Lane::Compute | Lane::Graphics | Lane::LaterSlow)) => list.push(n),
                Some(n) => return self.take(n),
                None => drop(self.groups.remove(group)),
            }
        }
        if self.drawn_first {
            return self.pop_drawn_first(workers).or_else(|| self.rt.pop_back().map(|item| (item, Lane::RtPass)));
        }
        if self.heavy_waiting > 0 && (self.heavy_waiting == self.waiting.len() || self.heavy_out < heavy_share(workers)) {
            while let Some((_, n)) = self.heavy.pop() {
                if self.waiting.contains_key(&n) {
                    return self.take(n);
                }
            }
        }
        if let Some(n) = first_known(&mut self.known, &self.waiting) {
            return self.take(n);
        }
        for lane in [Lane::Compute, Lane::Graphics] {
            if lane == Lane::Graphics {
                if let Some(n) = last_waiting(&mut self.depth, &self.waiting) {
                    return self.take(n);
                }
            }
            let list = if lane == Lane::Compute { &mut self.compute } else { &mut self.graphics };
            if let Some(n) = last_waiting(list, &self.waiting) {
                return self.take(n);
            }
        }
        self.rt.pop_back().map(|item| (item, Lane::RtPass))
    }

    /// A worker is done with what it took from `lane`.
    pub fn done(&mut self, lane: Lane) {
        match lane {
            Lane::Heavy => self.heavy_out = self.heavy_out.saturating_sub(1),
            Lane::ParkedPeer => self.peer_out -= 1,
            Lane::Slow | Lane::LaterSlow => self.slow_out = self.slow_out.saturating_sub(1),
            _ => {}
        }
        if self.later(lane) {
            self.later_out = self.later_out.saturating_sub(1);
        }
    }

    /// (pipelines waiting that an earlier run bound, that no run bound)
    pub fn drawn_split(&self) -> (usize, usize) {
        let later = self.waiting.values().filter(|(_, lane)| self.later(*lane)).count();
        (self.waiting.len() - later, later + self.rt.len())
    }

    /// (pipelines waiting for a worker, of them parked for memory)
    pub fn waiting(&self) -> (usize, usize) {
        (self.parked.len() + self.waiting.len() + self.rt.len(), self.parked.len())
    }

    pub fn has_parked(&self) -> bool {
        !self.parked.is_empty()
    }

    pub fn parked_peers(&self, recent: &[u64]) -> usize {
        self.parked.iter().filter(|(_, p)| p.group != 0 && recent.contains(&p.group)).count()
    }

    /// The smallest driver compile parked, in bytes of DXIL (0: none).
    pub fn least_parked(&self) -> usize {
        self.parked.first().map_or(0, |(len, _)| (*len).max(1))
    }
}

/// The workers that may build heavy pipelines at once while others wait: a quarter, one at least.
pub fn heavy_share(workers: usize) -> usize {
    workers.div_ceil(4).max(1)
}

/// Background workers for a PC of `threads` hardware threads on `cores` cores: one for each
/// core, and two threads left over at least; two at least, 32 at most. (One for each thread but
/// two – 22 on 12 cores, ~17 of them busy – made a retail game's loading 2.6 times as long: a
/// worker on a core's second thread takes from the title's thread on its first.) How many of
/// them build at a time follows what the title is doing (`Pace`).
pub fn background_workers(threads: usize, cores: usize) -> usize {
    cores.min(threads.saturating_sub(2)).clamp(2, 32)
}

/// The background workers that build while the title needs the PC most: a quarter, two at least.
pub fn least_workers(pool: usize) -> usize {
    (pool / 4).max(2).min(pool.max(1))
}

/// How the background workers go about it now (`Pace::step`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tempo {
    /// workers that may build at once
    pub workers: usize,
    /// they run at their lowest priority: behind title threads that run below normal themselves
    pub low: bool,
}

/// Steps (`Pace::step`) the title counts as loading for after its storage was last busy.
pub const LOADING_HOLD: u32 = 4;

/// Whether the title loads, from what its storage API did within `ms`: 16 requests or 8 MB a
/// second, or more. (A retail game ran 160 to 740 requests a second while it loaded, 55 with its
/// loader starved, and one request in half a minute once in its world.)
pub fn loading(requests: u64, bytes: u64, ms: u64) -> bool {
    requests * 1000 >= 16 * ms || bytes * 1000 >= (8 << 20) * ms
}

/// How many background workers build at once, from what the rest of the process is doing: the
/// cores its other threads (the title's, its loader's, the storage's) are not using – half of
/// those, at the lowest priority, while the title loads through its storage API – and never
/// fewer than `least_workers`. Fewer at once as soon as the title needs the cores, one more a
/// step when it does not. (The storage alone does not tell: a retail game's loader read
/// nothing for four minutes of a load that 22 workers had slowed down.)
pub struct Pace {
    pool: usize,
    cores: usize,
    workers: usize,
    calm: u32,
}

impl Pace {
    /// For `pool` workers on `cores` cores. It starts at the least: a title starts by loading.
    pub fn new(pool: usize, cores: usize) -> Self {
        Pace { pool: pool.max(1), cores, workers: least_workers(pool), calm: 0 }
    }

    /// One step (they come at even intervals). `others`: the CPU the process's other threads
    /// used since the last step, in thousandths of a thread; `loading`: the title's storage
    /// ran requests since then, or holds some.
    pub fn step(&mut self, others: u64, loading: bool) -> Tempo {
        self.calm = if loading { 0 } else { self.calm.saturating_add(1) };
        let loading = self.calm < LOADING_HOLD;
        let free = self.cores.saturating_sub(others.div_ceil(1000) as usize);
        let target = if loading { free / 2 } else { free }.clamp(least_workers(self.pool), self.pool);
        self.workers = if target > self.workers { self.workers + 1 } else { target };
        Tempo { workers: self.workers, low: loading }
    }
}

/// What a run's background compiles leave available at the least while the title loads, and
/// once it has not loaded for `RISE_KEPT_MS` (`Headroom::floor`).
pub const MIN_HEADROOM: u64 = 4 << 30;
pub const CALM_HEADROOM: u64 = 3 << 29;

/// Console bytes already backed, and the host's private commit at the same sample.
#[derive(Clone, Copy, Default)]
pub struct TitleMemory {
    pub backed: u64,
    pub commit: u64,
}

const STEADY_WINDOW_MS: u64 = 30_000;
const STEADY_GROWTH: u64 = 64 << 20;

/// A title can leave part of its budget unused. Use two 30-second medians so brief driver
/// compiles do not look like sustained title growth; backings and storage are sampled directly.
#[derive(Default)]
pub struct TitleReserve {
    samples: VecDeque<(u64, u64)>,
    backed: u64,
    projected: u64,
    steady: bool,
}

impl TitleReserve {
    pub const fn empty() -> Self {
        Self { samples: VecDeque::new(), backed: 0, projected: 0, steady: false }
    }

    pub fn note(&mut self, now: u64, memory: TitleMemory, budget: u64, loading: bool) {
        self.projected = memory.host_reserve(budget);
        if loading || memory.backed.abs_diff(self.backed) >= STEADY_GROWTH {
            self.samples.clear();
            self.backed = memory.backed;
        }
        self.samples.push_back((now, memory.commit));
        let window = 2 * STEADY_WINDOW_MS;
        while self.samples.len() > 1 && now - self.samples[1].0 >= window {
            self.samples.pop_front();
        }
        self.steady = false;
        if now - self.samples[0].0 < window {
            return;
        }
        let split = self.samples.partition_point(|&(at, _)| now - at >= STEADY_WINDOW_MS);
        if split < 3 || self.samples.len() - split < 3 {
            return;
        }
        let median = |lo, hi| {
            let mut bytes: Vec<_> = self.samples.range(lo..hi).map(|&(_, bytes)| bytes).collect();
            let mid = bytes.len() / 2;
            *bytes.select_nth_unstable(mid).1
        };
        let before = median(0, split);
        let recent = median(split, self.samples.len());
        self.steady = recent.saturating_sub(before) < STEADY_GROWTH;
        if self.steady {
            self.steady = self.samples.iter().rev().take_while(|&&(at, _)| now - at <= 2_000)
                .any(|&(_, bytes)| bytes.saturating_sub(recent) <= 512 << 20);
        }
    }

    pub fn effective(&self) -> u64 {
        if self.steady { 0 } else { self.projected }
    }

    pub fn is_steady(&self) -> bool {
        self.steady
    }
}

impl TitleMemory {
    /// Project the remaining console bytes at the observed host bytes per backed byte.
    /// Before a quarter of the budget is backed, use that quarter as the denominator
    /// so startup's fixed costs don't get multiplied by a nearly empty budget.
    pub fn host_reserve(self, budget: u64) -> u64 {
        let left = budget.saturating_sub(self.backed);
        let observed = self.backed.max(budget / 4).max(1) as u128;
        let host = observed.max(self.commit as u128);
        (u128::from(left) * host).div_ceil(observed).min(u128::from(u64::MAX)) as u64
    }
}
/// The span a rise of the process's memory is measured over, and how long a rise – and that
/// the title loaded – is kept in mind.
const RISE_SPAN_MS: u64 = 10_000;
const RISE_KEPT_MS: u64 = 60_000;

/// The memory a run's background compiles leave available, following what the title does:
/// the line a run is stopped at and twice the biggest rise of the process's private memory
/// within `RISE_SPAN_MS` of the last `RISE_KEPT_MS` – another rise like it still ends above
/// that line – and at the least one amount while the title loads (and for `RISE_KEPT_MS`
/// after, and from the start: a title starts by loading) and a smaller one once it is calm.
/// (With 1 GB left throughout, a retail game's world load committed 2.2 GB more within seconds
/// and took the PC to 688 MB available. With 4 GB left throughout, its ~1 100 compute programs
/// of 0.5 to 1 MB stayed parked in a world that left ~4 GB available, and each was compiled
/// when a bind already waited for it. Its rises within 10 s were 2.5 to 3 GB while it loaded
/// and 0.3 to 0.9 GB in its world; within 30 s over 5 GB – twice that would have kept every
/// background build out of its load.) The compiles' own memory is in the rise too: they hold
/// back from each other while it grows.
#[derive(Default)]
pub struct Headroom {
    /// (ms, private memory) within the last `RISE_SPAN_MS`, and the one before them
    seen: VecDeque<(u64, u64)>,
    /// (ms, the rise up to then) within the last `RISE_KEPT_MS`
    rises: VecDeque<(u64, u64)>,
    /// when it was last told (ms), and when it was last told the title loads
    now: u64,
    loaded: u64,
}

impl Headroom {
    pub const fn empty() -> Self {
        Self { seen: VecDeque::new(), rises: VecDeque::new(), now: 0, loaded: 0 }
    }

    /// The process's private memory is `commit` at `now_ms`, and the title loads or not (`loading`).
    pub fn note(&mut self, now_ms: u64, commit: u64, loading: bool) {
        // (the one before the span stays: after a pause in the samples the rise is measured from it)
        while self.seen.len() > 1 && self.seen[1].0 + RISE_SPAN_MS <= now_ms {
            self.seen.pop_front();
        }
        self.seen.push_back((now_ms, commit));
        let low = self.seen.iter().map(|s| s.1).min().unwrap_or(commit);
        while self.rises.front().is_some_and(|r| r.0 + RISE_KEPT_MS < now_ms) {
            self.rises.pop_front();
        }
        self.rises.push_back((now_ms, commit - low));
        self.now = now_ms;
        if loading {
            self.loaded = now_ms;
        }
    }

    /// Whether the title counts as loading now.
    pub fn loading(&self) -> bool {
        self.now < self.loaded + RISE_KEPT_MS
    }

    /// What background compiles leave available now, above `stop`, the line a run is stopped
    /// at: `calm` at the least, `loading` while the title loads. `bind_waits`: a bind waits for
    /// a pipeline now – the title is held anyway, and what is built meanwhile is one bind less
    /// to wait: no least amount then, only what follows the growth (`stop` itself, the floor of
    /// the compile the bind waits for, while the memory is flat). (With a least amount alone
    /// the memory available settled just over it once a retail game was in its world: nothing
    /// heavy started in the background, and the binds built 2.2 pipelines a minute, one at a
    /// time, 26 to 28 s each.)
    pub fn floor(&self, stop: u64, calm: u64, loading: u64, bind_waits: bool) -> u64 {
        let growth = stop + 2 * self.rises.iter().map(|r| r.1).max().unwrap_or(0);
        if bind_waits {
            return growth;
        }
        growth.max(if self.loading() { loading } else { calm.min(loading) })
    }

    pub fn title_floor(&self, stop: u64, calm: u64, loading: u64, bind_waits: bool, reserve: u64) -> u64 {
        let growth_floor = self.floor(stop, calm, loading, true);
        if bind_waits {
            growth_floor
        } else {
            let least = if self.loading() { loading } else { calm.min(loading) };
            least.max(growth_floor.saturating_add(reserve))
        }
    }
}

/// Which built pipelines nothing has bound yet to release, of `built` (id, when built): the ones
/// built `limit` or longer before `now`, and of the rest the oldest beyond the newest `keep`. A
/// run that builds every pipeline when the title makes it would otherwise hold thousands a
/// frame never binds (~400 KB each in an NVIDIA driver); one released is built again at its
/// bind from the driver's blob in ~0.1 s.
pub fn idle_to_release<I: Copy>(built: &mut Vec<(I, u64)>, now: u64, limit: u64, keep: usize) -> Vec<I> {
    built.sort_by_key(|&(_, at)| std::cmp::Reverse(at));
    let fresh = built.iter().take_while(|&&(_, at)| now.saturating_sub(at) < limit).count().min(keep);
    built[fresh..].iter().map(|&(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drawn first: what an earlier run bound, the slow ones of it quickest measured first on a
    /// quarter of the workers; what no run bound behind it, on a quarter of the workers.
    #[test]
    fn known_ones_go_in_the_order_an_earlier_run_first_used_them() {
        for drawn_first in [false, true] {
            let mut b = if drawn_first { Backlog::drawn_first() } else { Backlog::default() };
            let push = |b: &mut Backlog<&'static str>, first_use: Option<u64>, item: &'static str| {
                let n = b.next;
                b.push_as(n, 100, true, false, 0, None, first_use, item);
            };
            push(&mut b, None, "no first use, older");
            push(&mut b, Some(7), "used third");
            push(&mut b, Some(0), "bound first");
            push(&mut b, None, "no first use, newer");
            push(&mut b, Some(3), "bound later");
            let take = |b: &mut Backlog<&'static str>| b.pop(8, &[], |_| true, |_| true, |_| false).map(|(item, _)| item);
            for want in ["bound first", "bound later", "used third", "no first use, newer", "no first use, older"] {
                assert_eq!(take(&mut b), Some(want), "drawn_first {drawn_first}");
            }
            assert_eq!(take(&mut b), None);
        }
    }

    #[test]
    fn drawn_first_builds_what_runs_bound_before_the_rest_and_caps_slow_and_later() {
        let mut b = Backlog::drawn_first();
        b.push(100, false, false, 0, None, "later g0");
        b.push(100, false, true, 0, None, "later c1");
        b.push(100, false, false, 0, None, "later g1");
        b.push(100, true, false, 0, None, "known 1");
        b.push(100, true, true, 0, Some(5000), "known slow 5 s");
        b.push(100, true, true, 0, Some(1500), "known slow 1.5 s");
        b.push(HEAVY_SHADER_BYTES, true, true, 0, None, "known heavy");
        b.push(HEAVY_SHADER_BYTES, false, true, 0, None, "later heavy");
        b.push(100, true, false, 0, None, "known 2");
        assert_eq!(b.drawn_split(), (5, 4));
        let workers = 8;
        assert_eq!(heavy_share(workers), 2);
        let take = |b: &mut Backlog<&'static str>| b.pop(workers, &[], |_| true, |_| true, |_| false);
        assert_eq!(take(&mut b), Some(("known slow 1.5 s", Lane::Slow)));
        assert_eq!(take(&mut b), Some(("known slow 5 s", Lane::Slow)));
        assert_eq!(take(&mut b), Some(("known 2", Lane::Known)), "two workers hold slow ones");
        assert_eq!(take(&mut b), Some(("known 1", Lane::Known)));
        assert_eq!(take(&mut b), Some(("later c1", Lane::Compute)));
        assert_eq!(take(&mut b), Some(("later g1", Lane::Graphics)));
        assert_eq!(take(&mut b), None, "the later share and the slow share are both taken");
        b.done(Lane::Slow);
        assert_eq!(take(&mut b), Some(("known heavy", Lane::Slow)));
        b.done(Lane::Compute);
        assert_eq!(take(&mut b), Some(("later g0", Lane::Graphics)), "the slow share is taken again");
        assert_eq!(take(&mut b), None, "only a slow one waits, but the later share is taken");
        b.done(Lane::Graphics);
        assert_eq!(take(&mut b), Some(("later heavy", Lane::LaterSlow)), "nothing else waits: no slow share");
        assert_eq!(b.drawn_split(), (0, 0));
        for lane in [Lane::Slow, Lane::Slow, Lane::LaterSlow, Lane::Graphics] {
            b.done(lane);
        }
        assert_eq!((b.slow_out, b.later_out), (0, 0));
    }

    /// A pipeline no run bound is not taken through a group bound lately beyond the later share.
    #[test]
    fn drawn_first_keeps_later_group_peers_within_the_later_share() {
        let mut b = Backlog::drawn_first();
        b.push(100, false, false, 0x51, None, "peer 1");
        b.push(100, false, false, 0x51, None, "peer 2");
        b.push(100, true, false, 0, None, "known");
        let take = |b: &mut Backlog<&'static str>| b.pop(4, &[0x51], |_| true, |_| true, |_| false);
        assert_eq!(take(&mut b), Some(("peer 2", Lane::Graphics)));
        assert_eq!(take(&mut b), Some(("known", Lane::Known)));
        assert_eq!(take(&mut b), None);
        b.done(Lane::Graphics);
        assert_eq!(take(&mut b), Some(("peer 1", Lane::Graphics)));
        assert_eq!(b.waiting(), (0, 0));
    }

    const GB: u64 = 1 << 30;

    #[test]
    fn only_total_physical_capacity_can_make_a_runtime_compile_impossible() {
        let cost = peak(3_515_708);
        assert!(fits_on_machine(cost, GB, 32 * GB));
        assert!(fits_on_machine(cost, GB, cost + GB));
        assert!(!fits_on_machine(cost, GB, cost + GB - 1));
        assert!(fits_on_machine(cost, GB, 0), "unknown total capacity is not a refusal");
    }

    fn room(available: u64) -> Room {
        Room { available, floor: GB, urgent_floor: GB, commit: 0, commit_room: u64::MAX, fall: 0, commit_fall: 0, trend_fall: 0, title_fall: 0, compile_rise: 0 }
    }

    #[test]
    fn no_compile_starts_at_the_275_second_emergency_floor() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 2, urgent: 2, starving: 1, reserved: 2 * peak(512 << 10), ..Load::default() };
        for floor in [10_463 * MIB, 2_856 * MIB, GB] {
            let room = Room { available: 1_024 * MIB, floor, ..room(0) };
            assert_eq!(admit(&ask(121_004, false), &load, &room), Verdict::Hold);
            assert_eq!(admit(&ask(148_428, true), &load, &room), Verdict::Hold);
        }
    }

    #[test]
    fn urgent_admission_requires_all_in_flight_reservations_to_fit() {
        let len = 148_428;
        let load = Load { running: 2, urgent: 2, reserved: 2 * peak(512 << 10), ..Load::default() };
        let needed = GB + load.reserved + peak(len);
        assert_eq!(admit(&ask(len, true), &load, &room(needed - 1)), Verdict::Hold);
        assert_eq!(admit(&ask(len, true), &load, &room(needed)), Verdict::Urgent);
    }

    #[test]
    fn an_idle_governor_unblocks_the_oldest_waited_for_compile() {
        let mut breaker = IdleBreaker::new();
        let newer = std::thread::spawn(|| std::thread::current().id()).join().unwrap();
        let oldest = std::thread::current().id();
        let now = std::time::Instant::now();
        let giant = 3_241_524;
        let available = GB + peak(giant) - (256 << 20);
        assert!(fits_on_machine(peak(giant), GB, 32 * GB));
        assert_eq!(admit(&ask(giant, true), &Load::default(), &room(available)), Verdict::Hold);
        assert!(!breaker.hold(oldest, true, 0, available, GB, now));
        assert!(!breaker.hold(newer, true, 0, available, GB, now));
        let stalled = now + IDLE_BREAK_GRACE;
        assert!(!breaker.hold(newer, true, 0, available, GB, stalled));
        assert!(!breaker.hold(oldest, true, 0, GB, GB, stalled));
        assert!(breaker.hold(oldest, true, 0, available, GB, stalled));
        breaker.started(oldest);
        assert!(!breaker.hold(newer, true, 1, available, GB, stalled + IDLE_BREAK_GRACE));
        assert!(!breaker.hold(newer, true, 0, available, GB, stalled + IDLE_BREAK_GRACE));
    }

    #[test]
    fn a_timed_out_waiter_does_not_force_a_compile_or_block_the_next_waiter() {
        let mut breaker = IdleBreaker::new();
        let first = std::thread::current().id();
        let second = std::thread::spawn(|| std::thread::current().id()).join().unwrap();
        let now = std::time::Instant::now();
        assert!(!breaker.hold(first, true, 0, 4 * GB, GB, now));
        assert!(!breaker.hold(second, true, 0, 4 * GB, GB, now));
        let stalled = now + IDLE_BREAK_GRACE;
        assert!(!breaker.hold(first, false, 0, 4 * GB, GB, stalled));
        assert!(breaker.hold(second, true, 0, 4 * GB, GB, stalled));
    }

    /// play-92 at ~430 s: 5 live compiles running, 1 a cold giant, 12,902 MiB reserved, 12,514 MiB
    /// of it settled, 2,737 MiB available. Small live compiles saw no spare and the bind queue
    /// built ~0.6 pipelines a second; what the settled ones take shows in the memory available.
    #[test]
    fn the_small_lane_counts_settled_compiles_as_the_memory_shows_them() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 5, urgent: 5, giants: 1, reserved: 12_902 * MIB, settled: 12_514 * MIB, giant_reserved: 3_000 * MIB, ..Load::default() };
        let small = Ask { small_lane: true, ..ask(200 << 10, true) };
        assert_eq!(admit(&small, &load, &room(2_737 * MIB)), Verdict::Urgent);
        let falling = Room { trend_fall: SETTLE_FALL, ..room(2_737 * MIB) };
        assert_eq!(admit(&small, &load, &falling), Verdict::Urgent, "while the memory falls too (XCL_SETTLE_OVERDUE)");
        assert_eq!(admit(&small, &load, &room(1_100 * MIB)), Verdict::Hold, "the floor stays");
    }

    /// play-95 at ~305 s: 2 driver compiles running on 3 slots (1 a giant), all 8,137 MiB of their
    /// reservation past its expected time, 5,059 MiB available and falling ~70 MB/s as the title
    /// loaded, 73.5 GB of commit room. Four waited-for compiles waited 250 to 281 s: the overdue
    /// reservation, already missing from the memory available, was subtracted again.
    #[test]
    fn a_waited_for_compile_starts_while_the_title_drains_memory_behind_overdue_compiles() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 2, giants: 1, urgent: 2, starving: 4, reserved: 8_137 * MIB, settled: 8_137 * MIB, giant_reserved: 6_000 * MIB, live_recent: true, ..Load::default() };
        let loading = Room { trend_fall: 70 * MIB, commit_room: 73_500 * MIB, ..room(5_059 * MIB) };
        assert_eq!(admit(&ask(148_428, true), &load, &loading), Verdict::Urgent);
        assert_eq!(admit(&Ask { small_lane: true, ..ask(148_428, true) }, &load, &loading), Verdict::Urgent);
        assert_eq!(admit(&ask(148_428, false), &Load { starving: 0, ..load }, &loading), Verdict::Background, "small background work too (XCL_SETTLE_BACKGROUND)");
        let fresh = Load { settled: 0, ..load };
        assert_eq!(admit(&ask(148_428, true), &fresh, &loading), Verdict::Hold, "compiles within their expected time keep theirs");
        assert_eq!(admit(&ask(148_428, true), &load, &Room { available: peak(148_428) + GB - 1, ..loading }), Verdict::Hold, "the 1 GB floor stays");
        assert_eq!(admit(&ask(148_428, true), &load, &Room { commit_room: 8_137 * MIB + GB, ..loading }), Verdict::Hold, "and the commit check, by the trend rule");
    }

    /// play-95 again, the background side: no dispatch ran in 310 s. A 2 KB loading pass
    /// (CS_DecorationDilate, CS_BakeLUT_2DBezier_3Segment_LinearShaper) never built while 2 giants
    /// held 8,137 MiB of reservation, all past its expected time, with 5,059 MiB available.
    #[test]
    fn a_small_background_compile_starts_beside_overdue_giants() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 2, giants: 2, reserved: 8_137 * MIB, settled: 8_137 * MIB, giant_reserved: 8_137 * MIB, live_recent: true, ..Load::default() };
        let loading = Room { fall: 70 * MIB, trend_fall: 70 * MIB, commit_room: 73_500 * MIB, ..room(5_059 * MIB) };
        let small = ask(2 << 10, false);
        assert_eq!(admit_why(&small, &load, &loading), (Verdict::Background, ""));
        let fresh = Load { settled: 0, giant_reserved: 0, ..load };
        assert!(admit_why(&small, &fresh, &loading).1.starts_with("memory"), "compiles within their expected time keep their estimates");
        assert_eq!(admit(&small, &load, &Room { available: peak(2 << 10) + GB + growth(&loading, 2 << 10).0 - 1, ..loading }), Verdict::Hold, "the floor and the title's growth stay");
        assert!(admit_why(&small, &load, &Room { commit_room: 8_137 * MIB + GB, ..loading }).1.starts_with("commit room"), "the commit check counts every estimate");
        assert_eq!(admit_why(&ask(GIANT_DXIL, false), &load, &loading).1, "a background giant within 30 s of live demand");
        assert_eq!(admit_why(&small, &Load { starving: 1, ..load }, &loading).1, "a compile a bind waits for waits for memory first");
    }

    /// play-97 at 87 s: one cold giant running within its expected time (~52 s), reserved at its
    /// corrected 7,520 MiB, 9,468 MiB available. Background compiles of 9 to 612 KB were held for
    /// memory (1,786 lines); in the small lane they see the giant as the memory available shows it.
    #[test]
    fn small_background_compiles_never_wait_on_cold_giant_reservations() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 1, giants: 1, reserved: 7_520 * MIB, giant_reserved: 7_520 * MIB, ..Load::default() };
        let small = ask(30 << 10, false);
        assert_eq!(admit_why(&small, &load, &room(9_468 * MIB)), (Verdict::Background, ""));
        assert_eq!(admit(&ask(612 << 10, false), &load, &room(9_468 * MIB)), Verdict::Background);
        let growing = Room { fall: 400 * MIB, ..room(9_468 * MIB) };
        assert_eq!(admit(&small, &load, &growing), Verdict::Background, "the title's growth is kept, at most GROWTH_CAP");
        assert_eq!(admit(&small, &load, &room(peak(30 << 10) + GB - 1)), Verdict::Hold, "the floor stays");
        let others = Load { reserved: 7_520 * MIB + 2_000 * MIB, ..load };
        assert_eq!(admit(&small, &others, &room(3_000 * MIB)), Verdict::Hold, "other compiles' reservations stay");
        assert_eq!(admit(&ask(1_208_312, false), &Load { live_recent: false, ..load }, &room(9_468 * MIB)), Verdict::Hold, "a background giant still counts the giant");
    }

    /// play-97: FFX_FSR4_PASS_14, 16 and 18 asked at 159 s and were held 310 s. At 170 s:
    /// 3,691 MiB available, 11,256 MiB reserved; at 474 s: 2,614 MiB available, 7,502 MiB reserved.
    #[test]
    fn a_waited_for_compile_that_cannot_fit_with_every_compile_done_never_fits_now() {
        const MIB: u64 = 1 << 20;
        for (available, reserved) in [(3_691 * MIB, 11_256 * MIB), (2_614 * MIB, 7_502 * MIB)] {
            for est in [16_375 * MIB, 16_370 * MIB, 25_411 * MIB] {
                assert!(never_fits_now(est, available, reserved, GB));
            }
            assert!(!never_fits_now(3_992 * MIB, available, reserved, GB), "a 1,547 KB giant waits for the running compiles");
            assert!(!never_fits_now(3_085 * MIB, available, reserved, GB));
        }
        assert!(!never_fits_now(16_375 * MIB, 18_004 * MIB, 0, GB), "at boot (18 GB available) the 16 GB ones wait");
        assert!(never_fits_now(25_411 * MIB, 18_004 * MIB, 0, GB));
    }

    /// play-95 at `305 s as the backlog's sample saw it: 3,059 pipelines parked, 2 giants holding
    /// 8,137 MiB of reservation, all past its expected time, 5,059 MiB available. The sample alone
    /// holds nothing that `admit` starts (a small one starts beside the overdue giants), and what it
    /// holds `admit` holds whatever the commit room and the title's growth are.
    #[test]
    fn the_sample_alone_holds_only_what_admit_holds() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 2, giants: 2, reserved: 8_137 * MIB, settled: 8_137 * MIB, giant_reserved: 8_137 * MIB, live_recent: true, ..Load::default() };
        let loading = |available: u64| Room { fall: 70 * MIB, trend_fall: 70 * MIB, commit_room: 73_500 * MIB, ..room(available) };
        let held = |dxil_len: usize, load: &Load, available: u64| {
            let sample = room(available);
            held_on_sample(dxil_len, peak(dxil_len), 20, load, available, sample.floor, sample.urgent_floor)
        };
        assert!(!held(2 << 10, &load, 5_059 * MIB), "a 2 KB loading pass goes on to the full verdict");
        assert_eq!(admit(&ask(2 << 10, false), &load, &loading(5_059 * MIB)), Verdict::Background);
        let loads = [load, Load { settled: 0, ..load }, Load { starving: 1, ..load }, Load { running: 22, ..load }, Load { live_recent: false, ..load }, Load::default()];
        for load in loads {
            for available in [GB, 5_059 * MIB, 16 * GB] {
                for dxil_len in [2 << 10, 148_428, 1 << 20, GIANT_DXIL, 4 * GIANT_DXIL] {
                    for room in [room(available), loading(available), Room { commit_room: 8_137 * MIB + GB, ..room(available) }] {
                        let verdict = admit(&ask(dxil_len, false), &load, &room);
                        assert!(!held(dxil_len, &load, available) || verdict == Verdict::Hold, "{dxil_len} B: {load:?}, {room:?}");
                    }
                }
            }
        }
        assert!(held(2 << 10, &Load { settled: 0, giant_reserved: 0, ..load }, 5_059 * MIB), "compiles within their expected time keep their estimates");
        assert!(!held(2 << 10, &Load { settled: 0, ..load }, 5_059 * MIB), "a small one leaves cold giants' reservations out (XCL_BACKGROUND_SMALL_LANE)");
        assert!(held(2 << 10, &Load { starving: 1, ..load }, 5_059 * MIB), "a waited-for compile waits for memory first");
        assert!(held(2 << 10, &load, GB), "under the floor");
        assert!(held(GIANT_DXIL, &load, 5_059 * MIB), "a background giant within 30 s of live demand");
    }

    /// A take asks `fits` once for a DXIL size, however many parked pipelines have it and however
    /// many of the groups bound lately they are looked through for.
    #[test]
    fn a_take_asks_once_for_a_size() {
        let mut b = Backlog::<&'static str>::default();
        for id in 1..=40 {
            b.park(if id % 2 == 0 { 2_000 } else { 3_000 }, 7, id, None, "parked");
        }
        let asked = std::cell::Cell::new(0);
        let fits = |_| {
            asked.set(asked.get() + 1);
            false
        };
        assert_eq!(b.pop(4, &[7, 8], |_| true, fits, |_| false), None);
        assert_eq!(asked.get(), 2);
    }

    /// play-98 at ~340 s: a giant past its expected time kept growing; the memory available fell
    /// from 2,623 to 781 MB in ~40 s (~46 MB/s) and memwatch stopped the run. With the rise held,
    /// a waited-for compile no longer starts into the last of the floor.
    #[test]
    fn admissions_keep_the_floor_against_the_compiles_measured_rise() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 3, giants: 1, urgent: 3, reserved: 6_000 * MIB, settled: 6_000 * MIB, giant_reserved: 4_000 * MIB, ..Load::default() };
        let small = ask(150 << 10, true);
        assert_eq!(admit(&small, &load, &room(1_213 * MIB)), Verdict::Urgent, "the old rule: 1,213 MiB holds 128 MiB and the floor");
        let rising = Room { compile_rise: 46 * MIB, ..room(1_213 * MIB) };
        assert!(admit_why(&small, &load, &rising).1.starts_with("memory"), "46 MB/s over its 5 s takes the floor");
        assert_eq!(admit(&small, &load, &Room { compile_rise: 46 * MIB, ..room(2_623 * MIB) }), Verdict::Urgent, "with room for the rise it starts");
        assert_eq!(admit(&ask(150 << 10, false), &Load { urgent: 0, ..load }, &rising), Verdict::Hold, "background work too");
    }

    /// The game's hard rule for cold giants: one at a time, and only when the corrected estimate,
    /// the title's growth and the compiles' growth fit (play-91b's curve fallback is not taken).
    #[test]
    fn a_cold_giant_in_the_game_starts_alone_and_only_on_its_full_estimate() {
        const MIB: u64 = 1 << 20;
        let giant = 2_342_300;
        let hard = |wanted| Ask { cost: cost(giant, false, 3), hard_giant: true, ..ask(giant, wanted) };
        let one_running = Load { running: 1, giants: 1, urgent: 1, reserved: 4_000 * MIB, settled: 4_000 * MIB, giant_reserved: 4_000 * MIB, ..Load::default() };
        assert_eq!(admit_why(&hard(true), &one_running, &room(30_000 * MIB)).1, "a cold giant runs: one at a time in the game");
        let small = Load { running: 1, urgent: 1, reserved: 610 * MIB, ..Load::default() };
        assert_eq!(admit(&Ask { cost: cost(giant, false, 3), ..ask(giant, true) }, &small, &room(7_178 * MIB)), Verdict::Urgent, "play-91b: the curve fallback without the hard rule");
        assert!(admit_why(&hard(true), &small, &room(7_178 * MIB)).1.starts_with("a cold giant:"), "its corrected estimate does not fit 7,178 MiB");
        let fits = cost(giant, false, 3) + GB + 610 * MIB;
        assert_eq!(admit(&hard(true), &small, &room(fits)), Verdict::Urgent);
        assert_eq!(admit(&hard(true), &small, &Room { title_fall: 20 * MIB, ..room(fits) }), Verdict::Hold, "the title's growth is kept for a waited-for giant too");
        assert_eq!(admit(&hard(true), &small, &Room { compile_rise: 20 * MIB, ..room(fits) }), Verdict::Hold, "and the compiles' growth");
    }

    /// play-101: FFX_FSR4_PASS_14 in the giant helper (4,963 KB, reserved at 7,810 MiB) held 500 s
    /// with ~10 GB available, ~2 GB reserved and the compile heap rising 57 to 258 MB/s. Its child
    /// is held to the reservation and ended when the memory runs short, so a helper's compile keeps
    /// no headroom for either growth; the one-giant rule and the floor stay.
    #[test]
    fn a_giant_helpers_compile_keeps_no_growth_headroom() {
        const MIB: u64 = 1 << 20;
        let len = 4_963 << 10;
        let helper = |helper| Ask { cost: 7_810 * MIB, hard_giant: true, helper, ..ask(len, true) };
        let load = Load { running: 2, urgent: 2, reserved: 1_000 * MIB, ..Load::default() };
        let growing = Room { compile_rise: 64 * MIB, title_fall: 20 * MIB, ..room(10_000 * MIB) };
        assert!(admit_why(&helper(false), &load, &growing).1.starts_with("a cold giant:"), "in this process the rise holds it");
        assert_eq!(admit(&helper(true), &load, &growing), Verdict::Urgent, "in a helper it starts");
        assert_eq!(admit(&helper(true), &load, &Room { compile_rise: 64 * MIB, ..room(9_000 * MIB) }), Verdict::Hold, "the estimate and 1 GB must still fit");
        let giant_running = Load { giants: 1, giant_reserved: 3_000 * MIB, ..load };
        assert_eq!(admit_why(&helper(true), &giant_running, &room(30_000 * MIB)).1, "a cold giant runs: one at a time in the game");
    }

    fn ask(dxil_len: usize, wanted: bool) -> Ask {
        Ask { dxil_len, wanted, most: 20, cap: None, cost: peak(dxil_len), small_lane: false, hard_giant: false, helper: false }
    }

    /// play-89 at ~150 s: 4 live compiles running, 3 of them cold giants (978 + 972 + 1,274 MiB of
    /// estimate), 3,780 MiB reserved, 555 MiB of it settled, 3,927 MiB available. A small live
    /// compile saw 702 MiB of spare and waited; in its lane it sees the giants as the memory
    /// available shows them. A live cold giant waits on its corrected estimate.
    #[test]
    fn small_live_compiles_never_wait_on_giant_reservations() {
        const MIB: u64 = 1 << 20;
        let load = Load { running: 4, giants: 3, urgent: 4, reserved: 3_780 * MIB, settled: 555 * MIB, giant_reserved: 3_224 * MIB, ..Load::default() };
        let available = room(3_927 * MIB);
        let small = |len| Ask { cost: cost(len, false, 3), ..ask(len, true) };
        assert_eq!(admit(&small(39 << 10), &load, &available), Verdict::Hold, "the old rule: 702 MiB of spare");
        assert_eq!(admit(&Ask { small_lane: true, ..small(39 << 10) }, &load, &available), Verdict::Urgent);
        assert_eq!(admit(&Ask { small_lane: true, ..small(200 << 10) }, &load, &available), Verdict::Urgent);
        assert_eq!(admit(&Ask { small_lane: true, ..small(39 << 10) }, &load, &room(1_100 * MIB)), Verdict::Hold, "the floor stays");
        let giant = 1_127_196;
        assert_eq!(admit(&Ask { cost: cost(giant, false, 3), ..ask(giant, true) }, &load, &available), Verdict::Hold, "a cold giant waits on its corrected estimate");
        let alone = Load { running: 1, urgent: 1, reserved: 200 * MIB, ..Load::default() };
        assert_eq!(admit(&Ask { cost: cost(giant, false, 3), ..ask(giant, true) }, &alone, &room(4_200 * MIB)), Verdict::Urgent, "one at a time when it fits");
        let kept = Ask { cost: cost(2_342_300, true, 3), small_lane: true, ..ask(2_342_300, true) };
        assert_eq!(admit(&kept, &load, &available), Verdict::Urgent, "a kept blob loads in the small lane");
    }

    /// play-91b: one small live compile running (610 MiB reserved), 7,178 MiB available, a live
    /// cold giant of 2,342,300 B asking. At its corrected 7.9 GB it never fits, and it waited
    /// 250 s; with no other cold giant running it starts on the curve. Beside a running cold
    /// giant it waits on the corrected estimate.
    #[test]
    fn a_live_giant_alone_falls_back_to_the_curve() {
        const MIB: u64 = 1 << 20;
        let giant = 2_342_300;
        let ask = Ask { cost: cost(giant, false, 3), ..ask(giant, true) };
        let load = Load { running: 1, urgent: 1, reserved: 610 * MIB, ..Load::default() };
        let available = room(7_178 * MIB);
        assert_eq!(admit(&ask, &load, &available), Verdict::Urgent);
        let beside = Load { running: 2, giants: 1, reserved: 3_000 * MIB, giant_reserved: 2_390 * MIB, ..load };
        assert_eq!(admit(&ask, &beside, &room(9_000 * MIB)), Verdict::Hold, "one cold giant at a time while its corrected estimate does not fit");
        assert_eq!(admit(&ask, &load, &room(3_000 * MIB)), Verdict::Hold, "the curve and the floor still have to fit");
    }

    #[test]
    fn corrected_costs() {
        const MIB: u64 = 1 << 20;
        assert_eq!(cost(64 << 10, false, 3), peak(64 << 10));
        assert_eq!(cost(1_127_196, false, 1), peak(1_127_196));
        assert_eq!(cost(1_127_196, false, 3), 3 * peak(1_127_196));
        assert!(cost(1_509_056, false, 3) >= 3_537 * MIB, "play-89's 1,509,056 B peaked at +3,537 MiB");
        assert_eq!(cost(2_342_300, false, 3), 2_342_300 * 3380, "capped at the unmeasured-size guard");
        assert_eq!(cost(5 << 20, false, 3), peak(5 << 20));
        assert!(cost(2_342_300, true, 3) <= 2 << 30 && cost(2_342_300, true, 3) >= 223 * MIB, "a kept blob loaded at +223 MiB");
    }

    /// play-88: a background giant's 8,279 MiB estimate in flight, 3.2 GB available, a 1 MB
    /// compile a skipped draw waits for asking. It waits for the giant (the plain rule), and no
    /// background compile starts ahead of it; once it started, no new background giant starts
    /// until live demand has been quiet for `LIVE_QUIET`.
    #[test]
    fn live_demand_goes_first_and_holds_background_giants() {
        const MIB: u64 = 1 << 20;
        let (live, giant, small) = (1 << 20, 4 << 20, 64 << 10);
        let in_flight = Load { running: 1, giants: 1, reserved: 8_279 * MIB, ..Load::default() };
        let tight = room(3_200 * MIB);
        assert_eq!(admit(&ask(live, true), &in_flight, &tight), Verdict::Hold, "the live compile waits for the giant");
        let waiting = Load { starving: 1, live_recent: true, ..in_flight };
        assert_eq!(admit(&ask(giant, false), &waiting, &tight), Verdict::Hold);
        assert_eq!(admit(&ask(small, false), &waiting, &tight), Verdict::Hold);
        let freed = room(24 * GB);
        let after = Load { running: 0, giants: 0, reserved: 0, ..waiting };
        assert_eq!(admit(&ask(live, true), &after, &freed), Verdict::Urgent, "the live compile starts when the giant is done");
        assert_eq!(admit(&ask(giant, false), &after, &freed), Verdict::Hold, "no background giant while a live compile waits");
        assert_eq!(admit(&ask(small, false), &after, &freed), Verdict::Hold, "no background compile ahead of a waiting live one");
        let started = Load { running: 1, urgent: 1, starving: 0, reserved: peak(live), ..after };
        assert_eq!(admit(&ask(small, false), &started, &freed), Verdict::Background, "small background work goes on beside live demand");
        assert_eq!(admit(&ask(giant, false), &started, &freed), Verdict::Hold, "no new background giant within LIVE_QUIET of live demand");
        assert_eq!(admit(&ask(giant, true), &started, &freed), Verdict::Urgent, "a live giant is not held");
        let quiet = Load { live_recent: false, ..started };
        assert_eq!(admit(&ask(giant, false), &quiet, &freed), Verdict::Background, "background giants run while live demand is idle");
        let growing = Room { commit_fall: FAST_FALL, ..freed };
        assert_eq!(admit(&ask(giant, false), &quiet, &growing), Verdict::Hold, "nor while the title's commit grows fast");
    }

    /// play-88b: a background giant in flight since 60 s with 8,641 MiB of estimate, past its
    /// expected time from 130 s, and 7,572 MiB available at 153 s. A 39 KB live compile counts it
    /// as the memory available shows it; background work still counts its whole estimate.
    #[test]
    fn live_compiles_count_a_settled_giant_as_measured() {
        const MIB: u64 = 1 << 20;
        let (live, small) = (39 << 10, 64 << 10);
        let fresh = Load { running: 1, giants: 1, reserved: 8_641 * MIB, starving: 1, live_recent: true, ..Load::default() };
        let available = room(7_572 * MIB);
        assert_eq!(admit(&ask(live, true), &fresh, &available), Verdict::Hold, "a giant within its expected time keeps its estimate");
        let settled = Load { settled: 8_641 * MIB, ..fresh };
        assert_eq!(admit(&ask(live, true), &settled, &available), Verdict::Urgent);
        assert_eq!(admit(&ask(small, false), &Load { starving: 0, ..settled }, &available), Verdict::Background, "small background work too (XCL_SETTLE_BACKGROUND)");
        assert_eq!(admit(&ask(live, true), &settled, &room(976 * MIB)), Verdict::Hold, "the floor stays");
        let commit = Room { commit_room: 2 * GB, ..available };
        assert_eq!(admit(&ask(live, true), &settled, &commit), Verdict::Urgent, "the commit room shows it too");
        let growing = Room { trend_fall: 51 * MIB, ..available };
        assert_eq!(admit(&ask(live, true), &settled, &growing), Verdict::Urgent, "while the memory falls too (XCL_SETTLE_OVERDUE)");
        assert_eq!(admit(&ask(live, true), &settled, &Room { available: 976 * MIB, ..growing }), Verdict::Hold, "the floor stays");
        assert_eq!(admit(&ask(live, true), &settled, &Room { trend_fall: SETTLE_FALL - 1, ..available }), Verdict::Urgent);
    }

    #[test]
    fn peaks_stay_above_the_measured_costs() {
        const MB: u64 = 1 << 20;
        // (bytes of DXIL, the private peak measured in MB)
        for (len, measured) in [(533_812, 307), (660_536, 425), (1_717_176, 1067), (1_945_712, 1217), (3_241_524, 2897), (3_515_708, 4468)] {
            let estimate = peak_at(len, SMALL_PEAK_PER_BYTE);
            let (least, most) = if len < GIANT_DXIL { (measured * MB * 115 / 100, measured * MB * 3 / 2) } else { (measured * MB * 125 / 100, measured * MB * 9 / 4) };
            assert!(estimate >= least, "{len} B: {} MB estimated, {measured} MB measured", estimate / MB);
            assert!(estimate <= most, "{len} B: {} MB estimated is no longer near the {measured} MB measured", estimate / MB);
        }
        let at = |len| peak_at(len, SMALL_PEAK_PER_BYTE);
        assert_eq!(at(1000), 128 << 20);
        assert_eq!(at(512 << 10), (512 << 10) * 800);
        assert_eq!(at(GIANT_DXIL), 911 << 20);
        assert_eq!(at(6 << 20), 6 * 3380 << 20);
        // never less for more DXIL, whatever a byte under the giant line is expected to take
        let sizes = [0, 1000, 256 << 10, GIANT_DXIL - 1, GIANT_DXIL, 2 << 20, (4 << 20) - 1, 4 << 20, 27 << 20];
        for small in [256, SMALL_PEAK_PER_BYTE, 1024, 4096] {
            assert!(sizes.windows(2).all(|w| peak_at(w[0], small) <= peak_at(w[1], small)), "{small} B a byte");
        }
        assert_eq!(peak_at(700 << 10, 1024), 700 << 20, "what it was before");
        assert_eq!(peak_at(900 << 10, 4096), 2 << 30, "never over the least giant's");
        assert_eq!(peak_at(GIANT_DXIL, 4096), 2 << 30, "the small override cannot make the first giant cheaper");
    }

    #[test]
    fn the_giant_estimate_covers_isolated_rtx_peaks_without_the_old_overestimate() {
        const MIB: u64 = 1 << 20;
        for (bytes, observed, headroom) in [(1_208_312, 953u64, 110), (1_588_808, 1067, 125), (1_727_332, 1777, 125), (3_240_760, 3047, 125), (3_515_708, 4468, 125)] {
            let estimate = peak_at(bytes, SMALL_PEAK_PER_BYTE);
            assert_eq!(estimate, (observed * headroom).div_ceil(100) * MIB, "{bytes} B");
        }
        let giant = ask(3_240_760, true);
        assert_eq!(admit(&giant, &Load::default(), &room(4_833 * MIB)), Verdict::Urgent);
        assert_eq!(admit(&giant, &Load::default(), &room(4_832 * MIB)), Verdict::Hold);
    }

    /// A compile is admitted when its expected peak fits the memory available less the floor,
    /// and held – not refused – when it does not.
    #[test]
    fn a_compile_starts_when_its_peak_fits_above_the_floor() {
        let len = 512 << 10;
        let cost = peak(len);
        assert_eq!(admit(&ask(len, false), &Load::default(), &room(cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, false), &Load::default(), &room(cost + GB - 1)), Verdict::Hold);
        assert_eq!(admit(&ask(len, true), &Load::default(), &room(cost + GB)), Verdict::Urgent);
        assert_eq!(admit(&ask(len, true), &Load::default(), &room(cost + GB - 1)), Verdict::Hold);
    }

    /// All the expected peaks of the compiles running count against the memory available.
    #[test]
    fn running_compiles_count_against_the_memory() {
        let len = 512 << 10;
        let cost = peak(len);
        let load = Load { running: 2, reserved: 2 * cost, ..Load::default() };
        assert_eq!(admit(&ask(len, false), &load, &room(3 * cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(len, false), &load, &room(3 * cost + GB - 1)), Verdict::Hold);
    }

    /// The floor of a background compile is its own: a run keeps the 1 GB its watchdog stops
    /// at, a warm-up 6 GB – and a compile a bind waits for only ever its 1 GB.
    #[test]
    fn a_background_compile_keeps_its_own_floor() {
        let len = 512 << 10;
        let cost = peak(len);
        let warm = Room { available: cost + 2 * GB, floor: 6 * GB, urgent_floor: GB, commit: 0, commit_room: u64::MAX, fall: 0, commit_fall: 0, trend_fall: 0, title_fall: 0, compile_rise: 0 };
        assert_eq!(admit(&ask(len, false), &Load::default(), &warm), Verdict::Hold);
        assert_eq!(admit(&ask(len, true), &Load::default(), &warm), Verdict::Urgent);
        assert_eq!(admit(&ask(len, false), &Load::default(), &Room { floor: GB, ..warm }), Verdict::Background);
    }

    #[test]
    fn background_floor_cannot_drop_below_the_emergency_floor() {
        let below = Room { floor: 0, ..room(GB) };
        assert_eq!(admit(&ask(121_004, false), &Load::default(), &below), Verdict::Hold);
        assert_eq!(admit(&ask(121_004, true), &Load::default(), &room(0)), Verdict::Hold, "no RAM sample");
    }

    /// No count of giants at once: a giant starts whenever its peak fits in the memory available
    /// less the running peaks and the floor, as any compile does.
    #[test]
    fn giants_start_whenever_their_peak_fits() {
        let giant = 2 << 20;
        let cost = peak(giant);
        let three = Load { running: 3, giants: 3, reserved: 3 * cost, ..Load::default() };
        assert_eq!(admit(&ask(giant, false), &three, &room(4 * cost + GB)), Verdict::Background, "a fourth giant that fits");
        assert_eq!(admit(&ask(giant, false), &three, &room(4 * cost + GB - 1)), Verdict::Hold, "one that does not");
        assert_eq!(admit(&ask(giant, true), &three, &room(4 * cost + GB)), Verdict::Urgent, "a bind's giant");
        assert_eq!(admit(&ask(giant, true), &three, &room(4 * cost + GB - 1)), Verdict::Hold);
    }

    /// The commit headroom (the commit limit less the commit charge) holds the running peaks, the
    /// compile's own and 1 GB, whatever the physical memory says, for a bind's compile too.
    #[test]
    fn a_compile_needs_commit_headroom_for_its_peak() {
        let small = 512 << 10;
        let cost = peak(small);
        let load = Load { running: 1, reserved: cost, ..Load::default() };
        let with = |commit_room| Room { commit_room, ..room(64 * GB) };
        assert_eq!(admit(&ask(small, false), &load, &with(2 * cost + GB)), Verdict::Background);
        assert_eq!(admit(&ask(small, false), &load, &with(2 * cost + GB - 1)), Verdict::Hold);
        assert_eq!(admit(&ask(small, true), &load, &with(2 * cost + GB)), Verdict::Urgent);
        assert_eq!(admit(&ask(small, true), &load, &with(2 * cost + GB - 1)), Verdict::Hold, "a bind waits for commit too");
    }

    /// A waiting warm-up giant keeps smaller background compiles from taking the memory it needs,
    /// but not when it could not fit even with nothing else running.
    #[test]
    fn a_waiting_giant_holds_back_smaller_compiles() {
        let (giant, small) = (2 << 20, 512 << 10);
        let (gcost, scost) = (peak(giant), peak(small));
        let load = Load { running: 1, reserved: scost, giant_waiting: gcost, ..Load::default() };
        let tight = room(scost + GB + gcost + scost - 1);
        assert_eq!(admit(&ask(small, false), &load, &tight), Verdict::Hold);
        assert_eq!(admit(&ask(small, false), &load, &room(scost + GB + gcost + scost)), Verdict::Background);
        assert_eq!(admit(&ask(small, false), &Load { giant_waiting: 0, ..load }, &tight), Verdict::Background, "no giant waits");
        assert_eq!(admit(&ask(small, false), &Load { giant_waiting: 64 * GB, ..load }, &room(8 * GB)), Verdict::Background, "the giant could never fit");
        assert_eq!(admit(&ask(small, true), &load, &tight), Verdict::Urgent, "a bind is not held");
    }

    /// Background compiles hold back while a compile a bind waits for waits for memory, and stay
    /// within `most` at once and the private-memory cap; the ones binds wait for do not count
    /// towards `most`.
    #[test]
    fn background_compiles_give_way_and_stay_within_their_limits() {
        let len = 300 << 10;
        let plenty = room(64 * GB);
        assert_eq!(admit(&ask(len, false), &Load { starving: 1, ..Load::default() }, &plenty), Verdict::Hold);
        let busy = Load { running: 4, urgent: 2, ..Load::default() };
        assert_eq!(admit(&Ask { most: 2, ..ask(len, false) }, &busy, &plenty), Verdict::Hold);
        assert_eq!(admit(&Ask { most: 3, ..ask(len, false) }, &busy, &plenty), Verdict::Background);
        assert_eq!(admit(&Ask { most: 2, ..ask(len, true) }, &busy, &plenty), Verdict::Urgent);
        let capped = Ask { cap: Some(4 * GB), ..ask(len, false) };
        assert_eq!(admit(&capped, &Load::default(), &Room { commit: 4 * GB, ..plenty }), Verdict::Hold);
        assert_eq!(admit(&capped, &Load::default(), &Room { commit: GB, ..plenty }), Verdict::Background);
    }

    fn drain(b: &mut Backlog<&'static str>, workers: usize) -> Vec<&'static str> {
        std::iter::from_fn(|| {
            let (p, lane) = b.pop(workers, &[], |_| true, |_| true, |_| false)?;
            b.done(lane);
            Some(p)
        })
        .collect()
    }

    /// Heavy pipelines first, the biggest of them first; then the ones an earlier run used, then
    /// compute, then graphics, each in the order made.
    #[test]
    fn the_backlog_goes_heavy_known_compute_graphics() {
        let mut b = Backlog::default();
        b.push(100, false, false, 0, None, "g1");
        b.push(100, false, true, 0, None, "c1");
        b.push(HEAVY_SHADER_BYTES, false, true, 0, None, "heavy-small");
        b.push(100, true, false, 0, None, "k1");
        b.push(100, false, false, 0, None, "g2");
        b.push(4 * HEAVY_SHADER_BYTES, false, true, 0, None, "heavy-big");
        b.push(100, true, true, 0, None, "k2");
        b.push(100, false, true, 0, None, "c2");
        assert_eq!(b.waiting(), (8, 0));
        // (of each lane the newest first)
        assert_eq!(drain(&mut b, 8), ["heavy-big", "heavy-small", "k2", "k1", "c2", "c1", "g2", "g1"]);
        assert_eq!(b.waiting(), (0, 0));
    }

    /// Graphics pipelines without colour targets go before the other graphics ones, behind the
    /// known and the compute ones; one of another lane stays where it is.
    #[test]
    fn depth_only_pipelines_go_before_the_other_graphics_ones() {
        let mut b = Backlog::default();
        let d1 = b.push(100, false, false, 0, None, "d1");
        b.push(100, false, false, 0, None, "g1");
        b.push(100, false, true, 0, None, "c1");
        let d2 = b.push(100, false, false, 0, None, "d2");
        let k1 = b.push(100, true, false, 0, None, "k1");
        b.push(100, false, false, 0, None, "g2");
        for n in [d1, d2, k1, 0] {
            b.depth_only(n);
        }
        assert_eq!(drain(&mut b, 8), ["k1", "c1", "d2", "d1", "g2", "g1"]);
        assert_eq!(b.waiting(), (0, 0));
    }

    /// play-101: ray passes built with every ray a miss go behind every other pipeline waiting,
    /// heavy, slow and recently bound ones included, and are still built once nothing else waits.
    #[test]
    fn ray_passes_built_with_every_ray_a_miss_go_last() {
        for mut b in [Backlog::default(), Backlog::drawn_first()] {
            let r1 = b.push(100, true, true, 7, None, "r1");
            b.push(100, false, false, 0, None, "g1");
            let r2 = b.push(HEAVY_SHADER_BYTES, false, true, 0, None, "r2");
            b.push(100, true, true, 0, None, "k1");
            let r3 = b.push(100, true, true, 0, Some(900), "r3");
            for n in [r1, r2, r3, 99] {
                b.rt_pass(n);
            }
            assert_eq!(b.waiting(), (5, 0));
            let mut taken = Vec::new();
            while let Some((item, lane)) = b.pop(8, &[7], |_| true, |_| true, |_| false) {
                taken.push((item, lane == Lane::RtPass));
                b.done(lane);
            }
            assert_eq!(taken, [("k1", false), ("g1", false), ("r3", true), ("r2", true), ("r1", true)]);
            assert_eq!(b.waiting(), (0, 0));
        }
    }

    /// Only a share of the workers builds heavy pipelines while small ones wait: the rest go on
    /// with those; once nothing else waits, every worker takes heavy ones.
    #[test]
    fn heavy_pipelines_take_a_share_of_the_workers() {
        let mut b = Backlog::default();
        for name in ["h1", "h2", "h3", "h4"] {
            b.push(HEAVY_SHADER_BYTES, false, true, 0, None, name);
        }
        b.push(100, false, true, 0, None, "c1");
        b.push(100, false, true, 0, None, "c2");
        let workers = 8;
        assert_eq!(heavy_share(workers), 2);
        let take = |b: &mut Backlog<&'static str>| b.pop(workers, &[], |_| true, |_| true, |_| false).unwrap();
        assert_eq!(take(&mut b), ("h1", Lane::Heavy));
        assert_eq!(take(&mut b), ("h2", Lane::Heavy));
        assert_eq!(take(&mut b), ("c2", Lane::Compute), "two workers hold heavy ones: the third takes a small one");
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b), ("h3", Lane::Heavy), "a heavy one done frees its place");
        assert_eq!(take(&mut b), ("c1", Lane::Compute));
        assert_eq!(take(&mut b), ("h4", Lane::Heavy), "nothing else waits: no share");
        assert_eq!(heavy_share(1), 1);
        assert_eq!(heavy_share(22), 6);
    }

    /// A compile that does not fit is parked and the worker goes on: nothing waits behind it.
    /// It comes back ahead of everything once it fits, the biggest that fits first.
    #[test]
    fn a_parked_build_holds_nothing_up_and_returns_when_it_fits() {
        let mut b = Backlog::default();
        b.push(100, false, true, 0, None, "c1");
        b.push(100, false, false, 0, None, "g1");
        b.park(3 << 20, 0, 0, None, "giant");
        b.park(300 << 10, 0, 0, None, "mid");
        b.park(600 << 10, 0, 0, None, "big");
        assert_eq!(b.waiting(), (5, 3));
        assert!(b.has_parked());
        assert_eq!(b.least_parked(), 300 << 10);
        let none_fit = |b: &mut Backlog<&'static str>| b.pop(4, &[], |_| true, |_| false, |_| false).map(|(p, _)| p);
        assert_eq!(none_fit(&mut b), Some("c1"), "no memory for the parked ones: the others go on");
        assert_eq!(b.pop(4, &[], |_| true, |len| len < 1 << 20, |_| false), Some(("big", Lane::Parked)), "the biggest that fits");
        assert_eq!(b.pop(4, &[], |_| true, |_| true, |_| false), Some(("giant", Lane::Parked)));
        assert_eq!(b.pop(4, &[], |_| true, |len| len < 100 << 10, |_| false), Some(("g1", Lane::Graphics)));
        assert_eq!(none_fit(&mut b), None, "only a parked one left, and no memory for it: the worker waits");
        assert_eq!(b.pop(4, &[], |_| true, |_| true, |_| false), Some(("mid", Lane::Parked)));
        assert!(!b.has_parked());
        assert_eq!(b.least_parked(), 0);
    }

    /// A parked build a bind took over (or whose pipeline the title released) is dropped: it
    /// does not come back to a background worker.
    #[test]
    fn a_parked_build_something_else_took_is_dropped() {
        let mut b = Backlog::default();
        b.park(300 << 10, 0, 0, None, "taken by a bind");
        b.park(600 << 10, 0, 0, None, "still waiting");
        assert_eq!(b.pop(4, &[], |p| *p != "taken by a bind", |_| false, |_| false), None);
        assert_eq!(b.waiting(), (1, 1));
        assert_eq!(b.pop(4, &[], |_| true, |_| true, |_| false), Some(("still waiting", Lane::Parked)));
    }

    #[test]
    fn background_workers_are_one_for_each_core_and_leave_two_threads() {
        assert_eq!(background_workers(24, 12), 12);
        assert_eq!(background_workers(16, 8), 8);
        assert_eq!(background_workers(8, 8), 6);
        assert_eq!(background_workers(8, 4), 4);
        assert_eq!(background_workers(4, 4), 2);
        assert_eq!(background_workers(2, 1), 2);
        assert_eq!(background_workers(128, 64), 32);
        assert_eq!((least_workers(12), least_workers(8), least_workers(4), least_workers(1)), (3, 2, 2, 1));
    }

    /// Unbound pipelines are released after the idle limit, and before it the oldest beyond
    /// the number kept.
    #[test]
    fn idle_pipelines_go_by_age_and_by_number() {
        let mut built = vec![(1, 1_000), (2, 29_000), (3, 30_000), (4, 31_000), (5, 5_000)];
        let mut gone = idle_to_release(&mut built, 35_000, 30_000, 100);
        gone.sort();
        assert_eq!(gone, [1, 5], "30 s old or more");
        let mut gone = idle_to_release(&mut built, 35_000, 30_000, 2);
        gone.sort();
        assert_eq!(gone, [1, 2, 5], "and the oldest beyond the two newest");
        assert!(idle_to_release(&mut built, 30_500, 30_000, 5).is_empty());
        assert_eq!(idle_to_release(&mut built, 35_000, 30_000, 0).len(), 5);
    }

    #[test]
    fn the_pace_starts_low_and_rises_a_worker_a_step_while_the_title_is_idle() {
        let mut p = Pace::new(12, 12);
        // (the first steps still count as loading: a title starts by loading)
        for _ in 0..LOADING_HOLD - 1 {
            let t = p.step(1500, false);
            assert!(t.low && t.workers <= 5, "{t:?}");
        }
        let seen: Vec<Tempo> = (0..10).map(|_| p.step(1500, false)).collect();
        assert!(seen.iter().all(|t| !t.low));
        assert_eq!(seen.iter().map(|t| t.workers).collect::<Vec<_>>(), [6, 7, 8, 9, 10, 10, 10, 10, 10, 10], "up to the cores the title leaves");
    }

    #[test]
    fn the_pace_drops_at_once_when_the_title_takes_the_cores() {
        let mut p = Pace::new(12, 12);
        for _ in 0..20 {
            p.step(1000, false);
        }
        assert_eq!(p.step(1000, false).workers, 11);
        // the title's threads take six cores: six workers, at once; their priority stays
        assert_eq!(p.step(5200, false), Tempo { workers: 6, low: false });
        // … and all of them: never under the least
        assert_eq!(p.step(14_000, false), Tempo { workers: 3, low: false });
        assert_eq!(p.step(0, false), Tempo { workers: 4, low: false }, "back a worker a step");
    }

    #[test]
    fn the_pace_halves_and_lowers_the_workers_while_the_title_loads() {
        let mut p = Pace::new(12, 12);
        for _ in 0..20 {
            p.step(2000, false);
        }
        assert_eq!(p.step(2000, false), Tempo { workers: 10, low: false });
        assert_eq!(p.step(2000, true), Tempo { workers: 5, low: true });
        assert_eq!(p.step(6000, true), Tempo { workers: 3, low: true });
        // it still counts as loading for a few steps after the storage went quiet
        for _ in 0..LOADING_HOLD - 1 {
            assert!(p.step(2000, false).low);
        }
        assert_eq!(p.step(2000, false), Tempo { workers: 6, low: false });
        // a small pool keeps two workers going, a pool of one its one
        let mut small = Pace::new(4, 4);
        assert_eq!(small.step(4000, true).workers, 2);
        assert_eq!(Pace::new(1, 4).step(4000, true).workers, 1);
        // … and no pace goes beyond its pool
        let mut wide = Pace::new(6, 16);
        for _ in 0..20 {
            wide.step(0, false);
        }
        assert_eq!(wide.step(0, false).workers, 6);
    }

    #[test]
    fn the_headroom_is_its_least_while_the_memory_is_flat_or_shrinks() {
        let mut h = Headroom::default();
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        for s in 0..120u64 {
            h.note(s * 500, (10 << 30) - s * (8 << 20), true);
        }
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        assert_eq!(h.floor(0, 6 << 30, 6 << 30, false), 6 << 30, "the least is the caller's");
    }

    #[test]
    fn the_headroom_is_twice_the_biggest_rise_within_ten_seconds() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        // 3 GB within 6 s, as a world load commits
        for s in 0..=12u64 {
            h.note(s * 500, 10 * gb + s * gb / 4, true);
        }
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 6 * gb);
        // a slow rise does not add up beyond its ten seconds: 100 MB a second is 1 GB, under the least
        let mut slow = Headroom::default();
        for s in 0..300u64 {
            slow.note(s * 500, 10 * gb + s * (50 << 20), true);
        }
        assert_eq!(slow.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        assert_eq!(slow.floor(0, gb, gb, false), 2 * (20 * (50u64 << 20)), "twice the rise of the 20 steps in the span");
        assert_eq!(slow.floor(gb, gb, gb, false), gb + 2 * (20 * (50u64 << 20)), "above the line a run is stopped at");
    }

    #[test]
    fn the_headroom_forgets_a_rise_a_minute_after_its_ten_seconds() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        h.note(0, 10 * gb, true);
        h.note(2_000, 14 * gb, true);
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 8 * gb);
        for s in 1..=160u64 {
            let now = 2_000 + s * 500;
            h.note(now, 14 * gb, true);
            // (the rise was within the span until 9.5 s after it)
            let kept = now <= 2_000 + RISE_SPAN_MS - 500 + RISE_KEPT_MS;
            assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), if kept { 8 * gb } else { MIN_HEADROOM }, "at {now} ms");
        }
    }

    #[test]
    fn the_headroom_sees_a_rise_across_a_pause_in_the_samples() {
        let mut h = Headroom::default();
        let gb = 1u64 << 30;
        h.note(0, 10 * gb, true);
        h.note(500, 10 * gb, true);
        // nothing asked for 40 s, and the process is 5 GB bigger
        h.note(40_500, 15 * gb, true);
        assert_eq!(h.floor(0, MIN_HEADROOM, MIN_HEADROOM, false), 10 * gb);
    }

    #[test]
    fn the_pipelines_of_a_group_bound_lately_go_before_the_others() {
        let mut b = Backlog::default();
        let (shadow, sky, post) = (0x51, 0x52, 0x53);
        b.push(100, false, false, shadow, None, "shadow 1");
        b.push(100, true, false, sky, None, "sky 1, known");
        b.push(100, false, true, post, None, "post 1");
        b.push(HEAVY_SHADER_BYTES, false, true, post, None, "post heavy");
        b.push(100, false, false, shadow, None, "shadow 2");
        b.push(100, false, false, 0, None, "no group");
        assert_eq!(b.waiting(), (6, 0));
        let take = |b: &mut Backlog<&'static str>, recent: &[u64]| b.pop(1, recent, |_| true, |_| true, |_| false).map(|(p, _)| p);
        // post was bound last, shadow before it: post's first (its newest first), then shadow's
        let recent = [post, shadow];
        assert_eq!(b.pop(1, &recent, |_| true, |_| true, |_| false), Some(("post heavy", Lane::Heavy)));
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b, &recent), Some("post 1"));
        assert_eq!(take(&mut b, &recent), Some("shadow 2"));
        // a bind of another group since: that one first; an unknown group changes nothing
        assert_eq!(take(&mut b, &[0x99, sky, post, shadow]), Some("sky 1, known"));
        assert_eq!(take(&mut b, &recent), Some("shadow 1"));
        assert_eq!(b.waiting(), (1, 0));
        // nothing of those groups is left: the rest in its own order, and nothing is given twice
        assert_eq!(take(&mut b, &recent), Some("no group"));
        assert_eq!(take(&mut b, &recent), None);
        assert_eq!(b.waiting(), (0, 0));
    }

    #[test]
    fn a_pipeline_taken_through_its_lane_is_not_given_again_through_its_group() {
        let mut b = Backlog::default();
        b.push(100, false, false, 0x51, None, "a");
        b.push(HEAVY_SHADER_BYTES, false, true, 0x51, None, "heavy");
        b.push(100, false, true, 0x51, None, "c");
        let take = |b: &mut Backlog<&'static str>, recent: &[u64]| b.pop(4, recent, |_| true, |_| true, |_| false).map(|(p, _)| p);
        assert_eq!(take(&mut b, &[]), Some("heavy"));
        assert_eq!(take(&mut b, &[]), Some("c"));
        assert_eq!(b.waiting(), (1, 0));
        assert_eq!(take(&mut b, &[0x51]), Some("a"));
        assert_eq!(take(&mut b, &[0x51]), None);
        // … nor the other way round, and heavy ones taken through their group leave the count right
        b.push(HEAVY_SHADER_BYTES, false, true, 0x52, None, "h1");
        b.push(HEAVY_SHADER_BYTES * 2, false, true, 0x52, None, "h2");
        b.push(100, false, false, 0, None, "g");
        assert_eq!(take(&mut b, &[0x52]), Some("h2"));
        assert_eq!(take(&mut b, &[0x52]), Some("h1"));
        b.done(Lane::Heavy);
        b.done(Lane::Heavy);
        assert_eq!(take(&mut b, &[]), Some("g"));
        assert_eq!(take(&mut b, &[]), None);
        assert_eq!(b.waiting(), (0, 0));
    }

    #[test]
    fn while_a_bind_waits_the_headroom_is_the_binds_floor_and_the_growth() {
        let (mut h, gb) = (Headroom::default(), 1u64 << 30);
        // the title loads, its memory flat: a bind that waits lets background compiles go
        // down to the floor of the compile it waits for; none waits any more, and the least is back
        h.note(0, 12 * gb, true);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), MIN_HEADROOM);
        // … but not into a load that takes 3 GB within seconds: twice that stays above the floor
        h.note(4_000, 15 * gb, true);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), 7 * gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), 7 * gb);
        // calm, in the world: the bind's floor again once the rise is forgotten
        for s in 1..=160u64 {
            h.note(4_000 + s * 500, 15 * gb, false);
        }
        assert!(!h.loading());
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, true), gb);
        assert_eq!(h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false), CALM_HEADROOM);
    }

    #[test]
    fn parked_builds_of_a_group_bound_lately_come_back_before_bigger_ones() {
        let mut b = Backlog::default();
        b.park(900 << 10, 0x61, 0, None, "big, another pass");
        b.park(600 << 10, 0x62, 0, None, "this pass");
        b.park(500 << 10, 0x62, 0, None, "this pass, smaller");
        let take = |b: &mut Backlog<&'static str>, recent: &[u64], most: usize| b.pop(4, recent, |_| true, |len| len <= most, |_| false).map(|(p, _)| p);
        assert_eq!(take(&mut b, &[0x62, 0], 1 << 20), Some("this pass"));
        // only what fits: the smaller one of the pass, not the big one of another
        assert_eq!(take(&mut b, &[0x62], 550 << 10), Some("this pass, smaller"));
        assert_eq!(take(&mut b, &[0x62], 550 << 10), None);
        assert_eq!(take(&mut b, &[0x62], 1 << 20), Some("big, another pass"));
        assert!(!b.has_parked());
    }

    #[test]
    fn background_reserves_the_host_cost_of_unbacked_title_memory() {
        const MIB: u64 = 1 << 20;
        let budget = 8192 * MIB;
        let backed = (4095 + 1627 + 715) * MIB;
        let mut h = Headroom::empty();
        for (commit, growth, low, high) in [(15637, 2577, 7800, 8000), (15937, 1963, 7300, 7500)] {
            let reserve = TitleMemory { backed, commit: commit * MIB }.host_reserve(budget);
            h.note(0, (commit - growth / 2) * MIB, false);
            h.note(500, commit * MIB, false);
            let floor = h.title_floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, false, reserve) / MIB;
            assert!((low..=high).contains(&floor), "floor {floor} MiB");
            assert!((4100..=4400).contains(&(reserve / MIB)));
            assert_eq!(h.title_floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, true, reserve), h.floor(1024 * MIB, CALM_HEADROOM, MIN_HEADROOM, true));
            h = Headroom::empty();
        }
    }

    #[test]
    fn the_title_reserve_reaches_the_calm_floor_as_the_budget_fills() {
        let budget = 8 << 30;
        assert_eq!(TitleMemory::default().host_reserve(budget), budget);
        let mut h = Headroom::empty();
        h.note(60_000, 20 << 30, false);
        for backed in [budget - (64 << 20), budget, budget + (1 << 30)] {
            let reserve = TitleMemory { backed, commit: 20 << 30 }.host_reserve(budget);
            assert_eq!(h.title_floor(1 << 30, CALM_HEADROOM, MIN_HEADROOM, false, reserve), CALM_HEADROOM);
        }
        assert_eq!(TitleMemory { backed: 6 << 30, commit: 15 << 30 }.host_reserve(8 << 30), 5 << 30);
        assert_eq!(TitleMemory { backed: 6 << 30, commit: 3 << 30 }.host_reserve(8 << 30), 2 << 30);
    }

    #[test]
    fn passive_promotion_prefers_the_exact_pipeline_without_bypassing_memory() {
        let mut b = Backlog::default();
        let promoted = b.push(100, false, false, 11, None, "promoted");
        b.push(100, true, true, 12, None, "hot");
        b.promote(promoted);
        assert_eq!(b.pop(4, &[12], |_| true, |_| true, |_| false), Some(("promoted", Lane::Graphics)));

        b.park(700 << 10, 11, promoted, None, "promoted after parking");
        b.promote(promoted);
        assert_eq!(b.pop(4, &[12], |_| true, |_| false, |_| false), Some(("hot", Lane::Known)));
        assert_eq!(b.pop(4, &[12], |_| true, |_| false, |_| false), None);
        assert_eq!(b.pop(4, &[12], |_| true, |_| true, |_| false), Some(("promoted after parking", Lane::Parked)));
        assert_eq!(b.pop(4, &[12], |_| true, |_| true, |_| false), None);
    }

    #[test]
    fn a_parked_peer_uses_the_bind_window_before_unrelated_work() {
        let mut b = Backlog::default();
        b.park(900 << 10, 10, 1, None, "unrelated");
        b.park(700 << 10, 20, 2, None, "newest group, older pipeline");
        b.park(600 << 10, 30, 3, None, "previous group");
        b.park(800 << 10, 30, 4, None, "previous group, too big");
        b.park(500 << 10, 20, 5, None, "newest group, newer pipeline");
        b.push(64, false, false, 20, None, "unparked");
        let pop = |b: &mut Backlog<_>| b.pop(8, &[20, 30], |_| true, |_| false, |len| len <= 700 << 10);
        assert_eq!(pop(&mut b), Some(("newest group, newer pipeline", Lane::ParkedPeer)));
        assert_eq!(pop(&mut b), Some(("newest group, older pipeline", Lane::ParkedPeer)));
        assert_eq!(pop(&mut b), Some(("previous group", Lane::ParkedPeer)), "no count of peers at once: memory decides");
        assert_eq!(pop(&mut b), Some(("unparked", Lane::Graphics)));
        assert_eq!(pop(&mut b), None);
        b.done(Lane::ParkedPeer);
        assert_eq!(b.parked_peers(&[20, 30]), 1);
        assert_eq!(b.pop(8, &[20, 30], |_| true, |_| true, |_| false), Some(("previous group, too big", Lane::Parked)));
    }

    #[test]
    fn prior_compute_use_orders_only_fitting_work_within_the_current_priority() {
        let mut b = Backlog::default();
        b.park(950 << 10, 10, 4, Some(0), "earliest, does not fit");
        b.park(880 << 10, 10, 3, None, "unknown");
        b.park(600 << 10, 10, 2, Some(2), "second");
        b.park(500 << 10, 10, 1, Some(1), "first");
        b.park(400 << 10, 20, 5, Some(8), "recent group");
        let take = |b: &mut Backlog<_>, recent: &[u64]| b.pop(8, recent, |_| true, |len| len <= 900 << 10, |_| false);
        assert_eq!(take(&mut b, &[20]), Some(("recent group", Lane::Parked)));
        assert_eq!(b.pop(8, &[10], |_| true, |_| false, |len| len <= 900 << 10), Some(("first", Lane::ParkedPeer)));
        assert_eq!(take(&mut b, &[]), Some(("second", Lane::Parked)));
        assert_eq!(take(&mut b, &[]), Some(("unknown", Lane::Parked)));
        assert_eq!(take(&mut b, &[]), None);
    }

    #[test]
    fn steady_memory_releases_the_title_reserve_and_load_restores_it() {
        let mib = 1 << 20;
        let (budget, backed) = (8192 * mib, 5817 * mib);
        let mut reserve = TitleReserve::empty();
        let sample = |r: &mut TitleReserve, t, commit, backed, loading| {
            r.note(t * 1000, TitleMemory { backed, commit: commit * mib }, budget, loading);
        };
        for t in 0..60 {
            sample(&mut reserve, t, 17_300 + (t % 5) * 125, backed, false);
        }
        assert!(!reserve.is_steady());
        sample(&mut reserve, 60, 17_300, backed, false);
        assert!(reserve.is_steady(), "temporary driver noise is not sustained growth");
        assert_eq!(reserve.effective(), 0);
        sample(&mut reserve, 61, 17_300, backed + 64 * mib, false);
        assert!(!reserve.is_steady());
        assert!(reserve.effective() > 0);
        for t in 62..=121 {
            sample(&mut reserve, t, 17_300, backed + 64 * mib, false);
        }
        assert!(reserve.is_steady());
        sample(&mut reserve, 122, 17_300, backed + 64 * mib, true);
        assert!(!reserve.is_steady());
    }

    #[test]
    fn sustained_commit_growth_restores_the_title_reserve_without_storage_activity() {
        let mib = 1 << 20;
        let mut reserve = TitleReserve::empty();
        for t in 0u64..=90 {
            let commit = 17_300 + t.saturating_sub(60) * 8;
            reserve.note(t * 1000, TitleMemory { backed: 5817 * mib, commit: commit * mib }, 8192 * mib, false);
            if t == 60 { assert!(reserve.is_steady()); }
        }
        assert!(!reserve.is_steady());
        assert!(reserve.effective() > 0);
    }

    #[test]
    fn a_commit_step_restores_the_reserve_before_the_median_catches_up() {
        let mib = 1 << 20;
        let mut reserve = TitleReserve::empty();
        for t in 0..=64 {
            let commit = if t <= 60 { 17_300 } else { 17_900 };
            reserve.note(t * 1000, TitleMemory { backed: 5817 * mib, commit: commit * mib }, 8192 * mib, false);
            if t == 60 { assert!(reserve.is_steady()); }
        }
        assert!(reserve.effective() > 0);
    }

    #[test]
    fn the_title_loads_while_its_storage_runs_many_requests_or_bytes() {
        assert!(!loading(0, 0, 500));
        assert!(!loading(1, 1 << 20, 500), "a request now and then is not loading");
        assert!(loading(8, 0, 500));
        assert!(loading(1, 4 << 20, 500), "one big read is");
        assert!(!loading(8, 4 << 20, 1100));
    }

    #[test]
    fn the_headroom_is_smaller_once_the_title_has_not_loaded_for_a_minute() {
        let (mut h, gb) = (Headroom::default(), 1u64 << 30);
        let floor = |h: &Headroom| h.floor(gb, CALM_HEADROOM, MIN_HEADROOM, false);
        // from the start it counts as loading
        assert!(h.loading());
        assert_eq!(floor(&h), MIN_HEADROOM);
        for s in 0..=40u64 {
            h.note(s * 500, 12 * gb, s <= 20);
        }
        assert_eq!(floor(&h), MIN_HEADROOM, "10 s after its storage went quiet");
        for s in 41..=139u64 {
            h.note(s * 500, 12 * gb, false);
        }
        assert!(h.loading(), "59.5 s after");
        h.note(70_000, 12 * gb, false);
        assert!(!h.loading());
        assert_eq!(floor(&h), CALM_HEADROOM, "calm: background compiles go down to 1.5 GB available");
        // the world streams in 0.4 GB within seconds: twice that above the line a run is stopped at
        h.note(72_000, 12 * gb + 4 * gb / 10, false);
        assert_eq!(floor(&h), gb + 2 * (4 * gb / 10));
        // … and the title loads again: the bigger least at once
        h.note(72_500, 12 * gb + 4 * gb / 10, true);
        assert_eq!(floor(&h), MIN_HEADROOM);
        // a calm least over the loading one is the loading one
        assert_eq!(Headroom::default().floor(gb, 8 * gb, 2 * gb, false), 2 * gb);
    }

    #[test]
    fn a_trend_keeps_the_fastest_recent_fall_and_forgets_old_ones() {
        const MIB: u64 = 1 << 20;
        let mut falling = Trend::new();
        for s in 0..=10u64 {
            falling.note(s as f64, 20 * GB - s * 300 * MIB);
        }
        assert_eq!(falling.fall_rate(), 300 * MIB);
        let mut rising = Trend::new();
        for s in 0..=10u64 {
            rising.note(s as f64, GB + s * 100 * MIB);
        }
        assert_eq!(rising.fall_rate(), 0);
        let mut burst = Trend::new();
        for (at, value) in [(0.0, 20 * GB), (2.0, 16 * GB), (4.0, 16 * GB), (8.0, 16 * GB)] {
            burst.note(at, value);
        }
        assert_eq!(burst.fall_rate(), 2 * GB, "a burst stays the rate while in the window");
        burst.note(13.0, 16 * GB);
        assert_eq!(burst.fall_rate(), 0, "and is forgotten 10 s after it");
        let mut dense = Trend::new();
        dense.note(0.0, 10 * GB);
        dense.note(0.2, 5 * GB);
        assert_eq!(dense.fall_rate(), 0, "no span under TREND_SPAN, no sample under half a second apart");
    }

    #[test]
    fn a_compile_keeps_the_title_s_growth_over_its_duration_free_and_giants_pause_in_a_fast_fall() {
        const MIB: u64 = 1 << 20;
        let giant = 2 * MIB as usize;
        assert_eq!(compile_secs(giant), 50);
        assert_eq!(compile_secs(64 << 20), 90);
        assert_eq!(compile_secs(GIANT_DXIL - 1), 5);
        // play-86n at 18:04:03: about 16 GB available, 4 giants reserving 9.8 GB, the title taking
        // about 300 MB/s while it loads
        let load = Load { running: 4, giants: 4, reserved: 9_800 * MIB, ..Load::default() };
        let calm = room(16 * GB);
        assert_eq!(admit(&ask(giant, false), &load, &calm), Verdict::Background, "the old check: it fits at admission");
        let loading = Room { fall: 300 * MIB, ..calm };
        assert_eq!(growth(&loading, giant).0, GROWTH_CAP, "15 GB at 300 MB/s for 50 s, capped");
        assert_eq!(compile_rise(&Room { compile_rise: 258 * MIB, ..calm }, 64 << 20), GROWTH_CAP, "play-101's 258 MB/s rise for 90 s (23 GB), capped as the title's growth");
        assert_eq!(admit(&ask(giant, false), &load, &loading), Verdict::Hold);
        let slow = Room { fall: 20 * MIB, ..calm };
        assert_eq!(admit(&ask(giant, false), &load, &slow), Verdict::Background, "a slow rise over 50 s fits beside it");
        let fast = Room { fall: FAST_FALL, ..room(64 * GB) };
        assert_eq!(admit(&ask(giant, false), &Load::default(), &fast), Verdict::Hold, "no new background giant in a fast fall");
        assert_eq!(admit(&ask(4096, false), &Load::default(), &fast), Verdict::Background, "small ones go on");
        let commit = Room { commit_room: peak(giant) + GB + GB, commit_fall: 100 * MIB, ..room(64 * GB) };
        assert_eq!(admit(&ask(4096, false), &Load::default(), &commit), Verdict::Background);
        assert_eq!(admit(&ask(giant, false), &Load::default(), &commit), Verdict::Hold, "the capped commit growth does not fit beside it");
        assert_eq!(admit(&ask(giant, true), &Load::default(), &commit), Verdict::Urgent, "a waited-for one keeps none");
    }

    #[test]
    fn play_87n_a_compile_the_title_waits_for_starts_at_the_plain_rule_while_it_loads() {
        const MIB: u64 = 1 << 20;
        // play-87n in the world: the title loading at 180 MB/s, 4.25 GB available, 3 compiles
        // running; a 1.1 MB program asked for 7.2 GB of headroom and title threads waited on
        // "driver slot / RAM wait" for 313-390 s
        let live = 1_153_434;
        let load = Load { running: 3, reserved: 1_536 * MIB, ..Load::default() };
        let loading = Room { fall: 180 * MIB, commit_fall: 180 * MIB, ..room(4_352 * MIB) };
        assert_eq!(admit(&ask(live, true), &load, &loading), Verdict::Urgent, "waited for: the plain rule, available less running peaks less 1 GB");
        assert_eq!(admit(&ask(live, false), &load, &loading), Verdict::Hold, "background work keeps the capped headroom");
        let settled = Room { fall: 0, commit_fall: 0, ..loading };
        assert_eq!(admit(&ask(live, false), &load, &settled), Verdict::Background, "and starts once the burst is 10 s gone");
        let at_most = |fall| growth(&Room { fall, ..loading }, live).0;
        assert!(at_most(180 * MIB) == GROWTH_CAP && at_most(u64::MAX) == GROWTH_CAP, "the reserve never passes the cap");
        // a 4.3 MB program (expected 14 GB) never fits 2.8 GB, headroom or not
        assert_eq!(admit(&ask(4_300_000, true), &Load::default(), &Room { fall: 0, ..room(2_867 * MIB) }), Verdict::Hold);
    }
}
