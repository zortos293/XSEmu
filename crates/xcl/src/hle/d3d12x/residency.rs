//! D3D12 backing memory policy for the title's unified memory. Only fence-idle
//! allocations with complete use tracking can be evicted; D3D12 retains their
//! contents. New allocations can instead prefer non-local memory.
//!
//! Two switches ask for memory back, and both need the tracker (an allocation is pinned by the
//! submission that uses it and restamped when that submission's fence signals, so a pass can
//! never take memory the GPU is at work on):
//!
//! * `XCL_VRAM_EVICT=1` evicts under allocation pressure. An allocation that would take local
//!   memory past 90% of the adapter's budget is made room for from the coldest allocations,
//!   down to 80%, and the same check runs after every submission. This only relieves a moment:
//!   the next submission makes resident whatever it uses, so it exists for the allocation that
//!   has nowhere else to go.
//! * `XCL_VRAM_BUDGET=1` runs a pass over the budget once a second, whether or not anything is
//!   being allocated: above `XCL_VRAM_HIGH_PCT` (90) of the local budget the coldest
//!   allocations are evicted, least recently used first and only ones unused for the free
//!   bounds (`free_policy`: `XCL_FREE_AFTER_MS`, `XCL_FREE_AFTER_FRAMES`), until local usage is
//!   back at `XCL_VRAM_LOW_PCT` (85) or nothing else is cold enough. D3D12 keeps the contents
//!   of what it evicts, so nothing is lost; what the pass buys is that memory a title has
//!   streamed in and moved on from stops counting against the budget.
//!
//! Either switch turns the tracker on; with only `XCL_VRAM_BUDGET` set, allocations are tracked
//! and restored but nothing evicts under pressure.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, Result, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, E_OUTOFMEMORY, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use super::free_policy;

const MIB: u64 = 1 << 20;
const POLL_EVERY: Duration = Duration::from_millis(200);
const LOG_EVERY: Duration = Duration::from_secs(30);
/// the budget pass's period: the only rate at which it logs a pass that moved memory
const PASS_EVERY: Duration = Duration::from_secs(1);
/// a pressed pass that finds nothing cold enough says so this often, not every second
const PRESSED_EVERY: Duration = Duration::from_secs(30);
/// how long a pass that keeps being undone stands down
const BACKOFF: Duration = Duration::from_secs(15);

/// `XCL_VRAM_EVICT=1`: evict under allocation pressure and spill a protected allocation that
/// does not fit local memory. Off: no pressure eviction (`XCL_VRAM_BUDGET` is separate).
#[inline]
pub(super) fn pressure_eviction() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_VRAM_EVICT").is_some_and(|v| v == "1"))
}

/// `XCL_VRAM_BUDGET=1`: the once-a-second pass of `budget_pass`. Off: no pass. The name is exact
/// and not `XCL_VRAM_BUDGET_MB`, which overrides the budget both the pass and the spill gate
/// measure against.
#[inline]
pub(super) fn budget_pass_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_VRAM_BUDGET").is_some_and(|v| v == "1"))
}

/// Whether allocations are tracked, pinned and restored at all: every eviction needs it, from
/// whichever switch. Off means the allocator, the submissions and the releases never touch the
/// registry.
#[inline]
pub(super) fn eviction_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pressure_eviction() || budget_pass_enabled())
}

/// `XCL_VRAM_HIGH_PCT` (90) and `XCL_VRAM_LOW_PCT` (85): where a budget pass starts and stops,
/// as whole percents of the local budget. A value that is not 1 to 100, or a low that is not
/// below the high, is refused with one warning and the defaults are used: a pass that stops
/// where it starts either evicts everything or nothing at all, and neither is a mistake worth
/// starting a run with.
fn pass_pcts() -> (u64, u64) {
    static PCTS: OnceLock<(u64, u64)> = OnceLock::new();
    *PCTS.get_or_init(|| {
        let high = std::env::var("XCL_VRAM_HIGH_PCT").ok();
        let low = std::env::var("XCL_VRAM_LOW_PCT").ok();
        pcts(high.as_deref(), low.as_deref())
    })
}

/// `XCL_VRAM_HIGH_PCT` and `XCL_VRAM_LOW_PCT` as given (`None`: unset).
fn pcts(high: Option<&str>, low: Option<&str>) -> (u64, u64) {
    let read = |raw: Option<&str>, default: u64| match raw.map(str::trim).map(str::parse::<u64>) {
        None => default,
        Some(Ok(v)) if (1..=100).contains(&v) => v,
        Some(_) => default,
    };
    let (high, low) = (read(high, 90), read(low, 85));
    if low >= high { (90, 85) } else { (high, low) }
}

/// Whether a pass that evicted `moved` bytes was undone: less than 80% of that is still out of
/// local memory (`kept`: `State::evicted_local` now less what it was before the pass, so bytes an
/// earlier pass or a pressure trim evicted do not hide it). The share is of what the pass moved,
/// not of the budget, so a title that allocates a lot after the pass does not make it look
/// undone; what does is `before_submit` making resident what the pass took.
fn undone(kept: u64, moved: u64) -> bool {
    moved != 0 && kept.saturating_mul(5) < moved.saturating_mul(4)
}

#[derive(Clone, Copy, Debug)]
struct Heat {
    id: usize,
    bytes: u64,
    /// the submission that last used it (a counter over all submissions of the device)
    last_used: u64,
    /// when, and in which presented frame, that submission was
    last_used_at: Instant,
    last_frame: u64,
    in_flight: bool,
    resident: bool,
    high: bool,
}

/// What one eviction round asks for: the share of the budget it starts at and stops at, how
/// cold a candidate must be, and the clock it is judged against. [`Relief::pressure`] is
/// `XCL_VRAM_EVICT`'s – anything not in flight, no wait; [`Relief::pass`] is
/// `XCL_VRAM_BUDGET`'s, with the free bounds (`free_policy`) and its own water marks.
#[derive(Clone, Copy)]
struct Relief {
    high_pct: u64,
    low_pct: u64,
    now: Instant,
    frame: u64,
    cold: free_policy::Wait,
}

impl Relief {
    /// Making room for one allocation now: 90% down to 80%, anything not in flight. The
    /// submission that follows restores what it uses, so this is not where coldness matters.
    fn pressure() -> Self {
        Self { high_pct: 90, low_pct: 80, now: Instant::now(), frame: 0, cold: free_policy::Wait::NONE }
    }

    /// The budget pass: `XCL_VRAM_HIGH_PCT` down to `XCL_VRAM_LOW_PCT`, only allocations cold
    /// for the free bounds.
    fn pass() -> Self {
        let (high_pct, low_pct) = pass_pcts();
        Self { high_pct, low_pct, now: Instant::now(), frame: free_policy::frame(), cold: free_policy::wait() }
    }
}

/// What one eviction round moved.
#[derive(Clone, Copy, Default)]
struct Evicted {
    count: usize,
    bytes: u64,
}

fn percent(bytes: u64, percent: u64) -> u64 {
    bytes / 100 * percent + bytes % 100 * percent / 100
}

/// `usage` as a whole percent of `budget`.
fn share(usage: u64, budget: u64) -> u64 {
    usage.saturating_mul(100) / budget.max(1)
}

fn budget_mb(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok().filter(|&v| v != 0).and_then(|v| v.checked_mul(MIB))
}

/// The allocations to evict: coldest first (least recently used, other memory before protected
/// targets, the bigger before the smaller, then by id) until `usage` is back at the relief's low
/// water mark. Allocations in flight, evicted already, or not cold for the relief's bounds never
/// appear.
fn pick(heats: impl IntoIterator<Item = Heat>, usage: u64, budget: u64, relief: Relief) -> Vec<usize> {
    if budget == 0 || usage <= percent(budget, relief.high_pct) {
        return Vec::new();
    }
    let floor = percent(budget, relief.low_pct);
    let mut cold: Vec<_> = heats.into_iter()
        .filter(|h| h.resident && !h.in_flight
            && relief.cold.due(relief.now, relief.frame, h.last_used_at, h.last_frame))
        .collect();
    cold.sort_unstable_by_key(|h| (h.last_used, h.high, std::cmp::Reverse(h.bytes), h.id));
    let mut left = usage;
    cold.into_iter().take_while(|h| {
        if left <= floor {
            return false;
        }
        left = left.saturating_sub(h.bytes);
        true
    }).map(|h| h.id).collect()
}

struct BudgetEvent {
    handle: HANDLE,
    cookie: u32,
    adapter: IDXGIAdapter3,
}

impl Drop for BudgetEvent {
    fn drop(&mut self) {
        unsafe {
            self.adapter.UnregisterVideoMemoryBudgetChangeNotification(self.cookie);
            let _ = CloseHandle(self.handle);
        }
    }
}

#[derive(Default)]
struct Budget {
    adapter: Option<IDXGIAdapter3>,
    event: Option<BudgetEvent>,
    local: DXGI_QUERY_VIDEO_MEMORY_INFO,
    non_local: DXGI_QUERY_VIDEO_MEMORY_INFO,
    usage: u64,
    polled: Option<Instant>,
}

impl Budget {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        let mut budget = Self::default();
        budget.adapter = CreateDXGIFactory1::<IDXGIFactory4>().ok()
            .and_then(|factory| factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok());
        if let Some(adapter) = &budget.adapter {
            let Ok(handle) = CreateEventW(None, false, false, PCWSTR::null()) else { return budget };
            match adapter.RegisterVideoMemoryBudgetChangeNotificationEvent(handle) {
                Ok(cookie) => budget.event = Some(BudgetEvent { handle, cookie, adapter: adapter.clone() }),
                Err(_) => { let _ = CloseHandle(handle); }
            }
        }
        budget
    }

    unsafe fn poll(&mut self) {
        let Some(adapter) = &self.adapter else { return };
        let notified = self.event.as_ref().is_some_and(|e| WaitForSingleObject(e.handle, 0) == WAIT_OBJECT_0);
        let now = Instant::now();
        if !notified && self.polled.is_some_and(|last| now.duration_since(last) < POLL_EVERY) {
            return;
        }
        self.polled = Some(now);
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        if adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local).is_ok() {
            self.local = local;
            self.usage = local.CurrentUsage;
        }
        let _ = adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut self.non_local);
    }
}

type LastUse = Vec<(Arc<ID3D12Fence>, u64)>;

fn stamp(last: &mut LastUse, fence: &Arc<ID3D12Fence>, value: u64) {
    if let Some((_, previous)) = last.iter_mut().find(|(f, _)| Arc::ptr_eq(f, fence)) {
        *previous = value;
    } else {
        last.push((Arc::clone(fence), value));
    }
}

unsafe fn in_flight(last: &LastUse, completed: &mut HashMap<usize, u64>) -> bool {
    last.iter().any(|(f, value)| {
        let done = *completed.entry(f.as_raw() as usize).or_insert_with(|| f.GetCompletedValue());
        done == u64::MAX || done < *value
    })
}

struct Object {
    bytes: u64,
    heap_type: Option<D3D12_HEAP_TYPE>,
    local: bool,
    high: bool,
    evictable: bool,
    resident: bool,
    last_used: u64,
    last_used_at: Instant,
    last_frame: u64,
    last: LastUse,
    pins: u32,
    releasing: u32,
}

struct QueueFence {
    fence: Arc<ID3D12Fence>,
    value: u64,
}

struct State {
    /// the device this state was made for: the pass evicts through it
    device: ID3D12Device,
    budget: Budget,
    budget_override: Option<u64>,
    pending_local: u64,
    evicted_local: u64,
    objects: HashMap<usize, Object>,
    queues: HashMap<usize, QueueFence>,
    opaque_last: LastUse,
    opaque_used: u64,
    opaque_pending: u32,
    /// when, and in which frame, an opaque submission last went: what it used is unknown, so
    /// every allocation is as young as that submission for the free bounds (`Heat`)
    opaque_at: Instant,
    opaque_frame: u64,
    serial: u64,
    pass: Pass,
    log: Tally,
}

/// The budget pass's own bookkeeping.
#[derive(Default)]
struct Pass {
    /// what the last pass that evicted moved, and the evicted bytes before it ran: whether that
    /// much is still out of local memory is what says the pass helped
    freed: Option<(u64, u64)>,
    /// passes in a row whose freed memory came back
    undone: u32,
    /// no pass before this
    quiet_until: Option<Instant>,
    /// when a pressed pass last logged that nothing was cold enough
    pressed_log: Option<Instant>,
    /// the pass has said it is off for `XCL_SINGLE_QUEUE`
    single_queue: bool,
}

struct Tally {
    at: Instant,
    evicted: u64,
    restored: u64,
    spilled: u64,
    spills: u64,
    protected_spills: u64,
    reported_protected_spills: u64,
    worst_restore: Duration,
    trim_checks: u64,
    trim_opaque: u64,
}

impl State {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
        let budget_override = *OVERRIDE.get_or_init(|| std::env::var("XCL_VRAM_BUDGET_MB").ok().as_deref().and_then(budget_mb));
        let state = Self {
            device: dev.clone(),
            budget: Budget::new(dev),
            budget_override,
            pending_local: 0,
            evicted_local: 0,
            objects: HashMap::new(),
            queues: HashMap::new(),
            opaque_last: Vec::new(),
            opaque_used: 0,
            opaque_pending: 0,
            opaque_at: Instant::now(),
            opaque_frame: 0,
            serial: 0,
            pass: Pass::default(),
            log: Tally {
                at: Instant::now(), evicted: 0, restored: 0, spilled: 0, spills: 0,
                protected_spills: 0, reported_protected_spills: 0,
                worst_restore: Duration::ZERO, trim_checks: 0, trim_opaque: 0,
            },
        };
        start_budget_pass(dev);
        state
    }

    fn limit(&self) -> u64 {
        self.budget_override.unwrap_or(self.budget.local.Budget)
    }

    unsafe fn track(&mut self, raw: usize, bytes: u64, local: bool, high: bool) {
        let previous = self.objects.insert(raw, Object {
            bytes, heap_type: None, local, high, evictable: local, resident: true, last_used: self.serial,
            last_used_at: Instant::now(), last_frame: free_policy::frame(), last: Vec::new(), pins: 0, releasing: 0,
        });
        if let Some(old) = previous.filter(|o| !o.resident) {
            self.evicted_local = self.evicted_local.saturating_sub(old.bytes);
        }
    }

    /// Makes room for `incoming` bytes at `relief`'s high water mark and reports whether the
    /// memory now fits under it. A `incoming` larger than the high water mark can never fit: no
    /// amount of eviction gives it room it did not have.
    unsafe fn trim(&mut self, dev: &ID3D12Device, incoming: u64, relief: Relief) -> bool {
        self.budget.poll();
        let budget = self.limit();
        let usage = self.budget.usage.saturating_add(self.pending_local).saturating_add(incoming);
        let soft = percent(budget, relief.high_pct);
        if budget == 0 || usage <= soft {
            return true;
        }
        if incoming > soft {
            return false;
        }
        self.evict(dev, usage, budget, relief);
        self.budget.usage.saturating_add(self.pending_local).saturating_add(incoming) <= soft
    }

    /// Evicts the coldest allocations until `usage` is back at `relief`'s low water mark or
    /// nothing else can go. An opaque submission in flight (or one whose fence has not been
    /// checked to completion yet) stops it entirely: what such a submission used is unknown, so
    /// no allocation can be shown to be cold.
    ///
    /// The registry lock is held across `Evict` (as across every other D3D12 call here), so a
    /// submission can neither pin an allocation nor stamp it while the pass chooses and takes:
    /// the pins it drops and the fences it stamps are what keep the GPU's work and this list in
    /// agreement.
    unsafe fn evict(&mut self, dev: &ID3D12Device, usage: u64, budget: u64, relief: Relief) -> Evicted {
        self.log.trim_checks += 1;
        let mut completed = HashMap::new();
        if self.opaque_pending != 0 || in_flight(&self.opaque_last, &mut completed) {
            self.log.trim_opaque += 1;
            return Evicted::default();
        }
        let heats = self.objects.iter().filter(|(_, o)| o.evictable).map(|(&id, o)| Heat {
            id,
            bytes: o.bytes,
            last_used: o.last_used.max(self.opaque_used),
            last_used_at: o.last_used_at.max(self.opaque_at),
            last_frame: o.last_frame.max(self.opaque_frame),
            in_flight: o.pins != 0 || o.releasing != 0 || in_flight(&o.last, &mut completed),
            resident: o.resident,
            high: o.high,
        });
        let ids = pick(heats, usage, budget, relief);
        let mut moved = Evicted::default();
        if ids.is_empty() {
            return moved;
        }
        let objects: Vec<_> = ids.iter().map(|&id| id as *mut c_void).collect();
        match (dev.vtable().Evict)(dev.as_raw(), objects.len() as u32, objects.as_ptr()).ok() {
            Ok(()) => {
                for id in ids {
                    let o = self.objects.get_mut(&id).unwrap();
                    o.resident = false;
                    self.budget.usage = self.budget.usage.saturating_sub(o.bytes);
                    self.evicted_local += o.bytes;
                    self.log.evicted += 1;
                    moved.count += 1;
                    moved.bytes += o.bytes;
                }
            }
            Err(e) => crate::xlog_warn!("D3D12.X: Evict of {} allocations failed: {e}", ids.len()),
        }
        moved
    }

    /// One budget pass (`XCL_VRAM_BUDGET`): the adapter's local memory against its budget. Above
    /// `XCL_VRAM_HIGH_PCT` of it, `evict` takes the coldest allocations down to
    /// `XCL_VRAM_LOW_PCT`; a pass that moved anything logs once with what it moved and the usage
    /// and budget before and after, and a pressed pass that found nothing cold enough warns at
    /// most every `PRESSED_EVERY`.
    ///
    /// A pass that finds less than 80% of what it evicted still out of local memory did not help:
    /// a submission in between made it resident again (`before_submit` restores everything an
    /// opaque list may use), and evicting it a second time pays another round of `MakeResident`
    /// for nothing. Three such passes in a row stand the pass down for `BACKOFF`, logged once.
    unsafe fn budget_pass(&mut self) {
        // `XCL_SINGLE_QUEUE` hands submissions to the PC queue itself, past `before_submit`: an
        // eviction made here would never be restored, so the pass stays off in that mode
        if super::single::enabled() {
            let once = &mut self.pass.single_queue;
            if !*once {
                *once = true;
                crate::xlog_warn!("D3D12.X: VRAM budget pass is off while XCL_SINGLE_QUEUE runs: that path submits past the residency restore");
            }
            return;
        }
        self.budget.polled = None;
        self.budget.poll();
        let budget = self.limit();
        let before = self.budget.usage;
        let now = Instant::now();
        if budget == 0 || self.pass.quiet_until.is_some_and(|until| now < until) {
            return;
        }
        let relief = Relief::pass();
        if before <= percent(budget, relief.high_pct) {
            self.pass.freed = None;
            self.pass.undone = 0;
            return;
        }
        if let Some((moved, before_evicted)) = self.pass.freed.take() {
            if undone(self.evicted_local.saturating_sub(before_evicted), moved) {
                self.pass.undone += 1;
                if self.pass.undone >= 3 {
                    self.pass.undone = 0;
                    self.pass.quiet_until = Some(now + BACKOFF);
                    crate::xlog_warn!(
                        "D3D12.X: VRAM budget pass: less than 80% of the {:.0} MiB the last pass evicted is still out of local memory (a submission makes resident what it uses); no pass for {} s",
                        moved as f64 / MIB as f64, BACKOFF.as_secs(),
                    );
                }
            } else {
                self.pass.undone = 0;
            }
        }
        let device = self.device.clone();
        let already_evicted = self.evicted_local;
        let moved = self.evict(&device, before, budget, relief);
        if moved.count == 0 {
            if self.pass.pressed_log.is_none_or(|at| now.saturating_duration_since(at) >= PRESSED_EVERY) {
                self.pass.pressed_log = Some(now);
                let (tracked, busy, coldest) = self.coldness();
                crate::xlog_warn!(
                    "D3D12.X: VRAM budget pass: local {} of {} MiB ({share}%) and nothing cold enough to evict: {tracked} evictable tracked, {busy} in flight, coldest idle {}",
                    before / MIB, budget / MIB, coldest.map_or_else(|| "none".into(), |idle: Duration| format!("{:.1} s", idle.as_secs_f64())),
                    share = share(before, budget),
                );
            }
            return;
        }
        self.budget.polled = None;
        self.budget.poll();
        let after = self.budget.usage;
        self.pass.freed = Some((moved.bytes, already_evicted));
        crate::xlog_info!(
            "D3D12.X: VRAM budget pass: evicted {} MiB in {} allocations; local {} of {} MiB ({}%) -> {} MiB ({}%)",
            moved.bytes / MIB, moved.count, before / MIB, budget / MIB, share(before, budget), after / MIB, share(after, budget),
        );
    }

    /// (evictable allocations resident, how many of those are in flight, the shortest idle among
    /// the rest) – for the pressed pass's line.
    unsafe fn coldness(&mut self) -> (usize, usize, Option<Duration>) {
        let now = Instant::now();
        let mut completed = HashMap::new();
        let (mut tracked, mut busy, mut coldest) = (0, 0, None);
        for o in self.objects.values().filter(|o| o.evictable && o.resident) {
            tracked += 1;
            if o.pins != 0 || o.releasing != 0 || in_flight(&o.last, &mut completed) {
                busy += 1;
                continue;
            }
            let idle = now.saturating_duration_since(o.last_used_at);
            coldest = Some(coldest.map_or(idle, |coldest: Duration| coldest.min(idle)));
        }
        (tracked, busy, coldest)
    }

    /// Makes resident what a submission (or a `Hold`) is about to use. Making room for it is the
    /// pressure relief, not the budget pass: this is memory the GPU work in hand already owns, so
    /// the question is only what else may go to fit it – 90% down to 80%, as `XCL_VRAM_EVICT` has
    /// always done, whichever switch turned the tracker on.
    unsafe fn restore(&mut self, dev: &ID3D12Device, ids: &[usize]) -> Result<()> {
        let evicted: Vec<_> = ids.iter().copied().filter(|id| self.objects.get(id).is_some_and(|o| !o.resident)).collect();
        if evicted.is_empty() {
            return Ok(());
        }
        let incoming = evicted.iter().map(|id| self.objects[id].bytes).sum::<u64>();
        self.trim(dev, incoming, Relief::pressure());
        let start = Instant::now();
        let objects: Vec<_> = evicted.iter().map(|&id| id as *mut c_void).collect();
        let make_resident = || (dev.vtable().MakeResident)(dev.as_raw(), objects.len() as u32, objects.as_ptr()).ok();
        let result = make_resident().or_else(|_| {
            self.budget.polled = None;
            self.trim(dev, incoming, Relief::pressure());
            make_resident()
        });
        self.log.worst_restore = self.log.worst_restore.max(start.elapsed());
        result?;
        for id in evicted {
            self.objects.get_mut(&id).unwrap().resident = true;
            self.log.restored += 1;
        }
        self.budget.usage = self.budget.usage.saturating_add(incoming);
        self.evicted_local = self.evicted_local.saturating_sub(incoming);
        Ok(())
    }

    fn report(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.log.at) < LOG_EVERY {
            return;
        }
        let row = format!(
            "D3D12.X: VRAM local {} / {} MiB (DXGI budget {} MiB), non-local {} / {} MiB; {} evicted, {} made resident, {} total spills ({} protected, {:.2} MiB), worst MakeResident {:.2} ms, opaque blocked {}/{} pressure trim checks",
            self.budget.local.CurrentUsage / MIB, self.limit() / MIB, self.budget.local.Budget / MIB,
            self.budget.non_local.CurrentUsage / MIB, self.budget.non_local.Budget / MIB,
            self.log.evicted, self.log.restored, self.log.spills, self.log.protected_spills, self.log.spilled as f64 / MIB as f64, self.log.worst_restore.as_secs_f64() * 1000.0,
            self.log.trim_opaque, self.log.trim_checks,
        );
        if self.log.protected_spills != self.log.reported_protected_spills {
            crate::xlog_warn!("{row}");
            self.log.reported_protected_spills = self.log.protected_spills;
        } else {
            crate::xlog_info!("{row}");
        }
        self.log.at = now;
        self.log.evicted = 0;
        self.log.restored = 0;
        self.log.worst_restore = Duration::ZERO;
        self.log.trim_checks = 0;
        self.log.trim_opaque = 0;
    }
}

/// `XCL_VRAM_BUDGET=1`: the pass's thread, started on the first device. One pass every
/// `PASS_EVERY` – a look at DXGI more often than that costs more than the answer changes, and it
/// is also why a pass that moved memory is logged at most once a second. The registry lock is
/// taken for each pass and let go between them, so a pass never delays a submission by more than
/// its own Evict call.
unsafe fn start_budget_pass(dev: &ID3D12Device) {
    static STARTED: OnceLock<()> = OnceLock::new();
    if !budget_pass_enabled() || STARTED.set(()).is_err() {
        return;
    }
    let device = dev.clone();
    let spawned = std::thread::Builder::new().name("xcl-vram-budget".into()).spawn(move || {
        let (high, low) = pass_pcts();
        let wait = free_policy::wait();
        crate::xlog_info!(
            "D3D12.X: VRAM budget pass every {} s: above {high}% of the local budget, the coldest allocations go down to {low}%; a candidate must be {} frames and {} ms without use (XCL_VRAM_BUDGET, XCL_VRAM_HIGH_PCT, XCL_VRAM_LOW_PCT)",
            PASS_EVERY.as_secs(), wait.frames, wait.idle.as_millis(),
        );
        loop {
            std::thread::sleep(PASS_EVERY);
            let Some(lock) = REGISTRY.get() else { continue };
            let mut g = lock.lock().unwrap();
            let Some(s) = g.devices.get_mut(&(device.as_raw() as usize)) else { continue };
            s.budget_pass();
        }
    });
    if let Err(e) = spawned {
        crate::xlog_warn!("D3D12.X: cannot start the VRAM budget pass: {e}");
    }
}

#[derive(Default)]
struct ListUse {
    resources: HashSet<usize>,
    opaque: bool,
}

struct Link {
    heaps: Vec<usize>,
    generation: u64,
}

#[derive(Default)]
struct Registry {
    devices: HashMap<usize, State>,
    objects: HashMap<usize, (usize, u64)>,
    links: HashMap<usize, Link>,
    generation: u64,
    lists: HashMap<usize, ListUse>,
}

unsafe impl Send for Registry {}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
static RELEASED: Condvar = Condvar::new();

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn state<'a>(g: &'a mut Registry, dev: &ID3D12Device) -> &'a mut State {
    g.devices.entry(dev.as_raw() as usize).or_insert_with(|| unsafe { State::new(dev) })
}

pub(super) fn may_evict(flags: D3D12_HEAP_FLAGS) -> bool {
    const NO_SPILL: D3D12_HEAP_FLAGS = D3D12_HEAP_FLAGS(
        D3D12_HEAP_FLAG_ALLOW_DISPLAY.0 | D3D12_HEAP_FLAG_SHARED.0 | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER.0
        | D3D12_HEAP_FLAG_HARDWARE_PROTECTED.0 | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH.0
        | D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT.0,
    );
    flags.0 & NO_SPILL.0 == 0
}

pub(super) fn could_spill(props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS) -> bool {
    let gpu_only = props.Type == D3D12_HEAP_TYPE_DEFAULT
        || (props.Type == D3D12_HEAP_TYPE_CUSTOM && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE);
    may_evict(flags) && gpu_only
}

pub(super) fn protected_texture(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        && desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) != 0
}

unsafe fn spill_properties(dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS) -> Option<D3D12_HEAP_PROPERTIES> {
    if !could_spill(props, flags) || !local_memory(dev, props) {
        return None;
    }
    Some(D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..props
    })
}

unsafe fn local_memory(dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES) -> bool {
    if props.CreationNodeMask > 1 {
        return false;
    }
    match props.Type {
        D3D12_HEAP_TYPE_CUSTOM => props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1,
        D3D12_HEAP_TYPE_DEFAULT | D3D12_HEAP_TYPE_GPU_UPLOAD => {
            let node = if dev.GetNodeCount() == 1 { 0 } else { props.CreationNodeMask };
            dev.GetCustomHeapProperties(node, props.Type).MemoryPoolPreference == D3D12_MEMORY_POOL_L1
        }
        _ => false,
    }
}

fn allocate_with<T>(
    props: D3D12_HEAP_PROPERTIES, spill: D3D12_HEAP_PROPERTIES, fits_local: bool,
    mut create: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
) -> Result<(T, bool)> {
    if !fits_local {
        if let Ok(v) = create(&spill) {
            return Ok((v, true));
        }
        return create(&props).map(|v| (v, false));
    }
    match create(&props) {
        Err(e) if e.code() == E_OUTOFMEMORY => create(&spill).map(|v| (v, true)).or(Err(e)),
        other => other.map(|v| (v, false)),
    }
}

pub(super) unsafe fn allocate<T>(
    dev: &ID3D12Device, props: D3D12_HEAP_PROPERTIES, flags: D3D12_HEAP_FLAGS, bytes: u64, prefer_local: bool,
    mut create: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
) -> Result<(T, bool)> {
    let Some(spill) = (bytes != 0 && bytes != u64::MAX).then(|| spill_properties(dev, props, flags)).flatten() else {
        return create(&props).map(|v| (v, false));
    };
    let fits = {
        let mut g = registry().lock().unwrap();
        let s = state(&mut g, dev);
        if pressure_eviction() {
            s.trim(dev, bytes, Relief::pressure());
        } else {
            s.budget.poll();
        }
        let fits = prefer_local || s.limit() == 0
            || s.budget.usage.saturating_add(s.evicted_local).saturating_add(s.pending_local).saturating_add(bytes) <= percent(s.limit(), 90);
        s.pending_local = s.pending_local.saturating_add(bytes);
        fits
    };
    let made = allocate_with(props, spill, fits, create);
    let mut g = registry().lock().unwrap();
    let s = state(&mut g, dev);
    s.pending_local = s.pending_local.saturating_sub(bytes);
    if let Ok((_, spilled)) = &made {
        if *spilled {
            s.log.spilled = s.log.spilled.saturating_add(bytes);
            s.log.spills += 1;
            if prefer_local {
                s.log.protected_spills += 1;
                if s.log.spills <= 4 {
                    crate::xlog_warn!("D3D12.X: local allocation ran out of memory; spilled {bytes} protected bytes to non-local memory");
                }
            } else if s.log.spills <= 4 {
                crate::xlog_info!("D3D12.X: spilled {bytes} bytes to non-local memory");
            }
        } else {
            s.budget.usage = s.budget.usage.saturating_add(bytes);
        }
    }
    s.report();
    made
}

pub(super) unsafe fn track(dev: &ID3D12Device, raw: usize, props: D3D12_HEAP_PROPERTIES, bytes: u64, high: bool, evictable: bool, spilled: bool) {
    if !eviction_enabled() || bytes == 0 || bytes == u64::MAX {
        return;
    }
    let local = !spilled && local_memory(dev, props);
    let mut g = registry().lock().unwrap();
    g.generation += 1;
    let generation = g.generation;
    g.objects.insert(raw, (dev.as_raw() as usize, generation));
    let s = state(&mut g, dev);
    s.track(raw, bytes, local, high);
    let gpu_only = props.Type == D3D12_HEAP_TYPE_DEFAULT || props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
    let o = s.objects.get_mut(&raw).unwrap();
    o.heap_type = Some(if spilled { D3D12_HEAP_TYPE_CUSTOM } else { props.Type });
    o.evictable = local && evictable && gpu_only;
}

pub(super) fn describe_backing(resource: usize) -> Option<String> {
    let g = REGISTRY.get()?.lock().unwrap();
    let describe = |id: usize| {
        let &(dev, _) = g.objects.get(&id)?;
        let o = g.devices.get(&dev)?.objects.get(&id)?;
        let kind = match o.heap_type {
            Some(D3D12_HEAP_TYPE_DEFAULT) => "default",
            Some(D3D12_HEAP_TYPE_UPLOAD) => "upload",
            Some(D3D12_HEAP_TYPE_READBACK) => "readback",
            Some(D3D12_HEAP_TYPE_CUSTOM) => "custom",
            Some(D3D12_HEAP_TYPE_GPU_UPLOAD) => "gpu-upload",
            _ => "unknown type",
        };
        let pool = if o.local { "local" } else { "non-local" };
        let state = if o.resident { "resident" } else { "evicted" };
        Some(format!("{kind}/{pool}/{state}"))
    };
    if let Some(l) = g.links.get(&resource) {
        let mut heaps: Vec<_> = l.heaps.iter().filter_map(|&id| describe(id)).collect();
        heaps.sort_unstable();
        heaps.dedup();
        if !heaps.is_empty() {
            return Some(format!("tracked backing heaps {}", heaps.join(", ")));
        }
    }
    describe(resource).map(|o| format!("tracked allocation {o}"))
}

pub(super) unsafe fn create_heap(dev: &ID3D12Device, desc: &D3D12_HEAP_DESC, prefer_local: bool) -> Result<ID3D12Heap> {
    let (heap, spilled) = allocate(dev, desc.Properties, desc.Flags, desc.SizeInBytes, prefer_local, |properties| {
        let d = D3D12_HEAP_DESC { Properties: *properties, ..*desc };
        let mut heap: Option<ID3D12Heap> = None;
        super::title_textures::own_work(|| dev.CreateHeap(&d, &mut heap))?;
        heap.ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))
    })?;
    if eviction_enabled() {
        super::real::adopt(heap.as_raw(), super::real::Kind::Heap);
        track(dev, heap.as_raw() as usize, desc.Properties, desc.SizeInBytes, prefer_local, false, spilled);
    }
    Ok(heap)
}

pub(super) fn manage_heap(heap: &ID3D12Heap, high: bool) {
    if !eviction_enabled() {
        return;
    }
    let raw = heap.as_raw() as usize;
    let mut g = registry().lock().unwrap();
    if let Some((device, _)) = g.objects.get(&raw).copied() {
        let o = g.devices.get_mut(&device).unwrap().objects.get_mut(&raw).unwrap();
        o.high |= high;
        o.evictable = o.local;
    }
}

pub(super) fn link(resource: usize, heap: usize) {
    if !eviction_enabled() {
        return;
    }
    let mut g = registry().lock().unwrap();
    if !g.objects.contains_key(&heap) {
        return;
    }
    if let Some(link) = g.links.get_mut(&resource) {
        if !link.heaps.contains(&heap) {
            link.heaps.push(heap);
        }
    } else {
        g.generation += 1;
        let generation = g.generation;
        g.links.insert(resource, Link { heaps: vec![heap], generation });
    }
}

pub(super) fn copy_links(dst: usize, src: usize) {
    let heaps = registry().lock().unwrap().links.get(&src).map(|l| l.heaps.clone());
    for heap in heaps.into_iter().flatten() {
        link(dst, heap);
    }
}

pub(super) fn release(raw: usize, f: impl FnOnce() -> u32) -> u32 {
    if !eviction_enabled() {
        return f();
    }
    let Some(lock) = REGISTRY.get() else { return f() };
    let (owner, link) = {
        let mut g = lock.lock().unwrap();
        let owner = g.objects.get(&raw).copied();
        if let Some((dev, _)) = owner {
            g.devices.get_mut(&dev).unwrap().objects.get_mut(&raw).unwrap().releasing += 1;
        }
        (owner, g.links.get(&raw).map(|l| l.generation))
    };
    let n = f();
    if owner.is_none() && link.is_none() {
        return n;
    }
    let mut g = lock.lock().unwrap();
    if let Some((dev, _)) = owner.filter(|o| Some(o) == g.objects.get(&raw)) {
        let s = g.devices.get_mut(&dev).unwrap();
        if n == 0 {
            let o = s.objects.remove(&raw).unwrap();
            if o.resident && o.local {
                s.budget.usage = s.budget.usage.saturating_sub(o.bytes);
            } else if !o.resident {
                s.evicted_local = s.evicted_local.saturating_sub(o.bytes);
            }
            g.objects.remove(&raw);
        } else {
            s.objects.get_mut(&raw).unwrap().releasing -= 1;
        }
    }
    if n == 0 && link.is_some() && link == g.links.get(&raw).map(|l| l.generation) {
        g.links.remove(&raw);
    }
    RELEASED.notify_all();
    n
}

/// Whether the GPU work that last used `raw` has completed: the fence of the submission that
/// last used it has signaled, and no opaque submission is in flight (what one uses is unknown, so
/// nothing can be shown to be clear of it). False when allocations are not tracked: then no
/// fence exists to prove anything, and a caller that reuses memory on this answer must not.
pub(super) unsafe fn completed(raw: usize) -> bool {
    if !eviction_enabled() {
        return false;
    }
    let Some(lock) = REGISTRY.get() else { return false };
    let g = lock.lock().unwrap();
    let Some(&(dev, _)) = g.objects.get(&raw) else { return false };
    let Some(s) = g.devices.get(&dev) else { return false };
    let Some(o) = s.objects.get(&raw) else { return false };
    if o.releasing != 0 || s.opaque_pending != 0 {
        return false;
    }
    let mut done = HashMap::new();
    !in_flight(&o.last, &mut done) && !in_flight(&s.opaque_last, &mut done)
}

/// GPU work on `resource` that is not part of a tracked submission – an upload copy into a
/// placed texture, a readback of one – keeps what backs it resident until the guard is let go of:
/// `restore` makes it resident first, and the pin it takes keeps the budget pass from taking it
/// again while the work is handed over. Without tracking nothing was ever evicted and the guard
/// holds nothing.
#[must_use]
pub(super) struct Hold {
    device: usize,
    ids: Vec<usize>,
}

impl Drop for Hold {
    fn drop(&mut self) {
        let Some(lock) = REGISTRY.get() else { return };
        let mut g = lock.lock().unwrap();
        let Some(s) = g.devices.get_mut(&self.device) else { return };
        for id in &self.ids {
            if let Some(o) = s.objects.get_mut(id) {
                o.pins = o.pins.saturating_sub(1);
            }
        }
    }
}

pub(super) unsafe fn hold(resource: usize) -> Hold {
    let nothing = Hold { device: 0, ids: Vec::new() };
    if !eviction_enabled() {
        return nothing;
    }
    let Some(lock) = REGISTRY.get() else { return nothing };
    let mut g = lock.lock().unwrap();
    let ids: Vec<usize> = std::iter::once(resource)
        .chain(g.links.get(&resource).into_iter().flat_map(|l| l.heaps.iter().copied()))
        .filter(|id| g.objects.contains_key(id))
        .collect();
    let Some(&(device, _)) = ids.first().and_then(|id| g.objects.get(id)) else { return nothing };
    let raw = device as *mut c_void;
    let Some(dev) = ID3D12Device::from_raw_borrowed(&raw) else { return nothing };
    let Some(s) = g.devices.get_mut(&device) else { return nothing };
    for id in &ids {
        if let Some(o) = s.objects.get_mut(id) {
            o.pins += 1;
        }
    }
    if let Err(e) = s.restore(&dev, &ids) {
        crate::xlog_warn!("D3D12.X: MakeResident of the memory behind {resource:#x} before work outside a submission failed: {e}");
    }
    Hold { device, ids }
}

pub(super) fn note_list_resources(list: usize, resources: impl IntoIterator<Item = usize>) {
    if eviction_enabled() {
        registry().lock().unwrap().lists.entry(list).or_default().resources.extend(resources.into_iter().filter(|&r| r != 0));
    }
}

pub(super) fn note_list_untracked(list: usize) {
    if eviction_enabled() {
        registry().lock().unwrap().lists.entry(list).or_default().opaque = true;
    }
}

pub(super) fn list_reset(list: usize) {
    if !eviction_enabled() {
        return;
    }
    if let Some(lock) = REGISTRY.get() {
        lock.lock().unwrap().lists.remove(&list);
    }
}

pub(super) fn queue_released(queue: usize) {
    if !eviction_enabled() {
        return;
    }
    if let Some(lock) = REGISTRY.get() {
        for s in lock.lock().unwrap().devices.values_mut() {
            s.queues.remove(&queue);
        }
    }
}

pub(super) struct Submission {
    device: ID3D12Device,
    ids: Vec<usize>,
    opaque: bool,
}

pub(super) unsafe fn before_submit(queue: *mut c_void, lists: &[*mut c_void]) -> Option<Submission> {
    if !eviction_enabled() {
        return None;
    }
    before_use(queue, |g| {
        let mut resources = HashSet::new();
        for list in lists {
            match g.lists.get(&(*list as usize)) {
                Some(use_) if !use_.opaque => {
                    if use_.resources.iter().any(|r| !g.links.contains_key(r) && !g.objects.contains_key(r)) {
                        return None;
                    }
                    resources.extend(use_.resources.iter().copied());
                }
                _ => return None,
            }
        }
        Some(resources)
    }).unwrap_or_else(|e| {
        crate::xlog_warn!("D3D12.X: unable to prepare GPU residency; submitting anyway: {e}");
        None
    })
}

unsafe fn before_use<R: IntoIterator<Item = usize>>(
    queue: *mut c_void, resources: impl FnOnce(&Registry) -> Option<R>,
) -> Result<Option<Submission>> {
    let Some(lock) = REGISTRY.get() else { return Ok(None) };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return Ok(None) };
    let mut g = lock.lock().unwrap();
    if g.objects.is_empty() {
        return Ok(None);
    }
    let mut device = None;
    q.GetDevice(&mut device)?;
    let Some(device): Option<ID3D12Device> = device else { return Ok(None) };
    let dev = device.as_raw() as usize;
    let Some(s) = g.devices.get(&dev) else { return Ok(None) };
    let (mut ids, opaque): (Vec<_>, bool) = match resources(&g) {
        None => {
            let ids = if s.evicted_local == 0 { Vec::new() } else {
                s.objects.iter().filter_map(|(&id, o)| (!o.resident).then_some(id)).collect()
            };
            (ids, true)
        }
        Some(resources) => (resources.into_iter().flat_map(|r| {
            std::iter::once(r).chain(g.links.get(&r).into_iter().flat_map(|l| l.heaps.iter().copied()))
        }).filter(|id| g.objects.get(id).is_some_and(|&(owner, _)| owner == dev)).collect(), false),
    };
    ids.sort_unstable();
    ids.dedup();
    if !opaque && ids.is_empty() {
        return Ok(None);
    }
    let s = g.devices.get_mut(&dev).unwrap();
    if opaque {
        s.opaque_pending += 1;
    } else {
        for id in &ids {
            if let Some(o) = s.objects.get_mut(id) {
                o.pins += 1;
            }
        }
    }
    while ids.iter().any(|id| g.devices[&dev].objects.get(id).is_some_and(|o| !o.resident && o.releasing != 0)) {
        g = RELEASED.wait(g).unwrap();
    }
    let s = g.devices.get_mut(&dev).unwrap();
    if let Err(e) = s.restore(&device, &ids) {
        crate::xlog_warn!("D3D12.X: MakeResident failed after retry; submitting anyway: {e}");
    }
    Ok(Some(Submission { device, ids, opaque }))
}

pub(super) unsafe fn after_submit(queue: *mut c_void, submission: Option<Submission>) {
    let Some(Submission { device, ids, opaque }) = submission else { return };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let mut g = registry().lock().unwrap();
    let s = state(&mut g, &device);
    let key = queue as usize;
    let fence = match s.queues.entry(key) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => match super::real::create_residency_fence(&device) {
            Ok(fence) => e.insert(QueueFence { fence: Arc::new(fence), value: 0 }),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: cannot track submission completion: {e}");
                return;
            }
        },
    };
    fence.value += 1;
    if let Err(e) = q.Signal(&*fence.fence, fence.value) {
        crate::xlog_warn!("D3D12.X: cannot signal residency fence: {e}");
        return;
    }
    super::markers::queue_fence(key, "the latest submission (residency signal)", &fence.fence, fence.value);
    s.serial += 1;
    if opaque {
        stamp(&mut s.opaque_last, &fence.fence, fence.value);
        s.opaque_used = s.serial;
        s.opaque_at = Instant::now();
        s.opaque_frame = free_policy::frame();
        s.opaque_pending -= 1;
    } else {
        let now = Instant::now();
        let frame = free_policy::frame();
        for id in ids {
            if let Some(o) = s.objects.get_mut(&id) {
                stamp(&mut o.last, &fence.fence, fence.value);
                o.last_used = s.serial;
                o.last_used_at = now;
                o.last_frame = frame;
                o.pins = o.pins.saturating_sub(1);
            }
        }
    }
    if pressure_eviction() {
        s.trim(&device, 0, Relief::pressure());
    }
    s.report();
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn heat(id: usize, mib: u64, last_used: u64) -> Heat {
        heat_at(id, mib, last_used, Instant::now() - Duration::from_secs(60), 0)
    }

    fn heat_at(id: usize, mib: u64, last_used: u64, last_used_at: Instant, last_frame: u64) -> Heat {
        Heat { id, bytes: mib * MIB, last_used, last_used_at, last_frame, in_flight: false, resident: true, high: false }
    }

    fn pressure() -> Relief {
        Relief { high_pct: 90, low_pct: 80, now: Instant::now(), frame: 0, cold: free_policy::Wait::NONE }
    }

    #[test]
    fn nothing_goes_at_or_below_ninety_percent() {
        let heats = [heat(1, 512, 0), heat(2, 512, 0)];
        assert!(pick(heats, 9 * GIB, 10 * GIB, pressure()).is_empty());
        assert!(pick(heats, 0, 0, pressure()).is_empty(), "no budget known");
    }

    #[test]
    fn vram_budget_override_is_positive_and_checked() {
        assert_eq!(budget_mb("1"), Some(1 << 20));
        assert_eq!(budget_mb("8192"), Some(8 * GIB));
        for val in ["0", "-1", "mistyped", "18446744073709551615"] {
            assert_eq!(budget_mb(val), None);
        }
    }

    #[test]
    fn protected_allocation_attempts_local_and_only_spills_on_oom() {
        let local = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let l0 = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, ..Default::default() };
        let mut attempts = Vec::new();
        let made = allocate_with(local, l0, true, |props| {
            attempts.push(props.Type);
            if props.Type == local.Type {
                Err(windows::core::Error::from_hresult(E_OUTOFMEMORY))
            } else {
                Ok(())
            }
        }).unwrap();
        assert_eq!(attempts, [local.Type, l0.Type]);
        assert!(made.1);

        attempts.clear();
        let invalid = windows::Win32::Foundation::E_INVALIDARG;
        let e = allocate_with(local, l0, true, |props| {
            attempts.push(props.Type);
            Err::<(), _>(windows::core::Error::from_hresult(invalid))
        }).unwrap_err();
        assert_eq!(e.code(), invalid);
        assert_eq!(attempts, [local.Type]);
    }

    #[test]
    fn the_coldest_go_first_until_eighty_percent() {
        // 9.5 of 10 GiB: 1.5 GiB must go
        let heats = [heat(1, 512, 50), heat(2, 1024, 10), heat(3, 512, 20), heat(4, 1024, 30)];
        assert_eq!(pick(heats, 9 * GIB + 512 * MIB, 10 * GIB, pressure()), [2, 3]);
    }

    #[test]
    fn in_flight_and_evicted_allocations_are_not_candidates() {
        let mut busy = heat(1, 4096, 0);
        busy.in_flight = true;
        let mut gone = heat(2, 4096, 0);
        gone.resident = false;
        let warm = heat(3, 4096, 98);
        let cold = heat(4, 256, 97);
        assert_eq!(pick([busy, gone, warm, cold], 10 * GIB, 10 * GIB, pressure()), [4, 3]);
    }

    #[test]
    fn at_equal_age_other_memory_before_targets_and_the_bigger_first() {
        let mut target = heat(1, 2048, 10);
        target.high = true;
        let heats = [target, heat(2, 256, 10), heat(3, 2048, 10)];
        assert_eq!(pick(heats, 10 * GIB + 256 * MIB, 10 * GIB, pressure()), [3, 2]);
    }

    /// A budget pass evicts at its own water marks, and only allocations that have been cold for
    /// the free bounds: a thing used in the last frames, or within the last seconds, stays even
    /// when the budget is over its high mark.
    #[test]
    fn the_pass_stops_at_its_low_mark_and_spares_what_is_not_cold() {
        let now = Instant::now();
        let relief = Relief { high_pct: 90, low_pct: 85, now, frame: 100, cold: free_policy::Wait { frames: 2, idle: Duration::from_secs(1) } };
        // 9.2 of 10 GiB: 0.7 GiB must go to reach 85%, and the 768 MiB the coldest takes is enough
        let heats = [heat_at(1, 768, 1, now - Duration::from_secs(9), 40), heat_at(2, 256, 2, now - Duration::from_secs(9), 40)];
        assert_eq!(pick(heats, 9 * GIB + 205 * MIB, 10 * GIB, relief), [1]);
        // the coldest has been used in the last frame, the other 0.2 s ago: nothing may go
        let heats = [heat_at(1, 512, 1, now - Duration::from_secs(9), 99), heat_at(2, 4096, 2, now - Duration::from_millis(200), 40)];
        assert!(pick(heats, 9 * GIB + 205 * MIB, 10 * GIB, relief).is_empty());
        // at the high mark itself nothing goes (a pass frees from above it only)
        let heats = [heat_at(1, 512, 1, now - Duration::from_secs(9), 40)];
        assert!(pick(heats, 9 * GIB, 10 * GIB, relief).is_empty());
    }

    #[test]
    fn the_water_marks_are_ninety_and_eighty_five_and_a_bad_pair_is_refused() {
        assert_eq!(pcts(None, None), (90, 85));
        assert_eq!(pcts(Some("80"), Some("70")), (80, 70));
        assert_eq!(pcts(Some("90"), Some("85")), (90, 85), "explicit defaults");
        for (high, low) in [(Some("85"), Some("90")), (Some("90"), Some("90")), (Some("0"), Some("0")), (Some("high"), Some("low"))] {
            assert_eq!(pcts(high, low), (90, 85), "{high:?} {low:?}");
        }
        assert_eq!(pcts(Some("101"), Some("70")), (90, 70), "one bad value keeps the other's");
    }

    #[test]
    fn a_pass_whose_freed_memory_came_back_is_undone() {
        let mib = MIB;
        // evicted 400 MiB and all of it is still out: it helped
        assert!(!undone(400 * mib, 400 * mib));
        // 80% of it still out: it helped
        assert!(!undone(320 * mib, 400 * mib));
        // less than that: `before_submit` made it resident again, undone
        assert!(undone(319 * mib, 400 * mib));
        assert!(undone(0, 400 * mib));
        assert!(!undone(0, 0), "a pass that moved nothing cannot be undone");
    }

    #[test]
    fn a_share_of_the_budget_never_overflows_and_a_zero_budget_is_zero() {
        assert_eq!(share(0, 0), 0);
        assert_eq!(share(9 * GIB, 10 * GIB), 90);
        assert_eq!(share(6 * GIB + 512 * MIB, 6 * GIB), 108);
    }
}
