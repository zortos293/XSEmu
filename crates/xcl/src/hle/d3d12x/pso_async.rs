//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.
//!
//! A lazy pipeline (`lazy`) is not built in the call: it is built on the background workers when
//! the title makes it (`preload`: all the cores but two, below normal priority, in the order of
//! `pso_sched::Backlog`), and a bind waits for it, its build going ahead of all the others. A big
//! driver compile that does not fit the memory is put off and its worker goes on (`put_off`). One nobody binds is released again after a while
//! (`idle_limit`) and built again at its first bind. With `XCL_PSO_PRELOAD=0` it is built
//! when first bound, its shaders prepared before that on background threads
//! (`prepare_threads`): translated and compiled into the disk caches with the driver left out
//! (`preparing`). Without either the title's binding threads did every translation and DXC
//! compile themselves, one after another: a retail game showed one compile running on a
//! 24-thread PC while thousands of its pipelines waited. `XCL_PREPARE_PSO=0` turns that off.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

/// The priority of a preload worker: below the title's threads, which keep their cores while
/// every other core builds pipelines (at `THREAD_PRIORITY_LOWEST` the workers were also behind
/// every other process's threads).
const PRELOAD_PRIORITY: i32 = windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL;
/// … and while the title loads (`pso_sched::Tempo::low`): behind title threads that run below
/// normal themselves, as loaders do.
const LOADING_PRIORITY: i32 = windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST;
/// the priority a preload worker has now while no bind waits for what it builds (`pace`)
static REST_PRIORITY: AtomicI32 = AtomicI32::new(PRELOAD_PRIORITY);

/// Refcounted priority inheritance for a preload worker (`REST_PRIORITY`): while a guard
/// lives (a bind waits for the pipeline the worker builds), its OS thread runs at normal
/// priority, whatever the pace of the others; the last guard dropped restores its own.
struct BoostToken {
    tid: u32,
    count: Mutex<usize>,
}

struct BoostGuard(Arc<BoostToken>);

/// Whether preload workers run at `PRELOAD_PRIORITY` and are raised again on demand. Under
/// Wine lowering a thread's priority raises its Linux nice value (always allowed), while raising
/// it back needs `RLIMIT_NICE` / `CAP_SYS_NICE` and fails for a normal user: there the workers
/// stay at normal priority.
fn can_lower_preload_priority() -> bool {
    !crate::util::under_wine()
}

fn set_thread_priority_by_id(tid: u32, prio: i32) {
    if !can_lower_preload_priority() {
        return;
    }
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
    unsafe {
        let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
        if !h.is_null() {
            SetThreadPriority(h, prio);
            CloseHandle(h);
        }
    }
}

impl BoostToken {
    fn for_current_thread() -> Arc<Self> {
        let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        Arc::new(Self { tid, count: Mutex::new(0) })
    }

    #[cfg(test)]
    fn is_boosted(&self) -> bool {
        *self.count.lock().unwrap() > 0
    }

    fn boost(self: &Arc<Self>) -> BoostGuard {
        let mut c = self.count.lock().unwrap();
        if *c == 0 {
            set_thread_priority_by_id(self.tid, windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL);
        }
        *c += 1;
        BoostGuard(self.clone())
    }

    /// The workers' priority is `prio` from now on: this one's too, unless a bind waits for
    /// what it builds (then when that is built).
    fn rest(&self, prio: i32) {
        let c = self.count.lock().unwrap();
        if *c == 0 {
            set_thread_priority_by_id(self.tid, prio);
        }
    }
}

impl Drop for BoostGuard {
    fn drop(&mut self) {
        let mut c = self.0.count.lock().unwrap();
        *c = c.saturating_sub(1);
        if *c == 0 {
            set_thread_priority_by_id(self.0.tid, REST_PRIORITY.load(Ordering::Acquire));
        }
    }
}

/// Who builds a pipeline in the background ahead of its first bind, and whether a bind needs it.
#[derive(Default)]
struct PreloadOwner {
    /// the preload worker building it now
    worker: Option<Arc<BoostToken>>,
    /// a bind waits for it (`promote_preload`): a preload worker that has not started it skips it
    demanded: bool,
    /// the worker's raised priority while it builds a pipeline a bind waits for
    boost: Option<BoostGuard>,
}

struct PreloadJobGuard<'a>(&'a Pending);

impl Drop for PreloadJobGuard<'_> {
    fn drop(&mut self) {
        let mut owner = self.0.preload.lock().unwrap_or_else(|e| e.into_inner());
        owner.worker = None;
        owner.boost = None;
    }
}

struct Pending {
    /// when the title made it (`status`)
    created: std::time::Instant,
    /// the frame (`present::presents` + 1) a skipped bind last sent it to the bind queue
    last_bound: AtomicU64,
    /// the build, until a full-quality build of it finished
    job: Mutex<Option<Arc<JobFn>>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// a preload worker's build of it, and the bind that waits for it
    preload: Mutex<PreloadOwner>,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    image_only: std::sync::atomic::AtomicBool,
    depth_only: std::sync::atomic::AtomicBool,
    /// a pass over the screen (`mark_screen_pass`): its binds wait within `screen_budget`
    screen: std::sync::atomic::AtomicBool,
    /// a ray pass built with every ray a miss (`rt_pass_last`): built behind every other one
    rt_pass: std::sync::atomic::AtomicBool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    record: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a full-quality build started (one runs: `build_full`)
    full_started: std::sync::atomic::AtomicBool,
    /// a build that may take the fast tier started (one runs: `build_tiered`)
    tiered_started: std::sync::atomic::AtomicBool,
    /// `real` came from the fast tier: a full-quality build replaces it
    real_fast: std::sync::atomic::AtomicBool,
    /// a bind started a build of its own (`resolve_wait`)
    urgent: std::sync::atomic::AtomicBool,
    active_waiters: AtomicUsize,
    /// counted in `COMPUTE_BOUND_PENDING` until its build is done
    counted: std::sync::atomic::AtomicBool,
    /// the fast-tier pipeline the optimized one replaced, kept until the stand-in goes (lists
    /// recorded with it may still run)
    retired: AtomicUsize,
    /// a bind (or a lookup made for one) took it: its real pipeline stays (`release_idle`)
    bound: std::sync::atomic::AtomicBool,
    /// when its real pipeline was built (`since_start`, ms; 0: not built)
    built_at: std::sync::atomic::AtomicU64,
    /// its real pipeline was released, nothing having bound it (`release_idle`)
    released_idle: std::sync::atomic::AtomicBool,
    /// a bind asked for it again after that: its next build is a rebuild
    rebuild: std::sync::atomic::AtomicBool,
    /// what its last build waited for by a bind was (`note_build`)
    note: Mutex<Option<String>>,
    /// its shaders, for the log (stage and hash of each)
    what: String,
    /// a draw was skipped for want of it (`backstop`): later binds do not wait for it either
    skipped: std::sync::atomic::AtomicBool,
    /// the title's root signature of the pipeline (0: unknown): what is bound next is likely of
    /// the group of what was bound last (`pso_sched::Backlog`)
    group: u64,
    preload_seq: AtomicU64,
    /// built before every other build (`build_first`)
    first: std::sync::atomic::AtomicBool,
    /// never built (`give_up`): its draws and dispatches are skipped and counted
    left_out: std::sync::atomic::AtomicBool,
    /// a stage of the acceleration-structure builder (`mark_builder`): its real pipeline is one too
    builder: std::sync::atomic::AtomicBool,
    /// draws and dispatches skipped for want of it since a bind-queue worker last took it
    /// (`bound_by_skips`)
    skips_waiting: AtomicU64,
    /// in the bind queue and not taken by a worker yet (`bound_by_skips`: queued once)
    in_bound: std::sync::atomic::AtomicBool,
    /// a submit gave up waiting for it (`wait_for_submit`): later calls with it are skipped, not
    /// put off to the submit
    submit_gave_up: std::sync::atomic::AtomicBool,
    /// its last build was refused as one whose driver compile never fits the memory now
    /// (`real::take_never_fit`): that compile's estimate (0: not refused). Its binds skip, its
    /// build is kept, and it is built again once that much fits (`retry_never_fits`).
    never_fit: AtomicU64,
}

impl Drop for Pending {
    fn drop(&mut self) {
        release(*self.retired.get_mut());
    }
}

impl Pending {
    /// A bind waits for this pipeline. True if a preload worker is building it already: that
    /// worker is raised to normal priority until it is done, and no second build of the same
    /// program is started (its driver compile takes 1.4 GB; `Governor` counts it as waited for).
    /// False: no worker has it, and a worker that takes it later skips it.
    fn promote_preload(&self) -> bool {
        let mut owner = self.preload.lock().unwrap_or_else(|e| e.into_inner());
        owner.demanded = true;
        let Some(worker) = owner.worker.clone() else { return false };
        if owner.boost.is_none() {
            owner.boost = Some(worker.boost());
        }
        true
    }

    /// A bind (or a lookup for one) takes this pipeline: from now on its real pipeline stays.
    fn note_bind(&self) {
        self.bound.store(true, Ordering::SeqCst);
        note_bound_group(self.group);
        if self.released_idle.swap(false, Ordering::AcqRel) {
            REBUILT.fetch_add(1, Ordering::Relaxed);
            self.rebuild.store(true, Ordering::Release);
        }
        if !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
            super::pso_manifest::note_bound(self.record);
        }
    }
}

/// the groups (`Pending::group`) of the pipelines bound last, and how many were noted: the last
/// one is at that number less one, modulo the length
static RECENT_GROUPS: [AtomicU64; super::pso_sched::RECENT_GROUPS] = [const { AtomicU64::new(0) }; super::pso_sched::RECENT_GROUPS];
static RECENT_NOTED: AtomicUsize = AtomicUsize::new(0);

/// A pipeline of `group` is bound (every bind comes here: no lock, and a group bound again and
/// again is noted once).
fn note_bound_group(group: u64) {
    let noted = RECENT_NOTED.load(Ordering::Relaxed);
    if group == 0 || RECENT_GROUPS[noted.wrapping_sub(1) % RECENT_GROUPS.len()].load(Ordering::Relaxed) == group {
        return;
    }
    RECENT_GROUPS[noted % RECENT_GROUPS.len()].store(group, Ordering::Relaxed);
    RECENT_NOTED.store(noted.wrapping_add(1), Ordering::Relaxed);
}

/// The groups of the pipelines bound last, the last one first (0: none).
fn recent_groups() -> [u64; super::pso_sched::RECENT_GROUPS] {
    let noted = RECENT_NOTED.load(Ordering::Relaxed);
    std::array::from_fn(|k| RECENT_GROUPS[noted.wrapping_sub(1 + k) % RECENT_GROUPS.len()].load(Ordering::Relaxed))
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// A build finished (`real`: its pipeline, 0 = failed; `fast`: one of the fast tier,
/// `dxc::fast_tier`). The first pipeline is used at once; an optimized one replaces a fast one.
/// A failed fast build leaves the pipeline waiting for its full-quality build.
fn complete(p: &Pending, real: usize, fast: bool) {
    let never_fit = if real == 0 && !fast { super::real::take_never_fit() } else { 0 };
    let drop_now = {
        let mut done = p.done.lock().unwrap();
        if !fast {
            p.never_fit.store(never_fit, Ordering::Release);
        }
        let mut drop_now = 0;
        if p.orphan.load(Ordering::Acquire) {
            drop_now = real;
        } else if real != 0 {
            match p.real.load(Ordering::Acquire) {
                0 => {
                    BUILT.fetch_add(1, Ordering::Relaxed);
                    if p.builder.load(Ordering::Acquire) {
                        super::raytracing::note_builder_identity(real);
                    }
                    p.real.store(real, Ordering::SeqCst);
                    p.real_fast.store(fast, Ordering::Release);
                    p.built_at.store(since_start().max(1), Ordering::Release);
                }
                old if !fast && p.real_fast.swap(false, Ordering::AcqRel) => {
                    if p.builder.load(Ordering::Acquire) {
                        super::raytracing::note_builder_identity(real);
                    }
                    p.real.store(real, Ordering::Release);
                    drop_now = p.retired.swap(old, Ordering::AcqRel);
                }
                _ => drop_now = real,
            }
        } else if !fast && !*done {
            FAILED.fetch_add(1, Ordering::Relaxed);
        }
        if real != 0 || !fast {
            if p.counted.swap(false, Ordering::AcqRel) {
                COMPUTE_BOUND_PENDING.fetch_sub(1, Ordering::AcqRel);
            }
            *done = true;
            p.cv.notify_all();
        }
        drop_now
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    release(drop_now);
    // (one nothing bound keeps its build: its pipeline may be released and built again; so does
    // one refused for memory, built again when the memory comes back)
    if never_fit != 0 {
        start_never_fit_retry();
    } else if !fast && p.bound.load(Ordering::SeqCst) && !keeps_build(p) {
        *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// pipelines built again at their next bind once the memory they were refused for came back
/// (`retry_never_fits`)
static NEVER_FIT_RETRIED: AtomicUsize = AtomicUsize::new(0);

/// The pipelines whose build was refused as one that never fits the memory now
/// (`Pending::never_fit`) and whose driver compile `fits` now wait for a bind to build them
/// again, as after `release_used`: the number of them. One refused again goes back to waiting.
fn retry_never_fits(fits: impl Fn(u64) -> bool) -> usize {
    let refused: Vec<Arc<Pending>> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.values().filter(|p| p.never_fit.load(Ordering::Acquire) != 0).cloned().collect());
    let mut again = 0;
    for p in refused {
        let cost = p.never_fit.load(Ordering::Acquire);
        if cost == 0 || p.orphan.load(Ordering::Acquire) || !fits(cost) {
            continue;
        }
        let mut done = p.done.lock().unwrap();
        if !*done || p.real.load(Ordering::Acquire) != 0 || p.never_fit.swap(0, Ordering::AcqRel) == 0 {
            continue;
        }
        *done = false;
        unbuild(&p);
        again += 1;
    }
    again
}

/// The thread that looks every 2 s whether the memory came back for the pipelines refused as
/// never fitting it now (`retry_never_fits`; started by the first refusal).
fn start_never_fit_retry() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-memory".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let again = retry_never_fits(super::real::fits_now);
            if again > 0 {
                let total = NEVER_FIT_RETRIED.fetch_add(again, Ordering::Relaxed) + again;
                crate::xlog_info!("D3D12.X: {again} pipelines refused as never fitting the memory are built again at their next bind: the memory for their driver compiles came back ({total} so far)");
            }
        });
    });
}

/// Milliseconds since XCL first asked.
fn since_start() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// How long a pipeline built when the title made it may wait for its first bind before its real
/// pipeline is released (`XCL_PSO_IDLE`: seconds, 0 = never): its build is kept, and a bind
/// builds it again from the disk caches (the translation's DXIL, the driver's blob – ~0.1 s).
/// A retail game makes ~51 000 pipelines while it loads and binds few; all of them built and
/// kept ran the process out of commit.
fn idle_limit() -> Option<std::time::Duration> {
    static L: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *L.get_or_init(|| {
        let secs = std::env::var("XCL_PSO_IDLE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(30u64);
        (secs > 0).then(|| std::time::Duration::from_secs(secs))
    })
}

/// pipelines released after waiting `idle_limit` for a first bind, and of those built again
static IDLE_RELEASED: AtomicUsize = AtomicUsize::new(0);
static REBUILT: AtomicUsize = AtomicUsize::new(0);

/// Release `p`'s real pipeline if nothing bound it in the `limit` since it was built (true:
/// released). Its stand-in and build stay; it waits for a bind again, as before it was built.
///
/// (A bind marks `bound` before it reads `real` or `done`, and this checks `bound` after it
/// took `real` away, all `SeqCst` and under `done`: either the bind finds the pipeline gone and
/// waits for it to be built again, or this finds the bind and puts the pipeline back.)
fn release_idle(p: &Pending, limit: std::time::Duration) -> bool {
    let built_at = p.built_at.load(Ordering::Acquire);
    if built_at == 0 || since_start().saturating_sub(built_at) < limit.as_millis() as u64 || p.bound.load(Ordering::SeqCst) || p.orphan.load(Ordering::Acquire) {
        return false;
    }
    if job_of(p).is_none() {
        return false;
    }
    let real = {
        let mut done = p.done.lock().unwrap();
        if !*done {
            return false;
        }
        let real = p.real.swap(0, Ordering::SeqCst);
        if real == 0 {
            return false;
        }
        if p.bound.load(Ordering::SeqCst) {
            p.real.store(real, Ordering::SeqCst);
            return false;
        }
        *done = false;
        p.built_at.store(0, Ordering::Release);
        p.full_started.store(false, Ordering::Release);
        p.tiered_started.store(false, Ordering::Release);
        p.urgent.store(false, Ordering::Release);
        p.preload.lock().unwrap_or_else(|e| e.into_inner()).demanded = false;
        p.sent.store(false, Ordering::Release);
        p.released_idle.store(true, Ordering::Release);
        real
    };
    release(real);
    IDLE_RELEASED.fetch_add(1, Ordering::Relaxed);
    true
}

/// With a budget for the compiled code the lists use (`shader_code::budget`), a pipeline the
/// budget may take away (a compute one; a graphics one too with `shader_code::graphics`) keeps
/// its build after a bind too: `release_used` may take its pipeline away, and its next bind
/// builds it again.
fn keeps_build(p: &Pending) -> bool {
    super::shader_code::budget() != 0 && (p.compute || super::shader_code::graphics())
}

/// Take `real` away from `p`, a pipeline that lists have used: a compute one, or with `graphics`
/// (`shader_code::graphics`) a graphics one (true: taken; the caller
/// gives it to `shader_code::retire`, which keeps it until those lists are through). The stand-in
/// and its build stay; the next bind finds it not built and has it built again, as after
/// `release_idle` (a draw with it is skipped or waits meanwhile, as before it was first built).
///
/// (A bind that read `real` before this took it goes on to set it on its list, where
/// `shader_code::used` notes that list: that is why the pipeline is retired, not released here.)
fn release_used(p: &Pending, real: usize, graphics: bool) -> bool {
    if (!p.compute && !graphics) || p.orphan.load(Ordering::Acquire) || p.builder.load(Ordering::Acquire) || p.real_fast.load(Ordering::Acquire) || job_of(p).is_none() {
        return false;
    }
    let mut done = p.done.lock().unwrap();
    if !*done || p.real.compare_exchange(real, 0, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return false;
    }
    *done = false;
    unbuild(p);
    p.released_idle.store(true, Ordering::Release);
    true
}

/// `p`, not built now (`done` false, under `done`), as before it was first built: its next bind
/// finds it not built and has it built again (`release_used`, `retry_never_fits`).
fn unbuild(p: &Pending) {
    p.built_at.store(0, Ordering::Release);
    p.full_started.store(false, Ordering::Release);
    p.tiered_started.store(false, Ordering::Release);
    p.urgent.store(false, Ordering::Release);
    p.preload.lock().unwrap_or_else(|e| e.into_inner()).demanded = false;
    p.sent.store(false, Ordering::Release);
    // (a bind that was skipped for want of it before does not queue its build again)
    p.skipped.store(false, Ordering::Release);
    p.submit_gave_up.store(false, Ordering::Release);
}

/// The thread that keeps the compiled code the lists use within its budget
/// (`shader_code::budget`; started by the first bind when there is one): every second it
/// releases the retired pipelines whose lists are through, and while the code is over the budget
/// it takes the pipelines no list took for `shader_code::idle_ms` from their stand-ins (the
/// compute ones; the graphics ones too with `shader_code::graphics`), the longest unused first.
pub(super) fn start_code_release() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let (budget, idle_s) = (super::shader_code::budget() >> 20, super::shader_code::idle_ms() / 1000);
        let graphics = super::shader_code::graphics();
        // (with XCL_SHADER_CODE_GRAPHICS=0 every line here reads as it did for compute pipelines alone)
        let (kind, also) = if graphics { ("", " (graphics pipelines too; XCL_SHADER_CODE_GRAPHICS=0: compute ones only)") } else { ("compute ", "") };
        crate::xlog_info!("D3D12.X: compiled shader code budget {budget} MB (XCL_SHADER_CODE_MB; unset or 0: none): over it, {kind}pipelines no list took for {idle_s} s are released and built again at their next bind{also}");
        let _ = std::thread::Builder::new().name("xcl-pso-code".into()).spawn(move || {
            let (mut rounds, mut stuck) = (0usize, 0usize);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                super::shader_code::frame_mark(super::present::presents());
                let (through, parked) = super::shader_code::through();
                through.into_iter().for_each(release);
                let (in_use, over) = super::shader_code::over_budget();
                if over == 0 {
                    stuck = 0;
                    continue;
                }
                // (a member of a pass chain stays: while one is not built no pass of its chain
                // runs, `pass_chain`)
                let built: Vec<(usize, Arc<Pending>)> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.iter().filter(|(_, p)| (p.compute || graphics) && p.real.load(Ordering::Acquire) != 0).map(|(&standin, p)| (standin, p.clone())).collect());
                // (PENDING is free again here: `pass_chain` takes it under its own lock)
                let all: Vec<Arc<Pending>> = built.into_iter().filter(|(standin, _)| !super::pass_chain::member(*standin)).map(|(_, p)| p).collect();
                // (a retail game holds about 51 000 stand-ins and a few hundred built pipelines)
                let idle = all.iter().enumerate().filter(|(_, p)| (p.compute || graphics) && p.real.load(Ordering::Acquire) != 0).filter_map(|(at, p)| {
                    let (last, bytes) = super::shader_code::idle(p.real.load(Ordering::Acquire))?;
                    Some((last, bytes, at))
                });
                let (mut released, mut freed, mut drawing) = (0usize, 0usize, 0usize);
                for (at, bytes) in super::shader_code::to_release(idle.collect(), over) {
                    let real = all[at].real.load(Ordering::Acquire);
                    // (looked at again: a list may have taken it since the candidates were listed)
                    if real != 0 && super::shader_code::idle(real).is_some() && release_used(&all[at], real, graphics) {
                        super::shader_code::retire(real);
                        released += 1;
                        freed += bytes;
                        drawing += !all[at].compute as usize;
                    }
                }
                if released > 0 {
                    (rounds, stuck) = (rounds + 1, 0);
                    if rounds <= 4 || rounds.is_power_of_two() {
                        crate::xlog_info!(
                            "D3D12.X: {released} {kind}pipelines no list took for {idle_s} s released{}, {} MB of compiled code: {} MB were in use against a budget of {budget} MB; {} released ones are kept until their lists are through; each is built again at its next bind (#{rounds})",
                            if graphics { format!(" ({drawing} of them graphics)") } else { String::new() },
                            freed >> 20,
                            in_use >> 20,
                            parked + released
                        );
                        if let Some(line) = super::shader_code::by_age() {
                            crate::xlog_info!("{line}");
                        }
                    }
                } else {
                    stuck += 1;
                    if stuck == 5 || (stuck > 5 && stuck % 60 == 0) {
                        crate::xlog_warn!("D3D12.X: the compiled code in use ({} MB) is {} MB over its budget of {budget} MB and no {kind}pipeline has gone {idle_s} s without a list taking it: nothing is released", in_use >> 20, over >> 20);
                        if let Some(line) = super::shader_code::by_age() {
                            crate::xlog_warn!("{line}");
                        }
                    }
                }
            }
        });
    });
}

/// How many built pipelines nothing has bound yet are kept at most (`XCL_PSO_IDLE_KEEP`, 4096 by
/// default: ~1.6 GB in an NVIDIA driver): beyond it the oldest are released before `idle_limit`.
/// Workers on every core build a retail game's pipelines faster than 30 s release them.
fn idle_keep() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| std::env::var("XCL_PSO_IDLE_KEEP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4096))
}

/// pipelines built ahead since the idle thread last looked, and its wake-up
static BUILT_SINCE_LOOK: Mutex<usize> = Mutex::new(0);
static LOOK: Condvar = Condvar::new();

/// A preload worker built a pipeline: the idle thread looks again once a quarter of `idle_keep`
/// were built since it last did, without waiting out its 5 s.
fn note_built_ahead() {
    let mut n = BUILT_SINCE_LOOK.lock().unwrap();
    *n += 1;
    if *n >= (idle_keep() / 4).max(1) {
        LOOK.notify_one();
    }
}

/// The thread that releases pipelines nothing bound (`idle_limit`, `idle_keep`), every 5 s or sooner
/// (`note_built_ahead`); it also drops the builds of bound ones (never released, never built again).
fn start_idle_release() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let Some(limit) = idle_limit() else { return };
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-idle".into()).spawn(move || loop {
            {
                let n = BUILT_SINCE_LOOK.lock().unwrap();
                let (mut n, _) = LOOK.wait_timeout_while(n, std::time::Duration::from_secs(5), |n| *n < (idle_keep() / 4).max(1)).unwrap();
                *n = 0;
            }
            let all: Vec<(usize, Arc<Pending>)> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.iter().map(|(&standin, p)| (standin, p.clone())).collect());
            let mut unbound = Vec::new();
            for (at, (standin, p)) in all.iter().enumerate() {
                if p.bound.load(Ordering::SeqCst) {
                    if !keeps_build(p) && *p.done.lock().unwrap() && p.never_fit.load(Ordering::Acquire) == 0 {
                        *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    }
                } else if !super::pass_chain::member(*standin) {
                    // (a member of a pass chain stays built: while one is not built no pass of
                    // its chain runs, `pass_chain`; PENDING is free here)
                    match p.built_at.load(Ordering::Acquire) {
                        0 => {}
                        built_at => unbound.push((at, built_at)),
                    }
                }
            }
            let (kept, mut released, mut early) = (unbound.len(), 0, 0);
            for at in super::pso_sched::idle_to_release(&mut unbound, since_start(), limit.as_millis() as u64, idle_keep()) {
                let old = since_start().saturating_sub(all[at].1.built_at.load(Ordering::Acquire)) >= limit.as_millis() as u64;
                if release_idle(&all[at].1, if old { limit } else { std::time::Duration::ZERO }) {
                    released += 1;
                    early += !old as usize;
                }
            }
            if released > 0 {
                let (live, _, _) = live();
                crate::xlog_info!(
                    "D3D12.X: {released} pipelines nobody bound released ({} of them within {} s of their build: {kept} were kept, {} at most; {live} live; they are built again at their first bind)",
                    early,
                    limit.as_secs(),
                    idle_keep()
                );
            }
        });
    });
}

fn job_of(p: &Pending) -> Option<Arc<JobFn>> {
    p.job.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The full-quality build (one runs).
fn build_full(p: &Pending) {
    if p.orphan.load(Ordering::Acquire) || p.full_started.swap(true, Ordering::AcqRel) {
        return;
    }
    if let Some(job) = job_of(p) {
        complete(p, run(p, &*job), false);
    }
}

/// The build something waits for: a big program the cache has no optimized DXIL for compiled
/// at the fast tier first (`dxc::fast_tier`), its full-quality build queued behind everything
/// else and swapped in when done.
/// `hand_off`: a live build's cold giant that cannot start now goes to the giant lane
/// (`GIANT_LANE`) instead of holding this worker (`real::live_lanes`).
fn build_tiered(p: &Arc<Pending>, hand_off: bool) {
    if p.orphan.load(Ordering::Acquire) || *p.done.lock().unwrap() || p.tiered_started.swap(true, Ordering::AcqRel) {
        return;
    }
    let Some(job) = job_of(p) else { return };
    if hand_off && p.last_bound.load(Ordering::Acquire) != 0 {
        PUT_OFF.with(|d| d.set(PutOff::Allowed));
    }
    // (no fast tier: -O0 DXIL took an NVIDIA driver longer than -O1 – 182 s against 123 s on a
    // retail game's biggest compute program – and DXC only ~half the time)
    let (real, fast) = (run(p, &*job), false);
    if let (0, PutOff::Left(_)) = (real, PUT_OFF.with(|d| d.replace(PutOff::No))) {
        p.tiered_started.store(false, Ordering::Release);
        crate::progress::PIPELINES.queue();
        GIANT_LANE.0.lock().unwrap().push_back(p.clone());
        GIANT_LANE.1.notify_one();
        start_giant_lane();
        return;
    }
    if !fast {
        p.full_started.store(true, Ordering::Release);
    }
    complete(p, real, fast);
    if fast && !p.full_started.load(Ordering::Acquire) {
        enqueue_upgrade(p.clone());
    }
}

/// Live builds whose cold giant did not fit when a bind-queue worker came to it, built one at a
/// time, in order, on one thread that waits for the memory (`real::live_lanes`): the workers go
/// on with the small ones.
static GIANT_LANE: (Mutex<VecDeque<Arc<Pending>>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());
static GIANT_LANE_BUILDS: AtomicUsize = AtomicUsize::new(0);

fn start_giant_lane() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-giant".into()).stack_size(64 << 20).spawn(|| loop {
            let p = {
                let (queue, ready) = &GIANT_LANE;
                let mut queue = ready.wait_while(queue.lock().unwrap(), |q| q.is_empty()).unwrap();
                queue.pop_front().unwrap()
            };
            let _task = crate::progress::PIPELINES.take();
            let n = GIANT_LANE_BUILDS.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} live giants built in the giant lane ({} waiting)", GIANT_LANE.0.lock().unwrap().len());
            }
            BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
            build_tiered(&p, false);
            BUILDING.with(|b| *b.borrow_mut() = None);
        });
    });
}

/// `p`'s driver compile of `dxil_len` bytes of DXIL waits for memory (`pso_sched::Backlog::park`).
fn park(q: &Preload, dxil_len: usize, p: &Arc<Pending>) {
    let first_use = if p.compute { used::rank(p.key) } else { None };
    let mut list = q.lock();
    list.park(dxil_len, p.group, p.preload_seq.load(Ordering::Relaxed), first_use, Arc::downgrade(p));
    LEAST_PARKED.store(list.least_parked(), Ordering::Release);
}

static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub(super) fn failed_count() -> usize {
    FAILED.load(Ordering::Relaxed)
}
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Whether a bind is waiting for a pipeline still being built: one of the title's threads is
/// held where a console title never waits (`kernelx::Sleep` does not count that time).
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held<'a>(&'a Pending);

impl<'a> Held<'a> {
    fn new(p: &'a Pending) -> Self {
        HELD.fetch_add(1, Ordering::AcqRel);
        p.active_waiters.fetch_add(1, Ordering::AcqRel);
        Self(p)
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.0.active_waiters.fetch_sub(1, Ordering::AcqRel);
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed. Called again for the pipeline once
/// its shaders were prepared (`preparing`).
pub type Job = Box<JobFn>;
type JobFn = dyn Fn() -> usize + Send + Sync;

/// Run a job; a panic in it is a failed build (a worker must not die with it, and whoever
/// waits for the pipeline must be woken).
fn run(p: &Pending, job: &JobFn) -> usize {
    super::real::take_never_fit();
    let (started, waited) = (std::time::Instant::now(), super::real::memory_waited());
    let real = crate::live_stall::with_job(p.record, p.key, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or(0)
    });
    if real != 0 && !preparing() {
        let took = started.elapsed().saturating_sub(super::real::memory_waited().saturating_sub(waited));
        super::pso_manifest::note_build_time(p.record, took.as_millis() as u64);
    }
    real
}

thread_local! {
    static PREPARING: Cell<bool> = const { Cell::new(false) };
}

/// Inside a job run to prepare its shaders: the translation and DXC run, the driver does not
/// (creating the pipeline fails at once, and what fails is not reported – the real build
/// reports it).
pub fn preparing() -> bool {
    PREPARING.with(|p| p.get())
}

/// `f` with the driver, inside a job preparing its shaders: the driver's compile of a big
/// program is worth having in its disk cache before the program's first use.
pub fn with_driver<R>(f: impl FnOnce() -> R) -> R {
    let was = PREPARING.with(|p| p.replace(false));
    let r = f();
    PREPARING.with(|p| p.set(was));
    r
}

/// Background threads that prepare lazy pipelines' shaders: half the compile threads
/// (`XCL_PREPARE_THREADS` to choose), so a title thread's own compile always finds a free DXC
/// worker and the title keeps half the CPU. 0: none.
pub fn prepare_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var("XCL_PREPARE_PSO").is_ok_and(|v| v == "0") {
            return 0;
        }
        let n = crate::gpu::recomp::dxc::compile_threads();
        std::env::var("XCL_PREPARE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(n.div_ceil(2)).min(n)
    })
}

/// Prepare `p`'s shaders, unless it was built, started or released already.
fn prepare(p: &Pending) {
    if p.orphan.load(Ordering::Acquire) || p.full_started.load(Ordering::Acquire) || *p.done.lock().unwrap() {
        return;
    }
    let Some(job) = job_of(p) else { return };
    PREPARING.with(|f| f.set(true));
    let made = run(p, &*job);
    PREPARING.with(|f| f.set(false));
    release(made);
    let n = PREPARED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 || n % 2000 == 0 {
        let (running, compiled) = crate::gpu::recomp::dxc::compile_counts();
        crate::xlog_info!("D3D12.X: shaders of {n} pipelines prepared in the background ({compiled} DXC compiles so far, {running} running)");
    }
}

static PREPARED: AtomicUsize = AtomicUsize::new(0);

fn prepare_queue() -> Option<&'static Mutex<mpsc::Sender<Weak<Pending>>>> {
    static Q: OnceLock<Option<Mutex<mpsc::Sender<Weak<Pending>>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let n = prepare_threads();
        if n == 0 {
            return None;
        }
        let (tx, rx) = mpsc::channel::<Weak<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-prep-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                while (super::real::background_memory_paused() || super::real::live_compile_waiting()) && p.strong_count() != 0 {
                    std::thread::sleep(PARKED_POLL);
                }
                let _task = crate::progress::PIPELINES.take();
                // (None: the stand-in was released)
                if let Some(p) = p.upgrade() {
                    prepare(&p);
                }
            });
        }
        crate::xlog_info!("D3D12.X: shaders of pipelines made for later are prepared on {n} background threads");
        Some(Mutex::new(tx))
    })
    .as_ref()
}

#[derive(Default)]
struct BuildQueue {
    graphics: VecDeque<Arc<Pending>>,
    compute: VecDeque<Arc<Pending>>,
    /// optimized builds replacing fast-tier pipelines: after everything else
    upgrades: VecDeque<Arc<Pending>>,
    /// builds a bind waits for, with when the bind asked: before everything else
    urgent: VecDeque<(Arc<Pending>, std::time::Instant)>,
    /// builds of pipelines whose draws were skipped (`skip_until_built`), the last bound last:
    /// taken newest first, after `urgent` and before the rest, as background builds
    bound: VecDeque<Arc<Pending>>,
    /// builds the title patches put first (`build_first`): before everything, on every worker
    first: VecDeque<Arc<Pending>>,
    /// passes over the screen (`mark_screen_pass`): after `first`, and the only builds the
    /// screen workers take
    screen: VecDeque<Arc<Pending>>,
}

impl BuildQueue {
    fn push(&mut self, p: Arc<Pending>) {
        if p.compute {
            self.compute.push_back(p);
        } else {
            self.graphics.push_back(p);
        }
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        self.graphics.pop_front().or_else(|| self.compute.pop_front())
    }
}

static QUEUE: Mutex<BuildQueue> = Mutex::new(BuildQueue {
    graphics: VecDeque::new(),
    compute: VecDeque::new(),
    upgrades: VecDeque::new(),
    urgent: VecDeque::new(),
    bound: VecDeque::new(),
    first: VecDeque::new(),
    screen: VecDeque::new(),
});

/// pipelines the bind queue holds at most: the ones bound longest ago fall out (a later bind
/// sends a pipeline again)
const BOUND_QUEUE: usize = 4096;

fn enqueue_bound(p: Arc<Pending>) {
    if p.first.load(Ordering::Acquire) {
        return enqueue_first(p);
    }
    start_workers();
    crate::progress::PIPELINES.queue();
    let mut queue = QUEUE.lock().unwrap();
    queue.bound.push_back(p);
    if queue.bound.len() > BOUND_QUEUE {
        if let Some(old) = queue.bound.pop_front() {
            old.in_bound.store(false, Ordering::Release);
        }
        drop(crate::progress::PIPELINES.take());
    }
    BOUND_WAITING.store(queue.bound.len(), Ordering::Release);
    drop(queue);
    READY.notify_one();
}

/// pipelines in the bind queue no worker has taken (`bound_first`)
static BOUND_WAITING: AtomicUsize = AtomicUsize::new(0);
/// bind-queue builds taken by their skipped draws (`take_bound`)
static BOUND_TAKEN: AtomicUsize = AtomicUsize::new(0);

/// The bind queue holds each pipeline whose draws are skipped once and gives a worker the one
/// with the most draws and dispatches skipped while it waited, the newest of equals first
/// (`XCL_BOUND_BY_SKIPS=0`: every frame's skip queues it again, the last bound taken first).
/// play-97 ended with 654 pipelines pending, the 181 tiled-lighting variants drawn every frame
/// among them, behind builds bound once.
pub(super) fn bound_by_skips() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_BOUND_BY_SKIPS").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: the bind queue takes the pipeline bound last first (XCL_BOUND_BY_SKIPS=0)");
        }
        on
    })
}

/// While pipelines whose draws are skipped wait in the bind queue for a worker, the preload
/// workers start nothing new: the CPU and memory go to what the frame draws
/// (`XCL_BOUND_FIRST=0`: they go on). In play-98 small compiles ran 5 to 10 times their
/// expected time beside the preload's, while 797 pipelines stayed pending.
fn bound_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = bound_by_skips() && std::env::var("XCL_BOUND_FIRST").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: preload builds go on while skipped draws wait for theirs (XCL_BOUND_FIRST=0)");
        }
        on
    })
}

fn bound_waiting() -> bool {
    bound_first() && BOUND_WAITING.load(Ordering::Acquire) != 0
}

/// Of the skip counts of the bind queue's pipelines (oldest first), the one to build next: the
/// most skipped, the newest of equals. (A ray pass built with every ray a miss ranks below every
/// other pipeline: `rt_pass_last`.)
fn most_skipped<K: Ord>(skips: impl Iterator<Item = K>) -> Option<usize> {
    skips.enumerate().max_by(|a, b| (&a.1, a.0).cmp(&(&b.1, b.0))).map(|(at, _)| at)
}

/// The next pipeline of the bind queue (`bound_by_skips`); built and released ones leave it.
fn take_bound(bound: &mut VecDeque<Arc<Pending>>) -> Option<Arc<Pending>> {
    if !bound_by_skips() {
        return bound.pop_back();
    }
    bound.retain(|p| {
        let wanted = p.real.load(Ordering::Acquire) == 0 && !p.orphan.load(Ordering::Acquire);
        if !wanted {
            p.in_bound.store(false, Ordering::Release);
        }
        wanted
    });
    let at = most_skipped(bound.iter().map(|p| (!p.rt_pass.load(Ordering::Relaxed), p.skips_waiting.load(Ordering::Relaxed))));
    let p = at.and_then(|at| bound.remove(at));
    BOUND_WAITING.store(bound.len(), Ordering::Release);
    let p = p?;
    p.in_bound.store(false, Ordering::Release);
    let skipped = p.skips_waiting.swap(0, Ordering::Relaxed);
    let n = BOUND_TAKEN.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: bind queue: {n} builds taken by their skipped draws, the latest with {skipped} skipped while it waited ({}; {} more queued)",
            if p.what.is_empty() { "shaders unknown" } else { &p.what },
            bound.len()
        );
    }
    Some(p)
}

/// threads that build only what binds wait for, besides the workers (which take it first too)
const URGENT_THREADS: usize = 4;

/// The bind queue's workers besides the `URGENT_THREADS`: as many as the preload has
/// (`preload_threads`), or DXC's pool when that is more (`XCL_BIND_WORKERS_BY_CORES=0`: DXC's
/// pool, sized once by the memory left over its 6 GB reserve when it is first asked for;
/// `XCL_BIND_WORKERS` sets the count). Each compile is still admitted by the memory it takes
/// (`real::driver_slot`). play-101 built its skipped draws' pipelines on 4 threads, 2 more held
/// by FFX_FSR4 giants, while its 12 preload workers waited behind the bind queue (`bound_first`)
/// with ~9 GB available: ~1,850 pending for 500 s, 128 bind-queue builds in 740 s.
fn bind_workers() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let dxc = crate::gpu::recomp::dxc::compile_threads();
        let set = std::env::var("XCL_BIND_WORKERS").ok().and_then(|v| v.trim().parse::<usize>().ok());
        let by_cores = std::env::var("XCL_BIND_WORKERS_BY_CORES").map_or(true, |v| v.trim() != "0");
        let n = set.map_or(if by_cores { dxc.max(preload_threads()) } else { dxc }, |n| n.clamp(1, 64));
        crate::xlog_info!(
            "D3D12.X: the bind queue is built on {n} workers, and {URGENT_THREADS} more take only builds a bind waits for (DXC's pool: {dxc}; XCL_BIND_WORKERS sets the count{})",
            if by_cores { ", XCL_BIND_WORKERS_BY_CORES=0 keeps DXC's" } else { "" }
        );
        n
    })
}
static READY: Condvar = Condvar::new();

pub(super) fn diagnostic_queued() -> impl Iterator<Item = crate::live_stall::Sample> {
    use crate::live_stall::{Sample, Stage};
    let mut samples = [None; 4];
    if let Ok(q) = QUEUE.try_lock() {
        for (slot, (p, since)) in samples.iter_mut().zip(q.urgent.iter()) {
            *slot = Some(Sample {
                stage: Stage::Queued, since: *since, tid: 0, pipeline: p.record, shader: p.key, target: None, queue: None,
            });
        }
    }
    samples.into_iter().flatten()
}

fn start_workers() {
    static WORKERS: std::sync::Once = std::sync::Once::new();
    WORKERS.call_once(|| {
        let n = bind_workers();
        let screens = if screen_budget().is_some() { SCREEN_THREADS } else { 0 };
        for k in 0..n + URGENT_THREADS + screens {
            let screen_only = k >= n + URGENT_THREADS;
            let urgent_only = k >= n && !screen_only;
            let name = if screen_only {
                format!("xcl-pso-screen-{}", k - n - URGENT_THREADS)
            } else if urgent_only {
                format!("xcl-pso-urgent-{}", k - n)
            } else {
                format!("xcl-pso-{k}")
            };
            let _ = std::thread::Builder::new().name(name).stack_size(64 << 20).spawn(move || loop {
                let (p, upgrade, asked) = {
                    let mut queue = QUEUE.lock().unwrap();
                    loop {
                        if screen_only {
                            match queue.screen.pop_front() {
                                Some(p) => break (p, false, None),
                                None => {
                                    queue = SCREEN_READY.wait(queue).unwrap();
                                    continue;
                                }
                            }
                        }
                        if let Some(p) = queue.first.pop_front() {
                            break (p, false, None);
                        }
                        if let Some(p) = queue.screen.pop_front() {
                            break (p, false, None);
                        }
                        let next = queue.urgent.pop_front().map(|(p, at)| (p, false, Some(at)));
                        if let Some(next) = next {
                            break next;
                        }
                        if !urgent_only && !queue.bound.is_empty() {
                            if super::real::background_memory_paused() {
                                queue = READY.wait_timeout(queue, PARKED_POLL).unwrap().0;
                                continue;
                            }
                            if let Some(p) = take_bound(&mut queue.bound) {
                                break (p, false, None);
                            }
                        }
                        if !urgent_only && (!queue.compute.is_empty() || !queue.graphics.is_empty() || !queue.upgrades.is_empty()) {
                            if super::real::background_memory_paused() || super::real::live_compile_waiting() {
                                queue = READY.wait_timeout(queue, PARKED_POLL).unwrap().0;
                                continue;
                            }
                            if let Some(next) = queue.pop().map(|p| (p, false, None)).or_else(|| queue.upgrades.pop_front().map(|p| (p, true, None))) {
                                break next;
                            }
                        }
                        queue = READY.wait(queue).unwrap();
                    }
                };
                let _task = crate::progress::PIPELINES.take();
                BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
                let started = std::time::Instant::now();
                if upgrade {
                    build_full(&p);
                } else {
                    build_tiered(&p, asked.is_none() && super::real::live_lanes() && !p.first.load(Ordering::Acquire) && !p.screen.load(Ordering::Acquire));
                }
                BUILDING.with(|b| *b.borrow_mut() = None);
                if let Some(at) = asked {
                    let (queued, built) = (started.duration_since(at), started.elapsed());
                    if queued + built >= std::time::Duration::from_millis(100) {
                        crate::xlog_info!(
                            "D3D12.X: a waited-for {} pipeline: {} ms in the queue, {} ms to build{}; {}",
                            if p.compute { "compute" } else { "graphics" },
                            queued.as_millis(),
                            built.as_millis(),
                            if p.rebuild.load(Ordering::Acquire) { " (a rebuild of one released unbound)" } else { "" },
                            p.note.lock().unwrap().as_deref().unwrap_or("no build details")
                        );
                    }
                }
            });
        }
    });
}

thread_local! {
    /// the pipeline this worker builds now (`waited_on`)
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// What the pipeline this worker builds for a waiting bind was made of and how (`f`: shader
/// hashes, DXIL sizes, translation and driver times; logged with the wait).
pub fn note_build(f: impl FnOnce() -> String) {
    BUILDING.with(|b| {
        if let Some(p) = b.borrow().as_ref().filter(|p| p.urgent.load(Ordering::Acquire)) {
            *p.note.lock().unwrap() = Some(f());
        }
    });
}

/// builds a bind started, the ms binds waited, and draws skipped (`backstop`), since `take_window`
static URGENT_BUILDS: AtomicUsize = AtomicUsize::new(0);
static WAIT_MS: AtomicUsize = AtomicUsize::new(0);
static SKIPS: AtomicUsize = AtomicUsize::new(0);
static WINDOW: Mutex<(usize, usize, usize, usize)> = Mutex::new((0, 0, 0, 0));

/// (builds a bind started, binds that waited, ms they waited in all, draws skipped) since the
/// last call
pub fn take_window() -> (usize, usize, usize, usize) {
    let (bind, used, _) = waits();
    let now = (URGENT_BUILDS.load(Ordering::Relaxed), bind + used, WAIT_MS.load(Ordering::Relaxed), SKIPS.load(Ordering::Relaxed));
    let last = std::mem::replace(&mut *WINDOW.lock().unwrap(), now);
    (now.0 - last.0, now.1 - last.1, now.2 - last.2, now.3 - last.3)
}

/// draws skipped so far (`backstop`)
pub fn skips() -> usize {
    SKIPS.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// (tests) this thread's backstop instead of `XCL_BIND_BACKSTOP_MS`
    pub(super) static BACKSTOP: Cell<Option<Option<std::time::Duration>>> = const { Cell::new(None) };
    /// (tests) this thread's frame wait budget instead of `XCL_FRAME_STALL_BUDGET_MS`
    pub(super) static FRAME_BUDGET: Cell<Option<Option<std::time::Duration>>> = const { Cell::new(None) };
    pub(super) static COMPUTE_SKIP_LIMIT: Cell<Option<std::time::Duration>> = const { Cell::new(None) };
    /// (tests) this thread's `lazy_mesh` instead of `XCL_LAZY_MESH_PSO`
    pub(super) static LAZY_MESH: Cell<Option<bool>> = const { Cell::new(None) };
    /// (tests) this thread's `skip_until_built` (off unless a test sets it)
    pub(super) static SKIP_UNTIL_BUILT: Cell<bool> = const { Cell::new(false) };
}

thread_local! {
    /// this thread's binds wait for their pipelines (`waiting_if`)
    static WAIT_FOR_BUILD: Cell<bool> = const { Cell::new(false) };
}

/// `f` with this thread's binds waiting for pipelines still being built when `on` (a bundle:
/// recorded once, a draw skipped in it would never come back).
pub(super) fn waiting_if<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let was = WAIT_FOR_BUILD.with(|w| w.replace(on));
    let r = f();
    WAIT_FOR_BUILD.with(|w| w.set(was));
    r
}

/// `enabled`, read once.
pub(super) fn async_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(enabled)
}

/// `XCL_ASYNC_PSO` (default 1, `enabled`): a draw or dispatch whose pipeline is still being
/// built does not wait for it. Its build goes ahead of the others, ordered by the binds that
/// asked for it, and the draw is skipped (`bind::skip_failed_draws`; a dispatch only with
/// `bind::skip_failed_dispatches`, else it waits as before) until the pipeline is built.
fn skip_until_built() -> bool {
    if WAIT_FOR_BUILD.with(Cell::get) {
        return false;
    }
    #[cfg(test)]
    return SKIP_UNTIL_BUILT.with(Cell::get);
    #[cfg(not(test))]
    async_on()
}

/// Every 2 s while pipelines are pending or binds skip: pending (wanted, not built), built,
/// failed, the draws and dispatches skipped in those 2 s and the oldest pending pipeline's age.
/// One walk of the pending pipelines: those sent and not done (`counts`), those wanted (sent to
/// be built) and not built yet, and how long the oldest of the latter has waited.
fn walk_pending() -> Walk {
    let mut w = Walk { building: 0, wanted: 0, oldest: None };
    if let Some(m) = PENDING.lock().unwrap().as_ref() {
        for p in m.values().filter(|p| p.sent.load(Ordering::Acquire) && !*p.done.lock().unwrap()) {
            w.building += 1;
            if p.real.load(Ordering::Acquire) == 0 {
                w.wanted += 1;
                w.oldest = w.oldest.max(Some(p.created.elapsed()));
            }
        }
    }
    w
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Walk {
    building: usize,
    wanted: usize,
    oldest: Option<std::time::Duration>,
}

/// The status thread's latest walk and when it was taken.
static WALKED: Mutex<Option<(std::time::Instant, Walk)>> = Mutex::new(None);

/// What the console status line and the overlay show of the pipelines, from the status thread's
/// latest walk (at most 2 s old) so that neither walks `PENDING` itself: (built, failed,
/// building) as `counts` gives them, and the pipelines wanted and not built with the oldest's
/// age now.
pub fn status_counts() -> ((usize, usize, usize), (usize, Option<std::time::Duration>)) {
    shown(*WALKED.lock().unwrap(), BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed))
}

fn shown(walked: Option<(std::time::Instant, Walk)>, built: usize, failed: usize) -> ((usize, usize, usize), (usize, Option<std::time::Duration>)) {
    let (building, wanted, oldest) = walked.map_or((0, 0, None), |(at, w)| (w.building, w.wanted, w.oldest.map(|d| d + at.elapsed())));
    ((built, failed, building), (wanted, oldest))
}

pub(super) fn start_status() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-status".into()).spawn(|| {
            let mut skipped = pending_skips();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = pending_skips();
                let (built, failed) = (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed));
                let walk = walk_pending();
                *WALKED.lock().unwrap() = Some((std::time::Instant::now(), walk));
                let (pending, oldest) = (walk.wanted, walk.oldest);
                if pending != 0 || now != skipped {
                    crate::xlog_info!(
                        "D3D12.X: async pipelines: {pending} pending (oldest {:.1} s), {built} built, {failed} failed; {} draws/dispatches skipped in the last 2 s ({now} so far){}",
                        oldest.map_or(0.0, |d| d.as_secs_f32()),
                        now - skipped,
                        super::raytracing::status().unwrap_or_default()
                    );
                }
                skipped = now;
            }
        });
    });
}

thread_local! {
    /// this thread's bind waits in the frame being made: (frames shown when it began, time waited)
    static FRAME_WAIT: Cell<(u64, std::time::Duration)> = const { Cell::new((u64::MAX, std::time::Duration::ZERO)) };
}

/// draws skipped at once because their thread had used up its frame's wait budget
static BUDGET_SKIPS: AtomicUsize = AtomicUsize::new(0);

/// How long one thread's binds may wait for pipelines in one frame (until the next present)
/// before its draws with graphics pipelines still being built are skipped at once
/// (`XCL_FRAME_STALL_BUDGET_MS`, 1000 by default; 0: no budget). Dispatches always wait. A run
/// of new pipelines waited for one after another would otherwise hold a frame past the title's
/// GPU watchdog.
fn frame_budget() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(b) = FRAME_BUDGET.with(Cell::get) {
        return b;
    }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_FRAME_STALL_BUDGET_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1000u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// What this thread's binds have waited in the frame being made (a new frame starts at zero).
fn frame_waited() -> std::time::Duration {
    let frame = super::present::presents();
    FRAME_WAIT.with(|w| match w.get() {
        (f, waited) if f == frame => waited,
        _ => {
            w.set((frame, std::time::Duration::ZERO));
            std::time::Duration::ZERO
        }
    })
}

fn add_frame_wait(waited: std::time::Duration) {
    FRAME_WAIT.with(|w| {
        let (f, total) = w.get();
        w.set((f, total + waited));
    });
}

/// draws skipped at once for the frame wait budget so far (`frame_budget`)
pub fn budget_skips() -> usize {
    BUDGET_SKIPS.load(Ordering::Relaxed)
}

/// How long a draw waits for its graphics pipeline before it is skipped for that bind
/// (`XCL_BIND_BACKSTOP_MS`, 3000 by default; 0: for ever). The build goes on ahead of the others
/// and a later bind uses it. A dispatch always waits: a skipped one could leave what the title
/// computes wrong (it is logged once it waited this long).
fn backstop() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(b) = BACKSTOP.with(Cell::get) {
        return b;
    }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_BIND_BACKSTOP_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3000u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// A bind waits for the pipeline this thread builds now (it may have started as background work
/// before): its driver compile goes ahead of the memory governor's limits.
pub fn waited_on() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.urgent.load(Ordering::Acquire)))
}

/// The pipeline this worker builds now was bound and its draws skipped (`skip_until_built`): a
/// live build, ahead of the background ones (`real::live_priority`).
pub(super) fn live_build() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.last_bound.load(Ordering::Acquire) != 0))
}

pub(super) fn blocking_waiter() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.active_waiters.load(Ordering::Acquire) != 0))
}

fn enqueue(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().push(p);
    READY.notify_one();
}

fn enqueue_urgent(p: Arc<Pending>) {
    URGENT_BUILDS.fetch_add(1, Ordering::Relaxed);
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().urgent.push_back((p, std::time::Instant::now()));
    READY.notify_all();
}

fn enqueue_upgrade(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().upgrades.push_back(p);
    READY.notify_one();
}

fn enqueue_first(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().first.push_back(p);
    READY.notify_all();
}

/// Title patches (`title_gta6`): `pso`'s build goes before every other one, now and when bound,
/// on any worker, as a compile a bind waits for (`first_build`), never handed to the giant lane.
/// False: not a pipeline building in the background, built already, or marked before.
pub(super) fn build_first(pso: usize) -> bool {
    let Some(p) = entry(pso) else { return false };
    if *p.done.lock().unwrap() || p.first.swap(true, Ordering::AcqRel) {
        return false;
    }
    p.sent.store(true, Ordering::Release);
    if !p.promote_preload() {
        enqueue_first(p);
    }
    true
}

/// The pipeline this worker builds now was put first (`build_first`).
pub fn first_build() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.first.load(Ordering::Acquire)))
}

/// `pso` is a stage of the acceleration-structure builder (`raytracing::note_builder_stage`): the real
/// pipeline it gets is known as one too, so a draw or dispatch with either is held as the builder is.
pub(super) fn mark_builder(pso: usize) {
    let Some(p) = entry(pso) else { return };
    p.builder.store(true, Ordering::Release);
    let real = p.real.load(Ordering::Acquire);
    if real != 0 {
        super::raytracing::note_builder_identity(real);
    }
}

/// `pso` can be used as it is: a stand-in whose real pipeline is built, or no stand-in at all.
pub(super) fn ready(pso: usize) -> bool {
    entry(pso).is_none_or(|p| p.real.load(Ordering::Acquire) != 0)
}

/// `p`'s build is waited for: before every other on the workers (a preload worker building it
/// is raised instead), once.
fn urge(p: &Arc<Pending>) {
    if p.urgent.swap(true, Ordering::AcqRel) {
        return;
    }
    if p.compute && !*p.done.lock().unwrap() && !p.counted.swap(true, Ordering::AcqRel) {
        COMPUTE_BOUND_PENDING.fetch_add(1, Ordering::AcqRel);
    }
    if !p.promote_preload() {
        enqueue_urgent(p.clone());
    }
}

/// `XCL_DEFER_TO_SUBMIT` (default 1): a draw or dispatch that `skip_until_built` would skip
/// for want of its build gets a segment of its own, recorded at the submit (`list_x::defer`),
/// and the submit waits for the build (`wait_for_submit`): a pass the title records once (a LUT
/// bake, an init pass) is not lost for the whole run. 0: such a call is skipped, as before.
pub(super) fn defer_to_submit() -> bool {
    #[cfg(test)]
    if let Some(on) = DEFER_FOR_TEST.with(Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_DEFER_TO_SUBMIT").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: draws and dispatches whose pipeline is still being built are skipped, not put off to the submit (XCL_DEFER_TO_SUBMIT=0)");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// `XCL_DEFER_TO_SUBMIT` for this test thread
    static DEFER_FOR_TEST: Cell<Option<bool>> = const { Cell::new(None) };
    /// `SUBMIT_WAIT_MOST` for this test thread
    static SUBMIT_WAIT_FOR_TEST: Cell<Option<std::time::Duration>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(super) fn set_defer_for_test(on: Option<bool>) {
    DEFER_FOR_TEST.with(|d| d.set(on));
}

#[cfg(test)]
pub(super) fn set_submit_wait_for_test(most: Option<std::time::Duration>) {
    SUBMIT_WAIT_FOR_TEST.with(|d| d.set(most));
}

/// Whether a call with `pso` would be skipped now for want of its build (`skip_until_built`) and
/// may be put off to the submit instead (`defer_to_submit`). Not a pipeline left out (ray
/// tracing off: it is never built), an image-only one (`XCL_PSO_IMAGE_SKIP`: skipped by choice)
/// or one a submit already gave up on (a cold giant held for memory: later frames skip it as
/// before instead of standing still at every submit).
pub(super) fn deferrable(pso: usize) -> bool {
    if !defer_to_submit() || !skip_until_built() {
        return false;
    }
    let Some(p) = entry(pso) else { return false };
    if p.left_out.load(Ordering::Acquire) || p.image_only.load(Ordering::Acquire) || p.submit_gave_up.load(Ordering::Acquire) || *p.done.lock().unwrap() {
        return false;
    }
    if p.screen.load(Ordering::Acquire) && screen_left().is_some() {
        return false;
    }
    if p.compute { super::bind::skip_failed_dispatches() } else { super::bind::skip_failed_draws() }
}

/// A call with `pso` was put off to the submit (`list_x::defer`): the submit will wait for its
/// build, which goes before the others on the workers, as one a bind waits for.
pub(super) fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    p.note_bind();
    start_status();
    p.sent.store(true, Ordering::Release);
    urge(&p);
}

/// `XCL_DEFER_DRAWS` (default 1): draws are put off to the submit as dispatches are
/// (`defer_to_submit`). 0: only dispatches are; a draw whose pipeline is still being built is
/// skipped, as before.
pub(super) fn defer_draws() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_DEFER_DRAWS").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: draws whose pipeline is still being built are skipped, not put off to the submit; dispatches still are (XCL_DEFER_DRAWS=0)");
        }
        on
    })
}

/// Whether `pso` is a compute pipeline, its shaders for the log and its real pipeline (0: not
/// built) – None: not a pipeline built here.
pub(super) fn kind_of(pso: usize) -> Option<(bool, String, usize)> {
    entry(pso).map(|p| (p.compute, p.what.clone(), p.real.load(Ordering::Acquire)))
}

/// How long one submit waits at most for the pipelines of the calls put off to it, all of them
/// together (`XCL_SUBMIT_WAIT_MS`, at most 30 s; 0: it does not wait, and such a call not built
/// by its submit is skipped). The submitting thread is the title's, which a retail title's
/// watchdog lets stand still 30 s at most, and what other queues wait for (a fence it signals)
/// goes in only after the waits: a build that takes longer than this (a cold giant, an outlined
/// DXC build) is not waited for to its end.
const SUBMIT_WAIT_MOST: std::time::Duration = std::time::Duration::from_secs(5);

fn submit_wait_most() -> std::time::Duration {
    #[cfg(test)]
    if let Some(most) = SUBMIT_WAIT_FOR_TEST.with(Cell::get) {
        return most;
    }
    static MOST: OnceLock<std::time::Duration> = OnceLock::new();
    *MOST.get_or_init(|| {
        let set = std::env::var("XCL_SUBMIT_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok());
        let most = set.map_or(SUBMIT_WAIT_MOST, |ms| std::time::Duration::from_millis(ms.min(30_000)));
        if set.is_some() {
            crate::xlog_info!("D3D12.X: a submit waits {} ms at most for the pipelines of the calls put off to it (XCL_SUBMIT_WAIT_MS)", most.as_millis());
        }
        most
    })
}

/// When the waits of a submit starting now end (`wait_for_submit`).
pub(super) fn submit_deadline() -> std::time::Instant {
    std::time::Instant::now() + submit_wait_most()
}

/// put-off calls whose pipeline was not built by their submit's deadline
static SUBMIT_GAVE_UP: AtomicUsize = AtomicUsize::new(0);
/// put-off calls skipped at their submit without a wait, a submit having given up on their
/// pipeline already (`wait_for_submit`)
static SUBMIT_SKIPPED_AT_ONCE: AtomicUsize = AtomicUsize::new(0);

/// A submit records a call put off to it (`list_x::record_deferred`): wait until `pso` is built
/// (or its build failed) or until `deadline`, whichever comes first. The submitting thread is
/// held meanwhile (`title_held`), as a bind's wait holds it, and the build goes before the others.
/// The workers and the memory governor build it as any other: one held for memory or put off is
/// waited for no longer than `deadline` either. False: not built by then – the call goes the way
/// it went before calls were put off (skipped and counted), and later calls with it are not put
/// off again. One a submit gave up on already (put off before that) is not waited for again.
pub(super) fn wait_for_submit(pso: usize, deadline: std::time::Instant) -> bool {
    let Some(p) = entry(pso) else { return true };
    if *p.done.lock().unwrap() {
        return true;
    }
    if p.submit_gave_up.load(Ordering::Acquire) {
        let n = SUBMIT_SKIPPED_AT_ONCE.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: deferred {} skipped at its submit without a wait: a submit gave up on its pipeline {pso:#x} already (#{n})",
                if p.compute { "dispatch" } else { "draw" }
            );
        }
        return false;
    }
    let _held = Held::new(&p);
    let _live = crate::live_stall::begin(crate::live_stall::Stage::PipelineWait, p.record, p.key, None, None);
    urge(&p);
    let t0 = std::time::Instant::now();
    let built = {
        let mut done = p.done.lock().unwrap();
        while !*done {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                break;
            }
            done = p.cv.wait_timeout(done, left).unwrap().0;
        }
        *done
    };
    let ms = t0.elapsed().as_millis() as usize;
    WAIT_MS.fetch_add(ms, Ordering::Relaxed);
    let (kind, what) = (if p.compute { "compute" } else { "graphics" }, if p.what.is_empty() { "shaders unknown" } else { &p.what });
    if built {
        if ms >= 250 {
            crate::xlog_info!("D3D12.X: a submit waited {ms} ms for {kind} pipeline {pso:#x} ({what}) of a call put off to it");
        }
        return true;
    }
    p.submit_gave_up.store(true, Ordering::Release);
    if p.key != 0 && !p.skipped.swap(true, Ordering::AcqRel) {
        used::note_skipped(p.key);
    }
    let n = SUBMIT_GAVE_UP.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        crate::xlog_warn!(
            "D3D12.X: deferred {} skipped after waiting {ms} ms for its pipeline (a submit waits {} ms at most; {n} so far): {kind} pipeline {pso:#x} ({what}) – later calls with it are skipped until it is built",
            if p.compute { "dispatch" } else { "draw" },
            submit_wait_most().as_millis()
        );
    }
    false
}

/// put-off calls skipped at their submit so far (`wait_for_submit`)
#[cfg(test)]
fn submit_gave_up() -> usize {
    SUBMIT_GAVE_UP.load(Ordering::Relaxed)
}

/// Ray tracing off (`raytracing::left_out`): `pso` is never built; its draws and dispatches are
/// skipped as a failed build's, and counted (`raytracing::note_dispatch`).
pub(super) fn give_up(pso: usize) {
    let Some(p) = entry(pso) else { return };
    p.left_out.store(true, Ordering::Release);
    p.sent.store(true, Ordering::Release);
    *p.job.lock().unwrap_or_else(|e| e.into_inner()) = None;
    complete(&p, 0, false);
}

/// `give_up` for a pipeline whose build is not wanted any more (`fsr4_standin`). None: nothing to
/// drop, it is built, failed or left out already (or no stand-in). Some(started): a worker had
/// started its build – that build runs on to its next wait for a driver compile's memory
/// (`building_left_out`); one no worker had taken is never translated or compiled.
pub(super) fn drop_build(pso: usize) -> Option<bool> {
    let p = entry(pso)?;
    if p.real.load(Ordering::Acquire) != 0 || *p.done.lock().unwrap() {
        return None;
    }
    let started = p.full_started.load(Ordering::Acquire) || p.tiered_started.load(Ordering::Acquire) || p.preload.lock().unwrap_or_else(|e| e.into_inner()).worker.is_some();
    give_up(pso);
    Some(started)
}

/// The pipeline this worker builds was left out since the build started (`drop_build`): its
/// driver compile is not to start (`real::Governor::acquire_within`), and a helper process
/// compiling it is ended (`real::giant_helper`).
pub(super) fn building_left_out() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.left_out.load(Ordering::Acquire)))
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        super::shader_code::gone(p);
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, (0, 0), String::new(), 0, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built in the background (`submit_lazy`), not in the call. A
/// retail game makes ~51 000 pipelines in its first 30 s and uses few; each one built stays in
/// the driver (~400 KB), and the process ran out of commit – so one nobody binds is released
/// again (`idle_limit`).
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// Console mesh pipelines are lazy too (`XCL_LAZY_MESH_PSO=1`; off unless set): the title gets a
/// stand-in, and the pipeline is built when first bound or by the background workers. Without
/// it a mesh pipeline is built in the title's call and kept until the title releases it: a
/// retail game made 24 600 while it loaded, before it drew with any (two driver pipelines each
/// when made in two passes, ~150 KB of an NVIDIA driver's memory a pair, which that driver
/// keeps when the pipelines are released).
pub fn lazy_mesh() -> bool {
    #[cfg(test)]
    if let Some(on) = LAZY_MESH.with(Cell::get) {
        return on;
    }
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| lazy() && std::env::var("XCL_LAZY_MESH_PSO").is_ok_and(|v| v == "1"))
}

/// (tests) `pso` is a stand-in the title holds, built or not.
#[cfg(test)]
pub(super) fn is_standin(pso: usize) -> bool {
    entry(pso).is_some()
}

/// `submit`, built by the background workers (`preload`) below normal priority, in the order of
/// `pso_sched::Backlog`: the ones with the most console code first, then the ones of shaders an
/// earlier run used (`key`, 0 = none; `used`), then compute, then graphics. A bind sends it
/// ahead of all of them. (`XCL_PSO_PRELOAD=0`: built at its first `resolve`, its shaders prepared
/// before.)
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, (key, 0), String::new(), 0, 0);
}

/// `submit_lazy` with what the pipeline is made of (`what`: its shaders, for the log;
/// `shader_bytes`: its console shaders' size).
#[cfg(test)]
pub fn submit_lazy_described(standin: usize, compute: bool, job: Job, key: u64, what: String, shader_bytes: usize, group: u64) {
    submit_as(standin, compute, job, false, (key, 0), what, shader_bytes, group);
}

pub(super) fn submit_recorded(standin: usize, compute: bool, job: Job, (key, record): (u64, u64), what: String, shader_bytes: usize, group: u64) {
    let send = !lazy();
    submit_as(standin, compute, job, send, (if send { 0 } else { key }, record), what, shader_bytes, group);
}

#[allow(clippy::too_many_arguments)]
fn submit_as(standin: usize, compute: bool, job: Job, send: bool, (key, record): (u64, u64), what: String, shader_bytes: usize, group: u64) {
    let p = Arc::new(Pending {
        created: std::time::Instant::now(),
        last_bound: AtomicU64::new(0),
        job: Mutex::new(Some(Arc::from(job))),
        sent: std::sync::atomic::AtomicBool::new(send),
        preload: Default::default(),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        image_only: Default::default(),
        depth_only: Default::default(),
        screen: Default::default(),
        rt_pass: Default::default(),
        key,
        record,
        used: Default::default(),
        orphan: Default::default(),
        full_started: Default::default(),
        tiered_started: Default::default(),
        real_fast: Default::default(),
        urgent: Default::default(),
        active_waiters: Default::default(),
        counted: Default::default(),
        retired: Default::default(),
        bound: Default::default(),
        built_at: Default::default(),
        released_idle: Default::default(),
        never_fit: Default::default(),
        rebuild: Default::default(),
        note: Default::default(),
        what,
        skipped: Default::default(),
        group,
        preload_seq: Default::default(),
        first: Default::default(),
        left_out: Default::default(),
        builder: Default::default(),
        skips_waiting: Default::default(),
        in_bound: Default::default(),
        submit_gave_up: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else {
        start_idle_release();
        if let Some(q) = preload() {
            crate::progress::PIPELINES.queue();
            let known = used::known(key) || super::pso_manifest::bound_earlier(record);
            let n = PRELOAD_SEQ.fetch_add(1, Ordering::Relaxed);
            p.preload_seq.store(n, Ordering::Relaxed);
            let first_use = if known { first_use(key, record) } else { None };
            q.change(Change::Push { n, shader_bytes, known, compute, group, slow_ms: super::pso_manifest::slow_ms(record), first_use, item: Arc::downgrade(&p) });
        } else if let Some(q) = prepare_queue() {
            crate::progress::PIPELINES.queue();
            if q.lock().unwrap().send(Arc::downgrade(&p)).is_err() {
                drop(crate::progress::PIPELINES.take());
            }
        }
    }
}

/// Where an earlier run first used a pipeline, for the order the preload workers take the known
/// ones in (`pso_sched::Backlog::push_as`, smaller first): the exact pipelines the last runs bound,
/// in the order they were first bound (`pso_manifest::bound_order`), then the others of shader sets
/// an earlier run used, in the order those were first used (`used::rank`). Without it the known
/// lane took the newest made first, so of the thousands the title makes in a burst while it loads
/// the ones its first frames draw with could come last. `XCL_KNOWN_FIRST_USE=0`: as before.
fn first_use(key: u64, record: u64) -> Option<u64> {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var("XCL_KNOWN_FIRST_USE").map_or(true, |v| v.trim() != "0");
        if on {
            crate::xlog_info!(
                "D3D12.X: pipelines earlier runs used are built ahead in the order they were first used: {} exact ones as last bound, then the rest by shader set (XCL_KNOWN_FIRST_USE=0: the newest made first)",
                super::pso_manifest::bound_order().len()
            );
        } else {
            crate::xlog_info!("D3D12.X: pipelines earlier runs used are built ahead the newest made first (XCL_KNOWN_FIRST_USE=0)");
        }
        on
    });
    if !on {
        return None;
    }
    let exact = super::pso_manifest::bound_order().get(&record).map(|&rank| rank as u64);
    exact.or_else(|| used::rank(key).map(|rank| (1u64 << 32) + rank as u64))
}

/// The pipelines earlier runs used, by the hash of their shaders (`xcl warm-up` builds them first).
pub fn used_earlier() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(used::path()).unwrap_or_default())
}

/// The pipelines a draw of an earlier run was skipped for (`backstop`; the warm-up builds them
/// before the others an earlier run bound).
pub fn skipped_earlier() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(used::skipped_path()).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashMap<u64, usize>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    pub fn path() -> std::path::PathBuf {
        crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        keys(text).collect()
    }

    fn keys(text: &str) -> impl Iterator<Item = u64> + '_ {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0)
    }

    pub(super) fn first_use_order(text: &str) -> HashMap<u64, usize> {
        let mut order = HashMap::new();
        for (rank, key) in keys(text).enumerate() {
            order.entry(key).or_insert(rank);
        }
        order
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashMap::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = path();
            let earlier = first_use_order(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.keys().copied().collect(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        rank(key).is_some()
    }

    pub fn rank(key: u64) -> Option<usize> {
        if key == 0 { None } else { get().earlier.read().unwrap().get(&key).copied() }
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// `skipped-pipelines.txt` beside `used-pipelines.txt`: the pipelines a draw was skipped for
    /// (`backstop`), which the next warm-up builds first.
    pub fn skipped_path() -> std::path::PathBuf {
        crate::gpu::recomp::dxc::cache_dir().join("skipped-pipelines.txt")
    }

    /// A draw was skipped for want of a pipeline of these shaders: kept for the next warm-up
    /// (once a run; a test writes nothing).
    pub fn note_skipped(key: u64) {
        static NOTED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
        if cfg!(test) || !NOTED.lock().unwrap().get_or_insert_with(HashSet::new).insert(key) {
            return;
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(skipped_path()) {
            let _ = writeln!(f, "{key:016x}");
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key, 0);
    }
}

/// The workers that build every pipeline the title makes ahead of its first use
/// (`XCL_PSO_PRELOAD=0`: off), in the order of `pso_sched::Backlog`. Apart from the ones a bind
/// sends to (`queue`) and below normal priority: a pipeline that is bound is not held up by the ones
/// made ahead (whoever takes `job` first builds it), and the title's own threads are not either.
/// Their big driver compiles are governed (`real::DRIVER_COMPILES`), and none of them waits there:
/// one that does not fit the memory now is put off (`put_off`) and taken up again when it fits.
/// (Two workers that waited in the governor – one 278 s for a 1.9 MB program, one 324 s for a 0.6 MB
/// one, with 4 GB available and 6 GB asked for – built 256 of a retail game's 87 000 pipelines in
/// 14 minutes, and every bind compiled its own.)
struct Preload {
    list: Mutex<super::pso_sched::Backlog<Weak<Pending>>>,
    cv: Condvar,
    /// what the title's threads change in `list`, applied by the next worker that takes its lock
    /// (`change`)
    inbox: Mutex<Vec<Change>>,
}

/// A change to the preload backlog from one of the title's threads.
enum Change {
    Push { n: u64, shader_bytes: usize, known: bool, compute: bool, group: u64, slow_ms: Option<u64>, first_use: Option<u64>, item: Weak<Pending> },
    Promote(u64),
    DepthOnly(u64),
    RtPass(u64),
}

/// the numbers pipelines get in the backlog (`pso_sched::Backlog::push_as`)
static PRELOAD_SEQ: AtomicU64 = AtomicU64::new(1);

/// The title's threads never take the backlog's lock (`XCL_BACKLOG_INBOX=0`: they do): a worker
/// holds it through `Backlog::pop`, which asks every parked pipeline whether it fits. play-95's
/// three recording threads waited in SetPipelineState behind one worker's pop over 3,059 parked
/// pipelines (and five workers queued behind it) until the title's 30 s watchdog ended the run.
fn backlog_inbox() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_BACKLOG_INBOX").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: the title's threads change the preload backlog under its lock (XCL_BACKLOG_INBOX=0)");
        }
        on
    })
}

impl Preload {
    fn new(list: super::pso_sched::Backlog<Weak<Pending>>) -> Self {
        Preload { list: Mutex::new(list), cv: Condvar::new(), inbox: Mutex::new(Vec::new()) }
    }

    /// A title thread's change to the backlog: put in `inbox`, never waiting for a worker's pop.
    fn change(&self, change: Change) {
        if backlog_inbox() {
            self.inbox.lock().unwrap_or_else(|e| e.into_inner()).push(change);
        } else {
            apply(&mut self.list.lock().unwrap(), change);
        }
        self.cv.notify_one();
    }

    /// The backlog's lock, with the title's changes applied.
    fn lock(&self) -> std::sync::MutexGuard<'_, super::pso_sched::Backlog<Weak<Pending>>> {
        let mut list = self.list.lock().unwrap_or_else(|e| e.into_inner());
        let changes = std::mem::take(&mut *self.inbox.lock().unwrap_or_else(|e| e.into_inner()));
        for change in changes {
            apply(&mut list, change);
        }
        list
    }
}

fn apply(list: &mut super::pso_sched::Backlog<Weak<Pending>>, change: Change) {
    match change {
        Change::Push { n, shader_bytes, known, compute, group, slow_ms, first_use, item } => list.push_as(n, shader_bytes, known, compute, group, slow_ms, first_use, item),
        Change::Promote(n) => list.promote(n),
        Change::DepthOnly(n) => list.depth_only(n),
        Change::RtPass(n) => list.rt_pass(n),
    }
}

/// Every pipeline made goes in the preload order of before (`XCL_PSO_BUILD_ALL=1`), instead of
/// what an earlier run drew with first (`pso_sched::Backlog::drawn_first`). Tests keep the order
/// of before: `pso_sched`'s tests cover the other.
fn build_all() -> bool {
    cfg!(test) || std::env::var("XCL_PSO_BUILD_ALL").is_ok_and(|v| v.trim() == "1")
}

/// How many preload workers there are (`XCL_PSO_PRELOAD_THREADS`; `pso_sched::background_workers`
/// of the cores); how many of them build at once is `pace`'s. Under Wine they cannot run below
/// normal priority and stay few.
fn preload_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(n) = std::env::var("XCL_PSO_PRELOAD_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            return n.clamp(1, 64);
        }
        if !can_lower_preload_priority() {
            return (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        }
        super::pso_sched::background_workers(std::thread::available_parallelism().map_or(4, |n| n.get()), physical_cores())
    })
}

/// What a preload worker's build may do about a big driver compile that does not fit the memory
#[derive(Clone, Copy, PartialEq, Eq)]
enum PutOff {
    /// wait for the memory (every other thread)
    No,
    /// leave it (`may_put_off`)
    Allowed,
    /// it left one of so many bytes of DXIL (`put_off`)
    Left(usize),
}

thread_local! {
    static PUT_OFF: Cell<PutOff> = const { Cell::new(PutOff::No) };
}

/// On a preload worker, in a build nothing has to wait for: a big driver compile that does not
/// fit the memory now is not waited for (`real::driver_slot`).
pub fn may_put_off() -> bool {
    PUT_OFF.with(|d| d.get()) != PutOff::No
}

/// The build this worker runs leaves its driver compile of `dxil_len` bytes of DXIL for later: it
/// did not fit the memory. The build fails without an error, and its pipeline waits to be built
/// when the compile fits (`pso_sched::Backlog::park`) or a bind asks for it.
pub fn put_off(dxil_len: usize) {
    PUT_OFF.with(|d| {
        if d.get() != PutOff::No {
            d.set(PutOff::Left(dxil_len));
        }
    });
}

/// The job this thread runs made no pipeline and did not fail: it only prepared the shaders
/// (`preparing`), or its driver compile was put off for memory (`put_off`) and the pipeline is
/// built later.
pub fn left_for_later() -> bool {
    preparing() || matches!(PUT_OFF.with(|d| d.get()), PutOff::Left(_))
}

/// The full-quality build on a preload worker (one runs). Some(DXIL bytes): its big driver
/// compile did not fit the memory and nothing waits for it – not built, to be parked. A bind that
/// came for it meanwhile (`promote_preload` found this worker) gets it built here and now, waiting
/// for memory like every build a bind waits for.
fn build_ahead(p: &Pending) -> Option<usize> {
    if p.orphan.load(Ordering::Acquire) || p.full_started.swap(true, Ordering::AcqRel) {
        return None;
    }
    let job = job_of(p)?;
    PUT_OFF.with(|d| d.set(PutOff::Allowed));
    let mut real = run(p, &*job);
    if let (0, PutOff::Left(dxil_len)) = (real, PUT_OFF.with(|d| d.replace(PutOff::No))) {
        {
            let mut owner = p.preload.lock().unwrap_or_else(|e| e.into_inner());
            if !owner.demanded {
                // (under `preload`: a bind from here on finds no worker and starts its own build)
                owner.worker = None;
                p.full_started.store(false, Ordering::Release);
                return Some(dxil_len);
            }
        }
        real = run(p, &*job);
    }
    complete(p, real, false);
    None
}

/// pipelines the preload workers built, and builds they put off for memory (`build_ahead`)
static AHEAD: AtomicUsize = AtomicUsize::new(0);
static PUT_OFFS: AtomicUsize = AtomicUsize::new(0);

/// (tests) the DXIL lengths there is no memory for, instead of asking the governor
#[cfg(test)]
static NO_ROOM: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// (tests) Whether a driver compile of so many bytes of DXIL would start now.
#[cfg(test)]
fn room_for(dxil_len: usize) -> bool {
    !NO_ROOM.lock().unwrap().contains(&dxil_len)
}

/// the smallest driver compile parked for memory, in bytes of DXIL (0: none parked)
static LEAST_PARKED: AtomicUsize = AtomicUsize::new(0);

/// How often the workers ask whether memory has come for the parked builds: it comes back
/// unannounced (the game's own, another process's).
const PARKED_POLL: std::time::Duration = std::time::Duration::from_millis(200);

#[cfg(not(test))]
fn parked_room() -> Option<super::real::ParkedRoom> {
    let least = LEAST_PARKED.load(Ordering::Acquire);
    if least == 0 {
        return None;
    }
    static SAMPLE: Mutex<Option<(std::time::Instant, super::real::ParkedRoom)>> = Mutex::new(None);
    let mut sample = SAMPLE.lock().unwrap();
    if let Some((at, room)) = *sample {
        if at.elapsed() < PARKED_POLL {
            return Some(room);
        }
    }
    let room = super::real::background_room(least);
    *sample = Some((std::time::Instant::now(), room));
    Some(room)
}

thread_local! {
    static ASSISTING_BIND: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn assisting_bind() -> bool {
    ASSISTING_BIND.with(Cell::get)
}

/// Cores of the PC (its hardware threads when the system does not tell them apart).
fn physical_cores() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GetLogicalProcessorInformation, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION};
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut bytes = 0u32;
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut bytes) };
    let each = std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    let mut info: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = vec![unsafe { std::mem::zeroed() }; bytes as usize / each];
    let mut bytes = (info.len() * each) as u32;
    if info.is_empty() || unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut bytes) } == 0 {
        return threads;
    }
    match info[..(bytes as usize / each).min(info.len())].iter().filter(|i| i.Relationship == RelationProcessorCore).count() {
        0 => threads,
        cores => cores.min(threads),
    }
}

fn filetime(t: &windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64
}

/// CPU time this process has used, in 100 ns.
fn process_cpu() -> u64 {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let mut t = [windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [made, gone, kernel, user] = &mut t;
    if unsafe { GetProcessTimes(GetCurrentProcess(), made, gone, kernel, user) } == 0 {
        return 0;
    }
    filetime(kernel) + filetime(user)
}

/// CPU time thread `tid` has used, in 100 ns (0: gone).
fn thread_cpu(tid: u32) -> u64 {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    let mut t = [windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [made, gone, kernel, user] = &mut t;
    unsafe {
        let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if h.is_null() {
            return 0;
        }
        let ok = GetThreadTimes(h, made, gone, kernel, user);
        CloseHandle(h);
        if ok == 0 {
            return 0;
        }
    }
    filetime(kernel) + filetime(user)
}

/// How often the pace of the preload workers is set again (`pso_sched::Pace::step`), and how
/// long a worker that may not build now rests before it asks again.
#[cfg(not(test))]
const PACE_STEP: std::time::Duration = std::time::Duration::from_millis(500);
const PACE_REST: std::time::Duration = std::time::Duration::from_millis(100);

/// preload workers that may build at once now (`pace`; all of them until it has started), the
/// ones building, and the CPU the rest of the process used in the last step (thousandths of a
/// thread)
static PACE_WORKERS: AtomicUsize = AtomicUsize::new(usize::MAX);
static BUILDING_AHEAD: AtomicUsize = AtomicUsize::new(0);
static PACE_OTHERS: AtomicU64 = AtomicU64::new(0);
/// the title loads through its storage API (`pso_sched::loading`, set by `pace`; until it has
/// run it does: a title starts by loading)
static TITLE_LOADING: AtomicBool = AtomicBool::new(true);

/// Whether the title loads now, as far as its storage API tells.
pub fn title_loading() -> bool {
    TITLE_LOADING.load(Ordering::Relaxed)
}

/// The pace of the `pool` preload workers, set every `PACE_STEP` (`pso_sched::Pace`): how many
/// build at once – from the CPU the process uses apart from them, and whether the title loads
/// through DirectStorage (`pso_sched::loading`) – and their priority. A worker over the number finishes what it
/// builds and takes nothing new; one a bind waits on keeps normal priority (`BoostToken`), and a
/// bind that finds no worker on its pipeline builds it on a thread of its own (`resolve_wait`),
/// so nothing a bind waits for is held up here. Each step also samples the memory for the
/// floor of the background compiles (`real::background_floor`).
#[cfg(not(test))]
fn pace(pool: usize, workers: mpsc::Receiver<Arc<BoostToken>>) {
    let workers: Vec<Arc<BoostToken>> = workers.into_iter().take(pool).collect();
    let mut pace = super::pso_sched::Pace::new(pool, physical_cores());
    let cpu = |workers: &[Arc<BoostToken>]| (process_cpu(), workers.iter().map(|w| thread_cpu(w.tid)).sum::<u64>());
    let storage = || {
        let (requests, bytes, _) = crate::hle::dstorage::counters();
        (requests, bytes)
    };
    let (mut at, mut before, mut ran) = (std::time::Instant::now(), cpu(&workers), storage());
    super::real::sample_background_headroom();
    loop {
        std::thread::sleep(PACE_STEP);
        let (now, after, read) = (std::time::Instant::now(), cpu(&workers), storage());
        let wall = ((now - at).as_nanos() / 100).max(1);
        let others = ((after.0.saturating_sub(before.0)).saturating_sub(after.1.saturating_sub(before.1)) as u128 * 1000 / wall) as u64;
        let loading = super::pso_sched::loading(read.0.saturating_sub(ran.0), read.1.saturating_sub(ran.1), (now - at).as_millis() as u64);
        TITLE_LOADING.store(loading, Ordering::Relaxed);
        let tempo = pace.step(others, loading);
        let rest = if tempo.low { LOADING_PRIORITY } else { PRELOAD_PRIORITY };
        if REST_PRIORITY.swap(rest, Ordering::AcqRel) != rest {
            for w in &workers {
                w.rest(rest);
            }
        }
        PACE_WORKERS.store(tempo.workers, Ordering::Release);
        PACE_OTHERS.store(others, Ordering::Relaxed);
        super::real::sample_background_headroom();
        (at, before, ran) = (now, after, read);
    }
}

/// For the 30 s line: what the preload workers have built, what waits for them, what waits for
/// memory, and their pace now.
pub fn ahead_summary() -> String {
    let (waiting, parked, peers, (drawn, later)) = Q.get().and_then(|q| q.as_ref()).map_or((0, 0, 0, (0, 0)), |q| {
        let list = q.lock();
        let (waiting, parked) = list.waiting();
        (waiting, parked, list.parked_peers(&recent_groups()), list.drawn_split())
    });
    let pool = if Q.get().is_some_and(|q| q.is_some()) { preload_threads() } else { 0 };
    format!(
        "built ahead {} on {} workers ({} preload jobs waiting, {parked} of them for memory, {drawn} an earlier run drew with, {later} no run drew with (a quarter of the workers); put off {} times; now {} of up to {} preload jobs active{}, beside {:.1} threads of other work, {}{}; {peers} parked in recent root groups, {} driver compiles started on the bind floor)",
        AHEAD.load(Ordering::Relaxed),
        pool,
        waiting,
        PUT_OFFS.load(Ordering::Relaxed),
        BUILDING_AHEAD.load(Ordering::Relaxed),
        PACE_WORKERS.load(Ordering::Relaxed).min(pool),
        if REST_PRIORITY.load(Ordering::Relaxed) == LOADING_PRIORITY { " at the lowest priority" } else { "" },
        PACE_OTHERS.load(Ordering::Relaxed) as f64 / 1000.0,
        super::real::background_memory_summary(),
        if title_loading() { ", the title loads" } else { "" },
        super::real::peer_compiles_started(),
    )
}

static Q: OnceLock<Option<Preload>> = OnceLock::new();

fn preload() -> Option<&'static Preload> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            let list = if build_all() { Default::default() } else { super::pso_sched::Backlog::drawn_first() };
            Some(Preload::new(list))
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = preload_threads();
        crate::xlog_info!(
            "D3D12.X: pipelines are built when the title makes them on {n} background threads ({} cores): as many at once as the title's own threads leave cores for, fewer and at the lowest priority while it loads",
            physical_cores()
        );
        let (tokens, started) = mpsc::channel();
        #[cfg(not(test))]
        {
            // (from the least: a title starts by loading)
            PACE_WORKERS.store(super::pso_sched::least_workers(n), Ordering::Release);
            let _ = std::thread::Builder::new().name("xcl-pace".into()).spawn(move || pace(n, started));
        }
        #[cfg(test)]
        drop(started);
        for k in 0..n {
            let tokens = tokens.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                if can_lower_preload_priority() {
                    unsafe {
                        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                        SetThreadPriority(GetCurrentThread(), REST_PRIORITY.load(Ordering::Acquire));
                    }
                }
                let token = BoostToken::for_current_thread();
                let _ = tokens.send(token.clone());
                drop(tokens);
                loop {
                    let (p, lane) = loop {
                        // (not on `cv`: a pipeline queued must wake a worker that may take it)
                        if BUILDING_AHEAD.load(Ordering::Acquire) >= PACE_WORKERS.load(Ordering::Acquire) {
                            std::thread::sleep(PACE_REST);
                            continue;
                        }
                        #[cfg(not(test))]
                        let room = parked_room();
                        let mut g = q.lock();
                        // (under the backlog's lock, as every take is: never more than the pace allows)
                        if BUILDING_AHEAD.load(Ordering::Acquire) >= PACE_WORKERS.load(Ordering::Acquire) {
                            continue;
                        }
                        // (a job is yielded at its end: none is taken while a live compile waits for a slot)
                        if super::real::background_memory_paused() || super::real::live_compile_waiting() || bound_waiting() {
                            drop(q.cv.wait_timeout(g, PARKED_POLL).unwrap());
                            continue;
                        }
                        let wanted = |p: &Weak<Pending>| p.upgrade().is_some_and(|p| !p.orphan.load(Ordering::Acquire) && !p.full_started.load(Ordering::Acquire) && !*p.done.lock().unwrap());
                        #[cfg(not(test))]
                        let fits = |dxil_len| room.is_some_and(|room| room.fits(dxil_len));
                        #[cfg(not(test))]
                        let peer_fits = |dxil_len| room.is_some_and(|room| room.peer_fits(dxil_len));
                        #[cfg(test)]
                        let (fits, peer_fits) = (room_for, |_| false);
                        let next = g.pop(n, &recent_groups(), wanted, fits, peer_fits);
                        LEAST_PARKED.store(g.least_parked(), Ordering::Release);
                        if let Some(next) = next {
                            BUILDING_AHEAD.fetch_add(1, Ordering::AcqRel);
                            break next;
                        }
                        // (a change put in the inbox while this worker went to wait wakes nobody: it is
                        // seen within PARKED_POLL)
                        if g.has_parked() || backlog_inbox() {
                            drop(q.cv.wait_timeout(g, PARKED_POLL).unwrap());
                        } else {
                            drop(q.cv.wait(g).unwrap());
                        }
                    };
                    struct Taken(&'static Preload, super::pso_sched::Lane);
                    impl Drop for Taken {
                        fn drop(&mut self) {
                            self.0.list.lock().unwrap_or_else(|e| e.into_inner()).done(self.1);
                            BUILDING_AHEAD.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                    let _taken = Taken(q, lane);
                    // (a parked one was counted as taken when it was first taken)
                    let _task = (!matches!(lane, super::pso_sched::Lane::Parked | super::pso_sched::Lane::ParkedPeer)).then(|| crate::progress::PIPELINES.take());
                    let Some(p) = p.upgrade() else { continue };
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    if *p.done.lock().unwrap() || p.full_started.load(Ordering::Acquire) {
                        continue;
                    }
                    {
                        let mut owner = p.preload.lock().unwrap_or_else(|e| e.into_inner());
                        if owner.demanded {
                            // a bind's own build of it is queued or running (`resolve_wait`)
                            continue;
                        }
                        owner.worker = Some(token.clone());
                    }
                    let _owned = PreloadJobGuard(&p);
                    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
                    ASSISTING_BIND.with(|help| help.set(lane == super::pso_sched::Lane::ParkedPeer));
                    let left = build_ahead(&p);
                    ASSISTING_BIND.with(|help| help.set(false));
                    BUILDING.with(|b| *b.borrow_mut() = None);
                    if let Some(dxil_len) = left {
                        let times = PUT_OFFS.fetch_add(1, Ordering::Relaxed) + 1;
                        if times.is_power_of_two() {
                            crate::xlog_info!(
                                "D3D12.X: a background build put off: its driver compile of {dxil_len} B of DXIL does not fit the memory now ({}; #{times}) – built when it fits, or when a bind asks for it",
                                if p.what.is_empty() { "shaders unknown" } else { &p.what }
                            );
                        }
                        park(q, dxil_len, &p);
                        continue;
                    }
                    let n = AHEAD.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                    }
                    note_built_ahead();
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    #[test]
    fn use_history_preserves_first_occurrence() {
        let order = used::first_use_order("c\na\nc\nnot a key\n0\nb\n");
        assert_eq!(order.len(), 3);
        assert!(order[&0xc] < order[&0xa] && order[&0xa] < order[&0xb]);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        let (started, finish) = (Mutex::new(started), Mutex::new(finish));
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = started.lock().unwrap().send(());
                let _ = finish.lock().unwrap().recv();
                0x5252
            }),
            0,
        );
        let bind = std::thread::spawn(move || resolve_wait(standin));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    fn thread_priority_of(tid: u32) -> i32 {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{GetThreadPriority, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
        unsafe {
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
            assert!(!h.is_null(), "OpenThread({tid}) failed");
            let p = GetThreadPriority(h);
            CloseHandle(h);
            p
        }
    }

    /// The pace's change of priority reaches a worker at once, unless a bind waits for what it
    /// builds: that one stays at normal priority until it is built.
    #[test]
    fn a_worker_takes_the_priority_of_the_pace_unless_a_bind_waits_on_it() {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        if !can_lower_preload_priority() {
            return;
        }
        let (tok_tx, tok_rx) = mpsc::channel();
        let (end_tx, end_rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            tok_tx.send(BoostToken::for_current_thread()).unwrap();
            let _ = end_rx.recv();
        });
        let tok: Arc<BoostToken> = tok_rx.recv().unwrap();
        tok.rest(LOADING_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), LOADING_PRIORITY);
        tok.rest(PRELOAD_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), PRELOAD_PRIORITY);
        let waited_on = tok.boost();
        tok.rest(LOADING_PRIORITY);
        assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL, "a bind waits on it: not lowered");
        drop(waited_on);
        assert_eq!(thread_priority_of(tok.tid), REST_PRIORITY.load(Ordering::Acquire), "the workers' priority once its bind has it");
        drop(end_tx);
        t.join().unwrap();
    }

    /// What the pace is set from: the CPU a thread used is in the CPU the process used, and the
    /// PC has no more cores than hardware threads.
    #[test]
    fn the_cpu_of_a_thread_is_part_of_the_cpu_of_the_process() {
        let (tid_tx, tid_rx) = mpsc::channel();
        let (p0, started) = (process_cpu(), std::time::Instant::now());
        let t = std::thread::spawn(move || {
            tid_tx.send(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }).unwrap();
            let mut x = 1u64;
            while started.elapsed() < std::time::Duration::from_millis(300) {
                x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
            }
            // (its times are read while it lives)
            std::thread::sleep(std::time::Duration::from_millis(200));
        });
        let tid = tid_rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(350));
        let (thread, process) = (thread_cpu(tid), process_cpu() - p0);
        t.join().unwrap();
        // (100 ns units; a busy PC may give the thread less than its 300 ms)
        assert!(thread >= 500_000, "the thread's CPU time is read: {thread}");
        assert!(process >= thread * 9 / 10, "process {process}, thread {thread}");
        assert_eq!(thread_cpu(u32::MAX - 3), 0, "no such thread");
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        assert!((1..=threads).contains(&physical_cores()), "{} cores, {threads} threads", physical_cores());
    }

    #[test]
    fn boost_token_refcounts_nested_guards() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_NORMAL};
        let check_os = can_lower_preload_priority();
        let (tok_tx, tok_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            if check_os {
                unsafe { SetThreadPriority(GetCurrentThread(), PRELOAD_PRIORITY) };
            }
            tok_tx.send(BoostToken::for_current_thread()).unwrap();
            let _ = done_rx.recv();
        });
        let tok = tok_rx.recv().unwrap();
        assert!(!tok.is_boosted());
        let g1 = tok.boost();
        assert!(tok.is_boosted());
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL);
        }
        let g2 = tok.boost();
        drop(g1);
        assert!(tok.is_boosted(), "still boosted while the second guard lives");
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), THREAD_PRIORITY_NORMAL);
        }
        drop(g2);
        assert!(!tok.is_boosted(), "unboosted once the last guard drops");
        if check_os {
            assert_eq!(thread_priority_of(tok.tid), PRELOAD_PRIORITY, "restored once the last guard drops");
        }
        let _ = done_tx.send(());
        h.join().unwrap();
    }

    /// A bind that waits for a pipeline a preload worker is building raises that worker to normal
    /// priority until it is done (and starts no second build); a pipeline a bind waited for
    /// before a preload worker took it is skipped by that worker.
    #[test]
    fn resolve_wait_boosts_the_preload_worker_building_it_and_restores_it() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_NORMAL};
        let check_os = can_lower_preload_priority();
        let (standin, key) = (0x7E57_7011usize, 0x5EED_0011u64);
        used::earlier_run_used(key);
        let builds = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let counted = builds.clone();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                let tok = entry(standin).unwrap().preload.lock().unwrap().worker.clone().expect("a preload worker builds it");
                let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
                started_tx.send((tid, tok.clone(), unsafe { GetThreadPriority(GetCurrentThread()) })).unwrap();
                let _ = go_rx.lock().unwrap().recv();
                assert!(tok.is_boosted(), "raised while a bind waits for it");
                0x5678
            }),
            key,
        );
        let (tid, tok, before) = started_rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        assert_eq!(before, if check_os { PRELOAD_PRIORITY } else { THREAD_PRIORITY_NORMAL });
        let waiter = std::thread::spawn(move || resolve_wait(standin));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !tok.is_boosted() || (check_os && thread_priority_of(tid) != THREAD_PRIORITY_NORMAL) {
            assert!(std::time::Instant::now() < deadline, "the preload worker was not raised");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        go_tx.send(()).unwrap();
        assert_eq!(waiter.join().unwrap(), 0x5678);
        assert_eq!(builds.load(Ordering::SeqCst), 1, "the bind did not start a second build");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while tok.is_boosted() || (check_os && thread_priority_of(tid) != PRELOAD_PRIORITY) {
            assert!(std::time::Instant::now() < deadline, "the worker was not lowered again after the build");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// Every lazy pipeline is built when it is made, with no bind – those of shaders an earlier
    /// run used and new ones alike – and a first use is kept for the next run.
    #[test]
    fn a_pipeline_is_built_when_made_and_its_first_use_kept() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| if preparing() { 0 } else { 0x4321 }), known_key);
        submit_lazy(unknown, false, Box::new(|| if preparing() { 0 } else { 0x4322 }), unknown_key);
        let start = std::time::Instant::now();
        while (is_pending(known) || is_pending(unknown)) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known) && !is_pending(unknown), "built by the background workers, with no bind");
        assert!(!used::noted(unknown_key), "built is not used");
        assert_eq!(resolve(known), 0x4321);
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// A draw whose graphics pipeline is not built within the backstop is skipped (the stand-in,
    /// no hang) and draws after it are not held up; the build goes on, and once it is done a bind
    /// takes the real pipeline. A dispatch is never skipped.
    #[test]
    fn a_slow_graphics_pipeline_skips_its_draw_and_a_later_bind_uses_it() {
        BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(150)))));
        let (standin, compute) = (0x7E57_7101usize, 0x7E57_7102usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (g, gc) = (gate.clone(), gate.clone());
        let held = move |g: &(Mutex<bool>, Condvar), real: usize| {
            if preparing() {
                return 0;
            }
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            real
        };
        submit_lazy_described(standin, false, Box::new(move || held(&g, 0x4331)), 0, "VS 1 PS 2".into(), 0, 0);
        submit_lazy_described(compute, true, Box::new(move || held(&gc, 0x4332)), 0, "CS 3".into(), 0, 0);
        let skipped_before = skips();
        let t0 = std::time::Instant::now();
        assert_eq!(resolve_wait(standin), standin, "the draw is skipped: the stand-in");
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "no hang");
        let t1 = std::time::Instant::now();
        assert_eq!(resolve_wait(standin), standin, "skipped again while it is built");
        assert!(t1.elapsed() < std::time::Duration::from_millis(100), "a later draw does not wait again");
        assert!(skips() >= skipped_before + 2);
        let waiter = std::thread::spawn(move || {
            BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(150)))));
            resolve_wait(compute)
        });
        std::thread::sleep(std::time::Duration::from_millis(400));
        assert!(!waiter.is_finished(), "a dispatch waits past the backstop");
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        assert_eq!(waiter.join().unwrap(), 0x4332);
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait(standin), 0x4331, "built: a bind uses it");
        released(standin);
        released(compute);
    }

    /// A COM object whose Release counts (a pipeline `release` may drop).
    fn counted_object(releases: &'static AtomicUsize) -> usize {
        unsafe extern "system" fn query(_: *mut std::ffi::c_void, _: *const windows::core::GUID, _: *mut *mut std::ffi::c_void) -> windows::core::HRESULT {
            windows::Win32::Foundation::E_NOINTERFACE
        }
        unsafe extern "system" fn add_ref(_: *mut std::ffi::c_void) -> u32 {
            1
        }
        unsafe extern "system" fn release(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *const Counted)).1.fetch_add(1, Ordering::Relaxed);
            0
        }
        #[repr(C)]
        struct Vtable(
            unsafe extern "system" fn(*mut std::ffi::c_void, *const windows::core::GUID, *mut *mut std::ffi::c_void) -> windows::core::HRESULT,
            unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
            unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
        );
        #[repr(C)]
        struct Counted(&'static Vtable, &'static AtomicUsize);
        static VTABLE: Vtable = Vtable(query, add_ref, release);
        Box::into_raw(Box::new(Counted(&VTABLE, releases))) as usize
    }

    /// A build refused as one whose driver compile never fits the memory now leaves its pipeline
    /// done without a real one (its binds skip, nothing waits) but keeps its build: it is built
    /// again at a bind once that compile fits (`retry_never_fits`), and only then.
    #[test]
    fn a_pipeline_refused_for_memory_is_built_again_once_the_memory_comes_back() {
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let standin = 0x7E57_7009usize;
        const COST: u64 = 25_411 << 20;
        submit_lazy(standin, true, Box::new(|| {
            if BUILDS.fetch_add(1, Ordering::SeqCst) == 0 {
                super::super::real::refuse_as_never_fitting_for_test(COST);
                0
            } else {
                0x4339
            }
        }), 0);
        assert_eq!(resolve_wait(standin), standin, "refused: the stand-in");
        let p = entry(standin).unwrap();
        assert!(!is_pending(standin), "nothing waits for it");
        assert_eq!(p.never_fit.load(Ordering::Acquire), COST);
        assert!(job_of(&p).is_some(), "its build is kept although a bind took it");
        assert_eq!(build_state(standin), super::super::pass_chain::Build::Building, "a chain waits for it");
        let asked = std::sync::Mutex::new(Vec::new());
        assert_eq!(retry_never_fits(|cost| { asked.lock().unwrap().push(cost); cost != COST }), 0, "the memory has not come back");
        assert!(asked.lock().unwrap().contains(&COST));
        assert!(!is_pending(standin));
        assert_eq!(retry_never_fits(|cost| cost == COST), 1);
        assert!(is_pending(standin), "waiting for a bind again");
        assert_eq!(p.never_fit.load(Ordering::Acquire), 0);
        assert_eq!(resolve_wait(standin), 0x4339, "the bind builds it");
        assert_eq!(BUILDS.load(Ordering::SeqCst), 2);
        assert_eq!(retry_never_fits(|_| true), 0, "built: nothing to retry");
        assert_eq!(released(standin), 0x4339);
    }

    /// A pipeline nobody bound is released after its time (`release_idle`) and built again when
    /// a bind wants it; one that was bound is never released.
    #[test]
    fn a_pipeline_nobody_binds_is_released_and_built_again_at_its_first_bind() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let standin = 0x7E57_7004usize;
        let (first, second) = (counted_object(&RELEASES), counted_object(&RELEASES));
        submit_lazy(standin, false, Box::new(move || if BUILDS.fetch_add(1, Ordering::SeqCst) == 0 { first } else { second }), 0);
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let p = entry(standin).unwrap();
        assert_eq!(p.real.load(Ordering::SeqCst), first, "built when made");
        assert!(!release_idle(&p, std::time::Duration::from_secs(3600)), "not idle long enough");
        assert!(release_idle(&p, std::time::Duration::ZERO));
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1, "its real pipeline released");
        assert!(is_pending(standin), "waiting for a bind again");
        assert!(!p.sent.load(Ordering::Acquire), "not counted as building (`counts`)");
        assert_eq!(resolve_wait(standin), second, "a bind builds it again");
        assert_eq!(BUILDS.load(Ordering::SeqCst), 2);
        assert!(!release_idle(&p, std::time::Duration::ZERO), "bound: kept");
        assert_eq!(released(standin), second);
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    /// A graphics pipeline lists have used gives its real pipeline up to the code budget
    /// (`release_used`) only when the budget takes graphics pipelines, and its next bind builds
    /// it again; a compute one gives it up either way.
    #[test]
    fn a_used_graphics_pipeline_is_taken_away_only_when_the_budget_takes_graphics_ones() {
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let (draw, compute) = (0x7E57_7014usize, 0x7E57_7015usize);
        let build = || -> Arc<JobFn> { Arc::new(|| 0x7E57_9000 + 0x10 * BUILDS.fetch_add(1, Ordering::SeqCst)) };
        // (with no budget, as in the tests, a bound pipeline's build is dropped at any time: it is
        // put back until a call saw it)
        let with_build = |p: &Pending, take: &dyn Fn() -> bool| loop {
            *p.job.lock().unwrap() = Some(build());
            let taken = take();
            if taken || job_of(p).is_some() {
                return taken;
            }
        };
        submit_lazy(draw, false, Box::new(|| 0x7E57_8000), 0);
        submit_lazy(compute, true, Box::new(|| 0x7E57_8010), 0);
        let (p, c) = (entry(draw).unwrap(), entry(compute).unwrap());
        let first = resolve_wait(draw);
        assert_eq!(first, 0x7E57_8000, "built at its bind");
        assert!(!with_build(&p, &|| release_used(&p, first, false)), "graphics pipelines are not taken away: kept");
        assert_eq!(p.real.load(Ordering::SeqCst), first);
        assert!(!with_build(&p, &|| release_used(&p, first + 8, true)), "not the pipeline it has now");
        p.builder.store(true, Ordering::Release);
        assert!(!with_build(&p, &|| release_used(&p, first, true)), "a stage of the acceleration-structure builder stays");
        p.builder.store(false, Ordering::Release);
        p.skipped.store(true, Ordering::Release);
        p.submit_gave_up.store(true, Ordering::Release);
        assert!(with_build(&p, &|| release_used(&p, first, true)));
        // (the idle thread may have looked at it just before it was taken away)
        *p.job.lock().unwrap() = Some(build());
        assert_eq!(p.real.load(Ordering::SeqCst), 0, "taken away: the caller keeps it until its lists are through");
        assert!(!p.skipped.load(Ordering::Acquire) && !p.submit_gave_up.load(Ordering::Acquire), "its next bind waits for it and queues its build as a first bind does");
        // (the idle thread may rebuild it before the next bind gets there; that the bind builds it
        // again is what this proves, and the assertion below is that)
        let second = resolve_wait(draw);
        assert!(second >= 0x7E57_9000 && second != first, "a bind builds it again: {second:#x}");
        assert!(p.rebuild.load(Ordering::Acquire), "logged as a rebuild");
        let real = resolve_wait(compute);
        assert_eq!(real, 0x7E57_8010);
        assert!(with_build(&c, &|| release_used(&c, real, false)), "a compute pipeline is taken away either way");
        let _ = released(draw);
        let _ = released(compute);
    }

    #[test]
    fn a_new_compute_pipeline_is_built_without_a_bind() {
        let compute = 0x7E57_7003;
        submit_lazy(compute, true, Box::new(|| 0x4323), 0);
        let start = std::time::Instant::now();
        while is_pending(compute) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(compute));
        assert_eq!(released(compute), 0x4323);
    }

    #[test]
    fn a_pipeline_put_first_is_built() {
        let pso = 0x7E57_7008;
        assert!(!build_first(0x7E57_7FFF), "not a stand-in");
        submit_lazy(pso, true, Box::new(|| 0x4324), 0);
        build_first(pso);
        let start = std::time::Instant::now();
        while is_pending(pso) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(pso));
        assert!(!build_first(pso), "built: not again");
        assert_eq!(released(pso), 0x4324);
    }

    /// Ray tracing off: a pipeline left out is never waited for – its binds go on at once with the
    /// stand-in, nothing pending holds a fence or a frame back.
    #[test]
    fn a_pipeline_left_out_completes_at_once_and_its_binds_never_wait() {
        let pso = 0x7E57_7005;
        submit_lazy(pso, true, Box::new(|| 0), 0);
        give_up(pso);
        assert!(!is_pending(pso), "done: nothing waits for it");
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait(pso), pso, "the stand-in");
        assert_eq!(resolve_wait(pso), pso);
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
        assert!(!build_first(pso), "never put first");
        assert_eq!(released(pso), 0);
    }

    /// A call put off to the submit is waited for there (`wait_for_submit`): the submitting
    /// thread counts as held for the build (`title_held`) while it waits, and its build went
    /// before the others (`need_by_submit`). Its own wait is told by the pipeline's waiters, not
    /// by `title_held` alone: other tests running beside it hold the title too.
    #[test]
    fn a_call_put_off_to_the_submit_is_waited_for_there_and_holds_the_title() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_D201usize;
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        submit_lazy(standin, true, Box::new(move || {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0x1271
        }), 0);
        SKIP_UNTIL_BUILT.with(|s| s.set(true));
        set_defer_for_test(Some(true));
        assert!(deferrable(standin), "a pending pipeline a bind would skip for");
        need_by_submit(standin);
        assert!(entry(standin).unwrap().urgent.load(Ordering::Acquire), "a submit waits for it: urgent");
        let p = entry(standin).unwrap();
        let waiting = p.clone();
        let opener = std::thread::spawn(move || {
            let waits = || waiting.active_waiters.load(Ordering::Acquire) != 0;
            let t = Instant::now();
            while !waits() && t.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let held = waits() && title_held();
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            held
        });
        let built = wait_for_submit(standin, Instant::now() + Duration::from_secs(20));
        let held = opener.join().unwrap();
        SKIP_UNTIL_BUILT.with(|s| s.set(false));
        set_defer_for_test(None);
        assert!(held, "the submit waited without holding the title");
        assert!(built && !is_pending(standin));
        assert_eq!(resolve(standin), 0x1271);
        assert_eq!(p.active_waiters.load(Ordering::Acquire), 0, "the submit's hold outlived its wait");
        let _ = released(standin);
    }

    /// A submit waits for a put-off call's pipeline no longer than its bound (30 s, shorter
    /// here), then gives up, counts it, and later calls with it are skipped as before instead of
    /// put off again (a cold giant held for memory does not stand every submit still).
    #[test]
    fn a_submit_waits_no_longer_than_its_bound_and_later_calls_are_not_put_off() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_D202usize;
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        submit(standin, true, Box::new(move || {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0x1272
        }));
        SKIP_UNTIL_BUILT.with(|s| s.set(true));
        set_defer_for_test(Some(true));
        set_submit_wait_for_test(Some(Duration::from_millis(300)));
        assert!(deferrable(standin));
        let gave_up = submit_gave_up();
        let t = Instant::now();
        let built = wait_for_submit(standin, submit_deadline());
        let took = t.elapsed();
        let (counted, again, pending) = (submit_gave_up() > gave_up, deferrable(standin), is_pending(standin));
        set_submit_wait_for_test(None);
        SKIP_UNTIL_BUILT.with(|s| s.set(false));
        set_defer_for_test(None);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = released(standin);
        assert!(!built && pending, "the build ended before the bound");
        assert!(took >= Duration::from_millis(280) && took < Duration::from_secs(5), "the submit waited {took:?}");
        assert!(counted, "the call skipped at the submit was not counted");
        assert!(!again, "a later call with it is put off again");
    }

    /// A call put off before a submit gave up on its pipeline (another list of the same frame)
    /// is skipped at its own submit without a wait: the submit does not stand still for that
    /// build once more (play-102: twice 30 s for one outlined DXC build).
    #[test]
    fn a_submit_does_not_wait_again_for_a_pipeline_a_submit_gave_up_on() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_D20Cusize;
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        submit(standin, true, Box::new(move || {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0x1273
        }));
        SKIP_UNTIL_BUILT.with(|s| s.set(true));
        set_defer_for_test(Some(true));
        set_submit_wait_for_test(Some(Duration::from_millis(200)));
        let first = wait_for_submit(standin, submit_deadline());
        let t = Instant::now();
        let second = wait_for_submit(standin, Instant::now() + Duration::from_secs(5));
        let took = t.elapsed();
        let pending = is_pending(standin);
        set_submit_wait_for_test(None);
        SKIP_UNTIL_BUILT.with(|s| s.set(false));
        set_defer_for_test(None);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let built_later = wait_for_submit(standin, Instant::now());
        let _ = released(standin);
        assert!(!first && pending, "the build ended before the first bound");
        assert!(!second && took < Duration::from_millis(100), "the second submit waited {took:?}");
        assert!(built_later, "once built, a submit takes it");
    }

    /// `XCL_DEFER_TO_SUBMIT=0`: a call whose pipeline is still being built is not put off; it is
    /// skipped at once, as before. A pipeline left out (ray tracing off) is never put off either.
    #[test]
    fn with_defer_to_submit_off_a_pending_call_is_skipped_as_before() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_D203usize;
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        submit_lazy(standin, true, Box::new(move || {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0x1273
        }), 0);
        SKIP_UNTIL_BUILT.with(|s| s.set(true));
        set_defer_for_test(Some(false));
        let off = deferrable(standin);
        let t = Instant::now();
        let skipped = matches!(resolve_for_compute_use(standin), Ok(p) if p == standin);
        let quick = t.elapsed() < Duration::from_millis(500);
        set_defer_for_test(Some(true));
        let on = deferrable(standin);
        let left_out = 0x7E57_D204usize;
        submit_lazy(left_out, true, Box::new(|| 0), 0);
        give_up(left_out);
        let left_out_deferrable = deferrable(left_out);
        SKIP_UNTIL_BUILT.with(|s| s.set(false));
        let waiting = deferrable(standin);
        set_defer_for_test(None);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = released(standin);
        let _ = released(left_out);
        assert!(!off, "put off with XCL_DEFER_TO_SUBMIT=0");
        assert!(skipped && quick, "not skipped at once with XCL_DEFER_TO_SUBMIT=0");
        assert!(on, "not put off with XCL_DEFER_TO_SUBMIT=1");
        assert!(!left_out_deferrable, "a pipeline left out was put off");
        assert!(!waiting, "put off where the bind waits for its build");
    }

    /// A pipeline a submit will wait for is built before the workers' queue (`need_by_submit`).
    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        use std::time::{Duration, Instant};
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < Duration::from_secs(60) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let order = order.lock().unwrap().clone();
        for &standin in &standins {
            let _ = released(standin);
        }
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
    }

    /// The acceleration-structure builder runs as a whole or not at all (play-92: some stages
    /// built and ran while others were skipped, and the live ones wrote nodes over title objects).
    #[test]
    fn no_builder_stage_dispatches_while_one_is_not_built() {
        use super::super::raytracing::{gate_closed, note_builder_stage};
        let (a, b, wrapped_a) = (0x7E57_7006, 0x7E57_7007, 0x7E57_7106);
        let open = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let gate = open.clone();
        submit_lazy(a, true, Box::new(|| 0x4326), 0);
        submit_lazy(b, true, Box::new(move || {
            while !gate.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x4327
        }), 0);
        note_builder_stage(a, wrapped_a);
        note_builder_stage(b, b);
        mark_builder(a);
        let wait = |pso: usize| {
            let t = std::time::Instant::now();
            while is_pending(pso) && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        wait(a);
        assert!(ready(a) && !ready(b));
        for pso in [a, wrapped_a, 0x4326, b] {
            assert!(gate_closed(false, pso), "{pso:#x}: a stage is not built, so none dispatches");
        }
        assert!(!gate_closed(false, 0x7E57_7FFE), "not a builder stage");
        open.store(true, Ordering::Release);
        wait(b);
        for pso in [a, wrapped_a, 0x4326, b] {
            assert!(!gate_closed(false, pso), "{pso:#x}: every stage built");
            assert!(gate_closed(true, pso), "{pso:#x}: ray tracing off, never");
        }
        assert_eq!((released(a), released(b)), (0x4326, 0x4327));
    }
}

/// The preload workers' scheduling with the real threads: a build put off for memory, and the
/// binds that come for it. (The order of the backlog and the governor's admission are
/// `pso_sched`'s tests.)
#[cfg(test)]
mod ahead_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_until(what: &str, f: impl Fn() -> bool) {
        let t = Instant::now();
        while !f() {
            assert!(t.elapsed() < Duration::from_secs(10), "{what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn compute_deadline_leaves_the_build_urgent() {
        let (standin, real) = (0x7E57_D002usize, 0x4702usize);
        let (finish, gate) = mpsc::channel();
        let gate = Mutex::new(gate);
        submit_lazy(standin, true, Box::new(move || {
            let _ = gate.lock().unwrap().recv();
            real
        }), 0);
        let count = AtomicUsize::new(0);
        let Err(PendingComputeSkip::Deadline { waited_ms, .. }) = resolve_with_deadline(standin, &count, Some(Duration::from_millis(10))) else {
            panic!("pending compute did not hit its deadline");
        };
        assert!(waited_ms >= 10);
        assert_eq!(count.load(Ordering::Relaxed), 1);
        assert!(entry(standin).unwrap().urgent.load(Ordering::Acquire));
        assert_eq!(entry(standin).unwrap().active_waiters.load(Ordering::Acquire), 0);
        assert!(is_pending(standin));
        finish.send(()).unwrap();
        assert_eq!(resolve_wait(standin), real);
        assert_eq!(released(standin), real);
    }

    #[test]
    fn image_only_compute_skips_pending_binds_and_uses_then_runs() {
        let (standin, real) = (0x7E57_D003usize, 0x4703usize);
        let (finish, gate) = mpsc::channel();
        let gate = Mutex::new(gate);
        submit_lazy(standin, true, Box::new(move || {
            let _ = gate.lock().unwrap().recv();
            real
        }), 0);
        allow_image_skip(standin);
        let (tx, bound) = mpsc::channel();
        let bind = std::thread::spawn(move || {
            let bound = resolve_wait_at_bind(standin);
            let use_skipped = matches!(resolve_for_compute_use(standin), Err(PendingComputeSkip::ImageOnly));
            tx.send((bound, use_skipped)).unwrap();
        });
        let result = bound.recv_timeout(Duration::from_secs(2));
        let pending = entry(standin).unwrap();
        assert!(!pending.urgent.load(Ordering::Acquire), "a skipped pipeline stays non-urgent");
        finish.send(()).unwrap();
        assert_eq!(result.unwrap(), (standin, true));
        bind.join().unwrap();
        wait_until("the background build completes", || !is_pending(standin));
        assert!(matches!(resolve_for_compute_use(standin), Ok(pso) if pso == real));
        assert_eq!(resolve_wait_at_bind(standin), real);
        assert_eq!(released(standin), real);
    }

    #[test]
    fn a_passive_bind_leaves_pending_compute_to_the_first_use() {
        let standin = 0x7E57_D001usize;
        let real = 0x4701usize;
        let (finish, gate) = mpsc::channel();
        let gate = Mutex::new(gate);
        submit_lazy(standin, true, Box::new(move || {
            gate.lock().unwrap().recv().unwrap();
            real
        }), 0);
        let pending = entry(standin).unwrap();
        assert_eq!(resolve_bind(standin, true), standin);
        assert!(!pending.urgent.load(Ordering::Acquire));
        assert!(!pending.bound.load(Ordering::Acquire));

        let used = std::thread::spawn(move || {
            let count = AtomicUsize::new(0);
            (resolve_wait_from(standin, &count), count.load(Ordering::Relaxed))
        });
        wait_until("compute use makes the unfinished build urgent", || pending.urgent.load(Ordering::Acquire));
        assert!(!used.is_finished());
        finish.send(()).unwrap();
        assert_eq!(used.join().unwrap(), (real, 1));
        assert_eq!(resolve_bind(standin, true), real);
        assert_eq!(released(standin), real);
    }

    /// A build with a big driver compile of `dxil_len` bytes, as `real::driver_slot` treats it:
    /// with no memory for it (`NO_ROOM`) a preload worker nothing waits on leaves it; any other
    /// thread would wait for the memory (here: builds it).
    fn big(dxil_len: usize, real: usize, runs: Arc<AtomicUsize>) -> Job {
        Box::new(move || {
            runs.fetch_add(1, Ordering::SeqCst);
            if !room_for(dxil_len) && may_put_off() && !waited_on() {
                put_off(dxil_len);
                return 0;
            }
            real
        })
    }

    fn parked(standin: usize) -> bool {
        let p = entry(standin).unwrap();
        let owner = p.preload.lock().unwrap();
        owner.worker.is_none() && !p.full_started.load(Ordering::Acquire) && !*p.done.lock().unwrap()
    }

    /// A build whose driver compile does not fit the memory is put off: the pipelines made
    /// after it are all built meanwhile (two workers that waited in the governor held a whole
    /// run's builds behind them), it stays waiting to be built – not failed – and is built once it fits.
    #[test]
    fn a_build_put_off_for_memory_holds_nothing_up_and_is_built_when_it_fits() {
        let len = 0x7_0001;
        NO_ROOM.lock().unwrap().push(len);
        let runs = Arc::new(AtomicUsize::new(0));
        let (big_one, small) = (0x7E57_A001usize, 0x7E57_A100usize);
        submit_lazy_described(big_one, true, big(len, 0x4401, runs.clone()), 0, "CS big".into(), super::super::pso_sched::HEAVY_SHADER_BYTES, 0);
        for k in 0..64 {
            submit_lazy(small + k, k % 2 == 0, Box::new(move || 0x4500 + k), 0);
        }
        wait_until("the pipelines behind a build that waits for memory are built", || (0..64).all(|k| !is_pending(small + k)));
        wait_until("the big build was put off", || runs.load(Ordering::SeqCst) >= 1 && parked(big_one));
        std::thread::sleep(Duration::from_millis(500));
        assert!(is_pending(big_one), "not built while it does not fit – and not failed: it still waits to be built");
        assert_eq!(runs.load(Ordering::SeqCst), 1, "and not tried again while it does not fit");
        NO_ROOM.lock().unwrap().retain(|l| *l != len);
        wait_until("built once it fits", || !is_pending(big_one));
        assert_eq!(released(big_one), 0x4401);
        for k in 0..64 {
            assert_eq!(released(small + k), 0x4500 + k);
        }
    }

    /// The groups of the pipelines bound last are kept, the last one first; one bound again and
    /// again is noted once, and a pipeline of no group changes nothing.
    #[test]
    fn the_groups_bound_last_are_kept_the_last_one_first() {
        // (the other tests' pipelines have no group: only this one notes any)
        let (a, b, c) = (0x7E57_0000_0000_00A1u64, 0x7E57_0000_0000_00B2u64, 0x7E57_0000_0000_00C3u64);
        for group in [a, a, 0, a, b, 0, b, a, c, c] {
            note_bound_group(group);
        }
        assert_eq!(recent_groups()[..4], [c, a, b, a]);
        for _ in 0..super::super::pso_sched::RECENT_GROUPS {
            note_bound_group(b);
            note_bound_group(c);
        }
        assert!(recent_groups().iter().all(|g| *g == b || *g == c), "the older ones are gone");
        assert_eq!(recent_groups()[..2], [c, b]);
    }

    /// No more background workers build at once than the pace allows (`PACE_WORKERS`): the
    /// others take nothing until one is done.
    #[test]
    fn the_workers_build_no_more_at_once_than_the_pace_allows() {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                PACE_WORKERS.store(usize::MAX, Ordering::Release);
            }
        }
        let (at_once, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let first = 0x7E57_C100usize;
        let _restore = Restore;
        PACE_WORKERS.store(2, Ordering::Release);
        for k in 0..24 {
            let (at_once, most) = (at_once.clone(), most.clone());
            submit_lazy(
                first + k,
                k % 2 == 0,
                Box::new(move || {
                    most.fetch_max(at_once.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    at_once.fetch_sub(1, Ordering::SeqCst);
                    0x4600 + k
                }),
                0,
            );
        }
        wait_until("all of them are built, two at a time", || (0..24).all(|k| !is_pending(first + k)));
        assert!((1..=2).contains(&most.load(Ordering::SeqCst)), "{} at once", most.load(Ordering::SeqCst));
        for k in 0..24 {
            assert_eq!(released(first + k), 0x4600 + k);
        }
    }

    /// A bind promotes a build that waits for memory: its build starts at once on a thread of
    /// its own, ahead of everything queued, and waits for the memory itself.
    #[test]
    fn a_bind_takes_a_build_put_off_for_memory() {
        let len = 0x7_0002;
        NO_ROOM.lock().unwrap().push(len);
        let runs = Arc::new(AtomicUsize::new(0));
        let standin = 0x7E57_A002usize;
        submit_lazy_described(standin, true, big(len, 0x4402, runs.clone()), 0, "CS big".into(), 0, 0);
        wait_until("the build was put off", || runs.load(Ordering::SeqCst) >= 1 && parked(standin));
        let urgent_before = URGENT_BUILDS.load(Ordering::SeqCst);
        assert_eq!(resolve_wait(standin), 0x4402, "the bind's own build");
        assert!(URGENT_BUILDS.load(Ordering::SeqCst) > urgent_before, "started by the bind, ahead of the queue");
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        NO_ROOM.lock().unwrap().retain(|l| *l != len);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(runs.load(Ordering::SeqCst), 2, "the parked build is dropped, not built again");
        assert_eq!(released(standin), 0x4402);
    }

    /// A bind that comes while a preload worker is putting its build off is not left waiting
    /// for a build nobody runs: the worker builds it itself, now.
    #[test]
    fn a_bind_that_comes_while_its_build_is_put_off_gets_it_built() {
        let len = 0x7_0003;
        let standin = 0x7E57_A003usize;
        let (started_tx, started_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let threads = Arc::new(Mutex::new(Vec::new()));
        let ran_on = threads.clone();
        submit_lazy_described(
            standin,
            true,
            Box::new(move || {
                let first = {
                    let mut t = ran_on.lock().unwrap();
                    t.push(std::thread::current().id());
                    t.len() == 1
                };
                if !first {
                    return 0x4403;
                }
                assert!(may_put_off(), "on a preload worker");
                started_tx.send(()).unwrap();
                let _ = go_rx.lock().unwrap().recv();
                put_off(len);
                0
            }),
            0,
            "CS big".into(),
            0,
            0,
        );
        started_rx.recv_timeout(Duration::from_secs(10)).expect("a preload worker builds it");
        let waiter = std::thread::spawn(move || resolve_wait(standin));
        wait_until("the bind found the worker", || entry(standin).unwrap().preload.lock().unwrap().demanded);
        go_tx.send(()).unwrap();
        assert_eq!(waiter.join().unwrap(), 0x4403);
        let threads = threads.lock().unwrap();
        assert_eq!(threads.len(), 2, "built once more, and only once");
        assert_eq!(threads[0], threads[1], "by the worker that had it");
        assert_eq!(released(standin), 0x4403);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_bind();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            pso
        }
        r => r,
    }
}

/// A bind waits until its pipeline is built, graphics and compute alike: a stand-in bound
/// instead skips the draw or dispatch, and what it would have written stays unwritten (with
/// stand-ins bound for hundreds of pipelines a retail game crashed or hung the GPU while it
/// loaded). Waits this long or longer are logged.
const LONG_WAIT: std::time::Duration = std::time::Duration::from_secs(8);

/// the longest a bind waited so far (ms)
static WORST_WAIT: AtomicUsize = AtomicUsize::new(0);

/// `resolve`, waiting for a pipeline still being built: its build starts now on a thread of its
/// own (`build_tiered`; a queue of a title's hundreds of pipelines must not hold up the one it
/// binds now), and the bind waits until it is built (`LONG_WAIT`). A build that failed leaves
/// the stand-in (its failure is logged where it failed).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_from(pso, &USE_WAITS)
}

fn compute_skip_pending() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(limit) = COMPUTE_SKIP_LIMIT.with(Cell::get) {
        return Some(limit);
    }
    static LIMIT: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let ms = std::env::var("XCL_COMPUTE_SKIP_PENDING_MS").ok()?.trim().parse().ok()?;
        crate::xlog_warn!("D3D12.X: compute pending-skip diagnostic enabled ({ms} ms per call); skipped GPU writes are not replayed");
        Some(std::time::Duration::from_millis(ms))
    })
}

pub(super) fn image_skip_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_warn!("D3D12.X: image-only compute skip enabled; pending dispatches leave their textures unchanged and are not replayed");
        }
        on
    })
}

pub(super) fn allow_image_skip(pso: usize) {
    if let Some(p) = entry(pso).filter(|p| p.compute) {
        p.image_only.store(true, Ordering::Release);
    }
}

/// A graphics pipeline without colour targets (shadow maps, depth prepasses): what its draws
/// leave unwritten is read by every later draw of the frame, so its binds may wait this many
/// frame wait budgets (`frame_budget`) before they are skipped at once.
const DEPTH_ONLY_BUDGETS: u32 = 4;

/// Ray tracing off with every ray a miss (`raytracing::miss_pass`): `pso`'s build goes behind every
/// other one, in the bind queue and in the preload's backlog, so it never takes a worker or a
/// driver slot a pipeline the frame draws with is waiting for.
pub(super) fn rt_pass_last(pso: usize) {
    if let Some(p) = entry(pso) {
        p.rt_pass.store(true, Ordering::Release);
        if let Some(Some(q)) = Q.get() {
            q.change(Change::RtPass(p.preload_seq.load(Ordering::Relaxed)));
        }
    }
}

pub(super) fn allow_depth_wait(pso: usize) {
    if let Some(p) = entry(pso).filter(|p| !p.compute) {
        p.depth_only.store(true, Ordering::Release);
        if let Some(Some(q)) = Q.get() {
            q.change(Change::DepthOnly(p.preload_seq.load(Ordering::Relaxed)));
        }
    }
}

/// Workers that build passes over the screen alone, besides the others (`mark_screen_pass`).
const SCREEN_THREADS: usize = 2;
/// wakes the screen workers (`enqueue_screen`)
static SCREEN_READY: Condvar = Condvar::new();
/// the frame (`present::presents`) the wait for passes over the screen belongs to, and what was
/// waited of it
static SCREEN_WAITED: Mutex<(u64, std::time::Duration)> = Mutex::new((u64::MAX, std::time::Duration::ZERO));
static SCREEN_PASSES: AtomicUsize = AtomicUsize::new(0);
static SCREEN_SKIPS: AtomicUsize = AtomicUsize::new(0);

/// How long binds may wait for passes over the screen in one frame, over all threads
/// (`XCL_SCREEN_WAIT_MS`, 10 000 by default, below the title's GPU watchdog; 0: they are skipped
/// while building as other draws are, and get no workers of their own). The title fills its
/// lighting tables once, at the world's first frame, with such draws; skipped while their
/// pipelines build behind a world's worth of others, a run has no lamps and no diffuse light to
/// its end. Behaviour of CGBJ/cool_fox (codec-d/cool_fox f4b48ad, GPL-3)
/// crates/victory-gpu/src/pipeline/make.rs `screen_budget`, after Belobug 18e57db8.
fn screen_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000u64);
        if ms > 0 {
            crate::xlog_info!("D3D12.X: passes over the screen (no vertex input, no depth, a target): binds wait up to {ms} ms a frame for them, {SCREEN_THREADS} workers of their own (XCL_SCREEN_WAIT_MS)");
        } else {
            crate::xlog_info!("D3D12.X: passes over the screen are skipped while building, as other draws (XCL_SCREEN_WAIT_MS=0)");
        }
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// What is left of this frame's wait for passes over the screen (None: nothing).
fn screen_left() -> Option<std::time::Duration> {
    let budget = screen_budget()?;
    let frame = super::present::presents();
    let mut w = SCREEN_WAITED.lock().unwrap();
    if w.0 != frame {
        *w = (frame, std::time::Duration::ZERO);
    }
    Some(budget.saturating_sub(w.1)).filter(|left| !left.is_zero())
}

fn add_screen_wait(waited: std::time::Duration) {
    let frame = super::present::presents();
    let mut w = SCREEN_WAITED.lock().unwrap();
    if w.0 == frame {
        w.1 += waited;
    }
}

/// `XCL_SCREEN_PREBUILD` (default 1): a pass over the screen is built when the title makes it,
/// ahead of the others. 0: when first bound, as other builds.
fn screen_prebuild() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SCREEN_PREBUILD").map_or(true, |v| v.trim() != "0"))
}

/// `pso` is a pass over the screen (`real::screen_pass`): its binds wait for it within
/// `screen_budget` instead of skipping its draws or putting them off to the submit, and its build
/// goes to the screen workers, now (`screen_prebuild`) or when first bound.
pub(super) fn mark_screen_pass(pso: usize) {
    if screen_budget().is_none() {
        return;
    }
    let Some(p) = entry(pso).filter(|p| !p.compute) else { return };
    if p.screen.swap(true, Ordering::AcqRel) {
        return;
    }
    let n = SCREEN_PASSES.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: pipeline {pso:#x} ({}) is a pass over the screen (#{n})", if p.what.is_empty() { "shaders unknown" } else { &p.what });
    }
    if screen_prebuild() && !*p.done.lock().unwrap() {
        p.sent.store(true, Ordering::Release);
        if !p.promote_preload() {
            enqueue_screen(p);
        }
    }
}

fn enqueue_screen(p: Arc<Pending>) {
    start_workers();
    crate::progress::PIPELINES.queue();
    QUEUE.lock().unwrap().screen.push_back(p);
    SCREEN_READY.notify_all();
    READY.notify_one();
}

pub(super) enum PendingComputeSkip {
    Deadline { shader: u64, waited_ms: usize },
    ImageOnly,
}

pub(super) fn resolve_for_compute_use(pso: usize) -> Result<usize, PendingComputeSkip> {
    resolve_with_deadline(pso, &USE_WAITS, compute_skip_pending())
}

pub(super) fn resolve_wait_at_bind(pso: usize) -> usize {
    resolve_wait_from(pso, &BIND_WAITS)
}

fn resolve_wait_from(pso: usize, site_waits: &AtomicUsize) -> usize {
    match resolve_with_deadline(pso, site_waits, None) {
        Ok(real) => real,
        Err(PendingComputeSkip::ImageOnly) => pso,
        Err(PendingComputeSkip::Deadline { .. }) => unreachable!("no compute skip deadline"),
    }
}

fn resolve_with_deadline(pso: usize, site_waits: &AtomicUsize, compute_limit: Option<std::time::Duration>) -> Result<usize, PendingComputeSkip> {
    let Some(p) = entry(pso) else { return Ok(pso) };
    if p.left_out.load(Ordering::Acquire) {
        if !super::fsr4_standin::member(pso) {
            super::raytracing::note_dispatch();
        }
        return Ok(pso);
    }
    p.note_bind();
    if !*p.done.lock().unwrap() {
        if p.image_only.load(Ordering::Acquire) && p.real.load(Ordering::Acquire) == 0 {
            if !p.skipped.swap(true, Ordering::AcqRel) {
                IMAGE_SKIP_PIPELINES.fetch_add(1, Ordering::Relaxed);
                let seq = p.preload_seq.load(Ordering::Relaxed);
                if seq != 0 && !p.rebuild.load(Ordering::Acquire) {
                    if let Some(Some(q)) = Q.get() {
                        q.change(Change::Promote(seq));
                    }
                } else if !p.sent.swap(true, Ordering::AcqRel) {
                    enqueue(p);
                }
            }
            let waits = if std::ptr::eq(site_waits, &BIND_WAITS) { &IMAGE_SKIP_BINDS } else { &IMAGE_SKIP_USES };
            waits.fetch_add(1, Ordering::Relaxed);
            return Err(PendingComputeSkip::ImageOnly);
        }
        let screen = if p.screen.load(Ordering::Acquire) { screen_left() } else { None };
        if screen.is_none() && skip_until_built() && if p.compute { super::bind::skip_failed_dispatches() } else { super::bind::skip_failed_draws() } {
            start_status();
            p.sent.store(true, Ordering::Release);
            let by_skips = bound_by_skips();
            if by_skips {
                p.skips_waiting.fetch_add(1, Ordering::Relaxed);
            }
            let frame = super::present::presents().wrapping_add(1);
            if p.last_bound.swap(frame, Ordering::AcqRel) != frame {
                let seq = p.preload_seq.load(Ordering::Relaxed);
                if seq != 0 && !p.rebuild.load(Ordering::Acquire) {
                    if let Some(Some(q)) = Q.get() {
                        q.change(Change::Promote(seq));
                    }
                }
                if !p.promote_preload() && !(by_skips && p.in_bound.swap(true, Ordering::AcqRel)) {
                    enqueue_bound(p.clone());
                }
            }
            if p.key != 0 && !p.skipped.swap(true, Ordering::AcqRel) {
                used::note_skipped(p.key);
            }
            return Ok(pso);
        }
        // (the title's thread is held from here until the pipeline is built or the bind gives up)
        let _held = Held::new(&p);
        let _live = crate::live_stall::begin(crate::live_stall::Stage::PipelineWait, p.record, p.key, None, None);
        urge(&p);
        let t0 = std::time::Instant::now();
        let frame_waited = frame_waited();
        let budgets = if p.depth_only.load(Ordering::Acquire) { DEPTH_ONLY_BUDGETS } else { 1 };
        let over_budget = !p.compute && screen.is_none() && frame_budget().is_some_and(|b| frame_waited >= b * budgets);
        let limit = if p.compute {
            compute_limit.or_else(backstop)
        } else if screen.is_some() {
            screen
        } else if p.skipped.load(Ordering::Acquire) || over_budget {
            Some(std::time::Duration::ZERO)
        } else {
            backstop()
        };
        let (mut done, mut warned) = (p.done.lock().unwrap(), false);
        while !*done {
            let Some(left) = limit.map(|l| l.saturating_sub(t0.elapsed())) else {
                done = p.cv.wait(done).unwrap();
                continue;
            };
            if !left.is_zero() {
                done = p.cv.wait_timeout(done, left).unwrap().0;
            } else if !p.compute {
                drop(done);
                if screen.is_some() {
                    add_screen_wait(t0.elapsed());
                    let n = SCREEN_SKIPS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_warn!(
                            "D3D12.X: a pass over the screen skipped: graphics pipeline {pso:#x} ({}) not built after {} ms, this frame's XCL_SCREEN_WAIT_MS is used up (#{n})",
                            if p.what.is_empty() { "shaders unknown" } else { &p.what },
                            t0.elapsed().as_millis()
                        );
                    }
                }
                add_frame_wait(t0.elapsed());
                let ms = t0.elapsed().as_millis() as usize;
                super::pso_manifest::note_blocked(p.record, ms);
                site_waits.fetch_add(1, Ordering::Relaxed);
                WAIT_MS.fetch_add(ms, Ordering::Relaxed);
                SKIPS.fetch_add(1, Ordering::Relaxed);
                if over_budget && !p.skipped.load(Ordering::Acquire) {
                    let n = BUDGET_SKIPS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_warn!(
                            "D3D12.X: a draw skipped at once: graphics pipeline {pso:#x} ({}) not built, and this thread's binds have waited {} ms for pipelines since the last present (XCL_FRAME_STALL_BUDGET_MS) (#{n})",
                            if p.what.is_empty() { "shaders unknown" } else { &p.what },
                            frame_waited.as_millis()
                        );
                    }
                    p.skipped.store(true, Ordering::Release);
                    if p.key != 0 {
                        used::note_skipped(p.key);
                    }
                    return Ok(pso);
                }
                // the first skip of a pipeline is logged (and kept for the next warm-up); later
                // draws with it are skipped without a wait until it is built, and counted
                if !p.skipped.swap(true, Ordering::AcqRel) {
                    if p.key != 0 {
                        used::note_skipped(p.key);
                    }
                    crate::xlog_warn!(
                        "D3D12.X: a draw skipped: graphics pipeline {pso:#x} ({}) not built after {ms} ms of waiting (XCL_BIND_BACKSTOP_MS) – its build goes on ahead of the others; draws with it are skipped until it is built, then use it",
                        if p.what.is_empty() { "shaders unknown" } else { &p.what }
                    );
                }
                return Ok(pso);
            } else if compute_limit.is_some() {
                let waited = t0.elapsed();
                add_frame_wait(waited);
                let waited_ms = waited.as_millis() as usize;
                super::pso_manifest::note_blocked(p.record, waited_ms);
                site_waits.fetch_add(1, Ordering::Relaxed);
                WAIT_MS.fetch_add(waited_ms, Ordering::Relaxed);
                WORST_WAIT.fetch_max(waited_ms, Ordering::Relaxed);
                let shader = p.what.strip_prefix("CS ").and_then(|s| s.split_whitespace().next())
                    .and_then(|s| u64::from_str_radix(s, 16).ok()).unwrap_or_else(|| if p.key == 0 { pso as u64 } else { p.key });
                return Err(PendingComputeSkip::Deadline { shader, waited_ms });
            } else {
                if !std::mem::replace(&mut warned, true) {
                    crate::xlog_warn!(
                        "D3D12.X: a dispatch has waited {} ms for compute pipeline {pso:#x} ({}) – still waiting: a skipped dispatch could leave the title's state wrong",
                        t0.elapsed().as_millis(),
                        if p.what.is_empty() { "shaders unknown" } else { &p.what }
                    );
                }
                done = p.cv.wait(done).unwrap();
            }
        }
        drop(done);
        let waited = t0.elapsed();
        if screen.is_some() {
            add_screen_wait(waited);
        } else {
            add_frame_wait(waited);
        }
        let kind = if p.compute { "compute" } else { "graphics" };
        site_waits.fetch_add(1, Ordering::Relaxed);
        if waited >= LONG_WAIT {
            crate::xlog_warn!("D3D12.X: a bind waited {:.1} s for {kind} pipeline {pso:#x} (binds wait for every pipeline)", waited.as_secs_f32());
        }
        let ms = waited.as_millis() as usize;
        WAIT_MS.fetch_add(ms, Ordering::Relaxed);
        if p.real.load(Ordering::Acquire) != 0 {
            super::pso_manifest::note_blocked(p.record, ms);
        }
        if ms >= 250 {
            crate::xlog_info!(
                "D3D12.X: a bind waited {ms} ms for {kind} pipeline {pso:#x}{}; {}; manifest {:016x}",
                if p.rebuild.load(Ordering::Acquire) { " (a rebuild of one released unbound)" } else { "" },
                p.note.lock().unwrap().as_deref().unwrap_or("no build details"),
                p.record
            );
        }
        if WORST_WAIT.fetch_max(ms, Ordering::Relaxed) < ms && ms >= 250 {
            crate::xlog_info!("D3D12.X: longest bind wait so far: {ms} ms ({kind} pipeline {pso:#x})");
        }
    }
    Ok(match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    })
}

/// binds that waited for a pipeline still being built
static BIND_WAITS: AtomicUsize = AtomicUsize::new(0);
static USE_WAITS: AtomicUsize = AtomicUsize::new(0);
static IMAGE_SKIP_PIPELINES: AtomicUsize = AtomicUsize::new(0);
static IMAGE_SKIP_BINDS: AtomicUsize = AtomicUsize::new(0);
static IMAGE_SKIP_USES: AtomicUsize = AtomicUsize::new(0);

pub(super) fn image_skips() -> (usize, usize, usize) {
    (IMAGE_SKIP_PIPELINES.load(Ordering::Relaxed), IMAGE_SKIP_BINDS.load(Ordering::Relaxed), IMAGE_SKIP_USES.load(Ordering::Relaxed))
}

/// (bind-time waits, use-time waits, the longest wait in ms) so far
pub fn waits() -> (usize, usize, usize) {
    (BIND_WAITS.load(Ordering::Relaxed), USE_WAITS.load(Ordering::Relaxed), WORST_WAIT.load(Ordering::Relaxed))
}

pub(super) fn passive_bind() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| lazy() && std::env::var("XCL_PSO_PASSIVE_BIND").is_ok_and(|v| v.trim() == "1"))
}

/// Resolve without enqueuing or pinning a pipeline for a bookkeeping-only lookup.
pub(super) fn resolve_if_ready(pso: usize) -> usize {
    if !passive_for(pso) {
        return resolve(pso);
    }
    entry(pso).map_or(pso, |p| match p.real.load(Ordering::Acquire) {
        0 => pso,
        real => real,
    })
}

/// Opt-in passive binds keep the stand-in until an actual draw or dispatch asks for the PSO.
pub fn resolve_for_bind(pso: usize) -> usize {
    resolve_bind(pso, passive_for(pso))
}

fn passive_for(pso: usize) -> bool {
    passive_bind() || (compute_skip_pending().is_some() && entry(pso).is_some_and(|p| p.compute))
}

fn resolve_bind(pso: usize, passive: bool) -> usize {
    if !passive {
        return resolve_wait_at_bind(pso);
    }
    let Some(p) = entry(pso) else { return pso };
    if p.real.load(Ordering::Acquire) != 0 {
        p.bound.store(true, Ordering::SeqCst);
        match p.real.load(Ordering::SeqCst) {
            0 => (),
            real => return real,
        }
    }
    let seq = p.preload_seq.load(Ordering::Relaxed);
    if seq != 0 {
        if let Some(Some(q)) = Q.get() {
            q.change(Change::Promote(seq));
        }
    }
    pso
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    // a build still running (or an optimized one to come) releases what it makes
    p.orphan.store(true, Ordering::Release);
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        0
    }
}

/// compute pipelines a bind is waiting for (bound, not built yet)
static COMPUTE_BOUND_PENDING: AtomicUsize = AtomicUsize::new(0);

/// Whether a dispatch is waiting for its compute pipeline now.
pub fn compute_bound_pending() -> bool {
    COMPUTE_BOUND_PENDING.load(Ordering::Acquire) > 0
}

/// (its pipeline key, its shaders) of stand-in `pso` (0 and nothing: not built in the background).
pub fn key_of(pso: usize) -> (u64, String) {
    entry(pso).map_or((0, String::new()), |p| (p.key, p.what.clone()))
}

/// draws and dispatches skipped as their pipeline was still being built
static PENDING_SKIPS: AtomicUsize = AtomicUsize::new(0);

pub(super) fn note_pending_skip() {
    PENDING_SKIPS.fetch_add(1, Ordering::Relaxed);
}

/// draws and dispatches skipped so far as their pipeline was still being built
pub fn pending_skips() -> usize {
    PENDING_SKIPS.load(Ordering::Relaxed)
}

/// The shaders a console pipeline is made of, when known.
pub fn what(pso: usize) -> Option<String> {
    entry(pso).map(|p| p.what.clone()).filter(|w| !w.is_empty())
}

/// `pso` is a stand-in whose real pipeline is still being built, or was built just now.
pub(super) fn pending_or_built(pso: usize) -> bool {
    entry(pso).is_some_and(|p| p.real.load(Ordering::Acquire) != 0 || !*p.done.lock().unwrap())
}

/// Where `pso`'s build is (`pass_chain`): built (or no stand-in), still building (released
/// unbound and not built again yet, or refused for memory until it fits, included), or failed
/// (or left out).
pub(super) fn build_state(pso: usize) -> super::pass_chain::Build {
    use super::pass_chain::Build;
    let Some(p) = entry(pso) else { return Build::Built };
    if p.real.load(Ordering::Acquire) != 0 {
        Build::Built
    } else if !*p.done.lock().unwrap() || p.never_fit.load(Ordering::Acquire) != 0 {
        Build::Building
    } else {
        Build::Failed
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (live real pipelines, released as nobody bound them, built again at a bind) so far
pub fn live() -> (usize, usize, usize) {
    let live = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| p.real.load(Ordering::Acquire) != 0).count());
    (live, IDLE_RELEASED.load(Ordering::Relaxed), REBUILT.load(Ordering::Relaxed))
}

/// (stand-ins alive, those whose build is kept with the description and shaders it owns, those
/// built and bound), for `memdiag`.
pub fn diag_builds() -> (usize, usize, usize) {
    let all: Vec<Arc<Pending>> = PENDING.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.values().cloned().collect());
    let kept = all.iter().filter(|p| job_of(p).is_some()).count();
    let bound = all.iter().filter(|p| p.bound.load(Ordering::SeqCst) && p.real.load(Ordering::Acquire) != 0).count();
    (all.len(), kept, bound)
}

/// (built in the background, failed, still building): lazy pipelines count as building once
/// they were wanted (`resolve`), not while they wait for their first use.
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| p.sent.load(Ordering::Acquire) && !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_bind_queue_builds_the_most_skipped_first_the_newest_of_equals() {
        assert_eq!(super::most_skipped([3u64, 40, 7].into_iter()), Some(1), "drawn every frame before bound once");
        assert_eq!(super::most_skipped([5u64, 5, 2].into_iter()), Some(1), "the newest of equals");
        assert_eq!(super::most_skipped(std::iter::empty::<u64>()), None);
        assert_eq!(super::most_skipped([(false, 900u64), (true, 3), (false, 5)].into_iter()), Some(1), "a ray pass built with every ray a miss after every other");
        assert_eq!(super::most_skipped([(false, 900u64), (false, 5)].into_iter()), Some(0), "of ray passes, the most skipped");
    }

    #[test]
    fn a_put_off_or_prepared_job_is_left_for_later_not_failed() {
        assert!(!super::left_for_later());
        super::PUT_OFF.with(|d| d.set(super::PutOff::Allowed));
        assert!(!super::left_for_later(), "allowed, not put off");
        super::put_off(4096);
        assert!(super::left_for_later(), "put off for memory");
        super::PUT_OFF.with(|d| d.set(super::PutOff::No));
        super::PREPARING.with(|p| p.set(true));
        assert!(super::left_for_later(), "only its shaders prepared");
        super::PREPARING.with(|p| p.set(false));
    }

    use super::*;
    use std::time::{Duration, Instant};

    /// The status line and the overlay read the status thread's latest walk, its oldest wait
    /// aged to now, and never walk the pending pipelines themselves.
    #[test]
    fn status_counts_come_from_the_latest_walk() {
        let walk = Walk { building: 7, wanted: 5, oldest: Some(Duration::from_secs(40)) };
        let ((built, failed, building), (wanted, oldest)) = shown(Some((Instant::now() - Duration::from_secs(2), walk)), 300, 2);
        assert_eq!((built, failed, building, wanted), (300, 2, 7, 5));
        assert!(oldest.is_some_and(|d| d >= Duration::from_secs(42) && d < Duration::from_secs(50)));
        assert_eq!(shown(None, 300, 2), ((300, 2, 0), (0, None)), "before the first walk");
        let pending = PENDING.lock().unwrap();
        let _ = status_counts();
        drop(pending);
    }

    /// Once a thread's binds have waited the frame budget, its draws with pipelines still
    /// being built are skipped at once, while its dispatches still wait; a new frame starts the
    /// budget again.
    #[test]
    fn draws_past_the_frame_wait_budget_are_skipped_at_once_and_dispatches_still_wait() {
        let (first, second, compute) = (0x7E57_7211usize, 0x7E57_7212usize, 0x7E57_7213usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let held = |g: Arc<(Mutex<bool>, Condvar)>, real: usize| -> Job {
            Box::new(move || {
                if preparing() {
                    return 0;
                }
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                real
            })
        };
        submit_lazy(first, false, held(gate.clone(), 0x1251), 0);
        submit_lazy(second, false, held(gate.clone(), 0x1252), 0);
        submit_lazy(compute, true, held(gate.clone(), 0x1253), 0);
        BACKSTOP.with(|b| b.set(Some(Some(Duration::from_millis(300)))));
        FRAME_BUDGET.with(|b| b.set(Some(Some(Duration::from_millis(200)))));
        FRAME_WAIT.with(|w| w.set((u64::MAX, Duration::ZERO)));

        assert_eq!(resolve_wait(first), first, "skipped at the backstop");
        assert!(frame_waited() >= Duration::from_millis(300));
        let (skips, t0) = (budget_skips(), Instant::now());
        assert_eq!(resolve_wait(second), second);
        assert!(t0.elapsed() < Duration::from_millis(150), "skipped at once: {:?}", t0.elapsed());
        assert_eq!(budget_skips(), skips + 1);

        let opener = {
            let g = gate.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(500));
                *g.0.lock().unwrap() = true;
                g.1.notify_all();
            })
        };
        let t0 = Instant::now();
        assert_eq!(resolve_wait(compute), 0x1253, "a dispatch waits past the budget and the backstop");
        assert!(t0.elapsed() >= Duration::from_millis(400));
        opener.join().unwrap();

        FRAME_WAIT.with(|w| w.set((super::super::present::presents().wrapping_add(1), Duration::from_secs(5))));
        assert_eq!(frame_waited(), Duration::ZERO, "a new frame starts the budget again");
        for p in [first, second, compute] {
            released(p);
        }
    }

    /// With `XCL_ASYNC_PSO` neither a draw nor a dispatch waits for a pipeline still being built:
    /// it gets the stand-in at once (its draw is then skipped), its build is sent ahead, and once
    /// built the real pipeline is used.
    #[test]
    fn with_async_pipelines_a_bind_never_waits_and_its_build_goes_first() {
        let (draw, compute) = (0x7E57_7231usize, 0x7E57_7232usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let held = |g: Arc<(Mutex<bool>, Condvar)>, real: usize| -> Job {
            Box::new(move || {
                if preparing() {
                    return 0;
                }
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                real
            })
        };
        submit_lazy(draw, false, held(gate.clone(), 0x1261), 0);
        submit_lazy(compute, true, held(gate.clone(), 0x1262), 0);
        SKIP_UNTIL_BUILT.with(|s| s.set(true));
        let t0 = Instant::now();
        assert_eq!(resolve_wait(draw), draw);
        assert_eq!(resolve_for_bind(draw), draw);
        assert_eq!(resolve_wait(compute), compute, "a dispatch does not wait either");
        assert!(t0.elapsed() < Duration::from_millis(150), "no bind waited: {:?}", t0.elapsed());
        assert_ne!(entry(draw).unwrap().last_bound.load(Ordering::Acquire), 0, "its build was sent to the bind queue");
        assert!(!entry(draw).unwrap().urgent.load(Ordering::Acquire), "a skipped bind waits for nothing: its build is a background one");
        BACKSTOP.with(|b| b.set(Some(Some(Duration::from_millis(300)))));
        let t0 = Instant::now();
        assert_eq!(waiting_if(true, || resolve_wait(draw)), draw, "a bundle's bind waits (to the backstop here)");
        assert!(t0.elapsed() >= Duration::from_millis(250), "it waited: {:?}", t0.elapsed());
        BACKSTOP.with(|b| b.set(None));
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let t0 = Instant::now();
        while resolve(draw) == draw && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(resolve_wait(draw), 0x1261, "once built it is used");
        SKIP_UNTIL_BUILT.with(|s| s.set(false));
        for p in [draw, compute] {
            released(p);
        }
    }

    /// A draw without colour targets still waits for its pipeline once the frame budget is
    /// used up (a skipped shadow map is read by the rest of the frame), up to
    /// `DEPTH_ONLY_BUDGETS` of it; a draw with colour targets is skipped at once as before.
    #[test]
    fn depth_only_draws_wait_past_the_frame_wait_budget() {
        let (colour, depth, late) = (0x7E57_7221usize, 0x7E57_7222usize, 0x7E57_7223usize);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let held = |g: Arc<(Mutex<bool>, Condvar)>, real: usize| -> Job {
            Box::new(move || {
                if preparing() {
                    return 0;
                }
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                real
            })
        };
        submit_lazy(colour, false, held(gate.clone(), 0x1261), 0);
        submit_lazy(depth, false, held(gate.clone(), 0x1262), 0);
        submit_lazy(late, false, held(gate.clone(), 0x1263), 0);
        allow_depth_wait(depth);
        allow_depth_wait(late);
        BACKSTOP.with(|b| b.set(Some(Some(Duration::from_millis(2000)))));
        FRAME_BUDGET.with(|b| b.set(Some(Some(Duration::from_millis(200)))));
        FRAME_WAIT.with(|w| w.set((super::super::present::presents(), Duration::from_millis(250))));

        let (skips, t0) = (budget_skips(), Instant::now());
        assert_eq!(resolve_wait(colour), colour);
        assert!(t0.elapsed() < Duration::from_millis(150), "skipped at once: {:?}", t0.elapsed());
        assert!(budget_skips() > skips);

        FRAME_WAIT.with(|w| w.set((super::super::present::presents(), Duration::from_millis(200 * DEPTH_ONLY_BUDGETS as u64))));
        let t0 = Instant::now();
        assert_eq!(resolve_wait(late), late, "past its own budget a depth-only draw is skipped too");
        assert!(t0.elapsed() < Duration::from_millis(150), "skipped at once: {:?}", t0.elapsed());

        FRAME_WAIT.with(|w| w.set((super::super::present::presents(), Duration::from_millis(250))));
        let opener = {
            let g = gate.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                *g.0.lock().unwrap() = true;
                g.1.notify_all();
            })
        };
        let t0 = Instant::now();
        assert_eq!(resolve_wait(depth), 0x1262, "a depth-only draw waits for its pipeline past the frame budget");
        assert!(t0.elapsed() >= Duration::from_millis(200));
        opener.join().unwrap();
        for p in [colour, depth, late] {
            released(p);
        }
    }

    /// play-95: a preload worker held the backlog's lock through a pop over 3,059 parked
    /// pipelines while the title's recording threads waited for it in SetPipelineState. A bind's
    /// promotion goes in the inbox and returns at once; the next worker to take the lock applies it.
    #[test]
    fn a_bind_s_promotion_completes_while_a_worker_holds_a_long_scan() {
        let pending = |compute| Arc::new(Pending {
            created: Instant::now(),
            last_bound: AtomicU64::new(0),
            job: Mutex::new(None),
            sent: Default::default(),
            preload: Default::default(),
            real: Default::default(),
            done: Default::default(),
            cv: Default::default(),
            compute,
            image_only: Default::default(),
            depth_only: Default::default(),
            screen: Default::default(),
            rt_pass: Default::default(),
            key: 0,
            record: 0,
            used: Default::default(),
            orphan: Default::default(),
            full_started: Default::default(),
            tiered_started: Default::default(),
            real_fast: Default::default(),
            urgent: Default::default(),
            active_waiters: Default::default(),
            counted: Default::default(),
            retired: Default::default(),
            bound: Default::default(),
            built_at: Default::default(),
            released_idle: Default::default(),
            never_fit: Default::default(),
            rebuild: Default::default(),
            note: Default::default(),
            what: String::new(),
            skipped: Default::default(),
                group: 0,
            preload_seq: Default::default(),
            first: Default::default(),
            left_out: Default::default(),
            builder: Default::default(),
            skips_waiting: Default::default(),
            in_bound: Default::default(),
            submit_gave_up: Default::default(),
        });
        let q = Arc::new(Preload::new(Default::default()));
        let ps: Vec<Arc<Pending>> = (0..3).map(|_| pending(true)).collect();
        for (n, p) in (1..).zip(&ps) {
            q.change(Change::Push { n, shader_bytes: 1 << 10, known: false, compute: true, group: 0, slow_ms: None, first_use: None, item: Arc::downgrade(p) });
        }
        let (held, scanning) = std::sync::mpsc::channel();
        let worker = {
            let q = q.clone();
            std::thread::spawn(move || {
                let _scan = q.list.lock().unwrap();
                held.send(()).unwrap();
                std::thread::sleep(Duration::from_secs(2));
            })
        };
        scanning.recv().unwrap();
        let t0 = Instant::now();
        q.change(Change::Promote(1));
        assert!(t0.elapsed() < Duration::from_millis(500), "the bind does not wait for the worker's scan");
        worker.join().unwrap();
        let pop = || q.lock().pop(4, &[], |_| true, |_| false, |_| false).and_then(|(p, _)| p.upgrade());
        assert!(pop().is_some_and(|p| Arc::ptr_eq(&p, &ps[0])), "the next worker applies it: the promoted one first");
        assert!(pop().is_some_and(|p| Arc::ptr_eq(&p, &ps[2])), "then the newest, as before");
    }

    #[test]
    fn build_queue_prefers_graphics_and_keeps_order_within_each_class() {
        let pending = |compute| Arc::new(Pending {
            created: Instant::now(),
            last_bound: AtomicU64::new(0),
            job: Mutex::new(None),
            sent: Default::default(),
            preload: Default::default(),
            real: Default::default(),
            done: Default::default(),
            cv: Default::default(),
            compute,
            image_only: Default::default(),
            depth_only: Default::default(),
            screen: Default::default(),
            rt_pass: Default::default(),
            key: 0,
            record: 0,
            used: Default::default(),
            orphan: Default::default(),
            full_started: Default::default(),
            tiered_started: Default::default(),
            real_fast: Default::default(),
            urgent: Default::default(),
            active_waiters: Default::default(),
            counted: Default::default(),
            retired: Default::default(),
            bound: Default::default(),
            built_at: Default::default(),
            released_idle: Default::default(),
            never_fit: Default::default(),
            rebuild: Default::default(),
            note: Default::default(),
            what: String::new(),
            skipped: Default::default(),
                group: 0,
            preload_seq: Default::default(),
            first: Default::default(),
            left_out: Default::default(),
            builder: Default::default(),
            skips_waiting: Default::default(),
            in_bound: Default::default(),
            submit_gave_up: Default::default(),
        });
        let (c1, c2, g1, g2) = (pending(true), pending(true), pending(false), pending(false));
        let mut queue = BuildQueue::default();
        for p in [&c1, &g1, &c2, &g2] {
            queue.push(Arc::clone(p));
        }
        for expected in [&g1, &g2, &c1, &c2] {
            assert!(Arc::ptr_eq(&queue.pop().unwrap(), expected));
        }
        assert!(queue.pop().is_none());
    }

    /// A lazy pipeline's job runs once to prepare its shaders, before anything binds it, and
    /// once more – for the driver – when it is first bound.
    #[test]
    fn lazy_pipelines_are_prepared_before_their_first_use() {
        // (only without the background workers, `XCL_PSO_PRELOAD=0`: with them it is built when
        // made, `a_pipeline_is_built_when_made_and_its_first_use_kept`)
        if prepare_threads() == 0 || preload().is_some() {
            return;
        }
        let (standin, real) = (0x7300_0010usize, 0x7300_0020usize);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        submit_lazy(
            standin,
            false,
            Box::new(move || {
                seen.lock().unwrap().push(preparing());
                if preparing() { 0 } else { real }
            }),
            0,
        );
        let t0 = Instant::now();
        while calls.lock().unwrap().is_empty() && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*calls.lock().unwrap(), [true], "prepared in the background, the driver left out");
        assert!(is_pending(standin), "preparing does not build the pipeline");
        assert!(!entry(standin).unwrap().sent.load(Ordering::Acquire), "nor counted as building (`counts`)");
        assert_eq!(resolve_wait(standin), real);
        assert_eq!(*calls.lock().unwrap(), [true, false]);
        assert!(!preparing());
        assert_eq!(released(standin), real);
    }

    /// `XCL_BENCH_HLSL=<folder of dumped HLSL, *_{cs,ps,vs}.fssm*.hlsl>`: lazy pipelines whose
    /// builds compile real programs with DXC (into a fresh disk cache), all bound one after
    /// another by one thread right after they were made – a title's loading at its worst.
    /// Run once with `XCL_PREPARE_PSO=0` (each bind compiles its own) and once without.
    #[test]
    #[ignore]
    fn lazy_pipelines_bound_one_after_another() {
        let Some(dir) = std::env::var_os("XCL_BENCH_HLSL") else { return };
        let n: usize = std::env::var("XCL_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let max: u64 = std::env::var("XCL_BENCH_MAX_KB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024) << 10;
        let mut files: Vec<(u64, std::path::PathBuf, &'static str)> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let profile = [("_cs.fssm", "cs_6_6"), ("_ps.fssm", "ps_6_6"), ("_vs.fssm", "vs_6_6")].iter().find(|(s, _)| name.contains(s) && name.ends_with(".hlsl"))?.1;
                let len = e.metadata().ok()?.len();
                (len <= max).then(|| (len, e.path(), profile))
            })
            .collect();
        files.sort();
        let take = n.min(files.len());
        let cache = std::env::temp_dir().join(format!("xcl-lazy-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let mut standins = Vec::new();
        let t0 = Instant::now();
        for k in 0..take {
            let (_, path, profile) = &files[k * files.len() / take];
            let (src, profile, cache) = (std::fs::read_to_string(path).unwrap(), *profile, cache.clone());
            let (standin, real) = (0x7400_0000 + 16 * k, 0x7500_0000 + 16 * k);
            submit_lazy(standin, false, Box::new(move || crate::gpu::recomp::dxc::compile_cached_in(&cache, &src, "main", profile).map_or(0, |_| if preparing() { 0 } else { real })), 0);
            standins.push(standin);
        }
        let built = standins.iter().filter(|&&s| resolve_wait(s) != s).count();
        let secs = t0.elapsed().as_secs_f64();
        let (_, compiles) = crate::gpu::recomp::dxc::compile_counts();
        println!("{take} lazy pipelines ({built} built, {compiles} DXC compiles), {} prepare threads: all bound in {secs:.1} s", prepare_threads());
        for &s in &standins {
            released(s);
        }
        let _ = std::fs::remove_dir_all(&cache);
    }

    /// A build that panics is a failed build: whoever waits for it is woken, the worker lives.
    #[test]
    fn a_panicking_build_fails_instead_of_hanging() {
        let (standin, again) = (0x7300_0110usize, 0x7300_0120usize);
        submit(standin, true, Box::new(|| panic!("synthetic build failure")));
        assert_eq!(resolve_wait(standin), standin);
        submit(again, true, Box::new(|| 0x7300_0130));
        assert_eq!(resolve_wait(again), 0x7300_0130);
        assert_eq!(released(standin), 0);
        assert_eq!(released(again), 0x7300_0130);
    }
}
