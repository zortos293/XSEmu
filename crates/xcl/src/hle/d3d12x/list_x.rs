//! The console command list's own methods.
//!
//! The console's ID3D12GraphicsCommandList keeps the PC methods (some moved:
//! `real::console_remap`) and holds console extensions from #60 on (as observed on Fortnite):
//! CopyMemoryToMemoryX #60, FillMemoryWith{32,64,128}BitValueX #61–#63, depth bounds #68,
//! Write{32,64}BitValueBottomOfPipeX #69 / #70, Wait{32,64}BitValueX #71 / #72,
//! DispatchX #78, DrawIndexedX #81, CopyResourceX #84, SetPredicationBufferX #85,
//! Write{32,64}BitValueTopOfPipeX #88 / #89, Write32BitValueBottomOfShaderX #90,
//! KickoffX #93, RSSetDepthBiasX #94. A title call of any other extension (GDS, hints…)
//! is logged and does nothing (`real::class_for`).
//!
//! Occlusion queries may nest on the console; the PC allows one active per list. When one
//! begins while another is active, the outer one ends there on the PC and resolves as visible;
//! the inner one is a real query (Fortnite nests its per-object queries in a dummy outer one).
//!
//! Console GPU addresses in unified memory are the title's own addresses; memory operations
//! go to that window's buffer (`real::uma_window_buffer`), which is in COMMON between them.
//! An index buffer at such an address is given to the PC as the window buffer's GPU address.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::device_x::{self, DepthBias};
use super::real;

#[path = "native_buffer/mod.rs"]
pub(crate) mod native_buffer;
#[path = "native_buffer/ia_live.rs"]
pub(crate) mod ia_live;
pub(super) use ia_live::tracked_scope;

pub(crate) fn ia_recipe(this: *mut c_void, indexed: bool) -> Vec<native_buffer::recipes::Address> {
    use native_buffer::recipes::{Address, Binding};
    let g = LISTS.lock().unwrap();
    let Some(b) = g.as_ref().and_then(|lists| lists.get(&(this as usize))).map(|s| &s.bindings) else {
        return Vec::new();
    };
    let mut result: Vec<_> = b
        .vertex_titles
        .iter()
        .map(|(&slot, &(original, bytes, stride))| Address {
            original,
            binding: Binding::Vertex { slot, bytes, stride },
        })
        .collect();
    if indexed {
        if let Some((original, bytes, format)) = b.index_title {
            result.push(Address {
                original,
                binding: Binding::Index { bytes, format },
            });
        }
    }
    result
}

const PC_GET_TYPE: usize = 8;
const PC_CLOSE: usize = 9;
const PC_RESET: usize = 10;
const PC_COPY_BUFFER_REGION: usize = 15;
const PC_COPY_TEXTURE_REGION: usize = 16;
const PC_COPY_RESOURCE: usize = 17;
const PC_SET_PIPELINE_STATE: usize = 25;
const PC_RESOURCE_BARRIER: usize = 26;
const PC_IA_SET_INDEX_BUFFER: usize = 43;
const PC_BEGIN_QUERY: usize = 52;
const PC_END_QUERY: usize = 53;
const PC_RESOLVE_QUERY_DATA: usize = 54;
const PC_SET_PREDICATION: usize = 55;
const PC_OM_SET_DEPTH_BOUNDS: usize = 62;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
const PC_QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;

/// Write64BitValueBottomOfPipeX flags 1 / 2, Write64BitValueTopOfPipeX flags 4 / 8: the value
/// written is a GPU / system timestamp
const BOTTOM_TIMESTAMPS: u32 = 0x3;
const TOP_TIMESTAMPS: u32 = 0xC;
/// fills larger than this are not done
const MAX_FILL: u64 = 256 << 20;
const FILL_CHUNK: u64 = 128 << 10;
/// Bounded wait timeout for Wait32BitValueX / Wait64BitValueX so an unsatisfied wait never hangs.
/// Overridable via `XCL_WAIT_TIMEOUT_MS` for diagnostics.
const WAIT_TIMEOUT: Duration = Duration::from_secs(2);
/// Watchdog timeout for GPU `queue.Wait` so a genuinely hung GPU cycle never triggers a 30 s
/// hang, while giving heavy initial 3D frames (>1 s of dispatches) plenty of room to finish
/// cleanly without prematurely breaking synchronization.
const GPU_WAIT_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(5000);

/// `GPU_WAIT_WATCHDOG_TIMEOUT`, or `XCL_GPU_WAIT_RELEASE_MS` (a value above 0). A release on the
/// CPU lets the waiting queue go on before its producer has written the value: when the producer
/// is only late (its list held while a pipeline is built), the consumer reads what was not
/// written yet.
fn gpu_wait_watchdog_timeout() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| gpu_wait_watch().unwrap_or_else(|| release_after(std::env::var("XCL_GPU_WAIT_RELEASE_MS").ok().as_deref())))
}

/// `XCL_GPU_WAIT_WATCH_MS=<ms>` (a run that asks whether the GPU is stuck or slow): a GPU wait is
/// given that long, and from the age at which a run without it gives up
/// (`GPU_WAIT_WATCHDOG_TIMEOUT`) the ops the GPU completes are logged every second
/// (`watch_gpu_waits`). A value no longer than that timeout is none.
fn gpu_wait_watch() -> Option<Duration> {
    static T: OnceLock<Option<Duration>> = OnceLock::new();
    *T.get_or_init(|| watch_for(std::env::var("XCL_GPU_WAIT_WATCH_MS").ok().as_deref()))
}

fn watch_for(ms: Option<&str>) -> Option<Duration> {
    ms.and_then(|v| v.trim().parse::<u64>().ok()).map(Duration::from_millis).filter(|&d| d > GPU_WAIT_WATCHDOG_TIMEOUT)
}

/// `XCL_GPU_WAIT_WATCH_MS`: `oldest` is the age of the oldest GPU wait still armed. Once one is
/// as old as `GPU_WAIT_WATCHDOG_TIMEOUT` the state is logged as at a stop (`markers::snapshot`:
/// the ops the GPU is in and what they were given), then every second the ops and recordings the
/// GPU completed since (`markers::frontier`) – until no wait is that old any more (the GPU was
/// slow, not stuck) or one is `most` old (the stop or release of any run).
fn watch_gpu_waits(oldest: Duration, most: Duration) {
    // (since when a wait is that old, the last line)
    static WATCH: Mutex<Option<(Instant, Instant)>> = Mutex::new(None);
    let mut w = WATCH.lock().unwrap();
    if oldest < GPU_WAIT_WATCHDOG_TIMEOUT {
        if let Some((since, _)) = w.take() {
            crate::xlog_error!(
                "D3D12.X: GPU wait watch: no GPU wait is {} ms old any more, {:.1} s after the first was – the GPU was slow, not stuck; {}",
                GPU_WAIT_WATCHDOG_TIMEOUT.as_millis(),
                since.elapsed().as_secs_f64(),
                super::markers::frontier().join(" | ")
            );
        }
        return;
    }
    let Some((since, last)) = w.as_mut() else {
        *w = Some((Instant::now(), Instant::now()));
        drop(w);
        crate::xlog_error!(
            "D3D12.X: GPU wait watch: a GPU wait is {} ms old – a run without XCL_GPU_WAIT_WATCH_MS gives up here; this one waits up to {} ms and logs every second what the GPU completes",
            oldest.as_millis(),
            most.as_millis()
        );
        super::markers::snapshot("a GPU wait as old as the watchdog timeout, watched on");
        crate::xlog_error!("D3D12.X: GPU wait watch +0 s: {}", super::markers::frontier().join(" | "));
        return;
    };
    if last.elapsed() >= Duration::from_secs(1) {
        *last = Instant::now();
        crate::xlog_error!("D3D12.X: GPU wait watch +{:.0} s (the oldest wait {} ms): {}", since.elapsed().as_secs_f64(), oldest.as_millis(), super::markers::frontier().join(" | "));
    }
}

fn release_after(ms: Option<&str>) -> Duration {
    ms.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map_or(GPU_WAIT_WATCHDOG_TIMEOUT, Duration::from_millis)
}

fn wait_timeout() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("XCL_WAIT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(Duration::from_millis)
            .unwrap_or(WAIT_TIMEOUT)
    })
}

/// `XCL_WAIT_RELEASE=0`: a GPU wait that times out is never released on the CPU – the state is
/// logged (`hard_stop`) and the title stopped, so nothing runs on a dependency the CPU made up.
fn wait_release() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| !std::env::var("XCL_WAIT_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// The first GPU wait that timed out under `XCL_WAIT_RELEASE=0` (`what`): the GPU markers, the
/// title fences still pending and the indirect captures are logged, then DRED once the device
/// goes (it is waited for up to 10 s), and the title is stopped. A second thread that gets here
/// waits for the first.
pub(super) fn hard_stop(what: String) -> ! {
    static STOPPING: AtomicBool = AtomicBool::new(false);
    if STOPPING.swap(true, Ordering::AcqRel) {
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }
    crate::xlog_error!("D3D12.X: {what} – not released on the CPU (XCL_WAIT_RELEASE=0): the title is stopped after this snapshot");
    super::markers::snapshot("the first GPU wait that timed out");
    crate::xlog_error!("D3D12.X: pending title ID3D12Fences: {}", super::fence_x::pending_summary());
    let removed = || unsafe {
        let raw = super::real::device_raw();
        ID3D12Device::from_raw_borrowed(&raw).is_some_and(|d| d.GetDeviceRemovedReason().is_err())
    };
    let t0 = Instant::now();
    while !removed() && t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(100));
    }
    if removed() {
        super::real::log_device_removed();
    } else {
        crate::xlog_error!("D3D12.X: the device was not removed within 10 s of the snapshot");
    }
    crate::report::write_final(Some(format!("a GPU wait timed out and XCL_WAIT_RELEASE=0 stopped the title: {what}")));
    crate::log::flush();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), 0xC0DE_5741);
    }
    unreachable!("TerminateProcess returned")
}

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// GPU waits released on the CPU in the run, at every site (`cpu_release`)
static CPU_RELEASES: AtomicU64 = AtomicU64::new(0);

/// A GPU wait was released on the CPU at the site counting with `site`: Some(the site's count)
/// when it is to be logged (a WARN): the first 16 at each site, then each power of two.
fn cpu_release(site: &AtomicU64) -> Option<u64> {
    CPU_RELEASES.fetch_add(1, Ordering::Relaxed);
    let n = site.fetch_add(1, Ordering::Relaxed) + 1;
    (n <= 16 || n.is_power_of_two()).then_some(n)
}

/// GPU waits released on the CPU so far
pub(super) fn cpu_releases() -> u64 {
    CPU_RELEASES.load(Ordering::Relaxed)
}

/// `XCL_TRACE_QUEUES`: the queues (XCL's q numbers, comma-separated, or "all") whose submissions,
/// GPU waits and GPU signals are logged, with the GPU waits other queues arm on them.
fn traced_queue(q: usize) -> bool {
    static SET: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    let set = SET.get_or_init(|| {
        let v = std::env::var("XCL_TRACE_QUEUES").ok()?;
        Some(if v.trim() == "all" { Vec::new() } else { v.split(',').filter_map(|x| x.trim().trim_start_matches('q').parse().ok()).collect() })
    });
    set.as_ref().is_some_and(|s| s.is_empty() || s.contains(&queue_id(q))) || super::bind::indirect_world_trace()
}

fn trace_queue(line: impl FnOnce() -> String) {
    static N: AtomicU64 = AtomicU64::new(0);
    static WORLD: AtomicU64 = AtomicU64::new(0);
    let n = if super::bind::indirect_world_trace() { &WORLD } else { &N };
    if once(n, 50_000) {
        crate::xlog_info!("queue trace: {}", line());
    }
}

/// a PC method of the list (its runtime's own)
unsafe fn pc<F: Copy>(list: *mut c_void, slot: usize) -> F {
    transmute_copy(&real::orig_fn(list, slot))
}

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    pc::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, PC_GET_TYPE)(list)
}

pub(super) fn tracked_list_type(list: *mut c_void) -> Option<D3D12_COMMAND_LIST_TYPE> {
    LISTS.lock().unwrap().as_ref()?.get(&(list as usize))?.list_type
}

pub(super) unsafe fn transition(list: *mut c_void, res: usize, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    super::residency::note_list_resources(list as usize, [res]);
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: transmute_copy(&res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    super::gpu_timing::class(list, super::gpu_timing::Kind::Barrier, || {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, PC_RESOURCE_BARRIER)(list, 1, &b);
    });
    super::depth_state::legacy(list as usize, 1, &b);
}

/// `op` with the window buffer `res` as a copy destination (back to COMMON after it)
unsafe fn as_copy_dest(list: *mut c_void, res: usize, op: impl FnOnce()) {
    transition(list, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    op();
    transition(list, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn copy_buffer_region(list: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    super::gpu_timing::class(list, super::gpu_timing::Kind::Transfer, || {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(list, PC_COPY_BUFFER_REGION)(list, dst, dst_off, src, src_off, n);
    });
}

fn warn(counter: &AtomicU64, what: &str, why: std::fmt::Arguments) {
    if once(counter, 4) {
        crate::xlog_warn!("D3D12.X: {what}: {why} – not done");
    }
}

// ---------------------------------------------------------------------------------------
// per-list state & live queue tracking (KickoffX, Wait32/64BitValueX, RSSetDepthBiasX,
// index buffer restoration, nested occlusion queries)
// ---------------------------------------------------------------------------------------

/// a query: (heap, index)
type Query = (usize, u32);

/// One list's occlusion queries: the one active on the PC (with its type) and the outer ones
/// it ended early.
#[derive(Default)]
struct Occlusion {
    active: Option<(Query, D3D12_QUERY_TYPE)>,
    ended_early: Vec<Query>,
}

impl Occlusion {
    /// Begin `q` on the PC; Some(outer): end that one on the PC first.
    fn begin(&mut self, q: Query, ty: D3D12_QUERY_TYPE) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let outer = self.active.replace((q, ty));
        if let Some((o, _)) = outer {
            self.ended_early.push(o);
        }
        outer
    }

    /// End the currently active PC query early (e.g. before a `KickoffX` split), so `EndQuery`
    /// later resolves it as visible.
    fn end_active_early(&mut self) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let active = self.active.take()?;
        self.ended_early.push(active.0);
        Some(active)
    }

    /// Whether the PC ends `q` (not: an outer one already ended – it resolves as visible).
    fn end(&mut self, q: Query) -> bool {
        if self.active.is_some_and(|(a, _)| a == q) {
            self.active = None;
            return true;
        }
        match self.ended_early.iter().position(|e| *e == q) {
            Some(i) => {
                self.ended_early.swap_remove(i);
                false
            }
            None => true,
        }
    }
}

/// Recorded GPU-side memory wait from `Wait32BitValueX` (#71) or `Wait64BitValueX` (#72).
/// Evaluates `(mem_value & mask) <comparison> value` using `D3D12_COMPARISON_FUNC` (1..=8),
/// matching AMD PM4 `WAIT_REG_MEM` semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingWait {
    va: u64,
    width: u8, // 4 or 8
    comparison: u32,
    value: u64,
    mask: u64,
    /// True if this list already recorded a `Write64BitValue*` before this wait (potential
    /// intra-list fork-join where hoisting a GPU wait before list submission could deadlock).
    after_write64: bool,
}

fn compare_u64(op: u32, lhs: u64, rhs: u64) -> Option<bool> {
    match op {
        1 => Some(false),        // D3D12_COMPARISON_FUNC_NEVER
        2 => Some(lhs < rhs),    // D3D12_COMPARISON_FUNC_LESS
        3 => Some(lhs == rhs),   // D3D12_COMPARISON_FUNC_EQUAL
        4 => Some(lhs <= rhs),   // D3D12_COMPARISON_FUNC_LESS_EQUAL
        5 => Some(lhs > rhs),    // D3D12_COMPARISON_FUNC_GREATER
        6 => Some(lhs != rhs),   // D3D12_COMPARISON_FUNC_NOT_EQUAL
        7 => Some(lhs >= rhs),   // D3D12_COMPARISON_FUNC_GREATER_EQUAL
        8 => Some(true),         // D3D12_COMPARISON_FUNC_ALWAYS
        _ => None,
    }
}

impl PendingWait {
    unsafe fn read_value(&self) -> Option<u64> {
        // (as a copy, `memmap::title_words`: the title can unmap the page between a look at
        // it and the load)
        let addr = self.va as usize;
        match self.width {
            4 => Some(crate::memmap::title_words(addr, 1)?[0] as u64),
            8 => crate::memmap::title_words(addr, 2).map(|w| w[0] as u64 | (w[1] as u64) << 32),
            _ => None,
        }
    }

    unsafe fn is_satisfied(&self) -> bool {
        let Some(cur) = self.read_value() else {
            return true;
        };
        compare_u64(self.comparison, cur & self.mask, self.value).unwrap_or(true)
    }

    unsafe fn wait_until_satisfied(&self, timeout: Duration) -> bool {
        if self.is_satisfied() {
            return true;
        }
        for _ in 0..512 {
            std::hint::spin_loop();
            if self.is_satisfied() {
                return true;
            }
        }
        let start = Instant::now();
        while start.elapsed() < timeout {
            let step = if start.elapsed() < Duration::from_millis(10) {
                Duration::from_micros(100)
            } else {
                Duration::from_millis(1)
            };
            std::thread::sleep(step);
            if self.is_satisfied() {
                return true;
            }
        }
        self.is_satisfied()
    }
}

/// Recorded `OMSetRenderTargets` state for `KickoffX` restoration and `XCL_TRACE_FRAME`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderTargets {
    /// Requested `NumRenderTargetDescriptors` (clamped to D3D12 max 8; 0 if `handles` is empty).
    num_targets: u32,
    /// Saved descriptor handles: 1 base handle when `single != 0` and `num_targets > 0`,
    /// otherwise `num_targets` handles.
    handles: Vec<usize>,
    single: i32,
    dsv: Option<usize>,
}

/// What console code set on a list that a PC Reset clears: `KickoffX` closes, submits and
/// resets the PC list mid-recording, and on the console a kickoff leaves the list's state as it
/// was – so it is set again (descriptor heaps above all: translated shaders index the bound
/// heap, and a draw without one reads garbage descriptors).
#[derive(Default, Clone)]
struct Bindings {
    depth_capture_native_pso: usize,
    depth_capture_dsv_generation: Option<u64>,
    lineage_om_valid: bool,
    lineage_pass_dsv: Option<Option<(usize, u64)>>,
    heaps: Vec<usize>,
    topology: Option<u32>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<windows::Win32::Foundation::RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    vertex_buffers: std::collections::BTreeMap<u32, D3D12_VERTEX_BUFFER_VIEW>,
    render_targets: Option<RenderTargets>,
    /// slot → the title's (address, bytes, stride) of the vertex buffer (`XCL_TRACE_FRAME`)
    vertex_titles: std::collections::BTreeMap<u32, (u64, u32, u32)>,
    /// the title's (address, bytes, format) of the index buffer (`XCL_TRACE_FRAME`)
    index_title: Option<(u64, u32, i32)>,
    /// the title set the console's rect list topology
    rect_list: bool,
    predication: Option<(usize, u64, D3D12_PREDICATION_OP)>,
    /// a draw put off to the submit: the heaps its render targets were copied into
    /// (`snapshot_render_targets`)
    render_target_copies: Vec<ID3D12DescriptorHeap>,
}

/// Whether `list` draws rect lists now, and the title's vertex buffers (slot, (address, bytes, stride)).
pub(super) fn rect_list_state(list: *mut c_void) -> Option<Vec<(u32, (u64, u32, u32), Option<D3D12_VERTEX_BUFFER_VIEW>)>> {
    let g = LISTS.lock().unwrap();
    let b = &g.as_ref()?.get(&(list as usize))?.bindings;
    b.rect_list.then(|| b.vertex_titles.iter().map(|(&k, &v)| (k, v, b.vertex_buffers.get(&k).copied())).collect())
}

/// `pc_va` for other modules.
pub(super) fn pc_va_of(va: u64) -> u64 {
    pc_va(va)
}

/// `XCL_TRACE_FRAME`: the fixed-function state a list last set (topology, first viewport and
/// scissor, render targets) as text.
pub(super) fn state_summary(list: *mut c_void) -> String {
    let mut out = String::new();
    with_bindings(list, |b| {
        let vp = b.viewports.first().map(|v| (v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth));
        let sc = b.scissors.first().map(|r| (r.left, r.top, r.right, r.bottom));
        out = format!(
            "topology {:?}, stencil ref {:?}, viewport {vp:?} ({} set), scissor {sc:?} ({} set), render targets {:?}{}",
            b.topology,
            b.stencil,
            b.viewports.len(),
            b.scissors.len(),
            b.render_targets.as_ref().map(|rt| (rt.num_targets as usize, rt.dsv.is_some())),
            b.render_targets.as_ref().and_then(|rt| rt.dsv).map_or(String::new(), |d| format!(", DSV {d:#x}"))
        );
    });
    out
}

/// `XCL_TRACE_FRAME`: the vertex and index buffers a list binds, with their first vertices
/// and indices as the CPU sees the title's memory.
pub(super) unsafe fn geometry_summary(list: *mut c_void, draw: (bool, u32, u32, u32, i32)) -> String {
    let (vbs, ib) = {
        let mut out = (Vec::new(), None);
        with_bindings(list, |b| out = (b.vertex_titles.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(), b.index_title));
        out
    };
    let words = |a: u64, n: usize| -> Option<Vec<u32>> {
        (a != 0).then(|| crate::memmap::title_words(a as usize, n)).flatten()
    };
    let mut out = Vec::new();
    for (slot, (a, size, stride)) in vbs {
        let per = (stride as usize / 4).clamp(1, 12);
        let verts: Vec<String> = (0..4u64).filter_map(|i| words(a + i * stride as u64, per).map(|w| format!("{w:x?}"))).collect();
        out.push(format!("VB{slot} {a:#x} ({size} bytes, stride {stride}): {}", verts.join(" ")));
    }
    // the vertices the draw names: through its indices (indexed) or from its first vertex
    let (indexed, count, _, first, base) = draw;
    let n = count.min(6) as usize;
    let names: Option<Vec<u64>> = if indexed {
        ib.and_then(|(a, _, fmt)| {
            let w = if fmt == 57 { 2 } else { 4 };
            let at = a as usize + w * first as usize;
            (a != 0).then(|| crate::memmap::title_bytes(at, w * n)).flatten().map(|b| {
                b.chunks_exact(w).map(|c| if w == 2 { u16::from_le_bytes([c[0], c[1]]) as u64 } else { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            })
        })
    } else {
        Some((0..n as u64).map(|k| first as u64 + k).collect())
    };
    out.push(format!("IB {:x?}; the draw's vertices (as indexed now) {names:?}", ib.map(|(a, size, fmt)| (a, size, fmt))));
    // the title fills ring buffers after recording a draw: the vertices are read again when
    // the list is executed (`dump_pending_geometry`)
    if let Some((_, vb)) = vbs_first(list) {
        PENDING_GEOMETRY.lock().unwrap().push((list as usize, vb, ib, draw));
    }
    out.join("; ")
}

/// draws whose vertices are dumped when their list runs: (list, (VB address, bytes, stride),
/// (IB address, bytes, format), draw arguments)
#[allow(clippy::type_complexity)]
static PENDING_GEOMETRY: Mutex<Vec<(usize, (u64, u32, u32), Option<(u64, u32, i32)>, (bool, u32, u32, u32, i32))>> = Mutex::new(Vec::new());

/// `XCL_TRACE_FRAME`: the vertices of traced draws as the title's memory holds them at the
/// next present (the CPU has written the frame's vertices by then).
pub(super) unsafe fn dump_pending_geometry() {
    let mine: Vec<_> = std::mem::take(&mut *PENDING_GEOMETRY.lock().unwrap());
    for (_, (a, _, stride), ib, (indexed, count, _, first, base)) in mine {
        let n = count.min(48) as usize;
        let names: Vec<u64> = match (indexed, ib) {
            (true, Some((ia, _, fmt))) => {
                let w = if fmt == 57 { 2 } else { 4 };
                let at = ia as usize + w * first as usize;
                let Some(b) = crate::memmap::title_bytes(at, w * n) else { continue };
                b.chunks_exact(w).map(|c| if w == 2 { u16::from_le_bytes([c[0], c[1]]) as u64 } else { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            }
            _ => (0..n as u64).map(|k| first as u64 + k).collect(),
        };
        let per = (stride as usize / 4).clamp(1, 12);
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        let verts: Vec<String> = unique
            .iter()
            .take(24)
            .map(|&i| {
                let at = (a + i * stride as u64) as usize;
                let w: Option<Vec<u32>> = crate::memmap::title_words(at, per);
                format!("v{i}: {w:x?}")
            })
            .collect();
        crate::xlog_info!("frame trace: … its vertices at the present (VB {a:#x}, stride {stride}, indices {names:?}): {}", verts.join("; "));
    }
}

fn vbs_first(list: *mut c_void) -> Option<(u32, (u64, u32, u32))> {
    let mut first = None;
    with_bindings(list, |b| first = b.vertex_titles.iter().next().map(|(&k, &v)| (k, v)));
    first
}

/// The depth-stencil view handle a list last bound (`XCL_TRACE_FRAME`).
pub(super) fn bound_dsv(list: *mut c_void) -> Option<usize> {
    let mut d = None;
    with_bindings(list, |b| d = b.render_targets.as_ref().and_then(|rt| rt.dsv));
    d
}

/// depth-stencil views bound by draws that read a watched target: their clears are traced
static WATCHED_DSVS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(super) fn watch_dsv(dsv: usize) {
    let mut g = WATCHED_DSVS.lock().unwrap();
    if !g.contains(&dsv) && g.len() < 8 {
        g.push(dsv);
    }
}

fn with_bindings(list: *mut c_void, f: impl FnOnce(&mut Bindings)) {
    let mut g = LISTS.lock().unwrap();
    f(&mut g.get_or_insert_with(HashMap::new).entry(list as usize).or_default().bindings);
}

const PC_IA_SET_PRIMITIVE_TOPOLOGY: usize = 20;
const PC_RS_SET_VIEWPORTS: usize = 21;
const PC_RS_SET_SCISSOR_RECTS: usize = 22;
const PC_OM_SET_BLEND_FACTOR: usize = 23;
const PC_OM_SET_STENCIL_REF: usize = 24;
const PC_OM_SET_RENDER_TARGETS: usize = 46;

/// console #20 IASetPrimitiveTopology (PC #20)
/// The console's rect list (AMD's DI_PT_RECTLIST, 0x11): three vertices of a rectangle, the
/// fourth implied. The PC has no such topology: the draw is a triangle list, and a translated
/// pipeline of topology type 19 completes each rect in a geometry shader (`pso::rect_list_gs`);
/// where it has none (refused, see `real::rect_list_gs`) each rect is drawn as the triangle of
/// its three vertices (a retail game's full-screen passes).
pub(super) const RECT_LIST: u32 = 17;

pub(super) unsafe extern "system" fn ia_set_topology_x(this: *mut c_void, t: u32) {
    // a topology the PC does not have (console ones): logged once each
    if !matches!(t, 0..=5 | 10..=13 | 33..=64) {
        static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let new = {
            let mut g = SEEN.lock().unwrap();
            let n = !g.contains(&t);
            if n {
                g.push(t);
            }
            n
        };
        if new {
            crate::xlog_warn!("D3D12.X: IASetPrimitiveTopology({t}) – a console topology the PC does not have{}", if t == RECT_LIST { ": rect lists are drawn as triangle lists, completed by a GS in rect list pipelines" } else { "" });
        }
    }
    let rect = t == RECT_LIST;
    let t = if t == RECT_LIST { windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32 } else { t };
    with_bindings(this, |b| {
        b.topology = Some(t);
        b.rect_list = rect;
    });
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t)
}

/// console #21 RSSetViewports (PC #21)
pub(super) unsafe extern "system" fn rs_set_viewports_x(this: *mut c_void, n: u32, v: *const D3D12_VIEWPORT) {
    if !v.is_null() {
        let vs = std::slice::from_raw_parts(v, n as usize).to_vec();
        with_bindings(this, |b| b.viewports = vs);
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, n, v)
}

/// console #22 RSSetScissorRects (PC #22)
pub(super) unsafe extern "system" fn rs_set_scissors_x(this: *mut c_void, n: u32, r: *const windows::Win32::Foundation::RECT) {
    if !r.is_null() {
        let rs = std::slice::from_raw_parts(r, n as usize).to_vec();
        with_bindings(this, |b| b.scissors = rs);
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, n, r)
}

/// console #23 OMSetBlendFactor (PC #23)
pub(super) unsafe extern "system" fn om_set_blend_factor_x(this: *mut c_void, f: *const [f32; 4]) {
    let v = f.as_ref().copied();
    with_bindings(this, |b| b.blend = v);
    pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, f)
}

/// console #24 OMSetStencilRef (PC #24)
pub(super) unsafe extern "system" fn om_set_stencil_ref_x(this: *mut c_void, r: u32) {
    with_bindings(this, |b| b.stencil = Some(r));
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r)
}

/// OMSetRenderTargets as the PC list got it (`real::cl_om_set_render_targets`)
pub(super) unsafe fn note_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    let num_targets = if rtvs.is_null() { 0 } else { n.min(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) };
    let read_count = if single != 0 { num_targets.min(1) } else { num_targets } as usize;
    // SAFETY: `rtvs` is non-null whenever `read_count > 0`, and D3D12 requires at least 1
    // handle when `RTsSingleHandleToDescriptorRange != 0` or `num_targets` handles otherwise.
    let handles = if read_count == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_count).to_vec() };
    let d = dsv.as_ref().copied();
    let generation = d.and_then(|handle| super::depth_capture_views::note_bind(this as usize, handle, "dsv_om_bind"));
    with_bindings(this, |b| {
        b.depth_capture_dsv_generation = generation;
        b.lineage_om_valid = generation.is_some();
        b.render_targets = Some(RenderTargets { num_targets, handles, single, dsv: d });
    });
}

pub(super) fn note_lineage_render_pass(list: *mut c_void, beginning: bool, dsv: Option<usize>, depth_access: bool) {
    if !super::depth_lineage::enabled() {
        return;
    }
    let binding = dsv.and_then(|handle| {
        super::depth_capture_views::note_bind(list as usize, handle, "dsv_render_pass_bind").map(|generation| (handle, generation))
    });
    with_bindings(list, |b| {
        b.lineage_pass_dsv = beginning.then_some(binding.filter(|_| depth_access));
        if !beginning {
            b.lineage_om_valid = false;
        }
    });
}

pub(super) fn note_lineage_depth_draw(list: *mut c_void, call: &'static str) {
    if !super::depth_lineage::enabled() {
        return;
    }
    let state = {
        let lists = LISTS.lock().unwrap();
        lists.as_ref().and_then(|m| m.get(&(list as usize))).and_then(|s| {
            if s.using_fallback_pso || s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE) {
                return None;
            }
            let b = &s.bindings;
            let target = match b.lineage_pass_dsv {
                Some(target) => target,
                None if b.lineage_om_valid => b.render_targets.as_ref().and_then(|rt| rt.dsv).zip(b.depth_capture_dsv_generation),
                None => None,
            }?;
            Some((target, b.depth_capture_native_pso, s.draw_predicate.is_some() || b.predication.is_some() || s.predication.is_some()))
        })
    };
    if let Some(((handle, generation), pso, predicated)) = state {
        if device_x::lineage_depth_pso(pso) {
            super::depth_capture_views::note_depth_draw(list as usize, handle, generation, pso, call, predicated);
        }
    }
}

pub(super) fn capture_depth_binding(list: *mut c_void) -> Option<(usize, u64)> {
    let lists = LISTS.lock().unwrap();
    let s = lists.as_ref()?.get(&(list as usize))?;
    if s.bindings.predication.is_some() || s.predication.is_some() { return None }
    Some((s.bindings.render_targets.as_ref()?.dsv?, s.bindings.depth_capture_dsv_generation?))
}

pub(super) fn prepass_dsv(list: usize, pso: usize, planned:bool) -> Option<(usize, u64, Option<(f32, f32)>)> {
    let lists = LISTS.lock().unwrap();
    let s = lists.as_ref()?.get(&list)?;
    let b = &s.bindings;
    if s.list_type != Some(D3D12_COMMAND_LIST_TYPE_DIRECT) || s.using_fallback_pso || b.lineage_pass_dsv.is_some() || !b.lineage_om_valid
        || (!planned && b.depth_capture_native_pso != pso) || s.draw_predicate.is_some() || s.predication.is_some() || b.predication.is_some()
        || s.occlusion.active.is_some() || !device_x::lineage_depth_pso(pso) { return None; }
    Some((b.render_targets.as_ref()?.dsv?, b.depth_capture_dsv_generation?, s.depth_bounds))
}

/// The first viewport (x, y, width, height) and scissor `list` set, each with how many it set.
pub(super) fn viewport_scissor(list: usize) -> (Option<([f32; 4], usize)>, Option<([i32; 4], usize)>) {
    let g = LISTS.lock().unwrap();
    let Some(b) = g.as_ref().and_then(|m| m.get(&list)).map(|s| &s.bindings) else { return (None, None) };
    (
        b.viewports.first().map(|v| ([v.TopLeftX, v.TopLeftY, v.Width, v.Height], b.viewports.len())),
        b.scissors.first().map(|r| ([r.left, r.top, r.right, r.bottom], b.scissors.len())),
    )
}

pub(super) fn single_viewport(list: usize) -> Option<([f32;6],[i32;4])> {
    let g = LISTS.lock().unwrap();
    let b = &g.as_ref()?.get(&list)?.bindings;
    let [v] = b.viewports.as_slice() else { return None };
    let [s] = b.scissors.as_slice() else { return None };
    Some(([v.TopLeftX,v.TopLeftY,v.Width,v.Height,v.MinDepth,v.MaxDepth],[s.left,s.top,s.right,s.bottom]))
}

pub(super) fn capture_native_bind(list: *mut c_void, pso: usize) {
    if super::depth_capture_views::enabled() {
        with_bindings(list, |b| b.depth_capture_native_pso = pso);
    }
}

pub(super) fn capture_native_pso(list: *mut c_void) -> usize {
    LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).filter(|s| !s.using_fallback_pso)
        .map_or(0, |s| s.bindings.depth_capture_native_pso)
}

unsafe fn set_native_pso(list: *mut c_void, pso: usize) {
    super::shader_code::taken(list as usize, pso);
    pc::<unsafe extern "system" fn(*mut c_void, usize)>(list, PC_SET_PIPELINE_STATE)(list, pso);
    capture_native_bind(list, pso);
}

pub(super) fn bound_color_resources(list: *mut c_void) -> Vec<usize> {
    let mut targets = None;
    with_bindings(list, |b| targets = b.render_targets.clone());
    let Some(targets) = targets else { return Vec::new() };
    (0..targets.num_targets as usize).filter_map(|i| {
        let handle = if targets.single != 0 { *targets.handles.first()? + i * 32 } else { *targets.handles.get(i)? };
        super::present::rtv_resource(handle)
    }).collect()
}

/// Set again what `b` holds (after KickoffX's Reset or a deferred mid-list split).
unsafe fn restore_bindings(this: *mut c_void, b: &Bindings, ty: D3D12_COMMAND_LIST_TYPE) {
    if ty == D3D12_COMMAND_LIST_TYPE_COPY {
        return;
    }
    if !b.heaps.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, 28)(this, b.heaps.len() as u32, b.heaps.as_ptr());
    }
    if let Some((res, off, op)) = b.predication {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(this, PC_SET_PREDICATION)(this, res, off, op);
    }
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return;
    }
    if let Some(t) = b.topology {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t);
    }
    if !b.viewports.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, b.viewports.len() as u32, b.viewports.as_ptr());
    }
    if !b.scissors.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, b.scissors.len() as u32, b.scissors.as_ptr());
    }
    if let Some(f) = b.blend {
        pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, &f);
    }
    if let Some(r) = b.stencil {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r);
    }
    for (&slot, v) in &b.vertex_buffers {
        pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS)(this, slot, 1, v);
    }
    if let Some(rt) = &b.render_targets {
        let d = rt.dsv.unwrap_or(0);
        let (n, rts_ptr) = if rt.handles.is_empty() { (0, std::ptr::null()) } else { (rt.num_targets, rt.handles.as_ptr()) };
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(this, PC_OM_SET_RENDER_TARGETS)(
            this,
            n,
            rts_ptr,
            rt.single,
            if rt.dsv.is_some() { &d } else { std::ptr::null() },
        );
    }
}

#[derive(Default)]
struct ListState {
    helper: ia_live::Hazards,
    allocator: usize,
    list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    last_queue: Option<usize>,
    /// Title's currently bound PSO (before dynamic depth-bias override)
    bound_pso: usize,
    /// Active `RSSetDepthBiasX` override (None = zero / disabled)
    depth_bias: Option<DepthBias>,
    /// Last depth bias sent to `ID3D12GraphicsCommandList9::RSSetDepthBias` on this PC list
    /// since CreateCommandList / Reset / ClearState / KickoffX (resets to ZERO on each).
    native_bias: DepthBias,
    /// True when the PC list's bound PSO needs patching via the fallback cache before the next draw
    pso_dirty: bool,
    /// True when a fallback depth-bias derived PSO is currently bound on the PC list
    using_fallback_pso: bool,
    /// Bound index buffer (PC address), restored after `DrawIndexedX` and `KickoffX`
    index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>,
    /// Last `OMSetDepthBounds` values, restored after `KickoffX`
    depth_bounds: Option<(f32, f32)>,
    occlusion: Occlusion,
    /// Sorted, disjoint UMA address ranges written by this list since last Reset/KickoffX
    written_ranges: Vec<(u64, u64)>,
    /// Sparse-window `(va, bytes)` ranges written by memory ops (`Write32/64`, `Fill*`,
    /// `CopyMemoryToMemoryX`, timestamps) to copy back to the CPU view after execution.
    sparse_writes: Vec<(usize, usize)>,
    /// Non-timestamp 64-bit `(va, value)` writes recorded by `Write64BitValue{Bottom,Top}OfPipeX`
    /// so `ExecuteCommandLists` / `KickoffX` can signal per-address GPU fences for cross-queue
    /// `Wait64BitValueX` without blocking the submitter thread on the CPU.
    write64_signals: Vec<(u64, u64)>,
    /// External memory waits recorded on this list
    pending_waits: Vec<PendingWait>,
    /// bottom-of-pipe writes into memory the GPU does not see (`cpu_writes`)
    cpu_writes: Vec<super::cpu_writes::Write>,
    /// Dedicated 256-byte buffer holding the current predicate value (created lazily in COMMON state).
    pred_buf: Option<ID3D12Resource>,
    /// True while `pred_buf` is in `D3D12_RESOURCE_STATE_PREDICATION` on this open recording.
    pred_in_state: bool,
    /// Active `SetPredicationBufferX` op, restored across `KickoffX`.
    predication: Option<D3D12_PREDICATION_OP>,
    draw_predicate: Option<DrawPredicate>,
    /// state a PC Reset clears (restored after KickoffX)
    bindings: Bindings,
    /// what this segment records at the submit (`defer`)
    deferred: Option<Deferred>,
}

/// A call put off to the submit (`defer`), with its arguments.
#[derive(Clone, Copy, Debug)]
pub(super) enum Deferred {
    /// groups or, with `DispatchX`'s thread-dimension flag, threads; the `DispatchX` flags (None: `Dispatch`)
    Dispatch(u32, u32, u32, Option<u32>),
    /// signature, most commands, arguments and their offset, count buffer and its offset
    Indirect(usize, u32, usize, u64, usize, u64),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    /// `DrawIndexedX`: the indices' address, their format and count
    DrawIndexedX(u64, DXGI_FORMAT, u32),
}

impl Deferred {
    pub(super) fn draws(&self) -> bool {
        match *self {
            Deferred::Dispatch(..) => false,
            Deferred::Indirect(sig, ..) => !super::indirect::dispatches(sig),
            _ => true,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Deferred::Dispatch(..) => "dispatch",
            Deferred::Indirect(..) => "ExecuteIndirect",
            _ => "draw",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct DrawPredicate {
    pub title: Option<u64>,
    resource: usize,
    offset: u64,
    op: i32,
}

impl std::fmt::Debug for DrawPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "resource {:#x}+{:#x} title {:x?} op {}", self.resource, self.offset, self.title, self.op)
    }
}

pub(super) fn draw_gate_predicate(list: *mut c_void) -> Option<DrawPredicate> {
    LISTS.lock().unwrap().as_ref()?.get(&(list as usize))?.draw_predicate
}

pub(super) fn depth_copy_unpredicated(list: usize) -> bool {
    LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&list)).is_some_and(|s|
        s.draw_predicate.is_none() && s.bindings.predication.is_none() && s.predication.is_none())
}

pub(super) fn draw_gate_pso(list: *mut c_void) -> usize {
    LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).map_or(0, |s| s.bound_pso)
}

pub(super) fn note_draw_predicate(list: *mut c_void, title: Option<u64>, resource: usize, offset: u64, op: i32) {
    if super::bind::draw_gate_trace() || super::depth_capture::enabled() {
        let mut lists = LISTS.lock().unwrap();
        let s = lists.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        s.draw_predicate = (resource != 0).then_some(DrawPredicate { title, resource, offset, op });
    }
}

impl ListState {
    fn note_write(&mut self, va: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = va.saturating_add(size);
        let ranges = &mut self.written_ranges;
        let first = ranges.partition_point(|&(_, stop)| stop < va);
        let count = ranges[first..].partition_point(|&(start, _)| start <= end);
        if count == 0 {
            ranges.insert(first, (va, end));
        } else {
            ranges[first] = (va.min(ranges[first].0), end.max(ranges[first + count - 1].1));
            ranges.drain(first + 1..first + count);
        }
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(va as usize)) {
            if self.sparse_writes.is_empty() {
                SPARSE_WRITE_LISTS.fetch_add(1, Ordering::AcqRel);
            }
            self.sparse_writes.push((va as usize, size as usize));
        }
    }

    fn clear_sparse_writes(&mut self) -> Vec<(usize, usize)> {
        if self.sparse_writes.is_empty() {
            return Vec::new();
        }
        SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        std::mem::take(&mut self.sparse_writes)
    }

    fn push_write64_signal(&mut self, va: u64, value: u64) {
        if self.write64_signals.is_empty() {
            WRITE64_SIGNAL_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.write64_signals.push((va, value));
    }

    fn clear_write64_signals(&mut self) -> Vec<(u64, u64)> {
        if self.write64_signals.is_empty() {
            return Vec::new();
        }
        WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
        std::mem::take(&mut self.write64_signals)
    }

    fn has_prior_write(&self, va: u64, size: u64) -> bool {
        let end = va.saturating_add(size);
        let first = self.written_ranges.partition_point(|&(_, stop)| stop <= va);
        self.written_ranges.get(first).is_some_and(|&(start, _)| start < end)
    }

    fn clear_waits(&mut self) {
        if !self.pending_waits.is_empty() {
            self.pending_waits.clear();
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn push_wait(&mut self, w: PendingWait) {
        if self.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.pending_waits.push(w);
    }
}

#[derive(Clone, Copy)]
struct QueueInfo {
    queue: usize,
    device: usize,
    ty: D3D12_COMMAND_LIST_TYPE,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);
static LIVE_QUEUES: Mutex<Vec<QueueInfo>> = Mutex::new(Vec::new());
/// Number of lists with at least one pending `Wait32/64BitValueX` (fast-path guard for ExecuteCommandLists).
static PENDING_WAIT_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Number of lists with recorded sparse-window memory writes (fast-path guard for post-execute readback).
static SPARSE_WRITE_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Number of lists with recorded non-timestamp `Write64BitValue*` signals (fast-path guard).
static WRITE64_SIGNAL_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Set once any list uses `RSSetDepthBiasX` (fast-path guard for Draw* hooks).
static DEPTH_BIAS_USED: AtomicBool = AtomicBool::new(false);
/// Set once any list uses `SetPredicationBufferX` (fast-path guard for Close/Reset hooks).
static PREDICATION_USED: AtomicBool = AtomicBool::new(false);

/// Record a newly created title command queue so `KickoffX` can find a compatible queue.
pub fn on_queue_create(queue: usize, device: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if queue == 0 {
        return;
    }
    let mut g = LIVE_QUEUES.lock().unwrap();
    g.retain(|q| q.queue != queue);
    g.push(QueueInfo { queue, device, ty });
}

/// Remove a released command queue so `KickoffX` never submits to a dead queue.
pub(super) fn on_queue_release(queue: usize) {
    LIVE_QUEUES.lock().unwrap().retain(|q| q.queue != queue);
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.last_queue == Some(queue) {
                s.last_queue = None;
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Optional test interceptor for `try_native_depth_bias` so unit tests deterministically
    /// verify the native `ID3D12GraphicsCommandList9::RSSetDepthBias` path and its fallback
    /// even on Windows builds whose inbox D3D12Core.dll stops at `ID3D12GraphicsCommandList7`.
    static TEST_NATIVE_BIAS: std::cell::RefCell<Option<Box<dyn FnMut(usize, DepthBias) -> bool>>> = const { std::cell::RefCell::new(None) };
}

/// Apply depth bias natively via `ID3D12GraphicsCommandList9::RSSetDepthBias` (exact f32,
/// without rounding to i32 or deriving a patched PSO). Returns `false` if unavailable.
unsafe fn try_native_depth_bias(list: *mut c_void, bias: DepthBias) -> bool {
    if list.is_null() {
        return false;
    }
    #[cfg(test)]
    if let Some(res) = TEST_NATIVE_BIAS.with(|h| h.borrow_mut().as_mut().map(|f| f(list as usize, bias))) {
        return res;
    }
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else {
        return false;
    };
    let Ok(cl9) = cl.cast::<ID3D12GraphicsCommandList9>() else {
        return false;
    };
    cl9.RSSetDepthBias(bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias());
    true
}

/// Record a newly created command list's allocator, initial pipeline, and type.
///
/// # Safety
/// `list` must be null or a valid `ID3D12GraphicsCommandList` COM pointer.
pub unsafe fn on_create(list: *mut c_void, alloc: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    super::physical_directory::recordings::ended(list as usize);
    if list.is_null() {
        return;
    }
    let key = list as usize;
    let old = {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let old = m.remove(&key);
        if let Some(ref old) = old {
            if !old.pending_waits.is_empty() {
                PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
            if !old.sparse_writes.is_empty() {
                SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
            if !old.write64_signals.is_empty() {
                WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        m.insert(key, ListState { allocator: alloc, list_type: Some(ty), bound_pso: pso, ..Default::default() });
        old
    };
    drop(old);
    super::title_textures::list_reset(key);
    super::depth_state::reset(key);
    // Dynamic-depth-bias PSOs ignore baked RasterizerState depth bias; seed the list with the
    // initial PSO's static bias if non-zero (CreateCommandList resets RSSetDepthBias to 0).
    if ty != D3D12_COMMAND_LIST_TYPE_BUNDLE && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Clean up all tracked state when a command list is released for good.
pub(super) fn on_release(list: usize) {
    super::title_textures::list_reset(list);
    super::depth_state::reset(list);
    let old = LISTS.lock().unwrap().as_mut().and_then(|m| m.remove(&list));
    if let Some(old) = old {
        if !old.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
        if !old.sparse_writes.is_empty() {
            SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
        if !old.write64_signals.is_empty() {
            WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Clear per-recording sync/write tracking on a split segment moved into `title_list`'s spare pool.
pub(super) fn clear_segment_state(list: usize) {
    super::gpu_timing::forget(list);
    super::title_textures::list_reset(list);
    super::depth_state::reset(list);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&list)) {
        s.helper = Default::default();
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.clear_sparse_writes();
        s.clear_write64_signals();
        s.clear_waits();
        s.cpu_writes.clear();
        s.predication = None;
        s.draw_predicate = None;
        s.pred_in_state = false;
        s.bindings = Bindings::default();
        s.deferred = None;
    }
}

#[cfg(test)]
pub(super) fn has_list_state(list: usize) -> bool {
    LISTS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&list))
}

/// Forget a released command allocator so `KickoffX` will not reset a list with a dead allocator.
pub(super) fn on_allocator_release(alloc: usize) {
    on_allocator_reset(alloc);
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.allocator == alloc {
                s.allocator = 0;
            }
        }
    }
    // (released outside the lock: a companion's Release comes back here)
    let gone: Vec<_> = COMPANIONS.lock().unwrap().as_mut().map_or_else(Vec::new, |m| {
        let keys: Vec<_> = m.keys().filter(|k| k.0 == alloc).copied().collect();
        keys.into_iter().filter_map(|k| m.remove(&k)).collect()
    });
    drop(gone);
}

/// Track `SetPipelineState` on a command list (for `RSSetDepthBiasX` and `KickoffX`),
/// syncing `ID3D12GraphicsCommandList9::RSSetDepthBias` when the bound PSO uses native dynamic
/// depth bias.
pub(super) unsafe fn on_set_pipeline_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let sync_target = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        // Keep the title-visible base PSO key when `bind_abi` re-binds its resolved background PSO.
        if s.bound_pso == 0 || pso == s.bound_pso || super::pso_async::resolve_if_ready(s.bound_pso) != pso {
            s.bound_pso = pso;
        }
        let effective = s.bound_pso;
        s.using_fallback_pso = false;
        let is_bundle = s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE);
        let info = if effective != 0 { device_x::base_depth_bias_info(effective) } else { None };
        if !is_bundle {
            if let Some((true, static_bias)) = info {
                let target = s.depth_bias.unwrap_or(static_bias);
                s.pso_dirty = false;
                if target != s.native_bias {
                    Some((target, s.depth_bias.is_some()))
                } else {
                    None
                }
            } else {
                s.pso_dirty = s.depth_bias.is_some();
                if s.pso_dirty {
                    DEPTH_BIAS_USED.store(true, Ordering::Release);
                }
                None
            }
        } else {
            s.pso_dirty = s.depth_bias.is_some();
            if s.pso_dirty {
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some((target, has_override)) = sync_target {
        if try_native_depth_bias(list, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
        } else if has_override {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
    }
}

/// Track `ClearState` on a command list: resets dynamic state and binds `pso`.
pub(super) unsafe fn on_clear_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&key) {
            s.bound_pso = pso;
            s.depth_bias = None;
            s.native_bias = DepthBias::ZERO;
            s.pso_dirty = false;
            s.using_fallback_pso = false;
            s.index_buffer = None;
            s.depth_bounds = None;
            s.predication = None;
            s.draw_predicate = None;
            s.bindings = Bindings::default();
            s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
        } else {
            false
        }
    };
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Associate executed lists with `queue` and keep `queue` at the head of `LIVE_QUEUES`.
pub(super) unsafe fn note_queue_for_lists(queue: usize, lists: &[*mut c_void]) {
    if queue == 0 {
        return;
    }
    {
        let mut g = LIVE_QUEUES.lock().unwrap();
        if let Some(pos) = g.iter().position(|q| q.queue == queue) {
            let info = g.remove(pos);
            g.push(info);
        } else {
            let raw = queue as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                let ty = q.GetDesc().Type;
                let mut dev: Option<ID3D12Device> = None;
                let dev_raw = if q.GetDevice(&mut dev).is_ok() {
                    dev.as_ref().map_or(0, |d| d.as_raw() as usize)
                } else {
                    0
                };
                g.push(QueueInfo { queue, device: dev_raw, ty });
            }
        }
    }
    let mut g = LISTS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for &l in lists {
        if !l.is_null() {
            m.entry(l as usize).or_default().last_queue = Some(queue);
        }
    }
}

fn pick_queue(last_queue: Option<usize>, device: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<usize> {
    let g = LIVE_QUEUES.lock().unwrap();
    if let Some(q) = last_queue {
        if g.iter().any(|info| info.queue == q) {
            return Some(q);
        }
    }
    g.iter().rev().find(|info| info.ty == ty && (device == 0 || info.device == 0 || info.device == device)).map(|info| info.queue)
}

fn note_write(list: usize, va: u64, size: u64) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().note_write(va, size);
    unsafe { super::sparse::before_gpu_write(va as usize, size as usize) };
}

/// `[offset, offset + bytes)` of a buffer of `size` bytes at `base`, clamped to the buffer:
/// (address, bytes). None: nothing of it lies in the buffer.
fn buffer_span(base: usize, size: usize, offset: u64, bytes: u64) -> Option<(usize, usize)> {
    let begin = offset.min(size as u64) as usize;
    let end = offset.saturating_add(bytes).min(size as u64) as usize;
    (begin < end).then(|| (base + begin, end - begin))
}

/// The title's GPU work on `list` writes `[offset, offset + bytes)` of buffer `dst` (a copy or a
/// query resolve into it). On the console that is the title's memory once the work is done. A
/// buffer in a sparse window is in video memory here: the range joins the list's sparse writes,
/// so it is copied back when the list's submission returns (`readback_sparse_writes`) and not
/// only at the title's next Map. `cpu_read_only`: only for a buffer the title made to read on
/// the CPU (buffer to buffer copies are mostly uploads the CPU never reads).
fn note_buffer_write(list: usize, dst: usize, offset: u64, bytes: u64, cpu_read_only: bool) {
    if cpu_read_only && !real::is_sparse_readback(dst) {
        return;
    }
    let Some((va, n)) = real::console_buffer(dst).and_then(|(base, size)| buffer_span(base, size, offset, bytes)) else { return };
    if !crate::gpu::uma::get().is_some_and(|u| u.is_sparse(va)) {
        return;
    }
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(list).or_default();
    if s.sparse_writes.is_empty() {
        SPARSE_WRITE_LISTS.fetch_add(1, Ordering::AcqRel);
    }
    s.sparse_writes.push((va, n));
    drop(g);
    unsafe { super::sparse::before_gpu_write(va, n) };
}

// ---------------------------------------------------------------------------------------
// per-address GPU fence bridge for monotonic 64-bit Write64BitValue* -> Wait64BitValueX
// ---------------------------------------------------------------------------------------

/// Recent `Write32/64BitValue*` call for timeout diagnostics.
#[derive(Clone, Copy)]
struct WriteTraceEntry {
    va: u64,
    value: u64,
    flags: u32,
    list_type: i32,
    what: &'static str,
}

struct WriteTraceRing {
    entries: [Option<WriteTraceEntry>; 32],
    next: usize,
    total: u64,
}

impl WriteTraceRing {
    const fn new() -> Self {
        Self { entries: [None; 32], next: 0, total: 0 }
    }

    fn push(&mut self, e: WriteTraceEntry) {
        self.entries[self.next] = Some(e);
        self.next = (self.next + 1) % self.entries.len();
        self.total += 1;
    }

    fn snapshot(&self) -> (u64, Vec<WriteTraceEntry>) {
        let mut out = Vec::with_capacity(self.entries.len());
        for i in 0..self.entries.len() {
            let idx = (self.next + i) % self.entries.len();
            if let Some(e) = self.entries[idx] {
                out.push(e);
            }
        }
        (self.total, out)
    }
}

static WRITE_TRACE: Mutex<WriteTraceRing> = Mutex::new(WriteTraceRing::new());

fn record_write_trace(list: *mut c_void, va: u64, value: u64, flags: u32, what: &'static str) {
    let ty = LISTS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .and_then(|s| s.list_type)
        .map_or(-1, |t| t.0);
    WRITE_TRACE.lock().unwrap().push(WriteTraceEntry { va, value, flags, list_type: ty, what });
}

/// Active GPU `queue.Wait(&fence, val)` tracked for exact cycle detection and deadlock watchdog.
#[derive(Clone, Copy)]
struct ArmedGpuWait {
    val: u64,
    waiting_queue: usize,
    waiting_queue_ty: i32,
    /// Per-queue submission sequence number at which this `Wait` was issued on `waiting_queue`.
    wait_seq: u64,
    armed_at: Instant,
    after_write64: bool,
}

/// Exact GPU `queue.Signal(&fence, val)` submission point on `queue`.
#[derive(Clone, Copy)]
struct SubmittedSignal {
    val: u64,
    queue: usize,
    /// Per-queue submission sequence number at which this `Signal` was issued on `queue`.
    seq: u64,
}

/// Per-address GPU fence state bridging `Write64BitValue{Bottom,Top}OfPipeX` to cross-queue
/// `Wait64BitValueX` (`GREATER_EQUAL`, full 64-bit mask).
#[derive(Default)]
struct SyncCell {
    fence: Option<ID3D12Fence>,
    /// Raw `ID3D12Device` pointer owning `fence`.
    device: usize,
    /// Highest non-zero value recorded in a `Write64BitValue*` targeting this VA.
    recorded_max: u64,
    /// Command list type of the most recent recording advancing `recorded_max`.
    recorded_list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    /// Last queue the recording list was executed on (if known).
    recorded_last_queue: Option<usize>,
    /// Highest value submitted via `queue.Signal(&fence, v)` or released on CPU.
    submitted_max: u64,
    /// Queue that issued the latest GPU `Signal` (`None` if released on CPU).
    submitted_queue: Option<usize>,
    /// Highest value explicitly released on CPU (cycle-break or watchdog). Protects against
    /// vkd3d-proton rewinding `fence->virtual_value` when an older in-flight GPU signal retires.
    cpu_signaled_max: u64,
    /// Set once a strictly decreasing write (`value < submitted_max`) is observed at this VA,
    /// disabling GPU-fence bridging and falling back to CPU polling.
    non_monotonic: bool,
    /// Active GPU `queue.Wait(&fence, v)` calls on this cell not yet confirmed completed on GPU.
    armed_waits: Vec<ArmedGpuWait>,
    /// Active GPU `queue.Signal(&fence, v)` calls on this cell not yet confirmed completed on GPU.
    submitted_signals: Vec<SubmittedSignal>,
}

// SAFETY: `ID3D12Fence` is a free-threaded COM object in D3D12.
unsafe impl Send for SyncCell {}

/// Tracked title `ID3D12CommandQueue::Wait` / `Signal` edges on a title `ID3D12Fence`, used so
/// exact cycle detection sees mixed cycles involving both `Wait64BitValueX` and title fences.
#[derive(Default)]
struct TitleFenceSync {
    waits: Vec<(u64, usize, u64)>,   // (val, waiting_queue, wait_seq)
    signals: Vec<SubmittedSignal>,   // (val, signaling_queue, sig_seq)
}

#[derive(Default)]
struct SyncGraph {
    cells: HashMap<u64, SyncCell>,
    title_fences: HashMap<usize, TitleFenceSync>,
    queue_seq: HashMap<usize, u64>,
}

impl SyncGraph {
    fn next_seq(&mut self, queue: usize) -> u64 {
        let s = self.queue_seq.entry(queue).or_insert(0);
        *s += 1;
        *s
    }
}

static SYNC_GRAPH: Mutex<Option<SyncGraph>> = Mutex::new(None);

/// The last sync-graph seq of `queue` (its Signals and Waits so far), for naming submissions.
pub(super) fn queue_seq_now(queue: usize) -> u64 {
    SYNC_GRAPH.lock().unwrap().as_ref().and_then(|g| g.queue_seq.get(&queue).copied()).unwrap_or(0)
}
static ARMED_GPU_WAITS: AtomicUsize = AtomicUsize::new(0);
static SYNC_WATCHDOG_WAKE: OnceLock<(Mutex<()>, Condvar)> = OnceLock::new();
static QUEUE_IDS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static LIVE_GPU_WAIT: Mutex<Option<(Instant, crate::live_stall::Sample)>> = Mutex::new(None);

fn diagnostic_queue_id(raw: usize) -> Option<usize> {
    QUEUE_IDS.try_lock().ok()?.iter().position(|&q| q == raw)
}

pub(super) fn diagnostic_gpu_wait() -> Option<crate::live_stall::Sample> {
    if ARMED_GPU_WAITS.load(Ordering::Acquire) == 0 {
        return None;
    }
    let (observed, sample) = (*LIVE_GPU_WAIT.try_lock().ok()?)?;
    (observed.elapsed() < Duration::from_millis(500)).then_some(sample)
}

pub(super) fn queue_id(q: usize) -> usize {
    let mut g = QUEUE_IDS.lock().unwrap();
    if let Some(pos) = g.iter().position(|&x| x == q) {
        pos
    } else {
        let id = g.len();
        g.push(q);
        id
    }
}

/// Remove any `armed_waits` and `submitted_signals` already satisfied on the GPU (or if the device
/// was removed). Returns the fence's current completed value (or `0` if no fence exists yet).
unsafe fn prune_completed_waits(cell: &mut SyncCell) -> u64 {
    let Some(ref fence) = cell.fence else { return 0 };
    let completed = fence.GetCompletedValue();
    if completed == u64::MAX {
        let removed = cell.armed_waits.len();
        cell.armed_waits.clear();
        cell.submitted_signals.clear();
        if removed > 0 {
            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
        }
        return completed;
    }
    if !cell.armed_waits.is_empty() {
        let before = cell.armed_waits.len();
        cell.armed_waits.retain(|aw| aw.val > completed);
        let removed = before - cell.armed_waits.len();
        if removed > 0 {
            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
        }
    }
    if !cell.submitted_signals.is_empty() {
        cell.submitted_signals.retain(|s| s.val > completed);
    }
    completed
}

/// Prune completed waits/signals from a tracked title `ID3D12Fence`.
unsafe fn prune_title_fence(fence_raw: usize, tf: &mut TitleFenceSync) {
    if tf.waits.is_empty() && tf.signals.is_empty() {
        return;
    }
    let p = fence_raw as *mut c_void;
    let Some(f) = ID3D12Fence::from_raw_borrowed(&p) else { return };
    let completed = f.GetCompletedValue();
    if completed == u64::MAX {
        tf.waits.clear();
        tf.signals.clear();
    } else if completed > 0 {
        tf.waits.retain(|&(val, _, _)| val > completed);
        tf.signals.retain(|s| s.val > completed);
    }
}

/// Exact `(queue, seq)` cycle detection BEFORE emitting `sig_q.Signal(&cell_va_fence, sig_val)`.
///
/// Theorem: in any GPU deadlock cycle of length `k >= 1` (`q_0 -> q_1 -> ... -> q_0`), every queue
/// `q_i` submits its `Wait` strictly before its `Signal` (`wait_seq < sig_seq`). Therefore the
/// chronologically last call closing any cycle is ALWAYS a `Signal`. Checking the exact submitted
/// `(queue, seq)` graph right before emitting each `Signal` catches 100% of cycles (including
/// same-queue self-deadlocks, 2..N-queue cycles across rotating compute queues, and mixed cycles
/// involving title `ID3D12Fence` waits/signals) before any cyclic wait-signal pair reaches Vulkan.
///
/// Returns `true` iff emitting `sig_q.Signal` for `(target_va, sig_val)` would close a cycle.
unsafe fn would_signal_close_cycle(graph: &mut SyncGraph, sig_q: usize, target_va: u64, sig_val: u64) -> bool {
    // Prune completed waits/signals on target cell first.
    let target_waiters: Vec<(usize, u64)> = {
        let Some(target) = graph.cells.get_mut(&target_va) else { return false };
        let completed = prune_completed_waits(target);
        if completed == u64::MAX || target.armed_waits.is_empty() {
            return false;
        }
        target.armed_waits.iter().filter(|aw| aw.val <= sig_val).map(|aw| (aw.waiting_queue, aw.wait_seq)).collect()
    };
    if target_waiters.is_empty() {
        return false;
    }
    // k = 1: direct same-queue self-deadlock (`sig_q` is already waiting on `target_va <= sig_val`).
    if target_waiters.iter().any(|&(wq, _)| wq == sig_q) {
        return true;
    }

    // Prune all cells & title fences and build exact edge list: (waiter_q, wait_seq, prod_q, prod_sig_seq).
    // Include earlier submitted signals on `target_va` too (e.g. frame N-1 wait -> signal).
    let mut edges: Vec<(usize, u64, usize, u64)> = Vec::new();
    for (&_va, cell) in graph.cells.iter_mut() {
        if cell.armed_waits.is_empty() {
            continue;
        }
        let completed = prune_completed_waits(cell);
        if completed == u64::MAX || cell.armed_waits.is_empty() {
            continue;
        }
        for aw in &cell.armed_waits {
            // Earliest submitted GPU signal satisfying `aw.val` (matches vkd3d-proton update_count order).
            if let Some(s) = cell.submitted_signals.iter().find(|s| s.val >= aw.val) {
                edges.push((aw.waiting_queue, aw.wait_seq, s.queue, s.seq));
            }
        }
    }
    for (&tf_raw, tf) in graph.title_fences.iter_mut() {
        prune_title_fence(tf_raw, tf);
        for &(val, wq, wseq) in &tf.waits {
            if let Some(s) = tf.signals.iter().find(|s| s.val >= val) {
                edges.push((wq, wseq, s.queue, s.seq));
            }
        }
    }
    if edges.is_empty() {
        return false;
    }

    // BFS over `(queue, limit_seq)`: all waits on `sig_q` sit before our new `sig_q.Signal` (`u64::MAX`).
    let mut best_limit: HashMap<usize, u64> = HashMap::new();
    let mut stack: Vec<(usize, u64)> = vec![(sig_q, u64::MAX)];
    best_limit.insert(sig_q, u64::MAX);
    while let Some((cur_q, limit_seq)) = stack.pop() {
        for &(wq, wseq, pq, pseq) in &edges {
            if wq != cur_q || wseq >= limit_seq {
                continue;
            }
            // If `pq` has a wait on `target_va <= sig_val` strictly BEFORE `pq`'s signal `pseq`,
            // then `pq`'s signal cannot execute until `target_va` completes -> exact cycle!
            if target_waiters.iter().any(|&(twq, twseq)| twq == pq && twseq < pseq) {
                return true;
            }
            let prev = best_limit.get(&pq).copied().unwrap_or(0);
            if pseq > prev {
                best_limit.insert(pq, pseq);
                stack.push((pq, pseq));
            }
        }
    }
    false
}

/// Record a title `ID3D12CommandQueue::Wait(fence, value)` in the exact sync graph.
pub(super) unsafe fn note_title_queue_wait(queue: usize, fence: usize, value: u64) {
    if queue == 0 || fence == 0 || value == 0 {
        return;
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if once(&LOGGED, 8) {
        crate::xlog_info!("D3D12.X: title ID3D12CommandQueue::Wait on q{} fence {fence:#x} >= {value:#x}", queue_id(queue));
    }
    let mut g = SYNC_GRAPH.lock().unwrap();
    let graph = g.get_or_insert_with(SyncGraph::default);
    let seq = graph.next_seq(queue);
    let tf = graph.title_fences.entry(fence).or_default();
    prune_title_fence(fence, tf);
    tf.waits.push((value, queue, seq));
    if traced_queue(queue) {
        trace_queue(|| format!("q{} seq {seq}: title fence {fence:#x} Wait >= {value:#x}", queue_id(queue)));
    }
}

/// Record a title `ID3D12CommandQueue::Signal(fence, value)` in the exact sync graph.
pub(super) unsafe fn note_title_queue_signal(queue: usize, fence: usize, value: u64) {
    if queue == 0 || fence == 0 || value == 0 {
        return;
    }
    let mut g = SYNC_GRAPH.lock().unwrap();
    let graph = g.get_or_insert_with(SyncGraph::default);
    let seq = graph.next_seq(queue);
    let tf = graph.title_fences.entry(fence).or_default();
    prune_title_fence(fence, tf);
    tf.signals.push(SubmittedSignal { val: value, queue, seq });
    if traced_queue(queue) {
        trace_queue(|| format!("q{} seq {seq}: title fence {fence:#x} Signal = {value:#x}", queue_id(queue)));
    }
}

/// Before a title `ID3D12CommandQueue::Signal(title_fence, value)` is submitted, check if it would
/// close a mixed cycle passing through any `Wait64BitValueX` `SyncCell`. If so, release that
/// `SyncCell` on CPU BEFORE the title signal executes so Vulkan never enters a semaphore cycle.
pub(super) unsafe fn break_cycles_before_title_signal(sig_q: usize, title_fence: usize, sig_val: u64) {
    if sig_q == 0 || title_fence == 0 || sig_val == 0 {
        return;
    }
    let to_release: Vec<(u64, u64, ID3D12Fence)> = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let Some(graph) = g.as_mut() else { return };
        let Some(tf) = graph.title_fences.get_mut(&title_fence) else { return };
        prune_title_fence(title_fence, tf);
        let target_waiters: Vec<(usize, u64)> = tf.waits.iter().filter(|&&(v, _, _)| v <= sig_val).map(|&(_, wq, wseq)| (wq, wseq)).collect();
        if target_waiters.is_empty() {
            return;
        }
        // Build edges tagging whether each edge is a Wait64 SyncCell `(Some((va, aw_val)))` or title fence (`None`).
        let mut edges: Vec<(usize, u64, usize, u64, Option<(u64, u64)>)> = Vec::new();
        for (&va, cell) in graph.cells.iter_mut() {
            if cell.armed_waits.is_empty() {
                continue;
            }
            let completed = prune_completed_waits(cell);
            if completed == u64::MAX || cell.armed_waits.is_empty() {
                continue;
            }
            for aw in &cell.armed_waits {
                if let Some(s) = cell.submitted_signals.iter().find(|s| s.val >= aw.val) {
                    edges.push((aw.waiting_queue, aw.wait_seq, s.queue, s.seq, Some((va, aw.val))));
                }
            }
        }
        for (&tf_raw, other_tf) in graph.title_fences.iter_mut() {
            if tf_raw == title_fence {
                continue;
            }
            prune_title_fence(tf_raw, other_tf);
            for &(val, wq, wseq) in &other_tf.waits {
                if let Some(s) = other_tf.signals.iter().find(|s| s.val >= val) {
                    edges.push((wq, wseq, s.queue, s.seq, None));
                }
            }
        }
        // BFS carrying the first Wait64 SyncCell edge encountered on the path from `sig_q`.
        let mut best_limit: HashMap<usize, u64> = HashMap::new();
        let mut stack: Vec<(usize, u64, Option<(u64, u64)>)> = vec![(sig_q, u64::MAX, None)];
        best_limit.insert(sig_q, u64::MAX);
        let mut break_cells: Vec<(u64, u64)> = Vec::new();
        while let Some((cur_q, limit_seq, first_cell)) = stack.pop() {
            for &(wq, wseq, pq, pseq, cell_edge) in &edges {
                if wq != cur_q || wseq >= limit_seq {
                    continue;
                }
                let tracked = first_cell.or(cell_edge);
                if target_waiters.iter().any(|&(twq, twseq)| twq == pq && twseq < pseq) {
                    if let Some((va, val)) = tracked {
                        if !break_cells.iter().any(|&(bva, bval)| bva == va && bval >= val) {
                            break_cells.push((va, val));
                        }
                    }
                    continue;
                }
                let prev = best_limit.get(&pq).copied().unwrap_or(0);
                if pseq > prev {
                    best_limit.insert(pq, pseq);
                    stack.push((pq, pseq, tracked));
                }
            }
        }
        let mut out = Vec::new();
        for (va, val) in break_cells {
            if let Some(cell) = graph.cells.get_mut(&va) {
                cell.cpu_signaled_max = cell.cpu_signaled_max.max(val);
                let before = cell.armed_waits.len();
                cell.armed_waits.retain(|aw| aw.val > val);
                let removed = before - cell.armed_waits.len();
                if removed > 0 {
                    ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                }
                cell.submitted_signals.retain(|s| s.val > val);
                if let Some(ref f) = cell.fence {
                    out.push((va, val, f.clone()));
                }
            }
        }
        out
    };
    for (va, val, fence) in to_release {
        let _ = fence.Signal(val);
        static MIXED_WARN: AtomicU64 = AtomicU64::new(0);
        if let Some(n) = cpu_release(&MIXED_WARN) {
            crate::xlog_warn!(
                "D3D12.X: Wait64 fence {va:#x} >= {val:#x} released on CPU before title queue q{} Signal({title_fence:#x}, {sig_val:#x}) to break mixed GPU cycle (#{n} here)",
                queue_id(sig_q)
            );
        }
    }
}

/// Whether an armed GPU wait has outlasted the watchdog. Its clock stands while one of the title's
/// threads is held for a pipeline build (`pso_async::title_held`: a bind's wait, or a submit's for
/// the calls put off to it) and starts again after, as `kernelx::Sleep`'s does.
fn wait_expired(aw: &mut ArmedGpuWait, held: bool) -> bool {
    if held {
        aw.armed_at = Instant::now();
    }
    aw.armed_at.elapsed() >= gpu_wait_watchdog_timeout()
}

fn wake_sync_watchdog() {
    let pair = SYNC_WATCHDOG_WAKE.get_or_init(|| {
        let pair = (Mutex::new(()), Condvar::new());
        let _ = std::thread::Builder::new().name("xcl-sync-watchdog".into()).spawn(|| loop {
            let (lock, cv) = SYNC_WATCHDOG_WAKE.get().unwrap();
            let guard = lock.lock().unwrap();
            if ARMED_GPU_WAITS.load(Ordering::Acquire) == 0 {
                if let Some(most) = gpu_wait_watch() {
                    watch_gpu_waits(Duration::ZERO, most);
                }
                let _g = cv.wait(guard).unwrap();
                continue;
            }
            let _ = cv.wait_timeout(guard, Duration::from_millis(50)).unwrap();
            let mut oldest = Duration::ZERO;
            let mut live = None;
            let diagnostics = crate::live_stall::enabled();
            let (expired, rewinds, producers) = {
                let mut g = SYNC_GRAPH.lock().unwrap();
                let mut out: Vec<(u64, u64, ID3D12Fence, usize, i32, Option<usize>, Option<i32>, u64, u64, u64)> = Vec::new();
                let mut rew: Vec<(ID3D12Fence, u64)> = Vec::new();
                let (mut signal_seqs, mut producers) = (Vec::new(), Vec::new());
                if let Some(graph) = g.as_mut() {
                    for (&va, cell) in graph.cells.iter_mut() {
                        if cell.armed_waits.is_empty() && cell.cpu_signaled_max == 0 {
                            continue;
                        }
                        let completed = unsafe { prune_completed_waits(cell) };
                        if completed == u64::MAX {
                            continue;
                        }
                        // Guard against vkd3d-proton rewinding `fence->virtual_value` when an older
                        // in-flight GPU signal retires after a CPU cycle-break/watchdog signal.
                        if cell.cpu_signaled_max > completed {
                            if let Some(ref f) = cell.fence {
                                rew.push((f.clone(), cell.cpu_signaled_max));
                            }
                        }
                        if cell.armed_waits.is_empty() {
                            continue;
                        }
                        oldest = cell.armed_waits.iter().map(|aw| aw.armed_at.elapsed()).fold(oldest, Duration::max);
                        let mut max_exp = 0u64;
                        let mut waiter_q = 0usize;
                        let mut waiter_ty = -1i32;
                        let before = cell.armed_waits.len();
                        let held = super::pso_async::defer_to_submit() && super::pso_async::title_held();
                        cell.armed_waits.retain_mut(|aw| {
                            if wait_expired(aw, held) {
                                if aw.val >= max_exp {
                                    max_exp = aw.val;
                                    waiter_q = aw.waiting_queue;
                                    waiter_ty = aw.waiting_queue_ty;
                                }
                                return false;
                            }
                            if diagnostics && live.is_none_or(|(since, _, _)| aw.armed_at < since) {
                                live = Some((aw.armed_at, aw.val, aw.waiting_queue));
                            }
                            true
                        });
                        let removed = before - cell.armed_waits.len();
                        if removed > 0 {
                            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                        }
                        if max_exp > 0 {
                            cell.cpu_signaled_max = cell.cpu_signaled_max.max(max_exp);
                            let signal_seq = cell.submitted_signals.iter().find(|s| s.val >= max_exp).map(|s| s.seq);
                            cell.submitted_signals.retain(|s| s.val > max_exp);
                            if let Some(ref f) = cell.fence {
                                signal_seqs.push(signal_seq);
                                out.push((
                                    va,
                                    max_exp,
                                    f.clone(),
                                    waiter_q,
                                    waiter_ty,
                                    cell.submitted_queue.or(cell.recorded_last_queue),
                                    cell.recorded_list_type.map(|t| t.0),
                                    cell.recorded_max,
                                    cell.submitted_max,
                                    completed,
                                ));
                            }
                        }
                    }
                    producers = out.iter().zip(&signal_seqs).map(|(e, &seq)| producer_state(graph, e.5, seq)).collect();
                }
                (out, rew, producers)
            };
            if diagnostics {
                if let Ok(mut sample) = LIVE_GPU_WAIT.try_lock() {
                    *sample = live.map(|(since, target, queue)| (Instant::now(), crate::live_stall::Sample {
                        stage: crate::live_stall::Stage::QueueFenceWait, since, tid: 0, pipeline: 0, shader: 0,
                        target: Some(target), queue: diagnostic_queue_id(queue),
                    }));
                }
            }
            if let Some(most) = gpu_wait_watch() {
                watch_gpu_waits(oldest, most);
            }
            if !expired.is_empty() && !wait_release() {
                let lines: Vec<String> = expired
                    .iter()
                    .zip(&producers)
                    .map(|((va, val, _, wq, wty, pq, pty, rec, sub, done), producer)| {
                        format!(
                            "GPU queue.Wait at {va:#x} >= {val:#x} did not complete within {} ms (waiter=q{}/ty{wty}, prod=q{:?}/ty{pty:?}, recorded={rec:#x}, submitted={sub:#x}, completed={done:#x}); producer {producer}",
                            gpu_wait_watchdog_timeout().as_millis(),
                            queue_id(*wq),
                            pq.map(queue_id)
                        )
                    })
                    .collect();
                hard_stop(lines.join(" | "));
            }
            for (fence, val) in rewinds {
                unsafe {
                    let _ = fence.Signal(val);
                }
            }
            let had_expired = !expired.is_empty();
            static EXPIRED: AtomicU64 = AtomicU64::new(0);
            for ((va, val, fence, waiter_q, waiter_ty, prod_q, prod_ty, rec, sub, done), producer) in expired.into_iter().zip(producers) {
                unsafe {
                    let _ = fence.Signal(val);
                }
                let wq = queue_id(waiter_q);
                let pq = prod_q.map(queue_id);
                if let Some(n) = cpu_release(&EXPIRED) {
                    crate::xlog_warn!(
                        "D3D12.X: GPU queue.Wait at {va:#x} >= {val:#x} did not complete within {} ms (waiter=q{wq}/ty{waiter_ty}, prod=q{pq:?}/ty{prod_ty:?}, recorded={rec:#x}, submitted={sub:#x}, completed={done:#x}) – released on CPU to prevent GPU hang (#{n} here); producer {producer}",
                        gpu_wait_watchdog_timeout().as_millis()
                    );
                }
            }
            if had_expired {
                crate::xlog_warn!("D3D12.X: pending title ID3D12Fences during GPU wait timeout: {}", super::fence_x::pending_summary());
            }
        });
        pair
    });
    pair.1.notify_one();
}

/// What the queue meant to signal a timed-out GPU wait was doing: where that Signal sits among
/// its submissions, and the GPU waits it had armed itself (work after them waits for them).
fn producer_state(graph: &SyncGraph, prod: Option<usize>, signal_seq: Option<u64>) -> String {
    let Some(q) = prod else { return "none: no queue was given the Signal (no list writing it reached ExecuteCommandLists / KickoffX)".into() };
    let issued = graph.queue_seq.get(&q).copied().unwrap_or(0);
    let mut waits: Vec<(u64, String)> = Vec::new();
    for (&va, c) in &graph.cells {
        for aw in c.armed_waits.iter().filter(|aw| aw.waiting_queue == q) {
            waits.push((
                aw.wait_seq,
                format!("seq {} GPU Wait {va:#x} >= {:#x} for {} ms (recorded {:#x}, submitted {:#x} by q{:?})", aw.wait_seq, aw.val, aw.armed_at.elapsed().as_millis(), c.recorded_max, c.submitted_max, c.submitted_queue.map(queue_id)),
            ));
        }
    }
    for (&f, t) in &graph.title_fences {
        for &(val, _, seq) in t.waits.iter().filter(|w| w.1 == q) {
            waits.push((seq, format!("seq {seq} title fence {f:#x} Wait >= {val:#x}")));
        }
    }
    waits.sort_by_key(|w| w.0);
    let at = signal_seq.map_or_else(|| format!("its Signal not among its pending GPU signals ({issued} submissions)"), |s| format!("its Signal is submission seq {s} of {issued}"));
    if waits.is_empty() {
        return format!("q{}: {at}; no GPU wait of its own armed – the work before the Signal did not finish on the GPU", queue_id(q));
    }
    let n = waits.len();
    format!("q{}: {at}; its armed GPU waits ({n}): [{}]", queue_id(q), waits.into_iter().take(12).map(|w| w.1).collect::<Vec<_>>().join("; "))
}

unsafe fn queue_device_and_type(queue: *mut c_void) -> Option<(ID3D12Device, usize, D3D12_COMMAND_LIST_TYPE)> {
    if queue.is_null() {
        return None;
    }
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    let ty = q.GetDesc().Type;
    let mut dev: Option<ID3D12Device> = None;
    q.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let raw = dev.as_raw() as usize;
    Some((dev, raw, ty))
}

unsafe fn ensure_cell_fence(va: u64, dev: &ID3D12Device, dev_raw: usize) -> Option<ID3D12Fence> {
    {
        let g = SYNC_GRAPH.lock().unwrap();
        if let Some(cell) = g.as_ref().and_then(|gr| gr.cells.get(&va)) {
            if cell.non_monotonic {
                return None;
            }
            if let Some(ref f) = cell.fence {
                return (cell.device == dev_raw).then(|| f.clone());
            }
        }
    }
    let created: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let mut g = SYNC_GRAPH.lock().unwrap();
    let cell = g.get_or_insert_with(SyncGraph::default).cells.entry(va).or_default();
    if cell.non_monotonic {
        return None;
    }
    if let Some(ref existing) = cell.fence {
        return (cell.device == dev_raw).then(|| existing.clone());
    }
    cell.fence = Some(created.clone());
    cell.device = dev_raw;
    Some(created)
}

fn note_write64_recorded(list: *mut c_void, va: u64, value: u64) {
    if value == 0 || !va.is_multiple_of(8) {
        return;
    }
    let (lty, last_q) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        s.push_write64_signal(va, value);
        (s.list_type, s.last_queue)
    };
    let rescue: Option<(ID3D12Fence, u64, u64)> = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let cell = g.get_or_insert_with(SyncGraph::default).cells.entry(va).or_default();
        if cell.non_monotonic {
            return;
        }
        // Strictly decreasing write (`value < submitted_max`) marks this address non-monotonic.
        // Duplicate same-value writes from parallel command-list recording threads remain monotonic.
        if value < cell.submitted_max {
            let max_armed = cell.armed_waits.iter().map(|aw| aw.val).max().unwrap_or(0);
            let removed = cell.armed_waits.len();
            if removed > 0 {
                cell.armed_waits.clear();
                ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
            }
            cell.submitted_signals.clear();
            cell.non_monotonic = true;
            let prev_sub = cell.submitted_max;
            let old_fence = cell.fence.take();
            old_fence.filter(|_| max_armed > 0).map(|f| (f, max_armed, prev_sub))
        } else {
            if value > cell.recorded_max {
                cell.recorded_max = value;
                if lty.is_some() {
                    cell.recorded_list_type = lty;
                }
                if last_q.is_some() {
                    cell.recorded_last_queue = last_q;
                }
            }
            None
        }
    };
    // Rescue any armed GPU waits BEFORE dropping the fence so no vkd3d_queue thread can be orphaned.
    if let Some((old_fence, max_armed, prev_sub)) = rescue {
        unsafe {
            let _ = old_fence.Signal(max_armed);
        }
        static NON_MONO_WARN: AtomicU64 = AtomicU64::new(0);
        if let Some(n) = cpu_release(&NON_MONO_WARN) {
            crate::xlog_warn!(
                "D3D12.X: Write64 at {va:#x} became non-monotonic ({value:#x} < submitted {prev_sub:#x}) – released armed GPU waits up to {max_armed:#x} on CPU (#{n} here)"
            );
        }
    }
}

unsafe fn signal_write64_pairs(queue: *mut c_void, pairs: &[(u64, u64)]) {
    if queue.is_null() || pairs.is_empty() {
        return;
    }
    let Some((dev, dev_raw, q_ty)) = queue_device_and_type(queue) else { return };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let sig_q = queue as usize;
    // Coalesce per-VA max value within this batch.
    let mut max_by_va: Vec<(u64, u64)> = Vec::with_capacity(pairs.len());
    for &(va, val) in pairs {
        if val == 0 {
            continue;
        }
        if let Some(entry) = max_by_va.iter_mut().find(|e| e.0 == va) {
            entry.1 = entry.1.max(val);
        } else {
            max_by_va.push((va, val));
        }
    }
    for (va, val) in max_by_va {
        let skip = {
            let g = SYNC_GRAPH.lock().unwrap();
            g.as_ref()
                .and_then(|gr| gr.cells.get(&va))
                .is_some_and(|c| c.non_monotonic || c.submitted_max >= val)
        };
        if skip {
            continue;
        }
        let Some(fence) = ensure_cell_fence(va, &dev, dev_raw) else { continue };
        enum SignalAction {
            Skip,
            GpuSignal(u64),
            CpuCycleBreak(Vec<(usize, bool)>),
        }
        let action = {
            let mut g = SYNC_GRAPH.lock().unwrap();
            let graph = g.get_or_insert_with(SyncGraph::default);
            let eligible = match graph.cells.get_mut(&va) {
                Some(cell) => {
                    let completed = prune_completed_waits(cell);
                    if cell.non_monotonic || cell.device != dev_raw || val <= cell.submitted_max {
                        None
                    } else {
                        Some(completed)
                    }
                }
                None => None,
            };
            match eligible {
                None => SignalAction::Skip,
                Some(completed) => {
                    if would_signal_close_cycle(graph, sig_q, va, val) {
                        let cell = graph.cells.get_mut(&va).unwrap();
                        let waiters: Vec<(usize, bool)> =
                            cell.armed_waits.iter().filter(|aw| aw.val <= val).map(|aw| (aw.waiting_queue, aw.after_write64)).collect();
                        cell.submitted_max = val;
                        cell.submitted_queue = None;
                        cell.cpu_signaled_max = cell.cpu_signaled_max.max(val);
                        if val > cell.recorded_max {
                            cell.recorded_max = val;
                        }
                        let before = cell.armed_waits.len();
                        cell.armed_waits.retain(|aw| aw.val > val);
                        let removed = before - cell.armed_waits.len();
                        if removed > 0 {
                            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                        }
                        cell.submitted_signals.retain(|s| s.val > val);
                        SignalAction::CpuCycleBreak(waiters)
                    } else {
                        let sig_seq = graph.next_seq(sig_q);
                        let cell = graph.cells.get_mut(&va).unwrap();
                        cell.submitted_max = val;
                        cell.submitted_queue = Some(sig_q);
                        if val > cell.recorded_max {
                            cell.recorded_max = val;
                        }
                        if completed < val && cell.cpu_signaled_max < val {
                            cell.submitted_signals.push(SubmittedSignal { val, queue: sig_q, seq: sig_seq });
                            SignalAction::GpuSignal(sig_seq)
                        } else {
                            SignalAction::Skip
                        }
                    }
                }
            }
        };
        match action {
            SignalAction::Skip => {}
            SignalAction::GpuSignal(seq) => {
                if traced_queue(sig_q) {
                    trace_queue(|| format!("q{} seq {seq}: GPU Signal {va:#x} = {val:#x}", queue_id(sig_q)));
                }
                if let Err(e) = q.Signal(&fence, val) {
                    static FAILED: AtomicU64 = AtomicU64::new(0);
                    if let Some(n) = cpu_release(&FAILED) {
                        crate::xlog_warn!("D3D12.X: Write64 fence Signal({va:#x}, {val:#x}) failed ({e}) – signalling on CPU (#{n} here)");
                    }
                    let _ = fence.Signal(val);
                }
            }
            SignalAction::CpuCycleBreak(waiters) => {
                let _ = fence.Signal(val);
                static BREAK_WARN: AtomicU64 = AtomicU64::new(0);
                if let Some(n) = cpu_release(&BREAK_WARN) {
                    let sq = queue_id(sig_q);
                    let mut wids: Vec<String> = waiters
                        .into_iter()
                        .map(|(w, aw)| if aw { format!("q{}(after_w64)", queue_id(w)) } else { format!("q{}", queue_id(w)) })
                        .collect();
                    wids.sort();
                    wids.dedup();
                    crate::xlog_warn!(
                        "D3D12.X: Write64 Signal at {va:#x} = {val:#x} on q{sq}/ty{} would close a GPU cycle (waiters=[{}]) – signalled on CPU (#{n} here)",
                        q_ty.0,
                        wids.join(",")
                    );
                }
            }
        }
    }
}

/// After `queue.ExecuteCommandLists(lists)`: signal per-address GPU fences for any
/// `Write64BitValue{Bottom,Top}OfPipeX` commands recorded on `lists`.
pub(super) unsafe fn signal_write64_fences(queue: *mut c_void, lists: &[*mut c_void]) {
    if WRITE64_SIGNAL_LISTS.load(Ordering::Acquire) == 0 || queue.is_null() || lists.is_empty() {
        return;
    }
    let pairs: Vec<(u64, u64)> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let mut out = Vec::new();
        for &l in lists {
            if let Some(s) = m.get(&(l as usize)) {
                out.extend_from_slice(&s.write64_signals);
            }
        }
        out
    };
    signal_write64_pairs(queue, &pairs);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuWaitOutcome {
    /// Satisfied on GPU (via `queue.Wait` or same-queue FIFO order) — no CPU wait needed.
    Handled,
    /// Skipping this wait avoids a same-batch self-deadlock.
    CycleSkip,
    /// Not eligible for GPU fence bridging — fall back to bounded CPU poll.
    FallbackCpu,
}

/// Attempt to satisfy a `Wait64BitValueX(GREATER_EQUAL, u64::MAX)` on the GPU via
/// `queue.Wait(&fence, w.value)` (or same-queue FIFO ordering) instead of blocking the
/// submission thread on the CPU.
///
/// Cycle breaking is performed with 100% precision at `Signal` time (`signal_write64_pairs` /
/// `break_cycles_before_title_signal`) rather than guessing producer queues from previous frames,
/// so rotating multi-queue schedules neither skip valid barriers nor form Vulkan timeline cycles.
unsafe fn try_gpu_wait(queue: *mut c_void, w: &PendingWait, list_writes: &[(u64, u64)]) -> GpuWaitOutcome {
    if queue.is_null()
        || w.width != 8
        || w.comparison != D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32
        || w.mask != u64::MAX
        || w.value == 0
    {
        return GpuWaitOutcome::FallbackCpu;
    }
    let Some((dev, dev_raw, q_ty)) = queue_device_and_type(queue) else { return GpuWaitOutcome::FallbackCpu };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return GpuWaitOutcome::FallbackCpu };
    let q_raw = queue as usize;

    // Fast path: check eligibility, CPU-released bypass, and same-queue FIFO before allocating a fence.
    let pre = {
        let g = SYNC_GRAPH.lock().unwrap();
        g.as_ref().and_then(|gr| gr.cells.get(&w.va)).map(|c| {
            (c.non_monotonic, c.recorded_max, c.submitted_max, c.submitted_queue, c.cpu_signaled_max)
        })
    };
    let Some((non_mono, rec_max, sub_max, sub_q, cpu_sig_max)) = pre else {
        return GpuWaitOutcome::FallbackCpu;
    };
    if non_mono || rec_max < w.value {
        return GpuWaitOutcome::FallbackCpu;
    }
    // Already released on CPU (cycle break or watchdog): never re-arm a GPU wait that could
    // observe a vkd3d-proton fence rewind.
    if cpu_sig_max >= w.value {
        return GpuWaitOutcome::Handled;
    }
    // Producer already submitted earlier on THIS EXACT queue: D3D12 FIFO executes it strictly
    // before this list without needing a CPU wait or redundant queue.Wait.
    if sub_max >= w.value && sub_q == Some(q_raw) {
        return GpuWaitOutcome::Handled;
    }
    // Same-batch self-wait: this very submission writes `(w.va, >= w.value)`, so hoisting a GPU
    // Wait ahead of it on the same queue would immediately self-deadlock.
    if sub_max < w.value && list_writes.iter().any(|&(wva, wv)| wva == w.va && wv >= w.value) {
        static SELF_WARN: AtomicU64 = AtomicU64::new(0);
        if let Some(n) = cpu_release(&SELF_WARN) {
            crate::xlog_warn!(
                "D3D12.X: Wait64BitValueX at {:#x} >= {:#x} skipped to avoid same-batch self-deadlock (q{}/ty{}, #{n} here)",
                w.va,
                w.value,
                queue_id(q_raw),
                q_ty.0
            );
        }
        return GpuWaitOutcome::CycleSkip;
    }

    if super::single::host_of(queue).is_some() {
        // every title queue's work runs on this queue: a GPU wait for a write that comes after it
        // would never end. `single` holds a title queue's lists until the write is handed over;
        // what it cannot hold (KickoffX) waits here
        let t0 = Instant::now();
        while !cell_written(w.va, w.value) {
            if t0.elapsed() > Duration::from_secs(10) {
                hard_stop(format!("single queue: a list was handed over whose Wait64BitValueX at {:#x} >= {:#x} no queue's lists have written after 10 s", w.va, w.value));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        return GpuWaitOutcome::Handled;
    }

    let Some(fence) = ensure_cell_fence(w.va, &dev, dev_raw) else { return GpuWaitOutcome::FallbackCpu };

    let (unsubmitted, final_rec_ty, final_prod_q, wait_seq) = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let Some(graph) = g.as_mut() else { return GpuWaitOutcome::FallbackCpu };
        let pre_ok = match graph.cells.get_mut(&w.va) {
            Some(cell) => {
                if cell.non_monotonic || cell.device != dev_raw || cell.recorded_max < w.value {
                    return GpuWaitOutcome::FallbackCpu;
                }
                if cell.cpu_signaled_max >= w.value {
                    return GpuWaitOutcome::Handled;
                }
                let completed = prune_completed_waits(cell);
                if completed >= w.value {
                    // Fence already reached `w.value`: trust it if CPU memory agrees (not reset).
                    return if w.is_satisfied() { GpuWaitOutcome::Handled } else { GpuWaitOutcome::FallbackCpu };
                }
                if cell.submitted_max >= w.value && cell.submitted_queue == Some(q_raw) {
                    return GpuWaitOutcome::Handled;
                }
                (cell.submitted_max < w.value, cell.recorded_list_type, cell.submitted_queue.or(cell.recorded_last_queue))
            }
            None => return GpuWaitOutcome::FallbackCpu,
        };
        let wait_seq = graph.next_seq(q_raw);
        let cell = graph.cells.get_mut(&w.va).unwrap();
        cell.armed_waits.push(ArmedGpuWait {
            val: w.value,
            waiting_queue: q_raw,
            waiting_queue_ty: q_ty.0,
            wait_seq,
            armed_at: Instant::now(),
            after_write64: w.after_write64,
        });
        ARMED_GPU_WAITS.fetch_add(1, Ordering::AcqRel);
        (pre_ok.0, pre_ok.1, pre_ok.2, wait_seq)
    };

    if let Err(e) = q.Wait(&fence, w.value) {
        let mut g = SYNC_GRAPH.lock().unwrap();
        if let Some(cell) = g.as_mut().and_then(|gr| gr.cells.get_mut(&w.va)) {
            if let Some(pos) = cell.armed_waits.iter().rposition(|aw| aw.val == w.value && aw.waiting_queue == q_raw && aw.wait_seq == wait_seq) {
                cell.armed_waits.remove(pos);
                ARMED_GPU_WAITS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        crate::xlog_warn!("D3D12.X: Wait64 GPU queue.Wait({:#x}, {:#x}) failed ({e}) – falling back to CPU wait", w.va, w.value);
        return GpuWaitOutcome::FallbackCpu;
    }

    if super::gpu_timing::enabled() && !real::is_adopted(queue) {
        real::note_profiled_gpu_wait(queue, fence.as_raw() as usize, w.value);
    }
    wake_sync_watchdog();
    if traced_queue(q_raw) || final_prod_q.is_some_and(traced_queue) {
        trace_queue(|| format!("q{} seq {wait_seq}: GPU Wait {:#x} >= {:#x} (producer q{:?}{})", queue_id(q_raw), w.va, w.value, final_prod_q.map(queue_id), if unsubmitted { ", its Signal not submitted yet" } else { "" }));
    }
    if unsubmitted {
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if once(&LOGGED, 8) {
            crate::xlog_info!(
                "D3D12.X: Wait64BitValueX at {:#x} >= {:#x} bridged to GPU queue.Wait (waiter=q{}/ty{}, prod=q{:?}/ty{:?}; wait-before-signal, after_write64={})",
                w.va,
                w.value,
                queue_id(q_raw),
                q_ty.0,
                final_prod_q.map(queue_id),
                final_rec_ty.map(|t| t.0),
                w.after_write64
            );
        }
    }
    GpuWaitOutcome::Handled
}

unsafe fn wait_all_on(queue: *mut c_void, waits: &[PendingWait], list_writes: &[(u64, u64)]) {
    static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
    static RING_DUMPED: AtomicU64 = AtomicU64::new(0);
    let timeout = wait_timeout();
    let q_ty = queue_device_and_type(queue).map_or(-1, |(_, _, t)| t.0);
    for w in waits {
        if w.is_satisfied() {
            continue;
        }
        match try_gpu_wait(queue, w, list_writes) {
            GpuWaitOutcome::Handled | GpuWaitOutcome::CycleSkip => continue,
            GpuWaitOutcome::FallbackCpu => {}
        }
        if let Some((_, _, pso)) = super::indirect::unwritten_at(w.va) {
            // its writer was a skipped dispatch (`XCL_SKIP_FAILED_DISPATCHES`): nothing will write it
            static SKIPPED: AtomicU64 = AtomicU64::new(0);
            if once(&SKIPPED, 16) {
                crate::xlog_warn!("D3D12.X: Wait{}BitValueX at {:#x} released at once: the dispatch that writes it (pipeline {pso:#x}) was skipped", w.width * 8, w.va);
            }
            continue;
        }
        let t0 = Instant::now();
        let held = {
            let live = crate::live_stall::enabled().then(|| crate::live_stall::begin(
                crate::live_stall::Stage::GpuMemoryWait, 0, 0, Some(w.value), diagnostic_queue_id(queue as usize),
            ));
            let held = w.wait_until_satisfied(timeout);
            drop(live);
            held
        };
        super::wait_trace::memory(if w.width == 4 { "Wait32BitValueX" } else { "Wait64BitValueX" }, w.va as usize, w.comparison, w.value, t0.elapsed(), held, w.read_value());
        if !held && !wait_release() {
            hard_stop(format!(
                "Wait{}BitValueX at {:#x} on queue type {q_ty} timed out after {} ms (op {}, want {:#x}, mask {:#x}; memory holds {:x?})",
                w.width * 8,
                w.va,
                timeout.as_millis(),
                w.comparison,
                w.value,
                w.mask,
                w.read_value()
            ));
        }
        if let Some(n) = (!held).then(|| cpu_release(&TIMEOUTS)).flatten() {
            let cell_diag = {
                let g = SYNC_GRAPH.lock().unwrap();
                g.as_ref().and_then(|gr| gr.cells.get(&w.va)).map(|c| (c.recorded_max, c.submitted_max, c.non_monotonic, c.recorded_list_type.map(|t| t.0)))
            };
            crate::xlog_warn!(
                "D3D12.X: Wait{}BitValueX timed out after {} ms at {:#x} (op {}, want {:#x}, mask {:#x}; memory holds {:x?}; queue_type={}, after_write64={}, sync={cell_diag:x?}) – the submission goes on without it (#{n} here)",
                w.width * 8,
                timeout.as_millis(),
                w.va,
                w.comparison,
                w.value,
                w.mask,
                w.read_value(),
                q_ty,
                w.after_write64
            );
            if once(&RING_DUMPED, 2) {
                let (total, entries) = WRITE_TRACE.lock().unwrap().snapshot();
                let summary: Vec<String> = entries
                    .iter()
                    .rev()
                    .take(16)
                    .map(|e| format!("{}({:#x}={:#x},f={:#x},ty={})", e.what, e.va, e.value, e.flags, e.list_type))
                    .collect();
                crate::xlog_warn!("D3D12.X: recent Write32/64 calls ({total} total): [{}]", summary.join(", "));
            }
        }
    }
}

/// Copy any GPU memory writes into sparse VRAM windows recorded by `lists` back into their
/// CPU view after `queue.ExecuteCommandLists` so CPU reads and `Wait32/64BitValueX` see them.
pub(super) unsafe fn readback_sparse_writes(queue: *mut c_void, lists: &[*mut c_void]) {
    if SPARSE_WRITE_LISTS.load(Ordering::Acquire) == 0 || queue.is_null() || lists.is_empty() {
        return;
    }
    let ranges: Vec<(usize, usize)> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let mut out = Vec::new();
        for &l in lists {
            if let Some(s) = m.get(&(l as usize)) {
                out.extend_from_slice(&s.sparse_writes);
            }
        }
        out
    };
    if ranges.is_empty() {
        return;
    }
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        super::sparse::readback_after_execute(q, &ranges);
    }
}

/// After `lists` went to `queue`: their writes into memory the GPU does not see (`cpu_writes`).
pub(super) fn note_dispatch_timing(list: usize, readback: usize, write: super::cpu_writes::Write) -> bool {
    let mut g = LISTS.lock().unwrap();
    let Some(s) = g.as_mut().and_then(|m| m.get_mut(&list)) else { return false };
    s.cpu_writes.push(write);
    drop(g);
    super::residency::note_list_resources(list, [readback]);
    true
}

pub(super) unsafe fn cpu_writes_after(queue: *mut c_void, lists: &[*mut c_void]) {
    let writes: Vec<super::cpu_writes::Write> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        lists.iter().filter_map(|&l| m.get(&(l as usize))).flat_map(|s| s.cpu_writes.iter().cloned()).collect()
    };
    super::cpu_writes::after(queue, writes);
}

/// Submit `lists` through `submit`, splitting the batch before any list whose `Wait32/64BitValueX`
/// condition is not yet satisfied so earlier lists in the same batch can run on the GPU first.
/// When `queue` is non-null, monotonic 64-bit waits backed by a recorded `Write64BitValue*` are
/// bridged to GPU-side `queue.Wait` without blocking the submitter thread on the CPU.
pub(super) unsafe fn execute_with_waits_on(queue: *mut c_void, lists: &[*mut c_void], mut submit: impl FnMut(&[*mut c_void])) {
    if !queue.is_null() && traced_queue(queue as usize) {
        trace_queue(|| format!("q{}: ExecuteCommandLists of {} lists", queue_id(queue as usize), lists.len()));
    }
    if lists.is_empty() || PENDING_WAIT_LISTS.load(Ordering::Acquire) == 0 {
        submit(lists);
        return;
    }
    let per_list: Vec<(Vec<PendingWait>, Vec<(u64, u64)>)> = {
        let g = LISTS.lock().unwrap();
        lists
            .iter()
            .map(|&l| {
                g.as_ref()
                    .and_then(|m| m.get(&(l as usize)))
                    .map(|s| (s.pending_waits.clone(), s.write64_signals.clone()))
                    .unwrap_or_default()
            })
            .collect()
    };
    let mut start = 0usize;
    for (i, (waits, list_writes)) in per_list.iter().enumerate() {
        if waits.is_empty() || waits.iter().all(|w| w.is_satisfied()) {
            continue;
        }
        if i > start {
            submit(&lists[start..i]);
        }
        wait_all_on(queue, waits, list_writes);
        start = i;
    }
    if start < lists.len() {
        submit(&lists[start..]);
    }
}

/// (`single`) The first of `lists` with a `Wait64BitValueX` for a value whose write no queue
/// has been given yet, with that wait's address and value: it cannot go to the one PC queue
/// before the write does. Only waits `try_gpu_wait` would put on the GPU count; the others are
/// waited for on the CPU as ever.
pub(super) unsafe fn first_unwritten_wait(lists: &[usize]) -> Option<(usize, u64, u64)> {
    if PENDING_WAIT_LISTS.load(Ordering::Acquire) == 0 {
        return None;
    }
    let per_list: Vec<(Vec<PendingWait>, Vec<(u64, u64)>)> = {
        let g = LISTS.lock().unwrap();
        lists.iter().map(|l| g.as_ref().and_then(|m| m.get(l)).map(|s| (s.pending_waits.clone(), s.write64_signals.clone())).unwrap_or_default()).collect()
    };
    let g = SYNC_GRAPH.lock().unwrap();
    let graph = g.as_ref()?;
    for (i, (waits, writes)) in per_list.iter().enumerate() {
        for w in waits {
            if w.width != 8 || w.comparison != D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32 || w.mask != u64::MAX || w.value == 0 || w.is_satisfied() {
                continue;
            }
            let Some(c) = graph.cells.get(&w.va) else { continue };
            if c.non_monotonic || c.recorded_max < w.value || c.cpu_signaled_max >= w.value || c.submitted_max >= w.value {
                continue;
            }
            // the list writes it itself (`try_gpu_wait` skips such a wait)
            if writes.iter().any(|&(va, v)| va == w.va && v >= w.value) {
                continue;
            }
            return Some((i, w.va, w.value));
        }
    }
    None
}

/// (`single`) A queue has been given a write of `value` or more at `va`, or memory holds it.
pub(super) unsafe fn cell_written(va: u64, value: u64) -> bool {
    let given = SYNC_GRAPH.lock().unwrap().as_ref().and_then(|g| g.cells.get(&va)).is_some_and(|c| c.submitted_max >= value || c.cpu_signaled_max >= value);
    // (memory: as a copy, `memmap::title_words`)
    given || crate::memmap::title_words(va as usize, 2).is_some_and(|w| (w[0] as u64 | (w[1] as u64) << 32) >= value)
}

/// (`single`) One of `lists` writes `value` or more at `va`.
pub(super) fn lists_write(lists: &[usize], va: u64, value: u64) -> bool {
    let g = LISTS.lock().unwrap();
    lists.iter().filter_map(|l| g.as_ref()?.get(l)).any(|s| s.write64_signals.iter().any(|&(a, v)| a == va && v >= value))
}

#[cfg(test)]
pub(super) unsafe fn execute_with_waits(lists: &[*mut c_void], submit: impl FnMut(&[*mut c_void])) {
    execute_with_waits_on(std::ptr::null_mut(), lists, submit);
}

// ---------------------------------------------------------------------------------------
// memory copy / fill / immediate write / timestamp
// ---------------------------------------------------------------------------------------

/// command list #60 CopyMemoryToMemoryX(destination, source, size, flags)
pub(super) unsafe extern "system" fn copy_memory_x(this: *mut c_void, dst: u64, src: u64, size: u64, _flags: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    const W: &str = "CopyMemoryToMemoryX";
    super::dsv_trace::memory(W, Some(dst as usize), Some(src as usize), size);
    if super::bind::indirect_world_trace() {
        super::bind::indirect_world_event(this as usize, format_args!("{W} source {src:#x} destination {dst:#x}+{size:#x}"));
    }
    if size == 0 || !size.is_multiple_of(4) {
        return warn(&N, W, format_args!("size {size:#x} is not a non-zero multiple of 4"));
    }
    let (Some((d, doff)), Some((s, soff))) = (real::uma_window_buffer(dst as usize, size as usize), real::uma_window_buffer(src as usize, size as usize)) else {
        note_gpu_write("CopyMemoryToMemoryX", dst as usize, size);
        let (d, s, n) = (dst as usize, src as usize, size as usize);
        if super::cpu_writes::writable(d, n) && super::cpu_writes::readable(s, n) && defer_to_cpu(this, super::cpu_writes::Write::Copy { dst: d, src: s, bytes: n }) {
            return;
        }
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) is not in unified memory"));
    };
    note_gpu_write("CopyMemoryToMemoryX", dst as usize, size);
    if d == s && dst < src + size && src < dst + size {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) overlap"));
    }
    note_write(this as usize, dst, size);
    if d == s {
        // one window: the buffer is the source and the destination (the ranges do not overlap)
        as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
        return;
    }
    transition(this, s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
    transition(this, s, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

/// `w` done by the CPU once the GPU has finished the list's work before it (`cpu_writes`).
fn defer_to_cpu(list: *mut c_void, w: super::cpu_writes::Write) -> bool {
    let mut g = LISTS.lock().unwrap();
    let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else { return false };
    s.cpu_writes.push(w);
    true
}

unsafe fn device_of(list: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    dev
}

type FillUploads = HashMap<usize, HashMap<[u8; 16], ID3D12Resource>>;
static FILL_UPLOADS: Mutex<Option<FillUploads>> = Mutex::new(None);

/// The command allocator the list `list` records on (0: not known, or the lists' table is taken now –
/// the caller may be the one that holds it).
pub(super) fn allocator_of(list: usize) -> usize {
    LISTS.try_lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(&list)).map(|s| s.allocator)).unwrap_or(0)
}

/// Whether `list` is a bundle, by the driver's word or the title's (a title's bundle is not the
/// driver's with `single`; false when the lists' table is taken now and the driver says no).
pub(super) fn is_bundle(list: usize) -> bool {
    if list == 0 {
        return false;
    }
    let get: unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE = unsafe { std::mem::transmute(real::original_or_native(list as *mut c_void, PC_GET_TYPE).0) };
    let host = unsafe { get(list as *mut c_void) };
    host == D3D12_COMMAND_LIST_TYPE_BUNDLE
        || LISTS.try_lock().ok().is_some_and(|g| g.as_ref().and_then(|m| m.get(&list)).is_some_and(|s| s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)))
}

pub(super) fn on_allocator_reset(allocator: usize) {
    super::shader_code::allocator_reset(allocator);
    let uploads = FILL_UPLOADS.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator));
    drop(uploads);
    // the GPU is done with every list recorded on `allocator`, the put-off calls they made too
    let companions: Vec<_> = COMPANIONS.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.iter().filter(|(k, _)| k.0 == allocator).map(|(_, c)| c.clone()).collect());
    for c in companions {
        let _ = unsafe { c.lock().unwrap().Reset() };
    }
}

unsafe fn pattern_buffer(list: *mut c_void, pat: [u8; 16]) -> Option<usize> {
    let allocator = LISTS.lock().unwrap().as_ref()?.get(&(list as usize))?.allocator;
    if allocator == 0 {
        return None;
    }
    if let Some(raw) = FILL_UPLOADS.lock().unwrap().as_ref()
        .and_then(|m| m.get(&allocator)).and_then(|m| m.get(&pat)).map(|b| b.as_raw() as usize)
    {
        return Some(raw);
    }
    let dev = device_of(list)?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: FILL_CHUNK,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut b,
    )
    .ok()?;
    let b = b?;
    let mut p: *mut c_void = std::ptr::null_mut();
    b.Map(0, None, Some(&mut p)).ok()?;
    for chunk in std::slice::from_raw_parts_mut(p as *mut u8, FILL_CHUNK as usize).chunks_exact_mut(16) {
        chunk.copy_from_slice(&pat);
    }
    b.Unmap(0, None);
    let raw = FILL_UPLOADS.lock().unwrap().get_or_insert_with(HashMap::new)
        .entry(allocator).or_default().entry(pat).or_insert_with(|| b.clone()).as_raw() as usize;
    Some(raw)
}

unsafe fn fill(list: *mut c_void, va: u64, size: u64, pat: [u8; 16], unit: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    if size == 0 || !size.is_multiple_of(unit) || size > MAX_FILL {
        return warn(&N, what, format_args!("size {size:#x} (a non-zero multiple of {unit} up to {MAX_FILL:#x} is filled)"));
    }
    let Some((d, off)) = real::uma_window_buffer(va as usize, size as usize) else {
        if super::cpu_writes::writable(va as usize, size as usize) && defer_to_cpu(list, super::cpu_writes::Write::Fill(va as usize, pat, size as usize)) {
            return;
        }
        return warn(&N, what, format_args!("{va:#x} ({size:#x} bytes) is not in unified memory"));
    };
    if super::bind::indirect_world_trace() {
        super::bind::indirect_world_event(list as usize, format_args!("{what} destination {va:#x}+{size:#x} pattern {pat:02x?}"));
    }
    let Some(src) = pattern_buffer(list, pat) else {
        return warn(&N, what, format_args!("no pattern buffer for {pat:x?}"));
    };
    super::residency::note_list_resources(list as usize, [src]);
    note_write(list as usize, va, size);
    as_copy_dest(list, d, || {
        for offset in (0..size).step_by(FILL_CHUNK as usize) {
            copy_buffer_region(list, d, off + offset, src, 0, (size - offset).min(FILL_CHUNK));
        }
    });
}

fn pattern(bytes: &[u8]) -> [u8; 16] {
    let mut p = [0u8; 16];
    for (i, b) in p.iter_mut().enumerate() {
        *b = bytes[i % bytes.len()];
    }
    p
}

/// command list #61 FillMemoryWith32BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill32_x(this: *mut c_void, va: u64, size: u64, value: u32) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 4, "FillMemoryWith32BitValueX");
}

/// command list #62 FillMemoryWith64BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill64_x(this: *mut c_void, va: u64, size: u64, value: u64) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 8, "FillMemoryWith64BitValueX");
}

/// command list #63 FillMemoryWith128BitValueX(address, size, value*)
pub(super) unsafe extern "system" fn fill128_x(this: *mut c_void, va: u64, size: u64, value: *const [u8; 16]) {
    if !value.is_null() {
        fill(this, va, size, value.read_unaligned(), 16, "FillMemoryWith128BitValueX");
    }
}

/// Values written by the GPU at `va` (WriteBufferImmediate: MARKER_IN = when the command is
/// reached, MARKER_OUT = when the work before it is done).
unsafe fn write(list: *mut c_void, va: u64, words: &[u32], mode: D3D12_WRITEBUFFERIMMEDIATE_MODE, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = 4 * words.len() as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    let (Some((d, _)), Some(at)) = (real::uma_window_buffer(va as usize, n as usize), real::uma_va_of(va as usize)) else {
        // the title's own memory: written by the CPU once the GPU is there
        if super::cpu_writes::writable(va as usize, n as usize) && defer_to_cpu(list, super::cpu_writes::Write::Immediate(va as usize, words.to_vec())) {
            return;
        }
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    };
    if super::bind::indirect_world_trace() {
        super::bind::indirect_world_event(list as usize, format_args!("{what} destination {va:#x}+{n:#x} words {:08x?} mode {}", &words[..words.len().min(8)], mode.0));
    }
    note_write(list as usize, va, n);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = words.iter().enumerate().map(|(i, &w)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + 4 * i as u64, Value: w }).collect();
    let modes = vec![mode; words.len()];
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    as_copy_dest(list, d, || pc::<F>(list, PC_WRITE_BUFFER_IMMEDIATE)(list, params.len() as u32, params.as_ptr(), modes.as_ptr()));
}

/// timestamp queries (one heap per device, used round-robin) and a lazily created READBACK
/// buffer for timestamps targeting memory outside unified memory
struct TimestampPool {
    heap: ID3D12QueryHeap,
    /// `(buffer, mapped_base)` of `TIMESTAMP_SLOTS * 8` bytes in a READBACK heap
    readback: Option<(ID3D12Resource, usize)>,
}

// SAFETY: D3D12 query heaps and resources are free-threaded COM objects; the persistent
// readback mapping lives as long as `readback` is held.
unsafe impl Send for TimestampPool {}

static TIMESTAMPS: Mutex<Option<HashMap<usize, TimestampPool>>> = Mutex::new(None);
const TIMESTAMP_SLOTS: u32 = 4096;
static NEXT_TIMESTAMP: AtomicU32 = AtomicU32::new(0);

unsafe fn ensure_timestamp_readback(dev: &ID3D12Device, pool: &mut TimestampPool) -> Option<(ID3D12Resource, usize)> {
    if let Some((ref buf, base)) = pool.readback {
        return Some((buf.clone(), base));
    }
    let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let rd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: TIMESTAMP_SLOTS as u64 * 8,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).ok()?;
    let rb = rb?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut ptr)).ok()?;
    if ptr.is_null() {
        return None;
    }
    let base = ptr as usize;
    pool.readback = Some((rb.clone(), base));
    Some((rb, base))
}

/// The GPU timestamp (the PC queue's ticks) written at `va` when the work before it is done.
unsafe fn timestamp(list: *mut c_void, va: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, what, format_args!("a timestamp on a type-{} list", ty.0));
    }
    if !va.is_multiple_of(8) {
        return warn(&N, what, format_args!("{va:#x} is not 8-byte aligned"));
    }
    let uma_dst = real::uma_window_buffer(va as usize, 8);
    if uma_dst.is_none() && !super::cpu_writes::writable(va as usize, 8) {
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    }
    let Some(dev) = device_of(list) else { return };
    let (heap, readback) = {
        let mut m = TIMESTAMPS.lock().unwrap();
        let m = m.get_or_insert_with(HashMap::new);
        let pool = match m.entry(dev.as_raw() as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let mut h: Option<ID3D12QueryHeap> = None;
                let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: TIMESTAMP_SLOTS, NodeMask: 0 };
                match dev.CreateQueryHeap(&desc, &mut h) {
                    Ok(()) if h.is_some() => slot.insert(TimestampPool { heap: h.unwrap(), readback: None }),
                    _ => return warn(&N, what, format_args!("no timestamp query heap")),
                }
            }
        };
        let rb = if uma_dst.is_none() {
            match ensure_timestamp_readback(&dev, pool) {
                Some(rb) => Some(rb),
                None => return warn(&N, what, format_args!("no timestamp readback buffer")),
            }
        } else {
            None
        };
        (pool.heap.as_raw() as usize, rb)
    };
    let i = NEXT_TIMESTAMP.fetch_add(1, Ordering::Relaxed) % TIMESTAMP_SLOTS;
    type Resolve = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    if let Some((d, off)) = uma_dst {
        note_write(list as usize, va, 8);
        pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
        as_copy_dest(list, d, || super::gpu_timing::class(list, super::gpu_timing::Kind::Transfer, || pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, d, off)));
    } else if let Some((rb, base)) = readback {
        // Queue the CPU copy only if the list is tracked; otherwise avoid emitting an unused resolve.
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
        };
        let rb_raw = rb.as_raw() as usize;
        let off = i as u64 * 8;
        s.cpu_writes.push(super::cpu_writes::Write::ReadbackU64 { va: va as usize, src: base + off as usize, _buf: rb });
        drop(g);
        super::residency::note_list_resources(list as usize, [rb_raw]);
        pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
        super::gpu_timing::class(list, super::gpu_timing::Kind::Transfer, || pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, rb_raw, off));
    }
}

/// command list #69 Write32BitValueBottomOfPipeX(address, value, flush, flags)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn write32_bottom_x(this: *mut c_void, va: u64, value: u32, _flush: u32, flags: u32) {
    record_write_trace(this, va, value as u64, flags, "W32B");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfPipeX");
}

/// command list #70 Write64BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write64_bottom_x(this: *mut c_void, va: u64, value: u64, _flush: u32, flags: u32) {
    const W: &str = "Write64BitValueBottomOfPipeX";
    record_write_trace(this, va, value, flags, "W64B");
    if flags & BOTTOM_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    note_write64_recorded(this, va, value);
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, W);
}

/// Deferred mid-list split for `Wait32/64BitValueX` recorded after a `Write64BitValue*` in the
/// same command list: closes `list` (preserving its earlier `Write64` signals on segment N) and
/// opens/reuses a sibling PC command list on the same `ID3D12CommandAllocator` as segment N+1,
/// carrying all bound D3D12 + XCL root-argument state across the split.
///
/// At `ExecuteCommandLists`, `title_list::push_segments` expands the wrapper into `[seg_0, seg_1, ...]`
/// and `execute_with_waits_on` submits `seg_0` + emits `queue.Signal` for its `Write64`s BEFORE
/// emitting `queue.Wait` for `seg_1`'s `Wait64`, preserving exact Xbox chronological order with
/// zero artificial GPU cycles.
unsafe fn try_split_for_wait(list: *mut c_void) -> Option<*mut c_void> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("XCL_SPLIT_MIDLIST_WAIT").map_or(true, |v| v.trim() != "0")) {
        return None;
    }
    let wrapper = super::title_list::wrapper_of(list)?;
    if !super::title_list::can_split(wrapper) {
        return None;
    }
    let ty = list_type(list);
    let next_pc = split(list, wrapper, None)?;
    static SPLITS: AtomicU64 = AtomicU64::new(0);
    let n = SPLITS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || (n.is_power_of_two() && n <= 256) {
        crate::xlog_info!("D3D12.X: split command list (type {}) around mid-list Wait after Write64 (#{n})", ty.0);
    }
    Some(next_pc)
}

/// Ends `list`'s segment of `wrapper`'s recording and opens the next, with all of `list`'s
/// state; with `put_off` (a call put off to the submit and the closed segment it is recorded in
/// there, `defer`), that segment comes between the two.
unsafe fn split(list: *mut c_void, wrapper: *mut c_void, put_off: Option<(Deferred, *mut c_void)>) -> Option<*mut c_void> {
    let ty = list_type(list);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return None;
    }
    let (alloc, last_queue, bound_pso, depth_bias, index_buffer, depth_bounds, predication, draw_predicate, pred_buf, pred_in_state, active_query, ended_early, written_ranges, bindings) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.as_mut()?.get_mut(&(list as usize))?;
        if s.allocator == 0 {
            return None;
        }
        let active_query = s.occlusion.end_active_early();
        let ended_early = std::mem::take(&mut s.occlusion.ended_early);
        let pred_in_state = std::mem::replace(&mut s.pred_in_state, false);
        (
            s.allocator,
            s.last_queue,
            s.bound_pso,
            s.depth_bias,
            s.index_buffer,
            s.depth_bounds,
            s.predication,
            s.draw_predicate,
            s.pred_buf.clone(),
            pred_in_state,
            active_query,
            ended_early,
            s.written_ranges.clone(),
            s.bindings.clone(),
        )
    };

    if let Some(((h, i), qty)) = active_query {
        pc::<QueryFn>(list, PC_END_QUERY)(list, h, qty, i);
    }
    if pred_in_state {
        if let Some(ref pb) = pred_buf {
            transition(list, pb.as_raw() as usize, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    super::depth_uav::before_close(list);
    super::gpu_timing::before_close(list);
    let ext = *(list as *const usize);
    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(list, PC_CLOSE)(list);
    real::preserve_vtable(list, ext);
    if close_hr < 0 {
        static CLOSE_ERR: AtomicU64 = AtomicU64::new(0);
        warn(&CLOSE_ERR, "Wait64 mid-list split", format_args!("Close failed ({close_hr:#x})"));
        return None;
    }
    super::depth_state::reset(list as usize);

    let init_pso = if ty == D3D12_COMMAND_LIST_TYPE_COPY {
        0
    } else {
        super::pso_async::resolve_for_bind(bound_pso)
    };
    super::shader_code::taken_on(alloc, init_pso);
    let dev = device_of(list)?;
    let dev_raw = dev.as_raw() as usize;

    let mut next_pc: Option<*mut c_void> = None;
    if let Some(spare) = super::title_list::pop_spare(wrapper) {
        let ext_spare = *(spare as *const usize);
        let reset_hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(spare, PC_RESET)(spare, alloc, init_pso);
        real::preserve_vtable(spare, ext_spare);
        if reset_hr >= 0 {
            super::physical_directory::recordings::ended(spare as usize);
            super::gpu_timing::opened(spare, bound_pso);
            super::material_gate_views::prepass::native_reset(spare as usize);
            next_pc = Some(spare);
        }
    }
    let next_pc = match next_pc {
        Some(p) => p,
        None => match real::create_raw_pc_command_list(dev.as_raw(), ty, alloc, init_pso) {
            Some(p) => p,
            None => {
                static CREATE_ERR: AtomicU64 = AtomicU64::new(0);
                warn(&CREATE_ERR, "Wait64 mid-list split", format_args!("CreateCommandList failed"));
                return None;
            }
        },
    };

    let old_next = {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let old = m.remove(&(next_pc as usize));
        if let Some(ref o) = old {
            uncount(o);
        }
        m.insert(
            next_pc as usize,
            ListState {
                allocator: alloc,
                list_type: Some(ty),
                last_queue,
                bound_pso,
                depth_bias,
                native_bias: DepthBias::ZERO,
                pso_dirty: false,
                using_fallback_pso: false,
                index_buffer,
                depth_bounds,
                occlusion: Occlusion { active: None, ended_early },
                written_ranges,
                pred_buf: pred_buf.clone(),
                pred_in_state: false,
                predication,
                draw_predicate: draw_predicate.filter(|_| predication.is_some()),
                bindings: bindings.clone(),
                ..Default::default()
            },
        );
        old
    };
    drop(old_next);
    super::title_textures::list_reset(next_pc as usize);
    super::depth_state::reset(next_pc as usize);
    super::residency::list_reset(next_pc as usize);
    super::depth_uav::list_reset(next_pc as usize);
    super::bind::inherit_for_split(list, next_pc, alloc);
    restore_bindings(next_pc, &bindings, ty);

    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT {
        restore_direct(next_pc, dev_raw, bound_pso, depth_bias, index_buffer, depth_bounds);
    }
    if ty != D3D12_COMMAND_LIST_TYPE_COPY {
        if let (Some(op), Some(ref pb)) = (predication, pred_buf) {
            let pb_raw = pb.as_raw() as usize;
            transition(next_pc, pb_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
            pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(next_pc, PC_SET_PREDICATION)(next_pc, pb_raw, 0, op);
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(next_pc as usize))) {
                s.pred_in_state = true;
            }
        }
    }

    if let Some((what, seg)) = put_off {
        let mut captured = bindings.clone();
        if what.draws() {
            snapshot_render_targets(&dev, &mut captured);
        }
        let old = {
            let mut g = LISTS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            let old = m.remove(&(seg as usize));
            if let Some(ref o) = old {
                uncount(o);
            }
            m.insert(
                seg as usize,
                ListState {
                    allocator: alloc,
                    list_type: Some(ty),
                    last_queue,
                    bound_pso,
                    depth_bias,
                    index_buffer,
                    depth_bounds,
                    bindings: captured,
                    deferred: Some(what),
                    ..Default::default()
                },
            );
            old
        };
        drop(old);
        super::title_textures::list_reset(seg as usize);
        super::depth_state::reset(seg as usize);
        super::residency::list_reset(seg as usize);
        super::depth_uav::list_reset(seg as usize);
        super::bind::inherit_for_split(list, seg, alloc);
        super::title_list::push_segment(wrapper, seg);
    }

    super::title_list::activate_segment(wrapper, list, next_pc);
    Some(next_pc)
}

/// The rest of a direct list's state a PC Reset clears, on `list` (a segment going on from
/// another): the depth bias for `bound_pso`, the index buffer, the depth bounds.
unsafe fn restore_direct(list: *mut c_void, dev_raw: usize, bound_pso: usize, depth_bias: Option<DepthBias>, index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>, depth_bounds: Option<(f32, f32)>) {
    let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
    let mut applied_native = false;
    if let Some((true, static_bias)) = info {
        let target = depth_bias.unwrap_or(static_bias);
        if target.is_zero() {
            applied_native = true;
        } else if try_native_depth_bias(list, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
                s.native_bias = target;
            }
            applied_native = true;
        }
    } else if bound_pso == 0 {
        if let Some(b) = depth_bias {
            if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(list, b) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
                    s.native_bias = b;
                }
                applied_native = true;
            }
        }
    }
    if depth_bias.is_some() && !applied_native {
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
            s.pso_dirty = true;
        }
        DEPTH_BIAS_USED.store(true, Ordering::Release);
    }
    if let Some(ref view) = index_buffer {
        set_index_buffer(list, Some(view));
    }
    if let Some((min, max)) = depth_bounds {
        pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(list, PC_OM_SET_DEPTH_BOUNDS)(list, min, max);
    }
}

/// The counts a list's state is in, for a state that goes (`on_release` does the same).
fn uncount(o: &ListState) {
    if !o.pending_waits.is_empty() {
        PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
    }
    if !o.sparse_writes.is_empty() {
        SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
    }
    if !o.write64_signals.is_empty() {
        WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// PC allocators the calls put off to the submit record on (`record_deferred`), one per title
/// allocator and PC list type: by the submit the title's allocator may be recording another list.
/// Each is reset with its title allocator (`on_allocator_reset`): the title resets that once the
/// GPU is done with every list recorded on it, and so with their put-off calls too.
type Companions = HashMap<(usize, i32), std::sync::Arc<Mutex<ID3D12CommandAllocator>>>;
static COMPANIONS: Mutex<Option<Companions>> = Mutex::new(None);

unsafe fn companion(dev: &ID3D12Device, alloc: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<std::sync::Arc<Mutex<ID3D12CommandAllocator>>> {
    let ty = D3D12_COMMAND_LIST_TYPE(super::single::list_type(ty.0 as usize) as i32);
    let mut g = COMPANIONS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(c) = m.get(&(alloc, ty.0)) {
        return Some(c.clone());
    }
    let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).ok()?;
    let c = std::sync::Arc::new(Mutex::new(a));
    m.insert((alloc, ty.0), c.clone());
    Some(c)
}

/// Calls put off to the submit in one frame at most, over all lists (`defer`).
const DEFERRED_PER_FRAME: usize = 256;
static DEFER_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
static DEFERRED_IN_FRAME: AtomicUsize = AtomicUsize::new(0);

/// A call whose pipeline is still being built (`bind::put_off`) gets a closed segment of its
/// own in `list`'s recording, recorded at the submit (`record_deferred`); the list goes on in
/// the segment after it. False: the call stays in the list and is skipped as before (counted) –
/// not a title list, a bundle, a predicated call or one inside an occlusion query (its Begin and
/// End would be in different PC lists), or the budget used up: `title_list::MAX_DEFERRED` a
/// recording, `DEFERRED_PER_FRAME` a frame (upstream, a segment for every unbuilt call of a world
/// start ran a retail title out of command list memory: Close E_OUTOFMEMORY, then device removed).
pub(super) unsafe fn defer(list: *mut c_void, what: Deferred) -> bool {
    let Some(wrapper) = super::title_list::wrapper_of(list) else { return false };
    let ty = list_type(list);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return false;
    }
    let frame = super::present::presents();
    if DEFER_FRAME.swap(frame, Ordering::AcqRel) != frame {
        DEFERRED_IN_FRAME.store(0, Ordering::Release);
    }
    let room = DEFERRED_IN_FRAME.load(Ordering::Acquire) < DEFERRED_PER_FRAME && super::title_list::can_defer(wrapper);
    let done = room
        && match put_off_segment(list, wrapper, ty) {
            Some(seg) => split(list, wrapper, Some((what, seg))).is_some() || {
                super::title_list::put_off_spare(wrapper, seg);
                false
            },
            None => false,
        };
    if !done {
        static KEPT: AtomicU64 = AtomicU64::new(0);
        let n = KEPT.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: a {} with a pipeline still being built skipped, not put off to the submit: {} (#{n})",
                what.name(),
                if room { "predicated, inside an occlusion query, or no segment for it" } else { "the list's or the frame's budget is used up" }
            );
        }
        return false;
    }
    DEFERRED_IN_FRAME.fetch_add(1, Ordering::AcqRel);
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a {} with a pipeline still being built put off to the submit (#{n})", what.name());
    }
    true
}

/// A closed PC list of type `ty` for a call put off to the submit, holding no recording that has
/// not run: one of `wrapper`'s put-off segments whose last recording ran (`pop_put_off_spare`), or
/// a new one made closed with no recording (`CreateCommandList1`). Nothing is recorded in it here:
/// the submit resets it once, on the companion of `list`'s allocator (`companion`), and records
/// the call or nothing (`record_put_off`). A list reset and closed empty here was reset again at
/// the submit, dropping a recording that never ran: D3D12 destroyed an allocation in it, and the
/// kernel call did not return while the submitting thread's queues sat behind GPU waits only it
/// could release. None: the call stays in `list` (`defer`).
unsafe fn put_off_segment(list: *mut c_void, wrapper: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE) -> Option<*mut c_void> {
    let alloc = {
        let g = LISTS.lock().unwrap();
        let s = g.as_ref()?.get(&(list as usize))?;
        let plain = s.allocator != 0 && s.predication.is_none() && s.draw_predicate.is_none() && s.bindings.predication.is_none() && s.occlusion.active.is_none();
        plain.then_some(s.allocator)?
    };
    let dev = device_of(list)?;
    companion(&dev, alloc, ty)?;
    super::title_list::pop_put_off_spare(wrapper).or_else(|| {
        let seg = real::create_closed_pc_command_list(dev.as_raw(), ty);
        if seg.is_none() {
            static ERR: AtomicU64 = AtomicU64::new(0);
            warn(&ERR, "put-off call", format_args!("CreateCommandList1 of its segment failed"));
        }
        seg
    })
}

/// The render targets a draw put off to the submit binds, copied into heaps of its own (kept with
/// its segment): the title may write other views into its descriptors before the submit, and the
/// PC reads them at OMSetRenderTargets. Left as they are when a copy cannot be made.
unsafe fn snapshot_render_targets(dev: &ID3D12Device, b: &mut Bindings) {
    let Some(rt) = b.render_targets.as_ref() else { return };
    let mut copied = rt.clone();
    let mut heaps = Vec::new();
    let colour = if rt.handles.is_empty() { 0 } else { rt.num_targets };
    for (ty, count) in [(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, colour), (D3D12_DESCRIPTOR_HEAP_TYPE_DSV, rt.dsv.is_some() as u32)] {
        if count == 0 {
            continue;
        }
        let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: count, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 };
        let Ok(heap) = dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc) else { return };
        let base = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let step = dev.GetDescriptorHandleIncrementSize(ty) as usize;
        for i in 0..count as usize {
            let src = match (ty == D3D12_DESCRIPTOR_HEAP_TYPE_DSV, rt.single != 0) {
                (true, _) => rt.dsv.unwrap_or(0),
                (false, true) => rt.handles[0] + i * step,
                (false, false) => rt.handles[i],
            };
            dev.CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base + i * step }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src }, ty);
        }
        if ty == D3D12_DESCRIPTOR_HEAP_TYPE_DSV {
            copied.dsv = Some(base);
        } else {
            copied.handles = vec![base];
            copied.single = 1;
        }
        heaps.push(heap);
    }
    b.render_targets = Some(copied);
    b.render_target_copies = heaps;
}

/// Records the calls put off to the submit in `wrapper`'s recording (`defer`), each once its
/// pipeline is built: the submit waits for them all first, until `deadline` at most
/// (`pso_async::wait_for_submit`). A call whose pipeline is not built by then goes the way it
/// went before calls were put off: skipped and counted.
pub(super) unsafe fn record_deferred(wrapper: *mut c_void, deadline: Instant) {
    if !super::pso_async::defer_to_submit() {
        return;
    }
    let mut segs = Vec::new();
    if !super::title_list::push_segments(wrapper, &mut segs) || segs.len() < 2 {
        return;
    }
    let put_off: Vec<*mut c_void> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        segs.into_iter().filter(|&seg| m.get(&(seg as usize)).is_some_and(|s| s.deferred.is_some())).collect()
    };
    for &seg in &put_off {
        super::pso_async::wait_for_submit(super::bind::tracked_pso(seg), deadline);
    }
    let Some(&first) = put_off.first() else { return };
    if let Some(reason) = removed_reason(first) {
        if !REMOVAL_NOTED.swap(true, Ordering::AcqRel) {
            crate::xlog_error!("D3D12.X: the device was removed ({reason:#x}) before a submit recorded the {} calls put off to it", put_off.len());
        }
    }
    let t0 = Instant::now();
    let (n, draws) = (put_off.len(), put_off.into_iter().filter(|&seg| record_put_off(seg)).count());
    static SUBMITS: AtomicU64 = AtomicU64::new(0);
    let k = SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
    if k <= 8 || k.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a submit recorded the {n} calls put off to it ({draws} draws) in {} ms (#{k})", t0.elapsed().as_millis());
    }
}

/// The last calls put off to a submit and recorded there (`record_put_off`), oldest first, and
/// the one a submit is recording now: for the log when the device is removed
/// (`put_off_at_removal`).
static RECORDED: Mutex<std::collections::VecDeque<String>> = Mutex::new(std::collections::VecDeque::new());
const RECORDED_KEPT: usize = 64;
static RECORDING: Mutex<Option<String>> = Mutex::new(None);
/// the device's removal was logged by a submit (`record_deferred`, `record_put_off`)
static REMOVAL_NOTED: AtomicBool = AtomicBool::new(false);

fn note_recorded(line: String) {
    let mut kept = RECORDED.lock().unwrap_or_else(|e| e.into_inner());
    if kept.len() == RECORDED_KEPT {
        kept.pop_front();
    }
    kept.push_back(line);
}

/// The device was removed (`real::log_device_removed`): the calls put off to a submit and
/// recorded there last, and the one being recorded then.
pub(super) fn put_off_at_removal() {
    if let Some(now) = RECORDING.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        crate::xlog_error!("D3D12.X: at the removal a submit was recording put-off {now}");
    }
    let kept = RECORDED.lock().unwrap_or_else(|e| e.into_inner());
    if kept.is_empty() {
        return;
    }
    crate::xlog_error!("D3D12.X: the last {} calls put off to a submit and recorded there, oldest first:", kept.len());
    for line in kept.iter() {
        crate::xlog_error!("D3D12.X:   {line}");
    }
}

/// The device of `list` was removed: the reason GetDeviceRemovedReason gives.
unsafe fn removed_reason(list: *mut c_void) -> Option<u32> {
    device_of(list)?.GetDeviceRemovedReason().err().map(|e| e.code().0 as u32)
}

/// Why the call `what` put off to a segment of type `ty` must not be recorded there with the
/// state it kept (None: it may): its pipeline (`compute`, None: not known) is not of its kind,
/// no root signature of its kind is bound (`roots`: compute, graphics), its render targets
/// were not copied when it was put off (the title may have written other views into their
/// descriptors since), or an indexed draw has no index buffer.
fn refuse_put_off(what: &Deferred, ty: D3D12_COMMAND_LIST_TYPE, compute: Option<bool>, roots: (bool, bool), b: &Bindings, index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>) -> Option<&'static str> {
    let draws = what.draws();
    if draws && ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return Some("a draw on a list that is not a direct one");
    }
    if compute == Some(draws) {
        return Some(if draws { "a draw with a compute pipeline" } else { "a dispatch with a graphics pipeline" });
    }
    if !(if draws { roots.1 } else { roots.0 }) {
        return Some("no root signature of its kind bound");
    }
    if !draws {
        return None;
    }
    if b.render_targets.as_ref().is_some_and(|rt| !rt.handles.is_empty() || rt.dsv.is_some()) && b.render_target_copies.is_empty() {
        return Some("its render targets were not copied when it was put off");
    }
    if matches!(what, Deferred::DrawIndexed(..)) && index_buffer.is_none() {
        return Some("an indexed draw with no index buffer");
    }
    None
}

/// The call put off to the segment `seg`, recorded in it on its companion allocator with the
/// state its list had (`split`), unless that state is not fit for it (`refuse_put_off`):
/// true when it draws. `seg` is reset here once, whatever becomes of the call: a call not
/// recorded leaves it empty, never with the recording it had before (`put_off_segment`). What it was recorded with, and how, is kept for the log of a removal
/// (`put_off_at_removal`), and the device is checked after it: the first call after which the
/// device is found removed is logged.
unsafe fn record_put_off(seg: *mut c_void) -> bool {
    static ERR: AtomicU64 = AtomicU64::new(0);
    let taken = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(seg as usize))).and_then(|s| {
        Some((s.deferred.take()?, s.list_type?, s.allocator, s.bound_pso, s.depth_bias, s.index_buffer, s.depth_bounds, s.bindings.clone()))
    });
    let Some((what, ty, alloc, bound_pso, depth_bias, index_buffer, depth_bounds, bindings)) = taken else { return false };
    let draws = what.draws();
    let title_pso = super::bind::tracked_pso(seg);
    let kind = super::pso_async::kind_of(title_pso);
    let said = {
        let (shaders, real) = kind.as_ref().map_or(("not built here", 0), |(_, what, real)| (if what.is_empty() { "shaders unknown" } else { what.as_str() }, *real));
        let rt = bindings.render_targets.as_ref();
        format!(
            "{what:?} in frame {} on {} list segment {seg:p}: pipeline {title_pso:#x} (real {real:#x}, {shaders}), root signatures {:?}; {} render targets{}{}, {} viewports, {} scissors, topology {:?}, vertex buffers {:?}, index buffer {:x?}, {} descriptor heaps",
            super::present::presents(),
            match ty { D3D12_COMMAND_LIST_TYPE_DIRECT => "a direct", D3D12_COMMAND_LIST_TYPE_COMPUTE => "a compute", D3D12_COMMAND_LIST_TYPE_COPY => "a copy", _ => "another" },
            super::bind::roots_bound(seg),
            rt.map_or(0, |rt| rt.num_targets),
            if rt.is_some_and(|rt| rt.dsv.is_some()) { " and depth" } else { "" },
            if bindings.render_target_copies.is_empty() { "" } else { " (copied)" },
            bindings.viewports.len(),
            bindings.scissors.len(),
            bindings.topology,
            bindings.vertex_buffers.iter().map(|(slot, v)| (*slot, v.BufferLocation, v.SizeInBytes, v.StrideInBytes)).collect::<Vec<_>>(),
            index_buffer.map(|v| (v.BufferLocation, v.SizeInBytes, v.Format.0)),
            bindings.heaps.len(),
        )
    };
    // (an allocator the title released meanwhile: its lists must not run)
    let Some((dev, companion)) = device_of(seg).filter(|_| alloc != 0).and_then(|d| Some((companion(&d, alloc, ty)?, d)).map(|(c, d)| (d, c))) else {
        note_recorded(format!("skipped (its allocator is gone): {said}"));
        super::bind::deferred_skipped(draws);
        return draws;
    };
    let own = companion.lock().unwrap();
    let own_raw = own.as_raw() as usize;
    let reset = |init: usize| {
        let ext = *(seg as *const usize);
        let hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(seg, PC_RESET)(seg, own_raw, init);
        real::preserve_vtable(seg, ext);
        hr
    };
    let close = || {
        super::depth_uav::before_close(seg);
        super::gpu_timing::before_close(seg);
        let ext = *(seg as *const usize);
        let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(seg, PC_CLOSE)(seg);
        real::preserve_vtable(seg, ext);
        hr
    };
    if let Some(why) = refuse_put_off(&what, ty, kind.as_ref().map(|k| k.0), super::bind::roots_bound(seg), &bindings, index_buffer) {
        static REFUSED: AtomicU64 = AtomicU64::new(0);
        let n = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            crate::xlog_warn!("D3D12.X: a call put off to the submit skipped there, {why} (#{n}): {said}");
        }
        if reset(0) >= 0 {
            super::physical_directory::recordings::ended(seg as usize);
            super::gpu_timing::opened(seg, 0);
            super::material_gate_views::prepass::native_reset(seg as usize);
            close();
        }
        note_recorded(format!("skipped ({why}): {said}"));
        super::bind::deferred_skipped(draws);
        return draws;
    }
    // (a draw goes on with the pipeline its list had bound; a dispatch binds its own)
    let init_pso = if draws && ty != D3D12_COMMAND_LIST_TYPE_COPY { super::pso_async::resolve(title_pso) } else { 0 };
    super::shader_code::drawn_on(own_raw, init_pso);
    let hr = reset(init_pso);
    if hr < 0 {
        warn(&ERR, "put-off call", format_args!("Reset failed ({hr:#x})"));
        note_recorded(format!("skipped (Reset failed, {hr:#x}): {said}"));
        super::bind::deferred_skipped(draws);
        return draws;
    }
    *RECORDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(said.clone());
    super::physical_directory::recordings::ended(seg as usize);
    super::gpu_timing::opened(seg, init_pso);
    super::material_gate_views::prepass::native_reset(seg as usize);
    restore_bindings(seg, &bindings, ty);
    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT {
        restore_direct(seg, dev.as_raw() as usize, bound_pso, depth_bias, index_buffer, depth_bounds);
    }
    match what {
        Deferred::Dispatch(x, y, z, flags) => super::bind::dispatch_flags(seg, x, y, z, flags),
        Deferred::Indirect(sig, max, args, args_off, count_buf, count_off) => super::bind::execute_indirect(seg, sig, max, args, args_off, count_buf, count_off),
        Deferred::Draw(vertices, instances, first_vertex, first_instance) => super::bind::draw_instanced(seg, vertices, instances, first_vertex, first_instance),
        Deferred::DrawIndexed(indices, instances, first_index, base_vertex, first_instance) => super::bind::draw_indexed_instanced(seg, indices, instances, first_index, base_vertex, first_instance),
        Deferred::DrawIndexedX(va, format, count) => draw_indexed_x(seg, va, format, count, 0),
    }
    let hr = close();
    *RECORDING.lock().unwrap_or_else(|e| e.into_inner()) = None;
    if hr < 0 {
        // a list whose Close failed must not run: an empty one instead
        warn(&ERR, "put-off call", format_args!("Close failed ({hr:#x}); left out"));
        super::bind::deferred_skipped(draws);
        if reset(0) >= 0 {
            close();
        }
    }
    note_recorded(if hr < 0 { format!("left out (Close failed, {hr:#x}): {said}") } else { format!("recorded: {said}") });
    if let Some(reason) = removed_reason(seg) {
        if !REMOVAL_NOTED.swap(true, Ordering::AcqRel) {
            crate::xlog_error!("D3D12.X: the device was found removed ({reason:#x}) right after a submit recorded put-off {said}");
        }
    }
    draws
}

/// Record a `Wait32BitValueX` or `Wait64BitValueX` memory wait on `list`.
///
/// If an earlier write in the same open list already targeted an overlapping UMA range, the
/// `COMMON -> COPY_DEST -> COMMON` barrier around that write already orders subsequent work
/// on the GPU, so we do not stall before submission (which would deadlock waiting on our own
/// unsubmitted list). If this list already recorded a `Write64BitValue*` to a different cell,
/// we split into a new deferred PC segment so earlier `Write64` signals run BEFORE this wait.
unsafe fn wait_memory(list: *mut c_void, va: u64, width: u8, comparison: u32, value: u64, mask: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = width as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    // (the wait reads the memory on the CPU: the title's own memory too, which `cpu_writes` fills)
    if real::uma_window_buffer(va as usize, width as usize).is_none() && !crate::gpu::shader::readable(va as usize, width as usize) {
        return warn(&N, what, format_args!("{va:#x} is not readable memory"));
    }
    if compare_u64(comparison, 0, 0).is_none() {
        return warn(&N, what, format_args!("unknown comparison op {comparison}"));
    }
    if comparison == D3D12_COMPARISON_FUNC_ALWAYS.0 as u32 {
        return;
    }
    if comparison == D3D12_COMPARISON_FUNC_NEVER.0 as u32 {
        return warn(&N, what, format_args!("NEVER comparison at {va:#x} (skipped to avoid hang)"));
    }
    let need_split = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        if s.has_prior_write(va, n) {
            return;
        }
        if s.write64_signals.is_empty() {
            s.push_wait(PendingWait { va, width, comparison, value, mask, after_write64: false });
            return;
        }
        true
    };
    let target = if need_split { try_split_for_wait(list).unwrap_or(list) } else { list };
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(target as usize).or_default();
    let after_write64 = !s.write64_signals.is_empty();
    s.push_wait(PendingWait { va, width, comparison, value, mask, after_write64 });
}

/// command list #71 Wait32BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait32_x(this: *mut c_void, va: u64, comparison: u32, value: u32, mask: u32, _flags: u32) {
    wait_memory(this, va, 4, comparison, value as u64, mask as u64, "Wait32BitValueX");
}

/// command list #72 Wait64BitValueX(address, comparison, value, flags): the whole 64-bit value
/// is compared. What comes after the value is not a mask: a retail game passes a 32-bit 1 there (the
/// upper half of the stack slot is whatever was left in it – 0x231_0000_0001), and read as a
/// mask its waits for a counter at 2 ≥ 1, 4 ≥ 2 … never held: each timed out (2 s a frame, the
/// game stood still after its first cinematic).
pub(super) unsafe extern "system" fn wait64_x(this: *mut c_void, va: u64, comparison: u32, value: u64, _flags: u64, _more: u32) {
    wait_memory(this, va, 8, comparison, value, u64::MAX, "Wait64BitValueX");
}

/// command list #88 Write32BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write32_top_x(this: *mut c_void, va: u64, value: u32, flags: u32) {
    record_write_trace(this, va, value as u64, flags, "W32T");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, "Write32BitValueTopOfPipeX");
}

/// command list #89 Write64BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write64_top_x(this: *mut c_void, va: u64, value: u64, flags: u32) {
    const W: &str = "Write64BitValueTopOfPipeX";
    record_write_trace(this, va, value, flags, "W64T");
    if flags & TOP_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    note_write64_recorded(this, va, value);
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, W);
}

/// command list #90 Write32BitValueBottomOfShaderX(address, value, stage)
pub(super) unsafe extern "system" fn write32_bottom_of_shader_x(this: *mut c_void, va: u64, value: u32, stage: u32) {
    record_write_trace(this, va, value as u64, stage, "W32S");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfShaderX");
}

/// command list #68 OMSetDepthBounds(min, max) (PC #62)
pub(super) unsafe extern "system" fn set_depth_bounds_x(this: *mut c_void, min: f32, max: f32) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(this as usize).or_default().depth_bounds = Some((min, max));
    pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
}

/// command list #78 DispatchX(x, y, z, flags)
pub(super) unsafe extern "system" fn dispatch_x(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    super::bind::dispatch_flags(this, x, y, z, Some(flags));
}

/// command list #146 console compute/UAV barrier (used by `XboxBvhBuilder.dll` between BVH
/// build passes): emits a global UAV barrier so UMA writes become visible to the next pass
pub(super) unsafe extern "system" fn uav_barrier_x(this: *mut c_void, n: u32, barriers: *const c_void) {
    super::depth_uav::observe_slot146(this as usize, n, barriers, true);
    global_uav_barrier(this);
    super::depth_uav::note_slot146_global(this as usize);
}

unsafe fn global_uav_barrier(this: *mut c_void) {
    if super::bind::indirect_world_trace() {
        super::bind::indirect_world_event(this as usize, format_args!("global UAV barrier"));
    }
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }),
        },
    };
    super::gpu_timing::class(this, super::gpu_timing::Kind::Barrier, || {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(this, PC_RESOURCE_BARRIER)(this, 1, &b);
    });
}

/// A UAV barrier for all resources on `list`, where a list takes one (a copy list and a bundle
/// do not).
unsafe fn flush_barrier(list: *mut c_void) {
    let ty = list_type(list);
    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT || ty == D3D12_COMMAND_LIST_TYPE_COMPUTE {
        global_uav_barrier(list);
    }
}

/// command list #131 FlushGpuCacheRangeX / #132 FlushShaderCacheRangeX (cache flags, wait flags,
/// address, bytes), the newer layout's: the console writes its caches back (or drops what they
/// hold) for a range of memory and waits for the work before it. It is how a title orders a pass
/// behind the one that wrote what it reads – a retail title calls them thousands of times a minute,
/// and answered with nothing its passes had no order there. The PC has no ranges of caches: a UAV
/// barrier for everything.
pub(super) unsafe extern "system" fn flush_cache_range_x(this: *mut c_void, flags: u32, wait: u32, va: u64, bytes: u64) {
    if (flags != 0 || wait != 0) && bytes != 0 {
        if super::bind::indirect_world_trace() {
            super::bind::indirect_world_event(this as usize, format_args!("console cache flush {va:#x}+{bytes:#x} flags {flags:#x} wait {wait:#x}"));
        }
        flush_barrier(this);
    }
}

/// command list #67 (newer layout: #69) FlushPipelineX(flags, address, bytes)
pub(super) unsafe extern "system" fn flush_pipeline_x(this: *mut c_void, _flags: u32, _va: u64, _bytes: u64) {
    flush_barrier(this);
}

/// command list #84 CopyResourceX(destination, source, flags); in place (the console's
/// compression operations) there is nothing to do on PC
pub(super) unsafe extern "system" fn copy_resource_flags_x(this: *mut c_void, dst: usize, src: usize, _flags: u32) {
    if dst != src {
        copy_resource_x(this, dst, src);
    }
}

/// Return `list`'s dedicated 256-byte predication buffer, creating it lazily in `COMMON` state.
unsafe fn predication_buffer(list: *mut c_void) -> Option<usize> {
    if let Some(p) = LISTS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .and_then(|s| s.pred_buf.as_ref())
        .map(|r| r.as_raw() as usize)
    {
        return Some(p);
    }
    let dev = device_of(list)?;
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).ok()?;
    let buf = buf?;
    let (raw, extra) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        if let Some(existing) = s.pred_buf.as_ref() {
            (existing.as_raw() as usize, Some(buf))
        } else {
            let raw = buf.as_raw() as usize;
            s.pred_buf = Some(buf);
            (raw, None)
        }
    };
    drop(extra);
    Some(raw)
}

/// Before one of XCL's own passes in the middle of `list`'s recording (`depth_uav`): the title's
/// predication does not hold for it. True: it was on (`after_own_pass` turns it on again).
pub(super) unsafe fn before_own_pass(list: *mut c_void) -> bool {
    if !PREDICATION_USED.load(Ordering::Acquire) {
        return false;
    }
    let on = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).is_some_and(|s| s.bindings.predication.is_some() || (s.predication.is_some() && s.pred_in_state));
    if on {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(list, PC_SET_PREDICATION)(list, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    on
}

/// After one of XCL's own passes on `list`: what the pass set in place of the title's is set
/// again – the descriptor heaps, the pipeline, the predication and, after a pass that drew
/// (`drew`), the fixed-function state. Root signatures and root arguments are bound by every
/// draw and dispatch of the title's (`bind`).
pub(super) unsafe fn after_own_pass(list: *mut c_void, drew: bool, predicated: bool) {
    let ty = list_type(list);
    let (pso, predication) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else { return };
        if drew {
            restore_bindings(list, &s.bindings, ty);
        } else if !s.bindings.heaps.is_empty() {
            pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(list, 28)(list, s.bindings.heaps.len() as u32, s.bindings.heaps.as_ptr());
        }
        // a pipeline derived for a depth bias is bound again by the next draw
        s.pso_dirty = true;
        let standard = if drew { None } else { s.bindings.predication };
        let by_address = s.predication.zip(s.pred_buf.as_ref().map(|r| r.as_raw() as usize)).map(|(op, buf)| (buf, 0, op));
        (s.bound_pso, standard.or(by_address))
    };
    if pso != 0 {
        set_native_pso(list, super::pso_async::resolve_for_bind(pso));
    }
    if let (true, Some((buf, offset, op))) = (predicated, predication) {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(list, PC_SET_PREDICATION)(list, buf, offset, op);
    }
}

/// Before `Close()` on a command list: return `pred_buf` to `COMMON` if it is currently in
/// `D3D12_RESOURCE_STATE_PREDICATION`.
pub(super) unsafe fn before_close(list: *mut c_void) {
    if !PREDICATION_USED.load(Ordering::Acquire) || list.is_null() {
        return;
    }
    let pb = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if s.pred_in_state {
            s.pred_in_state = false;
            s.pred_buf.as_ref().map(|r| r.as_raw() as usize)
        } else {
            None
        }
    };
    if let Some(pb) = pb {
        transition(list, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
    }
}

/// Clear active predication state when `ID3D12GraphicsCommandList::Reset` resets a recording.
pub(super) fn on_pc_reset(list: *mut c_void) {
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.helper = Default::default();
    }
    super::physical_directory::recordings::ended(list as usize);
    if !list.is_null() && super::depth_capture_views::enabled() {
        with_bindings(list, |b| {
            b.depth_capture_native_pso = 0;
            b.depth_capture_dsv_generation = None;
            b.lineage_om_valid = false;
            b.lineage_pass_dsv = None;
        });
    }
    if !(PREDICATION_USED.load(Ordering::Acquire) || super::bind::draw_gate_trace()) || list.is_null() {
        return;
    }
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.predication = None;
        s.draw_predicate = None;
        s.pred_in_state = false;
        s.bindings.predication = None;
    }
}

/// Copy 8 predicate bytes from `(uma_buf, off)` into `pb` with exact barrier transitions and
/// bind `SetPredication(pb, 0, op)` on `this`.
unsafe fn apply_predication_copy(this: *mut c_void, uma_buf: usize, off: u64, pb: usize, op: D3D12_PREDICATION_OP) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let f = pc::<F>(this, PC_SET_PREDICATION);
    let (was_active, was_in_state) = LISTS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(this as usize)))
        .map_or((false, false), |s| (s.predication.is_some() || s.bindings.predication.is_some(), s.pred_in_state));
    // Disarm any active predicate before copying the new predicate value so a failing predicate
    // does not suppress `CopyBufferRegion` when switching directly between buffers.
    if was_active {
        f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    transition(this, uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    let pb_before = if was_in_state { D3D12_RESOURCE_STATE_PREDICATION } else { D3D12_RESOURCE_STATE_COMMON };
    transition(this, pb, pb_before, D3D12_RESOURCE_STATE_COPY_DEST);
    copy_buffer_region(this, pb, 0, uma_buf, off, 8);
    transition(this, uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    transition(this, pb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION);
    f(this, pb, 0, op);
    PREDICATION_USED.store(true, Ordering::Release);
    if let Some(s) = LISTS.lock().unwrap().get_or_insert_with(HashMap::new).get_mut(&(this as usize)) {
        s.predication = Some(op);
        s.pred_in_state = true;
        s.bindings.predication = None;
    }
}

/// command list #31 SetPredication(buffer, offset, operation) (PC #55): buffers placed below
/// their console address add `buffer_delta(buffer)` to the offset, as ResolveQueryData does, so
/// the predicate is read where the resolve wrote it. Kept for KickoffX and mid-list splits
/// (`restore_bindings`).
pub(super) unsafe extern "system" fn set_predication_x(this: *mut c_void, res: usize, offset: u64, op: D3D12_PREDICATION_OP) {
    let pc_offset = if res == 0 { offset } else { offset.saturating_add(real::buffer_delta(res)) };
    PREDICATION_USED.store(true, Ordering::Release);
    if res != 0 {
        super::residency::note_list_resources(this as usize, [res]);
    }
    {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        s.bindings.predication = (res != 0).then_some((res, pc_offset, op));
        // the last of #31 and #85 SetPredicationBufferX is the one in effect (#85's buffer goes
        // back to COMMON at Close or KickoffX: `pred_in_state`)
        s.predication = None;
    }
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    pc::<F>(this, PC_SET_PREDICATION)(this, res, pc_offset, op);
}

/// command list #85 SetPredicationBufferX({ u64 address; u32 operation }*)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer, and non-null `desc`
/// must point to readable memory if present.
pub unsafe extern "system" fn set_predication_buffer_x(this: *mut c_void, desc: *const u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(this);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, "SetPredicationBufferX", format_args!("unsupported command list type {}", ty.0));
    }
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let requested = if !desc.is_null() && crate::gpu::shader::readable(desc as usize, 12) {
        Some((desc.read_unaligned(), (desc.add(1) as *const u32).read_unaligned()))
    } else {
        None
    };
    if desc.is_null() || super::bind::ignore_predication() {
        let f = pc::<F>(this, PC_SET_PREDICATION);
        f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        let pb = {
            let mut g = LISTS.lock().unwrap();
            g.as_mut().and_then(|m| m.get_mut(&(this as usize))).and_then(|s| {
                s.predication = None;
                s.draw_predicate = None;
                s.bindings.predication = None;
                if s.pred_in_state {
                    s.pred_in_state = false;
                    s.pred_buf.as_ref().map(|r| r.as_raw() as usize)
                } else {
                    None
                }
            })
        };
        if let Some(pb) = pb {
            transition(this, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
        super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("requested {requested:x?} (cleared)"), requested.map(|(va, _)| va));
        return;
    }
    let Some((va, op_raw)) = requested else {
        super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("unreadable descriptor {desc:p}"), None);
        return warn(&N, "SetPredicationBufferX", format_args!("descriptor {desc:p} is not readable"));
    };
    if op_raw > 1 {
        super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("rejected {va:#x} op {op_raw}"), Some(va));
        return warn(&N, "SetPredicationBufferX", format_args!("unknown predication op {op_raw}"));
    }
    let gathered = if va.is_multiple_of(8) { native_buffer::destinations::predicate(this, va) } else { None };
    let Some((uma_buf, off)) = gathered.as_ref().map(|r| (r.as_raw() as usize, 0)).or_else(|| real::uma_window_buffer(va as usize, 8)).filter(|&(b, _)| b != 0 && va.is_multiple_of(8)) else {
        super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("rejected {va:#x} op {op_raw}"), Some(va));
        return warn(&N, "SetPredicationBufferX", format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    let Some(pb) = predication_buffer(this) else {
        super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("could not allocate for {va:#x} op {op_raw}"), Some(va));
        return warn(&N, "SetPredicationBufferX", format_args!("could not allocate predication buffer"));
    };
    apply_predication_copy(this, uma_buf, off, pb, D3D12_PREDICATION_OP(op_raw as i32));
    note_draw_predicate(this, Some(va), pb, 0, op_raw as i32);
    super::bind::draw_gate_predication(this, "SetPredicationBufferX", format_args!("title {va:#x} op {op_raw}"), Some(va));
}

/// command list #93 KickoffX(): flush commands recorded so far by closing the list, submitting
/// it to a compatible live command queue, and resetting it on the same command allocator so
/// recording can continue seamlessly until the title's normal `Close` + `ExecuteCommandLists`.
///
/// If no compatible queue or allocator is known yet (e.g. a bundle, or a list created before
/// any queue of its type exists), `KickoffX` is a safe logged no-op and the recorded work runs
/// at `Close` + `ExecuteCommandLists`.
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn kickoff_x(this: *mut c_void) {
    static NO_QUEUE: AtomicU64 = AtomicU64::new(0);
    static ERR: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(this);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return;
    }
    let dev_raw = device_of(this).map_or(0, |d| d.as_raw() as usize);
    let (alloc, queue, bound_pso, depth_bias, index_buffer, depth_bounds, predication, draw_predicate, pred_buf, pred_in_state, active_query, waits, sparse_writes) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        };
        if s.allocator == 0 {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        }
        let Some(q) = pick_queue(s.last_queue, dev_raw, s.list_type.unwrap_or(ty)) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – no live queue of type {} yet, deferring to Close + ExecuteCommandLists", ty.0);
            }
            return;
        };
        let active_query = s.occlusion.end_active_early();
        let waits = if s.pending_waits.is_empty() {
            Vec::new()
        } else {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            std::mem::take(&mut s.pending_waits)
        };
        let sparse_writes = s.clear_sparse_writes();
        let write64_signals = s.clear_write64_signals();
        s.written_ranges.clear();
        let cpu_writes = std::mem::take(&mut s.cpu_writes);
        let pred_in_state = std::mem::replace(&mut s.pred_in_state, false);
        let pred_buf = s.pred_buf.as_ref().map(|r| r.as_raw() as usize);
        (s.allocator, q, s.bound_pso, s.depth_bias, s.index_buffer, s.depth_bounds, s.predication, s.draw_predicate, pred_buf, pred_in_state, active_query, waits, (sparse_writes, write64_signals, cpu_writes))
    };
    let (sparse_writes, write64_signals, cpu_writes) = sparse_writes;

    // End any active occlusion query before Close (PC forbids closing with an open query).
    if let Some(((h, i), qty)) = active_query {
        pc::<QueryFn>(this, PC_END_QUERY)(this, h, qty, i);
    }
    if pred_in_state {
        if let Some(pb) = pred_buf {
            transition(this, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    super::depth_uav::before_close(this);
    super::gpu_timing::before_close(this);
    let ext = *(this as *const usize);
    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_CLOSE)(this);
    real::preserve_vtable(this, ext);
    if close_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Close failed ({close_hr:#x})"));
    }
    if let Some(wrapper) = super::title_list::wrapper_of(this) {
        record_deferred(wrapper, super::pso_async::submit_deadline());
    }

    let q_raw = super::single::kickoff_queue(this as usize, queue as *mut c_void);
    super::title_textures::refresh(q_raw);
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: ExecFn = transmute_copy(&real::orig_fn(q_raw, PC_QUEUE_EXECUTE_COMMAND_LISTS));
    let one = [this];
    let submission = if super::residency::eviction_enabled() {
        let mut segments = Vec::new();
        if let Some(wrapper) = super::title_list::wrapper_of(this).filter(|&w| super::title_list::backing(w) != Some(this)) {
            super::title_list::push_segments(wrapper, &mut segments);
        }
        let lists = if segments.is_empty() { &one[..] } else { &segments };
        super::residency::before_submit(q_raw, lists)
    } else {
        None
    };

    // If this list was previously split mid-recording around a `Wait64`, flush all earlier closed
    // segments in exact order before submitting `this`, then resume recording on `primary`.
    let kickoff_split = super::title_list::reclaim_for_kickoff(this);
    if let Some((_, ref prior, _)) = kickoff_split {
        if !prior.is_empty() {
            execute_with_waits_on(q_raw, prior, |chunk| {
                super::bind::indirect_world_submit(q_raw as usize, chunk);
                super::page_mappings::queue::submit(q_raw, chunk, || super::gpu_timing::submit(q_raw, chunk, || exec(q_raw, chunk.len() as u32, chunk.as_ptr())));
                super::present::submit_plane_writes(q_raw, chunk);
                signal_write64_fences(q_raw, chunk);
                readback_sparse_writes(q_raw, chunk);
                cpu_writes_after(q_raw, chunk);
                super::image_writeback::submitted(q_raw, chunk);
            });
            super::title_textures::submitted(prior);
            for &p in prior {
                super::title_textures::list_reset(p as usize);
            }
        }
    }

    if !waits.is_empty() {
        wait_all_on(q_raw, &waits, &write64_signals);
    }

    if traced_queue(q_raw as usize) {
        trace_queue(|| format!("q{}: KickoffX of one list ({} waits, {} Write64 signals)", queue_id(q_raw as usize), waits.len(), write64_signals.len()));
    }
    super::bind::indirect_world_submit(q_raw as usize, &one);
    super::page_mappings::queue::submit(q_raw, &one, || super::gpu_timing::submit(q_raw, &one, || exec(q_raw, 1, one.as_ptr())));
    super::present::submit_plane_writes(q_raw, &one);
    super::residency::after_submit(q_raw, submission);
    signal_write64_pairs(q_raw, &write64_signals);
    super::image_writeback::submitted(q_raw, &one);
    // what ran is in the textures' states; the list records anew after its Reset below
    super::title_textures::submitted(&one);
    super::title_textures::list_reset(this as usize);
    if !sparse_writes.is_empty() {
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&q_raw) {
            super::sparse::readback_after_execute(q, &sparse_writes);
        }
    }
    super::cpu_writes::after(q_raw, cpu_writes);

    let target = if let Some((primary, _, reclaimed)) = kickoff_split {
        let (ended_early, pred_buf_obj, bindings_copy) = {
            let mut g = LISTS.lock().unwrap();
            if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                (std::mem::take(&mut s.occlusion.ended_early), s.pred_buf.clone(), s.bindings.clone())
            } else {
                (Vec::new(), None, Bindings::default())
            }
        };
        super::bind::inherit_for_split(this, primary, alloc);
        for seg in reclaimed {
            super::bind::reset(seg as *mut c_void, 0, 0);
            clear_segment_state(seg);
        }
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(primary as usize))) {
            s.allocator = alloc;
            s.bound_pso = bound_pso;
            s.depth_bias = depth_bias;
            s.index_buffer = index_buffer;
            s.depth_bounds = depth_bounds;
            s.predication = predication;
            s.draw_predicate = draw_predicate.filter(|_| predication.is_some());
            s.pred_buf = pred_buf_obj;
            s.pred_in_state = false;
            s.occlusion = Occlusion { active: None, ended_early };
            s.written_ranges.clear();
            s.clear_sparse_writes();
            s.clear_write64_signals();
            s.clear_waits();
            s.cpu_writes.clear();
            s.bindings = bindings_copy;
        }
        primary
    } else {
        this
    };

    let ext_closed = *(target as *const usize);
    let init_pso = super::pso_async::resolve_for_bind(bound_pso);
    super::shader_code::taken_on(alloc, init_pso);
    let reset_hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(target, PC_RESET)(target, alloc, init_pso);
    real::preserve_vtable(target, ext_closed);
    if reset_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Reset after kickoff failed ({reset_hr:#x})"));
    }
    super::physical_directory::recordings::ended(target as usize);
    super::depth_state::reset(target as usize);
    super::gpu_timing::opened(target, bound_pso);
    super::material_gate_views::prepass::native_reset(target as usize);
    super::residency::list_reset(target as usize);
    super::depth_uav::list_reset(target as usize);
    let bindings = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(target as usize))).map(|s| s.bindings.clone());
    if let Some(b) = bindings {
        restore_bindings(target, &b, ty);
    }

    // Reset() cleared the PC list's native RSSetDepthBias state to (0, 0, 0) and bound init_pso.
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
        s.native_bias = DepthBias::ZERO;
        s.using_fallback_pso = false;
        s.pso_dirty = false;
        s.helper = Default::default();
    }
    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT {
        let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
        let mut applied_native = false;
        if let Some((true, static_bias)) = info {
            let target_bias = depth_bias.unwrap_or(static_bias);
            if target_bias.is_zero() {
                applied_native = true;
            } else if try_native_depth_bias(target, target_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                    s.native_bias = target_bias;
                }
                applied_native = true;
            }
        } else if bound_pso == 0 {
            if let Some(b) = depth_bias {
                if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(target, b) {
                    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                        s.native_bias = b;
                    }
                    applied_native = true;
                }
            }
        }
        if depth_bias.is_some() && !applied_native {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        if let Some(ref view) = index_buffer {
            set_index_buffer(target, Some(view));
        }
        if let Some((min, max)) = depth_bounds {
            pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(target, PC_OM_SET_DEPTH_BOUNDS)(target, min, max);
        }
    }
    if let (Some(op), Some(pb)) = (predication, pred_buf) {
        transition(target, pb, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(target, PC_SET_PREDICATION)(target, pb, 0, op);
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
            s.pred_in_state = true;
        }
    }
}

/// command list #94 RSSetDepthBiasX(depthBias, depthBiasClamp, slopeScaledDepthBias):
/// dynamic depth bias override. Uses `ID3D12GraphicsCommandList9::RSSetDepthBias` natively
/// when `D3D12_FEATURE_D3D12_OPTIONS16::DynamicDepthBiasSupported` and the bound PSO support
/// it (preserving exact `f32` bias with zero PSO derivation), falling back to the patched-PSO
/// cache (`device_x::depth_bias_pso`) otherwise.
pub(super) unsafe extern "system" fn set_depth_bias_x(this: *mut c_void, depth_bias: f32, depth_bias_clamp: f32, slope_scaled_depth_bias: f32) {
    static NON_FINITE: AtomicU64 = AtomicU64::new(0);
    let clean = |v: f32| {
        if v.is_finite() {
            v
        } else {
            if once(&NON_FINITE, 4) {
                crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – non-finite value {v} clamped to 0.0");
            }
            0.0
        }
    };
    let bias = DepthBias::new(clean(depth_bias), clean(depth_bias_clamp), clean(slope_scaled_depth_bias));
    let key = this as usize;

    let (bound_pso, is_bundle) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        (s.bound_pso, s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE))
    };

    let can_native = if is_bundle {
        false
    } else if bound_pso != 0 {
        matches!(device_x::base_depth_bias_info(bound_pso), Some((true, _)))
    } else {
        device_of(this).is_some_and(|dev| device_x::dynamic_depth_bias_supported(dev.as_raw()))
    };

    if can_native {
        let target = if bias.is_zero() { DepthBias::ZERO } else { bias };
        let (need_restore_base, need_native_call) = {
            let mut g = LISTS.lock().unwrap();
            let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
            s.depth_bias = if bias.is_zero() { None } else { Some(bias) };
            s.pso_dirty = false;
            let restore = s.using_fallback_pso && s.bound_pso != 0;
            s.using_fallback_pso = false;
            (if restore { Some(s.bound_pso) } else { None }, target != s.native_bias)
        };
        if let Some(base) = need_restore_base {
            let resolved = super::pso_async::resolve_wait_at_bind(base);
            set_native_pso(this, resolved);
        }
        if !need_native_call {
            return;
        }
        if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
            return;
        }
        // ID3D12GraphicsCommandList9 unavailable: fall through to patched-PSO fallback.
        if !bias.is_zero() {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        return;
    }

    let restore = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        if bias.is_zero() {
            let was_active = s.depth_bias.take().is_some();
            let was_fallback = std::mem::take(&mut s.using_fallback_pso);
            s.pso_dirty = false;
            if (was_active || was_fallback) && s.bound_pso != 0 {
                Some(s.bound_pso)
            } else {
                None
            }
        } else {
            if s.depth_bias != Some(bias) || !s.using_fallback_pso {
                s.depth_bias = Some(bias);
                s.pso_dirty = true;
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some(base) = restore {
        let resolved = super::pso_async::resolve_wait_at_bind(base);
        set_native_pso(this, resolved);
    }
}

/// Apply a pending fallback `RSSetDepthBiasX` override before a draw call on `list`.
pub(super) unsafe fn prepare_draw(list: *mut c_void) {
    if !DEPTH_BIAS_USED.load(Ordering::Acquire) {
        return;
    }
    let (bound_pso, bias) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if !s.pso_dirty {
            return;
        }
        s.pso_dirty = false;
        (s.bound_pso, s.depth_bias)
    };
    if bound_pso == 0 {
        return;
    }
    let (target, is_fallback) = match bias {
        Some(b) => {
            if super::pso_async::passive_bind() {
                super::pso_async::resolve_wait(bound_pso);
            }
            static MISS: AtomicU64 = AtomicU64::new(0);
            let Some(dev) = device_of(list) else { return };
            match device_x::depth_bias_pso(dev.as_raw(), bound_pso, b) {
                Some(derived) => (derived, true),
                None => {
                    if once(&MISS, 4) {
                        crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – could not derive pipeline for {bound_pso:#x}, using base pipeline");
                    }
                    return;
                }
            }
        }
        None => (bound_pso, false),
    };
    let resolved = super::pso_async::resolve_wait(target);
    if is_fallback && super::real::is_standin_pso(resolved) {
        // a derived pipeline still being built (or failed) has the title's root signature, not
        // the ABI's bound for this draw: the base pipeline stays, and the next draw tries again
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
            s.pso_dirty = true;
        }
        return;
    }
    set_native_pso(list, resolved);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.using_fallback_pso = is_fallback;
    }
}

// ---------------------------------------------------------------------------------------
// index buffers & DrawIndexedX
// ---------------------------------------------------------------------------------------

/// a console GPU address as the PC's (unified memory: the window buffer's)
/// A GPU address the title gave: its memory's PC address when it is the title's memory, else
/// as it is (a PC resource's GPU address – it may lie in a window's range by number).
fn pc_va(va: u64) -> u64 {
    if !crate::gpu::uma::get().is_some_and(|u| u.is_title_address(va as usize)) {
        return va;
    }
    real::uma_va_of(va as usize).unwrap_or(va)
}

/// `pc_va` for an index or vertex buffer of `bytes` bytes bound on `list`. A retail title also
/// binds buffers in its ordinary memory (a private allocation outside the unified-memory
/// windows – on the console the GPU reads all of the title's memory). The PC's GPU has no
/// address for that (IASetIndexBuffer of it: debug layer message 725, then the device hangs,
/// 0x887a0006), so the bytes are copied into the list's upload memory as they are at the bind;
/// what the title writes there afterwards is not seen.
///
/// A buffer that stays bound (`mirror`) gets one block of unified memory for good – its
/// *mirror*, the same for every bind of that address and size – and each draw copies the part
/// it reads there first (`sync_title_buffers`), as the title wrote it by then. (A copy of the
/// whole buffer at every bind filled unified memory within two minutes of a retail title's
/// menus, and missed what the title wrote after the bind.)
unsafe fn gpu_va(list: *mut c_void, va: u64, bytes: usize, what: &str, mirror: bool) -> u64 {
    let t = pc_va(va);
    if t != va || va == 0 || bytes == 0 {
        return t;
    }
    // in a window by number: a PC resource's GPU address
    if crate::gpu::uma::get().is_none_or(|u| u.contains(va as usize)) {
        return t;
    }
    if mirror {
        if let Some(p) = mirrored(va, bytes) {
            return pc_va(p as u64);
        }
    }
    if !crate::gpu::shader::readable(va as usize, bytes) {
        return t;
    }
    if mirror {
        if let Some(p) = mirror_of(va, bytes, what) {
            return pc_va(p as u64);
        }
    }
    let p = super::bind::allocate_upload_memory(list, bytes, 16);
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if p.is_null() {
        if n < 4 {
            crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory and there is no room for a copy");
        }
        return t;
    }
    std::ptr::copy_nonoverlapping(va as *const u8, p, bytes);
    if n < 4 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – bound as a copy made now (#{n})");
    }
    pc_va(p as u64)
}

/// Ranges of the title's memory with a mirror in unified memory: (start, bytes, mirror). A
/// title binds one buffer at many offsets (a ring it fills through the frame: a retail title's
/// views all end at the buffer's end), so a mirror covers a 64 KiB-aligned range and a view
/// inside a mirrored range uses that mirror at its offset.
static MIRRORS: Mutex<Vec<(u64, usize, usize)>> = Mutex::new(Vec::new());
/// bytes of mirrors made (never freed)
static MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);

/// Where `[va, va + bytes)` is in a mirror, if it is in one.
fn mirrored(va: u64, bytes: usize) -> Option<usize> {
    let end = va.checked_add(bytes as u64)?;
    MIRRORS.lock().unwrap().iter().find(|&&(s, l, _)| va >= s && end <= s + l as u64).map(|&(s, _, m)| m + (va - s) as usize)
}

/// The mirror address of the title's buffer at `va`; its range's mirror is made (and filled
/// once) the first time.
unsafe fn mirror_of(va: u64, bytes: usize, what: &str) -> Option<usize> {
    if let Some(p) = mirrored(va, bytes) {
        return Some(p);
    }
    let mut g = MIRRORS.lock().unwrap();
    // again under the lock: two lists binding one new range at once must share a mirror (a
    // second one would never be synced – every later lookup finds the first)
    if let Some(&(s, _, m)) = g.iter().find(|&&(s, l, _)| va >= s && va + bytes as u64 <= s + l as u64) {
        return Some(m + (va - s) as usize);
    }
    let start = va & !0xffff;
    let len = ((va + bytes as u64 + 0xffff) & !0xffff) - start;
    // only as much as is the title's readable memory (the view itself is)
    let (start, len) = if crate::gpu::shader::readable(start as usize, len as usize) { (start, len as usize) } else { (va, bytes) };
    if MIRROR_BYTES.load(Ordering::Relaxed) + len as u64 > 2 << 30 {
        static FULL: AtomicBool = AtomicBool::new(false);
        if !FULL.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("D3D12.X: 2 GiB of mirrors for buffers outside unified memory – further ones ({what} at {va:#x}) are bound as a copy made at the bind, which later writes do not reach");
        }
        return None;
    }
    let u = crate::gpu::uma::get()?;
    let p = match u.has_sparse().then(|| super::sparse::alloc(len, 0x1_0000)).flatten() {
        Some((p, _)) => p,
        None => u.alloc(len, 0x1_0000)? as usize,
    };
    std::ptr::copy_nonoverlapping(start as *const u8, p as *mut u8, len);
    g.push((start, len, p));
    let total = MIRROR_BYTES.fetch_add(len as u64, Ordering::Relaxed) + len as u64;
    if g.len() <= 8 || g.len().is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – {start:#x}..{:#x} mirrored at {p:#x}, each draw copies what it reads ({} mirrors, {} MB)", start + len as u64, g.len(), total >> 20);
    }
    Some(p + (va - start) as usize)
}

/// Before a draw on `list`: the parts of mirrored buffers it reads are copied from the title's
/// memory. `indexed`: (first index, index count, base vertex); else `first_vertex`, `vertices`.
pub(super) unsafe fn sync_title_buffers(list: *mut c_void, indexed: Option<(u32, u32, i32)>, first_vertex: u32, vertices: u32, instances: u32) {
    if MIRROR_BYTES.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (vbs, ib) = {
        let g = LISTS.lock().unwrap();
        let Some(b) = g.as_ref().and_then(|m| m.get(&(list as usize))).map(|s| &s.bindings) else { return };
        (b.vertex_titles.values().copied().collect::<Vec<_>>(), b.index_title)
    };
    let mirrors: Vec<Option<usize>> = {
        let mut v: Vec<Option<usize>> = vbs.iter().map(|&(va, bytes, _)| mirrored(va, bytes as usize)).collect();
        v.push(ib.and_then(|(va, bytes, _)| mirrored(va, bytes as usize)));
        if v.iter().all(Option::is_none) {
            return;
        }
        v
    };
    let copy = |va: u64, mirror: usize, bytes: u32, from: u64, len: u64| {
        let from = from.min(bytes as u64);
        let len = len.min(bytes as u64 - from) as usize;
        if len > 0 && crate::gpu::shader::readable(va as usize + from as usize, len) {
            // (as a copy: another thread of the title's can unmap the buffer under the read)
            super::title_textures::with_title_bytes(va as usize + from as usize, len, |now| std::ptr::copy_nonoverlapping(now.as_ptr(), (mirror + from as usize) as *mut u8, len));
        }
    };
    // the vertices read: [lo, hi)
    let (lo, hi) = match (indexed, ib) {
        (Some((first, count, base)), Some((va, bytes, format))) => {
            let size: u64 = if format == DXGI_FORMAT_R32_UINT.0 { 4 } else { 2 };
            let (from, len) = (first as u64 * size, count as u64 * size);
            if let Some(m) = mirrors[vbs.len()] {
                copy(va, m, bytes, from, len);
            }
            let n = (len.min((bytes as u64).saturating_sub(from)) / size) as usize;
            if n == 0 || !crate::gpu::shader::readable((va + from) as usize, n * size as usize) {
                (0, u64::MAX)
            } else {
                let at = (va + from) as *const u8;
                let (mut lo, mut hi) = (u32::MAX, 0u32);
                for k in 0..n {
                    let i = if size == 4 { at.add(4 * k).cast::<u32>().read_unaligned() } else { at.add(2 * k).cast::<u16>().read_unaligned() as u32 };
                    // (a strip's cut value counts as an index: whether 0xffff is one is the
                    // topology's to say, and a range too wide only copies more)
                    lo = lo.min(i);
                    hi = hi.max(i);
                }
                if lo > hi {
                    return;
                }
                ((lo as i64 + base as i64).max(0) as u64, (hi as i64 + base as i64 + 1).max(0) as u64)
            }
        }
        (Some(_), None) => (0, u64::MAX),
        (None, _) => (first_vertex as u64, first_vertex as u64 + vertices as u64),
    };
    for (k, &(va, bytes, stride)) in vbs.iter().enumerate() {
        let Some(m) = mirrors[k] else { continue };
        // a buffer stepped by instance is not told apart: with instances (callers pass more
        // than 1 for a first instance other than 0 too), all of it
        if instances > 1 || hi == u64::MAX || stride == 0 || hi.saturating_mul(stride as u64) > bytes as u64 {
            copy(va, m, bytes, 0, bytes as u64);
        } else {
            copy(va, m, bytes, lo * stride as u64, (hi - lo) * stride as u64);
        }
    }
}

unsafe fn set_index_buffer(list: *mut c_void, view: Option<&D3D12_INDEX_BUFFER_VIEW>) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_INDEX_BUFFER_VIEW)>(list, PC_IA_SET_INDEX_BUFFER);
    f(list, view.map_or(std::ptr::null(), |v| v as *const _));
}

/// The heaps a list binds for console code: a view heap in the shared shader-visible heap
/// is that one (`console_heap`), which is bound whatever the title names (a console
/// descriptor table may be in any heap); a view heap that kept its own shader-visible PC heap
/// (no shared heap) is bound itself, for its local indices; sampler heaps as they are. A view
/// heap whose views were meant for the shared heap but found no room there is CPU-only
/// (`dev_create_descriptor_heap_x` dropped the flag) and cannot be bound. While the shared
/// heap exists no other view heap replaces it: translated shaders reach memory through views
/// at fixed indices in it (`real::mem_views`).
pub(super) unsafe fn pc_heaps(n: u32, heaps: *const usize) -> Vec<Option<ID3D12DescriptorHeap>> {
    let shared = super::console_heap::global_heap();
    let mut view_heap = shared.clone();
    let mut out = Vec::new();
    for k in 0..n as usize {
        let p = *heaps.add(k) as *mut c_void;
        if p.is_null() || super::console_heap::in_global(p as usize) {
            continue;
        }
        let Some(h) = ID3D12DescriptorHeap::from_raw_borrowed(&p) else { continue };
        let desc = h.GetDesc();
        if desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV {
            if desc.Flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
                if shared.is_none() {
                    view_heap = Some(h.clone());
                } else {
                    static N: AtomicU64 = AtomicU64::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 4 {
                        crate::xlog_warn!("D3D12.X: SetDescriptorHeaps names a shader-visible view heap {p:?} outside the shared heap – the shared heap stays bound");
                    }
                }
            }
        } else {
            out.push(Some(h.clone()));
        }
    }
    out.extend(view_heap.map(Some));
    out
}

/// The shared view heap on `this` when the title bound no heap since the list's reset:
/// translated shaders reach memory through views in it (`real::mem_views`).
pub(super) unsafe fn ensure_view_heap(this: *mut c_void) {
    let bound = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize))).is_some_and(|s| !s.bindings.heaps.is_empty());
    let Some(h) = super::console_heap::global_heap() else { return };
    if bound {
        return;
    }
    let heaps = vec![h.as_raw() as usize];
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, 28)(this, 1, heaps.as_ptr());
    with_bindings(this, |b| b.heaps = heaps);
}

/// console #32 SetDescriptorHeaps (PC #28)
pub(super) unsafe extern "system" fn set_descriptor_heaps_x(this: *mut c_void, n: u32, heaps: *const usize) {
    const PC_SET_DESCRIPTOR_HEAPS: usize = 28;
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, PC_SET_DESCRIPTOR_HEAPS);
    if heaps.is_null() || super::console_heap::global_heap().is_none() {
        if !heaps.is_null() {
            let given = std::slice::from_raw_parts(heaps, n as usize).to_vec();
            with_bindings(this, |b| b.heaps = given);
        }
        return f(this, n, heaps);
    }
    let h = pc_heaps(n, heaps);
    let raw: Vec<usize> = h.iter().map(|x| x.as_ref().map_or(0, |x| x.as_raw() as usize)).collect();
    let kept = raw.clone();
    with_bindings(this, |b| b.heaps = kept);
    f(this, raw.len() as u32, raw.as_ptr())
}

const PC_IA_SET_VERTEX_BUFFERS: usize = 44;

/// command list #48 IASetVertexBuffers(start slot, count, views) (PC #44): the addresses are
/// the title's (its memory: the GPU address of that memory for the PC)
pub(super) unsafe extern "system" fn ia_set_vertex_buffers_x(this: *mut c_void, start: u32, n: u32, views: *const D3D12_VERTEX_BUFFER_VIEW) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS);
    if n == 0 {
        return f(this, start, n, views);
    }
    if views.is_null() {
        let cleared = |slot: u32| slot >= start && slot - start < n;
        with_bindings(this, |b| {
            b.vertex_buffers.retain(|&slot, _| !cleared(slot));
            b.vertex_titles.retain(|&slot, _| !cleared(slot));
        });
        return f(this, start, n, views);
    }
    let v: Vec<D3D12_VERTEX_BUFFER_VIEW> = (0..n as usize)
        .map(|k| {
            let mut v = views.add(k).read_unaligned();
            let title = v.BufferLocation;
            v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "vertex buffer", true);
            if super::present::trace_frame() {
                static N: AtomicU64 = AtomicU64::new(0);
                let i = N.fetch_add(1, Ordering::Relaxed);
                if i < 12 || (i.is_power_of_two() && i < 1 << 16) {
                    let n = (v.SizeInBytes as usize).min(256);
                    let data = (title != 0).then(|| crate::memmap::title_bytes(title as usize, n)).flatten().map(|b| b.iter().filter(|&&b| b != 0).count());
                    let win = crate::gpu::uma::get().and_then(|u| u.window_of(title as usize).map(|w| (w, u.window(w), u.is_sparse(title as usize))));
                    let mut mbi: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    let q = windows_sys::Win32::System::Memory::VirtualQuery(title as *const c_void, &mut mbi, std::mem::size_of_val(&mbi));
                    crate::xlog_info!(
                        "frame trace: vertex buffer #{i} slot {}: title {title:#x} → PC {:#x}, {} bytes, stride {}; non-zero in the first {n} bytes (CPU view): {data:?}; window (index, (base, size), sparse) {win:x?}; memory state {:#x} protect {:#x} (query {q})",
                        start + k as u32,
                        v.BufferLocation,
                        v.SizeInBytes,
                        v.StrideInBytes,
                        mbi.State,
                        mbi.Protect
                    );
                }
            }
            v
        })
        .collect();
    let kept = v.clone();
    let titles: Vec<(u64, u32, u32)> = (0..n as usize).map(|k| { let t = views.add(k).read_unaligned(); (t.BufferLocation, t.SizeInBytes, t.StrideInBytes) }).collect();
    with_bindings(this, |b| {
        for (k, view) in kept.into_iter().enumerate() {
            b.vertex_buffers.insert(start + k as u32, view);
            b.vertex_titles.insert(start + k as u32, titles[k]);
        }
    });
    f(this, start, n, v.as_ptr())
}

/// command list #47 IASetIndexBuffer(view) (PC #43): the address is the title's
pub(super) unsafe extern "system" fn ia_set_index_buffer_x(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
    if view.is_null() {
        s.index_buffer = None;
        s.bindings.index_title = None;
        drop(g);
        return set_index_buffer(this, None);
    }
    let mut v = view.read_unaligned();
    s.bindings.index_title = Some((v.BufferLocation, v.SizeInBytes, v.Format.0));
    drop(g);
    v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "index buffer", true);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        s.index_buffer = Some(v);
    }
    set_index_buffer(this, Some(&v));
}

/// command list #81 DrawIndexedX(index address, index format, index count, flags): one
/// instance of `count` indices read from that address; the bound index buffer stays
pub(super) unsafe extern "system" fn draw_indexed_x(this: *mut c_void, va: u64, format: DXGI_FORMAT, count: u32, _flags: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    let stride = match format {
        DXGI_FORMAT_R16_UINT => 2,
        DXGI_FORMAT_R32_UINT => 4,
        f => return warn(&N, "DrawIndexedX", format_args!("index format {}", f.0)),
    };
    if count == 0 {
        return;
    }
    if super::bind::put_off(this, Deferred::DrawIndexedX(va, format, count)) {
        return;
    }
    let immediate = D3D12_INDEX_BUFFER_VIEW { BufferLocation: va, SizeInBytes: count.saturating_mul(stride), Format: format };
    let ia = ia_live::prepare(this, true, Some(immediate));
    let Some(list) = super::bind::prepare_indexed_draw(this, count, 1, 0, 0, 0) else {
        return;
    };
    if ia.gathered() {
        return super::bind::emit_prepared_indexed_draw(&list, count, 1, 0, 0, 0);
    }
    let bound = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).and_then(|s| s.index_buffer));
    sync_title_buffers(this, None, 0, 0, 2);
    let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: gpu_va(this, va, count.saturating_mul(stride) as usize, "DrawIndexedX indices", false), SizeInBytes: count.saturating_mul(stride), Format: format };
    set_index_buffer(this, Some(&view));
    super::bind::emit_prepared_indexed_draw(&list, count, 1, 0, 0, 0);
    set_index_buffer(this, bound.as_ref());
}

// ---------------------------------------------------------------------------------------
// nested occlusion queries & Reset
// ---------------------------------------------------------------------------------------

/// outer queries ended early, and the result they resolve to
static VISIBLE: Mutex<Option<HashMap<Query, u64>>> = Mutex::new(None);
/// what such an OCCLUSION query reports (BINARY_OCCLUSION: 1)
const VISIBLE_SAMPLES: u64 = 0x1_0000;

/// what every OCCLUSION query reports with `occlusion_visible` (CGBJ's value)
const ALWAYS_VISIBLE_SAMPLES: u64 = 1 << 20;

/// Every occlusion query resolves as visible (2^20 samples; BINARY_OCCLUSION: 1), as in
/// CGBJ/cool_fox f4b48ad crates/victory-gpu/src/list.rs:1484-1515: the PC's real count depends
/// on a depth buffer that is not yet the console's, and a query that reads 0 hides whatever the
/// title gates on it (sun, lamps, flares). `XCL_OCCLUSION_VISIBLE=0`: the PC's real results.
fn occlusion_visible() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_OCCLUSION_VISIBLE").map_or(true, |v| v.trim() != "0");
        crate::xlog_info!("D3D12.X: occlusion queries {} (XCL_OCCLUSION_VISIBLE)", if on { "all resolve as visible" } else { "report the PC's real results" });
        on
    })
}

fn is_occlusion(ty: D3D12_QUERY_TYPE) -> bool {
    ty == D3D12_QUERY_TYPE_OCCLUSION || ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION
}

type QueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);

/// command list #10 Reset(allocator, initial pipeline): updates allocator/pipeline and clears
/// per-recording state (queries, index buffer, depth bias, pending waits, written ranges)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn reset_x(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    // A failed native Reset leaves the old recording alive. Do not discard its tracked state or
    // transfer its allocations until the driver accepts the new recording.
    let ext = *(this as *const usize);
    let resolved = super::pso_async::resolve_for_bind(pso);
    super::shader_code::taken_on(alloc, resolved);
    super::shader_code::bundle_takes(this as usize, resolved);
    let hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, alloc, resolved);
    real::preserve_vtable(this, ext);
    if hr < 0 {
        return hr;
    }

    super::physical_directory::recordings::ended(this as usize);
    super::gpu_timing::opened(this, pso);
    super::material_gate_views::prepass::native_reset(this as usize);
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        s.allocator = alloc;
        s.bound_pso = pso;
        s.depth_bias = None;
        s.native_bias = DepthBias::ZERO;
        s.pso_dirty = false;
        s.using_fallback_pso = false;
        s.index_buffer = None;
        s.depth_bounds = None;
        s.helper = Default::default();
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.clear_sparse_writes();
        s.clear_write64_signals();
        s.clear_waits();
        s.cpu_writes.clear();
        s.predication = None;
        s.draw_predicate = None;
        s.pred_in_state = false;
        s.bindings = Bindings::default();
        s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
    };
    super::title_textures::list_reset(this as usize);
    super::residency::list_reset(this as usize);
    super::depth_uav::list_reset(this as usize);
    super::depth_state::reset(this as usize);
    super::bind::reset(this, alloc, pso);
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(this, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
    hr
}

/// the D3D12_RESOURCE_STATES bits the PC knows
const PC_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// #26 ResourceBarrier(count, barriers): console barriers (D3D12_RESOURCE_BARRIER, 32 bytes) as
/// PC ones – state bits the PC does not know are dropped (logged: the console's own), a
/// transition left with nothing to do goes, aliasing barriers name placed textures
pub(super) unsafe extern "system" fn resource_barrier_x(this: *mut c_void, n: u32, barriers: *const [u32; 8]) {
    static N: AtomicU64 = AtomicU64::new(0);
    if barriers.is_null() || n == 0 {
        if n != 0 {
            super::depth_state::invalidate(this as usize);
        }
        return;
    }
    let mut out: Vec<[u32; 8]> = Vec::with_capacity(n as usize + 1);
    let mut unknown = false;
    // a UAV barrier for all resources is in `out` / a transition from or to UNORDERED_ACCESS
    let (mut any_uav, mut uav_transition) = (false, false);
    // copies to the shadow textures of depth buffers that follow these barriers (`depth_uav`)
    let mut depth_copies = Vec::new();
    for k in 0..n as usize {
        let mut b = barriers.add(k).read_unaligned();
        super::depth_state::guest_legacy_raw(this as usize, &b);
        let (ty, flags) = (b[0], b[1]);
        if super::present::trace_frame() && ty == 0 {
            let r = (b[2] as u64 | (b[3] as u64) << 32) as usize;
            if super::present::is_watched(r) {
                static W: AtomicU64 = AtomicU64::new(0);
                if W.fetch_add(1, Ordering::Relaxed) < 24 {
                    crate::xlog_info!("frame trace: ResourceBarrier on a watched target {r:#x}: subresource {:#x}, {:#x} → {:#x}, flags {flags:#x}", b[4], b[5], b[6]);
                }
            }
        }
        if ty > 2 || flags > 3 {
            unknown = true;
            if N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console barrier type {ty} flags {flags:#x} dropped ({:x?})", b);
            }
            continue;
        }
        if ty == 1 {
            // Keep the identity of textures that really share placed GPU memory.
            // Committed resources still need the wildcard form on the PC.
            for words in b[2..6].chunks_exact_mut(2) {
                let resource = (words[0] as u64 | (words[1] as u64) << 32) as usize;
                if resource != 0 && !super::vram::is_placed_texture(resource) {
                    words.fill(0);
                }
            }
        }
        if ty == 2 {
            // UAV (resource +8): translated shaders reach title memory through XCL's memory
            // window buffers – other resources over the same memory – which a UAV barrier for
            // the title's resource alone does not order (AMD drivers wait and flush for any UAV
            // barrier, NVIDIA's need not): a UAV barrier for all resources, once per batch
            b[2..4].fill(0);
            if any_uav {
                continue;
            }
            any_uav = true;
        }
        if ty == 0 {
            // transition: resource (+8), subresource (+16), before (+20), after (+24)
            let (before, after) = (b[5], b[6]);
            if (before | after) & !PC_STATES != 0 && N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console state bits {:#x} → {:#x} dropped", before & !PC_STATES, after & !PC_STATES);
            }
            b[5] &= PC_STATES;
            b[6] &= PC_STATES;
            // VERTEX_AND_CONSTANT_BUFFER / INDEX_BUFFER are buffer states on the PC; the console
            // takes them for textures too (Minecraft moves a texture to a generic-read
            // combination) – the PC list then fails to close
            if (b[5] | b[6]) & BUFFER_ONLY_STATES != 0 && !is_buffer((b[2] as u64 | (b[3] as u64) << 32) as usize) {
                b[5] &= !BUFFER_ONLY_STATES;
                b[6] &= !BUFFER_ONLY_STATES;
            }
            if b[5] == b[6] {
                continue;
            }
            // a depth buffer the console also gives unordered access: its shadow texture is
            // brought in step, and the PC depth buffer takes a state it has
            // UAV writes before (or after) the transition: the same holds for the window buffers
            uav_transition |= (b[5] | b[6]) & UNORDERED_ACCESS != 0;
            if (super::depth_uav::active() || (b[5] | b[6]) & UNORDERED_ACCESS != 0) && !super::depth_uav::on_transition(this, &mut b, &mut depth_copies) {
                continue;
            }
        }
        out.push(b);
    }
    if uav_transition && !any_uav {
        out.push([2, 0, 0, 0, 0, 0, 0, 0]);
    }
    // the states uploads into textures kept in step start from once this list has run
    if super::title_textures::any() {
        let t: Vec<(usize, u32, D3D12_RESOURCE_STATES)> =
            out.iter().filter(|b| b[0] == 0).map(|b| ((b[2] as u64 | (b[3] as u64) << 32) as usize, b[4], D3D12_RESOURCE_STATES(b[6] as i32))).collect();
        super::title_textures::note_transitions(this as usize, &t);
    }
    let named = out.iter().flat_map(|b| {
        let r = |i: usize| (b[i] as u64 | (b[i + 1] as u64) << 32) as usize;
        match b[0] {
            0 => [r(2), 0],
            1 => [r(2), r(4)],
            _ => [0, 0],
        }
    });
    let uavs = (0..n as usize).filter_map(|k| {
        let b = barriers.add(k).read_unaligned();
        (b[0] == 2 && b[1] <= 3).then(|| (b[2] as u64 | (b[3] as u64) << 32) as usize)
    });
    super::residency::note_list_resources(this as usize, named.chain(uavs));
    if super::residency::eviction_enabled() && out.iter().any(|b| b[0] == 1 && ((b[2] | b[3]) == 0 || (b[4] | b[5]) == 0)) {
        super::residency::note_list_untracked(this as usize);
    }
    if !out.is_empty() {
        if (any_uav || uav_transition) && super::bind::indirect_world_trace() {
            super::bind::indirect_world_event(this as usize, format_args!("ResourceBarrierX emitted global UAV, explicit {any_uav}, transition {uav_transition}"));
        }
        super::gpu_timing::class(this, super::gpu_timing::Kind::Barrier, || {
            pc::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(this, PC_RESOURCE_BARRIER)(this, out.len() as u32, out.as_ptr());
        });
        super::depth_state::legacy_raw(this as usize, &out);
    }
    if !depth_copies.is_empty() {
        super::depth_uav::after_barriers(this, depth_copies);
    }
    // R9G9B9E5 views read through copies of the R32_UINT textures whose writes these end (`e5_copy`)
    super::e5_copy::after_barriers(this, &out, barriers, n);
    if unknown {
        super::depth_state::invalidate(this as usize);
    }
}

/// D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | INDEX_BUFFER: valid for buffers only
const BUFFER_ONLY_STATES: u32 = 0x1 | 0x2;

/// D3D12_RESOURCE_STATE_UNORDERED_ACCESS
const UNORDERED_ACCESS: u32 = 0x8;

/// Whether `res` is a buffer (null / unknown: taken as one – its states are left alone).
unsafe fn is_buffer(res: usize) -> bool {
    let r = res as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&r).is_none_or(|x| x.GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
}

/// DMA list #9 Close: as on the PC; a failure is logged (the title stops on it)
pub(super) unsafe extern "system" fn dma_close_x(this: *mut c_void) -> i32 {
    super::gpu_timing::before_close(this);
    let ext = *(this as *const usize);
    let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, 9)(this);
    real::preserve_vtable(this, ext);
    if hr < 0 {
        let ty = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_GET_TYPE)(this);
        crate::xlog_warn!("D3D12.X: DMA list Close → {:#x} (PC list type {ty})", hr as u32);
    }
    hr
}

/// a copy location with the offset of a buffer placed below its console address added
unsafe fn pc_location(l: *const D3D12_TEXTURE_COPY_LOCATION) -> Option<D3D12_TEXTURE_COPY_LOCATION> {
    let mut l = std::ptr::read(l.as_ref()?);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        l.Anonymous.PlacedFootprint.Offset += real::buffer_delta(transmute_copy::<_, usize>(&l.pResource));
    }
    Some(l)
}

/// CopyBufferRegion(dst, dst offset, src, src offset, bytes) (PC #15; DMA list #11)
/// `XCL_TRACE_FRAME`: a GPU copy into title memory that a texture kept in step lies in (the
/// texture does not see GPU writes)
fn note_gpu_write(what: &str, va: usize, n: u64) {
    if !super::present::trace_frame() {
        return;
    }
    let hit = super::title_textures::overlapping(va, n as usize);
    if hit.is_empty() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k < 16 || k.is_power_of_two() {
        crate::xlog_info!("frame trace: {what} #{k} writes {n:#x} bytes at {va:#x} – into title textures {hit:x?}");
    }
}

pub(super) unsafe extern "system" fn copy_buffer_region_x(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    super::residency::note_list_resources(this as usize, [dst, src]);
    note_buffer_write(this as usize, dst, dst_off, n, true);
    if super::dsv_trace::on() {
        let at = |r: usize, off: u64| real::title_address(r).map(|a| a + off as usize);
        super::dsv_trace::memory("CopyBufferRegion", at(dst, dst_off), at(src, src_off), n);
    }
    if super::present::trace_frame() {
        if let Some(a) = real::title_address(dst) {
            note_gpu_write("CopyBufferRegion", a + dst_off as usize, n);
        }
    }
    if super::bind::indirect_world_trace() {
        let (from, to) = (super::bind::resource_title(src, src_off), super::bind::resource_title(dst, dst_off));
        super::bind::indirect_world_event(this as usize, format_args!("CopyBufferRegion source {from:x?} destination {to:x?}+{n:#x}"));
    }
    if !native_buffer::intercept_copy(this as usize, dst, dst_off, src, src_off, n) {
        copy_buffer_region(this, dst, dst_off + real::buffer_delta(dst), src, src_off + real::buffer_delta(src), n)
    }
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): copies out of it – that
/// object, or another at its title address (the first 16)
unsafe fn trace_watched_copy(what: &str, dst: usize, src: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let addr = |r: usize| real::title_address(r).or_else(|| super::title_textures::address(r));
    if !super::present::is_watched(src) && (addr(src).is_none() || addr(src) != addr(w)) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return;
    }
    if !super::present::is_presented(dst) {
        super::present::follow(dst);
    }
    let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0) });
    crate::xlog_info!(
        "frame trace: {what} out of a watched target {src:#x} {:?} into {dst:#x} {:?}{}",
        d(src),
        d(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

pub(crate) mod texture_ops;

/// CopyTextureRegion(dst, x, y, z, src, box) (PC #16; DMA list #12)
pub(super) unsafe extern "system" fn copy_texture_region_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX) {
    copy_texture_region_flags_x(this, dst, x, y, z, src, bx, 0)
}

pub(super) unsafe extern "system" fn copy_texture_region_flags_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX, flags: u32) {
    if super::dsv_trace::on() {
        let r = |l: *const D3D12_TEXTURE_COPY_LOCATION| l.as_ref().map_or(0, |l| transmute_copy::<_, usize>(&l.pResource));
        super::dsv_trace::transfer("CopyTextureRegion", r(dst), r(src));
    }
    if texture_ops::intercept(this, texture_ops::RecordingPoint::Adopted, || texture_ops::Operation::copy(dst, [x,y,z], src, bx, flags))
        .unwrap_or_else(|e| hard_stop(e)) { return; }
    super::depth_uav::on_copy_region(this, dst, src);
    super::depth_uav::note_op(this);
    if let Some(d) = dst.as_ref() {
        let r: usize = transmute_copy(&d.pResource);
        super::present::note_use(r, super::present::USE_COPY);
        if super::target_dump::recording() || (super::present::writer_trace_active() && super::present::is_writer_target(r)) {
            let source = src.as_ref().map_or(0, |s| transmute_copy::<_, usize>(&s.pResource));
            trace_plane_transfer(this, r, source, "texture copy");
        }
        super::title_textures::note_gpu_write(r);
        super::vram::note_gpu_write(r);
        if d.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
            let f = d.Anonymous.PlacedFootprint;
            note_buffer_write(this as usize, r, f.Offset, f.Footprint.RowPitch as u64 * f.Footprint.Height as u64 * f.Footprint.Depth as u64, false);
        }
        if super::present::trace_frame() {
            if let Some(va) = super::title_textures::address(r) {
                static N: AtomicU64 = AtomicU64::new(0);
                let k = N.fetch_add(1, Ordering::Relaxed);
                if k < 8 || k.is_power_of_two() {
                    // the source: a buffer footprint – where its bytes are for the CPU, and whether
                    // there is anything there
                    let src_info = src.as_ref().map(|sl| {
                        let sr: usize = transmute_copy(&sl.pResource);
                        let rr = sr as *mut c_void;
                        let heap = ID3D12Resource::from_raw_borrowed(&rr).and_then(|x| {
                            let mut hp = D3D12_HEAP_PROPERTIES::default();
                            let mut hf = D3D12_HEAP_FLAGS::default();
                            x.GetHeapProperties(Some(&mut hp), Some(&mut hf)).ok().map(|_| hp.Type.0)
                        });
                        let (ty, fp) = (sl.Type.0, if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT { Some(sl.Anonymous.PlacedFootprint) } else { None });
                        let cpu = real::title_address(sr);
                        let nonzero = match (cpu, fp) {
                            (Some(a), Some(f)) => {
                                let at = a + f.Offset as usize;
                                let n = (f.Footprint.RowPitch as usize * 4).min(4096);
                                crate::memmap::title_bytes(at, n).map(|b| b.iter().filter(|&&b| b != 0).count())
                            }
                            _ => None,
                        };
                        format!("src {sr:#x} type {ty} heap {heap:?} title address {:?} delta {:#x} footprint {:?} nonzero in the first rows {nonzero:?}", cpu.map(|a| format!("{a:#x}")), real::buffer_delta(sr), fp.map(|f| (f.Offset, f.Footprint.Format.0, f.Footprint.Width, f.Footprint.Height, f.Footprint.RowPitch)))
                    });
                    crate::xlog_info!("frame trace: CopyTextureRegion #{k} into title texture {r:#x} (at {va:#x}), subresource {}; {src_info:?}", d.Anonymous.SubresourceIndex);
                }
            }
        }
    }
    if super::present::trace_frame() {
        if let (Some(d), Some(sl)) = (dst.as_ref(), src.as_ref()) {
            trace_watched_copy("CopyTextureRegion", transmute_copy(&d.pResource), transmute_copy(&sl.pResource));
            // footprints the PC's copy rules do not allow (row pitch: 256 bytes, offset: 512)
            if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
                let f = sl.Anonymous.PlacedFootprint;
                if f.Footprint.RowPitch % 256 != 0 || f.Offset % 512 != 0 {
                    // each (format, pitch alignment, offset alignment) once
                    static SEEN: Mutex<Vec<(i32, u32, u64)>> = Mutex::new(Vec::new());
                    let key = (f.Footprint.Format.0, f.Footprint.RowPitch % 256, f.Offset % 512);
                    let new = {
                        let mut g = SEEN.lock().unwrap();
                        let n = !g.contains(&key) && g.len() < 24;
                        if n {
                            g.push(key);
                        }
                        n
                    };
                    if new {
                        crate::xlog_info!(
                            "frame trace: CopyTextureRegion from a footprint the PC does not take: offset {:#x}, format {}, {}x{}x{}, row pitch {} (box {:?}, to ({x}, {y}, {z}))",
                            f.Offset,
                            f.Footprint.Format.0,
                            f.Footprint.Width,
                            f.Footprint.Height,
                            f.Footprint.Depth,
                            f.Footprint.RowPitch,
                            bx.as_ref().map(|b| (b.left, b.top, b.right, b.bottom))
                        );
                    }
                }
            }
        }
    }
    let (Some(d), Some(s)) = (pc_location(dst), pc_location(src)) else { return };
    super::residency::note_list_resources(this as usize, [transmute_copy(&d.pResource), transmute_copy(&s.pResource)]);
    super::gpu_timing::class(this, super::gpu_timing::Kind::Transfer, || {
        pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(this, PC_COPY_TEXTURE_REGION)(this, &d, x, y, z, &s, bx);
    });
    std::mem::forget((d, s)); // borrowed resource pointers
}

const PC_CLEAR_DEPTH_STENCIL_VIEW: usize = 47;

/// command list #51 ClearDepthStencilView(dsv, flags, depth, stencil, rects, rect list) (PC #47)
pub(super) unsafe extern "system" fn clear_depth_stencil_x(this: *mut c_void, dsv: usize, flags: u32, depth: f32, stencil: u8, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    super::depth_uav::on_clear(this, dsv, flags);
    super::depth_uav::note_op(this);
    super::residency::note_list_untracked(this as usize);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        static W: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        let watched = WATCHED_DSVS.lock().unwrap().contains(&dsv);
        if k < 8 || k.is_power_of_two() || (watched && W.fetch_add(1, Ordering::Relaxed) < 12) {
            crate::xlog_info!(
                "frame trace: ClearDepthStencilView #{k} {dsv:#x}{}: flags {flags:#x} (1 depth, 2 stencil), depth {depth}, stencil {stencil}, {n} rects",
                if watched { " (bound by a draw reading a watched target)" } else { "" }
            );
        }
    }
    super::gpu_timing::class(this, super::gpu_timing::Kind::Clear, || {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u32, f32, u8, u32, *const windows::Win32::Foundation::RECT)>(this, PC_CLEAR_DEPTH_STENCIL_VIEW)(this, dsv, flags, depth, stencil, n, rects)
    });
    super::depth_capture_views::note_clear(this, dsv, flags, depth, n);
}

const PC_CLEAR_UAV_UINT: usize = 49;
const PC_CLEAR_UAV_FLOAT: usize = 50;

/// The handles a ClearUnorderedAccessView* of console code goes to the PC with.
#[derive(Debug, PartialEq)]
enum UavClearHandles {
    /// both are console slots: their PC UAVs (GPU: in the heap lists bind; CPU: in the
    /// title's CPU-only heap)
    Console { gpu: u64, cpu: u64 },
    /// neither is a console slot: PC handles already
    Pc,
    /// one console slot and one not, or a slot with no shader-visible UAV
    Unusable,
}

fn uav_clear_handles(gpu: Option<super::console_heap::Slot>, cpu: Option<super::console_heap::Slot>) -> UavClearHandles {
    match (gpu, cpu) {
        (None, None) => UavClearHandles::Pc,
        (Some(g), Some(c)) if !g.sampler && !c.sampler && g.pc_gpu_uav != 0 => UavClearHandles::Console { gpu: g.pc_gpu_uav, cpu: c.pc_uav },
        _ => UavClearHandles::Unusable,
    }
}

/// command list #53 / #54 ClearUnorderedAccessViewUint / Float(GPU handle, CPU handle,
/// resource, values[4], rects, rect list) (PC #49 / #50): console handles are addresses of
/// console heap slots (heap #9 / #10), not PC descriptors
unsafe fn clear_uav(this: *mut c_void, slot: usize, gpu: u64, cpu: u64, res: usize, values: *const c_void, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    let (gpu, cpu) = match uav_clear_handles(super::console_heap::slot(gpu), super::console_heap::slot(cpu)) {
        UavClearHandles::Console { gpu, cpu } => (gpu, cpu),
        UavClearHandles::Pc => (gpu, cpu),
        UavClearHandles::Unusable => {
            static N: AtomicU64 = AtomicU64::new(0);
            if once(&N, 1) {
                crate::xlog_warn!("D3D12.X: ClearUnorderedAccessView with GPU handle {gpu:#x} / CPU handle {cpu:#x}: not both a console slot with a shader-visible UAV – not done");
            }
            return;
        }
    };
    pc::<unsafe extern "system" fn(*mut c_void, u64, u64, usize, *const c_void, u32, *const windows::Win32::Foundation::RECT)>(this, slot)(this, gpu, cpu, res, values, n, rects)
}

pub(super) unsafe extern "system" fn clear_uav_uint_x(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const u32, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    clear_uav(this, PC_CLEAR_UAV_UINT, gpu, cpu, res, values as *const c_void, n, rects)
}

pub(super) unsafe extern "system" fn clear_uav_float_x(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const f32, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    clear_uav(this, PC_CLEAR_UAV_FLOAT, gpu, cpu, res, values as *const c_void, n, rects)
}

const PC_RESOLVE_SUBRESOURCE: usize = 19;
const PC_RESOLVE_SUBRESOURCE_REGION: usize = 64;

fn trace_plane_transfer(list: *mut c_void, dst: usize, src: usize, op: &'static str) {
    super::target_dump::record_write(list as usize, dst, op);
    if super::present::writer_trace_active() && super::present::is_writer_target(dst) {
        let backing = super::residency::describe_backing(src).unwrap_or_else(|| "untracked".into());
        super::present::record_plane_write(list as usize, dst, op, 0, vec![format!("source {src:#x} {backing}")]);
    }
}

/// `XCL_TRACE_FRAME`: a resolve (first 12 and powers of two) – what goes where
unsafe fn trace_resolve(what: &str, dst: usize, src: usize, format: i32, extra: &str) {
    if !super::present::trace_frame() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k >= 12 && !k.is_power_of_two() {
        return;
    }
    let desc = |r: usize| {
        let p = r as *mut c_void;
        ID3D12Resource::from_raw_borrowed(&p).map(|x| {
            let d = x.GetDesc();
            (d.Width, d.Height, d.Format.0, d.SampleDesc.Count)
        })
    };
    crate::xlog_info!(
        "frame trace: {what} #{k} {src:#x} {:?} → {dst:#x} {:?}{} format {format}{extra}",
        desc(src),
        desc(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

/// command list #19 ResolveSubresource(dst, dst sub, src, src sub, format) (PC #19)
pub(super) unsafe extern "system" fn resolve_subresource_x(this: *mut c_void, dst: usize, dst_sub: u32, src: usize, src_sub: u32, format: i32) {
    super::dsv_trace::transfer("ResolveSubresource", dst, src);
    if texture_ops::intercept(this, texture_ops::RecordingPoint::Adopted, || texture_ops::Operation::resolve(dst, dst_sub, src, src_sub, format, None))
        .unwrap_or_else(|e| hard_stop(e)) { return; }
    super::depth_uav::on_copy(this, dst, dst_sub, src, src_sub);
    super::depth_uav::note_op(this);
    super::residency::note_list_resources(this as usize, [dst, src]);
    super::present::note_use(dst, super::present::USE_COPY);
    super::vram::note_gpu_write(dst);
    trace_plane_transfer(this, dst, src, "resolve");
    trace_resolve("ResolveSubresource", dst, src, format, "");
    super::gpu_timing::class(this, super::gpu_timing::Kind::Transfer, || {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u32, usize, u32, i32)>(this, PC_RESOLVE_SUBRESOURCE)(this, dst, dst_sub, src, src_sub, format)
    })
}

/// command list #108 ResolveSubresourceRegion(dst, dst sub, x, y, src, src sub, rect, format,
/// mode) (PC #64)
#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "system" fn resolve_subresource_region_x(this: *mut c_void, dst: usize, dst_sub: u32, x: u32, y: u32, src: usize, src_sub: u32, rect: *const windows::Win32::Foundation::RECT, format: i32, mode: i32) {
    super::dsv_trace::transfer("ResolveSubresourceRegion", dst, src);
    if texture_ops::intercept(this, texture_ops::RecordingPoint::Adopted, || texture_ops::Operation::resolve(dst, dst_sub, src, src_sub, format,
        Some((x,y,rect.as_ref().copied(),D3D12_RESOLVE_MODE(mode)))))
        .unwrap_or_else(|e| hard_stop(e)) { return; }
    super::depth_uav::on_copy(this, dst, dst_sub, src, src_sub);
    super::depth_uav::note_op(this);
    super::residency::note_list_resources(this as usize, [dst, src]);
    super::present::note_use(dst, super::present::USE_COPY);
    super::vram::note_gpu_write(dst);
    trace_plane_transfer(this, dst, src, "region resolve");
    if super::present::trace_frame() {
        trace_resolve("ResolveSubresourceRegion", dst, src, format, &format!(" mode {mode} at ({x}, {y}) rect {:?}", rect.as_ref().map(|r| (r.left, r.top, r.right, r.bottom))));
    }
    super::gpu_timing::class(this, super::gpu_timing::Kind::Transfer, || {
        pc::<unsafe extern "system" fn(*mut c_void, usize, u32, u32, u32, usize, u32, *const windows::Win32::Foundation::RECT, i32, i32)>(this, PC_RESOLVE_SUBRESOURCE_REGION)(this, dst, dst_sub, x, y, src, src_sub, rect, format, mode)
    })
}

/// CopyResource(dst, src) (PC #17; DMA list #13): buffers placed below their console address
/// are copied as their own bytes (the PC resources also hold what lies before them)
pub(super) unsafe extern "system" fn copy_resource_x(this: *mut c_void, dst: usize, src: usize) {
    super::depth_uav::on_copy(this, dst, u32::MAX, src, u32::MAX);
    super::dsv_trace::transfer("CopyResource", dst, src);
    super::depth_uav::note_op(this);
    super::present::note_use(dst, super::present::USE_COPY);
    trace_plane_transfer(this, dst, src, "resource copy");
    super::title_textures::note_gpu_write(dst);
    super::vram::note_gpu_write(dst);
    if super::present::trace_frame() {
        trace_watched_copy("CopyResource", dst, src);
    }
    let (dd, sd) = (real::buffer_delta(dst), real::buffer_delta(src));
    if dd == 0 && sd == 0 {
        super::residency::note_list_resources(this as usize, [dst, src]);
        note_buffer_write(this as usize, dst, 0, u64::MAX, true);
        return super::gpu_timing::class(this, super::gpu_timing::Kind::Transfer, || {
            pc::<unsafe extern "system" fn(*mut c_void, usize, usize)>(this, PC_COPY_RESOURCE)(this, dst, src)
        });
    }
    let n = real::console_buffer(dst).or(real::console_buffer(src)).map_or(0, |b| b.1 as u64);
    copy_buffer_region_x(this, dst, 0, src, 0, n);
}

/// command list #28 BeginQuery(heap, type, index) (PC #52)
pub(super) unsafe extern "system" fn begin_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    ia_live::query_begin(this, heap, ty, index);
    super::material_gate_views::prepass::query_seen(this as usize);
    static N: AtomicU64 = AtomicU64::new(0);
    if is_occlusion(ty) {
        let outer = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .begin((heap, index), ty);
        if let Some(((h, i), oty)) = outer {
            if once(&N, 1) {
                crate::xlog_info!("D3D12.X: nested occlusion query – the outer one ends there and reports visible, the inner one counts");
            }
            pc::<QueryFn>(this, PC_END_QUERY)(this, h, oty, i);
        }
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, index));
        }
    }
    pc::<QueryFn>(this, PC_BEGIN_QUERY)(this, heap, ty, index);
}

pub(super) unsafe extern "system" fn observe_native_begin_query(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    super::material_gate_views::prepass::query_seen(this as usize);
    pc::<QueryFn>(this, PC_BEGIN_QUERY)(this, heap, ty, index);
}

/// command list #29 EndQuery(heap, type, index) (PC #53)
pub(super) unsafe extern "system" fn end_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    ia_live::query_end(this, heap, ty, index);
    if is_occlusion(ty) {
        let forward = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .end((heap, index));
        if !forward {
            let v = if ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION { 1 } else { VISIBLE_SAMPLES };
            VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, index), v);
            return;
        }
    }
    pc::<QueryFn>(this, PC_END_QUERY)(this, heap, ty, index);
}

/// command list #30 ResolveQueryData(heap, type, first, count, destination, offset) (PC #54):
/// outer occlusion queries ended early are written over the resolved data (after it, in COPY_DEST as
/// ResolveQueryData wants the destination). Buffers placed below their console address add
/// `buffer_delta(dest)` to the destination offset.
pub(super) unsafe extern "system" fn resolve_query_data_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, first: u32, count: u32, dest: usize, offset: u64) {
    super::residency::note_list_resources(this as usize, [dest]);
    let stride = match ty {
        D3D12_QUERY_TYPE_OCCLUSION | D3D12_QUERY_TYPE_BINARY_OCCLUSION | D3D12_QUERY_TYPE_TIMESTAMP => 8,
        _ => std::mem::size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>() as u64,
    };
    note_buffer_write(this as usize, dest, offset, stride * count as u64, false);
    let visible: Vec<(u32, u64)> = if is_occlusion(ty) && occlusion_visible() {
        let v = if ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION { 1 } else { ALWAYS_VISIBLE_SAMPLES };
        (first..first.saturating_add(count)).map(|i| (i, v)).collect()
    } else {
        match VISIBLE.lock().unwrap().as_ref() {
            Some(m) if is_occlusion(ty) && !m.is_empty() => (first..first.saturating_add(count)).filter_map(|i| m.get(&(heap, i)).map(|&v| (i, v))).collect(),
            _ => Vec::new(),
        }
    };
    if native_buffer::destinations::resolve(this, heap, ty, first, count, dest, offset, stride * count as u64, &visible) {
        return;
    }
    let pc_offset = offset.saturating_add(real::buffer_delta(dest));
    type F = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    super::gpu_timing::class(this, super::gpu_timing::Kind::Transfer, || pc::<F>(this, PC_RESOLVE_QUERY_DATA)(this, heap, ty, first, count, dest, pc_offset));
    let dest = dest as *mut c_void;
    let Some(res) = ID3D12Resource::from_raw_borrowed(&dest).filter(|_| !visible.is_empty()) else { return };
    let at = res.GetGPUVirtualAddress().saturating_add(pc_offset);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = visible
        .iter()
        .flat_map(|&(i, v)| {
            let a = at + 8 * (i - first) as u64;
            [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a, Value: v as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a + 4, Value: (v >> 32) as u32 }]
        })
        .collect();
    let modes = vec![D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; params.len()];
    type W = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    pc::<W>(this, PC_WRITE_BUFFER_IMMEDIATE)(this, params.len() as u32, params.as_ptr(), modes.as_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_gpu_wait_does_not_expire_while_the_title_is_held_for_a_build() {
        let old = Instant::now() - gpu_wait_watchdog_timeout() * 2;
        let mut aw = ArmedGpuWait { val: 1, waiting_queue: 0, waiting_queue_ty: 0, wait_seq: 1, armed_at: old, after_write64: false };
        assert!(!wait_expired(&mut aw, true));
        assert!(!wait_expired(&mut aw, false), "its clock starts again when the hold ends");
        aw.armed_at = old;
        assert!(wait_expired(&mut aw, false));
    }

    /// A dispatch whose pipeline is still being built gets a segment of its own and the list goes
    /// on without waiting; the submit waits for the build and records it there. Past the
    /// per-recording budget a dispatch is skipped as before; so is every one with
    /// `XCL_DEFER_TO_SUBMIT=0`.
    #[test]
    fn a_dispatch_with_a_pipeline_still_building_waits_at_the_submit() {
        use super::super::{bind, pso_async, title_list};
        use std::time::{Duration, Instant};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
                if warp.and_then(|w| D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev)).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let raw: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, None).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let standin = 0x7E57_D101usize;
            static BUILD: AtomicBool = AtomicBool::new(false);
            pso_async::submit(
                standin,
                true,
                Box::new(|| {
                    let t = Instant::now();
                    while !BUILD.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(10) {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    0
                }),
            );
            let pc0 = raw.into_raw();
            real::adopt(pc0, real::Kind::CommandList);
            on_create(pc0, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            bind::reset(pc0, alloc.as_raw() as usize, standin);
            let wrap = title_list::wrap(pc0);
            pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(true));

            // off: skipped as before, in the list
            pso_async::set_defer_for_test(Some(false));
            let skipped = bind::counts_snapshot()[4];
            bind::dispatch(pc0, 1, 1, 1);
            let mut segs = Vec::new();
            assert!(title_list::push_segments(wrap, &mut segs));
            assert_eq!(segs.len(), 1, "put off with XCL_DEFER_TO_SUBMIT=0");
            assert!(bind::counts_snapshot()[4] > skipped, "not skipped with XCL_DEFER_TO_SUBMIT=0");

            pso_async::set_defer_for_test(Some(true));
            const DISPATCHES: usize = 34;
            for n in 1..=DISPATCHES {
                let t = Instant::now();
                bind::dispatch(*segs.last().unwrap(), 1, 1, 1);
                assert!(t.elapsed() < Duration::from_millis(700), "dispatch {n} waited for the build ({:?})", t.elapsed());
                segs.clear();
                assert!(title_list::push_segments(wrap, &mut segs));
            }
            assert_eq!(segs.len(), 1 + 2 * 32, "the list before each put-off dispatch, the dispatch, the list after it; 32 a recording");
            assert!(pso_async::is_pending(standin));

            let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(9));
            assert_eq!(close_fn(wrap), 0);
            assert!(pso_async::is_pending(standin), "Close waited for the build");

            let skipped = bind::counts_snapshot()[4];
            let build = std::thread::spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                BUILD.store(true, Ordering::Release);
            });
            record_deferred(wrap, Instant::now() + Duration::from_secs(20));
            pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(false));
            pso_async::set_defer_for_test(None);
            assert!(!pso_async::is_pending(standin), "the submit waits for the build");
            assert!(bind::counts_snapshot()[4] >= skipped + 32, "the dispatches of a build that failed are skipped, at the submit");
            build.join().unwrap();

            type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let exec: ExecFn = std::mem::transmute(*(*(queue.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
            exec(queue.as_raw(), segs.len() as u32, segs.as_ptr());
            queue.Signal(&done, 1).unwrap();
            let t = Instant::now();
            while done.GetCompletedValue() < 1 && t.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(done.GetCompletedValue(), 1, "the split list runs");

            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(2));
            assert_eq!(rel_fn(wrap), 0);
            let _ = pso_async::released(standin);
        }
    }

    /// ClearUnorderedAccessView* handles: two console slots → their PC UAVs, two PC handles as
    /// they are, anything else not usable.
    #[test]
    fn uav_clear_handles_translate_console_slots() {
        use super::super::console_heap::Slot;
        let s = |pc_uav: u64, pc_gpu_uav: u64, sampler: bool| Some(Slot { at: 0x1000, index: 1, pc_srv: 0, pc_uav, pc_gpu_uav, sampler, shared_srv: 0, k: 0 });
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), s(0x20, 0, false)), UavClearHandles::Console { gpu: 0x5010, cpu: 0x20 });
        assert_eq!(uav_clear_handles(None, None), UavClearHandles::Pc);
        // a console slot beside a PC handle, a GPU slot of a heap that is not shader visible,
        // sampler slots
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), None), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(None, s(0x20, 0, false)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0x10, 0, false), s(0x20, 0, false)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), s(0, 0, true)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0, 0, true), s(0x20, 0, false)), UavClearHandles::Unusable);
    }

    /// Console code clears a UAV with its console heap slots (heap #9 / #10 give it their
    /// addresses): the clear reaches the PC with the slots' UAVs. Before, the slot addresses
    /// went to the PC method as handles. On WARP (a software adapter).
    #[test]
    fn clear_uav_with_console_handles_clears_the_slots_view() {
        use super::super::console_heap;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            crate::gpu::recomp::runner::skip_test("no unified memory");
            return;
        }
        if !crate::hle::d3d12x::sparse::windows_ready() {
            crate::gpu::recomp::runner::skip_test("no 4 GiB sparse memory window: this adapter cannot reserve one");
            return;
        }
        unsafe {
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            }
            let dev = dev.unwrap();
            const N: u32 = 64;
            let desc = |flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4 * N as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let committed = |ty, flags, state| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc(flags), state, None, &mut r).unwrap();
                r.unwrap()
            };
            let buf = committed(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let rb = committed(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            // the title's heaps of one view slot each (SRV, UAV): a CPU-only one and a shader-visible one
            let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let heap = |flags| -> ID3D12DescriptorHeap {
                dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: flags, NodeMask: 0 }).unwrap()
            };
            let (cpu_heap, gpu_heap) = (heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE), heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: N, ..Default::default() } },
            };
            for h in [&cpu_heap, &gpu_heap] {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + inc as usize };
                dev.CreateUnorderedAccessView(&buf, None, Some(&uav), at);
            }
            let (co, go) = (cpu_heap.as_raw() as usize, gpu_heap.as_raw() as usize);
            assert!(console_heap::register(co, 1, cpu_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, 0, inc, false, false, false));
            assert!(console_heap::register(go, 1, gpu_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, gpu_heap.GetGPUDescriptorHandleForHeapStart().ptr, inc, false, false, true));
            let (console_cpu, console_gpu) = (console_heap::start(co).unwrap(), console_heap::start(go).unwrap());

            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
            l.SetDescriptorHeaps(&[Some(gpu_heap.clone())]);
            // the title's call (#53) on the adopted list, then the PC vtable back
            let raw = l.as_raw();
            real::adopt(raw, real::Kind::CommandList);
            clear_uav_uint_x(raw, console_gpu, console_cpu, buf.as_raw() as usize, [0xDEAD_BEEFu32; 4].as_ptr(), 0, std::ptr::null());
            let ext = *(raw as *const *const usize);
            *(raw as *mut usize) = *ext.sub(1);
            let b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(buf.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            l.ResourceBarrier(&[b]);
            l.CopyResource(&rb, &buf);
            l.Close().unwrap();
            q.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            q.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 5000).expect("the clear completes");
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p as *const u32, N as usize).to_vec();
            rb.Unmap(0, None);
            console_heap::unregister(co);
            console_heap::unregister(go);
            assert!(got.iter().all(|&v| v == 0xDEAD_BEEF), "the UAV was not cleared: {:x?}", &got[..4]);
        }
    }
    #[test]
    fn a_gpu_wait_is_released_after_xcl_gpu_wait_release_ms_or_the_default() {
        assert_eq!(release_after(None), GPU_WAIT_WATCHDOG_TIMEOUT);
        assert_eq!(release_after(Some("180000")), Duration::from_secs(180));
        assert_eq!(release_after(Some(" 750 ")), Duration::from_millis(750));
        assert_eq!(release_after(Some("0")), GPU_WAIT_WATCHDOG_TIMEOUT, "0 keeps the default");
        assert_eq!(release_after(Some("soon")), GPU_WAIT_WATCHDOG_TIMEOUT);
    }

    /// The console's cache flushes are UAV barriers for all resources on lists that take them; a
    /// flush of nothing, a copy list's and a bundle's are not.
    #[test]
    fn cache_flushes_are_uav_barriers() {
        #[repr(C)]
        struct FakeFlushList {
            vt: *const usize,
            ty: i32,
            barriers: Mutex<Vec<(i32, usize)>>,
        }
        unsafe extern "system" fn get_type(this: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            D3D12_COMMAND_LIST_TYPE((*(this as *const FakeFlushList)).ty)
        }
        unsafe extern "system" fn barrier(this: *mut c_void, n: u32, b: *const D3D12_RESOURCE_BARRIER) {
            for k in 0..n as usize {
                let b = &*b.add(k);
                let res = (*b.Anonymous.UAV).pResource.as_ref().map_or(0, |r| r.as_raw() as usize);
                (*(this as *const FakeFlushList)).barriers.lock().unwrap().push((b.Type.0, res));
            }
        }
        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = get_type as usize;
        orig_vt[PC_RESOURCE_BARRIER] = barrier as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let mut fake = Box::new(FakeFlushList { vt: unsafe { ext_vt.as_ptr().add(1) }, ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0, barriers: Mutex::new(Vec::new()) });
        let uav = (D3D12_RESOURCE_BARRIER_TYPE_UAV.0, 0);
        unsafe {
            let list = &mut *fake as *mut FakeFlushList as *mut c_void;
            flush_cache_range_x(list, 0xc3c0, 0x8002_0000, 0, 0xFFFF_FFFF_FF00);
            flush_cache_range_x(list, 0, 0x8000_0000, 0x1000, 0x100);
            flush_pipeline_x(list, 1, 0, 0);
            assert_eq!(*fake.barriers.lock().unwrap(), vec![uav; 3]);
            // nothing asked for, or of no memory
            flush_cache_range_x(list, 0, 0, 0, 0x1000);
            flush_cache_range_x(list, 0xc3c0, 0, 0, 0);
            assert_eq!(fake.barriers.lock().unwrap().len(), 3);
            fake.ty = D3D12_COMMAND_LIST_TYPE_DIRECT.0;
            flush_cache_range_x(list, 0xc3c0, 0, 0, 1);
            assert_eq!(fake.barriers.lock().unwrap().len(), 4);
            for ty in [D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_BUNDLE] {
                fake.ty = ty.0;
                flush_cache_range_x(list, 0xc3c0, 0, 0, 1);
                flush_pipeline_x(list, 1, 0, 0);
            }
            assert_eq!(fake.barriers.lock().unwrap().len(), 4);
        }
    }

    /// `XCL_GPU_WAIT_WATCH_MS` is a watch only when it gives a GPU wait longer than the default.
    #[test]
    fn a_gpu_wait_is_watched_only_for_longer_than_the_default() {
        assert_eq!(watch_for(None), None);
        assert_eq!(watch_for(Some(" 60000 ")), Some(Duration::from_secs(60)));
        assert_eq!(watch_for(Some("5000")), None, "no longer than the default: nothing to watch");
        assert_eq!(watch_for(Some("0")), None);
        assert_eq!(watch_for(Some("long")), None);
    }

    static RESET_TEST_HR: AtomicU32 = AtomicU32::new(0);
    static RESET_TEST_REPLACEMENT_VT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn reset_test_native(this: *mut c_void, _alloc: usize, _pso: usize) -> i32 {
        *(this as *mut usize) = RESET_TEST_REPLACEMENT_VT.load(Ordering::SeqCst) as usize;
        RESET_TEST_HR.load(Ordering::SeqCst) as i32
    }

    #[test]
    fn null_input_assembler_views_clear_tracked_bindings() {
        unsafe extern "system" fn host_vertex_buffers(_: *mut c_void, _: u32, _: u32, _: *const D3D12_VERTEX_BUFFER_VIEW) {}
        unsafe extern "system" fn host_index_buffer(_: *mut c_void, _: *const D3D12_INDEX_BUFFER_VIEW) {}

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_IA_SET_VERTEX_BUFFERS] = host_vertex_buffers as usize;
        orig_vt[PC_IA_SET_INDEX_BUFFER] = host_index_buffer as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0];
        let mut object = unsafe { ext_vt.as_ptr().add(1) };
        let list = (&mut object as *mut *const usize).cast::<c_void>();
        with_bindings(list, |b| {
            for slot in 0..4 {
                let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: u64::from(slot + 1), SizeInBytes: 16, StrideInBytes: 4 };
                b.vertex_buffers.insert(slot, view);
                b.vertex_titles.insert(slot, (view.BufferLocation, view.SizeInBytes, view.StrideInBytes));
            }
            b.index_title = Some((16, 16, DXGI_FORMAT_R16_UINT.0));
        });

        unsafe { ia_set_vertex_buffers_x(list, 0, 0, std::ptr::null()) };
        with_bindings(list, |b| {
            assert_eq!(b.vertex_buffers.len(), 4);
            assert_eq!(b.vertex_titles.len(), 4);
        });
        unsafe {
            ia_set_vertex_buffers_x(list, 1, 2, std::ptr::null());
            ia_set_index_buffer_x(list, std::ptr::null());
        }
        with_bindings(list, |b| {
            assert_eq!(b.vertex_buffers.keys().copied().collect::<Vec<_>>(), [0, 3]);
            assert_eq!(b.vertex_titles.keys().copied().collect::<Vec<_>>(), [0, 3]);
            assert!(b.index_title.is_none());
        });
        on_release(list as usize);
    }

    #[test]
    fn nested_occlusion_queries_end_the_outer_one_early() {
        let ty = D3D12_QUERY_TYPE_OCCLUSION;
        let mut o = Occlusion::default();
        assert_eq!(o.begin((1, 0), ty), None);
        assert_eq!(o.begin((1, 1), ty), Some(((1, 0), ty)), "the outer one ends on the PC");
        assert!(o.end((1, 1)), "inner: a real query");
        // more per-object queries inside the same outer one
        assert_eq!(o.begin((1, 2), ty), None);
        assert!(o.end((1, 2)));
        assert!(!o.end((1, 0)), "outer: already ended");
        // overlapping, not nested: A, B, end A, end B
        assert_eq!(o.begin((1, 3), ty), None);
        assert_eq!(o.begin((1, 4), ty), Some(((1, 3), ty)));
        assert!(!o.end((1, 3)) && o.end((1, 4)));
        assert!(o.active.is_none() && o.ended_early.is_empty());
        assert!(o.end((1, 9)), "a query not begun here: the PC's call");
        // kickoff ends active query early so EndQuery later resolves it as visible
        assert_eq!(o.begin((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION), None);
        assert_eq!(o.end_active_early(), Some(((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION)));
        assert!(!o.end((1, 5)));
    }

    #[test]
    fn reset_commits_state_only_after_native_success_and_always_restores_vtable() {
        let mut native = vec![0usize; PC_RESET + 1];
        native[PC_RESET] = reset_test_native as usize;
        let extended = [native.as_ptr() as usize, 0usize];
        let ext = unsafe { extended.as_ptr().add(1) as usize };
        let replacement = 0x1234_5678usize;
        let mut object = [ext];
        let list = object.as_mut_ptr() as *mut c_void;
        let key = list as usize;

        unsafe { on_create(list, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            s.allocator = 0x111;
            s.bound_pso = 0x222;
            s.depth_bias = Some(DepthBias::new(-3.5, 0.25, 2.0));
            s.native_bias = DepthBias::new(-4.5, 0.5, 3.0);
            s.pso_dirty = true;
            s.using_fallback_pso = true;
            s.index_buffer = Some(D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: 0x5000,
                SizeInBytes: 12,
                Format: DXGI_FORMAT_R32_UINT,
            });
            s.depth_bounds = Some((0.25, 0.75));
            s.written_ranges.push((0x6000, 0x6010));
        }
        super::super::bind::seed_reset_state_for_test(list, 0x111, 0x222, vec![0x4444]);
        RESET_TEST_REPLACEMENT_VT.store(replacement as u64, Ordering::SeqCst);

        RESET_TEST_HR.store(0x8000_4005, Ordering::SeqCst);
        assert!(unsafe { reset_x(list, 0x333, 0x555) } < 0);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x111, 0x222));
            assert_eq!(s.depth_bias, Some(DepthBias::new(-3.5, 0.25, 2.0)));
            assert_eq!(s.native_bias, DepthBias::new(-4.5, 0.5, 3.0));
            assert!(s.pso_dirty && s.using_fallback_pso);
            assert_eq!(s.index_buffer.as_ref().unwrap().BufferLocation, 0x5000);
            assert_eq!(s.depth_bounds, Some((0.25, 0.75)));
            assert_eq!(s.written_ranges, vec![(0x6000, 0x6010)]);
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x111, 0x222, vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![0x4444])
        );
        assert!(super::super::console_list::take_pending_for_test(0x111).is_empty());

        RESET_TEST_HR.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { reset_x(list, 0x333, 0x555) }, 0);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x333, 0x555));
            assert!(s.depth_bias.is_none() && s.index_buffer.is_none() && s.depth_bounds.is_none());
            assert!(!s.pso_dirty && !s.using_fallback_pso && s.written_ranges.is_empty());
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x333, 0x555, Vec::new(), Vec::new(), Vec::new())
        );
        assert_eq!(super::super::console_list::take_pending_for_test(0x111), vec![0x4444]);

        LISTS.lock().unwrap().as_mut().unwrap().remove(&key);
        super::super::bind::remove_state_for_test(list);
    }

    #[test]
    fn fill_patterns_repeat_the_value() {
        assert_eq!(pattern(&0x1122_3344u32.to_le_bytes())[..8], [0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(pattern(&[7u8; 16]), [7u8; 16]);
    }

    #[test]
    fn wait_comparison_ops_cover_all_d3d12_funcs() {
        assert_eq!(compare_u64(0, 1, 1), None);
        assert_eq!(compare_u64(9, 1, 1), None);
        assert_eq!(compare_u64(1, 5, 5), Some(false)); // NEVER
        assert_eq!(compare_u64(2, 4, 5), Some(true));  // LESS
        assert_eq!(compare_u64(2, 5, 5), Some(false));
        assert_eq!(compare_u64(3, 5, 5), Some(true));  // EQUAL
        assert_eq!(compare_u64(3, 4, 5), Some(false));
        assert_eq!(compare_u64(4, 5, 5), Some(true));  // LESS_EQUAL
        assert_eq!(compare_u64(4, 6, 5), Some(false));
        assert_eq!(compare_u64(5, 6, 5), Some(true));  // GREATER
        assert_eq!(compare_u64(5, 5, 5), Some(false));
        assert_eq!(compare_u64(6, 4, 5), Some(true));  // NOT_EQUAL
        assert_eq!(compare_u64(6, 5, 5), Some(false));
        assert_eq!(compare_u64(7, 5, 5), Some(true));  // GREATER_EQUAL
        assert_eq!(compare_u64(7, 4, 5), Some(false));
        assert_eq!(compare_u64(8, 0, 1), Some(true));  // ALWAYS
    }

    #[test]
    fn pending_wait_32_and_64_bit_with_mask_and_cross_thread_signal() {
        let cell32 = Arc::new(AtomicU32::new(0xDEAD_0000));
        let w32 = PendingWait {
            va: Arc::as_ptr(&cell32) as u64,
            width: 4,
            comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
            value: 0x0000_005A,
            mask: 0x0000_00FF,
            after_write64: false,
        };
        unsafe {
            assert!(!w32.is_satisfied());
            let writer = cell32.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                writer.store(0xBEEF_005A, Ordering::Release);
            });
            assert!(w32.wait_until_satisfied(Duration::from_secs(1)));
            t.join().unwrap();
        }

        let cell64 = Arc::new(AtomicU64::new(0x0000_0001_0000_0000));
        let w64 = PendingWait {
            va: Arc::as_ptr(&cell64) as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 0x0000_0002_0000_0000,
            mask: u64::MAX,
            after_write64: false,
        };
        unsafe {
            assert!(!w64.is_satisfied());
            // bounded timeout returns false without hanging
            assert!(!w64.wait_until_satisfied(Duration::from_millis(5)));
            cell64.store(0x0000_0002_0000_0001, Ordering::Release);
            assert!(w64.is_satisfied());
        }
    }

    #[test]
    fn written_ranges_detect_same_list_overlap() {
        let mut s = ListState::default();
        s.note_write(0x1000, 8);
        assert!(s.has_prior_write(0x1000, 4));
        assert!(s.has_prior_write(0x1004, 4));
        assert!(s.has_prior_write(0x0FFC, 8));
        assert!(!s.has_prior_write(0x1008, 4));
        assert!(!s.has_prior_write(0x0FF8, 8));
    }

    #[test]
    fn buffer_writes_are_clamped_to_the_buffer() {
        assert_eq!(buffer_span(0x4_0000, 0x100, 0x10, 8), Some((0x4_0010, 8)));
        assert_eq!(buffer_span(0x4_0000, 0x100, 0xF8, 0x40), Some((0x4_00F8, 8)));
        assert_eq!(buffer_span(0x4_0000, 0x100, 0, u64::MAX), Some((0x4_0000, 0x100)));
        assert_eq!(buffer_span(0x4_0000, 0x100, 0x100, 8), None);
        assert_eq!(buffer_span(0x4_0000, 0x100, 0x20, 0), None);
    }

    #[test]
    fn scattered_writes_preserve_unwritten_wait_ranges() {
        let mut s = ListState::default();
        for i in (0..300).rev() {
            s.note_write(0x1000 + i * 16, 8);
        }
        let a = 0x1000 + 20 * 16;
        assert!(s.has_prior_write(a, 8));
        assert!(!s.has_prior_write(a + 8, 8));
        s.note_write(a + 8, 8);
        assert!(s.has_prior_write(a + 8, 8));
        assert!(!s.has_prior_write(a + 24, 8));
    }

    #[test]
    fn execute_with_waits_splits_batches_before_unsatisfied_list() {
        // Use synthetic non-zero list pointers distinct from real COM objects.
        let l0 = 0x7000_0001usize as *mut c_void;
        let l1 = 0x7000_0002usize as *mut c_void;
        let l2 = 0x7000_0003usize as *mut c_void;
        let cell = Box::new(AtomicU32::new(0));
        let va = &*cell as *const AtomicU32 as u64;

        unsafe {
            on_create(l0, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l1, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l2, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&(l1 as usize)).unwrap();
            s.push_wait(PendingWait {
                va,
                width: 4,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 42,
                mask: u32::MAX as u64,
                after_write64: false,
            });
        }

        let mut batches: Vec<Vec<usize>> = Vec::new();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
                // Simulate l0's GPU execution satisfying l1's wait!
                if chunk.contains(&l0) {
                    cell.store(42, Ordering::Release);
                }
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize], vec![l1 as usize, l2 as usize]]);

        // Now that cell == 42 is already satisfied, the whole batch submits in one call.
        batches.clear();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize, l1 as usize, l2 as usize]]);

        on_release(l0 as usize);
        on_release(l1 as usize);
        on_release(l2 as usize);
    }

    #[test]
    fn queue_picking_prefers_last_queue_and_respects_release() {
        let dev = 0x8000_0001usize;
        let q1 = 0x8000_0010usize;
        let q2 = 0x8000_0020usize;
        let q_copy = 0x8000_0030usize;
        on_queue_create(q1, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q2, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q_copy, dev, D3D12_COMMAND_LIST_TYPE_COPY);

        // Most recently created matching queue wins when no last_queue
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_COPY), Some(q_copy));
        // Explicit last_queue wins when still live
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q1));
        // Releasing q1 falls back to q2
        on_queue_release(q1);
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        on_queue_release(q2);
        on_queue_release(q_copy);
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), None);
    }

    #[test]
    fn depth_bias_state_tracks_fallback_and_clear() {
        let l = 0x9000_0001usize as *mut c_void;
        let key = l as usize;
        unsafe {
            on_create(l, 0x100, 0xAAA, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_set_pipeline_state(l, 0xBBB);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            assert_eq!(s.bound_pso, 0xBBB);
            assert!(!s.pso_dirty);
            s.depth_bias = Some(DepthBias::new(-10.0, 0.0, 1.0));
            s.pso_dirty = true;
        }
        // Switching to a non-dynamic PSO while depth bias is active keeps pso_dirty for fallback
        unsafe { on_set_pipeline_state(l, 0xCCC) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xCCC);
            assert!(s.pso_dirty);
        }
        // ClearState resets depth bias, native_bias, and dirty flag
        unsafe { on_clear_state(l, 0xDDD) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xDDD);
            assert!(s.depth_bias.is_none());
            assert_eq!(s.native_bias, DepthBias::ZERO);
            assert!(!s.pso_dirty);
        }
        on_release(key);
    }

    #[test]
    fn depth_bias_native_list9_and_bundle_fallback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let supported = device_x::probe_dynamic_depth_bias(&dev);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            let has_list9 = list.cast::<ID3D12GraphicsCommandList9>().is_ok();
            let raw = list.as_raw();
            let key = raw as usize;

            let pso_dyn_zero = 0x9100_0001usize;
            let pso_dyn_static = 0x9100_0002usize;
            let pso_static_only = 0x9100_0003usize;
            let static_bias = DepthBias::new(-500.0, 0.0, 0.0);
            device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
            device_x::register_test_base(pso_dyn_static, true, static_bias);
            device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

            let cache_before = device_x::depth_bias_cache_len();
            on_create(raw, alloc.as_raw() as usize, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_DIRECT);

            if supported && has_list9 {
                // Fractional f32 depth bias takes the native ID3D12GraphicsCommandList9 path
                // without rounding to i32 or dirtying the patched-PSO fallback cache.
                set_depth_bias_x(raw, -125.75, 0.05, 1.25);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.depth_bias, Some(DepthBias::new(-125.75, 0.05, 1.25)));
                    assert_eq!(s.native_bias, DepthBias::new(-125.75, 0.05, 1.25));
                    assert_eq!(s.native_bias.depth_bias(), -125.75);
                    assert!(!s.pso_dirty);
                }
                assert_eq!(device_x::depth_bias_cache_len(), cache_before);

                // Resetting depth bias to 0 updates native_bias to ZERO without touching fallback cache.
                set_depth_bias_x(raw, 0.0, 0.0, 0.0);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert!(s.depth_bias.is_none());
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }

                // Binding a dynamic PSO with non-zero static bias syncs native_bias to its static bias.
                on_set_pipeline_state(raw, pso_dyn_static);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, static_bias);
                    assert!(!s.pso_dirty);
                }

                // Switching back to a zero-static-bias dynamic PSO restores native_bias to ZERO.
                on_set_pipeline_state(raw, pso_dyn_zero);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }
            }

            // Non-dynamic PSO must fall back to the patched-PSO cache (pso_dirty = true).
            on_set_pipeline_state(raw, pso_static_only);
            set_depth_bias_x(raw, -50.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert_eq!(s.depth_bias, Some(DepthBias::new(-50.0, 0.0, 0.0)));
                assert!(s.pso_dirty);
            }
            on_release(key);

            // Bundle lists cannot call ID3D12GraphicsCommandList9::RSSetDepthBias and must fall back.
            let b_raw = 0x9100_00F0usize as *mut c_void;
            on_create(b_raw, 0x100, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_BUNDLE);
            set_depth_bias_x(b_raw, -75.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&(b_raw as usize)).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::ZERO);
            }
            on_release(b_raw as usize);

            device_x::forget_base(pso_dyn_zero);
            device_x::forget_base(pso_dyn_static);
            device_x::forget_base(pso_static_only);
        }
    }

    #[test]
    fn depth_bias_native_path_exact_f32_and_list9_fallback() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Verify PC vtable slot of ID3D12GraphicsCommandList9::RSSetDepthBias is in layout table.
        let cl9 = super::super::layout::find("ID3D12GraphicsCommandList9").expect("ID3D12GraphicsCommandList9 in layout");
        assert_eq!(super::super::layout::slot(cl9, "RSSetDepthBias"), Some(82));

        let calls: Rc<RefCell<Vec<(usize, f32, f32, f32)>>> = Rc::new(RefCell::new(Vec::new()));
        let allow = Rc::new(std::cell::Cell::new(true));
        let calls_cb = calls.clone();
        let allow_cb = allow.clone();
        TEST_NATIVE_BIAS.with(|h| {
            *h.borrow_mut() = Some(Box::new(move |list, bias| {
                if !allow_cb.get() {
                    return false;
                }
                calls_cb.borrow_mut().push((list, bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias()));
                true
            }));
        });

        let l = 0x9200_0001usize as *mut c_void;
        let key = l as usize;
        let pso_dyn_zero = 0x9200_0010usize;
        let pso_dyn_static = 0x9200_0011usize;
        let pso_static_only = 0x9200_0012usize;
        device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
        device_x::register_test_base(pso_dyn_static, true, DepthBias::new(-500.0, 0.25, 2.0));
        device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

        let cache_before = device_x::depth_bias_cache_len();
        unsafe {
            // 1. Initial PSO with baked static bias seeds native RSSetDepthBias on CreateCommandList.
            on_create(l, 0x100, pso_dyn_static, D3D12_COMMAND_LIST_TYPE_DIRECT);
            assert_eq!(*calls.borrow(), vec![(key, -500.0, 0.25, 2.0)]);

            // 2. Switching to zero-static-bias dynamic PSO resets native RSSetDepthBias to (0, 0, 0).
            on_set_pipeline_state(l, pso_dyn_zero);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));

            // 3. Fractional f32 override is forwarded verbatim (no i32 rounding, no fallback PSO cache).
            set_depth_bias_x(l, -1000.5, 0.05, 1.5);
            assert_eq!(calls.borrow().last(), Some(&(key, -1000.5, 0.05, 1.5)));
            assert_eq!(device_x::depth_bias_cache_len(), cache_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(!s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::new(-1000.5, 0.05, 1.5));
            }

            // 4. Switching between dynamic PSOs while override is active keeps the override without redundant calls.
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().len(), n_before);

            // 5. Resetting to (0, 0, 0) sets native bias to (0, 0, 0); ClearState with static-bias PSO re-seeds it.
            set_depth_bias_x(l, 0.0, 0.0, 0.0);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));
            on_clear_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().last(), Some(&(key, -500.0, 0.25, 2.0)));

            // 6. If ID3D12GraphicsCommandList9 is unavailable, set_depth_bias_x falls back cleanly to pso_dirty.
            on_set_pipeline_state(l, pso_dyn_zero);
            allow.set(false);
            set_depth_bias_x(l, -42.5, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.depth_bias, Some(DepthBias::new(-42.5, 0.0, 0.0)));
            }

            // 7. Non-dynamic PSO always uses the patched-PSO fallback and never invokes RSSetDepthBias.
            allow.set(true);
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_static_only);
            set_depth_bias_x(l, -10.0, 0.0, 0.0);
            assert_eq!(calls.borrow().len(), n_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
            }
        }

        TEST_NATIVE_BIAS.with(|h| *h.borrow_mut() = None);
        on_release(key);
        device_x::forget_base(pso_dyn_zero);
        device_x::forget_base(pso_dyn_static);
        device_x::forget_base(pso_static_only);
    }

    #[test]
    fn resolve_query_data_applies_buffer_delta_to_pc_resolve_and_occlusion_writeback() {
        if !crate::hle::d3d12x::sparse::windows_ready() {
            // The resolve reaches its PC data through a console heap, which needs window-0 storage a
            // software adapter cannot make; without it the whole buffer goes instead.
            return crate::gpu::recomp::runner::skip_test(
                "no 4 GiB sparse memory window: this adapter cannot reserve one, and the resolve needs a console heap");
        }
        use std::cell::RefCell;

        thread_local! {
            static RESOLVED: RefCell<Vec<(usize, u32, u32, usize, u64)>> = const { RefCell::new(Vec::new()) };
            static WRITES: RefCell<Vec<(u64, u32)>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_resolve(
            _this: *mut c_void,
            heap: usize,
            _ty: D3D12_QUERY_TYPE,
            first: u32,
            count: u32,
            dest: usize,
            offset: u64,
        ) {
            RESOLVED.with(|r| r.borrow_mut().push((heap, first, count, dest, offset)));
        }

        unsafe extern "system" fn fake_write_immediate(
            _this: *mut c_void,
            count: u32,
            params: *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER,
            _modes: *const D3D12_WRITEBUFFERIMMEDIATE_MODE,
        ) {
            let slice = std::slice::from_raw_parts(params, count as usize);
            WRITES.with(|w| {
                let mut w = w.borrow_mut();
                for p in slice {
                    w.push((p.Dest, p.Value));
                }
            });
        }

        const BASE_GPU_VA: u64 = 0x1_0000_0000;
        unsafe extern "system" fn fake_gpu_va(_this: *mut c_void) -> u64 {
            BASE_GPU_VA
        }

        #[repr(C)]
        struct FakeCom {
            vt: *const usize,
        }

        let mut orig_list_vt = [0usize; 70];
        orig_list_vt[PC_RESOLVE_QUERY_DATA] = fake_resolve as usize;
        orig_list_vt[PC_WRITE_BUFFER_IMMEDIATE] = fake_write_immediate as usize;
        // Adopted objects store `orig` at `ext[-1]`.
        let ext_list_vt = [orig_list_vt.as_ptr() as usize, 0usize];
        let mut fake_list = FakeCom {
            // SAFETY: `ext_list_vt` has 2 elements; `.add(1)` points at `ext[0]` with `ext[-1] = orig`.
            vt: unsafe { ext_list_vt.as_ptr().add(1) },
        };

        let mut res_vt = [0usize; 16];
        res_vt[11] = fake_gpu_va as usize;
        let mut fake_res = FakeCom { vt: res_vt.as_ptr() };

        let heap = 0x5555_0001usize;
        let dest = &mut fake_res as *mut FakeCom as usize;
        let delta = 0x3400u64;
        let title_offset = 0x100u64;

        real::set_test_buffer_delta(dest, Some(delta));
        VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, 2), VISIBLE_SAMPLES);
        RESOLVED.with(|r| r.borrow_mut().clear());
        WRITES.with(|w| w.borrow_mut().clear());

        unsafe {
            resolve_query_data_x(
                &mut fake_list as *mut FakeCom as *mut c_void,
                heap,
                D3D12_QUERY_TYPE_OCCLUSION,
                1,
                3,
                dest,
                title_offset,
            );
        }

        real::set_test_buffer_delta(dest, None);
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, 2));
        }

        let expected_off = title_offset + delta;
        RESOLVED.with(|r| {
            assert_eq!(*r.borrow(), vec![(heap, 1, 3, dest, expected_off)]);
        });
        WRITES.with(|w| {
            let expected_va = BASE_GPU_VA + expected_off + 8; // query index 2 = slot (2 - 1) = +8 bytes
            assert_eq!(
                *w.borrow(),
                vec![
                    (expected_va, VISIBLE_SAMPLES as u32),
                    (expected_va + 4, (VISIBLE_SAMPLES >> 32) as u32),
                ]
            );
        });
    }

    #[test]
    fn draw_indexed_x_routes_through_bind_and_restores_index_buffer() {
        #[repr(C)]
        struct FakeDrawList {
            vt: *const usize,
            refs: AtomicUsize,
            draws: Mutex<Vec<(u32, u32, u32, i32, u32)>>,
            index_buffers: Mutex<Vec<Option<D3D12_INDEX_BUFFER_VIEW>>>,
        }

        unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn fake_draw_indexed(this: *mut c_void, ic: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
            (*(this as *const FakeDrawList)).draws.lock().unwrap().push((ic, inst, si, bv, sinst));
        }
        unsafe extern "system" fn fake_set_ib(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
            let v = view.as_ref().copied();
            (*(this as *const FakeDrawList)).index_buffers.lock().unwrap().push(v);
        }
        unsafe extern "system" fn fake_noop_ptr(_this: *mut c_void, _p: *mut c_void) {}
        unsafe extern "system" fn fake_noop_uav(_this: *mut c_void, _slot: u32, _va: u64) {}

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[13] = fake_draw_indexed as usize;
        orig_vt[25] = fake_noop_ptr as usize; // SetPipelineState
        orig_vt[30] = fake_noop_ptr as usize; // SetGraphicsRootSignature
        orig_vt[42] = fake_noop_uav as usize; // SetGraphicsRootUnorderedAccessView
        orig_vt[43] = fake_set_ib as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeDrawList {
            // SAFETY: `ext_vt` has 65 entries so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            draws: Mutex::new(Vec::new()),
            index_buffers: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeDrawList as *mut c_void;

        // The index buffers' addresses: reserved, never committed, so not readable and never
        // mirrored into unified memory. A fixed address (0x1_0000_1000) was committed memory once
        // earlier tests had allocated there, and the draw then bound its mirror.
        let reserved = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(std::ptr::null(), 0x1_0000, windows_sys::Win32::System::Memory::MEM_RESERVE, windows_sys::Win32::System::Memory::PAGE_NOACCESS)
        } as u64;
        assert_ne!(reserved, 0, "reserve the index buffers' addresses");
        let ib = |k: u64| reserved + k * 0x1000;
        let orig_ib = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib(1),
            SizeInBytes: 64,
            Format: DXGI_FORMAT_R16_UINT,
        };

        unsafe {
            ia_set_index_buffer_x(list_ptr, &orig_ib);
            fake.index_buffers.lock().unwrap().clear();

            // 1. Untranslated (PC) PSO: routes through bind::draw_indexed_instanced -> DrawIndexedInstanced,
            // records DRAW_ARGS, binds temp IB, then restores orig_ib.
            super::super::bind::reset(list_ptr, 0x7400_0100, 0);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, ib(2), DXGI_FORMAT_R16_UINT, 6, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 6, 1, 0, 0));
            assert!(after[2] > before[2], "DRAWS_PC must increment via bind::draw_indexed_instanced");
            assert_eq!(*fake.draws.lock().unwrap(), vec![(6, 1, 0, 0, 0)]);
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            assert_eq!(ibs.len(), 2);
            assert_eq!(ibs[0].unwrap().BufferLocation, ib(2));
            assert_eq!(ibs[0].unwrap().SizeInBytes, 12);
            assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R16_UINT);
            assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            assert_eq!(ibs[1].unwrap().SizeInBytes, orig_ib.SizeInBytes);

            // 2. Translated graphics PSO: enters bind::graphics_bound translated path (DRAWS or
            // DRAWS_SKIPPED, NOT DRAWS_PC) instead of bypassing bind.rs directly to PC slot 13.
            let translated_pso = 0x7400_0055usize;
            real::register_translated_graphics_for_test(translated_pso);
            super::super::bind::reset(list_ptr, 0x7400_0100, translated_pso);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, ib(3), DXGI_FORMAT_R32_UINT, 12, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 12, 1, 0, 0));
            assert_eq!(
                (after[0] + after[1]) - (before[0] + before[1]),
                1,
                "translated graphics PSO must go through bind::graphics_bound (DRAWS or DRAWS_SKIPPED)"
            );
            assert_eq!(after[2], before[2], "translated graphics PSO must not take the PC draw path");
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            if after[0] > before[0] {
                assert_eq!(ibs.len(), 2);
                assert_eq!(ibs[0].unwrap().SizeInBytes, 48);
                assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R32_UINT);
                assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            } else {
                assert!(ibs.is_empty(), "a rejected preparation must not mutate the native index buffer");
            }

            // 3. Zero count or invalid format returns early without touching IB or draws.
            let draws_before = fake.draws.lock().unwrap().len();
            draw_indexed_x(list_ptr, ib(4), DXGI_FORMAT_R16_UINT, 0, 0);
            draw_indexed_x(list_ptr, ib(4), DXGI_FORMAT_R8_UINT, 3, 0);
            assert_eq!(fake.draws.lock().unwrap().len(), draws_before);
            assert!(fake.index_buffers.lock().unwrap().is_empty());

            real::forget_pso_for_test(translated_pso);
        }

        super::super::bind::forget(list_ptr);
        on_release(list_ptr as usize);
        unsafe { windows_sys::Win32::System::Memory::VirtualFree(reserved as *mut c_void, 0, windows_sys::Win32::System::Memory::MEM_RELEASE) };
    }

    #[test]
    fn kickoff_x_render_targets_preserve_mrt_count_for_single_descriptor_range() {
        #[derive(Debug, PartialEq, Eq)]
        struct RtCall {
            n: u32,
            handles: Vec<usize>,
            rtvs_was_null: bool,
            single: i32,
            dsv: Option<usize>,
        }

        #[repr(C)]
        struct FakeRtList {
            vt: *const usize,
            calls: Mutex<Vec<RtCall>>,
        }

        unsafe extern "system" fn fake_om_set_render_targets(
            this: *mut c_void,
            n: u32,
            rtvs: *const usize,
            single: i32,
            dsv: *const usize,
        ) {
            let read_n = if rtvs.is_null() {
                0
            } else if single != 0 {
                n.min(1) as usize
            } else {
                n as usize
            };
            let handles = if read_n == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_n).to_vec() };
            (*(this as *const FakeRtList)).calls.lock().unwrap().push(RtCall {
                n,
                handles,
                rtvs_was_null: rtvs.is_null(),
                single,
                dsv: dsv.as_ref().copied(),
            });
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_OM_SET_RENDER_TARGETS] = fake_om_set_render_targets as usize;
        let mut ext_vt = [0usize; 2];
        ext_vt[0] = orig_vt.as_ptr() as usize;

        let fake = Box::new(FakeRtList {
            // SAFETY: `ext_vt` has 2 entries so `.add(1)` has `orig_vt` at `-1` as `real::orig_fn` expects.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            calls: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeRtList as *mut c_void;

        let restore = || {
            let b = LISTS.lock().unwrap().as_ref().unwrap().get(&(list_ptr as usize)).unwrap().bindings.clone();
            unsafe { restore_bindings(list_ptr, &b, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        };

        unsafe {
            // 1. Single contiguous descriptor range for 4 MRTs + DSV: only 1 handle exists at `rtvs`,
            //    yet restore_bindings must pass `n = 4` (not 1!) to OMSetRenderTargets.
            let single_base = [0x1000usize];
            let dsv_val = 0x9000usize;
            note_render_targets(list_ptr, 4, single_base.as_ptr(), 1, &dsv_val);
            assert!(
                state_summary(list_ptr).contains("render targets Some((4, true))"),
                "state_summary should report full MRT count: {}",
                state_summary(list_ptr)
            );
            restore();

            // 2. Oversized count with single range is clamped to D3D12 max (8).
            note_render_targets(list_ptr, 12, single_base.as_ptr(), 1, std::ptr::null());
            restore();

            // 3. Separate handles (single == 0) for 3 MRTs.
            let multi = [0x2000usize, 0x2020, 0x2040];
            note_render_targets(list_ptr, 3, multi.as_ptr(), 0, std::ptr::null());
            restore();

            // 4. Depth-only (n == 0, rtvs == null, dsv != null).
            note_render_targets(list_ptr, 0, std::ptr::null(), 0, &dsv_val);
            restore();

            // 5. Null rtvs pointer with non-zero n must safely restore n == 0, null pointer.
            note_render_targets(list_ptr, 4, std::ptr::null(), 1, std::ptr::null());
            restore();
        }

        let calls = fake.calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![
                RtCall { n: 4, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: Some(0x9000) },
                RtCall { n: 8, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: None },
                RtCall { n: 3, handles: vec![0x2000, 0x2020, 0x2040], rtvs_was_null: false, single: 0, dsv: None },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 0, dsv: Some(0x9000) },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 1, dsv: None },
            ]
        );

        on_release(list_ptr as usize);
    }

    #[test]
    fn set_predication_buffer_state_transitions_and_lifecycle() {
        #[derive(Debug, PartialEq, Eq)]
        enum PredEvent {
            Barrier(usize, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES),
            Copy(usize, u64, usize, u64, u64),
            SetPred(usize, u64, D3D12_PREDICATION_OP),
        }

        #[repr(C)]
        struct PredDesc {
            va: u64,
            op: u32,
        }

        #[repr(C)]
        struct FakePredList {
            vt: *const usize,
            ty: Mutex<D3D12_COMMAND_LIST_TYPE>,
            events: Mutex<Vec<PredEvent>>,
        }

        #[repr(C)]
        struct FakeComRes {
            vt: *const usize,
            refs: AtomicUsize,
        }

        unsafe extern "system" fn fake_get_type(this: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            *(*(this as *const FakePredList)).ty.lock().unwrap()
        }

        unsafe extern "system" fn fake_resource_barrier(this: *mut c_void, count: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
            let mut events = (*(this as *const FakePredList)).events.lock().unwrap();
            for i in 0..count as usize {
                let b = &*barriers.add(i);
                assert_eq!(b.Type, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
                let tr = &*b.Anonymous.Transition;
                let res = (std::ptr::addr_of!(tr.pResource) as *const usize).read_unaligned();
                events.push(PredEvent::Barrier(res, tr.StateBefore, tr.StateAfter));
            }
        }

        unsafe extern "system" fn fake_copy_buffer_region(
            this: *mut c_void,
            dst: usize,
            dst_off: u64,
            src: usize,
            src_off: u64,
            n: u64,
        ) {
            (*(this as *const FakePredList))
                .events
                .lock()
                .unwrap()
                .push(PredEvent::Copy(dst, dst_off, src, src_off, n));
        }

        unsafe extern "system" fn fake_set_predication(
            this: *mut c_void,
            res: usize,
            off: u64,
            op: D3D12_PREDICATION_OP,
        ) {
            (*(this as *const FakePredList))
                .events
                .lock()
                .unwrap()
                .push(PredEvent::SetPred(res, off, op));
        }

        unsafe extern "system" fn res_add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeComRes)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }

        unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
            (*(this as *const FakeComRes)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = fake_get_type as usize;
        orig_vt[PC_COPY_BUFFER_REGION] = fake_copy_buffer_region as usize;
        orig_vt[PC_RESOURCE_BARRIER] = fake_resource_barrier as usize;
        orig_vt[PC_SET_PREDICATION] = fake_set_predication as usize;

        let mut ext_vt = [0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakePredList {
            // SAFETY: `ext_vt` has 65 entries so `.add(1)` points at slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            ty: Mutex::new(D3D12_COMMAND_LIST_TYPE_DIRECT),
            events: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakePredList as *mut c_void;
        let list_key = list_ptr as usize;

        let mut res_vt = [0usize; 16];
        res_vt[1] = res_add_ref as usize;
        res_vt[2] = res_release as usize;
        let fake_res = Box::new(FakeComRes {
            vt: res_vt.as_ptr(),
            refs: AtomicUsize::new(1),
        });
        let pb_raw = &*fake_res as *const FakeComRes as usize;
        let uma_buf = 0xDEAD_BEEFusize;

        let take_events = || std::mem::take(&mut *fake.events.lock().unwrap());

        unsafe {
            // Seed ListState with fake dedicated predication resource and verify initial rejections.
            on_create(list_ptr, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            {
                let mut g = LISTS.lock().unwrap();
                let s = g.as_mut().unwrap().get_mut(&list_key).unwrap();
                s.pred_buf = Some(ID3D12Resource::from_raw(pb_raw as *mut c_void));
            }

            // (a) BUNDLE and COPY lists are rejected without emitting any commands.
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_BUNDLE;
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert!(take_events().is_empty(), "BUNDLE list must reject SetPredicationBufferX");

            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_COPY;
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert!(take_events().is_empty(), "COPY list must reject SetPredicationBufferX");

            // (b) DIRECT and COMPUTE lists reject invalid descriptors / ops / alignments / non-UMA VAs.
            let non_uma_slot = 0u64;
            let non_uma_va = &non_uma_slot as *const u64 as u64;
            let bad_op = PredDesc { va: 0x1000, op: 2 };
            let unaligned = PredDesc { va: 0x1004, op: 0 };
            let non_uma = PredDesc { va: non_uma_va, op: 0 };
            let bad_desc = std::ptr::without_provenance::<u64>(1);
            for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE] {
                *fake.ty.lock().unwrap() = ty;
                set_predication_buffer_x(list_ptr, bad_desc);
                set_predication_buffer_x(list_ptr, &bad_op as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &unaligned as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &non_uma as *const PredDesc as *const u64);
                assert!(take_events().is_empty(), "invalid inputs on {ty:?} must emit no events");
            }

            // (c) Initial apply_predication_copy emits exact barriers + copy + SetPredication without disarm.
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_DIRECT;
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x100, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                ]
            );

            // Rejections while a predicate is active must neither emit events nor disturb tracked state.
            for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE] {
                *fake.ty.lock().unwrap() = ty;
                set_predication_buffer_x(list_ptr, bad_desc);
                set_predication_buffer_x(list_ptr, &bad_op as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &unaligned as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &non_uma as *const PredDesc as *const u64);
                assert!(take_events().is_empty());
            }
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_DIRECT;
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&list_key).unwrap();
                assert_eq!(s.predication, Some(D3D12_PREDICATION_OP_EQUAL_ZERO));
                assert!(s.pred_in_state);
            }

            // (d) Direct switch without NULL disarms active predicate BEFORE copy and transitions PREDICATION -> COPY_DEST.
            apply_predication_copy(list_ptr, uma_buf, 0x200, pb_raw, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x200, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO),
                ]
            );

            // (e) ClearState clears predication while keeping pred_in_state == true:
            // re-arming omits redundant SetPred(0, ...) yet transitions pb_raw from PREDICATION -> COPY_DEST,
            // and ClearState + before_close still transitions pb_raw from PREDICATION -> COMMON (once).
            on_clear_state(list_ptr, 0);
            apply_predication_copy(list_ptr, uma_buf, 0x300, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x300, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                ]
            );

            on_clear_state(list_ptr, 0);
            before_close(list_ptr);
            assert_eq!(
                take_events(),
                vec![PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON)]
            );
            before_close(list_ptr);
            assert!(take_events().is_empty(), "second before_close must be idempotent");

            // (f) NULL transitions PREDICATION -> COMMON when active; second NULL emits 0 barriers.
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            take_events();
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON),
                ]
            );
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert_eq!(take_events(), vec![PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO)]);

            // (g) on_pc_reset clears predication & pred_in_state so subsequent before_close emits 0 barriers.
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            take_events();
            on_pc_reset(list_ptr);
            before_close(list_ptr);
            assert!(take_events().is_empty(), "on_pc_reset must clear pred_in_state before before_close");
        }

        // (h) on_release removes ListState and releases the dedicated COM resource.
        on_release(list_key);
        assert!(!has_list_state(list_key));
        assert_eq!(fake_res.refs.load(Ordering::Relaxed), 0);
    }

    /// Write64BitValueBottomOfPipeX / Write64BitValueTopOfPipeX with timestamp flags targeting
    /// writable memory outside unified memory resolve through the timestamp readback buffer and
    /// land once the queue completes. Unaligned and unwritable addresses are ignored.
    #[test]
    fn timestamps_outside_unified_memory_land_after_the_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

            let list_ptr = list.as_raw();
            real::readopt(list_ptr, real::Kind::CommandList);
            on_create(list_ptr, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);

            let bottom_ts = Box::new(0u64);
            let top_ts = Box::new(0u64);
            let bottom_va = bottom_ts.as_ref() as *const u64 as u64;
            let top_va = top_ts.as_ref() as *const u64 as u64;

            // Unaligned and null targets are ignored without queueing CPU writes.
            write64_bottom_x(list_ptr, bottom_va + 4, 0, 0, BOTTOM_TIMESTAMPS);
            write64_bottom_x(list_ptr, 0, 0, 0, BOTTOM_TIMESTAMPS);
            {
                let g = LISTS.lock().unwrap();
                assert!(g.as_ref().unwrap()[&(list_ptr as usize)].cpu_writes.is_empty());
            }

            // Valid 8-byte aligned non-UMA targets record GPU timestamp queries + readback copies.
            write64_top_x(list_ptr, top_va, 0, TOP_TIMESTAMPS);
            write64_bottom_x(list_ptr, bottom_va, 0, 0, BOTTOM_TIMESTAMPS);
            {
                let g = LISTS.lock().unwrap();
                assert_eq!(g.as_ref().unwrap()[&(list_ptr as usize)].cpu_writes.len(), 2);
            }

            list.Close().unwrap();
            type Execute = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let (target, _) = real::original_or_native(q.as_raw(), 10);
            let execute: Execute = std::mem::transmute(target);
            execute(q.as_raw(), 1, &list_ptr);
            cpu_writes_after(q.as_raw(), &[list_ptr]);

            let t = Instant::now();
            while (*(bottom_va as *const AtomicU64)).load(Ordering::Acquire) == 0 || (*(top_va as *const AtomicU64)).load(Ordering::Acquire) == 0 {
                assert!(t.elapsed().as_secs() < 5, "non-UMA GPU timestamp never landed (top={}, bottom={})", *top_ts, *bottom_ts);
                std::thread::yield_now();
            }
            assert_ne!(*top_ts, 0);
            assert_ne!(*bottom_ts, 0);

        }
    }

    #[test]
    fn write_buffer_immediate_on_open_existing_heap_visible_to_cpu() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dev3: ID3D12Device3 = dev.cast().unwrap();

            let mem = VirtualAlloc(std::ptr::null(), 0x1_0000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
            assert!(!mem.is_null());
            let base = mem as usize;

            let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(mem).expect("OpenExistingHeapFromAddress");
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 0x1_0000,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
                ..Default::default()
            };
            let mut buf: Option<ID3D12Resource> = None;
            dev.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).unwrap();
            let buf = buf.unwrap();
            let gpu_va = buf.GetGPUVirtualAddress();

            // Upload buffer holding known u64 for CopyBufferRegion comparison
            let up_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &up_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut up,
            )
            .unwrap();
            let up = up.unwrap();
            let mut up_ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut up_ptr)).unwrap();
            *(up_ptr as *mut u64) = 0xCAFE_BABE_DEAD_BEEFu64;
            up.Unmap(0, None);

            // Readback buffer to inspect what the GPU itself sees at +0x100, +0x180, +0x200
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &up_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut rb,
            )
            .unwrap();
            let rb = rb.unwrap();

            // 1. DIRECT queue: WriteBufferImmediate at +0x100 and CopyBufferRegion at +0x200
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let dl: ID3D12GraphicsCommandList2 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let make_barrier = |before, after| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(buf.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            };

            dl.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let p_direct = [
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x100, Value: 0x0000_0677 },
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x104, Value: 0x0000_0000 },
            ];
            let m_direct = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; 2];
            dl.WriteBufferImmediate(2, p_direct.as_ptr(), Some(m_direct.as_ptr()));
            dl.CopyBufferRegion(&buf, 0x200, &up, 0, 8);
            dl.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            dl.Close().unwrap();
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            dq.ExecuteCommandLists(&[Some(dcl)]);
            dq.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 2000).unwrap();

            // 2. COMPUTE queue: WriteBufferImmediate at +0x180
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let ca: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let cl2: ID3D12GraphicsCommandList2 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca, None).unwrap();
            cl2.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let p_comp = [
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x180, Value: 0x0000_0677 },
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x184, Value: 0x1122_3344 },
            ];
            let m_comp = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; 2];
            cl2.WriteBufferImmediate(2, p_comp.as_ptr(), Some(m_comp.as_ptr()));
            cl2.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            cl2.Close().unwrap();
            let ccl: ID3D12CommandList = cl2.cast().unwrap();
            cq.ExecuteCommandLists(&[Some(ccl)]);
            cq.Signal(&fence, 2).unwrap();
            crate::gpu::fence::wait(&fence, 2, &dev, None, 2000).unwrap();

            // 3. GPU readback of +0x100, +0x180, +0x200
            let da2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let dl2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da2, None).unwrap();
            dl2.CopyBufferRegion(&rb, 0, &buf, 0x100, 8);
            dl2.CopyBufferRegion(&rb, 8, &buf, 0x180, 8);
            dl2.CopyBufferRegion(&rb, 16, &buf, 0x200, 8);
            dl2.Close().unwrap();
            let dcl2: ID3D12CommandList = dl2.cast().unwrap();
            dq.ExecuteCommandLists(&[Some(dcl2)]);
            dq.Signal(&fence, 3).unwrap();
            crate::gpu::fence::wait(&fence, 3, &dev, None, 2000).unwrap();

            let cpu_direct = *((base + 0x100) as *const u64);
            let cpu_comp = *((base + 0x180) as *const u64);
            let cpu_copy = *((base + 0x200) as *const u64);
            let mut rb_ptr = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut rb_ptr)).unwrap();
            let gpu_direct = *(rb_ptr as *const u64);
            let gpu_comp = *((rb_ptr as *const u64).add(1));
            let gpu_copy = *((rb_ptr as *const u64).add(2));
            rb.Unmap(0, None);

            drop(dl);
            drop(dl2);
            drop(cl2);
            drop(buf);
            drop(heap);
            VirtualFree(mem, 0, MEM_RELEASE);

            assert_eq!(cpu_direct, 0x677, "DIRECT WriteBufferImmediate must be visible to CPU (gpu={gpu_direct:#x})");
            assert_eq!(cpu_comp, 0x1122_3344_0000_0677, "COMPUTE WriteBufferImmediate must be visible to CPU (gpu={gpu_comp:#x})");
            assert_eq!(cpu_copy, 0xCAFE_BABE_DEAD_BEEF, "DIRECT CopyBufferRegion must be visible to CPU (gpu={gpu_copy:#x})");
        }
    }

    /// #31 SetPredication on a buffer placed below the title's address reads the predicate at the
    /// offset #30 ResolveQueryData writes it to (both add `real::buffer_delta`); a null buffer
    /// passes through.
    #[test]
    fn set_predication_x_adds_the_buffer_delta() {
        #[repr(C)]
        struct FakePredList {
            vt: *const usize,
            calls: Mutex<Vec<(usize, u64, i32)>>,
        }

        unsafe extern "system" fn fake_set_predication(this: *mut c_void, res: usize, off: u64, op: D3D12_PREDICATION_OP) {
            (*(this as *const FakePredList)).calls.lock().unwrap().push((res, off, op.0));
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_SET_PREDICATION] = fake_set_predication as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0];
        let fake = Box::new(FakePredList {
            // SAFETY: `ext_vt` has 2 entries so `.add(1)` has `orig_vt` at `-1` as `real::orig_fn` expects.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            calls: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakePredList as *mut c_void;
        let res = 0x31_5000usize;

        real::set_test_buffer_delta(res, Some(0x200));
        unsafe {
            set_predication_x(list_ptr, res, 0x10, D3D12_PREDICATION_OP_EQUAL_ZERO);
            set_predication_x(list_ptr, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
        real::set_test_buffer_delta(res, None);

        assert_eq!(*fake.calls.lock().unwrap(), vec![(res, 0x210, 0), (0, 0, 0)]);
        on_release(list_ptr as usize);
    }

    /// Predication set through #31 is set again after the PC Reset of KickoffX or a mid-list split
    /// (`restore_bindings`), on direct and compute lists, at the PC offset it was set with. A null
    /// buffer leaves nothing to restore, and the last of #31 / #85 SetPredicationBufferX is the
    /// one in effect.
    #[test]
    fn set_predication_x_is_restored_after_a_pc_reset() {
        #[repr(C)]
        struct FakePredList {
            vt: *const usize,
            calls: Mutex<Vec<(usize, u64, i32)>>,
        }

        unsafe extern "system" fn fake_get_type(_this: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            D3D12_COMMAND_LIST_TYPE_DIRECT
        }

        unsafe extern "system" fn fake_resource_barrier(_this: *mut c_void, _count: u32, _barriers: *const D3D12_RESOURCE_BARRIER) {}

        unsafe extern "system" fn fake_copy_buffer_region(_this: *mut c_void, _dst: usize, _dst_off: u64, _src: usize, _src_off: u64, _n: u64) {}

        unsafe extern "system" fn fake_set_predication(this: *mut c_void, res: usize, off: u64, op: D3D12_PREDICATION_OP) {
            (*(this as *const FakePredList)).calls.lock().unwrap().push((res, off, op.0));
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = fake_get_type as usize;
        orig_vt[PC_COPY_BUFFER_REGION] = fake_copy_buffer_region as usize;
        orig_vt[PC_RESOURCE_BARRIER] = fake_resource_barrier as usize;
        orig_vt[PC_SET_PREDICATION] = fake_set_predication as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0];
        let fake = Box::new(FakePredList {
            // SAFETY: `ext_vt` has 2 entries so `.add(1)` has `orig_vt` at `-1` as `real::orig_fn` expects.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            calls: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakePredList as *mut c_void;
        let res = 0x31_6000usize;
        let (uma_buf, pb) = (0x31_7000usize, 0x31_8000usize);
        let take = || std::mem::take(&mut *fake.calls.lock().unwrap());
        let restore = |ty: D3D12_COMMAND_LIST_TYPE| {
            let b = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list_ptr as usize))).map(|s| s.bindings.clone()).unwrap_or_default();
            take();
            unsafe { restore_bindings(list_ptr, &b, ty) };
            take()
        };
        let pred_85 = || LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list_ptr as usize))).and_then(|s| s.predication);

        real::set_test_buffer_delta(res, Some(0x200));
        let (direct, compute, null_restored, after_85, disarm, after_31, after_null_85);
        unsafe {
            set_predication_x(list_ptr, res, 0x10, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
            direct = restore(D3D12_COMMAND_LIST_TYPE_DIRECT);
            compute = restore(D3D12_COMMAND_LIST_TYPE_COMPUTE);
            set_predication_x(list_ptr, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
            null_restored = restore(D3D12_COMMAND_LIST_TYPE_DIRECT);

            // #85 after #31: the copy into the predication buffer runs unpredicated, and only
            // #85's state is restored (by KickoffX itself)
            set_predication_x(list_ptr, res, 0x10, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
            take();
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb, D3D12_PREDICATION_OP_EQUAL_ZERO);
            disarm = take();
            after_85 = (restore(D3D12_COMMAND_LIST_TYPE_DIRECT), pred_85());

            // #31 after #85: only #31's is restored
            set_predication_x(list_ptr, res, 0x10, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
            after_31 = (restore(D3D12_COMMAND_LIST_TYPE_DIRECT), pred_85());

            // #85 with no descriptor turns predication off: nothing to restore
            set_predication_buffer_x(list_ptr, std::ptr::null());
            after_null_85 = restore(D3D12_COMMAND_LIST_TYPE_DIRECT);
        }
        real::set_test_buffer_delta(res, None);
        on_release(list_ptr as usize);

        assert_eq!(direct, vec![(res, 0x210, 1)]);
        assert_eq!(compute, vec![(res, 0x210, 1)]);
        assert!(null_restored.is_empty());
        assert_eq!(disarm, vec![(0, 0, 0), (pb, 0, 0)]);
        assert_eq!(after_85, (vec![], Some(D3D12_PREDICATION_OP_EQUAL_ZERO)));
        assert_eq!(after_31, (vec![(res, 0x210, 1)], None));
        assert!(after_null_85.is_empty());
    }

    /// Submitting a DIRECT list with `Wait64BitValueX(va, >=, 0x677)` BEFORE submitting the
    /// COMPUTE list that recorded `Write64BitValueBottomOfPipeX(va, 0x677)` must bridge via
    /// GPU `queue.Wait`/`queue.Signal` without blocking the submitter thread for `WAIT_TIMEOUT`.
    #[test]
    fn cross_queue_wait64_before_signal_does_not_deadlock_submit_thread() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let ca: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let dl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let cl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca, None).unwrap();
            let done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as u64;

            let dl_ptr = dl.as_raw();
            let cl_ptr = cl.as_raw();
            on_create(dl_ptr, da.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(cl_ptr, ca.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Worker threads record both lists: COMPUTE writes `va = 0x677`, DIRECT waits `va >= 0x677`.
            note_write64_recorded(cl_ptr, va, 0x677);
            wait64_x(dl_ptr, va, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x677, 1, 0);
            dl.Close().unwrap();
            cl.Close().unwrap();

            // Single submitter thread submits DIRECT (consumer) FIRST — before COMPUTE (producer)!
            let t0 = Instant::now();
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            execute_with_waits_on(dq.as_raw(), &[dl_ptr], |chunk| {
                assert_eq!(chunk, &[dl_ptr]);
                dq.ExecuteCommandLists(&[Some(dcl.clone())]);
                signal_write64_fences(dq.as_raw(), chunk);
            });
            let direct_submit_ms = t0.elapsed().as_millis();
            assert!(
                direct_submit_ms < 500,
                "wait-before-signal must bridge to GPU queue.Wait without blocking CPU (took {direct_submit_ms} ms)"
            );
            dq.Signal(&done_fence, 1).unwrap();

            // Now submit COMPUTE (producer) which signals the per-VA GPU fence.
            let ccl: ID3D12CommandList = cl.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl_ptr], |chunk| {
                assert_eq!(chunk, &[cl_ptr]);
                cq.ExecuteCommandLists(&[Some(ccl.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });

            // DIRECT queue must now unblock and finish promptly on the GPU.
            crate::gpu::fence::wait(&done_fence, 1, &dev, None, 2000).expect("DIRECT queue must complete after COMPUTE signals per-VA fence");

            on_release(dl_ptr as usize);
            on_release(cl_ptr as usize);
            SYNC_GRAPH.lock().unwrap().as_mut().map(|gr| gr.cells.remove(&va));
        }
    }

    /// (`single`) A list whose `Wait64BitValueX` is for a value no queue has been given the
    /// write of is found, with the wait; once the writing list has gone to the queue the wait
    /// holds nothing, and the waiting list runs after it on the same queue with no GPU wait.
    #[test]
    fn a_wait64_is_unwritten_until_the_writing_list_is_handed_over() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let (wa, ra): (ID3D12CommandAllocator, ID3D12CommandAllocator) = (dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(), dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap());
            let waiter: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &wa, None).unwrap();
            let writer: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &ra, None).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as u64;
            let (waiter_ptr, writer_ptr) = (waiter.as_raw(), writer.as_raw());
            on_create(waiter_ptr, wa.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(writer_ptr, ra.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            note_write64_recorded(writer_ptr, va, 0x688);
            wait64_x(waiter_ptr, va, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x688, 1, 0);
            waiter.Close().unwrap();
            writer.Close().unwrap();

            let (waiter_l, writer_l) = (waiter_ptr as usize, writer_ptr as usize);
            assert_eq!(first_unwritten_wait(&[writer_l, waiter_l]), Some((1, va, 0x688)));
            assert!(!cell_written(va, 0x688));
            assert!(lists_write(&[writer_l], va, 0x688) && !lists_write(&[writer_l], va, 0x689) && !lists_write(&[waiter_l], va, 0x688));

            let cl: ID3D12CommandList = writer.cast().unwrap();
            execute_with_waits_on(q.as_raw(), &[writer_ptr], |chunk| {
                q.ExecuteCommandLists(&[Some(cl.clone())]);
                signal_write64_fences(q.as_raw(), chunk);
            });
            assert!(cell_written(va, 0x688));
            assert_eq!(first_unwritten_wait(&[waiter_l]), None);

            let cl: ID3D12CommandList = waiter.cast().unwrap();
            let t0 = Instant::now();
            execute_with_waits_on(q.as_raw(), &[waiter_ptr], |_| q.ExecuteCommandLists(&[Some(cl.clone())]));
            assert!(t0.elapsed() < Duration::from_millis(500), "no CPU wait: the write went to the same queue before");
            q.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 2000).expect("no GPU wait is left on the queue");

            on_release(waiter_l);
            on_release(writer_l);
            SYNC_GRAPH.lock().unwrap().as_mut().map(|gr| gr.cells.remove(&va));
        }
    }

    /// Cross-queue cycle detection (`DIRECT` waits on `COMPUTE` while `COMPUTE` is already
    /// blocked on an unsubmitted write from `DIRECT`) and same-queue FIFO bypass must both
    /// complete immediately without stalling CPU or hanging GPU.
    #[test]
    fn cross_queue_cycle_and_same_queue_fifo_do_not_deadlock() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let ca1: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let ca2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let dl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let cl1: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca1, None).unwrap();
            let cl2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca2, None).unwrap();
            let done_d: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done_c: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(0));
            let va_a = &*cell_a as *const AtomicU64 as u64;
            let va_b = &*cell_b as *const AtomicU64 as u64;

            let (dl_ptr, cl1_ptr, cl2_ptr) = (dl.as_raw(), cl1.as_raw(), cl2.as_raw());
            on_create(dl_ptr, da.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(cl1_ptr, ca1.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(cl2_ptr, ca2.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Fork-join cycle: DIRECT writes A then waits on B; COMPUTE1 waits on A then writes B;
            // COMPUTE2 waits on B (same-queue FIFO after COMPUTE1) after writing something else.
            note_write64_recorded(dl_ptr, va_a, 0x702);
            wait64_x(dl_ptr, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            wait64_x(cl1_ptr, va_a, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            note_write64_recorded(cl1_ptr, va_b, 0x702);
            wait64_x(cl2_ptr, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            dl.Close().unwrap();
            cl1.Close().unwrap();
            cl2.Close().unwrap();

            let t0 = Instant::now();
            let ccl1: ID3D12CommandList = cl1.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl1_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(ccl1.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            // Same-queue FIFO bypass: cl2 waits on va_b already submitted on cq.
            let ccl2: ID3D12CommandList = cl2.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl2_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(ccl2.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            cq.Signal(&done_c, 1).unwrap();

            // Cycle breaking: dl writes va_a (which cq is blocked on) and waits on va_b (from cq).
            // Must detect the cycle, skip hoisting va_b ahead of dl, submit dl immediately, and unblock BOTH queues!
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            execute_with_waits_on(dq.as_raw(), &[dl_ptr], |chunk| {
                dq.ExecuteCommandLists(&[Some(dcl.clone())]);
                signal_write64_fences(dq.as_raw(), chunk);
            });
            dq.Signal(&done_d, 1).unwrap();

            assert!(t0.elapsed().as_millis() < 500, "submits must not stall CPU (took {} ms)", t0.elapsed().as_millis());
            crate::gpu::fence::wait(&done_d, 1, &dev, None, 2000).expect("DIRECT must finish without GPU deadlock");
            crate::gpu::fence::wait(&done_c, 1, &dev, None, 2000).expect("COMPUTE must finish once DIRECT signals A");

            on_release(dl_ptr as usize);
            on_release(cl1_ptr as usize);
            on_release(cl2_ptr as usize);
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_a);
                gr.cells.remove(&va_b);
            }
        }
    }

    /// Exact `(queue, seq)` cycle detection must catch k=1 self-deadlocks, k=2 and k=3 transitive
    /// cycles (including mixed title-fence + Wait64 cycles) while NEVER false-positiving on
    /// pipelined frame N/N+1 or back-to-back cross-queue handshakes (`wait_seq > prod_sig_seq`).
    #[test]
    fn pipelined_next_frame_write_is_not_false_cycle() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();

            // 1. Back-to-back handshake (NOT a cycle):
            //    q0 signals A=10 at seq=2, then waits on B=10 at seq=3.
            //    q1 waits on A=10 at seq=1, then wants to signal B=10 (seq=2).
            //    Since q0's wait on B (seq=3) is AFTER q0's signal of A (seq=2), NOT a cycle!
            let f_a: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f_b: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut g = SyncGraph::default();
            g.cells.insert(
                0x1000,
                SyncCell {
                    fence: Some(f_a),
                    device: dev.as_raw() as usize,
                    recorded_max: 10,
                    submitted_max: 10,
                    submitted_queue: Some(0xC001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 10,
                        waiting_queue: 0xD001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 10, queue: 0xC001, seq: 2 }],
                    ..Default::default()
                },
            );
            g.cells.insert(
                0x2000,
                SyncCell {
                    fence: Some(f_b),
                    device: dev.as_raw() as usize,
                    recorded_max: 10,
                    armed_waits: vec![ArmedGpuWait {
                        val: 10,
                        waiting_queue: 0xC001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 3, // AFTER q0's signal of A (seq=2)!
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    ..Default::default()
                },
            );
            assert!(
                !would_signal_close_cycle(&mut g, 0xD001, 0x2000, 10),
                "back-to-back handshake (wait_seq=3 > sig_seq=2) must NOT be treated as a cycle"
            );
            // Now invert q0's order: q0 waited on B at seq=1 BEFORE signaling A at seq=2 -> TRUE cycle!
            g.cells.get_mut(&0x2000).unwrap().armed_waits[0].wait_seq = 1;
            assert!(
                would_signal_close_cycle(&mut g, 0xD001, 0x2000, 10),
                "2-queue cycle (wait_seq=1 < sig_seq=2) MUST be detected at Signal time"
            );
            // k = 1 self-deadlock: q0 signaling B when q0 itself is waiting on B.
            assert!(
                would_signal_close_cycle(&mut g, 0xC001, 0x2000, 10),
                "same-queue self-deadlock MUST be detected at Signal time"
            );

            // 2. 3-queue transitive cycle: Direct -> C1 -> C2 -> Direct.
            //    C2 waits on D (0x3000, seq=1), then signals C2_out (0x2000, seq=2).
            //    C1 waits on C2_out (0x2000, seq=1), then signals C1_out (0x1000, seq=2).
            //    Direct waits on C1_out (0x1000, seq=1), then wants to signal D (0x3000).
            let f1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f3: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut g3 = SyncGraph::default();
            g3.cells.insert(
                0x1000,
                SyncCell {
                    fence: Some(f1),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    submitted_max: 0x78d,
                    submitted_queue: Some(0xC001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xD001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x78d, queue: 0xC001, seq: 2 }],
                    ..Default::default()
                },
            );
            g3.cells.insert(
                0x2000,
                SyncCell {
                    fence: Some(f2),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    submitted_max: 0x78d,
                    submitted_queue: Some(0xC002),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xC001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x78d, queue: 0xC002, seq: 2 }],
                    ..Default::default()
                },
            );
            g3.cells.insert(
                0x3000,
                SyncCell {
                    fence: Some(f3),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xC002,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    ..Default::default()
                },
            );
            assert!(
                would_signal_close_cycle(&mut g3, 0xD001, 0x3000, 0x78d),
                "3-queue transitive cycle Direct -> C1 -> C2 -> Direct MUST be detected at Signal time"
            );
            // If C2 waited on D AFTER signaling C2_out (wait_seq=3 > sig_seq=2), NOT a cycle.
            g3.cells.get_mut(&0x3000).unwrap().armed_waits[0].wait_seq = 3;
            assert!(
                !would_signal_close_cycle(&mut g3, 0xD001, 0x3000, 0x78d),
                "pipelined 3-queue chain with wait_seq > sig_seq must NOT be treated as a cycle"
            );
        }
    }

    /// Reproduces two real multi-queue hazards seen in retail titles:
    /// 1. Rotating compute queues where cell `va` was previously written by `cq0` in frame N,
    ///    then in frame N+1 `cq1` waits on `va` before `cq1` itself submits the write to `va`
    ///    (same-queue self-deadlock despite stale `submitted_queue == cq0`).
    /// 2. A Wait64 cell becoming non-monotonic (`value < submitted_max`) while another queue
    ///    already has an armed `queue.Wait` on it — must rescue the waiter instead of orphaning it.
    #[test]
    fn rotating_queue_self_deadlock_and_non_monotonic_rescue_do_not_hang() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let cq0: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let cq1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let a0: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a1: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l0: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a0, None).unwrap();
            let l1: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a1, None).unwrap();
            let l2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a2, None).unwrap();
            let done1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell_rot = Box::new(AtomicU64::new(0));
            let va_rot = &*cell_rot as *const AtomicU64 as u64;
            let (l0_ptr, l1_ptr, l2_ptr) = (l0.as_raw(), l1.as_raw(), l2.as_raw());
            on_create(l0_ptr, a0.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l1_ptr, a1.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l2_ptr, a2.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Frame 0x69f: cq0 writes va_rot = 0x69f (sets stale submitted_queue = cq0).
            note_write64_recorded(l0_ptr, va_rot, 0x69f);
            l0.Close().unwrap();
            let cl0: ID3D12CommandList = l0.cast().unwrap();
            execute_with_waits_on(cq0.as_raw(), &[l0_ptr], |chunk| {
                cq0.ExecuteCommandLists(&[Some(cl0.clone())]);
                signal_write64_fences(cq0.as_raw(), chunk);
            });

            // Frame 0x6a0: l2 records write va_rot = 0x6a0, l1 waits on va_rot >= 0x6a0.
            // Both l1 and l2 are submitted on cq1 (rotated from cq0 -> cq1)!
            note_write64_recorded(l2_ptr, va_rot, 0x6a0);
            wait64_x(l1_ptr, va_rot, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x6a0, 1, 0);
            l1.Close().unwrap();
            l2.Close().unwrap();

            let t0 = Instant::now();
            let cl1: ID3D12CommandList = l1.cast().unwrap();
            let cl2: ID3D12CommandList = l2.cast().unwrap();
            execute_with_waits_on(cq1.as_raw(), &[l1_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl1.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            execute_with_waits_on(cq1.as_raw(), &[l2_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl2.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            cq1.Signal(&done1, 1).unwrap();

            // Must break self-deadlock immediately on CPU at Signal time (not wait 5s for watchdog!).
            crate::gpu::fence::wait(&done1, 1, &dev, None, 1500).expect("cq1 must complete immediately without self-deadlock");
            assert!(t0.elapsed().as_millis() < 1500, "took {} ms", t0.elapsed().as_millis());

            // Part 2: Non-monotonic rescue while a GPU wait is armed.
            let cell_nm = Box::new(AtomicU64::new(0));
            let va_nm = &*cell_nm as *const AtomicU64 as u64;
            let a3: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a4: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a5: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l3: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a3, None).unwrap();
            let l4: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a4, None).unwrap();
            let l5: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a5, None).unwrap();
            let done2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (l3_ptr, l4_ptr, l5_ptr) = (l3.as_raw(), l4.as_raw(), l5.as_raw());
            on_create(l3_ptr, a3.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l4_ptr, a4.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l5_ptr, a5.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Submit 10 first -> submitted_max = 10.
            note_write64_recorded(l3_ptr, va_nm, 10);
            l3.Close().unwrap();
            let cl3: ID3D12CommandList = l3.cast().unwrap();
            execute_with_waits_on(cq0.as_raw(), &[l3_ptr], |chunk| {
                cq0.ExecuteCommandLists(&[Some(cl3.clone())]);
                signal_write64_fences(cq0.as_raw(), chunk);
            });
            // Record 20 (unsubmitted), then arm cq1 wait for >= 20.
            note_write64_recorded(l4_ptr, va_nm, 20);
            wait64_x(l5_ptr, va_nm, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 20, 1, 0);
            l5.Close().unwrap();
            let cl5: ID3D12CommandList = l5.cast().unwrap();
            execute_with_waits_on(cq1.as_raw(), &[l5_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl5.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            cq1.Signal(&done2, 1).unwrap();
            // Now record a strictly decreasing write (5 < submitted_max 10) -> non_monotonic!
            // Must rescue cq1's armed wait on CPU before dropping the fence!
            note_write64_recorded(l4_ptr, va_nm, 5);
            crate::gpu::fence::wait(&done2, 1, &dev, None, 1500).expect("cq1 must be rescued when cell goes non-monotonic");

            for p in [l0_ptr, l1_ptr, l2_ptr, l3_ptr, l4_ptr, l5_ptr] {
                on_release(p as usize);
            }
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_rot);
                gr.cells.remove(&va_nm);
            }
        }
    }

    /// Verifies that CPU-signalling a fence while a GPU `queue.Signal` to the same value is
    /// already in flight (watchdog release path) does not hang subsequent queue completions.
    #[test]
    fn cpu_watchdog_signal_with_inflight_gpu_signal_completes_cleanly() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let f_a: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f_b: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // Deliberate 2-queue cycle with in-flight GPU signals to value 10 on both fences.
            q1.Wait(&f_a, 10).unwrap();
            q1.Signal(&f_b, 10).unwrap();
            q1.Signal(&done1, 1).unwrap();
            q2.Wait(&f_b, 10).unwrap();
            q2.Signal(&f_a, 10).unwrap();
            q2.Signal(&done2, 1).unwrap();

            // Simulate watchdog breaking the deadlock from CPU on both fences while GPU signals are in flight.
            f_a.Signal(10).unwrap();
            f_b.Signal(10).unwrap();

            crate::gpu::fence::wait(&done1, 1, &dev, None, 2000).expect("q1 must complete after watchdog CPU signal");
            crate::gpu::fence::wait(&done2, 1, &dev, None, 2000).expect("q2 must complete after watchdog CPU signal");
        }
    }

    /// Reproduces the exact intra-list fork-join pattern seen on retail compute queues:
    ///   q_compute list O_c: Write64(cell_A, frame) -> Wait64(cell_B >= frame) -> Write64(cell_C, frame)
    ///   q_direct  list O_d: Wait64(cell_A >= frame) -> Write64(cell_B, frame)
    /// Without deferred mid-list splitting, hoisting Wait64(cell_B) ahead of Write64(cell_A) creates
    /// a 2-queue deadlock cycle every frame. With deferred splitting, O_c expands into `[pc_c0, pc_c1]`,
    /// completes on GPU with ZERO CPU cycle breaks, and reuses `pc_c1` from the spare pool on Reset.
    #[test]
    fn mid_list_write64_wait64_split_avoids_cycle_and_reuses_spares() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q_comp: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let q_dir: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let a_comp: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a_dir: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let raw_c: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a_comp, None).unwrap();
            let raw_d: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a_dir, None).unwrap();
            let done_c: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done_d: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // Transfer raw_c into `title_list::wrap` (the wrapper owns the COM ref).
            let pc_c0 = raw_c.into_raw();
            real::adopt(pc_c0, real::Kind::CommandList);
            on_create(pc_c0, a_comp.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            super::super::bind::reset(pc_c0, a_comp.as_raw() as usize, 0);
            let wrap_c = super::super::title_list::wrap(pc_c0);

            let pc_d0 = raw_d.as_raw();
            real::adopt(pc_d0, real::Kind::CommandList);
            on_create(pc_d0, a_dir.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);

            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(0));
            let cell_c = Box::new(AtomicU64::new(0));
            let va_a = &*cell_a as *const AtomicU64 as u64;
            let va_b = &*cell_b as *const AtomicU64 as u64;
            let va_c = &*cell_c as *const AtomicU64 as u64;

            let mut first_seg1: *mut c_void = std::ptr::null_mut();
            for frame in 1u64..=2 {
                if frame > 1 {
                    a_comp.Reset().unwrap();
                    a_dir.Reset().unwrap();
                    // Reset through `title_list` slot #10: must switch back to `pc_c0` and keep `seg1` as spare.
                    let reset_fn: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 =
                        std::mem::transmute(*(*(wrap_c as *const *const usize)).add(10));
                    assert_eq!(reset_fn(wrap_c, a_comp.as_raw() as usize, 0), 0);
                    assert_eq!(reset_x(pc_d0, a_dir.as_raw() as usize, 0), 0);
                }

                // Record O_d: Wait64(A >= frame) -> Write64(B, frame) -> Close (seed cell_B recorded
                // first so O_c's Wait64(B) bridges to GPU even though O_c submits before O_d).
                note_write64_recorded(pc_c0, va_a, frame);
                let mut g = SYNC_GRAPH.lock().unwrap();
                let cell_b_entry = g.get_or_insert_with(SyncGraph::default).cells.entry(va_b).or_default();
                cell_b_entry.recorded_max = cell_b_entry.recorded_max.max(frame);
                cell_b_entry.recorded_list_type = Some(D3D12_COMMAND_LIST_TYPE_DIRECT);
                drop(g);
                // Call wait64_x on the active PC segment (`pc_c0`); it looks up `wrap_c`, splits into
                // `[pc_c0, seg1]`, and switches `wrap_c`'s `F_BACKING` to `seg1`.
                wait64_x(pc_c0, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, frame, 0, 0);

                // Verify the split created/reused a second segment.
                let mut segs_c = Vec::new();
                assert!(super::super::title_list::push_segments(wrap_c, &mut segs_c));
                assert_eq!(segs_c.len(), 2, "frame {frame}: mid-list Wait64 after Write64 must split into 2 segments");
                assert_eq!(segs_c[0], pc_c0);
                assert_ne!(segs_c[1], pc_c0);
                if frame == 1 {
                    first_seg1 = segs_c[1];
                } else {
                    assert_eq!(segs_c[1], first_seg1, "frame 2 split must reuse spare PC command list");
                }

                // Write64(C, frame) lands on the second segment.
                note_write64_recorded(segs_c[1], va_c, frame);

                // Close O_c via title_list slot #9 (closes segs_c[1]; segs_c[0] was already closed on split).
                let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 =
                    std::mem::transmute(*(*(wrap_c as *const *const usize)).add(9));
                assert_eq!(close_fn(wrap_c), 0, "frame {frame}: Close on split title_list failed");

                // Record O_d: Wait64(A >= frame) -> Write64(B, frame) -> Close.
                wait64_x(pc_d0, va_a, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, frame, 0, 0);
                note_write64_recorded(pc_d0, va_b, frame);
                let cl_d_raw = ID3D12GraphicsCommandList::from_raw_borrowed(&pc_d0).unwrap();
                cl_d_raw.Close().unwrap();

                // Execute O_c segments on q_comp FIRST (just like the real title!).
                execute_with_waits_on(q_comp.as_raw(), &segs_c, |chunk| {
                    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
                    let exec: ExecFn = std::mem::transmute(*(*(q_comp.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
                    exec(q_comp.as_raw(), chunk.len() as u32, chunk.as_ptr());
                    signal_write64_fences(q_comp.as_raw(), chunk);
                    if chunk.contains(&pc_c0) {
                        cell_a.store(frame, Ordering::Release);
                    }
                });
                q_comp.Signal(&done_c, frame).unwrap();

                // Execute O_d on q_dir SECOND.
                let one_d = [pc_d0];
                execute_with_waits_on(q_dir.as_raw(), &one_d, |chunk| {
                    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
                    let exec: ExecFn = std::mem::transmute(*(*(q_dir.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
                    exec(q_dir.as_raw(), chunk.len() as u32, chunk.as_ptr());
                    signal_write64_fences(q_dir.as_raw(), chunk);
                    cell_b.store(frame, Ordering::Release);
                });
                q_dir.Signal(&done_d, frame).unwrap();

                crate::gpu::fence::wait(&done_c, frame, &dev, None, 2000).expect("compute queue must complete without cycle");
                crate::gpu::fence::wait(&done_d, frame, &dev, None, 2000).expect("direct queue must complete without cycle");

                // CRITICAL: verify ZERO CPU cycle breaks occurred on ANY of the 3 cells!
                let g = SYNC_GRAPH.lock().unwrap();
                let gr = g.as_ref().unwrap();
                for (name, va) in [("cell_A", va_a), ("cell_B", va_b), ("cell_C", va_c)] {
                    let c = gr.cells.get(&va).unwrap();
                    assert_eq!(c.cpu_signaled_max, 0, "frame {frame}: {name} must NOT trigger a CPU cycle break");
                    assert_eq!(c.submitted_max, frame, "frame {frame}: {name} must reach frame {frame} on GPU");
                }
            }

            // Release wrapped list (must release primary + spare segment cleanly).
            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(*(*(wrap_c as *const *const usize)).add(2));
            assert_eq!(rel_fn(wrap_c), 0);
            on_release(pc_d0 as usize);
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_a);
                gr.cells.remove(&va_b);
                gr.cells.remove(&va_c);
            }
        }
    }
}
