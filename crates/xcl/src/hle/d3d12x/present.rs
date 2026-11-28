//! Presentation for the real backend.
//!
//! Console titles do not use DXGI swap chains: they render into their own back buffers
//! and call `PresentX` – command queue slot 30, `PresentX(planeCount, planes, parameters)`
//! with plane 0 `{ u64 frame token; u32 resource count @8; ID3D12Resource** @0x10 }` (as
//! observed on Fortnite). XCL queues the composition before telling the frame clock (`frame`)
//! the token flipped. The host window presents retired compositions on its own queue.
//!
//! Until a title's first `PresentX`, a fallback guesses the frame: XCL remembers which
//! resource every RTV descriptor points to and the first render target of each command list,
//! and after ExecuteCommandLists presents the latest back-buffer-like render target – on a
//! call of another console-only queue method, or rate-limited after submissions.
//!
//! The back buffer is copied on the title's own queue (implicit promotion from
//! COMMON/PRESENT state, which consoles require before PresentX).

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Once, OnceLock, TryLockError};
use std::time::{Duration, Instant};

use windows::core::{Interface, HSTRING, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, DXGI_STATUS_OCCLUDED, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, S_OK, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromPoint, MONITORINFO, MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY, MONITOR_FROM_FLAGS};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::CreateEventW;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::gpu::fence::{self, FenceError};
use crate::stubs::StubInfo;

static RTVS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static LISTS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static CANDIDATE: Mutex<Option<(usize, usize)>> = Mutex::new(None); // (queue, resource) – resource AddRef'd
static PRESENTS: AtomicU64 = AtomicU64::new(0);
/// When a frame was last presented, in `now_ms` (0: none yet): the background compiles give way
/// to a title that draws.
static LAST_PRESENT_MS: AtomicU64 = AtomicU64::new(0);
static HOST_PRESENTS: AtomicU64 = AtomicU64::new(0);
static TITLE_PRESENT_CALLS: AtomicU64 = AtomicU64::new(0);
static HOST_OCCLUDED: AtomicU64 = AtomicU64::new(0);
static HOST_STILL_DRAWING: AtomicU64 = AtomicU64::new(0);
static LAST_FRAME_DUMP: Mutex<Option<FrameDumpSample>> = Mutex::new(None);
static BOUNDARIES: AtomicU64 = AtomicU64::new(0);
static LAST_BOUNDARY_MS: AtomicU64 = AtomicU64::new(0);
static HWND_RAW: AtomicIsize = AtomicIsize::new(0);
/// the title called PresentX: the fallback guessing is off
static PRESENTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PRESENTX_LOCK: Mutex<()> = Mutex::new(());

/// command queue #30 PresentX(planeCount, planes, parameters): plane 0's first resource is
/// the frame; S_OK always (a console present does not fail for the title).
pub unsafe extern "system" fn present_x(queue: *mut c_void, plane_count: u32, planes: *const u8, _params: *const u8) -> i32 {
    if plane_count == 0 || planes.is_null() || !crate::gpu::shader::readable(planes as usize, 0x18) {
        return 0;
    }
    let tokens: Vec<u64> = (0..plane_count as usize)
        .map(|k| planes.add(PLANE_BYTES * k))
        .take_while(|&plane| crate::gpu::shader::readable(plane as usize, 8))
        .map(|plane| (plane as *const u64).read_unaligned())
        .collect();
    crate::control::wait_while_paused();
    let _present = PRESENTX_LOCK.lock().unwrap();
    TITLE_PRESENT_CALLS.fetch_add(1, Ordering::Relaxed);
    // The title can use flip statistics to decide when to reuse its buffer.
    // Queue our copy before reporting the flip.
    super::present_order::present(&tokens, |show| present_x_frame(queue, plane_count, planes, show));
    0
}

/// `show`: false for a present older than the frame on screen (`present_order`), which is
/// accounted for like any other but not composed.
unsafe fn present_x_frame(queue: *mut c_void, plane_count: u32, planes: *const u8, show: bool) {
    super::gpu_timing::boundary();
    if super::gpu_timing::enabled() {
        super::gpu_timing_waits::boundary(super::gpu_timing::epoch());
    }
    let frames = BOUNDARIES.fetch_add(1, Ordering::Relaxed) + 1;
    super::census::present(frames);
    super::depth_uav::probe_frame();
    if frames % 300 == 0 {
        super::dsv_trace::log_frame();
        // the title's frame rate, for comparing runs (every 300 frames)
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        if let Some(t) = LAST.lock().unwrap().replace(now) {
            crate::xlog_info!("D3D12.X: {frames} frames presented, {:.1} fps over the last 300", 300.0 / (now - t).as_secs_f64());
        }
        super::depth_uav::log_frame();
    }
    super::real::stop_if_device_removed();
    if !PRESENTX.swap(true, Ordering::Relaxed) {
        crate::xlog_console!("D3D12.X: PresentX – the title presents its own frames ({plane_count} plane(s))");
    }
    {
        // a change in the number of planes (the console's display composes them – a UI
        // plane over the game's): logged with the planes' raw words
        static LAST: AtomicU64 = AtomicU64::new(1);
        if LAST.swap(plane_count as u64, Ordering::Relaxed) != plane_count as u64 {
            let n = 8 * plane_count as usize;
            let words: Vec<u64> = if crate::gpu::shader::readable(planes as usize, 8 * n) { (0..n).map(|k| (planes as *const u64).add(k).read_unaligned()).collect() } else { Vec::new() };
            crate::xlog_info!("D3D12.X: PresentX now with {plane_count} plane(s): {words:x?}");
        }
    }
    if !enabled() {
        return;
    }
    let shown = extract_planes(plane_count, planes);
    let Some(&(res, _)) = shown.first() else { return };
    {
        let mut p = PRESENTED.lock().unwrap();
        for &(r, _) in &shown {
            if !p.contains(&r) {
                p.push(r);
            }
        }
    }
    if trace_frame() {
        super::list_x::dump_pending_geometry();
    }
    if trace_frame() {
        static K: AtomicU64 = AtomicU64::new(0);
        let k = K.fetch_add(1, Ordering::Relaxed);
        if k == 2 || k == 300 {
            super::console_heap::census();
        }
    }
    if trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 3 || n % 50 == 0 {
            let u = USES.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or_default();
            let d = ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0, d.Flags.0));
            let busiest: Vec<String> = {
                let g = USES.lock().unwrap();
                let mut v: Vec<(usize, [u64; 4])> = g.as_ref().map(|m| m.iter().map(|(&k, &u)| (k, u)).collect()).unwrap_or_default();
                v.sort_by_key(|(_, u)| std::cmp::Reverse(u[0] + u[1] + u[2]));
                v.iter().take(6).map(|(k, u)| format!("{k:#x}: rt {} uav {} copy {} clear {}", u[0], u[1], u[2], u[3])).collect()
            };
            crate::xlog_info!("frame trace: PresentX #{n} of {res:#x} {:?} – rt binds {}, UAV views {}, copy destination {}, clears {}; most used: {}", d, u[0], u[1], u[2], u[3], busiest.join("; "));
        }
    }
    if show {
        present_now(Duration::ZERO, Some((queue as usize, &shown)));
    }
}

/// `D3D12XBOX_PRESENT_PLANE_PARAMETERS`: token @0, resource count @8, resources @0x10, source
/// rects @0x18, destination placement @0x20, colour space / filter / flags after them (a
/// retail game's two planes are 0x48 bytes apart)
const PLANE_BYTES: usize = 0x48;

/// planes read from one PresentX at most
const MAX_PLANES: usize = 8;

/// A plane's whole resource as source UVs `[left, top, right, bottom]`.
const FULL_UV: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Each plane's first resource and its source rect (`RECT { left, top, right, bottom }` behind
/// the pointer @0x18; None when that pointer is null or unreadable), bottom (the game's) to top
/// (a UI plane over it). Adapted from ivall/xcl 2ca749ab `extract_planes`.
unsafe fn extract_planes(plane_count: u32, planes: *const u8) -> Vec<(usize, Option<[i32; 4]>)> {
    (0..(plane_count as usize).min(MAX_PLANES))
        .filter_map(|k| {
            let plane = planes.add(PLANE_BYTES * k);
            if !crate::gpu::shader::readable(plane as usize, 0x18) {
                return None;
            }
            let count = (plane.add(8) as *const u32).read_unaligned();
            let list = (plane.add(0x10) as *const usize).read_unaligned();
            if count == 0 || !crate::gpu::shader::readable(list, 8) {
                return None;
            }
            let res = Some((list as *const usize).read_unaligned()).filter(|&r| r != 0)?;
            let rect = if crate::gpu::shader::readable(plane as usize, 0x20) { (plane.add(0x18) as *const usize).read_unaligned() } else { 0 };
            let rect = crate::gpu::shader::readable(rect, 16).then(|| std::array::from_fn(|k| (rect as *const i32).add(k).read_unaligned()));
            Some((res, rect))
        })
        .collect()
}

/// The source UVs of `rect` on a `w` x `h` plane: the whole plane without a rect, or for one
/// that is empty, inverted or reaches outside the plane. ivall/xcl 2ca749ab `source_uv_rect`.
fn source_uv_rect(rect: Option<[i32; 4]>, w: u32, h: u32) -> [f32; 4] {
    let Some([l, t, r, b]) = rect else { return FULL_UV };
    if !(0 <= l && l < r && i64::from(r) <= i64::from(w) && 0 <= t && t < b && i64::from(b) <= i64::from(h)) {
        return FULL_UV;
    }
    [l as f32 / w as f32, t as f32 / h as f32, r as f32 / w as f32, b as f32 / h as f32]
}

/// `XCL_PRESENT_SOURCE_RECTS=0`: every plane is shown whole, its source rect is ignored and the
/// composer takes planes without a shader view as it did (the behaviour before source rects were
/// read; only the 8-plane cap of `extract_planes` stays).
fn source_rects_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PRESENT_SOURCE_RECTS").map_or(true, |v| v.trim() != "0"))
}

/// The part of plane `k` (`w` x `h`) the display shows (`source_uv_rect`); the first source rect
/// that is not the whole plane is logged once.
fn shown_uv(k: usize, rect: Option<[i32; 4]>, w: u32, h: u32) -> [f32; 4] {
    let uv = source_uv_rect(rect, w, h);
    if uv != FULL_UV {
        static LOGGED: Once = Once::new();
        LOGGED.call_once(|| {
            let off = if source_rects_enabled() { "" } else { " – ignored (XCL_PRESENT_SOURCE_RECTS=0)" };
            crate::xlog_info!("present: plane {k} shows source rect {rect:?} of its {w}x{h} target (uv {uv:?}){off}");
        });
    }
    if source_rects_enabled() { uv } else { FULL_UV }
}

thread_local! {
    static IN_PRESENT: Cell<bool> = const { Cell::new(false) };
}

fn now_ms() -> u64 {
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub fn counters() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), BOUNDARIES.load(Ordering::Relaxed))
}

/// How long ago the last frame was presented (`None`: none yet). A title that presents is drawing
/// now, however slowly, and the background compiles give way to it (`pso_async::pace`).
#[cfg(not(test))]
pub fn since_present() -> Option<std::time::Duration> {
    let at = LAST_PRESENT_MS.load(Ordering::Relaxed);
    (at != 0).then(|| std::time::Duration::from_millis(now_ms().saturating_sub(at)))
}

fn enabled() -> bool {
    !crate::state::config().present.eq_ignore_ascii_case("off")
}

unsafe fn add_ref(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
    f(p);
}

unsafe fn release(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(2));
    f(p);
}

pub unsafe fn note_rtv(handle: usize, res: *mut c_void) {
    if !enabled() {
        return;
    }
    let mut g = RTVS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !res.is_null() {
        add_ref(res as usize);
    }
    let old = if res.is_null() { m.remove(&handle) } else { m.insert(handle, res as usize) };
    if let Some(o) = old {
        release(o);
    }
}

/// `XCL_TRACE_FRAME=1`: what console code does with each resource it later presents –
/// render-target binds, UAV views, copy destinations, clears (logged at the first presents)
static USES: Mutex<Option<HashMap<usize, [u64; 4]>>> = Mutex::new(None);
pub const USE_RT: usize = 0;
pub const USE_UAV: usize = 1;
pub const USE_COPY: usize = 2;
pub const USE_CLEAR: usize = 3;

pub fn trace_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_FRAME").is_some())
}

/// `res` was used as `what` (USE_*)
pub fn note_use(res: usize, what: usize) {
    if trace_frame() && res != 0 {
        USES.lock().unwrap().get_or_insert_with(HashMap::new).entry(res).or_default()[what] += 1;
    }
}

/// resources the title presented
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Whether the title has presented a frame: the preload's memory floor applies from then on
/// (pso_sched::preload_floor), so the compiles made ahead of the title's first frame are not
/// held back by the RAM the title will need later.
pub fn title_presents() -> bool {
    PRESENTX.load(Ordering::Relaxed)
}

pub fn is_presented(res: usize) -> bool {
    PRESENTED.lock().unwrap().contains(&res)
}

fn writer_frames() -> &'static [u64] {
    static FRAMES: OnceLock<Vec<u64>> = OnceLock::new();
    FRAMES.get_or_init(|| std::env::var("XCL_PRESENT_WRITERS_AT").unwrap_or_default()
        .split(',').filter_map(|v| v.trim().parse().ok()).take(16).collect())
}

pub(super) fn writer_trace_configured() -> bool {
    !writer_frames().is_empty() || plane_schedule().is_some()
}

pub(super) fn draw_gate_capture(lead: u64) -> Option<(u64, u64)> {
    let slot = plane_schedule()?.due((crate::log::uptime() as u64).saturating_add(lead), NEXT_PLANE_CAPTURE.load(Ordering::Relaxed))?;
    Some((BOUNDARIES.load(Ordering::Relaxed), slot))
}

pub(super) fn writer_trace_active() -> bool {
    let frame = PRESENTS.load(Ordering::Relaxed);
    writer_frames().iter().any(|&target| (target.saturating_sub(3)..=target).contains(&frame))
        || plane_schedule().is_some_and(|schedule| {
            let next = NEXT_PLANE_CAPTURE.load(Ordering::Relaxed);
            schedule.due(crate::log::uptime() as u64 + 12, next).is_some()
        })
}

#[derive(Clone)]
struct PlaneWriter {
    resource: usize,
    kind: &'static str,
    shaders: Option<(u64, u64)>,
    pso: usize,
    inputs: Vec<String>,
    read_resources: Vec<usize>,
    candidate_args: Option<super::bind::LineageArgs>,
}

type SubmittedPlaneWriter = (u64, usize, Option<usize>, PlaneWriter);
static PLANE_WRITES: Mutex<Option<HashMap<usize, Vec<PlaneWriter>>>> = Mutex::new(None);
static PLANE_SUBMITS: Mutex<std::collections::VecDeque<SubmittedPlaneWriter>> = Mutex::new(std::collections::VecDeque::new());
static PLANE_INPUTS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(super) fn native_view_trace_configured() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("XCL_PRESENT_UPSTREAM_DESCS").is_some() || std::env::var_os("XCL_GPU_IMAGE_PROBES").is_some() || super::target_dump::configured())
}

pub(super) fn lineage_depth() -> usize {
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| std::env::var("XCL_PRESENT_UPSTREAM_DEPTH").ok()
        .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(2).clamp(1, 8))
}

pub(super) fn watch_plane_input(res: usize) {
    let mut resources = PLANE_INPUTS.lock().unwrap();
    if !resources.contains(&res) {
        if resources.len() == 32 {
            resources.remove(0);
        }
        resources.push(res);
    }
}

pub(super) fn is_known_writer_target(res: usize) -> bool {
    res != 0 && (is_presented(res) || PLANE_INPUTS.lock().unwrap().contains(&res))
}

pub(super) fn is_writer_target(res: usize) -> bool {
    res != 0 && (lineage_depth() > 2 || is_known_writer_target(res))
}

pub(super) fn record_plane_write(list: usize, resource: usize, kind: &'static str, pso: usize, inputs: Vec<String>) {
    record_plane_write_with_reads(list, resource, kind, pso, inputs, Vec::new(), None);
}

pub(super) fn record_plane_write_with_reads(list: usize, resource: usize, kind: &'static str, pso: usize, inputs: Vec<String>, read_resources: Vec<usize>, candidate_args: Option<super::bind::LineageArgs>) {
    if !writer_trace_active() || !is_writer_target(resource) {
        return;
    }
    let shaders = if pso == 0 { None } else { super::gpu_timing_stats::shaders(pso) };
    let mut records = PLANE_WRITES.lock().unwrap();
    let lists = records.get_or_insert_with(HashMap::new);
    if lists.len() >= 1024 && !lists.contains_key(&list) {
        return;
    }
    let writes = lists.entry(list).or_default();
    if lineage_depth() > 2 {
        if let Some(prev) = writes.last_mut().filter(|w| w.resource == resource && w.kind == kind && w.pso == pso) {
            *prev = PlaneWriter { resource, kind, shaders, pso, inputs, read_resources, candidate_args };
            return;
        }
    }
    if writes.len() == if lineage_depth() > 2 { 256 } else { 32 } {
        writes.remove(0);
    }
    writes.push(PlaneWriter { resource, kind, shaders, pso, inputs, read_resources, candidate_args });
}

#[derive(Clone, Copy)]
pub(super) struct ScalarProbe {
    pub selector: (usize, u32, u32),
    pub words: u32,
    pub va: Option<u64>,
}

type ScalarBindings = (u64, Vec<ScalarProbe>);
type ScalarSubmission = (u64, usize, ScalarBindings);
static SCALAR_LISTS: OnceLock<Mutex<HashMap<(usize, u64), Vec<ScalarProbe>>>> = OnceLock::new();
static SCALAR_SUBMITS: Mutex<Vec<ScalarSubmission>> = Mutex::new(Vec::new());

pub(super) fn record_scalar_probes(list: usize, shader: u64, values: Vec<ScalarProbe>) {
    let mut lists = SCALAR_LISTS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if lists.len() < 128 || lists.contains_key(&(list, shader)) {
        lists.insert((list, shader), values);
    }
}

pub(super) fn reset_plane_writes(list: usize) {
    super::target_dump::reset(list);
    super::submit_watch::reset(list);
    if let Some(probes) = SCALAR_LISTS.get() {
        probes.lock().unwrap().retain(|&(l, _), _| l != list);
    }
    if writer_trace_configured() {
        if let Some(lists) = PLANE_WRITES.lock().unwrap().as_mut() {
            lists.remove(&list);
        }
    }
}

pub(super) unsafe fn submit_plane_writes(queue: *mut c_void, lists: &[*mut c_void]) {
    super::target_dump::submitted(queue as usize, lists);
    super::submit_watch::submitted(queue, lists);
    if !writer_trace_active() {
        return;
    }
    let queue_id = super::list_x::queue_id(queue as usize);
    let timing_id = super::gpu_timing::queue_stamp(queue).map(|(_, id)| id);
    let frame = PRESENTS.load(Ordering::Relaxed);
    if let Some(recorded) = SCALAR_LISTS.get() {
        let recorded = recorded.lock().unwrap();
        let mut submitted = SCALAR_SUBMITS.lock().unwrap();
        for &list in lists {
            for (&(_, shader), values) in recorded.iter().filter(|&(&(l, _), _)| l == list as usize) {
                let item = (frame, queue as usize, (shader, values.clone()));
                if let Some(prev) = submitted.iter_mut().find(|(_, q, (h, _))| *q == queue as usize && *h == shader) {
                    *prev = item;
                } else if submitted.len() < 16 {
                    submitted.push(item);
                }
            }
        }
    }
    let recorded = PLANE_WRITES.lock().unwrap();
    let Some(recorded) = recorded.as_ref() else { return };
    let mut submits = PLANE_SUBMITS.lock().unwrap();
    for write in lists.iter().filter_map(|list| recorded.get(&(*list as usize))).flatten() {
        if submits.len() == if lineage_depth() > 2 { 32768 } else { 2048 } {
            submits.pop_front();
        }
        submits.push_back((frame, queue_id, timing_id, write.clone()));
    }
}

unsafe fn log_plane_writers(queue: *mut c_void, shown: &[(usize, Option<[i32; 4]>)], timed: bool) {
    let frame = PRESENTS.load(Ordering::Relaxed);
    if !timed && !writer_frames().contains(&frame) {
        return;
    }
    let present_q = super::list_x::queue_id(queue as usize);
    let timing_q = super::gpu_timing::queue_stamp(queue).map(|(_, id)| id);
    let submits = PLANE_SUBMITS.lock().unwrap().clone();
    let mut budget = 128;
    let mut visited = std::collections::HashSet::new();
    for (i, &(resource, _)) in shown.iter().enumerate() {
        crate::xlog_info!("present writer: frame {frame} p{i} resource {resource:#x}, present q{present_q}, timing q{timing_q:?}");
        let mut queues = Vec::new();
        for (at, (recorded_frame, q, timing, w)) in submits.iter().enumerate().rev().filter(|(_, (seen, _, _, w))| *seen >= frame.saturating_sub(3) && w.resource == resource) {
            if queues.contains(&(q, w.kind)) || queues.len() >= 8 {
                continue;
            }
            queues.push((q, w.kind));
            crate::xlog_info!("present writer: p{i}, submitted at frame {recorded_frame}, q{q} timing q{timing:?}: {} PSO {:#x} shader hashes {:x?}, candidates [{}]", w.kind, w.pso, w.shaders, w.inputs.join("; "));
            log_input_producers(&submits, i, frame, at, w, 1, &mut budget, &mut visited);
        }
        if queues.is_empty() {
            crate::xlog_info!("present writer: p{i}, no matching submitted writer in the capture window");
        }
    }
    if budget == 0 {
        crate::xlog_info!("present upstream inputs: frame {frame} reached the lineage row budget");
    }
}

fn log_input_producers(submits: &std::collections::VecDeque<SubmittedPlaneWriter>, plane: usize, frame: u64, at: usize, writer: &PlaneWriter, depth: usize, budget: &mut usize, visited: &mut std::collections::HashSet<usize>) {
    if depth > lineage_depth() || !visited.insert(at) {
        return;
    }
    let mut candidates = if writer.read_resources.is_empty() && lineage_depth() > 2 {
        writer.candidate_args.as_ref().map_or_else(Vec::new, super::bind::lineage_candidates)
    } else {
        writer.read_resources.iter().map(|&r| (r, String::new())).collect()
    };
    if !candidates.is_empty() && candidates[0].1.len() != 0 {
        candidates.sort_by_key(|(r, _)| !submits.range(..at).rev().any(|(seen, _, _, w)| *seen >= frame.saturating_sub(4) && w.resource == *r));
        candidates.truncate(8);
    }
    let mut next = Vec::new();
    for (input, description) in candidates {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let mut logged = [false; 2];
        let written = submits.iter().filter(|(seen, _, _, w)| *seen == frame && w.resource == input)
            .fold(0, |seen, (_, _, _, w)| seen | if w.kind == "compute binding candidate" { 1 } else { 2 });
        let observed = match written { 0 => "not-observed", 1 => "compute-binding-only", _ => "draw/copy/clear" };
        crate::xlog_info!("present input state: p{plane} depth {depth} input {input:#x} this-frame-submitted={observed} {description}");
        for (producer_at, (input_frame, q, timing, producer)) in submits.range(..at).enumerate().rev()
            .filter(|(_, (seen, _, _, producer))| *seen >= frame.saturating_sub(4) && producer.resource == input)
        {
            let kind = usize::from(producer.kind == "compute binding candidate");
            if !std::mem::replace(&mut logged[kind], true) {
                crate::xlog_info!("present input producer: p{plane} depth {depth} input {input:#x}, submitted at frame {input_frame}, q{q} timing q{timing:?}: {} PSO {:#x} shader hashes {:x?}", producer.kind, producer.pso, producer.shaders);
                if producer.inputs.first().is_some_and(|input| input.starts_with("selected ")) {
                    crate::xlog_info!("present upstream inputs: p{plane} output {input:#x} [{}]", producer.inputs.join("; "));
                }
                if kind == 0 && producer.pso != 0 {
                    next.push((producer_at, producer));
                }
            }
            if logged == [true; 2] {
                break;
            }
        }
        if logged == [false; 2] {
            crate::xlog_info!("present input producer: p{plane} depth {depth} input {input:#x}, no prior matching CPU submission in the trace");
        }
    }
    for (producer_at, producer) in next {
        log_input_producers(submits, plane, frame, producer_at, producer, depth + 1, budget, visited);
    }
}

/// the resource an RTV handle names (render-target binds)
pub fn rtv_resource(handle: usize) -> Option<usize> {
    RTVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

/// The resource bound as render target 0 on `list` (`XCL_TRACE_FRAME`).
pub fn rt0_resource(list: usize) -> Option<usize> {
    let h = *LISTS.lock().unwrap().as_ref()?.get(&list)?;
    rtv_resource(h)
}

pub fn note_rt0(list: usize, handle: usize) {
    if !enabled() {
        return;
    }
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(list, handle);
    watch_list(list, handle);
}

/// `XCL_TRACE_LIST=<presents>`: past that many presents, every console method call on a list
/// that binds a presented resource – or one that could be presented (`backbuffer_like`) – as
/// render target 0 is logged until the list closes (the first 4 such lists, 600 calls each;
/// `guard::make_list_tracer`)
pub fn list_trace_after() -> Option<u64> {
    static V: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("XCL_TRACE_LIST").ok().map(|v| v.parse().unwrap_or(300)))
}

/// lists being traced, with the calls logged on each
static WATCHED: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());
static WATCHES: AtomicU64 = AtomicU64::new(0);

fn watch_list(list: usize, handle: usize) {
    let Some(after) = list_trace_after() else { return };
    let Some(res) = rtv_resource(handle).filter(|&r| is_presented(r) || unsafe { backbuffer_like(r) }.is_some()) else { return };
    if activity().0 < after {
        return;
    }
    let mut w = WATCHED.lock().unwrap();
    if w.iter().any(|e| e.0 == list) || WATCHES.fetch_add(1, Ordering::Relaxed) >= 4 {
        return;
    }
    crate::xlog_info!("list trace: list {list:#x} binds presented {res:#x} as render target 0 (handle {handle:#x})");
    w.push((list, 0));
}

/// Whether a call of `slot` on `list` is logged (a close ends the trace of that list).
pub fn list_watched(list: usize, slot: usize) -> bool {
    let mut w = WATCHED.lock().unwrap();
    let Some(i) = w.iter().position(|e| e.0 == list) else { return false };
    w[i].1 += 1;
    if slot == 9 || w[i].1 >= 600 {
        w.swap_remove(i);
    }
    true
}

/// Format the swap chain uses for a back buffer of `f` (None = not presentable).
pub fn swap_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

unsafe fn backbuffer_like(res: usize) -> Option<D3D12_RESOURCE_DESC> {
    let raw = res as *mut c_void;
    let r = ID3D12Resource::from_raw_borrowed(&raw)?;
    let d = r.GetDesc();
    let ok = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Width >= 640
        && d.Height >= 360
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && swap_format(d.Format).is_some();
    ok.then_some(d)
}

/// queues the title submitted work to (AddRef'd: they stay valid for `drain`)
static QUEUES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// The queues the title submitted work to.
pub fn known_queues() -> Vec<usize> {
    QUEUES.lock().unwrap().clone()
}

static EXECUTES: AtomicU64 = AtomicU64::new(0);

/// Submitted composites and ExecuteCommandLists calls so far.
pub fn activity() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), EXECUTES.load(Ordering::Relaxed))
}

pub unsafe fn note_execute(queue: *mut c_void, lists: &[*mut c_void]) {
    EXECUTES.fetch_add(1, Ordering::Relaxed);
    {
        let mut q = QUEUES.lock().unwrap();
        if !q.contains(&(queue as usize)) {
            add_ref(queue as usize);
            q.push(queue as usize);
        }
    }
    if !enabled() || IN_PRESENT.with(|c| c.get()) || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    let handle = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        lists.iter().rev().find_map(|l| m.get(&(*l as usize)).copied())
    };
    let Some(h) = handle else { return };
    let res = RTVS.lock().unwrap().as_ref().and_then(|m| m.get(&h).copied());
    let Some(res) = res else { return };
    if backbuffer_like(res).is_none() {
        return;
    }
    {
        let mut c = CANDIDATE.lock().unwrap();
        if c.map(|(_, r)| r) != Some(res) {
            add_ref(res);
            if let Some((_, old)) = c.replace((queue as usize, res)) {
                release(old);
            }
        }
    }
    // no console present call seen recently → present on submissions (rate limited)
    if now_ms().saturating_sub(LAST_BOUNDARY_MS.load(Ordering::Relaxed)) > 500 {
        present_now(Duration::from_millis(15), None);
    }
}

/// StubHook for console-only queue methods (PresentX, frame events, …).
pub fn on_queue_console_method(_info: &StubInfo, _caller: usize) {
    if !enabled() || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    BOUNDARIES.fetch_add(1, Ordering::Relaxed);
    LAST_BOUNDARY_MS.store(now_ms(), Ordering::Relaxed);
    super::depth_uav::probe_frame();
    unsafe { present_now(Duration::from_millis(4), None) };
}

// ---------------------------------------------------------------------------------------
// window
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if wp.0 == 0x79 && matches!(msg, WM_KEYDOWN | WM_KEYUP | WM_SYSKEYDOWN | WM_SYSKEYUP)
        && super::live_status::key(msg, lp) {
        return LRESULT(0);
    }
    if let Some(r) = crate::hosting::on_message(hwnd.0, msg, wp.0, lp.0) {
        return LRESULT(r);
    }
    // keyboard and mouse as the gamepad: raw mouse input, the cursor held in the window
    if crate::hle::kbm::on_message(hwnd.0, msg, wp.0, lp.0) {
        return LRESULT(0);
    }
    match msg {
        WM_CLOSE => {
            crate::xlog_info!("XCL window closed – ending the game process");
            crate::control::exit(0, "stopped", "");
            flush_frame_dumps();
            crate::report::write_final(None);
            crate::hle::xgameruntime::gamesave::settle_for_exit();
            crate::log::flush();
            windows::Win32::System::Threading::ExitProcess(0);
        }
        WM_DESTROY => {
            super::live_status::stop();
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_TIMER if wp.0 == COMPILE_TIMER => {
            show_compiling(hwnd);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wp, lp),
    }
}

/// The game's name in the XCL window's title: its display name from MicrosoftGame.config,
/// else its package identity.
fn window_name() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let Some(b) = crate::state::boot() else { return "XCL".into() };
        crate::launcher::display_name(&b.game_root).unwrap_or_else(|| b.title.clone())
    })
    .clone()
}

/// the XCL window's timer for `show_compiling`
const COMPILE_TIMER: usize = 0x5C0;

struct FrameRate {
    sampled_at: Instant,
    presented: u64,
    fps: f64,
}

impl FrameRate {
    fn new(now: Instant, presented: u64) -> Self {
        Self {
            sampled_at: now,
            presented,
            fps: 0.0,
        }
    }

    fn sample(&mut self, now: Instant, presented: u64) -> f64 {
        let elapsed = now.duration_since(self.sampled_at);
        if elapsed >= Duration::from_secs(1) {
            self.fps = presented.saturating_sub(self.presented) as f64 / elapsed.as_secs_f64();
            self.sampled_at = now;
            self.presented = presented;
        }
        self.fps
    }
}

/// While XCL builds shaders (the title draws nothing with those pipelines meanwhile: after an
/// XCL update the screen stays black while thousands recompile), the window's title says how
/// far it got (`progress::Status::title`):
/// "xcl: building shaders 1234/2600 (47%), 3 large in flight | XCL – <game> | N FPS".
/// With `XCL_PROGRESS=0`: "XCL – <game> | N FPS | compiling shaders: N running, M done[, Q
/// pipelines queued][, F failed]", kept 2 s after the last compile (compiles in the title's own
/// calls come one by one). Otherwise only the game name and presented frame rate remain.
unsafe fn show_compiling(hwnd: HWND) {
    static SHOWN: Mutex<Option<String>> = Mutex::new(None);
    static FRAME_RATE: Mutex<Option<FrameRate>> = Mutex::new(None);
    let now = Instant::now();
    let presented = HOST_PRESENTS.load(Ordering::Relaxed);
    let fps = FRAME_RATE
        .lock()
        .unwrap()
        .get_or_insert_with(|| FrameRate::new(now, presented))
        .sample(now, presented);
    let mut title = format!("XCL – {} | {fps:.1} FPS", window_name());
    let advanced = super::live_status::advanced();
    if advanced {
        title.push_str(" | ");
        title.push_str(&super::live_status::title());
    }
    let text = if crate::progress::enabled() {
        match crate::progress::current() {
            Some(s) if advanced => format!("{title} | {}", s.title()),
            Some(s) => format!("{} | {title}", s.title()),
            None => title,
        }
    } else {
        compiling_title(title, advanced)
    };
    let mut shown = SHOWN.lock().unwrap();
    if shown.as_deref() != Some(text.as_str()) {
        let _ = SetWindowTextW(hwnd, &HSTRING::from(text.as_str()));
        *shown = Some(text);
    }
    drop(shown);
    super::live_status::refresh(hwnd, fps);
}

/// `title` with the DXC compiles running and done (`XCL_PROGRESS=0`).
fn compiling_title(title: String, advanced: bool) -> String {
    static LAST_ACTIVE: Mutex<Option<Instant>> = Mutex::new(None);
    let (failed, queued) = if advanced {
        (super::pso_async::failed_count(), crate::progress::PIPELINES.counts().pending)
    } else {
        let (_, failed, queued) = super::pso_async::counts();
        (failed, queued)
    };
    let (running, done) = crate::gpu::recomp::dxc::compile_counts();
    let active = running > 0 || queued > 0;
    let recent = {
        let mut last = LAST_ACTIVE.lock().unwrap();
        if active {
            *last = Some(Instant::now());
        }
        last.is_some_and(|t| t.elapsed() < Duration::from_secs(2))
    };
    if !recent {
        return title;
    }
    let queued = if queued > 0 { format!(", {queued} pipelines queued") } else { String::new() };
    let failed = if failed > 0 { format!(", {failed} failed") } else { String::new() };
    format!("{title} | compiling shaders: {running} running, {done} done{queued}{failed}")
}

fn window_thread() {
    unsafe {
        let Ok(inst) = GetModuleHandleW(PCWSTR::null()) else { return };
        let class = windows::core::w!("XCLGameWindow");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            ..Default::default()
        };
        RegisterClassW(&wc);
        let cfg = crate::state::config();
        let title = window_name();
        let mut rc = RECT { left: 0, top: 0, right: cfg.window_width as i32, bottom: cfg.window_height as i32 };
        let _ = AdjustWindowRect(&mut rc, WS_OVERLAPPEDWINDOW, false);
        let (w, h) = (rc.right - rc.left, rc.bottom - rc.top);
        let (x, y) = window_position(&cfg.window_pos, w, h);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            &HSTRING::from(format!("XCL – {title}")),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            x,
            y,
            w,
            h,
            None,
            None,
            Some(inst.into()),
            None,
        );
        let Ok(hwnd) = hwnd else {
            crate::xlog_error!("could not create the XCL window");
            return;
        };
        HWND_RAW.store(hwnd.0 as isize, Ordering::Release);
        crate::xlog_info!("XCL window created ({}x{})", cfg.window_width, cfg.window_height);
        crate::hosting::adopt(hwnd.0);
        crate::hle::kbm::attach(hwnd.0);
        if super::live_status::advanced() || crate::live_stall::enabled() {
            super::live_status::start();
        }
        super::live_status::install(hwnd, inst.into());
        SetTimer(Some(hwnd), COMPILE_TIMER, 500, None);
        let raw = hwnd.0 as isize;
        let _ = std::thread::Builder::new().name("xcl-window-watch".into()).spawn(move || watch_window(raw));
        if let Err(e) = std::thread::Builder::new().name("xcl-display".into()).spawn(move || display_thread(raw)) {
            crate::xlog_error!("present: cannot start the host display thread: {e}");
        }
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let t0 = Instant::now();
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
            let us = t0.elapsed().as_micros() as u64;
            if us > WORST_HANDLER.fetch_max(us, Ordering::Relaxed) {
                WORST_MESSAGE.store(msg.message, Ordering::Relaxed);
            }
        }
    }
}

/// The longest the window thread spent on one message since `watch_window` last looked, and
/// which message that was.
static WORST_HANDLER: AtomicU64 = AtomicU64::new(0);
static WORST_MESSAGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether the XCL window answers: every second a WM_NULL is sent to it and the round trip
/// timed (Windows calls a window that has not taken a message for 5 s "not responding"). A
/// round trip of a second or more is a warning; with `XCL_ACTIVITY` the worst round trip and
/// the worst message handler are logged every 10 s.
fn watch_window(hwnd: isize) {
    let activity = std::env::var_os("XCL_ACTIVITY").is_some();
    let mut worst_trip = Duration::ZERO;
    for tick in 1u64.. {
        std::thread::sleep(Duration::from_secs(1));
        let t0 = Instant::now();
        let mut result = 0usize;
        let answered = unsafe { SendMessageTimeoutW(HWND(hwnd as *mut c_void), WM_NULL, WPARAM(0), LPARAM(0), SMTO_NORMAL, 10_000, Some(&mut result as *mut usize)) }.0 != 0;
        let trip = t0.elapsed();
        worst_trip = worst_trip.max(trip);
        if !answered || trip >= Duration::from_secs(1) {
            crate::xlog_warn!("window: took {} ms to answer a message{}", trip.as_millis(), if answered { "" } else { " (no answer in 10 s)" });
        }
        if activity && tick % 10 == 0 {
            crate::xlog_info!(
                "window (10 s): worst answer {} ms, worst message handler {} ms (message {:#x})",
                worst_trip.as_millis(),
                WORST_HANDLER.swap(0, Ordering::Relaxed) / 1000,
                WORST_MESSAGE.load(Ordering::Relaxed)
            );
            worst_trip = Duration::ZERO;
        }
    }
}

/// The XCL window's top-left from `window_pos` (`XCL_WINDOW_POS` overrides): "auto" keeps
/// CW_USEDEFAULT (Windows may pick any monitor), "center" centers on the primary monitor's
/// work area, "x,y" is clamped into the nearest monitor's work area — a multi-monitor desktop
/// otherwise gets the window on whichever monitor Windows' cascade picks.
unsafe fn window_position(setting: &str, w: i32, h: i32) -> (i32, i32) {
    let env = std::env::var("XCL_WINDOW_POS").ok();
    let mode = env.as_deref().unwrap_or(setting).trim();
    if mode.is_empty() || mode.eq_ignore_ascii_case("auto") {
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    }
    let work = |pt: POINT, flags: u32| -> Option<RECT> {
        let mon = MonitorFromPoint(pt, MONITOR_FROM_FLAGS(flags));
        if mon.is_invalid() {
            return None;
        }
        let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
        GetMonitorInfoW(mon, &mut mi).as_bool().then_some(mi.rcWork)
    };
    if mode.eq_ignore_ascii_case("center") {
        let Some(r) = work(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY.0) else {
            return (CW_USEDEFAULT, CW_USEDEFAULT);
        };
        return (r.left + ((r.right - r.left) - w).max(0) / 2, r.top + ((r.bottom - r.top) - h).max(0) / 2);
    }
    let Some((xs, ys)) = mode.split_once(',') else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    let (Ok(x), Ok(y)) = (xs.trim().parse::<i32>(), ys.trim().parse::<i32>()) else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    // clamp into the work area of the monitor nearest the requested top-left
    let Some(r) = work(POINT { x, y }, MONITOR_DEFAULTTONEAREST.0) else {
        return (x, y);
    };
    (x.clamp(r.left, (r.right - w).max(r.left)), y.clamp(r.top, (r.bottom - h).max(r.top)))
}

fn ensure_window() -> Option<HWND> {
    static START: Once = Once::new();
    START.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-window".into()).spawn(window_thread);
    });
    let t0 = Instant::now();
    while HWND_RAW.load(Ordering::Acquire) == 0 {
        if t0.elapsed() > Duration::from_secs(3) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Some(HWND(HWND_RAW.load(Ordering::Acquire) as *mut c_void))
}

// ---------------------------------------------------------------------------------------
// presenter
// ---------------------------------------------------------------------------------------

struct Presenter {
    queue: usize,
    dev: ID3D12Device,
    surfaces: Vec<Arc<ID3D12Resource>>,
    w: u32,
    h: u32,
    fmt: DXGI_FORMAT,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    copy_sources: Vec<ID3D12Resource>,
    dumps: Vec<FrameDump>,
    dump_frame: Option<u64>,
    scalar_dump: Vec<(u64, usize, u64, ScalarProbe, u64)>,
    disabled: bool,
    value: u64,
    event: HANDLE,
    last: Instant,
    composer: Option<Composer>,
    /// the back buffer's view for the progress bar (`progress_bar`)
    bar_rtv: Option<ID3D12DescriptorHeap>,
}

/// What composes a PresentX of several planes into the swap chain's buffer (`compose`), for
/// one back-buffer format.
struct Composer {
    fmt: DXGI_FORMAT,
    rs: ID3D12RootSignature,
    /// the bottom plane, as it is
    under: ID3D12PipelineState,
    /// a plane over it, blended with its premultiplied alpha
    over: ID3D12PipelineState,
    srvs: ID3D12DescriptorHeap,
    rtv: ID3D12DescriptorHeap,
}

/// planes composed at most (the console's display has four)
const COMPOSED_PLANES: usize = 4;

/// The composer's shaders: a triangle over the whole output that samples the plane's source
/// UVs (`uv_rect`, root constants b0; ivall/xcl 2ca749ab).
const COMPOSER_VS: &str = "cbuffer R : register(b0) { float4 uv_rect; };\nstruct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };\nO main(uint id : SV_VertexID) { O o; float2 t = float2((id << 1) & 2, id & 2); o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = lerp(uv_rect.xy, uv_rect.zw, t); return o; }\n";
const COMPOSER_PS: &str = "Texture2D t : register(t0);\nSamplerState s : register(s0);\nfloat4 main(float4 p : SV_Position, float2 uv : TEXCOORD0) : SV_Target { return t.SampleLevel(s, uv, 0); }\n";

unsafe fn new_composer(dev: &ID3D12Device, fmt: DXGI_FORMAT) -> Option<Composer> {
    let (vs, ps) = (super::real::compile(COMPOSER_VS, "vs_5_0"), super::real::compile(COMPOSER_PS, "ps_5_0"));
    if vs.is_empty() || ps.is_empty() {
        crate::xlog_error!("present: the plane composer's shaders did not compile");
        return None;
    }
    let range = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 1, ..Default::default() };
    // the plane's source UVs (4 constants, b0) and its view (t0)
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxLOD: f32::MAX,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), NumStaticSamplers: 1, pStaticSamplers: &sampler, Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE };
    let mut blob = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).ok()?;
    let blob = blob?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).ok()?;
    let pso = |blend: bool| -> Option<ID3D12PipelineState> {
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&rs),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        d.RasterizerState.DepthClipEnable = true.into();
        d.RTVFormats[0] = fmt;
        d.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: blend.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        dev.CreateGraphicsPipelineState(&d).map_err(|e| crate::xlog_error!("present: plane composer pipeline: {e}")).ok()
    };
    let (under, over) = (pso(false)?, pso(true)?);
    let srvs = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: COMPOSED_PLANES as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .ok()?;
    let rtv = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).ok()?;
    Some(Composer { fmt, rs, under, over, srvs, rtv })
}

/// The console's display composes a PresentX's planes: the first (the game's) as it is, each
/// one after it (a UI plane, often at another size) over it with its alpha, each plane's source
/// rect (its source UVs, `shown_uv`) stretched over the whole output. Recorded on the
/// presenter's list; the back buffer ends in COPY_SOURCE. False: no composer, or a plane without
/// a shader view (multisampled, not 2D, or denying shader resources; ivall/xcl #2418 3f4a1291) –
/// the caller copies the bottom plane whole as before.
/// shortcut: the destination placement (@0x20) is not read (upstream does not read it either).
unsafe fn compose(p: &mut Presenter, bb: &ID3D12Resource, planes: &[(ID3D12Resource, [f32; 4])]) -> bool {
    let planes = &planes[..planes.len().min(COMPOSED_PLANES)];
    let viewable = |d: D3D12_RESOURCE_DESC| {
        d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count == 1 && !d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    };
    if source_rects_enabled() && !planes.iter().all(|(r, _)| viewable(r.GetDesc())) {
        return false;
    }
    if p.composer.as_ref().map_or(true, |c| c.fmt != p.fmt) {
        p.composer = new_composer(&p.dev, p.fmt);
    }
    let Some(c) = p.composer.as_ref() else { return false };
    let step = p.dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    let (cpu, gpu) = (c.srvs.GetCPUDescriptorHandleForHeapStart(), c.srvs.GetGPUDescriptorHandleForHeapStart());
    for (k, (r, _)) in planes.iter().enumerate() {
        let d = r.GetDesc();
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: swap_format(d.Format).unwrap_or(d.Format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
        };
        p.dev.CreateShaderResourceView(r, Some(&view), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + k * step });
    }
    let rtv = c.rtv.GetCPUDescriptorHandleForHeapStart();
    p.dev.CreateRenderTargetView(bb, None, rtv);
    // (the RTV hook keeps a reference to what a handle names: none on the swap chain's buffer)
    note_rtv(rtv.ptr, std::ptr::null_mut());
    p.list.ResourceBarrier(&[transition(bb, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
    p.list.OMSetRenderTargets(1, Some(&rtv), false, None);
    p.list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: p.w as f32, Height: p.h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
    p.list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: p.w as i32, bottom: p.h as i32 }]);
    p.list.SetGraphicsRootSignature(&c.rs);
    p.list.SetDescriptorHeaps(&[Some(c.srvs.clone())]);
    p.list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    for (k, (_, uv)) in planes.iter().enumerate() {
        p.list.SetPipelineState(if k == 0 { &c.under } else { &c.over });
        p.list.SetGraphicsRoot32BitConstants(0, 4, uv.as_ptr() as *const c_void, 0);
        p.list.SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + (k * step) as u64 });
        p.list.DrawInstanced(3, 1, 0, 0);
    }
    p.list.ResourceBarrier(&[transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    true
}

struct FrameDump {
    source: Option<ID3D12Resource>,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    format: DXGI_FORMAT,
    tag: String,
}

#[derive(Debug, PartialEq, Eq)]
enum DumpAction {
    NoCapture,
    Keep,
    Collect(u64),
}

fn dump_action(frame: Option<u64>, completed: bool) -> DumpAction {
    match frame {
        None => DumpAction::NoCapture,
        Some(frame) if completed => DumpAction::Collect(frame),
        Some(_) => DumpAction::Keep,
    }
}

type DumpBatch = (u64, Vec<FrameDump>);

enum DumpJob {
    Write(DumpBatch),
    Flush(Option<DumpBatch>, SyncSender<()>),
}

static DUMP_WRITER: OnceLock<Option<SyncSender<DumpJob>>> = OnceLock::new();
static DUMP_SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const SCALAR_READBACK_BYTES: u64 = 16 * 8 * 8 * 4;

fn scalar_readback() -> Option<(usize, u64, usize)> {
    static BUFFER: OnceLock<Option<(usize, u64, usize)>> = OnceLock::new();
    *BUFFER.get_or_init(|| unsafe { super::markers::readback_buffer(SCALAR_READBACK_BYTES) })
}

unsafe fn copy_scalar_probes(p: &Presenter, frame: u64) -> Vec<(u64, usize, u64, ScalarProbe, u64)> {
    let bindings: Vec<_> = SCALAR_SUBMITS.lock().unwrap().iter()
        .filter(|(f, q, _)| *q == p.queue && (frame.saturating_sub(2)..=frame).contains(f)).cloned().collect();
    if bindings.is_empty() {
        return Vec::new();
    }
    let Some((rb, _, _)) = scalar_readback() else { return Vec::new() };
    let rb = rb as *mut c_void;
    let Some(dst) = ID3D12Resource::from_raw_borrowed(&rb) else { return Vec::new() };
    let mut at = 0;
    let mut samples = Vec::new();
    for (recorded, queue, (shader, values)) in bindings {
        let q = super::list_x::queue_id(queue);
        for value in values {
            let (reg, desc, offset) = value.selector;
            let bytes = (value.words * 4) as u64;
            let source = value.va.and_then(|va| super::real::uma_window_buffer(va as usize, bytes as usize));
            let Some((src, off)) = source else {
                crate::xlog_info!("present scalar: frame {frame} q{q} submission frame {recorded} shader {shader:016x} s{reg}+{desc:#x}:{offset:#x}: no UMA source");
                continue;
            };
            let src = src as *mut c_void;
            let Some(src) = ID3D12Resource::from_raw_borrowed(&src) else { continue };
            p.list.ResourceBarrier(&[transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            p.list.CopyBufferRegion(dst, at, src, off, bytes);
            p.list.ResourceBarrier(&[transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
            samples.push((frame, q, shader, value, at));
            at += bytes;
        }
    }
    samples
}

fn take_frame_dumps(p: &mut Presenter, completed: bool) -> Option<DumpBatch> {
    if completed && !p.scalar_dump.is_empty() {
        if let Some((_, _, cpu)) = scalar_readback() {
            for (frame, q, shader, value, at) in p.scalar_dump.drain(..) {
                let (reg, desc, offset) = value.selector;
                let words: Vec<_> = (0..value.words as usize).map(|i| unsafe {
                    ((cpu + at as usize + i * 4) as *const u32).read_volatile()
                }).collect();
                let floats: Vec<_> = words.iter().copied().map(f32::from_bits).collect();
                crate::xlog_info!("present scalar: frame {frame} q{q} shader {shader:016x} s{reg}+{desc:#x}:{offset:#x} UMA {:x?} GPU bits {words:08x?} floats {floats:?}", value.va);
            }
        }
    }
    let DumpAction::Collect(frame) = dump_action(p.dump_frame, completed) else { return None };
    let mut dumps = std::mem::take(&mut p.dumps);
    p.dump_frame = None;
    for dump in &mut dumps {
        dump.source = None;
    }
    Some((frame, dumps))
}

fn frame_dump_writer() -> Option<&'static SyncSender<DumpJob>> {
    DUMP_WRITER.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel(1);
        match std::thread::Builder::new().name("xcl-frame-dump".into()).spawn(move || {
            while let Ok(job) = rx.recv() {
                let (batch, done) = match job {
                    DumpJob::Write(batch) => (Some(batch), None),
                    DumpJob::Flush(batch, done) => (batch, Some(done)),
                };
                if let Some((frame, dumps)) = batch {
                    for dump in dumps {
                        unsafe { write_frame(&dump.buffer, &dump.footprint, dump.format, &dump.tag, frame) };
                    }
                }
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
        }) {
            Ok(_) => Some(tx),
            Err(e) => {
                crate::xlog_error!("present: cannot start frame dump writer: {e}");
                None
            }
        }
    }).as_ref()
}

fn queue_frame_dumps(p: &mut Presenter, completed: bool) -> bool {
    let Some(batch) = take_frame_dumps(p, completed) else { return true };
    let Some(writer) = frame_dump_writer() else { return true };
    match writer.try_send(DumpJob::Write(batch)) {
        Ok(()) => true,
        Err(TrySendError::Full(DumpJob::Write((frame, dumps)))) => {
            p.dump_frame = Some(frame);
            p.dumps = dumps;
            false
        }
        Err(_) => {
            crate::xlog_warn!("present: frame dump writer disconnected");
            true
        }
    }
}

unsafe fn frame_dump_ready(p: &Presenter, wait_ms: u32) -> Result<bool, FenceError> {
    if wait_ms == 0 {
        fence::completed_value(&p.fence, &p.dev).map(|completed| completed >= p.value)
    } else {
        fence::wait(&p.fence, p.value, &p.dev, Some(p.event), wait_ms).map(|()| true)
    }
}

pub(crate) fn flush_frame_dumps() {
    if !DUMP_SHUTDOWN.swap(true, Ordering::AcqRel) {
        log_host_display();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let remaining = || deadline.saturating_duration_since(Instant::now());
    let mut guard = loop {
        match PRESENTER.try_lock() {
            Ok(guard) => break guard,
            Err(TryLockError::WouldBlock) if !remaining().is_zero() => {
                std::thread::sleep(Duration::from_millis(10).min(remaining()));
            }
            Err(_) => {
                crate::xlog_warn!("present: shutdown could not lock the presenter to collect its last dump");
                return;
            }
        }
    };
    let batch = guard.as_mut().filter(|p| p.dump_frame.is_some() || !p.scalar_dump.is_empty()).and_then(|p| {
        match unsafe { frame_dump_ready(p, remaining().as_millis() as u32) } {
            Ok(ready) => take_frame_dumps(p, ready),
            Err(e) => {
                crate::xlog_warn!("present: shutdown frame dump copy: {e}");
                None
            }
        }
    });
    let writer = if batch.is_some() { frame_dump_writer() } else { DUMP_WRITER.get().and_then(Option::as_ref) };
    let Some(writer) = writer else { return };
    let (done, finished) = mpsc::sync_channel(1);
    let mut job = DumpJob::Flush(batch, done);
    loop {
        match writer.try_send(job) {
            Ok(()) => {
                if finished.recv_timeout(remaining()).is_err() {
                    crate::xlog_warn!("present: shutdown timed out waiting for the frame dump writer");
                }
                return;
            }
            Err(TrySendError::Full(pending)) if !remaining().is_zero() => {
                job = pending;
                std::thread::sleep(Duration::from_millis(10).min(remaining()));
            }
            Err(_) => {
                crate::xlog_warn!("present: shutdown could not flush the frame dump writer");
                return;
            }
        }
    }
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

unsafe fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

unsafe fn new_presenter(queue: usize) -> Option<Presenter> {
    let dev_raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&dev_raw)?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    list.Close().ok()?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let event = CreateEventW(None, false, false, None).ok()?;
    Some(Presenter {
        queue, dev: dev.clone(), surfaces: Vec::new(), w: 0, h: 0, fmt: DXGI_FORMAT_UNKNOWN, alloc, list, fence,
        copy_sources: Vec::new(), dumps: Vec::new(), dump_frame: None, scalar_dump: Vec::new(), disabled: false, value: 0, event,
        last: Instant::now() - Duration::from_secs(1), composer: None, bar_rtv: None,
    })
}

#[derive(Clone)]
struct DisplayFrame {
    index: u64,
    surface: Arc<ID3D12Resource>,
}

struct DisplayFrames {
    pending: Option<(DisplayFrame, ID3D12Fence, u64)>,
    retired: Option<DisplayFrame>,
}

static DISPLAY_FRAMES: Mutex<DisplayFrames> = Mutex::new(DisplayFrames { pending: None, retired: None });

impl DisplayFrames {
    unsafe fn retire(&mut self, dev: &ID3D12Device) -> Result<(), FenceError> {
        if let Some((_, fence, value)) = &self.pending {
            if fence::completed_value(fence, dev)? >= *value {
                self.retired = self.pending.take().map(|(frame, _, _)| frame);
            }
        }
        Ok(())
    }
}

fn log_host_display() {
    crate::xlog_info!(
        "host display: {} valid PresentX calls, {} composites submitted, {} DXGI S_OK, {} occluded, {} still drawing",
        TITLE_PRESENT_CALLS.load(Ordering::Relaxed),
        PRESENTS.load(Ordering::Relaxed),
        HOST_PRESENTS.load(Ordering::Relaxed),
        HOST_OCCLUDED.load(Ordering::Relaxed),
        HOST_STILL_DRAWING.load(Ordering::Relaxed),
    );
}

unsafe fn composite_surface(p: &mut Presenter, w: u32, h: u32, fmt: DXGI_FORMAT) -> Option<Arc<ID3D12Resource>> {
    if (p.w, p.h, p.fmt) != (w, h, fmt) {
        p.surfaces.clear();
        (p.w, p.h, p.fmt) = (w, h, fmt);
    }
    if let Some(surface) = p.surfaces.iter().find(|surface| Arc::strong_count(surface) == 1) {
        return Some(surface.clone());
    }
    if p.surfaces.len() == 4 {
        return None;
    }
    let mut surface: Option<ID3D12Resource> = None;
    p.dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(w),
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COMMON,
        None,
        &mut surface,
    ).ok()?;
    let surface = Arc::new(surface?);
    p.surfaces.push(surface.clone());
    Some(surface)
}

struct HostDisplay {
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
    in_flight: Option<DisplayFrame>,
    shown: Option<u64>,
    sc: Option<IDXGISwapChain3>,
    w: u32,
    h: u32,
    fmt: DXGI_FORMAT,
}

impl HostDisplay {
    unsafe fn new(dev: &ID3D12Device) -> windows::core::Result<Self> {
        let desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
        type CreateQueue = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const windows::core::GUID, *mut *mut c_void) -> windows::core::HRESULT;
        let slot = std::mem::offset_of!(ID3D12Device_Vtbl, CreateCommandQueue) / std::mem::size_of::<usize>();
        let create: CreateQueue = std::mem::transmute(super::real::original_or_native(dev.as_raw(), slot).0);
        let mut queue = std::ptr::null_mut();
        create(dev.as_raw(), &desc, &ID3D12CommandQueue::IID, &mut queue).ok()?;
        let queue = ID3D12CommandQueue::from_raw(queue);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        Ok(Self {
            queue, alloc, list, fence: dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?, value: 0,
            in_flight: None, shown: None, sc: None, w: 0, h: 0, fmt: DXGI_FORMAT_UNKNOWN,
        })
    }

    unsafe fn flip(&mut self, dev: &ID3D12Device, hwnd: HWND, frame: DisplayFrame) -> Result<(), FenceError> {
        let desc = frame.surface.GetDesc();
        let err = |op, e| fence::api_error(dev, op, e);
        let sc = self.swapchain(hwnd, desc.Width as u32, desc.Height, desc.Format)
            .map_err(|e| err("host swap chain", e))?;
        let bb: ID3D12Resource = sc.GetBuffer(sc.GetCurrentBackBufferIndex()).map_err(|e| err("GetBuffer", e))?;
        self.alloc.Reset().map_err(|e| err("host allocator reset", e))?;
        self.list.Reset(&self.alloc, None).map_err(|e| err("host list reset", e))?;
        self.list.ResourceBarrier(&[
            transition(&frame.surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
        ]);
        self.list.CopyResource(&bb, frame.surface.as_ref());
        self.list.ResourceBarrier(&[
            transition(&frame.surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            transition(&bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT),
        ]);
        self.list.Close().map_err(|e| err("host list close", e))?;
        let list = self.list.cast().map_err(|e| err("host list cast", e))?;
        self.queue.ExecuteCommandLists(&[Some(list)]);
        let index = frame.index;
        self.in_flight = Some(frame);
        crate::gpu::pix_capture::before_present(hwnd);
        let status = sc.Present(1, DXGI_PRESENT_DO_NOT_WAIT);
        self.value += 1;
        self.queue.Signal(&self.fence, self.value).map_err(|e| err("host copy signal", e))?;
        if status == S_OK {
            self.shown = Some(index);
            if HOST_PRESENTS.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_info!("present: first host flip ({}x{}, composite {index})", desc.Width, desc.Height);
            }
        } else if status == DXGI_STATUS_OCCLUDED {
            HOST_OCCLUDED.fetch_add(1, Ordering::Relaxed);
        } else if status == DXGI_ERROR_WAS_STILL_DRAWING {
            HOST_STILL_DRAWING.fetch_add(1, Ordering::Relaxed);
        } else {
            status.ok().map_err(|e| err("host Present", e))?;
        }
        Ok(())
    }

    unsafe fn swapchain(&mut self, hwnd: HWND, w: u32, h: u32, fmt: DXGI_FORMAT) -> windows::core::Result<IDXGISwapChain3> {
        if let Some(sc) = &self.sc {
            if (self.w, self.h, self.fmt) != (w, h, fmt) {
                sc.ResizeBuffers(2, w, h, fmt, DXGI_SWAP_CHAIN_FLAG(0))?;
                (self.w, self.h, self.fmt) = (w, h, fmt);
                crate::xlog_info!("present: swap chain {w}x{h} {:?}", fmt);
            }
            return Ok(sc.clone());
        }
        let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: w,
            Height: h,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            ..Default::default()
        };
        let sc: IDXGISwapChain3 = factory.CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?.cast()?;
        let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        self.sc = Some(sc.clone());
        (self.w, self.h, self.fmt) = (w, h, fmt);
        crate::xlog_info!("present: swap chain {w}x{h} {:?}", fmt);
        Ok(sc)
    }
}

fn display_thread(raw: isize) {
    let hwnd = HWND(raw as *mut c_void);
    let mut display = None::<HostDisplay>;
    let mut last_error = None;
    let mut visibility_lost = false;
    let mut last_stats = Instant::now();
    let activity = std::env::var_os("XCL_ACTIVITY").is_some();
    loop {
        std::thread::sleep(Duration::from_millis(16));
        if activity && last_stats.elapsed() >= Duration::from_secs(10) {
            log_host_display();
            last_stats = Instant::now();
        }
        unsafe {
            if !IsWindow(Some(hwnd)).as_bool() {
                break;
            }
            if IsIconic(hwnd).as_bool() || !IsWindowVisible(hwnd).as_bool() {
                visibility_lost = true;
                continue;
            }
            let raw = super::real::device_raw();
            let Some(dev) = ID3D12Device::from_raw_borrowed(&raw) else { continue };
            if let Some(host) = display.as_mut().filter(|host| host.in_flight.is_some()) {
                match fence::completed_value(&host.fence, dev) {
                    Ok(value) if value >= host.value => host.in_flight = None,
                    Ok(_) => continue,
                    Err(e) => {
                        crate::xlog_error!("present: retire host copy: {e}");
                        break;
                    }
                }
            }
            let frame = {
                let mut frames = DISPLAY_FRAMES.lock().unwrap();
                if let Err(e) = frames.retire(dev) {
                    crate::xlog_error!("present: retire composite: {e}");
                    break;
                }
                frames.retired.clone()
            };
            let Some(frame) = frame else { continue };
            if display.is_none() {
                match HostDisplay::new(dev) {
                    Ok(host) => display = Some(host),
                    Err(e) => {
                        crate::xlog_error!("present: cannot create host display: {e}");
                        break;
                    }
                }
            }
            let host = display.as_mut().unwrap();
            if std::mem::take(&mut visibility_lost) {
                host.shown = None;
            }
            if host.shown.is_some_and(|shown| shown >= frame.index) {
                continue;
            }
            match host.flip(dev, hwnd, frame) {
                Ok(()) => last_error = None,
                Err(e) => {
                    let message = e.to_string();
                    if last_error.as_ref() != Some(&message) {
                        crate::xlog_error!("present: {e}");
                        last_error = Some(message);
                    }
                    if matches!(e, FenceError::Removed(_) | FenceError::Api("host copy signal", _)) {
                        break;
                    }
                }
            }
        }
    }
    if display.as_ref().is_some_and(|host| host.in_flight.is_some()) {
        std::mem::forget(display);
    }
}

unsafe fn present_now(min_interval: Duration, frame: Option<(usize, &[(usize, Option<[i32; 4]>)])>) {
    if IN_PRESENT.with(|c| c.replace(true)) {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_PRESENT.with(|c| c.set(false));
        }
    }
    let _reset = Reset;
    let (queue, src, mut guard) = if let Some((queue, planes)) = frame {
        let raw = planes[0].0 as *mut c_void;
        let Some(src) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        (queue, src.clone(), PRESENTER.lock().unwrap())
    } else {
        let Ok(guard) = PRESENTER.try_lock() else { return };
        if PRESENTX.load(Ordering::Relaxed) {
            return;
        }
        let candidate = CANDIDATE.lock().unwrap();
        let Some((queue, res)) = *candidate else { return };
        let raw = res as *mut c_void;
        let Some(src) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        (queue, src.clone(), guard)
    };
    // PresentX on another queue than the presenter was made for (the fallback presented once from
    // a queue before the title's first PresentX): retire the old queue's composition before
    // switching to the title's present queue.
    if frame.is_some() && guard.as_ref().is_some_and(|p| p.queue != queue) {
        if let Some(old) = guard.as_mut() {
            match fence::wait(&old.fence, old.value, &old.dev, Some(old.event), 1000) {
                Ok(()) => {
                    if !queue_frame_dumps(old, true) {
                        return;
                    }
                }
                Err(FenceError::Removed(_)) => {}
                Err(_) => return,
            }
        }
        crate::xlog_info!("D3D12.X: PresentX from another queue ({queue:#x}) – composition moves to it");
        *guard = None;
    }
    if guard.is_none() {
        *guard = new_presenter(queue);
    }
    let Some(p) = guard.as_mut() else { return };
    if p.disabled || p.last.elapsed() < min_interval {
        return;
    }
    let Some(desc) = backbuffer_like(src.as_raw() as usize) else { return };
    let Some(fmt) = swap_format(desc.Format) else { return };
    if ensure_window().is_none() {
        return;
    }
    if p.queue != queue {
        return;
    }
    // wait for the previous composition before reusing its allocator and descriptors
    if let Err(e) = fence::wait(&p.fence, p.value, &p.dev, Some(p.event), 100) {
        p.disabled = matches!(&e, FenceError::Removed(_));
        if !matches!(&e, FenceError::Timeout(_)) {
            crate::xlog_error!("present: previous copy: {e}");
        }
        return;
    }
    p.copy_sources.clear();
    queue_frame_dumps(p, true);
    let Some(bb) = composite_surface(p, desc.Width as u32, desc.Height, fmt) else { return };
    if p.alloc.Reset().is_err() || p.list.Reset(&p.alloc, None).is_err() {
        return;
    }
    // the planes over the first (a UI plane), and a first plane shown in part (its source rect:
    // a frame drawn into part of a larger target): composed as the console's display does
    let sources = frame.map_or(&[][..], |f| f.1);
    let mut planes = vec![(src.clone(), sources.first().and_then(|s| s.1))];
    planes.extend(sources.iter().skip(1).filter_map(|&(r, rect)| Some((ID3D12Resource::from_raw_borrowed(&(r as *mut c_void))?.clone(), rect))));
    let planes: Vec<(ID3D12Resource, [f32; 4])> = planes
        .into_iter()
        .enumerate()
        .map(|(k, (r, rect))| {
            let d = r.GetDesc();
            let uv = shown_uv(k, rect, u32::try_from(d.Width).unwrap_or(0), d.Height);
            (r, uv)
        })
        .collect();
    let composed = (planes.len() > 1 || planes[0].1 != FULL_UV) && compose(p, &bb, &planes);
    if !composed {
        p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        p.list.CopyResource(bb.as_ref(), &src);
        p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    }
    let index = PRESENTS.load(Ordering::Relaxed);
    let timed_capture = if DUMP_SHUTDOWN.load(Ordering::Acquire) {
        None
    } else {
        plane_schedule().and_then(|schedule| schedule.due(crate::log::uptime() as u64, NEXT_PLANE_CAPTURE.load(Ordering::Relaxed)))
    };
    let trace_planes = timed_capture.is_some() || trace_planes_at(index);
    if let Some((_, shown)) = frame {
        log_plane_writers(queue as *mut c_void, shown, timed_capture.is_some());
    }
    let due = frame_dump_due();
    let mut dumps = Vec::new();
    if due || trace_planes {
        // the composite before the host progress bar
        let (shown, d) = (bb.as_ref().clone(), bb.GetDesc());
        if let Some((buffer, footprint)) = readback_copy(&p.list, &shown, &d) {
            dumps.push(FrameDump { source: Some(shown), buffer, footprint, format: d.Format, tag: "frame".into() });
        }
    }
    if trace_planes {
        let queue = super::list_x::queue_id(queue);
        crate::xlog_info!("present planes: frame {index}, {:.1} s, q{queue}, {} source(s), composed {composed}", crate::log::uptime(), planes.len());
        for (k, (plane, uv)) in planes.iter().take(COMPOSED_PLANES).enumerate() {
            let res = plane.as_raw() as usize;
            let d = plane.GetDesc();
            let title = super::real::title_address(res).or_else(|| super::title_textures::address(res));
            let backing = super::residency::describe_backing(res).unwrap_or_else(|| "not in residency registry".into());
            crate::xlog_info!("present planes: p{k} resource {res:#x}, title {title:x?}, {}x{} format {} flags {:#x}, source uv {uv:?}, {backing}", d.Width, d.Height, d.Format.0, d.Flags.0);
            if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count == 1 && d.DepthOrArraySize == 1 {
                if let Some(format) = swap_format(d.Format).filter(|&f| f != DXGI_FORMAT_R16G16B16A16_FLOAT) {
                    if let Some((buffer, footprint)) = readback_copy(&p.list, plane, &d) {
                        dumps.push(FrameDump { source: Some(plane.clone()), buffer, footprint, format, tag: format!("plane{k}") });
                    }
                }
            }
        }
    }
    let target_dump = if frame.is_some() && super::target_dump::boundary() { super::target_dump::capture(&p.list, &planes.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>()) } else { None };
    // with `XCL_TRACE_FRAME`: also the render target the title binds most (its scene / UI)
    // (and the targets the watched one's contents went on to: rt_<n>_<k>)
    if due && trace_frame() {
        let busiest = busiest_rt();
        let followed = FOLLOWED.lock().unwrap();
        let targets = busiest
            .into_iter()
            .chain(followed.iter().copied())
            .filter_map(|r| {
                let rr = r as *mut c_void;
                let res = ID3D12Resource::from_raw_borrowed(&rr)?.clone();
                let d = res.GetDesc();
                if d.SampleDesc.Count > 1 {
                    return None;
                }
                let (buffer, footprint) = readback_copy(&p.list, &res, &d)?;
                Some((res, buffer, footprint, d.Format))
            });
        for (k, (res, buffer, footprint, format)) in targets.enumerate() {
            let tag = if k == 0 { "rt".into() } else { format!("rt{k}") };
            dumps.push(FrameDump { source: Some(res), buffer, footprint, format, tag });
        }
    }
    let scalar_dump = if timed_capture.is_some() { copy_scalar_probes(p, index) } else { Vec::new() };
    // the bar and the overlay come after the frame dumps: they hold the title's frame as it is
    let mut rects = crate::progress::current().map_or(Vec::new(), |s| crate::progress::bar(p.w, p.h, s.fraction()).to_vec());
    rects.extend(crate::control::overlay_rects(p.w, p.h));
    if rects.is_empty() {
        p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
    } else {
        fill_rects(p, &bb, &rects);
    }
    if p.list.Close().is_err() {
        return;
    }
    let qraw = queue as *mut c_void;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    let Ok(cl) = p.list.cast::<ID3D12CommandList>() else { return };
    q.ExecuteCommandLists(&[Some(cl)]);
    p.copy_sources = planes.into_iter().map(|(r, _)| r).collect();
    if let Some(frame) = p.dump_frame {
        crate::xlog_warn!("present: frame dump writer busy; skipped frame {frame}");
    }
    p.dump_frame = (!dumps.is_empty()).then_some(index);
    p.dumps = dumps;
    p.scalar_dump = scalar_dump;
    p.value += 1;
    super::markers::queue_fence(queue as usize, "the latest present copy (signal)", &p.fence, p.value);
    if let Err(e) = q.Signal(&p.fence, p.value) {
        // The copy is already submitted. Without its signal, keep its resources and allocator.
        p.disabled = true;
        let e = fence::api_error(&p.dev, "Signal", e);
        crate::xlog_error!("present: submit copy: {e}");
        return;
    }
    if let Some(t) = target_dump {
        super::target_dump::write_after(t, p.fence.clone(), p.value);
    }
    {
        let mut frames = DISPLAY_FRAMES.lock().unwrap();
        if let Err(e) = frames.retire(&p.dev) {
            p.disabled = true;
            crate::xlog_error!("present: retire previous composite: {e}");
            return;
        }
        frames.pending = Some((DisplayFrame { index, surface: bb }, p.fence.clone(), p.value));
    }
    if !PRESENTX.load(Ordering::Relaxed) {
        super::gpu_timing::boundary();
        if super::gpu_timing::enabled() {
            super::gpu_timing_waits::boundary(super::gpu_timing::epoch());
        }
    }
    p.last = Instant::now();
    PRESENTS.fetch_add(1, Ordering::Relaxed);
    LAST_PRESENT_MS.store(now_ms(), Ordering::Relaxed);
    crate::control::note_present();
    if let Some(slot) = timed_capture {
        NEXT_PLANE_CAPTURE.store(slot + 1, Ordering::Relaxed);
    }
    if p.dump_frame.is_some() || !p.scalar_dump.is_empty() {
        match frame_dump_ready(p, frame_dump_wait_ms()) {
            Ok(ready) => { queue_frame_dumps(p, ready); }
            Err(FenceError::Timeout(ms)) => {
                crate::xlog_info!("present: frame dump copy: GPU fence still pending after {ms} ms; will collect at a later present");
            }
            Err(e) => {
                p.disabled = matches!(&e, FenceError::Removed(_));
                crate::xlog_error!("present: frame dump copy: {e}");
            }
        }
    }
}

/// The progress bar (`progress::bar`) along the bottom edge of the back buffer and the
/// performance overlay (`control::overlay_rects`): `rects` filled with their colours on the
/// back buffer, which comes in COPY_SOURCE and leaves in COMMON; the rest of the frame stays as
/// it is.
unsafe fn fill_rects(p: &mut Presenter, bb: &ID3D12Resource, rects: &[(crate::progress::Rect, [f32; 4])]) {
    if p.bar_rtv.is_none() {
        p.bar_rtv = p.dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).ok();
    }
    let Some(heap) = &p.bar_rtv else {
        p.list.ResourceBarrier(&[transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        return;
    };
    let rtv = heap.GetCPUDescriptorHandleForHeapStart();
    p.dev.CreateRenderTargetView(bb, None, rtv);
    // (the RTV hook keeps a reference to what a handle names: none on the swap chain's buffer)
    note_rtv(rtv.ptr, std::ptr::null_mut());
    p.list.ResourceBarrier(&[transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
    for &([left, top, right, bottom], colour) in rects {
        if left < right && top < bottom {
            p.list.ClearRenderTargetView(rtv, &colour, Some(&[RECT { left, top, right, bottom }]));
        }
    }
    p.list.ResourceBarrier(&[transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON)]);
}

/// `XCL_FRAME_DUMP=1`: every 120th shown frame is written to the logs folder (`XCL_FRAME_DUMP=n`,
/// n > 1: every n-th; a title loading at a few frames a second shows few)
/// (`frame_<n>.bmp`, a quarter of each side; `XCL_FRAME_DUMP_FULL=1`: full size) – what the title presented, before the window.
/// `XCL_FRAME_DUMP_WAIT_MS` bounds the extra fence wait (default 2000; 0 polls only).
/// If the copy is still pending, the next completed present collects it before reusing the allocator.
/// Orderly shutdown gets a separate five-second budget to collect the last pending capture and finish writing.
fn frame_dump_due() -> bool {
    if DUMP_SHUTDOWN.load(Ordering::Acquire) {
        return false;
    }
    if LOOK.swap(false, Ordering::Relaxed) {
        return true;
    }
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| match std::env::var("XCL_FRAME_DUMP") {
        Ok(v) => v.parse().ok().filter(|&n| n > 1).unwrap_or(120),
        Err(_) => 0,
    });
    every > 0 && PRESENTS.load(Ordering::Relaxed) % every == 20.min(every - 1)
}

fn trace_planes_at(index: u64) -> bool {
    static FRAMES: OnceLock<Vec<u64>> = OnceLock::new();
    !DUMP_SHUTDOWN.load(Ordering::Acquire)
        && FRAMES.get_or_init(|| std::env::var("XCL_PRESENT_PLANES_AT").unwrap_or_default()
            .split(',').filter_map(|v| v.trim().parse().ok()).take(16).collect()).contains(&index)
}

struct PlaneSchedule {
    after: u64,
    stride: u64,
    count: u64,
}

impl PlaneSchedule {
    fn due(&self, elapsed: u64, next: u64) -> Option<u64> {
        let slot = (elapsed.checked_sub(self.after)? / self.stride).min(self.count - 1);
        (slot >= next).then_some(slot)
    }
}

static NEXT_PLANE_CAPTURE: AtomicU64 = AtomicU64::new(0);

fn plane_schedule() -> Option<&'static PlaneSchedule> {
    static SCHEDULE: OnceLock<Option<PlaneSchedule>> = OnceLock::new();
    SCHEDULE.get_or_init(|| {
        let integer = |key| std::env::var(key).ok().and_then(|v| v.trim().parse::<u64>().ok());
        Some(PlaneSchedule {
            after: integer("XCL_PRESENT_PLANES_AFTER_S")?,
            stride: integer("XCL_PRESENT_PLANES_STRIDE_S").unwrap_or(60).max(1),
            count: integer("XCL_PRESENT_PLANES_COUNT").unwrap_or(3).clamp(1, 16),
        })
    }).as_ref()
}

fn frame_dump_wait_ms() -> u32 {
    static WAIT_MS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WAIT_MS.get_or_init(|| parse_frame_dump_wait_ms(std::env::var("XCL_FRAME_DUMP_WAIT_MS").ok().as_deref()))
}

fn parse_frame_dump_wait_ms(setting: Option<&str>) -> u32 {
    setting.and_then(|ms| ms.trim().parse().ok()).filter(|&ms| ms != u32::MAX).unwrap_or(2000)
}

/// The next shown frame is dumped whatever `XCL_FRAME_DUMP` says (a live look, `XCL_INPUT_FILE`).
pub static LOOK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Compositions submitted so far, including host frames replaced before display.
pub fn presents() -> u64 {
    PRESENTS.load(Ordering::Relaxed)
}

/// Content sampled by an existing `XCL_FRAME_DUMP` readback, not proof of gameplay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameDumpSample {
    /// Zero-based presentation index, as in `frame_<n>.bmp`.
    pub frame: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
    /// One texel every `sample_step` pixels in each direction (normally 4).
    pub sample_step: u32,
    pub sampled_pixels: u64,
    /// Samples whose decoded 8-bit R+G+B exceeds 30; None for an undecoded format.
    pub non_dark_samples: Option<u64>,
}

pub fn last_frame_dump() -> Option<FrameDumpSample> {
    LAST_FRAME_DUMP.lock().ok().and_then(|s| *s)
}

/// a copy of subresource 0 of `src` into a new readback buffer, recorded on `list`
pub(super) unsafe fn readback_copy(list: &ID3D12GraphicsCommandList, src: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let dev: ID3D12Device = {
        let mut d: Option<ID3D12Device> = None;
        src.GetDevice(&mut d).ok()?;
        d?
    };
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    dev.GetCopyableFootprints(desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut rb,
    )
    .ok()?;
    let rb = rb?;
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
    let s = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(src), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    list.CopyTextureRegion(&dst, 0, 0, 0, &s, None);
    Some((rb, fp))
}

/// texel → 8-bit RGB for the formats titles present in (None: not decoded)
pub(super) fn rgb(format: DXGI_FORMAT, t: u32) -> Option<[u8; 3]> {
    Some(match format {
        DXGI_FORMAT_R10G10B10A2_UNORM => [(t >> 2 & 0xFF) as u8, (t >> 12 & 0xFF) as u8, (t >> 22 & 0xFF) as u8],
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => [t as u8, (t >> 8) as u8, (t >> 16) as u8],
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => [(t >> 16) as u8, (t >> 8) as u8, t as u8],
        // HDR: tone-mapped for viewing
        DXGI_FORMAT_R11G11B10_FLOAT => {
            let tm = |v: f32| ((v / (1.0 + v)).sqrt() * 255.0) as u8;
            [tm(small_float(t & 0x7FF, 6)), tm(small_float(t >> 11 & 0x7FF, 6)), tm(small_float(t >> 22, 5))]
        }
        _ => return None,
    })
}

/// the largest render target the draws `XCL_TRACE_SHADERS` picked drew into (AddRef'd): the
/// one the `XCL_TRACE_FRAME` dumps show instead of the busiest
static WATCHED_RT: Mutex<Option<usize>> = Mutex::new(None);
/// targets that what the watched one holds went on to (drawn or copied from it; AddRef'd)
static FOLLOWED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn watched_rt() -> Option<usize> {
    *WATCHED_RT.lock().unwrap()
}

/// The watched target or one its contents went on to.
pub fn is_watched(r: usize) -> bool {
    watched_rt() == Some(r) || FOLLOWED.lock().unwrap().contains(&r)
}

/// `r` now holds (some of) what the watched target held: its readers are traced too.
pub unsafe fn follow(r: usize) {
    let mut g = FOLLOWED.lock().unwrap();
    if r == 0 || g.contains(&r) || g.len() >= 6 || watched_rt() == Some(r) {
        return;
    }
    add_ref(r);
    g.push(r);
}

pub unsafe fn watch_rt(rt: usize) {
    let area = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map_or(0, |x| {
        let d = x.GetDesc();
        if d.SampleDesc.Count > 1 { 0 } else { d.Width * d.Height as u64 }
    });
    let mut g = WATCHED_RT.lock().unwrap();
    if *g == Some(rt) || area(rt) <= g.map_or(0, area) {
        return;
    }
    add_ref(rt);
    if let Some(old) = g.replace(rt) {
        release(old);
    }
}

/// the single-sampled resource bound as a render target most often (`XCL_TRACE_FRAME`; a
/// multisampled one cannot be copied to a buffer – the present list would not close)
fn busiest_rt() -> Option<usize> {
    if let Some(r) = *WATCHED_RT.lock().unwrap() {
        return Some(r);
    }
    let single = |r: usize| unsafe { ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| x.GetDesc().SampleDesc.Count <= 1) };
    USES.lock().unwrap().as_ref()?.iter().filter(|(&r, u)| u[USE_RT] > 0 && single(r)).max_by_key(|(_, u)| u[USE_RT]).map(|(&r, _)| r)
}

/// an 11- / 10-bit unsigned float (5-bit exponent) -> f32
fn small_float(v: u32, mantissa_bits: u32) -> f32 {
    let (e, m) = (v >> mantissa_bits, v & ((1 << mantissa_bits) - 1));
    let m = m as f32 / (1u32 << mantissa_bits) as f32;
    match e {
        0 => m * 2f32.powi(-14),
        31 => 65504.0,
        _ => (1.0 + m) * 2f32.powi(e as i32 - 15),
    }
}

unsafe fn write_frame(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, format: DXGI_FORMAT, tag: &str, n: u64) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let (w, h, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    // every 4th texel of every 4th row; `XCL_FRAME_DUMP_FULL=1`: every texel (1-pixel detail), =2: every 2nd
    let step = std::env::var("XCL_FRAME_DUMP_FULL").map_or(4, |v| v.trim().parse::<usize>().unwrap_or(1).clamp(1, 4));
    let (ow, oh) = ((w / step).max(1), (h / step).max(1));
    let row = (ow * 3).div_ceil(4) * 4;
    let mut bmp = Vec::with_capacity(54 + row * oh);
    let size = (54 + row * oh) as u32;
    bmp.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40, ow as u32, oh as u32] {
        bmp.extend_from_slice(&v.to_le_bytes());
    }
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        bmp.extend_from_slice(&0u32.to_le_bytes());
    }
    let (mut lit, mut decoded) = (0usize, true);
    let mut raw: HashMap<u32, usize> = HashMap::new();
    for y in (0..oh).rev() {
        let start = bmp.len();
        for x in 0..ow {
            let t = ((p as *const u8).add(fp.Offset as usize + step * y * pitch + 4 * step * x) as *const u32).read_unaligned();
            *raw.entry(t).or_default() += 1;
            let c = rgb(format, t).unwrap_or_else(|| {
                decoded = false;
                [0; 3]
            });
            if c.iter().map(|&v| v as u32).sum::<u32>() > 30 {
                lit += 1;
            }
            bmp.extend_from_slice(&[c[2], c[1], c[0]]);
        }
        bmp.resize(start + row, 0);
    }
    rb.Unmap(0, None);
    // `XCL_TRACE_FRAME` also dumps intermediate targets; only the presented image belongs here.
    if tag == "frame" {
        static BOOTED: std::sync::Once = std::sync::Once::new();
        if decoded && lit * 100 > ow * oh {
            BOOTED.call_once(|| crate::xlog_info!("boot: first frame with content at {:.1} s (frame {n}); {}", crate::log::uptime(), crate::gpu::recomp::dxc::stage_summary()));
        }
        if let Ok(mut last) = LAST_FRAME_DUMP.lock() {
            *last = Some(FrameDumpSample {
                frame: n,
                width: w as u32,
                height: h as u32,
                format: format.0,
                sample_step: step as u32,
                sampled_pixels: (ow * oh) as u64,
                non_dark_samples: decoded.then_some(lit as u64),
            });
        }
    }
    let Some(boot) = crate::state::boot() else { return };
    let path = boot.logs_dir().join(format!("{tag}_{n}.bmp"));
    let _ = std::fs::write(&path, &bmp);
    crate::xlog_info!(
        "present: {tag} {n} ({w}x{h}, format {}) written to {} – {lit} of {} samples not black{}",
        format.0,
        path.display(),
        ow * oh,
        if decoded { "" } else { " (format not decoded)" }
    );
    let mut top: Vec<(u32, usize)> = raw.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    crate::xlog_info!("present: {tag} {n} raw texels (most common): {}", top.iter().take(6).map(|(v, c)| format!("{v:#010x}×{c}")).collect::<Vec<_>>().join(" "));
}

impl Drop for Presenter {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_rects_become_uvs_only_inside_the_plane() {
        assert_eq!(source_uv_rect(Some([480, 270, 1440, 810]), 1920, 1080), [0.25, 0.25, 0.75, 0.75]);
        assert_eq!(source_uv_rect(Some([0, 0, 1280, 720]), 1920, 1080), [0.0, 0.0, 1280.0 / 1920.0, 720.0 / 1080.0]);
        assert_eq!(source_uv_rect(Some([0, 0, 1920, 1080]), 1920, 1080), FULL_UV);
        // no rect, inverted, empty, reaching outside the plane, negative
        for rect in [
            None,
            Some([800, 0, 400, 1080]),
            Some([0, 900, 1920, 100]),
            Some([100, 100, 100, 500]),
            Some([0, 300, 1920, 300]),
            Some([0, 0, 1921, 1080]),
            Some([0, 0, 1920, 1081]),
            Some([-1, 0, 960, 540]),
            Some([0, i32::MIN, 960, 540]),
        ] {
            assert_eq!(source_uv_rect(rect, 1920, 1080), FULL_UV, "{rect:?}");
        }
        // a plane of no size
        assert_eq!(source_uv_rect(Some([0, 0, 1, 1]), 0, 1080), FULL_UV);
        assert_eq!(source_uv_rect(Some([0, 0, 1, 1]), 1920, 0), FULL_UV);
    }

    #[test]
    fn composer_shaders_compile() {
        assert!(!super::super::real::compile(COMPOSER_VS, "vs_5_0").is_empty());
        assert!(!super::super::real::compile(COMPOSER_PS, "ps_5_0").is_empty());
    }

    #[test]
    fn planes_carry_their_first_resource_and_source_rect() {
        fn plane(buf: &mut [u8], k: usize, count: u32, list: usize, rect: usize) {
            let b = &mut buf[k * PLANE_BYTES..];
            b[8..12].copy_from_slice(&count.to_le_bytes());
            b[0x10..0x18].copy_from_slice(&list.to_le_bytes());
            b[0x18..0x20].copy_from_slice(&rect.to_le_bytes());
        }
        let rect = [0i32, 0, 1280, 720];
        let resources = [0x1111_0000usize, 0x2222_0000, 0];
        let (r0, r1, null) = (resources.as_ptr() as usize, resources.as_ptr() as usize + 8, resources.as_ptr() as usize + 16);
        let rect_at = rect.as_ptr() as usize;
        let mut buf = vec![0u8; (MAX_PLANES + 2) * PLANE_BYTES];
        plane(&mut buf, 0, 1, r0, rect_at);
        // a null rect pointer, an unreadable one
        plane(&mut buf, 1, 1, r1, 0);
        plane(&mut buf, 2, 2, r1, 0x10);
        // no resources, a null resource: no plane
        plane(&mut buf, 3, 0, r0, rect_at);
        plane(&mut buf, 4, 1, null, rect_at);
        for k in 5..MAX_PLANES + 2 {
            plane(&mut buf, k, 1, r0, rect_at);
        }
        let planes = unsafe { extract_planes((MAX_PLANES + 2) as u32, buf.as_ptr()) };
        let (with, without) = ((resources[0], Some(rect)), (resources[1], None));
        assert_eq!(planes, [with, without, without, with, with, with]);
        assert_eq!(unsafe { extract_planes(u32::MAX, buf.as_ptr()) }.len(), 6);
        assert_eq!(unsafe { extract_planes(0, buf.as_ptr()) }, []);
    }

    #[test]
    fn plane_capture_schedule_depends_on_time_and_does_not_repeat() {
        let schedule = PlaneSchedule { after: 200, stride: 60, count: 3 };
        assert_eq!(schedule.due(199, 0), None);
        assert_eq!(schedule.due(203, 0), Some(0));
        assert_eq!(schedule.due(250, 1), None);
        assert_eq!(schedule.due(262, 1), Some(1));
        assert_eq!(schedule.due(330, 0), Some(2));
        assert_eq!(schedule.due(390, 3), None);
    }

    #[test]
    fn frame_dump_keeps_pending_capture_until_its_fence_completes() {
        assert_eq!(dump_action(None, false), DumpAction::NoCapture);
        assert_eq!(dump_action(None, true), DumpAction::NoCapture);
        assert_eq!(dump_action(Some(23), false), DumpAction::Keep);
        assert_eq!(dump_action(Some(23), true), DumpAction::Collect(23));
    }

    #[test]
    fn frame_dump_wait_is_configurable_and_finite() {
        assert_eq!(parse_frame_dump_wait_ms(None), 2000);
        assert_eq!(parse_frame_dump_wait_ms(Some(" 5500 ")), 5500);
        assert_eq!(parse_frame_dump_wait_ms(Some("0")), 0);
        assert_eq!(parse_frame_dump_wait_ms(Some("-1")), 2000);
        assert_eq!(parse_frame_dump_wait_ms(Some("4294967295")), 2000);
    }

    #[test]
    fn frame_rate_uses_frame_and_time_deltas() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 1_000);
        assert_eq!(
            rate.sample(start + Duration::from_millis(1_500), 1_030),
            20.0
        );
        assert_eq!(
            rate.sample(start + Duration::from_millis(2_750), 1_080),
            40.0
        );
    }

    #[test]
    fn frame_rate_waits_for_a_full_sample() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start, 0), 0.0);
        assert_eq!(rate.sample(start + Duration::from_millis(500), 15), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 30), 30.0);
        assert_eq!(rate.sample(start + Duration::from_millis(1_500), 35), 30.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 40), 10.0);
    }

    #[test]
    fn frame_rate_reports_stalls_and_recovery() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 60), 60.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 60), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(4), 70), 5.0);
    }

    // XCL_WINDOW_POS is process-global; one test covers all modes in sequence.
    #[test]
    fn window_position_modes() {
        unsafe {
            std::env::remove_var("XCL_WINDOW_POS");
            assert_eq!(window_position("auto", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("nonsense", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("12,abc", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));

            let (cx, cy) = window_position("center", 1280, 720);
            if cx != CW_USEDEFAULT {
                // centered on the primary work area: inside it, and not off its left/top
                let mi = {
                    let mon = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
                    let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
                    assert!(GetMonitorInfoW(mon, &mut mi).as_bool());
                    mi
                };
                assert!(cx >= mi.rcWork.left && cy >= mi.rcWork.top);
                assert!(cx + 1280 <= mi.rcWork.right || mi.rcWork.right - mi.rcWork.left < 1280);
            }

            // a position far outside every monitor clamps into the nearest work area
            let (x, y) = window_position("100000,100000", 1280, 720);
            let mon = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST);
            assert!(!mon.is_invalid());

            // the env override wins over the setting
            std::env::set_var("XCL_WINDOW_POS", "center");
            assert_eq!(window_position("auto", 1280, 720), window_position("center", 1280, 720));
            std::env::remove_var("XCL_WINDOW_POS");
        }
    }
}

#[cfg(test)]
mod frame_dump_tests {
    use super::*;
    use super::super::real::D3d12Stats;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    #[test]
    fn reports_without_frame_dumps_remain_readable() {
        let stats = D3d12Stats { presents: 100, ..Default::default() };
        let mut json = serde_json::to_value(&stats).unwrap();
        assert!(json["last_frame_dump"].is_null(), "presents alone do not establish content");
        json.as_object_mut().unwrap().remove("last_frame_dump");
        let old: D3d12Stats = serde_json::from_value(json).unwrap();
        assert_eq!(old.presents, 100);
        assert_eq!(old.last_frame_dump, None);
    }

    #[test]
    fn mapped_frame_dump_distinguishes_dark_bright_and_undecoded_content() {
        unsafe {
            // WARP needs no physical GPU. Fill a CPU-visible buffer with the same padded
            // footprint used by a readback, so this exercises write_frame's actual decoder.
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev = dev.unwrap();
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512 + 256 * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer,
            ).unwrap();
            let buffer = buffer.unwrap();
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 512,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 8, Height: 8, Depth: 1, RowPitch: 256,
                },
            };
            let fill = |red: u8| {
                let mut p: *mut c_void = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let bytes = std::slice::from_raw_parts_mut(p as *mut u8, desc.Width as usize);
                bytes.fill(255); // padding must not be counted as bright pixels
                for y in 0..8 {
                    for x in 0..8 {
                        let start = 512 + y * 256 + x * 4;
                        bytes[start..start + 4].copy_from_slice(&[red, 0, 0, 255]);
                    }
                }
                buffer.Unmap(0, None);
            };
            fill(30);
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame", 17);
            let dark = last_frame_dump().unwrap();
            assert_eq!(dark.frame, 17);
            assert_eq!((dark.width, dark.height, dark.format), (8, 8, 28));
            assert_eq!(dark.sampled_pixels, (8 / dark.sample_step).pow(2) as u64);
            assert_eq!(dark.non_dark_samples, Some(0));

            fill(31);
            write_frame(&buffer, &fp, fp.Footprint.Format, "rt", 18);
            assert_eq!(last_frame_dump(), Some(dark), "intermediate targets must not replace frame content");
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame", 18);
            let bright = last_frame_dump().unwrap();
            assert_eq!(bright.frame, 18);
            assert_eq!(bright.non_dark_samples, Some(bright.sampled_pixels));
            write_frame(&buffer, &fp, DXGI_FORMAT_R32_FLOAT, "frame", 19);
            let json = serde_json::to_value(last_frame_dump().unwrap()).unwrap();
            assert!(json["non_dark_samples"].is_null(), "undecoded is different from dark");
        }
    }
}

/// The XCL window, made now if it is not up yet (None: it could not be made).
pub fn window() -> Option<isize> {
    ensure_window().map(|h| h.0 as isize)
}
