//! Real D3D12.X backend: the title gets genuine D3D12 objects from the PC driver.
//!
//! Every object handed to the title is "adopted": its vtable pointer is replaced with
//! an extended copy:
//!
//! ```text
//! ext[-1]                 original vtable (for calling originals)
//! ext[0 .. public)        original PC methods (a few overridden, see below)
//! ext[public .. internal) guard thunks: runtime/DXGI calls → original internal virtual,
//!                         calls from the game image → logged as console-only method
//! ext[internal .. +64)    logging stubs (console-only methods past the end)
//! ```
//!
//! Overrides: QueryInterface (logs unknown IIDs), every Create* on the device (adopts
//! the new object, retries with the PC IID when the console one is unknown, captures
//! shaders, builds fallback pipeline states when console shaders are rejected),
//! CreateRenderTargetView / OMSetRenderTargets / ExecuteCommandLists (presentation).

#![allow(non_snake_case)]

mod giant_helper;
mod record_variant;

pub use giant_helper::child as giant_helper_child;
pub(crate) mod image_views;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::Serialize;
use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows_sys::core::GUID;

use super::layout::{self, Iface};
use super::residency_list;
use crate::stubs;
use crate::util::{HRESULT, E_INVALIDARG, E_NOINTERFACE, S_FALSE, S_OK};

const DLL: &str = "d3d12_xs.dll";
const EXTRA_SLOTS: usize = 64;
const MAX_SCAN: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    /// the console's DMA (copy) list, ID3D12XboxDmaCommandList: a PC list with the console's
    /// shorter vtable (`console_remap`)
    DmaList,
    Fence,
    DescriptorHeap,
    Resource,
    Heap,
    PipelineState,
    RootSignature,
    QueryHeap,
    CommandSignature,
    PipelineLibrary,
    StateObject,
}

impl Kind {
    pub fn base(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList | Kind::DmaList => "ID3D12GraphicsCommandList",
            Kind::Fence => "ID3D12Fence",
            Kind::DescriptorHeap => "ID3D12DescriptorHeap",
            Kind::Resource => "ID3D12Resource",
            Kind::Heap => "ID3D12Heap",
            Kind::PipelineState => "ID3D12PipelineState",
            Kind::RootSignature => "ID3D12RootSignature",
            Kind::QueryHeap => "ID3D12QueryHeap",
            Kind::CommandSignature => "ID3D12CommandSignature",
            Kind::PipelineLibrary => "ID3D12PipelineLibrary",
            Kind::StateObject => "ID3D12StateObject",
        }
    }
}

// ---------------------------------------------------------------------------------------
// statistics for the report
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct ClassStat {
    pub interface: String,
    pub public_methods: usize,
    pub internal_slots: usize,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct D3d12Stats {
    pub backend: String,
    pub adapter: String,
    pub classes: Vec<ClassStat>,
    pub unknown_iids: Vec<String>,
    pub fallback_pipelines: u64,
    pub failed_pipelines: u64,
    pub fallback_root_signatures: u64,
    pub presents: u64,
    pub frame_boundaries: u64,
    /// Optional content evidence from the existing frame-dump readback; no new GPU readback.
    #[serde(default)]
    pub last_frame_dump: Option<super::present::FrameDumpSample>,
    /// unified memory region opened as a GPU heap (`gpu::uma`): size, bytes in use, buffer VA
    #[serde(default)]
    pub uma_mb: u64,
    #[serde(default)]
    pub uma_used_kb: u64,
    #[serde(default)]
    pub uma_gpu_va: String,
    #[serde(default)]
    pub uma_error: String,
    /// whether the GPU sees CPU writes to the coherent windows (`check_uma_import`)
    #[serde(default)]
    pub uma_import_check: String,
    /// graphics pipelines built from recompiled console shaders (XCL shader ABI root signature)
    #[serde(default)]
    pub translated_pipelines: u64,
    /// why console shaders could not be used (first messages)
    #[serde(default)]
    pub translation_errors: Vec<String>,
    /// pipelines built in the background behind a stand-in: built, failed, still building
    #[serde(default)]
    pub background_pipelines: (usize, usize, usize),
    /// DXIL cache and compiler statistics.
    #[serde(default)]
    pub dxil_cache: crate::gpu::recomp::dxc::CacheStats,
}

static ADAPTER: Mutex<String> = Mutex::new(String::new());
static UNKNOWN_IIDS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
static FALLBACK_PSO: AtomicU64 = AtomicU64::new(0);
static FAILED_PSO: AtomicU64 = AtomicU64::new(0);
static FALLBACK_RS: AtomicU64 = AtomicU64::new(0);
static TRANSLATED_PSO: AtomicU64 = AtomicU64::new(0);
static TRANSLATE_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// XCL shader ABI root signature on the backend device (raw, kept for the process lifetime)
static ABI_RS: AtomicUsize = AtomicUsize::new(0);

/// XCL's shader ABI root signature, once a translated pipeline created it.
pub fn abi_rs_cached() -> Option<*mut c_void> {
    match ABI_RS.load(Ordering::Acquire) {
        0 => None,
        p => Some(p as *mut c_void),
    }
}

thread_local! {
    /// the last compute pipeline created on this thread runs a translated console shader
    static LAST_COMPUTE_TRANSLATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// … with this root signature (static samplers; 0 = the plain ABI one)
    static LAST_COMPUTE_RS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Root signature of the pipelines built from translated console shaders (bind it, then the
/// XCL ABI root UAVs io / res / mem, to draw with them).
pub fn abi_root_signature() -> Option<ID3D12RootSignature> {
    let raw = ABI_RS.load(Ordering::Acquire) as *mut c_void;
    if raw.is_null() {
        return None;
    }
    unsafe { ID3D12RootSignature::from_raw_borrowed(&raw).cloned() }
}
/// Each UMA window as one D3D12 buffer (placed in a heap opened over the process memory):
/// the memory of recompiled shaders (`MEM_VIEWS`). (buffer, GPU VA) per window, kept for the
/// process lifetime.
static UMA_BUFFER: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
/// the committed UMA windows opened as D3D12 heaps (indexed by window `k`; `None` for sparse windows)
static UMA_HEAPS: Mutex<Vec<Option<usize>>> = Mutex::new(Vec::new());
/// console buffers placed in unified memory: resource → (address, bytes)
/// address, bytes, whether XCL allocated the memory (not: the title's own, CreatePlacedResourceX)
type UmaBuffer = (usize, usize, bool);
static UMA_BUFFERS: Mutex<Option<HashMap<usize, UmaBuffer>>> = Mutex::new(None);

/// (diagnostics) The console buffers released lately, the oldest first: address, bytes, whether
/// XCL allocated the memory, when (s of the run). A crash record asks what the GPU could write
/// at an address – also through a buffer that is gone by then.
static UMA_BUFFERS_GONE: Mutex<std::collections::VecDeque<(usize, usize, bool, f32)>> = Mutex::new(std::collections::VecDeque::new());
const UMA_BUFFERS_GONE_KEPT: usize = 8192;

/// (diagnostics, `alias::report`) What XCL knows of console buffers at `p` in a committed
/// window of unified memory – memory the GPU writes where it is, with no copy that XCL makes:
/// the buffers placed over `p` now, and those released there lately.
pub(crate) fn console_buffers_at(p: usize) -> Vec<String> {
    if !crate::gpu::uma::get().is_some_and(|u| u.is_committed(p)) {
        return Vec::new();
    }
    let (Some(live), Some(gone)) = (crate::alias::try_lock(&UMA_BUFFERS), crate::alias::try_lock(&UMA_BUFFERS_GONE)) else {
        return vec!["in unified memory the GPU writes where it is (the console buffers are locked by another thread)".into()];
    };
    let live: Vec<(usize, UmaBuffer)> = live.iter().flatten().map(|(&res, &b)| (res, b)).collect();
    describe_buffers_at(&live, gone.iter().copied().collect(), p, crate::log::uptime() as f32)
}

fn describe_buffers_at(live: &[(usize, UmaBuffer)], gone: Vec<(usize, usize, bool, f32)>, p: usize, now: f32) -> Vec<String> {
    let whose = |own: bool| if own { "memory XCL gave it" } else { "the title's own memory" };
    let over = |at: usize, n: usize| at <= p && p - at < n;
    let mut out: Vec<String> = live
        .iter()
        .filter(|(_, (at, n, _))| over(*at, *n))
        .take(4)
        .map(|(res, (at, n, own))| format!("a console buffer is placed over this address, and the GPU writes it where it is: PC buffer {res:#x} at {at:#x} + {n:#x} ({}), this is its byte {:#x}", whose(*own), p - at))
        .collect();
    let was: Vec<String> = gone
        .iter()
        .rev()
        .filter(|(at, n, ..)| over(*at, *n))
        .take(6)
        .map(|(at, n, own, t)| format!("{t:.3} s ({:.3} s before): {at:#x} + {n:#x} ({}), its byte {:#x}", now - t, whose(*own), p - at))
        .collect();
    if !was.is_empty() {
        out.push(format!("console buffers over this address were released, the latest first: {}", was.join("; ")));
    }
    if out.is_empty() {
        out.push(format!("in unified memory, where the GPU writes without a copy: no console buffer is placed over this address, and none of the last {} released was ({} live)", gone.len(), live.len()));
    }
    out
}

#[cfg(test)]
mod console_buffers_at_tests {
    use super::describe_buffers_at;

    /// The record of an address names the buffer over it with the byte it is, the buffers
    /// released there with when, the latest first, and says so when there is and was none.
    #[test]
    fn an_address_names_its_buffers_live_and_released() {
        let live = [(0x10usize, (0x1000usize, 0x400usize, false)), (0x20, (0x3000, 0x100, true))];
        let gone = vec![(0x1000usize, 0x800usize, true, 10.0f32), (0x5000, 0x100, false, 11.0), (0x0800, 0x1000, false, 12.5)];
        let at = describe_buffers_at(&live, gone.clone(), 0x1234, 14.0);
        assert_eq!(at.len(), 2, "{at:?}");
        assert!(at[0].contains("PC buffer 0x10 at 0x1000 + 0x400 (the title's own memory), this is its byte 0x234"), "{at:?}");
        assert!(at[1].contains("12.500 s (1.500 s before): 0x800 + 0x1000 (the title's own memory), its byte 0xa34; 10.000 s (4.000 s before): 0x1000 + 0x800 (memory XCL gave it), its byte 0x234"), "{at:?}");
        let none = describe_buffers_at(&live, gone, 0x9000, 14.0);
        assert!(none.len() == 1 && none[0].contains("none of the last 3 released was (2 live)"), "{none:?}");
    }
}

/// Where a console-created buffer lives in unified memory (its console GPU address).
pub fn console_buffer(res: usize) -> Option<(usize, usize)> {
    UMA_BUFFERS.lock().unwrap().as_ref()?.get(&res).map(|&(p, n, _)| (p, n))
}

pub(super) fn owned_console_buffer(res:usize)->Option<(usize,usize)> {
    let guard=UMA_BUFFERS.lock().unwrap();
    let &(base,len,owned)=guard.as_ref()?.get(&res)?;
    owned.then_some((base,len))
}

pub(super) fn owned_console_buffer_at(va:usize,bytes:usize)->Option<(ID3D12Resource,usize,usize)> {
    let end=va.checked_add(bytes)?;
    let guard=UMA_BUFFERS.lock().unwrap();
    let mut found=guard.as_ref()?.iter().filter(|(_, &(base,len,owned))| owned && va>=base && base.checked_add(len).is_some_and(|tail| end<=tail));
    let (&raw,&(base,len,_))=found.next()?;
    if found.next().is_some(){return None}
    Some((unsafe{reference(raw)}?,base,len))
}

/// Number of live console buffers in unified memory.
pub fn console_buffers() -> usize {
    UMA_BUFFERS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}
static UMA_ERROR: Mutex<String> = Mutex::new(String::new());

/// GPU virtual address of the unified memory buffer of window 0 (0: not available).
pub fn uma_gpu_va() -> u64 {
    uma_window_vas().first().copied().unwrap_or(0)
}

/// Descriptor index in the shared heap of window 0's raw view (window k's at + k): how
/// translated shaders reach memory (io[`IO_MEM_VIEWS`]). None: UMA is not open on the GPU.
static MEM_VIEWS: Mutex<Option<u32>> = Mutex::new(None);

pub fn mem_views() -> Option<u32> {
    *MEM_VIEWS.lock().unwrap()
}

/// GPU virtual addresses of the UMA window buffers.
pub fn uma_window_vas() -> Vec<u64> {
    UMA_BUFFER.lock().map(|g| g.iter().map(|b| b.1).collect()).unwrap_or_default()
}

pub(super) fn shader_window_base(windows: &[u64]) -> Result<u64, &'static str> {
    if windows.contains(&0) {
        return Err("a shader memory window is not open on the GPU");
    }
    windows.first().copied().ok_or("shader memory is not open on the GPU")
}

#[test]
fn translated_shaders_require_all_active_memory_windows() {
    assert!(shader_window_base(&[]).is_err());
    assert!(shader_window_base(&[0, 0, 0, 0x4000]).is_err());
    assert!(shader_window_base(&[0x1000, 0, 0x3000, 0x4000]).is_err());
    assert_eq!(shader_window_base(&[0x1000]), Ok(0x1000));
    assert_eq!(shader_window_base(&[0x1000, 0x2000, 0x3000, 0x4000]), Ok(0x1000));
}

/// The window buffer holding `len` bytes at UMA address `p`, and their offset in it.
pub(super) fn uma_window_buffer(p: usize, len: usize) -> Option<(usize, u64)> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    let (base, size) = u.window(k);
    if len > size - (p - base) {
        return None;
    }
    let raw = UMA_BUFFER.lock().ok()?.get(k)?.0;
    (raw != 0).then_some((raw, (p - base) as u64))
}

/// GPU virtual address of a UMA address (None: not in UMA or not open on the GPU).
pub fn uma_va_of(p: usize) -> Option<u64> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    let base = UMA_BUFFER.lock().ok()?.get(k)?.1;
    (base != 0).then(|| base + (p - u.window(k).0) as u64)
}

/// What the start-up check of the coherent UMA windows found (`check_uma_import`): empty – not
/// run, "ok …", "off (XCL_UMA_CHECK=0)", or why the GPU does not see the CPU's writes.
static UMA_IMPORT_CHECK: Mutex<String> = Mutex::new(String::new());

/// A word the CPU writes into each coherent window, read back by a GPU copy. The windows are
/// imported as heaps (`OpenExistingHeapFromAddress`), and a runtime that cannot import them may
/// still succeed with memory of its own: vkd3d-proton under a Wine without `userfaultfd` write
/// watches (#175). The GPU then reads zeros where the title wrote, and nothing else says so.
/// Off with `XCL_UMA_CHECK=0`.
unsafe fn check_uma_import(device: &ID3D12Device, u: &crate::gpu::uma::Uma, bufs: &[(usize, u64)]) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let wins: Vec<usize> = (0..bufs.len()).filter(|&k| u.is_committed_window(k) && bufs[k].0 != 0).collect();
    if wins.is_empty() {
        return Ok("no coherent window".into());
    }
    let queue: ID3D12CommandQueue =
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| format!("check not run: queue: {e}"))?;
    let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| format!("check not run: allocator: {e}"))?;
    let list: ID3D12GraphicsCommandList =
        device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| format!("check not run: list: {e}"))?;
    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("check not run: fence: {e}"))?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut readback: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)
        .map_err(|e| format!("check not run: readback buffer: {e}"))?;
    let readback = readback.ok_or("check not run: readback buffer")?;
    // a block of the window's own (title data is not touched), one word per window
    let mut probes = Vec::new();
    for (i, &k) in wins.iter().take(64).enumerate() {
        let Some(p) = u.alloc_in(k, 4, 0) else { continue };
        let canary = 0x5843_4C00 | k as u32; // "XCL" + the window
        std::ptr::write_volatile(p as *mut u32, canary);
        if let Some(src) = ID3D12Resource::from_raw_borrowed(&(bufs[k].0 as *mut c_void)) {
            list.CopyBufferRegion(&readback, 4 * i as u64, src, (p as usize - u.window(k).0) as u64, 4);
        }
        probes.push((k, i, p as usize, canary));
    }
    let run = || -> Result<Vec<u32>, String> {
        list.Close().map_err(|e| format!("check not run: Close: {e}"))?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(|e| e.to_string())?)]);
        queue.Signal(&fence, 1).map_err(|e| format!("check not run: Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, device, None, 2000).map_err(|e| format!("check not run: {e}"))?;
        let mut m = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut m)).map_err(|e| format!("check not run: Map: {e}"))?;
        let got = std::slice::from_raw_parts(m as *const u32, 64).to_vec();
        readback.Unmap(0, None);
        Ok(got)
    };
    let got = run();
    for &(_, _, p, _) in &probes {
        u.free(p);
    }
    let got = got?;
    let wrong: Vec<String> =
        probes.iter().filter(|&&(_, i, _, c)| got[i] != c).map(|&(k, i, _, c)| format!("window {k}: wrote {c:#x}, the GPU read {:#x}", got[i])).collect();
    if !wrong.is_empty() {
        return Err(format!(
            "the GPU does not see what the CPU writes to unified memory ({}): the host-memory import fell back to memory of its own (Wine without userfaultfd write watches?); shaders will read zeros or stale data where the title wrote",
            wrong.join("; ")
        ));
    }
    Ok(format!("ok, {} window(s), {} µs", probes.len(), t0.elapsed().as_micros()))
}

unsafe fn open_uma(device: &ID3D12Device, allow_driver_remap: bool) {
    let Some(u) = crate::gpu::uma::get() else {
        *UMA_ERROR.lock().unwrap() = "no UMA region".into();
        return;
    };
    let open = |k: usize| -> Result<(usize, u64, usize), String> {
        let (base, size) = u.window(k);
        // a driver heap at the window's address where that is faster for the GPU (`uma_driver`),
        // else the window's own pages imported; one line says which
        let (refused, why) = match super::uma_driver::wanted(device, allow_driver_remap) {
            Ok(()) => match super::uma_driver::remake(device, base, size) {
                Ok((heap, buf)) => {
                    u.set_driver_window(k);
                    return Ok(super::uma_driver::raw(heap, buf));
                }
                Err(e) => (true, format!("the driver heap was not made: {e}")),
            },
            Err(why) => (false, why),
        };
        let line = format!("D3D12.X: coherent window at {base:#x} ({} MB): its own pages imported (OpenExistingHeapFromAddress) – {why}", size >> 20);
        if refused {
            crate::xlog_warn!("{line}");
        } else {
            crate::xlog_info!("{line}");
        }
        let dev3: ID3D12Device3 = device.cast().map_err(|e| format!("ID3D12Device3: {e}"))?;
        let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(base as *const c_void).map_err(|e| format!("OpenExistingHeapFromAddress (window {k}): {e}"))?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let va = buf.GetGPUVirtualAddress();
        Ok((buf.into_raw() as usize, va, heap.into_raw() as usize))
    };
    let n = u.shader_windows();
    let mut bufs: Vec<(usize, u64)> = vec![(0, 0); n];
    let mut heaps: Vec<Option<usize>> = vec![None; n];
    for k in 0..n {
        if !u.is_committed_window(k) {
            continue;
        }
        match open(k) {
            Ok((buf_raw, va, heap_raw)) => {
                bufs[k] = (buf_raw, va);
                heaps[k] = Some(heap_raw);
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: UMA not available to the GPU: {e}");
                *UMA_ERROR.lock().unwrap() = e;
                return;
            }
        }
    }
    let check = if std::env::var("XCL_UMA_CHECK").is_ok_and(|v| v == "0") {
        Ok("off (XCL_UMA_CHECK=0)".into())
    } else {
        check_uma_import(device, u, &bufs)
    };
    match &check {
        Ok(s) => crate::xlog_info!("D3D12.X: UMA import check: {s}"),
        Err(e) => crate::xlog_error!("D3D12.X: UMA import check: {e}"),
    }
    *UMA_IMPORT_CHECK.lock().unwrap() = check.unwrap_or_else(|e| e);
    let sparse = super::sparse::init(device);
    for (k, buf_raw, va) in &sparse {
        if let Some(slot) = bufs.get_mut(*k) {
            *slot = (*buf_raw, *va);
        }
    }
    let w0_va = bufs.first().map_or(0, |b| b.1);
    crate::xlog_info!(
        "D3D12.X: UMA opened ({} sparse VRAM window(s), {} MB coherent window), window 0 buffer at VA {:#x}",
        sparse.len(),
        u.size() >> 20,
        w0_va
    );
    match open_mem_views(device, &bufs) {
        Ok(first) => {
            crate::xlog_info!("D3D12.X: the memory windows' raw views are descriptors {first}..{} of the shared heap, the null view {}", first + crate::gpu::recomp::MEM_NULL_VIEW, first + crate::gpu::recomp::MEM_NULL_VIEW);
            *MEM_VIEWS.lock().unwrap() = Some(first);
        }
        Err(e) => crate::xlog_error!("D3D12.X: translated shaders have no memory: {e}"),
    }
    *UMA_HEAPS.lock().unwrap() = heaps;
    *UMA_BUFFER.lock().unwrap() = bufs;
}

/// A raw view over each memory window buffer (all of its 4 GB) in the shared descriptor heap:
/// window k's at the returned index + k, four of them (a missing window gets window 0's), then
/// the null view (`MEM_NULL_VIEW`). Made once; an UMA opened again rewrites them.
unsafe fn open_mem_views(device: &ID3D12Device, bufs: &[(usize, u64)]) -> Result<u32, String> {
    if !super::console_heap::shared_heap(device) {
        return Err("no shared shader-visible descriptor heap".into());
    }
    static OWN: Mutex<Option<(u32, usize, u32)>> = Mutex::new(None);
    let mut own = OWN.lock().unwrap();
    if own.is_none() {
        *own = super::console_heap::reserve_own(crate::gpu::recomp::MEM_NULL_VIEW + 1).map(|(first, cpu, inc)| (first, cpu.ptr, inc));
    }
    let (first, cpu, inc) = own.ok_or("no room in the shared descriptor heap")?;
    let view = |bytes: u64| D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: (bytes.min(1 << 32) / 4) as u32, StructureByteStride: 0, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_RAW },
        },
    };
    let at = |k: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + k as usize * inc as usize };
    for k in 0..crate::gpu::recomp::MEM_NULL_VIEW {
        let raw = bufs.get(k as usize).filter(|b| b.0 != 0).or(bufs.first()).map_or(0, |b| b.0) as *mut c_void;
        let buf = ID3D12Resource::from_raw_borrowed(&raw).ok_or("a memory window has no buffer")?;
        device.CreateUnorderedAccessView(buf, None::<&ID3D12Resource>, Some(&view(buf.GetDesc().Width)), at(k));
        if super::console_heap::vsharp::enabled() {crate::xlog_info!("mesh V# provenance: created raw memory UAV index={} k={k} resource={raw:p} native_va={:#x} width={} num_elements={} raw=true",first+k,buf.GetGPUVirtualAddress(),buf.GetDesc().Width,view(buf.GetDesc().Width).Anonymous.Buffer.NumElements);}
    }
    device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&view(1 << 32)), at(crate::gpu::recomp::MEM_NULL_VIEW));
    if super::console_heap::vsharp::enabled() {crate::xlog_info!("mesh V# provenance: created raw memory UAV index={} k={} null=true resource=None num_elements={} raw=true",first+crate::gpu::recomp::MEM_NULL_VIEW,crate::gpu::recomp::MEM_NULL_VIEW,view(1<<32).Anonymous.Buffer.NumElements);}
    Ok(first)
}

pub fn stats() -> D3d12Stats {
    let classes = CLASSES
        .lock()
        .map(|g| {
            g.iter()
                .map(|c| ClassStat { interface: c.iface.name.to_string(), public_methods: c.public, internal_slots: c.scanned - c.public })
                .collect()
        })
        .unwrap_or_default();
    let (presents, boundaries) = super::present::counters();
    D3d12Stats {
        backend: format!("{:?}", super::backend()).to_lowercase(),
        adapter: ADAPTER.lock().map(|a| a.clone()).unwrap_or_default(),
        classes,
        unknown_iids: UNKNOWN_IIDS.lock().map(|s| s.iter().cloned().collect()).unwrap_or_default(),
        fallback_pipelines: FALLBACK_PSO.load(Ordering::Relaxed),
        failed_pipelines: FAILED_PSO.load(Ordering::Relaxed),
        fallback_root_signatures: FALLBACK_RS.load(Ordering::Relaxed),
        presents,
        frame_boundaries: boundaries,
        last_frame_dump: super::present::last_frame_dump(),
        uma_mb: crate::gpu::uma::get().map(|u| (u.size() >> 20) as u64).unwrap_or(0),
        uma_used_kb: crate::gpu::uma::get().map(|u| (u.used() >> 10) as u64).unwrap_or(0),
        uma_gpu_va: match uma_gpu_va() {
            0 => String::new(),
            v => format!("{v:#x}"),
        },
        uma_error: UMA_ERROR.lock().map(|e| e.clone()).unwrap_or_default(),
        uma_import_check: UMA_IMPORT_CHECK.lock().map(|e| e.clone()).unwrap_or_default(),
        translated_pipelines: TRANSLATED_PSO.load(Ordering::Relaxed),
        translation_errors: TRANSLATE_ERRORS.lock().map(|e| e.clone()).unwrap_or_default(),
        background_pipelines: super::pso_async::counts(),
        dxil_cache: crate::gpu::recomp::dxc::cache_stats(),
    }
}

// ---------------------------------------------------------------------------------------
// device creation
// ---------------------------------------------------------------------------------------

static DEVICE: AtomicUsize = AtomicUsize::new(0);

/// The device's adapter and its user-mode driver's version (`vendor-device-version`): what
/// a driver blob kept on disk is good for (`driver_cache`)
static DRIVER: OnceLock<String> = OnceLock::new();

pub fn device_raw() -> *mut c_void {
    DEVICE.load(Ordering::Acquire) as *mut c_void
}

/// The process's device: the one the unified memory, the physical directory and the descriptor
/// heaps belong to. A test that needs the directory uses this rather than a device of its own,
/// which the singletons refuse - and whose memory would belong to the other device anyway.
#[cfg(test)]
pub(crate) unsafe fn test_device() -> Option<ID3D12Device> {
    init().ok()?;
    ID3D12Device::from_raw_borrowed(&device_raw()).cloned()
}

/// (driver compiles running, giants among them, the bytes reserved for them), for `memdiag`.
pub(crate) fn diag_driver_compiles() -> (usize, usize, u64) {
    let s = DRIVER_COMPILES.state.lock().unwrap_or_else(|e| e.into_inner());
    (s.running, s.giants, s.reserved)
}

/// (console buffers placed in unified memory, title textures kept in step with their memory),
/// each (address, bytes), for `memdiag`.
pub(crate) fn diag_placed() -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let buffers = UMA_BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or_else(Vec::new, |m| m.values().map(|&(at, bytes, _)| (at, bytes)).collect());
    (buffers, super::title_textures::diag_ranges())
}

/// (diagnostics, `memdiag`) The physical directory's current version as (page, GPU address) of
/// each GPU page that names one, with the title textures XCL keeps a video-memory copy of and
/// the textures XCL placed in video memory, each as (address, bytes) (None: no directory).
#[allow(clippy::type_complexity)]
pub(crate) fn diag_imports() -> Option<(Vec<(usize, u64)>, Vec<(usize, usize)>, Vec<(usize, usize)>)> {
    let at = super::physical_directory::current_topology()?.diag_bound().map(|(va, binding)| (binding.token(), va)).collect();
    Some((at, super::title_textures::diag_ranges(), super::vram::diag_placed()))
}

/// (diagnostics, `memdiag`) The title's buffers outside unified memory: (mirrors, the bytes they
/// mirror, the bytes of their seen copies).
pub(crate) fn diag_external_buffers() -> (usize, usize, usize) {
    super::external_buffers::diag_bytes()
}

/// (console shader copies alive, their bytes), for `memdiag`: the copies the builds kept own
/// (`OwnedDesc`), each once (`intern_shader_blob`).
pub(crate) fn diag_shader_copies() -> (usize, u64) {
    let Some(cache) = SHADER_BLOBS.get() else { return (0, 0) };
    let cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    cache.blobs.iter().filter(|(_, blob)| blob.strong_count() != 0).fold((0, 0), |(n, bytes), ((_, len), _)| (n + 1, bytes + *len as u64))
}

/// The process's video memory on the device's adapter: (local, non-local) bytes in use.
/// Entries in XCL's long-lived maps, for `memdiag`.
pub(crate) fn diag_map_sizes() -> String {
    fn n<K, V>(m: &Mutex<Option<HashMap<K, V>>>) -> usize {
        m.lock().unwrap().as_ref().map_or(0, HashMap::len)
    }
    fn s<K>(m: &Mutex<Option<HashSet<K>>>) -> usize {
        m.lock().unwrap().as_ref().map_or(0, HashSet::len)
    }
    format!(
        "pipeline names {}, input layouts {}, shader pairs {}, root signatures {}, user SGPRs {}, translated compute {}, translated graphics {}, unified-memory buffers {}, heaps with blocks {}, derivable bases {}",
        n(&PSO_NAMES),
        n(&PSO_LAYOUTS),
        n(&PSO_SHADERS),
        n(&PSO_RS),
        n(&PSO_USER_SGPRS),
        s(&TRANSLATED_COMPUTE),
        s(&TRANSLATED_GRAPHICS),
        n(&UMA_BUFFERS),
        n(&HEAP_UMA),
        super::device_x::base_count()
    )
}

pub fn video_memory() -> Option<(u64, u64)> {
    let [local, non_local] = video_memory_info()?;
    Some((local.CurrentUsage, non_local.CurrentUsage))
}

pub(super) fn video_adapter_luid() -> Option<windows::Win32::Foundation::LUID> {
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    Some(unsafe { dev.GetAdapterLuid() })
}

pub(crate) fn video_memory_info() -> Option<[windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO; 2]> {
    use windows::Win32::Graphics::Dxgi::*;
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    unsafe {
        let luid = dev.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(luid).ok()?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local).ok()?;
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local).ok()?;
        Some([local, non_local])
    }
}

/// The device runs only 32-lane waves: wave64 compute, mesh and geometry programs need the
/// split form (`Options::split_wave64`). Unknown device: assume so.
fn wave32_only(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::wave32_only_for_device(dev)
    }
}

/// A wave64 VS / PS may run in 32-lane waves here (`pso::graphics_wave32`: the driver picks
/// the wave size of graphics shaders): the split flag for VS / PS translation. Unknown
/// device: assume so.
fn graphics_wave32(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::graphics_wave32_for_device(dev)
    }
}

/// Create the process-wide D3D12 device (the adapter `d3d12_adapter` / `XCL_ADAPTER` chooses,
/// by default the high-performance GPU → WARP); once – later calls
/// return the existing device's adapter.
pub fn init() -> Result<String, String> {
    init_with_remap(false)
}

/// `init`; `allow_driver_remap`: no title code has run, so the coherent window may be remade as a
/// driver heap (`uma_driver`).
pub(super) fn init_with_remap(allow_driver_remap: bool) -> Result<String, String> {
    if !device_raw().is_null() {
        return Ok(ADAPTER.lock().map(|a| a.clone()).unwrap_or_default());
    }
    // one caller makes the device: two at once (tests run side by side) each made one and
    // opened the unified memory twice – a removed device or a fault now and then
    static MAKING: Mutex<()> = Mutex::new(());
    let _making = MAKING.lock().unwrap_or_else(|e| e.into_inner());
    if !device_raw().is_null() {
        return Ok(ADAPTER.lock().map(|a| a.clone()).unwrap_or_default());
    }
    unsafe {
        let cfg = crate::state::config();
        match crate::gpu::agility::state() {
            crate::gpu::agility::State::Loaded { version, dir, .. } => {
                crate::xlog_info!("D3D12: Agility SDK v{version} ({})", dir.display());
            }
            crate::gpu::agility::State::Unavailable(reason) => {
                crate::xlog_warn!("D3D12: Agility SDK unavailable ({reason}) – using inbox D3D12");
            }
            crate::gpu::agility::State::NotFound => {}
        }
        let debug = cfg.d3d12_debug || std::env::var_os("XCL_D3D12_DEBUG").is_some();
        if debug {
            match crate::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
                Ok(d) => {
                    d.EnableDebugLayer();
                    crate::xlog_info!("D3D12: debug layer enabled");
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: debug layer requested but unavailable: {e}");
                }
            }
        }
        // `XCL_DRED=1`: on a device removal the faulting address and the last GPU operations
        // (logged by `log_device_removed`); no debug layer needed. Opt-in: with it XDefiant
        // stalled in its first seconds twice.
        if std::env::var_os("XCL_DRED").is_some() {
            match crate::gpu::agility::configuration_interface::<ID3D12DeviceRemovedExtendedDataSettings>(
                &CLSID_D3D12DeviceRemovedExtendedData,
            ) {
                Ok(d) => {
                    d.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    d.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    // DRED 1.2: the PIX context of each breadcrumb; 1.3: every op, not markers only
                    let context = d.cast::<ID3D12DeviceRemovedExtendedDataSettings1>().map(|s| s.SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON)).is_ok();
                    let every_op = d.cast::<ID3D12DeviceRemovedExtendedDataSettings2>().map(|s| s.UseMarkersOnlyAutoBreadcrumbs(false)).is_ok();
                    crate::xlog_info!("D3D12: DRED auto-breadcrumbs and page-fault reporting on (no debug layer; breadcrumb context {context}, every op rather than markers only {every_op}, settings from {})", if crate::gpu::agility::loaded().is_some() { "the Agility SDK device factory" } else { "the inbox runtime" });
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: DRED requested but unavailable: {e}");
                }
            }
        }
        start_trim();
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI: {e}"))?;
        let mut device: Option<ID3D12Device> = None;
        let mut name = String::new();
        let sel = crate::gpu::adapter::Selector::from_env_or(&cfg.d3d12_adapter);
        if sel != crate::gpu::adapter::Selector::Warp {
            let mut list = crate::gpu::adapter::candidates(&factory, &sel);
            if list.is_empty() {
                crate::xlog_warn!("D3D12: {} – using auto", crate::gpu::adapter::no_match(&factory, &sel));
                list = crate::gpu::adapter::candidates(&factory, &crate::gpu::adapter::Selector::Auto);
            }
            for (adapter, desc) in list {
                if let Ok(dev) = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0) {
                    device = Some(dev);
                    name = format!("{} ({} MiB VRAM)", crate::gpu::adapter::name(&desc), desc.DedicatedVideoMemory >> 20);
                    let umd = adapter.CheckInterfaceSupport(&IDXGIDevice::IID).unwrap_or(0);
                    let _ = DRIVER.set(format!("{:04x}-{:04x}-{umd:016x}", desc.VendorId, desc.DeviceId));
                    crate::gpu::recomp::pso::set_adapter_vendor(desc.VendorId);
                    break;
                }
            }
        }
        let device = match device {
            Some(d) => d,
            None => {
                let warp = crate::gpu::adapter::software_adapter(&factory)?;
                let d = crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0)
                    .map_err(|e| format!("D3D12 device (WARP): {e}"))?;
                name = "WARP (software renderer)".into();
                d
            }
        };
        if std::env::var_os("XCL_DRED").is_some() {
            let version = if device.cast::<ID3D12DeviceRemovedExtendedData2>().is_ok() { "2" } else if device.cast::<ID3D12DeviceRemovedExtendedData1>().is_ok() { "1" } else if device.cast::<ID3D12DeviceRemovedExtendedData>().is_ok() { "0" } else { "none" };
            crate::xlog_info!("D3D12: the device's DRED interface: version {version}");
        }
        if debug {
            // the debug layer's messages into XCL's log (they otherwise go to a debugger only)
            if let Ok(q) = device.cast::<ID3D12InfoQueue1>() {
                let mut cookie = 0u32;
                let _ = q.RegisterMessageCallback(Some(debug_message), D3D12_MESSAGE_CALLBACK_FLAG_NONE, std::ptr::null_mut(), &mut cookie);
            }
        }
        open_uma(&device, allow_driver_remap);
        crate::gpu::rebar::init(&device);
        let directory = if !super::page_mappings::queue::available() {
            "off (XCL_PHYSICAL_DIRECTORY=0)"
        } else if !super::physical_import::init(&device) {
            "off (native physical import capability check failed; play81 unified-memory path)"
        } else if !super::physical_directory::init(&device) {
            "off (directory resources could not be created; play81 unified-memory path)"
        } else {
            "on (queue-ordered GPU page directory for translated shaders)"
        };
        crate::xlog_info!("D3D12.X: physical directory: {directory}");
        super::image_writeback::init();
        super::device_x::probe_dynamic_depth_bias(&device);
        let raw = device.into_raw(); // one reference kept for the process lifetime
        adopt(raw, Kind::Device);
        DEVICE.store(raw as usize, Ordering::Release);
        *ADAPTER.lock().unwrap() = name.clone();
        watch_device_removal();
        Ok(name)
    }
}

/// Trims the files mapped into XCL (§trim_mapped_files§) every second while the PC has less than
/// §TRIM_BELOW§ available, from before the device is made. The driver maps its shader cache while
/// it makes the device, and the pages it reads stay resident: a game with no compile waiting
/// went from 7 GB available to 727 MB in 11 s, and play-91 (the first launch after a rekey, 15 GB
/// of cache views) from 15.5 GB to 21 MB, falling ~1 GB/s through the 9.5 s of device creation,
/// before this thread used to start. Its commit hardly moved: mapped file pages are not commit.
fn start_trim() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-trim".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if available_physical() < TRIM_BELOW {
                trim_mapped_files();
            }
        });
    });
}

/// A debug layer message: errors and warnings into the log – the first 8 of each message id
/// (3 000 in all). One cap for all of them (200) was used up in seconds by two per-frame state
/// errors of a retail game, long before the message that came with its device removal.
unsafe extern "system" fn debug_message(_category: D3D12_MESSAGE_CATEGORY, severity: D3D12_MESSAGE_SEVERITY, id: D3D12_MESSAGE_ID, text: windows::core::PCSTR, _context: *mut c_void) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static PER_ID: Mutex<Option<HashMap<i32, u32>>> = Mutex::new(None);
    if severity.0 > D3D12_MESSAGE_SEVERITY_WARNING.0 {
        return;
    }
    {
        let mut g = PER_ID.lock().unwrap();
        let n = g.get_or_insert_with(HashMap::new).entry(id.0).or_insert(0);
        *n += 1;
        if *n > 8 {
            return;
        }
    }
    if N.fetch_add(1, Ordering::Relaxed) >= 3000 {
        return;
    }
    let t = if text.is_null() { String::new() } else { String::from_utf8_lossy(text.as_bytes()).into_owned() };
    crate::xlog_warn!("D3D12 debug layer ({}, id {}): {t}", if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 { "error" } else { "warning" }, id.0);
}

type QiFn = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
type RefFn = unsafe extern "system" fn(*mut c_void) -> u32;

pub unsafe fn create_device(riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let dev = device_raw();
    if dev.is_null() {
        return E_INVALIDARG;
    }
    if !riid.is_null() {
        let qi: QiFn = std::mem::transmute(*vt(dev));
        let hr = qi(dev, riid as *const GUID, ppv);
        if hr >= 0 {
            crate::xlog_info!("D3D12.X: device {:p} for the title", *ppv);
            if std::env::var_os("XCL_WATCH_DEVICE").is_some() {
                crate::eh::watch_writes(*ppv as usize);
            }
            return hr;
        }
    }
    let add: RefFn = std::mem::transmute(*vt(dev).add(1));
    add(dev);
    *ppv = dev;
    S_OK
}

// ---------------------------------------------------------------------------------------
// vtable adoption
// ---------------------------------------------------------------------------------------

struct Class {
    orig: usize,
    kind: Kind,
    ext: usize,
    iface: &'static Iface,
    public: usize,
    scanned: usize,
}

static CLASSES: Mutex<Vec<Class>> = Mutex::new(Vec::new());
static EXTS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe fn vt(obj: *mut c_void) -> *const usize {
    *(obj as *const *const usize)
}

/// Original function of `slot` for an adopted object.
pub unsafe fn orig_fn(this: *mut c_void, slot: usize) -> usize {
    let ext = vt(this);
    let orig = *ext.sub(1) as *const usize;
    *orig.add(slot)
}

pub(super) unsafe fn original_or_native(this: *mut c_void, slot: usize) -> (usize, bool) {
    let table = vt(this);
    let adopted = is_ext(table as usize);
    let original = if adopted { *table.sub(1) as *const usize } else { table };
    (*original.add(slot), adopted)
}

fn module_of(addr: usize) -> usize {
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    let mut h = std::ptr::null_mut();
    unsafe {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut h);
    }
    h as usize
}

/// Whether `obj` is an object XCL adopted (its vtable is one of XCL's).
pub(super) unsafe fn is_adopted(obj: *mut c_void) -> bool {
    !obj.is_null() && crate::gpu::shader::readable(obj as usize, 8) && is_ext(obj.cast::<usize>().read_unaligned())
}

/// A struct-returning method's first two arguments: true when `a` is the object (the C++ way:
/// this, return buffer, …), false when `b` is (the C way: return buffer, this, …). A return
/// buffer on the stack can hold a stale copy of an object's vtable pointer and look adopted
/// itself: a retail game's call of GetResourceAllocationInfo the C way then had its result
/// written over the device's vtable pointer (the next call read slot 25 of "vtable" 0x1000). So
/// when both look adopted, the one on this thread's stack – or the one that is not the device –
/// is the buffer.
pub(super) unsafe fn this_first(a: *mut c_void, b: *mut c_void) -> bool {
    match (is_adopted(a), is_adopted(b)) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            let (mut lo, mut hi) = (0usize, 0usize);
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi);
            let on_stack = |p: *mut c_void| (lo..hi).contains(&(p as usize));
            let dev = device_raw();
            if on_stack(a) != on_stack(b) {
                on_stack(b)
            } else if a == dev || b == dev {
                a == dev
            } else {
                true
            }
        }
        (false, false) => false,
    }
}

fn is_ext(v: usize) -> bool {
    EXTS.lock().map(|g| g.as_ref().map(|s| s.contains(&v)).unwrap_or(false)).unwrap_or(false)
}

fn kind_of(ext: usize) -> Option<Kind> {
    CLASSES.lock().unwrap().iter().find(|c| c.ext == ext).map(|c| c.kind)
}

/// If a native D3D12 method (such as `Close` or `Reset` on Agility SDK 1.620+, or copy-list
/// `Reset`) replaced `this`'s vtable pointer with another native runtime table, adopt the
/// new table under the same `Kind` so XCL's hooks stay installed while forwarding to the
/// runtime's current state table.
pub(super) unsafe fn preserve_vtable(this: *mut c_void, prev_ext: usize) {
    let cur = *(this as *const usize);
    if cur == prev_ext || is_ext(cur) {
        return;
    }
    if let Some(kind) = kind_of(prev_ext) {
        adopt(this, kind);
    } else {
        *(this as *mut usize) = prev_ext;
    }
}

/// An adopted object gets the extended vtable of another kind (a DMA list: created as a PC
/// list, it needs the console's DMA layout).
pub(super) unsafe fn readopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() || !is_ext(vt(obj) as usize) {
        return adopt(obj, kind);
    }
    let orig = *vt(obj).sub(1);
    *(obj as *mut usize) = class_for(obj, orig, kind);
}

/// Give a genuine D3D12 object the extended vtable for `kind`.
pub unsafe fn adopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() {
        return;
    }
    let v = vt(obj) as usize;
    if is_ext(v) {
        return;
    }
    let ext = class_for(obj, v, kind);
    *(obj as *mut usize) = ext;
}

unsafe fn class_for(obj: *mut c_void, orig: usize, kind: Kind) -> usize {
    if let Some(c) = CLASSES.lock().unwrap().iter().find(|c| c.orig == orig && c.kind == kind) {
        return c.ext;
    }
    let ot = orig as *const usize;
    // most derived PC interface this object implements
    let chain = layout::chain(kind.base());
    let qi: QiFn = std::mem::transmute(*ot);
    let release: RefFn = std::mem::transmute(*ot.add(2));
    let mut iface = *chain.last().expect("layout chain");
    for i in &chain {
        let iid = layout::parse_guid(i.iid);
        let mut out = std::ptr::null_mut();
        if qi(obj, &iid, &mut out) >= 0 && !out.is_null() {
            release(out);
            iface = i;
            break;
        }
    }
    let public = iface.methods.len();
    let m0 = module_of(*ot);
    let mut scanned = 0;
    while scanned < public + MAX_SCAN && module_of(*ot.add(scanned)) == m0 {
        scanned += 1;
    }
    let scanned = scanned.max(public);
    let max_console = console_slots(kind)
        .iter()
        .map(|e| e.0 + 1)
        .chain(console_remap(kind).iter().map(|e| e.0 + 1))
        .max()
        .unwrap_or(0);
    let total = (scanned + EXTRA_SLOTS).max(max_console);
    let mut v: Vec<usize> = Vec::with_capacity(total + 1);
    v.push(orig);
    for i in 0..scanned {
        v.push(*ot.add(i));
    }
    let hook = if kind == Kind::CommandQueue { super::present::on_queue_console_method as usize } else { 0 };
    for i in public..scanned {
        let info = stubs::new_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v[1 + i] = super::guard::make_guard(v[1 + i], info);
    }
    for i in scanned..total {
        let (addr, info) = stubs::make_stub_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v.push(addr);
    }
    for (method, f) in overrides(kind) {
        if let Some(s) = layout::slot(iface, method) {
            if s < public {
                v[1 + s] = f;
            }
        }
    }
    if kind == Kind::CommandList {
        super::list_x::ia_live::observe_slots(iface, &mut v[1..1 + public]);
    }
    if super::residency::eviction_enabled() && matches!(kind, Kind::CommandList | Kind::DmaList) {
        for &method in residency_list::OPAQUE_METHODS {
            if let Some(s) = layout::slot(iface, method) {
                v[1 + s] = super::guard::make_opaque_list_guard(v[1 + s]);
            }
        }
    }
    // the console's layout of the same PC methods (title calls only; the runtime keeps PC's)
    let pc = v.clone();
    for (x, p) in console_remap(kind) {
        if x < public {
            v[1 + x] = super::guard::make_split(pc[1 + p], pc[1 + x]);
        } else if x < scanned {
            v[1 + x] = super::guard::make_split(pc[1 + p], *ot.add(x));
        } else if x < total {
            v[1 + x] = pc[1 + p];
        }
    }
    // slots where the console's method differs from the PC one at the same index
    for (slot, f) in console_slots(kind) {
        if slot < public {
            v[1 + slot] = super::guard::make_split(f, v[1 + slot]);
        } else if slot < scanned {
            // an internal method of the runtime there (e.g. queue #30 = PresentX): the title
            // gets the console method, the runtime keeps its own
            v[1 + slot] = super::guard::make_split(f, *ot.add(slot));
        } else if slot < total {
            v[1 + slot] = f;
        }
    }
    // the other console extensions (device, command list): a title call is logged and answered
    // S_OK, not run as the PC method at that index
    if let Some(first) = console_extensions(kind) {
        let own: HashSet<usize> = console_slots(kind).iter().map(|e| e.0).chain(console_remap(kind).iter().map(|e| e.0)).collect();
        for i in (first..public).filter(|i| !own.contains(i)) {
            let (stub, _) = stubs::make_stub_info(DLL, &format!("{}::#{i} (console extension)", iface.name), 0);
            v[1 + i] = super::guard::make_split(stub, v[1 + i]);
        }
    }
    // methods the newer console layout keeps at other slots: picked at each call
    if kind == Kind::CommandList {
        let older_handlers = console_slots(kind);
        for (newer, older) in NEWER_LIST_SLOTS {
            let Some(&(_, f)) = older_handlers.iter().find(|e| e.0 == older) else { continue };
            if newer >= total {
                continue;
            }
            let pc = if newer < scanned { *ot.add(newer) } else { v[1 + newer] };
            let pc = if super::residency::eviction_enabled() && newer < public {
                super::guard::make_opaque_list_guard(pc)
            } else { pc };
            v[1 + newer] = super::guard::make_layout_split(super::guard::LayoutTargets { older: v[1 + newer], newer: f, pc, newer_layout: &QUEUE_SHIFTED });
        }
        // mesh dispatches: each layout has them at its own slots (past the PC's public methods),
        // and the other layout has another method there, which keeps what the slot had
        for (older, newer, f) in MESH_LIST_SLOTS.map(|(o, n, k)| (o, n, mesh_list_method(k))) {
            if newer >= total || older < public {
                continue;
            }
            // a method of the runtime's own at that index keeps its calls (as `console_slots`)
            let title = |slot: usize| if slot < scanned { super::guard::make_split(f, *ot.add(slot)) } else { f };
            v[1 + older] = super::guard::make_layout_switch(title(older), v[1 + older], &QUEUE_SHIFTED);
            v[1 + newer] = super::guard::make_layout_switch(v[1 + newer], title(newer), &QUEUE_SHIFTED);
        }
    }
    if kind == Kind::CommandList && super::present::list_trace_after().is_some() {
        for i in 0..total {
            v[1 + i] = super::guard::make_list_tracer(v[1 + i], i);
        }
    }
    // XCL_D3D12_TRACE: every public method called from console code is logged
    if std::env::var_os("XCL_D3D12_TRACE").is_some() {
        for i in 0..public {
            let name = format!("{}::{} (#{i})", iface.name, iface.methods.get(i).copied().unwrap_or("?"));
            v[1 + i] = super::guard::make_tracer(v[1 + i], name);
        }
    }
    let leaked: &'static [usize] = Box::leak(v.into_boxed_slice());
    let ext = leaked.as_ptr() as usize + 8;
    EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(ext);
    crate::xlog_info!(
        "D3D12.X: {} – {public} PC methods, {} internal slots under the guard, +{EXTRA_SLOTS} stubs ({kind:?}, PC vtable {})",
        iface.name,
        scanned - public,
        crate::modules::describe_address(orig)
    );
    CLASSES.lock().unwrap().push(Class { orig, kind, ext, iface, public, scanned });
    ext
}

/// (console slot, PC slot) of PC methods that sit elsewhere in the console's vtable.
/// ID3D12GraphicsCommandList (from the Fortnite RHI call sites, matching XCL's own evidence
/// – FSR4 calls ClearUnorderedAccessViewUint at #53): 0–27 and 56–59 as on PC; the query
/// methods come right after ExecuteBundle (28–31 = PC 52–55: BeginQuery, EndQuery,
/// ResolveQueryData, SetPredication), so SetDescriptorHeaps … DiscardResource move up by 4
/// (32–55 = PC 28–51); ID3D12GraphicsCommandList1 is at 104–109 (PC 60–65), the shading-rate
/// methods at 124 / 125 (PC 77 / 78). The console-only methods in between: `list_x`.
fn console_remap(kind: Kind) -> Vec<(usize, usize)> {
    match kind {
        Kind::CommandList => (28..=31).map(|x| (x, x + 24)).chain((32..=55).map(|x| (x, x - 4))).chain((104..=109).map(|x| (x, x - 44))).chain([(124, 77), (125, 78)]).collect(),
        // the DMA list: Close / Reset, then the copy methods (XDefiant calls #12 with
        // CopyTextureRegion's arguments): 11–14 = PC 15–18 (CopyBufferRegion, CopyTextureRegion,
        // CopyResource, CopyTiles); what follows is not known yet (`console_extensions`)
        Kind::DmaList => (11..=14).map(|x| (x, x + 4)).collect(),
        _ => Vec::new(),
    }
}

/// console D3D12_DESCRIPTOR_HEAP_TYPE also used for CBV / SRV / UAV heaps
const XBOX_EXTRA_CBV_SRV_UAV: u32 = 4;

/// Map a console `D3D12_DESCRIPTOR_HEAP_TYPE` to its PC equivalent.
#[inline]
fn pc_heap_type(ty: u32) -> u32 {
    if ty == XBOX_EXTRA_CBV_SRV_UAV {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32
    } else {
        ty
    }
}

/// Where a class's console extensions start: the device's from #44 (`device_x`), the command
/// list's from #60 (`list_x`).
fn console_extensions(kind: Kind) -> Option<usize> {
    match kind {
        Kind::Device => Some(44),
        Kind::CommandList => Some(60),
        Kind::DmaList => Some(15),
        _ => None,
    }
}

/// Console vtable slots that differ from the PC layout (found on console code calling them):
/// * device #62: `CreateComputePipelineState` with two extra arguments (XboxBvhBuilder.dll
///   passes a D3D12_COMPUTE_PIPELINE_STATE_DESC, NULL, NULL, riid, ppv); PC #62 = CreateStateObject.
/// * device #76: `CreateCommittedResource` (heap properties, flags, desc, state, clear value,
///   riid, ppv) as on PC #27 (PC #76 = CreateCommittedResource3); the console resource desc
///   carries more after the 52 PC bytes (amd_fidelityfx_upscaler_xbox.dll writes 3 at +52).
/// * descriptor heap #9 / #10: the heap start handles come back in rax on the console.
/// * resource #11 GetGPUVirtualAddress: console buffers live in unified memory; their console
///   address (= CPU address, low 32 bits = offset in `mem`) is what console code gets.
/// * device #14 / #15 / #17 / #18 / #19 / #22 / #23 / #24 (CreateDescriptorHeap,
///   GetDescriptorHandleIncrementSize, CreateConstantBufferView, CreateShaderResourceView,
///   CreateUnorderedAccessView, CreateSampler, CopyDescriptors, CopyDescriptorsSimple):
///   CBV/SRV/UAV and sampler heaps of console code get 32 / 16-byte slots in unified memory
///   (`console_heap`). #14–#19 seen from console code with the PC layout; #17 and #22–#24
///   assumed to follow it (each checks its arguments and otherwise goes to the PC method).
/// * device #12 `CreateCommandList`: console list types; with console lists on, console code
///   gets a console-shaped list (D3D12.X inline code writes into it), XCL's own code a PC one.
/// * the device's own methods (`device_x`, `frame`), the command list's (`list_x`), queue #30
///   PresentX (`present`).
fn console_slots(kind: Kind) -> Vec<(usize, usize)> {
    use super::device_x as x;
    match kind {
        Kind::Device => {
            vec![
                (0, dev_qi_x as usize),
                (8, x::create_command_queue_x as usize),
                (25, x::get_resource_allocation_info_x as usize),
                (26, x::get_custom_heap_properties_x as usize),
                (29, dev_create_placed_resource_x as usize),
                // CreateCommittedResource at its PC index too (XDefiant): console code uses the
                // buffer's GPU address as a CPU pointer – it goes into unified memory
                (27, dev_create_committed_resource_x as usize),
                (9, x::create_command_allocator_x as usize),
                (10, x::create_graphics_pso_pc as usize),
                (12, x::create_command_list_pc as usize),
                (14, dev_create_descriptor_heap_x as usize),
                (15, dev_increment_size_x as usize),
                (17, dev_create_cbv_x as usize),
                (18, dev_create_srv_x as usize),
                (21, dev_create_dsv_x as usize),
                (19, dev_create_uav_x as usize),
                (22, dev_create_sampler_x as usize),
                (23, dev_copy_descriptors_x as usize),
                (24, dev_copy_descriptors_simple_x as usize),
                (38, x::get_copyable_footprints_x as usize),
                (41, x::create_command_signature_x as usize),
                (44, x::create_placed_resource_x as usize),
                (48, x::create_command_queue_x as usize),
                (52, x::set_debug_flags_x as usize),
                (55, x::create_derived_pso_x as usize),
                (56, x::serialize_pso_x as usize),
                (57, x::deserialize_pso_x as usize),
                (60, x::create_graphics_pso_x as usize),
                (61, x::create_sampler_x as usize),
                (62, dev_create_compute_pso_x as usize),
                (66, x::serialize_pso_x as usize),
                (67, x::deserialize_pso_x as usize),
                (70, dev_create_srv_at_x as usize),
                (71, dev_create_uav_at_x as usize),
                (73, x::create_command_list_x as usize),
                (74, x::register_page_pool_x as usize),
                (75, x::unregister_page_pool_x as usize),
                (76, dev_create_committed_resource_x as usize),
                // frame pacing (SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX,
                // GetFrameStatisticsX)
                (78, super::frame::set_frame_interval_x as usize),
                (79, super::frame::schedule_frame_event_x as usize),
                (80, super::frame::wait_frame_event_x as usize),
                (81, super::frame::frame_statistics_x as usize),
                (86, x::create_pipeline_state_x as usize),
                (108, x::get_resource_allocation_info2_x as usize),
                (127, dev_create_committed_resource_layout_x as usize),
                (133, x::create_placed_resource_layout_x as usize),
            ]
        }
        Kind::DescriptorHeap => vec![(9, heap_cpu_start_x as usize), (10, heap_gpu_start_x as usize)],
        Kind::Resource => vec![(11, res_gpu_va_x as usize)],
        Kind::PipelineState => vec![(11, pso_console_blob_x as usize)],
        Kind::Fence => vec![(12, super::fence_x::completed_value_address as usize)],
        // the title's calls from #14 on: Signal / Wait / … / SignalX / PresentX at the slots of
        // its runtime's layout (`queue_dispatch`)
        Kind::CommandQueue => QUEUE_SLOTS.iter().enumerate().map(|(k, &f)| (QUEUE_FIRST + k, f as usize)).collect(),
        Kind::DmaList => {
            use super::list_x as l;
            vec![(9, l::dma_close_x as usize), (10, l::reset_x as usize), (11, l::copy_buffer_region_x as usize), (12, l::copy_texture_region_x as usize), (13, l::copy_resource_x as usize)]
        }
        Kind::CommandList => {
            use super::bind as b;
            use super::list_x as l;
            vec![
                // compute root arguments of console code: kept for translated dispatches
                (12, b::draw_instanced as usize),
                (15, l::copy_buffer_region_x as usize),
                (16, l::copy_texture_region_x as usize),
                (17, l::copy_resource_x as usize),
                (19, l::resolve_subresource_x as usize),
                (51, image_views::clear_dsv_guest as usize),
                (53, l::clear_uav_uint_x as usize),
                (54, l::clear_uav_float_x as usize),
                (108, l::resolve_subresource_region_x as usize),
                (13, b::draw_indexed_instanced as usize),
                (14, b::dispatch as usize),
                (48, l::ia_set_vertex_buffers_x as usize),
                // state a PC Reset clears, kept for KickoffX
                (20, l::ia_set_topology_x as usize),
                (21, l::rs_set_viewports_x as usize),
                (22, l::rs_set_scissors_x as usize),
                (23, l::om_set_blend_factor_x as usize),
                (24, l::om_set_stencil_ref_x as usize),
                (32, l::set_descriptor_heaps_x as usize),
                (59, b::execute_indirect as usize),
                (83, l::copy_texture_region_flags_x as usize),
                (34, b::set_graphics_root_signature as usize),
                (36, b::set_graphics_root_table as usize),
                (38, b::set_graphics_root_constant as usize),
                (40, b::set_graphics_root_constants as usize),
                (42, b::set_graphics_root_descriptor as usize),
                (44, b::set_graphics_root_descriptor as usize),
                (46, b::set_graphics_root_descriptor as usize),
                (26, l::resource_barrier_x as usize),
                (25, b::set_pipeline_state as usize),
                (33, b::set_compute_root_signature as usize),
                (35, b::set_compute_root_table as usize),
                (37, b::set_compute_root_constant as usize),
                (39, b::set_compute_root_constants as usize),
                (41, b::set_compute_root_descriptor as usize),
                (43, b::set_compute_root_descriptor as usize),
                (45, b::set_compute_root_descriptor as usize),
                (10, l::reset_x as usize),
                (28, l::begin_query_x as usize),
                (29, l::end_query_x as usize),
                (31, l::set_predication_x as usize),
                (30, l::resolve_query_data_x as usize),
                (47, l::ia_set_index_buffer_x as usize),
                (60, l::copy_memory_x as usize),
                (61, l::fill32_x as usize),
                (62, l::fill64_x as usize),
                (63, l::fill128_x as usize),
                (67, l::flush_pipeline_x as usize),
                (68, l::set_depth_bounds_x as usize),
                (69, l::write32_bottom_x as usize),
                (70, l::write64_bottom_x as usize),
                (71, l::wait32_x as usize),
                (72, l::wait64_x as usize),
                (78, l::dispatch_x as usize),
                (81, l::draw_indexed_x as usize),
                (84, l::copy_resource_flags_x as usize),
                (85, l::set_predication_buffer_x as usize),
                (88, l::write32_top_x as usize),
                (89, l::write64_top_x as usize),
                (90, l::write32_bottom_of_shader_x as usize),
                (93, l::kickoff_x as usize),
                (94, l::set_depth_bias_x as usize),
                (131, l::flush_cache_range_x as usize),
                (132, l::flush_cache_range_x as usize),
                // console compute/BVH builder extensions (`XboxBvhBuilder.dll`)
                (146, l::uav_barrier_x as usize),
                (148, b::allocate_upload_memory as usize),
            ]
        }
        _ => vec![],
    }
}

/// descriptor heap #9 / #10 GetCPU / GetGPUDescriptorHandleForHeapStart: console code
/// takes the handle in rax (only `this` passed); the PC methods return it through a hidden
/// pointer
unsafe extern "system" fn heap_cpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 9));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn heap_gpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 10));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn dev_create_descriptor_heap_x(this: *mut c_void, desc: *const [u32; 4], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut d = if desc.is_null() { [u32::MAX; 4] } else { *desc };
    d[0] = pc_heap_type(d[0]);
    let desc = if desc.is_null() { desc } else { &d as *const [u32; 4] };
    let sampler = d[0] == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32;
    if d[0] != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 && !sampler {
        return dev_create_descriptor_heap(this, desc as usize, riid, ppv);
    }
    // views: an SRV / UAV pair per console slot (in the shared shader-visible heap, the
    // title's own a CPU-only mirror – `console_heap`); samplers: one per slot
    let pc_n = if sampler { d[1] } else { d[1].saturating_mul(2) };
    let visible = d[2] & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0;
    let shared = match ID3D12Device::from_raw_borrowed(&this) {
        Some(dev) if !sampler && !desc.is_null() => super::console_heap::shared_heap(dev),
        _ => false,
    };
    let flags = if shared { d[2] & !(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32) } else { d[2] };
    let pc = [d[0], pc_n, flags, d[3]];
    let hr = dev_create_descriptor_heap(this, pc.as_ptr() as usize, riid, ppv);
    if hr >= 0 {
        let raw = *ppv;
        let heap = ID3D12DescriptorHeap::from_raw_borrowed(&raw).unwrap();
        let dev = ID3D12Device::from_raw_borrowed(&this).unwrap();
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let gpu = if flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0 { heap.GetGPUDescriptorHandleForHeapStart().ptr } else { 0 };
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE(d[0] as i32));
        if sampler {
            // an S# of zeros – a slot nothing was written to – is a sampler on the console: point
            // filtering, wrap, mip 0 only. Every PC slot starts as that one, so a table entry the
            // title left empty samples as there (an unwritten PC descriptor is undefined).
            let zero = D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: 0.0,
            };
            for k in 0..d[1] as u64 {
                dev.CreateSampler(&zero, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize });
            }
        }
        if !super::console_heap::register(raw as usize, d[1], cpu, gpu, inc, sampler, shared, visible) {
            crate::xlog_warn!("console descriptor heap: no room in UMA for {} slots", d[1]);
        }
    }
    hr
}

unsafe extern "system" fn dev_increment_size_x(this: *mut c_void, ty: u32) -> u32 {
    let ty = pc_heap_type(ty);
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 {
        return super::console_heap::SLOT;
    }
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 {
        return super::console_heap::SAMPLER_SLOT;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32) -> u32 = std::mem::transmute(orig_fn(this, 15));
    f(this, ty)
}

/// What a console view slot describes: (GFX10 format, image type, width, height, depth, mips)
/// from the view (or the resource when the view is NULL); None for buffers.
unsafe fn view_shape(res: &ID3D12Resource, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, dim: i32, array: u32, mips: u32) -> Option<(u32, crate::gpu::desc::ImageType, u32, u32, u32, u32)> {
    use crate::gpu::desc::ImageType as T;
    let r = res.GetDesc();
    let f = if format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN { r.Format } else { format };
    let texel = if super::device_x::stencil_view(r.Format.0 as u32, f.0 as u32) {
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UINT
    } else {
        f
    };
    let fmt = crate::gpu::desc::gfx10_format(texel).unwrap_or(0);
    let mips = if mips == 0 || mips == u32::MAX { r.MipLevels as u32 } else { mips };
    let (w, h, d) = (r.Width as u32, r.Height, r.DepthOrArraySize as u32);
    let ty = match dim {
        2 => T::D1,
        3 => T::D1Array,
        4 => T::D2,
        5 => T::D2Array,
        6 => T::D2Msaa,
        7 => T::D2MsaaArray,
        8 => T::D3,
        9 | 10 => T::Cube,
        _ => return None,
    };
    let depth = match ty {
        T::D3 => d,
        T::D1Array | T::D2Array | T::D2MsaaArray => array.max(1),
        // a cube view: its faces (6 per cube)
        T::Cube => array.max(6),
        _ => 1,
    };
    Some((fmt, ty, w, h, depth, mips))
}

fn resource_dim_to_view(r: &D3D12_RESOURCE_DESC) -> i32 {
    match r.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if r.DepthOrArraySize > 1 {
                3
            } else {
                2
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            if r.DepthOrArraySize > 1 {
                5
            } else {
                4
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 8,
        _ => 1,
    }
}

static NON_UMA_BUFFER_VIEWS: AtomicU64 = AtomicU64::new(0);

/// V# of a buffer view (raw: 4-byte elements, num_records in bytes; structured / typed:
/// stride = element size, num_records in elements); empty when the buffer is not in UMA.
fn buffer_view(res: usize, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let Some((at, bytes)) = console_buffer(res) else {
        note_buffer_view();
        return [0; 4];
    };
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = if raw {
        (0, 4)
    } else if stride > 0 {
        (stride, stride)
    } else {
        let e = crate::gpu::desc::format_bytes(fmt);
        (e, e)
    };
    let base = at as u64 + first * elem as u64;
    let room = (bytes as u64).saturating_sub(first * elem as u64);
    let records = if stride == 0 { (n as u64 * 4).min(room) } else { (n as u64).min(room / stride as u64) };
    crate::gpu::desc::buffer_words(base, stride, records as u32, fmt)
}

/// Console buffer → a placed resource in the unified memory heap, so its console address is
/// what the recompiled shaders read through `mem`. None: not placed (fall back to committed).
/// A committed texture of console code: on the console its memory is unified memory too –
/// titles write it through its GPU address (XDefiant clears one with memset). It gets memory
/// of the console's size (XG; the PC's where XG has no model) and is created as a texture at
/// that address (`device_x::create_placed_resource_x`: kept in step with the memory where XG
/// lays it out); GetGPUVirtualAddress returns the address. The memory is the coherent
/// window's, as long as it keeps `COHERENT_KEPT` free; beyond that the texture is a plain PC
/// one (None). Unicorn Overlord's 2.2 GB of committed textures filled the 512 MB window in its
/// first second, leaving no room for readback heaps. (In the sparse windows instead, XDefiant
/// rendered black – not understood yet.)
/// bytes of the coherent window committed textures leave free: readback / write-back heaps
/// and buffers, XCL's blocks outside its arena pool
const COHERENT_KEPT: usize = 128 << 20;

unsafe fn place_console_texture(this: *mut c_void, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if ppv.is_null() || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if !(2..=4).contains(&u32::from_le_bytes(x[0..4].try_into().unwrap())) {
        return None;
    }
    let (size, align) = match crate::hle::xg::Texture::new(&x) {
        Ok(t) if t.size() > 0 => (t.size(), t.alignment()),
        _ => {
            type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
            let pc: F = std::mem::transmute(orig_fn(this, 25));
            let mut r = [0u64; 2];
            pc(this, &mut r, 0, 1, &super::device_x::pc_resource_desc(&x));
            (r[0], r[1])
        }
    };
    if size == 0 || size == u64::MAX {
        return None;
    }
    let align = (align as usize).max(0x1_0000);
    let bytes = (size as usize).div_ceil(align) * align;
    if u.size().saturating_sub(u.used()) < bytes + COHERENT_KEPT {
        return None;
    }
    let p = u.alloc(bytes, align)? as usize;
    let hr = super::device_x::create_placed_resource_x(this, p as u64, desc as *const u8, state as u32, clear as *const c_void, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, size as usize, true));
    Some(hr)
}

/// Largest single `D3D12_HEAP_TYPE_READBACK` allocation kept on the coherent window; larger
/// ones spill to sparse VRAM so one big readback heap/buffer cannot exhaust coherent headroom.
const COHERENT_READBACK_MAX: usize = 64 << 20;
/// Largest single `D3D12_HEAP_TYPE_CUSTOM` + `WRITE_BACK` `COPY_DEST` buffer kept on coherent RAM.
const COHERENT_CUSTOM_WB_MAX: usize = 16 << 20;

/// bytes of the coherent window readback heaps and buffers leave free (XCL's blocks outside its
/// arena pool)
const COHERENT_FLOOR: usize = 16 << 20;

/// Whether a readback heap / buffer of `rounded` bytes may take coherent memory. The room
/// committed textures leave (`COHERENT_KEPT`) is kept for these: with the textures at that
/// limit (Unicorn Overlord's), a second 128 MB of headroom sent every readback buffer to the
/// sparse windows.
fn has_coherent_headroom(u: &crate::gpu::uma::Uma, rounded: usize) -> bool {
    u.size().saturating_sub(u.used()) >= COHERENT_FLOOR.min(u.size() / 8).saturating_add(rounded)
}

/// Sparse-backed buffers whose `Map(0, NULL, ...)` should read back their full range from VRAM
/// (readback buffers that spilled to sparse VRAM or `COPY_DEST` buffers in `CUSTOM + WRITE_BACK`).
static SPARSE_READBACK: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn note_sparse_readback(res: usize) {
    SPARSE_READBACK.lock().unwrap().get_or_insert_with(HashSet::new).insert(res);
}

/// Whether `res` is a sparse-backed buffer the title made to read on the CPU (`SPARSE_READBACK`).
pub(super) fn is_sparse_readback(res: usize) -> bool {
    SPARSE_READBACK.lock().unwrap().as_ref().is_some_and(|s| s.contains(&res))
}

/// Whether `props` asks for CPU-cached readback memory. On Xbox UMA, `D3D12_HEAP_TYPE_CUSTOM`
/// with `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` is the general CPU-cached graphics pool (used for
/// hundreds of MB of VBs/IBs/CBs/UAVs), so only explicit `D3D12_HEAP_TYPE_READBACK` heaps — or
/// small non-UAV `COPY_DEST` buffers — stay in the 512 MB coherent window.
fn is_coherent_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS, bytes: usize) -> bool {
    if props.Type == D3D12_HEAP_TYPE_READBACK {
        return bytes <= COHERENT_READBACK_MAX;
    }
    props.Type == D3D12_HEAP_TYPE_CUSTOM
        && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
        && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0
        && bytes <= COHERENT_CUSTOM_WB_MAX
}

fn is_cpu_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS) -> bool {
    props.Type == D3D12_HEAP_TYPE_READBACK
        || (props.Type == D3D12_HEAP_TYPE_CUSTOM
            && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
            && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0)
}

unsafe fn place_console_buffer_sparse(this: *mut c_void, d: &D3D12_RESOURCE_DESC, bytes: usize, state: u32, readback_on_null_map: bool, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let (p, _) = super::sparse::alloc(bytes, 0x1_0000)?;
    if let Some((hr, delta)) = super::sparse::place_buffer(this, p, d, state, riid, ppv) {
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        if readback_on_null_map {
            note_sparse_readback(*ppv as usize);
        }
        return Some(hr);
    }
    super::sparse::free(p);
    None
}

pub(super) unsafe fn place_external_buffer(this: *mut c_void, title: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let bytes = desc.Width as usize;
    if let Some(why) = super::external_buffers::unsupported(title, bytes) {
        let mut m: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        windows_sys::Win32::System::Memory::VirtualQuery(title as *const c_void, &mut m, std::mem::size_of_val(&m));
        crate::xlog_warn!(
            "D3D12.X: buffer at {title:#x} ({bytes:#x} bytes) outside unified memory: {why} (allocation {:#x}, protection {:#x}, type {:#x})",
            m.AllocationBase as usize, m.AllocationProtect, m.Type
        );
        return crate::util::E_INVALIDARG;
    }
    let mut raw = std::ptr::null_mut();
    // inside a buffer that is mirrored already: the same memory on the console, and the same
    // video memory here – placed in that mirror, which lives as long as either
    if let Some((at, block)) = super::external_buffers::hold(title, bytes) {
        let Some((hr, delta)) = super::sparse::place_buffer(this, at, desc, state, riid, &mut raw) else {
            if let Some(block) = super::external_buffers::unhold(block) {
                super::sparse::release_owned_block(block);
            }
            return crate::util::E_FAIL;
        };
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(raw as usize, (at, bytes, false));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(raw as usize, delta as u64);
        }
        super::external_buffers::register_in(raw as usize, title, at, bytes, block);
        *ppv = raw;
        return hr;
    }
    let Some(hr) = place_console_buffer_sparse(this, desc, bytes, state, false, riid, &mut raw) else {
        return crate::util::E_FAIL;
    };
    // (the mirror's video memory is given back when the last buffer placed in it goes,
    // `external_buffers::forget` – not with this one)
    let mirror = match UMA_BUFFERS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(raw as usize))) {
        Some(entry) => {
            entry.2 = false;
            entry.0
        }
        None => return crate::util::E_FAIL,
    };
    super::external_buffers::register(raw as usize, title, mirror, bytes);
    *ppv = raw;
    hr
}

/// Sanitize a console buffer description and initial state for placement in a coherent
/// cross-adapter UMA heap (`OpenExistingHeapFromAddress`); `cross_adapter`: the window is such a heap, not a
/// driver heap (`uma_driver`), which takes no cross-adapter resource.
fn sanitize_coherent_buffer(desc: &D3D12_RESOURCE_DESC, state: usize, cross_adapter: bool) -> (D3D12_RESOURCE_DESC, u32) {
    let mut d = *desc;
    d.Flags = D3D12_RESOURCE_FLAGS(d.Flags.0 & 0xFF);
    if cross_adapter {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    (d, state as u32 & PC_RESOURCE_STATES)
}

unsafe fn place_console_buffer(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if UMA_BUFFER.lock().unwrap().is_empty() || ppv.is_null() {
        return None;
    }
    let d: D3D12_RESOURCE_DESC = std::ptr::read(desc as *const D3D12_RESOURCE_DESC);
    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || d.Width == 0 {
        return None;
    }
    let bytes = d.Width as usize;
    let st = state as u32 & PC_RESOURCE_STATES;
    let heap_props = (props != 0 && crate::gpu::shader::readable(props, std::mem::size_of::<D3D12_HEAP_PROPERTIES>()))
        .then(|| std::ptr::read_unaligned(props as *const D3D12_HEAP_PROPERTIES));
    let rounded = (bytes + 0xFFFF) & !0xFFFF;
    let prefer_coherent = heap_props.as_ref().is_some_and(|hp| is_coherent_readback_buffer(hp, st, d.Flags, bytes))
        && has_coherent_headroom(u, rounded);
    let readback_on_null_map = heap_props.as_ref().is_some_and(|hp| is_cpu_readback_buffer(hp, st, d.Flags));
    // Non-readback (and general CUSTOM+WRITE_BACK) committed console buffers live in sparse VRAM
    // windows; small genuine readback buffers prefer coherent committed RAM (spilling to sparse if full).
    if !prefer_coherent && u.has_sparse() {
        if let Some(hr) = place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv) {
            return Some(hr);
        }
    }
    let Some(p) = u.alloc(rounded, 0x1_0000).map(|a| a as usize) else {
        if prefer_coherent && u.has_sparse() {
            return place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv);
        }
        return None;
    };
    // the heap from OpenExistingHeapFromAddress is cross-adapter: row-major + sanitized flags & state
    let k = u.window_of(p)?;
    let (d, st) = sanitize_coherent_buffer(&d, state, !u.is_driver_window(k));
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    let hr = dev_create_placed_resource(this, heap, p - u.window(k).0, &d as *const _ as usize, st as usize, clear, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console buffer in UMA: CreatePlacedResource {hr:#x} – keeping a separate resource (the shader will not see it)");
        }
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
    // Map must hand out the same memory the shaders read (checked once)
    static CHECKED: AtomicU64 = AtomicU64::new(0);
    if CHECKED.fetch_add(1, Ordering::Relaxed) == 0 {
        let raw = *ppv;
        if let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) {
            let mut m: *mut c_void = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut m)).is_ok() {
                r.Unmap(0, None);
                if m as usize != p {
                    crate::xlog_warn!("console buffer in UMA: Map returns {m:?}, not {p:#x} – CPU writes will not reach the console address");
                } else {
                    crate::xlog_info!("console buffers in UMA: Map = console address ({p:#x})");
                }
            }
        }
    }
    Some(hr)
}

/// A buffer the title places at its own address (CreatePlacedResourceX): at a placement
/// boundary (64 KiB) in unified memory it is that memory, placed in the window's heap; the
/// title's allocation stays the title's.
pub(super) unsafe fn place_at_title_address(this: *mut c_void, va: usize, desc: &[u8; 0x38], state: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(va)?;
    let (base, size) = u.window(k);
    let mut d: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(desc.as_ptr() as *const D3D12_RESOURCE_DESC);
    let bytes = d.Width as usize;
    // a placed PC buffer starts at 64 KiB: one at another address starts below it and XCL adds
    // the difference where PC methods take an offset into it (`buffer_delta`)
    let start = va & !0xFFFF;
    let delta = va - start;
    if bytes == 0 || va - base + bytes > size {
        return None;
    }
    // the title's own memory (a sparse window): a resource over its video-memory tiles
    if u.is_sparse(va) {
        let (hr, delta) = super::sparse::place_buffer(this, va, &d, state, riid, ppv)?;
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        return Some(hr);
    }
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    if !u.is_driver_window(k) {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    d.Width = (delta + bytes) as u64;
    let hr = dev_create_placed_resource(this, heap, start - base, &d as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
    if delta != 0 {
        BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
    }
    Some(hr)
}

/// buffers placed below their console address: resource → bytes before it
static BUFFER_DELTA: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Where offset 0 of a console buffer is in its PC resource (0 for most).
pub fn buffer_delta(res: usize) -> u64 {
    BUFFER_DELTA.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or(0)
}

#[cfg(test)]
pub(super) fn set_test_buffer_delta(res: usize, delta: Option<u64>) {
    let mut g = BUFFER_DELTA.lock().unwrap();
    match delta {
        Some(d) => {
            g.get_or_insert_with(HashMap::new).insert(res, d);
        }
        None => {
            if let Some(m) = g.as_mut() {
                m.remove(&res);
            }
        }
    }
}

/// resource Release (every caller): a console buffer's unified memory goes with it
/// Map: on the console every resource is CPU-visible memory. One in unified memory that the
/// PC cannot map (a texture XCL keeps in step with it) gives its unified-memory address – what
/// the console gives; the CPU's writes there reach the texture (`title_textures`).
unsafe extern "system" fn res_map(this: *mut c_void, sub: u32, range: *const c_void, data: *mut *mut c_void) -> HRESULT {
    if let Some(backing) = super::external_buffers::get(this as usize) {
        if sub != 0 {
            return crate::util::E_INVALIDARG;
        }
        if let Some(r) = (range as *const D3D12_RANGE).as_ref() {
            let (begin, end) = (r.Begin.min(backing.bytes), r.End.min(backing.bytes));
            if begin < end && !backing.read(begin, end) {
                return crate::util::E_INVALIDARG;
            }
        }
        if !data.is_null() {
            *data = backing.title as *mut c_void;
        }
        UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
        return S_OK;
    }
    // a resource in unified memory: its console address (no PC mapping – the PC's would be
    // the start of the PC resource, before the address for a buffer placed below it)
    if let Some((p, bytes)) = console_buffer(this as usize) {
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(p)) {
            if !range.is_null() {
                let r = *(range as *const D3D12_RANGE);
                if r.End > r.Begin {
                    let begin = r.Begin.min(bytes);
                    let end = r.End.min(bytes);
                    if end > begin {
                        super::sparse::readback_now(p + begin, end - begin);
                    }
                }
            } else if SPARSE_READBACK.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
                super::sparse::readback_now(p, bytes);
            }
        }
        if !data.is_null() {
            *data = p as *mut c_void;
        }
        UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
        return S_OK;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "Map")));
    let hr = f(this, sub, range, data);
    if hr >= 0 {
        return hr;
    }
    match super::title_textures::address(this as usize).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied())) {
        Some(p) => {
            if !data.is_null() {
                *data = p as *mut c_void;
            }
            UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
            S_OK
        }
        None => {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
                crate::xlog_warn!("D3D12.X: Map of a resource not in unified memory → {:#x} ({:?})", hr as u32, d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0)));
            }
            hr
        }
    }
}

/// WriteToSubresource(subresource, box, data, row pitch, depth pitch): the CPU writing texels.
/// Every console resource is CPU-visible memory, a PC texture only in a CPU-visible heap.
unsafe extern "system" fn res_write_to_subresource(this: *mut c_void, sub: u32, dst_box: *const D3D12_BOX, data: *const c_void, row_pitch: u32, depth_pitch: u32) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_BOX, *const c_void, u32, u32) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "WriteToSubresource")));
    let hr = f(this, sub, dst_box, data, row_pitch, depth_pitch);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static FAILED: AtomicU64 = AtomicU64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    if hr < 0 && FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
        let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
        let b = dst_box.as_ref().map(|b| (b.left, b.top, b.right, b.bottom));
        crate::xlog_warn!(
            "D3D12.X: WriteToSubresource({sub}, box {b:?}, row pitch {row_pitch}) of {:?} → {:#x} (call #{n})",
            d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Layout.0, d.Flags.0)),
            hr as u32
        );
    }
    hr
}

/// resources the title placed at its own address that got a PC resource of their own: on the
/// console Map gives that address (the CPU's view of the resource)
static TITLE_ADDRESS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// The title address a resource stands for (buffers in unified memory, placed resources).
pub(super) fn title_address(res: usize) -> Option<usize> {
    console_buffer(res).map(|b| b.0).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()))
}

pub(super) fn note_title_address(res: usize, va: usize) {
    TITLE_ADDRESS.lock().unwrap().get_or_insert_with(HashMap::new).insert(res, va);
}

/// resources whose Map gave their unified-memory address (no PC mapping to undo)
static UNMAPPABLE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe extern "system" fn res_unmap(this: *mut c_void, sub: u32, range: *const c_void) {
    if UNMAPPABLE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void) = std::mem::transmute(orig_fn(this, slot_of(this, "Unmap")));
    f(this, sub, range)
}

/// Resources a Release of which is under way (`res_release`), each with whether that Release
/// was the object's last: then the object is gone and XCL's records of it are being removed.
static RELEASING: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());

/// A Release of a resource under way, from before the original Release until XCL's records of
/// an object that ended with it are gone.
struct Releasing(usize, bool);

impl Releasing {
    fn begin(res: usize) -> Releasing {
        RELEASING.lock().unwrap_or_else(|e| e.into_inner()).push((res, false));
        Releasing(res, false)
    }

    fn was_the_last(&mut self) {
        let mut g = RELEASING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = g.iter_mut().find(|r| **r == (self.0, false)) {
            r.1 = true;
        }
        self.1 = true;
    }
}

impl Drop for Releasing {
    fn drop(&mut self) {
        let mut g = RELEASING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = g.iter().position(|r| *r == (self.0, self.1)) {
            g.swap_remove(i);
        }
    }
}

/// A reference to resource `res` for a table of XCL's that knows it by its address alone and
/// holds no reference to it (`title_textures`, the targets of `vram`): called with that table's
/// lock held and `res` in the table. None: the object is gone – its last Release has returned and
/// XCL's records of it are being removed, the table's among them once the caller lets go of its
/// lock. A Release that has not returned yet may be the last and is waited for.
///
/// Such a table hears of an object's end after its last Release, behind the locks of the tables
/// told before it. In between it names an object that is gone, and a reference taken from the
/// bare address was a call into freed memory (an upload batch read the description of a texture
/// released on another thread).
pub(super) unsafe fn reference(res: usize) -> Option<ID3D12Resource> {
    let started = std::time::Instant::now();
    loop {
        {
            let g = RELEASING.lock().unwrap_or_else(|e| e.into_inner());
            match g.iter().filter(|r| r.0 == res).map(|r| r.1).max() {
                None => {
                    let raw = res as *mut c_void;
                    return ID3D12Resource::from_raw_borrowed(&raw).cloned();
                }
                Some(true) => return None,
                Some(false) => {}
            }
        }
        // (a Release takes none of XCL's table locks; the bound is for one that never returns)
        if started.elapsed().as_millis() > 100 {
            return None;
        }
        std::thread::yield_now();
    }
}

pub(super) unsafe fn try_reference_depth_capture(res: usize) -> Option<ID3D12Resource> {
    let g = RELEASING.try_lock().ok()?;
    if g.iter().any(|r| r.0 == res) { return None }
    ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).cloned()
}

pub(super) fn resource_release_pending(res: usize) -> bool {
    RELEASING.try_lock().map_or(true, |g| g.iter().any(|r| r.0 == res))
}

pub(super) unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
    let pins = super::sparse::buffer_pins(this as usize);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    // A texture kept in step with title memory is forgotten before its last reference goes, while
    // the object is there: its address can be another texture's the moment it is gone, and a
    // forget after that took the new texture's entry. (A reference taken from the table meanwhile
    // makes this Release not the last; the one that is forgets again.)
    let mut forgotten = false;
    if super::title_textures::any() {
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 1));
        add_ref(this);
        if f(this) == 1 {
            super::title_textures::forget(this as usize);
            forgotten = true;
        }
    }
    // From here until the records of an object that ends here are gone, no table gives out a
    // reference to it (`reference`).
    let mut under_way = Releasing::begin(this as usize);
    let n = super::residency::release(this as usize, || f(this));
    if n == 0 {
        under_way.was_the_last();
        image_views::forget_resource(this as usize);
        if super::depth_lineage::enabled() && super::depth_lineage::interested(this as usize) {
            super::console_heap::forget_lineage_uav_resource(this as usize);
        }
        if super::material_gate_views::enabled() {
            super::material_gate_views::forget_resource(this as usize);
        }
        super::depth_capture_views::forget_resource(this as usize);
        if let Some(s) = UNMAPPABLE.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(s) = SPARSE_READBACK.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(m) = BUFFER_DELTA.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        if let Some(m) = TITLE_ADDRESS.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        if let Some(block) = super::external_buffers::forget(this as usize) {
            super::sparse::release_owned_block(block);
        }
        if let Some(pins) = pins {
            super::sparse::forget_buffer(this as usize, pins);
        }
        super::vram::forget(this as usize);
        super::image_writeback::forget(this as usize);
        if !forgotten {
            // (two references given up at once: neither Release saw the last one coming)
            super::title_textures::forget(this as usize);
        }
        super::depth_uav::forget(this as usize);
        super::depth_state::forget(this as usize);
        let gone = UMA_BUFFERS.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)));
        if let Some((p, n, own)) = gone {
            let mut was = UMA_BUFFERS_GONE.lock().unwrap();
            if was.len() == UMA_BUFFERS_GONE_KEPT {
                was.pop_front();
            }
            was.push_back((p, n, own, crate::log::uptime() as f32));
        }
        if let (Some((p, _, true)), Some(u)) = (gone, crate::gpu::uma::get()) {
            if u.is_sparse(p) {
                super::sparse::release_owned_block(p);
            } else {
                u.free(p);
            }
        }
    }
    n
}

/// ID3D12PipelineState #11, past the PC interface: (this, 0, 0, ID3DBlob**) hands back a blob of
/// the console compile's data for the pipeline (a caller reads a size from its first word and
/// goes on without one). A PC pipeline has no console compile: no blob, the out pointer cleared.
unsafe extern "system" fn pso_console_blob_x(_this: *mut c_void, a: usize, b: usize, out: *mut *mut c_void) -> HRESULT {
    if !out.is_null() && crate::gpu::shader::readable(out as usize, 8) {
        out.write_unaligned(std::ptr::null_mut());
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: ID3D12PipelineState #11 ({a:#x}, {b:#x}): no console compile data for a PC pipeline – no blob (E_NOTIMPL)");
    }
    E_NOTIMPL_HR
}

const E_NOTIMPL_HR: HRESULT = 0x8000_4001u32 as i32;

/// The names the title gave its pipeline states (`SetName`), kept for `XCL_TRACE_NAMED_PSO`
static PSO_NAMES: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

/// `XCL_TRACE_NAMED_PSO`: the pipelines whose name holds this text ("all": every named one) are
/// traced – named, built or failed in the background, their first dispatch or indirect call.
fn traced_name(name: &str) -> bool {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| std::env::var("XCL_TRACE_NAMED_PSO").ok()).as_deref().is_some_and(|f| f == "all" || name.contains(f))
}

/// The name of pipeline `pso` when it is traced (`XCL_TRACE_NAMED_PSO`).
/// The name the title gave a pipeline (SetName).
pub fn pso_name(pso: usize) -> Option<String> {
    PSO_NAMES.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn traced_pso(pso: usize) -> Option<String> {
    PSO_NAMES.lock().unwrap().as_ref()?.get(&pso).filter(|n| traced_name(n)).cloned()
}

/// ID3D12Object::SetName of a pipeline state: kept, then passed on.
unsafe extern "system" fn pso_set_name(this: *mut c_void, name: *const u16) -> HRESULT {
    let readable = |i: usize| crate::gpu::shader::readable(name as usize + 2 * i, 2);
    if !name.is_null() {
        let len = (0..256).take_while(|&i| readable(i) && *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        if traced_name(&text) {
            crate::xlog_info!(
                "named pipeline '{text}' = {this:p}: translated compute {}, building in the background {}",
                is_translated_compute(this as usize),
                super::pso_async::is_pending(this as usize)
            );
        }
        super::title_gta6::named(this as usize, &text);
        super::pass_chain::named(this as usize, &text);
        super::fsr4_standin::named(this as usize, &text);
        PSO_NAMES.lock().unwrap().get_or_insert_with(HashMap::new).insert(this as usize, text);
    }
    let f: unsafe extern "system" fn(*mut c_void, *const u16) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "SetName")));
    f(this, name)
}

/// A stand-in pipeline released for good takes its background-built real pipeline with it.
unsafe extern "system" fn pso_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        forget_pso(this as usize);
        super::pass_chain::gone(this as usize);
        super::fsr4_standin::gone(this as usize);
        super::title_gta6_ambient::gone(this as usize);
        super::pso_async::release(super::pso_async::released(this as usize));
    }
    n
}

/// pipelines made with stand-in shaders (`dummy`): nothing to draw or dispatch with them
static STANDINS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// `hr`, noting the pipeline it made with stand-in shaders (`is_standin_pso`).
unsafe fn note_standin(hr: HRESULT, ppv: *mut *mut c_void) -> HRESULT {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        STANDINS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
    hr
}

/// Whether `pso` was made with stand-in shaders: a pipeline still being built behind it, or one
/// whose translation or build failed or the driver refused (`XCL_SKIP_FAILED_DRAWS`).
pub fn is_standin_pso(pso: usize) -> bool {
    pso != 0 && STANDINS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}
#[cfg(test)]
mod standin_tests {
    use super::*;

    #[test]
    fn a_pipeline_made_with_stand_in_shaders_is_known_until_it_is_gone() {
        let mut made = 0x7b5a_0040usize as *mut c_void;
        assert!(!is_standin_pso(made as usize));
        assert_eq!(unsafe { note_standin(E_INVALIDARG, &mut made) }, E_INVALIDARG);
        assert!(!is_standin_pso(made as usize), "a failed stand-in is not noted");
        assert_eq!(unsafe { note_standin(0, &mut made) }, 0);
        assert!(is_standin_pso(made as usize));
        assert!(!is_standin_pso(0));
        forget_pso(made as usize);
        assert!(!is_standin_pso(made as usize), "a later pipeline at the same address is not a stand-in");
    }
}


/// A pipeline state is gone: drop what XCL keeps by its address (a later pipeline may get the
/// same address and must not inherit its root signature or compute pass).
/// compute pipelines that run a translated console shader
static TRANSLATED_COMPUTE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// graphics pipelines built from translated console shaders
static TRANSLATED_GRAPHICS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` is a graphics pipeline of translated console shaders.
/// `XCL_TRACE_FRAME`: the console VS and PS (their dump names' hashes) of a translated PSO.
static PSO_SHADERS: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

/// translated pipeline → its input layout as text (`XCL_TRACE_FRAME`)
static PSO_LAYOUTS: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

pub fn pso_layout(pso: usize) -> Option<String> {
    PSO_LAYOUTS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

static DRAW_GATE_SHADERS: Mutex<Option<HashMap<usize, [u64; 6]>>> = Mutex::new(None);

static CAPTURE_DEPTH_PSOS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(super) fn capture_depth_equal_pso(pso: usize) -> bool {
    CAPTURE_DEPTH_PSOS.lock().unwrap().contains(&pso)
}

pub(super) fn draw_gate_shaders(pso: usize) -> Option<[u64; 6]> {
    DRAW_GATE_SHADERS.lock().unwrap().as_ref()?.get(&pso).copied()
}

fn note_draw_gate_shaders(pso: usize, stages: &[Option<&[u8]>]) {
    if !super::bind::draw_gate_hashes() && !super::depth_lineage::enabled() {
        return;
    }
    let mut hashes = [0; 6];
    for (dst, bytes) in hashes.iter_mut().zip(stages) {
        *dst = bytes.map_or(0, crate::gpu::shader::fnv64);
    }
    DRAW_GATE_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, hashes);
}

pub fn pso_shaders(pso: usize) -> Option<(u64, u64)> {
    PSO_SHADERS.lock().unwrap().as_ref()?.get(&pso).copied()
}

pub fn is_translated_graphics(pso: usize) -> bool {
    pso != 0 && TRANSLATED_GRAPHICS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// graphics pipelines given the generated rect list GS (`rect_list_gs`)
static RECT_LIST_GS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` (as `is_translated_graphics` takes it: a stand-in resolved) draws rect lists
/// whole with the generated GS. Its draws must not be expanded to two triangles a rect as well:
/// each half is three corners of the rect and the GS would complete both.
pub fn has_rect_list_gs(pso: usize) -> bool {
    pso != 0 && RECT_LIST_GS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// Note `pso`, made from `copy`, if it kept the generated rect list GS (`rect_gs`; a pipeline
/// made again without it, `retry_without_rect_gs`, did not).
fn note_rect_list_gs(pso: usize, copy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool) {
    if pso != 0 && rect_gs && !copy.GS.pShaderBytecode.is_null() {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

/// Whether `pso` runs a translated console compute shader.
pub fn is_translated_compute(pso: usize) -> bool {
    pso != 0 && TRANSLATED_COMPUTE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

#[cfg(test)]
pub(super) fn register_translated_compute_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_rect_list_gs_for_test(pso: usize) {
    if pso != 0 {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_translated_graphics_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn forget_pso_for_test(pso: usize) {
    forget_pso(pso);
}

fn forget_pso(pso: usize) {
    if super::depth_capture_views::enabled() {
        CAPTURE_DEPTH_PSOS.lock().unwrap().retain(|&key| key != pso);
    }
    super::markers::pipeline_gone(pso);
    if super::bind::draw_gate_hashes() || super::depth_lineage::enabled() {
        if let Some(m) = DRAW_GATE_SHADERS.lock().unwrap().as_mut() {
            m.remove(&pso);
        }
    }
    if let Some(m) = UNTRANSLATED.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(s) = STANDINS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    super::bind::forget_failed_skips(pso);
    if let Some(m) = PSO_NAMES.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(s) = TRANSLATED_COMPUTE.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = TRANSLATED_GRAPHICS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = RECT_LIST_GS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(m) = PSO_RS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(m) = PSO_USER_SGPRS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    super::dispatch_timing::forget(pso);
    super::device_x::forget_base(pso);
    if let Some(s) = NATIVE_MESH.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    // the compute pass goes with the last user of its entry (`Drop for TwoPassPso`)
    let gone = TWO_PASS.lock().unwrap().as_mut().and_then(|m| m.remove(&pso));
    drop(gone);
}

/// resource #11 GetGPUVirtualAddress for console code
unsafe extern "system" fn res_gpu_va_x(this: *mut c_void) -> u64 {
    if let Some((p, _)) = console_buffer(this as usize) {
        return p as u64;
    }
    if let Some(p) = super::title_textures::gpu_address(this) {
        return p as u64;
    }
    let f: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(orig_fn(this, 11));
    // a texture has no GPU address on the PC (0) – on the console every resource has one, and
    // titles use a texture's as the CPU's pointer to it (a retail game clears a 256x256 R32F one
    // through it: with 0 it wrote 256 KB at address 0): the title's address it was placed at
    match f(this) {
        0 => {
            let placed = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied());
            placed.or_else(|| texture_memory(this)).unwrap_or(0) as u64
        }
        va => va,
    }
}

const TEXTURE_ADDRESS_PAGE: u64 = 0x1_0000;
const CONSOLE_BC_FOOTPRINT: windows::core::GUID = windows::core::GUID::from_u128(0x049475e9_fa5f_4272_a86b_f244608d62df);

fn console_bc_footprint(desc: &[u8; 0x38]) -> Option<u64> {
    let format = u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap());
    if !crate::hle::xg::element(format).is_some_and(|(_, side)| side == 4) {
        return None;
    }
    let bytes = crate::hle::xg::Texture::new(desc).ok()?.size();
    (bytes != 0).then_some(bytes)
}

fn texture_address_reserve_size(payload: u64) -> Option<u64> {
    let padded = payload.checked_add(TEXTURE_ADDRESS_PAGE)?;
    (padded <= 1 << 30).then(|| padded.next_multiple_of(TEXTURE_ADDRESS_PAGE))
}

/// Memory for a committed texture's console address, with pages committed on demand.
/// Writes through this address are not uploaded to the texture.
unsafe fn texture_memory(res: *mut c_void) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap();
    if let Some(p) = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).copied()) {
        return Some(p);
    }
    let resource = ID3D12Resource::from_raw_borrowed(&res)?;
    let d = resource.GetDesc();
    if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let mut footprint = 0u64;
    let mut len = size_of::<u64>() as u32;
    let has_footprint = crate::hle::xg::element(d.Format.0 as u32).is_some_and(|(_, side)| side == 4)
        && resource.GetPrivateData(&CONSOLE_BC_FOOTPRINT, &mut len, Some((&mut footprint as *mut u64).cast())).is_ok()
        && len == size_of::<u64>() as u32
        && footprint != 0;
    if !has_footprint {
        let element_bytes = crate::gpu::desc::gfx10_format(d.Format).map_or(16, |f| crate::gpu::desc::format_bytes(f).max(1) as u64);
        let texels = d.Width.checked_mul(d.Height as u64)?.checked_mul(d.DepthOrArraySize.max(1) as u64)?;
        footprint = texels.checked_mul(element_bytes)?.checked_mul(4)? / 3;
    }
    let bytes = texture_address_reserve_size(footprint)?;
    let p = VirtualAlloc(std::ptr::null(), bytes as usize, MEM_RESERVE, PAGE_READWRITE) as usize;
    if p == 0 {
        return None;
    }
    crate::memmap::reserved(p, bytes as usize);
    for page in (p..p + bytes as usize).step_by(TEXTURE_ADDRESS_PAGE as usize) {
        crate::hle::kernelx::lazy::mark(page, None);
    }
    note_title_address(res as usize, p);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: GetGPUVirtualAddress of a committed {}x{} texture (format {}, flags {:#x}): {bytes:#x} bytes at {p:#x} for the title to write through (not uploaded)",
            d.Width,
            d.Height,
            d.Format.0,
            d.Flags.0
        );
    }
    Some(p)
}

fn note_buffer_view() {
    if NON_UMA_BUFFER_VIEWS.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("buffer view in the console heap: the buffer is not in UMA – the shader gets an empty V# (num_records 0)");
    }
}

/// #17 CreateConstantBufferView: a V# of float4 rows at the buffer's console address
/// (`desc::cbv_words`; unified memory, anything else gets an empty V#). No PC CBV: translated
/// shaders read buffers through `mem`.
unsafe extern "system" fn dev_create_cbv_x(this: *mut c_void, desc: *const D3D12_CONSTANT_BUFFER_VIEW_DESC, cpu: u64) {
    let Some(slot) = super::console_heap::lookup(cpu).filter(|s| !s.sampler) else {
        let f: unsafe extern "system" fn(*mut c_void, *const D3D12_CONSTANT_BUFFER_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 17));
        return f(this, desc, cpu);
    };
    let (va, size) = desc.as_ref().map_or((0, 0), |d| (d.BufferLocation, d.SizeInBytes));
    let in_uma = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    if !in_uma {
        note_buffer_view();
    }
    let (va, size) = if in_uma { (va, size) } else { (0, 0) };
    super::console_heap::write(&slot, &crate::gpu::desc::cbv_words(va, size));
}

/// #22 CreateSampler: the PC sampler at the slot's index, the S# in XCL ABI form (word 3
/// [11:0] = sampler heap index; the state lives in the PC sampler).
pub(super) unsafe extern "system" fn dev_create_sampler_x(this: *mut c_void, desc: *const D3D12_SAMPLER_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *const D3D12_SAMPLER_DESC, u64) = std::mem::transmute(orig_fn(this, 22));
    let Some(slot) = super::console_heap::slot(cpu).filter(|s| s.sampler) else { return f(this, desc, cpu) };
    f(this, desc, slot.pc_srv);
    super::console_heap::write(&slot, &[0, 0, 0, slot.index & 0xFFF]);
}

/// Copy `n` console slots from `src` to `dst` (both in console heaps of one kind): the PC
/// descriptors (2 per view slot) and the slots. False: not two console ranges.
unsafe fn copy_console_slots(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) -> bool {
    use super::console_heap::{copy_words, has_pc_views, lookup, slot, SAMPLER_SLOT, SLOT};
    let (Some(d0), Some(s0)) = (lookup(dst), lookup(src)) else { return false };
    if d0.sampler != s0.sampler {
        return false;
    }
    let ty = pc_heap_type(ty);
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, if d0.sampler { n } else { 2 * n }, d0.pc_srv, s0.pc_srv, ty);
    let step = if d0.sampler { SAMPLER_SLOT } else { SLOT } as u64;
    for k in 0..n as u64 {
        let Some(s) = lookup(src + k * step) else { continue };
        let d = if s.sampler || has_pc_views(&s) { slot(dst + k * step) } else { lookup(dst + k * step) };
        if let Some(d) = d {
            copy_words(&s, &d);
        }
    }
    if !d0.sampler {
        super::console_heap::publish_range(&d0, n);
    }
    true
}

/// `XCL_TRACE_FRAME`: the device's descriptor copies (first few, and a count)
fn note_copy_call(what: &str, n: u32, dst: u64, src: u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if super::present::trace_frame() && (k < 4 || k.is_power_of_two()) {
        crate::xlog_info!("frame trace: {what} #{k}: {n} from {src:#x} to {dst:#x}");
    }
}

static MIXED_COPIES: AtomicU64 = AtomicU64::new(0);

fn note_mixed_copy() {
    if MIXED_COPIES.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("CopyDescriptors (console): range outside the XCL console heaps – copying on the PC only");
    }
}

/// #24 CopyDescriptorsSimple(n, dst, src, type)
unsafe extern "system" fn dev_copy_descriptors_simple_x(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) {
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptorsSimple", n, dst, src);
    if copy_console_slots(this, n, dst, src, ty) {
        return;
    }
    if super::console_heap::lookup(dst).is_some() || super::console_heap::lookup(src).is_some() {
        note_mixed_copy();
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, n, dst, src, ty);
    image_views::copied(this, ty, 1, &dst, &n, 1, &src, &n);
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_DSV.0 as u32 && super::depth_capture_views::enabled() {
        let step = dev_increment_size_x(this, ty) as usize;
        super::depth_capture_views::copy_ranges(step, (&[dst], &n), (&[src], &n));
    }
}

/// #23 CopyDescriptors(dst ranges, src ranges, type): range by range over console slots.
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_copy_descriptors_x(this: *mut c_void, nd: u32, dsts: *const u64, dsizes: *const u32, ns: u32, srcs: *const u64, ssizes: *const u32, ty: u32) {
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptors", nd, if dsts.is_null() { 0 } else { *dsts }, if srcs.is_null() { 0 } else { *srcs });
    use super::console_heap::{lookup, SAMPLER_SLOT, SLOT};
    let console = !dsts.is_null() && !srcs.is_null() && nd > 0 && lookup(*dsts).is_some();
    if !console {
        if !srcs.is_null() && ns > 0 && lookup(*srcs).is_some() {
            note_mixed_copy();
            return;
        }
        type F = unsafe extern "system" fn(*mut c_void, u32, *const u64, *const u32, u32, *const u64, *const u32, u32);
        let f: F = std::mem::transmute(orig_fn(this, 23));
        f(this, nd, dsts, dsizes, ns, srcs, ssizes, ty);
        image_views::copied(this, ty, nd, dsts, dsizes, ns, srcs, ssizes);
        if ty == D3D12_DESCRIPTOR_HEAP_TYPE_DSV.0 as u32 && super::depth_capture_views::enabled() && nd != 0 && !dsts.is_null() {
            let dst = std::slice::from_raw_parts(dsts, nd as usize);
            let src = if ns == 0 || srcs.is_null() { &[] } else { std::slice::from_raw_parts(srcs, ns as usize) };
            let step = dev_increment_size_x(this, ty) as usize;
            super::depth_capture_views::copy_ranges(step, (dst, dsizes), (src, ssizes));
        }
        return;
    }
    // every descriptor as (dst, src) handle, ranges walked in step
    let step = if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 { SAMPLER_SLOT } else { SLOT } as u64;
    let expand = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<u64> {
        (0..n as usize).flat_map(|r| {
            let (h, len) = (*starts.add(r), if sizes.is_null() { 1 } else { *sizes.add(r) });
            (0..len as u64).map(move |k| h + k * step)
        })
        .collect()
    };
    let (d, s) = (expand(dsts, dsizes, nd), expand(srcs, ssizes, ns));
    for (a, b) in d.iter().zip(s.iter()) {
        if !copy_console_slots(this, 1, *a, *b, ty) {
            note_mixed_copy();
        }
    }
}

/// descriptor heap Release (every caller): a console heap's slots go with it
unsafe extern "system" fn heap_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let depth_heap = super::depth_capture_views::heap_generation(this as usize);
    let attachment_heap = image_views::heap_id(this as usize);
    let n = f(this);
    if n == 0 {
        if let Some((first, generation)) = depth_heap {
            super::depth_capture_views::forget_heap(first, generation);
        }
        if let Some(id) = attachment_heap { image_views::forget_heap(id); }
        super::console_heap::unregister(this as usize);
    }
    n
}

/// #21 CreateDepthStencilView: the console's view formats as the PC's (`pc_view_format`).
unsafe extern "system" fn dev_create_dsv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_DEPTH_STENCIL_VIEW_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 21));
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(c.Format.0 as u32) as i32);
    }
    // no description: the resource's format – which a depth buffer made typeless
    // (`device_x::pc_resource_desc`) is not one a view can have
    if copy.is_none() {
        copy = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| default_dsv(&r.GetDesc()));
    }
    f(this, res, copy.as_ref().map_or(desc, |c| c as *const _), cpu);
    image_views::dsv(this as usize, cpu as usize, res, copy.as_ref());
    if super::depth_capture_views::enabled() {
        super::depth_capture_views::record_view(cpu as usize, res, copy.as_ref());
    }
}

/// The view a DSV without a description has, for a depth buffer `pc_resource_desc` made
/// typeless (None: the resource's own format is a depth one).
fn default_dsv(d: &D3D12_RESOURCE_DESC) -> Option<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    let format = match d.Format.0 {
        19 => 20,
        39 => 40,
        44 => 45,
        53 => 55,
        _ => return None,
    };
    let mut v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT(format), ..Default::default() };
    if d.DepthOrArraySize > 1 {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { MipSlice: 0, FirstArraySlice: 0, ArraySize: d.DepthOrArraySize as u32 };
    } else {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    }
    Some(v)
}

/// A shader view `v` (None: the resource's default one) of `rd` with the PC's format names
/// (`device_x::pc_shader_view_format`) – None: as it is.
fn pc_srv(rd: &D3D12_RESOURCE_DESC, v: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    if rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let rf = rd.Format.0 as u32;
    let mut out = match v {
        Some(v) => *v,
        None => {
            // only a typeless depth buffer needs a description of its own
            if !matches!(rf, 19 | 39 | 44 | 53) {
                return None;
            }
            let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
            if rd.DepthOrArraySize > 1 {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: u32::MAX, ArraySize: rd.DepthOrArraySize as u32, ..Default::default() };
            } else {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                v.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: u32::MAX, ..Default::default() };
            }
            v
        }
    };
    if out.ViewDimension == D3D12_SRV_DIMENSION_BUFFER || out.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE {
        return None;
    }
    let f = super::device_x::pc_shader_view_format(rf, out.Format.0 as u32);
    let stencil = super::device_x::stencil_view(rf, f);
    if v.is_some() && f == out.Format.0 as u32 && !stencil {
        return None;
    }
    out.Format = DXGI_FORMAT(f as i32);
    // the PC reads stencil from plane 1; the console's view of it is plane 0 in a stencil format
    if stencil {
        if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D {
            out.Anonymous.Texture2D.PlaneSlice = 1;
        } else if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
            out.Anonymous.Texture2DArray.PlaneSlice = 1;
        }
    }
    Some(out)
}

unsafe fn trace_native_srv(
    slot: &super::console_heap::Slot, res: *mut c_void, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
    owner: *mut c_void, requested: Option<super::material_gate_views::GuestView>,
) -> Option<super::material_gate_views::NativeView> {
    let trace = super::present::native_view_trace_configured();
    let material = super::material_gate_views::enabled() && !slot.sampler && is_adopted(owner);
    if !trace && !material {
        return None;
    }
    let rd = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
    if trace {
        let (format, layout, flags) = rd.map_or((-1, -1, 0), |rd| (rd.Format.0, rd.Layout.0, rd.Flags.0));
        let (view, dim) = desc.map_or_else(
            || (format, rd.map_or(0, |rd| resource_dim_to_view(&rd))),
            |d| (d.Format.0, d.ViewDimension.0),
        );
        super::console_heap::note_native_srv(slot, Some((res as usize, view, dim, format, layout, flags)));
    }
    if !material {
        return None;
    }
    super::material_gate_views::NativeView::new(owner as usize, res as usize, &rd?, desc, requested)
}

unsafe extern "system" fn dev_create_srv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_SHADER_RESOURCE_VIEW_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
    let requested = desc.as_ref();
    let mut copy = requested.copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(c.Format.0 as u32) as i32);
    }
    // a depth buffer's view in the formats the PC has for it (a D24S8 format itself, R16_UNORM of
    // a D16S8 buffer …: an invalid call, the device removed)
    if let Some(v) = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| pc_srv(&r.GetDesc(), copy.as_ref())) {
        copy = Some(v);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    let Some(slot) = super::console_heap::slot(cpu) else { return f(this, res, desc, cpu) };
    if super::console_heap::native_uavs_kept() {
        super::console_heap::note_native_uav(&slot, None);
    }
    // Raytracing acceleration structure SRV (pResource == NULL, Location == console GPU VA):
    // translated shaders read BVH nodes from UMA (`bvh_intersect`), never from a PC DXR SRV,
    // so write the RDNA2 BVH descriptor directly and skip forwarding the console VA to PC.
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) {
        let va = v.Anonymous.RaytracingAccelerationStructure.Location;
        let in_uma = va != 0 && crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
        if in_uma {
            let w = crate::gpu::desc::bvh_words(va, true);
            return super::console_heap::write(&slot, &[w[0], w[1], w[2], w[3], 0, 0, 0, 0]);
        }
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: CreateShaderResourceView(RAYTRACING_ACCELERATION_STRUCTURE) at {va:#x} is not in unified memory – writing empty BVH descriptor");
        }
        return super::console_heap::write(&slot, &[0; 8]);
    }
    // translated shaders declare every cube T# as a TextureCubeArray: a title's TEXTURECUBE
    // view becomes the cube array of that one cube (a dimension mismatch is undefined in D3D12)
    let cube = desc.as_ref().and_then(cube_as_cube_array);
    let desc = cube.as_ref().map_or(desc, |v| v as *const _);
    f(this, res, desc, slot.pc_srv);
    let native = trace_native_srv(&slot, res, desc.as_ref(), res, requested.map(|v| super::material_gate_views::GuestView::Srv(v.Format.0)));
    super::console_heap::publish(&slot, false);
    super::console_heap::note_resource(&slot, res as usize);
    if super::present::trace_frame() && super::title_textures::address(res as usize).is_some() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 24 {
            let rd = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
            let v = desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0, v.Shader4ComponentMapping, v.Anonymous.Texture2D.MostDetailedMip, v.Anonymous.Texture2D.MipLevels, v.Anonymous.Texture2D.ResourceMinLODClamp));
            crate::xlog_info!(
                "frame trace: SRV of title texture {res:p} {:?}: view (format, dimension, mapping, mip, mips, clamp) {v:x?}",
                rd.map(|d| (d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0))
            );
        }
    }
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else {
        return super::console_heap::write(&slot, &[0; 8]);
    };
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        return super::console_heap::write_observed(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw),None,None,super::console_heap::vsharp::Writer::Api { kind: "resource-srv", input: res as u64, first: b.FirstElement, count: b.NumElements, stride: b.StructureByteStride, raw: raw });
    }
    let (format, dim, array, mips) = match desc.as_ref() {
        Some(v) => {
            let a = &v.Anonymous;
            let (array, mips) = match v.ViewDimension.0 {
                2 => (1, a.Texture1D.MipLevels),
                3 => (a.Texture1DArray.ArraySize, a.Texture1DArray.MipLevels),
                4 => (1, a.Texture2D.MipLevels),
                5 => (a.Texture2DArray.ArraySize, a.Texture2DArray.MipLevels),
                7 => (a.Texture2DMSArray.ArraySize, 1),
                8 => (1, a.Texture3D.MipLevels),
                9 => (6, a.TextureCube.MipLevels),
                10 => (a.TextureCubeArray.NumCubes.max(1) * 6, a.TextureCubeArray.MipLevels),
                _ => (1, 1),
            };
            (v.Format, v.ViewDimension.0, array, mips)
        }
        None => {
            let d = r.GetDesc();
            (d.Format, resource_dim_to_view(&d), d.DepthOrArraySize as u32, d.MipLevels as u32)
        }
    };
    match view_shape(r, format, dim, array, mips) {
        Some((fmt, ty, w, h, d, m)) => super::console_heap::write_view(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, w, h, d, m), None, native),
        None => {
            note_buffer_view();
            super::console_heap::write(&slot, &[0; 8]);
        }
    }
}

/// A TEXTURECUBE view as the TEXTURECUBEARRAY of its one cube (None: any other view).
fn cube_as_cube_array(v: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    if v.ViewDimension != D3D12_SRV_DIMENSION_TEXTURECUBE {
        return None;
    }
    // SAFETY: the union member the view dimension names
    let c = unsafe { v.Anonymous.TextureCube };
    Some(D3D12_SHADER_RESOURCE_VIEW_DESC {
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                MostDetailedMip: c.MostDetailedMip,
                MipLevels: c.MipLevels,
                First2DArrayFace: 0,
                NumCubes: 1,
                ResourceMinLODClamp: c.ResourceMinLODClamp,
            },
        },
        ..*v
    })
}

/// The texture a view by address describes: `desc` (a resource desc) placed at `va`, made once
/// per address and desc – titles make views of the same memory again and again.
unsafe fn resource_at(this: *mut c_void, desc: *const u8, va: u64) -> *mut c_void {
    type Key = (u64, [u8; 0x38]);
    static MADE: Mutex<Option<HashMap<Key, usize>>> = Mutex::new(None);
    if desc.is_null() || va == 0 || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return std::ptr::null_mut();
    }
    // the PC part of the desc; the flags (@0x30) are its last dword, padding follows
    let mut x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    x[0x34..].fill(0);
    if let Some(&r) = MADE.lock().unwrap().get_or_insert_with(HashMap::new).get(&(va, x)) {
        return r as *mut c_void;
    }
    let mut r = std::ptr::null_mut();
    let hr = super::device_x::create_placed_resource_x(this, va, x.as_ptr(), 0, std::ptr::null(), &ID3D12Resource::IID as *const _ as *const GUID, &mut r);
    if hr < 0 || r.is_null() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_warn!("D3D12.X: view at {va:#x}: the resource cannot be made ({hr:#x}); desc {x:02x?}");
        }
        r = std::ptr::null_mut();
    }
    MADE.lock().unwrap().get_or_insert_with(HashMap::new).insert((va, x), r as usize);
    r
}

/// device #70 (a retail game): a shader resource view of memory, without a resource object:
/// (resource desc, { the PC's SRV desc; GPU address @0x28; … }, destination).
unsafe extern "system" fn dev_create_srv_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_SHADER_RESOURCE_VIEW_DESC).read_unaligned();
    if v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
        let Some(slot) = super::console_heap::lookup(cpu) else { return };
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write_observed(&slot, &words, None,None,super::console_heap::vsharp::Writer::Api { kind: "by-address-srv", input: va, first: b.FirstElement, count: b.NumElements, stride: b.StructureByteStride, raw: raw });
    }
    dev_create_srv_x(this, resource_at(this, rdesc, va), &v, cpu);
}

/// device #71: the same for an unordered access view ({ the PC's UAV desc; GPU address @0x28;
/// the counter's address @0x30; … }).
unsafe extern "system" fn dev_create_uav_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_UNORDERED_ACCESS_VIEW_DESC).read_unaligned();
    if v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
        let Some(slot) = super::console_heap::lookup(cpu) else { return };
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write_observed(&slot, &words, None,None,super::console_heap::vsharp::Writer::Api { kind: "by-address-uav", input: va, first: b.FirstElement, count: b.NumElements, stride: b.StructureByteStride, raw: raw });
    }
    dev_create_uav_x(this, resource_at(this, rdesc, va), std::ptr::null_mut(), &v, cpu);
}

/// V# of `n` elements from element `first` of the buffer at `va`.
fn buffer_words_at(va: u64, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = match (raw, stride) {
        (true, _) => (0, 4),
        (false, 0) => {
            let e = crate::gpu::desc::format_bytes(fmt);
            (e, e)
        }
        (false, s) => (s, s),
    };
    let records = if stride == 0 { n.saturating_mul(4) } else { n };
    crate::gpu::desc::buffer_words(va + first * elem as u64, stride, records, fmt)
}

fn null_texture_uav(dimension: D3D12_UAV_DIMENSION) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let (dimension, view) = match dimension {
        D3D12_UAV_DIMENSION_TEXTURE1D | D3D12_UAV_DIMENSION_TEXTURE2D => (dimension, D3D12_UNORDERED_ACCESS_VIEW_DESC_0::default()),
        D3D12_UAV_DIMENSION_TEXTURE1DARRAY => (dimension, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture1DArray: D3D12_TEX1D_ARRAY_UAV { ArraySize: 1, ..Default::default() },
        }),
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => (dimension, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV { ArraySize: 1, ..Default::default() },
        }),
        D3D12_UAV_DIMENSION_TEXTURE3D => (dimension, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_UAV { WSize: 1, ..Default::default() },
        }),
        _ => (D3D12_UAV_DIMENSION_TEXTURE2D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0::default()),
    };
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT,
        ViewDimension: dimension,
        Anonymous: view,
    }
}

fn uav_companion_srv(rd: Option<&D3D12_RESOURCE_DESC>, uav: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
    let dim = uav.map(|v| v.ViewDimension).or_else(|| rd.map(|d| D3D12_UAV_DIMENSION(resource_dim_to_view(d))))?;
    let layers = rd.map_or(1, |d| d.DepthOrArraySize as u32);
    let (dimension, view) = unsafe {
        match dim {
            D3D12_UAV_DIMENSION_BUFFER => {
                let b = uav?.Anonymous.Buffer;
                let flags = if b.Flags.contains(D3D12_BUFFER_UAV_FLAG_RAW) { D3D12_BUFFER_SRV_FLAG_RAW } else { D3D12_BUFFER_SRV_FLAG_NONE };
                (D3D12_SRV_DIMENSION_BUFFER, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV { FirstElement: b.FirstElement, NumElements: b.NumElements, StructureByteStride: b.StructureByteStride, Flags: flags },
                })
            }
            D3D12_UAV_DIMENSION_TEXTURE1D => (D3D12_SRV_DIMENSION_TEXTURE1D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_SRV { MostDetailedMip: uav.map_or(0, |v| v.Anonymous.Texture1D.MipSlice), MipLevels: 1, ..Default::default() },
            }),
            D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
                let a = uav.map_or(D3D12_TEX1D_ARRAY_UAV { ArraySize: layers, ..Default::default() }, |v| v.Anonymous.Texture1DArray);
                (D3D12_SRV_DIMENSION_TEXTURE1DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1DArray: D3D12_TEX1D_ARRAY_SRV { MostDetailedMip: a.MipSlice, MipLevels: 1, FirstArraySlice: a.FirstArraySlice, ArraySize: a.ArraySize, ..Default::default() },
                })
            }
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                let a = uav.map_or_else(D3D12_TEX2D_UAV::default, |v| v.Anonymous.Texture2D);
                (D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: a.MipSlice, MipLevels: 1, PlaneSlice: a.PlaneSlice, ..Default::default() },
                })
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                let a = uav.map_or(D3D12_TEX2D_ARRAY_UAV { ArraySize: layers, ..Default::default() }, |v| v.Anonymous.Texture2DArray);
                (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: a.MipSlice, MipLevels: 1, FirstArraySlice: a.FirstArraySlice, ArraySize: a.ArraySize, PlaneSlice: a.PlaneSlice, ..Default::default() },
                })
            }
            D3D12_UAV_DIMENSION_TEXTURE3D => (D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: uav.map_or(0, |v| v.Anonymous.Texture3D.MipSlice), MipLevels: 1, ..Default::default() },
            }),
            _ => return None,
        }
    };
    let format = uav.map(|v| v.Format).filter(|f| *f != DXGI_FORMAT_UNKNOWN).or_else(|| rd.map(|d| d.Format))?;
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: dimension,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: view,
    };
    Some(rd.and_then(|d| pc_srv(d, Some(&srv))).unwrap_or(srv))
}

pub(super) unsafe extern "system" fn dev_create_uav_x(this: *mut c_void, res: *mut c_void, counter: *mut c_void, desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, cpu: u64) {
    // the view's format as the PC's for the resource (B5G5R5A1 of a target made R8G8B8A8,
    // `device_x::pc_resource_desc`: an invalid call, the device removed)
    let r = ID3D12Resource::from_raw_borrowed(&res);
    let rd = r.map(|r| r.GetDesc());
    let requested = desc.as_ref();
    let mut copy = requested.copied();
    if let (Some(c), Some(rd)) = (copy.as_mut(), rd.as_ref()) {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rd.Format.0 as u32, c.Format.0 as u32) as i32);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    super::present::note_use(res as usize, super::present::USE_UAV);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 19));
    if super::present::trace_frame() {
        if rd.is_some_and(|d| d.Width >= 1920 && d.Flags.0 & 1 != 0) {
            let s = super::console_heap::slot(cpu);
            crate::xlog_info!("frame trace: UAV of render target {res:p} {:?} at {cpu:#x}: console slot {:?}, view {:?}", rd.map(|d| (d.Width, d.Height, d.Format.0)), s.as_ref().map(|s| (s.index, s.pc_uav, s.at)), desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0)));
        }
    }
    // a resource without ALLOW_UNORDERED_ACCESS (a console depth buffer that also had it: the PC
    // refuses the pair, `device_x::pc_resource_desc`): a null view – a UAV of it is an invalid
    // call, and the device is removed (0x887A0001)
    let no_uav = rd.as_ref().filter(|r| r.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0);
    // a depth buffer's: a view of its shadow texture (`depth_uav`), where there is one
    if no_uav.is_some_and(|rd| rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) {
        super::device_x::watch_depth_first_view(res as usize, "UAV");
    }
    let shadow = no_uav.and_then(|rd| super::depth_uav::view(res, rd, desc.as_ref()));
    if let Some((twin, _, _)) = &shadow { image_views::note_twin(*twin as usize, res as usize); }
    let lineage = if super::depth_lineage::enabled() && super::depth_lineage::interested(res as usize) {
        no_uav.and_then(|rd| super::device_x::lineage_uav(
            res as usize, rd, desc.as_ref(), shadow.as_ref().map(|(p, v, _)| (*p as usize, v.Format.0)),
        ))
    } else {
        None
    };
    let pc_uav = |at: u64| {
        if let Some((texture, view, _)) = shadow.as_ref() {
            f(this, *texture, std::ptr::null_mut(), view, at)
        } else if let Some(rd) = no_uav {
            let dim = desc.as_ref().map_or_else(|| D3D12_UAV_DIMENSION(resource_dim_to_view(rd)), |d| d.ViewDimension);
            let null_desc = null_texture_uav(dim);
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: a UAV of a resource that cannot have one on the PC and has no shadow texture (format {}, {} samples, flags {:#x}) – a null view", rd.Format.0, rd.SampleDesc.Count, rd.Flags.0);
                if super::present::writer_trace_configured() {
                    crate::xlog_warn!("present writer: null UAV resource {res:p} {}x{}, requested view format {:?}, dimension {}", rd.Width, rd.Height, desc.as_ref().map(|d| d.Format.0), dim.0);
                }
            }
            f(this, std::ptr::null_mut(), std::ptr::null_mut(), &null_desc, at)
        } else {
            f(this, res, counter, desc, at)
        }
    };
    let Some(slot) = super::console_heap::slot(cpu) else { return pc_uav(cpu) };
    pc_uav(slot.pc_uav);
    // what the slot's views are of: the title's resource, or a depth buffer's shadow texture
    let (viewed, rd, desc): (*mut c_void, Option<D3D12_RESOURCE_DESC>, *const D3D12_UNORDERED_ACCESS_VIEW_DESC) = match shadow.as_ref() {
        Some((texture, view, texture_desc)) => (*texture, Some(*texture_desc), view as *const _),
        None => (res, rd, desc),
    };
    if super::console_heap::native_uavs_kept() {
        let details = rd.filter(|_| !viewed.is_null() && (shadow.is_some() || no_uav.is_none())).map(|d| {
            let descriptor = desc.as_ref();
            let view = descriptor.map_or((d.Format.0, resource_dim_to_view(&d)), |v| (v.Format.0, v.ViewDimension.0));
            let volume = if view.1 == D3D12_UAV_DIMENSION_TEXTURE3D.0 {
                Some(descriptor.map_or((0, 0, u32::MAX), |v| {
                    let d = v.Anonymous.Texture3D;
                    (d.MipSlice, d.FirstWSlice, d.WSize)
                }))
            } else {
                None
            };
            super::console_heap::NativeUav {
                resource: viewed as usize, view, volume,
                backing: (d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.Layout.0, d.Flags.0),
            }
        });
        super::console_heap::note_native_uav(&slot, details);
    }
    // RDNA images have no SRV / UAV distinction: console shaders may read this slot too.
    let native = if let Some(srv) = uav_companion_srv(rd.as_ref(), desc.as_ref()) {
        let srv_res = if rd.is_some_and(|d| d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) { std::ptr::null_mut() } else { viewed };
        let g: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
        g(this, srv_res, &srv, slot.pc_srv);
        let native = trace_native_srv(&slot, srv_res, Some(&srv), res, requested.map(|v| super::material_gate_views::GuestView::Uav(v.Format.0)));
        super::console_heap::publish_range(&slot, 1);
        native
    } else {
        super::console_heap::publish(&slot, true);
        None
    };
    super::console_heap::note_resource(&slot, res as usize);
    let (Some(r), Some(rd)) = (r, rd) else {
        return super::console_heap::write(&slot, &[0; 8]);
    };
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        return super::console_heap::write_observed(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw),None,None,super::console_heap::vsharp::Writer::Api { kind: "resource-uav", input: res as u64, first: b.FirstElement, count: b.NumElements, stride: b.StructureByteStride, raw: raw });
    }
    // UAV dimensions: 1 buffer, 2 tex1d, 3 tex1d array, 4 tex2d, 5 tex2d array, 8 tex3d
    let (format, dim, array, mip) = match desc.as_ref() {
        Some(v) => {
            let a = &v.Anonymous;
            let (array, mip) = match v.ViewDimension.0 {
                2 => (1, a.Texture1D.MipSlice),
                3 => (a.Texture1DArray.ArraySize, a.Texture1DArray.MipSlice),
                4 => (1, a.Texture2D.MipSlice),
                5 => (a.Texture2DArray.ArraySize, a.Texture2DArray.MipSlice),
                8 => (a.Texture3D.WSize, a.Texture3D.MipSlice),
                _ => (1, 0),
            };
            (v.Format, v.ViewDimension.0, array, mip)
        }
        None => (rd.Format, resource_dim_to_view(&rd), rd.DepthOrArraySize as u32, 0),
    };
    let Some((fmt, ty, w, h, d, _)) = view_shape(r, format, dim, array, 1) else {
        note_buffer_view();
        return super::console_heap::write(&slot, &[0; 8]);
    };
    let (w, h) = ((w >> mip).max(1), (h >> mip).max(1));
    super::console_heap::write_view(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, w, h, d, 1), lineage, native);
}

/// device #127 (a retail game): CreateCommittedResource with a barrier layout in the initial state's place
/// – the arguments of the PC's CreateCommittedResource3 (heap properties, flags, desc, layout,
/// clear value, protected session, castable formats, riid, resource).
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_create_committed_resource_layout_x(
    this: *mut c_void,
    props: usize,
    flags: u32,
    desc: usize,
    layout: u32,
    clear: usize,
    _protected: usize,
    _castable: u32,
    _formats: usize,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    note_newer_device_method(127);
    let state = super::device_x::layout_state(D3D12_BARRIER_LAYOUT(layout as i32)).unwrap_or(D3D12_RESOURCE_STATE_COMMON).0 as usize;
    if flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY.0 as u32 != 0 && desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        return create_display_target(this, props, desc, state, clear, riid, ppv);
    }
    dev_create_committed_resource_x(this, props, flags as usize, desc, state, clear, riid, ppv)
}

/// A display target: its console memory comes from the title's allocation hook, as the console
/// runtime's does (a retail game gives its back buffers one block and asserts that it is used up); on the
/// PC it is a resource of its own.
unsafe fn create_display_target(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let mut clear_buf = D3D12_CLEAR_VALUE::default();
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf, Some((&x, &d))) as usize;
    let hr = dev_create_committed_resource(this, props, 0, d.as_ptr() as usize, state, clear, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        crate::xlog_warn!("D3D12.X: display target: {hr:#x}; desc {x:02x?}");
        return hr;
    }
    let size = crate::hle::xg::Texture::new(&x).map_or(0, |t| t.size()) as usize;
    let at = if size > 0 { crate::hle::kernelx::runtime_alloc(size, crate::hle::kernelx::D3D_DISPLAY_TARGET) } else { None };
    if let Some(p) = at {
        note_title_address(*ppv as usize, p);
    }
    crate::xlog_info!(
        "D3D12.X: display target {}x{} format {}: {size:#x} bytes of the title's memory at {:#x}",
        u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()),
        u32::from_le_bytes(x[0x18..0x1C].try_into().unwrap()),
        u32::from_le_bytes(x[0x20..0x24].try_into().unwrap()),
        at.unwrap_or(0)
    );
    hr
}

unsafe extern "system" fn dev_create_committed_resource_x(this: *mut c_void, props: usize, flags: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if desc != 0 && std::env::var_os("XCL_D3D12_TRACE").is_some() && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        let w = |o: usize| *((desc + o) as *const u32);
        crate::xlog_info!(
            "CreateCommittedResource (console, slot 76): dimension {} {}x{} format {} flags {:#x}; console fields after the PC desc: +52 {:#x} +56 {:#x}",
            w(0),
            *((desc + 16) as *const u64),
            w(24),
            w(32),
            w(48),
            w(52),
            w(56)
        );
        crate::xlog_info!("  desc (dwords): {:x?}", (0..16).map(|k| w(4 * k)).collect::<Vec<_>>());
    }
    // console texture layouts (seen: 0x101, with 3 at +52) are not PC ones: the driver picks
    // its own (UNKNOWN); the console layout matters only where console code reads the memory
    let mut pc_desc = [0u8; 56];
    let console_desc = desc;
    let mut desc = desc;
    if desc != 0 {
        std::ptr::copy_nonoverlapping(desc as *const u8, pc_desc.as_mut_ptr(), 56);
        let layout = u32::from_le_bytes(pc_desc[44..48].try_into().unwrap());
        if layout > 3 {
            pc_desc[44..48].copy_from_slice(&0u32.to_le_bytes());
            pc_desc[52..56].copy_from_slice(&0u32.to_le_bytes());
            desc = pc_desc.as_ptr() as usize;
        }
    }
    if desc != 0 {
        if let Some(hr) = place_console_buffer(this, props, desc, state, clear, riid, ppv) {
            return hr;
        }
        if let Some(hr) = place_console_texture(this, console_desc, state, clear, riid, ppv) {
            return hr;
        }
    }
    // not placed in unified memory (e.g. its coherent window kept for readback heaps): a PC
    // resource of the PC's description – the console's flag bits (0x1000000, 0x300000 …) and
    // alignments made the PC refuse Fortnite's 3264x1836 scene targets (E_INVALIDARG, a crash)
    let console_full = (console_desc != 0 && crate::gpu::shader::readable(console_desc, 0x38))
        .then(|| (console_desc as *const [u8; 0x38]).read_unaligned());
    let pc_full = console_full.as_ref().map(super::device_x::pc_resource_desc);
    let desc = pc_full.as_ref().map_or(desc, |d| d.as_ptr() as usize);
    let mut clear_buf = D3D12_CLEAR_VALUE::default();
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf, console_full.as_ref().zip(pc_full.as_ref())) as usize;
    // console-only state bits (e.g. 0x100000 on a display target) are not the PC's; nor is
    // UNORDERED_ACCESS a state of a depth buffer (`depth_uav`)
    let mut state = state & PC_RESOURCE_STATES as usize;
    if console_full.as_ref().is_some_and(|x| u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()) & 2 != 0) {
        state = super::depth_uav::pc_state(state as u32) as usize;
    }
    let hr = dev_create_committed_resource(this, props, flags, desc, state, clear, riid, ppv);
    if hr >= 0 && props != 0 && desc != 0 && crate::gpu::shader::readable(props, std::mem::size_of::<D3D12_HEAP_PROPERTIES>()) {
        if let Some(dev) = ID3D12Device::from_raw_borrowed(&this) {
            let hp = std::ptr::read_unaligned(props as *const D3D12_HEAP_PROPERTIES);
            let rd = std::ptr::read_unaligned(desc as *const D3D12_RESOURCE_DESC);
            let size = dev.GetResourceAllocationInfo(0, &[rd]).SizeInBytes;
            crate::memdiag::TITLE_COMMITTED[crate::memdiag::heap_class(&hp)].add(if size == u64::MAX { 0 } else { size });
        }
    }
    if hr < 0 {
        let pw = |o: usize| if props == 0 { 0 } else { *((props + o) as *const u32) };
        crate::xlog_warn!(
            "CreateCommittedResource (console): {hr:#x}; heap type {} cpu {} pool {}, heap flags {flags:#x}, state {state:#x}",
            pw(0),
            pw(4),
            pw(8)
        );
        return hr;
    }
    if !ppv.is_null() {
        if let (Some(size), Some(unknown)) = (console_full.as_ref().and_then(console_bc_footprint), windows::core::IUnknown::from_raw_borrowed(&*ppv)) {
            if let Ok(obj) = unknown.cast::<ID3D12Object>() {
                let _ = obj.SetPrivateData(&CONSOLE_BC_FOOTPRINT, size_of::<u64>() as u32, Some((&size as *const u64).cast()));
            }
        }
    }
    hr
}

pub(super) unsafe extern "system" fn dev_create_command_list_x(this: *mut c_void, mask: usize, ty: usize, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pc_pso = super::console_list::pc_pso(pso as *mut c_void) as usize;
    super::shader_code::taken_on(alloc, pc_pso);
    let hr = dev_create_command_list(this, mask, ty, alloc, pc_pso, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        *ppv = super::console_list::create_with_allocator(*ppv, alloc as *mut c_void);
    }
    hr
}

unsafe extern "system" fn dev_create_compute_pso_x(this: *mut c_void, desc: *const c_void, a: usize, b: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if a != 0 || b != 0 {
        crate::xlog_warn!("CreateComputePipelineState (console, slot 62): unknown parameters {a:#x} {b:#x} ignored");
    }
    let hr = dev_create_compute_pso(this, desc, riid, ppv);
    let made = if hr >= 0 && !ppv.is_null() { *ppv as usize } else { 0 };
    // console code writes into pipeline objects (D3D12.X inline code): give it a console-shaped one
    if hr >= 0 && super::console_list::enabled() && !ppv.is_null() && !(*ppv).is_null() {
        *ppv = super::console_list::wrap_pso(*ppv, LAST_COMPUTE_TRANSLATED.with(|c| c.get()), LAST_COMPUTE_RS.with(|c| c.replace(0)));
    }
    if made != 0 {
        super::raytracing::note_builder_alias(made, *ppv as usize);
    }
    hr
}

fn overrides(kind: Kind) -> Vec<(&'static str, usize)> {
    let mut v: Vec<(&'static str, usize)> = vec![("QueryInterface", qi_override as usize)];
    match kind {
        Kind::Device => v.extend_from_slice(&[
            ("CreateCommandQueue", dev_create_command_queue as usize),
            ("CheckFeatureSupport", dev_check_feature_support as usize),
            ("CreateCommandAllocator", dev_create_command_allocator as usize),
            ("CreateGraphicsPipelineState", dev_create_graphics_pso as usize),
            ("CreateComputePipelineState", dev_create_compute_pso as usize),
            ("CreateCommandList", dev_create_command_list as usize),
            ("CreateDescriptorHeap", dev_create_descriptor_heap as usize),
            ("CreateRootSignature", dev_create_root_signature as usize),
            ("CreateRenderTargetView", dev_create_rtv as usize),
            ("CreateDepthStencilView", image_views::pc_create_dsv as usize),
            ("CopyDescriptors", image_views::pc_copy as usize),
            ("CopyDescriptorsSimple", image_views::pc_copy_simple as usize),
            ("CreateCommittedResource", dev_create_committed_resource as usize),
            ("CreateHeap", dev_create_heap as usize),
            ("CreatePlacedResource", dev_create_placed_resource as usize),
            ("CreateReservedResource", dev_create_reserved_resource as usize),
            ("CreateFence", dev_create_fence as usize),
            ("CreateQueryHeap", dev_create_query_heap as usize),
            ("CreateCommandSignature", dev_create_command_signature as usize),
            ("CreatePipelineLibrary", dev_create_pipeline_library as usize),
            ("CreatePipelineState", dev_create_pipeline_state as usize),
            ("CreateCommandList1", dev_create_command_list1 as usize),
            ("CreateCommittedResource1", dev_create_committed_resource1 as usize),
            ("CreateHeap1", dev_create_heap1 as usize),
            ("CreateReservedResource1", dev_create_reserved_resource1 as usize),
            ("CreateStateObject", if super::raytracing::off() { super::raytracing::create_state_object as usize } else { dev_create_state_object as usize }),
            ("CreateCommittedResource2", dev_create_committed_resource2 as usize),
            ("CreateCommittedResource3", dev_create_committed_resource3 as usize),
            ("CreatePlacedResource1", dev_create_placed_resource1 as usize),
            ("CreateCommandQueue1", dev_create_command_queue1 as usize),
        ]),
        Kind::CommandList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
            ("ClearState", cl_clear_state as usize),
            ("DrawInstanced", cl_draw_instanced as usize),
            ("DrawIndexedInstanced", cl_draw_indexed_instanced as usize),
            ("SetPipelineState", cl_set_pipeline_state as usize),
            ("OMSetRenderTargets", cl_om_set_render_targets as usize),
            ("ClearRenderTargetView", cl_clear_rtv as usize),
            ("ExecuteBundle", cl_execute_bundle as usize),
            ("ExecuteIndirect", cl_execute_indirect as usize),
            ("Dispatch", super::depth_uav::pc_dispatch as usize),
            ("ClearDepthStencilView", image_views::clear_dsv_pc as usize),
        ]),
        Kind::DmaList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
        ]),
        Kind::CommandQueue => v.extend_from_slice(&[
            ("Release", q_release as usize),
            ("ExecuteCommandLists", q_execute_command_lists as usize),
        ]),
        Kind::CommandAllocator => v.extend_from_slice(&[("Release", alloc_release as usize), ("Reset", alloc_reset as usize)]),
        Kind::Resource => v.extend_from_slice(&[
            ("Release", res_release as usize),
            ("Map", res_map as usize),
            ("Unmap", res_unmap as usize),
            ("WriteToSubresource", res_write_to_subresource as usize),
        ]),
        Kind::PipelineState => v.extend_from_slice(&[("Release", pso_release as usize), ("SetName", pso_set_name as usize)]),
        Kind::DescriptorHeap => v.push(("Release", heap_release as usize)),
        Kind::Heap => v.push(("Release", mem_heap_release as usize)),
        Kind::Fence => v.extend_from_slice(&[("Signal", fence_signal as usize), ("SetEventOnCompletion", fence_set_event as usize)]),
        _ => {}
    }
    let timing_hooks = cfg!(test) || super::gpu_timing::enabled();
    if kind == Kind::CommandList && super::material_gate_views::prepass::enabled() {
        v.push(("BeginQuery", super::list_x::observe_native_begin_query as usize));
    }
    if timing_hooks {
        match kind {
            Kind::CommandQueue => v.extend_from_slice(&[
                ("Wait", q_timed_wait as usize),
                ("Signal", q_timed_signal as usize),
            ]),
            Kind::Fence => v.push(("Release", timing_fence_release as usize)),
            Kind::CommandList => v.extend_from_slice(&[
                ("BeginRenderPass", cl_timed_begin_render_pass as usize),
                ("EndRenderPass", cl_timed_end_render_pass as usize),
                ("DispatchMesh", cl_timed_dispatch_mesh as usize),
            ]),
            _ => {}
        }
    }
    if kind == Kind::CommandList && !timing_hooks && (super::bind::draw_gate_hashes() || super::image_writeback::on()) {
        v.push(("DispatchMesh", cl_timed_dispatch_mesh as usize));
    }
    if !timing_hooks && kind == Kind::CommandList {
        v.extend_from_slice(&[
            ("BeginRenderPass", cl_timed_begin_render_pass as usize),
            ("EndRenderPass", cl_timed_end_render_pass as usize),
        ]);
    }
    if super::residency::eviction_enabled() || timing_hooks || super::depth_state::enabled() || super::image_writeback::on() {
        match kind {
            Kind::CommandList | Kind::DmaList => residency_list::overrides(&mut v),
            Kind::CommandQueue if super::residency::eviction_enabled() => v.extend_from_slice(&[
                ("UpdateTileMappings", q_update_tile_mappings as usize),
                ("CopyTileMappings", q_copy_tile_mappings as usize),
            ]),
            _ => {}
        }
    }
    if kind == Kind::CommandList && (super::bind::draw_gate_trace() || super::bind::ignore_predication() || super::depth_capture::enabled()) {
        v.retain(|(method, _)| *method != "SetPredication");
        v.push(("SetPredication", cl_set_predication as usize));
    }
    if kind == Kind::CommandList {
        super::raytracing::list_overrides(&mut v);
    }
    v
}

/// Slot of `method` in the class of an adopted object.
pub(super) unsafe fn slot_of(this: *mut c_void, method: &str) -> usize {
    let ext = vt(this) as usize;
    let g = CLASSES.lock().unwrap();
    let c = g.iter().find(|c| c.ext == ext).expect("adopted class");
    layout::slot(c.iface, method).expect("method in layout")
}

unsafe extern "system" fn qi_override(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let f: QiFn = std::mem::transmute(orig_fn(this, 0));
    let hr = f(this, riid, ppv);
    if hr == E_NOINTERFACE && !riid.is_null() {
        let s = layout::guid_string(&*riid);
        if UNKNOWN_IIDS.lock().unwrap().insert(s.clone()) {
            crate::xlog_warn!("D3D12.X: QueryInterface({s}) – interface unknown on the PC (console-only?)");
        }
    }
    hr
}

/// device #0 QueryInterface for console code: the console device is also an IDXGIDevice
/// (`dxgi_device`). Only the title gets it – DXGI asks a D3D12 device the same while creating
/// a swap chain and must keep getting the PC answer.
unsafe extern "system" fn dev_qi_x(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if !riid.is_null() && !ppv.is_null() && super::dxgi_device::answers(&std::mem::transmute::<GUID, windows::core::GUID>(*riid)) {
        *ppv = super::dxgi_device::create(this);
        return S_OK;
    }
    qi_override(this, riid, ppv)
}

fn pc_iid(kind: Kind) -> GUID {
    layout::parse_guid(layout::find(kind.base()).map(|i| i.iid).unwrap_or(""))
}

macro_rules! as_usize {
    ($x:ident) => {
        usize
    };
}

/// Close: a failure is logged with what was recorded (titles stop on it; the PC runtime
/// refuses some calls only here). Newer D3D12 runtimes (e.g. Agility SDK 1.620+) swap the
/// command list's vtable pointer on Close; adopt the runtime's new state vtable so
/// subsequent Reset and console calls dispatch through the current native table.
unsafe extern "system" fn cl_close(this: *mut c_void) -> HRESULT {
    super::list_x::before_close(this);
    super::gpu_timing::before_close(this);
    let ext = *(this as *const usize);
    let f: RefFn = std::mem::transmute(orig_fn(this, slot_of(this, "Close")));
    let hr = f(this) as HRESULT;
    preserve_vtable(this, ext);
    if hr < 0 {
        let ty: RefFn = std::mem::transmute(orig_fn(this, slot_of(this, "GetType")));
        crate::xlog_warn!("D3D12.X: command list {this:p} (type {}) Close → {:#x}", ty(this), hr as u32);
    }
    hr
}

/// Reset (PC path): re-adopt the runtime's new open-state vtable if Reset replaced it.
unsafe extern "system" fn cl_reset(this: *mut c_void, allocator: *mut c_void, initial_state: *mut c_void) -> HRESULT {
    super::single::before_list_reset(&[this as usize]);
    let ext = *(this as *const usize);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> HRESULT =
        std::mem::transmute(orig_fn(this, slot_of(this, "Reset")));
    let resolved = super::pso_async::resolve_for_bind(initial_state as usize) as *mut c_void;
    let hr = f(this, allocator, resolved);
    preserve_vtable(this, ext);
    if hr >= 0 {
        super::shader_code::taken_on(allocator as usize, resolved as usize);
        super::shader_code::bundle_takes(this as usize, resolved as usize);
        super::list_x::on_pc_reset(this);
        super::material_gate_views::prepass::native_reset(this as usize);
        super::list_x::capture_native_bind(this, resolved as usize);
        super::depth_state::reset(this as usize);
        super::gpu_timing::opened(this, initial_state as usize);
        super::residency::list_reset(this as usize);
        super::depth_uav::list_reset(this as usize);
        super::image_writeback::list_reset(this as usize);
    }
    hr
}

/// The PC size of a CheckFeatureSupport structure.
fn pc_feature_size(feature: i32) -> Option<usize> {
    use std::mem::size_of;
    Some(match D3D12_FEATURE(feature) {
        D3D12_FEATURE_D3D12_OPTIONS => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(),
        D3D12_FEATURE_ARCHITECTURE => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE>(),
        D3D12_FEATURE_FEATURE_LEVELS => size_of::<D3D12_FEATURE_DATA_FEATURE_LEVELS>(),
        D3D12_FEATURE_FORMAT_SUPPORT => size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>(),
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS => size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>(),
        D3D12_FEATURE_FORMAT_INFO => size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>(),
        D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT => size_of::<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT>(),
        D3D12_FEATURE_SHADER_MODEL => size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>(),
        D3D12_FEATURE_D3D12_OPTIONS1 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(),
        D3D12_FEATURE_ROOT_SIGNATURE => size_of::<D3D12_FEATURE_DATA_ROOT_SIGNATURE>(),
        D3D12_FEATURE_ARCHITECTURE1 => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>(),
        D3D12_FEATURE_D3D12_OPTIONS2 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>(),
        D3D12_FEATURE_SHADER_CACHE => size_of::<D3D12_FEATURE_DATA_SHADER_CACHE>(),
        D3D12_FEATURE_D3D12_OPTIONS3 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>(),
        D3D12_FEATURE_EXISTING_HEAPS => size_of::<D3D12_FEATURE_DATA_EXISTING_HEAPS>(),
        D3D12_FEATURE_D3D12_OPTIONS4 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>(),
        D3D12_FEATURE_D3D12_OPTIONS5 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(),
        D3D12_FEATURE_D3D12_OPTIONS6 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>(),
        D3D12_FEATURE_D3D12_OPTIONS7 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(),
        D3D12_FEATURE_D3D12_OPTIONS8 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS8>(),
        D3D12_FEATURE_D3D12_OPTIONS9 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(),
        D3D12_FEATURE_D3D12_OPTIONS10 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS10>(),
        D3D12_FEATURE_D3D12_OPTIONS11 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(),
        D3D12_FEATURE_D3D12_OPTIONS12 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>(),
        D3D12_FEATURE_D3D12_OPTIONS13 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS13>(),
        _ => return None,
    })
}

/// Apply `config.raytracing` / `XCL_RAYTRACING` override (if any) to a successful
/// `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5)` result and log the reported tier once.
unsafe fn adjust_raytracing_options5(feature: i32, data: *mut u8, size: u32) {
    if D3D12_FEATURE(feature) != D3D12_FEATURE_D3D12_OPTIONS5 || data.is_null() || (size as usize) < std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() {
        return;
    }
    let opts = &mut *(data as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5);
    let host = opts.RaytracingTier.0;
    if let Some(tier) = crate::state::config().raytracing_tier_override() {
        opts.RaytracingTier = D3D12_RAYTRACING_TIER(tier as i32);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: CheckFeatureSupport(D3D12_OPTIONS5) → RaytracingTier {} (host {})", opts.RaytracingTier.0, host);
    }
}

/// CheckFeatureSupport(feature, data, size): console structures can be larger than the PC's
/// (XDefiant: D3D12_OPTIONS with 0x40 bytes, 0x3C on the PC – and the title stops on the
/// error). A size the PC refuses is retried with the PC's, the console's extra bytes zero.
///
/// A query about one of the console's own formats (116 – 120, `device_x::pc_view_format`) asks
/// the PC about the format the resource gets there, and the caller's format field is put back.
/// A retail title asked FORMAT_SUPPORT of D16_UNORM_S8_UINT (118), the PC called it unsupported,
/// and the title stopped on its assertion.
unsafe extern "system" fn dev_check_feature_support(this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let f: CheckFeatureSupportFn = std::mem::transmute(orig_fn(this, slot_of(this, "CheckFeatureSupport")));
    check_feature_support_x(f, this, feature, data, size)
}

type CheckFeatureSupportFn = unsafe extern "system" fn(*mut c_void, i32, *mut u8, u32) -> HRESULT;

/// `dev_check_feature_support` with the PC device's own CheckFeatureSupport `f`.
pub(super) unsafe fn check_feature_support_x(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let console = console_format_query(feature, data, size);
    let hr = check_feature_support_pc(f, this, feature, data, size);
    if let Some(format) = console {
        (data as *mut u32).write_unaligned(format);
    }
    hr
}

/// The features whose data starts with a DXGI_FORMAT the caller asks about.
fn feature_takes_format(feature: i32) -> bool {
    matches!(
        D3D12_FEATURE(feature),
        D3D12_FEATURE_FORMAT_SUPPORT | D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS | D3D12_FEATURE_FORMAT_INFO | D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO
    )
}

/// For a format query (`feature_takes_format`) about a console-only format: its PC format written
/// into `data` in its place, and the console's returned (to be put back after the call).
unsafe fn console_format_query(feature: i32, data: *mut u8, size: u32) -> Option<u32> {
    if !feature_takes_format(feature) || data.is_null() || size < 4 {
        return None;
    }
    let format = (data as *const u32).read_unaligned();
    let pc = super::device_x::pc_view_format(format);
    (pc != format).then(|| {
        (data as *mut u32).write_unaligned(pc);
        format
    })
}

unsafe fn check_feature_support_pc(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let hr = f(this, feature, data, size);
    if hr >= 0 {
        adjust_raytracing_options5(feature, data, size);
        return hr;
    }
    let Some(pc) = pc_feature_size(feature).filter(|&pc| !data.is_null() && pc != size as usize) else { return hr };
    let n = pc.min(size as usize);
    let mut tmp = vec![0u8; pc];
    std::ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), n); // input fields (format, sample count …)
    let hr2 = f(this, feature, tmp.as_mut_ptr(), pc as u32);
    if hr2 >= 0 {
        std::ptr::write_bytes(data, 0, size as usize);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), data, n);
        adjust_raytracing_options5(feature, data, size);
    }
    static SEEN: Mutex<Vec<i32>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.contains(&feature) {
        seen.push(feature);
        crate::xlog_info!("D3D12.X: CheckFeatureSupport({feature}) with the console's {size} bytes, the PC's {pc} → {:#x}", hr2 as u32);
    }
    hr2
}

/// Device creation methods ending in (REFIID riid, void** ppv).
macro_rules! creator {
    ($name:ident, $method:literal, $kind:expr, [$($a:ident),*]) => {
        pub(super) unsafe extern "system" fn $name(this: *mut c_void, $($a: usize,)* riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
            type F = unsafe extern "system" fn(*mut c_void, $(as_usize!($a),)* *const GUID, *mut *mut c_void) -> HRESULT;
            let f: F = std::mem::transmute(orig_fn(this, slot_of(this, $method)));
            let mut hr = f(this, $($a,)* riid, ppv);
            if hr == E_NOINTERFACE {
                let iid = pc_iid($kind);
                hr = f(this, $($a,)* &iid, ppv);
                // an interface the PC does not have (a console one): which, once per method
                static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let asked = riid.as_ref().map(super::layout::guid_string).unwrap_or_default();
                let new = { let mut g = SEEN.lock().unwrap(); let n = !g.contains(&asked); if n { g.push(asked.clone()); } n };
                if new {
                    crate::xlog_info!("D3D12.X: {} for interface {asked} (not on the PC) – made as {}", $method, super::layout::guid_string(&iid));
                }
            }
            if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
                adopt(*ppv, $kind);
                if $method == "CreateCommandList" {
                    super::gpu_timing::opened(*ppv, [$($a,)*].get(3).copied().unwrap_or(0));
                }
                if $method == "CreateCommandQueue" || $method == "CreateCommandQueue1" {
                    if let Some(device) = ID3D12Device::from_raw_borrowed(&this) {
                        let _ = super::page_mappings::queue::register(*ppv, device);
                    }
                    super::gpu_timing::queue_created(*ppv);
                }
                if $method == "CreateFence" && super::gpu_timing::enabled() {
                    super::gpu_timing_waits::forget_fence(*ppv as usize);
                }
            }
            hr
        }
    };
}

creator!(dev_create_command_queue, "CreateCommandQueue", Kind::CommandQueue, [a1]);
creator!(dev_create_command_queue1, "CreateCommandQueue1", Kind::CommandQueue, [a1, a2]);
creator!(dev_create_command_allocator, "CreateCommandAllocator", Kind::CommandAllocator, [a1]);
creator!(dev_create_command_list, "CreateCommandList", Kind::CommandList, [a1, a2, a3, a4]);
creator!(dev_create_command_list1, "CreateCommandList1", Kind::CommandList, [a1, a2, a3]);

/// A raw adopted PC `ID3D12GraphicsCommandList` made closed, with no allocator and no recording
/// (`CreateCommandList1`), not wrapped in `title_list`: a call put off to the submit is its first
/// recording (`list_x::put_off_segment`).
pub(super) unsafe fn create_closed_pc_command_list(dev: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE) -> Option<*mut c_void> {
    if dev.is_null() {
        return None;
    }
    let ty = D3D12_COMMAND_LIST_TYPE(super::single::list_type(ty.0 as usize) as i32);
    if is_ext(vt(dev) as usize) {
        let iid = layout::parse_guid(layout::find("ID3D12GraphicsCommandList")?.iid);
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = dev_create_command_list1(dev, 0, ty.0 as usize, D3D12_COMMAND_LIST_FLAG_NONE.0 as usize, &iid, &mut out);
        return (hr >= 0 && !out.is_null()).then_some(out);
    }
    let dev4 = ID3D12Device::from_raw_borrowed(&dev)?.cast::<ID3D12Device4>().ok()?;
    let out = dev4.CreateCommandList1::<ID3D12GraphicsCommandList>(0, ty, D3D12_COMMAND_LIST_FLAG_NONE).ok()?.into_raw();
    adopt(out, Kind::CommandList);
    Some(out)
}

/// Create a raw adopted PC `ID3D12GraphicsCommandList` without wrapping it in `title_list`
/// (used for deferred mid-list split segments on the same allocator).
pub(super) unsafe fn create_raw_pc_command_list(
    dev: *mut c_void,
    ty: D3D12_COMMAND_LIST_TYPE,
    alloc: usize,
    init_pso: usize,
) -> Option<*mut c_void> {
    if dev.is_null() || alloc == 0 {
        return None;
    }
    let iid = layout::parse_guid(layout::find("ID3D12GraphicsCommandList")?.iid);
    let mut out: *mut c_void = std::ptr::null_mut();
    let ty = D3D12_COMMAND_LIST_TYPE(super::single::list_type(ty.0 as usize) as i32);
    let hr = if is_ext(vt(dev) as usize) {
        dev_create_command_list(dev, 0, ty.0 as usize, alloc, init_pso, &iid, &mut out)
    } else {
        type F = unsafe extern "system" fn(*mut c_void, u32, D3D12_COMMAND_LIST_TYPE, usize, usize, *const GUID, *mut *mut c_void) -> HRESULT;
        let slot = layout::slot(layout::find("ID3D12Device")?, "CreateCommandList")?;
        let f: F = std::mem::transmute(*vt(dev).add(slot));
        let hr = f(dev, 0, ty, alloc, init_pso, &iid, &mut out);
        if hr >= 0 && !out.is_null() {
            adopt(out, Kind::CommandList);
            super::gpu_timing::opened(out, init_pso);
        }
        hr
    };
    (hr >= 0 && !out.is_null()).then_some(out)
}
creator!(dev_create_descriptor_heap_raw, "CreateDescriptorHeap", Kind::DescriptorHeap, [a1]);

unsafe extern "system" fn dev_create_descriptor_heap(
    this: *mut c_void, a1: usize, riid: *const GUID, ppv: *mut *mut c_void,
) -> HRESULT {
    let hr = dev_create_descriptor_heap_raw(this, a1, riid, ppv);
    if hr >= 0 && image_views::enabled() && !ppv.is_null() && !(*ppv).is_null() {
        if let Some(d) = (a1 as *const D3D12_DESCRIPTOR_HEAP_DESC).as_ref() {
            if let Some(kind) = image_views::Kind::of(d.Type.0 as u32) {
                image_views::heap(*ppv as usize, this as usize, kind, heap_cpu_start_x(*ppv) as usize,
                    dev_increment_size_x(this, d.Type.0 as u32) as usize, d.NumDescriptors);
            }
        }
    }
    if hr >= 0 && super::depth_capture_views::enabled() && !ppv.is_null() && !(*ppv).is_null() {
        if let Some(desc) = (a1 as *const D3D12_DESCRIPTOR_HEAP_DESC).as_ref() {
            if desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV {
                let first = heap_cpu_start_x(*ppv) as usize;
                let step = dev_increment_size_x(this, desc.Type.0 as u32);
                super::depth_capture_views::record_heap(*ppv as usize, first, desc.NumDescriptors, step);
            }
        }
    }
    hr
}

macro_rules! committed_creator {
    ($name:ident, $raw:ident, $method:literal, $desc1:expr $(, $session:ident)?) => {
        creator!($raw, $method, Kind::Resource, [props, flags, desc, initial, clear $(, $session)?]);

        pub(super) unsafe extern "system" fn $name(
            this: *mut c_void, props: usize, flags: usize, desc: usize, initial: usize, clear: usize,
            $($session: usize,)? riid: *const GUID, ppv: *mut *mut c_void,
        ) -> HRESULT {
            let session = 0usize $(| $session)?;
            committed_with_residency(this, props, flags, desc, $desc1, session, ppv, ($method, "host_state", initial),
                |p| $raw(this, p, flags, desc, initial, clear $(, $session)?, riid, ppv))
        }
    };
}

committed_creator!(dev_create_committed_resource, dev_create_committed_resource_pc, "CreateCommittedResource", false);
committed_creator!(dev_create_committed_resource1, dev_create_committed_resource1_pc, "CreateCommittedResource1", false, session);
committed_creator!(dev_create_committed_resource2, dev_create_committed_resource2_pc, "CreateCommittedResource2", true, session);
creator!(dev_create_committed_resource3_pc, "CreateCommittedResource3", Kind::Resource, [props, flags, desc, layout, clear, session, n, formats]);

unsafe extern "system" fn dev_create_committed_resource3(
    this: *mut c_void, props: usize, flags: usize, desc: usize, layout: usize, clear: usize,
    session: usize, n: usize, formats: usize, riid: *const GUID, ppv: *mut *mut c_void,
) -> HRESULT {
    committed_with_residency(this, props, flags, desc, true, session, ppv, ("CreateCommittedResource3", "host_layout", layout),
        |p| dev_create_committed_resource3_pc(this, p, flags, desc, layout, clear, session, n, formats, riid, ppv))
}

unsafe fn committed_with_residency(
    this: *mut c_void, props: usize, flags: usize, desc: usize, desc1: bool, session: usize,
    ppv: *mut *mut c_void, creation: (&'static str, &'static str, usize), mut create: impl FnMut(usize) -> HRESULT,
) -> HRESULT {
    let mut create = |p| {
        let hr = create(p);
        note_depth_creation(hr, ppv, desc, creation);
        hr
    };
    if ppv.is_null() || super::title_textures::is_own_work()
        || !crate::gpu::shader::readable(props, size_of::<D3D12_HEAP_PROPERTIES>())
        || !crate::gpu::shader::readable(desc, size_of::<D3D12_RESOURCE_DESC>())
    {
        return create(props);
    }
    let Some(dev) = ID3D12Device::from_raw_borrowed(&this) else { return create(props) };
    let properties = (props as *const D3D12_HEAP_PROPERTIES).read_unaligned();
    let flags = D3D12_HEAP_FLAGS(flags as i32);
    if !super::residency::eviction_enabled() && (session != 0 || !super::residency::could_spill(properties, flags)) {
        return create(props);
    }
    let d = (desc as *const D3D12_RESOURCE_DESC).read_unaligned();
    let bytes = if desc1 {
        if !crate::gpu::shader::readable(desc, size_of::<D3D12_RESOURCE_DESC1>()) {
            return create(props);
        }
        let Ok(dev8) = dev.cast::<ID3D12Device8>() else { return create(props) };
        let d1 = (desc as *const D3D12_RESOURCE_DESC1).read_unaligned();
        dev8.GetResourceAllocationInfo2(properties.VisibleNodeMask, 1, &d1, None).SizeInBytes
    } else {
        dev.GetResourceAllocationInfo(properties.VisibleNodeMask, &[d]).SizeInBytes
    };
    let high = super::residency::protected_texture(&d);
    let mut alloc = |p: &D3D12_HEAP_PROPERTIES| {
        let hr = windows::core::HRESULT(create(p as *const _ as usize));
        hr.ok().map(|()| hr.0)
    };
    let made = if session == 0 { super::residency::allocate(dev, properties, flags, bytes, high, alloc) } else {
        alloc(&properties).map(|hr| (hr, false))
    };
    match made {
        Ok((hr, spilled)) => {
            if !(*ppv).is_null() {
                super::residency::track(dev, *ppv as usize, properties, bytes, high, session == 0 && super::residency::may_evict(flags), spilled);
            }
            hr
        }
        Err(e) => e.code().0,
    }
}

unsafe fn note_depth_creation(hr: HRESULT, ppv: *mut *mut c_void, desc: usize, (route, initial_kind, initial): (&str, &str, usize)) {
    if hr >= 0 && super::depth_lineage::enabled() && !ppv.is_null() && !(*ppv).is_null()
        && !super::title_textures::is_own_work()
        && crate::gpu::shader::readable(desc, size_of::<D3D12_RESOURCE_DESC>())
    {
        let d = (desc as *const D3D12_RESOURCE_DESC).read_unaligned();
        super::device_x::note_depth_resource_created(*ppv as usize, d.Flags, || format!(
            "route={route} {initial_kind}={initial:#x} format={}", d.Format.0,
        ));
    }
}

macro_rules! placed_creator {
    ($name:ident, $raw:ident, $method:literal) => {
        creator!($raw, $method, Kind::Resource, [heap, offset, desc, state, clear]);

        pub(super) unsafe extern "system" fn $name(
            this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize,
            riid: *const GUID, ppv: *mut *mut c_void,
        ) -> HRESULT {
            let hr = $raw(this, heap, offset, desc, state, clear, riid, ppv);
            if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
                super::residency::link(*ppv as usize, heap);
                note_depth_creation(hr, ppv, desc, ($method, "host_state", state));
            }
            hr
        }
    };
}

placed_creator!(dev_create_placed_resource, dev_create_placed_resource_pc, "CreatePlacedResource");

/// D3D12_RESOURCE_STATES the PC knows (as `list_x::PC_STATES`)
pub(super) const PC_RESOURCE_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// device #29 CreatePlacedResource(heap, offset, desc, state, clear, riid, resource) for console
/// code: a description the PC refuses (console layouts and flags – Fortnite's buffer in a
/// buffers-only heap) is made the PC's way (`device_x::pc_resource_desc`, the console's state
/// bits dropped). Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_placed_resource_x(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    // a buffer in a heap that has unified memory (`note_heap_uma`): that memory at its offset
    if desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
        let is_buffer = u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
        if let Some((block, cpu_readback)) = is_buffer.then(|| heap_block(heap)).flatten() {
            let d = super::device_x::pc_resource_desc(&x);
            let st = state as u32 & PC_RESOURCE_STATES;
            if let Some(hr) = place_at_title_address(this, block + offset, &d, st, clear as *const c_void, riid, ppv) {
                if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && cpu_readback && crate::gpu::uma::get().is_some_and(|u| u.is_sparse(block + offset)) {
                    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
                    if st == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32 && (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32) == 0 {
                        note_sparse_readback(*ppv as usize);
                    }
                }
                return hr;
            }
        }
    }
    if let Some(hr) = own_memory_if_larger_on_the_pc(this, heap, offset, desc, state, clear, riid, ppv) {
        return hr;
    }
    let hr = dev_create_placed_resource(this, heap, offset, desc, state, clear, riid, ppv);
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return hr;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let st = state as u32 & PC_RESOURCE_STATES;
    let mut clear_buf = D3D12_CLEAR_VALUE::default();
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf, Some((&x, &d))) as usize;
    let hr2 = dev_create_placed_resource(this, heap, offset, d.as_ptr() as usize, st as usize, clear, riid, ppv);
    let heap_desc = pc_heap_desc(heap);
    let pc_size = pc_allocation_size(this, &d);
    // A texture the title placed by the console's size (`get_resource_allocation_info_x`
    // answers with XG's) can be larger on the PC and run past the end of its heap, which the
    // PC refuses – Minecraft then ended (0xC0000409). It gets memory of its own, of the
    // heap's kind; the console's aliasing of that one placement is lost.
    let is_texture = u32::from_le_bytes(x[0..4].try_into().unwrap()) != D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
    let hr3 = match heap_desc {
        Some(h) if hr2 < 0 && is_texture => Some(dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear, riid, ppv)),
        _ => None,
    };
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        let sizes = format!(
            "heap {:#x} bytes, PC size {:#x}{}",
            heap_desc.map_or(0, |h| h.SizeInBytes),
            pc_size,
            if heap_desc.is_some_and(|h| offset as u64 + pc_size > h.SizeInBytes) { " – past the heap's end" } else { "" }
        );
        let own = hr3.map_or(String::new(), |h| format!("; memory of its own: {h:#x}"));
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource(dimension {}, {}x{}, format {}, layout {:#x}, flags {:#x}, state {state:#x}) at {offset:#x} refused ({hr:#x}); the PC's way: {hr2:#x} ({sizes}){own}",
            u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x2C), u(0x30)
        );
    }
    hr3.unwrap_or(hr2)
}

/// A texture the title placed in a heap by the console's size (`get_resource_allocation_info_x`
/// answers with XG's), when the PC's is larger: placed at the console's offset it would cover
/// the start of whatever the title placed after it, and each would write over the other.
/// Minecraft's logo (1937x333, 0x2A0000 bytes on the console, 0x2E0000 on the PC) ran into the
/// text and 16x16 textures packed after it – magenta / black squares and a white speck along
/// its bottom edge. Such a texture gets memory of its own, of the heap's kind; the console's
/// aliasing of that one placement is lost. None: placed as the title asked.
#[allow(clippy::too_many_arguments)]
unsafe fn own_memory_if_larger_on_the_pc(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 {
        return None;
    }
    let console = crate::hle::xg::Texture::new(&x).ok()?.size();
    let d = super::device_x::pc_resource_desc(&x);
    let pc = pc_allocation_size(this, &d);
    if console == u64::MAX || pc == u64::MAX || pc <= console {
        return None;
    }
    let h = pc_heap_desc(heap)?;
    let st = state as u32 & PC_RESOURCE_STATES;
    let mut clear_buf = D3D12_CLEAR_VALUE::default();
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf, Some((&x, &d))) as usize;
    let mut hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear, riid, ppv);
    if hr < 0 && st != 0 {
        // as in the past-the-end fallback: a state the PC refuses for this desc (e.g. UAV
        // without the UAV flag) – COMMON rather than falling back to the overlapping placement
        hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, 0, clear, riid, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource({}x{}, format {}, flags {:#x}) at {offset:#x}: {pc:#x} bytes on the PC, {console:#x} on the console – memory of its own ({hr:#x})",
            u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30)
        );
    }
    (hr >= 0).then_some(hr)
}

/// `XCL_DEBUG_NO_STENCIL` / `XCL_DEBUG_NO_CULL` `=hash,…` (pixel shader dump hashes, hex):
/// pipelines with those pixel shaders draw without their stencil test / culling – diagnostics
/// for draws the fixed-function stages reject.
fn debug_pso_switch(var: &str, ps: u64) -> bool {
    ps != 0 && std::env::var(var).is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == ps))
}

/// The PC description of a heap XCL adopted (ID3D12Heap::GetDesc, slot 8).
unsafe fn pc_heap_desc(heap: usize) -> Option<D3D12_HEAP_DESC> {
    if !is_adopted(heap as *mut c_void) {
        return None;
    }
    type F = unsafe extern "system" fn(*mut c_void, *mut D3D12_HEAP_DESC) -> *mut D3D12_HEAP_DESC;
    let f: F = std::mem::transmute(orig_fn(heap as *mut c_void, 8));
    let mut h = D3D12_HEAP_DESC::default();
    f(heap as *mut c_void, &mut h);
    Some(h)
}

/// The PC driver's size for a (PC) resource description (GetResourceAllocationInfo, slot 25).
unsafe fn pc_allocation_size(device: *mut c_void, d: &[u8; 0x38]) -> u64 {
    type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
    let f: F = std::mem::transmute(orig_fn(device, 25));
    let mut r = [0u64; 2];
    f(device, &mut r, 0, 1, d);
    r[0]
}
placed_creator!(dev_create_placed_resource1, dev_create_placed_resource1_pc, "CreatePlacedResource1");
creator!(dev_create_reserved_resource_raw, "CreateReservedResource", Kind::Resource, [desc, state, clear]);
creator!(dev_create_reserved_resource1_raw, "CreateReservedResource1", Kind::Resource, [desc, state, clear, session]);

pub(super) unsafe extern "system" fn dev_create_reserved_resource(
    this: *mut c_void, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void,
) -> HRESULT {
    let hr = dev_create_reserved_resource_raw(this, desc, state, clear, riid, ppv);
    note_depth_creation(hr, ppv, desc, ("CreateReservedResource", "host_state", state));
    hr
}

pub(super) unsafe extern "system" fn dev_create_reserved_resource1(
    this: *mut c_void, desc: usize, state: usize, clear: usize, session: usize, riid: *const GUID, ppv: *mut *mut c_void,
) -> HRESULT {
    let hr = dev_create_reserved_resource1_raw(this, desc, state, clear, session, riid, ppv);
    note_depth_creation(hr, ppv, desc, ("CreateReservedResource1", "host_state", state));
    hr
}
creator!(dev_create_heap_raw, "CreateHeap", Kind::Heap, [a1]);

unsafe fn dev_create_heap_pc(this: *mut c_void, desc: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    heap_with_residency(this, desc, 0, ppv, |d| dev_create_heap_raw(this, d, riid, ppv))
}

unsafe fn heap_with_residency(
    this: *mut c_void, desc: usize, session: usize, ppv: *mut *mut c_void, mut create: impl FnMut(usize) -> HRESULT,
) -> HRESULT {
    if ppv.is_null() || super::title_textures::is_own_work() || !crate::gpu::shader::readable(desc, size_of::<D3D12_HEAP_DESC>()) {
        return create(desc);
    }
    let Some(dev) = ID3D12Device::from_raw_borrowed(&this) else { return create(desc) };
    let d = (desc as *const D3D12_HEAP_DESC).read_unaligned();
    let high = !d.Flags.contains(D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);
    let mut alloc = |p: &D3D12_HEAP_PROPERTIES| {
        let chosen = D3D12_HEAP_DESC { Properties: *p, ..d };
        let hr = windows::core::HRESULT(create(&chosen as *const _ as usize));
        hr.ok().map(|()| hr.0)
    };
    let made = if session == 0 { super::residency::allocate(dev, d.Properties, d.Flags, d.SizeInBytes, high, alloc) } else {
        alloc(&d.Properties).map(|hr| (hr, false))
    };
    match made {
        Ok((hr, spilled)) => {
            if !(*ppv).is_null() {
                crate::memdiag::TITLE_HEAPS[crate::memdiag::heap_class(&d.Properties)].add(d.SizeInBytes);
                super::residency::track(dev, *ppv as usize, d.Properties, d.SizeInBytes, high, session == 0 && super::residency::may_evict(d.Flags), spilled);
            }
            hr
        }
        Err(e) => e.code().0,
    }
}

/// D3D12_HEAP_FLAGS the PC knows: SHARED 0x1, DENY_BUFFERS 0x4, ALLOW_DISPLAY 0x8,
/// SHARED_CROSS_ADAPTER 0x20, DENY_RT_DS_TEXTURES 0x40, DENY_NON_RT_DS_TEXTURES 0x80, then
/// 0x100 … 0x2000 (HARDWARE_PROTECTED … TOOLS_USE_MANUAL_WRITE_TRACKING); 0x2 and 0x10 are
/// the console's (Fortnite: 0xC2)
const PC_HEAP_FLAGS: u32 = 0x3FED;

/// CreateHeap(desc, riid, heap): a console description the PC refuses (console flags in the
/// high bits, alignments other than 64 KiB / 4 MiB, console heap types – Fortnite's distance
/// field atlas heap) is made the PC's way: known flags only, the alignment the PC has that
/// covers it, an unknown type as DEFAULT. Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_heap(this: *mut c_void, desc: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_heap_pc(this, desc, riid, ppv);
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return hr;
    }
    let mut d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    let orig = (d.SizeInBytes, d.Properties.Type.0, d.Alignment, d.Flags.0);
    d.Flags = D3D12_HEAP_FLAGS(d.Flags.0 & PC_HEAP_FLAGS as i32);
    d.Alignment = if d.Alignment <= 0x1_0000 { 0 } else { D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64 };
    if !(1..=5).contains(&d.Properties.Type.0) {
        d.Properties = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    }
    d.SizeInBytes = d.SizeInBytes.div_ceil(0x1_0000) * 0x1_0000;
    let hr2 = dev_create_heap_pc(this, &d as *const _ as usize, riid, ppv);
    if hr2 >= 0 {
        note_heap_uma(desc, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: CreateHeap({:#x} bytes, type {}, alignment {:#x}, flags {:#x}) refused ({hr:#x}); as type {}, alignment {:#x}, flags {:#x}: {hr2:#x}",
            orig.0, orig.1, orig.2, orig.3, d.Properties.Type.0, d.Alignment, d.Flags.0
        );
    }
    hr2
}
creator!(dev_create_heap1_pc, "CreateHeap1", Kind::Heap, [a1, a2]);

pub(super) unsafe extern "system" fn dev_create_heap1(this: *mut c_void, desc: usize, session: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = heap_with_residency(this, desc, session, ppv, |d| dev_create_heap1_pc(this, d, session, riid, ppv));
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    hr
}

struct HeapUma {
    size: usize,
    align: usize,
    /// Small explicit `D3D12_HEAP_TYPE_READBACK` heap: prefers coherent Window 3 RAM.
    prefer_coherent: bool,
    /// `READBACK` or `CUSTOM + WRITE_BACK`: if backed by sparse VRAM, `COPY_DEST` buffers placed
    /// in it read back on `Map(0, NULL, ...)`.
    cpu_readback: bool,
    block: Option<usize>,
    /// a heap of buffers only (no textures): its PC heap holds nothing once it has a block
    buffers_only: bool,
}

/// heap → its backing block (allocated lazily on the first placed buffer; sparse VRAM for
/// DEFAULT/UPLOAD/CUSTOM heaps, coherent committed window for small explicit READBACK heaps)
static HEAP_UMA: Mutex<Option<HashMap<usize, HeapUma>>> = Mutex::new(None);

/// A heap the title made that may hold buffers: a block of unified memory as large stands for
/// it. On the console every heap is memory the CPU and the GPU address alike, and console code
/// reads buffers (root descriptors, V#s) from that memory – a PC heap's buffer has a GPU
/// address only (Minecraft's constant and vertex buffers: a heap of its own; its UI drew
/// nothing – the constants read as zero). Buffers placed in the heap are placed in the block
/// (`dev_create_placed_resource_x`); textures stay in the PC heap. The block is taken when the
/// first buffer is placed (most heaps hold textures only: a block for each filled the 2 GiB of
/// unified memory, and draws then found no room for their arguments – the image flickered);
/// none when unified memory is full (the PC heap as before).
unsafe fn note_heap_uma(desc: usize, ppv: *mut *mut c_void) {
    const DENY_BUFFERS: i32 = 0x4;
    if super::title_textures::is_own_work() || desc == 0 || ppv.is_null() || (*ppv).is_null() || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return;
    }
    let d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    if d.Flags.0 & DENY_BUFFERS != 0 || d.SizeInBytes == 0 || UMA_BUFFER.lock().unwrap().is_empty() {
        return;
    }
    let size = d.SizeInBytes as usize;
    let is_rb = d.Properties.Type == D3D12_HEAP_TYPE_READBACK;
    let is_custom_wb = d.Properties.Type == D3D12_HEAP_TYPE_CUSTOM && d.Properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    HEAP_UMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(
        *ppv as usize,
        HeapUma {
            size,
            align: (d.Alignment as usize).max(0x1_0000),
            prefer_coherent: is_rb && size <= COHERENT_READBACK_MAX,
            cpu_readback: is_rb || is_custom_wb,
            block: None,
            buffers_only: d.Flags.0 & 0xC0 == 0xC0,
        },
    );
}

/// The block of unified memory standing for `heap` and whether it may hold CPU-readback buffers
/// (taken now if it has none yet; None: not a heap that takes one, or unified memory is full).
fn heap_block(heap: usize) -> Option<(usize, bool)> {
    let (size, align, prefer_coherent, cpu_readback) = {
        let g = HEAP_UMA.lock().unwrap();
        let e = g.as_ref()?.get(&heap)?;
        if let Some(p) = e.block {
            return Some((p, e.cpu_readback));
        }
        (e.size, e.align, e.prefer_coherent, e.cpu_readback)
    };
    let u = crate::gpu::uma::get()?;
    let rounded = size.div_ceil(0x1_0000) * 0x1_0000;
    let use_coherent = prefer_coherent && has_coherent_headroom(u, rounded);
    let p = if !use_coherent && u.has_sparse() {
        unsafe { super::sparse::alloc(size, align) }
            .map(|(a, _)| a)
            .or_else(|| u.alloc(rounded, align).map(|a| a as usize))
    } else {
        u.alloc(rounded, align)
            .map(|a| a as usize)
            .or_else(|| u.has_sparse().then(|| unsafe { super::sparse::alloc(size, align) }.map(|(a, _)| a)).flatten())
    };
    let Some(p) = p else {
        static FULL: AtomicU64 = AtomicU64::new(0);
        if FULL.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!(
                "D3D12.X: heap of {size:#x} bytes: unified memory is full ({} MB sparse, {} MB coherent used) – its buffers stay PC-only",
                u.sparse_used() >> 20,
                u.used() >> 20
            );
        }
        return None;
    };
    let mut g = HEAP_UMA.lock().unwrap();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&heap)) else {
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return None;
    };
    if let Some(existing) = e.block {
        let cr = e.cpu_readback;
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return Some((existing, cr));
    }
    e.block = Some(p);
    if e.buffers_only {
        crate::memdiag::BUFFER_HEAPS_IN_UMA.add(e.size as u64);
    }
    drop(g);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 4 {
        let kind = if u.is_sparse(p) { "sparse VRAM" } else { "coherent RAM" };
        crate::xlog_info!("D3D12.X: heap {heap:#x} ({size:#x} bytes) holds buffers: in {kind} at {p:#x}");
    }
    Some((p, cpu_readback))
}

/// ID3D12Heap::Release: its block of unified memory goes with it.
unsafe extern "system" fn mem_heap_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = super::residency::release(this as usize, || f(this));
    if n == 0 {
        let gone = HEAP_UMA.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize))).and_then(|e| e.block);
        if let (Some(p), Some(u)) = (gone, crate::gpu::uma::get()) {
            if u.is_sparse(p) {
                super::sparse::release_owned_block(p);
            } else {
                u.free(p);
            }
        }
    }
    n
}
creator!(dev_create_fence_pc, "CreateFence", Kind::Fence, [a1, a2]);

pub(super) unsafe fn create_residency_fence(dev: &ID3D12Device) -> windows::core::Result<ID3D12Fence> {
    if !is_ext(vt(dev.as_raw()) as usize) {
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        if super::gpu_timing::enabled() {
            super::gpu_timing_waits::forget_fence(fence.as_raw() as usize);
        }
        return Ok(fence);
    }
    let mut out = std::ptr::null_mut();
    let iid = pc_iid(Kind::Fence);
    windows::core::HRESULT(dev_create_fence_pc(dev.as_raw(), 0, 0, &iid, &mut out)).ok()?;
    Ok(ID3D12Fence::from_raw(out))
}

/// fences created through the device (AddRef'd: they are released at the end only)
/// (a read-write lock: `is_fence` reads it on every queue call, and `end_title` reads it with
/// the title's threads suspended – a reader that holds it then blocks no other reader)
static FENCES: std::sync::RwLock<Vec<usize>> = std::sync::RwLock::new(Vec::new());

/// CreateFence: as on the PC; the fence is remembered for `end_title`
unsafe extern "system" fn dev_create_fence(this: *mut c_void, value: usize, flags: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_fence_pc(this, value, flags, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        let add: RefFn = std::mem::transmute(*vt(*ppv).add(1));
        add(*ppv);
        let n = {
            let mut g = FENCES.write().unwrap();
            g.push(*ppv as usize);
            g.len()
        };
        // `XCL_WATCH_FENCE=n`: writes over the n-th fence's first 8 bytes (its vtable) are logged
        if std::env::var("XCL_WATCH_FENCE").ok().and_then(|v| v.parse::<usize>().ok()) == Some(n) {
            crate::xlog_info!("D3D12.X: fence #{n} {:p} (initial value {value:#x}, flags {flags:#x}) watched", *ppv);
            crate::eh::watch_writes(*ppv as usize);
        }
    }
    hr
}

/// End the title on purpose without leaving the process stuck in the driver: every fence it
/// created is signalled to the largest value (GPU waits on values the title would have signalled
/// later complete), the queues drain, and the process ends without running DLL detach code.
pub fn end_title(code: u32) -> ! {
    unsafe {
        // a last resort: the process ends within 15 s whatever happens below (a suspended
        // thread may hold a lock the drain needs)
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = std::thread::Builder::new().name("xcl-end-watchdog".into()).spawn(move || {
            let _ = tx.send(windows_sys::Win32::System::Threading::GetCurrentThreadId());
            std::thread::sleep(std::time::Duration::from_secs(15));
            windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
        });
        let watchdog = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap_or(0);
        crate::log::flush();
        // the title's threads stop first: with its fences at the maximum a running title thinks
        // the GPU is done and resets command allocators the GPU still runs (Fortnite: a device
        // hang at the end of a run)
        suspend_other_threads(watchdog);
        for &f in FENCES.read().unwrap_or_else(|e| e.into_inner()).iter() {
            let fr = f as *mut c_void;
            if let Some(f) = ID3D12Fence::from_raw_borrowed(&fr) {
                let _ = f.Signal(u64::MAX);
            }
        }
        drain_gpu();
        crate::log::flush();
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    std::process::abort()
}

/// Every thread of the process but this one and `keep` (XCL's end watchdog), suspended – twice
/// over: the title may start threads meanwhile.
unsafe fn suspend_other_threads(keep: u32) {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, SuspendThread, THREAD_SUSPEND_RESUME};
    let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
    let mut done: Vec<u32> = Vec::with_capacity(1024);
    for _ in 0..2 {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            let tid = te.th32ThreadID;
            if te.th32OwnerProcessID == pid && tid != me && tid != keep && !done.contains(&tid) {
                let h = OpenThread(THREAD_SUSPEND_RESUME, 0, tid);
                if !h.is_null() {
                    SuspendThread(h);
                    CloseHandle(h);
                }
                if done.len() < done.capacity() {
                    done.push(tid);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
    }
}
creator!(dev_create_query_heap, "CreateQueryHeap", Kind::QueryHeap, [a1]);
creator!(dev_create_command_signature, "CreateCommandSignature", Kind::CommandSignature, [a1, a2]);
creator!(dev_create_pipeline_library, "CreatePipelineLibrary", Kind::PipelineLibrary, [a1, a2]);
creator!(dev_create_state_object, "CreateStateObject", Kind::StateObject, [a1]);

// ---------------------------------------------------------------------------------------
// pipeline states: capture + fallback
// ---------------------------------------------------------------------------------------

struct Dummy {
    vs: Vec<u8>,
    ps: Vec<u8>,
    cs: Vec<u8>,
}

/// Stand-in mesh shader and pixel shader (DXIL, as mesh pipelines need): one degenerate
/// triangle (DXC wants every output written), nothing drawn.
fn dummy_mesh() -> Option<&'static (Vec<u8>, Vec<u8>)> {
    static D: OnceLock<Option<(Vec<u8>, Vec<u8>)>> = OnceLock::new();
    D.get_or_init(|| {
        use crate::gpu::recomp::dxc::compile;
        let ms = "struct V { float4 p : SV_Position; };\n[outputtopology(\"triangle\")]\n[numthreads(1, 1, 1)]\nvoid main(out vertices V v[1], out indices uint3 t[1]) { SetMeshOutputCounts(1, 1); v[0].p = float4(0, 0, 0, 1); t[0] = uint3(0, 0, 0); }\n";
        let ps = "float4 main() : SV_Target { return float4(1, 0, 1, 1); }\n";
        match (compile(ms, "main", "ms_6_5", &[]), compile(ps, "main", "ps_6_0", &[])) {
            (Ok(m), Ok(p)) => Some((m, p)),
            (m, p) => {
                crate::xlog_warn!("D3D12.X: stand-in mesh shaders: {:?} {:?}", m.err(), p.err());
                None
            }
        }
    })
    .as_ref()
}

pub(super) fn compile(src: &str, target: &str) -> Vec<u8> {
    unsafe {
        let mut blob = None;
        let t = format!("{target}\0");
        let r = D3DCompile(
            src.as_ptr() as *const c_void,
            src.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(t.as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec(),
            _ => Vec::new(),
        }
    }
}

/// Stand-in shaders used when the PC driver rejects a console shader.
fn dummy() -> &'static Dummy {
    static D: OnceLock<Dummy> = OnceLock::new();
    D.get_or_init(|| Dummy {
        vs: compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }", "vs_5_0"),
        ps: compile("float4 main() : SV_Target { return float4(1, 0, 1, 1); }", "ps_5_0"),
        cs: compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0"),
    })
}

#[cfg(test)]
pub(super) fn dummy_cs_bytecode() -> &'static [u8] {
    &dummy().cs
}

fn bytecode(b: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const c_void, BytecodeLength: b.len() }
}

fn note_fallback(what: &str, hr: HRESULT, ok: bool) {
    let n = if ok { FALLBACK_PSO.fetch_add(1, Ordering::Relaxed) } else { FAILED_PSO.fetch_add(1, Ordering::Relaxed) };
    if n < 5 || n % 1000 == 0 {
        if ok {
            crate::xlog_warn!("D3D12.X: {what} rejected by the driver ({hr:#x}) – fallback shaders used (#{})", n + 1);
        } else {
            crate::xlog_error!("D3D12.X: {what} rejected ({hr:#x}), the fallback version too (#{})", n + 1);
        }
    }
}

type PsoFn = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;

unsafe fn finish_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if super::pso_async::preparing() || blocking_warm_refusing() {
        return E_INVALIDARG;
    }
    let t0 = std::time::Instant::now();
    let mut hr = f(this, desc, riid, ppv);
    if hr == E_NOINTERFACE {
        if blocking_warm_refusing() {
            return E_INVALIDARG;
        }
        let iid = pc_iid(Kind::PipelineState);
        hr = f(this, desc, &iid, ppv);
    }
    crate::gpu::recomp::dxc::note_driver(t0.elapsed());
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::PipelineState);
    }
    hr
}

/// The pipeline library's key (`pso_library`) of the pipeline built as driver blob `blob` names it
/// from `desc`: a hash of the translation (its era), the memory windows, the blob's name (with the
/// root signature's static samplers) and the exact description – every shader's bytes and all
/// state. None: one the library does not keep (no blob name, stream output).
unsafe fn library_key(blob: Option<&std::path::Path>, desc: &super::pso_library::Desc) -> Option<u64> {
    use super::pso_library::Desc;
    let mut key = crate::gpu::recomp::pso::translation_era().to_le_bytes().to_vec();
    key.extend_from_slice(&(crate::gpu::recomp::pso::mem_windows() as u64).to_le_bytes());
    key.extend_from_slice(blob?.file_name()?.to_str()?.as_bytes());
    match desc {
        Desc::Compute(d) => {
            key.push(b'c');
            key.extend_from_slice(shader_bytes(&d.CS)?);
            key.extend_from_slice(&d.NodeMask.to_le_bytes());
            key.extend_from_slice(&d.Flags.0.to_le_bytes());
        }
        Desc::Graphics(d) => {
            key.push(b'g');
            key.extend_from_slice(driver_cache::graphics_name(d, &[])?.as_bytes());
        }
        Desc::Stream(s) => {
            key.push(b's');
            key.extend_from_slice(driver_cache::stream_path(s, &[])?.file_name()?.to_str()?.as_bytes());
        }
    }
    Some(crate::gpu::shader::fnv64(&key))
}

/// The PSO the pipeline library (`pso_library`) holds under the key of `desc` built as driver blob
/// `blob` names it (`library_key`), into `ppv` (`adopted`: as `finish_pso` adopts one). None: no
/// library, or none there of that key; the build goes on as before.
unsafe fn library_load(this: *mut c_void, blob: Option<&std::path::Path>, desc: super::pso_library::Desc, riid: *const GUID, ppv: *mut *mut c_void, adopted: bool) -> Option<HRESULT> {
    if ppv.is_null() || super::pso_async::preparing() {
        return None;
    }
    let libraries = library_of(this)?;
    let key = library_key(blob, &desc)?;
    let Some(hr) = libraries.load(key, desc, riid.cast(), ppv) else {
        note_blocking_driver(|c| c.library_misses += 1);
        return None;
    };
    if adopted {
        adopt(*ppv, Kind::PipelineState);
    }
    note_blocking_driver(|c| c.library_hits += 1);
    Some(hr.0)
}

/// Stores `pso`, a translated pipeline built from `desc` as driver blob `blob` names it, in this
/// process's pipeline library under its key (`library_key`); none without a description.
unsafe fn library_store(this: *mut c_void, blob: Option<&std::path::Path>, desc: Option<super::pso_library::Desc>, pso: *mut c_void) {
    if pso.is_null() {
        return;
    }
    let Some(libraries) = library_of(this) else { return };
    let Some(key) = desc.and_then(|desc| library_key(blob, &desc)) else { return };
    if libraries.store(key, pso) {
        note_blocking_driver(|c| c.library_stored += 1);
    }
}

/// Device `this`'s pipeline libraries (`pso_library::of`), made by its driver's own
/// `CreatePipelineLibrary`.
unsafe fn library_of(this: *mut c_void) -> Option<Arc<super::pso_library::Libraries>> {
    if this.is_null() || !super::pso_library::on() || !driver_cache::on() {
        return None;
    }
    let device = this as usize;
    let create = move |blob: &[u8]| {
        type Create = unsafe extern "system" fn(*mut c_void, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
        let this = device as *mut c_void;
        let f: Create = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineLibrary")));
        let mut out = std::ptr::null_mut();
        let hr = f(this, if blob.is_empty() { std::ptr::null() } else { blob.as_ptr().cast() }, blob.len(), (&ID3D12PipelineLibrary::IID as *const windows::core::GUID).cast(), &mut out);
        if hr < 0 || out.is_null() {
            return Err(windows::core::HRESULT(hr));
        }
        Ok(ID3D12PipelineLibrary::from_raw(out))
    };
    super::pso_library::of(device, driver_cache::library_dir, Box::new(create), || note_blocking_driver(|c| c.library_refused += 1))
}

/// The pipeline library's counters for the caches line; None while it is off (`XCL_PSO_LIBRARY`).
pub fn pipeline_library_summary() -> Option<String> {
    super::pso_library::on().then(super::pso_library::summary)
}

/// Saves this process's new pipelines in its pipeline library shard at its end, waiting
/// `pso_library::EXIT_WAIT` for the libraries' locks at most.
pub fn save_pipeline_libraries() {
    super::pso_library::save_all(Some(std::time::Instant::now() + super::pso_library::EXIT_WAIT));
}

thread_local! {
    /// translation errors noted during a cached-only attempt (`attempt`)
    static ATTEMPT_NOTES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// the translation error this thread noted last (`why_untranslated`)
    static LAST_NOTE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Why each pipeline the title holds without a translated console shader has none (its CS and
/// the error noted when the translation or the build failed).
static UNTRANSLATED: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

pub(super) fn note_untranslated(pso: usize, why: String) {
    if pso != 0 {
        UNTRANSLATED.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, why);
    }
}

fn take_last_note() -> Option<String> {
    LAST_NOTE.with(|n| n.borrow_mut().take())
}

/// Why pipeline `pso` has no translated console shader, when XCL made it.
pub fn why_untranslated(pso: usize) -> Option<String> {
    UNTRANSLATED.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// Run a translated-pipeline path without compiling (`dxc::cached_only`): (its result, whether
/// a shader was not compiled yet). Errors it noted count only when nothing was missing.
fn attempt<R>(f: impl FnOnce() -> R) -> (R, bool) {
    let (r, missed) = crate::gpu::recomp::dxc::cached_only(f);
    let notes = ATTEMPT_NOTES.with(|n| std::mem::take(&mut *n.borrow_mut()));
    if !missed {
        notes.into_iter().for_each(note_translation_error);
    }
    (r, missed)
}

fn note_translation_error(e: String) {
    LAST_NOTE.with(|n| *n.borrow_mut() = Some(e.clone()));
    // (`xcl warm-up` logs its failures: no run of the title follows to log them)
    if super::pso_async::preparing() && WARMING_UP.with(|w| w.get()).is_none() {
        return;
    }
    if crate::gpu::recomp::dxc::in_cached_only() {
        ATTEMPT_NOTES.with(|n| n.borrow_mut().push(e));
        return;
    }
    // every failure counted (`COMPILE_FAILURES`, on the caches line); each different one logged
    // once (the first five only hid a retail game's "mesh pipeline with AS" behind five earlier
    // ones). A pipeline left without a translation keeps its stand-in: nothing builds it again.
    COMPILE_FAILURES.fetch_add(1, Ordering::Relaxed);
    let mut g = TRANSLATE_ERRORS.lock().unwrap();
    if g.len() < 400 && !g.contains(&e) {
        crate::xlog_error!("D3D12.X: compile failed: {e}");
        g.push(e);
        return;
    }
    drop(g);
    // the same failure in another pipeline (the same shaders in another state): counted above,
    // and logged as its count doubles, so the failures logged add up to the ones counted
    let mut again = REPEATED_ERRORS.lock().unwrap();
    let n = again.get_or_insert_with(HashMap::new).entry(e.clone()).or_insert(1);
    *n += 1;
    if n.is_power_of_two() {
        crate::xlog_error!("D3D12.X: compile failed {n} times so far: {e}");
    }
}

/// how many pipelines each failure logged came up in again (`note_translation_error`)
static REPEATED_ERRORS: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// translations, DXC compiles and driver pipeline builds that failed in the run
static COMPILE_FAILURES: AtomicU64 = AtomicU64::new(0);

unsafe fn shader_bytes<'a>(b: &D3D12_SHADER_BYTECODE) -> Option<&'a [u8]> {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    (p != 0 && n > 0 && crate::gpu::shader::readable(p, n)).then(|| std::slice::from_raw_parts(p as *const u8, n))
}

/// Root signature of every pipeline built from translated console shaders (static samplers
/// make it per pipeline): the draw binds it instead of the title's.
static PSO_RS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

fn note_pso_rs(pso: *mut c_void, rs: *mut c_void) {
    if !pso.is_null() {
        PSO_RS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso as usize, rs as usize);
    }
}

/// user SGPRs and the console's `numthreads` of the console CS of a translated compute pipeline
static PSO_USER_SGPRS: Mutex<Option<HashMap<usize, (u32, [u32; 3])>>> = Mutex::new(None);

/// How many user SGPRs the console compute shader of a translated pipeline has (None: not
/// known). What is above the driver's four holds root arguments (`console_list::inline_for`).
pub fn pso_user_sgprs(pso: usize) -> Option<u32> {
    let pso = super::pso_async::resolve(pso);
    PSO_USER_SGPRS.lock().unwrap().as_ref()?.get(&pso).map(|e| e.0)
}

/// The console's group shape (`numthreads`) of a translated compute pipeline (None: not known).
pub fn pso_threads(pso: usize) -> Option<[u32; 3]> {
    let pso = super::pso_async::resolve(pso);
    PSO_USER_SGPRS.lock().unwrap().as_ref()?.get(&pso).map(|e| e.1)
}

/// The root signature a translated pipeline state was created with.
pub fn pso_root_signature(pso: usize) -> Option<usize> {
    let pso = super::pso_async::resolve(pso);
    PSO_RS.lock().unwrap().as_ref()?.get(&pso).copied()
}

/// ABI root signatures with static samplers, by sampler set (owned here, never released).
static ABI_RS_SAMPLERS: Mutex<Option<HashMap<Vec<u32>, usize>>> = Mutex::new(None);

/// The XCL ABI root signature with a pipeline's static samplers (the plain one without).
unsafe fn abi_rs_for(this: *mut c_void, samplers: &[crate::gpu::recomp::pso::StaticSampler]) -> Option<*mut c_void> {
    if samplers.is_empty() {
        return abi_rs_raw(this);
    }
    let key: Vec<u32> = samplers.iter().flat_map(|(w, c)| w.iter().copied().chain([*c as u32])).collect();
    let mut g = ABI_RS_SAMPLERS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&rs) = map.get(&key) {
        return Some(rs as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature_with(dev, true, samplers) {
        Ok(rs) => {
            let raw = rs.into_raw();
            map.insert(key, raw as usize);
            Some(raw)
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature with static samplers: {e}"));
            None
        }
    }
}

/// XCL ABI root signature, created once on the backend device.
unsafe fn abi_rs_raw(this: *mut c_void) -> Option<*mut c_void> {
    let cur = ABI_RS.load(Ordering::Acquire);
    if cur != 0 {
        return Some(cur as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature(dev, true) {
        Ok(rs) => {
            let raw = rs.into_raw();
            match ABI_RS.compare_exchange(0, raw as usize, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => Some(raw),
                Err(other) => {
                    drop(ID3D12RootSignature::from_raw(raw));
                    Some(other as *mut c_void)
                }
            }
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature: {e}"));
            None
        }
    }
}

/// A graphics pipeline whose VS (and PS) are console shaders: recompiled VS/PS + the XCL
/// shader ABI root signature. None: not console shaders, or not translatable (the caller
/// goes on with the title's description and the stand-in fallback).
/// A compute pipeline with a console CS: `None` = not a console shader, `Some(Err)` = console
/// shader the recompiler could not take, `Some(Ok(hr))` = translated pipeline created (or the
/// driver's verdict on it).
/// DXIL size from which a translated pipeline's driver compile goes to the background
const BIG_DRIVER_COMPILE: usize = 256 << 10;

/// Driver compiles of translated programs, within the memory left: a run's background builds and
/// `xcl warm-up` alike. An NVIDIA driver took 1.4 – 1.6 GB of process memory for each of a
/// retail game's 1 MB+ DXIL programs (40 – 180 s each), and a warm-up that started them on every
/// core took 25 GB of a 32 GB PC. A background compile starts only while the physical memory
/// available (read again each time it looks, so the game's own growth counts), less the full
/// expected peaks of those running (they may still be growing), stays over `KEEP_FREE` after
/// its own expected peak (`peak`), giants (`GIANT_DXIL`) alike; within `most` at once and the
/// process's private memory (with the running peaks) within `cap` when given. Every compile also
/// needs the system's commit headroom to hold the running peaks, its own and `URGENT_FLOOR`. A
/// compile a draw or dispatch waits for (`waited_on`, asked again while it waits) goes ahead of
/// background work when its peak leaves `URGENT_FLOOR` available. Current
/// memory pressure is not permanent: it waits regardless of whether other compiles are active.
struct Governor {
    state: Mutex<GovState>,
    cv: std::sync::Condvar,
    #[cfg(test)]
    test_ram: Option<(AtomicU64, u64)>,
}

struct GovState {
    running: usize,
    started: u64,
    giants: usize,
    /// compiles a bind waits for among `running`
    urgent: usize,
    /// compiles a bind waits for, waiting for memory (`URGENT_FLOOR`)
    starving: usize,
    /// the expected peaks of those running
    reserved: u64,
    /// the expected peaks of the giants a blocking warm-up waits to start (`admit` keeps them room)
    giant_waits: Vec<u64>,
    /// giants finished so far (a giant that gave way asks for room again after one)
    giants_finished: u64,
    idle_breaker: super::pso_sched::IdleBreaker,
    /// the last live demand (`live_priority`)
    live_at: Option<std::time::Instant>,
    /// the compiles running: (`GovGuard::started`, expected peak, started at, expected time, a cold giant)
    runs: Vec<(u64, u64, std::time::Instant, std::time::Duration, bool)>,
}

impl GovState {
    fn giant_waiting(&self) -> u64 {
        self.giant_waits.iter().copied().min().unwrap_or(0)
    }

    fn live_recent(&self) -> bool {
        live_priority() && self.live_at.is_some_and(|at| at.elapsed() < super::pso_sched::LIVE_QUIET)
    }

    /// (`pso_sched::Load::settled`)
    fn settled(&self) -> u64 {
        if !live_priority() {
            return 0;
        }
        self.runs.iter().filter(|(_, _, at, expected, _)| at.elapsed() >= *expected).map(|(_, cost, _, _, _)| cost).sum()
    }

    /// (`pso_sched::Load::giant_reserved`)
    fn giant_reserved(&self) -> u64 {
        self.runs.iter().filter(|run| run.4).map(|run| run.1).sum()
    }

    fn stop_waiting(&mut self, giant: bool, cost: u64) {
        if let Some(i) = self.giant_waits.iter().position(|&c| giant && c == cost) {
            self.giant_waits.swap_remove(i);
        }
    }
}

/// The waited-for compiles holding for a driver slot or memory: (thread, DXIL bytes, estimate,
/// since, which pipeline: `building`), for `log_slot_wait`.
static SLOT_WAITS: Mutex<Vec<(std::thread::ThreadId, usize, u64, std::time::Instant, String)>> = Mutex::new(Vec::new());

/// A waited-for compile in `SLOT_WAITS` until it starts or gives up.
struct SlotWait(std::thread::ThreadId);

impl SlotWait {
    fn new(id: std::thread::ThreadId, dxil_len: usize, cost: u64) -> Self {
        SLOT_WAITS.lock().unwrap_or_else(|e| e.into_inner()).push((id, dxil_len, cost, std::time::Instant::now(), building()));
        Self(id)
    }
}

impl Drop for SlotWait {
    fn drop(&mut self) {
        SLOT_WAITS.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != self.0);
    }
}

/// Every 10 s while a waited-for compile has waited 10 s or more: each waiter's estimate, its
/// multiplier over the plain curve (`peak`) and its wait; each running compile's estimate and age
/// against its expected time (play-95's waits of 250 to 281 s could not be read from the log).
fn log_slot_wait(s: &GovState, available: u64) -> Option<String> {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let ten = std::time::Duration::from_secs(10);
    let waits = SLOT_WAITS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if waits.iter().all(|w| w.3.elapsed() < ten) {
        return None;
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_some_and(|t| t.elapsed() < ten) {
        return None;
    }
    *last = Some(std::time::Instant::now());
    let waiting: Vec<String> = waits
        .iter()
        .map(|(_, len, cost, at, who)| format!("{who}: {} KB est {} MiB (x{:.2}) for {} s", len >> 10, cost >> 20, *cost as f64 / peak(*len).max(1) as f64, at.elapsed().as_secs()))
        .collect();
    let running: Vec<String> = s
        .runs
        .iter()
        .map(|&(_, cost, at, expected, giant)| format!("est {} MiB, {} s of ~{} s{}", cost >> 20, at.elapsed().as_secs(), expected.as_secs(), if giant { ", cold giant" } else { "" }))
        .collect();
    Some(format!(
        "D3D12.X: driver slot wait: {} waited-for compiles held [{}]; {} running [{}]: {} MiB reserved, {} MiB of it past its expected time, compile heap rising {} MB/s; {} MiB available",
        waits.len(),
        waiting.join("; "),
        s.running,
        running.join("; "),
        s.reserved >> 20,
        s.settled() >> 20,
        COMPILE_RISE.load(Ordering::Relaxed) >> 20,
        available >> 20
    ))
}

thread_local! {
    /// the title's name for the pipeline this thread builds (`build_named`)
    static BUILD_NAME: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// `f` builds stand-in `standin`'s pipeline: the governor's lines name it by the title's name.
fn build_named<R>(standin: usize, f: impl FnOnce() -> R) -> R {
    let name = PSO_NAMES.lock().unwrap().as_ref().and_then(|names| names.get(&standin).cloned());
    BUILD_NAME.with(|n| *n.borrow_mut() = name);
    let r = f();
    BUILD_NAME.with(|n| n.borrow_mut().take());
    r
}

/// The pipeline this thread builds, for the governor's lines: its manifest key, its shaders' key
/// (`live_stall::job`) and the title's name for it.
fn building() -> String {
    let (pipeline, shader) = crate::live_stall::job();
    let name = BUILD_NAME.with(|n| n.borrow().as_ref().map(|n| format!(" '{n}'")).unwrap_or_default());
    format!("pipeline {pipeline:016x}, shader {shader:016x}{name}")
}

thread_local! {
    /// why this thread's last admission held its compile (`pso_sched::admit_why`)
    static HOLD_REASON: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Why a driver compile is held, once per pipeline (`live_stall::job` names it; one outside a
/// pipeline build, once per DXIL size): the first 2,000 held, then every 100th (play-95's
/// starvation of 2 KB loading passes showed only in totals).
fn log_hold(dxil_len: usize, cost: u64, wanted: bool, available: u64, s: &GovState) -> Option<String> {
    static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
    static N: AtomicUsize = AtomicUsize::new(0);
    let job = crate::live_stall::job();
    let seen = if job == (0, 0) { (u64::MAX, dxil_len as u64) } else { job };
    if !SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(seen) {
        return None;
    }
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n > 2_000 && n % 100 != 0 {
        return None;
    }
    Some(format!(
        "D3D12.X: driver compile held ({}, {} KB of DXIL, est {} MiB, {}): {}; {} MiB available, {} MiB reserved, {} MiB of it past its expected time, {} running (#{n})",
        building(),
        dxil_len >> 10,
        cost >> 20,
        if wanted { "waited for" } else { "background" },
        HOLD_REASON.with(std::cell::Cell::get),
        available >> 20,
        s.reserved >> 20,
        s.settled() >> 20,
        s.running
    ))
}

/// Lines a governor wait logs, written once its lock is let go (`acquire_within`): for the log,
/// and as warnings (a warm-up child's, which reach the warm-up's own log on its stderr).
#[derive(Default)]
struct HeldLines(Vec<String>, Vec<String>);

impl HeldLines {
    fn flush(&mut self) {
        for line in self.0.drain(..) {
            crate::xlog_info!("{line}");
        }
        for line in self.1.drain(..) {
            crate::xlog_warn!("{line}");
        }
    }
}

/// `XCL_WARM_PEER_GIANTS_NET` (default 1): in a blocking warm-up, the cold giants the other
/// children run count against a compile for what they have not taken yet – their estimates less
/// what their processes' private bytes grew by since each started, which the memory available
/// already shows. 0: for their whole estimates on top of that, as before.
fn peer_giants_net() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_WARM_PEER_GIANTS_NET").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: the other warm-up children's cold giants count for their whole estimates (XCL_WARM_PEER_GIANTS_NET=0)");
        }
        on
    })
}

/// This process's private bytes when the first of its cold giants running now started
/// (`warm_peers::Held::publish_giants`).
static GIANT_BASE: AtomicU64 = AtomicU64::new(0);

/// What a warm-up child's held compile is held against (`acquire_within`): the floor with the
/// pass's margin, the other children's reservations and how much of them counts, and the
/// largest cold giant another child runs.
fn warm_hold_detail(floor: Option<u64>, gross: u64, counted: u64, giants: &super::warm_peers::PeerGiants) -> String {
    let margin = if WARM_UP_PROCESS.load(Ordering::Relaxed) { super::warm_children::governor_margin() } else { 0 };
    let base = floor.unwrap_or_else(keep_free).max(URGENT_FLOOR);
    let largest = giants.largest.map_or_else(
        || "no cold giant in another child".to_string(),
        |(slot, pid, est, used)| format!("the largest cold giant is child slot {slot}'s (pid {pid}): est {} MiB, {} MiB of it in use", est >> 20, used >> 20),
    );
    format!(
        "floor {} MiB ({} MiB and the pass's margin {} MiB); the other children reserve {} MiB, {} MiB of it counted ({} MiB of their cold giants' {} MiB in use already{}); {largest}",
        (base + margin) >> 20,
        base >> 20,
        margin >> 20,
        gross >> 20,
        counted >> 20,
        giants.used >> 20,
        giants.reserved >> 20,
        if peer_giants_net() { "" } else { ", counted whole: XCL_WARM_PEER_GIANTS_NET=0" }
    )
}

impl Drop for HeldLines {
    fn drop(&mut self) {
        self.flush();
    }
}

struct GovGuard {
    cost: u64,
    giant: bool,
    urgent: bool,
    available_at_idle_break: Option<u64>,
    started: u64,
    already_overlapping: bool,
    commit_measure: Option<GiantCommitMeasure>,
    _running: crate::progress::Task,
    _large: Option<crate::progress::Task>,
    live: Option<crate::live_stall::Span>,
}

struct GiantCommitMeasure {
    shader: String,
    dxil_len: usize,
    baseline: u64,
    began: std::time::Instant,
}

#[derive(Default)]
struct CommitPeak {
    bytes: u64,
    samples: usize,
}

impl CommitPeak {
    fn record(&mut self, bytes: u64) {
        if bytes != 0 {
            self.bytes = self.bytes.max(bytes);
            self.samples += 1;
        }
    }

    fn growth(&self, baseline: u64) -> Option<u64> {
        (baseline != 0 && self.samples != 0).then(|| self.bytes.saturating_sub(baseline))
    }
}

static GIANT_COMMIT: (Mutex<Vec<(u64, CommitPeak)>>, std::sync::Condvar) = (Mutex::new(Vec::new()), std::sync::Condvar::new());

impl GiantCommitMeasure {
    fn start(id: u64, shader: String, dxil_len: usize) -> Self {
        static START: std::sync::Once = std::sync::Once::new();
        START.call_once(|| {
            let _ = std::thread::Builder::new().name("xcl-giant-commit".into()).spawn(|| loop {
                let (active, ready) = &GIANT_COMMIT;
                let mut active = ready.wait_while(active.lock().unwrap(), |active| active.is_empty()).unwrap();
                let bytes = crate::util::process_commit_bytes();
                for (_, peak) in active.iter_mut() {
                    peak.record(bytes);
                }
                drop(active);
                std::thread::sleep(std::time::Duration::from_millis(50));
            });
        });
        let (active, ready) = &GIANT_COMMIT;
        let mut active = active.lock().unwrap();
        let baseline = crate::util::process_commit_bytes();
        active.push((id, CommitPeak::default()));
        ready.notify_one();
        Self { shader, dxil_len, baseline, began: std::time::Instant::now() }
    }

    fn finish(id: u64) -> CommitPeak {
        let mut active = GIANT_COMMIT.0.lock().unwrap();
        let Some(index) = active.iter().position(|(key, _)| *key == id) else { return CommitPeak::default() };
        let (_, mut peak) = active.swap_remove(index);
        peak.record(crate::util::process_commit_bytes());
        peak
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DriverDemand {
    Background,
    /// a skipped draw waits for it (`live_priority`)
    Live,
    Wanted,
    Blocking,
}

/// Live compiles go ahead of background ones (`XCL_LIVE_PRIORITY=0`: off): a build a skipped draw
/// waits for asks for its driver slot as one a bind waits for (the plain rule, ahead of every
/// background compile), no new background giant starts within `pso_sched::LIVE_QUIET` of live
/// demand, and the background workers take no new job while a live compile waits for a slot
/// (`live_compile_waiting`).
fn live_priority() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_LIVE_PRIORITY").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: live compiles have no priority over background ones (XCL_LIVE_PRIORITY=0)");
        }
        on
    })
}

/// A live compile waits for a driver slot: a background worker takes no new job until it starts.
pub(super) fn live_compile_waiting() -> bool {
    live_priority() && DRIVER_COMPILES.state.lock().unwrap().starving != 0
}

/// Live compiles in two lanes (`XCL_LIVE_LANES=0`: off): one that is not a cold giant never waits
/// on a cold giant's reservation, only on the memory available less 1 GB (`pso_sched::Ask::small_lane`);
/// cold giants are reserved at their corrected estimate (`pso_sched::cost`), and one a bind-queue
/// worker cannot start now goes to the giant lane's one thread instead of holding the worker
/// (`pso_async`'s `GIANT_LANE`); a compile from a kept driver blob is reserved at the small rate.
pub(super) fn live_lanes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = live_priority() && std::env::var("XCL_LIVE_LANES").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: no live compile lanes (XCL_LIVE_LANES=0 or XCL_LIVE_PRIORITY=0)");
        }
        on
    })
}

/// The cold giants' correction (`pso_sched::cost`).
const GIANT_FACTOR: u64 = 3;

thread_local! {
    /// the driver compile asking on this thread loads a kept driver blob (`driver_slot`)
    static KEPT_BLOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// the driver compile asking on this thread runs in a giant helper reserved at this
    /// (`giant_helper::build`): its job object's memory limit
    static HELPER_COST: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// The memory a driver compile asking on this thread is admitted and reserved at, and whether
/// it is a cold giant (`live_lanes`).
fn compile_cost(dxil_len: usize) -> (u64, bool) {
    if let Some(cost) = HELPER_COST.with(|c| c.get()) {
        return (cost, true);
    }
    if !live_lanes() {
        return (peak(dxil_len), false);
    }
    let kept = KEPT_BLOB.with(|k| k.get());
    (super::pso_sched::cost(dxil_len, kept, GIANT_FACTOR), !kept && dxil_len >= super::pso_sched::GIANT_DXIL)
}

/// What a compile leaves of the physical memory when nothing else is said: the user's rule
/// (2026-11-26 15:22Z): every compile is admitted when its expected peak fits in the memory
/// available less 1 GB, and waits otherwise.
const KEEP_FREE: u64 = URGENT_FLOOR;
/// Below this much available the files mapped into the process are trimmed (`trim_mapped_files`).
const TRIM_BELOW: u64 = 6 << 30;
/// The memory the governor keeps available now: `KEEP_FREE`, or the floor of `warm_up_giants` while it runs
static KEEP_FREE_NOW: AtomicU64 = AtomicU64::new(KEEP_FREE);

fn keep_free() -> u64 {
    KEEP_FREE_NOW.load(Ordering::Relaxed)
}
/// Memory a compile a bind waits for still leaves available: one too big for it keeps its
/// stand-in (a 27 MB program took an NVIDIA driver past 31 GB of a 32 GB PC).
pub(super) const URGENT_FLOOR: u64 = 1 << 30;
/// Why `Governor::acquire` gave no slot
#[derive(Clone, Copy, Debug)]
enum Refused {
    /// a warm-up's compile that did not fit for 10 s with nothing else running: it never will
    TooBig,
    /// the compile cannot fit even with all physical RAM available
    NoMemory,
    /// a warm-up's compile that did not fit for 10 s with nothing else running, but would in this
    /// PC's memory less `KEEP_FREE` and the warm-up's own: tried again at the end of its phase
    Later,
    /// a giant on a warm-up thread leaving them for later (`LEAVE_GIANTS`)
    Giant,
    /// a run's background compile nothing waits for that does not fit now: its build is put off
    /// and its worker goes on with the next (`pso_async::put_off`)
    NoRoom,
    /// a run's compile a bind waits for that would not fit even if every running compile ended
    /// (`pso_sched::never_fits_now`) for `pso_sched::NEVER_FIT_GRACE`: refused rather than holding its thread
    NeverFits,
    /// a warm-up's compile not started because the warm-up was cancelled (`cancel_warm_up`)
    Cancelled,
}

static HEADROOM: Mutex<(Option<std::time::Instant>, super::pso_sched::Headroom, super::pso_sched::TitleReserve, u64)> =
    Mutex::new((None, super::pso_sched::Headroom::empty(), super::pso_sched::TitleReserve::empty(), 0));

/// Count the full coherent window because it is precommitted, and the live, not pooled,
/// tiled memory. CPU shadows of tiles, VRAM arenas and physical-page VA aliases are not
/// counted a second time. Untracked title allocations keep the estimate conservative.
#[cfg(not(test))]
pub(super) fn sample_background_headroom() {
    let backed = crate::gpu::uma::get().map_or(0, |u| u.size())
        + super::sparse::pool_bytes().1
        + super::vram::page_bytes().0
        + crate::hle::kernelx::xmem_heap_bytes();
    let commit = crate::util::process_commit_bytes();
    let mut sampled = HEADROOM.lock().unwrap_or_else(|e| e.into_inner());
    let (start, headroom, reserve, backed_sample) = &mut *sampled;
    let now = start.get_or_insert_with(std::time::Instant::now).elapsed().as_millis() as u64;
    let loading = super::pso_async::title_loading();
    headroom.note(now, commit, loading);
    reserve.note(now, super::pso_sched::TitleMemory { backed: backed as u64, commit }, title_budget(), loading);
    *backed_sample = backed as u64;
}

/// The run's floor for speculative builds (`background_memory`). An actual bind wait uses the
/// emergency floor and the recent commit growth.
fn background_floor() -> u64 {
    background_floor_when(false)
}

fn title_budget() -> u64 {
    crate::state::config().title_memory_budget_mb().saturating_mul(1 << 20)
}

pub(super) fn background_memory_summary() -> String {
    let (floor, bind_floor, reserve, backed, steady) = background_memory();
    format!(
        "{} MB kept available ({} MB for parked peers during a bind wait), {} MB of commit headroom with {} MB kept; title budget {} of {} MB backed, {} MB projected host commit{}",
        floor >> 20, bind_floor >> 20, available_commit().min(u64::MAX >> 1) >> 20, URGENT_FLOOR >> 20,
        backed >> 20, title_budget() >> 20, reserve >> 20,
        if steady { " (steady)" } else { "" },
    )
}

/// `background_floor` while a bind waits for a pipeline (`pso_sched::Headroom::floor`) or not.
fn background_floor_when(bind_waits: bool) -> u64 {
    let (background, bind, _, _, _) = background_memory();
    if bind_waits { bind } else { background }
}

/// (the floor of speculative builds, the floor while a bind waits, the title's projected host
/// commit, the console bytes backed, that projection is steady). Speculative builds keep
/// `URGENT_FLOOR` of the physical memory (`XCL_PSO_BACKGROUND_FLOOR_MB`), as every compile
/// does (`KEEP_FREE`); the commit limit is kept by `pso_sched::admit` (`Room::commit_room`).
/// (A 14.4 GB projection of the title's host commit held back as available RAM made an 18.9 GB
/// floor that parked 1 850 of a retail game's 6 073 background builds with 7.5 – 9 GB
/// available, and its own threads compiled what it bound, ~30 s each.)
fn background_memory() -> (u64, u64, u64, u64, bool) {
    static KEEP: OnceLock<u64> = OnceLock::new();
    let keep = *KEEP.get_or_init(|| {
        std::env::var("XCL_PSO_BACKGROUND_FLOOR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(URGENT_FLOOR, |mb| mb.saturating_mul(1 << 20)).max(URGENT_FLOOR)
    });
    let budget = title_budget();
    let sampled = HEADROOM.lock().unwrap_or_else(|e| e.into_inner());
    let (start, _, reserve, backed) = &*sampled;
    let effective = if start.is_some() { reserve.effective() } else { budget };
    (
        keep,
        keep,
        effective,
        *backed,
        reserve.is_steady(),
    )
}

/// `XCL_PSO_SMALL_PEAK_PER_BYTE`: the memory expected for each byte of DXIL of a driver compile
/// under the giant line (`pso_sched::set_small_peak`), read once.
fn tune_peaks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(per_byte) = std::env::var("XCL_PSO_SMALL_PEAK_PER_BYTE").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            super::pso_sched::set_small_peak(per_byte);
            crate::xlog_info!("D3D12.X: a driver compile under 1 MB of DXIL is expected to take {} B for each byte (XCL_PSO_SMALL_PEAK_PER_BYTE)", per_byte.clamp(256, 4096));
        }
    });
}

/// Physical memory installed (0 when the system does not say).
fn total_physical() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return 0;
    }
    m.ullTotalPhys
}

/// compiles a bind waited for, refused (`Refused::NoMemory`, `Refused::NeverFits`): their pipelines keep the stand-in
static CANNOT_BUILD: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// the memory available and the reservations when this thread's compile was refused as one
    /// that never fits (`Refused::NeverFits`)
    static NEVER_FITS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}
pub(crate) use super::pso_sched::GIANT_DXIL;
use super::pso_sched::{peak, GIANTS_AT_ONCE};
/// background driver compiles at once at most; under `GIANT_DXIL` the memory left decides below that
const SMALL_AT_ONCE: usize = 20;
/// threads the title keeps for itself beside a run's background driver compiles
const TITLE_THREADS: usize = 4;

/// Physical memory available (zero when the system does not say).
pub fn available_physical() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return 0;
    }
    m.ullAvailPhys
}

/// The system's commit limit less its commit charge (u64::MAX when the system does not say).
fn available_commit() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return u64::MAX;
    }
    m.ullAvailPageFile
}

/// `available_commit`, after one try to commit `need` bytes when less is left, at most every 2 s
/// and only while no compile runs (`idle`): a system-managed page file grows only when something
/// asks for more than the commit limit, and nothing does while every compile waits for room.
fn commit_room(need: u64, idle: bool) -> u64 {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    static TRIED: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let room = available_commit();
    if room >= need || !idle {
        return room;
    }
    {
        let mut tried = TRIED.lock().unwrap_or_else(|e| e.into_inner());
        if tried.is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(2)) {
            return room;
        }
        *tried = Some(std::time::Instant::now());
    }
    let p = unsafe { VirtualAlloc(std::ptr::null(), need as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
    if p.is_null() {
        return room;
    }
    unsafe { VirtualFree(p, 0, MEM_RELEASE) };
    available_commit()
}

/// This process is a warm-up (in-process or a child): no title grows beside its compiles.
static WARM_UP_PROCESS: AtomicBool = AtomicBool::new(false);

/// How fast the title is taking memory now: the fastest recent fall of the memory available and
/// of the commit room (pso_sched::Trend), in bytes a second, with this sample of the memory
/// available noted. None in a warm-up process, where only its own compiles move them, and those
/// are reserved at their full peaks already. Also the faster of the two whatever
/// `XCL_ADMISSION_HEADROOM` says: while it is `pso_sched::SETTLE_FALL` or more, a compile past its
/// expected time may still be growing (`pso_sched::Room::trend_fall`).
#[cfg(not(test))]
fn title_growth(available: u64) -> ((u64, u64), u64) {
    use super::pso_sched::Trend;
    static TRENDS: Mutex<(Trend, Trend)> = Mutex::new((Trend::new(), Trend::new()));
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var("XCL_ADMISSION_HEADROOM").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: no memory headroom for the title's growth (XCL_ADMISSION_HEADROOM=0)");
        }
        on
    });
    if WARM_UP_PROCESS.load(Ordering::Relaxed) {
        return ((0, 0), 0);
    }
    let at = START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64();
    let commit = available_commit();
    let mut trends = TRENDS.lock().unwrap_or_else(|e| e.into_inner());
    trends.0.note(at, available);
    if commit != u64::MAX {
        trends.1.note(at, commit);
    }
    let rates = (trends.0.fall_rate(), trends.1.fall_rate());
    (if on { rates } else { (0, 0) }, rates.0.max(rates.1))
}

/// The last measured rise of the process heap less XCL's Rust heap (`compile_rise`), bytes a second.
static COMPILE_RISE: AtomicU64 = AtomicU64::new(0);

/// How fast the process heap less XCL's Rust heap rises now (`memdiag::compile_heap`, sampled at
/// most every half second): the driver compiles running still taking memory, which neither their
/// overdue reservations (`pso_sched::Load::settled`) nor the title's trend tell. 0 in a warm-up
/// process and with `XCL_COMPILE_RISE=0`. In play-98 a giant past its expected time took the
/// memory available from 2,623 to 781 MB in ~40 s while nothing held the floor.
fn compile_rise() -> u64 {
    #[cfg(test)]
    return 0;
    #[cfg(not(test))]
    {
        use super::pso_sched::Trend;
        static ON: OnceLock<bool> = OnceLock::new();
        static TREND: Mutex<(Trend, Option<std::time::Instant>)> = Mutex::new((Trend::new(), None));
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let on = *ON.get_or_init(|| {
            let on = std::env::var("XCL_COMPILE_RISE").map_or(true, |v| v.trim() != "0");
            if !on {
                crate::xlog_info!("D3D12.X: admissions keep no memory for the measured growth of the compiles running (XCL_COMPILE_RISE=0)");
            }
            on
        });
        if !on || WARM_UP_PROCESS.load(Ordering::Relaxed) {
            return 0;
        }
        let mut trend = TREND.lock().unwrap_or_else(|e| e.into_inner());
        if trend.1.map_or(true, |at| at.elapsed() >= std::time::Duration::from_millis(500)) {
            trend.1 = Some(std::time::Instant::now());
            let at = START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64();
            // (a rise is a fall of what is left below a fixed line: `Trend` measures falls)
            trend.0.note(at, (1u64 << 62).saturating_sub(crate::memdiag::compile_heap()));
            COMPILE_RISE.store(trend.0.fall_rate(), Ordering::Relaxed);
        }
        COMPILE_RISE.load(Ordering::Relaxed)
    }
}

/// Cold giants in a game follow `pso_sched::Ask::hard_giant` (`XCL_HARD_GIANTS=0`: off).
fn hard_giants() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_HARD_GIANTS").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: cold giants in the game may start beside each other and on the curve (XCL_HARD_GIANTS=0)");
        }
        on
    })
}

/// Logs the headroom an admission kept for the title's growth, at most every 10 s.
fn log_growth_headroom(room: &super::pso_sched::Room, dxil_len: usize, verdict: super::pso_sched::Verdict) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(10)) {
        return;
    }
    *last = Some(std::time::Instant::now());
    let (growth, commit_growth) = super::pso_sched::growth(room, dxil_len);
    let giants = if room.fall >= super::pso_sched::FAST_FALL { "; new background giants paused" } else { "" };
    crate::xlog_info!(
        "D3D12.X: memory headroom for the title's growth: available falling {} MB/s, commit room {} MB/s; a compile of {} KB of DXIL (~{} s) keeps {} MB more available and {} MB more commit ({} MB available, {:?}){giants}",
        room.fall >> 20, room.commit_fall >> 20, dxil_len >> 10, super::pso_sched::compile_secs(dxil_len), growth >> 20, commit_growth >> 20, room.available >> 20, verdict
    );
}

pub(super) fn background_memory_paused() -> bool {
    #[cfg(test)]
    return false;
    #[cfg(not(test))]
    return available_physical() < URGENT_FLOOR + peak(0);
}

/// Take the pages of files mapped into this process out of its working set (`VirtualUnlock` of a
/// range not locked). An NVIDIA driver maps its shader cache (DXCache `*.nvph`) and its lookups
/// leave most of it resident: 8.8 GB of a warm-up's 10 GB working set, with 1 GB private. Windows
/// does not count those pages as available until they are trimmed, so the 6 GB floor held back
/// compiles the PC had memory for (one at a time with 7 GB "available"; 13 GB after a trim). Clean
/// pages go to the standby list: available, and read back without a disk read if touched again.
/// Only file-backed views are trimmed, never the sections that back the title's memory. At most
/// every second; logged when it frees 1 GB or more (at most once a minute).
fn trim_mapped_files() {
    use windows_sys::Win32::System::Memory::{VirtualQuery, VirtualUnlock, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_MAPPED};
    use windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    static LAST: Mutex<Option<(std::time::Instant, Option<std::time::Instant>)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap();
    if last.is_some_and(|(t, _)| t.elapsed() < std::time::Duration::from_secs(1)) {
        return;
    }
    let before = available_physical();
    let (mut at, mut regions) = (0usize, 0usize);
    loop {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
            break;
        }
        let mut name = [0u16; 8];
        if m.State == MEM_COMMIT && m.Type == MEM_MAPPED && unsafe { K32GetMappedFileNameW(GetCurrentProcess(), m.BaseAddress, name.as_mut_ptr(), name.len() as u32) } != 0 {
            unsafe { VirtualUnlock(m.BaseAddress, m.RegionSize) };
            regions += 1;
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    let freed = available_physical().saturating_sub(before);
    let logged = last.and_then(|l| l.1);
    let log = freed >= 1 << 30 && logged.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(60));
    if log {
        crate::xlog_info!(
            "D3D12.X: {} MB of files mapped into XCL (its {regions} file views: the driver's shader cache) taken out of its working set; {} MB available now",
            freed >> 20,
            available_physical() >> 20
        );
    }
    *last = Some((std::time::Instant::now(), if log { Some(std::time::Instant::now()) } else { logged }));
}

impl Governor {
    fn physical_total(&self) -> u64 {
        #[cfg(test)]
        if let Some((_, total)) = self.test_ram.as_ref() {
            return *total;
        }
        total_physical()
    }

    fn available(&self) -> u64 {
        #[cfg(test)]
        if let Some((available, _)) = self.test_ram.as_ref() {
            return available.load(Ordering::Relaxed);
        }
        available_physical()
    }

    /// A slot for a driver compile of `dxil_len` bytes of DXIL, once it fits the limits. With
    /// `give_up`, one that did not fit for 10 s with no other compile running: `Refused::TooBig`
    /// when its peak is more than the PC's memory less `KEEP_FREE` and this process's private
    /// memory, so it never will (a 27 MB program's estimate is more than the PC has; started anyway
    /// it took an NVIDIA driver past 31 GB, leaving under 1 GB free), else `Refused::Later` (other
    /// processes hold the memory now). `Refused::NoMemory`: waited for, but
    /// its peak would leave under `URGENT_FLOOR` available with no other compile running.
    fn acquire(&self, dxil_len: usize, most: usize, cap: Option<u64>, give_up: bool, waited_on: impl Fn() -> bool) -> Result<GovGuard, Refused> {
        self.acquire_within(dxil_len, most, cap, give_up, |_| None, true, || if waited_on() { DriverDemand::Wanted } else { DriverDemand::Background })
    }

    /// Whether a compile of `dxil_len` bytes of DXIL starts now (`pso_sched::admit`), with the
    /// memory read now (after a trim of the files mapped in, when it is short). `floor`: what a
    /// background compile leaves available (`None`: `keep_free`).
    /// `own_starving`: the asker counts in `s.starving`. `others`: the compiles the other children
    /// of a blocking warm-up run (`warm_peers`), counted with this process's own against the
    /// memory and the giant cap.
    #[allow(clippy::too_many_arguments)]
    fn verdict(&self, s: &GovState, own_starving: bool, dxil_len: usize, wanted: bool, most: usize, cap: Option<u64>, floor: Option<u64>, others: super::warm_peers::Usage) -> (super::pso_sched::Verdict, u64) {
        let load = super::pso_sched::Load {
            running: s.running,
            giants: s.giants + others.giants,
            urgent: s.urgent,
            starving: s.starving - own_starving as usize,
            reserved: s.reserved.saturating_add(others.reserved),
            giant_waiting: s.giant_waiting(),
            live_recent: s.live_recent(),
            settled: s.settled(),
            giant_reserved: s.giant_reserved(),
        };
        let mut available = self.available();
        let mut verdict = self.verdict_in(&load, available, dxil_len, wanted, most, cap, floor);
        if verdict == super::pso_sched::Verdict::Hold {
            #[cfg(test)]
            if self.test_ram.is_some() {
                return (verdict, available);
            }
            if available_commit() < load.reserved.saturating_add(compile_cost(dxil_len).0).saturating_add(URGENT_FLOOR) {
                return (verdict, available);
            }
            trim_mapped_files();
            available = self.available();
            verdict = self.verdict_in(&load, available, dxil_len, wanted, most, cap, floor);
        }
        (verdict, available)
    }

    /// `verdict` for a load and the memory available as given.
    #[allow(clippy::too_many_arguments)]
    fn verdict_in(&self, load: &super::pso_sched::Load, available: u64, dxil_len: usize, wanted: bool, most: usize, cap: Option<u64>, floor: Option<u64>) -> super::pso_sched::Verdict {
        use super::pso_sched::{Ask, Room};
        #[cfg(test)]
        let commit_room = u64::MAX;
        #[cfg(not(test))]
        let commit_room = commit_room(load.reserved.saturating_add(compile_cost(dxil_len).0).saturating_add(URGENT_FLOOR), load.reserved == 0);
        #[cfg(test)]
        let ((fall, commit_fall), trend_fall, title_fall) = ((0, 0), 0, 0);
        // (a pipeline of the bind queue a background worker builds keeps none either: a skipped
        // draw waits for it; a cold giant does: `Ask::hard_giant`)
        #[cfg(not(test))]
        let ((fall, commit_fall), trend_fall, title_fall) = {
            let (rates, trend) = title_growth(available);
            (if wanted || super::pso_async::assisting_bind() { (0, 0) } else { rates }, trend, rates.0)
        };
        // (a warm-up keeps its pass's margin above the floor: `warm_children::Follow`)
        let margin = if WARM_UP_PROCESS.load(Ordering::Relaxed) { super::warm_children::governor_margin() } else { 0 };
        let room = Room { available, floor: floor.unwrap_or_else(keep_free) + margin, urgent_floor: URGENT_FLOOR + margin, commit: if cap.is_some() { crate::util::process_commit_bytes() } else { 0 }, commit_room, fall, commit_fall, trend_fall, title_fall, compile_rise: compile_rise() };
        let (cost, cold_giant) = compile_cost(dxil_len);
        let hard_giant = cold_giant && hard_giants() && !WARM_UP_PROCESS.load(Ordering::Relaxed) && !BLOCKING_WARM.with(|state| state.get().is_some());
        let helper = HELPER_COST.with(|c| c.get()).is_some() && super::pso_sched::helper_skips_growth();
        let (verdict, why) = super::pso_sched::admit_why(&Ask { dxil_len, wanted, most, cap, cost, small_lane: wanted && live_lanes() && !cold_giant, hard_giant, helper }, load, &room);
        HOLD_REASON.with(|c| c.set(why));
        if verdict == super::pso_sched::Verdict::Hold && (why.starts_with("memory") || why.starts_with("commit room")) {
            RAM_HOLDS.fetch_add(1, Ordering::Relaxed);
        }
        if fall != 0 || commit_fall != 0 {
            log_growth_headroom(&room, dxil_len, verdict);
        }
        if load.live_recent && !wanted && dxil_len >= super::pso_sched::GIANT_DXIL && verdict == super::pso_sched::Verdict::Hold {
            static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            if last.map_or(true, |t| t.elapsed() >= super::pso_sched::LIVE_QUIET) {
                *last = Some(std::time::Instant::now());
                crate::xlog_info!(
                    "D3D12.X: a background giant of {} KB of DXIL held: live compiles asked within {} s ({} MB available, {} waiting)",
                    dxil_len >> 10, super::pso_sched::LIVE_QUIET.as_secs(), available >> 20, load.starving
                );
            }
        }
        verdict
    }

    /// The compiles running and the memory available now, for a run's background compiles
    /// (`background_room`): the files mapped in are trimmed first when one of `least` bytes of
    /// DXIL would not start.
    #[cfg(not(test))]
    fn look(&self, least: usize, most: usize) -> (super::pso_sched::Load, u64) {
        let s = self.state.lock().unwrap();
        let load = super::pso_sched::Load { running: s.running, giants: s.giants, urgent: s.urgent, starving: s.starving, reserved: s.reserved, giant_waiting: s.giant_waiting(), live_recent: s.live_recent(), settled: s.settled(), giant_reserved: s.giant_reserved() };
        drop(s);
        if self.verdict_in(&load, available_physical(), least, false, most, None, Some(background_floor())) == super::pso_sched::Verdict::Hold {
            trim_mapped_files();
        }
        (load, available_physical())
    }

    /// Whether a run's background compile of `dxil_len` bytes of DXIL would start now.
    #[cfg(test)]
    fn fits(&self, dxil_len: usize, most: usize) -> bool {
        let s = self.state.lock().unwrap();
        self.verdict(&s, false, dxil_len, false, most, None, Some(background_floor()), Default::default()).0 == super::pso_sched::Verdict::Background
    }

    /// `acquire` with a background floor of its own (`verdict`). Without `wait`, a compile
    /// that does not start now is `Refused::NoRoom` at once: its thread is not held, and what is
    /// queued behind it goes on (one a bind waits for is then built again with `wait`:
    /// `pso_async::build_ahead`).
    #[allow(clippy::too_many_arguments)]
    fn acquire_within(&self, dxil_len: usize, most: usize, cap: Option<u64>, give_up: bool, floor: impl Fn(usize) -> Option<u64>, wait: bool, demand: impl Fn() -> DriverDemand) -> Result<GovGuard, Refused> {
        use super::pso_sched::Verdict;
        let mut clock = WaitClock(std::time::Instant::now(), false);
        let (cost, cold_giant) = compile_cost(dxil_len);
        if !give_up && !super::pso_sched::fits_on_machine(cost, URGENT_FLOOR, self.physical_total()) {
            CANNOT_BUILD.fetch_add(1, Ordering::Relaxed);
            return Err(Refused::NoMemory);
        }
        // (before `s`: dropped after it, so its lines are written outside the governor's lock)
        let mut lines = HeldLines::default();
        let mut s = self.state.lock().unwrap();
        let mut idle_since = None;
        let mut live = None;
        let mut starving = false;
        let id = std::thread::current().id();
        let blocking_giant = dxil_len >= GIANT_DXIL && BLOCKING_WARM.with(|state| state.get().is_some());
        let mut waiting_giant = false;
        let mut gave_way: Option<(u64, std::time::Instant)> = None;
        let mut slot_wait = None;
        let mut told = false;
        let mut never_fit_since: Option<std::time::Instant> = None;
        // a warm-up child admits against the other children's compiles under their common lock,
        // and publishes its own before it lets go (`warm_peers`)
        let peers = super::warm_peers::joined();
        let mut held;
        let (urgent, available_at_idle_break) = loop {
            // (`building_left_out`: the pipeline was left out while its build waited here)
            if blocking_warm_stopping() || super::pso_async::building_left_out() {
                s.starving -= starving as usize;
                s.stop_waiting(waiting_giant, cost);
                return Err(Refused::NoRoom);
            }
            if give_up && WARM_UP_CANCEL.load(Ordering::Relaxed) {
                s.starving -= starving as usize;
                s.stop_waiting(waiting_giant, cost);
                return Err(Refused::Cancelled);
            }
            let now_demand = demand();
            let wanted = now_demand != DriverDemand::Background;
            // (a live cold giant waits in its own lane: it holds no background work back, so
            // one that cannot fit does not stop everything else – play-91b)
            let holds_back = wanted && !cold_giant;
            if now_demand == DriverDemand::Live && !WARM_UP_PROCESS.load(Ordering::Relaxed) {
                s.live_at = Some(std::time::Instant::now());
            }
            // A bind waiting for room takes precedence over speculative builds.
            if holds_back != starving {
                if holds_back {
                    s.starving += 1;
                } else {
                    s.starving -= 1;
                }
                starving = holds_back;
            }
            held = peers.map(|p| p.lock());
            let peer_giants = held.as_ref().map_or_else(Default::default, |h| h.others_giants());
            let mut others = held.as_ref().map_or_else(Default::default, |h| h.others());
            let gross = others.reserved;
            if peer_giants_net() {
                others.reserved = others.reserved.saturating_sub(peer_giants.used);
            }
            let (verdict, available) = self.verdict(&s, starving, dxil_len, wanted, most, cap, floor(s.urgent), others);
            match verdict {
                Verdict::Urgent => break (true, None),
                Verdict::Background => break (false, None),
                Verdict::Hold => {}
            }
            if wanted && !WARM_UP_PROCESS.load(Ordering::Relaxed) {
                s.live_at = Some(std::time::Instant::now());
            }
            let never_fits = wanted && wait && !give_up && !WARM_UP_PROCESS.load(Ordering::Relaxed) && !BLOCKING_WARM.with(|state| state.get().is_some())
                && super::pso_sched::refuse_never_fit()
                && super::pso_sched::never_fits_now(cost, available, s.reserved.saturating_add(others.reserved), URGENT_FLOOR);
            if !never_fits {
                never_fit_since = None;
            } else if never_fit_since.get_or_insert_with(std::time::Instant::now).elapsed() >= super::pso_sched::NEVER_FIT_GRACE {
                s.starving -= starving as usize;
                s.stop_waiting(waiting_giant, cost);
                NEVER_FITS.with(|n| n.set((available, s.reserved.saturating_add(others.reserved))));
                return Err(Refused::NeverFits);
            }
            if !told {
                told = true;
                if let Some(line) = log_hold(dxil_len, cost, wanted, available, &s) {
                    if held.is_some() {
                        lines.1.push(format!("{line}; {}", warm_hold_detail(floor(s.urgent), gross, others.reserved, &peer_giants)));
                    } else {
                        lines.0.push(line);
                    }
                }
            }
            if wanted {
                slot_wait.get_or_insert_with(|| SlotWait::new(id, dxil_len, cost));
                lines.0.extend(log_slot_wait(&s, available));
            }
            if !wait {
                s.starving -= starving as usize;
                s.stop_waiting(waiting_giant, cost);
                return Err(Refused::NoRoom);
            }
            // a blocking warm-up's giant asks smaller compiles for room while some run; when they
            // drained and it still does not start, the memory is held elsewhere: it gives way until
            // a giant finishes or a minute passes, so it cannot hold them one at a time for the pass
            if blocking_giant {
                if waiting_giant && s.running == s.giants {
                    s.stop_waiting(true, cost);
                    waiting_giant = false;
                    gave_way = Some((s.giants_finished, std::time::Instant::now()));
                } else if !waiting_giant && s.running > s.giants && gave_way.map_or(true, |(finished, at)| finished != s.giants_finished || at.elapsed() >= std::time::Duration::from_secs(60)) {
                    s.giant_waits.push(cost);
                    waiting_giant = true;
                }
            }
            let running = s.running + others.running;
            if !give_up && cap.is_none() && s.idle_breaker.hold(id, demand() == DriverDemand::Blocking, running, available, URGENT_FLOOR, std::time::Instant::now())
                && (cfg!(test) || commit_room(cost.saturating_add(URGENT_FLOOR), true) >= cost.saturating_add(URGENT_FLOOR))
            {
                break (true, Some(available));
            }
            if give_up && running == 0 {
                let since = *idle_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() >= std::time::Duration::from_secs(10) {
                    s.starving -= starving as usize;
                    s.stop_waiting(waiting_giant, cost);
                    let room = total_physical().saturating_sub(keep_free() + crate::util::process_commit_bytes());
                    return Err(if cost > room { Refused::TooBig } else { Refused::Later });
                }
            } else {
                idle_since = None;
            }
            live.get_or_insert_with(|| crate::live_stall::stage(crate::live_stall::Stage::DriverSlot, || 0));
            drop(held.take());
            if !lines.0.is_empty() {
                drop(s);
                lines.flush();
                s = self.state.lock().unwrap();
            }
            clock.waiting();
            s = self.cv.wait_timeout(s, std::time::Duration::from_millis(200)).unwrap().0;
        };
        s.starving -= starving as usize;
        s.stop_waiting(waiting_giant, cost);
        s.idle_breaker.started(id);
        // (a giant a bind waits for counts too: the background ones leave it its place)
        let giant = dxil_len >= GIANT_DXIL;
        let already_overlapping = s.running != 0;
        s.started = s.started.wrapping_add(1);
        let run = (s.started, cost, std::time::Instant::now(), std::time::Duration::from_secs(super::pso_sched::compile_secs(dxil_len)), cold_giant);
        s.runs.push(run);
        s.running += 1;
        s.urgent += urgent as usize;
        s.giants += giant as usize;
        s.reserved += cost;
        if let Some(held) = &held {
            if cold_giant && s.giant_reserved() == cost {
                GIANT_BASE.store(crate::util::process_commit_bytes(), Ordering::Relaxed);
            }
            held.publish(super::warm_peers::Usage { running: s.running, giants: s.giants, reserved: s.reserved });
            held.publish_giants(s.giant_reserved(), GIANT_BASE.load(Ordering::Relaxed));
        }
        drop(held);
        Ok(GovGuard { cost, giant, urgent, available_at_idle_break, started: s.started, already_overlapping, commit_measure: None, live: None, _running: crate::progress::DRIVER.begin(), _large: giant.then(|| crate::progress::LARGE_DRIVER.begin()) })
    }

    fn release(&self, g: &GovGuard) -> bool {
        let mut s = self.state.lock().unwrap();
        s.running -= 1;
        s.urgent -= g.urgent as usize;
        s.giants -= g.giant as usize;
        s.giants_finished += g.giant as u64;
        s.reserved -= g.cost;
        if let Some(i) = s.runs.iter().position(|run| run.0 == g.started) {
            s.runs.swap_remove(i);
        }
        if let Some(peers) = super::warm_peers::joined() {
            let held = peers.lock();
            held.publish(super::warm_peers::Usage { running: s.running, giants: s.giants, reserved: s.reserved });
            held.publish_giants(s.giant_reserved(), GIANT_BASE.load(Ordering::Relaxed));
        }
        self.cv.notify_all();
        g.already_overlapping || g.started != s.started
    }

    /// (compiles running, giants among them, urgent ones among them)
    fn in_flight(&self) -> (usize, usize, usize) {
        let s = self.state.lock().unwrap();
        (s.running, s.giants, s.urgent)
    }
}

/// The run's compile failures, the driver compiles running now, the memory left, the binds that
/// waited and the warm driver cache, for the 30 s caches line.
pub fn driver_compiles_summary() -> String {
    let (n, giants, urgent, queued, reserved, settled) = {
        let s = DRIVER_COMPILES.state.lock().unwrap();
        (s.running, s.giants, s.urgent, s.starving, s.reserved >> 20, s.settled() >> 20)
    };
    let (bind_waits, use_waits, worst) = super::pso_async::waits();
    let waits = bind_waits + use_waits;
    let (urgent_builds, window_waits, window_ms, window_skips) = super::pso_async::take_window();
    let (image_pipelines, image_binds, image_uses) = super::pso_async::image_skips();
    format!(
        "failed {} ({} cannot build within memory); driver compiles running {n} ({giants} giants, {urgent} waited for), {reserved} MiB estimated in-flight reservation ({settled} MiB of it past its expected time, counted as measured for live compiles), {} MiB physical RAM available; {}; binds waited {waits} ({bind_waits} at bind, {use_waits} at use; worst {worst} ms; since the last line {urgent_builds} builds a bind started, {window_waits} waits took {window_ms} ms in all); image-only compute avoided {image_binds} bind waits and {image_uses} use waits over {image_pipelines} pipelines; draws skipped {} ({window_skips} since the last line; {} at once for the frame wait budget); GPU waits released on the CPU {}; {}; {queued} urgent compiles queued for RAM or a giant slot",
        COMPILE_FAILURES.load(Ordering::Relaxed),
        CANNOT_BUILD.load(Ordering::Relaxed),
        available_physical() >> 20,
        super::pso_async::ahead_summary(),
        super::pso_async::skips(),
        super::pso_async::budget_skips(),
        super::list_x::cpu_releases(),
        driver_cache::summary()
    )
}

impl Drop for GovGuard {
    fn drop(&mut self) {
        let measure = self.commit_measure.take().map(|measure| {
            let elapsed = measure.began.elapsed();
            let peak = GiantCommitMeasure::finish(self.started);
            (measure, elapsed, peak)
        });
        let overlapping = DRIVER_COMPILES.release(self);
        if let Some((measure, elapsed, peak)) = measure {
            let measured = peak.growth(measure.baseline).map_or_else(|| "unavailable".into(), |bytes| format!("+{} MiB", bytes >> 20));
            crate::xlog_info!(
                "D3D12.X: giant driver commit {} ({} B DXIL): estimated {} MiB, process private commit peak {measured} ({} samples over {} ms; other driver permits overlapped: {overlapping})",
                measure.shader, measure.dxil_len, self.cost >> 20, peak.samples, elapsed.as_millis()
            );
        }
    }
}

/// Which background driver compiles of a run (by their bytes of DXIL; true: one a bind asked
/// for, which starts as the compiles binds wait for do) would start now, with the memory and
/// the compiles running as they are now: a build put off for memory (`pso_async::put_off`) is
/// taken up again when its compile would. `least`: the smallest one
/// asked about – when not even that fits, the files mapped into the process are trimmed first
/// (`trim_mapped_files`), as before a compile that waits.
#[derive(Clone, Copy)]
#[cfg(not(test))]
pub(super) struct ParkedRoom {
    load: super::pso_sched::Load,
    available: u64,
    most: usize,
    floor: u64,
    bind_floor: u64,
}

#[cfg(not(test))]
impl ParkedRoom {
    /// (`pso_sched::held_on_sample` first: the full verdict reads the system's memory)
    pub(super) fn fits(self, dxil_len: usize) -> bool {
        !super::pso_sched::held_on_sample(dxil_len, compile_cost(dxil_len).0, self.most, &self.load, self.available, self.floor, URGENT_FLOOR)
            && DRIVER_COMPILES.verdict_in(&self.load, self.available, dxil_len, false, self.most, None, Some(self.floor))
                == super::pso_sched::Verdict::Background
    }

    pub(super) fn peer_fits(self, dxil_len: usize) -> bool {
        super::pso_async::title_held()
            && self.load.urgent > 0
            && !super::pso_sched::held_on_sample(dxil_len, compile_cost(dxil_len).0, self.most, &self.load, self.available, self.bind_floor, URGENT_FLOOR)
            && DRIVER_COMPILES.verdict_in(&self.load, self.available, dxil_len, false, self.most, None, Some(self.bind_floor))
                == super::pso_sched::Verdict::Background
    }
}

#[cfg(not(test))]
pub(super) fn background_room(least: usize) -> ParkedRoom {
    tune_peaks();
    let most = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(TITLE_THREADS).clamp(1, SMALL_AT_ONCE);
    let (load, available) = DRIVER_COMPILES.look(least, most);
    let (floor, bind_floor, _, _, _) = background_memory();
    ParkedRoom { load, available, most, floor, bind_floor }
}

static DRIVER_COMPILES: Governor = Governor {
    state: Mutex::new(GovState { running: 0, started: 0, giants: 0, urgent: 0, starving: 0, reserved: 0, giant_waits: Vec::new(), giants_finished: 0, idle_breaker: super::pso_sched::IdleBreaker::new(), live_at: None, runs: Vec::new() }),
    cv: std::sync::Condvar::new(),
    #[cfg(test)]
    test_ram: None,
};
static PEER_COMPILES: AtomicU64 = AtomicU64::new(0);

pub(super) fn peer_compiles_started() -> u64 {
    PEER_COMPILES.load(Ordering::Relaxed)
}


struct MaterialGatePipeline {
    original: ID3D12PipelineState,
    rs: ID3D12RootSignature,
    device: ID3D12Device,
    source: Vec<u8>,
    compiled: crate::gpu::recomp::pso::Compiled,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node_mask: u32,
    variant: OnceLock<Option<ID3D12PipelineState>>,
}

static MATERIAL_PIPELINES: Mutex<Vec<Arc<MaterialGatePipeline>>> = Mutex::new(Vec::new());

pub(super) unsafe fn material_gate_variant(original: usize) -> Option<usize> {
    let recipe = MATERIAL_PIPELINES.lock().unwrap().iter().find(|r| r.original.as_raw() as usize == original)?.clone();
    recipe.variant.get_or_init(|| {
        let make = || -> Result<ID3D12PipelineState, String> {
            let raw = recipe.device.as_raw();
            let variant = crate::gpu::recomp::pso::material_gate::compile(&recipe.source, wave32_only(raw), &recipe.compiled)?;
            let _slot = driver_slot(variant.dxil.len(), false, false, || "diagnostic material-gate compute PSO".into()).map_err(|_| "diagnostic driver compile not admitted")?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&recipe.rs),
                CS: bytecode(&variant.dxil), NodeMask: recipe.node_mask, Flags: recipe.flags, ..Default::default()
            };
            let create: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, "CreateComputePipelineState")));
            let mut out = std::ptr::null_mut();
            let hr = create(raw, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &pc_iid(Kind::PipelineState), &mut out);
            if hr < 0 || out.is_null() { return Err(format!("driver rejected diagnostic PSO: {hr:#x}")) }
            Ok(ID3D12PipelineState::from_raw(out))
        };
        match make() {
            Ok(p) => { crate::xlog_info!("material gate probe: isolated variant {:p} for native PSO {original:#x}", p.as_raw()); Some(p) }
            Err(reason) => { crate::xlog_warn!("material gate probe: keeping ordinary PSO {original:#x}: {reason}"); None }
        }
    }).as_ref().map(|p| p.as_raw() as usize)
}

unsafe fn translated_compute_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<Result<HRESULT, ()>> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let cs = shader_bytes(&d.CS)?;
    if !is_console(cs) {
        return None;
    }
    // made when first bound (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let translating = crate::progress::TRANSLATIONS.begin();
    let live = crate::live_stall::stage(crate::live_stall::Stage::ShaderCompile, || crate::gpu::shader::fnv64(cs));
    let compiled = match console_to_dxil_for(cs, Stage::Compute, 0, wave32_only(this)) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("CS {:016x}: {e}", crate::gpu::shader::fnv64(cs)));
            return Some(Err(()));
        }
    };
    drop(live);
    drop(translating);
    let dxil = compiled.dxil;
    // a big program: the driver's compile takes it seconds to a minute (a retail game's 1 MB
    // ray-tracing resolve programs ~30 s each, ~600 of them): built in the background like one
    // whose DXIL is missing – the title's loading does not wait for it, its first bind does
    let big = dxil.len() >= BIG_DRIVER_COMPILE;
    if big && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let Some(rs) = abi_rs_for(this, &compiled.samplers) else { return Some(Err(())) };
    LAST_COMPUTE_RS.with(|c| c.set(rs as usize));
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.CS = bytecode(&dxil);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    // A program built again gets the driver's own result of its first build: a retail title
    // made 14 big compute pipelines twice while loading, 13 - 30 s each the second time.
    let key = compute_blob_key(&dxil, rs as usize, copy.Flags.0, copy.NodeMask);
    // … and one built in an earlier run gets the blob kept on disk then (`driver_cache`)
    let disk = driver_cache::compute_path(&dxil, &compiled.samplers, copy.NodeMask, copy.Flags.0);
    let legacy = driver_cache::path(&dxil, copy.Flags.0).filter(|p| Some(p) != disk.as_ref());
    if BLOCKING_WARM.with(|state| state.get().is_some()) {
        let _slot = driver_slot(dxil.len(), false, disk.iter().chain(&legacy).any(|p| driver_cache::kept(p)), || format!("CS {:016x}", crate::gpu::shader::fnv64(cs))).ok()?;
        if let Some(hr) = library_load(this, disk.as_deref(), super::pso_library::Desc::Compute(&copy), riid, ppv, true) {
            return Some(Ok(hr));
        }
        let (hr, _) = driver_cache::build_with_legacy(disk.as_deref(), legacy.as_deref(), |blob| {
            copy.CachedPSO = blob.map_or_else(D3D12_CACHED_PIPELINE_STATE::default, |b| D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: b.as_ptr().cast(), CachedBlobSizeInBytes: b.len(),
            });
            finish_pso(this, f, (&copy as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), riid, ppv)
        }, || pso_blob(*ppv));
        if hr >= 0 {
            library_store(this, disk.as_deref(), Some(super::pso_library::Desc::Compute(&copy)), *ppv);
        }
        return Some(if hr < 0 { Err(()) } else { Ok(hr) });
    }
    let mut candidates = disk.as_deref().into_iter().chain(legacy.as_deref())
        .filter_map(|path| driver_cache::load(path).map(|b| (Arc::new(b), Some(path))));
    let mut cached = compute_blob(key).map(|b| (b, None)).or_else(|| candidates.next());
    let asked = std::time::Instant::now();
    // preparing (`pso_async::preparing`): a big program's driver compile is made too, for the
    // driver's disk cache (an NVIDIA driver: 46–131 s, the next time 0.07–0.13 s), and the
    // pipeline is released at once; one with a blob kept already needs no driver now
    let warm = super::pso_async::preparing();
    if warm && !(big && cached.is_none()) {
        return Some(Err(()));
    }
    // a cold giant compiles in a helper process within its reservation (`giant_helper`); the
    // pipeline is then made here from the driver blob it left
    if !warm && cached.is_none() && dxil.len() >= super::pso_sched::GIANT_DXIL && giant_helper::on() && !WARM_UP_PROCESS.load(Ordering::Relaxed) {
        if let Some(path) = disk.as_deref() {
            let what = || format!("CS {:016x}", crate::gpu::shader::fnv64(cs));
            match giant_helper::build(&dxil, &compiled.samplers, copy.Flags.0, copy.NodeMask, path, &what) {
                giant_helper::End::Blob => cached = driver_cache::load(path).map(|b| (Arc::new(b), Some(path))),
                giant_helper::End::Refused => return Some(Err(())),
                giant_helper::End::InProcess => {}
            }
        }
    }
    // in the background, within the memory left (`DRIVER_COMPILES`)
    let Ok(slot) = driver_slot(dxil.len(), warm, cached.is_some(), || format!("CS {:016x}", crate::gpu::shader::fnv64(cs))) else {
        return Some(Err(()));
    };
    // the driver's time, without the wait for a slot
    let (mut started, waited) = (std::time::Instant::now(), asked.elapsed());
    if warm {
        if super::pso_async::with_driver(|| finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)) >= 0 && !(*ppv).is_null() {
            let took = started.elapsed();
            keep_compute_blob(key, *ppv, dxil.len(), disk.as_deref(), took);
            driver_cache::note_miss(dxil.len(), took);
            library_store(this, disk.as_deref(), Some(super::pso_library::Desc::Compute(&copy)), *ppv);
            super::pso_async::release(std::mem::replace(&mut *ppv, std::ptr::null_mut()) as usize);
        }
        return Some(Err(()));
    }
    // an NVIDIA driver aborted the process reading a bad module ("Invalid record"): a create from
    // a blob is named on stderr before and after, so an abort names its blob
    let mut accepted = library_load(this, disk.as_deref(), super::pso_library::Desc::Compute(&copy), riid, ppv, true);
    let from_library = accepted.is_some();
    if from_library {
        cached = None;
    }
    while let Some((b, source)) = cached.take() {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr().cast(), CachedBlobSizeInBytes: b.len() };
        let at = source.map_or_else(|| "kept in memory".into(), |p| p.display().to_string());
        let crumb = format!("driver blob {:016x} ({} B, {at})", crate::gpu::shader::fnv64(&b), b.len());
        eprintln!("xcl: {crumb}: creating the pipeline");
        let hr = finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv);
        eprintln!("xcl: {crumb}: {hr:#x}");
        if hr >= 0 {
            accepted = Some(hr);
            if source.is_some() {
                driver_cache::note(true, None);
            }
            break;
        }
        if source.is_none() {
            forget_compute_blob(key);
        } else {
            driver_cache::note(false, source.filter(|p| Some(*p) == disk.as_deref()));
        }
        cached = candidates.next();
        started = std::time::Instant::now();
    }
    let reused = accepted.is_some();
    let hr = accepted.unwrap_or_else(|| {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)
    });
    let took = started.elapsed();
    drop(slot);
    if hr >= 0 && !reused {
        keep_compute_blob(key, *ppv, dxil.len(), disk.as_deref(), took);
        driver_cache::note_miss(dxil.len(), took);
    }
    if hr >= 0 && !from_library {
        library_store(this, disk.as_deref(), Some(super::pso_library::Desc::Compute(&copy)), *ppv);
    }
    if hr < 0 {
        note_translation_error(format!("CS {:016x}: the driver rejected the compute PSO ({hr:#x})", crate::gpu::shader::fnv64(cs)));
        return Some(Err(()));
    }
    TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    note_pso_rs(*ppv, rs);
    if let Some(i) = crate::gpu::recomp::pso::compute_interface(cs) {
        PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (i.user_sgprs, i.threads));
    }
    super::dispatch_timing::note(*ppv as usize, cs);
    note_translated_compute(hr, ppv);
    if super::material_gate_capture::enabled() && crate::gpu::recomp::pso::material_gate::selected_hash() == Some(crate::gpu::shader::fnv64(cs)) {
        if let (Some(original), Some(root), Some(device)) = (ID3D12PipelineState::from_raw_borrowed(&*ppv), ID3D12RootSignature::from_raw_borrowed(&rs), ID3D12Device::from_raw_borrowed(&this)) {
            let mut recipes = MATERIAL_PIPELINES.lock().unwrap();
            if recipes.len() < 8 {
                recipes.push(Arc::new(MaterialGatePipeline { original: original.clone(), rs: root.clone(), device: device.clone(),
                    source: cs.to_vec(), compiled: crate::gpu::recomp::pso::Compiled { dxil: dxil.clone(), samplers: compiled.samplers.clone() },
                    flags: copy.Flags, node_mask: copy.NodeMask, variant: OnceLock::new() }));
            }
        }
    }
    super::pso_async::note_build(|| {
        format!(
            "CS {:016x} {} B of DXIL; slot wait {} ms, driver {} ms{}",
            crate::gpu::shader::fnv64(cs),
            dxil.len(),
            waited.as_millis(),
            took.as_millis(),
            if reused { " (the driver's blob of an earlier build)" } else { "" }
        )
    });
    crate::xlog_info!(
        "D3D12.X: compute PSO with the translated console CS {:016x} ({} B DXIL, {} ms in the driver, {} ms waiting for a slot{})",
        crate::gpu::shader::fnv64(cs),
        dxil.len(),
        took.as_millis(),
        waited.as_millis(),
        if reused { ", the driver's blob of an earlier build" } else { "" }
    );
    Some(Ok(hr))
}

/// A slot for a translated driver compile (`what`: its stage and hash, for the log) within the memory
/// left (`DRIVER_COMPILES`): on XCL's build threads (`xcl-pso*`, the warm-up's within
/// `WARM_UP_COMMIT`) or while preparing (`warm`) as background compiles, on the title's own
/// threads as compiles it waits for. Err: not built, logged (`Governor::acquire`). `kept`: its
/// driver's blob is kept on disk (`driver_cache::kept`). On a thread leaving the giants
/// (`LEAVE_GIANTS`) a giant's cold compile is refused at once (`Refused::Giant`); one with a blob
/// kept starts if the memory lets it now, and is refused as a giant otherwise.
fn driver_slot(dxil_len: usize, warm: bool, kept: bool, what: impl FnOnce() -> String) -> Result<GovGuard, Refused> {
    TRANSLATION_SLOT.with(|slot| drop(slot.borrow_mut().take()));
    tune_peaks();
    let probe = crate::gpu::recomp::dxc::in_cached_only();
    // (`XCL_ASYNC_PSO`: a pipeline made on the title's thread whose driver compile is not kept on
    // disk is built in the background instead, behind a stand-in)
    if (dxil_len >= BIG_DRIVER_COMPILE || (!kept && super::pso_async::async_on())) && crate::gpu::recomp::dxc::defer() {
        return Err(Refused::NoRoom);
    }
    let leave_giant = dxil_len >= GIANT_DXIL && LEAVE_GIANTS.with(|l| l.get());
    if leave_giant && !kept {
        REFUSED.with(|r| r.set(Some(Refused::Giant)));
        LEFT_GIANT.with(|l| l.set(l.get().max(dxil_len)));
        return Err(Refused::Giant);
    }
    let background = warm || super::pso_async::preparing() || std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso"));
    let warming = WARMING_UP.with(|w| w.get());
    let in_game = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(TITLE_THREADS);
    let blocking_warm = BLOCKING_WARM.with(|state| state.get().is_some());
    let (most, cap) = if blocking_warm {
        (warming.unwrap_or(1), None)
    } else {
        warming.map_or((in_game.clamp(1, SMALL_AT_ONCE), None), |n| (n.min(SMALL_AT_ONCE), Some(WARM_UP_COMMIT)))
    };
    // a run's background builds keep the run's floor and never hold their worker; a warm-up and
    // the preparation (`warm`) keep `KEEP_FREE` and wait
    let run = warming.is_none() && !warm;
    let small_live = live_lanes() && run && super::pso_async::live_build() && (kept || dxil_len < GIANT_DXIL);
    let wait = !leave_giant && !probe && !(background && super::pso_async::may_put_off() && !small_live);
    let peer = std::cell::Cell::new(false);
    let floor = |urgent| run.then(|| {
        peer.set(urgent > 0 && super::pso_async::title_held() && super::pso_async::assisting_bind());
        background_floor_when(peer.get())
    }).or(blocking_warm.then_some(URGENT_FLOOR));
    let live = run && live_priority();
    KEPT_BLOB.with(|k| k.set(kept));
    let mut r = DRIVER_COMPILES.acquire_within(dxil_len, most, cap, warming.is_some() && !blocking_warm, floor, wait, || {
        if run && (!background || super::pso_async::blocking_waiter() || super::pso_async::first_build()) {
            DriverDemand::Blocking
        } else if !background || super::pso_async::waited_on() {
            DriverDemand::Wanted
        } else if live && super::pso_async::live_build() {
            DriverDemand::Live
        } else {
            DriverDemand::Background
        }
    });
    KEPT_BLOB.with(|k| k.set(false));
    if leave_giant && matches!(r, Err(Refused::NoRoom)) && !blocking_warm_stopping() {
        r = Err(Refused::Giant);
    }
    if peer.get() && r.as_ref().is_ok_and(|slot| !slot.urgent) {
        PEER_COMPILES.fetch_add(1, Ordering::Relaxed);
    }
    if let Err(e) = &r {
        REFUSED.with(|c| c.set(Some(*e)));
    }
    match &mut r {
        Err(Refused::NoRoom) => {
            if !crate::gpu::recomp::dxc::defer() {
                super::pso_async::put_off(dxil_len);
            }
        }
        Err(Refused::TooBig) => crate::xlog_warn!("D3D12.X: {}: {dxil_len} B of DXIL – too big to build within the memory limits, left for its first bind", what()),
        Err(Refused::Later) => eprintln!("  {}: {dxil_len} B of DXIL – no room now ({} MB available), tried again at the end of the phase", what(), available_physical() >> 20),
        Err(Refused::Giant) => LEFT_GIANT.with(|l| l.set(l.get().max(dxil_len))),
        Err(Refused::NeverFits) => {
            COMPILE_FAILURES.fetch_add(1, Ordering::Relaxed);
            let n = CANNOT_BUILD.fetch_add(1, Ordering::Relaxed) + 1;
            let (available, reserved) = NEVER_FITS.with(|n| n.get());
            let why = format!(
                "cannot fit in this run's memory: {} ({}): {dxil_len} B of DXIL, ~{} MiB expected peak; {} MiB available and {} MiB reserved by the compiles running, less the {} MiB floor, cannot hold it even with all of them done (XCL_REFUSE_NEVER_FIT=0: wait)",
                what(),
                building(),
                compile_cost(dxil_len).0 >> 20,
                available >> 20,
                reserved >> 20,
                URGENT_FLOOR >> 20,
            );
            static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64, usize)>>> = Mutex::new(None);
            let (pipeline, shader) = crate::live_stall::job();
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert((pipeline, shader, dxil_len)) {
                crate::xlog_warn!("D3D12.X: {why} – refused after {} s of not fitting, its binds skip (#{n} cannot build within memory)", super::pso_sched::NEVER_FIT_GRACE.as_secs());
            }
            LAST_NOTE.with(|n| *n.borrow_mut() = Some(why));
        }
        Err(Refused::Cancelled) => {}
        Err(Refused::NoMemory) => {
            COMPILE_FAILURES.fetch_add(1, Ordering::Relaxed);
            let why = format!(
                "cannot build within memory: {}: {dxil_len} B of DXIL, ~{} MB expected peak exceeds {} MB total physical RAM less the {} MB floor",
                what(),
                peak(dxil_len) >> 20,
                total_physical() >> 20,
                URGENT_FLOOR >> 20,
            );
            crate::xlog_error!("D3D12.X: {why} – its stand-in stays bound");
            // (why its pipeline has no translated shader: `why_untranslated`, for the skipped dispatch)
            LAST_NOTE.with(|n| *n.borrow_mut() = Some(why));
        }
        Ok(slot) => {
            slot.live = Some(crate::live_stall::stage(crate::live_stall::Stage::DriverBuild, || 0));
            if (run && slot.giant) || slot.available_at_idle_break.is_some() {
                let shader = what();
                if let Some(available) = slot.available_at_idle_break {
                    crate::xlog_warn!(
                        "D3D12.X: no driver compile in flight for {} ms; starting oldest blocked compile {} ({dxil_len} B DXIL, estimated peak {} MiB, {} MiB physical RAM available, {} MiB floor); this may page",
                        super::pso_sched::IDLE_BREAK_GRACE.as_millis(), shader, slot.cost >> 20, available >> 20, URGENT_FLOOR >> 20
                    );
                }
                if run && slot.giant {
                    slot.commit_measure = Some(GiantCommitMeasure::start(slot.started, shader, dxil_len));
                }
            }
        }
    }
    r
}

thread_local! {
    /// a blocking warm-up key's slot for its translation and DXC compiles (`translation_slot`),
    /// given back when its driver compile asks for one (`driver_slot`) or the key ends
    static TRANSLATION_SLOT: std::cell::RefCell<Option<GovGuard>> = const { std::cell::RefCell::new(None) };
    /// a warm-up thread, of this many
    static WARMING_UP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// a warm-up thread leaving the giants' driver compiles for later (`driver_slot` refuses them)
    static LEAVE_GIANTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// the DXIL length of the largest giant's driver compile this thread's key left (`Refused::Giant`)
    static LEFT_GIANT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// why `driver_slot` last refused this thread a slot (a warm-up's build clears it first)
    static REFUSED: std::cell::Cell<Option<Refused>> = const { std::cell::Cell::new(None) };
    static BLOCKING_WARM: std::cell::Cell<Option<(std::time::Instant, WarmDriverCounts)>> = const { std::cell::Cell::new(None) };
}

#[derive(Default, Clone, Copy)]
struct WarmDriverCounts {
    hits: usize,
    cold: usize,
    rejected: usize,
    saved: usize,
    /// pipelines loaded from, missed in and stored in the pipeline library (`pso_library`), and
    /// its shards refused
    library_hits: usize,
    library_misses: usize,
    library_stored: usize,
    library_refused: usize,
}

/// The pipeline library's counts of a warm-up, for its summary.
fn library_line(c: &WarmDriverCounts) -> String {
    format!("pipeline library {} hits, {} misses, {} stored, {} shards refused", c.library_hits, c.library_misses, c.library_stored, c.library_refused)
}

static BLOCKING_WARM_CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn cancel_blocking_warm_up() {
    BLOCKING_WARM_CANCELLED.store(true, Ordering::Relaxed);
}

fn blocking_warm_stopping() -> bool {
    BLOCKING_WARM.with(|state| state.get().is_some_and(|(deadline, _)| {
        BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed) || std::time::Instant::now() >= deadline
    }))
}

fn blocking_warm_refusing() -> bool {
    if !blocking_warm_stopping() {
        return false;
    }
    REFUSED.with(|r| r.set(Some(Refused::NoRoom)));
    true
}

fn note_blocking_driver(note: impl FnOnce(&mut WarmDriverCounts)) {
    BLOCKING_WARM.with(|state| {
        if let Some((deadline, mut counts)) = state.get() {
            note(&mut counts);
            state.set(Some((deadline, counts)));
        }
    });
}

thread_local! {
    /// the time this thread waited for admission so far (`memory_waited`)
    static MEMORY_WAITED: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// The time this thread has spent waiting for admission so far, in `Governor::acquire_within`:
/// for a driver compile (`driver_slot`), a translation (`translation_slot`) or the other warm-up
/// children's compiles (`warm_peers`) alike. A build's time less the difference across it is the
/// time it took to build.
pub(super) fn memory_waited() -> std::time::Duration {
    MEMORY_WAITED.with(|w| w.get())
}

/// Adds the time from its start to its drop to `MEMORY_WAITED`; in a warm-up child, the parent's
/// hang clock for the key stops from `waiting` to the drop (`warm_children::memory_wait`).
struct WaitClock(std::time::Instant, bool);

impl WaitClock {
    fn waiting(&mut self) {
        if !self.1 {
            self.1 = true;
            super::warm_children::memory_wait(true);
        }
    }
}

impl Drop for WaitClock {
    fn drop(&mut self) {
        let waited = self.0.elapsed();
        MEMORY_WAITED.with(|w| w.set(w.get() + waited));
        if self.1 {
            super::warm_children::memory_wait(false);
        }
    }
}

/// DXIL expected from one byte of console shader: a blocking warm-up key's translation and DXC
/// compiles are admitted as a driver compile of that much DXIL (`translation_slot`). An estimate,
/// not a measurement: the per-key line logs each key's console shader bytes to calibrate it.
pub(super) const TRANSLATED_DXIL_PER_GUEST_BYTE: usize = 24;
/// The most DXIL a key's translation is taken for: a giant of ~3.7 GB (`pso_sched::peak`), so a
/// huge key waits for memory like the largest giants instead of never fitting.
const TRANSLATION_DXIL_CAP: usize = 3_240_760;

/// The DXIL a blocking warm-up key with `guest_bytes` of console shaders is taken to translate to.
pub(super) fn translated_dxil(guest_bytes: usize) -> usize {
    guest_bytes.saturating_mul(TRANSLATED_DXIL_PER_GUEST_BYTE).min(TRANSLATION_DXIL_CAP)
}

/// A slot for the translation and DXC compiles of a blocking warm-up key with `guest_bytes` of
/// console shaders, within the memory left (`DRIVER_COMPILES`) as the driver compiles are: the
/// same 1 GiB floor, the same giant cap, waiting rather than refused, and a giant that cannot fit
/// gives way. Free RAM fell to 32 MB in a pass that admitted only the driver compiles.
fn translation_slot(guest_bytes: usize) -> Option<GovGuard> {
    let workers = WARMING_UP.with(|w| w.get()).unwrap_or(1);
    DRIVER_COMPILES.acquire_within(translated_dxil(guest_bytes), workers, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Background).ok()
}

/// What one key of a blocking warm-up came to (`blocking_key`).
struct KeyReport {
    kind: String,
    ok: bool,
    refused: bool,
    counts: WarmDriverCounts,
    elapsed: std::time::Duration,
    guest: usize,
    /// waiting for its translation's slot (`translation_slot`)
    admitted: std::time::Duration,
    /// `elapsed` less every wait for memory (`memory_waited`)
    build: std::time::Duration,
    /// the DXIL length of a driver compile of `GIANT_DXIL` or more left to a single-key child
    /// (`LEAVE_GIANTS`)
    giant: Option<usize>,
}

/// One key of a blocking warm-up on a worker thread: its translation's slot, then its pipeline
/// built (`warm_pipeline`).
fn blocking_key(dev: usize, deadline: std::time::Instant, record: &super::pso_manifest::Record) -> KeyReport {
    BLOCKING_WARM.with(|state| state.set(Some((deadline, WarmDriverCounts::default()))));
    REFUSED.with(|r| r.set(None));
    LEFT_GIANT.with(|l| l.set(0));
    let t0 = std::time::Instant::now();
    let waited = memory_waited();
    let guest = record.shaders.iter().map(|(_, s)| s.len()).sum::<usize>();
    let slot = translation_slot(guest);
    let admitted = t0.elapsed();
    TRANSLATION_SLOT.with(|t| *t.borrow_mut() = slot);
    let ok = std::panic::catch_unwind(|| unsafe { warm_pipeline(dev as *mut c_void, record) }).unwrap_or(false);
    TRANSLATION_SLOT.with(|t| drop(t.borrow_mut().take()));
    let counts = BLOCKING_WARM.with(|state| state.get().unwrap().1);
    let refused = REFUSED.with(|r| r.get());
    let elapsed = t0.elapsed();
    let build = elapsed.saturating_sub(memory_waited().saturating_sub(waited));
    let giant = matches!(refused, Some(Refused::Giant)).then(|| LEFT_GIANT.with(|l| l.get()));
    KeyReport { kind: format!("{:?}", record.kind), ok, refused: refused.is_some(), counts, elapsed, guest, admitted, build, giant }
}

type NextKey = Arc<dyn Fn() -> Option<(u64, super::pso_manifest::Record)> + Send + Sync>;
type KeyDone = Arc<dyn Fn(u64, KeyReport) + Send + Sync>;

/// `wanted` blocking warm-up workers (in this process or in a warm-up child), each building the
/// keys `next` gives (`blocking_key`) until it gives none or the pass stops, and handing each
/// key's report to `done`. `leave_giants`: a driver compile of `GIANT_DXIL` or more is refused
/// at once (`LEAVE_GIANTS`), for a pool child that hands its giants to single-key children.
fn blocking_workers(wanted: usize, dev: usize, deadline: std::time::Instant, leave_giants: bool, next: NextKey, done: KeyDone) -> Result<Vec<std::thread::JoinHandle<()>>, String> {
    let mut handles = Vec::with_capacity(wanted);
    for k in 0..wanted {
        let (next, done) = (next.clone(), done.clone());
        let spawned = std::thread::Builder::new().name(format!("xcl-pso-warm-blocking-{k}")).stack_size(64 << 20).spawn(move || {
            WARMING_UP.with(|state| state.set(Some(wanted)));
            LEAVE_GIANTS.with(|l| l.set(leave_giants));
            BLOCKING_WARM.with(|state| state.set(Some((deadline, WarmDriverCounts::default()))));
            while !blocking_warm_stopping() {
                let Some((key, record)) = next() else { break };
                done(key, blocking_key(dev, deadline, &record));
            }
        });
        match spawned {
            Ok(handle) => handles.push(handle),
            Err(e) if handles.is_empty() => return Err(format!("could not start the blocking warm-up workers: {e}")),
            Err(e) => {
                eprintln!("warm-up --blocking: started {} of {wanted} workers ({e})", handles.len());
                break;
            }
        }
    }
    Ok(handles)
}

const READY: &str = "ready";

/// The keys a blocking warm-up pass went through, by their status (`note`).
#[derive(Default)]
struct BlockingTally {
    done: usize,
    ready: usize,
    failed: usize,
    deferred: usize,
    unpersisted: usize,
    counts: WarmDriverCounts,
}

impl BlockingTally {
    /// One more key, with its driver counts: its status for the per-key line (`READY` once every
    /// pass of it has a driver blob to reuse).
    fn note(&mut self, ok: bool, refused: bool, last: &WarmDriverCounts) -> &'static str {
        self.done += 1;
        self.counts.hits += last.hits;
        self.counts.cold += last.cold;
        self.counts.rejected += last.rejected;
        self.counts.saved += last.saved;
        self.counts.library_hits += last.library_hits;
        self.counts.library_misses += last.library_misses;
        self.counts.library_stored += last.library_stored;
        self.counts.library_refused += last.library_refused;
        if refused {
            self.deferred += 1;
            "deferred"
        } else if !ok {
            self.failed += 1;
            "failed"
        } else if last.hits + last.saved + last.library_hits == 0 || last.saved != last.cold {
            self.unpersisted += 1;
            "no reusable blob for every pass"
        } else {
            self.ready += 1;
            READY
        }
    }

    fn line(&self, total: usize, kind: &str, key: u64, status: &str, last: &WarmDriverCounts, elapsed: std::time::Duration, admitted: std::time::Duration, guest: usize) -> String {
        format!(
            "  {}/{total} {kind} {key:016x}: {status}; {} accepted, {} cold attempts, {} saved; {:.2} s ({:.2} s waiting for memory, {guest} B console shaders)",
            self.done, last.hits, last.cold, last.saved, elapsed.as_secs_f64(), admitted.as_secs_f64()
        )
    }

    /// The periodic line (every 2 s without the console's own progress line).
    fn checked(&self, total: usize, running: usize, giants: usize, compiling: std::time::Instant, deadline: std::time::Instant) -> String {
        use std::time::Instant;
        let eta = if BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed) || Instant::now() >= deadline {
            "stopping after the current calls".into()
        } else if self.done == 0 {
            "estimating".into()
        } else {
            let estimate = compiling.elapsed().as_secs_f64() * (total - self.done) as f64 / self.done as f64;
            format!("{:.1} keys/s, ~{estimate:.0} s for all remaining ({} s budget left)", self.done as f64 / compiling.elapsed().as_secs_f64(), deadline.saturating_duration_since(Instant::now()).as_secs())
        };
        format!("  {}/{total} checked; {running} driver compiles running ({giants} giants); {} driver hits, {} cold attempts; {} MiB RAM available; {eta}", self.done, self.counts.hits, self.counts.cold, available_physical() >> 20)
    }
}

#[cfg(test)]
mod pso_library_ready_tests {
    use super::{BlockingTally, WarmDriverCounts, READY};

    #[test]
    fn a_key_the_pipeline_library_serves_is_ready() {
        let mut tally = BlockingTally::default();
        let library = WarmDriverCounts { library_hits: 1, ..Default::default() };
        assert_eq!(tally.note(true, false, &library), READY);
        let blob = WarmDriverCounts { hits: 1, ..Default::default() };
        assert_eq!(tally.note(true, false, &blob), READY);
        let cold_unsaved = WarmDriverCounts { cold: 1, library_hits: 1, ..Default::default() };
        assert_ne!(tally.note(true, false, &cold_unsaved), READY, "a cold pass without its blob is not ready");
        assert_ne!(tally.note(true, false, &WarmDriverCounts::default()), READY);
        assert_eq!((tally.ready, tally.unpersisted, tally.counts.library_hits), (2, 2, 2));
    }
}

/// The keys of a blocking warm-up pass, read for it (`blocking_keys`).
struct BlockingKeys {
    records: Vec<(u64, super::pso_manifest::Record)>,
    progress: Option<super::pso_manifest::Progress>,
    observed: usize,
    skipped: usize,
    unavailable: usize,
    /// left out: an earlier pass marked them bad (`warm_children::bad_keys`)
    bad: usize,
}

/// The keys of a blocking warm-up pass in translation `era` with driver `driver`, read from the
/// manifest: those an earlier pass made ready and those it marked bad left out, at most
/// `max_keys`. None: nothing to replay (said why).
fn blocking_keys(targets: Option<&std::path::Path>, mut keys: Vec<u64>, era: u64, driver: &str, fresh: bool, max_keys: Option<usize>, budget: std::time::Duration) -> Result<Option<BlockingKeys>, String> {
    use super::pso_manifest::{self, Store};
    let manifest_dir = pso_manifest::dir();
    let observed = keys.len();
    let store = Store::open(manifest_dir.clone());
    if fresh {
        store.blocking_progress_clear();
        super::warm_children::clear_bad(&manifest_dir);
    }
    let progress = match store.blocking_progress(era, driver) {
        Ok((progress, changed)) => {
            if changed {
                eprintln!("warm-up --blocking: the adapter or driver changed (now {driver}): every key is replayed");
            }
            Some(progress)
        }
        Err(e) => {
            eprintln!("warm-up --blocking: no progress record ({e}): this pass cannot be resumed");
            None
        }
    };
    let skipped = progress.as_ref().map_or(0, |p| {
        keys.retain(|key| !p.done.contains(key));
        observed - keys.len()
    });
    if skipped != 0 {
        eprintln!("warm-up --blocking: resuming translation {era:016x}, driver {driver}: {skipped} keys ready in an earlier pass are skipped (--fresh replays them)");
    }
    let bad = {
        let bad = super::warm_children::bad_keys(&manifest_dir, era, driver);
        let before = keys.len();
        keys.retain(|key| !bad.contains(key));
        before - keys.len()
    };
    if keys.is_empty() {
        let bad_note = if bad == 0 { String::new() } else { format!("; {bad} of them marked bad by an earlier pass (--fresh tries them again)") };
        eprintln!("warm-up --blocking: all {observed} keys were ready in an earlier pass{bad_note}");
        return Ok(None);
    }
    let mut found: HashMap<_, _> = Store::load_keys(&manifest_dir, &keys.iter().copied().collect()).records.into_iter().map(|r| (r.key(), r)).collect();
    let unavailable = keys.len() - found.len();
    let mut records: Vec<_> = keys.iter().filter_map(|key| found.remove(key).map(|r| (*key, r))).collect();
    // (ray passes built with every ray a miss after every other key: `raytracing::miss_pass`)
    records.sort_by_cached_key(|(_, r)| super::raytracing::miss_pass(r.kind == super::pso_manifest::Kind::Compute, r.shaders.iter().map(|(_, s)| s.as_slice())));
    records.truncate(max_keys.unwrap_or(usize::MAX));
    drop(found);
    let total = records.len();
    let limit_note = max_keys.map_or(String::new(), |n| format!(", at most {n} by --max-keys"));
    eprintln!("warm-up --blocking: {total} replayable, {unavailable} missing or unreadable, {skipped} ready earlier, {observed} observed{limit_note}; budget {} s; {bad} marked bad by an earlier pass left out", budget.as_secs());
    if total == 0 && targets.is_some() && unavailable == keys.len() && skipped == 0 && bad == 0 {
        return Err("no supplied target resolves to a replayable manifest descriptor".into());
    }
    if total == 0 {
        return Ok(None);
    }
    Ok(Some(BlockingKeys { records, progress, observed, skipped, unavailable, bad }))
}

/// `xcl warm-up --blocking`: every key that held a bind or a first use made ready, its pipeline
/// built so the driver keeps a blob of each pass, in child processes (`children`: `warm_children`)
/// or, with None (`--in-process`), on one worker per logical core here. Ok: whether it was cancelled.
pub fn warm_up_blocking(targets: Option<&std::path::Path>, budget: std::time::Duration, max_keys: Option<usize>, fresh: bool, children: Option<super::warm_children::ChildOptions>) -> Result<bool, String> {
    use super::pso_manifest::{self, Store};
    let started = std::time::Instant::now();
    let cancelled = || BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed);
    let default = pso_manifest::blocking_path();
    let path = targets.unwrap_or(&default);
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if targets.is_none() && e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("cannot read the blocking pipeline list: {e}")),
    };
    let min_run = targets.is_none().then(|| Store::latest_run(&pso_manifest::dir()).saturating_sub(pso_manifest::STALE_AFTER_RUNS - 1));
    let keys = match min_run {
        Some(min_run) => {
            let mut used = super::pso_async::used_earlier();
            used.extend(super::pso_async::skipped_earlier());
            let t = pso_manifest::blocking_targets(&pso_manifest::dir(), &text, min_run, &used);
            eprintln!("warm-up --blocking: {} pipelines runs drew with ({} waited for, {} bound or used); {} kept pipelines no run drew with are left for their first bind (xcl warm-up --full builds them)",
                t.keys.len(), t.waited, t.drawn, t.left);
            t.keys
        }
        None => pso_manifest::blocking_order(&text, None),
    };
    if keys.is_empty() {
        if targets.is_some() {
            return Err("the supplied target file contains no observed waits".into());
        }
        eprintln!("warm-up --blocking: no observed blocking descriptors; launch normally to collect them");
        return Ok(cancelled());
    }
    if !driver_cache::on() {
        return Err("the driver cache is disabled by XCL_PSO_DISK_CACHE".to_string());
    }
    match children {
        Some(options) => warm_up_blocking_in_children(targets, keys, started, budget, max_keys, fresh, &options),
        None => warm_up_blocking_in_process(targets, keys, started, budget, max_keys, fresh),
    }
}

fn warm_up_blocking_in_process(targets: Option<&std::path::Path>, keys: Vec<u64>, started: std::time::Instant, budget: std::time::Duration, max_keys: Option<usize>, fresh: bool) -> Result<bool, String> {
    use std::time::{Duration, Instant};
    let deadline = started + budget;
    let cancelled = || BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed);
    WARM_UP_PROCESS.store(true, Ordering::Relaxed);
    super::pso_library::set_warm_slot(0);
    init()?;
    if crate::gpu::recomp::pso::mem_windows() as usize != crate::gpu::uma::MAX_WINDOWS {
        return Err("could not initialize the runtime shader memory windows".to_string());
    }
    let era = crate::gpu::recomp::pso::translation_era();
    let driver = DRIVER.get().map_or("unknown", String::as_str);
    let Some(BlockingKeys { records, progress, observed, skipped, unavailable, bad }) = blocking_keys(targets, keys, era, driver, fresh, max_keys, budget)? else {
        return Ok(cancelled());
    };
    let total = records.len();
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    if cancelled() || Instant::now() >= deadline {
        return Ok(cancelled());
    }
    let dev = device_raw() as usize;
    trim_mapped_files();
    let wanted = cores.min(total);
    // DXC's pool is sized once, for a running title (its reserve, 1 GB a compile): 12 of 24
    // threads on a PC with 18 GB free, while the keys' compiles queued 30 s behind its giants. Here
    // each key's translation and DXC compiles are admitted within the memory (`translation_slot`).
    if std::env::var_os("XCL_DXC_THREADS").is_none() {
        std::env::set_var("XCL_DXC_THREADS", cores.saturating_sub(1).max(1).to_string());
    }
    let queue = Mutex::new(records.into_iter());
    let (tx, rx) = std::sync::mpsc::channel();
    let handles = blocking_workers(
        wanted,
        dev,
        deadline,
        false,
        Arc::new(move || queue.lock().unwrap_or_else(|e| e.into_inner()).next()),
        Arc::new(move |key, report| drop(tx.send((key, report)))),
    )?;
    let workers = handles.len();
    eprintln!("warm-up --blocking: {workers} workers on {cores} logical cores; driver compiles admitted by their estimated peaks within the RAM available less {} MiB and the commit headroom less {} MiB; {} DXC threads",
        URGENT_FLOOR >> 20, URGENT_FLOOR >> 20, crate::gpu::recomp::dxc::compile_threads());
    let monitor = crate::progress_window::Monitor::start("warm-up --blocking", "keys", total, Some(cancel_blocking_warm_up));

    let mut tally = BlockingTally::default();
    let compiling = Instant::now();
    let mut min_available = available_physical();
    let (mut most_running, mut most_giants) = (0, 0);
    let mut reported = Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok((key, r)) => {
                let status = tally.note(r.ok, r.refused, &r.counts);
                if status == READY {
                    if let Some(p) = &progress {
                        p.note(key);
                    }
                }
                super::pso_manifest::note_build_time(key, r.build.as_millis() as u64);
                if !(monitor.console() && status == READY) {
                    monitor.println(&tally.line(total, &r.kind, key, status, &r.counts, r.elapsed, r.admitted, r.guest));
                }
                monitor.advance(tally.done, tally.failed);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        min_available = min_available.min(available_physical());
        let (running, giants, _) = DRIVER_COMPILES.in_flight();
        most_running = most_running.max(running);
        most_giants = most_giants.max(giants);
        if !monitor.console() && reported.elapsed() >= Duration::from_secs(2) {
            reported = Instant::now();
            eprintln!("{}", tally.checked(total, running, giants, compiling, deadline));
        }
    }
    drop(monitor);
    let mut panicked = 0;
    for handle in handles {
        panicked += handle.join().is_err() as usize;
    }
    if panicked != 0 {
        return Err(format!("{panicked} blocking warm-up workers panicked"));
    }
    super::pso_library::save_all(None);
    let (peak_working_set, peak_commit) = peak_process_memory();
    let physical = total_physical();
    let BlockingTally { done, ready, failed, deferred, unpersisted, counts } = tally;
    eprintln!("warm-up --blocking: {workers} workers on {cores} logical cores; keys {ready} accepted (ready), {failed} failed, {unpersisted} unpersisted, {deferred} deferred, {unavailable} unavailable, {skipped} ready earlier, {} not started (out of {total} replayable, {observed} observed); {} driver hits, {} cold attempts, {} rejected blobs, {} saved; wall {:.2} s; peak RAM {} MiB in use of {} MiB ({} MiB minimum sampled available), {} MiB peak working set, {} MiB peak private commit; at most {most_running} driver compiles at once ({most_giants} giants); in process; {bad} marked bad by an earlier pass left out; {}",
        total - done, counts.hits, counts.cold, counts.rejected, counts.saved, started.elapsed().as_secs_f64(), physical.saturating_sub(min_available) >> 20, physical >> 20, min_available >> 20, peak_working_set >> 20, peak_commit >> 20, library_line(&counts));
    eprintln!("warm-up shaders: {}", crate::gpu::recomp::dxc::stage_summary());
    Ok(cancelled())
}

/// The parent's side of a blocking warm-up in children (`warm_children::Pass`): the progress record,
/// the per-key, periodic and final lines as in process, and the bad list.
struct ChildPass<'a> {
    targets: Option<&'a std::path::Path>,
    keys: Vec<u64>,
    fresh: bool,
    max_keys: Option<usize>,
    budget: std::time::Duration,
    deadline: std::time::Instant,
    plan: super::warm_children::Plan,
    cores: usize,
    peers: Arc<super::warm_peers::Peers>,
    era: u64,
    driver: String,
    keyed: Option<KeyedPass>,
}

/// A `ChildPass` once its keys are known.
struct KeyedPass {
    progress: Option<super::pso_manifest::Progress>,
    observed: usize,
    skipped: usize,
    unavailable: usize,
    bad_earlier: usize,
    total: usize,
    kinds: HashMap<u64, String>,
    monitor: crate::progress_window::Monitor,
    tally: BlockingTally,
    compiling: std::time::Instant,
    min_available: u64,
    most_running: usize,
    most_giants: usize,
    reported: std::time::Instant,
    shaders: Vec<String>,
}

impl KeyedPass {
    fn key_line(&self, key: u64, status: &str, tail: &str) -> String {
        let kind = self.kinds.get(&key).map_or("?", String::as_str);
        let zero = std::time::Duration::ZERO;
        format!("{}{tail}", self.tally.line(self.total, kind, key, status, &WarmDriverCounts::default(), zero, zero, 0))
    }
}

impl super::warm_children::Pass for ChildPass<'_> {
    fn hello(&mut self, era: u64, driver: &str) -> Result<Vec<super::warm_children::Job>, String> {
        use super::warm_children::{Job, Lane, GIANT_GUEST_BYTES};
        self.era = era;
        self.driver = driver.to_string();
        let keys = std::mem::take(&mut self.keys);
        let Some(BlockingKeys { records, progress, observed, skipped, unavailable, bad }) = blocking_keys(self.targets, keys, era, driver, self.fresh, self.max_keys, self.budget)? else {
            return Ok(Vec::new());
        };
        if self.stopping() {
            return Ok(Vec::new());
        }
        let total = records.len();
        let jobs: Vec<Job> = records.iter().map(|(key, r)| Job::new(*key, r.shaders.iter().map(|(_, s)| s.len()).sum(), super::pso_manifest::slow_ms(*key))).collect();
        let kinds = records.iter().map(|(key, r)| (*key, format!("{:?}", r.kind))).collect();
        drop(records);
        let plan = &self.plan;
        let follow = plan.follow.map_or_else(
            || "keys given as workers free up (XCL_WARM_FOLLOW_MEMORY=0)".to_string(),
            |f| format!("keys given while the RAM available holds {} MiB for each key not yet showing (its first {} s) above the floor and a {} MiB margin (XCL_WARM_FOLLOW_MEMORY=0: as workers free up; XCL_WARM_MARGIN_MB, XCL_WARM_KEY_MB)", f.per_key >> 20, super::warm_children::FRESH.as_secs(), f.margin >> 20),
        );
        let replaced = if plan.strict_pool { "once it has exited (XCL_WARM_STRICT_POOL=0: beside it)" } else { "beside it (XCL_WARM_STRICT_POOL=0)" };
        eprintln!("warm-up --blocking: {} workers on {} logical cores; driver compiles admitted by their estimated peaks within the RAM available less {} MiB (and the commit headroom less as much); {} DXC threads; {follow}; in child processes: {} pool children of {} workers, each replaced after {} keys or {} MiB private commit by another started {replaced}, and up to {} single-key children for the {} predicted giants ({GIANT_GUEST_BYTES} B of console shaders, or built in {} s or more before) and the keys a dead child had in flight; a key running {} s (waits for memory left out) is taken for a hang",
            plan.workers(), self.cores, (URGENT_FLOOR + plan.margin()) >> 20, plan.dxc_threads(Lane::Pool), plan.pool_children, plan.pool_threads, plan.recycle_keys, plan.recycle_commit >> 20, plan.single_children, jobs.iter().filter(|j| j.giant).count(), super::warm_children::GIANT_BUILD_MS / 1000, plan.key_timeout.as_secs());
        let now = std::time::Instant::now();
        self.keyed = Some(KeyedPass {
            progress,
            observed,
            skipped,
            unavailable,
            bad_earlier: bad,
            total,
            kinds,
            monitor: crate::progress_window::Monitor::start("warm-up --blocking", "keys", total, Some(cancel_blocking_warm_up)),
            tally: BlockingTally::default(),
            compiling: now,
            min_available: available_physical(),
            most_running: 0,
            most_giants: 0,
            reported: now,
            shaders: Vec::new(),
        });
        Ok(jobs)
    }

    fn outcome(&mut self, outcome: &super::warm_children::Outcome) {
        use super::warm_children::Outcome;
        let Some(k) = &mut self.keyed else { return };
        match outcome {
            Outcome::Done(d) => {
                let last = WarmDriverCounts { hits: d.hits, cold: d.cold, rejected: d.rejected, saved: d.saved, library_hits: d.library_hits, library_misses: d.library_misses, library_stored: d.library_stored, library_refused: d.library_refused };
                let status = k.tally.note(d.ok, d.refused, &last);
                if status == READY {
                    if let Some(p) = &k.progress {
                        p.note(d.key);
                    }
                }
                super::pso_manifest::note_build_time(d.key, d.build_ms);
                if !(k.monitor.console() && status == READY) {
                    let (elapsed, admitted) = (std::time::Duration::from_micros(d.elapsed_us), std::time::Duration::from_micros(d.admitted_us));
                    k.monitor.println(&k.tally.line(k.total, &d.kind, d.key, status, &last, elapsed, admitted, d.guest));
                }
            }
            Outcome::Missing(key) => {
                let status = k.tally.note(false, false, &WarmDriverCounts::default());
                k.monitor.println(&k.key_line(*key, status, "; not in the manifest when its child read it"));
            }
            Outcome::Retry { key, why } => k.monitor.println(&format!("  warm-up --blocking: {key:016x} {why}: tried again alone in a child of its own")),
            Outcome::Bad { key, why } => {
                let status = k.tally.note(false, false, &WarmDriverCounts::default());
                let listed = match super::warm_children::note_bad(&super::pso_manifest::dir(), *key, self.era, &self.driver, why) {
                    Ok(()) => super::warm_children::BAD_FILE.to_string(),
                    Err(e) => format!("not listed: {e}"),
                };
                k.monitor.println(&k.key_line(*key, status, &format!("; marked bad ({listed}): alone, it {why}")));
            }
            Outcome::Hung { key, why } => {
                let status = k.tally.note(true, true, &WarmDriverCounts::default());
                k.monitor.println(&k.key_line(*key, status, &format!("; left for the next pass: alone, it {why} again")));
            }
            Outcome::Mismatch { era, driver } => k.monitor.println(&format!(
                "  warm-up --blocking: a child reported translation {era:016x}, driver {driver}, not this pass's {:016x}, {}: it was killed and its keys go to the others",
                self.era, self.driver
            )),
            Outcome::Giant(key) => k.monitor.println(&format!("  {key:016x}: a giant (driver compile of {} MiB or more of DXIL): left for a single-key child", GIANT_DXIL >> 20)),
            Outcome::Shaders(line) => k.shaders.push(line.clone()),
            Outcome::Hello { .. } => {}
        }
        k.monitor.advance(k.tally.done, k.tally.failed);
    }

    fn tick(&mut self, coordinator: &super::warm_children::Coordinator) {
        let Some(k) = &mut self.keyed else { return };
        let usage = self.peers.lock().total();
        k.min_available = k.min_available.min(available_physical());
        k.most_running = k.most_running.max(usage.running);
        k.most_giants = k.most_giants.max(usage.giants);
        if !k.monitor.console() && k.reported.elapsed() >= std::time::Duration::from_secs(2) {
            k.reported = std::time::Instant::now();
            let (pool, single) = coordinator.children();
            eprintln!("{}; {pool} pool and {single} single-key children", k.tally.checked(k.total, usage.running, usage.giants, k.compiling, self.deadline));
        }
    }

    fn stopping(&self) -> bool {
        BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed) || std::time::Instant::now() >= self.deadline
    }

    fn room(&self) -> super::warm_children::Room {
        super::warm_children::Room { available: available_physical(), usage: self.peers.lock().total() }
    }
}

/// `warm_up_blocking` with its compiles in child processes (`warm_children`); this process
/// starts no device of its own.
fn warm_up_blocking_in_children(targets: Option<&std::path::Path>, keys: Vec<u64>, started: std::time::Instant, budget: std::time::Duration, max_keys: Option<usize>, fresh: bool, options: &super::warm_children::ChildOptions) -> Result<bool, String> {
    use super::warm_children::{self, Plan, ProcessSpawner};
    let deadline = started + budget;
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let plan = Plan::new(options, cores).with_env(|name| std::env::var(name).ok());
    let peers = Arc::new(super::warm_peers::Peers::create(std::process::id()).map_err(|e| format!("cannot share the memory governor with warm-up children: {e}"))?);
    let mut spawner = ProcessSpawner::new(plan, deadline, peers.clone()).map_err(|e| format!("cannot start warm-up children: {e}"))?;
    let mut pass = ChildPass { targets, keys, fresh, max_keys, budget, deadline, plan, cores, peers, era: 0, driver: String::new(), keyed: None };
    let stats = warm_children::run(plan, &mut spawner, &mut pass);
    drop(spawner);
    let stats = stats?;
    let cancelled = BLOCKING_WARM_CANCELLED.load(Ordering::Relaxed);
    let Some(KeyedPass { observed, skipped, unavailable, bad_earlier, total, monitor, tally, min_available, most_running, most_giants, shaders, .. }) = pass.keyed else {
        return Ok(cancelled);
    };
    drop(monitor);
    let physical = total_physical();
    let BlockingTally { done, ready, failed, deferred, unpersisted, counts } = tally;
    eprintln!("warm-up --blocking: {} workers on {cores} logical cores; keys {ready} accepted (ready), {failed} failed, {unpersisted} unpersisted, {deferred} deferred, {unavailable} unavailable, {skipped} ready earlier, {} not started (out of {total} replayable, {observed} observed); {} driver hits, {} cold attempts, {} rejected blobs, {} saved; wall {:.2} s; peak RAM {} MiB in use of {} MiB ({} MiB minimum sampled available), {} MiB peak working set, {} MiB peak private commit; at most {most_running} driver compiles at once ({most_giants} giants); children {} started, {} recycled, {} crashed, {} hung; {} suspects retried alone, {} keys marked bad, {bad_earlier} marked bad by an earlier pass left out; {} keys left after hanging alone, {} children killed for another translation or driver; {} giants handed to single-key children; {} key feeds held for memory; {}",
        plan.workers(), total - done, counts.hits, counts.cold, counts.rejected, counts.saved, started.elapsed().as_secs_f64(), physical.saturating_sub(min_available) >> 20, physical >> 20, min_available >> 20, stats.peak_working_set >> 20, stats.peak_commit >> 20,
        stats.started, stats.recycled, stats.crashed, stats.hung, stats.retried, stats.bad, stats.hung_keys, stats.mismatched, stats.handed, stats.memory_holds, library_line(&counts));
    eprintln!("warm-up shaders: {}", warm_children::merge_summaries(&shaders));
    Ok(cancelled)
}

/// `xcl warm-up --blocking-child`: one child of a blocking warm-up in children (`warm_children`).
/// It says hello once its device is up, builds the keys it reads on stdin on `threads` workers as a
/// pass in process does, with one reply line per key on stdout, and its governor sees the
/// other children's compiles (`warm_peers`). `leave_giants` (a pool child): a key whose driver
/// compile is a giant is handed back (`Msg::Giant`) for a single-key child, its translation and
/// DXC work kept in the caches.
pub fn warm_up_blocking_child(threads: usize, budget: std::time::Duration, leave_giants: bool) -> Result<(), String> {
    use super::pso_manifest::{self, Store};
    use super::warm_children::{Msg, Order, Reply};
    use std::collections::VecDeque;
    use std::io::BufRead;
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + budget;
    if !driver_cache::on() {
        return Err("the driver cache is disabled by XCL_PSO_DISK_CACHE".to_string());
    }
    WARM_UP_PROCESS.store(true, Ordering::Relaxed);
    super::warm_peers::join_from_env()?;
    if let Some(slot) = super::warm_peers::joined().and_then(|peers| peers.own_slot()) {
        super::pso_library::set_warm_slot(slot);
    }
    init()?;
    if crate::gpu::recomp::pso::mem_windows() as usize != crate::gpu::uma::MAX_WINDOWS {
        return Err("could not initialize the runtime shader memory windows".to_string());
    }
    if std::env::var_os("XCL_DXC_THREADS").is_none() {
        std::env::set_var("XCL_DXC_THREADS", threads.to_string());
    }
    let reply = Reply::stdout();
    reply.send(&Msg::Hello { era: crate::gpu::recomp::pso::translation_era(), driver: DRIVER.get().map_or("unknown", String::as_str).to_string() });
    let dev = device_raw() as usize;
    trim_mapped_files();
    // the list read once; each worker decodes the keys it takes, so reading orders never waits on it
    let index = Arc::new(Store::index(&pso_manifest::dir()));
    type Queue = (Mutex<(VecDeque<u64>, bool)>, std::sync::Condvar);
    let queue: Arc<Queue> = Arc::new((Mutex::new((VecDeque::new(), false)), std::sync::Condvar::new()));
    let orders = queue.clone();
    std::thread::Builder::new().name("xcl-warm-child-orders".into()).spawn(move || {
        let (state, ready) = &*orders;
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            match Order::parse(&line) {
                Some(Order::Keys(keys)) => state.lock().unwrap_or_else(|e| e.into_inner()).0.extend(keys),
                Some(Order::End) => state.lock().unwrap_or_else(|e| e.into_inner()).1 = true,
                Some(Order::Stop) => {
                    cancel_blocking_warm_up();
                    state.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
                }
                None => eprintln!("warm-up --blocking-child: not an order: {line:?}"),
            }
            ready.notify_all();
        }
        state.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
        ready.notify_all();
    }).map_err(|e| format!("could not read the warm-up's orders: {e}"))?;
    let next_queue = queue.clone();
    let handles = blocking_workers(
        threads.max(1),
        dev,
        deadline,
        leave_giants,
        Arc::new(move || {
            let (state, ready) = &*next_queue;
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(key) = state.0.pop_front() {
                    drop(state);
                    match index.get(key) {
                        Some(record) => {
                            reply.send(&Msg::Start { key });
                            super::warm_children::working_on(Some(key));
                            return Some((key, record));
                        }
                        None => reply.send(&Msg::Missing { key }),
                    }
                    state = next_queue.0.lock().unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                if state.1 || blocking_warm_stopping() {
                    return None;
                }
                state = ready.wait_timeout(state, Duration::from_millis(250)).unwrap_or_else(|e| e.into_inner()).0;
            }
        }),
        Arc::new(move |key, r: KeyReport| {
            super::warm_children::working_on(None);
            if let Some(dxil_len) = r.giant {
                return reply.send(&Msg::Giant { key, dxil_len });
            }
            reply.send(&Msg::Done(super::warm_children::Done {
                key,
                kind: r.kind,
                ok: r.ok,
                refused: r.refused,
                hits: r.counts.hits,
                cold: r.counts.cold,
                rejected: r.counts.rejected,
                saved: r.counts.saved,
                elapsed_us: r.elapsed.as_micros() as u64,
                admitted_us: r.admitted.as_micros() as u64,
                build_ms: r.build.as_millis() as u64,
                guest: r.guest,
                library_hits: r.counts.library_hits,
                library_misses: r.counts.library_misses,
                library_stored: r.counts.library_stored,
                library_refused: r.counts.library_refused,
            }))
        }),
    )?;
    let memory = || {
        let (peak_working_set, peak_commit) = peak_process_memory();
        Msg::Mem { commit: crate::util::process_commit_bytes(), peak_working_set, peak_commit }
    };
    let working = Arc::new(AtomicBool::new(true));
    let still = working.clone();
    let sampler = std::thread::Builder::new().name("xcl-warm-child-memory".into()).spawn(move || {
        while still.load(Ordering::Relaxed) {
            reply.send(&memory());
            std::thread::sleep(Duration::from_secs(2));
        }
    });
    let mut panicked = 0;
    for handle in handles {
        panicked += handle.join().is_err() as usize;
    }
    working.store(false, Ordering::Relaxed);
    drop(sampler);
    if panicked != 0 {
        return Err(format!("{panicked} blocking warm-up workers panicked"));
    }
    super::pso_library::save_all(None);
    reply.send(&memory());
    reply.send(&Msg::Bye { shaders: crate::gpu::recomp::dxc::stage_summary() });
    Ok(())
}

/// (peak working set, peak private commit) of this process.
fn peak_process_memory() -> (u64, u64) {
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        let mut m: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        m.cb = std::mem::size_of_val(&m) as u32;
        if K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut m, m.cb) != 0 { (m.PeakWorkingSetSize as u64, m.PeakPagefileUsage as u64) } else { (0, 0) }
    }
}

/// The private memory `xcl warm-up` keeps within (`DRIVER_COMPILES`).
const WARM_UP_COMMIT: u64 = 12 << 30;

/// `cancel_warm_up` was called
static WARM_UP_CANCEL: AtomicBool = AtomicBool::new(false);
/// times `pso_sched::admit_why` held a compile back on its memory tests (a "memory" or "commit room"
/// reason: RAM available less the reservations short of its estimate and floor, or the commit
/// headroom short of its estimate and `URGENT_FLOOR`), read as `WarmStatus::short_of_ram`
static RAM_HOLDS: AtomicUsize = AtomicUsize::new(0);

/// `warm_up` starts no more builds: a compile waiting for its slot gives up, those running
/// finish, and what was built stays (driver blobs, and the `--full` record of the pipelines gone
/// through, written as each one is), so the next warm-up goes on from there.
pub fn cancel_warm_up() {
    WARM_UP_CANCEL.store(true, Ordering::Relaxed);
}

fn warm_up_cancelled() -> bool {
    WARM_UP_CANCEL.load(Ordering::Relaxed)
}

#[derive(Default)]
pub struct WarmMesh {
    pub kept: usize,
    pub built: usize,
    pub failed: usize,
    pub too_big: usize,
    pub threads: usize,
    pub took: std::time::Duration,
}

type MeshBlobLocks = Mutex<HashMap<std::path::PathBuf, Arc<Mutex<()>>>>;

struct MeshWarmBuild<'a> {
    locks: &'a MeshBlobLocks,
    built: bool,
    blobs: Vec<std::path::PathBuf>,
}

impl MeshWarmBuild<'_> {
    /// `device`: the device it is made on, for the pipeline library.
    unsafe fn blob(&mut self, device: *mut c_void, path: Option<&std::path::Path>, len: usize, what: &'static str, create: impl FnOnce(*mut *mut c_void) -> HRESULT) -> Result<HRESULT, ()> {
        let Some(path) = path else {
            note_translation_error(format!("{what} has no driver cache key"));
            return Err(());
        };
        let lock = self.locks.lock().unwrap().entry(path.to_owned()).or_default().clone();
        let _lock = lock.lock().unwrap_or_else(|e| e.into_inner());
        if driver_cache::load(path).is_some() {
            self.blobs.push(path.to_owned());
            return Ok(S_OK);
        }
        let slot = driver_slot(len, true, false, || what.into()).map_err(|_| ())?;
        let mut pso = std::ptr::null_mut();
        let hr = create(&mut pso);
        let blob = (hr >= 0 && !pso.is_null()).then(|| pso_blob(pso)).flatten();
        if hr >= 0 {
            library_store(device, Some(path), None, pso);
        }
        super::pso_async::release(pso as usize);
        drop(slot);
        if hr < 0 {
            return Ok(hr);
        }
        if !blob.is_some_and(|b| driver_cache::write(path, &b)) {
            note_translation_error(format!("{what} did not yield a usable driver blob"));
            return Err(());
        }
        self.built = true;
        self.blobs.push(path.to_owned());
        Ok(S_OK)
    }
}

pub fn warm_up_mesh(recent_used: Option<std::num::NonZeroU32>) -> Result<WarmMesh, String> {
    use super::pso_manifest::Store;
    use super::pso_manifest::Kind::MeshStream;
    let started = std::time::Instant::now();
    let dir = super::pso_manifest::dir();
    let recent = if let Some(n) = recent_used {
        let (runs, created) = Store::recently_created(&dir, n).map_err(|e| format!("cannot read recent mesh pipeline history: {e}"))?;
        let mut used = super::pso_async::used_earlier();
        used.extend(super::pso_async::skipped_earlier());
        eprintln!("warm-up --mesh-only: scope is counted runs {}–{} and shaders previously bound or skipped", runs.start(), runs.end());
        Some((created, used))
    } else {
        None
    };
    let mut startup = None;
    let mut kept = 0;
    let loaded = Store::load_kind(&dir, MeshStream, |key, shader_key| {
        if let Some((created, used)) = &recent {
            if !created.contains(&key) || !shader_key.is_some_and(|key| used.contains(&key)) {
                return false;
            }
        }
        let result = startup.get_or_insert_with(|| {
            if !driver_cache::on() {
                return Err("the driver cache is disabled by XCL_PSO_DISK_CACHE".to_string());
            }
            init()?;
            if crate::gpu::recomp::pso::mem_windows() as usize != crate::gpu::uma::MAX_WINDOWS {
                return Err("could not initialize the runtime shader memory windows".to_string());
            }
            let dev = device_raw();
            Ok((dev as usize, driver_cache::mesh_coverage_dir(dev)))
        });
        let Ok((_, coverage)) = result else { return false };
        let hit = coverage.as_deref().is_some_and(|dir| driver_cache::mesh_covered(dir, key));
        kept += usize::from(hit);
        !hit
    });
    let Some(startup) = startup else {
        return Ok(WarmMesh { took: started.elapsed(), ..Default::default() });
    };
    let (dev, coverage) = startup?;
    let mut queue: Vec<_> = loaded.records.iter().map(|r| (r, false)).collect();
    let total = kept + loaded.broken + queue.len();
    eprintln!("warm-up --mesh-only: {total} selected pipelines, {kept} already covered, {} to check or build, {} unreadable", queue.len(), loaded.broken);
    if queue.is_empty() {
        return Ok(WarmMesh { kept, failed: loaded.broken, took: started.elapsed(), ..Default::default() });
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(queue.len());
    let results = Mutex::new(WarmMesh { kept, failed: loaded.broken, threads, ..Default::default() });
    let locks = MeshBlobLocks::default();
    queue.sort_unstable_by_key(|(r, _)| std::cmp::Reverse(r.size()));
    let ready = kept + loaded.broken;
    std::thread::scope(|s| -> Result<(), String> {
        let (_finish, stop) = std::sync::mpsc::channel::<()>();
        let results = &results;
        let compiling = std::time::Instant::now();
        s.spawn(move || {
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = stop.recv_timeout(std::time::Duration::from_secs(10)) {
                let w = results.lock().unwrap();
                let done = w.kept + w.built + w.failed + w.too_big;
                let checked = done.saturating_sub(ready);
                let eta = if checked == 0 {
                    "estimating".into()
                } else {
                    format!("{:.0} s", compiling.elapsed().as_secs_f64() * total.saturating_sub(done) as f64 / checked as f64)
                };
                eprintln!("warm-up --mesh-only: {done}/{total} pipelines checked, {} warmed, {} cached, {} failed or refused, ETA {eta}", w.built, w.kept, w.failed + w.too_big);
            }
        });
        for round in 0..=RETRY_ROUNDS {
            let (next, later) = (AtomicUsize::new(0), Mutex::new(Vec::new()));
            std::thread::scope(|scope| -> Result<(), String> {
                for k in 0..threads.min(queue.len()) {
                    let (next, later, queue, locks, results) = (&next, &later, &queue, &locks, &results);
                    let coverage = coverage.as_deref();
                    std::thread::Builder::new().name(format!("xcl-pso-warm-mesh-{k}")).stack_size(64 << 20).spawn_scoped(scope, move || {
                        WARMING_UP.with(|c| c.set(Some(threads)));
                        while let Some(&(r, built)) = queue.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let mut warm = MeshWarmBuild { locks, built, blobs: Vec::new() };
                            REFUSED.with(|c| c.set(None));
                            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                                let Some(owned) = OwnedDesc::from_record(r) else { return false };
                                let mut out = std::ptr::null_mut();
                                matches!(translated_mesh_stream(dev as *mut c_void, owned.ptr(), &pc_iid(Kind::PipelineState), &mut out, Some(&mut warm)), Some(hr) if hr >= 0)
                            })).unwrap_or_else(|p| {
                                eprintln!("  mesh pipeline {:016x} panicked: {}", r.key(), panic_text(&*p));
                                false
                            });
                            match REFUSED.with(|c| c.get()) {
                                Some(Refused::Later) if round < RETRY_ROUNDS => later.lock().unwrap().push((r, warm.built)),
                                Some(_) => results.lock().unwrap().too_big += 1,
                                None => {
                                    if let (true, Some(dir)) = (ok, coverage) {
                                        driver_cache::note_mesh_coverage(dir, r.key(), &warm.blobs);
                                    }
                                    let mut w = results.lock().unwrap();
                                    if !ok {
                                        w.failed += 1;
                                    } else if warm.built {
                                        w.built += 1;
                                    } else {
                                        w.kept += 1;
                                    }
                                }
                            }
                        }
                    }).map_err(|e| format!("could not start a mesh warm-up worker: {e}"))?;
                }
                Ok(())
            })?;
            queue = later.into_inner().unwrap();
            if queue.is_empty() {
                break;
            }
        }
        Ok(())
    })?;
    let mut w = results.into_inner().unwrap();
    w.took = started.elapsed();
    Ok(w)
}

/// One phase of `xcl warm-up`.
pub struct WarmPhase {
    pub name: &'static str,
    /// pipelines (or programs) it had
    pub of: usize,
    pub built: usize,
    pub failed: usize,
    /// giants (`GIANT_DXIL`) with no driver blob kept, left for `--full` and the run's own builds
    pub giants_left: usize,
    /// too big to build within the memory limits: at once (`Refused::TooBig`), or with no room
    /// still after `RETRY_ROUNDS` retries
    pub too_big: usize,
    /// not started when the budget ran out
    pub not_started: usize,
    pub took: std::time::Duration,
}

/// One giant of `warm_up_giants`: its DXIL (hash, length), what became of it, the seconds it took
pub struct GiantBuild {
    pub hash: u64,
    pub len: usize,
    pub outcome: &'static str,
    pub took: std::time::Duration,
}

/// `xcl warm-up --giants-only --floor-mb N`: the driver blobs of every giant compute program on
/// disk (`GIANT_DXIL` or more of DXIL, no blob kept), built one at a time keeping `floor`
/// available instead of the normal floor (`KEEP_FREE`). The game launches without a warm-up, and
/// a giant cannot build within the game's memory, so none is left for later. Only this explicit
/// command lowers the floor: never a run of the title or the normal warm-up. Returns each
/// giant's outcome and the process's peak commit.
pub fn warm_up_giants(floor: u64) -> Result<(Vec<GiantBuild>, u64), String> {
    init()?;
    let dev = device_raw() as usize;
    let programs = giants_without_blob();
    let mut out = Vec::new();
    for c in &programs {
        let (hash, len, t0) = (crate::gpu::shader::fnv64(&c.dxil), c.dxil.len(), std::time::Instant::now());
        KEEP_FREE_NOW.store(floor, Ordering::Relaxed);
        let outcome = match DRIVER_COMPILES.acquire(len, 1, None, true, || false) {
            Err(Refused::TooBig) => "too big even for the lowered floor",
            Err(_) => "no room within the lowered floor now",
            Ok(slot) => {
                let ok = std::panic::catch_unwind(|| unsafe { warm_one(dev as *mut c_void, c) }).unwrap_or(false);
                drop(slot);
                match (ok, driver_cache::compute_path(&c.dxil, &c.samplers, 0, 0).and_then(|p| driver_cache::load(&p)).is_some()) {
                    (true, true) => "built, its driver blob kept",
                    (true, false) => "built, but no driver blob kept",
                    (false, _) => "the driver build failed",
                }
            }
        };
        KEEP_FREE_NOW.store(KEEP_FREE, Ordering::Relaxed);
        out.push(GiantBuild { hash, len, outcome, took: t0.elapsed() });
    }
    let peak_commit = unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        let mut m: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        m.cb = std::mem::size_of_val(&m) as u32;
        if K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut m, m.cb) != 0 { m.PeakPagefileUsage as u64 } else { 0 }
    };
    Ok((out, peak_commit))
}

/// The giant compute programs translated on disk (`GIANT_DXIL` or more of DXIL) with no driver
/// blob kept, largest first; their counts logged.
fn giants_without_blob() -> Vec<crate::gpu::recomp::pso::Compiled> {
    let mut seen = HashSet::new();
    let mut programs = crate::gpu::recomp::pso::translated_on_disk(crate::gpu::recomp::pso::Stage::Compute);
    let on_disk = programs.len();
    programs.retain(|c| c.dxil.len() >= GIANT_DXIL && seen.insert((driver_cache::name(&c.dxil, 0), c.samplers.clone())));
    let giants = programs.len();
    programs.retain(|c| driver_cache::compute_path(&c.dxil, &c.samplers, 0, 0).is_some_and(|p| driver_cache::load(&p).is_none()));
    eprintln!("  {on_disk} compute programs translated on disk (translation {:016x}), {giants} giants, {} of them with a driver blob kept", crate::gpu::recomp::pso::translation_era(), giants - programs.len());
    programs.sort_by_key(|c| std::cmp::Reverse(c.dxil.len()));
    programs
}

pub struct WarmUp {
    /// `--full`: every pipeline and giant, no budget
    pub full: bool,
    /// pipelines earlier runs made (`pso_manifest`) and kept entries dropped (unreadable, a shader
    /// missing, or kept twice)
    pub pipelines: usize,
    pub dropped: usize,
    /// of them: last made in this translation era; left out as stale (no run of the last
    /// `STALE_AFTER_RUNS` made them), of the runs counted
    pub this_era: usize,
    pub stale_left: usize,
    pub runs: u32,
    /// `--full`: pipelines an earlier pass of this era went through, not built again; and whether
    /// this pass went through the rest (the next one starts again)
    pub resumed: usize,
    pub complete: bool,
    /// `cancel_warm_up` stopped it with builds not started (`WarmPhase::not_started`)
    pub cancelled: bool,
    pub phases: Vec<WarmPhase>,
    /// translated compute programs on disk, their blobs kept already
    pub programs: usize,
    pub kept: usize,
    /// driver blobs of DXIL no translation on disk has any more, deleted (count, bytes)
    pub pruned: (usize, u64),
    pub threads: usize,
    pub took: std::time::Duration,
    pub peak_commit: u64,
}

impl WarmUp {
    pub fn failed(&self) -> usize {
        self.phases.iter().map(|p| p.failed).sum()
    }

    pub fn too_big(&self) -> usize {
        self.phases.iter().map(|p| p.too_big).sum()
    }
}

/// How `warm_up` is getting on, told about once a second (`xcl warm-up --progress-json`).
#[derive(Clone, Copy, Debug)]
pub struct WarmStatus {
    pub phase: &'static str,
    /// builds of the phase gone through, of `total` (as `warm_up`'s `progress` counts them)
    pub done: usize,
    pub total: usize,
    /// builds that failed, in this phase and those before
    pub failed: usize,
    /// the pipelines this pass goes through (the phases' kept pipelines; giants and compute blobs
    /// are found as their phases start) and the threads it builds on
    pub pipelines: usize,
    pub threads: usize,
    /// driver compiles in flight (`DRIVER_COMPILES`) and the giants (`GIANT_DXIL` and up) among
    /// them; the memory alone decides how many run (`pso_sched::admit`), giants with no count of them
    pub in_flight: usize,
    pub giants: usize,
    /// a compile was held back on the memory tests of `pso_sched::admit` (RAM or commit headroom)
    /// since the last report
    pub short_of_ram: bool,
}

/// Rounds a warm-up phase tries again the builds it had no room for (`Refused::Later`).
const RETRY_ROUNDS: usize = 3;

/// The budgeted warm-up's last phase starts no build after this long (`XCL_WARMUP_BUDGET_S`).
const WARM_UP_BUDGET: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// `xcl warm-up`: the pipelines runs of the title made (`pso_manifest`), built as the title's
/// pipelines are – translated, compiled, the driver's compile – so their DXIL and the driver's
/// results are in the disk caches (an NVIDIA driver: 15 – 180 s for each big compute program,
/// 0.1 s the next time), in phases: (1) those an earlier run bound (`pso_async::used_earlier`); (2)
/// every compute pipeline, then the driver's blob of every translated compute program on disk
/// not kept yet (`driver_cache`; blobs of DXIL no translation has any more are deleted); (3) the
/// rest, within `WARM_UP_BUDGET` and without the stale ones, those a counted run made first. Each
/// phase largest first; the giants
/// (`GIANT_DXIL`) with no blob kept are left out of (2). `full` (`xcl warm-up --full`): every
/// pipeline and giant with no budget, going on where a pass of this translation era stopped.
/// Builds run on every core within the memory left (`DRIVER_COMPILES`, the private memory within
/// `WARM_UP_COMMIT`); the memory available and the builds running are logged every 10 s.
/// `progress` is told of each one built (done, of, bytes, how long, its key or DXIL hash), `status`
/// about once a second from when the builds start. After `cancel_warm_up` no build starts.
pub fn warm_up(full: bool, progress: impl Fn(usize, usize, usize, std::time::Duration, u64) + Sync, status: impl Fn(&WarmStatus) + Sync) -> Result<WarmUp, String> {
    use super::pso_manifest::{Kind, Record};
    let started = std::time::Instant::now();
    init()?;
    let dev = device_raw() as usize;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(GIANTS_AT_ONCE + 1);
    let done = std::sync::atomic::AtomicBool::new(false);
    let (built, failed, left, big) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let failed_before = AtomicUsize::new(0);
    let (phase, phase_total) = (std::sync::Mutex::new("pipelines"), AtomicUsize::new(0));
    let store = super::pso_manifest::Store::open(super::pso_manifest::dir());
    let loaded = store.load();
    let dropped = loaded.broken + loaded.repeated;
    if dropped > 0 {
        let _ = store.rewrite(&loaded.records);
    }
    let era = crate::gpu::recomp::pso::translation_era();
    let creations = store.creations();
    let mut pipelines: Vec<(Record, u64)> = loaded.records.into_iter().map(|r| {
        let key = r.key();
        (r, key)
    }).collect();
    let _ = store.compact_creations(&creations, &pipelines.iter().map(|p| p.1).collect());
    pipelines.sort_by_key(|p| std::cmp::Reverse(p.0.size()));
    let kept_pipelines = pipelines.len();
    let this_era = pipelines.iter().filter(|p| creations.era(p.1) == Some(era)).count();
    let record = full.then(|| store.progress(era).map_err(|e| eprintln!("  no --full progress record ({e}): this pass cannot be resumed")).ok()).flatten();
    let resumed = record.as_ref().map_or(0, |p| pipelines.iter().filter(|x| p.done.contains(&x.1)).count());
    if let Some(p) = &record {
        pipelines.retain(|x| !p.done.contains(&x.1));
    }
    let (used, skipped) = (super::pso_async::used_earlier(), super::pso_async::skipped_earlier());
    let (mut bound, mut later, mut compute, mut rest, mut stale_left) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0);
    for p in &pipelines {
        let shaders = unsafe { OwnedDesc::from_record(&p.0) }.map_or(0, |o| o.shaders_key());
        if skipped.contains(&shaders) {
            bound.push(p);
        } else if used.contains(&shaders) {
            later.push(p);
        } else if p.0.kind == Kind::Compute {
            compute.push(p);
        } else if full || !creations.stale(p.1) {
            rest.push(p);
        } else {
            stale_left += 1;
        }
    }
    // the pipelines a draw was skipped for first, then the others an earlier run bound
    bound.append(&mut later);
    // those a counted run made (graphics pipelines its boot and loading made) before those none did
    rest.sort_by_key(|p| creations.era(p.1).is_none());
    let budget = (!full).then(|| std::env::var("XCL_WARMUP_BUDGET_S").ok().and_then(|v| v.trim().parse().ok()).map_or(WARM_UP_BUDGET, std::time::Duration::from_secs));
    let mut phases = Vec::new();
    let (mut programs_found, mut kept, mut pruned) = (0, 0, (0, 0));
    let targets = bound.len() + compute.len() + rest.len();
    std::thread::scope(|s| {
        let (done, built, failed, left, big, phase, phase_total, failed_before, status) = (&done, &built, &failed, &left, &big, &phase, &phase_total, &failed_before, &status);
        // the logger ends with the builds, however they end (a panic unwinding out of here waited
        // for it for ever)
        struct Done<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _done = Done(done);
        // `XCL_WARMUP_THREAD_DUMP_S`: where every thread is, each so many seconds of the last phase
        let dump_every = std::env::var("XCL_WARMUP_THREAD_DUMP_S").ok().and_then(|v| v.trim().parse().ok()).map(std::time::Duration::from_secs);
        s.spawn(move || {
            let mut last = std::time::Instant::now();
            let mut last_dump = std::time::Instant::now();
            let mut ram_holds = RAM_HOLDS.load(Ordering::Relaxed);
            let mut tell = || {
                let (n, g, _) = DRIVER_COMPILES.in_flight();
                let holds = RAM_HOLDS.load(Ordering::Relaxed);
                status(&WarmStatus {
                    phase: *phase.lock().unwrap(),
                    done: built.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed) + left.load(Ordering::Relaxed) + big.load(Ordering::Relaxed),
                    total: phase_total.load(Ordering::Relaxed),
                    failed: failed_before.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed),
                    pipelines: targets,
                    threads,
                    in_flight: n,
                    giants: g,
                    short_of_ram: std::mem::replace(&mut ram_holds, holds) != holds,
                });
            };
            tell();
            let mut last_status = std::time::Instant::now();
            while !done.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if last_status.elapsed() >= std::time::Duration::from_secs(1) {
                    last_status = std::time::Instant::now();
                    tell();
                }
                if dump_every.is_some_and(|n| last_dump.elapsed() >= n) && *phase.lock().unwrap() == "the rest" {
                    last_dump = std::time::Instant::now();
                    crate::eh::dump_threads();
                }
                if last.elapsed() >= std::time::Duration::from_secs(10) {
                    last = std::time::Instant::now();
                    let (n, g, _) = DRIVER_COMPILES.in_flight();
                    eprintln!(
                        "  {:.0} s: {}: {} of {} done; {n} compute driver compiles ({g} giants), {} graphics pipelines building; {} MB available, {} MB private",
                        started.elapsed().as_secs_f64(),
                        phase.lock().unwrap(),
                        built.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed) + left.load(Ordering::Relaxed) + big.load(Ordering::Relaxed),
                        phase_total.load(Ordering::Relaxed),
                        WARM_GRAPHICS.load(Ordering::Relaxed),
                        available_physical() >> 20,
                        crate::util::process_commit_bytes() >> 20
                    );
                }
            }
        });
        let finish = |name: &'static str, of: usize, not_started: usize, t0: std::time::Instant| WarmPhase {
            name,
            of,
            built: built.swap(0, Ordering::Relaxed),
            failed: {
                let n = failed.swap(0, Ordering::Relaxed);
                failed_before.fetch_add(n, Ordering::Relaxed);
                n
            },
            giants_left: left.swap(0, Ordering::Relaxed),
            too_big: big.swap(0, Ordering::Relaxed),
            not_started,
            took: t0.elapsed(),
        };
        // kept pipelines, on threads XCL's governor knows (`driver_slot`); those it had no room for
        // (`Refused::Later`) go again at the end, `RETRY_ROUNDS` times at most
        let run = |name: &'static str, list: &[&(Record, u64)], leave_giants: bool, budget: Option<std::time::Duration>| {
            let t0 = std::time::Instant::now();
            let deadline = budget.map(|b| t0 + b);
            *phase.lock().unwrap() = name;
            phase_total.store(list.len(), Ordering::Relaxed);
            let (mut queue, mut not_started) = (list.to_vec(), 0);
            for round in 0..=RETRY_ROUNDS {
                let (next, later) = (AtomicUsize::new(0), Mutex::new(Vec::new()));
                std::thread::scope(|w| {
                    for k in 0..threads {
                        let (next, later, queue, progress, record) = (&next, &later, &queue, &progress, &record);
                        let _ = std::thread::Builder::new().name(format!("xcl-pso-warm-{k}")).stack_size(64 << 20).spawn_scoped(w, move || {
                            WARMING_UP.with(|c| c.set(Some(threads)));
                            LEAVE_GIANTS.with(|c| c.set(leave_giants));
                            while deadline.is_none_or(|d| std::time::Instant::now() < d) && !warm_up_cancelled() {
                                let Some(&p) = queue.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                                let (r, key) = (&p.0, p.1);
                                let t0 = std::time::Instant::now();
                                REFUSED.with(|c| c.set(None));
                                let ok = std::panic::catch_unwind(|| unsafe { warm_pipeline(dev as *mut c_void, r) }).unwrap_or_else(|p| {
                                    eprintln!("  a pipeline build panicked ({:?} {key:016x}, {} B of shaders): {}", r.kind, r.size(), panic_text(&*p));
                                    false
                                });
                                if t0.elapsed() >= std::time::Duration::from_secs(30) {
                                    eprintln!(
                                        "  slow, {name}: {:?} pipeline {key:016x}, {}: {:.0} s",
                                        r.kind,
                                        r.shaders.iter().map(|(at, s)| format!("{} {} B", super::pso_manifest::stage_at(r, *at), s.len())).collect::<Vec<_>>().join(", "),
                                        t0.elapsed().as_secs_f64()
                                    );
                                }
                                match REFUSED.with(|c| c.get()) {
                                    Some(Refused::Giant) => drop(left.fetch_add(1, Ordering::Relaxed)),
                                    Some(Refused::Later | Refused::Cancelled) => later.lock().unwrap().push(p),
                                    Some(Refused::TooBig) => drop(big.fetch_add(1, Ordering::Relaxed)),
                                    _ => {
                                        (if ok { built } else { failed }).fetch_add(1, Ordering::Relaxed);
                                        if let Some(p) = record {
                                            p.note(key);
                                        }
                                    }
                                }
                                progress(built.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed) + left.load(Ordering::Relaxed) + big.load(Ordering::Relaxed), list.len(), r.size(), t0.elapsed(), key);
                            }
                        });
                    }
                });
                not_started += queue.len().saturating_sub(next.into_inner());
                queue = later.into_inner().unwrap();
                if warm_up_cancelled() {
                    not_started += std::mem::take(&mut queue).len();
                }
                if queue.is_empty() {
                    break;
                }
                if round < RETRY_ROUNDS {
                    eprintln!("  {name}: {} had no room for their driver compile, tried again (round {} of {RETRY_ROUNDS})", queue.len(), round + 1);
                }
            }
            for p in &queue {
                eprintln!("  {name}: {:?} pipeline {:016x} still had no room after {RETRY_ROUNDS} retries: too big for now", p.0.kind, p.1);
            }
            big.fetch_add(queue.len(), Ordering::Relaxed);
            finish(name, list.len(), not_started, t0)
        };
        // the giants with no driver blob, one at a time, first: the PC's memory is at its freest
        // now (the driver's mapped shader cache trimmed, nothing else compiling)
        {
            let t0 = std::time::Instant::now();
            *phase.lock().unwrap() = "giants first";
            trim_mapped_files();
            let giants = giants_without_blob();
            phase_total.store(giants.len(), Ordering::Relaxed);
            let mut not_started = 0;
            for (k, c) in giants.iter().enumerate() {
                let (hash, len) = (crate::gpu::shader::fnv64(&c.dxil), c.dxil.len());
                let slot = DRIVER_COMPILES.acquire(len, 1, None, true, || false);
                if matches!(slot, Err(Refused::Cancelled)) {
                    not_started = giants.len() - k;
                    break;
                }
                let Ok(slot) = slot else {
                    eprintln!("  giants first: DXIL {hash:016x} ({len} B, ~{} MB expected peak) does not fit with {} MB available: for `xcl warm-up --giants-only`", peak(len) >> 20, available_physical() >> 20);
                    big.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let t1 = std::time::Instant::now();
                let ok = std::panic::catch_unwind(|| unsafe { warm_one(dev as *mut c_void, c) }).unwrap_or(false);
                drop(slot);
                eprintln!("  giants first: DXIL {hash:016x} ({len} B): {} in {:.0} s", if ok { "built" } else { "the driver build failed" }, t1.elapsed().as_secs_f64());
                (if ok { built } else { failed }).fetch_add(1, Ordering::Relaxed);
            }
            phases.push(finish("giants first", giants.len(), not_started, t0));
        }
        phases.push(run("bound by an earlier run", &bound, false, None));
        phases.push(run("compute pipelines", &compute, !full, None));
        // the compute programs on disk: the blobs not kept yet; those no program has any more go
        let t0 = std::time::Instant::now();
        *phase.lock().unwrap() = "compute blobs";
        let mut seen = HashSet::new();
        let mut programs = crate::gpu::recomp::pso::translated_on_disk(crate::gpu::recomp::pso::Stage::Compute);
        pruned = driver_cache::prune(&programs.iter().map(|c| (crate::gpu::shader::fnv64(&c.dxil), c.dxil.len())).collect());
        programs.retain(|c| c.dxil.len() >= COMPUTE_BLOB_MIN_DXIL && seen.insert((driver_cache::name(&c.dxil, 0), c.samplers.clone())));
        programs.sort_by_key(|c| std::cmp::Reverse(c.dxil.len()));
        programs_found = programs.len();
        programs.retain(|c| driver_cache::compute_path(&c.dxil, &c.samplers, 0, 0).is_some_and(|p| driver_cache::load(&p).is_none()));
        kept = programs_found - programs.len();
        let total = programs.len();
        phase_total.store(total, Ordering::Relaxed);
        let giants = programs.partition_point(|c| c.dxil.len() >= GIANT_DXIL);
        let mut queue: Vec<&crate::gpu::recomp::pso::Compiled> = if full {
            programs.iter().collect()
        } else {
            left.fetch_add(giants, Ordering::Relaxed);
            programs[giants..].iter().collect()
        };
        let mut blobs_not_started = 0;
        for round in 0..=RETRY_ROUNDS {
            let (next, later) = (AtomicUsize::new(0), Mutex::new(Vec::new()));
            let build = |c: &'_ crate::gpu::recomp::pso::Compiled| {
                let hash = crate::gpu::shader::fnv64(&c.dxil);
                let slot = match DRIVER_COMPILES.acquire(c.dxil.len(), threads.min(SMALL_AT_ONCE), Some(WARM_UP_COMMIT), true, || false) {
                    Ok(slot) => slot,
                    Err(Refused::Later | Refused::Cancelled) => return false,
                    Err(_) => {
                        eprintln!("  too big to build within the memory limits: DXIL {hash:016x} ({} B)", c.dxil.len());
                        big.fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                };
                let t0 = std::time::Instant::now();
                let ok = std::panic::catch_unwind(|| unsafe { warm_one(dev as *mut c_void, c) }).unwrap_or_else(|p| {
                    eprintln!("  a compute blob build panicked (DXIL {hash:016x}, {} B): {}", c.dxil.len(), panic_text(&*p));
                    false
                });
                drop(slot);
                (if ok { built } else { failed }).fetch_add(1, Ordering::Relaxed);
                progress(built.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed) + left.load(Ordering::Relaxed) + big.load(Ordering::Relaxed), total, c.dxil.len(), t0.elapsed(), hash);
                true
            };
            std::thread::scope(|w| {
                for _ in 0..threads {
                    let (build, next, later, queue) = (&build, &next, &later, &queue);
                    w.spawn(move || {
                        while !warm_up_cancelled() {
                            let Some(&c) = queue.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                            if !build(c) {
                                later.lock().unwrap().push(c);
                            }
                        }
                    });
                }
            });
            blobs_not_started += queue.len().saturating_sub(next.into_inner());
            queue = later.into_inner().unwrap();
            if warm_up_cancelled() {
                blobs_not_started += std::mem::take(&mut queue).len();
            }
            if queue.is_empty() {
                break;
            }
            if round < RETRY_ROUNDS {
                eprintln!("  compute blobs: {} had no room for their driver compile, tried again (round {} of {RETRY_ROUNDS})", queue.len(), round + 1);
            }
        }
        for c in &queue {
            eprintln!("  compute blobs: DXIL {:016x} ({} B) still had no room after {RETRY_ROUNDS} retries: too big for now", crate::gpu::shader::fnv64(&c.dxil), c.dxil.len());
        }
        big.fetch_add(queue.len(), Ordering::Relaxed);
        phases.push(finish("compute blobs", total, blobs_not_started, t0));
        phases.push(run("the rest", &rest, false, budget));
    });
    let cancelled = warm_up_cancelled() && phases.iter().any(|p| p.not_started != 0);
    let complete = record.is_some() && phases.iter().all(|p| p.not_started == 0);
    if complete {
        store.progress_complete();
    }
    let peak_commit = unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        let mut m: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        m.cb = std::mem::size_of_val(&m) as u32;
        if K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut m, m.cb) != 0 { m.PeakPagefileUsage as u64 } else { 0 }
    };
    Ok(WarmUp {
        full,
        pipelines: kept_pipelines,
        dropped,
        this_era,
        stale_left,
        runs: creations.runs,
        resumed,
        complete,
        cancelled,
        phases,
        programs: programs_found,
        kept,
        pruned,
        threads,
        took: started.elapsed(),
        peak_commit,
    })
}

/// The message of a caught panic.
fn panic_text(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>().copied().or_else(|| p.downcast_ref::<String>().map(String::as_str)).unwrap_or("(no message)")
}

/// Build kept pipeline `r` as the title's own is built (`go_background`'s builds, on this thread)
/// and release it: its translation and the driver's result stay in the disk caches.
/// Graphics and stream pipelines the warm-up is building now: their driver compiles take no
/// governor slot (`DRIVER_COMPILES` counts compute programs), so the progress line counts them here.
static WARM_GRAPHICS: AtomicUsize = AtomicUsize::new(0);

unsafe fn warm_pipeline(dev: *mut c_void, r: &super::pso_manifest::Record) -> bool {
    use super::pso_manifest::Kind;
    struct InFlight(bool);
    impl Drop for InFlight {
        fn drop(&mut self) {
            if self.0 {
                WARM_GRAPHICS.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
    let graphics = !matches!(r.kind, Kind::Compute);
    if graphics {
        WARM_GRAPHICS.fetch_add(1, Ordering::Relaxed);
    }
    let _in_flight = InFlight(graphics);
    let Some(owned) = OwnedDesc::from_record(r) else { return false };
    let (method, build): (&str, unsafe fn(Dev, &OwnedDesc, *mut *mut c_void) -> bool) = match r.kind {
        Kind::Compute => ("CreateComputePipelineState", build_compute),
        Kind::Graphics => ("CreateGraphicsPipelineState", build_graphics),
        Kind::MeshStream => ("CreatePipelineState", build_stream),
        Kind::VertexStream => ("CreatePipelineState", build_vertex_stream),
    };
    let mut out: *mut c_void = std::ptr::null_mut();
    let ok = build(Dev(dev as usize, orig_fn(dev, slot_of(dev, method))), &owned, &mut out) && !out.is_null();
    super::pso_async::release(out as usize);
    ok
}

/// Build `c` as `translated_compute_pso` does (its ABI root signature, no flags) and keep the
/// driver's blob.
unsafe fn warm_one(dev: *mut c_void, c: &crate::gpu::recomp::pso::Compiled) -> bool {
    let (Some(rs), Some(path), Some(device)) = (abi_rs_for(dev, &c.samplers), driver_cache::compute_path(&c.dxil, &c.samplers, 0, 0), ID3D12Device::from_raw_borrowed(&dev)) else { return false };
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
    desc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    desc.CS = bytecode(&c.dxil);
    let pso = match device.CreateComputePipelineState::<ID3D12PipelineState>(&desc) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  the driver refused compute program DXIL {:016x} ({} B): {:#x}", crate::gpu::shader::fnv64(&c.dxil), c.dxil.len(), e.code().0);
            return false;
        }
    };
    let Ok(blob) = pso.GetCachedBlob() else { return false };
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    driver_cache::write(&path, bytes)
}

/// The blobs of `COMPUTE_BLOBS`, and of translated graphics pipelines, kept on disk between runs.
///
/// `COMPUTE_BLOBS` saves the second build of a program in one run; the first build of every
/// run still goes through the driver's compiler. On an NVIDIA GPU that is 5 – 27 s for each of
/// a retail title's larger compute programs when its world loads: the title dispatches without
/// them meanwhile and is ended before all are built, so the next run starts the same compiles
/// again. A blob whose build took `KEEP` or longer is written under the hash of its DXIL and
/// its flags (`<DXIL cache>\driver\`) and read when a later run has none in memory; the
/// driver refuses the blob of another driver, adapter or root signature (the file is dropped
/// and the program compiled as before). A graphics pipeline's blob goes under the hash of all
/// the driver compiles it from (`graphics_name`): an NVIDIA driver takes 10 – 17 s for one big
/// translated pixel shader, and a draw waits for its pipeline. `XCL_PSO_DISK_CACHE=0`: off.
mod driver_cache {
    use super::{D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_INPUT_LAYOUT_DESC, D3D12_SHADER_BYTECODE, HRESULT};
    use crate::gpu::recomp::pso::StaticSampler;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// builds shorter than this are not worth a file
    pub(super) const KEEP: std::time::Duration = std::time::Duration::from_millis(250);
    static HITS: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    static STORED: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);

    /// The run's hits, misses, refused blobs and blobs kept, for the logs.
    pub fn summary() -> String {
        let n = |c: &AtomicU64| c.load(Ordering::Relaxed);
        format!("warm driver cache {} hits, {} misses, {} refused, {} failed their check, {} kept; {}", n(&HITS), n(&MISSES), n(&STALE), n(&CORRUPT), n(&STORED), super::super::pso_library::summary())
    }

    /// The pipeline library's folder (`pso_library`): this driver's and this translation's.
    pub(super) fn library_dir() -> PathBuf {
        dir().join("library").join(format!("{:016x}", crate::gpu::recomp::pso::translation_era()))
    }

    /// The file of the pipeline made from `dxil` with `flags`.
    pub fn name(dxil: &[u8], flags: i32) -> String {
        format!("{:016x}-{:x}-{:x}.pso", crate::gpu::shader::fnv64(dxil), dxil.len(), flags)
    }

    pub(super) fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("XCL_PSO_DISK_CACHE").map_or(true, |v| v.trim() != "0"))
    }

    pub fn path(dxil: &[u8], flags: i32) -> Option<PathBuf> {
        on().then(|| dir().join(name(dxil, flags)))
    }

    pub(super) fn compute_path(dxil: &[u8], samplers: &[StaticSampler], node_mask: u32, flags: i32) -> Option<PathBuf> {
        if node_mask == 0 && samplers.is_empty() {
            return path(dxil, flags);
        }
        if !on() {
            return None;
        }
        let mut h = Fnv(0xcbf2_9ce4_8422_2325);
        h.n(crate::gpu::recomp::pso::mem_windows());
        h.samplers(samplers);
        Some(dir().join(format!("cs{node_mask:x}-{:016x}-{}", h.0, name(dxil, flags))))
    }

    pub fn mesh_compute_path(dxil: &[u8], samplers: &[StaticSampler]) -> Option<PathBuf> {
        if !on() {
            return None;
        }
        let mut h = Fnv(0xcbf2_9ce4_8422_2325);
        h.samplers(samplers);
        Some(dir().join(format!("m{:016x}-{}", h.0, name(dxil, 0))))
    }

    pub(super) fn mesh_coverage_dir(device: *mut std::ffi::c_void) -> Option<PathBuf> {
        use crate::gpu::{agility, recomp::pso};
        let era = pso::translation_era();
        if era == 0 {
            return None;
        }
        super::DRIVER.get()?;
        let native = unsafe { windows::core::Interface::from_raw_borrowed(&device) }.and_then(pso::mesh_native_split);
        let sdk = match agility::state() {
            agility::State::Loaded { version, .. } => *version,
            _ => 0,
        };
        let mut opts = Fnv(0xcbf2_9ce4_8422_2325);
        for name in [
            "XCL_SHADER_BACKEND", "XCL_RECOMP_NO_STRUCTURED", "XCL_RECOMP_WHY", "XCL_X64_ALL_MASKS",
            "XCL_WAVE64_SPIN", "XCL_HLSL_COMMENTS", "XCL_DS_VALIDATE", "XCL_SAFE_BARRIER",
            "XCL_DEBUG_MAGENTA", "XCL_DEBUG_LOD0", "XCL_DEBUG_PS_SUB", "XCL_DEBUG_PS_EXPR",
        ] {
            if let Some(value) = std::env::var_os(name) {
                opts.bytes(name.as_bytes());
                opts.bytes(value.as_encoded_bytes());
            }
            opts.bytes(&[0]);
        }
        let (windows, cs, ps, opts) = (pso::mem_windows(), super::wave32_only(device) as u8, super::graphics_wave32(device) as u8, opts.0);
        Some(dir().join(format!("mesh-v1-{}-{era:016x}-{opts:016x}-w{windows}-c{cs}-p{ps}-n{native:?}-a{sdk}", env!("CARGO_PKG_VERSION"))))
    }

    fn blob_stamp(path: &Path) -> Option<(u64, u128)> {
        let m = std::fs::metadata(path).ok()?;
        if !m.is_file() || m.len() == 0 {
            return None;
        }
        let modified = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
        Some((m.len(), modified))
    }

    pub(super) fn mesh_covered(dir: &Path, key: u64) -> bool {
        let Some((note, cache)) = load(&dir.join(format!("{key:016x}.key"))).zip(dir.parent()) else { return false };
        let Ok(note) = std::str::from_utf8(&note) else { return false };
        let matches = |line: &str, prefix: char| {
            let Some((name, rest)) = line.split_once(' ') else { return false };
            let Some((len, modified)) = rest.split_once(' ') else { return false };
            if !name.starts_with(prefix) || !name.ends_with(".pso") || Path::new(name).file_name().is_none_or(|f| f != name) {
                return false;
            }
            blob_stamp(&cache.join(name)).is_some_and(|(n, m)| len.parse() == Ok(n) && modified.parse() == Ok(m))
        };
        let mut lines = note.lines();
        lines.next().is_some_and(|line| matches(line, 's'))
            && lines.next().is_none_or(|line| matches(line, 'm'))
            && lines.next().is_none()
    }

    pub(super) fn note_mesh_coverage(dir: &Path, key: u64, blobs: &[PathBuf]) {
        let mut note = String::new();
        for b in blobs {
            let Some((name, (len, modified))) = b.file_name().and_then(|n| n.to_str()).zip(blob_stamp(b)) else { return };
            use std::fmt::Write;
            let _ = writeln!(&mut note, "{name} {len} {modified}");
        }
        let _ = write(&dir.join(format!("{key:016x}.key")), note.as_bytes());
    }

    fn dir() -> PathBuf {
        crate::gpu::recomp::dxc::cache_dir().join("driver").join(super::DRIVER.get().map_or("unknown", String::as_str))
    }

    /// FNV-1a 64, fed field by field (the padding of the title's structures is not read).
    struct Fnv(u64);

    impl Fnv {
        fn bytes(&mut self, b: &[u8]) {
            for &x in b {
                self.0 = (self.0 ^ x as u64).wrapping_mul(0x100_0000_01b3);
            }
        }

        fn n(&mut self, v: impl Into<i64>) {
            self.bytes(&v.into().to_le_bytes());
        }

        unsafe fn shader(&mut self, b: &D3D12_SHADER_BYTECODE) {
            let bytes = super::shader_bytes(b).unwrap_or_default();
            self.n(bytes.len() as i64);
            self.n(crate::gpu::shader::fnv64(bytes) as i64);
        }

        fn samplers(&mut self, samplers: &[StaticSampler]) {
            self.n(samplers.len() as i64);
            for (w, cmp) in samplers {
                w.iter().for_each(|&v| self.n(v));
                self.n(*cmp as i32);
            }
        }

        unsafe fn input_layout(&mut self, il: &D3D12_INPUT_LAYOUT_DESC) {
            let elems = if il.NumElements > 0 && !il.pInputElementDescs.is_null() { std::slice::from_raw_parts(il.pInputElementDescs, il.NumElements as usize) } else { &[] };
            self.n(elems.len() as i64);
            for e in elems {
                if !e.SemanticName.is_null() {
                    self.bytes(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_bytes());
                }
                self.bytes(&[0]);
                for v in [e.SemanticIndex as i32, e.Format.0, e.InputSlot as i32, e.AlignedByteOffset as i32, e.InputSlotClass.0, e.InstanceDataStepRate as i32] {
                    self.n(v);
                }
            }
        }
    }

    /// The file of pipeline stream `stream` as the driver gets it – each shader by its bytes,
    /// the input layout by its elements, the other state as it is – with the static samplers
    /// `samplers` (its root signature). None: one with stream output or view instancing (state
    /// behind pointers not kept in the name), or a stream that does not parse.
    pub unsafe fn stream_path(stream: &[u8], samplers: &[StaticSampler]) -> Option<PathBuf> {
        if !on() {
            return None;
        }
        let mut h = Fnv(0xcbf2_9ce4_8422_2325);
        h.bytes(b"stream");
        h.samplers(samplers);
        for (ty, payload) in super::stream_subobjects(stream)? {
            h.n(ty as i64);
            match ty {
                0 | 19 => {}
                1..=6 | 24 | 25 => h.shader(&payload.as_ptr().cast::<D3D12_SHADER_BYTECODE>().read_unaligned()),
                12 => h.input_layout(&payload.as_ptr().cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned()),
                7 | 22 => return None,
                _ => h.bytes(payload),
            }
        }
        Some(dir().join(format!("s{:016x}.pso", h.0)))
    }

    /// The file of graphics pipeline `d` as the driver gets it – its shaders, fixed-function
    /// state, input layout, formats and flags – with the static samplers `samplers` (its root
    /// signature). None: one with stream output (not kept).
    pub unsafe fn graphics_name(d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, samplers: &[StaticSampler]) -> Option<String> {
        if d.StreamOutput.NumEntries != 0 {
            return None;
        }
        let mut h = Fnv(0xcbf2_9ce4_8422_2325);
        for s in [&d.VS, &d.PS, &d.DS, &d.HS, &d.GS] {
            h.shader(s);
        }
        h.samplers(samplers);
        let b = &d.BlendState;
        h.n(b.AlphaToCoverageEnable.0);
        h.n(b.IndependentBlendEnable.0);
        for t in &b.RenderTarget {
            for v in [t.BlendEnable.0, t.LogicOpEnable.0, t.SrcBlend.0, t.DestBlend.0, t.BlendOp.0, t.SrcBlendAlpha.0, t.DestBlendAlpha.0, t.BlendOpAlpha.0, t.LogicOp.0, t.RenderTargetWriteMask as i32] {
                h.n(v);
            }
        }
        h.n(d.SampleMask);
        let r = &d.RasterizerState;
        for v in [r.FillMode.0, r.CullMode.0, r.FrontCounterClockwise.0, r.DepthBias, r.DepthBiasClamp.to_bits() as i32, r.SlopeScaledDepthBias.to_bits() as i32, r.DepthClipEnable.0, r.MultisampleEnable.0, r.AntialiasedLineEnable.0, r.ForcedSampleCount as i32, r.ConservativeRaster.0] {
            h.n(v);
        }
        let z = &d.DepthStencilState;
        for v in [z.DepthEnable.0, z.DepthWriteMask.0, z.DepthFunc.0, z.StencilEnable.0, z.StencilReadMask as i32, z.StencilWriteMask as i32] {
            h.n(v);
        }
        for f in [&z.FrontFace, &z.BackFace] {
            for v in [f.StencilFailOp.0, f.StencilDepthFailOp.0, f.StencilPassOp.0, f.StencilFunc.0] {
                h.n(v);
            }
        }
        h.input_layout(&d.InputLayout);
        for v in [d.IBStripCutValue.0, d.PrimitiveTopologyType.0, d.NumRenderTargets as i32] {
            h.n(v);
        }
        d.RTVFormats.iter().for_each(|f| h.n(f.0));
        for v in [d.DSVFormat.0, d.SampleDesc.Count as i32, d.SampleDesc.Quality as i32, d.NodeMask as i32, d.Flags.0] {
            h.n(v);
        }
        Some(format!("g{:016x}.pso", h.0))
    }

    pub unsafe fn graphics_path(d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, samplers: &[StaticSampler]) -> Option<PathBuf> {
        if !on() {
            return None;
        }
        Some(dir().join(graphics_name(d, samplers)?))
    }

    /// Make a pipeline through the blob kept at `p`: `create(Some(blob))` when there is one
    /// (a blob the driver refuses – another driver, adapter or root signature, a damaged file –
    /// is dropped), else `create(None)`, the driver's own compile. The blob of a compile that
    /// took `KEEP` or longer is kept for the next run (`blob_of`, asked only then).
    /// Returns (the driver's result, made from the kept blob).
    pub fn build(p: Option<&Path>, create: impl FnMut(Option<&[u8]>) -> HRESULT, blob_of: impl FnOnce() -> Option<Vec<u8>>) -> (HRESULT, bool) {
        build_with_legacy(p, None, create, blob_of)
    }

    /// The lock of blob path `p` in a blocking warm-up: keys whose pipelines share a blob are
    /// built by one worker; the others wait and make theirs from its blob.
    fn blob_flight(p: &Path) -> std::sync::Arc<std::sync::Mutex<()>> {
        use std::sync::{Arc, Mutex, Weak};
        static FLIGHTS: Mutex<Vec<(PathBuf, Weak<Mutex<()>>)>> = Mutex::new(Vec::new());
        let mut flights = FLIGHTS.lock().unwrap_or_else(|e| e.into_inner());
        flights.retain(|(_, f)| f.strong_count() > 0);
        if let Some(f) = flights.iter().find(|(path, _)| path == p).and_then(|(_, f)| f.upgrade()) {
            return f;
        }
        let f = Arc::new(Mutex::new(()));
        flights.push((p.to_path_buf(), Arc::downgrade(&f)));
        f
    }

    pub(super) fn build_with_legacy(p: Option<&Path>, legacy: Option<&Path>, mut create: impl FnMut(Option<&[u8]>) -> HRESULT, blob_of: impl FnOnce() -> Option<Vec<u8>>) -> (HRESULT, bool) {
        let warming = super::BLOCKING_WARM.with(|state| state.get().is_some());
        let flight = p.filter(|_| warming).map(blob_flight);
        let _building = flight.as_ref().map(|f| f.lock().unwrap_or_else(|e| e.into_inner()));
        for path in p.into_iter().chain(legacy) {
            let Some(blob) = load(path) else { continue };
            if warming && super::blocking_warm_refusing() {
                return (super::E_INVALIDARG, false);
            }
            let hr = create(Some(&blob));
            if warming && hr < 0 && super::REFUSED.with(|r| r.get().is_some()) {
                return (hr, false);
            }
            note(hr >= 0, (Some(path) == p).then_some(path));
            if hr >= 0 {
                if warming {
                    if let Some(exact) = p.filter(|exact| *exact != path) {
                        let _ = write(exact, &blob);
                    }
                }
                super::note_blocking_driver(|counts| counts.hits += 1);
                return (hr, true);
            }
            super::note_blocking_driver(|counts| counts.rejected += 1);
        }
        if warming && super::blocking_warm_refusing() {
            return (super::E_INVALIDARG, false);
        }
        let started = std::time::Instant::now();
        let hr = create(None);
        let took = started.elapsed();
        let mut saved = false;
        if let (true, Some(p)) = (hr >= 0 && (warming || took >= KEEP), p) {
            if let Some(b) = blob_of() {
                if warming {
                    saved = write(p, &b);
                } else {
                    keep(p, &b, took);
                }
            }
        }
        if warming {
            super::note_blocking_driver(|counts| {
                counts.cold += 1;
                counts.saved += usize::from(saved);
            });
        }
        (hr, false)
    }

    /// What follows a blob in its file: its length, its hash, then this tag. A blob cut short or
    /// changed on disk is not handed to the driver (an NVIDIA driver aborted the process reading
    /// a bad module: "Invalid record").
    const TAG: &[u8; 8] = b"xclblob1";

    /// A blob is kept at `p` (`load` checks it as it reads it).
    pub fn kept(p: &Path) -> bool {
        p.metadata().is_ok_and(|m| m.len() > 0)
    }

    /// The blob kept at `p`, checked: one failing its length or hash is deleted (None, the
    /// program is compiled again). A file without the tag, kept before it, is taken as it is.
    pub fn load(p: &Path) -> Option<Vec<u8>> {
        let mut b = std::fs::read(p).ok().filter(|b| !b.is_empty())?;
        if !b.ends_with(TAG) {
            return Some(b);
        }
        let at = b.len().checked_sub(24)?;
        let word = |k: usize| u64::from_le_bytes(b[at + 8 * k..at + 8 * k + 8].try_into().unwrap());
        if word(0) != at as u64 || word(1) != crate::gpu::shader::fnv64(&b[..at]) {
            if !super::BLOCKING_WARM.with(|state| state.get().is_some()) {
                let _ = std::fs::remove_file(p);
            }
            if CORRUPT.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_warn!("D3D12.X: a driver blob kept on disk failed its length / hash check (cut short?) – its program is compiled again");
            }
            return None;
        }
        b.truncate(at);
        Some(b)
    }

    /// blobs on disk that failed their check (`load`)
    static CORRUPT: AtomicU64 = AtomicU64::new(0);

    /// Delete the blobs whose DXIL (hash, length) is not in `current`, the translations on disk:
    /// a translator change left them. (count, bytes) deleted; none when `current` is empty.
    pub fn prune(current: &std::collections::HashSet<(u64, usize)>) -> (usize, u64) {
        let Some(dir) = path(&[], 0).and_then(|p| p.parent().map(Path::to_path_buf)) else { return (0, 0) };
        if current.is_empty() {
            return (0, 0);
        }
        let mut pruned = (0, 0);
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let mut parts = name.strip_suffix(".pso").unwrap_or("").split('-');
            let (Some(Ok(hash)), Some(Ok(len))) = (parts.next().map(|h| u64::from_str_radix(h, 16)), parts.next().map(|l| usize::from_str_radix(l, 16))) else { continue };
            if !current.contains(&(hash, len)) {
                let bytes = e.metadata().map_or(0, |m| m.len());
                if std::fs::remove_file(e.path()).is_ok() {
                    pruned = (pruned.0 + 1, pruned.1 + bytes);
                }
            }
        }
        pruned
    }

    /// A pipeline was made from a kept blob (`ok`), or the driver refused the blob: dropped.
    pub fn note(ok: bool, p: Option<&Path>) {
        if ok {
            let n = HITS.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} pipelines made from the driver's blob of an earlier run (kept on disk)");
            }
        } else {
            if let Some(p) = p.filter(|_| !super::BLOCKING_WARM.with(|state| state.get().is_some())) {
                let _ = std::fs::remove_file(p);
            }
            if STALE.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_info!("D3D12.X: a driver blob kept on disk was refused (another driver or adapter?) – such pipelines are compiled again");
            }
        }
    }

    /// `bytes` to `p`, under another name first: a run that is cut short leaves no half file.
    pub fn write(p: &Path, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        static NEXT_TMP: AtomicU64 = AtomicU64::new(0);
        let tmp = p.with_extension(format!("tmp{}-{}", std::process::id(), NEXT_TMP.fetch_add(1, Ordering::Relaxed)));
        let mut file = Vec::with_capacity(bytes.len() + 24);
        file.extend_from_slice(bytes);
        file.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(&crate::gpu::shader::fnv64(bytes).to_le_bytes());
        file.extend_from_slice(TAG);
        let ok = std::fs::write(&tmp, &file).is_ok() && std::fs::rename(&tmp, p).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
        }
        ok
    }

    /// A compute pipeline went through the driver's compiler in the run: one `xcl warm-up`
    /// did not build (or built for another driver), when it took the driver `KEEP` or more.
    pub fn note_miss(dxil_len: usize, took: std::time::Duration) {
        if took < KEEP {
            return;
        }
        let n = MISSES.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 20 || n.is_power_of_two() {
            crate::xlog_warn!("D3D12.X: compute pipeline missed the warm driver cache ({dxil_len} B DXIL, {} ms in the driver; #{n})", took.as_millis());
        }
    }

    /// Keep `blob`, of a build that took `took`, for the next run.
    pub fn keep(p: &Path, blob: &[u8], took: std::time::Duration) {
        if took >= KEEP && write(p, blob) {
            let n = STORED.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} driver blobs kept on disk for the next run (this one: {} ms, {} KB)", took.as_millis(), blob.len() >> 10);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_stream_is_named_after_its_shader_bytes_not_where_they_are() {
            use super::super::{StreamOut, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE};
            let named = |code: &[u8], samplers: &[StaticSampler]| {
                let mut out = StreamOut::default();
                out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, code);
                out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
                unsafe { stream_path(&out.0, samplers) }.unwrap()
            };
            let (a, b, c) = (vec![1u8; 64], vec![1u8; 64], vec![2u8; 64]);
            assert_eq!(named(&a, &[]), named(&b, &[]));
            assert_ne!(named(&a, &[]), named(&c, &[]));
            assert!(named(&a, &[]).file_name().unwrap().to_string_lossy().starts_with('s'));
        }

        #[test]
        fn a_file_is_named_after_its_dxil_and_flags() {
            assert_eq!(name(b"abc", 0), name(b"abc", 0));
            assert_ne!(name(b"abc", 0), name(b"abd", 0));
            assert_ne!(name(b"abc", 0), name(b"abc", 1));
            // (the length is in the name: two programs of one hash and different lengths differ)
            assert!(name(&[0u8; 0x1234], 2).ends_with("-1234-2.pso"));
        }

        #[test]
        fn mesh_compute_blobs_are_keyed_by_dxil_and_samplers() {
            let key = mesh_compute_path(b"cs", &[]).unwrap();
            assert_eq!(mesh_compute_path(b"cs", &[]).as_ref(), Some(&key));
            assert_ne!(path(b"cs", 0).as_ref(), Some(&key));
            assert_ne!(mesh_compute_path(b"other cs", &[]).as_ref(), Some(&key));
            assert_ne!(mesh_compute_path(b"cs", &[([1, 2, 3, 4], false)]).as_ref(), Some(&key));
        }

        #[test]
        fn compute_blob_path_distinguishes_node_flags_and_samplers() {
            let exact = compute_path(b"cs", &[], 1, 0).unwrap();
            assert!(u64::from_str_radix(exact.file_name().unwrap().to_str().unwrap().split('-').next().unwrap(), 16).is_err());
            assert_eq!(compute_path(b"cs", &[], 0, 0), path(b"cs", 0));
            assert_ne!(Some(&exact), path(b"cs", 0).as_ref());
            assert_ne!(Some(&exact), compute_path(b"cs", &[], 1, 1).as_ref());
            assert_ne!(Some(&exact), compute_path(b"cs", &[([1, 2, 3, 4], false)], 1, 0).as_ref());
            assert_ne!(Some(&exact), compute_path(b"other", &[], 1, 0).as_ref());
        }

        #[test]
        fn blocking_warm_preserves_rejected_legacy_blob_and_reuses_the_exact_result() {
            use super::super::{WarmDriverCounts, BLOCKING_WARM, E_INVALIDARG, S_OK};
            let dir = std::env::temp_dir().join(format!("xcl-blocking-warm-{}", std::process::id()));
            let (exact, legacy) = (dir.join("exact.pso"), dir.join("legacy.pso"));
            assert!(write(&legacy, b"legacy"));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            BLOCKING_WARM.with(|state| state.set(Some((deadline, WarmDriverCounts::default()))));
            let mut attempts = Vec::new();
            let first = build_with_legacy(Some(&exact), Some(&legacy), |blob| {
                attempts.push(blob.is_some());
                if blob.is_some() { E_INVALIDARG } else { S_OK }
            }, || Some(b"exact".to_vec()));
            let next = build_with_legacy(Some(&exact), Some(&legacy), |blob| {
                assert_eq!(blob, Some(b"exact".as_slice()));
                S_OK
            }, || panic!("a cache hit cannot need a new blob"));
            let counts = BLOCKING_WARM.with(|state| state.take().unwrap().1);
            assert_eq!((first, next), ((S_OK, false), (S_OK, true)));
            assert_eq!(attempts, [true, false]);
            assert_eq!((counts.hits, counts.cold, counts.rejected, counts.saved), (1, 1, 1, 1));
            assert_eq!(load(&legacy).as_deref(), Some(b"legacy".as_slice()));
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn workers_sharing_a_blob_build_it_once() {
            use super::super::{WarmDriverCounts, BLOCKING_WARM, S_OK};
            use std::sync::atomic::{AtomicUsize, Ordering};
            let dir = std::env::temp_dir().join(format!("xcl-blocking-flight-{}", std::process::id()));
            let exact = dir.join("shared.pso");
            let cold = AtomicUsize::new(0);
            let results: Vec<_> = std::thread::scope(|s| {
                let workers: Vec<_> = (0..4)
                    .map(|_| {
                        s.spawn(|| {
                            BLOCKING_WARM.with(|state| state.set(Some((std::time::Instant::now() + std::time::Duration::from_secs(10), WarmDriverCounts::default()))));
                            build_with_legacy(Some(&exact), None, |blob| {
                                if blob.is_none() {
                                    cold.fetch_add(1, Ordering::Relaxed);
                                    std::thread::sleep(std::time::Duration::from_millis(100));
                                }
                                S_OK
                            }, || Some(b"shared".to_vec()))
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).collect()
            });
            assert_eq!(cold.load(Ordering::Relaxed), 1, "one worker compiles the shared blob");
            assert_eq!(results.iter().filter(|r| r.0 == S_OK && r.1).count(), 3, "the others make theirs from its blob");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn exact_rejection_then_accepted_legacy_needs_no_cold_create() {
            use super::super::{WarmDriverCounts, BLOCKING_WARM, E_INVALIDARG, S_OK};
            let dir = std::env::temp_dir().join(format!("xcl-blocking-legacy-{}", std::process::id()));
            let (exact, legacy) = (dir.join("exact.pso"), dir.join("legacy.pso"));
            assert!(write(&exact, b"exact") && write(&legacy, b"legacy"));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            BLOCKING_WARM.with(|state| state.set(Some((deadline, WarmDriverCounts::default()))));
            let result = build_with_legacy(Some(&exact), Some(&legacy), |blob| match blob {
                Some(b"exact") => E_INVALIDARG,
                Some(b"legacy") => S_OK,
                _ => panic!("did not try the distinct legacy blob"),
            }, || panic!("a cache hit cannot need a new blob"));
            let counts = BLOCKING_WARM.with(|state| state.take().unwrap().1);
            assert_eq!(result, (S_OK, true));
            assert_eq!((counts.hits, counts.cold, counts.rejected), (1, 0, 1));
            assert_eq!(load(&exact), load(&legacy));
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn warm_mesh_blob_hits_never_call_the_driver() {
            use super::super::{MeshBlobLocks, MeshWarmBuild, S_OK};
            let dir = std::env::temp_dir().join(format!("xcl-warm-mesh-{}", std::process::id()));
            let p = dir.join("mesh.pso");
            assert!(write(&p, b"driver blob"));
            let locks = MeshBlobLocks::default();
            let mut warm = MeshWarmBuild { locks: &locks, built: false, blobs: Vec::new() };
            let hr = unsafe { warm.blob(std::ptr::null_mut(), Some(&p), usize::MAX, "mesh", |_| panic!("called driver on a blob hit")) };
            assert_eq!(hr, Ok(S_OK));
            assert!(!warm.built);
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn mesh_coverage_requires_every_unchanged_blob() {
            let dir = std::env::temp_dir().join(format!("xcl-mesh-coverage-{}", std::process::id()));
            let coverage = dir.join("coverage");
            let (raster, compute) = (dir.join("sraster.pso"), dir.join("mcompute.pso"));
            assert!(write(&raster, b"raster") && write(&compute, b"compute"));
            note_mesh_coverage(&coverage, 1, &[raster.clone(), compute.clone()]);
            note_mesh_coverage(&coverage, 2, &[raster.clone()]);
            assert!(mesh_covered(&coverage, 1));
            assert!(mesh_covered(&coverage, 2));
            assert!(!mesh_covered(&coverage, 3));
            assert!(write(&compute, b"changed compute"));
            assert!(!mesh_covered(&coverage, 1));
            assert!(mesh_covered(&coverage, 2));
            note_mesh_coverage(&coverage, 1, &[raster.clone(), compute.clone()]);
            std::fs::remove_file(&compute).unwrap();
            assert!(!mesh_covered(&coverage, 1));
            std::fs::remove_file(&raster).unwrap();
            assert!(!mesh_covered(&coverage, 2));
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_kept_blob_is_read_back_and_a_refused_one_is_dropped() {
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-test-{}", std::process::id()));
            let p = dir.join("driver").join(name(b"program", 0));
            assert_eq!(load(&p), None, "nothing kept yet");
            assert!(!write(&p, &[]), "an empty blob is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_millis(10));
            assert_eq!(load(&p), None, "a build of milliseconds is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_secs(1));
            assert_eq!(load(&p).as_deref(), Some(&[1u8, 2, 3][..]));
            // no half file under the temporary name
            assert_eq!(std::fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
            // the driver took the blob: it stays; it refused it: gone, the program is compiled again
            note(true, Some(&p));
            assert!(p.exists());
            note(false, Some(&p));
            assert_eq!(load(&p), None);
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_blob_cut_short_or_changed_on_disk_is_not_used_and_one_kept_before_the_check_is() {
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-check-{}", std::process::id()));
            let p = dir.join(name(b"program", 0));
            let blob: Vec<u8> = (0..64u8).collect();
            assert!(write(&p, &blob));
            let kept = std::fs::read(&p).unwrap();
            assert_eq!(load(&p).as_deref(), Some(&blob[..]));
            std::fs::write(&p, &kept[..kept.len() - 1]).unwrap();
            assert_eq!(load(&p).as_deref(), Some(&kept[..kept.len() - 1]), "no tag: a file kept before the check");
            let mut changed = kept.clone();
            changed[3] ^= 1;
            std::fs::write(&p, &changed).unwrap();
            assert_eq!(load(&p), None, "a changed blob");
            assert!(!p.exists(), "deleted");
            std::fs::write(&p, &kept[blob.len() / 2..]).unwrap();
            assert_eq!(load(&p), None, "a blob cut short at its start");
            let _ = std::fs::remove_dir_all(&dir);
        }

        fn bytecode(b: &[u8]) -> D3D12_SHADER_BYTECODE {
            D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const std::ffi::c_void, BytecodeLength: b.len() }
        }

        /// A graphics pipeline's file follows everything the driver compiles it from, and only that.
        #[test]
        fn a_graphics_file_is_named_after_the_shaders_state_layout_and_samplers() {
            use windows::Win32::Graphics::Direct3D12::*;
            use windows::Win32::Graphics::Dxgi::Common::*;
            let (vs, ps, ps2) = (vec![1u8, 2, 3], vec![4u8, 5, 6], vec![4u8, 5, 7]);
            let (pos, nrm) = (c"POSITION", c"NORMAL");
            let elem = |name: &std::ffi::CStr| D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(name.as_ptr() as *const u8), Format: DXGI_FORMAT_R32G32B32_FLOAT, ..Default::default() };
            let layout = [elem(pos)];
            let desc = |ps: &[u8]| {
                let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
                d.VS = bytecode(&vs);
                d.PS = bytecode(ps);
                d.NumRenderTargets = 1;
                d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                d.SampleDesc.Count = 1;
                d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: layout.as_ptr(), NumElements: 1 };
                d
            };
            let samplers = [([1u32, 2, 3, 4], false)];
            unsafe {
                let name = |d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, s: &[StaticSampler]| graphics_name(d, s).unwrap();
                let base = name(&desc(&ps), &samplers);
                assert_eq!(base, name(&desc(&ps.clone()), &samplers), "the bytes count, not where they are");
                assert!(base.starts_with('g') && base.ends_with(".pso"));
                assert_ne!(base, name(&desc(&ps2), &samplers), "another PS");
                let mut d = desc(&ps);
                d.PS = D3D12_SHADER_BYTECODE::default();
                assert_ne!(base, name(&d, &samplers), "no PS");
                assert_ne!(base, name(&desc(&ps), &[]), "another root signature");
                assert_ne!(base, name(&desc(&ps), &[([1, 2, 3, 4], true)]), "another static sampler");
                let mut d = desc(&ps);
                d.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
                assert_ne!(base, name(&d, &samplers), "another target format");
                let mut d = desc(&ps);
                d.BlendState.RenderTarget[0].BlendEnable = true.into();
                assert_ne!(base, name(&d, &samplers), "another blend");
                let mut d = desc(&ps);
                d.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
                assert_ne!(base, name(&d, &samplers), "another cull mode");
                let mut d = desc(&ps);
                d.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;
                assert_ne!(base, name(&d, &samplers), "another stencil test");
                let other = [elem(nrm)];
                let mut d = desc(&ps);
                d.InputLayout.pInputElementDescs = other.as_ptr();
                assert_ne!(base, name(&d, &samplers), "another semantic");
                let mut d = desc(&ps);
                d.StreamOutput.NumEntries = 1;
                assert_eq!(graphics_name(&d, &samplers), None, "stream output is not kept");
            }
        }

        /// The second build of a slow pipeline comes from the blob its first build kept; a blob the
        /// driver refuses (a damaged file here) is dropped and the pipeline compiled again; a fast
        /// build keeps nothing and its blob is not asked for.
        #[test]
        fn a_slow_build_is_made_from_its_kept_blob_the_next_time() {
            use std::cell::Cell;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-build-test-{}", std::process::id()));
            let p = dir.join("driver").join("g0000000000000001.pso");
            let compiles = Cell::new(0);
            let blob = vec![9u8, 8, 7];
            let slow = |b: Option<&[u8]>| match b {
                Some(b) if b == blob.as_slice() => 0,
                Some(_) => super::super::E_INVALIDARG,
                None => {
                    compiles.set(compiles.get() + 1);
                    std::thread::sleep(KEEP + std::time::Duration::from_millis(20));
                    0
                }
            };
            assert_eq!(build(Some(&p), slow, || Some(blob.clone())), (0, false), "the first build compiles");
            assert_eq!(compiles.get(), 1);
            assert_eq!(load(&p).as_deref(), Some(blob.as_slice()), "and keeps its blob");
            assert_eq!(build(Some(&p), slow, || panic!("a build from the blob keeps nothing")), (0, true), "the second is made from it");
            assert_eq!(compiles.get(), 1, "without a compile");
            std::fs::write(&p, b"damaged").unwrap();
            assert_eq!(build(Some(&p), slow, || Some(blob.clone())), (0, false), "a refused blob: compiled");
            assert_eq!(compiles.get(), 2);
            assert_eq!(load(&p).as_deref(), Some(blob.as_slice()), "and the good blob kept in its place");
            // a build of milliseconds: nothing kept, its blob not asked for
            let q = dir.join("driver").join("g0000000000000002.pso");
            assert_eq!(build(Some(&q), |_| 0, || panic!("not asked for a fast build")), (0, false));
            assert_eq!(load(&q), None);
            // a failed build keeps nothing; no file: as before
            assert_eq!(build(Some(&q), |_| super::super::E_INVALIDARG, || panic!("not asked for a failed build")).0, super::super::E_INVALIDARG);
            assert_eq!(build(None, |b| if b.is_none() { 0 } else { -1 }, || None), (0, false));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// The driver's compiled form of translated compute programs (`GetCachedBlob`),
/// by (DXIL hash, root signature, flags), up to `COMPUTE_BLOBS_MAX` bytes in all.
type ComputeBlobKey = (u64, usize, i32, u32);
static COMPUTE_BLOBS: Mutex<Option<(HashMap<ComputeBlobKey, std::sync::Arc<Vec<u8>>>, usize)>> = Mutex::new(None);
const COMPUTE_BLOBS_MAX: usize = 512 << 20;
/// smaller programs compile in milliseconds: not kept
const COMPUTE_BLOB_MIN_DXIL: usize = 16 << 10;

fn compute_blob_key(dxil: &[u8], rs: usize, flags: i32, node_mask: u32) -> ComputeBlobKey {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dxil.hash(&mut h);
    (h.finish(), rs, flags, node_mask)
}

fn compute_blob(key: ComputeBlobKey) -> Option<std::sync::Arc<Vec<u8>>> {
    COMPUTE_BLOBS.lock().unwrap().as_ref().and_then(|(m, _)| m.get(&key).cloned())
}

pub(crate) fn resident_compute_blob_bytes() -> usize {
    COMPUTE_BLOBS.lock().unwrap().as_ref().map_or(0, |(_, total)| *total)
}

fn forget_compute_blob(key: ComputeBlobKey) {
    if let Some((m, total)) = COMPUTE_BLOBS.lock().unwrap().as_mut() {
        if let Some(b) = m.remove(&key) {
            *total -= b.len();
        }
    }
}

#[cfg(test)]
mod compute_blob_tests {
    #[test]
    fn a_full_resident_table_does_not_prevent_a_slow_blob_from_reaching_disk() {
        let dir = std::env::temp_dir().join(format!("xcl-compute-blob-full-{}", std::process::id()));
        let path = dir.join("compute.pso");
        let mut state = (super::HashMap::new(), super::COMPUTE_BLOBS_MAX);
        let blob = [1, 2, 3];
        super::save_compute_blob(&blob, Some(&path), super::driver_cache::KEEP, |bytes| {
            super::insert_compute_blob(&mut state, (1, 2, 0, 0), bytes);
        });
        assert!(state.0.is_empty());
        assert_eq!(super::driver_cache::load(&path).as_deref(), Some(blob.as_slice()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "native adapter diagnostic; run in isolation without the game"]
    fn native_single_node_compute_blob_acceptance() {
        use super::*;
        unsafe {
            let adapter = init().expect("D3D12 device");
            assert!(!adapter.starts_with("WARP"), "native hardware adapter required");
            eprintln!("compute blob node mask test: {adapter}");
            let raw = device_raw();
            let device = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            assert_eq!(device.GetNodeCount(), 1);
            let rs = abi_rs_for(raw, &[]).expect("root signature");
            let dxil = crate::gpu::recomp::dxc::compile("[numthreads(1,1,1)] void main() {}", "main", "cs_6_0", &[]).expect("DXC");
            for mask in [0, 1] {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
                desc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
                desc.CS = bytecode(&dxil);
                desc.NodeMask = mask;
                let pso = device.CreateComputePipelineState::<ID3D12PipelineState>(&desc).expect("native build");
                let blob = pso.GetCachedBlob().expect("native cache blob");
                assert_ne!(blob.GetBufferSize(), 0);
                desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: blob.GetBufferPointer(), CachedBlobSizeInBytes: blob.GetBufferSize() };
                assert!(device.CreateComputePipelineState::<ID3D12PipelineState>(&desc).is_ok(), "same-mask cached build");
                desc.NodeMask = 1 - mask;
                let result = device.CreateComputePipelineState::<ID3D12PipelineState>(&desc);
                eprintln!("compute blob node masks {mask} -> {}: {:#x}", desc.NodeMask, result.err().map_or(0, |e| e.code().0));
            }
        }
    }

    #[test]
    fn duplicate_results_do_not_consume_the_compute_blob_budget() {
        let mut cache = (super::HashMap::new(), 0);
        let key = (7, 1, 0, 0);
        super::insert_compute_blob(&mut cache, key, vec![1, 2]);
        super::insert_compute_blob(&mut cache, key, vec![3, 4, 5]);
        assert_eq!(cache.1, 2);
        assert_eq!(cache.0[&key].as_slice(), &[1, 2]);
    }

    #[test]
    fn a_blob_is_found_only_for_the_same_program_root_signature_and_flags_and_is_forgotten() {
        let dxil = vec![7u8; 1000];
        let key = super::compute_blob_key(&dxil, 0x1000, 0, 0);
        assert_eq!(key, super::compute_blob_key(&dxil.clone(), 0x1000, 0, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x2000, 0, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x1000, 1, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x1000, 0, 1));
        let mut other = dxil.clone();
        other[999] = 8;
        assert_ne!(key, super::compute_blob_key(&other, 0x1000, 0, 0));
        {
            let mut g = super::COMPUTE_BLOBS.lock().unwrap();
            let (m, total) = g.get_or_insert_with(|| (super::HashMap::new(), 0));
            m.insert(key, std::sync::Arc::new(vec![1, 2, 3]));
            *total += 3;
        }
        assert_eq!(super::compute_blob(key).as_deref().map(|v| v.len()), Some(3));
        let before = super::COMPUTE_BLOBS.lock().unwrap().as_ref().unwrap().1;
        super::forget_compute_blob(key);
        assert!(super::compute_blob(key).is_none());
        assert_eq!(super::COMPUTE_BLOBS.lock().unwrap().as_ref().unwrap().1, before - 3);
    }
}

/// The driver's blob of the adopted pipeline state `pso` (its PC `GetCachedBlob` through
/// `orig_fn`); None: none, or empty.
unsafe fn pso_blob(pso: *mut c_void) -> Option<Vec<u8>> {
    if pso.is_null() || !is_adopted(pso) {
        return None;
    }
    type GetBlob = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
    let get: GetBlob = std::mem::transmute(orig_fn(pso, slot_of(pso, "GetCachedBlob")));
    let mut raw = std::ptr::null_mut();
    if get(pso, &mut raw) < 0 || raw.is_null() {
        return None;
    }
    let blob = ID3DBlob::from_raw(raw);
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec();
    (!bytes.is_empty()).then_some(bytes)
}

unsafe fn keep_compute_blob(
    key: ComputeBlobKey,
    pso: *mut c_void,
    dxil_len: usize,
    disk: Option<&std::path::Path>,
    took: std::time::Duration,
) {
    let disk = disk.filter(|_| took >= driver_cache::KEEP);
    let memory = dxil_len >= COMPUTE_BLOB_MIN_DXIL && compute_blob(key).is_none();
    if (disk.is_none() && !memory) || pso.is_null() || !is_adopted(pso) {
        return;
    }
    type GetBlob = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
    let get: GetBlob = std::mem::transmute(orig_fn(pso, slot_of(pso, "GetCachedBlob")));
    let mut raw = std::ptr::null_mut();
    if get(pso, &mut raw) < 0 || raw.is_null() {
        return;
    }
    let blob = ID3DBlob::from_raw(raw);
    let len = blob.GetBufferSize();
    let ptr = blob.GetBufferPointer() as *const u8;
    if len == 0 || len > COMPUTE_BLOBS_MAX || ptr.is_null() {
        return;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    save_compute_blob(bytes, disk, took, |bytes| {
        if memory {
            let mut g = COMPUTE_BLOBS.lock().unwrap();
            insert_compute_blob(g.get_or_insert_with(|| (HashMap::new(), 0)), key, bytes);
        }
    });
}

fn save_compute_blob(bytes: &[u8], disk: Option<&std::path::Path>, took: std::time::Duration, resident: impl FnOnce(&[u8])) {
    if let Some(path) = disk {
        driver_cache::keep(path, bytes, took);
    }
    resident(bytes);
}

fn insert_compute_blob(
    (entries, total): &mut (HashMap<ComputeBlobKey, Arc<Vec<u8>>>, usize),
    key: ComputeBlobKey,
    bytes: impl AsRef<[u8]>,
) {
    let bytes = bytes.as_ref();
    if bytes.is_empty() || bytes.len() > COMPUTE_BLOBS_MAX.saturating_sub(*total) {
        return;
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = entries.entry(key) {
        *total += bytes.len();
        entry.insert(Arc::new(bytes.to_vec()));
    }
}

struct GraphicsShaders {
    vs: Vec<u8>,
    ps: Option<Vec<u8>>,
    samplers: Vec<crate::gpu::recomp::pso::StaticSampler>,
}

unsafe fn translate_graphics_stages(this: *mut c_void, vs: &[u8], ps: Option<&[u8]>) -> Option<GraphicsShaders> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    // a GPU that can run 32-lane waves may run a wave64 VS / PS in waves of 32 (the driver
    // picks; RDNA2 does for pixel shaders): the lanes 32..63 it reads come from `upper_lanes`
    // (split_wave64)
    let split = graphics_wave32(this);
    let _translating = crate::progress::TRANSLATIONS.begin();
    let _live = crate::live_stall::stage(crate::live_stall::Stage::ShaderCompile, || crate::gpu::shader::fnv64(vs));
    let vs_c = match console_to_dxil_for(vs, Stage::Vertex, 0, split) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("VS {:016x}: {e}", crate::gpu::shader::fnv64(vs)));
            return None;
        }
    };
    let mut samplers = vs_c.samplers;
    let vs_dxil = vs_c.dxil;
    let ps_dxil = match ps {
        Some(p) if is_console(p) => match console_to_dxil_for(p, Stage::Pixel, samplers.len() as u32, split) {
            Ok(b) => {
                samplers.extend(b.samplers);
                Some(b.dxil)
            }
            Err(e) => {
                note_translation_error(format!("PS {:016x}: {e}", crate::gpu::shader::fnv64(p)));
                return None;
            }
        },
        Some(p) => Some(p.to_vec()),
        None => None,
    };
    Some(GraphicsShaders { vs: vs_dxil, ps: ps_dxil, samplers })
}

fn note_translated_graphics(pso: *mut c_void, rs: *mut c_void) -> u64 {
    note_pso_rs(pso, rs);
    TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso as usize);
    TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed)
}

unsafe fn translated_graphics_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::is_console;
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let vs = shader_bytes(&d.VS)?;
    if !is_console(vs) {
        return None;
    }
    if shader_bytes(&d.HS).is_some() || shader_bytes(&d.DS).is_some() {
        note_translation_error("PSO with HS/DS – not supported yet".into());
        return None;
    }
    if let Some(gs) = shader_bytes(&d.GS).filter(|g| is_console(g)) {
        let _ = f;
        return Some(translated_gs_pso(this, d, vs, gs, riid, ppv));
    }
    // made when first used (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return None;
    }
    let translating = std::time::Instant::now();
    let GraphicsShaders { vs: vs_dxil, ps: ps_dxil, samplers } = translate_graphics_stages(this, vs, shader_bytes(&d.PS))?;
    let translated = translating.elapsed();
    let rs = abi_rs_for(this, &samplers)?;
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.VS = bytecode(&vs_dxil);
    copy.PS = ps_dxil.as_ref().map(|ps| bytecode(ps)).unwrap_or_default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    // a rect list pipeline: the fourth corner of each rect comes from a generated GS; without
    // it each rect is drawn as the triangle of its three vertices
    let rect_gs = if d.PrimitiveTopologyType.0 == RECT_LIST_TYPE && copy.GS.pShaderBytecode.is_null() {
        rect_list_gs(vs, shader_bytes(&d.PS)).map_err(|e| note_translation_error(format!("PS {:016x}: rect list drawn as triangles: {e}", shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64)))).ok()
    } else {
        None
    };
    if let Some(gs) = &rect_gs {
        copy.GS = bytecode(gs);
    }
    pc_topology_type(&mut copy);
    console_rt_blend(&mut copy.BlendState, &copy.RTVFormats, copy.NumRenderTargets);
    let ps_hash = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
    if debug_pso_switch("XCL_DEBUG_NO_STENCIL", ps_hash) {
        copy.DepthStencilState.StencilEnable = false.into();
    }
    if debug_pso_switch("XCL_DEBUG_NO_CULL", ps_hash) {
        copy.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    }
    let dxil_len = vs_dxil.len() + ps_dxil.as_ref().map_or(0, |ps| ps.len()) + copy.GS.BytecodeLength;
    // the driver's blob of an earlier run's build, when one is kept (`driver_cache`); a blob is
    // kept only of a build that kept its rect list GS (one made again without it is another
    // pipeline than its name says)
    let disk = driver_cache::graphics_path(&copy, &samplers);
    let asked = std::time::Instant::now();
    let slot = driver_slot(dxil_len, false, disk.as_deref().is_some_and(driver_cache::kept), || format!("VS {:016x} PS {ps_hash:016x}", crate::gpu::shader::fnv64(vs))).ok()?;
    let (driving, waited) = (std::time::Instant::now(), asked.elapsed());
    let gs = copy.GS.pShaderBytecode;
    let kept_gs = std::cell::Cell::new(true);
    let loaded = library_load(this, disk.as_deref(), super::pso_library::Desc::Graphics(&copy), riid, ppv, true);
    let (hr, from_blob) = if let Some(hr) = loaded { (hr, true) } else { driver_cache::build(
        disk.as_deref(),
        |blob| {
            copy.CachedPSO = blob.map_or_else(D3D12_CACHED_PIPELINE_STATE::default, |b| D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() });
            let hr = retry_without_rect_gs(&mut copy, rect_gs.is_some() && blob.is_none(), (crate::gpu::shader::fnv64(vs), ps_hash), |c| finish_pso(this, f, c as *const _ as *const c_void, riid, ppv));
            copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
            kept_gs.set(copy.GS.pShaderBytecode == gs);
            hr
        },
        || if kept_gs.get() && !ppv.is_null() { pso_blob(*ppv) } else { None },
    ) };
    if hr >= 0 && loaded.is_none() && kept_gs.get() && !ppv.is_null() {
        library_store(this, disk.as_deref(), Some(super::pso_library::Desc::Graphics(&copy)), *ppv);
    }
    drop(slot);
    super::pso_async::note_build(|| {
        format!(
            "VS {:016x} {} B, PS {ps_hash:016x} {} B of DXIL; translation {} ms, slot wait {} ms, driver {} ms{}",
            crate::gpu::shader::fnv64(vs),
            vs_dxil.len(),
            ps_dxil.as_ref().map_or(0, |ps| ps.len()),
            translated.as_millis(),
            waited.as_millis(),
            driving.elapsed().as_millis(),
            if from_blob { " (a driver blob kept on disk)" } else { "" }
        )
    });
    if hr < 0 {
        note_translation_error(format!(
            "PS {ps_hash:016x}: the driver rejected the PSO with the translated shaders ({hr:#x}; topology type {}, {} render targets, DSV format {}, HS {}, GS {})",
            copy.PrimitiveTopologyType.0,
            copy.NumRenderTargets,
            copy.DSVFormat.0,
            !copy.HS.pShaderBytecode.is_null(),
            !copy.GS.pShaderBytecode.is_null()
        ));
        return None;
    }
    if ppv.is_null() || (*ppv).is_null() {
        return Some(hr);
    }
    let n = note_translated_graphics(*ppv, rs);
    let z = &copy.DepthStencilState;
    super::device_x::note_lineage_depth_pso(*ppv as usize, z.DepthEnable.as_bool(), z.DepthWriteMask, z.DepthFunc);
    if super::depth_capture_views::selected_ps() == Some(ps_hash) {
        let equal = z.DepthEnable.as_bool() && z.DepthFunc == D3D12_COMPARISON_FUNC_EQUAL;
        if equal {
            let mut psos = CAPTURE_DEPTH_PSOS.lock().unwrap();
            if psos.len() < 64 { psos.push(*ppv as usize); }
        }
        crate::xlog_info!("depth capture: PS {ps_hash:016x} PSO {:#x}, enabled={} func={} write_mask={} EQUAL={equal} stencil_enabled={} stencil_read_mask={:#x} stencil_funcs={}/{} depth_bias={} slope_bias={}",
            *ppv as usize, z.DepthEnable.as_bool(), z.DepthFunc.0, z.DepthWriteMask.0, z.StencilEnable.as_bool(),
            z.StencilReadMask, z.FrontFace.StencilFunc.0, z.BackFace.StencilFunc.0, copy.RasterizerState.DepthBias, copy.RasterizerState.SlopeScaledDepthBias);
    }
    note_rect_list_gs(*ppv as usize, &copy, rect_gs.is_some());
    super::markers::note_graphics(*ppv as usize, crate::gpu::shader::fnv64(vs), shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64));
    note_draw_gate_shaders(*ppv as usize, &[Some(vs), shader_bytes(&d.PS), shader_bytes(&d.GS), shader_bytes(&d.HS), shader_bytes(&d.DS)]);
    let trace_frame = super::present::trace_frame();
    let ps = if trace_frame || super::gpu_timing::enabled() {
        let ps = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
        PSO_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (crate::gpu::shader::fnv64(vs), ps));
        ps
    } else {
        0
    };
    if trace_frame {
        // the input layout (semantic index, format, slot, offset) – what the vertices are read as
        let il = &d.InputLayout;
        let elems: Vec<String> = if il.NumElements > 0 && !il.pInputElementDescs.is_null() {
            std::slice::from_raw_parts(il.pInputElementDescs, il.NumElements as usize)
                .iter()
                .map(|e| format!("{}{} fmt {} slot {} @{}", e.SemanticName.to_string().unwrap_or_default(), e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset as i32))
                .collect()
        } else {
            Vec::new()
        };
        PSO_LAYOUTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, elems.join(", "));
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 400 {
            let b = &d.BlendState.RenderTarget[0];
            let z = &d.DepthStencilState;
            crate::xlog_info!(
                "frame trace: PSO vs {:016x} ps {ps:016x}: RTs {} {:?} DSV {} samples {}, depth (enable {}, write {}, func {}), stencil {}, blend0 (enable {}, src {} dst {} op {}, alpha src {} dst {}, mask {:#x}), cull {} fill {} depth clip {}, sample mask {:#x}",
                crate::gpu::shader::fnv64(vs),
                d.NumRenderTargets,
                &d.RTVFormats[..d.NumRenderTargets.min(8) as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                d.DSVFormat.0,
                d.SampleDesc.Count,
                z.DepthEnable.as_bool(),
                z.DepthWriteMask.0,
                z.DepthFunc.0,
                z.StencilEnable.as_bool(),
                b.BlendEnable.as_bool(),
                b.SrcBlend.0,
                b.DestBlend.0,
                b.BlendOp.0,
                b.SrcBlendAlpha.0,
                b.DestBlendAlpha.0,
                b.RenderTargetWriteMask,
                d.RasterizerState.CullMode.0,
                d.RasterizerState.FillMode.0,
                d.RasterizerState.DepthClipEnable.as_bool(),
                d.SampleMask
            );
            if z.StencilEnable.as_bool() {
                let f = |o: &D3D12_DEPTH_STENCILOP_DESC| (o.StencilFunc.0, o.StencilPassOp.0, o.StencilFailOp.0, o.StencilDepthFailOp.0);
                crate::xlog_info!(
                    "frame trace: … its stencil: read mask {:#x} write mask {:#x}, front (func, pass, fail, depth fail) {:?}, back {:?}",
                    z.StencilReadMask,
                    z.StencilWriteMask,
                    f(&z.FrontFace),
                    f(&z.BackFace)
                );
            }
        }
    }
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: graphics PSO from translated console shaders (#{})", n + 1);
    }
    Some(hr)
}

// ---------------------------------------------------------------------------------------
// non-blocking pipeline creation (see `pso_async`)
// ---------------------------------------------------------------------------------------

/// A pipeline description that owns what it points to (shader blobs, input layout and its
/// semantic names), so a worker can build the pipeline after the title's memory is gone.
/// The root signature pointer is kept as is (the translated paths replace it).
#[allow(dead_code)] // the vectors only keep the copies alive
enum OwnedDesc {
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>, Vec<Arc<Vec<u8>>>),
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>, Vec<Arc<Vec<u8>>>, Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>),
    Stream(Box<[usize; 2]>, Vec<u8>, Vec<Arc<Vec<u8>>>, Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>),
}

impl OwnedDesc {
    /// The title's root signature of the pipeline (0: none in it): the pipelines of one pass
    /// share theirs, and what a title binds next is likely of the pass it binds now
    /// (`pso_sched::Backlog`).
    fn root_signature(&self) -> usize {
        match self {
            OwnedDesc::Compute(d, _) => unsafe { std::mem::transmute_copy::<_, usize>(&d.pRootSignature) },
            OwnedDesc::Graphics(d, ..) => unsafe { std::mem::transmute_copy::<_, usize>(&d.pRootSignature) },
            OwnedDesc::Stream(_, buf, ..) => stream_fields(buf)
                .and_then(|fields| fields.into_iter().find(|f| f.0 == 0))
                .and_then(|(_, at, _)| buf.get(at..at + 8))
                .map_or(0, |p| usize::from_le_bytes(p.try_into().unwrap())),
        }
    }
}

unsafe impl Send for OwnedDesc {}
// read only once made: a fast-tier and a full build of one pipeline may read it at once
unsafe impl Sync for OwnedDesc {}

impl OwnedDesc {
    /// The bytes it owns, its shaders aside (shared copies, `diag_shader_copies`).
    fn held_bytes(&self) -> usize {
        use std::mem::size_of;
        let blobs = |b: &Vec<Arc<Vec<u8>>>| b.capacity() * size_of::<Arc<Vec<u8>>>();
        let layout = |e: &Vec<D3D12_INPUT_ELEMENT_DESC>, n: &Vec<std::ffi::CString>| {
            e.capacity() * size_of::<D3D12_INPUT_ELEMENT_DESC>()
                + n.capacity() * size_of::<std::ffi::CString>()
                + n.iter().map(|s| s.as_bytes_with_nul().len()).sum::<usize>()
        };
        size_of::<OwnedDesc>()
            + match self {
                OwnedDesc::Compute(_, b) => size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>() + blobs(b),
                OwnedDesc::Graphics(_, b, e, n) => size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>() + blobs(b) + layout(e, n),
                OwnedDesc::Stream(_, s, b, e, n) => size_of::<[usize; 2]>() + s.capacity() + blobs(b) + layout(e, n),
            }
    }
}

/// A background build's description while its job holds it, counted in `memdiag::KEPT_BUILDS`
/// with the bytes it owns and `more` (the job's text).
struct KeptBuild(OwnedDesc, u64);

impl KeptBuild {
    fn new(owned: OwnedDesc, more: usize) -> KeptBuild {
        let bytes = (owned.held_bytes() + more) as u64;
        crate::memdiag::KEPT_BUILDS.add(bytes);
        KeptBuild(owned, bytes)
    }
}

impl std::ops::Deref for KeptBuild {
    type Target = OwnedDesc;
    fn deref(&self) -> &OwnedDesc {
        &self.0
    }
}

impl Drop for KeptBuild {
    fn drop(&mut self) {
        crate::memdiag::KEPT_BUILDS.sub(self.1);
    }
}

/// A hash of a pipeline's shaders in their order (never 0): what `pso_async` knows a used
/// pipeline by in the next run.
fn shaders_key(blobs: &[Arc<Vec<u8>>]) -> u64 {
    super::pso_manifest::shaders_key(blobs.iter().map(|b| crate::gpu::shader::fnv64(b)))
}

#[cfg(test)]
mod shaders_key_tests {
    #[test]
    fn a_key_follows_the_shaders_and_their_order_and_is_never_zero() {
        use std::sync::Arc;
        let (a, b) = (Arc::new(vec![1u8, 2, 3]), Arc::new(vec![4u8, 5]));
        let key = super::shaders_key(&[a.clone(), b.clone()]);
        assert_eq!(key, super::shaders_key(&[a.clone(), b.clone()]));
        assert_ne!(key, super::shaders_key(&[b.clone(), a.clone()]));
        assert_ne!(key, super::shaders_key(&[a.clone()]));
        assert_ne!(key, super::shaders_key(&[a, Arc::new(vec![4u8, 6])]));
        assert_ne!(super::shaders_key(&[]), 0);
    }
}

impl OwnedDesc {
    fn shaders_key(&self) -> u64 {
        shaders_key(match self {
            OwnedDesc::Compute(_, b) => b,
            OwnedDesc::Graphics(_, b, ..) => b,
            OwnedDesc::Stream(_, _, b, ..) => b,
        })
    }

    fn ptr(&self) -> *const c_void {
        match self {
            OwnedDesc::Compute(d, _) => &**d as *const _ as *const c_void,
            OwnedDesc::Graphics(d, ..) => &**d as *const _ as *const c_void,
            OwnedDesc::Stream(d, ..) => &**d as *const _ as *const c_void,
        }
    }

    /// What `pso_manifest` keeps of it: the description with its pointers taken out, each console
    /// shader with where it is pointed at, the input layout's elements and names.
    fn record(&self, kind: super::pso_manifest::Kind) -> super::pso_manifest::Record {
        use std::mem::offset_of;
        fn bytes_of<T>(v: &T) -> Vec<u8> {
            unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }.to_vec()
        }
        fn zero(desc: &mut [u8], at: usize) {
            desc[at..at + 8].fill(0);
        }
        let layout = |at: usize, elems: &[D3D12_INPUT_ELEMENT_DESC], names: &[std::ffi::CString]| {
            (!elems.is_empty()).then(|| {
                let elems = elems.iter().zip(names).map(|(e, n)| {
                    let mut e = *e;
                    e.SemanticName = PCSTR::null();
                    (n.as_bytes().to_vec(), bytes_of(&e).try_into().unwrap())
                });
                (at as u32, elems.collect())
            })
        };
        let shader_at = |offset: usize| offset + offset_of!(D3D12_SHADER_BYTECODE, pShaderBytecode);
        let canonical = super::pso_canon::enabled();
        match self {
            OwnedDesc::Compute(d, blobs) => {
                let mut desc = bytes_of(&**d);
                if canonical {
                    super::pso_canon::compute(&mut desc);
                }
                let cs = shader_at(offset_of!(D3D12_COMPUTE_PIPELINE_STATE_DESC, CS));
                zero(&mut desc, offset_of!(D3D12_COMPUTE_PIPELINE_STATE_DESC, pRootSignature));
                zero(&mut desc, cs);
                super::pso_manifest::Record { kind, desc, shaders: blobs.iter().map(|b| (cs as u32, b.clone())).take(1).collect(), layout: None, canonical }
            }
            OwnedDesc::Graphics(d, blobs, elems, names) => {
                let mut desc = bytes_of(&**d);
                if canonical {
                    super::pso_canon::graphics(&mut desc);
                }
                zero(&mut desc, offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, pRootSignature));
                let stages = [
                    (offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, VS), &d.VS),
                    (offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, PS), &d.PS),
                    (offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, DS), &d.DS),
                    (offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, HS), &d.HS),
                    (offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, GS), &d.GS),
                ];
                let mut next = blobs.iter();
                let mut shaders = Vec::new();
                for (offset, stage) in stages {
                    if !stage.pShaderBytecode.is_null() {
                        if let Some(b) = next.next() {
                            shaders.push((shader_at(offset) as u32, b.clone()));
                        }
                    }
                    zero(&mut desc, shader_at(offset));
                }
                let elems_at = offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, InputLayout) + offset_of!(D3D12_INPUT_LAYOUT_DESC, pInputElementDescs);
                zero(&mut desc, elems_at);
                let so = offset_of!(D3D12_GRAPHICS_PIPELINE_STATE_DESC, StreamOutput);
                zero(&mut desc, so + offset_of!(D3D12_STREAM_OUTPUT_DESC, pSODeclaration));
                zero(&mut desc, so + offset_of!(D3D12_STREAM_OUTPUT_DESC, pBufferStrides));
                super::pso_manifest::Record { kind, desc, shaders, layout: layout(elems_at, elems, names), canonical }
            }
            OwnedDesc::Stream(_, buf, blobs, elems, names) => {
                let mut desc = buf.clone();
                if canonical {
                    super::pso_canon::stream(&mut desc);
                }
                let (mut shaders, mut elems_at) = (Vec::new(), None);
                for (ty, at, _) in stream_fields(buf).unwrap_or_default() {
                    // the stream's last payload may be 4 bytes at its very end (a sample mask)
                    let Some(p) = buf.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize) else { continue };
                    match ty {
                        1..=6 | 24 | 25 => {
                            if let Some(b) = blobs.iter().find(|b| b.as_ptr() as usize == p) {
                                shaders.push((at as u32, b.clone()));
                            }
                            zero(&mut desc, at);
                        }
                        12 => {
                            elems_at = Some(at);
                            zero(&mut desc, at);
                        }
                        0 => zero(&mut desc, at),
                        _ => {}
                    }
                }
                let layout = elems_at.and_then(|at| layout(at, elems, names));
                super::pso_manifest::Record { kind, desc, shaders, layout, canonical }
            }
        }
    }

    /// The description again from what `pso_manifest` kept (None: not one this XCL can read). Its
    /// root signature is none: the translated paths give it theirs.
    unsafe fn from_record(r: &super::pso_manifest::Record) -> Option<OwnedDesc> {
        use super::pso_manifest::Kind;
        fn put(desc: &mut [u8], at: u32, p: usize) -> Option<()> {
            desc.get_mut(at as usize..at as usize + 8)?.copy_from_slice(&(p as u64).to_le_bytes());
            Some(())
        }
        let mut desc = r.desc.clone();
        let blobs: Vec<Arc<Vec<u8>>> = r.shaders.iter().map(|(_, b)| b.clone()).collect();
        for (at, b) in &r.shaders {
            put(&mut desc, *at, b.as_ptr() as usize)?;
        }
        let (mut elems, mut names) = (Vec::new(), Vec::new());
        if let Some((_, kept)) = &r.layout {
            for (name, e) in kept {
                names.push(std::ffi::CString::new(name.clone()).ok()?);
                let mut e: D3D12_INPUT_ELEMENT_DESC = std::ptr::read_unaligned(e.as_ptr() as *const D3D12_INPUT_ELEMENT_DESC);
                e.SemanticName = PCSTR(names.last().unwrap().as_ptr() as *const u8);
                elems.push(e);
            }
            put(&mut desc, r.layout.as_ref().unwrap().0, elems.as_ptr() as usize)?;
        }
        fn read<T>(desc: &[u8]) -> Option<Box<T>> {
            (desc.len() == std::mem::size_of::<T>()).then(|| Box::new(unsafe { std::ptr::read_unaligned(desc.as_ptr() as *const T) }))
        }
        Some(match r.kind {
            Kind::Compute => OwnedDesc::Compute(read(&desc)?, blobs),
            Kind::Graphics => OwnedDesc::Graphics(read(&desc)?, blobs, elems, names),
            Kind::MeshStream | Kind::VertexStream => OwnedDesc::Stream(Box::new([desc.len(), desc.as_ptr() as usize]), desc, blobs, elems, names),
        })
    }
}

#[derive(Default)]
struct ShaderBlobs {
    blobs: HashMap<(u64, usize), Weak<Vec<u8>>>,
    inserts: u8,
}

static SHADER_BLOBS: OnceLock<Mutex<ShaderBlobs>> = OnceLock::new();

fn intern_shader_blob(bytes: &[u8]) -> Arc<Vec<u8>> {
    let key = (crate::gpu::shader::fnv64(bytes), bytes.len());
    let mut cache = SHADER_BLOBS.get_or_init(Mutex::default).lock().unwrap();
    if let Some(blob) = cache.blobs.get(&key).and_then(Weak::upgrade) {
        if blob.as_slice() == bytes {
            return blob;
        }
    }
    cache.inserts = cache.inserts.wrapping_add(1);
    if cache.inserts == 0 {
        cache.blobs.retain(|_, blob| blob.strong_count() != 0);
    }
    let blob = Arc::new(bytes.to_vec());
    cache.blobs.insert(key, Arc::downgrade(&blob));
    blob
}

unsafe fn own_shader(b: &mut D3D12_SHADER_BYTECODE, blobs: &mut Vec<Arc<Vec<u8>>>) {
    if let Some(bytes) = shader_bytes(b) {
        let blob = intern_shader_blob(bytes);
        *b = bytecode(blob.as_slice());
        blobs.push(blob);
    } else {
        *b = D3D12_SHADER_BYTECODE::default();
    }
}

unsafe fn own_compute(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_COMPUTE_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC));
    let mut blobs = Vec::new();
    own_shader(&mut d.CS, &mut blobs);
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Compute(d, blobs))
}

/// None with stream output (not copied; such pipelines are built in the call).
unsafe fn own_graphics(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    if d.StreamOutput.NumEntries > 0 {
        return None;
    }
    let mut blobs = Vec::new();
    for b in [&mut d.VS, &mut d.PS, &mut d.DS, &mut d.HS, &mut d.GS] {
        own_shader(b, &mut blobs);
    }
    let n = d.InputLayout.NumElements as usize;
    let mut elems = Vec::new();
    let mut names = Vec::new();
    if n > 0 && !d.InputLayout.pInputElementDescs.is_null() {
        elems = std::slice::from_raw_parts(d.InputLayout.pInputElementDescs, n).to_vec();
        for e in &mut elems {
            names.push(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_owned());
            e.SemanticName = PCSTR(names.last().unwrap().as_ptr() as *const u8);
        }
    }
    d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: if elems.is_empty() { std::ptr::null() } else { elems.as_ptr() }, NumElements: elems.len() as u32 };
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Graphics(d, blobs, elems, names))
}

/// A pipeline state stream with its shaders and input layout copied. None when another
/// subobject points at title memory (stream output, view instancing).
unsafe fn own_stream(desc: *const c_void) -> Option<OwnedDesc> {
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let mut blobs = Vec::new();
    let (mut elems, mut names) = (Vec::<D3D12_INPUT_ELEMENT_DESC>::new(), Vec::new());
    for (ty, at, psize) in stream_fields(&buf)? {
        let word = |buf: &[u8], k: usize| u64::from_le_bytes(buf[at + 8 * k..at + 8 * k + 8].try_into().unwrap());
        match ty {
            1..=6 | 24 | 25 => {
                let (p, n) = (word(&buf, 0) as usize, word(&buf, 1) as usize);
                if p != 0 && n != 0 && crate::gpu::shader::readable(p, n) {
                    let blob = intern_shader_blob(std::slice::from_raw_parts(p as *const u8, n));
                    buf[at..at + 8].copy_from_slice(&(blob.as_ptr() as u64).to_le_bytes());
                    blobs.push(blob);
                }
            }
            12 => {
                let (p, n) = (word(&buf, 0) as usize, u32::from_le_bytes(buf[at + 8..at + 12].try_into().unwrap()) as usize);
                if n != 0 {
                    if !crate::gpu::shader::readable(p, n * std::mem::size_of::<D3D12_INPUT_ELEMENT_DESC>()) {
                        return None;
                    }
                    elems = std::slice::from_raw_parts(p as *const D3D12_INPUT_ELEMENT_DESC, n).to_vec();
                    for e in &mut elems {
                        names.push(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_owned());
                        e.SemanticName = PCSTR(names.last().unwrap().as_ptr() as *const u8);
                    }
                    buf[at..at + 8].copy_from_slice(&(elems.as_ptr() as u64).to_le_bytes());
                }
            }
            7 | 22 if buf[at..at + psize].iter().any(|&b| b != 0) => return None,
            19 => buf[at..at + 16].fill(0),
            _ => {}
        }
    }
    let d = Box::new([buf.len(), buf.as_ptr() as usize]);
    Some(OwnedDesc::Stream(d, buf, blobs, elems, names))
}

/// The subobjects of a pipeline state stream: (type, offset of its payload, payload size). None:
/// a type this XCL does not know.
fn stream_fields(buf: &[u8]) -> Option<Vec<(u32, usize, usize)>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let at = off + if palign == 8 { 8 } else { 4 };
        if at + psize > buf.len() {
            break;
        }
        out.push((ty, at, psize));
        off += ((at - off) + psize + 7) & !7;
    }
    Some(out)
}

#[cfg(test)]
#[test]
fn owned_pipeline_shader_bytes_share_only_while_live() {
    let a = intern_shader_blob(b"shared shader");
    let b = intern_shader_blob(b"shared shader");
    let c = intern_shader_blob(b"unique shader");
    assert!(Arc::ptr_eq(&a, &b));
    assert!(!Arc::ptr_eq(&a, &c));
    let weak = Arc::downgrade(&a);
    drop(a);
    assert_eq!(b.as_slice(), b"shared shader");
    drop(b);
    assert!(weak.upgrade().is_none());
    assert_eq!(intern_shader_blob(b"shared shader").as_slice(), b"shared shader");
}

/// Raw pointers a worker needs (the device lives for the process).
#[derive(Clone, Copy)]
struct Dev(usize, usize);
unsafe impl Send for Dev {}

/// The stand-in for a pipeline with dummy shaders (draws / computes nothing).
unsafe fn standin_graphics(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let d = dummy();
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    copy.VS = bytecode(&d.vs);
    copy.PS = bytecode(&d.ps);
    copy.DS = D3D12_SHADER_BYTECODE::default();
    copy.HS = D3D12_SHADER_BYTECODE::default();
    copy.GS = D3D12_SHADER_BYTECODE::default();
    copy.StreamOutput = std::mem::zeroed();
    copy.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    pc_topology_type(&mut copy);
    console_rt_blend(&mut copy.BlendState, &copy.RTVFormats, copy.NumRenderTargets);
    note_standin(finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv), ppv)
}

unsafe fn standin_compute(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    note_standin(finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv), ppv)
}

unsafe fn standin_stream(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let Some((buf, _)) = fallback_stream(desc as *const u8) else {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 3 {
            crate::xlog_warn!("D3D12.X: a stand-in stream could not be made from the title's stream");
        }
        return E_INVALIDARG;
    };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    note_standin(finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv), ppv)
}

/// A background job that only prepared its shaders or had its driver compile put off for memory
/// (`pso_async::left_for_later`) is not noted as a failed build (`XCL_PUT_OFF_NOT_FAILED=0`: it
/// is, as "the background build failed: no error noted"). play-98 logged FSR4 PASS_12, PASS_17
/// and RCAS so, 0 to 16 ms into such a job, each built before and released unbound after 30 s.
fn later_not_failed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_PUT_OFF_NOT_FAILED").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: a put-off or prepared background build is noted as failed (XCL_PUT_OFF_NOT_FAILED=0)");
        }
        on
    })
}

/// Hand the title `standin`'s result now and build the real pipeline with `build` on a worker.
unsafe fn go_background(
    this: *mut c_void,
    f: PsoFn,
    desc: *const c_void,
    riid: *const GUID,
    ppv: *mut *mut c_void,
    kind: super::pso_manifest::Kind,
    owned: OwnedDesc,
    standin: unsafe fn(*mut c_void, PsoFn, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT,
    build: unsafe fn(Dev, &OwnedDesc, *mut *mut c_void) -> bool,
) -> Option<HRESULT> {
    let compute = kind == super::pso_manifest::Kind::Compute;
    let hr = standin(this, f, desc, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) < 3 {
            crate::xlog_warn!("D3D12.X: no stand-in pipeline ({hr:#x}) – building this one in the call");
        }
        return None;
    }
    let dev = Dev(this as usize, f as usize);
    let key = owned.shaders_key();
    let group = owned.root_signature() as u64;
    let kept = owned.record(kind);
    let image_only = compute && super::pso_async::image_skip_enabled()
        && kept.shaders.first().is_some_and(|(_, shader)| crate::gpu::compute_effects::big_compute_writes_only_textures(shader));
    let depth_only = match kind {
        super::pso_manifest::Kind::Compute => false,
        super::pso_manifest::Kind::Graphics => (*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC)).NumRenderTargets == 0,
        super::pso_manifest::Kind::MeshStream | super::pso_manifest::Kind::VertexStream => {
            let [size, base] = *desc.cast::<[usize; 2]>();
            stream_subobjects(std::slice::from_raw_parts(base as *const u8, size))
                .is_some_and(|subs| !subs.iter().any(|(ty, p)| *ty == 15 && p.get(32..36).is_some_and(|n| n != [0; 4])))
        }
    };
    let what = kept.shaders.iter().map(|(at, s)| format!("{} {:016x}", super::pso_manifest::stage_at(&kept, *at), crate::gpu::shader::fnv64(s))).collect::<Vec<_>>().join(" ");
    // (console shaders only: a PC shader's driver compile is not the translated program's)
    let shader_bytes = kept.shaders.iter().filter(|(_, s)| crate::gpu::recomp::pso::is_console(s)).map(|(_, s)| s.len()).sum::<usize>();
    let builder = super::raytracing::building_builder();
    let left_out = super::raytracing::left_out(compute, builder, kept.shaders.iter().map(|(_, s)| &s[..]));
    let rt_pass = left_out.is_none() && super::raytracing::miss_pass(compute, kept.shaders.iter().map(|(_, s)| &s[..]));
    let early = super::title_gta6::early_shader(kept.shaders.iter().map(|(_, s)| &s[..]));
    let record = super::pso_manifest::record(kept);
    let (standin, built_from) = (*ppv as usize, what.clone());
    let owned = KeptBuild::new(owned, built_from.capacity());
    let job: super::pso_async::Job = Box::new(move || {
        let mut out: *mut c_void = std::ptr::null_mut();
        take_last_note();
        let started = std::time::Instant::now();
        let built = build_named(standin, || unsafe { build(dev, &owned, &mut out) }) && !out.is_null();
        let later = !built && later_not_failed() && super::pso_async::left_for_later();
        if !built && !later {
            note_untranslated(standin, format!("{built_from}: the background build failed: {}", take_last_note().unwrap_or_else(|| "no error noted".into())));
        }
        if let Some(name) = traced_pso(standin) {
            let how = if built {
                "built".to_string()
            } else if later {
                let why = if super::pso_async::preparing() { "its shaders prepared, the driver build is still to come" } else { "its driver compile put off for memory, built when it fits or a bind asks" };
                format!("not built yet ({why})")
            } else {
                why_untranslated(standin).unwrap_or_default()
            };
            crate::xlog_info!("named pipeline '{name}' ({standin:#x}, {built_from}): {how} in {} ms", started.elapsed().as_millis());
        }
        if built {
            out as usize
        } else {
            0
        }
    });
    // (left out: a build that makes nothing, given up at once – no worker that takes it builds it)
    let job: super::pso_async::Job = if left_out.is_some() { Box::new(|| 0) } else { job };
    super::pso_async::submit_recorded(*ppv as usize, compute, job, (key, record), what, shader_bytes, group);
    if rt_pass {
        super::pso_async::rt_pass_last(*ppv as usize);
    }
    if let Some(why) = left_out {
        note_untranslated(*ppv as usize, why.to_string());
        super::pso_async::give_up(*ppv as usize);
    } else if builder {
        super::pso_async::mark_builder(*ppv as usize);
        super::pso_async::build_first(*ppv as usize);
    } else if let Some(shader) = early {
        super::title_gta6::build_early(*ppv as usize, shader);
    }
    if image_only {
        super::pso_async::allow_image_skip(standin);
    }
    if depth_only {
        super::pso_async::allow_depth_wait(standin);
    }
    Some(hr)
}

unsafe fn build_graphics(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_graphics_pso(dev.0 as *mut c_void, std::mem::transmute::<usize, PsoFn>(dev.1), d.ptr(), &iid, out), Some(hr) if hr >= 0)
}

unsafe fn build_compute(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_compute_pso(dev.0 as *mut c_void, std::mem::transmute::<usize, PsoFn>(dev.1), d.ptr(), &iid, out), Some(Ok(hr)) if hr >= 0)
}

unsafe fn build_vertex_stream(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let [size, base] = *d.ptr().cast::<[usize; 2]>();
    let Some(subs) = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size)) else { return false };
    let shader = |ty: u32| subs.iter().find(|e| e.0 == ty && e.1[8..16] != [0; 8]).and_then(|e| payload_shader(e.1));
    let Some(vs) = shader(1).filter(|vs| crate::gpu::recomp::pso::is_console(vs)) else { return false };
    let iid = pc_iid(Kind::PipelineState);
    translated_vertex_stream(dev.0 as *mut c_void, std::mem::transmute::<usize, PsoFn>(dev.1), &subs, vs, shader(2), &iid, out) >= 0
}

unsafe fn build_stream(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_mesh_stream(dev.0 as *mut c_void, d.ptr(), &iid, out, None), Some(hr) if hr >= 0)
}

pub(super) unsafe extern "system" fn dev_create_graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    graphics_pso(this, desc, riid, ppv, false)
}

/// `dev_create_graphics_pso`; `stream`: the desc is a pipeline stream's (`stream_pipeline`).
unsafe fn graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, stream: bool) -> HRESULT {
    super::null::capture_graphics_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateGraphicsPipelineState")));
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_graphics_pso(this, f, desc, riid, ppv)) } else { (translated_graphics_pso(this, f, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_graphics(desc).and_then(|o| go_background(this, f, desc, riid, ppv, super::pso_manifest::Kind::Graphics, o, standin_graphics, build_graphics)) {
            return hr;
        }
        if let Some(hr) = translated_graphics_pso(this, f, desc, riid, ppv) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return E_INVALIDARG;
    }
    let g = &*desc.cast::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>();
    let console = [&g.VS, &g.PS, &g.DS, &g.HS, &g.GS].into_iter().any(|stage| {
        shader_bytes(stage).is_some_and(crate::gpu::recomp::pso::is_console)
    });
    let hr = if console {
        E_INVALIDARG
    } else {
        let pc = untranslated_stream_desc(desc, stream);
        finish_pso(this, f, pc.as_ref().map_or(desc, |g| g as *const _ as *const c_void), riid, ppv)
    };
    if hr >= 0 {
        if !ppv.is_null() && !(*ppv).is_null() {
            let z = &g.DepthStencilState;
            super::device_x::note_lineage_depth_pso(*ppv as usize, z.DepthEnable.as_bool(), z.DepthWriteMask, z.DepthFunc);
        }
        return hr;
    }
    let hr2 = standin_graphics(this, f, desc, riid, ppv);
    note_fallback("graphics PSO", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe fn note_translated_compute(hr: HRESULT, ppv: *mut *mut c_void) {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
}

/// CreateComputePipelineState, console (slot 62) and PC alike: a pipeline whose CS lies in
/// XboxBvhBuilder.dll's image is a stage of the acceleration-structure builder (`raytracing`).
unsafe extern "system" fn dev_create_compute_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let builder = super::raytracing::builder_desc(desc);
    let hr = super::raytracing::creating_builder(builder, || create_compute_pso(this, desc, riid, ppv));
    if builder && hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        super::raytracing::note_builder_stage(*ppv as usize, *ppv as usize);
    }
    // the title's programs the stand-in for the indirect light follows are told by their name
    // here, once, and not at every dispatch
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && super::title_gta6_ambient::wanted() && crate::gpu::shader::readable(desc as usize, size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        if let Some(cs) = shader_bytes(&(*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC)).CS) {
            super::title_gta6_ambient::made(*ppv as usize, cs);
        }
    }
    hr
}

unsafe fn create_compute_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    super::null::capture_compute_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    take_last_note();
    let (mut console, missed) = if super::pso_async::enabled() { attempt(|| translated_compute_pso(this, f, desc, riid, ppv)) } else { (translated_compute_pso(this, f, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_compute(desc).and_then(|o| go_background(this, f, desc, riid, ppv, super::pso_manifest::Kind::Compute, o, standin_compute, build_compute)) {
            // a console program the worker is translating (its root signature comes with it);
            // the real PSO registers itself in TRANSLATED_COMPUTE once build_compute succeeds
            // (if build_compute fails, the stand-in stays unregistered so Dispatch /
            // ExecuteIndirect never bind XCL's ABI root signature onto it)
            LAST_COMPUTE_TRANSLATED.with(|c| c.set(true));
            LAST_COMPUTE_RS.with(|c| c.set(0));
            return hr;
        }
        console = translated_compute_pso(this, f, desc, riid, ppv);
    }
    LAST_COMPUTE_TRANSLATED.with(|c| c.set(matches!(console, Some(Ok(hr)) if hr >= 0)));
    if let Some(Ok(hr)) = console {
        return hr;
    }
    // a console shader the recompiler could not take must not reach the driver: AMD's shader
    // compiler crashes on the stripped DXIL (NOBC bitcode) instead of rejecting it
    let hr = if console.is_some() { E_INVALIDARG } else { finish_pso(this, f, desc, riid, ppv) };
    if hr >= 0 && !ppv.is_null() {
        note_untranslated(*ppv as usize, "a PC compute shader (no console code to translate)".into());
        if super::gpu_timing::enabled() && !(*ppv).is_null() && crate::gpu::shader::readable(desc as usize, size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
            if let Some(cs) = shader_bytes(&(*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC)).CS) {
                super::dispatch_timing::note(*ppv as usize, cs);
            }
        }
    }
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return hr;
    }
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let cs = shader_bytes(&copy.CS).map_or(0, crate::gpu::shader::fnv64);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    let hr2 = note_standin(finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv), ppv);
    if hr2 >= 0 && !ppv.is_null() {
        note_untranslated(*ppv as usize, format!("CS {cs:016x}: {}", take_last_note().unwrap_or_else(|| format!("the driver rejected it ({hr:#x}), no translation error noted"))));
    }
    note_fallback("compute PSO", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

// ---------------------------------------------------------------------------------------
// console geometry stages in two passes (MS / NGG, legacy GS): see `pso::TwoPass`
// ---------------------------------------------------------------------------------------

/// What XCL keeps for a mesh pipeline built from console MS / GS code: the title holds the
/// mesh pipeline state (pass-through MS + PS); the draw first runs `compute` (the console
/// program, one record per thread group into `res`), then the mesh pipeline over the records.
pub struct TwoPassPso {
    record_variant: Option<record_variant::Recipe>,
    pub prepass_fixed: Option<String>,
    /// compute pipeline state, owned here, and its root signature (static samplers)
    pub compute: usize,
    pub compute_rs: usize,
    /// root signature of the mesh pipeline (the title's PSO; its PS's static samplers)
    pub mesh_rs: usize,
    pub record: crate::gpu::recomp::MsInterface,
    /// DispatchMesh groups per record (see `pso::TwoPass`)
    pub mesh_groups_per_record: u32,
    /// "MS" or "GS"
    pub kind: &'static str,
}

impl TwoPassPso {
    pub(super) fn export_observer(&self) -> Option<[(usize,usize);2]> {
        self.record_variant.as_ref()?.pipeline(&self.record)
    }
}

impl Drop for TwoPassPso {
    fn drop(&mut self) {
        // the compute pipeline state is XCL's (the root signatures are cached or the title's)
        super::pso_async::release(self.compute);
    }
}

static TWO_PASS: Mutex<Option<HashMap<usize, std::sync::Arc<TwoPassPso>>>> = Mutex::new(None);

/// The compute pass of a pipeline state XCL built from console MS / GS code.
pub fn two_pass_of(pso: usize) -> Option<std::sync::Arc<TwoPassPso>> {
    let pso = super::pso_async::resolve(pso);
    TWO_PASS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn two_pass_count() -> usize {
    TWO_PASS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}

/// Mesh pipelines whose mesh stage is the console program itself, as a host mesh shader
/// (`pso::console_mesh_native`): translated graphics pipelines (`is_translated_graphics`) that a
/// DispatchMesh or a mesh ExecuteIndirect runs as one PC call.
static NATIVE_MESH: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

pub fn is_native_mesh(pso: usize) -> bool {
    let pso = super::pso_async::resolve(pso);
    pso != 0 && NATIVE_MESH.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

#[cfg(test)]
pub(super) fn register_native_mesh_for_test(pso: usize) {
    NATIVE_MESH.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
}

/// (native, two-pass) mesh pipelines made so far, for the caches line
pub fn mesh_pipeline_counts() -> (usize, usize) {
    (NATIVE_MESH.lock().unwrap().as_ref().map_or(0, |s| s.len()), two_pass_count())
}

/// A pipeline state stream being written: subobjects aligned as D3D12 expects
/// (type, payload at 4 or 8, size rounded to 8).
#[derive(Default)]
struct StreamOut(Vec<u8>);

impl StreamOut {
    fn raw(&mut self, ty: u32, payload: &[u8], align8: bool) {
        let at = self.0.len();
        self.0.extend_from_slice(&ty.to_le_bytes());
        self.0.resize(at + if align8 { 8 } else { 4 }, 0);
        let start = self.0.len();
        self.0.extend_from_slice(payload);
        if super::pso_canon::enabled() && super::null::subobject_payload(ty).is_some_and(|(size, _)| size == payload.len()) {
            super::pso_canon::payload(ty, &mut self.0[start..], None);
        }
        let end = (self.0.len() + 7) & !7;
        self.0.resize(end, 0);
    }
    fn put<T>(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, v: &T) {
        let b = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) };
        self.raw(ty.0 as u32, b, std::mem::align_of::<T>() == 8);
    }
    fn shader(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, code: &[u8]) {
        self.put(ty, &D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() });
    }
}

#[cfg(test)]
#[test]
fn streams_written_here_have_one_form() {
    let depth = |fill: u8| {
        let mut d = [fill; 56];
        d[..16].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
        d[52..].fill(0);
        d
    };
    let (mut a, mut b) = (StreamOut::default(), StreamOut::default());
    a.raw(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1.0 as u32, &depth(0x00), false);
    b.raw(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1.0 as u32, &depth(0xA7), false);
    assert_eq!(a.0, b.0);
}

/// The subobjects of a stream: (type, payload).
fn stream_subobjects(buf: &[u8]) -> Option<Vec<(u32, &[u8])>> {
    if buf.len() % 8 != 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        out.push((ty, buf.get(off + poff..off + poff + psize)?));
        off += (poff + psize + 7) & !7;
    }
    Some(out)
}

fn note_lineage_stream_depth(pso: usize, subs: &[(u32, &[u8])]) {
    if !super::depth_lineage::enabled() {
        return;
    }
    let mut depths = subs.iter().filter(|(ty, _)| matches!(*ty, 11 | 21 | 26));
    if let Some((_, bytes)) = depths.next() {
        if depths.next().is_none() && bytes.len() >= 12 {
            let word = |i: usize| i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
            super::device_x::note_lineage_depth_pso(pso, word(0) != 0, D3D12_DEPTH_WRITE_MASK(word(4)), D3D12_COMPARISON_FUNC(word(8)));
        }
    }
}

/// The stream's subobjects up to the first of a type XCL does not know (logged): a console
/// shader in front of it must still be found.
fn known_subobjects(buf: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::null::subobject_payload(ty) else {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: pipeline stream: unknown subobject type {ty} at {off:#x} of {:#x} - the rest is not read", buf.len());
            }
            break;
        };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(p) = buf.get(off + poff..off + poff + psize) else { break };
        out.push((ty, p));
        off += (poff + psize + 7) & !7;
    }
    out
}

unsafe fn payload_shader<'a>(p: &[u8]) -> Option<&'a [u8]> {
    let b = D3D12_SHADER_BYTECODE { pShaderBytecode: u64::from_le_bytes(p[0..8].try_into().ok()?) as usize as *const c_void, BytecodeLength: u64::from_le_bytes(p[8..16].try_into().ok()?) as usize };
    shader_bytes(&b)
}

fn console_rt_blend(blend: &mut D3D12_BLEND_DESC, formats: &[windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT; 8], count: u32) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let Some(formats) = formats.get(..count as usize) else { return false };
    let integer = |f: DXGI_FORMAT| {
        matches!(
            DXGI_FORMAT(super::device_x::pc_target_format(f.0 as u32) as i32),
            DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT
                | DXGI_FORMAT_R32G32B32_UINT | DXGI_FORMAT_R32G32B32_SINT
                | DXGI_FORMAT_R16G16B16A16_UINT | DXGI_FORMAT_R16G16B16A16_SINT
                | DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_SINT
                | DXGI_FORMAT_R10G10B10A2_UINT
                | DXGI_FORMAT_R8G8B8A8_UINT | DXGI_FORMAT_R8G8B8A8_SINT
                | DXGI_FORMAT_R16G16_UINT | DXGI_FORMAT_R16G16_SINT
                | DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT
                | DXGI_FORMAT_R8G8_UINT | DXGI_FORMAT_R8G8_SINT
                | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SINT
                | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_SINT
        )
    };
    if !blend.IndependentBlendEnable.as_bool() {
        let shared = blend.RenderTarget[0];
        if !shared.BlendEnable.as_bool() || !formats.iter().copied().any(integer) {
            return false;
        }
        if formats.iter().copied().all(integer) {
            blend.RenderTarget[0].BlendEnable = false.into();
            return true;
        }
        if shared.LogicOpEnable.as_bool() {
            return false;
        }
        blend.IndependentBlendEnable = true.into();
        blend.RenderTarget[..formats.len()].fill(shared);
    }
    let mut changed = false;
    for (target, &format) in blend.RenderTarget.iter_mut().zip(formats) {
        if target.BlendEnable.as_bool() && integer(format) {
            target.BlendEnable = false.into();
            changed = true;
        }
    }
    changed
}

fn console_stream_blend(subs: &[(u32, &[u8])]) -> Option<D3D12_BLEND_DESC> {
    let payload = |ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE| subs.iter().find(|e| e.0 == ty.0 as u32).map(|e| e.1);
    let formats = payload(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS)?;
    let blend = payload(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND)?;
    if formats.len() != std::mem::size_of::<D3D12_RT_FORMAT_ARRAY>() || blend.len() != std::mem::size_of::<D3D12_BLEND_DESC>() {
        return None;
    }
    let formats: D3D12_RT_FORMAT_ARRAY = unsafe { std::ptr::read_unaligned(formats.as_ptr().cast()) };
    let mut blend: D3D12_BLEND_DESC = unsafe { std::ptr::read_unaligned(blend.as_ptr().cast()) };
    console_rt_blend(&mut blend, &formats.RTFormats, formats.NumRenderTargets).then_some(blend)
}

fn vertex_stream_out(subs: &[(u32, &[u8])], rs: *mut c_void, vs: &[u8], ps: Option<&[u8]>, gs: Option<&[u8]>, pc_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE) -> StreamOut {
    const TOPOLOGY: u32 = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY.0 as u32;
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs);
    if let Some(ps) = ps {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
    }
    if let Some(gs) = gs {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs);
    }
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &pc_type);
    let blend = console_stream_blend(subs);
    for &(ty, payload) in subs {
        if !matches!(ty, 0..=6 | TOPOLOGY | 19 | 24 | 25) {
            if let Some(b) = blend.as_ref().filter(|_| ty == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND.0 as u32) {
                out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, b);
                continue;
            }
            let (_, align) = super::null::subobject_payload(ty).unwrap();
            out.raw(ty, payload, align == 8);
        }
    }
    out
}

unsafe fn translated_vertex_stream(this: *mut c_void, f: PsoFn, subs: &[(u32, &[u8])], vs: &[u8], ps: Option<&[u8]>, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    const TOPOLOGY: u32 = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY.0 as u32;
    let topology = subs.iter().find(|e| e.0 == TOPOLOGY).map(|e| i32::from_le_bytes(e.1[..4].try_into().unwrap()));
    // as `translated_graphics_pso` does for a desc: a rect list's fourth corners come from the
    // generated GS, and the topology type is one the PC has (`pc_topology_type`) – the console's
    // rect list type (19) or none made the driver refuse the stream, and the title then skipped
    // every draw with the pipeline (its full-screen passes and its UI)
    let rect_gs = if topology == Some(RECT_LIST_TYPE) {
        rect_list_gs(vs, ps).map_err(|e| note_translation_error(format!("PS {:016x}: rect list drawn as triangles: {e}", ps.map_or(0, crate::gpu::shader::fnv64)))).ok()
    } else {
        None
    };
    let pc_type = match topology {
        Some(t @ 1..=4) => D3D12_PRIMITIVE_TOPOLOGY_TYPE(t),
        _ => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    };
    let draw_gate_stages = [Some(vs), ps];
    let hashes = (crate::gpu::shader::fnv64(vs), ps.map_or(0, crate::gpu::shader::fnv64));
    let translating = std::time::Instant::now();
    let Some(GraphicsShaders { vs, ps, samplers }) = translate_graphics_stages(this, vs, ps) else { return E_INVALIDARG };
    let translated = translating.elapsed();
    let Some(rs) = abi_rs_for(this, &samplers) else { return E_INVALIDARG };
    let stream = |gs| vertex_stream_out(subs, rs, &vs, ps.as_deref(), gs, pc_type);
    let first = stream(rect_gs.as_deref());
    // the driver's blob of an earlier run's build, when one is kept (`driver_cache`); the
    // stream made again without the rect list GS is another pipeline: built without a blob
    let disk = driver_cache::stream_path(&first.0, &samplers);
    let len = vs.len() + ps.as_ref().map_or(0, Vec::len) + rect_gs.as_ref().map_or(0, Vec::len);
    let Ok(slot) = driver_slot(len, false, disk.as_deref().is_some_and(driver_cache::kept), || format!("VS {:016x} PS {:016x}", hashes.0, hashes.1)) else {
        return E_INVALIDARG;
    };
    let driving = std::time::Instant::now();
    let loaded = library_load(this, disk.as_deref(), super::pso_library::Desc::Stream(&first.0), riid, ppv, true);
    let (mut hr, from_blob) = loaded.map_or_else(|| driver_cache::build(disk.as_deref(), |blob| finish_stream(this, f, &first, blob, riid, ppv), || if ppv.is_null() { None } else { pso_blob(*ppv) }), |hr| (hr, true));
    let mut kept_gs = rect_gs.is_some();
    if hr < 0 && kept_gs {
        note_translation_error(format!("VS {:016x} PS {:016x}: the driver rejected the PSO with the rect list GS ({hr:#x}): rect list drawn as triangles", hashes.0, hashes.1));
        hr = finish_stream(this, f, &stream(None), None, riid, ppv);
        kept_gs = false;
    }
    if hr >= 0 && loaded.is_none() && kept_gs == rect_gs.is_some() && !ppv.is_null() {
        library_store(this, disk.as_deref(), Some(super::pso_library::Desc::Stream(&first.0)), *ppv);
    }
    drop(slot);
    super::pso_async::note_build(|| {
        format!(
            "stream VS {:016x} {} B, PS {:016x} {} B of DXIL; translation {} ms, driver {} ms{}",
            hashes.0,
            vs.len(),
            hashes.1,
            ps.as_ref().map_or(0, Vec::len),
            translated.as_millis(),
            driving.elapsed().as_millis(),
            if from_blob { " (a driver blob kept on disk)" } else { "" }
        )
    });
    if hr < 0 {
        note_translation_error(format!("the driver rejected the graphics stream with translated shaders ({hr:#x}; console topology type {topology:?})"));
    } else if !ppv.is_null() && !(*ppv).is_null() {
        if super::depth_lineage::enabled() {
            if let Some(subs) = stream_subobjects(&first.0) {
                note_lineage_stream_depth(*ppv as usize, &subs);
            }
        }
        if kept_gs {
            RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
        }
        if super::present::trace_frame() || super::gpu_timing::enabled() {
            PSO_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, hashes);
        }
        note_draw_gate_shaders(*ppv as usize, &draw_gate_stages);
        if super::depth_capture_views::selected_ps() == Some(hashes.1) {
            let depth = stream_subobjects(&first.0).and_then(|subs| {
                let mut depths = subs.into_iter().filter(|(ty, _)| matches!(*ty, 11 | 21 | 26));
                let (ty, bytes) = depths.next()?;
                if depths.next().is_some() || bytes.len() < 12 { return None }
                let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
                Some((ty, word(0), word(4), word(8)))
            });
            let equal = depth.is_some_and(|(_, on, _, func)| on != 0 && func == D3D12_COMPARISON_FUNC_EQUAL.0 as u32);
            let mut registered = false;
            if equal {
                let mut psos = CAPTURE_DEPTH_PSOS.lock().unwrap();
                if psos.len() < 64 {
                    psos.push(*ppv as usize);
                    registered = true;
                }
            }
            crate::xlog_info!("depth capture: VertexStream PS {:016x} native PSO {:#x}, submitted depth (type,enable,write,func) {depth:?} EQUAL={equal} registered={registered}", hashes.1, *ppv as usize);
        }
        super::markers::note_graphics(*ppv as usize, hashes.0, hashes.1);
        let n = note_translated_graphics(*ppv, rs);
        if n < 5 || n % 1000 == 0 {
            crate::xlog_info!("D3D12.X: graphics PSO stream from translated console shaders (#{})", n + 1);
        }
    }
    hr
}

/// Create the pipeline of `stream` (its RS and shaders in it), from the driver's blob `blob`
/// when there is one.
unsafe fn finish_stream(this: *mut c_void, f: PsoFn, stream: &StreamOut, blob: Option<&[u8]>, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut out = StreamOut(stream.0.clone());
    if let Some(b) = blob {
        out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() });
    }
    let desc = [out.0.len(), out.0.as_ptr() as usize];
    finish_pso(this, f, desc.as_ptr().cast(), riid, ppv)
}

/// One pipeline of a mesh shader built in the game (the pipeline library, then the blob cache):
/// `desc` what it is built from, into `out` (`adopted`: a pipeline `finish_pso` adopts).
#[allow(clippy::too_many_arguments)]
fn build_live_mesh_blob(this: *mut c_void, path: Option<&std::path::Path>, len: usize, what: impl FnOnce() -> String, desc: super::pso_library::Desc, riid: *const GUID, out: *mut *mut c_void, adopted: bool, create: impl FnMut(Option<&[u8]>) -> HRESULT, blob_of: impl FnOnce() -> Option<Vec<u8>>) -> Option<(HRESULT, bool)> {
    if super::pso_async::preparing() {
        return None;
    }
    let _slot = driver_slot(len, false, path.is_some_and(driver_cache::kept), what).ok()?;
    if let Some(hr) = unsafe { library_load(this, path, desc, riid, out, adopted) } {
        return Some((hr, true));
    }
    let built = driver_cache::build(path, create, blob_of);
    if built.0 >= 0 && !out.is_null() {
        unsafe { library_store(this, path, Some(desc), *out) };
    }
    Some(built)
}

/// The console PS of a two-pass pipeline (a PC PS passes as it is).
/// The PS of a two-pass pipeline (DXIL, static samplers – a PC PS passes as it is).
fn two_pass_ps(ps: Option<&[u8]>, split_wave64: bool) -> Result<(Option<Vec<u8>>, Vec<crate::gpu::recomp::pso::StaticSampler>), String> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    match ps {
        Some(p) if is_console(p) => {
            let _translating = crate::progress::TRANSLATIONS.begin();
            console_to_dxil_for(p, Stage::Pixel, 0, split_wave64).map(|c| (Some(c.dxil), c.samplers)).map_err(|e| format!("PS: {e}"))
        }
        Some(p) => Ok((Some(p.to_vec()), Vec::new())),
        None => Ok((None, Vec::new())),
    }
}

/// Create the mesh pipeline (`stream`, RS and shaders already in it) and the compute pass;
/// the title gets the mesh pipeline state.
#[allow(clippy::too_many_arguments)]
/// `mesh_samplers`: the static samplers of `mesh_rs`.
#[allow(clippy::too_many_arguments)]
unsafe fn finish_two_pass(this: *mut c_void, stream: &StreamOut, tp: crate::gpu::recomp::pso::TwoPass, mesh_rs: *mut c_void, mesh_samplers: &[crate::gpu::recomp::pso::StaticSampler], kind: &'static str, source: impl Fn() -> String, riid: *const GUID, ppv: *mut *mut c_void, warm: Option<&mut MeshWarmBuild<'_>>, inspect_records: bool, record_ps: Option<&[u8]>, record_ms: Option<&[u8]>) -> HRESULT {
    let Some(rs) = abi_rs_for(this, &tp.samplers) else { return E_INVALIDARG };
    let fs: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let fc: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    let disk = driver_cache::stream_path(&stream.0, mesh_samplers);
    let subs = stream_subobjects(&stream.0).unwrap_or_default();
    let raster_len = tp.ms.len() + subs.iter()
        .find(|e| e.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS.0 as u32)
        .and_then(|e| payload_shader(e.1)).map_or(0, <[u8]>::len);
    let raster_error = |hr: HRESULT| {
        let tags = subs.iter().map(|(kind, _)| *kind).collect::<Vec<_>>();
        note_translation_error(format!("the driver rejected the mesh pipeline from the console {kind} ({hr:#x}): {}; stream tags {tags:?}, raster shaders {raster_len} B", source()));
    };
    let compute_disk = (kind == "MS").then(|| driver_cache::mesh_compute_path(&tp.cs, &tp.samplers)).flatten();
    let iid = pc_iid(Kind::PipelineState);
    let create_compute = |blob: Option<&[u8]>, out| {
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        desc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
        desc.CS = bytecode(&tp.cs);
        if let Some(b) = blob {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr().cast(), CachedBlobSizeInBytes: b.len() };
        }
        if blocking_warm_refusing() {
            E_INVALIDARG
        } else if kind == "MS" {
            finish_pso(this, fc, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &iid, out)
        } else {
            fc(this, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &iid, out)
        }
    };
    if let Some(warm) = warm {
        match warm.blob(this, disk.as_deref(), raster_len, "MS raster pass", |out| finish_stream(this, fs, stream, None, riid, out)) {
            Ok(hr) if hr >= 0 => {}
            Ok(hr) => {
                raster_error(hr);
                return hr;
            }
            Err(()) => return E_INVALIDARG,
        }
        return match warm.blob(this, compute_disk.as_deref(), tp.cs.len(), "MS compute pass", |out| create_compute(None, out)) {
            Ok(hr) => {
                if hr < 0 {
                    note_translation_error(format!("the driver rejected the compute pass of the console MS ({hr:#x})"));
                }
                hr
            }
            Err(()) => E_INVALIDARG,
        };
    }
    let driving = std::time::Instant::now();
    let create = |blob: Option<&[u8]>| finish_stream(this, fs, stream, blob, riid, ppv);
    let blob_of = || if ppv.is_null() { None } else { pso_blob(*ppv) };
    let Some((hr, from_blob)) = build_live_mesh_blob(this, disk.as_deref(), raster_len, || format!("{kind} raster pass (DXIL {:016x})", crate::gpu::shader::fnv64(&tp.ms)), super::pso_library::Desc::Stream(&stream.0), riid, ppv, true, create, blob_of) else { return E_INVALIDARG };
    super::pso_async::note_build(|| {
        format!(
            "{kind} in two passes: mesh {} B, compute {} B of DXIL; mesh pipeline driver {} ms{}",
            tp.ms.len(),
            tp.cs.len(),
            driving.elapsed().as_millis(),
            if from_blob { " (a driver blob kept on disk)" } else { "" }
        )
    });
    if hr < 0 {
        raster_error(hr);
        return E_INVALIDARG;
    }
    let compute = std::cell::Cell::new(std::ptr::null_mut());
    let mut compute_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
    compute_desc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    compute_desc.CS = bytecode(&tp.cs);
    let hc = build_live_mesh_blob(this, compute_disk.as_deref(), tp.cs.len(), || format!("{kind} compute pass (DXIL {:016x})", crate::gpu::shader::fnv64(&tp.cs)), super::pso_library::Desc::Compute(&compute_desc), &iid, compute.as_ptr(), kind == "MS", |blob| create_compute(blob, compute.as_ptr()), || pso_blob(compute.get())).map(|(hr, _)| hr);
    let compute = compute.into_inner();
    if hc.is_none_or(|hc| hc < 0) || compute.is_null() {
        if let Some(hc) = hc {
            note_translation_error(format!("the driver rejected the compute pass of the console {kind} ({hc:#x})"));
        }
        super::pso_async::release(compute as usize);
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(*ppv as *const *const usize)).add(2));
        release(*ppv);
        *ppv = std::ptr::null_mut();
        return E_INVALIDARG;
    }
    note_pso_rs(*ppv, mesh_rs);
    if super::depth_lineage::enabled() {
        if let Some(subs) = stream_subobjects(&stream.0) {
            note_lineage_stream_depth(*ppv as usize, &subs);
        }
    }
    let prepass_fixed = super::material_gate_views::prepass::enabled().then(|| {
        let states = subs.iter().filter(|(tag, _)| matches!(tag, 9 | 10 | 11 | 16 | 17 | 21 | 26 | 27 | 28)).map(|(tag, data)| {
            let words = data.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<_>>();
            if matches!(tag,11|21|26) && words.len()>=4 {
                format!("depth_stencil type={tag} enable={} write_mask={} compare={} stencil_enable={} all_words={words:08x?}",words[0],words[1],words[2],words[3])
            } else if matches!(tag,10|27|28) && words.len()>=3 {
                format!("rasterizer type={tag} fill={} cull={} front_ccw={} all_words={words:08x?}",words[0],words[1],words[2])
            } else { format!("{tag}:{words:08x?}") }
        }).collect::<Vec<_>>();
        format!("native_stream[sample_mask,rasterizer,depth_stencil,dsv_format,sample_desc]={states:?}")
    });
    let record_variant = if inspect_records && mesh_samplers.is_empty() && tp.mesh_groups_per_record == 1 {
        match record_variant::Recipe::new(this, &subs, &tp, mesh_rs, rs, record_ps,record_ms) {
            Ok(recipe) => Some(recipe),
            Err(e) => { crate::xlog_warn!("mesh record probe: keeping ordinary MS PSO: {e}"); None }
        }
    } else { None };
    TWO_PASS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, std::sync::Arc::new(TwoPassPso { record_variant, prepass_fixed, compute: compute as usize, compute_rs: rs as usize, mesh_rs: mesh_rs as usize, record: tp.record, mesh_groups_per_record: tp.mesh_groups_per_record, kind }));
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: pipeline from the console {kind} in two passes (compute {} B + mesh {} B DXIL)", tp.cs.len(), tp.ms.len());
    }
    hr
}

fn mesh_shader_slots<'a>(ms_slot: Option<&'a [u8]>, as_slot: Option<&'a [u8]>) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
    use crate::gpu::recomp::pso::is_console;
    match (ms_slot, as_slot) {
        (None, Some(a)) if is_console(a) && crate::gpu::xbox::parse(a).is_some_and(|sh| sh.stage == "MS") => {
            (Some(a), None)
        }
        (ms, a) => (ms, a),
    }
}

#[cfg(test)]
mod mesh_slot_tests {
    use super::*;

    #[test]
    fn console_blend_bypasses_signed_and_unsigned_targets_only() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let mut blend = D3D12_BLEND_DESC::default();
        blend.IndependentBlendEnable = true.into();
        for (i, target) in blend.RenderTarget.iter_mut().enumerate() {
            target.BlendEnable = true.into();
            target.BlendOp = D3D12_BLEND_OP_MAX;
            target.RenderTargetWriteMask = (i + 1) as u8;
        }
        let formats = [
            DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R8_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN,
        ];
        assert!(console_rt_blend(&mut blend, &formats, 4));
        assert_eq!(blend.RenderTarget.map(|t| (t.BlendEnable.as_bool(), t.BlendOp, t.RenderTargetWriteMask)), [
            (true, D3D12_BLEND_OP_MAX, 1), (false, D3D12_BLEND_OP_MAX, 2), (false, D3D12_BLEND_OP_MAX, 3), (true, D3D12_BLEND_OP_MAX, 4),
            (true, D3D12_BLEND_OP_MAX, 5), (true, D3D12_BLEND_OP_MAX, 6), (true, D3D12_BLEND_OP_MAX, 7), (true, D3D12_BLEND_OP_MAX, 8),
        ]);
        assert!(!console_rt_blend(&mut blend, &formats, 4));
    }

    #[test]
    fn shared_console_blend_preserves_float_state_when_mixed_with_integer() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let mut blend = D3D12_BLEND_DESC::default();
        blend.AlphaToCoverageEnable = true.into();
        blend.RenderTarget[0].BlendEnable = true.into();
        blend.RenderTarget[0].BlendOp = D3D12_BLEND_OP_MAX;
        blend.RenderTarget[0].RenderTargetWriteMask = 7;
        blend.RenderTarget[4].RenderTargetWriteMask = 3;
        let mixed = [
            DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN,
        ];
        assert!(console_rt_blend(&mut blend, &mixed, 4));
        assert!(blend.IndependentBlendEnable.as_bool());
        assert!(blend.AlphaToCoverageEnable.as_bool());
        let expected = [false, true, false, true];
        for (i, target) in blend.RenderTarget[..4].iter().enumerate() {
            assert_eq!(target.BlendEnable.as_bool(), expected[i]);
            assert_eq!(target.BlendOp, D3D12_BLEND_OP_MAX);
            assert_eq!(target.RenderTargetWriteMask, 7);
        }
        assert_eq!(blend.RenderTarget[4].RenderTargetWriteMask, 3);
        let mut shared = D3D12_BLEND_DESC::default();
        shared.RenderTarget[0].BlendEnable = true.into();
        assert!(console_rt_blend(&mut shared, &[DXGI_FORMAT_R8_UINT; 8], 2));
        assert!(!shared.IndependentBlendEnable.as_bool());
        assert!(!shared.RenderTarget[0].BlendEnable.as_bool());
    }

    #[test]
    fn fallback_stream_uses_host_target_formats() {
        let mut stream = StreamOut::default();
        let rt = [115u32, 86, 67, 28, 0, 0, 0, 0, 4];
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &118u32);
        let desc = [stream.0.len(), stream.0.as_ptr() as usize];
        let (out, _) = unsafe { fallback_stream(desc.as_ptr().cast()) }.unwrap();
        let subs = stream_subobjects(&out).unwrap();
        let words = |bytes: &[u8]| {
            bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>()
        };
        assert_eq!(words(subs[0].1), [28, 28, 10, 28, 0, 0, 0, 0, 4]);
        assert_eq!(words(subs[1].1), [45]);
    }

    /// A title's stream may end without padding its last subobject to 8 bytes.
    #[test]
    fn fallback_stream_takes_a_stream_without_trailing_padding() {
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"console-vs");
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &D3D12_BLEND_DESC::default());
        let desc = [stream.0.len() - 4, stream.0.as_ptr() as usize];
        let (out, _) = unsafe { fallback_stream(desc.as_ptr().cast()) }.unwrap();
        let vs = known_subobjects(&out)[0];
        assert_eq!(unsafe { payload_shader(vs.1) }, Some(dummy().vs.as_slice()));
    }

    #[test]
    fn a_stand_in_stream_has_a_topology_type_the_pc_takes() {
        for (console, pc) in [(19u32, 3u32), (0, 3), (4, 3), (2, 2)] {
            let mut stream = StreamOut::default();
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"console-vs");
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &console);
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            let (out, _) = unsafe { fallback_stream(desc.as_ptr().cast()) }.unwrap();
            let topology = known_subobjects(&out)[1];
            assert_eq!(u32::from_le_bytes(topology.1[..4].try_into().unwrap()), pc, "console type {console}");
        }
    }

    #[test]
    fn an_owned_stream_keeps_its_own_input_layout() {
        let name = std::ffi::CString::new("TEXCOORD").unwrap();
        let elems = [D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(name.as_ptr().cast()), SemanticIndex: 3, ..Default::default() }];
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"console-vs");
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: 1 });
        let desc = [stream.0.len(), stream.0.as_ptr() as usize];
        let owned = unsafe { own_stream(desc.as_ptr().cast()) }.unwrap();
        drop((stream, name));
        let [size, base] = unsafe { *owned.ptr().cast::<[usize; 2]>() };
        let subs = known_subobjects(unsafe { std::slice::from_raw_parts(base as *const u8, size) });
        let layout = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT.0 as u32).unwrap();
        let copy = unsafe { &*(u64::from_le_bytes(layout.1[0..8].try_into().unwrap()) as *const D3D12_INPUT_ELEMENT_DESC) };
        assert_ne!(copy as *const D3D12_INPUT_ELEMENT_DESC, elems.as_ptr());
        assert_eq!((unsafe { std::ffi::CStr::from_ptr(copy.SemanticName.0.cast()) }.to_str().unwrap(), copy.SemanticIndex), ("TEXCOORD", 3));
    }
    use crate::gpu::xbox::tests::{sample, fssm, CODE};

    #[test]
    fn a_console_ms_in_the_as_slot_is_the_mesh_shader() {
        let ms = sample(13, &[fssm(&CODE, &[(8, 13)])]);
        let as_ = sample(14, &[fssm(&CODE, &[(8, 14)])]);
        let pc = b"DXBC-not-a-console-container".to_vec();
        let (m, a) = mesh_shader_slots(None, Some(&ms));
        assert!(m.is_some_and(|m| std::ptr::eq(m, ms.as_slice())) && a.is_none());
        let (m, a) = mesh_shader_slots(Some(&ms), Some(&as_));
        assert!(m.is_some_and(|m| std::ptr::eq(m, ms.as_slice())) && a.is_some_and(|a| std::ptr::eq(a, as_.as_slice())));
        let (m, a) = mesh_shader_slots(None, Some(&as_));
        assert!(m.is_none() && a.is_some_and(|a| std::ptr::eq(a, as_.as_slice())));
        let (m, a) = mesh_shader_slots(None, Some(&pc));
        assert!(m.is_none() && a.is_some());
        assert!(mesh_shader_slots(None, None) == (None, None));
    }

    #[test]
    fn standin_uses_one_mesh_slot_for_moved_mesh_bytecode() {
        let Some((dummy_ms, dummy_ps)) = dummy_mesh() else {
            crate::gpu::recomp::runner::skip_test("no stand-in mesh shaders");
            return;
        };
        let (as_type, ms_type, ps_type) = (
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
        );
        let ms = sample(13, &[fssm(&CODE, &[(8, 13)])]);
        for explicit_ms in [false, true] {
            let mut stream = StreamOut::default();
            stream.shader(as_type, &ms);
            stream.shader(ps_type, b"DXBC-test");
            if explicit_ms {
                stream.put(ms_type, &D3D12_SHADER_BYTECODE::default());
            }
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            let (out, _) = unsafe { fallback_stream(desc.as_ptr().cast()) }.unwrap();
            let subs = stream_subobjects(&out).unwrap();
            let mut mesh = subs.iter().filter(|s| s.0 == ms_type.0 as u32);
            let pixel = subs.iter().find(|s| s.0 == ps_type.0 as u32).unwrap();
            assert_eq!(unsafe { payload_shader(mesh.next().unwrap().1) }, Some(dummy_ms.as_slice()));
            assert!(mesh.next().is_none());
            assert_eq!(unsafe { payload_shader(pixel.1) }, Some(dummy_ps.as_slice()));
            assert!(subs.iter().filter(|s| s.0 == as_type.0 as u32).all(|s| s.1.iter().all(|&b| b == 0)));
        }
    }
}

/// CreatePipelineState with a console MS: Some(hr) when it was a console mesh pipeline
/// (translated, or E_INVALIDARG – console bytecode must not reach the driver).
unsafe fn translated_mesh_stream(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, mut warm: Option<&mut MeshWarmBuild<'_>>) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_mesh, is_console};
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return None;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let subs = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size))?;
    let as_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0 as u32).and_then(|s| payload_shader(s.1));
    let ms_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).and_then(|s| payload_shader(s.1));
    let (ms, as_slot) = mesh_shader_slots(ms_slot, as_slot);
    let Some(ms) = ms else { return None };
    if !is_console(ms) {
        return None;
    }
    if let Some(a) = as_slot {
        let types: Vec<u32> = subs.iter().map(|s| s.0).collect();
        note_translation_error(format!(
            "mesh pipeline with AS {:016x} (MS {:016x}) – not supported yet; subobjects {types:?}",
            crate::gpu::shader::fnv64(a),
            crate::gpu::shader::fnv64(ms)
        ));
        return Some(E_INVALIDARG);
    }
    // made when first bound (`pso_async::lazy`, `lazy_mesh`): not even translated now
    if (ms_slot.is_none() || super::pso_async::lazy_mesh()) && super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return Some(E_INVALIDARG);
    }
    let ps = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS.0 as u32).and_then(|s| payload_shader(s.1));
    // the console program as a host mesh shader where it can be one; else in two passes
    if let Some(hr) = native_mesh_stream(this, &subs, ms, ps, riid, ppv, warm.as_deref_mut()) {
        return Some(hr);
    }
    let draw_gate_ps = ps;
    let source = || {
        let key = own_stream(desc).map_or(0, |owned| owned.record(super::pso_manifest::Kind::MeshStream).key());
        format!("MS {:016x} PS {:016x}, mesh stream record {key:016x}", crate::gpu::shader::fnv64(ms), ps.map_or(0, crate::gpu::shader::fnv64))
    };
    let (tp, (ps, ps_samplers)) = match (console_mesh(ms, ps, wave32_only(this)), two_pass_ps(ps, graphics_wave32(this))) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            note_translation_error(format!("MS: {e}"));
            return Some(E_INVALIDARG);
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return Some(E_INVALIDARG);
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return Some(E_INVALIDARG) };
    let out = mesh_state_stream(rs, &tp.ms, ps.as_deref(), &subs);
    let inspect_records = super::material_gate_views::prepass::records::enabled()
        && draw_gate_ps.is_some() && super::material_gate_views::prepass::records::selected().is_none_or(|h| h == crate::gpu::shader::fnv64(ms));
    let hr = finish_two_pass(this, &out, tp, rs, &ps_samplers, "MS", source, riid, ppv, warm, inspect_records, draw_gate_ps,Some(ms));
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        note_draw_gate_shaders(*ppv as usize, &[Some(ms), draw_gate_ps]);
    }
    Some(hr)
}

/// A mesh pipeline's stream: the title's state (blend, rasteriser, depth, formats …) from its
/// subobjects `subs` with XCL's root signature and shaders.
unsafe fn mesh_state_stream(rs: *mut c_void, ms: &[u8], ps: Option<&[u8]>, subs: &[(u32, &[u8])]) -> StreamOut {
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if let Some(p) = ps {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, p);
    }
    let blend = console_stream_blend(subs);
    for (ty, payload) in subs {
        let skip = [0, 2, 19, 24, 25].contains(ty); // RS, PS, cached PSO, AS, MS
        if !skip {
            if let Some(b) = blend.as_ref().filter(|_| *ty == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND.0 as u32) {
                out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, b);
                continue;
            }
            let (_, palign) = super::null::subobject_payload(*ty).unwrap_or((0, 4));
            // RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT: the console's formats as the PC's
            if (*ty == 15 && payload.len() >= 32) || (*ty == 16 && payload.len() >= 4) {
                let mut p = payload.to_vec();
                for k in 0..if *ty == 15 { 8 } else { 1 } {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()));
                    p[4 * k..4 * k + 4].copy_from_slice(&f.to_le_bytes());
                }
                out.raw(*ty, &p, palign == 8);
                continue;
            }
            out.raw(*ty, payload, palign == 8);
        }
    }
    out
}

/// A console mesh pipeline with the mesh program as a host mesh shader (`pso::NativeMesh`): one
/// translated graphics pipeline, nothing else to run at the draw. None: this pipeline cannot be
/// one here – the adapter's wave size is not fixed at 32 lanes, the program is over a limit of
/// the host's mesh stage (`emit::Em::mesh_native_blocker`), or DXC or the driver refused it –
/// and the caller makes it in two passes. Some(E_INVALIDARG): a shader DXC has not compiled yet
/// inside `attempt` (the build goes to the background), or a failure two passes would share.
unsafe fn native_mesh_stream(this: *mut c_void, subs: &[(u32, &[u8])], ms: &[u8], ps: Option<&[u8]>, riid: *const GUID, ppv: *mut *mut c_void, warm: Option<&mut MeshWarmBuild<'_>>) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_mesh_native, console_to_dxil_for, is_console, mesh_native_split, Stage};
    let split = mesh_native_split(ID3D12Device::from_raw_borrowed(&this)?)?;
    static WHY: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let fall_back = |why: String| {
        // each different reason once: the pipeline is made in two passes instead
        if WHY.lock().unwrap().get_or_insert_with(HashSet::new).insert(why.clone()) {
            crate::xlog_info!("D3D12.X: a console mesh program runs in two passes, not as a host mesh shader: {why}");
        }
    };
    let native = {
        let _translating = crate::progress::TRANSLATIONS.begin();
        let _live = crate::live_stall::stage(crate::live_stall::Stage::ShaderCompile, || crate::gpu::shader::fnv64(ms));
        match console_mesh_native(ms, ps, split) {
            Ok(n) => n,
            Err(_) if crate::gpu::recomp::dxc::missed() => return Some(E_INVALIDARG),
            Err(e) => {
                fall_back(e);
                return None;
            }
        }
    };
    let draw_gate_ps = ps;
    // the pixel shader's static samplers follow the mesh shader's in the root signature
    let (ps, ps_samplers) = match ps {
        Some(p) if is_console(p) => {
            let _translating = crate::progress::TRANSLATIONS.begin();
            match console_to_dxil_for(p, Stage::Pixel, native.samplers.len() as u32, graphics_wave32(this)) {
                Ok(c) => (Some(c.dxil), c.samplers),
                Err(e) => {
                    note_translation_error(format!("PS: {e}"));
                    return Some(E_INVALIDARG);
                }
            }
        }
        Some(p) => (Some(p.to_vec()), Vec::new()),
        None => (None, Vec::new()),
    };
    let samplers = [native.samplers.as_slice(), ps_samplers.as_slice()].concat();
    let Some(rs) = abi_rs_for(this, &samplers) else { return Some(E_INVALIDARG) };
    let out = mesh_state_stream(rs, &native.ms, ps.as_deref(), subs);
    let fs: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let disk = driver_cache::stream_path(&out.0, &samplers);
    let len = native.ms.len() + ps.as_ref().map_or(0, Vec::len);
    if let Some(warm) = warm {
        return match warm.blob(this, disk.as_deref(), len, "host MS", |out_pso| finish_stream(this, fs, &out, None, riid, out_pso)) {
            Ok(hr) if hr < 0 => {
                fall_back(format!("the driver rejected the host mesh shader ({hr:#x})"));
                None
            }
            result => Some(result.unwrap_or(E_INVALIDARG)),
        };
    }
    let driving = std::time::Instant::now();
    let Some((hr, from_blob)) = build_live_mesh_blob(this, disk.as_deref(), len, || format!("host MS {:016x}", crate::gpu::shader::fnv64(ms)), super::pso_library::Desc::Stream(&out.0), riid, ppv, true, |blob| finish_stream(this, fs, &out, blob, riid, ppv), || if ppv.is_null() { None } else { pso_blob(*ppv) }) else {
        return Some(E_INVALIDARG);
    };
    super::pso_async::note_build(|| format!("MS as a host mesh shader: {} B of DXIL; driver {} ms{}", native.ms.len(), driving.elapsed().as_millis(), if from_blob { " (a driver blob kept on disk)" } else { "" }));
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        fall_back(format!("the driver rejected the host mesh shader ({hr:#x})"));
        return None;
    }
    let n = note_translated_graphics(*ppv, rs);
    if super::depth_lineage::enabled() {
        if let Some(subs) = stream_subobjects(&out.0) {
            note_lineage_stream_depth(*ppv as usize, &subs);
        }
    }
    note_draw_gate_shaders(*ppv as usize, &[Some(ms), draw_gate_ps]);
    NATIVE_MESH.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: mesh pipeline from the console MS as a host mesh shader ({} B DXIL, #{})", native.ms.len(), n + 1);
    }
    Some(hr)
}

/// A graphics PSO with a console VS + GS: the GS pipeline in two passes, as a mesh pipeline
/// with the description's state.
unsafe fn translated_gs_pso(this: *mut c_void, d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, vs: &[u8], gs: &[u8], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (tp, (ps, ps_samplers)) = match (
        crate::gpu::recomp::pso::console_geometry(vs, gs, shader_bytes(&d.PS), wave32_only(this)),
        two_pass_ps(shader_bytes(&d.PS), graphics_wave32(this)),
    ) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            // what the containers hold: each FSSM's hardware stages and its programs
            let describe = |blob: &[u8]| -> String {
                crate::gpu::xbox::parse(blob).map_or("not a console container".into(), |sh| {
                    sh.fssm
                        .iter()
                        .map(|f| {
                            format!(
                                "[{} | {} programs {:?}, version {:#x}, {} B, code {:?} (from header {}), error {:?}]",
                                f.stages.iter().map(|x| x.label()).collect::<Vec<_>>().join(", "),
                                f.programs.len(),
                                f.programs.iter().map(|p| p.len).collect::<Vec<_>>(),
                                f.version,
                                f.data.len(),
                                f.code,
                                f.code_from_header,
                                f.error
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            };
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: VS+GS pipeline not translated ({e}); VS {}; GS {}", describe(vs), describe(gs));
                if let Ok(dir) = std::env::var("XCL_DUMP_GS") {
                    let k = N.load(Ordering::Relaxed);
                    let _ = std::fs::write(format!("{dir}/vs{k}.bin"), vs);
                    let _ = std::fs::write(format!("{dir}/gs{k}.bin"), gs);
                }
            }
            note_translation_error(format!("VS+GS: {e}"));
            return E_INVALIDARG;
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return E_INVALIDARG;
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return E_INVALIDARG };
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &tp.ms);
    if let Some(p) = &ps {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, p);
    }
    let mut blend = d.BlendState;
    console_rt_blend(&mut blend, &d.RTVFormats, d.NumRenderTargets);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &d.SampleMask);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &d.RasterizerState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, &d.DepthStencilState);
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: d.NumRenderTargets, ..Default::default() };
    rt.RTFormats = d.RTVFormats;
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &d.DSVFormat);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &d.SampleDesc);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &d.NodeMask);
    if d.Flags.0 != 0 {
        out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &d.Flags);
    }
    let source = || format!("VS {:016x} GS {:016x} PS {:016x}", crate::gpu::shader::fnv64(vs), crate::gpu::shader::fnv64(gs), shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64));
    let hr = finish_two_pass(this, &out, tp, rs, &ps_samplers, "GS", source, riid, ppv, None, false, None,None);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        note_draw_gate_shaders(*ppv as usize, &[Some(vs), shader_bytes(&d.PS), Some(gs)]);
    }
    hr
}

const STANDIN_TOPOLOGY: u32 = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY.0 as u32;

/// Rewrite a pipeline state stream with stand-in shaders. None if not possible.
unsafe fn fallback_stream(desc: *const u8) -> Option<(Vec<u8>, [usize; 2])> {
    let size = *(desc as *const usize);
    let base = *(desc.add(8) as *const usize);
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let d = dummy();
    let subs = known_subobjects(&buf);
    let has_ms = subs.iter().any(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32);
    let move_ms = !has_ms && {
        let as_slot = subs.iter()
            .find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0 as u32)
            .and_then(|s| payload_shader(s.1));
        mesh_shader_slots(None, as_slot).0.is_some()
    };
    let mesh = has_ms || move_ms;
    let dm = if mesh { Some(dummy_mesh()?) } else { None };
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        let at = off + poff;
        let put = |buf: &mut Vec<u8>, b: &[u8]| {
            // no shader: a null pointer – an empty slice's is dangling (0x1), which the driver
            // refused (a retail game's mesh stand-ins, E_INVALIDARG)
            let ptr = if b.is_empty() { 0 } else { b.as_ptr() as u64 };
            buf[at..at + 8].copy_from_slice(&ptr.to_le_bytes());
            buf[at + 8..at + 16].copy_from_slice(&(b.len() as u64).to_le_bytes());
        };
        match ty {
            1 => put(&mut buf, &d.vs),
            2 => put(&mut buf, dm.map_or(&d.ps[..], |m| &m.1[..])),
            6 => put(&mut buf, &d.cs),
            24 if move_ms => {
                buf[off..off + 4].copy_from_slice(&(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).to_le_bytes());
                put(&mut buf, &dm?.0);
            }
            3 | 4 | 5 | 24 => put(&mut buf, &[]),
            25 => put(&mut buf, &dm?.0),
            15 | 16 => {
                let len = if ty == 15 { 32 } else { 4 };
                for format in buf.get_mut(at..at + len)?.chunks_exact_mut(4) {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(format.try_into().ok()?));
                    format.copy_from_slice(&f.to_le_bytes());
                }
            }
            12 | 19 => buf[at..at + 16].fill(0), // input layout, cached PSO
            7 => buf[at..at + 32].fill(0),        // stream output
            // with no HS, GS or DS left: a point, line or triangle type (the console's rect list
            // type 19, none, or PATCH made the driver refuse the stand-in)
            STANDIN_TOPOLOGY if !(1..=3).contains(&u32::from_le_bytes(buf[at..at + 4].try_into().ok()?)) => {
                buf[at..at + 4].copy_from_slice(&(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32).to_le_bytes());
            }
            _ => {}
        }
        off += (poff + psize + 7) & !7;
    }
    Some((buf, [size, 0]))
}

/// The console's pipeline topology type of rect lists (draws of topology 17, `list_x::RECT_LIST`).
const RECT_LIST_TYPE: i32 = 19;

/// `create(copy)`, and once more without the generated rect list GS (`rect_gs`) when the driver
/// refuses the pipeline with it: each rect is then drawn as the triangle of its three vertices,
/// as without the GS, rather than not at all (the untranslated fallback's dummy shaders).
fn retry_without_rect_gs(copy: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool, (vs_hash, ps_hash): (u64, u64), mut create: impl FnMut(&D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> HRESULT) -> HRESULT {
    let hr = create(copy);
    if hr >= 0 || !rect_gs {
        return hr;
    }
    note_translation_error(format!("VS {vs_hash:016x} PS {ps_hash:016x}: the driver rejected the PSO with the rect list GS ({hr:#x}): rect list drawn as triangles"));
    copy.GS = D3D12_SHADER_BYTECODE::default();
    create(copy)
}

/// What an untranslated pipeline from a stream is made from: a rect list's (type 19, which
/// `stream_pipeline` keeps for `translated_graphics_pso`) with the PC's topology type, as the
/// stream path made it before; None: `desc` as it is.
unsafe fn untranslated_stream_desc(desc: *const c_void, stream: bool) -> Option<D3D12_GRAPHICS_PIPELINE_STATE_DESC> {
    if !stream || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let mut g = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        return None;
    }
    pc_topology_type(&mut g);
    Some(g)
}

/// The GS that completes a rect list's rects behind the console VS `vs` and the PS `ps`; refused
/// for a PS that is not a console one (its inputs' modes are not known).
fn rect_list_gs(vs: &[u8], ps: Option<&[u8]>) -> Result<Vec<u8>, String> {
    use crate::gpu::recomp::pso;
    if ps.is_some_and(|p| !pso::is_console(p)) {
        return Err("a PC pixel shader".into());
    }
    pso::rect_list_gs(vs)
}

/// A topology type the PC has: the console's for rect lists (19, with draws of topology 17 –
/// `list_x::RECT_LIST`) is TRIANGLE, as those draws become triangle lists (the rects' fourth
/// corners come from `pso::rect_list_gs`, added by `translated_graphics_pso`); UNDEFINED (a stream
/// without one) is TRIANGLE, or PATCH with a hull shader – the PC refuses both with a VS.
fn pc_topology_type(g: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC) {
    // (and the console's D16S8 depth format the PC's: `device_x::pc_view_format`)
    g.DSVFormat = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(g.DSVFormat.0 as u32) as i32);
    for f in g.RTVFormats.iter_mut() {
        *f = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(f.0 as u32) as i32);
    }
    if !(1..=4).contains(&g.PrimitiveTopologyType.0) {
        g.PrimitiveTopologyType = if g.HS.pShaderBytecode.is_null() { D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE } else { D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH };
    }
}

enum StreamPipeline {
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>),
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>),
    /// neither with console shaders (or not readable)
    Unknown,
}

/// A pipeline stream with a console VS (+ PS / GS) or CS as the plain pipeline desc, which is
/// translated like any other (`dev_create_graphics_pso`). The stream itself must not reach the
/// driver: AMD's shader compiler fails on the console's stripped DXIL and takes the device with
/// it (0x887A0020 at the retail game's first graphics pipeline). Depth bounds, view instancing and the
/// float depth bias' fraction have no place in the plain desc.
unsafe fn stream_pipeline(desc: *const c_void) -> StreamPipeline {
    use crate::gpu::recomp::pso::is_console;
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return StreamPipeline::Unknown;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return StreamPipeline::Unknown;
    }
    let subs = known_subobjects(std::slice::from_raw_parts(base as *const u8, size));
    let console = |ty: u32| subs.iter().find(|s| s.0 == ty).and_then(|s| payload_shader(s.1)).is_some_and(is_console);
    if console(6) {
        let mut c: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::mem::zeroed();
        for (ty, p) in &subs {
            match ty {
                0 => put(&mut c.pRootSignature, p),
                6 => put(&mut c.CS, p),
                18 => put(&mut c.NodeMask, p),
                _ => {}
            }
        }
        return StreamPipeline::Compute(Box::new(c));
    }
    if !console(1) {
        return StreamPipeline::Unknown;
    }
    let mut g: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
    // what a stream leaves out has the PC's defaults
    g.SampleMask = u32::MAX;
    g.SampleDesc.Count = 1;
    g.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    g.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    g.RasterizerState.DepthClipEnable = true.into();
    for rt in &mut g.BlendState.RenderTarget {
        rt.SrcBlend = D3D12_BLEND_ONE;
        rt.DestBlend = D3D12_BLEND_ZERO;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = 15;
    }
    let keep = D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: D3D12_STENCIL_OP_KEEP, StencilDepthFailOp: D3D12_STENCIL_OP_KEEP, StencilPassOp: D3D12_STENCIL_OP_KEEP, StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS };
    g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: true.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
        StencilEnable: false.into(),
        StencilReadMask: 0xFF,
        StencilWriteMask: 0xFF,
        FrontFace: keep,
        BackFace: keep,
    };
    for (ty, p) in &subs {
        match ty {
            0 => put(&mut g.pRootSignature, p),
            1 => put(&mut g.VS, p),
            2 => put(&mut g.PS, p),
            3 => put(&mut g.DS, p),
            4 => put(&mut g.HS, p),
            5 => put(&mut g.GS, p),
            7 => put(&mut g.StreamOutput, p),
            8 => put(&mut g.BlendState, p),
            9 => put(&mut g.SampleMask, p),
            10 => put(&mut g.RasterizerState, p),
            // DEPTH_STENCIL; DEPTH_STENCIL1 is that and the depth bounds test
            11 | 21 => put(&mut g.DepthStencilState, p),
            12 => put(&mut g.InputLayout, p),
            13 => put(&mut g.IBStripCutValue, p),
            14 => put(&mut g.PrimitiveTopologyType, p),
            15 => {
                let mut rt: D3D12_RT_FORMAT_ARRAY = std::mem::zeroed();
                put(&mut rt, p);
                g.RTVFormats = rt.RTFormats;
                g.NumRenderTargets = rt.NumRenderTargets;
            }
            16 => put(&mut g.DSVFormat, p),
            17 => put(&mut g.SampleDesc, p),
            18 => put(&mut g.NodeMask, p),
            20 => put(&mut g.Flags, p),
            26 => {
                let mut d: D3D12_DEPTH_STENCIL_DESC2 = std::mem::zeroed();
                put(&mut d, p);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC1| D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: f.StencilFailOp, StencilDepthFailOp: f.StencilDepthFailOp, StencilPassOp: f.StencilPassOp, StencilFunc: f.StencilFunc };
                g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: d.DepthEnable,
                    DepthWriteMask: d.DepthWriteMask,
                    DepthFunc: d.DepthFunc,
                    StencilEnable: d.StencilEnable,
                    StencilReadMask: d.FrontFace.StencilReadMask,
                    StencilWriteMask: d.FrontFace.StencilWriteMask,
                    FrontFace: face(&d.FrontFace),
                    BackFace: face(&d.BackFace),
                };
            }
            27 | 28 => {
                // RASTERIZER1 / 2: fill, cull, winding, depth bias (float), clamp, slope, depth clip
                let mut d: D3D12_RASTERIZER_DESC1 = std::mem::zeroed();
                put(&mut d, p);
                g.RasterizerState = D3D12_RASTERIZER_DESC {
                    FillMode: d.FillMode,
                    CullMode: d.CullMode,
                    FrontCounterClockwise: d.FrontCounterClockwise,
                    DepthBias: d.DepthBias as i32,
                    DepthBiasClamp: d.DepthBiasClamp,
                    SlopeScaledDepthBias: d.SlopeScaledDepthBias,
                    DepthClipEnable: d.DepthClipEnable,
                    ..Default::default()
                };
            }
            _ => {}
        }
    }
    // a rect list stays one for translated_graphics_pso, which gives it its GS (untranslated, it
    // becomes the PC's in graphics_pso)
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        pc_topology_type(&mut g);
    }
    StreamPipeline::Graphics(Box::new(g))
}

/// The bytes of a stream payload as a desc field (no references counted: the desc only borrows).
unsafe fn put<T>(to: &mut T, from: &[u8]) {
    let n = std::mem::size_of::<T>().min(from.len());
    std::ptr::copy_nonoverlapping(from.as_ptr(), to as *mut T as *mut u8, n);
}

pub(super) unsafe extern "system" fn dev_create_pipeline_state(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    create_pipeline_stream(this, f, desc, riid, ppv)
}

unsafe fn create_pipeline_stream(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    use crate::gpu::recomp::pso::is_console;
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    if desc as usize % 8 != 0 || !crate::gpu::shader::readable(desc as usize, 16) {
        return E_INVALIDARG;
    }
    let [size, base] = *desc.cast::<[usize; 2]>();
    if !crate::gpu::shader::readable(base, size) || base % 8 != 0 {
        return E_INVALIDARG;
    }
    let canonical = super::pso_canon::enabled().then(|| {
        let mut s = StreamOut(std::slice::from_raw_parts(base as *const u8, size).to_vec());
        super::pso_canon::stream(&mut s.0);
        s
    });
    let canonical_desc = canonical.as_ref().map(|s| [s.0.len(), s.0.as_ptr() as usize]);
    let desc = canonical_desc.as_ref().map_or(desc, |d| d.as_ptr().cast());
    let base = canonical.as_ref().map_or(base, |s| s.0.as_ptr() as usize);
    let Some(subs) = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size)) else {
        crate::xlog_warn!("D3D12.X: unknown or truncated pipeline stream subobject");
        return E_INVALIDARG;
    };
    let mut shaders = [None; 26];
    let mut mesh_tags = [24, 25];
    let mut seen = 0u32;
    for &(ty, payload) in &subs {
        let bit = 1 << ty;
        if seen & bit != 0 {
            return E_INVALIDARG;
        }
        seen |= bit;
        if matches!(ty, 1..=6 | 24 | 25) && payload[8..16] != [0; 8] {
            let Some(shader) = payload_shader(payload) else { return E_INVALIDARG };
            let tag = if matches!(ty, 24 | 25) {
                let tag = match crate::gpu::xbox::program_kind(shader) {
                    Some(13) => D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32,
                    Some(14) => D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0 as u32,
                    _ => ty,
                };
                mesh_tags[(ty - 24) as usize] = tag;
                tag
            } else {
                ty
            };
            if shaders[tag as usize].replace(shader).is_some() {
                return E_INVALIDARG;
            }
        }
    }
    // RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT: the console's formats as the PC's
    let pc_formats = |ty: u32, payload: &[u8]| -> Option<Vec<u8>> {
        let n = match ty {
            15 if payload.len() >= 32 => 8,
            16 if payload.len() >= 4 => 1,
            _ => return None,
        };
        let mut p = payload.to_vec();
        for k in 0..n {
            let f = super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()));
            p[4 * k..4 * k + 4].copy_from_slice(&f.to_le_bytes());
        }
        (p != payload).then_some(p)
    };
    let normalized = (mesh_tags != [24, 25] || subs.iter().any(|&(ty, payload)| pc_formats(ty, payload).is_some())).then(|| {
        let mut stream = StreamOut::default();
        for &(ty, payload) in &subs {
            let tag = if matches!(ty, 24 | 25) {
                if payload[8..16] == [0; 8] {
                    continue;
                }
                mesh_tags[(ty - 24) as usize]
            } else {
                ty
            };
            let (_, align) = super::null::subobject_payload(tag).unwrap();
            match pc_formats(ty, payload) {
                Some(p) => stream.raw(tag, &p, align == 8),
                None => stream.raw(tag, payload, align == 8),
            }
        }
        stream
    });
    let adjusted = normalized.as_ref().map(|s| [s.0.len(), s.0.as_ptr() as usize]);
    let desc = adjusted.as_ref().map_or(desc, |d| d.as_ptr().cast());
    super::null::capture_stream_desc(desc as *const u8);
    if let Some(vs) = shaders[1].filter(|vs| is_console(vs)) {
        if (3..=6).chain([24, 25]).any(|ty| shaders[ty].is_some()) {
            note_translation_error("graphics stream with a console VS and additional shader stages – not supported yet".into());
            return E_INVALIDARG;
        }
        let subs = normalized.as_ref().map_or(Some(subs), |s| stream_subobjects(&s.0)).unwrap();
        if !super::pso_async::enabled() {
            return translated_vertex_stream(this, f, &subs, vs, shaders[2], riid, ppv);
        }
        // as the other translated paths: a shader DXC has not compiled yet gives the title a
        // stand-in now and the real pipeline from a worker (a retail game's loader made its
        // pipelines this way, each call waiting a second for DXC, three loader threads at a time)
        let (hr, missed) = attempt(|| translated_vertex_stream(this, f, &subs, vs, shaders[2], riid, ppv));
        if !missed {
            return hr;
        }
        if !ppv.is_null() && !(*ppv).is_null() {
            super::pso_async::release(std::mem::replace(&mut *ppv, std::ptr::null_mut()) as usize);
        }
        if let Some(hr) = own_stream(desc).and_then(|o| go_background(this, f, desc, riid, ppv, super::pso_manifest::Kind::VertexStream, o, standin_stream, build_vertex_stream)) {
            return hr;
        }
        return translated_vertex_stream(this, f, &subs, vs, shaders[2], riid, ppv);
    }
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_mesh_stream(this, desc, riid, ppv, None)) } else { (translated_mesh_stream(this, desc, riid, ppv, None), false) };
    if missed {
        if let Some(hr) = own_stream(desc).and_then(|o| go_background(this, f, desc, riid, ppv, super::pso_manifest::Kind::MeshStream, o, standin_stream, build_stream)) {
            return hr;
        }
        if let Some(hr) = translated_mesh_stream(this, desc, riid, ppv, None) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    match stream_pipeline(desc) {
        StreamPipeline::Graphics(d) => return graphics_pso(this, &*d as *const _ as *const c_void, riid, ppv, true),
        StreamPipeline::Compute(d) => return dev_create_compute_pso(this, &*d as *const _ as *const c_void, riid, ppv),
        StreamPipeline::Unknown => {}
    }
    let hr = if shaders.into_iter().flatten().any(is_console) {
        E_INVALIDARG
    } else {
        finish_pso(this, f, desc, riid, ppv)
    };
    if hr >= 0 {
        if !ppv.is_null() && !(*ppv).is_null() {
            note_lineage_stream_depth(*ppv as usize, &subs);
        }
        return hr;
    }
    let Some((buf, _)) = fallback_stream(desc as *const u8) else {
        note_fallback("PSO stream", hr, false);
        return hr;
    };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    let hr2 = note_standin(finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv), ppv);
    note_fallback("PSO stream", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe extern "system" fn dev_create_root_signature(this: *mut c_void, mask: u32, blob: *const c_void, len: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    crate::gpu::shader::capture("rootsig", blob as usize, len);
    type F = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRootSignature")));
    let mut hr = f(this, mask, blob, len, riid, ppv);
    if hr == E_NOINTERFACE {
        let iid = pc_iid(Kind::RootSignature);
        hr = f(this, mask, blob, len, &iid, ppv);
    }
    if hr < 0 {
        // console root signature rejected → empty one so the title keeps going
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let mut ser = None;
        if D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).is_ok() {
            if let Some(b) = ser {
                let iid = pc_iid(Kind::RootSignature);
                let hr2 = f(this, mask, b.GetBufferPointer(), b.GetBufferSize(), &iid, ppv);
                if hr2 >= 0 {
                    let n = FALLBACK_RS.fetch_add(1, Ordering::Relaxed);
                    if n < 5 {
                        crate::xlog_warn!("D3D12.X: root signature rejected ({hr:#x}) – empty fallback (#{})", n + 1);
                    }
                    hr = hr2;
                }
            }
        }
    }
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::RootSignature);
        if !blob.is_null() {
            super::console_list::note_root_signature(*ppv as usize, std::slice::from_raw_parts(blob as *const u8, len));
        }
    }
    hr
}

// ---------------------------------------------------------------------------------------
// presentation tracking
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn dev_create_rtv(this: *mut c_void, res: *mut c_void, desc: *const c_void, handle: usize) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, usize);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRenderTargetView")));
    // a view in a console-only format, or of a target made R8G8B8A8 (`device_x::pc_resource_desc`)
    let mut copy = (!desc.is_null()).then(|| (desc as *const D3D12_RENDER_TARGET_VIEW_DESC).read_unaligned());
    if let Some(c) = copy.as_mut() {
        let rf = ID3D12Resource::from_raw_borrowed(&res).map_or(0, |r| r.GetDesc().Format.0 as u32);
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rf, c.Format.0 as u32) as i32);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _ as *const c_void);
    f(this, res, desc, handle);
    image_views::rtv(this as usize, handle, res, copy.as_ref());
    super::present::note_rtv(handle, res);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let d = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0));
            crate::xlog_info!("frame trace: CreateRenderTargetView {handle:#x} → {res:p} {d:?}");
        }
    }
}

/// ClearRenderTargetView: as on the PC (`XCL_TRACE_FRAME`: the colours logged)
unsafe extern "system" fn cl_clear_rtv(this: *mut c_void, rtv: usize, color: *const [f32; 4], n: u32, rects: *const c_void) {
    type F = unsafe extern "system" fn(*mut c_void, usize, *const [f32; 4], u32, *const c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ClearRenderTargetView")));
    image_views::record(this, || image_views::clear(rtv, image_views::Clear::Color(*color.as_ref().ok_or("missing clear color")?), n, rects.cast()));
    if super::present::writer_trace_active() {
        if let Some(r) = super::present::rtv_resource(rtv) {
            super::present::record_plane_write(this as usize, r, "RTV clear", 0, vec![format!("colour {:?}, {n} rects", color.as_ref())]);
        }
    }
    if super::target_dump::recording() {
        if let Some(r) = super::present::rtv_resource(rtv) {
            super::target_dump::record_write(this as usize, r, "RTV clear");
        }
    }
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_info!("frame trace: ClearRenderTargetView {rtv:#x} → {:?} colour {:?}, {n} rects", super::present::rtv_resource(rtv), color.as_ref());
        }
        if let Some(r) = super::present::rtv_resource(rtv) {
            super::present::note_use(r, super::present::USE_CLEAR);
        }
    }
    super::gpu_timing::class(this, super::gpu_timing::Kind::Clear, || f(this, rtv, color, n, rects))
}

unsafe extern "system" fn cl_om_set_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "OMSetRenderTargets")));
    image_views::record(this, || image_views::bind(n, rtvs, single, dsv));
    f(this, n, rtvs, single, dsv);
    super::list_x::note_render_targets(this, n, rtvs, single, dsv);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let first = if n > 0 && !rtvs.is_null() { *rtvs } else { 0 };
            crate::xlog_info!("frame trace: OMSetRenderTargets({n}, first {first:#x} → {:?}, single {single}, dsv {:?})", super::present::rtv_resource(first), dsv.as_ref().copied());
        }
    }
    if n > 0 && !rtvs.is_null() {
        super::present::note_rt0(this as usize, *rtvs);
        if super::present::trace_frame() {
            for k in 0..n as usize {
                let h = if single != 0 { *rtvs + k * 32 } else { *rtvs.add(k) };
                if let Some(r) = super::present::rtv_resource(h) {
                    super::present::note_use(r, super::present::USE_RT);
                }
            }
        }
    }
}

/// A stand-in pipeline is bound as its real pipeline once that is built (compute: waited for).
/// SetPipelineState on the PC list (a pipeline still building in the background: its stand-in)
pub(super) unsafe fn bind_pipeline(this: *mut c_void, pso: *mut c_void) {
    cl_set_pipeline_state(this, pso)
}

unsafe extern "system" fn cl_set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "SetPipelineState")));
    let native = super::pso_async::resolve_for_bind(pso as usize);
    f(this, native as *mut c_void);
    if native != pso as usize {
        super::shader_code::used(this as usize, native);
    }
    super::list_x::on_set_pipeline_state(this, pso as usize);
    super::list_x::capture_native_bind(this, native);
    super::gpu_timing::pipeline(this, pso as usize);
}

unsafe extern "system" fn cl_clear_state(this: *mut c_void, pso: *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ClearState")));
    let native = super::pso_async::resolve_for_bind(pso as usize);
    f(this, native as *mut c_void);
    if native != pso as usize {
        super::shader_code::used(this as usize, native);
    }
    super::list_x::on_clear_state(this, pso as usize);
    super::list_x::capture_native_bind(this, native);
    super::bind::clear_state(this, pso as usize);
    super::gpu_timing::pipeline(this, pso as usize);
}

unsafe extern "system" fn cl_set_predication(this: *mut c_void, resource: usize, offset: u64, op: D3D12_PREDICATION_OP) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "SetPredication")));
    let title = super::bind::resource_title(resource, offset);
    if super::bind::ignore_predication() {
        f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        super::list_x::note_draw_predicate(this, None, 0, 0, 0);
    } else {
        super::residency::note_list_resources(this as usize, [resource]);
        f(this, resource, offset, op);
        super::list_x::note_draw_predicate(this, title, resource, offset, op.0);
    }
    super::bind::draw_gate_predication(this, "SetPredication(PC)", format_args!("resource {resource:#x}+{offset:#x} title {title:x?} op {}", op.0), title);
}

unsafe extern "system" fn cl_draw_instanced(this: *mut c_void, vcount: u32, icount: u32, sv: u32, si: u32) {
    if super::bind::draw_gate_calls() {
        let pso = super::list_x::draw_gate_pso(this);
        let skipped = super::bind::shader_skip(pso);
        super::bind::draw_gate_call(this, pso, "DrawInstanced(PC)", format_args!("vertices {vcount} instances {icount} first_vertex {sv} first_instance {si}"), skipped);
        if skipped { return }
    }
    super::list_x::prepare_draw(this);
    if vcount != 0 && icount != 0 {
        super::list_x::note_lineage_depth_draw(this, "draw");
    }
    super::depth_uav::note_op(this);
    super::image_writeback::note_work(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "DrawInstanced")));
    super::gpu_timing::op(this, super::gpu_timing::Kind::Draw, || 0, || f(this, vcount, icount, sv, si));
}

unsafe extern "system" fn cl_draw_indexed_instanced(this: *mut c_void, icount: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
    if super::bind::draw_gate_calls() {
        let pso = super::list_x::draw_gate_pso(this);
        let skipped = super::bind::shader_skip(pso);
        super::bind::draw_gate_call(this, pso, "DrawIndexedInstanced(PC)", format_args!("indices {icount} instances {inst} first_index {si} base_vertex {bv} first_instance {sinst}"), skipped);
        if skipped { return }
    }
    super::list_x::prepare_draw(this);
    if icount != 0 && inst != 0 {
        super::list_x::note_lineage_depth_draw(this, "draw_indexed");
    }
    super::depth_uav::note_op(this);
    super::image_writeback::note_work(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "DrawIndexedInstanced")));
    super::gpu_timing::op(this, super::gpu_timing::Kind::Draw, || 0, || f(this, icount, inst, si, bv, sinst));
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn cl_execute_indirect(
    this: *mut c_void,
    sig: *mut c_void,
    max_count: u32,
    arg_buf: *mut c_void,
    arg_off: u64,
    cnt_buf: *mut c_void,
    cnt_off: u64,
) {
    super::list_x::prepare_draw(this);
    let requested = max_count;
    let max_count = super::bind::indirect_max(requested);
    let mut event = None;
    if super::bind::draw_gate_calls() {
        let pso = super::list_x::draw_gate_pso(this);
        let skipped = super::bind::shader_skip(pso);
        event = super::bind::draw_gate_indirect(this, pso, sig as usize, requested, max_count, arg_buf as usize, arg_off, cnt_buf as usize, cnt_off, skipped);
        if skipped { return }
    }
    if super::depth_capture_views::enabled() && super::indirect::signature(sig as usize).is_some_and(|(kind, _, _)| matches!(kind, 0 | 1)) {
        if max_count != 0 {
            super::list_x::note_lineage_depth_draw(this, "indirect_draw");
        }
        if let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) {
            let native = super::list_x::capture_native_pso(this);
            let visible = super::pso_async::resolve_if_ready(super::list_x::draw_gate_pso(this));
            super::bind::capture_selected_depth(this, list, native, visible, event, true);
        }
    }
    super::residency::note_list_resources(this as usize, [arg_buf as usize, cnt_buf as usize]);
    super::depth_uav::note_op(this);
    super::image_writeback::note_work(this);
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, u64, *mut c_void, u64);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteIndirect")));
    super::gpu_timing::op(this, super::gpu_timing::Kind::Indirect, || 0, || f(this, sig, max_count, arg_buf, arg_off, cnt_buf, cnt_off));
}

unsafe extern "system" fn cl_timed_begin_render_pass(
    this: *mut c_void,
    count: u32,
    targets: *const D3D12_RENDER_PASS_RENDER_TARGET_DESC,
    depth: *const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC,
    flags: D3D12_RENDER_PASS_FLAGS,
) {
    let slot = std::mem::offset_of!(ID3D12GraphicsCommandList4_Vtbl, BeginRenderPass) / size_of::<usize>();
    let (target, adopted) = original_or_native(this, slot);
    let f: unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RENDER_PASS_RENDER_TARGET_DESC, *const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC, D3D12_RENDER_PASS_FLAGS) =
        std::mem::transmute(target);
    if adopted { image_views::record(this, || image_views::begin(count, targets, depth, flags)); }
    f(this, count, targets, depth, flags);
    if adopted {
        let depth_desc = depth.as_ref();
        super::list_x::note_lineage_render_pass(
            this, true, depth_desc.map(|d| d.cpuDescriptor.ptr),
            !flags.contains(D3D12_RENDER_PASS_FLAG_BIND_READ_ONLY_DEPTH)
                && depth_desc.is_some_and(|d| d.DepthBeginningAccess.Type != D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS),
        );
        super::depth_state::render_pass(this as usize, Some(flags));
        super::gpu_timing::render_pass(this, Some(flags));
        if super::depth_capture_views::enabled() && flags.0 & D3D12_RENDER_PASS_FLAG_RESUMING_PASS.0 == 0 {
            if let Some(d) = depth.as_ref().filter(|d| d.DepthBeginningAccess.Type == D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR) {
                let value = d.DepthBeginningAccess.Anonymous.Clear.ClearValue.Anonymous.DepthStencil.Depth;
                super::depth_capture_views::note_clear(this, d.cpuDescriptor.ptr, D3D12_CLEAR_FLAG_DEPTH.0 as u32, value, 0);
            }
            if super::depth_lineage::enabled() {
                if let Some(d) = depth_desc.filter(|d| d.StencilBeginningAccess.Type == D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR) {
                    super::depth_capture_views::note_clear(this, d.cpuDescriptor.ptr, D3D12_CLEAR_FLAG_STENCIL.0 as u32, 0.0, 0);
                }
            }
        }
    }
}

unsafe extern "system" fn cl_timed_end_render_pass(this: *mut c_void) {
    let slot = std::mem::offset_of!(ID3D12GraphicsCommandList4_Vtbl, EndRenderPass) / size_of::<usize>();
    let (target, adopted) = original_or_native(this, slot);
    let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(target);
    if adopted { image_views::record(this, || Ok(image_views::Event::End)); }
    f(this);
    if adopted {
        super::list_x::note_lineage_render_pass(this, false, None, false);
        super::depth_state::render_pass(this as usize, None);
        super::gpu_timing::render_pass(this, None);
    }
}

unsafe extern "system" fn cl_timed_dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    let slot = std::mem::offset_of!(ID3D12GraphicsCommandList6_Vtbl, DispatchMesh) / size_of::<usize>();
    let (target, adopted) = original_or_native(this, slot);
    let f: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
        std::mem::transmute(target);
    if !adopted {
        return f(this, x, y, z);
    }
    if super::bind::draw_gate_calls() {
        let pso = super::list_x::draw_gate_pso(this);
        let skipped = super::bind::shader_skip(pso);
        super::bind::draw_gate_call(this, pso, "DispatchMesh(PC)", format_args!("groups {x} {y} {z}"), skipped);
        if skipped { return }
    }
    super::image_writeback::note_work(this);
    if x != 0 && y != 0 && z != 0 {
        super::list_x::note_lineage_depth_draw(this, "native_mesh_dispatch");
    }
    super::gpu_timing::op_grid(this, super::gpu_timing::Kind::Draw, || 0, [x, y, z], || f(this, x, y, z));
}

/// ExecuteBundle: a bundle with a site recipe runs as its private instance for this list
/// (`recordings::bundle_site`), which reads the list's descriptor heaps; any other runs itself.
unsafe extern "system" fn cl_execute_bundle(this: *mut c_void, bundle: *mut c_void) {
    super::image_writeback::note_work(this);
    let native = super::console_list::pc_list(bundle).or_else(|| super::title_list::backing(bundle)).unwrap_or(bundle);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteBundle")));
    match super::physical_directory::recordings::bundle_site(this as usize, native as usize) {
        Some(site) => {
            super::list_x::ensure_view_heap(this);
            f(this, site.as_raw());
        }
        None => f(this, native),
    }
}

unsafe extern "system" fn cl_release(this: *mut c_void) -> u32 {
    let timing = super::gpu_timing::list_token(this);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = super::physical_directory::recordings::release(this as usize, || f(this));
    if n == 0 {
        super::gpu_timing::list_released(timing);
        super::residency::list_reset(this as usize);
        super::depth_uav::list_reset(this as usize);
        super::list_x::on_release(this as usize);
        super::bind::forget(this);
    }
    n
}

unsafe extern "system" fn q_release(this: *mut c_void) -> u32 {
    let timing = super::gpu_timing::queue_token(this);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::page_mappings::queue::released(this as usize);
        super::gpu_timing::queue_released(timing);
        super::residency::queue_released(this as usize);
        super::list_x::on_queue_release(this as usize);
        super::cpu_writes::forget_queue(this as usize);
    }
    n
}

unsafe extern "system" fn alloc_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::list_x::on_allocator_release(this as usize);
    }
    n
}

unsafe extern "system" fn alloc_reset(this: *mut c_void) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "Reset")));
    let hr = f(this);
    if hr >= 0 {
        // the GPU is done with what lists recorded from this allocator used
        super::console_list::allocator_reset(this);
        super::list_x::on_allocator_reset(this as usize);
    }
    hr
}

// ---------------------------------------------------------------------------------------
// command queue layouts
// ---------------------------------------------------------------------------------------
//
// Each console title runs on the game OS it ships with, and the D3D12 runtime there sets the
// queue's vtable. Two layouts are known, with the same D3D12 SDK version (0x20004) and the
// same slots up to ExecuteCommandLists (#10):
// * the older (XDefiant, Minecraft, Fortnite): the PC's #11–#18 (SetMarker … Signal #14, Wait
//   #15, GetTimestampFrequency #16, GetClockCalibration #17, GetDesc #18), SignalX #23,
//   PresentX #30;
// * a newer one (Unicorn Overlord): two more methods before Signal – Signal #16, Wait #17,
//   GetTimestampFrequency #18, GetClockCalibration #19, GetDesc #20, SignalX #25, PresentX #32.
// Nothing the title passes at device creation tells them apart, but its calls do: Signal and
// Wait take a fence, the older layout's #16 / #17 (GetTimestampFrequency / GetClockCalibration)
// never do. A title calling #16 / #17 with one of its fences uses the newer layout, and the
// slots that differ follow that from then on.
//
// The same runtime moves the command list's console methods too: Unicorn Overlord records its
// texture uploads with CopyTextureRegionX at #85 (the older layout's #83; its #85 is
// SetPredicationBufferX, which dropped them – every texture stayed zero). From #85 on the
// list's console methods sit two slots later (`NEWER_LIST_SLOTS`), and from #72 on as well: a retail game
// calls Write64BitValueBottomOfPipeX at #72 and Wait64BitValueX at #74, Write32BitValueBottomOfPipeX
// at #71 (address, value, 0, 0) and ExecuteIndirect at #61 (signature, 1, argument buffer, offset,
// NULL, 0) – sent to FillMemoryWith32BitValueX, the title's indirect draws and dispatches never ran.
// So the shift starts at #59 or below; where is not known. The newer layout's #139 is a Reset with one argument more (the same
// title: (allocator, NULL, 0) on each list it takes from its pool before recording – as a stub
// the list stayed closed, the debug layer saw every call on it refused, and Close failed).

/// the title's runtime has the newer layouts (queue: two methods before Signal; command list:
/// console methods from CopyTextureRegionX on two slots later)
pub(super) static QUEUE_SHIFTED: AtomicBool = AtomicBool::new(false);

/// the title's runtime has the older queue layout: it signalled / waited on a fence at #14 / #15
/// (on the newer layout those slots are other methods)
static QUEUE_OLDER: AtomicBool = AtomicBool::new(false);

/// command list slots of the newer layout → the older layout's slot of the same method
const NEWER_LIST_SLOTS: [(usize, usize); 22] = [(61, 59), (62, 60), (63, 61), (64, 62), (65, 63), (69, 67), (70, 68), (71, 69), (72, 70), (73, 71), (74, 72), (80, 78), (83, 81), (85, 83), (86, 84), (87, 85), (90, 88), (91, 89), (92, 90), (95, 93), (96, 94), (139, 10)];

/// Mesh dispatches of the command list: (the older layout's slot, the newer layout's, the
/// method). DispatchMesh(x, y, z) is the last method of the list's sixth version – #127, and two
/// slots later on the newer layout (a retail game calls #129 with three group counts);
/// DispatchMeshX(x, flags, indirect arguments, payload) is #98 / #100.
const MESH_LIST_SLOTS: [(usize, usize, usize); 2] = [(127, 129, 0), (98, 100, 1)];

/// The method of `MESH_LIST_SLOTS`' third field.
fn mesh_list_method(k: usize) -> usize {
    [super::bind::dispatch_mesh as usize, super::bind::dispatch_mesh_x as usize][k]
}

/// The title called device method `slot`, which only runtimes with the newer layouts have (a retail game
/// records its first lists before any queue call could tell).
pub(super) fn note_newer_device_method(slot: usize) {
    if !QUEUE_OLDER.load(Ordering::Acquire) && !QUEUE_SHIFTED.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (device method #{slot} called)");
    }
}

/// The queue layout flags `queue_dispatch` reads and sets: (`QUEUE_SHIFTED`, `QUEUE_OLDER`), or
/// (tests) this thread's own, so a test of the layout detection leaves the process's alone.
fn queue_flags() -> (&'static AtomicBool, &'static AtomicBool) {
    #[cfg(test)]
    if let Some(flags) = TEST_QUEUE_FLAGS.with(|f| f.get()) {
        return flags;
    }
    (&QUEUE_SHIFTED, &QUEUE_OLDER)
}

#[cfg(test)]
thread_local! {
    static TEST_QUEUE_FLAGS: std::cell::Cell<Option<(&'static AtomicBool, &'static AtomicBool)>> = const { std::cell::Cell::new(None) };
}

/// Whether `p` is a fence the title created.
pub(super) fn is_fence(p: usize) -> bool {
    p != 0 && FENCES.read().unwrap_or_else(|e| e.into_inner()).contains(&p)
}

/// first queue slot `queue_dispatch` handles
const QUEUE_FIRST: usize = 14;

/// A title's call of queue slot `slot` (arguments as integers; Signal / Wait / SignalX /
/// PresentX / GetTimestampFrequency / GetClockCalibration / GetDesc all fit).
unsafe fn queue_dispatch(slot: usize, this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
    const PC_SIGNAL: usize = 14;
    const PC_WAIT: usize = 15;
    const PC_TIMESTAMP_FREQUENCY: usize = 16;
    const PC_CLOCK_CALIBRATION: usize = 17;
    const PC_GET_DESC: usize = 18;
    let (shifted_flag, older_flag) = queue_flags();
    if (slot == 16 || slot == 17) && is_fence(a1) && !shifted_flag.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (queue: Signal #16, PresentX #32; command list: CopyTextureRegionX #85 – a fence passed to queue #{slot})");
    }
    let shifted = shifted_flag.load(Ordering::Acquire);
    if (slot == 14 || slot == 15) && !shifted && is_fence(a1) {
        older_flag.store(true, Ordering::Release);
    }
    // the method in the older layout's numbering
    let older = if shifted && slot >= 16 { slot - 2 } else if shifted { usize::MAX } else { slot };
    // (`single`) the title's call on one of its queues: a Signal or Wait takes its place in
    // the queue's held work, lists are held by ExecuteCommandLists itself, and anything else
    // runs on the PC queue after the queue's held work
    let this = match super::single::host_of(this) {
        Some(_) if older == 33 && shifted => this,
        Some(host) if super::single::running() => host,
        Some(host) => match older {
            14 | 23 if is_fence(a1) => {
                let host = host as usize;
                super::single::signal(this as usize, a1, a2 as u64, move || {
                    queue_dispatch(slot, host as *mut c_void, a1, a2, a3, a4);
                });
                return 0;
            }
            15 if is_fence(a1) => {
                super::single::wait(this as usize, a1, a2 as u64);
                return 0;
            }
            _ => super::single::host_after_held(this, "a queue method that cannot be held"),
        },
        None => this,
    };
    let pc = |s: usize| orig_fn(this, s);
    match older {
        // Signal / Wait take a fence: anything else is not one of them
        14 | 15 if is_fence(a1) => {
            if older == 14 {
                super::list_x::break_cycles_before_title_signal(this as usize, a1, a2 as u64);
            }
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(if older == 14 { PC_SIGNAL } else { PC_WAIT }));
            if older == 15 {
                super::wait_trace::wait(a1, a2 as u64, "queue Wait");
            }
            let hr = f(this, a1, a2);
            if hr >= 0 {
                if older == 14 {
                    super::fence_x::note_queue_signal(a1, a2 as u64);
                    super::list_x::note_title_queue_signal(this as usize, a1, a2 as u64);
                    super::wait_trace::signal(a1, a2 as u64, "queue Signal");
                    note_profiled_gpu_signal(this, a1, a2 as u64);
                } else {
                    super::fence_x::note_queue_wait(a1, a2 as u64);
                    super::list_x::note_title_queue_wait(this as usize, a1, a2 as u64);
                    note_profiled_gpu_wait(this, a1, a2 as u64);
                }
            }
            hr as u32 as usize
        }
        16 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> i32 = std::mem::transmute(pc(PC_TIMESTAMP_FREQUENCY));
            f(this, a1) as u32 as usize
        }
        17 => {
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(PC_CLOCK_CALIBRATION));
            f(this, a1, a2) as u32 as usize
        }
        // #18 before either layout is known: the older layout's GetDesc (a 16-byte structure) or
        // the newer one's GetTimestampFrequency (a UINT64). Only the first 8 bytes of the
        // description are written (type and priority), so an 8-byte out-param is not overrun.
        18 if !shifted && !older_flag.load(Ordering::Acquire) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: queue #18 before the title's first Signal / Wait: the queue layout is not known yet, GetDesc writes only its first 8 bytes"));
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            let mut desc = [0u8; 16];
            f(this, desc.as_mut_ptr() as usize);
            if a1 != 0 {
                std::ptr::copy_nonoverlapping(desc.as_ptr(), a1 as *mut u8, 8);
            }
            a1
        }
        // GetDesc returns its structure through the hidden pointer, and that pointer
        18 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            f(this, a1)
        }
        23 => q_signal_x(this, a1 as *mut c_void, a2 as u64, a3 as u32) as u32 as usize,
        29 => super::page_mappings::copy_batch(this, a1 as u32, a2 as *const super::page_mappings::Batch, a3, a4 as u32) as u32 as usize,
        30 => super::present::present_x(this, a1 as u32, a2 as *const u8, a3 as *const u8) as u32 as usize,
        // the newer layout's #35: ExecuteCommandListsX(count, lists, a value, flags) – a retail
        // game submits all its command lists this way (flags 3; 0 for the one it recorded at
        // start); as a stub nothing the title recorded ran on the GPU but its first two frames
        33 if shifted => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_info!("D3D12.X: queue #35 ExecuteCommandListsX ({} lists, value {a3:#x}, flags {a4:#x})", a1));
            q_execute_command_lists(this, a1 as u32, a2 as *const *mut c_void);
            0
        }
        _ => {
            // a console method not modelled: logged once, S_OK (as the stubs it replaces; frame
            // pacing on queue methods keeps working for titles without PresentX)
            static INFOS: Mutex<Vec<(usize, &'static stubs::StubInfo)>> = Mutex::new(Vec::new());
            let info = {
                let mut g = INFOS.lock().unwrap();
                match g.iter().find(|e| e.0 == slot) {
                    Some(e) => e.1,
                    None => {
                        let i = stubs::new_info(DLL, &format!("ID3D12CommandQueue::#{slot} ({} layout)", if shifted { "newer" } else { "older" }), 0);
                        i.hook.store(super::present::on_queue_console_method as usize, Ordering::Release);
                        g.push((slot, i));
                        i
                    }
                }
            };
            let _ = a4;
            stubs::record_call(info, 0);
            0
        }
    }
}

macro_rules! queue_slots {
    ($($slot:literal),*) => {
        /// the title's entry for each queue slot from `QUEUE_FIRST`
        static QUEUE_SLOTS: &[unsafe extern "system" fn(*mut c_void, usize, usize, usize, usize) -> usize] = &[$({
            unsafe extern "system" fn f(this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
                queue_dispatch($slot, this, a1, a2, a3, a4)
            }
            f
        }),*];
    };
}
queue_slots!(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

/// command queue #23 SignalX(fence, value, flush flags): Signal with the console's cache flush
/// options – the PC's Signal makes the queue's writes visible itself. (A stub that signalled
/// nothing left Fortnite's render fences pending: its render thread waited, textures were
/// never released, and the garbage collector ended the game after 10 s.)
unsafe extern "system" fn q_signal_x(this: *mut c_void, fence: *mut c_void, value: u64, _flags: u32) -> HRESULT {
    const PC_SIGNAL: usize = 14;
    super::list_x::break_cycles_before_title_signal(this as usize, fence as usize, value);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, PC_SIGNAL));
    let hr = f(this, fence, value);
    if hr >= 0 {
        super::fence_x::note_queue_signal(fence as usize, value);
        super::list_x::note_title_queue_signal(this as usize, fence as usize, value);
        super::wait_trace::signal(fence as usize, value, "queue SignalX");
        note_profiled_gpu_signal(this, fence as usize, value);
    }
    hr
}

/// Fence::Signal (CPU)
unsafe extern "system" fn fence_signal(this: *mut c_void, value: u64) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "Signal")));
    let hr = f(this, value);
    super::fence_x::note_cpu_signal(this as usize);
    super::wait_trace::signal(this as usize, value, "CPU Signal");
    if hr >= 0 && super::gpu_timing::enabled() {
        super::gpu_timing_waits::signal(None, super::gpu_timing::epoch(), this as usize, value);
    }
    super::single::poke();
    hr
}

unsafe extern "system" fn timing_fence_release(this: *mut c_void) -> u32 {
    let (target, adopted) = original_or_native(this, 2);
    let cookie = (adopted && super::gpu_timing::enabled())
        .then(|| super::gpu_timing_waits::fence_cookie(this as usize))
        .flatten();
    let f: RefFn = std::mem::transmute(target);
    let refs = f(this);
    if refs == 0 {
        if let Some(cookie) = cookie {
            super::gpu_timing_waits::forget_fence_if(this as usize, cookie);
        }
    }
    refs
}

/// Fence::SetEventOnCompletion (`XCL_TRACE_WAITS`: followed; without an event it blocks, timed)
unsafe extern "system" fn fence_set_event(this: *mut c_void, value: u64, event: *mut c_void) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, u64, *mut c_void) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "SetEventOnCompletion")));
    if !super::wait_trace::enabled() {
        return f(this, value, event);
    }
    if !event.is_null() {
        super::wait_trace::wait(this as usize, value, "CPU event wait");
        return f(this, value, event);
    }
    let t0 = std::time::Instant::now();
    let hr = f(this, value, event);
    super::wait_trace::waited(this as usize, value, "CPU blocking wait", t0.elapsed());
    hr
}

unsafe extern "system" fn q_update_tile_mappings(
    this: *mut c_void, resource: *mut c_void, n: u32, coordinates: *const D3D12_TILED_RESOURCE_COORDINATE,
    regions: *const D3D12_TILE_REGION_SIZE, heap: *mut c_void, ranges: u32, range_flags: *const D3D12_TILE_RANGE_FLAGS,
    offsets: *const u32, counts: *const u32, flags: D3D12_TILE_MAPPING_FLAGS,
) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const D3D12_TILED_RESOURCE_COORDINATE,
        *const D3D12_TILE_REGION_SIZE, *mut c_void, u32, *const D3D12_TILE_RANGE_FLAGS, *const u32, *const u32, D3D12_TILE_MAPPING_FLAGS);
    let this = super::single::host_after_held(this, "UpdateTileMappings");
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "UpdateTileMappings")));
    super::residency::link(resource as usize, heap as usize);
    note_profiled_queue_hole(this, || f(this, resource, n, coordinates, regions, heap, ranges, range_flags, offsets, counts, flags));
}

unsafe extern "system" fn q_copy_tile_mappings(
    this: *mut c_void, dst: *mut c_void, dst_coord: *const D3D12_TILED_RESOURCE_COORDINATE,
    src: *mut c_void, src_coord: *const D3D12_TILED_RESOURCE_COORDINATE, region: *const D3D12_TILE_REGION_SIZE,
    flags: D3D12_TILE_MAPPING_FLAGS,
) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_TILED_RESOURCE_COORDINATE,
        *mut c_void, *const D3D12_TILED_RESOURCE_COORDINATE, *const D3D12_TILE_REGION_SIZE, D3D12_TILE_MAPPING_FLAGS);
    let this = super::single::host_after_held(this, "CopyTileMappings");
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CopyTileMappings")));
    super::residency::copy_links(dst as usize, src as usize);
    note_profiled_queue_hole(this, || f(this, dst, dst_coord, src, src_coord, region, flags));
}

unsafe fn note_profiled_queue_hole(queue: *mut c_void, f: impl FnOnce()) {
    let token = super::gpu_timing::queue_stamp(queue)
        .map(|(epoch, id)| (id, super::gpu_timing_waits::submitted(id, epoch)));
    f();
    if let Some((id, seq)) = token {
        super::gpu_timing_waits::returned(id, seq);
    }
}

pub(super) unsafe fn note_profiled_gpu_wait(queue: *mut c_void, fence: usize, value: u64) {
    if let Some((epoch, id)) = super::gpu_timing::wait(queue) {
        super::gpu_timing_waits::wait(id, epoch, fence, value);
    }
}

unsafe fn note_profiled_gpu_signal(queue: *mut c_void, fence: usize, value: u64) {
    if let Some((epoch, id)) = super::gpu_timing::queue_stamp(queue) {
        super::gpu_timing_waits::signal(Some(id), epoch, fence, value);
    }
}

unsafe extern "system" fn q_timed_wait(this: *mut c_void, fence: *mut c_void, value: u64) -> HRESULT {
    let (target, adopted) = original_or_native(this, 15);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(target);
    let hr = f(this, fence, value);
    if adopted && hr >= 0 {
        note_profiled_gpu_wait(this, fence as usize, value);
    }
    hr
}

unsafe extern "system" fn q_timed_signal(this: *mut c_void, fence: *mut c_void, value: u64) -> HRESULT {
    let (target, adopted) = original_or_native(this, 14);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(target);
    let hr = f(this, fence, value);
    if adopted && hr >= 0 {
        note_profiled_gpu_signal(this, fence as usize, value);
    }
    hr
}

unsafe extern "system" fn q_execute_command_lists(this: *mut c_void, n: u32, lists: *const *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteCommandLists")));
    if n == 0 || lists.is_null() {
        f(this, n, lists);
        stop_if_device_removed();
        return;
    }
    // console-shaped lists run as the PC lists they were translated into (split title lists expand
    // into all of their recorded PC segments in exact chronological order)
    let mut mapped: Vec<*mut c_void> = Vec::with_capacity(n as usize);
    // (the calls put off to this submit are recorded first, their waits ending together)
    let deadline = super::pso_async::submit_deadline();
    for &l in std::slice::from_raw_parts(lists, n as usize) {
        if let Some(pc) = super::console_list::pc_list(l) {
            mapped.push(pc);
        } else {
            super::list_x::record_deferred(l, deadline);
            if !super::title_list::push_segments(l, &mut mapped) {
                mapped.push(l);
            }
        }
    }
    super::shader_code::submitted(std::slice::from_raw_parts(lists, n as usize).iter().chain(mapped.iter()).map(|&l| l as usize));
    // (`single`) the PC queue of every title queue gets them, when the queue's earlier work
    // and what these lists wait for have been handed over
    if super::single::host_of(this).is_some() {
        return super::single::execute(this as usize, mapped.iter().map(|&l| l as usize).collect());
    }
    execute_mapped(this, &mapped);
}

/// Hand the PC lists `mapped` to the queue `this`, with all that goes with a submission.
pub(super) unsafe fn execute_mapped(this: *mut c_void, mapped: &[*mut c_void]) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteCommandLists")));
    // textures in title memory the CPU wrote since: uploaded again first
    super::title_textures::refresh(this);
    super::list_x::note_queue_for_lists(this as usize, mapped);
    let submission = super::residency::before_submit(this, mapped);
    super::list_x::execute_with_waits_on(this, mapped, |chunk| {
        super::markers::submitted(this as usize, chunk);
        super::bind::indirect_world_submit(this as usize, chunk);
        super::page_mappings::queue::submit(this, chunk, || super::gpu_timing::submit(this, chunk, || f(this, chunk.len() as u32, chunk.as_ptr())));
        super::present::submit_plane_writes(this, chunk);
        super::list_x::signal_write64_fences(this, chunk);
        super::list_x::readback_sparse_writes(this, chunk);
        super::list_x::cpu_writes_after(this, chunk);
        super::image_writeback::submitted(this, chunk);
    });
    super::residency::after_submit(this, submission);
    // the next uploads into textures kept in step start from the states these lists leave
    super::title_textures::submitted(mapped);
    super::present::note_execute(this, mapped);
    stop_if_device_removed();
}

/// Wait (up to 10 s) until the GPU has finished the work submitted to the queues console code
/// used (before the process ends on purpose).
pub fn drain_gpu() {
    use crate::gpu::fence::{self, FenceError};

    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    unsafe {
        // a fence per queue (they finish in any order)
        let mut fences = Vec::new();
        for q in super::present::known_queues() {
            let qr = q as *mut c_void;
            let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qr) else { continue };
            let submit = || {
                let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|e| fence::api_error(dev, "CreateFence", e))?;
                q.Signal(&f, 1).map_err(|e| fence::api_error(dev, "Signal", e))?;
                Ok(f)
            };
            match submit() {
                Ok(f) => fences.push(f),
                Err(e) => {
                    crate::xlog_error!("D3D12 queue drain: {e}");
                    if matches!(e, FenceError::Removed(_)) {
                        return;
                    }
                }
            }
        }
        let start = std::time::Instant::now();
        loop {
            let mut pending = false;
            for f in &fences {
                match fence::completed_value(f, dev) {
                    Ok(value) => pending |= value < 1,
                    Err(e) => {
                        crate::xlog_error!("D3D12 queue drain: {e}");
                        return;
                    }
                }
            }
            if !pending {
                return;
            }
            if start.elapsed().as_secs() >= 10 {
                crate::xlog_warn!("D3D12 queue drain: GPU fences still pending after 10 s");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

/// The removal's reason by name with what DXGI documents for it, and this process's video memory
/// as DXGI counts it at the removal, for the local and the non-local segment group (each: usage,
/// budget, reservation, what may be reserved; bytes).
fn removal_report(code: u32, memory: Option<[[u64; 4]; 2]>) -> String {
    let (name, documented) = match code {
        0x887a_0005 => ("DXGI_ERROR_DEVICE_REMOVED", "the card was removed or its driver replaced; drivers and the system also give it when they take a device away without a GPU timeout"),
        0x887a_0006 => ("DXGI_ERROR_DEVICE_HUNG", "the device failed on commands it was sent; a GPU timeout gives this code"),
        0x887a_0007 => ("DXGI_ERROR_DEVICE_RESET", "the device failed on a badly formed command and was reset"),
        0x887a_0020 => ("DXGI_ERROR_DRIVER_INTERNAL_ERROR", "the driver met a problem of its own and removed the device"),
        0x887a_0001 => ("DXGI_ERROR_INVALID_CALL", "a call with wrong arguments removed the device"),
        0x8007_000e => ("E_OUTOFMEMORY", "the runtime or the driver had no memory left"),
        _ => ("a code XCL has no name for", "nothing known"),
    };
    let group = |g: [u64; 4]| format!("{} of {} MiB budget ({} MiB reserved, {} MiB may be reserved)", g[0] >> 20, g[1] >> 20, g[2] >> 20, g[3] >> 20);
    let memory = match memory {
        Some([local, non_local]) => format!("local {}, non-local {}", group(local), group(non_local)),
        None => "not read (no adapter answered)".to_string(),
    };
    format!("D3D12 device removed: reason {name} ({code:#x}: {documented}); this process's video memory by DXGI at the removal: {memory}")
}

/// If the device was removed: the reason, DRED's page fault (address, the allocations around
/// it) and the operations each command list got to – into the log, once.
pub fn log_device_removed() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    let Err(e) = (unsafe { dev.GetDeviceRemovedReason() }) else { return };
    if DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    crate::xlog_error!("D3D12 device removed: {:#x} {}", e.code().0 as u32, e.message());
    let memory = video_memory_info().map(|groups| groups.map(|g| [g.CurrentUsage, g.Budget, g.CurrentReservation, g.AvailableForReservation]));
    crate::xlog_error!("{}", removal_report(e.code().0 as u32, memory));
    super::markers::snapshot("the device was removed");
    super::submit_watch::log_removal();
    super::shader_code::log_removal();
    super::bind::indirect_args_at_removal();
    super::list_x::put_off_at_removal();
    unsafe {
        let Ok(dred) = dev.cast::<ID3D12DeviceRemovedExtendedData>() else {
            crate::xlog_error!("DRED: the device has no ID3D12DeviceRemovedExtendedData");
            return;
        };
        if let Ok(d2) = dred.cast::<ID3D12DeviceRemovedExtendedData2>() {
            let state = d2.GetDeviceState();
            let what = match state {
                D3D12_DRED_DEVICE_STATE_HUNG => "hung",
                D3D12_DRED_DEVICE_STATE_FAULT => "fault",
                D3D12_DRED_DEVICE_STATE_PAGEFAULT => "page fault",
                _ => "unknown (DRED was not on, or the removal was not this device's work)",
            };
            crate::xlog_error!("DRED: device state {} – {what}", state.0);
        }
        match dred.GetPageFaultAllocationOutput() {
            Err(e) => crate::xlog_error!("DRED: no page-fault output ({e})"),
            Ok(pf) => {
                let names = |mut n: *const D3D12_DRED_ALLOCATION_NODE| {
                    let mut v = Vec::new();
                    while !n.is_null() && v.len() < 8 {
                        let a = &*n;
                        let name = if a.ObjectNameA.is_null() { String::new() } else { std::ffi::CStr::from_ptr(a.ObjectNameA as *const std::ffi::c_char).to_string_lossy().into_owned() };
                        v.push(format!("type {} {}", a.AllocationType.0, name));
                        n = a.pNext;
                    }
                    v.join(", ")
                };
                let va = pf.PageFaultVA;
                let window = uma_window_vas().iter().position(|&w| va >= w && va < w + 0x1_0000_0000).map(|k| format!(" = window {k} + {:#x}", va - uma_window_vas()[k])).unwrap_or_default();
                // (no fault: DRED gives address 0 and names no allocation)
                if va == 0 && pf.pHeadExistingAllocationNode.is_null() && pf.pHeadRecentFreedAllocationNode.is_null() {
                    crate::xlog_error!("DRED: no page fault (fault address 0, no allocation named): the GPU did not stop on a bad address – a hang the watchdog ended, as a shader that never finishes is");
                } else {
                    crate::xlog_error!("DRED page fault at GPU VA {va:#x}{window}; allocations there: [{}]; recently freed: [{}]", names(pf.pHeadExistingAllocationNode), names(pf.pHeadRecentFreedAllocationNode));
                }
            }
        }
        let bc = match dred.GetAutoBreadcrumbsOutput() {
            Ok(bc) => bc,
            Err(e) => {
                crate::xlog_error!("DRED: no breadcrumbs output ({e})");
                return;
            }
        };
        let mut n = bc.pHeadAutoBreadcrumbNode;
        let (mut lists, mut unfinished) = (0, 0);
        while !n.is_null() && lists < 4096 {
            let node = &*n;
            let count = node.BreadcrumbCount as usize;
            let done = if node.pLastBreadcrumbValue.is_null() { 0 } else { *node.pLastBreadcrumbValue as usize };
            lists += 1;
            if done < count {
                unfinished += 1;
                if unfinished <= 64 {
                    let ops: Vec<String> = (done.saturating_sub(3)..(done + 3).min(count)).map(|i| format!("{}{}", if i == done { ">" } else { "" }, (*node.pCommandHistory.add(i)).0)).collect();
                    crate::xlog_error!("DRED: list {:p} on queue {:p}: {done} of {count} operations done, around: {}", node.pCommandList.as_ref().map_or(std::ptr::null_mut(), |l| l.as_raw()), node.pCommandQueue.as_ref().map_or(std::ptr::null_mut(), |q| q.as_raw()), ops.join(" "));
                }
            }
            n = node.pNext;
        }
        let why = if lists == 0 {
            " – none: DRED keeps only lists that were submitted and had not finished, so either no command list was in flight (the GPU stopped in work outside one: a tile mapping, a queue wait) or breadcrumbs were off for this device; the submit watch above names what XCL had in flight"
        } else if unfinished == 0 {
            " – every recorded list finished: the hang is in work without breadcrumbs"
        } else {
            ""
        };
        crate::xlog_error!("DRED: {lists} command lists with breadcrumbs, {unfinished} of them unfinished{why}");
    }
}

/// The device's removal (a GPU hang or reset) stops the title at once (`stop_if_device_removed`),
/// whichever of its calls meets it first: a fence's event for `u64::MAX` is set when the device is
/// removed. A title that goes on gets failing D3D12 calls, and code that does not check them
/// crashes on them (FSR4 read through a null model after its CreateModel failed), which hides
/// the hang behind a crash somewhere else.
unsafe fn watch_device_removal() {
    let raw = device_raw();
    let Some(dev) = ID3D12Device::from_raw_borrowed(&raw) else { return };
    let Ok(fence) = create_residency_fence(dev) else { return };
    let Ok(event) = windows::Win32::System::Threading::CreateEventW(None, false, false, None) else { return };
    if fence.SetEventOnCompletion(u64::MAX, event).is_err() {
        return;
    }
    let (fence, event) = (fence.into_raw() as usize, event.0 as usize);
    let _ = std::thread::Builder::new().name("xcl-device-removed".into()).spawn(move || {
        let _kept = fence;
        windows::Win32::System::Threading::WaitForSingleObject(windows::Win32::Foundation::HANDLE(event as *mut c_void), windows::Win32::System::Threading::INFINITE);
        stop_if_device_removed();
    });
}

/// The GPU hung or the device was removed: the title is stopped at once (a title that goes on
/// submitting work to a hung device can keep the driver from recovering), with the reason in
/// the log and the report.
pub fn stop_if_device_removed() {
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    if let Err(e) = unsafe { dev.GetDeviceRemovedReason() } {
        log_device_removed();
        let msg = format!("the D3D12 device was removed ({:#x}: {}) – the title is stopped", e.code().0 as u32, e.message());
        crate::xlog_error!("{msg}");
        crate::control::exit(e.code().0 as u32, "device_removed", &msg);
        crate::report::write_final(Some(msg));
        crate::log::flush();
        unsafe {
            windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), e.code().0 as u32);
        }
    }
}

#[cfg(test)]
mod removal_report_tests {
    use super::removal_report;

    #[test]
    fn the_reason_is_named_and_both_segment_groups_are_given_in_mib() {
        let line = removal_report(0x887a0005, Some([[5718 << 20, 9283 << 20, 0, 4897 << 20], [7376 << 20, 15570 << 20, 1 << 20, 7913 << 20]]));
        assert!(line.contains("reason DXGI_ERROR_DEVICE_REMOVED (0x887a0005:"), "{line}");
        assert!(line.contains("local 5718 of 9283 MiB budget (0 MiB reserved, 4897 MiB may be reserved), non-local 7376 of 15570 MiB budget (1 MiB reserved, 7913 MiB may be reserved)"), "{line}");
        assert!(removal_report(0x887a0006, None).contains("reason DXGI_ERROR_DEVICE_HUNG (0x887a0006:"));
        assert!(removal_report(0x887a0006, None).ends_with("not read (no adapter answered)"));
        assert!(removal_report(0x887a0020, None).contains("DXGI_ERROR_DRIVER_INTERNAL_ERROR"));
        assert!(removal_report(0x8007000e, None).contains("reason E_OUTOFMEMORY (0x8007000e:"));
        assert!(removal_report(0x80004005, None).contains("reason a code XCL has no name for (0x80004005:"));
    }
}

#[cfg(test)]
mod pipeline_state_console_tests {
    use super::*;

    #[test]
    fn console_pipeline_blob_query_gives_no_blob_and_clears_the_out_pointer() {
        let mut out = 0x1234usize as *mut c_void;
        let hr = unsafe { pso_console_blob_x(std::ptr::null_mut(), 0, 0, &mut out) };
        assert_eq!(hr, E_NOTIMPL_HR);
        assert!(out.is_null());
        assert!(console_slots(Kind::PipelineState).contains(&(11, pso_console_blob_x as usize)));
    }

    #[test]
    fn a_compute_pipeline_keeps_why_it_has_no_translation_until_it_is_released() {
        let pso = 0x7654_3210usize;
        note_untranslated(pso, "CS 0123456789abcdef: unsupported op".into());
        assert_eq!(why_untranslated(pso).as_deref(), Some("CS 0123456789abcdef: unsupported op"));
        forget_pso(pso);
        assert_eq!(why_untranslated(pso), None);
    }

    #[test]
    fn the_compile_budget_uses_the_calibrated_curve_and_keeps_the_unmeasured_bound() {
        use super::super::pso_sched::{peak_at, SMALL_PEAK_PER_BYTE};
        assert_eq!(peak(6 << 20), 6 * 3380 << 20);
        assert_eq!(peak_at(GIANT_DXIL, SMALL_PEAK_PER_BYTE), 911 << 20);
        assert_eq!(peak_at(1000, SMALL_PEAK_PER_BYTE), 128 << 20);
    }
}

#[cfg(test)]
mod rect_list_pso_tests {
    use super::*;

    const REFUSED: HRESULT = 0x8007_0057u32 as i32;

    #[test]
    fn translated_vertex_stream_replaces_empty_shader_slots() {
        let mut guest = StreamOut::default();
        for ty in [1, 2, 3, 4, 5, 6, 24, 25] {
            guest.raw(ty, &[0; 16], true);
        }
        guest.raw(14, &RECT_LIST_TYPE.to_le_bytes(), false);
        guest.raw(19, &[0; 16], true);
        guest.raw(9, &0x55aa_55aau32.to_le_bytes(), false);
        let subs = stream_subobjects(&guest.0).unwrap();
        let gs = &[42u8; 4];
        for code in [Some(gs.as_slice()), None] {
            let host = vertex_stream_out(&subs, std::ptr::null_mut(), &[1], Some(&[2]), code, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            let host = stream_subobjects(&host.0).unwrap();
            let tags = host.iter().map(|(ty, _)| *ty).collect::<Vec<_>>();
            if let Some(code) = code {
                assert_eq!(tags, [0, 1, 2, 5, 14, 9]);
                assert_eq!(host[3].1[8..16], code.len().to_le_bytes());
            } else {
                assert_eq!(tags, [0, 1, 2, 14, 9]);
            }
            assert_eq!(host.last().unwrap().1, 0x55aa_55aau32.to_le_bytes());
        }
    }

    fn with_gs(gs: &[u8]) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC { GS: D3D12_SHADER_BYTECODE { pShaderBytecode: gs.as_ptr() as _, BytecodeLength: gs.len() }, ..Default::default() }
    }

    /// A pipeline the driver refuses with the rect list GS is made again without it, once; one
    /// it accepts keeps the GS, and a refused pipeline without a rect list GS is not retried.
    #[test]
    fn a_refused_rect_list_gs_is_retried_without_it() {
        let gs = [0u8; 4];
        let mut copy = with_gs(&gs);
        let mut calls = Vec::new();
        let hr = retry_without_rect_gs(&mut copy, true, (0, 0), |c| {
            calls.push(c.GS.pShaderBytecode.is_null());
            if c.GS.pShaderBytecode.is_null() {
                0
            } else {
                REFUSED
            }
        });
        assert_eq!((hr, calls), (0, vec![false, true]));
        let mut copy = with_gs(&gs);
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, true, (0, 0), |_| {
                n += 1;
                0
            }),
            0
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, false, (0, 0), |_| {
                n += 1;
                REFUSED
            }),
            REFUSED
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());
    }

    /// `has_rect_list_gs`: a pipeline made with the rect list GS, not one made again without it
    /// (or never given it), and not once its address is forgotten.
    #[test]
    fn pipelines_with_the_rect_list_gs_are_known_until_forgotten() {
        let gs = [0u8; 4];
        let (kept, dropped, none) = (0x7E57_1000usize, 0x7E57_2000usize, 0x7E57_3000usize);
        let mut copy = with_gs(&gs);
        note_rect_list_gs(kept, &copy, true);
        assert!(retry_without_rect_gs(&mut copy, true, (0, 0), |c| if c.GS.pShaderBytecode.is_null() { 0 } else { REFUSED }) == 0);
        note_rect_list_gs(dropped, &copy, true);
        note_rect_list_gs(none, &with_gs(&gs), false);
        assert!(has_rect_list_gs(kept) && !has_rect_list_gs(dropped) && !has_rect_list_gs(none) && !has_rect_list_gs(0));
        forget_pso(kept);
        assert!(!has_rect_list_gs(kept));
    }

    /// An untranslated rect list pipeline from a stream gets the PC's topology type (TRIANGLE),
    /// as at 956a476; other types, and a direct CreateGraphicsPipelineState, are left as they are.
    #[test]
    fn untranslated_stream_rect_lists_get_the_pcs_topology_type() {
        let at = |t: i32| D3D12_GRAPHICS_PIPELINE_STATE_DESC { PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE(t), ..Default::default() };
        let desc = |g: &D3D12_GRAPHICS_PIPELINE_STATE_DESC| g as *const _ as *const c_void;
        let (rect, tri) = (at(RECT_LIST_TYPE), at(3));
        unsafe {
            let g = untranslated_stream_desc(desc(&rect), true).expect("a stream's rect list");
            assert_eq!(g.PrimitiveTopologyType, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            assert!(untranslated_stream_desc(desc(&tri), true).is_none());
            assert!(untranslated_stream_desc(desc(&rect), false).is_none());
        }
    }
}

#[cfg(test)]
mod this_first_tests {
    use super::*;

    /// `this_first`: either argument order, and a return buffer whose first word is a stale copy
    /// of the object's vtable pointer (7a033c8: the retail game's C-way call wrote its result
    /// over the device). Synthetic objects; no GPU and no title.
    #[test]
    fn a_return_buffer_with_a_stale_vtable_copy_is_not_the_object() {
        let pc = [0usize; 32];
        let ext = [pc.as_ptr() as usize, 0];
        let vtbl = ext.as_ptr() as usize + 8;
        EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(vtbl);
        let plain = 0x1234_5678usize;
        let p = &plain as *const usize as *mut c_void;
        // an object off the stack (as a real device is), and a stack buffer holding a stale copy
        // of its vtable pointer - the retail game's case
        let obj = Box::into_raw(Box::new([vtbl])) as *mut c_void;
        let buf = [vtbl];
        let b = buf.as_ptr() as *mut c_void;
        unsafe {
            assert!(is_adopted(obj) && is_adopted(b), "both look adopted");
            // one adopted: that one is the object, either position
            assert!(this_first(obj, p));
            assert!(!this_first(p, obj));
            // neither adopted: unchanged (the C way is assumed)
            assert!(!this_first(p, p) && !this_first(std::ptr::null_mut(), std::ptr::null_mut()));
            // one null argument: the other is taken as `this`, either position
            assert!(this_first(obj, std::ptr::null_mut()));
            assert!(!this_first(std::ptr::null_mut(), obj));
            // both adopted, one on this thread's stack: the stack one is the buffer, so the heap
            // one is `this` in either order - what 7a033c8 fixed
            assert!(this_first(obj, b));
            assert!(!this_first(b, obj));
            // both on the stack, neither the device: `a` (the C++ way)
            let stale = [vtbl];
            let s = stale.as_ptr() as *mut c_void;
            assert!(this_first(b, s) && this_first(s, b));
            drop(Box::from_raw(obj as *mut [usize; 1]));
        }
        if let Some(g) = EXTS.lock().unwrap().as_mut() {
            g.remove(&vtbl);
        }
    }
}

#[cfg(test)]
mod cache_report_tests {
    use super::*;

    #[test]
    fn reports_without_dxil_cache_statistics_still_deserialize() {
        let mut old = serde_json::to_value(D3d12Stats::default()).unwrap();
        old.as_object_mut().unwrap().remove("dxil_cache");
        let decoded: D3d12Stats = serde_json::from_value(old).unwrap();
        assert_eq!(
            decoded.dxil_cache,
            crate::gpu::recomp::dxc::CacheStats::default()
        );
    }
}

#[cfg(test)]
mod device_abi_tests {
    use super::*;

    #[test]
    fn fence_slot_12_exposes_live_completed_value() {
        use std::time::{Duration, Instant};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let method = console_slots(Kind::Fence).into_iter().find_map(|(slot, f)| (slot == 12).then_some(f))
            .expect("console completed-value address at fence slot 12");
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(7, D3D12_FENCE_FLAG_NONE).unwrap();
            let address: unsafe extern "system" fn(*mut c_void) -> *const AtomicU64 = std::mem::transmute(method);
            let word = address(fence.as_raw()).as_ref().expect("readable fence value");
            assert_eq!(word.load(Ordering::Acquire), 7);
            assert_eq!(address(fence.as_raw()), word as *const _);
            assert_ne!(address(gate.as_raw()), word as *const _);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let signal = queue.Signal(&fence, 21);
            let pending = word.load(Ordering::Acquire);
            gate.Signal(1).unwrap();
            signal.unwrap();
            assert_eq!(pending, 7, "an enqueued signal is not a completed signal");
            let deadline = Instant::now() + Duration::from_secs(2);
            while word.load(Ordering::Acquire) == 7 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(word.load(Ordering::Acquire), 21);
            assert_eq!(fence.GetCompletedValue(), 21);
        }
    }

    #[test]
    fn slot_133_mirrors_a_write_combined_title_pool_buffer() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        if !crate::hle::d3d12x::sparse::windows_ready() {
            return crate::gpu::recomp::runner::skip_test(
                "no 4 GiB sparse memory window: this adapter cannot reserve one, and this proves what a reservation does");
        }
        let raw = device_raw();
        let mut desc = [0u8; 0x38];
        let bytes = 0x70800;
        for (offset, value) in [(0, 1u32), (0x10, bytes), (0x18, 1), (0x1c, 0x10001), (0x24, 1), (0x30, 0x20004)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let allocation = crate::hle::kernelx::XMemVirtualAlloc(std::ptr::null(), 0x100000,
                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, 0x283, 0x2404);
            assert!(!allocation.is_null());
            let title = allocation.byte_add(0x20000);
            title.cast::<u8>().write(0x82);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            let hr = super::super::device_x::create_placed_resource_layout_x(raw, title as u64, desc.as_ptr(),
                D3D12_BARRIER_LAYOUT_UNDEFINED, std::ptr::null(), &iid, &mut resource);
            if hr < 0 {
                crate::hle::kernelx::VirtualFree(allocation, 0, MEM_RELEASE);
            }
            assert_eq!(hr, S_OK, "write-combined title pool placement: {hr:#x}");
            let resource = ID3D12Resource::from_raw(resource);
            let (mirror, size) = console_buffer(resource.as_raw() as usize).expect("UMA-backed buffer");
            assert_eq!(size, bytes as usize);
            assert_eq!(resource.GetDesc().Flags, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            assert_eq!((mirror as *const u8).read(), 0x82);
            let view = buffer_view(resource.as_raw() as usize, Default::default(), 0, bytes / 4, 0, true);
            assert_eq!(view[2], bytes);

            // a second buffer over the same memory: the same memory on the console, the same
            // video memory here (it was refused)
            let mut second = std::ptr::null_mut();
            assert_eq!(super::super::device_x::create_placed_resource_layout_x(raw, title as u64, desc.as_ptr(),
                D3D12_BARRIER_LAYOUT_UNDEFINED, std::ptr::null(), &iid, &mut second), S_OK);
            let second = ID3D12Resource::from_raw(second);
            assert_eq!(console_buffer(second.as_raw() as usize), Some((mirror, bytes as usize)));
            assert_eq!(res_gpu_va_x(second.as_raw()), mirror as u64);

            let queue: ID3D12CommandQueue = ID3D12Device::from_raw_borrowed(&raw).unwrap()
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let offset = 0x70000;
            title.byte_add(offset).cast::<u8>().write(0xc3);
            super::super::title_textures::refresh(queue.as_raw());
            let mirrored = (mirror + offset) as *mut u8;
            assert_eq!(mirrored.read(), 0xc3);
            mirrored.write(0);
            super::super::sparse::readback_after_execute(&queue, &[(mirror + offset, 1)]);
            assert_eq!(mirrored.read(), 0xc3, "the updated byte reached video memory");

            // the first goes: the mirror and its video memory stay, in step, for the second
            drop(resource);
            title.byte_add(offset).cast::<u8>().write(0x3c);
            super::super::title_textures::refresh(queue.as_raw());
            assert_eq!(mirrored.read(), 0x3c);
            mirrored.write(0);
            super::super::sparse::readback_after_execute(&queue, &[(mirror + offset, 1)]);
            assert_eq!(mirrored.read(), 0x3c, "the mirror outlives the buffer it was made for");
            let mut mapped = std::ptr::null_mut();
            second.Map(0, Some(&D3D12_RANGE { Begin: offset, End: offset + 1 }), Some(&mut mapped)).unwrap();
            assert_eq!(mapped, title);
            assert_eq!(title.byte_add(offset).cast::<u8>().read(), 0x3c);
            second.Unmap(0, None);

            drop(second);
            assert_eq!(crate::hle::kernelx::VirtualFree(allocation, 0, MEM_RELEASE), 1);
            crate::hle::kernelx::in_released_memory(allocation as usize, 0x100000, || assert!(!crate::hle::kernelx::in_graphics_reservation(title as usize)));
        }
    }

    #[test]
    fn slot_133_mirrors_a_remapped_buffer_in_a_host_reservation() {
        use crate::{gpu::uma, hle::kernelx};
        use super::super::{device_x, sparse, title_textures};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, PAGE_WRITECOMBINE};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        if !crate::hle::d3d12x::sparse::windows_ready() {
            return crate::gpu::recomp::runner::skip_test(
                "no 4 GiB sparse memory window: this adapter cannot reserve one, and this proves what a reservation does");
        }
        let raw = device_raw();
        let bytes = 50 << 20;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 1u32), (0x10, bytes), (0x18, 1), (0x1c, 0x10001), (0x24, 1), (0x30, 0x20004)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let allocation = kernelx::XMemVirtualAlloc(std::ptr::null(), uma::MAX_WINDOWS * uma::WINDOW + uma::GRAIN,
                MEM_RESERVE | 0x2040_0000, 0x23, 0x2404);
            assert!(!allocation.is_null());
            assert!(!uma::get().unwrap().contains(allocation as usize));
            let title = allocation.byte_add(2 * uma::GRAIN);
            let mut count = bytes as usize / uma::GRAIN;
            let mut pages = vec![0u64; count];
            assert_eq!(kernelx::XMemAllocatePhysicalPages(0x380, &mut count, pages.as_mut_ptr()), 1);
            assert_eq!(count, pages.len());
            assert_eq!(kernelx::VirtualAlloc(title, bytes as usize, MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE), title);
            assert_eq!(kernelx::XMemMapPhysicalPages(title, count, pages.as_mut_ptr()), title);
            title.cast::<u8>().write(0x82);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            let hr = device_x::create_placed_resource_layout_x(raw, title as u64, desc.as_ptr(),
                D3D12_BARRIER_LAYOUT_UNDEFINED, std::ptr::null(), &iid, &mut resource);
            assert_eq!(hr, S_OK, "placement in the host fallback reservation: {hr:#x}");
            let resource = ID3D12Resource::from_raw(resource);
            let (mirror, size) = console_buffer(resource.as_raw() as usize).unwrap();
            assert_eq!(size, bytes as usize);
            assert_eq!((mirror as *const u8).read(), 0x82);
            assert_eq!(res_gpu_va_x(resource.as_raw()), mirror as u64);

            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let last = bytes as usize - 1;
            title.cast::<u8>().add(last).write(0x5a);
            title_textures::refresh(queue.as_raw());
            let end = (mirror + last) as *mut u8;
            end.write(0);
            sparse::readback_after_execute(&queue, &[(mirror + last, 1)]);
            assert_eq!(end.read(), 0x5a);

            drop(resource);
            assert_eq!(kernelx::VirtualFree(allocation, 0, MEM_RELEASE), 1);
            kernelx::in_released_memory(allocation as usize, uma::MAX_WINDOWS * uma::WINDOW + uma::GRAIN, || assert!(!kernelx::in_graphics_reservation(title as usize)));
            assert_eq!(kernelx::XMemFreePhysicalPages(count, pages.as_mut_ptr()), 1);
        }
    }

    #[test]
    fn slot_133_mirrors_subpage_buffers_without_copying_neighbors() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        if !crate::hle::d3d12x::sparse::windows_ready() {
            return crate::gpu::recomp::runner::skip_test(
                "no 4 GiB sparse memory window: this adapter cannot reserve one, and this proves what a reservation does");
        }
        let raw = device_raw();
        unsafe {
            let allocation = crate::hle::kernelx::XMemVirtualAlloc(std::ptr::null(), 0x4000,
                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, 0x283, 0x2404);
            assert!(!allocation.is_null());
            let small_at = allocation.byte_add(0xc20);
            let large_at = small_at.byte_add(0x10);
            let resource = |at: *mut c_void, bytes: u32| {
                let mut desc = [0u8; 0x38];
                for (offset, value) in [(0, 1u32), (0x10, bytes), (0x18, 1), (0x1c, 0x10001), (0x24, 1), (0x30, 0x20004)] {
                    desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
                let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
                let mut resource = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_layout_x(raw, at as u64, desc.as_ptr(),
                    D3D12_BARRIER_LAYOUT_UNDEFINED, std::ptr::null(), &iid, &mut resource);
                assert_eq!(hr, S_OK, "subpage title pool placement: {hr:#x}");
                ID3D12Resource::from_raw(resource)
            };
            small_at.byte_sub(1).cast::<u8>().write(0x55);
            large_at.cast::<u8>().write(0x77);
            let small = resource(small_at, 0x10);
            let large = resource(large_at, 0x2000);
            let small_mirror = console_buffer(small.as_raw() as usize).unwrap().0 as *mut u8;
            let large_mirror = console_buffer(large.as_raw() as usize).unwrap().0 as *mut u8;
            let queue: ID3D12CommandQueue = ID3D12Device::from_raw_borrowed(&raw).unwrap()
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            super::super::title_textures::refresh(queue.as_raw());
            small_at.byte_add(7).cast::<u8>().write(0x31);
            large_at.cast::<u8>().write(0x62);
            large_at.byte_add(0x400).cast::<u8>().write(0x91);
            large_at.byte_add(0x1100).cast::<u8>().write(0xb4);
            super::super::title_textures::refresh(queue.as_raw());
            assert_eq!(small_mirror.add(7).read(), 0x31);
            assert_eq!(small_mirror.add(0x10).read(), 0);
            assert_eq!(large_mirror.read(), 0x62);
            large_mirror.add(0x400).write(0);
            large_mirror.add(0x1100).write(0);
            super::super::sparse::readback_after_execute(&queue, &[(large_mirror as usize + 0x400, 1), (large_mirror as usize + 0x1100, 1)]);
            assert_eq!((large_mirror.add(0x400).read(), large_mirror.add(0x1100).read()), (0x91, 0xb4));
            small_at.byte_add(7).cast::<u8>().write(0xe1);
            let mut mapped = std::ptr::null_mut();
            small.Map(0, Some(&D3D12_RANGE { Begin: 7, End: 8 }), Some(&mut mapped)).unwrap();
            assert_eq!(mapped, small_at);
            assert_eq!(mapped.byte_add(7).cast::<u8>().read(), 0xe1);
            small.Unmap(0, None);
            drop((small, large));
            assert_eq!(crate::hle::kernelx::VirtualFree(allocation, 0, MEM_RELEASE), 1);
        }
    }

    #[test]
    fn placed_color_texture_exposes_the_guest_address() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE};
        use super::super::{device_x, title_textures, vram};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        if !crate::hle::d3d12x::sparse::windows_ready() {
            return crate::gpu::recomp::runner::skip_test(
                "no 4 GiB sparse memory window: this adapter cannot reserve one, and this proves what a reservation does");
        }
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            for (format, bpe, layout, flags, offset) in [
                (DXGI_FORMAT_R32G32_FLOAT, 8, 0x100, 0x10_0005, 0x2000),
                (DXGI_FORMAT_R32_FLOAT, 4, 0, 0x30_0004, 0),
                (DXGI_FORMAT_R32G32_FLOAT, 8, 0, 0x30_0005, 0),
            ] {
                let mut desc = [0u8; 0x38];
                for (offset, value) in [(0, 3u32), (0x10, 256), (0x18, 256), (0x1c, 0x10001),
                    (0x20, format.0 as u32), (0x24, 1), (0x2c, layout), (0x30, flags)] {
                    desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
                let tex = crate::hle::xg::Texture::new(&desc).unwrap();
                let span = tex.size() as usize;
                let allocation = crate::hle::kernelx::XMemVirtualAlloc(std::ptr::null(), span + offset,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, 0x283, 0x2404);
                assert!(!allocation.is_null());
                let guest = allocation.byte_add(offset);
                assert_eq!(guest as u64 % tex.alignment(), 0, "valid console placement");
                let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
                let mut out = std::ptr::null_mut();
                let hr = device_x::create_placed_resource_layout_x(raw, guest as u64, desc.as_ptr(),
                    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, std::ptr::null(), &iid, &mut out);
                assert_eq!(hr, S_OK);
                let resource = ID3D12Resource::from_raw(out);
                assert_eq!(vram::is_placed_texture(out as usize), offset == 0);
                assert_eq!(resource.GetGPUVirtualAddress(), 0, "PC textures have no buffer GPU VA");
                assert_eq!(title_textures::address(out as usize), None);
                let address = res_gpu_va_x(out);
                assert_eq!(address, guest as u64, "the guest gets the storage it placed the texture in");
                let tiled = std::slice::from_raw_parts_mut(address as *mut u8, span);
                for phase in [21, 43] {
                    let linear: Vec<u8> = (0usize..256 * 256 * bpe)
                        .map(|i| (((i ^ (i >> 9)) * phase) & 127) as u8).collect();
                    tex.texels().unwrap().chain.tile_level(tiled, 0, 0, &linear);
                    title_textures::refresh(queue.as_raw());
                    let read = title_textures::read_gpu_subresource(dev, &resource, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).unwrap();
                    assert!(read == linear, "CPU-written texels must reach the native texture, format {format:?}, phase {phase}");
                }
                drop(resource);
                assert_eq!(title_textures::address(out as usize), None);
                assert_eq!(crate::hle::kernelx::VirtualFree(allocation, 0, MEM_RELEASE), 1);
            }
        }
    }

    #[test]
    #[ignore = "requires D3D12 depth UAV support, such as Agility 1.722 Preview with app-local WARP"]
    fn slot_133_creates_a_six_slice_depth_uav_texture() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_TYPELESS};

        type Console = unsafe extern "system" fn(*mut c_void, u64, *const u8, D3D12_BARRIER_LAYOUT, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 133).then_some(f))
            .expect("console placed resource adapter at slot 133");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 768), (0x18, 768), (0x1c, 0x10006), (0x20, DXGI_FORMAT_R16_TYPELESS.0 as u32), (0x24, 1), (0x30, 0x8030_0006)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D16_UNORM,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
        };
        unsafe {
            let console: Console = std::mem::transmute(method);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            let hr = console(device_raw(), 0x4_2800_0000, desc.as_ptr(), D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                (&clear as *const D3D12_CLEAR_VALUE).cast(), &iid, &mut resource);
            assert_eq!(hr, S_OK, "six-slice depth UAV texture: {hr:#x}");
            assert!(!resource.is_null());
            let resource = ID3D12Resource::from_raw(resource);
            let actual = resource.GetDesc();
            assert_eq!((actual.Width, actual.Height, actual.DepthOrArraySize, actual.MipLevels, actual.SampleDesc.Count), (768, 768, 6, 1, 1));
            assert_eq!((actual.Format, actual.Flags), (DXGI_FORMAT_R16_TYPELESS, flags));
        }
    }

    #[test]
    fn slot_133_creates_an_hdr_target_in_unordered_access_layout() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

        type Console = unsafe extern "system" fn(*mut c_void, u64, *const u8, D3D12_BARRIER_LAYOUT, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 133).then_some(f))
            .expect("console placed resource adapter at slot 133");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
        };
        let flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 2560), (0x18, 1440), (0x1c, 0x10001), (0x20, clear.Format.0 as u32), (0x24, 1), (0x30, 0x30_0000 | flags.0 as u32)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let console: Console = std::mem::transmute(method);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            assert_eq!(console(device_raw(), 0x4_1000_0000, desc.as_ptr(), D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                (&clear as *const D3D12_CLEAR_VALUE).cast(), &iid, &mut resource), S_OK);
            assert!(!resource.is_null());
            let resource = ID3D12Resource::from_raw(resource);
            let actual = resource.GetDesc();
            assert_eq!((actual.Width, actual.Height, actual.Format, actual.Flags), (2560, 1440, clear.Format, flags));
        }
    }

    #[test]
    fn slot_133_creates_an_rgba16f_volume_with_uav_layout() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

        type Console = unsafe extern "system" fn(*mut c_void, u64, *const u8, D3D12_BARRIER_LAYOUT, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 133).then_some(f))
            .expect("console placed resource adapter at slot 133");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 4u32), (8, 0x1000), (0x10, 33), (0x18, 33), (0x1c, 0x10021), (0x20, format.0 as u32), (0x24, 1), (0x2c, 0x105), (0x30, 0x30_0000 | flags.0 as u32)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let console: Console = std::mem::transmute(method);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let dev_ptr = device_raw();
            let mut resource = std::ptr::null_mut();
            assert_eq!(console(dev_ptr, 0x4_2001_1000, desc.as_ptr(), D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                std::ptr::null(), &iid, &mut resource), S_OK);
            assert!(!resource.is_null());
            let resource = ID3D12Resource::from_raw(resource);
            let actual = resource.GetDesc();
            assert_eq!(actual.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE3D);
            assert_eq!((actual.Width, actual.Height, actual.DepthOrArraySize, actual.MipLevels), (33, 33, 33, 1));
            assert_eq!((actual.Format, actual.Flags), (format, flags));
            let dev = ID3D12Device::from_raw_borrowed(&dev_ptr).unwrap();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0;
            let mut row_bytes = 0;
            dev.GetCopyableFootprints(&actual, 0, 1, 0, Some(&mut footprint), Some(&mut rows), Some(&mut row_bytes), None);
            assert_eq!((footprint.Footprint.Depth, rows, row_bytes), (33, 33, 33 * 8));
        }
    }

    #[test]
    fn slot_25_sizes_a_console_d16s8_texture() {
        type Console = unsafe extern "system" fn(*mut [u64; 2], *mut c_void, u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 25).then_some(f))
            .expect("console allocation info at slot 25");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 2560), (0x18, 1440), (0x1c, 0x10001), (0x20, 118), (0x24, 1), (0x30, 0x8030_0006)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let mut info = [0; 2];
        unsafe {
            let console: Console = std::mem::transmute(method);
            assert_eq!(console(&mut info, device_raw(), 1, 1, &desc), &raw mut info);
        }
        let htile = 3072 * 2048 / 16;
        let depth = 2560u64 * 1536 * 2;
        let stencil = 2560u64 * 1536;
        assert_eq!(info, [htile + depth + stencil, 0x10000]);
        let texture = crate::hle::xg::Texture::new(&desc).unwrap();
        let (d, d_at) = texture.plane_surface(1).unwrap();
        let (s, s_at) = texture.plane_surface(2).unwrap();
        assert_eq!((d.bpe, s.bpe), (2, 1));
        assert_eq!((d_at, s_at), (htile, htile + depth));
    }

    #[test]
    fn slot_133_places_an_uninitialized_uint_texture() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_UINT;
        use super::super::{device_x, title_textures, vram};
        use crate::gpu::texture::{self, Subresource};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let format = DXGI_FORMAT_R32G32B32A32_UINT;
        let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 1), (0x18, 1), (0x1c, 0x10001),
            (0x20, format.0 as u32), (0x24, 1), (0x2c, 0x102), (0x30, 0x30_0000 | flags.0 as u32)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
                std::mem::size_of_val(&options) as u32).is_err() || !options.EnhancedBarriersSupported.as_bool() {
                return;
            }
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            let hr = device_x::create_placed_resource_layout_x(raw, 0x4_2900_0000, desc.as_ptr(),
                D3D12_BARRIER_LAYOUT_UNDEFINED, std::ptr::null(), &iid, &mut resource);
            assert_eq!(hr, S_OK, "uninitialized placed texture: {hr:#x}");
            assert!(!resource.is_null());
            assert!(vram::is_placed_texture(resource as usize));
            assert_eq!(title_textures::address(resource as usize), None);
            let resource_ptr = resource;
            let resource = ID3D12Resource::from_raw(resource_ptr);
            let actual = resource.GetDesc();
            assert_eq!((actual.Dimension, actual.Width, actual.Height, actual.Format, actual.Flags),
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1, format, flags));
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList7 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = D3D12_TEXTURE_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_NONE, SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_NO_ACCESS, AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
                LayoutBefore: D3D12_BARRIER_LAYOUT_UNDEFINED, LayoutAfter: D3D12_BARRIER_LAYOUT_COMMON,
                pResource: std::mem::transmute_copy(&resource),
                Subresources: D3D12_BARRIER_SUBRESOURCE_RANGE { IndexOrFirstMipLevel: u32::MAX, ..Default::default() },
                Flags: D3D12_TEXTURE_BARRIER_FLAG_DISCARD,
            };
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_TEXTURE, NumBarriers: 1, Anonymous: D3D12_BARRIER_GROUP_0 { pTextureBarriers: &barrier }
            }]);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            texture::wait(dev, &queue).unwrap();
            let common = D3D12_RESOURCE_STATE_COMMON;
            let texel = Subresource { rows: 1, row_bytes: 16, depth: 1, data: (0..16).map(|i| (i * 13 + 9) as u8).collect() };
            texture::upload(dev, &queue, &resource, std::slice::from_ref(&texel), common, common).unwrap();
            assert_eq!(title_textures::read_gpu_subresource(dev, &resource, 0, common).unwrap(), texel.data);
            drop(resource);
            assert!(!vram::is_placed_texture(resource_ptr as usize));
        }
    }

    #[test]
    fn slot_133_places_an_uninitialized_depth_texture() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32G8X24_TYPELESS};
        use super::super::{device_x, title_textures, vram};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let format = DXGI_FORMAT_R32G8X24_TYPELESS;
        let flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 128), (0x18, 128), (0x1c, 0x10001),
            (0x20, format.0 as u32), (0x24, 1), (0x30, 0x80_30_0000 | flags.0 as u32)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.625, Stencil: 0x5a } },
        };
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
                std::mem::size_of_val(&options) as u32).is_err() || !options.EnhancedBarriersSupported.as_bool() {
                return;
            }
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let mut resource = std::ptr::null_mut();
            let hr = device_x::create_placed_resource_layout_x(raw, 0x4_2a00_0000, desc.as_ptr(),
                D3D12_BARRIER_LAYOUT_UNDEFINED, (&clear as *const D3D12_CLEAR_VALUE).cast(), &iid, &mut resource);
            assert_eq!(hr, S_OK, "uninitialized placed depth texture: {hr:#x}");
            assert!(!resource.is_null());
            assert!(vram::is_placed_texture(resource as usize));
            let resource = ID3D12Resource::from_raw(resource);
            let actual = resource.GetDesc();
            assert_eq!((actual.Dimension, actual.Width, actual.Height, actual.Format, actual.Flags),
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 128, 128, format, flags));
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList7 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default()
            }).unwrap();
            let handle = heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateDepthStencilView(&resource, Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: clear.Format, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 } },
                ..Default::default()
            }), handle);
            let transition = |barrier: &D3D12_TEXTURE_BARRIER| list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_TEXTURE, NumBarriers: 1, Anonymous: D3D12_BARRIER_GROUP_0 { pTextureBarriers: barrier }
            }]);
            let all = D3D12_BARRIER_SUBRESOURCE_RANGE { IndexOrFirstMipLevel: u32::MAX, ..Default::default() };
            transition(&D3D12_TEXTURE_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_NONE, SyncAfter: D3D12_BARRIER_SYNC_DEPTH_STENCIL,
                AccessBefore: D3D12_BARRIER_ACCESS_NO_ACCESS, AccessAfter: D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
                LayoutBefore: D3D12_BARRIER_LAYOUT_UNDEFINED, LayoutAfter: D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
                pResource: std::mem::transmute_copy(&resource), Subresources: all, Flags: D3D12_TEXTURE_BARRIER_FLAG_DISCARD,
            });
            list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.625, 0x5a, None);
            transition(&D3D12_TEXTURE_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_DEPTH_STENCIL, SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE, AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
                LayoutBefore: D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, LayoutAfter: D3D12_BARRIER_LAYOUT_COMMON,
                pResource: std::mem::transmute_copy(&resource), Subresources: all, Flags: D3D12_TEXTURE_BARRIER_FLAG_NONE,
            });
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            crate::gpu::texture::wait(dev, &queue).unwrap();
            let common = D3D12_RESOURCE_STATE_COMMON;
            let depth = title_textures::read_gpu_subresource(dev, &resource, 0, common).unwrap();
            let stencil = title_textures::read_gpu_subresource(dev, &resource, 1, common).unwrap();
            assert_eq!(depth.len(), 128 * 128 * 4);
            assert!(depth.chunks_exact(4).all(|v| v == 0.625f32.to_le_bytes()));
            assert_eq!(stencil.len(), 128 * 128);
            assert!(stencil.iter().all(|&v| v == 0x5a));
        }
    }

    #[test]
    fn slot_133_creates_a_1d_r32_texture_with_uav_layout() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;

        type Console = unsafe extern "system" fn(*mut c_void, u64, *const u8, D3D12_BARRIER_LAYOUT, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 133).then_some(f))
            .expect("console placed resource adapter at slot 133");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let format = DXGI_FORMAT_R32_FLOAT;
        let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 2u32), (0x10, 4096), (0x18, 1), (0x1c, 0x10001), (0x20, format.0 as u32), (0x24, 1), (0x2c, 0x100), (0x30, 0x30_0000 | flags.0 as u32)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let console: Console = std::mem::transmute(method);
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let dev = device_raw();
            let mut resource = std::ptr::null_mut();
            assert_eq!(console(dev, 0x4_2400_0000, desc.as_ptr(), D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                std::ptr::null(), &iid, &mut resource), S_OK);
            assert!(!resource.is_null());
            let resource = ID3D12Resource::from_raw(resource);
            let actual = resource.GetDesc();
            assert_eq!(actual.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D);
            assert_eq!((actual.Width, actual.Height, actual.DepthOrArraySize, actual.MipLevels), (4096, 1, 1, 1));
            assert_eq!((actual.Format, actual.Flags), (format, flags));
            let dev = ID3D12Device::from_raw_borrowed(&dev).unwrap();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0;
            let mut row_bytes = 0;
            dev.GetCopyableFootprints(&actual, 0, 1, 0, Some(&mut footprint), Some(&mut rows), Some(&mut row_bytes), None);
            assert_eq!((footprint.Footprint.Depth, rows, row_bytes), (1, 1, 4096 * 4));
        }
    }

    #[test]
    fn slot_133_creates_msaa_color_and_depth_targets() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM,
        };

        type Console = unsafe extern "system" fn(*mut c_void, u64, *const u8, D3D12_BARRIER_LAYOUT, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 133).then_some(f))
            .expect("console placed resource adapter at slot 133");
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        unsafe {
            let console: Console = std::mem::transmute(method);
            let raw = device_raw();
            let iid = GUID::from_u128(ID3D12Resource::IID.to_u128());
            let color = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: [1.0; 4] },
            };
            let depth = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.0, Stencil: 0 } },
            };
            for (va, format, flags, clear) in [
                (0x4_0000_0000, color.Format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, color),
                (0x4_0040_0000, DXGI_FORMAT_R32G8X24_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, depth),
            ] {
                let mut desc = [0u8; 0x38];
                for (offset, value) in [(0, 3u32), (0x10, 1), (0x18, 1), (0x1c, 0x10001), (0x20, format.0 as u32), (0x24, 4), (0x30, flags.0 as u32)] {
                    desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
                let mut resource = std::ptr::null_mut();
                assert_eq!(console(raw, va, desc.as_ptr(), D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, (&clear as *const D3D12_CLEAR_VALUE).cast(), &iid, &mut resource),
                    S_OK, "creating 4x MSAA texture with format {}", format.0);
                assert!(!resource.is_null());
                let resource = ID3D12Resource::from_raw(resource);
                let actual = resource.GetDesc();
                assert_eq!((actual.Width, actual.Height, actual.SampleDesc.Count), (1, 1, 4));
                assert_eq!((actual.Format, actual.Flags), (format, flags));
            }
        }
    }

    #[test]
    fn slot_108_returns_msaa_allocation_info_through_the_result_pointer() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 108).then_some(f))
            .expect("console allocation information adapter at slot 108");
        type Console = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, *const [u8; 0x48], *mut D3D12_RESOURCE_ALLOCATION_INFO1) -> *mut [u64; 2];
        // slot 108 takes D3D12_RESOURCE_DESC1 records (0x48), each a resource desc then its trailing
        // mip levels and sample desc
        let desc1 = |d: &[u8; 0x38]| {
            let mut b = [0u8; 0x48];
            b[..0x38].copy_from_slice(d);
            b
        };
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let mut desc = [0u8; 0x38];
        for (offset, value) in [(0, 3u32), (0x10, 1), (0x18, 1), (0x1c, 0x10001), (0x20, 28), (0x24, 4), (0x30, 1)] {
            desc[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device4::from_raw_borrowed(&raw).expect("D3D12 device4");
            let desktop = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 4, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut native_detail = D3D12_RESOURCE_ALLOCATION_INFO1::default();
            let native = dev.GetResourceAllocationInfo1(1, 1, &desktop, Some(&mut native_detail));
            assert_ne!(native.SizeInBytes, u64::MAX);
            let console: Console = std::mem::transmute(method);
            let mut detail = D3D12_RESOURCE_ALLOCATION_INFO1::default();
            let mut ret = [u64::MAX; 2];
            let out = &mut ret as *mut [u64; 2];
            assert_eq!(console(out.cast(), raw, 1, 1, &desc1(&desc), &mut detail), out);
            assert_eq!(ret, [native.SizeInBytes, native.Alignment]);
            assert_eq!(detail, native_detail);

            let mut volume = desc;
            for (offset, value) in [(0, 4u32), (8, 0x1000), (0x24, 1), (0x2c, 0x105), (0x30, 0)] {
                volume[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            let descs = [desc1(&volume), desc1(&desc), desc1(&volume)];
            let mut details = [D3D12_RESOURCE_ALLOCATION_INFO1::default(); 3];
            assert_eq!(console(out.cast(), raw, 1, 3, descs.as_ptr(), details.as_mut_ptr()), out);
            let ceil = |size: u64| size.next_multiple_of(native.Alignment);
            let msaa_offset = ceil(0x1000);
            let last_offset = msaa_offset + native.SizeInBytes;
            assert_eq!(ret, [ceil(last_offset + 0x1000), native.Alignment]);
            assert_eq!(details, [
                D3D12_RESOURCE_ALLOCATION_INFO1 { Offset: 0, Alignment: 0x1000, SizeInBytes: 0x1000 },
                D3D12_RESOURCE_ALLOCATION_INFO1 { Offset: msaa_offset, ..native_detail },
                D3D12_RESOURCE_ALLOCATION_INFO1 { Offset: last_offset, Alignment: 0x1000, SizeInBytes: 0x1000 },
            ]);
            assert_eq!(console(out.cast(), raw, 1, 1, &desc1(&volume), std::ptr::null_mut()), out);
            assert_eq!(ret, [0x1000, 0x1000]);
        }
    }

    #[test]
    fn a_console_pixel_shader_without_a_console_vertex_shader_uses_the_stream_standin() {
        use crate::gpu::xbox::tests::{container, dxil_part, fssm, CODE};

        unsafe extern "system" fn check_native(this: *mut c_void, desc: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
            let [size, base] = *desc.cast::<[usize; 2]>();
            let subs = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size)).unwrap();
            let shader = subs.iter().find(|(ty, _)| *ty == 2).and_then(|(_, p)| payload_shader(p)).unwrap();
            let calls = &mut *this.cast::<[usize; 2]>();
            let kind = usize::from(crate::gpu::recomp::pso::is_console(shader));
            calls[kind] += 1;
            S_OK
        }

        let code = dxil_part(0, &[fssm(&CODE, &[(5, 1)])]);
        let ps = container(&[(b"DXIL", &code)]);
        assert!(crate::gpu::recomp::pso::is_console(&ps));
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        let desc = [stream.0.len(), stream.0.as_ptr() as usize];
        let mut calls = [0usize; 2];
        let hr = unsafe {
            create_pipeline_stream(calls.as_mut_ptr().cast(), check_native, desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), std::ptr::null_mut())
        };
        assert_eq!(hr, S_OK);
        assert_eq!(calls, [1, 0]);
    }

    #[test]
    fn vertex_pixel_stream_translates_before_calling_d3d12() {
        use crate::gpu::xbox::{is_xbox, tests::{container, dxil_part, fssm, CODE}};
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

        thread_local! {
            static SAW_CONSOLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
            static SAW_TRANSLATED_STATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        unsafe extern "system" fn check_native(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
            let [size, base] = *desc.cast::<[usize; 2]>();
            let Some(subs) = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size)) else { return E_INVALIDARG };
            let stages: Vec<_> = subs.iter().filter(|(ty, _)| matches!(ty, 1 | 2)).filter_map(|(_, p)| payload_shader(p)).collect();
            if stages.iter().any(|s| is_xbox(s)) {
                SAW_CONSOLE.set(true);
                return E_INVALIDARG;
            }
            let translated = stages.len() == 2
                && stages.iter().all(|s| crate::gpu::shader::container_parts(s).is_some_and(|p| p.iter().any(|p| p.fourcc == "DXIL")))
                && subs.iter().any(|(ty, p)| *ty == 21 && p[4..8] == 1u32.to_le_bytes())
                && !subs.iter().any(|(ty, _)| *ty == 19);
            if !translated {
                return E_INVALIDARG;
            }
            SAW_TRANSLATED_STATE.set(true);
            let native: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
            native(this, desc, riid, ppv)
        }

        fn shader(kind: u32, hw: u8, api: u8, desktop: &[u8]) -> Vec<u8> {
            let parts = crate::gpu::shader::container_parts(desktop).expect("desktop shader");
            let sig = parts.iter().find(|p| p.fourcc == "OSGN").expect("output signature");
            let code = dxil_part(kind, &[fssm(&CODE, &[(hw, api)])]);
            container(&[(b"OSGN", &desktop[sig.offset..sig.offset + sig.size]), (b"DXIL", &code)])
        }

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let vs = shader(1, 4, 0, &dummy().vs);
        let ps = shader(0, 5, 1, &dummy().ps);
        let mut stream = StreamOut::default();
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &0usize);
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs);
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        for ty in [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS] {
            stream.put(ty, &D3D12_SHADER_BYTECODE::default());
        }
        let mut depth = [0u8; 56];
        depth[4..8].copy_from_slice(&1u32.to_le_bytes());
        depth[8..12].copy_from_slice(&8u32.to_le_bytes());
        for at in [20, 36] {
            depth[at..at + 16].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0]);
        }
        stream.raw(21, &depth, false);
        let mut rt = D3D12_RT_FORMAT_ARRAY::default();
        rt.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt.NumRenderTargets = 1;
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE::default());
        let desc = [stream.0.len(), stream.0.as_ptr() as usize];
        unsafe {
            let mut pso = std::ptr::null_mut();
            let hr = create_pipeline_stream(device_raw(), check_native, desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut pso);
            assert!(!SAW_CONSOLE.get(), "original Xbox shader reached the native boundary");
            assert!(SAW_TRANSLATED_STATE.get(), "native stream lost the translated shaders or graphics state");
            assert_eq!(hr, S_OK, "native stream creation failed: {hr:#x}");
            let pso = ID3D12PipelineState::from_raw(pso);
            assert!(is_translated_graphics(pso.as_raw() as usize));
            assert!(pso_root_signature(pso.as_raw() as usize).is_some());
        }
    }

    #[test]
    fn mesh_stream_accepts_xbox_mesh_in_subobject_24() {
        use crate::gpu::xbox::tests::{container, dxil_part, fssm};
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        // A console mesh program becomes a host mesh shader through the DX compiler: without it the
        // build misses and creation answers E_INVALIDARG (it goes to the background in a launcher).
        if let Err(e) = crate::gpu::recomp::dxc::location() {
            return crate::gpu::recomp::runner::skip_test(&format!("no DX compiler: {e}"));
        }
        let mut psv = [0u8; 56];
        psv[..4].copy_from_slice(&48u32.to_le_bytes());
        psv[16..18].copy_from_slice(&3u16.to_le_bytes());
        psv[18..20].copy_from_slice(&1u16.to_le_bytes());
        psv[28] = 13;
        psv[31] = 2;
        psv[40..44].copy_from_slice(&32u32.to_le_bytes());
        let dxil = dxil_part(13, &[fssm(&0xbf81_0000u32.to_le_bytes(), &[(7, 7)])]);
        let ms = container(&[(b"PSV0", &psv), (b"DXIL", &dxil)]);
        let (_, ps) = dummy_mesh().expect("DXIL fixtures");
        let mut rt = D3D12_RT_FORMAT_ARRAY::default();
        rt.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt.NumRenderTargets = 1;
        // a host mesh shader where the adapter runs console mesh programs as one, else two passes
        let host = unsafe {
            ID3D12Device::from_raw_borrowed(&device_raw()).is_some_and(|d| crate::gpu::recomp::pso::mesh_native_split(&d).is_some())
        };
        for tag in [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS] {
            let mut stream = StreamOut::default();
            stream.shader(tag, &ms);
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            unsafe {
                let mut raw = std::ptr::null_mut();
                let hr = dev_create_pipeline_state(device_raw(), desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut raw);
                assert_eq!(hr, S_OK, "Xbox MS in subobject {}: {hr:#x}", tag.0);
                let pso = ID3D12PipelineState::from_raw(raw);
                let p = pso.as_raw() as usize;
                assert_eq!((is_native_mesh(p), two_pass_of(p).is_some()), (host, !host));
            }
        }
    }

    /// `XCL_LAZY_MESH_PSO`: the title gets a stand-in for a console mesh pipeline, its first use
    /// builds the real one, and the stand-in's release takes it along.
    #[test]
    fn a_lazy_mesh_stream_is_a_stand_in_built_at_its_first_use() {
        use super::super::pso_async;
        use crate::gpu::xbox::tests::{container, dxil_part, fssm};
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let mut psv = [0u8; 56];
        psv[..4].copy_from_slice(&48u32.to_le_bytes());
        psv[16..18].copy_from_slice(&3u16.to_le_bytes());
        psv[18..20].copy_from_slice(&1u16.to_le_bytes());
        psv[28] = 13;
        psv[31] = 2;
        psv[40..44].copy_from_slice(&32u32.to_le_bytes());
        let dxil = dxil_part(13, &[fssm(&0xbf81_0000u32.to_le_bytes(), &[(7, 7)])]);
        let ms = container(&[(b"PSV0", &psv), (b"DXIL", &dxil)]);
        let (_, ps) = dummy_mesh().expect("DXIL fixtures");
        let mut rt = D3D12_RT_FORMAT_ARRAY::default();
        rt.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt.NumRenderTargets = 1;
        unsafe {
            if !ID3D12Device::from_raw_borrowed(&device_raw()).is_some_and(|d| crate::gpu::recomp::pso::mesh_native_split(&d).is_some()) {
                // no mesh shaders: the async build this waits for cannot finish
                return crate::gpu::recomp::runner::skip_test("no mesh shaders");
            }
            // (a title's stream has its root signature, and the stand-in is made with it)
            let rs = abi_rs_for(device_raw(), &[]).expect("root signature");
            let mut stream = StreamOut::default();
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
            stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            let mut raw = std::ptr::null_mut();
            pso_async::LAZY_MESH.with(|l| l.set(Some(true)));
            let hr = dev_create_pipeline_state(device_raw(), desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut raw);
            pso_async::LAZY_MESH.with(|l| l.set(None));
            assert_eq!(hr, S_OK, "{hr:#x}");
            let pso = ID3D12PipelineState::from_raw(raw);
            let p = raw as usize;
            assert!(pso_async::is_standin(p), "the title holds a stand-in");
            assert!(!is_native_mesh(p) && !TWO_PASS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&p)));
            let waiting = std::time::Instant::now();
            let real = loop {
                let real = pso_async::resolve_wait(p);
                if real != p {
                    break real;
                }
                assert!(waiting.elapsed() < std::time::Duration::from_secs(120), "the first use did not build the pipeline");
            };
            assert!(is_native_mesh(real) || two_pass_of(p).is_some(), "the stand-in resolves to the mesh pipeline");
            drop(pso);
            assert!(!pso_async::is_standin(p));
            assert!(!is_native_mesh(real) && !TWO_PASS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&real)), "the real pipeline went with its stand-in");
        }
    }

    /// Measurement, not a check: what a driver keeps for each pipeline state, what releasing
    /// them gives back and what making them again from their cached blobs costs. `XCL_PROBE_PSOS`:
    /// how many of each kind (4000); `XCL_PROBE_BLOB_DIR`: where a first run keeps the blobs of the
    /// pipelines with their own programs, which a second run then makes from them in a process
    /// whose driver compiled nothing. Run alone: `gpu_pipeline_memory_probe --ignored --nocapture`.
    #[test]
    #[ignore = "native adapter measurement; run in isolation without the game"]
    fn gpu_pipeline_memory_probe() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        // (private commit, working set) in KiB
        fn memory() -> (f64, f64) {
            use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
            let mut m: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
            m.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            unsafe { K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut m, m.cb) };
            (m.PagefileUsage as f64 / 1024.0, m.WorkingSetSize as f64 / 1024.0)
        }
        /// `disk`: the blobs are kept there by the first run and the pipelines made from them by
        /// a later run (a new process: the driver has compiled nothing in it).
        unsafe fn measure(what: &str, n: usize, disk: Option<&std::path::Path>, create: impl Fn(usize, Option<&[u8]>) -> ID3D12PipelineState) {
            let per = |a: (f64, f64), b: (f64, f64)| ((b.0 - a.0) / n as f64, (b.1 - a.1) / n as f64);
            let kept: Vec<Option<Vec<u8>>> = (0..n).map(|i| disk.and_then(|d| std::fs::read(d.join(format!("{i}.blob"))).ok())).collect();
            let from_disk = kept.iter().flatten().count();
            let start = memory();
            let t = std::time::Instant::now();
            let made: Vec<ID3D12PipelineState> = (0..n).map(|i| create(i, kept[i].as_deref())).collect();
            let first_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            let built = memory();
            let blobs: Vec<Vec<u8>> = made
                .iter()
                .map(|p| p.GetCachedBlob().map_or_else(|_| Vec::new(), |b| std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()))
                .collect();
            let blob_kib = blobs.iter().map(Vec::len).sum::<usize>() as f64 / 1024.0 / n as f64;
            if let Some(d) = disk.filter(|_| from_disk == 0) {
                std::fs::create_dir_all(d).unwrap();
                blobs.iter().enumerate().for_each(|(i, b)| std::fs::write(d.join(format!("{i}.blob")), b).unwrap());
            }
            println!("{what}, {n} pipelines (KiB each: private commit, working set)");
            println!("  made ({from_disk} from blobs on disk): {:+.1?} in {first_ms:.3} ms each; cached blob {blob_kib:.1} KiB each", per(start, built));
            let mut made = Some(made);
            for round in 1..=3 {
                let before = memory();
                drop(made.take());
                std::thread::sleep(std::time::Duration::from_secs(2));
                let released = memory();
                let t = std::time::Instant::now();
                made = Some((0..n).map(|i| create(i, Some(&blobs[i]).filter(|b| !b.is_empty()).map(Vec::as_slice))).collect());
                let again_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
                println!("  round {round}: released {:+.1?}, made again from the blobs {:+.1?} in {again_ms:.3} ms each", per(before, released), per(released, memory()));
            }
            let before = memory();
            drop(made);
            std::thread::sleep(std::time::Duration::from_secs(2));
            let end = memory();
            println!("  released at the end {:+.1?}; start to end {:+.1?} (the probe's copies of the blobs are in it)", per(before, end), per(start, end));
        }
        unsafe {
            let adapter = init().expect("D3D12 device");
            assert!(!adapter.starts_with("WARP"), "native hardware adapter required");
            println!("pipeline memory probe: {adapter}");
            let n = std::env::var("XCL_PROBE_PSOS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000usize);
            let raw = device_raw();
            let rs = abi_rs_for(raw, &[]).expect("root signature");
            let iid = pc_iid(Kind::PipelineState);
            // the driver's own calls: nothing of XCL's between the probe and the driver
            let make = |slot: &str, desc: *const c_void| -> ID3D12PipelineState {
                let f: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, slot)));
                let mut out = std::ptr::null_mut();
                let hr = f(raw, desc, &iid, &mut out);
                assert!(hr >= 0 && !out.is_null(), "{slot}: {hr:#x}");
                ID3D12PipelineState::from_raw(out)
            };
            let cached = |blob: Option<&[u8]>| blob.map_or_else(D3D12_CACHED_PIPELINE_STATE::default, |b| D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr().cast(), CachedBlobSizeInBytes: b.len() });
            // each pipeline its own state, as a title's differ: the depth bias
            let graphics = |i: usize, vs: &[u8], ps: &[u8], blob: Option<&[u8]>| {
                let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
                d.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
                d.VS = bytecode(vs);
                d.PS = bytecode(ps);
                d.BlendState.RenderTarget[0].RenderTargetWriteMask = 15;
                d.SampleMask = u32::MAX;
                d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                d.RasterizerState.DepthBias = i as i32;
                d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                d.NumRenderTargets = 1;
                d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                d.SampleDesc.Count = 1;
                d.CachedPSO = cached(blob);
                make("CreateGraphicsPipelineState", std::ptr::from_ref(&d).cast())
            };
            let d = dummy();
            measure("stand-in graphics pipelines (the dummy VS and PS)", n, None, |i, blob| graphics(i, &d.vs, &d.ps, blob));
            measure("stand-in compute pipelines (the dummy CS, one description)", n, None, |_, blob| {
                let mut c = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
                c.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
                c.CS = bytecode(&d.cs);
                c.CachedPSO = cached(blob);
                make("CreateComputePipelineState", std::ptr::from_ref(&c).cast())
            });
            // programs of a translated shader's size: pixel shaders of `statements` statements
            use crate::gpu::recomp::dxc::compile;
            let vs = compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }", "main", "vs_6_0", &[]).expect("DXC");
            let program = |k: usize, statements: usize| {
                let mut src = String::from("float4 main(float4 p : SV_Position) : SV_Target {\n  float4 a = p * 0.001;\n");
                for j in 0..statements {
                    src += &format!("  a = a * {:.4} + sin(a.yzwx * {:.4}) * cos(a.zwxy + {k}.5);\n", 1.0 + (j + k) as f32 * 1e-4, 1.5 + j as f32 * 0.01);
                }
                compile(&(src + "  return a;\n}\n"), "main", "ps_6_0", &[]).expect("DXC")
            };
            // many pipelines of few programs: what a pipeline's own state costs
            let shared: Vec<Vec<u8>> = (0..8).map(|k| program(k, 400)).collect();
            let what = format!("graphics pipelines of 8 pixel programs ({} B of DXIL each)", shared[0].len());
            measure(&what, n / 2, None, |i, blob| graphics(i, &vs, &shared[i % 8], blob));
            // a program for each pipeline (`XCL_PROBE_PROGRAMS`, 300): what a program costs
            let programs = std::env::var("XCL_PROBE_PROGRAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(300usize);
            let own: Vec<Vec<u8>> = (0..programs).map(|k| program(8 + k, 150)).collect();
            let what = format!("graphics pipelines each of its own pixel program ({} B of DXIL each)", own[0].len());
            let disk = std::env::var_os("XCL_PROBE_BLOB_DIR").map(std::path::PathBuf::from);
            measure(&what, programs, disk.as_deref(), |i, blob| graphics(i, &vs, &own[i], blob));
        }
    }

    #[test]
    fn custom_heap_properties_preserve_both_return_abis() {
        let method = console_slots(Kind::Device).into_iter().find_map(|(slot, f)| (slot == 26).then_some(f))
            .expect("console GetCustomHeapProperties adapter at slot 26");
        let console: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, D3D12_HEAP_TYPE) -> *mut D3D12_HEAP_PROPERTIES =
            unsafe { std::mem::transmute(method) };
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).expect("D3D12 device");
            assert_ne!(*vt(raw).add(26), orig_fn(raw, 26), "console dispatch slot is installed");
            for heap in [D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_TYPE_READBACK] {
                let native = dev.GetCustomHeapProperties(1, heap);
                assert_eq!(native.Type, D3D12_HEAP_TYPE_CUSTOM);
                let mut output = std::mem::MaybeUninit::<D3D12_HEAP_PROPERTIES>::uninit();
                let ret = output.as_mut_ptr();
                assert_eq!(console(ret.cast(), raw, 1, heap), ret);
                assert_eq!(output.assume_init(), native);
                let mut output = std::mem::MaybeUninit::<D3D12_HEAP_PROPERTIES>::uninit();
                let ret = output.as_mut_ptr();
                assert_eq!(console(raw, ret.cast(), 1, heap), ret);
                assert_eq!(output.assume_init(), native);
            }
        }
    }
}

#[cfg(test)]
mod readback_routing_tests {
    use super::*;

    /// Console #53 / #54 ClearUnorderedAccessViewUint / Float translate their console handles
    /// (not the PC #49 / #50 the remap alone gives).
    #[test]
    fn console_clear_uav_slots_translate_handles() {
        use super::super::list_x as l;
        let slots = console_slots(Kind::CommandList);
        assert!(slots.contains(&(53, l::clear_uav_uint_x as usize)));
        assert!(slots.contains(&(54, l::clear_uav_float_x as usize)));
    }


    #[test]
    fn custom_write_back_only_stays_coherent_for_small_copy_dest_buffers() {
        let rb = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let custom_wb = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let copy_dest = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let gen_read = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
        let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let none = D3D12_RESOURCE_FLAG_NONE;

        assert!(is_coherent_readback_buffer(&rb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&rb, copy_dest, none, COHERENT_READBACK_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&rb, copy_dest, none));

        // General CUSTOM+WRITE_BACK buffers route to sparse VRAM, not coherent Window 3.
        assert!(!is_coherent_readback_buffer(&custom_wb, gen_read, none, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, gen_read, none));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, uav, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, copy_dest, uav));

        // Small non-UAV COPY_DEST CUSTOM+WRITE_BACK buffers may use coherent Window 3; larger ones spill.
        assert!(is_coherent_readback_buffer(&custom_wb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, none, COHERENT_CUSTOM_WB_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&custom_wb, copy_dest, none));
    }

    /// Bug #4: `place_console_buffer` coherent fallback must strip console-only resource flags
    /// and state bits before creating the cross-adapter placed resource.
    #[test]
    fn coherent_buffer_sanitizes_console_flags_and_state() {
        let raw_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0x10_0000,
            Width: 0x1_0000,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT(0x101), // console layout
            Flags: D3D12_RESOURCE_FLAGS(0x0130_0000 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0),
            ..Default::default()
        };
        let console_state = 0x0010_0000usize | (D3D12_RESOURCE_STATE_COPY_DEST.0 as usize);

        let (clean, st) = sanitize_coherent_buffer(&raw_desc, console_state, true);
        assert_eq!(clean.Alignment, 0);
        assert_eq!(clean.Layout, D3D12_TEXTURE_LAYOUT_ROW_MAJOR);
        assert_eq!(
            clean.Flags,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        );
        assert_eq!(st, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);

        // Verify against a real D3D12 device: raw console flags/state are rejected, sanitized succeeds.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            if let Ok(dev3) = dev.cast::<ID3D12Device3>() {
                use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
                let mem = VirtualAlloc(std::ptr::null(), 0x1_0000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                assert!(!mem.is_null());
                if let Ok(heap) = dev3.OpenExistingHeapFromAddress::<ID3D12Heap>(mem) {
                    let mut unsanitized = raw_desc;
                    unsanitized.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
                    unsanitized.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    unsanitized.Alignment = 0;
                    let mut r_bad: Option<ID3D12Resource> = None;
                    assert!(
                        dev.CreatePlacedResource(&heap, 0, &unsanitized, D3D12_RESOURCE_STATES(console_state as i32), None, &mut r_bad).is_err(),
                        "PC D3D12 must reject unsanitized console flags/state"
                    );
                    let mut r_ok: Option<ID3D12Resource> = None;
                    dev.CreatePlacedResource(&heap, 0, &clean, D3D12_RESOURCE_STATES(st as i32), None, &mut r_ok)
                        .expect("sanitized coherent buffer must succeed on PC D3D12");
                    assert!(r_ok.is_some());
                    drop(r_ok);
                    drop(heap);
                }
                assert_ne!(VirtualFree(mem, 0, MEM_RELEASE), 0);
            }
        }
    }

    /// Bug #5: `CopyDescriptors` and `CopyDescriptorsSimple` must remap `XBOX_EXTRA_CBV_SRV_UAV`
    /// (`4`) to `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` (`0`) on both console and PC paths.
    #[test]
    fn copy_descriptors_remaps_xbox_extra_cbv_srv_uav_to_pc_enum() {
        use std::cell::RefCell;

        assert_eq!(pc_heap_type(XBOX_EXTRA_CBV_SRV_UAV), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        assert_eq!(pc_heap_type(0), 0);
        assert_eq!(pc_heap_type(1), 1);
        assert_eq!(pc_heap_type(2), 2);
        assert_eq!(pc_heap_type(3), 3);

        thread_local! {
            static SIMPLE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
            static RANGE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_copy_simple(_this: *mut c_void, _n: u32, _dst: u64, _src: u64, ty: u32) {
            SIMPLE_TYPES.with(|v| v.borrow_mut().push(ty));
        }
        #[allow(clippy::too_many_arguments)]
        unsafe extern "system" fn fake_copy_ranges(
            _this: *mut c_void,
            _nd: u32,
            _dsts: *const u64,
            _dsizes: *const u32,
            _ns: u32,
            _srcs: *const u64,
            _ssizes: *const u32,
            ty: u32,
        ) {
            RANGE_TYPES.with(|v| v.borrow_mut().push(ty));
        }

        #[repr(C)]
        struct FakeDev {
            vt: *const usize,
        }

        let mut orig_vt = [0usize; 32];
        orig_vt[23] = fake_copy_ranges as usize;
        orig_vt[24] = fake_copy_simple as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0usize];
        let mut dev = FakeDev {
            // SAFETY: `ext_vt` has 2 entries; `.add(1)` puts `orig_vt` at `ext[-1]`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
        };
        let dev_ptr = &mut dev as *mut FakeDev as *mut c_void;

        SIMPLE_TYPES.with(|v| v.borrow_mut().clear());
        RANGE_TYPES.with(|v| v.borrow_mut().clear());

        unsafe {
            // Non-console PC fallback paths
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);
            let (d, s, sz) = ([0x1000u64], [0x2000u64], [1u32]);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV.0 as u32);

            // Console-slot paths (`copy_console_slots` invoked from both Simple and Ranges)
            let heap_obj = 0x5555_DEADusize;
            if crate::gpu::uma::get().is_some()
                && super::super::console_heap::register(heap_obj, 4, 0x10_0000, 0, 32, false, false, false)
            {
                let start = super::super::console_heap::start(heap_obj).unwrap();
                let dst_h = start + super::super::console_heap::SLOT as u64;
                dev_copy_descriptors_simple_x(dev_ptr, 1, dst_h, start, XBOX_EXTRA_CBV_SRV_UAV);
                let (cd, cs, csz) = ([dst_h], [start], [1u32]);
                dev_copy_descriptors_x(dev_ptr, 1, cd.as_ptr(), csz.as_ptr(), 1, cs.as_ptr(), csz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
                super::super::console_heap::unregister(heap_obj);

                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1, 0, 0]));
            } else {
                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1]));
            }
            RANGE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 2]));
        }
    }

    /// Bug #6: `Kind::DmaList` must install `cl_release` so releasing a DMA command list removes
    /// its entry from `list_x::LISTS`.
    #[test]
    fn dma_list_release_cleans_up_list_state() {
        assert!(
            overrides(Kind::DmaList).iter().any(|&(name, f)| name == "Release" && f == cl_release as usize),
            "Kind::DmaList must override Release with cl_release"
        );

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).expect("copy alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None).expect("copy list");
            let raw = list.into_raw();
            let key = raw as usize;

            // Mirror `device_x::create_list`: adopt as CommandList, track via `list_x::on_create`,
            // then re-adopt as `Kind::DmaList`.
            adopt(raw, Kind::CommandList);
            super::super::list_x::on_create(raw, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COPY);
            readopt(raw, Kind::DmaList);
            assert!(super::super::list_x::has_list_state(key), "DMA list must be tracked in list_x::LISTS");

            // Invoke Release through the adopted vtable slot 2.
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt(raw).add(2));
            assert_eq!(release(raw), 0);
            assert!(!super::super::list_x::has_list_state(key), "releasing DMA list must remove its list_x::LISTS entry");
        }
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    // a fake adopted queue: the extension vtable's word before slot 0 points at the PC table
    unsafe extern "system" fn get_desc(_: *mut c_void, out: usize) -> usize {
        let d = [1u32, 2, 3, 4];
        std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, out as *mut u8, 16);
        out
    }
    unsafe extern "system" fn signal(_: *mut c_void, _: usize, _: usize) -> i32 {
        0
    }
    unsafe extern "system" fn completed_value(_: *mut c_void) -> u64 {
        u64::MAX
    }

    /// Queue #18 before the layout is known writes 8 bytes; after a Signal at #14 (the older
    /// layout) it is GetDesc with all 16.
    #[test]
    fn queue_slot_18_before_the_layout_is_known_writes_8_bytes() {
        let mut pc = [0usize; 32];
        pc[14] = signal as usize;
        pc[18] = get_desc as usize;
        let ext = [pc.as_ptr() as usize, 0];
        let vtbl = ext.as_ptr() as usize + 8;
        let mut obj = [vtbl];
        let this = obj.as_mut_ptr() as *mut c_void;
        // a fence a queue signalled stays in `list_x`'s sync graph, which asks other tests' cycle
        // checks for its completed value (#8): it lives on, everything signalled on it completed
        let mut fence_vtbl = Box::new([0usize; 9]);
        fence_vtbl[8] = completed_value as usize;
        let fence = Box::leak(Box::new([Box::leak(fence_vtbl).as_ptr() as usize])).as_ptr() as usize;
        let (shifted, older): (&'static AtomicBool, &'static AtomicBool) = (Box::leak(Box::default()), Box::leak(Box::default()));
        TEST_QUEUE_FLAGS.with(|f| f.set(Some((shifted, older))));
        unsafe {
            let mut out = [0xAAAA_AAAAu32; 4];
            let r = queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(r, out.as_ptr() as usize);
            assert_eq!(out, [1, 2, 0xAAAA_AAAA, 0xAAAA_AAAA], "only the first 8 bytes");
            FENCES.write().unwrap().push(fence);
            queue_dispatch(14, this, fence, 1, 0, 0);
            FENCES.write().unwrap().retain(|&f| f != fence);
            assert!(older.load(Ordering::Acquire) && !shifted.load(Ordering::Acquire));
            let mut out = [0xAAAA_AAAAu32; 4];
            queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(out, [1, 2, 3, 4], "older layout: the whole description");
        }
        TEST_QUEUE_FLAGS.with(|f| f.set(None));
    }
}

#[cfg(test)]
mod srv_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_UNORM_SRGB;

    #[test]
    fn a_cube_view_becomes_the_cube_array_of_one_cube() {
        let cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_BC1_UNORM_SRGB,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: 0x1688,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 2, MipLevels: 3, ResourceMinLODClamp: 1.5 },
            },
        };
        let v = cube_as_cube_array(&cube).expect("a cube view is rewritten");
        assert_eq!(v.ViewDimension, D3D12_SRV_DIMENSION_TEXTURECUBEARRAY);
        assert_eq!((v.Format, v.Shader4ComponentMapping), (DXGI_FORMAT_BC1_UNORM_SRGB, 0x1688), "format and mapping kept");
        let a = unsafe { v.Anonymous.TextureCubeArray };
        assert_eq!((a.MostDetailedMip, a.MipLevels, a.First2DArrayFace, a.NumCubes), (2, 3, 0, 1));
        assert_eq!(a.ResourceMinLODClamp, 1.5);

        // a cube array and every other view are forwarded as the title made them
        let mut array = cube;
        array.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        array.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { NumCubes: 2, ..Default::default() };
        assert!(cube_as_cube_array(&array).is_none());
        let mut d2 = cube;
        d2.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        assert!(cube_as_cube_array(&d2).is_none());
    }
}

#[cfg(test)]
mod console_format_query_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    #[test]
    fn committed_bc_address_reserve_covers_the_console_layout() {
        let mut desc = [0u8; 0x38];
        desc[0..4].copy_from_slice(&D3D12_RESOURCE_DIMENSION_TEXTURE2D.0.to_le_bytes());
        desc[0x10..0x18].copy_from_slice(&128u64.to_le_bytes());
        desc[0x18..0x1C].copy_from_slice(&16384u32.to_le_bytes());
        desc[0x1C..0x1E].copy_from_slice(&128u16.to_le_bytes());
        desc[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        desc[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        desc[0x2C..0x30].copy_from_slice(&0x109u32.to_le_bytes());
        for format in [DXGI_FORMAT_BC1_TYPELESS, DXGI_FORMAT_BC1_UNORM] {
            desc[0x20..0x24].copy_from_slice(&format.0.to_le_bytes());
            assert_eq!(console_bc_footprint(&desc).and_then(texture_address_reserve_size), Some((512 << 20) + TEXTURE_ADDRESS_PAGE));
        }
        desc[0x20..0x24].copy_from_slice(&DXGI_FORMAT_R8G8B8A8_UNORM.0.to_le_bytes());
        assert_eq!(console_bc_footprint(&desc), None);
        assert_eq!(texture_address_reserve_size(1 << 30), None);
    }

    #[test]
    fn console_formats_are_asked_as_the_pcs() {
        for (console, pc) in [(116u32, 10u32), (117, 10), (118, 45), (119, 46), (120, 47)] {
            let mut d = [0u32; 3];
            d[0] = console;
            let back = unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) };
            assert_eq!((back, d[0]), (Some(console), pc));
        }
        // a PC format, or a feature without one: untouched
        let mut d = [45u32, 0, 0];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) }, None);
        let mut d = [118u32; 16];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_D3D12_OPTIONS.0, d.as_mut_ptr() as *mut u8, 64) }, None);
        assert_eq!(d[0], 118);
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_INFO.0, std::ptr::null_mut(), 8) }, None);
    }

    /// A hardware device (None: no D3D12 GPU here – skipped).
    fn device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.ok()?;
        dev
    }

    /// CheckFeatureSupport of the PC device as `dev_check_feature_support` reaches it (vtable #13).
    fn pc_check(dev: &ID3D12Device) -> CheckFeatureSupportFn {
        unsafe { std::mem::transmute(*(*(dev.as_raw() as *const *const usize)).add(13)) }
    }

    /// GPU: the queries a title makes of D16_UNORM_S8_UINT (118) and its family succeed with the
    /// PC format's answers, and the caller's format field is the console's again. Asked as is, the
    /// PC has no such format.
    #[test]
    fn gpu_console_depth_formats_are_supported() {
        let Some(dev) = device() else { return };
        let (this, f) = (dev.as_raw(), pc_check(&dev));
        // the gap: the PC's own answer (unless another test adopted a device, and the vtable it
        // shares with this one already holds `dev_check_feature_support`)
        if f as usize != dev_check_feature_support as usize {
            let mut raw = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(118), ..Default::default() };
            let hr = unsafe { f(this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut raw as *mut _ as *mut u8, size_of_val(&raw) as u32) };
            assert!(hr < 0 || raw.Support1.0 == 0, "the PC knows format 118 ({hr:#x}, {:#x})", raw.Support1.0);
        }

        for (format, need) in [
            (118, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),
            (119, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (120, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (116, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
            (117, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
        ] {
            let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(format), ..Default::default() };
            let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) };
            assert!(hr >= 0, "FORMAT_SUPPORT {format}: {hr:#x}");
            assert_eq!(s.Format.0, format as i32, "the caller's format");
            assert!(s.Support1.0 & need.0 != 0, "FORMAT_SUPPORT {format}: {:#x}", s.Support1.0);
        }

        let mut ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(118), SampleCount: 4, ..Default::default() };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut ms as *mut _ as *mut u8, size_of_val(&ms) as u32) };
        assert!(hr >= 0 && ms.NumQualityLevels >= 1, "MULTISAMPLE_QUALITY_LEVELS 118 ×4: {hr:#x} {}", ms.NumQualityLevels);
        assert_eq!((ms.Format.0, ms.SampleCount), (118, 4));

        let mut info = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(118), PlaneCount: 0 };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_INFO.0, &mut info as *mut _ as *mut u8, size_of_val(&info) as u32) };
        assert!(hr >= 0, "FORMAT_INFO 118: {hr:#x}");
        assert_eq!((info.Format.0, info.PlaneCount), (118, 2));

        // a PC format is asked as it is
        let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() };
        assert!(unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) } >= 0);
        assert_eq!(s.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    /// GPU: allocation info of the console's 116 – 120 textures is a real size – the console's
    /// (XG), and the PC's for the resource made of the desc (`pc_resource_desc`), both sane.
    #[test]
    fn gpu_console_format_allocation_info_is_sane() {
        let Some(dev) = device() else { return };
        // (bytes a pixel at least, as XG lays them out: 119 alone is a plain 2-byte format, 120 alone
        // the stencil plane)
        for (format, flags, bytes) in [(118u32, 2u32, 3u64), (118, 0, 3), (119, 0, 2), (120, 0, 1), (116, 1, 4), (117, 1, 4)] {
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&3u32.to_le_bytes());
            x[0x10..0x18].copy_from_slice(&1920u64.to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&1080u32.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
            let t = crate::hle::xg::Texture::new(&x).unwrap_or_else(|hr| panic!("XG {format}: {hr:#x}"));
            let (size, align) = (t.size(), t.alignment());
            assert!(size >= 1920 * 1080 * bytes && size < 1 << 30 && align.is_power_of_two(), "XG {format}: {size:#x} {align:#x}");
            let d: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&x).as_ptr() as *const _) };
            let pc = unsafe { dev.GetResourceAllocationInfo(0, &[d]) };
            assert!(pc.SizeInBytes != u64::MAX && pc.SizeInBytes >= 1920 * 1080 * 4 && pc.Alignment > 0, "PC {format} as {}: {:#x}", d.Format.0, pc.SizeInBytes);
        }
    }
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    use super::super::pso_manifest::Kind;

    /// A pipeline kept for the warm-up (`OwnedDesc::record`) is made again as the title described
    /// it: its shaders, its input layout and the rest of its description, its root signature none.
    #[test]
    fn a_kept_pipeline_is_made_again_as_the_title_described_it() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let (vs, ps) = (b"vertex shader bytes".to_vec(), b"pixel shader".to_vec());
        let name = c"TEXCOORD";
        let elems = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(name.as_ptr() as *const u8),
            SemanticIndex: 1,
            Format: DXGI_FORMAT(2),
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        d.VS = bytecode(&vs);
        d.PS = bytecode(&ps);
        d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: 1 };
        d.NumRenderTargets = 2;
        d.RTVFormats[1] = DXGI_FORMAT(28);
        unsafe {
            let owned = own_graphics(&d as *const _ as *const c_void).unwrap();
            let kept = owned.record(Kind::Graphics);
            assert_eq!(kept.shaders.len(), 2);
            assert_eq!(kept.shaders.iter().map(|(at, _)| super::super::pso_manifest::stage_at(&kept, *at)).collect::<Vec<_>>(), ["VS", "PS"]);
            let again = OwnedDesc::from_record(&kept).unwrap();
            let OwnedDesc::Graphics(g, ..) = &again else { panic!("a graphics description") };
            assert_eq!(shader_bytes(&g.VS).unwrap(), vs.as_slice());
            assert_eq!(shader_bytes(&g.PS).unwrap(), ps.as_slice());
            assert!(g.GS.pShaderBytecode.is_null() && g.GS.BytecodeLength == 0);
            assert_eq!((g.NumRenderTargets, g.RTVFormats[1]), (2, DXGI_FORMAT(28)));
            assert_eq!(g.InputLayout.NumElements, 1);
            let e = &*g.InputLayout.pInputElementDescs;
            assert_eq!(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char), name);
            assert_eq!((e.SemanticIndex, e.AlignedByteOffset), (1, 12));
            assert_eq!(std::mem::transmute_copy::<_, usize>(&g.pRootSignature), 0);
            assert_eq!(again.record(Kind::Graphics), kept, "kept again the same");

            let cs = b"compute shader".to_vec();
            let mut c = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
            c.CS = bytecode(&cs);
            c.NodeMask = 1;
            let kept = own_compute(&c as *const _ as *const c_void).unwrap().record(Kind::Compute);
            let again = OwnedDesc::from_record(&kept).unwrap();
            let OwnedDesc::Compute(c2, _) = &again else { panic!("a compute description") };
            assert_eq!(shader_bytes(&c2.CS).unwrap(), cs.as_slice());
            assert_eq!(c2.NodeMask, 1);
            assert_eq!(again.record(Kind::Compute), kept);
        }
    }

    /// A stream whose last subobject is a 4-byte payload at its very end (a sample mask; a retail
    /// game's 688-byte stream) is kept with its shaders, read only within the stream.
    #[test]
    fn a_stream_ending_in_a_four_byte_payload_is_kept() {
        let vs = intern_shader_blob(b"vertex shader bytes");
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(&(vs.as_ptr() as u64).to_le_bytes());
        buf.extend_from_slice(&(vs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(stream_fields(&buf).unwrap().last().map(|f| (f.0, f.1 + f.2)), Some((9, buf.len())));
        let kept = OwnedDesc::Stream(Box::new([0, 0]), buf, vec![vs.clone()], Vec::new(), Vec::new()).record(Kind::MeshStream);
        assert_eq!(kept.shaders, vec![(8, vs)]);
        assert_eq!(super::super::pso_manifest::stage_at(&kept, 8), "VS");
        assert_eq!(&kept.desc[8..16], &[0; 8]);
    }
}


#[cfg(test)]
mod governor_tests {
    use super::*;

    fn governor() -> Governor {
        Governor { state: Mutex::new(GovState { running: 0, started: 0, giants: 0, urgent: 0, starving: 0, reserved: 0, giant_waits: Vec::new(), giants_finished: 0, idle_breaker: super::super::pso_sched::IdleBreaker::new(), live_at: None, runs: Vec::new() }), cv: std::sync::Condvar::new(), test_ram: None }
    }

    /// play-97's FFX_FSR4_PASS_18 (7,698 KB of DXIL, est 25,411 MiB) and _14 (4,961 KB, est
    /// 16,375 MiB), waited for with 2,614 MiB available: refused after `NEVER_FIT_GRACE` instead of
    /// holding their threads 310 s.
    #[test]
    fn a_waited_for_compile_that_never_fits_now_is_refused_after_its_grace() {
        const MIB: u64 = 1 << 20;
        let g = Governor { test_ram: Some((AtomicU64::new(2_614 * MIB), 32 << 30)), ..governor() };
        let asked = std::time::Instant::now();
        assert!(matches!(g.acquire_within(7_882_752, 20, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Wanted), Err(Refused::NeverFits)));
        let waited = asked.elapsed();
        assert!(waited >= super::super::pso_sched::NEVER_FIT_GRACE && waited < super::super::pso_sched::NEVER_FIT_GRACE + std::time::Duration::from_secs(2), "refused after the grace: {waited:?}");
        assert!(matches!(g.acquire_within(5_080_064, 20, None, false, |_| Some(URGENT_FLOOR), false, || DriverDemand::Background), Err(Refused::NoRoom)), "background work is put off as before");
        g.test_ram.as_ref().unwrap().0.store(18_004 * MIB, Ordering::Relaxed);
        let slot = g.acquire_within(5_080_064, 20, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Wanted).ok().expect("with 18 GB available it starts");
        g.release(&slot);
        std::mem::forget(slot);
        assert_eq!(g.in_flight(), (0, 0, 0));
    }

    #[test]
    fn giant_commit_keeps_the_peak_and_reports_missing_samples() {
        let mut peak = CommitPeak::default();
        peak.record(0);
        assert_eq!(peak.growth(1000), None);
        for bytes in [1300, 1700, 0, 900] {
            peak.record(bytes);
        }
        assert_eq!(peak.samples, 3);
        assert_eq!(peak.growth(1000), Some(700));
        assert_eq!(peak.growth(1800), Some(0));
        assert_eq!(peak.growth(0), None);
    }

    #[test]
    fn memory_waited_counts_the_time_an_admission_waited() {
        let g = Governor { test_ram: Some((AtomicU64::new(1 << 30), 32 << 30)), ..governor() };
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let before = memory_waited();
                let slot = g.acquire_within(GIANT_DXIL, 1, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Background).unwrap();
                g.release(&slot);
                std::mem::forget(slot);
                memory_waited() - before
            });
            std::thread::sleep(std::time::Duration::from_millis(400));
            g.test_ram.as_ref().unwrap().0.store(16 << 30, Ordering::Relaxed);
            g.cv.notify_all();
            // (the waiter's own start-up comes out of the 400 ms: under Wine it took up to ~100)
            assert!(waiter.join().unwrap() >= std::time::Duration::from_millis(250));
        });
        let before = memory_waited();
        let slot = g.acquire_within(1 << 10, 1, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Background).unwrap();
        g.release(&slot);
        std::mem::forget(slot);
        assert!(memory_waited() - before < std::time::Duration::from_millis(100), "admitted at once");
    }

    #[test]
    fn compile_permits_report_overlap_even_after_the_other_releases() {
        let g = Governor { test_ram: Some((AtomicU64::new(16 << 30), 32 << 30)), ..governor() };
        let first = g.acquire(GIANT_DXIL, 4, None, false, || true).unwrap();
        let second = g.acquire(1 << 10, 4, None, false, || true).unwrap();
        let second_overlapped = g.release(&second);
        let first_overlapped = g.release(&first);
        std::mem::forget(second);
        std::mem::forget(first);
        assert!(second_overlapped && first_overlapped);
        let alone = g.acquire(GIANT_DXIL, 4, None, false, || true).unwrap();
        let alone_overlapped = g.release(&alone);
        std::mem::forget(alone);
        assert!(!alone_overlapped);
    }

    #[test]
    fn a_waited_for_compile_that_could_fit_waits_even_when_no_others_run() {
        let g = Governor { test_ram: Some((AtomicU64::new(3 << 30), 32 << 30)), ..governor() };
        let giant = 3_515_708;
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                g.acquire(giant, 1, None, false, || true).map(|slot| {
                    let queued_count = g.state.lock().unwrap().starving;
                    g.release(&slot);
                    std::mem::forget(slot);
                    queued_count
                })
            });
            loop {
                if waiter.is_finished() || g.state.lock().unwrap().live_at.is_some() {
                    break;
                }
                std::thread::yield_now();
            }
            std::thread::sleep(std::time::Duration::from_millis(2150));
            let remained_queued = !waiter.is_finished();
            g.test_ram.as_ref().unwrap().0.store(compile_cost(giant).0 + (2 << 30), Ordering::Relaxed);
            g.cv.notify_all();
            assert!(remained_queued, "temporary pressure must not fail an idle governor's compile");
            assert_eq!(waiter.join().unwrap().unwrap(), 0);
        });
    }

    /// A blocking warm-up's giant keeps smaller compiles from taking its room while some run, and
    /// gives way once they drained and it still does not fit: it cannot hold them one at a time.
    #[test]
    fn a_waiting_giant_gives_way_when_draining_did_not_make_room() {
        let (giant, small) = (3_515_708, 512 << 10);
        let available = URGENT_FLOOR + compile_cost(giant).0 - (100 << 20);
        let g = Governor { test_ram: Some((AtomicU64::new(available), 32 << 30)), ..governor() };
        let ask = |len| g.acquire_within(len, 8, None, false, |_| Some(URGENT_FLOOR), false, || DriverDemand::Background);
        let state = |f: fn(&GovState) -> bool| {
            let t0 = std::time::Instant::now();
            while !f(&g.state.lock().unwrap()) {
                assert!(t0.elapsed() < std::time::Duration::from_secs(5));
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        };
        let first = ask(small).ok().unwrap();
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                BLOCKING_WARM.with(|b| b.set(Some((std::time::Instant::now() + std::time::Duration::from_secs(30), WarmDriverCounts::default()))));
                let slot = g.acquire_within(giant, 8, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Background).ok().unwrap();
                g.release(&slot);
                std::mem::forget(slot);
            });
            state(|s| s.giant_waits.len() == 1);
            assert!(matches!(ask(small), Err(Refused::NoRoom)), "a waiting giant keeps its room");
            g.release(&first);
            std::mem::forget(first);
            state(|s| s.giant_waits.is_empty());
            let (a, b) = (ask(small).ok().unwrap(), ask(small).ok().unwrap());
            for slot in [a, b] {
                g.release(&slot);
                std::mem::forget(slot);
            }
            g.test_ram.as_ref().unwrap().0.store(32 << 30, Ordering::Relaxed);
            g.cv.notify_all();
            waiter.join().unwrap();
        });
        assert_eq!(g.state.lock().unwrap().giants_finished, 1);
    }

    #[test]
    fn a_compile_exceeding_the_machines_total_plus_floor_is_refused() {
        let g = Governor { test_ram: Some((AtomicU64::new(3 << 30), 8 << 30)), ..governor() };
        assert!(matches!(g.acquire(4 << 20, 1, None, false, || true), Err(Refused::NoMemory)));
        let s = g.state.lock().unwrap();
        assert_eq!((s.running, s.starving, s.reserved), (0, 0, 0));
    }

    #[test]
    fn an_idle_governor_grants_one_over_budget_compile_to_unblock_a_bind() {
        let giant = 3_241_524;
        let available = URGENT_FLOOR + peak(giant) - (256 << 20);
        let g = Governor { test_ram: Some((AtomicU64::new(available), 32 << 30)), ..governor() };
        let test_ram = &g.test_ram.as_ref().unwrap().0;
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let slot = g.acquire_within(giant, 20, None, false, |_| Some(URGENT_FLOOR), true, || DriverDemand::Blocking).unwrap();
                let got = (slot.available_at_idle_break, g.in_flight());
                g.release(&slot);
                std::mem::forget(slot);
                send.send(got).unwrap();
            });
            let result = receive.recv_timeout(std::time::Duration::from_secs(5));
            if result.is_err() {
                test_ram.store(10 << 30, Ordering::Relaxed);
                g.cv.notify_all();
            }
            assert_eq!(result.unwrap(), (Some(available), (1, 1, 1)));
        });
    }

    /// A run's background compile nothing waits for is not held when it does not fit: refused at
    /// once (`Refused::NoRoom`), its thread free for the next build; nothing is left counted.
    #[test]
    fn a_background_compile_that_does_not_fit_is_put_off_at_once() {
        let g = Governor { test_ram: Some((AtomicU64::new(3 << 30), 32 << 30)), ..governor() };
        let too_big = 3_515_708;
        let t0 = std::time::Instant::now();
        assert!(matches!(g.acquire_within(too_big, 20, None, false, |_| Some(URGENT_FLOOR), false, || DriverDemand::Background), Err(Refused::NoRoom)));
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
        assert!(!g.fits(too_big, 20));
        let s = g.state.lock().unwrap();
        assert_eq!((s.running, s.starving, s.reserved), (0, 0, 0));
    }

    /// … and so is one a bind asked for on a thread that must not be held (a background worker
    /// that took a parked build up again): it is not counted as waiting for memory afterwards.
    #[test]
    fn a_compile_a_bind_asked_for_is_not_waited_for_on_a_thread_that_must_not_be_held() {
        let g = Governor { test_ram: Some((AtomicU64::new(3 << 30), 32 << 30)), ..governor() };
        let too_big = 3_515_708;
        let t0 = std::time::Instant::now();
        assert!(matches!(g.acquire_within(too_big, 20, None, false, |_| Some(URGENT_FLOOR), false, || DriverDemand::Blocking), Err(Refused::NoRoom)));
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
        let s = g.state.lock().unwrap();
        assert_eq!((s.running, s.starving, s.reserved), (0, 0, 0));
    }

    /// The background floor is the least headroom or more, whatever the process is doing.
    #[test]
    fn the_background_floor_is_at_least_the_least_headroom() {
        // (no pace runs in a test: the title counts as loading)
        if std::env::var_os("XCL_PSO_BACKGROUND_FLOOR_MB").is_none() {
            assert_eq!(background_floor_when(false), URGENT_FLOOR, "no title reserve, no 6 GB");
        }
        // while a bind waits it is the bind's floor and the growth: never under that floor
        assert!(background_floor_when(true) >= URGENT_FLOOR);
        assert!(background_floor() >= URGENT_FLOOR);
    }

    /// … while one that fits starts as a background compile, a giant counted as one.
    #[test]
    fn a_background_compile_that_fits_starts_without_waiting() {
        if available_physical() < peak(0) + background_floor_when(false) + (2 << 30) {
            return;
        }
        let g = governor();
        assert!(g.fits(1 << 10, 20));
        let slot = g.acquire_within(1 << 10, 20, None, false, |_| Some(URGENT_FLOOR), false, || DriverDemand::Background).ok().expect("started");
        assert!(!slot.urgent && !slot.giant);
        assert_eq!(g.in_flight(), (1, 0, 0));
        assert!(!g.fits(1 << 10, 1), "within the most at once");
        g.release(&slot);
        std::mem::forget(slot);
        assert_eq!(g.in_flight(), (0, 0, 0));
    }

    /// play-91b: a live cold giant that does not fit waited 250 s counted as an urgent compile
    /// queued, and the background held back for it (`live_compile_waiting` paused the build queue
    /// and preload); 1,830 pipelines waited with 7.5 GB available. It waits in its own lane now.
    #[test]
    fn a_waiting_live_giant_holds_no_background_work_back() {
        let gib = 1u64 << 30;
        let g = Governor { test_ram: Some((AtomicU64::new(7 * gib), 32 * gib)), ..governor() };
        let giant = 3_900_000;
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let slot = g.acquire_within(giant, 20, None, false, |_| None, true, || DriverDemand::Live).ok().unwrap();
                g.release(&slot);
                std::mem::forget(slot);
            });
            let t0 = std::time::Instant::now();
            while g.state.lock().unwrap().live_at.is_none() {
                assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the giant asked");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert!(!waiter.is_finished(), "it does not fit 7 GB, not even on the curve");
            assert_eq!(g.state.lock().unwrap().starving, 0, "a waiting cold giant is not an urgent compile queued");
            let small = g.acquire_within(64 << 10, 20, None, false, |_| None, false, || DriverDemand::Background).ok().expect("background work goes on beside it");
            g.release(&small);
            std::mem::forget(small);
            g.test_ram.as_ref().unwrap().0.store(32 * gib, Ordering::Relaxed);
            g.cv.notify_all();
            waiter.join().unwrap();
        });
        assert_eq!(g.in_flight(), (0, 0, 0));
    }

    #[test]
    fn a_waited_for_compile_that_fits_starts_at_once_outside_the_limits() {
        if available_physical() < peak(0) + URGENT_FLOOR + (1 << 30) {
            return;
        }
        let g = governor();
        let slot = g.acquire(1 << 10, 0, Some(0), false, || true).ok().expect("started");
        assert!(slot.urgent && !slot.giant);
        assert_eq!(g.in_flight(), (1, 0, 1));
        g.release(&slot);
        std::mem::forget(slot);
        assert_eq!(g.in_flight(), (0, 0, 0));
    }

    /// play-88/88b/89: a build a skipped draw waits for asks as a waited-for one, and a small one
    /// never waits on a giant's reservation; after it asked, no background giant starts for
    /// `LIVE_QUIET`; in the game a cold giant never starts beside another (`XCL_HARD_GIANTS`).
    #[test]
    fn live_builds_go_first_hold_background_giants_and_see_settled_giants_as_measured() {
        let gib = 1u64 << 30;
        let g = Governor { test_ram: Some((AtomicU64::new(32 * gib), 64 * gib)), ..governor() };
        let giant = 4_294_756;
        let held = g.acquire_within(giant, 20, None, false, |_| None, false, || DriverDemand::Background).ok().expect("a background giant while live demand is idle");
        assert!(!held.urgent && held.giant);
        g.test_ram.as_ref().unwrap().0.store(peak(giant) + gib + gib / 2, Ordering::Relaxed);
        let small = g.acquire_within(39 << 10, 20, None, false, |_| None, false, || DriverDemand::Live).ok().expect("a small live build never waits on a giant's reservation");
        assert!(small.urgent);
        g.release(&small);
        std::mem::forget(small);
        let live_giant = 1_127_196;
        assert!(matches!(g.acquire_within(live_giant, 20, None, false, |_| None, false, || DriverDemand::Live), Err(Refused::NoRoom)), "a fresh giant keeps its estimate from a live giant");
        let Some(long_ago) = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(600)) else { return };
        g.state.lock().unwrap().runs[0].2 = long_ago;
        let small_background = g.acquire_within(900 << 10, 20, None, false, |_| None, false, || DriverDemand::Background).ok().expect("small background work counts a settled giant as measured too (XCL_SETTLE_BACKGROUND)");
        g.release(&small_background);
        std::mem::forget(small_background);
        assert!(matches!(g.acquire_within(live_giant, 1, None, false, |_| None, false, || DriverDemand::Live), Err(Refused::NoRoom)), "the game's hard rule: no cold giant beside a running one (XCL_HARD_GIANTS)");
        g.release(&held);
        std::mem::forget(held);
        let live = g.acquire_within(live_giant, 1, None, false, |_| None, false, || DriverDemand::Live).ok().expect("a live giant once the other is done");
        assert!(live.urgent, "a live build asks as a waited-for one");
        g.test_ram.as_ref().unwrap().0.store(32 * gib, Ordering::Relaxed);
        assert!(matches!(g.acquire_within(giant, 20, None, false, |_| None, false, || DriverDemand::Background), Err(Refused::NoRoom)), "no background giant within LIVE_QUIET of live demand");
        g.state.lock().unwrap().live_at = Some(long_ago);
        assert!(matches!(g.acquire_within(giant, 20, None, false, |_| None, false, || DriverDemand::Background), Err(Refused::NoRoom)), "nor beside the live giant");
        g.release(&live);
        std::mem::forget(live);
        let next = g.acquire_within(giant, 20, None, false, |_| None, false, || DriverDemand::Background).ok().expect("background giants once live demand is quiet and no giant runs");
        g.release(&next);
        std::mem::forget(next);
        assert_eq!(g.in_flight(), (0, 0, 0));
        assert!(g.state.lock().unwrap().runs.is_empty());
    }
}

#[cfg(test)]
mod set_predication_slot_tests {
    use super::*;

    /// A title's #31 is SetPredication (PC #55) through `list_x::set_predication_x`, which adds
    /// the buffer delta as #30 ResolveQueryData does.
    #[test]
    fn title_slot_31_runs_set_predication_x() {
        assert!(console_remap(Kind::CommandList).contains(&(31, 55)));
        let own = console_slots(Kind::CommandList);
        assert_eq!(own.iter().find(|e| e.0 == 31).map(|e| e.1), Some(super::super::list_x::set_predication_x as usize));
    }
}

/// Unified memory opened as it is before any title code runs, when the coherent window may become
/// a driver heap (`uma_driver`): whichever way the window ends, the GPU reads what the CPU writes
/// to it (`check_uma_import`). `XCL_COHERENT_L0_TESTS=1` only, and alone (`--exact`): the window stays
/// what it became, and a window opened twice cannot be remade. It prints the path it ended on.
#[cfg(test)]
#[test]
fn unified_memory_opened_before_the_title_is_coherent_on_either_path() {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    if std::env::var_os("XCL_COHERENT_L0_TESTS").is_none() {
        return crate::gpu::recomp::runner::skip_test("XCL_COHERENT_L0_TESTS is not set (run alone: the coherent window stays what it becomes)");
    }
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        assert!(UMA_BUFFER.lock().unwrap().is_empty(), "unified memory is open already: run this test alone");
        open_uma(&dev.unwrap(), true);
        let u = crate::gpu::uma::get().expect("unified memory");
        let k = u.committed_window().expect("a coherent window");
        let (error, check) = (UMA_ERROR.lock().unwrap().clone(), UMA_IMPORT_CHECK.lock().unwrap().clone());
        let path = if u.is_driver_window(k) { "a driver L0 heap at its address" } else { "its own pages imported" };
        eprintln!("coherent window ({} MB) opened before the title: {path}; the GPU's read of a CPU write: {check}{error}", u.window(k).1 >> 20);
        if !error.is_empty() && !u.is_driver_window(k) {
            return crate::gpu::recomp::runner::skip_test("this D3D12 cannot import the window (as before the driver heap)");
        }
        assert!(check.starts_with("ok"), "{check}");
    }
}

/// Unified memory opened on `device` for one test. Dropping it gives the process back the state
/// it had before (later tests see no opened windows; the windows' buffers and heaps are leaked).
#[cfg(test)]
#[must_use]
pub(super) struct UmaForTest(Option<(super::sparse::SnapshotForTest, String, String)>);

#[cfg(test)]
pub(super) unsafe fn open_uma_for_test(device: &ID3D12Device) -> UmaForTest {
    if !UMA_BUFFER.lock().unwrap().is_empty() {
        return UmaForTest(None);
    }
    let before = (super::sparse::snapshot_for_test(), UMA_ERROR.lock().unwrap().clone(), UMA_IMPORT_CHECK.lock().unwrap().clone());
    open_uma(device, false);
    UmaForTest(Some(before))
}

#[cfg(test)]
impl Drop for UmaForTest {
    fn drop(&mut self) {
        let Some((sparse, error, check)) = self.0.take() else { return };
        UMA_BUFFER.lock().unwrap().clear();
        UMA_HEAPS.lock().unwrap().clear();
        *MEM_VIEWS.lock().unwrap() = None;
        *UMA_ERROR.lock().unwrap() = error;
        *UMA_IMPORT_CHECK.lock().unwrap() = check;
        super::sparse::restore_for_test(sparse);
    }
}


/// Whether DRED keeps breadcrumbs on XCL's device with this machine's driver. It removes the
/// device, so it runs alone, on a real GPU, while no game runs:
/// `cargo test --release -p xcl --lib -- --ignored --exact hle::d3d12x::real::dred_tests::dred_names_a_list_that_did_not_finish`.
/// A list waits on a fence that never comes, then the device is removed: DRED should name that
/// list as not finished.
#[cfg(test)]
mod dred_tests {
    use super::*;

    #[test]
    #[ignore]
    fn dred_names_a_list_that_did_not_finish() {
        std::env::set_var("XCL_DRED", "1");
        init().expect("a D3D12 device");
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap().clone();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let buffer = |dev: &ID3D12Device| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON, None, &mut r,
                ).unwrap();
                r.unwrap()
            };
            let (src, dst) = (buffer(&dev), buffer(&dev));
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.CopyBufferRegion(&dst, 0, &src, 0, 256);
            list.Close().unwrap();
            queue.Wait(&fence, 1).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            dev.cast::<ID3D12Device5>().unwrap().RemoveDevice();
            let dred = dev.cast::<ID3D12DeviceRemovedExtendedData>().expect("DRED on the device");
            let bc = dred.GetAutoBreadcrumbsOutput().expect("breadcrumbs output");
            let mut lists = 0;
            let mut n = bc.pHeadAutoBreadcrumbNode;
            while !n.is_null() {
                lists += 1;
                n = (*n).pNext;
            }
            assert!(lists >= 1, "DRED kept no breadcrumbs for a submitted list that never ran: they are off for XCL's device");
        }
    }
}

/// Whether resetting a command list waits for a queue that stands behind a GPU wait. In play-102t1
/// the submit thread stood in D3DKMTDestroyAllocation2 under `ID3D12GraphicsCommandList::Reset`
/// (`list_x::record_put_off`) while the compute queues waited for the Signals of that same
/// submit, until the 60 s limit. It needs a real GPU and runs alone, while no game runs:
/// `cargo test --release -p xcl --lib -- --ignored --exact --nocapture hle::d3d12x::real::put_off_reset_probe::reset_of_a_list_behind_a_gated_queue`.
/// Four lists are reset while a queue holds a list behind `Wait` on a fence nobody signals: one
/// closed and never run (what a put-off call's segment is at its submit), one made closed by
/// `CreateCommandList1`, one whose recording ran to the end on another queue, and the gated one
/// itself. Each prints whether its Reset came back within 3 s. The fence is then signalled from
/// the CPU, and every Reset must come back.
#[cfg(test)]
mod put_off_reset_probe {
    use super::*;

    #[test]
    #[ignore]
    fn reset_of_a_list_behind_a_gated_queue() {
        init().expect("a D3D12 device");
        unsafe {
            let raw = device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap().clone();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let queue = |dev: &ID3D12Device| -> ID3D12CommandQueue { dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap() };
            let (gated, free) = (queue(&dev), queue(&dev));
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let buffer = |dev: &ID3D12Device| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON, None, &mut r,
                ).unwrap();
                r.unwrap()
            };
            let (src, dst) = (buffer(&dev), buffer(&dev));
            let recorded = |dev: &ID3D12Device| -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
                list.CopyBufferRegion(&dst, 0, &src, 0, 256);
                list.Close().unwrap();
                (alloc, list)
            };
            let never_run = recorded(&dev);
            let made_closed: (ID3D12CommandAllocator, ID3D12GraphicsCommandList) = (
                dev.CreateCommandAllocator(ty).unwrap(),
                dev.cast::<ID3D12Device4>().unwrap().CreateCommandList1(0, ty, D3D12_COMMAND_LIST_FLAG_NONE).unwrap(),
            );
            let ran = recorded(&dev);
            free.ExecuteCommandLists(&[Some(ran.1.cast().unwrap())]);
            free.Signal(&done, 1).unwrap();
            let t0 = std::time::Instant::now();
            while done.GetCompletedValue() < 1 {
                assert!(t0.elapsed() < std::time::Duration::from_secs(5), "a list on a free queue did not finish in 5 s");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let behind = recorded(&dev);
            gated.Wait(&gate, 1).unwrap();
            gated.ExecuteCommandLists(&[Some(behind.1.cast().unwrap())]);

            let mut probes = Vec::new();
            for (name, (alloc, list)) in [
                ("a list closed and never run", &never_run),
                ("a list made closed by CreateCommandList1", &made_closed),
                ("a list whose recording ran to the end", &ran),
                ("the list behind the gate", &behind),
            ] {
                let (l, a) = (list.as_raw() as usize, alloc.as_raw() as usize);
                let (tx, rx) = std::sync::mpsc::channel();
                let thread = std::thread::spawn(move || {
                    let (l, a) = (l as *mut c_void, a as *mut c_void);
                    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&l).unwrap();
                    let alloc = ID3D12CommandAllocator::from_raw_borrowed(&a).unwrap();
                    let t0 = std::time::Instant::now();
                    let ok = list.Reset(alloc, None).is_ok();
                    let _ = tx.send((t0.elapsed(), ok));
                });
                let back = rx.recv_timeout(std::time::Duration::from_secs(3)).ok();
                match back {
                    Some((d, ok)) => println!("  {name}: Reset came back in {} ms (ok {ok})", d.as_millis()),
                    None => println!("  {name}: Reset did NOT come back within 3 s behind the gated queue"),
                }
                probes.push((name, thread, rx, back.is_some()));
            }
            gate.Signal(1).unwrap();
            for (name, thread, rx, back) in probes {
                if !back {
                    let late = rx.recv_timeout(std::time::Duration::from_secs(10)).ok();
                    assert!(late.is_some(), "{name}: Reset did not come back 10 s after the gate opened");
                    println!("  {name}: Reset came back {} ms after it was called, once the gate opened", late.unwrap().0.as_millis());
                }
                thread.join().unwrap();
            }
        }
    }
}
