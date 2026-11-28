//! d3d12_xs.dll – D3D12.X.
//!
//! Backends:
//! * `real` (default): genuine PC D3D12 objects from the GPU driver with extended
//!   vtables (PC methods go straight to the driver, console-only slots are guarded and
//!   logged), shader capture, fallback pipeline states, presentation window.
//! * `null`: stand-in objects that render nothing (`graphics = "null"` in xcl.toml or
//!   when no D3D12 device can be created).

#![allow(non_snake_case)]

pub mod bind;
mod indirect;
mod markers;
mod dispatch_timing;
mod gpu_timing;
mod gpu_timing_stats;
mod gpu_timing_waits;
mod hang;
pub mod console_heap;
pub mod console_list;
pub mod device_x;
mod castable;
pub mod dxgi_device;
mod cpu_writes;
mod depth_capture;
mod depth_capture_views;
mod depth_lineage;
mod depth_lineage_queue;
mod depth_state;
mod depth_twin_probe;
mod census;
mod dsv_trace;
mod material_gate_views;
mod material_gate_capture;
pub mod depth_uav;
mod e5_copy;
mod fence_x;
mod free_policy;
pub mod frame;
pub mod guard;
pub mod layout;
pub mod list_x;
pub mod null;
pub mod page_mappings;
pub mod present;
pub mod present_order;
pub(crate) mod texture_alias;
pub mod pso_async;
mod pass_chain;
pub(crate) mod fidelityfx;
mod fsr4_standin;
mod light_wait;
mod upscaler;
mod pso_fail;
mod pso_canon;
mod pso_library;
pub mod pso_manifest;
pub mod pso_sched;
mod mesh_attrib;
mod mesh_pso;
pub mod real;
pub mod residency;
mod residency_priority;
pub mod warm_children;
mod warm_peers;
mod single;
mod residency_list;
mod texture_gate;
pub mod sparse;
mod external_buffers;

/// (diagnostics) The buffers the title placed over `[va, va + len)` outside unified memory.
pub(crate) fn placed_buffers_over(va: usize, len: usize) -> Vec<String> {
    external_buffers::placed_over(va, len)
}
pub mod raytracing;
pub mod tex_census;
pub mod title_gta6;
mod title_gta6_ambient;
pub mod title_list;
pub mod title_textures;
mod uma_driver;
pub(crate) mod image_writeback;
pub mod vram;
mod wait_trace;
mod live_status;
pub mod console_status;
mod submit_watch;
mod hung_skip;
mod shader_code;
mod target_dump;

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::util::{HRESULT, E_INVALIDARG, E_NOTIMPL, S_OK};

/// data export: titles pass `&IID_ID3D12Device` to D3D12XboxCreateDevice / QueryInterface
static IID_ID3D12_DEVICE: windows::core::GUID = windows::core::GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);

pub fn exports() -> Vec<(&'static str, usize)> {
    let mut v = exports_fns();
    v.push(("IID_ID3D12Device", &IID_ID3D12_DEVICE as *const _ as usize));
    v
}

fn exports_fns() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        D3D12CreateDevice,
        D3D12XboxCreateDevice,
        D3D12ConfigureVirtualMemoryX,
        D3D12GetDebugInterface,
        D3D12SerializeRootSignature,
        D3D12SerializeVersionedRootSignature,
        D3D12CreateRootSignatureDeserializer,
        D3D12CreateVersionedRootSignatureDeserializer,
        D3D12EnableExperimentalFeatures,
        D3D12GetInterface,
        D3D10CreateBlob,
        D3D12XboxSetProcessDebugFlags,
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Real,
    Null,
}

/// Decided once: the real backend unless disabled or the driver refuses.
static BACKEND: OnceLock<Backend> = OnceLock::new();

fn backend_with_remap(allow_driver_remap: bool) -> Backend {
    *BACKEND.get_or_init(|| {
        let cfg = crate::state::config();
        if cfg.graphics.eq_ignore_ascii_case("null") {
            crate::xlog_info!("D3D12.X: backend 'null' (from the config)");
            return Backend::Null;
        }
        match real::init_with_remap(allow_driver_remap) {
            Ok(desc) => {
                crate::xlog_info!("D3D12.X: backend 'real' – {desc}");
                pso_async::defer_to_submit();
                Backend::Real
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: could not create a D3D12 device ({e}) – switching to the fallback device");
                Backend::Null
            }
        }
    })
}

pub fn backend() -> Backend {
    backend_with_remap(false)
}

/// `backend` decided before any title code runs: only then may the coherent window become a
/// driver heap (`uma_driver`) – nothing of the title is in it and none of its threads use it.
pub(crate) fn backend_before_guest() -> Backend {
    backend_with_remap(true)
}

/// Call a function of the system d3d12.dll by name.
unsafe fn sys(name: &str) -> Option<usize> {
    crate::resolver::system_proc("d3d12.dll", &crate::pe::ImportName::Name { hint: 0, name: name.into() })
}

macro_rules! forward {
    ($name:literal, $ty:ty, $null:expr, ($($a:ident),*)) => {{
        if backend() == Backend::Real {
            if let Some(f) = sys($name) {
                let f: $ty = std::mem::transmute(f);
                return f($($a),*);
            }
        }
        $null($($a),*)
    }};
}

pub unsafe extern "system" fn D3D12CreateDevice(adapter: *mut c_void, fl: u32, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if backend() == Backend::Real {
        return real::create_device(riid, ppv);
    }
    null::D3D12CreateDevice(adapter, fl, riid, ppv)
}

#[repr(C)]
struct VirtualMemoryConfig {
    flags: u32,
    page_table_blocks: u32,
    frame_buffer_preserve_bytes: u32,
    reserved: u32,
    outputs: [u64; 2],
}

unsafe extern "system" fn D3D12ConfigureVirtualMemoryX(config: *mut VirtualMemoryConfig) -> HRESULT {
    if config.is_null() || !config.is_aligned() || !crate::util::memory_is_writable(config as usize, std::mem::size_of::<VirtualMemoryConfig>()) {
        return E_INVALIDARG;
    }
    let config = &mut *config;
    if config.flags & !1 != 0 || config.page_table_blocks > 33 || config.reserved != 0 {
        return E_INVALIDARG;
    }
    if config.flags != 0 || config.frame_buffer_preserve_bytes != 0 || backend() != Backend::Real {
        return E_NOTIMPL;
    }
    let bytes = (config.page_table_blocks as usize) << 22;
    if let Err(hr) = sparse::reserve_tile_table(bytes) {
        return hr;
    }
    config.outputs = [0; 2];
    crate::xlog_info!("D3D12.X: configured graphics page tracking for {} MiB of requested tables", bytes >> 20);
    0x1000_0000
}

pub unsafe extern "system" fn D3D12XboxCreateDevice(adapter: *mut c_void, params: *const c_void, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    // D3D12XBOX_CREATE_DEVICE_PARAMETERS: the first word is the SDK version the title was
    // built with (the console's vtable layouts follow it)
    if !params.is_null() && crate::gpu::shader::readable(params as usize, 32) {
        let w: Vec<u32> = (0..8).map(|k| (params as *const u32).add(k).read_unaligned()).collect();
        crate::xlog_info!("D3D12XboxCreateDevice: parameters {w:x?} (SDK version {})", w[0]);
    }
    if backend() == Backend::Real {
        return real::create_device(riid, ppv);
    }
    null::D3D12XboxCreateDevice(adapter, params, riid, ppv)
}

pub unsafe extern "system" fn D3D12GetDebugInterface(riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!("D3D12GetDebugInterface", unsafe extern "system" fn(*const c_void, *mut *mut c_void) -> HRESULT, null::D3D12GetDebugInterface, (riid, ppv))
}

pub unsafe extern "system" fn D3D12SerializeRootSignature(d: *const c_void, v: u32, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12SerializeRootSignature",
        unsafe extern "system" fn(*const c_void, u32, *mut *mut c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12SerializeRootSignature,
        (d, v, blob, err)
    )
}

pub unsafe extern "system" fn D3D12SerializeVersionedRootSignature(d: *const c_void, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    let mut own_err: *mut c_void = std::ptr::null_mut();
    let e = if err.is_null() { &mut own_err as *mut _ } else { err };
    // console-only root signature flags (seen: 0x3000000 from XboxBvhBuilder.dll) – Windows
    // rejects unknown bits; the flags sit at +40 in every version of the desc (48 bytes)
    const PC_FLAGS: u32 = 0xFFF;
    let mut copy = [0u8; 48];
    let mut d = d;
    if !d.is_null() && *(d as *const u8).add(40).cast::<u32>() & !PC_FLAGS != 0 {
        std::ptr::copy_nonoverlapping(d as *const u8, copy.as_mut_ptr(), 48);
        let f = copy.as_mut_ptr().add(40).cast::<u32>();
        crate::xlog_info!("D3D12SerializeVersionedRootSignature: console flags {:#x} dropped", *f & !PC_FLAGS);
        *f &= PC_FLAGS;
        d = copy.as_ptr() as *const c_void;
    }
    type Serialize = unsafe extern "system" fn(*const c_void, *mut *mut c_void, *mut *mut c_void) -> HRESULT;
    let hr = match sys("D3D12SerializeVersionedRootSignature").filter(|_| backend() == Backend::Real) {
        Some(f) => std::mem::transmute::<_, Serialize>(f)(d, blob, e),
        None => null::D3D12SerializeVersionedRootSignature(d, blob, e),
    };
    if hr < 0 && !d.is_null() {
        // D3D12_VERSIONED_ROOT_SIGNATURE_DESC: version, then (1.1 / 1.2) parameters, samplers, flags
        let w = |o: usize| *(d as *const u8).add(o).cast::<u32>();
        let text = blob_text(*e);
        crate::xlog_warn!(
            "D3D12SerializeVersionedRootSignature: {hr:#x} (version {}, parameters {}, samplers {}, flags {:#x}): {text}",
            w(0),
            w(8),
            w(24),
            w(40)
        );
    }
    if err.is_null() && !own_err.is_null() {
        blob_release(own_err);
    }
    hr
}

/// text of an ID3DBlob (error messages)
unsafe fn blob_text(b: *mut c_void) -> String {
    if b.is_null() {
        return String::new();
    }
    let vt = *(b as *const *const usize);
    let ptr: unsafe extern "system" fn(*mut c_void) -> *const u8 = std::mem::transmute(*vt.add(3));
    let len: unsafe extern "system" fn(*mut c_void) -> usize = std::mem::transmute(*vt.add(4));
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr(b), len(b))).trim_end_matches('\0').trim().to_string()
}

unsafe fn blob_release(b: *mut c_void) {
    let vt = *(b as *const *const usize);
    let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
    release(b);
}

pub unsafe extern "system" fn D3D12CreateRootSignatureDeserializer(d: *const c_void, l: usize, r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12CreateRootSignatureDeserializer",
        unsafe extern "system" fn(*const c_void, usize, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12CreateRootSignatureDeserializer,
        (d, l, r, ppv)
    )
}

pub unsafe extern "system" fn D3D12CreateVersionedRootSignatureDeserializer(d: *const c_void, l: usize, r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12CreateVersionedRootSignatureDeserializer",
        unsafe extern "system" fn(*const c_void, usize, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12CreateVersionedRootSignatureDeserializer,
        (d, l, r, ppv)
    )
}

pub unsafe extern "system" fn D3D12EnableExperimentalFeatures(n: u32, i: *const c_void, c: *mut c_void, s: *mut u32) -> HRESULT {
    forward!(
        "D3D12EnableExperimentalFeatures",
        unsafe extern "system" fn(u32, *const c_void, *mut c_void, *mut u32) -> HRESULT,
        null::D3D12EnableExperimentalFeatures,
        (n, i, c, s)
    )
}

pub unsafe extern "system" fn D3D12GetInterface(clsid: *const c_void, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12GetInterface",
        unsafe extern "system" fn(*const c_void, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12GetInterface,
        (clsid, riid, ppv)
    )
}

pub unsafe extern "system" fn D3D10CreateBlob(size: usize, out: *mut *mut c_void) -> HRESULT {
    null::D3D10CreateBlob(size, out)
}

pub unsafe extern "system" fn D3D12XboxSetProcessDebugFlags(_flags: u32) -> HRESULT {
    S_OK
}

mod physical_import;
mod physical_directory;
