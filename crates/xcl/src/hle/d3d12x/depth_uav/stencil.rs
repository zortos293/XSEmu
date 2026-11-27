//! Unordered access to the stencil plane of a depth buffer.
//!
//! The console lets a shader write a depth buffer's stencil plane through an unordered access
//! view (X24_TYPELESS_G8_UINT of a D24S8 buffer, X32_TYPELESS_G8X24_UINT of a D32S8 one). The PC
//! has no unordered access to a stencil plane, so the view was null: loads zero, stores lost.
//! Here the view goes to a shadow, an R8_UINT texture of the buffer's size (8 bits a texel, as
//! the console's stencil surface), kept in step with the plane through a buffer (a stencil plane
//! copies to and from a buffer, not to another texture) at the title's transitions of the depth
//! buffer on a direct list:
//!
//! * out of a state the plane is written in (DEPTH_WRITE, COPY_DEST, RESOLVE_DEST): plane to shadow;
//! * into a state the plane is read in (depth, shader resource, copy or resolve source) from one
//!   it is not written in: shadow to plane, once the shadow was filled from the plane.
//!
//! The title need not name UNORDERED_ACCESS: its shaders' stores land in the plane at its next
//! depth or shader-resource use. Single-mip, single-slice, single-sample 2D depth buffers only;
//! other stencil views stay null.
//!
//! `XCL_STENCIL_UAV=0`: no shadows, null views as before.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{barriers, pc_state, raw, transition, ALL, COPY_DEST, COPY_SOURCE, COPY_TEXTURE_REGION, DEPTH_WRITE, GET_TYPE, PIXEL_READ, SHADER_READ, UNORDERED_ACCESS};

const COMMON: u32 = D3D12_RESOURCE_STATE_COMMON.0 as u32;
const PLANE_WRITES: u32 = DEPTH_WRITE | COPY_DEST | D3D12_RESOURCE_STATE_RESOLVE_DEST.0 as u32;
const PLANE_READS: u32 = DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32 | PIXEL_READ | SHADER_READ | COPY_SOURCE | D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0 as u32;

/// The shadow of one depth buffer's stencil plane.
struct Stencil {
    texture: ID3D12Resource,
    /// the plane's texels between the copies
    buffer: ID3D12Resource,
    /// the plane's footprint in the buffer, and the shadow's (laid out in the plane's rows)
    plane: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    own: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    /// a copy from the plane was recorded: copies back carry the plane's texels
    seeded: AtomicBool,
}

// D3D12 resources are free-threaded
unsafe impl Send for Stencil {}
unsafe impl Sync for Stencil {}

/// the stencil plane's subresource of a single-subresource depth buffer
const PLANE_SUB: u32 = 1;

/// depth buffer → its stencil shadow (None: none could be made)
static STENCILS: Mutex<Option<HashMap<usize, Option<Arc<Stencil>>>>> = Mutex::new(None);
/// a stencil view was asked for (the transition hook has something to do)
static ASKED: AtomicBool = AtomicBool::new(false);
static VIEWS: AtomicU64 = AtomicU64::new(0);
static NULL_VIEWS: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static COPIES_IN: AtomicU64 = AtomicU64::new(0);
static COPIES_OUT: AtomicU64 = AtomicU64::new(0);
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

/// Whether stencil planes get shadows (`XCL_STENCIL_UAV=0`: none).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_STENCIL_UAV").is_ok_and(|v| v.trim() == "0"))
}

/// The title makes an unordered access view (`desc`) of the stencil plane of `res`, a depth
/// buffer the PC made without unordered access (`rd`): the shadow, the view of it and its
/// description. None: a null view.
pub(super) unsafe fn view(
    res: *mut c_void,
    rd: &D3D12_RESOURCE_DESC,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<(*mut c_void, D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_RESOURCE_DESC)> {
    if !enabled() {
        return None;
    }
    ASKED.store(true, Ordering::Release);
    let n = VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
    let made = (|| {
        if rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.MipLevels > 1 || rd.DepthOrArraySize > 1 || rd.SampleDesc.Count > 1 {
            return Err("not a single-subresource 2D buffer".to_string());
        }
        if !matches!(rd.Format, DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
            return Err(format!("format {} has no stencil plane", rd.Format.0));
        }
        let mut v = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
        match desc.map(|d| d.ViewDimension) {
            None => {}
            Some(D3D12_UAV_DIMENSION_TEXTURE2D) if desc.map_or(0, |d| d.Anonymous.Texture2D.MipSlice) == 0 => {}
            Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) if desc.is_some_and(|d| d.Anonymous.Texture2DArray.MipSlice == 0 && d.Anonymous.Texture2DArray.FirstArraySlice == 0) => {
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: 1, PlaneSlice: 0 };
            }
            Some(d) => return Err(format!("view dimension {} past the first mip or slice", d.0)),
        }
        let s = stencil_of(res, rd)?;
        let texture_desc = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_R8_UINT, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, Alignment: 0, ..*rd };
        Ok((s.texture.as_raw(), v, texture_desc))
    })();
    if made.is_err() {
        NULL_VIEWS.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: unordered access view #{n} of a stencil plane ({}x{}, format {}; view format {}, dimension {}): {} (XCL_STENCIL_UAV=0 leaves it null)",
            rd.Width, rd.Height, rd.Format.0, desc.map_or(0, |d| d.Format.0), desc.map_or(0, |d| d.ViewDimension.0),
            match &made { Ok(_) => "of its R8_UINT shadow texture".to_string(), Err(e) => format!("a null view ({e})") }
        );
    }
    made.ok()
}

/// The stencil shadow of the depth buffer `res`, made now if it has none.
unsafe fn stencil_of(res: *mut c_void, rd: &D3D12_RESOURCE_DESC) -> Result<Arc<Stencil>, String> {
    if let Some(kept) = STENCILS.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).cloned()) {
        return kept.ok_or_else(|| "no shadow could be made for it earlier".to_string());
    }
    let (made, failed) = match make(rd) {
        Ok(s) => {
            let n = MADE.fetch_add(1, Ordering::Relaxed) + 1;
            crate::xlog_info!("D3D12.X: stencil plane of depth buffer {}x{} (format {}) with unordered access: views of it go to R8_UINT shadow texture #{n}, copied from and to the plane at the title's transitions", rd.Width, rd.Height, rd.Format.0);
            (Some(Arc::new(s)), None)
        }
        Err(e) => {
            crate::xlog_warn!("D3D12.X: no shadow texture for unordered access to the stencil plane of a depth buffer {}x{} format {} ({e}) – its views stay null", rd.Width, rd.Height, rd.Format.0);
            (None, Some(e))
        }
    };
    let kept = STENCILS.lock().unwrap().get_or_insert_with(HashMap::new).entry(res as usize).or_insert(made).clone();
    super::ACTIVE.store(true, Ordering::Release);
    kept.ok_or_else(|| failed.unwrap_or_else(|| "no shadow could be made for it".into()))
}

unsafe fn make(rd: &D3D12_RESOURCE_DESC) -> Result<Stencil, String> {
    let raw_device = super::super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw_device).ok_or("no device")?;
    super::super::title_textures::own_work(|| {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: rd.Width,
            Height: rd.Height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut texture)
            .map_err(|e| format!("the texture: {e}"))?;
        let texture = texture.ok_or("no texture")?;
        let (mut plane, mut own) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default());
        let mut total = u64::MAX;
        dev.GetCopyableFootprints(rd, PLANE_SUB, 1, 0, Some(&mut plane), None, None, Some(&mut total));
        dev.GetCopyableFootprints(&texture_desc, 0, 1, 0, Some(&mut own), None, None, None);
        let (p, o) = (plane.Footprint, own.Footprint);
        if total == u64::MAX || (p.Width, p.Height) != (o.Width, o.Height) || p.RowPitch < o.RowPitch {
            return Err(format!("the plane's footprint ({}x{}, {} B rows, format {}) is not the shadow's ({}x{}, {} B rows)", p.Width, p.Height, p.RowPitch, p.Format.0, o.Width, o.Height, o.RowPitch));
        }
        own.Footprint.RowPitch = p.RowPitch;
        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
            .map_err(|e| format!("the buffer: {e}"))?;
        Ok(Stencil { texture, buffer: buffer.ok_or("no buffer")?, plane, own, seeded: AtomicBool::new(false) })
    })
}

/// The depth buffer `res` is gone.
pub(super) fn forget(res: usize) {
    if !ASKED.load(Ordering::Acquire) {
        return;
    }
    let gone = STENCILS.lock().unwrap().as_mut().and_then(|m| m.remove(&res));
    drop(gone);
}

/// The title's transition `b` (PC state bits) of `res` on `list`, before the depth plane's own handling
/// (`super::on_transition`): a copy between the stencil plane and its shadow, recorded now, the
/// depth buffer back in the state the transition starts from.
pub(super) unsafe fn on_transition(list: *mut c_void, res: usize, b: &[u32; 8]) {
    if !ASKED.load(Ordering::Acquire) {
        return;
    }
    let Some(s) = STENCILS.lock().unwrap().as_ref().and_then(|m| m.get(&res).cloned()).flatten() else { return };
    let sub = b[4];
    if sub != ALL && sub != PLANE_SUB {
        return;
    }
    let (before, after) = (pc_state(b[5]), pc_state(b[6]));
    let into_shadow = before & PLANE_WRITES != 0 && after & PLANE_WRITES == 0;
    let into_plane = before & PLANE_WRITES == 0 && after & PLANE_READS != 0 && b[6] & UNORDERED_ACCESS == 0;
    if !into_shadow && !into_plane {
        return;
    }
    let direct = raw::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, GET_TYPE)(list) == D3D12_COMMAND_LIST_TYPE_DIRECT;
    if b[1] != 0 || !direct || (into_plane && !s.seeded.load(Ordering::Relaxed)) {
        LEFT_OUT.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let (t, buffer) = (s.texture.as_raw() as usize, s.buffer.as_raw() as usize);
    let at = |r: usize, k: u32| D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(&r),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k },
    };
    let placed = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(&buffer),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
    };
    let (held, shadow_state, from, to) = if into_shadow {
        (COPY_SOURCE, COPY_DEST, (at(res, PLANE_SUB), placed(s.plane)), (placed(s.own), at(t, 0)))
    } else {
        (COPY_DEST, COPY_SOURCE, (at(t, 0), placed(s.own)), (placed(s.plane), at(res, PLANE_SUB)))
    };
    let copy = raw::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(list, COPY_TEXTURE_REGION);
    let predicated = super::super::list_x::before_own_pass(list);
    barriers(list, &[transition(res, sub, before, held), transition(t, 0, UNORDERED_ACCESS, shadow_state)]);
    copy(list, &from.1, 0, 0, 0, &from.0, std::ptr::null());
    barriers(list, &[transition(buffer, ALL, COPY_DEST, COPY_SOURCE)]);
    copy(list, &to.1, 0, 0, 0, &to.0, std::ptr::null());
    barriers(list, &[transition(res, sub, held, before), transition(t, 0, shadow_state, UNORDERED_ACCESS), transition(buffer, ALL, COPY_SOURCE, COMMON)]);
    super::super::list_x::after_own_pass(list, false, predicated);
    if into_shadow {
        s.seeded.store(true, Ordering::Relaxed);
        COPIES_IN.fetch_add(1, Ordering::Relaxed);
    } else {
        COPIES_OUT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Every 300 frames the title presents (`super::log_frame`): the counts, when they changed.
pub(super) fn log_frame() {
    if !ASKED.load(Ordering::Acquire) {
        return;
    }
    static LAST: Mutex<Option<[u64; 6]>> = Mutex::new(None);
    let c = [&VIEWS, &NULL_VIEWS, &MADE, &COPIES_IN, &COPIES_OUT, &LEFT_OUT].map(|a| a.load(Ordering::Relaxed));
    if LAST.lock().unwrap().replace(c) == Some(c) {
        return;
    }
    crate::xlog_info!(
        "D3D12.X: stencil planes with unordered access (300 frames on; XCL_STENCIL_UAV=0 turns it off): {} views made ({} left null), {} shadow textures; copied plane to shadow {} times, shadow to plane {} times, {} transitions left out (split barrier, not a direct list, or nothing copied from the plane yet)",
        c[0], c[1], c[2], c[3], c[4], c[5]
    );
}
