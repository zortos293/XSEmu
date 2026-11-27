//! Console R9G9B9E5_SHAREDEXP render targets and UAV textures as R11G11B10_FLOAT castable to the
//! title's 32-bit views (`XCL_RGB9E5_CASTABLE`), and a log of the views the PC cannot make.
//!
//! The PC renders to and writes UAVs of no R9G9B9E5 texture (NVIDIA refuses the flags:
//! E_INVALIDARG), so `device_x::pc_resource_desc` makes such a texture another format. A title
//! views it in 32-bit formats: Grand Theft Auto VI's Series S light buffer is written as R32_UINT
//! (the packed bits) by a compute pass and read as R9G9B9E5 by its lighting. As
//! R16G16B16A16_FLOAT, 8 bytes a texel and another format family, neither view is one the PC can
//! make of it.
//!
//! With relaxed format casting (D3D12_OPTIONS12.RelaxedFormatCastingSupported) and
//! ID3D12Device10, such a texture is R11G11B10_FLOAT (4 bytes a texel; a render target and UAV
//! format on the PC) made with the castable formats `FORMATS` (CreateCommittedResource3,
//! CreatePlacedResource2, CreateReservedResource2): the R32_UINT and R9G9B9E5 views see the same
//! bits, as on the console. Render target and UAV views in R9G9B9E5, which the PC writes in
//! neither way, are R11G11B10_FLOAT (`device_x::pc_written_view_format`).
//!
//! Which creations: those a console call for such a texture makes (`Scope`), and of them the
//! texture itself (R11G11B10_FLOAT with the render target or UAV flag) - a title's own
//! R11G11B10_FLOAT textures stay as they are. One the PC refuses with the castable formats (or
//! whose initial state has no barrier layout) is made as before without them, logged once per
//! route: the same format and size, so it still fits the memory it was placed in.
//!
//! `XCL_RGB9E5_CASTABLE=1` turns it on (off by default since play-110/111: an RTV in R9G9B9E5 writes
//! R11G11B10_FLOAT bits that the R9G9B9E5 shader views decode wrongly); otherwise R16G16B16A16_FLOAT, as before.
//!
//! The behaviour is VIctory's (Marvisak/VIctory 372d56fc and b27faad6, GPL-3: the behaviour, none
//! of its code).
//!
//! Views (`check_view`, with or without the switch): an SRV, UAV or RTV in a format of another
//! family than its texture's and not among the texture's castable formats is logged once per
//! (kind, texture format, view format), the first `VIEW_PAIRS` pairs.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
use windows_sys::core::GUID;

use crate::util::HRESULT;

/// The formats such a texture can be viewed in.
pub(super) const FORMATS: [DXGI_FORMAT; 3] = [DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R11G11B10_FLOAT];

/// Private data of a texture made with `FORMATS`: those formats.
const TAG: windows::core::GUID = windows::core::GUID::from_u128(0x2d1b7e57_3b33_456c_9596_ec3362560b5f);

static ON: AtomicBool = AtomicBool::new(false);

/// Whether console R9G9B9E5 render targets and UAV textures are castable R11G11B10_FLOAT (false
/// until `init` says so - and in unit tests, which make no device this way).
pub(super) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// At device creation: `XCL_RGB9E5_CASTABLE` (off unless "1") and what the device can do.
pub(super) unsafe fn init(dev: &ID3D12Device) {
    // opt-in: a render target view in R9G9B9E5 writes the R11G11B10_FLOAT bits, which the
    // title's R9G9B9E5 shader views decode as shared-exponent ones (play-110/111: a posterized
    // false-colour sky over a black ground) - so on only with XCL_RGB9E5_CASTABLE=1
    if !std::env::var("XCL_RGB9E5_CASTABLE").is_ok_and(|v| v.trim() == "1") {
        crate::xlog_info!("XCL_RGB9E5_CASTABLE off (=1 turns it on): console R9G9B9E5 render targets and UAV textures are R16G16B16A16_FLOAT, as before");
        return;
    }
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
    let relaxed = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
        std::mem::size_of_val(&options) as u32).is_ok() && options.RelaxedFormatCastingSupported.as_bool();
    let device10 = dev.cast::<ID3D12Device10>().is_ok();
    ON.store(relaxed && device10, Ordering::Relaxed);
    if relaxed && device10 {
        crate::xlog_info!("XCL_RGB9E5_CASTABLE: console R9G9B9E5 render targets and UAV textures are R11G11B10_FLOAT, castable to R32_UINT and R9G9B9E5 (relaxed format casting, ID3D12Device10)");
    } else {
        crate::xlog_warn!("XCL_RGB9E5_CASTABLE: off - relaxed format casting {relaxed}, ID3D12Device10 {device10}: console R9G9B9E5 render targets and UAV textures are R16G16B16A16_FLOAT, as before");
    }
}

/// Whether a console texture of `format` with D3D12_RESOURCE_FLAGS `flags` is an R9G9B9E5 render
/// target or UAV texture.
pub fn rgb9e5_target(format: u32, flags: u32) -> bool {
    format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0 as u32 && flags & 5 != 0
}

thread_local! {
    static CONSOLE_RGB9E5: Cell<bool> = const { Cell::new(false) };
}

/// A console call that makes a texture (its console desc): while it lasts, the creations on this
/// thread know whether it is an R9G9B9E5 render target or UAV texture made castable.
pub(super) struct Scope(bool);

impl Scope {
    pub(super) fn enter(console: Option<&[u8; 0x38]>) -> Scope {
        let u = |x: &[u8; 0x38], o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        let ours = on() && console.is_some_and(|x| rgb9e5_target(u(x, 0x20), u(x, 0x30)));
        Scope(CONSOLE_RGB9E5.with(|c| c.replace(ours)))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        CONSOLE_RGB9E5.with(|c| c.set(self.0));
    }
}

/// Whether a PC texture of `format` with `flags` is to be made with `FORMATS`: the texture of
/// a console call in `Scope`.
pub(super) fn wanted(format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> bool {
    CONSOLE_RGB9E5.with(Cell::get) && format == DXGI_FORMAT_R11G11B10_FLOAT && flags.0 & 5 != 0
}

unsafe fn wanted_desc(desc: usize) -> Option<D3D12_RESOURCE_DESC> {
    if desc == 0 || !CONSOLE_RGB9E5.with(Cell::get) || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_RESOURCE_DESC>()) {
        return None;
    }
    let d = (desc as *const D3D12_RESOURCE_DESC).read_unaligned();
    wanted(d.Format, d.Flags).then_some(d)
}

/// The barrier layout a texture made in the legacy initial `state` starts in (None: no one
/// layout is that state).
pub fn legacy_layout(state: u32) -> Option<D3D12_BARRIER_LAYOUT> {
    Some(match state {
        0 => D3D12_BARRIER_LAYOUT_COMMON,
        0x4 => D3D12_BARRIER_LAYOUT_RENDER_TARGET,
        0x8 => D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
        0x40 | 0x80 | 0xC0 => D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
        0x400 => D3D12_BARRIER_LAYOUT_COPY_DEST,
        0x800 => D3D12_BARRIER_LAYOUT_COPY_SOURCE,
        0x1000 => D3D12_BARRIER_LAYOUT_RESOLVE_DEST,
        0x2000 => D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE,
        0xAC3 => D3D12_BARRIER_LAYOUT_GENERIC_READ,
        _ => return None,
    })
}

/// A creation route: its bit in the once-only logs, and its name.
#[derive(Clone, Copy)]
pub(super) struct Route(u32, &'static str);

pub(super) const COMMITTED: Route = Route(1, "committed");
pub(super) const PLACED: Route = Route(2, "placed in a heap");
pub(super) const RESERVED: Route = Route(4, "reserved, on the pages of its address");
pub(super) const ARENA: Route = Route(8, "placed in an arena with an undefined layout");

static MADE: AtomicU32 = AtomicU32::new(0);
static REFUSED: AtomicU32 = AtomicU32::new(0);

fn first(logged: &AtomicU32, route: Route) -> bool {
    logged.fetch_or(route.0, Ordering::Relaxed) & route.0 == 0
}

/// `resource`, made of `d` with `FORMATS`: tagged with them (`check_view`); the first of
/// each route logged.
pub(super) unsafe fn made(route: Route, resource: *mut c_void, d: &D3D12_RESOURCE_DESC) {
    if let Some(r) = ID3D12Resource::from_raw_borrowed(&resource) {
        let _ = r.SetPrivateData(&TAG, std::mem::size_of_val(&FORMATS) as u32, Some(FORMATS.as_ptr().cast()));
    }
    if first(&MADE, route) {
        crate::xlog_info!(
            "XCL_RGB9E5_CASTABLE: a console R9G9B9E5 texture ({}x{}, flags {:#x}) is R11G11B10_FLOAT castable to R32_UINT and R9G9B9E5 ({}; later ones on this route not logged)",
            d.Width, d.Height, d.Flags.0, route.1
        );
    }
}

/// The PC did not make `d` with `FORMATS` (`why`): it is made without them. Logged once per route.
pub(super) fn refused(route: Route, d: &D3D12_RESOURCE_DESC, why: &str) {
    if first(&REFUSED, route) {
        crate::xlog_warn!(
            "XCL_RGB9E5_CASTABLE: a console R9G9B9E5 texture ({}x{}, flags {:#x}, {}) not made castable: {why} - made without the castable formats, as R11G11B10_FLOAT; its R32_UINT and R9G9B9E5 views are not ones the PC can make (later ones on this route not logged)",
            d.Width, d.Height, d.Flags.0, route.1
        );
    }
}

unsafe fn settle(route: Route, hr: HRESULT, ppv: *mut *mut c_void, d: &D3D12_RESOURCE_DESC) -> Option<HRESULT> {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        made(route, *ppv, d);
        return Some(hr);
    }
    refused(route, d, &format!("{hr:#x}"));
    None
}

fn desc1(d: &D3D12_RESOURCE_DESC) -> D3D12_RESOURCE_DESC1 {
    D3D12_RESOURCE_DESC1 {
        Dimension: d.Dimension, Alignment: d.Alignment, Width: d.Width, Height: d.Height,
        DepthOrArraySize: d.DepthOrArraySize, MipLevels: d.MipLevels, Format: d.Format,
        SampleDesc: d.SampleDesc, Layout: d.Layout, Flags: d.Flags, ..Default::default()
    }
}

/// CreateCommittedResource of such a texture (`wanted`): CreateCommittedResource3 with
/// `FORMATS`. None: not one, or refused - the caller makes it as before.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn committed(this: *mut c_void, props: usize, flags: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let d = wanted_desc(desc)?;
    let Some(layout) = legacy_layout(state as u32) else {
        refused(COMMITTED, &d, &format!("its initial state {state:#x} has no barrier layout"));
        return None;
    };
    let d1 = desc1(&d);
    let hr = super::real::dev_create_committed_resource3_pc(this, props, flags, &d1 as *const _ as usize, layout.0 as usize, clear, 0, FORMATS.len(), FORMATS.as_ptr() as usize, riid, ppv);
    settle(COMMITTED, hr, ppv, &d)
}

/// CreatePlacedResource of such a texture: CreatePlacedResource2 with `FORMATS` (None: as `committed`).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn placed(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let d = wanted_desc(desc)?;
    let Some(layout) = legacy_layout(state as u32) else {
        refused(PLACED, &d, &format!("its initial state {state:#x} has no barrier layout"));
        return None;
    };
    let d1 = desc1(&d);
    let hr = super::real::dev_create_placed_resource2_pc(this, heap, offset, &d1 as *const _ as usize, layout.0 as usize, clear, FORMATS.len(), FORMATS.as_ptr() as usize, riid, ppv);
    settle(PLACED, hr, ppv, &d)
}

/// CreateReservedResource of such a texture: CreateReservedResource2 with `FORMATS` (None: as `committed`).
pub(super) unsafe fn reserved(this: *mut c_void, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let d = wanted_desc(desc)?;
    let Some(layout) = legacy_layout(state as u32) else {
        refused(RESERVED, &d, &format!("its initial state {state:#x} has no barrier layout"));
        return None;
    };
    let hr = super::real::dev_create_reserved_resource2_pc(this, desc, layout.0 as usize, clear, 0, FORMATS.len(), FORMATS.as_ptr() as usize, riid, ppv);
    settle(RESERVED, hr, ppv, &d)
}

/// The typeless format of a format's family (the format itself: a family of its own, or not known).
static VIEWS: Mutex<Vec<(&'static str, u32, u32, u64)>> = Mutex::new(Vec::new());

/// The views made of a castable console R9G9B9E5 texture (`made`), by kind and by the title's
/// format and the PC's: logged at the 1st, 2nd, 4th ... of each, so a run shows which views write
/// it in which encoding and which read it. An RTV or UAV whose PC format is R11G11B10_FLOAT writes
/// those bits; an SRV in R9G9B9E5 reads shared-exponent ones - the two disagree, and say so.
pub(super) unsafe fn note_view(kind: &'static str, res: *mut c_void, rd: &D3D12_RESOURCE_DESC, title: u32, pc: u32) {
    if rd.Format != DXGI_FORMAT_R11G11B10_FLOAT || !castable_to(res, DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0 as u32) {
        return;
    }
    let n = {
        let mut views = VIEWS.lock().unwrap();
        let Some(i) = views.iter().position(|v| (v.0, v.1, v.2) == (kind, title, pc)).or_else(|| (views.len() < 32).then(|| { views.push((kind, title, pc, 0)); views.len() - 1 })) else { return };
        views[i].3 += 1;
        views[i].3
    };
    if !n.is_power_of_two() {
        return;
    }
    let what = match (kind, pc) {
        ("an RTV" | "a UAV", 26) => "writes R11G11B10_FLOAT bits, which an R9G9B9E5 view reads as shared-exponent ones",
        (_, 67) => "reads shared-exponent bits",
        (_, 42) => "reads or writes the packed bits as they are",
        _ => "",
    };
    crate::xlog_info!("XCL_RGB9E5_CASTABLE: {kind} in title format {title} (PC {pc}) of a castable R9G9B9E5 texture ({}x{}, flags {:#x}) #{n}: {what}", rd.Width, rd.Height, rd.Flags.0);
}

pub fn family(f: u32) -> u32 {
    match f {
        1..=4 => 1,
        5..=8 => 5,
        9..=14 => 9,
        15..=18 => 15,
        19..=22 => 19,
        23..=25 => 23,
        27..=32 => 27,
        33..=38 => 33,
        39..=43 => 39,
        44..=47 => 44,
        48..=52 => 48,
        53..=59 => 53,
        60..=64 => 60,
        70..=72 => 70,
        73..=75 => 73,
        76..=78 => 76,
        79..=81 => 79,
        82..=84 => 82,
        87 | 90 | 91 => 90,
        88 | 92 | 93 => 92,
        94..=96 => 94,
        97..=99 => 97,
        f => f,
    }
}

/// Whether a view in format `view` of a texture in format `resource` is of another family (0:
/// not named, no check).
pub fn foreign(resource: u32, view: u32) -> bool {
    resource != 0 && view != 0 && family(resource) != family(view)
}

/// (kind, texture format, view format) of the foreign views logged
static VIEW_PAIRS: Mutex<Vec<(&'static str, u32, u32)>> = Mutex::new(Vec::new());
const VIEW_PAIRS_LOGGED: usize = 32;
static VIEW_PAIRS_FULL: AtomicBool = AtomicBool::new(false);

unsafe fn castable_to(res: *mut c_void, view: u32) -> bool {
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else { return false };
    let mut list = [DXGI_FORMAT(0); FORMATS.len()];
    let mut size = std::mem::size_of_val(&list) as u32;
    r.GetPrivateData(&TAG, &mut size, Some(list.as_mut_ptr().cast())).is_ok()
        && list.iter().take(size as usize / std::mem::size_of::<DXGI_FORMAT>()).any(|f| f.0 as u32 == view)
}

fn texel(f: u32) -> String {
    match crate::hle::xg::element(f) {
        Some((log2, 1)) => format!("{} B a texel", 1 << log2),
        Some((log2, block)) => format!("{} B a {block}x{block} block", 1 << log2),
        None => "size not known".into(),
    }
}

/// A `kind` view (`res`, its desc `rd`) in format `view`, as the PC is asked for it: logged
/// when the PC cannot make it (`foreign` and not castable), once per pair.
pub(super) unsafe fn check_view(kind: &'static str, res: *mut c_void, rd: &D3D12_RESOURCE_DESC, view: u32) {
    let resource = rd.Format.0 as u32;
    if rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || !foreign(resource, view) || VIEW_PAIRS_FULL.load(Ordering::Relaxed) || castable_to(res, view) {
        return;
    }
    let n = {
        let mut pairs = VIEW_PAIRS.lock().unwrap();
        if pairs.contains(&(kind, resource, view)) {
            return;
        }
        pairs.push((kind, resource, view));
        if pairs.len() >= VIEW_PAIRS_LOGGED {
            VIEW_PAIRS_FULL.store(true, Ordering::Relaxed);
        }
        pairs.len()
    };
    crate::xlog_warn!(
        "D3D12.X: {kind} in format {view} ({}) of a texture in format {resource} ({}), {}x{}, flags {:#x}: another format family and not among its castable formats - not a view the PC can make (a null view, garbage or the device removed); pair {n} of at most {VIEW_PAIRS_LOGGED} logged",
        texel(view), texel(resource), rd.Width, rd.Height, rd.Flags.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb9e5_targets_are_render_targets_or_uav_textures() {
        assert!(rgb9e5_target(67, 0x1) && rgb9e5_target(67, 0x4) && rgb9e5_target(67, 0x30_0005));
        assert!(!rgb9e5_target(67, 0) && !rgb9e5_target(67, 0x8) && !rgb9e5_target(26, 0x5));
        assert!(!on(), "off until a device says otherwise");
        assert!(!wanted(DXGI_FORMAT_R11G11B10_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), "outside a console call");
        let mut x = [0u8; 0x38];
        x[0x20] = 67;
        x[0x30] = 5;
        let _scope = Scope::enter(Some(&x));
        assert!(!wanted(DXGI_FORMAT_R11G11B10_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), "the switch is off in tests");
    }

    #[test]
    fn legacy_states_have_their_layouts() {
        assert_eq!(legacy_layout(0), Some(D3D12_BARRIER_LAYOUT_COMMON));
        assert_eq!(legacy_layout(0x4), Some(D3D12_BARRIER_LAYOUT_RENDER_TARGET));
        assert_eq!(legacy_layout(0x8), Some(D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS));
        assert_eq!(legacy_layout(0xC0), Some(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE));
        assert_eq!(legacy_layout(0xAC3), Some(D3D12_BARRIER_LAYOUT_GENERIC_READ));
        assert_eq!(legacy_layout(0x400), Some(D3D12_BARRIER_LAYOUT_COPY_DEST));
        assert_eq!(legacy_layout(0x4 | 0x8), None);
        assert_eq!(legacy_layout(0x10), None);
    }

    /// The views the PC cannot make without a castable list: another family, whatever the size.
    #[test]
    fn foreign_views_are_of_another_family() {
        assert!(foreign(10, 42) && foreign(26, 42) && foreign(26, 67) && foreign(10, 67));
        assert!(!foreign(44, 46) && !foreign(19, 21) && !foreign(39, 42) && !foreign(53, 56) && !foreign(28, 29));
        assert!(!foreign(87, 91) && !foreign(26, 26) && !foreign(0, 42) && !foreign(26, 0));
        assert!(foreign(28, 87) && foreign(41, 26));
        assert_eq!((texel(10), texel(67), texel(71)), ("8 B a texel".into(), "4 B a texel".into(), "8 B a 4x4 block".into()));
    }
}
