//! Unordered access to depth buffers.
//!
//! The console lets one texture be a depth buffer and take unordered access views
//! (ALLOW_DEPTH_STENCIL | ALLOW_UNORDERED_ACCESS: a depth pyramid that compute passes build, a
//! depth buffer a compute pass fills and draws then test against). The PC refuses the pair, so
//! the PC resource is a depth buffer only (`device_x::pc_resource_desc`) and a view of it for
//! unordered access had nowhere to go: a null view, whose loads are zero and whose stores are
//! lost.
//!
//! Each such depth buffer gets a shadow: a single-channel texture of the same size, mips and
//! slices (R32 for D32 / D32S8 / D24S8, R16 for D16) that takes the unordered access views and
//! the shader views beside them (`real::dev_create_uav_x`). On the console both are one
//! memory; here they are kept in step where the title says which use comes next – its
//! transitions (`list_x::resource_barrier_x`):
//!
//! * to UNORDERED_ACCESS: the depth plane is copied to the shadow;
//! * from UNORDERED_ACCESS: the shadow is copied back.
//!
//! A depth buffer without a stencil plane (D32, D16) has the format of its shadow: the copies
//! are CopyTextureRegion, every bit as it is, on any list type (as upstream XCL #1989 copies
//! them). The depth plane beside a stencil plane (D32S8, D24S8) is no format a colour texture
//! has: a compute pass reads the depth buffer's shader view into the shadow, and a full-screen
//! pass writing SV_Depth copies the shadow back (the stencil plane stays as it is; depth
//! outside the viewport's 0..1 is clamped there).
//!
//! All subresources of a transition, or the one it names. While the title has the depth
//! buffer in UNORDERED_ACCESS the PC one is in NON_PIXEL_SHADER_RESOURCE (a state every list
//! type can name). A copy is left out when the list it would be recorded on has put both in
//! step and recorded nothing since that could change either (`note_op`); what other lists
//! did is not known while one records, so the first transition of a list always copies. A
//! compute list cannot draw: the copy back by a pass that it owes is made by the next direct
//! list that names the depth buffer in a transition.
//!
//! Left as null views: multisampled depth buffers. The stencil plane has a shadow of its own
//! (`stencil`, `XCL_STENCIL_UAV=0`: null views as before).
//!
//! A title that never moves its depth buffers to or from UNORDERED_ACCESS (on the console the
//! two are one memory, so it need not) has them kept in step by use as well
//! (`XCL_DEPTH_UAV_SYNC`, on by default; `=0`: at those transitions only), as VICTORY does
//! (crates/victory-gpu/src/host.rs `twin_use`): a draw or dispatch whose unordered access
//! tables name a shadow leaves it ahead of its depth plane, and a draw with the depth buffer as
//! its depth target (not a read-only one), a clear or a copy into it leaves the plane ahead. The
//! side a command uses is first copied from the other when that is ahead, as at a transition of
//! the depth buffer, and what a list still has ahead when it closes is copied then, for the lists
//! after it, which may read the depth buffer through views XCL does not see. A copy needs the
//! state each of the plane's subresources is in on the list (its transitions, depth target,
//! clears and copies say it): until it is known the copy waits for a later use, on a later list
//! too (a list that closes leaves the side ahead with the shadow). Bundles take no part.
//!
//! A use by a draw or dispatch is found in its unordered access tables, and a table names a
//! shadow only when the mapping from each view slot to its resource is kept: a plain run keeps
//! that for no one, so the sync above sees no use at all. `XCL_DEPTH_UAV_OWNERS=1` keeps that
//! mapping, and with it a shadow no copy ever reached is copied from the plane before a draw or
//! dispatch is said to use it (off by default: an entry per view slot, and the copies the sync
//! then records are work the run does not do today). A use of a shadow that needed the plane and
//! did not get a copy of it says nothing of the plane: the model is left as it was rather than
//! told the shadow's side is the one ahead.
//!
//! `XCL_SHADOW_PROBE=1`: once a frame, a line for each shadow texture a shader pass sampled that
//! frame – its number and size, the copies it received and sent back, the depth plane's last
//! clear value and the pass that sampled it (`probe_frame`). It keeps a mapping of its own from a
//! view slot to its resource (`probe_slot`), so it watches a run as that run is, without the
//! mapping `XCL_DEPTH_UAV_OWNERS` keeps.
//!
//! `XCL_DEPTH_UAV_TWIN=0` (upstream's name): no shadows, null views as before.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::depth_lineage;
use super::depth_state::{self, Range};
use super::real;

mod stencil;

const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
const UNORDERED_ACCESS: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
const SHADER_READ: u32 = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32;
const PIXEL_READ: u32 = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32;
const DEPTH_WRITE: u32 = D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32;
const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
const COPY_SOURCE: u32 = D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32;

fn legacy_range(sub: u32) -> Range {
    if sub == ALL { Range::All } else { Range::One(sub) }
}

fn depth_range(s: &Shadow, sub: u32) -> Range {
    if sub == ALL {
        Range::Rect { ranges: [(0, s.mips), (0, s.slices), (0, 1)], mips: s.mips, slices: s.slices }
    } else {
        Range::One(sub)
    }
}

// PC command list methods (the runtime's own: `raw`)
const GET_TYPE: usize = 8;
const DRAW_INSTANCED: usize = 12;
const DISPATCH: usize = 14;
const COPY_TEXTURE_REGION: usize = 16;
const IA_SET_PRIMITIVE_TOPOLOGY: usize = 20;
const RS_SET_VIEWPORTS: usize = 21;
const RS_SET_SCISSOR_RECTS: usize = 22;
const SET_PIPELINE_STATE: usize = 25;
const RESOURCE_BARRIER: usize = 26;
const SET_DESCRIPTOR_HEAPS: usize = 28;
const SET_COMPUTE_ROOT_SIGNATURE: usize = 29;
const SET_GRAPHICS_ROOT_SIGNATURE: usize = 30;
const SET_COMPUTE_ROOT_DESCRIPTOR_TABLE: usize = 31;
const SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE: usize = 32;
const SET_COMPUTE_ROOT_CONSTANTS: usize = 35;
const SET_GRAPHICS_ROOT_CONSTANTS: usize = 36;
const OM_SET_RENDER_TARGETS: usize = 46;
const CLEAR_DEPTH_STENCIL_VIEW: usize = 47;

/// descriptors of a shadow's view heap: the depth plane, the shadow, then the shadow's mips
/// for unordered access
const VIEW_DEPTH: u32 = 0;
const VIEW_SHADOW: u32 = 1;
const VIEW_MIPS: u32 = 2;

/// The shadow of one depth buffer.
pub(super) struct Shadow {
    texture: ID3D12Resource,
    /// its number, in the order shadows were made (the probe's index)
    index: u64,
    /// kept for the descriptors below
    _views: ID3D12DescriptorHeap,
    _targets: ID3D12DescriptorHeap,
    views_heap: usize,
    views_gpu: u64,
    views_step: u32,
    /// depth views of the depth buffer: mip + slice · mips
    targets_cpu: usize,
    targets_step: u32,
    width: u32,
    height: u32,
    mips: u32,
    slices: u32,
    one_d: bool,
    depth_format: DXGI_FORMAT,
    typeless: DXGI_FORMAT,
    view_format: DXGI_FORMAT,
    /// the depth buffer has the shadow's format: CopyTextureRegion between them
    exact: bool,
    /// subresources whose copy back a list that cannot draw left to a direct list
    owed: Mutex<Vec<u32>>,
    /// copies the shadow received from the depth plane, and sent back to it
    copies_in: AtomicU64,
    copies_out: AtomicU64,
    /// a copy back was recorded: the title drew or dispatched while it had the depth buffer
    /// in unordered access
    written: AtomicBool,
    /// the side a list that closed left ahead of the other (`Ahead`, `XCL_DEPTH_UAV_SYNC`)
    ahead: AtomicU8,
}

// D3D12 resources and descriptor heaps are free-threaded
unsafe impl Send for Shadow {}
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The depth plane's (mip, slice) of the title's subresource index `sub` (all of them for
    /// ALL; none for another plane's or one past the end).
    fn parts(&self, sub: u32) -> Vec<(u32, u32)> {
        parts_of(self.mips, self.slices, sub)
    }

    /// `parts` as (mip, first slice, slices) runs, a mip at a time.
    fn runs(&self, sub: u32) -> Vec<(u32, u32, u32)> {
        if sub == ALL {
            return (0..self.mips).map(|m| (m, 0, self.slices)).collect();
        }
        self.parts(sub).into_iter().map(|(m, s)| (m, s, 1)).collect()
    }

    fn size(&self, mip: u32) -> (u32, u32) {
        ((self.width >> mip).max(1), (self.height >> mip).max(1))
    }

    fn view(&self, index: u32) -> u64 {
        self.views_gpu + (index * self.views_step) as u64
    }

    fn target(&self, mip: u32, slice: u32) -> usize {
        self.targets_cpu + ((mip + slice * self.mips) * self.targets_step) as usize
    }
}

/// The depth plane's (mip, slice) of subresource index `sub` of a texture of `mips` mips and
/// `slices` slices.
fn parts_of(mips: u32, slices: u32, sub: u32) -> Vec<(u32, u32)> {
    if sub == ALL {
        return (0..slices).flat_map(|s| (0..mips).map(move |m| (m, s))).collect();
    }
    if sub >= mips * slices {
        return Vec::new();
    }
    vec![(sub % mips, sub / mips)]
}

/// depth buffer → its shadow (None: none could be made)
static SHADOWS: Mutex<Option<HashMap<usize, Option<Arc<Shadow>>>>> = Mutex::new(None);
/// a depth buffer was asked for unordered access (the hooks below have something to do)
static ACTIVE: AtomicBool = AtomicBool::new(false);

static VIEWS: AtomicU64 = AtomicU64::new(0);
static NULL_VIEWS: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static COPIES_IN: AtomicU64 = AtomicU64::new(0);
static COPIES_IN_LEFT_OUT: AtomicU64 = AtomicU64::new(0);
static COPIES_OUT: AtomicU64 = AtomicU64::new(0);
static COPIES_OUT_LEFT_OUT: AtomicU64 = AtomicU64::new(0);
static COPIES_OWED: AtomicU64 = AtomicU64::new(0);
static WRITTEN: AtomicU64 = AtomicU64::new(0);
static USE_OUT: AtomicU64 = AtomicU64::new(0);
static USE_IN: AtomicU64 = AtomicU64::new(0);
static USE_LEFT: AtomicU64 = AtomicU64::new(0);

/// What was counted so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// unordered access views of depth buffers the title made, and those of them left null
    pub views: u64,
    pub null_views: u64,
    pub shadows: u64,
    /// copies depth → shadow recorded, and transitions that needed none
    pub copies_in: u64,
    pub copies_in_left_out: u64,
    /// copies shadow → depth recorded, transitions that needed none, and subresources a list
    /// that cannot draw left owed
    pub copies_out: u64,
    pub copies_out_left_out: u64,
    pub copies_owed: u64,
    /// depth buffers whose shadow was copied back at least once: the title drew or dispatched
    /// while it had them in unordered access (whether a shader stored to them is not known)
    pub written: u64,
    /// copies by use (`XCL_DEPTH_UAV_SYNC`): shadow → depth, depth → shadow, and sides lists
    /// left ahead when they closed (for the lists after them)
    pub by_use_out: u64,
    pub by_use_in: u64,
    pub by_use_left: u64,
}

pub fn counts() -> Counts {
    let c = |a: &AtomicU64| a.load(Ordering::Relaxed);
    Counts {
        views: c(&VIEWS),
        null_views: c(&NULL_VIEWS),
        shadows: c(&MADE),
        copies_in: c(&COPIES_IN),
        copies_in_left_out: c(&COPIES_IN_LEFT_OUT),
        copies_out: c(&COPIES_OUT),
        copies_out_left_out: c(&COPIES_OUT_LEFT_OUT),
        copies_owed: c(&COPIES_OWED),
        written: c(&WRITTEN),
        by_use_out: c(&USE_OUT),
        by_use_in: c(&USE_IN),
        by_use_left: c(&USE_LEFT),
    }
}

fn log_counts(why: &str) {
    let c = counts();
    crate::xlog_info!(
        "D3D12.X: depth buffers with unordered access ({why}): {} views made ({} left null), {} shadow textures, {} of them written back (draws or dispatches ran while the title had them in unordered access); copied to the shadow {} times ({} not needed), back {} times ({} not needed, {} left to a direct list); by use: into the plane {} times, into the shadow {} times, {} left to a later list",
        c.views, c.null_views, c.shadows, c.written, c.copies_in, c.copies_in_left_out, c.copies_out, c.copies_out_left_out, c.copies_owed, c.by_use_out, c.by_use_in, c.by_use_left
    );
}

/// Every 300 frames the title presents (`present`): the counts, when they changed.
pub(super) fn log_frame() {
    stencil::log_frame();
    if !active() {
        return;
    }
    static LAST: Mutex<Option<Counts>> = Mutex::new(None);
    let c = counts();
    if LAST.lock().unwrap().replace(c) == Some(c) {
        return;
    }
    log_counts("300 frames on");
    if c.views > c.null_views && c.copies_in + c.copies_in_left_out + c.copies_out + c.copies_out_left_out + c.copies_owed == 0 {
        crate::xlog_warn!("D3D12.X: the title has unordered access views of depth buffers and has moved none of them to or from UNORDERED_ACCESS: their shadow textures are not in step with the depth planes");
    }
}

/// Whether any depth buffer was asked for unordered access.
#[inline]
pub(super) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// The state the PC depth buffer is in while the title has it in `state`: UNORDERED_ACCESS –
/// not one it has – is NON_PIXEL_SHADER_RESOURCE.
pub(super) fn pc_state(state: u32) -> u32 {
    if state & UNORDERED_ACCESS != 0 {
        (state & !UNORDERED_ACCESS) | SHADER_READ
    } else {
        state
    }
}

/// (depth view format, the depth plane's shader view format, the shadow's typeless format, its
/// view format) for a depth buffer's resource format – typeless as `pc_resource_desc` makes
/// them (a typed depth format has no shader view to copy from).
fn formats(resource: DXGI_FORMAT) -> Option<(DXGI_FORMAT, DXGI_FORMAT, DXGI_FORMAT, DXGI_FORMAT)> {
    Some(match resource {
        DXGI_FORMAT_R32G8X24_TYPELESS => (DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT),
        DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT),
        DXGI_FORMAT_R24G8_TYPELESS => (DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT),
        DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM),
        _ => return None,
    })
}

/// The format of a shadow view for the format `view` the title gave its view of a depth buffer
/// of format `resource` (as `device_x::pc_shader_view_format` names it). None: the stencil
/// plane's.
fn view_format(resource: DXGI_FORMAT, view: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    let (_, _, typeless, default) = formats(resource)?;
    if matches!(view, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT | DXGI_FORMAT_X24_TYPELESS_G8_UINT | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_UNORM) {
        return None;
    }
    // the depth bits as integers: the same bits in the shadow of a depth buffer of that width
    let same_bits = match (resource, typeless) {
        (DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_R32G8X24_TYPELESS, _) => matches!(view, DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT),
        (DXGI_FORMAT_R16_TYPELESS, _) => matches!(view, DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SINT | DXGI_FORMAT_R16_SNORM | DXGI_FORMAT_R16_FLOAT),
        _ => false,
    };
    Some(if same_bits { view } else { default })
}

/// Whether depth buffers get shadows (`XCL_DEPTH_UAV_TWIN=0`: none).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_DEPTH_UAV_TWIN").is_ok_and(|v| v.trim() == "0"))
}

/// Whether `rd` is a depth buffer the PC could not also give unordered access that a shadow
/// can stand in for.
fn takes_shadow(rd: &D3D12_RESOURCE_DESC) -> bool {
    matches!(rd.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        && rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        && !rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        && rd.SampleDesc.Count <= 1
        && formats(rd.Format).is_some()
}

fn known(res: usize) -> Option<Option<Arc<Shadow>>> {
    if !active() {
        return None;
    }
    SHADOWS.lock().unwrap().as_ref()?.get(&res).cloned()
}

pub(super) fn capture_twin(res: usize) -> Option<(usize, i32, bool, bool)> {
    let s = known(res)??;
    Some((s.texture.as_raw() as usize, s.view_format.0, s.exact, s.written.load(Ordering::Relaxed)))
}

pub(super) fn select_twin_probe(owner: usize, mip: u32, slice: u32, sub: u32) {
    if !depth_lineage::twin_enabled() { return }
    if let Some(Some(s)) = known(owner) {
        super::depth_twin_probe::select(owner, &s, s.texture.as_raw() as usize, mip, slice, sub);
    }
}

/// The shadow of the depth buffer `res` (described by `rd`), made now if it has none.
unsafe fn shadow_of(res: *mut c_void, rd: &D3D12_RESOURCE_DESC) -> Option<Arc<Shadow>> {
    if let Some(s) = known(res as usize) {
        return s;
    }
    if !takes_shadow(rd) {
        return None;
    }
    let made = if enabled() { make(res, rd) } else { Err("XCL_DEPTH_UAV_TWIN=0".into()) };
    let shadow = match made {
        Ok(mut s) => {
            let n = MADE.fetch_add(1, Ordering::Relaxed) + 1;
            s.index = n;
            if n <= 32 || n.is_power_of_two() {
                crate::xlog_info!(
                    "D3D12.X: depth buffer {}x{} ({} slices, {} mips, format {}{}) with unordered access: views of it go to shadow texture #{n} (format {}), copied from and to the depth plane at the title's transitions ({})",
                    s.width, s.height, s.slices, s.mips, rd.Format.0, if s.one_d { ", 1D" } else { "" }, s.view_format.0,
                    if s.exact { "texture copies" } else { "a compute pass in, a depth-writing draw back" }
                );
            }
            Some(Arc::new(s))
        }
        Err(e) => {
            crate::xlog_warn!("D3D12.X: no shadow texture for unordered access to a depth buffer {}x{} format {} ({e}) – its views stay null", rd.Width, rd.Height, rd.Format.0);
            None
        }
    };
    let mut g = SHADOWS.lock().unwrap();
    let (kept, created) = match g.get_or_insert_with(HashMap::new).entry(res as usize) {
        Entry::Occupied(e) => (e.get().clone(), false),
        Entry::Vacant(e) => (e.insert(shadow).clone(), true),
    };
    ACTIVE.store(true, Ordering::Release);
    drop(g);
    if let Some(s) = kept.as_ref().filter(|_| created) {
        let twin = s.texture.as_raw() as usize;
        depth_lineage::start(res as usize, twin);
        depth_lineage::note(res as usize, 0, Range::All, "twin-created", || {
            format!("twin={twin:#x} size={}x{} mips={} slices={} depth_format={} twin_format={} exact={}",
                s.width, s.height, s.mips, s.slices, s.depth_format.0, s.typeless.0, s.exact)
        });
    }
    kept
}

unsafe fn make(res: *mut c_void, rd: &D3D12_RESOURCE_DESC) -> Result<Shadow, String> {
    let raw = real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
    let depth = ID3D12Resource::from_raw_borrowed(&res).ok_or("no resource")?;
    let (depth_format, depth_view, typeless, view_format) = formats(rd.Format).ok_or("not a depth format with a shader view")?;
    let one_d = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    let (mips, slices) = ((rd.MipLevels as u32).max(1), (rd.DepthOrArraySize as u32).max(1));
    super::title_textures::own_work(|| {
        let mut texture: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: rd.Dimension,
                Width: rd.Width,
                Height: rd.Height,
                DepthOrArraySize: slices as u16,
                MipLevels: mips as u16,
                Format: typeless,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut texture,
        )
        .map_err(|e| format!("the texture: {e}"))?;
        let texture = texture.ok_or("no texture")?;
        if rd.Format == typeless {
            if let Ok(mut placement) = crate::gpu::texture::image_bridge::GuestImage::get(depth) {
                placement.host = crate::gpu::texture::image_bridge::ownership::HostBacking::PRIVATE;
                let _ = placement.tag(texture.as_raw());
            }
        }
        let views: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: VIEW_MIPS + mips,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(|e| format!("its view heap: {e}"))?;
        let targets: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: mips * slices, ..Default::default() })
            .map_err(|e| format!("the depth view heap: {e}"))?;
        let (views_cpu, views_step) = (views.GetCPUDescriptorHandleForHeapStart().ptr, dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        let (targets_cpu, targets_step) = (targets.GetCPUDescriptorHandleForHeapStart().ptr, dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV));
        let at = |k: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: views_cpu + (k * views_step) as usize };
        let read = |format: DXGI_FORMAT| {
            let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
            if one_d {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MipLevels: mips, ArraySize: slices, ..Default::default() };
            } else {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: mips, ArraySize: slices, ..Default::default() };
            }
            v
        };
        dev.CreateShaderResourceView(depth, Some(&read(depth_view)), at(VIEW_DEPTH));
        dev.CreateShaderResourceView(&texture, Some(&read(view_format)), at(VIEW_SHADOW));
        for mip in 0..mips {
            let mut v = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: view_format, ..Default::default() };
            if one_d {
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: 0, ArraySize: slices };
            } else {
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: 0, ArraySize: slices, PlaneSlice: 0 };
            }
            dev.CreateUnorderedAccessView(&texture, None::<&ID3D12Resource>, Some(&v), at(VIEW_MIPS + mip));
            for slice in 0..slices {
                let mut d = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: depth_format, ..Default::default() };
                if one_d {
                    d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
                    d.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_DSV { MipSlice: mip, FirstArraySlice: slice, ArraySize: 1 };
                } else {
                    d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                    d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { MipSlice: mip, FirstArraySlice: slice, ArraySize: 1 };
                }
                dev.CreateDepthStencilView(depth, Some(&d), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: targets_cpu + ((mip + slice * mips) * targets_step) as usize });
            }
        }
        Ok(Shadow {
            views_heap: views.as_raw() as usize,
            views_gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
            views_step,
            targets_cpu,
            targets_step,
            texture,
            // numbered by the caller that counts it (`shadow_of`)
            index: 0,
            _views: views,
            _targets: targets,
            width: rd.Width as u32,
            height: if one_d { 1 } else { rd.Height },
            mips,
            slices,
            one_d,
            depth_format,
            typeless,
            view_format,
            exact: rd.Format == typeless,
            owed: Mutex::new(Vec::new()),
            copies_in: AtomicU64::new(0),
            copies_out: AtomicU64::new(0),
            written: AtomicBool::new(false),
            ahead: AtomicU8::new(Ahead::Neither as u8),
        })
    })
}

/// The depth buffer `res` is gone (its last reference released).
pub(super) fn forget(res: usize) {
    depth_lineage::forget(res);
    stencil::forget(res);
    if !active() {
        return;
    }
    let gone = SHADOWS.lock().unwrap().as_mut().and_then(|m| m.remove(&res));
    if gone.is_some() && depth_state::enabled() {
        if let Some(d) = SLOT146.lock().unwrap().as_mut() {
            d.samples.remove(&res);
        }
    }
    if let Some(m) = DSVS.lock().unwrap().as_mut() {
        m.retain(|_, d| d.depth != res);
    }
    drop(gone);
}

/// The title makes an unordered access view (`desc`, None: the default one) of `res`, a
/// resource the PC made without unordered access (`rd`): for a depth buffer, the shadow to make
/// the view of, the view there and the shadow's description. None: a null view.
pub(super) unsafe fn view(
    res: *mut c_void,
    rd: &D3D12_RESOURCE_DESC,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<(*mut c_void, D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_RESOURCE_DESC)> {
    if !rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
        return None;
    }
    let n = VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
    let wanted = desc.map_or(DXGI_FORMAT_UNKNOWN, |d| d.Format);
    let mut view_range = Range::All;
    let mut stencil = matches!(wanted, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT | DXGI_FORMAT_X24_TYPELESS_G8_UINT | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_UNORM);
    if depth_lineage::enabled() {
        let (mip, slice, count, plane) = match desc {
            Some(d) => match d.ViewDimension {
                D3D12_UAV_DIMENSION_TEXTURE1D => (d.Anonymous.Texture1D.MipSlice, 0, 1, 0),
                D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
                    let a = d.Anonymous.Texture1DArray;
                    (a.MipSlice, a.FirstArraySlice, a.ArraySize, 0)
                }
                D3D12_UAV_DIMENSION_TEXTURE2D => {
                    let a = d.Anonymous.Texture2D;
                    (a.MipSlice, 0, 1, a.PlaneSlice)
                }
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                    let a = d.Anonymous.Texture2DArray;
                    (a.MipSlice, a.FirstArraySlice, a.ArraySize, a.PlaneSlice)
                }
                _ => (0, 0, 0, 0),
            },
            None => (0, 0, u32::from(rd.DepthOrArraySize), 0),
        };
        stencil |= plane == 1;
        let (mips, slices) = (u32::from(rd.MipLevels), u32::from(rd.DepthOrArraySize));
        if mip < mips && count != 0 && slice.checked_add(count).is_some_and(|end| end <= slices) {
            view_range = Range::Rect { ranges: [(mip, 1), (slice, count), (if stencil { 1 } else { plane }, 1)], mips, slices };
        }
    }
    if stencil {
        if let Some(v) = stencil::view(res, rd, desc) {
            return Some(v);
        }
    }
    let made = (|| {
        let format = view_format(rd.Format, wanted).ok_or("unsupported-format")?;
        let s = shadow_of(res, rd).ok_or(if rd.SampleDesc.Count > 1 { "multisampled" } else { "no-twin" })?;
        let mut v = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ..Default::default() };
        match desc {
            Some(d) => {
                v.ViewDimension = d.ViewDimension;
                match d.ViewDimension {
                    D3D12_UAV_DIMENSION_TEXTURE1D if s.one_d => v.Anonymous.Texture1D = d.Anonymous.Texture1D,
                    D3D12_UAV_DIMENSION_TEXTURE1DARRAY if s.one_d => v.Anonymous.Texture1DArray = d.Anonymous.Texture1DArray,
                    D3D12_UAV_DIMENSION_TEXTURE2D if !s.one_d && d.Anonymous.Texture2D.PlaneSlice == 0 => v.Anonymous.Texture2D = d.Anonymous.Texture2D,
                    D3D12_UAV_DIMENSION_TEXTURE2DARRAY if !s.one_d && d.Anonymous.Texture2DArray.PlaneSlice == 0 => v.Anonymous.Texture2DArray = d.Anonymous.Texture2DArray,
                    _ => return Err("unsupported-dimension-or-plane"),
                }
            }
            None => match (s.one_d, s.slices > 1) {
                (true, false) => v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D,
                (true, true) => {
                    v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                    v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: s.slices };
                }
                (false, false) => v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D,
                (false, true) => {
                    v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                    v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: s.slices, PlaneSlice: 0 };
                }
            },
        }
        let texture_desc = D3D12_RESOURCE_DESC { Format: s.typeless, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, Alignment: 0, ..*rd };
        Ok((s.texture.as_raw(), v, texture_desc))
    })();
    match &made {
        Ok((twin, v, _)) => depth_lineage::note(res as usize, 0, view_range, "twin-uav-view", || {
            format!("twin={:#x} requested_format={} actual_format={} dimension={}", *twin as usize, wanted.0, v.Format.0, v.ViewDimension.0)
        }),
        Err(reason) => depth_lineage::note(res as usize, 0, view_range, "null-depth-uav-view", || {
            let reason = if stencil { "stencil-plane" } else { *reason };
            format!("reason={reason} resource_format={} requested_format={} dimension={}", rd.Format.0, wanted.0, desc.map_or(0, |d| d.ViewDimension.0))
        }),
    }
    if made.is_err() {
        NULL_VIEWS.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: unordered access view #{n} of a depth buffer ({}x{}, {} slices, {} mips, format {}; view format {}, dimension {}): {}",
            rd.Width, rd.Height, rd.DepthOrArraySize, rd.MipLevels, rd.Format.0,
            desc.map_or(0, |d| d.Format.0), desc.map_or(0, |d| d.ViewDimension.0),
            if made.is_ok() { "of its shadow texture" } else { "a null view (the stencil plane, a multisampled buffer, or no shadow)" }
        );
    }
    made.ok()
}

// ---------------------------------------------------------------------------------------
// what a recording knows
// ---------------------------------------------------------------------------------------

/// Of one recording: how many commands that can change a texture it has recorded (`note_op`),
/// and the (depth buffer, subresource) pairs whose shadow it put in step, with that count then.
#[derive(Default)]
struct Recording {
    ops: u64,
    in_step: Vec<(usize, u32, u64)>,
    /// what it did with each shadow it used (`XCL_DEPTH_UAV_SYNC`)
    uses: Vec<Use>,
}

static RECORDINGS: Mutex<Option<HashMap<usize, Recording>>> = Mutex::new(None);
/// how many recordings `RECORDINGS` holds (none: a draw has nothing to tell)
static RECORDING: AtomicU64 = AtomicU64::new(0);

/// `list` records a command that can change what a texture holds (a draw, a dispatch, a clear,
/// a copy): what it knew to be in step may not be any more.
#[inline]
pub(super) fn note_op(list: *mut c_void) {
    if !active() || RECORDING.load(Ordering::Acquire) == 0 {
        return;
    }
    if let Some(r) = RECORDINGS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        r.ops += 1;
    }
}

/// `list` starts another recording (or is gone).
pub(super) fn list_reset(list: usize) {
    if !active() || RECORDING.load(Ordering::Acquire) == 0 {
        return;
    }
    if RECORDINGS.lock().unwrap().as_mut().and_then(|m| m.remove(&list)).is_some() {
        RECORDING.fetch_sub(1, Ordering::AcqRel);
    }
}

fn in_step(list: *mut c_void, res: usize, sub: u32) -> bool {
    RECORDINGS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).is_some_and(|r| r.in_step.iter().any(|&(d, s, at)| d == res && at == r.ops && (s == sub || s == ALL)))
}

fn mark_in_step(list: *mut c_void, res: usize, sub: u32) {
    let mut g = RECORDINGS.lock().unwrap();
    let r = recording(&mut g, list);
    let ops = r.ops;
    r.in_step.retain(|&(d, s, at)| at == ops && !(d == res && (s == sub || sub == ALL)));
    r.in_step.push((res, sub, ops));
}

/// PC `Dispatch` of an adopted list (XCL's own calls of it for the title's dispatches).
pub(super) unsafe extern "system" fn pc_dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    super::census::dispatch(this as usize, [x, y, z]);
    note_op(this);
    super::image_writeback::note_work(this);
    let f = raw::<unsafe extern "system" fn(*mut c_void, u32, u32, u32)>(this, DISPATCH);
    super::gpu_timing::op_grid(this, super::gpu_timing::Kind::Dispatch, || 0, [x, y, z], || f(this, x, y, z))
}

/// PC `ClearDepthStencilView` of an adopted list.
pub(super) unsafe extern "system" fn pc_clear_depth_stencil(this: *mut c_void, dsv: usize, flags: u32, depth: f32, stencil: u8, n: u32, rects: *const RECT) {
    on_clear(this, dsv, flags);
    probe_clear(dsv, flags, depth);
    note_op(this);
    let f = raw::<unsafe extern "system" fn(*mut c_void, usize, u32, f32, u8, u32, *const RECT)>(this, CLEAR_DEPTH_STENCIL_VIEW);
    super::gpu_timing::class(this, super::gpu_timing::Kind::Clear, || f(this, dsv, flags, depth, stencil, n, rects));
    super::depth_capture_views::note_clear(this, dsv, flags, depth, n);
}

// ---------------------------------------------------------------------------------------
// the copies
// ---------------------------------------------------------------------------------------

/// a PC method of `list` (the runtime's own, past XCL's hooks)
unsafe fn raw<F: Copy>(list: *mut c_void, slot: usize) -> F {
    let f = if real::is_adopted(list) { real::orig_fn(list, slot) } else { *(*(list as *const *const usize)).add(slot) };
    std::mem::transmute_copy(&f)
}

fn transition(res: usize, sub: u32, before: u32, after: u32) -> [u32; 8] {
    [0, 0, res as u32, (res as u64 >> 32) as u32, sub, before, after, 0]
}

unsafe fn barriers(list: *mut c_void, b: &[[u32; 8]]) {
    raw::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(list, RESOURCE_BARRIER)(list, b.len() as u32, b.as_ptr());
    super::depth_state::legacy_raw(list as usize, b);
}

const COPY_IN_2D: &str = r#"
cbuffer Part : register(b0) { uint mip; uint first_slice; };
Texture2DArray<float> depth : register(t0);
RWTexture2DArray<float> shadow : register(u0);
[RootSignature("RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint3 at = uint3(id.xy, id.z + first_slice);
    shadow[at] = depth.Load(int4(at, mip));
}
"#;

const COPY_IN_1D: &str = r#"
cbuffer Part : register(b0) { uint mip; uint first_slice; };
Texture1DArray<float> depth : register(t0);
RWTexture1DArray<float> shadow : register(u0);
[RootSignature("RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0))")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 at = uint2(id.x, id.y + first_slice);
    shadow[at] = depth.Load(int3(at, mip));
}
"#;

const COPY_OUT_VS: &str = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(float(id & 1) * 4.0 - 1.0, float(id >> 1) * 4.0 - 1.0, 0.0, 1.0); }";

const COPY_OUT_2D: &str = r#"
cbuffer Part : register(b0) { uint mip; uint slice; };
Texture2DArray<float> shadow : register(t0);
[RootSignature("RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0))")]
float main(float4 at : SV_Position) : SV_Depth {
    return shadow.Load(int4(int2(at.xy), slice, mip));
}
"#;

const COPY_OUT_1D: &str = r#"
cbuffer Part : register(b0) { uint mip; uint slice; };
Texture1DArray<float> shadow : register(t0);
[RootSignature("RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0))")]
float main(float4 at : SV_Position) : SV_Depth {
    return shadow.Load(int3(int(at.x), slice, mip));
}
"#;

/// (root signature, pipeline) of XCL's own passes, raw; None: could not be made (logged once)
type Pass = Option<(usize, usize)>;
static PASSES_IN: Mutex<[Option<Pass>; 2]> = Mutex::new([None, None]);
static PASSES_OUT: Mutex<Option<HashMap<(bool, i32), Pass>>> = Mutex::new(None);

unsafe fn pass_in(one_d: bool) -> Pass {
    let mut g = PASSES_IN.lock().unwrap();
    if let Some(p) = g[one_d as usize] {
        return p;
    }
    let made = (|| -> Result<(usize, usize), String> {
        let code = real::compile(if one_d { COPY_IN_1D } else { COPY_IN_2D }, "cs_5_1");
        if code.is_empty() {
            return Err("its HLSL did not compile".into());
        }
        let raw = real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
        super::title_textures::own_work(|| {
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &code).map_err(|e| format!("root signature: {e}"))?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(|e| format!("pipeline: {e}"))?;
            Ok((rs.into_raw() as usize, pso.into_raw() as usize))
        })
    })();
    let p = match made {
        Ok(p) => Some(p),
        Err(e) => {
            crate::xlog_warn!("D3D12.X: no pass copying a depth buffer to its shadow texture ({e}): unordered access reads what the shadow holds");
            None
        }
    };
    g[one_d as usize] = Some(p);
    p
}

unsafe fn pass_out(one_d: bool, depth_format: DXGI_FORMAT) -> Pass {
    let mut g = PASSES_OUT.lock().unwrap();
    if let Some(p) = g.get_or_insert_with(HashMap::new).get(&(one_d, depth_format.0)) {
        return *p;
    }
    let made = (|| -> Result<(usize, usize), String> {
        let (vs, ps) = (real::compile(COPY_OUT_VS, "vs_5_1"), real::compile(if one_d { COPY_OUT_1D } else { COPY_OUT_2D }, "ps_5_1"));
        if vs.is_empty() || ps.is_empty() {
            return Err("its HLSL did not compile".into());
        }
        let raw = real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
        super::title_textures::own_work(|| {
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &ps).map_err(|e| format!("root signature: {e}"))?;
            let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rs),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 0,
                DSVFormat: depth_format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            d.RasterizerState.DepthClipEnable = true.into();
            // every texel takes the shadow's value; the stencil plane is not touched
            d.DepthStencilState.DepthEnable = true.into();
            d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            d.DepthStencilState.StencilEnable = false.into();
            let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&d).map_err(|e| format!("pipeline: {e}"))?;
            Ok((rs.into_raw() as usize, pso.into_raw() as usize))
        })
    })();
    let p = match made {
        Ok(p) => Some(p),
        Err(e) => {
            crate::xlog_warn!("D3D12.X: no pass copying a shadow texture back to its depth buffer of format {} ({e}): what shaders wrote through unordered access stays in the shadow", depth_format.0);
            None
        }
    };
    g.get_or_insert_with(HashMap::new).insert((one_d, depth_format.0), p);
    p
}

/// Records on `list` CopyTextureRegion of each of `subs` whole, from `src` to `dst`.
unsafe fn copy_subs(list: *mut c_void, dst: usize, src: usize, subs: &[u32]) {
    let f = raw::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(list, COPY_TEXTURE_REGION);
    for &sub in subs {
        let at = |r: usize| D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&r),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        f(list, &at(dst), 0, 0, 0, &at(src), std::ptr::null());
    }
}

/// Records on `list` the copy of subresource `sub` (or ALL) of the depth buffer `depth` to its
/// shadow of the same format. The depth buffer's are in NON_PIXEL_SHADER_RESOURCE.
unsafe fn copy_in_exact(list: *mut c_void, s: &Shadow, depth: usize, sub: u32) -> Result<(), &'static str> {
    let subs: Vec<u32> = s.parts(sub).iter().map(|&(m, sl)| m + sl * s.mips).collect();
    if subs.is_empty() {
        return Err("no-depth-parts");
    }
    let t = s.texture.as_raw() as usize;
    let predicated = super::list_x::before_own_pass(list);
    barriers(list, &[transition(depth, sub, SHADER_READ, COPY_SOURCE), transition(t, sub, UNORDERED_ACCESS, COPY_DEST)]);
    copy_subs(list, t, depth, &subs);
    barriers(list, &[transition(depth, sub, COPY_SOURCE, SHADER_READ), transition(t, sub, COPY_DEST, UNORDERED_ACCESS)]);
    super::list_x::after_own_pass(list, false, predicated);
    COPIES_IN.fetch_add(1, Ordering::Relaxed);
    s.copies_in.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Records on `list` the copy of the depth plane's `runs` to the shadow. The depth buffer's
/// are in NON_PIXEL_SHADER_RESOURCE.
unsafe fn copy_in(list: *mut c_void, s: &Shadow, runs: &[(u32, u32, u32)]) -> Result<(), &'static str> {
    if runs.is_empty() {
        return Err("no-depth-parts");
    }
    let (rs, pso) = pass_in(s.one_d).ok_or("compute-pass-unavailable")?;
    let predicated = super::list_x::before_own_pass(list);
    let heaps = [s.views_heap];
    raw::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(list, SET_DESCRIPTOR_HEAPS)(list, 1, heaps.as_ptr());
    raw::<unsafe extern "system" fn(*mut c_void, usize)>(list, SET_COMPUTE_ROOT_SIGNATURE)(list, rs);
    raw::<unsafe extern "system" fn(*mut c_void, usize)>(list, SET_PIPELINE_STATE)(list, pso);
    let table = raw::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, SET_COMPUTE_ROOT_DESCRIPTOR_TABLE);
    let constants = raw::<unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void, u32)>(list, SET_COMPUTE_ROOT_CONSTANTS);
    let dispatch = raw::<unsafe extern "system" fn(*mut c_void, u32, u32, u32)>(list, DISPATCH);
    table(list, 1, s.view(VIEW_DEPTH));
    for &(mip, first, slices) in runs {
        let part = [mip, first];
        constants(list, 0, 2, part.as_ptr().cast(), 0);
        table(list, 2, s.view(VIEW_MIPS + mip));
        let (w, h) = s.size(mip);
        if s.one_d {
            dispatch(list, w.div_ceil(64), slices, 1);
        } else {
            dispatch(list, w.div_ceil(8), h.div_ceil(8), slices);
        }
    }
    // what the title's shaders read and write next comes after these writes
    let t = s.texture.as_raw() as usize;
    barriers(list, &[[2, 0, t as u32, (t as u64 >> 32) as u32, 0, 0, 0, 0]]);
    super::list_x::after_own_pass(list, false, predicated);
    COPIES_IN.fetch_add(1, Ordering::Relaxed);
    s.copies_in.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// The state the PC depth buffer is in for a copy back from its shadow.
fn out_state(s: &Shadow) -> u32 {
    if s.exact {
        COPY_DEST
    } else {
        DEPTH_WRITE
    }
}

/// A copy back is recorded.
fn copied_out(s: &Shadow) {
    COPIES_OUT.fetch_add(1, Ordering::Relaxed);
    s.copies_out.fetch_add(1, Ordering::Relaxed);
    if !s.written.swap(true, Ordering::Relaxed) {
        let n = WRITTEN.fetch_add(1, Ordering::Relaxed) + 1;
        crate::xlog_info!("D3D12.X: depth buffer {}x{} ({} slices, {} mips): the first copy of its shadow texture back to the depth plane ({n} depth buffers written back so far)", s.width, s.height, s.slices, s.mips);
    }
}

/// Records on `list` the copy of the shadow's `parts` back to the depth plane of `depth`, whose
/// are in `out_state`: CopyTextureRegion (any list type), or the pass (a direct list).
unsafe fn copy_out(list: *mut c_void, s: &Shadow, depth: usize, parts: &[(u32, u32)]) -> Result<(), &'static str> {
    if parts.is_empty() {
        return Err("no-depth-parts");
    }
    let t = s.texture.as_raw() as usize;
    if s.exact {
        let subs: Vec<u32> = parts.iter().map(|&(m, sl)| m + sl * s.mips).collect();
        let read: Vec<[u32; 8]> = subs.iter().map(|&k| transition(t, k, UNORDERED_ACCESS, COPY_SOURCE)).collect();
        let written: Vec<[u32; 8]> = subs.iter().map(|&k| transition(t, k, COPY_SOURCE, UNORDERED_ACCESS)).collect();
        let predicated = super::list_x::before_own_pass(list);
        barriers(list, &read);
        copy_subs(list, depth, t, &subs);
        barriers(list, &written);
        super::list_x::after_own_pass(list, false, predicated);
        copied_out(s);
        return Ok(());
    }
    let (rs, pso) = pass_out(s.one_d, s.depth_format).ok_or("graphics-pass-unavailable")?;
    // (the pass's view is of the whole shadow)
    let read = [transition(t, ALL, UNORDERED_ACCESS, PIXEL_READ)];
    let written = [transition(t, ALL, PIXEL_READ, UNORDERED_ACCESS)];
    barriers(list, &read);
    let predicated = super::list_x::before_own_pass(list);
    let heaps = [s.views_heap];
    raw::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(list, SET_DESCRIPTOR_HEAPS)(list, 1, heaps.as_ptr());
    raw::<unsafe extern "system" fn(*mut c_void, usize)>(list, SET_GRAPHICS_ROOT_SIGNATURE)(list, rs);
    raw::<unsafe extern "system" fn(*mut c_void, usize)>(list, SET_PIPELINE_STATE)(list, pso);
    raw::<unsafe extern "system" fn(*mut c_void, u32)>(list, IA_SET_PRIMITIVE_TOPOLOGY)(list, 4);
    raw::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE)(list, 1, s.view(VIEW_SHADOW));
    let constants = raw::<unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void, u32)>(list, SET_GRAPHICS_ROOT_CONSTANTS);
    let viewports = raw::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(list, RS_SET_VIEWPORTS);
    let scissors = raw::<unsafe extern "system" fn(*mut c_void, u32, *const RECT)>(list, RS_SET_SCISSOR_RECTS);
    let targets = raw::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(list, OM_SET_RENDER_TARGETS);
    let draw = raw::<unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32)>(list, DRAW_INSTANCED);
    for &(mip, slice) in parts {
        let (w, h) = s.size(mip);
        let viewport = D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 };
        let scissor = RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 };
        let target = s.target(mip, slice);
        let part = [mip, slice];
        viewports(list, 1, &viewport);
        scissors(list, 1, &scissor);
        targets(list, 0, std::ptr::null(), 0, &target);
        constants(list, 0, 2, part.as_ptr().cast(), 0);
        draw(list, 3, 1, 0, 0);
    }
    barriers(list, &written);
    super::list_x::after_own_pass(list, true, predicated);
    copied_out(s);
    Ok(())
}

fn copy_event(
    res: usize, list: usize, range: Range, into_twin: bool, result: Result<(), &'static str>, path: &'static str,
) {
    let kind = match (into_twin, result.is_ok()) {
        (true, true) => "copy-in-recorded",
        (true, false) => "copy-in-skipped",
        (false, true) => "copy-out-recorded",
        (false, false) => "copy-out-skipped",
    };
    depth_lineage::note(res, list, range, kind, || {
        format!("path={path} reason={}", result.err().unwrap_or("recorded"))
    });
}

const LINEAGE_PARTS: usize = 16;

fn note_copy() {
    let n = COPIES_IN.load(Ordering::Relaxed) + COPIES_OUT.load(Ordering::Relaxed);
    if n.is_power_of_two() {
        log_counts("copies so far");
    }
}

// ---------------------------------------------------------------------------------------
// the title's transitions
// ---------------------------------------------------------------------------------------

/// A copy to the shadow that follows the barriers being gathered (`after_barriers`).
pub(super) struct Later {
    shadow: Arc<Shadow>,
    depth: usize,
    sub: u32,
    guest_sub: u32,
    runs: Vec<(u32, u32, u32)>,
}

/// The title's transition `b` on `list` (`transition_uav`), with what it says of the state a
/// depth buffer with a shadow is in (`XCL_DEPTH_UAV_SYNC`): the depth plane is used before it –
/// a shadow ahead of it is copied into it first – and is in the state after it then.
pub(super) unsafe fn on_transition(list: *mut c_void, b: &mut [u32; 8], later: &mut Vec<Later>) -> bool {
    let res = (b[2] as u64 | (b[3] as u64) << 32) as usize;
    let (sub, before, after) = (b[4], pc_state(b[5]), pc_state(b[6]));
    let begin = b[1] & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY.0 as u32 != 0;
    let names_uav = (b[5] | b[6]) & UNORDERED_ACCESS != 0;
    let shadow = if res != 0 && sync() { known(res).flatten() } else { None };
    if let Some(s) = shadow.as_ref().filter(|_| !names_uav && !begin) {
        use_side(list, res, s, Side::Plane, false, &plane_subs(s, sub), Some(before));
    }
    let kept = transition_uav(list, b, later);
    if let Some(s) = &shadow {
        let state = if begin { UNKNOWN } else { after };
        let mut g = RECORDINGS.lock().unwrap();
        let u = recording(&mut g, list).use_of(res, s);
        for k in plane_subs(s, sub) {
            if let Some(x) = u.states.get_mut(k as usize) {
                *x = state;
            }
        }
        if names_uav && !begin {
            u.ahead = Ahead::Neither;
        }
    }
    kept
}

/// The title's transition `b` (PC state bits only) on `list`, when it names UNORDERED_ACCESS or
/// depth buffers have shadows: for a depth buffer with a shadow the states become the PC's
/// (`pc_state`), a copy back to the depth plane is recorded now – before the barrier – and a copy
/// to the shadow is left in `later`. False: the barrier is dropped.
unsafe fn transition_uav(list: *mut c_void, b: &mut [u32; 8], later: &mut Vec<Later>) -> bool {
    let res = (b[2] as u64 | (b[3] as u64) << 32) as usize;
    let (before, after) = (b[5], b[6]);
    let names_uav = (before | after) & UNORDERED_ACCESS != 0;
    if res == 0 {
        return true;
    }
    stencil::on_transition(list, res, b);
    let list_key = list as usize;
    let sub = b[4];
    let flags = b[1];
    let range = legacy_range(sub);
    let (was, is) = (before & UNORDERED_ACCESS != 0, after & UNORDERED_ACCESS != 0);
    let egress = was && !is && flags & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY.0 as u32 == 0;
    let transition_kind = if egress { "guest-legacy-uav-egress" } else { "guest-legacy-transition" };
    let guest_transition = || depth_lineage::note(res, list_key, range, transition_kind, || {
        format!("before={before:#x} after={after:#x} flags={flags:#x}")
    });
    let s = match known(res) {
        Some(Some(s)) => s,
        // a depth buffer no shadow could be made for, or – not known yet – one the title moves
        // to unordered access before it makes a view
        other => {
            if !names_uav {
                return true;
            }
            let p = res as *mut c_void;
            let Some(rd) = ID3D12Resource::from_raw_borrowed(&p).map(|r| r.GetDesc()) else { return true };
            if !rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) || rd.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                return true;
            }
            match if other.is_some() { None } else { shadow_of(p, &rd) } {
                Some(s) => s,
                None => {
                    guest_transition();
                    if was != is {
                        copy_event(res, list_key, range, is, Err("no-twin"), "legacy");
                    }
                    // states the PC depth buffer has, at least
                    (b[5], b[6]) = (pc_state(before), pc_state(after));
                    return b[5] != b[6];
                }
            }
        }
    };
    guest_transition();
    // the two halves of a split barrier: the work is done at its end
    if flags & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY.0 as u32 != 0 {
        if was != is {
            copy_event(res, list_key, range, is, Err("split-begin"), "legacy");
        }
        return false;
    }
    b[1] = 0;
    let (pc_before, pc_after) = (pc_state(before), pc_state(after));
    let ty = raw::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, GET_TYPE)(list);
    let direct = ty == D3D12_COMMAND_LIST_TYPE_DIRECT;
    let index = |&(m, sl): &(u32, u32)| m + sl * s.mips;
    let parts = s.parts(sub);
    let copy_range = depth_range(&s, sub);
    let out_path = if s.exact { "texture-copy" } else { "graphics-pass" };
    // the copies back that lists which cannot draw left for these subresources
    let owed: Vec<(u32, u32)> = {
        let mut g = s.owed.lock().unwrap();
        if g.is_empty() || !direct {
            Vec::new()
        } else {
            let hit: Vec<(u32, u32)> = parts.iter().copied().filter(|p| g.contains(&index(p))).collect();
            g.retain(|o| !hit.iter().any(|p| index(p) == *o));
            hit
        }
    };
    if was && !is {
        let needed = !owed.is_empty() || !in_step(list, res, sub);
        if needed && (direct || s.exact) {
            let held = out_state(&s);
            barriers(list, &[transition(res, sub, pc_before, held)]);
            let copied = copy_out(list, &s, res, &parts);
            copy_event(res, list_key, copy_range, false, copied, out_path);
            if copied.is_ok() {
                mark_in_step(list, res, sub);
                note_copy();
            }
            (b[5], b[6]) = (held, pc_after);
            return b[5] != b[6];
        }
        if needed {
            // this list cannot draw: the next direct list that names the depth buffer does
            let mut g = s.owed.lock().unwrap();
            for p in &parts {
                if !g.contains(&index(p)) {
                    g.push(index(p));
                    COPIES_OWED.fetch_add(1, Ordering::Relaxed);
                }
            }
            drop(g);
            copy_event(res, list_key, copy_range, false, Err(if parts.is_empty() { "no-depth-parts" } else { "deferred-non-direct-list" }), out_path);
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: a depth buffer leaves unordered access on a command list of type {} (no draws there): its shadow texture is copied back by the next direct list that names it in a transition", ty.0);
            }
        } else {
            if !parts.is_empty() {
                COPIES_OUT_LEFT_OUT.fetch_add(1, Ordering::Relaxed);
            }
            copy_event(res, list_key, copy_range, false, Err(if parts.is_empty() { "no-depth-parts" } else { "already-in-step" }), out_path);
        }
    } else if !owed.is_empty() {
        // not leaving unordered access here, but the depth plane is behind its shadow
        let read: Vec<[u32; 8]> = owed.iter().map(|p| transition(res, index(p), pc_before, DEPTH_WRITE)).collect();
        let back: Vec<[u32; 8]> = owed.iter().map(|p| transition(res, index(p), DEPTH_WRITE, pc_before)).collect();
        if pc_before != DEPTH_WRITE {
            barriers(list, &read);
        }
        let copied = copy_out(list, &s, res, &owed);
        if depth_lineage::enabled() {
            if owed == parts {
                copy_event(res, list_key, copy_range, false, copied, "owed-flush");
            } else {
                for p in owed.iter().take(LINEAGE_PARTS) {
                    copy_event(res, list_key, Range::One(index(p)), false, copied, "owed-flush");
                }
                if owed.len() > LINEAGE_PARTS {
                    depth_lineage::note(res, list_key, copy_range, "copy-out-range-truncated", || {
                        format!("path=owed-flush omitted_parts={} result={copied:?}", owed.len() - LINEAGE_PARTS)
                    });
                }
            }
        }
        if copied.is_ok() {
            note_copy();
        }
        if pc_before != DEPTH_WRITE {
            barriers(list, &back);
        }
    }
    if is && !was && matches!(ty, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        // what a list that cannot draw left owed is newer in the shadow than in the depth plane
        let behind: Vec<u32> = s.owed.lock().unwrap().clone();
        let runs: Vec<(u32, u32, u32)> = if behind.is_empty() { s.runs(sub) } else { parts.iter().filter(|p| !behind.contains(&index(p))).map(|&(m, sl)| (m, sl, 1)).collect() };
        if in_step(list, res, sub) {
            if !parts.is_empty() {
                COPIES_IN_LEFT_OUT.fetch_add(1, Ordering::Relaxed);
            }
            copy_event(res, list_key, copy_range, true, Err(if parts.is_empty() { "no-depth-parts" } else { "already-in-step" }), "legacy");
        } else if !runs.is_empty() {
            later.push(Later { shadow: s.clone(), depth: res, sub: if behind.is_empty() { sub } else { u32::MAX - 1 }, guest_sub: sub, runs });
        } else {
            copy_event(res, list_key, copy_range, true, Err(if parts.is_empty() { "no-depth-parts" } else { "depth-older-than-twin" }), "legacy");
        }
    } else if is && !was {
        copy_event(res, list_key, copy_range, true, Err("unsupported-list-type"), "legacy");
    }
    (b[5], b[6]) = (pc_before, pc_after);
    b[5] != b[6]
}

/// The barriers gathered are on `list`: the copies to shadows that follow them.
pub(super) unsafe fn after_barriers(list: *mut c_void, later: Vec<Later>) {
    for l in later {
        let copied = if l.shadow.exact { copy_in_exact(list, &l.shadow, l.depth, l.sub) } else { copy_in(list, &l.shadow, &l.runs) };
        if depth_lineage::enabled() {
            let list_key = list as usize;
            let range = depth_range(&l.shadow, l.guest_sub);
            if l.sub != u32::MAX - 1 {
                copy_event(l.depth, list_key, range, true, copied, if l.shadow.exact { "texture-copy" } else { "compute-pass" });
            } else if l.shadow.exact {
                copy_event(l.depth, list_key, range, true, Err("exact-partial-copy-unavailable"), "texture-copy");
            } else {
                for &(mip, first, slices) in l.runs.iter().take(LINEAGE_PARTS) {
                    let part = Range::Rect {
                        ranges: [(mip, 1), (first, slices), (0, 1)],
                        mips: l.shadow.mips,
                        slices: l.shadow.slices,
                    };
                    copy_event(l.depth, list_key, part, true, copied, "partial-compute-pass");
                }
                depth_lineage::note(l.depth, list_key, range, "copy-in-partial", || {
                    format!("path=compute-pass excluded=owed-to-depth omitted_runs={} result={copied:?}", l.runs.len().saturating_sub(LINEAGE_PARTS))
                });
            }
        }
        if copied.is_ok() {
            // (a copy that left out subresources behind their shadow says nothing of the whole)
            if l.sub != u32::MAX - 1 {
                mark_in_step(list, l.depth, l.sub);
            }
            note_copy();
        }
    }
}

// ---------------------------------------------------------------------------------------
// in step by use (`XCL_DEPTH_UAV_SYNC`)
// ---------------------------------------------------------------------------------------

/// Whether shadows are also kept in step by use (`XCL_DEPTH_UAV_SYNC=0`: only at the title's
/// transitions to and from UNORDERED_ACCESS).
fn sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_DEPTH_UAV_SYNC").is_ok_and(|v| v.trim() == "0"))
}

/// Which side of a shadow holds what the other lacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Ahead {
    #[default]
    Neither = 0,
    Shadow = 1,
    Plane = 2,
}

impl Ahead {
    fn of(v: u8) -> Self {
        match v {
            1 => Self::Shadow,
            2 => Self::Plane,
            _ => Self::Neither,
        }
    }
}

/// The side of a shadow a command uses: the shadow, or the depth plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Shadow,
    Plane,
}

impl Side {
    /// Whether a use of this side needs the other copied into it first.
    fn behind(self, ahead: Ahead) -> bool {
        matches!((self, ahead), (Side::Shadow, Ahead::Plane) | (Side::Plane, Ahead::Shadow))
    }

    /// The side ahead after a use of this side that changes it (`write`) or not, the copy it
    /// needed first made (`copied`) or not.
    fn after(self, ahead: Ahead, write: bool, copied: bool) -> Ahead {
        match (write, self) {
            (true, Side::Shadow) => Ahead::Shadow,
            (true, Side::Plane) => Ahead::Plane,
            (false, _) if copied => Ahead::Neither,
            (false, _) => ahead,
        }
    }
}

/// a depth plane subresource whose state on a list is not known
const UNKNOWN: u32 = u32::MAX;

/// What one recording did with one shadow: the side it left ahead, and the state each of the
/// depth plane's subresources is in on it (`UNKNOWN`).
struct Use {
    depth: usize,
    shadow: Arc<Shadow>,
    ahead: Ahead,
    states: Vec<u32>,
}

impl Recording {
    /// What the recording did with the shadow `s` of `depth`; at its first use, the side a list
    /// that closed before it left ahead.
    fn use_of(&mut self, depth: usize, s: &Arc<Shadow>) -> &mut Use {
        let i = match self.uses.iter().position(|u| u.depth == depth && Arc::ptr_eq(&u.shadow, s)) {
            Some(i) => i,
            None => {
                self.uses.push(Use {
                    depth,
                    shadow: s.clone(),
                    ahead: Ahead::of(s.ahead.load(Ordering::Relaxed)),
                    states: vec![UNKNOWN; (s.mips * s.slices) as usize],
                });
                self.uses.len() - 1
            }
        };
        &mut self.uses[i]
    }
}

/// The recording of `list` in `RECORDINGS`, begun now if it has none.
fn recording(g: &mut Option<HashMap<usize, Recording>>, list: *mut c_void) -> &mut Recording {
    g.get_or_insert_with(HashMap::new).entry(list as usize).or_insert_with(|| {
        RECORDING.fetch_add(1, Ordering::AcqRel);
        Recording::default()
    })
}

/// The depth plane's subresource indices of the title's subresource index `sub` (all for ALL).
fn plane_subs(s: &Shadow, sub: u32) -> Vec<u32> {
    s.parts(sub).iter().map(|&(m, sl)| m + sl * s.mips).collect()
}

/// A depth-stencil view the title made: of which depth buffer, its subresources, and whether
/// it only reads the depth plane.
#[derive(Clone, Copy)]
struct Dsv {
    depth: usize,
    mip: u32,
    first: u32,
    count: u32,
    read_only: bool,
}

impl Dsv {
    /// The depth plane's subresources the view names, of a depth buffer with the shadow `s`.
    fn subs(&self, s: &Shadow) -> Vec<u32> {
        if self.mip >= s.mips {
            return Vec::new();
        }
        (self.first..self.first.saturating_add(self.count).min(s.slices)).map(|sl| self.mip + sl * s.mips).collect()
    }
}

/// depth-stencil view handle → the view (`XCL_DEPTH_UAV_SYNC`)
static DSVS: Mutex<Option<HashMap<usize, Dsv>>> = Mutex::new(None);

fn dsv_view(handle: usize) -> Option<Dsv> {
    DSVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

/// The title makes the depth-stencil view `desc` (None: the default one) of `res` at `handle`
/// (`real::dev_create_dsv_x`).
pub(super) unsafe fn note_dsv(handle: usize, res: *mut c_void, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) {
    if !sync() && !probe() {
        return;
    }
    let view = (|| {
        let rd = ID3D12Resource::from_raw_borrowed(&res)?.GetDesc();
        let (mip, first, count) = match desc {
            None => (0, 0, u32::from(rd.DepthOrArraySize)),
            Some(d) => match d.ViewDimension {
                D3D12_DSV_DIMENSION_TEXTURE1D => (d.Anonymous.Texture1D.MipSlice, 0, 1),
                D3D12_DSV_DIMENSION_TEXTURE2D => (d.Anonymous.Texture2D.MipSlice, 0, 1),
                D3D12_DSV_DIMENSION_TEXTURE1DARRAY => {
                    let v = d.Anonymous.Texture1DArray;
                    (v.MipSlice, v.FirstArraySlice, v.ArraySize)
                }
                D3D12_DSV_DIMENSION_TEXTURE2DARRAY => {
                    let v = d.Anonymous.Texture2DArray;
                    (v.MipSlice, v.FirstArraySlice, v.ArraySize)
                }
                _ => return None,
            },
        };
        let read_only = desc.is_some_and(|d| d.Flags.0 & D3D12_DSV_FLAG_READ_ONLY_DEPTH.0 != 0);
        Some(Dsv { depth: res as usize, mip, first, count, read_only })
    })();
    let mut g = DSVS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    match view {
        Some(v) => {
            m.insert(handle, v);
        }
        None => {
            m.remove(&handle);
        }
    }
}

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    raw::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, GET_TYPE)(list)
}

/// Records on `list` (of type `ty`) the copy of the shadow `s` into the depth plane of `depth`
/// (`to` the plane) or of the plane into the shadow, the plane's subresources moved from their
/// `states` to the state the copy needs and back.
unsafe fn copy_to(list: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE, depth: usize, s: &Shadow, to: Side, states: &[u32]) -> Result<(), &'static str> {
    if states.contains(&UNKNOWN) {
        return Err("depth-state-unknown");
    }
    if to == Side::Plane && !s.exact && ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return Err("deferred-non-direct-list");
    }
    let held = if to == Side::Plane { out_state(s) } else { SHADER_READ };
    let moves = |there: bool| -> Vec<[u32; 8]> {
        states
            .iter()
            .enumerate()
            .filter(|&(_, &st)| st != held)
            .map(|(k, &st)| if there { transition(depth, k as u32, st, held) } else { transition(depth, k as u32, held, st) })
            .collect()
    };
    let (go, back) = (moves(true), moves(false));
    if !go.is_empty() {
        barriers(list, &go);
    }
    let copied = match to {
        Side::Plane => copy_out(list, s, depth, &s.parts(ALL)),
        Side::Shadow if s.exact => copy_in_exact(list, s, depth, ALL),
        Side::Shadow => copy_in(list, s, &s.runs(ALL)),
    };
    if !back.is_empty() {
        barriers(list, &back);
    }
    copied?;
    let counter = if to == Side::Plane { &USE_OUT } else { &USE_IN };
    counter.fetch_add(1, Ordering::Relaxed);
    static TOLD: AtomicBool = AtomicBool::new(false);
    if !TOLD.swap(true, Ordering::Relaxed) {
        crate::xlog_info!("D3D12.X: depth-UAV shadows kept in step by use (XCL_DEPTH_UAV_SYNC=0: only at UNORDERED_ACCESS transitions)");
    }
    note_copy();
    Ok(())
}

/// A command about to be recorded on `list` uses `side` of the shadow `s` of `depth`, changing
/// it (`write`) or not; for it the depth plane's subresources `subs` are in `state` (None: as
/// the list left them). The side used is first copied from the other when that is ahead – and a
/// shadow no copy ever reached (it holds nothing of the plane) is copied from the plane first,
/// whatever the model says of it.
unsafe fn use_side(list: *mut c_void, depth: usize, s: &Arc<Shadow>, side: Side, write: bool, subs: &[u32], state: Option<u32>) {
    let ty = list_type(list);
    if !matches!(ty, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        return;
    }
    let (behind, states) = {
        let mut g = RECORDINGS.lock().unwrap();
        let u = recording(&mut g, list).use_of(depth, s);
        if let Some(st) = state {
            for &k in subs {
                if let Some(x) = u.states.get_mut(k as usize) {
                    *x = st;
                }
            }
        }
        let behind = side.behind(u.ahead) || (side == Side::Shadow && s.copies_in.load(Ordering::Relaxed) == 0);
        (behind, behind.then(|| u.states.clone()))
    };
    let copied = states.is_some_and(|st| copy_to(list, ty, depth, s, side, &st).is_ok());
    if copied {
        mark_in_step(list, depth, ALL);
    }
    let mut g = RECORDINGS.lock().unwrap();
    let u = recording(&mut g, list).use_of(depth, s);
    // a use of the shadow that needed the plane and did not get it says nothing of the plane:
    // what the shadow holds is not known to be the plane's
    if side != Side::Shadow || !behind || copied {
        u.ahead = side.after(u.ahead, write, copied);
    }
}

/// The depth buffers with shadows whose unordered access views the tables of `layout` name in
/// `args` (the first 64 views).
fn named_shadows(layout: &super::console_list::Layout, args: &[u8]) -> Vec<(usize, Arc<Shadow>)> {
    let mut out: Vec<(usize, Arc<Shadow>)> = Vec::new();
    let mut left = 64u32;
    for b in &layout.uavs {
        let super::console_list::UavBinding::Table { at, first, count, .. } = *b else { continue };
        let Some(t) = args.get(at as usize..at as usize + 4).map(|w| u32::from_le_bytes(w.try_into().unwrap())) else { continue };
        if t == 0 {
            continue;
        }
        let n = count.min(left);
        left -= n;
        for k in 0..n {
            let addr = t.wrapping_add(super::console_heap::SLOT.wrapping_mul(first.wrapping_add(k)));
            let Some(w) = super::console_heap::words_at(addr) else { break };
            if w[3] >> 28 < 8 {
                continue;
            }
            let Some(res) = super::console_heap::image_resource_at(addr, w[0]) else { continue };
            if out.iter().any(|(d, _)| *d == res) {
                continue;
            }
            if let Some(Some(s)) = known(res) {
                out.push((res, s));
            }
        }
        if left == 0 {
            break;
        }
    }
    out
}

/// A draw (`graphics`) or dispatch about to be bound on `list` with the root arguments `args`
/// of `layout` (`bind::bind_abi`): it writes the shadows its unordered access tables name, and a
/// draw uses the depth plane of its depth target (writes it, but through a read-only view).
pub(super) unsafe fn before_op(list: *mut c_void, graphics: bool, layout: Option<&super::console_list::Layout>, args: &[u8]) {
    if !active() || !sync() {
        return;
    }
    if graphics {
        if let Some(d) = super::list_x::bound_dsv(list).and_then(dsv_view) {
            if let Some(Some(s)) = known(d.depth) {
                use_side(list, d.depth, &s, Side::Plane, !d.read_only, &d.subs(&s), (!d.read_only).then_some(DEPTH_WRITE));
            }
        }
    }
    let Some(layout) = layout else { return };
    for (depth, s) in named_shadows(layout, args) {
        use_side(list, depth, &s, Side::Shadow, true, &[], None);
    }
}

// ---------------------------------------------------------------------------------------
// the probe (`XCL_SHADOW_PROBE`)
// ---------------------------------------------------------------------------------------

/// shadow textures a frame's passes are reported for, lines a frame prints of them, and view
/// slots whose resource is remembered
const PROBE_SHADOWS: usize = 512;
const PROBE_LINES: usize = 6;
const PROBE_OWNERS: usize = 4096;
/// descriptor slots a pass's tables are scanned for, and of one table
const PROBE_SLOTS: u32 = 256;
const PROBE_TABLE_SLOTS: u32 = 64;

/// Whether the probe runs (`XCL_SHADOW_PROBE=1`).
fn probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_SHADOW_PROBE").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!(
                "D3D12.X: shadow probe on (XCL_SHADOW_PROBE=1): once a frame, a line for each shadow texture a shader pass sampled, with the copies it received and sent back, the depth plane's clear value and the pass that sampled it (a depth buffer a pass names in a root descriptor instead of a table is not seen)"
            );
        }
        on
    })
}

/// Whether the mapping from each view slot to its resource is kept for the use-based sync
/// (`XCL_DEPTH_UAV_OWNERS=1`): without it `named_shadows` resolves no depth buffer, and no draw
/// or dispatch counts as a use of a shadow (so nothing is copied in or back by use).
pub(super) fn sync_owners() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_DEPTH_UAV_OWNERS").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!(
                "D3D12.X: depth-UAV shadows kept in step by use, with the mapping from each view slot to its resource (XCL_DEPTH_UAV_OWNERS=1; XCL_DEPTH_UAV_SYNC=0 turns the copies off again)"
            );
        }
        on
    })
}

/// Of one shadow a frame's passes sampled: how, by which pass first, and how often.
struct Sampled {
    shadow: Weak<Shadow>,
    depth: usize,
    via: &'static str,
    pass: String,
    calls: u64,
}

static PROBE: Mutex<Option<HashMap<usize, Sampled>>> = Mutex::new(None);
/// the view slot → the resource its view was made of (`probe_slot`)
static PROBED: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
/// the last depth clear value the title gave each depth buffer
static PROBE_CLEARS: Mutex<Option<HashMap<usize, f32>>> = Mutex::new(None);

/// The view in the slot at `at` names `res` (`console_heap::note_resource`): from here on, a pass
/// whose tables hold that slot sampled the shadow of `res`.
pub(super) fn probe_slot(at: usize, res: usize) {
    if !probe() || known(res).is_none() {
        return;
    }
    let mut g = PROBED.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() < PROBE_OWNERS || m.contains_key(&at) {
        m.insert(at, res);
    }
}

/// The depth buffers the descriptor tables of `layout` name in `args`, and the view each is
/// named through (its unordered access view, or a shader view of it) – the candidates for a
/// shadow; `probe_op` keeps those that have one.
fn probe_slots(layout: &super::console_list::Layout, args: &[u8]) -> Vec<(usize, &'static str)> {
    let mut out: Vec<(usize, &'static str)> = Vec::new();
    let Some(u) = crate::gpu::uma::get() else { return out };
    let owners = PROBED.lock().unwrap();
    let Some(owners) = owners.as_ref() else { return out };
    let mut left = PROBE_SLOTS;
    for (k, p) in layout.params.iter().enumerate() {
        let super::console_list::Param::Table { at } = *p else { continue };
        let Some(bytes) = args.get(at as usize..at as usize + 4) else { continue };
        let table = u32::from_le_bytes(bytes.try_into().unwrap());
        if table == 0 {
            continue;
        }
        let n = layout.tables.get(k).copied().unwrap_or(0).min(left).min(PROBE_TABLE_SLOTS);
        left -= n;
        for i in 0..n {
            let addr = table.wrapping_add(super::console_heap::SLOT.wrapping_mul(i));
            let Some(s) = super::console_heap::lookup((u.base() + addr as usize) as u64) else { break };
            if s.sampler {
                continue;
            }
            let w = unsafe { (s.at as *const [u32; 8]).read_unaligned() };
            let ty = w[3] >> 28;
            // buffer, constant and sampler slots are no image descriptor
            if ty < 4 {
                continue;
            }
            let Some(&res) = owners.get(&s.at) else { continue };
            if out.iter().any(|&(d, _)| d == res) {
                continue;
            }
            out.push((res, if ty >= 8 { "a UAV" } else { "a shader view" }));
        }
        if left == 0 {
            break;
        }
    }
    out
}

/// A draw or dispatch of `pso` (`call`) about to be recorded with the root arguments `args` of
/// `layout`: the shadows of the depth buffers its tables name are what it samples this frame.
pub(super) fn probe_op(pso: usize, call: &'static str, layout: Option<&super::console_list::Layout>, args: &[u8]) {
    let Some(layout) = layout.filter(|_| probe() && active()) else { return };
    let named = probe_slots(layout, args);
    if named.is_empty() {
        return;
    }
    let shadows: Vec<(usize, Arc<Shadow>, &'static str)> = {
        let g = SHADOWS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        named.into_iter().filter_map(|(res, via)| m.get(&res).and_then(|s| s.as_ref()).map(|s| (res, s.clone(), via))).collect()
    };
    if shadows.is_empty() {
        return;
    }
    let pass = match super::real::pso_name(pso) {
        Some(name) => format!("\"{name}\" ({call})"),
        None => format!("an unnamed pipeline ({call})"),
    };
    let mut g = PROBE.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for (depth, s, via) in shadows {
        let twin = s.texture.as_raw() as usize;
        if m.len() >= PROBE_SHADOWS && !m.contains_key(&twin) {
            continue;
        }
        let e = m.entry(twin).or_insert_with(|| Sampled { shadow: Arc::downgrade(&s), depth, via, pass: pass.clone(), calls: 0 });
        e.calls += 1;
    }
}

/// The title clears the depth-stencil view `handle` with `depth` (`flags` 1: the depth plane):
/// the value the probe reports for that depth buffer.
pub(super) fn probe_clear(handle: usize, flags: u32, depth: f32) {
    if !probe() || flags & 1 == 0 {
        return;
    }
    let Some(d) = dsv_view(handle) else { return };
    PROBE_CLEARS.lock().unwrap().get_or_insert_with(HashMap::new).insert(d.depth, depth);
}

/// A frame was presented: what its passes sampled of each shadow, one line each
/// (`XCL_SHADOW_PROBE`).
pub(super) fn probe_frame() {
    if !probe() {
        return;
    }
    let Some(sampled) = PROBE.lock().unwrap().take() else { return };
    let clears = PROBE_CLEARS.lock().unwrap();
    let mut entries: Vec<(usize, Sampled)> = sampled.into_iter().collect();
    entries.sort_by_key(|e| e.0);
    let total = entries.len();
    let mut shown = 0usize;
    for (twin, s) in entries {
        if shown >= PROBE_LINES {
            break;
        }
        let Some(shadow) = s.shadow.upgrade() else { continue };
        shown += 1;
        let clear = clears.as_ref().and_then(|m| m.get(&s.depth)).map_or_else(|| "none recorded".to_string(), |&v| format!("{v:.6}"));
        let ahead = match Ahead::of(shadow.ahead.load(Ordering::Relaxed)) {
            Ahead::Shadow => "the shadow texture",
            Ahead::Plane => "the depth plane",
            Ahead::Neither => "neither side",
        };
        crate::xlog_info!(
            "D3D12.X: shadow probe: shadow #{} {twin:#x} {}x{} ({} slices, {} mips, format {}) sampled {} time(s) this frame through {}, by {}; {} copies in, {} back; plane clear {clear}; the model has {ahead} ahead",
            shadow.index, shadow.width, shadow.height, shadow.slices, shadow.mips, shadow.depth_format.0,
            s.calls, s.via, s.pass, shadow.copies_in.load(Ordering::Relaxed), shadow.copies_out.load(Ordering::Relaxed),
        );
    }
    if total > shown {
        crate::xlog_info!("D3D12.X: shadow probe: {} more shadow texture(s) sampled this frame, not listed", total - shown);
    }
}

/// The title clears the depth-stencil view `handle` on `list` (`flags` 1: the depth plane).
pub(super) unsafe fn on_clear(list: *mut c_void, handle: usize, flags: u32) {
    if !active() || !sync() || flags & 1 == 0 {
        return;
    }
    let Some(d) = dsv_view(handle) else { return };
    let Some(Some(s)) = known(d.depth) else { return };
    use_side(list, d.depth, &s, Side::Plane, true, &d.subs(&s), Some(DEPTH_WRITE));
}

/// The title copies (or resolves) subresource `src_sub` of `src` into `dst_sub` of `dst` on
/// `list` (ALL: whole resources; 0 for no texture).
pub(super) unsafe fn on_copy(list: *mut c_void, dst: usize, dst_sub: u32, src: usize, src_sub: u32) {
    if !active() || !sync() {
        return;
    }
    for (res, sub, write, state) in [(src, src_sub, false, COPY_SOURCE), (dst, dst_sub, true, COPY_DEST)] {
        if res == 0 {
            continue;
        }
        if let Some(Some(s)) = known(res) {
            use_side(list, res, &s, Side::Plane, write, &plane_subs(&s, sub), Some(state));
        }
    }
}

/// `on_copy` for CopyTextureRegion's locations (a footprint names no texture).
pub(super) unsafe fn on_copy_region(list: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, src: *const D3D12_TEXTURE_COPY_LOCATION) {
    if !active() || !sync() {
        return;
    }
    let at = |l: *const D3D12_TEXTURE_COPY_LOCATION| match l.as_ref() {
        Some(l) if l.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX => (std::mem::transmute_copy::<_, usize>(&l.pResource), l.Anonymous.SubresourceIndex),
        _ => (0, 0),
    };
    let ((d, ds), (s, ss)) = (at(dst), at(src));
    on_copy(list, d, ds, s, ss);
}

/// `list` is about to close: the side of a shadow it left ahead of the other is copied into the
/// other where it can be, and what is still ahead is left with the shadow for the lists after it.
pub(super) unsafe fn before_close(list: *mut c_void) {
    if !active() || !sync() || RECORDING.load(Ordering::Acquire) == 0 {
        return;
    }
    let uses = RECORDINGS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))).map(|r| std::mem::take(&mut r.uses)).unwrap_or_default();
    if uses.is_empty() {
        return;
    }
    let ty = list_type(list);
    let records = matches!(ty, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE);
    for u in uses {
        let to = match u.ahead {
            Ahead::Neither => None,
            Ahead::Shadow => Some(Side::Plane),
            Ahead::Plane => Some(Side::Shadow),
        };
        let ahead = match to {
            Some(to) if records && copy_to(list, ty, u.depth, &u.shadow, to, &u.states).is_ok() => Ahead::Neither,
            _ => u.ahead,
        };
        if ahead != Ahead::Neither {
            USE_LEFT.fetch_add(1, Ordering::Relaxed);
        }
        u.shadow.ahead.store(ahead as u8, Ordering::Relaxed);
    }
}

#[repr(C)]
struct Slot146Texture {
    sync: [i32; 2],
    access: [i32; 2],
    layout: [i32; 2],
    resource: usize,
    subresources: D3D12_BARRIER_SUBRESOURCE_RANGE,
    flags: i32,
}

const _: () = {
    assert!(size_of::<Slot146Texture>() == size_of::<D3D12_TEXTURE_BARRIER>());
    assert!(std::mem::offset_of!(Slot146Texture, resource) == std::mem::offset_of!(D3D12_TEXTURE_BARRIER, pResource));
    assert!(std::mem::offset_of!(Slot146Texture, subresources) == std::mem::offset_of!(D3D12_TEXTURE_BARRIER, Subresources));
};

#[derive(Clone, Copy, Debug)]
struct Slot146Match {
    list: usize,
    epoch: u64,
    twin: usize,
    range: Range,
    sync: [i32; 2],
    access: [i32; 2],
    layout: [i32; 2],
    flags: i32,
}

impl Slot146Match {
    fn uav_egress(self) -> bool {
        let uav = |i: usize| {
            self.access[i] & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 != 0
                || matches!(D3D12_BARRIER_LAYOUT(self.layout[i]), D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS
                    | D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
                    | D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS)
        };
        uav(0) && !uav(1)
    }
}

#[derive(Default)]
struct Slot146Decoded {
    global: u64,
    first_global: Option<([i32; 2], [i32; 2])>,
    matches: Vec<(usize, Slot146Match)>,
}

#[derive(Default, Clone, Copy, Debug)]
struct Slot146Counts {
    calls: u64,
    global: u64,
    opaque: u64,
    empty: u64,
    known_twin: u64,
    dropped: u64,
}

#[derive(Default)]
struct Slot146Trace {
    counts: Slot146Counts,
    samples: HashMap<usize, Vec<Slot146Match>>,
}

static SLOT146: Mutex<Option<Slot146Trace>> = Mutex::new(None);

unsafe fn slot146_slice<'a, T>(p: *const T, n: usize) -> Option<&'a [T]> {
    (p.is_aligned() && crate::gpu::shader::readable(p as usize, n.checked_mul(size_of::<T>())?))
        .then(|| std::slice::from_raw_parts(p, n))
}

unsafe fn slot146_decode(
    list: usize, n: u32, groups: *const c_void,
) -> Option<Slot146Decoded> {
    if n > 32 {
        return None;
    }
    let groups = slot146_slice(groups.cast::<D3D12_BARRIER_GROUP>(), n as usize)?;
    let mut remaining = 256usize;
    let mut decoded = Slot146Decoded::default();
    let epoch = super::present::counters().1;
    for g in groups {
        let count = g.NumBarriers as usize;
        remaining = remaining.checked_sub(count)?;
        if count == 0 {
            continue;
        }
        match g.Type {
            D3D12_BARRIER_TYPE_GLOBAL => {
                let globals = slot146_slice(g.Anonymous.pGlobalBarriers, count)?;
                decoded.global += count as u64;
                if decoded.first_global.is_none() {
                    let g = &globals[0];
                    decoded.first_global = Some(([g.SyncBefore.0, g.SyncAfter.0], [g.AccessBefore.0, g.AccessAfter.0]));
                }
            }
            D3D12_BARRIER_TYPE_BUFFER => {
                slot146_slice(g.Anonymous.pBufferBarriers, count)?;
            }
            D3D12_BARRIER_TYPE_TEXTURE => {
                let textures = slot146_slice(g.Anonymous.pTextureBarriers.cast::<Slot146Texture>(), count)?;
                for t in textures {
                    let (twin, mips, slices, planes) = if let Some(Some(s)) = known(t.resource) {
                        let planes = if matches!(s.depth_format, DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_D32_FLOAT_S8X24_UINT) { 2 } else { 1 };
                        (s.texture.as_raw() as usize, s.mips, s.slices, planes)
                    } else if depth_lineage::twin_enabled() && depth_lineage::interested(t.resource) {
                        let Some((mips, slices, planes)) = super::vram::arena_depth_shape(t.resource) else { continue };
                        (0, mips, slices, planes)
                    } else { continue };
                    let range = Range::texture(t.subresources, mips, slices, planes)?;
                    if !t.layout.iter().all(|&x| (-1..=D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON.0).contains(&x))
                        || !matches!(t.flags, 0 | 1)
                    {
                        return None;
                    }
                    decoded.matches.push((t.resource, Slot146Match {
                        list, epoch, twin, range,
                        sync: t.sync, access: t.access, layout: t.layout, flags: t.flags,
                    }));
                }
            }
            _ => return None,
        }
    }
    Some(decoded)
}

pub(super) fn note_slot146_global(list: usize) {
    depth_lineage::note(0, list, Range::All, "slot146-actual-global", || {
        "legacy UAV barrier; no resource or texture layout transition".into()
    });
}

pub(super) unsafe fn observe_slot146(list: usize, n: u32, groups: *const c_void, native_now: bool) {
    let state = depth_state::enabled();
    let lineage = depth_lineage::enabled();
    if !state && !lineage {
        return;
    }
    let decoded = (n != 0).then(|| slot146_decode(list, n, groups)).flatten();
    if native_now {
        if let Some((owner, shadow, mip, slice, sub, generation, index, groups, gate)) = super::depth_twin_probe::take(list) {
            let valid = decoded.as_ref().is_some_and(|d| d.matches.iter().any(|(r, e)|
                *r == owner && e.twin == shadow.texture.as_raw() as usize && e.range.covers(sub)
                    && e.uav_egress() && e.flags == 0 && (e.sync[0] | e.sync[1]) & D3D12_BARRIER_SYNC_SPLIT.0 == 0
                    && e.access[0] & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 != 0
                    && e.layout[1] == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE.0));
            if valid && known(owner).flatten().is_some_and(|s| Arc::ptr_eq(&s, &shadow)) {
                if let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&(list as *mut c_void)) {
                    let context = format!("actual dispatch and named guest UAV egress; owner={owner:#x} twin={:p} list={list:#x}/{generation} descriptor={index} grid={groups:?} sub={sub}; before-state from private committed twin, not guest layout", shadow.texture.as_raw());
                    if let Err(reason) = super::depth_capture::record_twin(cl, &shadow.texture, mip, slice, context, gate) {
                        static REFUSALS: AtomicU64 = AtomicU64::new(0);
                        if REFUSALS.fetch_add(1, Ordering::Relaxed) < 4 { crate::xlog_warn!("twin egress capture: {reason}"); }
                    }
                }
            }
        }
    }
    if lineage {
        if let Some(d) = &decoded {
            if let Some((sync, access)) = d.first_global {
                depth_lineage::note(0, list, Range::All, "slot146-guest-global", || {
                    format!("groups={n} globals={} first_sync={sync:x?} first_access={access:x?}", d.global)
                });
            } else if d.matches.is_empty() {
                depth_lineage::note(0, list, Range::All, "slot146-guest-unmatched", || format!("groups={n}"));
            }
            for &(res, s) in &d.matches {
                let kind = if s.uav_egress() { "slot146-guest-uav-egress" } else { "slot146-guest-texture" };
                depth_lineage::note(res, list, s.range, kind, || {
                    format!("twin={:#x} sync={:x?} access={:x?} layout={:?} flags={:#x}", s.twin, s.sync, s.access, s.layout, s.flags)
                });
                if depth_lineage::twin_enabled() {
                    for (hit, kind) in [
                        (s.layout[0] == D3D12_BARRIER_LAYOUT_UNDEFINED.0, "slot146-guest-activation"),
                        (s.flags & D3D12_TEXTURE_BARRIER_FLAG_DISCARD.0 != 0, "slot146-guest-discard"),
                        ((s.sync[0] | s.sync[1]) & D3D12_BARRIER_SYNC_SPLIT.0 != 0, "slot146-guest-split"),
                    ] {
                        if hit { depth_lineage::note(res, list, s.range, kind, || format!("guest only; twin={:#x} sync={:x?} access={:x?} layout={:?} flags={:#x}", s.twin, s.sync, s.access, s.layout, s.flags)); }
                    }
                }
            }
        } else {
            let kind = if n == 0 { "slot146-guest-empty" } else { "slot146-guest-opaque" };
            depth_lineage::note(0, list, Range::All, kind, || format!("groups={n}"));
        }
    }
    if !state {
        return;
    }
    let counts = {
        let mut g = SLOT146.lock().unwrap();
        let trace = g.get_or_insert_with(Slot146Trace::default);
        let c = &mut trace.counts;
        c.calls += 1;
        if n == 0 {
            c.empty += 1;
        } else if let Some(d) = decoded {
            c.global += d.global;
            c.known_twin += d.matches.len() as u64;
            for (res, sample) in d.matches {
                if sample.twin == 0 { continue }
                if trace.samples.len() >= 512 && !trace.samples.contains_key(&res) {
                    c.dropped += 1;
                    continue;
                }
                let samples = trace.samples.entry(res).or_default();
                let egress = sample.uav_egress();
                samples.retain(|x| x.range != sample.range || x.twin != sample.twin || x.uav_egress() != egress);
                if samples.len() >= 16 {
                    c.dropped += 1;
                    if let Some(i) = samples.iter().position(|x| !x.uav_egress()) {
                        samples.remove(i);
                    } else if egress {
                        samples.remove(0);
                    } else {
                        continue;
                    }
                }
                samples.push(sample);
            }
        } else {
            c.opaque += 1;
        }
        *c
    };
    if counts.calls.is_power_of_two() {
        crate::xlog_info!("depth capture #146 CPU input totals (global entries, opaque calls): {counts:?}");
    }
}

pub(super) fn report_slot146(list: usize, resource: usize, subresource: u32) {
    static REPORTS: AtomicU64 = AtomicU64::new(0);
    if !depth_state::enabled() || REPORTS.fetch_add(1, Ordering::Relaxed) >= 32 {
        return;
    }
    let twin = known(resource).flatten().map(|s| s.texture.as_raw() as usize);
    let (counts, matched, egress) = SLOT146.lock().unwrap().as_ref().map_or((Slot146Counts::default(), None, None), |trace| {
        let (matched, egress) = trace.samples.get(&resource).map_or((None, None), |samples| {
            let mut matching = samples.iter().rev().copied().filter(|s| Some(s.twin) == twin && s.range.covers(subresource));
            let latest = matching.next();
            let egress = latest.filter(|s| s.uav_egress()).or_else(|| matching.find(|s| s.uav_egress()));
            (latest, egress)
        });
        (trace.counts, matched, egress)
    });
    crate::xlog_info!("depth capture #146 CPU input join: EQUAL list {list:#x} resource {resource:#x} subresource {subresource} twin {twin:x?}; {counts:?}");
    if let Some(s) = matched {
        crate::xlog_info!("depth capture #146 matched CPU record: list {:#x} title epoch {} twin {:#x} range {:?} sync {:x?} access {:x?} layout {:?} flags {:#x}",
            s.list, s.epoch, s.twin, s.range, s.sync, s.access, s.layout, s.flags);
    } else {
        crate::xlog_info!("depth capture #146: no matching checked texture record for this subresource");
    }
    if let Some(s) = egress.filter(|_| matched.is_some_and(|m| !m.uav_egress())) {
        crate::xlog_info!("depth capture #146 retained CPU UAV egress: list {:#x} title epoch {} twin {:#x} range {:?} sync {:x?} access {:x?} layout {:?} flags {:#x}",
            s.list, s.epoch, s.twin, s.range, s.sync, s.access, s.layout, s.flags);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{console_heap, list_x, title_textures};
    use super::*;
    use std::mem::ManuallyDrop;

    /// the counters are the process's: one GPU test at a time (and none beside the tests that
    /// put resources that are not there into the video memory records: `VRAM_TEST_LOCK`, which
    /// reading a texture back goes through)
    static GPU: Mutex<()> = Mutex::new(());

    #[test]
    fn a_use_takes_the_side_ahead_first_and_leaves_its_own_ahead_when_it_writes() {
        assert!(Side::Shadow.behind(Ahead::Plane));
        assert!(Side::Plane.behind(Ahead::Shadow));
        assert!(!Side::Plane.behind(Ahead::Plane));
        assert!(!Side::Shadow.behind(Ahead::Neither));
        assert_eq!(Side::Shadow.after(Ahead::Plane, true, true), Ahead::Shadow);
        assert_eq!(Side::Plane.after(Ahead::Shadow, true, true), Ahead::Plane);
        assert_eq!(Side::Plane.after(Ahead::Shadow, false, true), Ahead::Neither);
        assert_eq!(Side::Plane.after(Ahead::Shadow, false, false), Ahead::Shadow, "a copy not made leaves the shadow ahead");
        assert_eq!(Side::Shadow.after(Ahead::Neither, false, false), Ahead::Neither);
        for a in [Ahead::Neither, Ahead::Shadow, Ahead::Plane] {
            assert_eq!(Ahead::of(a as u8), a);
        }
    }

    #[test]
    fn states_formats_and_subresources() {
        assert_eq!(pc_state(UNORDERED_ACCESS), SHADER_READ);
        assert_eq!(pc_state(DEPTH_WRITE), DEPTH_WRITE);
        assert_eq!(pc_state(0), 0);
        // depth views in the float (or unorm) format of the shadow, whatever the title names
        assert_eq!(view_format(DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS), Some(DXGI_FORMAT_R32_FLOAT));
        assert_eq!(view_format(DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS), Some(DXGI_FORMAT_R32_FLOAT));
        assert_eq!(view_format(DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN), Some(DXGI_FORMAT_R32_FLOAT));
        assert_eq!(view_format(DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT), Some(DXGI_FORMAT_R32_UINT));
        assert_eq!(view_format(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM), Some(DXGI_FORMAT_R16_UNORM));
        assert_eq!(view_format(DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R32_UINT), Some(DXGI_FORMAT_R32_FLOAT), "24 depth bits are not the shadow's 32");
        // the stencil plane has no shadow
        assert_eq!(view_format(DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT), None);
        assert_eq!(view_format(DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT), None);
        assert_eq!(view_format(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM), None, "not a depth buffer");
        // subresources: mip + slice · mips, the stencil plane after the depth plane's
        assert_eq!(parts_of(3, 2, ALL), vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]);
        assert_eq!(parts_of(3, 2, 4), vec![(1, 1)]);
        assert_eq!(parts_of(3, 2, 6), vec![], "the stencil plane's");
        assert_eq!(parts_of(1, 1, 0), vec![(0, 0)]);
    }

    #[test]
    fn a_recording_knows_what_it_put_in_step_until_it_records_a_change() {
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let was = ACTIVE.swap(true, Ordering::AcqRel);
        let (list, other) = (0x7001_0000usize as *mut c_void, 0x7002_0000usize as *mut c_void);
        assert!(!in_step(list, 0x10, 2));
        mark_in_step(list, 0x10, 2);
        assert!(in_step(list, 0x10, 2));
        assert!(!in_step(list, 0x10, 3) && !in_step(list, 0x10, ALL) && !in_step(list, 0x20, 2) && !in_step(other, 0x10, 2));
        mark_in_step(list, 0x20, ALL);
        assert!(in_step(list, 0x20, 5) && in_step(list, 0x20, ALL) && in_step(list, 0x10, 2));
        note_op(other);
        assert!(in_step(list, 0x10, 2), "another list's commands");
        note_op(list);
        assert!(!in_step(list, 0x10, 2) && !in_step(list, 0x20, ALL), "a draw or dispatch may have changed either");
        mark_in_step(list, 0x10, 2);
        list_reset(list as usize);
        assert!(!in_step(list, 0x10, 2));
        ACTIVE.store(was, Ordering::Release);
    }

    struct Rig {
        dev: ID3D12Device,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        value: u64,
        out: ID3D12Resource,
        readback: ID3D12Resource,
        rs: ID3D12RootSignature,
        probe: ID3D12PipelineState,
        fill: ID3D12PipelineState,
        shared: ID3D12DescriptorHeap,
        step: u64,
    }

    unsafe fn committed(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, desc: &D3D12_RESOURCE_DESC) -> windows::core::Result<ID3D12Resource> {
        let mut r: Option<ID3D12Resource> = None;
        title_textures::own_work(|| dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut r))?;
        Ok(r.unwrap())
    }

    unsafe fn pc_transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: ALL, StateBefore: before, StateAfter: after }),
            },
            ..Default::default()
        }
    }

    impl Rig {
        /// The pipelines of a title's compute passes over a depth buffer: one that loads a texel
        /// through the view pair of a console descriptor (the shader view, or with `through` the
        /// unordered access view) and one that stores a ramp through the unordered access view.
        unsafe fn new(one_d: bool) -> Option<Rig> {
            let raw = real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw)?.clone();
            let shared = console_heap::global_heap()?;
            let (kind, at3, at2) = if one_d { ("1DArray", "int3(x, z, 0)", "uint2(x, z)") } else { ("2DArray", "int4(x, y, z, 0)", "uint3(x, y, z)") };
            let head = format!(
                "cbuffer Call : register(b0) {{ uint x; uint y; uint z; uint at; uint through; }};
Texture{kind}<float> read : register(t0);
RWTexture{kind}<float> written : register(u0);
RWByteAddressBuffer res : register(u1);
[RootSignature(\"RootConstants(num32BitConstants=5, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1)\")]
"
            );
            let probe = format!("{head}[numthreads(1, 1, 1)]
void main() {{ float v = read.Load({at3}); if (through != 0) v = written[{at2}]; res.Store(at, asuint(v)); }}
");
            let fill = if one_d {
                format!("{head}[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{ written[uint2(id.x, z)] = float(id.x + at) / 1024.0; }}
")
            } else {
                format!("{head}[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{ written[uint3(id.xy, z)] = float(id.x + 16 * id.y + at) / 1024.0; }}
")
            };
            let (probe, fill) = (real::compile(&probe, "cs_5_1"), real::compile(&fill, "cs_5_1"));
            assert!(!probe.is_empty() && !fill.is_empty(), "the test's shaders compile");
            let (rs, probe, fill) = title_textures::own_work(|| {
                let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &probe).unwrap();
                let pipeline = |code: &[u8]| -> ID3D12PipelineState {
                    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&rs),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() },
                        ..Default::default()
                    };
                    dev.CreateComputePipelineState(&desc).unwrap()
                };
                let (p, f) = (pipeline(&probe), pipeline(&fill));
                (rs, p, f)
            });
            let mut buffer = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let out = committed(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &buffer).unwrap();
            buffer.Flags = D3D12_RESOURCE_FLAG_NONE;
            let readback = committed(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &buffer).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as u64;
            Some(Rig { dev, queue, fence, value: 0, out, readback, rs, probe, fill, shared, step })
        }

        /// A list of the title's with its heaps bound the way console code binds them.
        unsafe fn list(&self) -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
            let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let heaps = [self.shared.as_raw() as usize];
            list_x::set_descriptor_heaps_x(list.as_raw(), 1, heaps.as_ptr());
            (alloc, list)
        }

        /// The title's ResourceBarrier: one transition of `res`.
        unsafe fn barrier(&self, list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, sub: u32, before: u32, after: u32) {
            let b = transition(res.as_raw() as usize, sub, before, after);
            list_x::resource_barrier_x(list.as_raw(), 1, &b);
        }

        /// The title's UAV barrier for `res`: what its dispatches stored there is what the next
        /// loads (the driver runs dispatches side by side without one).
        unsafe fn uav_barrier(&self, list: &ID3D12GraphicsCommandList, res: &ID3D12Resource) {
            let r = res.as_raw() as usize;
            let b = [2u32, 0, r as u32, (r as u64 >> 32) as u32, 0, 0, 0, 0];
            list_x::resource_barrier_x(list.as_raw(), 1, &b);
        }

        /// A dispatch of `pipeline` over the views of console descriptor `slot` (everything bound
        /// again, as each of the title's dispatches binds it).
        unsafe fn dispatch(&self, list: &ID3D12GraphicsCommandList, pipeline: &ID3D12PipelineState, slot: &console_heap::Slot, call: [u32; 5], groups: (u32, u32)) {
            list.SetComputeRootSignature(&self.rs);
            list.SetPipelineState(pipeline);
            list.SetComputeRoot32BitConstants(0, 5, call.as_ptr() as *const c_void, 0);
            list.SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: slot.pc_gpu_uav - self.step });
            list.SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: slot.pc_gpu_uav });
            list.SetComputeRootUnorderedAccessView(3, self.out.GetGPUVirtualAddress());
            list.Dispatch(groups.0, groups.1, 1);
        }

        /// Runs `list` and returns the first `words` words the probes stored.
        unsafe fn run(&mut self, list: &ID3D12GraphicsCommandList, words: usize) -> Vec<f32> {
            list.ResourceBarrier(&[pc_transition(&self.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&self.readback, 0, &self.out, 0, 256);
            list.ResourceBarrier(&[pc_transition(&self.out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            list.Close().unwrap();
            self.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            self.value += 1;
            self.queue.Signal(&self.fence, self.value).unwrap();
            crate::gpu::fence::wait(&self.fence, self.value, &self.dev, None, 20_000).unwrap();
            assert!(self.dev.GetDeviceRemovedReason().is_ok(), "the device is still there");
            let mut p = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p as *const f32, words).to_vec();
            self.readback.Unmap(0, None);
            got
        }
    }

    /// The depth plane of subresource `sub` of `depth` (in `state`), a value per texel.
    unsafe fn depth_texels(dev: &ID3D12Device, depth: &ID3D12Resource, format: DXGI_FORMAT, sub: u32, state: D3D12_RESOURCE_STATES) -> Vec<f32> {
        let bytes = title_textures::read_gpu_subresource(dev, depth, sub, state).unwrap();
        match format {
            DXGI_FORMAT_R16_TYPELESS => bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect(),
            DXGI_FORMAT_R24G8_TYPELESS => bytes.chunks_exact(4).map(|c| (u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0xFF_FFFF) as f32 / 16_777_215.0).collect(),
            _ => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        }
    }

    /// A console depth buffer with unordered access of resource format `format` (16 wide, two
    /// mips, two slices): what draws left in the depth plane is what a compute pass loads
    /// through the title's unordered access views – by the shader view beside each and by the
    /// view itself – and what compute passes store through them is in the depth plane when the
    /// title moves the buffer back, with the stencil plane and the texels not stored to as they
    /// were. Then one subresource alone, and transitions with nothing recorded between them.
    unsafe fn round_trip(format: DXGI_FORMAT, one_d: bool) {
        let what = format!("format {}{}", format.0, if one_d { " (1D)" } else { "" });
        let Some(mut rig) = Rig::new(one_d) else { return println!("SKIPPED depth_uav {what}: no device or shared descriptor heap") };
        let dev = rig.dev.clone();
        let (depth_format, title_view, _, shadow_view) = formats(format).unwrap();
        let stencil = matches!(depth_format, DXGI_FORMAT_D32_FLOAT_S8X24_UINT | DXGI_FORMAT_D24_UNORM_S8_UINT);
        let height = if one_d { 1 } else { 16 };
        // the PC resource `pc_resource_desc` makes of the console's: a depth buffer only
        let depth = match committed(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE, &D3D12_RESOURCE_DESC {
            Dimension: if one_d { D3D12_RESOURCE_DIMENSION_TEXTURE1D } else { D3D12_RESOURCE_DIMENSION_TEXTURE2D },
            Width: 16, Height: height, DepthOrArraySize: 2, MipLevels: 2,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        }) {
            Ok(d) => d,
            Err(e) => return println!("SKIPPED depth_uav {what}: the driver makes no such depth buffer ({e})"),
        };
        let res = depth.as_raw();
        let before = counts();

        // the title's views: one console descriptor per mip, both slices
        let ty = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: 4, ..Default::default() }).unwrap();
        let obj = heap.as_raw() as usize;
        assert!(console_heap::register(obj, 2, heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, 0, dev.GetDescriptorHandleIncrementSize(ty), false, true, true));
        let cpu = console_heap::start(obj).unwrap();
        let mut slots = Vec::new();
        for mip in 0..2u32 {
            let at = cpu + (mip * console_heap::SLOT) as u64;
            let mut v = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: title_view, ..Default::default() };
            if one_d {
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: 0, ArraySize: 2 };
            } else {
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: 0, ArraySize: 2, PlaneSlice: 0 };
            }
            real::dev_create_uav_x(real::device_raw(), res, std::ptr::null_mut(), &v, at);
            let s = console_heap::slot(at).unwrap();
            assert_ne!(s.pc_gpu_uav, 0, "{what}: the views are in the shared heap");
            assert_eq!((s.at as *const u32).read(), s.index, "{what}: the descriptor names its views");
            slots.push(s);
        }
        let made = counts();
        assert_eq!((made.views - before.views, made.null_views - before.null_views, made.shadows - before.shadows), (2, 0, 1), "{what}: two views, one shadow");

        // draws' part: each subresource cleared to a depth of its own (and the stencil plane)
        let targets: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 4, ..Default::default() }).unwrap();
        let target_step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
        let target = |sub: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: targets.GetCPUDescriptorHandleForHeapStart().ptr + sub as usize * target_step };
        const CLEARED: [f32; 4] = [0.25, 0.75, 0.5, 1.0];
        let (alloc, list) = rig.list();
        for sub in 0..4u32 {
            let mut d = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: depth_format, ..Default::default() };
            if one_d {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
                d.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_DSV { MipSlice: sub % 2, FirstArraySlice: sub / 2, ArraySize: 1 };
            } else {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { MipSlice: sub % 2, FirstArraySlice: sub / 2, ArraySize: 1 };
            }
            dev.CreateDepthStencilView(&depth, Some(&d), target(sub));
            let flags = if stencil { D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL } else { D3D12_CLEAR_FLAG_DEPTH };
            list.ClearDepthStencilView(target(sub), flags, CLEARED[sub as usize], 0x5A, None);
        }
        // the title: to unordered access, loads, stores, loads again, back to a depth buffer
        rig.barrier(&list, &depth, ALL, DEPTH_WRITE, UNORDERED_ACCESS);
        let through = (shadow_view == DXGI_FORMAT_R32_FLOAT) as u32;
        rig.dispatch(&list, &rig.probe, &slots[0], [3, 5 % height, 0, 0, 0], (1, 1));
        rig.dispatch(&list, &rig.probe, &slots[0], [3, 5 % height, 1, 4, 0], (1, 1));
        rig.dispatch(&list, &rig.probe, &slots[1], [1, 1 % height, 0, 8, 0], (1, 1));
        rig.dispatch(&list, &rig.probe, &slots[1], [1, 1 % height, 1, 12, 0], (1, 1));
        rig.dispatch(&list, &rig.probe, &slots[0], [3, 5 % height, 0, 16, through], (1, 1));
        let rows = |n: u32| if one_d { 1 } else { n };
        // (the loads before the stores, as the title's UAV barrier has them)
        rig.uav_barrier(&list, &depth);
        rig.dispatch(&list, &rig.fill, &slots[0], [0, 0, 1, 1, 0], (2, rows(2)));
        rig.dispatch(&list, &rig.fill, &slots[1], [0, 0, 0, 100, 0], (1, 1));
        rig.uav_barrier(&list, &depth);
        rig.dispatch(&list, &rig.probe, &slots[0], [2, 3 % height, 1, 20, 0], (1, 1));
        rig.dispatch(&list, &rig.probe, &slots[0], [2, 3 % height, 1, 24, through], (1, 1));
        rig.barrier(&list, &depth, ALL, UNORDERED_ACCESS, DEPTH_WRITE);
        let got = rig.run(&list, 7);
        let ramp = |x: u32, y: u32, seed: u32| (x + 16 * (y % height) + seed) as f32 / 1024.0;
        let step = match depth_format {
            DXGI_FORMAT_D16_UNORM => 1.0 / 65535.0,
            DXGI_FORMAT_D24_UNORM_S8_UINT => 1.0 / 16_777_215.0,
            _ => 0.0,
        };
        let close = |a: f32, b: f32| (a - b).abs() <= step;
        assert!(got[..5].iter().zip([0.25, 0.5, 0.75, 1.0, 0.25]).all(|(&v, want)| close(v, want)), "{what}: a compute pass loads the depth the depth buffer holds, not zero: {:?}", &got[..5]);
        if !(close(got[5], ramp(2, 3, 1)) && close(got[6], ramp(2, 3, 1))) {
            // what the shadow holds where the pass stored (its row 3 of slice 1 and of slice 0)
            let shadow = known(res as usize).flatten().unwrap();
            let row = |sub: u32| -> Vec<u8> { title_textures::read_gpu_subresource(&dev, &shadow.texture, sub, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).map(|t| t.chunks(t.len() / rows(16) as usize).nth(3 % height as usize).unwrap_or(&[]).to_vec()).unwrap_or_default() };
            panic!("{what}: a load after a store through the view: {:?}, wanted {}; the shadow's row of slice 1: {:x?}, of slice 0: {:x?}", &got[5..7], ramp(2, 3, 1), row(2), row(0));
        }
        let after = counts();
        assert_eq!((after.copies_in - made.copies_in, after.copies_out - made.copies_out), (1, 1), "{what}: one copy each way");

        // the depth buffer after the title moved it back
        let state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        let whole = |sub: u32, size: u32, value: f32| {
            let t = depth_texels(&dev, &depth, format, sub, state);
            assert_eq!(t.len() as u32, size * rows(size), "{what}: subresource {sub}");
            assert!(t.iter().all(|&v| close(v, value)), "{what}: subresource {sub} is as the clear left it ({value}): {:?}", &t[..4]);
        };
        let ramped = |sub: u32, size: u32, seed: u32, state: D3D12_RESOURCE_STATES| {
            let t = depth_texels(&dev, &depth, format, sub, state);
            for y in 0..rows(size) {
                for x in 0..size {
                    let (v, want) = (t[(x + y * size) as usize], ramp(x, y, seed));
                    assert!(close(v, want), "{what}: subresource {sub} at ({x}, {y}) holds what the compute pass stored: {v}, wanted {want}");
                }
            }
        };
        whole(0, 16, 0.25);
        ramped(1, 8, 100, state);
        ramped(2, 16, 1, state);
        whole(3, 8, 1.0);
        if stencil {
            for sub in 4..8 {
                let t = title_textures::read_gpu_subresource(&dev, &depth, sub, state).unwrap();
                assert!(!t.is_empty() && t.iter().all(|&b| b == 0x5A), "{what}: the stencil plane of subresource {} is not touched", sub - 4);
            }
        }
        drop((alloc, list));

        // one subresource alone (to a state that is not DEPTH_WRITE), and transitions with
        // nothing recorded in between
        let c0 = counts();
        let (alloc, list) = rig.list();
        rig.barrier(&list, &depth, 3, DEPTH_WRITE, UNORDERED_ACCESS);
        rig.dispatch(&list, &rig.fill, &slots[1], [0, 0, 1, 300, 0], (1, 1));
        rig.barrier(&list, &depth, 3, UNORDERED_ACCESS, D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32);
        rig.barrier(&list, &depth, 0, DEPTH_WRITE, UNORDERED_ACCESS);
        rig.barrier(&list, &depth, 0, UNORDERED_ACCESS, DEPTH_WRITE);
        rig.barrier(&list, &depth, 0, DEPTH_WRITE, UNORDERED_ACCESS);
        rig.dispatch(&list, &rig.probe, &slots[0], [9, 9 % height, 0, 0, 0], (1, 1));
        rig.barrier(&list, &depth, 0, UNORDERED_ACCESS, DEPTH_WRITE);
        let got = rig.run(&list, 1);
        let c1 = counts();
        assert!(close(got[0], 0.25), "{what}: the shadow of subresource 0 after the copies left out: {}", got[0]);
        assert_eq!(
            (c1.copies_in - c0.copies_in, c1.copies_in_left_out - c0.copies_in_left_out, c1.copies_out - c0.copies_out, c1.copies_out_left_out - c0.copies_out_left_out),
            (2, 1, 2, 1),
            "{what}: copies only where a draw or dispatch came between"
        );
        whole(0, 16, 0.25);
        ramped(1, 8, 100, state);
        ramped(2, 16, 1, state);
        ramped(3, 8, 300, D3D12_RESOURCE_STATE_DEPTH_READ);
        drop((alloc, list));

        console_heap::unregister(obj);
        drop(depth);
        assert!(known(res as usize).is_none(), "{what}: the shadow goes with its depth buffer");
    }

    fn gpu(name: &str) -> bool {
        if let Err(e) = real::init() {
            println!("SKIPPED {name}: {e}");
            return false;
        }
        true
    }

    #[test]
    #[ignore = "on the GPU, and not beside the rest of the suite (which then loses its device now and then; why is not known): --ignored --test-threads=1"]
    fn compute_reads_and_writes_a_d32_depth_buffer_through_console_unordered_access_views() {
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if gpu("compute_reads_and_writes_a_d32_depth_buffer_through_console_unordered_access_views") {
            unsafe { round_trip(DXGI_FORMAT_R32_TYPELESS, false) }
        }
    }

    #[test]
    #[ignore = "on the GPU, and not beside the rest of the suite (which then loses its device now and then; why is not known): --ignored --test-threads=1"]
    fn compute_reads_and_writes_a_d32s8_depth_plane_and_leaves_the_stencil() {
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if gpu("compute_reads_and_writes_a_d32s8_depth_plane_and_leaves_the_stencil") {
            unsafe { round_trip(DXGI_FORMAT_R32G8X24_TYPELESS, false) }
        }
    }

    #[test]
    #[ignore = "on the GPU, and not beside the rest of the suite (which then loses its device now and then; why is not known): --ignored --test-threads=1"]
    fn compute_reads_and_writes_a_d24s8_depth_plane_and_leaves_the_stencil() {
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if gpu("compute_reads_and_writes_a_d24s8_depth_plane_and_leaves_the_stencil") {
            unsafe { round_trip(DXGI_FORMAT_R24G8_TYPELESS, false) }
        }
    }

    #[test]
    #[ignore = "on the GPU, and not beside the rest of the suite (which then loses its device now and then; why is not known): --ignored --test-threads=1"]
    fn compute_reads_and_writes_a_d16_depth_buffer_through_console_unordered_access_views() {
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if gpu("compute_reads_and_writes_a_d16_depth_buffer_through_console_unordered_access_views") {
            unsafe { round_trip(DXGI_FORMAT_R16_TYPELESS, false) }
        }
    }

    #[test]
    #[ignore = "on the GPU, and not beside the rest of the suite (which then loses its device now and then; why is not known): --ignored --test-threads=1"]
    fn compute_reads_and_writes_a_1d_depth_buffer_through_console_unordered_access_views() {
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if gpu("compute_reads_and_writes_a_1d_depth_buffer_through_console_unordered_access_views") {
            unsafe { round_trip(DXGI_FORMAT_R32_TYPELESS, true) }
        }
    }
}
