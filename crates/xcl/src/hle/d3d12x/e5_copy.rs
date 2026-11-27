//! R9G9B9E5 views read through a copy of the R32_UINT texture the GPU writes at their address
//! (`XCL_E5_COPY`).
//!
//! A title packs light as R9G9B9E5 into R32_UINT textures with compute shaders, which cannot
//! write R9G9B9E5, and samples it through R9G9B9E5 textures it places at the same addresses.
//! `vram` gives the two one memory while the sampled texture is mapped tile for tile on the
//! pages of the one the GPU writes (`title_textures`: "stands on the tiles of a GPU-written
//! one"). When it is not (unmapped as a placement over its console range retired it, on memory
//! of its own, or committed), it shows the title's memory, which holds zeros, and every read of
//! the packed light is black.
//!
//! An R9G9B9E5 shader view of such a texture is made of a texture of XCL's own instead (`view`),
//! and that copy is brought in step with the R32_UINT texture where the title says its writes
//! are done (`after_barriers`): a transition of it out of UNORDERED_ACCESS, or a UAV barrier
//! naming it. The texels go as they are, through a buffer: the PC copies between the two
//! formats in no other way. The copy rests in NON_PIXEL_SHADER_RESOURCE | PIXEL_SHADER_RESOURCE,
//! so copies are recorded on direct lists only; one that a compute list would take is left out
//! and counted.
//!
//! Which textures: a sampled single-sample R9G9B9E5 texture (no render target, depth or
//! unordered access) and a live R32_UINT placement with unordered access at its address of its
//! dimension, size, mips and slices (`texture_alias::writer_at`). Not covered: an R9G9B9E5
//! texture with render target or unordered access, which the PC makes R16G16B16A16_FLOAT
//! (`device_x::pc_resource_desc`); its R32_UINT views are views of that one texture.
//!
//! `XCL_E5_COPY=0`: every view of the texture itself, as before.
//!
//! The behaviour is CGBJ's (codec-d/cool_fox f4b48ad766ddbec1a1867022cd42f7645b0dd55e, a
//! VIctory fork, crates/victory-gpu/src/resource.rs `alias_of`, src/texture.rs `alias`; GPL-3:
//! the behaviour, none of its code).

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_SAMPLE_DESC};

const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
const COMMON: u32 = D3D12_RESOURCE_STATE_COMMON.0 as u32;
const UNORDERED_ACCESS: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
const COPY_SOURCE: u32 = D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32;
/// the state a copy rests in between its updates
const READ: u32 = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32;

// PC command list methods (the runtime's own: `raw`)
const COPY_TEXTURE_REGION: usize = 16;
const RESOURCE_BARRIER: usize = 26;

/// How long a copy outlives its texture: lists recorded before may still run.
const GRACE: Duration = Duration::from_secs(2);

/// The copy one sampled R9G9B9E5 texture's views are made of.
struct Pair {
    /// the R32_UINT texture at its address (0: none is live there now)
    source: usize,
    copy: ID3D12Resource,
    buffer: ID3D12Resource,
    /// each subresource in the buffer: as the R32_UINT texture lays it out, as the copy does
    parts: Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
}

// D3D12 resources are free-threaded
unsafe impl Send for Pair {}

/// sampled R9G9B9E5 texture → its copy
static PAIRS: Mutex<Option<HashMap<usize, Pair>>> = Mutex::new(None);
/// copies whose texture is gone, and when it went
static GONE: Mutex<Vec<(Instant, Pair)>> = Mutex::new(Vec::new());
/// a copy was made (the hooks below have something to do)
static ACTIVE: AtomicBool = AtomicBool::new(false);

static MADE: AtomicU64 = AtomicU64::new(0);
static COPIES: AtomicU64 = AtomicU64::new(0);
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_E5_COPY").map_or(true, |v| v.trim() != "0"))
}

/// Whether two textures lay out their texels alike but for the format.
fn alike(a: &D3D12_RESOURCE_DESC, b: &D3D12_RESOURCE_DESC) -> bool {
    (a.Dimension, a.Width, a.Height, a.DepthOrArraySize, a.MipLevels, a.SampleDesc.Count)
        == (b.Dimension, b.Width, b.Height, b.DepthOrArraySize, b.MipLevels, b.SampleDesc.Count)
}

/// The title makes a shader resource view (`desc`; None: the default one) of `res`. An
/// R9G9B9E5 view of a sampled R9G9B9E5 texture that does not show what the GPU writes to the
/// R32_UINT texture at its address is made of the copy (made now if need be); any other view of
/// `res` itself.
pub(super) unsafe fn view(res: *mut c_void, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> *mut c_void {
    const WRITTEN: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    if !on() || desc.is_some_and(|d| d.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP || d.ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        return res;
    }
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else { return res };
    let rd = r.GetDesc();
    if rd.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP || rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || rd.Flags.0 & WRITTEN != 0 || rd.SampleDesc.Count > 1 {
        return res;
    }
    let source = super::texture_alias::writer_at(res as usize, DXGI_FORMAT_R32_UINT.0)
        .filter(|&w| super::real::reference(w).is_some_and(|w| alike(&w.GetDesc(), &rd)));
    if let Some(p) = PAIRS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(res as usize))) {
        p.source = source.unwrap_or(0);
        return p.copy.as_raw();
    }
    let Some(source) = source.filter(|&w| !super::vram::shows(res as usize, w)) else { return res };
    let mut pair = match make(&rd) {
        Ok(pair) => pair,
        Err(e) => {
            static FAILED: AtomicU64 = AtomicU64::new(0);
            if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("XCL_E5_COPY: no copy for an R9G9B9E5 texture {}x{} ({e}): its views show its own memory, as before", rd.Width, rd.Height);
            }
            return res;
        }
    };
    pair.source = source;
    let copy = match PAIRS.lock().unwrap().get_or_insert_with(HashMap::new).entry(res as usize) {
        Entry::Occupied(e) => return e.get().copy.as_raw(),
        Entry::Vacant(e) => e.insert(pair).copy.as_raw(),
    };
    ACTIVE.store(true, Ordering::Relaxed);
    let n = MADE.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 {
        crate::xlog_info!(
            "XCL_E5_COPY: an R9G9B9E5 texture {}x{} ({} mips, {} slices) does not share the memory of the R32_UINT texture the GPU writes at its address: its views read a copy of that texture, made after the title's barriers that end its unordered access (direct lists; XCL_E5_COPY=0 turns it off)",
            rd.Width, rd.Height, rd.MipLevels, rd.DepthOrArraySize
        );
    } else if n.is_power_of_two() {
        crate::xlog_info!("XCL_E5_COPY: {n} R9G9B9E5 textures read through copies so far");
    }
    copy
}

/// A copy of the texture `rd` describes, and the buffer the R32_UINT texels go through.
unsafe fn make(rd: &D3D12_RESOURCE_DESC) -> Result<Pair, String> {
    let raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
    let desc = D3D12_RESOURCE_DESC { Alignment: 0, Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN, Flags: D3D12_RESOURCE_FLAG_NONE, ..*rd };
    let count = u32::from(desc.MipLevels) * if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(desc.DepthOrArraySize) };
    if count == 0 {
        return Err("no subresources".into());
    }
    let footprints = |format: DXGI_FORMAT| {
        let mut parts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut bytes = 0u64;
        dev.GetCopyableFootprints(&D3D12_RESOURCE_DESC { Format: format, ..desc }, 0, count, 0, Some(parts.as_mut_ptr()), None, None, Some(&mut bytes));
        (parts, bytes)
    };
    let (written, bytes) = footprints(DXGI_FORMAT_R32_UINT);
    let (read, read_bytes) = footprints(desc.Format);
    let same = written.iter().zip(&read).all(|(a, b)| {
        let (fa, fb) = (a.Footprint, b.Footprint);
        a.Offset == b.Offset && (fa.Width, fa.Height, fa.Depth, fa.RowPitch) == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
    });
    if bytes == 0 || bytes != read_bytes || !same {
        return Err("the two formats lay the texels out differently".into());
    }
    super::title_textures::own_work(|| {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut copy: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATES(READ as i32), None, &mut copy)
            .map_err(|e| format!("the copy: {e}"))?;
        let mut buffer: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
        .map_err(|e| format!("its buffer: {e}"))?;
        Ok(Pair { source: 0, copy: copy.ok_or("no copy")?, buffer: buffer.ok_or("no buffer")?, parts: written.into_iter().zip(read).collect() })
    })
}

/// `res` is gone (its last reference released). As a sampled texture with a copy, the copy goes
/// `GRACE` later; as the R32_UINT texture of copies, nothing is copied from it any more.
pub(super) fn forget(res: usize) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let gone = {
        let mut g = PAIRS.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        for p in m.values_mut().filter(|p| p.source == res) {
            p.source = 0;
        }
        m.remove(&res)
    };
    let now = Instant::now();
    // (released after the lock: a release comes back here)
    let expired: Vec<(Instant, Pair)> = {
        let mut g = GONE.lock().unwrap();
        let (expired, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *g).into_iter().partition(|(at, _)| now.duration_since(*at) >= GRACE);
        *g = kept;
        g.extend(gone.map(|p| (now, p)));
        expired
    };
    drop(expired);
}

/// After the title's barriers on `list` (`out`: as recorded on the PC; `title`: the `n` it
/// gave): each R32_UINT texture with copies whose writes they end is copied into them. A
/// transition out of UNORDERED_ACCESS ends them (in the state it leaves the texture in), and so
/// does a UAV barrier naming the texture when no transition of it is among them.
pub(super) unsafe fn after_barriers(list: *mut c_void, out: &[[u32; 8]], title: *const [u32; 8], n: u32) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let at = |b: &[u32; 8]| (b[2] as u64 | (b[3] as u64) << 32) as usize;
    let transitions: Vec<&[u32; 8]> = out.iter().filter(|b| b[0] == 0).collect();
    let mut done: Vec<(usize, u32, u32)> = transitions
        .iter()
        .copied()
        .filter(|b| b[1] & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY.0 as u32 == 0 && b[5] & UNORDERED_ACCESS != 0 && b[6] & UNORDERED_ACCESS == 0)
        .map(|b| (at(b), b[4], b[6]))
        .collect();
    for k in 0..n as usize {
        let b = title.add(k).read_unaligned();
        let w = at(&b);
        if b[0] == 2 && b[1] <= 3 && w != 0 && !transitions.iter().any(|&t| at(t) == w) && !done.iter().any(|d| d.0 == w) {
            done.push((w, ALL, UNORDERED_ACCESS));
        }
    }
    let jobs: Vec<(usize, u32, u32, ID3D12Resource, ID3D12Resource, Vec<_>)> = {
        let g = PAIRS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        done.iter()
            .flat_map(|&(w, sub, state)| m.values().filter(move |p| w != 0 && p.source == w).map(move |p| (w, sub, state, p.copy.clone(), p.buffer.clone(), p.parts.clone())))
            .collect()
    };
    if jobs.is_empty() {
        return;
    }
    if super::list_x::tracked_list_type(list) != Some(D3D12_COMMAND_LIST_TYPE_DIRECT) {
        for _ in &jobs {
            count(&LEFT_OUT);
        }
        return;
    }
    let predicated = super::list_x::before_own_pass(list);
    for (source, sub, state, copy, buffer, parts) in &jobs {
        if record(list, *source, *sub, *state, copy.as_raw() as usize, buffer.as_raw() as usize, parts) {
            count(&COPIES);
        }
    }
    super::list_x::after_own_pass(list, false, predicated);
}

/// Records on `list` the copy of subresource `sub` (or all) of `source`, in `state`, through
/// `buffer` into `copy`; `source` is left in `state`. False: it has no such subresource.
unsafe fn record(
    list: *mut c_void, source: usize, sub: u32, state: u32, copy: usize, buffer: usize,
    parts: &[(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)],
) -> bool {
    let subs: Vec<u32> = match sub {
        ALL => (0..parts.len() as u32).collect(),
        k if (k as usize) < parts.len() => vec![k],
        _ => return false,
    };
    let texture = |res: usize, k: u32| location(res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k });
    let rows = |footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| location(buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint });
    let mut first = vec![transition(buffer, ALL, COMMON, COPY_DEST)];
    let mut second = vec![transition(buffer, ALL, COPY_DEST, COPY_SOURCE), transition(copy, sub, READ, COPY_DEST)];
    if state & COPY_SOURCE == 0 {
        first.push(transition(source, sub, state, COPY_SOURCE));
        second.push(transition(source, sub, COPY_SOURCE, state));
    }
    barriers(list, &first);
    for &k in &subs {
        copy_region(list, &rows(parts[k as usize].0), &texture(source, k));
    }
    barriers(list, &second);
    for &k in &subs {
        copy_region(list, &texture(copy, k), &rows(parts[k as usize].1));
    }
    barriers(list, &[transition(buffer, ALL, COPY_SOURCE, COMMON), transition(copy, sub, COPY_DEST, READ)]);
    true
}

/// One more copy recorded (`COPIES`) or left out (`LEFT_OUT`): the counts at each power of two.
fn count(counter: &AtomicU64) {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!(
            "XCL_E5_COPY: {} R9G9B9E5 textures read through copies; {} copies of their R32_UINT textures recorded, {} left out on lists that are not direct",
            MADE.load(Ordering::Relaxed),
            COPIES.load(Ordering::Relaxed),
            LEFT_OUT.load(Ordering::Relaxed)
        );
    }
}

/// a PC method of `list` (the runtime's own, past XCL's hooks)
unsafe fn raw<F: Copy>(list: *mut c_void, slot: usize) -> F {
    let f = if super::real::is_adopted(list) { super::real::orig_fn(list, slot) } else { *(*(list as *const *const usize)).add(slot) };
    std::mem::transmute_copy(&f)
}

fn transition(res: usize, sub: u32, before: u32, after: u32) -> [u32; 8] {
    [0, 0, res as u32, (res as u64 >> 32) as u32, sub, before, after, 0]
}

unsafe fn barriers(list: *mut c_void, b: &[[u32; 8]]) {
    raw::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(list, RESOURCE_BARRIER)(list, b.len() as u32, b.as_ptr());
}

unsafe fn location(res: usize, ty: D3D12_TEXTURE_COPY_TYPE, at: D3D12_TEXTURE_COPY_LOCATION_0) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&res), Type: ty, Anonymous: at }
}

unsafe fn copy_region(list: *mut c_void, dst: &D3D12_TEXTURE_COPY_LOCATION, src: &D3D12_TEXTURE_COPY_LOCATION) {
    let f = raw::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(list, COPY_TEXTURE_REGION);
    f(list, dst, 0, 0, 0, src, std::ptr::null());
}
