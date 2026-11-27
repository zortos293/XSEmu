//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The first upload is not done when the texture is made (a title makes thousands while it
//! streams – Fortnite's render thread then spent its time in CreatePlacedResourceX and its
//! render command fences ran late): it is pending until the title's next ExecuteCommandLists,
//! where all pending and rewritten textures go in batches on one list each.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.
//!
//! A copy starts and ends in the state the title's submitted work left the texture in: its
//! transitions of these textures are noted per PC command list as they are recorded
//! (`note_transitions`) and applied, in submission order, when the list is submitted
//! (`submitted`) – per subresource once a transition names one.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch with atomic reset of the reported writes.
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    pub(super) fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// (device, address its bytes are read from, layout, states) of an upload
type Upload = (usize, usize, Arc<xg::Texture>, Vec<D3D12_RESOURCE_STATES>);

struct Entry {
    device: usize,
    /// the title's address of the texture
    va: usize,
    /// where its bytes are read and its CPU writes watched: `va`, or – for a texture in a GPU
    /// view of the title's CPU memory with nothing mapped there – the same offset in the CPU
    /// view (`kernelx::mirrors`)
    src: usize,
    /// `va` lies in a GPU view: `src` is checked again when that view's mappings change
    in_gpu_view: bool,
    tex: Arc<xg::Texture>,
    /// the state the title's submitted work leaves the texture in (uploads leave it there):
    /// the creation state, then what its transitions give – one state for every subresource,
    /// or one per subresource (D3D12 order) while they differ
    states: Vec<D3D12_RESOURCE_STATES>,
    /// not uploaded yet (made since the last submit)
    pending: bool,
    /// the title's GPU work wrote it (CopyTextureRegion into it): its CPU view is stale from
    /// then on and no longer uploaded
    gpu_written: bool,
    /// backed by shared `vram::Pages` tiles (overlapping placements share video memory)
    placed: bool,
    /// tiles were unmapped after a different-range placement superseded this texture
    unmapped: bool,
    /// monotonic registration order (newest wins when uncovering shadowed textures)
    seq: u64,
}

#[derive(Clone, Copy)]
struct CpuColorView {
    device: usize,
    va: usize,
    desc: [u8; 0x38],
    state: D3D12_RESOURCE_STATES,
}

use crate::memmap::{committed_runs, copy_title};

static CPU_COLOR_VIEWS: Mutex<BTreeMap<usize, CpuColorView>> = Mutex::new(BTreeMap::new());

/// textures waiting for their first upload
static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

struct Table {
    /// Active PC textures (newest at each range) → its place in the title's memory
    by_res: BTreeMap<usize, Entry>,
    /// Active `(va, PC texture)` → end address (`va + size`); entries with different `(va, end)` are disjoint
    by_va: BTreeMap<(usize, usize), usize>,
    /// Older overlapped textures kept dormant until released or uncovered when a newer one is released
    shadowed: BTreeMap<usize, Entry>,
    /// Shadowed `(va, PC texture)` → end address (may overlap)
    shadowed_by_va: BTreeMap<(usize, usize), usize>,
    /// Largest byte span currently in `shadowed` (bounds range queries)
    max_shadowed_size: usize,
    next_seq: u64,
}

impl Table {
    const fn new() -> Self {
        Table {
            by_res: BTreeMap::new(),
            by_va: BTreeMap::new(),
            shadowed: BTreeMap::new(),
            shadowed_by_va: BTreeMap::new(),
            max_shadowed_size: 0,
            next_seq: 0,
        }
    }

    fn has_active_overlap(&self, va: usize, end: usize) -> bool {
        if let Some((_, &oend)) = self.by_va.range(..(va, 0)).next_back() {
            if oend > va {
                return true;
            }
        }
        end > va && self.by_va.range((va, 0)..(end, 0)).next().is_some()
    }

    fn active_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (&(ova, ores), &oend) in self.by_va.range(..(va, 0)).rev() {
            if oend <= va {
                break;
            }
            out.push((ova, ores, oend));
        }
        if end > va {
            for (&(ova, ores), &oend) in self.by_va.range((va, 0)..(end, 0)) {
                out.push((ova, ores, oend));
            }
        }
        out
    }

    fn shadowed_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        if self.shadowed.is_empty() || end <= va {
            return Vec::new();
        }
        let lo = va.saturating_sub(self.max_shadowed_size);
        self.shadowed_by_va
            .range((lo, 0)..(end, 0))
            .filter(|&(_, &send)| send > va)
            .map(|(&(sva, sres), &send)| (sva, sres, send))
            .collect()
    }

    /// Permanently retire active and shadowed entries overlapping `[va, end)` (used when a
    /// GPU-owned render target / depth / UAV overwrites the address range). Returns
    /// `(pending_delta, retired_count, placed_res_to_unmap)`.
    fn retire_range(&mut self, va: usize, end: usize) -> (isize, usize, Vec<usize>) {
        let (mut pending_delta, mut retired, mut to_unmap) = (0isize, 0usize, Vec::new());
        for (ova, ores, _) in self.active_overlaps(va, end) {
            self.by_va.remove(&(ova, ores));
            if let Some(old) = self.by_res.remove(&ores) {
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped {
                    to_unmap.push(ores);
                }
                retired += 1;
            }
        }
        for (sva, sres, _) in self.shadowed_overlaps(va, end) {
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(old) = self.shadowed.remove(&sres) {
                if old.placed && !old.unmapped {
                    to_unmap.push(sres);
                }
                retired += 1;
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (pending_delta, retired, to_unmap)
    }

    /// Insert `res`, shadowing older active entries whose memory `[va, va + size)` it overlaps so
    /// dead placements never re-upload over shared `vram::Pages` tiles (two committed resources
    /// with the exact same range may coexist as format aliases). Placed entries superseded by a
    /// *different* range are returned in `to_unmap` so their unshared tiles return to the pool;
    /// exact same-range entries stay mapped because they already share every tile.
    fn insert(&mut self, res: usize, mut entry: Entry) -> (isize, Vec<usize>) {
        let va = entry.va;
        let end = va.saturating_add(entry.tex.size() as usize);
        let mut pending_delta = 0isize;
        let mut to_unmap = Vec::new();
        if let Some(old) = self.by_res.remove(&res) {
            self.by_va.remove(&(old.va, res));
            if old.pending {
                pending_delta -= 1;
            }
        }
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
        }
        for (ova, ores, oend) in self.active_overlaps(va, end) {
            if !entry.placed && ova == va && oend == end && self.by_res.get(&ores).is_some_and(|e| !e.placed) {
                continue;
            }
            self.by_va.remove(&(ova, ores));
            if let Some(mut old) = self.by_res.remove(&ores) {
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped && (ova != va || oend != end) {
                    old.unmapped = true;
                    to_unmap.push(ores);
                }
                self.max_shadowed_size = self.max_shadowed_size.max(oend.saturating_sub(ova));
                self.shadowed_by_va.insert((ova, ores), oend);
                self.shadowed.insert(ores, old);
            }
        }
        // Also release tiles of any already-shadowed exact-range sibling now overwritten by a
        // different range (e.g. A and B shared an unaligned range, then C overwrites both).
        for (sva, sres, send) in self.shadowed_overlaps(va, end) {
            if sva != va || send != end {
                if let Some(old) = self.shadowed.get_mut(&sres) {
                    if old.placed && !old.unmapped {
                        old.unmapped = true;
                        to_unmap.push(sres);
                    }
                }
            }
        }
        self.next_seq += 1;
        entry.seq = self.next_seq;
        if entry.pending {
            pending_delta += 1;
        }
        self.by_va.insert((va, res), end);
        self.by_res.insert(res, entry);
        (pending_delta, to_unmap)
    }

    /// Remove `res` on COM release, reactivating any shadowed older textures it uncovered (newest
    /// first). Returns `(was_present, pending_delta, unmapped_res_to_remap)`.
    fn remove(&mut self, res: usize) -> (bool, isize, Vec<usize>) {
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
            if self.shadowed.is_empty() {
                self.max_shadowed_size = 0;
            }
            return (true, 0, Vec::new());
        }
        let Some(old) = self.by_res.remove(&res) else { return (false, 0, Vec::new()) };
        self.by_va.remove(&(old.va, res));
        let mut pending_delta = if old.pending { -1 } else { 0 };
        let va = old.va;
        let end = va.saturating_add(old.tex.size() as usize);
        let mut cands: Vec<(u64, usize, usize, usize)> = self
            .shadowed_overlaps(va, end)
            .into_iter()
            .filter_map(|(sva, sres, send)| self.shadowed.get(&sres).map(|e| (e.seq, sres, sva, send)))
            .collect();
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut to_remap = Vec::new();
        for (_, sres, sva, send) in cands {
            if self.has_active_overlap(sva, send) {
                continue;
            }
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(mut e) = self.shadowed.remove(&sres) {
                if e.unmapped {
                    e.unmapped = false;
                    to_remap.push(sres);
                    if !e.gpu_written {
                        e.pending = true;
                    }
                }
                if e.pending {
                    pending_delta += 1;
                }
                self.by_va.insert((sva, sres), send);
                self.by_res.insert(sres, e);
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (true, pending_delta, to_remap)
    }

    fn get_mut(&mut self, res: usize) -> Option<&mut Entry> {
        let Self { by_res, shadowed, .. } = self;
        by_res.get_mut(&res).or_else(|| shadowed.get_mut(&res))
    }

    fn overlapping(&self, va: usize, len: usize) -> Vec<(usize, usize)> {
        self.active_overlaps(va, va.saturating_add(len)).into_iter().take(4).map(|(ova, ores, _)| (ores, ova)).collect()
    }
}

fn adjust_pending(delta: isize) {
    if delta > 0 {
        PENDING.fetch_add(delta as usize, Ordering::Relaxed);
    } else if delta < 0 {
        PENDING.fetch_sub((-delta) as usize, Ordering::Relaxed);
    }
}

/// References to textures of the table, taken with its lock held (`real::reference`: one being
/// released for good on another thread is left out).
unsafe fn clone_resources(ptrs: &[usize]) -> Vec<ID3D12Resource> {
    ptrs.iter().filter_map(|&raw| super::real::reference(raw)).collect()
}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<Table> = Mutex::new(Table::new());

/// (address in the title's memory, bytes) of each texture kept in step, for `memdiag`.
pub(super) fn diag_ranges() -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap_or_else(|e| e.into_inner()).by_res.values().map(|e| (e.va, e.tex.size() as usize)).collect()
}
// The dirty-page scan consumes write-watch state for every submission thread.
// Keep the scan, pending-list update and GPU uploads together.
pub(super) static REFRESH: Mutex<()> = Mutex::new(());
static TEXTURES_GEN: AtomicU64 = AtomicU64::new(0);
static TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static WATCH_CACHE: Mutex<Option<((u64, u64, u64, u64, u64), Arc<Vec<(usize, usize)>>)>> = Mutex::new(None);
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES.lock().unwrap().by_res.len(), UPLOADS.load(Ordering::Relaxed))
}

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    // (a new texture object, or the address of an old one again: its first upload is not left out)
    LAST_CONTENT.lock().unwrap().remove(&(res as usize));
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // the console's 32-bit 116 / 117 floats are half floats on the PC (`pc_resource_desc`): the
    // title's bytes are not the resource's texels
    if matches!(u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap()), 116 | 117) {
        return Err("a console float format the PC holds as R16G16B16A16_FLOAT".into());
    }
    // Render targets and depth buffers are the GPU's (their metadata planes, empty planes) –
    // except a small single-sample colour target with texels: a retail title makes its glyph
    // caches that way (256x256 R8 with ALLOW_RENDER_TARGET, two planes) and fills them from the
    // CPU. Left to the GPU they stayed as they were made and its menus' body text lost most of
    // its letters. Such a target is uploaded like any texture when the CPU wrote its pages; one
    // the title only renders to is not CPU-written and so not uploaded over.
    // `XCL_SYNC_SMALL_TARGETS=0` turns this off.
    let small_target = {
        let (w, h) = (u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()), u32::from_le_bytes(desc[0x18..0x1c].try_into().unwrap()));
        let (samples, flags) = (u32::from_le_bytes(desc[0x24..0x28].try_into().unwrap()), u32::from_le_bytes(desc[0x30..0x34].try_into().unwrap()));
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("XCL_SYNC_SMALL_TARGETS").is_ok_and(|v| v.trim() == "0"));
        // (64x64 and up: 1x1 placeholders are not worth an entry; 0x2: a depth buffer)
        !off && w <= 1024 && h <= 1024 && w * h as u64 >= 64 * 64 && samples <= 1 && flags & 0x2 == 0
    };
    if tex.texels().is_none() || (tex.planes.len() != 1 && !small_target) {
        return Err("a render target / depth buffer".into());
    }
    if tex.planes.len() != 1 {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a small render target at {va:#x} ({} planes) is kept in step with the title's memory – a glyph cache the CPU fills ({n} so far)", tex.planes.len());
        }
    }
    if ID3D12Resource::from_raw_borrowed(&res).is_none() {
        return Err("no resource".into());
    }
    track(device, res, va, tex, state, false);
    Ok(())
}

pub(super) unsafe fn tag_created(ppv: *mut *mut c_void, va: u64, desc: [u8; 0x38], initial: [u32; 2]) {
    static FAILED: AtomicU64 = AtomicU64::new(0);
    if ppv.is_null() || (*ppv).is_null() {
        return;
    }
    let host = super::vram::native_image_backing(*ppv as usize);
    let image = texture::image_bridge::GuestImage { va, desc, host, initial };
    if let Err(e) = image.tag(*ppv) {
        if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture at {va:#x}: no guest image tag ({e}); it stays off the native image bridge");
        }
        return;
    }
    super::image_writeback::register(*ppv, &image);
}

pub(super) fn note_cpu_color_view(device: *mut c_void, res: *mut c_void, va: usize, desc: [u8; 0x38], state: D3D12_RESOURCE_STATES) {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};

    let word = |at| u32::from_le_bytes(desc[at..at + 4].try_into().unwrap());
    let format = word(0x20) as i32;
    if word(0) != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32 || word(0x24) != 1
        || (format != DXGI_FORMAT_R32_FLOAT.0 && format != DXGI_FORMAT_R32G32_FLOAT.0)
        || word(0x30) & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 as u32 != 0
    {
        return;
    }
    CPU_COLOR_VIEWS.lock().unwrap().insert(res as usize, CpuColorView { device: device as usize, va, desc, state });
}

fn writable_color_texture(view: &CpuColorView) -> Result<xg::Texture, String> {
    let tex = xg::Texture::new(&view.desc).map_err(|hr| format!("layout {hr:#x}"))?;
    if tex.texels().is_none() || !match tex.planes.as_slice() {
        [color] => color.usage == 1,
        [color, metadata] => color.usage == 1 && metadata.usage == 2,
        _ => false,
    } {
        return Err("unsupported color layout".into());
    }
    if !crate::hle::kernelx::writable_span(view.va, tex.size() as usize) {
        return Err("guest pages are not writable".into());
    }
    Ok(tex)
}

pub(super) unsafe fn gpu_address(res: *mut c_void) -> Option<usize> {
    if let Some(va) = address(res as usize) {
        return Some(va);
    }
    let view = CPU_COLOR_VIEWS.lock().unwrap().get(&(res as usize)).copied()?;
    match writable_color_texture(&view) {
        Ok(tex) => {
            track(view.device as *mut c_void, res, view.va, tex, view.state, true);
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
                let d = ID3D12Resource::from_raw_borrowed(&res)?.GetDesc();
                crate::xlog_info!("D3D12.X: exposed guest address for placed color texture {}x{} (format {})", d.Width, d.Height, d.Format.0);
            }
            Some(view.va)
        }
        Err(error) => {
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_warn!("D3D12.X: placed color texture has no writable guest view ({error})");
            }
            None
        }
    }
}

/// Keeps texture `res` at `va` in step with the title's memory (`only_new`: unless it already is).
unsafe fn track(device: *mut c_void, res: *mut c_void, va: usize, tex: xg::Texture, state: D3D12_RESOURCE_STATES, only_new: bool) {
    let src = mirror_source(va);
    if src != va {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a texture at {va:#x} lies in a GPU view of the title's CPU memory: kept in step with {src:#x} ({n} so far)");
        }
    }
    let in_gpu_view = crate::hle::kernelx::mirrors::source(va).is_some();
    let placed = super::vram::is_placed_texture(res as usize);
    // (one memory) a sampled texture that stands tile for tile on a GPU-written one, with no
    // byte in the title's memory: what it shows is what the GPU wrote there (an R9G9B9E5 view
    // of a buffer a compute pass writes as R32_UINT). It is the GPU's from the start – an
    // upload of the title's zeros went over those tiles – and is kept with that texture
    // (`vram::keep_with`). `XCL_TEXTURE_ONE_MEMORY=0`: uploaded, as before.
    let gpus = placed && super::texture_alias::one_memory()
        && super::vram::gpu_target_on_tiles(res as usize).is_some_and(|by| {
            with_title_bytes(src, tex.size() as usize, |bytes| starts_as_gpus(true, bytes)) && super::vram::keep_with(by, res as usize)
        });
    if gpus {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: one memory: a sampled texture at {va:#x} ({} bytes) stands on the tiles of a GPU-written one and the title's memory there is zero: not uploaded ({n} so far)", tex.size());
        }
    }
    let e = Entry {
        device: device as usize,
        va,
        src,
        in_gpu_view,
        tex: Arc::new(tex),
        states: vec![state],
        pending: !gpus,
        gpu_written: gpus,
        placed,
        unmapped: false,
        seq: 0,
    };
    let to_unmap = {
        let mut t = TEXTURES.lock().unwrap();
        if only_new && t.by_res.contains_key(&(res as usize)) {
            return;
        }
        let (delta, unmap_ptrs) = t.insert(res as usize, e);
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        clone_resources(&unmap_ptrs)
    };
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
}

/// (one memory) Whether a sampled texture starts as the GPU's – never uploaded from the
/// title's memory: it stands tile for tile on a mapped GPU-written texture at its address
/// (`on_gpu_tiles`) and the title's memory under it (`title_bytes`) has no byte that is not
/// zero. With a byte there the CPU filled it – a texture streamed into memory a target was
/// in – and it is uploaded like any other.
fn starts_as_gpus(on_gpu_tiles: bool, title_bytes: &[u8]) -> bool {
    on_gpu_tiles && !super::vram::any_nonzero(title_bytes)
}

/// Where a texture at `va` is read from: the CPU view behind a GPU view (`kernelx::mirrors`)
/// when nothing is mapped at `va` itself, else `va`. A physical page mapped at `va` counts
/// before its first use too, while it is still reserved memory (`kernelx::physical_page_at`).
fn mirror_source(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// Physical pages were mapped or unmapped in a GPU view since the last look: textures there
/// whose source changed (a page now mapped under one, or gone from it) read from the new one,
/// uploaded again at this submit.
fn recheck_gpu_view_sources() {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    if SEEN.swap(gen, Ordering::AcqRel) == gen {
        return;
    }
    let mut changed = false;
    // (the shadowed textures too: one comes back when the texture over it goes)
    let mut g = TEXTURES.lock().unwrap();
    let t = &mut *g;
    for e in t.by_res.values_mut().chain(t.shadowed.values_mut()) {
        changed |= recheck_source(e);
    }
    drop(g);
    if changed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// `e`'s source again (a texture in a GPU view): when it changed, `e` reads the new one and is
/// uploaded at the next submit. Whether it changed.
fn recheck_source(e: &mut Entry) -> bool {
    if !e.in_gpu_view {
        return false;
    }
    let src = mirror_source(e.va);
    if src == e.src {
        return false;
    }
    e.src = src;
    if !e.pending {
        e.pending = true;
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// The title's GPU work writes texture `res` (a copy into it): on the console the memory then
/// holds what the GPU wrote; here the CPU's view of it is stale, so it is not uploaded again
/// (a CPU write elsewhere on its pages – a neighbour, an allocator's header – would upload the
/// stale view over the copy: Fortnite's font atlas is zeros in memory, its glyphs are copied in
/// by the GPU). The first upload, if still pending, stays: it runs before the title's submit
/// that copies.
pub fn note_gpu_write(res: usize) {
    let mut t = TEXTURES.lock().unwrap();
    if let Some(e) = t.get_mut(res) {
        super::physical_directory::note_native_texture_access(e.va as u64, e.tex.size(), true);
        if !e.gpu_written {
            e.gpu_written = true;
            GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static GPU_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// (diagnostics, `alias::report`) The textures the title placed over `[va, va + len)` (the first 4
/// kept in step, 2 shadowed by newer ones): where each is, where its bytes are read from, and
/// whether the GPU wrote it. Not waited for while another thread holds the table.
pub(crate) fn placed_over(va: usize, len: usize) -> Vec<String> {
    let Some(t) = crate::alias::try_lock(&TEXTURES) else { return Vec::new() };
    let end = va.saturating_add(len);
    let line = |res: usize, e: &Entry, state: &str| {
        format!(
            "a texture is placed over this memory ({state}): PC texture {res:#x} at {:#x} + {:#x}, read from {:#x}{}{}",
            e.va,
            e.tex.size(),
            e.src,
            if e.in_gpu_view { " (in a GPU view)" } else { "" },
            if e.gpu_written { ", written by the GPU (no longer uploaded)" } else { "" }
        )
    };
    let mut out: Vec<String> = t.active_overlaps(va, end).into_iter().take(4).filter_map(|(_, res, _)| t.by_res.get(&res).map(|e| line(res, e, "kept in step"))).collect();
    out.extend(t.shadowed_overlaps(va, end).into_iter().take(2).filter_map(|(_, res, _)| t.shadowed.get(&res).map(|e| line(res, e, "shadowed by a newer one"))));
    out
}

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().overlapping(va, len)
}

/// A GPU-owned placement (render target / depth / UAV), `by`, now occupies `[va, va + len)`:
/// retire superseded sampled textures there so CPU dirty scans do not upload over the shared
/// tiles and any unshared extra tiles return to the pool. One that stands tile for tile on
/// `by`'s pages (`vram::keep_with`: the title's view of that memory in a format of its size,
/// made before the target was placed again) has no tile of its own and stays mapped: unmapped
/// it read as zero from then on, and `real::resource_at` hands the same texture out for every
/// later view of that address. It is unmapped when `by`'s pages are released and no other
/// target is on them (`XCL_TEXTURE_ALIAS=0` and `XCL_TEXTURE_ONE_MEMORY=0`: unmapped here, as
/// before).
pub(super) unsafe fn retire_overlapping(va: usize, len: usize, by: usize) {
    if !any() || len == 0 {
        return;
    }
    let (retired, mut to_unmap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (delta, retired, unmap_ptrs) = t.retire_range(va, va.saturating_add(len));
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (retired, clone_resources(&unmap_ptrs))
    };
    if super::texture_alias::enabled() || super::texture_alias::one_memory() {
        to_unmap.retain(|r| !super::vram::keep_with(by, r.as_raw() as usize) && !super::vram::kept_clear_of(r.as_raw() as usize, va, va.saturating_add(len)));
    }
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    if retired > 0 {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    CPU_COLOR_VIEWS.lock().unwrap().remove(&res);
    LAST_CONTENT.lock().unwrap().remove(&res);
    let (removed, to_remap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (removed, delta, remap_ptrs) = t.remove(res);
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (removed, unsafe { clone_resources(&remap_ptrs) })
    };
    for r in &to_remap {
        unsafe {
            super::vram::remap_if_unmapped(r);
        }
    }
    if removed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    let t = TEXTURES.lock().unwrap();
    t.by_res.get(&res).or_else(|| t.shadowed.get(&res)).map(|e| e.va)
}

/// (texture, subresource or ALL_SUBRESOURCES, state after) of a transition the title recorded
type Transition = (usize, u32, D3D12_RESOURCE_STATES);

/// PC command list → the transitions of textures kept in step it recorded since it was made or
/// last reset, in order. They apply when the list is submitted – each time: a closed list may
/// run again.
static LIST_NOTES: Mutex<BTreeMap<usize, Vec<Transition>>> = Mutex::new(BTreeMap::new());

/// Whether any texture is kept in step (callers skip collecting transitions otherwise).
pub fn any() -> bool {
    TEXTURES_LEN.load(Ordering::Relaxed) != 0
}

/// The title recorded `transitions` on PC list `list`: those of textures kept in step are noted
/// for when the list is submitted.
pub fn note_transitions(list: usize, transitions: &[Transition]) {
    if (!any() && !super::image_writeback::active()) || transitions.is_empty() {
        return;
    }
    let kept: Vec<Transition> = {
        let t = TEXTURES.lock().unwrap();
        transitions.iter().filter(|(r, _, _)| t.by_res.contains_key(r) || t.shadowed.contains_key(r) || super::image_writeback::tracks(*r)).copied().collect()
    };
    if !kept.is_empty() {
        LIST_NOTES.lock().unwrap().entry(list).or_default().extend(kept);
    }
}

/// PC list `list` starts a new recording (made, reset, released): its noted transitions go.
pub fn list_reset(list: usize) {
    super::image_writeback::list_reset(list);
    let mut g = LIST_NOTES.lock().unwrap();
    if !g.is_empty() {
        g.remove(&list);
    }
}

/// The transitions noted for the PC lists `lists`, in submission order.
pub(super) fn list_transitions(lists: &[*mut c_void]) -> Vec<Transition> {
    let g = LIST_NOTES.lock().unwrap();
    if g.is_empty() {
        return Vec::new();
    }
    lists.iter().filter_map(|&l| g.get(&(l as usize))).flatten().copied().collect()
}

/// The PC lists `lists` were submitted, in this order: the textures kept in step are in the
/// states their transitions leave them in.
pub fn submitted(lists: &[*mut c_void]) {
    let notes: Vec<Transition> = {
        let g = LIST_NOTES.lock().unwrap();
        if g.is_empty() {
            return;
        }
        lists.iter().filter_map(|&l| g.get(&(l as usize))).flatten().copied().collect()
    };
    if notes.is_empty() {
        return;
    }
    let mut t = TEXTURES.lock().unwrap();
    for (res, sub, after) in notes {
        if let Some(e) = t.get_mut(res) {
            let n = subresource_count(&e.tex);
            apply_transition(&mut e.states, n, sub, after);
        }
    }
}

/// Subresources of a texture kept in step (as `subresources` makes them).
fn subresource_count(tex: &xg::Texture) -> usize {
    tex.texels().map_or(1, |s| s.array as usize * s.chain.levels.len())
}

/// `states` (one for all `n` subresources, or one each) after a transition of subresource `sub`
/// (or all of them) to `after`; a subresource the texture does not have is ignored.
fn apply_transition(states: &mut Vec<D3D12_RESOURCE_STATES>, n: usize, sub: u32, after: D3D12_RESOURCE_STATES) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (n <= 1 && sub == 0) {
        *states = vec![after];
        return;
    }
    let k = sub as usize;
    if k >= n {
        return;
    }
    if states.len() != n {
        *states = vec![states[0]; n];
    }
    states[k] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

thread_local! {
    /// XCL's own submits on this thread (texture uploads): they read no title memory on the GPU,
    /// so they do not look for CPU writes first – that look is the title's submits' job
    static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Reusable `GetWriteWatch` output buffer (avoids 128 KiB alloc+zero+free per submit).
    static WATCH_BUF: std::cell::RefCell<Vec<*mut c_void>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable dirty page list for `refresh_once`.
    static DIRTY_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn cached_watch_ranges(u: &crate::gpu::uma::Uma) -> Arc<Vec<(usize, usize)>> {
    let key = (
        super::sparse::mapped_gen(),
        TEXTURES_GEN.load(Ordering::Acquire),
        crate::hle::kernelx::graphics_reservations_gen(),
        super::external_buffers::generation(),
        super::page_mappings::generation(),
    );
    {
        let g = WATCH_CACHE.lock().unwrap();
        if let Some((k, ref ranges)) = *g {
            if k == key {
                return Arc::clone(ranges);
            }
        }
    }
    // Recompute the watched ranges (textures kept in step + mapped sparse tiles clipped to
    // write-watched regions) only when the set of textures, tiles, or reservations changed.
    let mut regions = u.watch_regions();
    regions.extend(crate::hle::kernelx::graphics_reservations());
    let mut wanted: Vec<(usize, usize)> = TEXTURES.lock().unwrap().by_res.values().map(|e| (e.src & !0xFFF, (e.src + e.tex.size() as usize + 0xFFF) & !0xFFF)).collect();
    wanted.extend(super::sparse::mapped_ranges());
    wanted.extend(super::external_buffers::ranges());
    wanted.extend(super::page_mappings::pool_ranges());
    let computed = Arc::new(watch_ranges(&regions, wanted));
    *WATCH_CACHE.lock().unwrap() = Some((key, Arc::clone(&computed)));
    computed
}

/// memory the CPU wrote that the write watch may not see (`hle::bink`): taken as written by the
/// next submit's look
static CPU_WRITTEN: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// `len` bytes at `at` were written by the CPU: the next submit uploads what is there again
/// (and copies pool pages to the GPU views that show them).
pub fn mark_cpu_written(at: usize, len: usize) {
    CPU_WRITTEN.lock().unwrap().push((at, len));
}

/// Run `f` as XCL's own GPU work: ExecuteCommandLists inside it does not refresh.
pub(super) fn own_work<R>(f: impl FnOnce() -> R) -> R {
    let was = BUSY.with(|b| b.replace(true));
    let r = f();
    BUSY.with(|b| b.set(was));
    r
}

/// Whether the current thread is inside XCL's own internal GPU setup/upload/readback work.
pub(super) fn is_own_work() -> bool {
    BUSY.with(|b| b.get())
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) {
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.get()) {
        return;
    }
    let _refresh = REFRESH.lock().unwrap();
    own_work(|| refresh_once(queue));
}

/// The pages of `[base, base + size)` written since the last look, added to `dirty` and reset
/// (`page`: their size). A full `buf` goes on after the last page it reported: starting again at
/// `base` walked the pages already reset each time (4 GiB with every page written: 105 ms a
/// look, 47 ms going on).
unsafe fn take_written(base: usize, size: usize, buf: &mut [*mut c_void], dirty: &mut Vec<usize>, page: &mut u32) {
    let end = base + size;
    let mut at = base;
    while at < end {
        let mut n = buf.len();
        if GetWriteWatch(WRITE_WATCH_FLAG_RESET, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, page) != 0 {
            break;
        }
        dirty.extend(buf[..n].iter().map(|&p| p as usize));
        if n == 0 || n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + *page as usize;
    }
}

unsafe fn refresh_once(queue: *mut c_void) {
    let Some(u) = crate::gpu::uma::get() else { return };
    let sparse = u.has_sparse();
    if !sparse && TEXTURES_LEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    recheck_gpu_view_sources();
    let watch_t0 = std::time::Instant::now();
    let mut page = 0x1000u32;
    let ranges = cached_watch_ranges(u);
    let dirty_len = DIRTY_BUF.with(|dcell| {
        let mut dirty = dcell.borrow_mut();
        dirty.clear();
        WATCH_BUF.with(|wcell| {
            let mut buf = wcell.borrow_mut();
            if buf.len() < 16384 {
                buf.resize(16384, std::ptr::null_mut());
            }
            for &(base, size) in ranges.iter() {
                take_written(base, size, &mut buf, &mut dirty, &mut page);
            }
        });
        // (with every other address showing the physical pages written: views of a page are not
        // write-watched)
        let marked = std::mem::take(&mut *CPU_WRITTEN.lock().unwrap());
        let aliases = crate::hle::kernelx::alias_ranges(&marked);
        for (at, len) in marked.into_iter().chain(aliases) {
            dirty.extend((at & !0xFFF..at + len).step_by(0x1000));
        }
        super::external_buffers::unwatched_dirty(&mut dirty);
        dirty.sort_unstable();
        super::page_mappings::after_look(&mut dirty, &ranges);
        dirty.sort_unstable();
        super::external_buffers::sync_dirty(&mut dirty, page as usize);
        dirty.dedup();
        dirty.len()
    });
    super::sparse::note_watch(watch_t0.elapsed(), dirty_len, ranges.iter().map(|r| r.1).sum());
    if super::present::trace_frame() {
        // every 600th look: what the CPU's view of a few textures holds now (does the title's
        // data reach the address at all, and is the texture still kept in step?)
        static LOOKS: AtomicU64 = AtomicU64::new(0);
        if LOOKS.fetch_add(1, Ordering::Relaxed) % 600 == 599 {
            let t = TEXTURES.lock().unwrap();
            // 16 samples of 4 KiB over the whole texture (GFX10 keeps the small levels first,
            // level 0 last)
            let content = |e: &Entry| -> usize {
                let size = e.tex.size() as usize;
                (0..16usize)
                    .map(|k| e.src + (size.saturating_sub(4096)) * k / 15)
                    .map(|a| with_title_bytes(a, 4096, |bytes| bytes.iter().filter(|&&b| b != 0).count()))
                    .sum()
            };
            let (mut with, mut without) = (0, 0);
            let mut shown = 0;
            for e in t.by_res.values() {
                let c = content(e);
                if c > 0 { with += 1 } else { without += 1 }
                if c > 0 && shown < 4 {
                    shown += 1;
                    crate::xlog_info!("frame trace: title texture at {:#x} ({} bytes) has data in memory ({c} of 65536 sampled bytes); GPU-written {}", e.va, e.tex.size(), e.gpu_written);
                }
            }
            crate::xlog_info!("frame trace: title textures: {with} with data in memory, {without} all zero; {dirty_len} dirty pages this look");
            // the largest all-zero ones: what memory they are in (a GPU view of pool pages?)
            let mut zero: Vec<&Entry> = t.by_res.values().filter(|e| e.tex.size() >= 0x10000 && content(e) == 0).collect();
            zero.sort_by_key(|e| std::cmp::Reverse(e.tex.size()));
            for e in zero.iter().take(6) {
                crate::xlog_info!("frame trace: zero texture at {:#x} ({} bytes, GPU-written {}): first page {}; middle page {}", e.va, e.tex.size(), e.gpu_written, super::page_mappings::describe(e.va), super::page_mappings::describe(e.va + e.tex.size() as usize / 2));
            }
        }
    }
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        DIRTY_BUF.with(|dcell| {
            let dirty = dcell.borrow();
            super::sparse::before_execute(q, &dirty, page as usize);
            super::vram::before_execute(q);
        });
    }
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 {
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let todo: Vec<(ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.by_res
            .iter_mut()
            .filter(|(_, e)| {
                if e.pending {
                    return true;
                }
                if e.gpu_written {
                    return false;
                }
                let i = dirty.partition_point(|&p| p + page as usize <= e.src);
                dirty.get(i).is_some_and(|&p| p < e.src + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                let Some(r) = super::real::reference(res) else {
                    // being released on another thread: for good, or it is uploaded next time
                    if !e.pending {
                        e.pending = true;
                        PENDING.fetch_add(1, Ordering::Relaxed);
                    }
                    return None;
                };
                if e.pending {
                    e.pending = false;
                    PENDING.fetch_sub(1, Ordering::Relaxed);
                }
                super::physical_directory::note_native_texture_access(e.va as u64, e.tex.size(), false);
                Some((r, (e.device, e.src, Arc::clone(&e.tex), e.states.clone())))
            })
            .collect()
    });
    upload_all(todo);
}

/// texture → `content_hash` of its memory at its last upload
static LAST_CONTENT: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());
static UNCHANGED: AtomicU64 = AtomicU64::new(0);

/// `XCL_TEXTURE_SKIP_UNCHANGED=0` uploads a texture whenever its pages were written, as before.
fn unchanged_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_SKIP_UNCHANGED").map_or(true, |v| v.trim() != "0"))
}

/// Whether texture `res` was last uploaded from memory with hash `h`; `h` is remembered as its
/// content from here on (the caller uploads it when this is false).
fn same_as_last_upload(res: usize, h: u64) -> bool {
    LAST_CONTENT.lock().unwrap().insert(res, h) == Some(h)
}

/// A 64-bit hash of `b` for "the same bytes as before" (four lanes of multiply-rotate over
/// 8-byte words: gigabytes a second; not for anything adversarial).
fn content_hash(b: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = [K, K.rotate_left(17), K.rotate_left(31), K.rotate_left(47)];
    let mut chunks = b.chunks_exact(32);
    for c in &mut chunks {
        for (k, lane) in h.iter_mut().enumerate() {
            let v = u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
            *lane = (*lane ^ v).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut tail = b.len() as u64;
    for (k, &x) in chunks.remainder().iter().enumerate() {
        tail = (tail ^ ((x as u64) << (8 * (k % 8)))).wrapping_mul(K).rotate_left(23);
    }
    h.iter().fold(tail, |a, &l| (a ^ l).wrapping_mul(K).rotate_left(27))
}

/// staging bytes and textures of one batch at most (a texture larger than that is a batch of
/// its own). 256 MiB a batch kept an upload buffer of that size, and as much of detiled texels
/// on the heap while the batch was made (upstream XCL fb9d059: a count and 32 MiB).
const BATCH_BYTES: usize = 32 << 20;
const BATCH_COUNT: usize = 256;

/// Whether a batch of `picked` textures of `bytes` in all is closed before a texture of `next`
/// bytes: the textures of a batch, and the batches, are in the order the uploads were in.
fn batch_full(picked: usize, bytes: usize, next: usize) -> bool {
    picked != 0 && (picked >= BATCH_COUNT || bytes.saturating_add(next) > BATCH_BYTES)
}

/// The uploads `todo`, in batches of up to `BATCH_BYTES` (one list and one wait each).
unsafe fn upload_all(todo: Vec<(ID3D12Resource, Upload)>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    let mut todo = todo.into_iter().peekable();
    while todo.peek().is_some() {
        let mut picked = Vec::new();
        let mut bytes = 0usize;
        let mut device = 0usize;
        let mut spans = BTreeMap::new();
        while let Some((_, (dev, va, tex, _))) = todo.peek() {
            let end = va.saturating_add((tex.size() as usize).max(1));
            let overlaps = spans.range(..end).next_back().is_some_and(|(_, &previous_end)| previous_end > *va);
            if batch_full(picked.len(), bytes, tex.size() as usize) || (!picked.is_empty() && (*dev != device || overlaps)) {
                break;
            }
            let (r, (dev, va, tex, states)) = todo.next().unwrap();
            // The same bytes as at this texture's last upload: nothing to do. The write watch is
            // per page and a title writes next to its textures (and rewrites what it wrote): a
            // retail title's video planes and UI targets went up again for every submit, ~25 MB
            // a presented frame where ~3 MB had changed, and the submits waited here.
            if unchanged_skip() {
                let h = with_title_bytes(va, tex.size() as usize, content_hash);
                if same_as_last_upload(r.as_raw() as usize, h) {
                    let n = UNCHANGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n == 1 || (n.is_power_of_two() && n >= 1024) {
                        crate::xlog_info!("title textures: {n} uploads left out so far – the texture's memory held what it held at its last upload");
                    }
                    continue;
                }
            }
            device = dev;
            bytes += tex.size() as usize;
            spans.insert(va, end);
            picked.push((r, va, tex, states));
        }
        let subs = subresources_of_all(&picked, bytes);
        let mut batch = Vec::with_capacity(picked.len());
        for ((r, va, tex, states), subs) in picked.into_iter().zip(subs) {
            if super::present::trace_frame() {
                note_content(va, &r, &subs);
            }
            // video memory under the mips that have texels, before the copies into them
            if !subs.is_empty() && super::vram::wants_content(r.as_raw() as usize) {
                super::vram::note_content(&r, &with_texels(va, &tex, super::vram::zero_is_unmapped(r.GetDesc().Format), &subs));
            }
            batch.push((r, subs, states));
        }
        let d = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { continue };
        let queue = match upload_queue(dev, device) {
            Ok(q) => q,
            Err(e) => {
                warn(&e);
                continue;
            }
        };
        match own_work(|| {
            super::vram::before_execute(&queue);
            texture::upload_batch(dev, &queue, &batch)
        }) {
            Ok(failed) => {
                for (_, e) in failed {
                    warn(&e);
                }
                UPLOADS.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
            Err(e) => warn(&e),
        }
    }
}

/// `XCL_TRACE_FRAME`: how much of a title texture's top level is not zero, each time it is
/// uploaded (the first 48 uploads of textures of 64x64 and more).
fn note_content(va: usize, r: &ID3D12Resource, subs: &[texture::Subresource]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let Some(top) = subs.first() else { return };
    let d = unsafe { r.GetDesc() };
    if d.Width < 64 || d.Height < 64 || N.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let nonzero = top.data.iter().filter(|&&b| b != 0).count();
    crate::xlog_info!(
        "frame trace: title texture at {va:#x} ({}x{} format {}) uploaded: {nonzero} of {} bytes of level 0 not zero",
        d.Width,
        d.Height,
        d.Format.0,
        top.data.len()
    );
}

/// `XCL_TRACE_FRAME`: what the GPU holds in level 0 of title texture `res` (read back on XCL's
/// queue; the texture is taken to be in a shader-resource state) against the CPU's view of the
/// memory now. Each texture once, four at most.
pub unsafe fn probe_gpu_content(res: usize) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&res) || g.len() >= 4 {
            return;
        }
        g.push(res);
    }
    let Some((device, va, tex, gpu_written, r)) = ({
        let t = TEXTURES.lock().unwrap();
        t.by_res.get(&res).and_then(|e| Some((e.device, e.src, Arc::clone(&e.tex), e.gpu_written, super::real::reference(res)?)))
    }) else {
        return;
    };
    let d = device as *mut c_void;
    let (Some(dev), r) = (ID3D12Device::from_raw_borrowed(&d), &r) else { return };
    let rd = r.GetDesc();
    let result = read_gpu_subresource(dev, r, 0, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    let cpu = subresources(va, &tex).into_iter().next().map(|s| s.data).unwrap_or_default();
    match result {
        Ok(gpu) => {
            let nonzero = gpu.iter().filter(|&&b| b != 0).count();
            let same = gpu.iter().zip(cpu.iter()).filter(|(a, b)| a == b).count();
            crate::xlog_info!(
                "frame trace: GPU content of title texture {res:#x} at {va:#x} ({}x{} format {}, {} mips, GPU-written {gpu_written}): {nonzero} of {} bytes of level 0 not zero; {same} equal to the CPU view ({} of its bytes not zero); first bytes {:02x?}",
                rd.Width, rd.Height, rd.Format.0, rd.MipLevels, gpu.len(), cpu.iter().filter(|&&b| b != 0).count(), &gpu[..gpu.len().min(16)]
            );
        }
        Err(e) => crate::xlog_warn!("frame trace: GPU content of title texture {res:#x}: {e}"),
    }
}

pub(super) unsafe fn read_gpu_subresource(dev: &ID3D12Device, r: &ID3D12Resource, subresource: u32, state: D3D12_RESOURCE_STATES) -> Result<Vec<u8>, String> {
    let rd = r.GetDesc();
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
    dev.GetCopyableFootprints(&rd, subresource, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    own_work(|| -> Result<Vec<u8>, String> {
        let queue = upload_queue(dev, dev.as_raw() as usize)?;
        super::vram::before_execute(&queue);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
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
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| e.to_string())?;
        let rb = rb.ok_or("no readback buffer")?;
        let barrier = |a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: subresource, StateBefore: a, StateAfter: b }),
            },
        };
        list.ResourceBarrier(&[barrier(state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: subresource } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
        texture::wait(dev, &queue)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
        for y in 0..rows as usize {
            out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
        }
        rb.Unmap(0, None);
        Ok(out)
    })
}

/// XCL's queue for uploads on `dev`.
unsafe fn upload_queue(dev: &ID3D12Device, device: usize) -> Result<ID3D12CommandQueue, String> {
    let mut q = QUEUES.lock().unwrap();
    if let Some(q) = q.get(&device) {
        return Ok(q.clone());
    }
    let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
    q.insert(device, made.clone());
    Ok(made)
}

/// Ranges closer than this are scanned as one. A write-watch call costs ~3.5 µs however little it
/// covers, while one call over 2 GiB of committed pages takes ~1.3 ms (measured on Windows 11): a
/// retail game's intro video had its submitting thread at 42 % in GetWriteWatch with a 1 MiB gap
/// (thousands of calls before each submit over its mapped sparse tiles). The pages of a gap are
/// reported as written too; the uploads keep to what they track.
const WATCH_GAP: usize = 64 << 20;

/// The parts of `wanted` (start, end) inside `regions` (base, size), merged (gaps under
/// `WATCH_GAP` too) and sorted, as (base, size) – each within one region (a write-watch call
/// covers one allocation).
fn watch_ranges(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in wanted {
        match merged.last_mut() {
            Some(last) if a <= last.1.saturating_add(WATCH_GAP) => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    // each region against the merged ranges it meets (sorted and disjoint: a binary search, not
    // a pass over all of them – both lists run to thousands while a game streams)
    let mut out = Vec::new();
    for &(base, size) in regions {
        let end = base + size;
        let first = merged.partition_point(|&(_, b)| b <= base);
        for &(a, b) in merged[first..].iter().take_while(|&&(a, _)| a < end) {
            let (lo, hi) = (a.max(base), b.min(end));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
#[cfg(test)]
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    copy_title(at, &mut out);
    out
}

/// Which subresources of the texture at `va` (`subs`: all of them, as read from there) have
/// texels that a tile without video memory would not read as (`vram::note_content`): those
/// with a byte that is not zero – and, when zero texels of its format are not what such a tile
/// reads as (`zero_is_unmapped` false), those the title has memory under at all.
unsafe fn with_texels(va: usize, tex: &xg::Texture, zero_is_unmapped: bool, subs: &[Subresource]) -> Vec<bool> {
    let surface = tex.texels();
    subs.iter().enumerate().map(|(k, s)| {
        if super::vram::any_nonzero(&s.data) {
            return true;
        }
        if zero_is_unmapped {
            return false;
        }
        // where the subresource is in the title's memory: its level's span of its slice (a 3D
        // texture's levels are in every layer: all of it)
        let (at, len) = match surface {
            Some(t) if t.depth == 0 && !t.chain.levels.is_empty() => {
                let span = t.chain.level_span(k % t.chain.levels.len());
                (va + (k / t.chain.levels.len()) * t.chain.slice_bytes + span.start, span.len())
            }
            _ => (va, tex.size() as usize),
        };
        !committed_runs(at, len).is_empty()
    }).collect()
}

/// Batches of at least this many bytes are detiled on several threads (`subresources_of_all`)
const PARALLEL_BYTES: usize = 1 << 20;

/// `subresources` of every texture of a batch (`bytes` of them), in order: on several threads
/// when the batch is big. They were read and detiled one after another in the title's own
/// submit – most of a retail game's render thread while it streamed textures the CPU writes.
unsafe fn subresources_of_all(picked: &[(ID3D12Resource, usize, Arc<xg::Texture>, Vec<D3D12_RESOURCE_STATES>)], bytes: usize) -> Vec<Vec<Subresource>> {
    let picked: Vec<(usize, &xg::Texture)> = picked.iter().map(|(_, va, tex, _)| (*va, &**tex)).collect();
    let threads = if bytes < PARALLEL_BYTES { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get() / 2).clamp(1, 8) };
    subresources_on(&picked, threads)
}

/// `subresources` of each (address, texture), in order, on up to `threads` threads.
unsafe fn subresources_on(picked: &[(usize, &xg::Texture)], threads: usize) -> Vec<Vec<Subresource>> {
    let threads = threads.min(picked.len());
    if threads <= 1 {
        return picked.iter().map(|&(va, tex)| subresources(va, tex)).collect();
    }
    // the next texture to take; each thread hands back (index, subresources)
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Vec<Subresource>> = (0..picked.len()).map(|_| Vec::new()).collect();
    let done: Vec<Vec<(usize, Vec<Subresource>)>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(va, tex)) = picked.get(k) else { break };
                        // SAFETY: as in the serial path – the title's memory is only read
                        mine.push((k, unsafe { subresources(va, tex) }));
                    }
                    mine
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap_or_default()).collect()
    });
    for (k, subs) in done.into_iter().flatten() {
        out[k] = subs;
    }
    out
}

/// A thread's scratch for `with_title_bytes`, counted in `memdiag::TITLE_SCRATCH` while it holds
/// memory.
struct Scratch(Vec<u8>);

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.capacity() > 0 {
            crate::memdiag::TITLE_SCRATCH.sub(self.0.capacity() as u64);
        }
    }
}

thread_local! {
    static TITLE_BYTES: std::cell::RefCell<Scratch> = const { std::cell::RefCell::new(Scratch(Vec::new())) };
}

/// `f` with the title's `len` bytes at `at` as a copy of XCL's own, read so that memory the
/// title unmaps, decommits or frees meanwhile reads as zeros (`memmap::copy_title`): every bulk
/// read of the title's memory that XCL makes on its own schedule goes through here – its
/// textures (detile, content hash, frame sample) and the buffers it placed outside unified
/// memory (`external_buffers`). "Readable" a moment ago is not readable now.
pub(crate) unsafe fn with_title_bytes<R>(at: usize, len: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    const MAX_RETAINED_BYTES: usize = 32 << 20;
    TITLE_BYTES.with(|scratch| {
        if len <= MAX_RETAINED_BYTES {
            if let Ok(mut scratch) = scratch.try_borrow_mut() {
                let bytes = &mut scratch.0;
                if bytes.len() < len {
                    let was = bytes.capacity();
                    let additional = len - bytes.len();
                    bytes.reserve_exact(additional);
                    bytes.resize(len, 0);
                    if was > 0 {
                        crate::memdiag::TITLE_SCRATCH.sub(was as u64);
                    }
                    crate::memdiag::TITLE_SCRATCH.add(bytes.capacity() as u64);
                }
                copy_title(at, &mut bytes[..len]);
                return f(&bytes[..len]);
            }
        }
        let mut bytes = vec![0; len];
        copy_title(at, &mut bytes);
        f(&bytes)
    })
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        return with_title_bytes(va, tex.size() as usize, |mem| {
            c.levels.iter().enumerate().map(|(i, l)| Subresource { depth: l.depth, rows: l.height, row_bytes: l.width * bpe, data: c.detile_volume(mem, i) }).collect()
        });
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        with_title_bytes(va + s * c.slice_bytes, c.slice_bytes, |slice| {
            for (i, l) in c.levels.iter().enumerate() {
                out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(slice, i, s) });
            }
        });
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_images_keep_original_slot44_and_slot133_layouts_after_shadowing() {
        use crate::gpu::texture::image_bridge::{GuestImage, ImageUse};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let _vram = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _textures = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else {
                crate::gpu::recomp::runner::skip_test("no adapter");
                return;
            };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                crate::gpu::recomp::runner::skip_test("software adapter: placed video memory is a real-GPU path");
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no D3D12 device");
                return;
            }
            let dev = dev.unwrap();
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                crate::gpu::recomp::runner::skip_test("tiled tier < 2");
                return;
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory = vec![0u8; 2<<20];
            let va = (memory.as_ptr() as u64).next_multiple_of(65536);
            let a = console_desc(3, 64, 64, 1, 1, 28, 0x105);
            let b = console_desc(3, 128, 64, 1, 1, 28, 0x105);
            let iid = (&ID3D12Resource::IID as *const windows::core::GUID).cast();
            let mut r = std::ptr::null_mut();
            assert!(super::super::device_x::create_placed_resource_x(dev.as_raw(), va, a.as_ptr(), 0, std::ptr::null(), iid, &mut r) >= 0);
            let first = ID3D12Resource::from_raw(r);
            r = std::ptr::null_mut();
            assert!(super::super::device_x::create_placed_resource_layout_x(dev.as_raw(), va, b.as_ptr(), D3D12_BARRIER_LAYOUT_COMMON, std::ptr::null(), iid, &mut r) >= 0);
            let second = ID3D12Resource::from_raw(r);
            for (resource, desc, initial) in [(&first, a, [0,0]),(&second, b, [1,D3D12_BARRIER_LAYOUT_COMMON.0 as u32])] {
                let p = GuestImage::get(resource).unwrap();
                assert_eq!((p.va, p.desc, p.initial), (va, desc, initial));
                assert!(p.permits(ImageUse::Copy) && p.permits(ImageUse::Sample));
                assert!(!p.permits(ImageUse::UnorderedAccess) && !p.permits(ImageUse::RenderTarget));
            }
        }
    }

    /// A texture is left out only when its memory hashes as at its own last upload, and a
    /// texture that was forgotten (released, or its address registered again) is uploaded again.
    #[test]
    fn a_texture_with_unchanged_memory_is_left_out_until_it_is_forgotten() {
        let (a, b) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize);
        let mut mem = vec![0u8; 3 << 20];
        mem[5] = 1;
        let h = content_hash(&mem);
        assert!(!same_as_last_upload(a, h), "the first upload is not left out");
        assert!(same_as_last_upload(a, h), "the same bytes again: left out");
        assert!(!same_as_last_upload(b, h), "another texture over the same bytes has its own first upload");
        // one byte anywhere, also in the tail that is not a whole 32-byte step
        for at in [0usize, 31, 32, (3 << 20) - 1] {
            let mut m = mem.clone();
            m[at] ^= 0x80;
            assert_ne!(content_hash(&m), h, "byte {at} changed");
        }
        mem[(3 << 20) - 1] = 9;
        let h2 = content_hash(&mem);
        assert!(!same_as_last_upload(a, h2), "changed memory is uploaded");
        assert!(same_as_last_upload(a, h2));
        // the length is part of the content: a prefix of zeros is not the whole
        assert_ne!(content_hash(&[0u8; 64]), content_hash(&[0u8; 96]));
        forget(a);
        assert!(!same_as_last_upload(a, h2), "forgotten: its next upload is a first one");
        forget(a);
        forget(b);
    }

    /// Real uploads, dirty-page detection and a source changing after registration:
    /// bookkeeping alone cannot show that the bytes the GPU sees follow the mirror.
    #[test]
    fn gpu_mirror_upload_follows_cpu_writes_and_later_mapping_changes() {
        use crate::hle::kernelx;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
        use windows_sys::Win32::System::Memory as mem;

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        // Larger than all four UMA windows together. These public allocations
        // must use the watched Windows fallback, reserving space rather than RAM
        // (a reservation of placeholders shows the physical page itself, not a
        // lazily filled copy, `kernelx::physical`).
        let _copies = kernelx::without_page_views();
        const SIZE: usize = 24usize << 30;
        const OFFSET: usize = 0x2_0000;
        const PAGE: usize = 0x1_0000;
        struct Reservation(usize);
        impl Drop for Reservation {
            fn drop(&mut self) {
                unsafe { kernelx::VirtualFree(self.0 as *mut c_void, 0, mem::MEM_RELEASE); }
            }
        }
        struct RegisteredTexture(usize);
        impl Drop for RegisteredTexture {
            fn drop(&mut self) { forget(self.0); }
        }
        struct PhysicalPage { id: usize, at: usize }
        impl Drop for PhysicalPage {
            fn drop(&mut self) {
                unsafe {
                    if kernelx::physical_page_at(self.at) {
                        kernelx::MapTitlePhysicalPages(self.at as *mut c_void, 1, 0, 4, std::ptr::null_mut());
                    }
                    kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, &mut self.id);
                }
            }
        }
        unsafe fn reserve(protect: u32) -> Reservation {
            let at = kernelx::VirtualAlloc(std::ptr::null(), SIZE, 0x2040_2000, protect) as usize;
            assert_ne!(at, 0, "reserve-only fallback must succeed");
            let reservation = Reservation(at);
            assert!(!crate::gpu::uma::get().unwrap().contains(at), "exercise the Windows fallback");
            assert!(kernelx::in_graphics_reservation(at), "fallback must be write-watched");
            reservation
        }
        unsafe fn read_r8(dev: &ID3D12Device, queue: &ID3D12CommandQueue, texture: &ID3D12Resource) -> Vec<u8> {
            own_work(|| {
                let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut total = 0;
                dev.GetCopyableFootprints(&texture.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
                let mut readback = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback,
                ).unwrap();
                let readback: ID3D12Resource = readback.unwrap();
                let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                let barrier = |before, after| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(texture), Subresource: 0, StateBefore: before, StateAfter: after }) },
                    ..Default::default()
                };
                list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(texture), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                crate::gpu::fence::wait(&fence, 1, dev, None, 5_000).unwrap();
                let mut data = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut data)).unwrap();
                let mut out = Vec::with_capacity(64 * 64);
                for y in 0..64 {
                    out.extend_from_slice(std::slice::from_raw_parts((data as *const u8).add(footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize), 64));
                }
                readback.Unmap(0, None);
                out
            })
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let cpu = reserve(mem::PAGE_READWRITE);
            let gpu = reserve(0x801);
            let (src, va) = (cpu.0 + OFFSET, gpu.0 + OFFSET);
            assert_eq!(kernelx::mirrors::source(va), Some(src));
            assert_eq!(kernelx::VirtualAlloc(src as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, src);
            let console = console_desc(3, 64, 64, 1, 1, DXGI_FORMAT_R8_UNORM.0 as u32, 1);
            assert!(xg::Texture::new(&console).unwrap().size() <= PAGE as u64);
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = texture.as_raw() as usize;
            register(raw_device, texture.as_raw(), va, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
            let _registered = RegisteredTexture(key);
            for value in [0x3D, 0x9A] {
                std::ptr::write_bytes(src as *mut u8, value, PAGE);
                refresh(queue.as_raw());
                assert_eq!(read_r8(dev, &queue, &texture), vec![value; 64 * 64], "GPU contents follow CPU writes");
                assert_eq!(address(key), Some(va), "the title address stays in the GPU view");
            }
            let (mut count, mut id) = (1usize, 0usize);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, &mut id), 1);
            let mut mapped = PhysicalPage { id, at: va };
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, &mut mapped.id) as usize, va);
            assert!(kernelx::physical_page_at(va));
            let mut info: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(mem::VirtualQuery(va as *const c_void, &mut info, std::mem::size_of_val(&info)), 0);
            assert_eq!(info.State, mem::MEM_RESERVE, "exercise the lazy mapping before its first touch");
            assert_eq!(mirror_source(va), va, "a mapped lazy page overrides the CPU mirror");
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0; 64 * 64], "the untouched physical page supplies zeros");
            std::ptr::write_bytes(va as *mut u8, 0xC7, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0xC7; 64 * 64]);
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, va);
            assert!(!kernelx::physical_page_at(va));
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x9A; 64 * 64], "unmapping restores the earlier CPU view");
            assert_eq!(address(key), Some(va));
            dev.GetDeviceRemovedReason().unwrap();
        }
    }

    /// A physical page mapped into a GPU view but not used yet (reserved until its first
    /// access, `kernelx::lazy`) is that address's own source, not the CPU view behind it.
    /// (Test by Kalmarv, review of PR #283.)
    #[test]
    fn a_lazily_mapped_page_in_a_gpu_view_is_its_own_source() {
        use crate::hle::kernelx;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const VIEW: usize = 16 << 20;
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let cpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            let gpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            kernelx::mirrors::reserved(cpu, VIEW, 0x2040_2000, 0x04);
            kernelx::mirrors::reserved(gpu, VIEW, 0x2040_2000, 0x801);
            assert_eq!(mirror_source(gpu + 0x2_0000), cpu + 0x2_0000); // nothing mapped: the CPU view
            let (mut n, mut pages) = (1usize, [0usize; 1]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let at = gpu + 0x2_0000;
            // a texture placed there before the page is mapped reads the CPU view …
            let mut d = [0u8; 0x38];
            d[0..4].copy_from_slice(&3u32.to_le_bytes());
            d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
            d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
            d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
            d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
            let tex = Arc::new(xg::Texture::new(&d).unwrap());
            let mut e = Entry { device: 0, va: at, src: mirror_source(at), in_gpu_view: true, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: false, placed: false, unmapped: false, seq: 0 };
            assert_eq!(e.src, cpu + 0x2_0000);
            assert!(!recheck_source(&mut e), "nothing changed yet");
            assert_eq!(kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, at);
            // … and the page mapped under it later, uploaded again
            assert!(recheck_source(&mut e));
            assert_eq!(e.src, at);
            assert!(e.pending);
            PENDING.fetch_sub(1, Ordering::Relaxed);
            let before = mirror_source(at);
            (at as *mut u32).write_volatile(7); // its first use commits it
            let after = mirror_source(at);
            let mut none = [0usize; 1];
            kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr());
            kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr());
            kernelx::mirrors::released(cpu);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
            assert_eq!(after, at);
            assert_eq!(before, at, "a page mapped at the address but not used yet is its source too");
        }
    }

    /// More written pages than one call reports: each is found once, in order, and the look
    /// resets them – a buffer that fills in the middle of the range and one that fills exactly.
    #[test]
    fn written_pages_past_a_full_buffer_are_all_found_once() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 1000;
        const PAGE: usize = 0x1000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert!(base != 0);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 64];
            let (mut dirty, mut page) = (Vec::new(), 0u32);
            take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
            for written in [(0..PAGES).filter(|i| i % 7 != 3).collect::<Vec<_>>(), (0..128).map(|i| i * 5).collect()] {
                let want: Vec<usize> = written.iter().map(|i| base + i * PAGE).collect();
                for &a in &want {
                    (a as *mut u8).write_volatile(1);
                }
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
                assert_eq!(page as usize, PAGE);
                assert_eq!(dirty, want, "{} pages written", want.len());
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
                assert!(dirty.is_empty(), "the look reset what it reported");
            }
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn texture_read_preserves_pages_on_either_side_of_a_hole() {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
        };
        const PAGE: usize = 4096;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            base.write_bytes(0x35, PAGE);
            base.add(2 * PAGE).write_bytes(0x72, PAGE);
            assert_ne!(VirtualFree(base.add(PAGE).cast(), PAGE, MEM_DECOMMIT), 0);
            let mut out = [0; 2 * PAGE + 17];
            crate::memmap::copy_committed_region(base as usize + 23, out.as_mut_ptr(), out.len());
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            assert!(out[..PAGE - 23].iter().all(|&b| b == 0x35));
            assert!(out[PAGE - 23..2 * PAGE - 23].iter().all(|&b| b == 0));
            assert!(out[2 * PAGE - 23..].iter().all(|&b| b == 0x72));
        }
    }

    #[test]
    fn volume_upload_has_one_subresource_per_mip() {
        let mut d = [0u8; 0x38];
        d[..4].copy_from_slice(&4u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&17u64.to_le_bytes());
        d[0x18..0x1c].copy_from_slice(&9u32.to_le_bytes());
        d[0x1c..0x1e].copy_from_slice(&9u16.to_le_bytes());
        d[0x1e..0x20].copy_from_slice(&3u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2c..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let t = xg::Texture::new(&d).unwrap();
        let c = &t.texels().unwrap().chain;
        let mut tiled = vec![0u8; t.size() as usize];
        let linear: Vec<Vec<u8>> = c.levels.iter().enumerate().map(|(i, m)| {
            let pixels = (0..m.width * m.height * m.depth * 4).map(|k| (k as u8).wrapping_add(37 * i as u8)).collect::<Vec<_>>();
            c.tile_volume(&mut tiled, i, &pixels);
            pixels
        }).collect();
        let subs = unsafe { subresources(tiled.as_ptr() as usize, &t) };
        assert_eq!(subs.len(), 3);
        for (i, (sub, depth)) in subs.iter().zip([9, 4, 2]).enumerate() {
            assert_eq!((sub.depth, sub.rows, sub.row_bytes), (depth, (9 >> i).max(1), (17 >> i).max(1) * 4));
            assert_eq!(sub.data, linear[i]);
        }
    }

    const SRV: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// One state while every subresource has it, one each while they differ.
    #[test]
    fn transitions_per_subresource_and_back_to_one_state() {
        let mut s = vec![D3D12_RESOURCE_STATE_COMMON];
        apply_transition(&mut s, 4, ALL, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 2, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 9, D3D12_RESOURCE_STATE_COPY_DEST); // no subresource 9
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 2, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        apply_transition(&mut s, 4, ALL, D3D12_RESOURCE_STATE_COMMON);
        assert_eq!(s, vec![D3D12_RESOURCE_STATE_COMMON]);
        // a texture of one subresource: subresource 0 is all of it
        let mut one = vec![SRV];
        apply_transition(&mut one, 1, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(one, vec![D3D12_RESOURCE_STATE_COPY_SOURCE]);
    }

    /// A texture kept in step: the title's transitions count once their list is submitted, in
    /// submission order, each time it is; a Reset forgets them; other resources are not noted.
    #[test]
    fn transitions_apply_when_their_list_is_submitted() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // RGBA8 64×64, 3 mips, 2 slices (6 subresources), 4KB_S
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&2u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&3u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        assert_eq!(subresource_count(&tex), 6);
        // fake resource / list keys, never dereferenced: GPU-written, so no refresh uploads it
        let (res, other, l1, l2) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize, 0x7E57_0000_3000usize, 0x7E57_0000_4000usize);
        TEXTURES.lock().unwrap().insert(res, Entry { device: 0, va: 0x7E57_0000_0000, src: 0x7E57_0000_0000, in_gpu_view: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, placed: false, unmapped: false, seq: 0 });
        TEXTURES_LEN.fetch_add(1, Ordering::Relaxed);
        let states = || TEXTURES.lock().unwrap().by_res.get(&res).unwrap().states.clone();
        let (p1, p2) = (l1 as *mut c_void, l2 as *mut c_void);

        note_transitions(l1, &[(res, ALL, SRV), (other, ALL, D3D12_RESOURCE_STATE_RENDER_TARGET), (res, 4, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        assert_eq!(LIST_NOTES.lock().unwrap().get(&l1).map(|v| v.len()), Some(2), "only the texture kept in step is noted");
        assert_eq!(states(), vec![D3D12_RESOURCE_STATE_COMMON], "recorded, not run yet");
        note_transitions(l2, &[(res, 4, SRV), (res, 0, D3D12_RESOURCE_STATE_COPY_DEST)]);
        submitted(&[p1]);
        let mut want = vec![SRV; 6];
        want[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        assert_eq!(states(), want);
        submitted(&[p2]);
        want[4] = SRV;
        want[0] = D3D12_RESOURCE_STATE_COPY_DEST;
        assert_eq!(states(), want);
        // l1 runs again (a closed list may): its transitions again, after l2's
        submitted(&[p2, p1]);
        assert_eq!(states(), {
            let mut w = vec![SRV; 6];
            w[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            w
        });
        list_reset(l1);
        list_reset(l2);
        submitted(&[p1, p2]);
        assert_eq!(states()[4], D3D12_RESOURCE_STATE_COPY_SOURCE, "a reset list has nothing to apply");
        forget(res);
        assert!(LIST_NOTES.lock().unwrap().get(&l1).is_none());
    }

    /// A texture kept in step is forgotten before its last reference goes, not after: a refresh on
    /// another thread takes its reference for an upload from the table, and between the Release
    /// that ended the object and a forget after it the table named an object that was gone.
    #[test]
    fn a_texture_is_forgotten_before_its_last_reference_goes() {
        use std::sync::atomic::AtomicU32;
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // an object as `real::adopt` leaves one: XCL's vtable, the original one slot before it
        #[repr(C)]
        struct Object {
            vtable: *const usize,
            references: AtomicU32,
            /// 0: alive; when its last reference went: 1 forgotten by then, 2 still in the table
            at_its_end: AtomicU32,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const Object)).references.fetch_add(1, Ordering::SeqCst) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let o = &*(this as *const Object);
            let n = o.references.fetch_sub(1, Ordering::SeqCst) - 1;
            if n == 0 {
                // what a refresh on another thread finds from here on
                o.at_its_end.store(1 + u32::from(address(this as usize).is_some()), Ordering::SeqCst);
            }
            n
        }
        let original = [0usize, add_ref as usize, release as usize];
        let ours = [original.as_ptr() as usize, 0, add_ref as usize, super::super::real::res_release as usize];
        let object = Object { vtable: ours[1..].as_ptr(), references: AtomicU32::new(2), at_its_end: AtomicU32::new(0) };
        let res = &object as *const Object as usize;

        // RGBA8 64×64, one mip, 4KB_S
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        let va = 0x7E57_0010_0000usize;
        {
            let mut t = TEXTURES.lock().unwrap();
            t.insert(res, Entry { device: 0, va, src: va, in_gpu_view: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, placed: false, unmapped: false, seq: 0 });
            TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        }
        unsafe {
            assert_eq!(super::super::real::res_release(res as *mut c_void), 1);
            assert_eq!(address(res), Some(va), "a reference is left: still kept in step");
            assert_eq!(object.at_its_end.load(Ordering::SeqCst), 0);
            assert_eq!(super::super::real::res_release(res as *mut c_void), 0);
        }
        assert_eq!(object.at_its_end.load(Ordering::SeqCst), 1, "the table had forgotten the texture when its last reference went");
        assert_eq!(address(res), None);
    }

    /// Two references given up at the same moment: neither Release sees the last one coming, so
    /// the texture is still in the table when the object ends. Until the table has forgotten it,
    /// nothing takes a reference to it from the table (`real::reference`).
    #[test]
    fn no_reference_is_taken_from_the_table_once_the_last_release_is_under_way() {
        use std::sync::atomic::{AtomicBool, AtomicU32};
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        #[repr(C)]
        struct Object {
            vtable: *const usize,
            references: AtomicU32,
            releases: AtomicU32,
            ended: AtomicBool,
            asked: AtomicBool,
            referenced_after_its_end: AtomicU32,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let o = &*(this as *const Object);
            if o.ended.load(Ordering::SeqCst) {
                o.referenced_after_its_end.fetch_add(1, Ordering::SeqCst);
            }
            o.references.fetch_add(1, Ordering::SeqCst) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let o = &*(this as *const Object);
            // (the first Release is `res_release` reading the count; with the second, another
            // thread's Release of the other reference lands)
            let gone = if o.releases.fetch_add(1, Ordering::SeqCst) == 1 { 2 } else { 1 };
            let n = o.references.fetch_sub(gone, Ordering::SeqCst) - gone;
            if n == 0 {
                o.ended.store(true, Ordering::SeqCst);
                // the object is gone and this Release has not returned: a refresh asks for it now
                let t = std::time::Instant::now();
                while !o.asked.load(Ordering::SeqCst) && t.elapsed().as_secs() < 5 {
                    std::thread::yield_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            n
        }
        let original = [0usize, add_ref as usize, release as usize];
        let ours = [original.as_ptr() as usize, 0, add_ref as usize, super::super::real::res_release as usize];
        let object = Object {
            vtable: ours[1..].as_ptr(),
            references: AtomicU32::new(2),
            releases: AtomicU32::new(0),
            ended: AtomicBool::new(false),
            asked: AtomicBool::new(false),
            referenced_after_its_end: AtomicU32::new(0),
        };
        let res = &object as *const Object as usize;
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        let va = 0x7E57_0020_0000usize;
        {
            let mut t = TEXTURES.lock().unwrap();
            t.insert(res, Entry { device: 0, va, src: va, in_gpu_view: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, placed: false, unmapped: false, seq: 0 });
            TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        }
        std::thread::scope(|s| {
            s.spawn(|| unsafe { assert_eq!(super::super::real::res_release(res as *mut c_void), 0) });
            let t0 = std::time::Instant::now();
            while !object.ended.load(Ordering::SeqCst) && t0.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            let (taken, after_its_end) = {
                let t = TEXTURES.lock().unwrap();
                assert!(t.by_res.contains_key(&res), "the last Release has not returned: still in the table");
                object.asked.store(true, Ordering::SeqCst);
                let taken = unsafe { super::super::real::reference(res) };
                (taken, object.referenced_after_its_end.load(Ordering::SeqCst))
            };
            assert_eq!(after_its_end, 0, "a reference was taken to an object that was gone");
            assert!(taken.is_none());
        });
        assert_eq!(address(res), None, "forgotten once its last Release had returned");
        // a texture no Release of which is under way: its reference is there
        let alive = Object { vtable: ours[1..].as_ptr(), references: AtomicU32::new(1), releases: AtomicU32::new(2), ended: AtomicBool::new(false), asked: AtomicBool::new(true), referenced_after_its_end: AtomicU32::new(0) };
        let taken = unsafe { super::super::real::reference(&alive as *const Object as usize) };
        assert!(taken.is_some());
        assert_eq!(alive.references.load(Ordering::SeqCst), 2);
        drop(taken);
        assert_eq!(alive.references.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn title_bytes_snapshot_survives_source_decommit_and_zeros_reused_holes() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 << 12, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const c_void, 3 << 12, MEM_COMMIT, PAGE_READWRITE).is_null());
            for page in 0..3usize {
                std::ptr::write_bytes((base + (page << 12)) as *mut u8, 0xA0 + page as u8, 1 << 12);
            }
            with_title_bytes(base, 3 << 12, |whole| {
                assert_eq!(VirtualFree((base + (1 << 12)) as *mut c_void, 1 << 12, MEM_DECOMMIT), 1);
                assert!(whole[1 << 12..2 << 12].iter().all(|&b| b == 0xA1));
            });
            with_title_bytes(base, 3 << 12, |holed| {
                assert_eq!(holed, &read_committed(base, 3 << 12)[..]);
                assert!(holed[..1 << 12].iter().all(|&b| b == 0xA0) && holed[1 << 12..2 << 12].iter().all(|&b| b == 0) && holed[2 << 12..].iter().all(|&b| b == 0xA2));
            });
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Only the wanted parts of the watched regions are scanned, merged, one region each.
    #[test]
    fn watch_ranges_clip_and_merge() {
        let regions = [(0x10000, 0x10000), (0x40000, 0x10000), (0x1000_0000, 0x1000_0000)];
        let wanted = vec![(0x12000, 0x13000), (0x11000, 0x12800), (0x1F000, 0x41000), (0x60000, 0x61000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x11000, 0xF000), (0x40000, 0x10000)]);
        // ranges a gap of under WATCH_GAP apart: one scan; further apart: two
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1008_0000, 0x1008_1000), (0x1100_0000, 0x1100_1000), (0x1800_0000, 0x1800_1000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x1000_0000, 0x100_1000), (0x1800_0000, 0x1000)]);
        // a merged range over two regions: a piece in each, whatever order the regions come in
        let regions = [(0x40000, 0x10000), (0x10000, 0x10000)];
        assert_eq!(watch_ranges(&regions, vec![(0x18000, 0x48000)]), vec![(0x18000, 0x8000), (0x40000, 0x8000)]);
        // a region between two merged ranges, and one past them all
        let regions = [(0x2000_0000, 0x10_0000), (0x9000_0000, 0x1000)];
        assert_eq!(watch_ranges(&regions, vec![(0x1000_0000, 0x1000_1000), (0x2008_0000, 0x2009_0000)]), vec![(0x2008_0000, 0x1_0000)]);
    }

    /// The textures of a batch detiled on several threads come back as on one: complete and in
    /// order (2D with mips and slices, 3D, BC7, a tiny one; more textures than threads).
    #[test]
    fn a_batch_detiled_on_several_threads_matches_one_thread() {
        let descs = [
            console_desc(3, 64, 64, 2, 3, 28, 0x105),
            console_desc(3, 200, 72, 1, 5, 28, 0x105),
            console_desc(4, 40, 24, 6, 3, 28, 0x105),
            console_desc(3, 256, 256, 1, 9, 98, 0x10A),
            console_desc(3, 4, 4, 1, 1, 28, 0x105),
            console_desc(3, 128, 32, 3, 2, 28, 0x105),
            console_desc(3, 64, 64, 1, 1, 28, 0x105),
        ];
        let textures: Vec<xg::Texture> = descs.iter().map(|d| xg::Texture::new(d).unwrap()).collect();
        // each texture's console memory: distinct bytes per texture
        let mems: Vec<Vec<u8>> = textures.iter().enumerate().map(|(t, x)| (0..x.size() as usize).map(|i| (i * 7 + t * 61) as u8).collect()).collect();
        let picked: Vec<(usize, &xg::Texture)> = mems.iter().zip(&textures).map(|(m, x)| (m.as_ptr() as usize, x)).collect();
        unsafe {
            let one = subresources_on(&picked, 1);
            assert!(one.iter().all(|s| !s.is_empty()), "a texture gave no subresources");
            for threads in [2, 3, 8] {
                assert!(subresources_on(&picked, threads) == one, "{threads} threads differ from one");
            }
        }
    }

    fn console_desc(dim: u32, w: u64, h: u32, array_or_depth: u16, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&dim.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&w.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&array_or_depth.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        d
    }

    /// Overlapping textures shadow earlier ones so dead placements never re-upload over shared
    /// `vram::Pages` tiles; different-range overlaps request unmapping while exact same-range
    /// overlaps stay mapped; releasing a newer placement reactivates uncovered older ones.
    #[test]
    fn overlapping_placements_retire_superseded_entries() {
        let tex_16k = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 28, 0x105)).unwrap());
        let tex_64k = Arc::new(xg::Texture::new(&console_desc(3, 128, 128, 1, 1, 28, 0x105)).unwrap());
        assert_eq!((tex_16k.size(), tex_64k.size()), (0x4000, 0x1_0000));
        let entry = |va: usize, tex: &Arc<xg::Texture>, pending: bool, placed: bool| Entry {
            device: 0,
            va,
            src: va,
            in_gpu_view: false,
            tex: Arc::clone(tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending,
            gpu_written: false,
            placed,
            unmapped: false,
            seq: 0,
        };
        let mut tbl = Table::new();
        // Two adjacent placed 16 KiB textures + one placed 64 KiB texture after them.
        assert_eq!(tbl.insert(1, entry(0x10_0000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(2, entry(0x10_4000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(3, entry(0x11_0000, &tex_64k, false, true)), (0, vec![]));
        assert_eq!(tbl.by_res.len(), 3);
        // Exact same-range sibling over 2 shadows 2 without unmapping (tiles are shared).
        assert_eq!(tbl.insert(20, entry(0x10_4000, &tex_16k, false, true)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&2) && tbl.shadowed.contains_key(&2));
        // A 64 KiB placed texture at 0x10_0000 supersedes 1, 20, and shadowed sibling 2 (all different range).
        let (delta, mut unmap) = tbl.insert(4, entry(0x10_0000, &tex_64k, true, true));
        unmap.sort_unstable();
        assert_eq!((delta, unmap), (0, vec![1, 2, 20]));
        assert!(tbl.by_res.contains_key(&3) && tbl.by_res.contains_key(&4));
        // Releasing 4 uncovers newest non-overlapping shadowed entries (1 and 20) and requests remap.
        let (removed, delta, mut remap) = tbl.remove(4);
        remap.sort_unstable();
        assert_eq!((removed, delta, remap), (true, 1, vec![1, 20]));
        assert!(tbl.by_res.contains_key(&1) && tbl.by_res.contains_key(&20) && tbl.shadowed.contains_key(&2));
        // Releasing 20 uncovers 2; releasing middle of a 3-chain still uncovers oldest when top goes.
        let (removed, _, remap) = tbl.remove(20);
        assert_eq!((removed, remap), (true, vec![2]));
        assert!(tbl.by_res.contains_key(&2));
        assert_eq!(tbl.insert(5, entry(0x10_8000, &tex_64k, true, true)).0, 1);
        assert_eq!(tbl.overlapping(0x10_0000, 0x1_0000), vec![(1, 0x10_0000), (2, 0x10_4000), (5, 0x10_8000)]);
        // Two committed resources with the exact same range coexist; a shifted one shadows them.
        assert_eq!(tbl.insert(10, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert_eq!(tbl.insert(11, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert!(tbl.by_res.contains_key(&10) && tbl.by_res.contains_key(&11));
        assert_eq!(tbl.insert(12, entry(0x20_0000, &tex_64k, true, false)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&10) && !tbl.by_res.contains_key(&11) && tbl.by_res.contains_key(&12));
    }

    /// (one memory) A sampled texture on the tiles of a GPU-written one starts as the GPU's
    /// only while the title's memory under it is zero, and such an entry is never pending: no
    /// upload goes over what the GPU wrote, at its registration or at a later submit.
    #[test]
    fn a_sampled_texture_on_gpu_tiles_is_not_uploaded_while_its_memory_is_zero() {
        let zeros = vec![0u8; 3 * 4096];
        let mut texels = zeros.clone();
        texels[2 * 4096 + 17] = 1;
        assert!(starts_as_gpus(true, &zeros) && starts_as_gpus(true, &[]));
        assert!(!starts_as_gpus(true, &texels), "the CPU filled it: uploaded");
        assert!(!starts_as_gpus(false, &zeros), "on tiles of its own: uploaded (zeros read as zeros)");
        // the entry `track` then makes: GPU-written and not pending – the upload scan passes
        // over it (`pending` or a written page of one that is not GPU-written)
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 128, 128, 1, 1, 28, 0x105)).unwrap());
        let entry = |gpus: bool| Entry {
            device: 0, va: 0x30_4000, src: 0x30_4000, in_gpu_view: false, tex: Arc::clone(&tex), states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: !gpus, gpu_written: gpus, placed: true, unmapped: false, seq: 0,
        };
        let mut tbl = Table::new();
        assert_eq!(tbl.insert(1, entry(true)), (0, vec![]));
        assert!(tbl.by_res[&1].gpu_written && !tbl.by_res[&1].pending);
        assert_eq!(tbl.insert(2, entry(false)).0, 1, "one that starts as the CPU's is pending");
        // a target placed over both retires them: neither is uploaded from then on
        let (delta, retired, mut unmap) = tbl.retire_range(0x30_4000, 0x31_4000);
        unmap.sort_unstable();
        assert_eq!((delta, retired, unmap), (-1, 2, vec![1, 2]));
        assert!(tbl.by_res.is_empty() && tbl.shadowed.is_empty());
    }

    /// Real GPU (#248): placed title textures share `vram::Pages` so streaming unreleased
    /// overlapping placements stays bounded in video memory, superseded textures never corrupt
    /// newer placements on shared tiles, 4 KiB-aligned neighbours never false-alias, and mip
    /// chains (including packed mips), cubemap arrays (`extra_pages`) and 3D volumes upload intact.
    #[test]
    fn placed_title_textures_share_vram_and_preserve_pixels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no adapter");
                return;
            };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: software adapter");
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no D3D12 device");
                return;
            };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: tiled tier < 2");
                return;
            }
            // An adapter that reports tier 2 but refuses a reserved cube with packed mips
            // (GitHub's runner) does not run the shared-page path this test is about.
            {
                let mut cube: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&console_desc(3, 256, 256, 6, 9, 28, 0x105)).as_ptr() as *const _);
                cube.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                let mut r: Option<ID3D12Resource> = None;
                if dev.CreateReservedResource(&cube, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_err() || r.is_none() {
                    println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: this adapter makes no reserved cube with mips");
                    return;
                }
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let Some(uma) = crate::gpu::uma::get() else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no UMA");
                return;
            };
            let pool_size = 16usize << 20;
            let Some(base) = uma.alloc(pool_size, 0x1_0000).map(|p| p as usize) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: UMA alloc failed");
                return;
            };
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local_vram = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_tex = |va: usize, desc: &[u8; 0x38]| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), va as u64, desc.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "CreatePlacedResourceX failed: {hr:#x}");
                assert!(super::super::vram::is_placed_texture(p as usize), "expected shared vram::Pages texture at {va:#x}");
                ID3D12Resource::from_raw(p)
            };
            let write_tiled = |va: usize, desc: &[u8; 0x38], seed: u8| {
                let tex = xg::Texture::new(desc).unwrap();
                let s = tex.texels().unwrap();
                let bytes = std::slice::from_raw_parts_mut(va as *mut u8, tex.size() as usize);
                bytes.fill(0);
                if s.depth > 0 {
                    for (m, l) in s.chain.levels.iter().enumerate() {
                        let pat: Vec<u8> = (0..l.width * l.height * l.depth * s.bpe as usize).map(|k| seed.wrapping_add(m as u8).wrapping_add((k * 13) as u8)).collect();
                        s.chain.tile_volume(bytes, m, &pat);
                    }
                } else {
                    for slice in 0..s.array as usize {
                        let dst = &mut bytes[slice * s.chain.slice_bytes..(slice + 1) * s.chain.slice_bytes];
                        for (m, l) in s.chain.levels.iter().enumerate() {
                            let pat: Vec<u8> = (0..l.width * l.height * s.bpe as usize).map(|k| seed.wrapping_add((slice * 31 + m * 7) as u8).wrapping_add((k * 13) as u8)).collect();
                            s.chain.tile_level(dst, m, slice, &pat);
                        }
                    }
                }
            };
            let read_sub = |r: &ID3D12Resource, sub: u32| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
                .unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                own_work(|| {
                    super::super::vram::before_execute(&queue);
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let slice_pitch = fp.Footprint.RowPitch as usize * rows as usize;
                let mut out = Vec::with_capacity(fp.Footprint.Depth as usize * rows as usize * row_bytes as usize);
                for z in 0..fp.Footprint.Depth as usize {
                    for y in 0..rows as usize {
                        out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(z * slice_pitch + y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                    }
                }
                rb.Unmap(0, None);
                out
            };

            // 1. Streaming 512 unreleased 1 MiB textures (512 MiB placed) across 16 MiB stays bounded
            // with zero additional page growth after the 16 MiB working set is covered.
            let dev_key = dev.as_raw() as usize;
            let dev_pages = || super::super::vram::page_bytes_for(dev_key).0;
            let stream_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 64KB_S, 1 MiB
            // (counted from the pages in use: those earlier tests left idle on this device go
            // back while this one runs, and the count fell below its start – a subtraction
            // overflow in a run one test at a time)
            super::super::vram::sweep_idle_now(dev_key);
            let (pages0, vram0, kept0) = (dev_pages(), local_vram(), counts().0);
            let mut leaked_refs = Vec::with_capacity(512);
            for k in 0..16usize {
                leaked_refs.push(place_tex(base + k * (1 << 20), &stream_desc));
            }
            // (a sampled texture has pages where it has texels: none before its first upload)
            assert!(dev_pages() <= pages0, "a texture without texels takes no pages");
            for k in 0..16usize {
                write_tiled(base + k * (1 << 20), &stream_desc, 1 + k as u8);
            }
            refresh(queue.as_raw());
            let (pages16, vram16) = (dev_pages(), local_vram());
            for k in 16..512usize {
                leaked_refs.push(place_tex(base + (k % 16) * (1 << 20), &stream_desc));
            }
            refresh(queue.as_raw());
            let (pages_delta, vram_delta, vram_after_fill, kept_delta) = (
                dev_pages() - pages0,
                local_vram().saturating_sub(vram0) as usize,
                local_vram().saturating_sub(vram16) as usize,
                counts().0 - kept0,
            );
            println!("512 unreleased 1 MiB textures over 16 MiB: +{pages_delta} B pages (after fill: +{} B), +{vram_delta} B local VRAM (after fill: +{vram_after_fill} B), +{kept_delta} active kept", dev_pages() - pages16);
            assert_eq!(pages_delta, 16 << 20, "pages must equal covered 16 MiB, not 512 MiB");
            assert_eq!(dev_pages(), pages16, "no pages added after working set is covered");
            // (the process's local video memory is not asserted: other GPU tests in the process
            // allocate at the same time - on an RX 9070 it read +672 MB under --test-threads=4 and
            // stayed within the bound alone; the page count above is what this change controls)
            let _ = vram_after_fill;
            assert_eq!(kept_delta, 16, "superseded placements must be retired from active tracking");

            // 2. Overlap integrity (rows 4, 5, 6) + unmap/remap on different-range overlap:
            // (a) Exact same range: old object sees new uploads (row 4); releasing new object
            // reactivates old object for subsequent CPU rewrites (row 6).
            let same_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            let same_a = place_tex(base + 0x1000, &same_desc); // 4 KiB offset too!
            write_tiled(base + 0x1000, &same_desc, 0x1A);
            refresh(queue.as_raw());
            let same_b = place_tex(base + 0x1000, &same_desc);
            write_tiled(base + 0x1000, &same_desc, 0x2B);
            refresh(queue.as_raw());
            let want_2b = subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap()).swap_remove(0).data;
            assert_eq!(read_sub(&same_b, 0), want_2b, "row 3: new placement readback");
            assert_eq!(read_sub(&same_a, 0), want_2b, "row 4: older exact-range alias shares tiles");
            drop(same_b);
            write_tiled(base + 0x1000, &same_desc, 0x3C);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&same_a, 0), subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap())[0].data, "row 6: releasing newer placement reactivates older one");
            drop(same_a);

            // (b) Partial / different-range overlap (row 5 + unmap dead tail + remap on release):
            let old_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 1 MiB
            let new_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            drop(leaked_refs);
            let old_res = place_tex(base, &old_desc);
            write_tiled(base, &old_desc, 0xAA);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data);
            // Place B at base + 256 KiB (overlaps old_res tail); writing into old_res's first
            // 256 KiB must NOT re-upload old_res over B (row 5).
            let new_res = place_tex(base + 0x4_0000, &new_desc);
            write_tiled(base + 0x4_0000, &new_desc, 0x55);
            refresh(queue.as_raw());
            let want_b = subresources(base + 0x4_0000, &xg::Texture::new(&new_desc).unwrap()).swap_remove(0).data;
            *(base as *mut u8) ^= 0xFF; // dirty page outside B, inside old_res
            refresh(queue.as_raw());
            assert_eq!(read_sub(&new_res, 0), want_b, "row 5: stale shadowed texture corrupted B");
            // Releasing B remaps previously-unmapped old_res cleanly and restores its pixels.
            drop(new_res);
            write_tiled(base, &old_desc, 0xD4);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data, "reactivated unmapped texture remapped cleanly");
            drop(old_res);

            // 3. 4 KiB-aligned adjacent textures inside one 64 KiB page + straddling page boundary,
            // plus reclaiming dead extra_pages when overwritten by a larger placement without Release.
            let d16 = console_desc(3, 64, 64, 1, 1, 28, 0x105); // 4KB_S, 16 KiB
            let d64 = console_desc(3, 128, 128, 1, 1, 28, 0x105); // 4KB_S, 64 KiB
            let adj_specs = [(0x0000usize, d16, 0x11u8), (0x4000, d16, 0x22), (0x8000, d16, 0x33), (0xC000, d64, 0x44), (0x1_C000, d16, 0x55)];
            let adj: Vec<_> = adj_specs.iter().map(|&(off, d, _)| place_tex(base + off, &d)).collect();
            let arena = |r: &ID3D12Resource| super::super::vram::arena_of(r.as_raw() as usize);
            assert!(adj.iter().all(|r| arena(r).is_none()), "the multi-page title allocation must stay tiled");
            for &(off, d, seed) in &adj_specs {
                write_tiled(base + off, &d, seed);
            }
            refresh(queue.as_raw());
            for (r, &(off, d, _)) in adj.iter().zip(&adj_specs) {
                assert_eq!(read_sub(r, 0), subresources(base + off, &xg::Texture::new(&d).unwrap())[0].data, "4 KiB neighbour at +{off:#x} crosstalk");
            }
            let pages_before_reuse = dev_pages();
            let reused = place_tex(base + 0x4000, &d16);
            assert_eq!(dev_pages(), pages_before_reuse, "reusing unaligned va must share extra_pages tile");
            // Overwrite the first 64 KiB (covering +0x0000, +0x4000 (both adj[1] & reused), +0x8000, +0xC000)
            // WITHOUT releasing adj/reused: dead extra_pages (+0x4000, +0x8000) must drop immediately!
            let cover = place_tex(base, &d64);
            let pages_after_cover = dev_pages();
            assert!(pages_after_cover < pages_before_reuse, "superseded unaligned extra_pages must be reclaimed immediately ({pages_before_reuse} -> {pages_after_cover})");
            drop((cover, reused, adj));

            let pc_desc = |d| std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(d).as_ptr().cast::<D3D12_RESOURCE_DESC>());
            let cover_desc = pc_desc(&d64);
            let tiny_desc = D3D12_RESOURCE_DESC { Alignment: 0x1000, ..pc_desc(&d16) };
            let tiny_info = dev.GetResourceAllocationInfo(0, &[tiny_desc]);
            let cover_info = dev.GetResourceAllocationInfo(0, &[cover_desc]);
            if tiny_info.Alignment == 0x1000 && (1..=0x4000).contains(&tiny_info.SizeInBytes)
                && cover_info.Alignment == 0x1_0000 && cover_info.SizeInBytes == 0x1_0000
            {
                let group = uma.reserve(0x1_0000, 0x1_0000).expect("bounded title texture group") as usize;
                let offsets = [0, 0x4000, 0x8000, 0xC000];
                let packed: Vec<_> = offsets.iter().map(|&off| place_tex(group + off, &d16)).collect();
                let id = arena(&packed[0]).expect("bounded small textures must use the default compact route");
                assert!(packed.iter().all(|r| arena(r) == Some(id)), "the four neighboring textures must share one arena");
                for (i, &off) in offsets.iter().enumerate() {
                    write_tiled(group + off, &d16, 0x50 + i as u8);
                }
                refresh(queue.as_raw());
                for (r, &off) in packed.iter().zip(&offsets) {
                    assert_eq!(read_sub(r, 0), subresources(group + off, &xg::Texture::new(&d16).unwrap())[0].data, "compact neighbor at +{off:#x}");
                }
                let reused = place_tex(group + 0x4000, &d16);
                assert_eq!(arena(&reused), Some(id));
                write_tiled(group + 0x4000, &d16, 0x6B);
                refresh(queue.as_raw());
                let changed = subresources(group + 0x4000, &xg::Texture::new(&d16).unwrap()).swap_remove(0).data;
                assert_eq!(read_sub(&reused, 0), changed);
                assert_eq!(read_sub(&packed[1], 0), changed, "the original alias sees writes through the new placement");
                let cover = place_tex(group, &d64);
                assert_eq!(arena(&cover), Some(id), "a compatible larger placement must reuse the live compact arena");
                write_tiled(group, &d64, 0x92);
                refresh(queue.as_raw());
                assert_eq!(read_sub(&cover, 0), subresources(group, &xg::Texture::new(&d64).unwrap())[0].data, "compact cover pixels");
                drop((cover, reused, packed));
                assert!(uma.free(group));
            }

            // 4. Full mip chain (standard + packed mips, RGBA8 & BC7), tiny all-packed mips,
            // 6-slice cubemap (exercises extra_pages reuse), and 3D volume texture (Tier 3).
            let shapes = [
                ("2D 512x512 RGBA8 10 mips", console_desc(3, 512, 512, 1, 10, 28, 0x109)),
                ("2D 1024x1024 BC7 11 mips", console_desc(3, 1024, 1024, 1, 11, 98, 0x109)),
                ("2D 64x64 RGBA8 all-packed 7 mips", console_desc(3, 64, 64, 1, 7, 28, 0x105)),
                ("2D 256x256x6 cubemap 9 mips", console_desc(3, 256, 256, 6, 9, 28, 0x105)),
                ("3D 64x64x64 RGBA8 1 mip", console_desc(4, 64, 64, 64, 1, 28, 0x105)),
                ("3D 64x64x64 RGBA8 4 packed mips", console_desc(4, 64, 64, 64, 4, 28, 0x105)),
                ("3D 64x64x64 R11G11B10_FLOAT 4 packed mips", console_desc(4, 64, 64, 64, 4, 26, 0x105)),
                ("3D 64x64x64 R16_FLOAT 4 packed mips", console_desc(4, 64, 64, 64, 4, 54, 0x105)),
            ];
            for (label, d) in shapes {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), base as u64, d.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "{label}: hr {hr:#x}");
                let res = ID3D12Resource::from_raw(p);
                let is_3d = u32::from_le_bytes(d[0..4].try_into().unwrap()) == 4;
                let array = u16::from_le_bytes(d[0x1C..0x1E].try_into().unwrap());
                let placed = super::super::vram::is_placed_texture(res.as_raw() as usize);
                if array > 1 && !is_3d {
                    // arrays and cubes with packed mips: a driver may refuse them reserved (GitHub's
                    // runner did with this cube); the texture is then committed – what the pixels
                    // below check either way
                    if !placed {
                        println!("{label}: not reserved on this adapter (committed instead)");
                    }
                } else if is_3d && u16::from_le_bytes(d[0x1E..0x20].try_into().unwrap()) > 1 {
                    assert!(!placed, "{label}: packed volume mips must use committed detile/upload");
                } else if !is_3d || opts.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_3.0 {
                    assert!(placed, "{label} should use shared vram::Pages");
                }
                write_tiled(base, &d, 0x71);
                refresh(queue.as_raw());
                let want = subresources(base, &xg::Texture::new(&d).unwrap());
                for (sub, exp) in want.iter().enumerate() {
                    assert_eq!(read_sub(&res, sub as u32), exp.data, "{label}: subresource {sub} mismatch");
                }
                // Placing a second cubemap at the same va must reuse every tile including extra_pages.
                if label.contains("cubemap") {
                    let p0 = dev_pages();
                    let res2 = place_tex(base, &d);
                    assert_eq!(dev_pages(), p0, "cubemap extra_pages must be reused at same va");
                    drop(res2);
                }
                drop(res);
            }

            // 5. A sampled texture has video memory only where it has texels. A streamed texture
            // whose largest mip the title has not loaded takes none for it and reads zeros there;
            // the mip gets its memory with its texels and gives it back when they are zero again.
            // BC1, whose zero texels are opaque black, keeps memory under the zeros the title has
            // memory for. And the first GPU copy into a texture gives it all its memory.
            let backing = |r: &ID3D12Resource| super::super::vram::tile_backing(r.as_raw() as usize).map(|(local, non_local, none)| (local + non_local, none)).unwrap();
            let zero_top = |va: usize, desc: &[u8; 0x38]| {
                let tex = xg::Texture::new(desc).unwrap();
                let s = tex.texels().unwrap();
                let l = s.chain.levels[0];
                s.chain.tile_level(std::slice::from_raw_parts_mut(va as *mut u8, s.chain.slice_bytes), 0, 0, &vec![0u8; l.width * l.height * s.bpe as usize]);
            };
            let check = |r: &ID3D12Resource, desc: &[u8; 0x38], what: &str| {
                for (sub, exp) in subresources(base, &xg::Texture::new(desc).unwrap()).iter().enumerate() {
                    assert_eq!(read_sub(r, sub as u32), exp.data, "{what}: subresource {sub}");
                }
            };
            // 1024x1024 BC7: 16 tiles of mip 0 (tier 2: a mip of whole tiles is not packed)
            let bc7 = console_desc(3, 1024, 1024, 1, 11, 98, 0x109);
            let res = place_tex(base, &bc7);
            write_tiled(base, &bc7, 0x21);
            zero_top(base, &bc7);
            refresh(queue.as_raw());
            let (held, none) = backing(&res);
            println!("1024x1024 BC7 without its largest mip: {held} tiles with memory, {none} without");
            assert_eq!(none, 16, "the mip without texels has no memory");
            assert!(held >= 5, "the mips with texels have theirs");
            assert!(read_sub(&res, 0).iter().all(|&b| b == 0), "a mip without memory reads zeros");
            check(&res, &bc7, "largest mip not loaded");
            write_tiled(base, &bc7, 0x22);
            refresh(queue.as_raw());
            assert_eq!(backing(&res), (held + 16, 0), "the mip has memory when it has texels");
            check(&res, &bc7, "largest mip loaded");
            zero_top(base, &bc7);
            refresh(queue.as_raw());
            assert_eq!(backing(&res), (held, 16), "and gives it back when they are zero again");
            check(&res, &bc7, "largest mip let go");
            // the title's GPU copies into it: every tile from then on, whatever the CPU's bytes
            super::super::vram::note_gpu_write(res.as_raw() as usize);
            assert_eq!(backing(&res).1, 0, "a GPU-written texture has all its memory");
            write_tiled(base, &bc7, 0x23);
            zero_top(base, &bc7);
            refresh(queue.as_raw());
            assert_eq!(backing(&res).1, 0, "a GPU-written texture keeps all its memory");
            check(&res, &bc7, "GPU-written");
            drop(res);
            // 1024x1024 BC1: zeros the title has memory under are texels (opaque black)
            let bc1 = console_desc(3, 1024, 1024, 1, 11, 71, 0x109);
            let res = place_tex(base, &bc1);
            write_tiled(base, &bc1, 0x31);
            zero_top(base, &bc1);
            refresh(queue.as_raw());
            assert_eq!(backing(&res).1, 0, "BC1 zeros in the title's memory keep their video memory");
            check(&res, &bc1, "BC1 with a zero mip");
            drop(res);
            uma.free(base);
        }
    }

    /// Upload batches keep the uploads' order and hold at most 32 MiB and 256 textures – a
    /// larger texture alone; each batch is one upload buffer of its size and one wait.
    #[test]
    fn upload_batches_keep_order_within_the_caps() {
        const MIB: usize = 1 << 20;
        let split = |sizes: &[usize]| {
            let mut batches: Vec<Vec<usize>> = Vec::new();
            let (mut current, mut bytes) = (Vec::new(), 0usize);
            for (i, &size) in sizes.iter().enumerate() {
                if batch_full(current.len(), bytes, size) {
                    batches.push(std::mem::take(&mut current));
                    bytes = 0;
                }
                current.push(i);
                bytes += size;
            }
            batches.push(current);
            batches
        };
        // 10 MiB, 20 MiB | 5 MiB | 40 MiB alone | 1 MiB
        assert_eq!(split(&[10 * MIB, 20 * MIB, 5 * MIB, 40 * MIB, MIB]), [vec![0, 1], vec![2], vec![3], vec![4]]);
        assert_eq!(split(&[32 * MIB, 1]), [vec![0], vec![1]]);
        // small textures: 256 a batch
        let small = split(&[4096; 600]);
        assert_eq!(small.iter().map(Vec::len).collect::<Vec<_>>(), [256, 256, 88]);
        assert_eq!(small.concat(), (0..600).collect::<Vec<_>>(), "in order, each once");
        // a world load's burst of 1.4 MiB textures (256 MiB of them): 8 waits where one batch
        // of 256 MiB – and its upload buffer – was 1
        let burst = split(&[1_398_101; 192]);
        assert_eq!(burst.len(), 8);
        assert!(burst.iter().all(|b| b.len() * 1_398_101 <= BATCH_BYTES));
    }

    /// Which mips need video memory: those with a byte that is not zero, and – for a format
    /// whose zero texels are not what a tile without memory reads as – those the title has
    /// memory under at all.
    #[test]
    fn mips_have_texels_where_bytes_are_set_or_zero_is_a_texel() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            let tex = xg::Texture::new(&console_desc(3, 1024, 1024, 1, 11, 71, 0x109)).unwrap();
            let s = tex.texels().unwrap();
            // a streamed texture: the title has memory under every mip but the largest
            let top = s.chain.level_span(0);
            assert!(top.start > 0 && top.start % 0x1000 == 0 && top.end == s.chain.slice_bytes, "{top:?}");
            let base = VirtualAlloc(std::ptr::null(), (tex.size() as usize).next_multiple_of(0x1_0000), MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const c_void, top.start, MEM_COMMIT, PAGE_READWRITE).is_null());
            let l = s.chain.levels[2];
            s.chain.tile_level(std::slice::from_raw_parts_mut(base as *mut u8, top.start), 2, 0, &vec![0x5Au8; l.width * l.height * s.bpe as usize]);
            let subs = subresources(base, &tex);
            assert_eq!(subs.len(), 11);
            let set: Vec<bool> = (0..11).map(|k| k == 2).collect();
            assert_eq!(with_texels(base, &tex, true, &subs), set, "zero reads as no memory: the mips with bytes set");
            let memory: Vec<bool> = (0..11).map(|k| k != 0).collect();
            assert_eq!(with_texels(base, &tex, false, &subs), memory, "zero is a texel: every mip the title has memory under");
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `read_committed` copies committed pages and leaves uncommitted gaps zeroed, including
    /// unaligned start/end offsets.
    #[test]
    fn read_committed_copies_committed_pages_and_zeroes_gaps() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            assert_ne!(VirtualAlloc((base + 2 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            std::ptr::write_bytes(base as *mut u8, 0x3A, PAGE);
            std::ptr::write_bytes((base + 2 * PAGE) as *mut u8, 0x7C, PAGE);

            let start = base + 0x0B00;
            let len = 2 * PAGE + 0x0500;
            let got = read_committed(start, len);
            assert_eq!(got.len(), len);
            assert!(got[..PAGE - 0x0B00].iter().all(|&b| b == 0x3A));
            assert!(got[PAGE - 0x0B00..2 * PAGE - 0x0B00].iter().all(|&b| b == 0));
            assert!(got[2 * PAGE - 0x0B00..].iter().all(|&b| b == 0x7C));

            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `read_committed` must not fault with 0xC0000005 when another thread concurrently
    /// decommits/recommits pages between `VirtualQuery` and the copy, and surrounding committed
    /// pages must remain intact.
    #[test]
    fn read_committed_tolerates_concurrent_decommit() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        const PAGES: usize = 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGES * PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            // Keep first and last 64 KiB pages permanently committed with known sentinels; race the middle.
            std::ptr::write_bytes(base as *mut u8, 0x5A, PAGE);
            std::ptr::write_bytes((base + (PAGES - 1) * PAGE) as *mut u8, 0xA5, PAGE);

            let stop = Arc::new(AtomicBool::new(false));
            let stop_w = Arc::clone(&stop);
            let mutator = std::thread::spawn(move || {
                let mut i = 0usize;
                while !stop_w.load(Ordering::Relaxed) {
                    let slot = 1 + (i % (PAGES - 2));
                    let p = (base + slot * PAGE) as *const c_void;
                    let _ = VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT);
                    let q = VirtualAlloc(p, PAGE, MEM_COMMIT, PAGE_READWRITE);
                    if !q.is_null() {
                        std::ptr::write_bytes(q as *mut u8, 0x3C, 4096);
                    }
                    i = i.wrapping_add(1);
                }
            });

            // Unaligned start/end across the racing region (mirrors the streaming pool layout).
            let off = 0x0B00;
            let len = (PAGES - 1) * PAGE + 0x0500;
            for _ in 0..400 {
                let out = read_committed(base + off, len);
                assert_eq!(out.len(), len);
                assert!(out[..PAGE - off].iter().all(|&b| b == 0x5A));
                assert!(out[len - (off + 0x0500)..].iter().all(|&b| b == 0xA5));
            }

            stop.store(true, Ordering::Relaxed);
            mutator.join().unwrap();
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
