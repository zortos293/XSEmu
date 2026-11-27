//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels – one of another layout there is told
//! apart and has memory of its own, `texture_alias`). When the title requests the address
//! of a supported color texture, `title_textures` also mirrors writes from its CPU storage.
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).
//!
//! A sampled texture on pages has memory only where it has texels (`note_content`): a title
//! places a streamed texture's whole mip chain and maps memory under the levels it has loaded –
//! a retail game placed 5.2 GB of textures over 2.9 GB of mapped pages, and a page of video
//! memory under every tile of them took 2.3 GB the console never spends. A mip whose bytes are
//! all zero in the title's memory (nothing mapped there, or not written yet) has no tiles
//! mapped, which reads as the same zeros. What the GPU writes keeps all its memory: render
//! targets, depth buffers and UAV textures always, a sampled texture from the title's first
//! copy into it (`note_gpu_write`). `XCL_LAZY_TILES=0`: a page under every tile, as before.
//!
//! Textures under a page on the console (a retail game: 9 000 of 16x4 texels) took a 64 KiB
//! tile each, 0.9 GB for a few MB of texels: sampled ones the PC places at 4 KiB go to slots
//! of shared 1 MiB heaps (`Slabs`; `XCL_SMALL_SLABS=0`: tiles, as before).

use std::collections::{BTreeSet, HashSet};
use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

const SMALL: usize = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as usize;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Placement {
    va: usize,
    host_end: usize,
    title_end: usize,
    align: usize,
    /// `texture_alias::Shape::key` of the texture (0: not told – it shares by address alone)
    shape: u64,
}

impl Placement {
    fn new(va: usize, bytes: usize, align: usize, console_bytes: Option<u64>) -> Option<Self> {
        Some(Self {
            va,
            host_end: va.checked_add(bytes)?,
            title_end: va.checked_add(console_bytes.map_or(bytes, |b| b as usize))?,
            align,
            shape: 0,
        })
    }

    /// The placement of a texture of this shape (`texture_alias::Shape::key`).
    fn of_shape(self, shape: u64) -> Self {
        Self { shape, ..self }
    }

    fn small(self) -> bool {
        self.align == SMALL
    }

    fn host_overlaps(self, other: Self) -> bool {
        self.va < other.host_end && other.va < self.host_end
    }

    fn title_overlaps(self, va: usize, end: usize) -> bool {
        self.va < end && va < self.title_end
    }
}

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    device: usize,
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    compact: bool,
    heap: H,
    placements: Vec<Placement>,
    title_end: usize,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
    /// (`texture_alias`) made for a placement whose shape differs from the one at its address,
    /// which had joined an arena there by address: video memory the split takes (`split_bytes`)
    split: bool,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

struct NewArena {
    base: usize,
    size: usize,
    align: usize,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Arenas no resource used for `IDLE` go (their heap with the last reference).
    fn sweep(&mut self, now: std::time::Instant) -> Vec<Arena<H>> {
        self.list.extract_if(.., |a| a.placements.is_empty()
            && a.idle_since.is_some_and(|t| now.saturating_duration_since(t) >= IDLE)).collect()
    }

    fn overlaps_small(&self, device: usize, va: usize, end: usize) -> bool {
        self.overlaps(device, va, end, true)
    }

    fn overlaps(&self, device: usize, va: usize, end: usize, compact: bool) -> bool {
        self.list.iter().filter(|a| a.device == device && a.compact == compact && va < a.title_end && a.base < end).any(|a| {
            a.placements.iter().any(|&p| p.title_overlaps(va, end))
        })
    }

    fn selection(&self, device: usize, p: Placement, require_alias: bool) -> Result<Option<usize>, ()> {
        let fits = |a: &Arena<H>| {
            a.device == device && a.align >= p.align && a.base <= p.va
                && p.host_end <= a.base + a.size && (p.va - a.base) % p.align == 0
                && (!a.compact || p.title_end <= a.base + a.size)
                && !a.placements.iter().any(|&q| {
                    (p.small() || a.compact) && p.host_overlaps(q) && !p.title_overlaps(q.va, q.title_end)
                })
        };
        let mut alias = None;
        if p.small() || self.overlaps_small(device, p.va, p.title_end) {
            for (i, a) in self.list.iter().enumerate() {
                if a.device != device || p.va >= a.title_end || a.base >= p.title_end {
                    continue;
                }
                for &q in &a.placements {
                    if p.title_overlaps(q.va, q.title_end) {
                        if !p.host_overlaps(q) || alias.is_some_and(|prev| prev != i) {
                            return Err(());
                        }
                        alias = Some(i);
                    }
                }
            }
        }
        if let Some(i) = alias {
            if !fits(&self.list[i]) || (p.small() && !self.list[i].compact) {
                return Err(());
            }
            return Ok(Some(i));
        }
        if require_alias {
            return Err(());
        }
        if p.shape != 0 && !p.small() {
            // (`texture_alias`) the arena that holds its shape at its address is its memory; one
            // that holds another shape there is not – each would read the other's texels as the
            // PC lays out its own
            let at = |a: &Arena<H>, same: bool| a.placements.iter().any(|q| q.va == p.va && q.shape != 0 && (q.shape == p.shape) == same);
            let own = self.list.iter().rposition(|a| !a.compact && fits(a) && at(a, true));
            return Ok(own.or_else(|| self.list.iter().rposition(|a| !a.compact && fits(a) && !at(a, false))));
        }
        Ok(self.list.iter().rposition(|a| a.compact == p.small() && fits(a)))
    }

    /// Whether an arena holds a texture of `p`'s shape and size at `p`'s address: the PC lays
    /// `p` out there as it does that one.
    fn holds(&self, device: usize, p: Placement) -> bool {
        p.shape != 0 && self.list.iter().any(|a| {
            a.device == device && !a.compact && a.align >= p.align && a.base <= p.va
                && p.host_end <= a.base + a.size && (p.va - a.base) % p.align == 0
                && a.placements.iter().any(|q| q.va == p.va && q.shape == p.shape && q.host_end == p.host_end)
        })
    }

    /// (`texture_alias`) `p` as it is placed with `room` bytes left of the budget for memory
    /// of their own, and whether the arena made for it is one the split makes: it has no arena
    /// by its shape and had joined one by its address. Without room for that arena it is
    /// placed with no shape told – by address, as before.
    fn budgeted(&self, device: usize, p: Placement, require_alias: bool, room: usize) -> (Placement, bool) {
        if p.shape == 0 || !matches!(self.selection(device, p, require_alias), Ok(None))
            || !matches!(self.selection(device, p.of_shape(0), require_alias), Ok(Some(_))) {
            return (p, false);
        }
        if Self::new_arena(p).is_some_and(|new| new.size <= room) { (p, true) } else { (p.of_shape(0), false) }
    }

    /// bytes of the arenas the split made (`Arena::split`), those no resource uses any more
    /// too until they go
    fn split_bytes(&self) -> usize {
        self.list.iter().filter(|a| a.split).map(|a| a.size).sum()
    }

    fn new_arena(p: Placement) -> Option<NewArena> {
        let align = p.align.max(PAGE);
        let base = p.va & !(align - 1);
        let size = p.host_end.checked_next_multiple_of(align)? - base;
        Some(NewArena { base, size, align })
    }

    fn insert(&mut self, device: usize, p: Placement, new: NewArena, heap: H, split: bool) -> (H, usize, u64, bool) {
        let NewArena { base, size, align } = new;
        self.next_id += 1;
        self.list.push(Arena { device, base, size, align, compact: p.small(), heap, placements: Vec::new(), title_end: base, idle_since: None, id: self.next_id, split });
        self.join(self.list.len() - 1, p)
    }

    fn join(&mut self, i: usize, p: Placement) -> (H, usize, u64, bool) {
        let a = &mut self.list[i];
        a.placements.push(p);
        a.title_end = a.title_end.max(p.title_end);
        a.idle_since = None;
        (a.heap.clone(), p.va - a.base, a.id, a.compact)
    }

    #[cfg(test)]
    fn take(&mut self, device: usize, p: Placement, now: std::time::Instant, make: impl FnOnce(usize, usize, usize) -> Option<H>) -> Result<Option<(H, usize, u64, bool)>, ()> {
        self.take_within(device, p, now, usize::MAX, make)
    }

    /// (tests) `take` with `room` bytes left of the budget for memory of their own.
    #[cfg(test)]
    fn take_within(&mut self, device: usize, p: Placement, now: std::time::Instant, room: usize, make: impl FnOnce(usize, usize, usize) -> Option<H>) -> Result<Option<(H, usize, u64, bool)>, ()> {
        self.sweep(now);
        let (p, split) = self.budgeted(device, p, false, room);
        if let Some(i) = self.selection(device, p, false)? {
            return Ok(Some(self.join(i, p)));
        }
        let Some(new) = Self::new_arena(p) else { return Ok(None) };
        Ok(make(new.base, new.size, new.align).map(|heap| self.insert(device, p, new, heap, split)))
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, p: Placement, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            if let Some(i) = a.placements.iter().position(|&q| p == q) {
                a.placements.swap_remove(i);
            }
            if p.title_end == a.title_end {
                a.title_end = a.placements.iter().map(|p| p.title_end).max().unwrap_or(a.base);
            }
            if a.placements.is_empty() {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
#[derive(Clone, Copy, PartialEq, Eq)]
struct PlacementRange {
    device: usize,
    first_page: usize,
    last_page: usize,
}

impl PlacementRange {
    fn overlaps(self, other: Self) -> bool {
        self.device == other.device && self.first_page <= other.last_page && other.first_page <= self.last_page
    }
}

static IN_PLACEMENT: Mutex<Vec<PlacementRange>> = Mutex::new(Vec::new());
static PLACEMENT_FINISHED: Condvar = Condvar::new();

struct PendingPlacement(PlacementRange);

impl PendingPlacement {
    fn enter(device: usize, va: usize, end: usize) -> Self {
        let range = PlacementRange { device, first_page: va / PAGE, last_page: end.saturating_sub(1).max(va) / PAGE };
        let pending = IN_PLACEMENT.lock().unwrap();
        let mut pending = PLACEMENT_FINISHED.wait_while(pending, |all| all.iter().any(|r| range.overlaps(*r))).unwrap();
        pending.push(range);
        Self(range)
    }
}

impl Drop for PendingPlacement {
    fn drop(&mut self) {
        let mut pending = IN_PLACEMENT.lock().unwrap();
        let i = pending.iter().position(|r| *r == self.0).unwrap();
        pending.swap_remove(i);
        PLACEMENT_FINISHED.notify_all();
    }
}
const ARENA_PEERS: usize = 6;

#[derive(Clone, Copy)]
struct ArenaPeer {
    resource: usize,
    offset: usize,
    end: usize,
}

impl std::fmt::Debug for ArenaPeer {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{:#x}:[{:#x},{:#x})", self.resource, self.offset, self.end)
    }
}

#[derive(Clone)]
struct ArenaTrace {
    heap: usize,
    offset: usize,
    initial: TextureInitial,
    shape: Option<(u32, u32, u32)>,
    at_creation: Vec<ArenaPeer>,
    appeared_later: Vec<ArenaPeer>,
    first_peer_released: Option<ArenaPeer>,
    omitted_at_creation: usize,
    omitted_later: usize,
}

struct ArenaResource {
    id: u64,
    placement: Placement,
    trace: Option<Box<ArenaTrace>>,
}

impl ArenaResource {
    fn peer(&self, resource: usize, base: usize) -> ArenaPeer {
        ArenaPeer { resource, offset: self.placement.va - base, end: self.placement.host_end - base }
    }
}

static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, ArenaResource>>> = Mutex::new(None);

pub(super) fn arena_depth_shape(resource: usize) -> Option<(u32, u32, u32)> {
    ARENA_OF.lock().unwrap().as_ref()?.get(&resource)?.trace.as_deref()?.shape
}

pub(super) fn arena_lineage(resource: usize) -> Option<String> {
    let (id, p, trace) = {
        let table = ARENA_OF.lock().unwrap();
        let entry = table.as_ref()?.get(&resource)?;
        (entry.id, entry.placement, entry.trace.as_deref()?.clone())
    };
    Some(format!("arena={id} heap={:#x} host_heap_range=[{:#x},{:#x}) tracked_title_range=[{:#x},{:#x}) host_initial={:?} same_heap_host_overlaps_at_creation={:?} omitted_at_creation={} same_heap_host_overlaps_appeared_later={:?} omitted_later={} first_overlapping_peer_released={:?}; overlaps are observed arena placements, not GPU alias activation or evidence about other backings",
        trace.heap, trace.offset, trace.offset + (p.host_end - p.va), p.va, p.title_end, trace.initial,
        trace.at_creation, trace.omitted_at_creation, trace.appeared_later, trace.omitted_later, trace.first_peer_released))
}
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}

/// (diagnostics, `memdiag`) The title address ranges, as (address, bytes), of the textures placed
/// in video memory here: in the arenas and on the title's pages (the slabs' small sampled
/// textures aside).
pub(super) fn diag_placed() -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = ARENAS.lock().unwrap().list.iter().flat_map(|a| a.placements.iter().map(|p| (p.va, p.title_end.max(p.host_end) - p.va))).collect();
    out.extend(PAGES.lock().unwrap().values().flatten().flat_map(|p| p.targets.values().map(|t| (t.va, t.end().saturating_sub(t.va)))));
    out
}
use crate::gpu::texture::image_bridge::ownership::HostBacking;
static PLACED_TEXTURES: Mutex<Option<std::collections::HashMap<usize, HostBacking>>> = Mutex::new(None);

#[cfg(test)]
pub(super) fn arena_of(resource: usize) -> Option<u64> {
    ARENA_OF.lock().unwrap().as_ref()?.get(&resource).map(|entry| entry.id)
}

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains_key(&resource))
}

pub(super) fn native_image_backing(resource: usize) -> HostBacking {
    PLACED_TEXTURES.lock().unwrap().as_ref().and_then(|t| t.get(&resource)).copied().unwrap_or(HostBacking::PRIVATE)
}

pub(super) fn forget(resource: usize) {
    super::texture_alias::forget(resource);
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource).is_some());
    if placed {
        forget_pages(resource);
        if let Some(slabs) = SLABS.lock().unwrap().as_mut() {
            slabs.release(resource, std::time::Instant::now());
        }
        let tracing = super::depth_lineage::twin_enabled();
        let released = {
            let mut table = ARENA_OF.lock().unwrap();
            table.as_mut().and_then(|m| {
                let released = m.remove(&resource)?;
                if tracing {
                    for entry in m.values_mut().filter(|entry| entry.id == released.id
                        && entry.placement.host_overlaps(released.placement))
                    {
                        if let Some(trace) = &mut entry.trace {
                            let base = entry.placement.va - trace.offset;
                            trace.first_peer_released.get_or_insert_with(|| released.peer(resource, base));
                        }
                    }
                }
                Some(released)
            })
        };
        if let Some(released) = released {
            ARENAS.lock().unwrap().release(released.id, released.placement, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize, prefer_local: bool) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let heap = super::residency::create_heap(dev, &desc, prefer_local).map_err(|e| {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
    }).ok()?;
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    Some(heap)
}

/// The adapter's local video memory in use against its budget, for the log.
unsafe fn local_memory(dev: &ID3D12Device) -> String {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO};
    let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return String::new() };
    let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter3>(dev.GetAdapterLuid()) else { return String::new() };
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    if adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).is_err() {
        return String::new();
    }
    format!(" (video memory {} of {} MiB budget in use)", info.CurrentUsage >> 20, info.Budget >> 20)
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    /// the GPU writes it (a render target, depth buffer or UAV texture is on it, or the title
    /// copied into a texture on it): a sampled texture keeps it whatever its own texels are
    gpu: bool,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

/// Tiles of a target that one call maps: a standard mip from its first tile (`sub`: its
/// subresource), the packed mips, or every standard tile of a texture whose mips are not told
/// apart (`start`, `count`: in the resource's overall tile order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Unit {
    sub: u32,
    start: u32,
    count: u32,
}

/// What `GetResourceTiling` says of a reserved texture, as `units_of` needs it.
#[derive(Clone, Copy, Default)]
struct Tiling {
    total: u32,
    /// a 3D texture or one slice: the packed mips are one run of tiles after the standard ones
    single_slice: bool,
    standard_mips: u32,
    packed_mips: u32,
    packed_tiles: u32,
    packed_start: u32,
}

/// The units of a texture and whether each is one mip: a single-slice texture's packed mip tail
/// is mapped at its own subresource (the PC lays it out apart from the standard mips), the rest
/// from subresource 0 – one unit per standard mip when `mips` (first tile, tiles of each, in
/// subresource order) tile the standard range exactly, else one unit for all of it.
fn units_of(t: Tiling, mips: &[(u32, u32)]) -> (Vec<Unit>, bool) {
    let tail = t.single_slice
        && t.standard_mips != 0
        && t.packed_mips != 0
        && t.packed_start > 0
        && t.packed_start < t.total
        && t.packed_tiles == t.total - t.packed_start;
    let standard_end = if tail { t.packed_start } else { t.total };
    let mut next = 0u32;
    let per_mip = t.single_slice
        && (tail || t.packed_mips == 0)
        && !mips.is_empty()
        && mips.len() == t.standard_mips as usize
        && mips.iter().all(|&(start, count)| {
            let fits = start == next && count != 0;
            next = next.saturating_add(count);
            fits
        })
        && next == standard_end;
    let mut units: Vec<Unit> = if per_mip {
        mips.iter().enumerate().map(|(i, &(start, count))| Unit { sub: i as u32, start, count }).collect()
    } else if standard_end > 0 {
        vec![Unit { sub: 0, start: 0, count: standard_end }]
    } else {
        Vec::new()
    };
    if tail {
        units.push(Unit { sub: t.standard_mips, start: t.packed_start, count: t.total - t.packed_start });
    }
    (units, per_mip)
}

struct Target {
    va: usize,
    /// (format, width, height, depth or array size, mips): for `report_growth`
    shape: (i32, u64, u32, u16, u16),
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    /// the title address of its first whole page and how many of its tiles stand on such pages
    /// (`owned_pages`); the tiles after them are `extra_pages`
    first: usize,
    own: usize,
    /// the pool tile under each of its tiles (None: no memory there – it reads as zero)
    held: Vec<Option<Tile>>,
    units: Vec<Unit>,
    /// each unit is one mip, or the packed mips (`units_of`)
    per_mip: bool,
    /// a sampled texture: the units its texels need memory for (`note_content`). None: every
    /// tile has memory while the target is mapped
    wanted: Option<Vec<bool>>,
    /// placed as a sampled texture (`place_pages`), not as the GPU's (`place`): what it is by
    /// its flags on the console, whatever was written to it since
    sampled: bool,
    /// the PC's tiles of its first mip (`texture_alias::Grid`; None: a format it does not know)
    grid: Option<super::texture_alias::Grid>,
    /// 0, or it is kept apart from the texture of another grid that holds the pages of its
    /// address (`texture_alias::keying`): it has no page of the title's addresses (`own` is 0)
    /// and this is added to its tile indices in `extra_pages`
    apart: usize,
}

impl Target {
    fn end(&self) -> usize {
        self.va.saturating_add(self.console_bytes.map_or(self.total as usize * PAGE, |b| b as usize))
    }

    /// The page tile `i` stands on.
    fn key(&self, i: usize) -> PageKey {
        if i < self.own {
            PageKey::Own(self.first + i * PAGE)
        } else {
            PageKey::Extra(self.va, self.apart + i)
        }
    }

    /// The units subresource `sub`'s texels are in.
    fn units_for(&self, sub: usize) -> std::ops::Range<usize> {
        if self.per_mip {
            let u = sub.min(self.units.len() - 1);
            u..u + 1
        } else {
            0..self.units.len()
        }
    }

    fn tiles(&self, unit: usize) -> std::ops::Range<usize> {
        let u = self.units[unit];
        u.start as usize..(u.start + u.count) as usize
    }

    /// Whether both stand tile for tile on the same pages: placed at one address in one tile
    /// grid, so each shows what the other holds.
    fn same_tiles(&self, other: &Target) -> bool {
        self.va == other.va && self.total == other.total && self.grid.is_some() && self.grid == other.grid
            && (self.first, self.own, self.apart) == (other.first, other.own, other.apart)
    }
}

/// (`texture_alias`) What is under a class of textures kept apart at an address.
struct Class {
    grid: super::texture_alias::Grid,
    /// the textures under it, mapped or not
    textures: u32,
    /// the tiles of the largest of them so far: what the class may take of the budget
    tiles: u32,
}

/// (`texture_alias`) The sampled textures left mapped on the pages of a target placed over
/// them (`keep_with`), by that target. Nothing in `title_textures` unmaps them any more: they
/// go with the target.
#[derive(Default)]
struct Kept {
    of: std::collections::HashMap<usize, Vec<usize>>,
    /// those unmapped as their last target went: the next target on their tiles has them
    /// (`adopt`)
    waiting: Vec<usize>,
}

impl Kept {
    fn tie(&mut self, target: usize, view: usize) {
        let views = self.of.entry(target).or_default();
        if !views.contains(&view) {
            views.push(view);
        }
    }

    /// `resource` was released. As a kept texture it is nobody's any more. As a target, each
    /// texture kept with it goes to `heir` – another target on its tiles – or is returned: it
    /// is to be unmapped, its pages given back, and waits for the next target there.
    fn released(&mut self, resource: usize, heir: impl Fn(usize) -> Option<usize>) -> Vec<usize> {
        self.of.retain(|_, views| {
            views.retain(|&view| view != resource);
            !views.is_empty()
        });
        self.waiting.retain(|&view| view != resource);
        let mut left = Vec::new();
        for view in self.of.remove(&resource).unwrap_or_default() {
            match heir(view) {
                Some(target) => self.tie(target, view),
                None => left.push(view),
            }
        }
        self.waiting.extend(&left);
        left
    }

    /// (one memory) `target` was placed: the waiting textures that stand on its tiles (`same`)
    /// are kept with it, and returned – they are to be mapped again.
    fn adopt(&mut self, target: usize, same: impl Fn(usize) -> bool) -> Vec<usize> {
        let (taken, waiting): (Vec<usize>, Vec<usize>) = std::mem::take(&mut self.waiting).into_iter().partition(|&view| same(view));
        self.waiting = waiting;
        for &view in &taken {
            self.tie(target, view);
        }
        taken
    }
}

/// (one memory) How many of its tiles a new texture has on the pages of its address (`Target::own`)
/// when textures of its tile grid and tile count are mapped there (`peers`: GPU-written,
/// their `own`): what a GPU-written one has, else the first – tile i of each is then one page,
/// whatever each takes its size on the console to be. None: no such texture.
fn peer_own(peers: &[(bool, usize)]) -> Option<usize> {
    peers.iter().find(|peer| peer.0).or(peers.first()).map(|peer| peer.1)
}

/// (one memory) Whether a texture of the GPU's with these PC flags goes on the pages of an
/// address off the 64 KiB grid: written by unordered access only – not a render target, not a
/// depth buffer.
fn off_grid_on_pages(flags: i32) -> bool {
    flags & RT_DS == 0 && flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
}

/// (one memory) Whether a GPU-written texture at an address off the 64 KiB grid stays a
/// resource of its own: the PC has more than 4 times the tiles the console has pages for it
/// (an array of small slices has a tile for each slice).
fn wasteful(tiles: usize, console_bytes: u64) -> bool {
    tiles > usize::try_from(console_bytes).unwrap_or(usize::MAX).div_ceil(PAGE).max(1).saturating_mul(4)
}


/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    system_pools: HashSet<u32>,
    free: [Vec<Tile>; 2],
    /// tiles given back, reusable after `COOL`
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    target_starts: BTreeSet<(usize, usize)>,
    max_target_span: usize,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
    /// (`texture_alias`) (address, class) → the textures kept apart under that class there
    apart: std::collections::HashMap<(usize, usize), Class>,
    /// the class the next grid kept apart gets (never 0, never one given before: no two grids
    /// have the same pages)
    next_class: u32,
    kept: Kept,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

/// The sampled textures on pages that have memory only where they have texels (the targets
/// whose `wanted` is set): what `note_content` and `note_gpu_write` look at first, without
/// the lock placements hold while they make resources.
static LAZY: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn is_lazy(resource: usize) -> bool {
    LAZY.lock().unwrap().as_ref().is_some_and(|s| s.contains(&resource))
}

fn lazy_tiles() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_LAZY_TILES").map_or(true, |v| v.trim() != "0"))
}

pub(super) fn writer_tiles(resource: usize) -> Option<String> {
    let (va, end, mapped, total, tiles) = {
        let g = PAGES.lock().unwrap();
        let (p, t) = g.values().flatten().find_map(|p| p.targets.get(&resource).map(|t| (p, t)))?;
        let tiles: Vec<_> = (0..t.total.min(128) as usize).map(|i| t.held.get(i).copied().flatten().and_then(|(pool, tile)|
            p.pools.get(pool as usize).map(|heap| (heap.as_raw() as usize, tile)))).collect();
        (t.va, t.end(), t.mapped, t.total, tiles)
    };
    Some(format!("reserved title_range=[{va:#x},{end:#x}) mapped={mapped} total_tiles={total} first_128_heap_tile_order={tiles:x?}"))
}

pub(super) fn tile_backing(resource: usize) -> Option<(usize, usize, usize)> {
    let g = PAGES.lock().unwrap();
    let (p, t) = g.values().flatten().find_map(|p| p.targets.get(&resource).map(|t| (p, t)))?;
    let mut local = 0;
    let mut non_local = 0;
    for tile in t.held.iter().flatten() {
        if p.system_pools.contains(&tile.0) {
            non_local += 1;
        } else {
            local += 1;
        }
    }
    Some((local, non_local, (t.total as usize).saturating_sub(local + non_local)))
}


impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            system_pools: HashSet::new(),
            free: Default::default(),
            cooling: std::collections::VecDeque::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            target_starts: BTreeSet::new(),
            max_target_span: 0,
            waited: std::collections::HashMap::new(),
            apart: std::collections::HashMap::new(),
            next_class: 1,
            kept: Kept::default(),
        })
    }

    /// The mapped targets that start at `va`, as `texture_alias::keying` takes them: tile grid,
    /// whether the GPU writes it, its own keying.
    fn peers(&self, va: usize) -> Vec<(super::texture_alias::Grid, bool, usize)> {
        self.target_starts.range((va, 0)..=(va, usize::MAX)).filter_map(|&(_, resource)| {
            let t = self.targets.get(&resource)?;
            t.grid.filter(|_| t.mapped).map(|grid| (grid, t.wanted.is_none(), t.apart))
        }).collect()
    }

    /// (one memory) `peer_own` for `t` among the mapped textures at its address.
    fn own_of_peer(&self, t: &Target) -> Option<usize> {
        let peers: Vec<(bool, usize)> = self.target_starts.range((t.va, 0)..=(t.va, usize::MAX))
            .filter_map(|&(_, resource)| self.targets.get(&resource))
            .filter(|o| o.mapped && o.total == t.total && o.grid.is_some() && o.grid == t.grid && o.apart == t.apart)
            .map(|o| (o.wanted.is_none(), o.own))
            .collect();
        peer_own(&peers)
    }

    /// (one memory) The GPU-written `target` was placed: the sampled textures unmapped when the
    /// last target on its tiles went (`forget_pages`) are mapped on its pages and kept with it.
    /// The references are given up after the lock.
    unsafe fn adopt_waiting(&mut self, dev: &ID3D12Device, target: usize) -> Vec<ID3D12Resource> {
        let Pages { kept, targets, .. } = self;
        let Some(t) = targets.get(&target) else { return Vec::new() };
        let views = kept.adopt(target, |view| targets.get(&view).is_some_and(|v| !v.mapped && v.same_tiles(t)));
        let mut used = Vec::new();
        for view in views {
            let Some(mut v) = self.targets.remove(&view) else { continue };
            if let Some(res) = super::real::reference(view) {
                let all = 0..v.units.len();
                self.acquire(dev, &res, &mut v, all, false, false);
                v.mapped = true;
                used.push(res);
            }
            self.targets.insert(view, v);
        }
        used
    }

    /// The classes textures at `va` are kept apart under, with their grids (`texture_alias::keying`).
    fn classes(&self, va: usize) -> Vec<(super::texture_alias::Grid, usize)> {
        self.apart.iter().filter(|(key, _)| key.0 == va).map(|(key, class)| (class.grid, key.1)).collect()
    }

    /// `t` is kept apart: it is under its class from here on (until `leave_class`).
    fn join_class(&mut self, t: &Target) {
        let Some(grid) = t.grid.filter(|_| t.apart != 0) else { return };
        let class = self.apart.entry((t.va, t.apart)).or_insert(Class { grid, textures: 0, tiles: 0 });
        class.textures += 1;
        class.tiles = class.tiles.max(t.total);
    }

    fn leave_class(&mut self, t: &Target) {
        if let Some(class) = self.apart.get_mut(&(t.va, t.apart)).filter(|_| t.apart != 0) {
            class.textures = class.textures.saturating_sub(1);
            if class.textures == 0 {
                self.apart.remove(&(t.va, t.apart));
            }
        }
    }

    /// Bytes of pages that keeping grids apart takes, for the budget: the tiles of the largest
    /// texture of each class, with memory or not yet, and the pages of classes no texture is
    /// under any more until they are swept.
    fn apart_spent(&self) -> usize {
        let classes: usize = self.apart.values().map(|class| class.tiles as usize).sum();
        let left = self.extra_pages.keys().filter(|&&(va, i)| i >> 32 != 0 && !self.apart.contains_key(&(va, i >> 32 << 32))).count();
        (classes + left) * PAGE
    }

    fn overlapping_targets(&self, va: usize, end: usize) -> impl Iterator<Item = &Target> {
        let start = va.saturating_sub(self.max_target_span);
        self.target_starts.range((start, 0)..(end, 0)).filter_map(move |&(_, resource)| {
            let t = &self.targets[&resource];
            (va < t.end()).then_some(t)
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.cooling, now);
    }

    fn available_tile(&mut self, local_only: bool) -> Option<Tile> {
        self.free[0].pop()
            .or_else(|| (!local_only).then(|| self.free[1].pop()).flatten())
            .or_else(|| {
                let i = self.cooling.iter().take_while(|(_, at)| at.elapsed() >= COOL)
                    .position(|(tile, _)| !local_only || !self.system_pools.contains(&tile.0))?;
                self.cooling.remove(i).map(|(tile, _)| tile)
            })
    }

    unsafe fn take_tile(&mut self, dev: &ID3D12Device, prefer_local: bool, allow_system: &mut bool) -> Option<Tile> {
        if let Some(tile) = self.available_tile(!*allow_system) {
            return Some(tile);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            // every kind of texture (resource heap tier 2: the only flags for both)
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let heap = match super::residency::create_heap(dev, &desc, prefer_local) {
            Ok(heap) => heap,
            Err(e) => {
                if !*allow_system {
                    *allow_system = true;
                    if let Some(tile) = self.available_tile(false) {
                        return Some(tile);
                    }
                }
                crate::xlog_warn!("D3D12.X: no memory for more pages of title textures ({e}){}", local_memory(dev));
                return None;
            }
        };
        super::residency::manage_heap(&heap, prefer_local);
        let props = heap.GetDesc().Properties;
        let system = props.Type == D3D12_HEAP_TYPE_CUSTOM && props.MemoryPoolPreference == D3D12_MEMORY_POOL_L0;
        self.pools.push(heap);
        let h = self.pools.len() as u32 - 1;
        if system {
            self.system_pools.insert(h);
            *allow_system = true;
        }
        self.free[usize::from(system)].extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    fn page(&self, key: PageKey) -> Option<&Page> {
        match key {
            PageKey::Own(a) => self.pages.get(&a),
            PageKey::Extra(va, i) => self.extra_pages.get(&(va, i)),
        }
    }

    fn page_mut(&mut self, key: PageKey) -> Option<&mut Page> {
        match key {
            PageKey::Own(a) => self.pages.get_mut(&a),
            PageKey::Extra(va, i) => self.extra_pages.get_mut(&(va, i)),
        }
    }

    /// Map a linear range of tiles starting at the given subresource, one call per pool heap
    /// (`tiles[i]`: where tile i of the range goes; None: its mapping stays as it is).
    unsafe fn map(&self, res: &ID3D12Resource, subresource: u32, tiles: &[Option<Tile>]) {
        let mut heaps: Vec<u32> = tiles.iter().flatten().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| if r.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: subresource, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }


    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    /// Unmap the tiles of one unit of `res`: they read as zero from then on (the caller signals).
    unsafe fn unmap_unit(&self, res: &ID3D12Resource, unit: Unit) {
        let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: unit.sub, ..Default::default() };
        let region = D3D12_TILE_REGION_SIZE { NumTiles: unit.count, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&unit.count),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// `t` lets go of its tiles `range`. A page no target holds any more is kept `IDLE` for the
    /// next target at its address; with `immediate_cool` its tile goes back to the pool at once.
    fn release(&mut self, t: &mut Target, range: std::ops::Range<usize>, now: std::time::Instant, immediate_cool: bool) {
        for i in range {
            if t.held[i].take().is_none() {
                continue;
            }
            let key = t.key(i);
            let Some(p) = self.page_mut(key) else { continue };
            p.users = p.users.saturating_sub(1);
            if p.users != 0 {
                continue;
            }
            if immediate_cool {
                let gone = match key {
                    PageKey::Own(a) => self.pages.remove(&a),
                    PageKey::Extra(va, i) => self.extra_pages.remove(&(va, i)),
                };
                if let Some(p) = gone {
                    self.cooling.push_back((p.tile, now));
                }
            } else {
                p.idle_since = Some(now);
                self.idle.push_back((key, now));
            }
        }
    }

    /// Memory for `t`'s tiles of `units`, mapped on the `Pages` queue: the pages that exist at
    /// their addresses, shared, and with `create` new ones for the rest. `gpu`: the GPU writes
    /// them (`Page::gpu`). False: a page to create had no memory (what was taken stays `t`'s).
    unsafe fn acquire(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, t: &mut Target, units: std::ops::Range<usize>, create: bool, gpu: bool) -> bool {
        self.sweep(std::time::Instant::now());
        let prefer_local = super::residency::protected_texture(&res.GetDesc());
        let mut allow_system = !prefer_local;
        let mut complete = true;
        let mut mapped = false;
        for u in units {
            let range = t.tiles(u);
            let mut changed = false;
            for i in range.clone() {
                let key = t.key(i);
                if t.held[i].is_some() {
                    if gpu {
                        if let Some(p) = self.page_mut(key) {
                            p.gpu = true;
                        }
                    }
                    continue;
                }
                let tile = match self.page_mut(key) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.gpu |= gpu;
                        Some(p.tile)
                    }
                    None if create => self.take_tile(dev, prefer_local, &mut allow_system).inspect(|&tile| {
                        let page = Page { tile, users: 1, idle_since: None, gpu };
                        match key {
                            PageKey::Own(a) => self.pages.insert(a, page),
                            PageKey::Extra(va, i) => self.extra_pages.insert((va, i), page),
                        };
                    }),
                    None => None,
                };
                match tile {
                    Some(tile) => {
                        t.held[i] = Some(tile);
                        changed = true;
                    }
                    None if create => {
                        complete = false;
                        break;
                    }
                    None => {}
                }
            }
            if changed {
                self.map(res, t.units[u].sub, &t.held[range]);
                mapped = true;
            }
            if !complete {
                break;
            }
        }
        if mapped {
            self.value += 1;
            let _ = self.queue.Signal(&self.fence, self.value);
        }
        complete
    }

    /// A target the GPU writes got its pages in `va..end`: the sampled textures at those
    /// addresses that have memory only where they have texels take them too – what the GPU
    /// writes there is what they read, as on the console.
    ///
    /// Those textures are known here by their address alone: each is used through a reference of
    /// its own (`real::reference`; one being released for good on another thread is left out).
    /// The references are returned for the caller to let go of once the pages are unlocked – the
    /// last reference to a texture, let go of under that lock, came back for the lock (`forget`).
    #[must_use]
    unsafe fn share_with_sampled(&mut self, dev: &ID3D12Device, va: usize, end: usize) -> Vec<ID3D12Resource> {
        let start = va.saturating_sub(self.max_target_span);
        let keys: Vec<usize> = self.target_starts.range((start, 0)..(end, 0)).map(|&(_, resource)| resource)
            .filter(|r| self.targets.get(r).is_some_and(|t| t.mapped && t.wanted.is_some() && va < t.end()))
            .collect();
        let mut used = Vec::new();
        for key in keys {
            let Some(mut t) = self.targets.remove(&key) else { continue };
            if let Some(res) = super::real::reference(key) {
                let all = 0..t.units.len();
                self.acquire(dev, &res, &mut t, all, false, false);
                used.push(res);
            }
            self.targets.insert(key, t);
        }
        used
    }
}


/// `XCL_PAGES_REPORT=1`: each time the pages in use pass another 512 MiB, which targets hold
/// them – grouped by shape, with the tiles past their console size (`extra_pages`) apart.
fn report_growth(p: &Pages) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const STEP: usize = 512 << 20;
    static NEXT: AtomicUsize = AtomicUsize::new(STEP);
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let bytes = (p.pages.len() + p.extra_pages.len()) * PAGE;
    let next = NEXT.load(Ordering::Relaxed);
    if bytes < next || !*ON.get_or_init(|| std::env::var("XCL_PAGES_REPORT").is_ok_and(|v| v == "1")) || NEXT.compare_exchange(next, (bytes / STEP + 1) * STEP, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let mut groups: std::collections::HashMap<(i32, u64, u32, u16, u16), (usize, usize, usize)> = std::collections::HashMap::new();
    for t in p.targets.values().filter(|t| t.mapped) {
        let g = groups.entry(t.shape).or_default();
        g.0 += 1;
        g.1 += t.held[..t.own].iter().flatten().count();
        g.2 += t.held[t.own..].iter().flatten().count();
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by_key(|(_, (_, own, extra))| std::cmp::Reverse(own + extra));
    let idle = p.pages.values().chain(p.extra_pages.values()).filter(|pg| pg.users == 0).count();
    let mb = |tiles: usize| (tiles * PAGE) >> 20;
    let top: Vec<String> = groups
        .iter()
        .take(8)
        .map(|((f, w, h, d, m), (n, own, extra))| format!("fmt {f} {w}x{h}x{d} {m} mips ×{n}: {} MB ({} MB past console size)", mb(own + extra), mb(*extra)))
        .collect();
    crate::xlog_info!(
        "D3D12.X: target pages {} MB: {} MB at title addresses, {} MB past console sizes, {} MB idle; {} targets mapped in {} shapes; most: {}",
        bytes >> 20,
        mb(p.pages.len()),
        mb(p.extra_pages.len()),
        mb(idle),
        p.targets.values().filter(|t| t.mapped).count(),
        groups.len(),
        top.join("; ")
    );
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Option<Tile>]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &tile in tiles {
        let mine = tile.is_some_and(|(h, _)| h == heap);
        let t = tile.map_or(0, |(_, t)| t);
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

unsafe fn supports_reserved_texture(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut format = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(
        D3D12_FEATURE_FORMAT_SUPPORT,
        (&mut format as *mut D3D12_FEATURE_DATA_FORMAT_SUPPORT).cast(),
        std::mem::size_of_val(&format) as u32,
    ).is_ok() && format.Support2.0 & D3D12_FORMAT_SUPPORT2_TILED.0 == 0 {
        return false;
    }
    if desc.SampleDesc.Count <= 1 {
        return true;
    }
    let mut msaa = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
        Format: desc.Format,
        SampleCount: desc.SampleDesc.Count,
        Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE,
        NumQualityLevels: 0,
    };
    dev.CheckFeatureSupport(
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
        (&mut msaa as *mut D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS).cast(),
        std::mem::size_of_val(&msaa) as u32,
    ).is_err() || msaa.NumQualityLevels != 0
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). `lazy`: a
/// sampled texture – it shares the pages that exist at its addresses and gets more only where
/// it has texels (`note_content`). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    lazy: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    if !supports_reserved_texture(dev, desc) {
        return None;
    }
    // (`texture_alias`) what the split's arenas take of the budget for memory of their own
    let split_arenas = if super::texture_alias::enabled() { ARENAS.lock().unwrap().split_bytes() } else { 0 };
    let mut g = PAGES.lock().unwrap();
    g.entry(dev.as_raw() as usize).or_insert_with(|| Pages::new(dev)).as_ref()?;
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        if require_aligned {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({hr:#x}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
            }
        } else if !lazy {
            // (one memory: a GPU-written texture off the 64 KiB grid – no arena takes it there)
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_info!("D3D12.X: one memory: a GPU-written texture at {va:#x} ({}x{}, format {}) not reserved ({hr:#x}): a resource of its own, as before", desc.Width, desc.Height, desc.Format.0);
            }
        }
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut tiling = Tiling::default();
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let made = res.GetDesc();
    let mut n = u32::from(made.MipLevels);
    let mut standard = vec![D3D12_SUBRESOURCE_TILING::default(); n as usize];
    dev.GetResourceTiling(res, Some(&mut tiling.total), Some(&mut packed), None, Some(&mut n), 0, standard.as_mut_ptr());
    let total = tiling.total;
    tiling.single_slice = made.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || made.DepthOrArraySize == 1;
    tiling.standard_mips = u32::from(packed.NumStandardMips);
    tiling.packed_mips = u32::from(packed.NumPackedMips);
    tiling.packed_tiles = packed.NumTilesForPackedMips;
    tiling.packed_start = packed.StartTileIndexInOverallResource;
    let mips: Vec<(u32, u32)> = standard.iter().take((n.min(tiling.standard_mips)) as usize)
        .map(|s| (s.StartTileIndexInOverallResource, s.WidthInTiles.saturating_mul(u32::from(s.HeightInTiles)).saturating_mul(u32::from(s.DepthInTiles))))
        .collect();
    let (units, per_mip) = units_of(tiling, &mips);
    let fail = |g: std::sync::MutexGuard<'_, std::collections::BTreeMap<usize, Option<Pages>>>| {
        drop(g);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        None
    };
    let Some((first, own)) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten() else {
        return fail(g);
    };
    let shape = (desc.Format.0, desc.Width, desc.Height, desc.DepthOrArraySize, desc.MipLevels);
    let sampled = lazy;
    // (one memory) a GPU-written texture at an address off the 64 KiB grid (`place_impl`)
    let off_grid = !sampled && va % PAGE != 0;
    if off_grid && console_bytes.is_some_and(|bytes| wasteful(total as usize, bytes)) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: one memory: a GPU-written texture at {va:#x} ({}x{}x{}, format {}) off the 64 KiB grid has {total} tiles on the PC for {} bytes on the console: a resource of its own, as before ({n} so far)", desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format.0, console_bytes.unwrap_or(0));
        }
        return fail(g);
    }
    let lazy = lazy && lazy_tiles();
    // (`texture_alias`) its tile grid: the PC's own tiles of its first mip, when that is a standard one
    let first_mip = standard.first().filter(|_| tiling.standard_mips != 0 && n != 0)
        .map(|t| [t.WidthInTiles, u32::from(t.HeightInTiles), u32::from(t.DepthInTiles)]);
    let grid = super::texture_alias::Shape::of(&made).map(|s| super::texture_alias::Grid::of_tiling(&s, first_mip));
    let mut target = Target {
        va,
        shape,
        console_bytes,
        total,
        mapped: true,
        first,
        own,
        held: vec![None; total as usize],
        wanted: lazy.then(|| vec![false; units.len()]),
        sampled,
        units,
        per_mip,
        grid,
        apart: 0,
    };
    let all = 0..target.units.len();
    let Some(pages) = g.get_mut(&(dev.as_raw() as usize)).and_then(Option::as_mut) else {
        return fail(g);
    };
    // over a GPU-written texture of another tile grid it would show that one's tiles in the order
    // of its own (a wider one: the picture twice, side by side): pages of its own
    if let Some(grid) = grid.filter(|_| super::texture_alias::enabled()) {
        // (a GPU-written texture whose size on the console is not known – a multisampled target:
        // no `title_textures::retire_overlapping` follows, the sampled ones under it stay mapped)
        let stays = !sampled && console_bytes.is_none();
        let fresh = (pages.next_class as usize) << 32;
        let apart = super::texture_alias::keying(grid, &pages.peers(va), &pages.classes(va), fresh, stays);
        // the tiles it has more than its class: within the budget for memory of their own, else
        // it shares by address, as before
        let more = (total as usize).saturating_sub(pages.apart.get(&(va, apart)).map_or(0, |class| class.tiles as usize)) * PAGE;
        if apart != 0 && more != 0 && !super::texture_alias::fits(split_arenas + pages.apart_spent(), more, super::texture_alias::budget()) {
            super::texture_alias::over_budget(va);
        } else if apart != 0 {
            target.apart = apart;
            target.own = 0;
            if apart == fresh {
                pages.next_class = pages.next_class.wrapping_add(1).max(1);
            }
        }
    }
    // (one memory) on the tiles of a texture of its grid and tile count at its address: as many
    // of its tiles on the address's pages as that one has
    if super::texture_alias::one_memory() {
        if let Some(own) = pages.own_of_peer(&target) {
            target.own = own;
        }
    }
    let mut shared = Vec::new();
    if lazy {
        pages.acquire(dev, res, &mut target, all, false, false);
    } else if !pages.acquire(dev, res, &mut target, all, true, true) {
        pages.release(&mut target, 0..total as usize, std::time::Instant::now(), true);
        return fail(g);
    } else {
        shared = pages.share_with_sampled(dev, va, target.end());
    }
    pages.max_target_span = pages.max_target_span.max(target.end() - va);
    pages.target_starts.insert((va, raw as usize));
    pages.join_class(&target);
    pages.targets.insert(raw as usize, target);
    if !sampled && super::texture_alias::one_memory() {
        shared.extend(pages.adopt_waiting(dev, raw as usize));
    }
    report_growth(pages);
    drop(g);
    drop(shared);
    if off_grid {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: one memory: a GPU-written texture at {va:#x} ({}x{}x{}, format {}) is on the pages of its address, off the 64 KiB grid ({n} so far)", desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format.0);
        }
    }
    if lazy {
        LAZY.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(raw as usize, HostBacking::TILED);
    Some(hr)
}

/// Whether `note_content` wants to know what `resource` holds (a sampled texture on pages).
pub(super) fn wants_content(resource: usize) -> bool {
    is_lazy(resource)
}

/// Whether `bytes` has a byte that is not zero.
pub(super) fn any_nonzero(bytes: &[u8]) -> bool {
    bytes.chunks(4096).any(|c| c.iter().fold(0u8, |a, &b| a | b) != 0)
}

/// Whether texels of `format` that are all zero bytes read as a tile without memory does: zero
/// in every component the format has. Not BC1: its zero block is opaque black (alpha 1).
pub(super) fn zero_is_unmapped(format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC1_TYPELESS, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB};
    !matches!(format, DXGI_FORMAT_BC1_TYPELESS | DXGI_FORMAT_BC1_UNORM | DXGI_FORMAT_BC1_UNORM_SRGB)
}

/// The device of `res`.
unsafe fn device_of(res: &ID3D12Resource) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    res.GetDevice(&mut dev).ok()?;
    dev
}

/// What the sampled texture `res` on pages holds as it is about to be uploaded from the title's
/// memory (`texels[s]`: subresource s has texels that a tile without memory would not read
/// as): those mips get memory, and the others – nothing mapped under them on the console, or
/// zeros – give back the pages no GPU-written target shares; unmapped they read as the same
/// zeros. The title's queues wait for the mappings (`before_execute`).
pub(super) unsafe fn note_content(res: &ID3D12Resource, texels: &[bool]) {
    let key = res.as_raw() as usize;
    if !is_lazy(key) {
        return;
    }
    let Some(dev) = device_of(res) else { return };
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let Some(mut t) = pages.targets.remove(&key) else { return };
    if t.mapped && t.wanted.is_some() {
        let mut want = vec![false; t.units.len()];
        for (s, _) in texels.iter().enumerate().filter(|(_, &n)| n) {
            for u in t.units_for(s) {
                want[u] = true;
            }
        }
        let now = std::time::Instant::now();
        let mut unmapped = false;
        for (u, &wanted) in want.iter().enumerate() {
            let range = t.tiles(u);
            if wanted {
                if t.held[range].iter().any(Option::is_none) {
                    pages.acquire(&dev, res, &mut t, u..u + 1, true, false);
                }
            } else if t.held[range.clone()].iter().any(Option::is_some)
                && !range.clone().any(|i| t.held[i].is_some() && pages.page(t.key(i)).is_some_and(|p| p.gpu))
            {
                pages.unmap_unit(res, t.units[u]);
                pages.release(&mut t, range, now, false);
                unmapped = true;
            }
        }
        if unmapped {
            pages.value += 1;
            let _ = pages.queue.Signal(&pages.fence, pages.value);
        }
        t.wanted = Some(want);
    }
    pages.targets.insert(key, t);
    report_growth(pages);
}

/// The title's GPU work writes `resource` (a copy or a resolve into it): a sampled texture on
/// pages has memory under every tile from here on, as a render target has.
pub(super) unsafe fn note_gpu_write(resource: usize) {
    super::texture_alias::note_gpu_write(resource);
    if !is_lazy(resource) {
        return;
    }
    let raw = resource as *mut c_void;
    let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    let Some(dev) = device_of(res) else { return };
    // (declared before the lock: let go of after it)
    let mut shared = Vec::new();
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let Some(mut t) = pages.targets.remove(&resource) else { return };
    if t.wanted.take().is_some() {
        if let Some(s) = LAZY.lock().unwrap().as_mut() {
            s.remove(&resource);
        }
    }
    if t.mapped {
        let all = 0..t.units.len();
        pages.acquire(&dev, res, &mut t, all, true, true);
        shared = pages.share_with_sampled(&dev, t.va, t.end());
    }
    pages.targets.insert(resource, t);
    drop(g);
    drop(shared);
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(mut t) = pages.targets.remove(&key) else { continue };
        if t.mapped {
            let total = t.total;
            t.mapped = false;
            pages.unmap(res, total);
            pages.release(&mut t, 0..total as usize, now, true);
        }
        pages.targets.insert(key, t);
        return;
    }
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released
/// (a sampled texture: the pages at its addresses and those its texels needed).
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let Some(dev) = device_of(res) else { return false };
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let Some(mut t) = pages.targets.remove(&key) else { return false };
    let mut ok = true;
    if !t.mapped {
        let all = 0..t.units.len();
        ok = match t.wanted.clone() {
            None => pages.acquire(&dev, res, &mut t, all, true, true),
            Some(want) => {
                pages.acquire(&dev, res, &mut t, all, false, false);
                want.iter().enumerate().filter(|(_, &w)| w).all(|(u, _)| pages.acquire(&dev, res, &mut t, u..u + 1, true, false))
            }
        };
        if ok {
            t.mapped = true;
        } else {
            let total = t.total;
            pages.unmap(res, total);
            pages.release(&mut t, 0..total as usize, std::time::Instant::now(), true);
        }
    }
    pages.targets.insert(key, t);
    ok
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    if let Some(s) = LAZY.lock().unwrap().as_mut() {
        s.remove(&resource);
    }
    // (references given up after the lock: a last one released under it came back here)
    let mut unmapped: Vec<ID3D12Resource> = Vec::new();
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(mut t) = pages.targets.remove(&resource) else { continue };
        pages.target_starts.remove(&(t.va, resource));
        pages.leave_class(&t);
        let total = t.total as usize;
        pages.release(&mut t, 0..total, now, false);
        // (`texture_alias`) the sampled textures kept mapped on its pages (`keep_with`): another
        // GPU-written target on those tiles has them from here on; with none they are unmapped
        // and give their pages back – what happened to them when the target was placed, before
        let Pages { kept, targets, target_starts, .. } = pages;
        let left = kept.released(resource, |view| {
            let v = targets.get(&view)?;
            target_starts.range((v.va, 0)..=(v.va, usize::MAX)).map(|&(_, other)| other)
                .find(|&other| other != view && targets.get(&other).is_some_and(|o| o.mapped && o.wanted.is_none() && o.same_tiles(v)))
        });
        for view in left {
            let Some(mut v) = pages.targets.remove(&view) else { continue };
            if v.mapped {
                if let Some(res) = unsafe { super::real::reference(view) } {
                    let total = v.total;
                    v.mapped = false;
                    unsafe { pages.unmap(&res, total) };
                    // (its pages idle like the target's: a target made again at the address
                    // within `IDLE` has what was rendered there, and the view with it – `adopt`)
                    pages.release(&mut v, 0..total as usize, now, false);
                    unmapped.push(res);
                }
            }
            pages.targets.insert(view, v);
        }
        return;
    }
}


/// Before GPU work on `queue`: it waits for the pages mapped since its last work on that device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        super::markers::queue_fence(key, "VRAM page mapping", &pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// (tests) The pages of `device` that no resource uses go back to the pools now, not after
/// `IDLE`: a test that counts the device's pages (`page_bytes_for`) starts from those in use.
/// Tests of one adapter share its device, and the pages an earlier test left idle are swept
/// while the next one counts – in a run one test at a time, 2 s later: its count fell below
/// where it started.
#[cfg(test)]
pub(super) fn sweep_idle_now(device: usize) {
    let mut g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get_mut(&device) else { return };
    p.sweep(std::time::Instant::now() + IDLE);
}

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// (bytes of pages in use by tiled textures, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((0, 0), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Where `resource`'s video memory is, for `texture_alias`' log.
pub(super) fn memory(resource: usize) -> super::texture_alias::Memory {
    use super::texture_alias::Memory;
    if let Some(apart) = PAGES.lock().unwrap().values().flatten().find_map(|p| p.targets.get(&resource).map(|t| t.apart)) {
        return Memory::Pages(apart);
    }
    if let Some(id) = ARENA_OF.lock().unwrap().as_ref().and_then(|m| m.get(&resource)).map(|entry| entry.id) {
        return Memory::Arena(id);
    }
    if SLABS.lock().unwrap().as_ref().is_some_and(|s| s.of.contains_key(&resource)) {
        return Memory::Slab;
    }
    Memory::Own
}

/// (textures kept on pages of their own by `texture_alias::keying`, bytes of the pages under
/// their classes) – `XCL_ACTIVITY`
pub fn apart_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((0, 0), |(n, bytes), p| (
        n + p.apart.values().map(|class| class.textures as usize).sum::<usize>(),
        bytes + p.extra_pages.keys().filter(|key| key.1 >> 32 != 0).count() * PAGE,
    ))
}

/// Bytes of pages that keeping grids apart takes of the budget (`Pages::apart_spent`).
fn apart_spent() -> usize {
    PAGES.lock().unwrap().values().flatten().map(Pages::apart_spent).sum()
}

/// Bytes of the arenas made for placements kept apart from the one at their address – `XCL_ACTIVITY`
pub fn split_arena_bytes() -> usize {
    ARENAS.lock().unwrap().split_bytes()
}

/// Whether `a` and `b` stand tile for tile on the same pages: placed at one address in one tile
/// grid, so each shows what the other holds.
#[cfg(test)]
pub(super) fn same_tiles(a: usize, b: usize) -> bool {
    let g = PAGES.lock().unwrap();
    g.values().flatten().any(|p| matches!((p.targets.get(&a), p.targets.get(&b)), (Some(a), Some(b)) if a.same_tiles(b)))
}

/// (one memory) The GPU-written texture the sampled texture `view` stands tile for tile on:
/// mapped at its address in its tile grid, on the same pages.
pub(super) fn gpu_target_on_tiles(view: usize) -> Option<usize> {
    if !lazy_tiles() {
        // (every texture on pages has memory under every tile: none is told to be the GPU's)
        return None;
    }
    let g = PAGES.lock().unwrap();
    g.values().flatten().find_map(|p| {
        // (a texture placed as the GPU's that `title_textures` tracks too – a colour target the
        // CPU writes through its address – is not a view of its neighbour: it is uploaded, and
        // never kept with another; and only one placed as the GPU's is stood on: a sampled
        // texture a copy went into may yet be filled by the CPU)
        let v = p.targets.get(&view).filter(|v| v.sampled)?;
        p.target_starts.range((v.va, 0)..=(v.va, usize::MAX)).map(|&(_, other)| other)
            .find(|&other| other != view && p.targets.get(&other).is_some_and(|o| o.mapped && !o.sampled && o.wanted.is_none() && o.same_tiles(v)))
    })
}

/// Whether the sampled texture `view` stays mapped as target `by` is placed over it: it stands
/// tile for tile on `by`'s pages. It is kept with `by` from here on (`Kept`): unmapped when
/// `by`'s pages are released and no other target is on them.
pub(super) fn keep_with(by: usize, view: usize) -> bool {
    let mut g = PAGES.lock().unwrap();
    for p in g.values_mut().flatten() {
        if let (Some(a), Some(b)) = (p.targets.get(&by), p.targets.get(&view)) {
            // (only a sampled texture is another's view: one placed as the GPU's is unmapped
            // at the retire, as it was)
            let same = b.sampled && a.same_tiles(b);
            if same {
                p.kept.tie(by, view);
            }
            return same;
        }
    }
    false
}

/// (one memory) Whether the sampled texture `view`, kept with a mapped GPU-written texture on
/// whose tiles it stands (`keep_with`), stays mapped as a placement over `va..end` retires it:
/// that texture's memory is clear of `va..end`, and `view`'s pages are that texture's - only
/// its own console size reaches into `va..end`. Seen: an R9G9B9E5 view at an address off the
/// 64 KiB grid, 983040 bytes in its console layout (64KB_D), on the tiles of the R32_UINT
/// texture there, 921600 bytes in 256B_D; the next R32_UINT texture, placed 921600 bytes on,
/// unmapped the view and it read zero. `XCL_KEEP_TILE_VIEWS=0`: unmapped, as before.
pub(super) fn kept_clear_of(view: usize, va: usize, end: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("XCL_KEEP_TILE_VIEWS").map_or(true, |v| v.trim() != "0")) {
        return false;
    }
    let kept = {
        let g = PAGES.lock().unwrap();
        g.values().flatten().find_map(|p| {
            let v = p.targets.get(&view).filter(|v| v.sampled && v.mapped)?;
            p.kept.of.iter().filter(|(_, views)| views.contains(&view)).find_map(|(by, _)| p.targets.get(by))
                .filter(|t| t.mapped && t.same_tiles(v) && !(t.va < end && va < t.end()))
                .map(|t| (v.va, v.end() - v.va, t.end() - t.va))
        })
    };
    let Some((at, bytes, keeper)) = kept else { return false };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: one memory: a sampled texture at {at:#x} ({bytes} bytes in its console layout) stays mapped as the texture at {va:#x} is placed: it stands on the tiles of the GPU-written one at its address ({keeper} bytes), which that placement does not reach (XCL_KEEP_TILE_VIEWS; {n} so far)");
    }
    true
}

/// Whether `resource` is a texture on pages that is unmapped (`unmap_superseded`: it gave way
/// to a placement over it).
pub(super) fn unmapped(resource: usize) -> bool {
    PAGES.lock().unwrap().values().flatten().any(|p| p.targets.get(&resource).is_some_and(|t| !t.mapped))
}

/// (`e5_copy`) Whether the sampled texture `view` shows what the GPU writes to `target`: both
/// mapped, tile for tile on the same pages.
pub(super) fn shows(view: usize, target: usize) -> bool {
    let g = PAGES.lock().unwrap();
    g.values().flatten().any(|p| matches!((p.targets.get(&view), p.targets.get(&target)), (Some(v), Some(t)) if v.mapped && t.mapped && v.same_tiles(t)))
}

/// Bytes of tiles of mapped sampled textures that have no memory (`note_content`) – what a page
/// under every tile would take more – `XCL_ACTIVITY`
pub fn unbacked_bytes() -> usize {
    let g = PAGES.lock().unwrap();
    g.values().flatten().flat_map(|p| p.targets.values()).filter(|t| t.mapped && t.wanted.is_some())
        .map(|t| t.held.iter().filter(|h| h.is_none()).count() * PAGE).sum()
}


/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum TextureInitial {
    Legacy(u32),
    Undefined,
}

const RT_DS: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0;

/// `desc` with the PC's 4 KiB placement alignment and the bytes it then takes, when the PC
/// places it that way (a small texture that is no render target or depth buffer).
unsafe fn small_host_desc(
    dev: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    default_info: impl FnOnce() -> D3D12_RESOURCE_ALLOCATION_INFO,
) -> Option<(D3D12_RESOURCE_DESC, usize)> {
    if desc.SampleDesc.Count != 1 || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN
        || desc.Flags.0 & RT_DS != 0
        || !matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D
            | D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        || default_info().SizeInBytes > PAGE as u64
    {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC { Alignment: SMALL as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[desc]);
    if info.Alignment != SMALL as u64 || info.SizeInBytes == 0 || info.SizeInBytes >= PAGE as u64 {
        return None;
    }
    Some((desc, info.SizeInBytes as usize))
}

unsafe fn small_desc(
    dev: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    va: usize,
    console_bytes: Option<u64>,
    default_info: impl FnOnce() -> D3D12_RESOURCE_ALLOCATION_INFO,
) -> Option<(D3D12_RESOURCE_DESC, Placement)> {
    if va % SMALL != 0 {
        return None;
    }
    let (desc, bytes) = small_host_desc(dev, desc, default_info)?;
    Some((desc, Placement::new(va, bytes, SMALL, console_bytes)?))
}

fn compact_group(p: Placement, console_bytes: Option<u64>) -> Option<(usize, usize)> {
    console_bytes?;
    let uma = crate::gpu::uma::get()?;
    let (base, bytes) = if uma.is_sparse(p.va) {
        uma.sparse_reservation(p.va)?
    } else {
        uma.committed_reservation(p.va)?
    };
    bounded_compact_page(p, base, bytes)
}

fn bounded_compact_page(p: Placement, base: usize, bytes: usize) -> Option<(usize, usize)> {
    let end = base.checked_add(bytes)?;
    let page = base & !(PAGE - 1);
    let limit = page.checked_add(PAGE)?;
    (end <= limit && base <= p.va && p.title_end.max(p.host_end) <= end).then_some((page, limit))
}

/// A sampled texture of the title at `va`, backed by compact placement or shared tiles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    place_impl(this, va, desc, TextureInitial::Legacy(state), clear, console_bytes, riid, ppv, false)
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, initial: TextureInitial, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    place_impl(this, va, desc, initial, clear, console_bytes, riid, ppv, true)
}

#[allow(clippy::too_many_arguments)]
unsafe fn place_impl(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, initial: TextureInitial, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void, allow_arena: bool) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    let dev10 = if matches!(initial, TextureInitial::Undefined) {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS12).cast(),
            std::mem::size_of_val(&options) as u32).ok()?;
        if !options.EnhancedBarriersSupported.as_bool() {
            return None;
        }
        Some(dev.cast::<ID3D12Device10>().ok()?)
    } else {
        None
    };
    let info = std::cell::OnceCell::new();
    let default_info = || *info.get_or_init(|| dev.GetResourceAllocationInfo(0, &[*desc]));
    let candidate = small_desc(dev, desc, va, console_bytes, default_info);
    let end = console_bytes.and_then(|bytes| va.checked_add(bytes as usize))
        .unwrap_or_else(|| {
            let bytes = default_info().SizeInBytes;
            va.saturating_add(if bytes == u64::MAX { PAGE } else { (bytes as usize).max(PAGE) })
        });
    let device = this as usize;
    let group = candidate.and_then(|(_, p)| compact_group(p, console_bytes));
    let _placement = PendingPlacement::enter(device, va, end.max(candidate.map_or(end, |(_, p)| p.host_end)));
    let (has_compact_alias, legacy_in_group) = {
        let arenas = ARENAS.lock().unwrap();
        (arenas.overlaps_small(device, va, end),
            group.is_some_and(|(base, limit)| arenas.overlaps(device, base, limit, false)))
    };
    let tiled_in = |start, limit| PAGES.lock().unwrap().get(&device)
        .and_then(Option::as_ref).is_some_and(|pages| pages.overlapping_targets(start, limit).next().is_some());
    if has_compact_alias && tiled_in(va, end) {
        alias_conflict(va);
        return None;
    }
    let compact = candidate.filter(|_| has_compact_alias
        || (!legacy_in_group && group.is_some_and(|(base, limit)| !tiled_in(base, limit))));
    if let Some((small, p)) = compact {
        if let Some(hr) = place_in_arena(dev, &small, p, initial, dev10.as_ref(), clear, riid, ppv, has_compact_alias) {
            return Some(hr);
        }
    }
    // (`texture_alias`; 0 with `XCL_TEXTURE_ALIAS=0`: arenas are shared by address alone)
    let shape = Some(desc).filter(|_| super::texture_alias::enabled()).and_then(super::texture_alias::Shape::of).map_or(0, |s| s.key());
    let arena_placement = || {
        let info = default_info();
        let align = (info.Alignment as usize).max(PAGE);
        (info.SizeInBytes != 0 && info.SizeInBytes != u64::MAX && va % align == 0)
            .then(|| Placement::new(va, info.SizeInBytes as usize, align, console_bytes)).flatten()
            .map(|p| p.of_shape(shape))
    };
    if has_compact_alias {
        let placed = arena_placement().and_then(|p| place_in_arena(dev, desc, p, initial, dev10.as_ref(), clear, riid, ppv, true));
        if placed.is_none() {
            alias_conflict(va);
        }
        return placed;
    }
    // a texture of the shape and size of one that is in an arena at this address (a target the
    // title made UNDEFINED, or one the PC does not tile): in that arena, the same memory – on
    // pages or in a slab it had memory of its own and never showed what the target holds
    // (looked for by address first: the PC is asked for the texture's size only when one is there)
    let in_arena = shape != 0 && ARENAS.lock().unwrap().list.iter().any(|a| {
        a.device == device && !a.compact && a.placements.iter().any(|q| q.va == va && q.shape == shape)
    });
    if let Some(p) = Some(()).filter(|_| in_arena).and_then(|_| arena_placement()).filter(|&p| ARENAS.lock().unwrap().holds(device, p)) {
        if let Some(hr) = place_in_arena(dev, desc, p, initial, dev10.as_ref(), clear, riid, ppv, false) {
            return Some(hr);
        }
    }
    // a sampled texture under a page on the console that nothing tiled shares addresses with: a
    // slot of a slab, not a 64 KiB tile of its own
    if !allow_arena && console_bytes.is_some_and(|b| b < PAGE as u64) && small_slabs() && !tiled_in(va, end) {
        if let TextureInitial::Legacy(state) = initial {
            if let Some((small, bytes)) = small_host_desc(dev, desc, default_info) {
                if let Some(hr) = place_in_slab(dev, va, &small, bytes, state, clear, riid, ppv) {
                    return Some(hr);
                }
            }
        }
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let TextureInitial::Legacy(state) = initial {
            // (one memory) a texture written by unordered access whose size on the console is
            // known goes on the pages of its address off the 64 KiB grid too, as a sampled one
            // does: no arena takes it there, and as a resource of its own it never showed in the
            // sampled texture the title places at its address (`XCL_TEXTURE_ONE_MEMORY=0`: as
            // before). A render target or depth buffer there stays a resource of its own: on
            // pages it wants a clear or a discard first, and it is followed from the CPU's side
            // (`title_textures::register`) only as a resource of its own.
            let aligned = allow_arena && !(console_bytes.is_some() && off_grid_on_pages(desc.Flags.0) && super::texture_alias::one_memory());
            if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, aligned, !allow_arena, riid, ppv) {
                return Some(hr);
            }
        }
    }
    if allow_arena {
        return arena_placement().and_then(|p| place_in_arena(dev, desc, p, initial, dev10.as_ref(), clear, riid, ppv, false));
    }
    None
}

fn alias_conflict(va: usize) {
    static CONFLICTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = CONFLICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: compact texture alias conflict #{n} at {va:#x}: the live resources cannot share this placement, using the fallback");
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn place_in_arena(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, p: Placement, initial: TextureInitial, dev10: Option<&ID3D12Device10>, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void, preserve_alias: bool) -> Option<HRESULT> {
    let now = std::time::Instant::now();
    let device = dev.as_raw() as usize;
    // (`texture_alias`) an arena of its own beside the one at its address is within the budget
    // for memory of their own, arenas and pages together; past it: by address, as before
    let apart_pages = if p.shape != 0 && ARENAS.lock().unwrap().budgeted(device, p, preserve_alias, usize::MAX).1 { apart_spent() } else { 0 };
    let asked = p;
    let (expired, selected, p, split) = {
        let mut arenas = ARENAS.lock().unwrap();
        let expired = arenas.sweep(now);
        let room = super::texture_alias::budget().saturating_sub(arenas.split_bytes().saturating_add(apart_pages));
        let (p, split) = arenas.budgeted(device, p, preserve_alias, room);
        let selected = arenas.selection(device, p, preserve_alias).map(|i| i.map(|i| arenas.join(i, p)));
        (expired, selected, p, split)
    };
    drop(expired);
    if p.shape != asked.shape {
        super::texture_alias::over_budget(p.va);
    }
    let prefer_local = super::residency::protected_texture(desc);
    let (heap, offset, id, compact) = if let Some(existing) = selected.ok()? {
        existing
    } else {
        let new = Arenas::<ID3D12Heap>::new_arena(p)?;
        let heap = arena_heap(dev, new.base, new.size, new.align, prefer_local)?;
        ARENAS.lock().unwrap().insert(device, p, new, heap, split)
    };
    let hr = match initial {
        TextureInitial::Legacy(state) => {
            let hr = super::real::dev_create_placed_resource(dev.as_raw(), heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
            if hr < 0 || (*ppv).is_null() {
                ARENAS.lock().unwrap().release(id, p, now);
                return None;
            }
            hr
        }
        TextureInitial::Undefined => {
            let dev10 = dev10.unwrap();
            // a console R9G9B9E5 target made castable R11G11B10_FLOAT (`castable`)
            let castable = super::castable::wanted(desc.Format, desc.Flags).then_some(*desc);
            let desc = D3D12_RESOURCE_DESC1 {
                Dimension: desc.Dimension, Alignment: desc.Alignment, Width: desc.Width, Height: desc.Height,
                DepthOrArraySize: desc.DepthOrArraySize, MipLevels: desc.MipLevels, Format: desc.Format,
                SampleDesc: desc.SampleDesc, Layout: desc.Layout, Flags: desc.Flags, ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            let clear = (!clear.is_null()).then_some(clear.cast::<D3D12_CLEAR_VALUE>());
            let place = |formats: Option<&[windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT]>, out: &mut Option<ID3D12Resource>| {
                let mut placed = dev10.CreatePlacedResource2(&heap, offset as u64, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED, clear, formats, out);
                // a clear value is only a hint for fast clears: one the PC refuses for this resource
                // (a format its views cannot have) is dropped rather than the resource
                if let (Err(e), Some(c)) = (&placed, clear) {
                    if e.code() == windows::Win32::Foundation::E_INVALIDARG {
                        static DROPPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            crate::xlog_warn!("D3D12.X: texture format {} flags {:#x} refused clear format {} – placed without a clear value", desc.Format.0, desc.Flags.0, (*c).Format.0);
                        }
                        placed = dev10.CreatePlacedResource2(&heap, offset as u64, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED, None, formats, out);
                    }
                }
                placed
            };
            let mut placed = place(castable.map(|_| &super::castable::FORMATS[..]), &mut out);
            let mut castable = castable;
            if let (Err(e), Some(d)) = (&placed, castable.as_ref()) {
                super::castable::refused(super::castable::ARENA, d, &format!("{:#x}", e.code().0));
                castable = None;
                placed = place(None, &mut out);
            }
            if let Err(e) = placed {
                ARENAS.lock().unwrap().release(id, p, now);
                return (!p.small() && !preserve_alias).then_some(e.code().0);
            }
            let Some(out) = out else {
                ARENAS.lock().unwrap().release(id, p, now);
                return Some(crate::util::E_FAIL);
            };
            super::real::adopt(out.as_raw(), super::real::Kind::Resource);
            if let Some(d) = castable.as_ref() {
                super::castable::made(super::castable::ARENA, out.as_raw(), d);
            }
            super::residency::link(out.as_raw() as usize, heap.as_raw() as usize);
            *ppv = out.into_raw();
            crate::util::S_OK
        }
    };
    if !compact {
        super::residency::manage_heap(&heap, prefer_local);
    }
    let resource = *ppv as usize;
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(resource, HostBacking::arena(id, p.va, p.host_end));
    let tracing = super::depth_lineage::twin_enabled();
    let depth = desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    let source = super::material_gate_views::writer::enabled() && desc.Format.0 == 41 && desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    let trace = (tracing && (depth || source)).then(|| Box::new(ArenaTrace {
        heap: heap.as_raw() as usize,
        offset,
        initial,
        shape: (depth && matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D)
            && desc.MipLevels != 0 && desc.DepthOrArraySize != 0).then(|| {
                let planes = if matches!(desc.Format.0, 19..=22 | 44..=47) { 2 } else { 1 };
                (u32::from(desc.MipLevels), u32::from(desc.DepthOrArraySize), planes)
            }),
        at_creation: Vec::new(),
        appeared_later: Vec::new(),
        first_peer_released: None,
        omitted_at_creation: 0,
        omitted_later: 0,
    }));
    let mut entry = ArenaResource { id, placement: p, trace };
    let mut table = ARENA_OF.lock().unwrap();
    let table = table.get_or_insert_with(std::collections::HashMap::new);
    if tracing {
        let base = p.va - offset;
        let new_peer = entry.peer(resource, base);
        for (&other, previous) in table.iter_mut().filter(|(_, v)| v.id == id && p.host_overlaps(v.placement)) {
            let prior_peer = previous.peer(other, base);
            if let Some(trace) = &mut entry.trace {
                if trace.at_creation.len() < ARENA_PEERS {
                    trace.at_creation.push(prior_peer);
                } else {
                    trace.omitted_at_creation += 1;
                }
            }
            if let Some(trace) = &mut previous.trace {
                if trace.appeared_later.len() < ARENA_PEERS {
                    trace.appeared_later.push(new_peer);
                } else {
                    trace.omitted_later += 1;
                }
            }
        }
    }
    table.insert(resource, entry);
    Some(hr)
}

/// bytes of a slab heap: 256 slots of 4 KiB
const SLAB: usize = 1 << 20;

struct Slab<H> {
    device: usize,
    heap: H,
    /// its slots that are in use or cooling
    taken: Vec<bool>,
    live: usize,
}

/// (device, title address, bytes on the PC) of a small texture: resources of one key are one
/// memory, as placements at one address are on the console
type SlotKey = (usize, usize, usize);

/// Slots of shared heaps for the title's small sampled textures, apart from D3D12 (`H`: the
/// heap; numbers in the tests). A slot is as many 4 KiB steps as the texture takes on the PC;
/// resources placed again at its address with its size share it; it is free again `COOL` after
/// its last resource went (the GPU may still be at work on what the title released), and a slab
/// with no slot taken goes.
struct Slabs<H> {
    slabs: Vec<Option<Slab<H>>>,
    /// key → (slab, first slot, slots, resources there)
    slots: std::collections::HashMap<SlotKey, (usize, usize, usize, u32)>,
    of: std::collections::HashMap<usize, SlotKey>,
    cooling: std::collections::VecDeque<(usize, usize, usize, std::time::Instant)>,
}

impl<H: Clone> Slabs<H> {
    fn new() -> Self {
        Slabs { slabs: Vec::new(), slots: Default::default(), of: Default::default(), cooling: Default::default() }
    }

    /// Slots whose last resource went `COOL` ago are free again; the heaps of slabs left with
    /// none taken are handed back (to be dropped outside the lock).
    fn sweep(&mut self, now: std::time::Instant) -> Vec<H> {
        while self.cooling.front().is_some_and(|&(.., since)| now.saturating_duration_since(since) >= COOL) {
            let Some((slab, first, n, _)) = self.cooling.pop_front() else { break };
            self.free(slab, first, n);
        }
        let mut gone = Vec::new();
        for s in &mut self.slabs {
            if s.as_ref().is_some_and(|s| s.live == 0) {
                gone.extend(s.take().map(|s| s.heap));
            }
        }
        gone
    }

    fn free(&mut self, slab: usize, first: usize, n: usize) {
        if let Some(Some(s)) = self.slabs.get_mut(slab) {
            s.taken[first..first + n].fill(false);
            s.live -= n;
        }
    }

    /// The heap and the offset in it for a texture of `key`: the slot resources of that key
    /// have, else free slots of a slab of its device, else a new slab (`make`: its heap).
    fn take(&mut self, key: SlotKey, make: impl FnOnce() -> Option<H>) -> Option<(H, usize)> {
        if let Some(&mut (slab, first, _, ref mut users)) = self.slots.get_mut(&key) {
            *users += 1;
            return Some((self.slabs[slab].as_ref()?.heap.clone(), first * SMALL));
        }
        let n = key.2.div_ceil(SMALL);
        if n == 0 || n > SLAB / SMALL {
            return None;
        }
        let run = |s: &Slab<H>| {
            let mut first = 0;
            while first + n <= s.taken.len() {
                match s.taken[first..first + n].iter().rposition(|&t| t) {
                    Some(last) => first += last + 1,
                    None => return Some(first),
                }
            }
            None
        };
        let found = self.slabs.iter().enumerate().find_map(|(i, s)| s.as_ref().filter(|s| s.device == key.0).and_then(run).map(|first| (i, first)));
        let (slab, first) = match found {
            Some(found) => found,
            None => {
                let s = Slab { device: key.0, heap: make()?, taken: vec![false; SLAB / SMALL], live: 0 };
                let i = match self.slabs.iter().position(Option::is_none) {
                    Some(i) => {
                        self.slabs[i] = Some(s);
                        i
                    }
                    None => {
                        self.slabs.push(Some(s));
                        self.slabs.len() - 1
                    }
                };
                (i, 0)
            }
        };
        let s = self.slabs[slab].as_mut()?;
        s.taken[first..first + n].fill(true);
        s.live += n;
        self.slots.insert(key, (slab, first, n, 1));
        Some((s.heap.clone(), first * SMALL))
    }

    /// One resource fewer at `key`; the last: its slot cools from `now` (`cool` false: it was
    /// never used and is free at once).
    fn drop_use(&mut self, key: SlotKey, now: std::time::Instant, cool: bool) {
        let Some(slot) = self.slots.get_mut(&key) else { return };
        slot.3 -= 1;
        if slot.3 != 0 {
            return;
        }
        let (slab, first, n, _) = *slot;
        self.slots.remove(&key);
        if cool {
            self.cooling.push_back((slab, first, n, now));
        } else {
            self.free(slab, first, n);
        }
    }

    /// `resource` was released: whether it had a slot.
    fn release(&mut self, resource: usize, now: std::time::Instant) -> bool {
        let Some(key) = self.of.remove(&resource) else { return false };
        self.drop_use(key, now, true);
        true
    }

    /// (bytes of slots in use or cooling, bytes of the slabs)
    fn bytes(&self) -> (usize, usize) {
        let live: usize = self.slabs.iter().flatten().map(|s| s.live).sum();
        (live * SMALL, self.slabs.iter().flatten().count() * SLAB)
    }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Slabs<ID3D12Heap> {}

static SLABS: Mutex<Option<Slabs<ID3D12Heap>>> = Mutex::new(None);

fn small_slabs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SMALL_SLABS").map_or(true, |v| v.trim() != "0"))
}

/// (bytes of slab slots in use, bytes of slab heaps) – `XCL_ACTIVITY`
pub fn slab_bytes() -> (usize, usize) {
    SLABS.lock().unwrap().as_ref().map_or((0, 0), Slabs::bytes)
}

/// A small sampled texture of the title at `va` (`small`: its description with the 4 KiB
/// alignment, `bytes` what it takes) in a slot of a slab. None: no slot, or the PC refused it
/// there – the caller places it as before.
#[allow(clippy::too_many_arguments)]
unsafe fn place_in_slab(dev: &ID3D12Device, va: usize, small: &D3D12_RESOURCE_DESC, bytes: usize, state: u32, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let key = (dev.as_raw() as usize, va, bytes);
    let now = std::time::Instant::now();
    let (heap, offset, gone) = {
        let mut g = SLABS.lock().unwrap();
        let slabs = g.get_or_insert_with(Slabs::new);
        let gone = slabs.sweep(now);
        let (heap, offset) = slabs.take(key, || {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: SLAB as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            super::residency::create_heap(dev, &desc, false).map_err(|e| {
                crate::xlog_warn!("D3D12.X: no heap for small title textures: {e}");
            }).ok()
        })?;
        (heap, offset, gone)
    };
    drop(gone);
    let hr = super::real::dev_create_placed_resource(dev.as_raw(), heap.as_raw() as usize, offset, small as *const _ as usize, state as usize, clear as usize, riid, ppv);
    let mut g = SLABS.lock().unwrap();
    let slabs = g.as_mut()?;
    if hr < 0 || (*ppv).is_null() {
        slabs.drop_use(key, now, false);
        return None;
    }
    slabs.of.insert(*ppv as usize, key);
    drop(g);
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, HostBacking::slab(heap.as_raw() as usize, offset, offset + bytes));
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)].map(Some);
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)].map(Some)), [(true, 3, 1), (true, 1, 2)]);
        // tiles without memory are skipped like another heap's: their mapping stays as it is
        assert_eq!(heap_ranges(0, &[Some((0, 3)), None, None, Some((0, 4)), Some((1, 7))]), [(true, 3, 1), (false, 0, 2), (true, 4, 1), (false, 0, 1)]);
    }

    /// A single-slice texture's units are its standard mips, each from its own first tile, and
    /// its packed mips; a texture whose mips cannot be told apart is one unit (and its tail).
    #[test]
    fn units_are_mips_when_the_tiling_tells_them_apart() {
        // 1024x1024 BC7, 9 mips: 16 + 4 + 1 standard tiles, 6 packed mips in 1 tile
        let t = Tiling { total: 22, single_slice: true, standard_mips: 3, packed_mips: 6, packed_tiles: 1, packed_start: 21 };
        let per_mip = [Unit { sub: 0, start: 0, count: 16 }, Unit { sub: 1, start: 16, count: 4 }, Unit { sub: 2, start: 20, count: 1 }, Unit { sub: 3, start: 21, count: 1 }];
        assert_eq!(units_of(t, &[(0, 16), (16, 4), (20, 1)]), (per_mip.to_vec(), true));
        // tilings that do not cover the standard range in order: one unit for it
        let whole = [Unit { sub: 0, start: 0, count: 21 }, Unit { sub: 3, start: 21, count: 1 }];
        for mips in [&[][..], &[(0, 16), (16, 4)], &[(0, 16), (17, 4), (21, 1)], &[(0, 16), (16, 4), (20, 2)], &[(0, 16), (16, 0), (16, 5)]] {
            assert_eq!(units_of(t, mips), (whole.to_vec(), false), "{mips:?}");
        }
        // no packed mips: every mip a unit; an array: all of it one unit, no tail of its own
        let plain = Tiling { total: 5, single_slice: true, standard_mips: 2, ..Default::default() };
        assert_eq!(units_of(plain, &[(0, 4), (4, 1)]), (vec![Unit { sub: 0, start: 0, count: 4 }, Unit { sub: 1, start: 4, count: 1 }], true));
        let array = Tiling { total: 132, single_slice: false, standard_mips: 3, packed_mips: 6, packed_tiles: 1, packed_start: 21 };
        assert_eq!(units_of(array, &[(0, 16), (16, 4), (20, 1)]), (vec![Unit { sub: 0, start: 0, count: 132 }], false));
        // every mip packed: the one unit map_all made
        let tiny = Tiling { total: 1, single_slice: true, standard_mips: 0, packed_mips: 5, packed_tiles: 1, packed_start: 0 };
        assert_eq!(units_of(tiny, &[]), (vec![Unit { sub: 0, start: 0, count: 1 }], false));
        // a subresource's unit: its mip, the tail for the packed ones, everything when not told apart
        let target = |units: Vec<Unit>, per_mip| Target { va: 0, shape: (0, 0, 0, 0, 0), console_bytes: None, total: 22, mapped: true, first: 0, own: 22, held: vec![None; 22], units, per_mip, wanted: None, sampled: false, grid: None, apart: 0 };
        let fine = target(per_mip.to_vec(), true);
        assert_eq!([0, 1, 2, 3, 8].map(|s| fine.units_for(s)), [0..1, 1..2, 2..3, 3..4, 3..4]);
        assert_eq!(fine.tiles(1), 16..20);
        assert_eq!(target(whole.to_vec(), false).units_for(5), 0..2);
    }

    /// Small textures share slabs: 4 KiB steps, one slot for placements again at an address,
    /// free again `COOL` after the last resource went, and an empty slab's heap goes.
    #[test]
    fn small_textures_take_slots_of_shared_slabs() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut made = 0u32;
        let mut s: Slabs<u32> = Slabs::new();
        let mut take = |s: &mut Slabs<u32>, key: SlotKey| s.take(key, || { made += 1; Some(made) });
        // 300 textures of 4 KiB at 512-byte steps of the title's addresses: 2 slabs, not 300 tiles
        for k in 0..300usize {
            let (heap, offset) = take(&mut s, (1, 0x7000_0000 + k * 0x200, 4096)).unwrap();
            assert_eq!((heap, offset), (1 + (k / 256) as u32, (k % 256) * SMALL));
            s.of.insert(k, (1, 0x7000_0000 + k * 0x200, 4096));
        }
        assert_eq!(s.bytes(), (300 * SMALL, 2 * SLAB));
        // the same address and size again: the same memory; another size there: its own slot
        assert_eq!(take(&mut s, (1, 0x7000_0000, 4096)), Some((1, 0)));
        assert_eq!(take(&mut s, (1, 0x7000_0000, 3 * 4096)), Some((2, 44 * SMALL)));
        // another device: its own slab; a texture of a slab and more: none
        assert_eq!(take(&mut s, (2, 0x7000_0000, 4096)), Some((3, 0)));
        assert_eq!(take(&mut s, (1, 0x7100_0000, SLAB + 1)), None);
        assert_eq!(s.bytes(), (304 * SMALL, 3 * SLAB));
        // one of two resources of a slot goes: nothing changes; the last: the slot cools, then
        // the next texture takes it
        s.drop_use((1, 0x7000_0000, 4096), at(0), true);
        assert!(s.release(0, at(100)) && !s.release(0, at(100)));
        assert!(s.sweep(at(100 + 1999)).is_empty());
        assert_eq!(take(&mut s, (1, 0x7200_0000, 4096)), Some((2, 47 * SMALL)));
        assert!(s.sweep(at(100 + 2000)).is_empty());
        assert_eq!(take(&mut s, (1, 0x7300_0000, 4096)), Some((1, 0)));
        // a slot never used (the PC refused the resource) is free at once
        assert_eq!(take(&mut s, (1, 0x7400_0000, 8192)), Some((2, 48 * SMALL)));
        s.drop_use((1, 0x7400_0000, 8192), at(2200), false);
        assert_eq!(take(&mut s, (1, 0x7500_0000, 8192)), Some((2, 48 * SMALL)));
        // the other device's only texture goes: its slab's heap is handed back once it cooled
        s.drop_use((2, 0x7000_0000, 4096), at(3000), true);
        assert!(s.sweep(at(4999)).is_empty());
        assert_eq!(s.sweep(at(5000)), [3]);
        assert_eq!(s.bytes().1, 2 * SLAB);
        assert_eq!(take(&mut s, (2, 0x7000_0000, 4096)), Some((4, 0)));
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None, gpu: false })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None, gpu: false })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        let p = Placement::new(va, bytes, align, None).unwrap();
        a.take(0, p, now, |_, _, _| Some(n)).unwrap().unwrap().2
    }

    #[test]
    fn compact_neighbors_and_overlaps_use_one_64k_heap() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let base = 0x8_0000;
        let first = Placement::new(base + SMALL, 2 * SMALL, SMALL, None).unwrap();
        let (_, offset, id, compact) = arenas.take(0, first, now, |addr, size, align| {
            assert_eq!((addr, size, align), (base, PAGE, PAGE));
            Some(0)
        }).unwrap().unwrap();
        assert_eq!((offset, compact), (SMALL, true));
        for offset in [SMALL, 2 * SMALL, 3 * SMALL, PAGE - SMALL] {
            let p = Placement::new(base + offset, SMALL, SMALL, None).unwrap();
            let (_, actual, other, _) = arenas.take(0, p, now, |_, _, _| panic!("unexpected heap")).unwrap().unwrap();
            assert_eq!((actual, other), (offset, id));
        }
        assert_eq!(arenas.bytes(), PAGE);
    }

    #[test]
    fn compact_alias_cannot_split_live_content() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let base = 0x8_0000;
        let small = Placement::new(base + PAGE - SMALL, SMALL, SMALL, None).unwrap();
        let (_, _, id, _) = arenas.take(0, small, now, |_, _, _| Some(0)).unwrap().unwrap();
        let bigger = Placement::new(base, 2 * PAGE, PAGE, None).unwrap();
        let straddling = Placement::new(small.va, 2 * SMALL, SMALL, None).unwrap();
        let title_straddles = Placement::new(small.va, SMALL, SMALL, Some((2 * SMALL) as u64)).unwrap();
        for p in [bigger, straddling, title_straddles] {
            assert!(arenas.take(0, p, now, |_, _, _| panic!("split alias")).is_err());
        }
        let page = Placement::new(base, PAGE, PAGE, None).unwrap();
        assert_eq!(arenas.take(0, page, now, |_, _, _| panic!("split alias")).unwrap().unwrap().2, id);
        arenas.release(id, page, now);
        arenas.release(id, small, now);
        assert!(arenas.selection(0, bigger, true).is_err(), "the required live alias disappeared");
        assert!(arenas.take(0, bigger, now, |_, _, _| Some(1)).unwrap().is_some());
    }

    #[test]
    fn placement_reservations_only_conflict_on_shared_device_pages() {
        let same_page = PendingPlacement::enter(0, SMALL, 2 * SMALL);
        let next_page = PendingPlacement::enter(0, PAGE, PAGE + SMALL);
        let other_device = PendingPlacement::enter(1, SMALL, 2 * SMALL);
        let within_first = PlacementRange { device: 0, first_page: 0, last_page: 0 };
        assert!(same_page.0.overlaps(within_first));
        assert!(!next_page.0.overlaps(within_first));
        assert!(!other_device.0.overlaps(within_first));
    }

    #[test]
    fn known_suballocations_share_the_enclosing_page_only_if_their_extent_fits() {
        let page = 8 * PAGE;
        let a = Placement::new(page + SMALL, SMALL, SMALL, None).unwrap();
        let b = Placement::new(page + 2 * SMALL, SMALL, SMALL, None).unwrap();
        assert_eq!(bounded_compact_page(a, a.va, SMALL), Some((page, page + PAGE)));
        assert_eq!(bounded_compact_page(b, b.va, SMALL), Some((page, page + PAGE)));
        assert_eq!(bounded_compact_page(a, a.va, PAGE), None);
        assert_eq!(bounded_compact_page(b, a.va, SMALL), None);
    }

    #[test]
    fn compact_host_overlap_cannot_turn_disjoint_title_ranges_into_aliases() {
        let mut arenas = Arenas::new();
        let now = Instant::now();
        let first = Placement::new(0x8_0000, 2 * SMALL, SMALL, Some(SMALL as u64)).unwrap();
        let second = Placement::new(first.va + SMALL, SMALL, SMALL, None).unwrap();
        let (_, _, id, _) = arenas.take(0, first, now, |_, _, _| Some(0)).unwrap().unwrap();
        let (_, _, other, _) = arenas.take(0, second, now, |_, _, _| Some(1)).unwrap().unwrap();
        assert_ne!(id, other);
        let covering = Placement::new(first.va, 2 * SMALL, SMALL, None).unwrap();
        assert!(arenas.take(0, covering, now, |_, _, _| panic!("split alias")).is_err());
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let p = Placement::new(a0, 70 * MIB, PAGE, None).unwrap();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, p, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, p, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// (`texture_alias`) Placements of one shape at an address share its arena; one of another
    /// shape there gets an arena of its own size, which those of its shape then share. A
    /// placement no shape is told of (`XCL_TEXTURE_ALIAS=0`) goes by its addresses alone.
    #[test]
    fn a_placement_of_another_shape_at_an_address_gets_an_arena_of_its_own() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let take = |a: &mut Arenas<u32>, va: usize, bytes: usize, shape: u64| {
            let n = a.list.len() as u32;
            let p = Placement::new(va, bytes, PAGE, None).unwrap().of_shape(shape);
            (a.take(0, p, t, |_, _, _| Some(n)).unwrap().unwrap().2, p)
        };
        let (full, p_full) = take(&mut a, a0, 16 * MIB, 7);
        assert_eq!(take(&mut a, a0, 16 * MIB, 7).0, full);
        assert!(a.holds(0, p_full) && !a.holds(1, p_full) && !a.holds(0, p_full.of_shape(9)) && !a.holds(0, p_full.of_shape(0)));
        // (its shape in another size – fewer rows, mips or slices – is not known to be laid out alike)
        assert!(!a.holds(0, Placement::new(a0, 8 * MIB, PAGE, None).unwrap().of_shape(7)));
        // half the size at that address: another layout – an arena of its own, of its size
        let (half, p_half) = take(&mut a, a0, 4 * MIB, 9);
        assert_ne!(half, full);
        assert_eq!(a.bytes(), 20 * MIB);
        assert!(a.holds(0, p_half) && !a.holds(0, p_half.of_shape(11)));
        // each shape again: the memory of its own
        assert_eq!(take(&mut a, a0, 4 * MIB, 9).0, half);
        assert_eq!(take(&mut a, a0, 16 * MIB, 7).0, full);
        // a third shape is apart from both
        let (third, _) = take(&mut a, a0, 4 * MIB, 11);
        assert!(third != half && third != full);
        assert_eq!(a.bytes(), 24 * MIB);
        // inside the first arena where no texture starts: shared by address, as before
        assert_eq!(take(&mut a, a0 + 8 * MIB, 4 * MIB, 9).0, full);
        // no shape told: the newest arena that holds its addresses, whatever is there
        assert_eq!(take(&mut a, a0, 4 * MIB, 0).0, third);
        // the half-size textures released: their arena alone goes after `IDLE`
        a.release(half, p_half, t);
        a.release(half, p_half, t);
        let gone = a.sweep(t + IDLE);
        assert_eq!(gone.iter().map(|g| g.id).collect::<Vec<_>>(), [half]);
        assert_eq!(a.bytes(), 20 * MIB);
    }

    /// (`texture_alias`) An arena of its own for another shape is made while the budget for
    /// memory of their own has room for it. Past the budget the placement shares the arena at
    /// its address, with no shape told, as before; the arenas the split made are counted until
    /// they go, and then there is room again.
    #[test]
    fn a_split_arena_is_made_within_the_budget_and_shared_by_address_past_it() {
        let (t, a0, budget) = (Instant::now(), 0x1_0000_0000, 6 * MIB);
        let mut a = Arenas::new();
        let take = |a: &mut Arenas<u32>, bytes: usize, shape: u64| {
            let n = a.list.len() as u32;
            let p = Placement::new(a0, bytes, PAGE, None).unwrap().of_shape(shape);
            let room = budget.saturating_sub(a.split_bytes());
            a.take_within(0, p, t, room, |_, _, _| Some(n)).unwrap().unwrap().2
        };
        // the first arena at the address is no split: it had been made anyway
        let full = take(&mut a, 16 * MIB, 7);
        assert_eq!((a.bytes(), a.split_bytes()), (16 * MIB, 0));
        // another shape: an arena of its own, of the budget
        let half = take(&mut a, 4 * MIB, 9);
        assert_ne!(half, full);
        assert_eq!((a.bytes(), a.split_bytes()), (20 * MIB, 4 * MIB));
        // a third shape of 4 MiB with 2 MiB left: by address – the newest arena that holds its
        // addresses, as a placement no shape is told of – and nothing more is taken
        let third = take(&mut a, 4 * MIB, 11);
        assert_eq!(third, half);
        assert_eq!((a.bytes(), a.split_bytes()), (20 * MIB, 4 * MIB));
        let shared = Placement::new(a0, 4 * MIB, PAGE, None).unwrap();
        assert!(a.list.iter().any(|arena| arena.id == half && arena.placements.contains(&shared)), "placed with no shape told");
        assert!(!a.holds(0, shared.of_shape(11)));
        // one that fits what is left is still kept apart
        let small = take(&mut a, 2 * MIB, 13);
        assert!(small != half && small != full);
        assert_eq!(a.split_bytes(), 6 * MIB);
        // the split arenas released and gone: room again
        a.release(half, shared.of_shape(9), t);
        a.release(half, shared, t);
        a.release(small, Placement::new(a0, 2 * MIB, PAGE, None).unwrap().of_shape(13), t);
        assert_eq!(a.split_bytes(), 6 * MIB, "counted until they go");
        assert_eq!(a.sweep(t + IDLE).len(), 2);
        assert_eq!((a.bytes(), a.split_bytes()), (16 * MIB, 0));
        assert_ne!(take(&mut a, 4 * MIB, 11), full);
        assert_eq!(a.split_bytes(), 4 * MIB);
    }

    /// (`texture_alias`) A small placement over an address with targets of two shapes is
    /// refused an arena whether those have an arena each or share one, as they did before
    /// shapes were told: a small texture never joins an arena of large ones (`selection`), so
    /// the split takes no arena from it.
    #[test]
    fn a_small_placement_over_targets_of_two_shapes_is_refused_as_before() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        for (shapes, arenas) in [((7, 9), 2), ((0, 0), 1)] {
            let mut a = Arenas::new();
            for (bytes, shape) in [(16 * MIB, shapes.0), (4 * MIB, shapes.1)] {
                let n = a.list.len() as u32;
                a.take(0, Placement::new(a0, bytes, PAGE, None).unwrap().of_shape(shape), t, |_, _, _| Some(n)).unwrap().unwrap();
            }
            assert_eq!(a.list.len(), arenas);
            let small = Placement::new(a0, SMALL, SMALL, None).unwrap();
            assert!(a.selection(0, small, false).is_err() && a.selection(0, small, true).is_err(), "{shapes:?}");
        }
    }

    /// (`texture_alias`) A sampled texture kept mapped on a target's pages goes with that
    /// target: to another target on those tiles while there is one, else it is to be unmapped.
    #[test]
    fn a_kept_texture_goes_with_its_target() {
        let mut kept = Kept::default();
        kept.tie(1, 10);
        kept.tie(1, 11);
        kept.tie(1, 10);
        assert_eq!(kept.of[&1], [10, 11]);
        // a kept texture released: nobody's any more
        assert!(kept.released(11, |_| None).is_empty());
        assert_eq!(kept.of[&1], [10]);
        // its target released while another stands on its tiles: that one's
        assert!(kept.released(1, |view| (view == 10).then_some(2)).is_empty());
        assert_eq!((kept.of.len(), &kept.of[&2]), (1, &vec![10]));
        // that one released and none left: to be unmapped, and nobody's – it waits
        assert_eq!(kept.released(2, |_| None), [10]);
        assert!(kept.of.is_empty());
        assert_eq!(kept.waiting, [10]);
        // a resource that is neither: nothing
        assert!(kept.released(3, |_| Some(4)).is_empty() && kept.of.is_empty());
    }

    /// (one memory) A sampled texture unmapped as its last target went waits for the next
    /// target on its tiles: that one has it from then on (it is mapped again). A target on
    /// other tiles does not, and a texture released while it waits is forgotten.
    #[test]
    fn a_waiting_texture_is_kept_with_the_next_target_on_its_tiles() {
        let mut kept = Kept::default();
        for view in [10, 11, 12] {
            kept.tie(1, view);
        }
        assert_eq!(kept.released(1, |_| None), [10, 11, 12]);
        assert_eq!(kept.waiting, [10, 11, 12]);
        // a target on other tiles: none of them
        assert!(kept.adopt(2, |_| false).is_empty());
        assert_eq!((kept.of.len(), kept.waiting.len()), (0, 3));
        // one of them released meanwhile
        assert!(kept.released(11, |_| None).is_empty());
        assert_eq!(kept.waiting, [10, 12]);
        // the next target on the tiles of 10: it has 10, and 12 still waits
        assert_eq!(kept.adopt(3, |view| view == 10), [10]);
        assert_eq!((&kept.of[&3], &kept.waiting), (&vec![10], &vec![12]));
        // and goes with that target like any kept texture
        assert_eq!(kept.released(3, |_| None), [10]);
        assert_eq!(kept.waiting, [12, 10]);
    }

    /// (one memory) Textures of one tile grid and tile count at an address have the same
    /// number of tiles on the address's pages – a GPU-written one's first – so tile i of each
    /// is one page; a GPU-written texture off the 64 KiB grid goes on pages unless the PC has
    /// many times the tiles the console has pages for it.
    #[test]
    fn textures_on_one_address_agree_on_their_pages() {
        assert_eq!(peer_own(&[]), None);
        assert_eq!(peer_own(&[(false, 14)]), Some(14));
        assert_eq!(peer_own(&[(false, 14), (true, 15), (true, 16)]), Some(15));
        // 640x360 of 4 bytes: 15 tiles on the PC, 15 pages on the console
        assert!(!wasteful(15, 921_600));
        // an 8x8 array of 40 slices with a tile for each slice, 10 KiB on the console: not on pages
        assert!(wasteful(40, 10_240) && !wasteful(4, 10_240) && wasteful(5, 10_240));
        assert!(!wasteful(8, 2 * PAGE as u64) && wasteful(9, 2 * PAGE as u64) && !wasteful(4, 0));
        assert!(!wasteful(1 << 40, u64::MAX));
        // off the grid only what unordered access alone writes goes on pages
        let (rt, ds, ua) = (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0);
        assert!(off_grid_on_pages(ua) && !off_grid_on_pages(0));
        assert!(!off_grid_on_pages(rt) && !off_grid_on_pages(ds) && !off_grid_on_pages(rt | ua) && !off_grid_on_pages(ds | ua));
    }

    /// (`texture_alias`) A target kept apart stands on no page of the title's addresses: its
    /// tiles are `extra_pages` of its address under its class, where no tile of a texture that
    /// shares by address is, nor one of another class.
    #[test]
    fn a_target_kept_apart_stands_on_pages_of_its_own() {
        let target = |own, apart| Target {
            va: 0x10_4000, shape: (0, 0, 0, 0, 0), console_bytes: Some(0x3_0000), total: 4, mapped: true, first: 0x11_0000, own,
            held: vec![None; 4], units: vec![Unit { sub: 0, start: 0, count: 4 }], per_mip: false, wanted: None, sampled: false, grid: None, apart,
        };
        let keys = |t: &Target| (0..4).map(|i| match t.key(i) {
            PageKey::Own(page) => (0, page),
            PageKey::Extra(va, index) => (va, index),
        }).collect::<Vec<_>>();
        let shared = keys(&target(3, 0));
        assert_eq!(shared, [(0, 0x11_0000), (0, 0x12_0000), (0, 0x13_0000), (0x10_4000, 3)]);
        let apart = keys(&target(0, 7 << 32));
        assert_eq!(apart, [(0x10_4000, 7 << 32), (0x10_4000, (7 << 32) + 1), (0x10_4000, (7 << 32) + 2), (0x10_4000, (7 << 32) + 3)]);
        assert!(apart.iter().all(|k| !shared.contains(k)));
        assert!(keys(&target(0, 9 << 32)).iter().all(|k| !apart.contains(k)));
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<(u64, Placement)>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| {
                let va = a0 + start + k * S;
                (take(&mut a, va, S, PAGE, now), Placement::new(va, S, PAGE, None).unwrap())
            }).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for (id, p) in live.pop_front().unwrap() {
                    a.release(id, p, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    #[test]
    fn compact_gpu_textures_share_an_arena_without_neighbor_crosstalk() {
        use std::mem::ManuallyDrop;
        use crate::gpu::texture::{self, Subresource};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory4};

        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            // a CI runner's adapter is not WARP-flagged but has no local video memory of its own,
            // and what these prove (arenas living in it) needs some
            let hardware = adapter.GetDesc1().is_ok_and(|d| {
                d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 && d.DedicatedVideoMemory != 0
            });
            if !hardware {
                return crate::gpu::recomp::runner::skip_test("no hardware adapter: no local video memory");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(tier2) else { return };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 32, Height: 32, DepthOrArraySize: 1, MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let Some((_, p)) = small_desc(&dev, &desc, SMALL, None, || dev.GetResourceAllocationInfo(0, &[desc]))
                else { return };
            let stride = p.host_end - p.va;
            if SMALL + 2 * stride > PAGE {
                return;
            }
            let Some(uma) = crate::gpu::uma::get() else { return };
            let base = uma.reserve(PAGE, PAGE).expect("small texture reservation") as usize;
            let va = base + SMALL;
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let state = D3D12_RESOURCE_STATE_COMMON;
            let place_at = |va| {
                let mut raw = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, state.0 as u32, std::ptr::null(), Some(stride as u64), riid, &mut raw);
                assert!(hr.is_some_and(|hr| hr >= 0) && !raw.is_null(), "small texture at {va:#x}: {hr:?}");
                let res = ID3D12Resource::from_raw(raw);
                assert_eq!(res.GetDesc().Alignment, SMALL as u64);
                res
            };
            let a = place_at(va);
            let b = place_at(va + stride);
            let alias = place_at(va);
            let id = arena_of(a.as_raw() as usize).unwrap();
            assert_eq!(arena_of(b.as_raw() as usize), Some(id));
            assert_eq!(arena_of(alias.as_raw() as usize), Some(id));
            assert_eq!(ARENAS.lock().unwrap().list.iter().find(|arena| arena.id == id).unwrap().heap.GetDesc().Alignment, PAGE as u64);

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let activate = |before: Option<&ID3D12Resource>, after: &ID3D12Resource| {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 { Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                        pResourceBefore: before.map_or_else(|| ManuallyDrop::new(None), |r| std::mem::transmute_copy(r)),
                        pResourceAfter: std::mem::transmute_copy(after),
                    }) },
                    ..Default::default()
                }]);
                list.Close().unwrap();
                super::super::title_textures::own_work(|| {
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
            };
            for (res, byte) in [(&a, 0x4D), (&b, 0x71)] {
                activate(None, res);
                let pixels = Subresource { rows: 32, row_bytes: 128, depth: 1, data: vec![byte; 4096] };
                super::super::title_textures::own_work(|| texture::upload(&dev, &queue, res, &[pixels], state, state)).unwrap();
            }
            activate(Some(&a), &alias);
            for (res, byte) in [(&alias, 0x4D), (&b, 0x71)] {
                let got = super::super::title_textures::read_gpu_subresource(&dev, res, 0, state).unwrap();
                assert_eq!(got, vec![byte; 4096]);
            }
            drop((a, b, alias));
            assert!(uma.free(base));
        }
    }

    /// On the GPU (`texture_alias`): targets of one layout at an address are one memory – what
    /// is written through one reads through the other, whatever its format – and a target of
    /// another layout there has memory of its own, which the next of its layout shares. Needs a
    /// hardware adapter (skips on WARP); the texels are compared where the targets are on pages.
    #[test]
    fn targets_of_another_layout_at_an_address_get_memory_of_their_own() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::gpu::texture::{self, Subresource};
        use super::super::texture_alias::Memory;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory4};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            // a CI runner's adapter is not WARP-flagged but has no local video memory of its own,
            // and what these prove (arenas living in it) needs some
            let hardware = adapter.GetDesc1().is_ok_and(|d| {
                d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 && d.DedicatedVideoMemory != 0
            });
            if !hardware {
                return crate::gpu::recomp::runner::skip_test("no hardware adapter: no local video memory");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(tier2) else { return };
            if !super::super::texture_alias::enabled() {
                return;
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let state = D3D12_RESOURCE_STATE_COMMON;
            let va = 0x5_A11A_0000usize;
            let target = |format: DXGI_FORMAT, width: u64| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: width, Height: 512, DepthOrArraySize: 1, MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let mut raw = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, &desc, TextureInitial::Legacy(state.0 as u32), std::ptr::null(), None, riid, &mut raw);
                assert!(hr.is_some_and(|hr| hr >= 0) && !raw.is_null(), "{width}x512 target of format {}: {hr:?}", format.0);
                ID3D12Resource::from_raw(raw)
            };
            let at = |r: &ID3D12Resource| memory(r.as_raw() as usize);
            let a = target(DXGI_FORMAT_R8G8B8A8_UNORM, 512);
            let wide = target(DXGI_FORMAT_R8G8B8A8_UNORM, 1024);
            let cast = target(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 512);
            let wide_cast = target(DXGI_FORMAT_B8G8R8A8_UNORM, 1024);
            assert_ne!(at(&a), Memory::Own);
            assert_eq!(at(&cast), at(&a), "a target in another format of the element is the same memory");
            assert_ne!(at(&wide), at(&a), "a target twice as wide is not: its tiles are in another order");
            assert_eq!(at(&wide_cast), at(&wide), "the next of its layout shares its memory");
            if at(&a) == Memory::Pages(0) {
                let key = |r: &ID3D12Resource| r.as_raw() as usize;
                assert!(same_tiles(key(&a), key(&cast)) && same_tiles(key(&wide), key(&wide_cast)) && !same_tiles(key(&a), key(&wide)));
                assert_eq!(tile_backing(key(&wide)).map(|(local, non_local, none)| (local + non_local, none)), Some((32, 0)), "all its 8x4 tiles on pages of its own");
                let (apart, bytes) = apart_bytes();
                assert!(apart >= 2 && bytes >= 32 * PAGE, "{apart} textures apart, {bytes} bytes of the pages the two share");
                // texels through one target of a layout, read through the other
                let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
                for (to, from, width, seed) in [(&a, &cast, 512usize, 5usize), (&wide_cast, &wide, 1024, 11)] {
                    let data: Vec<u8> = (0..width * 512 * 4).map(|i| ((i / 4 % width + i / 4 / width * 3) * seed / 7) as u8).collect();
                    let pixels = Subresource { rows: 512, row_bytes: width * 4, depth: 1, data: data.clone() };
                    super::super::title_textures::own_work(|| {
                        before_execute(&queue);
                        texture::upload(&dev, &queue, to, &[pixels], state, state)
                    }).unwrap();
                    let got = super::super::title_textures::read_gpu_subresource(&dev, from, 0, state).unwrap();
                    assert!(got == data, "{width}x512: the texels written through one target read through the other");
                }
                // and the narrow targets still hold theirs: the wide ones wrote pages of their own
                let data: Vec<u8> = (0..512usize * 512 * 4).map(|i| ((i / 4 % 512 + i / 4 / 512 * 3) * 5 / 7) as u8).collect();
                assert!(super::super::title_textures::read_gpu_subresource(&dev, &a, 0, state).unwrap() == data, "a write through the wide target reached the narrow one");
            }
            drop((a, wide, cast, wide_cast));
        }
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            // a CI runner's adapter is not WARP-flagged but has no local video memory of its own,
            // and what these prove (arenas living in it) needs some
            let hardware = adapter.GetDesc1().is_ok_and(|d| {
                d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 && d.DedicatedVideoMemory != 0
            });
            if !hardware {
                return crate::gpu::recomp::runner::skip_test("no hardware adapter: no local video memory");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, TextureInitial::Legacy(D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32), &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }
}
