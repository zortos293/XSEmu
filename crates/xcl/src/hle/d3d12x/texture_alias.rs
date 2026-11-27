//! Several title textures at one address: which of them the PC shows one memory to, and a log
//! of every pair it cannot.
//!
//! A console title places textures over memory it uses again (`vram`): the transient targets
//! of a frame, a view of a target in another format of its size, the integer texture a compute
//! pass writes the blocks of a compressed one through. Each placement is a PC resource of its
//! own, and `vram` gives those at one address the same video memory: tile k of each on the
//! page at address + k x 64 KiB (`Pages`), or the same offset of a heap (arenas). That is the
//! console's sharing only while the PC lays both out alike. Tile k of a reserved texture is at
//! column k mod WidthInTiles of row k / WidthInTiles, so a texture twice as wide over the pages
//! of another shows that one's tile rows 2j and 2j + 1 side by side in its row j: its picture
//! twice at half the width, one copy from the even tile rows and one from the odd ones.
//!
//! So placements at one address are told apart by what the order of their elements in memory
//! depends on - the row. Rows follow each other, then mips, then slices: a texture with fewer
//! of them is the start of one with more (a target placed on the first face of a cube), on the
//! PC as on the console, and shares memory as before.
//!
//! * `Shape`: sizes in elements (texels, or 4x4 blocks) and the bytes of an element. Two
//!   placements with the same elements in a row are one layout whatever their formats - a view
//!   in another format of the size, a block-compressed texture and the plain one of its blocks.
//!   An arena placement whose row differs from that of the one at its address gets a heap of
//!   its own (`vram::Arenas::selection`).
//! * `Grid`: the PC's tiles in a row of the first mip (the standard 64 KiB tile shapes).
//!   Reserved textures of one grid have tile k at the same texels, also when their widths
//!   differ inside the last tile. A texture whose grid differs from that of the GPU-written one
//!   holding the pages of its address gets pages of its own (`keying`).
//!
//! What a texture kept apart holds is its own last contents, or what the CPU path uploads from
//! the title's memory (`title_textures`) - not the other texture's tiles in another order.
//!
//! Memory of their own is within a budget (`XCL_TEXTURE_ALIAS_BUDGET_MB`, 256: the arenas the
//! split makes and the pages of textures kept apart, together). Past it a texture shares by
//! address as before, and is logged.
//!
//! `XCL_TEXTURE_ALIAS=0`: every placement shares by address alone, as before. The log is
//! always on: the first `LOGGED` distinct pairs in full, then counts (`summary`, in the
//! activity lines, and at each power of two).
//!
//! The rule is the one VIctory keys its host textures by and copies between them under
//! (Marvisak/VIctory 13bbd6f, crates/victory-gpu/src/resource.rs memory_key / alias_source;
//! GPL-3: the behaviour, none of its code). VIctory holds one host texture per key and copies
//! between two keys of equal rows; here two PC resources of one layout are already the same
//! memory, so there is nothing to copy, and no shared resource object: the title holds the PC
//! objects themselves and records its own barriers on them.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

/// Whether placements of different layouts at one address are kept apart
/// (`XCL_TEXTURE_ALIAS=0`: they share by address, as before).
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_ALIAS").map_or(true, |v| v.trim() != "0"))
}

/// Whether placements of one layout at one address that had memory of their own each are made
/// one memory (`vram::place_impl`: a GPU-written texture off the 64 KiB grid on the pages of
/// its address; `title_textures::track`: a sampled texture on its tiles is the GPU's).
/// `XCL_TEXTURE_ONE_MEMORY=0`: as before.
pub(super) fn one_memory() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_ONE_MEMORY").map_or(true, |v| v.trim() != "0"))
}

/// The video memory that may go to keeping layouts apart, split arenas and pages of their own
/// together (`XCL_TEXTURE_ALIAS_BUDGET_MB`, 256): a title that keeps targets of many shapes at
/// the addresses of one pool had the sum of them in video memory, not the largest.
pub(super) fn budget() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("XCL_TEXTURE_ALIAS_BUDGET_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(256).saturating_mul(1 << 20)
    })
}

/// Whether `more` bytes fit a budget of which `spent` are taken (nothing more always does).
pub(super) fn fits(spent: usize, more: usize, budget: usize) -> bool {
    more == 0 || spent.saturating_add(more) <= budget
}

/// textures that shared by address because the budget was reached
static OVER: AtomicU64 = AtomicU64::new(0);

/// A texture at `va` that would have had memory of its own shares by address, as before: the
/// budget is reached. Said the first time, then counted (at each power of two, and `summary`).
pub(super) fn over_budget(va: usize) {
    let n = OVER.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 {
        crate::xlog_warn!("D3D12.X: texture alias: the budget for memory of their own ({} MB, XCL_TEXTURE_ALIAS_BUDGET_MB) is reached at {va:#x}: this texture shares by address, as before, and so do those after it while the budget is taken", budget() >> 20);
    } else if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: texture alias: {n} textures shared by address over the budget so far");
    }
}

/// What the PC's layout of a texture depends on, apart from its format: sizes in elements
/// (a texel, or a 4x4 block of a block-compressed format).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Shape {
    /// D3D12_RESOURCE_DIMENSION
    dimension: i32,
    /// elements of a row, and rows
    width: u64,
    height: u32,
    /// depth (3D) or array size
    layers: u16,
    mips: u16,
    samples: u32,
    /// bytes of an element and the texels of its side (4: block compressed)
    element: u32,
    block: u32,
}

impl Shape {
    /// The shape of `desc` (None: a buffer, or a format whose element is not known).
    pub(super) fn of(desc: &D3D12_RESOURCE_DESC) -> Option<Shape> {
        if !matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
            return None;
        }
        let (log2, block) = crate::hle::xg::element(desc.Format.0 as u32)?;
        let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u64::from(desc.DepthOrArraySize) } else { 1 };
        // (0 mips: all of them, down to 1x1)
        let all = 64 - desc.Width.max(u64::from(desc.Height)).max(depth).max(1).leading_zeros() as u16;
        Some(Shape {
            dimension: desc.Dimension.0,
            width: desc.Width.div_ceil(u64::from(block)),
            height: desc.Height.div_ceil(block),
            layers: desc.DepthOrArraySize.max(1),
            mips: if desc.MipLevels == 0 { all } else { desc.MipLevels },
            samples: desc.SampleDesc.Count.max(1),
            element: 1 << log2,
            block,
        })
    }

    fn volume(&self) -> bool {
        self.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0
    }

    /// What `shares` compares: the elements of a row and what they are (a 3D texture's rows
    /// too: its layers follow each other).
    fn row(&self) -> (i32, u32, u32, u64, u32) {
        (self.dimension, self.element, self.samples, self.width, if self.volume() { self.height } else { 0 })
    }

    /// Whether both have their elements in one order from their start: placed at one address
    /// each shows what the other holds, as far as it reaches.
    pub(super) fn shares(&self, other: &Shape) -> bool {
        self.row() == other.row()
    }

    /// `shares` as a number (never 0), for `vram::Placement`.
    pub(super) fn key(&self) -> u64 {
        hash(&self.row()) | 1
    }

    /// The same bytes in a row and the same rows without sharing: elements of another size.
    /// VIctory copies between such textures (texture.rs alias_copy); the PC has no layout in
    /// common for them, so here they are apart and the log says so.
    fn same_rows(&self, other: &Shape) -> bool {
        let rows = |s: &Shape| (s.dimension, s.width * u64::from(s.element), s.height, s.layers, s.samples);
        !self.shares(other) && rows(self) == rows(other)
    }
}

fn hash(v: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// The elements of a standard 64 KiB tile, (width, height, depth), of a texture with elements of
/// `element` bytes (a power of two up to 16). A block-compressed format has the tile of a plain
/// one of its block's bytes (BC1: 512x256 texels, the 128x64 of a 64-bit format in blocks).
/// Multisampled 2D tiles hold as many samples: fewer texels (D3D12 tiled resources).
fn tile_elements(element: u32, volume: bool, samples: u32) -> (u32, u32, u32) {
    let e = element.trailing_zeros().min(4) as usize;
    if volume {
        return [(64, 32, 32), (32, 32, 32), (32, 32, 16), (32, 16, 16), (16, 16, 16)][e];
    }
    let (w, h) = [(256, 256), (256, 128), (128, 128), (128, 64), (64, 64)][e];
    let (sw, sh) = match samples {
        2 => (2, 1),
        4 => (2, 2),
        8 => (4, 2),
        16 => (4, 4),
        _ => (1, 1),
    };
    (w / sw, h / sh, 1)
}

/// The PC's tiles of a reserved texture's first mip, as far as the place of tile k goes by
/// them: two textures of one grid have tile k at the same elements, for every tile both have.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Grid {
    /// what a tile holds: the bytes of an element, the samples, 3D or not
    element: u32,
    samples: u32,
    volume: bool,
    /// tiles in a row (tile k is at column k mod this); for a 3D texture also the rows of a
    /// layer of tiles – else 0: rows follow each other, however many there are
    columns: u32,
    rows: u32,
    /// a first mip under a tile in some dimension is packed with the others, laid out as the
    /// driver likes: only a texture of these very sizes and mips is known to match
    packed: Option<(u64, u32, u16, u16)>,
}

impl Grid {
    pub(super) fn of(shape: &Shape) -> Grid {
        let (tw, th, td) = tile_elements(shape.element, shape.volume(), shape.samples);
        let depth = if shape.volume() { u32::from(shape.layers) } else { 1 };
        let packed = shape.width < u64::from(tw) || shape.height < th || depth < td;
        Grid {
            element: shape.element,
            samples: shape.samples,
            volume: shape.volume(),
            columns: shape.width.div_ceil(u64::from(tw)).min(u64::from(u32::MAX)) as u32,
            rows: if shape.volume() { shape.height.div_ceil(th) } else { 0 },
            packed: packed.then_some((shape.width, shape.height, shape.layers, shape.mips)),
        }
    }

    /// The grid of a texture as `GetResourceTiling` tells it: the tiles of its first mip (in a
    /// row, rows, layers), or None when the PC packs that mip with the others. That is what
    /// `vram` keys pages by; where the standard tile shapes (`of`, which the log's pairs go by)
    /// give another grid, it is logged (4 times).
    pub(super) fn of_tiling(shape: &Shape, first_mip: Option<[u32; 3]>) -> Grid {
        let standard = Grid::of(shape);
        let told = match first_mip {
            Some(tiles) => Grid { columns: tiles[0], rows: if shape.volume() { tiles[1] } else { 0 }, packed: None, ..standard },
            None => Grid { packed: Some((shape.width, shape.height, shape.layers, shape.mips)), ..standard },
        };
        if told != standard {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: texture alias: the PC's tiles of a first mip ({first_mip:?}; None: packed) are not the standard tile shapes' ({standard:?}) - the PC's are taken");
            }
        }
        told
    }
}

/// How a new texture of grid `new` on pages is keyed among the mapped ones that start at its
/// address (`peers`: grid, whether the GPU writes it, its own keying): 0 - the pages of the
/// title's addresses, shared with whatever covers them - or the class its tiles are kept apart
/// under: where they are in `vram::Pages::extra_pages`, added to its tile indices (which are
/// under 2^32). Textures of one grid at one address share those pages too: the class is the
/// one a texture of its grid there has (`taken`: the classes of the address, also of textures
/// that are unmapped and so no peers), else `fresh` - one `vram` gives no other grid.
///
/// A GPU-written peer of its grid is the same memory: its keying. Else, when a peer of another
/// grid holds the address's pages, apart: on those pages it would read that peer's tiles in
/// another order, and its uploads would land in them. A GPU-written peer holds them; a sampled
/// one gives way when a target takes its address (`title_textures::retire_overlapping`) and
/// holds them only when it `stays`: under a GPU-written texture no retire follows (one whose
/// size on the console is not known - a multisampled target).
pub(super) fn keying(new: Grid, peers: &[(Grid, bool, usize)], taken: &[(Grid, usize)], fresh: usize, stays: bool) -> usize {
    if let Some(&(_, _, apart)) = peers.iter().find(|p| p.0 == new && p.1) {
        return apart;
    }
    if peers.iter().any(|p| p.0 != new && p.2 == 0 && (p.1 || stays)) {
        return taken.iter().find(|t| t.0 == new).map_or(fresh, |t| t.1);
    }
    peers.iter().find(|p| p.0 == new).map_or(0, |p| p.2)
}

/// Where a placement's video memory is (`vram::memory`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Memory {
    /// tiles on pages: 0, the pages of its addresses, else the class it is kept apart under
    Pages(usize),
    /// in the arena of this id
    Arena(u64),
    /// a slot of a small-texture slab (one slot for placements of its address and size)
    Slab,
    /// a resource of its own
    Own,
}

impl Memory {
    /// Whether two placements at one address in these memories are the same video memory.
    fn same(self, other: Memory) -> bool {
        self == other && self != Memory::Own
    }
}

struct Placed {
    device: usize,
    va: usize,
    format: i32,
    size: (u64, u32, u16),
    mips: u16,
    samples: u32,
    flags: i32,
    shape: Option<Shape>,
    memory: Memory,
    /// the GPU writes it: a render target, depth buffer or UAV texture, or the title copied or
    /// resolved into it
    gpu: bool,
}

impl Placed {
    fn describe(&self) -> String {
        format!(
            "format {} {}x{}x{} {} mips {} samples flags {:#x}, {}, {}",
            self.format, self.size.0, self.size.1, self.size.2, self.mips, self.samples, self.flags,
            if self.gpu { "written by the GPU" } else { "sampled" },
            match self.memory {
                Memory::Pages(0) => "on the pages of its addresses".to_string(),
                Memory::Pages(_) => "on pages of its own".to_string(),
                Memory::Arena(id) => format!("in arena {id}"),
                Memory::Slab => "in a slab".to_string(),
                Memory::Own => "a resource of its own".to_string(),
            }
        )
    }
}

/// What a pair of live placements at one address is to the PC.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Pair {
    /// one layout, one memory: a view in another format, or of fewer rows, mips or slices
    Cast,
    /// one layout in elements, one memory: block-compressed and plain
    Recast,
    /// one layout, but not one memory (one in an arena, the other on pages or on its own)
    Split,
    /// two layouts, each its own memory
    Apart,
    /// two layouts in one memory: each reads the other's tiles in another order
    Mixed,
}

const PAIRS: [(Pair, &str); 5] = [
    (Pair::Cast, "of one layout in one memory"),
    (Pair::Recast, "block-compressed over plain in one memory"),
    (Pair::Split, "of one layout NOT in one memory"),
    (Pair::Apart, "of two layouts kept apart"),
    (Pair::Mixed, "of two layouts IN ONE MEMORY"),
];

/// What the pair is: by the shapes, the memories, and for two textures on the pages of their
/// addresses the tile grids (None: a shape is not known - not counted).
fn pair(a: (&Shape, Memory), b: (&Shape, Memory)) -> Pair {
    let same = a.1.same(b.1);
    let on_pages = matches!((a.1, b.1), (Memory::Pages(_), Memory::Pages(_)));
    let one_layout = if on_pages { Grid::of(a.0) == Grid::of(b.0) } else { a.0.shares(b.0) };
    match (one_layout, same) {
        (true, true) if a.0.block != b.0.block => Pair::Recast,
        (true, true) => Pair::Cast,
        (true, false) => Pair::Split,
        (false, false) => Pair::Apart,
        (false, true) => Pair::Mixed,
    }
}

#[derive(Default)]
struct Live {
    by_res: HashMap<usize, Placed>,
    /// (device, address) -> the live placements that start there
    at: HashMap<(usize, usize), Vec<usize>>,
    /// the pairs logged in full: what they are and both descriptions, whatever the address
    seen: HashSet<u64>,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// pairs logged in full (the distinct ones; the rest are counted)
const LOGGED: usize = 32;

/// The title placed texture `resource` (`desc`: the PC's) at `va` of `device`: every pair it
/// makes with a live placement that starts there is counted, and one that is not a plain cast is
/// logged (the first `LOGGED` distinct ones in full, then at each power of two of its kind).
/// Called after `title_textures::retire_overlapping`, when one follows the placement: a sampled
/// texture is said to give way to a target when it is unmapped by then.
pub(super) fn placed(device: usize, resource: usize, va: usize, desc: &D3D12_RESOURCE_DESC) {
    for line in note(device, resource, va, desc, super::vram::memory(resource), super::vram::unmapped) {
        crate::xlog_info!("{line}");
    }
}

/// The lines to log (`gave_way`: whether a texture is unmapped - asked without the table's lock).
fn note(device: usize, resource: usize, va: usize, desc: &D3D12_RESOURCE_DESC, memory: Memory, gave_way: impl Fn(usize) -> bool) -> Vec<String> {
    let new = Placed {
        device,
        va,
        format: desc.Format.0,
        size: (desc.Width, desc.Height, desc.DepthOrArraySize),
        mips: desc.MipLevels,
        samples: desc.SampleDesc.Count,
        flags: desc.Flags.0,
        shape: Shape::of(desc),
        memory,
        gpu: super::vram::gpu_owned(desc.Flags.0 as u32),
    };
    let mut lines = Vec::new();
    {
        let mut g = LIVE.lock().unwrap();
        let live = g.get_or_insert_with(Live::default);
        forget_in(live, resource);
        let Live { by_res, at, seen } = live;
        let peers = at.entry((device, va)).or_default();
        for (&old_resource, old) in peers.iter().filter_map(|r| by_res.get(r).map(|old| (r, old))) {
            let (Some(a), Some(b)) = (&new.shape, &old.shape) else { continue };
            let what = pair((a, new.memory), (b, old.memory));
            let n = COUNTS[what as usize].fetch_add(1, Ordering::Relaxed) + 1;
            let text = PAIRS[what as usize].1;
            if what != Pair::Cast && seen.len() < LOGGED && seen.insert(hash(&(what as usize, a, b, new.format, old.format, new.gpu, old.gpu))) {
                let rows = if a.same_rows(b) { "; the same bytes a row and rows" } else { "" };
                let retired = (what == Pair::Mixed && new.gpu && !old.gpu).then_some(old_resource);
                lines.push((format!("D3D12.X: texture alias at {va:#x}: a pair {text}{rows}"), retired, format!(": new [{}] over [{}]", new.describe(), old.describe())));
            } else if what != Pair::Cast && n.is_power_of_two() {
                lines.push((format!("D3D12.X: texture alias: pairs {text}: {n} so far"), None, String::new()));
            }
        }
        peers.push(resource);
        by_res.insert(resource, new);
    }
    lines.into_iter().map(|(head, retired, tail)| {
        let retired = if retired.is_some_and(&gave_way) { "; the sampled one gives way to the target" } else { "" };
        format!("{head}{retired}{tail}")
    }).collect()
}

fn forget_in(live: &mut Live, resource: usize) {
    let Some(old) = live.by_res.remove(&resource) else { return };
    if let Some(peers) = live.at.get_mut(&(old.device, old.va)) {
        peers.retain(|&r| r != resource);
        if peers.is_empty() {
            live.at.remove(&(old.device, old.va));
        }
    }
}

/// `resource` was released for good.
pub(super) fn forget(resource: usize) {
    if let Some(live) = LIVE.lock().unwrap().as_mut() {
        forget_in(live, resource);
    }
}

/// (`dsv_trace`) The live placements that start at `va` on any device: (resource, PC format,
/// size, mips, flags).
pub(super) fn placed_at(va: usize) -> Vec<(usize, i32, (u64, u32, u16), u16, i32)> {
    let g = LIVE.lock().unwrap();
    let Some(live) = g.as_ref() else { return Vec::new() };
    live.at.iter().filter(|(k, _)| k.1 == va).flat_map(|(_, rs)| rs.iter())
        .filter_map(|r| live.by_res.get(r).map(|p| (*r, p.format, p.size, p.mips, p.flags)))
        .collect()
}

/// The title's GPU work wrote `resource` (a copy or resolve into it).
pub(super) fn note_gpu_write(resource: usize) {
    if let Some(p) = LIVE.lock().unwrap().as_mut().and_then(|live| live.by_res.get_mut(&resource)) {
        p.gpu = true;
    }
}

/// (`e5_copy`) The newest live placement at `resource`'s address, of its size, mips and samples,
/// that takes unordered access and has the PC format `format`.
pub(super) fn writer_at(resource: usize, format: i32) -> Option<usize> {
    let g = LIVE.lock().unwrap();
    let live = g.as_ref()?;
    let p = live.by_res.get(&resource)?;
    live.at.get(&(p.device, p.va))?.iter().rev().copied().find(|&r| {
        r != resource && live.by_res.get(&r).is_some_and(|o| {
            o.format == format && o.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0 && (o.size, o.mips, o.samples) == (p.size, p.mips, p.samples)
        })
    })
}

/// The pairs so far and what is kept apart now, for the activity lines (None: no placement has
/// met another at its address).
pub fn summary() -> Option<String> {
    let n: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if n.iter().all(|&c| c == 0) {
        return None;
    }
    let (shared, placements) = LIVE.lock().unwrap().as_ref().map_or((0, 0), |l| {
        let shared = l.at.values().filter(|v| v.len() > 1);
        (shared.clone().count(), shared.map(Vec::len).sum::<usize>())
    });
    let (apart, apart_bytes) = super::vram::apart_bytes();
    let over = OVER.load(Ordering::Relaxed);
    let pairs: Vec<String> = PAIRS.iter().map(|&(p, text)| format!("{} {text}", n[p as usize])).collect();
    Some(format!(
        "texture aliases: {placements} placements at {shared} shared addresses now; pairs so far: {}; {apart} textures on pages of their own now ({} MB), {} MB of arenas of their own; budget {} MB{}{}",
        pairs.join(", "),
        apart_bytes >> 20,
        super::vram::split_arena_bytes() >> 20,
        budget() >> 20,
        if over == 0 { String::new() } else { format!(", reached: {over} textures shared by address") },
        if enabled() { "" } else { "; XCL_TEXTURE_ALIAS=0: nothing is kept apart" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn desc(format: DXGI_FORMAT, w: u64, h: u32, layers: u16, mips: u16) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w,
            Height: h,
            DepthOrArraySize: layers,
            MipLevels: mips,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        }
    }

    fn shape(format: DXGI_FORMAT, w: u64, h: u32) -> Shape {
        Shape::of(&desc(format, w, h, 1, 1)).unwrap()
    }

    /// (the table of live placements and its counts are the process's: one test at a time)
    static TABLE: Mutex<()> = Mutex::new(());

    /// A shape is in elements: formats of one element size are one layout, a compressed
    /// texture is the layout of a plain one of its blocks, and anything else is not.
    #[test]
    fn shapes_tell_layouts_apart_not_formats() {
        let rgba8 = shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720);
        // a view in another format of the element: one memory
        for f in [DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_D24_UNORM_S8_UINT] {
            assert!(rgba8.shares(&shape(f, 1280, 720)) && rgba8.key() == shape(f, 1280, 720).key(), "{f:?}");
        }
        // BC3 written through a 16-byte integer texture of its blocks; BC1 through an 8-byte one
        let bc3 = shape(DXGI_FORMAT_BC3_UNORM, 1024, 512);
        let blocks = shape(DXGI_FORMAT_R32G32B32A32_UINT, 256, 128);
        assert!(bc3.shares(&blocks) && bc3.key() == blocks.key() && (bc3.block, blocks.block) == (4, 1));
        assert!(shape(DXGI_FORMAT_BC1_UNORM, 1024, 512).shares(&shape(DXGI_FORMAT_R16G16B16A16_UINT, 256, 128)));
        // (a compressed texture's odd sizes round up to blocks)
        assert!(shape(DXGI_FORMAT_BC1_UNORM, 1021, 510).shares(&shape(DXGI_FORMAT_R32G32_UINT, 256, 128)));
        assert!(!bc3.shares(&shape(DXGI_FORMAT_R16G16B16A16_UINT, 256, 128)));
        // the reported picture: a texture twice as wide over a target, and half the size over it
        for (w, h) in [(640, 720), (640, 360), (2560, 720)] {
            let other = shape(DXGI_FORMAT_R8G8B8A8_UNORM, w, h);
            assert!(!rgba8.shares(&other) && rgba8.key() != other.key() && !rgba8.same_rows(&other), "{w}x{h}");
        }
        // the same bytes a row and rows in other elements: no layout in common, and said so
        assert!(rgba8.same_rows(&shape(DXGI_FORMAT_R16_UNORM, 2560, 720)) && !rgba8.shares(&shape(DXGI_FORMAT_R16_UNORM, 2560, 720)));
        assert!(rgba8.same_rows(&shape(DXGI_FORMAT_R16G16B16A16_FLOAT, 640, 720)));
        assert!(!rgba8.same_rows(&shape(DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720)) && !rgba8.shares(&shape(DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720)));
        // rows, mips and slices follow each other: fewer of them are the start of more, as
        // before (a target on the first face of a cube, a view of the first mips)
        let mipped = |layers, mips| Shape::of(&desc(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720, layers, mips)).unwrap();
        assert_eq!(mipped(1, 0).mips, 11);
        for other in [shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 360), mipped(1, 5), mipped(6, 1), mipped(6, 4)] {
            assert!(rgba8.shares(&other) && rgba8.key() == other.key(), "{other:?}");
        }
        // samples and the dimension are layout; a 3D texture's layers follow its rows
        let mut msaa = desc(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720, 1, 1);
        msaa.SampleDesc.Count = 4;
        assert!(!rgba8.shares(&Shape::of(&msaa).unwrap()));
        let volume = |h, d| {
            let mut volume = desc(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, h, d, 1);
            volume.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            Shape::of(&volume).unwrap()
        };
        assert!(!rgba8.shares(&volume(720, 1)) && volume(720, 64).shares(&volume(720, 16)) && !volume(720, 64).shares(&volume(360, 64)));
        // buffers and formats without a known element have no shape
        let mut buffer = desc(DXGI_FORMAT_UNKNOWN, 65536, 1, 1, 1);
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        assert!(Shape::of(&buffer).is_none() && Shape::of(&desc(DXGI_FORMAT_NV12, 64, 64, 1, 1)).is_none());
    }

    /// A console R9G9B9E5 target made castable R11G11B10_FLOAT (`castable`) is one layout with
    /// the R32_UINT and R9G9B9E5 textures a title places at its address; as R16G16B16A16_FLOAT
    /// it was not.
    #[test]
    fn a_castable_rgb9e5_target_shares_with_its_32_bit_placements() {
        let target = shape(DXGI_FORMAT_R11G11B10_FLOAT, 1920, 1080);
        for f in [DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP] {
            assert!(target.shares(&shape(f, 1920, 1080)) && target.key() == shape(f, 1920, 1080).key(), "{f:?}");
        }
        assert!(!shape(DXGI_FORMAT_R16G16B16A16_FLOAT, 1920, 1080).shares(&shape(DXGI_FORMAT_R32_UINT, 1920, 1080)));
        assert!(shape(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 1280, 720).shares(&shape(DXGI_FORMAT_R32_UINT, 1280, 720)));
    }

    /// The standard 64 KiB tiles: 65 536 bytes of elements each, a block-compressed format's
    /// in blocks, a multisampled one's in samples.
    #[test]
    fn standard_tiles_hold_64_kib() {
        for element in [1u32, 2, 4, 8, 16] {
            for volume in [false, true] {
                let (w, h, d) = tile_elements(element, volume, 1);
                assert_eq!(w * h * d * element, 0x1_0000, "{element} bytes, 3D {volume}");
            }
            for samples in [2u32, 4, 8, 16] {
                let (w, h, _) = tile_elements(element, false, samples);
                assert_eq!(w * h * samples * element, 0x1_0000, "{element} bytes, {samples} samples");
            }
        }
        // D3D12's table in texels: 32 bits 128x128, 64 bits 128x64, BC1 512x256, BC3 256x256
        assert_eq!(tile_elements(4, false, 1), (128, 128, 1));
        assert_eq!(tile_elements(8, false, 1), (128, 64, 1));
        assert_eq!(Grid::of(&shape(DXGI_FORMAT_BC1_UNORM, 512, 256)).columns, 1);
        assert_eq!(Grid::of(&shape(DXGI_FORMAT_BC3_UNORM, 512, 256)).columns, 2);
        assert_eq!(tile_elements(4, true, 1), (32, 32, 16));
    }

    /// Where tile k of a texture is, and so which texels of a W-wide writer a 2W-wide reader
    /// over its pages shows: the writer's tile rows 2j and 2j + 1 side by side.
    #[test]
    fn a_wider_grid_over_the_same_pages_shows_the_picture_twice() {
        let writer = Grid::of(&shape(DXGI_FORMAT_R11G11B10_FLOAT, 640, 360));
        let reader = Grid::of(&shape(DXGI_FORMAT_R11G11B10_FLOAT, 1280, 720));
        assert_eq!((writer.columns, reader.columns), (5, 10));
        assert_ne!(writer, reader);
        // page k of the address is tile k of both: (column, row) of it in each
        let at = |g: &Grid, k: u32| (k % g.columns, k / g.columns);
        for row in 0..2 {
            for column in 0..10 {
                let k = row * 10 + column;
                let (x, y) = at(&writer, k);
                assert_eq!(at(&reader, k), (column, row));
                // the reader's left half is the writer's even tile row, its right half the odd one
                assert_eq!((x, y), (column % 5, 2 * row + column / 5));
            }
        }
        // widths inside the last tile are one grid, and so are fewer rows: tile k is at the same
        // texels of both
        let small = |w, h| Grid::of(&shape(DXGI_FORMAT_R8G8B8A8_UNORM, w, h));
        assert_eq!(small(1280, 720), Grid::of(&shape(DXGI_FORMAT_B8G8R8A8_UNORM, 1264, 711)));
        assert_eq!(small(1280, 720), small(1280, 360));
        assert_ne!(small(1280, 720), small(1152, 720));
        assert_eq!(Grid::of(&shape(DXGI_FORMAT_BC3_UNORM, 1024, 512)), Grid::of(&shape(DXGI_FORMAT_R32G32B32A32_UINT, 256, 128)));
        // other elements are other tiles even where the tiles in a row agree
        assert_ne!(small(1280, 768), Grid::of(&shape(DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 384)));
        // a first mip under a tile is packed: only the very same sizes are one grid
        assert!(small(1280, 64).packed.is_some() && small(1280, 128).packed.is_none());
        assert_eq!(small(96, 96), small(96, 96));
        assert_ne!(small(96, 96), small(100, 96));
        // mips and slices follow the first mip's tiles: they are not the grid
        let mipped = |layers, mips| Grid::of(&Shape::of(&desc(DXGI_FORMAT_R8G8B8A8_UNORM, 1024, 1024, layers, mips)).unwrap());
        assert!(mipped(1, 1) == mipped(1, 11) && mipped(6, 1) == mipped(6, 11) && mipped(1, 1) == mipped(6, 1));
        // what the driver tells is taken: its tiles of the first mip, or that it packs it
        let s = shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720);
        assert_eq!(Grid::of_tiling(&s, Some([10, 6, 1])), small(1280, 720));
        assert_eq!(Grid::of_tiling(&s, Some([5, 12, 1])).columns, 5);
        assert_ne!(Grid::of_tiling(&s, None), Grid::of_tiling(&shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1264, 711), None));
        assert_eq!(Grid::of_tiling(&shape(DXGI_FORMAT_R8G8B8A8_UNORM, 96, 96), None), small(96, 96));
    }

    /// A new texture shares the pages of its address unless a GPU-written texture of another
    /// grid holds them; textures of one grid are one memory wherever that is.
    #[test]
    fn a_texture_of_another_grid_is_kept_apart_from_the_target_at_its_address() {
        let g = |w, h| Grid::of(&shape(DXGI_FORMAT_R8G8B8A8_UNORM, w, h));
        let (half, full, other) = (g(640, 360), g(1280, 720), g(1920, 1080));
        // (the class `vram` has for a grid that has none at the address yet)
        let fresh = 5usize << 32;
        let key = |new, peers: &[(Grid, bool, usize)], taken: &[(Grid, usize)]| keying(new, peers, taken, fresh, false);
        // alone, or over textures of its grid: the address's pages
        assert_eq!(key(full, &[], &[]), 0);
        assert_eq!(key(full, &[(full, true, 0)], &[]), 0);
        assert_eq!(key(full, &[(full, false, 0)], &[]), 0);
        // over a GPU-written target of another grid: apart, under the class it is given
        let apart = key(full, &[(half, true, 0)], &[]);
        assert_eq!(apart, fresh);
        // a sampled texture of another grid does not hold the pages: it gives way
        assert_eq!(key(full, &[(half, false, 0)], &[]), 0);
        // the next of its grid joins it, GPU-written or sampled, also beside the first target
        assert_eq!(key(full, &[(half, true, 0), (full, true, apart)], &[(full, apart)]), apart);
        assert_eq!(key(full, &[(half, true, 0), (full, false, apart)], &[(full, apart)]), apart);
        // the class of its grid at the address is its also while the textures under it are
        // unmapped (no peers): one of them mapped again is then on its pages, not another grid's
        let old = 3usize << 32;
        assert_eq!(key(full, &[(half, true, 0)], &[(full, old)]), old);
        // a third grid is apart from both: the class it is given, never another grid's
        assert_eq!(keying(other, &[(half, true, 0), (full, true, apart)], &[(full, apart)], 6 << 32, false), 6 << 32);
        // the first target gone: one of its grid takes the address's pages again, not the
        // apart texture's; one of the apart texture's grid stays with it
        assert_eq!(key(half, &[(full, true, apart)], &[(full, apart)]), 0);
        assert_eq!(key(full, &[(full, true, apart)], &[(full, apart)]), apart);
        // a sampled texture of its grid on the address's pages beside a target of another grid
        // (it read that target's tiles): the new target does not join it there
        assert_eq!(key(full, &[(half, true, 0), (full, false, 0)], &[]), apart);
        // a GPU-written texture no retire follows (a multisampled target): the sampled texture
        // of another grid stays mapped under it, so it is apart from that one too - and shares
        // with one of its own grid as ever
        assert_eq!(keying(full, &[(half, false, 0)], &[], fresh, true), fresh);
        assert_eq!(keying(full, &[(full, false, 0)], &[], fresh, true), 0);
        assert_eq!(keying(full, &[(half, false, 0), (full, false, old)], &[(full, old)], fresh, true), old);
    }

    /// Memory of their own is within the budget: what fits is taken, what does not shares by
    /// address, and a texture that takes nothing more always fits.
    #[test]
    fn memory_of_their_own_is_within_the_budget() {
        let mb = 1usize << 20;
        assert!(fits(0, 256 * mb, 256 * mb) && fits(200 * mb, 56 * mb, 256 * mb));
        assert!(!fits(200 * mb, 57 * mb, 256 * mb) && !fits(0, 1, 0));
        // (a texture that joins the pages of its class takes nothing more)
        assert!(fits(300 * mb, 0, 256 * mb) && fits(usize::MAX, 0, 0));
        assert!(!fits(usize::MAX, 1, 256 * mb), "no overflow");
        assert_eq!(budget() % mb, 0);
    }

    /// A sampled texture under a new target is said to give way only when it was unmapped: a
    /// multisampled target has no size on the console, no retire follows it, and the sampled
    /// texture under it stays.
    #[test]
    fn a_sampled_texture_is_said_to_give_way_only_when_it_is_unmapped() {
        let _table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let (device, va) = (0xA11A_6000usize, 0x7A12_0000_0000usize);
        let over = |sampled: usize, format: DXGI_FORMAT, samples: u32, gave_way: bool| {
            let mut target = desc(format, 2048, 512, 1, 1);
            target.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            target.SampleDesc.Count = samples;
            assert!(note(device, sampled, va + sampled, &desc(format, 1024, 512, 1, 1), Memory::Pages(0), |_| true).is_empty());
            let lines = note(device, sampled + 1, va + sampled, &target, Memory::Pages(0), |r| gave_way && r == sampled);
            forget(sampled);
            forget(sampled + 1);
            lines
        };
        let stays = over(0xA11A_0010, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, false);
        assert!(stays.len() == 1 && stays[0].contains("two layouts IN ONE MEMORY") && !stays[0].contains("gives way"), "{stays:?}");
        let retired = over(0xA11A_0020, DXGI_FORMAT_R16G16_FLOAT, 1, true);
        assert!(retired.len() == 1 && retired[0].contains("; the sampled one gives way to the target: new ["), "{retired:?}");
    }

    /// What a pair is to the PC: by layout and by memory.
    #[test]
    fn pairs_are_named_by_layout_and_memory() {
        let rgba = shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720);
        let srgb = shape(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 1280, 720);
        let narrow = shape(DXGI_FORMAT_R8G8B8A8_UNORM, 1264, 711);
        let half = shape(DXGI_FORMAT_R8G8B8A8_UNORM, 640, 360);
        let bc3 = shape(DXGI_FORMAT_BC3_UNORM, 1024, 512);
        let blocks = shape(DXGI_FORMAT_R32G32B32A32_UINT, 256, 128);
        let (pages, own) = (Memory::Pages(0), Memory::Pages(7 << 32));
        assert_eq!(pair((&rgba, pages), (&srgb, pages)), Pair::Cast);
        assert_eq!(pair((&bc3, pages), (&blocks, pages)), Pair::Recast);
        // on pages the grid decides: widths inside the last tile are one layout, in a heap not
        assert_eq!(pair((&rgba, pages), (&narrow, pages)), Pair::Cast);
        assert_eq!(pair((&rgba, Memory::Arena(3)), (&narrow, Memory::Arena(3))), Pair::Mixed);
        assert_eq!(pair((&rgba, Memory::Arena(3)), (&narrow, Memory::Arena(4))), Pair::Apart);
        // two grids: in one memory (as before, or XCL_TEXTURE_ALIAS=0), or kept apart
        assert_eq!(pair((&rgba, pages), (&half, pages)), Pair::Mixed);
        assert_eq!(pair((&rgba, own), (&half, pages)), Pair::Apart);
        // one layout in two memories: a target made UNDEFINED in an arena, its view on pages
        assert_eq!(pair((&rgba, Memory::Arena(3)), (&srgb, pages)), Pair::Split);
        assert_eq!(pair((&rgba, Memory::Own), (&srgb, Memory::Own)), Pair::Split);
        assert_eq!(pair((&rgba, Memory::Slab), (&srgb, Memory::Slab)), Pair::Cast);
    }

    /// Placements are paired with the live ones at their address and device, and no longer once
    /// they are released.
    #[test]
    fn live_placements_are_paired_at_their_address() {
        let _table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
        // (addresses and resources no other test uses: the table is the process's)
        let (device, va) = (0xA11A_5000usize, 0x7A11_0000_0000usize);
        let count = |p: Pair| COUNTS[p as usize].load(Ordering::Relaxed);
        let before = [Pair::Cast, Pair::Mixed, Pair::Apart, Pair::Split].map(count);
        let rgba = desc(DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720, 1, 1);
        let mut target = rgba;
        target.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        note(device, 0xA11A_0001, va, &target, Memory::Pages(0), |_| false);
        note(device, 0xA11A_0002, va, &desc(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 1280, 720, 1, 1), Memory::Pages(0), |_| false);
        note(device, 0xA11A_0003, va, &desc(DXGI_FORMAT_R8G8B8A8_UNORM, 640, 360, 1, 1), Memory::Pages(0), |_| false);
        // another address, another device: no pair
        note(device, 0xA11A_0004, va + 0x1_0000, &rgba, Memory::Pages(0), |_| false);
        note(device + 1, 0xA11A_0005, va, &rgba, Memory::Pages(0), |_| false);
        let after = [Pair::Cast, Pair::Mixed, Pair::Apart, Pair::Split].map(count);
        assert_eq!([after[0] - before[0], after[1] - before[1]], [1, 2], "the cast, and the half-size texture against both");
        {
            let g = LIVE.lock().unwrap();
            let live = g.as_ref().unwrap();
            assert_eq!(live.at[&(device, va)], [0xA11A_0001, 0xA11A_0002, 0xA11A_0003]);
            assert!(live.by_res[&0xA11A_0001].gpu && !live.by_res[&0xA11A_0003].gpu);
        }
        note_gpu_write(0xA11A_0003);
        assert!(LIVE.lock().unwrap().as_ref().unwrap().by_res[&0xA11A_0003].gpu);
        assert!(summary().is_some_and(|s| s.contains("pairs so far")));
        // released: the next placement there pairs with what is left
        forget(0xA11A_0001);
        forget(0xA11A_0002);
        note(device, 0xA11A_0006, va, &rgba, Memory::Arena(9), |_| false);
        assert_eq!(count(Pair::Apart) - before[2], 1);
        // the same resource placed again (its address reused): its old entry goes
        note(device, 0xA11A_0006, va + 0x2_0000, &rgba, Memory::Arena(9), |_| false);
        assert_eq!(LIVE.lock().unwrap().as_ref().unwrap().at[&(device, va)], [0xA11A_0003]);
        for r in [0xA11A_0003usize, 0xA11A_0004, 0xA11A_0005, 0xA11A_0006] {
            forget(r);
        }
        let g = LIVE.lock().unwrap();
        let live = g.as_ref().unwrap();
        assert!(!live.at.contains_key(&(device, va)) && !live.by_res.contains_key(&0xA11A_0006));
        assert_eq!(count(Pair::Split) - before[3], 0);
    }
}
