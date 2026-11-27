//! Console descriptor heaps (CBV/SRV/UAV and samplers).
//!
//! On the console a descriptor is GPU-readable memory: 32 bytes per CBV/SRV/UAV slot (T# /
//! V#), 16 per sampler (S#), and a descriptor table in a root argument is a 32-bit GPU address
//! (shaders load it with the high half 0 – amd_fidelityfx_upscaler_xbox.dll, HANDOFF "FSR4").
//! XCL gives console code heaps whose slots live in unified memory (so the table addresses
//! console code computes from the heap start work as they are in the recompiler's `mem`
//! model) and keeps PC descriptors beside each: view slot k → an SRV and, next to it, a UAV;
//! sampler slot k → PC sampler k. A slot holds the descriptor in the XCL shader ABI
//! (`desc::xcl_image_words`: T# word 0 = the SRV's index in the shader-visible PC heap;
//! `desc::xcl_sampler_words`: S# word 3 [11:0] = k; V# with its unified-memory address).
//!
//! View descriptors are values on the console: titles create them in a CPU-only heap and copy
//! them into the shader-visible one as memory (the console header's inline
//! CopyDescriptorsSimple; XDefiant copies every texture that way), and a descriptor table may
//! be in any heap. So all console view heaps share one shader-visible PC heap (`Global`):
//! each gets a range there, T# word 0 is an index into it – a T# copied anywhere still names
//! its PC descriptor – and that heap is what lists bind. The title's own PC heap stays as a
//! CPU-only mirror (where views are made, copies come from and UAV clears take their CPU
//! handle). A heap gets its place in the shared heap a page of `PAGE_SLOTS` slots at a time,
//! when a view with PC descriptors is first made in or copied to a slot of that page (`slot`;
//! a slot that is read, or gets a view shaders read from memory, takes none: `lookup`): titles
//! make large heaps and fill few slots
//! (Minecraft: eight view heaps of 73 727 slots – 147 454 PC descriptors each, more in all
//! than the 1 000 000 an NVIDIA shared heap holds – with a few hundred views in them). A heap
//! left out of the shared heap would be lost: lists bind only the shared heap for views.
//! Without the shared heap (it could not be made) a heap keeps its own PC heap: SRV at 2k,
//! UAV at 2k + 1.

pub(crate) mod vsharp;
use vsharp::Writer;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

use super::material_gate_views::{self, Key, NativeView};

/// bytes of a CBV/SRV/UAV slot (T# / V#) and of a sampler slot (S#)
pub const SLOT: u32 = 32;
pub const SAMPLER_SLOT: u32 = 16;

/// Whose memory a heap's slots are in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Memory {
    /// a sparse VRAM allocation in window 0
    Sparse,
    /// committed system RAM
    Uma,
    /// the title's (its allocation hook gave it: `kernelx::runtime_alloc`)
    Title,
}

struct Heap {
    stamp: u64,
    obj: usize,
    shadow: usize,
    n: u32,
    pc_cpu: u64,
    /// 0: not shader visible
    pc_gpu: u64,
    inc: u32,
    sampler: bool,
    /// views in the shared heap: the index there of each page's first SRV (`UNMAPPED`: the
    /// page has no place yet); empty: the heap's views are not in the shared heap
    pages: Vec<u32>,
    /// for each page of `pages`, the frame (present count) a slot of it was last used in
    /// (`slot`, `lookup`, `table_bound`)
    used: Vec<u32>,
    /// the `EVICTED` pages: what their image slots named (`evict_words`)
    evicted: std::collections::HashMap<u32, Named>,
    /// (frame, page): in that frame a list bound a table taken to reach the heap's end from
    /// that page (`table_pages`), and the pages from there on are marked as used in it
    tail: Option<(u32, u32)>,
    /// shader visible as the title made it
    visible: bool,
    memory: Memory,
}

impl Heap {
    fn slot_bytes(&self) -> u32 {
        if self.sampler {
            SAMPLER_SLOT
        } else {
            SLOT
        }
    }
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// The shader-visible PC heap of all console view heaps.
struct Global {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    gpu: u64,
    inc: u32,
    cap: u32,
    /// the single descriptors' region (`reserve_single`), after the null views; empty: there is none
    singles: Region,
    /// the rest of the heap: the title heaps' pages and XCL's own ranges
    ranges: Region,
    /// descriptors held now, by `Holder`
    held: [u32; HOLDERS],
}

// D3D12 descriptor heaps and devices are free-threaded
unsafe impl Send for Global {}

static GLOBAL: Mutex<Option<Global>> = Mutex::new(None);
static GLOBAL_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// descriptors of the shared heap at most (the GPU may take more than it reports: `larger_heap`)
const GLOBAL_MAX: u32 = 1 << 21;

/// Who holds descriptors of the shared heap (`usage_line` counts each).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Holder {
    /// pages of the titles' view heaps (`page_of`)
    TitlePages,
    /// XCL's own ranges (`reserve_own`): the memory windows' raw views
    Windows,
    /// root versions of the physical directory
    Roots,
    /// its leaves
    Leaves,
    /// its recordings' locators
    Locators,
    /// its bound physical pages
    Pages,
}
const HOLDERS: usize = 6;

/// A part of the shared heap, `[first, end)`, with its own free ranges.
struct Region {
    first: u32,
    end: u32,
    next: u32,
    /// Free ranges (first, count), ordered by start and separated by live descriptors.
    free: Vec<(u32, u32)>,
    /// descriptors given out
    used: u32,
}

/// (first, end, descriptors given out, free ranges, the largest free range) of a region
type RegionUsage = (u32, u32, u32, usize, u32);

impl Region {
    fn new(first: u32, end: u32) -> Self {
        Self { first, end, next: first, free: Vec::new(), used: 0 }
    }

    fn take(&mut self, count: u32) -> Option<u32> {
        let first = take_range(&mut self.free, &mut self.next, self.end, count)?;
        self.used += count;
        Some(first)
    }

    fn give(&mut self, first: u32, count: u32) {
        return_range(&mut self.free, &mut self.next, first, count);
        self.used = self.used.saturating_sub(count);
    }

    /// more than three quarters of it are given out
    fn crowded(&self) -> bool {
        self.used as u64 * 4 > (self.end - self.first) as u64 * 3
    }

    /// The room never given out (after `next`) counts as one free range.
    fn usage(&self) -> RegionUsage {
        let rest = self.end - self.next;
        (self.first, self.end, self.used, self.free.len() + (rest > 0) as usize, self.free.iter().map(|&(_, n)| n).max().unwrap_or(0).max(rest))
    }
}

/// single descriptors kept for the physical directory's roots, leaves and locators, beside one
/// for each 64 KiB page of the title's memory
const SINGLES_VIEWS: u32 = 16_384;
/// single descriptors are named in 20 bits (`physical_directory`: a page table entry, a view)
const SINGLES_END: u32 = 1 << 20;

/// Descriptors of a shared heap of `cap` kept as the single descriptors' region
/// (`XCL_DESCRIPTOR_SINGLES`: that many, 0: no region), for a title with `title_mb` MiB of
/// memory: one for each 64 KiB physical page – the most the physical directory can bind – and
/// `SINGLES_VIEWS`, a quarter of the heap at most and all below `SINGLES_END`. The title heaps'
/// pages get none of these, so they cannot leave the directory without a descriptor, and single
/// descriptors do not land in a page's released range (512: a single one there keeps it from
/// being given whole again).
fn singles_region(cap: u32, title_mb: u64, setting: Option<&str>) -> u32 {
    let want = match setting.and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(n) => n,
        None => (title_mb.saturating_mul(16).min(u32::MAX as u64) as u32).saturating_add(SINGLES_VIEWS),
    };
    want.min(cap / 4).min(SINGLES_END - NULL_IMAGE_VIEWS)
}

/// A shared heap of `GLOBAL_MAX` is tried where the device reports a smaller most
/// (`XCL_DESCRIPTOR_HEAP_LARGE=0`: only the size it reports).
fn larger_heap() -> bool {
    std::env::var("XCL_DESCRIPTOR_HEAP_LARGE").map_or(true, |v| v != "0")
}

/// view slots per page of a heap's place in the shared heap (2 PC descriptors each)
pub const PAGE_SLOTS: u32 = 256;
/// a page with no place in the shared heap yet
const UNMAPPED: u32 = u32::MAX;
/// a page with no place in the shared heap whose slots were looked up (`lookup`): one the
/// lookup would have given a place before, counted by the usage line
const READ: u32 = u32::MAX - 1;
/// a page whose place in the shared heap another page took (`evict`) while its image slots
/// named its descriptors there: it gets a place again when one of its slots is used
const EVICTED: u32 = u32::MAX - 2;

/// Whether a page's entry is its place in the shared heap.
fn placed(first: u32) -> bool {
    first < EVICTED
}

/// When the shared heap has no room for a page of a title's view heap, the page takes the place
/// of the least recently used page of a shader-visible heap (`evict`;
/// `XCL_DESCRIPTOR_EVICT=0`: it gets none and its slots name the null views, `unplaced`).
fn evicting() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DESCRIPTOR_EVICT").map_or(true, |v| v != "0"))
}

/// Frames a page must have been out of use before another takes its place
/// (`XCL_DESCRIPTOR_EVICT_IDLE_FRAMES`). The GPU reads a page's descriptors when it runs a
/// list that binds a table over it, some frames after the list was recorded: as many as the
/// title lets the GPU fall behind. A title waits for a frame's lists before it records into
/// their allocators again and keeps two to four frames of those; 30 is ten times that, a second
/// at 30 Hz. No fence proves it: a list run later than that reads the views of the page that
/// got the place.
const EVICT_IDLE_FRAMES: u32 = 30;

fn evict_idle() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_DESCRIPTOR_EVICT_IDLE_FRAMES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(EVICT_IDLE_FRAMES))
}

/// the frame + 1 from which lists' descriptor tables are followed (`table_bound`; 0: not yet):
/// from when the ranges are three quarters given out, so that the pages lists bind are known
/// as used before the first eviction – no page goes before they were followed for the frames a
/// page must be out of use (`evict`)
static TABLES_FOLLOWED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// pages that lost their place (`evict`), and evicted pages that got one again (`page_of`)
static EVICTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PLACED_AGAIN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether lists' descriptor tables are followed (`table_bound`).
pub fn tables_followed() -> bool {
    TABLES_FOLLOWED.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// A table of more pages than this is taken to reach its heap's end (`table_pages`).
const WIDE_TABLE: u32 = 16;

/// The pages of a heap of `n` slots that a descriptor table of `slots` slots at slot `k`
/// reaches, and whether it is a wide one: a table of more than `WIDE_TABLE` pages – one with
/// a range without end, mostly – is taken to reach the heap's end, which `table_bound` marks
/// once a frame.
fn table_pages(n: u32, k: u32, slots: u32) -> (std::ops::Range<u32>, bool) {
    let first = k / PAGE_SLOTS;
    let end = k.saturating_add(slots.max(1)).min(n).div_ceil(PAGE_SLOTS).max(first + 1);
    let wide = end - first > WIDE_TABLE;
    (first..if wide { n.div_ceil(PAGE_SLOTS).max(end) } else { end }, wide)
}

/// The page to take the place of: of `pages` – each (heap, page, its descriptors, the frame it
/// was last used in) – the one longest out of use at frame `now` that has `need` descriptors
/// and was not used in the last `idle` frames. None: there is no such page.
fn victim(pages: impl Iterator<Item = (usize, usize, u32, u32)>, need: u32, now: u32, idle: u32) -> Option<(usize, usize)> {
    pages
        .filter(|&(_, _, count, used)| count >= need && now.wrapping_sub(used) >= idle)
        .max_by_key(|&(_, _, _, used)| now.wrapping_sub(used))
        .map(|(heap, page, _, _)| (heap, page))
}

/// What the image slots of an evicted page named: (slot in the page, the PC index it named less
/// the page's first, a hash of the slot's other words).
type Named = Vec<(u16, u16, u32)>;

/// A hash of words 1.. of a slot (FNV-1a): whether the slot still holds the T# it held.
fn rest_hash(w: &[u32; 8]) -> u32 {
    w[1..].iter().fold(0x811C_9DC5u32, |h, &x| (h ^ x).wrapping_mul(0x0100_0193))
}

/// The page whose slots are `slots` loses its place at `first` (`count` descriptors): every
/// image T# there that names one of those descriptors names the null views from now on (word 0
/// = 0: a shader reads zeros, its writes go nowhere), not the views of the page that gets the
/// place. What they named is returned, for `restore_words`.
fn evict_words(slots: &mut [[u32; 8]], first: u32, count: u32) -> Named {
    let mut named = Vec::new();
    for (k, w) in slots.iter_mut().enumerate() {
        if w[3] >> 28 >= 8 && w[0] >= first && w[0] - first < count {
            named.push((k as u16, (w[0] - first) as u16, rest_hash(w)));
            w[0] = 0;
        }
    }
    named
}

/// The page has a place again, at `first`: its slots that `evict_words` made null name their
/// descriptors there, as `copy_words` renumbers a copy for its destination. A slot written
/// since (another T#, a buffer, zeros) stays as it is. Returns the slots renamed.
fn restore_words(slots: &mut [[u32; 8]], named: &Named, first: u32) -> u32 {
    let mut renamed = 0;
    for &(k, at, hash) in named {
        if let Some(w) = slots.get_mut(k as usize).filter(|w| w[0] == 0 && w[3] >> 28 >= 8 && rest_hash(w) == hash) {
            w[0] = first + at as u32;
            renamed += 1;
        }
    }
    renamed
}

/// The place of page `p` of a heap (`pages`) for a lookup, which gives it none: a page without
/// one is marked `READ`. None: no place, or the heap is not in the shared heap.
fn seen_page(pages: &mut [u32], p: usize) -> Option<u32> {
    let first = pages.get_mut(p)?;
    if *first == UNMAPPED {
        *first = READ;
    }
    placed(*first).then_some(*first)
}

/// A page of a title's view heap takes its descriptors of the shared heap when a view with PC
/// descriptors is made in it or copied to it, not when one of its slots is only read (a draw
/// reads the tables it binds) or gets a view shaders read from memory – a constant buffer view,
/// a buffer view by address (`XCL_DESCRIPTOR_LAZY_PAGES=0`: on any use of a slot, as before).
fn lazy_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DESCRIPTOR_LAZY_PAGES").map_or(true, |v| v != "0"))
}

/// PC descriptors of page `p` of a heap of `n` slots: 2 per slot, and the last page only as
/// long as the heap (a heap of a few slots takes a few descriptors, not a whole page's).
fn page_descriptors(n: u32, p: u32) -> u32 {
    2 * n.saturating_sub(p * PAGE_SLOTS).clamp(1, PAGE_SLOTS)
}

/// A first-fit range of `count` in `free` (released ranges), else at `next` below `cap`.
fn take_range(free: &mut Vec<(u32, u32)>, next: &mut u32, cap: u32, count: u32) -> Option<u32> {
    if let Some(i) = free.iter().position(|&(_, n)| n >= count) {
        let (f, n) = free[i];
        if n == count {
            free.remove(i);
        } else {
            free[i] = (f + count, n - count);
        }
        return Some(f);
    }
    if cap - *next < count {
        return None;
    }
    *next += count;
    Some(*next - count)
}

fn return_range(free: &mut Vec<(u32, u32)>, next: &mut u32, mut first: u32, count: u32) {
    if count == 0 {
        return;
    }
    let mut end = first + count;
    let mut i = free.partition_point(|&(start, _)| start < first);
    if i > 0 && free[i - 1].0 + free[i - 1].1 == first {
        i -= 1;
        first = free.remove(i).0;
    }
    if i < free.len() && free[i].0 == end {
        end += free.remove(i).1;
    }
    if end == *next {
        *next = first;
    } else {
        free.insert(i, (first, end - first));
    }
}

/// The first descriptors of the shared heap, which no view of a title's or of XCL's is given:
/// null image views (`null_views`). A T# of zeros – an empty console slot – has PC index 0 in
/// its word 0, so an image shader that reads one takes descriptor 0 (SRV) and 1 (UAV) of the
/// heap. Whatever was reserved first lay there: the memory windows' 4 GiB raw buffer UAVs, read
/// as a texture, which hangs NVIDIA GPUs. An even count: SRV / UAV pairs.
pub const NULL_IMAGE_VIEWS: u32 = 4;

/// Null views at descriptors `0..NULL_IMAGE_VIEWS` of the shared heap (an SRV / UAV pair per
/// console slot: even SRV, odd UAV): a shader that lands there for a T# of zeros reads zeros,
/// and its writes go nowhere.
unsafe fn null_views(dev: &ID3D12Device, cpu: u64, inc: u32) {
    for k in 0..NULL_IMAGE_VIEWS as u64 {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize };
        if k % 2 == 0 {
            let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), at);
        } else {
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV::default() },
            };
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), at);
        }
    }
}

/// The shared shader-visible heap, made on first use. False: there is none (it could not be
/// made) – view heaps then keep their own.
pub unsafe fn shared_heap(dev: &ID3D12Device) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = GLOBAL.lock().unwrap();
    if g.is_none() {
        if GLOBAL_FAILED.load(Relaxed) {
            return false;
        }
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        let max = if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.MaxViewDescriptorHeapSize > 0 {
            o.MaxViewDescriptorHeapSize
        } else {
            1_000_000
        };
        let mut cap = max.min(GLOBAL_MAX);
        let mut d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: GLOBAL_MAX, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let mut larger: Option<ID3D12DescriptorHeap> = None;
        if cap < GLOBAL_MAX && larger_heap() {
            match dev.CreateDescriptorHeap(&d) {
                Ok(h) => {
                    crate::xlog_info!("console descriptor heaps: the device reports {max} view descriptors in a heap at most and made a shared heap of {GLOBAL_MAX} (XCL_DESCRIPTOR_HEAP_LARGE=0: one of {cap})");
                    larger = Some(h);
                    cap = GLOBAL_MAX;
                }
                Err(e) => crate::xlog_info!("console descriptor heaps: the device reports {max} view descriptors in a heap at most and refused a shared heap of {GLOBAL_MAX} ({e}): one of {cap}, as it reports"),
            }
        }
        d.NumDescriptors = cap;
        let heap: ID3D12DescriptorHeap = match larger.map_or_else(|| dev.CreateDescriptorHeap(&d), Ok) {
            Ok(h) => h,
            Err(e) => {
                GLOBAL_FAILED.store(true, Relaxed);
                crate::xlog_warn!("console descriptor heaps: no shared shader-visible heap of {cap} ({e}) – each keeps its own");
                return false;
            }
        };
        crate::xlog_info!("console descriptor heaps: one shader-visible PC heap of {cap} view descriptors for all of them; its first {NULL_IMAGE_VIEWS} are null image views (an empty slot's)");
        let singles = NULL_IMAGE_VIEWS + singles_region(cap, crate::state::config().title_memory_budget_mb(), std::env::var("XCL_DESCRIPTOR_SINGLES").ok().as_deref());
        if singles > NULL_IMAGE_VIEWS {
            crate::xlog_info!("console descriptor heaps: descriptors {NULL_IMAGE_VIEWS}..{singles} of the shared heap are kept for single views (the physical directory's roots, leaves, locators and page bindings); the title heaps' pages take {singles}..{cap} (XCL_DESCRIPTOR_SINGLES=0: one region for all, or the descriptors to keep)");
        }
        null_views(dev, heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        *g = Some(Global {
            dev: dev.clone(),
            cpu: heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
            gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
            inc: dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            heap,
            cap,
            singles: Region::new(NULL_IMAGE_VIEWS, singles),
            ranges: Region::new(singles, cap),
            held: [0; HOLDERS],
        });
    }
    true
}

/// `count` descriptors of the shared heap for `holder`: the index of the first. None: no shared
/// heap, or no room left in it. `single`: one descriptor, from the single descriptors' region
/// while that has room. `quiet`: no room is not logged (the caller makes room and asks again).
fn reserve_global(count: u32, holder: Holder, single: bool, quiet: bool) -> Option<u32> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = GLOBAL.lock().unwrap();
    let s = g.as_mut()?;
    let cap = s.cap;
    let mut r = if single { s.singles.take(count) } else { None };
    if !single && (s.ranges.crowded() || s.ranges.end - s.ranges.next < count) && evicting() && !tables_followed() {
        TABLES_FOLLOWED.store(super::present::presents() + 1, Relaxed);
    }
    if r.is_none() {
        r = s.ranges.take(count);
        if single && r.is_some() && s.singles.end > s.singles.first {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("console descriptor heaps: the single descriptors' region ({}..{}) is full – single views go among the title heaps' pages (XCL_DESCRIPTOR_SINGLES: the descriptors to keep)", s.singles.first, s.singles.end));
        }
    }
    let mut first_full = false;
    match r {
        Some(_) => s.held[holder as usize] += count,
        None if quiet => return None,
        None => {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            first_full = N.fetch_add(1, Relaxed) == 0;
            if first_full {
                crate::xlog_warn!("console descriptor heaps: the shared heap ({cap}) is full – shaders do not see the views of slots past it");
            }
        }
    }
    if r.is_none() || s.ranges.crowded() || s.singles.crowded() {
        let now = crate::log::uptime() as u64 + 1;
        if usage_owed(USAGE_AT.load(Relaxed), now, first_full) {
            USAGE_AT.store(now, Relaxed);
            USAGE_OWED.store(if r.is_none() { count + 1 } else { 1 }, Relaxed);
        }
    }
    r
}

/// seconds between usage lines
const USAGE_EVERY: u64 = 30;
/// title view heaps named in a usage line
const USAGE_HEAPS: usize = 8;
/// uptime (s) + 1 when a usage line was last owed (0: never)
static USAGE_AT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// a usage line is owed (0: none): 1, or the descriptors of the reservation that found no room + 1
static USAGE_OWED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether a usage line is owed at `now` (uptime + 1), the last at `last` (0: none yet): the
/// first time the heap is full, the first time it is asked, then every `USAGE_EVERY` seconds.
/// Asked while a reservation finds no room or a region is more than three quarters given out.
fn usage_owed(last: u64, now: u64, first_full: bool) -> bool {
    first_full || last == 0 || now >= last + USAGE_EVERY
}

/// The usage line, when one is owed (`reserve_global`). `heaps`: the heaps of a caller that
/// holds them; None: taken here, unless another call has them (the line then waits for the next).
fn report_usage(heaps: Option<&[Heap]>) {
    use std::sync::atomic::Ordering::Relaxed;
    if USAGE_OWED.load(Relaxed) == 0 {
        return;
    }
    let guard;
    let heaps = match heaps {
        Some(h) => h,
        None => match HEAPS.try_lock() {
            Ok(g) => {
                guard = g;
                &guard[..]
            }
            Err(_) => return,
        },
    };
    let owed = USAGE_OWED.swap(0, Relaxed);
    if owed == 0 {
        return;
    }
    let views: Vec<(u32, u32, u32)> = heaps.iter().filter(|h| !h.pages.is_empty()).map(|h| (h.n, h.pages.iter().filter(|&&f| placed(f)).count() as u32, h.pages.iter().filter(|&&f| f == READ).count() as u32)).collect();
    let evictions = (EVICTIONS.load(Relaxed), PLACED_AGAIN.load(Relaxed), heaps.iter().map(|h| h.evicted.len() as u64).sum());
    let g = GLOBAL.lock().unwrap();
    let Some(s) = g.as_ref() else { return };
    let line = usage_line(s.cap, owed - 1, s.held, s.ranges.usage(), s.singles.usage(), views, evictions);
    if owed > 1 {
        crate::xlog_warn!("{line}");
    } else {
        crate::xlog_info!("{line}");
    }
}

/// The usage line: the descriptors each `Holder` has, the title view heaps in the shared heap
/// (`views`: each one's slots, its pages with a place there and its pages only looked up, which
/// have none – `lookup`; the `USAGE_HEAPS` with the most pages are named) and the room left in
/// both regions. `failed`: the descriptors a reservation found no room for (0: none failed).
/// `evictions`: pages that lost their place to another so far (`evict`), evicted pages that got
/// a place again, and the pages that are evicted now.
fn usage_line(cap: u32, failed: u32, held: [u32; HOLDERS], ranges: RegionUsage, singles: RegionUsage, mut views: Vec<(u32, u32, u32)>, evictions: (u64, u64, u64)) -> String {
    let region = |name: &str, (first, end, used, free, largest): RegionUsage| format!("{name} {first}..{end}: {used} held, {free} free ranges, largest {largest}");
    views.sort_unstable_by_key(|&(n, mapped, _)| std::cmp::Reverse((mapped, n)));
    let named: Vec<String> = views.iter().take(USAGE_HEAPS).map(|&(n, mapped, _)| format!("{n} slots {mapped}/{} pages", n.div_ceil(PAGE_SLOTS).max(1))).collect();
    format!(
        "console descriptor heaps: shared heap usage{}: {} of {cap} descriptors held: title heap pages {}, memory windows {}, directory roots {}, directory leaves {}, directory locators {}, page bindings {}; {} title view heaps of {} slots, {} of their pages only looked up (no place), most pages first: [{}]; {} pages evicted, {} placed again, {} evicted now; {}; {}",
        if failed > 0 { format!(" (full: no room for {failed})") } else { String::new() },
        ranges.2 + singles.2,
        held[Holder::TitlePages as usize],
        held[Holder::Windows as usize],
        held[Holder::Roots as usize],
        held[Holder::Leaves as usize],
        held[Holder::Locators as usize],
        held[Holder::Pages as usize],
        views.len(),
        views.iter().map(|&(n, _, _)| n as u64).sum::<u64>(),
        views.iter().map(|&(_, _, seen)| seen as u64).sum::<u64>(),
        named.join(", "),
        evictions.0,
        evictions.1,
        evictions.2,
        region("ranges", ranges),
        region("singles", singles)
    )
}

/// The place in the shared heap of the page slot `k` of heap `i` is in – (index of the page's
/// first SRV there, the page's first slot) – given now if the page has none yet, where the
/// shared heap is full the place of a page long out of use (`evict`; `now`: the frame). An
/// evicted page gets its views and its slots' names back (`place_again`). None: the heap is
/// not in the shared heap, or that is full and no page can go.
fn page_of(heaps: &mut [Heap], i: usize, k: u32, now: u32) -> Option<(u32, u32)> {
    let p = (k / PAGE_SLOTS) as usize;
    let mut first = *heaps[i].pages.get(p)?;
    if !placed(first) {
        let need = page_descriptors(heaps[i].n, p as u32);
        let quiet = evicting();
        first = match reserve_global(need, Holder::TitlePages, false, quiet) {
            Some(first) => first,
            None if quiet => {
                evict(heaps, need, now);
                reserve_global(need, Holder::TitlePages, false, false)?
            }
            None => return None,
        };
        let h = &mut heaps[i];
        h.pages[p] = first;
        if let Some(named) = h.evicted.remove(&(p as u32)) {
            unsafe { place_again(h, p, first, &named) };
        }
    }
    Some((first, p as u32 * PAGE_SLOTS))
}

/// The title's memory of the slots of page `p` of `h` (None: not all of it is readable).
unsafe fn page_words(h: &Heap, p: usize) -> Option<&'static mut [[u32; 8]]> {
    let (at, slots) = (h.shadow + p * (PAGE_SLOTS * SLOT) as usize, (page_descriptors(h.n, p as u32) / 2) as usize);
    (at % 4 == 0 && crate::gpu::shader::readable(at, slots * SLOT as usize)).then(|| std::slice::from_raw_parts_mut(at as *mut [u32; 8], slots))
}

/// Room in the full shared heap for `need` descriptors: the page of a shader-visible heap
/// longest out of use (`victim`) loses its place. Its image slots name the null views until
/// it has a place again (`evict_words`), which it gets when one of its slots is used (`slot`,
/// `lookup`, `table_bound`). Only shader-visible heaps: a CPU-only heap is where a title
/// makes its views and copies them from, as memory too, so the descriptors of its pages are
/// named from slots XCL does not see used. What is not known here: a T# copied as memory out
/// of the page before, which names the place another page has now. False: no page can go.
fn evict(heaps: &mut [Heap], need: u32, now: u32) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    /// (frame << 32 | descriptors): no page of that many descriptors could go in that frame,
    /// so none can later in it – pages only come into use in a frame
    static NO_VICTIM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
    let refused = NO_VICTIM.load(Relaxed);
    if (refused >> 32) as u32 == now && need >= refused as u32 && evict_idle() != 0 {
        return false;
    }
    let followed = TABLES_FOLLOWED.load(Relaxed);
    if followed == 0 || now.wrapping_sub((followed - 1) as u32) < evict_idle() {
        return false;
    }
    let pages = heaps.iter().enumerate().filter(|(_, h)| h.visible).flat_map(|(i, h)| {
        h.pages.iter().zip(&h.used).enumerate().filter(|(_, (&f, _))| placed(f)).map(move |(p, (_, &used))| (i, p, page_descriptors(h.n, p as u32), used))
    });
    let Some((i, p)) = victim(pages, need, now, evict_idle()) else {
        NO_VICTIM.store((now as u64) << 32 | need as u64, Relaxed);
        return false;
    };
    let h = &mut heaps[i];
    let (first, count) = (h.pages[p], page_descriptors(h.n, p as u32));
    let named = unsafe { page_words(h, p) }.map_or_else(Vec::new, |slots| evict_words(slots, first, count));
    h.pages[p] = if named.is_empty() { UNMAPPED } else { EVICTED };
    if !named.is_empty() {
        h.evicted.insert(p as u32, named);
    }
    if let Some(res) = RES.lock().unwrap().as_mut() {
        for index in first..first + count {
            res.by_shared_index.remove(&index);
        }
    }
    if material_gate_views::enabled() {
        material_gate_views::forget_heap(0..0, std::iter::once((first, count)));
    }
    if super::depth_lineage::enabled() {
        if let Some(uavs) = LINEAGE_UAVS.lock().unwrap().as_mut() {
            for index in first..first + count {
                uavs.by_index.remove(&index);
            }
        }
    }
    release_held(first, count, Holder::TitlePages);
    if EVICTIONS.fetch_add(1, Relaxed) == 0 {
        crate::xlog_warn!("console descriptor heaps: the shared heap has no room for a view heap's page: it takes the place of the shader-visible heap page longest out of use ({} frames or more: XCL_DESCRIPTOR_EVICT_IDLE_FRAMES), whose image slots name the null views until one of its slots is used again (XCL_DESCRIPTOR_EVICT=0: the new page gets no place)", evict_idle());
    }
    true
}

/// Page `p` of `h`, evicted, has a place again at `first`: its PC descriptors come back from
/// the heap's own PC heap, where every view of it was made or copied, and its slots name them
/// (`restore_words`: `named`, from the eviction).
unsafe fn place_again(h: &Heap, p: usize, first: u32, named: &Named) {
    if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
        let own = h.pc_cpu + (2 * p as u32 * PAGE_SLOTS * h.inc) as u64;
        g.dev.CopyDescriptorsSimple(page_descriptors(h.n, p as u32), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (g.cpu + first as u64 * g.inc as u64) as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: own as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    if let Some(slots) = page_words(h, p) {
        restore_words(slots, named, first);
    }
    PLACED_AGAIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A list binds the descriptor table at console address `addr32` (once the ranges are nearly
/// given out: `TABLES_FOLLOWED`), `slots` slots by its root signature
/// (`console_list::Layout::tables`): the pages it reaches (`table_pages`) are in use this
/// frame – the shaders of the list read their slots from memory, where XCL does not see it –
/// and those that were evicted get a place again, while there is room.
pub fn table_bound(addr32: u32, slots: u32) {
    if !tables_followed() {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let cpu = u.base() + addr32 as usize;
    let mut heaps = HEAPS.lock().unwrap();
    let Some(i) = heaps.iter().position(|h| !h.pages.is_empty() && cpu >= h.shadow && cpu < h.shadow + (h.n * SLOT) as usize) else { return };
    let h = &mut heaps[i];
    let now = super::present::presents() as u32;
    let (pages, wide) = table_pages(h.n, ((cpu - h.shadow) / SLOT as usize) as u32, slots);
    if wide {
        if h.tail.is_some_and(|(frame, first)| frame == now && first <= pages.start) {
            return;
        }
        h.tail = Some((now, pages.start));
    }
    let mut room = true;
    for p in pages {
        let Some(used) = heaps[i].used.get_mut(p as usize) else { break };
        *used = now;
        if room && heaps[i].pages[p as usize] == EVICTED {
            room = page_of(&mut heaps, i, p * PAGE_SLOTS, now).is_some();
        }
    }
    report_usage(Some(&heaps));
}

/// The shared range starting at `first` (`count` descriptors) that `holder` had is free again.
pub fn release_held(first: u32, count: u32, holder: Holder) {
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        if first < s.singles.end {
            s.singles.give(first, count);
        } else {
            s.ranges.give(first, count);
        }
        s.held[holder as usize] = s.held[holder as usize].saturating_sub(count);
    }
}

/// The shared range starting at `first` (`count` descriptors, from `reserve_own`) is free again.
pub fn release_global(first: u32, count: u32) {
    release_held(first, count, Holder::Windows);
}

/// `count` descriptors of the shared heap for XCL's own views, kept for the process lifetime:
/// the index of the first and its CPU handle (the next ones `inc` apart). None: no shared heap,
/// or no room left in it.
pub fn reserve_own(count: u32) -> Option<(u32, D3D12_CPU_DESCRIPTOR_HANDLE, u32)> {
    reserve_for(count, Holder::Windows, false)
}

/// One descriptor of the shared heap for `holder` (given back with `release_held`), from the
/// single descriptors' region: its index and its CPU handle. None: no shared heap, or no room
/// left in it.
pub fn reserve_single(holder: Holder) -> Option<(u32, D3D12_CPU_DESCRIPTOR_HANDLE)> {
    reserve_for(1, holder, true).map(|(first, cpu, _)| (first, cpu))
}

fn reserve_for(count: u32, holder: Holder, single: bool) -> Option<(u32, D3D12_CPU_DESCRIPTOR_HANDLE, u32)> {
    let first = reserve_global(count, holder, single, false);
    report_usage(None);
    let first = first?;
    let g = GLOBAL.lock().unwrap();
    let s = g.as_ref()?;
    Some((first, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (s.cpu + first as u64 * s.inc as u64) as usize }, s.inc))
}

/// The shader-visible PC heap lists bind for view descriptors (None: none shared).
pub fn global_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's view heap) has its descriptors in the shared heap.
pub fn in_global(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && !h.pages.is_empty())
}

/// Memory for a heap of `bytes` from the title's allocation hook – where the title expects the
/// heap then. Only what descriptor tables can address (window 0) and is committed; anything else
/// goes back to the title.
unsafe fn title_memory(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    use crate::hle::kernelx::{runtime_alloc, runtime_free, D3D_DESCRIPTOR_HEAP};
    let p = runtime_alloc(bytes, D3D_DESCRIPTOR_HEAP)?;
    // (physical pages the title mapped there are committed on first use)
    crate::hle::kernelx::lazy::ensure(p, bytes);
    let readable = crate::gpu::shader::readable(p, bytes);
    if readable && u.window_of(p) == Some(0) && u.window_of(p + bytes - 1) == Some(0) {
        return Some(p);
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        crate::xlog_warn!("console descriptor heap: the title's memory {p:#x} ({bytes:#x} bytes) not used (window {:?}, committed {readable})", u.window_of(p));
    }
    runtime_free(p, D3D_DESCRIPTOR_HEAP);
    None
}

/// `obj` is a PC heap from `pc_cpu` / `pc_gpu` (increment `inc`; `pc_gpu` 0 when not shader
/// visible) with 2·`n` view descriptors or `n` samplers: give it `n` console slots in
/// unified memory. `shared`: its views go to the shared heap (`shared_heap`), a page at a
/// time as slots are used; `visible`: what the title asked for.
pub fn register(obj: usize, n: u32, pc_cpu: u64, pc_gpu: u64, inc: u32, sampler: bool, shared: bool, visible: bool) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let bytes = n.max(1) * if sampler { SAMPLER_SLOT } else { SLOT };
    // Descriptor tables are 32-bit console addresses (high dword 0): must live in window 0.
    // On the sparse-window path, `sparse::alloc_in(0, ...)` commits fresh zeroed pages so we
    // avoid touching unused pages with `write_bytes` (which would mark them dirty for write-watch).
    let (p, memory) = if let Some(p) = unsafe { title_memory(u, bytes as usize) } {
        (p as *mut u8, Memory::Title)
    } else if let Some(addr) = u.is_sparse_window(0).then(|| unsafe { super::sparse::alloc_in(0, bytes as usize, 256) }).flatten() {
        (addr.0 as *mut u8, Memory::Sparse)
    } else {
        // Window 0 first (a table's 32-bit console address names it), then anywhere XCL has committed
        // memory: when the windows are sparse and this adapter could not open them, window 0 keeps no
        // committed space, and a heap somewhere is still better than no descriptor heap at all.
        let p = match u.alloc_in(0, bytes as usize, 256) {
            Some(p) => p,
            None => {
                let Some(p) = u.alloc(bytes as usize, 256) else { return false };
                static SAID_WINDOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID_WINDOW.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::xlog_warn!("console descriptor heaps: no room in window 0, so heaps are kept in another committed window; a console 32-bit table address cannot name them");
                }
                p
            }
        };
        unsafe { std::ptr::write_bytes(p, 0, bytes as usize) };
        if u.is_sparse_window(0) {
            // The adapter could not make the 4 GiB sparse windows (a software adapter cannot), so the
            // heap lives in the committed window: a descriptor table still reaches its slots.
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::xlog_warn!("console descriptor heaps: the sparse VRAM windows are not open, so heaps are kept in the committed window (descriptor tables still work, the 32-bit slot aliasing does not)");
            }
        }
        (p, Memory::Uma)
    };
    {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
            crate::xlog_info!(
                "console descriptor heap {obj:#x}: {n} {} slots at {} {:#x}, {}",
                if sampler { "sampler" } else { "view" },
                match memory {
                    Memory::Sparse => "sparse VRAM",
                    Memory::Uma => "UMA",
                    Memory::Title => "the title's memory",
                },
                p as usize - u.base(),
                if visible { "shader visible" } else { "CPU only" }
            );
        }
    }
    let pages = if shared && !sampler && GLOBAL.lock().unwrap().is_some() { vec![UNMAPPED; n.div_ceil(PAGE_SLOTS).max(1) as usize] } else { Vec::new() };
    static STAMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let stamp = if vsharp::enabled() { STAMP.fetch_add(1,std::sync::atomic::Ordering::Relaxed)+1 } else {0};
    let used = vec![0; pages.len()];
    HEAPS.lock().unwrap().push(Heap { stamp, obj, shadow: p as usize, n, pc_cpu, pc_gpu, inc, sampler, pages, used, evicted: Default::default(), tail: None, visible, memory });
    if !sampler && n != 0 && super::depth_lineage::enabled() {
        LINEAGE_UAVS.lock().unwrap().get_or_insert_with(Default::default)
            .heaps.insert(p as usize, p as usize + n as usize * SLOT as usize);
    }
    true
}

/// The heap object was released for good: its slots go back to unified memory.
pub fn unregister(obj: usize) {
    let gone = {
        let mut heaps = HEAPS.lock().unwrap();
        let gone = heaps.iter().position(|h| h.obj == obj).map(|i| heaps.swap_remove(i));
        if let Some(h)=gone.as_ref().filter(|h|!h.sampler) { vsharp::forget(h.shadow..h.shadow+h.n as usize*SLOT as usize); }
        if let Some(h) = gone.as_ref().filter(|h| !h.sampler && material_gate_views::enabled()) {
            material_gate_views::forget_heap(h.shadow..h.shadow + h.n as usize * SLOT as usize,
                h.pages.iter().enumerate().filter(|&(_, &f)| placed(f)).map(|(p, &f)| (f, page_descriptors(h.n, p as u32))));
        }
        gone
    };
    if let Some(h) = gone {
        if !h.sampler && super::depth_lineage::enabled() {
            if let Some(uavs) = LINEAGE_UAVS.lock().unwrap().as_mut() {
                if let Some(end) = uavs.heaps.remove(&h.shadow) {
                    uavs.forget(|at, _| (h.shadow..end).contains(&at));
                }
            }
        }
        if let Some(res) = RES.lock().unwrap().as_mut() {
            res.forget(h.shadow..h.shadow + h.n as usize * h.slot_bytes() as usize);
        }
        for (p, &f) in h.pages.iter().enumerate().filter(|&(_, &f)| placed(f)) {
            release_held(f, page_descriptors(h.n, p as u32), Holder::TitlePages);
        }
        match h.memory {
            Memory::Sparse => {
                unsafe { super::sparse::free(h.shadow) };
            }
            Memory::Uma => {
                if let Some(u) = crate::gpu::uma::get() {
                    u.free(h.shadow);
                }
            }
            Memory::Title => unsafe { crate::hle::kernelx::runtime_free(h.shadow, crate::hle::kernelx::D3D_DESCRIPTOR_HEAP) },
        }
    }
}

/// Start of the console slots of heap `obj` (CPU handle = GPU handle on the console).
pub fn start(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| h.shadow as u64)
}

/// A console slot: where its descriptor goes and its PC descriptors.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub at: usize,
    /// views: PC heap index of the SRV (the UAV is at + 1); samplers: of the sampler
    pub index: u32,
    /// PC CPU handles of the SRV / UAV where views are made (the title's heap)
    pub pc_srv: u64,
    pub pc_uav: u64,
    /// PC GPU handle of the UAV (0: heap not shader visible)
    pub pc_gpu_uav: u64,
    pub sampler: bool,
    /// CPU handle of the SRV in the shared heap (0: the heap is not in it)
    pub shared_srv: u64,
    /// the slot's number in its heap
    pub k: u32,
}

/// The slot behind a console CPU handle (None: not one of XCL's console heaps). A view slot
/// of a heap in the shared heap gets its page's place there now if the page has none yet: for
/// a view with PC descriptors (an image, a view of a resource), which is made or cleared there.
pub fn slot(cpu: u64) -> Option<Slot> {
    let mut heaps = HEAPS.lock().unwrap();
    let s = slot_in(&mut heaps, cpu, true);
    report_usage(Some(&heaps));
    s
}

/// The slot behind a console CPU handle for a caller that needs none of its PC descriptors: a
/// reader of its words, or the writer of a view shaders read from memory (a buffer V#). As
/// `slot`, but a page with no place in the shared heap gets none (`lazy_pages`) and its slot
/// then names the null views, with no place of its own (`shared_srv` 0).
pub fn lookup(cpu: u64) -> Option<Slot> {
    let mut heaps = HEAPS.lock().unwrap();
    let s = slot_in(&mut heaps, cpu, !lazy_pages());
    report_usage(Some(&heaps));
    s
}

/// Whether a copy of view slot `s` has PC descriptors to carry: views were made in its page (it
/// has a place in the shared heap), or it holds an image T#, whose copy names its destination's
/// own PC descriptors (`copy_words`).
pub fn has_pc_views(s: &Slot) -> bool {
    s.shared_srv != 0 || unsafe { (s.at as *const u32).add(3).read_unaligned() } >> 28 >= 8
}

fn slot_in(heaps: &mut [Heap], cpu: u64, place: bool) -> Option<Slot> {
    let i = heaps.iter().position(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
    let h = &mut heaps[i];
    let k = ((cpu - h.shadow as u64) / h.slot_bytes() as u64) as u32;
    if h.sampler {
        return Some(Slot { at: h.shadow + (k * SAMPLER_SLOT) as usize, index: k, pc_srv: h.pc_cpu + (k * h.inc) as u64, pc_uav: 0, pc_gpu_uav: 0, sampler: true, shared_srv: 0, k });
    }
    let uav = ((2 * k + 1) * h.inc) as u64;
    let p = (k / PAGE_SLOTS) as usize;
    let now = super::present::presents() as u32;
    if let Some(used) = h.used.get_mut(p) {
        *used = now;
    }
    let page = if place || h.pages.get(p) == Some(&EVICTED) {
        page_of(heaps, i, k, now)
    } else {
        let marked = h.pages.get(p) == Some(&UNMAPPED);
        let first = seen_page(&mut h.pages, p);
        if marked {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_info!("console descriptor heaps: a view heap's page takes its descriptors of the shared heap when a view with PC descriptors is made in it or copied to it, not when a slot of it is only read or gets a constant buffer view or a buffer view by address (XCL_DESCRIPTOR_LAZY_PAGES=0: on any use, as before)"));
        }
        first.map(|first| (first, p as u32 * PAGE_SLOTS))
    };
    let h = &heaps[i];
    let shared = page.and_then(|(first, k0)| {
        let g = GLOBAL.lock().unwrap();
        let g = g.as_ref()?;
        let i = first + 2 * (k - k0);
        Some((i, g.gpu + (i as u64 + 1) * g.inc as u64, g.cpu + i as u64 * g.inc as u64))
    });
    let hosted = !h.pages.is_empty();
    let (index, pc_gpu_uav, shared_srv) = match shared {
        Some(place) => place,
        None if hosted && !place => (0, 0, 0),
        None => unplaced(k, hosted && null_when_full(), if h.pc_gpu == 0 { 0 } else { h.pc_gpu + uav }),
    };
    Some(Slot { at: h.shadow + (k * SLOT) as usize, index, pc_srv: h.pc_cpu + (2 * k * h.inc) as u64, pc_uav: h.pc_cpu + uav, pc_gpu_uav, sampler: false, shared_srv, k })
}

/// A view slot of a heap in the shared heap whose page found no room there names the null views
/// (`XCL_DESCRIPTOR_NULL_ON_FULL=0`: PC index 2k, as before).
fn null_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DESCRIPTOR_NULL_ON_FULL").map_or(true, |v| v != "0"))
}

/// (PC index, PC GPU handle of the UAV, CPU handle in the shared heap) of view slot `k` whose
/// page has no place in the shared heap. `hosted`: its heap is in the shared heap, which had no
/// room – the slot names the null views at descriptors 0 / 1 (`null_views`), so a shader reads
/// zeros there and its writes go nowhere. 2k, the slot's place in a heap's own PC heap
/// (`own_uav`: the GPU handle of its UAV there, 0: none), is in the shared heap another heap's
/// views or the memory windows' raw views – read as a texture, those hang NVIDIA GPUs – and
/// lists do not bind a hosted heap's own PC heap.
fn unplaced(k: u32, hosted: bool, own_uav: u64) -> (u32, u64, u64) {
    if !hosted {
        return (2 * k, own_uav, 0);
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::xlog_warn!("console descriptor heaps: a view slot whose page found no room in the shared heap names the null views (descriptors 0 / 1): shaders read zeros there, not another heap's views (XCL_DESCRIPTOR_NULL_ON_FULL=0: PC index 2k, as before)"));
    (0, 0, 0)
}

/// The view just made in slot `s` (its SRV, or with `uav` its UAV) goes to the shared heap.
pub unsafe fn publish(s: &Slot, uav: bool) {
    if s.shared_srv == 0 {
        return;
    }
    if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
        let (dst, src) = if uav { (s.shared_srv + g.inc as u64, s.pc_uav) } else { (s.shared_srv, s.pc_srv) };
        g.dev.CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

/// `n` view slots from `first` were copied (in the title's heap): they go to the shared heap,
/// a run per page (a heap's pages lie apart there), where their page has a place – one the
/// copy gave no place holds no view with PC descriptors (`has_pc_views`).
pub unsafe fn publish_range(first: &Slot, n: u32) {
    let mut done = 0;
    while done < n {
        let Some(s) = lookup((first.at + (done * SLOT) as usize) as u64) else { return };
        let run = (PAGE_SLOTS - s.k % PAGE_SLOTS).min(n - done);
        if s.shared_srv != 0 {
            if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
                g.dev.CopyDescriptorsSimple(2 * run, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.shared_srv as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
        }
        done += run;
    }
}

#[derive(Default)]
struct ViewOwners {
    by_slot: std::collections::HashMap<usize, usize>,
    by_shared_index: std::collections::HashMap<u32, (usize, usize)>,
    native_srvs: std::collections::HashMap<usize, NativeSrv>,
    native_uavs: std::collections::HashMap<usize, NativeUav>,
}

#[derive(Clone, Copy)]
pub(super) struct NativeUav {
    pub resource: usize,
    pub view: (i32, i32),
    pub backing: (u64, u32, u16, i32, i32, i32),
    pub volume: Option<(u32, u32, u32)>,
}

pub(super) type NativeSrv = (usize, i32, i32, i32, i32, i32);

impl ViewOwners {
    fn note(&mut self, s: &Slot, res: usize) {
        if res == 0 {
            self.by_slot.remove(&s.at);
            if s.shared_srv != 0 {
                self.by_shared_index.remove(&s.index);
            }
        } else {
            self.by_slot.insert(s.at, res);
            if s.shared_srv != 0 {
                self.by_shared_index.insert(s.index, (s.at, res));
            }
        }
    }

    fn forget(&mut self, slots: std::ops::Range<usize>) {
        self.by_slot.retain(|at, _| !slots.contains(at));
        self.by_shared_index.retain(|_, (at, _)| !slots.contains(at));
        self.native_srvs.retain(|at, _| !slots.contains(at));
        self.native_uavs.retain(|at, _| !slots.contains(at));
    }

    fn shared_owner(&self, index: u32) -> Option<usize> {
        self.by_shared_index.get(&index).map(|&(_, res)| res)
    }
}

static RES: Mutex<Option<ViewOwners>> = Mutex::new(None);

/// What each view slot names is kept (`note_resource`): for the traces and markers, for the
/// FSR4 stand-in, which tells its input and outputs by it (`bind::compute_images`), and for the
/// stand-in for the indirect light (`title_gta6_ambient`), which checks the resources behind the
/// views it writes through before it records anything.
fn owners_kept() -> bool {
    super::present::trace_frame() || super::markers::enabled() || super::present::writer_trace_configured() || super::fsr4_standin::wants_views() || super::title_gta6_ambient::wanted()
}

/// The unordered-access view of each slot is kept (`note_native_uav`): for the native-view
/// traces, and for the FSR4 stand-in (what a pass may write).
pub(super) fn native_uavs_kept() -> bool {
    super::present::native_view_trace_configured() || super::fsr4_standin::wants_views()
}

/// The view in slot `s` names `res` (0: none).
pub fn note_resource(s: &Slot, res: usize) {
    if s.sampler || !owners_kept() {
        return;
    }
    RES.lock().unwrap().get_or_insert_with(Default::default).note(s, res);
}

/// The resource of the view slot at UMA address `at`.
pub fn resource_at(at: usize) -> Option<usize> {
    RES.lock().unwrap().as_ref()?.by_slot.get(&at).copied()
}

pub(super) fn image_resource_at(addr32: u32, index: u32) -> Option<usize> {
    let shared = GLOBAL.lock().unwrap().is_some();
    if shared {
        return RES.lock().unwrap().as_ref()?.shared_owner(index);
    }
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    if s.sampler || s.index != index {
        return None;
    }
    resource_at(s.at)
}

pub(super) fn note_native_srv(s: &Slot, srv: Option<NativeSrv>) {
    let mut guard = RES.lock().unwrap();
    let owners = guard.get_or_insert_with(Default::default);
    if let Some(srv) = srv {
        owners.native_srvs.insert(s.at, srv);
    } else {
        owners.native_srvs.remove(&s.at);
    }
}

pub(super) fn native_srv_at(addr32: u32, index: u32) -> Option<NativeSrv> {
    native_view_at(addr32, index, |owners, at| owners.native_srvs.get(&at).copied())
}

pub(super) fn note_native_uav(s: &Slot, resource: Option<NativeUav>) {
    let mut guard = RES.lock().unwrap();
    let owners = guard.get_or_insert_with(Default::default);
    if let Some(resource) = resource {
        owners.native_uavs.insert(s.at, resource);
    } else {
        owners.native_uavs.remove(&s.at);
    }
}

/// The GPU handle of descriptor `index` of the shared view heap; None: there is no shared heap, or
/// it is not that large.
pub(super) fn shared_gpu(index: u32) -> Option<u64> {
    GLOBAL.lock().unwrap().as_ref().filter(|g| index < g.cap).map(|g| g.gpu + index as u64 * g.inc as u64)
}

pub(super) fn native_uav_at(addr32: u32, index: u32) -> Option<usize> {
    native_uav_details_at(addr32, index).map(|v| v.resource)
}

pub(super) fn native_uav_details_at(addr32: u32, index: u32) -> Option<NativeUav> {
    native_view_at(addr32, index, |owners, at| owners.native_uavs.get(&at).copied())
}

fn native_view_at<T>(addr32: u32, index: u32, get: impl FnOnce(&ViewOwners, usize) -> Option<T>) -> Option<T> {
    let shared = GLOBAL.lock().unwrap().is_some();
    if shared {
        let guard = RES.lock().unwrap();
        let owners = guard.as_ref()?;
        let &(at, _) = owners.by_shared_index.get(&index)?;
        return get(owners, at);
    }
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    if s.sampler || s.index != index {
        return None;
    }
    get(RES.lock().unwrap().as_ref()?, s.at)
}

type LineageUav = super::device_x::LineageUav;

#[derive(Default)]
struct LineageUavs {
    heaps: std::collections::BTreeMap<usize, usize>,
    by_slot: std::collections::HashMap<usize, ([u32; 8], LineageUav, Option<u32>)>,
    by_index: std::collections::HashMap<u32, usize>,
}

impl LineageUavs {
    fn forget(&mut self, mut forget: impl FnMut(usize, LineageUav) -> bool) {
        self.by_slot.retain(|&at, &mut (_, uav, _)| !forget(at, uav));
        let by_slot = &self.by_slot;
        self.by_index.retain(|_, at| by_slot.contains_key(at));
    }
}

static LINEAGE_UAVS: Mutex<Option<LineageUavs>> = Mutex::new(None);

pub(super) fn forget_lineage_uav_resource(resource: usize) {
    if let Some(uavs) = LINEAGE_UAVS.lock().unwrap().as_mut() {
        uavs.forget(|_, uav| uav.owner == resource || uav.shadow == Some(resource));
    }
}

pub(super) fn has_lineage_uavs() -> bool {
    LINEAGE_UAVS.lock().unwrap().as_ref().is_some_and(|uavs| !uavs.by_index.is_empty())
}

pub(super) fn lineage_uavs_in_table(table: u32, first: u32, count: u32) -> Vec<(u32, u32, LineageUav)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let Some(start) = first.checked_mul(SLOT).and_then(|off| table.checked_add(off)).and_then(|off| u.base().checked_add(off as usize)) else {
        return Vec::new();
    };
    let uavs = LINEAGE_UAVS.lock().unwrap();
    let Some(uavs) = uavs.as_ref() else { return Vec::new() };
    let Some((&base, &end)) = uavs.heaps.range(..=start).next_back() else {
        return Vec::new();
    };
    if start >= end || (start - base) % SLOT as usize != 0 {
        return Vec::new();
    }
    let count = count.min(((end - start) / SLOT as usize) as u32);
    if count == 0 || !crate::gpu::shader::readable(start, count as usize * SLOT as usize) {
        return Vec::new();
    }
    (0..count).filter_map(|i| {
        let at = start + i as usize * SLOT as usize;
        let words = unsafe { (at as *const [u32; 8]).read_unaligned() };
        if words[3] >> 28 < 8 {
            return None;
        }
        let source_at = uavs.by_index.get(&words[0])?;
        let &(expected, uav, index) = uavs.by_slot.get(source_at)?;
        (index == Some(words[0]) && expected == words).then_some((first + i, words[0], uav))
    }).collect()
}

pub fn write(s: &Slot, words: &[u32]) {
    write_view(s, words, None, None);
}

pub(super) fn write_view(s: &Slot, words: &[u32], view: Option<LineageUav>, native: Option<NativeView>) {
    write_observed(s,words,view,native,Writer::Other)
}

pub(super) fn write_observed(s: &Slot, words: &[u32], view: Option<LineageUav>, native: Option<NativeView>, writer: Writer) {
    if !s.sampler && vsharp::enabled() {vsharp::written(s.at,words,writer);}
    let n = words.len().min(if s.sampler { 4 } else { 8 });
    if !s.sampler && material_gate_views::enabled() {
        material_gate_views::note(s, words, native);
    }
    if !s.sampler && super::depth_lineage::enabled() {
        let mut g = LINEAGE_UAVS.lock().unwrap();
        if let Some(uavs) = g.as_mut() {
            if let Some((_, _, Some(index))) = uavs.by_slot.remove(&s.at) {
                uavs.by_index.remove(&index);
            }
        }
        if let Some((view, words)) = view.zip(<[u32; 8]>::try_from(words).ok()).filter(|(_, w)| w[3] >> 28 >= 8) {
            let uavs = g.get_or_insert_with(Default::default);
            let index = (s.shared_srv != 0).then_some(s.index);
            uavs.by_slot.insert(s.at, (words, view, index));
            if let Some(index) = index {
                uavs.by_index.insert(index, s.at);
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
        return;
    }
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
}

/// Copy a console slot's descriptor to another slot of the same kind, the PC heap index
/// in it (T# word 0, S# word 3 [11:0]) renumbered for the destination. Buffer V# carry
/// their address and stay as they are.
pub fn copy_words(src: &Slot, dst: &Slot) {
    let n = if src.sampler { 4 } else { 8 };
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, w.as_mut_ptr(), n) };
    let lineage = if !src.sampler && super::depth_lineage::enabled() {
        LINEAGE_UAVS.lock().unwrap().as_ref().and_then(|uavs| uavs.by_slot.get(&src.at))
            .and_then(|&(words, uav, _)| (w == words).then_some(uav))
    } else {
        None
    };
    let native = if !src.sampler && material_gate_views::enabled() {
        material_gate_views::copied(src, &w)
    } else {
        None
    };
    if src.sampler {
        w[3] = (w[3] & !0xFFF) | (dst.index & 0xFFF);
    } else if w[3] >> 28 >= 8 {
        w[0] = dst.index;
    }
    write_observed(dst, &w[..n], lineage, native,Writer::Copy(src.at));
    if !src.sampler {
        note_resource(dst, resource_at(src.at).unwrap_or(0));
        if native_uavs_kept() {
            let (srv, uav) = RES.lock().unwrap().as_ref().map_or((None, None), |owners| {
                (owners.native_srvs.get(&src.at).copied(), owners.native_uavs.get(&src.at).copied())
            });
            if super::present::native_view_trace_configured() {
                note_native_srv(dst, srv);
            }
            note_native_uav(dst, uav);
        }
    }
}

/// `XCL_TRACE_FRAME`: what the slots of each shader-visible view heap hold – images whose
/// PC index is the slot's own (made or copied through the device) or another (copied by the
/// title itself, as memory), buffers, empty.
pub fn census() {
    let heaps = HEAPS.lock().unwrap();
    // samplers: S# word 3 [11:0] is the PC index in the sampler heap the slot belongs to
    for h in heaps.iter().filter(|h| h.sampler) {
        let (mut empty, mut own, mut other) = (0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SAMPLER_SLOT) as usize) as *const u32, 4) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] & 0xFFF == k {
                own += 1;
            } else {
                other += 1;
                if examples.len() < 8 {
                    examples.push((k, w[3] & 0xFFF));
                }
            }
        }
        crate::xlog_info!(
            "frame trace: sampler heap {:#x} ({} slots, {}): {own} with their own index, {other} with another (slot, index: {examples:?}), {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
    for h in heaps.iter().filter(|h| !h.sampler) {
        let (mut empty, mut own, mut other, mut buffers) = (0u32, 0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        let own_index = |k: u32| match h.pages.get((k / PAGE_SLOTS) as usize) {
            Some(&f) if placed(f) => f + 2 * (k % PAGE_SLOTS),
            _ => 2 * k,
        };
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SLOT) as usize) as *const u32, 8) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] >> 28 >= 8 {
                if w[0] == own_index(k) {
                    own += 1;
                } else {
                    other += 1;
                    if examples.len() < 8 {
                        examples.push((k, w[0]));
                    }
                }
            } else {
                buffers += 1;
            }
        }
        crate::xlog_info!(
            "frame trace: view heap {:#x} ({} slots, {}): {own} images with their own PC index, {other} with another (slot, index: {examples:?}), {buffers} buffers, {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
}

/// The descriptor at a console GPU address (low 32 bits, as in a descriptor table), if it
/// is in one of the console heaps.
pub fn words_at(addr32: u32) -> Option<[u32; 8]> {
    let u = crate::gpu::uma::get()?;
    let s = lookup((u.base() + addr32 as usize) as u64)?;
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, w.as_mut_ptr(), if s.sampler { 4 } else { 8 }) };
    Some(w)
}

pub(super) fn material_input(table: u32, offset: u32) -> Option<(u32, [u32; 8], Option<Key>)> {
    let u = crate::gpu::uma::get()?;
    let addr = table.checked_add(offset)?;
    let base = u.base().checked_add(table as usize)?;
    let at = u.base().checked_add(addr as usize)?;
    let heaps = HEAPS.lock().unwrap();
    let h = heaps.iter().find(|h| !h.sampler && base >= h.shadow && at >= base
        && (base - h.shadow) % SLOT as usize == 0 && (at - h.shadow) / (SLOT as usize) < h.n as usize)?;
    if offset % SLOT != 0 || !crate::gpu::shader::readable(at, SLOT as usize) {
        return None;
    }
    let words = unsafe { (at as *const [u32; 8]).read_unaligned() };
    let key = if GLOBAL.lock().unwrap().is_some() {
        Some(Key::Shared(words[0]))
    } else {
        let k = ((at - h.shadow) / SLOT as usize) as u32;
        (h.visible && words[0] == 2 * k).then_some(Key::Local(at))
    };
    Some((addr, words, key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_owner_follows_descriptor_index_across_raw_copies() {
        let source = Slot { at: 0x1000, index: 17, pc_srv: 0, pc_uav: 0, pc_gpu_uav: 0, sampler: false, shared_srv: 1, k: 0 };
        let copy = Slot { at: 0x2000, index: 19, ..source };
        let mut owners = ViewOwners::default();
        owners.note(&source, 101);
        owners.note(&copy, 102);
        let words = crate::gpu::desc::xcl_image_for_view(source.index, 22, crate::gpu::desc::ImageType::D3, 4, 4, 4, 1);
        assert_eq!(owners.shared_owner(words[0]), Some(101));
        assert_eq!(owners.by_slot.get(&copy.at), Some(&102));
        owners.forget(source.at..source.at + SLOT as usize);
        assert_eq!(owners.shared_owner(words[0]), None);
        assert_eq!(owners.shared_owner(copy.index), Some(102));
    }

    #[test]
    fn texture3d_uav_companion_reads_written_mip() {
        use super::super::real;
        use std::mem::ManuallyDrop;
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};

        if let Err(e) = crate::gpu::recomp::dxc::location().and_then(|_| real::init()) {
            println!("SKIPPED texture3d_uav_companion_reads_written_mip: {e}");
            return;
        }
        if let Some(why) = sparse_windows_missing() {
            return crate::gpu::recomp::runner::skip_test(&why);
        }
        unsafe {
            let raw = real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap();
            let ty = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: 2,
                ..Default::default()
            }).unwrap();
            let shared = global_heap().expect("shared descriptor heap");
            let obj = heap.as_raw() as usize;
            assert!(register(obj, 1, heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, 0, dev.GetDescriptorHandleIncrementSize(ty), false, true, true));
            let cpu = start(obj).unwrap();
            let s = slot(cpu).unwrap();
            assert_ne!(s.shared_srv, 0);
            let resource = |heap, state, desc: &D3D12_RESOURCE_DESC| {
                let mut r: Option<ID3D12Resource> = None;
                super::super::title_textures::own_work(|| dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE, desc, state, None, &mut r,
                )).unwrap();
                r.unwrap()
            };
            let volume = resource(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                Width: 4, Height: 4, DepthOrArraySize: 4, MipLevels: 2,
                Format: DXGI_FORMAT_R32_UINT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            });
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() },
                },
            }), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize });
            publish(&s, false);
            real::dev_create_uav_x(raw, volume.as_raw(), std::ptr::null_mut(), &D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 0, WSize: u32::MAX },
                },
            }, cpu);
            let index = (s.at as *const u32).read();
            assert_eq!(index, s.index);

            let hlsl = format!("
                RWByteAddressBuffer res : register(u1);
                [numthreads(1, 1, 1)]
                void main() {{
                    Texture3D<uint> tex = ResourceDescriptorHeap[{index}];
                    uint w, h, d, mips;
                    tex.GetDimensions(0, w, h, d, mips);
                    res.Store4(0, uint4(tex.Load(int4(1, 1, 1, 0)), w, h, d));
                }}
            ");
            let dxil = crate::gpu::recomp::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
            let rs = crate::gpu::recomp::pso::abi_root_signature(dev, false).unwrap();
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            ManuallyDrop::drop(&mut desc.pRootSignature);
            let mut buffer = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let out = resource(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &buffer);
            buffer.Flags = D3D12_RESOURCE_FLAG_NONE;
            let readback = resource(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &buffer);
            let transition = |r, before, after| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(r),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before, StateAfter: after,
                    }),
                },
                ..Default::default()
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            const TEXEL: u32 = 0x1234_ABCD;
            list.SetDescriptorHeaps(&[Some(shared)]);
            list.ClearUnorderedAccessViewUint(
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: s.pc_gpu_uav },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_uav as usize },
                &volume, &[TEXEL, 0, 0, 0], &[],
            );
            list.ResourceBarrier(&[transition(&volume, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&readback, 0, &out, 0, 16);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, dev, None, 10_000).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let got = p.cast::<[u32; 4]>().read();
            readback.Unmap(0, None);
            assert_eq!(got, [TEXEL, 2, 2, 2]);
        }
    }

    /// Ranges of the shared heap: released ones first (first fit, split), then new ones, and
    /// none past its size.
    #[test]
    fn shared_heap_ranges() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(0));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(40));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), None, "20 left");
        return_range(&mut free, &mut next, 0, 40);
        assert_eq!(take_range(&mut free, &mut next, 100, 30), Some(0));
        assert_eq!(free, vec![(30, 10)]);
        assert_eq!(take_range(&mut free, &mut next, 100, 20), Some(80));
        assert_eq!(take_range(&mut free, &mut next, 100, 10), Some(30));
        assert!(free.is_empty());
    }

    /// The single descriptors' region: one for each 64 KiB page of the title's memory and the
    /// directory's views, a quarter of the heap at most, below what a page table entry can name.
    #[test]
    fn singles_region_is_sized_for_the_titles_pages() {
        assert_eq!(singles_region(1_000_000, 8_192, None), 131_072 + SINGLES_VIEWS, "Series S");
        assert_eq!(singles_region(1_000_000, 13_824, None), 221_184 + SINGLES_VIEWS, "Series X");
        assert_eq!(singles_region(1_000_000, 13_824, Some("0")), 0, "off");
        assert_eq!(singles_region(1_000_000, 13_824, Some(" 5000 ")), 5000);
        assert_eq!(singles_region(1_000_000, 13_824, Some("on")), 221_184 + SINGLES_VIEWS, "not a number: the default");
        assert_eq!(singles_region(1000, 8_192, None), 250, "a quarter of a small heap");
        assert_eq!(singles_region(GLOBAL_MAX, u64::MAX, None), GLOBAL_MAX / 4);
        assert!(NULL_IMAGE_VIEWS + singles_region(u32::MAX, u64::MAX, Some("4294967295")) <= SINGLES_END);
    }

    /// Title heap pages and single descriptors each keep to their region: pages that fill theirs
    /// leave the singles their room, and a single given back leaves a page's range whole.
    #[test]
    fn single_descriptors_keep_their_region() {
        let mut singles = Region::new(NULL_IMAGE_VIEWS, 104);
        let mut ranges = Region::new(104, 104 + 3 * 512);
        assert_eq!(singles.usage(), (4, 104, 0, 1, 100));
        let pages: Vec<u32> = (0..3).map(|_| ranges.take(512).unwrap()).collect();
        assert_eq!(pages, [104, 616, 1128]);
        assert_eq!(ranges.take(512), None, "the ranges are full");
        assert!(ranges.crowded() && !singles.crowded());
        assert_eq!(ranges.usage(), (104, 1640, 1536, 0, 0));
        let views: Vec<u32> = (0..76).map(|_| singles.take(1).unwrap()).collect();
        assert_eq!((views[0], views[75]), (NULL_IMAGE_VIEWS, 79), "after the null views, below the pages");
        assert!(singles.crowded(), "76 of 100");
        singles.give(views[10], 1);
        singles.give(views[11], 1);
        singles.give(views[40], 1);
        assert_eq!(singles.usage(), (4, 104, 73, 3, 24));
        assert!(!singles.crowded());
        assert_eq!(singles.take(1), Some(views[10]));
        ranges.give(pages[1], 512);
        assert_eq!(ranges.usage(), (104, 1640, 1024, 1, 512));
        assert_eq!(ranges.take(512), Some(pages[1]), "a page's range comes back whole");
        let (mut none, mut all) = (Region::new(NULL_IMAGE_VIEWS, NULL_IMAGE_VIEWS), Region::new(NULL_IMAGE_VIEWS, 100));
        assert_eq!((none.take(1), none.crowded(), none.usage()), (None, false, (4, 4, 0, 0, 0)), "no region: nothing to take there");
        assert_eq!(all.take(1), Some(NULL_IMAGE_VIEWS), "the singles then come from the ranges, as before");
    }

    /// A slot of a heap in the shared heap that found no room there names the null SRV / UAV
    /// pair – what a T# of zeros finds – and has no UAV to clear through; a heap with its own PC
    /// heap keeps 2k and its own UAV.
    #[test]
    fn a_slot_past_a_full_shared_heap_names_the_null_views() {
        assert_eq!(unplaced(7, true, 0x5000), (0, 0, 0));
        assert!(NULL_IMAGE_VIEWS >= 2, "descriptors 0 / 1 are the null pair");
        assert_eq!(unplaced(7, false, 0x5000), (14, 0x5000, 0));
        assert_eq!(unplaced(7, false, 0), (14, 0, 0));
    }

    /// A lookup gives a page no place: it marks the page, once, and finds the place a view
    /// made there took before or takes after.
    #[test]
    fn a_page_only_looked_up_takes_no_place() {
        let mut pages = vec![UNMAPPED, 616, UNMAPPED];
        assert_eq!(seen_page(&mut pages, 0), None);
        assert_eq!(seen_page(&mut pages, 0), None);
        assert_eq!(seen_page(&mut pages, 1), Some(616));
        assert_eq!(seen_page(&mut pages, 3), None, "past the heap");
        assert_eq!(pages, [READ, 616, UNMAPPED]);
        assert!(!placed(READ) && !placed(UNMAPPED) && placed(616) && placed(0));
        assert_eq!(pages.iter().filter(|&&f| placed(f)).count(), 1, "one page's descriptors held");
        pages[0] = 104;
        assert_eq!(seen_page(&mut pages, 0), Some(104), "a view was made there since");
        assert!(seen_page(&mut [], 0).is_none(), "a heap outside the shared heap");
    }

    /// The page that goes: the one longest out of use that is large enough and was not used in
    /// the last frames, whatever the frame counter's wrap.
    #[test]
    fn the_page_longest_out_of_use_goes() {
        let pages = [(0, 0, 512, 100), (0, 1, 512, 40), (1, 0, 8, 10), (1, 3, 512, 990), (2, 7, 512, 960)];
        assert_eq!(victim(pages.into_iter(), 512, 1000, 30), Some((0, 1)), "the 8 descriptors of the oldest are too few");
        assert_eq!(victim(pages.into_iter(), 8, 1000, 30), Some((1, 0)), "a short page takes a short page's place");
        assert_eq!(victim(pages.into_iter(), 512, 1000, 961), None, "none out of use that long");
        assert_eq!(victim(pages.into_iter(), 512, 1000, 960), Some((0, 1)));
        assert_eq!(victim(pages[3..].iter().copied(), 512, 1000, 30), Some((2, 7)), "used 10 frames ago: stays");
        assert_eq!(victim(pages[3..4].iter().copied(), 512, 1000, 30), None);
        assert_eq!(victim(pages[3..4].iter().copied(), 512, 1000, 10), Some((1, 3)));
        assert_eq!(victim([(0, 0, 512, u32::MAX - 5), (0, 1, 512, 3)].into_iter(), 512, 40, 30), Some((0, 0)), "46 frames ago, across the wrap");
        assert_eq!(victim(std::iter::empty(), 512, 1000, 30), None);
    }

    /// The pages a table reaches: from its first slot's to its last's, to the heap's end at
    /// most – and to the heap's end when it is a wide one.
    #[test]
    fn a_table_reaches_the_pages_of_its_slots() {
        assert_eq!(table_pages(1_245_184, 0, 1), (0..1, false));
        assert_eq!(table_pages(1_245_184, 255, 1), (0..1, false));
        assert_eq!(table_pages(1_245_184, 255, 2), (0..2, false));
        assert_eq!(table_pages(1_245_184, 1000, 64), (3..5, false), "slots 1000..1064: pages 3 and 4");
        assert_eq!(table_pages(1_245_184, 1024, 256), (4..5, false));
        assert_eq!(table_pages(1_245_184, 1024, 0), (4..5, false), "a table of no range: its first slot");
        assert_eq!(table_pages(1_245_184, 512, 16 * 256), (2..18, false));
        assert_eq!(table_pages(1_245_184, 512, 16 * 256 + 1), (2..4864, true), "more than 16 pages: to the heap's end");
        assert_eq!(table_pages(1_245_184, 512, u32::MAX), (2..4864, true), "a range without end");
        assert_eq!(table_pages(3128, 3000, u32::MAX), (11..13, false), "the heap ends in page 12");
        assert_eq!(table_pages(3128, 3127, 500), (12..13, false));
        assert_eq!(table_pages(4, 3, 9), (0..1, false));
    }

    /// An evicted page's image slots name the null views, and their own descriptors again where
    /// the page gets a place – but not a slot the title wrote meanwhile, and never a T# that
    /// names another page, a buffer or an empty slot.
    #[test]
    fn an_evicted_pages_slots_name_null_views_until_it_has_a_place_again() {
        let image = |index: u32, format: u32| [index, format, 0, 9 << 28, 0, 0, 0, 0];
        let buffer = [0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0];
        let mut slots = [image(616, 0x0470_0000), image(618, 0x0470_0000), image(616 + 2 * 7, 0x0480_0000), image(104, 0x0470_0000), buffer, [0; 8], image(616 + 10, 0x0490_0000), image(1128, 0x0470_0000)];
        let before = slots;
        let named = evict_words(&mut slots, 616, 512);
        assert_eq!(named.iter().map(|&(k, at, _)| (k, at)).collect::<Vec<_>>(), [(0, 0), (1, 2), (2, 14), (6, 10)]);
        for k in [0, 1, 2, 6] {
            assert_eq!((slots[k][0], &slots[k][1..]), (0, &before[k][1..]), "slot {k} names the null views and is the same view otherwise");
        }
        for k in [3, 4, 5, 7] {
            assert_eq!(slots[k], before[k], "slot {k}: another page's image (104, 1128), a buffer, an empty slot");
        }
        slots[1] = image(0, 0x0499_0000);
        slots[6] = buffer;
        assert_eq!(restore_words(&mut slots, &named, 2152), 2);
        assert_eq!((slots[0], slots[2]), (image(2152, 0x0470_0000), image(2152 + 14, 0x0480_0000)), "slot 2 names slot 7's views, as it did");
        assert_eq!((slots[1], slots[6]), (image(0, 0x0499_0000), buffer), "written since: as the title left them");
        assert_eq!(restore_words(&mut slots, &named, 4000), 0, "already named");
        assert!(evict_words(&mut [buffer, [0; 8], image(104, 1)], 616, 512).is_empty(), "nothing named: the page is as one never placed");
        assert!(!placed(EVICTED) && placed(EVICTED - 1));
    }

    #[test]
    fn usage_lines_are_owed_when_full_and_every_30_s() {
        assert!(usage_owed(0, 200, false), "the first time it is asked");
        assert!(!usage_owed(200, 229, false));
        assert!(usage_owed(200, 230, false));
        assert!(usage_owed(200, 201, true), "full for the first time");
    }

    #[test]
    fn usage_line_counts_every_holder() {
        let held = [900_000, 5, 12, 30, 200, 60_000];
        let line = usage_line(1_000_000, 512, held, (147_460, 1_000_000, 852_540, 2, 300), (4, 147_460, 60_242, 1, 87_214), vec![(4, 1, 0), (73_727, 3, 40), (500_000, 1954, 0)], (700, 12, 688));
        assert_eq!(
            line,
            "console descriptor heaps: shared heap usage (full: no room for 512): 912782 of 1000000 descriptors held: title heap pages 900000, memory windows 5, directory roots 12, directory leaves 30, directory locators 200, page bindings 60000; 3 title view heaps of 573731 slots, 40 of their pages only looked up (no place), most pages first: [500000 slots 1954/1954 pages, 73727 slots 3/288 pages, 4 slots 1/1 pages]; 700 pages evicted, 12 placed again, 688 evicted now; ranges 147460..1000000: 852540 held, 2 free ranges, largest 300; singles 4..147460: 60242 held, 1 free ranges, largest 87214"
        );
        assert!(usage_line(100, 0, [0; HOLDERS], (4, 100, 0, 1, 96), (4, 4, 0, 0, 0), Vec::new(), (0, 0, 0)).starts_with("console descriptor heaps: shared heap usage: 0 of 100 descriptors held: "));
    }

    #[test]
    fn adjacent_shared_heap_ranges_are_reusable_together() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        for first in [0, 32, 64, 96] {
            assert_eq!(take_range(&mut free, &mut next, 160, 32), Some(first));
        }
        for first in [64, 0, 32] {
            return_range(&mut free, &mut next, first, 32);
        }
        assert_eq!(free, [(0, 96)]);
        assert_eq!(take_range(&mut free, &mut next, 160, 96), Some(0));
        return_range(&mut free, &mut next, 96, 32);
        assert_eq!(take_range(&mut free, &mut next, 160, 64), Some(96));
    }

    /// A T# of zeros (an empty slot) names descriptors 0 (SRV) and 1 (UAV) of the shared heap:
    /// those are null views, written before anything is reserved, and nothing is ever reserved
    /// there – not the memory windows' raw buffer UAVs (a texture read of one hangs the GPU).
    #[test]
    fn a_tsharp_of_zeros_finds_null_views_and_never_a_memory_window() {
        use std::ffi::c_void;
        use windows::core::Interface;
        static SEEN: Mutex<Vec<(bool, usize, usize)>> = Mutex::new(Vec::new());
        #[repr(C)]
        struct Object {
            vt: *const usize,
        }
        unsafe extern "system" fn srv(_this: *mut c_void, r: *mut c_void, _d: *const D3D12_SHADER_RESOURCE_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((false, r as usize, at.ptr));
        }
        unsafe extern "system" fn uav(_this: *mut c_void, r: *mut c_void, _c: *mut c_void, _d: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((true, r as usize, at.ptr));
        }
        assert!(NULL_IMAGE_VIEWS >= 2, "the SRV / UAV pair of a T# of zeros is not kept");
        assert_eq!(NULL_IMAGE_VIEWS % 2, 0, "the kept descriptors split a console slot's SRV / UAV pair");
        // (ID3D12Device: CreateShaderResourceView and CreateUnorderedAccessView)
        let mut vt = [0usize; 20];
        vt[18] = srv as usize;
        vt[19] = uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = std::mem::ManuallyDrop::new(ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void));
            null_views(&dev, 0x1000, 32);
        }
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), NULL_IMAGE_VIEWS as usize);
        for (k, &(is_uav, r, at)) in seen.iter().enumerate() {
            assert_eq!((is_uav, r, at), (k % 2 == 1, 0, 0x1000 + 32 * k), "descriptor {k}");
        }
        // a heap as it is made: the first range given starts after the null views, and one
        // given back and taken again does not move below them
        let (mut free, mut next) = (Vec::new(), NULL_IMAGE_VIEWS);
        let first = take_range(&mut free, &mut next, 1 << 20, crate::gpu::recomp::MEM_NULL_VIEW + 1).unwrap();
        assert_eq!(first, NULL_IMAGE_VIEWS, "the memory windows' views, reserved first, start after the null views");
        return_range(&mut free, &mut next, first, crate::gpu::recomp::MEM_NULL_VIEW + 1);
        assert!(take_range(&mut free, &mut next, 1 << 20, 2).unwrap() >= NULL_IMAGE_VIEWS);
        // with a device: the views XCL reserved for the memory windows
        if super::super::real::init().is_ok() {
            if let Some(views) = super::super::real::mem_views() {
                assert!(views >= NULL_IMAGE_VIEWS, "the memory windows' views start at descriptor {views}");
            }
        }
    }

    /// The 4 GiB sparse VRAM windows (a software adapter such as the CI runner's WARP cannot reserve
    /// them). Tests that prove what only they do skip with this reason; XCL_REQUIRE_GPU=window makes
    /// the skip a failure.
    fn sparse_windows_missing() -> Option<String> {
        if crate::hle::d3d12x::sparse::windows_ready() {
            return None;
        }
        Some("no 4 GiB sparse memory window: this adapter cannot reserve one".to_string())
    }

    #[test]
    fn slots_copy_and_release() {
        if let Some(why) = sparse_windows_missing() {
            return crate::gpu::recomp::runner::skip_test(&why);
        }
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        // views: 4 slots over a PC heap at 0x1000 (increment 32, GPU 0x5000); samplers: 2 at 0x9000
        assert!(register(0xA1, 4, 0x1000, 0x5000, 32, false, false, true));
        assert!(register(0xA2, 2, 0x9000, 0, 16, true, false, true));
        let (v, sm) = (start(0xA1).unwrap(), start(0xA2).unwrap());
        assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");
        assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");
        assert!(u.is_sparse(v as usize) && u.is_sparse(sm as usize), "descriptor heaps backed by sparse VRAM");
        let s1 = slot(v + 32).unwrap();
        assert_eq!((s1.index, s1.pc_srv, s1.pc_uav, s1.pc_gpu_uav), (2, 0x1040, 0x1060, 0x5060));
        // a 2D T# from slot 1 to slot 3: word 0 = the destination's SRV index
        write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        let s3 = slot(v + 96).unwrap();
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], [6, 0x0470_0000, 0, 9 << 28]);
        // a buffer V# keeps its address
        write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[0], 0x1234);
        // a raytracing BVH SRSRC keeps its base address and box-sort flag verbatim
        let bvh = crate::gpu::desc::bvh_words(0x12_3456_7800, true);
        write(&s1, &[bvh[0], bvh[1], bvh[2], bvh[3], 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], bvh);
        // samplers: 16-byte slots, S# word 3 [11:0] = sampler index
        let (q0, q1) = (slot(sm).unwrap(), slot(sm + 16).unwrap());
        assert!(q1.sampler && q1.index == 1 && q1.pc_srv == 0x9010);
        write(&q0, &[1, 2, 3, 0xABC0_0000]);
        copy_words(&q0, &q1);
        assert_eq!(words_at(rel(sm + 16)).unwrap()[..4], [1, 2, 3, 0xABC0_0001]);
        assert!(slot(sm + 32).is_none(), "past the last sampler slot");
        unregister(0xA1);
        unregister(0xA2);
        assert!(start(0xA1).is_none() && slot(v).is_none() && slot(sm).is_none());
    }

    /// Minecraft's view heaps: eight of 73 727 slots, 147 454 PC descriptors each – more in all
    /// than an NVIDIA shared heap holds (1 000 000). Each gets a place there for the pages it
    /// uses; before, the seventh and eighth got none and their views were lost (flicker).
    #[test]
    fn large_heaps_get_shared_pages_as_used() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            return;
        }
        if let Some(why) = sparse_windows_missing() {
            return crate::gpu::recomp::runner::skip_test(&why);
        }
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            dev.expect("D3D12 device")
        };
        assert!(unsafe { shared_heap(&dev) });
        const N: u32 = 73_727;
        let objs: Vec<usize> = (0..8).map(|i| 0xB000 + i).collect();
        for &o in &objs {
            assert!(register(o, N, 0x10_0000, 0, 32, false, true, true));
            assert!(in_global(o));
        }
        let mut seen = Vec::new();
        for &o in &objs {
            let v = start(o).unwrap();
            // slot 0, the last of page 0, the first of page 1 and the heap's last slot
            let s: Vec<Slot> = [0, PAGE_SLOTS - 1, PAGE_SLOTS, N - 1].iter().map(|&k| slot(v + (k * SLOT) as u64).unwrap()).collect();
            assert!(s.iter().all(|x| x.shared_srv != 0), "heap {o:#x}: a slot without a place in the shared heap");
            assert_eq!(s[1].index, s[0].index + 2 * (PAGE_SLOTS - 1), "a page is contiguous");
            assert_eq!(s[0].pc_srv, 0x10_0000, "views are still made in the title's heap");
            assert_eq!(s[3].pc_srv, 0x10_0000 + (2 * (N - 1) * 32) as u64);
            for x in &s {
                assert!(!seen.contains(&x.index), "two slots share the PC index {}", x.index);
                seen.push(x.index);
            }
            // the same slot again: the same place
            assert_eq!(slot(v).unwrap().index, s[0].index);
        }
        for &o in &objs {
            unregister(o);
        }
        // a heap smaller than a page takes 2 descriptors per slot, as it did when a heap's
        // whole range was reserved (a title with many small heaps must not fill the shared heap)
        assert_eq!((page_descriptors(4, 0), page_descriptors(N, 0), page_descriptors(N, N / PAGE_SLOTS)), (8, 2 * PAGE_SLOTS, 2 * (N % PAGE_SLOTS)));
        assert!(register(0xB100, 4, 0x10_0000, 0, 32, false, true, true) && register(0xB101, 4, 0x20_0000, 0, 32, false, true, true));
        let (a, b) = (slot(start(0xB100).unwrap()).unwrap(), slot(start(0xB101).unwrap()).unwrap());
        assert!(a.shared_srv != 0 && b.shared_srv != 0);
        assert_eq!(a.index.abs_diff(b.index), 8, "two heaps of 4 slots lie 8 descriptors apart");
        unregister(0xB100);
        unregister(0xB101);
    }

    /// A list's heaps when the title names its own (`list_x::pc_heaps`): its console view heap
    /// and a shader-visible PC heap outside the shared one. Only the shared heap is bound, and a
    /// dispatch through exactly that finds the runtime's memory views (`real::mem_views`) at
    /// io[`IO_MEM_VIEWS`] + k: window 3 reads back what the CPU wrote, and the view
    /// after the four windows is the null view – a store through it lands nowhere and a load
    /// reads 0 – which translated shaders select for an address outside the windows.
    #[test]
    fn title_heaps_keep_the_memory_views_bound() {
        use crate::gpu::recomp::{IO_MEM_VIEWS, KERNEL_SGPRS, MEM_NULL_VIEW};
        use windows::core::Interface;
        if crate::gpu::recomp::dxc::location().is_err() || super::super::real::init().is_err() {
            return;
        }
        let (Some(u), Some(views)) = (crate::gpu::uma::get(), super::super::real::mem_views()) else { return };
        unsafe {
            let raw = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).expect("device").clone();
            assert!(shared_heap(&dev));
            let title = 0xDEAD_4EA9usize;
            assert!(register(title, 4, 0x1000, 0, 32, false, true, true) && in_global(title));
            let foreign: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 16, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
                .unwrap();
            let named = [title, foreign.as_raw() as usize];
            let bound = super::super::list_x::pc_heaps(2, named.as_ptr());
            let shared = global_heap().unwrap();
            assert_eq!(bound.iter().map(|h| h.as_ref().map(|h| h.as_raw())).collect::<Vec<_>>(), [Some(shared.as_raw())], "only the shared heap is bound");

            let w3 = u.alloc(0x1000, 256).unwrap() as usize;
            *(w3 as *mut u32) = 0x89AB_CDEF;
            *((w3 + 4) as *mut u32) = 0xCAFE_BABE;
            let (io_buf, res_buf) = (u.alloc(0x1000, 256).unwrap() as usize, u.alloc(0x1000, 256).unwrap() as usize);
            let mut io = vec![0u32; KERNEL_SGPRS];
            io[IO_MEM_VIEWS] = views;
            io[1] = (w3 - u.window(3).0 as usize) as u32;
            io[2] = io[1] + 4;
            std::ptr::copy_nonoverlapping(io.as_ptr(), io_buf as *mut u32, io.len());
            std::ptr::write_bytes(res_buf as *mut u8, 0xFF, 16);
            let hlsl = format!(
                "RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n[numthreads(1, 1, 1)]\nvoid main()\n{{\n    uint v = io.Load({} * 4);\n    RWByteAddressBuffer w3 = ResourceDescriptorHeap[v + 3];\n    RWByteAddressBuffer none = ResourceDescriptorHeap[v + {}];\n    none.Store(io.Load(8), 0xBADu);\n    res.Store(4, w3.Load(io.Load(4)));\n    res.Store(8, none.Load(io.Load(8)));\n    res.Store(12, none.Load(0));\n}}\n",
                IO_MEM_VIEWS, MEM_NULL_VIEW
            );
            let dxil = crate::gpu::recomp::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
            let rs = crate::gpu::recomp::pso::abi_root_signature(&dev, false).unwrap();
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, super::super::real::uma_va_of(io_buf).unwrap());
            list.SetComputeRootUnorderedAccessView(1, super::super::real::uma_va_of(res_buf).unwrap());
            list.SetDescriptorHeaps(&bound);
            list.Dispatch(1, 1, 1);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            if fence.GetCompletedValue() < 1 {
                let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(1, ev).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
            let res = std::slice::from_raw_parts(res_buf as *const u32, 4);
            assert_eq!(res[1..], [0x89AB_CDEF, 0, 0], "window 3, the null view after a store, the null view at 0");
            assert_eq!(*((w3 + 4) as *const u32), 0xCAFE_BABE, "a store through the null view lands nowhere");
            unregister(title);
        }
    }
}
