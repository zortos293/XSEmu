//! Command queue CopyPageMappingsBatchX (the newer layout's #31, the older one's #29): the GPU's
//! view of physical pages. A title maps its physical pages into a pool (its CPU view, through
//! XMem), then has the queue map pool pages at GPU addresses – a streaming pool's pages into a
//! tiled heap, where its textures are. The GPU address then shows the physical page itself: the
//! title unmaps the pool slot right after (a retail game reuses its pool slots for the next
//! streaming request) and fills the page later (DirectStorage into physical pages, a video
//! decoder). As a stub the GPU addresses stayed unmapped, every texture there uploaded as zeros.
//!
//! `(count, batches, pool, pool pages, flags)`: `count` batches of 24 bytes (a destination
//! address, a range count, the ranges), each range 12 bytes (kind, first pool page, pages). Kind 0
//! maps pool pages from the first on, kind 1 the first pool page at each page, kind 2 unmaps, kind
//! 3 leaves the pages as they are; pages are 64 KiB. (Measured from a retail game's builder of
//! these batches – kind 1, one page each, and kind 2; it matches VIctory's model.)
//!
//! A GPU address in the title's reservations of 64 KiB pages becomes a view of the physical page
//! the pool slot shows (`kernelx::physical`): the same memory, wherever the title writes it. Views
//! are not write-watched, so the textures there are uploaded again when XCL wrote the page
//! (`kernelx::note_written`) and when a bounded comparison finds it changed: each submit's look
//! hashes up to `XCL_VIEW_SWEEP` pages (default 32) the GPU views show, in turn.
//!
//! Elsewhere (a GPU address that cannot hold a view, or a pool slot showing no physical page) the
//! view is a copy kept in step with what the slot showed – the physical page wherever its data is,
//! or the slot's memory: copied to the views when the look found that place written, or when it
//! moved (then compared once). A view that is unmapped keeps what it showed.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::util::{E_INVALIDARG, HRESULT, S_OK};

pub(super) mod queue;
mod pages;
use super::physical_directory::{self, Change, Mapping, Plane};

/// Paths where the GPU directory keeps what it had before (an entry inherits the canonical
/// view, or a version or reader is not ordered) while the guest call goes on as before.
#[derive(Clone, Copy)]
pub(super) enum Fallback {
    GuestRange,
    Unbindable,
    SourceRefused,
    Hole,
    SelfMapped,
    Unencodable,
    PrepareFailed,
    Unordered,
    PublishFailed,
    UnroutedReader,
    LaneMissing,
}

const FALLBACKS: [&str; 11] = [
    "map entry inherits: pool slot shows guest memory with no owned physical identity",
    "map entry inherits: physical page cannot be bound in the directory",
    "map entry inherits: pool slot source refused by the physical owner",
    "map entry inherits: pool slot shows no physical page",
    "map entry inherits: pool slot mapped onto itself",
    "map entry skipped: GPU address outside the 48-bit directory",
    "map version dropped: cannot be prepared",
    "map version dropped: queue not ordered",
    "map version dropped: native queue refused it",
    "reader submitted directly",
    "queue has no ordering lane",
];
static FALLBACK_N: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

pub(super) fn fallback(class: Fallback, reason: &str) {
    let n = FALLBACK_N[class as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: physical directory: {} ({reason}); {n} so far", FALLBACKS[class as usize]);
    }
}

/// The queue-ordered GPU directory version of one CopyPageMappingsBatchX, published on the
/// title's queue after the canonical CPU views were made.
pub(super) struct Update {
    version: physical_directory::PreparedVersion,
}

impl Update {
    /// None while the directory is inactive, the batch maps nothing, or it cannot be prepared.
    fn prepare(maps: &[(usize, Option<usize>)]) -> Option<Self> {
        use crate::hle::kernelx::{snapshot_queue_sources, QueuePageSource};
        if maps.is_empty() || physical_directory::current_topology().is_none() {
            return None;
        }
        let mut slots = Vec::new();
        let mut index = HashMap::new();
        for &(at, slot) in maps {
            if let Some(slot) = slot.filter(|&s| s != at) {
                index.entry(slot).or_insert_with(|| {
                    slots.push(slot);
                    slots.len() - 1
                });
            }
        }
        let sources: Vec<Result<QueuePageSource, &'static str>> = match snapshot_queue_sources(&slots) {
            Ok(all) => all.into_iter().map(Ok).collect(),
            Err(_) => slots.iter().map(|&s| snapshot_queue_sources(&[s]).map(|mut v| v.remove(0))).collect(),
        };
        let targets: Vec<Mapping> = sources
            .into_iter()
            .map(|source| match source {
                Ok(QueuePageSource::Physical(p)) => physical_directory::bind(&p).map_or_else(
                    |reason| {
                        fallback(Fallback::Unbindable, reason);
                        Mapping::Inherit
                    },
                    Mapping::Page,
                ),
                Ok(QueuePageSource::Hole) => {
                    fallback(Fallback::Hole, "canonical copied view of the slot");
                    Mapping::Inherit
                }
                Ok(QueuePageSource::GuestRange { .. }) => {
                    fallback(Fallback::GuestRange, "awaiting owned backing");
                    Mapping::Inherit
                }
                Err(reason) => {
                    fallback(Fallback::SourceRefused, reason);
                    Mapping::Inherit
                }
            })
            .collect();
        let mut changes = Vec::with_capacity(maps.len());
        for &(at, slot) in maps {
            if at >> 48 != 0 {
                fallback(Fallback::Unencodable, "not encodable");
                continue;
            }
            let target = match slot {
                None => Mapping::Unmapped,
                Some(s) if s == at => {
                    fallback(Fallback::SelfMapped, "no separate source");
                    Mapping::Inherit
                }
                Some(s) => targets[index[&s]].clone(),
            };
            changes.push(Change { guest: at as u64, plane: Plane::Gpu, target });
        }
        match physical_directory::prepare_version(changes) {
            Ok(version) => Some(Self { version }),
            Err(reason) => {
                fallback(Fallback::PrepareFailed, reason);
                None
            }
        }
    }
}

const PAGE: usize = 0x1_0000;

#[repr(C)]
pub(super) struct Range {
    kind: u32,
    start: u32,
    count: u32,
}

#[repr(C)]
pub(super) struct Batch {
    destination: u64,
    count: u32,
    ranges: *const Range,
}

/// What a copied view shows: a physical page (wherever it is), or – a pool slot that showed none
/// – the memory at that address.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Source {
    Page(usize),
    Address(usize),
}

#[derive(Default)]
struct Shown {
    /// where the source was when last copied to the views
    at: Option<usize>,
    views: BTreeSet<usize>,
}

/// The copied views.
#[derive(Default)]
struct Views {
    /// view page → what it shows
    shown: BTreeMap<usize, Source>,
    sources: HashMap<Source, Shown>,
    /// view pages written since the last submit's look
    written: Vec<usize>,
}

/// The views that are the physical page itself.
#[derive(Default)]
struct Gpu {
    /// view page → the physical page
    at: BTreeMap<usize, usize>,
    /// physical page → its views, and a hash of what it held when last compared
    pages: BTreeMap<usize, (Vec<usize>, u64)>,
    /// the next page to compare
    cursor: usize,
    /// views made since the last look (their textures are uploaded again)
    written: Vec<usize>,
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);
static GPU: Mutex<Option<Gpu>> = Mutex::new(None);
static ZERO: [u8; PAGE] = [0; PAGE];
/// copied view pages (no lock taken while there are none)
static SHOWN: AtomicUsize = AtomicUsize::new(0);
/// views that are the page itself
static VIEWED: AtomicUsize = AtomicUsize::new(0);

fn readable(a: usize) -> bool {
    crate::gpu::shader::readable(a, PAGE)
}

/// Whether the 64 KiB at `a` and at `b` hold the same, read as copies (None: one of them is
/// not there any more – a page compared where it is can be unmapped under the comparison).
fn same(a: usize, b: usize) -> Option<bool> {
    Some(crate::memmap::title_bytes(a, PAGE)? == crate::memmap::title_bytes(b, PAGE)?)
}

/// A hash of the 64 KiB at `a` (readable; read as a copy, what goes meanwhile as zeros).
unsafe fn hash(a: usize) -> u64 {
    let mut words = vec![0u64; PAGE / 8];
    crate::memmap::copy_title(a, std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), PAGE));
    let mut h = [0x243F_6A88_85A3_08D3u64, 0x1319_8A2E_0370_7344, 0xA409_3822_299F_31D0, 0x082E_FA98_EC4E_6C89];
    for c in words.chunks_exact(4) {
        for i in 0..4 {
            h[i] = (h[i] ^ c[i]).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        }
    }
    h[0] ^ h[1].rotate_left(16) ^ h[2].rotate_left(32) ^ h[3].rotate_left(48)
}

fn zero_hash() -> u64 {
    static H: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *H.get_or_init(|| unsafe { hash(ZERO.as_ptr() as usize) })
}

/// Where `source` is now.
fn place(held: &crate::hle::kernelx::Held, source: Source) -> Option<usize> {
    match source {
        Source::Page(p) => held.location(p),
        Source::Address(a) => Some(a),
    }
}

/// The view page `at` gets what `from` holds (zeros where nothing readable is).
unsafe fn copy_to(at: usize, from: Option<usize>) {
    if from != Some(at) {
        // (XCL's write into the title's memory)
        crate::alias::mapping_changed("a copied GPU page view is overwritten with what it shows", "CopyPageMappingsBatchX / a submit's look", at, PAGE);
        crate::eh::note_guest_write("copied GPU page view overwritten with what it shows", at, PAGE);
    }
    // (read as a copy and written by Windows: the title can unmap either page meanwhile)
    match from.filter(|&a| readable(a)) {
        Some(a) if a != at => {
            let mut page = vec![0u8; PAGE];
            crate::memmap::copy_title(a, &mut page);
            crate::memmap::write_title(at, &page);
        }
        Some(_) => {}
        None => {
            crate::memmap::write_title(at, &ZERO);
        }
    }
}

impl Views {
    fn unmap(&mut self, at: usize) {
        if let Some(source) = self.shown.remove(&at) {
            if let Some(s) = self.sources.get_mut(&source) {
                s.views.remove(&at);
                if s.views.is_empty() {
                    self.sources.remove(&source);
                    POOLS_GEN.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
    }
}

impl Gpu {
    fn insert(&mut self, at: usize, page: usize) {
        self.at.insert(at, page);
        self.pages.entry(page).or_insert_with(|| (Vec::new(), zero_hash())).0.push(at);
        self.written.push(at);
    }

    fn remove(&mut self, at: usize) {
        let Some(page) = self.at.remove(&at) else { return };
        if let Some((views, _)) = self.pages.get_mut(&page) {
            views.retain(|&v| v != at);
            if views.is_empty() {
                self.pages.remove(&page);
            }
        }
    }
}

/// CopyPageMappingsBatchX(count, batches, pool, pool pages, flags); the flags (a fifth argument
/// on the stack) only order the mapping against other queue work, which has all been submitted.
pub(super) unsafe fn copy_batch(queue: *mut std::ffi::c_void, count: u32, batches: *const Batch, pool: usize, pool_pages: u32) -> HRESULT {
    if count > 0 && batches.is_null() {
        return E_INVALIDARG;
    }
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("XCL_PAGE_MAPPINGS").is_ok_and(|v| v == "0")) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_info!("XCL_PAGE_MAPPINGS=0: CopyPageMappingsBatchX maps nothing"));
        return S_OK;
    }
    let mut maps: Vec<(usize, Option<usize>)> = Vec::new();
    let mut refused = 0;
    for b in (0..count as usize).map(|i| &*batches.add(i)) {
        if b.ranges.is_null() {
            continue;
        }
        let mut at = b.destination as usize & !(PAGE - 1);
        for r in (0..b.count as usize).map(|i| &*b.ranges.add(i)) {
            let n = r.count as usize;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = |k: usize| pool + (r.start as usize + k) * PAGE;
            if crate::hle::kernelx::trace_maps() {
                crate::xlog_info!("map trace: CopyPageMappingsBatchX kind {} {at:#x}..{:#x} ({n} pages) from pool page {} (pool {pool:#x}, {pool_pages} pages)", r.kind, at + n * PAGE, r.start);
            }
            match r.kind {
                0 if inside(r.count) => maps.extend((0..n).map(|k| (at + k * PAGE, Some(source(k))))),
                1 if inside(1) => maps.extend((0..n).map(|k| (at + k * PAGE, Some(source(0))))),
                2 => maps.extend((0..n).map(|k| (at + k * PAGE, None))),
                3 => {}
                _ => refused += 1,
            }
            at += n * PAGE;
        }
    }
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: queue CopyPageMappingsBatchX: {count} batches, pool {pool:#x} ({pool_pages} pages), first {} page mappings", maps.len()));
    let snapshot = maps.clone();
    let mut failed = 0;
    let (mut viewed, mut copied) = (0usize, 0usize);
    crate::hle::kernelx::hold_physical_pages(|held| {
        let mut g = VIEWS.lock().unwrap();
        let v = g.get_or_insert_with(Views::default);
        let mut gg = GPU.lock().unwrap();
        let gpu = gg.get_or_insert_with(Gpu::default);
        for (at, slot) in maps {
            v.unmap(at);
            gpu.remove(at);
            held.unmap(at);
            pages::gone(at);
            let Some(slot) = slot.filter(|&p| p != at) else { continue };
            let page = held.page_at(slot);
            if let Some(p) = page {
                held.note_source(at, slot, p);
            }
            if let Some(p) = page.filter(|&p| held.view(at, p)) {
                gpu.insert(at, p);
                viewed += 1;
                continue;
            }
            if !held.claim(at) {
                failed += 1;
                continue;
            }
            copied += 1;
            pages::placed(at);
            let source = page.map_or(Source::Address(slot), Source::Page);
            let now = place(held, source);
            copy_to(at, now);
            v.shown.insert(at, source);
            let s = v.sources.entry(source).or_insert_with(|| {
                POOLS_GEN.fetch_add(1, Ordering::AcqRel);
                Shown::default()
            });
            s.at = now;
            s.views.insert(at);
            v.written.push(at);
        }
        pages::evict(held);
        SHOWN.store(v.shown.len(), Ordering::Release);
        VIEWED.store(gpu.at.len(), Ordering::Release);
    });
    if refused + failed > 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("D3D12.X: CopyPageMappingsBatchX: {refused} ranges refused (kind or pool page out of range), {failed} pages could not be mapped (pool {pool:#x}, {pool_pages} pages)"));
    }
    if copied > 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_info!("D3D12.X: CopyPageMappingsBatchX: {copied} GPU view pages are copies ({viewed} views of the page itself in this batch): their address cannot hold a view or the pool slot shows no physical page"));
    }
    // after the canonical effects: a pool page whose data was in its copy is in its section now
    // that a GPU address shows it (`map_at`), and the directory names that section
    let directory = Update::prepare(&snapshot);
    if let Some(update) = directory {
        queue::publish(queue, update);
    }
    S_OK
}

/// The title maps something else at `[at, at + len)` (XMem physical pages): no longer a GPU view.
/// With the physical pages held.
pub(crate) fn forget(at: usize, len: usize) {
    let end = at.saturating_add(len);
    if SHOWN.load(Ordering::Acquire) > 0 {
        let mut g = VIEWS.lock().unwrap();
        if let Some(v) = g.as_mut() {
            let pages: Vec<usize> = v.shown.range(at..end).map(|(&a, _)| a).collect();
            for a in pages {
                v.unmap(a);
                pages::gone(a);
            }
            SHOWN.store(v.shown.len(), Ordering::Release);
        }
    }
    if VIEWED.load(Ordering::Acquire) > 0 {
        let mut g = GPU.lock().unwrap();
        if let Some(gpu) = g.as_mut() {
            let pages: Vec<usize> = gpu.at.range(at..end).map(|(&a, _)| a).collect();
            for a in pages {
                gpu.remove(a);
            }
            VIEWED.store(gpu.at.len(), Ordering::Release);
        }
    }
}

/// The GPU's views of the memory an address of a crash record shows (`alias::report`).
#[derive(Default)]
pub(crate) struct AliasFacts {
    /// what the address itself is to the GPU
    pub here: Option<String>,
    /// the other addresses (64 KiB aligned) that are the GPU's views of it, or that its copies are
    /// taken from, and as what
    pub aliases: Vec<(usize, &'static str)>,
}

/// For the 64 KiB page at `va`, which shows physical page `page` (None: none): whether it is a GPU
/// view, the views that are the page itself, and the copied views XCL keeps in step with the page
/// or with this address (XCL copies 64 KiB over each of those when its source changes). Locks
/// another thread holds are not waited for.
pub(crate) fn alias_facts(va: usize, page: Option<usize>) -> AliasFacts {
    let at = va & !(PAGE - 1);
    let mut f = AliasFacts::default();
    let mut here = Vec::new();
    if VIEWED.load(Ordering::Acquire) > 0 {
        match crate::alias::try_lock(&GPU) {
            Some(g) => {
                if let Some(gpu) = g.as_ref() {
                    if let Some(p) = gpu.at.get(&at) {
                        here.push(format!("a GPU view of physical page {p:#x} (the page itself)"));
                    }
                    if let Some((views, _)) = page.and_then(|p| gpu.pages.get(&p)) {
                        f.aliases.extend(views.iter().filter(|&&v| v != at).map(|&v| (v, "a GPU view (the page itself)")));
                    }
                }
            }
            None => here.push("(the GPU page views are locked by another thread)".to_string()),
        }
    }
    if SHOWN.load(Ordering::Acquire) > 0 {
        match crate::alias::try_lock(&VIEWS) {
            Some(g) => {
                if let Some(v) = g.as_ref() {
                    match v.shown.get(&at) {
                        Some(Source::Page(p)) => here.push(format!("a copied GPU view of physical page {p:#x}: XCL copies the page's 64 KiB over this address whenever the page changes")),
                        Some(&Source::Address(a)) => {
                            here.push(format!("a copied GPU view of the memory at {a:#x}: XCL copies those 64 KiB over this address whenever they change"));
                            f.aliases.push((a, "the address this copied GPU view is filled from"));
                        }
                        None => {}
                    }
                    let copied = |s: Source| v.sources.get(&s).into_iter().flat_map(|s| s.views.iter().copied()).filter(|&a| a != at).collect::<Vec<usize>>();
                    if let Some(p) = page {
                        f.aliases.extend(copied(Source::Page(p)).into_iter().map(|a| (a, "a copied GPU view of the page (XCL copies the page over it)")));
                    }
                    f.aliases.extend(copied(Source::Address(at)).into_iter().map(|a| (a, "a copied GPU view of this address (XCL copies this memory over it)")));
                }
            }
            None => here.push("(the copied GPU page views are locked by another thread)".to_string()),
        }
    }
    if !here.is_empty() {
        f.here = Some(format!("to the GPU this address is {}", here.join("; ")));
    }
    f
}

/// What the 64 KiB page at `va` is (diagnostics): a GPU view (of what, what both hold) or the
/// physical page model's account of it.
pub(super) fn describe(va: usize) -> String {
    let at = va & !(PAGE - 1);
    let nonzero = |a: usize| crate::memmap::title_bytes(a, PAGE).map_or("not committed".into(), |page| page.iter().filter(|&&b| b != 0).count().to_string());
    if let Some(p) = GPU.lock().unwrap().as_ref().and_then(|g| g.at.get(&at).copied()) {
        return format!("GPU view of page {p:#x} (the page itself): {} non-zero bytes", nonzero(at));
    }
    let source = VIEWS.lock().unwrap().as_ref().and_then(|v| v.shown.get(&at).copied());
    match source {
        Some(s) => {
            let now = unsafe { crate::hle::kernelx::hold_physical_pages(|held| place(held, s)) };
            format!("GPU view (copy) of {s:x?}, now at {now:x?}: non-zero bytes {} in the view, {} there", nonzero(at), now.map_or("-".into(), nonzero))
        }
        None => format!("not a GPU view; {}; {} non-zero bytes", crate::hle::kernelx::describe_address(at), nonzero(at)),
    }
}

/// The memory to be write-watched with the textures for the copied views: where each source they
/// show is now.
pub(super) fn pool_ranges() -> Vec<(usize, usize)> {
    if SHOWN.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let sources: Vec<Source> = VIEWS.lock().unwrap().as_ref().map_or(Vec::new(), |v| v.sources.keys().copied().collect());
    let places = unsafe { crate::hle::kernelx::hold_physical_pages(|held| sources.iter().filter_map(|&s| place(held, s)).collect::<Vec<_>>()) };
    places.into_iter().map(|a| (a, a + PAGE)).collect()
}

/// changes when `pool_ranges` does
pub(super) fn generation() -> u64 {
    POOLS_GEN.load(Ordering::Acquire)
}

static POOLS_GEN: AtomicU64 = AtomicU64::new(0);

/// A title submission was issued on `queue`: the placed pages of the look before it (`pages`) are
/// ordered after its fence.
pub(super) unsafe fn submitted(queue: *mut std::ffi::c_void) {
    pages::submitted(queue);
}

#[link(name = "kernel32")]
extern "system" {
    fn ResetWriteWatch(base: *const std::ffi::c_void, size: usize) -> u32;
}

/// pages compared per look (`XCL_VIEW_SWEEP`)
fn sweep_budget() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_VIEW_SWEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

/// The views that are the page itself: new ones, and those whose page changed since it was last
/// compared (up to `sweep_budget` pages, in turn), have their 4 KiB pages added to `dirty`.
/// (counts: pages compared, pages changed)
unsafe fn sweep(dirty: &mut Vec<usize>) -> (usize, usize) {
    if VIEWED.load(Ordering::Acquire) == 0 {
        return (0, 0);
    }
    let mut g = GPU.lock().unwrap();
    let Some(gpu) = g.as_mut() else { return (0, 0) };
    for at in std::mem::take(&mut gpu.written) {
        dirty.extend((at..at + PAGE).step_by(0x1000));
    }
    let budget = sweep_budget().min(gpu.pages.len());
    let start = gpu.cursor;
    let pages: Vec<usize> = gpu.pages.range(start..).map(|(&p, _)| p).chain(gpu.pages.range(..start).map(|(&p, _)| p)).take(budget).collect();
    let mut changed = 0;
    for &p in &pages {
        let (views, last) = gpu.pages.get_mut(&p).unwrap();
        let Some(&first) = views.first().filter(|&&a| readable(a)) else { continue };
        let h = hash(first);
        if h != *last {
            *last = h;
            changed += 1;
            for &at in views.iter() {
                dirty.extend((at..at + PAGE).step_by(0x1000));
            }
        }
    }
    if let Some(&last) = pages.last() {
        gpu.cursor = last + 1;
    }
    (pages.len(), changed)
}

/// After a submit's look for CPU writes (`dirty`: the written 4 KiB pages, sorted; `watched`:
/// the ranges looked at, sorted): the views of pages that changed, and the copied views whose
/// source changed (copied to them first), have their 4 KiB pages added to `dirty` (unsorted).
pub(super) unsafe fn after_look(dirty: &mut Vec<usize>, watched: &[(usize, usize)]) {
    let t0 = std::time::Instant::now();
    let found = dirty.len();
    let (swept, changed) = sweep(dirty);
    let (mut copied, mut compared, mut moved) = (0usize, 0usize, 0usize);
    let (mut views, mut sources, mut written_n) = (0, 0, 0);
    let mut used: Vec<usize> = Vec::new();
    if SHOWN.load(Ordering::Acquire) > 0 {
        let written_now = |at: usize, dirty: &[usize]| {
            let i = dirty[..found].partition_point(|&p| p < at);
            i < found && dirty[i] < at + PAGE
        };
        let is_watched = |at: usize| {
            let i = watched.partition_point(|&(base, size)| base + size <= at);
            watched.get(i).is_some_and(|&(base, size)| base <= at && at + PAGE <= base + size)
        };
        let (written, v_n, s_n, looked_at) = crate::hle::kernelx::hold_physical_pages(|held| {
            let mut g = VIEWS.lock().unwrap();
            let Some(v) = g.as_mut() else { return (Vec::new(), 0, 0, Vec::new()) };
            let mut written = std::mem::take(&mut v.written);
            for (&source, s) in v.sources.iter_mut() {
                let now = place(held, source);
                let changed = match now {
                    _ if now != s.at => {
                        moved += 1;
                        // moved: the same contents somewhere else, unless written meanwhile
                        match (now, s.views.iter().next()) {
                            (Some(a), Some(&first)) if readable(a) && readable(first) => {
                                compared += 1;
                                same(a, first) != Some(true)
                            }
                            (None, Some(&first)) if readable(first) => {
                                compared += 1;
                                crate::memmap::title_bytes(first, PAGE).is_none_or(|page| page[..] != ZERO[..])
                            }
                            _ => true,
                        }
                    }
                    Some(a) if is_watched(a) => written_now(a, dirty),
                    Some(a) => {
                        // a place outside the watched memory: compared with its first view
                        compared += 1;
                        s.views.iter().next().is_some_and(|&first| readable(a) && readable(first) && same(a, first) == Some(false))
                    }
                    None => false,
                };
                s.at = now;
                if !changed {
                    continue;
                }
                copied += 1;
                for &at in s.views.iter().filter(|&&a| readable(a)) {
                    copy_to(at, now);
                    written.push(at);
                }
            }
            let looked_at: Vec<usize> = if pages::on() { v.shown.keys().copied().collect() } else { Vec::new() };
            (written, v.shown.len(), v.sources.len(), looked_at)
        });
        if moved > 0 {
            // the places to watch changed (`pool_ranges`)
            POOLS_GEN.fetch_add(1, Ordering::AcqRel);
        }
        for &at in &written {
            // (the copies are this look's writes: not found again by the next one)
            ResetWriteWatch(at as *const std::ffi::c_void, PAGE);
            dirty.extend((at..at + PAGE).step_by(0x1000));
        }
        (views, sources, written_n) = (v_n, s_n, written.len());
        used = looked_at;
    }
    pages::look(&used);
    static LOOKS: AtomicUsize = AtomicUsize::new(0);
    if LOOKS.fetch_add(1, Ordering::Relaxed) % 2000 == 0 && VIEWED.load(Ordering::Relaxed) + SHOWN.load(Ordering::Relaxed) > 0 {
        crate::xlog_info!(
            "D3D12.X: GPU page views: {} views of the page itself ({swept} pages compared this look, {changed} changed); {views} copied views of {sources} pages / addresses ({moved} moved, {compared} compared, {copied} copied, {written_n} view pages written); in {:?}",
            VIEWED.load(Ordering::Relaxed),
            t0.elapsed()
        );
    }
}

#[cfg(test)] mod queue_order_test;
