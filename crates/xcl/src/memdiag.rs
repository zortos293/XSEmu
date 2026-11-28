//! Where the process's memory is, logged every 10 s as `memdiag:` lines (`XCL_MEMDIAG=0`: off).
//!
//! Every 10 s: the process counters (commit, working set, private and shareable), XCL's own Rust
//! heap (counted by the global allocator: live bytes, the peak since the last line as seen at
//! allocations of 64 KiB and more, the allocations of 64 MiB and more), the Windows heaps (HeapSummary: the process heap, which
//! XCL's Rust code, DXC and the CRT share, and every other heap, the driver's among them), the
//! D3D12 objects XCL saw the title make, by heap type (heaps and committed resources made on the
//! PC, buffer-only heaps given a unified-memory block as well), the title page segments the GPU
//! imports, the title's physical pages (allocated, live, committed, committed in freed pages) and
//! the entry counts of XCL's long-lived maps.
//!
//! Every 30 s (XCL_MEMDIAG_WALK=0: off) also a walk of the address space: committed private
//! memory by owner, mapped memory split into pagefile-backed sections, the driver's shader cache
//! files, the pipeline library and other files, and images; with an estimate of what of each is
//! resident and of that locked, which Windows cannot page out (QueryWorkingSetEx on one page in every
//! 256 KiB, XCL_MEMDIAG_RESIDENT=0: off). Each resident page sampled is also read, where its
//! protection allows (ReadProcessMemory on this process: a page gone meanwhile fails the read, and
//! a page not resident is never read, so nothing is paged in or made), and counted when all of it
//! is zeros (XCL_MEMDIAG_ZERO=0: off): committed, touched and empty memory. A page with more than
//! one mapping is counted too: the working set counts it once for each. The coherent window's
//! samples are also told apart by the block they are in (a line of its own): each block of
//! 32 MB and more with its zero pages, the console buffers and title textures placed in it and
//! the XCL code that asked for it, then the smaller blocks together and the memory in no block.
//!
//! Every second (XCL_MEMDIAG_POLL=0: off) the process heap less the Rust heap is noted under the
//! driver compiles running then, and the 10 s line gives its least, mean and most over the run
//! with no compile running, with no giant, with one and with more: what the heap holds when the
//! driver compiles nothing, and what a giant adds. The same line gives the Rust heap by
//! allocation size, the console shader copies alive and the pipeline stand-ins whose build (the
//! description and shaders it owns) is kept.
//!
//! Every 30 s, with the walk or without it: the title page segments imported as D3D12 heaps
//! against the pages bound in the physical directory, of those the pages the title freed, and
//! what XCL keeps at their GPU addresses (a texture copy in video memory, another texture it
//! placed there, neither); and XCL's own holders the size classes cannot name (kept builds,
//! external-buffer mirrors and seen copies, title-read scratch, sparse sent hashes, DirectStorage
//! read-ahead, the diagnostics rings). The walk's line is followed by what of its "the rest" is
//! thread stacks, other allocations with a guard page, the title's image and the other images
//! XCL's loader mapped. Once at the start: whether the Windows heap figures count a 32 MiB block.
//! The walk also names every other private allocation by its owner (XCL_MEMDIAG_OWNERS=0: off),
//! one line: the Windows heaps' segments (the NT heap's by their signature and its blocks from
//! VirtualAlloc by the list links and sizes at their start; the segment heap's, which XCL's
//! manifest asks for, by the word at +0x10 that is the same in all of them once their own
//! address is taken out), memory that can be executed, and the rest by the image most pointers
//! in up to nine of its resident pages go into (the driver, the D3D12 runtime, XCL, the title),
//! or none; by size, with the largest ones and what each reserves.
//! Nothing here changes what XCL does.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

const BIG: usize = 64 << 20;
/// allocations from which the peak is taken (one contended update per allocation, not two)
const PEAK_FROM: usize = 64 << 10;

/// The global allocator: System, with the bytes it holds counted.
pub struct Counting;

/// The live bytes, and of them by allocation size (`SIZES`), on one cache line: an allocation
/// writes two of them.
#[repr(C, align(64))]
struct Live {
    all: AtomicI64,
    sized: [AtomicI64; 5],
}

static LIVE: Live = Live { all: AtomicI64::new(0), sized: [const { AtomicI64::new(0) }; 5] };
static PEAK: AtomicI64 = AtomicI64::new(0);

const SIZES: [&str; 5] = ["under 1 KiB", "1-64 KiB", "64 KiB-1 MiB", "1-64 MiB", "64 MiB+"];

fn sized(n: usize) -> usize {
    match n {
        0..=0x3ff => 0,
        0x400..=0xffff => 1,
        0x1_0000..=0xf_ffff => 2,
        0x10_0000..=0x3ff_ffff => 3,
        _ => 4,
    }
}
static BIG_LIVE: AtomicI64 = AtomicI64::new(0);
static BIG_COUNT: AtomicI64 = AtomicI64::new(0);

fn grew(n: usize) {
    let now = LIVE.all.fetch_add(n as i64, Relaxed) + n as i64;
    LIVE.sized[sized(n)].fetch_add(n as i64, Relaxed);
    if n >= PEAK_FROM {
        PEAK.fetch_max(now, Relaxed);
    }
    if n >= BIG {
        BIG_LIVE.fetch_add(n as i64, Relaxed);
        BIG_COUNT.fetch_add(1, Relaxed);
    }
}

fn shrank(n: usize) {
    LIVE.all.fetch_sub(n as i64, Relaxed);
    LIVE.sized[sized(n)].fetch_sub(n as i64, Relaxed);
    if n >= BIG {
        BIG_LIVE.fetch_sub(n as i64, Relaxed);
        BIG_COUNT.fetch_sub(1, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        shrank(layout.size());
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, size);
        if !q.is_null() {
            shrank(layout.size());
            grew(size);
        }
        q
    }
}

/// A count of objects and the bytes they hold.
pub struct Tally {
    count: AtomicI64,
    bytes: AtomicI64,
}

impl Tally {
    pub const fn new() -> Tally {
        Tally { count: AtomicI64::new(0), bytes: AtomicI64::new(0) }
    }

    pub fn add(&self, bytes: u64) {
        self.count.fetch_add(1, Relaxed);
        self.bytes.fetch_add(bytes as i64, Relaxed);
    }

    pub fn sub(&self, bytes: u64) {
        self.count.fetch_sub(1, Relaxed);
        self.bytes.fetch_sub(bytes as i64, Relaxed);
    }

    fn show(&self) -> String {
        format!("{}/{} MB", self.count.load(Relaxed), self.bytes.load(Relaxed) >> 20)
    }

    /// (objects, bytes) as the line shows them, for the tests that watch a counter move.
    #[cfg(test)]
    pub fn counts(&self) -> (i64, i64) {
        (self.count.load(Relaxed), self.bytes.load(Relaxed))
    }
}

impl Default for Tally {
    fn default() -> Tally {
        Tally::new()
    }
}

const CLASSES: [&str; 6] = ["DEFAULT", "UPLOAD", "READBACK", "CUSTOM L0", "CUSTOM L1", "other"];

/// Heaps the title made, by heap_class (made, never counted down: XCL sees no heap's release).
pub static TITLE_HEAPS: [Tally; 6] = [const { Tally::new() }; 6];
/// Of them, heaps that hold only buffers and were also given a block of unified memory, where
/// their buffers are placed (real::heap_block).
pub static BUFFER_HEAPS_IN_UMA: Tally = Tally::new();
/// Of them, the title heaps whose PC heap is one alignment unit instead of the title's size
/// (`XCL_HEAP_LAZY`, real::lazy_placeholder) and whose full PC heap has not been made: the bytes,
/// and the commit behind them, that CreateHeap did not take. Alive now: a heap whose full PC heap
/// is made leaves for `LAZY_HEAP_MATERIALIZED`, and one the title releases leaves both.
pub static LAZY_HEAP_PLACEHOLDERS: Tally = Tally::new();
/// Of `LAZY_HEAP_PLACEHOLDERS`, the heaps whose full-size PC heap was made because something was
/// placed in them (real::materialize_lazy), alive now: the video memory and commit that came back.
pub static LAZY_HEAP_MATERIALIZED: Tally = Tally::new();
/// Committed resources the title made that reached the PC as committed resources, by class
/// (made, never counted down).
pub static TITLE_COMMITTED: [Tally; 6] = [const { Tally::new() }; 6];
/// Title page segments imported as D3D12 heaps and alive now (physical_import).
pub static IMPORTED_SEGMENTS: Tally = Tally::new();
/// Background builds whose job holds its description now (real::go_background): the bytes the
/// description and the job's text own, the shaders aside (each shared copy counted once by
/// real::diag_shader_copies).
pub static KEPT_BUILDS: Tally = Tally::new();
/// Threads whose scratch for reading the title's memory holds memory, and its bytes
/// (title_textures::with_title_bytes).
pub static TITLE_SCRATCH: Tally = Tally::new();

/// TITLE_HEAPS / TITLE_COMMITTED index of props: DEFAULT, UPLOAD, READBACK, CUSTOM in L0, in L1,
/// anything else.
pub fn heap_class(props: &windows::Win32::Graphics::Direct3D12::D3D12_HEAP_PROPERTIES) -> usize {
    use windows::Win32::Graphics::Direct3D12::*;
    match props.Type {
        D3D12_HEAP_TYPE_DEFAULT => 0,
        D3D12_HEAP_TYPE_UPLOAD => 1,
        D3D12_HEAP_TYPE_READBACK => 2,
        D3D12_HEAP_TYPE_CUSTOM if props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1 => 4,
        D3D12_HEAP_TYPE_CUSTOM => 3,
        _ => 5,
    }
}

/// A diagnostic that is on by default: off in a production build (`--features production`), where
/// the variable still turns it on. An alpha build measures exactly as before.
fn on(name: &str) -> bool {
    match std::env::var(name).ok().map(|v| v.trim().to_string()).as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => !cfg!(feature = "production"),
    }
}

/// Starts the thread that logs the lines (once).
pub fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !on("XCL_MEMDIAG") {
        return;
    }
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-memdiag".into()).spawn(|| {
            heap_check();
            let poll = on("XCL_MEMDIAG_POLL");
            let mut seen = [Seen::default(); 4];
            for tick in 1u64.. {
                for _ in 0..10 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    if poll {
                        note_heap(&mut seen);
                    }
                }
                report(poll.then_some(&seen));
                if tick % 3 == 0 {
                    if on("XCL_MEMDIAG_WALK") {
                        walk();
                    }
                    imports_line();
                    holders_line();
                }
            }
        });
    });
}

fn mb(b: u64) -> u64 {
    b >> 20
}

/// The process heap less the Rust heap as seen with some number of driver compiles running.
#[derive(Default, Clone, Copy)]
struct Seen {
    samples: u64,
    least: u64,
    most: u64,
    sum: u64,
}

const RUNNING: [&str; 4] = ["none", "no giant", "1 giant", "2+ giants"];

/// The process heap less XCL's Rust heap: DXC's and the driver's compiles among what else shares it
/// (`real::compile_rise`).
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn compile_heap() -> u64 {
    process_heap().saturating_sub(LIVE.all.load(Relaxed).max(0) as u64)
}

/// Committed bytes of the process heap.
fn process_heap() -> u64 {
    use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSummary, HEAP_SUMMARY};
    let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
    s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
    if unsafe { HeapSummary(GetProcessHeap(), 0, &mut s) } == 0 {
        return 0;
    }
    s.cbCommitted as u64
}

const PROBE: usize = 32 << 20;

/// Whether the Windows heap figures count a block of `PROBE` bytes, one made of its own pages
/// rather than in a heap segment: the process heap rose by most of it while it was held.
fn counts_large(before: u64, with: u64) -> bool {
    with.saturating_sub(before) >= PROBE as u64 / 2
}

/// Once at the start: the process heap before, with and after a touched `PROBE`-byte Rust
/// allocation. If large blocks are not counted, the Rust heap's large allocations are in the
/// walk's "the rest", not in its "process heap".
fn heap_check() {
    let before = process_heap();
    let probe = std::hint::black_box(vec![1u8; PROBE]);
    let with = process_heap();
    drop(probe);
    let after = process_heap();
    crate::xlog_info!(
        "memdiag: heap check: process heap committed {} MB before a 32 MiB Rust allocation, {} MB while it was held, {} MB after it was freed: large blocks are {}counted in the Windows heap figures",
        mb(before),
        mb(with),
        mb(after),
        if counts_large(before, with) { "" } else { "not " }
    );
}

fn note_heap(seen: &mut [Seen; 4]) {
    let (running, giants, _) = crate::hle::d3d12x::real::diag_driver_compiles();
    let heap = process_heap().saturating_sub(LIVE.all.load(Relaxed).max(0) as u64);
    let s = &mut seen[match (running, giants) {
        (0, _) => 0,
        (_, 0) => 1,
        (_, 1) => 2,
        _ => 3,
    }];
    s.least = if s.samples == 0 { heap } else { s.least.min(heap) };
    s.most = s.most.max(heap);
    s.sum += heap;
    s.samples += 1;
}

fn report(seen: Option<&[Seen; 4]>) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX2};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut p = PROCESS_MEMORY_COUNTERS_EX2 { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32, ..Default::default() };
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), std::ptr::from_mut(&mut p).cast(), p.cb) };
    let live = LIVE.all.load(Relaxed).max(0) as u64;
    let peak = PEAK.swap(live as i64, Relaxed).max(0) as u64;
    let (process_heap, others) = heaps();
    let largest: Vec<String> = others.iter().take(3).map(|b| format!("{} MB", mb(*b))).collect();
    crate::xlog_info!(
        "memdiag: process commit {} MB, working set {} MB ({} MB private, {} MB shareable); XCL Rust heap {} MB live (peak since the last line, at allocations of 64 KB+, {} MB; {} allocations of 64 MB+ holding {} MB); Windows heaps committed: process heap {} MB, {} other heaps {} MB (largest {})",
        mb(p.PrivateUsage as u64),
        mb(p.WorkingSetSize as u64),
        mb(p.PrivateWorkingSetSize as u64),
        mb(p.WorkingSetSize.saturating_sub(p.PrivateWorkingSetSize) as u64),
        mb(live),
        mb(peak),
        BIG_COUNT.load(Relaxed),
        BIG_LIVE.load(Relaxed) >> 20,
        mb(process_heap),
        others.len(),
        mb(others.iter().sum()),
        if largest.is_empty() { "none".to_string() } else { largest.join(", ") }
    );
    let classes = |t: &[Tally; 6]| CLASSES.iter().zip(t).map(|(n, t)| format!("{n} {}", t.show())).collect::<Vec<_>>().join(", ");
    let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
    crate::xlog_info!(
        "memdiag: D3D12: video memory {} MB local, {} MB non-local; title heaps made (count/MB): {}; lazy PC heaps still placeholders {} (their PC heap and the commit behind it never made), materialized {}; buffer-only heaps also given unified memory {}; title committed resources made on the PC: {}; imported title page segments alive {}; {}",
        mb(local),
        mb(non_local),
        classes(&TITLE_HEAPS),
        LAZY_HEAP_PLACEHOLDERS.show(),
        LAZY_HEAP_MATERIALIZED.show(),
        BUFFER_HEAPS_IN_UMA.show(),
        classes(&TITLE_COMMITTED),
        IMPORTED_SEGMENTS.show(),
        crate::gpu::rebar::summary()
    );
    if let Some(census) = crate::hle::d3d12x::tex_census::line() {
        crate::xlog_info!("memdiag: {census}");
    }
    let (live_psos, idle, again) = crate::hle::d3d12x::pso_async::live();
    crate::xlog_info!(
        "memdiag: title physical pages: {}; caches: compute blobs {} MB; pipelines live {live_psos} (released unbound {idle}, built again {again}); XCL maps (entries): {}",
        crate::hle::kernelx::physical_page_summary(),
        crate::hle::d3d12x::real::resident_compute_blob_bytes() >> 20,
        crate::hle::d3d12x::real::diag_map_sizes()
    );
    let (running, giants, reserved) = crate::hle::d3d12x::real::diag_driver_compiles();
    let by_running = seen.map_or_else(String::new, |seen| {
        let shown: Vec<String> = RUNNING
            .iter()
            .zip(seen)
            .map(|(name, s)| match s.samples {
                0 => format!("{name} -"),
                n => format!("{name} {}/{}/{} ({n})", mb(s.least), mb(s.sum / n), mb(s.most)),
            })
            .collect();
        format!("; process heap less the Rust heap by driver compiles running, MB least/mean/most over the run (1 s samples): {}", shown.join(", "))
    });
    let sizes: Vec<String> = SIZES.iter().zip(&LIVE.sized).map(|(name, bytes)| format!("{name} {} MB", bytes.load(Relaxed).max(0) >> 20)).collect();
    let (copies, copy_bytes) = crate::hle::d3d12x::real::diag_shader_copies();
    let (standins, kept, bound) = crate::hle::d3d12x::pso_async::diag_builds();
    crate::xlog_info!(
        "memdiag: driver compiles running {running} ({giants} giants, {} MiB reserved){by_running}; Rust heap by allocation size: {}; console shader copies {copies}/{} MB; pipeline stand-ins {standins} ({kept} with their build kept, {bound} built and bound)",
        reserved >> 20,
        sizes.join(", "),
        mb(copy_bytes)
    );
}

/// Committed bytes of the process heap and of every other heap, the others largest first.
fn heaps() -> (u64, Vec<u64>) {
    use windows_sys::Win32::System::Memory::{GetProcessHeap, GetProcessHeaps, HeapSummary, HEAP_SUMMARY};
    let mut handles = vec![std::ptr::null_mut(); 256];
    let n = unsafe { GetProcessHeaps(handles.len() as u32, handles.as_mut_ptr()) } as usize;
    handles.truncate(n.min(256));
    let main = unsafe { GetProcessHeap() };
    let (mut process_heap, mut others) = (0u64, Vec::new());
    for h in handles {
        let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
        s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
        if unsafe { HeapSummary(h, 0, &mut s) } == 0 {
            continue;
        }
        if h == main {
            process_heap = s.cbCommitted as u64;
        } else {
            others.push(s.cbCommitted as u64);
        }
    }
    others.sort_unstable_by(|a, b| b.cmp(a));
    (process_heap, others)
}

#[derive(Default, Clone, Copy)]
struct Part {
    committed: u64,
    sampled: u64,
    resident: u64,
    /// resident and locked: pinned (non-local video memory made resident is), so Windows cannot page it
    locked: u64,
    /// resident with more than one mapping (of this process or others): the working set counts each
    shared: u64,
    /// resident and read
    read: u64,
    /// of those, all zeros
    zero: u64,
}

const STRIDE: usize = 256 << 10;

/// The coherent window's blocks, by address, and what the samples in each showed.
struct WindowBlocks {
    /// (address, bytes)
    at: Vec<(usize, usize)>,
    /// for each block: resident pages read, and of those all zeros
    seen: Vec<(u64, u64)>,
    /// the same for the pages in no block
    free: (u64, u64),
}

impl WindowBlocks {
    fn note(&mut self, a: usize, zero: bool) {
        let i = self.at.partition_point(|b| b.0 <= a);
        let seen = match i.checked_sub(1).filter(|&i| a < self.at[i].0 + self.at[i].1) {
            Some(i) => &mut self.seen[i],
            None => &mut self.free,
        };
        seen.0 += 1;
        seen.1 += zero as u64;
    }
}

/// blocks of the coherent window shown one by one
const BLOCK_SHOWN: usize = 32 << 20;

fn window_line(blocks: &[(usize, usize, Option<&'static std::panic::Location<'static>>)], window: &WindowBlocks, start: usize, committed: u64) {
    let zero = |bytes: u64, (read, zero): (u64, u64)| if read > 0 { format!("~{}", mb(bytes * zero / read)) } else { "?".to_string() };
    let (buffers, textures) = crate::hle::d3d12x::real::diag_placed();
    let placed = |ranges: &[(usize, usize)], at: usize, len: usize| ranges.iter().map(|&(p, n)| (p + n).min(at + len).saturating_sub(p.max(at)) as u64).sum::<u64>();
    let mut shown: Vec<(usize, String)> = Vec::new();
    let (mut small, mut small_bytes, mut small_seen, mut in_blocks) = (0usize, 0u64, (0u64, 0u64), 0u64);
    for (&(at, len, by), &seen) in blocks.iter().zip(&window.seen) {
        in_blocks += len as u64;
        if len < BLOCK_SHOWN {
            small += 1;
            small_bytes += len as u64;
            small_seen = (small_seen.0 + seen.0, small_seen.1 + seen.1);
            continue;
        }
        let by = by.map_or_else(|| "?".to_string(), |l| format!("{}:{}", l.file().rsplit(['/', '\\']).next().unwrap_or(""), l.line()));
        shown.push((len, format!(
            "{} at {}: zero {}, buffers {}, textures {}, {by}",
            mb(len as u64),
            mb(at.saturating_sub(start) as u64),
            zero(len as u64, seen),
            mb(placed(&buffers, at, len)),
            mb(placed(&textures, at, len))
        )));
    }
    shown.sort_by(|a, b| b.0.cmp(&a.0));
    let more = shown.len().saturating_sub(24);
    shown.truncate(24);
    let free = committed.saturating_sub(in_blocks);
    crate::xlog_info!(
        "memdiag: coherent window blocks of 32 MB+ (MB at MB from its start: zero pages MB, console buffers placed MB, title textures placed MB, asked by): {}{}; {small} smaller blocks {} MB: zero {} MB; in no block {} MB: zero {} MB",
        if shown.is_empty() { "none".to_string() } else { shown.into_iter().map(|s| s.1).collect::<Vec<_>>().join("; ") },
        if more > 0 { format!("; and {more} more") } else { String::new() },
        mb(small_bytes),
        zero(small_bytes, small_seen),
        mb(free),
        zero(free, window.free)
    );
}

const NAMES: [&str; 11] = [
    "private: coherent window",
    "private: sparse windows",
    "private: title reservations",
    "private: write-combined (behind video memory)",
    "private: other (heaps, driver, title)",
    "mapped: sections (title pages, video memory)",
    "mapped: driver shader cache",
    "mapped: pipeline library",
    "mapped: other files",
    "mapped: views under 1 MB",
    "images",
];

/// The address-space walk: committed and (sampled) resident bytes by owner.
fn walk() {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEM_RESERVE};
    use windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let started = std::time::Instant::now();
    let uma = crate::gpu::uma::get();
    let resident = on("XCL_MEMDIAG_RESIDENT");
    let mut parts = [Part::default(); NAMES.len()];
    let stacks = thread_stacks();
    let (threads, reserved) = (stacks.len(), stacks.iter().map(|s| s.1 as u64).sum::<u64>());
    let main = crate::modules::main_module().map(|m| (m.base, m.size));
    let images: Vec<(usize, usize)> = crate::modules::mapped_modules().iter().map(|m| (m.base, m.size)).filter(|&m| Some(m) != main).collect();
    let places = Places { stacks: merged(stacks), title: merged(main), images: merged(images.iter().copied()) };
    let mut named = Named::default();
    let owners = on("XCL_MEMDIAG_OWNERS");
    let mut owned: Vec<Owned> = Vec::new();
    // (the allocation the walk is in and what of it is reserved only, before its first committed region)
    let mut ahead = (0usize, 0u64);
    let zero = resident && on("XCL_MEMDIAG_ZERO");
    let mut samples: Vec<(usize, usize, bool)> = Vec::new();
    let mut at = 0x1_0000usize;
    while at < 0x7FFF_FFFF_0000 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const std::ffi::c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.RegionSize == 0 {
            break;
        }
        let (base, size) = (m.BaseAddress as usize, m.RegionSize);
        at = base + size;
        if owners && m.State == MEM_RESERVE && m.Type == MEM_PRIVATE {
            let allocation = m.AllocationBase as usize;
            match owned.last_mut() {
                Some(o) if o.base == allocation => o.size += size as u64,
                _ => ahead = (allocation, if ahead.0 == allocation { ahead.1 } else { 0 } + size as u64),
            }
        }
        if m.State != MEM_COMMIT {
            continue;
        }
        let part = match m.Type {
            MEM_PRIVATE if uma.is_some_and(|u| u.is_sparse(base)) => 1,
            // (as a driver heap, `hle::d3d12x::uma_driver`, the window is a mapping)
            MEM_PRIVATE | MEM_MAPPED if uma.is_some_and(|u| u.is_committed(base)) => 0,
            MEM_PRIVATE if crate::hle::kernelx::in_graphics_reservation(base) || crate::hle::kernelx::in_reservation(base) || crate::memmap::reservation(base).is_some() => 2,
            MEM_PRIVATE if m.Protect & 0x600 != 0 => 3,
            MEM_PRIVATE => 4,
            MEM_MAPPED if size < 1 << 20 => 9,
            MEM_MAPPED => {
                let mut name = [0u16; 512];
                let n = unsafe { K32GetMappedFileNameW(GetCurrentProcess(), base as *const _, name.as_mut_ptr(), name.len() as u32) } as usize;
                if n == 0 {
                    5
                } else {
                    let name = String::from_utf16_lossy(&name[..n]).to_ascii_lowercase();
                    if name.contains("dxcache") || name.contains("nv_cache") || name.contains("glcache") {
                        6
                    } else if name.contains("\\library\\") && name.contains("library-") {
                        7
                    } else {
                        8
                    }
                }
            }
            MEM_IMAGE => 10,
            _ => 4,
        };
        parts[part].committed += size as u64;
        if part == 4 && !named.note(&places, base, size, m.AllocationBase as usize, m.Protect & 0x100 != 0) && owners {
            owned_note(&mut owned, ahead, m.AllocationBase as usize, base, size, m.Protect);
        }
        if resident {
            // (readable, and no guard page: a read would take the guard away)
            let readable = zero && m.Protect & 0x100 == 0 && m.Protect & 0xEE != 0;
            let mut a = base.next_multiple_of(STRIDE);
            while a < base + size {
                samples.push((a, part, readable));
                a += STRIDE;
            }
        }
    }
    let blocks = if zero { crate::gpu::uma::diag_blocks() } else { Vec::new() };
    let mut window = WindowBlocks { at: blocks.iter().map(|b| (b.0, b.1)).collect(), seen: vec![(0, 0); blocks.len()], free: (0, 0) };
    if resident {
        sample_resident(&samples, &mut parts, &mut window);
    }
    let shown: Vec<String> = NAMES
        .iter()
        .zip(parts)
        .filter(|(_, p)| p.committed > 0)
        .map(|(n, p)| {
            if p.sampled > 0 {
                let resident = p.committed * p.resident / p.sampled;
                let zero = if p.read > 0 { format!(", zero pages ~{} MB of it", mb(resident * p.zero / p.read)) } else { String::new() };
                let shared = if p.shared > 0 { format!(", ~{} MB of it mapped more than once", mb(p.committed * p.shared / p.sampled)) } else { String::new() };
                format!("{n} {} MB (resident ~{} MB{zero}{shared}, locked ~{} MB)", mb(p.committed), mb(resident), mb(p.committed * p.locked / p.sampled))
            } else {
                format!("{n} {} MB", mb(p.committed))
            }
        })
        .collect();
    let (process_heap, others) = heaps();
    let heaps = process_heap + others.iter().sum::<u64>();
    crate::xlog_info!(
        "memdiag: address space (committed{}{}): {}; of private other: process heap {} MB, other heaps {} MB, the rest {} MB; walk {} ms",
        if resident { ", resident sampled every 256 KiB" } else { "" },
        if zero { ", zero pages among the resident ones read" } else { "" },
        shown.join("; "),
        mb(process_heap),
        mb(heaps - process_heap),
        mb(parts[4].committed.saturating_sub(heaps)),
        started.elapsed().as_millis()
    );
    let rest = parts[4].committed.saturating_sub(heaps);
    crate::xlog_info!(
        "memdiag: the rest of private other {} MB: thread stacks {} MB committed of {} MB reserved ({threads} threads); other allocations with a guard page as a stack has (fiber stacks and the like) {} MB; the title's image {} MB; {} other images XCL's loader mapped {} MB; still unnamed {} MB",
        mb(rest),
        mb(named.stacks),
        mb(reserved),
        mb(named.guarded),
        mb(named.title),
        images.len(),
        mb(named.images),
        mb(rest.saturating_sub(named.stacks + named.guarded + named.title + named.images))
    );
    if let Some(start) = uma.filter(|_| zero).and_then(|u| u.committed_window().map(|k| u.window(k).0)) {
        window_line(&blocks, &window, start, parts[0].committed);
    }
    if owners {
        owners_line(&owned, heaps);
    }
}

/// `(start, length)` ranges as sorted `(start, end)` ranges that neither overlap nor touch.
fn merged(ranges: impl IntoIterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut all: Vec<(usize, usize)> = ranges.into_iter().filter(|r| r.1 > 0).map(|(at, n)| (at, at.saturating_add(n))).collect();
    all.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(all.len());
    for (at, end) in all {
        match out.last_mut() {
            Some(last) if at <= last.1 => last.1 = last.1.max(end),
            _ => out.push((at, end)),
        }
    }
    out
}

/// Bytes of `[at, at + len)` in `ranges` (`merged`).
fn covered(ranges: &[(usize, usize)], at: usize, len: usize) -> u64 {
    let end = at.saturating_add(len);
    let first = ranges.partition_point(|r| r.1 <= at);
    ranges[first..].iter().take_while(|r| r.0 < end).map(|r| (r.1.min(end) - r.0.max(at)) as u64).sum()
}

/// The threads' stacks as `(start, length)`: from each thread's TEB, the bottom of its
/// reservation (DeallocationStack, +0x1478) to its base (NT_TIB.StackBase, +0x8), read with
/// ReadProcessMemory (a thread gone meanwhile fails the read and is left out).
fn thread_stacks() -> Vec<(usize, usize)> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let read = |at: usize| {
        let (mut v, mut done) = (0usize, 0usize);
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const _, std::ptr::from_mut(&mut v).cast(), 8, &mut done) } != 0;
        (ok && done == 8).then_some(v)
    };
    unsafe { crate::tls::thread_tebs() }
        .into_iter()
        .filter_map(|teb| {
            let teb = teb as usize;
            let (base, bottom) = (read(teb + 0x8)?, read(teb + 0x1478)?);
            (bottom != 0 && bottom < base).then_some((bottom, base - bottom))
        })
        .collect()
}

/// Where the walk names private memory: the threads' stacks, the title's image and the other
/// images XCL's loader mapped (each `merged`).
struct Places {
    stacks: Vec<(usize, usize)>,
    title: Vec<(usize, usize)>,
    images: Vec<(usize, usize)>,
}

/// What of the walk's private other memory is named, the Windows heaps aside.
#[derive(Default)]
struct Named {
    stacks: u64,
    /// in allocations with a guard page, as a stack has, that are no thread's stack (fibers')
    guarded: u64,
    /// the allocation of the last such guard page
    guard_base: usize,
    title: u64,
    images: u64,
}

impl Named {
    /// A committed region of private other memory: `[base, base + len)` in the allocation at
    /// `allocation`, `guard`: it is a guard page. The walk goes up the address space, and a
    /// stack's guard page lies below its committed part.
    /// Whether a place (a stack, an image) names it.
    fn note(&mut self, places: &Places, base: usize, len: usize, allocation: usize, guard: bool) -> bool {
        let stack = covered(&places.stacks, base, len);
        let title = covered(&places.title, base, len);
        let image = covered(&places.images, base, len);
        self.stacks += stack;
        self.title += title;
        self.images += image;
        if stack + title + image > 0 {
            return true;
        }
        if guard {
            self.guard_base = allocation;
        }
        if allocation != 0 && allocation == self.guard_base {
            self.guarded += len as u64;
        }
        false
    }
}

/// a page of the physical directory
const DIRECTORY_PAGE: usize = 64 << 10;

/// The pages bound in the physical directory, by what XCL keeps at their GPU addresses.
#[derive(Default, Debug, PartialEq, Eq)]
struct Bound {
    /// GPU pages that name a page
    mapped: u64,
    /// the pages they name, each once
    pages: u64,
    /// of those, pages the title has freed
    freed: u64,
    /// of those, by the first that any of the page's GPU addresses lies under: a title texture
    /// XCL keeps a video-memory copy of, another texture XCL placed in video memory, neither
    under: [u64; 3],
}

/// `at`: (page, GPU address) of each GPU page that names one; `allocated`: whether every
/// page given is allocated still; `copied`, `placed`: the textures as (address, bytes).
fn sort_bound(mut at: Vec<(usize, u64)>, allocated: impl Fn(&[usize]) -> bool, copied: &[(usize, usize)], placed: &[(usize, usize)]) -> Bound {
    let (copied, placed) = (merged(copied.iter().copied()), merged(placed.iter().copied()));
    at.sort_unstable();
    let mut b = Bound { mapped: at.len() as u64, ..Bound::default() };
    let mut pages = Vec::new();
    for group in at.chunk_by(|x, y| x.0 == y.0) {
        let class = group
            .iter()
            .map(|&(_, va)| match va as usize {
                va if covered(&copied, va, DIRECTORY_PAGE) > 0 => 0,
                va if covered(&placed, va, DIRECTORY_PAGE) > 0 => 1,
                _ => 2,
            })
            .min()
            .unwrap_or(2);
        b.under[class] += 1;
        pages.push(group[0].0);
    }
    b.pages = pages.len() as u64;
    b.freed = pages.chunks(256).filter(|c| !allocated(c)).map(|c| c.iter().filter(|&&p| !allocated(&[p])).count() as u64).sum();
    b
}

/// The imported title page segments against the pages bound in the physical directory (a
/// snapshot of its current version, no lock held while it is looked at), and what XCL keeps at
/// the bound pages' GPU addresses. Imported bytes bound at no GPU page are the neighbours an
/// import took in with a bound page, and pages only an older version still names.
fn imports_line() {
    let started = std::time::Instant::now();
    let (segments, imported) = (IMPORTED_SEGMENTS.count.load(Relaxed), IMPORTED_SEGMENTS.bytes.load(Relaxed).max(0) as u64);
    let Some((at, copied, placed)) = crate::hle::d3d12x::real::diag_imports() else {
        crate::xlog_info!("memdiag: imports: title page segments imported {segments} holding {} MB; no physical directory", mb(imported));
        return;
    };
    let b = sort_bound(at, crate::hle::kernelx::pages_allocated, &copied, &placed);
    let bytes = |pages: u64| pages * DIRECTORY_PAGE as u64;
    crate::xlog_info!(
        "memdiag: imports: title page segments imported {segments} holding {} MB; pages bound in the physical directory {} ({} MB at 64 KiB, at {} GPU pages), {} of them ({} MB) pages the title has freed; bound MB under title textures XCL keeps a video-memory copy of {}, under other textures XCL placed in video memory (GPU-owned targets) {}, under neither {}; imported and bound at no GPU page {} MB; {} ms",
        mb(imported),
        b.pages,
        mb(bytes(b.pages)),
        b.mapped,
        b.freed,
        mb(bytes(b.freed)),
        mb(bytes(b.under[0])),
        mb(bytes(b.under[1])),
        mb(bytes(b.under[2])),
        mb(imported.saturating_sub(bytes(b.pages))),
        started.elapsed().as_millis()
    );
}

/// XCL's own holders of memory that the Rust heap's size classes cannot name.
fn holders_line() {
    let started = std::time::Instant::now();
    let (_, shader_bytes) = crate::hle::d3d12x::real::diag_shader_copies();
    let (mirrors, mirrored, seen) = crate::hle::d3d12x::real::diag_external_buffers();
    let (sent_pages, sent_bytes) = crate::hle::d3d12x::sparse::diag_sent();
    let ring = |(entries, each, held): (usize, usize, usize)| format!("{entries} x {each} B ({} MB held)", mb(held as u64));
    crate::xlog_info!(
        "memdiag: XCL holders: kept pipeline builds (count/MB) {} for their descriptions, {} MB for their shaders (each shared copy once); external buffers {mirrors} mirrors of {} MB in the sparse windows, seen copies {} MB; title-read scratch (threads/MB) {}; sparse sent-chunk hashes {} MB ({sent_pages} pages); DirectStorage read-ahead holds {} MB; diagnostics rings: DirectStorage payloads {}, physical map history {}, unified-memory block history {}; {} ms",
        KEPT_BUILDS.show(),
        mb(shader_bytes),
        mb(mirrored as u64),
        mb(seen as u64),
        TITLE_SCRATCH.show(),
        mb(sent_bytes as u64),
        mb(crate::hle::dstorage::pool::diag_held()),
        ring(crate::alias::diag_payloads()),
        ring(crate::hle::kernelx::diag_map_history()),
        ring(crate::gpu::uma::diag_history()),
        started.elapsed().as_millis()
    );
}

fn sample_resident(samples: &[(usize, usize, bool)], parts: &mut [Part], window: &mut WindowBlocks) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut page = [0u8; 4096];
    for chunk in samples.chunks(16384) {
        let mut info: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = chunk
            .iter()
            .map(|&(a, _, _)| {
                let mut i: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
                i.VirtualAddress = a as *mut _;
                i
            })
            .collect();
        let bytes = (info.len() * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr().cast(), bytes) } == 0 {
            return;
        }
        for (i, &(a, part, readable)) in info.iter().zip(chunk) {
            parts[part].sampled += 1;
            let flags = unsafe { i.VirtualAttributes.Flags };
            if flags & 1 != 0 {
                parts[part].resident += 1;
                if flags & (1 << 22) != 0 {
                    parts[part].locked += 1;
                }
                // (ShareCount, bits 1 to 3)
                if (flags >> 1) & 7 > 1 {
                    parts[part].shared += 1;
                }
                let mut done = 0usize;
                if readable && unsafe { ReadProcessMemory(GetCurrentProcess(), a as *const _, page.as_mut_ptr().cast(), page.len(), &mut done) } != 0 && done == page.len() {
                    parts[part].read += 1;
                    let zero = page.iter().all(|&b| b == 0);
                    parts[part].zero += zero as u64;
                    if part == 0 {
                        window.note(a, zero);
                    }
                }
            }
        }
    }
}

/// Pages asked for of one allocation at most: the first page of its first readable region, the
/// page at that region's middle, and the first page of `SPOTS` later regions.
const SPOTS: usize = 7;

/// A private allocation no place names: its committed bytes, its whole size, and where the
/// owners line reads it.
struct Owned {
    base: usize,
    committed: u64,
    /// committed and reserved
    size: u64,
    /// `(start, length)` of the first committed region that can be read, `(0, 0)`: none can
    first: (usize, usize),
    /// the starts of later readable regions: the 2nd, 3rd, 5th, 9th and so on (0: none)
    spots: [usize; SPOTS],
    /// readable regions so far
    readable: u32,
    /// some of it can be executed
    code: bool,
}

/// A committed region `[base, base + len)` of the allocation at `allocation` (the walk goes up
/// the address space, so an allocation's regions follow one another). `ahead`: an allocation
/// and what the walk found reserved of it so far.
fn owned_note(owned: &mut Vec<Owned>, ahead: (usize, u64), allocation: usize, base: usize, len: usize, protect: u32) {
    if !matches!(owned.last(), Some(o) if o.base == allocation) {
        let size = if ahead.0 == allocation { ahead.1 } else { 0 };
        owned.push(Owned { base: allocation, committed: 0, size, first: (0, 0), spots: [0; SPOTS], readable: 0, code: false });
    }
    let o = owned.last_mut().unwrap();
    o.committed += len as u64;
    o.size += len as u64;
    o.code |= protect & 0xF0 != 0;
    // (readable, and no guard page: a read would take the guard away)
    if protect & 0x100 == 0 && protect & 0xEE != 0 {
        if o.readable == 0 {
            o.first = (base, len);
        } else if o.readable.is_power_of_two() {
            if let Some(spot) = o.spots.get_mut(o.readable.trailing_zeros() as usize) {
                *spot = base;
            }
        }
        o.readable += 1;
    }
}

/// The pages the owners line asks for of an allocation (0: none; one page is asked for once).
fn owned_asks(o: &Owned) -> [usize; SPOTS + 2] {
    let mut at = [0usize; SPOTS + 2];
    if o.first.1 != 0 {
        at[0] = o.first.0;
        let middle = o.first.0 + (o.first.1 / 2 & !0xFFF);
        at[1] = if middle == o.first.0 { 0 } else { middle };
        at[2..].copy_from_slice(&o.spots);
    }
    at
}

/// The segment heap's segments among `starts`, each an allocation's base and the word at +0x10
/// of it: a segment keeps there a signature made of its own address and of values every
/// segment of the heap shares, so with the address taken out many allocations show one value.
/// The bases of those that share theirs with `SHARED` and more, sorted.
fn heap_segments(starts: &[(usize, usize)]) -> Vec<usize> {
    const SHARED: usize = 8;
    let mut keys: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(base, word) in starts {
        *keys.entry(base ^ word).or_default() += 1;
    }
    let mut out: Vec<usize> = starts.iter().filter(|&&(base, word)| keys[&(base ^ word)] >= SHARED).map(|s| s.0).collect();
    out.sort_unstable();
    out
}

/// Whom a private allocation belongs to, as far as a page or two of it tell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Owner {
    /// a segment of a Windows heap: the NT heap's signature at +0x10, or `heap_segments`
    HeapSegment,
    /// a block a Windows heap got from VirtualAlloc: its first two words link it to other
    /// allocations or to a heap, and the two sizes after them fit it
    HeapBlock,
    /// memory that can be executed
    Code,
    /// most pointers in the pages read go into this image (its place in the sorted list)
    Image(usize),
    Zero,
    /// no two pointers into one image
    Plain,
    /// none of the pages asked for was resident, so none was read
    Away,
}

const OWNED_SIZES: [&str; 4] = ["under 1 MiB", "1-16 MiB", "16-64 MiB", "64 MiB+"];

fn owned_size(bytes: u64) -> usize {
    match bytes >> 20 {
        0 => 0,
        1..=15 => 1,
        16..=63 => 2,
        _ => 3,
    }
}

/// The owner of an allocation of `committed` bytes by `pages` of it (the first is its first
/// page when `at_base`). `segment`: `heap_segments` found it; `bases`: the bases of the
/// private allocations, sorted; `heaps`: the Windows heaps' handles; `images`: `(start, end)`
/// of every image, sorted.
#[allow(clippy::too_many_arguments)]
fn owner(pages: &[&[u8]], at_base: bool, segment: bool, code: bool, committed: u64, bases: &[usize], heaps: &[usize], images: &[(usize, usize)]) -> Owner {
    if segment {
        return Owner::HeapSegment;
    }
    let Some(first) = pages.first() else {
        return Owner::Away;
    };
    let word = |p: &[u8], i: usize| usize::from_le_bytes(p[8 * i..8 * i + 8].try_into().unwrap());
    if at_base && first.len() >= 48 {
        let signature = word(first, 2) as u32;
        if signature == 0xFFEE_FFEE || signature == 0xDDEE_DDEE {
            return Owner::HeapSegment;
        }
        let linked = |p: usize| bases.binary_search(&p).is_ok() || heaps.iter().any(|&h| p >= h && p - h < 0x1000);
        let (commit, reserve) = (word(first, 4), word(first, 5));
        let sized = commit != 0 && commit % 0x1000 == 0 && reserve % 0x1000 == 0 && commit <= reserve && (commit as u64).abs_diff(committed) <= 0x1_0000;
        if sized && linked(word(first, 0)) && linked(word(first, 1)) {
            return Owner::HeapBlock;
        }
    }
    if code {
        return Owner::Code;
    }
    if pages.iter().all(|p| p.iter().all(|&b| b == 0)) {
        return Owner::Zero;
    }
    let mut votes: Vec<(usize, u32)> = Vec::new();
    for p in pages {
        for i in 0..p.len() / 8 {
            let w = word(p, i);
            let k = images.partition_point(|m| m.1 <= w);
            if images.get(k).is_some_and(|m| m.0 <= w) {
                match votes.iter_mut().find(|v| v.0 == k) {
                    Some(v) => v.1 += 1,
                    None => votes.push((k, 1)),
                }
            }
        }
    }
    match votes.iter().max_by_key(|v| v.1) {
        Some(&(k, n)) if n >= 2 => Owner::Image(k),
        _ => Owner::Plain,
    }
}

/// Count and bytes of the allocations of one owner, by size (`OWNED_SIZES`).
type Sizes = [(u64, u64); 4];

/// "count/MB" of an owner, with its sizes when it holds 64 MB and more.
fn sizes_text(sizes: &Sizes) -> String {
    let (count, bytes) = sizes.iter().fold((0, 0), |a, s| (a.0 + s.0, a.1 + s.1));
    if bytes < 64 << 20 {
        return format!("{count}/{} MB", mb(bytes));
    }
    let split: Vec<String> = OWNED_SIZES.iter().zip(sizes).filter(|(_, s)| s.0 > 0).map(|(n, s)| format!("{n} {}/{} MB", s.0, mb(s.1))).collect();
    format!("{count}/{} MB ({})", mb(bytes), split.join(", "))
}

/// Every image of the process as sorted `(start, end)` and its name: the modules Windows
/// loaded and the ones XCL's loader mapped.
fn images() -> (Vec<(usize, usize)>, Vec<String>) {
    use windows_sys::Win32::System::ProcessStatus::{K32EnumProcessModules, K32GetModuleBaseNameW, K32GetModuleInformation, MODULEINFO};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut all: Vec<(usize, usize, String)> = Vec::new();
    let mut handles = vec![std::ptr::null_mut(); 1024];
    let mut bytes = 0u32;
    if unsafe { K32EnumProcessModules(GetCurrentProcess(), handles.as_mut_ptr(), (handles.len() * std::mem::size_of::<usize>()) as u32, &mut bytes) } != 0 {
        for &h in handles.iter().take(bytes as usize / std::mem::size_of::<usize>()) {
            let mut info: MODULEINFO = unsafe { std::mem::zeroed() };
            if unsafe { K32GetModuleInformation(GetCurrentProcess(), h, &mut info, std::mem::size_of::<MODULEINFO>() as u32) } == 0 {
                continue;
            }
            let mut name = [0u16; 128];
            let n = unsafe { K32GetModuleBaseNameW(GetCurrentProcess(), h, name.as_mut_ptr(), name.len() as u32) } as usize;
            let start = info.lpBaseOfDll as usize;
            all.push((start, start + info.SizeOfImage as usize, String::from_utf16_lossy(&name[..n.min(name.len())]).to_ascii_lowercase()));
        }
    }
    let main = crate::modules::main_module().map(|m| m.base);
    all.extend(crate::modules::main_module().iter().chain(crate::modules::mapped_modules().iter()).map(|m| {
        (m.base, m.base + m.size, if Some(m.base) == main { "the title's image".to_string() } else { format!("{} (the title's)", m.name) })
    }));
    all.sort_unstable_by_key(|m| m.0);
    all.dedup_by_key(|m| m.0);
    all.into_iter().map(|m| ((m.0, m.1), m.2)).unzip()
}

/// The owners line: the private allocations no place names, by owner. `heaps`: what the Windows
/// heaps report as committed, to hold against the heaps' part here. Only resident pages are
/// read, so nothing is paged in or made.
fn owners_line(owned: &[Owned], heaps: u64) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Memory::GetProcessHeaps;
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const SHOWN: usize = 6;
    let started = std::time::Instant::now();
    let (images, names) = images();
    let mut handles = vec![std::ptr::null_mut(); 256];
    let n = unsafe { GetProcessHeaps(handles.len() as u32, handles.as_mut_ptr()) } as usize;
    let handles: Vec<usize> = handles.iter().take(n).map(|&h| h as usize).collect();
    let bases: Vec<usize> = owned.iter().map(|o| o.base).collect();
    let mut by: std::collections::HashMap<Owner, Sizes> = std::collections::HashMap::new();
    let mut largest: Vec<(u64, u64, Owner)> = Vec::new();
    let resident = |at: &[usize]| -> Vec<bool> {
        let mut info: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = at
            .iter()
            .map(|&a| {
                let mut i: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
                i.VirtualAddress = a as *mut _;
                i
            })
            .collect();
        let bytes = (info.len() * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        let known = unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr().cast(), bytes) } != 0;
        at.iter().zip(&info).map(|(&a, i)| known && a != 0 && unsafe { i.VirtualAttributes.Flags } & 1 != 0).collect()
    };
    let read = |at: usize, to: &mut [u8]| {
        let mut done = 0usize;
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const _, to.as_mut_ptr().cast(), to.len(), &mut done) };
        ok != 0 && done == to.len()
    };
    // (a segment of the segment heap is 1 MiB or 16 MiB; any power of two from 1 MiB to 64 MiB is looked at)
    let candidates: Vec<usize> = owned.iter().filter(|o| o.first.0 == o.base && o.size.is_power_of_two() && (1 << 20..=64 << 20).contains(&o.size)).map(|o| o.base).collect();
    let mut starts: Vec<(usize, usize)> = Vec::new();
    for chunk in candidates.chunks(16384) {
        for (&base, _) in chunk.iter().zip(resident(chunk)).filter(|x| x.1) {
            let mut word = [0u8; 8];
            if read(base + 0x10, &mut word) {
                starts.push((base, usize::from_le_bytes(word)));
            }
        }
    }
    let segments = heap_segments(&starts);
    let mut in_segments = 0u64;
    let mut pages = vec![[0u8; 4096]; SPOTS + 2];
    for chunk in owned.chunks(1024) {
        let asked: Vec<usize> = chunk.iter().flat_map(owned_asks).collect();
        let there = resident(&asked);
        for (k, o) in chunk.iter().enumerate() {
            let (mut got, mut at_base) = (0usize, false);
            for j in k * (SPOTS + 2)..(k + 1) * (SPOTS + 2) {
                if there[j] && read(asked[j], &mut pages[got]) {
                    at_base |= got == 0 && asked[j] == o.base;
                    got += 1;
                }
            }
            let seen: Vec<&[u8]> = pages[..got].iter().map(|p| &p[..]).collect();
            let who = owner(&seen, at_base, segments.binary_search(&o.base).is_ok(), o.code, o.committed, &bases, &handles, &images);
            let size = &mut by.entry(who).or_default()[owned_size(o.committed)];
            *size = (size.0 + 1, size.1 + o.committed);
            if matches!(who, Owner::HeapSegment | Owner::HeapBlock) {
                in_segments += o.committed;
            } else {
                largest.push((o.committed, o.size, who));
                largest.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                largest.truncate(SHOWN);
            }
        }
    }
    let name = |who: Owner| match who {
        Owner::HeapSegment => "Windows heap segments".to_string(),
        Owner::HeapBlock => "blocks the NT heaps got from VirtualAlloc".to_string(),
        Owner::Code => "memory that can be executed".to_string(),
        Owner::Image(k) => format!("pointing into {}", names.get(k).map_or("?", |n| n.as_str())),
        Owner::Zero => "all zeros where read".to_string(),
        Owner::Plain => "no two pointers into one image".to_string(),
        Owner::Away => "not resident where asked".to_string(),
    };
    let total = |s: &Sizes| s.iter().map(|x| x.1).sum::<u64>();
    let mut pointing: Vec<(Owner, Sizes)> = by.iter().filter(|(k, _)| matches!(k, Owner::Image(_))).map(|(k, v)| (*k, *v)).collect();
    pointing.sort_unstable_by_key(|p| (std::cmp::Reverse(total(&p.1)), match p.0 { Owner::Image(k) => k, _ => 0 }));
    let mut shown: Vec<String> = [Owner::HeapSegment, Owner::HeapBlock, Owner::Code]
        .into_iter()
        .chain(pointing.iter().take(SHOWN).map(|p| p.0))
        .chain([Owner::Plain, Owner::Zero, Owner::Away])
        .filter_map(|who| by.get(&who).map(|s| format!("{} {}", name(who), sizes_text(s))))
        .collect();
    if pointing.len() > SHOWN {
        let mut others = Sizes::default();
        for (_, s) in &pointing[SHOWN..] {
            for (o, x) in others.iter_mut().zip(s) {
                *o = (o.0 + x.0, o.1 + x.1);
            }
        }
        shown.push(format!("pointing into {} other images {}", pointing.len() - SHOWN, sizes_text(&others)));
    }
    let largest: Vec<String> = largest.iter().map(|&(bytes, size, who)| format!("{} MB of {} MB {}", mb(bytes), mb(size), name(who))).collect();
    static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    crate::xlog_info!(
        "memdiag: private other by owner, each allocation named by up to nine resident pages of it (allocations/MB): {}; the Windows heaps report {} MB, so {} MB of theirs is in blocks of their own that are counted under what their pages hold; the largest outside the heap segments (committed of reserved): {}; {} ms{}",
        shown.join("; "),
        mb(heaps),
        mb(heaps.saturating_sub(in_segments)),
        if largest.is_empty() { "none".to_string() } else { largest.join(", ") },
        started.elapsed().as_millis(),
        if TOLD.swap(true, Relaxed) { "" } else { " (XCL_MEMDIAG_OWNERS=0: this line off)" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_merged_and_counted_where_they_cover() {
        let r = merged([(10, 5), (0, 4), (12, 10), (40, 0), (30, 2)]);
        assert_eq!(r, [(0, 4), (10, 22), (30, 32)]);
        assert_eq!(covered(&r, 2, 30), 16);
        assert_eq!(covered(&r, 4, 6), 0);
        assert_eq!(covered(&r, 31, 100), 1);
        assert_eq!(covered(&[], 0, 100), 0);
    }

    #[test]
    fn a_bound_page_is_counted_once_under_the_first_of_its_addresses() {
        const P: usize = DIRECTORY_PAGE;
        let at = vec![(7, (0x10 * P) as u64), (7, (0x40 * P) as u64), (3, (0x20 * P) as u64), (5, (0x30 * P) as u64), (9, (0x50 * P) as u64)];
        let copied = [(0x40 * P, 100)];
        let placed = [(0x10 * P, P), (0x30 * P + 0x8000, 1)];
        let freed = [5usize, 9];
        let b = sort_bound(at, |pages| pages.iter().all(|p| !freed.contains(p)), &copied, &placed);
        assert_eq!(b, Bound { mapped: 5, pages: 4, freed: 2, under: [1, 1, 2] });
    }

    #[test]
    fn freed_pages_are_found_in_every_chunk() {
        let at: Vec<(usize, u64)> = (0..600).map(|p| (p, (p as u64) << 16)).collect();
        let b = sort_bound(at, |pages| pages.iter().all(|&p| p % 250 != 1), &[], &[]);
        assert_eq!((b.mapped, b.pages, b.freed, b.under), (600, 600, 3, [0, 0, 600]));
    }

    #[test]
    fn private_regions_are_named_by_stack_image_or_guard_page() {
        let places = Places { stacks: merged([(0x1000, 0x4000)]), title: merged([(0x10000, 0x2000)]), images: merged([(0x20000, 0x1000)]) };
        let mut n = Named::default();
        n.note(&places, 0x3000, 0x2000, 0x1000, false);
        n.note(&places, 0x2000, 0x1000, 0x1000, true);
        n.note(&places, 0x10000, 0x2000, 0x10000, false);
        n.note(&places, 0x20000, 0x1000, 0x20000, false);
        n.note(&places, 0x30000, 0x1000, 0x2f000, true);
        n.note(&places, 0x31000, 0x3000, 0x2f000, false);
        n.note(&places, 0x40000, 0x1000, 0x40000, false);
        n.note(&places, 0x41000, 0x1000, 0x2f000, false);
        assert_eq!((n.stacks, n.title, n.images, n.guarded), (0x3000, 0x2000, 0x1000, 0x5000));
    }

    #[test]
    fn an_allocation_gathers_its_regions_and_its_first_readable_one() {
        let mut owned = Vec::new();
        owned_note(&mut owned, (0x10000, 0x7000), 0x10000, 0x10000, 0x1000, 0x104);
        owned_note(&mut owned, (0x10000, 0x7000), 0x10000, 0x11000, 0x3000, 0x04);
        owned_note(&mut owned, (0x10000, 0x7000), 0x10000, 0x20000, 0x1000, 0x20);
        owned_note(&mut owned, (0x10000, 0x7000), 0x40000, 0x40000, 0x2000, 0x01);
        assert_eq!(owned.len(), 2);
        assert_eq!((owned[0].base, owned[0].committed, owned[0].size, owned[0].first, owned[0].code), (0x10000, 0x5000, 0xC000, (0x11000, 0x3000), true));
        assert_eq!((owned[1].base, owned[1].committed, owned[1].size, owned[1].first, owned[1].code), (0x40000, 0x2000, 0x2000, (0, 0), false));
        assert_eq!(owned_asks(&owned[0]), [0x11000, 0x12000, 0x20000, 0, 0, 0, 0, 0, 0]);
        assert_eq!(owned_asks(&owned[1]), [0; 9]);
        // (of many readable regions: the 1st, then the 2nd, 3rd, 5th, 9th, 17th, 33rd and 65th)
        let mut many = Vec::new();
        for k in 0..200usize {
            owned_note(&mut many, (0, 0), 0x100_0000, 0x100_0000 + k * 0x2000, 0x1000, 0x04);
        }
        let region = |n: usize| 0x100_0000 + (n - 1) * 0x2000;
        assert_eq!(owned_asks(&many[0]), [region(1), 0, region(2), region(3), region(5), region(9), region(17), region(33), region(65)]);
        assert_eq!([0u64, (1 << 20) - 1, 1 << 20, (16 << 20) - 1, 16 << 20, 64 << 20, 1 << 40].map(owned_size), [0, 0, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn an_allocation_is_named_by_its_pages() {
        fn page(words: &[(usize, usize)]) -> Vec<u8> {
            let mut p = vec![0u8; 4096];
            for &(i, w) in words {
                p[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
            }
            p
        }
        let bases = [0x10_0000usize, 0x20_0000, 0x30_0000];
        let heaps = [0x5000_0000usize];
        let images = [(0x7000_0000usize, 0x7010_0000usize), (0x7100_0000, 0x7100_1000)];
        let name = |p: &[u8], at_base: bool, code: bool, committed: u64| owner(&[p], at_base, false, code, committed, &bases, &heaps, &images);
        assert_eq!(owner(&[], true, false, true, 0x1000, &bases, &heaps, &images), Owner::Away);
        assert_eq!(owner(&[], false, true, true, 0x1000, &bases, &heaps, &images), Owner::HeapSegment);
        let segment = page(&[(2, 0xFFEE_FFEE), (9, 0x7000_0010), (10, 0x7000_0020)]);
        assert_eq!(name(&segment, true, false, 0x1000), Owner::HeapSegment);
        assert_eq!(name(&segment, false, false, 0x1000), Owner::Image(0));
        // (a block between another block and the heap's list head, 2 MiB committed)
        let block = page(&[(0, 0x30_0000), (1, 0x5000_0118), (4, 0x20_1000), (5, 0x20_1000), (9, 7)]);
        assert_eq!(name(&block, true, false, 0x20_1000), Owner::HeapBlock);
        assert_eq!(name(&block, true, false, 0x80_0000), Owner::Plain);
        assert_eq!(name(&block, false, false, 0x20_1000), Owner::Plain);
        let unlinked = page(&[(0, 0x30_0000), (1, 0x5000_2000), (4, 0x20_1000), (5, 0x20_1000)]);
        assert_eq!(name(&unlinked, true, false, 0x20_1000), Owner::Plain);
        assert_eq!(name(&unlinked, true, true, 0x20_1000), Owner::Code);
        assert_eq!(name(&page(&[]), true, false, 0x1000), Owner::Zero);
        assert_eq!(name(&page(&[(3, 0x7100_0800)]), true, false, 0x1000), Owner::Plain);
        assert_eq!(name(&page(&[(3, 0x7100_0800), (7, 0x7100_0000), (8, 0x7000_0000), (9, 0x7100_1000)]), true, false, 0x1000), Owner::Image(1));
        // (the second page decides when the first has no pointers)
        let (plain, pointers) = (page(&[(1, 5)]), page(&[(1, 0x700f_fff8), (2, 0x7000_0000)]));
        assert_eq!(owner(&[&plain, &pointers], true, false, false, 0x1000, &bases, &heaps, &images), Owner::Image(0));
        assert_eq!(owner(&[&page(&[]), &page(&[])], true, false, false, 0x1000, &bases, &heaps, &images), Owner::Zero);
    }

    #[test]
    fn segments_of_the_segment_heap_share_a_word_once_their_address_is_out() {
        let key = 0x1234_5678_9abc_def0usize;
        let mut starts: Vec<(usize, usize)> = (0..8).map(|k| (0x2000_0000 + (k << 24), (0x2000_0000 + (k << 24)) ^ key)).collect();
        starts.extend((0..7).map(|k| (0x9000_0000 + (k << 24), (0x9000_0000 + (k << 24)) ^ 0x55)));
        starts.push((0x1000_0000, 0x77));
        starts.push((0x1100_0000, 0));
        let found = heap_segments(&starts);
        assert_eq!(found, (0..8).map(|k| 0x2000_0000 + (k << 24)).collect::<Vec<usize>>());
        assert!(heap_segments(&[]).is_empty());
    }

    #[test]
    fn an_owner_shows_its_sizes_from_64_mb() {
        assert_eq!(sizes_text(&[(3, 3 << 20), (1, 8 << 20), (0, 0), (0, 0)]), "4/11 MB");
        assert_eq!(sizes_text(&[(3, 3 << 20), (0, 0), (2, 80 << 20), (1, 100 << 20)]), "6/183 MB (under 1 MiB 3/3 MB, 16-64 MiB 2/80 MB, 64 MiB+ 1/100 MB)");
    }

    #[test]
    fn a_large_block_counts_when_the_heap_rose_by_most_of_it() {
        assert!(counts_large(100 << 20, (100 << 20) + PROBE as u64));
        assert!(!counts_large(100 << 20, (100 << 20) + (1 << 20)));
        assert!(!counts_large(100 << 20, 90 << 20));
    }
}
