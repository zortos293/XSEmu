//! Title physical pages (64 KiB) as shared memory, as on the console: one page shows the same
//! memory at every address it is mapped at.
//!
//! An allocation (`AllocateTitlePhysicalPages`) is a pagefile-backed section of its size, reserved
//! (SEC_RESERVE: a page takes memory when it is first used), with one view of the whole
//! allocation that XCL reads and writes the pages through (`read_physical_pages` /
//! `write_physical_pages`). A page number counts 4 KiB frames (`page_number`; 0 means "none":
//! unmap).
//!
//! The title maps pages into address space it reserved. Its reservations of 64 KiB pages outside
//! unified memory are placeholders (`reserve`), and a page mapped there is a view of the page's
//! section offset (MapViewOfFile3 into the 64 KiB placeholder split out of the reservation;
//! unmapping turns the address back into a placeholder): its pool, the addresses it fills the
//! pages at, and the GPU's views of them (`d3d12x::page_mappings`) all show the one page, with no
//! copy. (A retail game maps all its streaming pages into a pool, maps each again elsewhere to fill
//! it, and has the GPU map the pool's slots.)
//!
//! Unified memory cannot hold views (a sparse window is one write-watched reservation, the
//! coherent window is imported by D3D12 as it is), so a page mapped there is a copy: filled from
//! the page when it is first used (`lazy`), or at once when the page holds data. While a page has
//! no view its copy is where its data lives: unmapping the last copy puts the data back in the
//! section, and the first view of the page takes the copy's data along.
//!
//! Views are not write-watched, so the GPU's copies of textures in them are kept in step by what
//! XCL writes there (`note_written`: DirectStorage, file reads, `write_physical_pages`, the video
//! decoder) – reported at every address showing the page (`alias_ranges`) – and by a bounded
//! comparison of the pages the GPU views show (`d3d12x::page_mappings`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};

use windows_sys::Win32::Foundation as fnd;
use windows_sys::Win32::System::Memory as mem;
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId};

use super::lazy;

type HANDLE = *mut c_void;
type BOOL = i32;

pub(crate) const PHYS_PAGE: usize = 0x1_0000;

/// Size of one console physical page (64 KiB).
pub const PHYS_PAGE_SIZE: usize = PHYS_PAGE;

/// Page numbers count 4 KiB frames, as on the console: a 64 KiB page's is a multiple of 16 (a retail game
/// packs them shifted right by 4). Index 0 is page 16 - 0 means "no page".
const FRAMES: usize = PHYS_PAGE >> 12;

/// VirtualFree: MEM_RELEASE with MEM_COALESCE_PLACEHOLDERS / MEM_PRESERVE_PLACEHOLDER
const COALESCE: u32 = mem::MEM_RELEASE | 1;
const PRESERVE: u32 = mem::MEM_RELEASE | 2;

/// the next page index handed out (page numbers are not addresses)
static NEXT_PAGE_INDEX: AtomicUsize = AtomicUsize::new(0);

fn page_number(index: usize) -> usize {
    (index + 1) * FRAMES
}

fn page_index(page: usize) -> Option<usize> {
    page.is_multiple_of(FRAMES).then(|| (page / FRAMES).checked_sub(1)).flatten()
}

fn take_page_indices(next: &AtomicUsize, count: usize, clustered: bool) -> Option<usize> {
    let mut old = next.load(Ordering::Relaxed);
    loop {
        let first = if clustered { old.checked_add(1)?.checked_next_multiple_of(32)? - 1 } else { old };
        let end = first.checked_add(count).filter(|&n| n <= usize::MAX / FRAMES)?;
        match next.compare_exchange_weak(old, end, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Some(first),
            Err(actual) => old = actual,
        }
    }
}

struct Block {
    count: usize,
    /// pages not freed yet
    live: usize,
    freed: Vec<usize>,
    section: Arc<Section>,
    clustered: bool,
    section_home: Option<(usize, usize)>,
    /// the view of the whole allocation
    view: usize,
    /// unified memory the title gets for these pages when it maps them at no chosen address
    uma_home: Option<usize>,
}

#[derive(Debug)]
struct Section { raw: usize, len: usize }

impl Drop for Section {
    fn drop(&mut self) {
        unsafe { fnd::CloseHandle(self.raw as HANDLE) };
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PhysicalPageRef {
    token: usize,
    offset: usize,
    section: Arc<Section>,
}

impl PhysicalPageRef {
    pub(crate) fn section_identity(&self) -> usize {
        Arc::as_ptr(&self.section) as usize
    }
    pub(crate) fn id(&self) -> usize {
        self.token
    }
    pub(crate) fn section_offset(&self) -> usize {
        self.offset
    }
    pub(crate) fn same_section(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.section, &other.section)
    }
}

pub(crate) const PHYSICAL_SEGMENT_BYTES: usize = 8 << 20;

/// The title's physical pages for `memdiag`: allocations, pages allocated and not freed, pages
/// committed (used) and, of those, pages freed while their allocation lives on (a section page
/// cannot be decommitted alone, so its memory stays until the whole allocation is freed).
pub(crate) fn physical_page_summary() -> String {
    let g = PHYS.lock().unwrap();
    let used = USED.read().unwrap_or_else(|e| e.into_inner());
    let (mut pages, mut live, mut freed_used) = (0usize, 0usize, 0usize);
    for b in g.blocks.values() {
        pages += b.count;
        live += b.live;
        if b.live < b.count {
            freed_used += (0..b.count).filter(|&i| !b.is_live(i) && used.contains(&(b.view + i * PHYS_PAGE))).count();
        }
    }
    let mb = |n: usize| (n * PHYS_PAGE) >> 20;
    format!(
        "{} allocations of {} MB, {} MB not freed, {} MB committed, {} MB committed in freed pages of live allocations",
        g.blocks.len(),
        mb(pages),
        mb(live),
        mb(used.len()),
        mb(freed_used)
    )
}

impl PhysicalPageRef {
    pub(crate) fn pin_committed_segment(&self, offsets: std::ops::Range<usize>) -> Option<(usize, PhysicalSectionView)> {
        let lo = offsets.start;
        let hi = offsets.end.min(self.section.len);
        let len = hi.checked_sub(lo)?;
        if len == 0 || len > PHYSICAL_SEGMENT_BYTES || !lo.is_multiple_of(PHYS_PAGE) || !hi.is_multiple_of(PHYS_PAGE)
            || self.offset < lo || self.offset.checked_add(PHYS_PAGE)? > hi {
            return None;
        }
        unsafe {
            let view = PhysicalSectionView { base: view_anywhere(self.section.raw, lo, len)?, len };
            let mut at = view.base;
            let end = view.base.checked_add(len)?;
            while at < end {
                let mut mbi = mem::MEMORY_BASIC_INFORMATION::default();
                if mem::VirtualQuery(at as *const _, &mut mbi, std::mem::size_of_val(&mbi)) == 0 || mbi.AllocationBase as usize != view.base {
                    return None;
                }
                let next = (mbi.BaseAddress as usize).checked_add(mbi.RegionSize)?.min(end);
                if next <= at { return None; }
                let start = lo + (at - view.base).checked_next_multiple_of(PHYS_PAGE)?;
                let stop = lo + ((next - view.base) & !(PHYS_PAGE - 1));
                if start <= self.offset && self.offset + PHYS_PAGE <= stop {
                    if mbi.Type != mem::MEM_MAPPED || mbi.State != mem::MEM_COMMIT || mbi.Protect != mem::PAGE_READWRITE {
                        return None;
                    }
                    if start == lo && stop == hi { return Some((lo, view)); }
                    let bytes = stop - start;
                    let base = view_anywhere(self.section.raw, start, bytes)?;
                    return Some((start, PhysicalSectionView { base, len: bytes }));
                }
                at = next;
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) enum QueuePageSource {
    Physical(PhysicalPageRef),
    // GuestRange base and len are not read yet; kept for the next fix.
    #[allow(dead_code)]
    GuestRange { base: usize, len: usize },
    Hole,
}

pub(crate) fn snapshot_queue_sources(slots: &[usize]) -> Result<Vec<QueuePageSource>, &'static str> {
    let g = PHYS.lock().unwrap();
    slots.iter().map(|&at| g.queue_source(at)).collect()
}

// The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn snapshot_default_gpu_sources(slots: &[usize]) -> Result<Vec<Option<QueuePageSource>>, &'static str> {
    let g = PHYS.lock().unwrap();
    slots.iter().map(|&at| g.cpu_reservation(at).filter(|r| r.class.default_gpu).map(|_| g.queue_source(at)).transpose()).collect()
}

impl Phys {
    fn queue_source(&self, at: usize) -> Result<QueuePageSource, &'static str> {
        if !at.is_multiple_of(PHYS_PAGE) {
            return Err("unaligned source");
        }
        let mapped = self.at.get(&at);
        let page = mapped.map(|m| m.page).or_else(|| {
            self.blocks.iter().find_map(|(&first, b)| {
                let i = at.checked_sub(b.view)? / PHYS_PAGE;
                b.is_live(i).then(|| page_number(first + i))
            })
        });
        if let Some(page) = page {
            if !self.has_views(page) && !self.copies(page).is_empty() {
                return Err("physical source data lives in a copy");
            }
            if let Some(owned) = mapped.and_then(|m| m.backing.as_ref()) {
                return Ok(QueuePageSource::Physical(owned.clone()));
            }
            let (b, offset) = self.block_of(page).ok_or("physical source freed")?;
            if !unsafe { use_page(b.view + offset) } {
                return Err("physical source commitment failed");
            }
            return Ok(QueuePageSource::Physical(PhysicalPageRef { token: page, offset, section: Arc::clone(&b.section) }));
        }
        if let Some(r) = self.reservations.range(..=at).next_back().map(|(_, r)| r).filter(|r| at < r.end) {
            if r.slots.get(&at) != Some(&Slot::Private) {
                return Ok(QueuePageSource::Hole);
            }
        }
        let end = at.checked_add(PHYS_PAGE - 1).ok_or("guest source range wraps")?;
        if let Some(u) = crate::gpu::uma::get().filter(|u| u.contains(at)) {
            let region = u.sparse_reservation(at).or_else(|| u.committed_reservation(at)).or_else(|| u.block_size(at).map(|n| (at, n)));
            if !region.is_some_and(|(base, len)| end - base < len) {
                return Err("guest source is not covered by one known UMA allocation");
            }
        }
        match crate::memmap::access(at, PHYS_PAGE) {
            Some(crate::memmap::Access::Read | crate::memmap::Access::Write) => Ok(QueuePageSource::GuestRange { base: at, len: PHYS_PAGE }),
            _ => Err("unclassified or inaccessible guest source"),
        }
    }
}

// The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn cpu_reservation_at(at: usize) -> Option<CpuReservation> {
    PHYS.lock().unwrap().cpu_reservation(at)
}

pub(crate) fn pin_physical_view_run(pages: &[PhysicalPageRef]) -> Option<PhysicalSectionView> {
    let first = pages.first()?;
    if pages.len() > (8 << 20) / PHYS_PAGE
        || !pages.iter().enumerate().all(|(i, p)| first.same_section(p) && first.offset + i * PHYS_PAGE == p.offset)
    {
        return None;
    }
    let len = pages.len() * PHYS_PAGE;
    let base = unsafe { view_anywhere(first.section.raw, first.offset, len) }?;
    Some(PhysicalSectionView { base, len })
}

pub(crate) struct PhysicalSectionView {
    base: usize,
    len: usize,
}

impl PhysicalSectionView {
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Drop for PhysicalSectionView {
    fn drop(&mut self) {
        unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.base as *mut c_void }) };
    }
}

pub(crate) fn physical_import_probe_views() -> Option<(PhysicalSectionView, PhysicalSectionView)> {
    unsafe {
        let section = create_section(3 * PHYS_PAGE)?;
        let make = || view_anywhere(section, PHYS_PAGE, PHYS_PAGE).map(|base| PhysicalSectionView { base, len: PHYS_PAGE });
        let a = make();
        let b = make();
        fnd::CloseHandle(section as HANDLE);
        let (a, b) = (a?, b?);
        if mem::VirtualAlloc(a.base as *const c_void, a.len, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
            return None;
        }
        Some((a, b))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VirtualGrain {
    K64,
    // The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    M2,
}

impl VirtualGrain {
    fn bytes(self) -> usize {
        match self {
            Self::K64 => PHYS_PAGE,
            Self::M2 => 32 * PHYS_PAGE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapClass {
    pub(crate) grain: VirtualGrain,
    pub(crate) default_gpu: bool,
}

pub(crate) struct GdkPhysicalAbi {
    pub(crate) mappable_reservation: fn(u32, u64, u32) -> Option<MapClass>,
    pub(crate) source_pool_allocation: fn(u32, u64, u32) -> Option<MapClass>,
    pub(crate) clustered_allocation: fn(u64) -> bool,
}

static SECTION_BACKEND: OnceLock<GdkPhysicalAbi> = OnceLock::new();

// The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn install_section_backend(abi: GdkPhysicalAbi) -> Result<(), GdkPhysicalAbi> {
    SECTION_BACKEND.set(abi)
}

#[derive(Clone, Copy, Debug)]
// Fields of the GDK CPU reservation; not read yet, kept for the next fix.
#[allow(dead_code)]
pub(crate) struct CpuReservation {
    pub(crate) id: usize,
    pub(crate) requested: usize,
    pub(crate) base: usize,
    pub(crate) len: usize,
    pub(crate) allocation: u32,
    pub(crate) flags: u64,
    pub(crate) protection: u32,
    pub(crate) accepts_physical_map: bool,
    pub(crate) class: MapClass,
}

impl CpuReservation {
    pub(crate) fn cache(&self) -> u32 { self.protection & (mem::PAGE_NOCACHE | mem::PAGE_WRITECOMBINE) }
    pub(crate) fn cpu_access(&self) -> u32 { super::sanitize_protect(self.protection) & 0xff }
}

static NEXT_RESERVATION: AtomicUsize = AtomicUsize::new(1);

impl Block {
    const WORD_BITS: usize = usize::BITS as usize;

    fn is_live(&self, offset: usize) -> bool {
        offset < self.count && self.freed[offset / Self::WORD_BITS] & (1 << (offset % Self::WORD_BITS)) == 0
    }

    fn release(&mut self, offset: usize) -> bool {
        if !self.is_live(offset) {
            return false;
        }
        self.freed[offset / Self::WORD_BITS] |= 1 << (offset % Self::WORD_BITS);
        self.live -= 1;
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// a view of the page's section
    View,
    /// a copy (unified memory, or address space that is not a placeholder)
    Copy,
}

#[derive(Clone)]
struct Mapping {
    page: usize,
    kind: Kind,
    backing: Option<PhysicalPageRef>,
}

/// A 64 KiB slot of a placeholder reservation split out of it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Placeholder,
    View,
    /// private memory the title (or a GPU view XCL had to fill itself) committed there
    Private,
}

// Raw GDK64 reservations are not observed yet; kept for the next fix.
#[allow(dead_code)]
struct Reservation {
    cpu: Option<CpuReservation>,
    raw_gdk64: Option<RawGdk64>,
    end: usize,
    slots: BTreeMap<usize, Slot>,
}

#[derive(Clone, Copy, Debug)]
// Fields of the raw GDK64 reservation; not read yet, kept for the next fix.
#[allow(dead_code)]
pub(crate) struct RawGdk64 {
    pub(crate) requested: usize,
    pub(crate) requested_size: usize,
    pub(crate) allocation: u32,
    pub(crate) flags: u64,
    pub(crate) protection: u32,
}

// The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn compat_gdk64_reservation_at(at: usize) -> Option<(usize, usize, RawGdk64)> {
    let g = PHYS.lock().unwrap();
    let (&base, r) = g.reservations.range(..=at).next_back()?;
    if at >= r.end { return None; }
    Some((base, r.end - base, r.raw_gdk64?))
}

#[derive(Default)]
struct Phys {
    /// first page index → its allocation
    blocks: BTreeMap<usize, Block>,
    /// address → the page mapped there
    at: HashMap<usize, Mapping>,
    /// page → the addresses it is mapped at
    places: HashMap<usize, Vec<usize>>,
    /// the title's reservations of 64 KiB pages made of placeholders: base → reservation
    reservations: BTreeMap<usize, Reservation>,
    /// the latest maps, unmaps, commits and frees, for a crash record (`alias_facts`)
    history: crate::alias::MapHistory,
    /// the call the changes being made are for (in `history`)
    site: &'static str,
}

fn kind_name(kind: Kind, lazy: bool) -> &'static str {
    match kind {
        Kind::View => "view",
        Kind::Copy if lazy => "copy (not used yet)",
        Kind::Copy => "copy",
    }
}

static PHYS: LazyLock<Mutex<Phys>> = LazyLock::new(|| Mutex::new(Phys::default()));

/// How many addresses show a page as a copy (`Kind::Copy` in `Phys::at`): none while the title
/// maps its pages in placeholder reservations only, and then `title_freed` has nothing to do.
static COPIES: AtomicUsize = AtomicUsize::new(0);

/// The section pages committed (used), by their place in their allocation's view. (Kept apart
/// from `PHYS` for `lazy`'s fault handler; and asking Windows – VirtualQuery – walks the whole run
/// of committed pages around the address: in a view of gigabytes that took 10 ms per call.)
static USED: LazyLock<RwLock<HashSet<usize>>> = LazyLock::new(|| RwLock::new(HashSet::new()));

/// Whether the section page XCL sees at `home` holds data (was committed).
pub(super) fn used(home: usize) -> bool {
    USED.read().unwrap_or_else(|e| e.into_inner()).contains(&home)
}

/// The section page at `home` committed (false: it could not be).
unsafe fn use_page(home: usize) -> bool {
    if used(home) {
        return true;
    }
    if !commit(home, PHYS_PAGE) {
        return false;
    }
    USED.write().unwrap_or_else(|e| e.into_inner()).insert(home);
    true
}

fn committed(a: usize) -> bool {
    if let Some(x) = crate::memmap::access(a, 1) {
        return x != crate::memmap::Access::None;
    }
    let mut m: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    unsafe { mem::VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == mem::MEM_COMMIT }
}

unsafe fn commit(a: usize, len: usize) -> bool {
    let ok = lazy::commit_memory(a, len);
    if ok {
        crate::memmap::committed(a, len, mem::PAGE_READWRITE);
    }
    ok
}

unsafe fn decommit(a: usize, len: usize) {
    mem::VirtualFree(a as *mut c_void, len, mem::MEM_DECOMMIT);
    crate::memmap::decommitted(a, len);
}

// --- the Windows calls --------------------------------------------------------------------

/// A pagefile-backed section of `bytes`, its pages committed when used.
unsafe fn create_section(bytes: usize) -> Option<usize> {
    let h = mem::CreateFileMappingW(fnd::INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_RESERVE, (bytes as u64 >> 32) as u32, bytes as u32, std::ptr::null());
    (!h.is_null()).then_some(h as usize)
}

/// A view of `len` bytes of `section` from `offset`, where Windows places it.
unsafe fn view_anywhere(section: usize, offset: usize, len: usize) -> Option<usize> {
    let v = mem::MapViewOfFile3(section as HANDLE, GetCurrentProcess(), std::ptr::null(), offset as u64, len, 0, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
    (!v.Value.is_null()).then_some(v.Value as usize)
}

/// A placeholder reservation of `len` bytes.
unsafe fn reserve_placeholder(len: usize) -> Option<usize> {
    let p = mem::VirtualAlloc2(GetCurrentProcess(), std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0);
    (!p.is_null()).then_some(p as usize)
}

unsafe fn cpu_reserve(g: &mut Phys, requested: usize, size: usize, allocation: u32, flags: u64, protection: u32, class: MapClass) -> Option<usize> {
    let grain = class.grain.bytes();
    if size == 0 || !requested.is_multiple_of(grain) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return None;
    }
    let len = size.checked_next_multiple_of(grain)?;
    let mut req = mem::MEM_ADDRESS_REQUIREMENTS { Alignment: grain, ..Default::default() };
    let mut ext = mem::MEM_EXTENDED_PARAMETER {
        Anonymous1: mem::MEM_EXTENDED_PARAMETER_0 { _bitfield: mem::MemExtendedParameterAddressRequirements as u64 },
        Anonymous2: mem::MEM_EXTENDED_PARAMETER_1 { Pointer: (&mut req as *mut mem::MEM_ADDRESS_REQUIREMENTS).cast() },
    };
    let (ext, n) = if requested == 0 && grain > PHYS_PAGE { (&mut ext as *mut _, 1) } else { (std::ptr::null_mut(), 0) };
    let base = mem::VirtualAlloc2(
        GetCurrentProcess(),
        requested as *const c_void,
        len,
        mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
        mem::PAGE_NOACCESS,
        ext,
        n,
    ) as usize;
    if base == 0 {
        return None;
    }
    let cpu = CpuReservation { id: NEXT_RESERVATION.fetch_add(1, Ordering::Relaxed), requested, base, len, allocation, flags, protection, class, accepts_physical_map: true };
    g.reservations.insert(base, Reservation { end: base + len, slots: BTreeMap::new(), cpu: Some(cpu), raw_gdk64: None });
    crate::memmap::reserved(base, len);
    Some(base)
}

pub(super) unsafe fn gdk_reserve(addr: *const c_void, size: usize, t: u32, flags: u64, protection: u32) -> Option<*mut c_void> {
    gdk_reserve_with(SECTION_BACKEND.get()?, addr, size, t, flags, protection)
}

unsafe fn gdk_reserve_with(abi: &GdkPhysicalAbi, addr: *const c_void, size: usize, t: u32, flags: u64, protection: u32) -> Option<*mut c_void> {
    let committed = match t & (mem::MEM_RESERVE | mem::MEM_COMMIT | mem::MEM_REPLACE_PLACEHOLDER) {
        mem::MEM_RESERVE => false,
        bits if bits == mem::MEM_RESERVE | mem::MEM_COMMIT || (bits == mem::MEM_COMMIT && addr.is_null()) => true,
        _ => return None,
    };
    let (class, mappable) = (abi.mappable_reservation)(t, flags, protection)
        .map(|class| (class, true)).or_else(|| (abi.source_pool_allocation)(t, flags, protection).map(|class| (class, false)))?;
    let mut g = PHYS.lock().unwrap();
    let Some(base) = cpu_reserve(&mut g, addr as usize, size, t, flags, protection, class) else { return Some(std::ptr::null_mut()) };
    g.reservation(base).unwrap().cpu.as_mut().unwrap().accepts_physical_map = mappable;
    if committed {
        if !g.commit_anonymous(base, size, protection) {
            let error = fnd::GetLastError();
            free_reservation(&mut g, base, 0, mem::MEM_RELEASE);
            fnd::SetLastError(error);
            return Some(std::ptr::null_mut());
        }
    }
    Some(base as *mut c_void)
}

/// The 64 KiB at `at` split out of the placeholder holding them.
unsafe fn split(at: usize) -> bool {
    mem::VirtualFree(at as *mut c_void, PHYS_PAGE, PRESERVE) != 0
}

/// The 64 KiB placeholder at `at` replaced by a view of `section` from `offset`.
#[cfg(test)]
unsafe fn map_into(section: usize, offset: usize, at: usize) -> bool {
    map_with_access(section, offset, at, mem::PAGE_READWRITE)
}

unsafe fn map_with_access(section: usize, offset: usize, at: usize, access: u32) -> bool {
    let v = mem::MapViewOfFile3(section as HANDLE, GetCurrentProcess(), at as *const c_void, offset as u64, PHYS_PAGE, mem::MEM_REPLACE_PLACEHOLDER, access, std::ptr::null_mut(), 0);
    v.Value as usize == at
}

/// The view at `at` unmapped, the address a placeholder again.
unsafe fn unmap_into_placeholder(at: usize) -> bool {
    crate::memmap::decommitted(at, PHYS_PAGE);
    mem::UnmapViewOfFile2(GetCurrentProcess(), mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }, mem::MEM_PRESERVE_PLACEHOLDER) != 0
}

/// The 64 KiB placeholder at `at` replaced by private memory (reserved, write-watched).
unsafe fn private_into(at: usize) -> bool {
    mem::VirtualAlloc2(GetCurrentProcess(), at as *const c_void, PHYS_PAGE, mem::MEM_RESERVE | mem::MEM_REPLACE_PLACEHOLDER | super::MEM_WRITE_WATCH, mem::PAGE_READWRITE, std::ptr::null_mut(), 0) as usize == at
}

// --- the model ----------------------------------------------------------------------------

impl Reservation {
    /// The slot at `at` as a placeholder of its own.
    unsafe fn placeholder(&mut self, at: usize) -> bool {
        match self.slots.get(&at).copied() {
            None => {
                // (a reservation of one page is already that placeholder: the split fails and the
                // replacement works)
                split(at);
            }
            Some(Slot::Placeholder) => {}
            Some(Slot::View) => {
                if !unmap_into_placeholder(at) {
                    return false;
                }
                lazy::unview(at);
            }
            Some(Slot::Private) => {
                crate::memmap::decommitted(at, PHYS_PAGE);
                if mem::VirtualFree(at as *mut c_void, PHYS_PAGE, PRESERVE) == 0 {
                    return false;
                }
                if super::GRAPHICS_RESERVATIONS.lock().unwrap().remove(&at).is_some() {
                    super::GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
                }
            }
        }
        self.slots.insert(at, Slot::Placeholder);
        true
    }

    /// The slot at `at` as private memory (reserved; kept if it is already): GPU-visible title
    /// memory, write-watched like a graphics reservation so `d3d12x::title_textures` sees the
    /// CPU's writes to textures there.
    unsafe fn private(&mut self, at: usize) -> bool {
        if self.slots.get(&at) == Some(&Slot::Private) {
            return true;
        }
        if !self.placeholder(at) || !private_into(at) {
            return false;
        }
        self.slots.insert(at, Slot::Private);
        super::note_graphics_reservation(at, PHYS_PAGE);
        true
    }
}

impl Phys {
    fn backing_block(&self, page: usize) -> Option<(&Block, usize)> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range(..=index).next_back()?;
        let offset = index - first;
        (offset < b.count).then_some((b, offset))
    }

    fn block_of(&self, page: usize) -> Option<(&Block, usize)> {
        let (b, offset) = self.backing_block(page)?;
        b.is_live(offset).then(|| (b, offset * PHYS_PAGE))
    }

    /// Where XCL sees page `page`: its place in the view of its allocation.
    fn home(&self, page: usize) -> Option<usize> {
        self.backing_block(page).map(|(b, offset)| b.view + offset * PHYS_PAGE)
    }

    fn reservation(&mut self, at: usize) -> Option<&mut Reservation> {
        let (_, r) = self.reservations.range_mut(..=at).next_back()?;
        (at < r.end).then_some(r)
    }

    fn in_reservation(&self, at: usize) -> bool {
        self.reservations.range(..=at).next_back().is_some_and(|(_, r)| at < r.end)
    }

    fn cpu_reservation(&self, at: usize) -> Option<CpuReservation> {
        self.reservations.range(..=at).next_back().and_then(|(_, r)| (at < r.end).then_some(r.cpu).flatten())
    }

    unsafe fn commit_anonymous(&mut self, at: usize, size: usize, protect: u32) -> bool {
        let cpu = self.cpu_reservation(at).unwrap();
        let grain = cpu.class.grain.bytes();
        let first = at & !(grain - 1);
        let Some(end) = at.checked_add(size.max(1)).and_then(|e| e.checked_next_multiple_of(grain)).filter(|&e| e <= cpu.base + cpu.len) else {
            fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
            return false;
        };
        let access = super::sanitize_protect(protect) & 0xff;
        let mut installed = Vec::new();
        let mut slot = first;
        let install = |g: &mut Phys, installed: &mut Vec<usize>, start: usize, count: usize| -> Option<()> {
            let len = count * PHYS_PAGE;
            let section = Arc::new(Section { raw: create_section(len)?, len });
            let view = PhysicalSectionView { base: view_anywhere(section.raw, 0, len)?, len };
            if mem::VirtualAlloc(view.base as *const _, len, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() { return None; }
            let index = take_page_indices(&NEXT_PAGE_INDEX, count, false)?;
            for i in 0..count {
                let a = start + i * PHYS_PAGE;
                let r = g.reservation(a)?;
                if !r.placeholder(a) || !map_with_access(section.raw, i * PHYS_PAGE, a, access) { return None; }
                r.slots.insert(a, Slot::View);
                let page = page_number(index + i);
                let owned = PhysicalPageRef { token: page, offset: i * PHYS_PAGE, section: Arc::clone(&section) };
                g.record(a, page, Kind::View, Some(owned));
                lazy::view(a, PHYS_PAGE);
                crate::memmap::committed(a, PHYS_PAGE, access);
                installed.push(a);
            }
            Some(())
        };
        while slot < end {
            if self.at.contains_key(&slot) { slot += PHYS_PAGE; continue; }
            let start = slot;
            while slot < end && slot - start < 8 << 20 && !self.at.contains_key(&slot)
                && self.reservation(slot).is_some_and(|r| r.slots.get(&slot) != Some(&Slot::Private)) {
                slot += PHYS_PAGE;
            }
            if start == slot || install(self, &mut installed, start, (slot - start) / PHYS_PAGE).is_none() {
                for a in installed { self.unmap_at(a); }
                fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
                return false;
            }
        }
        true
    }

    unsafe fn home_for_mapping(&mut self, page: usize, protection: u32) -> Option<usize> {
        if SECTION_BACKEND.get().is_none() {
            return self.uma_home(page);
        }
        self.section_home_for_mapping(page, protection)
    }

    unsafe fn section_home_for_mapping(&mut self, page: usize, protection: u32) -> Option<usize> {
        let (block, offset) = self.block_of(page)?;
        let first = page_index(page)? - offset / PHYS_PAGE;
        let base = match block.section_home.filter(|&(home, id)| self.cpu_reservation(home).is_some_and(|r| r.id == id)) {
            Some((home, _)) => home,
            None => {
                let bytes = block.count * PHYS_PAGE;
                let home = cpu_reserve(self, 0, bytes, mem::MEM_RESERVE, 0, protection, MapClass { grain: VirtualGrain::K64, default_gpu: true })?;
                let id = self.cpu_reservation(home)?.id;
                self.blocks.get_mut(&first)?.section_home = Some((home, id));
                home
            }
        };
        base.checked_add(offset)
    }

    fn places(&self, page: usize) -> &[usize] {
        self.places.get(&page).map_or(&[], |v| v.as_slice())
    }

    fn has_views(&self, page: usize) -> bool {
        self.places(page).iter().any(|a| self.at.get(a).is_some_and(|m| m.kind == Kind::View))
    }

    /// The copies of `page` in use (committed, not waiting for their first use).
    fn copies(&self, page: usize) -> Vec<usize> {
        self.places(page).iter().copied().filter(|a| self.at.get(a).is_some_and(|m| m.kind == Kind::Copy) && !lazy::is_lazy(*a)).collect()
    }

    /// Where the page's data is: a copy in use while the page has no view, else the section (None:
    /// never used – zeros).
    fn owner(&self, page: usize) -> Option<usize> {
        if !self.has_views(page) {
            if let Some(&a) = self.copies(page).first() {
                return Some(a);
            }
        }
        self.home(page).filter(|&h| used(h))
    }

    /// A change of the 64 KiB pages of `[at, at + len)` (`len` 0: none) or of `page` (0: none) kept
    /// in `history`.
    fn note(&mut self, what: &'static str, at: usize, len: usize, page: usize) {
        self.history.push(crate::alias::MapEvent { t: crate::log::uptime() as f32, thread: unsafe { GetCurrentThreadId() }, at, len, page, what, site: self.site });
    }

    fn record(&mut self, at: usize, page: usize, kind: Kind, owned: Option<PhysicalPageRef>) {
        self.note(
            match kind {
                Kind::View => "map view of",
                Kind::Copy if lazy::is_lazy(at) => "map copy (filled on first use) of",
                Kind::Copy => "map copy of",
            },
            at,
            PHYS_PAGE,
            page,
        );
        if kind == Kind::Copy {
            COPIES.fetch_add(1, Ordering::Release);
        }
        let backing = if kind == Kind::View {
            self.block_of(page).map(|(b, offset)| PhysicalPageRef { token: page, offset, section: Arc::clone(&b.section) })
        } else { None };
        self.at.insert(at, Mapping { page, kind, backing: owned.or(backing) });
        self.places.entry(page).or_default().push(at);
    }

    /// The addresses in `[lo, hi)` (`lo` a multiple of 64 KiB) that show a page, in order.
    fn mapped_in(&self, lo: usize, hi: usize) -> Vec<usize> {
        let slots = hi.saturating_sub(lo).div_ceil(PHYS_PAGE);
        let mut found: Vec<usize> = if slots <= self.at.len() {
            (0..slots).map(|i| lo + i * PHYS_PAGE).filter(|a| self.at.contains_key(a)).collect()
        } else {
            self.at.keys().copied().filter(|&a| a >= lo && a < hi).collect()
        };
        found.sort_unstable();
        found
    }

    /// Page `page` mapped at `at` (false: it could not be; the last error says why).
    unsafe fn map_at(&mut self, at: usize, page: usize) -> bool {
        if self.at.get(&at).is_some_and(|m| m.page == page) {
            return true;
        }
        if !self.unmap_at(at) {
            return false;
        }
        if self.in_reservation(at) {
            let Some((section, offset)) = self.block_of(page).map(|(b, off)| (b.section.raw, off)) else { return false };
            let home = self.home(page).unwrap();
            let from = if self.has_views(page) { None } else { self.copies(page).first().copied() };
            // (committed now: the page is written through its views)
            if !use_page(home) {
                fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
                return false;
            }
            if let Some(from) = from {
                // the copy in use has the data: into the section, which the view shows
                std::ptr::copy_nonoverlapping(from as *const u8, home as *mut u8, PHYS_PAGE);
                crate::eh::note_guest_write("physical page section filled from the page's copy (its first view)", home, PHYS_PAGE);
            }
            let r = self.reservation(at).unwrap();
            let access = r.cpu.map_or(mem::PAGE_READWRITE, |cpu| cpu.cpu_access());
            if !r.placeholder(at) || !map_with_access(section, offset, at, access) {
                fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
                return false;
            }
            r.slots.insert(at, Slot::View);
            lazy::view(at, PHYS_PAGE);
            crate::memmap::committed(at, PHYS_PAGE, access);
            self.record(at, page, Kind::View, None);
            return true;
        }
        let uma = crate::gpu::uma::get();
        let in_committed = uma.is_some_and(|u| u.is_committed(at));
        let sparse = uma.is_some_and(|u| u.is_sparse(at));
        let owner = self.owner(page);
        // a page never used: committed on first use, with what the page holds then (a retail
        // game's pool of 3.9 GB in a sparse window took that much video memory, and every
        // submit's look for CPU writes walked it)
        if owner.is_none() && !in_committed && !committed(at) {
            lazy::mark(at, self.home(page));
            self.record(at, page, Kind::Copy, None);
            return true;
        }
        let fresh = !in_committed && !committed(at);
        if fresh {
            if !commit(at, PHYS_PAGE) {
                fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
                return false;
            }
            if sparse {
                crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
            }
        }
        if owner.is_some() || !fresh {
            // (XCL's write into the title's memory: what was committed there is overwritten)
            crate::alias::mapping_changed("a physical page is mapped as a copy over memory", self.site, at, PHYS_PAGE);
            crate::eh::note_guest_write(if owner.is_some() { "physical page copy filled (mapped)" } else { "physical page copy zeroed (mapped, page never used)" }, at, PHYS_PAGE);
        }
        match owner {
            Some(from) => std::ptr::copy_nonoverlapping(from as *const u8, at as *mut u8, PHYS_PAGE),
            None if !fresh => std::ptr::write_bytes(at as *mut u8, 0, PHYS_PAGE),
            None => {} // freshly committed: zeros
        }
        self.record(at, page, Kind::Copy, None);
        true
    }

    /// Whatever page is mapped at `at` unmapped. False: Windows refused to unmap the view there –
    /// the address still shows the page, the model still says so, and the call this is for
    /// fails (an unmap taken as done left the title's address showing a page XCL no longer
    /// counted among the page's addresses).
    unsafe fn unmap_at(&mut self, at: usize) -> bool {
        let Some(m) = self.at.remove(&at) else { return true };
        if m.kind == Kind::Copy {
            COPIES.fetch_sub(1, Ordering::Release);
        }
        if let Some(v) = self.places.get_mut(&m.page) {
            v.retain(|&a| a != at);
            if v.is_empty() {
                self.places.remove(&m.page);
            }
        }
        self.note(
            match m.kind {
                Kind::View => "unmap view of",
                Kind::Copy if lazy::is_lazy(at) => "unmap copy (never used) of",
                Kind::Copy => "unmap copy of",
            },
            at,
            PHYS_PAGE,
            m.page,
        );
        crate::alias::mapping_changed("a physical page is unmapped", self.site, at, PHYS_PAGE);
        match m.kind {
            Kind::View => {
                if let Some(r) = self.reservation(at) {
                    if unmap_into_placeholder(at) {
                        r.slots.insert(at, Slot::Placeholder);
                    } else {
                        let error = fnd::GetLastError();
                        self.at.insert(at, m.clone());
                        self.places.entry(m.page).or_default().push(at);
                        self.note("UNMAP REFUSED BY WINDOWS, the view stays: of", at, PHYS_PAGE, m.page);
                        static SITES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
                        let mut sites = SITES.lock().unwrap_or_else(|e| e.into_inner());
                        if !sites.contains(&self.site) {
                            sites.push(self.site);
                            crate::xlog_warn!("physical pages: Windows refused to unmap the view of a page (error {error}) for {}: the address still shows the page, the call fails", self.site);
                        }
                        fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
                        return false;
                    }
                }
                lazy::unview(at);
                if !self.has_views(m.page) {
                    // the copies have the data again
                    let home = self.home(m.page).filter(|&h| used(h));
                    for a in self.copies(m.page) {
                        crate::alias::mapping_changed("a physical page's copy is overwritten with its section (its last view unmapped)", self.site, a, PHYS_PAGE);
                        crate::eh::note_guest_write("physical page copy overwritten with its section (its last view unmapped)", a, PHYS_PAGE);
                        match home {
                            Some(h) => std::ptr::copy_nonoverlapping(h as *const u8, a as *mut u8, PHYS_PAGE),
                            None => std::ptr::write_bytes(a as *mut u8, 0, PHYS_PAGE),
                        }
                    }
                }
            }
            Kind::Copy => {
                if lazy::unmark(at) {
                    return true; // never used
                }
                if !self.has_views(m.page) && self.copies(m.page).is_empty() {
                    if let Some(home) = self.home(m.page) {
                        if use_page(home) {
                            std::ptr::copy_nonoverlapping(at as *const u8, home as *mut u8, PHYS_PAGE);
                            crate::eh::note_guest_write("physical page section filled from the page's last copy (unmapped)", home, PHYS_PAGE);
                        }
                    }
                }
                let uma = crate::gpu::uma::get();
                if !uma.is_some_and(|u| u.is_committed(at)) {
                    if uma.is_some_and(|u| u.is_sparse(at)) {
                        crate::hle::d3d12x::sparse::decommitted(at, PHYS_PAGE);
                    }
                    decommit(at, PHYS_PAGE);
                }
            }
        }
        true
    }

    /// Unified memory for the block of `page` (mapping its pages at no chosen address).
    fn uma_home(&mut self, page: usize) -> Option<usize> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range_mut(..=index).next_back()?;
        if !b.is_live(index - first) {
            return None;
        }
        if b.uma_home.is_none() {
            let u = crate::gpu::uma::get()?;
            let bytes = b.count * PHYS_PAGE;
            b.uma_home = u.sparse_alloc(bytes, PHYS_PAGE).map(|(p, _)| p).or_else(|| u.alloc(bytes, PHYS_PAGE).map(|p| p as usize));
        }
        b.uma_home.map(|h| h + (index - first) * PHYS_PAGE)
    }

    /// The page shown at `va`: one mapped there, or the page whose place in its allocation's view
    /// (or unified memory home) it is.
    fn page_at(&self, va: usize) -> Option<usize> {
        let at = va & !(PHYS_PAGE - 1);
        if let Some(m) = self.at.get(&at) {
            return Some(m.page);
        }
        self.blocks.iter().find_map(|(&first, b)| {
            let within = |base: usize| {
                let offset = at.checked_sub(base)? / PHYS_PAGE;
                b.is_live(offset).then(|| page_number(first + offset))
            };
            within(b.view).or_else(|| b.uma_home.and_then(within))
        })
    }
}

pub unsafe extern "system" fn AllocateTitlePhysicalPages(_p: HANDLE, _t: u32, num: *mut usize, arr: *mut usize) -> BOOL {
    allocate_pages(num, arr, false)
}

unsafe fn allocate_pages(num: *mut usize, arr: *mut usize, clustered: bool) -> BOOL {
    if num.is_null() || arr.is_null() || *num == 0 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let n = *num;
    if clustered && !n.is_multiple_of(32) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let Some(bytes) = n.checked_mul(PHYS_PAGE).filter(|&b| b <= 1 << 40) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let Some(section) = create_section(bytes) else {
        crate::xlog_warn!("AllocateTitlePhysicalPages({n}): no section (error {})", fnd::GetLastError());
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let Some(view) = view_anywhere(section, 0, bytes) else {
        crate::xlog_warn!("AllocateTitlePhysicalPages({n}): no view of its section (error {})", fnd::GetLastError());
        fnd::CloseHandle(section as HANDLE);
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let Some(first) = take_page_indices(&NEXT_PAGE_INDEX, n, clustered) else {
        mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut c_void });
        fnd::CloseHandle(section as HANDLE);
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    lazy::view(view, bytes);
    crate::memmap::reserved(view, bytes);
    for i in 0..n {
        *arr.add(i) = page_number(first + i);
    }
    PHYS.lock().unwrap().blocks.insert(first, Block { count: n, live: n, freed: vec![0; n.div_ceil(Block::WORD_BITS)], section: Arc::new(Section { raw: section, len: bytes }), clustered, section_home: None, view, uma_home: None });
    1
}

pub unsafe extern "system" fn FreeTitlePhysicalPages(_p: HANDLE, n: usize, arr: *mut usize) -> BOOL {
    if arr.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // freed pages still mapped somewhere: those addresses keep showing them until they are
    // unmapped (a view keeps its section)
    let mut g = PHYS.lock().unwrap();
    g.site = "FreeTitlePhysicalPages";
    for i in 0..n {
        let page = *arr.add(i);
        let Some(index) = page_index(page) else { continue };
        let Some((&first, b)) = g.blocks.range_mut(..=index).next_back() else { continue };
        let released = b.release(index - first);
        let emptied = released && b.live == 0;
        if released {
            g.note("freed:", 0, 0, page);
        }
        if !emptied {
            continue;
        }
        let b = g.blocks.remove(&first).unwrap();
        g.note("every page of the allocation freed, XCL's view of it unmapped: first", b.view, b.count * PHYS_PAGE, page_number(first));
        crate::alias::mapping_changed("physical pages are freed (XCL's view of them unmapped)", "FreeTitlePhysicalPages", b.view, b.count * PHYS_PAGE);
        lazy::unview(b.view);
        lazy::forget_fills(b.view, b.count * PHYS_PAGE);
        USED.write().unwrap_or_else(|e| e.into_inner()).retain(|&h| h < b.view || h >= b.view + b.count * PHYS_PAGE);
        // (forgotten before it goes: once unmapped, the address may be another thread's)
        crate::memmap::released(b.view);
        mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: b.view as *mut c_void });
        if let Some((home, _)) = b.section_home.filter(|&(h, id)| g.cpu_reservation(h).is_some_and(|r| r.id == id)) {
            free_reservation(&mut g, home, 0, mem::MEM_RELEASE);
        }
        if let (Some(h), Some(u)) = (b.uma_home, crate::gpu::uma::get()) {
            // (the unified memory the allocation had goes back: nothing stays mapped in it)
            for at in g.mapped_in(h, h + b.count * PHYS_PAGE) {
                g.unmap_at(at);
            }
            if u.is_sparse(h) {
                crate::hle::d3d12x::sparse::free(h);
            } else {
                u.free(h);
            }
        }
    }
    1
}

pub unsafe extern "system" fn MapTitlePhysicalPages(va: *mut c_void, n: usize, _t: u32, p: u32, arr: *mut usize) -> *mut c_void {
    let t0 = std::time::Instant::now();
    let mut g = PHYS.lock().unwrap();
    let locked = t0.elapsed();
    g.site = "MapTitlePhysicalPages";
    let r = map_pages(&mut g, va, n, arr, Some(p));
    trace_map("MapPhysicalPages", va as usize, n, arr, r as usize, (locked, t0.elapsed()));
    r
}

/// Whether a physical page is mapped at the page holding `va` – a view or a copy, used or not
/// yet (`lazy`: a page never used is committed at its first access).
pub fn physical_page_at(va: usize) -> bool {
    PHYS.lock().unwrap().at.contains_key(&(va & !(PHYS_PAGE - 1)))
}

/// Counts the calls that map or unmap physical pages in a GPU view (`mirrors`): what
/// `physical_page_at` says there may have changed.
static GPU_VIEW_MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation of the physical-page mappings inside GPU views.
pub fn gpu_view_map_gen() -> u64 {
    GPU_VIEW_MAP_GEN.load(Ordering::Acquire)
}

unsafe fn map_pages(g: &mut Phys, va: *mut c_void, n: usize, arr: *mut usize, legacy_protection: Option<u32>) -> *mut c_void {
    if n == 0 || (va.is_null() && arr.is_null()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    let pages: Vec<usize> = if arr.is_null() { vec![0; n] } else { (0..n).map(|i| *arr.add(i)).collect() };
    if pages.iter().any(|&p| p != 0 && g.block_of(p).is_none()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    let base = if va.is_null() {
        // consecutive pages of one allocation at no chosen address: unified memory of its own
        let consecutive = pages[0] != 0 && (1..n).all(|i| pages[0].checked_add(i * FRAMES) == Some(pages[i]));
        let same_block = consecutive && g.block_of(pages[n - 1]).zip(g.block_of(pages[0])).is_some_and(|((a, _), (b, _))| std::ptr::eq(a, b));
        let Some(home) = same_block.then(|| match legacy_protection {
            Some(p) => g.home_for_mapping(pages[0], p),
            None => g.uma_home(pages[0]),
        }).flatten() else {
            fnd::SetLastError(fnd::ERROR_NOT_SUPPORTED);
            return std::ptr::null_mut();
        };
        home
    } else {
        va as usize
    };
    if !base.is_multiple_of(PHYS_PAGE) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    // (no longer the GPU's view of anything there)
    if g.cpu_reservation(base).is_none() {
        crate::hle::d3d12x::page_mappings::forget(base, n * PHYS_PAGE);
    }
    if super::mirrors::source(base).is_some() {
        GPU_VIEW_MAP_GEN.fetch_add(1, Ordering::Release);
    }
    for (i, &p) in pages.iter().enumerate() {
        let at = base + i * PHYS_PAGE;
        if p == 0 {
            if !g.unmap_at(at) {
                return std::ptr::null_mut();
            }
        } else if !g.map_at(at, p) {
            return std::ptr::null_mut();
        }
    }
    base as *mut c_void
}

/// `BOOL XMemAllocatePhysicalPages(XALLOC_ATTRIBUTES, SIZE_T* count, ULONG64* pages)`: the
/// title physical pages above (64 KiB)
pub unsafe extern "system" fn XMemAllocatePhysicalPages(flags: u64, num: *mut usize, arr: *mut u64) -> BOOL {
    allocate_pages(num, arr.cast(), SECTION_BACKEND.get().is_some_and(|abi| (abi.clustered_allocation)(flags)))
}

/// `void* XMemMapPhysicalPages(void* address, SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemMapPhysicalPages(va: *mut c_void, n: usize, arr: *mut u64) -> *mut c_void {
    let t0 = std::time::Instant::now();
    let mut g = PHYS.lock().unwrap();
    let locked = t0.elapsed();
    if let Some(cpu) = g.cpu_reservation(va as usize) {
        g.site = "XMemMapPhysicalPages";
        return map_gdk(&mut g, cpu, va as usize, n, arr).map_or(std::ptr::null_mut(), |_| va);
    }
    g.site = "MapTitlePhysicalPages";
    let r = map_pages(&mut g, va, n, arr.cast(), None);
    trace_map("MapPhysicalPages", va as usize, n, arr.cast(), r as usize, (locked, t0.elapsed()));
    r
}

unsafe fn map_gdk(g: &mut Phys, cpu: CpuReservation, at: usize, n: usize, arr: *const u64) -> Option<()> {
    let grain = cpu.class.grain.bytes();
    let fail = || {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        None
    };
    if !cpu.accepts_physical_map || n == 0 || !at.is_multiple_of(grain) || n > (cpu.base + cpu.len - at) / grain {
        return fail();
    }
    let mut tokens = Vec::new();
    if tokens.try_reserve_exact(n * (grain / PHYS_PAGE)).is_err() {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return None;
    }
    for i in 0..n {
        let head = if arr.is_null() { 0 } else { *arr.add(i) as usize };
        if head != 0 {
            let Some((b, offset)) = g.block_of(head) else { return fail() };
            if grain > PHYS_PAGE && (!b.clustered || !head.is_multiple_of(grain >> 12) || !offset.is_multiple_of(grain)) {
                return fail();
            }
            if grain > b.count * PHYS_PAGE - offset {
                return fail();
            }
            for j in 0..grain / PHYS_PAGE {
                if !b.is_live(offset / PHYS_PAGE + j) {
                    return fail();
                }
                let token = head + j * FRAMES;
                if g.places(token).iter().any(|&old_at| {
                    !(at..at + n * grain).contains(&old_at)
                        && g.cpu_reservation(old_at).is_some_and(|c| c.cache() != cpu.cache())
                }) {
                    return fail();
                }
                tokens.push(token);
            }
        } else {
            tokens.resize(tokens.len() + grain / PHYS_PAGE, 0);
        }
    }
    for (i, token) in tokens.into_iter().enumerate() {
        let va = at + i * PHYS_PAGE;
        if !(if token == 0 { g.unmap_at(va) } else { g.map_at(va, token) }) {
            return None;
        }
    }
    Some(())
}

/// `BOOL XMemMapPhysicalPagesScatter(void** addresses, SIZE_T count, ULONG64* pages)`: one 64 KiB
/// page at each address (page 0: the address is unmapped; no page array: every address is, as
/// with `MapTitlePhysicalPages` – a retail game unmaps its streaming pages this way and asserts
/// on failure)
pub unsafe extern "system" fn XMemMapPhysicalPagesScatter(addresses: *const *mut c_void, n: usize, pages: *mut u64) -> BOOL {
    if addresses.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let t0 = std::time::Instant::now();
    let mut g = PHYS.lock().unwrap();
    let locked = t0.elapsed();
    g.site = "XMemMapPhysicalPagesScatter";
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if pages.is_null() && *KEEP.get_or_init(|| std::env::var("XCL_SCATTER_UNMAP").is_ok_and(|v| v == "0")) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_info!("XCL_SCATTER_UNMAP=0: XMemMapPhysicalPagesScatter with no page array leaves the pages mapped"));
        return 1;
    }
    let page = |i: usize| if pages.is_null() { std::ptr::null_mut() } else { pages.add(i) as *mut usize };
    for i in 0..n {
        let at = *addresses.add(i) as usize;
        let t1 = std::time::Instant::now();
        let r = if let Some(cpu) = g.cpu_reservation(at) {
            if cpu.class.grain != VirtualGrain::K64 {
                fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
                return 0;
            }
            map_gdk(&mut g, cpu, at, 1, page(i).cast()).map_or(std::ptr::null_mut(), |_| at as *mut c_void)
        } else {
            map_pages(&mut g, at as *mut c_void, 1, page(i), None)
        };
        trace_map("MapPhysicalPagesScatter", at, 1, page(i), r as usize, (if i == 0 { locked } else { Default::default() }, t1.elapsed()));
        if r.is_null() {
            return 0;
        }
    }
    1
}

/// `BOOL XMemFreePhysicalPages(SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemFreePhysicalPages(n: usize, arr: *mut u64) -> BOOL {
    FreeTitlePhysicalPages(std::ptr::null_mut(), n, arr as *mut usize)
}

// --- the title's reservations of 64 KiB pages ---------------------------------------------

/// A reservation of 64 KiB pages for the title (MEM_RESERVE | MEM_64K_PAGES, no address chosen)
/// as a placeholder, so the pages it maps there are views (None: Windows refused).
// The GDK physical ABI scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) unsafe fn reserve(size: usize) -> Option<usize> {
    reserve_compat(size, None)
}

pub(super) unsafe fn reserve_compat(size: usize, raw_gdk64: Option<RawGdk64>) -> Option<usize> {
    let len = size.checked_next_multiple_of(PHYS_PAGE).filter(|&l| l > 0)?;
    let base = reserve_placeholder(len)?;
    crate::memmap::reserved(base, len);
    PHYS.lock().unwrap().reservations.insert(base, Reservation { end: base + len, slots: BTreeMap::new(), cpu: None, raw_gdk64 });
    Some(base)
}

/// Whether `a` is in one of those reservations.
pub(crate) fn in_reservation(a: usize) -> bool {
    PHYS.lock().unwrap().in_reservation(a)
}

/// The title commits `[a, a + size)` of one of those reservations (None: `a` is not in one):
/// private memory in the slots that hold no page, the page's own memory in those that do.
pub(crate) unsafe fn commit_in_reservation(a: usize, size: usize, protect: u32) -> Option<*mut c_void> {
    let mut g = PHYS.lock().unwrap();
    if g.cpu_reservation(a).is_some() {
        return Some(if g.commit_anonymous(a, size, protect) { a as *mut c_void } else { std::ptr::null_mut() });
    }
    let r = g.reservation(a)?;
    let Some(end) = a.checked_add(size.max(1)).filter(|&e| e <= r.end) else {
        fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
        return Some(std::ptr::null_mut());
    };
    let first = a & !(PHYS_PAGE - 1);
    let mut slot = first;
    let mut failed = false;
    while slot < end {
        let (lo, hi) = (a.max(slot), end.min(slot + PHYS_PAGE));
        if r.slots.get(&slot) != Some(&Slot::View) && !r.private(slot) {
            fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
            failed = true;
            break;
        }
        if mem::VirtualAlloc((lo & !0xFFF) as *const c_void, hi - (lo & !0xFFF), mem::MEM_COMMIT, protect).is_null() {
            failed = true;
            break;
        }
        crate::memmap::committed(lo, hi - lo, protect);
        slot += PHYS_PAGE;
    }
    if slot > first {
        g.site = "VirtualAlloc";
        g.note("the title commits memory in its reservation (private memory in the slots that show no page)", first, slot - first, 0);
    }
    if failed {
        return Some(std::ptr::null_mut());
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::xlog_info!("physical pages: the title commits memory in its reservation of 64 KiB pages at {a:#x} ({size:#x} bytes): private memory there, write-watched"));
    Some(a as *mut c_void)
}

/// The title frees in one of those reservations (None: `a` is not in one): MEM_RELEASE of the
/// whole reservation unmaps its pages and gives the address space back; MEM_DECOMMIT unmaps the
/// pages in the range and decommits the private memory there.
pub(crate) unsafe fn free_in_reservation(a: usize, size: usize, t: u32) -> Option<BOOL> {
    let mut g = PHYS.lock().unwrap();
    free_reservation(&mut g, a, size, t)
}

unsafe fn free_reservation(g: &mut Phys, a: usize, size: usize, t: u32) -> Option<BOOL> {
    let (&base, r) = g.reservations.range(..=a).next_back().filter(|(_, r)| a < r.end)?;
    let end = r.end;
    let legacy = r.cpu.is_none();
    let fail = || {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        Some(0)
    };
    let (release, decommitting) = (t & mem::MEM_RELEASE != 0, t & mem::MEM_DECOMMIT != 0);
    if release == decommitting || (release && (a != base || size != 0)) {
        return fail();
    }
    let grain = r.cpu.map_or(PHYS_PAGE, |cpu| cpu.class.grain.bytes());
    let range_end = if release || size == 0 { end } else if legacy { a.saturating_add(size).min(end) } else {
        a.saturating_add(size).min(end).next_multiple_of(grain)
    };
    let a = if legacy { a } else { a & !(grain - 1) };
    g.site = "VirtualFree";
    g.note(if release { "the title releases its reservation (pages unmapped, private memory gone)" } else { "the title decommits in its reservation (pages unmapped, private memory decommitted)" }, a & !(PHYS_PAGE - 1), range_end - (a & !(PHYS_PAGE - 1)), 0);
    crate::alias::mapping_changed(if release { "the title's reservation of 64 KiB pages is released" } else { "a range of the title's reservation of 64 KiB pages is decommitted" }, "VirtualFree", a & !(PHYS_PAGE - 1), range_end - (a & !(PHYS_PAGE - 1)));
    if legacy {
        crate::hle::d3d12x::page_mappings::forget(a & !(PHYS_PAGE - 1), range_end - (a & !(PHYS_PAGE - 1)));
    }
    let mapped: Vec<usize> = g.at.keys().copied().filter(|&x| x >= (a & !(PHYS_PAGE - 1)) && x < range_end).collect();
    for x in mapped {
        g.unmap_at(x);
    }
    let r = g.reservations.get_mut(&base).unwrap();
    let private: Vec<usize> = r.slots.range(a & !(PHYS_PAGE - 1)..range_end).filter(|(_, &s)| s == Slot::Private).map(|(&x, _)| x).collect();
    if decommitting {
        for x in private {
            let (lo, hi) = (a.max(x), range_end.min(x + PHYS_PAGE));
            decommit(lo, hi - lo);
        }
        return Some(1);
    }
    for x in private {
        r.placeholder(x);
    }
    mem::VirtualFree(base as *mut c_void, end - base, COALESCE);
    // (forgotten before it goes: once released, the address may be another thread's)
    crate::memmap::released(base);
    let ok = mem::VirtualFree(base as *mut c_void, 0, mem::MEM_RELEASE);
    if ok == 0 {
        // (still there: a view Windows would not unmap. The model keeps the reservation.)
        crate::xlog_warn!("physical pages: Windows refused to release a reservation of 64 KiB pages (error {}): kept", fnd::GetLastError());
        crate::memmap::reserved(base, end - base);
        return Some(0);
    }
    g.reservations.remove(&base);
    // (gone with everything in it: no address there shows a page any more)
    for x in g.mapped_in(base, end) {
        if let Some(m) = g.at.remove(&x) {
            if m.kind == Kind::Copy {
                COPIES.fetch_sub(1, Ordering::Release);
            }
            if let Some(v) = g.places.get_mut(&m.page) {
                v.retain(|&a| a != x);
                if v.is_empty() {
                    g.places.remove(&m.page);
                }
            }
            lazy::unview(x);
        }
    }
    Some(ok)
}

/// The title frees or decommits `[start, start + len)` outside those reservations (VirtualFree,
/// XMemFree): the physical pages mapped there are unmapped first, each keeping its data
/// (`unmap_at`) – on the console a mapping goes with its address space. Left in the model, a copy
/// there stayed a place XCL writes the page's data to (`write_physical_pages`, the copies of a
/// page whose last view goes) and takes it from (the page's first view) after the address had
/// become other memory. Returns how many pages were unmapped.
pub(crate) unsafe fn title_freed(start: usize, len: usize, site: &'static str) -> usize {
    if len == 0 || COPIES.load(Ordering::Acquire) == 0 {
        return 0;
    }
    let (lo, hi) = (start & !(PHYS_PAGE - 1), start.saturating_add(len));
    let mut g = PHYS.lock().unwrap();
    let mapped = g.mapped_in(lo, hi);
    if mapped.is_empty() {
        return 0;
    }
    g.site = site;
    g.note("the title frees memory that still shows physical pages (unmapped first, each keeps its data)", lo, hi - lo, 0);
    crate::hle::d3d12x::page_mappings::forget(lo, hi - lo);
    for &at in &mapped {
        g.unmap_at(at);
    }
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!("physical pages: the title freed memory with {} page(s) still mapped in it ({site}): unmapped first, each page keeps its data", mapped.len());
    }
    mapped.len()
}

// --- XCL's access to the pages ------------------------------------------------------------

/// Copy `dst.len()` bytes starting at `offset` within `pages` (each a 64 KiB physical page
/// number from `AllocateTitlePhysicalPages`). Returns `false` if `offset >= PHYS_PAGE_SIZE`,
/// `pages` is too short, or any page is not a live allocation.
pub unsafe fn read_physical_pages(pages: &[usize], offset: usize, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return offset < PHYS_PAGE;
    }
    let g = PHYS.lock().unwrap();
    let Some(need) = span(pages, offset, dst.len()) else { return false };
    if pages[..need].iter().any(|&p| g.block_of(p).is_none()) {
        return false;
    }
    let mut done = 0;
    let mut page_off = offset;
    for &p in &pages[..need] {
        let chunk = (PHYS_PAGE - page_off).min(dst.len() - done);
        match g.owner(p) {
            Some(a) => std::ptr::copy_nonoverlapping((a + page_off) as *const u8, dst.as_mut_ptr().add(done), chunk),
            None => dst[done..done + chunk].fill(0),
        }
        done += chunk;
        page_off = 0;
    }
    true
}

/// Copy `src` into `pages` starting at `offset` within the first page. Validates every page
/// before writing so the destination gets all bytes or nothing.
pub unsafe fn write_physical_pages(pages: &[usize], offset: usize, src: &[u8]) -> bool {
    if src.is_empty() {
        return offset < PHYS_PAGE;
    }
    // (each page's piece hashed before the pages are held: `alias::payload_written_as`)
    let hashes: Vec<u64> = if offset < PHYS_PAGE { crate::alias::pieces(offset, src.len()).map(|(at, n)| crate::alias::hash(&src[at - offset..at - offset + n])).collect() } else { Vec::new() };
    let g = PHYS.lock().unwrap();
    let Some(need) = span(pages, offset, src.len()) else { return false };
    if pages[..need].iter().any(|&p| g.block_of(p).is_none()) {
        return false;
    }
    if pages[..need].iter().any(|&p| g.owner(p).is_none() && !use_page(g.home(p).unwrap())) {
        return false;
    }
    let mut done = 0;
    let mut page_off = offset;
    for (i, &p) in pages[..need].iter().enumerate() {
        let chunk = (PHYS_PAGE - page_off).min(src.len() - done);
        let home = g.home(p).unwrap();
        let owner = g.owner(p).unwrap_or_else(|| {
            use_page(home);
            home
        });
        // the data's place, and the copies too while the section has it
        let mut targets = vec![owner];
        if owner == home {
            targets.extend(g.copies(p));
        }
        for &t in &targets {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(done), (t + page_off) as *mut u8, chunk);
            crate::eh::note_guest_write("physical page write", t + page_off, chunk);
            let kind = if t == home { "written through XCL's view of its section: a physical destination" } else { "written through its copy: a physical destination" };
            crate::alias::payload_written_as(t + page_off, &src[done..done + chunk], hashes[i], Some(crate::alias::Backing { page: p, kind }));
        }
        // every address showing the page: the textures there are uploaded again
        for &a in g.places(p) {
            crate::hle::d3d12x::title_textures::mark_cpu_written(a + page_off, chunk);
        }
        done += chunk;
        page_off = 0;
    }
    true
}

fn span(pages: &[usize], offset: usize, len: usize) -> Option<usize> {
    if offset >= PHYS_PAGE {
        return None;
    }
    let need = offset.checked_add(len)?.div_ceil(PHYS_PAGE);
    (pages.len() >= need).then_some(need)
}

/// XCL (or a kernel call it made for the title) wrote `len` bytes at `va`: memory in a view of a
/// physical page is not write-watched, so the next submit takes it as written (with every other
/// address showing the page: `alias_ranges`).
pub(crate) fn note_written(va: usize, len: usize) {
    if len > 0 && lazy::overlaps_view(va, len) {
        crate::hle::d3d12x::title_textures::mark_cpu_written(va, len);
    }
}

/// For memory written at `ranges` (address, bytes): the same bytes at the other addresses showing
/// the pages there (views of one page are one memory).
pub(crate) fn alias_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if !ranges.iter().any(|&(a, n)| lazy::overlaps_view(a, n)) {
        return out;
    }
    let g = PHYS.lock().unwrap();
    for &(a, n) in ranges {
        let end = a.saturating_add(n);
        let mut x = a;
        while x < end {
            let slot = x & !(PHYS_PAGE - 1);
            let hi = end.min(slot + PHYS_PAGE);
            if let Some(m) = g.at.get(&slot).filter(|m| m.kind == Kind::View) {
                for &b in g.places(m.page) {
                    if b != slot && g.at.get(&b).is_some_and(|o| o.kind == Kind::View) {
                        out.push((b + (x - slot), hi - x));
                    }
                }
            }
            x = hi;
        }
    }
    out
}

// --- for the GPU's views (`d3d12x::page_mappings`) ----------------------------------------

/// Runs `f` with the physical pages held still (no page is mapped or unmapped meanwhile).
pub(crate) unsafe fn hold_physical_pages<R>(f: impl FnOnce(&mut Held) -> R) -> R {
    let mut g = PHYS.lock().unwrap();
    g.site = "a GPU page view (CopyPageMappingsBatchX)";
    f(&mut Held(&mut g))
}

/// The physical pages held still (`hold_physical_pages`).
pub(crate) struct Held<'a>(&'a mut Phys);

impl Held<'_> {
    /// Page `page` at the GPU view address `at` as a view of it (false: `at` cannot hold views –
    /// not in one of the title's placeholder reservations – or the page is not allocated).
    pub(crate) unsafe fn view(&mut self, at: usize, page: usize) -> bool {
        self.0.in_reservation(at) && self.0.block_of(page).is_some() && self.0.map_at(at, page)
    }

    /// Whatever page is mapped at `at` unmapped (true: there was one).
    pub(crate) unsafe fn unmap(&mut self, at: usize) -> bool {
        let was = self.0.at.contains_key(&at);
        self.0.unmap_at(at);
        was
    }

    /// The 64 KiB at `at` as memory of its own for a GPU view XCL fills itself (a pool slot that
    /// shows no page): committed (false: it could not be).
    pub(crate) unsafe fn claim(&mut self, at: usize) -> bool {
        if !self.0.unmap_at(at) {
            return false;
        }
        self.0.note("claimed as memory of its own for a GPU view XCL fills (a copied view)", at, PHYS_PAGE, 0);
        if let Some(r) = self.0.reservation(at) {
            if !r.private(at) {
                return false;
            }
            return commit(at, PHYS_PAGE);
        }
        let Some(u) = crate::gpu::uma::get() else { return false };
        if u.is_committed(at) || committed(at) {
            return true;
        }
        if !commit(at, PHYS_PAGE) {
            return false;
        }
        if u.is_sparse(at) {
            crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
        }
        true
    }

    /// The 64 KiB at `at` that `claim` committed given back: decommitted, its slot a placeholder
    /// again (`claim`'s inverse; false: it is not such a page any more).
    pub(crate) unsafe fn release(&mut self, at: usize) -> bool {
        if let Some(r) = self.0.reservation(at) {
            return r.slots.get(&at) == Some(&Slot::Private) && r.placeholder(at);
        }
        let uma = crate::gpu::uma::get();
        if uma.is_some_and(|u| u.is_committed(at)) {
            return false;
        }
        if uma.is_some_and(|u| u.is_sparse(at)) {
            crate::hle::d3d12x::sparse::decommitted(at, PHYS_PAGE);
        }
        decommit(at, PHYS_PAGE);
        true
    }

    /// The physical page the address `va` shows.
    pub(crate) fn page_at(&self, va: usize) -> Option<usize> {
        self.0.page_at(va)
    }

    /// (diagnostics) The GPU view at `at` is about to show `page`, which `slot` shows: kept in
    /// the map history when no page is mapped at `slot` – it is `page`'s own place in its
    /// allocation (`page_at`), whatever the title has mapped the page for since.
    pub(crate) fn note_source(&mut self, at: usize, slot: usize, page: usize) {
        if !self.0.at.contains_key(&(slot & !(PHYS_PAGE - 1))) {
            self.0.note("a GPU view here takes its page from a pool slot where no page is mapped (the page's own place in its allocation):", at, PHYS_PAGE, page);
        }
    }

    /// Where page `p`'s data is (None: not allocated, or never used – zeros).
    pub(crate) fn location(&self, p: usize) -> Option<usize> {
        self.0.owner(p)
    }
}

// --- diagnostics --------------------------------------------------------------------------

/// XCL's own bookkeeping of physical pages is consistent (`XCL_DS_VALIDATE`): the blocks do not
/// overlap, the address and page maps agree, and every view is in a slot recorded as one. The
/// first broken rule, if any. (In-memory checks only: no system call per page.)
pub(crate) fn validate_physical() -> Result<(), String> {
    let g = PHYS.lock().unwrap();
    let mut end = 0;
    for (&first, b) in g.blocks.iter() {
        if first < end {
            return Err(format!("page block {first} overlaps the block before it (ends at {end})"));
        }
        if b.live > b.count {
            return Err(format!("page block {first}: {} pages live of {}", b.live, b.count));
        }
        end = first + b.count;
    }
    let listed: usize = g.places.values().map(|v| v.len()).sum();
    if listed != g.at.len() {
        return Err(format!("{} addresses mapped but {listed} listed by page", g.at.len()));
    }
    for (&at, m) in &g.at {
        if !g.places(m.page).contains(&at) {
            return Err(format!("{at:#x} shows page {:#x}, which does not list it", m.page));
        }
        if m.kind == Kind::View {
            let slot = g.reservations.range(..=at).next_back().filter(|(_, r)| at < r.end).and_then(|(_, r)| r.slots.get(&at).copied());
            if slot != Some(Slot::View) {
                return Err(format!("{at:#x} is a view of page {:#x} but its slot is not recorded as one", m.page));
            }
        }
    }
    Ok(())
}

/// What `va` is in XCL's model of physical pages (diagnostics).
pub(crate) fn describe_address(va: usize) -> String {
    let at = va & !(PHYS_PAGE - 1);
    let g = PHYS.lock().unwrap();
    if let Some(m) = g.at.get(&at) {
        let n = g.places(m.page).len();
        return format!(
            "page {:#x} mapped here as a {}{} ({n} places)",
            m.page,
            if m.kind == Kind::View { "view" } else { "copy" },
            if lazy::is_lazy(at) { ", not used yet" } else { "" }
        );
    }
    match g.page_at(at) {
        Some(p) => format!("XCL's view of page {p:#x}"),
        None if g.in_reservation(at) => "a placeholder of a reservation of 64 KiB pages (no page)".into(),
        None => "no physical page".into(),
    }
}

/// The physical page the title's address `va` shows, for a payload written there
/// (`alias::payload_written`): the page mapped at it, or a slot of one of the title's reservations
/// that shows none. None: plain memory. (A lookup of the address only: no search of the
/// allocations.)
pub(crate) fn backing(va: usize) -> Option<crate::alias::Backing> {
    let at = va & !(PHYS_PAGE - 1);
    let g = PHYS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = g.at.get(&at) {
        return Some(crate::alias::Backing { page: m.page, kind: kind_name(m.kind, false) });
    }
    g.in_reservation(at).then_some(crate::alias::Backing { page: 0, kind: "a slot of the title's reservation of 64 KiB pages that shows no physical page" })
}

/// (diagnostics) Where physical page `page`'s data is now: its section through XCL's view, or its
/// copy in use while it has no view. None: freed with its whole allocation, never used, or the
/// model is locked by another thread.
pub(crate) fn page_data(page: usize) -> Option<usize> {
    crate::alias::try_lock(&PHYS)?.owner(page)
}

/// XCL's account of the 64 KiB page at an address of a crash record (`alias::report`).
pub(crate) struct AliasFacts {
    pub what: String,
    /// the physical page shown there (page 0: a slot of a reservation that shows none)
    pub backing: Option<crate::alias::Backing>,
    /// the other addresses showing the page (64 KiB aligned), and as what
    pub aliases: Vec<(usize, String)>,
    /// the latest maps and unmaps of the address / of the page elsewhere, the latest first
    pub history_here: Vec<String>,
    pub history_page: Vec<String>,
}

/// What the 64 KiB page at `va` is in the model of physical pages: the page it shows, the page's
/// allocation, where its data is, every other address showing it, and the latest changes. None:
/// nothing (plain memory no page was mapped at within the history kept), or the model is locked
/// by another thread – at a crash it is not waited for.
pub(crate) fn alias_facts(va: usize) -> Option<AliasFacts> {
    alias_facts_inner(va, &PHYS, false)
}

pub(crate) fn crash_alias_facts(va: usize) -> Option<AliasFacts> {
    alias_facts_inner(va, LazyLock::get(&PHYS)?, true)
}

fn alias_facts_inner(va: usize, phys: &Mutex<Phys>, compact: bool) -> Option<AliasFacts> {
    let at = va & !(PHYS_PAGE - 1);
    let g = crate::alias::try_lock(phys)?;
    let mapped = g.at.get(&at);
    let page = mapped.map(|m| m.page).or_else(|| g.page_at(at));
    let slot = g.reservations.range(..=at).next_back().filter(|(_, r)| at < r.end).map(|(&base, r)| (base, r.end, r.slots.get(&at).copied()));
    let history_here: Vec<String> = g.history.at(at, 8).iter().map(|e| e.to_string()).collect();
    if page.is_none() && slot.is_none() && history_here.is_empty() {
        return None;
    }
    let mut what = match (mapped, page) {
        (Some(m), _) => format!("physical page {:#x} is mapped here as a {}", m.page, kind_name(m.kind, !compact && lazy::is_lazy(at))),
        (None, Some(p)) => format!("XCL's own view of physical page {p:#x} (its place in its allocation's view, or its unified-memory home)"),
        (None, None) => "no physical page is mapped here".to_string(),
    };
    if let Some((base, end, s)) = slot {
        let s = match s {
            None => "never split out of it",
            Some(Slot::Placeholder) => "a placeholder now",
            Some(Slot::View) => "a view",
            Some(Slot::Private) => "private memory",
        };
        what += &format!("; a slot of the title's reservation of 64 KiB pages {base:#x}..{end:#x} ({s})");
    }
    let mut aliases = Vec::new();
    let mut history_page = Vec::new();
    if let Some(p) = page {
        match g.backing_block(p) {
            Some((b, offset)) => {
                let first = page_number(page_index(p).map_or(0, |i| i - offset));
                what += &format!("; the page is page {offset} of an allocation of {} from page {first:#x}, {} of them live{}", b.count, b.live, if b.is_live(offset) { "" } else { " – THIS PAGE WAS FREED by the title" });
            }
            None => what += "; the page's allocation is gone (all its pages were freed)",
        }
        let home = g.home(p);
        what += &match g.owner(p) {
            Some(o) if Some(o) == home => format!("; its data is in its section (XCL's view of it: {o:#x})"),
            Some(o) => format!("; its data is in its copy at {o:#x} (it has no view)"),
            None => "; it was never used (zeros)".to_string(),
        };
        if !compact {
            if let Some(h) = home.filter(|&h| h != at) {
                aliases.push((h, "XCL's view of the page's section".to_string()));
            }
            for &a in g.places(p) {
                if a != at {
                    aliases.push((a, g.at.get(&a).map_or("?", |m| kind_name(m.kind, lazy::is_lazy(a))).to_string()));
                }
            }
        }
        history_page = g.history.of_page(p, at, 8).iter().map(|e| e.to_string()).collect();
    }
    let backing = match (mapped, page) {
        (Some(m), _) => Some(crate::alias::Backing { page: m.page, kind: kind_name(m.kind, false) }),
        (None, Some(p)) => Some(crate::alias::Backing { page: p, kind: "XCL's own view of it" }),
        (None, None) => slot.map(|_| crate::alias::Backing { page: 0, kind: "a slot of the title's reservation of 64 KiB pages that shows no physical page" }),
    };
    Some(AliasFacts { what, backing, aliases, history_here, history_page })
}

/// At how many addresses physical page `page` is mapped now.
pub(crate) fn page_places(page: usize) -> usize {
    PHYS.lock().unwrap().places(page).len()
}

/// Whether every one of `pages` is an allocated physical page.
pub(crate) fn pages_allocated(pages: &[usize]) -> bool {
    let g = PHYS.lock().unwrap();
    pages.iter().all(|&p| g.block_of(p).is_some())
}

/// (diagnostics, `memdiag`) The map history: (events, bytes of one, bytes the ring holds).
pub(crate) fn diag_map_history() -> (usize, usize, usize) {
    let g = PHYS.lock().unwrap();
    (g.history.len(), std::mem::size_of::<crate::alias::MapEvent>(), g.history.bytes())
}

/// `XCL_TRACE_MAPS=1`: every physical page map and unmap (`what`, the address range, the pages,
/// the result); a failed map is logged always.
pub(crate) fn trace_maps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_MAPS").is_ok_and(|v| v != "0"))
}

/// `took`: (waiting for the physical pages' lock, the whole call)
unsafe fn trace_map(what: &str, va: usize, n: usize, arr: *const usize, result: usize, took: (std::time::Duration, std::time::Duration)) {
    if result != 0 && !trace_maps() {
        return;
    }
    let pages: Vec<usize> = if arr.is_null() { Vec::new() } else { (0..n.min(4)).map(|i| *arr.add(i)).collect() };
    let pages = if arr.is_null() { "none (unmap)".to_string() } else if pages.iter().all(|&p| p == 0) && n <= 4 { "0 (unmap)".to_string() } else { format!("{pages:x?}{}", if n > 4 { " …" } else { "" }) };
    let range = format!("{va:#x}..{:#x}", va + n * PHYS_PAGE);
    if result == 0 {
        crate::xlog_warn!("map trace: {what} {range} ({n} pages {pages}) failed (error {})", fnd::GetLastError());
    } else {
        crate::xlog_info!("map trace: {what} {range} ({n} pages {pages}) → {result:#x} in {:?} ({:?} waiting for the lock)", took.1, took.0);
    }
}

/// (tests) pages physical pages shown at two reserved addresses: (address, alias, pages).
#[cfg(test)]
pub(crate) unsafe fn test_alias_views(pages: usize) -> Option<(usize, usize, Vec<usize>)> {
    let mut n = pages;
    let mut ids = vec![0usize; pages];
    if allocate_pages(&mut n, ids.as_mut_ptr(), false) == 0 {
        return None;
    }
    let at = reserve(pages * PHYS_PAGE)?;
    let alias = reserve(pages * PHYS_PAGE)?;
    hold_physical_pages(|h| (0..pages).all(|i| h.view(at + i * PHYS_PAGE, ids[i]) && h.view(alias + i * PHYS_PAGE, ids[i]))).then_some((at, alias, ids))
}

/// (tests) What test_alias_views made goes.
#[cfg(test)]
pub(crate) unsafe fn test_free_alias_views(at: usize, alias: usize, mut ids: Vec<usize>) {
    free_in_reservation(at, 0, mem::MEM_RELEASE);
    free_in_reservation(alias, 0, mem::MEM_RELEASE);
    FreeTitlePhysicalPages(std::ptr::null_mut(), ids.len(), ids.as_mut_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_segment_excludes_holes_and_obeys_live_segment_boundaries() {
        unsafe {
            let mut n = 8;
            let mut ids = [0usize; 8];
            assert_ne!(allocate_pages(&mut n, ids.as_mut_ptr(), false), 0);
            let at = reserve(3 * PHYS_PAGE).unwrap();
            assert!(hold_physical_pages(|h| h.view(at, ids[2]) && h.view(at + PHYS_PAGE, ids[3]) && h.view(at + 2 * PHYS_PAGE, ids[6])));
            let get = |va| match snapshot_queue_sources(&[va]).unwrap().remove(0) {
                QueuePageSource::Physical(page) => page,
                _ => panic!("expected a committed physical page"),
            };
            let p = get(at + PHYS_PAGE);
            let (off, view) = p.pin_committed_segment(0..PHYSICAL_SEGMENT_BYTES).unwrap();
            assert_eq!((off, view.len()), (2 * PHYS_PAGE, 2 * PHYS_PAGE));
            assert_eq!(get(at + 2 * PHYS_PAGE).pin_committed_segment(0..PHYSICAL_SEGMENT_BYTES).unwrap().0, 6 * PHYS_PAGE);
            assert!(hold_physical_pages(|h| h.view(at, ids[4]) && h.view(at + PHYS_PAGE, ids[5])));
            let joined = get(at);
            let (off, clipped) = joined.pin_committed_segment(4 * PHYS_PAGE..6 * PHYS_PAGE).unwrap();
            assert_eq!((off, clipped.len()), (4 * PHYS_PAGE, 2 * PHYS_PAGE));
            assert!(joined.pin_committed_segment(6 * PHYS_PAGE..8 * PHYS_PAGE).is_none());
            assert_eq!(free_in_reservation(at, 0, mem::MEM_RELEASE), Some(1));
            assert_ne!(FreeTitlePhysicalPages(std::ptr::null_mut(), n, ids.as_mut_ptr()), 0);
            assert_eq!(p.pin_committed_segment(0..PHYSICAL_SEGMENT_BYTES).unwrap().1.len(), 5 * PHYS_PAGE);
        }
    }

    #[test]
    fn clustered_reservation_maps_cluster_heads_and_retains_queue_sources() {
        unsafe {
            let cluster = 32 * PHYS_PAGE;
            let mut n = 64usize;
            let mut ids = [0usize; 64];
            assert_ne!(allocate_pages(&mut n, ids.as_mut_ptr(), true), 0);
            assert_eq!(ids[0] % (cluster >> 12), 0);
            let class = MapClass { grain: VirtualGrain::M2, default_gpu: true };
            let reserve = |addr, access| cpu_reserve(&mut PHYS.lock().unwrap(), addr, 2 * cluster, mem::MEM_RESERVE, 0, access, class);
            let va = reserve(0, mem::PAGE_READWRITE).unwrap();
            assert_eq!(va % cluster, 0);
            assert!(reserve(va, mem::PAGE_READWRITE).is_none());
            let ro = reserve(0, mem::PAGE_READONLY).unwrap();
            let mut heads = [ids[32] as u64, ids[0] as u64];
            assert_eq!(XMemMapPhysicalPages(va as *mut _, 2, heads.as_mut_ptr()) as usize, va);
            assert_eq!(XMemMapPhysicalPages(ro as *mut _, 2, heads.as_mut_ptr()) as usize, ro);
            *((va + 31 * PHYS_PAGE) as *mut u32) = 0xafa513;
            assert_eq!(*((ro + 31 * PHYS_PAGE) as *const u32), 0xafa513);
            let mut mbi = mem::MEMORY_BASIC_INFORMATION::default();
            assert_ne!(mem::VirtualQuery(ro as *const _, &mut mbi, std::mem::size_of_val(&mbi)), 0);
            assert_eq!(mbi.Protect, mem::PAGE_READONLY);
            let take = |s| match s { QueuePageSource::Physical(p) => p, _ => panic!("not physical") };
            let sources = snapshot_queue_sources(&[va, va + 31 * PHYS_PAGE, ro + 31 * PHYS_PAGE]).unwrap();
            let old = take(sources[1].clone());
            assert_eq!(old.id(), ids[63]);
            assert_eq!(take(sources[2].clone()).id(), old.id());
            assert!(take(sources[0].clone()).same_section(&old));
            assert!(matches!(snapshot_default_gpu_sources(&[va]).unwrap()[0], Some(QueuePageSource::Physical(_))));
            assert!(XMemMapPhysicalPagesScatter([va as *mut _].as_ptr(), 1, heads.as_mut_ptr()) == 0);
            let mut unaligned = ids[1] as u64;
            assert!(XMemMapPhysicalPages(va as *mut _, 1, &mut unaligned).is_null());
            assert_eq!(XMemMapPhysicalPages(va as *mut _, 2, std::ptr::null_mut()) as usize, va);
            assert!(matches!(snapshot_queue_sources(&[va]).unwrap()[0], QueuePageSource::Hole));
            assert!(matches!(snapshot_default_gpu_sources(&[va]).unwrap()[0], Some(QueuePageSource::Hole)));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), n, ids.as_mut_ptr()), 1);
            assert_eq!(take(snapshot_queue_sources(&[ro + 31 * PHYS_PAGE]).unwrap().remove(0)).id(), ids[63]);
            assert_eq!(free_in_reservation(va, 0, mem::MEM_RELEASE), Some(1));
            assert_eq!(free_in_reservation(ro, 0, mem::MEM_RELEASE), Some(1));
            let pin = pin_physical_view_run(&[old]).unwrap();
            assert_eq!(*(pin.base() as *const u32), 0xafa513);
        }
    }

    #[test]
    fn legacy_section_home_is_stable_and_default_gpu_mapping_is_selective() {
        unsafe {
            let mut n = 2usize;
            let mut ids = [0usize; 2];
            assert_ne!(allocate_pages(&mut n, ids.as_mut_ptr(), false), 0);
            let mut g = PHYS.lock().unwrap();
            let home = g.section_home_for_mapping(ids[0], mem::PAGE_READWRITE).unwrap();
            assert_eq!(g.section_home_for_mapping(ids[1], mem::PAGE_READWRITE), Some(home + PHYS_PAGE));
            assert!(g.map_at(home, ids[0]));
            let cls = MapClass { grain: VirtualGrain::K64, default_gpu: false };
            let cpu = cpu_reserve(&mut g, 0, PHYS_PAGE, mem::MEM_RESERVE, 0, mem::PAGE_READWRITE, cls).unwrap();
            assert!(g.map_at(cpu, ids[0]));
            drop(g);
            *(home as *mut u32) = 77;
            assert_eq!(*(cpu as *const u32), 77);
            let defaults = snapshot_default_gpu_sources(&[cpu, home]).unwrap();
            assert!(defaults[0].is_none());
            assert!(matches!(defaults[1], Some(QueuePageSource::Physical(_))));
            assert_eq!(free_in_reservation(cpu, 0, mem::MEM_RELEASE), Some(1));
            assert_ne!(FreeTitlePhysicalPages(std::ptr::null_mut(), n, ids.as_mut_ptr()), 0);
            assert!(cpu_reservation_at(home).is_none());
        }
    }

    #[test]
    fn compatibility_storage_records_raw_xmem_request_without_granting_mapping_rights() {
        unsafe {
            let flags = 0xca73_4000_0000_0000u64;
            let allocation = mem::MEM_RESERVE | super::super::MEM_64K_PAGES;
            let access = mem::PAGE_READWRITE;
            let bytes = 5 * crate::gpu::uma::WINDOW;
            let v = super::super::XMemVirtualAlloc(std::ptr::null(), bytes, allocation, flags, access) as usize;
            let (base, len, req) = compat_gdk64_reservation_at(v + PHYS_PAGE).unwrap();
            assert_eq!((base, len, req.requested, req.requested_size, req.allocation, req.flags, req.protection), (v, bytes, 0, bytes, allocation, flags, access));
            assert!(cpu_reservation_at(v).is_none());
            assert!(snapshot_default_gpu_sources(&[v]).unwrap()[0].is_none());
            let mut n = 1;
            let mut id = 0u64;
            assert_ne!(XMemAllocatePhysicalPages(0, &mut n, &mut id), 0);
            assert_eq!(XMemMapPhysicalPages(v as _, 1, &mut id) as usize, v);
            let page = match snapshot_queue_sources(&[v]).unwrap().remove(0) { QueuePageSource::Physical(p) => p, _ => unreachable!() };
            *(v as *mut u32) = 0x6125;
            let next = v + PHYS_PAGE;
            assert_eq!(super::super::VirtualAlloc((next + 4096) as _, 4096, mem::MEM_COMMIT, access) as usize, next + 4096);
            *((next + 4096) as *mut u32) = 7;
            assert_eq!(written_pages(next, PHYS_PAGE), vec![next + 4096]);
            assert_ne!(super::super::VirtualFree(v as _, 0, mem::MEM_RELEASE), 0);
            assert!(compat_gdk64_reservation_at(v).is_none());
            assert_ne!(XMemFreePhysicalPages(1, &mut id), 0);
            assert_eq!(*(pin_physical_view_run(&[page]).unwrap().base() as *const u32), 0x6125);
        }
    }

    #[test]
    fn physical_source_snapshot_distinguishes_private_memory_from_a_hole() {
        unsafe {
            let va = reserve(2 * PHYS_PAGE).unwrap();
            assert!(commit_in_reservation(va, PHYS_PAGE, mem::PAGE_READWRITE).is_some_and(|p| p as usize == va));
            let sources = snapshot_queue_sources(&[va, va + PHYS_PAGE]).unwrap();
            assert!(matches!(sources[0], QueuePageSource::GuestRange { base, len: PHYS_PAGE } if base == va));
            assert!(matches!(sources[1], QueuePageSource::Hole));
            assert_eq!(free_in_reservation(va, 0, mem::MEM_RELEASE), Some(1));
        }
    }

    #[test]
    fn anonymous_pool_refs_survive_decommit_recommit_and_release() {
        unsafe {
            let va = cpu_reserve(&mut PHYS.lock().unwrap(), 0, 2 * PHYS_PAGE, mem::MEM_RESERVE, 0, mem::PAGE_READWRITE,
                MapClass { grain: VirtualGrain::K64, default_gpu: true }).unwrap();
            assert_eq!(commit_in_reservation(va, 2 * PHYS_PAGE, mem::PAGE_READWRITE), Some(va as *mut _));
            *(va as *mut u32) = 0xdeca_fbad;
            let refs = |slots: &[usize]| snapshot_queue_sources(slots).unwrap().into_iter().map(|s| match s {
                QueuePageSource::Physical(p) => p, _ => panic!("missing anonymous PFN"),
            }).collect::<Vec<_>>();
            let original = refs(&[va, va + PHYS_PAGE]);
            assert!(original[0].same_section(&original[1]));
            assert_eq!(free_in_reservation(va, PHYS_PAGE, mem::MEM_DECOMMIT), Some(1));
            assert!(matches!(snapshot_queue_sources(&[va]).unwrap()[0], QueuePageSource::Hole));
            assert_eq!(commit_in_reservation(va, PHYS_PAGE, mem::PAGE_READWRITE), Some(va as *mut _));
            let replacement = refs(&[va, va + PHYS_PAGE]);
            assert_ne!(original[0].id(), replacement[0].id());
            assert_eq!(original[1].id(), replacement[1].id());
            assert_eq!(*(va as *const u32), 0);
            assert_eq!(free_in_reservation(va, 0, mem::MEM_RELEASE), Some(1));
            let pin = pin_physical_view_run(&original).unwrap();
            assert_eq!(*(pin.base() as *const u32), 0xdeca_fbad);
        }
    }

    #[test]
    fn abi_classifier_observes_unmodified_gdk_flags() {
        const FIXTURE: u64 = 0x7812_ba32_eb43_9107;
        let abi = GdkPhysicalAbi {
            mappable_reservation: |_, _, _| None,
            source_pool_allocation: |t, flags, p| (flags == FIXTURE && p == mem::PAGE_READWRITE && t == mem::MEM_COMMIT | mem::MEM_RESERVE)
                .then_some(MapClass { grain: VirtualGrain::K64, default_gpu: true }),
            clustered_allocation: |_| false,
        };
        unsafe {
            let t = mem::MEM_RESERVE | mem::MEM_COMMIT;
            assert!(gdk_reserve_with(&abi, std::ptr::null(), PHYS_PAGE, t, 0, mem::PAGE_READWRITE).is_none());
            let va = gdk_reserve_with(&abi, std::ptr::null(), PHYS_PAGE, t, FIXTURE, mem::PAGE_READWRITE).unwrap() as usize;
            let r = cpu_reservation_at(va).unwrap();
            assert_eq!((r.requested, r.base, r.allocation, r.flags, r.protection), (0, va, t, FIXTURE, mem::PAGE_READWRITE));
            assert!(!r.accepts_physical_map);
            let owned = match snapshot_queue_sources(&[va]).unwrap().remove(0) { QueuePageSource::Physical(p) => p, _ => panic!("missing pool owner") };
            *(va as *mut u32) = 0x1234;
            assert_eq!(free_in_reservation(va, 0, mem::MEM_RELEASE), Some(1));
            let pin = pin_physical_view_run(&[owned]).unwrap();
            assert_eq!(*(pin.base() as *const u32), 0x1234);
        }
    }

    /// The pages of `[base, base + len)` written since the last look (write-watched memory).
    unsafe fn written_pages(base: usize, len: usize) -> Vec<usize> {
        extern "system" {
            fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
        }
        let (mut found, mut n, mut page) = (vec![std::ptr::null_mut(); 16], 16usize, 0u32);
        assert_eq!(GetWriteWatch(1, base as *const c_void, len, found.as_mut_ptr(), &mut n, &mut page), 0);
        found[..n].iter().map(|&p| p as usize).collect()
    }

    /// The Windows calls the model rests on: a section committed through one view shows the write
    /// in every other, a view goes into a placeholder split out of a reservation and back.
    #[test]
    fn one_page_is_the_same_memory_at_every_address() {
        unsafe {
            let s = create_section(4 * PHYS_PAGE).unwrap();
            let home = view_anywhere(s, 0, 4 * PHYS_PAGE).unwrap();
            let r = reserve_placeholder(8 * PHYS_PAGE).unwrap();
            assert!(split(r + 2 * PHYS_PAGE));
            assert!(map_into(s, PHYS_PAGE, r + 2 * PHYS_PAGE));
            assert!(split(r + 5 * PHYS_PAGE));
            assert!(map_into(s, PHYS_PAGE, r + 5 * PHYS_PAGE));
            assert!(!committed(r + 2 * PHYS_PAGE), "reserved until used");
            assert!(commit(r + 2 * PHYS_PAGE, PHYS_PAGE));
            std::ptr::write_volatile((r + 2 * PHYS_PAGE + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 5 * PHYS_PAGE + 8) as *const u32), 7);
            assert_eq!(std::ptr::read_volatile((home + PHYS_PAGE + 8) as *const u32), 7);
            assert!(unmap_into_placeholder(r + 2 * PHYS_PAGE));
            assert!(map_into(s, 3 * PHYS_PAGE, r + 2 * PHYS_PAGE), "a placeholder again");
            assert!(private_into(r + 7 * PHYS_PAGE) || (split(r + 7 * PHYS_PAGE) && private_into(r + 7 * PHYS_PAGE)));
            assert!(commit(r + 7 * PHYS_PAGE, PHYS_PAGE));
            for a in [r + 2 * PHYS_PAGE, r + 5 * PHYS_PAGE] {
                assert!(unmap_into_placeholder(a));
            }
            assert_ne!(mem::VirtualFree((r + 7 * PHYS_PAGE) as *mut c_void, PHYS_PAGE, PRESERVE), 0);
            assert_ne!(mem::VirtualFree(r as *mut c_void, 8 * PHYS_PAGE, COALESCE), 0);
            assert_ne!(mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE), 0);
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: home as *mut c_void });
            fnd::CloseHandle(s as HANDLE);
            crate::memmap::decommitted(r, 8 * PHYS_PAGE);
        }
    }

    /// The title's flow: pages mapped into a pool, each mapped again elsewhere and written there,
    /// and a GPU view of the pool slot – all one memory; unmapping leaves the others showing it.
    #[test]
    fn pages_in_a_reservation_are_views_of_one_memory() {
        unsafe {
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            assert!(pages[0].is_multiple_of(16) && pages[1] == pages[0] + 16);
            let pool = reserve(4 * PHYS_PAGE).unwrap();
            let fill = reserve(4 * PHYS_PAGE).unwrap();
            assert!(in_reservation(pool + 3 * PHYS_PAGE));
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 2, 0, 4, pages.as_mut_ptr()) as usize, pool);
            assert_eq!(std::ptr::read_volatile((pool + PHYS_PAGE) as *const u32), 0, "a page never used reads as zeros");
            let at = [(fill + 3 * PHYS_PAGE) as *mut c_void];
            let mut one = [pages[1] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, one.as_mut_ptr()), 1);
            std::ptr::write_volatile((fill + 3 * PHYS_PAGE + 4) as *mut u32, 0xC0FFEE);
            assert_eq!(std::ptr::read_volatile((pool + PHYS_PAGE + 4) as *const u32), 0xC0FFEE, "the pool slot shows the page");
            let view = pool + 2 * PHYS_PAGE;
            assert!(hold_physical_pages(|h| {
                let p = h.page_at(pool + PHYS_PAGE).unwrap();
                h.view(view, p)
            }));
            assert_eq!(std::ptr::read_volatile((view + 4) as *const u32), 0xC0FFEE);
            assert_eq!(alias_ranges(&[(fill + 3 * PHYS_PAGE + 4, 4)]).len(), 2, "the pool slot and the GPU view");
            // unmapped where it was filled: still the pool's and the view's
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, std::ptr::null_mut()), 1);
            assert_eq!(std::ptr::read_volatile((view + 4) as *const u32), 0xC0FFEE);
            let mut buf = [0u8; 8];
            assert!(read_physical_pages(&pages[1..], 0, &mut buf));
            assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 0xC0FFEE);
            assert!(write_physical_pages(&pages[1..], 4, &7u32.to_le_bytes()));
            assert_eq!(std::ptr::read_volatile((pool + PHYS_PAGE + 4) as *const u32), 7);
            validate_physical().unwrap();
            // the title commits beside the pages, then gives the reservations back
            assert_eq!(commit_in_reservation(fill, 0x2000, mem::PAGE_READWRITE), Some(fill as *mut c_void));
            assert!(super::super::in_graphics_reservation(fill), "memory committed beside the pages is write-watched");
            std::ptr::write_volatile((fill + 0x1000) as *mut u32, 1);
            assert_eq!(written_pages(fill, PHYS_PAGE), vec![fill + 0x1000]);
            // (released memory is the next allocation's, of any test: looked at while nobody has it)
            assert_eq!(free_in_reservation(fill, 0, mem::MEM_RELEASE), Some(1));
            super::super::in_released_memory(fill, 4 * PHYS_PAGE, || assert!(!super::super::in_graphics_reservation(fill) && !in_reservation(fill)));
            assert_eq!(free_in_reservation(pool, 0, mem::MEM_RELEASE), Some(1));
            super::super::in_released_memory(pool, 4 * PHYS_PAGE, || assert!(!in_reservation(pool)));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            assert!(!pages_allocated(&pages));
            validate_physical().unwrap();
        }
    }

    /// Pages mapped in unified memory are copies: filled from the page on first use, and the page
    /// keeps what was written there when it goes.
    #[test]
    fn pages_in_unified_memory_are_copies() {
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let p = MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr());
            assert!(!p.is_null() && u.contains(p as usize) && (p as usize - u.base()) % PHYS_PAGE == 0);
            *(p as *mut u32).add(PHYS_PAGE / 4) = 0x55AA; // page 1
            assert_eq!(MapTitlePhysicalPages(p, 3, 0, 4, pages.as_mut_ptr()), p, "the same pages again: the same memory");
            let mut scattered = [pages[2], pages[0]];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, scattered.as_mut_ptr()).is_null(), "scattered pages need an address");
            // unmapped from unified memory: the page keeps what was written there
            let mut none = [0usize; 3];
            assert_eq!(MapTitlePhysicalPages(p, 3, 0, 4, none.as_mut_ptr()), p);
            let mut buf = [0u8; 4];
            assert!(read_physical_pages(&pages[1..], 0, &mut buf));
            assert_eq!(u32::from_le_bytes(buf), 0x55AA);
            let r = super::super::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let mut one = [pages[1]];
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(std::ptr::read_volatile(r as *const u32), 0x55AA, "a copy in plain memory");
            let mut outside = [usize::MAX & !15];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, outside.as_mut_ptr()).is_null());
            let mut wrapping = [usize::MAX];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, wrapping.as_mut_ptr()).is_null());
            validate_physical().unwrap();
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr()).is_null());
            assert_eq!(super::super::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert!(PHYS.lock().unwrap().places(pages[1]).is_empty(), "released: the page is mapped nowhere");
        }
    }

    /// The title decommits, then releases, memory that shows physical pages as copies: the pages
    /// are unmapped and keep what was written there, and the memory at those addresses afterwards
    /// is not theirs – XCL neither writes a page's data there nor takes it from there.
    #[test]
    fn memory_the_title_frees_shows_no_page_any_more() {
        unsafe {
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let r = super::super::VirtualAlloc(std::ptr::null(), 4 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert!(r != 0);
            // page 0 holds data and is written through its copy; page 1 is mapped and never used
            assert!(write_physical_pages(&pages[..1], 0, &[1u8; 8]));
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 2, 0, 4, pages.as_mut_ptr()) as usize, r);
            assert!(!lazy::is_lazy(r) && lazy::is_lazy(r + PHYS_PAGE));
            std::ptr::write_volatile((r + 16) as *mut u32, 0xC0FFEE);
            // decommitted with the pages still mapped, committed again, used for something else
            assert_eq!(super::super::VirtualFree(r as *mut c_void, 2 * PHYS_PAGE, mem::MEM_DECOMMIT), 1);
            assert!(!physical_page_at(r) && !physical_page_at(r + PHYS_PAGE) && !lazy::is_lazy(r + PHYS_PAGE));
            assert_eq!(super::super::VirtualAlloc(r as *const c_void, 2 * PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, r);
            std::ptr::write_bytes(r as *mut u8, 0x5A, 2 * PHYS_PAGE);
            // the page kept what was written through its copy, and a write to the pages by their
            // numbers goes to the pages
            let mut buf = [0u8; 4];
            assert!(read_physical_pages(&pages[..1], 16, &mut buf));
            assert_eq!(u32::from_le_bytes(buf), 0xC0FFEE);
            assert!(write_physical_pages(&pages, 0x100, &[0xB7; PHYS_PAGE]));
            assert!(std::slice::from_raw_parts(r as *const u8, 2 * PHYS_PAGE).iter().all(|&b| b == 0x5A), "the memory there now is not the pages'");
            // the pages' first views show the pages, not that memory
            let pool = reserve(2 * PHYS_PAGE).unwrap();
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 2, 0, 4, pages.as_mut_ptr()) as usize, pool);
            assert_eq!(std::ptr::read_volatile((pool + 16) as *const u32), 0xC0FFEE);
            assert_eq!(std::ptr::read_volatile((pool + 0x100) as *const u8), 0xB7);
            assert_eq!(std::ptr::read_volatile((pool + PHYS_PAGE + 0xFF) as *const u8), 0xB7);
            assert_eq!(std::ptr::read_volatile((pool + PHYS_PAGE + 0x100) as *const u8), 0, "page 1 past the write");
            // released with a page mapped in it: the page is left at its view only
            let mut one = [pages[0]];
            assert_eq!(MapTitlePhysicalPages((r + 2 * PHYS_PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r + 2 * PHYS_PAGE);
            assert_eq!(std::ptr::read_volatile((r + 2 * PHYS_PAGE + 16) as *const u32), 0xC0FFEE);
            assert_eq!(super::super::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!(PHYS.lock().unwrap().places(pages[0]), [pool]);
            validate_physical().unwrap();
            assert_eq!(free_in_reservation(pool, 0, mem::MEM_RELEASE), Some(1));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
        }
    }

    /// An unmap Windows refuses leaves the view up: the title's call fails and the model still
    /// says the address shows the page.
    #[test]
    fn an_unmap_windows_refuses_fails_and_the_page_stays_mapped() {
        unsafe {
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let pool = reserve(2 * PHYS_PAGE).unwrap();
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, pool);
            // (the view taken away behind the model, so that Windows refuses the unmap)
            assert!(unmap_into_placeholder(pool));
            assert!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, std::ptr::null_mut()).is_null(), "the unmap fails");
            assert_eq!(PHYS.lock().unwrap().places(pages[0]), [pool], "still counted among the page's addresses");
            assert!(alias_facts(pool).unwrap().history_here[0].contains("UNMAP REFUSED BY WINDOWS"));
            // (put back: now it goes)
            let (section, offset) = PHYS.lock().unwrap().block_of(pages[0]).map(|(b, off)| (b.section.raw, off)).unwrap();
            assert!(map_into(section, offset, pool));
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, pool);
            assert!(PHYS.lock().unwrap().places(pages[0]).is_empty());
            assert_eq!(free_in_reservation(pool, 0, mem::MEM_RELEASE), Some(1));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            validate_physical().unwrap();
        }
    }

    /// Pages freed while still mapped in the unified memory they were given are unmapped from it:
    /// that memory goes back to be handed out again.
    #[test]
    fn pages_freed_are_unmapped_from_their_unified_memory() {
        unsafe {
            if crate::gpu::uma::get().is_none() {
                return;
            }
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let p = MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, pages.as_mut_ptr()) as usize;
            assert!(p != 0);
            std::ptr::write_volatile(p as *mut u32, 5);
            assert_eq!(PHYS.lock().unwrap().places(pages[0]), [p]);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            let g = PHYS.lock().unwrap();
            assert!(g.places(pages[0]).is_empty() && g.places(pages[1]).is_empty());
        }
    }

    /// A crash record's account of an address: the page it shows, every other address showing it,
    /// what was written through those, whether a payload is still there, and the page's maps.
    #[test]
    fn a_crash_record_names_the_other_addresses_of_a_page_and_what_was_written_through_them() {
        unsafe {
            // (the addresses used here were other memory before, another test's a moment ago, with
            // a history of its own, which a crash record keeps: what is counted is what this
            // thread did since)
            std::thread::sleep(std::time::Duration::from_millis(3));
            let t0 = crate::log::uptime();
            let mine = format!("thread {})", GetCurrentThreadId());
            let since = |h: &[String]| h.iter().filter(|e| e.ends_with(&mine) && e.split(' ').next().and_then(|t| t.parse::<f64>().ok()).is_some_and(|t| t > t0 - 0.001)).count();
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let pool = reserve(4 * PHYS_PAGE).unwrap();
            let fill = reserve(4 * PHYS_PAGE).unwrap();
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, pool);
            let at = [(fill + PHYS_PAGE) as *mut c_void];
            let mut one = [pages[0] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, one.as_mut_ptr()), 1);
            assert_eq!(backing(fill + PHYS_PAGE + 0x123), Some(crate::alias::Backing { page: pages[0], kind: "view" }));
            assert_eq!(backing(fill).map(|b| b.page), Some(0), "a slot of the reservation that shows no page");
            assert_eq!(backing(&n as *const usize as usize), None, "plain memory");
            // a record written where the page was mapped to be filled
            let record = [0x5Au8; 0xC00];
            let dst = fill + PHYS_PAGE + 0xE000;
            std::ptr::copy_nonoverlapping(record.as_ptr(), dst as *mut u8, record.len());
            crate::alias::payload_written(dst, &record);
            let f = alias_facts(pool + 0xE010).unwrap();
            assert!(f.what.contains(&format!("physical page {:#x} is mapped here as a view", pages[0])) && f.what.contains("page 0 of an allocation of 2") && f.what.contains("its data is in its section"), "{}", f.what);
            assert_eq!(f.backing.map(|b| b.page), Some(pages[0]));
            let home = PHYS.lock().unwrap().home(pages[0]).unwrap();
            let mut aliases: Vec<usize> = f.aliases.iter().map(|a| a.0).collect();
            aliases.sort_unstable();
            let mut expected = vec![home, fill + PHYS_PAGE];
            expected.sort_unstable();
            assert_eq!(aliases, expected, "XCL's view of the section and the address it was filled at");
            assert!(since(&f.history_here) == 1 && f.history_here[0].contains("map view of") && f.history_here[0].contains("MapTitlePhysicalPages"), "{:?}", f.history_here);
            assert!(since(&f.history_page) == 1 && f.history_page[0].contains("XMemMapPhysicalPagesScatter") && f.history_page[0].contains(&format!("at {:#x}", fill + PHYS_PAGE)), "{:?}", f.history_page);
            // other bytes written into the same page by its number: no write to either address
            assert!(write_physical_pages(&pages[..1], 0xE000, &[0xB7; 0xC00]));
            let r = crate::alias::report(pool + 0xE010).unwrap();
            assert!(r.contains(&format!("payload written through {:#x}", dst + 0x10)) && r.contains("NO LONGER HOLDS it, and no later payload was written to that address"), "{r}");
            assert!(r.contains(&format!("payload written through {:#x}", home + 0xE010)) && r.contains("STILL HOLDS it") && r.contains("a physical destination"), "{r}");
            assert!(r.contains("the same memory at 2 other address(es)"), "{r}");
            // a second record, left alone
            let other = fill + PHYS_PAGE + 0x2000;
            std::ptr::copy_nonoverlapping(record.as_ptr(), other as *mut u8, record.len());
            crate::alias::payload_written(other, &record);
            // a third, found from the pool slot later by the page it was written to
            let third = fill + PHYS_PAGE + 0x5000;
            std::ptr::copy_nonoverlapping(record.as_ptr(), third as *mut u8, record.len());
            crate::alias::payload_written(third, &record);
            // the fill address shows another page now: its payload is told apart by the page
            one[0] = pages[1] as u64;
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, one.as_mut_ptr()), 1);
            let r = crate::alias::report(dst + 0x20).unwrap();
            assert!(r.contains(&format!("showing physical page {:#x} (view) then, physical page {:#x} (view) NOW", pages[0], pages[1])), "{r}");
            assert!(r.contains(&format!("physical page {:#x} itself no longer holds it either (at {:#x})", pages[0], home + 0xE000)), "the page it showed then was written by its number: {r}");
            let r = crate::alias::report(other + 0x20).unwrap();
            assert!(r.contains("NO LONGER HOLDS it") && r.contains(&format!("physical page {:#x} itself STILL HOLDS it (at {:#x})", pages[0], home + 0x2000)), "only the address moved: {r}");
            let r = crate::alias::report(pool + 0x5010).unwrap();
            assert!(r.contains(&format!("payload written to physical page {:#x} through {third:#x}, an address that shows the page no longer", pages[0])) && r.contains("the page STILL HOLDS it"), "written where the page is no longer mapped: {r}");
            let f = alias_facts(dst).unwrap();
            assert_eq!(since(&f.history_here), 3, "{:?}", f.history_here);
            assert!(f.history_here[0].contains(&format!("map view of page {:#x}", pages[1])) && f.history_here[1].contains(&format!("unmap view of page {:#x}", pages[0])), "{:?}", f.history_here);
            // unmapped and freed: the slot and the history remain
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, std::ptr::null_mut()), 1);
            let f = alias_facts(dst).unwrap();
            assert!(f.what.starts_with("no physical page is mapped here") && f.what.contains("a placeholder now") && f.backing.map(|b| b.page) == Some(0), "{}", f.what);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            let f = alias_facts(pool).unwrap();
            assert!(f.what.contains("THIS PAGE WAS FREED") && f.history_page.iter().any(|e| e.contains("freed:")), "{} {:?}", f.what, f.history_page);
            assert_eq!(free_in_reservation(fill, 0, mem::MEM_RELEASE), Some(1));
            assert_eq!(free_in_reservation(pool, 0, mem::MEM_RELEASE), Some(1));
            assert!(alias_facts(pool).unwrap().history_here.iter().any(|e| e.contains("the title releases its reservation") && e.contains("VirtualFree")));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages[1..].as_mut_ptr()), 1);
            validate_physical().unwrap();
        }
    }
}
