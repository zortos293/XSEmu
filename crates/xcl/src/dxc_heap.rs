//! A heap of its own for a DXC compile worker (`gpu::recomp::dxc`).
//!
//! DXC allocates through the COM allocator its compiler is created with. By default that is
//! the task allocator, on the process heap, where what DXC frees after a compile stays with
//! the heap among everything else's blocks. Here a worker's compiler is created
//! (`DxcCreateInstance2`) with an `IMalloc` over a heap made for that worker. When the worker
//! has been idle its compiler is released and the heap destroyed, which gives the system back
//! all that DXC took for its compiles – but only when DXC left nothing allocated in the heap
//! and holds no reference to the allocator (`WorkerHeap::end`): otherwise the heap stays.
//!
//! Not in the translator's sources: nothing here changes what DXC makes.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Memory::{HeapAlloc, HeapCompact, HeapCreate, HeapDestroy, HeapFree, HeapReAlloc, HeapSize, HeapValidate};

const S_OK: i32 = 0;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
const E_POINTER: i32 = 0x8000_4003u32 as i32;
const IID_IUNKNOWN: GUID = GUID { data1: 0, data2: 0, data3: 0, data4: [0xC0, 0, 0, 0, 0, 0, 0, 0x46] };
const IID_IMALLOC: GUID = GUID { data1: 2, data2: 0, data3: 0, data4: [0xC0, 0, 0, 0, 0, 0, 0, 0x46] };

/// heaps made, heaps destroyed, and the bytes allocated in those that are alive
static MADE: AtomicU64 = AtomicU64::new(0);
static ENDED: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
/// whether a block is looked up in its heap before it is freed or resized (`check`), and the
/// blocks that were not the heap's: left alone, since another allocator made them
static CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FOREIGN: AtomicU64 = AtomicU64::new(0);

/// Look every block up in its heap before it is freed or resized (slower: for finding out
/// whether DXC hands an allocator a block of another's).
pub fn check(on: bool) {
    CHECK.store(on, Relaxed);
}

/// Blocks handed to a heap's allocator that were not that heap's (counted only with `check`).
pub fn foreign() -> u64 {
    FOREIGN.load(Relaxed)
}

/// (heaps made, heaps destroyed, bytes DXC has allocated in the heaps alive)
pub fn counts() -> (u64, u64, usize) {
    (MADE.load(Relaxed), ENDED.load(Relaxed), BYTES.load(Relaxed))
}

#[repr(C)]
struct Vtbl {
    query_interface: unsafe extern "system" fn(*mut Malloc, *const GUID, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(*mut Malloc) -> u32,
    release: unsafe extern "system" fn(*mut Malloc) -> u32,
    alloc: unsafe extern "system" fn(*mut Malloc, usize) -> *mut c_void,
    realloc: unsafe extern "system" fn(*mut Malloc, *mut c_void, usize) -> *mut c_void,
    free: unsafe extern "system" fn(*mut Malloc, *mut c_void),
    get_size: unsafe extern "system" fn(*mut Malloc, *mut c_void) -> usize,
    did_alloc: unsafe extern "system" fn(*mut Malloc, *mut c_void) -> i32,
    heap_minimize: unsafe extern "system" fn(*mut Malloc),
}

static VTBL: Vtbl = Vtbl { query_interface, add_ref, release, alloc, realloc, free, get_size, did_alloc, heap_minimize };

/// An `IMalloc` over one heap: the COM object DXC gets (its first field is the vtable).
#[repr(C)]
struct Malloc {
    vtbl: *const Vtbl,
    refs: AtomicU32,
    heap: usize,
    /// blocks and bytes allocated now, and the most bytes there were
    live: AtomicUsize,
    bytes: AtomicUsize,
    peak: AtomicUsize,
}

impl Malloc {
    fn heap(&self) -> HANDLE {
        self.heap as HANDLE
    }

    /// The size the heap has for block `p` (0: it does not know it).
    unsafe fn size(&self, p: *mut c_void) -> usize {
        match HeapSize(self.heap(), 0, p) {
            usize::MAX => 0,
            n => n,
        }
    }

    fn grew(&self, by: usize) {
        let now = self.bytes.fetch_add(by, Relaxed) + by;
        self.peak.fetch_max(now, Relaxed);
        BYTES.fetch_add(by, Relaxed);
    }

    fn shrank(&self, by: usize) {
        self.bytes.fetch_sub(by, Relaxed);
        BYTES.fetch_sub(by, Relaxed);
    }

    /// With `check`: whether block `p` is not this heap's (it is counted, and left alone).
    unsafe fn foreign(&self, p: *mut c_void) -> bool {
        let foreign = CHECK.load(Relaxed) && HeapValidate(self.heap(), 0, p) == 0;
        if foreign {
            FOREIGN.fetch_add(1, Relaxed);
        }
        foreign
    }
}

fn same(a: &GUID, b: &GUID) -> bool {
    (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4)
}

unsafe extern "system" fn query_interface(this: *mut Malloc, iid: *const GUID, out: *mut *mut c_void) -> i32 {
    if out.is_null() || iid.is_null() {
        return E_POINTER;
    }
    if same(&*iid, &IID_IUNKNOWN) || same(&*iid, &IID_IMALLOC) {
        add_ref(this);
        *out = this.cast();
        return S_OK;
    }
    *out = std::ptr::null_mut();
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(this: *mut Malloc) -> u32 {
    (*this).refs.fetch_add(1, Relaxed) + 1
}

/// The last reference: the heap goes with whatever is still in it (nothing can free it now).
unsafe extern "system" fn release(this: *mut Malloc) -> u32 {
    let left = (*this).refs.fetch_sub(1, Relaxed) - 1;
    if left == 0 {
        let m = Box::from_raw(this);
        BYTES.fetch_sub(m.bytes.load(Relaxed), Relaxed);
        HeapDestroy(m.heap());
        ENDED.fetch_add(1, Relaxed);
    }
    left
}

unsafe extern "system" fn alloc(this: *mut Malloc, bytes: usize) -> *mut c_void {
    let m = &*this;
    let p = HeapAlloc(m.heap(), 0, bytes);
    if !p.is_null() {
        m.live.fetch_add(1, Relaxed);
        m.grew(m.size(p));
    }
    p
}

unsafe extern "system" fn realloc(this: *mut Malloc, p: *mut c_void, bytes: usize) -> *mut c_void {
    if p.is_null() {
        return alloc(this, bytes);
    }
    if bytes == 0 {
        free(this, p);
        return std::ptr::null_mut();
    }
    let m = &*this;
    if m.foreign(p) {
        return std::ptr::null_mut();
    }
    let old = m.size(p);
    let q = HeapReAlloc(m.heap(), 0, p, bytes);
    if !q.is_null() {
        m.shrank(old);
        m.grew(m.size(q));
    }
    q
}

unsafe extern "system" fn free(this: *mut Malloc, p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let m = &*this;
    if m.foreign(p) {
        return;
    }
    let size = m.size(p);
    if HeapFree(m.heap(), 0, p) != 0 {
        m.live.fetch_sub(1, Relaxed);
        m.shrank(size);
    }
}

unsafe extern "system" fn get_size(this: *mut Malloc, p: *mut c_void) -> usize {
    if p.is_null() {
        return usize::MAX;
    }
    HeapSize((*this).heap(), 0, p)
}

/// -1: not determined (COM allows it; DXC does not ask).
unsafe extern "system" fn did_alloc(_this: *mut Malloc, _p: *mut c_void) -> i32 {
    -1
}

unsafe extern "system" fn heap_minimize(this: *mut Malloc) {
    HeapCompact((*this).heap(), 0);
}

/// A worker's heap and its allocator. Dropped: the worker's reference goes, and the heap with
/// the last one.
pub struct WorkerHeap {
    m: *mut Malloc,
}

impl WorkerHeap {
    /// A new heap. None: the system made none. It has no limit of its own: DXC does not
    /// survive an allocation that fails (DXC 1.9 read a null pointer when one did), so what a
    /// compile may take is decided before it starts (`dxc::wait_for_memory`).
    pub fn new() -> Option<WorkerHeap> {
        let heap = unsafe { HeapCreate(0, 0, 0) };
        if heap as usize == 0 {
            return None;
        }
        MADE.fetch_add(1, Relaxed);
        let m = Malloc { vtbl: &VTBL, refs: AtomicU32::new(1), heap: heap as usize, live: AtomicUsize::new(0), bytes: AtomicUsize::new(0), peak: AtomicUsize::new(0) };
        Some(WorkerHeap { m: Box::into_raw(Box::new(m)) })
    }

    /// The `IMalloc` for `DxcCreateInstance2` (not counted: the callee takes its own reference).
    pub fn imalloc(&self) -> *mut c_void {
        self.m.cast()
    }

    /// (blocks, bytes) allocated in it now.
    pub fn live(&self) -> (usize, usize) {
        let m = unsafe { &*self.m };
        (m.live.load(Relaxed), m.bytes.load(Relaxed))
    }

    /// References to the allocator other than the worker's.
    pub fn held(&self) -> u32 {
        unsafe { &*self.m }.refs.load(Relaxed) - 1
    }

    /// The most bytes that were allocated in it at once.
    pub fn peak(&self) -> usize {
        unsafe { &*self.m }.peak.load(Relaxed)
    }

    /// The worker is done with it for now, its compiler released: the heap is destroyed if
    /// nothing is allocated in it and nobody else holds the allocator – its peak is returned.
    /// Otherwise it is given back, to be used on.
    pub fn end(self) -> Result<usize, WorkerHeap> {
        if self.held() != 0 || self.live().0 != 0 {
            return Err(self);
        }
        Ok(self.peak())
    }
}

impl Drop for WorkerHeap {
    fn drop(&mut self) {
        unsafe { release(self.m) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allocator counts what is allocated through it, and its heap ends only when that is
    /// nothing and nobody else holds it.
    #[test]
    fn a_worker_heap_ends_when_nothing_is_left_in_it() {
        let heap = WorkerHeap::new().expect("a heap");
        let m = heap.imalloc() as *mut Malloc;
        unsafe {
            let a = alloc(m, 100);
            let b = alloc(m, 0);
            assert!(!a.is_null() && !b.is_null());
            assert_eq!(heap.live().0, 2);
            assert!(heap.live().1 >= 100 && get_size(m, a) >= 100);
            let a = realloc(m, a, 1 << 20);
            assert!(!a.is_null() && heap.live().0 == 2 && heap.live().1 >= 1 << 20);
            let c = realloc(m, std::ptr::null_mut(), 64);
            assert_eq!(heap.live().0, 3);
            assert!(realloc(m, c, 0).is_null());
            free(m, a);
            free(m, std::ptr::null_mut());
            let heap = heap.end().err().expect("a block is left");
            free(m, b);
            assert_eq!(heap.live(), (0, 0));
            let mut out = std::ptr::null_mut();
            assert_eq!(query_interface(m, &IID_IMALLOC, &mut out), S_OK);
            assert_eq!((out, heap.held()), (m.cast(), 1));
            let heap = heap.end().err().expect("the allocator is held");
            assert_eq!(query_interface(m, &GUID { data1: 9, ..IID_IMALLOC }, &mut out), E_NOINTERFACE);
            assert_eq!(release(m), 1);
            let (ended, peak) = (counts().1, heap.peak());
            assert_eq!(heap.end().ok(), Some(peak));
            assert!(peak >= 1 << 20 && counts().1 > ended);
        }
    }
}
