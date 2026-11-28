use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};

use crate::hle::kernelx::{physical_import_probe_views, PhysicalPageRef, PhysicalSectionView, PHYSICAL_SEGMENT_BYTES, PHYS_PAGE_SIZE};

struct Segment {
    heap: ID3D12Heap,
    view: PhysicalSectionView,
    start: usize,
    /// the buffer over all of the heap, placed when a run view first asks for it (`resolve_run`)
    whole: OnceLock<Option<ID3D12Resource>>,
}

impl Drop for Segment {
    fn drop(&mut self) {
        crate::memdiag::IMPORTED_SEGMENTS.sub(self.view.len() as u64);
    }
}

pub(super) struct ImportedPage {
    resource: ID3D12Resource,
    _segment: Arc<Segment>,
    source: PhysicalPageRef,
    _bucket: Arc<Mutex<Bucket>>,
}

impl ImportedPage {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    // ImportedPage::len has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(super) fn len(&self) -> usize {
        PHYS_PAGE_SIZE
    }
    pub(super) fn id(&self) -> usize {
        self.source.id()
    }
}

/// A committed run of a section, imported as one heap (8 MiB at most), and the buffer over all
/// of it: what one raw view shows every page of the run through.
pub(super) struct ImportedRun(Arc<Segment>);

impl ImportedRun {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        self.0.whole.get().and_then(Option::as_ref).expect("placed run")
    }

    pub(super) fn len(&self) -> usize {
        self.0.view.len()
    }
    pub(super) fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

#[derive(Default)]
struct Slot {
    page: Weak<ImportedPage>,
    segment: Weak<Segment>,
}

type Bucket = [Slot; PHYSICAL_SEGMENT_BYTES / PHYS_PAGE_SIZE];
#[derive(Default)]
struct Groups {
    by_section: HashMap<(usize, usize), Weak<Mutex<Bucket>>>,
    since_sweep: usize,
}

struct Backend {
    dev: ID3D12Device,
    dev13: ID3D12Device13,
    groups: Mutex<Groups>,
    open: unsafe extern "system" fn(*mut c_void, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT,
    place: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u64,
        *const D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_STATES,
        *const D3D12_CLEAR_VALUE,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT,
}

static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();

impl Backend {
    unsafe fn import(&self, view: PhysicalSectionView, start: usize) -> Option<Segment> {
        let mut h = std::ptr::null_mut();
        let hr = (self.open)(self.dev13.as_raw(), view.base() as *const _, view.len(), &ID3D12Heap::IID, &mut h);
        if hr.is_err() || h.is_null() {
            return None;
        }
        let heap = ID3D12Heap::from_raw(h);
        if heap.GetDesc().SizeInBytes != view.len() as u64 {
            return None;
        }
        crate::memdiag::IMPORTED_SEGMENTS.add(view.len() as u64);
        Some(Segment { heap, view, start, whole: OnceLock::new() })
    }

    unsafe fn place_page(&self, segment: &Segment, offset: usize) -> Option<ID3D12Resource> {
        let relative = offset.checked_sub(segment.start)?;
        if !relative.is_multiple_of(PHYS_PAGE_SIZE) || relative.checked_add(PHYS_PAGE_SIZE)? > segment.view.len() {
            return None;
        }
        self.place(segment, relative, PHYS_PAGE_SIZE)
    }

    unsafe fn place(&self, segment: &Segment, relative: usize, bytes: usize) -> Option<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r = std::ptr::null_mut();
        let hr = (self.place)(
            self.dev.as_raw(),
            segment.heap.as_raw(),
            relative as u64,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            std::ptr::null(),
            &ID3D12Resource::IID,
            &mut r,
        );
        if hr.is_err() || r.is_null() {
            return None;
        }
        Some(ID3D12Resource::from_raw(r))
    }
}

pub(super) fn supported(dev: &ID3D12Device) -> bool {
    BACKEND
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|b| b.dev.as_raw() == dev.as_raw())
}

pub(super) fn resolve_page(dev: &ID3D12Device, page: &PhysicalPageRef) -> Option<Arc<ImportedPage>> {
    let backend = BACKEND.get()?.as_ref().filter(|b| b.dev.as_raw() == dev.as_raw())?;
    resolve_on(backend, page)
}

/// The run `page` is in and the page's byte offset in it. The run's buffer is placed over the
/// heap the page's own buffer is in, when first asked for. None: it cannot be placed.
pub(super) fn resolve_run(dev: &ID3D12Device, page: &ImportedPage) -> Option<(ImportedRun, usize)> {
    let backend = BACKEND.get()?.as_ref().filter(|b| b.dev.as_raw() == dev.as_raw())?;
    run_on(backend, page)
}

fn run_on(backend: &Backend, page: &ImportedPage) -> Option<(ImportedRun, usize)> {
    let segment = &page._segment;
    let offset = page.source.section_offset().checked_sub(segment.start)?;
    segment.whole.get_or_init(|| unsafe { backend.place(segment, 0, segment.view.len()) }).as_ref()?;
    Some((ImportedRun(Arc::clone(segment)), offset))
}

fn resolve_on(backend: &Backend, page: &PhysicalPageRef) -> Option<Arc<ImportedPage>> {
    let offset = page.section_offset();
    let base = offset & !(PHYSICAL_SEGMENT_BYTES - 1);
    let index = (offset - base) / PHYS_PAGE_SIZE;
    let group = {
        let mut groups = backend.groups.lock().unwrap();
        let key = (page.section_identity(), base);
        if let Some(group) = groups.by_section.get(&key).and_then(Weak::upgrade) {
            group
        } else {
            if groups.since_sweep >= 128 {
                groups.by_section.retain(|_, w| w.strong_count() != 0);
                groups.since_sweep = 0;
            }
            let group = Arc::new(Mutex::new(std::array::from_fn(|_| Slot::default())));
            groups.by_section.insert(key, Arc::downgrade(&group));
            groups.since_sweep += 1;
            group
        }
    };
    let mut slots = group.lock().unwrap();
    if let Some(old) = slots[index].page.upgrade() {
        return (old.id() == page.id()).then_some(old);
    }
    let segment = if let Some(segment) = slots[index].segment.upgrade() {
        segment
    } else {
        let lo = slots[..index]
            .iter()
            .rposition(|s| s.segment.strong_count() != 0)
            .map_or(0, |i| i + 1);
        let hi = slots[index + 1..]
            .iter()
            .position(|s| s.segment.strong_count() != 0)
            .map_or(slots.len(), |i| index + i + 1);
        let (start, view) = page.pin_committed_segment(base + lo * PHYS_PAGE_SIZE..base + hi * PHYS_PAGE_SIZE)?;
        let wide = view.len() > PHYS_PAGE_SIZE;
        let segment = Arc::new(unsafe { backend.import(view, start) }.or_else(|| {
            if !wide {
                return None;
            }
            let (start, view) = page.pin_committed_segment(offset..offset + PHYS_PAGE_SIZE)?;
            unsafe { backend.import(view, start) }
        })?);
        let first = (segment.start - base) / PHYS_PAGE_SIZE;
        let stop = first + segment.view.len() / PHYS_PAGE_SIZE;
        for slot in &mut slots[first..stop] {
            slot.segment = Arc::downgrade(&segment);
        }
        segment
    };
    let resource = unsafe { backend.place_page(&segment, offset) }?;
    let imported = Arc::new(ImportedPage {
        resource,
        _segment: segment,
        source: page.clone(),
        _bucket: Arc::clone(&group),
    });
    slots[index].page = Arc::downgrade(&imported);
    Some(imported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hle::kernelx;

    #[repr(C)]
    struct RecordedObject {
        table: *const usize,
        bytes: usize,
    }

    unsafe extern "system" fn release(p: *mut c_void) -> u32 {
        drop(Box::from_raw(p.cast::<RecordedObject>()));
        0
    }
    unsafe extern "system" fn get_desc(p: *mut c_void, d: *mut D3D12_HEAP_DESC) {
        d.write(D3D12_HEAP_DESC {
            SizeInBytes: (*p.cast::<RecordedObject>()).bytes as u64,
            ..Default::default()
        });
    }
    fn object(bytes: usize) -> *mut c_void {
        unsafe extern "system" fn unexpected() {
            std::process::abort()
        }
        static VT: OnceLock<[usize; 256]> = OnceLock::new();
        let vt = VT.get_or_init(|| {
            let mut vt = [unexpected as *const () as usize; 256];
            vt[2] = release as *const () as usize;
            vt[8] = get_desc as *const () as usize;
            vt
        });
        Box::into_raw(Box::new(RecordedObject { table: vt.as_ptr(), bytes })).cast()
    }
    static OPENS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    static PLACEMENTS: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn open(_: *mut c_void, _: *const c_void, len: usize, _: *const GUID, out: *mut *mut c_void) -> HRESULT {
        OPENS.lock().unwrap().push(len);
        out.write(object(len));
        HRESULT(0)
    }
    unsafe extern "system" fn place(
        _: *mut c_void,
        heap: *mut c_void,
        off: u64,
        desc: *const D3D12_RESOURCE_DESC,
        _: D3D12_RESOURCE_STATES,
        _: *const D3D12_CLEAR_VALUE,
        _: *const GUID,
        out: *mut *mut c_void,
    ) -> HRESULT {
        PLACEMENTS.lock().unwrap().push((heap as usize, off, (*desc).Width));
        out.write(object((*desc).Width as usize));
        HRESULT(0)
    }

    #[test]
    fn separate_lookups_share_only_the_committed_heap_not_the_page_resources() {
        unsafe {
            let b = Backend {
                dev: ID3D12Device::from_raw(object(0)),
                dev13: ID3D12Device13::from_raw(object(0)),
                open,
                place,
                groups: Mutex::default(),
            };
            let mut n = 6;
            let mut ids = [0; 6];
            assert_ne!(
                kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, ids.as_mut_ptr()),
                0
            );
            let at = kernelx::reserve(4 * PHYS_PAGE_SIZE).unwrap();
            let mut selected = [ids[1], ids[2], ids[4], ids[1]];
            assert_eq!(kernelx::MapTitlePhysicalPages(at as _, 4, 0, 4, selected.as_mut_ptr()) as usize, at);
            let pages: Vec<_> = kernelx::snapshot_queue_sources(&(0..4).map(|i| at + i * PHYS_PAGE_SIZE).collect::<Vec<_>>())
                .unwrap()
                .into_iter()
                .map(|p| match p {
                    kernelx::QueuePageSource::Physical(p) => p,
                    _ => unreachable!(),
                })
                .collect();
            kernelx::free_in_reservation(at, 0, windows_sys::Win32::System::Memory::MEM_RELEASE).unwrap();
            assert_ne!(kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 6, ids.as_mut_ptr()), 0);
            let a = resolve_on(&b, &pages[0]).unwrap();
            let b2 = resolve_on(&b, &pages[1]).unwrap();
            let c = resolve_on(&b, &pages[2]).unwrap();
            let alias = resolve_on(&b, &pages[3]).unwrap();
            assert!(Arc::ptr_eq(&a, &alias));
            assert!(Arc::ptr_eq(&a._segment, &b2._segment));
            assert!(!Arc::ptr_eq(&a._segment, &c._segment));
            assert_ne!(a.resource().as_raw(), b2.resource().as_raw());
            assert_eq!(*OPENS.lock().unwrap(), [2 * PHYS_PAGE_SIZE, PHYS_PAGE_SIZE]);
            let placements = PLACEMENTS.lock().unwrap();
            let h = a._segment.heap.as_raw() as usize;
            let size = PHYS_PAGE_SIZE as u64;
            assert_eq!(
                placements.as_slice(),
                [(h, 0, size), (h, size, size), (c._segment.heap.as_raw() as usize, 0, size)]
            );
            drop(placements);
            let (run, at) = run_on(&b, &a).unwrap();
            let (same, next) = run_on(&b, &b2).unwrap();
            assert_eq!((run.id(), run.len(), at, next), (same.id(), 2 * PHYS_PAGE_SIZE, 0, PHYS_PAGE_SIZE));
            assert_ne!(run_on(&b, &c).unwrap().0.id(), run.id());
            assert_eq!(run_on(&b, &alias).unwrap().1, 0);
            assert_eq!(
                PLACEMENTS.lock().unwrap()[3..],
                [(h, 0, 2 * size), (c._segment.heap.as_raw() as usize, 0, size)],
                "one buffer over each run, placed once"
            );
        }
    }
}

pub(super) fn init(dev: &ID3D12Device) -> bool {
    BACKEND.get_or_init(|| unsafe {
        let dev13: ID3D12Device13 = dev.cast().ok()?;
        let mut support = D3D12_FEATURE_DATA_EXISTING_HEAPS::default();
        dev.CheckFeatureSupport(
            D3D12_FEATURE_EXISTING_HEAPS,
            (&mut support as *mut D3D12_FEATURE_DATA_EXISTING_HEAPS).cast(),
            std::mem::size_of_val(&support) as u32,
        )
        .ok()?;
        if !support.Supported.as_bool() {
            return None;
        }
        let b = Backend {
            open: dev13.vtable().OpenExistingHeapFromAddress1,
            place: dev.vtable().CreatePlacedResource,
            dev13,
            dev: dev.clone(),
            groups: Mutex::default(),
        };
        probe(&b)?;
        Some(b)
    });
    supported(dev)
}

unsafe fn probe(b: &Backend) -> Option<()> {
    let dev = &b.dev;
    let (cpu, target) = physical_import_probe_views()?;
    let segment = b.import(target, 0)?;
    let resource = b.place_page(&segment, 0)?;
    let mut rb = None;
    let desc = D3D12_RESOURCE_DESC {
        Width: 4,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..resource.GetDesc()
    };
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut rb,
    )
    .ok()?;
    let rb: ID3D12Resource = rb?;
    let descriptor = |flags| {
        dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: flags,
            NodeMask: 0,
        })
        .ok()
    };
    let staging = descriptor(D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
    let visible = descriptor(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?;
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 4,
                NumElements: 4,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    let handle = staging.GetCPUDescriptorHandleForHeapStart();
    dev.CreateUnorderedAccessView(&resource, None, Some(&uav), handle);
    dev.CopyDescriptorsSimple(
        1,
        visible.GetCPUDescriptorHandleForHeapStart(),
        handle,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    );
    let queue: ID3D12CommandQueue = dev
        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })
        .ok()?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    const CPU_WORD: u32 = 0x1c30_954f;
    const GPU_WORD: u32 = 0x749a_e305;
    let words = cpu.base() as *mut u32;
    words.write_volatile(CPU_WORD);
    list.SetDescriptorHeaps(&[Some(visible.clone())]);
    list.ClearUnorderedAccessViewUint(visible.GetGPUDescriptorHandleForHeapStart(), handle, &resource, &[GPU_WORD; 4], &[]);
    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(&resource),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
        ..Default::default()
    }]);
    list.CopyBufferRegion(&rb, 0, &resource, 0, 4);
    list.Close().ok()?;
    queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().ok()?)]);
    if queue.Signal(&fence, 1).is_err() || crate::gpu::fence::wait(&fence, 1, dev, None, 2000).is_err() {
        if dev.GetDeviceRemovedReason().is_ok() {
            std::mem::forget((queue, alloc, list, fence, staging, visible, rb, resource, segment));
        }
        return None;
    }
    let mut mapped = std::ptr::null_mut();
    rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 4 }), Some(&mut mapped)).ok()?;
    let read = !mapped.is_null() && mapped.cast::<u32>().read_volatile() == CPU_WORD;
    rb.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
    (read && words.add(4).read_volatile() == GPU_WORD && words.read_volatile() == CPU_WORD).then_some(())
}
