use super::*;
use crate::hle::d3d12x::page_mappings::queue;
use crate::hle::d3d12x::physical_directory::{self, prepare_version, Change, Mapping, Plane};
use crate::hle::d3d12x::{console_heap, indirect};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Dxgi::Common::*;

unsafe fn buffer(dev: &ID3D12Device, ty: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, n: u64) -> ID3D12Resource {
    let d = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: n,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut r = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r).unwrap();
    r.unwrap()
}

/// A title copy run through the directory: unaligned on both sides and split on both sides'
/// page boundaries, the bytes landing in the destination's own memory, except on a page the
/// directory explicitly unmapped, whose writes are dropped (no stale mirror takes them).
#[test]
fn a_directory_copy_splits_both_sides_and_drops_writes_to_an_unmapped_page() {
    unsafe {
        let mut dev = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let dev: ID3D12Device = dev.unwrap();
        assert!(physical_directory::init(&dev));
        let base = 0x0618_0000_0000u64;
        prepare_version(vec![Change { guest: base + 5 * 65536, plane: Plane::Gpu, target: Mapping::Unmapped }]).unwrap().publish_at_ready();
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        queue::register(q.as_raw(), &dev).unwrap();
        let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();

        let size = 6 * 65536u64;
        let mut bytes: Vec<u8> = (0..size as usize).map(|i| if i < 4 * 65536 { (i * 7 + i / 251) as u8 } else { 0xEE }).collect();
        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, size);
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
        upload.Unmap(0, None);
        let data = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, size);
        l.CopyBufferRegion(&data, 0, &upload, 0, size);
        l.ResourceBarrier(&[indirect::transition(data.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        let (view, cpu, _) = console_heap::reserve_own(1).unwrap();
        dev.CreateUnorderedAccessView(
            &data,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Format: DXGI_FORMAT_R32_TYPELESS,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: size as u32 / 4, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
            }),
            cpu,
        );
        let locator = physical_directory::recordings::packet_index(l.as_raw() as usize);
        assert_ne!(locator, 0);
        let (from, to, n) = (base + 65536 + 65000 + 3, base + 4 * 65536 + 65536 - 700 + 1, 1501u64);
        let copy = Copy::prepare(Transfer::new(from, to, n).unwrap(), locator, |page, _| {
            let off = u32::try_from((u64::from(page) << 16).checked_sub(base)?).ok()?;
            Some(FallbackPage::Raw { view, page_base: off })
        })
        .unwrap();
        let pipeline = shared_pipeline(&dev).unwrap();
        let (plan_bytes, data_bytes) = copy.storage_bytes();
        let (plan, scratch) = (live::scratch(&dev, plan_bytes).unwrap(), live::scratch(&dev, data_bytes).unwrap());
        pipeline.record_copy(&l, &console_heap::global_heap().unwrap(), &copy, &plan, &scratch).unwrap();
        let readback = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, size);
        l.ResourceBarrier(&[indirect::transition(data.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        l.CopyBufferRegion(&readback, 0, &data, 0, size);
        l.Close().unwrap();
        queue::reader(q.as_raw(), &[l.as_raw()], None);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        q.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 10000).unwrap();
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let got = std::slice::from_raw_parts(p.cast::<u8>(), size as usize).to_vec();
        readback.Unmap(0, None);
        console_heap::release_global(view, 1);
        let (src, dst) = ((from - base) as usize, (to - base) as usize);
        let kept = (5 * 65536 - dst).min(n as usize);
        for i in 0..kept {
            bytes[dst + i] = bytes[src + i];
        }
        let first_diff = got.iter().zip(&bytes).position(|(g, e)| g != e);
        assert_eq!(first_diff, None, "byte {first_diff:?}: copied into page 4 only, page 5's writes dropped, everything else untouched");
        assert!(kept > 0 && kept < n as usize);
    }
}

/// One page-transfer run on its own device and directory: title pages at base..base + 6 pages
/// back a UAV buffer the test fills and reads back; page 5 is explicitly unmapped on the GPU.
struct Rig {
    dev: ID3D12Device,
    q: ID3D12CommandQueue,
    l: ID3D12GraphicsCommandList,
    data: ID3D12Resource,
    _keep: (ID3D12CommandAllocator, ID3D12Resource),
    view: u32,
    locator: u32,
    bytes: Vec<u8>,
    base: u64,
}

const RIG_SIZE: u64 = 6 * 65536;

impl Rig {
    unsafe fn new(base: u64) -> Option<Rig> {
        let mut dev = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return None;
        }
        let dev: ID3D12Device = dev.unwrap();
        assert!(physical_directory::init(&dev));
        prepare_version(vec![Change { guest: base + 5 * 65536, plane: Plane::Gpu, target: Mapping::Unmapped }]).unwrap().publish_at_ready();
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        queue::register(q.as_raw(), &dev).unwrap();
        let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
        let bytes: Vec<u8> = (0..RIG_SIZE as usize).map(|i| if i < 5 * 65536 { (i * 7 + i / 251) as u8 } else { 0xEE }).collect();
        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, RIG_SIZE);
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
        upload.Unmap(0, None);
        let data = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, RIG_SIZE);
        l.CopyBufferRegion(&data, 0, &upload, 0, RIG_SIZE);
        l.ResourceBarrier(&[indirect::transition(data.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        let (view, cpu, _) = console_heap::reserve_own(1).unwrap();
        dev.CreateUnorderedAccessView(
            &data,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Format: DXGI_FORMAT_R32_TYPELESS,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: RIG_SIZE as u32 / 4, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
            }),
            cpu,
        );
        let locator = physical_directory::recordings::packet_index(l.as_raw() as usize);
        assert_ne!(locator, 0);
        Some(Rig { dev, q, l, data, _keep: (a, upload), view, locator, bytes, base })
    }

    fn fallback(&self) -> impl FnMut(u32) -> Option<FallbackPage> + '_ {
        |page| Some(FallbackPage::Raw { view: self.view, page_base: u32::try_from((u64::from(page) << 16).checked_sub(self.base)?).ok()? })
    }

    /// Copies `from` (in COPY_SOURCE) to a readback buffer, runs the list and returns those bytes.
    unsafe fn run(self, from: &ID3D12Resource, n: u64) -> Vec<u8> {
        let readback = buffer(&self.dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, n);
        self.l.CopyBufferRegion(&readback, 0, from, 0, n);
        self.l.Close().unwrap();
        queue::reader(self.q.as_raw(), &[self.l.as_raw()], None);
        let fence: ID3D12Fence = self.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        self.q.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &self.dev, None, 10000).unwrap();
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let got = std::slice::from_raw_parts(p.cast::<u8>(), n as usize).to_vec();
        readback.Unmap(0, None);
        console_heap::release_global(self.view, 1);
        got
    }
}

/// ResolveQueryData into a title destination the directory maps: the queries resolve into scratch,
/// XCL's visible-sample results go over theirs, and the scatter writes the bytes to the
/// destination's own memory up to the explicitly unmapped page, whose share is dropped.
#[test]
fn a_query_resolve_reaches_the_destination_owner_and_drops_an_unmapped_page() {
    unsafe {
        let Some(rig) = Rig::new(0x0619_0000_0000) else { return };
        let to = rig.base + 5 * 65536 - 16;
        let copy = Copy::write(to, 32, rig.locator, rig.fallback()).unwrap();
        let pipeline = shared_pipeline(&rig.dev).unwrap();
        let (plan_bytes, data_bytes) = copy.storage_bytes();
        let (plan, scratch) = (live::scratch(&rig.dev, plan_bytes).unwrap(), live::scratch(&rig.dev, data_bytes).unwrap());
        let heap: ID3D12QueryHeap = {
            let mut h = None;
            rig.dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION, Count: 4, NodeMask: 0 }, &mut h).unwrap();
            h.unwrap()
        };
        for i in 0..4 {
            rig.l.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, i);
            rig.l.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, i);
        }
        rig.l.ResolveQueryData(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0, 4, &*scratch, 0);
        let visible = 0x1122_3344_5566_7788u64;
        let at = scratch.GetGPUVirtualAddress() + 8;
        let params = [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at, Value: visible as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + 4, Value: (visible >> 32) as u32 }];
        rig.l.cast::<ID3D12GraphicsCommandList2>().unwrap().WriteBufferImmediate(2, params.as_ptr(), Some([D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; 2].as_ptr()));
        rig.l.ResourceBarrier(&[indirect::transition(scratch.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        pipeline.record_write(&rig.l, &console_heap::global_heap().unwrap(), &copy, &plan, &scratch).unwrap();
        rig.l.ResourceBarrier(&[indirect::transition(rig.data.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let mut expected = rig.bytes.clone();
        let data = rig.data.clone();
        let got = rig.run(&data, RIG_SIZE);
        let at = (to - 0x0619_0000_0000) as usize;
        expected[at..at + 8].copy_from_slice(&0u64.to_le_bytes());
        expected[at + 8..at + 16].copy_from_slice(&visible.to_le_bytes());
        let first_diff = got.iter().zip(&expected).position(|(g, e)| g != e);
        assert_eq!(first_diff, None, "byte {first_diff:?}: queries 0 and 1 resolved into page 4, page 5's two dropped, everything else untouched");
    }
}

/// A predicate gathered through the directory: one in a page the directory has no entry for reads
/// the title's memory, one in the explicitly unmapped page reads zero (never a stale mirror).
#[test]
fn a_predicate_gather_reads_the_owner_and_zero_from_an_unmapped_page() {
    unsafe {
        let Some(rig) = Rig::new(0x061A_0000_0000) else { return };
        let pipeline = shared_pipeline(&rig.dev).unwrap();
        let (mapped, unmapped) = (rig.base + 2 * 65536 + 40, rig.base + 5 * 65536 + 8);
        let out = buffer(&rig.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, 16);
        let mut kept = Vec::new();
        for (i, va) in [mapped, unmapped].into_iter().enumerate() {
            let read = Copy::read(va, 8, rig.locator, rig.fallback()).unwrap();
            let (plan_bytes, data_bytes) = read.storage_bytes();
            let (plan, scratch) = (live::scratch(&rig.dev, plan_bytes).unwrap(), live::scratch(&rig.dev, data_bytes).unwrap());
            pipeline.record_read(&rig.l, &console_heap::global_heap().unwrap(), &read, &plan, &scratch).unwrap();
            let s = scratch.as_raw() as usize;
            rig.l.ResourceBarrier(&[indirect::transition(s, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)]);
            rig.l.ResourceBarrier(&[indirect::transition(s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            rig.l.CopyBufferRegion(&out, 8 * i as u64, &*scratch, 0, 8);
            kept.push((plan, scratch));
        }
        rig.l.ResourceBarrier(&[indirect::transition(out.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let at = (mapped - rig.base) as usize;
        let expected: Vec<u8> = rig.bytes[at..at + 8].iter().copied().chain([0; 8]).collect();
        let got = rig.run(&out, 16);
        drop(kept);
        assert_eq!(got, expected, "the mapped predicate's 8 bytes, then zero from the unmapped page");
    }
}

