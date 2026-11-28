use super::*;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

#[test]
fn immutable_versions_merge_only_after_ready_and_preserve_inflight_readers() {
    unsafe {
        let mut dev = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
        let dev: ID3D12Device = dev.unwrap();
        assert!(init(&dev));
        let hi = 0x617u16;
        let change = |page: u16, plane, target| Change {
            guest: (u64::from(hi) << 32) | (u64::from(page) << 16),
            plane,
            target,
        };
        let one = prepare_version(vec![
            change(1, Plane::Gpu, Mapping::Unmapped),
            change(3, Plane::Default, Mapping::Unmapped),
        ])
        .unwrap();
        let two = prepare_version(vec![
            change(2, Plane::Gpu, Mapping::Unmapped),
            change(2, Plane::Gpu, Mapping::Inherit),
            change(2, Plane::Gpu, Mapping::Unmapped),
        ])
        .unwrap();
        note_native_texture_access((u64::from(hi) << 32) + 65536 + 7, 2, false);
        note_native_texture_access((u64::from(hi) << 32) + 0x40000, 65536, true);
        assert_eq!(pending_native_texture_hits(), (1, 0));
        let old = current_topology().unwrap();
        assert!(!old.leaves.contains_key(&(hi, Plane::Gpu)));
        let q = || -> ID3D12CommandQueue {
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap()
        };
        let f = || -> ID3D12Fence { dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
        let (first, second) = (q(), q());
        let (future, ready1, ready2, grant1, grant2, done1, done2) =
            (f(), f(), f(), f(), f(), f(), f());
        first.Wait(&future, 1).unwrap();
        first.Signal(&ready1, 1).unwrap();
        first.Wait(&grant1, 1).unwrap();
        second.Signal(&ready2, 1).unwrap();
        second.Wait(&grant2, 1).unwrap();
        crate::gpu::fence::wait(&ready2, 1, &dev, None, 3000).unwrap();
        assert_eq!(ready1.GetCompletedValue(), 0);
        two.publish_at_ready();
        let v2 = current_topology().unwrap();
        let addr = u64::from(hi) << 32;
        assert!(matches!(v2.lookup(addr + 131073), Ok(PageLookup::Unmapped)));
        assert!(matches!(v2.lookup(addr + 65537), Ok(PageLookup::Absent)));
        assert!(v2.lookup(1 << 48).is_err());
        let rb = buffer(&dev, 32, D3D12_HEAP_TYPE_READBACK);
        let rb = rb.unwrap();
        let copy = |at: u64,
                    v: &Topology,
                    range: (u64, u64),
                    plane: Plane|
         -> (ID3D12CommandAllocator, ID3D12CommandList) {
            let a: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let l: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
                .unwrap();
            let leaf = &v.leaves[&(hi, plane)].view;
            l.CopyBufferRegion(&rb, at, &leaf.resource, range.0, range.1);
            l.Close().unwrap();
            (a, l.cast().unwrap())
        };
        let read2 = copy(0, &v2, (4, 8), Plane::Gpu);
        second.ExecuteCommandLists(&[Some(read2.1.clone())]);
        second.Signal(&done2, 1).unwrap();
        second.Signal(&future, 1).unwrap();
        grant2.Signal(1).unwrap();
        crate::gpu::fence::wait(&ready1, 1, &dev, None, 3000).unwrap();
        one.publish_at_ready();
        let v1 = current_topology().unwrap();
        note_native_texture_access((u64::from(hi) << 32) + 65536, 4, true);
        assert_eq!(pending_native_texture_hits(), (1, 0));
        assert!(matches!(v1.lookup(addr + 65537), Ok(PageLookup::Unmapped)));
        assert!(matches!(v1.lookup(addr + 196609), Ok(PageLookup::Unmapped)));
        assert!(matches!(v2.lookup(addr + 196609), Ok(PageLookup::Absent)));
        assert_ne!(v1.index(), v2.index());
        assert_ne!(
            v1.leaves[&(hi, Plane::Gpu)].view.resource.as_raw(),
            v2.leaves[&(hi, Plane::Gpu)].view.resource.as_raw()
        );
        let read1 = copy(8, &v1, (4, 8), Plane::Gpu);
        let historic = copy(16, &v2, (4, 8), Plane::Gpu);
        let alt = copy(24, &v1, (12, 4), Plane::Default);
        first.ExecuteCommandLists(&[
            Some(read1.1.clone()),
            Some(historic.1.clone()),
            Some(alt.1.clone()),
        ]);
        first.Signal(&done1, 1).unwrap();
        grant1.Signal(1).unwrap();
        crate::gpu::fence::wait(&done1, 1, &dev, None, 3000).unwrap();
        let mut p = std::ptr::null_mut();
        rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 28 }), Some(&mut p))
            .unwrap();
        let words = (p as *const [u32; 7]).read_unaligned();
        rb.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        let t = ownership::UNMAPPED;
        assert_eq!(words, [0, t, t, t, 0, t, t]);
    }
}
