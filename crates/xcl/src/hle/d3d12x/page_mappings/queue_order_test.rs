use super::*;
use crate::hle::d3d12x::physical_directory::{self as dir, recordings};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

#[test]
fn actual_copy_page_mapping_handler_and_worker_respect_a_future_queue_signal() {
    unsafe {
        let mut dev=None;
        D3D12CreateDevice(None,D3D_FEATURE_LEVEL_11_0,&mut dev).unwrap();
        let dev:ID3D12Device=dev.unwrap();
        assert!(dir::init(&dev));
        let make=||->ID3D12CommandQueue{dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC{Type:D3D12_COMMAND_LIST_TYPE_DIRECT,..Default::default()}).unwrap()};
        let (q1,q2)=(make(),make());
        for q in [&q1,&q2] {queue::register(q.as_raw(),&dev).unwrap();}
        let fence=||->ID3D12Fence {dev.CreateFence(0,D3D12_FENCE_FLAG_NONE).unwrap()};
        let (dep,hold,done1,done2)=(fence(),fence(),fence(),fence());
        let addr=(0x618u64<<32)+65536;
        let old=dir::current_topology().unwrap();
        let map=|q:&ID3D12CommandQueue,guest| {
            let range=Range{kind:2,start:0,count:1};
            let batch=Batch{destination:guest,count:1,ranges:&range};
            assert_eq!(copy_batch(q.as_raw(),1,&batch,0,0),0);
        };
        let mut rb=None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES{Type:D3D12_HEAP_TYPE_READBACK,..Default::default()},D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC{Dimension:D3D12_RESOURCE_DIMENSION_BUFFER,Width:8,Height:1,DepthOrArraySize:1,MipLevels:1,SampleDesc:DXGI_SAMPLE_DESC{Count:1,Quality:0},Layout:D3D12_TEXTURE_LAYOUT_ROW_MAJOR,..Default::default()},
            D3D12_RESOURCE_STATE_COPY_DEST,None,&mut rb).unwrap();
        let rb:ID3D12Resource=rb.unwrap();
        let record=|at| {
            let alloc:ID3D12CommandAllocator=dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list:ID3D12GraphicsCommandList=dev.CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,&alloc,None).unwrap();
            let pc=list.as_raw() as usize;
            let _ = recordings::packet_index(pc);
            let cell=recordings::test_locator_buffer(pc);
            list.CopyBufferRegion(&rb,at,&cell,0,4);list.Close().unwrap();
            (alloc,list)
        };
        let (_a,l1)=record(0);let (_b,l2)=record(4);
        q1.Wait(&dep,1).unwrap();map(&q1,addr);
        queue::reader(q1.as_raw(),&[l1.as_raw()],None);q1.Signal(&done1,1).unwrap();
        map(&q2,addr+65536);
        queue::reader(q2.as_raw(),&[l2.as_raw()],None);q2.Signal(&done2,1).unwrap();
        q2.Wait(&hold,1).unwrap();q2.Signal(&dep,1).unwrap();
        crate::gpu::fence::wait(&done2,1,&dev,None,5000).unwrap();
        let second=dir::current_topology().unwrap();
        assert_eq!(done1.GetCompletedValue(),0);
        assert!(matches!(second.lookup(addr),Ok(dir::PageLookup::Absent)));
        assert!(matches!(second.lookup(addr+65536),Ok(dir::PageLookup::Unmapped)));
        hold.Signal(1).unwrap();
        crate::gpu::fence::wait(&done1,1,&dev,None,5000).unwrap();
        let first=dir::current_topology().unwrap();
        assert!(matches!(first.lookup(addr),Ok(dir::PageLookup::Unmapped)));
        assert!(matches!(first.lookup(addr+65536),Ok(dir::PageLookup::Unmapped)));
        let mut p=std::ptr::null_mut();
        rb.Map(0,Some(&D3D12_RANGE{Begin:0,End:8}),Some(&mut p)).unwrap();
        let roots=(p as *const [u32;2]).read_unaligned();
        rb.Unmap(0,Some(&D3D12_RANGE{Begin:0,End:0}));
        assert_eq!(roots,[first.index(),second.index()]);
        assert_ne!(roots[0],roots[1]);assert!(!roots.contains(&old.index()));
        recordings::ended(l1.as_raw() as usize);recordings::ended(l2.as_raw() as usize);
        queue::released(q1.as_raw() as usize);queue::released(q2.as_raw() as usize);
    }
}
