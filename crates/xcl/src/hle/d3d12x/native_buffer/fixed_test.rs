use super::*;
use crate::hle::d3d12x::{console_heap, physical_directory};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

unsafe fn buffer(
    dev: &ID3D12Device,
    ty: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
    n: u64,
) -> ID3D12Resource {
    let d = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: n,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut r = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES {
            Type: ty,
            ..Default::default()
        },
        D3D12_HEAP_FLAG_NONE,
        &d,
        state,
        None,
        &mut r,
    )
    .unwrap();
    r.unwrap()
}

#[test]
fn a_fixed_packet_gathers_args_and_predicate_then_executes_only_the_guest_command_conditionally() {
    unsafe {
        let mut dev = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
        let dev: ID3D12Device = dev.unwrap();
        assert!(physical_directory::init(&dev));
        let q: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        queue::register(q.as_raw(), &dev).unwrap();
        let a: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let l: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
            .unwrap();
        let size = 2 * 65536;
        let data = uav(&dev, size).unwrap();
        let upload = buffer(
            &dev,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            size,
        );
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::write_bytes(p.cast::<u8>(), 0, size as usize);
        for i in 0..3 {
            p.cast::<u32>().add(65528 / 4 + i).write(1);
        }
        p.cast::<u64>().add(5).write(1);
        upload.Unmap(0, None);
        transition(&l, &data, D3D12_RESOURCE_STATE_COPY_DEST);
        l.CopyBufferRegion(&data, 0, &upload, 0, size);
        let result = uav(&dev, 64).unwrap();
        let readback = buffer(
            &dev,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            64,
        );
        let barrier = |list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, from, to| {
            let mut b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                ..Default::default()
            };
            b.Anonymous.Transition =
                std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: from,
                    StateAfter: to,
                });
            list.ResourceBarrier(std::slice::from_ref(&b));
            std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
        };
        barrier(
            &l,
            &data,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        transition(&l, &result, D3D12_RESOURCE_STATE_COPY_DEST);
        l.CopyBufferRegion(&result, 0, &upload, 128, 64);
        barrier(
            &l,
            &result,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        l.Close().unwrap();
        q.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
        let bytes=crate::gpu::recomp::dxc::compile("RWByteAddressBuffer outBuf:register(u0); [RootSignature(\"UAV(u0)\")][numthreads(1,1,1)]void main(){outBuf.Store(0,42);}","main","cs_6_0",&[]).unwrap();
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, &bytes).unwrap();
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
        std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let mut sig = None;
        dev.CreateCommandSignature(
            &D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 12,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            },
            None,
            &mut sig,
        )
        .unwrap();
        let sig: ID3D12CommandSignature = sig.unwrap();
        crate::hle::d3d12x::indirect::store_record_layout(sig.as_raw(), 0, 12, &[[2, 0, 0, 0]]);
        let base = 0x2222_0000_0000;
        let (view, cpu, _) = console_heap::reserve_own(1).unwrap();
        dev.CreateUnorderedAccessView(
            &data,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: size as u32 / 4,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            cpu,
        );
        let pipeline = Arc::new(Pipeline::new(&dev).unwrap());
        let mut packets = Vec::new();
        let mut storage = Vec::new();
        for i in 0..2u64 {
            let op = D3D12_PREDICATION_OP_EQUAL_ZERO;
            let pred_offset = 32 + i * 8;
            let need = Need {
                input: Input::Guest {
                    va: base + 65528,
                    bytes: 12,
                },
                state: D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            };
            let pred = Input::Guest {
                va: base + pred_offset,
                bytes: 8,
            };
            let recipe = Recipe {
                needs: vec![need],
                predicate: Some((pred, op)),
            };
            let consumer = FixedConsumer {
                inputs: recipe,
                ia: vec![],
                op: GuestOp::Indirect {
                    signature: sig.clone(),
                    max: 1,
                    count: false,
                },
            };
            let pkt = consumer
                .instantiate(
                    &dev,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    pipeline.clone(),
                    &console_heap::global_heap().unwrap(),
                    |_, page| {
                        let off = u32::try_from((u64::from(page) << 16).checked_sub(base)?).ok()?;
                        Some((
                            FallbackPage::Raw {
                                view,
                                page_base: off,
                            },
                            Arc::new(data.clone()),
                        ))
                    },
                    |l| {
                        l.SetComputeRootSignature(&root);
                        l.SetComputeRootUnorderedAccessView(
                            0,
                            result.GetGPUVirtualAddress() + 16 * i,
                        );
                        l.SetPipelineState(&pso);
                        Ok(Arc::new((
                            root.clone(),
                            pso.clone(),
                            sig.clone(),
                            result.clone(),
                        )))
                    },
                )
                .unwrap();
            storage.push(pkt.scratch_ids[0]);
            packets.push(pkt);
        }
        assert_ne!(storage[0], storage[1]);
        for p in packets {
            p.submit(&q).unwrap();
        }
        let tail: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
            .unwrap();
        for i in 0..2u64 {
            tail.SetPredication(&upload, 32 + i * 8, D3D12_PREDICATION_OP_EQUAL_ZERO);
            tail.SetComputeRootSignature(&root);
            tail.SetPipelineState(&pso);
            tail.SetComputeRootUnorderedAccessView(0, result.GetGPUVirtualAddress() + 32 + 16 * i);
            tail.ExecuteIndirect(&sig, 1, &upload, 65528, None, 0);
        }
        tail.SetPredication(None, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        transition(&tail, &result, D3D12_RESOURCE_STATE_COPY_SOURCE);
        tail.CopyBufferRegion(&readback, 0, &result, 0, 64);
        tail.Close().unwrap();
        q.ExecuteCommandLists(&[Some(tail.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        q.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 5000).unwrap();
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let observed = [p.cast::<u32>().read(), p.cast::<u32>().add(4).read()];
        let expected = [
            p.cast::<u32>().add(8).read(),
            p.cast::<u32>().add(12).read(),
        ];
        readback.Unmap(0, None);
        console_heap::release_global(view, 1);
        assert!(
            matches!(observed, [0, 42] | [42, 0]),
            "one predicate passes and one rejects: {observed:?}"
        );
        assert_eq!(
            observed, expected,
            "fixed inputs must use exactly the guest predicate polarity"
        );
    }
}
