use super::*;
use crate::hle::d3d12x::indirect::{self, IndirectInput, IndirectReads};
use crate::hle::d3d12x::list_x::native_buffer::{live, Copy, FallbackPage};
use crate::hle::d3d12x::page_mappings::queue;
use crate::hle::d3d12x::{console_heap, physical_directory, real};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Dxgi::Common::*;

const SHADER: &str = r#"
RWByteAddressBuffer row : register(u0);
RWByteAddressBuffer res : register(u1);
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), UAV(u0), UAV(u1)"
[RootSignature(RS)]
float4 VS(uint id : SV_VertexID) : SV_Position { float2 uv = float2((id << 1) & 2, id & 2); return float4(uv * 2 - 1, 0, 1); }
[RootSignature(RS)]
float4 PS() : SV_Target { uint old; res.InterlockedAdd(0, row.Load(32), old); res.InterlockedAdd(4, 1, old); return 0; }
"#;

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

/// Root-changing indirect draws on a translated-style graphics ABI: arguments and count read from
/// guest pages (across a page boundary) by the GPU gather, one io row per record carrying its own
/// root constant, a record past the draw budget emptied and a record past the GPU count left out.
#[test]
fn translated_draw_rows_take_gathered_guest_records_and_bound_their_draws() {
    unsafe {
        let mut dev = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let dev: ID3D12Device = dev.unwrap();
        let _uma = real::open_uma_for_test(&dev);
        assert!(!real::uma_window_vas().is_empty(), "unified memory must be open on the device");
        assert!(physical_directory::init(&dev));
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        queue::register(q.as_raw(), &dev).unwrap();
        let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();

        let size = 2 * 65536u64;
        let base = 0x3333_0000_0000u64;
        let (args_at, count_at) = (65500u64, 200u64);
        let mut guest = vec![0u32; size as usize / 4];
        let records: [(u32, [u32; 4]); 4] = [(100, [3, 1, 0, 0]), (7, [3, 0x7fff_ffff, 0, 0]), (300, [3, 1, 0, 0]), (5000, [3, 1, 0, 0])];
        for (i, (constant, draw)) in records.iter().enumerate() {
            let at = (args_at as usize + 20 * i) / 4;
            guest[at] = *constant;
            guest[at + 1..at + 5].copy_from_slice(draw);
        }
        guest[count_at as usize / 4] = 3;
        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, size);
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(guest.as_ptr(), p.cast::<u32>(), guest.len());
        upload.Unmap(0, None);
        let data = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, size);
        l.CopyBufferRegion(&data, 0, &upload, 0, size);
        let res = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, 256);
        l.CopyBufferRegion(&res, 0, &upload, 4096, 256);
        l.ResourceBarrier(&[indirect::transition(data.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        l.ResourceBarrier(&[indirect::transition(res.as_raw() as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);

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
        let page = |page: u32| {
            let off = u32::try_from((u64::from(page) << 16).checked_sub(base)?).ok()?;
            Some(FallbackPage::Raw { view, page_base: off })
        };
        let reads = IndirectReads {
            arguments: IndirectInput::Gather(Copy::read(base + args_at, 80, locator, page).unwrap()),
            count: Some(IndirectInput::Gather(Copy::read(base + count_at, 4, locator, page).unwrap())),
        };

        let layout = Layout { params: vec![Param::Constants { at: 0, count: 1 }], size: 4, inline: 4, inline_gfx: 4, ..Default::default() };
        let signature = [[5, 0, 0, 1], [0, 0, 0, 0]];
        let model = Rows::prepare(20, &signature, &layout, true, None, &[0u8; 4], &[0u32; 128]).unwrap();
        assert_eq!(model.action, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW);

        let vs = crate::gpu::recomp::dxc::compile(SHADER, "VS", "vs_6_0", &[]).unwrap();
        let ps = crate::gpu::recomp::dxc::compile(SHADER, "PS", "ps_6_0", &[]).unwrap();
        let abi: ID3D12RootSignature = dev.CreateRootSignature(0, &vs).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(abi.clone())),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&desc).unwrap();
        std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);

        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut rt).unwrap();
        let rt = rt.unwrap();
        let rtv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rt, None, rtv);
        l.OMSetRenderTargets(1, Some(&rtv), false, None);
        l.RSSetViewports(&[D3D12_VIEWPORT { Width: 1.0, Height: 1.0, MaxDepth: 1.0, ..Default::default() }]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: 1, bottom: 1 }]);
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        let u = crate::gpu::uma::get().unwrap();
        let window = u.committed_window().unwrap();
        let mut taken = Vec::new();
        let take = |n: usize| {
            let p = u.alloc_in(window, n.max(16), 256)? as usize;
            taken.push(p);
            Some(p)
        };
        let generated = dispatch::record_rows(&l, &model, 4, reads, vec![Arc::new(data.clone())], take, &live::Scope::packet(), true).unwrap();
        let rows_at = taken[1];

        let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() };
        let mut plain = None;
        dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: model.output_stride(), NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut plain).unwrap();
        let plain: ID3D12CommandSignature = plain.unwrap();
        l.SetGraphicsRootSignature(&abi);
        l.SetGraphicsRootUnorderedAccessView(0, generated.first_row);
        l.SetGraphicsRootUnorderedAccessView(1, res.GetGPUVirtualAddress());
        l.SetPipelineState(&pso);
        l.ExecuteIndirect(&plain, 4, &**generated.commands, 8, generated.count.as_ref(), 0);

        let readback = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, 512);
        l.ResourceBarrier(&[
            indirect::transition(res.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            indirect::transition(generated.commands.as_raw() as usize, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        l.CopyBufferRegion(&readback, 0, &res, 0, 256);
        l.CopyBufferRegion(&readback, 256, &**generated.commands, 0, 4 * u64::from(model.output_stride()));
        l.Close().unwrap();
        queue::reader(q.as_raw(), &[l.as_raw()], None);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        q.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 10000).unwrap();
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let words = std::slice::from_raw_parts(p.cast::<u32>(), 128).to_vec();
        readback.Unmap(0, None);
        console_heap::release_global(view, 1);
        assert_eq!((words[0], words[1]), (200, 2), "only records 0 and 2 draw: 1 is over the draw budget, 3 past the count");
        let stride = model.output_stride() as usize / 4;
        let row = |i: usize| rows_at as u64 + (i * model.row_bytes as usize) as u64;
        for (i, (constant, draw)) in records.iter().enumerate().take(3) {
            let command = &words[64 + i * stride..64 + (i + 1) * stride];
            let va = generated.first_row + (i * model.row_bytes as usize) as u64;
            assert_eq!((command[0], command[1]), (va as u32, (va >> 32) as u32), "record {i} binds its own row");
            let expect = if i == 1 { [0, 0, 0, 0] } else { *draw };
            assert_eq!(&command[2..6], &expect, "record {i}'s draw");
            assert_eq!((row(i) as *const u32).add(8).read_volatile(), *constant, "record {i}'s root constant in its row");
        }
    }
}
