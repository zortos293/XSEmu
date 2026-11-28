use std::mem::{size_of, ManuallyDrop};
use std::process::Command;
use std::ptr::{copy_nonoverlapping, null_mut};
use std::slice;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory4, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND};

use super::super::{real, title_textures};
use super::{Budget, Relief, State};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 16;
const ROW_BYTES: usize = WIDTH as usize * size_of::<u32>();
const TEX_BYTES: usize = ROW_BYTES * HEIGHT as usize;
const BUFFER_BYTES: usize = 4096;
const TEXTURE_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

unsafe fn discrete_device() -> Option<ID3D12Device> {
    let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("create DXGI factory");
    for index in 0.. {
        let adapter = match factory.EnumAdapters1(index) {
            Ok(adapter) => adapter,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => panic!("enumerate DXGI adapter: {e}"),
        };
        let desc = adapter.GetDesc1().expect("get DXGI adapter description");
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || desc.DedicatedVideoMemory == 0 {
            continue;
        }
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            continue;
        }
        let dev = dev.unwrap();
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
        dev.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE,
            (&mut arch as *mut D3D12_FEATURE_DATA_ARCHITECTURE).cast(),
            size_of::<D3D12_FEATURE_DATA_ARCHITECTURE>() as u32,
        )
        .expect("query D3D12 memory architecture");
        if !arch.UMA.as_bool() {
            return Some(dev);
        }
    }
    eprintln!("skipping residency GPU test: no discrete D3D12 adapter");
    None
}

struct Gpu {
    dev: ID3D12Device,
    q: ID3D12CommandQueue,
}

impl Gpu {
    unsafe fn new(dev: ID3D12Device) -> Self {
        let q = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        Self { dev, q }
    }

    unsafe fn run(&self, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
        let (_alloc, _list, fence) = self.submit(None, record);
        crate::gpu::fence::wait(&fence, 1, &self.dev, None, 10_000).expect("complete residency GPU test commands");
    }

    unsafe fn submit(
        &self,
        stage: Option<&str>,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12Fence) {
        let trace = |step| {
            if let Some(stage) = stage {
                eprintln!("adopted residency: {stage}: {step}");
            }
        };
        trace("allocating list");
        let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        trace("allocating completion fence");
        let fence = real::create_residency_fence(&self.dev).unwrap();
        trace("recording");
        record(&list);
        trace("closing list");
        list.Close().unwrap();
        trace("entering ExecuteCommandLists");
        title_textures::own_work(|| self.q.ExecuteCommandLists(&[Some(list.cast().unwrap())]));
        trace("queueing completion signal");
        self.q.Signal(&fence, 1).unwrap();
        trace("submitted");
        (alloc, list, fence)
    }
}

fn texture_desc() -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: WIDTH as u64,
        Height: HEIGHT,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    }
}

fn buffer_desc(bytes: usize) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

unsafe fn committed_buffer(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, ty: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES { Type: ty, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() };
    let mut resource = None;
    title_textures::own_work(|| dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut resource))
        .expect("create staging or shader output buffer");
    resource.unwrap()
}

unsafe fn placed(
    dev: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    flags: D3D12_HEAP_FLAGS,
    create_heap: &mut impl FnMut(&D3D12_HEAP_DESC) -> ID3D12Heap,
) -> (ID3D12Heap, ID3D12Resource) {
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    let heap = create_heap(&D3D12_HEAP_DESC {
        SizeInBytes: info.SizeInBytes,
        Alignment: info.Alignment,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        Flags: flags,
    });
    let mut resource = None;
    dev.CreatePlacedResource(&heap, 0, desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut resource).expect("create placed test resource");
    (heap, resource.unwrap())
}

unsafe fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) {
    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(r),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
        ..Default::default()
    }]);
}

unsafe fn activate(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: ManuallyDrop::new(None),
                pResourceAfter: std::mem::transmute_copy(r),
            }),
        },
        ..Default::default()
    }
}

unsafe fn copy_location(r: &ID3D12Resource, footprint: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>) -> D3D12_TEXTURE_COPY_LOCATION {
    let (ty, location) = match footprint {
        Some(fp) => (D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp }),
        None => (D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 }),
    };
    D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: ty, Anonymous: location }
}

fn assert_bytes(got: &[u8], expected: &[u8], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label} size");
    if let Some(i) = got.iter().zip(expected).position(|(a, b)| a != b) {
        panic!("{label}: byte {i} was {:#04x}, expected {:#04x}", got[i], expected[i]);
    }
}

struct Fixture {
    buffer: ID3D12Resource,
    texture: ID3D12Resource,
    heaps: [ID3D12Heap; 2],
    upload: ID3D12Resource,
    readback: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    total: usize,
    buffer_data: Vec<u8>,
    texels: Vec<u8>,
}

impl Fixture {
    unsafe fn new(dev: &ID3D12Device, mut create_heap: impl FnMut(&D3D12_HEAP_DESC) -> ID3D12Heap) -> Self {
        Self::with_texture(dev, texture_desc(), &mut create_heap)
    }

    unsafe fn with_texture(dev: &ID3D12Device, texture_desc: D3D12_RESOURCE_DESC, mut create_heap: impl FnMut(&D3D12_HEAP_DESC) -> ID3D12Heap) -> Self {
        let (bh, buffer) = placed(dev, &buffer_desc(BUFFER_BYTES), D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, &mut create_heap);
        let (th, texture) = placed(dev, &texture_desc, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, &mut create_heap);
        Self::with_resources(dev, &texture_desc, [bh, th], buffer, texture)
    }

    unsafe fn with_resources(
        dev: &ID3D12Device,
        texture_desc: &D3D12_RESOURCE_DESC,
        heaps: [ID3D12Heap; 2],
        buffer: ID3D12Resource,
        texture: ID3D12Resource,
    ) -> Self {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut texture_size = 0;
        dev.GetCopyableFootprints(texture_desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut texture_size));
        footprint.Offset = BUFFER_BYTES as u64;
        let total = BUFFER_BYTES + texture_size as usize;
        let staging = buffer_desc(total);
        let upload = committed_buffer(dev, &staging, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let readback = committed_buffer(dev, &staging, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let buffer_data: Vec<u8> = (0..BUFFER_BYTES as u32 / 4).flat_map(|i| (i.wrapping_mul(0x9e37_79b9) | 0x8000_0000).to_le_bytes()).collect();
        let texels: Vec<u8> = (0..WIDTH * HEIGHT).flat_map(|i| (0x3f80_0000 | (i.wrapping_mul(0x9e37_79b9) & 0x007f_ffff)).to_le_bytes()).collect();
        let mut p = null_mut();
        upload.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).unwrap();
        let p = p.cast::<u8>();
        copy_nonoverlapping(buffer_data.as_ptr(), p, BUFFER_BYTES);
        for (row, bytes) in texels.chunks_exact(ROW_BYTES).enumerate() {
            copy_nonoverlapping(bytes.as_ptr(), p.add(BUFFER_BYTES + row * footprint.Footprint.RowPitch as usize), ROW_BYTES);
        }
        upload.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: total }));
        Self { buffer, texture, heaps, upload, readback, footprint, total, buffer_data, texels }
    }

    unsafe fn upload(&self, gpu: &Gpu) {
        gpu.run(|list| {
            list.ResourceBarrier(&[activate(&self.buffer), activate(&self.texture)]);
            list.CopyBufferRegion(&self.buffer, 0, &self.upload, 0, BUFFER_BYTES as u64);
            list.CopyTextureRegion(&copy_location(&self.texture, None), 0, 0, 0, &copy_location(&self.upload, Some(self.footprint)), None);
            transition(list, &self.texture, D3D12_RESOURCE_STATE_COPY_DEST, TEXTURE_READ);
        });
    }

    unsafe fn verify_copies(&self, gpu: &Gpu, label: &str) {
        gpu.run(|list| {
            list.CopyBufferRegion(&self.readback, 0, &self.buffer, 0, BUFFER_BYTES as u64);
            transition(list, &self.texture, TEXTURE_READ, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyTextureRegion(&copy_location(&self.readback, Some(self.footprint)), 0, 0, 0, &copy_location(&self.texture, None), None);
            transition(list, &self.texture, D3D12_RESOURCE_STATE_COPY_SOURCE, TEXTURE_READ);
        });
        self.verify_readback(label);
    }

    unsafe fn verify_readback(&self, label: &str) {
        let mut p = null_mut();
        self.readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: self.total }), Some(&mut p)).unwrap();
        let p = p.cast::<u8>();
        assert_bytes(slice::from_raw_parts(p, BUFFER_BYTES), &self.buffer_data, label);
        for (row, bytes) in self.texels.chunks_exact(ROW_BYTES).enumerate() {
            let offset = self.footprint.Offset as usize + row * self.footprint.Footprint.RowPitch as usize;
            assert_bytes(slice::from_raw_parts(p.add(offset), ROW_BYTES), bytes, &format!("{label}, texture row {row}"));
        }
        self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
    }
}

struct Sample {
    srv: ID3D12DescriptorHeap,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    output: ID3D12Resource,
    readback: ID3D12Resource,
}

impl Sample {
    unsafe fn new(dev: &ID3D12Device, texture: &ID3D12Resource) -> Self {
        let range = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 1, ..Default::default() };
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, ..Default::default() },
        ];
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let mut blob = None;
        D3D12SerializeRootSignature(
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                ..Default::default()
            },
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            None,
        )
        .expect("serialize sampling root signature");
        let blob = blob.unwrap();
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())).unwrap();
        let shader = super::super::real::compile(
            r#"
            Texture2D<float> tex : register(t0);
            SamplerState pointSampler : register(s0);
            RWByteAddressBuffer outputBits : register(u0);
            [numthreads(8, 8, 1)]
            void main(uint3 id : SV_DispatchThreadID) {
                uint w, h;
                tex.GetDimensions(w, h);
                float2 uv = (float2(id.xy) + 0.5) / float2(w, h);
                outputBits.Store(4 * (id.y * w + id.x), asuint(tex.SampleLevel(pointSampler, uv, 0.0)));
            }
        "#,
            "cs_5_0",
        );
        assert!(!shader.is_empty(), "compile residency sample shader");
        let pso = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr().cast(), BytecodeLength: shader.len() },
                ..Default::default()
            })
            .expect("create residency sample pipeline");
        let srv: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .unwrap();
        dev.CreateShaderResourceView(texture, None, srv.GetCPUDescriptorHandleForHeapStart());
        let output = committed_buffer(
            dev,
            &D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..buffer_desc(TEX_BYTES) },
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let readback = committed_buffer(dev, &buffer_desc(TEX_BYTES), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        Self { srv, root, pso, output, readback }
    }

    unsafe fn verify(&self, gpu: &Gpu, fixture: &Fixture, label: &str) {
        gpu.run(|list| self.record(list, fixture));
        self.verify_output(fixture, label);
    }

    unsafe fn record(&self, list: &ID3D12GraphicsCommandList, fixture: &Fixture) {
        list.CopyBufferRegion(&self.output, 0, &fixture.upload, 0, TEX_BYTES as u64);
        transition(list, &self.output, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list.SetPipelineState(&self.pso);
        list.SetComputeRootSignature(&self.root);
        list.SetDescriptorHeaps(&[Some(self.srv.clone())]);
        list.SetComputeRootDescriptorTable(0, self.srv.GetGPUDescriptorHandleForHeapStart());
        list.SetComputeRootUnorderedAccessView(1, self.output.GetGPUVirtualAddress());
        list.Dispatch(WIDTH / 8, HEIGHT / 8, 1);
        transition(list, &self.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list.CopyBufferRegion(&self.readback, 0, &self.output, 0, TEX_BYTES as u64);
    }

    unsafe fn verify_output(&self, fixture: &Fixture, label: &str) {
        let mut p = null_mut();
        self.readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: TEX_BYTES }), Some(&mut p)).unwrap();
        assert_bytes(slice::from_raw_parts(p.cast::<u8>(), TEX_BYTES), &fixture.texels, label);
        self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
    }

    unsafe fn verify_pixel(&self, gpu: &Gpu, fixture: &Fixture) {
        let vs = real::compile(
            "float4 main(uint id : SV_VertexID) : SV_Position {
                float2 uv = float2((id << 1) & 2, id & 2);
                return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
            }",
            "vs_5_0",
        );
        let ps = real::compile(
            "Texture2D<float> tex : register(t0);
             SamplerState s : register(s0);
             float main(float4 pos : SV_Position) : SV_Target0 {
                 uint w, h;
                 tex.GetDimensions(w, h);
                 return tex.SampleLevel(s, pos.xy / float2(w, h), 0);
             }",
            "ps_5_0",
        );
        assert!(!vs.is_empty() && !ps.is_empty(), "compile L0 graphics shaders");
        let desc = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..texture_desc() };
        let mut pipeline = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&self.root),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            SampleDesc: desc.SampleDesc,
            NumRenderTargets: 1,
            ..Default::default()
        };
        pipeline.RTVFormats[0] = desc.Format;
        pipeline.RasterizerState =
            D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
        pipeline.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let pso: ID3D12PipelineState = gpu.dev.CreateGraphicsPipelineState(&pipeline).expect("create L0 pixel-sampling pipeline");
        let rt = committed_buffer(&gpu.dev, &desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        let heap: ID3D12DescriptorHeap = gpu
            .dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .unwrap();
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        gpu.dev.CreateRenderTargetView(&rt, None, rtv);
        gpu.run(|list| {
            transition(list, &fixture.texture, TEXTURE_READ, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            list.ClearRenderTargetView(rtv, &[0.0; 4], None);
            list.SetPipelineState(&pso);
            list.SetGraphicsRootSignature(&self.root);
            list.SetDescriptorHeaps(&[Some(self.srv.clone())]);
            list.SetGraphicsRootDescriptorTable(0, self.srv.GetGPUDescriptorHandleForHeapStart());
            list.RSSetViewports(&[D3D12_VIEWPORT { Width: WIDTH as f32, Height: HEIGHT as f32, MaxDepth: 1.0, ..Default::default() }]);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: WIDTH as i32, bottom: HEIGHT as i32 }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
            transition(list, &fixture.texture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, TEXTURE_READ);
            transition(list, &rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyTextureRegion(&copy_location(&fixture.readback, Some(fixture.footprint)), 0, 0, 0, &copy_location(&rt, None), None);
            list.CopyBufferRegion(&fixture.readback, 0, &fixture.buffer, 0, BUFFER_BYTES as u64);
        });
        fixture.verify_readback("pixel shader sampled a reserved texture from L0");
    }
}

unsafe fn native_heap(dev: &ID3D12Device, desc: &D3D12_HEAP_DESC) -> windows::core::Result<ID3D12Heap> {
    let mut heap = None;
    title_textures::own_work(|| dev.CreateHeap(desc, &mut heap))?;
    Ok(heap.unwrap())
}

#[test]
fn evicted_local_buffer_and_texture_keep_contents() {
    unsafe {
        let Some(dev) = discrete_device() else { return };
        let gpu = Gpu::new(dev.clone());
        let fixture = Fixture::new(&dev, |desc| native_heap(&dev, desc).expect("create test local heap"));
        let sampling = Sample::new(&dev, &fixture.texture);
        let [buffer, texture] = fixture.heaps.each_ref().map(|heap| heap.as_raw() as usize);
        let sizes = fixture.heaps.each_ref().map(|heap| heap.GetDesc().SizeInBytes);
        let mut state = State::new(&dev);
        let budget = sizes.iter().sum();
        state.budget = Budget { usage: budget, ..Default::default() };
        state.budget_override = Some(budget);
        state.track(buffer, sizes[0], true, false);
        state.track(texture, sizes[1], true, false);
        fixture.upload(&gpu);
        sampling.verify(&gpu, &fixture, "texture sampled before eviction");
        state.objects.get_mut(&texture).unwrap().last_used = 1;

        let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let submitted = std::sync::Arc::new(dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap());
        gpu.q.Wait(&gate, 1).unwrap();
        gpu.q.Signal(&*submitted, 1).unwrap();
        super::stamp(&mut state.opaque_last, &submitted, 1);
        let trimmed_while_pending = state.trim(&dev, 0, Relief::pressure());
        gate.Signal(1).unwrap();
        crate::gpu::fence::wait(&submitted, 1, &dev, None, 10_000).unwrap();
        assert!(!trimmed_while_pending, "an opaque list pins all reachable memory until its queue completes");
        assert!(state.objects[&buffer].resident && state.objects[&texture].resident);

        assert!(state.trim(&dev, 0, Relief::pressure()), "evicting the older buffer should bring usage under budget");
        assert!(!state.objects[&buffer].resident && state.objects[&texture].resident, "LRU should evict only the buffer");
        state.restore(&dev, &[buffer]).expect("restore the buffer heap");
        assert!(state.objects[&buffer].resident && !state.objects[&texture].resident, "the texture should make room for the buffer");

        state.budget_override = Some(2 * budget);
        state.restore(&dev, &[texture]).expect("restore the texture heap");
        assert!(state.objects[&buffer].resident && state.objects[&texture].resident, "both heaps should fit the raised budget");
        sampling.verify(&gpu, &fixture, "texture sampled after eviction with the same SRV");
        fixture.verify_copies(&gpu, "contents copied after eviction and restore");
    }
}

#[test]
fn low_budget_spills_buffer_and_texture_to_l0_without_changing_contents() {
    unsafe {
        let Some(dev) = discrete_device() else { return };
        let gpu = Gpu::new(dev.clone());
        let mut state = State::new(&dev);
        state.budget = Budget::default();
        state.budget_override = Some(1);
        let fixture = Fixture::new(&dev, |desc| {
            let local_fit = state.trim(&dev, desc.SizeInBytes, Relief::pressure());
            assert!(!local_fit, "heap unexpectedly fit in a one-byte local budget");
            let mut spill_error = None;
            let l0 = super::spill_properties(&dev, desc.Properties, desc.Flags).expect("discrete adapter has an L0 spill heap type");
            let (heap, spilled) = super::allocate_with(desc.Properties, l0, local_fit, |props| {
                let result = native_heap(&dev, &D3D12_HEAP_DESC { Properties: *props, ..*desc });
                if props.MemoryPoolPreference == D3D12_MEMORY_POOL_L0 {
                    spill_error = result.as_ref().err().cloned();
                }
                result
            })
            .expect("allocate test heap via the L0 spill path");
            let props = heap.GetDesc().Properties;
            assert!(spilled, "forced L0 allocation fell back to local memory: {spill_error:?}");
            assert_eq!(props.Type, D3D12_HEAP_TYPE_CUSTOM);
            assert_eq!(props.MemoryPoolPreference, D3D12_MEMORY_POOL_L0);
            assert_eq!(props.CPUPageProperty, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE);
            heap
        });
        fixture.upload(&gpu);
        fixture.verify_copies(&gpu, "contents copied from L0 spill heaps");
        Sample::new(&dev, &fixture.texture).verify(&gpu, &fixture, "texture sampled from an L0 spill heap");
    }
}

#[test]
fn reserved_buffer_uses_a_non_local_tile_heap() {
    unsafe {
        let Some(dev) = discrete_device() else { return };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
        .unwrap();
        if options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_1.0 {
            eprintln!("skipping L0 tile test: no tiled resources");
            return;
        }
        let gpu = Gpu::new(dev.clone());
        let tile = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as usize;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: tile as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            ..Default::default()
        };
        let l0 = super::spill_properties(&dev, desc.Properties, desc.Flags).unwrap();
        let (heap, spilled) = super::allocate_with(desc.Properties, l0, false, |props| native_heap(&dev, &D3D12_HEAP_DESC { Properties: *props, ..desc }))
            .expect("create tiled spill heap");
        assert!(spilled);
        let d = buffer_desc(tile);
        let mut tiled: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&d, D3D12_RESOURCE_STATE_COMMON, None, &mut tiled).unwrap();
        let tiled = tiled.unwrap();
        gpu.q.UpdateTileMappings(
            &tiled,
            1,
            Some(&D3D12_TILED_RESOURCE_COORDINATE::default()),
            None,
            &heap,
            1,
            None,
            Some(&0),
            Some(&1),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );

        let upload = committed_buffer(&dev, &d, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let readback = committed_buffer(&dev, &d, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let expected: Vec<u8> = (0..tile as u32 / 4).flat_map(|i| i.wrapping_mul(0xc27d_4eb1).to_le_bytes()).collect();
        let mut p = null_mut();
        upload.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).unwrap();
        copy_nonoverlapping(expected.as_ptr(), p.cast::<u8>(), tile);
        upload.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: tile }));
        gpu.run(|list| {
            list.CopyBufferRegion(&tiled, 0, &upload, 0, tile as u64);
            transition(list, &tiled, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyBufferRegion(&readback, 0, &tiled, 0, tile as u64);
        });
        readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: tile }), Some(&mut p)).unwrap();
        assert_bytes(slice::from_raw_parts(p.cast::<u8>(), tile), &expected, "buffer mapped to an L0 tile heap");
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
    }
}

#[test]
fn reserved_texture_samples_from_a_non_local_tile_heap() {
    unsafe {
        let Some(dev) = discrete_device() else { return };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
        .unwrap();
        if options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_1.0 {
            eprintln!("skipping L0 tiled texture: no tiled resources");
            return;
        }
        let gpu = Gpu::new(dev.clone());
        let desc = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..texture_desc() };
        let mut texture: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut texture).expect("create reserved texture");
        let texture = texture.unwrap();
        let mut tiles = 0;
        dev.GetResourceTiling(&texture, Some(&mut tiles), None, None, None, 0, null_mut());
        assert!(tiles > 0);
        let flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        let default = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() };
        let heap = native_heap(
            &dev,
            &D3D12_HEAP_DESC {
                SizeInBytes: u64::from(tiles) * D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
                Properties: super::spill_properties(&dev, default, flags).unwrap(),
                Flags: flags,
                ..Default::default()
            },
        )
        .expect("create L0 tile pool for reserved texture");
        assert_eq!(heap.GetDesc().Properties.MemoryPoolPreference, D3D12_MEMORY_POOL_L0);
        let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles, ..Default::default() };
        gpu.q.UpdateTileMappings(
            &texture,
            1,
            Some(&D3D12_TILED_RESOURCE_COORDINATE::default()),
            Some(&region),
            &heap,
            1,
            None,
            Some(&0),
            Some(&tiles),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        let (bh, buffer) = placed(&dev, &buffer_desc(BUFFER_BYTES), D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, &mut |d| native_heap(&dev, d).unwrap());
        let fixture = Fixture::with_resources(&dev, &desc, [bh, heap], buffer, texture);
        fixture.upload(&gpu);
        Sample::new(&dev, &fixture.texture).verify_pixel(&gpu, &fixture);
        fixture.verify_copies(&gpu, "bitwise readback from an L0 tiled texture");
    }
}

#[test]
fn unordered_access_texture_writes_and_samples_on_l0() {
    unsafe {
        let Some(dev) = discrete_device() else { return };
        let gpu = Gpu::new(dev.clone());
        let tex = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..texture_desc() };
        let mut fixture = Fixture::with_texture(&dev, tex, |desc| {
            let props = super::spill_properties(&dev, desc.Properties, desc.Flags).unwrap();
            native_heap(&dev, &D3D12_HEAP_DESC { Properties: props, ..*desc }).expect("create L0 UAV heap")
        });
        fixture.upload(&gpu);
        assert_eq!(fixture.heaps[1].GetDesc().Properties.MemoryPoolPreference, D3D12_MEMORY_POOL_L0);
        let make_uav = |flags| {
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: flags,
                    ..Default::default()
                })
                .unwrap();
            dev.CreateUnorderedAccessView(&fixture.texture, None, None, heap.GetCPUDescriptorHandleForHeapStart());
            heap
        };
        let cpu = make_uav(D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let visible = make_uav(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        let value = 0.625f32;
        gpu.run(|list| {
            transition(list, &fixture.texture, TEXTURE_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list.SetDescriptorHeaps(&[Some(visible.clone())]);
            list.ClearUnorderedAccessViewFloat(
                visible.GetGPUDescriptorHandleForHeapStart(),
                cpu.GetCPUDescriptorHandleForHeapStart(),
                &fixture.texture,
                &[value; 4],
                &[],
            );
            transition(list, &fixture.texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, TEXTURE_READ);
        });
        for texel in fixture.texels.chunks_exact_mut(4) {
            texel.copy_from_slice(&value.to_le_bytes());
        }
        Sample::new(&dev, &fixture.texture).verify(&gpu, &fixture, "L0 UAV clear sampled through an SRV");
        fixture.verify_copies(&gpu, "bitwise readback of a GPU write to an L0 UAV");
    }
}

#[test]
#[ignore = "requires a discrete GPU and one-shot XCL_VRAM_EVICT=1 XCL_VRAM_BUDGET_MB=1"]
fn tiny_budget_adopted_queues_keep_pending_heaps_and_restore_sampled_texture() {
    adopted_queue_case(true);
}

#[test]
#[ignore = "requires a discrete GPU, XCL_VRAM_BUDGET_MB=1, and XCL_VRAM_EVICT unset"]
fn adopted_queues_spill_only_does_not_hang() {
    adopted_queue_case(false);
}

fn adopted_queue_case(evict: bool) {
    if evict {
        assert_eq!(std::env::var("XCL_VRAM_EVICT").as_deref(), Ok("1"), "set XCL_VRAM_EVICT=1");
    } else {
        assert!(std::env::var_os("XCL_VRAM_EVICT").is_none(), "unset XCL_VRAM_EVICT for the spill-only test");
    }
    assert_eq!(std::env::var("XCL_VRAM_BUDGET_MB").as_deref(), Ok("1"), "set XCL_VRAM_BUDGET_MB=1");
    const CHILD: &str = "_XCL_ADOPTED_RESIDENCY_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let current = std::thread::current();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([current.name().expect("libtest name"), "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .spawn()
            .expect("start isolated adopted residency test");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = child.try_wait().expect("check adopted residency child") {
                assert!(status.success(), "isolated adopted residency test failed: {status}");
                return;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                panic!("isolated adopted residency test timed out after 30 seconds; see last stage above");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
    unsafe {
        let dev = discrete_device().expect("adopted residency test requires a discrete D3D12 adapter");
        eprintln!("adopted residency: uploading native fixture");
        let setup = Gpu::new(dev.clone());
        let fixture = Fixture::new(&dev, |desc| {
            native_heap(&dev, &D3D12_HEAP_DESC { SizeInBytes: desc.SizeInBytes.max(2 << 20), ..*desc })
                .expect("create local fixture before applying title budget")
        });
        fixture.upload(&setup);
        let sampling = Sample::new(&dev, &fixture.texture);
        sampling.verify(&setup, &fixture, "baseline before adopted queue eviction");

        eprintln!("adopted residency: adopting device and queues");
        real::adopt(dev.as_raw(), real::Kind::Device);
        let foreground = Gpu::new(dev.clone());
        let background = Gpu::new(dev.clone());
        assert!(real::is_adopted(foreground.q.as_raw()) && real::is_adopted(background.q.as_raw()));
        let track_heap = |heap: &ID3D12Heap, res: &ID3D12Resource| {
            real::adopt(heap.as_raw(), real::Kind::Heap);
            real::adopt(res.as_raw(), real::Kind::Resource);
            let desc = heap.GetDesc();
            super::track(&dev, heap.as_raw() as usize, desc.Properties, desc.SizeInBytes, false, true, false);
            super::link(res.as_raw() as usize, heap.as_raw() as usize);
        };
        if evict {
            track_heap(&fixture.heaps[0], &fixture.buffer);
            for (resource, ty) in
                [(&fixture.readback, D3D12_HEAP_TYPE_READBACK), (&fixture.upload, D3D12_HEAP_TYPE_UPLOAD), (&sampling.readback, D3D12_HEAP_TYPE_READBACK)]
            {
                real::adopt(resource.as_raw(), real::Kind::Resource);
                super::track(
                    &dev,
                    resource.as_raw() as usize,
                    D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() },
                    resource.GetDesc().Width,
                    false,
                    false,
                    false,
                );
            }
        }
        let gate = real::create_residency_fence(&dev).unwrap();
        let (unblock, wait) = mpsc::channel();
        let watchdog_gate = gate.clone();
        let watchdog = std::thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(Duration::from_secs(5)) {
                eprintln!("adopted residency: watchdog released the stalled queue gate after five seconds");
            }
            watchdog_gate.Signal(1).expect("release GPU queue gate");
        });
        eprintln!("adopted residency: queueing background fence wait");
        background.q.Wait(&gate, 1).unwrap();
        let (_alloc, _list, done) = background.submit(Some("background copy"), |list| {
            list.CopyBufferRegion(&fixture.readback, 0, &fixture.buffer, 0, BUFFER_BYTES as u64);
        });
        if evict {
            track_heap(&fixture.heaps[1], &fixture.texture);
        }
        let (_foreground_alloc, _foreground_list, foreground_done) = foreground.submit(Some("foreground trim"), |list| {
            list.CopyBufferRegion(&sampling.readback, 0, &fixture.upload, 0, TEX_BYTES as u64);
        });
        let snapshot = evict.then(|| {
            eprintln!("adopted residency: reading residency state before releasing background queue");
            let registry = super::registry().lock().unwrap();
            let state = &registry.devices[&(dev.as_raw() as usize)];
            let buffer = &state.objects[&(fixture.heaps[0].as_raw() as usize)];
            let texture = &state.objects[&(fixture.heaps[1].as_raw() as usize)];
            (state.limit(), state.budget.usage, buffer.resident && buffer.last.iter().any(|(f, v)| f.GetCompletedValue() < *v), !texture.resident)
        });
        let was_gated = gate.GetCompletedValue() == 0;
        let _ = unblock.send(());
        watchdog.join().expect("release gated queue");
        eprintln!("adopted residency: waiting for copy completion");
        crate::gpu::fence::wait(&done, 1, &dev, None, 10_000).expect("complete gated queue work");
        crate::gpu::fence::wait(&foreground_done, 1, &dev, None, 10_000).expect("complete foreground trim-triggering copy");
        assert!(was_gated, "the watchdog had to release the gate before the trim could finish");
        if let Some((limit, usage, buffer_pending, texture_evicted)) = snapshot {
            assert_eq!(limit, 1 << 20, "focused runner must enable the real 1 MiB budget");
            assert!(buffer_pending, "the adopted copy list did not protect its heap with an unfinished queue fence");
            assert!(texture_evicted, "the adopted queue did not evict the idle texture heap");
            assert!(usage > super::percent(limit, 90), "budget should still be exceeded while the busy heap remains resident");
        }

        let (_sample_alloc, _sample_list, sample_done) = foreground.submit(Some("restore and dispatch"), |list| {
            sampling.record(list, &fixture);
            assert!(
                !evict || super::registry().lock().unwrap().lists.get(&(list.as_raw() as usize)).is_some_and(|uses| uses.opaque),
                "the adopted dispatch must mark its implicit SRV reads opaque",
            );
        });
        crate::gpu::fence::wait(&sample_done, 1, &dev, None, 10_000).expect("complete restore and dispatch");
        sampling.verify_output(&fixture, "adopted dispatch sampled the texture without rewriting it");
    }
}
