use super::commands::Kernels;
use crate::gpu::recomp::{dxc, GPU_PAGE_DIRECTORY_HLSL};
use windows::Win32::Graphics::Direct3D12::*;

// These kernels are not compiled until the bridge is wired; kept for the next fix.
#[allow(dead_code)]
const TRANSFER: &str = include_str!("transfer.hlsl");
// These kernels are not compiled until the bridge is wired; kept for the next fix.
#[allow(dead_code)]
const ROOT: &str = r#"#define IMAGE_BRIDGE_ROOT [RootSignature("RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),RootConstants(num32BitConstants=8,b0),SRV(t0),UAV(u0),UAV(u1)")]
"#;
// These kernels are not compiled until the bridge is wired; kept for the next fix.
#[allow(dead_code)]
const RESOLVE: &str = r#"
uint4 ResolveImagePage(uint vpn, uint directory, uint umaHi, uint umaViews, uint windows) {
    uint2 va = uint2(vpn << 16, vpn >> 16);
    uint2 p = xcl_lookup_gpu_pte(va, directory);
    if (p.x == XCL_PTE_MAPPED) {
        uint uav = p.y & 0xfffffu;
        uint page = (p.y >> 20) & 127u;
        return uint4(uav, page << 16, 0, IMAGE_PAGE_PHYSICAL);
    }
    if (p.x != XCL_PTE_ABSENT) return 0;
    uint w = va.y - umaHi;
    if (w < windows) return uint4(umaViews + w, va.x, 0, IMAGE_PAGE_UMA);
    return uint4(0, 0, 0, IMAGE_PAGE_NATIVE);
}
"#;

impl Kernels {
    // These kernels are not compiled until the bridge is wired; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn compile(dev: &ID3D12Device) -> Result<Self, String> {
        let capture_source = format!("{ROOT}#define IMAGE_BRIDGE_SNAPSHOT\n{}",
            TRANSFER.replace("#include \"image_page_resolver.hlsli\"", &format!("{GPU_PAGE_DIRECTORY_HLSL}\n{RESOLVE}")));
        let transfer_source = format!("{ROOT}{TRANSFER}");
        let capture = dxc::compile_cached(&capture_source, "SnapshotImagePages", "cs_6_6")?.0;
        let gather = dxc::compile_cached(&transfer_source, "GatherImage", "cs_6_6")?.0;
        let scatter = dxc::compile_cached(&transfer_source, "ScatterImage", "cs_6_6")?.0;
        let signature: ID3D12RootSignature = dev.CreateRootSignature(0, &capture).map_err(|e| e.to_string())?;
        let make = |code: &[u8]| -> Result<ID3D12PipelineState, String> {
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(signature.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            let made = dev.CreateComputePipelineState(&desc);
            std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
            made.map_err(|e| e.to_string())
        };
        Ok(Self { capture: make(&capture)?, gather: make(&gather)?, scatter: make(&scatter)?, signature })
    }
}
