pub(crate) mod recipes;
pub(crate) mod translated;
pub(crate) mod fixed;
pub(crate) mod live;
pub(crate) mod destinations;
#[cfg(test)]
mod copy_test;
mod transfer;
pub(crate) use transfer::Transfer;

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) struct Operand {
    pub(crate) guest_address: Option<u64>,
    pub(crate) resource: ID3D12Resource,
    pub(crate) native_offset: u64,
}

// Prebuilt native-input assembly has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct NativeIa<L> {
    vertex: Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)>,
    index: Option<D3D12_INDEX_BUFFER_VIEW>,
    leases: Vec<L>,
}

// Prebuilt native-input assembly has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum IaDomain {
    Guest,
    Native(ID3D12Resource),
}

// Prebuilt native-input assembly has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum IaInput {
    Null,
    Native(ID3D12Resource),
    Gather(Read),
}

// Prebuilt native-input assembly has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn prebuilt_ia(
    recipe: &[recipes::Address],
    locator: u32,
    mut classify: impl FnMut(recipes::Address) -> Option<IaDomain>,
    mut fallback: impl FnMut(recipes::Address, u32) -> Option<FallbackPage>,
) -> Option<Vec<(recipes::Address, IaInput)>> {
    recipe
        .iter()
        .map(|&a| {
            let bytes = a.known_bytes()?;
            let input = if a.original == 0 {
                IaInput::Null
            } else {
                match classify(a)? {
                    IaDomain::Guest => {
                        IaInput::Gather(Copy::read(a.original, bytes, locator, |p| fallback(a, p))?)
                    }
                    IaDomain::Native(r) => {
                        let d = unsafe { r.GetDesc() };
                        let base = unsafe { r.GetGPUVirtualAddress() };
                        if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                            || base == 0
                            || a.original.checked_sub(base)?.checked_add(bytes)? > d.Width
                        {
                            return None;
                        }
                        IaInput::Native(r)
                    }
                }
            };
            Some((a, input))
        })
        .collect()
}

impl<L> NativeIa<L> {
    // Prebuilt native-input assembly has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn resolve(
        recipe: &[recipes::Address],
        mut resolver: impl FnMut(recipes::Address) -> Option<recipes::Resolved<L>>,
    ) -> Option<Self> {
        use recipes::Binding;
        let mut native = Self {
            vertex: Vec::new(),
            index: None,
            leases: Vec::new(),
        };
        for &address in recipe {
            let (va, lease) = address.resolve(&mut resolver)?;
            match address.binding {
                Binding::Vertex {
                    slot,
                    bytes,
                    stride,
                } => native.vertex.push((
                    slot,
                    D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: va,
                        SizeInBytes: bytes,
                        StrideInBytes: stride,
                    },
                )),
                Binding::Index { bytes, format } => {
                    native.index = Some(D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: va,
                        SizeInBytes: bytes,
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format),
                    })
                }
                _ => return None,
            }
            native.leases.extend(lease);
        }
        Some(native)
    }

    // Prebuilt native-input assembly has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn bind(&self, list: &ID3D12GraphicsCommandList) {
        for (slot, view) in &self.vertex {
            list.IASetVertexBuffers(*slot, Some(std::slice::from_ref(view)));
        }
        if let Some(view) = self.index.as_ref() {
            list.IASetIndexBuffer(Some(view));
        }
    }
}

impl Operand {
    pub(crate) unsafe fn retain(raw: usize, offset: u64) -> Option<Self> {
        let resource = ID3D12Resource::from_raw_borrowed(&(raw as *mut c_void))?.clone();
        let guest_address = match super::real::title_address(raw) {
            Some(base) => Some((base as u64).checked_add(offset)?),
            None => None,
        };
        let native_offset = offset.checked_add(super::real::buffer_delta(raw))?;
        Some(Self {
            resource,
            guest_address,
            native_offset,
        })
    }
}

static PIPELINE: std::sync::Mutex<Option<(ID3D12Device, std::sync::Arc<Pipeline>)>> = std::sync::Mutex::new(None);

/// The device's Capture/Gather/Scatter pipeline, made once per device.
pub(crate) unsafe fn shared_pipeline(dev: &ID3D12Device) -> Result<std::sync::Arc<Pipeline>, String> {
    let mut cache = PIPELINE.lock().unwrap();
    if cache.as_ref().is_none_or(|(d, _)| d != dev) {
        *cache = Some((dev.clone(), std::sync::Arc::new(Pipeline::new(dev)?)));
    }
    Ok(cache.as_ref().unwrap().1.clone())
}

/// A title's CopyBufferRegion on `list` (offsets as the title gave them). When a page of either
/// title range has an entry in the GPU page directory, the copy runs on the GPU through the
/// recording's locator after the list's earlier writers: Capture, Gather and Scatter, split on
/// both sides' page boundaries, each write to its page's one owner (a mapped physical page, the
/// title buffer's own memory, or nothing for an explicit unmap). true: recorded; false: the caller
/// copies natively as before, a refusal of a directory-mapped range counted.
pub(crate) unsafe fn intercept_copy(list: usize, dst: usize, dst_offset: u64, src: usize, src_offset: u64, bytes: u64) -> bool {
    match directory_copy(list, dst, dst_offset, src, src_offset, bytes) {
        Ok(recorded) => recorded,
        Err(why) => {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: CopyBufferRegion of a directory-mapped title range copied natively: {why}; {n} so far");
            }
            false
        }
    }
}

/// Whether the GPU page directory has an entry (a mapping or an explicit unmap) for a page of the
/// title range `va`..`va + bytes` in its newest version.
pub(crate) fn directory_touched(va: u64, bytes: u64) -> bool {
    use super::super::physical_directory::{current_topology, PageLookup};
    let Some(topology) = current_topology() else { return false };
    let Some(last) = va.checked_add(bytes.saturating_sub(1)) else { return true };
    (va >> 16..=last >> 16).any(|page| !matches!(topology.lookup(page << 16), Ok(PageLookup::Absent)))
}

/// What a helper on the title's list `list` needs: the list, its device, the shared view heap,
/// the proof its state can be given back (`scope`), and its recording's locator.
pub(crate) struct HelperContext {
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) device: ID3D12Device,
    pub(crate) views: ID3D12DescriptorHeap,
    pub(crate) scope: live::Scope,
    pub(crate) locator: u32,
}

pub(crate) unsafe fn helper_context(list: *mut c_void) -> Result<HelperContext, String> {
    let com = ID3D12GraphicsCommandList::from_raw_borrowed(&list).ok_or("no command list")?.clone();
    if !matches!(com.GetType(), D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        return Err("a COPY list cannot run page transfers".into());
    }
    let scope = super::tracked_scope(list).ok_or("the list's queries, predication or heaps are not known to be restorable")?;
    let locator = match super::super::physical_directory::recordings::packet_index(list as usize) {
        0 => return Err("no recording locator (unified-memory path)".into()),
        l => l,
    };
    let mut device: Option<ID3D12Device> = None;
    com.GetDevice(&mut device).map_err(|e| e.to_string())?;
    let device = device.ok_or("the command list has no device")?;
    let views = super::super::console_heap::global_heap().ok_or("no shared view heap for the page transfer")?;
    Ok(HelperContext { list: com, device, views, scope, locator })
}

/// Whether title buffer `raw` at `offset`..`offset + bytes` is title memory with a page the
/// directory has an entry for. Asks nothing of the COM object.
pub(crate) fn title_range_touched(raw: usize, offset: u64, bytes: u64) -> bool {
    bytes != 0 && super::real::title_address(raw).and_then(|base| (base as u64).checked_add(offset)).is_some_and(|va| directory_touched(va, bytes))
}

/// Directory-touched CopyBufferRegions recorded on COPY lists: copied natively (the play81 path),
/// counted apart from the other refusals. A COPY list's copy operands are fixed when it is
/// recorded, before its execution's ready point selects the topology.
static COPY_LIST_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

unsafe fn directory_copy(list: usize, dst: usize, dst_offset: u64, src: usize, src_offset: u64, bytes: u64) -> Result<bool, String> {
    if !title_range_touched(src, src_offset, bytes) && !title_range_touched(dst, dst_offset, bytes) {
        return Ok(false);
    }
    if super::list_type(list as *mut c_void) == D3D12_COMMAND_LIST_TYPE_COPY {
        let n = COPY_LIST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: CopyBufferRegion on a COPY list touches directory-mapped title memory, copied natively: {n} so far");
        }
        return Ok(false);
    }
    let (Some(source), Some(destination)) = (Operand::retain(src, src_offset), Operand::retain(dst, dst_offset)) else {
        return Ok(false);
    };
    let (Some(from), Some(to)) = (source.guest_address, destination.guest_address) else {
        return Err("one side of the copy is not title memory".into());
    };
    let ctx = helper_context(list as *mut c_void)?;
    let mut leases: Vec<fixed::Lease> = Vec::new();
    let copy = Copy::prepare(Transfer::new(from, to, bytes).ok_or("the copy leaves the 48-bit range")?, ctx.locator, |page, write| {
        let op = if write { &destination } else { &source };
        let (meta, lease) = live::fallback(&op.resource, op.guest_address?, op.native_offset, page)?;
        leases.push(lease);
        Some(meta)
    })
    .ok_or("a page of the copy has no proven owner")?;
    let pipeline = shared_pipeline(&ctx.device)?;
    let (plan_bytes, data_bytes) = copy.storage_bytes();
    let storage = std::sync::Arc::new((live::scratch(&ctx.device, plan_bytes)?, live::scratch(&ctx.device, data_bytes)?));
    super::super::physical_directory::recordings::retain_native(list, std::sync::Arc::new((leases, storage.clone(), pipeline.clone(), ctx.views.clone(), source.resource, destination.resource)))?;
    live::with_helper(&ctx.list, &ctx.scope, || pipeline.record_copy(&ctx.list, &ctx.views, &copy, &storage.0, &storage.1).map_err(String::from))?;
    Ok(true)
}

pub(crate) struct Pipeline {
    root: ID3D12RootSignature,
    capture: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    scatter: ID3D12PipelineState,
}

#[derive(Clone, Copy)]
pub(crate) enum FallbackPage {
    Unmapped,
    Raw { view: u32, page_base: u32 },
}

pub(crate) struct Copy {
    steps: Vec<([u32; 8], [u32; 8])>,
    plan_bytes: u64,
    scratch_bytes: u64,
}

pub(crate) struct Scratch<'a> {
    plan: &'a ID3D12Resource,
    data: &'a ID3D12Resource,
}

impl Copy {
    pub(crate) fn prepare(
        range: Transfer,
        locator: u32,
        mut fallback: impl FnMut(u32, bool) -> Option<FallbackPage>,
    ) -> Option<Self> {
        if locator == 0 || locator >= 1 << 20 {
            return None;
        }
        let mut steps = Vec::new();
        let mut plan_bytes = 0u64;
        for p in range.pieces() {
            let mut constants = |vpn, offset, write| {
                let (view, base) = match fallback(vpn, write)? {
                    FallbackPage::Unmapped => (0, 0),
                    FallbackPage::Raw { view, page_base }
                        if view != 0
                            && view < 1 << 20
                            && page_base & 3 == 0
                            && page_base <= u32::MAX - 65535 =>
                    {
                        (view, page_base)
                    }
                    _ => return None,
                };
                Some([locator, vpn, view, base, 0, offset, 0, p.bytes])
            };
            steps.push((
                constants(p.source_page, p.source_offset, false)?,
                constants(p.destination_page, p.destination_offset, true)?,
            ));
            plan_bytes = plan_bytes.checked_add(16)?;
        }
        Some(Self {
            steps,
            plan_bytes,
            scratch_bytes: range.bytes.checked_add(3)? & !3,
        })
    }

    pub(crate) fn storage_bytes(&self) -> (u64, u64) {
        (self.plan_bytes, self.scratch_bytes)
    }

    pub(crate) fn read(
        address: u64,
        bytes: u64,
        locator: u32,
        mut fallback: impl FnMut(u32) -> Option<FallbackPage>,
    ) -> Option<Read> {
        let copy = Self::prepare(
            Transfer::new(address, address, bytes)?,
            locator,
            |page, write| {
                if write {
                    Some(FallbackPage::Unmapped)
                } else {
                    fallback(page)
                }
            },
        )?;
        Some(Read(copy))
    }

    /// The title range `address`..`address + bytes` as a transfer's destination: scratch bytes
    /// written to each page's owner (`fallback` for a page the directory has no entry for).
    pub(crate) fn write(
        address: u64,
        bytes: u64,
        locator: u32,
        mut fallback: impl FnMut(u32) -> Option<FallbackPage>,
    ) -> Option<Copy> {
        Self::prepare(Transfer::new(address, address, bytes)?, locator, |page, write| {
            if write {
                fallback(page)
            } else {
                Some(FallbackPage::Unmapped)
            }
        })
    }

    pub(crate) unsafe fn bind_storage<'a>(
        &self,
        plan: &'a ID3D12Resource,
        data: &'a ID3D12Resource,
    ) -> Option<Scratch<'a>> {
        let accepts = |r: &ID3D12Resource, bytes| {
            let d = r.GetDesc();
            d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                && d.Width >= bytes
                && d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
                && r.GetGPUVirtualAddress().checked_add(bytes).is_some()
                && r.GetGPUVirtualAddress() != 0
        };
        (plan.as_raw() != data.as_raw()
            && accepts(plan, self.plan_bytes)
            && accepts(data, self.scratch_bytes))
        .then_some(Scratch { plan, data })
    }
}

pub(crate) struct Read(Copy);

impl Read {
    pub(crate) fn storage_bytes(&self) -> (u64, u64) {
        self.0.storage_bytes()
    }
}

impl Pipeline {
    pub(crate) unsafe fn record_read(
        &self,
        list: &ID3D12GraphicsCommandList,
        views: &ID3D12DescriptorHeap,
        read: &Read,
        plan: &ID3D12Resource,
        data: &ID3D12Resource,
    ) -> Result<(), &'static str> {
        if read.0.steps.is_empty() {
            return Ok(());
        }
        if !matches!(
            list.GetType(),
            D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE
        ) {
            return Err("a native read requires a direct or compute command list");
        }
        let buffers = read
            .0
            .bind_storage(plan, data)
            .ok_or("native read storage does not fit")?;
        self.record_phase(list, views, &read.0, &buffers, Phase::CaptureSource);
        self.record_phase(list, views, &read.0, &buffers, Phase::Gather);
        Ok(())
    }
    pub(crate) unsafe fn new(dev: &ID3D12Device) -> Result<Self, String> {
        let source = [
            crate::gpu::recomp::GPU_PAGE_DIRECTORY_HLSL,
            include_str!("transfer.hlsl"),
        ]
        .concat();
        let compile = |entry| crate::gpu::recomp::dxc::compile(&source, entry, "cs_6_6", &[]);
        let code = compile("Capture")?;
        let root: ID3D12RootSignature = dev
            .CreateRootSignature(0, &code)
            .map_err(|e| e.to_string())?;
        let create = |code: &[u8]| {
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let made = dev
                .CreateComputePipelineState(&desc)
                .map_err(|e| e.to_string());
            ManuallyDrop::drop(&mut desc.pRootSignature);
            made
        };
        let capture = create(&code)?;
        let gather = create(&compile("Gather")?)?;
        let scatter = create(&compile("Scatter")?)?;
        Ok(Self {
            root,
            capture,
            gather,
            scatter,
        })
    }

    pub(crate) unsafe fn record_copy(
        &self,
        list: &ID3D12GraphicsCommandList,
        views: &ID3D12DescriptorHeap,
        copy: &Copy,
        plan: &ID3D12Resource,
        scratch: &ID3D12Resource,
    ) -> Result<(), &'static str> {
        if copy.steps.is_empty() {
            return Ok(());
        }
        if !matches!(
            list.GetType(),
            D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE
        ) {
            return Err("native page transfers require a direct or compute command list");
        }
        let buffers = copy
            .bind_storage(plan, scratch)
            .ok_or("native page transfer storage does not fit")?;
        self.record_phase(list, views, copy, &buffers, Phase::Capture);
        self.record_phase(list, views, copy, &buffers, Phase::Gather);
        self.record_phase(list, views, copy, &buffers, Phase::Scatter);
        Ok(())
    }

    /// `copy` (made by `Copy::write`) from scratch bytes already in `scratch` to its destination.
    pub(crate) unsafe fn record_write(
        &self,
        list: &ID3D12GraphicsCommandList,
        views: &ID3D12DescriptorHeap,
        copy: &Copy,
        plan: &ID3D12Resource,
        scratch: &ID3D12Resource,
    ) -> Result<(), &'static str> {
        if copy.steps.is_empty() {
            return Ok(());
        }
        if !matches!(list.GetType(), D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
            return Err("native page transfers require a direct or compute command list");
        }
        let buffers = copy.bind_storage(plan, scratch).ok_or("native page transfer storage does not fit")?;
        self.record_phase(list, views, copy, &buffers, Phase::Capture);
        self.record_phase(list, views, copy, &buffers, Phase::Scatter);
        Ok(())
    }

    pub(crate) unsafe fn record_phase(
        &self,
        list: &ID3D12GraphicsCommandList,
        views: &ID3D12DescriptorHeap,
        copy: &Copy,
        buffers: &Scratch<'_>,
        pass: Phase,
    ) {
        let (plan_va, data_va) = (
            buffers.plan.GetGPUVirtualAddress(),
            buffers.data.GetGPUVirtualAddress(),
        );
        list.SetDescriptorHeaps(&[Some(views.clone())]);
        list.SetComputeRootSignature(&self.root);
        let barrier = || {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }])
        };
        barrier();
        {
            let pso = match pass {
                Phase::Capture | Phase::CaptureSource => &self.capture,
                Phase::Gather => &self.gather,
                Phase::Scatter => &self.scatter,
            };
            live::set_pipeline(list, pso);
            let mut offset = 0u64;
            for (i, &(s, d)) in copy.steps.iter().enumerate() {
                let data_offset = offset & !15;
                list.SetComputeRootUnorderedAccessView(2, data_va + data_offset);
                let record = |mut c: [u32; 8], is_dst: bool| {
                    list.SetComputeRootUnorderedAccessView(1, plan_va + 16 * i as u64);
                    c[4] = if is_dst { 8 } else { 0 };
                    c[6] = (offset - data_offset) as u32;
                    list.SetComputeRoot32BitConstants(0, 8, c.as_ptr() as *const c_void, 0);
                    let displacement = if pass == Phase::Gather {
                        c[6] & 3
                    } else {
                        c[5] & 3
                    };
                    live::dispatch(
                        list,
                        if matches!(pass, Phase::Capture | Phase::CaptureSource) {
                            1
                        } else {
                            (c[7] + displacement).div_ceil(256)
                        },
                        1,
                        1,
                    );
                };
                if pass != Phase::Scatter {
                    record(s, false);
                }
                if matches!(pass, Phase::Scatter | Phase::Capture) {
                    record(d, true);
                }
                if pass == Phase::Scatter {
                    barrier();
                }
                offset += u64::from(s[7]);
            }
            barrier();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Capture,
    CaptureSource,
    Gather,
    Scatter,
}
