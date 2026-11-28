//! GPU commands for an isolated list between native chunks, never inside a render pass.
//! The submitter fences the chunks in order, keeps all resources, heaps, PSOs and
//! allocators alive, and rebinds guest state when it resumes the native list.

use super::{
    ownership::{CapturedTopology, Location, PayloadEvidence, Request, WritePermit},
    Plan, Subresource,
};
use crate::gpu::physical_binding::PinnedPage;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use windows::core::{IUnknown, Interface};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
type Result<T> = std::result::Result<T, String>;
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
const COMMON: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COMMON;
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
const UAV: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
const SRC: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_SOURCE;
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
const DST: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_DEST;

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
fn unique_id() -> Result<u64> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "image bridge identity exhausted".into())
}

/// The three real SM 6.6 PSOs use one directly indexed root signature:
/// eight constants b0, raw SRV t0, raw UAVs u0 and u1 (all root descriptors).
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Kernels {
    pub signature: ID3D12RootSignature,
    pub capture: ID3D12PipelineState,
    pub gather: ID3D12PipelineState,
    pub scatter: ID3D12PipelineState,
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Directory {
    pub root_srv: u32,
    pub uma_base_hi: u32,
    pub uma_first_uav: u32,
    pub windows: u32,
}

impl Directory {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    fn valid(&self, heap_len: u32) -> bool {
        (1..heap_len).contains(&self.root_srv)
            && (1..=4).contains(&self.windows)
            && self
                .uma_first_uav
                .checked_add(4)
                .is_some_and(|miss| miss < heap_len && !(self.uma_first_uav..=miss).contains(&self.root_srv))
    }
}

/// Entry/exit states for one selected native subresource.
#[derive(Clone, Copy)]
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct States {
    pub before: D3D12_RESOURCE_STATES,
    pub after: D3D12_RESOURCE_STATES,
}

/// The actual (descriptor, page offset, token) records live on the GPU. The
/// submitter must pin the referenced physical runs and immutable descriptors
/// through the final gather/scatter fence, including work on other queues.
#[derive(Clone)]
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct CapturedPages {
    id: u64,
    recorder: u64,
    native: ID3D12Resource,
    buffer: ID3D12Resource,
    heap: ID3D12DescriptorHeap,
    first: u32,
    count: u32,
}

impl CapturedPages {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn capture_id(&self) -> u64 {
        self.id
    }
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Plane {
    native: ID3D12Resource,
    native_id: usize,
    plan: Plan,
    params: ID3D12Resource,
    footprint: ID3D12Resource,
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn buffer(dev: &ID3D12Device, width: u64, heap: D3D12_HEAP_TYPE) -> Result<ID3D12Resource> {
    let state = match heap {
        D3D12_HEAP_TYPE_UPLOAD => D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_HEAP_TYPE_DEFAULT => COMMON,
        D3D12_HEAP_TYPE_READBACK => DST,
        _ => return Err("unsupported image bridge buffer heap".into()),
    };
    let properties = D3D12_HEAP_PROPERTIES {
        Type: heap,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if heap == D3D12_HEAP_TYPE_DEFAULT {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut out = None;
    dev.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
        .map_err(|e| format!("image bridge buffer ({width} B): {e}"))?;
    out.ok_or_else(|| "image bridge buffer creation returned no resource".into())
}

impl Plane {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(super) fn index(&self, sub: u32) -> Result<usize> {
        self.plan
            .subs
            .binary_search_by_key(&sub, |entry| entry.sub)
            .map_err(|_| "native subresource outside this plane".into())
    }

    /// COM identity of the actual native texture, not a physical page identity.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn native_id(&self) -> usize {
        self.native_id
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn request(&self, topology: &CapturedTopology, sub: u32, write: bool, evidence: PayloadEvidence) -> Result<Request> {
        if u64::from(topology.first_vpn()) != self.plan.page_base >> 16 || topology.len() != self.plan.page_count as usize {
            return Err("captured topology does not cover this native image".into());
        }
        let i = self.index(sub)?;
        topology
            .image_request(self.native_id, sub, &self.plan.subs[i].words, write, evidence, self.plan.host)
            .map_err(str::to_owned)
    }

    /// `plan` must have been created by Plan::expanded_plane for this exact texture.
    /// Fence or serialize reuse of this plane's private footprint.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn new(dev: &ID3D12Device, native: &ID3D12Resource, plan: Plan) -> Result<Self> {
        let data_bytes = plan.subs.len().checked_mul(size_of::<[u32; 68]>()).ok_or("image plan too large")?;
        if data_bytes == 0 || plan.page_count == 0 {
            return Err("empty image bridge plane".into());
        }
        let native_id = native.cast::<IUnknown>().map_err(|e| e.to_string())?.as_raw() as usize;
        let params = buffer(dev, data_bytes as u64, D3D12_HEAP_TYPE_UPLOAD)?;
        let footprint = buffer(dev, plan.footprint_bytes, D3D12_HEAP_TYPE_DEFAULT)?;
        let mut mapped = std::ptr::null_mut();
        params.Map(0, Some(&D3D12_RANGE::default()), Some(&mut mapped)).map_err(|e| e.to_string())?;
        for (i, sub) in plan.subs.iter().enumerate() {
            std::ptr::copy_nonoverlapping(sub.words.as_ptr(), mapped.cast::<u32>().add(i * 68), 68);
        }
        params.Unmap(0, None);
        Ok(Self {
            native: native.clone(),
            native_id,
            plan,
            params,
            footprint,
        })
    }
}

/// Only creates fresh lists. The open list isn't exposed to native recorders.
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Recorder<'a, 'grant> {
    grants: PhantomData<&'grant WritePermit>,
    id: u64,
    list: ID3D12GraphicsCommandList,
    allocator: ID3D12CommandAllocator,
    kind: D3D12_COMMAND_LIST_TYPE,
    dev: &'a ID3D12Device,
    heap: &'a ID3D12DescriptorHeap,
    kernels: &'a Kernels,
}

impl<'a, 'grant> Recorder<'a, 'grant> {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn begin(
        dev: &'a ID3D12Device,
        allocator: &ID3D12CommandAllocator,
        kind: D3D12_COMMAND_LIST_TYPE,
        heap: &'a ID3D12DescriptorHeap,
        kernels: &'a Kernels,
    ) -> Result<Self> {
        if !matches!(kind, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
            return Err("image bridge needs a dedicated DIRECT or COMPUTE list".into());
        }
        let desc = heap.GetDesc();
        if desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !desc.Flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
            return Err("image bridge needs a shader-visible canonical resource heap".into());
        }
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, kind, allocator, None)
            .map_err(|e| format!("image bridge command list: {e}"))?;
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(&kernels.signature);
        Ok(Self {
            id: unique_id()?,
            grants: PhantomData,
            list,
            allocator: allocator.clone(),
            kind,
            dev,
            heap,
            kernels,
        })
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn finish(self) -> Result<Recorded<'grant>> {
        self.list.Close().map_err(|e| format!("closing image bridge list: {e}"))?;
        Ok(Recorded {
            list: self.list.cast().map_err(|e| e.to_string())?,
            _allocator: self.allocator,
            dev: self.dev.clone(),
            kind: self.kind,
            _grants: PhantomData,
        })
    }

    /// Record at the consumer or producer boundary, after all PTE updates.
    /// The directory and leaves must be shader-readable, and canonical raw
    /// page buffers UAV-accessible; queue waits and their states belong to the submitter.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn capture(&self, plane: &Plane, dir: &Directory) -> Result<CapturedPages> {
        let plan = &plane.plan;
        if !dir.valid(self.heap.GetDesc().NumDescriptors) {
            return Err("image bridge requires a page directory and 1..=4 canonical UMA windows with a miss slot".into());
        }
        let pages = CapturedPages {
            id: unique_id()?,
            recorder: self.id,
            native: plane.native.clone(),
            buffer: buffer(self.dev, u64::from(plan.page_count) * 16, D3D12_HEAP_TYPE_DEFAULT)?,
            heap: self.heap.clone(),
            first: (plan.page_base >> 16) as u32,
            count: plan.page_count,
        };
        transition(&self.list, &pages.buffer, ALL, COMMON, UAV);
        self.list.SetPipelineState(&self.kernels.capture);
        self.bind(plane, &pages, 0);
        self.dispatch(
            [pages.first, pages.count, dir.root_srv, dir.uma_base_hi, dir.uma_first_uav, 0, dir.windows, 0],
            [pages.count, 1, 1],
        );
        transition(&self.list, &pages.buffer, ALL, UAV, COMMON);
        Ok(pages)
    }

    /// Seal this capture list with a copy of its GPU-produced owner records.
    /// Requiring the original recorder prevents certifying an unsubmitted capture.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn read_owners(self, pages: &CapturedPages) -> Result<OwnerReadback<'grant>> {
        if self.id != pages.recorder {
            return Err("owner readback must seal the list that captured the pages".into());
        }
        let bytes = u64::from(pages.count) * 16;
        let dst = buffer(self.dev, bytes, D3D12_HEAP_TYPE_READBACK)?;
        let fence = self.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
        transition(&self.list, &pages.buffer, ALL, COMMON, SRC);
        self.list.CopyBufferRegion(&dst, 0, &pages.buffer, 0, bytes);
        transition(&self.list, &pages.buffer, ALL, SRC, COMMON);
        Ok(OwnerReadback {
            list: self.finish()?,
            pages: pages.clone(),
            dst,
            fence,
            state: ReadbackState::Recorded,
        })
    }

    /// Seed from the current native owner, then replace mapped bytes and
    /// explicit holes. An ordinary absent page outside UMA retains native data.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn materialize(
        &mut self,
        plane: &Plane,
        pages: &CapturedPages,
        permit: &'grant WritePermit,
        native_subs: &[(u32, States)],
    ) -> Result<()> {
        if native_subs.is_empty() {
            return Ok(());
        }
        let selected = self.select(plane, pages, native_subs)?;
        if !native_subs.iter().all(|&(sub, _)| permit.can_read(pages.capture_id(), plane.native_id(), sub)) {
            return Err("image bridge permit does not certify every native read for this capture".into());
        }
        transition(&self.list, &plane.footprint, ALL, COMMON, DST);
        for &(i, s) in &selected {
            let sub = &plane.plan.subs[i];
            transition(&self.list, &plane.native, sub.sub, s.before, SRC);
            copy(&self.list, &plane.native, &plane.footprint, sub, false);
        }
        transition(&self.list, &plane.footprint, ALL, DST, UAV);
        transition(&self.list, &pages.buffer, ALL, COMMON, UAV);
        uav_order(&self.list);
        self.run_plane(plane, pages, &selected, false);
        uav_order(&self.list);
        transition(&self.list, &pages.buffer, ALL, UAV, COMMON);
        transition(&self.list, &plane.footprint, ALL, UAV, SRC);
        for &(i, s) in &selected {
            let sub = &plane.plan.subs[i];
            transition(&self.list, &plane.native, sub.sub, SRC, DST);
            copy(&self.list, &plane.native, &plane.footprint, sub, true);
            transition(&self.list, &plane.native, sub.sub, DST, s.after);
        }
        transition(&self.list, &plane.footprint, ALL, SRC, COMMON);
        Ok(())
    }

    /// Publish native writes to the captured physical owners, without reading
    /// the current directory. Each dirty native subresource must own all its
    /// mapped bytes exclusively until publication. This is not a partial
    /// write mask and cannot merge unordered writes through a raw alias.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn publish(
        &mut self,
        plane: &Plane,
        pages: &CapturedPages,
        permit: &'grant WritePermit,
        dirty_native_subs: &[(u32, States)],
    ) -> Result<()> {
        if dirty_native_subs.is_empty() {
            return Ok(());
        }
        let selected = self.select(plane, pages, dirty_native_subs)?;
        if !dirty_native_subs
            .iter()
            .all(|&(sub, _)| permit.allows(pages.capture_id(), plane.native_id(), sub))
        {
            return Err("image bridge write permit does not certify every dirty subresource for this capture and native texture".into());
        }
        transition(&self.list, &plane.footprint, ALL, COMMON, DST);
        for &(i, s) in &selected {
            let sub = &plane.plan.subs[i];
            transition(&self.list, &plane.native, sub.sub, s.before, SRC);
            copy(&self.list, &plane.native, &plane.footprint, sub, false);
            transition(&self.list, &plane.native, sub.sub, SRC, s.after);
        }
        transition(&self.list, &plane.footprint, ALL, DST, UAV);
        transition(&self.list, &pages.buffer, ALL, COMMON, UAV);
        uav_order(&self.list);
        self.run_plane(plane, pages, &selected, true);
        uav_order(&self.list);
        transition(&self.list, &pages.buffer, ALL, UAV, COMMON);
        transition(&self.list, &plane.footprint, ALL, UAV, COMMON);
        Ok(())
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    fn select(&self, plane: &Plane, pages: &CapturedPages, native_subs: &[(u32, States)]) -> Result<Vec<(usize, States)>> {
        if pages.native != plane.native
            || pages.heap != *self.heap
            || u64::from(pages.first) != plane.plan.page_base >> 16
            || pages.count != plane.plan.page_count
        {
            return Err("captured owners don't cover this image in this descriptor heap".into());
        }
        select(&plane.plan.subs, self.kind, native_subs)
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    unsafe fn bind(&self, plane: &Plane, pages: &CapturedPages, sub: usize) {
        self.list
            .SetComputeRootShaderResourceView(1, plane.params.GetGPUVirtualAddress() + (sub * size_of::<[u32; 68]>()) as u64);
        self.list.SetComputeRootUnorderedAccessView(2, pages.buffer.GetGPUVirtualAddress());
        self.list.SetComputeRootUnorderedAccessView(3, plane.footprint.GetGPUVirtualAddress());
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    unsafe fn dispatch(&self, params: [u32; 8], dims: [u32; 3]) {
        tiles(dims, |base, groups| {
            let mut constants = params;
            for i in 0..3 {
                constants[5 + i] += base[i];
            }
            self.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            self.list.Dispatch(groups[0], groups[1], groups[2]);
        });
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    unsafe fn run_plane(&self, plane: &Plane, pages: &CapturedPages, selected: &[(usize, States)], scatter: bool) {
        self.list.SetPipelineState(if scatter { &self.kernels.scatter } else { &self.kernels.gather });
        for (n, &(i, _)) in selected.iter().enumerate() {
            let sub = &plane.plan.subs[i];
            if scatter && n != 0 {
                uav_order(&self.list);
            }
            self.bind(plane, pages, i);
            self.dispatch([pages.first, pages.count, 0, 0, 0, 0, 0, 0], [sub.words[0], sub.words[1], sub.words[2]]);
        }
    }
}

/// No raw list escapes. A recorded capture or certified scatter can execute
/// only once; replay would resolve a newer directory or reuse a retired permit.
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Recorded<'a> {
    list: ID3D12CommandList,
    _allocator: ID3D12CommandAllocator,
    dev: ID3D12Device,
    kind: D3D12_COMMAND_LIST_TYPE,
    _grants: PhantomData<&'a WritePermit>,
}

impl Recorded<'_> {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    unsafe fn enqueue(&self, queue: &ID3D12CommandQueue, native: Option<ID3D12CommandList>) -> Result<()> {
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev).map_err(|e| e.to_string())?;
        if queue.GetDesc().Type != self.kind || dev.as_ref() != Some(&self.dev) {
            return Err("image bridge list and queue do not match".into());
        }
        self.dev.GetDeviceRemovedReason().map_err(|e| e.to_string())?;
        let lists = [Some(self.list.clone()), native];
        queue.ExecuteCommandLists(&lists[..if lists[1].is_some() { 2 } else { 1 }]);
        Ok(())
    }

    /// Caller retains the allocator, resources, descriptor pins and access
    /// grants through its own successful queue signal and fence completion.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn execute(self, queue: &ID3D12CommandQueue) -> Result<()> {
        self.enqueue(queue, None)
    }

    /// Submit gather and the native consumer in one ExecuteCommandLists so
    /// promoted resources do not decay between them.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(super) unsafe fn execute_before(self, queue: &ID3D12CommandQueue, native: ID3D12CommandList) -> Result<()> {
        self.enqueue(queue, Some(native))
    }
}

#[derive(PartialEq)]
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
enum ReadbackState {
    Recorded,
    Unconfirmed,
    Signalled,
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct OwnerReadback<'a> {
    list: Recorded<'a>,
    pages: CapturedPages,
    dst: ID3D12Resource,
    fence: ID3D12Fence,
    state: ReadbackState,
}

/// Only constructed from a completed owner readback. This is resolver output,
/// not a PFN declaration; the backend must join it to pinned run metadata.
// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct OwnerRecords {
    pages: CapturedPages,
    records: Vec<[u32; 4]>,
}

impl OwnerRecords {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn capture_id(&self) -> u64 {
        self.pages.capture_id()
    }
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn first_vpn(&self) -> u32 {
        self.pages.first
    }
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn records(&self) -> &[[u32; 4]] {
        &self.records
    }
}

impl OwnerReadback<'_> {
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn try_topology(
        &self,
        resolve: impl FnMut(u32, Location) -> std::result::Result<Arc<PinnedPage>, &'static str>,
    ) -> Result<Option<CapturedTopology>> {
        let Some(owners) = self.try_records()? else {
            return Ok(None);
        };
        CapturedTopology::resolve(owners.capture_id(), owners.first_vpn(), owners.records(), resolve)
            .map(Some)
            .map_err(str::to_owned)
    }

    /// Keep the handle and all other submission resources until completion.
    /// On a signal failure the queue may have accepted the list: retain it
    /// through an independent retirement proof or device teardown, never replay.
    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn submit(&mut self, queue: &ID3D12CommandQueue) -> Result<()> {
        if self.state != ReadbackState::Recorded {
            return Err("owner readback already submitted".into());
        }
        self.list.enqueue(queue, None)?;
        self.state = ReadbackState::Unconfirmed;
        queue.Signal(&self.fence, 1).map_err(|e| format!("signalling owner readback: {e}"))?;
        self.state = ReadbackState::Signalled;
        Ok(())
    }

    // Image-bridge command recording is not wired into the title path yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn try_records(&self) -> Result<Option<OwnerRecords>> {
        if self.state != ReadbackState::Signalled {
            return Err("owner readback has no confirmed queue signal".into());
        }
        self.list.dev.GetDeviceRemovedReason().map_err(|e| e.to_string())?;
        match self.fence.GetCompletedValue() {
            u64::MAX => return Err("device removed before owner readback completed".into()),
            0 => return Ok(None),
            1 => (),
            _ => return Err("unexpected private owner fence value".into()),
        }
        let count = self.pages.count as usize;
        let range = D3D12_RANGE {
            Begin: 0,
            End: count * size_of::<[u32; 4]>(),
        };
        let mut ptr = std::ptr::null_mut();
        self.dst
            .Map(0, Some(&range), Some(&mut ptr))
            .map_err(|e| format!("mapping captured owner records: {e}"))?;
        let records = std::slice::from_raw_parts(ptr.cast::<[u32; 4]>(), count).to_vec();
        self.dst.Unmap(0, Some(&D3D12_RANGE::default()));
        self.list.dev.GetDeviceRemovedReason().map_err(|e| e.to_string())?;
        Ok(Some(OwnerRecords {
            pages: self.pages.clone(),
            records,
        }))
    }
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
fn select(subs: &[Subresource], kind: D3D12_COMMAND_LIST_TYPE, native_subs: &[(u32, States)]) -> Result<Vec<(usize, States)>> {
    let mut selected = Vec::with_capacity(native_subs.len());
    for &(sub, s) in native_subs {
        let i = subs
            .binary_search_by_key(&sub, |entry| entry.sub)
            .map_err(|_| "native subresource outside this plane")?;
        if !state_ok(kind, s.before) || !state_ok(kind, s.after) {
            return Err("unsupported native subresource state on this queue".into());
        }
        selected.push((i, s));
    }
    selected.sort_unstable_by_key(|e| e.0);
    if selected.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("duplicate native subresource in image bridge operation".into());
    }
    Ok(selected)
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
fn tiles([w, h, d]: [u32; 3], mut emit: impl FnMut([u32; 3], [u32; 3])) {
    const MAX: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    for z in (0..d).step_by(MAX as usize) {
        for y in (0..h).step_by(MAX as usize) {
            for x in (0..w).step_by((MAX * 64) as usize) {
                emit([x, y, z], [(w - x).div_ceil(64).min(MAX), (h - y).min(MAX), (d - z).min(MAX)]);
            }
        }
    }
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
pub(super) fn state_ok(kind: D3D12_COMMAND_LIST_TYPE, s: D3D12_RESOURCE_STATES) -> bool {
    let direct = kind == D3D12_COMMAND_LIST_TYPE_DIRECT;
    if !direct && kind != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return false;
    }
    if matches!(s, DST | UAV)
        || direct
            && matches!(
                s,
                D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_RESOLVE_DEST
            )
    {
        return true;
    }
    let mut reads = SRC | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if direct {
        reads |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_DEPTH_READ
            | D3D12_RESOURCE_STATE_RESOLVE_SOURCE
            | D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
    }
    reads.contains(s)
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn uav_order(list: &ID3D12GraphicsCommandList) {
    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER::default()),
        },
        ..Default::default()
    }]);
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, sub: u32, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    if before == after {
        return;
    }
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(r.clone())),
                Subresource: sub,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    };
    list.ResourceBarrier(std::slice::from_ref(&b));
    ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

// Image-bridge command recording is not wired into the title path yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn copy(list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, scratch: &ID3D12Resource, sub: &Subresource, to_texture: bool) {
    let mut tex = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub.sub },
    };
    let mut buf = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(scratch.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: sub.footprint,
        },
    };
    let (dst, src) = if to_texture { (&tex, &buf) } else { (&buf, &tex) };
    list.CopyTextureRegion(dst, 0, 0, 0, src, None);
    ManuallyDrop::drop(&mut tex.pResource);
    ManuallyDrop::drop(&mut buf.pResource);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_indices_address_only_the_requested_native_plane_members() {
        let make = |sub| Subresource {
            sub,
            words: [0; 68],
            footprint: Default::default(),
        };
        let plane = [make(8), make(9), make(10), make(11)];
        let s = States { before: COMMON, after: SRC };
        let pick = |ids: &[u32]| {
            select(&plane, D3D12_COMMAND_LIST_TYPE_DIRECT, &ids.iter().map(|&i| (i, s)).collect::<Vec<_>>())
                .map(|v| v.into_iter().map(|(i, _)| i).collect::<Vec<_>>())
        };
        assert_eq!(pick(&[11, 9]).unwrap(), vec![1, 3]);
        assert!(pick(&[]).unwrap().is_empty());
        assert!(pick(&[9, 11, 9]).is_err());
        assert!(pick(&[7]).is_err());
        assert!(pick(&[12]).is_err());
    }

    #[test]
    fn tiles_preserve_origins_and_cover_partial_groups() {
        let width = 65_535 * 64 + 3;
        let mut calls = Vec::new();
        tiles([width, 65_536, 65_537], |base, count| calls.push((base, count)));
        assert_eq!(calls.len(), 8);
        for (i, (base, count)) in calls.into_iter().enumerate() {
            assert_eq!(base, [(i & 1) as u32 * 65_535 * 64, ((i >> 1) & 1) as u32 * 65_535, (i >> 2) as u32 * 65_535]);
            assert_eq!(
                count,
                [
                    if i & 1 == 0 { 65_535 } else { 1 },
                    if i & 2 == 0 { 65_535 } else { 1 },
                    if i & 4 == 0 { 65_535 } else { 2 }
                ]
            );
        }
    }

    #[test]
    fn compute_refuses_graphics_states_and_copy_cannot_dispatch() {
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        for state in [COMMON, SRC, DST, UAV, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE] {
            assert!(state_ok(D3D12_COMMAND_LIST_TYPE_COMPUTE, state));
            assert!(!state_ok(D3D12_COMMAND_LIST_TYPE_COPY, state));
        }
        for state in [read, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_RENDER_TARGET] {
            assert!(state_ok(D3D12_COMMAND_LIST_TYPE_DIRECT, state));
            assert!(!state_ok(D3D12_COMMAND_LIST_TYPE_COMPUTE, state));
        }
        assert!(!state_ok(D3D12_COMMAND_LIST_TYPE_DIRECT, DST | SRC));
    }

    #[test]
    fn directory_needs_an_explicit_root_and_room_for_the_miss_slot() {
        let mut dir = Directory {
            root_srv: 1,
            uma_base_hi: 5,
            uma_first_uav: 10,
            windows: 4,
        };
        assert!(dir.valid(15));
        assert!(!dir.valid(14));
        dir.windows = 5;
        assert!(!dir.valid(15));
        dir.windows = 1;
        dir.root_srv = 0;
        assert!(!dir.valid(15));
        dir.root_srv = 1;
        dir.uma_first_uav = u32::MAX;
        assert!(!dir.valid(u32::MAX));
        dir.uma_first_uav = 0;
        assert!(!dir.valid(16));
        dir.root_srv = 5;
        assert!(dir.valid(16));
    }

    #[test]
    fn identity_does_not_depend_on_reused_addresses() {
        let a = unique_id().unwrap();
        let b = unique_id().unwrap();
        assert!(a != 0 && b > a);
    }
}
