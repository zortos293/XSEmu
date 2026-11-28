//! One execution-ready native operation on one reserved queue lane. The scheduler
//! must not overtake a waiting operation with later commands on that lane.
use super::commands::{state_ok, CapturedPages, Directory, Kernels, OwnerReadback, Plane, Recorder, States};
use super::ownership::{CapturedTopology, Location, Ownership, PayloadEvidence, Pending, Request, WritePermit};
use super::{GuestImage, ImageUse, Plan};
use crate::gpu::physical_binding::PinnedPage;
use std::collections::BTreeMap;
use std::sync::Arc;
use windows::core::{IUnknown, Interface};
use windows::Win32::Graphics::Direct3D12::*;

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Access {
    Read,
    PartialWrite,
    FullWrite,
    Discard,
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Operand {
    pub image: GuestImage,
    pub native: ID3D12Resource,
    pub guest_plane: u32,
    pub sub: u32,
    pub role: ImageUse,
    pub access: Access,
    /// Native subresource texels, exclusive right/bottom/back. None means the whole subresource.
    pub region: Option<D3D12_BOX>,
    /// Required state on entering the native chunk and its actual state on exit.
    pub states: States,
    pub evidence: PayloadEvidence,
}

/// Retains every binding and temporary referenced by the native operation.
/// Record the one original operation on the provided fresh list, never submit
/// here. Resolve only against the retained descriptor/run snapshot.
// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) trait NativeOperation: Send + Sync {
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList) -> Result<()>;
    fn pin(&self, image: &GuestImage, vpn: u32, location: Location) -> std::result::Result<Arc<PinnedPage>, &'static str>;
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Epoch {
    pub directory: Directory,
    pub heap: ID3D12DescriptorHeap,
    pub version: Arc<dyn Send + Sync>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Progress {
    WaitingCapture,
    WaitingOwnership,
    WaitingNative,
    WaitingRetire,
    Complete,
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
struct Use {
    states: States,
    evidence: PayloadEvidence,
    gather: bool,
    publish: bool,
    discard: bool,
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
struct Group {
    plane: Plane,
    capture: usize,
    uses: BTreeMap<u32, Use>,
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
struct Capture {
    image: GuestImage,
    pages: CapturedPages,
    readback: OwnerReadback<'static>,
    topology: Option<CapturedTopology>,
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
fn same_image(a: &GuestImage, b: &GuestImage) -> bool {
    a.va == b.va && a.desc == b.desc && a.host == b.host && a.initial == b.initial
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
fn describe_use(op: &Operand, desc: D3D12_RESOURCE_DESC, kind: D3D12_COMMAND_LIST_TYPE) -> Result<Use> {
    if !op.image.permits(op.role) || (matches!(op.role, ImageUse::Sample) && op.access != Access::Read) {
        return Err("guest image does not permit native operation role".into());
    }
    if !state_ok(kind, op.states.before) || !state_ok(kind, op.states.after) {
        return Err("unsupported native operation state for this queue".into());
    }
    let mip = op.sub % u32::from(desc.MipLevels);
    let w = desc.Width.checked_shr(mip).unwrap_or(0).max(1);
    let h = desc.Height.checked_shr(mip).unwrap_or(0).max(1);
    let d = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(desc.DepthOrArraySize).checked_shr(mip).unwrap_or(0).max(1)
    } else {
        1
    };
    let whole = match op.region {
        None => true,
        Some(b) => {
            if b.left >= b.right || b.top >= b.bottom || b.front >= b.back || u64::from(b.right) > w || b.bottom > h || b.back > d {
                return Err("empty or out-of-range native operation extent".into());
            }
            (b.left, b.top, b.front, u64::from(b.right), b.bottom, b.back) == (0, 0, 0, w, h, d)
        }
    };
    if matches!(op.access, Access::FullWrite | Access::Discard) && !whole {
        return Err("full overwrite or discard requires the entire native subresource".into());
    }
    if matches!(op.evidence, PayloadEvidence::MetadataFreeFullTypedOverwriteFromBinary16) && op.access != Access::FullWrite {
        return Err("typed overwrite evidence requires a full write".into());
    }
    Ok(Use {
        states: op.states,
        evidence: op.evidence,
        gather: matches!(op.access, Access::Read | Access::PartialWrite),
        publish: matches!(op.access, Access::PartialWrite | Access::FullWrite),
        discard: op.access == Access::Discard,
    })
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
enum Stage {
    Start,
    Capture,
    Admit,
    Native,
    Retire,
    Failed(String),
}

// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
struct Inner {
    stage: Stage,
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    kind: D3D12_COMMAND_LIST_TYPE,
    allocator: ID3D12CommandAllocator,
    kernels: Arc<Kernels>,
    epoch: Epoch,
    op: Box<dyn NativeOperation>,
    native: Option<ID3D12CommandList>,
    groups: Vec<Group>,
    captures: Vec<Capture>,
    requests: Vec<Request>,
    permit: Option<WritePermit>,
    native_done: ID3D12Fence,
    retirement: Option<Pending>,
    in_flight: bool,
}

#[must_use]
// The native-operation lifecycle is not scheduled yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Operation(Option<Box<Inner>>);

impl Operation {
    /// Call at actual execution readiness. The epoch and all raw requests must
    /// describe this same reserved queue position, not guest recording time.
    // The native-operation lifecycle is not scheduled yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn new(
        queue: &ID3D12CommandQueue,
        epoch: Epoch,
        kernels: Arc<Kernels>,
        operands: Vec<Operand>,
        raw: Vec<Request>,
        op: Box<dyn NativeOperation>,
    ) -> Result<Self> {
        if operands.is_empty() {
            return Err("empty native image operation".into());
        }
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev).map_err(|e| e.to_string())?;
        let dev = dev.ok_or("queue without a device")?;
        let kind = queue.GetDesc().Type;
        if !matches!(kind, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) {
            return Err("operation needs an executable direct or compute queue".into());
        }
        let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(kind).map_err(|e| e.to_string())?;
        let mut groups: Vec<Group> = Vec::new();
        let mut captures: Vec<Capture> = Vec::new();
        let mut by_plane = BTreeMap::new();
        let mut by_image = BTreeMap::new();
        let mut sub_planes = BTreeMap::new();
        for operand in operands {
            let id = operand.native.cast::<IUnknown>().map_err(|e| e.to_string())?.as_raw() as usize;
            let desc = operand.native.GetDesc();
            if desc.MipLevels == 0 {
                return Err("native image has no mip levels".into());
            }
            let use_ = describe_use(&operand, desc, kind)?;
            if sub_planes
                .insert((id, operand.sub), operand.guest_plane)
                .is_some_and(|p| p != operand.guest_plane)
            {
                return Err("two guest planes target the same native subresource".into());
            }
            let tagged = GuestImage::get(&operand.native).map_err(|e| e.to_string())?;
            if !same_image(&tagged, &operand.image) {
                return Err("pinned guest image does not match native placement".into());
            }
            let i = match by_plane.get(&(id, operand.guest_plane)).copied() {
                Some(i) => i,
                None => {
                    let plan = Plan::expanded_plane(&dev, &operand.native, operand.image, operand.guest_plane)
                        .map_err(|e| format!("unsupported native image plane: {e:?}"))?;
                    let plane = Plane::new(&dev, &operand.native, plan)?;
                    let capture = match by_image.get(&id).copied() {
                        Some(i) => i,
                        None => {
                            let rec = Recorder::begin(&dev, &allocator, kind, &epoch.heap, &kernels)?;
                            let pages = rec.capture(&plane, &epoch.directory)?;
                            let readback = rec.read_owners(&pages)?;
                            let i = captures.len();
                            captures.push(Capture {
                                image: operand.image,
                                pages,
                                readback,
                                topology: None,
                            });
                            by_image.insert(id, i);
                            i
                        }
                    };
                    let i = groups.len();
                    by_plane.insert((id, operand.guest_plane), i);
                    groups.push(Group {
                        plane,
                        capture,
                        uses: BTreeMap::new(),
                    });
                    i
                }
            };
            let group = &mut groups[i];
            group.plane.index(operand.sub)?;
            if !same_image(&captures[group.capture].image, &operand.image) {
                return Err("conflicting guest placement in operation".into());
            }
            match group.uses.entry(operand.sub) {
                std::collections::btree_map::Entry::Vacant(v) => {
                    v.insert(use_);
                }
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    let prior = e.get_mut();
                    if prior.discard != use_.discard
                        || prior.states.before != use_.states.before
                        || prior.states.after != use_.states.after
                        || std::mem::discriminant(&prior.evidence) != std::mem::discriminant(&use_.evidence)
                    {
                        return Err("contradictory uses of one native subresource".into());
                    }
                    prior.gather |= use_.gather;
                    prior.publish |= use_.publish;
                }
            }
        }
        let native: ID3D12GraphicsCommandList = dev.CreateCommandList(0, kind, &allocator, None).map_err(|e| e.to_string())?;
        op.record(&native)?;
        native.Close().map_err(|e| format!("native operation failed to close: {e}"))?;
        let native = native.cast().map_err(|e| e.to_string())?;
        let native_done = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
        Ok(Self(Some(Box::new(Inner {
            stage: Stage::Start,
            dev,
            queue: queue.clone(),
            kind,
            allocator,
            kernels,
            epoch,
            op,
            native: Some(native),
            groups,
            captures,
            requests: raw,
            permit: None,
            native_done,
            retirement: None,
            in_flight: false,
        }))))
    }

    // The native-operation lifecycle is not scheduled yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn poll(&mut self) -> Result<Progress> {
        let Some(s) = self.0.as_mut() else {
            return Ok(Progress::Complete);
        };
        if let Stage::Failed(e) = &s.stage {
            return Err(e.clone());
        }
        let result = s.advance();
        if let Err(e) = &result {
            s.stage = Stage::Failed(e.clone());
        }
        if matches!(result, Ok(Progress::Complete)) {
            self.0.take();
        }
        result
    }
}

impl Inner {
    // The native-operation lifecycle is not scheduled yet; kept for the next fix.
    #[allow(dead_code)]
    unsafe fn advance(&mut self) -> Result<Progress> {
        match self.stage {
            Stage::Start => {
                self.in_flight = true;
                for capture in &mut self.captures {
                    capture.readback.submit(&self.queue)?;
                }
                self.stage = Stage::Capture;
                Ok(Progress::WaitingCapture)
            }
            Stage::Capture => {
                for c in &mut self.captures {
                    if c.topology.is_none() {
                        c.topology = c.readback.try_topology(|vpn, loc| self.op.pin(&c.image, vpn, loc))?;
                        if c.topology.is_none() {
                            return Ok(Progress::WaitingCapture);
                        }
                    }
                }
                self.in_flight = false;
                for g in &self.groups {
                    let topology = self.captures[g.capture].topology.as_ref().unwrap();
                    for (&sub, u) in &g.uses {
                        self.requests.push(g.plane.request(topology, sub, u.publish || u.discard, u.evidence)?);
                    }
                }
                self.stage = Stage::Admit;
                Ok(Progress::WaitingOwnership)
            }
            Stage::Admit => {
                let Some(permit) = Ownership::shared().try_admit(&self.requests).map_err(str::to_owned)? else {
                    return Ok(Progress::WaitingOwnership);
                };
                self.permit = Some(permit);
                let mut rec = Recorder::begin(&self.dev, &self.allocator, self.kind, &self.epoch.heap, &self.kernels)?;
                for g in &self.groups {
                    let reads: Vec<_> = g
                        .uses
                        .iter()
                        .filter(|(_, u)| u.gather)
                        .map(|(&sub, u)| {
                            (
                                sub,
                                States {
                                    before: u.states.before,
                                    after: u.states.before,
                                },
                            )
                        })
                        .collect();
                    rec.materialize(&g.plane, &self.captures[g.capture].pages, self.permit.as_ref().unwrap(), &reads)?;
                }
                self.in_flight = true;
                rec.finish()?
                    .execute_before(&self.queue, self.native.take().expect("native operation submitted once"))?;
                self.queue.Signal(&self.native_done, 1).map_err(|e| format!("native operation signal: {e}"))?;
                self.stage = Stage::Native;
                Ok(Progress::WaitingNative)
            }
            Stage::Native => {
                self.dev.GetDeviceRemovedReason().map_err(|e| e.to_string())?;
                match self.native_done.GetCompletedValue() {
                    0 => return Ok(Progress::WaitingNative),
                    1 => (),
                    _ => return Err("native operation did not reach a valid completion".into()),
                }
                let mut rec = Recorder::begin(&self.dev, &self.allocator, self.kind, &self.epoch.heap, &self.kernels)?;
                for g in &self.groups {
                    let writes: Vec<_> = g
                        .uses
                        .iter()
                        .filter(|(_, u)| u.publish)
                        .map(|(&sub, u)| {
                            (
                                sub,
                                States {
                                    before: u.states.after,
                                    after: u.states.after,
                                },
                            )
                        })
                        .collect();
                    rec.publish(&g.plane, &self.captures[g.capture].pages, self.permit.as_ref().unwrap(), &writes)?;
                }
                rec.finish()?.execute(&self.queue)?;
                self.retirement = Some(self.permit.take().unwrap().signal_after_last_use(std::slice::from_ref(&self.queue))?);
                self.stage = Stage::Retire;
                Ok(Progress::WaitingRetire)
            }
            Stage::Retire => {
                self.dev.GetDeviceRemovedReason().map_err(|e| e.to_string())?;
                if !self.retirement.as_mut().unwrap().retire_if_complete().map_err(str::to_owned)? {
                    return Ok(Progress::WaitingRetire);
                }
                self.in_flight = false;
                Ok(Progress::Complete)
            }
            Stage::Failed(ref e) => Err(e.clone()),
        }
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        let Some(s) = self.0.take() else {
            return;
        };
        if s.in_flight && unsafe { s.dev.GetDeviceRemovedReason().is_ok() } {
            // No retirement proof exists. Never free backing still reachable through a submitted list.
            std::mem::forget(s);
        }
    }
}
