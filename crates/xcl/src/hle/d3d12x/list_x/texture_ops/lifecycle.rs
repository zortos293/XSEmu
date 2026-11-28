use super::{Layout, Location, Operation};
use crate::gpu::{
    physical_binding::PinnedPage,
    texture::image_bridge::{
        commands::{Kernels, States}, lifecycle::{self, Access, Epoch, NativeOperation, Operand},
        ownership::{Location as PageLocation, PayloadEvidence, Request}, GuestImage, ImageUse,
    },
};
use std::sync::Arc;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::DXGI_ERROR_NOT_FOUND};

// List-X footprint binding is not executed yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct FootprintBinding {
    pub resource: ID3D12Resource,
    pub native: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub requests: Vec<Request>,
}

/// Facts for this single reserved queue epoch, never observations from guest
/// recording time. The provider owns all bindings to the retirement fence.
/// For a raw footprint, return only a native buffer that already aliases its
/// actual bytes at execution, plus every required canonical ownership request.
/// A footprint requiring unimplemented gather/scatter must return Err.
// List-X footprint binding is not executed yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) unsafe trait ExecutionFacts: Send + Sync {
    fn certify_native_equivalence(&self, op: &Operation) -> Result<(),String>;
    fn prove_native_only(&self, resource: &ID3D12Resource, sub: u32, write: bool, state: D3D12_RESOURCE_STATES) -> Result<(),String>;
    fn state_and_payload(&self, native: &ID3D12Resource, image: &GuestImage, sub: u32, region: D3D12_BOX, access: Access,
        required: D3D12_RESOURCE_STATES) -> Result<(States,PayloadEvidence),String>;
    fn bind_footprint(&self, original: &Location, region: D3D12_BOX, write: bool, required: D3D12_RESOURCE_STATES) -> Result<FootprintBinding,String>;
    fn pin(&self, image: &GuestImage, vpn: u32, location: PageLocation) -> Result<Arc<PinnedPage>, &'static str>;
}

// List-X footprint binding is not executed yet; kept for the next fix.
#[allow(dead_code)]
struct Bound {
    op: Operation,
    facts: Arc<dyn ExecutionFacts>,
}

impl NativeOperation for Bound {
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList) -> Result<(),String> { self.op.issue(list) }
    fn pin(&self, image: &GuestImage, vpn: u32, location: PageLocation) -> Result<Arc<PinnedPage>, &'static str> {
        self.facts.pin(image, vpn, location)
    }
}

impl Operation {
    // List-X footprint binding is not executed yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn prepare(mut self, queue: &ID3D12CommandQueue, epoch: Epoch, kernels: Arc<Kernels>,
        facts: Arc<dyn ExecutionFacts>) -> Result<lifecycle::Operation,String> {
        if !self.permits(queue.GetDesc().Type) { return Err("invalid native operation on this execution queue".into()); }
        facts.certify_native_equivalence(&self)?;
        let (src_box, dst_box) = self.regions()?;
        let (src, dst, role, required) = match &mut self {
            Self::Copy { src, dst, .. } => {
                let format = |loc: &Location| match loc.layout {
                    Layout::Footprint { native, .. } => native.Footprint.Format,
                    Layout::Subresource(_) => loc.resource.GetDesc().Format,
                };
                if format(src) != format(dst) { return Err("cross-format copy requires a proven destination coordinate transform".into()); }
                (src,dst,ImageUse::Copy,[D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST])
            }
            Self::Resolve { src, dst, region, .. } => {
                if region.as_ref().is_some_and(|(_,_,_,m)| !matches!(*m, D3D12_RESOLVE_MODE_MIN|D3D12_RESOLVE_MODE_MAX|D3D12_RESOLVE_MODE_AVERAGE)) {
                    return Err("decompress or sampler-feedback resolve requires a different payload adapter".into());
                }
                (src,dst,ImageUse::Resolve,[D3D12_RESOURCE_STATE_RESOLVE_SOURCE,D3D12_RESOURCE_STATE_RESOLVE_DEST])
            }
        };
        let mut operands = Vec::new();
        let mut raw = Vec::new();
        for (loc,region,write,state) in [(src,src_box,false,required[0]),(dst,dst_box,true,required[1])] {
            let [w,h,d] = loc.extent()?;
            if region.left >= region.right || region.top >= region.bottom || region.front >= region.back ||
                region.right > w || region.bottom > h || region.back > d { return Err("invalid native operation coverage".into()); }
            let whole = [region.left,region.top,region.front,region.right,region.bottom,region.back] == [0,0,0,w,h,d];
            let sub = match loc.layout {
                Layout::Footprint { guest, native } => {
                    let binding = facts.bind_footprint(loc, region, write, state)?;
                    if binding.native.Footprint != native.Footprint || binding.resource.GetDesc().Dimension != D3D12_RESOURCE_DIMENSION_BUFFER {
                        return Err("buffer binding changed the native footprint layout".into());
                    }
                    loc.resource = binding.resource;
                    loc.layout = Layout::Footprint { guest, native: binding.native };
                    raw.extend(binding.requests);
                    continue;
                }
                Layout::Subresource(sub) => sub,
            };
            let image = match GuestImage::get(&loc.resource) {
                Ok(image) => image,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => {
                    facts.prove_native_only(&loc.resource, sub, write, state)?;
                    continue;
                }
                Err(e) => return Err(e.to_string()),
            };
            if !image.permits(role) { return Err("guest image does not permit this operation".into()); }
            let guest_plane = image.plane_for_sub(&loc.resource.GetDesc(),sub).map_err(|e| format!("unsupported guest image plane: {e:?}"))?;
            let access = if !write { Access::Read } else if whole { Access::FullWrite } else { Access::PartialWrite };
            let (states,evidence) = facts.state_and_payload(&loc.resource,&image,sub,region,access,state)?;
            if !states.before.contains(state) || states.after != states.before {
                return Err("unproven entry/exit states for native copy/resolve".into());
            }
            if matches!(role,ImageUse::Copy) && matches!(evidence,PayloadEvidence::MetadataFreeFullTypedOverwriteFromBinary16) {
                return Err("a bitwise copy is not a typed producer".into());
            }
            operands.push(Operand { image, native:loc.resource.clone(), guest_plane, sub, role, access, region:(!whole).then_some(region), states, evidence });
        }
        lifecycle::Operation::new(queue,epoch,kernels,operands,raw,Box::new(Bound { op:self, facts }))
    }
}
