use super::{Copy, FallbackPage, Pipeline};
use crate::hle::d3d12x::{page_mappings::queue, physical_directory::recordings};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) type Lease = Arc<dyn Send + Sync>;
#[derive(Clone)]
// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Input {
    Native {
        resource: ID3D12Resource,
        offset: u64,
        bytes: u64,
    },
    Guest {
        va: u64,
        bytes: u64,
    },
}
#[derive(Clone)]
// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Need {
    pub(crate) input: Input,
    pub(crate) state: D3D12_RESOURCE_STATES,
}
// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Bound {
    pub(crate) resource: ID3D12Resource,
    pub(crate) offset: u64,
}
// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Packet {
    raw: ID3D12CommandList,
    capture: Arc<queue::Recordings>,
    kind: D3D12_COMMAND_LIST_TYPE,
    device: ID3D12Device,
    #[cfg(test)]
    scratch_ids: Vec<usize>,
}

// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Recipe {
    pub(crate) needs: Vec<Need>,
    pub(crate) predicate: Option<(Input, D3D12_PREDICATION_OP)>,
}

#[derive(Clone)]
// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum GuestOp {
    Draw {
        count: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    },
    DrawIndexed {
        count: u32,
        instances: u32,
        first: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    Indirect {
        signature: ID3D12CommandSignature,
        max: u32,
        count: bool,
    },
}

// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct FixedConsumer {
    pub(crate) inputs: Recipe,
    pub(crate) ia: Vec<super::recipes::Binding>,
    pub(crate) op: GuestOp,
}

impl FixedConsumer {
    // Fixed native-buffer recipes have no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn instantiate(
        &self,
        dev: &ID3D12Device,
        ty: D3D12_COMMAND_LIST_TYPE,
        pipeline: Arc<Pipeline>,
        views: &ID3D12DescriptorHeap,
        fallback: impl FnMut(usize, u32) -> Option<(FallbackPage, Lease)>,
        state: impl FnOnce(&ID3D12GraphicsCommandList) -> Result<Lease, String>,
    ) -> Result<Packet, String> {
        use super::recipes::Binding;
        let n = self.ia.len();
        if n > 0 && ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
            return Err("IA requires a direct packet".into());
        }
        let indirect = match &self.op {
            GuestOp::Indirect {
                signature,
                count,
                max,
            } => {
                let (kind, stride) = crate::hle::d3d12x::indirect::plain_action(signature)
                    .ok_or("not a plain indirect signature")?;
                if kind != 2 && ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
                    return Err("graphics action needs DIRECT".into());
                }
                let required = [u64::from(*max) * u64::from(stride), 4];
                let number = 1 + usize::from(*count);
                for (k, &min) in required[..number].iter().enumerate() {
                    let input = &self
                        .inputs
                        .needs
                        .get(n + k)
                        .ok_or("missing indirect input")?
                        .input;
                    let (bytes, offset) = match input {
                        Input::Guest { bytes, va } => (*bytes, *va),
                        Input::Native { bytes, offset, .. } => (*bytes, *offset),
                    };
                    if bytes < min || offset % 4 != 0 {
                        return Err("invalid indirect range".into());
                    }
                }
                number
            }
            _ if ty != D3D12_COMMAND_LIST_TYPE_DIRECT => return Err("draw needs DIRECT".into()),
            _ => 0,
        };
        if self.inputs.needs.len() != n + indirect {
            return Err("wrong number of fixed guest inputs".into());
        }
        for (item, need) in self.ia.iter().zip(&self.inputs.needs) {
            let (state, required) = match *item {
                Binding::Vertex { bytes, .. } => {
                    (D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, bytes)
                }
                Binding::Index { bytes, .. } => (D3D12_RESOURCE_STATE_INDEX_BUFFER, bytes),
                _ => return Err("not an IA binding".into()),
            };
            let bytes = match &need.input {
                Input::Guest { bytes, .. } | Input::Native { bytes, .. } => *bytes,
            };
            if need.state != state || bytes < u64::from(required) {
                return Err("wrong IA input range/state".into());
            }
        }
        if self.inputs.needs[n..]
            .iter()
            .any(|p| p.state != D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
        {
            return Err("wrong indirect input state".into());
        }
        self.inputs
            .instantiate(dev, ty, pipeline, views, fallback, |list, b, predicate| {
                let retained = state(list)?;
                if let Some((p, op)) = predicate {
                    list.SetPredication(&p.resource, p.offset, op);
                } else {
                    list.SetPredication(None, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
                }
                for (&binding, r) in self.ia.iter().zip(b) {
                    let va = r
                        .resource
                        .GetGPUVirtualAddress()
                        .checked_add(r.offset)
                        .ok_or("IA address overflow")?;
                    match binding {
                        Binding::Vertex {
                            slot,
                            bytes,
                            stride,
                        } => list.IASetVertexBuffers(
                            slot,
                            Some(&[D3D12_VERTEX_BUFFER_VIEW {
                                BufferLocation: va,
                                SizeInBytes: bytes,
                                StrideInBytes: stride,
                            }]),
                        ),
                        Binding::Index { bytes, format } => {
                            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                                BufferLocation: va,
                                SizeInBytes: bytes,
                                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format),
                            }))
                        }
                        _ => unreachable!(),
                    }
                }
                match &self.op {
                    GuestOp::Draw {
                        count,
                        instances,
                        first,
                        first_instance,
                    } => list.DrawInstanced(*count, *instances, *first, *first_instance),
                    GuestOp::DrawIndexed {
                        count,
                        instances,
                        first,
                        base_vertex,
                        first_instance,
                    } => list.DrawIndexedInstanced(
                        *count,
                        *instances,
                        *first,
                        *base_vertex,
                        *first_instance,
                    ),
                    GuestOp::Indirect {
                        signature,
                        max,
                        count,
                    } => {
                        let a = &b[n];
                        let c = if *count { Some(&b[n + 1]) } else { None };
                        list.ExecuteIndirect(
                            signature,
                            *max,
                            &a.resource,
                            a.offset,
                            c.map(|b| &b.resource),
                            c.map_or(0, |b| b.offset),
                        );
                        return Ok(Arc::new((retained, signature.clone())));
                    }
                }
                Ok(retained)
            })
    }
}

impl Recipe {
    // Fixed native-buffer recipes have no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn instantiate(
        &self,
        dev: &ID3D12Device,
        ty: D3D12_COMMAND_LIST_TYPE,
        pipeline: Arc<Pipeline>,
        views: &ID3D12DescriptorHeap,
        fallback: impl FnMut(usize, u32) -> Option<(FallbackPage, Lease)>,
        guest: impl FnOnce(
            &ID3D12GraphicsCommandList,
            &[Bound],
            Option<(&Bound, D3D12_PREDICATION_OP)>,
        ) -> Result<Lease, String>,
    ) -> Result<Packet, String> {
        Packet::record(
            dev,
            ty,
            pipeline,
            views,
            self.needs.clone(),
            self.predicate.clone(),
            fallback,
            guest,
        )
    }
}

#[cfg(test)]
#[path = "fixed_test.rs"]
mod tests;

// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
struct Open(usize);
impl Drop for Open {
    fn drop(&mut self) {
        recordings::ended(self.0);
    }
}

// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn uav(dev: &ID3D12Device, bytes: u64) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut out = None;
    dev.CreateCommittedResource(
        &props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        None,
        &mut out,
    )
    .map_err(|e| e.to_string())?;
    out.ok_or("no native gather allocation".into())
}

// Fixed native-buffer recipes have no caller yet; kept for the next fix.
#[allow(dead_code)]
unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
    r: &ID3D12Resource,
    state: D3D12_RESOURCE_STATES,
) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        ..Default::default()
    };
    b.Anonymous.Transition = std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        StateAfter: state,
    });
    list.ResourceBarrier(std::slice::from_ref(&b));
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

impl Packet {
    // Fixed native-buffer recipes have no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn record(
        dev: &ID3D12Device,
        ty: D3D12_COMMAND_LIST_TYPE,
        pipeline: Arc<Pipeline>,
        views: &ID3D12DescriptorHeap,
        mut needs: Vec<Need>,
        predicate: Option<(Input, D3D12_PREDICATION_OP)>,
        mut fallback: impl FnMut(usize, u32) -> Option<(FallbackPage, Lease)>,
        guest: impl FnOnce(
            &ID3D12GraphicsCommandList,
            &[Bound],
            Option<(&Bound, D3D12_PREDICATION_OP)>,
        ) -> Result<Lease, String>,
    ) -> Result<Self, String> {
        if !matches!(
            ty,
            D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE
        ) {
            return Err("fixed gather needs direct or compute".into());
        }
        let alloc: ID3D12CommandAllocator =
            dev.CreateCommandAllocator(ty).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, ty, &alloc, None)
            .map_err(|e| e.to_string())?;
        let raw = list.as_raw() as usize;
        let _open = Open(raw);
        let locator = match recordings::packet_index(raw) { 0 => return Err("no recording locator".into()), l => l };
        if let Some((p, _)) = &predicate {
            let (offset, bytes) = match p {
                Input::Guest { va, bytes } => (*va, *bytes),
                Input::Native { offset, bytes, .. } => (*offset, *bytes),
            };
            if bytes != 8 || offset % 8 != 0 {
                return Err("invalid guest predicate width or alignment".into());
            }
        }
        let op = predicate.map(|(input, op)| {
            needs.push(Need {
                input,
                state: D3D12_RESOURCE_STATE_PREDICATION,
            });
            op
        });
        let mut leases: Vec<Lease> = vec![
            pipeline.clone(),
            Arc::new(views.clone()),
            Arc::new(alloc),
            Arc::new(list.clone()),
        ];
        let mut inputs = Vec::new();
        for (i, need) in needs.into_iter().enumerate() {
            let b = match need.input {
                Input::Native {
                    resource,
                    offset,
                    bytes,
                } => {
                    let desc = resource.GetDesc();
                    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                        || offset.checked_add(bytes).is_none_or(|end| end > desc.Width)
                    {
                        return Err("invalid native input".into());
                    }
                    Bound { resource, offset }
                }
                Input::Guest { va, bytes } => {
                    if bytes == 0 {
                        return Err("empty guest input".into());
                    }
                    let read = Copy::read(va, bytes, locator, |p| {
                        let (meta, lease) = fallback(i, p)?;
                        leases.push(lease);
                        Some(meta)
                    })
                    .ok_or("fixed guest range has no proven fallback")?;
                    let (pb, db) = read.storage_bytes();
                    let (plan, data) = (uav(dev, pb)?, uav(dev, db)?);
                    pipeline.record_read(&list, views, &read, &plan, &data)?;
                    transition(&list, &data, need.state);
                    leases.push(Arc::new(plan));
                    Bound {
                        resource: data,
                        offset: 0,
                    }
                }
            };
            leases.push(Arc::new(b.resource.clone()));
            inputs.push(b);
        }
        let predicate = if let Some(op) = op {
            let p = inputs.pop().unwrap();
            if p.offset % 8 != 0
                || p.offset
                    .checked_add(8)
                    .is_none_or(|e| e > p.resource.GetDesc().Width)
            {
                return Err("invalid predicate range".into());
            }
            Some((p, op))
        } else {
            None
        };
        leases.push(guest(
            &list,
            &inputs,
            predicate.as_ref().map(|(p, op)| (p, *op)),
        )?);
        list.Close().map_err(|e| e.to_string())?;
        let cmd: ID3D12CommandList = list.cast().map_err(|e| e.to_string())?;
        recordings::retain_native(raw, Arc::new(leases))?;
        let ptrs = [cmd.as_raw()];
        let capture = queue::Recordings::hold(&ptrs, &ptrs);
        Ok(Self {
            raw: cmd,
            capture,
            kind: ty,
            device: dev.clone(),
            #[cfg(test)]
            scratch_ids: inputs
                .iter()
                .map(|b| b.resource.as_raw() as usize)
                .collect(),
        })
    }

    // Fixed native-buffer recipes have no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn submit(self, q: &ID3D12CommandQueue) -> Result<(), String> {
        let mut device: Option<ID3D12Device> = None;
        q.GetDevice(&mut device).map_err(|e| e.to_string())?;
        if device.as_ref() != Some(&self.device) || q.GetDesc().Type != self.kind {
            return Err("fixed packet queue type or device differs".into());
        }
        queue::reader(q.as_raw(), &[self.raw.as_raw()], Some(&self.capture));
        Ok(())
    }
}
