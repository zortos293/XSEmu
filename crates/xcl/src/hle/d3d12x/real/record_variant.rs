use super::super::material_gate_views::prepass::records::Layout;
use super::*;
use crate::gpu::recomp::{
    dxc,
    emit::mesh_passthrough_for,
    pso::{material_gate::stage_bindings, TwoPass},
};

pub(super) struct Recipe {
    dev: ID3D12Device,
    rs: ID3D12RootSignature,
    fixed: Vec<(u32, Vec<u8>)>,
    ms: Vec<u8>,
    ps: Vec<u8>,
    inputs: Vec<crate::gpu::dxbc::SigElement>,
    producer: Option<(Vec<u8>, Vec<u8>, bool, bool)>,
    variant: OnceLock<Option<[(usize, usize); 2]>>,
}

impl Recipe {
    pub(super) unsafe fn new(
        this: *mut c_void,
        subs: &[(u32, &[u8])],
        tp: &TwoPass,
        rs: *mut c_void,
        compute_rs: *mut c_void,
        guest_ps: Option<&[u8]>,
        guest_ms: Option<&[u8]>,
    ) -> Result<Self, String> {
        let ms = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32;
        let root: Vec<_> = subs.iter().filter(|s| s.0 == 0).collect();
        let meshes: Vec<_> = subs.iter().filter(|s| s.0 == ms).collect();
        let pixels: Vec<_> = subs.iter().filter(|s| s.0 == 2).collect();
        let raw = |r| [7, 0, r, r, 11, 0];
        if root.len() != 1
            || abi_rs_raw(this) != Some(rs)
            || root[0].1 != (rs as usize).to_le_bytes()
            || meshes.len() != 1
            || payload_shader(meshes[0].1) != Some(tp.ms.as_slice())
            || subs.iter().any(|(t, bytes)| match t {
                2 => false,
                1 | 3..=6 | 12 | 24 => bytes.iter().any(|&v| v != 0),
                0 | 8..=11 | 13..=18 | 20..=21 | 25..=28 => false,
                _ => true,
            })
        {
            return Err("not an isolated two-root mesh/pixel stream".into());
        }
        if pixels.len() != 1 {
            return Err("not exactly one PS".into());
        }
        let ps = payload_shader(pixels[0].1).ok_or("missing native PS")?;
        if stage_bindings(ps, 0)? != [raw(0)] {
            return Err("PS is outside the raw-u0-only class".into());
        }
        for blob in [ps, tp.ms.as_slice()] {
            if crate::gpu::shader::container_parts(blob)
                .ok_or("invalid stage")?
                .iter()
                .any(|p| p.fourcc == "RTS0")
            {
                return Err("embedded stage root signature".into());
            }
        }
        let inputs = crate::gpu::dxbc::signatures(guest_ps.ok_or("missing guest PS")?).0;
        if inputs.is_empty()
            || tp
                .record
                .outputs
                .iter()
                .any(|e| matches!(e.system_value, 2 | 3))
        {
            return Err("unsupported mesh export interface".into());
        }
        if stage_bindings(&tp.ms, 13)? != [raw(1)] {
            return Err("ordinary mesh uses more than the raw record UAV u1".into());
        }
        let producer = if super::super::material_gate_views::prepass::records::producer() {
            if !tp.samplers.is_empty() || compute_rs != rs {
                return Err(
                    "ordinary compute stage uses a different root or static samplers".into(),
                );
            }
            let sh = crate::gpu::xbox::parse(guest_ms.ok_or("missing guest MS")?)
                .ok_or("invalid guest MS")?;
            let f = sh
                .fssm
                .iter()
                .find(|f| f.stages.first().is_some_and(|s| s.hw == 7))
                .ok_or("missing guest NGG variant")?;
            let pr = f.programs.first().ok_or("no producer program")?;
            let code = f
                .data
                .get(pr.offset..pr.offset.checked_add(pr.len).ok_or("producer bounds")?)
                .ok_or("producer bounds")?;
            if crate::gpu::shader::container_parts(&tp.cs)
                .ok_or("invalid baseline CS")?
                .iter()
                .any(|p| p.fourcc == "RTS0")
            {
                return Err("embedded CS root".into());
            }
            Some((
                code.to_vec(),
                tp.cs.clone(),
                crate::gpu::recomp::fssm_wave64(f, code),
                wave32_only(this),
            ))
        } else {
            None
        };
        Ok(Self {
            producer,
            dev: ID3D12Device::from_raw_borrowed(&this)
                .ok_or("missing device")?
                .clone(),
            rs: ID3D12RootSignature::from_raw_borrowed(&rs)
                .ok_or("missing root signature")?
                .clone(),
            fixed: subs
                .iter()
                .filter(|(t, _)| *t != 0 && *t != 2 && *t != ms)
                .map(|(t, b)| (*t, b.to_vec()))
                .collect(),
            ms: tp.ms.clone(),
            ps: ps.to_vec(),
            inputs,
            variant: OnceLock::new(),
        })
    }

    pub(super) fn pipeline(
        &self,
        record: &crate::gpu::recomp::MsInterface,
    ) -> Option<[(usize, usize); 2]> {
        if dxc::in_cached_only() {
            return None;
        }
        *self
            .variant
            .get_or_init(|| match unsafe { self.build(record) } {
                Ok(p) => {
                    crate::xlog_info!(
                        "mesh record probe: verified private mesh u2; ordinary MS u1 / unchanged PS u0; pso={:#x} extended_rs={:#x} original_rs={:#x} producer_pso={:#x}",p[0].0,p[0].1,self.rs.as_raw() as usize,p[1].0
                    );
                    Some(p)
                }
                Err(e) => {
                    crate::xlog_warn!("mesh record probe: keeping ordinary MS PSO: {e}");
                    None
                }
            })
    }

    unsafe fn build(
        &self,
        record: &crate::gpu::recomp::MsInterface,
    ) -> Result<[(usize, usize); 2], String> {
        let this = self.dev.as_raw();
        let layout = Layout::new(record).ok_or("unsupported export layout")?;
        let raw = |r| [7, 0, r, r, 11, 0];
        let src = mesh_passthrough_for(record, Some(&self.inputs));
        let ordinary = dxc::cached_only(|| dxc::compile_cached(&src, "main", "ms_6_6")).0;
        if !ordinary.is_ok_and(|(b, hit)| hit && b == self.ms)
            && dxc::compile(&src, "main", "ms_6_6", &[])? != self.ms
        {
            return Err("regenerated mesh is not the ordinary compiled mesh".into());
        }
        let variant = layout.instrument(src)?;
        let dxil = dxc::for_program("qualified mesh export observer", || {
            dxc::compile(&variant, "main", "ms_6_6", &[])
        })?;
        if crate::gpu::dxbc::psv(&dxil) != crate::gpu::dxbc::psv(&self.ms)
            || crate::gpu::dxbc::signatures(&dxil) != crate::gpu::dxbc::signatures(&self.ms)
        {
            return Err("variant changed the ordinary mesh interface".into());
        }
        if stage_bindings(&dxil, 13)? != [raw(1), raw(2)] {
            return Err("variant did not add solely the raw observation UAV u2".into());
        }
        let ext = extended(&self.dev, D3D12_SHADER_VISIBILITY_MESH)?;
        let mut out = StreamOut::default();
        out.put(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            &ext.as_raw(),
        );
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &dxil);
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &self.ps);
        for (ty, bytes) in &self.fixed {
            let ty = *ty;
            let (_, align) = super::super::null::subobject_payload(ty)
                .ok_or("unknown fixed-function stream subobject")?;
            out.raw(ty, bytes, align == 8);
        }
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        COUNT
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                (v < 8).then_some(v + 1)
            })
            .map_err(|_| "private PSO cap")?;
        let _slot = driver_slot(dxil.len() + self.ps.len(), false, false, || {
            "qualified mesh export observer".into()
        })
        .map_err(|_| "driver admission declined")?;
        let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
        let desc = [out.0.len(), out.0.as_ptr() as usize];
        let mut p = std::ptr::null_mut();
        let hr = f(
            this,
            desc.as_ptr().cast(),
            &pc_iid(Kind::PipelineState),
            &mut p,
        );
        if hr < 0 || p.is_null() {
            return Err(format!("driver HRESULT {hr:#x}"));
        }
        MESH_DRIVER_COMPILES.fetch_add(1, Ordering::Relaxed);
        let root = ext.as_raw() as usize;
        let raster = ID3D12PipelineState::from_raw(p);
        drop(_slot);
        static KEEP: Mutex<Vec<(ID3D12PipelineState, ID3D12RootSignature)>> =
            Mutex::new(Vec::new());
        let mut compute = (0, 0);
        if let Some((code, ordinary, wave, split)) = &self.producer {
            let dxil = crate::gpu::recomp::pso::material_gate::producer::compile(
                code, record, *wave, *split, ordinary,
            )?;
            let rs = extended(&self.dev, D3D12_SHADER_VISIBILITY_ALL)?;
            let _slot = driver_slot(dxil.len(), false, false, || {
                "paired private compute observer".into()
            })
            .map_err(|_| "producer driver admission declined")?;
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
            desc.pRootSignature = std::mem::transmute::<
                *mut c_void,
                std::mem::ManuallyDrop<Option<ID3D12RootSignature>>,
            >(rs.as_raw());
            desc.CS = bytecode(&dxil);
            let f: PsoFn =
                std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
            let mut c = std::ptr::null_mut();
            let hr = f(
                this,
                (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(),
                &pc_iid(Kind::PipelineState),
                &mut c,
            );
            if hr < 0 || c.is_null() {
                return Err(format!("private producer driver HRESULT {hr:#x}"));
            }
            MESH_DRIVER_COMPILES.fetch_add(1, Ordering::Relaxed);
            compute = (c as usize, rs.as_raw() as usize);
            KEEP.lock()
                .unwrap()
                .push((ID3D12PipelineState::from_raw(c), rs));
            crate::xlog_info!(
                "mesh producer probe: verified original CS, sibling bindings and private u2"
            );
        }
        KEEP.lock().unwrap().push((raster, ext));
        Ok([(p as usize, root), compute])
    }
}

unsafe fn extended(
    dev: &ID3D12Device,
    visibility: D3D12_SHADER_VISIBILITY,
) -> Result<ID3D12RootSignature, String> {
    let params = [0, 1, 2].map(|i| D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: i,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: if i == 2 {
            visibility
        } else {
            D3D12_SHADER_VISIBILITY_ALL
        },
    });
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: 3,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    let mut blob = None;
    D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| e.to_string())?;
    let blob = blob.ok_or("missing extended root blob")?;
    dev.CreateRootSignature(
        0,
        std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()),
    )
    .map_err(|e| e.to_string())
}
