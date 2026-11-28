use super::super::super::{gpu_timing, list_x, real};
use super::super::super::console_heap::vsharp;
use crate::gpu::recomp::{ranked, MsInterface};
use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};
use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList_Vtbl;

const MAGIC: u32 = 0x4d455850;
const GROUPS: usize = 2;
static COMPLETE: AtomicUsize = AtomicUsize::new(0);
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        super::enabled() && std::env::var("XCL_MESH_EXPORT_CLASS").as_deref() == Ok("raw-ps")
    })
}
pub(crate) fn done() -> bool {
    COMPLETE.load(Ordering::Relaxed) >= 8
}
pub(crate) fn producer() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled() && std::env::var_os("XCL_MESH_PRODUCER_SITES").is_some())
}
const HEAD: usize = 8;

pub(crate) fn selected() -> Option<u64> {
    static HASH: OnceLock<Option<u64>> = OnceLock::new();
    *HASH.get_or_init(|| {
        if !super::enabled() {
            return None;
        }
        let s = std::env::var("XCL_MESH_EXPORT_MS").ok()?;
        u64::from_str_radix(s.trim().trim_start_matches("0x"), 16)
            .ok()
            .filter(|v| *v != 0)
    })
}

pub(crate) struct Layout {
    mv: usize,
    mp: usize,
    vd: usize,
    rd: usize,
    distances: Vec<(usize, usize, String)>,
}

impl Layout {
    pub(crate) fn new(ms: &MsInterface) -> Option<Self> {
        if !(1..=256).contains(&ms.max_vertices) || !(1..=256).contains(&ms.max_primitives) {
            return None;
        }
        let mut distances = Vec::new();
        for (k, e) in ranked(&ms.outputs, true)
            .into_iter()
            .enumerate()
            .filter(|(_, e)| matches!(e.system_value, 2 | 3))
        {
            let n = (8 - e.mask.leading_zeros()) as usize;
            if !(1..=4).contains(&n) || e.component_type != 3 {
                return None;
            }
            for c in 0..n {
                distances.push((
                    k,
                    if n == 1 { 4 } else { c },
                    format!("{}[{}]", e.semantic(), c),
                ));
            }
        }
        if distances.len() > 8 {
            return None;
        }
        Some(Self {
            mv: ms.max_vertices as usize,
            mp: ms.max_primitives as usize,
            vd: ms.vert_dwords(),
            rd: ms.record_dwords(),
            distances,
        })
    }
    fn vw(&self) -> usize {
        5 + self.distances.len()
    }
    fn pw(&self) -> usize {
        HEAD + self.mv * self.vw()
    }
    fn gw(&self) -> usize {
        self.pw() + self.mp * 5
    }
    fn words(&self) -> usize {
        HEAD + GROUPS * self.gw()
    }

    pub(crate) fn instrument(&self, mut src: String) -> Result<String, String> {
        let mut replace = |from: &str, to: String| -> Result<(), String> {
            if src.matches(from).count() != 1 {
                return Err("unsupported ordinary mesh source shape".into());
            }
            src = src.replacen(from, &to, 1);
            Ok(())
        };
        replace(
            "RWByteAddressBuffer io : register(u0);",
            "RWByteAddressBuffer io : register(u2);".into(),
        )?;
        let original = format!(
            "    uint nv = min(res.Load(rec * 4u), {}u);\n    uint np = min(res.Load(rec * 4u + 4u), {}u);",
            self.mv, self.mp
        );
        replace(
            &original,
            format!(
                "    uint raw_nv = res.Load(rec * 4u); uint raw_np = res.Load(rec * 4u + 4u); uint nv = min(raw_nv, {}u); uint np = min(raw_np, {}u); bool observe = gid.x < {GROUPS}u && gid.y == 0u && gid.z == 0u && io.Load(0u) == {MAGIC}u; uint ob = 4u * ({HEAD}u + gid.x * {}u); if (observe && gi == 0u) {{ uint was; io.InterlockedAdd(ob, 1u, was); io.Store4(ob + 4u,uint4(raw_nv,raw_np,nv,np)); io.Store(ob + 20u,gid.x); }}",
                self.mv,
                self.mp,
                self.gw()
            ),
        )?;
        let mut vertex = format!(
            "        verts[i] = o; if (observe) {{ uint ov = ob + 4u * ({HEAD}u + i * {}u); io.Store(ov,1u); io.Store4(ov + 4u,asuint(o.pos));",
            self.vw()
        );
        for (j, (k, c, _)) in self.distances.iter().enumerate() {
            let component = [".x", ".y", ".z", ".w", ""][*c];
            vertex.push_str(&format!(
                " io.Store(ov + {}u,asuint(o.p{k}{component}));",
                20 + j * 4
            ));
        }
        vertex.push_str(" }");
        replace("        verts[i] = o;", vertex)?;
        replace(
            "        tris[j] = select(bad, uint3(0u, 0u, 0u), x);",
            format!(
                "        uint3 emitted = select(bad, uint3(0u, 0u, 0u), x); tris[j] = emitted; if (observe) {{ uint op = ob + 4u * ({}u + j * 5u); io.Store2(op,uint2(1u,p)); io.Store3(op+8u,emitted); }}",
                self.pw()
            ),
        )?;
        Ok(src)
    }
}

pub(crate) struct Lease {
    cpu: usize,
    gpu: u64,
    variant: [(usize, usize); 2],
    layout: Layout,
    slot: u32,
    generation: u64,
    original: usize,
    root: usize,
    views: [u64; 2],
    threads: u64,
    viewport: [f32; 6],
    provenance: Option<vsharp::Snapshot>,
    scissor: [i32; 4],
}
fn describe_producer(
    rows: &[[u32; 16]; 14],
    sites: [u64; 14],
    hits: [u32; 5],
    key: &str,
    record: &[u32],
    l: &Layout,
    provenance: Option<&vsharp::Snapshot>,
) -> Result<(), &'static str> {
    let pc = [
        0, sites[0], sites[1], sites[2], sites[4], sites[5], sites[6], sites[7], sites[8],
        sites[9], sites[10], sites[11], sites[12], sites[13],
    ];
    let mut present = [false; 14];
    for (i, row) in rows.iter().enumerate() {
        present[i] = row[0] == 0x50524f44 && row[1] as u64 == pc[i];
        if !present[i] && row.iter().any(|&w| w != 0) {
            return Err("torn, stale or wrong-PC producer row");
        }
    }
    let names = [
        "entry",
        "load_srd",
        "header_load",
        "header_post",
        "readfirst_counts",
        "primitive_load",
        "primitive_post",
        "pre_fallback_counts",
        "allocation",
        "bit_reverse_result",
        "fallback_position_no_write",
        "fallback_primitive",
        "normal_primitive",
        "normal_position",
    ];
    for (i, r) in rows.iter().enumerate().filter(|(i, _)| present[*i]) {
        crate::xlog_info!(
            "mesh producer probe: operand {key} site={} pc={:#x} exec={:08x?} words={:08x?}",
            names[i],
            pc[i],
            &r[2..4],
            &r[4..]
        );
    }
    if ![0usize, 1, 2, 3, 4, 7, 8].iter().all(|&i| present[i]) {
        return Err("incomplete group-zero allocation path");
    }
    if rows[0][4..7] != [0, 0, 0] {
        return Err("not logical group-zero slot zero");
    }
    let ptr = &rows[1][4..];
    let hdr = &rows[2][4..];
    let post = &rows[3][4..];
    let cnt = &rows[7][4..];
    let alloc = &rows[8][4..];
    if ptr[5..9] != hdr[8..12] {
        return Err("SRD changed between its load and the meshlet header");
    }
    vsharp::report(provenance,key,&rows[0],ptr,hdr[7]);
    let bounds = hdr[1];
    let resource = |r: &[u32], bytes: usize| {
        let u = crate::gpu::uma::get()?;
        let k = r[7] as usize;
        if k >= u.shader_windows() as usize {
            return None;
        }
        let (base, len) = u.window(k);
        let off = r[5] as usize;
        if off.checked_add(bytes)? > len {
            return None;
        }
        let at = base.checked_add(off)?;
        Some((
            at,
            real::uma_window_buffer(at, bytes),
            real::console_buffers_at(at),
        ))
    };
    let range = resource(hdr, 8);
    let usable = [0, 1].map(|i| {
        (hdr[0] != 0 && bounds & (1 << i) != 0 && range.as_ref().is_some_and(|r| r.1.is_some()))
            .then_some(hdr[2 + i])
    });
    let decode = usable[0].map(|v| [(v & 127) + 1, ((v >> 7) & 127) + 1]);
    let active = |i: usize| present[i] && rows[i][4] != 0;
    let fallback = active(11);
    let normal = active(12);
    let e = |i: usize| active(i).then_some(rows[i][7]);
    let word = if record[4] > 0 {
        Some(record[l.pw() + 1])
    } else {
        None
    };
    let pos = if record[3] > 0 {
        Some(&record[HEAD + 1..HEAD + 5])
    } else {
        None
    };
    let normal_pos = active(13).then_some(&rows[13][7..11]);
    let counts = [cnt[0], cnt[1]];
    let producer_counts = [alloc[1], alloc[2]];
    crate::xlog_info!("mesh producer probe: group branch counts {key} alloc_header_writers={} fallback_pos_EN0_reaches={} fallback_primitive_exports={} normal_primitive_exports={} normal_position_exports={}",hits[0],hits[1],hits[2],hits[3],hits[4]);
    let pf = present[5].then(|| resource(&rows[5][4..], 4)).flatten();
    let primitive_valid = (present[5]
        && rows[5][4] != 0
        && rows[5][5] & 1 != 0
        && pf.as_ref().is_some_and(|r| r.1.is_some()))
    .then_some(rows[5][6]);
    crate::xlog_info!("mesh producer probe: joined {key} group=0 slot_in_group=0 reached={present:?} entry_s17={:#x} descriptor_source_lo_hi_window={:x?} SRD={:08x?} header_off_addrlo_addrhi_window={:08x?} header_active={} component_bounds={bounds:#x} valid_raw_header={usable:x?} post_gate_v1_v2={:08x?} raw_decoded_plus_one={decode:?} firstlane_counts={:x?} counts_pre_fallback={counts:?} out_of_guest_range={} m0={:#x} alloc_counts={producer_counts:?} record_counts={:?} sampled_primitive_input={primitive_valid:x?} reverse_active={} reverse_expected_actual={:08x?} fallback_position_enabled={} primitive_fallback_normal=[{fallback},{normal}] primitive_words_fallback_normal_record={:08x?} normal_position_emitted={normal_pos:08x?} consumed_position={pos:08x?} header_host_range_at_report={range:x?} primitive_host_range_at_report={pf:x?}",rows[0][8],&ptr[2..5],&ptr[5..9],&hdr[4..8],hdr[0],&post[1..3],&rows[4][4..6],counts.iter().any(|&v|v==0 || v>128),alloc[0],&record[3..5],active(9),[rows[9][6],rows[9][5]],active(10)&&rows[10][5]!=0,[e(11),e(12),word]);
    Ok(())
}

unsafe fn set(
    raw: *mut c_void,
    root: usize,
    pso: usize,
    views: [u64; 2],
    probe: Option<u64>,
    gfx: bool,
) {
    type V = ID3D12GraphicsCommandList_Vtbl;
    let n = std::mem::size_of::<usize>();
    gpu_timing::native::<unsafe extern "system" fn(*mut c_void, usize)>(
        raw,
        (if gfx {
            std::mem::offset_of!(V, SetGraphicsRootSignature)
        } else {
            std::mem::offset_of!(V, SetComputeRootSignature)
        }) / n,
    )(raw, root);
    let uav = gpu_timing::native::<unsafe extern "system" fn(*mut c_void, u32, u64)>(
        raw,
        (if gfx {
            std::mem::offset_of!(V, SetGraphicsRootUnorderedAccessView)
        } else {
            std::mem::offset_of!(V, SetComputeRootUnorderedAccessView)
        }) / n,
    );
    uav(raw, 0, views[0]);
    uav(raw, 1, views[1]);
    if let Some(va) = probe {
        uav(raw, 2, va);
    }
    gpu_timing::native::<unsafe extern "system" fn(*mut c_void, usize)>(
        raw,
        std::mem::offset_of!(V, SetPipelineState) / n,
    )(raw, pso);
}

impl Lease {
    pub(crate) unsafe fn bind(
        raw: *mut c_void,
        variant: [(usize, usize); 2],
        original: usize,
        root: usize,
        views: [u64; 2],
        ms: &MsInterface,
        slot: u32,
        generation: u64,
        staged_io: &[u32],
    ) -> Option<Self> {
        let (viewport, scissor) = list_x::single_viewport(raw as usize)?;
        let layout = Layout::new(ms)?;
        const BYTES: usize = 16 * 1024;
        if layout.words() * 4 > (if producer() { 12288 } else { BYTES })
            || views.contains(&0)
            || (producer() && variant[1].0 == 0)
        {
            return None;
        }
        static AREA: OnceLock<Option<usize>> = OnceLock::new();
        let base = (*AREA.get_or_init(|| {
            let u = crate::gpu::uma::get()?;
            let n = super::limit().checked_mul(BYTES)?;
            let at = u.alloc(n, 256)? as usize;
            (u.in_committed_window(at, n) && real::uma_window_buffer(at, n).is_some()).then_some(at)
        }))?;
        let cpu = base + slot as usize * BYTES;
        let gpu = real::uma_va_of(cpu)?;
        std::ptr::write_bytes(cpu as *mut u8, 0, BYTES);
        (cpu as *mut [u32; 4]).write([MAGIC, slot, generation as u32, (generation >> 32) as u32]);
        Some(Self {
            variant,
            gpu,
            cpu,
            layout,
            slot,
            generation,
            original,
            root,
            views,
            viewport,
            scissor,
            provenance: vsharp::capture(staged_io),
            threads: ms.threads.clamp(1, 128) as u64,
        })
    }

    pub(crate) unsafe fn restore(&self, raw: *mut c_void) {
        set(raw, self.root, self.original, self.views, None, true)
    }
    pub(crate) unsafe fn mesh(&self, raw: *mut c_void) {
        let (p, r) = self.variant[0];
        set(raw, r, p, self.views, Some(self.gpu), true);
    }
    pub(crate) unsafe fn compute(&self, raw: *mut c_void, tp: &real::TwoPassPso, restore: bool) {
        let (p, r) = self.variant[1];
        if restore {
            set(raw, tp.compute_rs, tp.compute, self.views, None, false);
        } else {
            set(raw, r, p, self.views, Some(self.gpu), false);
        }
    }
    pub(crate) fn report(
        self,
        list: usize,
        generation: u64,
        q: usize,
        seq: u64,
        submit: u64,
        slot: u32,
        args: &[u32; 7],
        occ: u64,
        stats: Option<[u64; 14]>,
        desc: String,
    ) {
        let grid = [args[3], args[4], args[5]];
        if args[6] != 1 || grid[0] == 0 {
            return;
        }
        if generation != self.generation
            || slot != self.slot
            || grid[0] as usize > GROUPS
            || grid[1] != 1
            || grid[2] != 1
            || args[..3] != grid
            || (!producer() && occ != 0)
            || stats.is_none_or(|s| s[12] != grid[0] as u64 * self.threads || s[13] == 0)
        {
            crate::xlog_info!(
                "mesh record probe: unavailable slot={slot} generation={generation} unsupported grid/identity/query grid={grid:?}"
            );
            return;
        }
        let data: Vec<u32> = (0..self.layout.words())
            .map(|i| unsafe { (self.cpu as *const u32).add(i).read_volatile() })
            .collect();
        if data[..4] != [MAGIC, slot, generation as u32, (generation >> 32) as u32] {
            crate::xlog_warn!("mesh record probe: unavailable header slot={slot}");
            return;
        }
        let producer_rows = if producer() {
            let Ok(pcs) = crate::gpu::recomp::pso::material_gate::producer::sites() else {
                return;
            };
            Some((
                pcs,
                std::array::from_fn::<[u32; 16], 14, _>(|i| {
                    std::array::from_fn(|j| unsafe {
                        ((self.cpu + 12288) as *const u32)
                            .add(i * 16 + j)
                            .read_volatile()
                    })
                }),
                std::array::from_fn(|i| unsafe {
                    ((self.cpu + 13248) as *const u32).add(i).read_volatile()
                }),
            ))
        } else {
            None
        };
        let result=std::thread::Builder::new().name("xcl-export-report".into()).spawn(move||{
            let l=&self.layout; let groups=grid[0] as usize;
            let key=format!("slot={slot} list={list:#x} generation={generation} q={q} seq={seq} submit={submit}");
            let unavailable=|why|crate::xlog_warn!("mesh record probe: unavailable {key}: {why}");
            let (mut verts,mut prims,mut null,mut oor,mut repeated,mut nonfinite,mut nonpositive_w,mut zero_area,mut pos_area,mut neg_area)=(0usize,0usize,0usize,0usize,0usize,0usize,0usize,0usize,0usize,0usize);
            let mut rows=Vec::new(); let mut headers=Vec::new();
            for (g,b) in data[HEAD..].chunks_exact(l.gw()).enumerate() {
                if g>=groups { if b[0]!=0 {return unavailable("unexpected group heartbeat")} continue }
                let (nv,np)=(b[3] as usize,b[4] as usize);
                if b[0]!=1 || b[5]!=g as u32 || b[1] as usize!=nv || b[2] as usize!=np || nv>l.mv || np>l.mp {return unavailable("replayed, missing, or overflowing group header")}
                if (0..nv).any(|i|b[HEAD+i*l.vw()]!=1) || (0..np).any(|i|b[l.pw()+5*i]!=1) {return unavailable("missing output lane")}
                headers.push((nv,np)); verts+=nv; prims+=np;
                for j in 0..np {
                    let tri=&b[l.pw()+5*j..][..5]; let word=tri[1]; let idx=[word&1023,(word>>10)&1023,(word>>20)&1023];
                    let killed=word>>31!=0; let bad=idx.iter().any(|&i|i as usize>=nv);
                    let emitted=if killed||bad {[0;3]} else {idx};
                    if tri[2..5]!=emitted {return unavailable("raw-to-emitted index mismatch")}
                    null+=usize::from(killed);oor+=usize::from(bad);
                    repeated+=usize::from(emitted[0]==emitted[1]||emitted[0]==emitted[2]||emitted[1]==emitted[2]);
                    let at=|i|HEAD+i as usize*l.vw();
                    let valid_indices=!bad;
                    let raw=valid_indices.then(||idx.map(|i|<[u32;4]>::try_from(&b[at(i)+1..at(i)+5]).unwrap()));
                    let distances=valid_indices.then(||idx.map(|i|b[at(i)+5..at(i)+l.vw()].to_vec()));
                    let emitted_clip=(emitted.iter().all(|&i| (i as usize)<nv)).then(||emitted.map(|i|<[u32;4]>::try_from(&b[at(i)+1..at(i)+5]).unwrap()));
                    let projected=emitted_clip.and_then(|raw|{
                        let p=raw.map(|v|v.map(|x|f32::from_bits(x) as f64));
                        if !p.iter().flatten().all(|v|v.is_finite()){nonfinite+=1;return None}
                        if p.iter().any(|v|v[3]<=0.0){nonpositive_w+=1;return None}
                        let [x,y,w,h,zmin,zmax]=self.viewport.map(f64::from);
                        let sp=p.map(|[cx,cy,cz,cw]|[x+(1.0+cx/cw)*0.5*w,y+(1.0-cy/cw)*0.5*h,zmin+(zmax-zmin)*cz/cw]);
                        let area=(sp[1][0]-sp[0][0])*(sp[2][1]-sp[0][1])-(sp[1][1]-sp[0][1])*(sp[2][0]-sp[0][0]);
                        if !killed && !bad {if area==0.0 {zero_area+=1} else if area>0.0 {pos_area+=1} else {neg_area+=1}}
                        Some((sp,area))
                    });
                    if rows.len()<32 {rows.push(format!("g={g} prim={j} word={word:08x} raw_indices={idx:?} emitted={emitted:?} raw_ref_clip_bits={raw:08x?} emitted_clip_xyzw_bits={emitted_clip:08x?} raw_ref_clip_cull_bits={distances:08x?} emitted_target_xyz_and_signed_area2={projected:?}"))}
                }
            }
            if Some(prims as u64)!=stats.map(|s|s[13]) {return unavailable("emitted count does not match the retired primitive query")}
            if prims>32 && producer_rows.is_none() {return unavailable("primitive row budget exceeded")}
            if let Some((pcs,producer,hits))=&producer_rows {
                if groups!=1 {return unavailable("producer pairing requires exactly one observed group")}
                if let Err(e)=describe_producer(producer,*pcs,*hits,&key,&data[HEAD..HEAD+l.gw()],l,self.provenance.as_ref()){return unavailable(e)}
            }
            if COMPLETE.try_update(Ordering::Relaxed,Ordering::Relaxed,|n|(n<8).then_some(n+1)).is_err(){return}
            let semantics=l.distances.iter().map(|d|d.2.as_str()).collect::<Vec<_>>();
            if prims>rows.len(){crate::xlog_info!("mesh record probe: {key} triangle_rows_shown={} triangle_rows_omitted={} (all {} primitive records checked)",rows.len(),prims-rows.len(),prims);}
            crate::xlog_info!("mesh record probe: complete {key} grid={grid:?} stats={stats:?} passed={occ} mv={} mp={} record_dwords={} record_vert_dwords={} record_header_dwords=4 tri_base_dwords={} distance_semantics={semantics:?} headers_nv_np={headers:?} vertices={verts} primitives={prims} raw_null={null} raw_out_of_range={oor} emitted_repeat={repeated} nonfinite_position={nonfinite} nonpositive_w={nonpositive_w} valid_target_area2_zero_pos_neg=[{zero_area},{pos_area},{neg_area}] viewport={:?} scissor={:?}; {desc}",l.mv,l.mp,l.rd,l.vd,4+l.mv*l.vd,self.viewport,self.scissor);
            for row in rows {crate::xlog_info!("mesh record probe: triangle {key} {row}");}
        });
        if let Err(e) = result {
            crate::xlog_warn!("mesh record probe: reporter unavailable: {e}");
        }
    }
}
