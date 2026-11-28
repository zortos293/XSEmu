//! RDNA2 → HLSL recompiler (design §5.7a / §5.7b).
//!
//! Execution model – exact without control-flow structurisation:
//! * every branch in AMD GPU code is scalar, i.e. uniform across the wave, so the program
//!   becomes a dispatcher loop over basic blocks (`switch (pc)`) with one `pc` per wave;
//!   branch-free programs are emitted as straight-line code,
//! * structured form where possible (DXC compiles it much faster): forward skips as `if`,
//!   if / else, loops with the back edge at their end (also entered in the middle: rotated),
//!   and any other forward jump – loop breaks, jumps past a join, s_endpgm before the end –
//!   as a skip flag (`xcl_skip_<target>`: set at the jump, the code up to the target runs
//!   under `if (!flag)`, loops `break`); retail pack: 90% of the programs need no `pc`,
//! * SGPRs, VCC, EXEC, M0 and SCC are ordinary variables that every lane computes alike
//!   (uniform); lane masks produced by vector instructions come from `WaveActiveBallot`,
//! * VGPRs are per-lane variables; vector writes are masked by the lane's EXEC bit,
//! * cross-lane reads use `WaveReadLaneAt`.
//!
//! Modes: `Kernel` – a compute shader that loads the initial register file from a buffer,
//! runs the program and stores registers and exports back (instruction tests on the real
//! GPU, see `runner`). `Vertex` / `Pixel` – graphics entry points whose inputs and outputs
//! follow the DXBC signatures (rules measured on retail data, design §5.7b):
//! * VS: the fetch shader puts ISG1 element r into v[4 + 4r + c] (ES variant: v[9 + 4r + c]),
//!   v0 = SV_VertexID, v3 = SV_InstanceID; `exp pos0` = SV_Position, `exp param N` = N-th
//!   non-system OSG1 element (register order), then SV_ClipDistance / SV_CullDistance,
//! * PS: the initial VGPRs follow SPI_PS_INPUT_ENA (FSSM +0x248); `v_interp_* attrN` reads the
//!   N-th non-system ISG1 element, interpolated by the PC rasteriser in the mode given by the
//!   I/J pair it uses; `exp mrtN` / `mrtz` = SV_Target / SV_Depth; lanes without a DONE export
//!   are discarded.
//! * MS (NGG, 9 282 programs in the retail pack): a compute shader (`[WaveSize(64)]`, the
//!   thread group of PSV0) runs the program; `s_sendmsg 9` (GS_ALLOC_REQ, M0 = primitives << 12
//!   | vertices), `exp pos/param` (vertex = thread index in the group) and `exp prim` (packed
//!   indices, bit 31 = null) go to a record per group in `res`; a generated pass-through mesh
//!   shader ([`mesh_passthrough`]) rasterises the records. DXC allows `WaveSize` only for
//!   compute, and the console code relies on 64-lane ballots, hence two passes. Inputs:
//!   s3[27:24] = wave index in the group, v5 = group · threads + thread (the group id is v5 >> 7
//!   in the retail code, as in RADV's gfx10.3 mesh emulation), other SGPRs from `io`.
//! * GS (legacy on-chip GS, `GS_*` + the ES variant of its VS): ES and GS code of one pipeline
//!   become one compute shader (wave64, one group = up to 21 triangles = 63 ES vertices); the
//!   ES writes its vertices to LDS (`groupshared`, one item of 16 bytes per ES output
//!   register – the ESGS item size the console driver derives from the ES) and `s_setpc_b64`
//!   continues into the GS code; GS inputs v0/v1 = LDS dword offsets of the triangle's
//!   vertices (packed 16-bit), s3 = vertices | primitives << 8, s2 = start of the GS output
//!   area in LDS (after 64 ES items); `s_sendmsg`
//!   emit/cut builds triangles from the output strips, and the emitted vertices (found from
//!   the lowest LDS address each GS lane wrote) go to the mesh-mode record, drawn by
//!   [`mesh_passthrough`]. ES vertex attributes come pre-fetched from `mem` (float4 per ISG1
//!   register, address in io[`IO_GS_VERTICES`]); the primitive count is io[`IO_GS_PRIMS`].
//! * CS (real compute ABI): the FSSM stage descriptor holds COMPUTE_PGM_RSRC2 at +0x38 (user
//!   SGPR count, TGID_X/Y/Z and TG_SIZE enables, thread-id components) and numthreads at
//!   +0x3C (= PSV0 on all 36 retail CS checked); user SGPRs come from `io`, then the enabled
//!   group ids, v0..v2 = thread id in the group.
//! Memory instructions use a flat model: descriptor/base addresses index one
//! `ByteAddressBuffer` (`mem`); initial SGPRs (user data) come from `io`. Textures come next.

pub mod dxc;
pub(crate) mod emit;
pub mod mesh_dispatch;
mod kept_pages;
mod wave_pack;
pub mod mesh_wave;
pub mod x64_masks;
pub mod pso;
pub use emit::mesh_passthrough;
pub mod runner;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod coherence_tests;
#[cfg(test)]
mod structure_tests;
#[cfg(test)]
mod branch_free_tests;
#[cfg(test)]
mod quad_read_tests;
#[cfg(test)]
mod page_split_tests;
#[cfg(test)]
mod compact_access_tests;
#[cfg(test)]
mod page_group_tests;
#[cfg(test)]
mod rt_miss_tests;
#[cfg(test)]
mod prim_index_tests;
#[cfg(test)]
mod permute_tests;
#[cfg(test)]
mod rect_corner_tests;
#[cfg(test)]
mod thread_limit_tests;
#[cfg(test)]
mod loop_guard_tests;
#[cfg(test)]
mod lane_independent_tests;
pub mod upper_lanes;
mod loop_profile;
mod lds_start;
mod structure;
mod lane_guards;
mod lane_independent;
#[cfg(test)]
mod lane_independent_form_tests;
#[cfg(test)]
mod d16_tests;
#[cfg(test)]
mod lds_start_tests;
#[cfg(test)]
mod loop_skip_tests;
pub use loop_profile::{loop_profile_enabled, LOOP_PROFILE_SITES, LOOP_PROFILE_WORDS};

use std::collections::{BTreeMap, BTreeSet};

use super::dxbc::SigElement;
use super::rdna2::{self, Decoded};

/// Kernel ABI: register file slots in the `io` / `res` buffers (dwords).
pub const KERNEL_SGPRS: usize = 128;
pub const KERNEL_VGPRS: usize = 256;
pub const KERNEL_LANES: usize = 64;
/// SCC is passed in the slot of the `null` register (125).
pub const KERNEL_SCC_SLOT: usize = 125;
/// first dword of the export area in `res`: 64 targets × 64 lanes × 4 components
pub const KERNEL_EXPORTS: usize = KERNEL_SGPRS + KERNEL_LANES * KERNEL_VGPRS;
/// split wave64 (`Options::split_wave64`): the upper half's SGPRs (+ SCC slot), then
/// `KERNEL_SPLIT_MAGIC` – the runner checks both halves ended with the same scalar state
pub const KERNEL_SPLIT_SGPRS: usize = KERNEL_EXPORTS + 64 * KERNEL_LANES * 4;
pub const KERNEL_SPLIT_MAGIC: u32 = 0x5741_5645;
pub const KERNEL_RES_DWORDS: usize = KERNEL_SPLIT_SGPRS + KERNEL_SGPRS + 1;

/// Vertex shader interface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VsInterface {
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// first attribute VGPR: 4 (hardware VS) or 9 (ES variant)
    pub attr_base: u64,
    /// the OSG1 register of each `exp param N` from the console's table (`vs_output_table`;
    /// None: the N-th distinct register)
    pub params: Option<Vec<u32>>,
}

impl VsInterface {
    /// The first VGPR the fetch shader puts ISG1 register `register` in (None: not fetched).
    pub fn input_vgpr(&self, register: u32) -> Option<u64> {
        fetched_registers(&self.inputs).iter().position(|&r| r == register).map(|k| self.attr_base + 4 * k as u64)
    }
}

/// The ISG1 registers the console's fetch shader loads, in the order of their VGPRs (four
/// each from the first attribute VGPR): those the program reads (an element's ReadWriteMask).
/// An input the program never reads gets no VGPRs – Minecraft's UI vertex shader declares
/// COLOR0 r0, NORMAL0 r1 (unread), POSITION0 r2, TEXCOORD0 r3 and reads POSITION in v8, not
/// in v[4 + 4·2]: counting the unread register put NORMAL where the position is read, and
/// every vertex of those draws on one point (the create-world screen drew nothing). A
/// signature that marks nothing as read (no such information) keeps every register.
pub fn fetched_registers(inputs: &[SigElement]) -> Vec<u32> {
    let any_read = inputs.iter().any(|e| e.rw_mask != 0);
    let mut regs: Vec<u32> = inputs.iter().filter(|e| !any_read || e.rw_mask != 0).map(|e| e.register).collect();
    regs.sort_unstable();
    regs.dedup();
    regs
}

/// Pixel shader interface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PsInterface {
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// SPI_PS_INPUT_ENA: which system values occupy the initial VGPRs
    pub input_ena: u32,
    /// the console's PS input table (`ps_input_table`; None: attributes in register order)
    pub attrs: Option<Vec<PsAttr>>,
}

/// Mesh (NGG) shader interface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MsInterface {
    /// per-vertex outputs (OSG1)
    pub outputs: Vec<SigElement>,
    /// threads per group (PSV0 numthreads.x)
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
}

/// dwords before the vertices of a mesh record: vertex count, primitive count, 2 spare
pub const MESH_HEADER: usize = 4;

impl MsInterface {
    /// dwords per vertex: pos0 + every param (4 each, raw bits)
    pub fn vert_dwords(&self) -> usize {
        4 * (1 + ranked(&self.outputs, true).len())
    }
    /// dwords of one group's record: header, vertices, primitives (1 dword each)
    pub fn record_dwords(&self) -> usize {
        MESH_HEADER + self.max_vertices as usize * self.vert_dwords() + self.max_primitives as usize
    }
}

/// Legacy geometry shader emulated in compute (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GsInterface {
    /// ES (vertex) inputs: ISG1 of the VS whose ES variant feeds the GS
    pub es_inputs: Vec<SigElement>,
    /// ES outputs (OSG1 of that VS): their registers make the ESGS item size
    pub es_outputs: Vec<SigElement>,
    /// GS outputs (OSG1): LDS layout = one 16-byte slot per register
    pub outputs: Vec<SigElement>,
    /// vertices one GS invocation may emit (PSV0)
    pub max_vertex_count: u32,
    /// byte offset of the GS code in the program (the ES code comes first and its
    /// `s_setpc_b64` jumps here); None: ES alone (`s_setpc_b64` ends it); Some(0): GS alone
    pub gs_start: Option<u64>,
}

/// triangles per thread group (their 3 vertices each fit the 64 ES lanes)
pub const GS_PRIMS_PER_GROUP: u32 = 21;
/// io slots (trap-temp SGPR range, unused by titles): primitive count, address of the
/// pre-fetched vertex attributes in `mem`, the high 32 bits of UMA window 0's base
/// (`Uma::base() >> 32`, used when `mem_windows > 1` to range-check memory window selection),
/// and the descriptor heap index of window 0's raw view (window k's at + k: the shader's only
/// way to memory, `emit::mem_accessors`).
pub const IO_GS_PRIMS: usize = 110;
pub const IO_GS_VERTICES: usize = 111;
pub const IO_UMA_BASE_HI: usize = 112;
pub const IO_MEM_VIEWS: usize = 113;
/// `XCL_DS_VALIDATE`: the window 0 offset of the word a translated shader sets when a lane's
/// address leaves the window its scalar base selected (`emit::Em::effective_address`); 0: none.
pub const IO_CARRY_AT: usize = 114;
/// The offset, in memory window io[`IO_TRIP_WINDOW`], of four counters a translated shader adds
/// to when a guard gives up
/// (`xcl_trip` in the prelude, one atomic per lane and kind): `TRIP_KINDS`; 0: none. Pixel
/// shaders do not count (a write to memory would take their early depth test away).
pub const IO_TRIP_AT: usize = 115;
/// The memory window the counters at io[`IO_TRIP_AT`] are in: one the CPU reads (the coherent
/// window, which is not window 0 in the runtime layout).
pub const IO_TRIP_WINDOW: usize = 116;
pub const IO_NGG_USER_DISPLACEMENT: usize = 117;
/// Raw SRV of this recording's locator. Its first dword selects the execution's topology SRV.
pub const IO_PHYSICAL_TABLE: usize = 118;
/// io slots (trap-temp range) of a `DispatchX` with thread dimensions (flag 0x20): the thread
/// count of each axis when it is not a whole number of groups (0: no limit, the last group is
/// whole). Every compute prologue reads the three at once (one 12-byte load: they must stay
/// consecutive) and leaves a lane out of its launch EXEC when its console thread along an axis
/// (group × group size + thread in the group) is at or past the count. Adapted from ivall/xcl
/// a3e820d8 (PR #2610, the dispatch) and 12758ed0 (nvidia-dev, the prologue).
pub const IO_THREAD_LIMIT: [usize; 3] = [119, 120, 121];
const _: () = assert!(IO_THREAD_LIMIT[1] == IO_THREAD_LIMIT[0] + 1 && IO_THREAD_LIMIT[2] == IO_THREAD_LIMIT[0] + 2);
/// io slots (trap-temp range) of a compute dispatch spread past D3D12's 65,535 groups on an axis
/// (`dispatch_spread`): the console's x and y group counts (0: not spread). Such a dispatch
/// launches `GROUP_SPREAD_WIDTH` x ceil(x * y / `GROUP_SPREAD_WIDTH`) x z groups. Every compute
/// prologue reads the two at once (one 8-byte load: they must stay consecutive), numbers its group
/// n = gid.x + `GROUP_SPREAD_WIDTH` * gid.y on the console's x-y plane, leaves (the whole group)
/// when n is at or past x * y, and runs as console group (n % x, n / x, gid.z).
pub const IO_GROUP_SPREAD: [usize; 2] = [122, 123];
const _: () = assert!(IO_GROUP_SPREAD[1] == IO_GROUP_SPREAD[0] + 1 && IO_GROUP_SPREAD[0] > IO_THREAD_LIMIT[2]);
/// groups along x of a spread dispatch (`IO_GROUP_SPREAD`): D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION
pub const GROUP_SPREAD_WIDTH: u32 = 65535;

/// Shader and native gather/scatter ABI. `va` is the normalized guest byte address (low, high).
/// The result is the absolute raw UAV index and its byte offset, including on direct UMA.
pub const GPU_PAGE_DIRECTORY_HLSL: &str = r#"
static const uint XCL_PTE_ABSENT = 0u, XCL_PTE_UNMAPPED = 1u, XCL_PTE_MAPPED = 2u, XCL_PTE_INVALID = 3u;
uint xcl_leaf_pte(uint srv, uint lo) {
    if (srv == 0u) return 0u;
    ByteAddressBuffer leaf = ResourceDescriptorHeap[NonUniformResourceIndex(srv)];
    return leaf.Load((lo >> 16) * 4u);
}
uint2 xcl_lookup_gpu_pte(uint2 va, uint directory) {
    if (va.y > 65535u) return uint2(XCL_PTE_INVALID, 0u);
    if (directory == 0u) return uint2(XCL_PTE_ABSENT, 0u);
    ByteAddressBuffer root = ResourceDescriptorHeap[directory];
    uint2 leaves = root.Load2(va.y * 8u);
    uint pte = xcl_leaf_pte(leaves.x, va.x);
    if (pte == 0u) pte = xcl_leaf_pte(leaves.y, va.x);
    if (pte == 0u) return uint2(XCL_PTE_ABSENT, 0u);
    if (pte == 0x80000000u) return uint2(XCL_PTE_UNMAPPED, pte);
    if (or((pte & 0xf8000000u) != 0x80000000u, (pte & 0xfffffu) == 0u)) return uint2(XCL_PTE_INVALID, pte);
    return uint2(XCL_PTE_MAPPED, pte);
}
uint2 xcl_raw_gpuva(uint2 va, uint directory, uint mem_views, uint uma_hi, uint windows) {
    uint2 miss = uint2(mem_views + 4u, 0u);
    uint2 p = xcl_lookup_gpu_pte(va, directory);
    if (p.x == XCL_PTE_MAPPED) return uint2(p.y & 0xfffffu, ((p.y >> 20) & 127u) * 65536u + (va.x & 65535u));
    if (p.x != XCL_PTE_ABSENT) return miss;
    uint w = va.y - uma_hi;
    return uint2(select(w < windows, mem_views + w, miss.x), select(w < windows, va.x, miss.y));
}
"#;
pub const NGG_USER_DISPLACEMENT: usize = (KERNEL_SGPRS - 8) * 4;
/// What the counters at io[`IO_TRIP_AT`] count, in their order: lanes whose loop budget
/// (`XCL_LOOP_GUARD`) ran out, whose split-wave64 meet gave up (`xcl_x_meet`), whose
/// compare-exchange ran out of retries. The third counter counts waves whose barrier
/// waited without progress for 1024 polls, not an abandoned barrier.
pub const TRIP_KINDS: [&str; 4] = ["loop guard", "wave64 meet", "barrier long-wait waves", "compare-exchange retries"];
/// The view after the four windows' (at io[`IO_MEM_VIEWS`] + 4): a null view, which shaders
/// with several windows select for an address outside them (reads 0, writes dropped).
pub const MEM_NULL_VIEW: u32 = 4;

impl GsInterface {
    /// bytes of one ES vertex in LDS (ESGS item: 16 per ES output register)
    pub fn es_item_bytes(&self) -> u32 {
        16 * (self.es_outputs.iter().map(|e| e.register).max().unwrap_or(0) + 1)
    }
    /// LDS bytes for the ES vertices of a group; the GS output area follows (s2)
    pub fn es_ring_bytes(&self) -> u32 {
        64 * self.es_item_bytes()
    }
    /// bytes of one output vertex in LDS
    pub fn vertex_bytes(&self) -> u32 {
        16 * (self.outputs.iter().map(|e| e.register).max().unwrap_or(0) + 1)
    }
    /// float4 slots per pre-fetched input vertex
    pub fn input_registers(&self) -> u32 {
        self.es_inputs.iter().filter(|e| e.system_value == 0).map(|e| e.register + 1).max().unwrap_or(0)
    }
    /// The record the emulation writes per group (drawn by `mesh_passthrough`).
    pub fn record(&self) -> MsInterface {
        let mvc = self.max_vertex_count.max(1);
        MsInterface {
            outputs: self.outputs.clone(),
            threads: 64,
            max_vertices: (GS_PRIMS_PER_GROUP * mvc).clamp(1, 256),
            max_primitives: (GS_PRIMS_PER_GROUP * mvc.saturating_sub(2).max(1)).clamp(1, 256),
        }
    }
}

/// Compute shader with the console's dispatch ABI (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CsInterface {
    pub threads: [u32; 3],
    pub user_sgprs: u32,
    /// TGID_X / Y / Z enabled: the group id follows the user SGPRs in this order
    pub tgid: [bool; 3],
    /// TG_SIZE SGPR after the group ids (wave count | wave index << 6 – not used by the pack)
    pub tg_size: bool,
    /// thread id VGPRs: 1 (v0), 2 (v0, v1) or 3 (v0..v2)
    pub tid_components: u32,
}

impl CsInterface {
    /// From the FSSM stage descriptor at `desc` (COMPUTE_PGM_RSRC2 at +0x38, numthreads +0x3C).
    pub fn from_descriptor(fssm: &[u8], desc: usize) -> Option<CsInterface> {
        let rd = |o: usize| fssm.get(desc + o..desc + o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let r = rd(0x38)?;
        let threads = [rd(0x3C)?, rd(0x40)?, rd(0x44)?];
        if threads.iter().any(|&t| t == 0 || t > 1024) || threads.iter().product::<u32>() > 1024 {
            return None;
        }
        Some(Self::from_rsrc2(threads, r))
    }

    fn from_rsrc2(threads: [u32; 3], r: u32) -> Self {
        Self {
            threads,
            user_sgprs: (r >> 1) & 31,
            tgid: [(r >> 7) & 1 == 1, (r >> 8) & 1 == 1, (r >> 9) & 1 == 1],
            tg_size: (r >> 10) & 1 == 1,
            tid_components: ((r >> 11) & 3) + 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    Kernel,
    Compute(CsInterface),
    Vertex(VsInterface),
    Pixel(PsInterface),
    Mesh(MsInterface),
    Geometry(GsInterface),
}

/// ES variant of a VS container + a GS container → one program (ES code, then GS code) and
/// its interface. Triangle input and triangle-strip output only.
pub fn geometry_program(es_blob: &[u8], gs_blob: &[u8]) -> Result<(Vec<u8>, GsInterface), String> {
    use super::xbox;
    let pick = |blob: &[u8], hw: u8| -> Result<Vec<u8>, String> {
        let sh = xbox::parse(blob).ok_or("not a console shader container")?;
        let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == hw)).ok_or_else(|| format!("no variant {}", xbox::hw_stage_name(hw)))?;
        let p = f.programs.first().ok_or("FSSM without a program")?;
        Ok(f.data[p.offset..p.offset + p.len].to_vec())
    };
    let (es, gs) = (pick(es_blob, 2)?, pick(gs_blob, 3)?);
    let psv = super::dxbc::psv(gs_blob).ok_or("GS without PSV0")?;
    if psv.gs_input_primitive != 3 || psv.gs_output_topology != 5 {
        return Err(format!("GS: input {} / output {} – supported: triangles → triangle strip", psv.gs_input_primitive, psv.gs_output_topology));
    }
    let (es_inputs, es_outputs) = super::dxbc::signatures(es_blob);
    let g = GsInterface {
        es_inputs,
        es_outputs,
        outputs: super::dxbc::signatures(gs_blob).1,
        max_vertex_count: psv.gs_max_vertex_count as u32,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);
    Ok((code, g))
}

/// Non-system elements in register order – the order of `exp param N` / `v_interp attrN`.
pub fn ranked(sig: &[SigElement], with_clip: bool) -> Vec<&SigElement> {
    let mut user: Vec<&SigElement> = sig.iter().filter(|e| e.system_value == 0 && !e.name.starts_with("SV_")).collect();
    user.sort_by_key(|e| e.register);
    if with_clip {
        let mut clip: Vec<&SigElement> = sig.iter().filter(|e| matches!(e.system_value, 2 | 3)).collect();
        clip.sort_by_key(|e| e.register);
        user.extend(clip);
    }
    user
}

/// What `exp param N` (VS, MS) and `v_interp attrN` (PS) name on the console is a signature
/// *register*, not an element: elements packed into one register (XDefiant: UV1 in .xy and UV0
/// in .zw of one) share a param, and the console's compiler numbers only the registers it uses
/// (Fortnite's Slate PS reads COLOR0 and TEXCOORD1 of COLOR0, ORIGINAL_POSITION0, TEXCOORD0,
/// TEXCOORD1: attr1 is TEXCOORD1's register). XCL used to count elements on both sides – right
/// only while the VS and PS signatures were alike and unpacked (the two errors cancelled), and
/// Fortnite's UI sampled at ORIGINAL_POSITION. Now a param / attribute is a register: the VS
/// writes channel c of param N into every element at that register (component c of each), the
/// PS reads channel c of an attribute from an element at its register – the PC linkage stays
/// by semantic, the struct declarations are unchanged.
///
/// The registers in `exp param` / attribute order when the console's own table is not at hand:
/// the distinct registers of `ranked(sig, with_clip)` (for unpacked signatures: one per element,
/// as before).
pub fn param_registers(sig: &[SigElement], with_clip: bool) -> Vec<u32> {
    let mut regs = Vec::new();
    for e in ranked(sig, with_clip) {
        if !regs.contains(&e.register) {
            regs.push(e.register);
        }
    }
    regs
}

/// The elements (places in `ranked(sig, with_clip)`) at signature register `register`.
pub fn elements_at(sig: &[SigElement], with_clip: bool, register: u32) -> Vec<usize> {
    ranked(sig, with_clip).iter().enumerate().filter(|(_, e)| e.register == register).map(|(j, _)| j).collect()
}

/// FSSM sub-version (+0x10): 5 in Fortnite's GDK; 6 in XDefiant's and Minecraft's, which has 8
/// more bytes before the stage tables.
fn fssm_shift(fssm: &[u8]) -> Option<usize> {
    match fssm.get(0x10..0x14).map(|b| u32::from_le_bytes(b.try_into().unwrap()))? {
        5 => Some(0),
        6 => Some(8),
        _ => None,
    }
}

/// One attribute of the console's PS input table (FSSM): the ISG1 register `v_interp attrN`
/// reads and its components; `kind` 6 = a user input (0x11 seen once, for a system value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsAttr {
    pub kind: u8,
    pub register: u8,
    pub mask: u8,
    pub flat: bool,
}

/// The PS input table of an FSSM: SPI_PS_INPUT_ENA and, per `v_interp attrN`, the register it
/// reads. 32 records of 8 bytes at +0x130 (+ the sub-version shift): kind, register, component
/// mask, attribute number, 0, 0, register again, 1 flat / 2 interpolated; then the record count
/// (+0x100) and SPI_PS_INPUT_ENA (+0x110). Checked on every cached PS of three titles (472, 34,
/// 1): consistent, and covering every attribute the code reads.
pub fn ps_input_table(fssm: &[u8]) -> Option<(u32, Vec<PsAttr>)> {
    let base = 0x130 + fssm_shift(fssm)?;
    let rd = |o: usize| fssm.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (n, ena) = (rd(base + 0x100)? as usize, rd(base + 0x110)?);
    if n > 32 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let r = fssm.get(base + 8 * k..base + 8 * k + 8)?;
        if r[3] as usize != k || r[4] != 0 || r[5] != 0 || r[6] != r[1] {
            return None;
        }
        out.push(PsAttr { kind: r[0], register: r[1], mask: r[2], flat: r[7] == 1 });
    }
    Some((ena, out))
}

/// The VS output table of an FSSM: per `exp param N`, the OSG1 register it writes. 32 records
/// of 6 bytes at +0x1D4 (+ the sub-version shift): kind (6 user, 0 position), register, param
/// number, 0, 0, 0; the record count after them (+0xC0). Usually param N = the N-th output
/// register, but not always (a Fortnite VS exports its SV_Position register as a param).
/// Checked on every cached VS of three titles (984, 13, 1).
pub fn vs_output_table(fssm: &[u8]) -> Option<Vec<u32>> {
    let base = 0x1D4 + fssm_shift(fssm)?;
    let n = fssm.get(base + 0xC0..base + 0xC4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))? as usize;
    if n > 32 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let r = fssm.get(base + 6 * k..base + 6 * k + 6)?;
        if r[2] as usize != k || r[3..6] != [0, 0, 0] {
            return None;
        }
        out.push(r[1] as u32);
    }
    Some(out)
}

/// The register `v_interp attr` reads: the console's table, else the attr-th distinct register.
pub fn ps_attribute_register(ps: &PsInterface, attr: usize) -> Option<u32> {
    match &ps.attrs {
        Some(t) => t.get(attr).map(|a| a.register as u32),
        None => param_registers(&ps.inputs, false).get(attr).copied(),
    }
}

/// The place in `ranked(inputs, false)` (the PC input struct's a0, a1 …) to read channel `chan`
/// of attribute `attr` from: an element at its register (the one declaring `chan`, if any).
pub fn ps_attribute_slot(ps: &PsInterface, attr: usize, chan: usize) -> Option<usize> {
    let r = ps_attribute_register(ps, attr)?;
    let all = ranked(&ps.inputs, false);
    all.iter().position(|e| e.register == r && e.mask >> chan & 1 == 1).or_else(|| all.iter().position(|e| e.register == r))
}

/// The attribute the input at `slot` of `ranked(inputs, false)` belongs to (None: not read).
pub fn ps_slot_attribute(ps: &PsInterface, slot: usize) -> Option<usize> {
    let r = ranked(&ps.inputs, false).get(slot)?.register;
    match &ps.attrs {
        Some(t) => t.iter().position(|a| a.register as u32 == r),
        None => param_registers(&ps.inputs, false).iter().position(|&x| x == r),
    }
}

/// The register `exp param N` of a VS writes: the console's table, else the N-th distinct one.
pub fn vs_param_register(vs: &VsInterface, n: usize) -> Option<u32> {
    match &vs.params {
        Some(t) => t.get(n).copied(),
        None => param_registers(&vs.outputs, true).get(n).copied(),
    }
}

/// Hardware order and VGPR sizes of the PS inputs enabled in SPI_PS_INPUT_ENA (bit → size):
/// persp sample/center/centroid (2), pull model (3), linear sample/center/centroid (2),
/// line stipple, pos x/y/z/w, front face, ancillary, sample coverage, pos fixed (1 each).
pub const PS_INPUT_SIZES: [u64; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

/// First VGPR of PS input `bit` (None if not enabled).
pub fn ps_input_vgpr(ena: u32, bit: u32) -> Option<u64> {
    if ena >> bit & 1 == 0 {
        return None;
    }
    Some((0..bit).filter(|b| ena >> b & 1 == 1).map(|b| PS_INPUT_SIZES[b as usize]).sum())
}

/// Interface of a console program from its container (None: not a VS/PS we can drive).
pub fn interface(blob: &[u8], sh: &super::xbox::XboxShader, f: &super::xbox::Fssm) -> Option<Mode> {
    let (inputs, outputs) = super::dxbc::signatures(blob);
    let hw = f.stages.first()?.hw;
    match (sh.stage.as_str(), hw) {
        ("VS", 4) => Some(Mode::Vertex(VsInterface { inputs, outputs, attr_base: 4, params: vs_output_table(&f.data) })),
        ("PS", 5) => {
            let (ena, attrs) = match ps_input_table(&f.data) {
                Some((ena, t)) => (ena, Some(t)),
                None => (f.data.get(0x248..0x24C).map(|b| u32::from_le_bytes(b.try_into().unwrap()))?, None),
            };
            Some(Mode::Pixel(PsInterface { inputs, outputs, input_ena: ena, attrs }))
        }
        // ES variant of a VS (before a GS) and the GS itself, each translatable on its own
        ("VS", 2) => Some(Mode::Geometry(GsInterface { es_inputs: inputs, es_outputs: outputs, max_vertex_count: 1, ..Default::default() })),
        ("GS", 3) => {
            let p = super::dxbc::psv(blob)?;
            Some(Mode::Geometry(GsInterface { outputs, max_vertex_count: p.gs_max_vertex_count.max(1) as u32, gs_start: Some(0), ..Default::default() }))
        }
        ("CS", 6) => Some(Mode::Compute(CsInterface::from_descriptor(&f.data, f.stages.first()?.offset)?)),
        ("MS", 7) => {
            let p = super::dxbc::psv(blob)?;
            let threads = p.numthreads.map(|t| t[0] * t[1].max(1) * t[2].max(1)).filter(|&t| t > 0).unwrap_or(128);
            if p.ms_topology != 2 {
                return None; // lines / points: not yet
            }
            Some(Mode::Mesh(MsInterface {
                outputs,
                threads,
                max_vertices: (p.ms_max_vertices as u32).clamp(1, 256),
                max_primitives: (p.ms_max_primitives as u32).clamp(1, 256),
            }))
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub mode: Mode,
    /// wave64 (64-bit EXEC/VCC) or wave32 program
    pub wave64: bool,
    /// kernel mode only: 64 forces `[WaveSize(64)]` for a wave32 program, which then runs as
    /// two independent halves – the wave-size independent form VS/PS always get (tests)
    pub hw_wave: u32,
    /// the GPU runs only 32-lane waves (`D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveLaneCountMax`
    /// < 64, e.g. NVIDIA): each console wave of a wave64 kernel / compute / mesh / geometry
    /// program runs as two hardware wave32 halves that exchange masks and lane values through
    /// groupshared memory (`docs/wave64-on-wave32.md`). For a wave64 VS / PS (wave size picked
    /// by the driver) it means the GPU *can* run 32-lane waves (`pso::graphics_wave32`, also
    /// RDNA): the program gets the values of lanes 32..63 it reads where a 32-lane wave lacks
    /// them (`upper_lanes`). No effect on wave32 programs.
    pub split_wave64: bool,
    pub legacy_compute_wave64: bool,
    /// a wave64 compute program every lane of which computes on its own (`lane_independent`),
    /// for a GPU with 32-lane waves: one console lane per host thread, without `WaveSize`,
    /// ballots and lane reads. `emit_program` decides it from the program; what a caller sets
    /// is not read
    pub lane_independent: bool,
    pub compute_per_access_coherence: bool,
    pub bvh_node_fix: bool,
    /// ray tracing off as "every ray misses" (`rt_all_miss`): an `image_bvh(64)_intersect_ray` reads no
    /// node and gives what a scene without an acceleration structure gives, 0xFFFFFFFF in all
    /// four result registers
    pub rt_all_miss: bool,
    /// compute: the launch EXEC leaves out the lanes past a `DispatchX` with thread dimensions'
    /// per-axis thread counts (io[`IO_THREAD_LIMIT`]; `dispatch_thread_dimensions`)
    pub thread_limits: bool,
    /// compute: the prologue runs a dispatch spread past 65,535 groups on an axis as the console's
    /// groups (io[`IO_GROUP_SPREAD`]; `dispatch_spread`)
    pub group_spread: bool,
    pub loop_profile: bool,
    pub guard_trip_counters: bool,
    /// first register (space 1) of the program's static samplers (another stage of the same
    /// pipeline may come first)
    pub static_sampler_base: u32,
    /// 4 GiB memory windows the program may address (`gpu::uma`): 1 = `mem` only (the HLSL is
    /// then exactly the single-window form); 2..4 add `mem1`.. (u3..) and pick the window of
    /// every access from the high address bits
    pub mem_windows: u32,
    /// a compute program's memory functions stay inline whatever its size (`emit::outline_from`):
    /// the form without calls, whose DXIL size decides whether the program is outlined
    /// (`pso::outline_budget`)
    pub no_outline: bool,
    /// structured control flow when the program allows it (false, or `XCL_RECOMP_NO_STRUCTURED`:
    /// always the guarded block chain)
    pub structured: bool,
    /// no lane guard where EXEC provably holds every lane the wave started with (false, or
    /// `XCL_RECOMP_NO_EXEC_PROOF`: every vector write guarded)
    pub exec_proof: bool,
    /// float compares test their operands for NaN explicitly (`emit::float_compare`): AMD
    /// drivers need it (`pso::nan_guards`), the DXIL's ordered compares are exact without
    pub nan_guards: bool,
    /// `Mode::Mesh` only: the program as a host mesh shader (ms_6_6) instead of the compute
    /// first pass (`MeshNative`)
    pub mesh_native: Option<MeshNative>,
    /// `Mode::Mesh` only: a group of more than one console wave as one host mesh shader
    /// (`pso::mesh_native_host_threads`), instead of two passes. Off: the gate in
    /// `decode_program` refuses such a group, as it always has. `pso::mesh_multi_wave` reads
    /// the switch.
    pub mesh_multi_wave: bool,
    /// the constant data the console compiler put after the program in its code region
    /// (`code_data`): what a load through an `s_getpc_b64` address reads. None: such a load
    /// reads guest memory at the program-relative address, as a kernel test's `mem` holds it
    pub code_data: Option<CodeData>,
}

/// Constant data in a program's code region: the dwords from `at` (bytes from the program's
/// first instruction, the address `s_getpc_b64` counts in).
#[derive(Debug, Clone)]
pub struct CodeData {
    pub at: u32,
    pub words: Vec<u32>,
}

/// The code region of `f` from the first data block after program `p` to the end of the last
/// one, the bytes between them included. None: no data block follows the program.
pub fn code_data(f: &super::xbox::Fssm, p: &super::xbox::Program) -> Option<CodeData> {
    let program_end = p.offset.checked_add(p.len)?;
    let after = || f.data_blocks.iter().filter(|b| b.len > 0 && b.offset >= program_end);
    let at = (after().map(|b| b.offset).min()? - p.offset) & !3;
    let last = after().max_by_key(|b| b.offset)?;
    let words = f.data.get(p.offset + at..last.offset.checked_add(last.len)?)?
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect::<Vec<u32>>();
    Some(CodeData { at: at.try_into().ok()?, words })
}

impl Default for Options {
    fn default() -> Self {
        Options { mode: Mode::Kernel, wave64: true, hw_wave: 0, split_wave64: false, legacy_compute_wave64: legacy_compute_wave64(), lane_independent: false, compute_per_access_coherence: compute_per_access_coherence(), bvh_node_fix: bvh_node_fix(), rt_all_miss: rt_all_miss(), thread_limits: dispatch_thread_dimensions(), group_spread: dispatch_spread(), loop_profile: false, guard_trip_counters: true, static_sampler_base: 0, mem_windows: 1, no_outline: false, structured: true, exec_proof: true, nan_guards: true, mesh_native: None, mesh_multi_wave: false, code_data: None }
    }
}

pub(super) fn legacy_compute_wave64() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| std::env::var("XCL_COMPUTE_WAVE64_LEGACY").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// tests: the lane-independent form is taken on this thread. The tests of the packed and
    /// the split form translate programs that qualify for it, and keep getting those forms.
    pub(super) static LANE_INDEPENDENT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// tests: translations on this thread spread their groups (`dispatch_spread`). Off, the
    /// interpreter's generated io words would hold a spread nothing launched, and every group
    /// past the first would leave.
    pub(super) static GROUP_SPREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `XCL_RECOMP_NO_LANE_INDEPENDENT`: every wave64 compute program keeps the packed or the split
/// form, with the HLSL it had before there was a lane-independent one. A translation-time
/// setting, in the translated cache's identity (`pso::translated::identity`).
pub(super) fn lane_independent_enabled() -> bool {
    #[cfg(test)]
    if !LANE_INDEPENDENT.get() {
        return false;
    }
    std::env::var_os("XCL_RECOMP_NO_LANE_INDEPENDENT").is_none()
}

pub(super) fn compute_per_access_coherence() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_COMPUTE_PER_ACCESS_COHERENCE").is_ok_and(|v| v.trim() == "1"))
}

pub(super) fn bvh_node_fix() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_BVH_NODE_FIX").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_RT_GATE`, for a run with ray tracing off (`hle::d3d12x::raytracing`): "hold" leaves the
/// pipelines that traverse acceleration structures and the title's ray passes out unbuilt;
/// "miss", the default, builds and runs them with every intersection a miss. The
/// acceleration-structure builder is off in both, so there is no structure a ray could hit.
/// Upstream XCL (ivall/xcl amd-dev 1aebe666) measured a retail title dark with its ray programs
/// held back - 707 of the 1 073 programs left out write memory that later passes and the CPU
/// read (screen-probe ray lists, ray-traced resolves, particle collisions) - and lit with every
/// ray a miss.
pub fn rt_gate_miss() -> bool {
    static MISS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MISS.get_or_init(|| rt_gate_is_miss(std::env::var("XCL_RT_GATE").ok().as_deref()))
}

fn rt_gate_is_miss(value: Option<&str>) -> bool {
    !value.is_some_and(|v| v.trim().eq_ignore_ascii_case("hold"))
}

/// Console programs are translated with every ray a miss (`Options::rt_all_miss`): ray tracing is
/// off and the gate is "miss" (`rt_gate_miss`). Never in unit tests, which translate for a
/// structure they built.
pub fn rt_all_miss() -> bool {
    !cfg!(test) && !crate::state::config().raytracing_on() && rt_gate_miss()
}

/// The console program traverses acceleration structures: an `image_bvh(64)_intersect_ray` in its
/// native code (a driver FSPX container's too).
pub fn traces_rays(blob: &[u8]) -> bool {
    any_program(blob, code_traces_rays)
}

/// A program of the container may traverse acceleration structures: the opcode words alone,
/// without the decoder. The translation cache keys on this (`pso::translated_identity`): the
/// emitter writes a miss for every intersect it decodes, also behind data words that stop
/// `code_traces_rays`' linear decode, and a program keyed as tracing without need is only
/// translated once more.
pub fn may_trace_rays(blob: &[u8]) -> bool {
    any_program(blob, has_bvh_opcode)
}

fn any_program(blob: &[u8], test: fn(&[u8]) -> bool) -> bool {
    if let Some(f) = pso::fspx(blob) {
        return test(f.code);
    }
    super::xbox::parse(blob).is_some_and(|sh| sh.fssm.iter().any(|f| f.programs.iter().any(|p| f.data.get(p.offset..p.offset + p.len).is_some_and(test))))
}

fn has_bvh_opcode(code: &[u8]) -> bool {
    code.chunks_exact(4).any(|b| {
        let w = u32::from_le_bytes(b.try_into().unwrap());
        w >> 26 == 0x3c && (w >> 18) & 0x7e == 0x66
    })
}

/// `traces_rays` of one native program: a cheap scan for the instruction's MIMG opcode words
/// first (DexNom RayTime's filter), then the decoder confirms one.
pub fn code_traces_rays(code: &[u8]) -> bool {
    let maybe = has_bvh_opcode(code);
    let mut off = 0;
    while maybe && off + 4 <= code.len() {
        let Ok(d) = super::rdna2::decode(&code[off..], off as u64) else { return false };
        if d.inst.name.get(..9).is_some_and(|n| n.eq_ignore_ascii_case("image_bvh")) {
            return true;
        }
        off += d.len;
    }
    false
}

/// `DispatchX` flag 0x20 (USE_THREAD_DIMENSIONS) as the console runs it: the counts are threads,
/// launched as whole groups with the lanes past each axis's count out of EXEC (D3D12.X writes the
/// counts to io[`IO_THREAD_LIMIT`], every compute prologue reads them). Off with
/// `XCL_DISPATCH_THREAD_DIMENSIONS=0`: the flags are ignored and the counts launched as groups,
/// as before. Part of the translation cache's identity (the compute prologue differs).
pub fn dispatch_thread_dimensions() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_DISPATCH_THREAD_DIMENSIONS").map_or(true, |v| v.trim() != "0"))
}

/// `XCL_DISPATCH_SPREAD` (default on; 0: off): a compute dispatch with more than 65,535 groups on
/// an axis (D3D12's limit; the console has none) runs spread over x and y (io[`IO_GROUP_SPREAD`])
/// instead of capped or passed on. Part of the translation cache's identity (the compute prologue
/// differs).
pub fn dispatch_spread() -> bool {
    #[cfg(test)]
    if !GROUP_SPREAD.get() {
        return false;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_DISPATCH_SPREAD").map_or(true, |v| v.trim() != "0"))
}

pub(super) fn compute_wave64_pack_max_insts() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("XCL_COMPUTE_WAVE64_PACK_MAX_INSTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(32_768))
}

/// A console mesh (NGG) program as a host mesh shader: the program's exports go to the
/// thread's own vertex and primitive (lane n of the group exports vertex n and primitive n) and
/// `s_sendmsg` GS_ALLOC_REQ gives `SetMeshOutputCounts` its counts, so one `DispatchMesh` of the
/// title's groups draws it – no record, no second pass.
///
/// HLSL has no `WaveSize` for mesh shaders, so the caller must know the host's wave size: 32
/// lanes (one guest lane per thread for wave32, up to two for wave64).
/// `pso::mesh_native_split` says whether the adapter qualifies. A group of more than one console
/// wave is refused unless `Options::mesh_multi_wave` is set, which gives it one host wave per
/// console wave (`pso::mesh_native_host_threads`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshNative {
    /// the pixel shader's input signature: the vertices carry exactly these (as
    /// `emit::mesh_passthrough_for`); None: every output of the program
    pub ps_inputs: Option<Vec<SigElement>>,
    pub lds_bytes: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Translation {
    pub hlsl: String,
    packed_compute: bool,
    /// a compute program `emit::outline_from` names: its memory functions are outlined unless
    /// `Options::no_outline` keeps them inline
    pub outline_candidate: bool,
    /// the program got the lane-independent form (`Options::lane_independent`)
    pub lane_independent: bool,
    /// instructions we cannot translate yet (mnemonic → count); the HLSL still compiles,
    /// these become comments
    pub unsupported: BTreeMap<String, usize>,
    /// "mnemonic: reason" → count
    pub unsupported_why: BTreeMap<String, usize>,
    /// instructions without an effect in this model (waits, prefetch, mode changes…)
    pub ignored: BTreeMap<String, usize>,
    pub instructions: usize,
    pub blocks: usize,
    /// static samplers the program samples with (S# built from constants in its code – the
    /// console root signature's static samplers): (S#, comparison), register
    /// `static_sampler_base + i` in space 1
    pub static_samplers: Vec<([u32; 4], bool)>,
    /// loops that may run longer than `loop_budget` gives one in a working program (`long_loops`)
    pub long_loops: Vec<String>,
    /// the program's loop budget (`loop_budget`; None only in a default value)
    pub loop_budget: Option<LoopBudget>,
}

impl Translation {
    pub fn complete(&self) -> bool {
        self.unsupported.is_empty()
    }
}

struct Ctx<'a> {
    insts: &'a [Decoded],
    starts: &'a BTreeSet<u64>,
    leaders: &'a [u64],
    /// how the block of each leader ends (`term` of [leader, next leader))
    terms: &'a [Term],
    loops: &'a [(u64, u64)],
    loop_profile: loop_profile::Sites,
    /// loops being emitted around the code being written (`Ctx::loop_enter`)
    loop_nest: std::cell::Cell<usize>,
    /// every branch: (address of the branch, target)
    edges: &'a [(u64, u64)],
    /// `structurable_to` results: (lo, hi, exit) → targets leaving the range
    memo: std::cell::RefCell<std::collections::HashMap<MemoKey, Option<BTreeSet<u64>>>>,
}

/// How a block ends (from its last instruction).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Term {
    Next,
    Cond(u64),
    Jump(u64),
    End,
    /// s_setpc and other terminators the structured form does not handle
    Other,
}

/// The instructions in [start, end) (`insts` is in address order: a binary search, not a scan –
/// retail programs have tens of thousands of instructions and blocks).
fn insts_in<'a>(c: &Ctx<'a>, start: u64, end: u64) -> &'a [Decoded] {
    let a = c.insts.partition_point(|d| d.pc < start);
    let b = c.insts.partition_point(|d| d.pc < end);
    &c.insts[a..b.max(a)]
}

fn block_end(c: &Ctx, i: usize) -> u64 {
    c.leaders.get(i + 1).copied().unwrap_or(u64::MAX)
}

fn term(c: &Ctx, start: u64, end: u64) -> Term {
    term_of(insts_in(c, start, end).last())
}

fn term_of(last: Option<&Decoded>) -> Term {
    let Some(d) = last else { return Term::Next };
    match (rdna2::branch_target(d), d.inst.name) {
        (Some(t), n) if n.starts_with("S_CBRANCH") => Term::Cond(t),
        (Some(t), "S_BRANCH" | "S_SETPC_B64") => Term::Jump(t),
        (Some(_), _) => Term::Other,
        (None, "S_ENDPGM") => Term::End,
        (None, _) if d.inst.flags & rdna2::F_TERMINATOR != 0 => Term::Other,
        _ => Term::Next,
    }
}

/// `XCL_RECOMP_WHY`: print why a program stays on the guarded block chain.
fn why(_at: u64, msg: &str) {
    if std::env::var_os("XCL_RECOMP_WHY").is_some() {
        eprintln!("unstructured: {msg}");
    }
}

impl Ctx<'_> {
    /// Before an emitted loop. One in no other loop is a loop region (upstream's are the
    /// outermost of `loop_ranges`; the same loops in the `pc` chain, the outermost natural loops
    /// in the structured form): control only moves forward between regions, so each is entered
    /// once, here, where a program with region budgets gives it its own (`LOOP_REGION_ENTER`).
    fn loop_enter(&self, em: &emit::Em, body: &mut String, pad: &str) {
        let outermost = self.loop_nest.replace(self.loop_nest.get() + 1) == 0;
        if outermost && matches!(em.loop_budget, LoopBudget::Regions { .. }) {
            body.push_str(&format!("{pad}{LOOP_REGION_ENTER}\n"));
        }
    }

    /// After the loop `loop_enter` opened.
    fn loop_leave(&self, em: &emit::Em, body: &mut String, pad: &str) {
        self.loop_nest.set(self.loop_nest.get() - 1);
        if self.loop_nest.get() == 0 && matches!(em.loop_budget, LoopBudget::Regions { .. }) {
            body.push_str(&format!("{pad}{LOOP_REGION_LEAVE}\n"));
        }
    }
}

/// (lo, hi, exit, nest, loop head of `continue`, targets entering the range)
type MemoKey = (u64, u64, Option<u64>, usize, Option<u64>, Vec<u64>);

/// At the top of every emitted loop: a lane's loop iterations are capped (`xcl_budget`, in
/// the prelude: `loop_budget`) – console code that loops on what XCL passed it wrong (an address, a
/// count) must not keep the PC GPU busy for minutes (a process with such work in flight cannot
/// even be terminated). A lane that has spent its budget leaves before the next iteration.
///
/// The test is the lane's own. Every branch of console code is scalar, so the lanes of a console
/// wave take the same path and count the same guards: their budgets are equal, the lane's own
/// test is the wave's, and the wave leaves together where the liveness passes expect it to
/// (`x64_masks::GuardExits`). That holds in every form: a native or packed wave is one
/// hardware wave; the halves of a split wave64 are two, which a wave operation never spanned
/// anyway; and a mesh group's lanes are one console wave however the host cuts it into waves.
/// Budgets differ only between console waves: the two wave32 halves of a 64-lane hardware wave
/// (`xcl_w32`) each leave by their own count, and a pixel shader's helper lane – which wave
/// operations leave out, and give undefined results in – is stopped by its own count too, where
/// `LOOP_GUARD_WAVE` could leave it looping after the live lanes had gone.
const LOOP_GUARD: &str = "if (xcl_budget == 0u) { xcl_trip(0u); break; } xcl_budget--;";

/// The guard of the base form (`XCL_RECOMP_NO_LOOP_REGIONS`, `LoopBudget::Wave`): if any active lane
/// has spent its budget, the whole hardware wave leaves before the next iteration.
const LOOP_GUARD_WAVE: &str = "if (WaveActiveAnyTrue(xcl_budget == 0u)) { xcl_trip(0u); break; } xcl_budget -= min(xcl_budget, 1u);";

/// Before a loop region of a program with region budgets (`LoopBudget::Regions`): the region
/// takes its budget out of what the lane has left for all its regions ...
const LOOP_REGION_ENTER: &str = "xcl_budget = min(XCL_LOOP_REGION, xcl_btotal); xcl_btotal -= xcl_budget;";
/// ... and after it gives back what it did not spend. So the loops themselves count one value,
/// as with a single budget.
const LOOP_REGION_LEAVE: &str = "xcl_btotal += xcl_budget;";

/// A program with region budgets (`loop_regions`) gives each loop region this many times `loop_guard`: a
/// title's own spin capped at one (0x10000 tries) must fit, also when it sits in an outer loop
/// whose head spends from the same count (upstream ivall/xcl 287a931b).
pub(super) const LOOP_BUDGETS_PER_REGION: u32 = 2;
/// ... and all of its regions together at most this many: what stops a program that runs away.
pub(super) const LOOP_BUDGETS_IN_ALL: u32 = 4;

/// The loop iterations a lane of one program may run (`loop_budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopBudget {
    /// The base form (`XCL_RECOMP_NO_LOOP_REGIONS`): this many in all its loops together, and the
    /// hardware wave leaves a loop when any of its lanes has spent them (`LOOP_GUARD_WAVE`).
    Wave(u32),
    /// This many in all its loops together, each lane by its own count (`LOOP_GUARD`): what a
    /// program has unless a switch says otherwise.
    Lane(u32),
    /// `XCL_LOOP_GUARD_REGIONS`, compute and kernel programs: `region` in each loop region – an
    /// outermost loop with the loops in it, counted anew when it is entered – and `total` in all
    /// regions together.
    Regions { region: u32, total: u32 },
}

impl LoopBudget {
    /// The most iterations one loop can get: a loop with a longer static bound is cut short
    /// (`long_loops`).
    pub fn of_a_loop(self) -> u32 {
        match self {
            LoopBudget::Wave(n) | LoopBudget::Lane(n) => n,
            LoopBudget::Regions { region, .. } => region,
        }
    }

    /// The `#define`s the prelude's counters start from.
    pub(super) fn defines(self) -> String {
        match self {
            LoopBudget::Wave(n) | LoopBudget::Lane(n) => format!("#define XCL_LOOP_GUARD {n}u\n"),
            LoopBudget::Regions { region, total } => format!("#define XCL_LOOP_REGION {region}u\n#define XCL_LOOP_TOTAL {total}u\n"),
        }
    }
}

/// `LOOP_GUARD` of the lane-independent form: a thread is alone, and leaves alone.
const LOOP_GUARD_LANE: &str = "if (xcl_budget == 0u) { xcl_trip(0u); break; } xcl_budget -= 1u;";

#[cfg(test)]
thread_local! {
    /// tests: the base form's loop guard (`XCL_RECOMP_NO_LOOP_REGIONS`) on this thread
    pub(super) static NO_LOOP_REGIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// tests: region budgets (`XCL_LOOP_GUARD_REGIONS`) on this thread
    pub(super) static LOOP_REGIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `XCL_RECOMP_NO_LOOP_REGIONS` set: every program has the wave-wide guard it had before
/// (`LoopBudget::Wave`), the same HLSL byte for byte, whatever `XCL_LOOP_GUARD_REGIONS` says.
/// Unset: loops are guarded by the lane's own test (`LOOP_GUARD`).
fn wave_wide_loop_guard() -> bool {
    #[cfg(test)]
    if NO_LOOP_REGIONS.get() {
        return true;
    }
    std::env::var_os("XCL_RECOMP_NO_LOOP_REGIONS").is_some()
}

/// `XCL_LOOP_GUARD_REGIONS` set (and not 0): a compute or kernel program spends its budget by
/// loop region (`LoopBudget::Regions`). Only where asked for: a lane may then run
/// `LOOP_BUDGETS_IN_ALL` times the iterations `loop_guard` gives it, and a title with loops that
/// run to whatever budget they get is slower for every iteration more (a retail run with a guard
/// of 65 536 instead of 1 024: 5 s a frame).
fn loop_regions() -> bool {
    #[cfg(test)]
    if LOOP_REGIONS.get() {
        return true;
    }
    std::env::var("XCL_LOOP_GUARD_REGIONS").is_ok_and(|v| v.trim() != "0")
}

/// The loop budget of a program (its instructions, the stage it is translated for): the one
/// place a translation's budget comes from – the prelude's counters, the guards, the named long
/// loops (`long_loops`) and the log all take it from here, so a class of programs that must not
/// loop for long can be given less by returning its budget here (e.g. ray-tracing programs,
/// those with an `image_bvh_intersect_ray` / `image_bvh64_intersect_ray`, whose traversal
/// loops only end on a hit: upstream's 512 where every ray is made to miss).
///
/// Every program takes `loop_guard` (`XCL_LOOP_GUARD`) for all its loops together, each lane by
/// its own count (`LoopBudget::Lane`): no lane runs more iterations than that, as before.
///
/// With `XCL_LOOP_GUARD_REGIONS` a compute or kernel program spends it by loop region instead
/// (upstream ivall/xcl 287a931b, scoped by 22c5f68c): `LOOP_BUDGETS_PER_REGION` of it in each,
/// `LOOP_BUDGETS_IN_ALL` in all, so that a spin of the title's own that gives up after 0x10000
/// tries leaves the loops after it their iterations (a retail FSR pass otherwise published a
/// flag made from a register its main loop never set). Graphics, geometry and mesh programs keep
/// one budget then too: upstream's run with region budgets in every stage hung an NVIDIA GPU
/// 580 s into a title's world, where a pixel shader loop that ran away had four budgets to spend
/// instead of one, and ran to the end with this scope. (Its other reason, an AMD compiler crash
/// on the loop-carried values the regions added to mesh loop headers, does not apply here:
/// `LOOP_REGION_ENTER` keeps them out of the loops.)
pub(super) fn loop_budget(_insts: &[Decoded], mode: &Mode) -> LoopBudget {
    let guard = loop_guard();
    if wave_wide_loop_guard() {
        return LoopBudget::Wave(guard);
    }
    match mode {
        Mode::Compute(_) | Mode::Kernel if loop_regions() => LoopBudget::Regions { region: guard.saturating_mul(LOOP_BUDGETS_PER_REGION), total: guard.saturating_mul(LOOP_BUDGETS_IN_ALL) },
        _ => LoopBudget::Lane(guard),
    }
}

/// The guard (`LOOP_GUARD`) as a program has it: one that waits for other hardware waves by
/// polling (`emit::Em::waits`) also reports the iteration as progress (`xcl_tick`).
fn loop_guard_line(em: &emit::Em) -> String {
    if em.lane_independent() {
        return LOOP_GUARD_LANE.into();
    }
    let guard = if matches!(em.loop_budget, LoopBudget::Wave(_)) { LOOP_GUARD_WAVE } else { LOOP_GUARD };
    format!("{guard}{}", if em.waits() { " xcl_tick();" } else { "" })
}

/// `XCL_LOOP_GUARD`: the loop iterations a lane may run, all its loops together (65 536 unless
/// set; `loop_budget` makes a program's budget of it). A translation-time
/// setting: it is in the HLSL (`LoopBudget::defines`, so in the DXIL cache's key) and in the
/// translated cache's identity (`pso::translated::identity`).
pub fn loop_guard() -> u32 {
    static GUARD: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GUARD.get_or_init(|| std::env::var("XCL_LOOP_GUARD").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(1 << 16))
}

/// The loops that may run more than `guard` iterations in a working program, as text: every
/// loop but those whose latch compares a counter (set to a constant before the loop, stepped
/// by a constant once in it) with a constant no more than `guard` steps away (`LoopBudget::of_a_loop`).
/// A loop the guard cuts short leaves its results unfinished.
pub(super) fn long_loops(insts: &[Decoded], guard: u32) -> Vec<String> {
    loop_bounds(insts).filter_map(|(a, b, cmp, bound)| match bound {
        Some(n) if n <= guard as u64 => None,
        Some(n) => Some(format!("{a:#x}..{b:#x}: {n} iterations")),
        None => Some(format!("{a:#x}..{b:#x}: no static bound{}", cmp.map_or(String::new(), |c| format!(" (`{}`)", rdna2::format(c))))),
    }).collect()
}

fn loop_bounds(insts: &[Decoded]) -> impl Iterator<Item = (u64, u64, Option<&Decoded>, Option<u64>)> {
    let at = |pc: u64| insts.partition_point(|d| d.pc < pc);
    let number = |s: &str| match s.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok().map(|v| v as i32 as i64),
        None => s.parse::<i64>().ok(),
    };
    let trips = move |first: usize, body: &[Decoded], cmp: &Decoded| -> Option<u64> {
        let text = rdna2::format(cmp);
        let (x, y) = text.split_once(' ')?.1.split_once(", ")?;
        let (reg, limit) = match (number(x), number(y)) {
            (None, Some(k)) => (x, k),
            (Some(k), None) => (y, k),
            _ => return None,
        };
        let n: u64 = reg.strip_prefix('s')?.parse().ok()?;
        let writes = |d: &&Decoded| rdna2::sgpr_uses(d).1.contains(&n);
        let mut steps = body.iter().filter(writes);
        let (step, None) = (steps.next()?, steps.next()) else { return None };
        let step = rdna2::format(step);
        let (name, ops) = step.split_once(' ')?;
        let by = match (name, ops.split(", ").collect::<Vec<_>>().as_slice()) {
            ("s_add_u32" | "s_add_i32", [d, s, k]) if *d == reg && *s == reg => number(k)?,
            ("s_sub_u32" | "s_sub_i32", [d, s, k]) if *d == reg && *s == reg => -number(k)?,
            ("s_addk_i32", [d, k]) if *d == reg => number(k)?,
            _ => return None,
        };
        let init = rdna2::format(insts[..first].iter().rev().find(writes)?);
        let start = match init.split_once(' ')? {
            ("s_mov_b32" | "s_movk_i32", ops) => ops.split_once(", ").filter(|(d, _)| *d == reg).and_then(|(_, k)| number(k))?,
            _ => return None,
        };
        let span = limit - start;
        (by != 0 && (span == 0 || (span > 0) == (by > 0))).then(|| span.unsigned_abs().div_ceil(by.unsigned_abs()) + 1)
    };
    loop_ranges(insts)
        .into_iter()
        .filter_map(move |(a, b)| {
            let (first, end) = (at(a), at(b));
            let body = &insts[first..end];
            let latch = body.last()?;
            // the compare the latch's condition comes from: the last one of its block
            let cmp = body.iter().rev().skip(1).take_while(|d| rdna2::branch_target(d).is_none()).find(|d| d.inst.name.starts_with("S_CMP"));
            let cmp = cmp.filter(|_| latch.inst.name.starts_with("S_CBRANCH_SCC"));
            Some((a, b, cmp, cmp.and_then(|c| trips(first, body, c))))
        })
}

/// The loops the translator emits: ranges spanned by backward jumps (far jumps included),
/// partly overlapping ones merged until they nest, sorted. `x64_masks` uses them for the
/// exits through `LOOP_GUARD`.
pub(super) fn loop_ranges(insts: &[Decoded]) -> Vec<(u64, u64)> {
    let mut loops: Vec<(u64, u64)> = Vec::new();
    for d in insts {
        if let Some(t) = rdna2::branch_target(d) {
            if t <= d.pc {
                loops.push((t, d.pc + d.len as u64));
            }
        }
    }
    loop {
        let mut merged = false;
        'scan: for i in 0..loops.len() {
            for j in 0..loops.len() {
                let (a, b) = (loops[i], loops[j]);
                if i != j && a.0 < b.0 && b.0 < a.1 && a.1 < b.1 {
                    loops[i] = (a.0, b.1);
                    loops.remove(j);
                    merged = true;
                    break 'scan;
                }
            }
        }
        if !merged {
            break;
        }
    }
    loops.sort();
    loops.dedup();
    loops
}

/// Target of an s_endpgm before the end of the program: "skip to the end".
const END: u64 = u64::MAX;
/// "Target" of a jump back to the head of the innermost loop: `continue`.
const CONT: u64 = u64::MAX - 1;
/// Deepest if / loop nesting of structured code: deeper conditional jumps become skip flags
/// (flat). Pixel shaders check for "no live lane left" after every discard and jump to one
/// shared exit – as ifs that would nest hundreds deep (the translator's recursion and DXC's
/// run out of stack).
const MAX_NEST: usize = 24;

/// Flag of a forward jump out of nested structured code: set at the jump, every level skips
/// its following code while it is set, cleared where its target starts.
fn skip_flag(t: u64) -> String {
    if t == END {
        "xcl_skip_end".into()
    } else {
        format!("xcl_skip_{t:x}")
    }
}

/// One item of a structured level.
#[derive(Clone, Copy, Debug)]
enum Item {
    /// loop [a, b) whose last block (`latch`) branches back to a; `entry`: entered by a jump
    /// to this point inside (the rotated form `for (;;) { [entry, latch) latch [a, entry) }`)
    Loop { a: u64, b: u64, latch: u64, entry: Option<u64> },
    /// `if (!cond) { [end, t) } [else { [t, u) }]` after the block [start, end)
    If { start: u64, end: u64, t: u64, u: Option<u64> },
    /// a plain block; `raise`: the forward target it jumps to past the code that follows
    /// (conditional / unconditional branch, s_endpgm before the end → END)
    Block { start: u64, end: u64, raise: Option<u64> },
}

impl Item {
    fn range(&self) -> (u64, u64) {
        match *self {
            Item::Loop { a, b, .. } => (a, b),
            Item::If { start, t, u, .. } => (start, u.unwrap_or(t)),
            Item::Block { start, end, .. } => (start, end),
        }
    }
}

/// The item starting at leader `i` of a level [lo, hi) and the index after it (`exit`: the
/// jump ending an if branch over its else branch; `pending`: forward targets of earlier jumps
/// at this level – no if may swallow one, a conditional jump past it becomes a skip flag).
/// Err: a shape the structured form cannot express (backward jumps other than loop latches,
/// s_setpc …).
fn item_at(c: &Ctx, i: usize, lo: u64, hi: u64, exit: Option<u64>, nest: usize, cont: Option<u64>, pending: &BTreeSet<u64>) -> Result<(Item, usize), String> {
    let start = c.leaders[i];
    let end = block_end(c, i);
    let next = |x: u64| c.leaders.partition_point(|&l| l < x);
    let inside = |a: u64, b: u64| pending.iter().any(|&p| p > a && p < b);
    // (a range that is exactly a loop – e.g. the body of an if – is that loop)
    if let Some(&(a, b)) = c.loops.iter().find(|&&(a, b)| a == start && b <= hi) {
        let last = next(b) - 1;
        let latch = c.leaders[last];
        if !matches!(term(c, latch, b), Term::Cond(t) | Term::Jump(t) if t == a) {
            return Err(format!("loop {a:#x}..{b:#x}: end {:?}", term(c, latch, b)));
        }
        return Ok((Item::Loop { a, b, latch, entry: None }, next(b)));
    }
    let last_pc = c.insts.last().map_or(0, |d| d.pc);
    let item = match term(c, start, end) {
        Term::Next => Item::Block { start, end, raise: None },
        // back to the head of the innermost loop: continue
        Term::Cond(t) | Term::Jump(t) if Some(t) == cont => Item::Block { start, end, raise: Some(CONT) },
        // a jump into the loop that starts right after: the rotated form (the jump itself is
        // the loop entry, emitted as nothing)
        Term::Jump(u) if rotated_loop(c, start, end, u, lo, hi).is_some() => Item::Block { start, end, raise: None },
        Term::Cond(t) if nest < MAX_NEST && t >= end && t <= hi && (t == hi || c.leaders.binary_search(&t).is_ok()) && !inside(start, t) => {
            // if / else only when both branches are structured on their own, the if branch
            // does not jump into the else branch and no earlier jump lands in it; otherwise a
            // plain if (the else branch then runs under the skip flag of the jump over it)
            let u = else_end(c, end, t, hi).filter(|&u| !inside(start, u) && else_ok(c, end, t, u, nest, cont));
            return Ok((Item::If { start, end, t, u }, next(u.unwrap_or(t))));
        }
        // past the level, or over a target an earlier jump waits for: a skip flag
        Term::Cond(t) if t > end => Item::Block { start, end, raise: Some(t) },
        Term::Jump(u) if Some(u) == exit && end == hi => Item::Block { start, end, raise: None },
        Term::Jump(u) if u >= end => Item::Block { start, end, raise: Some(u) },
        // the last block of the program: the natural end
        Term::End if end == u64::MAX || last_pc < end => Item::Block { start, end, raise: None },
        Term::End => Item::Block { start, end, raise: Some(END) },
        other => return Err(format!("block {start:#x}..{end:#x} in {lo:#x}..{hi:#x}: {other:?}")),
    };
    Ok((item, i + 1))
}

/// The rotated loop entered by the jump ending block `i` (see `rotated_loop`).
fn rotated_after(c: &Ctx, i: usize, lo: u64, hi: u64) -> Option<Item> {
    let (start, end) = (c.leaders[i], block_end(c, i));
    let Term::Jump(u) = term(c, start, end) else { return None };
    let (a, b, latch) = rotated_loop(c, start, end, u, lo, hi)?;
    Some(Item::Loop { a, b, latch, entry: Some(u) })
}

/// A loop [a, b) starting at `end` (the block [start, end) ends with `s_branch u` into it),
/// entered only by that jump: (a, b, latch).
fn rotated_loop(c: &Ctx, start: u64, end: u64, u: u64, lo: u64, hi: u64) -> Option<(u64, u64, u64)> {
    // the loop starts right after the jump, past s_nop padding (dead code, not emitted)
    let padding = |a: u64| insts_in(c, end, a).iter().all(|d| d.inst.name == "S_NOP");
    let &(a, b) = c.loops.iter().find(|&&(a, b)| a >= end && u > a && u < b && b <= hi && (a, b) != (lo, hi) && padding(a))?;
    if c.edges.iter().any(|&(_, dst)| dst >= end && dst < a) {
        return None;
    }
    c.leaders.binary_search(&u).ok()?;
    let last = c.leaders.partition_point(|&l| l < b) - 1;
    let latch = c.leaders[last];
    if u > latch || !matches!(term(c, latch, b), Term::Cond(t) | Term::Jump(t) if t == a) {
        return None;
    }
    let outside_in = c.edges.iter().any(|&(src, dst)| dst >= a && dst < b && (src < a || src >= b) && !(dst == u && src >= start && src < end));
    (!outside_in).then_some((a, b, latch))
}

/// `if (!c) { [end, t) } else { [t, u) }` holds: both branches structured, the if branch does
/// not jump into the else branch.
fn else_ok(c: &Ctx, end: u64, t: u64, u: u64, nest: usize, cont: Option<u64>) -> bool {
    if c.edges.iter().any(|&(src, dst)| (end..t).contains(&src) && dst == t) {
        return false;
    }
    let none = BTreeSet::new();
    match structurable_to(c, end, t, Some(u), nest + 1, cont, &none) {
        Some(r) if !r.iter().any(|&x| x >= t && x < u) => structurable_to(c, t, u, None, nest + 1, cont, &none).is_some(),
        _ => false,
    }
}

/// Can [lo, hi) be emitted as structured code (no `pc` guards)? Linear blocks, forward
/// conditional skips, if / else, loops whose only way back is a (conditional or not) branch
/// at their end, and
/// forward jumps out of nested code (loop breaks, jumps past a join, s_endpgm before the end)
/// whose target starts an item of an enclosing level. Some(targets that leave the range).
fn structurable(c: &Ctx, lo: u64, hi: u64) -> Option<BTreeSet<u64>> {
    structurable_to(c, lo, hi, None, 0, None, &BTreeSet::new())
}

/// The else branch of `if (cond) { [then_lo, t) } else { [t, u) }`: the last block of the
/// then branch ends with `s_branch u`, u past t and inside `hi`.
fn else_end(c: &Ctx, then_lo: u64, t: u64, hi: u64) -> Option<u64> {
    let last = c.leaders.partition_point(|&l| l < t).checked_sub(1)?;
    let lb = c.leaders[last];
    if lb < then_lo {
        return None;
    }
    match term(c, lb, t) {
        Term::Jump(u) if u > t && u <= hi && (u == hi || c.leaders.binary_search(&u).is_ok()) => Some(u),
        _ => None,
    }
}

/// `cont`: head of the innermost loop (a jump back to it is `continue`); `incoming`: forward
/// targets of jumps from before the range that land inside it (a loop entered in its middle:
/// the code before them runs under their skip flags, i.e. from the second iteration on).
fn structurable_to(c: &Ctx, lo: u64, hi: u64, exit: Option<u64>, nest: usize, cont: Option<u64>, incoming: &BTreeSet<u64>) -> Option<BTreeSet<u64>> {
    let key = (lo, hi, exit, nest.min(MAX_NEST), cont, incoming.iter().copied().collect::<Vec<_>>());
    if let Some(r) = c.memo.borrow().get(&key) {
        return r.clone();
    }
    let r = structurable_uncached(c, lo, hi, exit, nest, cont, incoming);
    c.memo.borrow_mut().insert(key, r.clone());
    r
}

/// Targets of `pending` strictly inside loop [a, b): they enter it (taken out of `pending`).
fn enter_loop(pending: &mut BTreeSet<u64>, a: u64, b: u64) -> BTreeSet<u64> {
    let inner: BTreeSet<u64> = pending.range(a + 1..b).copied().collect();
    for t in &inner {
        pending.remove(t);
    }
    inner
}

fn structurable_uncached(c: &Ctx, lo: u64, hi: u64, exit: Option<u64>, nest: usize, cont: Option<u64>, incoming: &BTreeSet<u64>) -> Option<BTreeSet<u64>> {
    let mut pending: BTreeSet<u64> = incoming.clone();
    let mut i = c.leaders.partition_point(|&l| l < lo);
    let mut queued: Option<Item> = None;
    while queued.is_some() || (i < c.leaders.len() && c.leaders[i] < hi) {
        let it = match queued.take() {
            Some(q) => q,
            None => {
                pending.remove(&c.leaders[i]);
                match item_at(c, i, lo, hi, exit, nest, cont, &pending) {
                    Ok((it, n)) => {
                        queued = rotated_after(c, i, lo, hi);
                        i = if let Some(Item::Loop { b, .. }) = queued { c.leaders.partition_point(|&l| l < b) } else { n };
                        it
                    }
                    Err(e) => {
                        why(lo, &e);
                        return None;
                    }
                }
            }
        };
        let (s, e) = it.range();
        pending.remove(&s);
        let inner = match it {
            Item::Loop { a, b, entry: None, .. } => enter_loop(&mut pending, a, b),
            _ => BTreeSet::new(),
        };
        if let Some(t) = pending.iter().find(|&&t| t > s && t < e) {
            why(s, &format!("branch to {t:#x} inside {s:#x}..{e:#x}"));
            return None;
        }
        let none = BTreeSet::new();
        let raised: BTreeSet<u64> = match it {
            Item::Loop { a, latch, entry: None, .. } => {
                let mut r = structurable_to(c, a, latch, None, nest + 1, Some(a), &inner)?;
                r.remove(&latch);
                r
            }
            Item::Loop { a, latch, entry: Some(u), .. } => {
                // the head [a, u) runs at the end of an iteration and may only fall into u
                if structurable_to(c, a, u, None, nest + 1, None, &none)?.iter().any(|&t| t != u) {
                    why(a, &format!("rotated loop {a:#x}: the head exits elsewhere than {u:#x}"));
                    return None;
                }
                let mut r = structurable_to(c, u, latch, None, nest + 1, None, &none)?;
                r.remove(&latch);
                r
            }
            Item::If { end, t, u, .. } => {
                let mut r = structurable_to(c, end, t, u, nest + 1, cont, &none)?;
                if let Some(u) = u {
                    if r.contains(&t) {
                        why(t, &format!("branch from the if to the start of the else {t:#x}"));
                        return None;
                    }
                    r.extend(structurable_to(c, t, u, None, nest + 1, cont, &none)?);
                }
                r
            }
            Item::Block { raise: Some(CONT), .. } => BTreeSet::new(),
            Item::Block { raise, .. } => raise.into_iter().collect(),
        };
        if let Some(t) = raised.iter().find(|&&t| t < e) {
            why(s, &format!("backward/inward branch to {t:#x} from {s:#x}..{e:#x}"));
            return None;
        }
        pending.extend(raised);
    }
    Some(pending)
}

/// The instructions of block [start, end) at `depth`; the flow of the last one.
fn emit_block(c: &Ctx, em: &mut emit::Em, body: &mut String, start: u64, end: u64, depth: usize, blocks: &mut usize) -> emit::Flow {
    em.indent = depth;
    let mut flow = emit::Flow::Next;
    let block: Vec<&Decoded> = insts_in(c, start, end).iter().collect();
    for (k, d) in block.iter().enumerate() {
        let next = block.get(k + 1).map(|n| n.pc).unwrap_or(end);
        flow = em.inst(d, next);
        if !matches!(flow, emit::Flow::Next) {
            break;
        }
    }
    body.push_str(&em.take());
    *blocks += 1;
    flow
}

/// Emit a range `structurable` accepted; returns the jump targets that leave it (their flags
/// are set; `flags` collects every flag for the declarations).
#[allow(clippy::too_many_arguments)]
fn emit_structured(
    c: &Ctx,
    em: &mut emit::Em,
    body: &mut String,
    lo: u64,
    hi: u64,
    exit: Option<u64>,
    nest: usize,
    cont: Option<u64>,
    incoming: &BTreeSet<u64>,
    depth: usize,
    blocks: &mut usize,
    flags: &mut BTreeSet<u64>,
) -> BTreeSet<u64> {
    let pad = emit::pad(depth);
    let mut pending: BTreeSet<u64> = incoming.clone();
    let none = BTreeSet::new();
    // flags of the open `if (!(skip …)) {` around the following items
    let mut guard: Option<BTreeSet<u64>> = None;
    let mut i = c.leaders.partition_point(|&l| l < lo);
    let mut queued: Option<Item> = None;
    while queued.is_some() || (i < c.leaders.len() && c.leaders[i] < hi) {
        let resolved = queued.is_none() && pending.remove(&c.leaders[i]);
        let it = match queued.take() {
            Some(q) => q,
            None => {
                let Ok((it, n)) = item_at(c, i, lo, hi, exit, nest, cont, &pending) else { break };
                queued = rotated_after(c, i, lo, hi);
                i = if let Some(Item::Loop { b, .. }) = queued { c.leaders.partition_point(|&l| l < b) } else { n };
                it
            }
        };
        let (s, _) = it.range();
        // targets inside a loop enter it: not a guard around the loop
        let inner = match it {
            Item::Loop { a, b, entry: None, .. } => enter_loop(&mut pending, a, b),
            _ => BTreeSet::new(),
        };
        if guard.as_ref() != Some(&pending).filter(|p| !p.is_empty()) {
            if guard.take().is_some() {
                body.push_str(&format!("{pad}}}\n"));
            }
            if !pending.is_empty() {
                let cond: Vec<String> = pending.iter().map(|&t| skip_flag(t)).collect();
                body.push_str(&format!("{pad}if (!({})) {{\n", cond.join(" || ")));
                guard = Some(pending.clone());
            }
        }
        let d = depth + guard.is_some() as usize;
        let ipad = emit::pad(d);
        if resolved {
            body.push_str(&format!("{ipad}{} = false;\n", skip_flag(s)));
        }
        let raised: BTreeSet<u64> = match it {
            Item::Loop { a, b, latch, entry } => {
                c.loop_enter(em, body, &ipad);
                let probe = c.loop_profile.probe(a, b);
                probe.before(body, &ipad);
                body.push_str(&format!("{ipad}[loop] for (;;) {{\n"));
                probe.guard(body, &ipad, em);
                let (lcont, lin) = if entry.is_none() { (Some(a), &inner) } else { (None, &none) };
                let mut r = emit_structured(c, em, body, entry.unwrap_or(a), latch, None, nest + 1, lcont, lin, d + 1, blocks, flags);
                for &t in &r {
                    if t == latch {
                        body.push_str(&format!("{ipad}    {} = false;\n", skip_flag(t)));
                    } else {
                        body.push_str(&format!("{ipad}    if ({}) break;\n", skip_flag(t)));
                    }
                }
                r.remove(&latch);
                match emit_block(c, em, body, latch, b, d + 1, blocks) {
                    emit::Flow::Cond(cond, _) => body.push_str(&format!("{ipad}    if (!({cond})) break;\n")),
                    // unconditional back edge: the loop is left through the skip flags' breaks
                    emit::Flow::Jump(_) | emit::Flow::JumpExpr(_) => {}
                    _ => body.push_str(&format!("{ipad}    break;\n")),
                }
                if let Some(u) = entry {
                    // rotated: the head after the latch, falling into the next iteration
                    for t in emit_structured(c, em, body, a, u, None, nest + 1, None, &none, d + 1, blocks, flags) {
                        body.push_str(&format!("{ipad}    {} = false;\n", skip_flag(t)));
                    }
                }
                body.push_str(&format!("{ipad}}}\n"));
                probe.after(body, &ipad);
                c.loop_leave(em, body, &ipad);
                r
            }
            Item::If { start, end, t, u } => {
                let cond = match emit_block(c, em, body, start, end, d, blocks) {
                    emit::Flow::Cond(cond, _) => cond,
                    _ => "false".into(),
                };
                body.push_str(&format!("{ipad}if (!({cond})) {{\n"));
                let mut r = emit_structured(c, em, body, end, t, u, nest + 1, cont, &none, d + 1, blocks, flags);
                if let Some(u) = u {
                    body.push_str(&format!("{ipad}}} else {{\n"));
                    r.extend(emit_structured(c, em, body, t, u, None, nest + 1, cont, &none, d + 1, blocks, flags));
                }
                body.push_str(&format!("{ipad}}}\n"));
                r
            }
            Item::Block { start, end, raise } => {
                let flow = emit_block(c, em, body, start, end, d, blocks);
                match raise {
                    None => BTreeSet::new(),
                    Some(CONT) => {
                        match flow {
                            emit::Flow::Cond(cond, _) => body.push_str(&format!("{ipad}if ({cond}) continue;\n")),
                            _ => body.push_str(&format!("{ipad}continue;\n")),
                        }
                        BTreeSet::new()
                    }
                    Some(t) => {
                        flags.insert(t);
                        match flow {
                            emit::Flow::Cond(cond, _) => body.push_str(&format!("{ipad}if ({cond}) {} = true;\n", skip_flag(t))),
                            _ => body.push_str(&format!("{ipad}{} = true;\n", skip_flag(t))),
                        }
                        BTreeSet::from([t])
                    }
                }
            }
        };
        if guard.as_ref().is_some_and(|tested| !tested.is_disjoint(&raised)) {
            guard = None;
            body.push_str(&format!("{pad}}}\n"));
        }
        pending.extend(raised);
    }
    if guard.is_some() {
        body.push_str(&format!("{pad}}}\n"));
    }
    pending
}

fn flag_name(cfg: &structure::Cfg, f: structure::Flag) -> String {
    match f {
        structure::Flag::Skip(y) if y == cfg.n => skip_flag(END),
        structure::Flag::Skip(y) => skip_flag(cfg.start[y]),
        structure::Flag::Cont(h) => format!("xcl_cont_{:x}", cfg.start[h]),
    }
}

/// Every flag `nodes` set.
fn node_flags(nodes: &[structure::Node], out: &mut BTreeSet<structure::Flag>) {
    use structure::Node;
    for n in nodes {
        match n {
            Node::Set(f) => {
                out.insert(*f);
            }
            Node::If { then, els, .. } => {
                node_flags(then, out);
                node_flags(els, out);
            }
            Node::Loop { body, .. } | Node::Guard(_, body) | Node::IfFlag(_, body) => node_flags(body, out),
            _ => {}
        }
    }
}

/// Emit the structured form (`structure`).
fn emit_nodes(c: &Ctx, cfg: &structure::Cfg, em: &mut emit::Em, body: &mut String, nodes: &[structure::Node], depth: usize, blocks: &mut usize) {
    use structure::Node;
    let pad = emit::pad(depth);
    let flags = |fs: &[structure::Flag]| fs.iter().map(|&f| flag_name(cfg, f)).collect::<Vec<_>>().join(" || ");
    for n in nodes {
        match n {
            Node::Code(b) => {
                emit_block(c, em, body, cfg.start[*b], cfg.end[*b], depth, blocks);
            }
            Node::If { block, invert, then, els } => {
                let cond = match emit_block(c, em, body, cfg.start[*block], cfg.end[*block], depth, blocks) {
                    emit::Flow::Cond(cond, _) => cond,
                    emit::Flow::Jump(_) | emit::Flow::JumpExpr(_) => "true".into(),
                    _ => "false".into(),
                };
                let test = if *invert { format!("!({cond})") } else { cond };
                body.push_str(&format!("{pad}if ({test}) {{\n"));
                emit_nodes(c, cfg, em, body, then, depth + 1, blocks);
                if !els.is_empty() {
                    body.push_str(&format!("{pad}}} else {{\n"));
                    emit_nodes(c, cfg, em, body, els, depth + 1, blocks);
                }
                body.push_str(&format!("{pad}}}\n"));
            }
            Node::Loop { header, body: inner } => {
                c.loop_enter(em, body, pad);
                let probe = c.loop_profile.probe(cfg.start[*header], cfg.loop_end(*header));
                probe.before(body, pad);
                body.push_str(&format!("{pad}[loop] for (;;) {{\n"));
                probe.guard(body, pad, em);
                emit_nodes(c, cfg, em, body, inner, depth + 1, blocks);
                body.push_str(&format!("{pad}}}\n"));
                probe.after(body, pad);
                c.loop_leave(em, body, pad);
            }
            Node::Set(f) => body.push_str(&format!("{pad}{} = true;\n", flag_name(cfg, *f))),
            Node::Clear(f) => body.push_str(&format!("{pad}{} = false;\n", flag_name(cfg, *f))),
            Node::Break => body.push_str(&format!("{pad}break;\n")),
            Node::Continue => body.push_str(&format!("{pad}continue;\n")),
            Node::Guard(fs, inner) => {
                body.push_str(&format!("{pad}if (!({})) {{\n", flags(fs)));
                emit_nodes(c, cfg, em, body, inner, depth + 1, blocks);
                body.push_str(&format!("{pad}}}\n"));
            }
            Node::IfFlag(f, inner) => {
                body.push_str(&format!("{pad}if ({}) {{\n", flag_name(cfg, *f)));
                emit_nodes(c, cfg, em, body, inner, depth + 1, blocks);
                body.push_str(&format!("{pad}}}\n"));
            }
        }
    }
}

/// The longest range [start, x) (x ≤ hi) entered only at `start`, left only by falling into
/// x, that `structurable` accepts – emitted as plain structured code under one `pc` guard.
fn structured_span(c: &Ctx, i: usize, hi: u64) -> Option<u64> {
    let start = c.leaders[i];
    // candidate ends: block boundaries where every forward branch from inside is resolved
    let mut ends = Vec::new();
    let mut reach = start;
    let mut j = i;
    while j < c.leaders.len() && c.leaders[j] < hi {
        let (s, e) = (c.leaders[j], block_end(c, j));
        match c.terms[j] {
            Term::End | Term::Other => break,
            Term::Cond(t) | Term::Jump(t) if t > s => reach = reach.max(t),
            Term::Cond(t) | Term::Jump(t) if t < start => break,
            _ => {}
        }
        let x = e.min(hi);
        if reach <= x && x > block_end(c, i) {
            ends.push(x);
        }
        j += 1;
    }
    // entered only at start: no branch from outside lands strictly inside
    let entered_inside = |x: u64| c.edges.iter().any(|&(src, dst)| dst > start && dst < x && (src < start || src >= x));
    ends.iter().rev().take(64).copied().find(|&x| !entered_inside(x) && structurable_to(c, start, x, None, 0, None, &BTreeSet::new()).is_some_and(|r| r.iter().all(|&t| t == x)))
}

/// Emit the blocks starting in [lo, hi) at nesting `depth`; loop ranges become `while`.
fn emit_range(c: &Ctx, em: &mut emit::Em, body: &mut String, lo: u64, hi: u64, depth: usize, blocks: &mut usize, flags: &mut BTreeSet<u64>) {
    let pad = emit::pad(depth);
    let mut i = c.leaders.partition_point(|&l| l < lo);
    while i < c.leaders.len() && c.leaders[i] < hi {
        let start = c.leaders[i];
        let inner = c.loops.iter().filter(|&&(a, b)| a == start && b <= hi && (a, b) != (lo, hi)).max_by_key(|x| x.1).copied();
        if let Some((a, b)) = inner {
            c.loop_enter(em, body, &pad);
            let probe = c.loop_profile.probe(a, b);
            probe.before(body, &pad);
            body.push_str(&format!("{pad}[loop] while (pc >= {a}u && pc < {b}u) {{\n"));
            probe.guard(body, &pad, em);
            emit_range(c, em, body, a, b, depth + 1, blocks, flags);
            body.push_str(&format!("{pad}}}\n"));
            probe.after(body, &pad);
            c.loop_leave(em, body, &pad);
            i = c.leaders.partition_point(|&l| l < b);
            continue;
        }
        // a single-entry, single-exit stretch: structured code under one guard
        if let Some(x) = structured_span(c, i, hi) {
            body.push_str(&format!("{pad}if (pc == {start}u) {{\n"));
            for t in emit_structured(c, em, body, start, x, None, 0, None, &BTreeSet::new(), depth + 1, blocks, flags) {
                body.push_str(&format!("{pad}    {} = false;\n", skip_flag(t)));
            }
            let next = if x == u64::MAX || !c.starts.contains(&x) { "0xFFFFFFFFu".to_string() } else { format!("{x}u") };
            body.push_str(&format!("{pad}    pc = {next};\n{pad}}}\n"));
            i = c.leaders.partition_point(|&l| l < x);
            continue;
        }
        let end = c.leaders.get(i + 1).copied().unwrap_or(u64::MAX);
        body.push_str(&format!("{pad}if (pc == {start}u) {{\n"));
        em.indent = depth + 1;
        let mut flow = emit::Flow::Next;
        let block: Vec<&Decoded> = insts_in(c, start, end).iter().collect();
        for (k, d) in block.iter().enumerate() {
            let next = block.get(k + 1).map(|n| n.pc).unwrap_or(end);
            flow = em.inst(d, next);
            if !matches!(flow, emit::Flow::Next) {
                break;
            }
        }
        body.push_str(&em.take());
        let fallthrough = if end == u64::MAX || !c.starts.contains(&end) { "0xFFFFFFFFu".to_string() } else { format!("{end}u") };
        let tail = match flow {
            emit::Flow::Next => format!("pc = {fallthrough};"),
            emit::Flow::Jump(t) => format!("pc = {t}u;"),
            emit::Flow::JumpExpr(e) => format!("pc = {e};"),
            emit::Flow::Cond(cond, t) => format!("pc = ({cond}) ? {t}u : {fallthrough};"),
            emit::Flow::End => "pc = 0xFFFFFFFFu;".to_string(),
        };
        body.push_str(&format!("{pad}    {tail}\n{pad}}}\n"));
        *blocks += 1;
        i += 1;
    }
}

/// Cross-lane reads of a wave64 program that are still wrong when the GPU runs it in waves of
/// 32 lanes (VS / PS: the driver picks the wave size), by kind: reads of a lane ≥ 32 – or of a
/// lane chosen at run time – whose value `upper_lanes` cannot follow. The form for 32-lane
/// GPUs (`Options::split_wave64`) resolves the others. A wave32 program in 64-lane waves is
/// handled (the halves act as separate waves).
pub fn wave_hazards(code: &[u8]) -> BTreeSet<&'static str> {
    wave_lane_reads(code).0
}

/// `wave_hazards` and the kinds of reads `upper_lanes` resolves.
pub fn wave_lane_reads(code: &[u8]) -> (BTreeSet<&'static str>, BTreeSet<&'static str>) {
    let mut insts = Vec::new();
    let mut off = 0;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
        off += d.len;
        insts.push(d);
    }
    rdna2::resolve_far_jumps(&mut insts);
    let a = upper_lanes::analyze(&insts);
    (a.hazards(&insts), a.resolved())
}

/// Operations that cannot be split into two independent native wave32 halves while
/// preserving one logical wave64. The split form (`Options::split_wave64`) exchanges masks
/// and lane values between the halves and needs no such list; a program without any of these
/// could run as two independent halves without exchanges (a possible fast path).
pub fn geometry_wave32_hazards(code: &[u8]) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    let mut mask_regs = BTreeSet::from([106u64, 107, 126, 127]); // VCC and EXEC
    let mut mask_scc = false;
    let mut off = 0;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
        off += d.len;
        let n = d.inst.name;
        let (reads, writes) = rdna2::sgpr_uses(&d);
        let reads_mask = reads.iter().any(|r| mask_regs.contains(r));
        // a v_cmp result is a mask too, also in an SGPR pair (VOP3)
        if (reads_mask && n.ends_with("_B64")) || n.starts_with("V_CMP") {
            mask_regs.extend(writes.iter().copied());
        }
        if (n.starts_with("S_CMP") || n.starts_with("S_CMPK")) && !reads_mask {
            mask_scc = false;
        }
        if reads_mask
            && matches!(
                n,
                "S_AND_B64"
                    | "S_OR_B64"
                    | "S_XOR_B64"
                    | "S_ANDN2_B64"
                    | "S_ORN2_B64"
                    | "S_NAND_B64"
                    | "S_NOR_B64"
                    | "S_XNOR_B64"
            )
        {
            mask_scc = true;
        }
        if mask_scc && matches!(n, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1") {
            out.insert("branch on a half-local scalar condition");
        }
        if matches!(
            n,
            "V_READLANE_B32"
                | "V_WRITELANE_B32"
                | "V_READFIRSTLANE_B32"
                | "V_PERMLANE16_B32"
                | "V_PERMLANEX16_B32"
                | "DS_BPERMUTE_B32"
                | "DS_PERMUTE_B32"
                | "DS_SWIZZLE_B32"
        ) || d.enc.name.contains("DPP")
        {
            out.insert("cross-half lane exchange");
        }
        if matches!(n, "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" | "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ") {
            out.insert("branch on a whole-wave mask");
        }
        if matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64" | "S_FLBIT_I32_B64" | "S_FLBIT_I32_I64") {
            out.insert("whole-wave mask reduction");
        }
        // one 32-bit word of a mask: a half sees its own lanes in its own word only
        if reads_mask && matches!(n, "S_BCNT0_I32_B32" | "S_BCNT1_I32_B32" | "S_FF1_I32_B32" | "S_FLBIT_I32_B32") {
            out.insert("whole-wave mask reduction");
        }
        if reads_mask && matches!(n, "S_QUADMASK_B32" | "S_QUADMASK_B64" | "S_BITREPLICATE_B64_B32") {
            out.insert("whole-wave mask rearrangement");
        }
        // v_mbcnt over a mask: lanes 32..63 also count the active lanes of 0..31, which the
        // high half does not have (the lane-index idiom, a constant SRC0, is fine)
        if n.starts_with("V_MBCNT") && !matches!(rdna2::field_value(d.enc, &d.words, "SRC0"), Some(128..=208 | 240..=248 | 255)) {
            out.insert("prefix count over a whole-wave mask");
        }
        if matches!(n, "S_CMP_EQ_U64" | "S_CMP_LG_U64" | "S_BITCMP0_B64" | "S_BITCMP1_B64") {
            out.insert("scalar condition from a whole-wave mask");
        }
    }
    let report = mesh_wave::analyze(code);
    if !report.mask_scc_branches.is_empty() {
        out.insert("branch on a half-local scalar condition");
    }
    if report.undecoded_at.is_some() {
        out.insert("incomplete geometry program decode");
    }
    if !report.invalid_branch_targets.is_empty() {
        out.insert("invalid geometry branch target");
    }
    out
}

/// Translate a complete ES+GS program; `split_wave64`: for a GPU with 32-lane waves only.
/// `mem_windows`: the title's memory windows (`pso::mem_windows`), as for VS / PS / CS.
pub fn translate_geometry(code: &[u8], interface: GsInterface, split_wave64: bool, mem_windows: u32) -> Result<Translation, String> {
    translate(code, &Options { mode: Mode::Geometry(interface), wave64: true, split_wave64, mem_windows, ..Default::default() })
}

/// Translate an NGG mesh program; `split_wave64`: for a GPU with 32-lane waves only;
/// `mem_windows`: the title's memory windows (`pso::mem_windows`), as for VS / PS / CS – with
/// one, a program reading title memory above the first window read the wrong memory.
pub fn translate_mesh(code: &[u8], ms: MsInterface, wave64: bool, split_wave64: bool, mem_windows: u32) -> Result<Translation, String> {
    translate(code, &Options { mode: Mode::Mesh(ms), wave64, split_wave64, mem_windows, ..Default::default() })
}

pub fn fssm_wave64(f: &super::xbox::Fssm, code: &[u8]) -> bool {
    !f.wave32() && detect_wave64(code)
}

/// Wave size the program was compiled for: 64-bit EXEC/VCC writes mean wave64.
pub fn detect_wave64(code: &[u8]) -> bool {
    let mut off = 0;
    let (mut w64, mut w32) = (0, 0);
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
        let n = d.inst.name;
        if n.ends_with("_SAVEEXEC_B64") || n.ends_with("_WREXEC_B64") || n.starts_with("V_CMPX") {
            w64 += n.ends_with("_B64") as i32;
        }
        if n.ends_with("_SAVEEXEC_B32") || n.ends_with("_WREXEC_B32") {
            w32 += 1;
        }
        // s_mov_b64 exec / s_and_b64 exec …: SDST 126
        if let Some(sd) = rdna2::field_value(d.enc, &d.words, "SDST") {
            if sd == 126 && n.ends_with("_B64") {
                w64 += 1;
            }
            if sd == 126 && n.ends_with("_B32") {
                w32 += 1;
            }
        }
        off += d.len;
    }
    !(w32 > 0 && w64 == 0)
}

/// Value of a scalar constant source operand (inline constant or literal), if it is one.
fn scalar_constant(d: &Decoded, code: u64) -> Option<u32> {
    Some(match code {
        128..=192 => (code - 128) as u32,
        193..=208 => (-((code - 192) as i32)) as u32,
        240..=247 => [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(code - 240) as usize].to_bits(),
        248 => 0x3E22_F983, // 1 / (2π)
        255 => d.literal?,
        _ => return None,
    })
}

/// S# built from constants in the code – how the console compiler places the static
/// samplers of a root signature: for each MIMG instruction whose four sampler SGPRs were all
/// set by s_mov / s_movk / s_brev with the same constants on every path to it, those words.
fn constant_samplers(insts: &[Decoded], leaders: &BTreeSet<u64>) -> BTreeMap<u64, [u32; 4]> {
    type Known = std::collections::HashMap<u64, u32>;
    let step = |known: &mut Known, d: &Decoded, out: Option<&mut BTreeMap<u64, [u32; 4]>>| {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n);
        if let (Some(out), "ENC_MIMG") = (out, d.enc.name) {
            if let Some(s) = f("SSAMP").map(|v| v * rdna2::register_scale("SSAMP")) {
                if let (Some(a), Some(b), Some(c), Some(e)) = (known.get(&s), known.get(&(s + 1)), known.get(&(s + 2)), known.get(&(s + 3))) {
                    out.insert(d.pc, [*a, *b, *c, *e]);
                }
            }
        }
        let set: Vec<(u64, u32)> = match d.inst.name {
            "S_MOV_B32" => f("SDST").zip(f("SSRC0").and_then(|c| scalar_constant(d, c))).map(|(r, v)| vec![(r, v)]).unwrap_or_default(),
            "S_BREV_B32" => f("SDST").zip(f("SSRC0").and_then(|c| scalar_constant(d, c))).map(|(r, v)| vec![(r, v.reverse_bits())]).unwrap_or_default(),
            "S_MOVK_I32" => f("SDST").zip(f("SIMM16")).map(|(r, v)| vec![(r, v as u16 as i16 as i32 as u32)]).unwrap_or_default(),
            // 64-bit moves of inline integers (sign-extended): S# words that are 0 / -1
            "S_MOV_B64" => match (f("SDST"), f("SSRC0")) {
                (Some(r), Some(c @ 128..=208)) => {
                    let v = scalar_constant(d, c).unwrap() as i32 as i64;
                    vec![(r, v as u32), (r + 1, (v >> 32) as u32)]
                }
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        for w in rdna2::sgpr_uses(d).1 {
            known.remove(&w);
        }
        for (r, v) in set {
            if r < 106 {
                known.insert(r, v);
            }
        }
    };
    let starts: Vec<u64> = leaders.iter().copied().collect();
    let block = |i: usize| {
        let a = insts.partition_point(|d| d.pc < starts[i]);
        let b = starts.get(i + 1).map_or(insts.len(), |&end| insts.partition_point(|d| d.pc < end));
        &insts[a..b.max(a)]
    };
    let mut followed = insts.iter().all(|d| d.inst.flags & rdna2::F_BRANCH == 0 || rdna2::branch_target(d).is_some());
    let succs: Vec<Vec<usize>> = (0..starts.len())
        .map(|i| {
            let next = (i + 1 < starts.len()).then_some(i + 1);
            let mut to = |t: u64| {
                let at = starts.binary_search(&t).ok();
                followed &= at.is_some();
                at
            };
            match term_of(block(i).last()) {
                Term::Next => next.into_iter().collect(),
                Term::Cond(t) => to(t).into_iter().chain(next).collect(),
                Term::Jump(t) => to(t).into_iter().collect(),
                Term::End => Vec::new(),
                _ => {
                    followed = false;
                    Vec::new()
                }
            }
        })
        .collect();
    let mut entry: Vec<Option<Known>> = vec![None; starts.len()];
    if followed {
        let mut dirty = vec![false; starts.len()];
        entry[0] = Some(Known::new());
        dirty[0] = true;
        while dirty.contains(&true) {
            for i in 0..starts.len() {
                if !std::mem::take(&mut dirty[i]) {
                    continue;
                }
                let mut known = entry[i].clone().unwrap_or_default();
                for d in block(i) {
                    step(&mut known, d, None);
                }
                for &s in &succs[i] {
                    if let Some(cur) = &mut entry[s] {
                        let before = cur.len();
                        cur.retain(|r, v| known.get(r) == Some(v));
                        dirty[s] |= cur.len() != before;
                    } else {
                        entry[s] = Some(known.clone());
                        dirty[s] = true;
                    }
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    for (i, known) in entry.into_iter().enumerate() {
        let mut known = known.unwrap_or_default();
        for d in block(i) {
            step(&mut known, d, Some(&mut out));
        }
    }
    out
}

struct DecodedProgram<'a> {
    code: &'a [u8],
    insts: Vec<Decoded>,
    starts: BTreeSet<u64>,
    leaders: BTreeSet<u64>,
    branches: usize,
    gs_jump: Option<Option<u64>>,
    static_samplers: BTreeMap<u64, [u32; 4]>,
    fmt_regs: BTreeSet<u64>,
    max_vgpr: u64,
    loop_budget: LoopBudget,
    long_loops: Vec<String>,
}

fn decode_program<'a>(code: &'a [u8], opt: &Options) -> Result<DecodedProgram<'a>, String> {
    if let (Mode::Mesh(ms), Some(_)) = (&opt.mode, &opt.mesh_native) {
        if ms.threads > if opt.wave64 { 64 } else { 32 } {
            let multi = opt.mesh_multi_wave && pso::mesh_native_host_threads(ms.threads, opt.wave64).is_some();
            if !multi {
                return Err("not a host mesh shader: the group has more than one console wave".into());
            }
        }
        if opt.wave64 && !opt.split_wave64 {
            return Err("not a host mesh shader: the packed entry requires a 32-lane host wave".into());
        }
    }
    let mut insts: Vec<Decoded> = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("+{off:#x}: cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    rdna2::resolve_far_jumps(&mut insts);
    let starts: BTreeSet<u64> = insts.iter().map(|d| d.pc).collect();
    let mut leaders: BTreeSet<u64> = BTreeSet::from([0]);
    let mut branches = 0usize;
    // geometry mode: the ES's s_setpc_b64 continues into the GS code (or ends the ES alone)
    let gs_jump = match &opt.mode {
        Mode::Geometry(g) => Some(g.gs_start.filter(|&s| s > 0)),
        _ => None,
    };
    for d in &insts {
        let next = d.pc + d.len as u64;
        if let (Some(target), "S_SETPC_B64", None) = (gs_jump, d.inst.name, d.far_target) {
            if let Some(t) = target.filter(|t| starts.contains(t)) {
                leaders.insert(t);
                branches += 1;
            }
            leaders.insert(next);
            continue;
        }
        if let Some(t) = rdna2::branch_target(d) {
            if !starts.contains(&t) {
                return Err(format!("+{:#x}: branch to {t:#x}, which is not the start of an instruction", d.pc));
            }
            leaders.insert(t);
            leaders.insert(next);
            branches += 1;
        } else if d.inst.flags & rdna2::F_TERMINATOR != 0 {
            leaders.insert(next);
        }
    }
    let static_samplers = constant_samplers(&insts, &leaders);
    let fmt_regs = emit::format_descriptor_regs(&insts);
    let max_vgpr = insts.iter().flat_map(|d| {
        let (r, w) = rdna2::vgpr_uses(d);
        r.into_iter().chain(w)
    }).max().unwrap_or(0);
    let loop_budget = loop_budget(&insts, &opt.mode);
    let long_loops = long_loops(&insts, loop_budget.of_a_loop());
    Ok(DecodedProgram { code, insts, starts, leaders, branches, gs_jump, static_samplers, fmt_regs, max_vgpr, loop_budget, long_loops })
}

/// The census of the lane-independent form (`lane_independent`): one line per wave64 compute
/// program translated for a GPU with 32-lane waves, saying whether every lane of it computes on
/// its own, or the first instruction that keeps it on the wave form and why; counted in
/// `dxc::stage_summary`. An instrumented translation of a program is not a second program.
fn lane_census(code: &[u8], insts: &[Decoded], opt: &Options) -> Option<lane_independent::Proof> {
    let verdict = lane_independent::classify(insts, opt);
    if matches!(&verdict, Err(r) if r.rule == lane_independent::Rule::Instrumented) {
        // what is profiled or gated is a wave form of the program, not the form it runs in
        if !opt.legacy_compute_wave64 && lane_independent_enabled() && lane_independent::prove(insts).is_ok() {
            crate::xlog_info!("[SHADER] compute code {:016x}: instrumented translation (loop profile, material gate) in a wave form; without the instrument the program runs lane-independent", super::shader::fnv64(code));
        }
        return None;
    }
    // the legacy form translated again for a program that was already counted (`pso`'s packed
    // or legacy choice, the overrides)
    if !opt.legacy_compute_wave64 {
        dxc::note_lane_census(verdict.is_ok());
    }
    let hash = super::shader::fnv64(code);
    match verdict {
        Ok(proof) => {
            crate::xlog_info!("[SHADER] compute code {hash:016x}: lane-independent census: qualifies ({} mask branches, {} first-lane reads proved)", proof.mask_branches, proof.first_lane_reads);
            Some(proof)
        }
        Err(reject) => {
            crate::xlog_info!("[SHADER] compute code {hash:016x}: lane-independent census: keeps the wave form: {reject}");
            None
        }
    }
}

pub fn translate(code: &[u8], opt: &Options) -> Result<Translation, String> {
    emit_program(&decode_program(code, opt)?, opt)
}

fn emit_program(program: &DecodedProgram<'_>, opt: &Options) -> Result<Translation, String> {
    // resolutions kept across blocks (`kept_pages`): a first translation notes the scalar loads,
    // the second follows the plan made from them
    if !(matches!(opt.mode, Mode::Compute(_)) && opt.mem_windows > 1 && emit::keep_pages()) {
        return emit_program_with(program, opt, Default::default(), &mut None, true);
    }
    let mut first = Some(kept_pages::Record::default());
    emit_program_with(program, opt, Default::default(), &mut first, false)?;
    let mut record = first.unwrap_or_default();
    loop {
        let kept = kept_pages::plan(&program.insts, &record);
        // the text that follows the plan answers for itself: a write of the plan that this text
        // lets nothing follow is closed for the next plan, until a plan holds in its own text
        let writes: Vec<u64> = kept.after.keys().copied().collect();
        let mut second = Some(kept_pages::Record::default());
        let translation = emit_program_with(program, opt, kept, &mut second, true)?;
        let closed = second.unwrap_or_default().closed;
        let unfollowed: Vec<u64> = writes.into_iter().filter(|pc| closed.contains(pc)).collect();
        if unfollowed.is_empty() {
            return Ok(translation);
        }
        record.closed.extend(unfollowed);
    }
}

/// `emit_program` with the kept resolutions of `kept`, taking the notes for them in `record` (the
/// loads matter of a translation without a plan, the closed writes of any). `log`: the one
/// translation of a program that says which form it has.
fn emit_program_with(program: &DecodedProgram<'_>, opt: &Options, kept: kept_pages::Plan, record: &mut Option<kept_pages::Record>, log: bool) -> Result<Translation, String> {
    let DecodedProgram { code, insts, starts, leaders, branches, gs_jump, .. } = program;
    let (branches, gs_jump) = (*branches, *gs_jump);
    let mut em_options = opt.clone();
    if matches!(opt.mode, Mode::Compute(_)) {
        // the lane-independent form before the packed and the split one: it needs no wave of
        // the host at all. The legacy override is a request for the split form.
        let proved = opt.wave64 && opt.split_wave64 && lane_census(code, insts, opt).is_some();
        em_options.lane_independent = proved && !opt.legacy_compute_wave64 && lane_independent_enabled();
        let path = match (opt.wave64, opt.split_wave64) {
            (false, _) => "native-wave32",
            (true, false) => "native-wave64",
            (true, true) if em_options.lane_independent => "lane-independent",
            (true, true) if opt.legacy_compute_wave64 => "legacy-wave64 override",
            (true, true) if insts.len() > compute_wave64_pack_max_insts() => {
                em_options.legacy_compute_wave64 = true;
                "legacy-wave64 size cutoff"
            }
            (true, true) => "packed-wave64 candidate",
        };
        if log {
            crate::xlog_info!("[SHADER] compute code {:016x}: {path}, {} guest instructions (pack limit {})", super::shader::fnv64(code), insts.len(), compute_wave64_pack_max_insts());
        }
    } else {
        em_options.lane_independent = false;
    }
    let lane_independent = em_options.lane_independent;
    let mut em = emit::Em::new(em_options);
    em.kept = kept;
    em.kept_record = record.take();
    em.configure_round_modes(insts, branches == 0);
    // xsemu: from xcl-dev/xcl nvidia-dev 9b455a1a,67a08dbe crates/xcl/src/gpu/recomp/mod.rs
    em.writes_lds = emit::lds_order_wanted() && emit::writes_lds(insts);
    em.static_ss = program.static_samplers.clone();
    em.loop_budget = program.loop_budget;
    em.code_loads = opt.code_data.as_ref().is_some_and(|data| !data.words.is_empty())
        && insts.iter().any(|d| d.inst.name == "S_GETPC_B64");
    // the program's blocks and, when the graph is reducible, its structured form
    let leaders: Vec<u64> = leaders.iter().copied().filter(|l| starts.contains(l)).collect();
    let terms: Vec<Term> = leaders
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let e = leaders.get(i + 1).copied().unwrap_or(u64::MAX);
            term_of(insts[..insts.partition_point(|d| d.pc < e)].last().filter(|d| d.pc >= s))
        })
        .collect();
    let structured = if branches > 0 && gs_jump.is_none() && opt.structured && std::env::var_os("XCL_RECOMP_NO_STRUCTURED").is_none() && std::env::var_os("XCL_RECOMP_NO_CFG").is_none() {
        match structure::Cfg::new(&leaders, &terms) {
            Ok(cfg) => {
                let nodes = cfg.layout();
                Some((cfg, nodes))
            }
            Err(e) => {
                why(0, &e);
                None
            }
        }
    } else {
        None
    };
    // where each emitted loop's guard may leave it to
    let guard: x64_masks::GuardExits = match &structured {
        Some((cfg, nodes)) => cfg
            .loop_exits(nodes)
            .into_iter()
            .map(|(h, to)| (cfg.loops[&h].iter().map(|&b| (cfg.start[b], cfg.end[b])).collect(), to.map(|b| cfg.start[b])))
            .collect(),
        None => x64_masks::range_exits(insts),
    };
    if let Some(record) = &mut em.kept_record {
        record.guard = guard.clone();
    }
    // split wave64: masks only read lane by lane skip the halves' exchange (geometry mode
    // continues from the ES into the GS through s_setpc: left as it is)
    if opt.wave64 && opt.split_wave64 && !em.packed_compute() && !lane_independent && matches!(opt.mode, Mode::Kernel | Mode::Compute(_) | Mode::Mesh(_)) && std::env::var_os("XCL_X64_ALL_MASKS").is_none() {
        em.x64_whole_live = x64_masks::by_pc(insts, matches!(opt.mode, Mode::Kernel), &guard);
    }
    em.fmt_regs = program.fmt_regs.clone();
    // a wave64 VS / PS on a GPU that can run 32-lane waves: what lanes 32..63 hold where the
    // program reads them (the driver may still pick 64 lanes: decided at run time); only a
    // program with a read to change gets the form
    if opt.wave64 && opt.split_wave64 && matches!(opt.mode, Mode::Vertex(_) | Mode::Pixel(_)) {
        let a = upper_lanes::analyze(insts);
        if a.reads.values().any(|r| matches!(r, upper_lanes::Read::Const(_) | upper_lanes::Read::Shadow)) {
            em.upper = Some(a);
        }
    }
    // instructions every lane the wave started with runs: no lane guard (a packed wave starts
    // from its valid lanes, the emitter skips it; a wave64 mesh wave starts from 32 lanes; a
    // kernel's EXEC is its input, so only what the program sets is known there)
    if opt.exec_proof && gs_jump.is_none() && !(opt.wave64 && matches!(opt.mode, Mode::Mesh(_))) && std::env::var_os("XCL_RECOMP_NO_EXEC_PROOF").is_none() {
        em.exec_full = lane_guards::full_exec(insts, opt.wave64, !matches!(opt.mode, Mode::Kernel), &guard);
        if em.upper.is_none() && matches!(opt.mode, Mode::Kernel | Mode::Compute(_) | Mode::Vertex(_) | Mode::Pixel(_)) {
            em.loose = lane_guards::loose_writes(insts, &em.exec_full, matches!(opt.mode, Mode::Kernel), &guard);
        }
    }
    em.max_vgpr = program.max_vgpr;
    // `XCL_LDS_START`: what the program's first store leaves in the group's LDS, set before its
    // code (`emit::Em::lds_start_body`)
    if lds_start::lds_start_on() && gs_jump.is_none() && matches!(opt.mode, Mode::Kernel | Mode::Compute(_) | Mode::Mesh(_)) {
        em.lds_start = lds_start::lds_start(insts, &leaders, &terms);
        if log && !em.lds_start.is_empty() {
            crate::xlog_info!("[SHADER] LDS start: {} dword(s) of the group's LDS set before the program's first store (XCL_LDS_START)", em.lds_start.len());
        }
    }
    if let Mode::Pixel(ps) = &opt.mode {
        em.ps_modes = emit::ps_attr_modes(insts, ps.input_ena);
    }
    let mut body = String::new();
    let mut blocks = 0usize;
    let kept_at_entry = em.keep_pages_at_entry();
    if branches == 0 {
        // straight line: stop at the first s_endpgm (the rest is unreachable)
        body.push_str(&kept_at_entry);
        em.indent = 1;
        for (i, d) in insts.iter().enumerate() {
            let next = insts.get(i + 1).map(|n| n.pc).unwrap_or(u64::MAX);
            if let emit::Flow::End = em.inst(d, next) {
                break;
            }
        }
        body.push_str(&em.take());
        blocks = 1;
    } else {
        let loop_profile = loop_profile::Sites::new(insts, opt.loop_profile && matches!(opt.mode, Mode::Compute(_)));
        let edges: Vec<(u64, u64)> = insts.iter().filter_map(|d| rdna2::branch_target(d).map(|t| (d.pc, t))).collect();
        let loops = if structured.is_some() { Vec::new() } else { loop_ranges(insts) };
        let ctx = Ctx { insts, starts, leaders: &leaders, terms: &terms, loops: &loops, loop_profile, loop_nest: Default::default(), edges: &edges, memo: Default::default() };
        let mut code = String::new();
        if let Some((cfg, nodes)) = &structured {
            emit_nodes(&ctx, cfg, &mut em, &mut code, nodes, 1, &mut blocks);
            let mut flags = BTreeSet::new();
            node_flags(nodes, &mut flags);
            for f in flags {
                body.push_str(&format!("    bool {} = false;\n", flag_name(cfg, f)));
            }
        } else {
            // The `pc` chain: blocks in address order as a chain of `if (pc == start)`: a
            // forward branch is found further down the chain, so only ranges spanned by
            // backward branches need a loop (`while (pc in range)`); ranges are merged until
            // they nest. Same semantics as a dispatcher loop over all blocks, but DXC sees an
            // acyclic graph almost everywhere (optimising one big dispatcher loop took minutes
            // on large retail programs).
            let mut flags = BTreeSet::new();
            if gs_jump.is_none() && opt.structured && std::env::var_os("XCL_RECOMP_NO_STRUCTURED").is_none() && structurable(&ctx, 0, u64::MAX).is_some() {
                emit_structured(&ctx, &mut em, &mut code, 0, u64::MAX, None, 0, None, &BTreeSet::new(), 1, &mut blocks, &mut flags);
            } else {
                if gs_jump.is_some() && opt.split_wave64 && opt.wave64 {
                    code.push_str("    uint pc = xcl_entry_pc;\n");
                } else {
                    code.push_str("    uint pc = 0;\n");
                }
                emit_range(&ctx, &mut em, &mut code, 0, u64::MAX, 1, &mut blocks, &mut flags);
            }
            for &t in &flags {
                body.push_str(&format!("    bool {} = false;\n", skip_flag(t)));
            }
        }
        body.push_str(&kept_at_entry);
        body.push_str(&code);
    }
    *record = em.kept_record.take();
    if let Some(why) = em.mesh_native_blocker() {
        return Err(format!("not a host mesh shader: {why}"));
    }
    let lds_start = em.lds_start_body();
    if !lds_start.is_empty() {
        body.insert_str(0, &lds_start);
    }
    let hlsl = em.finish(&body);
    let static_samplers = em.static_used.clone();
    Ok(Translation { hlsl, packed_compute: em.packed_compute(), lane_independent, outline_candidate: em.outline_candidate, unsupported: em.unsupported, unsupported_why: em.unsupported_why, ignored: em.ignored, instructions: insts.len(), blocks, static_samplers, long_loops: program.long_loops.clone(), loop_budget: Some(program.loop_budget) })
}

#[cfg(test)]
mod attribute_tests {
    use super::*;

    fn input(name: &str, index: u32, register: u32, rw_mask: u8) -> SigElement {
        SigElement { name: name.into(), index, system_value: 0, component_type: 3, register, mask: 0xF, rw_mask, stream: 0 }
    }

    /// Fortnite's Slate PS (FSSM sub-version 5): attr0 = COLOR0, attr1 = TEXCOORD1; the table's
    /// layout one sub-version on (6) sits 8 bytes later
    #[test]
    fn ps_attributes_from_the_input_table() {
        let mut pos = input("SV_Position", 0, 0, 0);
        pos.system_value = 1;
        let inputs = vec![pos, input("COLOR", 0, 1, 0xF), input("ORIGINAL_POSITION", 0, 2, 0), input("TEXCOORD", 0, 3, 0), input("TEXCOORD", 1, 4, 3)];
        for (sub, base) in [(5u32, 0x130usize), (6, 0x138)] {
            let mut f = vec![0u8; 0x400];
            f[0x10..0x14].copy_from_slice(&sub.to_le_bytes());
            f[base..base + 16].copy_from_slice(&[6, 1, 0xF, 0, 0, 0, 1, 2, 6, 4, 0xF, 1, 0, 0, 4, 2]);
            f[base + 0x100] = 2;
            f[base + 0x110] = 2;
            let (ena, attrs) = ps_input_table(&f).unwrap();
            assert_eq!((ena, attrs.len(), attrs[1].register), (2, 2, 4));
            let ps = PsInterface { inputs: inputs.clone(), attrs: Some(attrs), ..Default::default() };
            let all = ranked(&ps.inputs, false);
            assert_eq!(all[ps_attribute_slot(&ps, 0, 3).unwrap()].semantic(), "COLOR0");
            assert_eq!(all[ps_attribute_slot(&ps, 1, 0).unwrap()].semantic(), "TEXCOORD1");
            assert_eq!(ps_attribute_slot(&ps, 2, 0), None);
            assert_eq!((0..4).map(|j| ps_slot_attribute(&ps, j)).collect::<Vec<_>>(), [Some(0), None, None, Some(1)]);
        }
        // no table: the attr-th register
        let ps = PsInterface { inputs, ..Default::default() };
        assert_eq!(ps_attribute_slot(&ps, 1, 0), Some(1));
    }

    /// Minecraft's UI vertex shader: an ISG1 input the program never reads (NORMAL0, r1) gets
    /// no fetch-shader VGPRs – POSITION0 (r2) is read in v8, TEXCOORD0 (r3) in v12.
    #[test]
    fn unread_inputs_take_no_fetch_vgprs() {
        let inputs = vec![input("COLOR", 0, 0, 0xF), input("NORMAL", 0, 1, 0), input("POSITION", 0, 2, 7), input("TEXCOORD", 0, 3, 3)];
        assert_eq!(fetched_registers(&inputs), [0, 2, 3]);
        let vs = VsInterface { inputs: inputs.clone(), attr_base: 4, ..Default::default() };
        assert_eq!([0, 1, 2, 3].map(|r| vs.input_vgpr(r)), [Some(4), None, Some(8), Some(12)]);
        // the ES variant starts at v9
        let es = VsInterface { inputs, attr_base: 9, ..Default::default() };
        assert_eq!(es.input_vgpr(3), Some(17));
        // a signature without read masks keeps every register (v[base + 4r])
        let unmarked = vec![input("COLOR", 0, 0, 0), input("NORMAL", 0, 1, 0), input("POSITION", 0, 2, 0)];
        assert_eq!(fetched_registers(&unmarked), [0, 1, 2]);
    }

    /// Elements packed into one register (XDefiant: UV1 in .xy, UV0 in .zw) share a param /
    /// attribute; channel c is read from the element declaring it.
    #[test]
    fn packed_registers_are_one_param() {
        let mut uv1 = input("UV", 1, 0, 3);
        uv1.mask = 0x3;
        let mut uv0 = input("UV", 0, 0, 0xC);
        uv0.mask = 0xC;
        let sig = vec![uv1, uv0, input("COLOR", 0, 1, 0xF), input("NORMAL", 0, 2, 0x7)];
        assert_eq!(param_registers(&sig, true), [0, 1, 2]);
        assert_eq!(elements_at(&sig, true, 0), [0, 1]);
        let ps = PsInterface { inputs: sig.clone(), ..Default::default() };
        assert_eq!((ps_attribute_slot(&ps, 0, 0), ps_attribute_slot(&ps, 0, 3), ps_attribute_slot(&ps, 1, 0)), (Some(0), Some(1), Some(2)));
        assert_eq!((0..4).map(|j| ps_slot_attribute(&ps, j)).collect::<Vec<_>>(), [Some(0), Some(0), Some(1), Some(2)]);
        let vs = VsInterface { outputs: sig, ..Default::default() };
        assert_eq!((vs_param_register(&vs, 1), vs_param_register(&vs, 3)), (Some(1), None));
        // the console's table: a Fortnite VS whose SV_Position register is param 1
        let mut f = vec![0u8; 0x400];
        f[0x10] = 5;
        f[0x1D4..0x1D4 + 18].copy_from_slice(&[6, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 6, 2, 2, 0, 0, 0]);
        f[0x1D4 + 0xC0] = 3;
        assert_eq!(vs_output_table(&f), Some(vec![0, 1, 2]));
        f[0x1D4 + 8] = 7;
        assert_eq!(vs_output_table(&f), None);
    }
}
