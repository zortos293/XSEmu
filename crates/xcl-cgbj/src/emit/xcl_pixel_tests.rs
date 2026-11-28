// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../LICENSE and ../../PROVENANCE.md), so it carries that licence.
//! Tests of pixel programs under the XCL binding on hand-assembled code: the entry, the
//! inputs, the exports and the signatures xsemu links.

use super::{DISAGREE, LinkScope, NOT_COVERED, NOT_WRITTEN, link_inputs, ours};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};
use crate::emit::bitcode::census;
use crate::emit::container::{self, SigElement};
use crate::emit::{HostShader, io, lower, structure};
use crate::native::{HwStage, NativeProgram, StageIo};
use crate::xcl::{BindingScope, XclConfig};
use crate::{ir, resource};

/// `SPI_PS_INPUT_ADDR` bits: POS_X_FLOAT, FRONT_FACE.
const POS_X: u32 = 1 << 8;
const FRONT_FACE: u32 = 1 << 12;
const S_ENDPGM: u32 = 0xBF81_0000;

/// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
fn exp_mrt0(r: u32) -> [u32; 2] {
    [0xF800_180F, r * 0x0101_0101]
}

/// v_mov_b32 v1, s`n`.
const fn v1_from(n: u32) -> u32 {
    0x7E02_0200 | n
}

fn el(name: &str, index: u32, kind: u8, row: Option<u32>, col: u8, cols: u8) -> SigElement {
    SigElement { name: name.into(), index, kind, comp: if kind == 13 { 5 } else { 9 }, interp: 0, row, col, cols, usage: 0, half: false }
}

fn position() -> SigElement {
    el("SV_Position", 0, 3, Some(0), 0, 4)
}

/// A DXIL container with only the output signature `outputs`, as a linked stage.
fn stage(outputs: &[SigElement]) -> Vec<u8> {
    crate::dxbc::build(&[(FourCC::OSG1, &container::signature(outputs, false))])
}

struct Pixel {
    shader: HostShader,
    counts: census::Census,
    inputs: Vec<SigElement>,
    outputs: Vec<SigElement>,
}

/// Whether a container's module declares a wave size (`WaveSize` in its entry metadata). The
/// metadata names the tag only when a shader declares one.
fn declares_wave_size(dxil: &[u8]) -> bool {
    dxil.windows(8).any(|w| w == b"WaveSize")
}

impl Pixel {
    fn count(&self, kind: &str) -> u32 {
        self.counts.counts[census::KINDS.iter().position(|k| *k == kind).expect("a kind")]
    }

    fn row(&self, name: &str, index: u32) -> (Option<u32>, u8, u8) {
        let e = self.inputs.iter().find(|e| e.name == name && e.index == index).expect("an input");
        (e.row, e.col, e.cols)
    }
}

/// Translates a wave32 pixel program with the console signatures `inputs` and one SV_Target0
/// of one column, linked to `links`, under the binding, on a host of 32-lane waves.
fn pixel(words: &[u32], input_addr: u32, inputs: &[SigElement], links: &[&[u8]]) -> Result<Pixel, ShaderError> {
    pixel_at(false, 32, words, input_addr, inputs, links)
}

/// A console container of the pixel program `words` of either wave size, with the signatures
/// `inputs` and one SV_Target0 of one column.
fn console(words: &[u32], wave64: bool, input_addr: u32, inputs: &[SigElement]) -> (Vec<u8>, NativeProgram) {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps { inputs: vec![], input_addr, db_shader_control: 0, patches: vec![] },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let target = el("SV_Target", 0, 16, Some(0), 0, 1);
    let (isg1, osg1) = (container::signature(inputs, true), container::signature(&[target], false));
    (crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]), native)
}

/// The same for a pixel program of either wave size, on a host whose widest wave is
/// `host_wave` lanes.
fn pixel_at(wave64: bool, host_wave: u32, words: &[u32], input_addr: u32, inputs: &[SigElement], links: &[&[u8]]) -> Result<Pixel, ShaderError> {
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let r = pixel_at_host(wave64, words, input_addr, inputs, links);
    lower::TEST_HOST_WAVE.set(was);
    r
}

/// What [`pixel_at`] runs, with the host wave already set.
fn pixel_at_host(wave64: bool, words: &[u32], input_addr: u32, inputs: &[SigElement], links: &[&[u8]]) -> Result<Pixel, ShaderError> {
    let (bytes, native) = console(words, wave64, input_addr, inputs);
    let c = Container::parse(&bytes).unwrap();
    let _scope = BindingScope::enter(XclConfig { thread_limits: false, ..XclConfig::default() });
    let _link = LinkScope::enter(&c, links)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    crate::xcl::clear_stats();
    let shader = crate::emit::emit(&c, &native, &p, &st, &io, &res)?;
    let (_, counts) = crate::xcl::take_stats(&shader.dxil);
    let out = Container::parse(&shader.dxil).unwrap();
    let sig = |fourcc| crate::emit::signature(out.part(fourcc).unwrap().data).unwrap();
    let (inputs, outputs) = (sig(FourCC::ISG1), sig(FourCC::OSG1));
    Ok(Pixel { shader, counts, inputs, outputs })
}

#[test]
fn a_pixel_program_reads_its_position_and_exports_with_nothing_bound_but_io() {
    let [e0, e1] = exp_mrt0(0);
    let p = pixel(&[e0, e1, S_ENDPGM], POS_X, &[position()], &[]).unwrap();
    // No thread limits and no memory: no load at all, and only io bound (u0).
    assert_eq!(p.count("buffer_load"), 0);
    assert_eq!(p.shader.root_uavs, vec![0]);
    assert!(p.shader.user_data.is_empty() && p.shader.root_cbvs.is_empty() && p.shader.root_srvs.is_empty());
    assert_eq!(p.row("SV_Position", 0), (Some(0), 0, 4));
    // The colour output is four columns wide, as xsemu's own translator declares it.
    assert_eq!(p.outputs.iter().map(|o| (o.kind, o.row, o.col, o.cols)).collect::<Vec<_>>(), [(16, Some(0), 0, 4)]);
}

#[test]
fn a_pixel_entry_sgpr_is_its_io_dword() {
    let [e0, e1] = exp_mrt0(1);
    let p = pixel(&[v1_from(3), e0, e1, S_ENDPGM], 0, &[position()], &[]).unwrap();
    assert_eq!(p.count("buffer_load"), 1);
    assert_eq!(p.shader.root_uavs, vec![0]);
}

#[test]
fn inputs_take_the_rows_of_the_vertex_outputs_they_link_to() {
    let [e0, e1] = exp_mrt0(0);
    // Console rows: TEXCOORD0 in row 1 (xy) and TEXCOORD1 beside it (zw), COLOR in row 2.
    let inputs = [position(), el("TEXCOORD", 0, 0, Some(1), 0, 2), el("TEXCOORD", 1, 0, Some(1), 2, 2), el("COLOR", 0, 0, Some(2), 0, 4)];
    let code = [e0, e1, S_ENDPGM];
    // Without links: xsemu's own pixel layout, a row each in the console's order.
    let p = pixel(&code, POS_X, &inputs, &[]).unwrap();
    assert_eq!([p.row("TEXCOORD", 0), p.row("TEXCOORD", 1), p.row("COLOR", 0)], [(Some(1), 0, 2), (Some(2), 2, 2), (Some(3), 0, 4)]);
    // Behind a vertex program that writes them in other rows, and more besides.
    let vs = stage(&[position(), el("NORMAL", 0, 0, Some(1), 0, 4), el("TEXCOORD", 1, 0, Some(2), 0, 4), el("color", 0, 0, Some(3), 0, 4), el("TEXCOORD", 0, 0, Some(4), 0, 4)]);
    let p = pixel(&code, POS_X, &inputs, &[&vs]).unwrap();
    assert_eq!(p.row("SV_Position", 0), (Some(0), 0, 4));
    assert_eq!([p.row("TEXCOORD", 0), p.row("TEXCOORD", 1), p.row("COLOR", 0)], [(Some(4), 0, 2), (Some(2), 2, 2), (Some(3), 0, 4)]);
    // A rect-list GS behind it with the same rows changes nothing.
    let p = pixel(&code, POS_X, &inputs, &[&vs, &vs]).unwrap();
    assert_eq!(p.row("TEXCOORD", 0), (Some(4), 0, 2));
}

#[test]
fn a_front_face_no_output_has_gets_a_row_after_them() {
    // v_mov_b32 v1, v0 (the front face, the only input VGPR), exported.
    let [e0, e1] = exp_mrt0(1);
    let inputs = [position(), el("TEXCOORD", 0, 0, Some(1), 0, 4), el("SV_IsFrontFace", 0, 13, None, 0, 1)];
    let vs = stage(&[position(), el("TEXCOORD", 0, 0, Some(5), 0, 4)]);
    let p = pixel(&[0x7E02_0300, e0, e1, S_ENDPGM], FRONT_FACE, &inputs, &[&vs]).unwrap();
    assert_eq!(p.row("TEXCOORD", 0), (Some(5), 0, 4));
    assert_eq!(p.row("SV_IsFrontFace", 0), (Some(6), 0, 1));
}

#[test]
fn inputs_that_cannot_link_are_refused() {
    let [e0, e1] = exp_mrt0(0);
    let code = [e0, e1, S_ENDPGM];
    let inputs = [position(), el("TEXCOORD", 0, 0, Some(1), 0, 4)];
    let refused = |links: &[&[u8]]| match pixel(&code, POS_X, &inputs, links) {
        Err(ShaderError::Unsupported { what, .. }) => what,
        other => panic!("{:?}", other.map(|p| p.inputs)),
    };
    let without = stage(&[position(), el("NORMAL", 0, 0, Some(1), 0, 4)]);
    assert_eq!(refused(&[&without]), NOT_WRITTEN);
    let narrow = stage(&[position(), el("TEXCOORD", 0, 0, Some(1), 0, 2)]);
    assert_eq!(refused(&[&narrow]), NOT_COVERED);
    let (a, b) = (stage(&[position(), el("TEXCOORD", 0, 0, Some(1), 0, 4)]), stage(&[position(), el("TEXCOORD", 0, 0, Some(2), 0, 4)]));
    assert_eq!(refused(&[&a, &b]), DISAGREE);
    assert!(pixel(&code, POS_X, &inputs, &[&a]).is_ok());
}

#[test]
fn our_layout_puts_clip_distances_after_the_user_inputs() {
    let inputs = [el("SV_ClipDistance", 0, 6, Some(0), 2, 2), position(), el("B", 0, 0, Some(3), 0, 1), el("A", 0, 0, Some(1), 1, 3)];
    let rows: Vec<(String, Option<u32>, u8, u8)> = ours(&inputs).into_iter().map(|e| (e.name, e.row, e.col, e.cols)).collect();
    let want = [("SV_Position", 0, 0, 4), ("A", 1, 0, 4), ("B", 2, 0, 4), ("SV_ClipDistance", 3, 0, 4)];
    assert_eq!(rows, want.map(|(n, r, c, w)| (n.to_string(), Some(r), c, w)));
    let mut linked = inputs.to_vec();
    link_inputs(&mut linked, &[ours(&inputs)]).unwrap();
    assert_eq!(linked.iter().map(|e| (e.row, e.col)).collect::<Vec<_>>(), [(Some(3), 2), (Some(0), 0), (Some(2), 0), (Some(1), 1)]);
}

#[test]
fn pixel_entry_points_take_their_own_stage_only() {
    let cfg = XclConfig::default();
    assert_eq!(crate::xcl::translate_pixel(&[], &cfg, &[]), Err(ShaderError::NotDxbc));
    let one = XclConfig { mem_windows: 1, ..cfg };
    assert!(matches!(crate::xcl::translate_pixel(&[], &one, &[]), Err(ShaderError::Unsupported { .. })));
}

/// s_mov_b32 s`n`, a 32-bit literal.
fn s_mov(n: u32, value: u32) -> [u32; 2] {
    [0xBE80_03FF | n << 16, value]
}

#[test]
fn pixel_programs_take_samplers_and_image_classes_by_the_compute_rules() {
    let [e0, e1] = exp_mrt0(1);
    // s_brev_b32 s6, 40 and s_brev_b32 s7, 0x1 before image_sample v[1:4], v[0:1], s[8:15], s[4:7]:
    // the sampler is static, as in a compute program.
    let sample = |last: &[u32]| {
        let mut words: Vec<u32> = [s_mov(4, 0x92), s_mov(5, 0x00C0_0000)].concat();
        words.extend([0xBE80_0BA8 | 6 << 16]);
        words.extend(last);
        words.extend([0xF09C_0F08, 0x0022_0100, e0, e1, S_ENDPGM]);
        pixel(&words, POS_X, &[position()], &[]).unwrap()
    };
    assert_eq!(sample(&[0xBE80_0BFF | 7 << 16, 1]).shader.samplers, vec![[0x92, 0x00C0_0000, 0x1400_0000, 0x8000_0000]]);
    // s_not_b32 s7, 0 is not followed: the sampler comes from the heap.
    assert!(sample(&[0xBE80_0780 | 7 << 16]).shader.samplers.is_empty());
    // image_store v[1:4], v[0:1], s[8:15] to a T# of 32_UINT keeps the registers' bits; one of
    // 32_FLOAT converts them.
    let store = |format: u32| {
        let mut words: Vec<u32> = (1..5).map(|k| 0x7E00_0200 | k << 17 | (k - 1)).collect();
        words.extend(s_mov(8, 3));
        words.extend(s_mov(9, format << 20));
        words.extend([0xF020_0F08, 0x0002_0100, e0, e1, S_ENDPGM]);
        pixel(&words, POS_X, &[position()], &[]).unwrap().count("bitcast")
    };
    assert!(store(20) < store(22), "{} {}", store(20), store(22));
}

#[test]
fn a_branch_on_the_scc_of_a_lane_mask_asks_the_host_wave() {
    let [e0, e1] = exp_mrt0(1);
    // v_cmp_lt_f32 vcc_lo, 0, v0; s_and_b32 s0, vcc_lo, exec_lo; s_cbranch_scc0 over
    // v_mov_b32 v1, 1.0: the wave skips the block only when no lane's bit is set.
    let words = [0x7C02_0080, 0x8700_7E6A, 0xBF84_0001, 0x7E02_02F2, e0, e1, S_ENDPGM];
    let p = pixel(&words, POS_X, &[position()], &[]).unwrap();
    assert!(p.count("wave_op") >= 1);
}

#[test]
fn the_wave_takes_the_discard_idiom_and_holds_the_exports_before_the_last() {
    // s_mov_b32 s0, exec_lo; v_cmpx_gt_f32 0, v0; s_andn2_b32 s0, s0, exec_lo;
    // s_cbranch_scc0 to the end; s_and_b32 exec_lo, exec_lo, s0; exp mrt0 v0; exp mrt0 v0 done.
    // The end: s_mov_b32 exec_lo, 0; exp null done.
    let words = [
        0xBE80_037E, 0x7C28_0080, 0x8A00_7E00, 0xBF84_0006, 0x877E_007E, 0xF800_100F, 0, 0xF800_180F, 0, S_ENDPGM,
        0xBEFE_0380, 0xF800_1890, 0, S_ENDPGM,
    ];
    let p = pixel(&words, POS_X, &[position()], &[]).unwrap();
    assert!(p.count("wave_op") >= 1);
}

#[test]
fn a_wave64_pixel_program_is_one_container_at_either_host_wave_and_takes_xsemu_layout() {
    let [e0, e1] = exp_mrt0(0);
    let words = [e0, e1, S_ENDPGM];
    let on32 = pixel_at(true, 32, &words, POS_X, &[position()], &[]).unwrap();
    let on64 = pixel_at(true, 64, &words, POS_X, &[position()], &[]).unwrap();
    // D3D12 has no `WaveSize` for a pixel shader: the driver picks the host wave, and the
    // lowered code does not depend on which size it picked.
    assert_eq!(on32.shader.dxil, on64.shader.dxil);
    assert!(!declares_wave_size(&on64.shader.dxil));
    assert_eq!(on64.row("SV_Position", 0), (Some(0), 0, 4));
    assert_eq!(on64.outputs.iter().map(|o| (o.kind, o.row, o.col, o.cols)).collect::<Vec<_>>(), [(16, Some(0), 0, 4)]);
    assert_eq!(on64.count("buffer_load"), 0);
    assert_eq!(on64.shader.root_uavs, vec![0]);
}

#[test]
fn a_wave32_pixel_program_needs_a_host_of_32_lane_waves() {
    for (lanes, wave64, runs) in [(32, true, true), (64, true, true), (32, false, true), (64, false, false)] {
        let was = lower::TEST_HOST_WAVE.replace(Some(lanes));
        assert_eq!(super::pixel_runs_in_host_waves(wave64), runs, "{lanes} lanes, wave64 {wave64}");
        lower::TEST_HOST_WAVE.set(was);
    }
    let [e0, e1] = exp_mrt0(0);
    let words = [e0, e1, S_ENDPGM];
    assert!(pixel_at(false, 32, &words, POS_X, &[position()], &[]).is_ok());
}
