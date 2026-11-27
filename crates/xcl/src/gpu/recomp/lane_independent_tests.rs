//! The lane-independent form's classifier (`lane_independent::classify`): programs of every shape the
//! form is meant for qualify, and one planted instruction per rule ends the proof at that
//! instruction for that reason. No GPU and no DXC needed.

use std::collections::HashMap;

use super::lane_independent::{classify, Proof, Reject, Rule};
use super::{CsInterface, Mode, Options};
use crate::gpu::rdna2::encode;

/// One line of a test program: an instruction (with its literal), a label, or a branch to a label.
#[derive(Clone)]
pub(super) enum A {
    I(&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>),
    L(&'static str),
    B(&'static str, &'static str),
}

pub(super) const V: u64 = 256;
pub(super) const VCC: u64 = 106;
pub(super) const EXEC: u64 = 126;
/// the inline constants 0 and -1
pub(super) const ZERO: u64 = 128;
pub(super) const ONES: u64 = 193;

pub(super) fn i(name: &'static str, enc: &'static str, f: &[(&'static str, u64)]) -> A {
    A::I(name, enc, f.to_vec(), None)
}

/// The program's bytes (an `s_endpgm` after the last line) and the address of each line.
pub(super) fn asm_at(items: &[A]) -> (Vec<u8>, Vec<u64>) {
    let size = |it: &A| match it {
        A::L(_) => 0,
        A::I(n, e, f, lit) => encode(n, e, f, *lit).unwrap_or_else(|x| panic!("{n}: {x}")).len() as u64,
        A::B(..) => 4,
    };
    let mut labels = HashMap::new();
    let mut at = Vec::new();
    let mut pc = 0u64;
    for it in items {
        if let A::L(l) = it {
            labels.insert(*l, pc);
        }
        at.push(pc);
        pc += size(it);
    }
    let mut out = Vec::new();
    for it in items {
        match it {
            A::L(_) => {}
            A::I(n, e, f, lit) => out.extend(encode(n, e, f, *lit).unwrap()),
            A::B(n, l) => {
                let off = ((labels[l] as i64 - (out.len() as i64 + 4)) / 4) as i16 as u16 as u64;
                out.extend(encode(n, "ENC_SOPP", &[("SIMM16", off)], None).unwrap());
            }
        }
    }
    out.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    (out, at)
}

pub(super) fn asm(items: &[A]) -> Vec<u8> {
    asm_at(items).0
}

/// A wave64 compute program for a GPU with 32-lane waves: `threads` in a row, eight user SGPRs.
pub(super) fn options(threads: [u32; 3]) -> Options {
    let cs = CsInterface { threads, user_sgprs: 8, tgid: [false; 3], tg_size: false, tid_components: 3 };
    Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, ..Default::default() }
}

fn verdict_with(items: &[A], opt: &Options) -> Result<Proof, Reject> {
    let code = asm(items);
    classify(&super::decode_program(&code, opt).unwrap().insts, opt)
}

fn verdict(items: &[A]) -> Result<Proof, Reject> {
    verdict_with(items, &options([64, 1, 1]))
}

fn qualifies(items: &[A]) -> Proof {
    verdict(items).unwrap_or_else(|r| panic!("{r}"))
}

/// `before` and `after` qualify together; with `plant` between them the proof ends at the
/// planted line `at` (the first, unless the rule is about a later one). Returns the reason.
fn planted_at(before: &[A], plant: &[A], at: usize, after: &[A]) -> Reject {
    qualifies(&[before, after].concat());
    let all = [before, plant, after].concat();
    let reject = verdict(&all).expect_err("the planted instruction must end the proof");
    assert_eq!(reject.pc, Some(asm_at(&all).1[before.len() + at]), "{reject}");
    reject
}

fn planted(before: &[A], plant: &[A], after: &[A]) -> Rule {
    planted_at(before, plant, 0, after).rule
}

// ---------------------------------------------------------------- building blocks

/// v[d] = v[a] + v[b]
pub(super) fn vadd(d: u64, a: u64, b: u64) -> A {
    i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)])
}

/// v[d] = s (a scalar source code: a register, an inline constant)
pub(super) fn vmov(d: u64, s: u64) -> A {
    i("V_MOV_B32", "ENC_VOP1", &[("VDST", d), ("SRC0", s)])
}

/// vcc = k > v[a] (unsigned; k an inline constant 0..64), active lanes
pub(super) fn vbelow(a: u64, k: u64) -> A {
    i("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 128 + k), ("VSRC1", a)])
}

/// s[d : d + 1] = v[a] < v[b] (unsigned), active lanes
pub(super) fn vless_to(d: u64, a: u64, b: u64) -> A {
    i("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", d), ("SRC0", V + a), ("SRC1", V + b)])
}

pub(super) fn sop2(name: &'static str, d: u64, a: u64, b: u64) -> A {
    i(name, "ENC_SOP2", &[("SDST", d), ("SSRC0", a), ("SSRC1", b)])
}

pub(super) fn sop1(name: &'static str, d: u64, a: u64) -> A {
    i(name, "ENC_SOP1", &[("SDST", d), ("SSRC0", a)])
}

pub(super) fn sopc(name: &'static str, a: u64, b: u64) -> A {
    i(name, "ENC_SOPC", &[("SSRC0", a), ("SSRC1", b)])
}

/// mem[s[4:7] + 4 * v0] = v[data]: the thread's own dword of the buffer in the user SGPRs
pub(super) fn store(data: u64) -> A {
    i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", data), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1)])
}

/// if (v0 < 8) { v3 = v0 + v0 } in the LLVM shape: the lanes that pass in EXEC, the rest back
/// at `join`
pub(super) fn if_then() -> (Vec<A>, Vec<A>) {
    let head = vec![vbelow(0, 8), sop1("S_AND_SAVEEXEC_B64", 10, VCC), A::B("S_CBRANCH_EXECZ", "join"), vadd(3, 0, 0)];
    let tail = vec![A::L("join"), sop2("S_OR_B64", EXEC, EXEC, 10), store(3)];
    (head, tail)
}

/// do { v3 += v1 } while (v3 < v0), lane by lane: a lane leaves EXEC when its own count is
/// reached, the wave goes round while any lane is left
pub(super) fn lane_loop(latch: &'static str) -> Vec<A> {
    let mut p = vec![vmov(3, ZERO), vmov(1, 129), sop1("S_MOV_B64", 12, EXEC), A::L("loop"), vadd(3, 3, 1)];
    p.push(i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 0)]));
    p.push(sop2("S_AND_B64", EXEC, EXEC, VCC));
    p.push(A::B(latch, "loop"));
    p.extend([sop1("S_MOV_B64", EXEC, 12), store(3)]);
    p
}

// ---------------------------------------------------------------- programs that qualify

#[test]
fn straight_line_code_qualifies() {
    assert_eq!(qualifies(&[vadd(3, 0, 0), vmov(4, 5), store(3)]), Proof::default());
}

#[test]
fn a_compare_into_saveexec_and_a_branch_qualifies() {
    let (head, tail) = if_then();
    assert_eq!(qualifies(&[head, tail].concat()).mask_branches, 1);
}

/// The LLVM shape of if / else: the else lanes kept in an SGPR pair, each side skipped when
/// none of its lanes is left.
#[test]
fn if_else_qualifies() {
    let p = [
        vbelow(0, 8),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        sop2("S_XOR_B64", 10, EXEC, 10),
        A::B("S_CBRANCH_EXECZ", "else"),
        vadd(3, 0, 0),
        A::L("else"),
        sop1("S_OR_SAVEEXEC_B64", 12, 10),
        sop2("S_XOR_B64", EXEC, EXEC, 12),
        A::B("S_CBRANCH_EXECZ", "join"),
        vmov(3, 135),
        A::L("join"),
        sop2("S_OR_B64", EXEC, EXEC, 12),
        store(3),
    ];
    assert_eq!(qualifies(&p).mask_branches, 2);
}

#[test]
fn nested_masks_qualify() {
    let p = [
        vbelow(0, 40),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        A::B("S_CBRANCH_EXECZ", "outer"),
        vadd(3, 0, 0),
        vbelow(0, 8),
        sop1("S_AND_SAVEEXEC_B64", 12, VCC),
        A::B("S_CBRANCH_VCCZ", "inner"),
        vadd(3, 3, 0),
        A::L("inner"),
        sop2("S_OR_B64", EXEC, EXEC, 12),
        vadd(4, 3, 0),
        A::L("outer"),
        sop2("S_OR_B64", EXEC, EXEC, 10),
        store(3),
        store(4),
    ];
    assert_eq!(qualifies(&p).mask_branches, 2);
}

#[test]
fn loops_with_a_lane_exit_mask_qualify() {
    for latch in ["S_CBRANCH_EXECNZ", "S_CBRANCH_SCC1", "S_CBRANCH_VCCNZ"] {
        assert_eq!(qualifies(&lane_loop(latch)).mask_branches, 1, "{latch}");
    }
    // the lanes that are done collected in an SGPR pair that starts as no lane, EXEC restored
    // from it after the loop
    let p = [
        vmov(3, ZERO),
        sop1("S_MOV_B64", 14, ZERO),
        A::L("loop"),
        vadd(3, 3, 0),
        i("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 1)]),
        sop2("S_OR_B64", 14, VCC, 14),
        sop2("S_ANDN2_B64", EXEC, EXEC, 14),
        A::B("S_CBRANCH_EXECNZ", "loop"),
        sop2("S_OR_B64", EXEC, EXEC, 14),
        store(3),
    ];
    assert_eq!(qualifies(&p).mask_branches, 1);
}

/// EXEC put back from a copy made before the branch, inside the code a lane's thread skips: the
/// lane is on again there, which nothing reads before EXEC is set again where the thread went.
#[test]
fn exec_restored_from_a_copy_qualifies() {
    let before = [vbelow(0, 8), sop1("S_MOV_B64", 10, EXEC), sop2("S_AND_B64", EXEC, EXEC, VCC), A::B("S_CBRANCH_EXECZ", "join"), vadd(3, 0, 0), sop1("S_MOV_B64", EXEC, 10), A::L("join")];
    let after = [sop1("S_MOV_B64", EXEC, 10), store(3)];
    assert_eq!(qualifies(&[&before[..], &after[..]].concat()).mask_branches, 1);
    // a vector instruction before that: the lane works in the wave and not in its thread
    let reject = verdict(&[&before[..], &[vadd(4, 0, 0)], &after[..]].concat()).unwrap_err();
    assert_eq!((reject.rule, reject.detail.as_str()), (Rule::SoloScalar, "exec"), "{reject}");
}

/// A loop every lane runs the same number of times (a scalar counter) with lane masks inside:
/// the counter is the same number in every thread.
#[test]
fn a_counted_loop_around_masks_qualifies() {
    let (head, tail) = if_then();
    let mut p = vec![i("S_MOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 3)]), A::L("again")];
    p.extend(head);
    p.extend(tail);
    p.extend([sop2("S_ADD_I32", 20, 20, ONES), sopc("S_CMP_GT_I32", 20, ZERO), A::B("S_CBRANCH_SCC1", "again"), vmov(5, 20), store(5)]);
    assert_eq!(qualifies(&p).mask_branches, 1);
}

#[test]
fn a_select_on_a_compare_qualifies() {
    let vop2 = [vbelow(0, 8), i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 1)]), store(3)];
    qualifies(&vop2);
    let vop3 = [vless_to(16, 0, 1), i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("SRC2", 16)]), store(3)];
    qualifies(&vop3);
    // a carry from one add into the next
    let carry = [
        i("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 3), ("SDST", VCC), ("SRC0", V), ("SRC1", V + 1)]),
        i("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 1), ("VSRC1", 2)]),
        store(4),
    ];
    qualifies(&carry);
}

/// Every lane switched on for a stretch (`s_or_saveexec_b64 s, -1`), EXEC set to all lanes
/// and to none, a mask register cleared, and masks compared with 0 and with -1.
#[test]
fn the_all_lanes_and_no_lanes_constants_qualify() {
    let (head, tail) = if_then();
    let whole = [vec![sop1("S_OR_SAVEEXEC_B64", 14, ONES), vmov(5, 131), sop1("S_MOV_B64", EXEC, 14)], head, tail].concat();
    assert_eq!(qualifies(&whole).mask_branches, 1);
    qualifies(&[sop1("S_MOV_B64", 14, EXEC), sop1("S_MOV_B64", EXEC, ZERO), vmov(5, 131), sop1("S_MOV_B64", EXEC, ONES), vmov(6, 131), sop1("S_MOV_B64", EXEC, 14)]);
    // skip the fix-up of the lanes that fail when every lane passes
    let all = [
        vless_to(16, 0, 1),
        sop1("S_MOV_B64", 14, EXEC),
        sopc("S_CMP_EQ_U64", 16, ONES),
        A::B("S_CBRANCH_SCC1", "skip"),
        sop2("S_ANDN2_B64", EXEC, EXEC, 16),
        vmov(3, 135),
        A::L("skip"),
        sop1("S_MOV_B64", EXEC, 14),
        store(3),
    ];
    assert_eq!(qualifies(&all).mask_branches, 1);
    // skip the work when no lane asks for it
    let none = [
        vless_to(16, 0, 1),
        sopc("S_CMP_LG_U64", 16, ZERO),
        A::B("S_CBRANCH_SCC0", "skip"),
        sop1("S_AND_SAVEEXEC_B64", 14, 16),
        vmov(3, 135),
        sop1("S_MOV_B64", EXEC, 14),
        A::L("skip"),
        store(3),
    ];
    assert_eq!(qualifies(&none).mask_branches, 1);
}

/// Lanes outside the image leave at the start; the wave goes on to the end of the program,
/// wherever that is.
#[test]
fn an_early_out_qualifies() {
    let p = [vbelow(0, 8), sop1("S_AND_SAVEEXEC_B64", 10, VCC), A::B("S_CBRANCH_EXECZ", "out"), vadd(3, 0, 0), store(3), i("S_ENDPGM", "ENC_SOPP", &[]), A::L("out")];
    assert_eq!(qualifies(&p).mask_branches, 1);
}

/// `v_readfirstlane_b32` of a register every lane got the same number in.
#[test]
fn first_lane_of_a_wave_uniform_value_qualifies() {
    let read = |s: u64, v: u64| i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", s), ("SRC0", V + v)]);
    // from a user SGPR and a constant, through vector arithmetic, read inside a lane's branch
    let (head, tail) = if_then();
    let p = [vec![vmov(5, 2), vmov(6, 131), vadd(7, 5, 6)], head, vec![read(20, 7), vmov(8, 20)], tail, vec![store(8)]].concat();
    assert_eq!(qualifies(&p), Proof { mask_branches: 1, first_lane_reads: 1 });
}

// ---------------------------------------------------------------- one rule each

#[test]
fn cross_lane_instructions_end_the_proof() {
    let (head, tail) = if_then();
    let plants = [
        i("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 4 << 2)]),
        i("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 0), ("DATA0", 1)]),
        i("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", ONES), ("SRC1", ZERO)]),
        i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 131)]),
        i("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 2), ("SRC1", 131)]),
        i("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
        i("V_ADD_NC_U32", "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 0), ("VSRC1", 1), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
        // the level of detail from the neighbouring lanes
        i("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 0), ("SSAMP", 1), ("DMASK", 1), ("DIM", 1)]),
    ];
    for plant in plants {
        assert_eq!(planted(&head, &[plant], &tail), Rule::CrossLane);
    }
    // sampling a given level is the lane's own
    let lz = i("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 0), ("SSAMP", 1), ("DMASK", 1), ("DIM", 1)]);
    qualifies(&[head, vec![lz], tail].concat());
}

/// A mask as a number: counted, searched, added, shifted, compared with another mask, moved
/// into a vector register (from where it could be stored).
#[test]
fn a_mask_read_as_a_number_ends_the_proof() {
    let (head, tail) = if_then();
    let plants = [
        sop1("S_BCNT1_I32_B64", 20, EXEC),
        sop1("S_FF1_I32_B64", 20, VCC),
        sop1("S_FF1_I32_B64", 20, 10),
        sop2("S_ADD_U32", 20, EXEC, 129),
        sop2("S_LSHR_B64", 20, 10, 129),
        sop1("S_MOV_B32", 20, VCC),
        sop2("S_AND_B32", 20, 10, 11),
        sopc("S_CMP_EQ_U64", EXEC, VCC),
        sopc("S_CMP_LG_U64", 10, 129),
        sopc("S_CMP_EQ_U32", EXEC, ZERO),
        vmov(6, VCC),
        vmov(6, 11),
    ];
    for plant in plants {
        assert_eq!(planted(&head, &[plant], &tail), Rule::MaskAsData);
    }
}

#[test]
fn a_lane_set_from_a_number_ends_the_proof() {
    let (head, tail) = if_then();
    // constants: lane 0 alone, the low 16 lanes
    assert_eq!(planted(&head, &[sop1("S_MOV_B64", EXEC, 129)], &tail), Rule::ConstantLaneSet);
    assert_eq!(planted(&head, &[A::I("S_AND_B64", "ENC_SOP2", vec![("SDST", EXEC), ("SSRC0", EXEC), ("SSRC1", 255)], Some(0xFFFF))], &tail), Rule::ConstantLaneSet);
    assert_eq!(planted(&head, &[i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V), ("SRC1", V + 1), ("SRC2", 129)])], &tail), Rule::ConstantLaneSet);
    // user data, arithmetic
    assert_eq!(planted(&head, &[sop2("S_AND_B64", EXEC, EXEC, 2)], &tail), Rule::DataAsMask);
    assert_eq!(planted(&head, &[sop1("S_MOV_B64", EXEC, 2)], &tail), Rule::DataAsMask);
    assert_eq!(planted(&head, &[sop2("S_LSHL_B64", EXEC, 2, 129)], &tail), Rule::DataAsMask);
    assert_eq!(planted(&head, &[sop1("S_MOV_B32", EXEC, ONES)], &tail), Rule::DataAsMask);
    assert_eq!(planted(&head, &[i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V), ("SRC1", V + 1), ("SRC2", 2)])], &tail), Rule::DataAsMask);
    // a register that is a mask on one path only
    let mixed = [sopc("S_CMP_EQ_U32", 2, ZERO), A::B("S_CBRANCH_SCC1", "over"), sop1("S_MOV_B64", 16, 4), A::L("over"), sop2("S_AND_B64", EXEC, EXEC, 16)];
    assert_eq!(planted_at(&[head.clone(), vec![sop1("S_MOV_B64", 16, EXEC)]].concat(), &mixed, 4, &tail).rule, Rule::MixedRegister);
}

/// A wave-wide test of a mask in arithmetic, and one the proof has lost the mask of.
#[test]
fn mask_tests_outside_branches_end_the_proof() {
    let (head, tail) = if_then();
    let test = sop2("S_AND_B64", 16, VCC, EXEC);
    assert_eq!(planted_at(&head, &[test.clone(), sop2("S_CSELECT_B32", 20, 2, 3)], 1, &tail).rule, Rule::MaskTestAsData);
    assert_eq!(planted_at(&head, &[test.clone(), sop2("S_ADDC_U32", 20, 2, 3)], 1, &tail).rule, Rule::MaskTestAsData);
    assert_eq!(planted_at(&head, &[test.clone(), sop2("S_ADD_U32", 20, 253, 3)], 1, &tail).rule, Rule::MaskTestAsData);
    // the tested mask written again before the branch
    let lost = [test.clone(), sop1("S_MOV_B64", 16, ZERO), A::B("S_CBRANCH_SCC1", "join")];
    assert_eq!(planted_at(&head, &lost, 2, &tail).rule, Rule::OpaqueBranch);
    // tested without a destination
    let null = [sop2("S_AND_B64", 125, VCC, EXEC), A::B("S_CBRANCH_SCC1", "join")];
    assert_eq!(planted_at(&head, &null, 1, &tail).rule, Rule::OpaqueBranch);
}

#[test]
fn lds_and_barriers_end_the_proof() {
    let (head, tail) = if_then();
    assert_eq!(planted(&head, &[i("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)])], &tail), Rule::Lds);
    assert_eq!(planted(&head, &[i("DS_READ_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 1)])], &tail), Rule::Lds);
    assert_eq!(planted(&head, &[i("S_BARRIER", "ENC_SOPP", &[])], &tail), Rule::Barrier);
    // a buffer access redirected to LDS
    let buffer = |lds: u64| i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1), ("LDS", lds)]);
    assert_eq!(planted(&head, &[buffer(1)], &tail), Rule::Lds);
    qualifies(&[head, vec![buffer(0)], tail].concat());
}

/// A scalar load is one for the wave and one per thread in the form: the same number only
/// from memory no thread is writing.
#[test]
fn a_scalar_load_of_shared_memory_ends_the_proof() {
    let (head, tail) = if_then();
    let load = |flag: &'static str| i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("SOFFSET", 125), ("OFFSET", 4), (flag, 1)]);
    assert_eq!(planted(&head, &[load("GLC")], &tail), Rule::SharedScalar);
    assert_eq!(planted(&head, &[load("DLC")], &tail), Rule::SharedScalar);
    assert_eq!(planted(&head, &[i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 2), ("SOFFSET", 125), ("GLC", 1)])], &tail), Rule::SharedScalar);
    let plain = i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("SOFFSET", 125), ("OFFSET", 4)]);
    // the lanes' own loads and atomics of such memory are theirs either way
    let lane = i("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 6), ("ADDR", 0), ("SADDR", 0x7D), ("GLC", 1)]);
    qualifies(&[head, vec![plain, lane], tail].concat());
}

#[test]
fn first_lane_of_a_lane_value_ends_the_proof() {
    let read = |s: u64, v: u64| i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", s), ("SRC0", V + v)]);
    let (head, tail) = if_then();
    // the thread id; a sum with it
    assert_eq!(planted(&head, &[read(20, 0)], &tail), Rule::FirstLane);
    assert_eq!(planted_at(&[vec![vmov(5, 2)], head.clone()].concat(), &[vadd(5, 5, 0), read(20, 5)], 1, &tail).rule, Rule::FirstLane);
    // written from a number, but only in the lanes of the branch: the others keep what they had
    assert_eq!(planted_at(&head, &[vmov(5, 2), read(20, 5)], 1, &tail).rule, Rule::FirstLane);
    // selected by a mask
    let select = [vmov(5, 2), vmov(6, 3), vbelow(0, 8), i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 6)]), read(20, 5)];
    assert_eq!(planted_at(&[], &select, 4, &[head, tail].concat()).rule, Rule::FirstLane);
}

/// A register is the same number in every lane only after an operation that writes it whole
/// from such numbers: not after one that keeps bits of what it held, swaps it, or writes past
/// the registers it names.
#[test]
fn first_lane_of_what_an_instruction_leaves_ends_the_proof() {
    let read = |s: u64, v: u64| i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", s), ("SRC0", V + v)]);
    let uniform = [vmov(5, 2), vmov(6, 3)];
    // v[dst] after `op` on the wave-uniform v5 and v6, read by its first lane
    let after = |op: A, dst: u64| verdict(&[&uniform[..], &[op, read(20, dst), vmov(8, 20), store(8)]].concat());
    let vop2 = |name: &'static str, dst: u64| i(name, "ENC_VOP2", &[("VDST", dst), ("SRC0", V + 5), ("VSRC1", 6)]);
    assert_eq!(after(vop2("V_MUL_F32", 0), 0).unwrap().first_lane_reads, 1);
    assert_eq!(after(vop2("V_MUL_F32", 7), 7).unwrap().first_lane_reads, 1);
    // accumulators: the thread id in v0 stays in the sum
    for name in ["V_FMAC_F32", "V_FMAC_F16", "V_PK_FMAC_F16", "V_DOT2C_F32_F16", "V_DOT4C_I32_I8"] {
        assert_eq!(after(vop2(name, 0), 0).unwrap_err().rule, Rule::FirstLane, "{name}");
    }
    // 16-bit results keep the other half of the register (op_sel[3]: the low one)
    for sel in [0, 8] {
        let half = i("V_ADD_NC_U16", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 5), ("SRC1", V + 6), ("OP_SEL", sel)]);
        assert_eq!(after(half, 0).unwrap_err().rule, Rule::FirstLane, "op_sel {sel}");
    }
    // a swap with the thread id: v5 is the id now, v0 the number
    let swap = i("V_SWAP_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V)]);
    assert_eq!(after(swap.clone(), 5).unwrap_err().rule, Rule::FirstLane);
    assert_eq!(after(swap.clone(), 0).unwrap().first_lane_reads, 1);
    // ... in the lanes of a branch only: neither is one number any more
    let (head, tail) = if_then();
    for dst in [0, 5] {
        let p = [&uniform[..], &head[..], &[swap.clone(), read(20, dst), vmov(8, 20)], &tail[..], &[store(8)]].concat();
        assert_eq!(verdict(&p).unwrap_err().rule, Rule::FirstLane, "v{dst}");
    }
    // the relative swap writes v[5 + m0] and v[6 + m0]: any register
    let swaprel = i("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 6)]);
    assert_eq!(planted_at(&[vmov(7, 2)], &[swaprel], 1, &[read(20, 7), vmov(8, 20), store(8)]).rule, Rule::FirstLane);
    // loads with TFE / LWE write a status dword past their four registers
    let tfe = i("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1), ("TFE", 1)]);
    let lwe = i("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("DMASK", 15), ("DIM", 1), ("LWE", 1)]);
    for load in [tfe, lwe] {
        assert_eq!(planted_at(&[vmov(5, 2)], &[load], 1, &[read(20, 5), vmov(8, 20), store(8)]).rule, Rule::FirstLane);
    }
}

/// `v_perm_b32` picks bytes inside one lane; the lane permutes read other lanes.
#[test]
fn a_byte_permute_qualifies_and_a_lane_permute_does_not() {
    let (head, tail) = if_then();
    qualifies(&[head.clone(), vec![i("V_PERM_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V), ("SRC1", V + 1)])], tail.clone()].concat());
    for name in ["V_PERMLANE16_B32", "V_PERMLANEX16_B32"] {
        assert_eq!(planted(&head, &[i(name, "ENC_VOP3", &[("VDST", 6), ("SRC0", V), ("SRC1", 2), ("SRC2", 3)])], &tail), Rule::CrossLane, "{name}");
    }
}

/// The divide sequence: `v_div_scale` leaves each lane's scale bit in VCC, `v_div_fmas` takes
/// the lane's own.
#[test]
fn a_divide_qualifies() {
    let scale = i("V_DIV_SCALE_F32", "VOP3_SDST_ENC", &[("VDST", 6), ("SDST", VCC), ("SRC0", V + 1), ("SRC1", V + 1), ("SRC2", V)]);
    let fmas = i("V_DIV_FMAS_F32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", V + 2), ("SRC2", V + 6)]);
    qualifies(&[scale, fmas.clone(), store(7)]);
    // VCC a number: no lane's scale
    let reject = verdict(&[sop1("S_MOV_B64", VCC, 2), fmas, store(7)]).unwrap_err();
    assert_eq!(reject.rule, Rule::DataAsMask, "{reject}");
}

/// EXECZ and VCCZ as operands are wave-wide tests of EXEC and VCC, read as numbers.
#[test]
fn a_mask_test_as_an_operand_ends_the_proof() {
    let (head, tail) = if_then();
    assert_eq!(planted(&head, &[vmov(6, 252)], &tail), Rule::MaskAsData);
    assert_eq!(planted(&head, &[sop1("S_MOV_B32", 20, 251)], &tail), Rule::MaskAsData);
    assert_eq!(planted(&head, &[sop2("S_ADD_U32", 20, 2, 252)], &tail), Rule::MaskAsData);
}

/// A lane's load of memory flagged as written by other threads, where the program's own
/// stores or atomics can come before it: another lane of the wave may have written it.
#[test]
fn a_shared_load_after_the_programs_stores_ends_the_proof() {
    let load = |flag: &'static str| i("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 6), ("ADDR", 0), ("SADDR", 0x7D), (flag, 1)]);
    let buffer = i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1), ("GLC", 1)]);
    let plain = i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1)]);
    let atomic = i("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("IDXEN", 1)]);
    for first in [store(3), atomic] {
        for shared in [load("GLC"), load("DLC"), buffer.clone()] {
            assert_eq!(planted(&[first.clone()], &[shared.clone()], &[vadd(4, 6, 0), store(4)]), Rule::SharedLoad);
            // before every store: nothing of this program's to read
            qualifies(&[shared, vadd(4, 6, 0), first.clone()]);
        }
        qualifies(&[first, plain.clone(), vadd(4, 6, 0), store(4)]);
    }
    // a store later in a loop comes before the load the next time round
    let p = [i("S_MOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 3)]), A::L("again"), load("GLC"), vadd(4, 6, 0), store(4), sop2("S_ADD_I32", 20, 20, ONES), sopc("S_CMP_GT_I32", 20, ZERO), A::B("S_CBRANCH_SCC1", "again")];
    let reject = verdict(&p).unwrap_err();
    assert_eq!((reject.rule, reject.pc), (Rule::SharedLoad, Some(asm_at(&p).1[2])), "{reject}");
}

/// Where the wave runs on for other lanes, what it does must not reach a lane that left.
#[test]
fn code_a_thread_skips_must_not_change_what_it_reads() {
    // the loop count of the slowest lane read after the loop
    let mut counted = lane_loop("S_CBRANCH_EXECNZ");
    counted.insert(4, sop2("S_ADD_U32", 20, 20, 129));
    let at = counted.len() - 3;
    qualifies(&counted);
    counted.push(vmov(5, 20));
    let reject = verdict(&counted).unwrap_err();
    assert_eq!((reject.rule, reject.pc, reject.detail.as_str()), (Rule::SoloScalar, Some(asm_at(&counted).1[at]), "s20"), "{reject}");

    let (head, tail) = if_then();
    let branch = asm_at(&[head.clone(), tail.clone()].concat()).1[2];
    // SCC set where the lane's thread is not, read after
    let before = [vbelow(0, 8), sop1("S_MOV_B64", 10, EXEC), sop2("S_AND_B64", EXEC, EXEC, VCC), sopc("S_CMP_EQ_U32", 2, ZERO), A::B("S_CBRANCH_EXECZ", "join"), vadd(3, 0, 0)];
    let after = [A::L("join"), sop2("S_CSELECT_B32", 20, 2, 3), sop1("S_MOV_B64", EXEC, 10), vmov(5, 20), store(5)];
    qualifies(&[&before[..], &after[..]].concat());
    let scc = [&before[..], &[sopc("S_CMP_EQ_U32", 3, ZERO)], &after[..]].concat();
    let reject = verdict(&scc).unwrap_err();
    assert_eq!((reject.rule, reject.pc, reject.detail.as_str()), (Rule::SoloScalar, Some(asm_at(&scc).1[4]), "scc"), "{reject}");
    // a mask register the skipped code leaves different for the lane
    let wrecked = [sop1("S_MOV_B64", 10, ZERO)];
    let reject = verdict(&[head.clone(), wrecked.to_vec(), tail.clone()].concat()).unwrap_err();
    assert_eq!((reject.rule, reject.pc, reject.detail.as_str()), (Rule::SoloScalar, Some(branch), "s10"), "{reject}");
    // every lane switched on inside: the lane that left works there after all
    let on = [sop1("S_OR_SAVEEXEC_B64", 14, ONES), vmov(5, 131), sop1("S_MOV_B64", EXEC, 14)];
    let reject = verdict(&[head.clone(), on.to_vec(), tail.clone()].concat()).unwrap_err();
    assert_eq!((reject.rule, reject.pc), (Rule::SoloVector, Some(branch)), "{reject}");
    assert!(reject.detail.contains("v_mov_b32"), "{reject}");
    // the wave ends in there while the lane's thread goes on
    let end = [i("S_ENDPGM", "ENC_SOPP", &[])];
    let reject = verdict(&[head.clone(), end.to_vec(), tail.clone()].concat()).unwrap_err();
    assert_eq!((reject.rule, reject.pc), (Rule::SoloEnd, Some(branch)), "{reject}");
}

/// A wave that goes round a loop for every lane while any lane asks for it: a lane that is done
/// works on in the wave, not in its thread.
#[test]
fn a_loop_all_lanes_run_while_any_asks_ends_the_proof() {
    let p = [vmov(3, ZERO), vmov(1, 129), A::L("loop"), vadd(3, 3, 1), i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 0)]), A::B("S_CBRANCH_VCCNZ", "loop"), store(3)];
    let reject = verdict(&p).unwrap_err();
    assert_eq!((reject.rule, reject.pc), (Rule::SoloVector, Some(asm_at(&p).1[5])), "{reject}");
}

/// A compare the wave makes again on its way is a new mask: that the lane's bit of the old one
/// was clear says nothing about it. (Here the lane's compare gives the same again, which the
/// proof does not follow; it must not take the new mask for the tested one.)
#[test]
fn a_mask_made_again_is_not_the_tested_one() {
    let p = [
        vmov(3, ZERO),
        vmov(1, 129),
        sop1("S_MOV_B64", 12, EXEC),
        A::L("loop"),
        i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 0)]),
        sop2("S_AND_B64", EXEC, 12, VCC),
        vadd(3, 3, 1),
        sop1("S_MOV_B64", EXEC, 12),
        A::B("S_CBRANCH_VCCNZ", "loop"),
        store(3),
    ];
    let reject = verdict(&p).unwrap_err();
    assert_eq!((reject.rule, reject.pc), (Rule::SoloVector, Some(asm_at(&p).1[8])), "{reject}");
    assert!(reject.detail.contains("v_add_nc_u32"), "{reject}");
}

#[test]
fn what_the_proof_does_not_cover_ends_it() {
    let (head, tail) = if_then();
    assert_eq!(planted(&head, &[sop1("S_MOVRELS_B32", 20, 2)], &tail), Rule::IndexedRegisters);
    assert_eq!(planted(&head, &[sop1("S_SETPC_B64", 0, 2)], &tail), Rule::IndirectJump);
    assert_eq!(planted(&head, &[i("S_SETREG_B32", "ENC_SOPK", &[("SDST", 2), ("SIMM16", 1)])], &tail), Rule::Unclassified);
    assert_eq!(planted(&head, &[i("S_SETHALT", "ENC_SOPP", &[("SIMM16", 1)])], &tail), Rule::Unclassified);
    assert_eq!(planted(&head, &[i("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF)])], &tail), Rule::Unclassified);
}

/// Only wave64 compute programs, and not their instrumented translations.
#[test]
fn other_programs_keep_their_form() {
    let (head, tail) = if_then();
    let p = [head, tail].concat();
    let cs = options([64, 1, 1]);
    assert!(verdict_with(&p, &cs).is_ok());
    for opt in [Options { wave64: false, ..cs.clone() }, Options { mode: Mode::Kernel, ..cs.clone() }, Options { mode: Mode::Pixel(Default::default()), ..cs.clone() }, Options { mode: Mode::Vertex(Default::default()), ..cs.clone() }] {
        let reject = verdict_with(&p, &opt).unwrap_err();
        assert_eq!((reject.rule, reject.pc), (Rule::NotWave64Compute, None), "{reject}");
    }
    assert_eq!(verdict_with(&p, &Options { loop_profile: true, ..cs.clone() }).unwrap_err().rule, Rule::Instrumented);
    // what the census says of a profiled program: the form it has without the profile
    let code = asm(&p);
    assert!(super::lane_independent::prove(&super::decode_program(&code, &cs).unwrap().insts).is_ok());
}

#[test]
fn the_reason_names_the_instruction() {
    let (head, tail) = if_then();
    let reject = planted_at(&head, &[sop1("S_BCNT1_I32_B64", 20, EXEC)], 0, &tail);
    assert_eq!(reject.to_string(), "+0x10 s_bcnt1_i32_b64: a lane mask read as a number (exec)");
}
