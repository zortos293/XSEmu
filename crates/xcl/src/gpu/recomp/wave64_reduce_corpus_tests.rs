//! The wave64 pixel/vertex reduction fixture (`upper_reduce_tests`, `retail wave64`): the
//! pack's whole-wave reduction with every operation, DPP form, upper-half step and mask, and
//! the corpus digest dump the base tree's translator is compared against
//! (`XCL_UPPER_REDUCE_DIGESTS`). Self-contained: it needs nothing of `upper_reduce`, so the
//! same file builds against a tree without the switch.

use super::{translate, Mode, Options};
use crate::gpu::dxbc::SigElement;
use crate::gpu::rdna2::encode;

const V: u64 = 256; // VALU source code of v0
const LANE_SEL: [&str; 8] = ["LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3", "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"];

fn ins(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, fields, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

/// A program from instructions, ending in `s_endpgm`.
fn program_of(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut v: Vec<u8> = parts.concat();
    v.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> SigElement {
    SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// The pixel interface of the pack's reduction: SV_Position in v0 / v1, COLOR in v2 / v3.
pub(super) fn psi() -> super::PsInterface {
    super::PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    }
}

/// The 32-lane form of a wave64 pixel shader (`Options::split_wave64` on, as a GPU that can
/// run 32-lane waves gets it).
pub(super) fn options() -> Options {
    Options { mode: Mode::Pixel(psi()), wave64: true, split_wave64: true, ..Default::default() }
}

/// What the reduction's mask is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mask {
    /// the saved EXEC itself: every lane the wave had
    Wave,
    /// the odd lanes under the saved EXEC (the draw tests' mask)
    Odd,
    /// `-1`: not under the saved EXEC, so not a whole-wave mask
    Any,
}

/// The pack's fold: `v6` = the identity where the mask is 0, DPP8 / DPP16 xor 1, 2 and 4,
/// row_mirror, the other row through `v_permlanex16`, `v_readlane` 32 for the upper half
/// (`read32`) and `v_readlane` 0 for the total. The total lands in `v40`.
fn fold(op: &str, identity: u64, lit: Option<u32>, mask: Mask, dpp16: bool, read32: bool) -> Vec<Vec<u8>> {
    let mut c = vec![ins("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None)];
    match mask {
        Mask::Odd => {
            // s[18:19] = the odd lanes, s[10:11] = that under the saved EXEC
            c.push(ins("V_AND_B32", "ENC_VOP2", &[("VDST", 14), ("SRC0", 129), ("VSRC1", 12)], None));
            c.push(ins("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 14)], None));
            c.push(ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 18), ("SSRC0", 106)], None));
            c.push(ins("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 30)], None));
        }
        Mask::Wave => c.push(ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 30)], None)),
        Mask::Any => c.push(ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 193)], None)),
    }
    c.push(ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", identity), ("SRC1", V + 13), ("SRC2", 10)], lit));
    for n in [1u64, 2, 4] {
        if dpp16 {
            c.push(ins(op, "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x160 + n), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None));
        } else {
            let mut f: Vec<(&str, u64)> = vec![("VDST", 6), ("SRC0", 0xE9), ("VSRC0", 6), ("VSRC1", 6)];
            f.extend((0..8u64).map(|i| (LANE_SEL[i as usize], i ^ n)));
            c.push(ins(op, "VOP2_VOP_DPP8", &f, None));
        }
    }
    c.push(ins(op, "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None));
    c.push(ins("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 128), ("SRC2", 128)], None));
    c.push(ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 7), ("VSRC1", 6)], None));
    if read32 {
        c.push(ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 6), ("SRC1", 128 + 32)], None));
        c.push(ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", 11), ("VSRC1", 6)], None));
    }
    c.push(ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 40), ("SRC0", V + 6), ("SRC1", 128)], None));
    c.push(ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None));
    c
}

/// A pixel shader whose value is `0x100 + lane` (`negate`: `-(0x100 + lane)`), reduction
/// result exported as a float.
pub(super) fn ps(negate: bool, cvt: &str, op: &str, identity: u64, lit: Option<u32>, mask: Mask, dpp16: bool, read32: bool) -> Vec<u8> {
    let mut parts = vec![
        // v12 = the lane (v_mbcnt of all lanes)
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", 193), ("SRC1", V + 12)], None),
        // v13 = the value: 0x100 + lane, or its negation
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 255)], Some(0x100)),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 13), ("VSRC1", 12)], None),
    ];
    if negate {
        parts.push(ins("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 128), ("VSRC1", 13)], None));
    }
    parts.extend(fold(op, identity, lit, mask, dpp16, read32));
    parts.push(ins(cvt, "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None));
    parts.push(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None));
    program_of(&parts)
}

/// The pack's min reduction: `v_min_u32`, the identity -1.
pub(super) fn min_ps() -> Vec<u8> {
    ps(false, "V_CVT_F32_U32", "V_MIN_U32", 193, None, Mask::Wave, false, true)
}

/// The corpus: every operation the pack's fold uses with the identity the console compiler
/// installs, both DPP forms, the upper-half step on and off, and three masks (two whole-wave
/// ones and one that is not under the saved EXEC), plus programs with cross-lane reads of
/// other kinds and a program with two reductions.
pub(super) fn corpus() -> Vec<(String, Vec<u8>)> {
    let ops: [(&str, u64, Option<u32>, bool, &str); 8] = [
        ("V_MIN_U32", 193, None, false, "V_CVT_F32_U32"),
        ("V_MAX_U32", 128, None, false, "V_CVT_F32_U32"),
        ("V_OR_B32", 128, None, false, "V_CVT_F32_U32"),
        ("V_AND_B32", 193, None, false, "V_CVT_F32_U32"),
        ("V_XOR_B32", 128, None, false, "V_CVT_F32_U32"),
        ("V_ADD_NC_U32", 128, None, false, "V_CVT_F32_U32"),
        ("V_MIN_I32", 255, Some(i32::MAX as u32), true, "V_CVT_F32_I32"),
        ("V_MAX_I32", 255, Some(i32::MIN as u32), true, "V_CVT_F32_I32"),
    ];
    let mut out = Vec::new();
    for (op, id, lit, negate, cvt) in ops {
        for mask in [Mask::Wave, Mask::Odd, Mask::Any] {
            for dpp16 in [false, true] {
                for read32 in [false, true] {
                    let name = format!("{op}-{mask:?}-{}-{}", if dpp16 { "dpp16" } else { "dpp8" }, if read32 { "read32" } else { "read0" });
                    out.push((name, ps(negate, cvt, op, id, lit, mask, dpp16, read32)));
                }
            }
        }
    }
    // a reduction whose identity is not the operation's
    out.push(("V_MIN_U32-wrong-identity".into(), ps(false, "V_CVT_F32_U32", "V_MIN_U32", 128, None, Mask::Wave, false, true)));
    // a fold without the row mirror: not the pack's reduction
    let mut no_mirror = fold("V_MIN_U32", 193, None, Mask::Wave, false, true);
    no_mirror.remove(6); // the row mirror, after saveexec, mask, seed and the three xors
    let mut parts = vec![
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 255)], Some(0x100)),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 13), ("VSRC1", 12)], None),
    ];
    parts.extend(no_mirror);
    parts.push(ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None));
    parts.push(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None));
    out.push(("no-row-mirror".into(), program_of(&parts)));
    // two reductions in one program, the second after the first restores EXEC
    let mut parts = vec![
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 255)], Some(0x100)),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 13), ("VSRC1", 12)], None),
    ];
    parts.extend(fold("V_MIN_U32", 193, None, Mask::Wave, false, true));
    parts.extend(fold("V_MAX_U32", 128, None, Mask::Wave, false, true));
    parts.push(ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None));
    parts.push(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None));
    out.push(("two-reductions".into(), program_of(&parts)));
    // the fold behind a branch (a block of its own: the seed's copy and the read are in the
    // same brace scope, and the branch target is past the read)
    let folded = fold("V_MIN_U32", 193, None, Mask::Wave, false, true);
    let skip = (folded.iter().map(|c| c.len()).sum::<usize>() / 4) as u64;
    let mut parts = vec![
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 255)], Some(0x100)),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 13), ("VSRC1", 12)], None),
        // skip the fold when SCC is 0 (the branch target is the instruction past it)
        ins("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", skip)], None),
    ];
    parts.extend(folded);
    parts.push(ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None));
    parts.push(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None));
    out.push(("branch-around".into(), program_of(&parts)));
    // no reduction: a waterfall read, a spill slot, a lane read of a compare mask
    out.push((
        "waterfall".into(),
        program_of(&[
            ins("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 18), ("SSRC1", 126)], None),
            ins("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 13), ("SRC1", 0)], None),
            ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 4)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None),
        ]),
    ));
    out.push((
        "spill".into(),
        program_of(&[
            ins("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", 20), ("SRC1", 128 + 5)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 30), ("SRC1", 128 + 5)], None),
            ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 4)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None),
        ]),
    ));
    out
}

/// One corpus program by name.
pub(super) fn corpus_named(name: &str) -> Vec<u8> {
    corpus().into_iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no corpus program {name}")).1
}

/// The HLSL digest of every corpus program, with the switch off, for the base tree and this
/// one to be compared: `XCL_UPPER_REDUCE_DIGESTS=<file>` writes `name<TAB>fnv64(hlsl)`.
#[test]
#[ignore]
fn wave64_reduction_corpus_digests() {
    let Ok(out) = std::env::var("XCL_UPPER_REDUCE_DIGESTS") else { return };
    let mut s = String::new();
    let mut ok = 0usize;
    for (name, code) in corpus() {
        let t = translate(&code, &options()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
        ok += 1;
        s.push_str(&format!("{name}\t{:016x}\n", crate::gpu::shader::fnv64(t.hlsl.as_bytes())));
    }
    std::fs::write(out, s).unwrap();
    println!("{ok} programs");
}
