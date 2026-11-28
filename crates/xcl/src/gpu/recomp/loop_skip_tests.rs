//! A skip around a loop that goes round on a scalar test is the wave's (`super::recomp`'s
//! execution model: every console branch is scalar, so a test of the lane mask is an OR over
//! the wave's lanes and the loop's bound is one value for all of them). Pinned here as the
//! behaviour the translator already has: the title's tiled light has such a loop, and running
//! it with no lane left would run its group barriers for a wave the console never starts.

use std::collections::BTreeMap;

use xcl_dxil::interp;

use super::{translate, Mode, Options};

const V: u64 = 256;

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    use crate::gpu::rdna2::encode;
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

/// `if (t < s0) { for (s6 = 0; s6 < s1; s6++) { v1 += 1; s_barrier } } v2 = s6`: the lanes below
/// `s0` are live, the loop goes round on a scalar compare, and its body has a group barrier – a
/// lane that took the loop while another skipped it would leave the barrier waiting. `v2` (every
/// lane writes it once EXEC is back) is the rounds the wave ran.
fn skip_around_a_loop() -> Vec<u8> {
    prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 0), ("VSRC1", V)], None), // vcc = s0 > v0
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 106)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 128)], None), // s6 = 0
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 5)], None), // to the restore, 0x24
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", V + 1)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 6), ("SSRC1", 129)], None),
        ("S_CMP_GE_U32", "ENC_SOPC", &[("SSRC0", 6), ("SSRC1", 1)], None),
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", -5i16 as u16 as u64)], None), // back to 0x10
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 12)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 6)], None), // v2 = s6
    ])
}

fn kernel_hlsl(code: &[u8], wave64: bool) -> String {
    let t = translate(code, &Options { mode: Mode::Kernel, wave64, ..Default::default() }).unwrap_or_else(|e| panic!("translate: {e}"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

fn run_kernel(hlsl: &str, patches: &[(usize, u32)]) -> BTreeMap<usize, u32> {
    let program = interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap();
    let id = |word: usize| ((interp::res::IDENT_IO as u64) << 32) | word as u64;
    let input = interp::Input { seed: 2, patches: patches.iter().map(|&(w, v)| (id(w), v)).collect() };
    let run = interp::run_input(&program, &input, &interp::Config { force_ballots: false, ..Default::default() }, false).0;
    assert!(run.stop.is_none(), "{:?}\n{hlsl}", run.stop);
    let res = interp::res::IDENT_IO | 1;
    run.written.iter().filter(|w| w.0 == 0 && w.1 == res).map(|w| (w.2 as usize / 4, w.3[0])).collect()
}

/// Kernel ABI (`io` words are the registers: word `n` is SGPR `n`, word 128 + 256·lane + n
/// is VGPR `n` of the lane), one wave: the lanes below `s0` are live, `s1` = the loop's
/// rounds, `v1` = 0.
fn lanes(s0: u32, s1: u32, count: usize) -> Vec<(usize, u32)> {
    let mut p = vec![(0, s0), (1, s1), (126, 0xFFFF_FFFF), (127, 0xFFFF_FFFF)];
    p.extend((0..count).map(|l| (128 + l * 256, l as u32)));
    p.extend((0..count).map(|l| (128 + l * 256 + 1, 0)));
    p
}

fn vgpr(lane: usize, reg: usize) -> usize {
    super::KERNEL_SGPRS + lane * super::KERNEL_VGPRS + reg
}

/// The skip reads the whole mask (`s_cbranch_execz` is an OR over the wave's lanes, not a
/// lane's own bit), and a wave with no lane left does not run the loop at all: `s6` stays 0 and
/// the barrier is never reached. With lanes live the whole wave runs the same rounds.
#[test]
fn a_skip_around_a_loop_on_a_scalar_test_is_the_waves() {
    let code = skip_around_a_loop();
    // the console wave's mask, both words of it in the wave64 form
    let wide = kernel_hlsl(&code, true);
    assert!(wide.contains("(exec_lo | exec_hi) == 0u"), "{wide}");
    let narrow = kernel_hlsl(&code, false);
    assert!(narrow.contains("exec_lo == 0u"), "{narrow}");
    // the scalar the loop goes round on: one value for the whole wave
    for (s0, s1) in [(0u32, 3u32), (3, 3), (3, 5)] {
        let res = run_kernel(&wide, &lanes(s0, s1, 64));
        let rounds = if s0 == 0 { 0 } else { s1 };
        for lane in 0..32 {
            assert_eq!(res[&vgpr(lane, 2)], rounds, "wave64 lane {lane} of s0 = {s0}, s1 = {s1}: the wave's rounds");
        }
    }
    for (s0, s1) in [(0u32, 3u32), (3, 3), (3, 5)] {
        let res = run_kernel(&narrow, &lanes(s0, s1, 32));
        let rounds = if s0 == 0 { 0 } else { s1 };
        for lane in 0..32 {
            assert_eq!(res[&vgpr(lane, 2)], rounds, "lane {lane} of s0 = {s0}, s1 = {s1}: the wave's rounds");
            assert_eq!(res[&vgpr(lane, 1)], if lane < s0 as usize { s1 } else { 0 }, "lane {lane} of s0 = {s0}: v1");
        }
    }
}
