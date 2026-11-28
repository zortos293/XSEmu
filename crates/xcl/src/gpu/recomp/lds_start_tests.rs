//! `XCL_LDS_START` (`super::lds_start`): the group's LDS starts as the program's first store
//! leaves it, against the host's memory (the direct backend's own DXIL, run by its interpreter).

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

/// The title's tiled light in miniature: thread 0 alone stores the tile's starting maximum (0,
/// dword 0) and minimum (-1, dword 1) behind the per-thread early-outs, every live thread
/// `ds_min_u32`s its key (its own id) into dword 1, then every lane reads it back. `stray`: a
/// write at an address that is not a constant before the set-up; `read_first`: a read before
/// the set-up.
fn tile_min_max(stray: bool, read_first: bool) -> Vec<u8> {
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("V_CMP_LE_U32", "ENC_VOPC", vec![("SRC0", 0), ("VSRC1", V)], None), // vcc = s0 <= v0
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 12), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 5), ("SRC0", 128)], None), // v5 = 0
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 6), ("SRC0", 193)], None), // v6 = -1
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 14), ("SSRC0", 126)], None),
    ];
    if stray {
        list.push(("DS_WRITE_B32", "ENC_DS", vec![("ADDR", V), ("DATA0", V + 3), ("OFFSET0", 16)], None));
    }
    if read_first {
        list.push(("DS_READ_B32", "ENC_DS", vec![("VDST", 8), ("ADDR", 5), ("OFFSET0", 4)], None));
    }
    list.extend([
        ("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", V)], None), // exec = thread 0
        ("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 5), ("DATA0", 5), ("DATA1", 6), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 14)], None),
        ("DS_MIN_U32", "ENC_DS", vec![("ADDR", 5), ("DATA0", V), ("OFFSET0", 4)], None),
        ("DS_READ_B32", "ENC_DS", vec![("VDST", 7), ("ADDR", 5), ("OFFSET0", 4)], None),
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 12)], None),
    ]);
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    prog(&refs)
}

/// A kernel-mode translation with `XCL_LDS_START` on or off on this thread.
fn kernel(code: &[u8], wave64: bool, split: bool, on: bool) -> String {
    let was = super::lds_start::TEST_LDS_START.replace(Some(on));
    let t = translate(code, &Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() });
    super::lds_start::TEST_LDS_START.set(was);
    let t = t.unwrap_or_else(|e| panic!("translate: {e}"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

/// What a run of a kernel-mode program stored in `res` (u1) per word, with the `io` words
/// `patches` set.
fn run_kernel(hlsl: &str, patches: &[(usize, u32)]) -> BTreeMap<usize, u32> {
    let program = interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap();
    let id = |word: usize| ((interp::res::IDENT_IO as u64) << 32) | word as u64;
    let input = interp::Input { seed: 2, patches: patches.iter().map(|&(w, v)| (id(w), v)).collect() };
    let run = interp::run_input(&program, &input, &interp::Config { force_ballots: false, ..Default::default() }, false).0;
    assert!(run.stop.is_none(), "{:?}\n{hlsl}", run.stop);
    let res = interp::res::IDENT_IO | 1;
    run.written.iter().filter(|w| w.0 == 0 && w.1 == res).map(|w| (w.2 as usize / 4, w.3[0])).collect()
}

/// Kernel ABI, one 32-lane wave: `io` words of the lanes' `v0` = the lane, EXEC all lanes, `s0`
/// the first live lane.
fn lanes(s0: u32) -> Vec<(usize, u32)> {
    let mut p = vec![(0, s0), (126, 0xFFFF_FFFF), (127, 0xFFFF_FFFF)];
    p.extend((0..32).map(|l| (128 + l * 256, l as u32)));
    p
}

fn vgpr(lane: usize, reg: usize) -> usize {
    super::KERNEL_SGPRS + lane * super::KERNEL_VGPRS + reg
}

/// A group whose thread 0 is not live does not run the program's own set-up of its LDS. With
/// `XCL_LDS_START` every lane stores that set-up at the group's start, so the tile's minimum is
/// the minimum over the live lanes; without it, dword 1 is whatever the host's memory held
/// (0 in the interpreter: the minimum comes out 0, the tile's whole minimum wrong).
#[test]
fn a_groups_lds_starts_as_the_programs_first_store_leaves_it() {
    let code = tile_min_max(false, false);
    // the wave64 forms (the title's programs) carry the same two stores, before the program
    for split in [false, true] {
        let hlsl = kernel(&code, true, split, true);
        assert!(hlsl.contains("xcl_lds[0u] = 0x00000000u;") && hlsl.contains("xcl_lds[1u] = 0xffffffffu;"), "split {split}\n{hlsl}");
        assert!(hlsl.contains("GroupMemoryBarrierWithGroupSync(); // XCL_LDS_START"), "split {split}\n{hlsl}");
        assert!(!kernel(&code, true, split, false).contains("XCL_LDS_START"), "split {split}: off by default");
    }
    // the run: one 32-lane wave, thread 0 not live (s0 = 1)
    let (on, off) = (kernel(&code, false, false, true), kernel(&code, false, false, false));
    assert_ne!(on, off);
    for s0 in [1usize, 8, 20] {
        let with = run_kernel(&on, &lanes(s0 as u32));
        let without = run_kernel(&off, &lanes(s0 as u32));
        for lane in s0..32 {
            assert_eq!(with[&vgpr(lane, 7)], s0 as u32, "lane {lane} of s0 = {s0}: the minimum over the live lanes");
        }
        for lane in s0..32 {
            assert_eq!(without[&vgpr(lane, 7)], 0, "lane {lane} of s0 = {s0}: without the set-up the host's 0 stands");
        }
    }
}

/// Only a plain store of constants that is the program's first LDS access is a set-up: an
/// earlier access, or a write whose address is not a constant, leaves the group's LDS as the
/// host's memory is.
#[test]
fn only_a_first_store_of_constants_is_a_set_up() {
    assert!(kernel(&tile_min_max(false, false), true, true, true).contains("XCL_LDS_START"));
    // a read before the set-up: the read would take whatever the host's memory held
    assert!(!kernel(&tile_min_max(false, true), true, true, true).contains("XCL_LDS_START"));
    // a write at an address that is not a constant: what it touches is not known
    assert!(!kernel(&tile_min_max(true, false), true, true, true).contains("XCL_LDS_START"));
}
