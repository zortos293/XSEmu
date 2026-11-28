// SPDX-License-Identifier: GPL-3.0-or-later
// xsemu: regression tests for the wave-wide SCC branch fixes taken from DoubleGhosty/cool_fox (9a4aa8d3, bd8b4f52).
// GPL-3.0-or-later, see crates/xcl-cgbj/LICENSE and PROVENANCE.md. native()/native_of() are CGBJ's (cgbj-reupload
// f4b48ad, crates/victory-shader/src/emit/wave64_tests.rs:46-98); the fixtures and the store-guard test are
// steven-seagull27's for DoubleGhosty/cool_fox#66 and #78 (bd8b4f52, same file, 1150-1243). The exact-policy test and the
// packed-form tests (M1d) are xsemu's.
//! A console wave64 compute program on a host of 32-lane waves branching on the SCC of a lane-mask
//! operation (`s_and_b64 s, vcc, exec` then `s_cbranch_scc0/1`): the whole console wave takes one
//! arm, so the branch must not be split per lane, and each store in an arm keeps its EXEC guard.

use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

use super::lower;

const S_ENDPGM: u32 = 0xBF81_0000;

fn native(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// With two host waves for one console wave, one active lane is enough to set SCC
/// for all lanes still in EXEC. For example, with 48 active lanes and lane 37
/// selected, all 48 store 2; with lane 49 selected they all store 1.
#[derive(Clone, Copy, Debug)]
enum SccUse {
    Combined,
    Branch,
    Select,
    StoreOnClear,
    StoreOnSet,
}

fn nested_mask_scc(which: SccUse) -> NativeProgram {
    let choice: &[u32] = match which {
        SccUse::StoreOnClear => return stores_in_scc_arms(false),
        SccUse::StoreOnSet => return stores_in_scc_arms(true),
        SccUse::Combined => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0206, // v_mov_b32 v2, s6
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Branch => &[
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0282, // v_mov_b32 v2, 2
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Select => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0x7E04_0206, // v_mov_b32 v2, s6
        ],
    };
    let mut body = vec![
        0xBF8C_0000, // s_waitcnt 0: wait for the fixture's scalar descriptor load
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | (5 + choice.len() as u32), // s_cbranch_execz to the restore
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
    ];
    body.extend_from_slice(choice);
    body.extend_from_slice(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0201, // buffer_store_dword v2, v1, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&body)
}

/// Each active lane stores to its own dword in the chosen arm. The address
/// is set before EXEC narrows, so every inactive lane also has a distinct
/// in-bounds dword whose initial zero makes an unwanted store observable.
fn stores_in_scc_arms(on_set: bool) -> NativeProgram {
    let (branch, fallthrough, taken) = if on_set {
        (0xbf85_0004, 1, 2) // s_cbranch_scc1
    } else {
        (0xbf84_0004, 2, 1) // s_cbranch_scc0
    };
    native(&[
        0xbf8c_0000, // s_waitcnt 0 for the descriptor
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7d88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xbe8c_246a, // s_and_saveexec_b64 s[12:13], vcc
        0xbf88_000a, // s_cbranch_execz to the restore
        0x7d84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0x8792_7e6a, // s_and_b64 s[18:19], vcc, exec
        branch,
        0x7e04_0280 | fallthrough, // v_mov_b32 v2, fallthrough
        0xe070_1000, 0x8002_0201, // buffer_store_dword v2 at v1
        0xbf82_0003, // s_branch to the restore
        0x7e04_0280 | taken, // v_mov_b32 v2, taken
        0xe070_1000, 0x8002_0201, // buffer_store_dword v2 at v1
        0xbefe_040c, // s_mov_b64 exec, s[12:13]
    ])
}

#[test]
fn both_arms_of_a_wave_wide_scc_branch_keep_their_store_guards() {
    for on_set in [false, true] {
        let native = stores_in_scc_arms(on_set);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
        let form = ir::without_skips(&p, false).expect("removable EXEC skip");
        let st = structure::structure(&form).expect("structure");
        let mut stores = 0;
        for (block, guards) in form.blocks.iter().zip(&st.guards) {
            for (inst, guard) in block.insts.iter().zip(guards) {
                if inst.name() == "buffer_store_dword" {
                    stores += 1;
                    assert!(guard.is_some(), "store at {:#x} lost EXEC", inst.machine.offset);
                }
            }
        }
        assert_eq!(stores, 2);
    }
}

/// xsemu: under the exact wave policy (crate::xcl::set_wave_exact) on a host of 32-lane waves, a
/// branch on the SCC of a lane-mask operation whose arms keep EXEC is never split per lane (each lane
/// by its own bit, DoubleGhosty/cool_fox#66: 188 of 576 words wrong on an RTX 5080). Before the fix
/// Branch, StoreOnClear and StoreOnSet translated in the first form, split per lane (2,092-2,100
/// bytes); with it every case is refused or takes a retry form whose branch is the wave's. Sets the
/// process-wide policy, so it is ignored by default; run it alone with --ignored.
#[test]
#[ignore = "sets the process-wide exact wave policy; run alone with --ignored"]
fn xsemu_exact_policy_never_splits_a_mask_scc_branch_per_lane() {
    crate::xcl::set_wave_exact(true);
    let mut outcomes = Vec::new();
    for which in [SccUse::Branch, SccUse::Combined, SccUse::StoreOnClear, SccUse::StoreOnSet] {
        let native = nested_mask_scc(which);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
        let io = io::raise(&p, &native).expect("io");
        let c = crate::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        crate::xcl::take_form();
        let out = super::emit(&c, &native, &p, &st, &io, &res);
        let form = crate::xcl::take_form();
        lower::TEST_HOST_WAVE.set(was);
        outcomes.push((which, out.is_ok(), form));
    }
    crate::xcl::set_wave_exact(false);
    for (which, translated, form) in outcomes {
        assert!(!translated || form.is_some(), "{which:?} translated in the first form, split per lane");
    }
}

/// xsemu (M1d): `native` lowered straight in the packed form under the XCL binding on a host of
/// 32-lane waves: the DXIL, and what crossed lanes.
fn packed(native: &NativeProgram) -> (Result<Vec<u8>, crate::ShaderError>, Vec<&'static str>) {
    let _scope = crate::xcl::BindingScope::enter(crate::xcl::XclConfig::default());
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
    let c = crate::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    lower::PACKED_SEEN.with(|s| s.borrow_mut().clear());
    let out = super::xcl::packed(&c, native, &p).expect("the packed form applies").map(|(h, _)| h.dxil);
    lower::TEST_HOST_WAVE.set(was);
    (out, lower::PACKED_SEEN.with(|s| s.take()))
}

/// xsemu (M1d): `body` with the fixture's descriptor wait, `v1 = 4 * v0`, and a store of v2 at v1.
fn storing(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xbf8c_0000, // s_waitcnt 0 for the descriptor
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[0xe070_1000, 0x8002_0201]); // buffer_store_dword v2 at v1
    native(&words)
}

/// xsemu (M1d): every SCC fixture translates in the packed form, and each decision is one wave op
/// over both halves of each host lane: `s_cbranch_execz` and `s_cbranch_scc0` when every lane's
/// bit is clear, `s_cbranch_scc1` when some lane's is set, `s_cselect` by the SCC. The stored
/// results are checked by running these containers (xsemu_write_packed_fixture_containers), not here.
#[test]
fn the_packed_form_takes_the_scc_of_a_mask_operation_for_the_whole_console_wave() {
    let every = "branch on every lane";
    for (which, want) in [
        (SccUse::Branch, vec![every, every]),
        (SccUse::StoreOnClear, vec![every, every]),
        (SccUse::StoreOnSet, vec![every, "branch on some lane"]),
        (SccUse::Combined, vec![every, "scc", every]),
        (SccUse::Select, vec![every, "scc"]),
    ] {
        let (out, seen) = packed(&nested_mask_scc(which));
        assert!(out.is_ok(), "{which:?}: {out:?}");
        assert_eq!(seen, want, "{which:?}");
    }
}

/// xsemu (M1d): the packed form declares 32 threads per console wave of 64 and waves of 32.
#[test]
fn the_packed_form_declares_32_threads_per_console_wave() {
    assert_eq!(crate::xcl::packed_threads([64, 1, 1]), [32, 1, 1]);
    assert_eq!(crate::xcl::packed_threads([8, 8, 4]), [128, 1, 1]);
    assert_eq!(crate::xcl::packed_threads([48, 1, 1]), [32, 1, 1]);
    assert_eq!(crate::xcl::packed_threads([16, 1, 1]), [32, 1, 1]);
    let (out, _) = packed(&nested_mask_scc(SccUse::Branch));
    let dxil = out.expect("translates");
    let c = crate::Container::parse(&dxil).expect("a container");
    let psv = c.part(crate::FourCC::PSV0).expect("PSV0").data;
    // (numthreads are three consecutive dwords of the runtime info, container.rs)
    let words: Vec<u32> = psv.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    assert!(words.windows(3).any(|w| w == [32, 1, 1]), "numthreads 32, 1, 1 in PSV0");
}

/// xsemu (M1d): a mask read as data is the ballot of the low half's bits and of the high half's.
#[test]
fn a_ballot_in_the_packed_form_takes_both_halves() {
    let n = storing(&[
        0x7d88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xbe86_106a, // s_bcnt1_i32_b64 s6, vcc
        0x7e04_0206, // v_mov_b32 v2, s6
    ]);
    let p = ir::build(&n.code, true, n.vgprs).expect("IR");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_bcnt1_i32_b64"));
    let (out, seen) = packed(&n);
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(seen, ["ballot"]);
}

/// xsemu (M1d): v_readlane of a lane at or past 32 reads the high half of the host lane 32 below.
#[test]
fn a_lane_read_past_32_takes_the_high_half() {
    let n = storing(&[
        0xd760_0006, 0x0001_5100, // v_readlane_b32 s6, v0, 40
        0x7e04_0206, // v_mov_b32 v2, s6
    ]);
    let p = ir::build(&n.code, true, n.vgprs).expect("IR");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_readlane_b32"));
    let (out, seen) = packed(&n);
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(seen, ["lane read"]);
}

/// xsemu (M1e): a group of more than one console wave whose program already declares the D3D12
/// limit of 32 KiB of LDS leaves no dword for the counter the host waves take their console
/// threads from, so the packed form refuses it instead of making a container dxil.dll rejects.
/// One console wave needs no counter, and a program below the limit gets its dword.
#[test]
fn the_packed_form_refuses_a_claim_dword_past_32_kib() {
    let shaped = |threads: [u32; 3], lds_blocks: u32| {
        let mut n = nested_mask_scc(SccUse::Branch);
        n.io = StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | lds_blocks << 15, threads });
        n
    };
    let full = shaped([128, 1, 1], 64);
    let StageIo::Cs(cs) = &full.io else { unreachable!() };
    assert_eq!(cs.lds_bytes(), 32768);
    let (out, _) = packed(&full);
    assert!(matches!(out, Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::NO_CLAIM_DWORD), "{out:?}");
    for (threads, blocks) in [([64, 1, 1], 64), ([128, 1, 1], 63), ([8, 8, 2], 0)] {
        let (out, _) = packed(&shaped(threads, blocks));
        assert!(out.is_ok(), "{threads:?} with {blocks} blocks of LDS: {out:?}");
    }
    assert_eq!(lower::claim_dword(128, 8191), Ok(Some(8191)));
    assert_eq!(lower::claim_dword(64, 8192), Ok(None));
    assert_eq!(lower::claim_dword(65, 8192), Err(lower::NO_CLAIM_DWORD));
}

/// xsemu (M1d): writes the packed containers of the SCC fixtures, the ballot and the lane read
/// to the folder `XCL_M1D_FIXTURES` names, to be run on an interpreter: every lane below s0 stores
/// 2 when lane s1 is one of them and 1 otherwise, the ballot stores min(s0, 64) and the lane read 40.
#[test]
#[ignore = "writes files; run with XCL_M1D_FIXTURES set"]
fn xsemu_write_packed_fixture_containers() {
    let dir = std::path::PathBuf::from(std::env::var_os("XCL_M1D_FIXTURES").expect("XCL_M1D_FIXTURES"));
    let mut all: Vec<(String, NativeProgram)> = [SccUse::Branch, SccUse::StoreOnClear, SccUse::StoreOnSet, SccUse::Combined, SccUse::Select]
        .into_iter()
        .map(|w| (format!("{w:?}"), nested_mask_scc(w)))
        .collect();
    all.push(("Ballot".into(), storing(&[0x7d88_0000, 0xbe86_106a, 0x7e04_0206])));
    all.push(("ReadLane40".into(), storing(&[0xd760_0006, 0x0001_5100, 0x7e04_0206])));
    for (name, n) in all {
        let (out, _) = packed(&n);
        std::fs::write(dir.join(format!("{name}.dxil")), out.expect("translates")).expect("write");
    }
}
