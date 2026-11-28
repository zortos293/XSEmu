// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../../LICENSE and ../../../PROVENANCE.md), so it carries that licence.
//! Tests of the packed form's console thread base and of its whole-wave 64-bit float sum, on
//! hand-assembled wave64 compute programs (the encoders are `emit/wave.rs`'s).

use crate::emit::lower;
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{ir, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC: u8 = 126;

const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
}

const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
    (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `v_lshlrev_b32` (opcode 0x1a): `vdst = vsrc1 << src0`.
const fn vop2_lshl(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (0x1a << 25) | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
    (row_mask as u32) << 28 | (bank_mask as u32) << 24 | (bound_ctrl as u32) << 19 | (ctrl as u32) << 8 | vsrc0 as u32
}

const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xD400_0000 | (op as u32) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
    ]
}

/// A VGPR source field.
const fn v(n: u8) -> u16 {
    256 + n as u16
}

/// An inline integer constant's field (128 is 0).
const fn k(n: u16) -> u16 {
    128 + n
}

/// A wave64 compute program of `threads` threads that declares `lds_blocks` blocks of 512 B of
/// LDS and `body`, with 5 root constants in s0..s4 and a raw UAV's descriptor in s[8:11].
fn native(threads: [u32; 3], lds_blocks: u32, body: &[u32]) -> NativeProgram {
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
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
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
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | lds_blocks << 15, threads }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The packed container of `native` in this thread's form, and what crossed lanes.
fn packed(native: &NativeProgram) -> (Result<Vec<u8>, crate::ShaderError>, Vec<&'static str>) {
    let _scope = crate::xcl::BindingScope::enter(crate::xcl::XclConfig::default());
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
    let c = crate::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    lower::PACKED_SEEN.with(|s| s.borrow_mut().clear());
    let out = crate::emit::xcl::packed(&c, native, &p).expect("the packed form applies").map(|(h, _)| h.dxil);
    lower::TEST_HOST_WAVE.set(was);
    (out, lower::PACKED_SEEN.with(|s| s.take()))
}

/// [`packed`] with [`lower::flat_thread_base`] set for this thread.
fn packed_flat(native: &NativeProgram, on: bool) -> (Result<Vec<u8>, crate::ShaderError>, Vec<&'static str>) {
    let was = lower::TEST_FLAT_BASE.replace(Some(on));
    let r = packed(native);
    lower::TEST_FLAT_BASE.set(was);
    r
}

/// [`packed`] with [`lower::f64_sum`] set for this thread.
fn packed_f64(native: &NativeProgram, on: bool) -> (Result<Vec<u8>, crate::ShaderError>, Vec<&'static str>) {
    let was = lower::TEST_F64_SUM.replace(Some(on));
    let r = packed(native);
    lower::TEST_F64_SUM.set(was);
    r
}

/// `v0` is this thread's console thread id and `v1 = 4 * v0` is the dword it stores at; store
/// s`n` there.
fn store_s(n: u8) -> Vec<u32> {
    vec![
        vop2_lshl(1, k(2), 0),  // v_lshlrev_b32 v1, 2, v0
        vop1(2, 0x01, n as u16), // v_mov_b32 v2, s<n>
        0xE070_1000,
        0x8002_0201,            // buffer_store_dword v2, v1, s[8:11], 0 offen
    ]
}

/// A program of `threads` threads that stores its own console thread id: the packed form has to
/// work the id out for every thread, whatever its group.
fn thread_id_stores(threads: [u32; 3], lds_blocks: u32) -> NativeProgram {
    let body = vec![
        0xBF8C_0000,             // s_waitcnt 0 for the descriptor
        vop2_lshl(1, k(2), 0),   // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0001,             // buffer_store_dword v0, v1, s[8:11], 0 offen
    ];
    native(threads, lds_blocks, &body)
}

// ---- the console thread base ----

/// xsemu: a one-dimensional group of more than one console wave takes its console threads from
/// the host thread index (`VICTORY_XCL_PACKED_THREAD_BASE=1`), so a program that declares all
/// 32 KiB of LDS has no dword to give for the counter and needs none. The switch is off by
/// default: the counter's refusal stands.
#[test]
fn a_one_dimensional_group_takes_its_console_thread_from_the_host_thread() {
    let full = thread_id_stores([1024, 1, 1], 64);
    let StageIo::Cs(cs) = &full.io else { unreachable!() };
    assert_eq!(cs.lds_bytes(), 32768);
    let (out, _) = packed_flat(&full, false);
    assert!(matches!(out, Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::NO_CLAIM_DWORD), "{out:?}");
    let (out, _) = packed_flat(&full, true);
    assert!(out.is_ok(), "{out:?}");
    // and the same group below the LDS limit takes the host thread index too: no counter dword
    // is added, so the container has 32 KiB and not 32 KiB + 4 B of LDS
    let (with, _) = packed_flat(&thread_id_stores([1024, 1, 1], 0), true);
    let (without, _) = packed_flat(&thread_id_stores([1024, 1, 1], 0), false);
    assert!(with.is_ok() && without.is_ok(), "{with:?} {without:?}");
    assert!(with.unwrap().len() < without.unwrap().len(), "the flat base is the smaller container");
}

/// xsemu: a group of more than one console wave that is not one-dimensional keeps the counter,
/// and still refuses when its program leaves no dword for it (`VICTORY_XCL_PACKED_THREAD_BASE`
/// does not apply: the host thread index does not order those threads).
#[test]
fn a_multi_dimensional_group_keeps_the_counter_dword() {
    let full = thread_id_stores([64, 16, 1], 64);
    let StageIo::Cs(cs) = &full.io else { unreachable!() };
    assert_eq!(cs.lds_bytes(), 32768);
    for flat in [false, true] {
        let (out, _) = packed_flat(&full, flat);
        assert!(
            matches!(out, Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::NO_CLAIM_DWORD),
            "flat {flat}: {out:?}"
        );
    }
    // with a dword to spare the counter is used, and the container grows by it
    let (out, _) = packed_flat(&thread_id_stores([64, 16, 1], 0), true);
    assert!(out.is_ok(), "{out:?}");
}

/// xsemu: which base a group takes, from the group's threads, its LDS and the switch alone.
#[test]
fn the_thread_base_of_a_group_is_its_shape() {
    use lower::ThreadBase;
    let base = |threads, lds, flat| lower::thread_base(threads, lds, flat);
    // one console wave: thread 0, whatever the switch says, and no dword
    assert_eq!(base([64, 1, 1], 8192, true), Ok(ThreadBase { claim: None, flat: false }));
    assert_eq!(base([32, 2, 1], 0, true), Ok(ThreadBase { claim: None, flat: false }));
    assert_eq!(base([8, 8, 1], 0, true), Ok(ThreadBase { claim: None, flat: false }));
    // one-dimensional over one console wave: the host thread index, and no dword
    assert_eq!(base([1024, 1, 1], 8192, true), Ok(ThreadBase { claim: None, flat: true }));
    assert_eq!(base([128, 1, 1], 0, true), Ok(ThreadBase { claim: None, flat: true }));
    // off: the counter, and its refusal past 32 KiB
    assert_eq!(base([1024, 1, 1], 8192, false), Err(lower::NO_CLAIM_DWORD));
    assert_eq!(base([128, 1, 1], 0, false), Ok(ThreadBase { claim: Some(0), flat: false }));
    // multi-dimensional: the counter, and its refusal past 32 KiB
    assert_eq!(base([64, 16, 1], 8192, true), Err(lower::NO_CLAIM_DWORD));
    assert_eq!(base([8, 16, 1], 3, true), Ok(ThreadBase { claim: Some(3), flat: false }));
    // a group under a console wave still leaves the program's own LDS alone
    assert_eq!(base([1024, 1, 1], 8191, false), Ok(ThreadBase { claim: Some(8191), flat: false }));
}

// ---- the 64-bit float wave sum ----

/// `v_mov_b32 vdst, <literal>`.
fn mov_literal(vdst: u8, literal: u32) -> [u32; 2] {
    [vop1(vdst, 0x01, 255), literal]
}

/// `v_cmp_lt_u32 vcc, v0, v6` (VOPC opcode 0xC1): lanes 0..31 of the console wave.
const fn cmp_lt_u32_v0_v6() -> u32 {
    (0x3E << 25) | (0xC1 << 17) | (6 << 9) | v(0) as u32
}

/// Each lane's term: the two words `f` where the lane is below 32 and `t` where it is not
/// (`v_cndmask_b32 v2, f, t, vcc`: vcc takes the second source), into v[2:3].
fn f64_terms(f: [u32; 2], t: [u32; 2]) -> Vec<u32> {
    let mut w = vec![vop1(6, 0x01, k(32) as u16)]; // v_mov_b32 v6, 32
    w.push(cmp_lt_u32_v0_v6());
    for (dst, (a, b)) in [(2u8, (f[0], t[0])), (3, (f[1], t[1]))] {
        w.extend(mov_literal(4, a));
        w.extend(mov_literal(5, b));
        w.push(vop2_cndmask(dst, 4, 5));
    }
    w
}

/// VOP2 `v_cndmask_b32 vdst, v`a`, v`b`, vcc (opcode 0x01): vcc takes the second source.
const fn vop2_cndmask(vdst: u8, a: u8, b: u8) -> u32 {
    (0x01 << 25) | (vdst as u32) << 17 | (b as u32) << 9 | v(a) as u32
}

/// The mask-and-fold the console expands a wave sum of 64-bit floats into: the two words of each
/// lane's term masked to 0.0 outside the saved EXEC (s[12:13]), the 32-lane butterfly, the upper
/// half's total added in, and the result read from lane 0.
fn f64_sum_to_s6_s7() -> Vec<u32> {
    let mut w = vec![
        sop1(12, 0x25, 193),                     // s_or_saveexec_b64 s[12:13], -1
        vop3(2, 0x101, k(0), v(2), 12)[0],       // v_cndmask_b32 v2, 0, v2, s[12:13]
        vop3(2, 0x101, k(0), v(2), 12)[1],
        vop3(3, 0x101, k(0), v(3), 12)[0],       // v_cndmask_b32 v3, 0, v3, s[12:13]
        vop3(3, 0x101, k(0), v(3), 12)[1],
    ];
    let fold = |w: &mut Vec<u32>, a: [u32; 2]| w.extend_from_slice(&a);
    for ctrl in [0x161u16, 0x162, 0x164, 0x168] {
        w.push(vop1(4, 0x01, 0xFA)); // v_mov_b32_dpp v4, v2 row_xmask:<n>
        w.push(dpp16(2, ctrl, 0xF, 0xF, false));
        w.push(vop1(5, 0x01, 0xFA)); // v_mov_b32_dpp v5, v3 row_xmask:<n>
        w.push(dpp16(3, ctrl, 0xF, 0xF, false));
        fold(&mut w, vop3(2, 0x164, v(4), v(2), 0)); // v_add_f64 v[2:3], v[4:5], v[2:3]
    }
    w.extend(vop3(4, 0x378, v(2), k(0), k(0))); // v_permlanex16_b32 v4, v2, 0, 0
    w.extend(vop3(5, 0x378, v(3), k(0), k(0))); // v_permlanex16_b32 v5, v3, 0, 0
    w.extend(vop3(2, 0x164, v(4), v(2), 0)); // v_add_f64 v[2:3], v[4:5], v[2:3]
    w.extend(vop3(6, 0x360, v(2), k(32), 0)); // v_readlane_b32 s6, v2, 32
    w.extend(vop3(7, 0x360, v(3), k(32), 0)); // v_readlane_b32 s7, v3, 32
    w.extend(vop3(2, 0x164, 6, v(2), 0)); // v_add_f64 v[2:3], s[6:7], v[2:3]
    w.extend(vop3(6, 0x360, v(2), k(0), 0)); // v_readlane_b32 s6, v2, 0
    w.extend(vop3(7, 0x360, v(3), k(0), 0)); // v_readlane_b32 s7, v3, 0
    w.push(sop1(EXEC, 0x04, 12)); // s_mov_b64 exec, s[12:13]
    w
}

/// The sum's cases: each lane's term for lanes below 32 and for the lanes above, as the f64's two
/// words, and what the 64-lane sum of them is.
const CASES: [(&str, [u32; 2], [u32; 2]); 5] = [
    // 64 lanes of 2.0
    ("all-equal", [0x0000_0000, 0x4000_0000], [0x0000_0000, 0x4000_0000]),
    // 32 lanes of 1e16 and 32 of -1e16: each half sums exactly, the halves cancel exactly
    ("cancel", [0x37E0_8000, 0x4341_C379], [0x37E0_8000, 0xC341_C379]),
    // the lower half finite, the upper half a NaN
    ("nan-half", [0x0000_0000, 0x3FF0_0000], [0x0000_0000, 0x7FF8_0000]),
    // the lower half finite, the upper half an infinity
    ("inf-half", [0x0000_0000, 0x3FF0_0000], [0x0000_0000, 0x7FF0_0000]),
    // no lane contributes: both halves are the empty sum, +0.0
    ("empty-halves", [0x0000_0000, 0x0000_0000], [0x0000_0000, 0x0000_0000]),
];

/// A wave64 compute program of 64 threads (one console wave, the thread id in v0) whose terms
/// are v0 and v1 as the two words of an f64, and which stores the wave sum's two words.
fn f64_sum_program(case: usize) -> NativeProgram {
    let (_, f, t) = CASES[case];
    let mut body = f64_terms(f, t);
    body.extend(f64_sum_to_s6_s7());
    body.extend(store_s(6));
    body.extend(store_s(7));
    body.push(0xBF8C_0000); // s_waitcnt 0
    native([64, 1, 1], 0, &body)
}

/// The fixture is the instructions it is meant to be (the disassembler, not the lowering).
#[test]
fn the_f64_sum_fixture_is_the_console_expansion_it_models() {
    let p = ir::build(&f64_sum_program(0).code, true, 16).expect("IR");
    let text: Vec<String> = p.blocks[0].insts.iter().map(|i| i.machine.text(true)).collect();
    for (at, want) in [
        (2, "v_mov_b32 v6, 32"),
        (3, "v_cmp_lt_u32 vcc, v0, v6"),
        (6, "v_cndmask_b32 v2, v4, v5, vcc"),
        (10, "s_or_saveexec_b64 s[12:13], -1"),
        (11, "v_cndmask_b32 v2, 0, v2, s[12:13]"),
        (13, "v_mov_b32 v4, v2 row_xmask:1"),
        (14, "v_mov_b32 v5, v3 row_xmask:1"),
        (15, "v_add_f64 v[2:3], v[4:5], v[2:3]"),
        (30, "v_add_f64 v[2:3], s[6:7], v[2:3]"),
        (33, "s_mov_b64 exec, s[12:13]"),
        (35, "v_mov_b32 v2, s6"),
    ] {
        assert_eq!(text[at], want, "instruction {at}");
    }
    assert!(text.iter().any(|t| t.starts_with("v_permlanex16_b32 v4, v2")), "{text:?}");
    assert!(text.iter().any(|t| t == "v_readlane_b32 s6, v2, 32"), "{text:?}");
    assert!(text.iter().any(|t| t == "v_add_f64 v[2:3], s[6:7], v[2:3]"), "{text:?}");
    assert!(text.iter().any(|t| t == "s_mov_b64 exec, s[12:13]"), "{text:?}");
}

/// xsemu: the packed form lowers a whole-wave sum of 64-bit floats as one wave operation per
/// half and one add of the two halves (`VICTORY_XCL_PACKED_F64_SUM=1`), the split form's tree.
/// Off by default: the refusal stands.
#[test]
fn the_packed_form_lowers_a_wave_sum_of_64_bit_floats() {
    for case in 0..CASES.len() {
        let n = f64_sum_program(case);
        let (out, seen) = packed_f64(&n, true);
        assert!(out.is_ok(), "{}: {out:?}", CASES[case].0);
        // once for each word of the sum the program stores
        assert_eq!(seen, ["wave sum of 64-bit floats"; 2], "{}", CASES[case].0);
    }
    let n = f64_sum_program(0);
    let (out, _) = packed_f64(&n, false);
    assert!(matches!(out, Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::F64_SUM), "{out:?}");

}

/// xsemu: the split form lowers the same program, so the two containers can be compared bit for
/// bit (`xdiff`). Written out with `XCL_PACKED_FIXTURES` set, for that run.
#[test]
#[ignore = "writes files; run with XCL_PACKED_FIXTURES set"]
fn xsemu_write_the_packed_and_split_containers_of_the_fixtures() {
    let dir = std::path::PathBuf::from(std::env::var_os("XCL_PACKED_FIXTURES").expect("XCL_PACKED_FIXTURES"));
    std::fs::create_dir_all(&dir).expect("the folder");
    for case in 0..CASES.len() {
        let name = format!("f64-{}", CASES[case].0);
        let n = f64_sum_program(case);
        let split = split(&n).expect("the split form");
        std::fs::write(dir.join(format!("{name}.split.dxil")), split).expect("write");
        let f64 = lower::TEST_F64_SUM.replace(Some(true));
        let packed = packed(&n).0.expect("the packed form");
        lower::TEST_F64_SUM.set(f64);
        std::fs::write(dir.join(format!("{name}.packed.dxil")), packed).expect("write");
    }
    // the console thread ids of a 1-D group, from the host thread index and from the counter
    let n = thread_id_stores([1024, 1, 1], 0);
    let flat = lower::TEST_FLAT_BASE.replace(Some(true));
    let packed = packed(&n).0.expect("the flat base");
    lower::TEST_FLAT_BASE.set(Some(false));
    let counter = packed_flat(&n, false).0.expect("the counter base");
    lower::TEST_FLAT_BASE.set(flat);
    std::fs::write(dir.join("threads-flat.dxil"), packed).expect("write");
    std::fs::write(dir.join("threads-counter.dxil"), counter).expect("write");
    // a two-console-wave one-dimensional group that translates at the base: the counter's own
    // container, for the byte-identity check
    let n = thread_id_stores([128, 1, 1], 63);
    std::fs::write(dir.join("counter128.dxil"), packed_flat(&n, false).0.expect("the counter base")).expect("write");
}

/// The container the ladder `emit` builds for `native` without the packed form: the split form
/// for a wave64 compute program on 32-lane waves.
fn split(native: &NativeProgram) -> Result<Vec<u8>, crate::ShaderError> {
    let _scope = crate::xcl::BindingScope::enter(crate::xcl::XclConfig::default());
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let io = crate::emit::io::raise(&p, native)?;
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let c = crate::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = crate::emit::emit_program(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    r.map(|(shader, _)| shader.dxil)
}

/// xsemu: the split form lowers the f64 sum, so the packed form's result can be compared with
/// it bit for bit.
#[test]
fn the_split_form_lowers_the_f64_sum_too() {
    for case in 0..CASES.len() {
        assert!(split(&f64_sum_program(case)).is_ok(), "{}", CASES[case].0);
    }
}




