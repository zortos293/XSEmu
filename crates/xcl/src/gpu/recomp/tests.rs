//! Instruction semantics checked on the real GPU: small RDNA2 programs are assembled,
//! recompiled to HLSL, compiled with DXC and run as kernels. Skipped (with a note) when
//! there is no D3D12 GPU with SM 6.6 + wave64 or no DXC. Tests that use `gpu_waves` also run
//! as wave32 programs, and only need a GPU and DXC.

use super::runner::{self, KernelState};
use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;

/// A GPU and DXC for the instruction tests. Wave64 programs run natively where the adapter
/// has 64-lane waves and in the split form (`runner::options`) where it has only 32 (an
/// adapter with neither, such as WARP, skips).
fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), super::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!("GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err());
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

/// 64-lane hardware waves (a test of how a program runs in them).
fn wave64_hardware() -> bool {
    let w = runner::wave64_supported() == Ok(true);
    if !w {
        println!("no 64-lane waves on this GPU – skipping");
        runner::skip_test("no wave64");
    }
    w
}

/// `o` for a program run on this adapter (the split wave64 form where it is needed).
fn on_gpu(o: Options) -> Options {
    Options { split_wave64: runner::split_for(&o.mode), ..o }
}

/// `runner::mesh_draw` needs mesh shaders (not every adapter with wave64 has them).
fn mesh_ready() -> bool {
    match runner::mesh_shaders_supported() {
        Ok(true) => true,
        r => {
            println!("GPU without mesh shaders – skipping ({r:?})");
            runner::skip_test(if r.is_err() { "no GPU" } else { "no mesh shaders" });
            false
        }
    }
}

/// The wave sizes a per-lane test runs in, as the `wave64` option: true where the GPU has
/// wave64, and false on every GPU, so the test also runs on NVIDIA. Empty without a GPU or DXC.
fn gpu_waves() -> Vec<bool> {
    if !gpu_ready() {
        return Vec::new();
    }
    if runner::wave64_supported() == Ok(true) { vec![true, false] } else { vec![false] }
}

/// A kernel compiled as a wave64 or wave32 program, and the code as memory from address 0,
/// where `runner::run_program` puts it.
fn kernel_wave(code: &[u8], wave64: bool) -> (Vec<u8>, Vec<u32>) {
    let t = translate(code, &Options { wave64, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    (dxil, code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// `runner::run_program` as a wave64 or wave32 program, with the number of lanes that ran.
fn run_wave(code: &[u8], st: &KernelState, wave64: bool) -> (runner::KernelResult, usize) {
    let (dxil, mem) = kernel_wave(code, wave64);
    (runner::run(&dxil, st, &mem).unwrap(), if wave64 { 64 } else { 32 })
}

/// Assemble a program from (name, encoding, fields, literal).
fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

const V: u64 = 256; // VALU source code of v0

#[test]
fn quad_sad_preserves_windows_accumulator_and_exec() {
    let code = prog(&[
        ("V_QSAD_PK_U16_U8", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 6)], None),
        ("V_MQSAD_PK_U16_U8", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 6)], None),
        ("V_MQSAD_U32_U8", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 13), ("CLAMP", 1)], None),
    ]);
    let translated = translate(&code, &Options::default()).unwrap();
    assert!(translated.complete(), "{:?}", translated.unsupported_why);
    if !gpu_ready() {
        return;
    }

    let mut state = KernelState::default();
    state.sgpr[126] = 0x5555_5555;
    state.sgpr[127] = 0xAAAA_AAAA;
    for v in &mut state.vgpr {
        v[2] = 0x4030_2010;
        v[3] = 0xFF60_5000;
        v[4] = 0x4000_2015;
        v[6] = 0x0002_0001;
        v[7] = 0x0004_0003;
        v[13..17].copy_from_slice(&[u32::MAX, 20, 30, 40]);
    }
    for (wave64, lanes) in [(false, 32), (true, 64)] {
        let opts = on_gpu(Options { wave64, ..Default::default() });
        let translated = translate(&code, &opts).unwrap();
        let dxil = super::dxc::compile(&translated.hlsl, "main", "cs_6_6", &[]).unwrap();
        let out = runner::run(&dxil, &state, &[]).unwrap();
        for (lane, v) in out.vgpr.iter().enumerate().take(lanes) {
            let exec = state.sgpr[126 + lane / 32];
            if exec >> (lane % 32) & 1 == 0 {
                assert_eq!(&v[8..17], &state.vgpr[lane][8..17]);
                continue;
            }
            assert_eq!(&v[8..12], &[54 | (157 << 16), 78 | (191 << 16), 6 | (93 << 16), 78 | (111 << 16)]);
            assert_eq!(&v[12..17], &[u32::MAX, 111, 105, 147, 40]);
        }
    }
}

#[test]
fn quad_sad_rejects_register_spans_past_v255() {
    for (op, dst, src0, src2) in [
        ("V_QSAD_PK_U16_U8", 255, V, V),
        ("V_MQSAD_PK_U16_U8", 4, V + 255, V),
        ("V_MQSAD_PK_U16_U8", 4, V, V + 255),
        ("V_MQSAD_U32_U8", 253, V, V),
        ("V_MQSAD_U32_U8", 4, V, V + 253),
        ("V_MQSAD_U32_U8", 4, V, 128),
    ] {
        let fields = [("VDST", dst), ("SRC0", src0), ("SRC1", V + 2), ("SRC2", src2)];
        let code = prog(&[(op, "ENC_VOP3", &fields, None)]);
        assert!(!translate(&code, &Options::default()).unwrap().complete(), "{op}: {fields:?}");
    }
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn gpu_valu_minmax_orders_signed_zeros() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 8), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_MIN3_F32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_MAX3_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_MIN_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 3), ("VSRC1", 4)], None),
        ("V_MAX_F16", "ENC_VOP2", &[("VDST", 12), ("SRC0", V + 3), ("VSRC1", 4)], None),
        ("V_MIN3_F16", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", V + 5)], None),
        ("V_MAX3_F16", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", V + 5)], None),
        ("V_PK_MIN_F16", "ENC_VOP3P", &[("VDST", 15), ("SRC0", V + 3), ("SRC1", V + 4), ("OP_SEL_HI", 3)], None),
        ("V_PK_MAX_F16", "ENC_VOP3P", &[("VDST", 16), ("SRC0", V + 3), ("SRC1", V + 4), ("OP_SEL_HI", 3)], None),
    ]);
    let mut st = KernelState::default();
    for (lane, v) in st.vgpr.iter_mut().enumerate() {
        for i in 0..3 {
            v[i] = ((lane >> i) as u32 & 1) << 31;
            let lo = v[i] >> 16;
            v[3 + i] = ((lo ^ 0x8000) << 16) | lo;
        }
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for (v, orig) in r.vgpr.iter().zip(st.vgpr.iter()) {
        let [a, b, c] = [orig[0], orig[1], orig[2]];
        assert_eq!(&v[7..11], &[a | b, a & b, a | b | c, a & b & c]);
        assert_eq!(&v[11..15], &[(a | b) >> 16, (a & b) >> 16, (a | b | c) >> 16, (a & b & c) >> 16]);
        let [a, b] = [orig[3], orig[4]];
        assert_eq!(&v[15..17], &[a | b, a & b]);
    }
}

#[test]
fn geometry_wave32_rejects_cross_half_operations() {
    let one = |name: &str, enc: &str, fields: &[(&str, u64)]| encode(name, enc, fields, None).unwrap();

    let lane = one("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)]);
    assert!(super::geometry_wave32_hazards(&lane).contains("cross-half lane exchange"));

    let branch = one("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)]);
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a whole-wave mask"));

    let reduce = one("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]);
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));

    let compare = one("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 126), ("SSRC1", 106)]);
    assert!(super::geometry_wave32_hazards(&compare).contains("scalar condition from a whole-wave mask"));

    let mut scalar = one("S_AND_B64", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 126), ("SSRC1", 106)]);
    scalar.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&scalar).contains("branch on a half-local scalar condition"));

    // a prefix count over EXEC, VCC or a v_cmp result in an SGPR pair; the lane index is fine
    for mask in [126, 107, 10] {
        let count = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", mask), ("SRC1", 128)]);
        assert!(super::geometry_wave32_hazards(&count).contains("prefix count over a whole-wave mask"), "SRC0 {mask}");
    }
    let mut lane = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", 128)]);
    lane.extend(one("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", V + 1)]));
    assert!(super::geometry_wave32_hazards(&lane).is_empty());

    // a v_cmp into s[0:1]: a mask, its 32-bit words too
    let cmp = one("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 168), ("SRC1", V)]);
    let mut reduce = cmp.clone();
    reduce.extend(one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut reduce = one("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)]);
    reduce.extend(one("S_FLBIT_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 127)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut branch = cmp.clone();
    branch.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]));
    branch.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a half-local scalar condition"));
    // the same scalar work on a non-mask SGPR stays accepted
    let mut plain = one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)]);
    plain.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]));
    plain.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    plain.extend(one("S_ENDPGM", "ENC_SOPP", &[]));
    assert!(super::geometry_wave32_hazards(&plain).is_empty());
}

#[test]
fn geometry_wave_path_selection_preserves_native_wave64() {
    assert!(super::pso::wave32_only(32, 32));
    assert!(!super::pso::wave32_only(32, 64));
    assert!(!super::pso::wave32_only(64, 64));
    // a VS / PS: the driver picks, so any range with 32 lanes gets the 32-lane form (#42)
    assert!(super::pso::graphics_wave32(32, 32));
    assert!(super::pso::graphics_wave32(32, 64));
    assert!(!super::pso::graphics_wave32(64, 64));
}

#[test]
fn structured_branches_recheck_pending_targets_and_enter_else_targets() {
    if !gpu_ready() {
        return;
    }
    let add = |r| ("S_ADD_U32", "ENC_SOP2", vec![("SDST", r), ("SSRC0", r), ("SSRC1", 129)]);
    let bit = |r| ("S_BITCMP1_B32", "ENC_SOPC", vec![("SSRC0", r), ("SSRC1", 0)]);
    let branch = |n, k| (n, "ENC_SOPP", vec![("SIMM16", k)]);
    let cond = |k| branch("S_CBRANCH_SCC1", k);
    let exits = vec![
        add(0), bit(10), cond(9),
        add(1), bit(11), cond(6),
        add(2), bit(12), cond(1),
        add(4),
        ("S_CMP_LT_U32", "ENC_SOPC", vec![("SSRC0", 0), ("SSRC1", 131)]),
        cond(0xFFF4),
        add(3),
    ];
    let into_else = vec![
        add(0), bit(10), cond(5),
        add(1), bit(11), cond(2),
        add(2), branch("S_BRANCH", 1),
        add(3),
        add(4),
    ];
    for (instructions, want) in [(exits, [1, 1, 0, 1, 0]), (into_else, [1, 1, 0, 1, 1])] {
        let lines: Vec<_> = instructions.iter().map(|(name, enc, fields)| (*name, *enc, fields.as_slice(), None)).collect();
        let code = prog(&lines);
        let mem: Vec<_> = code.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut st = KernelState::default();
        st.sgpr[11] = 2;
        for structured in [true, false] {
            let t = translate(&code, &Options { wave64: false, structured, ..Default::default() }).unwrap();
            assert!(t.complete(), "{:?}", t.unsupported_why);
            let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let out = runner::run(&dxil, &st, &mem).unwrap();
            assert_eq!(out.sgpr[..5], want, "structured={structured}\n{}", t.hlsl);
        }
    }
}

/// `XCL_LOOP_GUARD`: the guard is the translation's (a `#define` ahead of the prelude), and a
/// translation names the loops the guard may cut short in a working program: all but those
/// whose latch compares a constant-stepped counter with a constant within the guard.
#[test]
fn loops_without_a_static_bound_within_the_guard_are_named() {
    let counted = |cmp: (&'static str, u64, Option<u32>), branch: &'static str, back: u64| {
        prog(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            (cmp.0, "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", cmp.1)], cmp.2),
            (branch, "ENC_SOPP", &[("SIMM16", back)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
        ])
    };
    let loops = |code: &[u8]| {
        let t = translate(code, &Options::default()).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported);
        assert!(t.hlsl.contains(&format!("#define XCL_LOOP_GUARD {}u\n", super::loop_guard())) && t.hlsl.contains("static uint xcl_budget = XCL_LOOP_GUARD;"), "{}", t.hlsl);
        assert!(t.hlsl.contains("for (;;)") && t.hlsl.contains("if (xcl_budget == 0u) { xcl_trip(0u); break; }"), "{}", t.hlsl);
        t.long_loops
    };
    // s0 = 0; do { s0 += 1 } while (s0 < 3)
    assert_eq!(loops(&counted(("S_CMP_LT_U32", 131, None), "S_CBRANCH_SCC1", 0xFFFD)), Vec::<String>::new());
    // … while (s0 != s6): the count is a register
    let by_register = loops(&counted(("S_CMP_EQ_U32", 6, None), "S_CBRANCH_SCC0", 0xFFFD));
    assert!(by_register.len() == 1 && by_register[0].contains("no static bound (`s_cmp_eq_u32 s0, s6`)"), "{by_register:?}");
    // … while (s0 < 0x100000): static, beyond any guard in use
    let far = loops(&counted(("S_CMP_LT_U32", 255, Some(0x10_0000)), "S_CBRANCH_SCC1", 0xFFFC));
    assert!(far.len() == 1 && far[0].ends_with(": 1048577 iterations"), "{far:?}");
    assert!(super::loop_guard() < 0x10_0000);
}

/// A guard that gives up counts its lanes in the words at io[`IO_TRIP_AT`] (`xcl_trip`): a
/// loop that never meets its exit runs out of the loop budget in every lane, counted once each.
#[test]
fn gpu_a_spent_loop_budget_is_counted_once_per_lane() {
    if !gpu_ready() {
        return;
    }
    // do { s10 += 1 } while (s10 != 0)
    let p = prog(&[
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 3 };
    let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let (at, views) = (format!("uint at = io.Load({}u);", super::IO_TRIP_AT * 4), format!("ResourceDescriptorHeap[io.Load({}u) + io.Load({}u)]", super::IO_MEM_VIEWS * 4, super::IO_TRIP_WINDOW * 4));
    assert!(t.hlsl.contains(&at) && t.hlsl.contains(&views), "{}", t.hlsl);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let mut io = vec![0u32; 128];
    io[super::IO_TRIP_AT] = 0x40;
    let out = runner::dispatch(&dxil, [2, 1, 1], &io, &vec![0u32; 64]).unwrap();
    assert_eq!(out[0x10..0x14], [128, 0, 0, 0], "{} per lane of 2 groups of 64", super::TRIP_KINDS[0]);
}

/// `f` with the base form's loop guard (`XCL_RECOMP_NO_LOOP_REGIONS`): for the tests of its one budget
/// and its wave-wide exit.
fn base_loop_guard<R>(f: impl FnOnce() -> R) -> R {
    super::NO_LOOP_REGIONS.set(true);
    let r = f();
    super::NO_LOOP_REGIONS.set(false);
    r
}

/// `f` with region budgets (`XCL_LOOP_GUARD_REGIONS`).
fn with_loop_regions<R>(f: impl FnOnce() -> R) -> R {
    super::LOOP_REGIONS.set(true);
    let r = f();
    super::LOOP_REGIONS.set(false);
    r
}

/// With region budgets, a program's own spin that gives up at the loop budget, or just before it, must not leave the
/// loops after it short of iterations: a retail FSR pass then published a flag built from a
/// register its main loop never set, and the pass waiting on it spun for 260 ms a frame. Both
/// HLSL forms (upstream ivall/xcl 287a931b; `loop_guard_tests` runs the same without a GPU).
#[test]
fn gpu_a_loop_after_a_spin_to_the_budget_still_runs() {
    if !gpu_ready() {
        return;
    }
    let guard = super::loop_guard();
    for (spin, n, nested) in [(guard, 3, false), (guard - 16, 1000, false), (guard, 3, true)] {
        let code = super::loop_guard_tests::spin_then_loop(spin, n, nested);
        for structured in [true, false] {
            let opt = Options { structured, ..runner::options(Mode::Kernel) };
            let t = with_loop_regions(|| translate(&code, &opt)).unwrap();
            let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let r = runner::run(&dxil, &KernelState::default(), &[]).unwrap();
            let what = format!("spin {spin:#x}, loop {n}, nested {nested}, structured {structured}");
            assert_eq!(r.sgpr[0], spin, "{what}: the spin ran to its own end");
            assert_eq!((r.sgpr[1], r.sgpr[2]), (n, 7), "{what}: the loop after it ran all its iterations");
        }
    }
}

/// With region budgets, loops that never end still stop: `LOOP_BUDGETS_IN_ALL` budgets in all, then every loop leaves
/// (upstream ivall/xcl 287a931b).
#[test]
fn gpu_endless_loops_in_many_regions_stop_at_the_total() {
    if !gpu_ready() {
        return;
    }
    // s0 = s1 = 0; six regions, each `s0 += 1` until s0 == 0 (2^32 tries); s1 counts the regions reached
    let mut list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = vec![("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None), ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)], None)];
    for _ in 0..6 {
        list.push(("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None));
        list.push(("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None));
        list.push(("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None));
        list.push(("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None));
    }
    let t = with_loop_regions(|| translate(&prog(&list), &runner::options(Mode::Kernel))).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let r = runner::run(&dxil, &KernelState::default(), &[]).unwrap();
    assert_eq!(r.sgpr[1], 6, "every region was reached");
    assert_eq!(r.sgpr[0], super::LOOP_BUDGETS_IN_ALL * super::loop_guard(), "the program's total, and no more");
}

fn packed_compute_source(code: &[u8], threads: [u32; 3]) -> String {
    let cs = super::CsInterface { threads, user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 3 };
    let t = translate(code, &Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: false, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_x_meet(") && !t.hlsl.contains("XCL_X_SPIN"));
    t.hlsl
}

fn run_packed_compute(hlsl: &str, slots: usize) -> Vec<u32> {
    let dxil = super::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let mut io = vec![0u32; 128];
    io[0] = 0x100;
    io[super::IO_TRIP_AT] = 0x40;
    runner::dispatch(&dxil, [1, 1, 1], &io, &vec![0u32; 64 + slots]).unwrap()
}

#[test]
fn gpu_packed_compute_cross_half_exchange_in_a_branch_and_partial_wave() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", 131), ("SRC2", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 133), ("SRC2", V + 10)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 10)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 10), ("SRC1", 160)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 10)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 9)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 160), ("VSRC1", 10)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 1)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 3), ("ADDR", 1), ("DATA0", 3)], None),
        ("V_CMPX_LE_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 10)], None),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 10)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 11), ("VSRC1", 3)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 11), ("VSRC1", 3)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 10), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let hlsl = packed_compute_source(&code, [8, 4, 3]);
    assert!(hlsl.contains("[numthreads(64, 1, 1)]"));
    assert!(!hlsl.contains("GroupMemoryBarrier") && !hlsl.contains("groupshared"));
    let out = run_packed_compute(&hlsl, 128);
    assert_eq!(out[16..20], [0; 4]);
    for i in 0..128 {
        let expected = if i < 64 { (i ^ 32) + 64 } else if i < 96 { i } else { 0 };
        assert_eq!(out[64 + i], expected as u32, "thread {i}");
    }
}

#[test]
fn compute_flat_memory_is_coherent_only_for_flagged_accesses_atomics_or_invalidation() {
    let plain = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("SOFFSET", 125)][..], None);
    let cases: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 0), ("SOFFSET", 125), ("GLC", 1)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 0), ("SOFFSET", 125), ("DLC", 1)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("SLC", 1), ("SOFFSET", 128)], None),
        ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("SADDR", 0x7d)], None),
        ("FLAT_ATOMIC_ADD", "ENC_FLAT", &[], None),
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("SOFFSET", 128)], None),
        ("S_DCACHE_INV", "ENC_SMEM", &[], None),
        ("S_GL1_INV", "ENC_SMEM", &[], None),
        ("BUFFER_GL0_INV", "ENC_MUBUF", &[], None),
        ("BUFFER_GL1_INV", "ENC_MUBUF", &[], None),
    ];
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 };
    for mem_windows in [1, 4] {
        let opt = Options { mode: Mode::Compute(cs.clone()), wave64: true, split_wave64: true, mem_windows, ..Default::default() };
        let hlsl = translate(&prog(&[plain]), &opt).unwrap().hlsl;
        assert!(!hlsl.contains("globallycoherent RWByteAddressBuffer"));
        for &op in cases {
            let t = translate(&prog(&[plain, op]), &opt).unwrap();
            assert!(t.complete(), "{}: {:?}", op.0, t.unsupported_why);
            assert!(t.hlsl.contains("globallycoherent RWByteAddressBuffer"), "{}", op.0);
        }
    }
}

#[test]
fn gpu_packed_compute_scalar_load_chain() {
    let code = prog(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 1), ("OFFSET", 16), ("SOFFSET", 125), ("GLC", 1)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 1), ("OFFSET", 20), ("SOFFSET", 125), ("DLC", 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 12), ("SSRC1", 13)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 14)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 15)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 16)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 17)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 18)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let hlsl = packed_compute_source(&code, [96, 1, 1]);
    for s in 8..16 {
        assert!(!hlsl.contains(&format!("s{s} = WaveReadLaneFirst(")));
    }
    assert!(hlsl.contains("s16 = WaveReadLaneFirst("));
    assert!(hlsl.contains("s17 = WaveReadLaneFirst("));
    if !gpu_ready() {
        return;
    }
    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut io = vec![0u32; 128];
    io[0] = 0x100;
    let mut mem = vec![0u32; 160];
    mem[..6].copy_from_slice(&[64, 0, 12, 0, 17, 23]);
    mem[16..20].copy_from_slice(&[101, 103, 107, 1009]);
    let out = runner::dispatch(&dxil, [1, 1, 1], &io, &mem).unwrap();
    assert_eq!(out[64..160], [351; 96]);
}

/// High half of the addresses of memory window 0 in the runs of the load chain over several
/// windows. A program with several windows drops a store through a pointer whose high half is
/// 0 (`Em::window_write`), so the chain's pointers carry this one.
const CHAIN_WINDOW: u32 = 16;

/// `io` and memory of the load chain over several windows: its table pointer (s2, s3), its
/// buffer descriptor and its store address are in window 0.
fn chain_state(threads: usize) -> (Vec<u32>, Vec<u32>) {
    let mut io = vec![0u32; 128];
    io[0] = 0x100;
    io[3] = CHAIN_WINDOW;
    io[super::IO_UMA_BASE_HI] = CHAIN_WINDOW;
    let mut mem = vec![0u32; 64 + threads];
    mem[..6].copy_from_slice(&[64, CHAIN_WINDOW, 12, 0, 17, 23]);
    mem[16..20].copy_from_slice(&[101, 103, 107, 1009]);
    (io, mem)
}

/// The memory a compute program of `runner::dispatch`'s ABI leaves in window 0, in the
/// interpreter (no GPU): what the test's state makes the program do is checked on every machine.
fn interpreted(hlsl: &str, io: &[u32], mem: &[u32]) -> Vec<u32> {
    use xcl_dxil::interp::{self, res, Input};
    let hlsl = hlsl.replace("vector<uint64_t, 4>(", "uint64_t4(").replace("vector<uint, 4>(", "uint4(");
    let p = interp::load(&xcl_dxil::compile(&hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap();
    let word = |ident: u32, w: usize| (ident as u64) << 32 | w as u64;
    // the heap slot of the windows is the interpreter's: the program reads it from io
    let mut patches: Vec<(u64, u32)> = io.iter().enumerate().filter(|(w, _)| *w != super::IO_MEM_VIEWS).map(|(w, v)| (word(res::IDENT_IO, w), *v)).collect();
    let cfg = interp::Config::default();
    let trace = interp::run_input(&p, &Input { seed: 0, patches: patches.clone() }, &cfg, true).1.unwrap();
    let views = trace.reads.iter().find(|r| r.0 == word(res::IDENT_IO, super::IO_MEM_VIEWS)).expect("the program reads its memory views").1;
    let window = res::IDENT_HEAP | views;
    patches.extend(mem.iter().enumerate().map(|(w, v)| (word(window, w), *v)));
    let run = interp::run_input(&p, &Input { seed: 0, patches }, &cfg, false).0;
    assert!(run.stop.is_none() && run.ub == 0, "{:?} {}", run.stop, run.ub);
    let mut after = mem.to_vec();
    for w in run.written.iter().filter(|w| w.0 == 0 && w.1 == window) {
        after[w.2 as usize / 4] = w.3[0];
    }
    after
}

/// The load chain of `gpu_packed_compute_scalar_load_chain` over four memory windows, packed and per
/// wave, with its memory functions kept out of line (`emit::outlined`; DXC compiles such a
/// program and keeps the calls): it leaves the memory the plain program leaves, with the results
/// of the kept functions handed back through `out` parameters and through statics
/// (`emit::outline_statics`).
#[test]
fn gpu_outlined_memory_functions_compute_the_same() {
    let code = prog(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 1), ("OFFSET", 16), ("SOFFSET", 125), ("GLC", 1)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 1), ("OFFSET", 20), ("SOFFSET", 125), ("DLC", 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 12), ("SSRC1", 13)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 14)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 15)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 16)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 17)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128 + CHAIN_WINDOW as u64)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 18)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let forms: Vec<(bool, usize, String, [String; 2])> = [true, false]
        .into_iter()
        .map(|packed| {
            let threads = if packed { 96 } else { 64 };
            let cs = super::CsInterface { threads: [threads as u32, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 3 };
            let opt = if packed {
                Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: false, mem_windows: 4, ..Default::default() }
            } else {
                Options { mode: Mode::Compute(cs), wave64: false, split_wave64: true, mem_windows: 4, ..Default::default() }
            };
            let hlsl = |outline: bool, statics: bool| {
                super::emit::OUTLINE.set(Some(outline));
                super::emit::OUTLINE_FORM.set(Some((statics, super::emit::OutlineAtomics::All)));
                let t = translate(&code, &opt);
                super::emit::OUTLINE.set(None);
                super::emit::OUTLINE_FORM.set(None);
                let t = t.unwrap();
                assert!(t.complete(), "{:?}", t.unsupported_why);
                t.hlsl
            };
            let (plain, out, stat) = (hlsl(false, false), hlsl(true, false), hlsl(true, true));
            assert!(!plain.contains("[noinline]"), "{plain}");
            for (text, kept) in [(&out, ["xcl_mLoad_n", "xcl_mLoad4_n", "xcl_mStore_n"]), (&stat, ["xcl_mLoad_r", "xcl_mLoad4_r", "xcl_mStore_n"])] {
                for f in kept {
                    assert!(text.contains(&format!("[noinline] void {f}(")), "{f}
{text}");
                }
            }
            assert!(out.contains("out uint xr0") && !stat.contains("out uint xr0") && stat.contains("static uint xcl_ret_uint_0;"), "{stat}");
            (packed, threads, plain, [out, stat])
        })
        .collect();
    const FORMS: [&str; 2] = ["out parameters", "statics"];
    for (packed, threads, plain, outlined) in &forms {
        let (io, mem) = chain_state(*threads);
        let x = interpreted(plain, &io, &mem);
        assert_eq!(x[64..64 + threads], vec![351; *threads], "interpreted, packed {packed}: the plain program");
        for (out, form) in outlined.iter().zip(FORMS) {
            assert_eq!(x, interpreted(out, &io, &mem), "interpreted, packed {packed}: the outlined program, results through {form}");
        }
    }
    if !gpu_ready() {
        return;
    }
    for (packed, threads, plain, outlined) in forms {
        let run = |hlsl: &str| {
            let dxil = super::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{hlsl}"));
            let (io, mem) = chain_state(threads);
            runner::dispatch(&dxil, [1, 1, 1], &io, &mem).unwrap()
        };
        let a = run(&plain);
        assert_eq!(a[64..64 + threads], vec![351; threads], "packed {packed}: the plain program");
        for (out, form) in outlined.iter().zip(FORMS) {
            assert_eq!(a, run(out), "packed {packed}: the outlined program, results through {form}");
        }
    }
}

/// Passes of the unrolled part of `outline_gate_code` and of its loop.
const GATE_REPS: usize = 128;
const GATE_LOOP: u32 = 8;

/// A program of the shape of the title's big batch-cull programs, for what
/// `gpu_outlined_memory_functions_compute_the_same` leaves out: `GATE_REPS` unrolled passes, each
/// with a four-address buffer load (`xcl_loadx4`, a kept function that calls a kept function in
/// every second pass, where the buffer's dwords are in two 64 KiB blocks), a four-dword and a one-dword load, a load, an atomic add whose old value is used and a store on the
/// thread's own words, and one of six atomics on words every thread shares; then a counted loop of
/// `GATE_LOOP` passes around a load, an atomic and a load and store. Outlined, that is about 900
/// calls in `main` and, with `out` parameters, about 1,400 results handed back through pointers.
fn outline_gate_code() -> Vec<u8> {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &str, f: &[(&str, u64)]| i(n, "ENC_FLAT_GLBL", &[f, &[("SADDR", 0x7D)]].concat());
    let mov = |d: u64, s: u64| i("V_MOV_B32", "ENC_VOP1", &[("VDST", d), ("SRC0", s)]);
    let sadd = |d: u64, s: u64| i("S_ADD_U32", "ENC_SOP2", &[("SDST", d), ("SSRC0", d), ("SSRC1", s)]);
    let vadd = |d: u64, s: u64| i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + s), ("VSRC1", d)]);
    let window = 128 + CHAIN_WINDOW as u64;
    // v[1:2]: the thread's words; v[8:9]: the shared words; v10 = 1 << (thread & 31), v11 = 1, v12 = !v10, v13 = thread + 5
    let mut code = [
        i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("SOFFSET", 125)]),
        i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 28), ("SBASE", 1), ("OFFSET", 32), ("SOFFSET", 125)]),
        i("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)]),
        mov(2, window),
        mov(8, 128),
        mov(9, window),
        mov(11, 129),
        i("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V), ("VSRC1", 11)]),
        i("V_NOT_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 10)]),
        i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 133), ("VSRC1", 0)]),
        mov(6, 128),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 18), ("SSRC0", 128)]),
    ]
    .concat();
    const SHARED: [(&str, u64); 6] = [("GLOBAL_ATOMIC_ADD", 11), ("GLOBAL_ATOMIC_OR", 10), ("GLOBAL_ATOMIC_UMIN", 13), ("GLOBAL_ATOMIC_AND", 12), ("GLOBAL_ATOMIC_UMAX", 13), ("GLOBAL_ATOMIC_XOR", 10)];
    for r in 0..GATE_REPS {
        let (op, data) = SHARED[r % 6];
        code.extend(
            [
                // s[8:11]: a buffer in one 64 KiB block; s[28:31]: one in two, read a dword a call
                i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", if r % 2 == 0 { 4 } else { 14 }), ("SOFFSET", 125)]),
                i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 1), ("SOFFSET", 125)]),
                i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 1), ("OFFSET", 16), ("SOFFSET", 125)]),
                sadd(18, 12),
                sadd(18, 24),
                sadd(18, 16),
                glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 1), ("OFFSET", 0x200)]),
                glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 5), ("ADDR", 1), ("DATA", 11), ("OFFSET", 0x200), ("GLC", 1)]),
                vadd(6, 4),
                vadd(6, 5),
                glbl("GLOBAL_STORE_DWORD", &[("ADDR", 1), ("DATA", 6), ("OFFSET", 0)]),
                glbl(op, &[("ADDR", 8), ("DATA", data), ("OFFSET", 128 + 4 * (r as u64 % 6))]),
            ]
            .concat(),
        );
    }
    code.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)]));
    let body = [
        i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 1), ("OFFSET", 20), ("SOFFSET", 125)]),
        sadd(18, 17),
        glbl("GLOBAL_ATOMIC_ADD", &[("ADDR", 8), ("DATA", 11), ("OFFSET", 128)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 1), ("OFFSET", 0)]),
        vadd(4, 11),
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 1), ("DATA", 4), ("OFFSET", 0)]),
        i("S_ADD_U32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 20), ("SSRC1", 129)]),
        i("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 20), ("SSRC1", 128 + GATE_LOOP as u64)]),
    ]
    .concat();
    code.extend(&body);
    code.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", (-((body.len() as i64 + 4) / 4)) as u16 as u64)]));
    code.extend(mov(3, 18));
    code.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 1), ("DATA", 3), ("OFFSET", 0x400)]));
    code.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
    code
}

/// `io` and memory of `outline_gate_code` (`chain_state`'s table and a second buffer at 0xFFF8; words
/// 32..38 are the shared ones, 64.., 192.. and 320.. the threads' own), and the memory the program leaves.
fn outline_gate_state(threads: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (io, mut mem) = chain_state(threads);
    mem.resize(0x4100, 0);
    mem[8..12].copy_from_slice(&[0xFFF8, CHAIN_WINDOW, 12, 0]);
    mem[0x3FFE..0x4001].copy_from_slice(&[211, 223, 227]);
    mem[34] = u32::MAX;
    mem[35] = u32::MAX;
    let (mut after, n) = (mem.clone(), GATE_REPS as u32);
    for t in 0..threads {
        let (bit, id) = (1u32 << (t & 31), t as u32 + 5);
        for r in 0..GATE_REPS {
            match r % 6 {
                0 => after[32] += 1,
                1 => after[33] |= bit,
                2 => after[34] = after[34].min(id),
                3 => after[35] &= !bit,
                4 => after[36] = after[36].max(id),
                _ => after[37] ^= bit,
            }
        }
        after[32] += GATE_LOOP;
        after[64 + t] = n * (n - 1) + GATE_LOOP;
        after[192 + t] = n;
        after[320 + t] = n / 2 * (101 + 211) + n * (64 + 17) + GATE_LOOP * 23;
    }
    (io, mem, after)
}

/// A form of outlining: `emit::outline_statics`, `emit::outline_atomics`, `emit::outline_flat`.
type OutlineForm = (bool, super::emit::OutlineAtomics, bool);

/// `code` as the compute program of a driver container (`pso::fspx`): 64 threads a group, 4 user
/// registers, three thread id components.
fn fspx_container(code: &[u8]) -> Vec<u8> {
    let mut head = vec![0u8; 0x1C];
    head[0x0C..0x12].copy_from_slice(&[64, 0, 1, 0, 1, 0]);
    head[0x18..].copy_from_slice(&(4u32 << 1 | 2 << 11).to_le_bytes());
    let body = [vec![0u8; 0x24], code.to_vec()].concat();
    let table = 0x10 + 2 * 16;
    let mut part = [b"FSPX".as_slice(), &[0; 4], &2u32.to_le_bytes(), &[0; 4]].concat();
    for (kind, at, len) in [(0u32, table, head.len()), (6, table + head.len(), body.len())] {
        part.extend([kind, at as u32, len as u32, 0].iter().flat_map(|v| v.to_le_bytes()));
    }
    part.extend(head);
    part.extend(body);
    crate::gpu::xbox::tests::container(&[(b"ACCC", &part)])
}

/// The outline budget (`pso::outline_budget`) on `outline_gate_code` as a console program, built
/// the way a run builds it: within the budget it is the packed program without calls; over it
/// the legacy wave64 form without calls when that one fits; over it in both forms, and with no
/// budget, the outlined program. Needs a DXC (`XCL_DXC_DIR`) for the outlined programs and a
/// scratch `XCL_DXIL_CACHE` (the build writes the caches of a run); without them it does nothing.
#[test]
fn outline_budget_keeps_a_program_that_fits_without_calls() {
    if std::env::var_os("XCL_DXC_DIR").is_none() || std::env::var_os("XCL_DXIL_CACHE").is_none() {
        println!("XCL_DXC_DIR or XCL_DXIL_CACHE not set");
        return;
    }
    let blob = fspx_container(&outline_gate_code());
    let build = |budget: usize, legacy: bool| {
        super::emit::OUTLINE.set(Some(true));
        super::pso::OUTLINE_BUDGET.set(Some(budget));
        let c = super::pso::compile_console_shader(&blob, super::pso::Stage::Compute, 0, 4, true, legacy);
        super::emit::OUTLINE.set(None);
        super::pso::OUTLINE_BUDGET.set(None);
        c.unwrap_or_else(|e| panic!("budget {budget}, legacy {legacy}: {e}")).dxil
    };
    // a module of the direct backend is one function; DXC's outlined module is not one the interpreter reads
    let calls = |dxil: &[u8]| xcl_dxil::interp::load(dxil).is_err();
    let (packed, legacy) = (build(usize::MAX, false), build(usize::MAX, true));
    assert!(!calls(&packed) && !calls(&legacy) && legacy.len() < packed.len(), "{} B packed, {} B legacy", packed.len(), legacy.len());
    let (old, both_over) = (build(0, false), build(1, false));
    assert!(calls(&old) && old.len() < packed.len(), "{} B outlined, {} B without calls", old.len(), packed.len());
    assert!(both_over == old, "over the budget in both forms: {} B, not the outlined program's {} B", both_over.len(), old.len());
    assert!(build(packed.len(), false) == packed, "a budget the packed program just fits");
    assert!(build(packed.len() - 1, false) == legacy && build(legacy.len(), false) == legacy, "over the budget packed, within it in the legacy form");
    assert!(build(legacy.len() - 1, false) == old, "a byte over the budget in the legacy form too");
    assert!(calls(&build(1, true)) && build(legacy.len(), true) == legacy, "the legacy form alone");
    println!("outline budget: {} B packed and {} B legacy without calls, {} B outlined", packed.len(), legacy.len(), old.len());
}

/// The forms of outlining the gate tests run, that of this run first.
fn outline_forms() -> Vec<OutlineForm> {
    use super::emit::OutlineAtomics::{Add, All, None};
    let current = (super::emit::outline_statics(), super::emit::outline_atomics(), super::emit::outline_flat());
    let mut forms = vec![current];
    forms.extend([(false, All, false), (true, All, false), (false, Add, false), (true, Add, false), (false, None, false), (true, None, false), (false, All, true), (true, All, true), (true, None, true)].into_iter().filter(|f| *f != current));
    forms
}

/// A form as its switches read: `XCL_RECOMP_OUTLINE_RESULTS`/`XCL_RECOMP_OUTLINE_ATOMICS`, and
/// "/flat" for `XCL_RECOMP_OUTLINE_NESTED=flat`.
fn outline_form_name((statics, atomics, flat): OutlineForm) -> String {
    format!("{}/{}{}", if statics { "static" } else { "out" }, format!("{atomics:?}").to_lowercase(), if flat { "/flat" } else { "" })
}

/// `outline_gate_code` plain and in the forms of outlining (`outline_forms`) leaves the memory worked
/// out by hand: in the interpreter on every machine, and the outlined forms on the GPU through DXC
/// (the plain program is 1.3 MB of DXIL packed, minutes and gigabytes in a driver, and no form is
/// compared with it). The GPU part runs this test again in a process a form, the form of this run
/// first, so a device one form loses is not the next one's: one `outline gate:` line a program
/// (FINISHED and its wrong words, or FAILED and the error), and the form of this run has to pass.
/// `XCL_OUTLINE_GATE_FORMS` (e.g. "static/all,out/none/flat") keeps the test to those forms, in this
/// process; `XCL_OUTLINE_GATE_DUMP` names a folder for the program's code and the HLSL of every form.
#[test]
fn gpu_outlined_calls_atomics_and_loops_compute_the_same() {
    // the accessors of this test are the earlier ones (`XCL_RECOMP_NO_COMPACT_ACCESS`); the next test has the compact ones
    outline_gate(false, "gpu_outlined_calls_atomics_and_loops_compute_the_same");
}

/// The same gate with the compact accessors (`emit::compact_accessors`). Their `xcl_loadx4` is one
/// loop of its own, so it calls no kept function in either nesting.
#[test]
fn gpu_outlined_calls_atomics_and_loops_compute_the_same_with_compact_accessors() {
    outline_gate(true, "gpu_outlined_calls_atomics_and_loops_compute_the_same_with_compact_accessors");
}

/// The gate of the two tests above; `test` is the name the GPU part runs again by.
fn outline_gate(compact: bool, test: &str) {
    super::emit::NO_COMPACT_ACCESS.set(!compact);
    use super::emit::OutlineAtomics::{self, All};
    let code = outline_gate_code();
    // a child of this test (below) leaves the interpreter to its parent
    let (name, child) = (outline_form_name, std::env::var_os("XCL_OUTLINE_GATE_CHILD").is_some());
    let only: Option<Vec<String>> = std::env::var("XCL_OUTLINE_GATE_FORMS").ok().map(|v| v.split(',').map(|f| f.trim().to_string()).collect());
    let forms: Vec<OutlineForm> = outline_forms().into_iter().filter(|f| only.as_ref().is_none_or(|o| o.contains(&name(*f)))).collect();
    assert!(!forms.is_empty(), "XCL_OUTLINE_GATE_FORMS names none of {:?}", outline_forms().into_iter().map(name).collect::<Vec<_>>());
    let mut programs = Vec::new();
    for packed in [true, false] {
        let threads = if packed { 96 } else { 64 };
        let cs = super::CsInterface { threads: [threads as u32, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 3 };
        let opt = if packed {
            Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: false, mem_windows: 4, ..Default::default() }
        } else {
            Options { mode: Mode::Compute(cs), wave64: false, split_wave64: true, mem_windows: 4, ..Default::default() }
        };
        let hlsl = |form: Option<OutlineForm>| {
            super::emit::OUTLINE.set(Some(form.is_some()));
            super::emit::OUTLINE_FORM.set(form.map(|f| (f.0, f.1)));
            super::emit::OUTLINE_FLAT.set(form.map(|f| f.2));
            let t = translate(&code, &opt);
            super::emit::OUTLINE.set(None);
            super::emit::OUTLINE_FORM.set(None);
            super::emit::OUTLINE_FLAT.set(None);
            let t = t.unwrap();
            assert!(t.complete(), "{:?}", t.unsupported_why);
            t.hlsl
        };
        let (io, mem, want) = outline_gate_state(threads);
        let plain = hlsl(None);
        assert!(!plain.contains("[noinline]"));
        assert_eq!(plain.contains("uint2 xcl_at("), compact);
        let differ = |got: &[u32]| got.iter().zip(&want).enumerate().filter(|(_, (a, b))| a != b).map(|(w, (a, b))| format!("word {w}: {a:#x}, not {b:#x}")).take(8).collect::<Vec<_>>();
        if !child {
            assert_eq!(differ(&interpreted(&plain, &io, &mem)), Vec::<String>::new(), "interpreted, packed {packed}: the plain program");
        }
        programs.push((packed, threads, "plain".to_string(), plain));
        for form in &forms {
            let text = hlsl(Some(*form));
            let kept = |f: &str| text.contains(&format!("[noinline] void {f}("));
            let r = if form.0 { "_r" } else { "_n" };
            // a kept function that calls a kept function, and the atomics of the form
            assert!(kept(&format!("xcl_loadx4{r}")) && kept(&format!("xcl_mLoad{r}")) && kept("xcl_mStore_n"), "{}\n{text}", name(*form));
            assert_eq!(kept(&format!("xcl_mInterlockedAdd{r}")), form.1 != OutlineAtomics::None, "{}", name(*form));
            assert_eq!(kept("xcl_mInterlockedOr_n") && kept("xcl_mInterlockedMin_n") && kept("xcl_mInterlockedXor_n"), form.1 == All, "{}", name(*form));
            assert_eq!(text.contains(" out uint xr0"), !form.0, "{}", name(*form));
            assert_eq!(text.contains("return uint4(xcl_mLoad_i(a.x), xcl_mLoad_i(a.y), xcl_mLoad_i(a.z), xcl_mLoad_i(a.w)); }"), form.2 && !compact, "{}", name(*form));
            if !child {
                assert_eq!(differ(&interpreted(&text, &io, &mem)), Vec::<String>::new(), "interpreted, packed {packed}: outlined, {}", name(*form));
            }
            programs.push((packed, threads, name(*form), text));
        }
    }
    if let Some(dir) = std::env::var_os("XCL_OUTLINE_GATE_DUMP") {
        std::fs::write(std::path::Path::new(&dir).join("gate-code.bin"), &code).unwrap();
        for (packed, _, form, hlsl) in &programs {
            std::fs::write(std::path::Path::new(&dir).join(format!("gate-{}-{}.hlsl", if *packed { "packed" } else { "wave" }, form.replace('/', "-"))), hlsl).unwrap();
        }
    }
    if only.is_none() {
        let mut lost = Vec::new();
        for form in &forms {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([format!("gpu::recomp::tests::{test}").as_str(), "--exact", "--nocapture", "--test-threads=1"])
                .envs([("XCL_OUTLINE_GATE_FORMS", name(*form)), ("XCL_OUTLINE_GATE_CHILD", "1".to_string())])
                .env_remove("XCL_OUTLINE_GATE_DUMP")
                .output()
                .unwrap();
            // the harness and a panic print in front of a line
            let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            let lines: Vec<&str> = text.lines().filter_map(|l| l.find("outline gate: ").map(|at| &l[at..])).collect();
            println!("{}", if lines.is_empty() { format!("outline gate: {}: no line, the process ended with {}", name(*form), out.status) } else { lines.join("\n") });
            if !out.status.success() {
                lost.push(name(*form));
            }
        }
        println!("outline gate: forms that failed: {lost:?}");
        assert!(!lost.contains(&name(forms[0])), "the form of this run, {}, failed (failed: {lost:?})", name(forms[0]));
        return;
    }
    let gpu = gpu_ready();
    for (packed, threads, form, hlsl) in programs.iter().filter(|p| p.2 != "plain") {
        if !gpu {
            println!("outline gate: {form}, packed {packed}: no GPU, not dispatched");
            continue;
        }
        let (io, mem, want) = outline_gate_state(*threads);
        let t0 = std::time::Instant::now();
        let dxil = super::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("outline gate: {form}, packed {packed}: NOT BUILT: {}", e.lines().next().unwrap_or_default()));
        let (built, t1) = (t0.elapsed().as_secs_f64(), std::time::Instant::now());
        let got = runner::dispatch(&dxil, [1, 1, 1], &io, &mem).unwrap_or_else(|e| panic!("outline gate: {form}, packed {packed}: {} B of DXIL in {built:.1} s, FAILED after {:.2} s: {}", dxil.len(), t1.elapsed().as_secs_f64(), e.replace('\n', " | ")));
        let wrong = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("outline gate: {form}, packed {packed}: {} B of DXIL in {built:.1} s, FINISHED in {:.2} s on {:?}, {wrong} words wrong", dxil.len(), t1.elapsed().as_secs_f64(), runner::adapter().unwrap_or_default());
        assert_eq!(wrong, 0, "{form}, packed {packed}: first {:?}", got.iter().zip(&want).enumerate().find(|(_, (a, b))| a != b));
    }
}

/// The load chain of `gpu_packed_compute_scalar_load_chain` over four memory windows, packed and per
/// wave: its two loads before the first flagged one read through the view that is not coherent
/// (`emit::plain_load_limit`) and the program leaves the memory it leaves with every load coherent.
#[test]
fn gpu_plain_loads_read_what_coherent_loads_read() {
    // through a group's pages, then with them folded (`emit::group_fold`)
    // and with the buffer load, alone on its base, in a group of its own or not (`emit::lone_loads`)
    for (fold, lone) in [(false, false), (true, false), (false, true), (true, true)] {
        super::emit::NO_GROUP_FOLD.set(!fold);
        super::emit::NO_LONE_LOADS.set(!lone);
        gpu_plain_loads(fold, lone);
    }
    super::emit::NO_LONE_LOADS.set(false);
}

fn gpu_plain_loads(fold: bool, lone: bool) {
    let code = prog(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 1), ("OFFSET", 16), ("SOFFSET", 125), ("GLC", 1)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 1), ("OFFSET", 20), ("SOFFSET", 125), ("DLC", 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 12), ("SSRC1", 13)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 14)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 15)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 16)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 18), ("SSRC1", 17)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128 + CHAIN_WINDOW as u64)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 18)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let forms: Vec<(bool, usize, String, String)> = [true, false]
        .into_iter()
        .map(|packed| {
            let threads = if packed { 96 } else { 64 };
            let cs = super::CsInterface { threads: [threads as u32, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 3 };
            let opt = if packed {
                Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: false, mem_windows: 4, ..Default::default() }
            } else {
                Options { mode: Mode::Compute(cs), wave64: false, split_wave64: true, mem_windows: 4, ..Default::default() }
            };
            let hlsl = |plain: bool| {
                super::emit::PLAIN_LOADS.set(Some(plain));
                let t = translate(&code, &opt);
                super::emit::PLAIN_LOADS.set(None);
                let t = t.unwrap();
                assert!(t.complete(), "{:?}", t.unsupported_why);
                t.hlsl
            };
            let (coherent, plain) = (hlsl(false), hlsl(true));
            assert!(coherent.contains("globallycoherent ") && !coherent.contains("_cached"), "{coherent}");
            let body = plain.split_once("void main(").unwrap().1;
            let four = if fold && !packed && !lone { "xcl_m_cachedLoad4X(" } else if lone && !packed { "xcl_loadx4_cached(" } else { "xcl_loadx4" };
            assert!(body.contains(" = xcl_m_cachedLoad4") && body.contains(four) && body.contains("_cached") && body.contains(" = xcl_mLoad"), "{body}");
            assert_eq!(body.contains("Fold("), fold && !packed);
            (packed, threads, coherent, plain)
        })
        .collect();
    for (packed, threads, coherent, plain) in &forms {
        let (io, mem) = chain_state(*threads);
        let (x, y) = (interpreted(coherent, &io, &mem), interpreted(plain, &io, &mem));
        assert_eq!(x[64..64 + threads], vec![351; *threads], "interpreted, packed {packed}: every load coherent");
        assert_eq!(x, y, "interpreted, packed {packed}: the first loads on the view that is not coherent");
    }
    if !gpu_ready() {
        return;
    }
    for (packed, threads, coherent, plain) in forms {
        let run = |hlsl: &str| {
            let dxil = super::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{hlsl}"));
            let (io, mem) = chain_state(threads);
            runner::dispatch(&dxil, [1, 1, 1], &io, &mem).unwrap()
        };
        let (a, b) = (run(&coherent), run(&plain));
        assert_eq!(a[64..64 + threads], vec![351; threads], "packed {packed}: every load coherent");
        assert_eq!(a, b, "packed {packed}: the first loads on the view that is not coherent");
    }
}

#[test]
fn gpu_packed_compute_lds_orders_both_slots_at_the_guest_barrier() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 161), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 3), ("ADDR", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let barrier = encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap();
    let at = code.windows(barrier.len()).position(|w| w == barrier).unwrap();
    let mut wave_only = code.clone();
    wave_only.drain(at..at + barrier.len());
    for (code, syncs) in [(&code, 1), (&wave_only, 0)] {
        let hlsl = packed_compute_source(code, [64, 1, 1]);
        assert_eq!(hlsl.matches("GroupMemoryBarrierWithGroupSync();").count(), syncs);
        let out = run_packed_compute(&hlsl, 64);
        assert_eq!(out[16..20], [0; 4]);
        assert_eq!(out[64..128], std::array::from_fn::<_, 64, _>(|i| i as u32 ^ 33));
    }
}

#[test]
fn gpu_packed_loop_profile_reports_natural_and_shared_budget_exits() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_GT_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 134)], None),
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 128)], None),
    ]);
    let mode = Mode::Compute(super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 });
    let hlsl = base_loop_guard(|| translate(&code, &Options { mode, wave64: true, split_wave64: true, legacy_compute_wave64: false, loop_profile: true, ..Default::default() })).unwrap().hlsl;
    assert!(hlsl.contains("xcl_lp_done(0u, 5u, xcl_lp_n, xcl_lp_exit, 7u);"));
    if !gpu_ready() {
        return;
    }
    let out = run_packed_compute(&hlsl, 64);
    assert_eq!(out[20..30], [5, 4, 7, 448, 0, 64, 0, 64, 0, 0]);
    let short = hlsl.replace(&format!("#define XCL_LOOP_GUARD {}u", super::loop_guard()), "#define XCL_LOOP_GUARD 2u");
    let out = run_packed_compute(&short, 64);
    assert_eq!(out[16], 64);
    assert_eq!(out[20..30], [5, 2, 2, 128, 0, 64, 0, 0, 64, 0]);
}

#[test]
fn gpu_packed_compute_nested_loops_share_the_total_budget() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 129)], None),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 131)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFA)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 10)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let hlsl = base_loop_guard(|| packed_compute_source(&code, [96, 1, 1])).replace(&format!("#define XCL_LOOP_GUARD {}u", super::loop_guard()), "#define XCL_LOOP_GUARD 5u");
    assert_eq!(hlsl.matches("static uint xcl_budget =").count(), 1);
    assert_eq!(hlsl.matches("xcl_budget -= min(xcl_budget, 1u);").count(), 2);
    let out = run_packed_compute(&hlsl, 96);
    assert_eq!(out[16..20], [96, 0, 0, 0]);
    assert_eq!(out[64..160], [3; 96]);
}

#[test]
fn gpu_loop_budget_exits_with_the_wave_when_a_nonfirst_lane_runs_out() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 11)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 130), ("SRC2", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 10)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    for (wave64, legacy_compute_wave64, lanes) in [(false, false, 32), (true, false, 64), (true, true, 64)] {
        let cs = super::CsInterface { threads: [lanes, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 };
        let t = base_loop_guard(|| translate(&code, &Options { mode: Mode::Compute(cs), wave64, split_wave64: true, legacy_compute_wave64, ..Default::default() })).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let entry = t.hlsl.rfind("void main(").unwrap();
        assert_eq!(t.hlsl[entry..].matches("[loop] for (;;) {").count(), 1);
        let hlsl = format!("{}{}", &t.hlsl[..entry], t.hlsl[entry..].replace("[loop] for (;;) {", "xcl_budget = select((lane_id & 31u) == 17u, 1u, 4u);\n    [loop] for (;;) {"));
        let out = run_packed_compute(&hlsl, lanes as usize);
        assert_eq!(out[16..20], [lanes, 0, 0, 0]);
        assert!(out[64..].iter().all(|&v| v == 1), "wave64={wave64}, legacy={legacy_compute_wave64}");
    }
}

#[test]
fn hlsl_is_generated_for_straight_line_and_branches() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 133)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert_eq!(t.blocks, 3, "{}", t.hlsl);
    // forward branches only: a plain `if`, no loop and no block dispatch on pc
    assert!(t.hlsl.contains("if (!(") && !t.hlsl.contains("pc ==") && !t.hlsl.contains("for (;;)") && t.hlsl.contains("[WaveSize(64)]"), "{}", t.hlsl);
    // the guarded block chain (fallback form) for the same program
    // (an option, not the environment variable: other tests translate at the same time)
    let g = translate(&code, &Options { structured: false, ..Options::default() }).unwrap();
    assert!(g.hlsl.contains("if (pc == 0u)") && !g.hlsl.contains("while"), "{}", g.hlsl);
    // a backward branch: only its range becomes a loop
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None), // back to s_sub
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    // only the range of the backward branch loops; its condition is the loop's exit test
    assert!(t.hlsl.contains("[loop] for (;;)") && t.hlsl.contains("break;") && !t.hlsl.contains("pc =="), "{}", t.hlsl);
    let t = translate(&prog(&[("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)], None)]), &Options::default()).unwrap();
    assert!(!t.hlsl.contains("pc =="), "{}", t.hlsl);
    // unknown to the recompiler: reported, HLSL still produced
    let t = translate(&prog(&[("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 0), ("VSRC", 0), ("ATTR", 0)], None)]), &Options::default()).unwrap();
    assert_eq!(t.unsupported.get("v_interp_p1_f32"), Some(&1));
    let _ = Mode::Kernel;
}

#[test]
fn scalar_bfe_64_handles_unsigned_signed_and_zero_width() {
    let code = prog(&[
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 4), ("SSRC1", 9)], None),
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 4), ("SSRC1", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("& (0xFFFFFFFFFFFFFFFFull >> (64u -") && t.hlsl.contains("(int64_t)(t") && t.hlsl.contains("== 0u) ? 0ull"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_quadmask_and_bitreplicate() {
    let code = prog(&[
        ("S_QUADMASK_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_QUADMASK_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)], None),
        ("S_BITREPLICATE_B64_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 6)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("quadmask(s4) | (quadmask(s5) << 8u)"));
    assert!(t.hlsl.contains("bitrep16(s6);") && t.hlsl.contains("bitrep16(s6 >> 16u);"));
}

#[test]
fn scalar_bitcmp_64_selects_the_indexed_half() {
    let code = prog(&[
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 9)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("select((s8 & 32u) == 0u, s4, s5)") && t.hlsl.contains("select((s9 & 32u) == 0u, s4, s5)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("scc = (uint)(") && t.hlsl.contains("== 0u);"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_64_bit_count_scan_and_set_cluster() {
    let code = prog(&[
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 10)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 11)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("64u - countbits(s4) - countbits(s5)") && t.hlsl.contains("firstbitlow(~s4)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(s5)") && t.hlsl.contains("s7 | select((s10 & 32u) != 0u") && t.hlsl.contains("s6 & ~select((s11 & 32u) == 0u"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_signed_leading_bit_scans() {
    let code = prog(&[
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 6)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("ffbh_i(s4)"), "{}", t.hlsl);
    // i64: normalize on the 64-bit sign, then scan the high half first and the low half second
    assert!(t.hlsl.contains("select((int)s7 < 0, ~s6, s6)") && t.hlsl.contains("select((int)s7 < 0, ~s7, s7)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(") && t.hlsl.contains("63u - firstbithigh(") && t.hlsl.contains("0xFFFFFFFFu"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_relative_move_cluster() {
    let code = prog(&[
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("switch (m0)") && t.hlsl.contains("switch ((m0 & 0x3FFu))") && t.hlsl.contains("switch (((m0 >> 16) & 0x3FFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_scalar_relative_moves() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 130)], None),
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0x0003_0001)),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[10..14].copy_from_slice(&[100, 101, 102, 103]);
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!((r.sgpr[0], r.sgpr[2], r.sgpr[3]), (102, 102, 103), "relative scalar reads");
    assert_eq!((r.sgpr[22], r.sgpr[26], r.sgpr[27]), (100, 100, 101), "relative scalar writes");
    assert_eq!(r.sgpr[33], 101, "split source/destination offsets");
}

/// v_swaprel_b32: both sides are read-write (VDST + M0[25:16] and SRC0 + M0[9:0]); emitted
/// as the same bounded switches as v_movrelsd_2_b32, both old values captured before writes.
#[test]
fn v_swaprel_b32_uses_the_m0_relative_switch() {
    let code = prog(&[("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None)]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // both M0-relative registers read into temporaries, each with its own index
    let (dst, src) = ("switch (((m0 >> 16) & 0x3FFu)) { case 0u:", "switch ((m0 & 0x3FFu)) { case 0u:");
    assert!(t.hlsl.contains(&format!("uint t1 = 0u; {dst} t1 = v5; break; case 1u: t1 = v6; break;")), "{}", t.hlsl);
    assert!(t.hlsl.contains(&format!("uint t2 = 0u; {src} t2 = v10; break; default: break; }}")), "{}", t.hlsl);
    // EXEC-gated exchange of the two captured values
    assert!(t.hlsl.contains(&format!("if (xcl_on) {{ {dst} v5 = t2; break;")), "{}", t.hlsl);
    assert!(t.hlsl.contains(&format!("{src} v10 = t1; break; default: break; }} }}")), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_v_swaprel_b32() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None), // v5 <-> v13
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None), // exec = s[30:31]: even lanes
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 11)], None), // v6 <-> v14 (active only)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 32)], None), // exec = s[32:33]: all lanes
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 129)], None), // m0 = 1
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 6)], None), // v7 <-> v7: no-op
        // the ISA's two indices: VDST + M0[25:16], SRC0 + M0[9:0]
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some((2 << 16) | 3)),
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None), // v7 <-> v13, v5 untouched
        // M0 bits outside [25:16] and [9:0] ignored: indices 1 and 2
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xFC01_FC02)),
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 20)], None), // v17 <-> v22
        // relative targets lie within the program's VGPRs (as in gpu_v_movrel): v15 / v22 used directly
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 15)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", V + 22)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[30] = 0x5555_5555;
    st.sgpr[31] = 0x5555_5555;
    st.sgpr[32] = 0xFFFF_FFFF;
    st.sgpr[33] = 0xFFFF_FFFF;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = 1000 + l as u32;
        r[13] = 2000 + l as u32;
        r[6] = 3000 + l as u32;
        r[14] = 4000 + l as u32;
        r[7] = 700 + l as u32;
        r[17] = 5000 + l as u32;
        r[22] = 6000 + l as u32;
        r[16] = 7000 + l as u32;
        r[20] = 8000 + l as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let l32 = l as u32;
        // v5 <-> v13 (m0 = 3), then v7 (700, after the no-op) <-> v13 (1000) with m0 = (2 << 16) | 3
        assert_eq!(v[5], 2000 + l32, "lane {l} ordinary swap; v5 untouched by the second");
        let expected = if l % 2 == 0 { (4000 + l32, 3000 + l32) } else { (3000 + l32, 4000 + l32) };
        assert_eq!((v[6], v[14]), expected, "lane {l} EXEC-gated swap");
        assert_eq!((v[7], v[13]), (1000 + l32, 700 + l32), "lane {l} overlapping no-op, then the destination index");
        assert_eq!((v[16], v[17], v[20], v[22]), (7000 + l32, 6000 + l32, 8000 + l32, 5000 + l32), "lane {l} M0 bits outside the two fields");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0xFFFF_FFFF, 0xFFFF_FFFF), "exec restored");
}

#[test]
fn atomic_inc_dec_x2_encoding_families() {
    let code = prog(&[
        ("BUFFER_ATOMIC_INC_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("BUFFER_ATOMIC_DEC_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("FLAT_ATOMIC_INC_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14)], None),
        ("FLAT_ATOMIC_DEC_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_INC_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2)], None),
        ("GLOBAL_ATOMIC_DEC_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("xcl_mInterlockedCompareExchange64(t").count(), 6, "{}", t.hlsl);
    assert!(t.hlsl.contains(") ? (uint64_t)0 :") && t.hlsl.contains("== (uint64_t)0 ||"), "{}", t.hlsl);
    // memory above 4 GiB: every access of the CAS loop (its first read too) goes through the
    // window accessors (the flat forms' on the lane's window)
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(!w.hlsl.contains("mem.Load<") && w.hlsl.matches("= xcl_mLoad2(").count() + w.hlsl.matches("= xcl_vLoad2(").count() >= 6, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{}", w.hlsl));
    }
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn atomic_fcmpswap_x2_uses_float_equality_and_a_64_bit_cas() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FCMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("xcl_mInterlockedCompareExchange64(t").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("_cmp) == 0 ?").count(), 3, "{}", t.hlsl);
    // [DATA:DATA+1] is the swap value, [DATA+2:DATA+3] the compare value (128-bit operand)
    assert!(t.hlsl.contains("_v = (uint64_t)v4 | ((uint64_t)v5 << 32); uint64_t") && t.hlsl.contains("_cmp = (uint64_t)v6 | ((uint64_t)v7 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v18 | ((uint64_t)v19 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v26 | ((uint64_t)v27 << 32);"), "{}", t.hlsl);
    // 8-byte alignment of all three cells; the buffer form's bounds check reads records in dwords
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("_off + 8u <= s10"), "{}", t.hlsl);
    // GLC returns the old u64 as a low / high dword pair – in place in VDATA for the buffer
    // form, into VDST for flat / global
    assert_eq!(t.hlsl.matches("_old >> 32").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("v4 = (uint)t") && t.hlsl.contains("v5 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (uint)t") && t.hlsl.contains("v13 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v20 = (uint)t") && t.hlsl.contains("v21 = (uint)(t"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn atomic_fmin_fmax_x2_use_ordered_comparisons_and_a_64_bit_cas() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FMIN_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_FMAX_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("FLAT_ATOMIC_FMIN_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FMAX_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_FMIN_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("xcl_mInterlockedCompareExchange64(t").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("_c) == -1 ?").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("_c) == 1 ?").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 6, "{}", t.hlsl);
    // memory above 4 GiB: the same accessors, on the window `xcl_w` selects (the flat forms' on
    // the lane's, `xcl_wv`)
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert_eq!((w.hlsl.matches("xcl_mInterlockedCompareExchange64(t").count(), w.hlsl.matches("xcl_vInterlockedCompareExchange64(t").count()), (4, 2), "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", w.hlsl));
    }
}

#[test]
fn unary_f64_math_vop1_and_vop3() {
    let code = prog(&[
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 10)], None),
        ("V_CEIL_F64", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FRACT_F64", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_SQRT_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("ceil_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    assert_eq!(t.hlsl.matches("fract_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    // f64 roots through sqrt_d (HLSL's sqrt is f32): V_SQRT / V_RSQ, twice each
    assert_eq!(t.hlsl.matches("sqrt_d(").count(), 1 + 4, "{}", t.hlsl); // helper plus four calls
    assert_eq!(t.hlsl.matches("rcp_d(sqrt_d(").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn f64_class_compare_vopc_and_vop3() {
    let code = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 12)], None),
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 10), ("SRC1", V + 12), ("ABS", 1)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 13)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOP3", &[("VDST", 126), ("SRC0", V + 10), ("SRC1", V + 13), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("fclass64(").count(), 5, "{}", t.hlsl); // helper plus four calls
    // SRC0 is the 64-bit f64 operand (VGPR/SGPR pair), SRC1 stays a plain 32-bit mask;
    // abs/neg act on the double operand only
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v13) >> fclass64(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    // mask destinations go through the shared compare/write-mask path: vcc, an SGPR pair, exec
    assert!(t.hlsl.contains("vcc_lo = t") && t.hlsl.contains("vcc_hi = t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("s20 = t") && t.hlsl.contains("s21 = t"), "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_lo = t").count(), 2, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_hi = t").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_f64_class_compare() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 52)], None),                        // vcc
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 50), ("SRC1", V + 52)], None),      // s[20:21]
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 50), ("SRC1", 8), ("ABS", 1)], None), // |x|, mask in s8
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 53)], None),                       // exec
    ]);
    let classes = [
        0x7FF0_0000_0000_0001u64,
        0x7FF8_0000_0000_0000,
        f64::NEG_INFINITY.to_bits(),
        (-1.0f64).to_bits(),
        0x8000_0000_0000_0001,
        (-0.0f64).to_bits(),
        0.0f64.to_bits(),
        1,
        1.0f64.to_bits(),
        f64::INFINITY.to_bits(),
    ];
    let mut st = KernelState::default();
    st.sgpr[8] = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 9; // positive non-NaN classes
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let class = l % classes.len();
        let bits = classes[class];
        (r[50], r[51]) = (bits as u32, (bits >> 32) as u32);
        r[52] = 1 << class;
        r[53] = 1 << 6;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, u32::MAX), "each lane matches its f64 class (vcc)");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (u32::MAX, u32::MAX), "each lane matches its f64 class (sdst)");
    // |x| ∈ {+0, +denormal, +normal, +inf} exactly when x is not a NaN: negative lanes fold
    let pos: u64 = (0..64).filter(|l| l % 10 >= 2).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, pos, "abs: every non-NaN becomes positive");
    let zero: u64 = (0..64).filter(|l| l % 10 == 6).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[126] as u64 | (r.sgpr[127] as u64) << 32, zero, "cmpx keeps positive-zero lanes");
}

/// Normalized f16 conversions – v_cvt_norm_u16_f16 / _i16_f16 (VOP1 single half-word
/// result, VOP3 with op_sel picking source/destination halves) and v_cvt_pknorm_u16_f16 /
/// _i16_f16 (VOP3, two f16 halves → a packed dword). Emitted text is deterministic.
#[test]
fn cvt_norm_f16_instructions() {
    let code = prog(&[
        ("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 9)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 9), ("SRC1", V + 10), ("SRC2", V + 11), ("OP_SEL", 0b1001)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 9), ("SRC1", V + 10)], None),
        ("V_CVT_PKNORM_I16_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 9), ("SRC1", V + 10), ("OP_SEL", 0b0011)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 5), ("SRC0", 242), ("SRC1", 243)], None), // f16 +1.0, −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    // single f16 source (low half) → unsigned / signed normalized short, 16-bit result
    assert!(t.hlsl.contains("cvt_norm_u16_f16(((v9) & 0xFFFFu))"), "{}", t.hlsl);
    // op_sel[0] reads the source's high half, op_sel[3] writes the destination's high
    // half preserving the low one; abs/neg apply directly on the binary16 sign bit
    assert!(t.hlsl.contains("cvt_norm_i16_f16(((v9) >> 16))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v2 & 0xFFFFu) | (t") && t.hlsl.contains("<< 16), v2)"), "{}", t.hlsl);
    // packed: two whole dwords written; each source half picked by op_sel
    assert!(t.hlsl.contains("(cvt_norm_u16_f16((v9 & 0xFFFFu)) | (cvt_norm_u16_f16((v10 & 0xFFFFu)) << 16))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(cvt_norm_i16_f16((v9 >> 16)) | (cvt_norm_i16_f16((v10 >> 16)) << 16))"), "{}", t.hlsl);
    // inline float constants decode as f16 bit patterns (1.0 = 0x3C00, −1.0 = 0xBC00)
    assert!(t.hlsl.contains("cvt_norm_u16_f16((0x00003C00u & 0xFFFFu))") && t.hlsl.contains("cvt_norm_u16_f16((0x0000BC00u & 0xFFFFu))"), "{}", t.hlsl);
}

#[test]
fn v_cvt_off_f32_i4_sign_extends_the_low_nibble() {
    let code = prog(&[
        ("V_CVT_OFF_F32_I4", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_CVT_OFF_F32_I4", "ENC_VOP3", &[("VDST", 2), ("SRC0", V), ("OMOD", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("(float)(((int)(v0 << 28)) >> 28) * 0.0625f"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_v_cvt_off_f32_i4() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CVT_OFF_F32_I4", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_CVT_OFF_F32_I4", "ENC_VOP3", &[("VDST", 2), ("SRC0", V), ("OMOD", 1)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 0xABCD_EF70 | (l as u32 % 16);
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let n = (l % 16) as i32;
        let want = (if n >= 8 { n - 16 } else { n }) as f32 / 16.0;
        assert_eq!(f32::from_bits(r.vgpr[l][1]), want, "lane {l} nibble {n}");
        assert_eq!(f32::from_bits(r.vgpr[l][2]), want * 2.0, "lane {l} nibble {n} omod *2");
    }
}

/// GPU verification of exact RNE rounding and edge cases for `V_CVT_NORM_*_F16` and
/// `V_CVT_PKNORM_*_F16`: catches binary32 double-rounding (`0x3801`/`0xB801`), DXIL
/// `clamp(NaN, -1, 1)` returning ±32767 instead of 0, ±Inf, ±0, subnormals, and VOP3 abs/neg.
#[test]
fn gpu_cvt_norm_f16_exact_rounding_and_edge_cases() {
    if !gpu_ready() {
        return;
    }
    // 16 test vectors (packed into lane % 16): (f16_bits, expected_unorm16, expected_snorm16)
    const CASES: &[(u16, u16, u16)] = &[
        (0x3801, 32799, 16399),         // 1025/2048: fp32 double-rounds to 32800 / 16400
        (0xB801, 0, (-16399i16) as u16), // -1025/2048: fp32 double-rounds to -16400
        (0x3800, 32768, 16384),         // +0.5: exact half tie-to-even
        (0xB800, 0, (-16384i16) as u16), // -0.5: exact half tie-to-even
        (0x7E00, 0, 0),                 // +qNaN -> 0 (not ±32767)
        (0xFC01, 0, 0),                 // -sNaN -> 0
        (0x7C00, 65535, 32767),         // +Inf
        (0xFC00, 0, (-32767i16) as u16), // -Inf
        (0x0000, 0, 0),                 // +0.0
        (0x8000, 0, 0),                 // -0.0
        (0x0080, 0, 0),                 // subnormal 128 * 2^-24: *65535 = 0.499992 -> 0
        (0x0081, 1, 0),                 // subnormal 129 * 2^-24: *65535 = 0.503899 -> 1
        (0x0101, 1, 1),                 // subnormal 257 * 2^-24: *32767 = 0.501938 -> 1
        (0x3BFF, 65503, 32751),         // max f16 < 1.0 (2047/2048)
        (0x3C00, 65535, 32767),         // +1.0
        (0xBC00, 0, (-32767i16) as u16), // -1.0
    ];
    let code = prog(&[
        // v1 = V_CVT_NORM_U16_F16(v0.lo), v2 = V_CVT_NORM_I16_F16(v0.lo)
        ("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        // Packed: v3 = pknorm_u16(v0.lo, v0.hi), v4 = pknorm_i16(v0.lo, v0.hi)
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V), ("OP_SEL", 0b0010)], None),
        ("V_CVT_PKNORM_I16_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V), ("OP_SEL", 0b0010)], None),
        // VOP3 modifiers:
        // - abs(-1025/2048) -> +1025/2048 -> u16 32799 / i16 16399
        // - neg(+1025/2048) -> -1025/2048 -> i16 -16399 (0xBFF1)
        // - neg(abs(-1025/2048)) -> -1025/2048 -> i16 -16399 (verifies abs before neg)
        // - neg(qNaN) -> 0
        ("V_CVT_NORM_U16_F16", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 5), ("ABS", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 5), ("ABS", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 6), ("NEG", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 12), ("NEG", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 5), ("ABS", 1), ("NEG", 1)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lo = CASES[l % CASES.len()].0 as u32;
        let hi = CASES[(l + 1) % CASES.len()].0 as u32;
        r[0] = lo | (hi << 16);
        r[5] = 0xB801;  // -1025/2048
        r[6] = 0x7E00;  // +qNaN
        r[12] = 0x3801; // +1025/2048
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let (h_lo, exp_u_lo, exp_i_lo) = CASES[l % CASES.len()];
        let (h_hi, exp_u_hi, exp_i_hi) = CASES[(l + 1) % CASES.len()];
        assert_eq!(r.vgpr[l][1] & 0xFFFF, exp_u_lo as u32, "lane {l} u16(0x{h_lo:04X})");
        assert_eq!(r.vgpr[l][2] & 0xFFFF, exp_i_lo as u32, "lane {l} i16(0x{h_lo:04X})");
        assert_eq!(
            r.vgpr[l][3],
            (exp_u_lo as u32) | ((exp_u_hi as u32) << 16),
            "lane {l} pknorm_u16(0x{h_lo:04X}, 0x{h_hi:04X})"
        );
        assert_eq!(
            r.vgpr[l][4],
            (exp_i_lo as u32) | ((exp_i_hi as u32) << 16),
            "lane {l} pknorm_i16(0x{h_lo:04X}, 0x{h_hi:04X})"
        );
        assert_eq!(r.vgpr[l][7] & 0xFFFF, 32799, "lane {l} abs(0xB801) u16");
        assert_eq!(r.vgpr[l][8] & 0xFFFF, 16399, "lane {l} abs(0xB801) i16");
        assert_eq!(r.vgpr[l][9] & 0xFFFF, 0, "lane {l} neg(qNaN) i16");
        assert_eq!(r.vgpr[l][10] & 0xFFFF, (-16399i16) as u16 as u32, "lane {l} neg(0x3801) i16");
        assert_eq!(r.vgpr[l][11] & 0xFFFF, (-16399i16) as u16 as u32, "lane {l} -abs(0xB801) i16");
    }
}

/// The f64 frexp pair: v_frexp_mant_f64 (m in [0.5,1) keeping the sign, input back for
/// 0/inf/NaN) → an f64 pair, v_frexp_exp_i32_f64 (input = m·2^exp, 0 for 0/inf/NaN) → one
/// dword. VOP1 and VOP3 (source abs/neg come from the f64 modifiers). Emitted text is
/// deterministic; helpers normalise denormals bit-exactly.
#[test]
fn frexp_f64_instructions() {
    let code = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", 243)], None), // inline constant −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // the source pair becomes a double; the mantissa stays f64, the exponent is one dword
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(uint)frexp_exp_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_exp_d(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    // the f64 pair of inline −1.0 = 0xBFF0000000000000
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(0x00000000u, 0xBFF00000u))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn rndne_f64_instructions() {
    let code = prog(&[
        ("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_RNDNE_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 10), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_rndne_f64() {
    if !gpu_ready() { return; }
    let p = prog(&[("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None)]);
    const INPUTS: [u64; 16] = [
        0.0f64.to_bits(), (-0.0f64).to_bits(), 0.5f64.to_bits(), (-0.5f64).to_bits(),
        1.5f64.to_bits(), 2.5f64.to_bits(), (-1.5f64).to_bits(), (-2.5f64).to_bits(),
        0.49999999999999994f64.to_bits(), 0.5000000000000001f64.to_bits(),
        4503599627370495.5f64.to_bits(), 4503599627370496.0f64.to_bits(),
        f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits(), 0x7FF8_0000_0000_1234, 1,
    ];
    let mut st = KernelState::default();
    for (lane, regs) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[lane % INPUTS.len()];
        (regs[0], regs[1]) = (x as u32, (x >> 32) as u32);
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for (lane, regs) in r.vgpr.iter().enumerate() {
        let x = f64::from_bits(INPUTS[lane % INPUTS.len()]);
        let got = regs[2] as u64 | ((regs[3] as u64) << 32);
        let want = if x.is_nan() { x.to_bits() } else { x.round_ties_even().to_bits() };
        assert_eq!(got, want, "lane {lane}: {x:?}");
    }
}

#[test]
fn gpu_frexp_f64_exp_and_mant() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 0)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 0)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 0), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 0), ("ABS", 1)], None),
    ]);
    const INPUTS: [u64; 16] = [
        12.5f64.to_bits(),            // 0.78125 · 2^4
        (-4.0f64).to_bits(),          // −0.5 · 2^3
        0.5f64.to_bits(),             // 0.5 · 2^0
        1.0f64.to_bits(),             // 0.5 · 2^1
        0.0f64.to_bits(),             // special → 0/0
        (-0.0f64).to_bits(),          // special → input/0
        f64::INFINITY.to_bits(),      // special → input/0
        f64::NEG_INFINITY.to_bits(),  // special → input/0
        0x7FF8_0000_0000_0001,        // NaN → input payload, 0
        0x0000_0000_0000_0001,        // smallest denormal → 0.5 · 2^−1073
        0x0000_0000_000C_0000,        // denormal 3·2^−1056 → 0.75 · 2^−1054
        0x000F_FFFF_FFFF_FFFF,        // largest denormal → (1−2^−52) · 2^−1022
        0x0010_0000_0000_0000,        // smallest normal → 0.5 · 2^−1021
        0x7FEF_FFFF_FFFF_FFFF,        // max normal → (1−2^−53) · 2^1024
        0x4059_0000_0000_0000,        // 100.0 → 0.78125 · 2^7
        0x800F_FFFF_FFFF_FFFF,        // negative denormal → −(1−2^−52) · 2^−1022
    ];
    // CPU mirror of the ISA semantics: e = topmost set bit of the 52-bit fraction − 1073
    // for denormals, biased exponent − 1022 for normals; mant = normalised [0.5,1) value.
    fn frexp_ref(bits: u64) -> (u64, i32) {
        let (h, l) = ((bits >> 32) as u32, bits as u32);
        let (e, m, s) = ((h >> 20) & 0x7FF, h & 0xF_FFFF, h & 0x8000_0000);
        if e == 0x7FF || (e == 0 && (m | l) == 0) {
            return (bits, 0);
        }
        if e != 0 {
            return ((((s | m | 0x3FE0_0000) as u64) << 32) | l as u64, e as i32 - 1022);
        }
        let top = if m != 0 { 63 - m.leading_zeros() } else { 31 - l.leading_zeros() };
        let f = ((m as u64) << 32) | l as u64;
        let ff = (f << (52 - top)) & 0x000F_FFFF_FFFF_FFFF;
        ((((s | 0x3FE0_0000 | (ff >> 32) as u32) as u64) << 32) | (ff & 0xFFFF_FFFF), top as i32 - 1073)
    }
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[l % INPUTS.len()];
        r[0] = x as u32;
        r[1] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (mant, exp) = frexp_ref(INPUTS[l % INPUTS.len()]);
        assert_eq!(v[2] as u64 | (v[3] as u64) << 32, mant, "lane {l} v_frexp_mant_f64");
        assert_eq!(v[4] as i32, exp, "lane {l} v_frexp_exp_i32_f64");
        // negated source: mantissa flips its sign, exponent is unchanged
        assert_eq!(v[6] as u64 | (v[7] as u64) << 32, mant ^ 0x8000_0000_0000_0000, "lane {l} v_frexp_mant_f64 neg");
        assert_eq!(v[8] as i32, exp, "lane {l} v_frexp_exp_i32_f64 abs");
    }
}

#[test]
fn gpu_frexp_mant_f32_and_f16_preserve_sign() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_FREXP_MANT_F32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_FREXP_MANT_F16", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)], None),
    ]);
    let inputs = [
        ((-10.125f32).to_bits(), (-0.6328125f32).to_bits(), 0xC910, 0xB910),
        (10.125f32.to_bits(), 0.6328125f32.to_bits(), 0x4910, 0x3910),
        ((-0.0f32).to_bits(), (-0.0f32).to_bits(), 0x8000, 0x8000),
        (f32::NEG_INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 0xFC00, 0xFC00),
    ];
    let mut st = KernelState::default();
    for (lane, regs) in st.vgpr.iter_mut().enumerate() {
        let (f, _, h, _) = inputs[lane % inputs.len()];
        regs[0] = f;
        regs[1] = h;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for (lane, regs) in r.vgpr.iter().enumerate() {
        let (_, f, _, h) = inputs[lane % inputs.len()];
        assert_eq!(regs[2], f, "f32 lane {lane}");
        assert_eq!(regs[3] & 0xFFFF, h, "f16 lane {lane}");
    }
}

/// Edge inputs of the f64 operations: |x| ≥ 2^63, values outside the f32 range, denormals,
/// signed zeros, ±inf and NaN.
const F64_EDGES: [u64; 22] = [
    0x0000_0000_0000_0000,       // +0
    0x8000_0000_0000_0000,       // −0
    0.5f64.to_bits(),
    (-0.5f64).to_bits(),
    (-1.5f64).to_bits(),
    2.5f64.to_bits(),
    4.0f64.to_bits(),
    123456789.75f64.to_bits(),
    (-4503599627370495.5f64).to_bits(),
    9223372036854775808.0f64.to_bits(),     // 2^63
    (-18446744073709551616.0f64).to_bits(), // −2^64
    1e300f64.to_bits(),
    (-1e300f64).to_bits(),
    1e-300f64.to_bits(),
    (-1e-300f64).to_bits(),
    0x0000_0000_0000_0001,                  // smallest denormal
    0x000F_FFFF_FFFF_FFFF,                  // largest denormal
    f64::MAX.to_bits(),
    f64::INFINITY.to_bits(),
    f64::NEG_INFINITY.to_bits(),
    0x7FF8_0000_0000_1234,                  // NaN
    4294967295.0f64.to_bits(),
];

fn f64_lanes(st: &mut KernelState, inputs: &[u64]) {
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let x = inputs[l % inputs.len()];
        (r[0], r[1]) = (x as u32, (x >> 32) as u32);
    }
}

fn pair(v: &[u32], i: usize) -> f64 {
    f64::from_bits(v[i] as u64 | (v[i + 1] as u64) << 32)
}

/// v_floor, v_ceil, v_trunc and v_fract_f64 and the f64 to i32 and u32 conversions against
/// the CPU's IEEE results, bit for bit, including the sign of zero, |x| ≥ 2^63, ±inf and NaN.
#[test]
fn gpu_f64_rounding_and_conversions() {
    for wave64 in gpu_waves() {
        f64_rounding_and_conversions(wave64);
    }
}

fn f64_rounding_and_conversions(wave64: bool) {
    let p = prog(&[
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_TRUNC_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V)], None),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        // the inline constant 0 compiles as a constant, where 0.0 − floor(0.0) gave −0
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 12), ("SRC0", 128)], None),
    ]);
    let mut st = KernelState::default();
    f64_lanes(&mut st, &F64_EDGES);
    let (r, lanes) = run_wave(&p, &st, wave64);
    let same = |got: f64, want: f64| got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let x = f64::from_bits(F64_EDGES[l % F64_EDGES.len()]);
        // V_FRACT_F64 is x − floor(x) kept below 1.0, +0 for ±0, and NaN for ±inf and NaN
        let fract = if x.is_finite() { (x - x.floor()).min(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)) } else { f64::NAN };
        for (name, got, want) in [
            ("floor", pair(v, 2), x.floor()),
            ("ceil", pair(v, 4), x.ceil()),
            ("trunc", pair(v, 6), x.trunc()),
            ("fract", pair(v, 8), fract),
        ] {
            if !same(got, want) {
                bad.push(format!("v_{name}_f64({x:e}) = {got:e}, want {want:e}"));
            }
        }
        if pair(v, 12).to_bits() != 0 {
            bad.push(format!("v_fract_f64(inline 0) = {:e}, want +0", pair(v, 12)));
        }
        // saturating, and NaN gives 0, which is what Rust's `as` does
        if v[10] as i32 != x as i32 {
            bad.push(format!("v_cvt_i32_f64({x:e}) = {}, want {}", v[10] as i32, x as i32));
        }
        if v[11] != x as u32 {
            bad.push(format!("v_cvt_u32_f64({x:e}) = {}, want {}", v[11], x as u32));
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// v_sqrt_f64 and v_rsq_f64 over the whole f64 range. Denormals and values outside the f32
/// range get real roots within an ulp of the CPU's, a negative x gives NaN and −0 gives −0.
#[test]
fn gpu_f64_roots() {
    for wave64 in gpu_waves() {
        f64_roots(wave64);
    }
}

fn f64_roots(wave64: bool) {
    let p = prog(&[
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_RCP_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    f64_lanes(&mut st, &F64_EDGES);
    let (r, lanes) = run_wave(&p, &st, wave64);
    let ulps =|got: f64, want: f64| -> u64 {
        if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
            0
        } else if got.is_nan() || want.is_nan() || got.is_sign_negative() != want.is_sign_negative() {
            u64::MAX
        } else {
            got.to_bits().abs_diff(want.to_bits())
        }
    };
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let x = f64::from_bits(F64_EDGES[l % F64_EDGES.len()]);
        for (name, got, want, tol) in [("sqrt", pair(v, 2), x.sqrt(), 1), ("rsq", pair(v, 4), x.sqrt().recip(), 2), ("rcp", pair(v, 6), x.recip(), 1)] {
            if ulps(got, want) > tol {
                bad.push(format!("v_{name}_f64({x:e}) = {got:e}, want {want:e}"));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// v_ldexp_f64 = x · 2^n, rounded once, for every i32 n: results that over- or underflow,
/// denormal results, exponents beyond the f32 range of 2^n.
#[test]
fn gpu_ldexp_f64() {
    for wave64 in gpu_waves() {
        ldexp_f64(wave64);
    }
}

fn ldexp_f64(wave64: bool) {
    let p = prog(&[("V_LDEXP_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2)], None)]);
    // 2^e as f64 bits for a normal e
    const fn p2(e: i32) -> u64 {
        ((e + 1023) as u64) << 52
    }
    const ONE: u64 = p2(0);
    const CASES: [(u64, i32, u64); 21] = [
        (ONE, 0, ONE),
        (ONE, 1023, p2(1023)),
        (ONE, 1024, 0x7FF0_0000_0000_0000),                 // overflows to inf
        (ONE, -1022, p2(-1022)),                            // smallest normal
        (ONE, -1074, 1),                                    // smallest denormal
        (ONE, -1075, 0),                                    // half of it, a tie that rounds to even 0
        (1.5f64.to_bits(), -1074, 2),                       // 1.5 · 2^−1074, a tie that rounds to even 2
        (1.5f64.to_bits(), -1075, 1),                       // 0.75 · 2^−1074 rounds to 1
        (1, 2000, p2(926)),                                 // 2^−1074 · 2^2000
        (1, 1074, ONE),                                     // back to 1.0
        (p2(1023), -2046, 1 << 51),                         // 2^−1023, a denormal
        (1e300f64.to_bits(), -2000, 0x0137_E43C_8800_759C),  // only the exponent field moves
        (1e300f64.to_bits(), -3000, 0),                     // underflows to 0
        ((-2.0f64).to_bits(), 200, p2(201) | 1 << 63),      // −2^201
        (2.0f64.to_bits(), -200, p2(-199)),
        (ONE, i32::MAX, 0x7FF0_0000_0000_0000),             // inf
        ((-1.0f64).to_bits(), i32::MIN, 1 << 63),           // −0
        (1 << 63, 5, 1 << 63),                              // −0 stays −0
        (0x7FF0_0000_0000_0000, -5000, 0x7FF0_0000_0000_0000), // inf stays inf
        (0x7FF8_0000_0000_1234, 7, 0x7FF8_0000_0000_1234),  // NaN stays NaN
        (0, 3000, 0),                                       // 0 stays 0
    ];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, n, _) = CASES[l % CASES.len()];
        (r[0], r[1], r[2]) = (x as u32, (x >> 32) as u32, n as u32);
    }
    let (r, lanes) = run_wave(&p, &st, wave64);
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let (x, n, want) = CASES[l % CASES.len()];
        let (got, want, x) = (pair(v, 4), f64::from_bits(want), f64::from_bits(x));
        if !(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())) {
            bad.push(format!("v_ldexp_f64({x:e}, {n}) = {got:e} ({:#018x}), want {want:e}", got.to_bits()));
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

#[test]
fn gpu_legacy_fma_adds_the_zero_product() {
    let code = prog(&[
        ("V_FMA_LEGACY_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_FMAC_LEGACY_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V), ("VSRC1", 1)], None),
    ]);
    let cases = [
        (0.0f32, 3.0f32, -0.0f32, 0.0f32),
        (-0.0, 3.0, -0.0, 0.0),
        (3.0, -0.0, -0.0, 0.0),
        (0.0, f32::INFINITY, -0.0, 0.0),
        (f32::NAN, -0.0, -0.0, 0.0),
        (-0.0, 3.0, -1.0, -1.0),
        (2.0, 3.0, -1.0, 5.0),
    ];
    for wave64 in gpu_waves() {
        let mut st = KernelState::default();
        for (lane, regs) in st.vgpr.iter_mut().enumerate() {
            let (a, b, c, _) = cases[lane % cases.len()];
            (regs[0], regs[1], regs[2], regs[5]) = (a.to_bits(), b.to_bits(), c.to_bits(), c.to_bits());
        }
        let (result, lanes) = run_wave(&code, &st, wave64);
        for (lane, regs) in result.vgpr.iter().take(lanes).enumerate() {
            let expected = cases[lane % cases.len()].3.to_bits();
            assert_eq!(regs[4], expected, "wave{lanes} lane {lane}: v_fma_legacy_f32");
            assert_eq!(regs[5], expected, "wave{lanes} lane {lane}: v_fmac_legacy_f32");
        }
    }
}

#[test]
fn gpu_valu_float_and_int() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 128), ("VSRC1", 3)], None), // 0 * inf = 0
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", 244), ("SRC2", 243)], None), // v1*2-1
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 1), ("VSRC1", 2)], None), // v13 += v1*v2
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 4), ("VSRC1", 5)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 15), ("SRC0", 132), ("VSRC1", 4)], None), // v4 << 4
        ("V_BFE_U32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 6), ("SRC1", 136), ("SRC2", 132)], None), // bits 8..11
        ("V_CVT_F32_UBYTE1", "ENC_VOP1", &[("VDST", 17), ("SRC0", V + 6)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", 128)], None), // lane index (low half)
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", V + 18)], None), // + high half
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", 128), ("SRC2", 242)], None), // clamp to [0,1]
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", V + 2), ("NEG", 1), ("OMOD", 1)], None), // (-v1 + v2) * 2
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = f(l as f32 * 0.25);
        r[2] = f(1.5);
        r[3] = f(f32::INFINITY);
        r[4] = l as u32;
        r[5] = 1000;
        r[6] = 0x0000_AB00 | l as u32;
        r[13] = f(10.0);
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let x = l as f32 * 0.25;
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!(fl(10), x + 1.5, "lane {l} v_add_f32");
        assert_eq!(v[11], 0, "lane {l} v_mul_legacy_f32 0*inf");
        assert_eq!(fl(12), x * 2.0 - 1.0, "lane {l} v_fma_f32");
        assert_eq!(fl(13), 10.0 + x * 1.5, "lane {l} v_mac_f32");
        assert_eq!(v[14], l as u32 + 1000, "lane {l} v_add_nc_u32");
        assert_eq!(v[15], (l as u32) << 4, "lane {l} v_lshlrev_b32");
        assert_eq!(v[16], 0xB, "lane {l} v_bfe_u32");
        assert_eq!(fl(17), 0xAB as f32, "lane {l} v_cvt_f32_ubyte1");
        assert_eq!(v[18], l as u32, "lane {l} v_mbcnt");
        assert_eq!(fl(19), x.clamp(0.0, 1.0), "lane {l} v_med3_f32");
        assert_eq!(fl(20), (1.5 - x) * 2.0, "lane {l} VOP3 neg+omod");
    }
}

/// ABS / NEG source modifiers on the VOP3 forms of the untyped move and select (applied on
/// the sign bit: NaN payloads, -0 and denormals pass bit-exact) and of the f16 → f32 and
/// f32 → f64 conversions (RDNA 2 ISA, V_CNDMASK_B32 / V_MOV_B32: "suitable for negating or
/// taking the absolute value"; LLVM `VOPSelectModsPat`, MC `gfx10_asm_vop2.s` / `gfx10_asm_vop1.s`).
#[test]
fn gpu_abs_neg_on_move_select_and_conversions() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v_cndmask_b32_e64 v5, -v1, |v2|, vcc
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", 106), ("NEG", 1), ("ABS", 2)], None),
        ("V_MOV_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 1), ("NEG", 1)], None), // -v1
        ("V_MOV_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 2), ("ABS", 1), ("NEG", 1)], None), // -|v2|
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 3), ("NEG", 1)], None), // -v3.h
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 3), ("ABS", 1)], None), // |v3.h|
        ("V_CVT_F64_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("NEG", 1)], None), // -v4
        ("V_CVT_F64_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 4), ("ABS", 1)], None), // |v4|
    ]);
    // (v1, v2): ordinary, a quiet NaN with a payload against a negative signalling NaN, -0 against
    // -0, and denormals
    let pairs: [(u32, u32); 4] =
        [(f(1.0), f(-2.0)), (0x7FC1_2345, 0xFFA0_0001), (0x8000_0000, 0x8000_0000), (0x0000_0001, 0x8000_0001)];
    // v3 (f16) with -v3 / |v3| as f32
    let halves: [(u32, f32, f32); 4] = [(0x3C00, -1.0, 1.0), (0x8000, 0.0, 0.0), (0xC000, 2.0, 2.0), (0x3800, -0.5, 0.5)];
    let singles: [f32; 4] = [1.0, -0.0, -3.5, 0.25];
    let mut st = KernelState::default();
    st.sgpr[106] = 0x5555_5555; // vcc: even lanes take src1
    st.sgpr[107] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let k = (l / 2) % 4; // each input in a VCC-set and a VCC-clear lane
        (r[1], r[2]) = pairs[k];
        r[3] = 0xDEAD_0000 | halves[k].0;
        r[4] = f(singles[k]);
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let k = (l / 2) % 4;
        let (a, b) = pairs[k];
        let want = if l % 2 == 0 { b & 0x7FFF_FFFF } else { a ^ 0x8000_0000 };
        assert_eq!(v[5], want, "lane {l} v_cndmask_b32 -v1, |v2| ({a:#010X}, {b:#010X})");
        assert_eq!(v[6], a ^ 0x8000_0000, "lane {l} v_mov_b32 -v1 ({a:#010X})");
        assert_eq!(v[7], b | 0x8000_0000, "lane {l} v_mov_b32 -|v2| ({b:#010X})");
        assert_eq!(v[8], f(halves[k].1), "lane {l} v_cvt_f32_f16 -v3 ({:#06X})", halves[k].0);
        assert_eq!(v[9], f(halves[k].2), "lane {l} v_cvt_f32_f16 |v3| ({:#06X})", halves[k].0);
        let x = singles[k] as f64;
        let d = |i: usize| ((v[i + 1] as u64) << 32) | v[i] as u64;
        assert_eq!(d(10), (-x).to_bits(), "lane {l} v_cvt_f64_f32 -v4 ({x})");
        assert_eq!(d(12), x.abs().to_bits(), "lane {l} v_cvt_f64_f32 |v4| ({x})");
    }
}

/// Output modifiers each instruction kind applies: omod on f64 (clamp stays in double
/// precision) and on the f64 → f32 / f32 → f16 conversions, and the integer clamp (saturation) of the
/// unsigned adds, the packed 16-bit adds and the integer dot products.
#[test]
fn gpu_clamp_and_omod_per_instruction() {
    if !gpu_ready() {
        return;
    }
    let p = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3P", f, None);
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let code = prog(&[
        // f64: v[1:2] = 0.1, v[3:4] = 0.2, v[5:6] = 1.25
        v("V_ADD_F64", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)]),
        v("V_ADD_F64", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 3), ("OMOD", 1)]),
        v("V_ADD_F64", &[("VDST", 14), ("SRC0", V + 5), ("SRC1", V + 1), ("CLAMP", 1)]),
        v("V_CVT_F32_F64", &[("VDST", 16), ("SRC0", V + 5), ("CLAMP", 1)]),
        v("V_CVT_F32_F64", &[("VDST", 17), ("SRC0", V + 5), ("OMOD", 3)]),
        // f32 → f16: v7 = 3.0
        v("V_CVT_F16_F32", &[("VDST", 18), ("SRC0", V + 7), ("CLAMP", 1)]),
        v("V_CVT_F16_F32", &[("VDST", 19), ("SRC0", V + 7), ("OMOD", 1)]),
        // unsigned: v8 = 0xFFFFFFF0, v9 = 0x20, v20 = 0x10
        v("V_ADD_NC_U32", &[("VDST", 21), ("SRC0", V + 8), ("SRC1", V + 9), ("CLAMP", 1)]),
        v("V_SUB_NC_U32", &[("VDST", 22), ("SRC0", V + 20), ("SRC1", V + 9), ("CLAMP", 1)]),
        v("V_SUBREV_NC_U32", &[("VDST", 23), ("SRC0", V + 9), ("SRC1", V + 20), ("CLAMP", 1)]),
        v("V_ADD3_U32", &[("VDST", 24), ("SRC0", V + 8), ("SRC1", V + 9), ("SRC2", V + 9), ("CLAMP", 1)]),
        v("V_ADD_NC_U32", &[("VDST", 25), ("SRC0", V + 8), ("SRC1", V + 9)]), // no clamp: wraps
        // packed 16-bit: v26 = (0xFFF0, 0x0005), v27 = (0x0020, 0x0002), v28 = (0x7FF0, 0x8010), v29 = (0x0020, 0xFFE0)
        p("V_PK_ADD_U16", &[("VDST", 30), ("SRC0", V + 26), ("SRC1", V + 27), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_PK_SUB_U16", &[("VDST", 31), ("SRC0", V + 27), ("SRC1", V + 26), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_PK_ADD_I16", &[("VDST", 32), ("SRC0", V + 28), ("SRC1", V + 29), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        // halves of src1 swapped: (0x7FF0 − (−32), −32752 − 32)
        p("V_PK_SUB_I16", &[("VDST", 33), ("SRC0", V + 28), ("SRC1", V + 29), ("OP_SEL", 2), ("OP_SEL_HI", 1), ("CLAMP", 1)]),
        p("V_PK_MAD_U16", &[("VDST", 34), ("SRC0", V + 26), ("SRC1", V + 27), ("SRC2", V + 27), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)]),
        p("V_PK_MAD_I16", &[("VDST", 35), ("SRC0", V + 28), ("SRC1", V + 27), ("SRC2", V + 29), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)]),
        p("V_PK_ADD_U16", &[("VDST", 36), ("SRC0", V + 26), ("SRC1", V + 27), ("OP_SEL_HI", 3)]), // no clamp: wraps
        // dot products: v37 = 0x7F7F7F7F, v38 = 0xFFFFFFFF, v39 = 0x7FFFFFF0, v40 = 0x80000010
        p("V_DOT4_I32_I8", &[("VDST", 41), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("CLAMP", 1)]),
        p("V_DOT4_I32_I8", &[("VDST", 42), ("SRC0", V + 37), ("SRC1", V + 38), ("SRC2", V + 40), ("CLAMP", 1)]),
        p("V_DOT4_U32_U8", &[("VDST", 43), ("SRC0", V + 38), ("SRC1", V + 38), ("SRC2", V + 8), ("CLAMP", 1)]),
        p("V_DOT8_I32_I4", &[("VDST", 44), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("CLAMP", 1)]),
        p("V_DOT8_U32_U4", &[("VDST", 45), ("SRC0", V + 38), ("SRC1", V + 38), ("SRC2", V + 8), ("CLAMP", 1)]),
        p("V_DOT2_I32_I16", &[("VDST", 46), ("SRC0", V + 28), ("SRC1", V + 28), ("SRC2", V + 39), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_DOT2_U32_U16", &[("VDST", 47), ("SRC0", V + 26), ("SRC1", V + 26), ("SRC2", V + 8), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_DOT4_I32_I8", &[("VDST", 48), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("OP_SEL_HI", 3)]), // no clamp: wraps
    ]);
    let d = |x: f64, r: &mut [u32], i: usize| {
        r[i] = x.to_bits() as u32;
        r[i + 1] = (x.to_bits() >> 32) as u32;
    };
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        d(0.1, r, 1);
        d(0.2, r, 3);
        d(1.25, r, 5);
        r[7] = f(3.0);
        r[8] = 0xFFFF_FFF0;
        r[9] = 0x20;
        r[20] = 0x10;
        r[26] = 0x0005_FFF0;
        r[27] = 0x0002_0020;
        r[28] = 0x8010_7FF0;
        r[29] = 0xFFE0_0020;
        r[37] = 0x7F7F_7F7F;
        r[38] = 0xFFFF_FFFF;
        r[39] = 0x7FFF_FFF0;
        r[40] = 0x8000_0010;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    let d64 = |v: &[u32], i: usize| f64::from_bits(v[i] as u64 | (v[i + 1] as u64) << 32);
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(d64(v, 10), 0.1 + 0.2, "lane {l} v_add_f64 clamp keeps double precision");
        assert_eq!(d64(v, 12), (0.1 + 0.2) * 2.0, "lane {l} v_add_f64 omod *2");
        assert_eq!(d64(v, 14), 1.0, "lane {l} v_add_f64 clamp");
        assert_eq!(f32::from_bits(v[16]), 1.0, "lane {l} v_cvt_f32_f64 clamp");
        assert_eq!(f32::from_bits(v[17]), 0.625, "lane {l} v_cvt_f32_f64 omod /2");
        assert_eq!(v[18] & 0xFFFF, 0x3C00, "lane {l} v_cvt_f16_f32 clamp");
        assert_eq!(v[19] & 0xFFFF, 0x4600, "lane {l} v_cvt_f16_f32 omod *2");
        assert_eq!(v[21..26], [0xFFFF_FFFF, 0, 0, 0xFFFF_FFFF, 0x10], "lane {l} add / sub / subrev / add3 u32 clamp, add without");
        assert_eq!(v[30], 0x0007_FFFF, "lane {l} v_pk_add_u16 clamp");
        assert_eq!(v[31], 0x0000_0000, "lane {l} v_pk_sub_u16 clamp");
        assert_eq!(v[32], 0x8000_7FFF, "lane {l} v_pk_add_i16 clamp");
        assert_eq!(v[33], 0x8000_7FFF, "lane {l} v_pk_sub_i16 clamp");
        assert_eq!(v[34], 0x000C_FFFF, "lane {l} v_pk_mad_u16 clamp");
        assert_eq!(v[35], 0x8000_7FFF, "lane {l} v_pk_mad_i16 clamp");
        assert_eq!(v[36], 0x0007_0010, "lane {l} v_pk_add_u16 without clamp");
        assert_eq!(v[41], 0x7FFF_FFFF, "lane {l} v_dot4_i32_i8 clamp high");
        assert_eq!(v[42], 0x8000_0000, "lane {l} v_dot4_i32_i8 clamp low");
        assert_eq!(v[43], 0xFFFF_FFFF, "lane {l} v_dot4_u32_u8 clamp");
        assert_eq!(v[44], 0x7FFF_FFFF, "lane {l} v_dot8_i32_i4 clamp");
        assert_eq!(v[45], 0xFFFF_FFFF, "lane {l} v_dot8_u32_u4 clamp");
        assert_eq!(v[46], 0x7FFF_FFFF, "lane {l} v_dot2_i32_i16 clamp");
        assert_eq!(v[47], 0xFFFF_FFFF, "lane {l} v_dot2_u32_u16 clamp");
        assert_eq!(v[48], 0x7FFF_FFF0u32.wrapping_add(4 * 127 * 127), "lane {l} v_dot4_i32_i8 without clamp");
    }
}

/// 16-bit VALU integer CLAMP=1 saturation vs CLAMP=0 wrapping, plus OP_SEL half selection and
/// preservation of the untouched half of the destination VGPR.
#[test]
fn gpu_int_clamp_16bit() {
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let code = prog(&[
        // v_add_nc_u16
        v("V_ADD_NC_U16", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        v("V_ADD_NC_U16", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1)]),
        v("V_ADD_NC_U16", &[("VDST", 12), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL", 0b1011), ("CLAMP", 1)]),
        v("V_ADD_NC_U16", &[("VDST", 13), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL", 0b1000), ("CLAMP", 1)]),
        // v_sub_nc_u16
        v("V_SUB_NC_U16", &[("VDST", 14), ("SRC0", V + 1), ("SRC1", V + 0), ("CLAMP", 1)]),
        v("V_SUB_NC_U16", &[("VDST", 15), ("SRC0", V + 1), ("SRC1", V + 0)]),
        v("V_SUB_NC_U16", &[("VDST", 16), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL", 0b0011), ("CLAMP", 1)]),
        v("V_SUB_NC_U16", &[("VDST", 17), ("SRC0", V + 1), ("SRC1", V + 1), ("CLAMP", 1)]),
        // v_add_nc_i16
        v("V_ADD_NC_I16", &[("VDST", 18), ("SRC0", V + 2), ("SRC1", V + 3), ("CLAMP", 1)]),
        v("V_ADD_NC_I16", &[("VDST", 19), ("SRC0", V + 2), ("SRC1", V + 3)]),
        v("V_ADD_NC_I16", &[("VDST", 20), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0011), ("CLAMP", 1)]),
        v("V_ADD_NC_I16", &[("VDST", 21), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0011)]),
        // v_sub_nc_i16
        v("V_SUB_NC_I16", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0010), ("CLAMP", 1)]),
        v("V_SUB_NC_I16", &[("VDST", 23), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0010)]),
        v("V_SUB_NC_I16", &[("VDST", 24), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0001), ("CLAMP", 1)]),
        v("V_SUB_NC_I16", &[("VDST", 25), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0001)]),
        // v_mad_u16: max intermediate 65535*65535 + 65535 = 0xFFFF_0000 fits in u32
        v("V_MAD_U16", &[("VDST", 26), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 4), ("CLAMP", 1)]),
        v("V_MAD_U16", &[("VDST", 27), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 4)]),
        v("V_MAD_U16", &[("VDST", 28), ("SRC0", V + 7), ("SRC1", V + 7), ("SRC2", V + 8), ("OP_SEL", 0b0010), ("CLAMP", 1)]),
        // v_mad_i16: extreme positive (-32768)*(-32768)+1 and negative (-32768)*32767+(-32768)
        v("V_MAD_I16", &[("VDST", 29), ("SRC0", V + 5), ("SRC1", V + 5), ("SRC2", V + 6), ("CLAMP", 1)]),
        v("V_MAD_I16", &[("VDST", 30), ("SRC0", V + 5), ("SRC1", V + 5), ("SRC2", V + 6)]),
        v("V_MAD_I16", &[("VDST", 31), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 5), ("OP_SEL", 0b0010), ("CLAMP", 1)]),
        v("V_MAD_I16", &[("VDST", 32), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 5), ("OP_SEL", 0b0010)]),
        v("V_MAD_I16", &[("VDST", 33), ("SRC0", V + 3), ("SRC1", V + 7), ("SRC2", V + 8), ("OP_SEL", 0b0001), ("CLAMP", 1)]),
        // Regression check for negative-clamped low half preserving upper half on packed/cvt i16:
        // v_pk_add_i16 with lo = hi(v2)+hi(v3) (-32752 + -32 -> 0x8000), hi = lo(v2)+lo(v3) (+32752 + 32 -> 0x7FFF)
        ("V_PK_ADD_I16", "ENC_VOP3P", &[("VDST", 34), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 3), ("CLAMP", 1)], None),
        // v_cvt_pk_i16_i32 with lo = -40000 -> 0x8000, hi = 1 -> 0x0001
        v("V_CVT_PK_I16_I32", &[("VDST", 35), ("SRC0", V + 9), ("SRC1", 129)]),
        // v_cvt_i16_f16 with -40000.0 (f16 = 0xF8E2), NaN (0x7E00), +inf (0x7C00), -inf (0xFC00)
        v("V_CVT_I16_F16", &[("VDST", 36), ("SRC0", V + 37)]),
        v("V_CVT_I16_F16", &[("VDST", 38), ("SRC0", V + 37), ("OP_SEL", 0b0001)]),
        v("V_CVT_I16_F16", &[("VDST", 39), ("SRC0", V + 40)]),
        v("V_CVT_I16_F16", &[("VDST", 41), ("SRC0", V + 40), ("OP_SEL", 0b0001)]),
        // v_mul_lo_u16: 65535*65535 = 0xFFFE_0001 (overflow -> 0xFFFF / wrap -> 0x0001), 100*200 = 20000 (0x4E20)
        v("V_MUL_LO_U16", &[("VDST", 42), ("SRC0", V + 4), ("SRC1", V + 4), ("CLAMP", 1)]),
        v("V_MUL_LO_U16", &[("VDST", 43), ("SRC0", V + 4), ("SRC1", V + 4)]),
        v("V_MUL_LO_U16", &[("VDST", 44), ("SRC0", V + 7), ("SRC1", V + 7), ("OP_SEL", 0b1010), ("CLAMP", 1)]),
        // v_pk_mul_lo_u16: lo = 0xFFF0*0x0020 = 0x1F_FE00 (ov -> 0xFFFF / wrap -> 0xFE00), hi = 5*2 = 10 (in-range)
        ("V_PK_MUL_LO_U16", "ENC_VOP3P", &[("VDST", 45), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_PK_MUL_LO_U16", "ENC_VOP3P", &[("VDST", 46), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3)], None),
        // Negative-result i16 min/max/med3 sanity check (verifies upper half preservation on NVIDIA SPIR-V)
        v("V_MIN_I16", &[("VDST", 47), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0011)]),
        v("V_MAX_I16", &[("VDST", 48), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 0b0011)]),
        v("V_MED3_I16", &[("VDST", 49), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", V + 5), ("OP_SEL", 0b0011)]),
        v("V_MIN3_I16", &[("VDST", 50), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", V + 5), ("OP_SEL", 0b0011)]),
        v("V_MAX3_I16", &[("VDST", 51), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", V + 5), ("OP_SEL", 0b0011)]),
        ("V_PK_MIN_I16", "ENC_VOP3P", &[("VDST", 52), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 3)], None),
        ("V_PK_MAX_I16", "ENC_VOP3P", &[("VDST", 53), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 3)], None),
        // Exact boundary pairs: 0xFFFF + 0 vs +1, 32767 + 0 vs +1, -32768 - 0 vs -1
        v("V_ADD_NC_U16", &[("VDST", 54), ("SRC0", V + 4), ("SRC1", 128), ("CLAMP", 1)]),
        v("V_ADD_NC_U16", &[("VDST", 55), ("SRC0", V + 4), ("SRC1", 129), ("CLAMP", 1)]),
        v("V_ADD_NC_U16", &[("VDST", 56), ("SRC0", V + 4), ("SRC1", 129)]),
        v("V_ADD_NC_I16", &[("VDST", 57), ("SRC0", V + 6), ("SRC1", 128), ("OP_SEL", 0b0001), ("CLAMP", 1)]),
        v("V_ADD_NC_I16", &[("VDST", 58), ("SRC0", V + 6), ("SRC1", 129), ("OP_SEL", 0b0001), ("CLAMP", 1)]),
        v("V_ADD_NC_I16", &[("VDST", 59), ("SRC0", V + 6), ("SRC1", 129), ("OP_SEL", 0b0001)]),
        v("V_SUB_NC_I16", &[("VDST", 60), ("SRC0", V + 5), ("SRC1", 128), ("CLAMP", 1)]),
        v("V_SUB_NC_I16", &[("VDST", 61), ("SRC0", V + 5), ("SRC1", 129), ("CLAMP", 1)]),
        v("V_SUB_NC_I16", &[("VDST", 62), ("SRC0", V + 5), ("SRC1", 129)]),
    ]);
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[0] = 0x0005_FFF0;
        r[1] = 0x0002_0020;
        r[2] = 0x8010_7FF0;
        r[3] = 0xFFE0_0020;
        r[4] = 0xFFFF_FFFF;
        r[5] = 0x8000_8000;
        r[6] = 0x7FFF_0001;
        r[7] = 0x00C8_0064;
        r[8] = 0x0000_0037;
        r[9] = (-40_000i32) as u32;
        r[37] = 0x7E00_F8E2; // hi: NaN, lo: -40000.0 in binary16
        r[40] = 0xFC00_7C00; // hi: -inf, lo: +inf in binary16
        for d in (10..=36).chain([38, 39]).chain(41..=62) {
            r[d] = 0xDEAD_BEEF;
        }
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&code, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[10], 0xDEAD_FFFF, "wave{lanes} lane {l} v_add_nc_u16 clamp");
            assert_eq!(v[11], 0xDEAD_0010, "wave{lanes} lane {l} v_add_nc_u16 wrap");
            assert_eq!(v[12], 0x0007_BEEF, "wave{lanes} lane {l} v_add_nc_u16 clamp hi in-range");
            assert_eq!(v[13], 0xFFFF_BEEF, "wave{lanes} lane {l} v_add_nc_u16 clamp dst hi");
            assert_eq!(v[14], 0xDEAD_0000, "wave{lanes} lane {l} v_sub_nc_u16 clamp underflow");
            assert_eq!(v[15], 0xDEAD_0030, "wave{lanes} lane {l} v_sub_nc_u16 wrap");
            assert_eq!(v[16], 0xDEAD_0003, "wave{lanes} lane {l} v_sub_nc_u16 clamp in-range");
            assert_eq!(v[17], 0xDEAD_0000, "wave{lanes} lane {l} v_sub_nc_u16 clamp zero");
            assert_eq!(v[18], 0xDEAD_7FFF, "wave{lanes} lane {l} v_add_nc_i16 clamp +ov");
            assert_eq!(v[19], 0xDEAD_8010, "wave{lanes} lane {l} v_add_nc_i16 wrap +ov");
            assert_eq!(v[20], 0xDEAD_8000, "wave{lanes} lane {l} v_add_nc_i16 clamp -ov");
            assert_eq!(v[21], 0xDEAD_7FF0, "wave{lanes} lane {l} v_add_nc_i16 wrap -ov");
            assert_eq!(v[22], 0xDEAD_7FFF, "wave{lanes} lane {l} v_sub_nc_i16 clamp +ov");
            assert_eq!(v[23], 0xDEAD_8010, "wave{lanes} lane {l} v_sub_nc_i16 wrap +ov");
            assert_eq!(v[24], 0xDEAD_8000, "wave{lanes} lane {l} v_sub_nc_i16 clamp -ov");
            assert_eq!(v[25], 0xDEAD_7FF0, "wave{lanes} lane {l} v_sub_nc_i16 wrap -ov");
            assert_eq!(v[26], 0xDEAD_FFFF, "wave{lanes} lane {l} v_mad_u16 clamp max");
            assert_eq!(v[27], 0xDEAD_0000, "wave{lanes} lane {l} v_mad_u16 wrap max");
            assert_eq!(v[28], 0xDEAD_4E57, "wave{lanes} lane {l} v_mad_u16 clamp in-range");
            assert_eq!(v[29], 0xDEAD_7FFF, "wave{lanes} lane {l} v_mad_i16 clamp +ov");
            assert_eq!(v[30], 0xDEAD_0001, "wave{lanes} lane {l} v_mad_i16 wrap +ov");
            assert_eq!(v[31], 0xDEAD_8000, "wave{lanes} lane {l} v_mad_i16 clamp -ov");
            assert_eq!(v[32], 0xDEAD_0000, "wave{lanes} lane {l} v_mad_i16 wrap -ov");
            assert_eq!(v[33], 0xDEAD_F3B7, "wave{lanes} lane {l} v_mad_i16 clamp in-range neg");
            assert_eq!(v[34], 0x7FFF_8000, "wave{lanes} lane {l} v_pk_add_i16 clamp neg lo + pos hi");
            assert_eq!(v[35], 0x0001_8000, "wave{lanes} lane {l} v_cvt_pk_i16_i32 neg lo + pos hi");
            assert_eq!(v[36], 0xDEAD_8000, "wave{lanes} lane {l} v_cvt_i16_f16 neg clamp");
            assert_eq!(v[38], 0xDEAD_0000, "wave{lanes} lane {l} v_cvt_i16_f16 NaN");
            assert_eq!(v[39], 0xDEAD_7FFF, "wave{lanes} lane {l} v_cvt_i16_f16 +inf");
            assert_eq!(v[41], 0xDEAD_8000, "wave{lanes} lane {l} v_cvt_i16_f16 -inf");
            assert_eq!(v[42], 0xDEAD_FFFF, "wave{lanes} lane {l} v_mul_lo_u16 clamp");
            assert_eq!(v[43], 0xDEAD_0001, "wave{lanes} lane {l} v_mul_lo_u16 wrap");
            assert_eq!(v[44], 0x4E20_BEEF, "wave{lanes} lane {l} v_mul_lo_u16 clamp hi in-range");
            assert_eq!(v[45], 0x000A_FFFF, "wave{lanes} lane {l} v_pk_mul_lo_u16 clamp");
            assert_eq!(v[46], 0x000A_FE00, "wave{lanes} lane {l} v_pk_mul_lo_u16 wrap");
            assert_eq!(v[47], 0xDEAD_8010, "wave{lanes} lane {l} v_min_i16 neg preserves hi");
            assert_eq!(v[48], 0xDEAD_FFE0, "wave{lanes} lane {l} v_max_i16 neg preserves hi");
            assert_eq!(v[49], 0xDEAD_8010, "wave{lanes} lane {l} v_med3_i16 neg preserves hi");
            assert_eq!(v[50], 0xDEAD_8000, "wave{lanes} lane {l} v_min3_i16 neg preserves hi");
            assert_eq!(v[51], 0xDEAD_FFE0, "wave{lanes} lane {l} v_max3_i16 neg preserves hi");
            assert_eq!(v[52], 0x0020_8010, "wave{lanes} lane {l} v_pk_min_i16 neg lo + pos hi");
            assert_eq!(v[53], 0x7FF0_FFE0, "wave{lanes} lane {l} v_pk_max_i16 neg lo + pos hi");
            assert_eq!(v[54], 0xDEAD_FFFF, "wave{lanes} lane {l} v_add_nc_u16 0xFFFF + 0 clamp");
            assert_eq!(v[55], 0xDEAD_FFFF, "wave{lanes} lane {l} v_add_nc_u16 0xFFFF + 1 clamp");
            assert_eq!(v[56], 0xDEAD_0000, "wave{lanes} lane {l} v_add_nc_u16 0xFFFF + 1 wrap");
            assert_eq!(v[57], 0xDEAD_7FFF, "wave{lanes} lane {l} v_add_nc_i16 32767 + 0 clamp");
            assert_eq!(v[58], 0xDEAD_7FFF, "wave{lanes} lane {l} v_add_nc_i16 32767 + 1 clamp");
            assert_eq!(v[59], 0xDEAD_8000, "wave{lanes} lane {l} v_add_nc_i16 32767 + 1 wrap");
            assert_eq!(v[60], 0xDEAD_8000, "wave{lanes} lane {l} v_sub_nc_i16 -32768 - 0 clamp");
            assert_eq!(v[61], 0xDEAD_8000, "wave{lanes} lane {l} v_sub_nc_i16 -32768 - 1 clamp");
            assert_eq!(v[62], 0xDEAD_7FFF, "wave{lanes} lane {l} v_sub_nc_i16 -32768 - 1 wrap");
        }
    }
}

/// 32-bit integer CLAMP=1 saturation on MAD (u16/i16/u24/i24), MUL24 (u24/i24), SAD/MSAD,
/// SDWA CLAMP=1 on MUL24, and V_CVT_PKRTZ_F16_F32 ignoring VOP3 OMOD/CLAMP.
#[test]
fn gpu_int_clamp_32bit_mad_mul_sad() {
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let code = prog(&[
        // v_mad_u32_u16: v0 = 0xFFFF_0003, v1 = 0xFFFF_0004, v2 = 0x0002_0000, v3 = 0x0001_FFFE
        v("V_MAD_U32_U16", &[("VDST", 20), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("OP_SEL", 3), ("CLAMP", 1)]),
        v("V_MAD_U32_U16", &[("VDST", 21), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("OP_SEL", 3)]),
        v("V_MAD_U32_U16", &[("VDST", 22), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 3), ("OP_SEL", 3), ("CLAMP", 1)]),
        // v_mad_i32_i16: v4 = 0x8000_7FFF, v5 = 0x7000_0000, v6 = 0x9000_0000 (-0x7000_0000)
        // hi*hi + v5 = (-32768)*(-32768) + 0x7000_0000 = +0xB000_0000 -> +ov
        v("V_MAD_I32_I16", &[("VDST", 23), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 5), ("OP_SEL", 3), ("CLAMP", 1)]),
        v("V_MAD_I32_I16", &[("VDST", 24), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 5), ("OP_SEL", 3)]),
        // hi*lo + v6 = (-32768)*32767 + (-0x7000_0000) -> -ov
        v("V_MAD_I32_I16", &[("VDST", 25), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 6), ("OP_SEL", 1), ("CLAMP", 1)]),
        v("V_MAD_I32_I16", &[("VDST", 26), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 6), ("OP_SEL", 1)]),
        // v_mul_u32_u24: v7 = 0xABFF_FFFF (u24 = 0xFFFFFF), v8 = 0x0001_0000, v9 = 0x0000_FFFF
        v("V_MUL_U32_U24", &[("VDST", 27), ("SRC0", V + 7), ("SRC1", V + 7), ("CLAMP", 1)]),
        v("V_MUL_U32_U24", &[("VDST", 28), ("SRC0", V + 7), ("SRC1", V + 7)]),
        v("V_MUL_U32_U24", &[("VDST", 29), ("SRC0", V + 8), ("SRC1", V + 9), ("CLAMP", 1)]),
        // v_mul_i32_i24: v10 = 0x0080_0000 (i24 = -2^23), v11 = 0x007F_FFFF (i24 = 2^23 - 1)
        v("V_MUL_I32_I24", &[("VDST", 30), ("SRC0", V + 10), ("SRC1", V + 10), ("CLAMP", 1)]),
        v("V_MUL_I32_I24", &[("VDST", 31), ("SRC0", V + 10), ("SRC1", V + 10)]),
        v("V_MUL_I32_I24", &[("VDST", 32), ("SRC0", V + 10), ("SRC1", V + 11), ("CLAMP", 1)]),
        v("V_MUL_I32_I24", &[("VDST", 33), ("SRC0", V + 10), ("SRC1", V + 11)]),
        // v_mad_u32_u24: product overflow and add-only overflow
        v("V_MAD_U32_U24", &[("VDST", 34), ("SRC0", V + 7), ("SRC1", V + 7), ("SRC2", 133), ("CLAMP", 1)]),
        v("V_MAD_U32_U24", &[("VDST", 35), ("SRC0", V + 7), ("SRC1", V + 7), ("SRC2", 133)]),
        v("V_MAD_U32_U24", &[("VDST", 36), ("SRC0", V + 8), ("SRC1", V + 9), ("SRC2", V + 2), ("CLAMP", 1)]),
        v("V_MAD_U32_U24", &[("VDST", 37), ("SRC0", V + 8), ("SRC1", V + 9), ("SRC2", V + 2)]),
        // v_mad_i32_i24: positive overflow, negative underflow, and wrapping control
        v("V_MAD_I32_I24", &[("VDST", 38), ("SRC0", V + 11), ("SRC1", V + 11), ("SRC2", 138), ("CLAMP", 1)]),
        v("V_MAD_I32_I24", &[("VDST", 39), ("SRC0", V + 11), ("SRC1", V + 11), ("SRC2", 138)]),
        v("V_MAD_I32_I24", &[("VDST", 40), ("SRC0", V + 10), ("SRC1", V + 11), ("SRC2", 198), ("CLAMP", 1)]),
        v("V_MAD_I32_I24", &[("VDST", 41), ("SRC0", V + 10), ("SRC1", V + 11), ("SRC2", 198)]),
        // v_sad_u32: v12 = 0xFFFF_FFF0, v13 = 0x0000_0010, v14 = 0x0000_0100
        v("V_SAD_U32", &[("VDST", 42), ("SRC0", V + 12), ("SRC1", V + 13), ("SRC2", V + 14), ("CLAMP", 1)]),
        v("V_SAD_U32", &[("VDST", 43), ("SRC0", V + 12), ("SRC1", V + 13), ("SRC2", V + 14)]),
        v("V_SAD_U32", &[("VDST", 44), ("SRC0", V + 13), ("SRC1", V + 12), ("SRC2", V + 14), ("CLAMP", 1)]),
        // v_sad_u8 / v_sad_hi_u8 / v_sad_u16 / v_msad_u8: v15 = 0xFFFF_FFFF, v16 = 0x0000_0000, v17 = 0x0001_0101
        v("V_SAD_U8", &[("VDST", 45), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 12), ("CLAMP", 1)]),
        v("V_SAD_U8", &[("VDST", 46), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 12)]),
        v("V_SAD_HI_U8", &[("VDST", 47), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 18), ("CLAMP", 1)]),
        v("V_SAD_HI_U8", &[("VDST", 48), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 18)]),
        v("V_SAD_U16", &[("VDST", 49), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 19), ("CLAMP", 1)]),
        v("V_SAD_U16", &[("VDST", 50), ("SRC0", V + 15), ("SRC1", V + 16), ("SRC2", V + 19)]),
        v("V_MSAD_U8", &[("VDST", 51), ("SRC0", V + 15), ("SRC1", V + 17), ("SRC2", V + 12), ("CLAMP", 1)]),
        v("V_MSAD_U8", &[("VDST", 52), ("SRC0", V + 15), ("SRC1", V + 17), ("SRC2", V + 12)]),
        v("V_MSAD_U8", &[("VDST", 53), ("SRC0", V + 15), ("SRC1", V + 17), ("SRC2", 138), ("CLAMP", 1)]),
        // v_cvt_pkrtz_f16_f32 ignores VOP3 OMOD and CLAMP on RDNA2 (ABS/NEG still apply):
        // v54 = -2.0 (f16 0xC000), v55 = 3.0 (0x4200), v56 = 1.5 (0x3E00), v57 = -0.5 (0xB800)
        v("V_CVT_PKRTZ_F16_F32", &[("VDST", 58), ("SRC0", V + 54), ("SRC1", V + 55), ("CLAMP", 1)]),
        v("V_CVT_PKRTZ_F16_F32", &[("VDST", 59), ("SRC0", V + 56), ("SRC1", V + 57), ("OMOD", 1)]),
        // SDWA CLAMP=1 path: v_mul_u32_u24_sdwa & v_mul_i32_i24_sdwa (+ov and -ov)
        ("V_MUL_U32_U24", "VOP2_VOP_SDWA", &[("VDST", 60), ("SRC0", 0xF9), ("VSRC0", 7), ("VSRC1", 7), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6), ("CLAMP", 1)], None),
        ("V_MUL_I32_I24", "VOP2_VOP_SDWA", &[("VDST", 61), ("SRC0", 0xF9), ("VSRC0", 10), ("VSRC1", 10), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6), ("CLAMP", 1)], None),
        ("V_MUL_I32_I24", "VOP2_VOP_SDWA", &[("VDST", 62), ("SRC0", 0xF9), ("VSRC0", 10), ("VSRC1", 11), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6), ("CLAMP", 1)], None),
    ]);
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[0] = 0xFFFF_0003;
        r[1] = 0xFFFF_0004;
        r[2] = 0x0002_0000;
        r[3] = 0x0001_FFFE;
        r[4] = 0x8000_7FFF;
        r[5] = 0x7000_0000;
        r[6] = 0x9000_0000;
        r[7] = 0xABFF_FFFF;
        r[8] = 0x0001_0000;
        r[9] = 0x0000_FFFF;
        r[10] = 0x0080_0000;
        r[11] = 0x007F_FFFF;
        r[12] = 0xFFFF_FFF0;
        r[13] = 0x0000_0010;
        r[14] = 0x0000_0100;
        r[15] = 0xFFFF_FFFF;
        r[16] = 0x0000_0000;
        r[17] = 0x0001_0101;
        r[18] = 0xFF00_0000;
        r[19] = 0xFFFF_0000;
        r[54] = f(-2.0);
        r[55] = f(3.0);
        r[56] = f(1.5);
        r[57] = f(-0.5);
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&code, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[20], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mad_u32_u16 clamp");
            assert_eq!(v[21], 0x0000_0001, "wave{lanes} lane {l} v_mad_u32_u16 wrap");
            assert_eq!(v[22], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mad_u32_u16 exact max");
            assert_eq!(v[23], 0x7FFF_FFFF, "wave{lanes} lane {l} v_mad_i32_i16 clamp +ov");
            assert_eq!(v[24], 0xB000_0000, "wave{lanes} lane {l} v_mad_i32_i16 wrap +ov");
            assert_eq!(v[25], 0x8000_0000, "wave{lanes} lane {l} v_mad_i32_i16 clamp -ov");
            assert_eq!(
                v[26],
                ((-32768i32).wrapping_mul(32767).wrapping_add(-0x7000_0000i32)) as u32,
                "wave{lanes} lane {l} v_mad_i32_i16 wrap -ov"
            );
            assert_eq!(v[27], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mul_u32_u24 clamp");
            assert_eq!(v[28], 0xFE00_0001, "wave{lanes} lane {l} v_mul_u32_u24 wrap");
            assert_eq!(v[29], 0xFFFF_0000, "wave{lanes} lane {l} v_mul_u32_u24 clamp in-range");
            assert_eq!(v[30], 0x7FFF_FFFF, "wave{lanes} lane {l} v_mul_i32_i24 clamp +ov");
            assert_eq!(v[31], 0x0000_0000, "wave{lanes} lane {l} v_mul_i32_i24 wrap +ov");
            assert_eq!(v[32], 0x8000_0000, "wave{lanes} lane {l} v_mul_i32_i24 clamp -ov");
            assert_eq!(v[33], 0x0080_0000, "wave{lanes} lane {l} v_mul_i32_i24 wrap -ov");
            assert_eq!(v[34], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mad_u32_u24 clamp prod ov");
            assert_eq!(v[35], 0xFE00_0006, "wave{lanes} lane {l} v_mad_u32_u24 wrap prod ov");
            assert_eq!(v[36], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mad_u32_u24 clamp add ov");
            assert_eq!(v[37], 0x0001_0000, "wave{lanes} lane {l} v_mad_u32_u24 wrap add ov");
            assert_eq!(v[38], 0x7FFF_FFFF, "wave{lanes} lane {l} v_mad_i32_i24 clamp +ov");
            assert_eq!(
                v[39],
                ((0x007F_FFFFi32).wrapping_mul(0x007F_FFFF).wrapping_add(10)) as u32,
                "wave{lanes} lane {l} v_mad_i32_i24 wrap +ov"
            );
            assert_eq!(v[40], 0x8000_0000, "wave{lanes} lane {l} v_mad_i32_i24 clamp -ov");
            assert_eq!(
                v[41],
                ((-0x0080_0000i32).wrapping_mul(0x007F_FFFF).wrapping_add(-6)) as u32,
                "wave{lanes} lane {l} v_mad_i32_i24 wrap -ov"
            );
            assert_eq!(v[42], 0xFFFF_FFFF, "wave{lanes} lane {l} v_sad_u32 clamp a>b");
            assert_eq!(v[43], 0x0000_00E0, "wave{lanes} lane {l} v_sad_u32 wrap");
            assert_eq!(v[44], 0xFFFF_FFFF, "wave{lanes} lane {l} v_sad_u32 clamp a<b");
            assert_eq!(v[45], 0xFFFF_FFFF, "wave{lanes} lane {l} v_sad_u8 clamp");
            assert_eq!(v[46], 0xFFFF_FFF0u32.wrapping_add(1020), "wave{lanes} lane {l} v_sad_u8 wrap");
            assert_eq!(v[47], 0xFFFF_FFFF, "wave{lanes} lane {l} v_sad_hi_u8 clamp");
            assert_eq!(v[48], 0xFF00_0000u32.wrapping_add(1020 << 16), "wave{lanes} lane {l} v_sad_hi_u8 wrap");
            assert_eq!(v[49], 0xFFFF_FFFF, "wave{lanes} lane {l} v_sad_u16 clamp");
            assert_eq!(v[50], 0xFFFF_0000u32.wrapping_add(131070), "wave{lanes} lane {l} v_sad_u16 wrap");
            assert_eq!(v[51], 0xFFFF_FFFF, "wave{lanes} lane {l} v_msad_u8 clamp");
            assert_eq!(v[52], 0xFFFF_FFF0u32.wrapping_add(3 * 254), "wave{lanes} lane {l} v_msad_u8 wrap");
            assert_eq!(v[53], 3 * 254 + 10, "wave{lanes} lane {l} v_msad_u8 clamp in-range");
            assert_eq!(v[58], 0x4200_C000, "wave{lanes} lane {l} v_cvt_pkrtz_f16_f32 ignores VOP3 clamp");
            assert_eq!(v[59], 0xB800_3E00, "wave{lanes} lane {l} v_cvt_pkrtz_f16_f32 ignores VOP3 omod");
            assert_eq!(v[60], 0xFFFF_FFFF, "wave{lanes} lane {l} v_mul_u32_u24 sdwa clamp");
            assert_eq!(v[61], 0x7FFF_FFFF, "wave{lanes} lane {l} v_mul_i32_i24 sdwa clamp +ov");
            assert_eq!(v[62], 0x8000_0000, "wave{lanes} lane {l} v_mul_i32_i24 sdwa clamp -ov");
        }
    }
}

/// Carry-out/carry-in 32-bit add/sub (VOP3 + SDWA) and 64-bit integer MAD with CLAMP=1 vs CLAMP=0:
/// destination saturates while SDST/VCC still reports the unclamped carry/borrow/sign mask per lane.
#[test]
fn gpu_int_clamp_carry_and_64bit() {
    let vs = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "VOP3_SDST_ENC", f, None);
    let code = prog(&[
        // v_add_co_u32: even lanes (0xFFFF_FFF0 + 0x20) overflow, odd lanes (10 + 20) do not
        vs("V_ADD_CO_U32", &[("VDST", 10), ("SDST", 0), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        vs("V_ADD_CO_U32", &[("VDST", 11), ("SDST", 2), ("SRC0", V + 0), ("SRC1", V + 1)]),
        // v_sub_co_u32 / v_subrev_co_u32: even lanes (0x10 - 0x20) borrow, odd lanes (0x30 - 0x20) do not
        vs("V_SUB_CO_U32", &[("VDST", 12), ("SDST", 4), ("SRC0", V + 2), ("SRC1", V + 1), ("CLAMP", 1)]),
        vs("V_SUB_CO_U32", &[("VDST", 13), ("SDST", 6), ("SRC0", V + 2), ("SRC1", V + 1)]),
        vs("V_SUBREV_CO_U32", &[("VDST", 14), ("SDST", 8), ("SRC0", V + 1), ("SRC1", V + 2), ("CLAMP", 1)]),
        vs("V_SUBREV_CO_U32", &[("VDST", 15), ("SDST", 10), ("SRC0", V + 1), ("SRC1", V + 2)]),
        // carry-in = s[20:21] (all 1s):
        // v_add_co_ci_u32: even (0xFFFF_FFFF + 0 + 1) overflows, odd (0xFFFF_FFFE + 0 + 1) = 0xFFFF_FFFF no carry
        vs("V_ADD_CO_CI_U32", &[("VDST", 16), ("SDST", 12), ("SRC0", V + 3), ("SRC1", 128), ("SRC2", 20), ("CLAMP", 1)]),
        vs("V_ADD_CO_CI_U32", &[("VDST", 17), ("SDST", 14), ("SRC0", V + 3), ("SRC1", 128), ("SRC2", 20)]),
        // v_sub_co_ci_u32 / v_subrev_co_ci_u32: even (0 - 0 - 1) borrows, odd (1 - 0 - 1) = 0 no borrow
        vs("V_SUB_CO_CI_U32", &[("VDST", 18), ("SDST", 16), ("SRC0", V + 4), ("SRC1", 128), ("SRC2", 20), ("CLAMP", 1)]),
        vs("V_SUB_CO_CI_U32", &[("VDST", 19), ("SDST", 18), ("SRC0", V + 4), ("SRC1", 128), ("SRC2", 20)]),
        vs("V_SUBREV_CO_CI_U32", &[("VDST", 20), ("SDST", 22), ("SRC0", 128), ("SRC1", V + 4), ("SRC2", 20), ("CLAMP", 1)]),
        vs("V_SUBREV_CO_CI_U32", &[("VDST", 21), ("SDST", 24), ("SRC0", 128), ("SRC1", V + 4), ("SRC2", 20)]),
        // Corner cases with b = 0xFFFF_FFFF (v44) and alternating carry-in s[34:35] (even=1, odd=0):
        // v_add_co_ci_u32: 0xFFFF_FFFF + 0xFFFF_FFFF + cin (all lanes carry; clamp=0xFFFF_FFFF, wrap=0xFFFF_FFFF/0xFFFF_FFFE)
        vs("V_ADD_CO_CI_U32", &[("VDST", 22), ("SDST", 36), ("SRC0", V + 44), ("SRC1", V + 44), ("SRC2", 34), ("CLAMP", 1)]),
        vs("V_ADD_CO_CI_U32", &[("VDST", 23), ("SDST", 38), ("SRC0", V + 44), ("SRC1", V + 44), ("SRC2", 34)]),
        // v_sub_co_ci_u32 & v_subrev_co_ci_u32 with b+cin 32-bit wraparound (b = 0xFFFF_FFFF, cin = 1 on even):
        // v45 = even: 0, odd: 0xFFFF_FFFF; cin = s[20:21] (all 1s) -> even: 0 - 0xFFFF_FFFF - 1 (-2^32, borrow=1),
        // odd: 0xFFFF_FFFF - 0xFFFF_FFFF - 1 (-1, borrow=1, wrap=0xFFFF_FFFF, clamp=0)
        vs("V_SUB_CO_CI_U32", &[("VDST", 24), ("SDST", 40), ("SRC0", V + 45), ("SRC1", V + 44), ("SRC2", 20), ("CLAMP", 1)]),
        vs("V_SUB_CO_CI_U32", &[("VDST", 25), ("SDST", 42), ("SRC0", V + 45), ("SRC1", V + 44), ("SRC2", 20)]),
        vs("V_SUBREV_CO_CI_U32", &[("VDST", 26), ("SDST", 44), ("SRC0", V + 44), ("SRC1", V + 45), ("SRC2", 20), ("CLAMP", 1)]),
        vs("V_SUBREV_CO_CI_U32", &[("VDST", 27), ("SDST", 46), ("SRC0", V + 44), ("SRC1", V + 45), ("SRC2", 20)]),
        // SDWA CLAMP=1 on V_ADD_CO_CI_U32 & V_SUB_CO_CI_U32 (reads/writes VCC = s[106:107]):
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 20)], None),
        ("V_ADD_CO_CI_U32", "VOP2_VOP_SDWA", &[("VDST", 28), ("SRC0", 0xF9), ("VSRC0", 3), ("VSRC1", 46), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6), ("CLAMP", 1)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 48), ("SSRC0", 106)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 20)], None),
        ("V_SUB_CO_CI_U32", "VOP2_VOP_SDWA", &[("VDST", 29), ("SRC0", 0xF9), ("VSRC0", 4), ("VSRC1", 46), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6), ("CLAMP", 1)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 50), ("SSRC0", 106)], None),
        // v_mad_u64_u32: even lanes overflow, odd lanes hit exact u64::MAX without carry
        vs("V_MAD_U64_U32", &[("VDST", 30), ("SDST", 26), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7), ("CLAMP", 1)]),
        vs("V_MAD_U64_U32", &[("VDST", 32), ("SDST", 28), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7)]),
        // v_mad_i64_i32: 8 cases across lanes (l % 8) covering +ov, -ov, exact bounds, opposite signs
        vs("V_MAD_I64_I32", &[("VDST", 34), ("SDST", 30), ("SRC0", V + 40), ("SRC1", V + 41), ("SRC2", V + 42), ("CLAMP", 1)]),
        vs("V_MAD_I64_I32", &[("VDST", 36), ("SDST", 32), ("SRC0", V + 40), ("SRC1", V + 41), ("SRC2", V + 42)]),
    ]);
    let set64 = |r: &mut [u32], idx: usize, val: u64| {
        r[idx] = val as u32;
        r[idx + 1] = (val >> 32) as u32;
    };
    let get64 = |v: &[u32], idx: usize| v[idx] as u64 | ((v[idx + 1] as u64) << 32);
    let mut st = KernelState::default();
    st.sgpr[20] = u32::MAX;
    st.sgpr[21] = u32::MAX;
    st.sgpr[34] = 0x5555_5555;
    st.sgpr[35] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let even = l & 1 == 0;
        r[0] = if even { 0xFFFF_FFF0 } else { 10 };
        r[1] = 0x20;
        r[2] = if even { 0x10 } else { 0x30 };
        r[3] = if even { 0xFFFF_FFFF } else { 0xFFFF_FFFE };
        r[4] = if even { 0 } else { 1 };
        r[5] = if even { 0xFFFF_FFFF } else { 0x0001_0000 };
        r[6] = if even { 0xFFFF_FFFF } else { 0x0001_0000 };
        set64(r, 7, if even { u64::MAX } else { 0xFFFF_FFFE_FFFF_FFFF });
        r[44] = 0xFFFF_FFFF;
        r[45] = if even { 0 } else { 0xFFFF_FFFF };
        r[46] = 0;
        match l % 8 {
            0 => {
                // (-2^31)*(-2^31) + 2^62 = +2^63 > i64::MAX (bit64 = 0)
                r[40] = i32::MIN as u32;
                r[41] = i32::MIN as u32;
                set64(r, 42, 0x4000_0000_0000_0000);
            }
            1 => {
                // (-2^31)*(2^31-1) + i64::MIN < i64::MIN (bit64 = 1)
                r[40] = i32::MIN as u32;
                r[41] = i32::MAX as u32;
                set64(r, 42, i64::MIN as u64);
            }
            2 => {
                // 1*(-1) + i64::MIN = i64::MIN - 1 < i64::MIN (bit64 = 1)
                r[40] = 1;
                r[41] = (-1i32) as u32;
                set64(r, 42, i64::MIN as u64);
            }
            3 => {
                // (-1)*1 + (i64::MIN + 1) = exact i64::MIN (bit64 = 1)
                r[40] = (-1i32) as u32;
                r[41] = 1;
                set64(r, 42, (i64::MIN + 1) as u64);
            }
            4 => {
                // 1*1 + (i64::MAX - 1) = exact i64::MAX (bit64 = 0)
                r[40] = 1;
                r[41] = 1;
                set64(r, 42, (i64::MAX - 1) as u64);
            }
            5 => {
                // 1*1 + i64::MAX = i64::MAX + 1 > i64::MAX (bit64 = 0)
                r[40] = 1;
                r[41] = 1;
                set64(r, 42, i64::MAX as u64);
            }
            6 => {
                // (-2^31)*(2^31-1) + i64::MAX: opposite signs, > 0 (bit64 = 0)
                r[40] = i32::MIN as u32;
                r[41] = i32::MAX as u32;
                set64(r, 42, i64::MAX as u64);
            }
            _ => {
                // (-2^31)*(-2^31) + i64::MIN: opposite signs, -2^62 < 0 (bit64 = 1)
                r[40] = i32::MIN as u32;
                r[41] = i32::MIN as u32;
                set64(r, 42, i64::MIN as u64);
            }
        }
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&code, &st, wave64);
        let even_mask = (0x5555_5555u32, if wave64 { 0x5555_5555u32 } else { 0 });
        let sdwa_vcc_mask = (0x5555_5555u32, if wave64 { 0x5555_5555u32 } else { u32::MAX });
        let all_mask = (u32::MAX, if wave64 { u32::MAX } else { 0 });
        let mad_i64_mask = (0x8E8E_8E8Eu32, if wave64 { 0x8E8E_8E8Eu32 } else { 0 });
        for (slot, want, label) in [
            (0, even_mask, "v_add_co_u32 clamp sdst"),
            (2, even_mask, "v_add_co_u32 wrap sdst"),
            (4, even_mask, "v_sub_co_u32 clamp sdst"),
            (6, even_mask, "v_sub_co_u32 wrap sdst"),
            (8, even_mask, "v_subrev_co_u32 clamp sdst"),
            (10, even_mask, "v_subrev_co_u32 wrap sdst"),
            (12, even_mask, "v_add_co_ci_u32 clamp sdst"),
            (14, even_mask, "v_add_co_ci_u32 wrap sdst"),
            (16, even_mask, "v_sub_co_ci_u32 clamp sdst"),
            (18, even_mask, "v_sub_co_ci_u32 wrap sdst"),
            (22, even_mask, "v_subrev_co_ci_u32 clamp sdst"),
            (24, even_mask, "v_subrev_co_ci_u32 wrap sdst"),
            (36, all_mask, "v_add_co_ci_u32 max+max+cin clamp sdst"),
            (38, all_mask, "v_add_co_ci_u32 max+max+cin wrap sdst"),
            (40, all_mask, "v_sub_co_ci_u32 b+cin wraparound clamp sdst"),
            (42, all_mask, "v_sub_co_ci_u32 b+cin wraparound wrap sdst"),
            (44, all_mask, "v_subrev_co_ci_u32 b+cin wraparound clamp sdst"),
            (46, all_mask, "v_subrev_co_ci_u32 b+cin wraparound wrap sdst"),
            (48, sdwa_vcc_mask, "v_add_co_ci_u32 sdwa clamp vcc"),
            (50, sdwa_vcc_mask, "v_sub_co_ci_u32 sdwa clamp vcc"),
            (26, even_mask, "v_mad_u64_u32 clamp sdst"),
            (28, even_mask, "v_mad_u64_u32 wrap sdst"),
            (30, mad_i64_mask, "v_mad_i64_i32 clamp sdst (65-bit sign)"),
            (32, mad_i64_mask, "v_mad_i64_i32 wrap sdst (65-bit sign)"),
        ] {
            assert_eq!((r.sgpr[slot], r.sgpr[slot + 1]), want, "wave{lanes} {label}");
        }
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let even = l & 1 == 0;
            assert_eq!(v[10], if even { 0xFFFF_FFFF } else { 10 + 0x20 }, "wave{lanes} lane {l} v_add_co_u32 clamp");
            assert_eq!(v[11], if even { 0x10 } else { 10 + 0x20 }, "wave{lanes} lane {l} v_add_co_u32 wrap");
            assert_eq!(v[12], if even { 0 } else { 0x10 }, "wave{lanes} lane {l} v_sub_co_u32 clamp");
            assert_eq!(v[13], if even { 0xFFFF_FFF0 } else { 0x10 }, "wave{lanes} lane {l} v_sub_co_u32 wrap");
            assert_eq!(v[14], if even { 0 } else { 0x10 }, "wave{lanes} lane {l} v_subrev_co_u32 clamp");
            assert_eq!(v[15], if even { 0xFFFF_FFF0 } else { 0x10 }, "wave{lanes} lane {l} v_subrev_co_u32 wrap");
            assert_eq!(v[16], 0xFFFF_FFFF, "wave{lanes} lane {l} v_add_co_ci_u32 clamp");
            assert_eq!(v[17], if even { 0 } else { 0xFFFF_FFFF }, "wave{lanes} lane {l} v_add_co_ci_u32 wrap");
            assert_eq!(v[18], 0, "wave{lanes} lane {l} v_sub_co_ci_u32 clamp");
            assert_eq!(v[19], if even { 0xFFFF_FFFF } else { 0 }, "wave{lanes} lane {l} v_sub_co_ci_u32 wrap");
            assert_eq!(v[20], 0, "wave{lanes} lane {l} v_subrev_co_ci_u32 clamp");
            assert_eq!(v[21], if even { 0xFFFF_FFFF } else { 0 }, "wave{lanes} lane {l} v_subrev_co_ci_u32 wrap");
            assert_eq!(v[22], 0xFFFF_FFFF, "wave{lanes} lane {l} v_add_co_ci_u32 max+max+cin clamp");
            assert_eq!(v[23], if even { 0xFFFF_FFFF } else { 0xFFFF_FFFE }, "wave{lanes} lane {l} v_add_co_ci_u32 max+max+cin wrap");
            assert_eq!(v[24], 0, "wave{lanes} lane {l} v_sub_co_ci_u32 b+cin wraparound clamp");
            assert_eq!(v[25], if even { 0 } else { 0xFFFF_FFFF }, "wave{lanes} lane {l} v_sub_co_ci_u32 b+cin wraparound wrap");
            assert_eq!(v[26], 0, "wave{lanes} lane {l} v_subrev_co_ci_u32 b+cin wraparound clamp");
            assert_eq!(v[27], if even { 0 } else { 0xFFFF_FFFF }, "wave{lanes} lane {l} v_subrev_co_ci_u32 b+cin wraparound wrap");
            assert_eq!(v[28], 0xFFFF_FFFF, "wave{lanes} lane {l} v_add_co_ci_u32 sdwa clamp");
            assert_eq!(v[29], 0, "wave{lanes} lane {l} v_sub_co_ci_u32 sdwa clamp");
            assert_eq!(get64(v, 30), u64::MAX, "wave{lanes} lane {l} v_mad_u64_u32 clamp");
            assert_eq!(
                get64(v, 32),
                if even { 0xFFFF_FFFE_0000_0000 } else { u64::MAX },
                "wave{lanes} lane {l} v_mad_u64_u32 wrap"
            );
            let (want_clamp, want_wrap) = match l % 8 {
                0 => (i64::MAX as u64, i64::MIN as u64),
                1 => (i64::MIN as u64, 0x4000_0000_8000_0000u64),
                2 => (i64::MIN as u64, i64::MAX as u64),
                3 => (i64::MIN as u64, i64::MIN as u64),
                4 => (i64::MAX as u64, i64::MAX as u64),
                5 => (i64::MAX as u64, i64::MIN as u64),
                6 => (0x4000_0000_7FFF_FFFFu64, 0x4000_0000_7FFF_FFFFu64),
                _ => (0xC000_0000_0000_0000u64, 0xC000_0000_0000_0000u64),
            };
            assert_eq!(get64(v, 34), want_clamp, "wave{lanes} lane {l} v_mad_i64_i32 clamp");
            assert_eq!(get64(v, 36), want_wrap, "wave{lanes} lane {l} v_mad_i64_i32 wrap");
        }
    }
}

/// CPU-only codegen test so GPU-less CI exercises the integer CLAMP=1 HLSL emission paths
/// and verifies V_CVT_PKRTZ_F16_F32 ignores VOP3 OMOD/CLAMP.
#[test]
fn int_clamp_hlsl_codegen() {
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let vs = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "VOP3_SDST_ENC", f, None);
    let code = prog(&[
        v("V_ADD_NC_U16", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        v("V_ADD_NC_I16", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        v("V_MUL_LO_U16", &[("VDST", 12), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        ("V_PK_MUL_LO_U16", "ENC_VOP3P", &[("VDST", 13), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        v("V_MAD_U32_U16", &[("VDST", 14), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)]),
        v("V_MAD_I32_I24", &[("VDST", 15), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)]),
        v("V_SAD_U32", &[("VDST", 16), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)]),
        vs("V_ADD_CO_U32", &[("VDST", 17), ("SDST", 0), ("SRC0", V + 0), ("SRC1", V + 1), ("CLAMP", 1)]),
        vs("V_SUB_CO_CI_U32", &[("VDST", 18), ("SDST", 2), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", 0), ("CLAMP", 1)]),
        vs("V_MAD_U64_U32", &[("VDST", 20), ("SDST", 4), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)]),
        vs("V_MAD_I64_I32", &[("VDST", 22), ("SDST", 6), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)]),
        v("V_CVT_PKRTZ_F16_F32", &[("VDST", 24), ("SRC0", V + 0), ("SRC1", V + 1), ("OMOD", 1), ("CLAMP", 1)]),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("0u - min((uint)(r >> 32), 1u)"), "expected branchless DXC-safe sat_u32");
    assert!(t.hlsl.contains("sat_i16("), "expected sat_i16 helper call");
    assert!(t.hlsl.contains("sat_u32("), "expected sat_u32 helper call");
    assert!(t.hlsl.contains("sat_i32("), "expected sat_i32 helper call");
    assert!(t.hlsl.contains("0xFFFFu)"), "expected 16-bit unsigned saturation");
    assert!(t.hlsl.contains("_s64 ="), "expected 65-bit signed carry-out for v_mad_i64_i32");
    assert!(t.hlsl.contains("f16_rtz(asfloat(v0)) | (f16_rtz(asfloat(v1)) << 16)"), "v_cvt_pkrtz_f16_f32 must ignore VOP3 omod/clamp");
}

#[test]
fn gpu_bench_really_runs() {
    if !gpu_ready() {
        return;
    }
    println!("GPU: {} / DXC: {}", runner::adapter().unwrap(), super::dxc::location().unwrap().display());
    // v1 = lane * 3 must differ per lane – proves the kernel ran on the GPU, per lane
    let code = prog(&[("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 1), ("SRC0", 131), ("VSRC1", 0)], None)]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 0xDEAD;
    }
    // EXEC = only even lanes
    st.sgpr[126] = 0x5555_5555;
    st.sgpr[127] = 0x5555_5555;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let want = if l % 2 == 0 { 3 * l as u32 } else { 0xDEAD };
        assert_eq!(r.vgpr[l][1], want, "lane {l}");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0x5555_5555, 0x5555_5555));
}

#[test]
fn gpu_scalar_loop_and_scc() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),             // 0: s0 = 0
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),             // 4: s1 = 10
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),  // 8: loop: s0 += s1
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None), // 12: s1 -= 1
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),         // 16
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),                 // 20: → 8
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),                // v1 = s0
        // SCC rules
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 255), ("SSRC1", 130)], Some(0xFFFF_FFFF)), // carry
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 128), ("SSRC1", 128)], None),             // 0+0+carry
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 255), ("SSRC1", 129)], Some(0x7FFF_FFFF)), // overflow
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 135), ("SSRC1", 136)], None),           // scc ? 7 : 8
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 255), ("SSRC1", 20)], Some(0x00AB_CD00)), // s20 = (8 << 16) | 8
        ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 255)], Some(0x0000_0100)),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 126)], None), // popcount(exec)
        ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 255), ("SSRC1", 160)], Some(0x0000_0003)), // 3 << 32
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 193), ("SSRC1", 255)], Some(1234)),  // -1 * 1234
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 24), ("SSRC1", 22)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 24), ("SSRC1", 23)], None),
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 26)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 27), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 28)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 29), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 34), ("SSRC0", 24)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 35), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 36), ("SSRC0", 24)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 32)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 33)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 37), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 38), ("SSRC0", 24)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 39), ("SSRC0", 128)], None),       // 0: nothing differs from the sign
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 40), ("SSRC0", 193)], None),       // -1: all bits are the sign
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 41), ("SSRC0", 44)], None),    // high half 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 42), ("SSRC0", 46)], None),    // negative, ~high = 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 43), ("SSRC0", 128)], None),   // 0 → all bits equal the sign
    ]);
    let mut st = KernelState::default();
    st.sgpr[20] = (8 << 16) | 8;
    st.sgpr[22] = (20 << 16) | 12;
    st.sgpr[23] = (4 << 16) | 60;
    st.sgpr[24] = 0x7654_3210;
    st.sgpr[25] = 0xFEDC_BA98;
    st.sgpr[26] = 63;
    st.sgpr[28] = 0;
    st.sgpr[30] = 1 << 4;
    st.sgpr[31] = 0;
    st.sgpr[32] = 36;
    st.sgpr[33] = 4;
    st.sgpr[44] = 0x8000;                                                        // s[44:45] = 0x0000_0000_0000_8000
    st.sgpr[45] = 0;
    st.sgpr[46] = 0x8000_0000;                                                   // s[46:47] = 0xFFFF_FFFF_8000_0000
    st.sgpr[47] = 0xFFFF_FFFF;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!(r.sgpr[0], 55, "loop");
    assert!(r.vgpr.iter().all(|v| v[1] == 55));
    assert_eq!(r.sgpr[2], 1, "s_add_u32 wraps");
    assert_eq!(r.sgpr[3], 1, "s_addc_u32 uses the carry");
    assert_eq!(r.sgpr[4], 0x8000_0000, "s_add_i32");
    assert_eq!(r.sgpr[5], 7, "s_cselect after signed overflow → scc=1");
    assert_eq!(r.sgpr[6], 0xCD, "s_bfe_u32");
    assert_eq!(r.sgpr[7], 8, "s_ff1");
    assert_eq!(r.sgpr[8], 64, "s_bcnt1 exec");
    assert_eq!((r.sgpr[10], r.sgpr[11]), (0, 3), "s_lshl_b64");
    assert_eq!(r.sgpr[12] as i32, -1234, "s_mul_i32");
    assert_eq!((r.sgpr[14], r.sgpr[15]), (0x0007_6543, 0), "s_bfe_u64");
    assert_eq!((r.sgpr[16], r.sgpr[17]), (u32::MAX, u32::MAX), "s_bfe_i64 sign extension");
    assert_eq!((r.sgpr[27], r.sgpr[29]), (7, 7), "s_bitcmp1/0_b64");
    assert_eq!((r.sgpr[34], r.sgpr[35], r.sgpr[36]), (32, 0, 0), "s_bcnt0/ff0/flbit 64");
    assert_eq!((r.sgpr[30], r.sgpr[31]), (0, 1 << 4), "s_bitset1/0_b64");
    assert_eq!((r.sgpr[37], r.sgpr[38]), (1, 7), "signed scalar leading-bit scans");
    assert_eq!((r.sgpr[39], r.sgpr[40]), (u32::MAX, u32::MAX), "flbit: nothing differs from the sign");
    assert_eq!((r.sgpr[41], r.sgpr[42], r.sgpr[43]), (48, 33, u32::MAX), "flbit_i64 high-then-low scan");
}

#[test]
fn gpu_divergence_through_exec() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None), // vcc = 32 > lane
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)], None), // lanes 0..31: 1
        ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 130)], None), // lanes 32..63: 2
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128), ("VSRC1", 3)], None), // vcc ? v3 : 0
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V), ("SRC1", 255)], Some(45)),
        // nobody matches: exec = 0 → the block is skipped
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(1000)), // 36: (literal: 8 bytes)
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),                    // 44 → 56
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 192)], None), // 48 (skipped)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 129)], None),             // 52 (skipped)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),           // 56: exec = -1
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 8), ("SDST", 20), ("SRC0", V + 9), ("SRC1", 255)], Some(0x8000_0000)),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[3] = 7 + l as u32;
        r[6] = 0x66;
        r[9] = if l % 4 == 0 { 0x8000_0000 } else { 1 };
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], if l < 32 { 1 } else { 2 }, "lane {l} masked writes");
        assert_eq!(r.vgpr[l][2], if l < 32 { 7 + l as u32 } else { 0 }, "lane {l} cndmask");
        assert_eq!(r.vgpr[l][6], 0x66, "lane {l}: skipped block must not run");
    }
    assert_eq!((r.sgpr[2], r.sgpr[3]), (u32::MAX, u32::MAX), "saved exec");
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, 0), "vcc from v_cmp");
    assert_eq!(r.sgpr[4], 7, "readfirstlane");
    assert_eq!(r.sgpr[5], 45, "readlane 45");
    assert_eq!(r.sgpr[7], 0, "scalar code in the skipped block");
    assert_eq!((r.sgpr[126], r.sgpr[127]), (u32::MAX, u32::MAX));
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x1111_1111, 0x1111_1111), "carry mask of v_add_co_u32");
}

#[test]
fn gpu_memory_and_exports() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 6), ("OFFSET", 4), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("OFFSET", 0x400), ("SOFFSET", 125)], None), // out of range
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("OFFEN", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 8)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 3), ("VSRC1", 4), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1)], None),
        ("V_CVT_PKRTZ_F16_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 242), ("VSRC1", 6)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0x3), ("VSRC0", 5), ("VSRC1", 5), ("COMPR", 1), ("DONE", 1), ("VM", 1)], None),
    ]);
    let mut mem = vec![0u32; 256];
    for (i, m) in mem.iter_mut().enumerate() {
        *m = 0x1000 + i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0; // flat pointer
    st.sgpr[12] = 0x40; // V#: base 0x40, stride 0, 0x100 bytes
    st.sgpr[13] = 0;
    st.sgpr[14] = 0x100;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 4 * l as u32;
        r[6] = f(0.5);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!((r.sgpr[8], r.sgpr[9]), (0x1004, 0x1005), "s_load_dwordx2");
    assert_eq!(r.sgpr[10], 0x1000 + 0x44 / 4, "s_buffer_load_dword");
    assert_eq!(r.sgpr[11], 0, "s_buffer_load out of range → 0");
    for l in 0..64 {
        let want = if 8 + 4 * l < 0x100 { 0x1000 + (0x40 + 8 + 4 * l as u32) / 4 } else { 0 };
        assert_eq!(r.vgpr[l][3], want, "lane {l} buffer_load_dword offen");
        assert_eq!(r.export(12, l, 0), want, "lane {l} pos0.x");
        assert_eq!(r.export(12, l, 1), 0x1004, "lane {l} pos0.y");
        assert_eq!(r.export(12, l, 2), 4 * l as u32, "lane {l} pos0.z");
        assert_eq!((r.export(0, l, 0), r.export(0, l, 1)), (0x3C00, 0x3800), "lane {l} mrt0 compr (1.0, 0.5)");
    }
}

/// Multi-dword loads as the emitter groups them (four dwords per memory access, bounds checks
/// without branches): an unaligned `s_load`, `s_buffer_load`s that straddle num_records, and
/// vector buffer loads with lanes past the end and inactive lanes (which keep their registers).
#[test]
fn gpu_grouped_loads_bounds_and_inactive_lanes() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x22), ("SOFFSET", 125)], None), // low bits ignored
        ("S_BUFFER_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)], None), // 5 of 8 in range
        ("S_BUFFER_LOAD_DWORDX16", "ENC_SMEM", &[("SDATA", 32), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 48), ("SBASE", 2), ("OFFSET", 16), ("SOFFSET", 125)], None), // 1 of 2
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 48), ("VSRC1", 1)], None), // lanes 0..47
        ("BUFFER_LOAD_DWORDX3", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_DWORDX2", "ENC_MUBUF", &[("VDATA", 13), ("VADDR", 2), ("SRSRC", 13), ("SOFFSET", 128), ("OFFEN", 1)], None),
    ]);
    let mut mem = vec![0u32; 1024];
    for (i, m) in mem.iter_mut().enumerate() {
        *m = 0x1000 + i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0; // flat pointer 0
    st.sgpr[4..8].copy_from_slice(&[0x80, 0, 20, 0]); // V#: base 0x80, 20 bytes
    st.sgpr[8..12].copy_from_slice(&[0x100, 0, 0x100, 0]); // V#: base 0x100, 0x100 bytes
    st.sgpr[12..16].copy_from_slice(&[0x100, 12 << 16, 40, 0]); // V#: base 0x100, stride 12, 40 records
    st.sgpr[52..56].copy_from_slice(&[0x400, 0, 0x104, 0]); // V#: base 0x400, 0x104 bytes
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = l as u32;
        r[2] = 8 * l as u32;
        for k in 10..15 {
            r[k] = 0xDEAD;
        }
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for i in 0..8 {
        assert_eq!(r.sgpr[16 + i], 0x1008 + i as u32, "s_load_dwordx8 dword {i}");
        assert_eq!(r.sgpr[24 + i], if i < 5 { 0x1020 + i as u32 } else { 0 }, "s_buffer_load_dwordx8 dword {i}");
    }
    for i in 0..16 {
        assert_eq!(r.sgpr[32 + i], 0x1040 + i as u32, "s_buffer_load_dwordx16 dword {i}");
    }
    assert_eq!((r.sgpr[48], r.sgpr[49]), (0x1024, 0), "s_buffer_load_dwordx2 across num_records");
    for l in 0..64usize {
        for k in 0..3usize {
            let want = if l >= 48 { 0xDEAD } else if l < 40 { 0x1040 + 3 * l as u32 + k as u32 } else { 0 };
            assert_eq!(r.vgpr[l][10 + k], want, "lane {l} buffer_load_dwordx3 (idxen) dword {k}");
        }
        for k in 0..2usize {
            let off = 8 * l + 4 * k;
            let want = if l >= 48 { 0xDEAD } else if off + 4 <= 0x104 { 0x1100 + (off / 4) as u32 } else { 0 };
            assert_eq!(r.vgpr[l][13 + k], want, "lane {l} buffer_load_dwordx2 (offen) dword {k}");
        }
    }
}

/// A real console vertex shader from the retail pack, run on the PC GPU
/// (needs XCL_SAMPLE_RPF=…\shaders.rpf).
#[test]
fn gpu_retail_vs_blit_gen_amb_cube() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else {
        println!("XCL_SAMPLE_RPF not set – skipping");
        return;
    };
    if !gpu_ready() {
        return;
    }
    let data = std::fs::read(&path).unwrap();
    let rec = crate::gpu::pack::scan(&data, 0x40).into_iter().find(|r| r.name == "VS_BlitGenAmbCube_0cfdb358_Wrapped").expect("record");
    let blob = crate::gpu::pack::inflate(&data, &rec).unwrap();
    let sh = crate::gpu::xbox::parse(&blob).unwrap();
    let fs = &sh.fssm[0];
    let p = fs.programs[0];
    let code = &fs.data[p.offset..p.offset + p.len];
    let t = translate(code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported, t.hlsl);
    println!("{}", t.hlsl);
    // the fetch shader would load POSITION into v4..; give every lane its own vertex
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[4] = f((l % 8) as f32 / 7.0);
        r[5] = f((l / 8) as f32 / 7.0);
    }
    let (r, _) = runner::run_program(code, &st, &[]).unwrap();
    for l in 0..64 {
        let x = (l % 8) as f32 / 7.0;
        let y = (l / 8) as f32 / 7.0;
        let pos: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(12, l, c))).collect();
        assert_eq!(pos, vec![x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0], "lane {l}: SV_Position");
        let p0: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(32, l, c))).collect();
        assert_eq!(p0, vec![0.0, 0.0, 0.0, 1.0], "lane {l}: param0");
    }
}

#[test]
fn gpu_sdwa_dpp_and_cross_lane() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v10 = (float)(v1 byte 2)      SRC0=0xF9 (SDWA), VSRC0=1, SRC0_SEL=BYTE_2, DST_SEL=DWORD
        ("V_CVT_F32_U32", "VOP1_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 1), ("SRC0_SEL", 2), ("DST_SEL", 6)], None),
        // v11 = sext(v1 word1) + v2 written into word 0 of v11, upper word preserved
        ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 2), ("SRC0_SEL", 5), ("SRC0_SEXT", 1), ("SRC1_SEL", 6), ("DST_SEL", 4), ("DST_UNUSED", 2)], None),
        // v12 = v0 of lane (lane ^ 1 within quad): quad_perm [1,0,3,2] = 0b10_11_00_01 = 0xB1
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 12), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
        // v13 = v0 of lane-1 within row; lane 0 of each row has no source → 0 (bound_ctrl)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 13), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)], None),
        // v14 = row_mirror, rows 1 and 3 only (row_mask 0b1010)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 14), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xA), ("BANK_MASK", 0xF)], None),
        // v15 = v0 of lane 7 - (lane & 7) in each group of 8 (DPP8)
        ("V_MOV_B32", "VOP1_VOP_DPP8", &[("VDST", 15), ("SRC0", 0xE9), ("VSRC0", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)], None),
        // v16 = swizzle: lane ^ 4 (BitMode: and 0x1F, or 0, xor 4)
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 16), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 4 << 2)], None),
        // v17 = permlanex16 with identity selects → lane ^ 16
        ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V), ("SRC1", 255), ("SRC2", 3)], Some(0x7654_3210)),
        // v18 = fma_mix(f16 lo of v3, f16 hi of v3, f32 v4) = 1.5 * 2.0 + 0.25
        ("V_FMA_MIX_F32", "ENC_VOP3P", &[("VDST", 18), ("SRC0", V + 3), ("SRC1", V + 3), ("SRC2", V + 4), ("OP_SEL", 0b010), ("OP_SEL_HI", 0b11), ("OP_SEL_HI_2", 0)], None),
        // v19 = cube face of (v5, v6, v7)
        ("V_CUBEID_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[3] = 0xFEDC_BA98; // permlanex16 selects for lanes 8..15 of a row
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
        r[1] = 0x8001_0000 | ((l as u32) << 16 & 0x00F0_0000) ;
        r[2] = 10;
        r[11] = 0xAAAA_5555;
        r[14] = 0xDEAD;
        r[3] = 0x4000_3E00; // hi = 2.0h, lo = 1.5h
        r[4] = f(0.25);
        let axis = l % 3;
        r[5] = f(if axis == 0 { -3.0 } else { 0.5 });
        r[6] = f(if axis == 1 { 3.0 } else { 0.5 });
        r[7] = f(if axis == 2 { -3.0 } else { 0.5 });
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let v1 = 0x8001_0000u32 | ((l as u32) << 16 & 0x00F0_0000);
        assert_eq!(f32::from_bits(v[10]), ((v1 >> 16) & 0xFF) as f32, "lane {l} sdwa byte2");
        let w1 = ((v1 >> 16) as u16 as i16 as i32) as u32;
        assert_eq!(v[11], 0xAAAA_0000 | (w1.wrapping_add(10) & 0xFFFF), "lane {l} sdwa word1 sext + preserve");
        assert_eq!(v[12], 100 + (l ^ 1) as u32, "lane {l} quad_perm");
        assert_eq!(v[13], if l % 16 == 0 { 0 } else { 100 + l as u32 - 1 }, "lane {l} row_shr:1 bound_ctrl");
        let row = l / 16;
        assert_eq!(v[14], if row % 2 == 1 { 100 + (row * 16 + 15 - l % 16) as u32 } else { 0xDEAD }, "lane {l} row_mirror row_mask");
        assert_eq!(v[15], 100 + ((l & !7) | (7 - (l & 7))) as u32, "lane {l} dpp8");
        assert_eq!(v[16], 100 + (l ^ 4) as u32, "lane {l} ds_swizzle");
        let n = if l % 16 < 8 { (0x7654_3210u32 >> (4 * (l % 8))) & 15 } else { (0xFEDC_BA98u32 >> (4 * (l % 8))) & 15 };
        assert_eq!(v[17], 100 + ((l & !31) | ((l & 16) ^ 16) | n as usize) as u32, "lane {l} permlanex16");
        assert_eq!(f32::from_bits(v[18]), 1.5 * 2.0 + 0.25, "lane {l} fma_mix");
        let face = match l % 3 {
            0 => 1.0,
            1 => 2.0,
            _ => 5.0,
        };
        assert_eq!(f32::from_bits(v[19]), face, "lane {l} cubeid");
    }
}

#[test]
fn buffer_format_table_matches_llvm() {
    // every BUF_FMT_* in LLVM's gfx10 MTBUF tests must decode to the same name
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/llvm-amdgpu");
    let mut checked = 0;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        for line in std::fs::read_to_string(e.path()).unwrap_or_default().lines() {
            let Some(i) = line.find("format:[BUF_FMT_") else { continue };
            if !line.starts_with("# GFX10") {
                continue;
            }
            let name = &line[i + 16..line[i..].find(']').unwrap() + i];
            let Some(enc) = line.split("encoding: [").nth(1) else { continue };
            let b: Vec<u8> = enc.trim_end_matches(']').split(',').filter_map(|x| u8::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()).collect();
            let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            assert_eq!(super::emit::BUF_FORMATS[((w >> 19) & 0x7F) as usize], name, "{line}");
            checked += 1;
        }
    }
    assert!(checked >= 10, "{checked}");
}

#[test]
fn buffer_format_d16_packs_and_unpacks_halves() {
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_D16_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 12), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("TBUFFER_STORE_FORMAT_D16_XYZ", "ENC_MTBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 71)], None),
        ("BUFFER_STORE_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 16), ("SRSRC", 3), ("SOFFSET", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("v10 = t1_ok ? (select(xcl_f15_int") && t.hlsl.contains("v11 = t1_ok ? (select(xcl_f15_int"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (v12 & 0xFFFFu) | (t2_ok ? (select(xcl_f15_int") && t.hlsl.contains("<< 16) : 0u)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint t3_ty = (t3_info >> 3) & 7u") && t.hlsl.contains("f16tof32(((v14) & 0xFFFFu))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("f16tof32(((v16 >> 16) & 0xFFFFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// `fmt_comp` reads a component from one dword: no present component of any buffer format
/// crosses a dword boundary.
#[test]
fn buffer_format_components_stay_in_one_dword() {
    for f in 0..super::emit::BUF_FORMATS.len() {
        for c in 0..4 {
            let e = super::emit::buf_format_comp(super::emit::buf_format_info(f), c);
            assert!(e[0] >> 26 & 1 != 0 || (e[0] & 31) + e[2].count_ones() <= 32, "format {f} component {c}: {e:08X?}");
        }
    }
}

/// The buffer format decode of `emit::PRELUDE_MEM` (`fmt_comp` on the `FMT_COMP` entries,
/// `fmt_slot` / `fmt_const` / `fmt_pick`) against the form it replaced – all four components decoded from the format's
/// `FMT_INFO` word, then `DST_SEL` picked from them – bit for bit on the GPU: every format,
/// every `DST_SEL` and every component, for zero, all-ones, sign-bit and hashed raw data.
#[test]
fn gpu_buffer_format_decode_matches_the_four_component_form() {
    if !gpu_ready() {
        return;
    }
    let prelude = super::emit::PRELUDE_MEM;
    let item = |sig: &str| -> String {
        let at = prelude.find(sig).unwrap_or_else(|| panic!("{sig} not in PRELUDE_MEM"));
        let rest = &prelude[at..];
        let end = if rest.lines().next().unwrap().trim_end().ends_with('}') { rest.find('\n').unwrap() + 1 } else { rest.find("\n}\n").unwrap() + 3 };
        rest[..end].to_string()
    };
    let n = super::emit::BUF_FORMATS.len();
    let infos: Vec<String> = (0..n).map(|i| format!("{}u", super::emit::buf_format_info(i))).collect();
    let comps: Vec<String> = (0..n).flat_map(|i| (0..4).map(move |c| super::emit::buf_format_comp(super::emit::buf_format_info(i), c))).map(|e| format!("uint4({}u, {}u, {}u, {}u)", e[0], e[1], e[2], e[3])).collect();
    let old = r#"
float uf_small(uint v, uint mb) {
    uint e = v >> mb; uint m = v & ((1u << mb) - 1u); float fm = (float)m / (float)(1u << mb);
    float special = select(m != 0u, asfloat(0x7FC00000u), asfloat(0x7F800000u));
    return select(e == 31u, special, select(e == 0u, fm * exp2(-14.0f), exp2((float)e - 15.0f) * (1.0f + fm)));
}
uint fmt_comp_old(uint4 r, uint info, uint c) {
    uint n = info & 7u; uint ty = (info >> 3) & 7u;
    bool isint = or(ty == 4u, ty == 5u);
    uint off = select(c > 0u, (info >> 8) & 63u, 0u) + select(c > 1u, (info >> 14) & 63u, 0u) + select(c > 2u, (info >> 20) & 63u, 0u);
    uint w = (info >> ((8u + 6u * c) & 31u)) & 63u;
    uint v = bits_at(r, off, w);
    uint sh = (32u - w) & 31u;
    int sv = select(w >= 32u, (int)v, ((int)(v << sh)) >> sh);
    float maxu = select(w >= 32u, 4294967295.0f, (float)((1u << (w & 31u)) - 1u));
    float maxs = select(w >= 32u, 2147483647.0f, (float)((1u << ((w - 1u) & 31u)) - 1u));
    float fv = (float)v; float fsv = (float)sv;
    uint f_un = asuint(select(v == 0u, 0.0f, select(fv >= maxu, 1.0f, fv / maxu)));
    uint f_sn = asuint(select(sv == 0, 0.0f, select(fsv >= maxs, 1.0f, max(fsv / maxs, -1.0f))));
    uint flt = select(w == 32u, v, select(w == 16u, asuint(f16tof32(v)), asuint(uf_small(v, select(w == 11u, 6u, 5u)))));
    uint res = select(ty == 0u, f_un, select(ty == 1u, f_sn, select(ty == 2u, asuint(fv), select(ty == 3u, asuint(fsv), select(ty == 4u, v, select(ty == 5u, (uint)sv, flt))))));
    return select(c >= n, select(c == 3u, select(isint, 1u, 0x3F800000u), 0u), res);
}
uint dst_sel_old(uint4 c, uint sel, bool isint) { return select(sel == 0u, 0u, select(sel == 1u, select(isint, 1u, 0x3F800000u), c[(sel - 4u) & 3u])); }
uint hash(uint x) { x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x; }
[numthreads(64, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    uint bad = 0u; uint last = 0u;
    [loop] for (uint f = 0u; f < NFMT; f++) {
        uint info = FMT_INFO[f]; uint ty = (info >> 3) & 7u; bool isint = or(ty == 4u, ty == 5u);
        [loop] for (uint k = 0u; k < 80u; k++) {
            uint s = (gi * NFMT + f) * 16u + k;
            uint4 r = uint4(hash(s), hash(s ^ 0x9E3779B9u), hash(s + 0x632BE5ABu), hash(s * 3u + 1u));
            r = select(k == 0u, uint4(0u, 0u, 0u, 0u), select(k == 1u, uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu), select(k == 2u, uint4(0x80008000u, 0x80000000u, 0x80808080u, 0x7FFF7FFFu), r)));
            // every 11- and 10-bit value (64 threads x 32) in each packed position: 11-11-10
            // from bit 0, then 10-11-11
            uint e = gi * 32u + ((k - 16u) & 31u);
            uint packed = select(k < 48u, e | (e << 11) | ((e & 0x3FFu) << 22), (e & 0x3FFu) | (e << 10) | (e << 21));
            r = select(k >= 16u, uint4(packed, e | (e << 16), e, e), r);
            uint4 c = uint4(fmt_comp_old(r, info, 0u), fmt_comp_old(r, info, 1u), fmt_comp_old(r, info, 2u), fmt_comp_old(r, info, 3u));
            [loop] for (uint q = 0u; q < 4u; q++) {
                if (fmt_comp(r, FMT_COMP[f * 4u + q]) != c[q]) { bad++; last = (f << 16) | (q << 8) | k; }
            }
            [loop] for (uint sel = 0u; sel < 8u; sel++) {
                if (fmt_pick(r, fmt_slot(f * 4u, sel), sel >= 2u, fmt_const(sel, isint), false) != dst_sel_old(c, sel, isint)) { bad++; last = (f << 16) | ((8u + sel) << 8) | k; }
                if (fmt_pick(r, fmt_slot(f * 4u, sel), sel >= 2u, fmt_const(sel, isint), true) != select(sel <= 1u, dst_sel_old(c, sel, isint), 0u)) { bad++; last = (f << 16) | ((8u + sel) << 8) | k; }
            }
        }
    }
    uint o;
    res.InterlockedAdd(0u, bad, o);
    res.InterlockedMax(4u, last, o);
}
"#;
    let hlsl = format!(
        "RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n#define NFMT {n}u\nstatic const uint FMT_INFO[{n}] = {{ {} }};\nstatic const uint4 FMT_COMP[{}] = {{ {} }};\n{}{}{}{}{}{old}",
        infos.join(", "),
        comps.len(),
        comps.join(", "),
        item("uint bits_at("),
        item("uint fmt_comp("),
        item("uint4 fmt_slot("),
        item("uint fmt_const("),
        item("uint fmt_pick(")
    );
    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let out = runner::run_with_window_view(&dxil, 1 << 16).unwrap();
    assert_eq!(out[0], 0, "decodes that differ; the last: format {}, case {}, data {}", out[1] >> 16, (out[1] >> 8) & 255, out[1] & 255);
}

#[test]
fn gpu_tbuffer_load_identity_dst_sel() {
    if !gpu_ready() {
        return;
    }
    let bgra = 6 | 5 << 3 | 4 << 6 | 7 << 9;
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("TBUFFER_LOAD_FORMAT_XYZW", "ENC_MTBUF", &[("VDATA", 14), ("SRSRC", 3), ("SOFFSET", 128), ("FORMAT", 60)], None),
        ("TBUFFER_LOAD_FORMAT_XYZW", "ENC_MTBUF", &[("VDATA", 18), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 29)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XYZW", "ENC_MTBUF", &[("VDATA", 22), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 29)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XY", "ENC_MTBUF", &[("VDATA", 24), ("SRSRC", 3), ("SOFFSET", 128), ("FORMAT", 60)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 25), ("SRSRC", 4), ("SOFFSET", 128)], None),
    ]);
    let mut mem = vec![0u32; 128];
    mem[0x100 / 4] = u32::from_le_bytes([1, 2, 3, 4]);
    mem[0x110 / 4] = 0x3C00 | 0xC000 << 16;
    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x100, 0, 0x100, bgra | 60 << 12]);
    st.sgpr[16..20].copy_from_slice(&[0x110, 0, 0x100, 11 << 12]);
    for r in st.vgpr.iter_mut() {
        r[25] = 0xDEAD_BEEF;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(v[10..14], [3, 2, 1, 4], "lane {l} buffer_load_format applies the V# DST_SEL");
        assert_eq!(v[14..18], [1, 2, 3, 4], "lane {l} tbuffer xyzw: identity DST_SEL");
        assert_eq!(v[18..22], [f(1.0), f(-2.0), 0, f(1.0)], "lane {l} tbuffer 16_16_FLOAT: V# SEL_0 ignored, z = 0, w = 1");
        assert_eq!(v[22..24], [0xC000_3C00, 0x3C00_0000], "lane {l} tbuffer d16 xyzw");
        assert_eq!(v[24], 1 | 2 << 16, "lane {l} tbuffer d16 xy: identity DST_SEL");
        assert_eq!(v[25], 0, "lane {l} buffer_load_format 16_UINT (0x3C00) with SEL_0 everywhere");
    }
}

/// Format loads through one descriptor whose dword 3 (format, DST_SEL) is rewritten between
/// them – by a move, `s_movreld` through M0, a bit set and a lane read – each decode the
/// format the descriptor holds then (`Em::fmt_regs`: decoded at the start and after writes).
#[test]
fn gpu_buffer_format_loads_follow_descriptor_rewrites() {
    if !gpu_ready() {
        return;
    }
    let sel = |x: u32, y: u32, z: u32, w: u32| x | y << 3 | z << 6 | w << 9;
    let (uint8x4, uint32, float32) = (60u32, 20u32, 22u32);
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 15), ("SSRC0", 255)], Some(sel(7, 6, 5, 4) | uint8x4 << 12)),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 255)], Some(sel(4, 5, 0, 1) | uint32 << 12)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 20)], None), // s15 = s20
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 15), ("SSRC0", 128 + 14)], None), // format 20 -> 16 (8_8_USCALED)
        ("BUFFER_LOAD_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V)], None),
        ("BUFFER_LOAD_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 24), ("SRSRC", 3), ("SOFFSET", 128)], None),
    ]);
    let mut mem = vec![0u32; 128];
    mem[0x100 / 4] = u32::from_le_bytes([1, 2, 3, 4]);
    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x100, 0, 0x100, sel(4, 5, 6, 7) | uint8x4 << 12]);
    for r in st.vgpr.iter_mut() {
        r[0] = sel(5, 4, 0, 0) | float32 << 12;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(v[10..14], [1, 2, 3, 4], "lane {l}: 8_8_8_8_UINT, identity DST_SEL");
        assert_eq!(v[14..18], [4, 3, 2, 1], "lane {l}: after s_mov, reversed DST_SEL");
        assert_eq!(v[18..22], [0x0403_0201, 0, 0, 1], "lane {l}: after s_movreld, 32_UINT with y absent, z = 0, w = 1");
        assert_eq!(v[22..24], [f(1.0), f(2.0)], "lane {l}: after s_bitset0, 8_8_USCALED");
        assert_eq!(v[24..26], [0, 0x0403_0201], "lane {l}: after v_readfirstlane, 32_FLOAT with x from the absent y");
    }
}

#[test]
fn gpu_two_byte_typed_loads_use_effective_address_alignment() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 0), ("OFFEN", 1), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("TBUFFER_LOAD_FORMAT_X", "ENC_MTBUF", &[("VDATA", 11), ("VADDR", 0), ("OFFEN", 1), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 11)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 12), ("VADDR", 0), ("OFFEN", 1), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 13), ("VADDR", 0), ("OFFEN", 1), ("SRSRC", 5), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 14), ("VADDR", 0), ("OFFEN", 1), ("SRSRC", 3), ("SOFFSET", 128)], None),
    ]);
    let mut mem = vec![0u32; 128];
    mem[0x100 / 4] = 0x7856_3412;
    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x101, 0, 8, (11 << 12) | 4]);
    st.sgpr[16..20].copy_from_slice(&[0x100, 0, 8, (5 << 12) | 4]);
    st.sgpr[20..24].copy_from_slice(&[0x101, 0, 8, (11 << 12) | 1]);
    for (lane, regs) in st.vgpr.iter_mut().enumerate() {
        regs[0] = (lane & 1) as u32;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for (lane, regs) in r.vgpr.iter().enumerate() {
        let expected = if lane & 1 == 0 { [0, 0x3412, 0x12, 1, 0] } else { [0x7856, 0, 0x34, 1, 0x7856] };
        assert_eq!(regs[10..15], expected, "lane {lane}");
    }
}

#[test]
fn gpu_typed_buffer_loads_and_stores() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let bgra = 6 | 5 << 3 | 4 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32, sel: u32| [base, 0, 0x100, sel | fmt << 12];
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),             // A
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128)], None),             // B
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 5), ("SOFFSET", 128)], None),             // C
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None), // D (unaligned)
        ("BUFFER_LOAD_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 23), ("SRSRC", 7), ("SOFFSET", 128)], None),              // E
        ("TBUFFER_LOAD_FORMAT_XY", "ENC_MTBUF", &[("VDATA", 26), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None), // F
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 28), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 0x200)], None), // out of range
        ("BUFFER_LOAD_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 31), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 32), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XY", "ENC_MTBUF", &[("VDATA", 33), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None),
        ("BUFFER_STORE_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 34), ("SRSRC", 10), ("SOFFSET", 128)], None),
        ("V_CVT_PK_U8_F32", "ENC_VOP3", &[("VDST", 29), ("SRC0", V + 30), ("SRC1", 130), ("SRC2", 255)], Some(0x1122_3344)),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 1), ("SRSRC", 9), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("S_BITSET1_B32", "ENC_SOP1", &[("SDST", 50), ("SSRC0", 133)], None),
    ]);
    let mut mem = vec![0u32; 512];
    mem[0x100 / 4] = u32::from_le_bytes([0x00, 0x80, 0xFF, 0x40]); // A: 8_8_8_8_UNORM
    mem[0x110 / 4] = 0x3C00 | 0xC000 << 16; // B: 16_16_FLOAT (1.0, -2.0)
    mem[0x114 / 4] = 1023 | 512 << 20 | 3 << 30; // C: 2_10_10_10_UNORM
    mem[0x118 / 4] = 0x8001 << 16; // D: 16_SNORM at byte 0x11A
    mem[0x11C / 4] = 0x3C0 | 0x400 << 11 | 0x1C0 << 22; // E: 10_11_11_FLOAT (1.0, 2.0, 0.5)
    mem[0x120 / 4] = f(1.5);
    mem[0x124 / 4] = f(-3.25); // F: 32_32_FLOAT
    let mut st = KernelState::default();
    for (i, v) in [
        vs(0x100, 56, bgra),
        vs(0x110, 29, ident),
        vs(0x114, 50, ident),
        vs(0x118, 8, ident),
        vs(0x11C, 36, ident),
        vs(0x120, 0, ident),
        [0x400, 0, 0x100, 0],
    ]
    .iter()
    .enumerate()
    {
        st.sgpr[12 + 4 * i..16 + 4 * i].copy_from_slice(v);
    }
    st.sgpr[40..44].copy_from_slice(&vs(0x130, 29, ident));
    st.sgpr[50] = 1;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 0xC0DE_0000 | l as u32;
        r[1] = 4 * l as u32;
        r[30] = f(200.6);
        r[34] = 0xB800_4000; // packed f16: +2.0, -0.5
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let fl = |x: u32| f32::from_bits(x);
    // normalised mid values: within 2 ULP (DXC divides approximately); endpoints exact
    let near = |got: u32, want: f32| (got as i64 - want.to_bits() as i64).abs() <= 2;
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert!(
            v[10] == f(1.0) && near(v[11], 128.0 / 255.0) && v[12] == 0 && near(v[13], 64.0 / 255.0),
            "lane {l} A unorm8 bgra {:?}",
            [fl(v[10]), fl(v[11]), fl(v[12]), fl(v[13])]
        );
        assert_eq!([fl(v[14]), fl(v[15]), fl(v[16]), fl(v[17])], [1.0, -2.0, 0.0, 1.0], "lane {l} B 16_16_float");
        assert!(
            v[18] == f(1.0) && v[19] == 0 && near(v[20], 512.0 / 1023.0) && v[21] == f(1.0),
            "lane {l} C 2_10_10_10 {:?}",
            [fl(v[18]), fl(v[19]), fl(v[20]), fl(v[21])]
        );
        assert_eq!(fl(v[22]), -1.0, "lane {l} D 16_snorm unaligned");
        assert_eq!([fl(v[23]), fl(v[24]), fl(v[25])], [1.0, 2.0, 0.5], "lane {l} E 10_11_11_float");
        assert_eq!([fl(v[26]), fl(v[27])], [1.5, -3.25], "lane {l} F tbuffer 32_32_float");
        assert_eq!(v[28], 0, "lane {l} out of range");
        assert_eq!(v[31], 0xC000_3C00, "lane {l} d16 16_16_float");
        assert_eq!(v[32], 0x0000_BC00, "lane {l} d16 16_snorm");
        assert_eq!(v[33], 0xC280_3E00, "lane {l} d16 tbuffer 32_32_float");
        assert_eq!(v[29], 0x11C9_3344, "lane {l} v_cvt_pk_u8_f32 into byte 2");
        assert_eq!(r.mem[0x400 / 4 + l], 0xC0DE_0000 | l as u32, "lane {l} buffer_store_dword");
    }
    assert_eq!(r.mem[0x130 / 4], 0xB800_4000, "d16 store 16_16_float");
    assert_eq!(r.sgpr[50], 1 | 1 << 5, "s_bitset1");
}

#[test]
fn st_bytes_writes_every_dword_of_unaligned_stores() {
    let code = prog(&[
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 3)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(
        t.hlsl.contains("uint rem = min(4u, n - k * 4u);") && t.hlsl.contains("st_part(b + 4u * k, m << s, w << s);"),
        "st_bytes must loop over all dwords on unaligned/non-multiple-of-4 stores:\n{}",
        t.hlsl
    );
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_unaligned_multi_dword_format_stores() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32| [base, 0, 0x100, ident | fmt << 12];
    let code = prog(&[
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None), // lane 0 only
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 1)], None),   // 8 B @ +1
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 4), ("SOFFSET", 128), ("OFFSET", 2)], None),  // 12 B @ +2
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 3)], None), // 16 B @ +3
    ]);
    let mem = vec![0xAAAA_AAAAu32; 256];
    let mut st = KernelState::default();
    for (i, v) in [vs(0x100, 62), vs(0x140, 72), vs(0x180, 75)].iter().enumerate() {
        st.sgpr[12 + 4 * i..16 + 4 * i].copy_from_slice(v);
    }
    let words = [0x1122_3344u32, 0x5566_7788, 0x99AA_BBCC, 0xDDEE_FF01];
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[10..14].copy_from_slice(&words);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let bytes: Vec<u8> = r.mem.iter().flat_map(|w| w.to_le_bytes()).collect();
    for (base, off, count) in [(0x100usize, 1usize, 2usize), (0x140, 2, 3), (0x180, 3, 4)] {
        let mut expected = vec![0xAAu8; 24];
        for k in 0..count {
            expected[off + 4 * k..off + 4 * k + 4].copy_from_slice(&words[k].to_le_bytes());
        }
        assert_eq!(&bytes[base..base + 24], expected.as_slice(), "store of {} bytes at {base:#x}+{off}", 4 * count);
    }
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// VS + PS pair recompiled to graphics stages and drawn on the GPU: VS inputs from the
/// fetch-shader VGPRs, exp pos0/param0, PS interpolation through v_interp, discard via v_cmpx.
#[test]
fn gpu_graphics_vs_ps_draw() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(8.0f32.to_bits())), // keep pos.x > 8
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 240), ("VSRC1", 4)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let vt = translate(&vs_code, &Options { mode: Mode::Vertex(vsi), wave64: true, ..Default::default() }).unwrap();
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(vt.complete() && pt.complete(), "{:?} {:?}", vt.unsupported, pt.unsupported);
    let vs = super::dxc::compile(&vt.hlsl, "main", "vs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", vt.hlsl));
    let ps = super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let clear = [0.0, 0.0, 1.0, 1.0];
    let px = runner::draw(&vs, &ps, &layout, &verts, 24, &[], &[], 16, clear).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let want = if x >= 8 { [0.4, 0.6, 0.4, 1.0] } else { clear };
            let got = px[y * 16 + x];
            assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "pixel ({x},{y}) = {got:?}, expected {want:?}");
        }
    }
}

/// One raw UAV in the descriptor heap spans a whole 4 GB memory window: the shader's stores to
/// its first and last dwords come back, and a dword in an unmapped tile between reads 0.
#[test]
fn gpu_raw_view_spans_a_memory_window() {
    if let Err(e) = runner::adapter() {
        return runner::skip_test(&e);
    }
    let src = "RWByteAddressBuffer io : register(u0);\n\
        [numthreads(1, 1, 1)] void main() {\n\
            globallycoherent RWByteAddressBuffer w = ResourceDescriptorHeap[0];\n\
            w.Store(0u, 0x11111111u); w.Store(0xFFFFFFFCu, 0x22222222u);\n\
            DeviceMemoryBarrier();\n\
            io.Store(0u, w.Load(0u)); io.Store(4u, w.Load(0xFFFFFFFCu)); io.Store(8u, w.Load(0x80000000u));\n\
        }\n";
    let dxil = super::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
    let r = match runner::run_with_window_view(&dxil, 1 << 32) {
        Ok(r) => r,
        // The window itself is the subject: an adapter that cannot reserve 4 GiB (WARP on the CI
        // runner) has nothing for this test to prove.
        Err(e) if e.contains("CreateReservedResource") => {
            return runner::skip_test(&format!("no 4 GiB memory window ({e})"));
        }
        Err(e) => panic!("{e}"),
    };
    assert_eq!(&r[..3], &[0x1111_1111, 0x2222_2222, 0]);
}

/// `emit::track_exec` follows every EXEC assignment with the `xcl_on` recompute, inside a
/// block so a braceless `if` still guards both, and leaves reads, comparisons, other
/// registers and comments alone.
#[test]
fn track_exec_recomputes_after_exec_writes() {
    let tr = |s| super::emit::track_exec(s, "T").into_owned();
    assert_eq!(tr("exec_lo = s0;"), "{ exec_lo = s0; xcl_on = T; }");
    assert_eq!(tr("if (scc != 0u) exec_hi &= f(a, (b; c)); v0 = 1u;"), "if (scc != 0u) { exec_hi &= f(a, (b; c)); xcl_on = T; } v0 = 1u;");
    assert_eq!(tr("s0 = exec_lo; exec_lo = exec_lo & s1; exec_hi <<= 1u;"), "s0 = exec_lo; { exec_lo = exec_lo & s1; xcl_on = T; } { exec_hi <<= 1u; xcl_on = T; }");
    for s in ["scc = exec_lo == 0u;", "if (exec_lo <= s0) v1 = 0u;", "xcl_exec_lo = 1u;", "exec_low = 1u;", "// exec_lo = s0", "v0 = 1u; // exec_lo = s0"] {
        assert_eq!(tr(s), s);
    }
}

/// `emit::rect_list_gs` declares the VS's output struct; the fourth vertex is v[0] with its
/// float outputs continued from the other three, its integer (flat) ones left as v[0]'s.
#[test]
fn rect_list_gs_outputs() {
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF), el("TEXCOORD", 1, 0, 1, 2, 0xF), el("SV_ClipDistance", 0, 2, 3, 3, 1)];
    let h = super::emit::rect_list_gs(&outputs);
    assert!(h.contains("triangle XclOut v[3]") && h.contains("XclOut d = v[0];"), "{h}");
    assert!(h.contains("d.p0 = a.p0 + b.p0 - k.p0;") && h.contains("d.p2 = a.p2 + b.p2 - k.p2;") && !h.contains("d.p1 ="), "{h}");
    if super::dxc::location().is_ok() {
        super::dxc::compile(&h, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{h}"));
    }
}

/// `emit::rect_list_gs` on the full-screen pass a downstream build (theaustinayers/xcl, build 5,
/// via #189) reports from a retail title: positions (−1,1) (−1,−1) (1,−1), texture coordinates
/// (0,0) (0,1) (1,1), the shared corner v1 and the implied one (1,1) with (1,0). Plain HLSL
/// VS/PS around the GS. Every pixel must be drawn with its texture coordinate, for w 1 and 2;
/// without the GS about half are. The other half of that rect as a triangle (v0, v2 and the
/// implied corner – what a draw expanded to two triangles a rect on the CPU sends next) is three
/// corners of the same rect, so the GS completes it to the whole rect as well: such a draw must
/// not also get this GS, or each rect is drawn twice.
#[test]
fn gpu_rect_list_fullscreen_order() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF)];
    let vs_hlsl = "struct V { float4 pos : SV_Position; float4 uv : TEXCOORD0; };\nV main(float4 pos : POSITION, float4 uv : TEXCOORD0) { V o; o.pos = pos; o.uv = uv; return o; }\n";
    let ps_hlsl = "float4 main(float4 pos : SV_Position, float4 uv : TEXCOORD0) : SV_Target { return uv; }\n";
    let vs = super::dxc::compile(vs_hlsl, "main", "vs_6_6", &[]).unwrap();
    let ps = super::dxc::compile(ps_hlsl, "main", "ps_6_6", &[]).unwrap();
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "TEXCOORD", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 16 },
    ];
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    let first = [([-1.0f32, 1.0], [0.0f32, 0.0]), ([-1.0, -1.0], [0.0, 1.0]), ([1.0, -1.0], [1.0, 1.0])];
    let second = [([-1.0f32, 1.0], [0.0f32, 0.0]), ([1.0, -1.0], [1.0, 1.0]), ([1.0, 1.0], [1.0, 0.0])];
    for (name, corners, w) in [("v0 v1 v2", first, 1.0f32), ("v0 v1 v2", first, 2.0), ("v0 v2 v3", second, 1.0)] {
        let v: Vec<f32> = corners.iter().flat_map(|&([x, y], [s, t])| [x * w, y * w, 0.0, w, s, t, 0.0, 1.0]).collect();
        let run = |with_gs: bool| runner::draw_gs(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, &v, 32, &[], &[], n as u32, clear).unwrap();
        let px = run(true);
        for y in 0..n {
            for x in 0..n {
                let want = [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, 0.0, 1.0];
                let got = px[y * n + x];
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{name}, w {w}: pixel ({x},{y}) = {got:?}, expected {want:?}");
            }
        }
        // the negative control: the three vertices alone draw half the rect
        let drawn = run(false).iter().filter(|p| **p != clear).count();
        assert!(drawn > n * n / 3 && drawn < n * n * 2 / 3, "{name}, w {w}: without the GS {drawn} of {} pixels drawn", n * n);
    }
}

/// A console rect list with flat outputs – an integer one, and a float one the PS reads
/// `nointerpolation` (what `v_interp_mov_f32` is translated to) – drawn indexed through
/// `emit::rect_list_gs`, the corner at v0, v1 or v2: the whole rect is drawn, the smooth output
/// continues into the implied corner's half, and both flat ones are the rect's first vertex's
/// in both halves. On the console that is the provoking vertex under D3D's setting
/// (PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST = 0, see `emit::rect_list_gs`). Synthetic: not a
/// measurement of the hardware. Without the GS part of every rect stays clear.
#[test]
fn gpu_rect_list_flat_outputs() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF), el("TEXCOORD", 1, 0, 1, 2, 0xF), el("TEXCOORD", 2, 0, 3, 3, 0xF)];
    let vs_hlsl = "struct V { float4 pos : SV_Position; float4 uv : TEXCOORD0; uint4 id : TEXCOORD1; float4 tint : TEXCOORD2; };\nV main(float4 pos : POSITION, uint id : ID) { V o; o.pos = pos; o.uv = float4(pos.xy, 0.0, 1.0); o.id = uint4(id, 0u, 0u, 0u); o.tint = float4(id * 0.125, 0.0, 0.0, 0.0); return o; }\n";
    let ps_hlsl = "float4 main(float4 pos : SV_Position, float4 uv : TEXCOORD0, nointerpolation uint4 id : TEXCOORD1, nointerpolation float4 tint : TEXCOORD2) : SV_Target { return float4(uv.xy, id.x, tint.x); }\n";
    let vs = super::dxc::compile(vs_hlsl, "main", "vs_6_6", &[]).unwrap();
    let ps = super::dxc::compile(ps_hlsl, "main", "ps_6_6", &[]).unwrap();
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "ID", index: 0, format: DXGI_FORMAT_R32_UINT, offset: 16 },
    ];
    // a 3 × 2 grid: columns x = -0.75, 0, 0.75, rows y = -0.5, 0.5; vertex 2·column + row has id = its number + 1
    let grid: Vec<f32> = (0..6u32).flat_map(|v| [[-0.75f32, 0.0, 0.75][v as usize / 2], [-0.5f32, 0.5][v as usize % 2], 0.0, 1.0, f32::from_bits(v + 1)]).collect();
    let (left, right) = ([-0.75f32, 0.0, -0.5, 0.5], [0.0f32, 0.75, -0.5, 0.5]);
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    // (indices, the rects (x1, x2, y1, y2) and their first vertex's id)
    for (name, indices, rects) in [
        ("corner v0", vec![0u32, 1, 2], vec![(left, 1.0f32)]),
        ("corner v1", vec![0, 1, 3], vec![(left, 1.0)]),
        ("corner v1, the other way round", vec![1, 0, 2], vec![(left, 2.0)]),
        ("corner v2", vec![1, 2, 0], vec![(left, 2.0)]),
        ("two rects sharing vertices", vec![0, 1, 2, 5, 3, 2], vec![(left, 1.0), (right, 6.0)]),
    ] {
        let run = |with_gs: bool| runner::draw_gs_indexed(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, &grid, 20, &indices, n as u32, clear).unwrap();
        let wrong = |px: &[[f32; 4]], report: bool| {
            let mut wrong = 0;
            for y in 0..n {
                for x in 0..n {
                    let (fx, fy) = ((x as f32 + 0.5) / 8.0 - 1.0, 1.0 - (y as f32 + 0.5) / 8.0);
                    let rect = rects.iter().find(|&&([x1, x2, y1, y2], _)| fx > x1 && fx < x2 && fy > y1 && fy < y2);
                    let want = rect.map_or(clear, |&(_, id)| [fx, fy, id, id * 0.125]);
                    let got = px[y * n + x];
                    if !got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5) {
                        assert!(!report, "{name}: pixel ({x},{y}) = {got:?}, expected {want:?}");
                        wrong += 1;
                    }
                }
            }
            wrong
        };
        wrong(&run(true), true);
        // the negative control: as a triangle list part of every rect is not drawn
        let missing = wrong(&run(false), false);
        assert!(missing >= 12 * rects.len(), "{name}: without the GS {missing} pixels differ");
    }
}

/// Console rect lists (DI_PT_RECTLIST) drawn through `emit::rect_list_gs`: three corners of a
/// screen-aligned rectangle, the corner between the other two at v0 (Mesa radeonsi's blits),
/// v1 (xf86-video-ati's r600 EXA) or v2. Each covers the whole rectangle and its attribute – the
/// vertex's own position – is the pixel's position everywhere, the implied corner's quarter
/// included (both drivers' texture coordinates depend on it). Without the GS (as before) part
/// of the rectangle stays clear; three vertices that are not such corners, or differ in w, draw
/// the same with and without it. Synthetic: the expected images are the rectangle and the
/// triangle, not a measurement of the hardware.
#[test]
fn gpu_rect_list_corners() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let vs_code = prog(&[
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    };
    let mut ps_ops: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for c in 0..2u64 {
        ps_ops.push(("V_INTERP_P1_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", c)], None));
        ps_ops.push(("V_INTERP_P2_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", c)], None));
    }
    ps_ops.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 6), ("SRC0", 128)], None));
    ps_ops.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 242)], None));
    ps_ops.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let ps_list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = ps_ops.iter().map(|(n, e, f, l)| (*n, *e, f.as_slice(), *l)).collect();
    let ps_code = prog(&ps_list);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let outputs = vsi.outputs.clone();
    let vt = translate(&vs_code, &Options { mode: Mode::Vertex(vsi), wave64: true, ..Default::default() }).unwrap();
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(vt.complete() && pt.complete(), "{:?} {:?}", vt.unsupported, pt.unsupported);
    let vs = super::dxc::compile(&vt.hlsl, "main", "vs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", vt.hlsl));
    let ps = super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 16 },
    ];
    // a vertex at NDC (x, y) with clip w: position (x·w, y·w, 0, w), COLOR (x, y, 0, 1)
    let verts = |ps: &[[f32; 3]]| -> Vec<f32> { ps.iter().flat_map(|&[x, y, w]| [x * w, y * w, 0.0, w, x, y, 0.0, 1.0]).collect() };
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    let ndc = |x: usize, y: usize| ((x as f32 + 0.5) / 8.0 - 1.0, 1.0 - (y as f32 + 0.5) / 8.0);
    let run = |v: &[f32], with_gs: bool| runner::draw_gs(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, v, 32, &[], &[], n as u32, clear).unwrap();
    let (x1, x2, y1, y2) = (-0.5f32, 0.5, -0.5, 0.5);
    for (name, corners) in [
        ("corner v0 (radeonsi)", [[x1, y1], [x1, y2], [x2, y1]]),
        ("corner v1 (r600 EXA)", [[x1, y1], [x1, y2], [x2, y2]]),
        ("corner v2", [[x1, y2], [x2, y1], [x1, y1]]),
    ] {
        let v = verts(&corners.map(|[x, y]| [x, y, 1.0]));
        let px = run(&v, true);
        for y in 0..n {
            for x in 0..n {
                let (fx, fy) = ndc(x, y);
                let want = if fx > x1 && fx < x2 && fy > y1 && fy < y2 { [fx, fy, 0.0, 1.0] } else { clear };
                let got = px[y * n + x];
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{name}: pixel ({x},{y}) = {got:?}, expected {want:?}");
            }
        }
        // the negative control: as a triangle list only part of the rectangle is drawn
        let tri = run(&v, false);
        let missing = (0..n * n).filter(|&i| px[i] != clear && tri[i] == clear).count();
        assert!(missing >= 16, "{name}: without the GS {missing} of the rectangle's pixels stay clear");
    }
    // left as the triangle they make: not three rectangle corners, and corners with unequal w
    for (name, v) in [
        ("not a rectangle", verts(&[[-0.5, -0.5, 1.0], [0.25, 0.5, 1.0], [0.5, -0.25, 1.0]])),
        ("unequal w", verts(&[[x1, y1, 1.0], [x1, y2, 2.0], [x2, y1, 1.0]])),
    ] {
        let (with, without) = (run(&v, true), run(&v, false));
        assert!(without.iter().any(|p| *p != clear), "{name}: the triangle draws");
        assert!(with.iter().zip(&without).all(|(a, b)| a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-5)), "{name}: the GS changed the image");
    }
}

/// One instruction (programs built in loops).
fn ins(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, fields, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

/// A program from instructions, ending in `s_endpgm`.
fn program_of(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut v: Vec<u8> = parts.concat();
    v.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

/// The pixel interface of `gpu_graphics_vs_ps_draw`: v0 / v1 = i / j, v2 / v3 = pixel centre
/// x / y, attribute 0 = COLOR.
fn draw_psi() -> super::PsInterface {
    super::PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    }
}

/// The vertex interface of `gpu_graphics_vs_ps_draw`: POSITION in v4 / v5, COLOR in v8..v11;
/// exports pos0 and param0 (COLOR).
fn draw_vsi() -> super::VsInterface {
    super::VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    }
}

fn draw_passthrough_vs() -> Vec<u8> {
    prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ])
}

#[test]
fn gpu_ps_p0_reads_flat_integer_selectors() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32G32B32A32_SINT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32_FLOAT,
    };

    let vs_code = draw_passthrough_vs();
    // Match the parameter word against 1, 2, and all ones, one output channel per match.
    let ps_code = prog(&[
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 2), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 242)], None), // alpha and matching channel: 1.0
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 129), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 128), ("VSRC1", 8)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 8)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 193), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 128), ("VSRC1", 8)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 5), ("VSRC1", 6), ("VSRC2", 7), ("VSRC3", 8), ("DONE", 1), ("VM", 1)], None),
    ]);
    let compile = |code: &[u8], mode: Mode, profile: &str| {
        let t = translate(code, &runner::options(mode)).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        super::dxc::compile(&t.hlsl, "main", profile, &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
    };
    let cases = [
        (0, [0.0, 0.0, 0.0, 1.0]),
        (1, [1.0, 0.0, 0.0, 1.0]),
        (2, [0.0, 1.0, 0.0, 1.0]),
        (u32::MAX, [0.0, 0.0, 1.0, 1.0]),
    ];
    for (component_type, format) in [(1, DXGI_FORMAT_R32G32B32A32_UINT), (2, DXGI_FORMAT_R32G32B32A32_SINT)] {
        let mut vsi = draw_vsi();
        vsi.inputs[1].component_type = component_type;
        vsi.outputs[1].component_type = component_type;
        let mut psi = draw_psi();
        psi.inputs[1].component_type = component_type;
        let vs = compile(&vs_code, Mode::Vertex(vsi), "vs_6_6");
        let ps = compile(&ps_code, Mode::Pixel(psi), "ps_6_6");
        let layout = [
            runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
            runner::VertexElement { semantic: "COLOR", index: 0, format, offset: 8 },
        ];
        for (selector, expected) in cases {
            // The runner uploads these vertex bytes directly. The input format reads the word as an integer.
            let vertices: Vec<f32> = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]]
                .into_iter()
                .flat_map(|[x, y]| [x, y, f32::from_bits(selector), 0.0, 0.0, 0.0])
                .collect();
            let pixels = runner::draw(&vs, &ps, &layout, &vertices, 24, &[], &[], 16, [-1.0; 4]).unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                assert_eq!(*pixel, expected, "{format:?} selector {selector:#010x}, pixel {i}");
            }
        }
    }
}

/// Draw console wave64 VS + PS (`vs` None: position and COLOR passed through; `ps` None:
/// COLOR interpolated) with triangles given in pixels of a 16 × 16 target, cleared to -1.
/// Translated for this adapter (`runner::options`: on a GPU with 32-lane waves only, the
/// form of `upper_lanes`).
fn draw_console(vs: Option<&[u8]>, ps: Option<&[u8]>, tris_px: &[[f32; 2]]) -> Vec<[f32; 4]> {
    draw_console_views(vs, ps, tris_px, &[], &[], &[])
}

/// `draw_console` with initial SGPRs and textures / samplers (see `runner::ViewTexture`).
fn draw_console_views(
    vs: Option<&[u8]>,
    ps: Option<&[u8]>,
    tris_px: &[[f32; 2]],
    sgprs: &[u32],
    textures: &[runner::ViewTexture],
    samplers: &[windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC],
) -> Vec<[f32; 4]> {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let pass_vs = draw_passthrough_vs();
    let pass_ps = prog(&[
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 3)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 3)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let compile = |code: &[u8], mode: Mode, profile: &str| {
        let t = translate(code, &runner::options(mode)).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        super::dxc::compile(&t.hlsl, "main", profile, &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
    };
    let vsb = compile(vs.unwrap_or(&pass_vs), Mode::Vertex(draw_vsi()), "vs_6_6");
    let psb = compile(ps.unwrap_or(&pass_ps), Mode::Pixel(draw_psi()), "ps_6_6");
    // pixels → NDC (y down), COLOR 0
    let verts: Vec<f32> = tris_px.iter().flat_map(|p| [p[0] / 8.0 - 1.0, 1.0 - p[1] / 8.0, 0.0, 0.0, 0.0, 0.0]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    runner::draw_views(&vsb, &psb, &layout, &verts, 24, sgprs, &[], textures, samplers, 16, [-1.0; 4]).unwrap()
}

/// `image_sample_o` in a pixel shader, implicit LOD between two mips (trilinear): the console
/// moves each mip's texel coordinate by the offset in that mip's texels. A 4-level texture
/// (8², 4², 2², 1²; level k texel x holds 100·k + x), point min / mag, linear mip, clamp;
/// u = v = pixel centre · 2^0.25 / 8 gives LOD 0.25; offset (+1, 0).
#[test]
fn gpu_ps_image_sample_offset_implicit_lod_between_mips() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_SAMPLE_DESC};
    let scale = 2f32.powf(0.25) / 8.0;
    let ps = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 2)], Some(scale.to_bits())),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 3)], Some(scale.to_bits())),
        ("IMAGE_SAMPLE_O", "ENC_MIMG", &[("VDATA", 8), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 8), ("VSRC2", 8), ("VSRC3", 8), ("DONE", 1), ("VM", 1)], None),
    ]);
    let mut data = Vec::new();
    for k in 0..4u32 {
        let n = 8 >> k;
        for _y in 0..n {
            for x in 0..n {
                data.extend_from_slice(&[(100 * k + x) as f32; 4].map(f32::to_le_bytes).concat());
            }
        }
    }
    let res = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 8,
        Height: 8,
        DepthOrArraySize: 1,
        MipLevels: 4,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 4, ..Default::default() };
    let tex = runner::ViewTexture { res, srv, data, row_bytes: 8 * 16 };
    // S#: clamp u / v / w, point min / mag, MIP_FILTER linear (2), LOD 0..15.99, sampler 0
    let s_sharp = [2 | 2 << 3 | 2 << 6, 0xFFF << 12, 2 << 26, 0];
    let mut sgprs = vec![0u32; 32];
    sgprs[16..24].copy_from_slice(&tsharp(0, 77, 9));
    sgprs[24..28].copy_from_slice(&s_sharp);
    let px = draw_console_views(None, Some(&ps), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]], &sgprs, &[tex], &[crate::gpu::desc::sampler_d3d12(&s_sharp, false)]);
    for y in 0..16 {
        for x in 0..16 {
            let u = (x as f32 + 0.5) * scale;
            let texel = |size: f32| ((u * size).floor() + 1.0).clamp(0.0, size - 1.0);
            let want = 0.75 * texel(8.0) + 0.25 * (100.0 + texel(4.0));
            let got = px[y * 16 + x][0];
            assert!((got - want).abs() < 0.02, "pixel ({x},{y}) = {got}, expected {want}");
        }
    }
}

/// A 2-level texture (8², 4²; level k texel x holds 100·k + x); `r32`: one channel, else four.
fn two_mip_texture(r32: bool) -> runner::ViewTexture {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
    let format = if r32 { DXGI_FORMAT_R32_FLOAT } else { DXGI_FORMAT_R32G32B32A32_FLOAT };
    let mut data = Vec::new();
    for k in 0..2u32 {
        let n = 8 >> k;
        for _y in 0..n {
            for x in 0..n {
                let v = (100 * k + x) as f32;
                data.extend_from_slice(&if r32 { v.to_le_bytes().to_vec() } else { [v; 4].map(f32::to_le_bytes).concat() });
            }
        }
    }
    let res = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 8,
        Height: 8,
        DepthOrArraySize: 1,
        MipLevels: 2,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 2, ..Default::default() };
    runner::ViewTexture { res, srv, data, row_bytes: 8 * if r32 { 4 } else { 16 } }
}

/// Draws `ps` (its sample's VDATA in v20) with `two_mip_texture`, point min / mag / mip, clamp
/// (`cmp`: one channel and a LESS_EQUAL comparison sampler); every pixel's red is `want(u)`,
/// u = v = pixel centre · `scale`.
fn assert_two_mip_sample(name: &str, ps: &[u8], cmp: bool, scale: f32, want: &dyn Fn(f32) -> f32) {
    // S#: clamp u / v / w, point min / mag (0), MIP_FILTER point (1), DEPTH_COMPARE LESS_EQUAL (3 + 1)
    let s_sharp = [2 | 2 << 3 | 2 << 6 | if cmp { 3 << 12 } else { 0 }, 0xFFF << 12, 1 << 26, 0];
    let mut sgprs = vec![0u32; 32];
    sgprs[16..24].copy_from_slice(&tsharp(0, 77, 9));
    sgprs[24..28].copy_from_slice(&s_sharp);
    let px = draw_console_views(None, Some(ps), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]], &sgprs, &[two_mip_texture(cmp)], &[crate::gpu::desc::sampler_d3d12(&s_sharp, cmp)]);
    for y in 0..16 {
        for x in 0..16 {
            let u = (x as f32 + 0.5) * scale;
            let (got, w) = (px[y * 16 + x][0], want(u));
            assert!((got - w).abs() < 0.02, "{name}: pixel ({x},{y}) = {got}, expected {w}");
        }
    }
}

/// The texel of mip `k` (point filter) at u, moved by one of that mip's texels (offset +1, clamp).
fn texel_moved(k: u32, u: f32) -> f32 {
    let size = (8 >> k) as f32;
    100.0 * k as f32 + ((u * size).floor() + 1.0).clamp(0.0, size - 1.0)
}

const EXP_V20: (&str, &str, &[(&str, u64)], Option<u32>) =
    ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 20), ("VSRC2", 20), ("VSRC3", 20), ("DONE", 1), ("VM", 1)], None);

/// Control for the tests below: `image_sample_o` at implicit LOD 0.75 with a point mip filter
/// reads mip 1 alone, moved by one of its texels (right on main since #229).
#[test]
fn gpu_ps_image_sample_offset_point_mip() {
    if !gpu_ready() {
        return;
    }
    let s = 2f32.powf(0.75) / 8.0;
    let ps = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 2)], Some(s.to_bits())),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 3)], Some(s.to_bits())),
        ("IMAGE_SAMPLE_O", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
        EXP_V20,
    ]);
    assert_two_mip_sample("sample_o", &ps, false, s, &|u| texel_moved(1, u));
}

/// `image_sample_b_o` with bias 1, u = v = pixel centre · `s`: v4 offset, v5 bias, v6 / v7 u / v.
fn sample_b_o(s: f32) -> Vec<u8> {
    prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 242)], None), // bias 1.0
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 2)], Some(s.to_bits())),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 3)], Some(s.to_bits())),
        ("IMAGE_SAMPLE_B_O", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
        EXP_V20,
    ])
}

/// `image_sample_b_o`: implicit LOD 0 + bias 1 reads mip 1, so the offset is one of mip 1's texels.
#[test]
fn gpu_ps_image_sample_offset_bias_lod() {
    if !gpu_ready() {
        return;
    }
    let s = 1f32 / 8.0;
    assert_two_mip_sample("sample_b_o", &sample_b_o(s), false, s, &|u| texel_moved(1, u));
}

/// `image_sample_b_o` magnified: implicit LOD -1 + bias 1 reads mip 0 (the bias applies to the
/// LOD before it is clamped to 0), so the offset is one of mip 0's texels.
#[test]
fn gpu_ps_image_sample_offset_bias_lod_magnified() {
    if !gpu_ready() {
        return;
    }
    let s = 1f32 / 16.0;
    assert_two_mip_sample("sample_b_o magnified", &sample_b_o(s), false, s, &|u| texel_moved(0, u));
}

/// `image_sample_d_o` / `image_sample_cd_o`: gradients of 2 mip-0 texels per pixel (LOD 1) read
/// mip 1, so the offset is one of mip 1's texels.
#[test]
fn gpu_ps_image_sample_offset_gradient_lod() {
    if !gpu_ready() {
        return;
    }
    let (s, g) = (1f32 / 8.0, 2f32 / 8.0);
    for op in ["IMAGE_SAMPLE_D_O", "IMAGE_SAMPLE_CD_O"] {
        let ps = prog(&[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 255)], Some(g.to_bits())), // du/dx
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 255)], Some(g.to_bits())), // dv/dy
            ("V_MUL_F32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 2)], Some(s.to_bits())),
            ("V_MUL_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 255), ("VSRC1", 3)], Some(s.to_bits())),
            (op, "ENC_MIMG", &[("VDATA", 20), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
            EXP_V20,
        ]);
        assert_two_mip_sample(op, &ps, false, s, &|u| texel_moved(1, u));
    }
}

/// `image_sample_c_o` at implicit LOD 1, LESS_EQUAL: z is the moved mip-1 texel - 0.5, so each
/// pixel passes (1) only if the offset was one of mip 1's texels; u = v = pixel centre / 4 - 1/16
/// puts mip 0's single texel (half of mip 1's) short of the next mip-1 texel.
#[test]
fn gpu_ps_image_sample_offset_compare_lod() {
    if !gpu_ready() {
        return;
    }
    let s = 2f32 / 8.0;
    let ps = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 2)], Some(s.to_bits())),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 3)], Some(s.to_bits())),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 6)], Some((-1f32 / 16.0).to_bits())),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 7)], Some((-1f32 / 16.0).to_bits())),
        // z = 100 + min(floor(4u) + 1, 3) - 0.5
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 246), ("VSRC1", 6)], None),
        ("V_FLOOR_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 5)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 242), ("VSRC1", 5)], None),
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 5)], Some(3f32.to_bits())),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 5)], Some(99.5f32.to_bits())),
        ("IMAGE_SAMPLE_C_O", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
        EXP_V20,
    ]);
    assert_two_mip_sample("sample_c_o", &ps, true, s, &|_| 1.0);
}

/// 48 SGPR spill slots in lanes 0..47 of v30 (`v_writelane` / `v_readlane`, as 287 retail
/// pixel shaders do – 11 up to lane 47): slot k = `slot(k)`, then s44 = the sum of slots
/// 0..31, s45 = the sum of slots 32..47.
fn spill_slots(slot: fn(u32) -> u32) -> Vec<u8> {
    let mut code = Vec::new();
    for k in 0..48u32 {
        code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 255)], Some(slot(k))));
        code.extend(ins("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", 20), ("SRC1", 128 + k as u64)], None));
    }
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 44), ("SSRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 45), ("SSRC0", 128)], None));
    for k in 0..48u64 {
        code.extend(ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 30), ("SRC1", 128 + k)], None));
        let sum = if k < 32 { 44 } else { 45 };
        code.extend(ins("S_ADD_U32", "ENC_SOP2", &[("SDST", sum), ("SSRC0", sum), ("SSRC1", 20)], None));
    }
    code
}

/// The whole-wave reduction of the retail pixel shaders (`upper_lanes`), run as a wave64 PS:
/// identity into the lanes outside the mask, DPP8 xor 1 / 2 / 4, row_mirror, the other row
/// through `v_permlanex16`, `v_readlane` 32 for the upper half's result, `v_readlane` 0 for
/// the total – as min (identity -1, as in the pack) and as a sum (identity 0: wrong if lane
/// 32 is read as anything but the identity, e.g. as lane 0, where a 32-lane GPU's wave read
/// wraps). Over the odd pixels of each wave; also the waterfall read of the pack
/// (`v_readlane` of the lane `s_ff1` picks from mask & EXEC). The waves are found from the
/// pixels: each exports its wave's `v_readfirstlane` of the pixel number and its lane
/// (`v_mbcnt`), and the CPU computes what a console wave of those lanes gives.
#[test]
fn gpu_wave64_ps_whole_wave_reductions() {
    use std::collections::{BTreeMap, BTreeSet};
    if !gpu_ready() {
        return;
    }
    let reduce = |op: &str, identity: u64, out: u64| -> Vec<u8> {
        let mut c = vec![
            ins("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None),
            ins("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 30)], None),
            ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", identity), ("SRC1", V + 13), ("SRC2", 10)], None),
        ];
        let names = ["LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3", "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"];
        for sel in [[1u64, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5], [4, 5, 6, 7, 0, 1, 2, 3]] {
            let mut f = vec![("VDST", 6u64), ("SRC0", 0xE9), ("VSRC0", 6), ("VSRC1", 6)];
            f.extend(names.iter().copied().zip(sel));
            c.push(ins(op, "VOP2_VOP_DPP8", &f, None));
        }
        c.extend([
            ins(op, "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ins("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 128), ("SRC2", 128)], None),
            ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 7), ("VSRC1", 6)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
            ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", 10), ("VSRC1", 6)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", out), ("SRC0", V + 6), ("SRC1", 128)], None),
            ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None),
        ]);
        c.concat()
    };
    let code = program_of(&[
        // v12 = pixel number y·16 + x, v13 = v12 + 1, s[18:19] = the odd pixels
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 2)], None),
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 3)], None),
        ins("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 11), ("SRC1", 128 + 4), ("SRC2", V + 10)], None),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 129), ("VSRC1", 12)], None),
        ins("V_AND_B32", "ENC_VOP2", &[("VDST", 14), ("SRC0", 129), ("VSRC1", 12)], None),
        ins("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 14)], None),
        ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 18), ("SSRC0", 106)], None),
        // the wave: its first lane's pixel number; the lane
        ins("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 62), ("SRC0", V + 12)], None),
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", 193), ("SRC1", V + 15)], None),
        reduce("V_MIN_U32", 193, 40),
        reduce("V_ADD_NC_U32", 128, 41),
        ins("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 18), ("SSRC1", 126)], None),
        ins("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)], None),
        ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 43), ("SRC0", V + 13), ("SRC1", 0)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 41)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 22), ("SRC0", 43)], None),
        ins("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 62), ("SRC1", 128 + 6), ("SRC2", V + 15)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 23)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 21), ("VSRC2", 22), ("VSRC3", 23), ("DONE", 1), ("VM", 1)], None),
    ]);
    let px = draw_console(None, Some(&code), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    // by wave (its first lane's pixel number): (pixel number, lane, results)
    let mut waves: BTreeMap<u32, Vec<(u32, u32, [f32; 4])>> = BTreeMap::new();
    for (id, p) in px.iter().enumerate() {
        let kl = p[3] as u32;
        waves.entry(kl >> 6).or_default().push((id as u32, kl & 63, *p));
    }
    println!("waves of {:?} pixels", waves.values().map(|w| w.len()).collect::<Vec<_>>());
    for (first, w) in &waves {
        let lanes: BTreeSet<u32> = w.iter().map(|x| x.1).collect();
        assert_eq!(lanes.len(), w.len(), "wave of pixel {first}: a lane twice");
        let lowest = w.iter().min_by_key(|x| x.1).unwrap();
        assert_eq!(lowest.0, *first, "v_readfirstlane: the wave's lowest lane");
        let odd: Vec<&(u32, u32, [f32; 4])> = w.iter().filter(|x| x.0 % 2 == 1).collect();
        let Some(first_odd) = odd.iter().min_by_key(|x| x.1) else { continue };
        let want = [odd.iter().map(|x| x.0 + 1).min().unwrap(), odd.iter().map(|x| x.0 + 1).sum(), first_odd.0 + 1];
        for (id, lane, p) in w {
            let got = [p[0] as u32, p[1] as u32, p[2] as u32];
            assert_eq!(got, want, "pixel {id} (lane {lane}, wave of pixel {first}): min / sum over the odd pixels, waterfall read");
        }
    }
}

/// SGPR spill slots in lanes 0..47 of a VGPR (`spill_slots`) in a wave64 PS: slots ≥ 32 do
/// not exist in a 32-lane wave, and a slot in a lane that was not launched is lost in a
/// partial wave (a primitive's edge). Drawn over the whole target and as a 6-pixel triangle.
#[test]
fn gpu_wave64_ps_spill_slots() {
    if !gpu_ready() {
        return;
    }
    let slot: fn(u32) -> u32 = |k| k * k + 7 * k + 1;
    let code = program_of(&[
        spill_slots(slot),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 44)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 45)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", 242)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", 242)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 21), ("VSRC2", 22), ("VSRC3", 23), ("DONE", 1), ("VM", 1)], None),
    ]);
    let want = [(0..32).map(slot).sum::<u32>() as f32, (32..48).map(slot).sum::<u32>() as f32];
    let full = draw_console(None, Some(&code), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    for (i, p) in full.iter().enumerate() {
        assert_eq!([p[0], p[1]], want, "pixel {i}, full waves: slots 0..31, 32..47");
    }
    // pixels (0..2, 0), (0..1, 1), (0, 2): three quads
    let small = draw_console(None, Some(&code), &[[0.0, 0.0], [3.2, 0.0], [0.0, 3.2]]);
    let covered: Vec<(usize, &[f32; 4])> = small.iter().enumerate().filter(|(_, p)| p[3] == 1.0).collect();
    assert_eq!(covered.len(), 6);
    // the 32-lane form keeps slots < 32 whatever the wave size, slots ≥ 32 in 32-lane waves;
    // the native form (GPUs with 64-lane waves) still loses the slots of lanes not launched
    let (lo, hi) = (runner::graphics_wave32(), runner::wave64_supported() == Ok(false));
    for (i, p) in covered {
        println!("pixel {i}, partial wave: {:?} (want {want:?})", &p[..2]);
        if lo {
            assert_eq!(p[0], want[0], "pixel {i}, partial wave: slots 0..31");
        }
        if hi {
            assert_eq!(p[1], want[1], "pixel {i}, partial wave: slots 32..47");
        }
    }
}

/// A wave64 VS with SGPR spill slots in lanes 0..47 (`spill_slots`) and a whole-wave identity
/// read from lane 32 (`v_cndmask` with the saved EXEC, `v_readlane` 32): three vertices, so a
/// wave of three lanes. The VS exports the slots' sums and whether lane 32 held the identity
/// as param0, which the PS interpolates (the same at every vertex: exact).
#[test]
fn gpu_wave64_vs_spill_slots_and_identity() {
    if !gpu_ready() {
        return;
    }
    let slot: fn(u32) -> u32 = |k| 3 * k * k + k + 5;
    let code = program_of(&[
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        spill_slots(slot),
        ins("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None),
        ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 193), ("SRC1", V), ("SRC2", 30)], None),
        ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
        ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None),
        ins("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 193)], None),
        ins("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 242), ("SSRC1", 128)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 44)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 45)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 11)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 242)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let px = draw_console(Some(&code), None, &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    let want = [(0..32).map(slot).sum::<u32>() as f32, (32..48).map(slot).sum::<u32>() as f32, 1.0, 1.0];
    // as in `gpu_wave64_ps_spill_slots`: slots < 32 in the 32-lane form; slots ≥ 32 and the
    // identity in lane 32 in 32-lane waves (in a 64-lane wave of three vertices, lane 32 was
    // not launched either)
    let (lo, hi) = (runner::graphics_wave32(), runner::wave64_supported() == Ok(false));
    println!("{:?} (want {want:?})", px[0]);
    for (i, p) in px.iter().enumerate() {
        if lo {
            assert_eq!(p[0], want[0], "pixel {i}: slots 0..31");
        }
        if hi {
            assert_eq!(&p[1..], &want[1..], "pixel {i}: slots 32..47, lane 32");
        }
    }
}

/// XCL image descriptor (T#, GFX10 layout): BASE_ADDRESS = SRV heap index, FORMAT, TYPE.
fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

/// A cube T# of two cubes: the face coordinate (face + 8 · cube, as v_cubeid and the console's
/// compiler give it) picks the cube, and image_get_resinfo gives the layers (6 per cube).
#[test]
fn gpu_cube_array_slices() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // v[0:2] = (s, t, face + 8 · cube), T# in s[16:23], S# point in s[24:27]
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 3)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 20), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 3)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 11));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (face, cube) = (l % 6, l / 6 % 2);
        r[0] = f(1.5); // the middle of the face
        r[1] = f(1.5);
        r[2] = f((face + 8 * cube) as f32);
        r[3] = 0; // mip
    }
    // 2 x 2 faces, every texel of layer (cube, face) = (face, cube, 6 · cube + face, 1)
    let mut data = Vec::new();
    for cube in 0..2 {
        for face in 0..6 {
            for _ in 0..4 {
                for v in [face as f32, cube as f32, (6 * cube + face) as f32, 1.0] {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    let tex = [TestTexture { dim: TexDim::Cube, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 2, height: 2, depth: 12, texel_bytes: 16, data, uav: false }];
    let samp = [TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (face, cube) = ((l % 6) as f32, (l / 6 % 2) as f32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [face, cube, 6.0 * cube + face, 1.0], "lane {l}: cube {cube} face {face}");
        assert_eq!(&v[20..24], &[2, 2, 12, 1], "lane {l} resinfo (w, h, layers, mips)");
    }
}

#[test]
fn sample_g16_uses_half_gradients_and_aligned_coordinates() {
    let code = prog(&[
        ("IMAGE_SAMPLE_D_G16", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None),
        ("IMAGE_SAMPLE_CD_G16", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 10), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 2), ("A16", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let h = |v, shift| format!("f16tof32((v{v} >> {shift}u) & 0xFFFFu)");
    let a = format!(
        "float2(asfloat(v2), asfloat(v3)), float2({}, {}), float2({}, {})",
        h(0, 0), h(0, 16), h(1, 0), h(1, 16)
    );
    let b = format!(
        "float3({}, {}, {}), float3({}, {}, {}), float3({}, {}, {})",
        h(14, 0), h(14, 16), h(15, 0), h(10, 0), h(10, 16), h(11, 0), h(12, 0), h(12, 16), h(13, 0)
    );
    assert!(t.hlsl.contains(&a), "{a}");
    assert!(t.hlsl.contains(&b), "{b}");
}

#[test]
fn gpu_cube_sample_grad_matches_face_texture() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
    let mut code = Vec::new();
    for (dim, addr, tex, result) in [(3, 0, 4, 20), (5, 8, 8, 21)] {
        code.extend(ins("IMAGE_SAMPLE_D", "ENC_MIMG", &[("DIM", dim), ("VADDR", addr), ("SRSRC", tex), ("SSAMP", 6), ("DMASK", 1), ("VDATA", result)], None));
    }
    code.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 22, 11));
    st.sgpr[32..40].copy_from_slice(&tsharp(2, 22, 13));
    for (lane, r) in st.vgpr.iter_mut().enumerate() {
        let face = lane % 6;
        let cube = lane / 6 % 2;
        let gradients = match lane / 12 % 4 {
            0 => [8.0, 0.0, 0.0, 1.0],
            1 => [1.0, 0.0, 0.0, 8.0],
            2 => [6.0, 6.0, -0.5, 0.5],
            _ => [6.0, -6.0, 0.5, 0.5],
        };
        for start in [0, 8] {
            r[start..start + 4].copy_from_slice(&gradients.map(|g| f(g / 32.0)));
        }
        let (u, v) = (12.5 / 32.0, 19.5 / 32.0);
        r[4..7].copy_from_slice(&[f(1.0 + u), f(1.0 + v), f((8 * cube + face) as f32)]);
        r[12..15].copy_from_slice(&[f(u), f(v), f((6 * cube + face) as f32)]);
    }
    let data: Vec<u8> = (0..12).flat_map(|slice| (0..32 * 32).flat_map(move |i| {
        let (x, y) = (i % 32, i / 32);
        let column = if x == 12 { 1.0 } else { 0.0 };
        let diagonal = if x + 7 == y { 0.5 } else { 0.0 };
        (slice as f32 + column + diagonal + y as f32 / 32.0).to_le_bytes()
    })).collect();
    let textures = [TexDim::Cube, TexDim::D2Array].map(|dim| TestTexture {
        dim, format: DXGI_FORMAT_R32_FLOAT, width: 32, height: 32, depth: 12,
        texel_bytes: 4, data: data.clone(), uav: false,
    });
    let sampler = TestSampler { filter: D3D12_FILTER_ANISOTROPIC, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None };
    let result = runner::run_code_with(&code, &st, &[], &textures, &[sampler]).unwrap();
    let reference = |lane: usize| f32::from_bits(result.vgpr[lane][21]);
    assert!(reference(12) > reference(0) + 0.25, "the reference must resolve the two footprint directions");
    assert!(reference(24) > reference(36) + 0.15, "the reference must resolve the two diagonals");
    for (lane, r) in result.vgpr.iter().enumerate() {
        let (cube, face) = (f32::from_bits(r[20]), f32::from_bits(r[21]));
        assert!((cube - face).abs() < 0.04, "lane {lane}: cube {cube}, 2D face {face}");
    }
}

#[test]
fn gpu_cube_sample_grad_edges() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
    let code = prog(&[("IMAGE_SAMPLE_D", "ENC_MIMG", &[("DIM", 3), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 1), ("VDATA", 10)], None)]);
    let adjacent = [[4, 5, 2, 3], [5, 4, 2, 3], [1, 0, 5, 4],
                    [1, 0, 4, 5], [1, 0, 2, 3], [0, 1, 2, 3]];
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 22, 11));
    let mut expected = Vec::new();
    for (lane, r) in st.vgpr.iter_mut().enumerate() {
        let (face, edge) = (lane / 4 % 6, lane % 4);
        let inset = if lane / 24 % 2 == 0 { 0.0 } else { 0.25 };
        let d = inset / 8.0;
        let (u, v) = [(d, 0.375), (1.0 - d, 0.375), (0.375, d), (0.375, 1.0 - d)][edge];
        r[..7].copy_from_slice(&[f(0.125), 0, 0, f(0.125), f(1.0 + u), f(1.0 + v), f(face as f32)]);
        expected.push((0.5 + inset) * face as f32 + (0.5 - inset) * adjacent[face][edge] as f32);
    }
    let data = (0..6).flat_map(|face| (0..64).flat_map(move |_| (face as f32).to_le_bytes())).collect();
    let tex = TestTexture { dim: TexDim::Cube, format: DXGI_FORMAT_R32_FLOAT, width: 8, height: 8, depth: 6, texel_bytes: 4, data, uav: false };
    let samp = TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None };
    let result = runner::run_code_with(&code, &st, &[], &[tex], &[samp]).unwrap();
    for (lane, want) in expected.into_iter().enumerate() {
        let got = f32::from_bits(result.vgpr[lane][10]);
        assert!((got - want).abs() < 0.02, "lane {lane}: seam {got}, expected {want}");
    }
}

#[test]
fn gpu_cube_sample_grad_min_lod() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
    let mut ps = Vec::new();
    for (k, value) in [0.125, 0.0, 0.0, 0.125, 1.5, 1.5, 0.0, 2.0].into_iter().enumerate() {
        ps.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 4 + k as u64), ("SRC0", 255)], Some(f(value))));
    }
    for (op, dst) in [("IMAGE_SAMPLE_D", 12), ("IMAGE_SAMPLE_D_CL", 13)] {
        ps.extend(ins(op, "ENC_MIMG", &[("DIM", 3), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 1), ("VDATA", dst)], None));
    }
    for register in [4, 7] {
        ps.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", register), ("SRC0", 255)], Some(f(0.5))));
    }
    for (minimum, dst) in [(1.0, 14), (2.5, 15)] {
        ps.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 255)], Some(f(minimum))));
        ps.extend(ins("IMAGE_SAMPLE_D_CL", "ENC_MIMG", &[("DIM", 3), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 1), ("VDATA", dst)], None));
    }
    ps.extend(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 12), ("VSRC1", 13), ("VSRC2", 14), ("VSRC3", 15), ("DONE", 1), ("VM", 1)], None));
    ps.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    let res = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 8, Height: 8, DepthOrArraySize: 6, MipLevels: 4,
        Format: DXGI_FORMAT_R32_FLOAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
    };
    let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: res.Format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default()
    };
    srv.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { MipLevels: 4, NumCubes: 1, ..Default::default() };
    let mut data = Vec::new();
    for mip in 0..4 {
        let width = 8 >> mip;
        for _ in 0..width * width {
            data.extend_from_slice(&(1.0 + mip as f32).to_le_bytes());
        }
    }
    let texture = runner::ViewTexture { res, srv, data, row_bytes: 8 * 4 };
    let mut sgprs = vec![0u32; 32];
    sgprs[16..24].copy_from_slice(&tsharp(0, 22, 11));
    let sampler = [2 | 2 << 3 | 2 << 6, 0xFFF << 12, 2 << 26, 0];
    sgprs[24..28].copy_from_slice(&sampler);
    let pixels = draw_console_views(None, Some(&ps), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]], &sgprs, &[texture], &[crate::gpu::desc::sampler_d3d12(&sampler, false)]);
    for p in pixels {
        assert_eq!(p, [1.0, 3.0, 3.0, 3.5]);
    }
}

#[test]
fn gpu_gather4_explicit_lod_uses_the_named_mip() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
    for (op, unorm, coords, expected) in [
        ("IMAGE_GATHER4_L", 0, [0.375, 0.375, 1.0, 0.0], [109.0, 110.0, 106.0, 105.0]),
        ("IMAGE_GATHER4_L_O", 1, [f32::from_bits(63), 1.5, 1.5, 1.0], [108.0, 109.0, 105.0, 104.0]),
        ("IMAGE_GATHER4_C_L", 0, [107.0, 0.375, 0.375, 1.0], [1.0, 1.0, 0.0, 0.0]),
    ] {
        let mut ps = Vec::new();
        for (k, value) in coords.into_iter().enumerate() {
            ps.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 4 + k as u64), ("SRC0", 255)], Some(f(value))));
        }
        ps.extend(ins(op, "ENC_MIMG", &[("DIM", 1), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 1), ("VDATA", 12), ("UNORM", unorm)], None));
        ps.extend(ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 15), ("VSRC0", 12), ("VSRC1", 13), ("VSRC2", 14), ("VSRC3", 15), ("DONE", 1), ("VM", 1)], None));
        ps.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
        let res = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 8, Height: 8, DepthOrArraySize: 1, MipLevels: 2,
            Format: DXGI_FORMAT_R32_FLOAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
        };
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: res.Format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default()
        };
        srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 2, ..Default::default() };
        let data = (0..64).map(|i| i as f32).chain((100..116).map(|i| i as f32)).flat_map(f32::to_le_bytes).collect();
        let texture = runner::ViewTexture { res, srv, data, row_bytes: 8 * 4 };
        let mut sgprs = vec![0u32; 32];
        sgprs[16..24].copy_from_slice(&tsharp(0, 22, 9));
        let sampler = [2 | 2 << 3 | 2 << 6 | 1 << 12, 0xFFF << 12, 2 << 26, 0];
        sgprs[24..28].copy_from_slice(&sampler);
        let pixels = draw_console_views(None, Some(&ps), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]], &sgprs, &[texture], &[crate::gpu::desc::sampler_d3d12(&sampler, false)]);
        for pixel in pixels {
            assert_eq!(pixel, expected, "{op}");
        }
    }
}

/// A gather reads an integer view as integers: the T#'s FORMAT class picks the view type at run
/// time, as for image_load, and a float view keeps the float gather.
#[test]
fn gather4_reads_integer_views_as_integers() {
    let gather: &[(&str, u64)] = &[("VDATA", 40), ("VADDR", 33), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)];
    let t = translate(&prog(&[("IMAGE_GATHER4_LZ", "ENC_MIMG", gather, None)]), &Options::default()).unwrap();
    for view in ["Texture2D<uint4>", "Texture2D<int4>", "Texture2D<float4>"] {
        assert!(t.hlsl.contains(view), "{view} gather");
    }
    assert_eq!(t.hlsl.matches(".GatherRed(").count(), 3, "one gather per view class");
}

#[test]
fn gpu_images_sample_load_store_atomic() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // T# tex0 in s[16:23] (SRSRC = 4), S# point in s[24:27] (SSAMP = 6), S# linear in s[28:31] (7)
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_SAMPLE_L", &[("VDATA", 20), ("VADDR", 30), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_GATHER4_LZ", &[("VDATA", 40), ("VADDR", 33), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 14), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 50), ("VADDR", 5), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1)]));
    // tex1: R32_UINT UAV, T# in s[32:39] (SRSRC = 8); tex2: 1x1 R32_UINT UAV in s[40:47] (10)
    code.extend(mimg("IMAGE_STORE", &[("VDATA", 6), ("VADDR", 37), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_ATOMIC_UMAX", &[("VDATA", 7), ("VADDR", 35), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    // tex3: 1x1 R32_UINT UAV in s[48:55] (12): image_atomic_inc (no HLSL intrinsic: CAS loop)
    code.extend(mimg("IMAGE_ATOMIC_INC", &[("VDATA", 39), ("VADDR", 35), ("SRSRC", 12), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    st.sgpr[24..28].copy_from_slice(&[0, 0, 0, 0]);
    st.sgpr[28..32].copy_from_slice(&[0, 0, 0, 1]);
    st.sgpr[32..40].copy_from_slice(&tsharp(2, 20, 9));
    st.sgpr[40..48].copy_from_slice(&tsharp(4, 20, 9));
    st.sgpr[48..56].copy_from_slice(&tsharp(6, 20, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = f((x as f32 + 0.5) / 4.0);
        r[1] = f((y as f32 + 0.5) / 4.0);
        r[3] = x;
        r[4] = y;
        r[5] = 0;
        r[6] = if l < 16 { 10 * l as u32 } else { 0 };
        r[7] = l as u32;
        r[30] = f(0.5); // between texel columns 1 and 2
        r[31] = f(0.125); // row 0
        r[32] = f(0.0); // lod
        r[33] = f(0.5);
        r[34] = f(0.5);
        r[35] = 0;
        r[36] = 0;
        r[37] = x; // lanes 16.. land outside the 4x4 texture: their stores are dropped
        r[38] = l as u32 / 4;
        r[39] = 100;
    }
    // tex0: 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: vec![0xEE; 64], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
    ];
    let samp = vec![
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
    ];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [x, y, x + 4.0 * y, 1.0], "lane {l} sample_lz point");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l} sample_l linear");
        assert_eq!([fl(40), fl(41), fl(42), fl(43)], [1.0, 2.0, 2.0, 1.0], "lane {l} gather4 red");
        assert_eq!([fl(14), fl(15)], [x, x + 4.0 * y], "lane {l} image_load dmask x,z");
        assert_eq!(&v[50..54], &[4, 4, 1, 1], "lane {l} get_resinfo");
    }
    let t1: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for i in 0..16 {
        assert_eq!(t1[i], 10 * i as u32, "image_store texel {i}");
    }
    assert_eq!(u32::from_le_bytes(r.textures[1][..4].try_into().unwrap()), 63, "image_atomic_umax");
    assert_eq!(u32::from_le_bytes(r.textures[2][..4].try_into().unwrap()), 64, "image_atomic_inc (64 lanes, limit 100)");
}

/// Static samplers: the program builds two S# from constants (point / linear, clamp) – as
/// the console compiler places a root signature's static samplers – and samples the same
/// texel boundary with each; they become static samplers of the root signature.
#[test]
fn gpu_static_samplers_from_constants_in_code() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let clamp = 2 | 2 << 3 | 2 << 6;
    let point = [clamp, 0x00FF_F000u32, 0, 0];
    let linear = [clamp, 0x00FF_F000, 1 << 20 | 1 << 22, 0];
    let mov = |r: u64, v: u32| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        if v == 0 {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 128)], None)
        } else {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 255)], Some(v))
        }
    };
    let mut list = Vec::new();
    for k in 0..4 {
        list.push(mov(24 + k as u64, point[k]));
        list.push(mov(28 + k as u64, linear[k]));
    }
    // T# tex0 in s[16:23] (SRSRC = 4); S# point s[24:27] (SSAMP = 6), linear s[28:31] (7)
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None));
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = translate(&code, &Options::default()).unwrap();
    assert_eq!(t.static_samplers, vec![(point, false), (linear, false)], "two static samplers from the code");
    assert!(!t.hlsl.contains("SamplerDescriptorHeap"), "no sampler heap lookups left");

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for r in st.vgpr.iter_mut() {
        r[0] = f(0.5); // the boundary between texel columns 1 and 2
        r[1] = f(0.125); // row 0
    }
    // 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [2.0, 0.0, 2.0, 1.0], "lane {l}: static point sampler");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l}: static linear sampler");
    }
}

/// Static samplers across blocks: an S# built from constants before a branch stays that constant
/// where every path to the sample agrees, and is no constant once one path rewrites a word.
#[test]
fn static_samplers_from_constants_in_an_earlier_block() {
    let clamp = 2 | 2 << 3 | 2 << 6;
    let point = [clamp, 0x00FF_F000u32, 0, 0];
    let mov = |r: u64, v: u32| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        if v == 0 {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 128)], None)
        } else {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 255)], Some(v))
        }
    };
    let translated = |arm: bool| {
        let mut list: Vec<_> = (0..4).map(|k| mov(24 + k as u64, point[k])).collect();
        list.push(("S_CBRANCH_SCC1", "ENC_SOPP", vec![("SIMM16", arm as u64)], None));
        if arm {
            list.push(mov(24, 0));
        }
        list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None));
        let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
        translate(&prog(&refs), &Options::default()).unwrap()
    };
    let agreed = translated(false);
    assert_eq!(agreed.static_samplers, vec![(point, false)], "the constants of the block before the branch");
    assert!(!agreed.hlsl.contains("SamplerDescriptorHeap"), "no sampler heap lookup left");
    let rewritten = translated(true);
    assert!(rewritten.static_samplers.is_empty(), "one path rewrites word 0");
    assert!(rewritten.hlsl.contains("SamplerDescriptorHeap"));
}

#[test]
fn gpu_images_a16_d16() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // coordinates (u, v) as f16 in v0; result as 4 halves in v[10:11]
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1), ("A16", 1), ("D16", 1)]));
    // integer coordinates (x, y) as 16-bit in v1; x and z in v[12:13] as 32-bit
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 12), ("VADDR", 1), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1), ("A16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    let h = |x: f32| half_bits(x);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = h((x as f32 + 0.5) / 4.0) | h((y as f32 + 0.5) / 4.0) << 16;
        r[1] = x | y << 16;
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        assert_eq!((v[10], v[11]), (h(x) | h(y) << 16, h(x + 4.0 * y) | h(1.0) << 16), "lane {l} a16+d16 sample");
        assert_eq!([f32::from_bits(v[12]), f32::from_bits(v[13])], [x, x + 4.0 * y], "lane {l} a16 load");
    }
}

/// f32 → IEEE half bits (exact for the small values used in tests).
fn half_bits(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    (b >> 16 & 0x8000) | ((e as u32) << 10) | ((b >> 13) & 0x3FF)
}

#[test]
fn gpu_format_stores_atomics_global_lds() {
    if !gpu_ready() {
        return;
    }
    let mubuf = |n: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(n, "ENC_MUBUF", f, None).unwrap() };
    let mut code = Vec::new();
    // V# A (s[12:15]): 8_8_8_8_UNORM at 0x100, stride 4 (idxen with v0 = lane)
    code.extend(mubuf("BUFFER_STORE_FORMAT_XYZW", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# B (s[16:19]): 16_FLOAT at 0x300, stride 2 → neighbouring lanes share dwords
    code.extend(mubuf("BUFFER_STORE_FORMAT_X", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# C (s[20:23]): raw at 0x400: every lane adds 1, umax returns the old value
    code.extend(mubuf("BUFFER_ATOMIC_ADD", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128)]));
    code.extend(mubuf("BUFFER_ATOMIC_UMAX", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 4), ("GLC", 1)]));
    // global: store lane*3 to s[2:3] + v1 (lane*4), load it back through the 64-bit form
    code.extend(encode("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 17), ("SADDR", 2), ("OFFSET", 0)], None).unwrap());
    code.extend(encode("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 18), ("ADDR", 20), ("SADDR", 0x7D), ("OFFSET", 0xFFC)], None).unwrap()); // offset -4
    // LDS: write lane at lane*4, barrier, read reversed, read2 of two neighbours, counter, bpermute
    code.extend(encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(encode("DS_READ_B32", "ENC_DS", &[("VDST", 22), ("ADDR", 23)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 24), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28), ("OFFSET0", 0), ("OFFSET1", 2)], None).unwrap()); // at 512
    code.extend(encode("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 29), ("ADDR", 23), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x500; // global base
    st.sgpr[12..16].copy_from_slice(&[0x100, 4 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 56 << 12]);
    st.sgpr[16..20].copy_from_slice(&[0x300, 2 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 13 << 12]);
    st.sgpr[20..24].copy_from_slice(&[0x400, 0, 16, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32;
        r[1] = 4 * l32;
        r[10] = f(1.0);
        r[11] = f(0.0);
        r[12] = f(l as f32 / 63.0);
        r[13] = f(0.5);
        r[14] = f(l as f32);
        r[15] = 1;
        r[16] = 100 + l32;
        r[17] = 3 * l32;
        r[20] = 0x500 + 4 * l32 + 4; // 64-bit address (hi in v21 = 0), offset -4
        r[23] = 4 * (63 - l32);
        r[27] = 0;
        r[28] = 1;
    }
    let r = runner::run_code(&code, &st, &vec![0u32; 1024]).unwrap();
    let byte = |a: usize| (r.mem[a / 4] >> (8 * (a % 4))) & 0xFF;
    for l in 0..64usize {
        let px = [byte(0x100 + 4 * l), byte(0x100 + 4 * l + 1), byte(0x100 + 4 * l + 2), byte(0x100 + 4 * l + 3)];
        assert_eq!(px, [255, 0, ((l as f32 / 63.0) * 255.0).round() as u32, 128], "lane {l} store 8_8_8_8_UNORM");
        let h = (r.mem[(0x300 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF;
        assert_eq!(h, half_bits(l as f32), "lane {l} store 16_FLOAT (shared dwords)");
        let v = &r.vgpr[l];
        assert_eq!(r.mem[0x500 / 4 + l], 3 * l as u32, "lane {l} global_store saddr");
        assert_eq!(v[18], 3 * l as u32, "lane {l} global_load 64-bit address, offset -4");
        assert_eq!(v[22], 63 - l as u32, "lane {l} LDS reversed");
        assert_eq!(v[24], l as u32, "lane {l} ds_read2 (first)");
        if l < 63 {
            assert_eq!(v[25], l as u32 + 1, "lane {l} ds_read2 (second; LDS is not zeroed past the written area)");
        }
        assert_eq!(v[29], 63 - l as u32, "lane {l} ds_bpermute");
        assert!(v[16] < 164, "lane {l} umax old value");
    }
    assert_eq!(r.mem[0x400 / 4], 64, "buffer_atomic_add");
    assert_eq!(r.mem[0x404 / 4], 163, "buffer_atomic_umax");
    let olds: Vec<u32> = r.vgpr.iter().map(|v| v[26]).collect();
    // LDS is not zeroed between dispatches (nor on the console): consecutive values from any
    // start, which may wrap (0xFFFFFFFF, 0, 1, …)
    let base = *olds.iter().find(|&&v| !olds.contains(&v.wrapping_sub(1))).unwrap_or(&olds[0]);
    let mut got: Vec<u32> = olds.iter().map(|v| v.wrapping_sub(base)).collect();
    got.sort();
    assert_eq!(got, (0..64).collect::<Vec<u32>>(), "ds_add_rtn_u32 returns every old value once (from {base:#x}): {olds:?}");
}

/// global_*_dword_addtid (ENC_FLAT_GLBL): the spec gives them no ADDR operand – the
/// address is the scalar SADDR pair ("off"/null encodes a zero base) plus 4 times the
/// lane index plus the signed 12-bit OFFSET. Emitted HLSL is deterministic.
#[test]
fn global_dword_addtid_address_is_saddr_plus_4_per_lane() {
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 0x7D), ("OFFSET", 0xFFC)])); // null base, offset -4
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // s[2:3] base, signed +8 offset; the ADDR field bits are never read as a VGPR
    assert!(t.hlsl.contains("uint t1_a = s2 + 4u * lane_id + 0x00000008u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("xcl_mStore((t1_a + 0u) & ~3u, v5);"), "{}", t.hlsl);
    // "off"/null SADDR → base 0; the 12-bit offset sign-extends (0xFFC → -4)
    assert!(t.hlsl.contains("uint t2_a = 4u * lane_id + 0xFFFFFFFCu;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v7 = xcl_mLoad((t2_a + 0u) & ~3u);"), "{}", t.hlsl);
    // exactly the two address expressions above – no ADDR VGPR read anywhere
    assert_eq!(t.hlsl.matches("_a = ").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// On hardware: every lane touches its own dword (EA = SADDR + 4·lane + OFFSET), the
/// "off" SADDR sentinel gives a zero base and the signed offset can point below the base.
#[test]
fn gpu_dword_addtid_per_lane_addresses() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    // each lane writes its signature at s[2:3] + 4·lane + 8, then reads it back (+8) and
    // the previous lane's dword (+4)
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 8), ("SADDR", 2), ("OFFSET", 4)]));
    // s[4:5] + 4·lane − 4 → preset values just below 0x200; offset −4
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 10), ("SADDR", 4), ("OFFSET", 0xFFC)]));
    // null ("off") SADDR: base 0 → per-lane dwords at 0x40 + 4·lane
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 9), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 11), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[4] = 0x200;
    let sig = |l: usize| 0xA500 + l as u32;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = sig(l);
        r[9] = 0xBEEF + l as u32;
    }
    let mut mem = vec![0u32; 512];
    mem[0x604 / 4] = 0xDEAD_CAFE; // in the gap below the stores: lane 0's +4 load lands here
    for l in 0..64usize {
        mem[0x1FC / 4 + l] = 0xF00 + l as u32; // under s[4:5] for the offset −4 loads
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for l in 0..64usize {
        assert_eq!(r.mem[0x608 / 4 + l], sig(l), "lane {l}: store at saddr + 4·lane + 8");
        assert_eq!(r.vgpr[l][7], sig(l), "lane {l}: load its own dword");
        let prev = if l == 0 { 0xDEAD_CAFE } else { sig(l - 1) };
        assert_eq!(r.vgpr[l][8], prev, "lane {l}: offset +4 reads the previous lane's dword");
        assert_eq!(r.vgpr[l][10], 0xF00 + l as u32, "lane {l}: signed offset −4 below the base");
        assert_eq!(r.mem[0x40 / 4 + l], 0xBEEF + l as u32, "lane {l}: store with a null SADDR base");
        assert_eq!(r.vgpr[l][11], 0xBEEF + l as u32, "lane {l}: load with a null SADDR base");
    }
}

#[test]
fn gpu_valu_16_and_64_bit() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)], None), // low halves, high kept
        ("V_MUL_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 0b1001)], None), // src0.hi * src1.lo → dst.hi
        ("V_LSHRREV_B16", "ENC_VOP3", &[("VDST", 4), ("SRC0", 132), ("SRC1", V + 5)], None),
        ("V_CMP_LT_F16", "ENC_VOPC", &[("SRC0", V + 6), ("VSRC1", 7)], None),
        ("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 10), ("SRC1", V + 12)], None),
        ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 8), ("SRC1", V + 14)], None),
        ("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 16), ("SDST", 22), ("SRC0", V + 18), ("SRC1", V + 19), ("SRC2", V + 24)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 26), ("SRC1", 242)], None), // + 1.0 (double inline constant)
        ("V_CVT_F32_F64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 26)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 40)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 32), ("SRC0", V + 40)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 34), ("SRC0", V + 40)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 36), ("SRC0", V + 40)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[2] = 0xBEEF_1234;
        r[3] = 0x0000_1111;
        r[4] = 0xCAFE_0000;
        r[5] = 0xFFFF_8421;
        r[6] = h(l as f32);
        r[7] = h(31.5);
        r[10] = l32;
        r[12] = 1;
        let e = if l % 2 == 0 { 1u64 << l } else { 0 };
        r[14] = e as u32;
        r[15] = (e >> 32) as u32;
        r[18] = 0xFFFF_FFFF;
        r[19] = l32 + 1;
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        r[24] = c as u32;
        r[25] = (c >> 32) as u32;
        let dv = (l as f64 + 0.25).to_bits();
        r[26] = dv as u32;
        r[27] = (dv >> 32) as u32;
        let x = (l as f64 + 1.25).powi(2).to_bits();
        r[40] = x as u32;
        r[41] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], 0xBEEF_0000 | h(3.75), "lane {l} v_add_f16 keeps the high half");
        assert_eq!(v[3], 0x1111 | h(-4.5) << 16, "lane {l} v_mul_f16 op_sel");
        assert_eq!(v[4], 0xCAFE_0842, "lane {l} v_lshrrev_b16");
        assert_eq!(v[8] as u64 | (v[9] as u64) << 32, 1u64 << l, "lane {l} v_lshlrev_b64");
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        let (w, carry) = (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(c);
        assert_eq!(v[16] as u64 | (v[17] as u64) << 32, w, "lane {l} v_mad_u64_u32");
        let _ = carry;
        assert_eq!(f64::from_bits(v[26] as u64 | (v[27] as u64) << 32), l as f64 + 1.25, "lane {l} v_add_f64");
        assert_eq!(f32::from_bits(v[28]), l as f32 + 1.25, "lane {l} v_cvt_f32_f64");
        let d = |n: usize| f64::from_bits(v[n] as u64 | (v[n + 1] as u64) << 32);
        let root = l as f64 + 1.25;
        let input = root * root;
        assert_eq!((d(30), d(32), d(34)), (input.ceil(), input.fract(), root), "lane {l} unary f64");
        assert!((d(36) - root.recip()).abs() < 1e-15, "lane {l} v_rsq_f64");
    }
    assert_eq!((r.sgpr[106], r.sgpr[107]), (0xFFFF_FFFF, 0), "v_cmp_lt_f16 → vcc");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x5555_5555, 0x5555_5555), "v_cmp_eq_u64 → sdst");
    let carries: u64 = (0..64).filter(|&l| (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(if l % 2 == 1 { u64::MAX } else { l as u64 }).1).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, carries, "v_mad_u64_u32 carry mask");
}

#[test]
fn gpu_packed_math_dots_movrel() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 2), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        // lo = src0.hi * src1.lo, hi = -(src0.lo * src1.hi)
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 1), ("OP_SEL_HI", 2), ("NEG_HI", 1)], None),
        ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 5), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_U32_U8", "ENC_VOP3P", &[("VDST", 6), ("SRC0", V + 7), ("SRC1", V + 8), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_I32_I8", "ENC_VOP3P", &[("VDST", 11), ("SRC0", V + 7), ("SRC1", V + 12), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_PK_FMAC_F16", "ENC_VOP2", &[("VDST", 13), ("SRC0", V), ("VSRC1", 1)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_MOVRELS_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 20)], None), // v10 = v23
        ("V_MOVRELD_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 9)], None), // v33 = v9
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 40), ("SRC0", V + 33)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[5] = f(10.0);
        r[7] = 0x0102_0304;
        r[8] = 0x0506_0708;
        r[9] = l as u32;
        r[12] = 0xFF06_0708; // -1, 6, 7, 8
        r[13] = h(1.0) | h(2.0) << 16;
        for k in 0..8 {
            r[20 + k] = 100 + k as u32;
        }
        r[33] = 0xDEAD;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], h(3.75) | h(-1.5) << 16, "lane {l} v_pk_add_f16");
        assert_eq!(v[3], h(-4.5) | h(-0.75) << 16, "lane {l} v_pk_mul_f16 op_sel / neg_hi");
        assert_eq!(f32::from_bits(v[4]), 12.375, "lane {l} v_dot2_f32_f16");
        assert_eq!(v[6], 70 + l as u32, "lane {l} v_dot4_u32_u8");
        assert_eq!(v[11] as i32, 4 * 8 + 3 * 7 + 2 * 6 - 1 + l as i32, "lane {l} v_dot4_i32_i8 (bytes from the low end)");
        assert_eq!(v[13], h(1.0 + 1.5 * 2.25) | h(2.0 - 2.0 * 0.5) << 16, "lane {l} v_pk_fmac_f16");
        assert_eq!(v[10], 103, "lane {l} v_movrels_b32");
        assert_eq!(v[40], l as u32, "lane {l} v_movreld_b32");
    }
}

#[test]
fn dxc_mesh_shader_probe() {
    use super::dxc;
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    let src = r#"
struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };
struct P { bool cull : SV_CullPrimitive; };
groupshared uint xcl_nv;
[WaveSize(64)]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint gi : SV_GroupIndex, out vertices V verts[128], out indices uint3 tris[128], out primitives P prims[128]) {
    if (gi == 0) xcl_nv = 3;
    GroupMemoryBarrierWithGroupSync();
    uint n = xcl_nv;
    SetMeshOutputCounts(n, 1);
    if (gi < n) { verts[gi].pos = float4(gi, 0, 0, 1); verts[gi].p0 = 0; }
    if (gi < 1) { tris[gi] = uint3(0, 1, 2); prims[gi].cull = false; }
}
"#;
    for prof in ["ms_6_6", "ms_6_8"] {
        let r = dxc::compile(src, "main", prof, &[]);
        println!("{prof}: {:?}", r.as_ref().map(|b| b.len()).map_err(|e| e.lines().take(3).collect::<Vec<_>>().join(" | ")));
    }
}

/// Mesh mode end to end: an NGG program in the console's style (wave index from s3, group
/// from v5 >> 7, GS_ALLOC_REQ, exp pos0/param0/prim, two waves per group) runs as a compute
/// shader, the pass-through mesh shader rasterises its records.
#[test]
fn gpu_mesh_mode_two_passes() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, MsInterface, MESH_HEADER};
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)], threads: 128, max_vertices: 128, max_primitives: 128 };
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None)
    };
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)), // wave index
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None), // thread in group
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None), // group = v5 >> 7
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None), // x += group
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None)); // thread 0
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = translate(&code, &on_gpu(Options { mode: Mode::Mesh(ms.clone()), wave64: true, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let pass = mesh_passthrough(&ms);
    let msb = dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass}"));
    let ps_src = "struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    // vertex table at byte 0x100, V# in s[16:19] (stride 16, 3 records)
    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[f(*x), f(*y), f(0.0), f(1.0)]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    let (px, rec) = runner::mesh_draw(&cs, &msb, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    let (rd, vd) = (ms.record_dwords(), ms.vert_dwords());
    for g in 0..2usize {
        let r = &rec[g * rd..(g + 1) * rd];
        assert_eq!((r[0], r[1]), (3, 1), "group {g}: GS_ALLOC_REQ counts");
        for v in 0..3usize {
            let b = MESH_HEADER + v * vd;
            let x = [-1.0f32, -1.0, 0.0][v] + g as f32;
            assert_eq!(f32::from_bits(r[b]), x, "group {g} vertex {v} pos.x");
            assert_eq!([f32::from_bits(r[b + 4]), f32::from_bits(r[b + 5]), f32::from_bits(r[b + 7])], [g as f32, v as f32, 1.0], "group {g} vertex {v} param0");
        }
        assert_eq!(r[MESH_HEADER + 128 * vd], 1 << 10 | 2 << 20, "group {g} primitive");
    }
    let at = |x: usize, y: usize| px[y * 32 + x];
    assert_eq!((at(3, 29)[0], at(3, 29)[3]), (0.0, 1.0), "group 0 triangle (left)");
    assert_eq!((at(19, 29)[0], at(19, 29)[3]), (1.0, 1.0), "group 1 triangle (right)");
    assert_eq!(at(29, 3), [-1.0; 4], "outside both triangles");

    // the same records through a pass-through for a pixel shader with 16 inputs: over the
    // mesh output limit (17 · 16 B · 128 vertices), so 4 mesh groups per record, each a run
    // of triangles with their own vertex copies; inputs the program does not write read 0
    let ins: Vec<crate::gpu::dxbc::SigElement> = (0..16).map(|k| crate::gpu::dxbc::SigElement { index: k, register: k + 1, ..sig("TEXCOORD", 0, 0) }).collect();
    let split = super::emit::passthrough_split(&ms, super::emit::passthrough_elements(&ms, Some(&ins)));
    assert_eq!(split, 4);
    let pass16 = super::emit::mesh_passthrough_for(&ms, Some(&ins));
    let msb16 = dxc::compile(&pass16, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass16}"));
    let fields: String = (0..16).map(|k| format!(" float4 t{k} : TEXCOORD{k};")).collect();
    let ps16 = dxc::compile(&format!("struct V {{ float4 pos : SV_Position;{fields} }};\nfloat4 main(V v) : SV_Target0 {{ return v.t0 + v.t15; }}\n"), "main", "ps_6_6", &[]).unwrap();
    let (px16, _) = runner::mesh_draw_split(&cs, &msb16, &ps16, 2, split, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert!(px16 == px, "split pass-through draws the same pixels");
}

#[test]
fn mesh_passthrough_budgets_whole_vertex_chunks() {
    use super::emit::{mesh_passthrough, passthrough_split};
    use super::MsInterface;
    let mut outputs = vec![el("SV_Position", 0, 1, 3, 0, 0xF)];
    outputs.extend((0..21).map(|i| el("TEXCOORD", i, 0, 3, i + 1, 0xF)));

    let mut ms = MsInterface { outputs, threads: 64, max_vertices: 64, max_primitives: 128 };
    assert_eq!(passthrough_split(&ms, 21), 1);
    assert!(mesh_passthrough(&ms).contains("verts[64]"));

    ms.max_vertices = 65;
    assert_eq!(passthrough_split(&ms, 21), 7);
    assert!(mesh_passthrough(&ms).contains("verts[57]"));

    ms.max_primitives = 1;
    assert_eq!(passthrough_split(&ms, 21), 1);
    assert!(mesh_passthrough(&ms).contains("verts[3]"));
}

/// DXC optimisation levels on a saved HLSL file (`XCL_TIME_HLSL=path`, profile in `XCL_TIME_PROFILE`).
#[test]
#[ignore]
fn dxc_time_levels() {
    use super::dxc;
    let Ok(path) = std::env::var("XCL_TIME_HLSL") else { return };
    let src = std::fs::read_to_string(path).unwrap();
    let prof = std::env::var("XCL_TIME_PROFILE").unwrap_or("cs_6_6".into());
    for args in [vec!["-O0"], vec!["-O1"], vec!["-O2"], vec!["-O3"]] {
        let t0 = std::time::Instant::now();
        let r = dxc::compile(&src, "main", &prof, &args);
        println!("{args:?}: {:.1} s, {:?}", t0.elapsed().as_secs_f64(), r.map(|b| b.len()).map_err(|e| e.chars().take(200).collect::<String>()));
    }
}

/// Minimal RGB8 PNG writer (zlib via miniz_oxide).
fn write_png(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
    fn crc(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    let chunk = |out: &mut Vec<u8>, ty: &[u8], d: &[u8]| {
        out.extend_from_slice(&(d.len() as u32).to_be_bytes());
        let mut td = ty.to_vec();
        td.extend_from_slice(d);
        out.extend_from_slice(&td);
        out.extend_from_slice(&crc(&td).to_be_bytes());
    };
    let mut raw = Vec::with_capacity((w * 3 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6));
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out).unwrap();
}

/// Retail pixel shaders (no textures) translated and drawn full screen on the GPU:
/// attributes = screen uv, constants from a pattern. `XCL_SAMPLE_RPF`, `XCL_GALLERY_DIR`.
#[test]
#[ignore]
fn retail_ps_gallery() {
    use super::{dxc, ranked};
    let (Ok(path), Ok(dir)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_GALLERY_DIR")) else { return };
    if !gpu_ready() {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = std::fs::read(&path).unwrap();
    // descriptor table at 0x100: every V# points at the constants at 0x4000
    let mut mem = vec![0u32; 0x3000];
    for i in 0..256 {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[0x4000, 0, 0x8000, 0x27FAC]);
    }
    for i in 0..0x2000 {
        mem[0x1000 + i] = f(0.5 + 0.45 * ((i as f32) * 0.37).sin());
    }
    let mut io = vec![0u32; 128];
    for k in (0..106).step_by(2) {
        io[k] = 0x100;
    }
    let size = 160u32;
    let t0 = std::time::Instant::now();
    let (mut tried, mut kept) = (0, Vec::new());
    for rec in crate::gpu::pack::scan(&data, 0x40) {
        if kept.len() >= 24 || t0.elapsed().as_secs() > 150 {
            break;
        }
        if !rec.name.starts_with("PS_") {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &rec) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(fs) = sh.fssm.first() else { continue };
        let Some(p) = fs.programs.first() else { continue };
        let Some(mode) = super::interface(&blob, &sh, fs) else { continue };
        let Mode::Pixel(ps) = mode.clone() else { continue };
        if !ps.outputs.iter().any(|e| e.system_value == 64 && e.index == 0) {
            continue;
        }
        let code = &fs.data[p.offset..p.offset + p.len];
        let Ok(t) = translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }) else { continue };
        if !t.complete() || t.hlsl.contains("ResourceDescriptorHeap") || t.hlsl.len() > 150_000 {
            continue;
        }
        tried += 1;
        let mut vs = String::from("struct O {\n    float4 pos : SV_Position;\n");
        let ins = ranked(&ps.inputs, false);
        for (k, e) in ins.iter().enumerate() {
            let ty = match e.component_type {
                1 => "uint",
                2 => "int",
                _ => "float",
            };
            vs.push_str(&format!("    {ty}4 a{k} : {};\n", e.semantic()));
        }
        vs.push_str("};\nO main(uint id : SV_VertexID) {\n    O o;\n    float2 uv = float2((id << 1) & 2, id & 2);\n");
        vs.push_str("    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0.5, 1);\n");
        for (k, e) in ins.iter().enumerate() {
            let v = match e.component_type {
                1 => "uint4(uv * 255.0, 0, 1)",
                2 => "int4(uv * 255.0, 0, 1)",
                _ => "float4(uv, 1.0 - uv.x, 1.0)",
            };
            vs.push_str(&format!("    o.a{k} = {v};\n"));
        }
        vs.push_str("    return o;\n}\n");
        let (Ok(vsb), Ok(psb)) = (dxc::compile(&vs, "main", "vs_6_6", &[]), dxc::compile(&t.hlsl, "main", "ps_6_6", &[])) else { continue };
        let px = match runner::draw(&vsb, &psb, &[], &[0.0f32; 3], 4, &io, &mem, size, [0.0, 0.0, 0.0, 1.0]) {
            Ok(px) => px,
            Err(e) => {
                println!("{}: {e}", rec.name);
                break; // device trouble: stop
            }
        };
        let rgb: Vec<f32> = px.iter().flat_map(|c| [c[0], c[1], c[2]]).map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
        let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
        let var = rgb.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / rgb.len() as f32;
        if var < 0.002 {
            continue; // flat colour: not worth a picture
        }
        let bytes: Vec<u8> = rgb.iter().map(|x| (x.powf(1.0 / 2.2) * 255.0).round() as u8).collect();
        let file = format!("{:02}_{}.png", kept.len(), rec.name.split('_').take(4).collect::<Vec<_>>().join("_"));
        write_png(&dir.join(&file), size, size, &bytes);
        println!("{file}: {} instructions, variance {var:.3}", t.instructions);
        kept.push(format!("{file}\t{}\t{}", rec.name, t.instructions));
    }
    std::fs::write(dir.join("index.txt"), kept.join("\n")).unwrap();
    println!("tried {tried}, saved {} in {:.0} s", kept.len(), t0.elapsed().as_secs_f64());
}

#[test]
fn gpu_image_sample_offset() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut code = encode("IMAGE_SAMPLE_LZ_O", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        r[0] = 1 | 0x3F << 8; // offset (+1, -1)
        r[1] = f((x + 0.5) / 4.0);
        r[2] = f((y + 0.5) / 4.0);
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, 0.0, 1.0] {
                t0.extend_from_slice(&f32::to_le_bytes(v));
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let got = [f32::from_bits(r.vgpr[l][10]), f32::from_bits(r.vgpr[l][11])];
        assert_eq!(got, [(x + 1.0).min(3.0), (y - 1.0).max(0.0)], "lane {l} sample_lz_o");
    }
}

/// Console unified memory: the GPU reads constants and buffers straight from process memory
/// (4 GiB-aligned arena opened as a D3D12 heap) and its stores land there, no copies.
#[test]
fn gpu_unified_memory_arena() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::new(1 << 20).unwrap();
    let va = arena.base() as u64;
    assert_eq!(va & 0xFFFF_FFFF, 0, "4 GiB alignment");
    let words = || unsafe { std::slice::from_raw_parts_mut(arena.base() as *mut u32, arena.size() / 4) };
    let w = words();
    w[0x1000 / 4] = 0xC0FFEE;
    w[0x1000 / 4 + 1] = 0x1234;
    for l in 0..64 {
        w[0x2000 / 4 + l] = 1000 + l as u32;
    }
    let p = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] = console address
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let t = translate(&p, &on_gpu(Options::default())).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let hi = (va >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(va + 0x1000) as u32, hi]);
    st.sgpr[8..12].copy_from_slice(&[(va + 0x2000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(va + 0x3000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[2] = 7 * l as u32;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from process memory");
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], 1000 + l as u32, "lane {l} buffer_load from process memory");
        assert_eq!(words()[0x3000 / 4 + l], 7 * l as u32, "lane {l} buffer_store visible to the CPU");
    }
}

/// An NGG program (mesh mode) reading title memory above the first window: its vertex table
/// is in window 1, decoys sit at the same offset in window 0. Translated with the title's
/// windows (`translate_mesh`'s `mem_windows`, as `pso::console_mesh` now does) it reads window
/// 1; with one window – what MS / GS programs got before – it read the decoys.
#[test]
fn gpu_mesh_mode_reads_memory_windows() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, MsInterface, MESH_HEADER};
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0)], threads: 64, max_vertices: 64, max_primitives: 64 };
    let list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None), // thread
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("EXP", "ENC_EXP", vec![("TGT", 12), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 11), ("VSRC2", 12), ("VSRC3", 13), ("DONE", 1)], None),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        for (k, v) in [*x, *y, 0.0, 1.0].iter().enumerate() {
            *word(w1 + 0x100 + 16 * i as u64 + 4 * k as u64) = f(*v);
            *word(w0 + 0x100 + 16 * i as u64 + 4 * k as u64) = f(9.0); // must not be read
        }
    }
    let mut st = KernelState::default();
    // V#: the vertex table in window 1, stride 16, 3 records
    st.sgpr[16..20].copy_from_slice(&[(w1 + 0x100) as u32, ((w1 >> 32) as u32) & 0xFFFF | 16 << 16, 3, 0]);
    let vd = ms.vert_dwords();
    let positions = |windows: u32| -> Vec<[f32; 4]> {
        let t = super::translate_mesh(&code, ms.clone(), true, runner::split_wave64(), windows).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let r = runner::run_in_arena(&cs, &st, &arena).unwrap();
        assert_eq!((r.raw[0], r.raw[1]), (3, 1), "GS_ALLOC_REQ counts ({windows} windows)");
        (0..3).map(|v| std::array::from_fn(|c| f32::from_bits(r.raw[MESH_HEADER + v * vd + c]))).collect()
    };
    assert_eq!(positions(2), vec![[-1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]], "the vertex table from window 1");
    assert_eq!(positions(1)[0], [9.0; 4], "one window: window 0 at the same offset (the old MS / GS translation)");
}

/// Memory above 4 GiB: two UMA windows 4 GiB apart; every access kind picks its window from
/// the high address bits (V#, SGPR pair, 64-bit VGPR address, SADDR pair) and reaches the
/// right one – s_load, buffer load / store, global load / store / atomic in window 1 next to
/// accesses in window 0.
#[test]
fn gpu_memory_windows_above_4_gib() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    *word(w1 + 0x1000) = 0xC0FFEE;
    *word(w1 + 0x1004) = 0x1234;
    for l in 0..64u64 {
        *word(w1 + 0x2000 + 4 * l) = 1000 + l as u32;
        *word(w1 + 0x4000 + 4 * l) = 5000 + l as u32;
        *word(w0 + 0x4000 + 4 * l) = 9; // same low address in window 0: must not be read
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        ("S_LOAD_DWORDX2", "ENC_SMEM", vec![("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] → window 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 1
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 24), ("SBASE", 4), ("OFFSET", 4)], None), // that V#: one switch, 4 dwords
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 0
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 6), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[4:5] → window 1
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 8), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[8:9] → window 1
        glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 10), ("ADDR", 3), ("DATA", 11), ("SADDR", 16), ("OFFSET", 0), ("GLC", 1)]), // s[16:17] + v3 → window 1
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let one = translate(&p, &Options::default()).unwrap();
    assert!(!one.hlsl.contains("xcl_w =") && !one.hlsl.contains("min(xcl_w") && one.hlsl.contains("ResourceDescriptorHeap[xcl_mem_views]"), "one window: no window selection");
    let t = translate(&p, &on_gpu(Options { mem_windows: 2, ..Default::default() })).unwrap();
    let earlier = t.hlsl.contains("ResourceDescriptorHeap[xcl_mem_views + xcl_w]") && t.hlsl.contains("ResourceDescriptorHeap[NonUniformResourceIndex(xcl_mem_views + xcl_wv)]");
    let compact = t.hlsl.contains("select(f, xcl_mem_views + xcl_w, p.x)") && t.hlsl.contains("select(f, xcl_mem_views + xcl_wv, p.x)");
    assert!((earlier || compact) && t.hlsl.contains("_w < 2u, "), "{}", t.hlsl);
    assert!(!t.hlsl.contains("register(u2)") && !t.hlsl.contains("switch (xcl_w)"));
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let hi = |a: u64| (a >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(w1 + 0x1000) as u32, hi(w1)]);
    st.sgpr[8..12].copy_from_slice(&[(w1 + 0x2000) as u32, hi(w1) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(w0 + 0x3000) as u32, hi(w0) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[16..18].copy_from_slice(&[(w1 + 0x6000) as u32, hi(w1)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u64;
        r[0] = l as u32;
        r[2] = 7 * l as u32;
        r[3] = 0; // atomic offset: all lanes on one counter
        r[4] = (w1 + 0x4000 + 4 * l) as u32;
        r[5] = hi(w1);
        r[8] = (w1 + 0x5000 + 4 * l) as u32;
        r[9] = hi(w1);
        r[11] = 1;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from window 1");
    assert_eq!(r.sgpr[24..28], [1001, 1002, 1003, 1004], "s_buffer_load_dwordx4 from window 1");
    for l in 0..64u64 {
        assert_eq!(r.vgpr[l as usize][1], 1000 + l as u32, "lane {l}: buffer_load from window 1");
        assert_eq!(r.vgpr[l as usize][6], 5000 + l as u32, "lane {l}: global_load from window 1");
        assert_eq!(*word(w0 + 0x3000 + 4 * l), 7 * l as u32, "lane {l}: buffer_store into window 0");
        assert_eq!(*word(w1 + 0x5000 + 4 * l), 7 * l as u32, "lane {l}: global_store into window 1");
    }
    assert_eq!(*word(w1 + 0x6000), 64, "global_atomic_add in window 1");
    assert_eq!(*word(w0 + 0x6000), 0, "nothing in window 0 at the same low address");
}

#[test]
fn gpu_page_directory_carry_scatter_alias_and_override_uma() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0, arena.window(1).0);
    let views = runner::mem_views_at(0);
    let directory = views + super::MEM_NULL_VIEW + 2;
    let uma_hi = (w0 >> 32) as u32;
    let hi = uma_hi + 32;
    let mut tables = vec![vec![0u32; 65536 * 2], vec![0; 65536], vec![0; 65536], vec![0; 65536], vec![0; 65536]];
    for high in [hi, hi + 1] {
        tables[0][2 * high as usize..2 * high as usize + 2].copy_from_slice(&[directory + 1, directory + 2]);
    }
    tables[0][2 * uma_hi as usize..2 * uma_hi as usize + 2].copy_from_slice(&[directory + 3, directory + 4]);
    let pte = |view: u32, page: u32| 0x80000000 | page << 20 | view;
    tables[1][65535] = pte(views, 4);
    tables[2][0] = pte(views + 1, 7);
    tables[1][8] = pte(views + 1, 7);
    tables[1][7] = 0x80000000;
    tables[2][7] = pte(views + 1, 7);
    tables[3][9] = pte(views + 1, 7);
    tables[4][9] = pte(views, 4);
    tables[3][10] = 0x80000000;
    tables[4][10] = pte(views, 4);
    tables[4][11] = pte(views, 4);
    let word = |p: usize| unsafe { &mut *(p as *mut u32) };
    *word(w0 + 0x4fffc) = 0x44332211;
    *word(w0 + 0x50000) = 0xdeadbeef;
    *word(w1 + 0x70000) = 0x88776655;
    *word(w0 + 0x40000) = 0xcafef00d;
    *word(w0 + 0xa0000) = 0xdeadbeef;
    *word(w0 + 0xc0000) = 0xaabbccdd;
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[0xfffffffc, hi, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[8 * 65536 + 16, hi + 1, 64, 0]);
    st.sgpr[16..20].copy_from_slice(&[7 * 65536 + 16, hi + 1, 64, 0]);
    st.sgpr[20..24].copy_from_slice(&[0x90000, uma_hi, 4 * 65536, 0]);
    st.sgpr[28..32].copy_from_slice(&[0x90000, 0, 64, 0]);
    tables.insert(0, vec![directory]);
    let run = |code: &[u8], st: &KernelState| {
        let t = translate(code, &Options { wave64: false, mem_windows: 2, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        runner::run_arena_kernel(&dxil, st, &arena, &tables).unwrap()
    };
    for (i, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = (i as u32 & 1) * 4;
        r[7] = 1;
        r[8] = 1;
        r[4] = (i as u32 & 3) * 65536;
        r[10] = 1;
    }
    let r = run(&prog(&[
        ("S_BUFFER_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 4), ("SOFFSET", 125)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("BUFFER_LOAD_USHORT", "ENC_MUBUF", &[("VDATA", 3), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", 3)], None),
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 7), ("SRSRC", 3), ("SOFFSET", 128), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 8), ("SRSRC", 4), ("SOFFSET", 128), ("GLC", 1)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 9), ("SRSRC", 5), ("SOFFSET", 128), ("VADDR", 4), ("OFFEN", 1)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 36), ("SBASE", 14), ("SOFFSET", 125)], None),
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 7), ("SOFFSET", 128), ("GLC", 1)], None),
    ]), &st);
    assert_eq!(&r.sgpr[24..26], &[0x44332211, 0x88776655]);
    assert_eq!(r.sgpr[36], 0x88776655);
    for (i, lane) in r.vgpr[..32].iter().enumerate() {
        assert_eq!(lane[2], if i % 2 == 0 { 0x44332211 } else { 0x88776655 });
        assert_eq!(lane[3], 0x5544);
        assert_eq!(lane[8], 0);
        assert_eq!(lane[9], [0x88776655, 0, 0xcafef00d, 0xaabbccdd][i % 4]);
        assert_eq!(lane[10], 0);
    }
    assert_eq!(*word(w1 + 0x70010), 32);
    st.sgpr[126] = 1;
    st.vgpr[0][1] = 0xabcd;
    st.vgpr[0][2] = u32::MAX;
    let r = run(&prog(&[
        ("BUFFER_STORE_SHORT", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", 3)], None),
        ("BUFFER_ATOMIC_ADD_X2", "ENC_MUBUF", &[("VDATA", 2), ("SRSRC", 2), ("SOFFSET", 128), ("GLC", 1)], None),
    ]), &st);
    assert_eq!(&r.vgpr[0][2..4], &[0, 0]);
    assert_eq!(*word(w0 + 0x4fffc), 0xcd332211);
    assert_eq!(*word(w1 + 0x70000), 0x887766ab);
    assert_eq!(*word(w0 + 0x50000), 0xdeadbeef);
}

/// Out-of-UMA addresses must NEVER alias onto live UMA windows via `& 3`:
/// - high dwords outside `[base_hi, base_hi + mem_windows)` (such as `base_hi + 4` / `+ 5` /
///   `base_hi - 1` / `0xFFFFFFFF` that share low bits with live windows) drop stores & atomics
///   (including CAS-loop atomics like `global_atomic_inc`) and read 0;
/// - `hi == 0` reads still reach window 0 (32-bit console descriptor tables), while `hi == 0`
///   writes are dropped so null pointers cannot clobber window 0;
/// - V# stride bits `d1[29:16]` are masked off before window selection, and 64-bit carry/borrow
///   on `global_load`/`global_store` updates the target window accurately (a 64-bit VGPR address
///   per lane; an SGPR base and its offset select the window for the wave).
#[test]
fn gpu_memory_windows_reject_out_of_uma_aliasing() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let base_hi = (w0 >> 32) as u32;
    assert!(base_hi >= 4);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };

    // Canaries in window 0 and window 1 that out-of-UMA stores/atomics must NOT touch
    const CANARY0: u32 = 0xCAFE_BABE;
    const CANARY1: u32 = 0x1337_BEEF;
    *word(w0 + 0x1000) = CANARY0;
    *word(w1 + 0x1000) = CANARY1;
    *word(w0 + 0x2000) = CANARY0;
    *word(w1 + 0x2000) = CANARY1;
    *word(w0 + 0x3000) = CANARY0;
    *word(w1 + 0x3000) = CANARY1;
    *word(w0 + 0x4000) = 0xABCD_0123; // 32-bit hi=0 descriptor-table style read target
    *word(w1 + 0x5000) = 0x5566_7788; // valid strided V# buffer_load target
    *word(w1 + 0x0010) = 0x99AA_BBCC; // 64-bit carry target: (w0 + 0xFFFF_FFF0) + 0x20 == w1 + 0x10

    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        // 1. Out-of-UMA s_load from hi = base_hi + 4 (would alias w0 under `& 3`) -> must read 0
        ("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 30), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3]
        // 2. 32-bit hi == 0 s_load at offset 0x4000 -> MUST read w0 + 0x4000
        ("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 31), ("SBASE", 2), ("OFFSET", 0)], None), // s[4:5]
        // 3. Out-of-UMA buffer_store with V# hi = base_hi + 5 and stride = 4 -> must NOT touch w1 + 0x1000
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None), // s[8:11]
        // 4. Out-of-UMA buffer_load with V# hi = base_hi + 5 and stride = 4 -> must read 0 (v2)
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // 5. Valid strided V# buffer_load from w1 + 0x5000 (d1 has non-zero stride 4 in bits 29:16) -> must read 0x5566_7788 (v3)
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None), // s[12:15]
        // 6. Out-of-UMA global_store with hi = base_hi + 4 (would alias w0 + 0x1000) -> dropped
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 4), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0)]),
        // 7. Out-of-UMA global_store with hi = 0 (null/32-bit pointer at 0x2000) -> dropped, must NOT touch w0 + 0x2000
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0)]),
        // 8. Out-of-UMA global_atomic_add (hi = base_hi + 5) & global_atomic_inc CAS loop (hi = 0xFFFF_FFFF) -> dropped & terminates cleanly
        glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 10), ("ADDR", 8), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]),
        glbl("GLOBAL_ATOMIC_INC", &[("VDST", 11), ("ADDR", 12), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]),
        // 9. Out-of-UMA buffer_atomic_add with V# hi = base_hi + 4 (would alias w0 + 0x3000) -> dropped
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", vec![("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1), ("GLC", 1)], None), // s[16:19]
        // 10. 64-bit carry on global_load: v[16:17] = w0 + 0xFFFF_FFF0, + 0x20 -> w1 + 0x10 -> must read 0x99AA_BBCC (v19)
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 19), ("ADDR", 16), ("SADDR", 0x7D), ("OFFSET", 0x20)]),
        // 11. 64-bit signed borrow out of UMA: v[24:25] = w0 + 0x0 + (-16) -> (base_hi - 1, 0xFFFF_FFF0) -> dropped, reads 0 (v18)
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 18), ("ADDR", 24), ("SADDR", 0x7D), ("OFFSET", 0xFF0)]),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let t = translate(&p, &on_gpu(Options { mem_windows: 2, ..Default::default() })).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();

    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[0x1000, base_hi + 4]); // s[2:3]: out-of-UMA hi = base_hi + 4
    st.sgpr[4..6].copy_from_slice(&[0x4000, 0]); // s[4:5]: hi == 0 32-bit descriptor read
    st.sgpr[8..12].copy_from_slice(&[0x1000, ((base_hi + 5) & 0xFFFF) | (4 << 16), 64, 0]); // out-of-UMA V# (aliases w1 under & 3)
    st.sgpr[12..16].copy_from_slice(&[0x5000, ((base_hi + 1) & 0xFFFF) | (4 << 16), 64, 0]); // valid strided V# in w1
    st.sgpr[16..20].copy_from_slice(&[0x3000, ((base_hi + 4) & 0xFFFF) | (4 << 16), 64, 0]); // out-of-UMA atomic V# (aliases w0 under & 3)
    for r in st.vgpr.iter_mut() {
        r[0] = 0; // index 0
        r[1] = 0xBAD0_DEAD; // poison value to store
        r[2] = 0x7777_7777; // sentinel overwritten to 0 by out-of-UMA buffer_load
        r[10] = 0x7777_7777;
        r[11] = 0x7777_7777;
        r[14] = 0xBAD0_DEAD;
        r[4] = 0x1000;
        r[5] = base_hi + 4; // out-of-UMA global_store (aliases w0 under & 3)
        r[6] = 0x2000;
        r[7] = 0; // hi == 0 global_store (null pointer -> must be dropped)
        r[8] = 0x2000;
        r[9] = base_hi + 5; // out-of-UMA global_atomic_add (aliases w1 under & 3)
        r[12] = 0x3000;
        r[13] = 0xFFFF_FFFF; // out-of-UMA global_atomic_inc CAS loop
        r[16] = 0xFFFF_FFF0; // carry: (base_hi, 0xFFFF_FFF0) + 0x20 = w1 + 0x10
        r[17] = base_hi;
        r[24] = 0; // borrow: (base_hi, 0) - 16 = (base_hi - 1, 0xFFFF_FFF0)
        r[25] = base_hi;
        r[18] = 0x7777_7777;
    }

    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!(r.sgpr[30], 0, "out-of-UMA s_load must return 0");
    assert_eq!(r.sgpr[31], 0xABCD_0123, "hi == 0 s_load must still read window 0 (32-bit descriptor tables)");
    assert_eq!(r.vgpr[0][2], 0, "out-of-UMA buffer_load must return 0");
    assert_eq!(r.vgpr[0][3], 0x5566_7788, "valid strided V# buffer_load must mask stride out of d1");
    assert_eq!(r.vgpr[0][10], 0, "out-of-UMA global_atomic_add GLC must return 0");
    assert_eq!(r.vgpr[0][11], 0, "out-of-UMA global_atomic_inc CAS loop must return 0 and terminate");
    assert_eq!(r.vgpr[0][14], 0, "out-of-UMA buffer_atomic_add GLC must return 0");
    assert_eq!(r.vgpr[0][19], 0x99AA_BBCC, "64-bit carry global_load must cross from w0 into w1");
    assert_eq!(r.vgpr[0][18], 0, "64-bit signed borrow below window 0 must be rejected and read 0");

    assert_eq!(*word(w0 + 0x1000), CANARY0, "out-of-UMA global_store must not clobber w0");
    assert_eq!(*word(w1 + 0x1000), CANARY1, "out-of-UMA buffer_store must not clobber w1");
    assert_eq!(*word(w0 + 0x2000), CANARY0, "hi == 0 global_store must not clobber w0");
    assert_eq!(*word(w1 + 0x2000), CANARY1, "out-of-UMA global_atomic_add must not clobber w1");
    assert_eq!(*word(w0 + 0x3000), CANARY0, "out-of-UMA buffer_atomic_add must not clobber w0");
    assert_eq!(*word(w1 + 0x3000), CANARY1, "out-of-UMA global_atomic_inc must not clobber w1");
}

/// The start-up check of the coherent UMA windows (`real::check_uma_import`): the GPU sees what
/// the CPU writes there. Fails, as it should, where the import fell back to memory of its own
/// (vkd3d-proton under a Wine without `userfaultfd` write watches, #175).
#[test]
fn gpu_uma_import_check_at_init() {
    if !gpu_ready() {
        return;
    }
    crate::hle::d3d12x::real::init().expect("real D3D12.X backend");
    let s = crate::hle::d3d12x::real::stats().uma_import_check;
    assert!(s.starts_with("ok") || s.starts_with("off"), "{s}");
}

/// End-to-end GPU test of the sparse-window VRAM runtime layout:
/// - Window 0..2 = sparse VRAM (tiled reserved buffers on DEFAULT heaps)
/// - Window 3 = committed coherent RAM (`OpenExistingHeapFromAddress`)
/// Exercises a DXC-compiled SM 6.6 translated RDNA2 compute shader (`mem_windows = 4`)
/// reading a 64-bit spilled root-argument constant from Window 3 (`s[0:1]`), loading two
/// V# buffer descriptors through a 32-bit `hi = 0` descriptor table pointer in Window 0
/// sparse VRAM (`console_heap`), reading input from `XMemAlloc` graphics memory in sparse
/// VRAM, writing to Window 3, and then mutating both the buffer and the descriptor slot in
/// Window 0 sparse VRAM between submits.
#[test]
fn gpu_sparse_vram_descriptors_buffers_and_xmem() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;

    if !gpu_ready() {
        return;
    }
    crate::hle::d3d12x::real::init().expect("real D3D12.X backend");
    let u = crate::gpu::uma::get().expect("UMA");
    assert_eq!(u.shader_windows(), 4, "runtime layout has 4 shader windows");
    assert!(u.is_sparse_window(0) && u.is_sparse_window(1) && u.is_sparse_window(2), "windows 0..2 sparse VRAM");
    assert!(u.is_committed_window(3), "window 3 coherent committed RAM");
    let windows = crate::hle::d3d12x::real::uma_window_vas();
    assert_eq!(windows.len(), 4);
    assert!(windows.iter().all(|&va| va != 0), "all 4 window GPU VAs must be valid: {windows:x?}");

    // 1. Console descriptor heap in Window 0 sparse VRAM (32-bit hi = 0 table addressing)
    let heap_obj = 0xDEAD_5EA7usize;
    assert!(crate::hle::d3d12x::console_heap::register(heap_obj, 4, 0x1000, 0x5000, 32, false, false, true));
    let heap_va = crate::hle::d3d12x::console_heap::start(heap_obj).unwrap() as usize;
    assert_eq!(u.window_of(heap_va), Some(0), "descriptor heap must live in window 0");
    assert!(u.is_sparse(heap_va), "descriptor heap must live in sparse VRAM");

    // 2. XMem graphics allocations in sparse VRAM
    let gfx_attrs = 1u64 << 29;
    let in_a = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    let in_b = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    assert!(in_a != 0 && in_b != 0 && u.is_sparse(in_a) && u.is_sparse(in_b), "XMem graphics buffers must live in sparse VRAM");

    // 3. Coherent Window 3 allocations: spilled root argument table + output buffer + io/res
    let arg_table = u.alloc(0x1_0000, 256).unwrap() as usize;
    let out_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let io_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let res_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    assert_eq!(u.window_of(arg_table), Some(3));
    assert_eq!(u.window_of(out_buf), Some(3));

    // Populate initial data & descriptors (slot 0 = V# in_a, slot 1 = V# out_buf)
    let vsharp = |addr: usize| -> [u32; 4] { [addr as u32, (((addr >> 32) as u32) & 0xFFFF) | (4 << 16), 64, 0] };
    unsafe {
        *(arg_table as *mut u32) = 77;
        for l in 0..64usize {
            *((in_a + 4 * l) as *mut u32) = 1000 + l as u32;
            *((in_b + 4 * l) as *mut u32) = 5000 + 2 * l as u32;
        }
    }
    let slot0 = crate::hle::d3d12x::console_heap::slot(heap_va as u64).unwrap();
    let slot1 = crate::hle::d3d12x::console_heap::slot(heap_va as u64 + 32).unwrap();
    crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_a));
    crate::hle::d3d12x::console_heap::write(&slot1, &vsharp(out_buf));

    // Translate + DXC-compile RDNA2 compute shader (4 windows, 32-bit descriptor table with hi = 0)
    let p = prog(&[
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 0), ("OFFSET", 0)], None), // s[0:1] -> Window 3 root arg table
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)], None), // s5 = 0 (32-bit descriptor table pointer)
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0)], None), // s[4:5] + 0 -> Window 0 sparse desc slot 0
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 32)], None), // s[4:5] + 32 -> Window 0 sparse desc slot 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 20), ("VSRC1", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 6, ..Default::default() };
    let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), mem_windows: 4, ..Default::default() })).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();

    // Execute on the real D3D12.X device + queue (goes through q_execute_command_lists -> sparse::before_execute)
    unsafe {
        let dev_raw = crate::hle::d3d12x::real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&dev_raw).expect("device").clone();
        let rs = crate::gpu::recomp::pso::abi_root_signature(&dev, false).expect("ABI root signature");
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .expect("CreateComputePipelineState");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

        // User SGPRs: s[0:1] = 64-bit arg_table (Window 3), s4 = 32-bit descriptor table offset in Window 0
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
        io[crate::gpu::recomp::IO_MEM_VIEWS] = crate::hle::d3d12x::real::mem_views().expect("memory window views");
        io[0] = arg_table as u32;
        io[1] = (arg_table >> 32) as u32;
        io[4] = (heap_va - u.base()) as u32;
        io[5] = 0xDEAD_BEEF; // shader overwrites s5 = 0 to prove hi == 0 32-bit addressing hits Window 0 sparse VRAM
        std::ptr::copy_nonoverlapping(io.as_ptr() as *const u8, io_buf as *mut u8, io.len() * 4);

        let record_and_run = |fence_val: u64, first: bool| {
            if !first {
                alloc.Reset().unwrap();
                list.Reset(&alloc, Some(&pso)).unwrap();
            }
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, crate::hle::d3d12x::real::uma_va_of(io_buf).unwrap());
            list.SetComputeRootUnorderedAccessView(1, crate::hle::d3d12x::real::uma_va_of(res_buf).unwrap());
            list.SetDescriptorHeaps(&[crate::hle::d3d12x::console_heap::global_heap()]);
            list.Dispatch(1, 1, 1);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, fence_val).unwrap();
            if fence.GetCompletedValue() < fence_val {
                let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(fence_val, ev).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
        };

        // Submit 1: reads slot 0 -> in_a (sparse VRAM) + arg_table constant 77 (Window 3)
        record_and_run(1, true);
        for l in 0..64usize {
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, 1000 + l as u32 + 77, "submit 1 lane {l} mismatch");
        }

        // Submit 2: mutate descriptor slot 0 in Window 0 sparse VRAM -> in_b, and mutate in_b[3] + arg_table
        *(arg_table as *mut u32) = 13;
        *((in_b + 4 * 3) as *mut u32) = 9999;
        crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_b));
        record_and_run(2, false);
        for l in 0..64usize {
            let expected = if l == 3 { 9999 + 13 } else { 5000 + 2 * l as u32 + 13 };
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, expected, "submit 2 (mutated sparse descriptor + XMem buffer) lane {l} mismatch");
        }

        // Clean up
        crate::hle::d3d12x::console_heap::unregister(heap_obj);
        crate::hle::kernelx::XMemFreeDefault(in_a as *mut _, gfx_attrs);
        crate::hle::kernelx::XMemFreeDefault(in_b as *mut _, gfx_attrs);
        assert!(u.free(arg_table) && u.free(out_buf) && u.free(io_buf) && u.free(res_buf));
    }
}

/// Experiment: does OpenExistingHeapFromAddress accept a reservation with only part committed?
#[test]
#[ignore]
fn gpu_arena_partial_commit_probe() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows_sys::Win32::System::Memory::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev3: ID3D12Device3 = dev.unwrap().cast().unwrap();
        let size = 256usize << 20;
        let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE);
        let c = VirtualAlloc(p, 1 << 20, MEM_COMMIT, PAGE_READWRITE);
        println!("reserve {p:?} commit {c:?}");
        let r: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(p);
        println!("partial commit: {:?}", r.as_ref().map(|h| h.GetDesc().SizeInBytes));
        let q = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        let r2: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(q);
        println!("full commit 256 MB: {:?}", r2.as_ref().map(|h| h.GetDesc().SizeInBytes));
        for mb in [1024usize, 2047, 2048, 3072, 4095] {
            let big = mb << 20;
            let b = VirtualAlloc(std::ptr::null(), big, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            let t0 = std::time::Instant::now();
            let r3: windows::core::Result<ID3D12Heap> = if b.is_null() { Err(windows::core::Error::from_thread()) } else { dev3.OpenExistingHeapFromAddress(b) };
            println!("full commit {mb} MB: {:?} in {:.2} s", r3.as_ref().map(|h| h.GetDesc().SizeInBytes), t0.elapsed().as_secs_f64());
            drop(r3);
            VirtualFree(b, 0, MEM_RELEASE);
        }
    }
}

/// Experiment (vram61): what a heap opened from an address costs in RAM – before and after the
/// open, after the working set is emptied, with a buffer placed, evicted – and whether the tiles
/// of a reserved buffer can be mapped onto such heaps (a window committed in pieces).
#[test]
#[ignore]
fn gpu_imported_heap_residency_probe() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows_sys::Win32::System::Memory::*;
    #[repr(C)]
    struct MemStatus {
        len: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        rest: [u64; 5],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn K32QueryWorkingSetEx(process: isize, pv: *mut [u64; 2], cb: u32) -> i32;
        fn K32EmptyWorkingSet(process: isize) -> i32;
        fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
        fn GetLastError() -> u32;
    }
    const MIB: usize = 1 << 20;
    // (resident MiB, with the locked bit MiB) of a range, MiB of RAM available to the system
    unsafe fn look(p: usize, len: usize) -> (usize, usize, u64) {
        let mut v: Vec<[u64; 2]> = (0..len / 4096).map(|i| [(p + i * 4096) as u64, 0]).collect();
        let (mut valid, mut locked) = (0usize, 0usize);
        for c in v.chunks_mut(1 << 14) {
            if K32QueryWorkingSetEx(-1, c.as_mut_ptr(), (c.len() * 16) as u32) == 0 {
                return (usize::MAX, usize::MAX, 0);
            }
            for e in c.iter() {
                if e[1] & 1 != 0 {
                    valid += 1;
                    locked += usize::from(e[1] & 0x40_0000 != 0);
                }
            }
        }
        let mut m = MemStatus { len: 64, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
        GlobalMemoryStatusEx(&mut m);
        (valid * 4096 / MIB, locked * 4096 / MIB, m.avail_phys >> 20)
    }
    unsafe fn trim() {
        K32EmptyWorkingSet(-1);
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev = dev.unwrap();
        let dev3: ID3D12Device3 = dev.cast().unwrap();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).unwrap();
        let video = || {
            let (mut l, mut n) = (DXGI_QUERY_VIDEO_MEMORY_INFO::default(), DXGI_QUERY_VIDEO_MEMORY_INFO::default());
            let _ = adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut l);
            let _ = adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut n);
            (l.CurrentUsage >> 20, n.CurrentUsage >> 20)
        };
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let mut fence_value = 0u64;
        let buffer = |bytes: usize, flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &buffer(4096, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
        let readback = readback.unwrap();
        // copies of 4 bytes (source, offset) to the readback buffer; what the GPU read, or why not
        let mut read = |srcs: &[(&ID3D12Resource, usize)]| -> Result<Vec<u32>, String> {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
            for (i, (r, at)) in srcs.iter().enumerate() {
                list.CopyBufferRegion(&readback, 4 * i as u64, *r, *at as u64, 4);
            }
            list.Close().map_err(|e| format!("Close: {e}"))?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            fence_value += 1;
            queue.Signal(&fence, fence_value).map_err(|e| format!("Signal: {e}"))?;
            crate::gpu::fence::wait(&fence, fence_value, &dev, None, 5000).map_err(|e| format!("wait: {e}"))?;
            let mut m = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut m)).map_err(|e| format!("Map: {e}"))?;
            let got = std::slice::from_raw_parts(m as *const u32, srcs.len()).to_vec();
            readback.Unmap(0, None);
            Ok(got)
        };
        let cross = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

        // 0. control: plain memory the CPU wrote, trimmed – how fast the system gets it back
        let size = 1024 * MIB;
        let watch = |what: &str, p: usize| {
            K32EmptyWorkingSet(-1);
            let line: Vec<String> = (0..7).map(|_| {
                std::thread::sleep(std::time::Duration::from_millis(1000));
                let (r, _, a) = look(p, size);
                format!("{r}/{a}")
            }).collect();
            println!("{what}: resident/available each second after the working set is emptied: {}", line.join(" "));
        };
        let c = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
        for at in (0..size).step_by(4096) {
            ((c + at) as *mut u32).write_volatile(at as u32 | 1);
        }
        println!("control, 1024 MiB written:     {:?}", look(c, size));
        watch("control (plain written memory)", c);
        VirtualFree(c as *mut std::ffi::c_void, 0, MEM_RELEASE);
        std::thread::sleep(std::time::Duration::from_millis(1000));
        // 1. what the open costs

        println!("(resident MiB, locked-bit MiB, system available MiB); video (local, non-local) MiB");
        println!("start:                         available {} video {:?}", look(0x10000, 4096).2, video());
        let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
        println!("1024 MiB committed, untouched: {:?} video {:?}", look(p, size), video());
        let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(p as *const std::ffi::c_void).unwrap();
        let hd = heap.GetDesc();
        println!("heap: type {} cpu page {} pool {} flags {:#x} size {}", hd.Properties.Type.0, hd.Properties.CPUPageProperty.0, hd.Properties.MemoryPoolPreference.0, hd.Flags.0, hd.SizeInBytes);
        println!("opened as a heap:              {:?} video {:?}", look(p, size), video());
        watch("opened heap", p);
        let mut placed: Option<ID3D12Resource> = None;
        let made = dev.CreatePlacedResource(&heap, 0, &buffer(size, cross), D3D12_RESOURCE_STATE_COMMON, None, &mut placed);
        println!("buffer placed over it ({:?}):  {:?} video {:?}", made.is_ok(), look(p, size), video());
        (p as *mut u32).add(1024).write_volatile(0xC0FF_EE00);
        if let Some(b) = &placed {
            println!("the GPU reads the CPU's word:  {:x?}", read(&[(b, 4096)]));
        }
        trim();
        println!("used, working set emptied:     {:?} video {:?}", look(p, size), video());
        let pageable: ID3D12Pageable = heap.cast().unwrap();
        println!("Evict: {:?}", dev.Evict(&[Some(pageable.clone())]));
        trim();
        println!("evicted, working set emptied:  {:?} video {:?}", look(p, size), video());
        println!("MakeResident: {:?}", dev.MakeResident(&[Some(pageable.clone())]));
        println!("resident again:                {:?} video {:?}", look(p, size), video());
        if let Some(b) = &placed {
            println!("the GPU reads the word again:  {:x?}", read(&[(b, 4096)]));
        }
        drop(placed);
        drop(pageable);
        drop(heap);
        trim();
        println!("heap released, emptied:        {:?} video {:?}", look(p, size), video());
        VirtualFree(p as *mut std::ffi::c_void, 0, MEM_RELEASE);

        // 2. a reserved buffer over two heaps opened from adjacent allocations
        let piece = 64 * MIB;
        let b = VirtualAlloc(std::ptr::null(), 2 * piece, MEM_RESERVE, PAGE_NOACCESS) as usize;
        VirtualFree(b as *mut std::ffi::c_void, 0, MEM_RELEASE);
        let c0 = VirtualAlloc(b as *const std::ffi::c_void, piece, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
        let c1 = VirtualAlloc((b + piece) as *const std::ffi::c_void, piece, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
        println!("pieces at {c0:#x} and {c1:#x} (asked {b:#x}, {:#x})", b + piece);
        if c0 != 0 && c1 != 0 {
            let h0: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(c0 as *const std::ffi::c_void);
            let h1: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(c1 as *const std::ffi::c_void);
            println!("piece heaps: {:?} {:?}", h0.as_ref().map(|h| h.GetDesc().SizeInBytes), h1.as_ref().map(|h| h.GetDesc().SizeInBytes));
            if let (Ok(h0), Ok(h1)) = (h0, h1) {
                for (name, flags) in [("cross-adapter", cross)] {
                    let mut reserved: Option<ID3D12Resource> = None;
                    let made = dev.CreateReservedResource(&buffer(2 * piece, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved);
                    println!("reserved buffer ({name}): {:?}", made.as_ref().map(|_| ()));
                    let Some(reserved) = reserved else { continue };
                    let tiles = (piece >> 16) as u32;
                    for (k, h) in [&h0, &h1].into_iter().enumerate() {
                        let coord = D3D12_TILED_RESOURCE_COORDINATE { X: k as u32 * tiles, ..Default::default() };
                        let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles, UseBox: false.into(), ..Default::default() };
                        let (flag, offset, count) = (D3D12_TILE_RANGE_FLAG_NONE, 0u32, tiles);
                        queue.UpdateTileMappings(&reserved, 1, Some(&coord), Some(&region), h, 1, Some(&flag), Some(&offset), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
                    }
                    println!("  after UpdateTileMappings: device removed reason {:?}", dev.GetDeviceRemovedReason());
                    (c0 as *mut u32).add(1024).write_volatile(0xC0FF_EE10);
                    ((c0 + piece - 4) as *mut u32).write_volatile(0xC0FF_EE11);
                    (c1 as *mut u32).add(2048).write_volatile(0xC0FF_EE12);
                    println!("  the GPU reads through the tiles (want c0ffee10, c0ffee11, c0ffee12): {:x?}", read(&[(&reserved, 4096), (&reserved, piece - 4), (&reserved, piece + 8192)]));
                    println!("  device removed reason {:?}", dev.GetDeviceRemovedReason());
                    if dev.GetDeviceRemovedReason().is_err() {
                        break;
                    }
                }
                // 3. pages of an opened heap given back
                let freed = VirtualFree((c1 + 16 * MIB) as *mut std::ffi::c_void, 16 * MIB, MEM_DECOMMIT);
                println!("decommit inside an opened heap: {freed} (error {})", GetLastError());
                drop(h1);
                let freed = VirtualFree((c1 + 32 * MIB) as *mut std::ffi::c_void, 16 * MIB, MEM_DECOMMIT);
                println!("decommit after its release: {freed} (error {})", GetLastError());
                drop(h0);
            }
        }
        VirtualFree(c0 as *mut std::ffi::c_void, 0, MEM_RELEASE);
        VirtualFree(c1 as *mut std::ffi::c_void, 0, MEM_RELEASE);
    }
}

/// Decoded GFX10 T#s give resource descriptions the driver accepts.
#[test]
fn gpu_tsharp_to_d3d12_resources() {
    if !gpu_ready() {
        return;
    }
    use crate::gpu::desc;
    let t = |fmt: u32, w: u32, h: u32, depth: u32, ty: u32, mips: u32| -> [u32; 8] {
        let (w1, h1) = (w - 1, h - 1);
        [0, fmt << 20 | (w1 & 3) << 30, (w1 >> 2) | h1 << 14, 4 | 5 << 3 | 6 << 6 | 7 << 9 | (mips - 1) << 16 | ty << 28, depth - 1, (mips - 1) << 4, 0, 0]
    };
    for (name, words) in [
        ("2D RGBA8 9 mips", t(56, 256, 256, 1, 9, 9)),
        ("2D BC7 sRGB", t(182, 128, 64, 1, 9, 1)),
        ("cube RGBA16F", t(71, 64, 64, 6, 11, 7)),
        ("3D R11G11B10", t(36, 32, 32, 16, 10, 1)),
        ("2D array BC1 ×8", t(169, 64, 64, 8, 13, 4)),
        ("1D R32F", t(22, 1024, 1, 1, 8, 1)),
    ] {
        let img = desc::image(&words).unwrap();
        let (res, _srv) = desc::image_d3d12(&img).unwrap();
        runner::try_create_texture(&res).unwrap_or_else(|e| panic!("{name}: {e} ({res:?})"));
    }
}

/// Prints signatures / PSV0 of named retail shaders (`XCL_SAMPLE_RPF`, `XCL_DUMP=a,b`).
#[test]
#[ignore]
fn dump_retail_interfaces() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_DUMP")) else { return };
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| r.name.contains(n)) {
            continue;
        }
        let blob = crate::gpu::pack::inflate(&data, &r).unwrap();
        let (i, o) = crate::gpu::dxbc::signatures(&blob);
        let sh = crate::gpu::xbox::parse(&blob).unwrap();
        println!("{} hw {:?}", r.name, sh.fssm.iter().map(|f| f.stage_label()).collect::<Vec<_>>());
        println!("  ISG1: {:?}", i.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        println!("  OSG1: {:?}", o.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        if let Some(parts) = crate::gpu::shader::container_parts(&blob) {
            if let Some(p) = parts.iter().find(|p| p.fourcc == "PSV0") {
                let d = &blob[p.offset..p.offset + p.size];
                println!("  PSV0 {:02x?}", &d[..d.len().min(60)]);
            }
        }
    }
}

/// Geometry mode end to end: an ES program (vertex → LDS, s_setpc to the GS) and a GS
/// program (triangle from LDS through the v0/v1 offsets, a 4-vertex strip = 2 triangles,
/// emit / cut) run as one compute shader; 25 triangles = 2 groups (21 + 4); records checked
/// vertex by vertex (strip winding included), then drawn by the mesh pass-through.
#[test]
fn gpu_geometry_mode_es_gs_strip() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, GsInterface, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    // ES: vertex (v9..v12) → LDS at lane·32, then into the GS
    let mut es = Vec::new();
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 133), ("VSRC1", 20)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 9), ("DATA1", 10), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 11), ("DATA1", 12), ("OFFSET0", 2), ("OFFSET1", 3)], None));
    es.extend(e("S_BARRIER", "ENC_SOPP", &[], None));
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    // GS: EXEC = primitives of this wave (s3[15:8]), vertices from LDS, parallelogram strip
    let mut gs = Vec::new();
    gs.extend(e("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)));
    gs.extend(e("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None));
    gs.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    gs.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)], None));
    gs.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 30)], Some(0x3FFFC)));
    gs.extend(e("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 40), ("ADDR", 30), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 42), ("ADDR", 31), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 44), ("ADDR", 32), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    for (d, a, b) in [(46u64, 42u64, 44u64), (47, 43, 45)] {
        gs.extend(e("V_ADD_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None));
        gs.extend(e("V_SUB_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + d), ("VSRC1", d - 6)], None)); // − x0 / − y0
    }
    // output base: s2 + lane · (4 vertices · 32 bytes)
    gs.extend(e("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128)));
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 51), ("SRC0", 240)], None)); // 0.5
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 52), ("SRC0", 242)], None)); // 1.0
    for (k, (x, y)) in [(40u64, 41u64), (42, 43), (44, 45), (46, 47)].into_iter().enumerate() {
        let o = 8 * k as u64;
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", x), ("DATA1", y), ("OFFSET0", o), ("OFFSET1", o + 1)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 51), ("DATA1", 52), ("OFFSET0", o + 2), ("OFFSET1", o + 3)], None));
        gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 53), ("SRC0", 128 + k as u64)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 2), ("DATA1", 53), ("OFFSET0", o + 4), ("OFFSET1", o + 5)], None));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None)); // emit
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)], None)); // cut
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));

    let sig = |name: &str, sv: u32, reg: u32, ty: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: ty,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0, 3)],
        // two ES output registers: 32-byte ESGS items (the ES above writes at lane · 32)
        es_outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 1)],
        max_vertex_count: 4,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);
    let native = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, split_wave64: false, ..Default::default() }).unwrap();
    assert!(native.hlsl.contains("[WaveSize(64)]"));
    assert!(!native.hlsl.contains("xcl_es_phase") && !native.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    let t = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(32)]"));
    assert!(t.hlsl.contains("xcl_es_phase") && t.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let rec = g.record();
    let msb = dxc::compile(&mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap();
    let ps_src = "struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, (float)v.p0.y, 0.0, 1.0); }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();

    // 25 triangles, vertex attributes pre-fetched at 0x1000 (one float4 per vertex)
    let total = 25usize;
    let vtx = |p: usize, i: usize| [p as f32 * 0.03 - 0.6 + i as f32 * 0.2, p as f32 * 0.02 - 0.5 + (i == 2) as u32 as f32 * 0.25, 0.5, 1.0];
    let mut mem = vec![0u32; 0x1000 / 4 + total * 3 * 4];
    for p in 0..total {
        for i in 0..3 {
            let at = 0x1000 / 4 + (p * 3 + i) * 4;
            mem[at..at + 4].copy_from_slice(&vtx(p, i).map(f32::to_bits));
        }
    }
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = total as u32;
    io[IO_GS_VERTICES] = 0x1000;
    let (px, recs) = runner::mesh_draw(&cs, &msb, &ps, 2, rec.record_dwords(), &io, &mem, 32, [0.0; 4]).unwrap();
    let (rd, vd) = (rec.record_dwords(), rec.vert_dwords());
    for grp in 0..2usize {
        let r = &recs[grp * rd..(grp + 1) * rd];
        let np = if grp == 0 { 21 } else { 4 };
        assert_eq!((r[0], r[1]), (4 * np as u32, 2 * np as u32), "group {grp}: vertices / triangles");
        for lane in 0..np {
            let p = grp * 21 + lane;
            let (a, b, c) = (vtx(p, 0), vtx(p, 1), vtx(p, 2));
            let xy = [[a[0], a[1]], [b[0], b[1]], [c[0], c[1]], [b[0] + c[0] - a[0], b[1] + c[1] - a[1]]];
            for (k, want) in xy.iter().enumerate() {
                let v = MESH_HEADER + (4 * lane + k) * vd;
                let pos = [f32::from_bits(r[v]), f32::from_bits(r[v + 1]), f32::from_bits(r[v + 2]), f32::from_bits(r[v + 3])];
                assert_eq!(pos, [want[0], want[1], 0.5, 1.0], "group {grp} lane {lane} vertex {k}");
                assert_eq!((r[v + 4], r[v + 5]), (p as u32, k as u32), "group {grp} lane {lane} vertex {k}: TEXCOORD0");
            }
            let pk = |i: usize, j: usize, l: usize| (i | j << 10 | l << 20) as u32;
            let tris = MESH_HEADER + rec.max_vertices as usize * vd;
            let b0 = 4 * lane;
            assert_eq!(r[tris + 2 * lane], pk(b0, b0 + 1, b0 + 2), "group {grp} lane {lane}: strip triangle 0");
            assert_eq!(r[tris + 2 * lane + 1], pk(b0 + 2, b0 + 1, b0 + 3), "group {grp} lane {lane}: strip triangle 1 (odd: swapped)");
        }
    }
    let covered = px.iter().filter(|p| p[0] == 1.0).count();
    assert!(covered > 20, "pass-through drew the GS output: {covered} pixels");
}

/// v_mbcnt over a v_cmp mask in a geometry program (the compaction index of an ES / GS):
/// lanes 32..63 count the active lanes of 0..31 too. The split form (two wave32 halves) must
/// give the logical wave64 counts – it exchanges the halves' mask words – and so must the
/// native form where the GPU has 64-lane waves (the compute pass alone: no mesh shaders needed).
#[test]
fn gpu_geometry_wave32_prefix_count_over_a_mask() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, translate_geometry, GsInterface, IO_GS_PRIMS};
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    let mut es = Vec::new();
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None)); // lane
    es.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 131), ("VSRC1", 20)], None));
    es.extend(e("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 21)], None)); // vcc = lane & 3 != 0
    es.extend(e("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", 168), ("SRC1", V + 20)], None)); // s[10:11] = lane < 40
    // v22: below this lane in VCC, v23: in s[10:11], v24: in EXEC (= VCC; the others keep 0xFFFF)
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 106), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 107), ("SRC1", V + 22)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 10), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 11), ("SRC1", V + 23)], None));
    es.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", 255)], Some(0xFFFF)));
    es.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 106)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 126), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 127), ("SRC1", V + 24)], None));
    es.extend(e("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 12)], None));
    // s[8:9] + lane · 4: v22 at +0, v23 at +256, v24 at +512
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 25), ("SRC0", 130), ("VSRC1", 20)], None));
    for (k, d) in [22u64, 23, 24].into_iter().enumerate() {
        es.extend(e("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 25), ("DATA", d), ("SADDR", 8), ("OFFSET", 256 * k as u64)], None));
    }
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    let mut gs = Vec::new();
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done, nothing emitted
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![],
        es_outputs: vec![sig("SV_Position", 1, 0)],
        outputs: vec![sig("SV_Position", 1, 0)],
        max_vertex_count: 3,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);

    // what one wave64 computes
    let below = |m: u64, l: usize| (m & ((1u64 << l) - 1)).count_ones();
    let (vcc, s10) = ((0..64).filter(|l| l & 3 != 0).fold(0u64, |m, l| m | 1 << l), (1u64 << 40) - 1);
    let want: Vec<[u32; 3]> = (0..64).map(|l| [below(vcc, l), below(s10, l), if vcc >> l & 1 == 1 { below(vcc, l) } else { 0xFFFF }]).collect();
    let mut io = vec![0u32; 128];
    io[8] = 0x400;
    io[IO_GS_PRIMS] = 21;
    let run = |t: &super::Translation| {
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let mem = runner::dispatch(&cs, [1, 1, 1], &io, &vec![0u32; 0x400 / 4 + 3 * 64]).unwrap();
        let got: Vec<[u32; 3]> = (0..64).map(|l| [0, 1, 2].map(|k| mem[0x100 + 64 * k + l])).collect();
        let wrong: Vec<_> = (0..64).filter(|&l| got[l] != want[l]).map(|l| (l, got[l], want[l])).collect();
        assert!(wrong.is_empty(), "lanes (got [vcc, s[10:11], exec], want): {wrong:?}");
    };
    run(&translate_geometry(&code, g.clone(), true, 1).unwrap());
    // with a title's four memory windows (`pso::console_geometry`): all of them declared and
    // the program still compiles
    let windowed = translate_geometry(&code, g.clone(), true, 4).unwrap();
    assert!(windowed.hlsl.contains("xcl_w = min("));
    super::dxc::compile(&windowed.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    // the native wave64 form keeps the program, and computes the reference where it can run
    let native = translate_geometry(&code, g, false, 1).unwrap();
    if runner::wave64_supported() == Ok(true) {
        run(&native);
    } else {
        runner::skip_test("no wave64");
    }
}

/// A DS read whose destination overlaps its address register reads the address first
/// (retail GS code: `ds_read2_b32 v[0:1], v0 offset1:1`).
#[test]
fn gpu_ds_read_destination_overlaps_address() {
    if !gpu_ready() {
        return;
    }
    let mut code = Vec::new();
    code.extend(encode("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2), ("DATA1", 3), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = 8 * l as u32; // address
        r[2] = 1000 + l as u32;
        r[3] = 2000 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1000 + l as u32, 2000 + l as u32), "lane {l}");
    }
}

/// `DS_WRITE_ADDTID_B32` / `DS_READ_ADDTID_B32` compute `M0[15:0] + offset16 + 4 * lane_id`
/// without touching unused `ADDR`/`DATA1` VGPR fields, and reject `GDS=1`.
#[test]
fn ds_addtid_translates_with_m0_offset_and_lane_id() {
    let mut code = Vec::new();
    code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xDEAD_0100)).unwrap());
    // Populate unused ADDR/DATA1 (and DATA0 on read) with nonzero indices to verify they are ignored.
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 2), ("DATA1", 11), ("OFFSET0", 0x40), ("OFFSET1", 0x01)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("ADDR", 10), ("DATA0", 12), ("DATA1", 13), ("VDST", 3), ("OFFSET0", 0x40), ("OFFSET1", 0x01)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.unsupported.is_empty(), "{:?}\n{}", t.unsupported, t.hlsl);
    assert!(t.hlsl.contains("groupshared uint xcl_lds"), "{}", t.hlsl);
    assert!(t.hlsl.contains("((m0 & 0xFFFFu) + 320u + 4u * lane_id)"), "{}", t.hlsl);
    for unused in ["uint v9 =", "uint v10 =", "uint v11 =", "uint v12 =", "uint v13 ="] {
        assert!(!t.hlsl.contains(unused), "unexpected {unused} in:\n{}", t.hlsl);
    }

    // GDS=1 must be reported as unsupported and not allocate LDS.
    let mut gds_code = Vec::new();
    gds_code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 3), ("GDS", 1)], None).unwrap());
    gds_code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let tg = translate(&gds_code, &Options::default()).unwrap();
    assert_eq!(tg.unsupported.get("ds_read_addtid_b32"), Some(&1), "{:?}", tg.unsupported);
    assert!(tg.hlsl.contains("GDS"), "{}", tg.hlsl);
    assert!(!tg.hlsl.contains("groupshared uint xcl_lds"), "{}", tg.hlsl);
}

/// GPU verification for `DS_WRITE_ADDTID_B32` / `DS_READ_ADDTID_B32`: verifies exact LDS byte
/// address (`M0[15:0] + ((OFFSET1 << 8) | OFFSET0) + 4 * lane_id`, ignoring `M0[31:16]`) in both
/// directions against explicit `DS_READ_B32` / `DS_WRITE_B32`, plus EXEC masking on writes & reads.
#[test]
fn gpu_ds_addtid_roundtrip_and_exec_masking() {
    if !gpu_ready() {
        return;
    }
    let mut code = Vec::new();
    // m0 = 0xDEAD_0100 -> M0[15:0] = 256, offset = 0x4040 = 16448 -> base byte offset = 16704 + 4 * lane
    code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xDEAD_0100)).unwrap());
    // Direction 1: seed LDS via explicit DS_WRITE_B32 at v1 (= 16704 + 4 * lane) and read via DS_READ_ADDTID_B32 -> v7
    code.extend(encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 8)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 7), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    // Direction 2: all lanes write v2 via DS_WRITE_ADDTID_B32 and verify via explicit DS_READ_B32 -> v4
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 2), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 1)], None).unwrap());
    // Mask EXEC to even lanes (s[0:1] = 0x5555_5555_5555_5555): overwrite LDS with v6 and read into v3
    code.extend(encode("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)], None).unwrap());
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 6), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 3), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    // Restore all 64 lanes (EXEC = -1) and read final LDS contents into v5
    code.extend(encode("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 5), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[0] = 0x5555_5555;
    st.sgpr[1] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u32;
        r[1] = 16704 + 4 * l; // explicit byte address (256 + 0x4040 + 4 * lane)
        r[2] = 0x1000 + l;    // ADDTID write value
        r[3] = 0xCAFE_0000 + l; // sentinel preserved on inactive (odd) lanes
        r[6] = 0x2000 + l;    // masked overwrite value (even lanes only)
        r[8] = 0xBEEF_0000 + l; // seeded via DS_WRITE_B32 for ADDTID read check
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let lu = l as u32;
        assert_eq!(r.vgpr[l][7], 0xBEEF_0000 + lu, "lane {l} DS_WRITE_B32 -> DS_READ_ADDTID_B32");
        assert_eq!(r.vgpr[l][4], 0x1000 + lu, "lane {l} DS_WRITE_ADDTID_B32 -> DS_READ_B32");
        if l % 2 == 0 {
            assert_eq!(r.vgpr[l][3], 0x2000 + lu, "even lane {l} masked DS_READ_ADDTID_B32");
            assert_eq!(r.vgpr[l][5], 0x2000 + lu, "even lane {l} final DS_READ_ADDTID_B32");
        } else {
            assert_eq!(r.vgpr[l][3], 0xCAFE_0000 + lu, "odd lane {l} sentinel must be preserved");
            assert_eq!(r.vgpr[l][5], 0x1000 + lu, "odd lane {l} LDS slot must be untouched");
        }
    }
}

/// Ray tracing: image_bvh(64)_intersect_ray on a hand-built BVH (box32 root with 4 children,
/// a box16 node, triangles), several rays; sorted and unsorted box results, triangle hit as
/// (t, denominator, i, j) numerators, misses, and the 64-bit node pointer form.
#[test]
fn gpu_bvh_intersect_ray() {
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let (root, box16, tri_a, tri_b) = (5u32, (0x80u32 >> 3) | 4, 0xC0u32 >> 3, 0x100u32 >> 3);
    // root box32 at 0: children + boxes (min xyz, max xyz)
    w(&mut mem, 0, &[tri_a, box16, 0xFFFF_FFFF, tri_b]);
    w(&mut mem, 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0])); // triangle A's box: entry t = 5
    w(&mut mem, 40, &ff(&[-1.0, -1.0, 2.0, 1.0, 1.0, 3.0])); // box16 node's box: t = 2
    w(&mut mem, 64, &ff(&[-1.0, -1.0, 0.5, 1.0, 1.0, 0.7])); // invalid child: never reported
    w(&mut mem, 88, &ff(&[10.0, -1.0, 8.0, 11.0, 1.0, 9.0])); // off the ray
    // box16 node at 0x80: one child (triangle B) in z 6..7, half floats
    let h = |x: f32| half_bits(x);
    w(&mut mem, 0x80, &[tri_b, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    w(&mut mem, 0x90, &[h(-1.0) | h(-1.0) << 16, h(6.0) | h(1.0) << 16, h(1.0) | h(7.0) << 16]);
    // triangles: A in z = 5 (hit at u = v = 0.5 from the origin), B in z = 6.5
    w(&mut mem, 0xC0, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    w(&mut mem, 0x100, &ff(&[-1.0, -1.0, 6.5, 1.0, -1.0, 6.5, -1.0, 1.0, 6.5]));
    // triangle id 9: I = vertex 1, J = vertex 2 (no rotation)
    w(&mut mem, 0xC0 + 60, &[9]);
    w(&mut mem, 0x100 + 60, &[9]);
    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0, 2)); // root, sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 104, 0, 3)); // root, no sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 20, 2)); // box16 node
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 112, 40, 2)); // triangle A
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 116, 60, 2)); // triangle A, 64-bit node id
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    st.sgpr[12..16].copy_from_slice(&[(base >> 8) as u32, 0, u32::MAX, 0]);
    // lane 0: ray along +z from the origin; lane 1: the same with t max 3.5; lane 2: from (5, 5)
    let ray = |l: usize| -> ([f32; 3], f32) {
        match l % 3 {
            0 => ([0.0, 0.0, 0.0], 100.0),
            1 => ([0.0, 0.0, 0.0], 3.5),
            _ => ([5.0, 5.0, 0.0], 100.0),
        }
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (o, tmax) = ray(l);
        let rest = |r: &mut Vec<u32>, at: usize| {
            let v = [tmax, o[0], o[1], o[2], 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
            for (k, x) in v.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root;
        rest(r, 1);
        r[20] = box16;
        rest(r, 21);
        r[40] = tri_a;
        rest(r, 41);
        r[60] = tri_a;
        r[61] = 0;
        rest(r, 62);
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    const F: u32 = 0xFFFF_FFFF;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        match l % 3 {
            0 => {
                assert_eq!(q(100), [box16, tri_a, F, F], "lane {l}: box32, closest first");
                assert_eq!(q(104), [tri_a, box16, F, F], "lane {l}: box32 without sorting keeps child order");
                assert_eq!(q(108), [tri_b, F, F, F], "lane {l}: box16");
            }
            1 => {
                assert_eq!(q(100), [box16, F, F, F], "lane {l}: t max 3.5 excludes the box at t = 5");
                assert_eq!(q(108), [F, F, F, F], "lane {l}: box16 at t = 6 beyond t max");
            }
            _ => {
                assert_eq!(q(100), [F, F, F, F], "lane {l}: ray from (5, 5) misses every box");
                assert_eq!(q(112)[0], f32::INFINITY.to_bits(), "lane {l}: triangle missed → t = inf");
            }
        }
        if l % 3 != 2 {
            let tri = q(112).map(f32::from_bits);
            let (t, i, j) = (tri[0] / tri[1], tri[2] / tri[1], tri[3] / tri[1]);
            assert_eq!((t, i, j), (5.0, 0.5, 0.5), "lane {l}: triangle hit (t, i, j)");
            assert_eq!(q(116), q(112), "lane {l}: 64-bit node pointer gives the same");
        }
    }
}

/// Checks the compute ABI read from the FSSM stage descriptor (+0x38 COMPUTE_PGM_RSRC2,
/// +0x3C.. numthreads) on every CS of the pack against PSV0 and the SGPRs/VGPRs the code reads.
#[test]
#[ignore]
fn compute_abi_survey() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let data = std::fs::read(path).unwrap();
    let (mut n, mut threads_ok, mut sgpr_ok, mut vgpr_ok) = (0, 0, 0, 0);
    for r in crate::gpu::pack::scan(&data, 0x40) {
        // compute records are recognised by their hardware stage below, not by name
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(f) = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6)) else { continue };
        let d = f.stages[0].offset;
        let rd = |o: usize| u32::from_le_bytes(f.data[d + o..d + o + 4].try_into().unwrap());
        let rsrc2 = rd(0x38);
        let nt = [rd(0x3C), rd(0x40), rd(0x44)];
        let user = (rsrc2 >> 1) & 31;
        let tg = [(rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1, (rsrc2 >> 10) & 1];
        let tid = (rsrc2 >> 11) & 3;
        let sys = tg.iter().sum::<u32>();
        n += 1;
        let psv = crate::gpu::dxbc::psv(&blob).and_then(|p| p.numthreads);
        threads_ok += (psv == Some(nt)) as usize;
        // registers read before written
        let p = f.programs[0];
        let code = &f.data[p.offset..p.offset + p.len];
        let (mut sr, mut sw, mut vr, mut vw) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        let mut off = 0;
        while off + 4 <= code.len() {
            let Ok(dd) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { break };
            off += dd.len;
            let (r, w) = crate::gpu::rdna2::sgpr_uses(&dd);
            sr.extend(r.into_iter().filter(|x| !sw.contains(x) && *x < 106));
            sw.extend(w);
            let (r, w) = crate::gpu::rdna2::vgpr_uses(&dd);
            vr.extend(r.into_iter().filter(|x| !vw.contains(x)));
            vw.extend(w);
        }
        let max_s = sr.iter().max().copied().unwrap_or(0);
        let s_in = max_s < (user + sys) as u64;
        sgpr_ok += s_in as usize;
        let v_in = vr.iter().all(|&v| v <= tid as u64);
        vgpr_ok += v_in as usize;
        if !(s_in && v_in && psv == Some(nt)) {
            println!("{}: rsrc2 {rsrc2:#x} user {user} tgid {tg:?} tid {tid} threads {nt:?} psv {psv:?} SGPRs read {sr:?} VGPR {vr:?}", r.name);
        }
    }
    println!("CS: {n}, numthreads = PSV0: {threads_ok}, SGPRs within user+system: {sgpr_ok}, VGPRs within the thread ID: {vgpr_ok}");
}

/// Compute mode with the console dispatch ABI: numthreads 16×8 (two wave64 waves per group),
/// 2×3 groups; group ids from the SGPRs after the user data, thread ids in v0/v1; LDS shared
/// across the two waves through a barrier.
#[test]
fn gpu_compute_mode_dispatch_abi() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let cs = CsInterface { threads: [16, 8, 1], user_sgprs: 2, tgid: [true, true, false], tg_size: false, tid_components: 2 };
    let p = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 132), ("VSRC1", 1)], None), // y · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 2), ("VSRC1", 0)], None), // index in group
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 132)], None), // group x · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 4), ("VSRC1", 0)], None), // global x
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 3), ("SSRC1", 131)], None), // group y · 8
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 5), ("VSRC1", 1)], None), // global y
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 144), ("VSRC1", 4)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 3)], None), // x | y << 16
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 6), ("DATA0", 5)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 2)], Some(127)), // partner in the other wave
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 5)], None),
        ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 1)], None), // y · width
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 9), ("VSRC1", 3)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 9)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 0), ("VSRC1", 9)], None), // + output base
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 9), ("DATA", 11), ("SADDR", 0x7D)], None),
    ]);
    let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let (w, h, base) = (32usize, 24usize, 0x100usize);
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    io[1] = w as u32;
    let out = runner::dispatch(&dxil, [2, 3, 1], &io, &vec![0u32; (base + w * h * 8) / 4]).unwrap();
    for y in 0..h {
        for x in 0..w {
            let at = (base + (y * w + x) * 8) / 4;
            assert_eq!(out[at], (x | y << 16) as u32, "({x}, {y}): own value");
            let (gx, gy, i) = (x / 16, y / 8, (y % 8) * 16 + x % 16);
            let (px, py) = (gx * 16 + (127 - i) % 16, gy * 8 + (127 - i) / 16);
            assert_eq!(out[at + 1], (px | py << 16) as u32, "({x}, {y}): value of thread {} through LDS", 127 - i);
        }
    }
}

/// A wave32 program in a 64-lane hardware wave (what VS / PS may get from the driver): the
/// two halves behave as separate waves – ballots, lane counts, readlane and mbcnt per half.
#[test]
fn gpu_wave32_program_in_wave64_hardware() {
    if !gpu_ready() || !wave64_hardware() {
        return;
    }
    let p = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 138), ("VSRC1", 0)], None), // vcc_lo = lanes with v0 < 10
        ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 131)], None), // s1 = v0 of lane 3
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 1)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 193), ("SRC1", 128)], None),
    ]);
    let t = translate(&p, &Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(64)]") && t.hlsl.contains("xcl_half"), "{}", t.hlsl);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut st = KernelState::default();
    st.sgpr[126] = u32::MAX; // exec of each half
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = (l % 32 + if l >= 32 { 5 } else { 0 }) as u32; // second half shifted by 5
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();
    for l in 0..64 {
        let half = l / 32;
        let v = &r.vgpr[l];
        assert_eq!(v[1], if half == 0 { 10 } else { 5 }, "lane {l}: active lanes of its own half");
        assert_eq!(v[2], 3 + 5 * half as u32, "lane {l}: readlane 3 of its own half");
        assert_eq!(v[3], (l % 32) as u32, "lane {l}: lane index within the half");
    }
}

/// SV_VertexID / SV_InstanceID listed in ISG1 with a register arrive like attributes,
/// v[4 + 4r] (retail VInstanced* shaders read the instance id in v8 = register 1).
#[test]
fn vs_system_values_in_isg1_registers() {
    use super::VsInterface;
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: if sv == 0 { 3 } else { 1 },
        register: reg,
        mask: if sv == 0 { 7 } else { 1 },
        rw_mask: 1,
        stream: 0,
    };
    let vs = VsInterface {
        inputs: vec![sig("POSITION", 0, 0), sig("SV_InstanceID", 8, 1), sig("SV_VertexID", 6, 2)],
        outputs: vec![sig("SV_Position", 1, 0)],
        attr_base: 4,
        params: None,
    };
    let code = prog(&[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 8), ("VSRC1", 12)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 21), ("SRC0", V + 4), ("VSRC1", 5)], None),
    ]);
    let t = translate(&code, &Options { mode: Mode::Vertex(vs), wave64: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("uint v8 = xi.xcl_iid;") && t.hlsl.contains("uint v12 = xi.xcl_vid;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint v4 = asuint(xi.a0[0]);"), "{}", t.hlsl);
}

#[test]
fn vs_packed_inputs_use_semantic_types_and_signature_masks() {
    let vs = super::VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("TEXCOORD", 0, 0, 3, 1, 0b0101), el("BLENDINDICES", 0, 0, 1, 1, 0b1000)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)],
        attr_base: 4,
        params: None,
    };
    let code = prog(&[
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 8), ("VSRC1", 10)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 21), ("SRC0", V + 9), ("VSRC1", 11)], None),
    ]);
    let t = translate(&code, &Options { mode: Mode::Vertex(vs), wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    for expected in [
        "float4 a0 : POSITION0;",
        "float2 a1_1 : TEXCOORD0;",
        "uint1 a1_2 : BLENDINDICES0;",
        "uint v8 = asuint(xi.a1_1[0]);",
        "uint v9 = 0u;",
        "uint v10 = asuint(xi.a1_1[1]);",
        "uint v11 = asuint(xi.a1_2[0]);",
    ] {
        assert!(t.hlsl.contains(expected), "{expected}:\n{}", t.hlsl);
    }
}

#[test]
#[ignore]
fn probe_standard_swizzle_support() {
    use windows::Win32::Graphics::Direct3D12::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev = dev.unwrap();
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut _, std::mem::size_of_val(&o) as u32).unwrap();
        println!("StandardSwizzle64KBSupported = {:?}", o.StandardSwizzle64KBSupported);
    }
}

/// A BVH from the builder (gpu::bvh) through the translated image_bvh_intersect_ray: every
/// lane tests one built node (box32 or triangle) with its own ray; the GPU must agree with
/// the CPU node test the builder's traversal uses.
#[test]
fn gpu_built_bvh_nodes_match_the_cpu_node_test() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let tris = bvh::tests::random_tris(40, 5);
    let b = bvh::build(&tris, 0);
    // every node id, root first
    let mut nodes = vec![];
    let mut todo = vec![b.root];
    while let Some(n) = todo.pop() {
        nodes.push(n);
        if n & 7 == bvh::BOX32 {
            let at = ((n & !7) as usize) << 3;
            for k in 0..4 {
                let c = u32::from_le_bytes(b.data[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
                if c != bvh::INVALID {
                    todo.push(c);
                }
            }
        }
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; (base + b.data.len()) / 4 + 64];
    for (k, c) in b.data.chunks(4).enumerate() {
        mem[base / 4 + k] = u32::from_le_bytes(c.try_into().unwrap());
    }
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(17);
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % nodes.len()];
        let o = [r(-12.0, 12.0), r(-12.0, 12.0), r(-12.0, 12.0)];
        let to = [r(-6.0, 6.0), r(-6.0, 6.0), r(-6.0, 6.0)];
        let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code_with_options(&code, &st, &mem, &Options { bvh_node_fix: true, ..runner::options(Mode::Kernel) }).unwrap();
    let (mut boxes, mut tri_hits) = (0, 0);
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&b.data, node, 100.0, o, d, inv, bvh::NodeOptions { sort: true, ..Default::default() });
        if node & 7 == bvh::BOX32 {
            assert_eq!(gpu, cpu, "lane {l}: box node {node:#x}");
            boxes += 1;
        } else {
            let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
            assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: triangle {node:#x} hit / miss");
            if !c[0].is_infinite() {
                assert!((g[0] / g[1] - c[0] / c[1]).abs() < 1e-4, "lane {l}: t {g:?} vs {c:?}");
                tri_hits += 1;
            }
        }
    }
    assert!(boxes > 10 && tri_hits > 0, "boxes {boxes}, triangle hits {tri_hits}");
}

/// The console's node forms (seen in the console builder's output): a box16 node and a GPURT
/// triangle pair – type 0 = (v0, v1, v2), type 1 = (v1, v3, v2), barycentrics put back in the
/// original vertex order by the triangle id. GPU (HLSL emulation) = CPU mirror, and the
/// barycentrics of a known point come out right.
#[test]
fn gpu_triangle_pair_and_box16_nodes() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // corners of a quad in z = 5; triangle 0 = (p0, p1, p2) stored rotated: v0 = p1, v1 = p2,
    // v2 = p0, so its I barycentric (weight of p1) is vertex 0 and J (p2) vertex 1
    let (p0, p1, p2, p3) = ([-1.0f32, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]);
    let pair = 0x40usize;
    w(&mut mem, pair, &ff(&[p1, p2, p0, p3].concat()));
    // triangle 1 = (v1, v3, v2) = (p2, p3, p0), original order (p0, p3, p2): I (p3) is vertex 1
    // of the triple, J (p2) vertex 0. Triangle id: byte 0 = 0x04, byte 1 = 0x01
    w(&mut mem, pair + 48, &[0, 11, 12, 0x0104]);
    // box16 root with the two triangles
    let h = |x: f32| half_bits(x);
    let (t0, t1) = ((pair as u32) >> 3, (pair as u32) >> 3 | 1);
    w(&mut mem, 0, &[t0, t1, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    for k in 0..2 {
        w(&mut mem, 16 + 12 * k, &[h(-1.0) | h(-1.0) << 16, h(4.5) | h(1.0) << 16, h(1.0) | h(5.5) << 16]);
    }
    let root = 4u32; // box16 at 0
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(5);
    let nodes = [root, t0, t1];
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % 3];
        let o = [r(-1.5, 1.5), r(-1.5, 1.5), 0.0];
        let d = [r(-0.1, 0.1), r(-0.1, 0.1), 1.0];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code_with_options(&code, &st, &mem, &Options { bvh_node_fix: true, ..runner::options(Mode::Kernel) }).unwrap();
    let data: Vec<u8> = mem[base / 4..].iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut checked = 0;
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&data, node, 100.0, o, d, inv, bvh::NodeOptions { sort: true, ..Default::default() });
        if node == root {
            assert_eq!(gpu, cpu, "lane {l}: box16");
            continue;
        }
        let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
        assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: hit / miss");
        if c[0].is_infinite() {
            continue;
        }
        for k in 0..4 {
            assert!((g[k] / g[1] - c[k] / c[1]).abs() < 1e-5, "lane {l}: {g:?} vs {c:?}");
        }
        // hit point from the barycentrics in the ORIGINAL order of the stored triangle
        let (t, bi, bj) = (c[0] / c[1], c[2] / c[1], c[3] / c[1]);
        let hit = [0, 1, 2].map(|a| o[a] + t * d[a]);
        if node == t0 {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p1[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 0 barycentrics {q:?} vs {hit:?}");
        } else {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p3[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 1 barycentrics {q:?} vs {hit:?}");
        }
        checked += 1;
    }
    assert!(checked > 5, "{checked} triangle hits");
}

#[test]
fn bvh_intersection_loads_are_inside_the_lane_predicate() {
    for (op, bvh_node_fix) in [("IMAGE_BVH_INTERSECT_RAY", false), ("IMAGE_BVH64_INTERSECT_RAY", false), ("IMAGE_BVH_INTERSECT_RAY", true), ("IMAGE_BVH64_INTERSECT_RAY", true)] {
        let code = prog(&[(op, "ENC_MIMG", &[("VDATA", 100), ("VADDR", 12), ("SRSRC", 2), ("DMASK", 15)], None)]);
        let t = translate(&code, &Options { bvh_node_fix, ..Options::default() }).unwrap();
        let call = t.hlsl.lines().find(|line| line.contains(" = bvh_intersect(")).expect("BVH intersection call");
        assert!(call.trim_start().starts_with("if ("), "{op}: {call}");
    }
}

#[test]
fn gpu_bvh_node_types_and_a16() {
    use crate::gpu::bvh::tests::{triangle_ray, TRIANGLE_REMAP, TRIANGLE_VERTICES};
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    for (k, v) in TRIANGLE_VERTICES.iter().enumerate() {
        let at = base / 4 + k * 3;
        mem[at..at + 3].copy_from_slice(&v.map(f32::to_bits));
    }
    mem[(base + 60) / 4] = TRIANGLE_REMAP;
    for (offset, packed) in [(0x40, true), (0x80, false)] {
        let at = (base + offset) / 4;
        mem[at..at + 4].copy_from_slice(&[8, 16, 24, 32]);
        for (k, z) in [6.0f32, -2.0, -2.0, 18.0].into_iter().enumerate() {
            let bounds = [-16.0, -16.0, z, 16.0, 16.0, z + 8.0];
            if packed {
                let h = bounds.map(half_bits);
                let dst = at + 4 + 3 * k;
                mem[dst..dst + 3].copy_from_slice(&[h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16]);
            } else {
                let dst = at + 4 + 6 * k;
                mem[dst..dst + 6].copy_from_slice(&bounds.map(f32::to_bits));
            }
        }
    }
    let mut code = Vec::new();
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&crate::gpu::desc::bvh_words(base as u64, true));
    st.sgpr[11] = 0x8100_0000;
    let forms = [(false, false), (false, true), (true, false), (true, true)];
    for (k, (wide, a16)) in forms.into_iter().enumerate() {
        let (dst, src) = (100 + 4 * k, 16 * k);
        let op = if wide { "IMAGE_BVH64_INTERSECT_RAY" } else { "IMAGE_BVH_INTERSECT_RAY" };
        code.extend(encode(op, "ENC_MIMG", &[("VDATA", dst as u64), ("VADDR", src as u64), ("SRSRC", 2), ("DMASK", 15), ("A16", a16 as u64)], None).unwrap());
        for (lane, v) in st.vgpr.iter_mut().enumerate() {
            let case = lane % 32;
            let (node, tmax, o, d) = if case < 20 {
                let (ty, ray) = (case / 5, case % 5);
                let (o, d, _, _) = triangle_ray(ty, ray);
                (ty as u32, 1.0, o, d)
            } else {
                let node = if case < 24 { 12 } else if case < 28 { 21 } else { 6 | (case & 1) as u32 };
                let tmax = [4.0, -0.5, 100.0, 100.0][case % 4];
                let o = [if case % 4 == 3 { 64.0 } else { 0.0 }, 0.0, 0.0];
                (node, tmax, o, [0.5, -0.25, 2.0])
            };
            let start = src + 1 + usize::from(wide);
            v[src] = node;
            v[start..start + 4].copy_from_slice(&[tmax, o[0], o[1], o[2]].map(f32::to_bits));
            let [x, y, z] = d;
            let dir_inv = [x, y, z, 1.0 / x, 1.0 / y, 1.0 / z];
            if a16 {
                let h = dir_inv.map(half_bits);
                v[start + 4..start + 7].copy_from_slice(&[h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16]);
            } else {
                v[start + 4..start + 10].copy_from_slice(&dir_inv.map(f32::to_bits));
            }
        }
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let out = runner::run_code_with_options(&code, &st, &mem, &Options { bvh_node_fix: true, ..runner::options(Mode::Kernel) }).unwrap();
    const M: u32 = u32::MAX;
    for (lane, v) in out.vgpr.iter().enumerate() {
        let case = lane % 32;
        for k in 0..forms.len() {
            let got: [u32; 4] = v[100 + k * 4..104 + k * 4].try_into().unwrap();
            let context = format!("lane {lane}, width/A16 {:?}: {got:x?}", forms[k]);
            if case < 20 {
                let (_, d, t, [i, j]) = triangle_ray(case / 5, case % 5);
                let [tn, det, bi, bj] = got.map(f32::from_bits);
                assert_eq!(det.is_sign_negative(), d[2] > 0.0, "{context}");
                assert_eq!([bi / det, bj / det], [i, j], "{context}");
                if t < 0.0 {
                    assert!(tn.is_infinite() && tn.is_sign_negative() == det.is_sign_negative(), "{context}");
                } else {
                    assert_eq!(tn / det, t, "{context}");
                }
            } else {
                let expect = if case >= 28 { [M; 4] } else { [[24, 16, 8, M], [M; 4], [24, 16, 8, 32], [M; 4]][case % 4] };
                assert_eq!(got, expect, "{context}");
            }
        }
    }
}

#[test]
fn gpu_bvh_box_growth() {
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![u32::MAX; 0x2000 / 4];
    let bounds = [-1.0f32, -1.0, 2.0, 1.0, 1.0, 4.0];
    let h = bounds.map(half_bits);
    let at = base / 4;
    mem[at] = 8;
    mem[at + 4..at + 7].copy_from_slice(&[h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16]);
    let at = (base + 64) / 4;
    mem[at] = 8;
    mem[at + 4..at + 10].copy_from_slice(&bounds.map(f32::to_bits));

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&crate::gpu::desc::bvh_words(base as u64, false));
    st.sgpr[12..16].copy_from_slice(&crate::gpu::desc::bvh_words(base as u64, true));
    st.sgpr[13] |= 6 << 23;
    let mut code = Vec::new();
    for (dst, desc) in [(100, 2), (104, 3)] {
        code.extend(encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", dst), ("VADDR", 0), ("SRSRC", desc), ("DMASK", 15)], None).unwrap());
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let near = 2.0f32.to_bits();
    for (lane, v) in st.vgpr.iter_mut().enumerate() {
        v[0] = if lane & 1 == 0 { 4 } else { 13 };
        v[1] = [near - 1, near - 16, near, (-1.0f32).to_bits()][(lane / 2) % 4];
        v[7] = 1.0f32.to_bits();
        v[8..11].copy_from_slice(&[f32::INFINITY, f32::INFINITY, 1.0].map(f32::to_bits));
    }
    let out = runner::run_code_with_options(&code, &st, &mem, &Options { bvh_node_fix: true, ..runner::options(Mode::Kernel) }).unwrap();
    for (lane, v) in out.vgpr.iter().enumerate() {
        let case = (lane / 2) % 4;
        assert_eq!(v[100], if case == 2 { 8 } else { u32::MAX }, "lane {lane}, grow 0");
        assert_eq!(v[104], if case == 0 || case == 2 { 8 } else { u32::MAX }, "lane {lane}, grow 6");
        assert_eq!(v[101..104], [u32::MAX; 3]);
        assert_eq!(v[105..108], [u32::MAX; 3]);
    }
}

/// Multi-window `image_bvh64_intersect_ray` (full 64-bit node pointer into UMA window 1 with
/// base = 0, plus 48-bit base in window 1) and axis-aligned grazing ray / NaN planes.
#[test]
fn gpu_bvh64_intersect_ray_across_memory_windows_and_grazing() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let write_words = |addr: u64, vals: &[u32]| unsafe {
        let p = addr as *mut u32;
        for (k, &v) in vals.iter().enumerate() {
            *p.add(k) = v;
        }
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let base_off = 0x1000u64;
    let bvh_addr = w1 + base_off;
    let tri_rel = 0x80u32 >> 3;
    let nan_child = 0xC0u32 >> 3;
    let inv_child = 0x100u32 >> 3;
    // Window 1 root box32 at bvh_addr: child 0 = normal box [-1..1, -1..1, 5..6],
    // child 1 = open NaN plane, child 2 = inverted box (mn > mx), child 3 = all near planes NaN
    write_words(bvh_addr, &[tri_rel, nan_child, inv_child, 0x140 >> 3]);
    write_words(bvh_addr + 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0]));
    write_words(bvh_addr + 40, &ff(&[f32::NAN, -1.0, 2.0, 1.0, 1.0, 3.0]));
    write_words(bvh_addr + 64, &ff(&[2.0, -1.0, 2.0, -2.0, 1.0, 3.0]));
    write_words(bvh_addr + 88, &ff(&[f32::NAN, f32::NAN, f32::NAN, 1.0, 1.0, 3.0]));
    // Triangle at bvh_addr + 0x80 in z = 5
    write_words(bvh_addr + 0x80, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    write_words(bvh_addr + 0x80 + 60, &[9]);
    // Poison window 0 at the same low offset: reading window 0 would return bogus child 0xDEAD_BEEF
    write_words(w0 + base_off, &[0xDEAD_BEEF, u32::MAX, u32::MAX, u32::MAX]);
    write_words(w0 + base_off + 16, &ff(&[-100.0, -100.0, 1.0, 100.0, 100.0, 2.0]));

    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    // s[8:11] = base 0 (box sort enabled), v[0:1] = full 64-bit node pointer into window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 100, 0, 2));
    // s[8:11] = base 0, v[20:21] = full 64-bit triangle node pointer in window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 104, 20, 2));
    // s[12:15] = 48-bit base pointing at window 1 (`bvh_words(bvh_addr, true)`), v40 = 32-bit root
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 40, 3));
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 112, 60, 4));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let t = translate(&code, &on_gpu(Options { mem_windows: 2, bvh_node_fix: true, ..Default::default() })).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&crate::gpu::desc::bvh_words(0, true));
    st.sgpr[12..16].copy_from_slice(&crate::gpu::desc::bvh_words(bvh_addr, true));
    st.sgpr[16..20].copy_from_slice(&crate::gpu::desc::bvh_words(w0, true));
    let root64 = (bvh_addr >> 3) | 5;
    let tri64 = (bvh_addr + 0x80) >> 3;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        // Even lanes graze the box edge (o.x = -1.0 == mn.x, dx = 0, inv.x = +inf);
        // odd lanes fire through the center (0, 0, 0) along +z.
        let ox = if l % 2 == 0 { -1.0f32 } else { 0.0f32 };
        let ray = [100.0f32, ox, 0.0, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        let put_ray = |r: &mut Vec<u32>, at: usize| {
            for (k, x) in ray.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root64 as u32;
        r[1] = (root64 >> 32) as u32;
        put_ray(r, 2);
        r[20] = tri64 as u32;
        r[21] = (tri64 >> 32) as u32;
        put_ray(r, 22);
        r[40] = 5; // relative root box32 at bvh_addr
        put_ray(r, 41);
        r[60] = (if l % 2 == 0 { (bvh_addr - w0) >> 3 } else { base_off >> 3 }) as u32 | 5;
        put_ray(r, 61);
    }
    let res = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    const F: u32 = u32::MAX;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        let window_one = [nan_child, tri_rel, F, F];
        assert_eq!(q(100), window_one, "lane {l}: 64-bit box32 in window 1 (grazing/NaN/inverted)");
        assert_eq!(q(108), window_one, "lane {l}: 48-bit base box32 in window 1");
        let expected = if l % 2 == 0 { window_one } else { [0xDEAD_BEEF, F, F, F] };
        assert_eq!(q(112), expected, "lane {l}: 32-bit relative node uses the lane's effective window");
        let tri = q(104).map(f32::from_bits);
        assert!(!tri[0].is_infinite() && ((tri[0] / tri[1]) - 5.0).abs() < 1e-4, "lane {l}: 64-bit triangle in window 1 → {tri:?}");
    }
}

/// Instructions first met in XboxBvhBuilder.dll's shaders: global atomics (float min / max by
/// a CAS loop), 8 / 16-bit buffer loads / stores and 8 / 16-bit LDS accesses – neighbouring
/// lanes share dwords, so a non-atomic read-modify-write would lose writes.
#[test]
fn gpu_global_atomics_and_sub_dword_access() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let ds = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_DS", f, None).unwrap();
    let mut code = Vec::new();
    // global atomics on s[2:3] + v30 (= 0 for every lane) + offset
    code.extend(glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 40), ("ADDR", 30), ("DATA", 31), ("SADDR", 2), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_SUB", &[("VDST", 41), ("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMAX", &[("VDST", 42), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMIN", &[("VDST", 43), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 12)]));
    code.extend(glbl("GLOBAL_ATOMIC_SMAX", &[("VDST", 44), ("ADDR", 30), ("DATA", 33), ("SADDR", 2), ("OFFSET", 16)]));
    // V# s[24:27]: raw at 0x700; short at lane*2, byte at 0x80 + lane (offen)
    code.extend(mubuf("BUFFER_STORE_SHORT", &[("VDATA", 34), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_STORE_BYTE", &[("VDATA", 36), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_USHORT", &[("VDATA", 45), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SSHORT", &[("VDATA", 46), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_UBYTE", &[("VDATA", 47), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SBYTE", &[("VDATA", 48), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    // LDS: 16-bit at lane*2, 8-bit at 256 + lane; after the barrier read the neighbour's (lane ^ 1)
    code.extend(ds("DS_WRITE_B16", &[("ADDR", 2), ("DATA0", 34)]));
    code.extend(ds("DS_WRITE_B8", &[("ADDR", 4), ("DATA0", 36)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(ds("DS_READ_U16", &[("VDST", 49), ("ADDR", 5)]));
    code.extend(ds("DS_READ_I16", &[("VDST", 50), ("ADDR", 5)]));
    code.extend(ds("DS_READ_U8", &[("VDST", 51), ("ADDR", 6)]));
    code.extend(ds("DS_READ_I8", &[("VDST", 52), ("ADDR", 6)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[24..28].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        r[4] = 256 + l32;
        r[5] = 2 * (l32 ^ 1);
        r[6] = 256 + (l32 ^ 1);
        r[30] = 0;
        r[31] = l32 + 1;
        r[32] = f(l as f32 - 20.5);
        r[33] = (l as i32 - 30) as u32;
        r[34] = 0xAB00 + l32;
        r[35] = 1;
        r[36] = 0x90 + l32;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 1000;
    mem[0x608 / 4] = f32::NEG_INFINITY.to_bits();
    mem[0x60C / 4] = f32::INFINITY.to_bits();
    mem[0x610 / 4] = (-100i32) as u32;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 * 65 / 2, "global_atomic_add");
    let mut olds: Vec<u32> = r.vgpr.iter().map(|v| v[40]).collect();
    olds.sort_unstable();
    olds.dedup();
    assert_eq!(olds.len(), 64, "global_atomic_add with GLC returns 64 distinct old values");
    assert_eq!(r.mem[0x604 / 4], 1000 - 64, "global_atomic_sub");
    assert_eq!(f32::from_bits(r.mem[0x608 / 4]), 42.5, "global_atomic_fmax");
    assert_eq!(f32::from_bits(r.mem[0x60C / 4]), -20.5, "global_atomic_fmin");
    assert_eq!(r.mem[0x610 / 4] as i32, 33, "global_atomic_smax");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (s, b) = (0xAB00 + l as u32, 0x90 + l as u32);
        let n = l ^ 1;
        assert_eq!((r.mem[(0x700 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF, s, "lane {l}: buffer_store_short");
        assert_eq!((r.mem[(0x780 + l) / 4] >> (8 * (l % 4))) & 0xFF, b, "lane {l}: buffer_store_byte");
        assert_eq!((v[45], v[46]), (s, s | 0xFFFF_0000), "lane {l}: buffer_load_ushort / sshort");
        assert_eq!((v[47], v[48]), (b, b | 0xFFFF_FF00), "lane {l}: buffer_load_ubyte / sbyte");
        let (sn, bn) = (0xAB00 + n as u32, 0x90 + n as u32);
        assert_eq!((v[49], v[50]), (sn, sn | 0xFFFF_0000), "lane {l}: ds_read_u16 / i16 of the neighbour");
        assert_eq!((v[51], v[52]), (bn, bn | 0xFFFF_FF00), "lane {l}: ds_read_u8 / i8 of the neighbour");
    }
}

/// The instructions added for code outside the retail pack: global inc / dec / csub /
/// 64-bit add, LDS signed min / max, cmpst, mskor, float add (CAS loops), global sub-dword
/// and d16 accesses, scratch, SAD / lerp, the division helpers, saturating adds, packs, swap.
#[test]
fn gpu_more_atomics_sub_dword_scratch_valu() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let ds = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_DS", f);
    let vop3 = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_VOP3", f);
    let mut code = Vec::new();
    // global atomics at 0x600 + offset (s[2:3] + v30 = 0)
    code.extend(glbl("GLOBAL_ATOMIC_INC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 0)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_CSUB", &[("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_ADD_X2", &[("ADDR", 30), ("DATA", 38), ("SADDR", 2), ("OFFSET", 16)]));
    code.extend(glbl("GLOBAL_ATOMIC_INC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 24)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 32)]));
    // LDS: initial values, then the atomics, then read back
    for (a, v) in [(7, 8), (17, 9), (18, 10), (19, 12), (20, 15)] {
        code.extend(ds("DS_WRITE_B32", &[("ADDR", a), ("DATA0", v)]));
    }
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    code.extend(ds("DS_MAX_I32", &[("ADDR", 7), ("DATA0", 33)]));
    code.extend(ds("DS_MIN_I32", &[("ADDR", 17), ("DATA0", 33)]));
    code.extend(ds("DS_CMPST_RTN_B32", &[("VDST", 53), ("ADDR", 18), ("DATA0", 10), ("DATA1", 11)]));
    code.extend(ds("DS_MSKOR_B32", &[("ADDR", 19), ("DATA0", 13), ("DATA1", 14)]));
    code.extend(ds("DS_ADD_F32", &[("ADDR", 20), ("DATA0", 16)]));
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    for (k, a) in [7u64, 17, 18, 19, 20].iter().enumerate() {
        code.extend(ds("DS_READ_B32", &[("VDST", 54 + k as u64), ("ADDR", *a)]));
    }
    // global bytes at 0x780 + lane, shorts at 0x700 + 2 · lane (s[2:3] + v + 0x100 / 0x180)
    code.extend(glbl("GLOBAL_STORE_BYTE", &[("ADDR", 3), ("DATA", 36), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE", &[("VDST", 59), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_SBYTE", &[("VDST", 60), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE_D16_HI", &[("VDST", 61), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_STORE_SHORT_D16_HI", &[("ADDR", 2), ("DATA", 62), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_USHORT", &[("VDST", 63), ("ADDR", 2), ("SADDR", 2), ("OFFSET", 0x100)]));
    // scratch (per lane): a dword at 2 · lane + 16, a byte at 2 · lane + 40
    code.extend(enc("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 11), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_STORE_BYTE", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 36), ("SADDR", 0x7F), ("OFFSET", 40)]));
    code.extend(enc("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 64), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_LOAD_UBYTE", "ENC_FLAT_SCRATCH", &[("VDST", 65), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 40)]));
    // VALU
    let vv = |n: u64| V + n;
    code.extend(vop3("V_SAD_U8", &[("VDST", 66), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(24))]));
    code.extend(vop3("V_MSAD_U8", &[("VDST", 67), ("SRC0", vv(22)), ("SRC1", vv(25)), ("SRC2", vv(24))]));
    code.extend(vop3("V_LERP_U8", &[("VDST", 68), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(26))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 69), ("SRC0", vv(37)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 70), ("SRC0", vv(37)), ("SRC1", 128), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FMAS_F32", &[("VDST", 71), ("SRC0", vv(28)), ("SRC1", vv(27)), ("SRC2", vv(28))]));
    code.extend(enc("V_DIV_SCALE_F32", "VOP3_SDST_ENC", &[("VDST", 72), ("SDST", 106), ("SRC0", vv(27)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_ADD_NC_I32", &[("VDST", 73), ("SRC0", vv(29)), ("SRC1", vv(35)), ("CLAMP", 1)]));
    code.extend(vop3("V_CVT_PK_U16_U32", &[("VDST", 74), ("SRC0", vv(76)), ("SRC1", vv(24))]));
    code.extend(vop3("V_MAD_U32_U16", &[("VDST", 75), ("SRC0", vv(77)), ("SRC1", vv(78)), ("SRC2", vv(35))]));
    code.extend(vop3("V_CVT_PKNORM_U16_F32", &[("VDST", 79), ("SRC0", 242), ("SRC1", 240)]));
    // normalized f16 conversions: v82 = halves (0.5, +1.0), v83 = (−1.0, −0.5)
    code.extend(vop3("V_CVT_PKNORM_U16_F16", &[("VDST", 84), ("SRC0", vv(82)), ("SRC1", vv(83))]));
    code.extend(vop3("V_CVT_PKNORM_I16_F16", &[("VDST", 85), ("SRC0", vv(82)), ("SRC1", vv(83)), ("OP_SEL", 3)]));
    code.extend(vop3("V_CVT_NORM_U16_F16", &[("VDST", 86), ("SRC0", vv(82))]));
    code.extend(vop3("V_CVT_NORM_I16_F16", &[("VDST", 86), ("SRC0", vv(83)), ("OP_SEL", 9)]));
    code.extend(enc("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 87), ("SRC0", 242)]));
    code.extend(enc("V_SWAP_B32", "ENC_VOP1", &[("VDST", 80), ("SRC0", vv(81))]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        for (k, a) in [(7, 512u32), (17, 516), (18, 520), (19, 524), (20, 528)] {
            r[k] = a;
        }
        r[8] = (-100i32) as u32;
        r[9] = 100;
        r[10] = 7;
        r[11] = l32 + 1;
        r[12] = 0xFF00;
        r[13] = 0xF0F0;
        r[14] = 0x0101;
        r[15] = 0;
        r[16] = f(0.5);
        r[21] = 10;
        r[22] = 0x1020_3040;
        r[23] = 0x4030_2010;
        r[24] = 5;
        r[25] = 0x0030_2010;
        r[26] = 0x0101_0101;
        r[27] = f(7.0);
        r[28] = f(2.0);
        r[29] = 0x7FFF_FFFF;
        r[30] = 0;
        r[33] = (l as i32 - 30) as u32;
        r[35] = 1;
        r[36] = 0x90 + l32;
        r[37] = f(3.5);
        r[38] = 1;
        r[39] = 0;
        r[61] = 0x1234;
        r[62] = 0xBEEF_1111;
        r[76] = 70000;
        r[77] = 0x1_0003;
        r[78] = 0x2_0004;
        r[80] = 1;
        r[81] = 2;
        r[82] = 0x3C00_3800; // f16 halves: lo +0.5, hi +1.0
        r[83] = 0xB800_BC00; // lo −1.0, hi −0.5
        r[90] = 10;
        r[91] = 0;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 5;
    mem[0x608 / 4] = 50;
    mem[0x610 / 4] = 0xFFFF_FFF0;
    mem[0x620 / 4] = 5;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 % 11, "global_atomic_inc wraps at 10");
    assert_eq!(r.mem[0x604 / 4], 7, "global_atomic_dec wraps to 10 below 0");
    assert_eq!(r.mem[0x608 / 4], 0, "global_atomic_csub clamps at 0");
    assert_eq!((r.mem[0x610 / 4], r.mem[0x614 / 4]), (0x30, 1), "global_atomic_add_x2 carries into the high dword");
    assert_eq!((r.mem[0x618 / 4], r.mem[0x61C / 4]), (64 % 11, 0), "global_atomic_inc_x2 wraps at 10");
    assert_eq!((r.mem[0x620 / 4], r.mem[0x624 / 4]), (7, 0), "global_atomic_dec_x2 wraps to 10 below 0");
    assert_eq!(r.vgpr.iter().filter(|v| v[53] == 7).count(), 1, "ds_cmpst: one lane finds the compare value");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let b = 0x90 + l as u32;
        assert_eq!((v[54] as i32, v[55] as i32), (33, -30), "lane {l}: ds_max_i32 / ds_min_i32");
        assert!((1..=64).contains(&v[56]), "lane {l}: ds_cmpst stored a lane's value");
        assert_eq!((v[57], f32::from_bits(v[58])), (0x0F01, 32.0), "lane {l}: ds_mskor_b32 / ds_add_f32");
        assert_eq!((v[59], v[60], v[61]), (b, b | 0xFFFF_FF00, 0x1234 | b << 16), "lane {l}: global ubyte / sbyte / ubyte_d16_hi");
        assert_eq!(v[63], 0xBEEF, "lane {l}: global_store_short_d16_hi + load_ushort");
        assert_eq!((v[64], v[65]), (l as u32 + 1, b), "lane {l}: scratch dword / byte");
        assert_eq!((v[66], v[67], v[68]), (133, 85, 0x2828_2828), "lane {l}: v_sad_u8 / v_msad_u8 / v_lerp_u8");
        assert_eq!((f32::from_bits(v[69]), f32::from_bits(v[70]), f32::from_bits(v[71])), (3.5, f32::INFINITY, 16.0), "lane {l}: div_fixup / fixup(x/0) / div_fmas");
        assert_eq!((f32::from_bits(v[72]), v[73]), (7.0, 0x7FFF_FFFF), "lane {l}: v_div_scale_f32 / v_add_nc_i32 clamp");
        assert_eq!((v[74], v[75], v[79]), (0x0005_FFFF, 13, 0x8000_FFFF), "lane {l}: cvt_pk_u16_u32 / mad_u32_u16 / cvt_pknorm_u16_f32");
        assert_eq!((v[84], v[85], v[86], v[87]), (0x0000_8000, 0xC000_7FFF, 0xC000_8000, 0xFFFF), "lane {l}: cvt norm f16 (unorm/snorm, packed, hi halves)");
        assert_eq!((v[80], v[81]), (2, 1), "lane {l}: v_swap_b32");
    }
    assert_eq!(r.sgpr[106] | r.sgpr[107], 0, "v_div_scale_f32: VCC 0 (no post-scale)");
}

#[test]
fn gpu_atomic_fcmpswap_x2() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let mut code = Vec::new();
    // buffer: V# s[8:11] = raw base 0x600; per-lane cell at base + voff (OFFEN), voff = 8·lane
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 2), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]));
    // flat: cell at v14 (0x800 + 8·lane)
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)]));
    // global: cell at s2 + v22 (0xA00 + 8·lane)
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)]));
    // one cell shared by all 64 lanes per family: buffer 0xC00 (V# s[12:15]), flat 0xE00,
    // global s3 + v9 (0x1200)
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 30), ("VADDR", 8), ("SRSRC", 3), ("SOFFSET", 128), ("GLC", 1)]));
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 40), ("ADDR", 10), ("DATA", 34), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 44), ("ADDR", 9), ("DATA", 46), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0x1200;
    st.sgpr[8..12].copy_from_slice(&[0x600, 0, 0x200, 0]); // V#: base, stride/flags, records, –
    st.sgpr[12..16].copy_from_slice(&[0xC00, 0, 0x200, 0]);
    const INIT: u64 = 0x3FF8_0000_0000_0000; // f64 bits of 1.5
    const NEG_ZERO: u64 = 0x8000_0000_0000_0000; // -0.0: == +0.0 numerically, different bits
    let newv = |tag: u64, l: u64| tag << 48 | l;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l64 = l as u64;
        r[2] = 8 * l as u32; // buffer offset -> cell 0x600 + 8·lane
        r[14] = 0x800 + 8 * l as u32;
        r[22] = 8 * l as u32; // + s2 -> 0xA00 + 8·lane
        r[10] = 0xE00; // flat shared cell (all lanes)
        let cmp = if l % 2 == 0 { INIT } else { NEG_ZERO };
        let put = |r: &mut Vec<u32>, at: usize, v: u64| {
            r[at] = v as u32;
            r[at + 1] = (v >> 32) as u32;
        };
        put(r, 4, newv(0xB000, l64)); // buffer: new at [4:5], cmp at [6:7]
        put(r, 6, cmp);
        put(r, 16, newv(0xF000, l64));
        put(r, 18, cmp);
        put(r, 24, newv(0xA000, l64));
        put(r, 26, cmp);
        put(r, 30, newv(0x5B00, l64)); // shared cells: every lane compares against INIT
        put(r, 32, INIT);
        put(r, 34, newv(0x5F00, l64));
        put(r, 36, INIT);
        put(r, 46, newv(0x5A00, l64));
        put(r, 48, INIT);
    }
    let mut mem = vec![0u32; 0x1400 / 4];
    for l in 0..64usize {
        let b = if l % 2 == 0 { INIT } else { 0 };
        for a in [0x600 + 8 * l, 0x800 + 8 * l, 0xA00 + 8 * l] {
            mem[a / 4] = b as u32;
            mem[a / 4 + 1] = (b >> 32) as u32;
        }
    }
    for a in [0xC00, 0xE00, 0x1200] {
        mem[a / 4] = INIT as u32;
        mem[a / 4 + 1] = (INIT >> 32) as u32;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let old = |v: &[u32], at: usize| v[at] as u64 | (v[at + 1] as u64) << 32;
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let even = l % 2 == 0;
        let l64 = l as u64;
        let (bs, fs, gs) = (newv(0xB000, l64), newv(0xF000, l64), newv(0xA000, l64));
        let cmp_old = if even { INIT } else { 0 }; // the cell's pre-swap value
        assert_eq!(cell(0x600 + 8 * l), bs, "lane {l}: buffer cell");
        assert_eq!(cell(0x800 + 8 * l), fs, "lane {l}: flat cell");
        assert_eq!(cell(0xA00 + 8 * l), gs, "lane {l}: global cell");
        // GLC old value: VDATA in place for buffer, VDST for flat / global
        assert_eq!(old(v, 4), cmp_old, "lane {l}: buffer GLC old");
        assert_eq!(old(v, 12), cmp_old, "lane {l}: flat GLC old");
        assert_eq!(old(v, 20), cmp_old, "lane {l}: global GLC old");
        // the compare register half is untouched
        assert_eq!(old(v, 6), if even { INIT } else { NEG_ZERO }, "lane {l}: buffer cmp regs");
    }
    // shared cells: exactly one lane sees INIT (the winner); the cell holds its value
    for (i, (dst, a, tag)) in [(30, 0xC00usize, 0x5B00u64), (40, 0xE00, 0x5F00), (44, 0x1200, 0x5A00)].iter().enumerate() {
        let winners: Vec<usize> = r.vgpr.iter().enumerate().filter(|(_, v)| old(v, *dst) == INIT).map(|(l, _)| l).collect();
        assert_eq!(winners.len(), 1, "shared cell {i}: exactly one lane wins the swap");
        assert_eq!(cell(*a), newv(*tag, winners[0] as u64), "shared cell {i}: the winner's value landed");
    }
}

#[test]
fn gpu_atomic_f32_matches_ordered_float_comparisons() {
    if !gpu_ready() {
        return;
    }
    let ops = ["GLOBAL_ATOMIC_FMIN", "GLOBAL_ATOMIC_FMAX", "GLOBAL_ATOMIC_FCMPSWAP"];
    let mut code = Vec::new();
    for (i, op) in ops.into_iter().enumerate() {
        code.extend(encode(op, "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 4 * i as u64), ("VDST", 40 + i as u64), ("GLC", 1)], None).unwrap());
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut state = KernelState::default();
    state.sgpr[2] = 0x600;
    let values = [
        f32::NEG_INFINITY.to_bits(), (-5.0f32).to_bits(), 0x8000_0001, (-0.0f32).to_bits(),
        0, 1, 3.0f32.to_bits(), f32::INFINITY.to_bits(), 0x7fff_ffff, 0xffc0_0001,
        0x7f80_0001, 0xff80_0001,
    ];
    let mut mem = vec![0u32; 0x900 / 4];
    for (i, regs) in state.vgpr.iter_mut().enumerate() {
        let old = values[i % values.len()];
        regs[30] = 12 * i as u32;
        regs[32] = values[(i * 7 + 3) % values.len()];
        regs[33] = if old & 0x7fff_ffff == 0 { old ^ 0x8000_0000 } else if i % 4 == 1 { regs[32] } else { old };
        let at = 0x600 / 4 + i * 3;
        mem[at..at + 3].fill(old);
    }
    let result = runner::run_code(&code, &state, &mem).unwrap();
    for (i, regs) in state.vgpr.iter().enumerate() {
        let at = 0x600 / 4 + i * 3;
        let (old, v, cmp) = (mem[at], regs[32], regs[33]);
        let (a, b, c) = (f32::from_bits(old), f32::from_bits(v), f32::from_bits(cmp));
        let want = [if b < a { v } else { old }, if b > a { v } else { old }, if a == c { v } else { old }];
        assert_eq!(result.mem[at..at + 3], want, "lane {i}: old={old:08x} operand={v:08x} cmp={cmp:08x}");
        assert_eq!(result.vgpr[i][40..43], [old; 3], "lane {i}: GLC old bits");
    }
}

#[test]
fn gpu_atomic_fmin_fmax_negative_zero() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let mut code = Vec::new();
    // global: s2 + v30 (= 0) + offset; buffer: V# s[8:11], raw base 0x700, + offset
    for (op, off) in [("FMIN", 0u64), ("FMAX", 4), ("FMIN", 8), ("FMAX", 12)] {
        let (g, b) = match op {
            "FMIN" => ("GLOBAL_ATOMIC_FMIN", "BUFFER_ATOMIC_FMIN"),
            _ => ("GLOBAL_ATOMIC_FMAX", "BUFFER_ATOMIC_FMAX"),
        };
        code.extend(glbl(g, &[("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", off)]));
        code.extend(mubuf(b, &[("VDATA", 32), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[8..12].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for r in st.vgpr.iter_mut() {
        r[30] = 0;
        r[32] = (-0.0f32).to_bits();
    }
    let mut mem = vec![0u32; 0x800 / 4];
    for base in [0x600usize, 0x700] {
        mem[base / 4] = (-5.0f32).to_bits();
        mem[base / 4 + 1] = (-5.0f32).to_bits();
        mem[base / 4 + 2] = 3.0f32.to_bits();
        mem[base / 4 + 3] = 3.0f32.to_bits();
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for (family, base) in [("global", 0x600usize), ("buffer", 0x700)] {
        let cell = |k: usize| r.mem[base / 4 + k];
        assert_eq!(cell(0), (-5.0f32).to_bits(), "{family}: fmin(-5, -0) = -5");
        assert_eq!(cell(1), (-0.0f32).to_bits(), "{family}: fmax(-5, -0) = -0");
        assert_eq!(cell(2), (-0.0f32).to_bits(), "{family}: fmin(3, -0) = -0");
        assert_eq!(cell(3), 3.0f32.to_bits(), "{family}: fmax(3, -0) = 3");
    }
}

/// *_atomic_fmin_x2 / fmax_x2 on hardware, all three families. Six cells each: 64 lanes
/// race on a max (from -inf) and a min (from +inf) of lane - 20.5, then the -0.0 cases of
/// the f32 test. A seventh, per-lane cell checks the GLC return (global).
#[test]
fn gpu_atomic_fmin_fmax_x2() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut code = Vec::new();
    // (op, data VGPR, cell offset): the lane values at v[32:33], -0.0 at v[34:35]
    let ops = [("FMAX", 32u64, 0u64), ("FMIN", 32, 8), ("FMIN", 34, 16), ("FMAX", 34, 24), ("FMIN", 34, 32), ("FMAX", 34, 40)];
    for (op, data, off) in ops {
        let x2 = |family: &str| -> &'static str {
            match (family, op) {
                ("BUFFER", "FMIN") => "BUFFER_ATOMIC_FMIN_X2",
                ("BUFFER", _) => "BUFFER_ATOMIC_FMAX_X2",
                ("FLAT", "FMIN") => "FLAT_ATOMIC_FMIN_X2",
                ("FLAT", _) => "FLAT_ATOMIC_FMAX_X2",
                (_, "FMIN") => "GLOBAL_ATOMIC_FMIN_X2",
                _ => "GLOBAL_ATOMIC_FMAX_X2",
            }
        };
        // buffer: V# s[8:11], raw base 0x800; flat: address v[36:37] = 0x900; global: s2 = 0xA00
        code.extend(enc(x2("BUFFER"), "ENC_MUBUF", &[("VDATA", data), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
        code.extend(enc(x2("FLAT"), "ENC_FLAT", &[("ADDR", 36), ("DATA", data), ("OFFSET", off)]));
        code.extend(enc(x2("GLOBAL"), "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", data), ("SADDR", 2), ("OFFSET", off)]));
    }
    // per-lane cell at s3 + v38 (0xB00 + 8·lane): max(1.0, 2.0), the old 1.0 back in v[40:41]
    code.extend(enc("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 40), ("ADDR", 38), ("DATA", 42), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0xB00;
    st.sgpr[8..12].copy_from_slice(&[0x800, 0, 0x100, 0]);
    let put = |r: &mut Vec<u32>, at: usize, v: u64| {
        r[at] = v as u32;
        r[at + 1] = (v >> 32) as u32;
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[30] = 0;
        put(r, 32, (l as f64 - 20.5).to_bits());
        put(r, 34, (-0.0f64).to_bits());
        put(r, 36, 0x900);
        r[38] = 8 * l as u32;
        put(r, 42, 2.0f64.to_bits());
    }
    let init = [f64::NEG_INFINITY, f64::INFINITY, -5.0, -5.0, 3.0, 3.0];
    let mut mem = vec![0u32; 0xE00 / 4];
    let set = |mem: &mut Vec<u32>, a: usize, v: f64| {
        mem[a / 4] = v.to_bits() as u32;
        mem[a / 4 + 1] = (v.to_bits() >> 32) as u32;
    };
    for base in [0x800usize, 0x900, 0xA00] {
        for (k, v) in init.iter().enumerate() {
            set(&mut mem, base + 8 * k, *v);
        }
    }
    for l in 0..64usize {
        set(&mut mem, 0xB00 + 8 * l, 1.0);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    let want = [42.5f64.to_bits(), (-20.5f64).to_bits(), (-5.0f64).to_bits(), (-0.0f64).to_bits(), (-0.0f64).to_bits(), 3.0f64.to_bits()];
    let what = ["max of the lanes", "min of the lanes", "fmin(-5, -0) = -5", "fmax(-5, -0) = -0", "fmin(3, -0) = -0", "fmax(3, -0) = 3"];
    for (family, base) in [("buffer", 0x800usize), ("flat", 0x900), ("global", 0xA00)] {
        for k in 0..6 {
            assert_eq!(f64::from_bits(cell(base + 8 * k)), f64::from_bits(want[k]), "{family}: {}", what[k]);
            assert_eq!(cell(base + 8 * k), want[k], "{family}: {} (bits)", what[k]);
        }
    }
    for (l, v) in r.vgpr.iter().enumerate() {
        assert_eq!(cell(0xB00 + 8 * l), 2.0f64.to_bits(), "lane {l}: per-lane fmax_x2");
        let old = v[40] as u64 | (v[41] as u64) << 32;
        assert_eq!(old, 1.0f64.to_bits(), "lane {l}: GLC returns the old f64");
    }
}

/// `XCL_BLOB=<console CS container>`: HLSL size and shape of its translation.
#[test]
#[ignore]
fn blob_hlsl_shape() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    let blob = std::fs::read(p).unwrap();
    let (h, _) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    println!("HLSL {} KB, pc-dispatch: {}, if: {}, for(;;): {}", h.len() / 1024, h.matches("pc ==").count(), h.matches("if (!(").count(), h.matches("for (;;)").count());
    if let Some(out) = std::env::var_os("XCL_HLSL_OUT") {
        std::fs::write(out, &h).unwrap();
    }
}

/// `XCL_BLOB=<console CS>` `XCL_DXC_ARGS="-O3"`: DXC time and DXIL size for its HLSL (this
/// process is capped at 8 GB of commit).
#[test]
#[ignore]
fn blob_dxc_cost() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let blob = std::fs::read(p).unwrap();
    let (h, profile) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let t0 = std::time::Instant::now();
    let r = super::dxc::compile(&h, "main", profile, &extra);
    println!("HLSL {} KB, DXC {args}: {:.1} s → {}", h.len() / 1024, t0.elapsed().as_secs_f32(), match r {
        Ok(b) => format!("DXIL {} KB", b.len() / 1024),
        Err(e) => format!("error: {}", e.chars().take(200).collect::<String>()),
    });
}

/// `XCL_RAW=<file of raw RDNA2 code>`: its disassembly (reverse-engineering helper).
#[test]
#[ignore]
fn raw_rdna2_listing() {
    let Some(p) = std::env::var_os("XCL_RAW") else { return };
    let code = std::fs::read(p).unwrap();
    for l in crate::gpu::rdna2::disassemble(&code, 0, false) {
        println!("  {:06x}: {}", l.pc, l.text);
    }
}

/// Every FSSM variant of the pack through the translator (no GPU): how many translate
/// completely, per stage, and which instructions are missing (weighted by shaders).
/// `XCL_SAMPLE_RPF=…\shaders.rpf`, `XCL_SURVEY_EVERY=N` (every N-th record, default 1),
/// `XCL_SURVEY_FIND=text` (only records whose name contains it), `XCL_SURVEY_SKIP=a|b` (leave out
/// records whose name contains one of these), `XCL_SURVEY_TRACE=1` (start / time / result per program);
/// `XCL_SURVEY_DXC=1` also compiles each one with DXC (4 threads, process capped at 8 GB):
/// failures and compile times.
#[test]
#[ignore]
fn retail_translation_survey() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let every: usize = std::env::var("XCL_SURVEY_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let data = std::fs::read(path).unwrap();
    // XCL_SURVEY_SEED=<n> or "random": a random 1/every of the records (splitmix64), else every N-th
    let seed = std::env::var("XCL_SURVEY_SEED").ok().map(|v| {
        v.parse::<u64>().unwrap_or_else(|_| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
    });
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut all = crate::gpu::pack::scan(&data, 0x40);
    if let Ok(find) = std::env::var("XCL_SURVEY_FIND") {
        all.retain(|r| r.name.contains(&find));
        println!("{} records whose name contains {find:?}", all.len());
    }
    if let Ok(skip) = std::env::var("XCL_SURVEY_SKIP") {
        all.retain(|r| !skip.split('|').any(|s| !s.is_empty() && r.name.contains(s)));
        println!("{} records after skipping {skip:?}", all.len());
    }
    let records: Vec<_> = match seed {
        Some(sd) => {
            println!("random sample: seed {sd}");
            all.into_iter().enumerate().filter(|(i, _)| mix(sd ^ (*i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) % every as u64 == 0).map(|(_, r)| r).collect()
        }
        None => all.into_iter().step_by(every).collect(),
    };
    #[derive(Default)]
    struct Acc {
        // stage → (variants, complete)
        stages: BTreeMap<String, (usize, usize)>,
        // "mnemonic: reason" → shaders
        missing: BTreeMap<String, usize>,
        // error / no interface → shaders (with one example name)
        errors: BTreeMap<String, (usize, String)>,
        instructions: usize,
        // DXC: seconds per variant, failures "stage: first line" → (count, example)
        dxc_times: Vec<(f64, String)>,
        // DXC seconds / variants: fully structured, with `pc` guards
        dxc_structured: (f64, usize),
        dxc_hybrid: (f64, usize),
        hybrid: usize,
        hybrid_names: Vec<(usize, String)>,
        dxc_errors: BTreeMap<String, (usize, String)>,
    }
    let with_dxc = std::env::var_os("XCL_SURVEY_DXC").is_some() && super::dxc::location().is_ok();
    if with_dxc {
        unsafe {
            use windows_sys::Win32::System::JobObjects::*;
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = 8 << 30;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
            AssignProcessToJobObject(job, GetCurrentProcess());
        }
    }
    let workers = if with_dxc { 4 } else { std::thread::available_parallelism().map_or(4, |n| n.get()) };
    let acc = Mutex::new(Acc::default());
    let next = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for f in &sh.fssm {
                    let hw = f.stages.first().map_or(0, |s| s.hw);
                    let stage = format!("{} ({})", sh.stage, crate::gpu::xbox::hw_stage_name(hw));
                    let res = std::panic::catch_unwind(|| -> Result<(super::Translation, &'static str), String> {
                        let mode = super::interface(&blob, &sh, f).ok_or("no interface")?;
                        let profile = match &mode {
                            Mode::Vertex(_) => "vs_6_6",
                            Mode::Pixel(_) => "ps_6_6",
                            // MS (NGG) and GS run as compute shaders feeding a pass-through mesh shader
                            Mode::Compute(_) | Mode::Mesh(_) | Mode::Geometry(_) | Mode::Kernel => "cs_6_6",
                        };
                        let p = f.programs.first().ok_or("FSSM without a program")?;
                        let code = &f.data[p.offset..p.offset + p.len];
                        // XCL_SURVEY_WINDOWS=N: translate for N memory windows (above 4 GiB)
                        let mem_windows = std::env::var("XCL_SURVEY_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                        let trace = std::env::var_os("XCL_SURVEY_TRACE").is_some();
                        if trace {
                            eprintln!("translate start {} ({} B)", r.name, code.len());
                        }
                        let t1 = std::time::Instant::now();
                        let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), mem_windows, ..Default::default() });
                        if trace {
                            let state = match &t {
                                Ok(t) if t.complete() => "complete".to_string(),
                                Ok(t) => format!("partial {:?}", t.unsupported_why.keys().collect::<Vec<_>>()),
                                Err(e) => format!("error {e}"),
                            };
                            eprintln!("translate done {} {:.2} s {state}", r.name, t1.elapsed().as_secs_f64());
                        }
                        Ok((t?, profile))
                    });
                    // DXC outside the lock
                    let guarded = matches!(&res, Ok(Ok((t, _))) if t.hlsl.contains("pc =="));
                    // XCL_SURVEY_ONLY=a|b: DXC only for programs whose HLSL contains one of these
                    let only = std::env::var("XCL_SURVEY_ONLY").ok();
                    let wanted = |h: &str| only.as_ref().is_none_or(|o| o.split('|').any(|x| h.contains(x)));
                    let dxc = match (&res, with_dxc) {
                        (Ok(Ok((t, profile))), true) if wanted(&t.hlsl) => {
                            if std::env::var_os("XCL_SURVEY_TRACE").is_some() {
                                eprintln!("dxc start {} ({} KB)", r.name, t.hlsl.len() / 1024);
                            }
                            let t1 = std::time::Instant::now();
                            // XCL_DXC_ARGS="-O1 …": extra DXC arguments (comparing optimisation levels)
                            let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
                            let args: Vec<&str> = args.split_whitespace().collect();
                            let r = super::dxc::compile(&t.hlsl, "main", profile, &args);
                            // the specific message (validation failures list it after "validation errors")
                            let first = |e: &str| {
                                e.lines().find(|l| l.contains("error") && !l.contains("validation errors")).or_else(|| e.lines().find(|l| l.contains("error"))).unwrap_or(e.lines().next().unwrap_or("")).to_string()
                            };
                            Some((t1.elapsed().as_secs_f64(), r.err().map(|e| first(&e))))
                        }
                        _ => None,
                    };
                    let mut a = acc.lock().unwrap();
                    let e = a.stages.entry(stage.clone()).or_default();
                    e.0 += 1;
                    let error = |a: &mut Acc, msg: String| {
                        let x = a.errors.entry(format!("{stage}: {msg}")).or_insert((0, r.name.clone()));
                        x.0 += 1;
                    };
                    a.hybrid += guarded as usize;
                    if let (true, Ok(Ok((t, _)))) = (guarded, &res) {
                        a.hybrid_names.push((t.instructions, r.name.clone()));
                    }
                    if let Some((secs, err)) = dxc {
                        a.dxc_times.push((secs, r.name.clone()));
                        let b = if guarded { &mut a.dxc_hybrid } else { &mut a.dxc_structured };
                        b.0 += secs;
                        b.1 += 1;
                        if let Some(e) = err {
                            // drop the file:line:col prefix so equal errors group
                            let e = e.split("error:").nth(1).unwrap_or(&e).trim().to_string();
                            let x = a.dxc_errors.entry(format!("{stage}: {e}")).or_insert((0, r.name.clone()));
                            x.0 += 1;
                        }
                    }
                    match res {
                        Ok(Ok((t, _))) => {
                            a.instructions += t.instructions;
                            if t.complete() {
                                a.stages.get_mut(&stage).unwrap().1 += 1;
                            }
                            for k in t.unsupported_why.keys() {
                                *a.missing.entry(k.clone()).or_default() += 1;
                            }
                        }
                        Ok(Err(m)) => error(&mut a, m),
                        Err(_) => error(&mut a, "translator panic".into()),
                    }
                }
            });
        }
    });
    let _ = std::panic::take_hook();
    let a = acc.into_inner().unwrap();
    let (all, done): (usize, usize) = a.stages.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("records: {} (1 in {every}), FSSM variants: {all}, fully translated: {done} ({:.1}%), instructions: {}, time {:.1} s", records.len(), 100.0 * done as f64 / all.max(1) as f64, a.instructions, t0.elapsed().as_secs_f64());
    println!("with pc blocks (not fully structured): {} / {all}", a.hybrid);
    let mut hn = a.hybrid_names;
    hn.sort();
    println!("  smallest: {:?}", hn.iter().take(6).collect::<Vec<_>>());
    for (st, (n, ok)) in &a.stages {
        println!("  {st:<14} {ok:>6} / {n:<6} ({:.1}%)", 100.0 * *ok as f64 / (*n).max(1) as f64);
    }
    let mut missing: Vec<_> = a.missing.into_iter().collect();
    missing.sort_by(|x, y| y.1.cmp(&x.1));
    println!("missing instructions (in how many variants):");
    for (k, n) in missing.iter().take(40) {
        println!("  {n:>6}  {k}");
    }
    let mut errors: Vec<_> = a.errors.into_iter().collect();
    errors.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
    println!("errors:");
    for (k, (n, ex)) in errors.iter().take(20) {
        println!("  {n:>6}  {k}   (e.g. {ex})");
    }
    if with_dxc {
        let mut t = a.dxc_times;
        t.sort_by(|x, y| x.0.total_cmp(&y.0));
        let q = |f: f64| t.get(((t.len() as f64 * f) as usize).min(t.len().saturating_sub(1))).map_or(0.0, |x| x.0);
        let failed: usize = a.dxc_errors.values().map(|v| v.0).sum();
        println!(
            "DXC structured: {} variants, {:.0} s (mean {:.2} s); with pc blocks: {} variants, {:.0} s (mean {:.2} s)",
            a.dxc_structured.1,
            a.dxc_structured.0,
            a.dxc_structured.0 / a.dxc_structured.1.max(1) as f64,
            a.dxc_hybrid.1,
            a.dxc_hybrid.0,
            a.dxc_hybrid.0 / a.dxc_hybrid.1.max(1) as f64
        );
        println!("DXC: {} compiled, {failed} errors; time median {:.2} s, p90 {:.2} s, max {:.1} s ({}), total {:.0} s", t.len(), q(0.5), q(0.9), t.last().map_or(0.0, |x| x.0), t.last().map_or("", |x| x.1.as_str()), t.iter().map(|x| x.0).sum::<f64>());
        for (secs, name) in t.iter().rev().take(10) {
            println!("  {secs:>6.2} s  {name}");
        }
        let mut errs: Vec<_> = a.dxc_errors.into_iter().collect();
        errs.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
        for (k, (n, ex)) in errs.iter().take(20) {
            println!("  {n:>6}  {k}   (e.g. {ex})");
        }
    }
}

/// Shape of the HLSL and DXC cost of named retail shaders (reverse engineering of compile
/// times): `XCL_SAMPLE_RPF`, `XCL_NAMES=a,b` (substring of the record name), `XCL_DXC_ARGS`,
/// `XCL_HLSL_OUT=<folder>` (writes the HLSL). The process is capped at 8 GB.
#[test]
#[ignore]
fn retail_shader_cost() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_NAMES")) else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| !n.is_empty() && r.name.contains(n)) {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        // effect-qualified names (`effect:CS_…`) are not file names (`:` opens an NTFS stream)
        let stem = r.name.replace(['/', '\\', ':'], "_");
        for f in &sh.fssm {
            let Some(mode) = super::interface(&blob, &sh, f) else { continue };
            let profile = match &mode {
                Mode::Vertex(_) => "vs_6_6",
                Mode::Pixel(_) => "ps_6_6",
                _ => "cs_6_6",
            };
            let p = f.programs[0];
            let code = &f.data[p.offset..p.offset + p.len];
            let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }).unwrap();
            let owned = t.hlsl.clone();
            let h = &owned;
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
                eprintln!("HLSL written: {} ({} KB)", r.name, h.len() / 1024);
            }
            let t0 = std::time::Instant::now();
            let r2 = super::dxc::compile(h, "main", profile, &extra);
            let secs = t0.elapsed().as_secs_f64();
            if let (Err(_), Ok(dir)) = (&r2, std::env::var("XCL_HLSL_OUT")) {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.failed.hlsl")), h);
            }
            println!(
                "{} [{profile}] instructions {}, HLSL {} KB, lines {}, pc-dispatch {}, if {}, for(;;) {}, select {} → DXC {:.2} s, {}",
                r.name,
                t.instructions,
                h.len() / 1024,
                h.lines().count(),
                h.matches("pc ==").count(),
                h.matches("if (").count(),
                h.matches("for (;;)").count(),
                h.matches("select(").count(),
                secs,
                r2.as_ref().map(|b| format!("DXIL {} KB", b.len() / 1024)).unwrap_or_else(|e| e.lines().next().unwrap_or("").to_string())
            );
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
            }
        }
    }
}

/// ISA coverage: every instruction the decoder knows (RDNA2 spec + console extensions), one
/// at a time with zero operands, through the translator (compute kernel mode, then pixel
/// mode for graphics-only ones). Per encoding: translated / all, and the missing ones.
#[test]
#[ignore]
fn isa_coverage() {
    use crate::gpu::rdna2::{all_instructions, encode};
    use crate::gpu::rdna2_spec::ENCODINGS;
    use std::collections::BTreeMap;
    let mut per_enc: BTreeMap<&str, (usize, usize, Vec<&str>)> = BTreeMap::new();
    let try_mode = |code: &[u8], mode: Mode| match translate(code, &Options { mode, ..Default::default() }) {
        Ok(t) => t.complete(),
        Err(_) => false,
    };
    std::panic::set_hook(Box::new(|_| {}));
    for inst in all_instructions() {
        // the first plain encoding (not an SDWA / DPP / literal variant)
        let Some(ie) = inst.encs.iter().find(|e| ENCODINGS[e.encoding as usize].primary) else { continue };
        let enc = ENCODINGS[ie.encoding as usize].name;
        let lit = inst.always_literal.then_some(0);
        // operands that must not be 0: MIMG channel mask, VOP3 SDST (VCC)
        let fields: &[(&str, u64)] = match enc {
            "ENC_MIMG" => &[("DMASK", 1)],
            "VOP3_SDST_ENC" => &[("SDST", 106)],
            _ if inst.name == "V_SWAPREL_B32" => &[("SRC0", V)],
            _ if inst.name == "V_MQSAD_U32_U8" => &[("SRC2", V)],
            _ => &[],
        };
        let Ok(mut code) = encode(inst.name, enc, fields, lit) else { continue };
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let ok = std::panic::catch_unwind(|| {
            try_mode(&code, Mode::Kernel) || try_mode(&code, Mode::Pixel(super::PsInterface { inputs: vec![], outputs: vec![], input_ena: 0x7F, attrs: None }))
        })
        .unwrap_or(false);
        let e = per_enc.entry(enc.trim_start_matches("ENC_")).or_default();
        e.1 += 1;
        if ok {
            e.0 += 1;
        } else {
            e.2.push(inst.name);
        }
    }
    let _ = std::panic::take_hook();
    let (ok, all) = per_enc.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("RDNA2 instructions (spec + console): {all}, translated: {ok} ({:.1}%)", 100.0 * ok as f64 / all.max(1) as f64);
    for (enc, (ok, all, missing)) in &per_enc {
        let m: Vec<String> = missing.iter().map(|n| n.to_lowercase()).collect();
        let full = std::env::var_os("XCL_ISA_FULL").is_some();
        println!("  {enc:<12} {ok:>4} / {all:<4} missing: {}", if m.len() > 14 && !full { format!("{} … (+{})", m[..14].join(", "), m.len() - 14) } else { m.join(", ") });
    }
}

/// Census of the cross-lane / control-flow forms the translator special-cases: every
/// `ds_swizzle_b32` offset in the pack, and the instructions around each `s_setpc_b64` /
/// `s_swappc_b64` of the records whose name contains one of `XCL_NAMES` (`a,b`).
#[test]
#[ignore]
fn retail_swizzle_and_setpc_census() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let names = std::env::var("XCL_NAMES").unwrap_or_default();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let swz: Mutex<BTreeMap<u64, (usize, String)>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                let show = names.split(',').any(|n| !n.is_empty() && r.name.contains(n));
                for f in &sh.fssm {
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        if let (true, Ok(dir)) = (show, std::env::var("XCL_RAW_OUT")) {
                            let stem = r.name.replace(['/', '\\', ':'], "_");
                            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.{:x}.bin", p.offset)), code);
                        }
                        let mut off = 0usize;
                        let mut recent: Vec<usize> = Vec::new();
                        while off + 4 <= code.len() {
                            let Ok(d) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { off += 4; continue };
                            let name = d.inst.name;
                            if name.eq_ignore_ascii_case("ds_swizzle_b32") {
                                let v = |n: &str| crate::gpu::rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
                                let mut m = swz.lock().unwrap();
                                m.entry(v("OFFSET0") | v("OFFSET1") << 8).or_insert((0, r.name.clone())).0 += 1;
                            }
                            if show && (name.eq_ignore_ascii_case("s_setpc_b64") || name.eq_ignore_ascii_case("s_swappc_b64")) {
                                let from = recent.get(recent.len().saturating_sub(12)).copied().unwrap_or(off);
                                let to = (off + 24).min(code.len());
                                let text: Vec<String> = crate::gpu::rdna2::disassemble(&code[from..to], from as u64, false)
                                    .iter()
                                    .map(|l| format!("  {:06x}: {}", l.pc, l.text))
                                    .collect();
                                println!("{} @{off:#x}:
{}", r.name, text.join("
"));
                            }
                            recent.push(off);
                            off += d.len;
                        }
                    }
                }
            });
        }
    });
    println!("ds_swizzle_b32 offsets (uses, example):");
    for (o, (n, ex)) in swz.into_inner().unwrap() {
        println!("  {o:#06x} {n:>6}  {ex}");
    }
}

/// Every cross-lane read `upper_lanes` resolves (or not) in the pack's wave64 VS / PS
/// programs, with the instructions around it: `XCL_SAMPLE_RPF`, `XCL_CENSUS_OUT=<file>` (the
/// listing; marks `>>` unresolved, `C ` constant, `S ` shadow, `N ` native), `XCL_CENSUS_FULL=1`
/// (whole programs), `XCL_CENSUS_UNKNOWN=1` (only programs with an unresolved read). Prints
/// sites and programs (all, and distinct code) per resolution.
#[test]
#[ignore]
fn retail_wave64_graphics_lane_census() {
    use super::upper_lanes::Read;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let (Ok(path), Ok(out)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_CENSUS_OUT")) else { return };
    let full = std::env::var_os("XCL_CENSUS_FULL").is_some();
    let only_unknown = std::env::var_os("XCL_CENSUS_UNKNOWN").is_some();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let text: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    // resolution → (sites, programs, distinct programs by code hash)
    let stats: Mutex<BTreeMap<&'static str, (usize, usize, BTreeSet<u64>)>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for (fi, f) in sh.fssm.iter().enumerate() {
                    let mode = match super::interface(&blob, &sh, f) {
                        Some(Mode::Vertex(_)) => "VS",
                        Some(Mode::Pixel(_)) => "PS",
                        _ => continue,
                    };
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        if !super::detect_wave64(code) {
                            continue;
                        }
                        let mut insts = Vec::new();
                        let mut off = 0;
                        while off + 4 <= code.len() {
                            let Ok(d) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { break };
                            off += d.len;
                            insts.push(d);
                        }
                        crate::gpu::rdna2::resolve_far_jumps(&mut insts);
                        let a = super::upper_lanes::analyze(&insts);
                        if a.reads.is_empty() || (only_unknown && !a.reads.values().any(|r| *r == Read::Unknown)) {
                            continue;
                        }
                        let hash = crate::gpu::shader::fnv64(code);
                        {
                            let mut st = stats.lock().unwrap();
                            let mut kinds = BTreeSet::new();
                            for r in a.reads.values() {
                                let k = match r {
                                    Read::Native => "native (lanes >= 32 unwritten / inactive)",
                                    Read::Const(_) => "constant (whole-wave identity)",
                                    Read::Shadow => "spill slot shadow",
                                    Read::Unknown => "unresolved",
                                };
                                st.entry(k).or_default().0 += 1;
                                kinds.insert(k);
                            }
                            for k in kinds {
                                let e = st.entry(k).or_default();
                                e.1 += 1;
                                e.2.insert(hash);
                            }
                        }
                        let lines = crate::gpu::rdna2::disassemble(code, 0, false);
                        let mut s = format!("=== {} FSSM#{fi} {mode} {} B {hash:016x}
", r.name, code.len());
                        if a.reads.values().any(|r| *r == Read::Unknown) {
                            // where EXEC's upper word and the VGPRs' lanes ≥ 32 stopped being known
                            super::upper_lanes::TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
                            super::upper_lanes::analyze(&insts);
                            for l in super::upper_lanes::TRACE.with(|t| t.borrow_mut().take()).unwrap_or_default() {
                                s.push_str(&format!("   trace {l}
"));
                            }
                        }
                        let mark = |pc: u64| match a.reads.get(&pc) {
                            Some(Read::Unknown) => ">>",
                            Some(Read::Const(_)) => "C ",
                            Some(Read::Shadow) => "S ",
                            Some(Read::Native) => "N ",
                            None => "  ",
                        };
                        let sites: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| a.reads.contains_key(&l.pc)).map(|(k, _)| k).collect();
                        let mut shown = 0usize;
                        for &k in &sites {
                            let (from, to) = if full { (0, lines.len()) } else { (k.saturating_sub(40), (k + 8).min(lines.len())) };
                            let from = from.max(shown);
                            if from > shown || shown == 0 {
                                s.push_str("  ...
");
                            }
                            for l in &lines[from..to.max(from)] {
                                s.push_str(&format!("{} {:05x}: {}
", mark(l.pc), l.pc, l.text));
                            }
                            shown = shown.max(to);
                        }
                        text.lock().unwrap().push((r.name.clone(), s));
                    }
                }
            });
        }
    });
    let mut v = text.into_inner().unwrap();
    v.sort();
    println!("{} programs listed", v.len());
    for (k, (sites, progs, distinct)) in stats.into_inner().unwrap() {
        println!("  {k:<45} {sites:>6} sites {progs:>6} programs {:>6} distinct", distinct.len());
    }
    std::fs::write(out, v.into_iter().map(|(_, s)| s).collect::<String>()).unwrap();
}

/// The 32-lane form of the pack's wave64 VS / PS (`upper_lanes`): every program whose
/// translation it changes is compiled with DXC. Digests of every program's translation in the
/// native form (`split_wave64` off – the form a GPU with 64-lane waves gets) and of the other
/// modes' 32-lane form are printed, to compare with another build: `XCL_SAMPLE_RPF`;
/// `XCL_FORMS_NO_DXC=1` skips the compiles.
#[test]
#[ignore]
fn retail_wave64_graphics_lane_forms() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let dxc = std::env::var_os("XCL_FORMS_NO_DXC").is_none();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let (digest, programs, changed, compiled) = (AtomicU64::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let split_digest = AtomicU64::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for (fi, f) in sh.fssm.iter().enumerate() {
                    let mode = super::interface(&blob, &sh, f).unwrap_or(Mode::Kernel);
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        let wave64 = super::detect_wave64(code);
                        let opt = Options { mode: mode.clone(), wave64, ..Default::default() };
                        let Ok(native) = translate(code, &opt) else { continue };
                        programs.fetch_add(1, Ordering::Relaxed);
                        let key = format!("{} FSSM#{fi} {:x}", r.name, p.offset);
                        let h = crate::gpu::shader::fnv64(format!("{key}
{}", native.hlsl).as_bytes());
                        digest.fetch_add(h, Ordering::Relaxed);
                        let profile = match &mode {
                            Mode::Vertex(_) => "vs_6_6",
                            Mode::Pixel(_) => "ps_6_6",
                            _ => {
                                // the other modes' 32-lane form: unchanged by `upper_lanes`
                                if let Ok(t) = translate(code, &Options { split_wave64: true, ..opt }) {
                                    split_digest.fetch_add(crate::gpu::shader::fnv64(format!("{key}\\n{}", t.hlsl).as_bytes()), Ordering::Relaxed);
                                }
                                continue;
                            }
                        };
                        if !wave64 {
                            continue;
                        }
                        let t = translate(code, &Options { split_wave64: true, ..opt }).unwrap();
                        if t.hlsl == native.hlsl {
                            continue;
                        }
                        changed.fetch_add(1, Ordering::Relaxed);
                        if dxc {
                            match super::dxc::compile(&t.hlsl, "main", profile, &[]) {
                                Ok(_) => {
                                    compiled.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(e) => errors.lock().unwrap().push(format!("{key}: {}", e.lines().take(4).collect::<Vec<_>>().join(" | "))),
                            }
                        }
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    println!(
        "{} programs, native-form digest {:016x}, 32-lane-form digest of the other modes {:016x}; 32-lane form differs for {} wave64 VS / PS, {} compiled, {} DXC errors",
        programs.into_inner(),
        digest.into_inner(),
        split_digest.into_inner(),
        changed.into_inner(),
        compiled.into_inner(),
        errors.len()
    );
    for e in errors.iter().take(10) {
        println!("  {e}");
    }
    assert!(errors.is_empty());
}

/// `ds_swizzle_b32`'s four modes against the RDNA ISA (its FFT examples, rotate, quad
/// permute, and the bit mode with xor ≥ 16, which retail compute shaders use to swap the
/// 16-lane halves: 0x401F).
#[test]
fn ds_swizzle_modes_follow_the_isa() {
    use super::emit::swizzle_lane;
    let lanes = |off: u64| (0..32).map(|i| swizzle_lane(off, i)).collect::<Vec<u32>>();
    // ISA examples (inputs 1..0x20, so each value − 1 is the source lane)
    let fft0 = [0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17, 0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c, 0x08, 0x18, 0x10, 0x20];
    let fft10 = [0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c, 0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e, 0x14, 0x1c, 0x18, 0x20];
    assert_eq!(lanes(0xE000), fft0.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE010), fft10.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE01F), (0..32).collect::<Vec<u32>>(), "FFT mask 0x1F: no swizzle");
    // rotate left by 3 / right by 5 (offset[10]), whole 32 lanes
    assert_eq!(lanes(0xC000 | 3 << 5), (0..32).map(|i| (i + 3) % 32).collect::<Vec<u32>>());
    assert_eq!(lanes(0xC400 | 5 << 5), (0..32).map(|i| (i + 27) % 32).collect::<Vec<u32>>());
    // quad permute [3, 2, 1, 0]
    assert_eq!(lanes(0x8000 | 0x1B), (0..32).map(|i| (i & !3) | (3 - (i & 3))).collect::<Vec<u32>>());
    // bit mode: and 0x1F, xor 16 (swap halves) / xor 18
    assert_eq!(lanes(0x401F), (0..32).map(|i| i ^ 16).collect::<Vec<u32>>());
    assert_eq!(lanes(0x481F), (0..32).map(|i| i ^ 18).collect::<Vec<u32>>());
    assert_eq!(lanes(0x041F), (0..32).map(|i| i ^ 1).collect::<Vec<u32>>());
}

#[test]
fn gpu_ds_swizzle_modes() {
    if !gpu_ready() {
        return;
    }
    let offsets = [0x401Fu64, 0x481F, 0xE000, 0xE010, 0xC000 | 3 << 5, 0xC400 | 5 << 5 | 0x3, 0x8000 | 0x1B];
    let list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = offsets
        .iter()
        .enumerate()
        .map(|(k, &o)| ("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", 16 + k as u64), ("ADDR", 0), ("OFFSET0", o & 0xFF), ("OFFSET1", o >> 8)], None))
        .collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(n, e, f, l)| (*n, *e, f.as_slice(), *l)).collect();
    let code = prog(&list);
    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64u32 {
        for (k, &o) in offsets.iter().enumerate() {
            let want = 100 + ((l & 0x20) | super::emit::swizzle_lane(o, l));
            assert_eq!(r.vgpr[l as usize][16 + k], want, "lane {l} offset {o:#06x}");
        }
    }
}

/// The compiler's far jump (`s_getpc_b64`, add of a 64-bit constant, `s_setpc_b64`) out of
/// `s_branch` range is a direct branch: it gets its target, the program stays structured and
/// complete, and the code it jumps over does not run.
#[test]
fn far_jumps_are_direct_branches() {
    let far = |sub: bool, offset: u32| -> Vec<u8> {
        let (lo, hi) = if sub { ("S_SUB_U32", "S_SUBB_U32") } else { ("S_ADD_U32", "S_ADDC_U32") };
        prog(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
            (lo, "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(offset)),
            (hi, "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)], None), // 0x14: v1 = 7 (jumped over)
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 137)], None), // 0x18: v2 = 9
        ])
    };
    let decoded = |code: &[u8]| {
        let mut v = Vec::new();
        let mut off = 0;
        while off + 4 <= code.len() {
            let d = crate::gpu::rdna2::decode(&code[off..], off as u64).unwrap();
            off += d.len;
            v.push(d);
        }
        crate::gpu::rdna2::resolve_far_jumps(&mut v);
        v
    };
    // s_getpc_b64 returns 0x4 (the next instruction): 0x4 + 0x14 = 0x18
    let code = far(false, 0x14);
    let insts = decoded(&code);
    assert_eq!(insts[3].far_target, Some(0x18));
    assert_eq!(crate::gpu::rdna2::branch_target(&insts[3]), Some(0x18));
    // backward form: 0x4 − 4 = 0x0
    assert_eq!(decoded(&far(true, 4))[3].far_target, Some(0));
    // an s_setpc_b64 of another register pair stays indirect
    let mut other = code.clone();
    other[0x10..0x14].copy_from_slice(&encode("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None).unwrap());
    assert_eq!(decoded(&other)[3].far_target, None);
    let t = super::translate(&other, &Options::default()).unwrap();
    assert!(!t.complete());

    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert!(!t.hlsl.contains("pc =="), "far jump left the structured form:\n{}", t.hlsl);
    if !gpu_ready() {
        return;
    }
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[1] = 1;
        r[2] = 2;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1, 9), "lane {l}");
    }
}

/// The console compiler keeps constant tables in the code region after the program and
/// addresses them through `s_getpc_b64`: its address, a table at 0x100, two loads through
/// it, and a load through an ordinary pointer in s12:s13.
fn code_table_program() -> Vec<u8> {
    prog(&[
        ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None), // s4 = 0x4
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(0xFC)),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 2), ("OFFSET", 8), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("SOFFSET", 125)], None),
        ("GLOBAL_LOAD_DWORDX2", "ENC_FLAT_GLBL", &[("VDST", 8), ("ADDR", 0), ("SADDR", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 5)], None),
        ("FLAT_LOAD_DWORDX3", "ENC_FLAT", &[("VDST", 12), ("ADDR", 4), ("OFFSET", 4)], None),
        ("GLOBAL_LOAD_DWORDX3", "ENC_FLAT_GLBL", &[("VDST", 16), ("ADDR", 4), ("SADDR", 0x7D)], None),
        ("FLAT_LOAD_USHORT", "ENC_FLAT", &[("VDST", 20), ("ADDR", 4), ("OFFSET", 3)], None),
    ])
}

fn code_table() -> super::CodeData {
    super::CodeData { at: 0x100, words: vec![0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444] }
}

#[test]
fn code_region_table_bounds_are_relative_to_the_program() {
    use crate::gpu::xbox::{Fssm, Program};
    let p = Program { offset: 5, len: 8 };
    let mut f = Fssm { data: (0..32).collect(), data_blocks: vec![Program { offset: 0, len: 4 }], ..Default::default() };
    assert!(super::code_data(&f, &p).is_none());
    f.data_blocks.extend([Program { offset: 24, len: 4 }, Program { offset: 19, len: 2 }]);
    let table = super::code_data(&f, &p).unwrap();
    assert_eq!(table.at, 12);
    assert_eq!(table.words, [0x14131211, 0x18171615, 0x001b1a19]);
}

fn far_jump_table_program() -> Vec<u8> {
    prog(&[
        ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(0x14)),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0xe8), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("SOFFSET", 125)], None),
    ])
}

/// With the code region's data (`Options::code_data`) a scalar load through an `s_getpc_b64`
/// address reads the table. A far jump can leave its target in a register pair used by a load.
#[test]
fn code_region_tables_are_read_through_getpc_addresses() {
    let code = code_table_program();
    let with = super::translate(&code, &Options { code_data: Some(code_table()), ..Default::default() }).unwrap();
    assert!(with.complete(), "{:?}", with.unsupported);
    assert!(with.hlsl.contains("static const uint xcl_code_d[4] = { 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u };"), "{}", with.hlsl);
    assert!(with.hlsl.contains("uint xcl_code(uint a) { uint i = (a - 0x00000100u) >> 2; return i < 4u ? xcl_code_d[i] : 0u; }"), "{}", with.hlsl);
    assert!(with.hlsl.contains("s4 = 0x00000004u; s5 = 0xC0DE0000u;"), "{}", with.hlsl);
    assert_eq!(with.hlsl.matches("_pc) {").count(), 3, "every scalar load tells the code region from guest memory:\n{}", with.hlsl);

    let without = super::translate(&code, &Options::default()).unwrap();
    assert!(without.hlsl.contains("s4 = 0x00000004u; s5 = 0u;"), "{}", without.hlsl);
    assert!(!without.hlsl.contains("xcl_code") && !without.hlsl.contains("0xC0DE0000u") && !without.hlsl.contains("_pc"), "{}", without.hlsl);

    let far = far_jump_table_program();
    let plain = super::translate(&far, &Options::default()).unwrap();
    let given = super::translate(&far, &Options { code_data: Some(code_table()), ..Default::default() }).unwrap();
    assert!(plain.complete(), "{:?}", plain.unsupported);
    assert!(given.complete(), "{:?}", given.unsupported);
    assert!(!plain.hlsl.contains("xcl_code"));
    assert!(given.hlsl.contains("s5 = 0xC0DE0000u;"), "a resolved jump's target remains a code pointer");
    let no_getpc = prog(&[("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("SOFFSET", 125)], None)]);
    assert_eq!(super::translate(&no_getpc, &Options::default()).unwrap().hlsl,
        super::translate(&no_getpc, &Options { code_data: Some(code_table()), ..Default::default() }).unwrap().hlsl);
}

/// The table is not guest memory: the kernel's `mem` holds other values at the table's
/// program-relative addresses, and the loads through the `s_getpc_b64` address return the
/// table's words. The load through s12:s13 reads `mem` either way.
#[test]
fn gpu_code_region_tables_are_not_guest_memory() {
    if !gpu_ready() {
        return;
    }
    let code = code_table_program();
    let mut mem = vec![0u32; 68];
    mem[64..68].copy_from_slice(&[0xDEC0_0000, 0xDEC0_0001, 0xDEC0_0002, 0xDEC0_0003]);
    mem[32] = 0x5555_AAAA;
    let mut st = KernelState::default();
    st.sgpr[12] = 0x80;
    for (lane, v) in st.vgpr.iter_mut().enumerate() {
        v[0] = 4 * (lane as u32 & 1);
    }
    let r = runner::run_code_with_options(&code, &st, &mem, &Options { code_data: Some(code_table()), ..runner::options(Mode::Kernel) }).unwrap();
    assert_eq!(r.sgpr[8..12], [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x5555_AAAA], "the table's words and the guest dword");
    let words = code_table().words;
    for (lane, v) in r.vgpr.iter().enumerate() {
        let i = lane & 1;
        assert_eq!(v[8..10], words[i..i + 2], "global scalar base, lane {lane}");
        assert_eq!(v[12..15], words[1..4], "flat vector base, lane {lane}");
        assert_eq!(v[16..19], words[..3], "global vector base, lane {lane}");
        assert_eq!(v[20], 0x2211, "unaligned ushort, lane {lane}");
    }
    let r = runner::run_code_with_options(&code, &st, &mem, &runner::options(Mode::Kernel)).unwrap();
    assert_eq!(r.sgpr[8..12], [0xDEC0_0000, 0xDEC0_0001, 0xDEC0_0002, 0x5555_AAAA], "without the data: guest memory, as a kernel test places its code");
    let far = far_jump_table_program();
    for (table, want) in [(None, 0xDEC0_0000), (Some(code_table()), 0x1111_1111)] {
        let r = runner::run_code_with_options(&far, &st, &mem, &Options { code_data: table, ..runner::options(Mode::Kernel) }).unwrap();
        assert_eq!(r.vgpr[0][1], 0, "the far jump skipped the move");
        assert_eq!((r.sgpr[4], r.sgpr[8], r.sgpr[11]), (0x18, want, 0x5555_AAAA));
    }
}

/// Split wave64: a meet in a branch only one half takes keeps the bounded spin. The GS phase of
/// geometry mode runs in the lower half alone (`if (gi < 32u)`), so the EXEC ballot it starts
/// with (`xcl_x_ballot`) has no other half to meet: a group barrier there would wait for the
/// upper half forever. The same ballot in a one-wave64 compute group meets at a group barrier.
#[test]
fn split_wave64_meet_in_a_half_divergent_branch_keeps_the_spin() {
    let end: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[("S_ENDPGM", "ENC_SOPP", &[], None)];
    let es = prog(&[("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None)]);
    let mut code = es.clone();
    code.extend(prog(&[("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None), end[0]]));
    let position = crate::gpu::dxbc::SigElement { name: "SV_Position".into(), index: 0, system_value: 1, component_type: 3, register: 0, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let g = super::GsInterface { outputs: vec![position], max_vertex_count: 1, gs_start: Some(es.len() as u64), ..Default::default() };
    let gs = translate(&code, &Options { mode: Mode::Geometry(g), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(gs.complete(), "{:?}", gs.unsupported_why);
    let lower = gs.hlsl.split_once("if (gi < 32u)").map(|(_, rest)| rest).unwrap_or_else(|| panic!("no lower-half GS phase\n{}", gs.hlsl));
    assert!(lower.contains("xcl_x_ballot(true)"), "{}", gs.hlsl);
    assert!(gs.hlsl.contains("#define XCL_X_SPIN 1"), "{}", gs.hlsl);
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let one_wave = translate(&prog(end), &Options { mode: Mode::Compute(cs.clone()), wave64: true, split_wave64: true, legacy_compute_wave64: true, thread_limits: false, ..Default::default() }).unwrap();
    assert!(one_wave.hlsl.contains("xcl_x_ballot(true)"), "{}", one_wave.hlsl);
    // with the dispatch's thread limits the launch ballot is of the lanes below them, met the same way
    let limited = translate(&prog(end), &Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: true, thread_limits: true, ..Default::default() }).unwrap();
    assert!(limited.hlsl.contains("uint2 e0 = xcl_x_ballot(all(gid * uint3(64u, 1u, 1u) + gtid <= xcl_tl));"), "{}", limited.hlsl);
    if std::env::var_os("XCL_WAVE64_SPIN").is_none() {
        for t in [&one_wave, &limited] {
            assert!(t.hlsl.contains("#define XCL_X_SPIN 0\n#define XCL_X_PAIR 1"), "{}", t.hlsl);
        }
    }
    if super::dxc::location().is_ok() {
        for t in [&gs, &one_wave, &limited] {
            super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        }
    }
}

// xsemu: from xcl-dev/xcl nvidia-dev 9b455a1a,67a08dbe crates/xcl/src/gpu/recomp/tests.rs
/// The split form keeps the console wave's order of LDS accesses across its halves
/// (`xcl_x_lw`): an LDS read meets the other half first when LDS may have been written since the
/// halves last met, a write when it may have been read (on the console every lane reads before
/// any lane of a later instruction writes), an atomic in either case; every meet and `s_barrier`
/// clears the bits. A program that writes no LDS gets no such meet, nor does the native form.
#[test]
fn split_wave64_lds_reads_meet_after_writes() {
    type Inst = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);
    let write: Inst = ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None);
    let read: Inst = ("DS_READ_B32", "ENC_DS", &[("VDST", 2), ("ADDR", 1)], None);
    let add: Inst = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None);
    let barrier: Inst = ("S_BARRIER", "ENC_SOPP", &[], None);
    let lane: Inst = ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)], None);
    // a read waits for writes (bit 1), a write for reads (bit 2), an atomic for both
    const AFTER_WRITE: &str = "if ((xcl_x_lw & 1u) != 0u) xcl_x_meet();";
    const AFTER_READ: &str = "if ((xcl_x_lw & 2u) != 0u) xcl_x_meet();";
    const AFTER_ANY: &str = "if ((xcl_x_lw & 3u) != 0u) xcl_x_meet();";
    const WROTE: &str = "xcl_x_lw |= 1u;";
    const READ: &str = "xcl_x_lw |= 2u;";
    const BOTH: &str = "xcl_x_lw |= 3u;";
    const CLEAR: &str = "xcl_x_lw = 0u;";
    let split = Options { mode: Mode::Kernel, wave64: true, split_wave64: true, ..Default::default() };
    let hlsl = |list: &[Inst], o: &Options| {
        let t = translate(&prog(list), o).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if super::dxc::location().is_ok() {
            super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        }
        t.hlsl
    };
    let main = |h: &str| h[h.find("void main(").unwrap()..].to_string();
    // every meet clears the bits, also one without a partner half
    let h = hlsl(&[write, read], &split);
    let meet = &h[h.find("bool xcl_x_meet()").unwrap()..];
    assert!(meet.find(CLEAR).unwrap() < meet.find("if (xcl_xn == 0u) return false;").unwrap(), "{meet}");
    // the hazard: a write, then a read with nothing between: the read meets first
    let m = main(&h);
    let (wrote, wait, at) = (m.find(WROTE).unwrap(), m.find(AFTER_WRITE).unwrap(), m.rfind("= xcl_lds[").unwrap());
    assert!(wrote < wait && wait < at, "the meet between the write and the read:\n{m}");
    // the other way round: a read, then a write – the write meets first, so the other half has
    // read before this one writes
    let m = main(&hlsl(&[read, write], &split));
    let (read_at, wait, write_at) = (m.find("= xcl_lds[").unwrap(), m.find(AFTER_READ).unwrap(), m.find("] = v0;").unwrap());
    assert!(m.find(READ).unwrap() < wait && read_at < wait && wait < write_at, "the meet between the read and the write:\n{m}");
    // an atomic waits for both and sets both
    let m = main(&hlsl(&[read, write, add, read], &split));
    assert_eq!([AFTER_WRITE, AFTER_READ, AFTER_ANY, WROTE, READ, BOTH].map(|x| m.matches(x).count()), [2, 1, 1, 1, 2, 1], "{m}");
    // s_barrier: both halves made their accesses before it
    let m = main(&hlsl(&[write, barrier, read], &split));
    let (sync, clear, wait) = (m.find("GroupMemoryBarrierWithGroupSync();").unwrap(), m.find(CLEAR).unwrap(), m.find(AFTER_WRITE).unwrap());
    assert!(sync < clear && clear < wait, "{m}");
    // no LDS write: no bits at all, LDS or not; the native form has none either
    for (what, list, o) in [
        ("reads only", &[read, read][..], split.clone()),
        ("no LDS", &[lane][..], split.clone()),
        ("native wave64", &[write, read, add][..], Options { split_wave64: false, ..split.clone() }),
    ] {
        let m = main(&hlsl(list, &o));
        assert!(!m.contains("xcl_x_lw"), "{what}:\n{m}");
    }
    // a compute group of several console waves (the counters' meet) and one of a single half
    // (xsemu: the halves form of compute is the legacy wave64 form; the packed form runs both
    // halves in one hardware wave, in order, and needs no meet)
    for threads in [128u32, 32] {
        let cs = super::CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
        let h = hlsl(&[write, read], &Options { mode: Mode::Compute(cs.clone()), legacy_compute_wave64: true, ..split.clone() });
        assert!(h.contains(AFTER_WRITE) && h.contains(WROTE), "{threads} threads");
        let packed = hlsl(&[write, read], &Options { mode: Mode::Compute(cs), legacy_compute_wave64: false, ..split.clone() });
        assert!(!packed.contains("xcl_x_lw"), "{threads} threads, packed:\n{packed}");
    }
    // `XCL_X_LDS_ORDER=0` turns it off (the earlier form)
    use super::emit::lds_order_for;
    assert!(lds_order_for(None) && lds_order_for(Some("1")) && lds_order_for(Some("")));
    assert!(!lds_order_for(Some("0")) && !lds_order_for(Some(" 0 ")));
}

/// `XCL_X_LDS_ORDER=0` gives the split form's text from before the LDS order exactly: no
/// `xcl_x_lw` anywhere, and the ordered text is that one plus its `xcl_x_lw` lines alone.
#[test]
fn split_wave64_lds_order_off_is_the_earlier_text() {
    type Inst = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);
    let write: Inst = ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None);
    let read: Inst = ("DS_READ_B32", "ENC_DS", &[("VDST", 2), ("ADDR", 1)], None);
    let add: Inst = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None);
    let barrier: Inst = ("S_BARRIER", "ENC_SOPP", &[], None);
    let code = prog(&[write, read, add, barrier, read, write]);
    let kernel = Options { mode: Mode::Kernel, wave64: true, split_wave64: true, ..Default::default() };
    let cs = |threads: u32| Options {
        mode: Mode::Compute(super::CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 }),
        legacy_compute_wave64: true,
        ..kernel.clone()
    };
    for (what, o) in [("kernel", kernel.clone()), ("compute 64", cs(64)), ("compute 128", cs(128))] {
        let on = translate(&code, &o).unwrap().hlsl;
        super::emit::NO_LDS_ORDER.set(true);
        let off = translate(&code, &o).unwrap().hlsl;
        super::emit::NO_LDS_ORDER.set(false);
        assert!(on.contains("xcl_x_lw") && !off.contains("xcl_x_lw"), "{what}:\n{off}");
        let stripped: Vec<&str> = on.lines().filter(|l| !l.contains("xcl_x_lw")).collect();
        assert_eq!(stripped, off.lines().collect::<Vec<_>>(), "{what}");
    }
}

/// Mesh mode for a GPU with 32-lane waves only: the console's two wave64s per group as four
/// hardware wave32s exchanging masks and lane reads (counter-paired: two console waves),
/// programs with cross-half operations included; native wave64 and wave32 programs unchanged.
#[test]
fn mesh_split_wave64_form() {
    let ms = super::MsInterface { threads: 128, max_vertices: 64, max_primitives: 64, outputs: vec![] };
    let lane = prog(&[("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)], None)]);
    let branch = prog(&[("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)], None)]);
    for code in [&lane, &branch] {
        let native = super::translate_mesh(code, ms.clone(), true, false, 1).unwrap();
        assert!(native.hlsl.contains("[WaveSize(64)]") && native.hlsl.contains("[numthreads(128, 1, 1)]") && !native.hlsl.contains("xcl_x_"));
        let split = super::translate_mesh(code, ms.clone(), true, true, 1).unwrap();
        assert!(split.complete(), "{:?}", split.unsupported_why);
        assert!(split.hlsl.contains("[WaveSize(32)]") && split.hlsl.contains("[numthreads(128, 1, 1)]"));
        assert!(split.hlsl.contains("uint xcl_tid = xcl_x_thread(gi);") && split.hlsl.contains("#define XCL_X_SPIN 1"));
        assert!(split.hlsl.contains("xcl_x_ballot(true)"));
        // a wave32 program is not split
        let w32 = super::translate_mesh(code, ms.clone(), false, true, 1).unwrap();
        assert!(w32.hlsl.contains("[WaveSize(32)]") && !w32.hlsl.contains("xcl_x_"));
        if super::dxc::location().is_ok() {
            super::dxc::compile(&split.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", split.hlsl));
        }
    }
    assert!(super::translate_mesh(&lane, ms, true, true, 1).unwrap().hlsl.contains("xcl_x_rla_uniform("));
}

/// Operations whose result spans both 32-lane halves of a wave64 – what the split form
/// exchanges: masks with bits in either half (counted, scanned, branched on), readlane /
/// readfirstlane of a lane in the other half, ds_bpermute across the halves, a GLC scalar
/// load. The same results natively (64-lane waves) and split (32-lane waves only); split,
/// the runner also checks both halves end with the same SGPRs.
#[test]
fn gpu_split_wave64_cross_half_operations() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)), // vcc = lanes 0..39
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 1), ("SRC1", 2)], None), // lane s2 = 37
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 106), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 107), ("SRC1", V + 5)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 128)], None), // exec = lanes 48..63
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0xFFFF_0000)),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)], None),
        ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 126)], None),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), // not taken: only the upper half is active
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 3)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 4), ("DATA0", 1)], None), // lane 63 - l
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(50)), // vcc = lane 50 only
        ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), // not taken
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 126), ("SSRC1", 106)], None), // scc = vcc != 0
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 7), ("SSRC0", 129), ("SSRC1", 130)], None),
        // s[8:9] = 0: the test memory
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[2] = 37;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 1000 + l as u32;
        r[2] = 0x77;
        r[4] = (63 - l as u32) * 4;
    }
    let mem = [0xABCDu32, 0x1234, 0x5678, 0x9ABC];
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.sgpr[0], 40, "s_bcnt1 of a mask over both halves");
    assert_eq!(r.sgpr[1], 1037, "readlane of lane 37 (upper half)");
    assert_eq!(r.sgpr[3], 1048, "readfirstlane with only the upper half active");
    assert_eq!(r.sgpr[4], 48, "s_ff1 of exec");
    assert_eq!(r.sgpr[5], 1, "branch on a vcc with a bit in the upper half only");
    assert_eq!(r.sgpr[7], 1, "scc from a 64-bit mask operation");
    assert_eq!((r.sgpr[10], r.sgpr[11]), (0, 1 << 18), "s_and_b64 exec, vcc");
    assert_eq!(&r.sgpr[12..16], &mem, "s_load_dwordx4 glc");
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(v[5], (l as u32).min(40), "lane {l}: v_mbcnt over vcc");
        assert_eq!(v[2], if l >= 48 { 1048 } else { 0x77 }, "lane {l}: write under exec of the upper half");
        assert_eq!(v[6], 1000 + 63 - l as u32, "lane {l}: ds_bpermute from lane {}", 63 - l);
    }
}

// xsemu: from xcl-dev/xcl nvidia-dev 9b455a1a,67a08dbe crates/xcl/src/gpu/recomp/tests.rs
/// LDS written by the upper half and read by lane 0 with nothing between that makes the halves
/// meet: the console's wave runs both for all 64 lanes in order, so lane 0 sees the writes. The
/// upper half writes only after a chain of dependent global loads its lanes alone make, so in
/// the split form (32-lane GPUs) the lower half reaches the read first unless the read waits
/// for it (`xcl_x_lw`). A plain store and an atomic or; then the other order: every lane reads
/// (the upper half late) and lane 0 overwrites what they read, which the lower half must not do
/// before the upper half has read. Groups of one console wave (the group barrier's meet) and
/// of several (the counters'). xsemu: compute in the halves form is the legacy wave64 form
/// (the packed form runs both halves in one hardware wave), and HLSL is the only translator.
#[test]
fn gpu_split_wave64_lds_write_then_read_across_halves() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    const LOADS: usize = 16;
    const MARKER: u32 = 0x600D;
    // s0 = output base, s1 = the load chain's first address, s2 = MARKER - 1, s3 =
    // log2(threads), s4 = MARKER, s5 = group id; v0 = thread index in the group
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 5), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 6), ("SRC0", 128)], None),
        ("V_AND_B32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 128 + 31), ("VSRC1", 0)], None), // lane & 31
        ("V_LSHRREV_B32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 134), ("VSRC1", 0)], None), // console wave
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 130), ("VSRC1", 1)], None), // its LDS dword
        // lane 0 clears the wave's two dwords (at v1 and v1 + 256)
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 127), ("SSRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 3), ("SRC0", 128)], None),
        ("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 1), ("DATA0", 3), ("DATA1", 3), ("OFFSET0", 0), ("OFFSET1", 64)], None),
        // a mask read whole: the halves meet (after the clear, before the writes below)
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 255), ("VSRC1", 0)], Some(40)),
        ("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 20), ("SSRC0", 106)], None),
        // lanes 32..63: the load chain, then lds[v1] = MARKER, lds[v1 + 256] |= 1 << (lane - 32)
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 127), ("SSRC0", 193)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 10), ("SRC0", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", 128)], None),
    ];
    for _ in 0..LOADS {
        list.push(("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 10), ("ADDR", 10), ("SADDR", 0x7D)], None));
    }
    list.extend([
        ("DS_WRITE_B32", "ENC_DS", vec![("ADDR", 1), ("DATA0", 10)], None),
        ("V_SUBREV_NC_U32", "ENC_VOP2", vec![("VDST", 12), ("SRC0", 2), ("VSRC1", 10)], None), // v10 - s2 = 1
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 12), ("SRC0", V + 2), ("VSRC1", 12)], None),
        ("DS_OR_B32", "ENC_DS", vec![("ADDR", 1), ("DATA0", 12), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        // lane 0 reads both, no exchange since the writes
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 127), ("SSRC0", 128)], None),
        ("DS_READ2_B32", "ENC_DS", vec![("VDST", 5), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 64)], None),
        // read, then write: every lane reads lds[v1] (MARKER), the upper half only after a
        // second load chain (its address depends on it); then lane 0 clears it with nothing
        // between – the console's lanes all read before the clear
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", 4)], None), // MARKER
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 127), ("SSRC0", 193)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", 128)], None),
    ]);
    for _ in 0..LOADS {
        list.push(("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 14), ("ADDR", 14), ("SADDR", 0x7D)], None));
    }
    list.extend([
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None),
        ("V_SUBREV_NC_U32", "ENC_VOP2", vec![("VDST", 15), ("SRC0", 4), ("VSRC1", 14)], None), // 0
        ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 15), ("SRC0", V + 1), ("VSRC1", 15)], None), // v1
        ("DS_READ_B32", "ENC_DS", vec![("VDST", 7), ("ADDR", 15)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 127), ("SSRC0", 128)], None),
        ("DS_WRITE_B32", "ENC_DS", vec![("ADDR", 1), ("DATA0", 3)], None),
        // out[group · threads + thread] = (v5, v6, v7, 0)
        ("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 128)], None),
        ("S_LSHL_B32", "ENC_SOP2", vec![("SDST", 16), ("SSRC0", 5), ("SSRC1", 3)], None), // s3 = log2(threads)
        ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 20), ("SRC0", 16), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 20), ("SRC0", 132), ("VSRC1", 20)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 20), ("SRC0", 0), ("VSRC1", 20)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", vec![("ADDR", 20), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(n, e, f, l)| (*n, *e, &f[..], *l)).collect();
    let p = prog(&refs);
    let chain = 0x40u32;
    let base = 0x1000usize;
    let groups = 64usize;
    for threads in [64u32, 128, 256] {
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 5, tgid: [true, false, false], tg_size: false, tid_components: 1 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64: true, legacy_compute_wave64: true, ..Default::default() });
        let t = translate(&p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if runner::split_wave64() {
            // every LDS instruction tests the bits (2 writes, the or, 2 reads, the clear)
            assert_eq!(t.hlsl.matches(") != 0u) xcl_x_meet();").count(), 6, "{threads} threads");
        }
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let mut io = vec![0u32; 128];
        io[..5].copy_from_slice(&[base as u32, chain, MARKER - 1, threads.trailing_zeros(), MARKER]);
        let mut mem = vec![0u32; (base + groups * threads as usize * 16) / 4];
        // mem[chain + 4k] = chain + 4 (k + 1), the last one MARKER
        for k in 0..LOADS {
            mem[chain as usize / 4 + k] = if k + 1 == LOADS { MARKER } else { chain + 4 * (k as u32 + 1) };
        }
        let out = runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &mem).unwrap();
        for g in 0..groups {
            for tid in 0..threads as usize {
                let at = (base + (g * threads as usize + tid) * 16) / 4;
                let want = if tid % 64 == 0 { (MARKER, u32::MAX) } else { (0, 0) };
                assert_eq!((out[at], out[at + 1]), want, "{threads} threads, group {g} thread {tid}: the upper half's store and or");
                assert_eq!(out[at + 2], MARKER, "{threads} threads, group {g} thread {tid}: read before lane 0's clear");
            }
        }
    }
}

/// Masks only read lane by lane (`x64_masks`): a v_cndmask selector and a 64-bit carry chain,
/// both with bits in the two halves, need no exchange in the split form – and give the same
/// lanes as a whole-wave mask.
#[test]
fn gpu_split_wave64_lane_local_masks() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)), // vcc = lanes 0..39
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)], None),
        // s[20:21] = lanes 0..49, as a VOP3 selector
        ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 255)], Some(50)),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", 20)], None),
        // v[4:5] + v[6:7] as 64-bit numbers: the carry of the low words feeds the high words
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 4), ("SDST", 106), ("SRC0", V + 4), ("SRC1", V + 6)], None),
        ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 7)], None),
        // the masks end dead (the kernel runner reads every SGPR back)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)], None),
    ]);
    if runner::split_wave64() {
        let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
        assert_eq!(t.hlsl.matches("= xcl_x_ballot_local(and(").count(), 4, "every mask stays in its half");
        assert_eq!(t.hlsl.matches("= xcl_x_ballot(and(").count(), 0);
    }
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u32;
        r[0] = l;
        r[2] = 100 + l;
        r[3] = 200 + l;
        // a carry out of the low word in every third lane
        r[4] = if l.is_multiple_of(3) { 0xFFFF_FFF0 } else { l };
        r[5] = 1000 + l;
        r[6] = 0x20;
        r[7] = 7;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64u32 {
        let v = &r.vgpr[l as usize];
        assert_eq!(v[1], if l < 40 { 200 + l } else { 100 + l }, "lane {l}: v_cndmask on vcc");
        assert_eq!(v[8], if l < 50 { 200 + l } else { 100 + l }, "lane {l}: v_cndmask on s[20:21]");
        let a = ((1000 + l as u64) << 32) | if l % 3 == 0 { 0xFFFF_FFF0 } else { l as u64 };
        let sum = a + ((7u64 << 32) | 0x20);
        assert_eq!((v[4], v[5]), (sum as u32, (sum >> 32) as u32), "lane {l}: 64-bit add with carry");
    }
}

/// Masks that are read whole after a write `x64_masks` must not take for an overwrite (an
/// SDWA compare into its SDST, `s_bitset1`, `s_movreld`) or through a read the operand
/// tables do not show (`s_movrels`, an SMEM register offset, an SDWA scalar src1): each is
/// still exchanged, and both halves end with the same SGPRs (the runner checks) and the
/// whole-wave values. A mask that goes nowhere is not exchanged.
#[test]
fn gpu_split_wave64_masks_read_whole_after_partial_writes() {
    if !gpu_ready() {
        return;
    }
    type Inst = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);
    // vcc / s[20:21] = lanes 0..39, or lane 2 only (low word 4)
    let cmp_vcc: Inst = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40));
    let cmp_s20: Inst = ("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 255), ("SRC1", V)], Some(40));
    let lane2_s20: Inst = ("V_CMP_EQ_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 130)], None);
    let s20_dead: Inst = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)], None);
    let cases: Vec<(&str, Vec<Inst>)> = vec![
        (
            "SDWA compare into s[22:23]",
            vec![
                cmp_vcc,
                ("V_CMP_EQ_U32", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)], None),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
            ],
        ),
        ("s_bitset1_b64", vec![cmp_s20, ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 191)], None), ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)], None)]),
        (
            "s_movrels_b64 from VCC",
            vec![
                cmp_vcc,
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 134)], None), // m0 = 6
                ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 100)], None), // s[0:1] = s[106:107]
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)], None),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)], None),
            ],
        ),
        (
            "s_movreld_b64 past the mask",
            vec![
                cmp_s20,
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 132)], None), // m0 = 4
                ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 2)], None), // s[24:25] = s[2:3]
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)], None),
            ],
        ),
        ("SMEM register offset", vec![lane2_s20, ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 0), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 20)], None), s20_dead]),
        (
            "SDWA scalar src1",
            vec![
                lane2_s20,
                ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 9), ("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 20), ("S1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
                s20_dead,
            ],
        ),
    ];
    let mem = [11u32, 22, 33, 44];
    for (what, list) in cases {
        let code = prog(&list);
        if runner::split_wave64() {
            let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
            assert!(t.hlsl.contains("= xcl_x_ballot(and("), "{what}: the mask is exchanged");
        }
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            r[0] = l as u32;
        }
        st.sgpr[2] = 0x1234;
        let r = runner::run_code(&code, &st, &mem).unwrap_or_else(|e| panic!("{what}: {e}"));
        match what {
            "SDWA compare into s[22:23]" => {
                assert_eq!(r.sgpr[0], 40, "{what}");
                assert_eq!((r.sgpr[22], r.sgpr[23]), (!0, !0), "{what}");
            }
            "s_bitset1_b64" => assert_eq!(r.sgpr[0], 41, "{what}"),
            "s_movrels_b64 from VCC" => assert_eq!(r.sgpr[0], 40, "{what}"),
            "s_movreld_b64 past the mask" => {
                assert_eq!(r.sgpr[0], 40, "{what}");
                assert_eq!(r.sgpr[24], 0x1234, "{what}");
            }
            "SMEM register offset" => assert_eq!(r.sgpr[0], mem[1], "{what}"),
            _ => {
                for l in 0..64u32 {
                    assert_eq!(r.vgpr[l as usize][9], l + 4, "{what}: lane {l}");
                }
            }
        }
    }
    if runner::split_wave64() {
        // v_cmp into NULL: nothing to exchange
        let code = prog(&[("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 125), ("SRC0", V), ("SRC1", V + 1)], None)]);
        let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
        assert_eq!(t.hlsl.matches("= xcl_x_ballot(and(").count(), 0);
        assert_eq!(t.hlsl.matches("= xcl_x_ballot_local(and(").count(), 1);
    }
}

/// Four wave64s per group whose waves loop a different number of times (the counter-paired
/// form on 32-lane GPUs): each iteration a mask read lane by lane (`v_cndmask`) that only the
/// odd waves also read whole (`s_bcnt1`, on their own path), and a 64-bit carry chain (not
/// exchanged). The exchanged and the local masks interleave without the halves losing step.
#[test]
fn gpu_split_wave64_local_and_exchanged_masks_in_diverging_waves() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let p = prog(&[
        // v0 = thread index in the group, v1 = wave, v2 = lane (as gpu_split_wave64_waves_diverge)
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", 1), ("SRC2", V)], None),
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 2), ("SRC1", 2), ("SRC2", V + 20)], None),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 134), ("VSRC1", 0)], None),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 0)], Some(63)),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 1)], None),
        ("S_AND_B32", "ENC_SOP2", &[("SDST", 17), ("SSRC0", 10), ("SSRC1", 129)], None), // odd wave
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None), // n = 2 (w + 1)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None), // sum (odd waves)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None), // i
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)], None), // per-lane count
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 2)], Some(0x9E37_79B9)),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None),
        // loop: k = 5 i + 3
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 12), ("SSRC1", 133)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 13), ("SSRC1", 131)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 13), ("VSRC1", 2)], None), // vcc = lanes < k
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)], None), // count += lane < k
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 17), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)], None), // even waves: skip
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 106)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 14)], None),
        // v[8:9] += 0x9E3779B9 + lane, through s[20:21]
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 8), ("SDST", 20), ("SRC0", V + 8), ("SRC1", V + 6)], None),
        ("V_ADD_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 9), ("SDST", 20), ("SRC0", V + 9), ("SRC1", 128), ("SRC2", 20)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF0)], None), // → loop
        // out[group · 256 + thread] = (count, sum, v8, v9)
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", 11)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", V + 8)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 9)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 4), ("SSRC1", 136)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 16), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 132), ("VSRC1", 30)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 0), ("VSRC1", 30)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 31), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", 24), ("SADDR", 0x7D)], None),
    ]);
    for dims in [[256u32, 1, 1], [8, 8, 4]] {
        let threads = dims.iter().product::<u32>();
        let cs = CsInterface { threads: dims, user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 3 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() });
        let t = translate(&p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if runner::split_wave64() {
            let legacy = translate(&p, &Options { legacy_compute_wave64: true, ..opt }).unwrap();
            assert_eq!(legacy.hlsl.matches("= xcl_x_ballot(and(").count(), 1, "the v_cmp: read whole by the odd waves");
            assert_eq!(legacy.hlsl.matches("= xcl_x_ballot_local(and(").count(), 2, "the carries");
        }
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let base = 0x100usize;
        let mut io = vec![0u32; 128];
        io[0] = base as u32;
        io[1] = dims[0];
        io[2] = dims[0] * dims[1];
        let groups = 3usize;
        let out = runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &vec![0u32; (base + groups * 256 * 16) / 4]).unwrap();
        for g in 0..groups {
            for tid in 0..threads as usize {
                let (w, l) = (tid / 64, (tid % 64) as u32);
                let n = 2 * (w as u32 + 1);
                let count = (0..n).filter(|i| 5 * i + 3 > l).count() as u32;
                let sum = if w % 2 == 1 { (0..n).map(|i| (5 * i + 3).min(64)).sum() } else { 0 };
                let acc = n as u64 * (0x9E37_79B9u64 + l as u64);
                let at = (base + (g * 256 + tid) * 16) / 4;
                let what = format!("{dims:?}, group {g} thread {tid} (wave {w}, lane {l})");
                assert_eq!(out[at], count, "{what}: per-lane count");
                assert_eq!(out[at + 1], sum, "{what}: s_bcnt1 sum");
                assert_eq!((out[at + 2], out[at + 3]), (acc as u32, (acc >> 32) as u32), "{what}: 64-bit carry chain");
            }
        }
    }
}

/// A group of four wave64s whose waves take different paths: each loops 2 · (wave + 1)
/// times through a ballot, s_bcnt1 and a readlane, then exchanges lanes across its halves
/// with ds_bpermute. Split (32-lane GPUs) this is the counter-paired form – a group barrier
/// per exchange would deadlock or mix the waves up. 3-D groups too: NVIDIA does not form its
/// waves from consecutive thread indices there (8 × 8 × 4), the console does.
#[test]
fn gpu_split_wave64_waves_diverge() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let p = prog(&[
        // v0 = thread index in the group: v0 + v1 · s1 + v2 · s2 (s1 = x, s2 = x · y)
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", 1), ("SRC2", V)], None),
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 2), ("SRC1", 2), ("SRC2", V + 20)], None),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 134), ("VSRC1", 0)], None), // wave
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 0)], Some(63)), // lane
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None), // n = 2 (w + 1)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None), // sum
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None), // i
        // loop: k = 5 i + 3; sum += popcount(lanes < k) + v0 of lane k
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 12), ("SSRC1", 133)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 13), ("SSRC1", 131)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 13), ("VSRC1", 2)], None),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 106)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 14)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V), ("SRC1", 13)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 15)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF5)], None), // → loop
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 2)], Some(33)),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 3), ("DATA0", 0)], None), // v0 of lane ^ 33
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 11)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 4), ("SSRC1", 136)], None), // group · 256
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 16), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 131), ("VSRC1", 7)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    for dims in [[256u32, 1, 1], [64, 1, 1], [8, 8, 4], [16, 4, 1]] {
        let threads = dims.iter().product::<u32>();
        let cs = CsInterface { threads: dims, user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 3 };
        let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() })).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if runner::split_wave64() && super::legacy_compute_wave64() {
            let spin = threads > 64 || std::env::var("XCL_WAVE64_SPIN").is_ok_and(|v| v.trim() != "0");
            assert!(t.hlsl.contains("[WaveSize(32)]") && t.hlsl.contains(&format!("#define XCL_X_SPIN {}\n#define XCL_X_PAIR 1", spin as u32)), "{dims:?}");
        } else if runner::split_wave64() {
            assert!(t.hlsl.contains("[WaveSize(32)]") && t.hlsl.contains(&format!("[numthreads({}, 1, 1)]", threads.div_ceil(64) * 32)), "{dims:?}");
            assert!(!t.hlsl.contains("xcl_x_meet("), "{dims:?}");
        }
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let base = 0x100usize;
        let mut io = vec![0u32; 128];
        io[0] = base as u32;
        io[1] = dims[0];
        io[2] = dims[0] * dims[1];
        let groups = 3usize;
        let out = runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &vec![0u32; (base + groups * 256 * 8) / 4]).unwrap();
        for g in 0..groups {
            for tid in 0..threads as usize {
                let (w, l) = (tid / 64, tid % 64);
                let want: u32 = (0..2 * (w + 1)).map(|i| (5 * i + 3).min(64) as u32 + (w * 64 + 5 * i + 3) as u32).sum();
                let at = (base + (g * 256 + tid) * 8) / 4;
                assert_eq!(out[at], want, "{dims:?}, group {g} thread {tid}: loop of wave {w}");
                assert_eq!(out[at + 1], (w * 64 + (l ^ 33)) as u32, "{dims:?}, group {g} thread {tid}: ds_bpermute");
            }
        }
    }
}

/// What the split wave64 form costs (`cargo test --release -- --ignored bench_split_wave64
/// --nocapture`): a loop of a few VALU instructions and one mask (`v_cmp`, an exchange in the
/// split form), plus a readfirstlane (another exchange) in the second variant, as a wave32
/// program (no exchanges) and as a wave64 program – natively on a GPU with 64-lane waves,
/// split on one with 32-lane waves only. GPU time of the dispatch (timestamps), median of 9.
#[test]
#[ignore]
fn bench_split_wave64() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let body = |readfirstlane: bool| {
        let mut l: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 10), ("SSRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", V)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", 128)], None),
            // loop: an LCG step, a mask from it, a conditional count
            ("V_MUL_LO_U32", "ENC_VOP3", vec![("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(1_103_515_245)),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(12345)),
            ("V_LSHRREV_B32", "ENC_VOP2", vec![("VDST", 3), ("SRC0", 144), ("VSRC1", 2)], None),
            ("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 255), ("VSRC1", 3)], Some(0x4000)),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 5), ("SRC0", 129), ("VSRC1", 4)], None),
            ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)], None),
        ];
        let mut back = 0xFFF3u64; // → the loop start: 48 bytes of loop body before the branch
        if readfirstlane {
            l.push(("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", V + 3)], None));
            l.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 11), ("VSRC1", 4)], None));
            back -= 2;
        }
        l.extend([
            ("S_ADD_U32", "ENC_SOP2", vec![("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", vec![("SSRC0", 10), ("SSRC1", 1)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", vec![("SIMM16", back)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 128)], None),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", vec![("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ]);
        let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = l.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
        prog(&refs)
    };
    let iterations = 32768u32;
    println!("GPU: {} ({} lanes per wave at most)", runner::adapter().unwrap(), if runner::wave64_supported() == Ok(true) { 64 } else { 32 });
    for readfirstlane in [false, true] {
        let p = body(readfirstlane);
        for threads in [64u32, 256] {
            let mut ms = Vec::new();
            let mut exchanges = 0;
            for wave64 in [false, true] {
                let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
                let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), wave64, ..Default::default() })).unwrap();
                // the loop's exchanges: masks read whole (`x64_masks`: not the v_cndmask
                // selector, unless `XCL_X64_ALL_MASKS`) and the readfirstlane
                exchanges = t.hlsl.matches("= xcl_x_ballot(and(").count() + t.hlsl.matches("xcl_x_rla_uniform(").count().saturating_sub(1);
                assert!(t.complete(), "{:?}", t.unsupported_why);
                let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
                let groups = (1u32 << 18) / threads;
                let mut io = vec![0u32; 128];
                io[0] = 0;
                io[1] = iterations;
                let mem = vec![0u32; threads as usize];
                // GPU time of each dispatch (timestamps: no pipeline creation, uploads or CPU
                // wait in it), the median of 9
                let mut t = runner::dispatch_timed(&dxil, [groups, 1, 1], &io, &mem, 9).unwrap();
                t.sort_by(f64::total_cmp);
                ms.push(t[t.len() / 2]);
            }
            let form = if runner::split_wave64() { if threads > 64 { "split, counter-paired" } else { "split, barrier" } } else { "native" };
            println!(
                "v_cmp + v_cndmask{}: {exchanges} exchange(s) / iteration, {threads} threads per group, 2^18 threads x {iterations} iterations: wave32 {:.1} ms, wave64 ({form}) {:.1} ms = {:.2}x",
                if readfirstlane { " + v_readfirstlane" } else { "" },
                ms[0],
                ms[1],
                ms[1] / ms[0]
            );
        }
    }
}

/// TG_SIZE (the SGPR after the group ids): the group's wave count | the thread's wave index
/// << 6. A group that is not a multiple of the wave size has a last, partial wave: 96 threads
/// are 2 console waves of 64 (3 of 32), not 96 / 64 = 1.
#[test]
fn gpu_compute_tg_size_counts_a_partial_last_wave() {
    if !gpu_ready() {
        return;
    }
    // global_store_dword [v0 * 4], s2 (TG_SIZE: user SGPRs s0, s1, no group ids)
    let p = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 2)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ]);
    for (threads, wave64) in [(96u32, true), (96, false), (64, true), (40, false), (128, true)] {
        let lanes = if wave64 { 64 } else { 32 };
        let cs = super::CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: true, tid_components: 1 };
        let t = translate(&p, &on_gpu(Options { mode: Mode::Compute(cs), wave64, ..Default::default() })).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let mem = runner::dispatch(&dxil, [1, 1, 1], &[0u32; 128], &vec![0xFFFF_FFFFu32; threads as usize]).unwrap();
        let waves = threads.div_ceil(lanes);
        for (k, &got) in mem.iter().take(threads as usize).enumerate() {
            assert_eq!(got, waves | (k as u32 / lanes) << 6, "{threads} threads, wave{lanes}: thread {k}");
        }
    }
}

/// v_fract_f32 at the edges, as RDNA computes it (x - floor(x) kept below 1.0): ±inf gives
/// NaN and a NaN stays NaN – a `min()` clamp returned 0.99999994 there on NVIDIA (HLSL min
/// returns its non-NaN operand); a tiny negative value rounds to the largest float below 1.
#[test]
fn gpu_fract_f32_edges() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[("V_FRACT_F32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)], None)]);
    let inputs = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0, 1.5, -0.25, -(2f32.powi(-25)), 7.0];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = inputs[l % inputs.len()].to_bits();
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let (x, got) = (inputs[l % inputs.len()], f32::from_bits(r.vgpr[l][2]));
        if x.is_infinite() || x.is_nan() {
            assert!(got.is_nan(), "lane {l}: fract({x}) = {got}, want NaN");
        } else if x == -(2f32.powi(-25)) {
            assert_eq!(r.vgpr[l][2], 0x3F7F_FFFF, "lane {l}: fract({x}) below 1.0");
        } else {
            assert_eq!(got.to_bits(), (x - x.floor()).to_bits(), "lane {l}: fract({x})");
        }
    }
}

#[test]
fn gpu_trig_reduces_turns_before_scaling() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_SIN_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 1)], None),
        ("V_COS_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 1)], None),
        ("V_SIN_F16", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 2)], None),
        ("V_COS_F16", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 2)], None),
    ]);
    let cases: [(f32, u32); 6] = [
        (0.5, 0x3800),
        (-1.5, 0xBE00),
        (-0.0, 0x8000),
        (1_048_576.25, 0x5C01),
        (0.125, 0x3000),
        (f32::INFINITY, 0x7C00),
    ];
    let mut st = KernelState::default();
    for (lane, &(x, half)) in cases.iter().enumerate() {
        st.vgpr[lane][1] = x.to_bits();
        st.vgpr[lane][2] = half;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for lane in 0..2 {
        assert_eq!(r.vgpr[lane][4], 0);
        assert_eq!(r.vgpr[lane][6] as u16, 0);
    }
    assert_eq!(r.vgpr[2][4], (-0.0f32).to_bits());
    assert_eq!(r.vgpr[2][6] as u16, 0x8000);
    assert!((f32::from_bits(r.vgpr[3][4]) - 1.0).abs() < 1e-6);
    assert_eq!(r.vgpr[3][5], 0);
    assert_eq!(r.vgpr[3][6] as u16, 0x3C00);
    assert_eq!(r.vgpr[3][7] as u16, 0);
    assert!((f32::from_bits(r.vgpr[4][4]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!(f32::from_bits(r.vgpr[5][4]).is_nan());
    assert!(f32::from_bits(r.vgpr[5][5]).is_nan());
}

#[test]
fn gpu_half_fract_and_float_median_edges() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_FRACT_F16", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 7)], None),
        ("V_MED3_F16", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)], None),
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", V + 5), ("SRC2", V + 6)], None),
    ]);
    let mut st = KernelState::default();
    for (i, (half, single)) in [(0x4000, 2.0f32), (0x3C00, 1.0), (0x4200, 3.0)].into_iter().enumerate() {
        for lane in 0..4 {
            st.vgpr[lane][1 + i] = if lane == i { 0x7E00 } else { half };
            st.vgpr[lane][4 + i] = if lane == i { f32::NAN } else { single }.to_bits();
        }
    }
    st.vgpr[0][7] = 0x8400;
    st.vgpr[1][7] = 0xB400;
    st.vgpr[2][7] = 0x7C00;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!(r.vgpr[0][8] as u16, 0x3BFF);
    assert_eq!(r.vgpr[1][8] as u16, 0x3A00);
    assert_eq!(r.vgpr[2][8] & 0x7C00, 0x7C00);
    assert_ne!(r.vgpr[2][8] & 0x03FF, 0);
    for (lane, (half, single)) in [(0x3C00, 1.0f32), (0x4000, 2.0), (0x3C00, 1.0), (0x4000, 2.0)].into_iter().enumerate() {
        assert_eq!(r.vgpr[lane][9] as u16, half);
        assert_eq!(r.vgpr[lane][10], single.to_bits());
    }
}

/// f64 compares with NaN operands, both the ordered ones and the negated ones that are true
/// when unordered, bit for bit against IEEE.
#[test]
fn gpu_f64_compares_with_nan() {
    for wave64 in gpu_waves() {
        f64_compares_with_nan(wave64);
    }
}

fn f64_compares_with_nan(wave64: bool) {
    const OPS: [&str; 8] = ["O", "U", "NEQ", "NLT", "NGE", "EQ", "LT", "LG"];
    let insts: Vec<(String, u64)> = OPS.iter().enumerate().map(|(i, op)| (format!("V_CMP_{op}_F64"), 20 + 2 * i as u64)).collect();
    let fields: Vec<[(&str, u64); 3]> = insts.iter().map(|(_, d)| [("VDST", *d), ("SRC0", V), ("SRC1", V + 2)]).collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> =
        insts.iter().zip(&fields).map(|((n, _), f)| (n.as_str(), "ENC_VOP3", &f[..], None)).collect();
    let p = prog(&list);
    let nan = f64::from_bits(0x7FF8_0000_0000_1234);
    let pairs = [(1.0, 1.0), (nan, 1.0), (2.0, nan), (-0.0, 0.0), (f64::INFINITY, f64::INFINITY), (nan, nan), (1.0, 2.0)];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (a, b): (f64, f64) = pairs[l % pairs.len()];
        (r[0], r[1], r[2], r[3]) = (a.to_bits() as u32, (a.to_bits() >> 32) as u32, b.to_bits() as u32, (b.to_bits() >> 32) as u32);
    }
    let (r, lanes) = run_wave(&p, &st, wave64);
    let mut bad = Vec::new();
    for (i, op) in OPS.iter().enumerate() {
        let mask = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
        for l in 0..lanes {
            let (a, b) = pairs[l % pairs.len()];
            let want = match *op {
                "O" => !a.is_nan() && !b.is_nan(),
                "U" => a.is_nan() || b.is_nan(),
                "NEQ" => !(a == b),
                "NLT" => !(a < b),
                "NGE" => !(a >= b),
                "EQ" => a == b,
                "LT" => a < b,
                _ => a < b || a > b,
            };
            if (mask >> l & 1 == 1) != want {
                bad.push(format!("v_cmp_{}_f64({a:e}, {b:e}) = {}, want {want}", op.to_lowercase(), !want));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// Every f32 compare with NaN operands against IEEE. Before the explicit NaN tests, lt, ge
/// and lg came back true and nlg false for a NaN operand on an AMD RX 7900 XTX. The driver
/// gave lt the result of the nge before it, as `float_compare` describes. Without the NaN
/// tests (`Options::nan_guards`) too where the adapter is not AMD's.
#[test]
fn gpu_f32_compares_with_nan() {
    for wave64 in gpu_waves() {
        f32_compares_with_nan(wave64, true);
        if !super::pso::nan_guards() {
            f32_compares_with_nan(wave64, false);
        }
    }
}

fn f32_compares_with_nan(wave64: bool, guards: bool) {
    const OPS: [&str; 12] = ["O", "U", "NEQ", "NLT", "NGE", "EQ", "LT", "LG", "LE", "GT", "GE", "NLG"];
    let insts: Vec<(String, u64)> = OPS.iter().enumerate().map(|(i, op)| (format!("V_CMP_{op}_F32"), 20 + 2 * i as u64)).collect();
    let fields: Vec<[(&str, u64); 3]> = insts.iter().map(|(_, d)| [("VDST", *d), ("SRC0", V), ("SRC1", V + 1)]).collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> =
        insts.iter().zip(&fields).map(|((n, _), f)| (n.as_str(), "ENC_VOP3", &f[..], None)).collect();
    let p = prog(&list);
    let nan = f32::from_bits(0x7FC0_1234);
    let pairs = [(1.0f32, 1.0f32), (nan, 1.0), (2.0, nan), (-0.0, 0.0), (f32::INFINITY, f32::INFINITY), (nan, nan), (1.0, 2.0)];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (a, b) = pairs[l % pairs.len()];
        (r[0], r[1]) = (a.to_bits(), b.to_bits());
    }
    let t = translate(&p, &Options { wave64, nan_guards: guards, ..Default::default() }).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let mem: Vec<u32> = p.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let (r, lanes) = (runner::run(&dxil, &st, &mem).unwrap(), if wave64 { 64 } else { 32 });
    let mut bad = Vec::new();
    for (i, op) in OPS.iter().enumerate() {
        let mask = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
        for l in 0..lanes {
            let (a, b) = pairs[l % pairs.len()];
            let want = match *op {
                "O" => !a.is_nan() && !b.is_nan(),
                "U" => a.is_nan() || b.is_nan(),
                "NEQ" => !(a == b),
                "NLT" => !(a < b),
                "NGE" => !(a >= b),
                "EQ" => a == b,
                "LT" => a < b,
                "LG" => a < b || a > b,
                "LE" => a <= b,
                "GT" => a > b,
                "GE" => a >= b,
                _ => !(a < b || a > b),
            };
            if (mask >> l & 1 == 1) != want {
                bad.push(format!("v_cmp_{}_f32({a:e}, {b:e}) = {}, want {want}", op.to_lowercase(), !want));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}, NaN tests {guards}:\n{}", bad.join("\n"));
}

/// xorshift64, the fixed-seed generator of the randomized GPU tests
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

/// A random f64 weighted towards the edges: specials, any bit pattern, values next to
/// integers up to 2^64, denormals, magnitudes around 2^±65, small values on 1/8 steps.
fn rand_f64(r: &mut Rng) -> f64 {
    const SPECIAL: [u64; 12] = [0, 1 << 63, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000,
        0x7FF0_0000_0000_0001, 0xFFF8_0000_0000_1234, 1, 0x000F_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000, 0x8000_0000_0000_0001];
    match r.below(6) {
        0 => f64::from_bits(SPECIAL[r.below(12) as usize]),
        1 => f64::from_bits(r.next()),                                        // any bits
        2 => { // near an integer of any size up to 2^64
            let k = (r.next() >> r.below(64)) as f64;
            let d = [0.0, 0.5, -0.5, 0.25, 1e-9, -1e-9, 0.9999999999][r.below(7) as usize];
            let v = k + d;
            if r.below(2) == 0 { -v } else { v }
        }
        3 => f64::from_bits(r.next() & 0x800F_FFFF_FFFF_FFFF),                 // denormal
        4 => { let e = 1023 + r.below(130) - 65; f64::from_bits((e << 52) | (r.next() & 0x800F_FFFF_FFFF_FFFF)) } // |x| around 2^±65
        _ => (r.below(2000) as f64 - 1000.0) / 8.0,                           // small values on 1/8 steps
    }
}

/// 2^e for a normal e
fn p2f(e: i32) -> f64 { f64::from_bits(((e + 1023) as u64) << 52) }

/// x · 2^n on the CPU with a single rounding, since each step but the last is exact. None when
/// that is not possible.
fn ldexp_ref(x: f64, n: i32) -> Option<f64> {
    if x.is_nan() || x.is_infinite() || x == 0.0 { return Some(x); }
    if n > 4000 { return Some(f64::INFINITY.copysign(x)); }
    if n < -4000 { return Some(x * 0.0); }
    if (-1022..=1023).contains(&n) { return Some(x * p2f(n)); }
    if n > 1023 && n <= 2046 { return Some((x * p2f(1023)) * p2f(n - 1023)); }
    if (-2044..-1022).contains(&n) && (1.0..2.0).contains(&x.abs()) { return Some((x * p2f(-1022)) * p2f(n + 1022)); }
    None
}

/// The f64 math against the CPU on random inputs, one wave per dispatch in 50 dispatches, or
/// n dispatches with XCL_FUZZ_ITERS=n, as wave64 and as wave32. floor, ceil, trunc, fract and the i32 and u32
/// conversions must match bit for bit, and so must ldexp wherever the CPU can compute it with
/// one rounding. sqrt must be within 1 ulp and rsq within 2.
#[test]
fn gpu_f64_math_random() {
    for wave64 in gpu_waves() {
        f64_math_random(wave64);
    }
}

fn f64_math_random(wave64: bool) {
    let iters: usize = std::env::var("XCL_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let p = prog(&[
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_TRUNC_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V)], None),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 12), ("SRC0", V)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 14), ("SRC0", V)], None),
        ("V_LDEXP_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V), ("SRC1", V + 20)], None),
        ("V_TRUNC_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V), ("NEG", 1)], None),
        ("V_FLOOR_F64", "ENC_VOP3", &[("VDST", 24), ("SRC0", V), ("ABS", 1)], None),
    ]);
    let (dxil, mem) = kernel_wave(&p, wave64);
    let lanes = if wave64 { 64 } else { 32 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let same = |g: f64, w: f64| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
    let ulps = |g: f64, w: f64| -> u64 {
        if same(g, w) { 0 } else if g.is_nan() || w.is_nan() || g.is_sign_negative() != w.is_sign_negative() { u64::MAX } else { g.to_bits().abs_diff(w.to_bits()) }
    };
    let (mut checked, mut ldexp_checked, mut sqrt_exact, mut sqrt_1ulp, mut rsq_hist) = (0usize, 0usize, 0usize, 0usize, [0usize; 4]);
    let mut bad: Vec<String> = Vec::new();
    for _ in 0..iters {
        let mut st = KernelState::default();
        let mut xs = Vec::new();
        for r in st.vgpr.iter_mut() {
            let x = rand_f64(&mut rng);
            let n: i32 = match rng.below(5) {
                0 => rng.below(2047) as i32 - 1023,
                1 => rng.below(1025) as i32 + 1023,
                2 => -(rng.below(1023) as i32) - 1022,
                3 => rng.next() as i32,
                _ => rng.below(120) as i32 - 60,
            };
            (r[0], r[1], r[20]) = (x.to_bits() as u32, (x.to_bits() >> 32) as u32, n as u32);
            xs.push((x, n));
        }
        let r = runner::run(&dxil, &st, &mem).unwrap();
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (x, n) = xs[l];
            checked += 1;
            let fract = if x.is_finite() { (x - x.floor()).min(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)) } else { f64::NAN };
            for (name, g, w) in [("floor", pair(v, 2), x.floor()), ("ceil", pair(v, 4), x.ceil()), ("trunc", pair(v, 6), x.trunc()),
                                 ("fract", pair(v, 8), fract), ("trunc(-x)", pair(v, 22), (-x).trunc()), ("floor(|x|)", pair(v, 24), x.abs().floor())] {
                if !same(g, w) {
                    bad.push(format!("{name}({x:e} / {:#018x}) = {g:e}, want {w:e}", x.to_bits()));
                }
            }
            if v[10] as i32 != x as i32 { bad.push(format!("cvt_i32({x:e}) = {}, want {}", v[10] as i32, x as i32)); }
            if v[11] != x as u32 { bad.push(format!("cvt_u32({x:e}) = {}, want {}", v[11], x as u32)); }
            let (gs, ws) = (pair(v, 12), x.sqrt());
            match ulps(gs, ws) { 0 => sqrt_exact += 1, 1 => sqrt_1ulp += 1, _ => bad.push(format!("sqrt({x:e} / {:#018x}) = {gs:e}, want {ws:e}", x.to_bits())) }
            let u = ulps(pair(v, 14), 1.0 / x.sqrt());
            if u <= 2 { rsq_hist[u as usize] += 1 } else { rsq_hist[3] += 1; bad.push(format!("rsq({x:e}) = {:e}, want {:e}", pair(v, 14), 1.0 / x.sqrt())) }
            if let Some(w) = ldexp_ref(x, n) {
                ldexp_checked += 1;
                if !same(pair(v, 16), w) { bad.push(format!("ldexp({x:e} / {:#018x}, {n}) = {:e}, want {w:e}", x.to_bits(), pair(v, 16))); }
            }
        }
    }
    println!("wave{lanes}: lanes {checked}, ldexp checked {ldexp_checked}, sqrt exact {sqrt_exact} / 1 ulp {sqrt_1ulp}, rsq ulps 0/1/2/>2 {rsq_hist:?}");
    bad.sort(); bad.dedup();
    println!("{} distinct failures", bad.len());
    for b in bad.iter().take(40) { println!("  {b}"); }
    assert!(bad.is_empty());
}

fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1F;
    let m = (h & 0x3FF) as f64;
    match e {
        0 => s * m * (2.0f64).powi(-24),
        31 => if m == 0.0 { s * f64::INFINITY } else { f64::NAN },
        _ => s * (1.0 + m / 1024.0) * (2.0f64).powi(e as i32 - 15),
    }
}

/// All 16 float compares of f32, f64 and f16 and v_cmpx_lt against the CPU on random operand
/// pairs, a quarter of them equal. The VOP3 forms write SGPR pairs and the VOPC forms VCC.
/// One wave of pairs per dispatch in 50 dispatches, or n dispatches with XCL_FUZZ_ITERS=n,
/// as wave64 and as wave32. An f32
/// result that only matches with f32 denormals flushed to zero is counted and does not fail
/// the test. That comes from the shader's denormal mode, not from the compare.
#[test]
fn gpu_float_compares_random() {
    for wave64 in gpu_waves() {
        float_compares_random(wave64);
    }
}

fn float_compares_random(wave64: bool) {
    let iters: usize = std::env::var("XCL_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    const OPS: [&str; 16] = ["F", "LT", "EQ", "LE", "GT", "LG", "GE", "O", "U", "NGE", "NLG", "NGT", "NLE", "NEQ", "NLT", "TRU"];
    let want_of = |op: &str, a: f64, b: f64| match op {
        "F" => false, "LT" => a < b, "EQ" => a == b, "LE" => a <= b, "GT" => a > b, "LG" => a < b || a > b, "GE" => a >= b,
        "O" => !a.is_nan() && !b.is_nan(), "U" => a.is_nan() || b.is_nan(), "NGE" => !(a >= b), "NLG" => !(a < b || a > b),
        "NGT" => !(a > b), "NLE" => !(a <= b), "NEQ" => !(a == b), "NLT" => !(a < b), _ => true,
    };
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut total_bad = 0usize;
    for ty in ["F32", "F64", "F16"] {
        let (s0, s1) = if ty == "F64" { (V + 2, 4u64) } else { (V, 1u64) };
        let names: Vec<String> = OPS.iter().map(|o| format!("V_CMP_{o}_{ty}")).collect();
        let f3: Vec<[(&str, u64); 3]> = (0..16).map(|i| [("VDST", 20 + 2 * i as u64), ("SRC0", s0), ("SRC1", V + s1)]).collect();
        let fc: Vec<[(&str, u64); 2]> = (0..16).map(|_| [("SRC0", s0), ("VSRC1", s1)]).collect();
        let fm: Vec<[(&str, u64); 2]> = (0..16).map(|i| [("SDST", 60 + 2 * i as u64), ("SSRC0", 106)]).collect();
        let cmpx = format!("V_CMPX_LT_{ty}");
        let fx = [("SRC0", s0), ("VSRC1", s1)];
        let mut list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = Vec::new();
        for i in 0..16 { list.push((names[i].as_str(), "ENC_VOP3", &f3[i][..], None)); }
        for i in 0..16 { list.push((names[i].as_str(), "ENC_VOPC", &fc[i][..], None)); list.push(("S_MOV_B64", "ENC_SOP1", &fm[i][..], None)); }
        list.push((cmpx.as_str(), "ENC_VOPC", &fx[..], None));
        let p = prog(&list);
        let (dxil, mem) = kernel_wave(&p, wave64);
        let n = if wave64 { 64 } else { 32 };
        let mut bad: Vec<String> = Vec::new();
        let mut lanes = 0usize;
        let mut denorm = 0usize;
        for _ in 0..iters {
            let mut st = KernelState::default();
            let mut ab = Vec::new();
            for r in st.vgpr.iter_mut() {
                let (a, b) = match ty {
                    "F64" => {
                        let a = rand_f64(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { rand_f64(&mut rng) };
                        (r[2], r[3], r[4], r[5]) = (a.to_bits() as u32, (a.to_bits() >> 32) as u32, b.to_bits() as u32, (b.to_bits() >> 32) as u32);
                        (a, b)
                    }
                    "F32" => {
                        let g = |rng: &mut Rng| -> f32 { match rng.below(3) { 0 => rand_f64(rng) as f32, 1 => f32::from_bits(rng.next() as u32),
                            _ => f32::from_bits([0u32, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0xFFC1_2345, 1][rng.below(8) as usize]) } };
                        let a = g(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { g(&mut rng) };
                        (r[0], r[1]) = (a.to_bits(), b.to_bits());
                        (a as f64, b as f64)
                    }
                    _ => {
                        let g = |rng: &mut Rng| -> u16 { if rng.below(3) == 0 { [0u16, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE12, 1, 0x3C00][rng.below(9) as usize] } else { rng.next() as u16 } };
                        let a = g(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { g(&mut rng) };
                        // the high halves hold junk: compares read the low half
                        (r[0], r[1]) = (a as u32 | (rng.next() as u32) << 16, b as u32 | (rng.next() as u32) << 16);
                        (f16_to_f64(a), f16_to_f64(b))
                    }
                };
                ab.push((a, b));
            }
            let r = runner::run(&dxil, &st, &mem).unwrap();
            ab.truncate(n);
            lanes += n;
            for (i, op) in OPS.iter().enumerate() {
                let m3 = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
                let mc = r.sgpr[60 + 2 * i] as u64 | (r.sgpr[61 + 2 * i] as u64) << 32;
                for (l, &(a, b)) in ab.iter().enumerate() {
                    let w = want_of(op, a, b);
                    let ftz = |x: f64| if ty == "F32" && x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64 { 0.0f64.copysign(x) } else { x };
                    let wz = want_of(op, ftz(a), ftz(b));
                    for (enc, m) in [("VOP3", m3), ("VOPC", mc)] {
                        let g = m >> l & 1 == 1;
                        if g != w && g != wz { bad.push(format!("{ty} {enc} {op}({a:e}, {b:e}) = {g}")); } else if g != w { denorm += 1; }
                    }
                }
            }
            let ex = r.sgpr[126] as u64 | (r.sgpr[127] as u64) << 32;
            for (l, &(a, b)) in ab.iter().enumerate() {
                let ftz = |x: f64| if ty == "F32" && x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64 { 0.0f64.copysign(x) } else { x };
                let g = ex >> l & 1 == 1;
                if g != (a < b) && g != (ftz(a) < ftz(b)) { bad.push(format!("{ty} CMPX LT({a:e}, {b:e}) = {g}")); } else if g != (a < b) { denorm += 1; }
            }
        }
        bad.sort(); bad.dedup();
        println!("wave{n} {ty}: {lanes} lanes x 33 compares, {} distinct failures, {denorm} results that match only a flush of f32 denormals", bad.len());
        for b in bad.iter().take(15) { println!("  {b}"); }
        total_bad += bad.len();
    }
    assert_eq!(total_bad, 0);
}

fn f16_exact(h: u16) -> Option<i128> {
    let (e, m) = ((h >> 10) & 0x1F, (h & 0x3FF) as i128);
    let mag = match e {
        31 => return None,
        0 => m,
        _ => (m | 0x400) << (e - 1),
    };
    Some(if h & 0x8000 != 0 { -mag } else { mag })
}

fn f16_round_exact(v: i128, u: u32, neg_zero: bool) -> u16 {
    let sign = if v < 0 || (v == 0 && neg_zero) { 0x8000u16 } else { 0 };
    let m = v.unsigned_abs();
    let bits = |x: u128| 127 - x.leading_zeros(); // 2^bits <= x
    let min_normal = 1u128 << (u - 14);
    let q_log = if m < min_normal { u - 24 } else { bits(m) - 10 };
    let mut k = m >> q_log;
    if q_log > 0 {
        let (r, half) = (m & ((1u128 << q_log) - 1), 1u128 << (q_log - 1));
        if r > half || (r == half && k & 1 == 1) {
            k += 1;
        }
    }
    let rounded = k << q_log;
    if rounded < min_normal {
        return sign | (rounded >> (u - 24)) as u16;
    }
    let l = bits(rounded);
    let e = l as i32 - u as i32 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    sign | (e as u16) << 10 | ((rounded >> (l - 10)) as u16 & 0x3FF)
}

#[test]
fn f16_exact_reference_round_trips() {
    for h in 0..=u16::MAX {
        if let Some(v) = f16_exact(h) {
            assert_eq!(v as f64 / 2f64.powi(24), f16_to_f64(h), "{h:#06x}");
            for u in [24, 48] {
                assert_eq!(f16_round_exact(v << (u - 24), u, h & 0x8000 != 0), h, "{h:#06x} in units of 2^-{u}");
            }
        }
    }
    let (one, ulp) = (1i128 << 48, 1i128 << 38);
    let r = |v: i128| f16_round_exact(v, 48, false);
    assert_eq!(r(one + ulp / 2), 0x3C00, "a tie to the even 1.0");
    assert_eq!(r(one + 3 * ulp / 2), 0x3C02, "a tie to the even 1 + 2^-9");
    assert_eq!(r(one + ulp / 2 + 1), 0x3C01, "just above the tie");
    assert_eq!(r(one + ulp / 2 - 1), 0x3C00, "just below the tie");
    assert_eq!(r(-(one + ulp / 2 + 1)), 0xBC01);
    let max = f16_exact(0x7BFF).unwrap() << 24; // 65504; 65520 is the tie with 2^16
    assert_eq!(r(max + (16 << 48)), 0x7C00, "65520 rounds to inf");
    assert_eq!(r(max + (16 << 48) - 1), 0x7BFF);
    assert_eq!(r(1 << 23), 0x0000, "half the smallest subnormal: a tie to 0");
    assert_eq!(r((1 << 23) + 1), 0x0001);
    assert_eq!(r(3 << 23), 0x0002, "1.5 subnormal quanta: a tie to the even 2");
    assert_eq!(f16_round_exact(0, 48, true), 0x8000);
}

#[test]
fn gpu_cvt_f16_from_16_bit_integers_exhaustive() {
    for wave64 in gpu_waves() {
        cvt_f16_from_16_bit_integers(wave64);
    }
}

fn cvt_f16_from_16_bit_integers(wave64: bool) {
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>)> = Vec::new();
    for k in 0..16u64 {
        list.push(("V_CVT_F16_U16", "ENC_VOP1", vec![("VDST", 16 + k), ("SRC0", V + k)]));
        list.push(("V_CVT_F16_I16", "ENC_VOP1", vec![("VDST", 32 + k), ("SRC0", V + k)]));
    }
    let p = prog(&list.iter().map(|(n, e, f)| (*n, *e, &f[..], None)).collect::<Vec<_>>());
    let (dxil, mem) = kernel_wave(&p, wave64);
    let n = if wave64 { 64 } else { 32 };
    let mut rng = Rng(0xC0F1_6000_0000_0001);
    let mut bad: Vec<String> = Vec::new();
    let (mut checked, mut wrong_u, mut wrong_i) = (0usize, 0usize, 0usize);
    for d in 0..(65536 / (16 * n)) {
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().take(n).enumerate() {
            for k in 0..16 {
                let x = ((d * n + l) * 16 + k) as u32;
                r[k] = x | (rng.next() as u32) << 16;
            }
        }
        let r = runner::run(&dxil, &st, &mem).unwrap();
        for l in 0..n {
            for k in 0..16 {
                let x = ((d * n + l) * 16 + k) as u16;
                let (want_u, want_i) = (f16_round_exact((x as i128) << 24, 24, false), f16_round_exact((x as i16 as i128) << 24, 24, false));
                let (got_u, got_i) = (r.vgpr[l][16 + k] as u16, r.vgpr[l][32 + k] as u16);
                checked += 1;
                if got_u != want_u {
                    wrong_u += 1;
                    bad.push(format!("v_cvt_f16_u16({x}) = {got_u:#06x} ({}), want {want_u:#06x} ({})", f16_to_f64(got_u), f16_to_f64(want_u)));
                }
                if got_i != want_i {
                    wrong_i += 1;
                    bad.push(format!("v_cvt_f16_i16({}) = {got_i:#06x} ({}), want {want_i:#06x} ({})", x as i16, f16_to_f64(got_i), f16_to_f64(want_i)));
                }
            }
        }
    }
    println!("wave{n}: {checked} inputs each; wrong: v_cvt_f16_u16 {wrong_u}, v_cvt_f16_i16 {wrong_i}");
    for b in bad.iter().take(20) {
        println!("  {b}");
    }
    assert!(bad.is_empty(), "{} results differ from the exact reference", bad.len());
}

fn hwreg_imm(id: u64, offset: u64, width: u64) -> u64 {
    id | (offset << 6) | ((width - 1) << 11)
}

#[test]
fn round_mode_classification_and_lowering() {
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };

    // Default RNE (0x0): exact NOOP even when f16/f64 ops are present.
    let def = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&def, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_round_mode"), Some(&1));

    // Retail graphics pattern (`s_round_mode 0xC`: f32=RNE, f16/f64=RTZ) with no f16/f64 ops:
    // exact NOOP, zero HLSL overhead.
    let f32_only = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 2)], None),
    ]);
    let t = translate(&f32_only, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_round_mode"), Some(&1));

    // `s_round_mode 0xC` with f16 ops: emits `xcl_fp16_rm` and routes f16 rounding through `f32_to_f16_rm`.
    let f16_rtz = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&f16_rtz, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("static uint xcl_fp16_rm = 0u;"));
    assert!(t.hlsl.contains("xcl_fp16_rm = 3u;"));
    assert!(t.hlsl.contains("f32_to_f16_rm("));

    // Non-default f64 round mode: rejected instead of silently running RNE.
    let f64_rtz = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&f64_rtz, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_round_mode: non-default f64 round mode"), "{:?}", t.unsupported_why);

    // Non-default f32 round mode: rejected.
    for sp in [1u64, 2, 3] {
        let f32_rm = prog(&[
            ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", sp)], None),
            ("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 2)], None),
        ]);
        let t = translate(&f32_rm, &opt).unwrap();
        assert!(!t.complete(), "sp={sp}");
        assert!(t.unsupported_why.contains_key("s_round_mode: non-default f32 round mode"), "{:?}", t.unsupported_why);
    }

    // `s_setreg_imm32_b32`: full MODE[3:0] write, partial bit writes, non-MODE & denorm-bit NOOPs, rejections.
    let setreg_partial = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 1))], Some(1)),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 3, 1))], Some(1)),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_partial, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("xcl_fp16_rm = (xcl_fp16_rm & 2u) | 1u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("xcl_fp16_rm = (xcl_fp16_rm & 1u) | 2u;"), "{}", t.hlsl);

    let setreg_f32_bad = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 4))], Some(0x1)),
    ]);
    let t = translate(&setreg_f32_bad, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_imm32_b32: non-default f32 round mode"), "{:?}", t.unsupported_why);

    let setreg_f64_bad = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2))], Some(0x3)),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&setreg_f64_bad, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_imm32_b32: non-default f64 round mode"), "{:?}", t.unsupported_why);

    let setreg_noop = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(20, 0, 32))], Some(0x1234_5678)), // flat scratch
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 4, 4))], Some(0xF)),           // MODE denorm bits
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_noop, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_setreg_imm32_b32"), Some(&2));

    // `s_setreg_b32`: dynamic SGPR write to MODE[3:2] with f16 ops succeeds; f32/f64 rejected.
    let setreg_dyn = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_dyn, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("xcl_fp16_rm = (s0 & 3u);"), "{}", t.hlsl);

    let setreg_dyn_f32 = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 2)), ("SDST", 0)], None),
    ]);
    let t = translate(&setreg_dyn_f32, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_b32: dynamic f32 round mode"), "{:?}", t.unsupported_why);

    let setreg_dyn_f64 = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&setreg_dyn_f64, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_b32: dynamic f64 round mode"), "{:?}", t.unsupported_why);

    // Straight-line pre-scan stops at `s_endpgm`: unreachable trailing f64 op does not break `s_round_mode 0xC`.
    let mut trailing = prog(&[("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None)]);
    trailing.extend(encode("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None).unwrap());
    let t = translate(&trailing, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
}

#[test]
fn mode_round_fields_can_be_read_without_half_arithmetic() {
    let code = prog(&[
        ("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 4)), ("SDST", 2)], None),
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 12)], None),
        ("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 1, 3)), ("SDST", 3)], None),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 3, 1))], Some(0)),
        ("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 4)], None),
        ("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 2)), ("SDST", 5)], None),
        ("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 125)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("static uint xcl_fp16_rm = 0u;"));
    assert!(!t.hlsl.contains("0u = ((("));
    if gpu_ready() {
        let mut st = KernelState::default();
        st.sgpr[2..=5].fill(u32::MAX);
        let r = runner::run_code(&code, &st, &[]).unwrap();
        assert_eq!(&r.sgpr[2..=5], &[0, 6, 1, 0]);
    }

    for (id, offset, width) in [(2, 0, 1), (1, 3, 2), (1, 0, 32)] {
        let code = prog(&[("S_GETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(id, offset, width)), ("SDST", 2)], None)]);
        let t = translate(&code, &Options::default()).unwrap();
        assert!(t.unsupported_why.contains_key("s_getreg_b32: unmodeled hardware register field"));
    }
}

#[test]
fn gpu_nonfused_mad_and_mac_prevent_fma_contraction() {
    if !gpu_ready() {
        return;
    }
    // Two golden vectors where fused FMA and non-fused multiply-then-add disagree:
    //   lane 0 (overflow):     a = f32::MAX, b = 1 + 2^-23, c = -f32::MAX
    //     non-fused -> +inf (0x7F800000), fused -> 0x73FFFFFF
    //   lane 1 (cancellation): a = 1 + 2^-23, b = 1 + 2^-23, c = -(1 + 2^-22)
    //     non-fused -> +0.0 (0x00000000), fused -> 2^-46 (0x28800000)
    let code = prog(&[
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 0), ("VSRC1", 1)], None),
        ("V_MAC_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1)], None),
        ("V_MAD_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_MADMK_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 0), ("VSRC1", 2)], Some(0x3F80_0001)),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 0), ("VSRC1", 1)], Some(0xFF7F_FFFF)),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 15), ("SRC0", V + 0), ("VSRC1", 1)], Some(0xBF80_0002)),
        // Fused control on the exact same inputs proves the backend has hardware FMA and `precise` mattered.
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();

    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    let vectors = [
        (0x7F7F_FFFFu32, 0x3F80_0001u32, 0xFF7F_FFFFu32), // overflow
        (0x3F80_0001u32, 0x3F80_0001u32, 0xBF80_0002u32), // cancellation
    ];
    for (l, &(a, b, c)) in vectors.iter().enumerate() {
        st.vgpr[l][0] = a;
        st.vgpr[l][1] = b;
        st.vgpr[l][2] = c;
        st.vgpr[l][10] = c;
        st.vgpr[l][11] = c;
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();

    let v0 = &r.vgpr[0];
    assert_eq!(v0[10], 0x7F80_0000, "v_mac_f32 (VOP2) overflow must not fuse");
    assert_eq!(v0[11], 0x7F80_0000, "v_mac_f32 (VOP3) overflow must not fuse");
    assert_eq!(v0[12], 0x7F80_0000, "v_mad_f32 overflow must not fuse");
    assert_eq!(v0[13], 0x7F80_0000, "v_madmk_f32 overflow must not fuse");
    assert_eq!(v0[14], 0x7F80_0000, "v_madak_f32 overflow must not fuse");
    assert_eq!(v0[16], 0x73FF_FFFF, "v_fma_f32 overflow control should stay fused");

    let v1 = &r.vgpr[1];
    assert_eq!(v1[10], 0x0000_0000, "v_mac_f32 (VOP2) cancellation must not fuse");
    assert_eq!(v1[11], 0x0000_0000, "v_mac_f32 (VOP3) cancellation must not fuse");
    assert_eq!(v1[12], 0x0000_0000, "v_mad_f32 cancellation must not fuse");
    assert_eq!(v1[13], 0x0000_0000, "v_madmk_f32 cancellation must not fuse");
    assert_eq!(v1[15], 0x0000_0000, "v_madak_f32 cancellation must not fuse");
    assert_eq!(v1[16], 0x2880_0000, "v_fma_f32 cancellation control should stay fused");
}

#[test]
fn gpu_f16_add_and_sub_round_to_nearest_even() {
    let code = prog(&[
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_SUB_F16", "ENC_VOP2", &[("VDST", 5), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_SUBREV_F16", "ENC_VOP2", &[("VDST", 6), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 7), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL_HI", 3)], None),
    ]);
    let mut st = KernelState::default();
    for regs in &mut st.vgpr {
        regs[..4].copy_from_slice(&[0x3C00, 0x1200, 0x3C00_3C00, 0x1000_1200]);
    }
    for wave64 in gpu_waves() {
        let (result, lanes) = run_wave(&code, &st, wave64);
        for regs in result.vgpr.iter().take(lanes) {
            assert_eq!(regs[4..8], [0x3C01, 0x3BFE, 0xBBFE, 0x3C00_3C01]);
        }
    }
}

#[test]
fn gpu_f16_narrowing_uses_default_rounding_for_unary_results() {
    let code = prog(&[
        ("V_SQRT_F16", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_LDEXP_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_LDEXP_F16", "ENC_VOP2", &[("VDST", 12), ("SRC0", V + 3), ("VSRC1", 4)], None),
        ("V_FRACT_F16", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 5)], None),
    ]);
    let mut st = KernelState::default();
    for regs in &mut st.vgpr {
        regs[..6].copy_from_slice(&[0x3420, 0x7BFF, 1, 0x3C03, (-15i32) as u32, 0x8400]);
    }
    for wave64 in gpu_waves() {
        let (result, lanes) = run_wave(&code, &st, wave64);
        for regs in result.vgpr.iter().take(lanes) {
            assert_eq!(regs[10..14], [0x3810, 0x7C00, 0x0202, 0x3BFF]);
        }
    }
}

#[test]
fn gpu_f16_defaults_to_round_to_nearest_even() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();

    let cases: [(f32, u16); 7] = [
        (f32::from_bits(3075.0f32.to_bits() - 1), 0x6A01),
        (3075.0, 0x6A02),
        (f32::from_bits(3075.0f32.to_bits() + 1), 0x6A02),
        (3081.0, 0x6A04),
        (-3075.0, 0xEA02),
        (65520.0, 0x7C00),
        (f32::from_bits(0x3300_0001), 0x0001),
    ];
    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    for (lane, &(value, _)) in cases.iter().enumerate() {
        st.vgpr[lane][0] = value.to_bits();
    }
    // 3072 * (1 + 2^-10) is the exact binary16 midpoint 3075.
    st.vgpr[0][1] = 0xEA00_6A00;
    st.vgpr[0][2] = 0x3C01_3C01;

    let r = runner::run(&dxil, &st, &[]).unwrap();
    let actual: [u16; 7] = std::array::from_fn(|lane| r.vgpr[lane][10] as u16);
    let expected = cases.map(|(_, bits)| bits);
    let product = r.vgpr[0][11] as u16;
    let packed = r.vgpr[0][12];
    assert_eq!(
        (actual, product, packed),
        (expected, 0x6A02, 0xEA02_6A02),
        "conversion {actual:04x?}, multiplication {product:04x}, packed multiplication {packed:08x}",
    );
}

#[test]
fn gpu_s_round_mode_controls_f16_rounding_across_modes() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // Default RNE (0) -> v10 (v_cvt_f16_f32), v20 (v_mul_f16), v21 (v_pk_mul_f16), v22 (v_fma_mixlo_f16)
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 0)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 21), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_FMA_MIXLO_F16", "ENC_VOP3P", &[("VDST", 22), ("SRC0", V + 0), ("SRC1", 242), ("SRC2", 128)], None),
        // Switch to RTZ (3) via S_ROUND_MODE 0xC -> v11, v23, v24, v25
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 0)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 23), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 24), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_FMA_MIXLO_F16", "ENC_VOP3P", &[("VDST", 25), ("SRC0", V + 0), ("SRC1", 242), ("SRC2", 128)], None),
        // Switch to +inf (1) via S_ROUND_MODE 0x4 -> v12
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x4)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 0)], None),
        // Switch to -inf (2) via S_ROUND_MODE 0x8 -> v13
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x8)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 0)], None),
        // Switch back to RNE (0) via S_ROUND_MODE 0x0 -> v14
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 0)], None),
        // Switch to RTZ (3) via S_SETREG_IMM32_B32, then clear bit 0 -> -inf (2) -> v15
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2))], Some(3)),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 1))], Some(0)),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 0)], None),
        // Switch dynamically via S_SETREG_B32 from s0 (= 1 -> +inf) -> v16
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 0)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();

    // (f32 bits, RNE, RTZ, +inf, -inf)
    let cases: &[(u32, u16, u16, u16, u16)] = &[
        (0x3F80_0800, 0x3C00, 0x3C00, 0x3C01, 0x3C00), // +(1 + 2^-12)
        (0xBF80_0800, 0xBC00, 0xBC00, 0xBC00, 0xBC01), // -(1 + 2^-12)
        (0x3F80_1000, 0x3C00, 0x3C00, 0x3C01, 0x3C00), // +(1 + 2^-11): tie to even (0x3C00)
        (0x3F80_3000, 0x3C02, 0x3C01, 0x3C02, 0x3C01), // +(1 + 3*2^-11): tie to odd -> rounds up to 0x3C02
        (0xBF80_3000, 0xBC02, 0xBC01, 0xBC01, 0xBC02), // -(1 + 3*2^-11)
        (0x477F_E040, 0x7BFF, 0x7BFF, 0x7C00, 0x7BFF), // +65504.5
        (0xC77F_E040, 0xFBFF, 0xFBFF, 0xFBFF, 0xFC00), // -65504.5
        (0x477F_F000, 0x7C00, 0x7BFF, 0x7C00, 0x7BFF), // +65520.0 (tie to +inf)
        (0xC77F_F000, 0xFC00, 0xFBFF, 0xFBFF, 0xFC00), // -65520.0 (tie to -inf)
        (0x4788_B800, 0x7C00, 0x7BFF, 0x7C00, 0x7BFF), // +70000.0 (> 65536 overflow)
        (0xC788_B800, 0xFC00, 0xFBFF, 0xFBFF, 0xFC00), // -70000.0
        (0x33C0_0000, 0x0002, 0x0001, 0x0002, 0x0001), // +1.5 * 2^-24 (subnormal tie to even)
        (0x387F_FFFF, 0x0400, 0x03FF, 0x0400, 0x03FF), // subnormal-to-normal carry
        (0x3300_0000, 0x0000, 0x0000, 0x0001, 0x0000), // +2^-25 (exact tie to 0)
        (0x3300_0001, 0x0001, 0x0000, 0x0001, 0x0000), // +(2^-25 + eps)
        (0x8D80_0000, 0x8000, 0x8000, 0x8000, 0x8001), // tiny negative underflow
        (0x0000_0000, 0x0000, 0x0000, 0x0000, 0x0000), // +0.0
        (0x8000_0000, 0x8000, 0x8000, 0x8000, 0x8000), // -0.0
        (0x7F80_0000, 0x7C00, 0x7C00, 0x7C00, 0x7C00), // +inf
        (0xFF80_0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00), // -inf
    ];

    let mut st = KernelState::default();
    st.sgpr[0] = 1; // +inf for S_SETREG_B32
    st.sgpr[127] = 0;
    for (l, &(bits, ..)) in cases.iter().enumerate() {
        st.vgpr[l][0] = bits;
        // Pack two f16 overflow pairs: 65504 * (1 + 2^-10) = 65567.96875 -> +inf (RNE) vs 65504 (RTZ)
        // High half: -65504 * (1 + 2^-10) -> -inf (RNE) vs -65504 (RTZ)
        st.vgpr[l][1] = 0xFBFF_7BFF;
        st.vgpr[l][2] = 0x3C01_3C01;
    }
    // Lane 20: quiet NaN preservation check
    st.vgpr[20][0] = 0x7FC0_2000;

    let r = runner::run(&dxil, &st, &[]).unwrap();
    for (l, &(bits, rne, rtz, rtp, rtn)) in cases.iter().enumerate() {
        let v = &r.vgpr[l];
        assert_eq!(v[10] as u16, rne, "lane {l} ({bits:#010x}) RNE");
        assert_eq!(v[11] as u16, rtz, "lane {l} ({bits:#010x}) RTZ");
        assert_eq!(v[12] as u16, rtp, "lane {l} ({bits:#010x}) +inf");
        assert_eq!(v[13] as u16, rtn, "lane {l} ({bits:#010x}) -inf");
        assert_eq!(v[14] as u16, rne, "lane {l} ({bits:#010x}) reset to RNE");
        assert_eq!(v[15] as u16, rtn, "lane {l} ({bits:#010x}) setreg_imm32 partial -> -inf");
        assert_eq!(v[16] as u16, rtp, "lane {l} ({bits:#010x}) setreg_b32 dynamic -> +inf");
        // v_fma_mixlo_f16 is x * 1.0 + 0.0: -0.0 + +0.0 is +0.0 in both modes (IEEE 754 6.3)
        let mix = |want: u16| if bits == 0x8000_0000 { 0 } else { want };
        assert_eq!(v[22] as u16, mix(rne), "lane {l} ({bits:#010x}) v_fma_mixlo_f16 RNE");
        assert_eq!(v[25] as u16, mix(rtz), "lane {l} ({bits:#010x}) v_fma_mixlo_f16 RTZ");
    }
    // Verify V_MUL_F16 & V_PK_MUL_F16 honor RNE vs RTZ
    assert_eq!(r.vgpr[0][20] as u16, 0x7C00, "v_mul_f16 overflow RNE");
    assert_eq!(r.vgpr[0][23] as u16, 0x7BFF, "v_mul_f16 overflow RTZ");
    assert_eq!(r.vgpr[0][21], 0xFC00_7C00, "v_pk_mul_f16 overflow RNE");
    assert_eq!(r.vgpr[0][24], 0xFBFF_7BFF, "v_pk_mul_f16 overflow RTZ");
    // Verify NaN stays NaN across all 4 rounding modes
    for reg in [10, 11, 12, 13] {
        let h = r.vgpr[20][reg] as u16;
        assert!((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0, "reg v{reg} NaN corrupted: {h:#06x}");
    }
}

#[test]
fn gpu_mad_nf_f16_rounds_intermediate_in_binary16() {
    if !gpu_ready() {
        return;
    }
    // Compile a wave32 kernel containing the recompiler PRELUDE and exercise `mad_nf_f16` directly.
    let base = prog(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 0)], None)]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let mut hlsl = translate(&base, &opt).unwrap().hlsl;
    let probe = "\
    if (lane_id == 0u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x7BFFu), f16tof32(0x3C01u), f16tof32(0xFBFFu), 0u), 0u);\n\
    if (lane_id == 1u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x7BFFu), f16tof32(0x3C01u), f16tof32(0xFBFFu), 3u), 3u);\n\
    if (lane_id == 2u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x3C01u), f16tof32(0x3C01u), f16tof32(0xBC02u), 0u), 0u);\n";
    let marker = "    v0 = select(xcl_on, (v0), v0);\n";
    assert!(hlsl.contains(marker), "unexpected kernel body:\n{hlsl}");
    hlsl = hlsl.replace(marker, probe);

    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    let r = runner::run(&dxil, &st, &[]).unwrap();

    assert_eq!(r.vgpr[0][0], 0x7C00, "mad_nf_f16 RNE overflow: 65504*(1+2^-10)-65504 -> +inf");
    assert_eq!(r.vgpr[1][0], 0x0000, "mad_nf_f16 RTZ clamp: 65504*(1+2^-10)->65504, -65504 -> 0");
    assert_eq!(r.vgpr[2][0], 0x0000, "mad_nf_f16 cancellation: (1+2^-10)^2-(1+2^-9) -> 0 (not 2^-20)");
}

#[test]
fn ps_interp_p0_preserves_bits_and_p2_converts_to_float() {
    use super::PsInterface;
    let ps_code = prog(&[
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 2), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 1)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 2), ("ATTR", 2), ("ATTRCHAN", 0)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 8), ("VSRC", 2), ("ATTR", 3), ("ATTRCHAN", 0)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![
            el("SV_Position", 0, 1, 3, 0, 0xF),
            el("TEXCOORD", 0, 0, 3, 1, 0xF), // float4
            el("TEXCOORD", 1, 0, 1, 2, 0xF), // uint4
            el("TEXCOORD", 2, 0, 2, 3, 0xF), // int4
            el("TEXCOORD", 3, 0, 3, 4, 0xF), // flat float4
        ],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
    for read in ["asuint((float)xi.a0[0])", "asuint(xi.a1[0])", "asuint((float)xi.a1[1])", "asuint(xi.a2[0])", "asuint(xi.a3[0])"] {
        assert!(pt.hlsl.contains(read), "{read}:\n{}", pt.hlsl);
    }
    if super::dxc::location().is_ok() {
        super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    }
}

#[test]
fn gpu_ds_swizzle_and_bpermute_zero_inactive_lanes() {
    let run_case = |wave64: bool, split_wave64: bool| {
        let mut insns: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0x5555_5555)),
        ];
        if wave64 {
            insns.push(("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0x5555_5555)));
        }
        insns.extend_from_slice(&[
            // swizzle XOR 1: even lanes read inactive odd neighbor -> must be 0
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 1 << 2)], None),
            // bpermute targeting odd neighbor -> must be 0
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 1), ("DATA0", 0)], None),
            // bpermute targeting cross-half odd lane (lane ^ 33) -> must be 0
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 2), ("DATA0", 0)], None),
            // swizzle XOR 2: even lanes read active even lane -> must return its value
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 13), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 2 << 2)], None),
        ]);
        let code = prog(&insns);
        let t = translate(&code, &Options { wave64, split_wave64, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
        let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let mut st = KernelState::default();
        let lanes = if wave64 { 64 } else { 32 };
        for (l, r) in st.vgpr.iter_mut().take(lanes).enumerate() {
            r[0] = if l & 1 == 1 { 0xDEAD } else { 0x1000 + l as u32 };
            r[1] = 4 * (l ^ 1) as u32;
            r[2] = 4 * ((l ^ 33) & (lanes - 1)) as u32;
            for dst in 10..=13 {
                r[dst] = 0xCAFE_BABE;
            }
        }
        let r = runner::run(&dxil, &st, &[]).unwrap();
        for l in 0..lanes {
            let v = &r.vgpr[l];
            if l & 1 == 0 {
                assert_eq!(v[10], 0, "wave{lanes} split={split_wave64} lane {l}: ds_swizzle inactive src");
                assert_eq!(v[11], 0, "wave{lanes} split={split_wave64} lane {l}: ds_bpermute inactive src");
                assert_eq!(v[12], 0, "wave{lanes} split={split_wave64} lane {l}: ds_bpermute cross-half inactive src");
                assert_eq!(v[13], 0x1000 + (l ^ 2) as u32, "wave{lanes} split={split_wave64} lane {l}: ds_swizzle active src");
            } else {
                for dst in 10..=13 {
                    assert_eq!(v[dst], 0xCAFE_BABE, "wave{lanes} split={split_wave64} lane {l}: inactive dst v{dst}");
                }
            }
        }
    };
    for wave64 in gpu_waves() {
        run_case(wave64, false);
    }
    if gpu_ready() {
        run_case(true, true);
    }
}

fn ds_permute_reference(lanes: usize, exec: u64, addr: &[u32], offset: u32, data: &[u32], old: u32) -> Vec<u32> {
    let mut tmp = vec![0u32; lanes];
    for i in 0..lanes {
        if exec >> i & 1 == 1 {
            tmp[(addr[i].wrapping_add(offset) >> 2) as usize % lanes] = data[i];
        }
    }
    (0..lanes).map(|i| if exec >> i & 1 == 1 { tmp[i] } else { old }).collect()
}

#[test]
fn gpu_ds_permute_b32() {
    let run_case = |wave64: bool, split_wave64: bool| {
        let lanes = if wave64 { 64 } else { 32 };
        let exec: u64 = if wave64 { 0xF0F0_FF0F_7777_FFFE } else { 0x7777_FFFE };
        let mut insns: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = vec![("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(exec as u32))];
        if wave64 {
            insns.push(("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some((exec >> 32) as u32)));
        }
        insns.extend_from_slice(&[
            ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 1), ("DATA0", 0)], None),
            ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 2), ("DATA0", 0)], None),
            ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 3), ("DATA0", 0), ("OFFSET0", 8)], None),
        ]);
        let code = prog(&insns);
        let t = translate(&code, &Options { wave64, split_wave64, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
        let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let mut st = KernelState::default();
        let (mut data, mut a1, mut a2, mut a3) = (vec![], vec![], vec![], vec![]);
        for l in 0..lanes {
            data.push(0x1000 + l as u32);
            a1.push(4 * (lanes - 1 - l) as u32);
            a2.push(4 * (l / 2) as u32 + 0x0001_0000);
            a3.push(4 * l as u32);
        }
        for (l, r) in st.vgpr.iter_mut().take(lanes).enumerate() {
            (r[0], r[1], r[2], r[3]) = (data[l], a1[l], a2[l], a3[l]);
            for dst in 10..=12 {
                r[dst] = 0xCAFE_BABE;
            }
        }
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let want = [
            ds_permute_reference(lanes, exec, &a1, 0, &data, 0xCAFE_BABE),
            ds_permute_reference(lanes, exec, &a2, 0, &data, 0xCAFE_BABE),
            ds_permute_reference(lanes, exec, &a3, 8, &data, 0xCAFE_BABE),
        ];
        for l in 0..lanes {
            for (k, w) in want.iter().enumerate() {
                assert_eq!(r.vgpr[l][10 + k], w[l], "wave{lanes} split={split_wave64} lane {l}: v{}", 10 + k);
            }
        }
    };
    for wave64 in gpu_waves() {
        run_case(wave64, false);
    }
    if gpu_ready() {
        run_case(true, true);
    }
}

#[test]
fn ds_permute_exchange_fits_before_bpermute() {
    let operands: &[(&str, u64)] = &[("VDST", 4), ("ADDR", 1), ("DATA0", 2)];
    let code = prog(&[
        ("DS_PERMUTE_B32", "ENC_DS", operands, None),
        ("DS_BPERMUTE_B32", "ENC_DS", operands, None),
    ]);
    let t = translate(&code, &Options { wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("#define XCL_X_SLOT 64u"));
}

#[test]
fn ds_permute_b32_refused_where_upper_lanes_do_not_exist() {
    let code = prog(&[("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 1), ("DATA0", 0)], None)]);
    let ps = Mode::Pixel(super::PsInterface { inputs: vec![], outputs: vec![], input_ena: 0x7F, attrs: None });
    let t = translate(&code, &Options { mode: ps.clone(), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert_eq!(t.unsupported.get("ds_permute_b32"), Some(&1), "{:?}", t.unsupported);
    let t = translate(&code, &Options { mode: ps, wave64: true, split_wave64: false, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let t = translate(&code, &Options { wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
}

#[test]
fn gpu_ldexp_f32() {
    for wave64 in gpu_waves() {
        let p = prog(&[("V_LDEXP_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1)], None)]);
        const CASES: [(u32, i32, u32); 15] = [
            (1e30f32.to_bits(), -150, 1e30f32.to_bits() - (150 << 23)),
            (0.5f32.to_bits(), 128, 0x7F00_0000),            // 2^127
            (2.0f32.to_bits(), -127, 0x0080_0000),           // 2^-126 (min normal)
            (0.0f32.to_bits(), 200, 0x0000_0000),            // +0 stays +0
            (0x8000_0000, 200, 0x8000_0000),                 // -0 stays -0
            (f32::INFINITY.to_bits(), -200, 0x7F80_0000),    // +inf stays +inf
            (f32::NEG_INFINITY.to_bits(), -200, 0xFF80_0000),// -inf stays -inf
            (0x7FC0_1234, 10, 0x7FC0_1234),                  // NaN stays NaN
            (1.0f32.to_bits(), -149, 0x0000_0001),           // min subnormal
            (1.0f32.to_bits(), -150, 0x0000_0000),           // tie to even -> 0
            (1.5f32.to_bits(), -149, 0x0000_0002),           // tie to even -> 2
            (0x0000_0001, 149, 1.0f32.to_bits()),            // subnormal input -> 1.0
            (0x0000_0001, 276, 0x7F00_0000),                 // subnormal input -> 2^127
            (1.0f32.to_bits(), i32::MAX, 0x7F80_0000),       // overflow to +inf
            ((-1.0f32).to_bits(), i32::MIN, 0x8000_0000),    // underflow to -0
        ];
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let (x, n, _) = CASES[l % CASES.len()];
            (r[0], r[1]) = (x, n as u32);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (x, n, want) = CASES[l % CASES.len()];
            let got = v[4];
            let both_nan = f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan();
            assert!(
                got == want || both_nan,
                "wave{lanes} lane {l} v_ldexp_f32({x:#010x}, {n}) = {got:#010x}, want {want:#010x}"
            );
        }
    }
}

#[test]
fn gpu_ldexp_f16_special_values_and_signed_exponents() {
    let code = prog(&[("V_LDEXP_F16", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 1)], None)]);
    let cases: [(u16, i16, u16); 10] = [
        (0x0000, 200, 0x0000),
        (0x8000, i16::MAX, 0x8000),
        (0x8000, i16::MIN, 0x8000),
        (0x7C00, -200, 0x7C00),
        (0xFC00, i16::MIN, 0xFC00),
        (0x7E00, 5, 0x7E00),
        (0x3C00, -30, 0x0000),
        (0xBC00, -300, 0x8000),
        (0x0001, 10, 0x0400),
        (0x3C00, 3, 0x4800),
    ];
    for wave64 in gpu_waves() {
        let mut st = KernelState::default();
        for (lane, regs) in st.vgpr.iter_mut().enumerate() {
            let (x, n, _) = cases[lane % cases.len()];
            (regs[0], regs[1], regs[4]) = (x.into(), n as u16 as u32, 0xA5A5_0000);
        }
        let (result, lanes) = run_wave(&code, &st, wave64);
        for (lane, regs) in result.vgpr.iter().take(lanes).enumerate() {
            let (x, n, expected) = cases[lane % cases.len()];
            let actual = regs[4] as u16;
            assert_eq!(regs[4] >> 16, 0xA5A5);
            let is_nan = |h: u16| h & 0x7FFF > 0x7C00;
            assert!(actual == expected || (is_nan(actual) && is_nan(expected)),
                "wave{lanes} lane {lane}: ldexp_f16({x:#06x}, {n}) = {actual:#06x}, expected {expected:#06x}");
        }
    }
}

#[test]
fn gpu_f64_literal_high_word() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // 1.0 + literal(1.5 = 0x3FF8_0000_0000_0000) == 2.5
            ("V_ADD_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", 255)], Some(0x3FF8_0000)),
            // compare result against literal(2.5 = 0x4004_0000_0000_0000)
            ("V_CMP_EQ_F64", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 4), ("SRC1", 255)], Some(0x4004_0000)),
            // verify integer 64-bit literal still goes in low 32 bits
            ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 6), ("SRC1", 255)], Some(0x1234_5678)),
        ]);
        let mut st = KernelState::default();
        let one = 1.0f64.to_bits();
        for r in st.vgpr.iter_mut() {
            (r[0], r[1]) = (one as u32, (one >> 32) as u32);
            (r[6], r[7]) = (0x1234_5678, 0);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(pair(v, 4), 2.5f64, "wave{lanes} lane {l} v_add_f64 literal");
        }
        let expected_hi = if wave64 { u32::MAX } else { 0 };
        assert_eq!((r.sgpr[0], r.sgpr[1]), (u32::MAX, expected_hi), "wave{lanes} v_cmp_eq_f64 literal");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (u32::MAX, expected_hi), "wave{lanes} v_cmp_eq_u64 literal");
    }
}

#[test]
fn gpu_pknorm_i16_f32_nan_and_fmt_encode_snorm_ufloat() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32| [base, 0, 0x100, ident | fmt << 12];
    // Formats: 43 = 11_11_10_FLOAT (X:10b, Y:11b, Z:11b), 36 = 10_11_11_FLOAT (X:11b, Y:11b, Z:10b), 24 = 16_16_SNORM
    let p = prog(&[
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 0)], None), // (NaN, NaN)
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1)], None), // (NaN, 1.0)
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 0)], None), // (-1.0, NaN)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 4), ("SOFFSET", 128)], None), // 11_11_10_FLOAT (-1, -inf, 1)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 24), ("SRSRC", 5), ("SOFFSET", 128)], None), // 11_11_10_FLOAT (NaN, +inf, -0.5)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 6), ("SOFFSET", 128)], None), // 10_11_11_FLOAT (-1, -inf, 1)
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 28), ("SRSRC", 7), ("SOFFSET", 128)], None),  // 16_16_SNORM (NaN, -1.0)
    ]);
    let mut st = KernelState::default();
    st.sgpr[16..20].copy_from_slice(&vs(0x200, 43));
    st.sgpr[20..24].copy_from_slice(&vs(0x204, 43));
    st.sgpr[24..28].copy_from_slice(&vs(0x208, 36));
    st.sgpr[28..32].copy_from_slice(&vs(0x20C, 24));
    for r in st.vgpr.iter_mut() {
        r[0] = f32::NAN.to_bits();
        r[1] = 1.0f32.to_bits();
        r[2] = (-1.0f32).to_bits();
        // (-1.0, -Inf, 1.0)
        r[20] = (-1.0f32).to_bits();
        r[21] = f32::NEG_INFINITY.to_bits();
        r[22] = 1.0f32.to_bits();
        // (NaN, +Inf, -0.5)
        r[24] = f32::NAN.to_bits();
        r[25] = f32::INFINITY.to_bits();
        r[26] = (-0.5f32).to_bits();
        // (NaN, -1.0)
        r[28] = f32::NAN.to_bits();
        r[29] = (-1.0f32).to_bits();
    }
    let mem = vec![0u32; 0x100];
    let r = runner::run_code(&p, &st, &mem).unwrap();
    assert_eq!(r.vgpr[0][10], 0x0000_0000, "v_cvt_pknorm_i16_f32(NaN, NaN)");
    assert_eq!(r.vgpr[0][11], 0x7FFF_0000, "v_cvt_pknorm_i16_f32(NaN, 1.0)");
    assert_eq!(r.vgpr[0][12], 0x0000_8001, "v_cvt_pknorm_i16_f32(-1.0, NaN)");
    // 11_11_10_FLOAT (-1, -inf, 1): X(10b)=0, Y(11b)=0, Z(11b)=0x3C0 -> 0x3C0 << 21 = 0x7800_0000
    assert_eq!(r.mem[0x200 / 4], 0x3C0 << 21, "11_11_10_FLOAT (-1, -inf, 1)");
    // 11_11_10_FLOAT (NaN, +inf, -0.5): X(10b)=0x3F0 (NaN), Y(11b)=0x7C0 (+inf), Z(11b)=0
    assert_eq!(r.mem[0x204 / 4], 0x3F0 | (0x7C0 << 10), "11_11_10_FLOAT (NaN, +inf, -0.5)");
    // 10_11_11_FLOAT (-1, -inf, 1): X(11b)=0, Y(11b)=0, Z(10b)=0x1E0 -> 0x1E0 << 22 = 0x7800_0000
    assert_eq!(r.mem[0x208 / 4], 0x1E0 << 22, "10_11_11_FLOAT (-1, -inf, 1)");
    // 16_16_SNORM (NaN, -1.0): lo=0, hi=-32767 (0x8001)
    assert_eq!(r.mem[0x20C / 4], 0x8001_0000, "16_16_SNORM (NaN, -1.0)");
}

#[test]
fn gpu_f32_norm_instructions_and_stores_round_once() {
    if !gpu_ready() {
        return;
    }
    let inputs = [
        f32::from_bits(0x37c0_00c0),
        f32::from_bits(0x3840_0180),
        -f32::from_bits(0x3840_0180),
        f32::from_bits(0x3b00_8081),
        -f32::from_bits(0x3c41_8306),
        0.5,
        -0.5,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let formats = [("16_UNORM", 16, false), ("16_SNORM", 16, true), ("8_UNORM", 8, false), ("8_SNORM", 8, true)];
    let code = prog(&[
        ("V_CVT_PKNORM_U16_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 1)], None),
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 1), ("SRC1", V + 1)], None),
        ("BUFFER_STORE_FORMAT_X", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    for (format, width, signed) in formats {
        let fmt = super::emit::BUF_FORMATS.iter().position(|&f| f == format).unwrap() as u32;
        let mut st = KernelState::default();
        let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
        st.sgpr[12..16].copy_from_slice(&[0x100, 4 << 16, 64, ident | fmt << 12]);
        for (lane, regs) in st.vgpr.iter_mut().enumerate() {
            regs[0] = lane as u32;
            regs[1] = inputs[lane % inputs.len()].to_bits();
        }
        let r = runner::run_code(&code, &st, &[0u32; 128]).unwrap();
        for (lane, x) in inputs.into_iter().enumerate() {
            let reference = |width: u32, signed| {
                let mask = (1u32 << width) - 1;
                let scale = if signed { mask >> 1 } else { mask };
                let clamped = if x.is_nan() { 0.0 } else { x.clamp(if signed { -1.0 } else { 0.0 }, 1.0) };
                ((f64::from(clamped) * f64::from(scale)).round_ties_even() as i32 as u32) & mask
            };
            let (u, s) = (reference(16, false), reference(16, true));
            assert_eq!(r.vgpr[lane][10], u | u << 16, "UNORM pack, lane {lane}");
            assert_eq!(r.vgpr[lane][11], s | s << 16, "SNORM pack, lane {lane}");
            let mask = (1u32 << width) - 1;
            assert_eq!(r.mem[64 + lane] & mask, reference(width, signed), "{format}, lane {lane}");
        }
    }
}

#[test]
fn gpu_signed_bfe_arithmetic_shift() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // s_bfe_i64(-1, offset=8, width=64) -> -1
            ("S_BFE_I64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 4), ("SSRC1", 6)], None),
            // s_bfe_i64(0x8000_0000_0000_0000, offset=32, width=40) -> 0xFFFF_FFFF_8000_0000
            ("S_BFE_I64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 8), ("SSRC1", 10)], None),
            // s_bfe_i32(0x8000_0000, offset=16, width=20) -> 0xFFFF_8000
            ("S_BFE_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 11), ("SSRC1", 13)], None),
            // v_bfe_i32(0x8000_0000, offset=16, width=20) -> 0xFFFF_8000
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 0), ("SRC1", 144), ("SRC2", 148)], None),
            // v_bfe_i32(0x0000_8000, offset=0, width=16) -> 0xFFFF_8000 (o + w < 32)
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("SRC1", 128), ("SRC2", 144)], None),
            // v_bfe_i32(0xFFFF_FFFF, offset=4, width=0) -> 0
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 2), ("SRC1", 132), ("SRC2", 128)], None),
        ]);
        let mut st = KernelState::default();
        (st.sgpr[4], st.sgpr[5]) = (u32::MAX, u32::MAX);
        st.sgpr[6] = (64 << 16) | 8;
        (st.sgpr[8], st.sgpr[9]) = (0, 0x8000_0000);
        st.sgpr[10] = (40 << 16) | 32;
        st.sgpr[11] = 0x8000_0000;
        st.sgpr[13] = (20 << 16) | 16;
        for r in st.vgpr.iter_mut() {
            r[0] = 0x8000_0000;
            r[1] = 0x0000_8000;
            r[2] = u32::MAX;
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        assert_eq!((r.sgpr[0], r.sgpr[1]), (u32::MAX, u32::MAX), "wave{lanes} s_bfe_i64(-1, 8, 64)");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (0x8000_0000, u32::MAX), "wave{lanes} s_bfe_i64(i64::MIN, 32, 40)");
        assert_eq!(r.sgpr[12], 0xFFFF_8000, "wave{lanes} s_bfe_i32(i32::MIN, 16, 20)");
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[4], 0xFFFF_8000, "wave{lanes} lane {l} v_bfe_i32(i32::MIN, 16, 20)");
            assert_eq!(v[5], 0xFFFF_8000, "wave{lanes} lane {l} v_bfe_i32(0x8000, 0, 16)");
            assert_eq!(v[6], 0, "wave{lanes} lane {l} v_bfe_i32(width=0)");
        }
    }
}

#[test]
fn gpu_cmp_class_f16_preserves_snan() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // 0x7C01 (sNaN) tested against mask 1 << 0 (sNaN) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 0), ("SRC1", 129)], None),
            // 0x7C01 (sNaN) tested against mask 1 << 1 (qNaN) -> false
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 0), ("SRC1", 130)], None),
            // 0x7E01 (qNaN) tested against mask 1 << 0 (sNaN) -> false
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 1), ("SRC1", 129)], None),
            // 0x7E01 (qNaN) tested against mask 1 << 1 (qNaN) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 1), ("SRC1", 130)], None),
            // 0xBC00 (-1.0) with ABS tested against mask 1 << 8 (+normal) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 255), ("ABS", 1)], Some(1 << 8)),
            // 0xBC00 (-1.0) with NEG tested against mask 1 << 8 (+normal) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 255), ("NEG", 1)], Some(1 << 8)),
        ]);
        let mut st = KernelState::default();
        for r in st.vgpr.iter_mut() {
            r[0] = 0x7C01; // f16 sNaN
            r[1] = 0x7E01; // f16 qNaN
            r[2] = 0xBC00; // f16 -1.0
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        let all = (u32::MAX, if wave64 { u32::MAX } else { 0 });
        assert_eq!((r.sgpr[0], r.sgpr[1]), all, "wave{lanes} f16 sNaN matches sNaN bit");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (0, 0), "wave{lanes} f16 sNaN does not match qNaN bit");
        assert_eq!((r.sgpr[4], r.sgpr[5]), (0, 0), "wave{lanes} f16 qNaN does not match sNaN bit");
        assert_eq!((r.sgpr[6], r.sgpr[7]), all, "wave{lanes} f16 qNaN matches qNaN bit");
        assert_eq!((r.sgpr[8], r.sgpr[9]), all, "wave{lanes} f16 abs(-1.0) matches +normal");
        assert_eq!((r.sgpr[10], r.sgpr[11]), all, "wave{lanes} f16 neg(-1.0) matches +normal");
    }
}

#[test]
fn wave32_src_vccz_execz_and_saveexec_b32_hlsl() {
    let w32_code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 124)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 251)], None),
    ]);
    let t32 = translate(&w32_code, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t32.complete(), "{:?}", t32.unsupported_why);
    assert!(t32.hlsl.contains("(uint)(vcc_lo == 0u)"), "{}", t32.hlsl);
    assert!(t32.hlsl.contains("(uint)(exec_lo == 0u)"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("vcc_lo | vcc_hi"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("exec_lo | exec_hi"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("& exec_hi"), "{}", t32.hlsl);

    let w64_code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 252)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 251)], None),
    ]);
    let t64 = translate(&w64_code, &Options { wave64: true, ..Default::default() }).unwrap();
    assert!(t64.complete(), "{:?}", t64.unsupported_why);
    assert!(t64.hlsl.contains("(uint)((vcc_lo | vcc_hi) == 0u)"), "{}", t64.hlsl);
    assert!(t64.hlsl.contains("(uint)((exec_lo | exec_hi) == 0u)"), "{}", t64.hlsl);
}

#[test]
fn sdwa_f16_inline_constants_hlsl() {
    let code = prog(&[
        ("V_ADD_F16", "VOP2_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("DST_SEL", 6)], None),
        ("V_MUL_F16", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 244), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CVT_F32_F16", "VOP1_VOP_SDWA", &[("VDST", 12), ("SRC0", 0xF9), ("VSRC0", 243), ("S0", 1), ("SRC0_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("SD", 1), ("SDST", 20)], None),
    ]);
    let t = translate(&code, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("sdwa_sel(0x00003C00u"), "{}", t.hlsl);
    assert!(t.hlsl.contains("sdwa_sel(0x00004000u"), "{}", t.hlsl);
    assert!(t.hlsl.contains("sdwa_sel(0x0000BC00u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0x3F800000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0x40000000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0xBF800000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("uint v244"), "{}", t.hlsl);

    let int_src = prog(&[("V_CVT_F16_U16", "ENC_VOP1", &[("VDST", 13), ("SRC0", 242)], None)]);
    let t_int = translate(&int_src, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t_int.complete(), "{:?}", t_int.unsupported_why);
    assert!(t_int.hlsl.contains("((0x3F800000u) & 0xFFFFu)"), "{}", t_int.hlsl);
    assert!(!t_int.hlsl.contains("0x00003C00u"), "{}", t_int.hlsl);
}

#[test]
fn gpu_wave32_and_wave64_src_vccz_execz() {
    if !gpu_ready() {
        return;
    }
    let code32 = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 251)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 252)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 124)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 126)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 5)], None),
    ]);
    let mut st32 = KernelState::default();
    st32.sgpr[106] = 0;
    st32.sgpr[107] = 0x1234_5678;
    st32.sgpr[127] = 1;
    st32.sgpr[124] = 0x0000_000F;
    let (r32, lanes32) = run_wave(&code32, &st32, false);
    assert_eq!(lanes32, 32);
    assert_eq!(r32.sgpr[0], 1, "wave32 src_vccz ignores vcc_hi");
    assert_eq!(r32.sgpr[1], 0, "wave32 src_vccz clears once vcc_lo != 0");
    assert_eq!(r32.sgpr[3], 1, "wave32 src_execz ignores exec_hi");
    assert_eq!(r32.sgpr[4], 0, "wave32 src_execz clears once exec_lo != 0");
    assert_eq!(r32.sgpr[6], 0xF, "wave32 s_and_saveexec_b32 with m0");
    for (l, v) in r32.vgpr.iter().take(32).enumerate() {
        assert_eq!(v[0], 1, "wave32 lane {l} v_mov_b32 src_vccz");
    }

    let code64 = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 107), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 251)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 251)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 2)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 252)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)], None),
    ]);
    let mut st64 = KernelState::default();
    st64.sgpr[106] = 0;
    st64.sgpr[107] = 1;
    st64.sgpr[2] = 0;
    st64.sgpr[3] = 1;
    let r64 = runner::run_code(&code64, &st64, &[]).unwrap();
    assert_eq!(r64.sgpr[0], 0, "wave64 src_vccz checks vcc_hi");
    assert_eq!(r64.sgpr[1], 1, "wave64 src_vccz once vcc_hi == 0");
    assert_eq!((r64.sgpr[8], r64.sgpr[9]), (1, 0), "wave64 s_mov_b64 src_vccz");
    assert_eq!(r64.sgpr[6], 0, "wave64 src_execz checks exec_hi");
    assert_eq!(r64.sgpr[7], 1, "wave64 src_execz once exec_hi == 0");
}

#[test]
fn gpu_sdwa_f16_inline_constants() {
    let code = prog(&[
        ("V_ADD_F16", "VOP2_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("DST_SEL", 6)], None),
        ("V_MUL_F16", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 244), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CVT_F32_F16", "VOP1_VOP_SDWA", &[("VDST", 12), ("SRC0", 0xF9), ("VSRC0", 243), ("S0", 1), ("SRC0_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("SD", 1), ("SDST", 20)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 2), ("VSRC1", 242), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)], None),
    ]);
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[1] = 0x4000_3800; // WORD_1 = 2.0h, WORD_0 = 0.5h
        r[2] = 0x3800_0000; // WORD_1 = 0.5h
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&code, &st, wave64);
        let all = (u32::MAX, if wave64 { u32::MAX } else { 0 });
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[10], 0x4200, "wave{lanes} lane {l} v_add_f16_sdwa 1.0 + 2.0");
            assert_eq!(v[11], 0x4400, "wave{lanes} lane {l} v_mul_f16_sdwa 2.0 * 2.0");
            assert_eq!(v[12], (-1.0f32).to_bits(), "wave{lanes} lane {l} v_cvt_f32_f16_sdwa -1.0");
        }
        assert_eq!((r.sgpr[20], r.sgpr[21]), (0, 0), "wave{lanes} v_cmp_lt_f16_sdwa 1.0 < 0.5");
        assert_eq!((r.sgpr[22], r.sgpr[23]), all, "wave{lanes} v_cmp_lt_f16_sdwa 0.5 < 1.0");
    }
}

#[test]
fn fspx_compute_user_sgprs_from_rsrc2() {
    use crate::gpu::xbox::tests::container;

    let counts = [4u32, 8].map(|count| {
        // Two directory entries: type 0 holds the compute registers, type 6 holds the code.
        let mut part = Vec::from(*b"FSPX");
        for word in [1u32, 2, 0x74, 0, 0x30, 0x1c, 0, 6, 0x4c, 0x28, 0] {
            part.extend_from_slice(&word.to_le_bytes());
        }
        part.resize(0x74, 0);
        let header = &mut part[0x30..0x4c];
        header[0x0c..0x12].copy_from_slice(&[1, 0, 1, 0, 1, 0]); // numthreads(1, 1, 1)
        let rsrc2 = (count << 1) | 0x81; // set unrelated bits on both sides of USER_SGPR
        header[0x18..0x1c].copy_from_slice(&rsrc2.to_le_bytes());
        part[0x70..].copy_from_slice(&0xbf81_0000u32.to_le_bytes()); // s_endpgm
        let blob = container(&[(b"ACCC", &part)]);
        assert_eq!(super::pso::fspx(&blob).unwrap().rsrc2, rsrc2);
        super::pso::compute_user_sgprs(&blob)
    });
    assert_eq!(counts, [Some(4), Some(8)]);
}

/// Verifies early-exit-safe, bounded `xcl_s_barrier()` on real multi-warp compute shaders:
/// 1. All 4 hardware wave32s in `numthreads(128, 1, 1)` exchange LDS values across two back-to-back
///    `s_barrier` generations (both split-wave64 and native wave32).
/// 2. Upper console wave (`tid >= 64`) takes an early `s_endpgm` while the lower console wave
///    (`tid 0..63`, 2 hardware wave32s) executes two `s_barrier`s and exchanges LDS across its
///    surviving warps — legal on RDNA2, would permanently deadlock `bar.sync 0` on NVIDIA.
#[test]
fn gpu_safe_s_barrier_multi_warp_and_early_endpgm() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};

    // Shader 1: 4 hardware wave32s, 2 consecutive s_barrier generations exchanging across waves
    // using disjoint LDS banks (bank 0: 0..512, bank 1: 512..1024) so there is no WAR data race.
    let p_all = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),           // v1 = tid * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 1)], Some(512)),       // v9 = 512 + tid * 4 (bank 1)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 192), ("VSRC1", 0)], None),               // v2 = tid ^ 64
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)], None),           // v2 = (tid ^ 64) * 4 (bank 0)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 160), ("VSRC1", 0)], None),               // v3 = tid ^ 32
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),           // v3 = (tid ^ 32) * 4
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(512)),       // v3 = 512 + (tid ^ 32) * 4 (bank 1)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 0)], Some(0x100)),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 6)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 129), ("VSRC1", 4)], None),            // +1
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 4)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 3)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 1)], None),              // s0 + tid*4
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);

    let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [false, false, false], tg_size: false, tid_components: 1 };
    let base = 0x100usize;
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    for wave64 in [true, false] {
        let t = translate(&p_all, &on_gpu(Options { mode: Mode::Compute(cs.clone()), wave64, ..Default::default() })).unwrap();
        assert!(t.complete(), "wave64={wave64}: {:?}", t.unsupported_why);
        if !t.hlsl.contains("#define XCL_X_THREADS") {
            let body = t.hlsl.split_once("void main(").unwrap().1;
            assert_eq!(body.matches("GroupMemoryBarrierWithGroupSync();").count(), 2);
            assert!(!body.contains("xcl_s_barrier();"));
        }
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let out = runner::dispatch(&dxil, [1, 1, 1], &io, &vec![0u32; (base + 128 * 4) / 4]).unwrap();
        for tid in 0..128usize {
            let want = ((tid ^ 96) + 0x101) as u32;
            assert_eq!(out[(base + tid * 4) / 4], want, "wave64={wave64} tid={tid}: 2-gen xcl_s_barrier LDS exchange");
        }
    }

    // Shader 2: upper console wave (tid >= 64) exits early via `s_endpgm` before 2 `s_barrier`s;
    // lower console wave (tid 0..63, 2 hardware wave32s) synchronizes and exchanges LDS across halves.
    let p_early = prog(&[
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),                    // s10 = first tid in wave
        ("S_CMP_GE_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 192)], None),                       // scc = (s10 >= 64)
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None),                                     // skip s_endpgm if < 64
        ("S_ENDPGM", "ENC_SOPP", &[], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),           // v1 = tid * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 1)], Some(512)),       // v9 = 512 + tid * 4 (bank 1)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 160), ("VSRC1", 0)], None),               // v3 = tid ^ 32
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),           // v3 = (tid ^ 32) * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 3)], Some(512)),       // v2 = 512 + (tid ^ 32) * 4 (bank 1)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 0)], Some(0x200)),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 6)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 3)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 135), ("VSRC1", 4)], None),            // +7
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 4)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    for wave64 in [true, false] {
        let t = translate(&p_early, &on_gpu(Options { mode: Mode::Compute(cs.clone()), wave64, ..Default::default() })).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let out = runner::dispatch(&dxil, [2, 1, 1], &io, &vec![0u32; (base + 128 * 4) / 4]).unwrap();
        for tid in 0..64usize {
            assert_eq!(out[(base + tid * 4) / 4], (tid + 0x207) as u32, "wave64={wave64} tid={tid}: surviving warps must pass 2 s_barriers after upper waves retire");
        }
        for tid in 64..128usize {
            assert_eq!(out[(base + tid * 4) / 4], 0, "wave64={wave64} tid={tid}: exited wave must not store");
        }
    }
}

fn split_wait_test_code() -> Vec<u8> {
    prog(&[
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("S_CMP_GE_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 192)], None),
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 1)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 126)], None),
        ("V_CMP_LE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None),
        ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 8), ("SSRC1", 106)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 12), ("VSRC1", 0)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ])
}

fn split_wait_fixture() -> (String, Vec<u32>) {
    let cs = super::CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let t = translate(&split_wait_test_code(), &Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, legacy_compute_wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let mut io = vec![0u32; 128];
    io[0] = 0x100;
    io[super::IO_TRIP_AT] = 0x40;
    (t.hlsl, io)
}

#[test]
fn gpu_split_waves_wait_for_a_long_live_barrier_phase() {
    if !gpu_ready() || std::env::var("XCL_SAFE_BARRIER").is_ok_and(|v| v.trim() == "0") {
        return;
    }
    let (hlsl, mut io) = split_wait_fixture();
    io[1] = 896;
    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let out = runner::dispatch(&dxil, [1, 1, 1], &io, &[0u32; 192]).unwrap();
    assert_eq!(out[16..20], [0; 4]);
    for tid in 0..128 {
        assert_eq!(out[64 + tid], tid as u32 + if tid < 64 { 0 } else { 896 });
    }
}

#[test]
fn gpu_counted_barrier_sorts_with_thirty_two_waves_after_a_long_wait() {
    if !gpu_ready() || std::env::var("XCL_SAFE_BARRIER").is_ok_and(|v| v.trim() == "0") {
        return;
    }
    let code = prog(&[
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFE)], None),
    ]);
    let cs = super::CsInterface { threads: [1024, 1, 1], user_sgprs: 0, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let t = translate(&code, &Options { mode: Mode::Compute(cs), wave64: false, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let (prelude, _) = t.hlsl.split_once("[WaveSize(").unwrap();
    let report = "if (_still == 1024u) xcl_trip(2u);";
    assert!(prelude.contains(report));
    let prelude = prelude.replace(report, "if (_still == 1024u) { uint o; InterlockedExchange(test_release, 1u, o); xcl_trip(2u); }");
    let hlsl = format!("groupshared uint test_release;
{prelude}
groupshared uint keys[1024];
[WaveSize(32)]
[numthreads(1024, 1, 1)]
void main(uint gi : SV_GroupIndex) {{
    if (gi == 0u) test_release = 0u;
    xcl_s_barrier_init(gi);
    if (WaveActiveAnyTrue(gi == 1023u)) {{
        uint release = 0u;
        [loop] while (release == 0u) {{
            if (WaveIsFirstLane()) InterlockedAdd(test_release, 0u, release);
            release = WaveReadLaneFirst(release);
        }}
    }}
    keys[gi] = (gi * 37u) & 1023u;
    xcl_s_barrier();
    [loop] for (uint width = 2u; width <= 1024u; width *= 2u) {{
        [loop] for (uint step = width / 2u; step != 0u; step /= 2u) {{
            uint a = keys[gi];
            uint b = keys[gi ^ step];
            xcl_s_barrier();
            keys[gi] = ((gi & width) == 0u) == ((gi & step) == 0u) ? min(a, b) : max(a, b);
            xcl_s_barrier();
        }}
    }}
    RWByteAddressBuffer mem = ResourceDescriptorHeap[io.Load(452u)];
    mem.Store(256u + 4u * gi, keys[gi]);
    xcl_s_barrier_exit();
}}");
    let mut io = vec![0u32; 128];
    io[super::IO_TRIP_AT] = 0x40;
    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let out = runner::dispatch(&dxil, [1, 1, 1], &io, &[0u32; 64 + 1024]).unwrap();
    assert!(out[18] > 0, "the first barrier must report a long wait");
    assert_eq!(&out[64..64 + 1024], &(0..1024u32).collect::<Vec<_>>());
}

/// `XCL_SAMPLE_FSPX=<folder>`: every `*.bin` there that is an FSPX (driver) compute container –
/// e.g. the FSR4 pipelines XCL dumps to `<data>\shaders` – translates and compiles.
#[test]
#[ignore]
fn fspx_containers_translate_and_compile() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(dir) = std::env::var_os("XCL_SAMPLE_FSPX") else { return };
    let files: Vec<std::path::PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
    let next = AtomicUsize::new(0);
    let (n, ok) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..super::dxc::compile_threads() {
            s.spawn(|| loop {
                let Some(path) = files.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let blob = std::fs::read(path).unwrap();
                let Some(f) = super::pso::fspx(&blob) else { continue };
                n.fetch_add(1, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let name = path.file_name().unwrap().to_string_lossy();
                match super::pso::console_to_dxil(&blob, super::pso::Stage::Compute, 0) {
                    Ok(c) => {
                        ok.fetch_add(1, Ordering::Relaxed);
                        println!("FSPX {name}: threads {:?} rsrc2 {:#x}, {} B code -> {} B DXIL, {:.2} s", f.threads, f.rsrc2, f.code.len(), c.dxil.len(), t0.elapsed().as_secs_f64());
                    }
                    Err(err) => println!("FSPX {name}: FAILED {err}"),
                }
            });
        }
    });
    let (n, ok) = (n.into_inner(), ok.into_inner());
    println!("FSPX: {ok} / {n} translated and compiled");
    assert_eq!(ok, n);
}

/// The NGG program of `gpu_mesh_mode_two_passes` (three vertices per group from a vertex table,
/// x moved by the group; param0 = group, vertex, 0, 1; one primitive from thread 0) with its
/// interface, SGPRs and memory. `null_in_group_1`: group 1 exports its primitive null (bit 31
/// of the primitive export) – how a console program culls one.
fn ngg_triangle_per_group(null_in_group_1: bool) -> (Vec<u8>, super::MsInterface, Vec<u32>, Vec<u32>) {
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let ms = super::MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)], threads: 64, max_vertices: 64, max_primitives: 64 };
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None)
    };
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)), // wave index
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None), // thread in group
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 134)], None), // group = v5 >> 6
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None), // x += group
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None)); // thread 0
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    if null_in_group_1 {
        list.push(("S_LSHL_B32", "ENC_SOP2", vec![("SDST", 2), ("SSRC0", 1), ("SSRC1", 159)], None)); // group << 31
        list.push(("V_OR_B32", "ENC_VOP2", vec![("VDST", 21), ("SRC0", 2), ("VSRC1", 21)], None));
    }
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    // vertex table at byte 0x100, V# in s[16:19] (stride 16, 3 records)
    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[f(*x), f(*y), f(0.0), f(1.0)]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    (prog(&refs), ms, sg, mem)
}

/// The split flag of console mesh programs as host mesh shaders on the test adapter; None (the
/// test skips): its waves are not fixed at 32 lanes (`pso::mesh_native_waves`), or the run asks
/// for two passes (`XCL_MESH=two-pass`).
fn mesh_native_ready() -> Option<bool> {
    let split = runner::mesh_native_split().ok().flatten();
    if split.is_none() {
        println!("no host mesh form of console mesh programs on this adapter – skipping");
        if !super::pso::mesh_two_pass_forced() {
            runner::skip_test("no mesh shaders");
        }
    }
    split
}

/// `ngg_triangle_per_group`'s program as a host mesh shader for `ps_inputs` (DXIL).
fn native_mesh(code: &[u8], ms: &super::MsInterface, split: bool, ps_inputs: Option<Vec<crate::gpu::dxbc::SigElement>>) -> Vec<u8> {
    let t = translate(code, &Options { mode: Mode::Mesh(ms.clone()), wave64: true, split_wave64: split, mesh_native: Some(super::MeshNative { ps_inputs, lds_bytes: 0 }), ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    super::dxc::compile(&t.hlsl, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
}

const MESH_TEST_PS: &str = "struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n";

#[test]
fn gpu_mesh_native_converged_barrier_and_wave_exchange() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    let Some(split) = mesh_native_ready() else { return };
    let (mut code, ms, sg, mem) = ngg_triangle_per_group(false);
    let barrier = encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap();
    let at = code.windows(barrier.len()).position(|w| w == barrier).unwrap() + barrier.len();
    let read = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V), ("SRC1", 160)], None).unwrap();
    code.splice(at..at, read);
    let color = encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 20)], None).unwrap();
    let at = code.windows(color.len()).position(|w| w == color).unwrap();
    let from_peer = encode("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 11)], None).unwrap();
    code[at..at + color.len()].copy_from_slice(&from_peer);
    let t = translate(&code, &Options { mode: Mode::Mesh(ms), wave64: true, split_wave64: split, mesh_native: Some(super::MeshNative { ps_inputs: None, lds_bytes: 0 }), ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("GroupMemoryBarrier") && !t.hlsl.contains("xcl_x_meet("));
    assert!(t.hlsl.contains("[numthreads(32, 1, 1)]") && t.hlsl.contains(" = xcl_pack_read("));
    let dxil = super::dxc::compile(&t.hlsl, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    let ps = super::dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let px = runner::mesh_draw_native(&dxil, &ps, runner::MeshGroups::Direct(1), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert_eq!((px[29 * 32 + 3][0], px[29 * 32 + 3][3]), (32.0, 1.0));
    assert_eq!(px[29 * 32 + 19], [-1.0; 4]);
}

#[test]
fn native_mesh_uses_wave_intrinsics_in_branches_and_rejects_multi_wave_groups() {
    let options = Options {
        mode: Mode::Mesh(super::MsInterface { threads: 64, max_vertices: 3, max_primitives: 1, outputs: vec![] }),
        wave64: true,
        split_wave64: true,
        mesh_native: Some(super::MeshNative { ps_inputs: None, lds_bytes: 0 }),
        ..Default::default()
    };
    let barrier = prog(&[
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
    ]);
    let t = translate(&barrier, &options).unwrap();
    assert!(!t.hlsl.contains("GroupMemoryBarrier"));
    let meet = prog(&[
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let packed = translate(&meet, &options).unwrap();
    assert!(packed.hlsl.contains("xcl_pack_read(v0, hi_v0, first_lane(exec_lo, exec_hi))"));
    assert!(!packed.hlsl.contains("GroupMemoryBarrier") && !packed.hlsl.contains("groupshared"));
    let wide = Options { mode: Mode::Mesh(super::MsInterface { threads: 128, max_vertices: 3, max_primitives: 1, outputs: vec![] }), ..options };
    let err = translate(&meet, &wide).unwrap_err();
    assert!(err.contains("more than one console wave"), "{err}");
    let fallback = translate(&meet, &Options { mesh_native: None, ..wide }).unwrap();
    assert!(fallback.hlsl.contains("#define XCL_X_SPIN 1"));
}

#[test]
fn gpu_mesh_native_packed_warp_reads_and_exports_upper_lanes_in_a_branch() {
    if !gpu_ready() || !mesh_ready() || mesh_native_ready() != Some(true) {
        return;
    }
    let (mut code, ms, mut sg, mem) = ngg_triangle_per_group(false);
    let barrier = encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap();
    let at = code.windows(barrier.len()).position(|w| w == barrier).unwrap() + barrier.len();
    let insert = prog(&[
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 126)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 160), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 3), ("ADDR", 1), ("DATA0", 3)], None),
        ("V_CMPX_LE_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 3)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 3)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 12)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 6)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 0), ("SRC0", 255), ("VSRC1", 0)], Some(0xFFFF_FFE0)),
    ]);
    code.splice(at..at, insert[..insert.len() - 4].iter().copied());
    let mut replace = |old: Vec<u8>, new: Vec<u8>| {
        assert_eq!(old.len(), new.len());
        let at = code.windows(old.len()).position(|w| w == old).unwrap();
        code[at..at + old.len()].copy_from_slice(&new);
    };
    let mov = |dst, imm| encode("S_MOV_B32", "ENC_SOP1", &[("SDST", dst), ("SSRC0", 255)], Some(imm)).unwrap();
    replace(mov(124, 1 << 12 | 3), mov(124, 33 << 12 | 35));
    replace(
        encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)).unwrap(),
        encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 255)], Some(32 | 33 << 10 | 34 << 20)).unwrap(),
    );
    replace(
        encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 20)], None).unwrap(),
        encode("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 11)], None).unwrap(),
    );
    sg[super::IO_TRIP_AT] = 0x40;
    let t = translate(&code, &Options { mode: Mode::Mesh(ms), wave64: true, split_wave64: true, mesh_native: Some(super::MeshNative { ps_inputs: None, lds_bytes: 0 }), ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("GroupMemoryBarrier") && !t.hlsl.contains("xcl_x_meet("));
    if let Ok(file) = std::env::var("XCL_PACKED_MESH_HLSL") {
        std::fs::write(file, &t.hlsl).unwrap();
    }
    let dxil = super::dxc::compile(&t.hlsl, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    let ps = super::dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let (px, flat) = runner::mesh_draw_native_flat(&dxil, &ps, 1, &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert_eq!(flat[16..20], [0; 4]);
    assert_eq!((px[29 * 32 + 3][0], px[29 * 32 + 3][3]), (32.0, 1.0));
}

#[test]
fn gpu_mesh_native_packed_lds_orders_writes_and_cross_lane_reads() {
    if !gpu_ready() || !mesh_ready() || mesh_native_ready() != Some(true) {
        return;
    }
    let (mut code, ms, mut sg, mem) = ngg_triangle_per_group(false);
    let barrier = encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap();
    let at = code.windows(barrier.len()).position(|w| w == barrier).unwrap() + barrier.len();
    let insert = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 161), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 3), ("ADDR", 2)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 3)], None),
    ]);
    code.splice(at..at, insert[..insert.len() - 4].iter().copied());
    let color = encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 20)], None).unwrap();
    let at = code.windows(color.len()).position(|w| w == color).unwrap();
    let from_peer = encode("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 11)], None).unwrap();
    code[at..at + color.len()].copy_from_slice(&from_peer);
    sg[super::IO_TRIP_AT] = 0x40;
    let t = translate(&code, &Options { mode: Mode::Mesh(ms), wave64: true, split_wave64: true, mesh_native: Some(super::MeshNative { ps_inputs: None, lds_bytes: 256 }), ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert_eq!(t.hlsl.matches("GroupMemoryBarrierWithGroupSync();").count(), 2);
    assert!(!t.hlsl.contains("xcl_x_meet("));
    let dxil = super::dxc::compile(&t.hlsl, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    let ps = super::dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let (px, flat) = runner::mesh_draw_native_flat(&dxil, &ps, 1, &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert_eq!(flat[16..20], [0; 4]);
    assert_eq!((px[29 * 32 + 3][0], px[29 * 32 + 3][3]), (33.0, 1.0));
}

#[test]
fn native_mesh_refuses_lds_in_a_divergent_site() {
    let (_, ms, _, _) = ngg_triangle_per_group(false);
    let code = prog(&[
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None),
    ]);
    let err = translate(&code, &Options { mode: Mode::Mesh(ms), wave64: true, split_wave64: true, mesh_native: Some(super::MeshNative { ps_inputs: None, lds_bytes: 256 }), ..Default::default() }).unwrap_err();
    assert!(err.contains("LDS accesses or barriers are in divergent control flow"), "{err}");
}

/// A console mesh program as a host mesh shader (`MeshNative`): one DispatchMesh of the
/// title's groups draws what the two passes draw – the exports are the thread's own vertex and
/// primitive, GS_ALLOC_REQ the output counts.
#[test]
fn gpu_mesh_mode_native_draws_what_two_passes_draw() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    let Some(split) = mesh_native_ready() else { return };
    use super::dxc;
    let (code, ms, sg, mem) = ngg_triangle_per_group(false);
    let ps = dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let px = runner::mesh_draw_native(&native_mesh(&code, &ms, split, None), &ps, runner::MeshGroups::Direct(2), &sg, &mem, 32, [-1.0; 4]).unwrap();
    let at = |x: usize, y: usize| px[y * 32 + x];
    assert_eq!((at(3, 29)[0], at(3, 29)[3]), (0.0, 1.0), "group 0 triangle (left)");
    assert_eq!((at(19, 29)[0], at(19, 29)[3]), (1.0, 1.0), "group 1 triangle (right)");
    assert_eq!(at(29, 3), [-1.0; 4], "outside both triangles");
    // the same program in two passes
    let t = translate(&code, &on_gpu(Options { mode: Mode::Mesh(ms.clone()), wave64: true, ..Default::default() })).unwrap();
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let pass = dxc::compile(&super::mesh_passthrough(&ms), "main", "ms_6_6", &[]).unwrap();
    let (px2, _) = runner::mesh_draw(&cs, &pass, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert!(px == px2, "the host mesh shader and the two passes draw the same pixels");

    // vertices that carry a pixel shader's inputs (as the pipeline makes them): an input the
    // program does not export reads 0
    let input = |k: u32| crate::gpu::dxbc::SigElement { name: "TEXCOORD".into(), index: k, system_value: 0, component_type: 3, register: k + 1, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let ps2 = dxc::compile("struct V { float4 pos : SV_Position; float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1; };\nfloat4 main(V v) : SV_Target0 { return v.t0 + v.t1; }\n", "main", "ps_6_6", &[]).unwrap();
    let linked = native_mesh(&code, &ms, split, Some(vec![input(0), input(1)]));
    let px3 = runner::mesh_draw_native(&linked, &ps2, runner::MeshGroups::Direct(2), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert!(px3 == px, "linked to a pixel shader's inputs");
}

/// What a host mesh shader cannot hold is refused at translation, so the pipeline is made in two
/// passes: vertices over D3D12's 32 KB of mesh outputs (32 pixel shader inputs, 64 vertices).
#[test]
fn mesh_native_refuses_what_the_host_mesh_stage_cannot_hold() {
    let (code, ms, _, _) = ngg_triangle_per_group(false);
    let inputs = |n: u32| -> Vec<crate::gpu::dxbc::SigElement> {
        (0..n).map(|k| crate::gpu::dxbc::SigElement { name: "TEXCOORD".into(), index: k, system_value: 0, component_type: 3, register: k + 1, mask: 0xF, rw_mask: 0xF, stream: 0 }).collect()
    };
    let native = |ms: &super::MsInterface, n: u32| translate(&code, &Options { mode: Mode::Mesh(ms.clone()), wave64: true, split_wave64: true, mesh_native: Some(super::MeshNative { ps_inputs: Some(inputs(n)), lds_bytes: 0 }), ..Default::default() });
    let t = native(&ms, 31).unwrap();
    assert!(t.hlsl.contains("SetMeshOutputCounts(") && t.hlsl.contains("out vertices XclVert verts[64]"), "a mesh shader with no record");
    assert!(native(&ms, 32).unwrap_err().contains("32 KB"));
    let wide = super::MsInterface { threads: 256, max_vertices: 256, max_primitives: 256, ..ms.clone() };
    assert!(native(&wide, 1).unwrap_err().contains("more than one console wave"));
    // only an adapter whose waves are 32 lanes and nothing else runs the form
    assert_eq!(super::pso::mesh_native_waves(32, 32), Some(true));
    for (min, max) in [(32, 64), (8, 32), (4, 4), (64, 64)] {
        assert_eq!(super::pso::mesh_native_waves(min, max), None, "{min}..{max}");
    }
}

/// A null primitive (bit 31 of the primitive export) draws nothing, as a host mesh shader and in
/// two passes: group 1 culls its triangle, group 0 draws.
#[test]
fn gpu_mesh_mode_culls_null_primitives() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::dxc;
    let (code, ms, sg, mem) = ngg_triangle_per_group(true);
    let ps = dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let check = |px: &[[f32; 4]], form: &str| {
        let at = |x: usize, y: usize| px[y * 32 + x];
        assert_eq!((at(3, 29)[0], at(3, 29)[3]), (0.0, 1.0), "{form}: group 0 triangle (left)");
        assert_eq!(at(19, 29), [-1.0; 4], "{form}: group 1's primitive is null");
        assert!(px[32 * 16..].iter().all(|p| p[0] != 1.0), "{form}: nothing of group 1");
    };
    let t = translate(&code, &on_gpu(Options { mode: Mode::Mesh(ms.clone()), wave64: true, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let pass = dxc::compile(&super::mesh_passthrough(&ms), "main", "ms_6_6", &[]).unwrap();
    let (px, rec) = runner::mesh_draw(&cs, &pass, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    let prim = |g: usize| rec[g * ms.record_dwords() + super::MESH_HEADER + ms.max_vertices as usize * ms.vert_dwords()];
    assert_eq!((prim(0), prim(1)), (1 << 10 | 2 << 20, 1 << 31 | 1 << 10 | 2 << 20));
    check(&px, "two passes");
    let Some(split) = mesh_native_ready() else { return };
    let px = runner::mesh_draw_native(&native_mesh(&code, &ms, split, None), &ps, runner::MeshGroups::Direct(2), &sg, &mem, 32, [-1.0; 4]).unwrap();
    check(&px, "host mesh shader");
}

/// An indirect mesh call of a host mesh pipeline takes its group counts and its command count
/// from GPU memory: commands past the count do not run.
#[test]
fn gpu_mesh_mode_native_indirect_takes_the_gpu_count() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    let Some(split) = mesh_native_ready() else { return };
    let (code, ms, sg, mem) = ngg_triangle_per_group(false);
    let ps = super::dxc::compile(MESH_TEST_PS, "main", "ps_6_6", &[]).unwrap();
    let msb = native_mesh(&code, &ms, split, None);
    // command 0: group 0 alone (the left triangle); command 1: both groups
    let records = [[1, 1, 1], [2, 1, 1]];
    let draw = |count: u32| runner::mesh_draw_native(&msb, &ps, runner::MeshGroups::Indirect { records: &records, count, max: 2 }, &sg, &mem, 32, [-1.0; 4]).unwrap();
    let one = draw(1);
    assert_eq!((one[29 * 32 + 3][0], one[29 * 32 + 3][3]), (0.0, 1.0), "command 0: group 0");
    assert_eq!(one[29 * 32 + 19], [-1.0; 4], "command 1 is past the GPU's count");
    let two = draw(2);
    assert_eq!((two[29 * 32 + 3][0], two[29 * 32 + 19][0]), (0.0, 1.0), "both commands");
    let none = draw(0);
    assert!(none.iter().all(|p| *p == [-1.0; 4]), "a count of 0 draws nothing");
}

/// XCL_SAMPLE_MESH=<directory of console MS containers (*_ms.bin, *_as.bin)>: each as a host
/// mesh shader for 32-lane waves – translated, and compiled by DXC. Prints what refuses.
#[test]
#[ignore]
fn mesh_containers_compile_as_host_mesh_shaders() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(dir) = std::env::var_os("XCL_SAMPLE_MESH") else { return };
    let every: usize = std::env::var("XCL_SAMPLE_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| { let n = n.to_string_lossy(); n.ends_with("_ms.bin") || n.ends_with("_as.bin") }))
        .collect();
    files.sort();
    let files: Vec<_> = files.into_iter().step_by(every.max(1)).collect();
    let next = AtomicUsize::new(0);
    let (n, ok) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let why: std::sync::Mutex<BTreeMap<String, (usize, String)>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..super::dxc::compile_threads() {
            s.spawn(|| loop {
                let Some(path) = files.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let blob = std::fs::read(path).unwrap();
                if !crate::gpu::xbox::parse(&blob).is_some_and(|sh| sh.stage == "MS") {
                    continue;
                }
                n.fetch_add(1, Ordering::Relaxed);
                match super::pso::console_mesh_native(&blob, None, true) {
                    Ok(_) => {
                        ok.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        let first = e.lines().find(|l| l.contains("error")).unwrap_or(e.lines().next().unwrap_or("")).to_string();
                        let mut w = why.lock().unwrap();
                        let slot = w.entry(first).or_insert_with(|| (0, path.file_name().unwrap().to_string_lossy().into_owned()));
                        slot.0 += 1;
                    }
                }
            });
        }
    });
    let (n, ok) = (n.into_inner(), ok.into_inner());
    println!("MS containers as host mesh shaders: {ok} / {n} translated and compiled");
    for (e, (count, example)) in why.into_inner().unwrap() {
        println!("  {count} x {e} (e.g. {example})");
    }
}

/// A wave32 pixel program under `s_wqm` with the march loop of a retail volumetric PS after
/// another loop: every lane counts steps in a vector register, and the loop goes on as long as
/// a lane of the mask taken from EXEC has not reached 6 (`v_cmp_le_f32`, `s_andn2 mask, mask,
/// vcc`, `s_cbranch_scc1`). Exports (passes of the march, the lane's steps, 1, 1).
fn mask_loop_ps() -> super::Translation {
    use super::PsInterface;
    let before = [
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 2)], None),
        ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 9), ("SSRC0", 9), ("SSRC1", 129)], None),
        ins("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 130)], None),
    ]
    .concat();
    let march = [
        ins("V_ADD_F32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 242), ("VSRC1", 9)], None),
        ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None),
        ins("V_CMP_LE_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 9)], Some(0x40C0_0000)),
        ins("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 106)], None),
    ]
    .concat();
    let back = |body: &[u8]| (-((body.len() as i64 + 4) / 4)) as u16 as u64;
    let mut code = [
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 126)], None),
        ins("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None),
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 128)], None),
    ]
    .concat();
    code.extend(&before);
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back(&before))], None));
    code.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 126)], None));
    code.extend(&march);
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back(&march))], None));
    code.extend(
        [
            ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None),
            ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 8)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 242)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 9), ("VSRC2", 11), ("VSRC3", 11), ("DONE", 1), ("VM", 1)], None),
            ins("S_ENDPGM", "ENC_SOPP", &[], None),
        ]
        .concat(),
    );
    let psi = PsInterface { inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x302, attrs: None };
    translate(&code, &Options { mode: Mode::Pixel(psi), wave64: false, ..Default::default() }).unwrap()
}

/// Draws `mask_loop_ps` on two triangles sharing a diagonal (helper lanes along it): the march
/// ends after its 6 passes in every pixel. A helper lane's steps cannot be read from the second
/// pass on with an RDNA2 driver, its bit stayed in the mask and the loop ran to the loop budget
/// (SCC must count only live lanes).
#[test]
fn gpu_ps_mask_loop_on_a_primitive_edge_ends() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs = super::dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
    let pt = mask_loop_ps();
    assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
    assert!(pt.hlsl.contains("& ballot(true).x) != 0u);"), "{}", pt.hlsl);
    // the fourth channel: the hardware wave's lane count
    let hlsl = pt.hlsl.replacen("    if (!xcl_live) discard;", "    xo.t0[3] = (float)WaveGetLaneCount();\n    if (!xcl_live) discard;", 1);
    let ps = super::dxc::compile(&hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
    let size = 32usize;
    let px = runner::draw(&vs, &ps, &layout, &quad, 8, &[], &[], size as u32, [9.0; 4]).unwrap();
    let lanes: std::collections::BTreeSet<u32> = px.iter().map(|p| p[3] as u32).collect();
    let bad: Vec<(usize, usize, [f32; 4])> = (0..size * size).filter(|&i| px[i][0] != 6.0 || px[i][1] != 6.0).map(|i| (i % size, i / size, px[i])).collect();
    println!("{:?}: lane counts {lanes:?}; {} of {} pixels wrong, first {:?}", runner::adapter(), bad.len(), size * size, bad.iter().take(8).collect::<Vec<_>>());
    assert!(bad.is_empty(), "{} pixels wrong (passes, steps, 1, lane count), first {:?}", bad.len(), &bad[..bad.len().min(8)]);
}

/// A program outside a title run, for what a driver does with it (a device removed at once, as
/// with the outlined batch-cull programs of play-102). `XCL_REPRO=<file or folder>` names dumped
/// console compute shaders (`*_cs.bin`: built the way a run builds them, so the translator's
/// switches apply), HLSL (`*.hlsl`: a cs_6_6 program with the entry `main`, through
/// `dxc::compile`) and DXIL (`*.dxil`: as it is, e.g. a file of the DXIL cache). Each becomes a
/// pipeline on the test device and is dispatched once (`XCL_REPRO_GROUPS=x,y,z`, default 1,1,1)
/// over stand-in state with no title data, in a process of its own (`XCL_REPRO_ONE`): a device one program loses is not the next one's. One
/// `repro:` line an input; a process that gives none within `XCL_REPRO_TIMEOUT` seconds
/// (default 600) is ended. `XCL_DXIL_CACHE` must name a scratch folder, so the emulator's cache
/// is never written. The state (`XCL_REPRO_STATE`): "soup", the default, makes every pair of words
/// in the user registers and in 256 KiB of window 0 read as a pointer into that window, as a
/// count and as a buffer's base, so a title program gets past its first tests and its loads
/// stay in the window; "zero" leaves both zeroed, and most programs leave at their first count.
/// `XCL_REPRO_SAVE=<folder>` keeps each program's DXIL there as `<input's name>.dxil`.
/// Without `XCL_REPRO` the test does nothing.
#[test]
fn gpu_repro_programs() {
    use std::io::Write;
    if let Some(one) = std::env::var_os("XCL_REPRO_ONE") {
        let path = std::path::PathBuf::from(one);
        println!("repro: {}: {}", path.file_name().unwrap_or_default().to_string_lossy(), repro_one(&path));
        let _ = std::io::stdout().flush();
        return;
    }
    let Some(input) = std::env::var_os("XCL_REPRO") else {
        return;
    };
    assert!(std::env::var_os("XCL_DXIL_CACHE").is_some(), "set XCL_DXIL_CACHE to a scratch folder: this test must not write the emulator's DXIL cache");
    let input = std::path::PathBuf::from(input);
    let mut files: Vec<std::path::PathBuf> = if input.is_dir() {
        std::fs::read_dir(&input).unwrap().flatten().map(|e| e.path()).filter(|p| ["bin", "hlsl", "dxil"].iter().any(|x| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(x)))).collect()
    } else {
        vec![input.clone()]
    };
    files.sort();
    assert!(!files.is_empty(), "no *.bin, *.hlsl or *.dxil in {input:?}");
    println!("repro: {} programs, {:?} groups each", files.len(), repro_groups());
    for f in &files {
        println!("{}", repro_child(f, &[]));
        let _ = std::io::stdout().flush();
    }
}

/// One input of `gpu_repro_programs` in a process of its own, with `env` added to this one's: its
/// `repro:` line. A process that gives none within `XCL_REPRO_TIMEOUT` seconds is ended.
fn repro_child(f: &std::path::Path, env: &[(&str, &str)]) -> String {
    use std::io::Read;
    let limit = std::time::Duration::from_secs(std::env::var("XCL_REPRO_TIMEOUT").ok().and_then(|v| v.parse().ok()).unwrap_or(600));
    let name = f.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let t0 = std::time::Instant::now();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["gpu::recomp::tests::gpu_repro_programs", "--exact", "--nocapture", "--test-threads=1"])
        .env("XCL_REPRO_ONE", f)
        .envs(env.iter().copied())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = pipe.read_to_string(&mut s);
        s
    });
    let status = loop {
        match child.try_wait().unwrap() {
            Some(s) => break Some(s),
            None if t0.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    };
    let out = reader.join().unwrap_or_default();
    // the harness prints "test <name> ... " in front of the child's line unless something ended that line first
    match (out.lines().find_map(|l| l.find("repro: ").map(|at| &l[at..])), status) {
        (Some(line), _) => line.to_string(),
        (None, Some(s)) => format!("repro: {name}: NO RESULT, the process ended with {s} after {:.1} s", t0.elapsed().as_secs_f64()),
        (None, None) => format!("repro: {name}: NO RESULT in {:.0} s, the process was ended", t0.elapsed().as_secs_f64()),
    }
}

/// Dumped console compute shaders (`XCL_OUTLINE_GATE_PROGRAMS=<file or folder of *_cs.bin>`, e.g.
/// the title's big batch-cull programs) built in every form of outlining and dispatched once each
/// the way `gpu_repro_programs` does it (its groups, state, time limit and scratch `XCL_DXIL_CACHE`),
/// a process a program and form: one `outline gate:` line each, and the form of this run has to
/// finish on every program. `XCL_OUTLINE_GATE_FORMS` keeps it to the forms named. A program below
/// `XCL_RECOMP_OUTLINE_MIN_INSTS` is plain in every form. Without the variable the test does nothing.
#[test]
fn gpu_outlined_title_programs_finish() {
    use std::io::Write;
    let Some(input) = std::env::var_os("XCL_OUTLINE_GATE_PROGRAMS") else {
        return;
    };
    assert!(std::env::var_os("XCL_DXIL_CACHE").is_some(), "set XCL_DXIL_CACHE to a scratch folder: this test must not write the emulator's DXIL cache");
    let input = std::path::PathBuf::from(input);
    let mut files: Vec<std::path::PathBuf> = if input.is_dir() { std::fs::read_dir(&input).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("bin"))).collect() } else { vec![input.clone()] };
    files.sort();
    assert!(!files.is_empty(), "no *.bin in {input:?}");
    let only: Option<Vec<String>> = std::env::var("XCL_OUTLINE_GATE_FORMS").ok().map(|v| v.split(',').map(|f| f.trim().to_string()).collect());
    let forms = outline_forms();
    let mut lost = Vec::new();
    for f in &files {
        for form in forms.iter().filter(|f| only.as_ref().is_none_or(|o| o.contains(&outline_form_name(**f)))) {
            let name = outline_form_name(*form);
            let part: Vec<&str> = name.split('/').collect();
            let line = repro_child(f, &[("XCL_RECOMP_OUTLINE_RESULTS", part[0]), ("XCL_RECOMP_OUTLINE_ATOMICS", part[1]), ("XCL_RECOMP_OUTLINE_NESTED", part.get(2).copied().unwrap_or("keep"))]);
            println!("outline gate: {name}: {}", line.trim_start_matches("repro: "));
            let _ = std::io::stdout().flush();
            if *form == forms[0] && !line.contains("; FINISHED ") && !line.contains("; no GPU, not dispatched") {
                lost.push(line);
            }
        }
    }
    assert!(lost.is_empty(), "the form of this run, {}, did not finish:\n{}", outline_form_name(forms[0]), lost.join("\n"));
}

fn repro_groups() -> [u32; 3] {
    let v: Vec<u32> = std::env::var("XCL_REPRO_GROUPS").unwrap_or_default().split(',').filter_map(|x| x.trim().parse().ok()).collect();
    [v.first().copied().unwrap_or(1).max(1), v.get(1).copied().unwrap_or(1).max(1), v.get(2).copied().unwrap_or(1).max(1)]
}

/// One input of `gpu_repro_programs`: what became of it, in one line.
fn repro_one(path: &std::path::Path) -> String {
    let t0 = std::time::Instant::now();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return format!("NOT READ: {e}"),
    };
    let is = |x: &str| path.extension().is_some_and(|e| e.eq_ignore_ascii_case(x));
    let dxil = if is("dxil") {
        Ok(bytes)
    } else if is("hlsl") {
        String::from_utf8(bytes).map_err(|e| e.to_string()).and_then(|s| super::dxc::compile(&s, "main", "cs_6_6", &[]))
    } else {
        super::pso::console_to_dxil_in(&bytes, super::pso::Stage::Compute, 0, true, 4).map(|c| c.dxil)
    };
    let dxil = match dxil {
        Ok(d) => d,
        Err(e) => return format!("NOT BUILT in {:.1} s: {}", t0.elapsed().as_secs_f64(), e.lines().next().unwrap_or_default()),
    };
    let built = format!("{} B of DXIL in {:.1} s", dxil.len(), t0.elapsed().as_secs_f64());
    if let Some(dir) = std::env::var_os("XCL_REPRO_SAVE") {
        let _ = std::fs::write(std::path::Path::new(&dir).join(path.file_name().unwrap_or_default()).with_extension("dxil"), &dxil);
    }
    if !gpu_ready() {
        return format!("{built}; no GPU, not dispatched");
    }
    let soup = std::env::var("XCL_REPRO_STATE").map_or(true, |v| v.trim() != "zero");
    let word = |k: usize| if !soup { 0 } else if k % 2 == 0 { 0x1000 } else { CHAIN_WINDOW };
    let mut io: Vec<u32> = (0..128).map(|k| if k < 32 { word(k) } else { 0 }).collect();
    io[super::IO_UMA_BASE_HI] = CHAIN_WINDOW;
    let mem: Vec<u32> = (0..1 << 16).map(word).collect();
    let (groups, t1) = (repro_groups(), std::time::Instant::now());
    match runner::dispatch(&dxil, groups, &io, &mem) {
        Ok(after) => format!("{built}; FINISHED {groups:?} groups in {:.2} s on {:?} ({} memory words changed)", t1.elapsed().as_secs_f64(), runner::adapter().unwrap_or_default(), after.iter().zip(&mem).filter(|(a, b)| a != b).count()),
        Err(e) => format!("{built}; FAILED {groups:?} groups after {:.2} s: {}", t1.elapsed().as_secs_f64(), e.replace('\n', " | ")),
    }
}
