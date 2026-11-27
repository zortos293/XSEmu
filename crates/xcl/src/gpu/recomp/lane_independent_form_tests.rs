//! The lane-independent form (`Options::lane_independent`) computes what the wave forms compute:
//! every kind of program the classifier lets through, as the form and as the 64-lane and the
//! packed form, run by the DXIL interpreter over thread groups with whole and partial console
//! waves. And for the rules a wrong answer can be shown for: the form forced onto a program
//! the classifier refuses computes something else. No GPU and no DXC needed.

use xcl_dxil::interp::{self, Config, Verdict};

use super::lane_independent::{classify, Rule, ASSUME};
use super::lane_independent_tests::{asm, i, if_then, lane_loop, sop1, sop2, sopc, vadd, vbelow, vless_to, vmov, A, EXEC, ONES, V, VCC, ZERO};
use super::{translate, CsInterface, Mode, Options, Translation};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Form {
    /// `[WaveSize(64)]`: a host wave is the console's
    Wave64,
    Packed,
    Split,
    Lane,
}

fn interface(threads: [u32; 3]) -> CsInterface {
    CsInterface { threads, user_sgprs: 8, tgid: [true; 3], tg_size: true, tid_components: 3 }
}

fn translation(code: &[u8], threads: [u32; 3], form: Form) -> Translation {
    translation_in(code, threads, form, 1)
}

/// `translation` for a title with `windows` memory windows.
fn translation_in(code: &[u8], threads: [u32; 3], form: Form, windows: u32) -> Translation {
    let opt = Options { mode: Mode::Compute(interface(threads)), wave64: true, split_wave64: form != Form::Wave64, legacy_compute_wave64: form == Form::Split, mem_windows: windows, ..Default::default() };
    super::LANE_INDEPENDENT.set(form == Form::Lane);
    let t = translate(code, &opt);
    super::LANE_INDEPENDENT.set(false);
    let t = t.unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t
}

fn load(hlsl: &str) -> interp::Program {
    interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap()
}

fn literal(d: u64, value: u32) -> A {
    A::I("S_MOV_B32", "ENC_SOP1", vec![("SDST", d), ("SSRC0", 255)], Some(value))
}

/// The start of every program here. The buffers: s[4:7] the results, 16 bytes a thread from
/// 1 MiB on, s[12:15] the inputs, 16 bytes a thread from 0 (never written, so whatever the seed
/// makes of them). v0 = the thread's index in its group (the console's order: x, then y), v1 =
/// its first input word, 0..63, v2 = its second.
fn head(threads: [u32; 3]) -> Vec<A> {
    let mut p = vec![literal(4, 0x10_0000), literal(5, 16 << 16), literal(6, 0x7FFF_FFFF), literal(7, 0)];
    p.extend([literal(12, 0), literal(13, 16 << 16), literal(14, 0x7FFF_FFFF), literal(15, 0)]);
    if threads[1] > 1 {
        assert!(threads[0].is_power_of_two() && threads[2] == 1);
        p.push(i("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 128 + threads[0].trailing_zeros() as u64), ("VSRC1", 1)]));
        p.push(vadd(0, 0, 1));
    }
    p.push(i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)]));
    p.push(i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 4), ("IDXEN", 1)]));
    p.push(i("V_AND_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 191), ("VSRC1", 1)]));
    p
}

/// result word k of the thread = v[data]
fn out(k: u64, data: u64) -> A {
    i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", data), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("OFFSET", 4 * k), ("IDXEN", 1)])
}

const GROUPS: [[u32; 3]; 5] = [[64, 1, 1], [40, 1, 1], [96, 1, 1], [8, 8, 1], [8, 12, 1]];

fn config() -> Config {
    // a forced ballot would be the wave forms' alone: the form has none
    Config { groups: 2, force_ballots: false, ..Config::default() }
}

/// Every buffer word the program has written when two groups have run, from the inputs of
/// `seed`. (Not the atomics: the split form and the 64-lane form of a 2-D group count their
/// hardware waves with some, the programs here have none.)
fn written(p: &interp::Program, seed: u64) -> Vec<interp::res::Written> {
    let run = interp::run(p, seed, &config());
    assert!(run.stop.is_none(), "seed {seed}: {:?}", run.stop);
    run.written
}

/// The program qualifies, its form reads no wave, and it writes what the 64-lane form writes
/// (and the packed and the split form), whatever the inputs. Returns the form's HLSL for the
/// group of 64.
fn same(body: &[A]) -> String {
    let mut first = String::new();
    for threads in GROUPS {
        let code = asm(&[head(threads), body.to_vec()].concat());
        let lane = translation(&code, threads, Form::Lane);
        assert!(lane.lane_independent, "{threads:?}: not taken");
        let main = lane.hlsl.split_once("void main(").unwrap().1;
        for wave in ["Wave", "groupshared", "GroupMemoryBarrier", "xcl_x_", "rla("] {
            assert!(!lane.hlsl.contains(wave), "{threads:?}: {wave} in the form\n{main}");
        }
        assert!(lane.hlsl.contains(&format!("\n[numthreads({}, {}, {})]\nvoid main(", threads[0], threads[1], threads[2])), "{main}");
        let program = load(&lane.hlsl);
        let runs: Vec<_> = (0..6).map(|seed| written(&program, seed)).collect();
        assert!(runs.iter().all(|r| !r.is_empty()), "{threads:?}: nothing written");
        let forms: &[Form] = if threads == [64, 1, 1] || threads == [8, 12, 1] { &[Form::Wave64, Form::Packed, Form::Split] } else { &[Form::Wave64, Form::Packed] };
        for &form in forms {
            let other = translation(&code, threads, form);
            assert!(!other.lane_independent && other.hlsl.contains("[WaveSize("));
            let other = load(&other.hlsl);
            for (seed, run) in runs.iter().enumerate() {
                assert!(written(&other, seed as u64) == *run, "{threads:?}, {form:?}, seed {seed}: {:?}\n{main}", interp::compare(&interp::run(&other, seed as u64, &config()), &interp::run(&program, seed as u64, &config())));
            }
        }
        if threads == [64, 1, 1] {
            first = lane.hlsl;
        }
    }
    first
}

// ---------------------------------------------------------------- what the form computes

#[test]
fn straight_line_code_with_the_thread_ids_and_the_wave_number() {
    // the group's ids and TG_SIZE (waves in the group, the thread's wave above them) in
    // s8..s11, the thread's own ids in v0..v2 before the head folds them
    let mut p = vec![vmov(20, 8), vmov(21, 9), vmov(22, 11), vadd(23, 0, 2)];
    p.extend([out(0, 20), out(1, 22), out(2, 23), out(3, 1)]);
    let hlsl = same(&p);
    assert!(hlsl.contains("    lane_id = gi % 64u;\n") && hlsl.contains("s11 = 1u | ((gi / 64u) << 6);"), "{hlsl}");
}

#[test]
fn a_compare_into_saveexec_and_a_branch() {
    // if (v0 < v1) v3 = v0 + v2
    let p = [vmov(3, 2), vless_to(VCC, 0, 1), sop1("S_AND_SAVEEXEC_B64", 10, VCC), A::B("S_CBRANCH_EXECZ", "join"), vadd(3, 0, 2), A::L("join"), sop2("S_OR_B64", EXEC, EXEC, 10), out(0, 3)];
    let hlsl = same(&p);
    assert!(hlsl.contains("uint2 ballot(bool b) { uint m = select(b, 0xFFFFFFFFu, 0u); return uint2(m, m); }"));
    // the stage 1 shape, on the thread's index alone
    let (head, tail) = if_then();
    same(&[head, tail, vec![out(1, 3)]].concat());
}

#[test]
fn if_else() {
    let p = [
        vless_to(VCC, 1, 0),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        sop2("S_XOR_B64", 10, EXEC, 10),
        A::B("S_CBRANCH_EXECZ", "else"),
        vadd(3, 0, 2),
        A::L("else"),
        sop1("S_OR_SAVEEXEC_B64", 16, 10),
        sop2("S_XOR_B64", EXEC, EXEC, 16),
        A::B("S_CBRANCH_EXECZ", "join"),
        vadd(3, 1, 1),
        A::L("join"),
        sop2("S_OR_B64", EXEC, EXEC, 16),
        out(0, 3),
    ];
    same(&p);
}

#[test]
fn nested_masks() {
    let p = [
        vmov(3, ZERO),
        vmov(4, ZERO),
        vbelow(1, 40),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        A::B("S_CBRANCH_EXECZ", "outer"),
        vadd(3, 0, 2),
        vless_to(VCC, 1, 0),
        sop1("S_AND_SAVEEXEC_B64", 16, VCC),
        A::B("S_CBRANCH_VCCZ", "inner"),
        vadd(3, 3, 1),
        A::L("inner"),
        sop2("S_OR_B64", EXEC, EXEC, 16),
        vadd(4, 3, 0),
        A::L("outer"),
        sop2("S_OR_B64", EXEC, EXEC, 10),
        out(0, 3),
        out(1, 4),
    ];
    same(&p);
}

#[test]
fn loops_with_a_lane_exit_mask() {
    // do { v3 += 1 } while (v3 < v0): every latch the compiler uses
    for latch in ["S_CBRANCH_EXECNZ", "S_CBRANCH_SCC1", "S_CBRANCH_VCCNZ"] {
        let mut p = lane_loop(latch);
        p.pop();
        p.push(out(0, 3));
        let hlsl = same(&p);
        assert!(hlsl.contains("if (xcl_budget == 0u) { xcl_trip(0u); break; } xcl_budget -= 1u;"), "{hlsl}");
    }
    // the count from the thread's input, the lanes that are done collected in s[14:15]
    let p = [
        vmov(3, ZERO),
        vmov(5, ZERO),
        sop1("S_MOV_B64", 14, ZERO),
        A::L("loop"),
        vadd(3, 3, 2),
        i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129), ("VSRC1", 5)]),
        i("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V + 5), ("VSRC1", 1)]),
        sop2("S_OR_B64", 14, VCC, 14),
        sop2("S_ANDN2_B64", EXEC, EXEC, 14),
        A::B("S_CBRANCH_EXECNZ", "loop"),
        sop2("S_OR_B64", EXEC, EXEC, 14),
        out(0, 3),
        out(1, 5),
    ];
    same(&p);
}

#[test]
fn a_counted_loop_around_masks() {
    let mut p = vec![vmov(3, ZERO), i("S_MOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 3)]), A::L("again")];
    p.extend([vless_to(VCC, 3, 1), sop1("S_AND_SAVEEXEC_B64", 10, VCC), A::B("S_CBRANCH_EXECZ", "join"), vadd(3, 3, 0), vadd(3, 3, 2), A::L("join"), sop2("S_OR_B64", EXEC, EXEC, 10)]);
    p.extend([sop2("S_ADD_I32", 20, 20, ONES), sopc("S_CMP_GT_I32", 20, ZERO), A::B("S_CBRANCH_SCC1", "again"), vmov(5, 20), out(0, 3), out(1, 5)]);
    same(&p);
}

#[test]
fn selects_and_carries_on_compare_results() {
    let p = [
        vless_to(VCC, 0, 1),
        i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 2)]),
        vless_to(16, 1, 2),
        i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", 16)]),
        i("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 5), ("SDST", VCC), ("SRC0", V + 2), ("SRC1", V + 2)]),
        i("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 1), ("VSRC1", 0)]),
        // masks combined: (v0 < v1) & ~(v1 < v2), and its complement
        vless_to(18, 0, 1),
        sop2("S_ANDN2_B64", 18, 18, 16),
        sop1("S_NOT_B64", 20, 18),
        i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", 20)]),
        out(0, 3),
        out(1, 4),
        out(2, 6),
        out(3, 7),
    ];
    same(&p);
}

#[test]
fn the_all_lanes_and_no_lanes_constants() {
    // every lane on for a stretch inside a lane's branch, EXEC none and all, a mask cleared
    let whole = [
        vmov(3, ZERO),
        vless_to(VCC, 0, 1),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        sop1("S_OR_SAVEEXEC_B64", 14, ONES),
        vmov(5, 131),
        sop1("S_MOV_B64", EXEC, 14),
        vadd(3, 0, 2),
        sop1("S_MOV_B64", EXEC, ZERO),
        vmov(3, 135),
        sop1("S_MOV_B64", EXEC, ONES),
        vadd(6, 3, 5),
        sop1("S_MOV_B64", EXEC, 10),
        out(0, 3),
        out(1, 6),
    ];
    same(&whole);
    // masks compared with -1 and with 0: the fix-up skipped when every lane passes, the work
    // when none asks
    let tests = [
        vmov(3, ZERO),
        vmov(4, ZERO),
        vless_to(16, 0, 1),
        sop1("S_MOV_B64", 14, EXEC),
        sopc("S_CMP_EQ_U64", 16, ONES),
        A::B("S_CBRANCH_SCC1", "skip"),
        sop2("S_ANDN2_B64", EXEC, EXEC, 16),
        vadd(3, 0, 2),
        A::L("skip"),
        sop1("S_MOV_B64", EXEC, 14),
        vless_to(18, 1, 2),
        sopc("S_CMP_LG_U64", 18, ZERO),
        A::B("S_CBRANCH_SCC0", "none"),
        sop1("S_AND_SAVEEXEC_B64", 14, 18),
        vadd(4, 1, 2),
        sop1("S_MOV_B64", EXEC, 14),
        A::L("none"),
        out(0, 3),
        out(1, 4),
    ];
    same(&tests);
}

#[test]
fn an_early_out() {
    let p = [vless_to(VCC, 0, 1), sop1("S_AND_SAVEEXEC_B64", 10, VCC), A::B("S_CBRANCH_EXECZ", "out"), vadd(3, 0, 2), out(0, 3), i("S_ENDPGM", "ENC_SOPP", &[]), A::L("out")];
    same(&p);
}

#[test]
fn first_lane_of_a_wave_uniform_value() {
    let read = |s: u64, v: u64| i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", s), ("SRC0", V + v)]);
    // from a user SGPR and a constant, read inside a lane's branch
    let p = [
        vmov(5, 2),
        vmov(6, 131),
        vadd(7, 5, 6),
        vmov(8, ZERO),
        vless_to(VCC, 0, 1),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        A::B("S_CBRANCH_EXECZ", "join"),
        read(20, 7),
        vmov(8, 20),
        A::L("join"),
        sop2("S_OR_B64", EXEC, EXEC, 10),
        out(0, 8),
    ];
    let hlsl = same(&p);
    assert!(!hlsl.contains("first_lane(exec_lo"), "{hlsl}");
}

#[test]
fn loads_and_stores_addressed_by_the_thread() {
    // a second buffer read at an index the thread computes from its input, and a 64-bit address
    let p = [
        vadd(5, 0, 1),
        i("BUFFER_LOAD_DWORDX2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 5), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("IDXEN", 1)]),
        i("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 132), ("VSRC1", 0)]),
        vmov(9, ZERO),
        i("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 10), ("ADDR", 8), ("SADDR", 0x7D), ("OFFSET", 12)]),
        vless_to(VCC, 6, 7),
        i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 6), ("VSRC1", 7)]),
        out(0, 6),
        out(1, 7),
        out(2, 10),
        out(3, 11),
    ];
    same(&p);
}

#[test]
fn a_divide_and_a_byte_permute() {
    let cvt = |d: u64, s: u64| i("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", d), ("SRC0", V + s)]);
    let p = [
        cvt(5, 1),
        vadd(6, 0, 1),
        cvt(6, 6),
        // v5 / v6 as the compiler divides: scale, reciprocal, multiply-add with the scale, fix up
        i("V_DIV_SCALE_F32", "VOP3_SDST_ENC", &[("VDST", 7), ("SDST", VCC), ("SRC0", V + 6), ("SRC1", V + 6), ("SRC2", V + 5)]),
        i("V_RCP_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 6)]),
        i("V_MUL_F32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 5), ("VSRC1", 8)]),
        i("V_DIV_FMAS_F32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 9), ("SRC1", V + 7), ("SRC2", V + 9)]),
        i("V_DIV_FIXUP_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 9), ("SRC1", V + 6), ("SRC2", V + 5)]),
        i("V_PERM_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 2), ("SRC1", V + 1)]),
        out(0, 9),
        out(1, 10),
        out(2, 11),
    ];
    same(&p);
}

// ---------------------------------------------------------------- the size report

/// One round of what compute programs are made of, on the thread's inputs and the round's
/// number: a branch on a compare with a nested one and a load at an index the lane computed, a
/// select, and a loop each lane leaves on its own count.
fn round(k: u64) -> Vec<A> {
    let label = |s: &str| -> &'static str { Box::leak(format!("{s}{k}").into_boxed_str()) };
    let (outer, inner, again) = (label("outer"), label("inner"), label("again"));
    vec![
        vmov(6, 128 + (k & 63)),
        vadd(7, 6, 1),
        vless_to(VCC, 0, 7),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        A::B("S_CBRANCH_EXECZ", outer),
        vadd(3, 3, 2),
        vless_to(VCC, 1, 6),
        sop1("S_AND_SAVEEXEC_B64", 16, VCC),
        A::B("S_CBRANCH_VCCZ", inner),
        vadd(4, 4, 0),
        A::L(inner),
        sop2("S_OR_B64", EXEC, EXEC, 16),
        i("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 3), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("IDXEN", 1)]),
        vadd(4, 4, 10),
        A::L(outer),
        sop2("S_OR_B64", EXEC, EXEC, 10),
        vless_to(18, 3, 4),
        i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", 18)]),
        vmov(8, ZERO),
        i("V_AND_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 1)]),
        sop1("S_MOV_B64", 20, EXEC),
        A::L(again),
        vadd(3, 3, 5),
        i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 129), ("VSRC1", 8)]),
        i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V + 8), ("VSRC1", 9)]),
        sop2("S_AND_B64", EXEC, EXEC, VCC),
        A::B("S_CBRANCH_EXECNZ", again),
        sop1("S_MOV_B64", EXEC, 20),
    ]
}

fn rounds(n: u64) -> Vec<A> {
    let mut p = vec![vmov(3, ZERO), vmov(4, ZERO)];
    p.extend((0..n).flat_map(round));
    p.extend([out(0, 3), out(1, 4), out(2, 5)]);
    p
}

/// The size report (`--nocapture` prints it): the DXIL of the direct backend
/// (`xcl_dxil::compile`) for programs of 1 to 128 rounds as the form, the packed and the split
/// form, in bytes, and the time the backend took, for a title with one memory window and with
/// four. The form is the smallest of the three, by
/// more the larger the program.
#[test]
fn the_form_is_the_smallest() {
    same(&rounds(3));
    let threads = [8, 8, 1];
    println!("lane-independent size report, group {threads:?}: memory windows, guest instructions; DXIL bytes (backend ms) lane-independent / packed / split; packed and split over lane-independent");
    for (n, windows) in [(1, 1), (4, 1), (16, 1), (64, 1), (128, 1), (16, 4), (64, 4), (128, 4)] {
        let code = asm(&[head(threads), rounds(n)].concat());
        let size = |form: Form| {
            let t = translation_in(&code, threads, form, windows);
            assert_eq!(t.lane_independent, form == Form::Lane);
            let start = std::time::Instant::now();
            let dxil = xcl_dxil::compile(&t.hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{form:?}: {e}"));
            (t.instructions, dxil.len(), start.elapsed().as_millis())
        };
        let (insts, lane, lane_ms) = size(Form::Lane);
        let ((_, packed, packed_ms), (_, split, split_ms)) = (size(Form::Packed), size(Form::Split));
        println!(
            "{n:>4} rounds, {windows} window(s), {insts:>5} instructions: {lane:>8} ({lane_ms} ms) / {packed:>8} ({packed_ms} ms) / {split:>8} ({split_ms} ms); {:.2}x, {:.2}x",
            packed as f64 / lane as f64,
            split as f64 / lane as f64
        );
        assert!(lane < packed && lane < split, "{n} rounds: {lane} / {packed} / {split}");
    }
}

// ---------------------------------------------------------------- the choice of the form

/// Off (`XCL_RECOMP_NO_LANE_INDEPENDENT`; tests: the thread has not asked): the packed form, as
/// if there were no other. On: only for a wave64 compute program on a GPU with 32-lane waves,
/// and not against the legacy override.
#[test]
fn the_form_is_taken_only_where_it_is_meant() {
    let (a, b) = if_then();
    let threads = [64, 1, 1];
    let code = asm(&[head(threads), a, b, vec![out(0, 3)]].concat());
    let cs = || Mode::Compute(interface(threads));
    let packed = translate(&code, &Options { mode: cs(), wave64: true, split_wave64: true, legacy_compute_wave64: false, ..Default::default() }).unwrap();
    assert!(!packed.lane_independent && packed.hlsl.contains("[WaveSize(32)]\n[numthreads(32, 1, 1)]"));
    assert_eq!(translation(&code, threads, Form::Packed).hlsl, packed.hlsl);
    super::LANE_INDEPENDENT.set(true);
    let taken = |opt: Options| translate(&code, &opt).unwrap().lane_independent;
    assert!(taken(Options { mode: cs(), wave64: true, split_wave64: true, legacy_compute_wave64: false, ..Default::default() }));
    // what a caller sets is not read
    assert!(!taken(Options { mode: cs(), wave64: true, split_wave64: false, lane_independent: true, ..Default::default() }));
    assert!(!taken(Options { mode: cs(), wave64: true, split_wave64: true, legacy_compute_wave64: true, ..Default::default() }));
    assert!(!taken(Options { mode: cs(), wave64: true, split_wave64: true, legacy_compute_wave64: false, loop_profile: true, ..Default::default() }));
    assert!(!taken(Options { mode: Mode::Kernel, wave64: true, split_wave64: true, lane_independent: true, ..Default::default() }));
    // a wave32 program has its own lanes on such a GPU
    let wave32 = translate(&code, &Options { mode: cs(), wave64: false, split_wave64: true, ..Default::default() });
    assert!(!wave32.is_ok_and(|t| t.lane_independent));
    super::LANE_INDEPENDENT.set(false);
}

// ---------------------------------------------------------------- what the rules are for

/// The classifier refuses the program for `rule`, so it keeps a wave form; and were it let
/// through, the form would write something else than the console's wave does.
fn refused(body: &[A], rule: Rule) {
    let threads = [64, 1, 1];
    let code = asm(&[head(threads), body.to_vec()].concat());
    let opt = Options { mode: Mode::Compute(interface(threads)), wave64: true, split_wave64: true, ..Default::default() };
    let reject = classify(&super::decode_program(&code, &opt).unwrap().insts, &opt).expect_err("the program must not qualify");
    assert_eq!(reject.rule, rule, "{reject}");
    let kept = translation(&code, threads, Form::Lane);
    assert!(!kept.lane_independent && kept.hlsl.contains("[WaveSize(32)]"));
    assert_eq!(kept.hlsl, translation(&code, threads, Form::Packed).hlsl);
    ASSUME.set(true);
    let forced = translation(&code, threads, Form::Lane);
    ASSUME.set(false);
    assert!(forced.lane_independent);
    let (verdict, seed) = interp::diff(&load(&translation(&code, threads, Form::Wave64).hlsl), &load(&forced.hlsl), 0..6, &config());
    assert!(matches!(verdict, Verdict::Different(_)), "seed {seed}: {verdict:?}\n{}", forced.hlsl.split_once("void main(").unwrap().1);
}

#[test]
fn a_lane_count_would_be_the_threads_own() {
    let p = [vless_to(VCC, 0, 1), sop1("S_BCNT1_I32_B64", 20, VCC), vmov(5, 20), out(0, 5)];
    refused(&p, Rule::MaskAsData);
}

#[test]
fn the_first_lane_of_a_lane_value_would_be_the_thread() {
    let p = [i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 20), ("SRC0", V + 1)]), vmov(5, 20), out(0, 5)];
    refused(&p, Rule::FirstLane);
}

/// A number in every lane swapped with the thread's input: the register read is the input now.
#[test]
fn the_first_lane_of_a_swapped_register_would_be_the_thread() {
    let p = [vmov(5, 2), i("V_SWAP_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 1)]), i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 20), ("SRC0", V + 5)]), vmov(6, 20), out(0, 6)];
    refused(&p, Rule::FirstLane);
}

#[test]
fn a_wave_wide_test_read_as_a_number_would_be_the_threads_own() {
    let p = [vless_to(16, 0, 1), sop2("S_AND_B64", 18, 16, EXEC), sop2("S_CSELECT_B32", 20, 129, ZERO), vmov(5, 20), out(0, 5)];
    refused(&p, Rule::MaskTestAsData);
}

#[test]
fn a_loop_all_lanes_run_while_any_asks_would_stop_early() {
    // do { v3 += v2 } while (v5++ < v1), every lane for as long as one asks
    let p = [vmov(3, ZERO), vmov(5, ZERO), A::L("loop"), vadd(3, 3, 2), i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129), ("VSRC1", 5)]), vless_to(VCC, 5, 1), A::B("S_CBRANCH_VCCNZ", "loop"), out(0, 3)];
    refused(&p, Rule::SoloVector);
}

#[test]
fn a_scalar_set_where_other_lanes_ask_would_be_missed() {
    let p = [
        i("S_MOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)]),
        vless_to(VCC, 0, 1),
        sop1("S_AND_SAVEEXEC_B64", 10, VCC),
        A::B("S_CBRANCH_EXECZ", "join"),
        i("S_MOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 7)]),
        A::L("join"),
        sop2("S_OR_B64", EXEC, EXEC, 10),
        vmov(5, 20),
        out(0, 5),
    ];
    refused(&p, Rule::SoloScalar);
}
