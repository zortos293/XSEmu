//! The compute prologue's thread limits (io[`IO_THREAD_LIMIT`], a `DispatchX` with thread
//! dimensions) against the same limits tested by the console program itself: a program that
//! stores its console thread's ids, translated with the limits in its launch EXEC, and the same
//! program behind an explicit `v_cmp_le_u32` / `s_and_saveexec` on each axis's limit (read from
//! the same io words as `ttmp11`..`ttmp13`), translated without them. Both are compiled by the
//! direct backend (`xcl_dxil`) and run by its interpreter over several groups, and must store
//! the same words (their group-shared atomics differ: the oracle's compares are more meets of
//! the split form's halves): no limit, a limit inside the first wave, on and next to a wave boundary, in a
//! later group, on each axis of 2-D and 3-D groups; for every wave64 form (native, packed,
//! split), and with a barrier, where the waves left with no thread must not hang. A wrong
//! compare and a store left unguarded must show.

use super::{translate, CsInterface, Mode, Options};
use crate::gpu::rdna2::encode;
use xcl_dxil::interp::{self, res, Config, Input};

type Inst = (&'static str, &'static str, Vec<(&'static str, u64)>);

const V: u64 = 256;

/// Groups each run dispatches (along x: the interpreter's groups are (g, 0, 0)).
const GROUPS: u32 = 3;

/// One translated form: wave64, split wave64, the legacy split form (not packed), the console
/// group shape and whether the program has an `s_barrier`.
#[derive(Clone, Copy, Debug)]
struct Form {
    wave64: bool,
    split: bool,
    legacy: bool,
    threads: [u32; 3],
    barrier: bool,
}

/// v5 = the console thread's ids, x | y << 16 | z << 24 (x across groups: s2 is the group id),
/// stored at its own address (v5 × 4); `oracle` first leaves the threads past each axis's limit
/// (`ttmp11` + k, a count of 0 passing every thread) out of EXEC, as the console never starts
/// them.
fn program(f: Form, oracle: bool) -> Vec<u8> {
    let [x, y, z] = f.threads;
    let mut v: Vec<Inst> = Vec::new();
    if f.barrier {
        v.push(("S_BARRIER", "ENC_SOPP", vec![]));
    }
    v.push(("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 4), ("SRC0", 2), ("SRC1", 128 + x.trailing_zeros() as u64), ("SRC2", V)]));
    if oracle {
        let saveexec = if f.wave64 { "S_AND_SAVEEXEC_B64" } else { "S_AND_SAVEEXEC_B32" };
        v.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 9), ("SRC0", 193)]));
        for (k, id) in [4u64, 1, 2].into_iter().enumerate() {
            v.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 8), ("SRC0", 119 + k as u64), ("VSRC1", 9)]));
            v.push(("V_CMP_LE_U32", "ENC_VOPC", vec![("SRC0", V + id), ("VSRC1", 8)]));
            v.push((saveexec, "ENC_SOP1", vec![("SDST", 20), ("SSRC0", 106)]));
        }
    }
    v.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 5), ("SRC0", V + 4)]));
    if y > 1 {
        v.push(("V_LSHL_OR_B32", "ENC_VOP3", vec![("VDST", 5), ("SRC0", V + 1), ("SRC1", 144), ("SRC2", V + 5)]));
    }
    if z > 1 {
        v.push(("V_LSHL_OR_B32", "ENC_VOP3", vec![("VDST", 5), ("SRC0", V + 2), ("SRC1", 152), ("SRC2", V + 5)]));
    }
    v.push(("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 130), ("VSRC1", 5)]));
    v.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 128)]));
    v.push(("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", vec![("ADDR", 6), ("DATA", 5), ("SADDR", 0x7D)]));
    v.push(("S_ENDPGM", "ENC_SOPP", vec![]));
    v.iter().flat_map(|(n, e, fl)| encode(n, e, fl, None).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

fn translate_cs(f: Form, oracle: bool) -> String {
    translation(f, oracle).hlsl
}

fn translation(f: Form, oracle: bool) -> super::Translation {
    let [_, y, z] = f.threads;
    let cs = CsInterface { threads: f.threads, user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: if z > 1 { 3 } else if y > 1 { 2 } else { 1 } };
    let opt = Options { mode: Mode::Compute(cs), wave64: f.wave64, split_wave64: f.split, legacy_compute_wave64: f.legacy, thread_limits: !oracle, ..Default::default() };
    let t = translate(&program(f, oracle), &opt).unwrap();
    assert!(t.complete(), "{f:?}: {:?}", t.unsupported_why);
    t
}

fn load(hlsl: &str) -> interp::Program {
    interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap()
}

/// Both programs run to their end on one input, with the limits `l` in io (None: what the seed
/// generates): whether they stored the same words, and how many the oracle stored.
fn run(limited: &interp::Program, oracle: &interp::Program, seed: u64, l: Option<[u32; 3]>) -> (bool, usize) {
    let patches = l.map_or(Vec::new(), |l| (0..3).map(|k| (((res::IDENT_IO as u64) << 32) | super::IO_THREAD_LIMIT[k] as u64, l[k])).collect());
    let input = Input { seed, patches };
    let cfg = Config { groups: GROUPS, force_ballots: false, ..Config::default() };
    let (a, b) = (interp::run_input(oracle, &input, &cfg, false).0, interp::run_input(limited, &input, &cfg, false).0);
    assert!(a.stop.is_none() && b.stop.is_none(), "stopped: {:?} / {:?}", a.stop, b.stop);
    (a.written == b.written, a.written.len())
}

/// The threads a dispatch of `GROUPS` groups of `threads` starts under the limits `l`.
fn started(threads: [u32; 3], l: [u32; 3]) -> usize {
    let span = [threads[0] * GROUPS, threads[1], threads[2]];
    (0..3).map(|k| if l[k] == 0 { span[k] } else { l[k].min(span[k]) } as usize).product()
}

fn forms() -> Vec<Form> {
    let f = |wave64, split, legacy, threads, barrier| Form { wave64, split, legacy, threads, barrier };
    vec![
        // wave32; wave64 native, packed, split
        f(false, false, false, [64, 1, 1], false),
        f(true, false, false, [64, 1, 1], false),
        f(true, false, false, [128, 1, 1], false),
        f(true, true, false, [64, 1, 1], false),
        f(true, true, false, [128, 1, 1], false),
        f(true, true, true, [64, 1, 1], false),
        f(true, true, true, [128, 1, 1], false),
        // 2-D and 3-D groups (the console's thread order, whatever the PC driver's)
        f(false, false, false, [8, 8, 1], false),
        f(true, true, false, [8, 8, 1], false),
        f(true, true, true, [8, 8, 1], false),
        f(false, false, false, [4, 4, 4], false),
        // several waves and a barrier: the waves left with no thread leave without a hang
        f(false, false, false, [128, 1, 1], true),
        f(true, false, false, [128, 1, 1], true),
        f(true, true, false, [128, 1, 1], true),
        f(true, true, true, [128, 1, 1], true),
    ]
}

/// No limit; x limits inside the first wave, on and next to the wave32 and wave64 boundaries,
/// inside the group and in later groups; then each other axis, alone and with the others.
fn limits(threads: [u32; 3]) -> Vec<[u32; 3]> {
    let [x, y, z] = threads;
    let mut l = vec![[0, 0, 0]];
    for lx in [1, 5, 31, 32, 33, 63, 64, 65, x - 1, x + 3, 2 * x + 1] {
        l.push([lx, 0, 0]);
    }
    if y > 1 {
        l.extend([[0, 1, 0], [0, 3, 0], [0, y - 1, 0], [13, 5, 0]]);
    }
    if z > 1 {
        l.extend([[0, 0, 1], [0, 0, z - 1], [6, 3, 2]]);
    }
    l
}

#[test]
fn thread_limits_leave_the_threads_past_each_count_out_of_exec() {
    for f in forms() {
        let (limited, oracle) = (translate_cs(f, false), translate_cs(f, true));
        assert!(limited.contains(&format!("uint3 xcl_tl = io.Load3({}u) - 1u;", super::IO_THREAD_LIMIT[0] * 4)), "{f:?}\n{limited}");
        assert!(!oracle.contains("xcl_tl"), "{f:?}\n{oracle}");
        let (a, b) = (load(&limited), load(&oracle));
        for l in limits(f.threads) {
            let (same, stored) = run(&a, &b, 1, Some(l));
            assert!(same, "{f:?}, limits {l:?}\n{limited}");
            assert_eq!(stored, started(f.threads, l), "{f:?}, limits {l:?}: words the oracle stored");
        }
        for seed in 0..6 {
            assert!(run(&a, &b, seed, None).0, "{f:?}, seed {seed}\n{limited}");
        }
    }
}

/// The lane-independent form (`Options::lane_independent`) starts every thread as its own wave:
/// the limits leave the same threads out of it as out of the wave forms, with no wave operation.
#[test]
fn the_lane_independent_form_keeps_the_thread_limits() {
    for threads in [[64, 1, 1], [128, 1, 1], [8, 8, 1], [4, 4, 4]] {
        let f = Form { wave64: true, split: true, legacy: false, threads, barrier: false };
        super::LANE_INDEPENDENT.set(true);
        let lone = translation(f, false);
        super::LANE_INDEPENDENT.set(false);
        assert!(lone.lane_independent, "{f:?}: the program keeps a wave form\n{}", lone.hlsl);
        assert!(lone.hlsl.contains("xcl_tl") && !lone.hlsl.contains("WaveActive"), "{f:?}\n{}", lone.hlsl);
        let oracle = translation(f, true);
        assert!(!oracle.lane_independent);
        let (a, b) = (load(&lone.hlsl), load(&oracle.hlsl));
        for l in limits(f.threads) {
            let (same, stored) = run(&a, &b, 1, Some(l));
            assert!(same, "{f:?}, limits {l:?}\n{}", lone.hlsl);
            assert_eq!(stored, started(f.threads, l), "{f:?}, limits {l:?}: words the oracle stored");
        }
        for seed in 0..6 {
            assert!(run(&a, &b, seed, None).0, "{f:?}, seed {seed}\n{}", lone.hlsl);
        }
    }
}

/// A wave with every lane past the count leaves at once, through the early-exit-safe barrier
/// where the program has one; where the program waits at a plain group barrier (a barrier the
/// waves are proven to reach together) it stays, as leaving would hang the others.
#[test]
fn waves_with_no_thread_leave_unless_a_plain_barrier_waits_for_them() {
    for f in forms() {
        let hlsl = translate_cs(f, false);
        let body = hlsl.split_once("if ((exec_lo | exec_hi) == 0u)").map_or(hlsl.as_str(), |(_, rest)| rest);
        let leaves = hlsl.lines().find(|l| l.contains("if ((exec_lo | exec_hi) == 0u)")).map(str::trim);
        let want = if body.contains("xcl_s_barrier();") {
            Some("if ((exec_lo | exec_hi) == 0u) { xcl_s_barrier_exit(); return; }")
        } else if f.barrier {
            None
        } else {
            Some("if ((exec_lo | exec_hi) == 0u) { return; }")
        };
        assert_eq!(leaves, want, "{f:?}\n{hlsl}");
    }
}

/// The oracle tells a wrong limit compare and a store left unguarded where EXEC is the launch
/// EXEC (`lane_guards::full_exec`: before the limits, every lane) from the right ones.
#[test]
fn a_wrong_limit_compare_or_an_unguarded_store_is_caught() {
    let f = Form { wave64: false, split: false, legacy: false, threads: [64, 1, 1], barrier: false };
    let (limited, oracle) = (translate_cs(f, false), translate_cs(f, true));
    let b = load(&oracle);
    assert!(run(&load(&limited), &b, 1, Some([5, 0, 0])).0);
    let off_by_one = limited.replacen("+ gtid <= xcl_tl)", "+ gtid <= xcl_tl + 1u)", 1);
    assert_ne!(off_by_one, limited);
    assert!(!run(&load(&off_by_one), &b, 1, Some([5, 0, 0])).0);
    let main = limited.find("void main(").unwrap();
    let store = limited[main..].lines().find(|l| l.contains("xcl_on") && !l.contains("bool xcl_on")).unwrap_or_else(|| panic!("no guarded store\n{limited}"));
    let unguarded = limited.replacen(store, &store.replace("xcl_on", "true"), 1);
    assert!(!run(&load(&unguarded), &b, 1, Some([5, 0, 0])).0, "{store}");
}
