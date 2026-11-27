//! The loop guard with the lane's own test (`LOOP_GUARD`) and, where asked for, region budgets
//! (`XCL_LOOP_GUARD_REGIONS`, `LoopBudget`) against the base form (`XCL_RECOMP_NO_LOOP_REGIONS`: one
//! budget, the wave-wide test) on the same programs: the forms are translated, compiled by the direct backend (`xcl_dxil`) and run by its
//! wave interpreter, with the budgets in the HLSL made small enough to be spent. No GPU and no
//! DXC needed.

use std::collections::{BTreeMap, HashMap};

use xcl_dxil::interp::{self, Config, Verdict};

use super::{translate, CsInterface, GsInterface, LoopBudget, Mode, MsInterface, Options, PsInterface, VsInterface, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::rdna2::encode;

/// One line of a test program: an instruction, a label, or a branch to a label.
enum A {
    I(&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>),
    L(&'static str),
    B(&'static str, &'static str),
}

const V: u64 = 256;
/// scalar source code of the literal that follows the instruction
const LIT: u64 = 255;

fn i(name: &'static str, enc: &'static str, f: &[(&'static str, u64)]) -> A {
    A::I(name, enc, f.to_vec(), None)
}

fn asm(items: &[A]) -> Vec<u8> {
    let mut labels = HashMap::new();
    let mut pc = 0u64;
    for it in items {
        match it {
            A::L(l) => {
                labels.insert(*l, pc);
            }
            A::I(n, e, f, lit) => pc += encode(n, e, f, *lit).unwrap_or_else(|x| panic!("{n}: {x}")).len() as u64,
            A::B(..) => pc += 4,
        }
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
    out
}

/// s[d] = k (0..=64)
fn smov(d: u64, k: u64) -> A {
    i("S_MOV_B32", "ENC_SOP1", &[("SDST", d), ("SSRC0", 128 + k)])
}

/// s[d] += 1
fn sinc(d: u64) -> A {
    i("S_ADD_U32", "ENC_SOP2", &[("SDST", d), ("SSRC0", d), ("SSRC1", 129)])
}

/// SCC = s[d] != k
fn sne(d: u64, k: u32) -> A {
    A::I("S_CMP_LG_U32", "ENC_SOPC", vec![("SSRC0", d), ("SSRC1", LIT)], Some(k))
}

/// v[d] += 1 in the lanes of EXEC
fn vinc(d: u64) -> A {
    i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", d), ("SRC0", 129), ("VSRC1", d)])
}

/// `code` translated with one of the loop guard's forms.
fn hlsl(code: &[u8], opt: &Options, guard: Guard) -> String {
    let t = guard.of(|| translate(code, opt)).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

/// The loop guard's forms: the base one (`XCL_RECOMP_NO_LOOP_REGIONS`), the default (the lane's own
/// test, one budget) and the default with region budgets (`XCL_LOOP_GUARD_REGIONS`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Guard {
    Base,
    Lane,
    Regions,
}

impl Guard {
    /// `f` with this form on this thread.
    fn of<R>(self, f: impl FnOnce() -> R) -> R {
        super::NO_LOOP_REGIONS.set(self == Guard::Base);
        super::LOOP_REGIONS.set(self == Guard::Regions);
        let r = f();
        super::NO_LOOP_REGIONS.set(false);
        super::LOOP_REGIONS.set(false);
        r
    }
}

/// A translation with its loop budget replaced: `region` iterations in a loop region and `total`
/// in all for a program with region budgets, `total` for one with a single budget.
fn with_budget(hlsl: &str, region: u32, total: u32) -> String {
    let guard = super::loop_guard();
    let regions = LoopBudget::Regions { region: guard * super::LOOP_BUDGETS_PER_REGION, total: guard * super::LOOP_BUDGETS_IN_ALL }.defines();
    let single = LoopBudget::Lane(guard).defines();
    assert!(hlsl.contains(&regions) != hlsl.contains(&single), "{hlsl}");
    hlsl.replace(&regions, &LoopBudget::Regions { region, total }.defines()).replace(&single, &LoopBudget::Lane(total).defines())
}

fn load(hlsl: &str, profile: &str) -> interp::Program {
    interp::load(&xcl_dxil::compile(hlsl, "main", profile).unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap()
}

/// The words a kernel-mode program stored in `res` (u1) on a run whose `io` words are `patches`
/// where given, and the run.
fn run_kernel(hlsl: &str, patches: &[(usize, u32)]) -> (BTreeMap<usize, u32>, interp::Run) {
    let input = interp::Input { seed: 2, patches: patches.iter().map(|&(w, v)| ((interp::res::IDENT_IO as u64) << 32 | w as u64, v)).collect() };
    let run = interp::run_input(&load(hlsl, "cs_6_6"), &input, &Config { force_ballots: false, ..Config::default() }, false).0;
    assert!(run.stop.is_none(), "{:?}\n{hlsl}", run.stop);
    let res = interp::res::IDENT_IO | 1;
    (run.written.iter().filter(|w| w.0 == 0 && w.1 == res).map(|w| (w.2 as usize / 4, w.3[0])).collect(), run)
}

/// The word of `res` that holds v[`reg`] of `lane`.
fn vgpr(lane: usize, reg: usize) -> usize {
    KERNEL_SGPRS + lane * KERNEL_VGPRS + reg
}

/// A spin of `spin` tries (a retail FSR pass polls a flag up to 0x10000 times, the loop budget),
/// inside an outer loop that runs once when `nested` (as in that pass), then a loop of `n`
/// iterations, then `s2 = 7`: upstream's `spin_then_loop` (ivall/xcl 287a931b, recomp/tests.rs).
pub(super) fn spin_then_loop(spin: u32, n: u32, nested: bool) -> Vec<u8> {
    let mut p = vec![smov(0, 0), smov(1, 0), smov(2, 0), smov(3, 0)];
    p.push(A::L("outer"));
    if nested {
        p.push(sinc(3));
    }
    p.push(A::L("spin"));
    p.push(sinc(0));
    p.push(sne(0, spin));
    p.push(A::B("S_CBRANCH_SCC1", "spin"));
    if nested {
        p.push(sne(3, 1));
        p.push(A::B("S_CBRANCH_SCC1", "outer"));
    }
    p.push(A::L("loop"));
    p.push(sinc(1));
    p.push(sne(1, n));
    p.push(A::B("S_CBRANCH_SCC1", "loop"));
    p.push(smov(2, 7));
    asm(&p)
}

/// (b) With region budgets (`XCL_LOOP_GUARD_REGIONS`) a spin that spends its loop region's budget,
/// or all but the budget the base form had, leaves the loop after it every iteration: that loop
/// is another region, with a budget of its own. With one budget for the program (the base form,
/// and the default) the second loop leaves at its first guard.
#[test]
fn a_loop_after_a_spent_region_still_runs() {
    for wave64 in [false, true] {
        let opt = Options { mode: Mode::Kernel, wave64, ..Default::default() };
        // budgets as `loop_budget` makes them from a guard of 8: 16 a region, 32 in all
        // a spin that never gives up is cut at its region's 16; the 16 iterations after it run
        let code = spin_then_loop(1000, 16, false);
        let (new, base) = (hlsl(&code, &opt, Guard::Regions), hlsl(&code, &opt, Guard::Base));
        assert!(!new.contains("uint pc = ") && new.matches(super::LOOP_REGION_ENTER).count() == 2, "{new}");
        let s = run_kernel(&with_budget(&new, 16, 32), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (16, 16, 7), "wave64 {wave64}");
        let s = run_kernel(&with_budget(&base, 16, 16), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (16, 0, 7), "wave64 {wave64}: the base form starves the second loop");
        let s = run_kernel(&with_budget(&hlsl(&code, &opt, Guard::Lane), 16, 16), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (16, 0, 7), "wave64 {wave64}: and so does the default, which has one budget too");
        // the title's shape: a spin of exactly one guard (8 tries) in an outer loop, whose head
        // spends from the same count
        let code = spin_then_loop(8, 3, true);
        let (new, base) = (hlsl(&code, &opt, Guard::Regions), hlsl(&code, &opt, Guard::Base));
        let s = run_kernel(&with_budget(&new, 16, 32), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (8, 3, 7), "wave64 {wave64}");
        let s = run_kernel(&with_budget(&base, 8, 8), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (7, 0, 7), "wave64 {wave64}: the base form cuts the spin and starves the loop");
        // the `pc chain` too, where the spin fits its region (a guard that gives up there ends
        // the program, as before: no later block matches the `pc` it leaves with)
        let chain = hlsl(&code, &Options { structured: false, ..opt.clone() }, Guard::Regions);
        assert!(chain.contains("uint pc = ") && chain.matches(super::LOOP_REGION_ENTER).count() == 2, "{chain}");
        let s = run_kernel(&with_budget(&chain, 16, 32), &[]).0;
        assert_eq!((s[&0], s[&1], s[&2]), (8, 3, 7), "wave64 {wave64}, pc chain");
    }
}

/// (d) Loops that never end still stop: with region budgets every region is reached and leaves,
/// the first ones after their region's budget and the later ones at once, when the lane has
/// spent the program's total (upstream's `gpu_endless_loops_in_many_regions_stop_after_four_budgets`,
/// ivall/xcl 287a931b). The lanes that gave up are counted once each (`xcl_trip`). By default,
/// as in the base form, the program's total is its one budget: no lane runs an iteration more.
#[test]
fn loops_that_never_end_stop_at_the_programs_total() {
    // six regions, each `s0 += 1` until s0 == 0 (2^32 tries); s1 counts the regions reached
    let mut p = vec![smov(0, 0), smov(1, 0)];
    for l in ["a", "b", "c", "d", "e", "f"] {
        p.push(sinc(1));
        p.push(A::L(l));
        p.push(sinc(0));
        p.push(sne(0, 0));
        p.push(A::B("S_CBRANCH_SCC1", l));
    }
    let code = asm(&p);
    let trips = [(super::IO_TRIP_AT, 0x40), (super::IO_TRIP_WINDOW, 0)];
    for wave64 in [false, true] {
        let opt = Options { mode: Mode::Kernel, wave64, ..Default::default() };
        let new = hlsl(&code, &opt, Guard::Regions);
        assert_eq!(new.matches(super::LOOP_REGION_ENTER).count(), 6, "{new}");
        let (s, run) = run_kernel(&with_budget(&new, 8, 20), &trips);
        assert_eq!((s[&0], s[&1]), (20, 6), "wave64 {wave64}: 8 + 8 + 4 iterations, every region reached");
        let lanes = if wave64 { 64 } else { 32 };
        assert_eq!(run.atomics.count, lanes, "one count a lane, whatever the loops it gave up in");
        assert!(run.atomics.sample.iter().all(|a| (a.addr, a.a) == (0x40, 1)), "{:?}", run.atomics.sample);
        // one budget (the default, the base form, and what a graphics stage always has): the same stop
        for guard in [Guard::Lane, Guard::Base] {
            let one = hlsl(&code, &opt, guard);
            assert!(!one.contains("xcl_btotal") && one.contains("static uint xcl_budget = XCL_LOOP_GUARD;"), "{one}");
            let (s, run) = run_kernel(&with_budget(&one, 20, 20), &trips);
            assert_eq!((s[&0], s[&1], run.atomics.count), (20, 6, lanes), "wave64 {wave64}, {guard:?}");
        }
    }
}

/// (c) A lane leaves by its own count, which the lanes of a console wave share: a wave32 program
/// in a 64-lane hardware wave is two console waves (`xcl_w32`), each with its own scalar state.
/// Here the inner loop runs 10 times an outer iteration in the lower half and twice in the upper
/// one, twice over: the lower half spends its budget of 12 in the second round. With the lane's
/// own test the upper half computes what it does with no budget at all; the wave-wide test of
/// the base form took it out of its inner loop with the lower half.
#[test]
fn a_spent_console_wave_leaves_the_other_one_in_its_hardware_wave_alone() {
    let p = [
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]), // exec = every lane
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)]),
        i("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)]), // s4 = v0 of the wave's first lane
        smov(1, 0),
        A::L("outer"),
        smov(0, 0),
        A::L("inner"),
        vinc(1),
        sinc(0),
        i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 4)]),
        A::B("S_CBRANCH_SCC1", "inner"),
        sinc(1),
        sne(1, 2),
        A::B("S_CBRANCH_SCC1", "outer"),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 135)]),
    ];
    let code = asm(&p);
    let opt = Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() };
    let (new, base) = (hlsl(&code, &opt, Guard::Lane), hlsl(&code, &opt, Guard::Base));
    assert!(new.contains("[WaveSize(64)]") && new.contains(super::LOOP_GUARD) && !new.contains("WaveActiveAnyTrue(xcl_budget"), "{new}");
    // v0 of each half's first lane: the inner loop's count
    let counts = [(vgpr(0, 0), 10), (vgpr(32, 0), 2)];
    let v1 = |hlsl: &str| {
        let s = run_kernel(hlsl, &counts).0;
        let half = |h: usize| {
            let lanes: Vec<u32> = (0..32).map(|l| s[&vgpr(32 * h + l, 1)]).collect();
            assert!(lanes.iter().all(|&v| v == lanes[0]) && (0..32).all(|l| s[&vgpr(32 * h + l, 2)] == 7), "half {h}: {lanes:?}");
            lanes[0]
        };
        (half(0), half(1))
    };
    assert_eq!(v1(&with_budget(&new, 1 << 20, 1 << 20)), (20, 4), "no budget spent");
    // lower half: 1 + 10 in the first round, 1 in the second, and the inner loop leaves at once
    assert_eq!(v1(&with_budget(&new, 12, 12)), (10, 4), "the upper half is not disturbed");
    assert_eq!(v1(&with_budget(&base, 12, 12)), (10, 2), "the base form's wave-wide test takes the upper half along");
}

/// A lane that has spent its budget is not killed, also where EXEC is proven full after the
/// loop (`lane_guards::full_exec`: set to all lanes before it, not written in it): the write and
/// the store after the loop have no lane guard, and every lane writes and stores what it had
/// counted when the guard took it out of the loop (6 of 10 iterations) – what the base form
/// stores too. Nothing after a guard's exit depends on a lane having finished its loops.
#[test]
fn a_spent_lane_stores_where_exec_is_proven_full_after_the_loop() {
    let code = asm(&[
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)]),
        smov(1, 0),
        A::L("loop"),
        vinc(1),
        sinc(1),
        sne(1, 10),
        A::B("S_CBRANCH_SCC1", "loop"),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 135)]),
        i("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 1), ("SADDR", 0x7D)]),
    ]);
    let opt = Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() };
    let (new, base) = (hlsl(&code, &opt, Guard::Lane), hlsl(&code, &opt, Guard::Base));
    let tail = &new[new.rfind(super::LOOP_GUARD).unwrap()..new.rfind("if (xcl_hw_lane == 0u)").unwrap()];
    assert!(tail.contains("v2 = (7u);") && tail.contains("if (true) {"), "{tail}");
    let inputs: Vec<(usize, u32)> = (0..64).map(|l| (vgpr(l, 6), 4 * l as u32)).chain([(super::IO_TRIP_AT, 0)]).collect();
    let run = |hlsl: &str| {
        let (s, run) = run_kernel(hlsl, &inputs);
        let mem: BTreeMap<u64, u32> = run.written.iter().filter(|w| w.0 == 0 && w.1 >> 30 == 2).map(|w| (w.2, w.3[0])).collect();
        assert_eq!(mem.len(), 64, "every lane stores");
        (0..64).map(|l| (s[&vgpr(l, 1)], s[&vgpr(l, 2)], mem[&(4 * l as u64)])).collect::<Vec<_>>()
    };
    assert_eq!(run(&with_budget(&new, 1 << 20, 1 << 20)), vec![(10, 7, 10); 64]);
    assert_eq!(run(&with_budget(&new, 6, 6)), vec![(6, 7, 6); 64]);
    assert_eq!(run(&with_budget(&base, 6, 6)), vec![(6, 7, 6); 64]);
}

/// The divergent loop of `a_spent_wave_runs_on_where_the_saved_exec_is_put_back_after_the_loops`:
/// every lane on and EXEC saved; twice, a loop each lane leaves after its own v12 iterations
/// (v1 counts them); then the saved EXEC back, `v2 = 7` and a store of v1 at the lane's address.
fn divergent_then_store() -> Vec<u8> {
    asm(&[
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 126)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)]),
        smov(1, 0),
        A::L("outer"),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 128)]),
        smov(20, 0),
        A::L("inner"),
        vinc(1),
        vinc(13),
        i("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 13), ("VSRC1", 12)]),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 106), ("SSRC1", 20)]),
        i("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        A::B("S_CBRANCH_EXECNZ", "inner"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        sinc(1),
        sne(1, 2),
        A::B("S_CBRANCH_SCC1", "outer"),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 5)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 135)]),
        i("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 1), ("SADDR", 0x7D)]),
    ])
}

/// A lane that has spent its budget is not killed: it leaves its loop by the guard's exit, the
/// edge the lane-guard proofs follow (`GuardExits`, `lane_guards::full_exec`), with EXEC as the
/// program has it and no other mark on the lane, and runs on with its console wave. So what the
/// proofs say after the loop holds as it did with the wave-wide guard: here the saved EXEC is
/// put back after the loops, the write and the store after it have no lane guard, and every lane
/// writes and stores – a spent wave (the lower half: its inner loop is cut at 5 of up to 10
/// iterations, with lanes already out of EXEC, and its second round never starts) what it had
/// counted, exactly as in the base form, and the other wave of the hardware wave (2 iterations
/// a round, 6 guards) what it stores with no budget at all.
#[test]
fn a_spent_wave_runs_on_where_the_saved_exec_is_put_back_after_the_loops() {
    let code = divergent_then_store();
    let opt = Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() };
    let (new, base) = (hlsl(&code, &opt, Guard::Lane), hlsl(&code, &opt, Guard::Base));
    // the proof at work after the loops: the guarded form of the same program differs there
    let guarded = hlsl(&code, &Options { exec_proof: false, ..opt.clone() }, Guard::Lane);
    let tail = |h: &str| h[h.rfind(super::LOOP_GUARD).unwrap()..h.rfind("if (xcl_hw_lane == 0u)").unwrap()].to_string();
    assert!(tail(&new).contains("v2 = (7u);") && tail(&new).contains("if (true) {"), "{}", tail(&new));
    assert!(tail(&guarded).contains("v2 = select(xcl_on, (7u), v2);") && !tail(&guarded).contains("if (true) {"), "{}", tail(&guarded));
    // each lane's count (v12: up to 10 in the lower half, up to 2 in the upper) and address (v6);
    // no trip counters, which are stored in memory too
    let count = |lane: usize| if lane < 32 { 1 + lane as u32 % 10 } else { 1 + lane as u32 % 2 };
    let inputs: Vec<(usize, u32)> = (0..64).flat_map(|l| [(vgpr(l, 12), count(l)), (vgpr(l, 6), 4 * l as u32)]).chain([(super::IO_TRIP_AT, 0)]).collect();
    let run = |hlsl: &str| {
        let (s, run) = run_kernel(hlsl, &inputs);
        let mem: BTreeMap<u64, u32> = run.written.iter().filter(|w| w.0 == 0 && w.1 >> 30 == 2).map(|w| (w.2, w.3[0])).collect();
        assert_eq!(mem.len(), 64, "every lane stores");
        (0..64).map(|l| (s[&vgpr(l, 1)], s[&vgpr(l, 2)], mem[&(4 * l as u64)])).collect::<Vec<_>>()
    };
    let free: Vec<_> = (0..64).map(|l| (2 * count(l), 7, 2 * count(l))).collect();
    assert_eq!(run(&with_budget(&new, 1 << 20, 1 << 20)), free);
    let spent: Vec<_> = (0..64).map(|l| if l < 32 { (count(l).min(5), 7, count(l).min(5)) } else { free[l] }).collect();
    assert_eq!(run(&with_budget(&new, 6, 6)), spent);
    // the base form: the same in the spent wave; the other wave is taken out of its second round
    let dragged: Vec<_> = (0..64).map(|l| if l < 32 { spent[l] } else { (count(l), 7, count(l)) }).collect();
    assert_eq!(run(&with_budget(&base, 6, 6)), dragged);
}

type Form = (&'static str, Options, &'static str);

fn cs(threads: u32) -> Mode {
    Mode::Compute(CsInterface { threads: [threads, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 })
}

fn el(name: &str, sv: u32, reg: u32) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 }
}

fn ps() -> Mode {
    Mode::Pixel(PsInterface { inputs: vec![el("SV_Position", 1, 0), el("COLOR", 0, 1)], outputs: vec![el("SV_Target", 64, 0)], input_ena: 0x302, attrs: None })
}

/// The emission forms a loop guard is written in.
fn forms() -> Vec<Form> {
    let o = Options::default;
    vec![
        ("kernel wave32", Options { mode: Mode::Kernel, wave64: false, ..o() }, "cs_6_6"),
        ("kernel wave64", Options { mode: Mode::Kernel, wave64: true, ..o() }, "cs_6_6"),
        ("kernel wave32 in a 64-lane wave", Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..o() }, "cs_6_6"),
        ("kernel pc chain", Options { mode: Mode::Kernel, wave64: true, structured: false, ..o() }, "cs_6_6"),
        ("kernel split wave64", Options { mode: Mode::Kernel, wave64: true, split_wave64: true, ..o() }, "cs_6_6"),
        ("compute native wave64", Options { mode: cs(64), wave64: true, ..o() }, "cs_6_6"),
        ("compute packed wave64", Options { mode: cs(64), wave64: true, split_wave64: true, legacy_compute_wave64: false, ..o() }, "cs_6_6"),
        ("compute split wave64", Options { mode: cs(64), wave64: true, split_wave64: true, legacy_compute_wave64: true, ..o() }, "cs_6_6"),
        ("compute split wave64, two console waves", Options { mode: cs(128), wave64: true, split_wave64: true, legacy_compute_wave64: true, ..o() }, "cs_6_6"),
        ("compute packed wave64, loop profile", Options { mode: cs(64), wave64: true, split_wave64: true, legacy_compute_wave64: false, loop_profile: true, ..o() }, "cs_6_6"),
        ("pixel wave32", Options { mode: ps(), wave64: false, ..o() }, "ps_6_6"),
        ("pixel wave64", Options { mode: ps(), wave64: true, ..o() }, "ps_6_6"),
        ("pixel wave64 on 32-lane waves", Options { mode: ps(), wave64: true, split_wave64: true, ..o() }, "ps_6_6"),
    ]
}

/// A program of every form with loops that spend a small budget: a counted nest (3 x 4 inner
/// iterations, 16 guards), a loop of 5, then a divergent loop the lanes leave one by one
/// (`structure_tests::a_divergent_loop`), each adding into v8 / v9, which a pixel shader exports,
/// a compute shader stores to memory at 4 * its thread id and a kernel stores with its registers.
fn looping_program(mode: &Mode) -> Vec<u8> {
    let mut p = Vec::new();
    if let Mode::Pixel(_) = mode {
        for c in 0..4u64 {
            p.push(i("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4 + c), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", c)]));
            p.push(i("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4 + c), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", c)]));
        }
    }
    p.push(i("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)]));
    p.push(i("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)]));
    p.push(smov(10, 0));
    p.push(A::L("outer"));
    p.push(smov(11, 0));
    p.push(A::L("inner"));
    p.push(vinc(8));
    p.push(sinc(11));
    p.push(sne(11, 4));
    p.push(A::B("S_CBRANCH_SCC1", "inner"));
    p.push(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 8), ("VSRC1", 9)]));
    p.push(sinc(10));
    p.push(sne(10, 3));
    p.push(A::B("S_CBRANCH_SCC1", "outer"));
    p.push(smov(12, 0));
    p.push(A::L("second"));
    p.push(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 131), ("VSRC1", 8)]));
    p.push(sinc(12));
    p.push(sne(12, 5));
    p.push(A::B("S_CBRANCH_SCC1", "second"));
    // lanes leave as v13 reaches their v12 = v9 & 7
    p.push(i("V_AND_B32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 135), ("VSRC1", 9)]));
    p.push(i("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 128)]));
    p.push(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)]));
    p.push(A::L("divergent"));
    p.push(vinc(9));
    p.push(vinc(13));
    p.push(i("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 13), ("VSRC1", 12)]));
    p.push(i("S_OR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 106), ("SSRC1", 20)]));
    p.push(i("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]));
    p.push(A::B("S_CBRANCH_EXECNZ", "divergent"));
    p.push(i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]));
    match mode {
        Mode::Pixel(_) => p.push(i("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 8), ("VSRC3", 9), ("DONE", 1), ("VM", 1)])),
        Mode::Compute(_) => {
            p.push(i("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 131), ("SRC2", 128)]));
            p.push(i("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)]));
            p.push(i("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 8), ("SRC1", 136), ("SRC2", V + 9)]));
            p.push(i("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)]));
        }
        _ => {}
    }
    asm(&p)
}

/// (c) In a hardware wave whose lanes are one console wave the lanes count the same guards, so
/// the lane's own test leaves a loop exactly where the wave-wide one did: with the same single
/// budget (a region budget equal to the total is one) the forms compute the same in every
/// emission form – the default, and a compute or kernel program with region budgets – whether
/// the budget is spent in the nest's inner loop (9), between two loops (16, 21), in the
/// divergent loop (23) or not at all. One iteration more is a different result (the runs are not
/// blind to where the guard leaves; the counted meets of a split wave64 group of several console
/// waves race by design, which the verdict then names).
#[test]
fn the_lanes_own_test_leaves_where_the_wave_wide_one_did() {
    for (name, opt, profile) in forms() {
        let code = looping_program(&opt.mode);
        let base = hlsl(&code, &opt, Guard::Base);
        assert!(base.contains("WaveActiveAnyTrue(xcl_budget == 0u)"), "{name}\n{base}");
        let cfg = Config::default();
        for guard in [Guard::Lane, Guard::Regions] {
            let regions = guard == Guard::Regions;
            if regions && !matches!(opt.mode, Mode::Kernel | Mode::Compute(_)) {
                continue;
            }
            let new = hlsl(&code, &opt, guard);
            assert!(!new.contains("WaveActiveAnyTrue(xcl_budget") && new.contains("if (xcl_budget == 0u) {"), "{name}\n{new}");
            assert_eq!(new.contains(super::LOOP_REGION_ENTER), regions, "{name}");
            // (region budgets: where the two differ in text, in and between loops)
            for budget in if regions { vec![9, 21] } else { vec![9, 16, 21, 23, 1 << 16] } {
                let a = load(&with_budget(&base, budget, budget), profile);
                let b = load(&with_budget(&new, budget, budget), profile);
                let (verdict, seed) = interp::diff(&a, &b, 0..4, &cfg);
                assert_eq!(verdict, Verdict::Equal, "{name}, {guard:?}, budget {budget}, seed {seed}\n{new}");
                if budget == 9 {
                    let more = load(&with_budget(&new, budget + 1, budget + 1), profile);
                    let told = interp::diff(&a, &more, 0..4, &cfg).0;
                    assert!(matches!(told, Verdict::Different(_) | Verdict::Race(_)), "{name}, {guard:?}: a budget of {} is not told from {budget}: {told:?}", budget + 1);
                }
            }
        }
    }
}

/// The text of the base form's guard and counter, as the tree had them before the regions
/// (live tree 2067d94a, recomp/mod.rs `LOOP_GUARD` and emit.rs `PRELUDE_START`).
const BASE_GUARD: &str = "if (WaveActiveAnyTrue(xcl_budget == 0u)) { xcl_trip(0u); break; } xcl_budget -= min(xcl_budget, 1u);";
const BASE_PROFILED: [&str; 2] = ["if (WaveActiveAnyTrue(xcl_budget == 0u)) { xcl_lp_exit = 1u; xcl_trip(0u); break; }", "xcl_budget -= min(xcl_budget, 1u); xcl_lp_n++;"];
const BASE_COUNTER: &str = "\nstatic uint lane_id;\n// loop iterations a lane may run (`LOOP_GUARD`; the translation's `loop_guard`)\nstatic uint xcl_budget = XCL_LOOP_GUARD;\n";

/// (a) `XCL_RECOMP_NO_LOOP_REGIONS` gives the base form's HLSL: its guard line at the top of
/// every loop and its one counter, none of the new text, and nothing else of the program differs
/// from the present forms (putting the base text back into either gives the same bytes). The
/// default's counter is the base form's as it is: one budget, `loop_guard`.
#[test]
fn the_switch_gives_the_base_forms_hlsl() {
    let guard = super::loop_guard();
    let mesh = Mode::Mesh(MsInterface { threads: 64, max_vertices: 64, max_primitives: 64, ..Default::default() });
    let more: Vec<Form> = vec![
        ("vertex", Options { mode: Mode::Vertex(VsInterface::default()), wave64: false, ..Default::default() }, "vs_6_6"),
        ("mesh, two passes", Options { mode: mesh.clone(), wave64: true, ..Default::default() }, "cs_6_6"),
        ("mesh, two passes, split wave64", Options { mode: mesh, wave64: true, split_wave64: true, ..Default::default() }, "cs_6_6"),
        ("geometry", Options { mode: Mode::Geometry(GsInterface { max_vertex_count: 3, gs_start: Some(0), ..Default::default() }), wave64: true, ..Default::default() }, "cs_6_6"),
    ];
    for (name, opt, _) in forms().into_iter().chain(more) {
        let code = looping_program(&opt.mode);
        let base = hlsl(&code, &opt, Guard::Base);
        let program = &base[base.rfind(" main(").unwrap()..];
        let loops = program.matches("[loop] for (;;) {").count() + program.matches("[loop] while (pc >= ").count();
        assert_eq!(loops, 4, "{name}\n{base}");
        let profiled = opt.loop_profile;
        assert_eq!(base.matches(BASE_GUARD).count(), if profiled { 0 } else { loops }, "{name}\n{base}");
        assert!(BASE_PROFILED.iter().all(|line| base.matches(line).count() == if profiled { loops } else { 0 }), "{name}\n{base}");
        assert_eq!(base.matches(&format!("#define XCL_LOOP_GUARD {guard}u\n{BASE_COUNTER}")).count(), 1, "{name}\n{base}");
        assert!(["xcl_btotal", "XCL_LOOP_REGION", "XCL_LOOP_TOTAL", "xcl_budget--", "if (xcl_budget == 0u)"].iter().all(|new| !base.contains(new)), "{name}\n{base}");
        // the present forms with the base text put back
        for form in [Guard::Lane, Guard::Regions] {
            let new = hlsl(&code, &opt, form);
            let regions = form == Guard::Regions && matches!(opt.mode, Mode::Kernel | Mode::Compute(_));
            assert_eq!((new.matches(super::LOOP_REGION_ENTER).count(), new.matches(super::LOOP_REGION_LEAVE).count()), if regions { (3, 3) } else { (0, 0) }, "{name}\n{new}");
            assert_eq!(new.matches(&format!("#define XCL_LOOP_GUARD {guard}u\n{BASE_COUNTER}")).count(), if regions { 0 } else { 1 }, "{name}\n{new}");
            let lines: Vec<&str> = new.lines().filter(|l| ![super::LOOP_REGION_ENTER, super::LOOP_REGION_LEAVE].contains(&l.trim_start())).collect();
            let back = (lines.join("\n") + "\n")
                .replace("if (xcl_budget == 0u) { xcl_trip(0u); break; } xcl_budget--;", BASE_GUARD)
                .replace("if (xcl_budget == 0u) { xcl_lp_exit = 1u; xcl_trip(0u); break; }", BASE_PROFILED[0])
                .replace("xcl_budget--; xcl_lp_n++;", BASE_PROFILED[1])
                .replace(&format!("#define XCL_LOOP_REGION {}u\n#define XCL_LOOP_TOTAL {}u\n{}", guard * 2, guard * 4, super::emit::PRELUDE_START_REGIONS), &format!("#define XCL_LOOP_GUARD {guard}u\n{BASE_COUNTER}"));
            assert!(back == base, "{name}, {form:?}: the forms differ in more than the loop guard\n{new}\n{base}");
        }
    }
}

/// Region budgets (`XCL_LOOP_GUARD_REGIONS`) are a compute or kernel program's; a graphics, geometry
/// or mesh stage has one budget for all its loops then too (upstream ivall/xcl 22c5f68c), the
/// lane's own test in every stage. A loop in another loop is in that loop's region. By default
/// every stage has the one budget, `loop_guard`.
#[test]
fn region_budgets_are_compute_and_kernel_only() {
    let mesh = MsInterface { threads: 64, max_vertices: 64, max_primitives: 64, ..Default::default() };
    let guard_line = "if (xcl_budget == 0u) { xcl_trip(0u); break; } xcl_budget--;";
    for (mode, regions) in [(Mode::Kernel, true), (cs(64), true), (ps(), false), (Mode::Vertex(VsInterface::default()), false), (Mode::Mesh(mesh), false)] {
        let opt = Options { mode: mode.clone(), ..Default::default() };
        let new = hlsl(&looping_program(&mode), &opt, Guard::Regions);
        assert_eq!(new.matches(guard_line).count(), 4, "{mode:?}\n{new}");
        assert_eq!(new.contains("static uint xcl_btotal = XCL_LOOP_TOTAL;"), regions, "{mode:?}");
        assert_eq!(new.contains("static uint xcl_budget = XCL_LOOP_GUARD;"), !regions, "{mode:?}");
        assert_eq!(new.matches(super::LOOP_REGION_ENTER).count(), if regions { 3 } else { 0 }, "{mode:?}\n{new}");
        let lane = hlsl(&looping_program(&mode), &opt, Guard::Lane);
        assert_eq!(lane.matches(guard_line).count(), 4, "{mode:?}\n{lane}");
        assert!(lane.contains(&LoopBudget::Lane(super::loop_guard()).defines()) && lane.contains("static uint xcl_budget = XCL_LOOP_GUARD;"), "{mode:?}\n{lane}");
        assert!(!lane.contains("xcl_btotal") && !lane.contains("XCL_LOOP_REGION"), "{mode:?}\n{lane}");
    }
}

/// A loop region has `LOOP_BUDGETS_PER_REGION` guards to spend, so a loop whose static bound is
/// between one guard and that is not named as one the guard may cut short in a compute program
/// with region budgets (`long_loops`); with one budget it is.
#[test]
fn long_loops_are_named_by_what_a_loop_can_get() {
    let guard = super::loop_guard();
    let code = asm(&[smov(0, 0), A::L("l"), sinc(0), A::I("S_CMP_LT_U32", "ENC_SOPC", vec![("SSRC0", 0), ("SSRC1", LIT)], Some(guard + 9)), A::B("S_CBRANCH_SCC1", "l")]);
    let named = |mode: Mode, form: Guard| {
        let t = form.of(|| translate(&code, &Options { mode, ..Default::default() })).unwrap();
        (t.long_loops.len(), t.loop_budget.unwrap())
    };
    assert_eq!(named(Mode::Kernel, Guard::Regions), (0, LoopBudget::Regions { region: 2 * guard, total: 4 * guard }));
    assert_eq!(named(ps(), Guard::Regions), (1, LoopBudget::Lane(guard)));
    assert_eq!(named(Mode::Kernel, Guard::Lane), (1, LoopBudget::Lane(guard)));
    assert_eq!(named(Mode::Kernel, Guard::Base), (1, LoopBudget::Wave(guard)));
}
