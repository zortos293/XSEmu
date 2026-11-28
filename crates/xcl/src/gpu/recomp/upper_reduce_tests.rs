//! `XCL_UPPER_LANES_REDUCTION`: a whole-wave reduction read as a wave operation over the
//! lanes the wave runs (`upper_reduce`), and the partial-wave numbers it fixes.
//!
//! The fixture is the pack's reduction (`docs/wave64-on-wave32.md`, "What the retail pack
//! does"): the identity into the lanes the mask leaves out, DPP8 / DPP16 xor 1, 2 and 4,
//! row_mirror, the other row through `v_permlanex16`, `v_readlane` 32 for the upper half and
//! `v_readlane` 0 for the total. Translated for a GPU that can run 32-lane waves and run by
//! `xcl-dxil`'s interpreter, once with the switch off (the transcribed fold) and once on
//! (the wave operation), at a full wave and at a partial one (`Config::live`): the console's
//! lanes that the host wave did not launch hold the identity there, and what the fold reads
//! from them instead has no defined value on a GPU.

use std::collections::BTreeMap;

use super::upper_reduce::{self, Op, TEST_ON};
use super::wave64_reduce_corpus_tests::{corpus, corpus_named, min_ps, options, ps, Mask};
use super::translate;
use crate::gpu::rdna2::Decoded;
use xcl_dxil::interp::{self, Config};

/// A translated program with the switch set for this thread only.
fn translate_with(code: &[u8], on: bool) -> super::Translation {
    TEST_ON.with(|t| t.set(Some(on)));
    let t = translate(code, &options());
    TEST_ON.with(|t| t.set(None));
    t.unwrap_or_else(|e| panic!("{e}"))
}

/// A program's instructions, far jumps resolved, as `upper_lanes` and the emitter see them.
fn decoded(code: &[u8]) -> Vec<Decoded> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).unwrap();
        off += d.len;
        out.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut out);
    out
}

/// The reduction plan of a program.
fn plan_of(code: &[u8]) -> (super::upper_lanes::UpperLanes, upper_reduce::Plan) {
    let insts = decoded(code);
    let a = super::upper_lanes::analyze(&insts);
    let p = upper_reduce::plan(&insts, &a, &[]);
    (a, p)
}

/// Every exported component 0 of target 0, by lane.
fn exports(run: &interp::Run) -> BTreeMap<u32, f32> {
    let mut out = BTreeMap::new();
    for &(kind, element, at, v) in &run.written {
        if kind == 2 && element == 0 {
            out.insert((at >> 16) as u32, f32::from_bits(v[0]));
        }
    }
    out
}

/// Run a translated pixel shader in the interpreter as one wave of `wave` lanes, of which
/// the first `live` run (None: all).
fn run_px(hlsl: &str, wave: u32, live: Option<u32>) -> BTreeMap<u32, f32> {
    let dxil = xcl_dxil::compile(hlsl, "main", "ps_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let p = interp::load_with(&dxil, wave).unwrap_or_else(|e| panic!("{e}"));
    let cfg = Config { force_ballots: false, live, ..Config::default() };
    exports(&interp::run(&p, 0, &cfg))
}

#[test]
fn the_pack_reduction_is_recognised() {
    let (_, plan) = plan_of(&min_ps());
    assert_eq!(plan.reds.len(), 1, "{plan:?}");
    assert_eq!(plan.reds[0].op, Op::MinU32);
    assert_eq!(plan.reds[0].vgpr, 6);
    let insts = decoded(&min_ps());
    let seed = insts.iter().find(|d| d.pc == plan.reds[0].pc).unwrap();
    assert_eq!(seed.inst.name, "V_CNDMASK_B32");
    let read = insts.iter().find(|d| plan.sites.contains_key(&d.pc)).unwrap();
    assert_eq!(read.inst.name, "V_READLANE_B32");
    // the DPP16 xor steps (a different encoding of the same fold) and no upper-half step
    let (_, p) = plan_of(&ps(false, "V_CVT_F32_U32", "V_MIN_U32", 193, None, Mask::Wave, true, false));
    assert_eq!(p.reds.len(), 1);
    // the odd-lane mask under the saved EXEC, as the draw tests build it
    let (_, p) = plan_of(&ps(false, "V_CVT_F32_U32", "V_MIN_U32", 193, None, Mask::Odd, false, true));
    assert_eq!(p.reds.len(), 1);
}

#[test]
fn a_reduction_that_is_not_the_packs_is_left_alone() {
    // a mask that is not under the saved EXEC: its bits may hold lanes the wave never launched
    let (_, p) = plan_of(&ps(false, "V_CVT_F32_U32", "V_MIN_U32", 193, None, Mask::Any, false, true));
    assert!(p.is_empty(), "{p:?}");
    // a constant that is not the operation's identity: the console's lanes 32..63 would join
    let (_, p) = plan_of(&ps(false, "V_CVT_F32_U32", "V_MIN_U32", 128, None, Mask::Wave, false, true));
    assert!(p.is_empty(), "{p:?}");
    // the pack's fold without its row mirror folds 8 lanes, not 32
    let (_, p) = plan_of(&corpus_named("no-row-mirror"));
    assert!(p.is_empty(), "{p:?}");
}

#[test]
fn a_program_with_two_reductions_folds_both() {
    let (_, p) = plan_of(&corpus_named("two-reductions"));
    assert_eq!(p.reds.len(), 2, "{p:?}");
    assert_eq!((p.seeds.len(), p.sites.len()), (2, 2));
    assert_ne!(p.reds[0].pc, p.reds[1].pc);
    assert_eq!(p.reds[0].op, Op::MinU32);
    assert_eq!(p.reds[1].op, Op::MaxU32);
}

#[test]
fn the_switch_reads_the_reduction_as_a_wave_operation() {
    let code = min_ps();
    let off = translate_with(&code, false);
    let on = translate_with(&code, true);
    assert!(off.complete() && on.complete(), "{:?} {:?}", off.unsupported_why, on.unsupported_why);
    assert!(!off.hlsl.contains("WaveActiveMin") && !off.hlsl.contains("xcl_wre"), "switch off: the fold, as before");
    assert!(off.hlsl.contains("rla(v6, 0u & 63u)"), "switch off: lane 0 read through the fold");
    assert!(on.hlsl.contains("uint xcl_wre0 = v6;"), "the seed is kept before the fold overwrites it");
    assert!(on.hlsl.contains("WaveActiveMin(xcl_wre0)"), "the read is the wave operation");
    assert!(!on.hlsl.contains("rla(v6, 0u & 63u)"), "the fold's lane 0 read is gone");
}

/// The switch off emits today's HLSL, byte for byte: no corpus program's translation
/// mentions the switch's names, and every one of them translates.
#[test]
fn the_switch_off_changes_nothing() {
    for (name, code) in corpus() {
        let t = translate_with(&code, false);
        assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
        assert!(!t.hlsl.contains("xcl_wre"), "{name}: the switch is off");
        for wave in ["WaveActiveMin(", "WaveActiveMax(", "WaveActiveBitOr(", "WaveActiveBitAnd(", "WaveActiveBitXor(", "WaveActiveSum("] {
            assert!(!t.hlsl.contains(wave), "{name}: {wave} with the switch off");
        }
    }
}

#[test]
fn the_corpus_translates_the_same_programs() {
    let mut off_ok = 0;
    let mut on_ok = 0;
    let mut folded = 0;
    for (name, code) in corpus() {
        TEST_ON.with(|t| t.set(Some(false)));
        let off = translate(&code, &options());
        TEST_ON.with(|t| t.set(Some(true)));
        let on = translate(&code, &options());
        TEST_ON.with(|t| t.set(None));
        assert_eq!(off.is_ok(), on.is_ok(), "{name}: the switch changes whether it translates");
        let (Ok(off), Ok(on)) = (off, on) else { continue };
        assert_eq!((off.complete(), on.complete()), (true, true), "{name}");
        off_ok += 1;
        on_ok += 1;
        let (_, plan) = plan_of(&code);
        if !plan.is_empty() {
            folded += 1;
            assert_ne!(off.hlsl, on.hlsl, "{name}: recognised ({plan:?}) and unchanged");
            for i in 0..plan.reds.len() {
                assert!(on.hlsl.contains(&format!("xcl_wre{i}")), "{name}: reduction {i} folded");
            }
        } else {
            assert_eq!(off.hlsl, on.hlsl, "{name}: not recognised, so the switch must not change the HLSL");
        }
    }
    assert_eq!(off_ok, on_ok);
    assert!(folded > 40, "{folded} of {} corpus programs folded", corpus().len());
    println!("corpus: {} programs, {folded} with a recognised reduction", corpus().len());
}

/// The two forms of the pack's min reduction as DXIL, for the tree's own comparison tool:
/// `XCL_UPPER_REDUCE_DXIL=<prefix>` writes `<prefix>.fold.dxil` and `<prefix>.wave.dxil`
/// (both a pixel shader that declares no wave size). With lanes 0..2 of a 32-lane wave live,
/// `xdiff <prefix>.fold.dxil <prefix>.wave.dxil --wave 32 --lanes 3` reports the difference
/// this switch fixes; without `--lanes` it reports the two equal.
#[test]
#[ignore]
fn wave64_reduction_dxil_pair() {
    let Ok(prefix) = std::env::var("XCL_UPPER_REDUCE_DXIL") else { return };
    let code = min_ps();
    for (form, on) in [("fold", false), ("wave", true)] {
        let hlsl = translate_with(&code, on).hlsl;
        let dxil = xcl_dxil::compile(&hlsl, "main", "ps_6_6").unwrap_or_else(|e| panic!("{form}: {e}"));
        std::fs::write(format!("{prefix}.{form}.dxil"), dxil).unwrap();
    }
}

/// Both forms of every corpus program compile (the fold's and the wave operation's), and a
/// full wave gives the same exports: the switch changes nothing a full wave can see.
#[test]
fn the_corpus_agrees_at_a_full_wave() {
    for (name, code) in corpus() {
        let off = translate_with(&code, false).hlsl;
        let on = translate_with(&code, true).hlsl;
        for (form, hlsl) in [("fold", &off), ("wave operation", &on)] {
            xcl_dxil::compile(hlsl, "main", "ps_6_6").unwrap_or_else(|e| panic!("{name}, {form}: {e}\n{hlsl}"));
        }
        let (a, b) = (run_px(&off, 32, None), run_px(&on, 32, None));
        assert_eq!(a, b, "{name}: a full wave differs between the forms");
    }
}

#[test]
fn a_partial_wave_reduction_is_the_console_answer() {
    // One quad of a 32-lane wave: lanes 0, 1 and 2 run, lane 3 and the rest of the wave were
    // never launched. The value is 0x100 + lane, so the console's min over the launched lanes
    // is 0x100 (256) and the fold's read of an unlaunched lane's register (0) gives 0.
    let code = min_ps();
    let fold = run_px(&translate_with(&code, false).hlsl, 32, Some(3));
    let wave = run_px(&translate_with(&code, true).hlsl, 32, Some(3));
    println!("min, 3 of 32 lanes live, values 0x100+lane: fold {fold:?}, wave operation {wave:?}");
    assert_eq!([fold[&0], fold[&1], fold[&2]], [0.0; 3], "the fold reads an unlaunched lane");
    assert_eq!([wave[&0], wave[&1], wave[&2]], [256.0; 3], "the wave operation is the console's min");
    // a full wave: both forms are the console's answer (lanes 0..31 hold 0x100..0x11F)
    for (on, form) in [(false, "fold"), (true, "wave operation")] {
        let px = run_px(&translate_with(&code, on).hlsl, 32, None);
        assert_eq!(px.len(), 32, "32 lanes export ({form})");
        assert_eq!(px[&1], 256.0, "a full wave ({form})");
    }
    // max of signed values, identity i32::MIN: -(0x100 + lane), so the console's max is
    // -0x100 (-256) and the fold's read of the unlaunched lane (0) gives 0
    let code = ps(true, "V_CVT_F32_I32", "V_MAX_I32", 255, Some(i32::MIN as u32), Mask::Wave, false, true);
    let fold = run_px(&translate_with(&code, false).hlsl, 32, Some(3));
    let wave = run_px(&translate_with(&code, true).hlsl, 32, Some(3));
    println!("max, 3 of 32 lanes live, values -(0x100+lane): fold {fold:?}, wave operation {wave:?}");
    assert_eq!([fold[&0], fold[&1], fold[&2]], [0.0; 3], "the fold reads an unlaunched lane");
    assert_eq!([wave[&0], wave[&1], wave[&2]], [-256.0; 3], "the wave operation is the console's max");
    // a full wave, max: lane 0's -0x100 is the largest
    for (on, form) in [(false, "fold"), (true, "wave operation")] {
        let px = run_px(&translate_with(&code, on).hlsl, 32, None);
        assert_eq!(px[&0], -256.0, "a full wave ({form})");
    }
}
