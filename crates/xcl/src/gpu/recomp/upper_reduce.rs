//! A whole-wave reduction as a wave operation (`XCL_UPPER_LANES_REDUCTION=1`).
//!
//! The console runs whole-wave code in all 64 lanes, the inactive ones included. A reduction
//! in a wave64 vertex or pixel shader reaches the program as: `s_or_saveexec_b64 s, -1`; a
//! mask under the saved EXEC; a `v_cndmask` of the operation's identity with it, which puts
//! the identity into every lane outside the mask; then a fold of the 32 lanes of each half
//! (DPP8 / DPP16 xor 1, 2 and 4, then row_mirror), the other half through a
//! `v_permlanex16`, and two `v_readlane`s – lane 32 for the upper half, lane 0 for the
//! total. `docs/wave64-on-wave32.md` ("What the retail pack does") counts 1 082 of these
//! reads in the pack.
//!
//! `upper_lanes.rs` gives the lanes ≥ 32 of a 32-lane host wave the identity, which is right
//! (they are all missing, and all hold the identity). A lane the host wave did **not**
//! launch – a primitive's edge, down to one quad of a wave – is missing among 0..31, and the
//! transcribed fold reads it anyway: its register holds what an earlier wave left there, so
//! a min reduction (identity -1, `v_min_u32`) reads 0 where the console has the identity and
//! the picture goes black at the edge.
//!
//! With the switch on, the fold is not transcribed at the read: the seed's value (the
//! mask's constant, in every lane) is reduced over the lanes the wave runs, as
//! `WaveActiveMin` / `WaveActiveMax` / `WaveActiveBitOr` / … D3D's wave operations run over
//! the active lanes only, and in a pixel shader helper lanes are not active ones
//! (`DirectX-Specs`, `HLSL_ShaderModel6_6`, "helper lanes do not participate"), so the total
//! is exactly the console's: the mask's lanes hold their real values, every lane outside the
//! mask holds the identity, and a lane that was not launched is in neither set.
//!
//! Two facts make that exact, and `upper_lanes.rs` proves both before this module fires:
//! - the mask holds only lanes the wave launched (`State::subset`, `UpperLanes::outside_const`),
//!   so a lane that was not launched is outside it and its console value is the constant;
//! - the constant is the operation's identity (`Op::identity`), so the lanes the console's
//!   half has and the host wave does not (32..63) join the total without changing it.
//!
//! Not folded here: a reduction whose mask can hold a lane ≥ 32 (its constant is not the
//! read's), one whose fold is not the pack's shape, and one whose result is only ever read
//! from the folded register rather than through `v_readlane` – those keep the transcribed
//! form. With the switch off, nothing in this module runs and the emitted HLSL is the old
//! one byte for byte.

use std::collections::{BTreeMap, BTreeSet};

use super::emit::ext_encoding;
use super::upper_lanes::UpperLanes;
use crate::gpu::rdna2::{self, Decoded, Variant};

/// `XCL_UPPER_LANES_REDUCTION=1`: a whole-wave reduction is a wave operation over the lanes
/// the wave runs (off by default: the transcribed DPP fold). Read once per process; a test
/// sets it for its own thread.
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_ON.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_UPPER_LANES_REDUCTION").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`on`], for its thread only.
    pub(super) static TEST_ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The operation a reduction applies to its lanes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Op {
    MinU32,
    MaxU32,
    MinI32,
    MaxI32,
    OrB32,
    AndB32,
    XorB32,
    AddU32,
}

impl Op {
    /// The instruction that applies it in the pack's fold.
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            "V_MIN_U32" => Op::MinU32,
            "V_MAX_U32" => Op::MaxU32,
            "V_MIN_I32" => Op::MinI32,
            "V_MAX_I32" => Op::MaxI32,
            "V_OR_B32" => Op::OrB32,
            "V_AND_B32" => Op::AndB32,
            "V_XOR_B32" => Op::XorB32,
            "V_ADD_NC_U32" => Op::AddU32,
            _ => return None,
        })
    }

    /// What the operation leaves a total as when a lane holding this joins it: the constant
    /// the shader installs outside its mask.
    fn identity(self) -> u32 {
        match self {
            Op::MinU32 | Op::AndB32 => u32::MAX,
            Op::MaxU32 | Op::OrB32 | Op::XorB32 | Op::AddU32 => 0,
            Op::MinI32 => i32::MAX as u32,
            Op::MaxI32 => i32::MIN as u32,
        }
    }

    /// The same reduction over the wave's active lanes, in HLSL.
    pub(super) fn expr(self, x: &str) -> String {
        match self {
            Op::MinU32 => format!("WaveActiveMin({x})"),
            Op::MaxU32 => format!("WaveActiveMax({x})"),
            Op::MinI32 => format!("asuint(WaveActiveMin(asint({x})))"),
            Op::MaxI32 => format!("asuint(WaveActiveMax(asint({x})))"),
            Op::OrB32 => format!("WaveActiveBitOr({x})"),
            Op::AndB32 => format!("WaveActiveBitAnd({x})"),
            Op::XorB32 => format!("WaveActiveBitXor({x})"),
            Op::AddU32 => format!("WaveActiveSum({x})"),
        }
    }
}

/// One recognised reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Reduction {
    /// where its seed (the `v_cndmask`) is: the value the wave operation reduces is copied
    /// here, before the fold overwrites it
    pub pc: u64,
    /// the VGPR the seed writes
    pub vgpr: u64,
    pub op: Op,
}

/// A program's reductions, by seed and by the read each one is consumed at.
#[derive(Clone, Default, Debug)]
pub(super) struct Plan {
    /// in program order; the emitter names each one's copy `xcl_wre{i}`
    pub reds: Vec<Reduction>,
    /// seed pc → index in `reds`
    pub seeds: BTreeMap<u64, usize>,
    /// pc of the `v_readlane` that reads lane 0 of the folded register → index in `reds`
    pub sites: BTreeMap<u64, usize>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.reds.is_empty()
    }

    /// The name of reduction `i`'s copy of its seed value.
    pub fn name(&self, i: usize) -> String {
        format!("xcl_wre{i}")
    }
}

/// A VALU source operand: a VGPR or an SGPR word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reg {
    V(u64),
    S(u64),
}

impl Reg {
    /// The register a 9-bit source operand names (an inline constant or a literal: none).
    fn of(code: u64) -> Option<Reg> {
        match code {
            256..=511 => Some(Reg::V(code - 256)),
            0..=124 | 126 | 127 => Some(Reg::S(code)),
            _ => None,
        }
    }

    /// A VGPR source operand, when it is one.
    fn vgpr(code: u64) -> Option<u64> {
        match Reg::of(code)? {
            Reg::V(v) => Some(v),
            Reg::S(_) => None,
        }
    }
}

/// The VGPR an instruction writes.
fn vdst(d: &Decoded) -> Option<u64> {
    rdna2::field_value(d.enc, &d.words, "VDST")
}

/// An instruction's two VALU sources (VOP2: SRC0 / VSRC1, VOP3: SRC0 / SRC1), with the VOP3
/// modifiers required to be none.
fn sources(d: &Decoded) -> Option<(Reg, Reg)> {
    let f = |n: &str| rdna2::field_value(d.enc, &d.words, n);
    match d.enc.name {
        "ENC_VOP2" => Some((Reg::of(f("SRC0")?)?, Reg::of(256 + f("VSRC1")?)?)),
        "ENC_VOP3" => {
            let x = |n: &str| f(n).unwrap_or(0);
            (x("ABS") == 0 && x("NEG") == 0 && x("CLAMP") == 0 && x("OMOD") == 0).then_some(())?;
            Some((Reg::of(f("SRC0")?)?, Reg::of(f("SRC1")?)?))
        }
        _ => None,
    }
}

/// A DPP step of the fold: `v = op(dpp(v, lane ^ n), v)` over all 32 lanes, `n` in 1, 2, 4.
fn dpp_xor(d: &Decoded) -> Option<u32> {
    if d.variant != Variant::Dpp8 && d.variant != Variant::Dpp16 {
        return None;
    }
    let e = ext_encoding(d)?;
    let x = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
    let v = vdst(d)?;
    if x("VSRC0") != v || x("VSRC1") != v {
        return None;
    }
    let n = match d.variant {
        // lane i of each 8-lane group reads lane i ^ n
        Variant::Dpp8 => {
            let sels: BTreeSet<u32> = (0..8u32).map(|i| i ^ x(&format!("LANE_SEL_{i}")) as u32).collect();
            (sels.len() == 1).then(|| *sels.first().unwrap())?
        }
        // row_xor: DPP_CTRL 0x160 | n, every lane of the row written
        _ => {
            let (ctrl, row, bank) = (x("DPP_CTRL"), x("ROW_MASK"), x("BANK_MASK"));
            if !(0x160..=0x16F).contains(&ctrl) || row != 0xF || bank != 0xF {
                return None;
            }
            (ctrl & 15) as u32
        }
    };
    [1, 2, 4].contains(&n).then_some(n)
}

/// The row mirror of the fold: `v = op(v[mirror(lane)], v)` over all 32 lanes.
fn is_row_mirror(d: &Decoded) -> bool {
    if d.variant != Variant::Dpp16 {
        return false;
    }
    let Some(e) = ext_encoding(d) else { return false };
    let x = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
    vdst(d).is_some_and(|v| x("VSRC0") == v && x("VSRC1") == v && x("DPP_CTRL") == 0x140 && x("ROW_MASK") == 0xF && x("BANK_MASK") == 0xF)
}

/// `v_permlanex16 dst, v`: every lane takes lane 0 of the other 16-lane row.
fn permlanex16(d: &Decoded) -> Option<(u64, u64)> {
    if d.inst.name != "V_PERMLANEX16_B32" || d.enc.name != "ENC_VOP3" {
        return None;
    }
    let f = |n: &str| rdna2::field_value(d.enc, &d.words, n);
    let x = |n: &str| f(n).unwrap_or(0);
    if x("ABS") != 0 || x("NEG") != 0 || x("CLAMP") != 0 || x("OMOD") != 0 || f("SRC1")? != 128 || f("SRC2")? != 128 {
        return None;
    }
    Some((vdst(d)?, Reg::vgpr(f("SRC0")?)?))
}

/// `v = op(a, b)` (either order) into VGPR `v`.
fn combine(d: &Decoded, op: Op, v: Reg, other: Reg) -> bool {
    if d.variant != Variant::Plain || Op::of(d.inst.name) != Some(op) || vdst(d).map(Reg::V) != Some(v) {
        return false;
    }
    matches!(sources(d), Some((a, b)) if (a == v && b == other) || (a == other && b == v))
}

/// `dst, v[lane]` of a `v_readlane` with a constant lane: the destination is the scalar the
/// emitter writes (the instruction names a VGPR or an SGPR, and the translation keeps it an
/// SGPR – the reduction's total is the wave's).
fn readlane(d: &Decoded, lane: u64) -> Option<(Reg, Reg)> {
    if d.inst.name != "V_READLANE_B32" {
        return None;
    }
    let f = |n: &str| rdna2::field_value(d.enc, &d.words, n);
    (f("SRC1")? == 128 + lane).then_some(())?;
    let dst = vdst(d)?;
    Some((Reg::S((dst < 128).then_some(dst)?), Reg::of(f("SRC0")?)?))
}

/// Recognise the reduction that starts at `insts[i]`, with the index of the instruction
/// after it.
fn recognise(insts: &[Decoded], i: usize, upper: &UpperLanes, leaders: &BTreeSet<u64>) -> Option<(Reduction, usize)> {
    let seed = &insts[i];
    // the seed: the operation's identity into every lane its mask leaves out, the mask under
    // the saved EXEC (`upper_lanes` proves it holds only launched lanes)
    let constant = *upper.outside_const.get(&seed.pc)?;
    // the fold's operation: the first step belongs to it
    let op = Op::of(&insts.get(i + 1)?.inst.name)?;
    if constant != op.identity() {
        return None;
    }
    let v = vdst(seed)?;
    let mut k = i + 1;
    let mut xors = 0u32;
    while let Some(n) = dpp_xor(insts.get(k)?) {
        if vdst(&insts[k]) != Some(v) {
            return None;
        }
        xors |= n;
        k += 1;
    }
    // the pack's fold: xor 1, 2 and 4 (the 8-lane group's total), then the row mirror
    if xors != 7 || !is_row_mirror(insts.get(k)?) {
        return None;
    }
    k += 1;
    let (t, src) = permlanex16(insts.get(k)?)?;
    if src != v || t == v || !combine(insts.get(k + 1)?, op, Reg::V(v), Reg::V(t)) {
        return None;
    }
    k += 2;
    // the upper half's total (`v_readlane` 32), the identity on a 32-lane wave: optional
    if let Some((r, src)) = readlane(insts.get(k)?, 32) {
        if src == Reg::V(v) && combine(insts.get(k + 1)?, op, Reg::V(v), r) {
            k += 2;
        }
    }
    let (dst, src) = readlane(insts.get(k)?, 0)?;
    if src != Reg::V(v) || dst == Reg::V(v) {
        return None;
    }
    // a path that enters the fold in its middle would not have taken the seed's copy
    if insts[i + 1..=k].iter().any(|d| leaders.contains(&d.pc)) {
        return None;
    }
    Some((Reduction { pc: seed.pc, vgpr: v, op }, k + 1))
}

/// Reductions folded since the process started: the first is logged (the switch engaged),
/// then every power of ten (`nan_scrub`'s pattern).
pub(super) fn note(n: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static FOLDED: AtomicU64 = AtomicU64::new(0);
    let before = FOLDED.fetch_add(n as u64, Ordering::Relaxed);
    let after = before + n as u64;
    if before == 0 || (after > 0 && after.ilog10() != before.ilog10()) {
        crate::xlog_info!("[SHADER] XCL_UPPER_LANES_REDUCTION: {after} whole-wave reduction(s) as wave operations over the lanes the wave runs (unset XCL_UPPER_LANES_REDUCTION to turn it off)");
    }
}

/// The reductions of a program (`Options::upper_reduce`).
pub(super) fn plan(insts: &[Decoded], upper: &UpperLanes, leaders: &[u64]) -> Plan {
    let leaders: BTreeSet<u64> = leaders.iter().copied().collect();
    let mut out = Plan::default();
    let mut i = 0;
    while i < insts.len() {
        match recognise(insts, i, upper, &leaders) {
            Some((r, next)) => {
                out.seeds.insert(r.pc, out.reds.len());
                out.sites.insert(insts[next - 1].pc, out.reds.len());
                out.reds.push(r);
                i = next;
            }
            None => i += 1,
        }
    }
    out
}
