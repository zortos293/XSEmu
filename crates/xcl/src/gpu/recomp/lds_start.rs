//! The group's LDS starts as the program's first store leaves it (`XCL_LDS_START`).
//!
//! Ported from the emulator work that made the title's tiled water light correct on 32-lane
//! cards: `fxceless/victory-fork` f299fba, `crates/victory-shader/src/emit/lower.rs`
//! (`lds_start`, `Lower::lds_start`), GPL-3.0; the idea's first form there is `a918163`
//! (`VICTORY_LDS_INIT_HOIST`). Provenance: THIRD_PARTY.md.
//!
//! The console does not clear a group's LDS: a group finds what an earlier wave left. A
//! program that sets its LDS up in one thread (`if (thread == 0) { nearest = ~0; farthest = 0; }`
//! behind its per-pixel early-outs: the title's tiled light) does not set it up in a group
//! whose thread 0 is not live (a tile whose first pixel is not water), and that group works
//! with what was left: on the console an earlier group's values, on the host whatever the
//! memory held, this translation's own exchange words among it. Every LDS dword the program
//! sets up that way is stored by every thread at the group's start instead, before any LDS
//! access and then a group barrier: the group then holds the program's own values whether or
//! not the setting-up thread got there.
//!
//! Exact: where the program's own store runs it stores the same value again before the
//! dword's first use in that order, and a read that would see the early store instead of a
//! later write of this group would have read memory this group never wrote (undefined on the
//! console). A dword whose first access could be reached without passing the store, or a
//! store whose address or value is not a constant, is not set up and keeps today's behaviour.

use std::collections::BTreeMap;

use super::Term;
use crate::gpu::rdna2::{self, Decoded};

/// `XCL_LDS_START=1`: the group's LDS starts as the program's first store leaves it (off by
/// default: the host's memory, as before). Read per program, so a shader cache entry written
/// with the switch on is only reused with it on.
pub(super) fn lds_start_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_LDS_START.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_LDS_START").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`lds_start_on`], for its thread only.
    pub(super) static TEST_LDS_START: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The (dword, value) a group's LDS is set to before the program runs: the plain store of
/// constants at a constant address that is the program's first LDS access, when its block is
/// on the way to every other LDS access. Empty without such a store.
pub(super) fn lds_start(insts: &[Decoded], leaders: &[u64], terms: &[Term]) -> Vec<(u32, u32)> {
    // a block whose end the graph cannot follow makes "on the way to" unprovable
    if leaders.is_empty() || terms.iter().any(|t| matches!(t, Term::Other)) {
        return Vec::new();
    }
    let block_of = |pc: u64| leaders.partition_point(|&l| l <= pc).saturating_sub(1);
    let is_lds = |d: &Decoded| {
        d.enc.name == "ENC_DS"
            && super::emit::is_lds(&d.inst.name.to_ascii_lowercase())
            && rdna2::field_value(d.enc, &d.words, "GDS") != Some(1)
    };
    // the program's first LDS access, and the blocks of all of them
    let mut first: Option<&Decoded> = None;
    let mut blocks: Vec<usize> = Vec::new();
    for d in insts.iter().filter(|d| is_lds(d)) {
        let b = block_of(d.pc);
        if blocks.last() != Some(&b) {
            blocks.push(b);
        }
        if first.is_none() {
            first = Some(d);
        }
    }
    let Some(store) = first else { return Vec::new() };
    // what the store writes, in (byte offset, data source) pairs; only a plain write of
    // constants counts (a read first, or anything else, leaves the dwords as they are)
    let field = |d: &Decoded, name: &str| rdna2::field_value(d.enc, &d.words, name);
    // the byte offsets the store writes and the registers their values come from
    let written: Option<(Vec<u32>, Vec<u64>)> = match store.inst.name {
        "DS_WRITE_B32" => {
            let offset = |name| field(store, name).unwrap_or(0);
            field(store, "DATA0").map(|data| (vec![(offset("OFFSET0") | offset("OFFSET1") << 8) as u32], vec![data]))
        }
        "DS_WRITE2_B32" => field(store, "DATA0").zip(field(store, "DATA1")).map(|(a, b)| {
            (vec![4 * field(store, "OFFSET0").unwrap_or(0) as u32, 4 * field(store, "OFFSET1").unwrap_or(0) as u32], vec![a, b])
        }),
        _ => None,
    };
    let Some((bytes, data)) = written else { return Vec::new() };
    let store_block = block_of(store.pc);
    for &b in &blocks {
        if b != store_block && !dominates(leaders, terms, store_block, b) {
            return Vec::new();
        }
    }
    // what the store's address and data registers hold: only plain moves of constants may have
    // run since the block started, so nothing else can have changed them
    let mut values: BTreeMap<u64, u32> = BTreeMap::new();
    let start = leaders[store_block];
    for d in insts.iter().take_while(|d| d.pc < store.pc).skip_while(|d| d.pc < start) {
        let enc = d.enc.name;
        // scalar code writes no VGPR (scalar memory loads would: they are not here), and a
        // compare writes VCC / EXEC (or a scalar SDST), not a VGPR
        if enc.starts_with("ENC_SOP") || enc.starts_with("SOP") || enc.starts_with("ENC_VOPC") || enc.starts_with("VOPC") {
            continue;
        }
        if d.inst.name == "V_NOP" {
            continue;
        }
        if d.inst.name == "V_MOV_B32" && matches!(enc, "ENC_VOP1" | "VOP1_INST_LITERAL") {
            let (Some(dst), Some(value)) = (field(d, "VDST"), constant(d)) else { return Vec::new() };
            values.insert(dst, value);
            continue;
        }
        return Vec::new();
    }
    let Some(base) = field(store, "ADDR").and_then(|r| values.get(&r).copied()) else { return Vec::new() };
    let base = base as u64;
    let mut out = Vec::new();
    for (byte, reg) in bytes.iter().zip(&data) {
        let (Some(value), Some(at)) = (values.get(reg).copied(), base.checked_add(u64::from(*byte))) else { return Vec::new() };
        if at % 4 != 0 {
            return Vec::new();
        }
        out.push(((at / 4) as u32, value));
    }
    out
}

/// The value a plain `v_mov_b32` moves: an inline integer constant or a literal.
fn constant(d: &Decoded) -> Option<u32> {
    let code = rdna2::field_value(d.enc, &d.words, "SRC0")?;
    Some(match code {
        128..=192 => (code - 128) as u32,
        193..=208 => (-(code as i64 - 192)) as u32,
        255 => d.literal?,
        _ => return None,
    })
}

/// Whether block `a` is on every path from the program's entry (block 0) to block `b`.
fn dominates(leaders: &[u64], terms: &[Term], a: usize, b: usize) -> bool {
    if a == 0 {
        return true;
    }
    let target = leaders.get(b).copied();
    let at = |pc: u64| {
        let y = leaders.partition_point(|&l| l < pc);
        (y < leaders.len() && leaders[y] == pc).then_some(y)
    };
    let mut seen = vec![false; leaders.len()];
    let mut stack = vec![0usize];
    while let Some(x) = stack.pop() {
        if std::mem::replace(&mut seen[x], true) || x == a {
            continue;
        }
        if Some(leaders[x]) == target {
            return false;
        }
        match terms.get(x) {
            Some(Term::Cond(t)) => {
                stack.extend(at(*t));
                if x + 1 < leaders.len() {
                    stack.push(x + 1);
                }
            }
            Some(Term::Jump(t)) => stack.extend(at(*t)),
            Some(Term::Next) if x + 1 < leaders.len() => stack.push(x + 1),
            _ => {}
        }
    }
    true
}
