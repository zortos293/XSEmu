//! Page resolutions kept across blocks (`emit::keep_pages`): the plan.
//!
//! A page group (`emit::PageGroup`) lives in one block of the emitted program, so a base that
//! many blocks load from – a program's descriptor table, read hundreds of times through two
//! registers it writes a dozen times – is resolved in each. A kept resolution is a group in
//! variables of the whole program instead: resolved after each instruction that may write the
//! base's registers (and at the start, where a load may read what the program began with), and
//! read by every load those are the only writes to reach. Which those are is a question about
//! the instructions' graph (`x64_masks::graph`, the one the register liveness uses), asked here
//! once the loads are known: a first translation writes them down (`Record`), a second one
//! follows the `Plan`.

use super::x64_masks::{self, GuardExits, Regs};
use crate::gpu::rdna2::Decoded;
use std::collections::{BTreeMap, BTreeSet};

/// A scalar load of the first translation: its instruction, its group's key and base registers
/// (`Em::page_group`), the bytes `[imm, imm + span)` it reads from the base, and the block of
/// the emitted text it is in.
pub(super) struct Site {
    pub pc: u64,
    pub key: String,
    pub regs: Regs,
    pub imm: i64,
    pub span: i64,
    pub block: usize,
}

/// What the first translation found: the loads, the instructions no text can follow (a branch,
/// or one whose text leaves its block early), and where the loops' guards leave to.
#[derive(Default)]
pub(super) struct Record {
    pub sites: Vec<Site>,
    pub closed: BTreeSet<u64>,
    pub guard: GuardExits,
}

/// The kept resolutions of a program: each one's key and anchor offset (`keys`), the loads
/// that read through one (`sites`, by instruction and key), and where each is resolved: after
/// the instructions of `after`, and at the start of the program (`entry`).
#[derive(Default)]
pub(super) struct Plan {
    pub keys: Vec<(String, i64)>,
    pub sites: BTreeMap<(u64, String), usize>,
    pub after: BTreeMap<u64, Vec<usize>>,
    pub entry: Vec<usize>,
}

/// The writes of a base's registers that reach each instruction: bit 0 is the program's start,
/// bit k + 1 the k-th instruction that may write one of the base's registers (`at`, their
/// places in `insts`). A write is the last one on its path whether it happened or not, since
/// the resolution after it reads the registers as they are. The sets are `words` words an
/// instruction, one after the other. None: the graph has a jump whose target is not known.
fn reaching(insts: &[Decoded], succ: &[Option<Vec<usize>>], exits: &[Vec<Option<usize>>], at: &[usize]) -> Option<(Vec<u64>, usize)> {
    let words = (at.len() + 1).div_ceil(64);
    let mut before = vec![0u64; insts.len() * words];
    if insts.is_empty() {
        return Some((before, words));
    }
    before[0] |= 1;
    let mut write = vec![0usize; insts.len()];
    for (k, &i) in at.iter().enumerate() {
        write[i] = k + 1;
    }
    // every instruction is read once, and again when what reaches it has grown: a program has
    // hundreds of bases and tens of thousands of instructions, nearly all of them read once
    let mut due = vec![true; insts.len()];
    let mut out = vec![0u64; words];
    let mut join = |before: &mut [u64], due: &mut [bool], to: usize, from: &[u64]| {
        for (t, f) in before[to * words..][..words].iter_mut().zip(from) {
            due[to] |= *t | *f != *t;
            *t |= *f;
        }
        due[to]
    };
    loop {
        let mut again = false;
        for i in 0..insts.len() {
            if !std::mem::take(&mut due[i]) {
                continue;
            }
            out.copy_from_slice(&before[i * words..][..words]);
            // a loop's guard leaves before the instruction
            for &to in exits[i].iter().flatten() {
                again |= join(&mut before, &mut due, to, &out) && to <= i;
            }
            if write[i] != 0 {
                out.fill(0);
                out[write[i] / 64] |= 1 << (write[i] % 64);
            }
            for &to in succ[i].as_ref()? {
                again |= join(&mut before, &mut due, to, &out) && to <= i;
            }
        }
        if !again {
            return Some((before, words));
        }
    }
}

/// The plan for the loads of `record`. A key's anchor is its lowest offset, and its loads are
/// those whose bytes lie in the 64 KiB from there on, a multiple of four bytes away (the group's
/// rule). Loads with the same reaching writes are served together when that takes fewer
/// resolutions than the blocks they are in – what the groups of today cost – or none at all;
/// a write no text can follow serves no load, and a load no write reaches by the graph is left
/// alone.
pub(super) fn plan(insts: &[Decoded], record: &Record) -> Plan {
    let mut plan = Plan::default();
    let (succ, exits) = x64_masks::graph(insts, &record.guard);
    let index: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let written: Vec<Regs> = insts.iter().map(x64_masks::may_write).collect();
    let mut by_key: BTreeMap<&str, Vec<&Site>> = BTreeMap::new();
    for s in &record.sites {
        by_key.entry(&s.key).or_default().push(s);
    }
    for (key, sites) in by_key {
        // (a key spells out every register its address reads)
        let regs = sites[0].regs;
        debug_assert!(sites.iter().all(|s| s.regs == regs));
        let anchor = sites.iter().map(|s| s.imm).min().unwrap();
        let sites: Vec<&Site> = sites.into_iter().filter(|s| (s.imm - anchor) % 4 == 0 && s.imm - anchor + s.span <= 65536 && index.contains_key(&s.pc)).collect();
        if sites.iter().map(|s| s.block).collect::<BTreeSet<_>>().len() < 2 {
            continue;
        }
        let at: Vec<usize> = (0..insts.len()).filter(|&i| written[i] & regs != 0).collect();
        let Some((before, words)) = reaching(insts, &succ, &exits, &at) else { return Plan::default() };
        let open = |bit: usize| bit == 0 || !record.closed.contains(&insts[at[bit - 1]].pc);
        let bits = |set: &[u64]| (0..=at.len()).filter(|&b| set[b / 64] >> (b % 64) & 1 != 0).collect::<Vec<_>>();
        // loads by their reaching writes, with the blocks they are in
        let mut classes: BTreeMap<Vec<usize>, (BTreeSet<usize>, Vec<&Site>)> = BTreeMap::new();
        for s in sites {
            let class = classes.entry(bits(&before[index[&s.pc] * words..][..words])).or_default();
            class.0.insert(s.block);
            class.1.push(s);
        }
        let mut resolved = BTreeSet::new();
        let mut served = Vec::new();
        loop {
            let mut changed = false;
            classes.retain(|writes, (blocks, sites)| {
                let new = writes.iter().filter(|b| !resolved.contains(*b)).count();
                // (no write at all: the graph has no path to the load, and neither does the plan)
                if writes.is_empty() || !writes.iter().all(|&b| open(b)) || new >= blocks.len() && new != 0 {
                    return true;
                }
                resolved.extend(writes.iter().copied());
                served.append(sites);
                changed = true;
                false
            });
            if !changed {
                break;
            }
        }
        if served.is_empty() {
            continue;
        }
        let n = plan.keys.len();
        plan.keys.push((key.to_string(), anchor));
        for s in served {
            plan.sites.insert((s.pc, key.to_string()), n);
        }
        for bit in resolved {
            match bit {
                0 => plan.entry.push(n),
                _ => plan.after.entry(insts[at[bit - 1]].pc).or_default().push(n),
            }
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::{self, encode};

    /// A program of the plan's tests: `site` stands for a scalar load of the base s0:s1 (the
    /// plan asks nothing else of it), `write` writes s0, `other` writes a register that is not
    /// the base's.
    enum A {
        Site,
        Write,
        Other,
        L(&'static str),
        B(&'static str, &'static str),
    }

    const KEY: &str = "smem|s1|s0|0u";

    /// The decoded instructions and the pc of each item (a label's is that of what follows).
    fn program(items: &[A]) -> (Vec<Decoded>, Vec<u64>) {
        let bytes = |a: &A| match a {
            A::Site => encode("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap(),
            A::Write => encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 131)], None).unwrap(),
            A::Other => encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 131)], None).unwrap(),
            A::L(_) => Vec::new(),
            A::B(n, _) => encode(n, "ENC_SOPP", &[("SIMM16", 0)], None).unwrap(),
        };
        let mut at = Vec::new();
        let mut pc = 0u64;
        for a in items {
            at.push(pc);
            pc += bytes(a).len() as u64;
        }
        let label = |l: &str| at[items.iter().position(|a| matches!(a, A::L(x) if *x == l)).unwrap()];
        let mut code = Vec::new();
        for (a, &pc) in items.iter().zip(&at) {
            match a {
                A::B(n, l) => code.extend(encode(n, "ENC_SOPP", &[("SIMM16", ((label(l) as i64 - (pc as i64 + 4)) / 4) as i16 as u16 as u64)], None).unwrap()),
                a => code.extend(bytes(a)),
            }
        }
        let mut insts = Vec::new();
        let mut off = 0;
        while off < code.len() {
            let d = rdna2::decode(&code[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        (insts, at)
    }

    /// The record of a first translation of `items`: the item `k` of `sites` is a load of
    /// `[imm, imm + 4)` in block `block`; no text can follow the items of `closed`.
    fn record(insts: &[Decoded], at: &[u64], sites: &[(usize, i64, usize)], closed: &[usize]) -> Record {
        Record {
            sites: sites.iter().map(|&(k, imm, block)| Site { pc: at[k], key: KEY.into(), regs: 3, imm, span: 4, block }).collect(),
            closed: closed.iter().map(|&k| at[k]).collect(),
            guard: x64_masks::range_exits(insts),
        }
    }

    /// (the served items, the items resolved after, whether the start resolves) of the plan
    fn planned(items: &[A], sites: &[(usize, i64, usize)], closed: &[usize]) -> (Vec<usize>, Vec<usize>, bool) {
        let (insts, at) = program(items);
        let plan = plan(&insts, &record(&insts, &at, sites, closed));
        let item = |pc: u64| at.iter().position(|&p| p == pc).unwrap();
        // (a label has the pc of the item after it: the tests name the items, never the labels)
        let item = |pc: u64| (item(pc)..).find(|&k| !matches!(items[k], A::L(_))).unwrap();
        (plan.sites.keys().map(|(pc, _)| item(*pc)).collect(), plan.after.keys().map(|&pc| item(pc)).collect(), !plan.entry.is_empty())
    }

    /// A base the program never writes, read in two blocks, is resolved once, at the start; read
    /// in one block it is left to that block's group.
    #[test]
    fn an_unwritten_base_is_resolved_at_the_start() {
        let p = [A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::Other, A::B("S_ENDPGM", "end")];
        assert_eq!(planned(&p, &[(0, 0, 0), (2, 8, 1)], &[]), (vec![0, 2], vec![], true));
        assert_eq!(planned(&p, &[(0, 0, 0), (2, 8, 0)], &[]), (vec![], vec![], false));
        // a write of another register changes nothing
        assert_eq!(planned(&[A::Site, A::Other, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")], &[(0, 0, 0), (3, 8, 1)], &[]), (vec![0, 3], vec![], true));
    }

    /// The anchor is the lowest offset: a load past its 64 KiB, or not a multiple of four bytes
    /// from it, is not served.
    #[test]
    fn loads_outside_the_anchors_pages_are_not_served() {
        let p = [A::Site, A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let (insts, at) = program(&p);
        let plan = plan(&insts, &record(&insts, &at, &[(0, -8, 0), (1, 65536 - 12, 1), (3, 65536 - 8, 2), (4, 2, 2)], &[]));
        assert_eq!(plan.keys, [(KEY.to_string(), -8)]);
        assert_eq!(plan.sites.keys().map(|(pc, _)| *pc).collect::<Vec<_>>(), [at[0], at[1]]);
    }

    /// A write of the base is followed by a resolution where the loads it reaches are in more
    /// blocks than that costs; the loads before it read the resolution of the start.
    #[test]
    fn a_write_is_resolved_after_where_it_pays() {
        // blocks: 0 (start), 1 (after the write, one side of a branch), 2 (the other side)
        let p = [A::Site, A::B("S_CBRANCH_SCC1", "skip"), A::Site, A::L("skip"), A::Write, A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        assert_eq!(planned(&p, &[(0, 0, 0), (2, 0, 1), (5, 0, 2), (7, 0, 3)], &[]), (vec![0, 2, 5, 7], vec![4], true));
        // loads after the write in one block only: that block's group is as good
        assert_eq!(planned(&p, &[(0, 0, 0), (2, 0, 1), (5, 0, 2), (7, 0, 2)], &[]), (vec![0, 2], vec![], true));
    }

    /// No text can follow a branch or an instruction whose text leaves its block: such a write
    /// serves no load, and neither does the start for the loads the write reaches too. The loads
    /// only other writes reach are served as before.
    #[test]
    fn a_write_no_text_can_follow_serves_no_load() {
        let p = [A::Site, A::B("S_CBRANCH_SCC1", "skip"), A::Site, A::Write, A::L("skip"), A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let sites = [(0, 0, 0), (2, 0, 1), (5, 0, 2), (7, 0, 3)];
        // open: the loads after the join are reached by the start and by the write, and read
        // what was resolved last
        assert_eq!(planned(&p, &sites, &[]), (vec![0, 2, 5, 7], vec![3], true));
        // closed: only the loads the write does not reach
        assert_eq!(planned(&p, &sites, &[3]), (vec![0, 2], vec![], true));
    }

    /// A write in a loop reaches the loop's loads of the next turn and the loads after it, by
    /// the branch back and by the loop guard's exit.
    #[test]
    fn a_write_in_a_loop_reaches_the_next_turn_and_the_exit() {
        let p = [A::Site, A::L("turn"), A::Site, A::Write, A::B("S_CBRANCH_SCC1", "turn"), A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let sites = [(0, 0, 0), (2, 0, 1), (5, 0, 2), (7, 0, 3)];
        // the loads after the loop: the write alone; the load in it: the start too
        assert_eq!(planned(&p, &sites, &[]), (vec![0, 2, 5, 7], vec![3], true));
        // a write no text can follow leaves the load before the loop alone in its block
        assert_eq!(planned(&p, &sites, &[3]), (vec![], vec![], false));
    }

    /// The loads after a loop its branch never leaves are reached through the loop guard's exit
    /// alone (`x64_masks::GuardExits`): by the write in the loop, and by the start where the guard
    /// leaves before the write. Without the exit the graph has no path to them, and they are
    /// left to their blocks.
    #[test]
    fn a_write_reaches_a_load_through_a_loop_guards_exit() {
        let p = [A::Site, A::L("turn"), A::Write, A::B("S_BRANCH", "turn"), A::Site, A::B("S_CBRANCH_SCC1", "b"), A::Site, A::L("b"), A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let sites = [(0, 0, 0), (4, 0, 1), (6, 0, 2), (9, 0, 3)];
        assert_eq!(planned(&p, &sites, &[]), (vec![0, 4, 6, 9], vec![2], true));
        let (insts, at) = program(&p);
        let mut rec = record(&insts, &at, &sites, &[]);
        assert_eq!(rec.guard.len(), 1, "the loop's guard");
        rec.guard.clear();
        assert!(plan(&insts, &rec).keys.is_empty());
        // with the write closed, the loads the guard's exit brings it to are not served either
        assert_eq!(planned(&p, &sites, &[2]), (vec![], vec![], false));
    }

    /// A call writes its link register and jumps: where that register is a base's, the emitter
    /// finds no text can follow it (the first translation's `closed`), and the loads it reaches
    /// stay with their blocks.
    #[test]
    fn a_call_that_writes_the_base_serves_no_load() {
        let p = [A::Site, A::B("S_CBRANCH_SCC1", "skip"), A::Site, A::L("skip"), A::Other, A::L("to"), A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let (mut insts, at) = program(&p);
        let sites = [(0, 0, 0), (2, 0, 1), (6, 0, 2), (8, 0, 3)];
        // (the write of another register in its place: every load is served from the start)
        assert_eq!(plan(&insts, &record(&insts, &at, &sites, &[])).sites.len(), 4);
        let call = encode("S_CALL_B64", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 0)], None).unwrap();
        insts[3] = rdna2::decode(&call, at[4]).unwrap();
        assert_ne!(x64_masks::may_write(&insts[3]) & 3, 0, "the call writes s0:s1");
        let plan = plan(&insts, &record(&insts, &at, &sites, &[4]));
        assert_eq!(plan.sites.keys().map(|(pc, _)| *pc).collect::<Vec<_>>(), [at[0], at[2]]);
        assert!(plan.after.is_empty() && plan.entry == [0]);
    }

    /// A jump whose target is not known may go anywhere: nothing is kept.
    #[test]
    fn an_unknown_jump_keeps_nothing() {
        let p = [A::Site, A::B("S_CBRANCH_SCC1", "end"), A::Site, A::L("end"), A::B("S_ENDPGM", "end")];
        let (mut insts, at) = program(&p);
        let rec = record(&insts, &at, &[(0, 0, 0), (2, 0, 1)], &[]);
        assert_eq!(plan(&insts, &rec).keys.len(), 1);
        let jump = encode("S_SETPC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None).unwrap();
        let pc = insts.last().unwrap().pc;
        *insts.last_mut().unwrap() = rdna2::decode(&jump, pc).unwrap();
        assert!(plan(&insts, &rec).keys.is_empty());
    }
}
