//! Which vector writes keep their lane guard (`v = select(xcl_on, x, v)`: a lane whose EXEC
//! bit is clear keeps its old value). A guard can go where no lane can tell the difference:
//!
//! * [`full_exec`]: EXEC provably holds every lane the wave started with, so every lane's bit is
//!   set and the guard is `true`, for every guard of the instruction (memory included);
//! * [`loose_writes`]: what a lane outside EXEC would keep is never read again, so the write
//!   needs no guard: until EXEC next changes or the code branches only lane-local
//!   instructions read the register (in their own lane, inside EXEC), and after that it is
//!   dead (overwritten before any read by an instruction that writes every lane).
//!
//! Both are exact: the program computes the same values in every lane whose result can be
//! read. Anything the analyses cannot follow (an indirect jump, indexed registers) keeps
//! every guard.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::x64_masks::{graph, sgpr_use, GuardExits, ALL, EXEC};
use crate::gpu::rdna2::{self, Decoded, Variant};

/// Lane predicates as reduced ordered BDDs: node 0 is false, 1 true, every other node tests a
/// variable (an atom: a mask value the analysis does not follow, named by where it was
/// made). Equal predicates are the same node.
pub(super) struct Bdd {
    pub(super) nodes: Vec<(u32, u32, u32)>,
    /// lowest and highest variable under each node (a quick test for `usesrange`)
    span: Vec<(u32, u32)>,
    unique: HashMap<(u32, u32, u32), u32>,
    memo: HashMap<(u8, u32, u32), u32>,
}

const NODE_LIMIT: usize = 1 << 18;

impl Bdd {
    pub(super) fn new() -> Bdd {
        Bdd { nodes: vec![(u32::MAX, 0, 0), (u32::MAX, 1, 1)], span: vec![(u32::MAX, 0), (u32::MAX, 0)], unique: HashMap::new(), memo: HashMap::new() }
    }

    pub(super) fn mk(&mut self, v: u32, lo: u32, hi: u32) -> u32 {
        if lo == hi {
            return lo;
        }
        if let Some(&n) = self.unique.get(&(v, lo, hi)) {
            return n;
        }
        let n = self.nodes.len() as u32;
        let (a, b) = (self.span[lo as usize], self.span[hi as usize]);
        self.nodes.push((v, lo, hi));
        self.span.push((v.min(a.0).min(b.0), v.max(if a.0 == u32::MAX { 0 } else { a.1 }).max(if b.0 == u32::MAX { 0 } else { b.1 })));
        self.unique.insert((v, lo, hi), n);
        n
    }

    pub(super) fn var(&mut self, v: u32) -> u32 {
        self.mk(v, 0, 1)
    }

    /// op 0: and, 1: or, 2: xor; None when the predicates grow past `NODE_LIMIT`
    pub(super) fn apply(&mut self, op: u8, a: u32, b: u32) -> Option<u32> {
        let (a, b) = (a.min(b), a.max(b));
        match (op, a, b) {
            (0, 0, _) => return Some(0),
            (0, 1, _) => return Some(b),
            (1, 0, _) => return Some(b),
            (1, 1, _) => return Some(1),
            (2, 0, _) => return Some(b),
            (0 | 1, _, _) if a == b => return Some(a),
            (2, _, _) if a == b => return Some(0),
            _ => {}
        }
        if let Some(&n) = self.memo.get(&(op, a, b)) {
            return Some(n);
        }
        if self.nodes.len() > NODE_LIMIT {
            return None;
        }
        let ((va, la, ha), (vb, lb, hb)) = (self.nodes[a as usize], self.nodes[b as usize]);
        let v = va.min(vb);
        let (la, ha) = if va == v { (la, ha) } else { (a, a) };
        let (lb, hb) = if vb == v { (lb, hb) } else { (b, b) };
        let lo = self.apply(op, la, lb)?;
        let hi = self.apply(op, ha, hb)?;
        let n = self.mk(v, lo, hi);
        self.memo.insert((op, a, b), n);
        Some(n)
    }

    pub(super) fn not(&mut self, a: u32) -> Option<u32> {
        self.apply(2, a, 1)
    }

    /// Whether a variable in [lo, hi) is under node `a` (each node once: the graph shares
    /// nodes; below a variable at or past `hi` there are only later ones).
    pub(super) fn uses_range(&self, a: u32, lo: u32, hi: u32) -> bool {
        let (min, max) = self.span[a as usize];
        if a < 2 || max < lo || min >= hi {
            return false;
        }
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![a];
        while let Some(n) = stack.pop() {
            let (min, max) = self.span[n as usize];
            if n < 2 || max < lo || min >= hi || !seen.insert(n) {
                continue;
            }
            let (v, l, h) = self.nodes[n as usize];
            if (lo..hi).contains(&v) {
                return true;
            }
            stack.extend([l, h]);
        }
        false
    }
}

/// Which lanes a mask word stands for: a word of the wave's EXEC is relative to the lanes the
/// wave has in that half (`trueBdd`: every one of them on), a constant to none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Half {
    Lo,
    Hi,
    Any,
}

/// What an SGPR holds where the analysis is: a predicate of each lane's bit, or nothing known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Word {
    Known(u32, Half),
    Unknown,
}

/// Atoms made by instruction `i`: its results' words and its unknown operands.
pub(super) const ATOMS_PER_INST: u32 = 32;

/// The instructions before which EXEC provably holds every lane the wave started with (its
/// entry EXEC, every live thread's bit, when `entry_full`), by pc: there a lane's EXEC bit is
/// known set and the emitter's per-lane guard (`xcl_on`) can go.
///
/// Every SGPR word is a predicate of the lane (`Bdd`) over atoms, masks the analysis does not
/// follow: a compare's mask is EXEC and the compare's own atom (inactive lanes give 0), the
/// mask operations and saveexecs combine predicates exactly, anything else is a new atom. So
/// an `if` / `else`'s `exec | (old & !vcc)` restores the entry EXEC exactly, which a plain
/// "holds every lane" flag cannot follow. An atom made again (a loop) forgets every word made
/// from its old value; words that differ where paths meet are unknown; a loop guard's exit
/// takes the state before any instruction of its loop. An indirect jump proves nothing.
pub(super) fn full_exec(insts: &[Decoded], wave64: bool, entry_full: bool, guard: &GuardExits) -> BTreeSet<u64> {
    let (succ, exits) = graph(insts, guard);
    if insts.is_empty() || succ.iter().any(Option::is_none) || insts.len() > (u32::MAX / ATOMS_PER_INST) as usize - 2 {
        return BTreeSet::new();
    }
    let succ: Vec<Vec<usize>> = succ.into_iter().map(Option::unwrap).collect();
    let n = insts.len();
    let mut preds = vec![0usize; n];
    for s in &succ {
        for &j in s {
            preds[j] += 1;
        }
    }
    let mut leader = vec![false; n];
    leader[0] = true;
    for i in 0..n {
        for &j in &succ[i] {
            if j != i + 1 || succ[i].len() != 1 || preds[j] != 1 {
                leader[j] = true;
            }
        }
        for &j in exits[i].iter().flatten() {
            leader[j] = true;
        }
    }
    let mut bdd = Bdd::new();
    let mut entry = vec![Word::Unknown; 128];
    if entry_full {
        entry[126] = Word::Known(1, Half::Lo);
        entry[127] = Word::Known(1, Half::Hi);
    } else {
        let base = n as u32 * ATOMS_PER_INST;
        entry[126] = Word::Known(bdd.var(base), Half::Lo);
        entry[127] = Word::Known(bdd.var(base + 1), Half::Hi);
    }
    let mut state: HashMap<usize, Vec<Word>> = HashMap::from([(0, entry)]);
    let mut work = BTreeSet::from([0usize]);
    let cyclic = on_cycles(&succ);
    let mut full = vec![false; n];
    let meet = |state: &mut HashMap<usize, Vec<Word>>, work: &mut BTreeSet<usize>, j: usize, s: &[Word]| match state.get_mut(&j) {
        None => {
            state.insert(j, s.to_vec());
            work.insert(j);
        }
        Some(t) => {
            let mut changed = false;
            for (a, b) in t.iter_mut().zip(s) {
                if *a != *b && *a != Word::Unknown {
                    *a = Word::Unknown;
                    changed = true;
                }
            }
            if changed {
                work.insert(j);
            }
        }
    };
    let is_full = |s: &[Word]| {
        let on = |w: Word, h: Half| matches!(w, Word::Known(1, x) if x == h || x == Half::Any);
        on(s[126], Half::Lo) && (!wave64 || on(s[127], Half::Hi))
    };
    while let Some(l) = work.pop_first() {
        let mut s = state[&l].clone();
        let mut i = l;
        loop {
            full[i] = is_full(&s);
            for &j in exits[i].iter().flatten() {
                meet(&mut state, &mut work, j, &s);
            }
            mask_step(&mut bdd, &insts[i], i as u32, wave64, cyclic[i], &mut s);
            match succ[i].as_slice() {
                [j] if *j == i + 1 && !leader[*j] => i = *j,
                next => {
                    for &j in next {
                        meet(&mut state, &mut work, j, &s);
                    }
                    break;
                }
            }
        }
    }
    insts.iter().zip(&full).filter(|(d, &f)| f && sgpr_use(d).maybe & EXEC == 0).map(|(d, _)| d.pc).collect()
}

/// `full_exec`'s step: the SGPR words after instruction `i` (`d`), from those before.
pub(super) fn mask_step(bdd: &mut Bdd, d: &Decoded, i: u32, wave64: bool, cyclic: bool, s: &mut [Word]) {
    let name = d.inst.name;
    let u = sgpr_use(d);
    let base = i * ATOMS_PER_INST;
    // what this instruction makes anew replaces every word made from its earlier atoms (only an
    // instruction on a cycle can have run before)
    for w in s.iter_mut().filter(|_| cyclic) {
        if let Word::Known(p, _) = *w {
            if bdd.uses_range(p, base, base + ATOMS_PER_INST) {
                *w = Word::Unknown;
            }
        }
    }
    if u.maybe == ALL {
        s.fill(Word::Unknown);
        return;
    }
    let field = |f: &str| rdna2::field_value(d.enc, &d.words, f);
    let b64 = name.ends_with("_B64") && wave64;
    let words = if b64 { 2 } else { 1 };
    // dword k of source operand f (its index among the operands: its atom if unknown)
    let src = |bdd: &mut Bdd, f: &str, which: u32, k: u64| -> Word {
        let fresh = |bdd: &mut Bdd| Word::Known(bdd.var(base + 16 + which * 2 + k as u32), Half::Any);
        match field(f) {
            Some(193) => Word::Known(1, Half::Any),
            Some(128) => Word::Known(0, Half::Any),
            Some(255) if !b64 && d.literal == Some(u32::MAX) => Word::Known(1, Half::Any),
            Some(c) if c <= 127 && c != 125 && c + k <= 127 => match s[(c + k) as usize] {
                Word::Unknown => fresh(bdd),
                w => w,
            },
            _ => fresh(bdd),
        }
    };
    let combine = |bdd: &mut Bdd, op: &[u8], a: Word, b: Word, not_a: bool, not_b: bool, not_r: bool| -> Word {
        let (Word::Known(x, hx), Word::Known(y, hy)) = (a, b) else { return Word::Unknown };
        let h = match (hx, hy) {
            (Half::Any, h) | (h, Half::Any) => h,
            (p, q) if p == q => p,
            _ => return Word::Unknown,
        };
        let r = (|| {
            let x = if not_a { bdd.not(x)? } else { x };
            let y = if not_b { bdd.not(y)? } else { y };
            let r = bdd.apply(op[0], x, y)?;
            if not_r { bdd.not(r) } else { Some(r) }
        })();
        r.map_or(Word::Unknown, |r| Word::Known(r, h))
    };
    // (op, not a, not b, not result) of a two-operand mask operation
    let logic = |op: &str| -> Option<(u8, bool, bool, bool)> {
        Some(match op {
            "AND" => (0, false, false, false),
            "OR" => (1, false, false, false),
            "XOR" => (2, false, false, false),
            "ANDN2" => (0, false, true, false),
            "ORN2" => (1, false, true, false),
            "ANDN1" => (0, true, false, false),
            "ORN1" => (1, true, false, false),
            "NAND" => (0, false, false, true),
            "NOR" => (1, false, false, true),
            "XNOR" => (2, false, false, true),
            _ => return None,
        })
    };
    let op = name.strip_prefix("S_").and_then(|r| r.split('_').next()).unwrap_or("");
    let sdst = field("SDST").filter(|&c| c <= 127 && c != 125);
    let mut out: Vec<(u64, Word)> = Vec::new();
    let generic_ok;
    if name.contains("SAVEEXEC") || name.contains("WREXEC") {
        let Some((o, na, nb, nr)) = logic(op) else { return unknown_writes(bdd, &u, base, s) };
        let mut exec_new = Vec::new();
        for k in 0..words {
            let old = s[126 + k as usize];
            let a = src(bdd, "SSRC0", 0, k);
            let new = combine(bdd, &[o], a, old, na, nb, nr);
            exec_new.push((126 + k, new));
            if let Some(dst) = sdst.filter(|&dst| dst + k <= 127) {
                out.push((dst + k, if name.contains("WREXEC") { new } else { old }));
            }
        }
        out.extend(exec_new);
        generic_ok = true;
    } else if d.enc.name == "ENC_SOP1" || d.enc.name == "ENC_SOP2" {
        let two = d.enc.name == "ENC_SOP2";
        generic_ok = match (op, name.contains("_B32") || name.contains("_B64"), sdst) {
            (_, false, _) | (_, _, None) => false,
            ("MOV", _, Some(dst)) if !two => {
                for k in 0..words {
                    out.push((dst + k, src(bdd, "SSRC0", 0, k)));
                }
                true
            }
            ("NOT", _, Some(dst)) if !two => {
                for k in 0..words {
                    let a = src(bdd, "SSRC0", 0, k);
                    out.push((dst + k, combine(bdd, &[2], a, Word::Known(1, Half::Any), false, false, false)));
                }
                true
            }
            ("WQM", _, Some(dst)) if !two => {
                for k in 0..words {
                    let a = src(bdd, "SSRC0", 0, k);
                    out.push((dst + k, if matches!(a, Word::Known(1, _)) { a } else { Word::Known(bdd.var(base + k as u32), Half::Any) }));
                }
                true
            }
            (o, _, Some(dst)) if two && logic(o).is_some() => {
                let (o, na, nb, nr) = logic(o).unwrap();
                for k in 0..words {
                    let (a, b) = (src(bdd, "SSRC0", 0, k), src(bdd, "SSRC1", 1, k));
                    out.push((dst + k, combine(bdd, &[o], a, b, na, nb, nr)));
                }
                true
            }
            _ => false,
        };
    } else if d.enc.name.starts_with("ENC_VOP") && (name.starts_with("V_CMP") || u.written & EXEC != 0) {
        // a compare: EXEC and the lane's condition, per mask word (V_CMPX: into EXEC)
        let c = bdd.var(base + 15);
        let dst: Vec<u64> = (0..128u64).filter(|&r| u.written >> r & 1 != 0).collect();
        let mask_words = if wave64 { 2 } else { 1 };
        for (k, &r) in dst.iter().enumerate().take(mask_words) {
            let e = s[126 + k];
            out.push((r, combine(bdd, &[0], e, Word::Known(c, Half::Any), false, false, false)));
        }
        generic_ok = dst.is_empty() || dst.len() == mask_words || (!wave64 && dst.len() == 2);
    } else {
        generic_ok = false;
    }
    if !generic_ok {
        return unknown_writes(bdd, &u, base, s);
    }
    unknown_writes(bdd, &u, base, s);
    for (r, w) in out {
        s[r as usize] = w;
    }
}

/// Which instructions lie on a cycle of the graph (Tarjan's strongly connected components).
pub(super) fn on_cycles(succ: &[Vec<usize>]) -> Vec<bool> {
    let n = succ.len();
    let (mut index, mut low, mut on_stack) = (vec![usize::MAX; n], vec![0; n], vec![false; n]);
    let (mut stack, mut cyclic, mut next) = (Vec::new(), vec![false; n], 0);
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut calls = vec![(root, 0usize)];
        while let Some(&mut (v, ref mut e)) = calls.last_mut() {
            if *e == 0 {
                index[v] = next;
                low[v] = next;
                next += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if let Some(&w) = succ[v].get(*e) {
                *e += 1;
                if index[w] == usize::MAX {
                    calls.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            calls.pop();
            if let Some(&(p, _)) = calls.last() {
                low[p] = low[p].min(low[v]);
            }
            if low[v] == index[v] {
                let mut members = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack[w] = false;
                    members.push(w);
                    if w == v {
                        break;
                    }
                }
                let cycle = members.len() > 1 || succ[v].contains(&v);
                for w in members {
                    cyclic[w] = cycle;
                }
            }
        }
    }
    cyclic
}

/// Every word an instruction may write: a new atom where it is overwritten, unknown where it
/// is only partly or maybe written.
fn unknown_writes(bdd: &mut Bdd, u: &super::x64_masks::SgprUse, base: u32, s: &mut [Word]) {
    let mut k = 0;
    let mut maybe = u.maybe;
    while maybe != 0 {
        let r = maybe.trailing_zeros() as u64;
        maybe &= maybe - 1;
        s[r as usize] = if u.written >> r & 1 != 0 && k < 15 {
            k += 1;
            Word::Known(bdd.var(base + k - 1), Half::Any)
        } else {
            Word::Unknown
        };
    }
}

/// VGPRs as a bit set.
type Vregs = [u128; 2];

fn has(s: &Vregs, v: u64) -> bool {
    v < 256 && s[(v / 128) as usize] >> (v % 128) & 1 != 0
}

fn set(regs: &[u64]) -> Vregs {
    let mut s = [0u128; 2];
    for &v in regs.iter().filter(|&&v| v < 256) {
        s[(v / 128) as usize] |= 1 << (v % 128);
    }
    s
}

fn or(a: Vregs, b: Vregs) -> Vregs {
    [a[0] | b[0], a[1] | b[1]]
}

/// Reads VGPRs of lanes other than its own, or of lanes outside EXEC: DPP, lane reads and
/// writes, LDS lane exchanges, image operations (implicit derivatives read the quad) and
/// exports.
fn reads_other_lanes(d: &Decoded) -> bool {
    let n = d.inst.name;
    matches!(d.variant, Variant::Dpp16 | Variant::Dpp8)
        || matches!(d.enc.name, "ENC_MIMG" | "ENC_EXP")
        || ["V_READLANE", "V_READFIRSTLANE", "V_WRITELANE", "V_PERMLANE", "V_MOVREL", "DS_SWIZZLE", "DS_PERMUTE", "DS_BPERMUTE", "V_INTERP"].iter().any(|p| n.starts_with(p))
}

/// Replaces its VGPR results whole in every lane it runs in.
fn writes_whole(d: &Decoded) -> bool {
    d.enc.name.starts_with("ENC_VOP") && d.variant == Variant::Plain && !["V_WRITELANE", "V_MOVREL"].iter().any(|p| d.inst.name.starts_with(p))
}

/// By pc: the VGPR results of an instruction (outside `full_exec`) whose lane guard can go,
/// because what lanes outside EXEC kept is never read again. `end_reads_all`: the program's
/// end reads every register (kernel mode).
pub(super) fn loose_writes(insts: &[Decoded], full: &BTreeSet<u64>, end_reads_all: bool, guard: &GuardExits) -> BTreeMap<u64, Vec<u64>> {
    if insts.iter().any(|d| d.inst.name.starts_with("V_MOVREL") || d.inst.name.starts_with("S_SET_GPR_IDX")) {
        return BTreeMap::new();
    }
    let (succ, exits) = graph(insts, guard);
    if succ.iter().any(Option::is_none) {
        return BTreeMap::new();
    }
    let succ: Vec<Vec<usize>> = succ.into_iter().map(Option::unwrap).collect();
    let at_end: Vregs = if end_reads_all { [!0, !0] } else { [0, 0] };
    let uses: Vec<(Vregs, Vregs)> = insts.iter().map(|d| {
        let (r, w) = rdna2::vgpr_uses(d);
        (set(&r), set(&w))
    }).collect();
    let kill: Vec<Vregs> = insts.iter().zip(&uses).map(|(d, u)| if full.contains(&d.pc) && writes_whole(d) { u.1 } else { [0, 0] }).collect();
    // what may be read after an instruction, and where a loop guard leaves before it
    let mut live_in = vec![[0u128; 2]; insts.len()];
    let mut live_out = vec![[0u128; 2]; insts.len()];
    let mut left = vec![[0u128; 2]; insts.len()];
    loop {
        let mut changed = false;
        for i in (0..insts.len()).rev() {
            let out = if succ[i].is_empty() { at_end } else { succ[i].iter().fold([0, 0], |m, &j| or(m, live_in[j])) };
            let l = exits[i].iter().fold([0, 0], |m, e| or(m, e.map_or(at_end, |j| live_in[j])));
            let inn = or(or(uses[i].0, [out[0] & !kill[i][0], out[1] & !kill[i][1]]), l);
            if out != live_out[i] || inn != live_in[i] || l != left[i] {
                (live_out[i], live_in[i], left[i]) = (out, inn, l);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // after instruction i, before EXEC changes or the code branches: what is read outside
    // the lane, or anywhere once that happens
    let mut seen = vec![[0u128; 2]; insts.len()];
    for i in (0..insts.len()).rev() {
        seen[i] = if succ[i] != [i + 1] {
            live_out[i]
        } else if sgpr_use(&insts[i + 1]).maybe & EXEC != 0 {
            live_in[i + 1]
        } else {
            let other = if reads_other_lanes(&insts[i + 1]) { uses[i + 1].0 } else { [0, 0] };
            or(or(seen[i + 1], other), left[i + 1])
        };
    }
    insts
        .iter()
        .enumerate()
        .filter(|(_, d)| !full.contains(&d.pc) && sgpr_use(d).maybe & EXEC == 0)
        .filter_map(|(i, d)| {
            let w: Vec<u64> = rdna2::vgpr_uses(d).1.into_iter().filter(|&v| v < 256 && !has(&seen[i], v)).collect();
            (!w.is_empty()).then(|| (d.pc, w))
        })
        .collect()
}
