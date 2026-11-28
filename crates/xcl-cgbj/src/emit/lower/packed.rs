// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, M1d). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../../LICENSE and ../../../PROVENANCE.md), so it carries that licence.
//! The packed form of a console wave64 compute program on a host of 32-lane waves.
//!
//! One host wave of 32 runs one console wave of 64. Host lane i holds console lanes i and
//! i + 32, the low and the high half. The program declares `WaveSize(32)` and
//! `numthreads(ceil(T / 64) * 32, 1, 1)`.
//!
//! The form is entered with a structure in which every branch on a lane mask is the console
//! wave's ([`crate::structure::structure_wide`]), so no branch is per lane and a host wave
//! runs exactly the blocks its console wave runs. EXEC is each vector instruction's guard.
//! A vector instruction is then lowered twice, once per half, each with its own register
//! values, lane booleans and caches ([`Half`]). A scalar instruction is lowered once and
//! both halves share what it writes. What crosses lanes takes both halves at once: masks as
//! data, wave-wide decisions, lane reads, permutes and the whole-wave sections. Nothing
//! needs another host wave, so the form adds no barrier and no wait.
//!
//! The hooks in lower.rs are marked `xsemu:`.

use std::cell::Cell;
use std::collections::HashMap;

use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use super::super::wave::{Op, Shape};
use super::{Lower, Stop, Val, Why, unsupported};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg, Value};
use crate::native::StageIo;
use crate::resource::Site;
use crate::structure::{Atom, Bool};

thread_local! {
    /// See [`with_packed`].
    static PACKED: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// What crossed lanes in the packed form, in the order it was lowered, for the tests (the
    /// bitcode writer packs names, so the module can't be read back).
    pub(in super::super) static SEEN: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Notes for the tests that `what` was lowered across both halves.
fn seen(what: &'static str) {
    #[cfg(test)]
    SEEN.with(|s| s.borrow_mut().push(what));
    let _ = what;
}

/// Runs `f` with the lowering in the packed form.
pub(in super::super) fn with_packed<R>(f: impl FnOnce() -> R) -> R {
    let was = PACKED.replace(true);
    let r = f();
    PACKED.set(was);
    r
}

/// Whether the lowering on this thread is in the packed form.
pub(super) fn on() -> bool {
    PACKED.get()
}

/// What each half has of its own. The fields are the emitter's of the same names.
#[derive(Default)]
pub(super) struct Half {
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    strides: HashMap<(usize, Site, u8), V>,
    src0: Option<V>,
    root_copy: Option<V>,
    page_there: Option<V>,
    /// The binding's launch mask.
    launch: Option<V>,
}

pub(super) struct Packed {
    /// The half being lowered: 0 low, 1 high.
    half: u8,
    /// The other half's state.
    other: Half,
    /// Console threads in a group.
    group: u32,
    /// How each host wave finds the first console thread of its console wave.
    thread_base: ThreadBase,
    /// The host lane, and the first console thread of this host wave, made in the entry block.
    host_lane: Option<V>,
    base: Option<V>,
    /// Per half: the console lane and the console thread in the group, made in the entry block.
    lanes: [Option<V>; 2],
    threads: [Option<V>; 2],
    /// The instruction being lowered.
    site: Site,
}

/// The most LDS a D3D12 compute program may declare: 32 KiB.
const LDS_LIMIT_DWORDS: u32 = 8192;

/// The refusal of a group of more than one console wave whose program leaves no LDS dword for
/// the counter its host waves take their console threads from ([`Lower::p_base`]).
pub(in super::super) const NO_CLAIM_DWORD: Why = "packed wave64 form needs an LDS dword past 32 KiB";

/// The LDS dword of that counter for a group of `group` console threads whose program uses
/// `lds_dwords` of LDS: none with one console wave, else the first after the program's.
pub(in super::super) fn claim_dword(group: u32, lds_dwords: u32) -> Result<Option<u32>, Why> {
    if group <= 64 {
        return Ok(None);
    }
    if lds_dwords >= LDS_LIMIT_DWORDS {
        return Err(NO_CLAIM_DWORD);
    }
    Ok(Some(lds_dwords))
}

/// Whether a one-dimensional group of more than one console wave takes its console threads
/// from the host thread index instead of from the LDS counter (`XCL_CGBJ_PACKED_THREAD_BASE=1`).
/// `SV_GroupThreadID.x` of the 1-D host group is the host wave's place in the group and the
/// thread's place in that wave, so console thread `t + 32 * (t >> 5)` needs no counter and no
/// LDS ([`ThreadBase::Flat`]). Read under an `XCL_` name first and the binding's own
/// `VICTORY_` name second, both collected into the translation fingerprint, so cached
/// containers follow the setting.
pub(in super::super) fn flat_thread_base() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_FLAT_BASE.get() {
        return on;
    }
    switch_on("XCL_CGBJ_PACKED_THREAD_BASE") || switch_on("VICTORY_XCL_PACKED_THREAD_BASE")
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`flat_thread_base`], for its thread only (the process's is the
    /// variable, which tests must not set under one another).
    pub(in super::super) static TEST_FLAT_BASE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// The same for [`f64_sum`].
    pub(in super::super) static TEST_F64_SUM: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a switch is on: the variable set to `1`.
fn switch_on(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v == "1")
}

/// The refusal of a wave sum of 64-bit floats while [`f64_sum`] is off.
pub(in super::super) const F64_SUM: Why = "a wave sum of 64-bit floats in the packed form";

/// Whether the packed form lowers a whole-wave sum of 64-bit floats
/// (`XCL_CGBJ_PACKED_F64_SUM=1`) instead of refusing it ([`Lower::p_shape`]). Read under an
/// `XCL_` name first and the binding's own `VICTORY_` name second, both collected into the
/// translation fingerprint, so cached containers follow the setting.
pub(in super::super) fn f64_sum() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_F64_SUM.get() {
        return on;
    }
    switch_on("XCL_CGBJ_PACKED_F64_SUM") || switch_on("VICTORY_XCL_PACKED_F64_SUM")
}

/// Where a host wave of a group finds the first console thread of its console wave.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in super::super) struct ThreadBase {
    /// The LDS dword of the counter the host waves take their turns on, for a group of more
    /// than one console wave whose console threads do not follow the host thread index.
    pub claim: Option<u32>,
    /// Whether the console thread follows the host thread index ([`flat_thread_base`], a
    /// one-dimensional group): console thread `t + 32 * (t >> 5)`, so the base is
    /// `64 * (t >> 5)`.
    pub flat: bool,
}

/// How a group of `threads` threads finds each host wave's console thread: the host thread
/// index in a one-dimensional group when `flat` is on and the group is more than one console
/// wave, else the counter dword [`claim_dword`] gives. Pure, for testing.
pub(in super::super) fn thread_base(threads: [u32; 3], lds_dwords: u32, flat: bool) -> Result<ThreadBase, Why> {
    let group = threads.iter().product::<u32>();
    if group > 64 && flat && threads[1] == 1 && threads[2] == 1 {
        return Ok(ThreadBase { claim: None, flat: true });
    }
    let claim = claim_dword(group, lds_dwords)?;
    Ok(ThreadBase { claim, flat: false })
}

/// What [`Lower::section_inst`] says of a value it does not know.
const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";

impl<'a> Lower<'a> {
    /// Sets up the packed form after [`Lower::new`] when the lowering is in it.
    pub(super) fn p_setup(&mut self) {
        if !on() || !self.x.native.wave64 {
            return;
        }
        let StageIo::Cs(cs) = &self.x.native.io else {
            return;
        };
        let group = cs.threads.iter().product::<u32>();
        self.console_lanes = true;
        self.s.wave_size = Some(32);
        self.s.host_threads = Some([group.div_ceil(64) * 32, 1, 1]);
        let base = thread_base(cs.threads, self.s.lds_dwords, flat_thread_base()).expect("emit::xcl::packed refuses a group whose thread base does not fit");
        if base.claim.is_some() {
            self.s.lds_dwords += 1;
        }
        self.packed = Some(Box::new(Packed {
            half: 0,
            other: Half::default(),
            group,
            thread_base: base,
            host_lane: None,
            base: None,
            lanes: [None; 2],
            threads: [None; 2],
            site: (0, 0),
        }));
    }

    fn pk(&mut self) -> &mut Packed {
        self.packed.as_mut().expect("the packed form")
    }

    /// Swaps the state of the half being lowered with the other half's.
    pub(super) fn p_swap(&mut self) {
        let Some(mut p) = self.packed.take() else { return };
        let o = &mut p.other;
        std::mem::swap(&mut self.vals, &mut o.vals);
        std::mem::swap(&mut self.floats, &mut o.floats);
        std::mem::swap(&mut self.bools, &mut o.bools);
        std::mem::swap(&mut self.lane_bits, &mut o.lane_bits);
        std::mem::swap(&mut self.mask_phis, &mut o.mask_phis);
        std::mem::swap(&mut self.pending_masks, &mut o.pending_masks);
        std::mem::swap(&mut self.data_phis, &mut o.data_phis);
        std::mem::swap(&mut self.handles, &mut o.handles);
        std::mem::swap(&mut self.strides, &mut o.strides);
        std::mem::swap(&mut self.src0, &mut o.src0);
        std::mem::swap(&mut self.root_copy, &mut o.root_copy);
        std::mem::swap(&mut self.page_there, &mut o.page_there);
        if let Some(launch) = self.x_launch_slot() {
            std::mem::swap(launch, &mut o.launch);
        }
        p.half ^= 1;
        self.packed = Some(p);
    }

    /// `f` in the low half and in the high half, in that order, whichever is being lowered.
    fn p_halves<T>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, Why>) -> Result<(T, T), Why> {
        let first = self.pk().half;
        let a = f(self)?;
        self.p_swap();
        let b = f(self);
        self.p_swap();
        let b = b?;
        Ok(if first == 0 { (a, b) } else { (b, a) })
    }

    // ---- entry ----

    /// The high half's input VGPRs. Input SGPRs are shared.
    pub(super) fn p_inputs(&mut self) {
        if self.packed.is_none() {
            return;
        }
        let io: HashMap<Value, crate::io::Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            if !matches!(reg, Reg::V(_))
                && let Some(&val) = self.vals.get(&v)
            {
                self.pk().other.vals.insert(v, val);
            }
        }
        self.p_swap();
        for &(reg, v) in &self.x.p.inputs {
            if matches!(reg, Reg::V(_)) {
                let val = match io.get(&v) {
                    Some(&input) => self.input(input),
                    None => Val::I32(self.s.m.i32(0)),
                };
                self.vals.insert(v, val);
            }
        }
        self.p_swap();
    }

    /// The host lane (0..31), made in the entry block.
    fn p_host_lane_index(&mut self) -> V {
        if let Some(v) = self.pk().host_lane {
            return v;
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        self.s.flags |= flags::WAVE_OPS;
        let v = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        self.pk().host_lane = Some(v);
        v
    }

    /// The console lane of the half being lowered: the host lane, plus 32 in the high half.
    pub(super) fn p_lane(&mut self) -> Option<V> {
        let half = usize::from(self.packed.as_ref()?.half);
        if let Some(v) = self.pk().lanes[half] {
            return Some(v);
        }
        let host = self.p_host_lane_index();
        let v = if half == 0 {
            host
        } else {
            let k32 = self.s.m.i32(32);
            self.s.m.bin(self.s.entry, Bin::Add, host, k32)
        };
        self.pk().lanes[half] = Some(v);
        Some(v)
    }

    /// The host lane that holds console lane `lane` of the same half.
    pub(super) fn p_host_lane(&mut self, lane: V) -> V {
        if self.packed.is_none() {
            return lane;
        }
        let k31 = self.s.m.i32(31);
        self.s.m.bin(self.cur, Bin::And, lane, k31)
    }

    /// The first console thread of this host wave: the host thread's console wave
    /// (`64 * (t >> 5)`) where the group is one-dimensional and
    /// [`flat_thread_base`] is on, 0 with one console wave in a group, else the next 64 from
    /// an LDS counter, which is CGBJ's `wave_base` with steps of 64. Every thread zeroes the
    /// counter, the group meets, and each host wave's first lane adds 64.
    /// In the entry block, which every thread runs.
    fn p_base(&mut self) -> V {
        if let Some(v) = self.pk().base {
            return v;
        }
        let e = self.s.entry;
        if let Some(slot) = self.pk().thread_base.claim {
            let lds = self.s.lds();
            let k = self.s.m.i32(slot);
            let ptr = self.s.m.element(e, lds, k);
            let zero = self.s.m.i32(0);
            self.s.m.store(e, ptr, zero);
            self.group_sync(e);
            let lane = self.p_host_lane_index();
            let first = self.s.m.cmp(e, pred::EQ, lane, zero);
            let k64 = self.s.m.i32(64);
            let add = self.s.m.select(e, first, k64, zero);
            let ptr = self.s.m.element(e, lds, k);
            let old = self.s.m.atomic(e, super::super::bitcode::Rmw::Add, ptr, add);
            let base = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, self.s.t.i32, 118, &[old]);
            self.pk().base = Some(base);
            return base;
        }
        let v = if self.pk().thread_base.flat {
            // console thread = t + 32 * (t >> 5), and the halves' lanes add 0 and 32 to the base
            let Val::I32(t) = self.compute_id("threadIdInGroup.i32", 95, 0) else { unreachable!() };
            let k5 = self.s.m.i32(5);
            let wave = self.s.m.bin(e, Bin::LShr, t, k5);
            let k64 = self.s.m.i32(64);
            self.s.m.bin(e, Bin::Mul, wave, k64)
        } else {
            self.s.m.i32(0)
        };
        self.pk().base = Some(v);
        v
    }

    /// The console thread in the group of the half being lowered.
    fn p_thread(&mut self) -> V {
        let half = usize::from(self.pk().half);
        if let Some(v) = self.pk().threads[half] {
            return v;
        }
        let base = self.p_base();
        let lane = self.p_lane().expect("the packed form");
        let v = self.s.m.bin(self.s.entry, Bin::Add, base, lane);
        self.pk().threads[half] = Some(v);
        v
    }

    /// Component `c` of the console thread id of the half being lowered.
    pub(super) fn p_thread_id(&mut self, c: u8) -> Option<Val> {
        self.packed.as_ref()?;
        let [x, y, _] = self.s.threads.expect("compute");
        let f = self.p_thread();
        let e = self.s.entry;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Some(Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        }))
    }

    /// Whether the console lane of the half being lowered exists: none where the group ends
    /// inside a console wave. `None` when every lane exists.
    pub(super) fn p_valid(&mut self) -> Option<V> {
        let group = self.packed.as_ref()?.group;
        if group % 64 == 0 {
            return None;
        }
        let f = self.p_thread();
        let t = self.s.m.i32(group);
        Some(self.s.m.cmp(self.s.entry, pred::IULT, f, t))
    }

    // ---- instructions ----

    /// Lowers instruction `i` of IR block `b`: once for a scalar one, once per half for a
    /// vector one. Outside the packed form, as CGBJ does.
    pub(super) fn p_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        if self.packed.is_none() {
            return self.inst(b, i);
        }
        self.pk().site = (b, i);
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let name = inst.name();
        if name == "ds_permute_b32" {
            return Err(unsupported(inst.machine.offset, "a forward lane permute in the packed form"));
        }
        let lds = matches!(inst.fields, Fields::Ds { .. }) && !(name.starts_with("ds_swizzle") || name.contains("permute"));
        let once = match inst.fields {
            Fields::Valu(_) => matches!(name, "v_readlane_b32" | "v_readfirstlane_b32"),
            Fields::Ds { .. } => name == "ds_bpermute_b32",
            Fields::Buffer { .. } | Fields::Image { .. } | Fields::Flat { .. } | Fields::Vintrp { .. } | Fields::Exp { .. } => false,
            _ => true,
        };
        if once {
            self.inst(b, i)?;
            self.p_share(inst);
            return Ok(());
        }
        self.inst(b, i)?;
        self.p_swap();
        let r = self.inst(b, i);
        self.p_swap();
        r?;
        if lds {
            // the console's lanes see each other's LDS accesses of one instruction: both halves'
            // are made before the next (a groupshared fence, no group sync, as xsemu's translator)
            let (mode, void) = (self.s.m.i32(8), self.s.t.void);
            self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        }
        Ok(())
    }

    /// What a scalar instruction wrote is the other half's too.
    fn p_share(&mut self, inst: &Inst) {
        for &(reg, v) in &inst.writes {
            if !matches!(reg, Reg::V(_))
                && let Some(&val) = self.vals.get(&v)
            {
                self.pk().other.vals.insert(v, val);
            }
        }
    }

    /// The high half's join of a VGPR at a block's entry; any other join is shared.
    pub(super) fn p_phi(&mut self, b: usize, i: usize, val: Val) {
        if self.packed.is_none() {
            return;
        }
        let phi = &self.x.p.blocks[b].phis[i];
        if !matches!(phi.reg, Reg::V(_)) {
            self.pk().other.vals.insert(phi.value, val);
            return;
        }
        self.p_swap();
        let val = match self.bad.get(&phi.value) {
            Some(&why) => Val::Missing(why),
            None => {
                let v = self.s.m.phi(self.cur, self.s.t.i32);
                self.data_phis.push((v, b, i));
                Val::I32(v)
            }
        };
        self.vals.insert(phi.value, val);
        self.p_swap();
    }

    /// The high half's joins, filled as CGBJ fills the low half's.
    pub(super) fn p_fill_phis(&mut self) -> Result<(), Stop> {
        if self.packed.is_none() {
            return Ok(());
        }
        self.p_swap();
        let r = self.fill_phis();
        self.p_swap();
        r
    }

    // ---- across lanes ----

    /// A lane mask as data: the ballot of the low half's bits and of the high half's.
    pub(super) fn p_mask_dword(&mut self, lo: Value, high: bool, m: Bool) -> Result<V, Why> {
        if let Some(&(x, y)) = self.ballots.get(&lo) {
            return Ok(if high { y } else { x });
        }
        let at = match self.def_block.get(&lo) {
            None => self.s.entry,
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bits = self.p_halves(|l| l.boolean(m, at));
        self.balloting.remove(&lo);
        let (a, b) = bits?;
        seen("ballot");
        let xy = (self.p_ballot(at, a), self.p_ballot(at, b));
        self.ballots.insert(lo, xy);
        Ok(if high { xy.1 } else { xy.0 })
    }

    /// The host wave's ballot of `bit`: one half's 32 bits.
    fn p_ballot(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        self.s.flags |= flags::WAVE_OPS;
        let r = self.s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.m.extract(at, r, 0)
    }

    /// SCC of a mask operation: some lane of either half has its bit.
    pub(super) fn p_any(&mut self, m: Bool, at: usize) -> Result<V, Why> {
        let (a, b) = self.p_halves(|l| l.boolean(m, at))?;
        seen("scc");
        let either = self.s.m.bin(at, Bin::Or, a, b);
        self.s.flags |= flags::WAVE_OPS;
        let i1t = self.s.t.i1;
        Ok(self.s.op(at, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[either]))
    }

    /// The console's decision of a wave-wide branch: some lane's `when` (`any`) or every
    /// lane's. Every host lane is here, and a console lane outside EXEC says what the console
    /// takes it to say.
    pub(super) fn p_wave_wide(&mut self, when: Bool, any: bool) -> Result<V, Why> {
        let c = self.cur;
        let (a, b) = self.p_halves(|l| l.boolean(when, c))?;
        seen(if any { "branch on some lane" } else { "branch on every lane" });
        let c = self.cur;
        let (op, name, code) = if any { (Bin::Or, "waveAnyTrue", 113) } else { (Bin::And, "waveAllTrue", 114) };
        let both = self.s.m.bin(c, op, a, b);
        self.s.flags |= flags::WAVE_OPS;
        let i1t = self.s.t.i1;
        Ok(self.s.op(c, name, Attr::NoUnwind, i1t, code, &[both]))
    }

    /// Refuses a branch that is still per lane: the halves of a host lane could part there.
    pub(super) fn p_lane_branch(&self, b: usize, when: Bool) -> Result<(), Why> {
        let st = self.x.st;
        if self.packed.is_none() || st.wide.contains(&b) || st.wave_scc.contains(&b) {
            return Ok(());
        }
        if st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))) {
            Ok(())
        } else {
            Err("a lane branch in the packed form")
        }
    }

    /// `v_readlane_b32` at console lane `lane`, which the low or the high half holds.
    pub(super) fn p_lane_at(&mut self, inst: &Inst, lane: V) -> Result<V, Why> {
        let (lo, hi) = self.p_halves(|l| l.src(inst, 0))?;
        seen("lane read");
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let up = self.s.m.bin(c, Bin::And, lane, k32);
        let up = self.s.m.cmp(c, pred::NE, up, z);
        let a = self.read_lane(lo, at);
        let b = self.read_lane(hi, at);
        self.s.flags |= flags::WAVE_OPS;
        Ok(self.s.m.select(c, up, b, a))
    }

    /// `v_readfirstlane_b32`: the value of the lowest console lane in EXEC, the low half's
    /// lanes first. With no lane in EXEC the console reads lane 0; this reads the host's first
    /// lane, as CGBJ does.
    pub(super) fn p_first_lane(&mut self, inst: &Inst) -> Result<V, Why> {
        let (b, i) = self.pk().site;
        let ((a, g), (ah, gh)) = self.p_halves(|l| Ok((l.src(inst, 0)?, l.guard(b, i)?)))?;
        seen("first lane");
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[a]);
        let (Some(g), Some(gh)) = (g, gh) else {
            return Ok(first);
        };
        let (lo, hi) = (self.p_ballot(c, g), self.p_ballot(c, gh));
        let z = self.s.m.i32(0);
        // FirstbitLo (dx.op 32): the lowest set bit
        let flo = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let in_lo = self.s.m.cmp(c, pred::NE, lo, z);
        let in_hi = self.s.m.cmp(c, pred::NE, hi, z);
        let flo = self.s.m.select(c, in_lo, flo, z);
        let fhi = self.s.m.select(c, in_hi, fhi, z);
        let from_lo = self.read_lane(a, flo);
        let from_hi = self.read_lane(ah, fhi);
        let r = self.s.m.select(c, in_hi, from_hi, first);
        Ok(self.s.m.select(c, in_lo, from_lo, r))
    }

    /// `ds_bpermute_b32`: each lane takes the data of the console lane its address names,
    /// in either half; a lane outside EXEC gives 0.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn p_bpermute(
        &mut self,
        inst: &Inst,
        addr: Option<&Operand>,
        data: Option<&Operand>,
        offset: u32,
        dst: Option<u8>,
        b: usize,
        i: usize,
    ) -> Result<(), Why> {
        let addr = addr.ok_or("a permute address")?;
        let data = data.ok_or("a permute source")?;
        let reg = dst.ok_or("a permute result")?;
        self.cross_lane(false)?;
        let (lo, hi) = self.p_halves(|l| {
            let c = l.cur;
            let g = l.guard(b, i)?;
            let at = l.vgprs(inst, addr, 1)?[0];
            let x = l.vgprs(inst, data, 1)?[0];
            let o = l.s.m.i32(offset);
            let at = l.s.m.bin(c, Bin::Add, at, o);
            let two = l.s.m.i32(2);
            let src = l.s.m.bin(c, Bin::LShr, at, two);
            let x = match g {
                Some(g) => {
                    let z = l.s.m.i32(0);
                    l.s.m.select(c, g, x, z)
                }
                None => x,
            };
            Ok((src, x, g))
        })?;
        for (half, (src, _, g)) in [lo, hi].into_iter().enumerate() {
            let c = self.cur;
            let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
            let up = self.s.m.bin(c, Bin::And, src, k32);
            let up = self.s.m.cmp(c, pred::NE, up, z);
            let a = self.read_lane(lo.1, src);
            let b = self.read_lane(hi.1, src);
            let r = self.s.m.select(c, up, b, a);
            if half == 1 {
                self.p_swap();
            }
            self.write(inst, Reg::V(reg), r, g);
            if half == 1 {
                self.p_swap();
            }
        }
        Ok(())
    }

    // ---- whole-wave sections ----

    /// An instruction inside a whole-wave section ([`Lower::section_inst`] in the packed form).
    pub(super) fn p_section_inst(&mut self, b: usize, i: usize, start: usize) {
        if self.packed.is_none() {
            return self.section_inst(b, i, start);
        }
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let r = match self.sections[&b][&start].shapes.get(&w).copied() {
                Some(sh) if self.section_out.contains(&w) => self.p_shape(sh),
                _ => Err(UNKNOWN),
            };
            let (lo, hi) = match r {
                Ok((lo, hi)) => (Val::I32(lo), Val::I32(hi)),
                Err(why) => (Val::Missing(why), Val::Missing(why)),
            };
            self.vals.insert(w, lo);
            self.pk().other.vals.insert(w, hi);
        }
    }

    /// One lane of a value a section left, read by `v_readlane_b32`: only values every lane
    /// holds alike.
    pub(super) fn p_section_lane(&mut self, shape: Shape) -> Result<V, Why> {
        match shape {
            Shape::K(_) | Shape::Value(_) | Shape::Active(..) => Ok(self.p_shape(shape)?.0),
            _ => Err("a lane of a whole-wave section's per-lane value in the packed form"),
        }
    }

    /// A value a section leaves, in the low half and in the high half.
    fn p_shape(&mut self, shape: Shape) -> Result<(V, V), Why> {
        match shape {
            Shape::K(k) => {
                let v = self.s.m.i32(k);
                Ok((v, v))
            }
            Shape::Value(v) => self.p_halves(|l| l.i32_of(v)),
            Shape::Masked(src, mask, ident) => self.p_halves(|l| l.p_masked(src, mask, ident)),
            Shape::Active(op, src, mask, ident) => {
                let r = self.p_reduce(op, src, mask, ident)?;
                Ok((r, r))
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => self.p_prefix(op, src, mask, ident, inclusive),
            Shape::ActiveF64(lo, hi, mask, high) => {
                if !f64_sum() {
                    return Err(F64_SUM);
                }
                let v = self.p_f64_sum(lo, hi, mask, high)?;
                Ok((v, v))
            }
        }
    }

    /// `Shape::ActiveF64`: the sum of the console lanes' `mask ? f64(lo, hi) : 0.0`, returning
    /// the low (`!high`) or high (`high`) 32-bit word. One wave operation per half, then one
    /// add of the two halves: the split form's tree (`Lower::active_f64_sum`), so the two forms'
    /// results are the same bits. Adding each host lane's two console lanes first would be
    /// another tree.
    fn p_f64_sum(&mut self, lo: Value, hi: Value, mask: Value, high: bool) -> Result<V, Why> {
        // per half: the two words of its lanes' masked terms
        let (lower, upper) = self.p_halves(|l| Ok((l.p_masked(lo, mask, 0)?, l.p_masked(hi, mask, 0)?)))?;
        seen("wave sum of 64-bit floats");
        self.s.flags |= flags::WAVE_OPS | flags::DOUBLES;
        let (i32t, f64t) = (self.s.t.i32, self.s.m.double());
        let c = self.cur;
        let sum = |l: &mut Self, lo: V, hi: V| {
            let x = l.s.op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]);
            let (k0, k0b) = (l.s.m.i8(0), l.s.m.i8(0));
            l.s.op(c, "waveActiveOp.f64", Attr::NoUnwind, f64t, 119, &[x, k0, k0b])
        };
        let lower = sum(self, lower.0, lower.1);
        let upper = sum(self, upper.0, upper.1);
        let total = self.s.m.bin(c, Bin::Add, lower, upper);
        let split_ty = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let split = self.s.op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[total]);
        Ok(self.s.m.extract(c, split, u32::from(high)))
    }

    /// The lane's own `mask ? src : ident` in the half being lowered.
    fn p_masked(&mut self, src: Value, mask: Value, ident: u32) -> Result<V, Why> {
        let c = self.cur;
        let &m = self.x.st.masks.get(&mask).ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        Ok(self.s.m.select(c, on, x, id))
    }

    /// `op` over every console lane: the halves combine in each host lane, then one wave op.
    fn p_reduce(&mut self, op: Op, src: Value, mask: Value, ident: u32) -> Result<V, Why> {
        if op == Op::AddH {
            return Err("a wave sum of 16-bit floats in the packed form");
        }
        let (lo, hi) = self.p_halves(|l| l.p_masked(src, mask, ident))?;
        seen("wave reduction");
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let (ty, sfx) = if float { (f32t, "f32") } else { (i32t, "i32") };
        let (lo, hi) = if float { (self.as_float(lo), self.as_float(hi)) } else { (lo, hi) };
        let c = self.cur;
        let x = match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            _ => {
                // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                let code = match op {
                    Op::MaxF => 35,
                    Op::MinF => 36,
                    Op::MaxI => 37,
                    Op::MinI => 38,
                    Op::MaxU => 39,
                    _ => 40,
                };
                self.s.op(c, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[lo, hi])
            }
        };
        self.s.flags |= flags::WAVE_OPS;
        let r = match op {
            Op::Or | Op::And | Op::Xor => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s.op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[x, k])
            }
            _ => self.p_wave_op(op, x),
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    /// `waveActiveOp` of an arithmetic `op` (a float one on f32 values).
    fn p_wave_op(&mut self, op: Op, x: V) -> V {
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (ty, sfx) = if float { (self.s.t.f32, "f32") } else { (self.s.t.i32, "i32") };
        let (code, unsigned) = match op {
            Op::AddU => (0, 1),
            Op::AddF => (0, 0),
            Op::MinU => (2, 1),
            Op::MaxU => (3, 1),
            Op::MinI | Op::MinF => (2, 0),
            _ => (3, 0),
        };
        let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
        let c = self.cur;
        self.s.op(c, &format!("waveActiveOp.{sfx}"), Attr::NoUnwind, ty, 119, &[x, k, u])
    }

    /// A sum over the console lanes below each lane (`inclusive`: and the lane): the low half's
    /// own prefix, and in the high half the low half's total plus its own prefix.
    fn p_prefix(&mut self, op: Op, src: Value, mask: Value, ident: u32, inclusive: bool) -> Result<(V, V), Why> {
        if !matches!(op, Op::AddU | Op::AddF) {
            return Err("a wave prefix of an operation other than a sum");
        }
        let float = op == Op::AddF;
        let (lo, hi) = self.p_halves(|l| l.p_masked(src, mask, ident))?;
        let (lo, hi) = if float { (self.as_float(lo), self.as_float(hi)) } else { (lo, hi) };
        let (ty, sfx) = if float { (self.s.t.f32, "f32") } else { (self.s.t.i32, "i32") };
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
        let prefix = |l: &mut Self, x: V| {
            let r = l.s.op(c, &format!("wavePrefixOp.{sfx}"), Attr::NoUnwind, ty, 121, &[x, k0, k1]);
            if inclusive { l.s.m.bin(c, Bin::Add, r, x) } else { r }
        };
        let p_lo = prefix(self, lo);
        let p_hi = prefix(self, hi);
        let total = self.p_wave_op(op, lo);
        let p_hi = self.s.m.bin(c, Bin::Add, total, p_hi);
        Ok(if float { (self.as_int(p_lo), self.as_int(p_hi)) } else { (p_lo, p_hi) })
    }
}
