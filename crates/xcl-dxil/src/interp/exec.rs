//! The wave interpreter: one thread group of a [`Program`] over a [`World`].
//!
//! # Lanes, waves and reconvergence
//!
//! The threads of a group are cut into waves of the program's wave size in thread-index
//! order. A wave executes one block at a time for a set of lanes; every wave operation in
//! that block sees exactly those lanes. Which lanes run next is one rule: among the lanes
//! that can run, those waiting at the lowest position of the schedule order
//! ([`super::cfg`]), where a lane that took a back edge waits behind every block of that
//! loop. Hence lanes that diverge at a branch run their sides and meet again at the first
//! block both reach; lanes that left a loop wait at its exit until the last lane is out; a
//! lane that finished an iteration waits at the header for the lanes still in the body,
//! so a wave operation in iteration k sees the lanes that are in iteration k.
//!
//! # Scheduling points
//!
//! A lane that reaches a group-sync barrier stops until no lane of the group can run (all
//! are at a barrier or done), then all continue: lanes of every wave pass the barrier
//! together. With several waves in a group, a wave runs until it has executed an atomic
//! (or can not run), then the next wave runs: the interleaving is a function of each
//! wave's sequence of atomics, which optimisation does not change, and a wave that spins
//! on an atomic waiting for another wave lets that wave run.

use super::decode::*;
use super::math::{self, *};
use super::res::{self, Atomic, World};
use crate::consts::OpCode;

/// A lane indexed a local array out of its bounds (undefined in HLSL).
pub const UB_LOCAL_BOUNDS: u32 = 1;
/// A wave or quad read named a lane that was not executing the instruction.
pub const UB_INACTIVE_LANE: u32 = 2;
/// A lane reached `unreachable`.
pub const UB_UNREACHABLE: u32 = 4;
/// A load or store through a pointer to nothing.
pub const UB_POINTER: u32 = 8;
/// Two lanes of a wave or two waves accessed the same memory, one writing, in an order the
/// program does not fix ([`res::Clock`]), or a lane read groupshared memory another lane of its wave had
/// written since the wave's last call that may write memory (where a compiler may still
/// use the lane's own earlier value).
pub const UB_RACE: u32 = 16;
/// An operation returned a NaN the IR does not fix ([math::take_nan_open]).
pub const UB_NAN: u32 = 32;

pub fn describe_ub(ub: u32) -> String {
    let names = [(UB_LOCAL_BOUNDS, "local array index out of bounds"), (UB_INACTIVE_LANE, "wave read of an inactive lane"), (UB_UNREACHABLE, "unreachable executed"), (UB_POINTER, "access through a bad pointer"), (UB_RACE, "a data race between lanes or waves"), (UB_NAN, "a NaN whose payload the IR leaves open")];
    names.iter().filter(|(bit, _)| ub & bit != 0).map(|(_, n)| *n).collect::<Vec<_>>().join(", ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// the step budget ran out
    Budget,
    /// something the interpreter does not model, found while running
    Unsupported(String),
}

const DONE: u32 = u32::MAX;
const IDLE: u64 = u64::MAX;

struct Wave {
    regs: Vec<u64>,
    frame: Vec<u64>,
    /// per lane: the block it runs next, [`DONE`] when it returned
    next: Vec<u32>,
    /// per lane: its place in the schedule, [`IDLE`] while it can not run
    key: Vec<u64>,
    /// per lane: the key it gets back when the barrier it waits at opens, else [`IDLE`]
    held: Vec<u64>,
    /// per lane: the block it ran last
    pred: Vec<u32>,
    lanes: usize,
    /// lanes that have not returned
    live: usize,
    first_thread: u32,
    /// [`res::Clock`] ticks: the last block execution with every live lane, and the last
    /// call that may write memory
    region_start: u32,
    clobber: u32,
    /// WaveActiveBallot calls executed
    ballots: u32,
    /// the pixels that were discarded, as a mask of lanes
    dead: u128,
}

enum Flow {
    Next,
    Yield,
}

pub struct Group<'a> {
    p: &'a Program,
    world: &'a mut World,
    shared: Vec<u64>,
    /// who wrote each groupshared cell last, and when
    shared_owner: Vec<u32>,
    shared_stamp: Vec<u32>,
    id: [u32; 3],
    waves: Vec<Wave>,
    pub steps: u64,
    budget: u64,
    pub ub: u32,
    tmp: Vec<u64>,
}

#[inline]
fn mask(w: u8) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

#[inline]
fn sext(x: u64, w: u8) -> i64 {
    let s = 64 - (w as u32).clamp(1, 64);
    ((x << s) as i64) >> s
}

#[inline]
fn forward_key(t: u32) -> u64 {
    (t as u64) << 33
}

#[inline]
fn back_key(loop_end: u32, t: u32) -> u64 {
    (loop_end as u64) << 33 | 1 << 32 | (u32::MAX - t) as u64
}

/// DXIL AtomicBinOpCode numbers.
mod atomic {
    pub const ADD: u8 = 0;
    pub const AND: u8 = 1;
    pub const OR: u8 = 2;
    pub const XOR: u8 = 3;
    pub const IMIN: u8 = 4;
    pub const IMAX: u8 = 5;
    pub const UMIN: u8 = 6;
    pub const UMAX: u8 = 7;
    pub const EXCHANGE: u8 = 8;
}

fn atomic_apply(op: u8, w: u8, old: u64, v: u64) -> Option<u64> {
    let m = mask(w);
    Some(match op {
        atomic::ADD => old.wrapping_add(v) & m,
        atomic::AND => old & v,
        atomic::OR => old | v,
        atomic::XOR => old ^ v,
        atomic::IMIN => {
            if sext(v, w) < sext(old, w) {
                v
            } else {
                old
            }
        }
        atomic::IMAX => {
            if sext(v, w) > sext(old, w) {
                v
            } else {
                old
            }
        }
        atomic::UMIN => old.min(v),
        atomic::UMAX => old.max(v),
        atomic::EXCHANGE => v,
        _ => return None,
    })
}

/// An atomic that can not change memory whatever it finds there (a poll): not part of the
/// multiset, since how often a wave polls depends on the schedule.
fn atomic_is_poll(op: u8, w: u8, v: u64) -> bool {
    let m = mask(w);
    match op {
        atomic::ADD | atomic::OR | atomic::XOR | atomic::UMAX => v == 0,
        atomic::AND | atomic::UMIN => v == m,
        atomic::IMIN => v == m >> 1,
        atomic::IMAX => v == (m >> 1) + 1,
        _ => false,
    }
}

/// The dx.ops [`Group`] executes.
pub fn supports(op: u32) -> bool {
    min_operands(op) != usize::MAX
}

/// Operands (without the opcode) a dx.op call must have; usize::MAX for one that is not
/// implemented.
pub fn min_operands(op: u32) -> usize {
    match op {
        4 | 5 => 4,
        6..=34 => 1,
        35..=40 => 2,
        46..=52 => 3,
        53 => 4,
        54 => 4,
        55 => 6,
        56 => 8,
        58 => 3,
        59 => 2,
        60 => 10,
        61 => 11,
        62 => 10,
        63 => 16,
        64 => 11,
        65 => 10,
        66 => 8,
        67 => 9,
        68 => 3,
        69 => 8,
        70 => 2,
        71 => 1,
        72 => 2,
        73 => 9,
        74 => 10,
        78 => 6,
        79 => 6,
        80 => 1,
        81 => 6,
        82..=86 => 1,
        87 => 5,
        88 => 4,
        89 => 3,
        90..=92 => 0,
        93..=95 => 1,
        96 => 0,
        101 => 2,
        102 => 1,
        108 => 0,
        110..=112 => 0,
        113..=116 => 1,
        117 => 2,
        118 => 1,
        119 => 3,
        120 => 2,
        121 => 3,
        122 => 2,
        124..=134 => 1,
        135 | 136 => 1,
        138 => 0,
        139 => 5,
        140 => 9,
        163 | 164 => 3,
        165 => 1,
        216 => 2,
        217 => 2,
        218 => 3,
        219 => 2,
        220 => 5,
        221 => 0,
        224 => 11,
        _ => usize::MAX,
    }
}

const KIND_TEXTURE1D: u32 = 1;
const KIND_TEXTURE2D: u32 = 2;
const KIND_TEXTURE2DMS: u32 = 3;
const KIND_TEXTURE3D: u32 = 4;
const KIND_TEXTURECUBE: u32 = 5;
const KIND_TEXTURE1DARRAY: u32 = 6;
const KIND_TEXTURE2DARRAY: u32 = 7;
const KIND_TEXTURE2DMSARRAY: u32 = 8;
const KIND_TEXTURECUBEARRAY: u32 = 9;
const KIND_TYPEDBUFFER: u32 = 10;
const KIND_RAWBUFFER: u32 = 11;
const KIND_STRUCTUREDBUFFER: u32 = 12;

/// Which of x, y, z a texel address of this resource kind uses.
fn texel_dims(kind: u32) -> Option<[bool; 3]> {
    match kind {
        KIND_TEXTURE1D => Some([true, false, false]),
        KIND_TEXTURE2D | KIND_TEXTURE2DMS | KIND_TEXTURE1DARRAY => Some([true, true, false]),
        KIND_TEXTURE3D | KIND_TEXTURE2DARRAY | KIND_TEXTURE2DMSARRAY => Some([true, true, true]),
        _ => None,
    }
}

fn is_texture(kind: u32) -> bool {
    (KIND_TEXTURE1D..=KIND_TEXTURECUBEARRAY).contains(&kind)
}

/// What the operations of one block need besides the registers.
struct Cx<'a> {
    p: &'a Program,
    world: &'a mut World,
    shared: &'a mut [u64],
    shared_owner: &'a mut [u32],
    shared_stamp: &'a mut [u32],
    /// the tick of the wave's last call that may write memory
    clobber: u32,
    frame: &'a mut [u64],
    id: [u32; 3],
    first_thread: u32,
    /// the lanes executing, as a mask
    active: u128,
    /// the wave's number in the group and its count of ballots
    wave: u32,
    ballots: &'a mut u32,
    /// the pixels that were discarded: they run on, as the other pixels of their quads
    /// need them for derivatives, but store nothing
    dead: &'a mut u128,
    ub: u32,
}

impl Cx<'_> {
    /// (identity, resource kind, second property word) of a handle.
    #[inline]
    fn handle(&self, h: u64) -> (u32, u32, u32) {
        match (h >> 32) as usize {
            0 => (h as u32, 0, 0),
            pid => {
                let props = self.p.props[pid - 1];
                (h as u32, props[0] & 0xff, props[1])
            }
        }
    }

    /// The byte address of a buffer element.
    fn buffer_addr(&self, kind: u32, stride: u32, c0: u64, c1: u64) -> Result<u64, Stop> {
        match kind {
            KIND_RAWBUFFER => Ok(c0 & 0xffff_ffff),
            KIND_STRUCTUREDBUFFER => Ok((c0 & 0xffff_ffff) * stride as u64 + (c1 & 0xffff_ffff)),
            KIND_TYPEDBUFFER => Ok((c0 & 0xffff_ffff) * 16),
            _ => Err(Stop::Unsupported(format!("buffer access to a resource of kind {kind}"))),
        }
    }

    /// The cell a pointer names, for a load (`write` false) or a store or atomic.
    fn cell(&mut self, ptr: u64, lane: usize, write: bool) -> Option<&mut u64> {
        let w = self.p.wave_size as usize;
        let cell = (ptr & CELL_MASK) as usize;
        match ptr >> REGION_SHIFT {
            REGION_FRAME if cell < self.p.frame_init.len() => self.frame.get_mut(cell * w + lane),
            REGION_SHARED if cell < self.shared.len() => {
                let clock = &mut self.world.clock;
                clock.me = clock.me & !0xff | lane as u32;
                let (owner, stamp) = (self.shared_owner[cell], self.shared_stamp[cell]);
                if clock.races(owner, stamp) || (!write && clock.same_wave_other_lane(owner) && stamp >= self.clobber) {
                    self.ub |= UB_RACE;
                }
                if write {
                    self.shared_owner[cell] = clock.me;
                    self.shared_stamp[cell] = clock.tick;
                }
                self.shared.get_mut(cell)
            }
            _ => None,
        }
    }
}

impl<'a> Group<'a> {
    /// A thread group with id `id` (SV_GroupID), every lane at the entry.
    pub fn new(p: &'a Program, world: &'a mut World, id: [u32; 3], budget: u64, live: Option<u32>) -> Group<'a> {
        let w = p.wave_size as usize;
        let threads = p.threads() as usize;
        // the clock runs on from the group before: what that one wrote is older than
        // anything this one can race with
        let tick = world.clock.tick.wrapping_add(1);
        let mut waves = Vec::new();
        for k in 0..p.waves() as usize {
            let lanes = (threads - k * w).min(w);
            let mut regs = vec![0u64; p.rows as usize * w];
            for &(row, v) in &p.row_init {
                regs[row as usize * w..(row as usize + 1) * w].fill(v);
            }
            let mut frame = vec![0u64; p.frame_init.len() * w];
            for (cell, &v) in p.frame_init.iter().enumerate() {
                if v != 0 {
                    frame[cell * w..(cell + 1) * w].fill(v);
                }
            }
            waves.push(Wave { regs, frame, next: vec![0; lanes], key: vec![forward_key(0); lanes], held: vec![IDLE; lanes], pred: vec![DONE; lanes], lanes, live: lanes, first_thread: (k * w) as u32, region_start: tick, clobber: tick, ballots: 0, dead: 0 });
        }
        // a partial launch (`Config::live`): the lanes past it never run
        if let Some(live) = live {
            for wave in &mut waves {
                let n = (live as usize).min(wave.lanes);
                for l in n..wave.lanes {
                    wave.next[l] = DONE;
                    wave.key[l] = IDLE;
                }
                wave.live = n;
            }
        }
        world.clock = res::Clock { tick, epoch_start: tick, ..Default::default() };
        world.covered.resize(p.blocks.len(), false);
        let cells = p.shared_cells as usize;
        Group { p, world, shared: vec![0; cells], shared_owner: vec![0; cells], shared_stamp: vec![0; cells], id, waves, steps: 0, budget, ub: 0, tmp: Vec::new() }
    }

    /// Run the group to completion.
    pub fn run(&mut self) -> Result<(), Stop> {
        let mut lanes: Vec<u8> = Vec::with_capacity(self.p.wave_size as usize);
        loop {
            let mut ran = false;
            for wi in 0..self.waves.len() {
                loop {
                    let wave = &self.waves[wi];
                    let best = wave.key.iter().copied().min().unwrap_or(IDLE);
                    if best == IDLE {
                        break;
                    }
                    lanes.clear();
                    lanes.extend((0..wave.lanes).filter(|&l| wave.key[l] == best).map(|l| l as u8));
                    let block = wave.next[lanes[0] as usize];
                    ran = true;
                    if let Flow::Yield = self.block(wi, block, &lanes)? {
                        break;
                    }
                }
            }
            if ran {
                continue;
            }
            // nothing can run: open the barrier, or the group is done
            let mut opened = false;
            for wave in &mut self.waves {
                for l in 0..wave.lanes {
                    if wave.held[l] != IDLE {
                        wave.key[l] = wave.held[l];
                        wave.held[l] = IDLE;
                        opened = true;
                    }
                }
            }
            if !opened {
                let threads = self.p.threads();
                for wave in &self.waves {
                    for l in (0..wave.lanes as u32).filter(|l| wave.dead >> l & 1 != 0) {
                        self.world.discard(self.id[0] * threads + wave.first_thread + l);
                    }
                }
                return Ok(());
            }
            let clock = &mut self.world.clock;
            clock.tick = clock.tick.wrapping_add(1);
            clock.epoch_start = clock.tick;
        }
    }

    fn block(&mut self, wi: usize, bi: u32, lanes: &[u8]) -> Result<Flow, Stop> {
        let p = self.p;
        let w = p.wave_size as usize;
        let blk = &p.blocks[bi as usize];
        self.steps += (blk.end - blk.first + blk.nphis) as u64 * lanes.len() as u64;
        if self.steps > self.budget {
            return Err(Stop::Budget);
        }
        self.world.covered[bi as usize] = true;
        let Wave { regs, frame, next, key, held, pred, live, first_thread, region_start, clobber, ballots, dead, .. } = &mut self.waves[wi];
        let regs = &mut regs[..];

        macro_rules! each {
            (|$l:ident| $body:expr) => {
                for &lane in lanes {
                    let $l = lane as usize;
                    $body;
                }
            };
        }

        if blk.nphis != 0 {
            let preds = &p.preds[blk.pred_first as usize..(blk.pred_first + blk.npreds) as usize];
            let n = blk.npreds as usize;
            let first = pred[lanes[0] as usize];
            let uniform = lanes.iter().all(|&l| pred[l as usize] == first);
            let find = |pred: u32| preds.iter().position(|&x| x == pred).ok_or_else(|| Stop::Unsupported("block entered from a block its phis do not name".into()));
            let dsts = &p.phi_dst[blk.phi_first as usize..(blk.phi_first + blk.nphis) as usize];
            let srcs = &p.phi_src[blk.src_first as usize..blk.src_first as usize + dsts.len() * n];
            if uniform && !blk.parallel {
                let j = find(first)?;
                for (k, &dst) in dsts.iter().enumerate() {
                    let (s, d) = (srcs[k * n + j] as usize * w, dst as usize * w);
                    each!(|l| regs[d + l] = regs[s + l]);
                }
            } else {
                // incoming index per lane, then every phi reads before any is written
                let mut js = [0usize; 128];
                let (mut last, mut last_j) = (DONE, 0);
                for &l in lanes {
                    let from = pred[l as usize];
                    if from != last {
                        last = from;
                        last_j = find(from)?;
                    }
                    js[l as usize] = last_j;
                }
                let nl = lanes.len();
                self.tmp.clear();
                self.tmp.resize(dsts.len() * nl, 0);
                for k in 0..dsts.len() {
                    for (i, &l) in lanes.iter().enumerate() {
                        self.tmp[k * nl + i] = regs[srcs[k * n + js[l as usize]] as usize * w + l as usize];
                    }
                }
                for (k, &dst) in dsts.iter().enumerate() {
                    for (i, &l) in lanes.iter().enumerate() {
                        regs[dst as usize * w + l as usize] = self.tmp[k * nl + i];
                    }
                }
            }
        }

        let mut active = 0u128;
        for &l in lanes {
            active |= 1 << l;
        }
        // a tick per block execution (overflow would need 2^32 of them: the budget ends a
        // run long before)
        let clock = &mut self.world.clock;
        clock.tick = clock.tick.wrapping_add(1);
        clock.block_start = clock.tick;
        if lanes.len() == *live {
            *region_start = clock.tick;
        }
        clock.region_start = *region_start;
        clock.me = (wi as u32 + 1) << 8;
        let mut cx = Cx {
            p,
            world: &mut *self.world,
            shared: &mut self.shared[..],
            shared_owner: &mut self.shared_owner[..],
            shared_stamp: &mut self.shared_stamp[..],
            clobber: *clobber,
            frame: &mut frame[..],
            id: self.id,
            first_thread: *first_thread,
            active,
            wave: wi as u32,
            ballots,
            dead,
            ub: 0,
        };
        macro_rules! clobbered {
            () => {{
                cx.world.clock.tick = cx.world.clock.tick.wrapping_add(1);
                cx.clobber = cx.world.clock.tick;
                *clobber = cx.clobber;
            }};
        }
        let mut flow = Flow::Next;

        for inst in &p.insts[blk.first as usize..blk.end as usize] {
            let d = inst.d as usize * w;
            let a = inst.a as usize * w;
            let b = inst.b as usize * w;
            macro_rules! int2 {
                (|$x:ident, $y:ident| $e:expr) => {{
                    let m = mask(inst.w);
                    each!(|l| {
                        let ($x, $y) = (regs[a + l], regs[b + l]);
                        regs[d + l] = ($e) & m;
                    })
                }};
            }
            macro_rules! float2 {
                (|$x:ident, $y:ident| $e:expr) => {{
                    match inst.w {
                        32 => each!(|l| {
                            let ($x, $y) = (f32::from_bits(regs[a + l] as u32), f32::from_bits(regs[b + l] as u32));
                            regs[d + l] = nan2_32($e, $x, $y).to_bits() as u64;
                        }),
                        64 => each!(|l| {
                            let ($x, $y) = (f64::from_bits(regs[a + l]), f64::from_bits(regs[b + l]));
                            regs[d + l] = nan2_64($e, $x, $y).to_bits();
                        }),
                        _ => each!(|l| {
                            let ($x, $y) = (f16_to_f32(regs[a + l] as u16), f16_to_f32(regs[b + l] as u16));
                            regs[d + l] = f32_to_f16(nan2_32($e, $x, $y)) as u64;
                        }),
                    }
                }};
            }
            match inst.op {
                Opc::Add => int2!(|x, y| x.wrapping_add(y)),
                Opc::Sub => int2!(|x, y| x.wrapping_sub(y)),
                Opc::Mul => int2!(|x, y| x.wrapping_mul(y)),
                Opc::And => int2!(|x, y| x & y),
                Opc::Or => int2!(|x, y| x | y),
                Opc::Xor => int2!(|x, y| x ^ y),
                // DXIL takes shift counts modulo the width
                Opc::Shl => int2!(|x, y| x << (y & (inst.w as u64 - 1))),
                Opc::LShr => int2!(|x, y| x >> (y & (inst.w as u64 - 1))),
                Opc::AShr => int2!(|x, y| (sext(x, inst.w) >> (y & (inst.w as u64 - 1))) as u64),
                // D3D: an unsigned division by zero gives all ones, quotient and remainder;
                // the signed ones are given the same quotient and a zero remainder
                Opc::UDiv => int2!(|x, y| if y == 0 { u64::MAX } else { x / y }),
                Opc::URem => int2!(|x, y| if y == 0 { u64::MAX } else { x % y }),
                Opc::SDiv => int2!(|x, y| if y == 0 { u64::MAX } else { sext(x, inst.w).wrapping_div(sext(y, inst.w)) as u64 }),
                Opc::SRem => int2!(|x, y| if y == 0 { 0 } else { sext(x, inst.w).wrapping_rem(sext(y, inst.w)) as u64 }),
                Opc::FAdd => float2!(|x, y| x + y),
                Opc::FSub => float2!(|x, y| x - y),
                Opc::FMul => float2!(|x, y| x * y),
                Opc::FDiv => float2!(|x, y| x / y),
                Opc::FRem => float2!(|x, y| x % y),
                Opc::ICmp => {
                    let wd = inst.w;
                    if let Some(trace) = &mut cx.world.trace {
                        each!(|l| trace.compare(regs[a + l], regs[b + l]));
                    }
                    macro_rules! cmp {
                        (|$x:ident, $y:ident| $e:expr) => {
                            each!(|l| {
                                let ($x, $y) = (regs[a + l], regs[b + l]);
                                regs[d + l] = ($e) as u64;
                            })
                        };
                    }
                    match inst.aux {
                        32 => cmp!(|x, y| x == y),
                        33 => cmp!(|x, y| x != y),
                        34 => cmp!(|x, y| x > y),
                        35 => cmp!(|x, y| x >= y),
                        36 => cmp!(|x, y| x < y),
                        37 => cmp!(|x, y| x <= y),
                        38 => cmp!(|x, y| sext(x, wd) > sext(y, wd)),
                        39 => cmp!(|x, y| sext(x, wd) >= sext(y, wd)),
                        40 => cmp!(|x, y| sext(x, wd) < sext(y, wd)),
                        _ => cmp!(|x, y| sext(x, wd) <= sext(y, wd)),
                    }
                }
                Opc::FCmp => {
                    if let Some(trace) = &mut cx.world.trace {
                        if inst.w == 32 {
                            each!(|l| trace.compare(regs[a + l], regs[b + l]));
                        }
                    }
                    each!(|l| regs[d + l] = fcmp(inst.aux, widen(regs[a + l], inst.w), widen(regs[b + l], inst.w)) as u64)
                }
                Opc::Select => {
                    let c = inst.c as usize * w;
                    each!(|l| regs[d + l] = if regs[a + l] & 1 != 0 { regs[b + l] } else { regs[c + l] });
                }
                Opc::Trunc => {
                    let m = mask(inst.w);
                    each!(|l| regs[d + l] = regs[a + l] & m);
                }
                Opc::SExt => {
                    let m = mask(inst.aux as u8);
                    each!(|l| regs[d + l] = sext(regs[a + l], inst.w) as u64 & m);
                }
                // D3D float to integer: toward zero, saturating, NaN gives 0
                Opc::FPToUI => {
                    let m = mask(inst.aux as u8);
                    each!(|l| regs[d + l] = (widen(regs[a + l], inst.w) as u64).min(m));
                }
                Opc::FPToSI => {
                    let bits = inst.aux as u32;
                    let (lo, hi) = if bits >= 64 { (i64::MIN, i64::MAX) } else { (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1) };
                    let m = mask(inst.aux as u8);
                    each!(|l| regs[d + l] = (widen(regs[a + l], inst.w) as i64).clamp(lo, hi) as u64 & m);
                }
                Opc::UIToFP => match inst.aux {
                    32 => each!(|l| regs[d + l] = (regs[a + l] as f32).to_bits() as u64),
                    64 => each!(|l| regs[d + l] = (regs[a + l] as f64).to_bits()),
                    _ => each!(|l| regs[d + l] = f64_to_f16(regs[a + l] as f64) as u64),
                },
                Opc::SIToFP => match inst.aux {
                    32 => each!(|l| regs[d + l] = (sext(regs[a + l], inst.w) as f32).to_bits() as u64),
                    64 => each!(|l| regs[d + l] = (sext(regs[a + l], inst.w) as f64).to_bits()),
                    _ => each!(|l| regs[d + l] = f64_to_f16(sext(regs[a + l], inst.w) as f64) as u64),
                },
                Opc::FPTrunc | Opc::FPExt => each!(|l| regs[d + l] = narrow(widen(regs[a + l], inst.w), inst.aux as u8)),
                Opc::GepC => each!(|l| regs[d + l] = regs[a + l].wrapping_add(inst.b as u64)),
                Opc::GepI => {
                    let scale = p.extra[inst.c as usize] as u64;
                    let bound = p.extra[inst.c as usize + 1] as u64;
                    each!(|l| {
                        let mut ix = sext(regs[b + l], inst.w) as u64;
                        if ix >= bound {
                            cx.ub |= UB_LOCAL_BOUNDS;
                            ix = 0;
                        }
                        regs[d + l] = regs[a + l].wrapping_add(ix * scale);
                    });
                }
                Opc::LoadFrame => each!(|l| regs[d + l] = cx.frame[a + l]),
                Opc::StoreFrame => each!(|l| cx.frame[a + l] = regs[b + l]),
                Opc::Load => each!(|l| {
                    let ptr = regs[a + l];
                    regs[d + l] = if ptr >> REGION_SHIFT == REGION_CONST {
                        p.const_mem.get((ptr & CELL_MASK) as usize).copied().unwrap_or_else(|| {
                            cx.ub |= UB_POINTER;
                            0
                        })
                    } else {
                        match cx.cell(ptr, l, false) {
                            Some(c) => *c,
                            None => {
                                cx.ub |= UB_POINTER;
                                0
                            }
                        }
                    };
                }),
                Opc::Store => each!(|l| {
                    let v = regs[b + l];
                    match cx.cell(regs[a + l], l, true) {
                        Some(c) => *c = v,
                        None => cx.ub |= UB_POINTER,
                    }
                }),
                Opc::AtomicRmw => {
                    cx.world.clock.me |= res::ATOMIC_OWNER;
                    // LLVM rmw numbers to DXIL's; sub is an add of the negation
                    let (op, negate) = match inst.aux {
                        0 => (atomic::EXCHANGE, false),
                        1 => (atomic::ADD, false),
                        2 => (atomic::ADD, true),
                        3 => (atomic::AND, false),
                        5 => (atomic::OR, false),
                        6 => (atomic::XOR, false),
                        7 => (atomic::IMAX, false),
                        8 => (atomic::IMIN, false),
                        9 => (atomic::UMAX, false),
                        10 => (atomic::UMIN, false),
                        other => return Err(Stop::Unsupported(format!("atomicrmw operation {other}"))),
                    };
                    let m = mask(inst.w);
                    each!(|l| {
                        let ptr = regs[a + l];
                        let v = if negate { regs[b + l].wrapping_neg() & m } else { regs[b + l] };
                        let old = match cx.cell(ptr, l, true) {
                            Some(c) => {
                                let old = *c;
                                *c = atomic_apply(op, inst.w, old, v).unwrap_or(old);
                                old
                            }
                            None => {
                                cx.ub |= UB_POINTER;
                                0
                            }
                        };
                        if ptr >> REGION_SHIFT == REGION_SHARED && !atomic_is_poll(op, inst.w, v) {
                            cx.world.atomics.record(Atomic { res: res::SHARED, addr: p.shared_addr.get((ptr & CELL_MASK) as usize).copied().unwrap_or(u64::MAX), op, a: v, b: 0 });
                        }
                        regs[d + l] = old;
                    });
                    cx.world.clock.me &= !res::ATOMIC_OWNER;
                    clobbered!();
                }
                Opc::CmpXchg => {
                    cx.world.clock.me |= res::ATOMIC_OWNER;
                    let c = inst.c as usize * w;
                    each!(|l| {
                        let ptr = regs[a + l];
                        let (expected, new) = (regs[b + l], regs[c + l]);
                        let (old, ok) = match cx.cell(ptr, l, true) {
                            Some(cell) => {
                                let old = *cell;
                                if old == expected {
                                    *cell = new;
                                }
                                (old, old == expected)
                            }
                            None => {
                                cx.ub |= UB_POINTER;
                                (0, false)
                            }
                        };
                        if ptr >> REGION_SHIFT == REGION_SHARED {
                            cx.world.atomics.record(Atomic { res: res::SHARED, addr: p.shared_addr.get((ptr & CELL_MASK) as usize).copied().unwrap_or(u64::MAX), op: res::ATOMIC_CMPXCHG, a: expected, b: new });
                        }
                        regs[d + l] = old;
                        regs[d + w + l] = ok as u64;
                    });
                    cx.world.clock.me &= !res::ATOMIC_OWNER;
                    clobbered!();
                }
                Opc::Dx => {
                    dx(&mut cx, inst, regs, lanes)?;
                    cx.world.clock.me &= !res::ATOMIC_OWNER;
                    if inst.c != 0 {
                        clobbered!();
                    }
                }
                Opc::Br | Opc::Yield => {
                    let t = inst.a;
                    let k = if t <= bi { back_key(p.blocks[t as usize].loop_end, t) } else { forward_key(t) };
                    each!(|l| {
                        next[l] = t;
                        key[l] = k;
                        pred[l] = bi;
                    });
                    if inst.op == Opc::Yield {
                        flow = Flow::Yield;
                    }
                }
                Opc::CondBr => {
                    let key_of = |t: u32| if t <= bi { back_key(p.blocks[t as usize].loop_end, t) } else { forward_key(t) };
                    let (t, f) = (inst.b, inst.c);
                    let (kt, kf) = (key_of(t), key_of(f));
                    each!(|l| {
                        let taken = regs[a + l] & 1 != 0;
                        next[l] = if taken { t } else { f };
                        key[l] = if taken { kt } else { kf };
                        pred[l] = bi;
                    });
                }
                Opc::Switch => {
                    let cases = &p.switches[inst.c as usize];
                    if let Some(trace) = &mut cx.world.trace {
                        let v = regs[a + lanes[0] as usize];
                        for c in cases.iter().take(16) {
                            trace.compare(v, c.0);
                        }
                    }
                    each!(|l| {
                        let v = regs[a + l];
                        let t = match cases.binary_search_by_key(&v, |c| c.0) {
                            Ok(k) => cases[k].1,
                            Err(_) => inst.b,
                        };
                        next[l] = t;
                        key[l] = if t <= bi { back_key(p.blocks[t as usize].loop_end, t) } else { forward_key(t) };
                        pred[l] = bi;
                    });
                }
                Opc::Ret | Opc::Unreachable => {
                    if inst.op == Opc::Unreachable {
                        cx.ub |= UB_UNREACHABLE;
                    }
                    each!(|l| {
                        next[l] = DONE;
                        key[l] = IDLE;
                    });
                    *live -= lanes.len();
                }
                Opc::Barrier => each!(|l| {
                    next[l] = inst.a;
                    key[l] = IDLE;
                    held[l] = forward_key(inst.a);
                    pred[l] = bi;
                }),
            }
        }
        self.ub |= cx.ub;
        if self.world.races != 0 {
            self.ub |= UB_RACE;
        }
        Ok(flow)
    }
}

/// One dx.op for the lanes of a block.
fn dx(cx: &mut Cx, inst: &DInst, regs: &mut [u64], lanes: &[u8]) -> Result<(), Stop> {
    let p = cx.p;
    let w = p.wave_size as usize;
    let d = inst.d as usize * w;
    let ex = &p.extra[inst.a as usize..(inst.a + inst.b) as usize];
    let op = inst.aux as u32;
    let ovl = inst.w;
    let bits = ovl & !OVL_FLOAT;
    let float = ovl & OVL_FLOAT != 0;
    let m = mask(bits);
    let owner = cx.world.clock.me & !0xff | if matches!(op, ATOMIC_BIN_OP | ATOMIC_CMPXCHG | BUFFER_UPDATE_COUNTER) { res::ATOMIC_OWNER } else { 0 };
    // operand k of a lane
    macro_rules! arg {
        ($k:expr, $l:expr) => {
            regs[ex[$k] as usize * w + $l]
        };
    }
    macro_rules! each {
        (|$l:ident| $body:expr) => {
            for &lane in lanes {
                let $l = lane as usize;
                cx.world.clock.me = owner | lane as u32;
                $body;
            }
        };
    }
    const FABS: u32 = OpCode::FAbs as u32;
    const ISNAN: u32 = OpCode::IsNaN as u32;
    const ISINF: u32 = OpCode::IsInf as u32;
    const ISFINITE: u32 = OpCode::IsFinite as u32;
    const ISNORMAL: u32 = OpCode::IsNormal as u32;
    const ROUND_Z: u32 = OpCode::Round_z as u32;
    const BFREV: u32 = OpCode::Bfrev as u32;
    const COUNTBITS: u32 = OpCode::Countbits as u32;
    const FIRSTBIT_LO: u32 = OpCode::FirstbitLo as u32;
    const FIRSTBIT_HI: u32 = OpCode::FirstbitHi as u32;
    const FIRSTBIT_SHI: u32 = OpCode::FirstbitSHi as u32;
    const FMAX: u32 = OpCode::FMax as u32;
    const FMIN: u32 = OpCode::FMin as u32;
    const IMAX: u32 = OpCode::IMax as u32;
    const IMIN: u32 = OpCode::IMin as u32;
    const UMAX: u32 = OpCode::UMax as u32;
    const UMIN: u32 = OpCode::UMin as u32;
    const FMAD: u32 = OpCode::FMad as u32;
    const FMA: u32 = OpCode::Fma as u32;
    const IMAD: u32 = OpCode::IMad as u32;
    const UMAD: u32 = OpCode::UMad as u32;
    const MSAD: u32 = OpCode::Msad as u32;
    const IBFE: u32 = OpCode::Ibfe as u32;
    const UBFE: u32 = OpCode::Ubfe as u32;
    const BFI: u32 = OpCode::Bfi as u32;
    const DOT2: u32 = OpCode::Dot2 as u32;
    const DOT4: u32 = OpCode::Dot4 as u32;
    const CBUFFER_LOAD: u32 = OpCode::CBufferLoad as u32;
    const CBUFFER_LOAD_LEGACY: u32 = OpCode::CBufferLoadLegacy as u32;
    const SAMPLE: u32 = OpCode::Sample as u32;
    const SAMPLE_BIAS: u32 = OpCode::SampleBias as u32;
    const SAMPLE_CMP: u32 = OpCode::SampleCmp as u32;
    const SAMPLE_CMP_LEVEL_ZERO: u32 = OpCode::SampleCmpLevelZero as u32;
    const SAMPLE_CMP_LEVEL: u32 = OpCode::SampleCmpLevel as u32;
    const TEXTURE_LOAD: u32 = OpCode::TextureLoad as u32;
    const TEXTURE_STORE: u32 = OpCode::TextureStore as u32;
    const BUFFER_LOAD: u32 = OpCode::BufferLoad as u32;
    const BUFFER_STORE: u32 = OpCode::BufferStore as u32;
    const BUFFER_UPDATE_COUNTER: u32 = OpCode::BufferUpdateCounter as u32;
    const CHECK_ACCESS: u32 = OpCode::CheckAccessFullyMapped as u32;
    const GET_DIMENSIONS: u32 = OpCode::GetDimensions as u32;
    const TEXTURE_GATHER: u32 = OpCode::TextureGather as u32;
    const TEXTURE_GATHER_CMP: u32 = OpCode::TextureGatherCmp as u32;
    const ATOMIC_BIN_OP: u32 = OpCode::AtomicBinOp as u32;
    const ATOMIC_CMPXCHG: u32 = OpCode::AtomicCompareExchange as u32;
    const BARRIER: u32 = OpCode::Barrier as u32;
    const CALCULATE_LOD: u32 = OpCode::CalculateLOD as u32;
    const THREAD_ID: u32 = OpCode::ThreadId as u32;
    const GROUP_ID: u32 = OpCode::GroupId as u32;
    const THREAD_ID_IN_GROUP: u32 = OpCode::ThreadIdInGroup as u32;
    const FLATTENED_THREAD_ID: u32 = OpCode::FlattenedThreadIdInGroup as u32;
    const MAKE_DOUBLE: u32 = OpCode::MakeDouble as u32;
    const SPLIT_DOUBLE: u32 = OpCode::SplitDouble as u32;
    const WAVE_IS_FIRST_LANE: u32 = OpCode::WaveIsFirstLane as u32;
    const WAVE_GET_LANE_INDEX: u32 = OpCode::WaveGetLaneIndex as u32;
    const WAVE_GET_LANE_COUNT: u32 = OpCode::WaveGetLaneCount as u32;
    const WAVE_ANY_TRUE: u32 = OpCode::WaveAnyTrue as u32;
    const WAVE_ALL_TRUE: u32 = OpCode::WaveAllTrue as u32;
    const WAVE_ALL_EQUAL: u32 = OpCode::WaveActiveAllEqual as u32;
    const WAVE_BALLOT: u32 = OpCode::WaveActiveBallot as u32;
    const WAVE_READ_LANE_AT: u32 = OpCode::WaveReadLaneAt as u32;
    const WAVE_READ_LANE_FIRST: u32 = OpCode::WaveReadLaneFirst as u32;
    const WAVE_ACTIVE_OP: u32 = OpCode::WaveActiveOp as u32;
    const WAVE_ACTIVE_BIT: u32 = OpCode::WaveActiveBit as u32;
    const WAVE_PREFIX_OP: u32 = OpCode::WavePrefixOp as u32;
    const QUAD_READ_LANE_AT: u32 = OpCode::QuadReadLaneAt as u32;
    const F32_TO_F16: u32 = OpCode::LegacyF32ToF16 as u32;
    const F16_TO_F32: u32 = OpCode::LegacyF16ToF32 as u32;
    const DOUBLE_TO_FLOAT: u32 = OpCode::LegacyDoubleToFloat as u32;
    const DOUBLE_TO_SINT: u32 = OpCode::LegacyDoubleToSInt32 as u32;
    const DOUBLE_TO_UINT: u32 = OpCode::LegacyDoubleToUInt32 as u32;
    const WAVE_ALL_BIT_COUNT: u32 = OpCode::WaveAllBitCount as u32;
    const WAVE_PREFIX_BIT_COUNT: u32 = OpCode::WavePrefixBitCount as u32;
    const RAW_BUFFER_LOAD: u32 = OpCode::RawBufferLoad as u32;
    const RAW_BUFFER_STORE: u32 = OpCode::RawBufferStore as u32;
    const DOT4_ADD_I8: u32 = OpCode::Dot4AddI8Packed as u32;
    const DOT4_ADD_U8: u32 = OpCode::Dot4AddU8Packed as u32;
    const WAVE_MATCH: u32 = OpCode::WaveMatch as u32;
    const ANNOTATE_HANDLE: u32 = OpCode::AnnotateHandle as u32;
    const HANDLE_FROM_BINDING: u32 = OpCode::CreateHandleFromBinding as u32;
    const HANDLE_FROM_HEAP: u32 = OpCode::CreateHandleFromHeap as u32;
    const UNPACK4X8: u32 = OpCode::Unpack4x8 as u32;
    const PACK4X8: u32 = OpCode::Pack4x8 as u32;
    const IS_HELPER_LANE: u32 = OpCode::IsHelperLane as u32;
    const LOAD_INPUT: u32 = OpCode::LoadInput as u32;
    const STORE_OUTPUT: u32 = OpCode::StoreOutput as u32;
    const DISCARD: u32 = OpCode::Discard as u32;
    const DERIV_COARSE_X: u32 = OpCode::DerivCoarseX as u32;
    const DERIV_COARSE_Y: u32 = DERIV_COARSE_X + 1;
    const DERIV_FINE_X: u32 = OpCode::DerivFineX as u32;
    const DERIV_FINE_Y: u32 = DERIV_FINE_X + 1;
    const EVAL_SNAPPED: u32 = OpCode::EvalSnapped as u32;
    const EVAL_SAMPLE_INDEX: u32 = OpCode::EvalSampleIndex as u32;
    const EVAL_CENTROID: u32 = OpCode::EvalCentroid as u32;
    const COVERAGE: u32 = OpCode::Coverage as u32;
    const INNER_COVERAGE: u32 = OpCode::InnerCoverage as u32;
    const SAMPLE_INDEX: u32 = OpCode::SampleIndex as u32;
    const VIEW_ID: u32 = OpCode::ViewID as u32;
    const PRIMITIVE_ID: u32 = OpCode::PrimitiveID as u32;

    let unsupported = |what: &str| Err(Stop::Unsupported(format!("dx.op {op}: {what}")));
    // a discarded pixel writes nothing: its stores are dropped and its atomics return 0
    let mut alive = [0u8; 128];
    let lanes = if *cx.dead != 0 && matches!(op, RAW_BUFFER_STORE | BUFFER_STORE | TEXTURE_STORE | ATOMIC_BIN_OP | ATOMIC_CMPXCHG | BUFFER_UPDATE_COUNTER | STORE_OUTPUT) {
        let mut n = 0;
        for &lane in lanes {
            if *cx.dead >> lane & 1 == 0 {
                alive[n] = lane;
                n += 1;
            } else if inst.d != NONE {
                regs[d + lane as usize] = 0;
            }
        }
        &alive[..n]
    } else {
        lanes
    };
    match op {
        ISNAN | ISINF | ISFINITE | ISNORMAL => each!(|l| {
            let x = widen(arg!(0, l), bits);
            let normal = match bits {
                64 => x.is_normal(),
                32 => (x as f32).is_normal(),
                _ => x.is_finite() && x.abs() >= 6.103515625e-5,
            };
            regs[d + l] = match op {
                ISNAN => x.is_nan(),
                ISINF => x.is_infinite(),
                ISFINITE => x.is_finite(),
                _ => normal,
            } as u64;
        }),
        FABS..=ROUND_Z => match bits {
            32 => each!(|l| regs[d + l] = unary32(op, f32::from_bits(arg!(0, l) as u32)).to_bits() as u64),
            64 => each!(|l| regs[d + l] = unary64(op, f64::from_bits(arg!(0, l))).to_bits()),
            _ => each!(|l| regs[d + l] = f32_to_f16(unary32(op, f16_to_f32(arg!(0, l) as u16))) as u64),
        },
        BFREV => each!(|l| regs[d + l] = (arg!(0, l) & m).reverse_bits() >> (64 - bits as u32)),
        COUNTBITS => each!(|l| regs[d + l] = (arg!(0, l) & m).count_ones() as u64),
        // -1 when no bit is set; the high ones count from the top bit down (DXIL's
        // definition; DXC subtracts from the width for HLSL's firstbithigh)
        FIRSTBIT_LO => each!(|l| {
            let x = arg!(0, l) & m;
            regs[d + l] = if x == 0 { 0xffff_ffff } else { x.trailing_zeros() as u64 };
        }),
        FIRSTBIT_HI | FIRSTBIT_SHI => each!(|l| {
            let mut x = arg!(0, l) & m;
            if op == FIRSTBIT_SHI && x >> (bits - 1) != 0 {
                x = !x & m;
            }
            regs[d + l] = if x == 0 { 0xffff_ffff } else { (x.leading_zeros() - (64 - bits as u32)) as u64 };
        }),
        FMAX | FMIN => each!(|l| {
            let (x, y) = (arg!(0, l), arg!(1, l));
            let (fx, fy) = (widen(x, bits), widen(y, bits));
            let first = if op == FMAX { fmax(fx, fy, fx.is_nan(), fy.is_nan()) } else { fmin(fx, fy, fx.is_nan(), fy.is_nan()) };
            // the operand itself, bit for bit (the sign of a zero, a NaN's payload)
            regs[d + l] = if first.to_bits() == fx.to_bits() { x } else { y };
        }),
        IMAX => each!(|l| regs[d + l] = if sext(arg!(0, l), bits) > sext(arg!(1, l), bits) { arg!(0, l) } else { arg!(1, l) }),
        IMIN => each!(|l| regs[d + l] = if sext(arg!(0, l), bits) < sext(arg!(1, l), bits) { arg!(0, l) } else { arg!(1, l) }),
        UMAX => each!(|l| regs[d + l] = arg!(0, l).max(arg!(1, l))),
        UMIN => each!(|l| regs[d + l] = arg!(0, l).min(arg!(1, l))),
        FMAD => each!(|l| {
            regs[d + l] = match bits {
                32 => {
                    let (x, y, z) = (f32::from_bits(arg!(0, l) as u32), f32::from_bits(arg!(1, l) as u32), f32::from_bits(arg!(2, l) as u32));
                    let prod = nan2_32(x * y, x, y);
                    nan2_32(prod + z, prod, z).to_bits() as u64
                }
                64 => {
                    let (x, y, z) = (f64::from_bits(arg!(0, l)), f64::from_bits(arg!(1, l)), f64::from_bits(arg!(2, l)));
                    let prod = nan2_64(x * y, x, y);
                    nan2_64(prod + z, prod, z).to_bits()
                }
                _ => {
                    let (x, y, z) = (f16_to_f32(arg!(0, l) as u16), f16_to_f32(arg!(1, l) as u16), f16_to_f32(arg!(2, l) as u16));
                    let prod = f16_to_f32(f32_to_f16(nan2_32(x * y, x, y)));
                    f32_to_f16(nan2_32(prod + z, prod, z)) as u64
                }
            };
        }),
        FMA => each!(|l| {
            let (x, y, z) = (widen(arg!(0, l), bits), widen(arg!(1, l), bits), widen(arg!(2, l), bits));
            let r = x.mul_add(y, z);
            let r = if r.is_nan() { nan2_64(nan2_64(r, x, y), if x.is_nan() || y.is_nan() { x } else { z }, z) } else { r };
            regs[d + l] = narrow(r, bits);
        }),
        IMAD | UMAD => each!(|l| regs[d + l] = arg!(0, l).wrapping_mul(arg!(1, l)).wrapping_add(arg!(2, l)) & m),
        MSAD => each!(|l| {
            let (reference, src, mut acc) = (arg!(0, l) as u32, arg!(1, l) as u32, arg!(2, l) as u32);
            for k in 0..4 {
                let (r, s) = ((reference >> (8 * k)) & 0xff, (src >> (8 * k)) & 0xff);
                if r != 0 {
                    acc = acc.wrapping_add(r.abs_diff(s));
                }
            }
            regs[d + l] = acc as u64;
        }),
        IBFE | UBFE => each!(|l| regs[d + l] = bfe(bits as u32, arg!(0, l), arg!(1, l), arg!(2, l), op == IBFE)),
        BFI => each!(|l| regs[d + l] = bfi(bits as u32, arg!(0, l), arg!(1, l), arg!(2, l), arg!(3, l))),
        DOT2..=DOT4 => {
            // DXC's folder: a sum from +0 of the products, each rounded, in order
            let n = (op - DOT2) as usize + 2;
            each!(|l| {
                regs[d + l] = match bits {
                    32 => {
                        let mut sum = 0f32;
                        for k in 0..n {
                            let (x, y) = (f32::from_bits(arg!(k, l) as u32), f32::from_bits(arg!(n + k, l) as u32));
                            let prod = nan2_32(x * y, x, y);
                            sum = nan2_32(sum + prod, sum, prod);
                        }
                        sum.to_bits() as u64
                    }
                    _ => {
                        let mut sum = 0f64;
                        for k in 0..n {
                            let (x, y) = (widen(arg!(k, l), bits), widen(arg!(n + k, l), bits));
                            let prod = widen(narrow(nan2_64(x * y, x, y), bits), bits);
                            sum = widen(narrow(nan2_64(sum + prod, sum, prod), bits), bits);
                        }
                        narrow(sum, bits)
                    }
                };
            });
        }
        CBUFFER_LOAD_LEGACY => each!(|l| {
            let (ident, _, _) = cx.handle(arg!(0, l));
            let row = (arg!(1, l) & 0xffff_ffff) * 4;
            match bits {
                64 => {
                    for k in 0..2u64 {
                        regs[d + k as usize * w + l] = cx.world.constant(ident, row + 2 * k) as u64 | (cx.world.constant(ident, row + 2 * k + 1) as u64) << 32;
                    }
                }
                16 => {
                    for k in 0..8u64 {
                        regs[d + k as usize * w + l] = (cx.world.constant(ident, row + k / 2) >> (16 * (k & 1))) as u64 & 0xffff;
                    }
                }
                _ => {
                    for k in 0..4u64 {
                        regs[d + k as usize * w + l] = cx.world.constant(ident, row + k) as u64;
                    }
                }
            }
        }),
        CBUFFER_LOAD => each!(|l| {
            let (ident, _, _) = cx.handle(arg!(0, l));
            let word = (arg!(1, l) & 0xffff_ffff) / 4;
            let lo = cx.world.constant(ident, word) as u64;
            regs[d + l] = if bits == 64 { lo | (cx.world.constant(ident, word + 1) as u64) << 32 } else { lo & m };
        }),
        SAMPLE..=SAMPLE_CMP_LEVEL_ZERO | SAMPLE_CMP_LEVEL | TEXTURE_GATHER | TEXTURE_GATHER_CMP | CALCULATE_LOD => {
            if bits != 32 && op != CALCULATE_LOD {
                return unsupported("a sample that is not of floats");
            }
            each!(|l| {
                let (texture, kind, _) = cx.handle(arg!(0, l));
                let (sampler, _, _) = cx.handle(arg!(1, l));
                if !is_texture(kind) {
                    return unsupported("sample of a resource that is not a texture");
                }
                let mut operands = [0u64; 24];
                let mut n = (ex.len() - 2).min(16);
                for (k, o) in operands[..n].iter_mut().enumerate() {
                    *o = arg!(k + 2, l);
                }
                if cx.p.stage == Stage::Pixel && matches!(op, SAMPLE | SAMPLE_BIAS | SAMPLE_CMP | CALCULATE_LOD) {
                    // the level comes from the quad: the result is also a function of
                    // the coordinates' derivatives across it
                    let quad = l & !3;
                    if cx.active >> quad & 0xf != 0xf {
                        cx.ub |= UB_INACTIVE_LANE;
                    }
                    for k in 2..if op == CALCULATE_LOD { 5 } else { 6 } {
                        let c = |lane: usize| f32::from_bits(arg!(k, lane) as u32);
                        for to in [quad + 1, quad + 2] {
                            operands[n] = nan2_32(c(to) - c(quad), c(to), c(quad)).to_bits() as u64;
                            n += 1;
                        }
                    }
                }
                let v = cx.world.sample(texture, sampler, op, &operands[..n]);
                if op == CALCULATE_LOD {
                    regs[d + l] = (f32::from_bits(v[0]) * 8.0).to_bits() as u64;
                } else {
                    for k in 0..4 {
                        regs[d + k * w + l] = v[k] as u64;
                    }
                    regs[d + 4 * w + l] = 0;
                }
            });
        }
        TEXTURE_LOAD => each!(|l| {
            let (ident, kind, _) = cx.handle(arg!(0, l));
            let Some(used) = texel_dims(kind) else { return unsupported("load from a resource that is not a 1D, 2D or 3D texture") };
            let mip = if kind == KIND_TEXTURE2DMS || kind == KIND_TEXTURE2DMSARRAY { 0 } else { arg!(1, l) as u32 };
            let coord = [arg!(2, l).wrapping_add(arg!(5, l)) as u32, arg!(3, l).wrapping_add(arg!(6, l)) as u32, arg!(4, l).wrapping_add(arg!(7, l)) as u32];
            let v = cx.world.texel(ident, coord, mip, used, float);
            for k in 0..4 {
                regs[d + k * w + l] = v[k] as u64 & m;
            }
            regs[d + 4 * w + l] = 0;
        }),
        TEXTURE_STORE => each!(|l| {
            let (ident, kind, _) = cx.handle(arg!(0, l));
            let Some(used) = texel_dims(kind) else { return unsupported("store to a resource that is not a 1D, 2D or 3D texture") };
            let coord = [arg!(1, l) as u32, arg!(2, l) as u32, arg!(3, l) as u32];
            let v = [arg!(4, l) as u32, arg!(5, l) as u32, arg!(6, l) as u32, arg!(7, l) as u32];
            cx.world.set_texel(ident, coord, 0, used, v);
        }),
        BUFFER_LOAD => each!(|l| {
            let (ident, kind, stride) = cx.handle(arg!(0, l));
            let addr = cx.buffer_addr(kind, stride, arg!(1, l), arg!(2, l))?;
            for k in 0..4 {
                regs[d + k * w + l] = cx.world.read(ident, addr + 4 * k as u64, 4) & m;
            }
            regs[d + 4 * w + l] = 0;
        }),
        BUFFER_STORE => each!(|l| {
            let (ident, kind, stride) = cx.handle(arg!(0, l));
            let addr = cx.buffer_addr(kind, stride, arg!(1, l), arg!(2, l))?;
            let write_mask = arg!(7, l);
            for k in 0..4 {
                if write_mask >> k & 1 != 0 {
                    cx.world.write(ident, addr + 4 * k as u64, 4, arg!(3 + k, l));
                }
            }
        }),
        RAW_BUFFER_LOAD => {
            let size = (bits as u32 / 8).max(1);
            each!(|l| {
                let (ident, kind, stride) = cx.handle(arg!(0, l));
                let addr = cx.buffer_addr(kind, stride, arg!(1, l), arg!(2, l))?;
                let read_mask = arg!(3, l);
                for k in 0..4 {
                    regs[d + k * w + l] = if read_mask >> k & 1 != 0 { cx.world.read(ident, addr + (size * k as u32) as u64, size) } else { 0 };
                }
                regs[d + 4 * w + l] = 0;
            });
        }
        RAW_BUFFER_STORE => {
            let size = (bits as u32 / 8).max(1);
            each!(|l| {
                let (ident, kind, stride) = cx.handle(arg!(0, l));
                let addr = cx.buffer_addr(kind, stride, arg!(1, l), arg!(2, l))?;
                let write_mask = arg!(7, l);
                for k in 0..4 {
                    if write_mask >> k & 1 != 0 {
                        cx.world.write(ident, addr + (size * k as u32) as u64, size, arg!(3 + k, l));
                    }
                }
            });
        }
        BUFFER_UPDATE_COUNTER => each!(|l| {
            let (ident, _, _) = cx.handle(arg!(0, l));
            let delta = arg!(1, l) as u8 as i8 as i32;
            let (before, after) = cx.world.counter(ident, delta);
            cx.world.atomics.record(Atomic { res: ident, addr: 0, op: res::ATOMIC_COUNTER, a: delta as u64, b: 0 });
            regs[d + l] = if delta > 0 { before } else { after } as u64;
        }),
        CHECK_ACCESS => each!(|l| regs[d + l] = 1),
        GET_DIMENSIONS => each!(|l| {
            let (ident, kind, stride) = cx.handle(arg!(0, l));
            let mip = arg!(1, l) as u32;
            let out = match kind {
                KIND_RAWBUFFER => [cx.world.buffer_size(ident) as u32, 0, 0, 0],
                KIND_STRUCTUREDBUFFER => [(cx.world.buffer_size(ident) / (stride as u64).max(1)) as u32, 0, 0, 0],
                KIND_TYPEDBUFFER => [(cx.world.buffer_size(ident) / 16) as u32, 0, 0, 0],
                _ if is_texture(kind) => {
                    let t = cx.world.texture_dims(ident);
                    if mip >= t[3] {
                        [0, 0, 0, t[3]]
                    } else {
                        let (tw, th) = ((t[0] >> mip).max(1), (t[1] >> mip).max(1));
                        match kind {
                            KIND_TEXTURE1D => [tw, 0, 0, t[3]],
                            KIND_TEXTURE1DARRAY => [tw, t[2], 0, t[3]],
                            KIND_TEXTURE2D | KIND_TEXTURE2DMS => [tw, th, 0, t[3]],
                            KIND_TEXTURECUBE => [tw, tw, 0, t[3]],
                            KIND_TEXTURECUBEARRAY => [tw, tw, t[2], t[3]],
                            _ => [tw, th, t[2], t[3]],
                        }
                    }
                }
                _ => return unsupported("dimensions of this resource kind"),
            };
            for k in 0..4 {
                regs[d + k * w + l] = out[k] as u64;
            }
        }),
        ATOMIC_BIN_OP | ATOMIC_CMPXCHG => {
            let size = (bits as u32 / 8).max(1);
            let binop = op == ATOMIC_BIN_OP;
            // the coordinates start after the operation of a binary atomic
            let c = if binop { 2 } else { 1 };
            each!(|l| {
                let (ident, kind, stride) = cx.handle(arg!(0, l));
                let (rmw, v, v2) = if binop { (arg!(1, l) as u8, arg!(5, l) & m, 0) } else { (res::ATOMIC_CMPXCHG, arg!(4, l) & m, arg!(5, l) & m) };
                let apply = |old: u64| -> Result<u64, Stop> {
                    if binop {
                        atomic_apply(rmw, bits, old, v).ok_or_else(|| Stop::Unsupported(format!("atomic operation {rmw}")))
                    } else {
                        Ok(if old == v { v2 } else { old })
                    }
                };
                let old = if let Some(used) = texel_dims(kind) {
                    let coord = [arg!(c, l) as u32, arg!(c + 1, l) as u32, arg!(c + 2, l) as u32];
                    let mut texel = cx.world.texel(ident, coord, 0, used, false);
                    let old = texel[0] as u64 & m;
                    texel[0] = apply(old)? as u32;
                    cx.world.set_texel(ident, coord, 0, used, texel);
                    (coord[0] as u64 | (coord[1] as u64) << 16 | (coord[2] as u64) << 32, old)
                } else {
                    let addr = cx.buffer_addr(kind, stride, arg!(c, l), arg!(c + 1, l))?;
                    let old = cx.world.read(ident, addr, size);
                    cx.world.write(ident, addr, size, apply(old)?);
                    (addr, old)
                };
                if !(binop && atomic_is_poll(rmw, bits, v)) {
                    cx.world.atomics.record(Atomic { res: ident, addr: old.0, op: rmw, a: v, b: v2 });
                }
                regs[d + l] = old.1;
            });
        }
        BARRIER => {}
        THREAD_ID | GROUP_ID | THREAD_ID_IN_GROUP | FLATTENED_THREAD_ID => each!(|l| {
            let t = cx.first_thread + l as u32;
            let n = p.numthreads;
            let in_group = [t % n[0], t / n[0] % n[1], t / (n[0] * n[1])];
            let c = if op == FLATTENED_THREAD_ID { 0 } else { (arg!(0, l) as usize).min(2) };
            regs[d + l] = match op {
                THREAD_ID => cx.id[c].wrapping_mul(n[c]).wrapping_add(in_group[c]),
                GROUP_ID => cx.id[c],
                THREAD_ID_IN_GROUP => in_group[c],
                _ => t,
            } as u64;
        }),
        MAKE_DOUBLE => each!(|l| regs[d + l] = arg!(0, l) & 0xffff_ffff | arg!(1, l) << 32),
        SPLIT_DOUBLE => each!(|l| {
            let x = arg!(0, l);
            regs[d + l] = x & 0xffff_ffff;
            regs[d + w + l] = x >> 32;
        }),
        F32_TO_F16 => each!(|l| regs[d + l] = f32_to_f16(f32::from_bits(arg!(0, l) as u32)) as u64),
        F16_TO_F32 => each!(|l| regs[d + l] = f16_to_f32(arg!(0, l) as u16).to_bits() as u64),
        DOUBLE_TO_FLOAT => each!(|l| regs[d + l] = (f64::from_bits(arg!(0, l)) as f32).to_bits() as u64),
        DOUBLE_TO_SINT => each!(|l| regs[d + l] = (f64::from_bits(arg!(0, l)) as i32) as u32 as u64),
        DOUBLE_TO_UINT => each!(|l| regs[d + l] = (f64::from_bits(arg!(0, l)) as u32) as u64),
        DOT4_ADD_I8 | DOT4_ADD_U8 => each!(|l| {
            let (mut acc, x, y) = (arg!(0, l) as u32, arg!(1, l) as u32, arg!(2, l) as u32);
            for k in 0..4 {
                let (xb, yb) = ((x >> (8 * k)) as u8, (y >> (8 * k)) as u8);
                acc = acc.wrapping_add(if op == DOT4_ADD_I8 { (xb as i8 as i32 * yb as i8 as i32) as u32 } else { xb as u32 * yb as u32 });
            }
            regs[d + l] = acc as u64;
        }),
        UNPACK4X8 => each!(|l| {
            let (signed, pk) = (arg!(0, l) & 1 != 0, arg!(1, l) as u32);
            for k in 0..4 {
                let byte = (pk >> (8 * k)) as u8;
                regs[d + k * w + l] = if signed { byte as i8 as i64 as u64 & m } else { byte as u64 };
            }
        }),
        PACK4X8 => each!(|l| {
            let mode = arg!(0, l);
            let mut out = 0u64;
            for k in 0..4 {
                let x = sext(arg!(1 + k, l), bits);
                let byte = match mode {
                    1 => x.clamp(0, 255) as u64,
                    2 => x.clamp(-128, 127) as u64 & 0xff,
                    _ => x as u64 & 0xff,
                };
                out |= byte << (8 * k);
            }
            regs[d + l] = out;
        }),
        IS_HELPER_LANE => each!(|l| regs[d + l] = (*cx.dead >> l & 1) as u64),
        LOAD_INPUT | EVAL_SNAPPED | EVAL_SAMPLE_INDEX | EVAL_CENTROID => {
            if cx.p.stage == Stage::Compute || bits == 64 {
                return unsupported("an input of this stage or width");
            }
            let pixel = cx.p.stage == Stage::Pixel;
            each!(|l| {
                // an attribute is evaluated at the pixel's centre wherever it is asked for
                let element = arg!(0, l) as u32;
                let (kind, flat) = cx.p.inputs.iter().find(|e| e.id == element).map_or((0, false), |e| (e.kind, e.interp == 1));
                let v = cx.world.input(pixel, kind, flat, element, arg!(1, l) as u32, arg!(2, l) as u32, cx.id[0] * w as u32 + l as u32, float);
                regs[d + l] = if float && bits == 16 { f32_to_f16(f32::from_bits(v)) as u64 } else { v as u64 & m };
            })
        }
        STORE_OUTPUT => {
            if cx.p.stage == Stage::Compute || bits == 64 {
                return unsupported("an output of this stage or width");
            }
            each!(|l| {
                let v = arg!(3, l);
                let v = if float && bits == 16 { f16_to_f32(v as u16).to_bits() } else { (v & m) as u32 };
                cx.world.store_output(arg!(0, l) as u32, arg!(1, l) as u32, arg!(2, l) as u32, cx.id[0] * w as u32 + l as u32, v);
            })
        }
        DISCARD => {
            if cx.p.stage != Stage::Pixel {
                return unsupported("discard outside a pixel shader");
            }
            each!(|l| if arg!(0, l) & 1 != 0 {
                *cx.dead |= 1 << l;
            })
        }
        DERIV_COARSE_X..=DERIV_FINE_Y => {
            if cx.p.stage != Stage::Pixel || !float || bits != 32 {
                return unsupported("a derivative outside a pixel shader or not of 32-bit floats");
            }
            // of the lane's quad: coarse ones from its first row and column, fine ones
            // from the lane's own; every lane reads before any result is written
            let mut out = [0u32; 128];
            for &lane in lanes {
                let l = lane as usize;
                let quad = l & !3;
                let (from, to) = match op {
                    DERIV_COARSE_X => (quad, quad + 1),
                    DERIV_COARSE_Y => (quad, quad + 2),
                    DERIV_FINE_X => (quad + (l & 2), quad + (l & 2) + 1),
                    _ => (quad + (l & 1), quad + (l & 1) + 2),
                };
                if cx.active >> from & cx.active >> to & 1 == 0 {
                    cx.ub |= UB_INACTIVE_LANE;
                }
                let (x, y) = (f32::from_bits(arg!(0, from) as u32), f32::from_bits(arg!(0, to) as u32));
                out[l] = nan2_32(y - x, y, x).to_bits();
            }
            for &lane in lanes {
                regs[d + lane as usize] = out[lane as usize] as u64;
            }
        }
        COVERAGE => each!(|l| regs[d + l] = 1),
        INNER_COVERAGE | SAMPLE_INDEX | VIEW_ID => each!(|l| regs[d + l] = 0),
        PRIMITIVE_ID => each!(|l| regs[d + l] = cx.world.input(cx.p.stage == Stage::Pixel, res::SEM_PRIMITIVE_ID, true, u32::MAX, 0, 0, cx.id[0] * w as u32 + l as u32, false) as u64),
        HANDLE_FROM_BINDING => {
            let bind = p.binds[ex[0] as usize];
            let index = ex[1] as usize * w;
            each!(|l| {
                let i = regs[index + l] as u32;
                regs[d + l] = if i < bind.lower || i > bind.upper || i >= 1 << 20 { res::IDENT_NULL } else { res::IDENT_BOUND | bind.class << 28 | bind.space << 20 | i } as u64;
            });
        }
        HANDLE_FROM_HEAP => each!(|l| {
            let i = arg!(0, l) as u32;
            let tag = if arg!(1, l) & 1 != 0 { res::IDENT_SAMPLER_HEAP } else { res::IDENT_HEAP };
            regs[d + l] = if i >= res::HEAP_SIZE { res::IDENT_NULL } else { tag | i } as u64;
        }),
        ANNOTATE_HANDLE => {
            let h = ex[0] as usize * w;
            let props = (ex[1] as u64 + 1) << 32;
            each!(|l| regs[d + l] = regs[h + l] & 0xffff_ffff | props);
        }

        // wave operations: the lanes of this block are the active lanes
        WAVE_IS_FIRST_LANE => each!(|l| regs[d + l] = (l == lanes[0] as usize) as u64),
        WAVE_GET_LANE_INDEX => each!(|l| regs[d + l] = l as u64),
        WAVE_GET_LANE_COUNT => each!(|l| regs[d + l] = w as u64),
        WAVE_ANY_TRUE | WAVE_ALL_TRUE | WAVE_ALL_BIT_COUNT => {
            let count = lanes.iter().filter(|&&l| arg!(0, l as usize) & 1 != 0).count();
            let v = match op {
                WAVE_ANY_TRUE => (count != 0) as u64,
                WAVE_ALL_TRUE => (count == lanes.len()) as u64,
                _ => count as u64,
            };
            each!(|l| regs[d + l] = v);
        }
        WAVE_PREFIX_BIT_COUNT => {
            let mut count = 0u64;
            each!(|l| {
                let bit = arg!(0, l) & 1;
                regs[d + l] = count;
                count += bit;
            });
        }
        WAVE_ALL_EQUAL => {
            let first = arg!(0, lanes[0] as usize);
            let equal = lanes.iter().all(|&l| {
                let x = arg!(0, l as usize);
                if float {
                    widen(x, bits) == widen(first, bits)
                } else {
                    x == first
                }
            });
            each!(|l| regs[d + l] = equal as u64);
        }
        WAVE_BALLOT => {
            let mut ballot = 0u128;
            each!(|l| ballot |= ((arg!(0, l) & 1) as u128) << l);
            *cx.ballots += 1;
            if ballot == 0 && cx.world.force_ballots {
                // a function of the seed and of how many ballots the wave has taken: the
                // same in two compilations of one program, which take the same ballots
                let h = res::mix(cx.world.seed ^ (cx.wave as u64) << 40 ^ *cx.ballots as u64);
                if h & 3 != 0 {
                    let pick = if h & 4 != 0 { u128::MAX } else { (res::mix(h) as u128) << 64 | res::mix(h ^ 1) as u128 };
                    ballot = cx.active & pick;
                    if ballot == 0 {
                        ballot = 1 << lanes[0];
                    }
                }
            }
            each!(|l| {
                for k in 0..4 {
                    regs[d + k * w + l] = (ballot >> (32 * k)) as u32 as u64;
                }
            });
        }
        WAVE_MATCH => {
            let values: Vec<u64> = lanes.iter().map(|&l| arg!(0, l as usize)).collect();
            each!(|l| {
                let mine = arg!(0, l);
                let mut same = 0u128;
                for (i, &other) in lanes.iter().enumerate() {
                    let eq = if float { widen(values[i], bits) == widen(mine, bits) } else { values[i] == mine };
                    same |= (eq as u128) << other;
                }
                for k in 0..4 {
                    regs[d + k * w + l] = (same >> (32 * k)) as u32 as u64;
                }
            });
        }
        WAVE_READ_LANE_FIRST => {
            let v = arg!(0, lanes[0] as usize);
            each!(|l| regs[d + l] = v);
        }
        WAVE_READ_LANE_AT | QUAD_READ_LANE_AT => {
            // every lane reads before any result is written (the result row may be the
            // operand's when the decoder reuses scratch rows; it never is today)
            let values: Vec<u64> = lanes
                .iter()
                .map(|&l| {
                    let l = l as usize;
                    let from = if op == QUAD_READ_LANE_AT { (l & !3) as u64 + (arg!(1, l) & 3) } else { arg!(1, l) & 0xffff_ffff };
                    if from < 128 && cx.active >> from & 1 != 0 {
                        arg!(0, from as usize)
                    } else {
                        cx.ub |= UB_INACTIVE_LANE;
                        0
                    }
                })
                .collect();
            for (i, &l) in lanes.iter().enumerate() {
                regs[d + l as usize] = values[i];
            }
        }
        WAVE_ACTIVE_BIT => {
            let kind = arg!(1, lanes[0] as usize);
            let mut acc = if kind == 0 { m } else { 0 };
            each!(|l| {
                let x = arg!(0, l) & m;
                acc = match kind {
                    0 => acc & x,
                    1 => acc | x,
                    _ => acc ^ x,
                };
            });
            each!(|l| regs[d + l] = acc);
        }
        WAVE_ACTIVE_OP | WAVE_PREFIX_OP => {
            // operation: 0 sum, 1 product, 2 min, 3 max; sign: 0 signed (and floats), 1 unsigned
            let kind = arg!(1, lanes[0] as usize);
            let unsigned = arg!(2, lanes[0] as usize) & 1 != 0;
            if kind > 3 || (op == WAVE_PREFIX_OP && kind > 1) {
                return unsupported("wave operation kind");
            }
            let combine = |acc: u64, x: u64| -> u64 {
                if float {
                    let (fa, fx) = (widen(acc, bits), widen(x, bits));
                    match kind {
                        0 => narrow(nan2_64(fa + fx, fa, fx), bits),
                        1 => narrow(nan2_64(fa * fx, fa, fx), bits),
                        2 => {
                            if fmin(fa, fx, fa.is_nan(), fx.is_nan()).to_bits() == fa.to_bits() {
                                acc
                            } else {
                                x
                            }
                        }
                        _ => {
                            if fmax(fa, fx, fa.is_nan(), fx.is_nan()).to_bits() == fa.to_bits() {
                                acc
                            } else {
                                x
                            }
                        }
                    }
                } else {
                    match (kind, unsigned) {
                        (0, _) => acc.wrapping_add(x) & m,
                        (1, _) => acc.wrapping_mul(x) & m,
                        (2, true) => acc.min(x),
                        (3, true) => acc.max(x),
                        (2, false) => {
                            if sext(x, bits) < sext(acc, bits) {
                                x
                            } else {
                                acc
                            }
                        }
                        _ => {
                            if sext(x, bits) > sext(acc, bits) {
                                x
                            } else {
                                acc
                            }
                        }
                    }
                }
            };
            if op == WAVE_ACTIVE_OP {
                let mut acc = arg!(0, lanes[0] as usize) & m;
                for &l in &lanes[1..] {
                    acc = combine(acc, arg!(0, l as usize) & m);
                }
                each!(|l| regs[d + l] = acc);
            } else {
                // the lanes before this one; the identity for the first
                let mut acc = match (kind, float) {
                    (0, _) => 0,
                    (_, false) => 1,
                    (_, true) => narrow(1.0, bits),
                };
                each!(|l| {
                    let x = arg!(0, l) & m;
                    regs[d + l] = acc;
                    acc = combine(acc, x);
                });
            }
        }
        _ => return unsupported("not implemented"),
    }
    let _ = math::NAN32;
    Ok(())
}
