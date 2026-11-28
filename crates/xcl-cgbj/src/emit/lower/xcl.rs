// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../../LICENSE and ../../../PROVENANCE.md), so it carries that licence.
//! The XCL binding: memory instructions on register values, the way xsemu's own translator
//! lowers them.
//!
//! CGBJ traces each descriptor to a root parameter and turns it into a host heap slot. Under
//! this binding nothing is traced. An entry SGPR is a dword of io (root UAV u0, space 0). A
//! scalar load reads title memory at the address its registers hold. A buffer, image or
//! sampler descriptor is the register values the instruction names. Every byte of title
//! memory is reached through one resolver ([`Lower::x_resolve`]) that follows the emulator's
//! windows and page directory.
//!
//! The rules are those of xsemu's translator (its memory access notes, sections 1 to 7). The
//! hooks in lower.rs are marked `xsemu:`.

use std::collections::HashMap;

use super::super::bitcode::{Attr, Bin, Cast, Term, Ty, V, pred};
use super::super::shader::{Shader, flags};
use super::{Lower, Parts, Val, Why};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::{Site, Source};
use crate::xcl::{d16_half_bits, XclConfig, XclStats};

#[path = "xcl_formats.rs"]
mod formats;

/// What stands for a descriptor's source where CGBJ's code wants one: under the binding a
/// descriptor is the values of its registers.
pub(super) static BY_VALUE: Source = Source::Constant(Vec::new());

/// Memory windows in the view heap. The null view follows the last one.
const WINDOWS: u32 = 4;

/// io dwords the entry code reads.
const IO_UMA_BASE: u32 = 112;
const IO_PHYSICAL_TABLE: u32 = 118;

/// The words the entry code loads for memory accesses.
#[derive(Clone, Copy)]
struct Memory {
    /// Heap index of window 0's raw view.
    views: V,
    /// High dword of window 0's addresses.
    uma_base: V,
    /// The root's handle, made once in the entry block.
    root: V,
    /// Heap index of the null view.
    null: V,
}

/// A base that scalar loads read at constant offsets, and which 64 KiB of offsets: what
/// makes two loads members of one page group. `T` is a value of the program, or the
/// module's value for it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Base<T> {
    lo: T,
    hi: T,
    soffset: Option<T>,
    /// A buffer load's lane offset (`buffer_load_dword*`), the VGPR the instruction names.
    voff: Option<T>,
    /// A V#'s base (`s_buffer_load_*`, `buffer_load_*`): the high dword is the second
    /// word's low 16 bits.
    buffer: bool,
    /// The constant offset from bit 16 up.
    upper: i32,
}

/// The two 64 KiB pages the loads of one [`Base`] lie in, resolved once
/// ([`Lower::x_group`]).
#[derive(Clone, Copy)]
struct Group {
    /// The heap index of each page's view.
    views: [V; 2],
    /// Per page: the byte in its view of the member whose offset's low 16 bits are 0, were
    /// it in that page. A member's byte is this plus those bits.
    bytes: [V; 2],
    /// The low 16 bits of the base (with the offset above bit 16), on a dword.
    low: V,
    /// The handles of the two pages' views.
    handles: [V; 2],
    /// The byte of the second page's first dword in its view.
    next: V,
    /// The second page's first three dwords, read once where the group is resolved when the
    /// program has no access past the caches (scalar loads may give an older value there).
    /// Otherwise each member reads them itself.
    over: Option<[V; 3]>,
}

/// The page groups of a program.
#[derive(Default)]
struct Groups {
    /// The bases more than one load reads, in the order of their second load.
    wanted: Vec<Base<Value>>,
    /// Per value a wanted base is made of: where the base is in `wanted`.
    by_value: HashMap<Value, Vec<usize>>,
    made: HashMap<Base<V>, Group>,
}

/// Whether a scalar load reads a V#'s buffer, and its dwords.
fn smem_shape(name: &str) -> Option<(bool, u32)> {
    let (buffer, rest) = match (name.strip_prefix("s_load_dword"), name.strip_prefix("s_buffer_load_dword")) {
        (Some(rest), _) => (false, rest),
        (_, Some(rest)) => (true, rest),
        _ => return None,
    };
    let count = match rest {
        "" => 1,
        "x2" => 2,
        "x4" => 4,
        "x8" => 8,
        "x16" => 16,
        _ => return None,
    };
    Some((buffer, count))
}

/// The base of a scalar load that can be a member of a page group, and the low 16 bits of
/// its offset. A member's dwords all lie in the base's page or the next: its offset is on a
/// dword and its last dword starts below 64 KiB.
fn smem_base(inst: &Inst) -> Option<(Base<Value>, u32)> {
    let Fields::Smem {
        base: Operand::Scalar { code, .. },
        offset,
        soffset,
        ..
    } = &inst.fields
    else {
        return None;
    };
    let (buffer, count) = smem_shape(inst.name())?;
    let soffset = match soffset {
        Operand::Scalar { code: NULL, .. } => None,
        Operand::Scalar { code, .. } => Some(inst.read(Reg::S(*code))?),
        _ => return None,
    };
    // (the offset of a buffer load has no sign)
    let offset = if buffer { ((*offset as u32) & 0x1f_ffff) as i32 } else { *offset };
    let low = (offset & 0xffff) as u32;
    if low % 4 != 0 || low + 4 * (count - 1) > 0xfffc {
        return None;
    }
    let base = Base {
        lo: inst.read(Reg::S(*code))?,
        hi: inst.read(Reg::S(code + 1))?,
        soffset,
        voff: None,
        buffer,
        upper: offset >> 16,
    };
    Some((base, low))
}

/// The base of a `buffer_load_dword*` that can be a member of a page group, the low 16 bits
/// of its offset, and whether it reads several dwords: the descriptor's base is a value, the
/// lane's offset another, and the instruction's a constant on a dword whose last dword
/// starts below 64 KiB.
fn buffer_base(inst: &Inst) -> Option<(Base<Value>, u32, bool)> {
    let Fields::Buffer {
        addr,
        rsrc,
        soffset,
        offset,
        offen,
        idxen,
        lds,
        tfe,
        ..
    } = &inst.fields
    else {
        return None;
    };
    if *lds || *tfe || *idxen || !*offen {
        return None;
    }
    let count = match inst.name() {
        "buffer_load_dword" => 1,
        "buffer_load_dwordx2" => 2,
        "buffer_load_dwordx3" => 3,
        "buffer_load_dwordx4" => 4,
        _ => return None,
    };
    let Operand::Scalar { code: rs, .. } = *rsrc else {
        return None;
    };
    let Operand::Vector { reg: areg, .. } = *addr else {
        return None;
    };
    let off = u32::from(*offset);
    if off % 4 != 0 || off + 4 * (count - 1) > 0xfffc {
        return None;
    }
    let soffset = match soffset {
        Operand::Scalar { code: NULL, .. } | Operand::Const(128) => None,
        Operand::Scalar { code, .. } => Some(inst.read(Reg::S(*code))?),
        _ => return None,
    };
    let base = Base {
        lo: inst.read(Reg::S(rs))?,
        hi: inst.read(Reg::S(rs + 1))?,
        soffset,
        voff: Some(inst.read(Reg::V(areg))?),
        buffer: true,
        upper: 0,
    };
    Some((base, off, count > 1))
}

impl Groups {
    /// The bases of `p` that are worth a group: those more than one load reads. A base
    /// read once is resolved by its load, which walks one page where a group walks two.
    fn find(p: &Program) -> Groups {
        let mut loads: HashMap<Base<Value>, u32> = HashMap::new();
        let mut g = Groups::default();
        for inst in p.blocks.iter().flat_map(|b| &b.insts) {
            // (a buffer load of one dword is served by its own resolver, which walks one
            // page where a group walks two)
            let base = match smem_base(inst) {
                Some((base, _)) => base,
                None => match buffer_base(inst) {
                    Some((base, _, true)) => base,
                    _ => continue,
                },
            };
            let n = loads.entry(base).or_insert(0);
            *n += 1;
            if *n == 2 {
                let terms = [Some(base.lo), Some(base.hi), base.soffset, base.voff];
                for v in terms.into_iter().flatten() {
                    let at = g.by_value.entry(v).or_default();
                    if at.last() != Some(&g.wanted.len()) {
                        at.push(g.wanted.len());
                    }
                }
                g.wanted.push(base);
            }
        }
        g
    }
}

pub(super) struct State {
    cfg: XclConfig,
    /// The program has an instruction that reaches memory.
    memory: bool,
    /// Data views are globally coherent (decided once per program).
    coherent: bool,
    /// `dx.types.ResRet.i32`.
    ret: Ty,
    io: Option<V>,
    words: Option<Memory>,
    /// Sampler words known to be constants, by value.
    consts: HashMap<Value, Option<u32>>,
    /// Where each join of the program is: its block and its place there.
    phis: Option<HashMap<Value, (usize, usize)>>,
    /// What each s_brev_b32 of a constant gives, by value.
    reversed: HashMap<Value, u32>,
    /// Whether this lane is inside the thread limits, when they are honoured.
    launch: Option<V>,
    /// Raw dwords for the next [`Lower::format_load`].
    format_words: Option<Vec<V>>,
    /// The tables of the formats a V# names, once a load or a store needs them.
    formats: Option<formats::Tables>,
    groups: Groups,
    stats: XclStats,
}

impl State {
    /// The binding's state, when the translation on this thread is under it. Sets the places
    /// of io and of the static samplers in `s`.
    pub(super) fn new(x: &Parts, s: &mut Shader) -> Option<Box<State>> {
        let cfg = crate::xcl::binding()?;
        s.view_space = 0;
        s.sampler_space = 1;
        s.sampler_base = cfg.sampler_base;
        let insts = || x.p.blocks.iter().flat_map(|b| &b.insts);
        let memory = insts().any(|i| matches!(i.fields, Fields::Smem { .. } | Fields::Buffer { .. } | Fields::Flat { .. }));
        // The simple form of the coherence rule: any access past the caches, any cache
        // invalidate or any memory atomic makes every data view coherent.
        let coherent = insts().any(|i| {
            let n = i.name();
            match &i.fields {
                Fields::Smem { cache, .. } => cache.glc || cache.dlc || cache.slc || matches!(n, "s_dcache_inv" | "s_gl1_inv"),
                Fields::Buffer { cache, .. } => cache.glc || cache.dlc || cache.slc || n.starts_with("buffer_atomic_") || matches!(n, "buffer_gl0_inv" | "buffer_gl1_inv"),
                Fields::Flat { cache, .. } => !n.starts_with("scratch_") && (cache.glc || cache.dlc || cache.slc || n.contains("_atomic_")),
                _ => false,
            }
        });
        let i32t = s.t.i32;
        let ret = s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
        Some(Box::new(State {
            cfg,
            memory,
            coherent,
            ret,
            io: None,
            words: None,
            launch: None,
            consts: HashMap::new(),
            phis: None,
            reversed: HashMap::new(),
            format_words: None,
            formats: None,
            groups: if memory { Groups::find(x.p) } else { Groups::default() },
            stats: XclStats::default(),
        }))
    }
}

/// Where an access's base comes from, which decides the addresses it may reach.
#[derive(Clone, Copy)]
enum Origin {
    /// A load through a base whose high dword is this. The flag: known to be at most 0xffff.
    Read(V, bool),
    /// A load at a 64-bit address each lane holds, whose high dword before the offset is this.
    Lane(V),
    /// A store or an atomic through a base whose high dword is this.
    Write(V),
}

/// The bound test of a buffer access.
struct Bounds {
    /// The stride is 0: the offset is tested, else the index.
    stride0: V,
    /// The VGPR offset, if any, and the instruction's.
    voff: Option<V>,
    offset: u32,
    records: V,
    by_index: V,
}

impl<'a> Lower<'a> {
    fn xi(&mut self, v: u32) -> V {
        self.s.m.i32(v)
    }

    fn xb(&mut self, op: Bin, a: V, b: V) -> V {
        self.s.m.bin(self.cur, op, a, b)
    }

    fn xc(&mut self, p: u8, a: V, b: V) -> V {
        self.s.m.cmp(self.cur, p, a, b)
    }

    fn xs(&mut self, c: V, t: V, f: V) -> V {
        self.s.m.select(self.cur, c, t, f)
    }

    fn x_and(&mut self, a: Option<V>, b: Option<V>) -> Option<V> {
        match (a, b) {
            (Some(a), Some(b)) => Some(self.xb(Bin::And, a, b)),
            (a, b) => a.or(b),
        }
    }

    fn x_state(&mut self) -> &mut State {
        self.xcl.as_mut().expect("the XCL binding")
    }

    fn x_words(&self) -> Memory {
        self.xcl.as_ref().and_then(|x| x.words).expect("the memory words")
    }

    /// `n` dwords (at most 4) of the raw view `h` from `byte` on, read in `block`.
    fn x_raw(&mut self, block: usize, h: V, byte: V, n: u32) -> Vec<V> {
        let ret = self.x_state().ret;
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let mask = self.s.m.i8((1u8 << n) - 1);
        let r = self.s.op(block, "rawBufferLoad.i32", Attr::ReadOnly, ret, 139, &[h, byte, u, mask, align]);
        (0..n).map(|k| self.s.m.extract(block, r, k)).collect()
    }

    /// One dword stored to the raw view `h` at `byte`.
    fn x_raw_store(&mut self, h: V, byte: V, value: V) {
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let (mask, void) = (self.s.m.i8(1), self.s.t.void);
        self.s.op(self.cur, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &[h, byte, u, value, u, u, u, mask, align]);
    }

    /// A handle from a descriptor heap at `index`, as it stands: the host's indices need no
    /// clamp.
    fn x_heap(&mut self, index: V, sampler: bool, nonuniform: bool, props: (u32, u32)) -> V {
        self.s.flags |= if sampler { flags::SAMPLER_HEAP } else { flags::RESOURCE_HEAP };
        self.s.nonuniform_handles += u32::from(nonuniform);
        let (s, n) = (self.s.m.bool(sampler), self.s.m.bool(nonuniform));
        let handle = self.s.t.handle;
        let h = self.s.op(self.cur, "createHandleFromHeap", Attr::ReadNone, handle, 218, &[index, s, n]);
        self.s.annotate(self.cur, h, props.0, props.1)
    }

    /// The handle of a raw view of title memory.
    fn x_data(&mut self, view: V) -> V {
        let coherent = if self.x_state().coherent { 0x4000 } else { 0 };
        self.x_heap(view, false, true, (11 | 0x1000 | coherent, 0))
    }

    /// The entry code: the io handle, the thread limits and the memory words. Leaves the
    /// shader's entry block as the block the program starts in.
    pub(super) fn x_prologue(&mut self) {
        let Some(x) = self.xcl.as_ref() else { return };
        let (cfg, memory) = (x.cfg, x.memory);
        let first = self.s.entry;
        let io = self.s.root_uav_handle(0);
        self.x_state().io = Some(io);
        // (only a compute program has thread limits)
        let limits = cfg.thread_limits && self.s.threads.is_some();
        if !limits && !memory {
            return;
        }
        let zero = self.s.m.i32(0);
        // The locator and the three thread limits are consecutive dwords.
        let at = self.s.m.i32(4 * IO_PHYSICAL_TABLE);
        let w = self.x_raw(first, io, at, 4);
        if limits {
            // A thread at or past a limit that is not 0 runs with its EXEC bit clear, as on
            // the console: it stores nothing, but a program that sets EXEC wider can read its
            // registers. The launch mask of the lane structure is this test (`crate::structure::Atom::Launch`).
            let runs = self.x_limits(first, &w);
            self.x_state().launch = Some(runs);
            // The packed form's high half has console threads of its own (lower/packed.rs).
            if self.packed.is_some() {
                self.p_swap();
                let runs = self.x_limits(first, &w);
                self.x_state().launch = Some(runs);
                self.p_swap();
            }
        }
        if memory {
            let head = self.s.entry;
            let at = self.s.m.i32(4 * IO_UMA_BASE);
            let v = self.x_raw(head, io, at, 2);
            let (uma_base, views) = (v[0], v[1]);
            let k = self.s.m.i32(WINDOWS);
            let null = self.s.m.bin(head, Bin::Add, views, k);
            // The directory's root index is the first dword of the view the locator names.
            // Without a locator, and when the locator names no root, the index is the null
            // view's, whose dwords are 0. Both reads are then valid, so the root is made
            // here once and every access walks the directory: the null root answers "no
            // entry" for every page, which is the window rule's own answer.
            let located = self.s.m.cmp(head, pred::NE, w[0], zero);
            let index = self.s.m.select(head, located, w[0], null);
            let locator = self.x_heap(index, false, false, (11, 0));
            let found = self.x_raw(head, locator, zero, 1)[0];
            let named = self.s.m.cmp(head, pred::NE, found, zero);
            let index = self.s.m.select(head, named, found, null);
            let root = self.x_heap(index, false, false, (11, 0));
            self.x_state().words = Some(Memory { views, uma_base, root, null });
        }
        self.cur = self.s.entry;
    }

    /// Whether this thread is inside the thread limits `w[1..4]`, made in the entry block `first`. In
    /// the packed form, for the half being lowered and only where its console lane exists.
    fn x_limits(&mut self, first: usize, w: &[V]) -> V {
        let threads = self.s.threads.expect("a compute program");
        let one = self.s.m.i32(1);
        let mut runs = self.p_valid();
        for c in 0..3u8 {
            let (Val::I32(group), Val::I32(thread)) = (self.compute_id("groupId.i32", 94, c), self.thread_id(c)) else {
                unreachable!()
            };
            let width = threads[usize::from(c)];
            let start = if width == 1 {
                group
            } else {
                let width = self.s.m.i32(width);
                self.s.m.bin(first, Bin::Mul, group, width)
            };
            let at = self.s.m.bin(first, Bin::Add, start, thread);
            // (a limit of 0 becomes the largest number)
            let last = self.s.m.bin(first, Bin::Sub, w[1 + usize::from(c)], one);
            let inside = self.s.m.cmp(first, pred::IULE, at, last);
            runs = Some(match runs {
                Some(r) => self.s.m.bin(first, Bin::And, r, inside),
                None => inside,
            });
        }
        runs.expect("three axes")
    }

    /// The launch mask's place, for the packed form to give each half its own.
    pub(super) fn x_launch_slot(&mut self) -> Option<&mut Option<V>> {
        Some(&mut self.xcl.as_mut()?.launch)
    }

    /// Whether this lane is inside the thread limits ([`crate::structure::Atom::Launch`]).
    pub(super) fn x_launch(&mut self) -> Result<V, Why> {
        self.xcl.as_ref().and_then(|x| x.launch).ok_or("a launch mask without the thread limits")
    }

    /// The value of an input register under the binding, where it differs from CGBJ's: an
    /// SGPR the hardware does not set itself (`system`) is its io dword.
    pub(super) fn x_input(&mut self, reg: Reg, system: bool) -> Option<Val> {
        let io = self.xcl.as_ref()?.io?;
        let Reg::S(n) = reg else { return None };
        // s0 to s105 and M0. Masks keep CGBJ's handling.
        if system || !(n <= 105 || n == 124) {
            return None;
        }
        // (also past the SGPRs the console sets, where CGBJ takes 0: the host leaves io 0 there)
        let (entry, at) = (self.s.entry, self.s.m.i32(4 * u32::from(n)));
        Some(Val::I32(self.x_raw(entry, io, at, 1)[0]))
    }

    /// Why an instruction the binding does not cover is refused.
    pub(super) fn x_refused(&self, inst: &Inst) -> Option<Why> {
        self.xcl.as_ref()?;
        inst.name().starts_with("image_bvh").then_some("ray tests under the XCL binding")
    }

    /// The counts of what was emitted.
    pub(super) fn x_stats(&mut self) -> Option<XclStats> {
        let samplers = self.samplers.len() as u32;
        let x = self.xcl.as_mut()?;
        x.stats.static_samplers = samplers;
        Some(x.stats)
    }

    /// The raw dwords the caller of [`Lower::format_load`] read for it.
    pub(super) fn x_format_words(&mut self) -> Option<Vec<V>> {
        self.xcl.as_mut()?.format_words.take()
    }

    // ---- addresses ----

    /// `lo` on a dword.
    fn x_aligned(&mut self, lo: V) -> V {
        let m = self.xi(!3);
        self.xb(Bin::And, lo, m)
    }

    /// The 64-bit address `lo`, `hi` plus `terms` (each an unsigned dword), `k` (signed) and the
    /// 64-bit product of two dwords.
    fn x_address(&mut self, lo: V, hi: V, terms: &[V], k: i32, product: Option<(V, V)>) -> (V, V) {
        if terms.is_empty() && product.is_none() {
            return self.address_plus(lo, hi, k);
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let mut sum: Option<V> = None;
        if let Some((a, b)) = product {
            let (a, b) = (self.s.m.cast(c, Cast::ZExt, a, i64t), self.s.m.cast(c, Cast::ZExt, b, i64t));
            sum = Some(self.xb(Bin::Mul, a, b));
        }
        for &t in terms {
            let t = self.s.m.cast(c, Cast::ZExt, t, i64t);
            sum = Some(match sum {
                Some(s) => self.xb(Bin::Add, s, t),
                None => t,
            });
        }
        let mut sum = sum.expect("a term");
        if k != 0 {
            let k = self.s.m.int_const(i64t, i64::from(k));
            sum = self.xb(Bin::Add, sum, k);
        }
        let base = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let total = self.xb(Bin::Add, base, sum);
        let (lo, carry) = self.split64(total);
        (lo, self.xb(Bin::Add, hi, carry))
    }

    // ---- the resolver ----

    /// Whether an access at high dword `hi` with this origin reaches nothing.
    fn x_bad(&mut self, hi: V, origin: Origin, kill: Option<V>) -> V {
        let (top, zero) = (self.xi(0xffff), self.xi(0));
        let bad = match origin {
            Origin::Read(o, small) => {
                let mut bad = self.xc(pred::IUGT, hi, top);
                if o != hi {
                    if !small {
                        let over = self.xc(pred::IUGT, o, top);
                        bad = self.xb(Bin::Or, bad, over);
                    }
                    // A 32-bit pointer that carried, or a 64-bit one that fell to 0.
                    let (oz, hz) = (self.xc(pred::EQ, o, zero), self.xc(pred::EQ, hi, zero));
                    let moved = self.xb(Bin::Xor, oz, hz);
                    bad = self.xb(Bin::Or, bad, moved);
                }
                bad
            }
            Origin::Lane(v) => {
                let mut bad = self.xc(pred::IUGT, hi, top);
                if v != hi {
                    let over = self.xc(pred::IUGT, v, top);
                    bad = self.xb(Bin::Or, bad, over);
                    let (vz, hn) = (self.xc(pred::EQ, v, zero), self.xc(pred::NE, hi, zero));
                    let carried = self.xb(Bin::And, vz, hn);
                    bad = self.xb(Bin::Or, bad, carried);
                }
                bad
            }
            Origin::Write(o) => {
                // Neither 0 nor past 0xffff, for the base and for the address.
                let (one, most) = (self.xi(1), self.xi(0xfffe));
                let t = self.xb(Bin::Sub, o, one);
                let mut bad = self.xc(pred::IUGT, t, most);
                if o != hi {
                    let t = self.xb(Bin::Sub, hi, one);
                    let out = self.xc(pred::IUGT, t, most);
                    bad = self.xb(Bin::Or, bad, out);
                }
                bad
            }
        };
        match kill {
            Some(k) => self.xb(Bin::Or, bad, k),
            None => bad,
        }
    }

    /// The page directory's entry for the 64 KiB page of address `lo`, `hi`, 0 when it has
    /// none. A `hi` past 0xffff (an access the caller drops) reads some root entry or, past
    /// the root's 512 KiB, 0: the leaf indices it gives are still the directory's own.
    /// `root` is its handle, made once in the entry block, and `null` the heap index of the
    /// null view, whose dwords are 0. The only place that knows the directory's layout: a
    /// root of two leaf indices per high dword, and leaves of one dword per page. The second
    /// leaf counts when the first has no entry. Both are read, in one block: a loop or a
    /// branch per access costs the driver's compiler more than the read. A leaf index of 0
    /// reads the null view instead, which answers 0, the directory's "no entry".
    fn x_directory(&mut self, lo: V, hi: V, null: V, root: V) -> V {
        let zero = self.xi(0);
        let k3 = self.xi(3);
        let at = self.xb(Bin::Shl, hi, k3);
        let leaves = self.x_raw(self.cur, root, at, 2);
        let (k16, k2) = (self.xi(16), self.xi(2));
        let page = self.xb(Bin::LShr, lo, k16);
        let at = self.xb(Bin::Shl, page, k2);
        let entries: Vec<V> = leaves
            .into_iter()
            .map(|leaf| {
                let none = self.xc(pred::EQ, leaf, zero);
                let index = self.xs(none, null, leaf);
                let h = self.x_heap(index, false, true, (11, 0));
                self.x_raw(self.cur, h, at, 1)[0]
            })
            .collect();
        let found = self.xc(pred::NE, entries[0], zero);
        self.xs(found, entries[0], entries[1])
    }

    /// The raw view (a heap index) and the byte in it of the dword at 64-bit address `lo`,
    /// `hi`, for an access of this origin. `kill`: the access reaches nothing. Pure: any
    /// lane may compute it. Goes on in another block.
    fn x_resolve(&mut self, lo: V, hi: V, origin: Origin, kill: Option<V>, multi: bool) -> (V, V) {
        let m = self.x_words();
        let stats = &mut self.x_state().stats;
        if multi {
            stats.resolves_multi += 1;
        } else {
            stats.resolves_single += 1;
        }
        let zero = self.xi(0);
        let bad = self.x_bad(hi, origin, kill);
        // A 32-bit pointer lies in window 0.
        let hi = match origin {
            Origin::Write(_) => hi,
            _ => {
                let short = self.xc(pred::EQ, hi, zero);
                self.xs(short, m.uma_base, hi)
            }
        };
        let w = self.xb(Bin::Sub, hi, m.uma_base);
        let k = self.xi(WINDOWS);
        let inside = self.xc(pred::IULT, w, k);
        let window = self.xb(Bin::Add, m.views, w);
        // The directory, when there is one: the null root answers "no entry" when there is not.
        let entry = self.x_directory(lo, hi, m.null, m.root);
        // An entry: 0 absent (the window rule), else bit 31 set, bits 30:27 zero, the page's
        // place in the view in bits 26:20 and the view's heap index in bits 19:0. An index of
        // 0 is unmapped: the null view, as anything else is.
        let unmapped = self.xi(0x8000_0000);
        let entry = self.xs(bad, unmapped, entry);
        // (mapped: the entry without its place is from 0x80000001 to 0x800fffff)
        let (index_bits, rest, least) = (self.xi(0xf_ffff), self.xi(0xf80f_ffff), self.xi(0x8000_0001));
        let index = self.xb(Bin::And, entry, index_bits);
        let named = self.xb(Bin::And, entry, rest);
        let named = self.xb(Bin::Sub, named, least);
        let mapped = self.xc(pred::IULT, named, index_bits);
        let absent = self.xc(pred::EQ, entry, zero);
        let direct = self.xb(Bin::And, absent, inside);
        let view = self.xs(direct, window, m.null);
        let view = self.xs(mapped, index, view);
        let (k4, slots, low) = (self.xi(4), self.xi(0x7f_0000), self.xi(0xffff));
        let slot = self.xb(Bin::LShr, entry, k4);
        let slot = self.xb(Bin::And, slot, slots);
        let within = self.xb(Bin::And, lo, low);
        let at = self.xb(Bin::Or, slot, within);
        // The byte does not matter in the null view: it reads 0 everywhere. (xsemu's
        // translator reads it at byte 0, or at the address's low 16 bits, by the kind of
        // load. A test host that keeps data in the null view sees that difference.)
        let byte = self.xs(mapped, at, lo);
        (view, byte)
    }

    /// For an access of `n` dwords from `lo`, `hi`: whether it crosses a 64 KiB page, its
    /// bytes in the first page, and the address of the next page.
    fn x_pages(&mut self, lo: V, hi: V, n: u32) -> (V, V, V, V) {
        let (low, room, page, zero) = (self.xi(0xffff), self.xi(0x1_0000 - 4 * n), self.xi(0x1_0000), self.xi(0));
        let within = self.xb(Bin::And, lo, low);
        let crosses = self.xc(pred::IUGT, within, room);
        let first = self.xb(Bin::Sub, page, within);
        let lo2 = self.xb(Bin::Add, lo, first);
        let wrapped = self.xc(pred::EQ, lo2, zero);
        let carry = self.s.m.cast(self.cur, Cast::ZExt, wrapped, self.s.t.i32);
        let hi2 = self.xb(Bin::Add, hi, carry);
        (crosses, first, lo2, hi2)
    }

    /// `n` dwords of title memory from the address `lo`, `hi` (on a dword) on. The page of
    /// the first dword and the page after it are both resolved, in one block: an access of
    /// `n` dwords lies in those two at most, and dword j is in the second when its offset
    /// reaches the first page's end. The dwords are read four at a time, from the page of the
    /// first of the four, and those of a four that lie past the page's end from one more read
    /// of the next page's dwords there.
    fn x_load(&mut self, lo: V, hi: V, origin: Origin, kill: Option<V>, n: u32) -> Vec<V> {
        if n == 1 {
            let (view, byte) = self.x_resolve(lo, hi, origin, kill, false);
            let h = self.x_data(view);
            return self.x_raw(self.cur, h, byte, 1);
        }
        let (_, first, lo2, hi2) = self.x_pages(lo, hi, n);
        let (view, byte) = self.x_resolve(lo, hi, origin, kill, true);
        let (next, next_byte) = self.x_resolve(lo2, hi2, origin, kill, true);
        let (h, h2) = (self.x_data(view), self.x_data(next));
        // The next page's first byte, less the first page's bytes from it: dword j of the
        // second page is at this plus 4j.
        let at2 = self.xb(Bin::Sub, next_byte, first);
        let mut out = Vec::with_capacity(n as usize);
        for c in (0..n).step_by(4) {
            let m = (n - c).min(4);
            let off = self.xi(4 * c);
            let at = self.xb(Bin::Add, byte, off);
            let read = self.x_raw(self.cur, h, at, m);
            // Dword 0 is in the first page (the crossing needs an offset of 4 at least).
            // Later dwords of the four read the next page's dword of the same offset: from
            // its byte `at2` on, one `c` further in for a four that does not start at 0.
            let over: Vec<V> = match (c, m) {
                (0, 1) => Vec::new(),
                (0, _) => {
                    let four = self.xi(4);
                    let at = self.xb(Bin::Add, at2, four);
                    self.x_raw(self.cur, h2, at, m - 1)
                }
                (_, _) => {
                    let at = self.xb(Bin::Add, at2, off);
                    self.x_raw(self.cur, h2, at, m)
                }
            };
            let skip = usize::from(c == 0);
            for k in 0..m as usize {
                if c == 0 && k == 0 {
                    out.push(read[0]);
                    continue;
                }
                let at = self.xi(4 * (c as usize + k) as u32);
                let second = self.xc(pred::IUGE, at, first);
                out.push(self.xs(second, over[k - skip], read[k]));
            }
        }
        out
    }

    /// Resolves the page groups whose bases the program has just made: by the joins of
    /// block `b` (and the program's inputs, in its first block), or by what its instruction
    /// `i` wrote. Every load of a base is below the place its last value is made, so a group
    /// resolved there serves them all. Goes on in another block.
    pub(super) fn x_ready(&mut self, b: usize, i: Option<usize>) {
        let Some(x) = self.xcl.as_ref() else { return };
        if x.groups.wanted.is_empty() {
            return;
        }
        let p = self.x.p;
        let mut found: Vec<usize> = Vec::new();
        let mut made = |v: Value| found.extend(x.groups.by_value.get(&v).into_iter().flatten());
        match i {
            Some(i) => p.blocks[b].insts[i].writes.iter().for_each(|&(_, v)| made(v)),
            None => {
                p.blocks[b].phis.iter().for_each(|phi| made(phi.value));
                if self.rpo[b] == 0 {
                    p.inputs.iter().for_each(|&(_, v)| made(v));
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        for w in found {
            let base = self.x_state().groups.wanted[w];
            // (a value not made yet, or not a dword: the loads resolve on their own)
            let known = |v: Value| match self.vals.get(&v) {
                Some(&Val::I32(x)) => Some(x),
                _ => None,
            };
            let (Some(lo), Some(hi)) = (known(base.lo), known(base.hi)) else { continue };
            let term = |v: Option<Value>| match v.map(known) {
                Some(None) => Err(()),
                Some(s) => Ok(s),
                None => Ok(None),
            };
            let (Ok(soffset), Ok(voff)) = (term(base.soffset), term(base.voff)) else { continue };
            let key = Base {
                lo,
                hi,
                soffset,
                voff,
                buffer: base.buffer,
                upper: base.upper,
            };
            if self.x_state().groups.made.contains_key(&key) {
                continue;
            }
            let g = self.x_group(key);
            self.x_state().groups.made.insert(key, g);
        }
    }

    /// The page group of `base`: the page of the base plus the offset's upper bits, and the
    /// next page, through one resolver that runs twice. Goes on in another block. A scalar
    /// load and a buffer load never share one: their origins differ.
    fn x_group(&mut self, base: Base<V>) -> Group {
        self.x_state().stats.page_groups += 1;
        let (i32t, zero) = (self.s.t.i32, self.xi(0));
        let (hi, origin) = if base.buffer {
            let (hi, _) = self.x_buffer_words(base.hi);
            (hi, Origin::Read(hi, true))
        } else {
            (base.hi, Origin::Read(base.hi, false))
        };
        let terms: Vec<V> = base.soffset.into_iter().chain(base.voff).collect();
        let (lo, hi) = self.x_address(base.lo, hi, &terms, base.upper << 16, None);
        let (pages, dwords, size) = (self.xi(0xffff_0000), self.xi(0xfffc), self.xi(0x1_0000));
        let page = self.xb(Bin::And, lo, pages);
        let low = self.xb(Bin::And, lo, dwords);
        let lo2 = self.xb(Bin::Add, page, size);
        let wrapped = self.xc(pred::EQ, lo2, zero);
        let carry = self.s.m.cast(self.cur, Cast::ZExt, wrapped, i32t);
        let hi2 = self.xb(Bin::Add, hi, carry);
        // The group's two pages, one resolver each: what used to be a loop of exactly two
        // rounds, with a phi for each value it carried.
        let (view0, byte0) = self.x_resolve(page, hi, origin, None, true);
        let (view, byte) = self.x_resolve(lo2, hi2, origin, None, true);
        // Both pages were resolved at their first byte. A member's dword is at the base's
        // low bits plus its offset's, less 64 KiB in the second page.
        let first_byte = self.xb(Bin::Add, byte0, low);
        let down = self.xb(Bin::Sub, low, size);
        let next_byte = self.xb(Bin::Add, byte, down);
        let handles = [self.x_data(view0), self.x_data(view)];
        let over = (!self.x_state().coherent).then(|| {
            let w = self.x_raw(self.cur, handles[1], byte, 3);
            [w[0], w[1], w[2]]
        });
        Group {
            views: [view0, view],
            bytes: [first_byte, next_byte],
            low,
            handles,
            next: byte,
            over,
        }
    }

    /// The page group that serves the load `inst` with these values for its base, and the
    /// low 16 bits of its offset.
    fn x_group_of(&mut self, inst: &Inst, lo: V, hi: V, soffset: Option<V>, voff: Option<V>) -> Option<(Group, u32)> {
        let (base, low) = match smem_base(inst) {
            Some((base, low)) => (base, low),
            None => {
                let (base, low, _) = buffer_base(inst)?;
                (base, low)
            }
        };
        let key = Base {
            lo,
            hi,
            soffset,
            voff,
            buffer: base.buffer,
            upper: base.upper,
        };
        let g = *self.x_state().groups.made.get(&key)?;
        self.x_state().stats.grouped_loads += 1;
        Some((g, low))
    }

    /// `n` dwords of a member of the page group `g` whose offset's low 16 bits are `low`,
    /// without a branch. They are read four at a time, from the page of the first of the
    /// four. The others of the four may lie in the next page, when the base is not on 16
    /// bytes: those are taken from one more read, of the next page's first dwords.
    fn x_group_load(&mut self, g: Group, low: u32, n: u32) -> Vec<V> {
        let over = match (n, g.over) {
            (1, _) => Vec::new(),
            (_, Some(w)) => w.to_vec(),
            (_, None) => self.x_raw(self.cur, g.handles[1], g.next, (n - 1).min(3)),
        };
        let mut out = Vec::new();
        for first in (0..n).step_by(4) {
            let (c, m) = (low + 4 * first, (n - first).min(4));
            #[cfg(test)]
            PIECES.with(|p| p.borrow_mut().push((c, m)));
            let (h, at) = if c == 0 {
                // (the base's own dword is in the first page)
                (g.handles[0], g.bytes[0])
            } else {
                // In the next page when the base's low bits plus `c` reach 64 KiB.
                let (room, k) = (self.xi(0xffff - c), self.xi(c));
                let next = self.xc(pred::IUGT, g.low, room);
                let view = self.xs(next, g.views[1], g.views[0]);
                let byte = self.xs(next, g.bytes[1], g.bytes[0]);
                (self.x_data(view), self.xb(Bin::Add, byte, k))
            };
            let read = self.x_raw(self.cur, h, at, m);
            out.push(read[0]);
            // `d` of the four are in the first page and the rest in the next, when the base's
            // low bits are 4 `d` below 64 KiB less `c`. Dword `k` is then the next page's
            // dword `k - d`.
            let split: Vec<V> = (1..m)
                .map(|d| {
                    let at = self.xi(0x1_0000 - c - 4 * d);
                    self.xc(pred::EQ, g.low, at)
                })
                .collect();
            for k in 1..m as usize {
                let mut x = read[k];
                for d in 1..=k {
                    x = self.xs(split[d - 1], over[k - d], x);
                }
                out.push(x);
            }
        }
        out
    }

    /// Stores `vals` as dwords from the address `lo`, `hi` (on a dword) on, each where
    /// its `oks` entry holds (none: always). A dword that is not stored goes to the null
    /// view. As [`Lower::x_load`] for a page crossing. `keeps`: a dword's bits outside its
    /// entry are left as they are, and its value's replace the others, by two atomics. Goes
    /// on in another block.
    fn x_store(&mut self, lo: V, hi: V, origin: Origin, kill: Option<V>, vals: &[V], oks: &[Option<V>], keeps: Option<&[V]>) {
        let (n, null) = (vals.len() as u32, self.x_words().null);
        if n == 1 {
            let (view, byte) = self.x_resolve(lo, hi, origin, kill, false);
            let view = match oks[0] {
                Some(ok) => self.xs(ok, view, null),
                None => view,
            };
            let h = self.x_data(view);
            match keeps {
                Some(keeps) => self.x_and_or(h, byte, keeps[0], vals[0]),
                None => self.x_raw_store(h, byte, vals[0]),
            }
            return;
        }
        let (i32t, i1t) = (self.s.t.i32, self.s.t.i1);
        let (zero, no, yes) = (self.xi(0), self.s.m.bool(false), self.s.m.bool(true));
        let (crosses, first, lo2, hi2) = self.x_pages(lo, hi, n);
        let whole = self.xb(Bin::Xor, crosses, yes);
        // (the first pass stores dword j)
        let early: Vec<V> = (0..n)
            .map(|j| {
                let at = self.xi(4 * j);
                let before = self.xc(pred::IULT, at, first);
                self.xb(Bin::Or, whole, before)
            })
            .collect();
        let pre = self.cur;
        let head = self.s.m.new_block();
        self.s.m.terminate(pre, Term::Br(head));
        self.cur = head;
        let (lo_now, hi_now, second) = (self.s.m.phi(head, i32t), self.s.m.phi(head, i32t), self.s.m.phi(head, i1t));
        let (view, byte) = self.x_resolve(lo_now, hi_now, origin, kill, true);
        let back = self.xs(second, first, zero);
        let at = self.xb(Bin::Sub, byte, back);
        for j in 0..n as usize {
            let mine = self.xb(Bin::Xor, second, early[j]);
            let mine = match oks[j] {
                Some(ok) => self.xb(Bin::And, mine, ok),
                None => mine,
            };
            let v = self.xs(mine, view, null);
            let h = self.x_data(v);
            let a = if j == 0 {
                at
            } else {
                let off = self.xi(4 * j as u32);
                self.xb(Bin::Add, at, off)
            };
            match keeps {
                Some(keeps) => self.x_and_or(h, a, keeps[j], vals[j]),
                None => self.x_raw_store(h, a, vals[j]),
            }
        }
        let fresh = self.xb(Bin::Xor, second, yes);
        let again = self.xb(Bin::And, crosses, fresh);
        let tail = self.cur;
        let done = self.s.m.new_block();
        self.s.m.terminate(tail, Term::CondBr(again, head, done));
        self.s.m.set_phi(lo_now, vec![(lo, pre), (lo2, tail)]);
        self.s.m.set_phi(hi_now, vec![(hi, pre), (hi2, tail)]);
        self.s.m.set_phi(second, vec![(no, pre), (yes, tail)]);
        self.cur = done;
    }

    /// `n` dwords from the byte address `lo`, `hi`, which need not lie on a dword: read as
    /// `n + 1` dwords from the dword below and shifted.
    fn x_unaligned(&mut self, lo: V, hi: V, origin: Origin, n: u32) -> Vec<V> {
        let (three, zero, k32, k31) = (self.xi(3), self.xi(0), self.xi(32), self.xi(31));
        let within = self.xb(Bin::And, lo, three);
        let aligned = self.xb(Bin::Sub, lo, within);
        let down = self.xb(Bin::Shl, within, three);
        let up = self.xb(Bin::Sub, k32, down);
        let up = self.xb(Bin::And, up, k31);
        let whole = self.xc(pred::EQ, within, zero);
        let w = self.x_load(aligned, hi, origin, None, n + 1);
        (0..n as usize)
            .map(|k| {
                let (a, b) = (self.xb(Bin::LShr, w[k], down), self.xb(Bin::Shl, w[k + 1], up));
                let joined = self.xb(Bin::Or, a, b);
                self.xs(whole, w[k], joined)
            })
            .collect()
    }

    /// The dword that holds the byte at `lo`, `hi`, shifted so the byte is its lowest.
    fn x_byte(&mut self, lo: V, hi: V, origin: Origin) -> V {
        let three = self.xi(3);
        let within = self.xb(Bin::And, lo, three);
        let aligned = self.xb(Bin::Sub, lo, within);
        let down = self.xb(Bin::Shl, within, three);
        let w = self.x_load(aligned, hi, origin, None, 1)[0];
        self.xb(Bin::LShr, w, down)
    }

    /// `x`'s low `bits` bits, zero- or sign-extended.
    fn x_extend(&mut self, x: V, bits: u32, signed: bool) -> V {
        if signed {
            let up = self.xi(32 - bits);
            let t = self.xb(Bin::Shl, x, up);
            self.xb(Bin::AShr, t, up)
        } else {
            let ones = self.xi((1 << bits) - 1);
            self.xb(Bin::And, x, ones)
        }
    }

    /// A byte or 16-bit load at the byte address `lo`, `hi`.
    fn x_small_load(&mut self, lo: V, hi: V, origin: Origin, bits: u32, signed: bool) -> V {
        let x = if bits == 8 { self.x_byte(lo, hi, origin) } else { self.x_unaligned(lo, hi, origin, 1)[0] };
        self.x_extend(x, bits, signed)
    }

    /// A byte or 16-bit store at the byte address `lo`, `hi`, in the lanes of `guard`: two
    /// atomics per dword it touches, which leave the dword's other bits alone.
    fn x_small_store(&mut self, lo: V, hi: V, base_hi: V, value: V, bits: u32, guard: Option<V>) {
        let (three, all) = (self.xi(3), self.xi(u32::MAX));
        let within = self.xb(Bin::And, lo, three);
        let aligned = self.xb(Bin::Sub, lo, within);
        let up = self.xb(Bin::Shl, within, three);
        let ones = self.xi((1 << bits) - 1);
        let value = self.xb(Bin::And, value, ones);
        let mask = self.xb(Bin::Shl, ones, up);
        let keep = self.xb(Bin::Xor, mask, all);
        let set = self.xb(Bin::Shl, value, up);
        let at = self.begin_guarded(guard);
        let (view, byte) = self.x_resolve(aligned, hi, Origin::Write(base_hi), None, false);
        let h = self.x_data(view);
        self.x_and_or(h, byte, keep, set);
        if bits == 16 {
            // The second byte of a value at a dword's last byte is in the next dword.
            let straddles = self.xc(pred::EQ, within, three);
            let (keep, k8) = (self.xi(0xffff_ff00), self.xi(8));
            let set = self.xb(Bin::LShr, value, k8);
            let (lo2, hi2) = self.address_plus(aligned, hi, 4);
            let next = self.begin_guarded(Some(straddles));
            let (view, byte) = self.x_resolve(lo2, hi2, Origin::Write(base_hi), None, false);
            let h = self.x_data(view);
            self.x_and_or(h, byte, keep, set);
            self.end_guarded(next, &[]);
        }
        self.end_guarded(at, &[]);
    }

    /// Replaces the bits of a dword that `keep` leaves out with `set`.
    fn x_and_or(&mut self, h: V, byte: V, keep: V, set: V) {
        let (i32t, u) = (self.s.t.i32, self.s.m.undef(self.s.t.i32));
        for (op, x) in [(1, keep), (2, set)] {
            let op = self.xi(op);
            self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, op, byte, u, u, x]);
        }
    }

    /// A 32-bit atomic `op` of the instruction `name` at the byte address `lo`, `hi`, in the
    /// lanes of `guard`: the value from before, 0 in the other lanes. An address off a dword
    /// reaches nothing.
    fn x_atomic(&mut self, name: Why, op: &str, lo: V, hi: V, base_hi: V, vals: &[V], guard: Option<V>) -> Result<V, Why> {
        // DXIL's operations: add 0, and 1, or 2, xor 3, signed minimum 4 and maximum 5,
        // unsigned 6 and 7, exchange 8.
        enum Way {
            Op(u32, V),
            Exchange,
            Loop,
        }
        let zero = self.xi(0);
        let way = match op {
            "add" => Way::Op(0, vals[0]),
            "sub" => Way::Op(0, self.xb(Bin::Sub, zero, vals[0])),
            "and" => Way::Op(1, vals[0]),
            "or" => Way::Op(2, vals[0]),
            "xor" => Way::Op(3, vals[0]),
            "smin" => Way::Op(4, vals[0]),
            "smax" => Way::Op(5, vals[0]),
            "umin" => Way::Op(6, vals[0]),
            "umax" => Way::Op(7, vals[0]),
            "swap" => Way::Op(8, vals[0]),
            "cmpswap" => Way::Exchange,
            "fmin" | "fmax" => Way::Loop,
            // (the 64-bit forms are not written)
            _ => return Err(name),
        };
        let three = self.xi(3);
        let within = self.xb(Bin::And, lo, three);
        let aligned = self.xb(Bin::Sub, lo, within);
        let odd = self.xc(pred::NE, within, zero);
        let (i32t, u) = (self.s.t.i32, self.s.m.undef(self.s.t.i32));
        let at = self.begin_guarded(guard);
        let (view, byte) = self.x_resolve(aligned, hi, Origin::Write(base_hi), Some(odd), false);
        let h = self.x_data(view);
        let was = match way {
            Way::Op(code, x) => {
                let code = self.xi(code);
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, code, byte, u, u, x])
            }
            // The data is the new value, then the value to compare with.
            Way::Exchange => self.s.op(self.cur, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &[h, byte, u, u, vals[1], vals[0]]),
            Way::Loop => self.x_atomic_loop(op == "fmax", h, byte, vals[0]),
        };
        Ok(self.end_guarded(at, &[(was, zero)])[0])
    }

    /// fmax (or fmin), which the host has no operation for, as xsemu's translator does it:
    /// the dword is exchanged for the larger (smaller) of it and `v` until no other lane got
    /// in between, at most 1024 times. The result is the value seen last.
    fn x_atomic_loop(&mut self, max: bool, h: V, byte: V, v: V) -> V {
        let (i32t, head) = (self.s.t.i32, self.cur);
        let u = self.s.m.undef(i32t);
        let first = self.x_raw(head, h, byte, 1)[0];
        let (pass, done) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(head, Term::Br(pass));
        self.cur = pass;
        let (c, n) = (self.s.m.phi(pass, i32t), self.s.m.phi(pass, i32t));
        let (zero, one) = (self.xi(0), self.xi(1));
        // The value when it orders above (below) the dword. A NaN or an equal value leaves
        // the dword.
        let (nan, equal, below) = if max { self.x_order(v, c) } else { self.x_order(c, v) };
        let kept = self.xb(Bin::Or, nan, equal);
        let kept = self.xb(Bin::Or, kept, below);
        let new = self.xs(kept, c, v);
        let found = self.s.op(pass, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &[h, byte, u, u, c, new]);
        let next = self.xb(Bin::Add, n, one);
        let (lost, limit) = (self.xc(pred::NE, found, c), self.xi(1024));
        let more = self.xc(pred::IULT, next, limit);
        let again = self.xb(Bin::And, lost, more);
        self.s.m.terminate(pass, Term::CondBr(again, pass, done));
        self.s.m.set_phi(c, vec![(first, head), (found, pass)]);
        self.s.m.set_phi(n, vec![(zero, head), (next, pass)]);
        self.cur = done;
        found
    }

    /// How two floats (as bits) order, as xsemu's translator orders them for atomics: on
    /// the bits, so a denormal is not 0. Whether either is a NaN, whether they are equal
    /// (+0 equals -0), and whether `a` is below `b` (meant only when neither holds).
    fn x_order(&mut self, a: V, b: V) -> (V, V, V) {
        let (mags, inf, zero) = (self.xi(0x7fff_ffff), self.xi(0x7f80_0000), self.xi(0));
        let (ax, bx) = (self.xb(Bin::And, a, mags), self.xb(Bin::And, b, mags));
        let (an, bn) = (self.xc(pred::IUGT, ax, inf), self.xc(pred::IUGT, bx, inf));
        let nan = self.xb(Bin::Or, an, bn);
        let both = self.xb(Bin::Or, ax, bx);
        let (same, zeros) = (self.xc(pred::EQ, a, b), self.xc(pred::EQ, both, zero));
        let equal = self.xb(Bin::Or, same, zeros);
        let (aneg, bneg) = (self.xc(pred::NE, a, ax), self.xc(pred::NE, b, bx));
        let signs = self.xb(Bin::Xor, aneg, bneg);
        let (more, fewer) = (self.xc(pred::IUGT, ax, bx), self.xc(pred::IULT, ax, bx));
        let by_size = self.xs(aneg, more, fewer);
        (nan, equal, self.xs(signs, aneg, by_size))
    }

    // ---- instructions ----

    fn x_sgpr(&mut self, inst: &Inst, code: u8, what: Why) -> Result<V, Why> {
        let v = inst.read(Reg::S(code)).ok_or(what)?;
        self.i32_of(v)
    }

    /// The high dword of a V#'s base and its stride, from its second word.
    fn x_buffer_words(&mut self, d1: V) -> (V, V) {
        let (top, k16, strides) = (self.xi(0xffff), self.xi(16), self.xi(0x3fff));
        let base_hi = self.xb(Bin::And, d1, top);
        let stride = self.xb(Bin::LShr, d1, k16);
        (base_hi, self.xb(Bin::And, stride, strides))
    }

    /// `s_load_dword*`: title memory at the address in an SGPR pair. `s_buffer_load_dword*`:
    /// the buffer of the V# in four SGPRs.
    pub(super) fn x_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            data,
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            return Err("a scalar memory access without a base");
        };
        let name = inst.name();
        if matches!(name, "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb") {
            return Ok(());
        }
        if matches!(name, "s_memtime" | "s_memrealtime") {
            let Some(Operand::Scalar { code, .. }) = *data else {
                return Err(name);
            };
            let zero = self.xi(0);
            for k in 0..2 {
                self.write(inst, Reg::S(code + k), zero, None);
            }
            return Ok(());
        }
        let (buffer, count) = smem_shape(name).ok_or(name)?;
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar load result");
        };
        let soff = match soffset {
            Operand::Scalar { code: NULL, .. } => None,
            op => Some(self.operand(inst, op)?),
        };
        let terms: Vec<V> = soff.into_iter().collect();
        let d0 = self.x_sgpr(inst, *base, "an address not in SGPRs")?;
        let d1 = self.x_sgpr(inst, base + 1, "an address not in SGPRs")?;
        let grouped = self.x_group_of(inst, d0, d1, soff, None);
        let stats = &mut self.x_state().stats;
        if buffer {
            stats.buffer_accesses += 1;
        } else if count >= 4 {
            stats.record_loads += 1;
        } else {
            stats.scalar_loads += 1;
        }
        let out = if buffer {
            let d2 = self.x_sgpr(inst, base + 2, "a buffer descriptor not in SGPRs")?;
            let (base_hi, stride) = self.x_buffer_words(d1);
            // (the offset of a buffer load has no sign)
            let imm = (*offset as u32) & 0x1f_ffff;
            // A dword is in range when its first byte is below the buffer's bytes, both as
            // 64-bit numbers.
            let (c, zero, i64t) = (self.cur, self.xi(0), self.s.m.int(64));
            self.s.flags |= flags::INT64;
            let records = self.s.m.cast(c, Cast::ZExt, d2, i64t);
            let wide = self.s.m.cast(c, Cast::ZExt, stride, i64t);
            let bytes = self.xb(Bin::Mul, records, wide);
            let stride0 = self.xc(pred::EQ, stride, zero);
            let limit = self.xs(stride0, records, bytes);
            let soff = soff.map(|s| self.s.m.cast(c, Cast::ZExt, s, i64t));
            let oks: Vec<V> = (0..count)
                .map(|k| {
                    let at = self.s.m.int_const(i64t, i64::from(imm + 4 * k));
                    let at = match soff {
                        Some(s) => self.xb(Bin::Add, s, at),
                        None => at,
                    };
                    self.xc(pred::IULT, at, limit)
                })
                .collect();
            let vals = match grouped {
                Some((g, low)) => self.x_group_load(g, low, count),
                None => {
                    let (lo, hi) = self.x_address(d0, base_hi, &terms, imm as i32, None);
                    let lo = self.x_aligned(lo);
                    self.x_load(lo, hi, Origin::Read(base_hi, true), None, count)
                }
            };
            vals.into_iter().zip(oks).map(|(x, ok)| self.xs(ok, x, zero)).collect()
        } else if let Some((g, low)) = grouped {
            self.x_group_load(g, low, count)
        } else {
            let (lo, hi) = self.x_address(d0, d1, &terms, *offset, None);
            let lo = self.x_aligned(lo);
            self.x_load(lo, hi, Origin::Read(d1, false), None, count)
        };
        for (k, x) in out.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// Whether an access of `bytes` bytes is inside its buffer.
    fn x_bound(&mut self, b: &Bounds, bytes: u32) -> V {
        let k = self.xi(b.offset + bytes);
        let by_offset = match b.voff {
            Some(voff) => {
                // (as a 64-bit sum: one that wraps is outside)
                let end = self.xb(Bin::Add, voff, k);
                let whole = self.xc(pred::IUGE, end, k);
                let inside = self.xc(pred::IULE, end, b.records);
                self.xb(Bin::And, whole, inside)
            }
            None => self.xc(pred::IULE, k, b.records),
        };
        self.xs(b.stride0, by_offset, b.by_index)
    }

    /// [`Lower::x_bound`] for a number of bytes (at most 16) only the shader knows.
    fn x_bound_sized(&mut self, b: &Bounds, bytes: V) -> V {
        let k = self.xi(b.offset);
        let k = self.xb(Bin::Add, bytes, k);
        let by_offset = match b.voff {
            Some(voff) => {
                let end = self.xb(Bin::Add, voff, k);
                let whole = self.xc(pred::IUGE, end, k);
                let inside = self.xc(pred::IULE, end, b.records);
                self.xb(Bin::And, whole, inside)
            }
            None => self.xc(pred::IULE, k, b.records),
        };
        self.xs(b.stride0, by_offset, b.by_index)
    }

    /// `buffer_load_*`, `buffer_store_*`, `buffer_atomic_*` and `tbuffer_load_format_*`: the
    /// buffer of the V# in four SGPRs, at an offset and an index each lane has.
    pub(super) fn x_buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if matches!(name, "buffer_gl0_inv" | "buffer_gl1_inv") {
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        enum Kind {
            Load(u32),
            Store(u32),
            SmallLoad(u32, bool),
            SmallStore(u32),
            Atomic(&'static str),
            /// The format the instruction names, the channels, and whether they are halves.
            Format(u8, u32, bool),
            /// A load in the format the V# names, of so many channels, and whether they are
            /// halves (Some) to the high half of a register (true).
            Described(u32, Option<bool>),
            /// A store of so many channels, in the format the V# or the instruction names,
            /// with halves as above.
            FormatStore(u32, Option<bool>),
        }
        let d16 = name.contains("_format_d16_");
        let channels = |rest: &str| match rest {
            "x" => Some(1),
            "xy" => Some(2),
            "xyz" => Some(3),
            "xyzw" => Some(4),
            _ => None,
        };
        // The channels of a format form, and its halves.
        let shaped = |rest: &str| match rest.strip_prefix("d16_") {
            Some("hi_x") => Some((1, Some(true))),
            Some(rest) => channels(rest).map(|n| (n, Some(false))),
            None => channels(rest).map(|n| (n, None)),
        };
        let kind = match name {
            "buffer_load_dword" => Kind::Load(1),
            "buffer_load_dwordx2" => Kind::Load(2),
            "buffer_load_dwordx3" => Kind::Load(3),
            "buffer_load_dwordx4" => Kind::Load(4),
            "buffer_store_dword" => Kind::Store(1),
            "buffer_store_dwordx2" => Kind::Store(2),
            "buffer_store_dwordx3" => Kind::Store(3),
            "buffer_store_dwordx4" => Kind::Store(4),
            "buffer_load_ubyte" => Kind::SmallLoad(8, false),
            "buffer_load_sbyte" => Kind::SmallLoad(8, true),
            "buffer_load_ushort" => Kind::SmallLoad(16, false),
            "buffer_load_sshort" => Kind::SmallLoad(16, true),
            "buffer_store_byte" => Kind::SmallStore(8),
            "buffer_store_short" => Kind::SmallStore(16),
            _ => {
                if let Some(op) = name.strip_prefix("buffer_atomic_") {
                    Kind::Atomic(op)
                } else if let Some((n, h)) = name.strip_prefix("buffer_load_format_").and_then(shaped) {
                    Kind::Described(n, h)
                } else if let Some((n, h)) = name.strip_prefix("buffer_store_format_").and_then(shaped) {
                    Kind::FormatStore(n, h)
                } else if let (Some((n, h)), true) = (name.strip_prefix("tbuffer_store_format_").and_then(shaped), format.is_some()) {
                    Kind::FormatStore(n, h)
                } else if let (Some(n), Some(f)) = (name.strip_prefix("tbuffer_load_format_").and_then(|r| channels(r.trim_start_matches("d16_"))), *format) {
                    Kind::Format(f, n, d16)
                } else {
                    return Err(name);
                }
            }
        };
        let Operand::Scalar { code: rs, .. } = *rsrc else {
            return Err("a buffer descriptor not in SGPRs");
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let voff = if *offen { Some(vaddr(self, u8::from(*idxen))?) } else { None };
        let soff = match soffset {
            Operand::Scalar { code: NULL, .. } | Operand::Const(128) => None,
            op => Some(self.operand(inst, op)?),
        };
        let d0 = self.x_sgpr(inst, rs, "a buffer descriptor not in SGPRs")?;
        let d1 = self.x_sgpr(inst, rs + 1, "a buffer descriptor not in SGPRs")?;
        let d2 = self.x_sgpr(inst, rs + 2, "a buffer descriptor not in SGPRs")?;
        // The V#'s last word names the format, where the instruction does not.
        let d3 = match (&kind, *format) {
            (Kind::Described(..) | Kind::FormatStore(..), None) => Some(self.x_sgpr(inst, rs + 3, "a buffer descriptor not in SGPRs")?),
            _ => None,
        };
        let reg = match *data {
            Some(Operand::Vector { reg, .. }) => reg,
            _ => return Err("buffer data"),
        };
        // What a store or an atomic takes, read before any code is emitted.
        let vals = match kind {
            Kind::Store(n) | Kind::FormatStore(n, None) => self.vgprs(inst, data.as_ref().expect("checked"), n as u8)?,
            Kind::FormatStore(n, Some(high)) => self.vgprs(inst, data.as_ref().expect("checked"), if high { 1 } else { n.div_ceil(2) as u8 })?,
            Kind::SmallStore(_) => self.vgprs(inst, data.as_ref().expect("checked"), 1)?,
            Kind::Atomic(op) => self.vgprs(inst, data.as_ref().expect("checked"), if op == "cmpswap" { 2 } else { 1 })?,
            _ => Vec::new(),
        };
        let guard = self.guard(b, i)?;
        self.x_state().stats.buffer_accesses += 1;
        let zero = self.xi(0);
        let (base_hi, stride) = self.x_buffer_words(d1);
        let terms: Vec<V> = soff.into_iter().chain(voff).collect();
        let (lo, hi) = self.x_address(d0, base_hi, &terms, i32::from(*offset), index.map(|x| (stride, x)));
        // With a stride the index is tested, without one the offset.
        let bounds = Bounds {
            stride0: self.xc(pred::EQ, stride, zero),
            voff,
            offset: u32::from(*offset),
            records: d2,
            by_index: match index {
                Some(x) => self.xc(pred::IULT, x, d2),
                None => self.xc(pred::NE, d2, zero),
            },
        };
        let read = Origin::Read(base_hi, true);
        // A vector load whose offset is a constant on a dword and whose lane offset is a
        // value joins the page group of its descriptor and that lane offset: several loads
        // of one descriptor at one lane offset share the two pages' resolutions.
        let grouped = match kind {
            Kind::Load(n) if n > 1 => self.x_group_of(inst, d0, d1, soff, voff),
            _ => None,
        };
        match kind {
            Kind::Load(n) => {
                let lo = self.x_aligned(lo);
                let oks: Vec<V> = (0..n).map(|k| self.x_bound(&bounds, 4 * k + 4)).collect();
                let got = match grouped {
                    Some((g, low)) => self.x_group_load(g, low, n),
                    None => self.x_load(lo, hi, read, None, n),
                };
                for (k, (x, ok)) in got.into_iter().zip(oks).enumerate() {
                    let x = self.xs(ok, x, zero);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
            }
            Kind::Store(n) => {
                let lo = self.x_aligned(lo);
                let oks: Vec<Option<V>> = (0..n).map(|k| Some(self.x_bound(&bounds, 4 * k + 4))).collect();
                let at = self.begin_guarded(guard);
                self.x_store(lo, hi, Origin::Write(base_hi), None, &vals, &oks, None);
                self.end_guarded(at, &[]);
            }
            Kind::Described(n, halves) => {
                let d3 = d3.expect("read above");
                let mut f = self.x_word_format(d3);
                if halves.is_some() && d16_half_bits() {
                    f = self.x_d16_twin(f, Some(d3), n);
                }
                let (mut xs, int) = self.x_format_load(f, Some(d3), lo, hi, read, &bounds, n);
                if let Some(high) = halves {
                    // (the high form keeps the register's low half)
                    let old = match (high, inst.read(Reg::V(reg))) {
                        (true, Some(v)) => Some(self.i32_of(v)?),
                        (true, None) => Some(zero),
                        _ => None,
                    };
                    xs = self.x_halves(&xs, int, old);
                }
                for (k, x) in xs.into_iter().enumerate() {
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
            }
            Kind::FormatStore(n, halves) => {
                let mut f = match (d3, *format) {
                    (Some(d3), _) => self.x_word_format(d3),
                    (None, f) => self.xi(u32::from(f.expect("the instruction's format"))),
                };
                if halves.is_some() && d16_half_bits() {
                    f = self.x_d16_twin(f, None, n);
                }
                self.x_format_store(f, lo, hi, base_hi, &bounds, &vals, halves.map(|h| (n, h)), guard);
            }
            Kind::SmallLoad(bits, signed) => {
                let ok = self.x_bound(&bounds, bits / 8);
                let x = self.x_small_load(lo, hi, read, bits, signed);
                let x = self.xs(ok, x, zero);
                self.write(inst, Reg::V(reg), x, guard);
            }
            Kind::SmallStore(bits) => {
                let ok = self.x_bound(&bounds, bits / 8);
                let lanes = self.x_and(guard, Some(ok));
                self.x_small_store(lo, hi, base_hi, vals[0], bits, lanes);
            }
            Kind::Atomic(op) => {
                let ok = self.x_bound(&bounds, 4);
                let lanes = self.x_and(guard, Some(ok));
                let was = self.x_atomic(name, op, lo, hi, base_hi, &vals, lanes)?;
                if cache.glc {
                    self.write(inst, Reg::V(reg), was, guard);
                }
            }
            Kind::Format(f, channels, halves) => {
                let f = if halves && d16_half_bits() { d16_twin(f, channels).unwrap_or(f) } else { f };
                let (bits, _) = super::buffer_format(f).ok_or("a buffer format")?;
                let total: u32 = bits.iter().sum();
                let mut ok = self.x_bound(&bounds, total / 8);
                // The bytes at the address, wherever it lies in a dword.
                let words = match total {
                    8 => vec![self.x_byte(lo, hi, read)],
                    16 => {
                        // (a 16-bit format at an odd address reads nothing)
                        let one = self.xi(1);
                        let odd = self.xb(Bin::And, lo, one);
                        let even = self.xc(pred::EQ, odd, zero);
                        ok = self.xb(Bin::And, ok, even);
                        self.x_unaligned(lo, hi, read, 1)
                    }
                    _ => self.x_unaligned(lo, hi, read, total / 32),
                };
                self.x_state().format_words = Some(words);
                let mut xs = self.format_load(lo, lo, f, channels)?;
                if halves {
                    xs = self.d16_channels(f, xs)?;
                }
                for (k, x) in xs.into_iter().enumerate() {
                    let x = self.xs(ok, x, zero);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
            }
        }
        Ok(())
    }

    /// The `global_*` and `flat_*` loads, stores and atomics: title memory at an address
    /// each lane holds, or at an SGPR pair plus each lane's offset.
    pub(super) fn x_flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds {
            return Err("a flat access to LDS");
        }
        enum Kind {
            Load(u32),
            Store(u32),
            SmallLoad(u32, bool),
            SmallStore(u32),
            Atomic(&'static str),
        }
        // (scratch memory and the forms by thread id are refused by name)
        let rest = name.strip_prefix("global_").or(name.strip_prefix("flat_")).ok_or(name)?;
        let kind = match rest {
            "load_dword" => Kind::Load(1),
            "load_dwordx2" => Kind::Load(2),
            "load_dwordx3" => Kind::Load(3),
            "load_dwordx4" => Kind::Load(4),
            "store_dword" => Kind::Store(1),
            "store_dwordx2" => Kind::Store(2),
            "store_dwordx3" => Kind::Store(3),
            "store_dwordx4" => Kind::Store(4),
            "load_ubyte" => Kind::SmallLoad(8, false),
            "load_sbyte" => Kind::SmallLoad(8, true),
            "load_ushort" => Kind::SmallLoad(16, false),
            "load_sshort" => Kind::SmallLoad(16, true),
            "store_byte" => Kind::SmallStore(8),
            "store_short" => Kind::SmallStore(16),
            _ => Kind::Atomic(rest.strip_prefix("atomic_").ok_or(name)?),
        };
        let Some(Operand::Vector { reg: areg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a global address")?;
            l.i32_of(v)
        };
        let v0 = vaddr(self, 0)?;
        // The base: the VGPR pair, each lane its own, or an SGPR pair.
        let base = match *saddr {
            None | Some(Operand::Scalar { code: NULL, .. }) => None,
            Some(Operand::Scalar { code, .. }) => Some((
                self.x_sgpr(inst, code, "a global address base")?,
                self.x_sgpr(inst, code + 1, "a global address base")?,
            )),
            Some(_) => return Err("a global address base"),
        };
        let v1 = if base.is_none() { Some(vaddr(self, 1)?) } else { None };
        let vals = match kind {
            Kind::Store(n) => self.vgprs(inst, data.as_ref().ok_or("global store data")?, n as u8)?,
            Kind::SmallStore(_) => self.vgprs(inst, data.as_ref().ok_or("global store data")?, 1)?,
            Kind::Atomic(op) => {
                self.vgprs(inst, data.as_ref().ok_or("global atomic data")?, if op == "cmpswap" { 2 } else { 1 })?
            }
            _ => Vec::new(),
        };
        let result = match *dst {
            Some(Operand::Vector { reg, .. }) => Some(reg),
            _ => None,
        };
        if matches!(kind, Kind::Load(_) | Kind::SmallLoad(..)) && result.is_none() {
            return Err("a global load result");
        }
        let guard = self.guard(b, i)?;
        self.x_state().stats.global_accesses += 1;
        let (lo, hi, from, read) = match (base, v1) {
            (Some((s0, s1)), _) => {
                let (lo, hi) = self.x_address(s0, s1, &[v0], i32::from(*offset), None);
                (lo, hi, s1, Origin::Read(s1, false))
            }
            (None, Some(v1)) => {
                let (lo, hi) = self.address_plus(v0, v1, i32::from(*offset));
                (lo, hi, v1, Origin::Lane(v1))
            }
            (None, None) => unreachable!(),
        };
        match kind {
            Kind::Load(n) => {
                let lo = self.x_aligned(lo);
                let got = self.x_load(lo, hi, read, None, n);
                for (k, x) in got.into_iter().enumerate() {
                    self.write(inst, Reg::V(result.expect("checked") + k as u8), x, guard);
                }
            }
            Kind::Store(n) => {
                let lo = self.x_aligned(lo);
                let at = self.begin_guarded(guard);
                self.x_store(lo, hi, Origin::Write(from), None, &vals, &vec![None; n as usize], None);
                self.end_guarded(at, &[]);
            }
            Kind::SmallLoad(bits, signed) => {
                let x = self.x_small_load(lo, hi, read, bits, signed);
                self.write(inst, Reg::V(result.expect("checked")), x, guard);
            }
            Kind::SmallStore(bits) => self.x_small_store(lo, hi, from, vals[0], bits, guard),
            Kind::Atomic(op) => {
                let was = self.x_atomic(name, op, lo, hi, from, &vals, guard)?;
                if let (Some(reg), true) = (result, cache.glc) {
                    self.write(inst, Reg::V(reg), was, guard);
                }
            }
        }
        Ok(())
    }

    // ---- images and samplers ----

    /// The constant an SGPR value holds, when the module has it as one: a constant, or a
    /// join whose every way in brings the same one.
    fn x_const(&mut self, v: Value) -> Option<u32> {
        if let Some(&known) = self.x_state().consts.get(&v) {
            return known;
        }
        if self.x_state().phis.is_none() {
            let p = self.x.p;
            let phis = p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| block.phis.iter().enumerate().map(move |(n, phi)| (phi.value, (b, n))))
                .collect();
            // A constant reversed is a constant: xsemu's translator takes S# words made by
            // s_brev_b32 of an inline constant or a literal as known.
            let reversed = p
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| inst.name() == "s_brev_b32")
                .filter_map(|inst| {
                    let Fields::Sop { src, .. } = &inst.fields else {
                        return None;
                    };
                    let k = match *src.first()? {
                        Operand::Const(c) => crate::resource::inline(c)?,
                        Operand::Literal(x) => x,
                        _ => return None,
                    };
                    let to = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))?.1;
                    Some((to, k.reverse_bits()))
                })
                .collect();
            let x = self.x_state();
            x.phis = Some(phis);
            x.reversed = reversed;
        }
        let known = self.x_const_in(v, &mut Vec::new()).ok().flatten();
        self.x_state().consts.insert(v, known);
        known
    }

    /// [`Lower::x_const`] below the joins on `path`. Ok(None): a join on the path, which
    /// brings what the others bring.
    fn x_const_in(&mut self, v: Value, path: &mut Vec<Value>) -> Result<Option<u32>, ()> {
        let phi = self.xcl.as_ref().and_then(|x| x.phis.as_ref()).and_then(|p| p.get(&v)).copied();
        let Some((b, n)) = phi else {
            if let Some(&k) = self.xcl.as_ref().and_then(|x| x.reversed.get(&v)) {
                return Ok(Some(k));
            }
            return match self.vals.get(&v) {
                Some(&Val::I32(x)) => self.s.m.int_k(x).map(|(_, k)| Some(k as u32)).ok_or(()),
                _ => Err(()),
            };
        };
        if path.contains(&v) {
            return Ok(None);
        }
        path.push(v);
        let p = self.x.p;
        let mut found: Option<u32> = None;
        for &arg in &p.blocks[b].phis[n].args {
            match self.x_const_in(arg, path)? {
                None => {}
                Some(k) if found.is_none_or(|f| f == k) => found = Some(k),
                Some(_) => return Err(()),
            }
        }
        path.pop();
        Ok(found)
    }

    /// The handle of the image or the sampler the instruction at `site` names: the view or
    /// the sampler its registers hold.
    pub(super) fn x_handle(&mut self, site: Site, sampler: bool, props: (u32, u32)) -> Result<V, Why> {
        let p = self.x.p;
        let inst = &p.blocks[site.0].insts[site.1];
        let Fields::Image { rsrc, sampler: state, .. } = &inst.fields else {
            return Err("a descriptor outside an image instruction");
        };
        if !sampler {
            // The view heap index is the T#'s first word, and the writable view follows it.
            let Operand::Scalar { code, .. } = *rsrc else {
                return Err("an image descriptor not in SGPRs");
            };
            let w0 = self.x_sgpr(inst, code, "an image descriptor not in SGPRs")?;
            self.x_state().stats.image_ops += 1;
            let index = if props.0 & 0x1000 != 0 {
                let one = self.xi(1);
                self.xb(Bin::Add, w0, one)
            } else {
                w0
            };
            return Ok(self.x_heap(index, false, true, props));
        }
        let Some(Operand::Scalar { code, .. }) = *state else {
            return Err("a sampler descriptor not in SGPRs");
        };
        let mut words = [None; 4];
        for (k, word) in words.iter_mut().enumerate() {
            let v = inst.read(Reg::S(code + k as u8)).ok_or("a sampler descriptor not in SGPRs")?;
            *word = self.x_const(v);
        }
        if let [Some(a), Some(b), Some(c), Some(d)] = words {
            // Constants: a sampler of the root signature.
            let n = self.register_sampler(&[a, b, c, d], props.0 & 0x8000 != 0)?;
            return Ok(self.s.sampler_handle(self.cur, n));
        }
        // The host wrote the sampler heap index into the S#'s last word.
        let w3 = self.x_sgpr(inst, code + 3, "a sampler descriptor not in SGPRs")?;
        self.x_state().stats.heap_samplers += 1;
        let low = self.xi(0xfff);
        let index = self.xb(Bin::And, w3, low);
        Ok(self.x_heap(index, true, true, props))
    }

    /// The numeric class of the image a T#'s second word names: its format (bits 28:20)
    /// looked up in a table of two bits per format, as xsemu's translator does with
    /// `img_cls`. Branch-free.
    fn x_image_class(&mut self, w1: V) -> V {
        let (k20, k511, k4, k15, k3) = (self.xi(20), self.xi(0x1ff), self.xi(4), self.xi(15), self.xi(3));
        let format = self.xb(Bin::LShr, w1, k20);
        let format = self.xb(Bin::And, format, k511);
        let word = self.xb(Bin::LShr, format, k4);
        let mut bits = self.xi(0);
        for (n, entry) in Class::table().into_iter().enumerate() {
            let (n, entry) = (self.xi(n as u32), self.xi(entry));
            let here = self.xc(pred::EQ, word, n);
            bits = self.xs(here, entry, bits);
        }
        let place = self.xb(Bin::And, format, k15);
        let one = self.xi(1);
        let place = self.xb(Bin::Shl, place, one);
        let class = self.xb(Bin::LShr, bits, place);
        self.xb(Bin::And, class, k3)
    }

    /// The view an image instruction names: the heap index of its read-only or writable
    /// view, and the T#'s numeric class, known when the module has its second word as a
    /// constant. Refuses before it emits anything that is not pure.
    fn x_view(&mut self, inst: &Inst, props0: u32) -> Result<View, Why> {
        let Fields::Image { rsrc, .. } = &inst.fields else {
            return Err("a descriptor outside an image instruction");
        };
        let Operand::Scalar { code, .. } = *rsrc else {
            return Err("an image descriptor not in SGPRs");
        };
        let w0 = self.x_sgpr(inst, code, "an image descriptor not in SGPRs")?;
        let word1 = inst.read(Reg::S(code + 1)).ok_or("an image descriptor not in SGPRs")?;
        let w1 = self.i32_of(word1)?;
        let index = if props0 & 0x1000 != 0 {
            let one = self.xi(1);
            self.xb(Bin::Add, w0, one)
        } else {
            w0
        };
        if let Some(k) = self.x_const(word1) {
            let known = Class::of(k >> 20 & 0x1ff);
            return Ok(View { index, class: self.xi(known as u32), known: Some(known) });
        }
        Ok(View { index, class: self.x_image_class(w1), known: None })
    }

    /// An image access through a view whose component type follows the T#'s numeric class,
    /// in the shape of xsemu's translator: a branch on the class, and in each way a view of
    /// that type (`access` emits the access on it and returns its results as i32 values).
    /// Gives the results joined. A known class takes its one way only.
    fn x_by_class(&mut self, view: &View, props0: u32, mut access: impl FnMut(&mut Self, V, Class) -> Vec<V>) -> Vec<V> {
        if let Some(class) = view.known {
            let h = self.x_heap(view.index, false, true, (props0, class.component()));
            return access(self, h, class);
        }
        let merge = self.s.m.new_block();
        let mut incoming: Vec<Vec<(V, usize)>> = Vec::new();
        for way in [Class::Uint, Class::Sint, Class::Float] {
            if way == Class::Float {
                self.x_class_way(view.index, props0, way, merge, &mut access, &mut incoming);
                continue;
            }
            let (arm, next) = (self.s.m.new_block(), self.s.m.new_block());
            let k = self.xi(way as u32);
            let is = self.xc(pred::EQ, view.class, k);
            self.s.m.terminate(self.cur, Term::CondBr(is, arm, next));
            self.cur = arm;
            self.x_class_way(view.index, props0, way, merge, &mut access, &mut incoming);
            self.cur = next;
        }
        self.cur = merge;
        let i32t = self.s.t.i32;
        incoming
            .into_iter()
            .map(|ways| {
                let phi = self.s.m.phi(merge, i32t);
                self.s.m.set_phi(phi, ways);
                phi
            })
            .collect()
    }

    /// One way of [`Lower::x_by_class`], from the current block to `merge`.
    fn x_class_way(&mut self, index: V, props0: u32, class: Class, merge: usize, access: &mut impl FnMut(&mut Self, V, Class) -> Vec<V>, incoming: &mut Vec<Vec<(V, usize)>>) {
        let h = self.x_heap(index, false, true, (props0, class.component()));
        let results = access(self, h, class);
        incoming.resize_with(results.len(), Vec::new);
        for (ways, x) in incoming.iter_mut().zip(results) {
            ways.push((x, self.cur));
        }
        self.s.m.terminate(self.cur, Term::Br(merge));
    }

    /// `base` (an image intrinsic without its overload) on a view of `class`, with `args`
    /// after the handle: the i32 overload for integer views, the f32 one for float views.
    fn x_class_op(&mut self, base: &str, class: Class, opcode: u32, h: V, args: &[V]) -> V {
        let overload = if class == Class::Float { "f32" } else { "i32" };
        let (attr, ty) = match base {
            "textureStore" => (Attr::NoUnwind, self.s.t.void),
            _ if class == Class::Float => (Attr::ReadOnly, self.s.t.res_ret),
            _ => (Attr::ReadOnly, self.x_state().ret),
        };
        let name = format!("{base}.{overload}");
        #[cfg(test)]
        TYPED.with(|t| t.borrow_mut().push((name.clone(), class.component())));
        let mut all = vec![h];
        all.extend(args);
        self.s.op(self.cur, &name, attr, ty, opcode, &all)
    }

    /// `image_store` under the binding: `cs` the coordinates, the data from VGPR `reg` on,
    /// through a view of the T#'s class. The channels of the mask take the registers in
    /// order and the others are 0. A float view takes the registers as floats (d16: halves
    /// widened), an integer view their bits (d16: the halves' 16 bits), as xsemu's
    /// translator stores them.
    pub(super) fn x_image_store(&mut self, inst: &Inst, guard: Option<V>, cs: &[V], reg: u8, props0: u32) -> Result<(), Why> {
        let Fields::Image { d16, dmask, .. } = inst.fields else {
            return Err("an image store");
        };
        let (mut bits, mut floats) = (Vec::new(), Vec::new());
        let mut k = 0u8;
        for ch in 0..4 {
            if dmask >> ch & 1 == 0 {
                bits.push(self.xi(0));
                floats.push(self.s.m.f32(0));
                continue;
            }
            let v = inst.read(Reg::V(reg + if d16 { k / 2 } else { k })).ok_or("image store data")?;
            if d16 {
                let mut x = self.i32_of(v)?;
                if k % 2 == 1 {
                    let k16 = self.xi(16);
                    x = self.xb(Bin::LShr, x, k16);
                }
                let low = self.xi(0xffff);
                bits.push(self.xb(Bin::And, x, low));
                let f32t = self.s.t.f32;
                floats.push(self.s.op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]));
            } else {
                bits.push(self.i32_of(v)?);
                floats.push(self.float_of(v)?);
            }
            k += 1;
        }
        let view = self.x_view(inst, props0)?;
        self.x_state().stats.image_ops += 1;
        let at = self.begin_guarded(guard);
        self.x_by_class(&view, props0, |l, h, class| {
            let mut args = cs.to_vec();
            args.extend(if class == Class::Float { &floats } else { &bits });
            args.push(l.s.m.i8(0xF));
            l.x_class_op("textureStore", class, 67, h, &args);
            Vec::new()
        });
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// `image_load*` (`gather` false) or a gather (true) under the binding, `args` after the
    /// handle, through a view of the T#'s class; writes the result registers. A load's d16
    /// halves are rounded floats from a float view and the low 16 bits from an integer
    /// view. A gather's are rounded floats, as in xsemu's translator and CGBJ's own code.
    pub(super) fn x_image_read(&mut self, inst: &Inst, guard: Option<V>, props0: u32, gather: bool, args: &[V]) -> Result<(), Why> {
        let Fields::Image { data, dmask, d16, .. } = inst.fields else {
            return Err("an image read");
        };
        let Operand::Vector { reg, .. } = data else {
            return Err("an image result");
        };
        let dmask = if dmask == 0 { 1 } else { dmask };
        let (base, opcode) = if gather { ("textureGather", 73) } else { ("textureLoad", 66) };
        let view = self.x_view(inst, props0)?;
        let texel = self.x_by_class(&view, props0, |l, h, class| {
            let r = l.x_class_op(base, class, opcode, h, args);
            (0..4)
                .map(|ch| {
                    let x = l.s.m.extract(l.cur, r, ch);
                    if class == Class::Float { l.as_int(x) } else { x }
                })
                .collect()
        });
        let channels: Vec<usize> = (0..4).filter(|&ch| gather || dmask >> ch & 1 != 0).collect();
        let mut xs = Vec::new();
        for ch in channels {
            let x = texel[ch];
            if !d16 {
                xs.push(x);
                continue;
            }
            let (f, i32t) = (self.as_float(x), self.s.t.i32);
            let half = self.s.op(self.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            xs.push(if gather {
                half
            } else {
                let (zero, low) = (self.xi(0), self.xi(0xffff));
                let float = self.xc(pred::EQ, view.class, zero);
                let bits = self.xb(Bin::And, x, low);
                self.xs(float, half, bits)
            });
        }
        if d16 {
            let k16 = self.xi(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.xb(Bin::Shl, hi, k16);
                        self.xb(Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}

/// What [`Lower::x_view`] gives.
struct View {
    index: V,
    /// The class as a value: 0 float, 1 uint, 2 sint.
    class: V,
    known: Option<Class>,
}

// xsemu: from fxceless/victory-fork d4b786f (PR 927, gpu/half-views) crates/victory-gpu/src/descriptor/host.rs (half_format), GPL-3.0
/// The unsigned format of the same layout for a d16 access whose 16 bits move unchanged
/// (`XCL_RECOMP_D16_HALF_BITS`): the three 16-bit float formats, when the format has every
/// channel the access names. `None` for every other format (16-bit integers already move
/// their bits; normalised and scaled ones the console converts).
fn d16_twin(f: u8, channels: u32) -> Option<u8> {
    let twin = match f {
        13 => 11,
        29 => 27,
        73 => 69,
        _ => return None,
    };
    (u32::from(f & 7) >= channels).then_some(twin)
}

/// The numeric class of an image format, as xsemu's translator reads it from the buffer
/// format table (its memory access notes, section 7.1): the uint and sint formats, and
/// float for every other one, formats past 77 included. The value is the translator's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Float = 0,
    Uint = 1,
    Sint = 2,
}

impl Class {
    fn of(format: u32) -> Class {
        if format > formats::LAST {
            return Class::Float;
        }
        match formats::shape(format).1 {
            4 => Class::Uint,
            5 => Class::Sint,
            _ => Class::Float,
        }
    }

    /// The second word of a four-channel view's properties: the component type.
    fn component(self) -> u32 {
        0x400
            | match self {
                Class::Float => 9,
                Class::Uint => 5,
                Class::Sint => 4,
            }
    }

    /// Two bits per format, 16 formats per word, for [`Lower::x_image_class`]. Formats
    /// past the table's words read 0, float.
    fn table() -> [u32; 5] {
        let mut t = [0; 5];
        for format in 0..=formats::LAST {
            t[format as usize / 16] |= (Class::of(format) as u32) << (format % 16 * 2);
        }
        t
    }
}

#[cfg(test)]
thread_local! {
    /// The image intrinsics [`Lower::x_class_op`] emitted on this thread, with the view's
    /// component type, for the tests.
    pub(super) static TYPED: std::cell::RefCell<Vec<(String, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The pieces of four dwords [`Lower::x_group_load`] read on this thread: the byte of
    /// each piece's first dword from the base's low bits, and its dwords, for the tests.
    pub(super) static PIECES: std::cell::RefCell<Vec<(u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
#[path = "xcl_tests.rs"]
mod tests;
