// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../../LICENSE and ../../../PROVENANCE.md), so it carries that licence.
//! Buffer formats that a V# names: decoded and encoded through tables, because the format is
//! only known when the shader runs. The rules are those of xsemu's translator (its memory
//! access notes, section 3.7): what a format's components are, how a component is decoded
//! and encoded, and how the bytes are stored.

use super::super::super::bitcode::{Attr, Bin, Cast, V, pred};
use super::super::super::shader::flags;
use super::super::Lower;
use super::{Bounds, Origin};

/// The last format's number. A larger number is taken as this one.
pub(super) const LAST: u32 = 77;

/// The formats in the order of their numbers from 1 on: the widths as the names list them,
/// the most significant component first, and how many numeric types follow one another.
/// 6: unorm, snorm, uscaled, sscaled, uint, sint. 7: those and float. 3: uint, sint, float.
const SHAPES: [(&[u32], u32); 14] = [
    (&[8], 6),
    (&[16], 7),
    (&[8, 8], 6),
    (&[32], 3),
    (&[16, 16], 7),
    (&[10, 11, 11], 7),
    (&[11, 11, 10], 7),
    (&[10, 10, 10, 2], 6),
    (&[2, 10, 10, 10], 6),
    (&[8, 8, 8, 8], 6),
    (&[32, 32], 3),
    (&[16, 16, 16, 16], 7),
    (&[32, 32, 32], 3),
    (&[32, 32, 32, 32], 3),
];

/// A format's component widths, the least significant first, and its numeric type: 0 unorm,
/// 1 snorm, 2 uscaled, 3 sscaled, 4 uint, 5 sint, 6 float. Format 0 has no components.
pub(super) fn shape(format: u32) -> (Vec<u32>, u32) {
    let Some(mut n) = format.min(LAST).checked_sub(1) else {
        return (Vec::new(), 0);
    };
    for (widths, types) in SHAPES {
        if n < types {
            let ty = if types == 3 { n + 4 } else { n };
            return (widths.iter().rev().copied().collect(), ty);
        }
        n -= types;
    }
    unreachable!("77 formats")
}

/// What a load or a store asks of a format first: its bytes [7:0], whether it is an integer
/// format [8], and whether it has a component at all [9].
fn meta_entry(format: u32) -> u32 {
    let (widths, ty) = shape(format);
    widths.iter().sum::<u32>() / 8 | u32::from(ty == 4 || ty == 5) << 8 | u32::from(!widths.is_empty()) << 9
}

/// How component c of a format is decoded. Word 0: bit offset [7:0], width [13:8], class
/// [15:14] (norm, scaled, integer, float), the shift that sign-extends [20:16] (0: no sign),
/// the shift that makes an 11- or 10-bit float a half [25:21], absent [26], 32-bit float
/// [27], 11- or 10-bit float [28], signed [29]. Word 1: the norm maximum as a float. Word
/// 2: the width's mask. Word 3: the value of an absent component.
pub(super) fn decode_entry(format: u32, c: u32) -> [u32; 4] {
    let (widths, ty) = shape(format);
    let n = widths.len() as u32;
    let width = |k: u32| widths.get(k as usize).copied().unwrap_or(0);
    let (off, w) = ((0..c).map(width).sum::<u32>(), width(c));
    let signed = ty & 1 != 0 && ty <= 5;
    let max = if signed {
        if w >= 32 { 2147483647.0f32 } else { (1u32 << (w.wrapping_sub(1) & 31)).wrapping_sub(1) as f32 }
    } else if w >= 32 {
        4294967295.0f32
    } else {
        ((1u32 << (w & 31)) - 1) as f32
    };
    let mask = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
    let class = match ty {
        0 | 1 => 0,
        2 | 3 => 1,
        4 | 5 => 2,
        _ => 3,
    };
    let sh = if signed { 32u32.wrapping_sub(w) & 31 } else { 0 };
    let small = class == 3 && w != 32 && w != 16;
    let half_shift = if small { 15u32.wrapping_sub(w) & 31 } else { 0 };
    let absent = match c {
        _ if c < n => 0,
        3 if ty == 4 || ty == 5 => 1,
        3 => 0x3f80_0000,
        _ => 0,
    };
    let x = off
        | w << 8
        | class << 14
        | sh << 16
        | half_shift << 21
        | u32::from(c >= n) << 26
        | u32::from(w == 32) << 27
        | u32::from(small) << 28
        | u32::from(signed) << 29;
    [x, max.to_bits(), mask, absent]
}

/// How a value is encoded as a component ([\`encode_entry\`]).
const UNORM: u32 = 1;
const SNORM: u32 = 2;
const USCALED: u32 = 3;
const SSCALED: u32 = 4;
const BITS: u32 = 5;
const HALF: u32 = 6;
const FLOAT11: u32 = 7;
const FLOAT10: u32 = 8;

/// How component c of a format is encoded. Word 0: the shift in its dword [4:0], the dword
/// [6:5], the kind [11:8]. Word 1: the width's mask, 0 for a component the format lacks.
/// Word 2: the norm maximum.
pub(super) fn encode_entry(format: u32, c: u32) -> [u32; 3] {
    let (widths, ty) = shape(format);
    let Some(&w) = widths.get(c as usize) else {
        return [0; 3];
    };
    let off: u32 = widths[..c as usize].iter().sum();
    let mask = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
    let kind = match (ty, w) {
        (0, _) => UNORM,
        (1, _) => SNORM,
        (2, _) => USCALED,
        (3, _) => SSCALED,
        (4 | 5, _) | (_, 32) => BITS,
        (_, 16) => HALF,
        (_, 11) => FLOAT11,
        _ => FLOAT10,
    };
    [off | kind << 8, mask, if ty == 1 { mask >> 1 } else { mask }]
}

/// The format tables of a module.
#[derive(Clone, Copy)]
pub(super) struct Tables {
    meta: V,
    decode: V,
    encode: V,
}

impl<'a> Lower<'a> {
    fn x_tables(&mut self) -> Tables {
        if let Some(t) = self.x_state().formats {
            return t;
        }
        let formats = 0..=LAST;
        let meta: Vec<u32> = formats.clone().map(meta_entry).collect();
        let decode: Vec<u32> = formats.clone().flat_map(|f| (0..4).flat_map(move |c| decode_entry(f, c))).collect();
        let encode: Vec<u32> = formats.flat_map(|f| (0..4).flat_map(move |c| encode_entry(f, c))).collect();
        let mut table = |name: &str, words: Vec<u32>| {
            let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
            let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
            let init = self.s.m.aggregate(arr, &elems);
            self.s.m.constant_global(name, init)
        };
        let t = Tables {
            meta: table("xcl.format", meta),
            decode: table("xcl.decode", decode),
            encode: table("xcl.encode", encode),
        };
        self.x_state().formats = Some(t);
        t
    }

    fn x_entry(&mut self, table: V, index: V) -> V {
        let at = self.s.m.element(self.cur, table, index);
        self.s.m.load(self.cur, at)
    }

    /// Whether \`bit\` of \`x\` is set.
    fn x_bit(&mut self, x: V, bit: u32) -> V {
        let (k, zero) = (self.xi(1 << bit), self.xi(0));
        let b = self.xb(Bin::And, x, k);
        self.xc(pred::NE, b, zero)
    }

    /// \`x\` shifted right by \`by\` and masked.
    fn x_field(&mut self, x: V, by: u32, mask: u32) -> V {
        let (by, mask) = (self.xi(by), self.xi(mask));
        let x = self.xb(Bin::LShr, x, by);
        self.xb(Bin::And, x, mask)
    }

    /// The format a V#'s last word names.
    pub(super) fn x_word_format(&mut self, d3: V) -> V {
        self.x_field(d3, 12, 0x7f)
    }

    /// The format number as the tables take it, its bytes, whether it is an integer format,
    /// and whether an access of its bytes is one at all: the format has components and the
    /// bytes are inside the buffer.
    fn x_format(&mut self, format: V, bounds: &Bounds) -> (V, V, V, V) {
        let t = self.x_tables();
        let last = self.xi(LAST);
        let over = self.xc(pred::IUGT, format, last);
        let f = self.xs(over, last, format);
        let meta = self.x_entry(t.meta, f);
        let bytes = self.xi(0xff);
        let size = self.xb(Bin::And, meta, bytes);
        let (int, some) = (self.x_bit(meta, 8), self.x_bit(meta, 9));
        let inside = self.x_bound_sized(bounds, size);
        (f, size, int, self.xb(Bin::And, some, inside))
    }

    /// A component of the raw dwords \`r\`, decoded by the entry \`e\` ([\`decode_entry\`]).
    fn x_format_component(&mut self, r: &[V], e: &[V]) -> V {
        let c = self.cur;
        let (zero, one, k31, f32t, i32t) = (self.xi(0), self.xi(1), self.xi(31), self.s.t.f32, self.s.t.i32);
        // No component crosses a dword.
        let off = self.x_field(e[0], 0, 0xff);
        let i = self.x_field(off, 5, 7);
        let (two, is0, is1) = (self.xi(2), self.xc(pred::EQ, i, zero), self.xc(pred::EQ, i, one));
        let is2 = self.xc(pred::EQ, i, two);
        let w = self.xs(is2, r[2], r[3]);
        let w = self.xs(is1, r[1], w);
        let w = self.xs(is0, r[0], w);
        let by = self.xb(Bin::And, off, k31);
        let v = self.xb(Bin::LShr, w, by);
        let v = self.xb(Bin::And, v, e[2]);
        // Sign-extended (without a sign: a shift of 0), and as a float either way.
        let sh = self.x_field(e[0], 16, 31);
        let sv = self.xb(Bin::Shl, v, sh);
        let sv = self.xb(Bin::AShr, sv, sh);
        let signed = self.x_bit(e[0], 29);
        let (fs, fu) = (self.s.m.cast(c, Cast::SiToFp, sv, f32t), self.s.m.cast(c, Cast::UiToFp, v, f32t));
        let fx = self.xs(signed, fs, fu);
        // A norm: exact at 0 and at the maximum, and not below -1.
        let m = self.as_float(e[1]);
        let (f0, f1, below) = (self.s.m.f32(0), self.s.m.f32(0x3f80_0000), self.s.m.f32(0xbf80_0000));
        let q = self.xb(Bin::SDiv, fx, m);
        let q = self.s.op(c, "binary.f32", Attr::ReadNone, f32t, 35, &[q, below]);
        let most = self.xc(pred::OGE, fx, m);
        let q = self.xs(most, f1, q);
        let none = self.xc(pred::EQ, v, zero);
        let norm = self.xs(none, f0, q);
        let norm = self.as_int(norm);
        // A float: 32 bits as they are, 16 bits widened. 11 and 10 bits are made halves
        // first, and their NaNs are the one NaN.
        let by = self.x_field(e[0], 21, 31);
        let h = self.xb(Bin::Shl, v, by);
        let wide = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        let wide = self.as_int(wide);
        let (payload, inf, nan) = (self.xi(0x7fff), self.xi(0x7c00), self.xi(0x7fc0_0000));
        let p = self.xb(Bin::And, h, payload);
        let p = self.xc(pred::IUGT, p, inf);
        let small = self.x_bit(e[0], 28);
        let p = self.xb(Bin::And, small, p);
        let float = self.xs(p, nan, wide);
        let whole = self.x_bit(e[0], 27);
        let float = self.xs(whole, v, float);
        let class = self.x_field(e[0], 14, 3);
        let (c0, c1, c2) = (self.xc(pred::EQ, class, zero), self.xc(pred::EQ, class, one), self.xc(pred::EQ, class, two));
        let scaled = self.s.m.cast(c, Cast::Bitcast, fx, i32t);
        let x = self.xs(c2, sv, float);
        let x = self.xs(c1, scaled, x);
        let x = self.xs(c0, norm, x);
        let absent = self.x_bit(e[0], 26);
        self.xs(absent, e[3], x)
    }

    /// A load in the format \`format\` (a number: past the last, the last) from the byte
    /// address \`lo\`, \`hi\`: one value for each of \`channels\` registers. \`swizzle\`: the V#'s
    /// last word, whose low bits say for each register which component it takes, or 0 or 1.
    /// Without it register k takes component k. Also whether the format is an integer one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn x_format_load(&mut self, format: V, swizzle: Option<V>, lo: V, hi: V, read: Origin, bounds: &Bounds, channels: u32) -> (Vec<V>, V) {
        let t = self.x_tables();
        let (f, size, int, ok) = self.x_format(format, bounds);
        let (zero, one, two, k3, k4, k7) = (self.xi(0), self.xi(1), self.xi(2), self.xi(3), self.xi(4), self.xi(7));
        // A 16-bit format at an odd address reads nothing.
        let halves = self.xc(pred::EQ, size, two);
        let odd = self.x_bit(lo, 0);
        let odd = self.xb(Bin::And, halves, odd);
        // The bytes at the address, wherever it lies in a dword.
        let words = self.x_unaligned(lo, hi, read, 4);
        let row = self.xb(Bin::Shl, f, k4);
        let unit = self.xi(0x3f80_0000);
        let unit = self.xs(int, one, unit);
        let mut out = Vec::new();
        for k in 0..channels {
            let (at, sel) = match swizzle {
                Some(d3) => {
                    let by = self.xi(3 * k);
                    let sel = if k == 0 { d3 } else { self.xb(Bin::LShr, d3, by) };
                    let sel = self.xb(Bin::And, sel, k7);
                    // (4 to 7 are x to w, and 2 and 3 are z and w)
                    let slot = self.xb(Bin::And, sel, k3);
                    let slot = self.xb(Bin::Shl, slot, two);
                    (self.xb(Bin::Or, row, slot), Some(sel))
                }
                None => {
                    let slot = self.xi(4 * k);
                    (self.xb(Bin::Or, row, slot), None)
                }
            };
            let mut e = Vec::new();
            for j in 0..4 {
                let n = self.xi(j);
                let at = if j == 0 { at } else { self.xb(Bin::Or, at, n) };
                e.push(self.x_entry(t.decode, at));
            }
            let x = self.x_format_component(&words, &e);
            let mut x = self.xs(odd, zero, x);
            if let Some(sel) = sel {
                // 0 gives 0, and 1 gives 1 as the format counts.
                let first = self.xc(pred::EQ, sel, one);
                let constant = self.xs(first, unit, zero);
                let decoded = self.xc(pred::IUGE, sel, two);
                x = self.xs(decoded, x, constant);
            }
            out.push(self.xs(ok, x, zero));
        }
        (out, int)
    }

    // xsemu: from fxceless/victory-fork d4b786f (PR 927, gpu/half-views) crates/victory-gpu/src/descriptor/host.rs (half_view, half_format), GPL-3.0
    /// The unsigned format of the same layout where a d16 access's 16 bits move unchanged
    /// (`XCL_RECOMP_D16_HALF_BITS`): a 16-bit float format whose view has every channel the
    /// access names and picks them without a DST_SEL constant. The format itself otherwise.
    pub(super) fn x_d16_twin(&mut self, f: V, swizzle: Option<V>, channels: u32) -> V {
        let (seven, four, ch) = (self.xi(7), self.xi(4), self.xi(channels));
        let has = self.xb(Bin::And, f, seven);
        let mut ok = self.xc(pred::IUGE, has, ch);
        if let Some(d3) = swizzle {
            for k in 0..channels {
                let sel = self.x_field(d3, 3 * k, 7);
                let ident = self.xc(pred::IUGE, sel, four);
                ok = self.xb(Bin::And, ok, ident);
            }
        }
        let (k11, k27, k69, k13, k29, k73) = (self.xi(11), self.xi(27), self.xi(69), self.xi(13), self.xi(29), self.xi(73));
        let e13 = self.xc(pred::EQ, f, k13);
        let e29 = self.xc(pred::EQ, f, k29);
        let e73 = self.xc(pred::EQ, f, k73);
        let t = self.xs(e73, k69, f);
        let t = self.xs(e29, k27, t);
        let t = self.xs(e13, k11, t);
        self.xs(ok, t, f)
    }

    /// Loaded components `xs` as halves, two to a register and the low half first: an
    /// integer format's low 16 bits, any other as a half float. With `high` (the register's
    /// value) the one half goes above that value's low half.
    pub(super) fn x_halves(&mut self, xs: &[V], int: V, high: Option<V>) -> Vec<V> {
        let c = self.cur;
        let (i32t, low, k16) = (self.s.t.i32, self.xi(0xffff), self.xi(16));
        let halves: Vec<V> = xs
            .iter()
            .map(|&x| {
                let bits = self.xb(Bin::And, x, low);
                let f = self.as_float(x);
                let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                self.xs(int, bits, h)
            })
            .collect();
        if let Some(old) = high {
            let kept = self.xb(Bin::And, old, low);
            let up = self.xb(Bin::Shl, halves[0], k16);
            return vec![self.xb(Bin::Or, kept, up)];
        }
        halves
            .chunks(2)
            .map(|pair| match *pair {
                [lo, hi] => {
                    let up = self.xb(Bin::Shl, hi, k16);
                    self.xb(Bin::Or, lo, up)
                }
                [lo] => lo,
                _ => unreachable!("two halves at most"),
            })
            .collect()
    }

    /// The halves a store of halves takes from `regs`, widened for the format `f` (a number
    /// the tables take): an unsigned integer format's as they are, a signed one's
    /// sign-extended, any other's as a half float. `channels` halves, two to a register and
    /// the low half first, or with `high` the one high half.
    fn x_widen(&mut self, f: V, int: V, regs: &[V], channels: u32, high: bool) -> Vec<V> {
        let c = self.cur;
        let t = self.x_tables();
        let (f32t, low, k4, k16) = (self.s.t.f32, self.xi(0xffff), self.xi(4), self.xi(16));
        let row = self.xb(Bin::Shl, f, k4);
        let e0 = self.x_entry(t.decode, row);
        let signed = self.x_bit(e0, 29);
        (0..channels)
            .map(|k| {
                let r = regs[if high { 0 } else { k as usize / 2 }];
                let r = if high || k % 2 == 1 { self.xb(Bin::LShr, r, k16) } else { r };
                let h = self.xb(Bin::And, r, low);
                let up = self.xb(Bin::Shl, h, k16);
                let sext = self.xb(Bin::AShr, up, k16);
                let n = self.xs(signed, sext, h);
                let wide = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                let wide = self.as_int(wide);
                self.xs(int, n, wide)
            })
            .collect()
    }

    /// The float \`x\` (as bits) as a norm of maximum \`scale\`, rounded to the nearest even. A
    /// NaN is 0, and so is a negative number without \`signed\`.
    fn x_norm(&mut self, x: V, scale: V, signed: V) -> V {
        let c = self.cur;
        let (i64t, i32t, zero) = (self.s.m.int(64), self.s.t.i32, self.xi(0));
        self.s.flags |= flags::INT64;
        let (mags, inf, unit, yes) = (self.xi(0x7fff_ffff), self.xi(0x7f80_0000), self.xi(0x3f80_0000), self.s.m.bool(true));
        let mag = self.xb(Bin::And, x, mags);
        let neg = self.xc(pred::NE, x, mag);
        let nan = self.xc(pred::IUGT, mag, inf);
        let unsigned = self.xb(Bin::Xor, signed, yes);
        let under = self.xb(Bin::And, unsigned, neg);
        let nothing = self.xb(Bin::Or, nan, under);
        // Below 1: the mantissa times the maximum, shifted down by the exponent.
        let below = self.xc(pred::IULT, mag, unit);
        let (k23, k150, k64, k63) = (self.xi(23), self.xi(150), self.xi(64), self.xi(63));
        let exp = self.xb(Bin::LShr, mag, k23);
        let shift = self.xb(Bin::Sub, k150, exp);
        let tiny = self.xc(pred::IUGE, shift, k64);
        let tiny = self.xb(Bin::And, below, tiny);
        let nothing = self.xb(Bin::Or, nothing, tiny);
        // (a shift that is not used stays a shift the type has)
        let shift = self.xb(Bin::And, shift, k63);
        let (frac, top) = (self.xi(0x7f_ffff), self.xi(0x80_0000));
        let man = self.xb(Bin::And, mag, frac);
        let man = self.xb(Bin::Or, man, top);
        let (man, wide) = (self.s.m.cast(c, Cast::ZExt, man, i64t), self.s.m.cast(c, Cast::ZExt, scale, i64t));
        let prod = self.xb(Bin::Mul, man, wide);
        let by = self.s.m.cast(c, Cast::ZExt, shift, i64t);
        let q = self.xb(Bin::LShr, prod, by);
        let q = self.s.m.cast(c, Cast::Trunc, q, i32t);
        let one64 = self.s.m.int_const(i64t, 1);
        let bit = self.xb(Bin::Shl, one64, by);
        let low = self.xb(Bin::Sub, bit, one64);
        let rem = self.xb(Bin::And, prod, low);
        let half = self.xb(Bin::LShr, bit, one64);
        let (more, tie) = (self.xc(pred::IUGT, rem, half), self.xc(pred::EQ, rem, half));
        let odd = self.x_bit(q, 0);
        let tie = self.xb(Bin::And, tie, odd);
        let up = self.xb(Bin::Or, more, tie);
        let up = self.s.m.cast(c, Cast::ZExt, up, i32t);
        let q = self.xb(Bin::Add, q, up);
        let q = self.xs(below, q, scale);
        let minus = self.xb(Bin::Sub, zero, q);
        let q = self.xs(neg, minus, q);
        self.xs(nothing, zero, q)
    }

    /// The register value \`x\` as a component, by the entry's first word and its norm
    /// maximum ([\`encode_entry\`]). Not yet masked to the width.
    fn x_format_encode(&mut self, x: V, e0: V, scale: V) -> V {
        let c = self.cur;
        let (i32t, zero) = (self.s.t.i32, self.xi(0));
        let kind = self.x_field(e0, 8, 15);
        let is = |l: &mut Self, k: u32| {
            let k = l.xi(k);
            l.xc(pred::EQ, kind, k)
        };
        let (snorm, unorm) = (is(self, SNORM), is(self, UNORM));
        let norm = self.x_norm(x, scale, snorm);
        let f = self.as_float(x);
        let (us, ss) = (self.s.m.cast(c, Cast::FpToUi, f, i32t), self.s.m.cast(c, Cast::FpToSi, f, i32t));
        let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        // 11 and 10 bits: no sign, a NaN is the one NaN, and the half's low bits are cut.
        let eleven = is(self, FLOAT11);
        let (h11, h10) = (self.x_field(h, 4, 0x7ff), self.x_field(h, 5, 0x3ff));
        let small = self.xs(eleven, h11, h10);
        let f0 = self.s.m.f32(0);
        let low = self.xc(pred::OLE, f, f0);
        let small = self.xs(low, zero, small);
        let (n11, n10) = (self.xi(0x7e0), self.xi(0x3f0));
        let nan = self.xs(eleven, n11, n10);
        let unordered = self.xc(pred::UNO, f, f);
        let small = self.xs(unordered, nan, small);
        let (half, bits, sscaled, uscaled) = (is(self, HALF), is(self, BITS), is(self, SSCALED), is(self, USCALED));
        let v = self.xs(half, h, small);
        let v = self.xs(bits, x, v);
        let v = self.xs(sscaled, ss, v);
        let v = self.xs(uscaled, us, v);
        let normed = self.xb(Bin::Or, snorm, unorm);
        self.xs(normed, norm, v)
    }

    /// A store of the register values \`vals\` in the format \`format\` at the byte address \`lo\`,
    /// \`hi\`, in the lanes of \`guard\`. Components the format lacks are dropped. `halves`:
    /// the registers hold so many halves, or the one high half ([`Lower::x_widen`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn x_format_store(&mut self, format: V, lo: V, hi: V, base_hi: V, bounds: &Bounds, vals: &[V], halves: Option<(u32, bool)>, guard: Option<V>) {
        let t = self.x_tables();
        let (f, size, int, ok) = self.x_format(format, bounds);
        let wide;
        let vals = match halves {
            Some((channels, high)) => {
                wide = self.x_widen(f, int, vals, channels, high);
                &wide[..]
            }
            None => vals,
        };
        let lanes = self.x_and(guard, Some(ok));
        let (zero, k12) = (self.xi(0), self.xi(12));
        let row = self.xb(Bin::Mul, f, k12);
        let mut words = [zero; 4];
        for (c, &x) in vals.iter().enumerate().take(4) {
            let mut e = Vec::new();
            for j in 0..3 {
                let n = self.xi(3 * c as u32 + j);
                let at = self.xb(Bin::Add, row, n);
                e.push(self.x_entry(t.encode, at));
            }
            let v = self.x_format_encode(x, e[0], e[2]);
            let v = self.xb(Bin::And, v, e[1]);
            let by = self.x_field(e[0], 0, 31);
            let placed = self.xb(Bin::Shl, v, by);
            let dword = self.x_field(e[0], 5, 3);
            for (d, word) in words.iter_mut().enumerate() {
                let n = self.xi(d as u32);
                let here = self.xc(pred::EQ, dword, n);
                let part = self.xs(here, placed, zero);
                *word = if *word == zero { part } else { self.xb(Bin::Or, *word, part) };
            }
        }
        self.x_bytes_store(lo, hi, base_hi, &words, size, lanes);
    }

    /// Stores the first \`size\` bytes (at most 16) of the dwords \`r\` at the byte address \`lo\`,
    /// \`hi\`, in the lanes of \`lanes\`. Whole dwords on a dword are plain stores. Anything
    /// else replaces the bytes inside each dword it touches by two atomics, which leave the
    /// dword's other bytes alone.
    // (xsemu's translator writes a dword that takes bytes of two registers by two such
    // pairs. Here it is one pair with both.)
    fn x_bytes_store(&mut self, lo: V, hi: V, base_hi: V, r: &[V; 4], size: V, lanes: Option<V>) {
        let (zero, three, all, k31, k32, one) = (self.xi(0), self.xi(3), self.xi(u32::MAX), self.xi(31), self.xi(32), self.xi(1));
        let within = self.xb(Bin::And, lo, three);
        let aligned = self.xb(Bin::Sub, lo, within);
        let on = self.xc(pred::EQ, within, zero);
        let rest = self.xb(Bin::And, size, three);
        let whole = self.xc(pred::EQ, rest, zero);
        let plain = self.xb(Bin::And, on, whole);
        // (dword k has bytes of the value)
        let mut used = Vec::new();
        for k in 0..4 {
            let at = self.xi(4 * k);
            used.push(self.xc(pred::IULT, at, size));
        }
        let oks: Vec<Option<V>> = used.iter().map(|&u| Some(u)).collect();
        let go = self.x_and(lanes, Some(plain));
        let at = self.begin_guarded(go);
        self.x_store(aligned, hi, Origin::Write(base_hi), None, r, &oks, None);
        self.end_guarded(at, &[]);
        // In parts: dword k of the value covers bits of the dword it starts in and, off a
        // dword, of the next one.
        let up = self.xb(Bin::Shl, within, three);
        let down = self.xb(Bin::Sub, k32, up);
        let down = self.xb(Bin::And, down, k31);
        let bits = self.xb(Bin::Shl, size, three);
        let (mut masks, mut sets) = (Vec::new(), Vec::new());
        let mut over: Option<(V, V)> = None;
        for k in 0..5usize {
            let mut mask = None;
            if k < 4 {
                // (the value's bits in this dword: 32, or what is left of the size)
                let before = self.xi(32 * k as u32);
                let left = self.xb(Bin::Sub, bits, before);
                let full = self.xc(pred::IUGE, left, k32);
                let left = self.xb(Bin::And, left, k31);
                let part = self.xb(Bin::Shl, one, left);
                let part = self.xb(Bin::Sub, part, one);
                let m = self.xs(full, all, part);
                let m = self.xs(used[k], m, zero);
                let v = self.xb(Bin::And, r[k], m);
                mask = Some((m, v));
            }
            let mut here = mask.map(|(m, v)| (self.xb(Bin::Shl, m, up), self.xb(Bin::Shl, v, up)));
            if let Some((m, v)) = over {
                here = Some(match here {
                    Some((m2, v2)) => (self.xb(Bin::Or, m, m2), self.xb(Bin::Or, v, v2)),
                    None => (m, v),
                });
            }
            let (m, v) = here.expect("a part");
            masks.push(m);
            sets.push(v);
            over = mask.map(|(m, v)| {
                let (m, v) = (self.xb(Bin::LShr, m, down), self.xb(Bin::LShr, v, down));
                (self.xs(on, zero, m), self.xs(on, zero, v))
            });
        }
        let keeps: Vec<V> = masks.iter().map(|&m| self.xb(Bin::Xor, m, all)).collect();
        let oks: Vec<Option<V>> = masks.iter().map(|&m| Some(self.xc(pred::NE, m, zero))).collect();
        let yes = self.s.m.bool(true);
        let parts = self.xb(Bin::Xor, plain, yes);
        let go = self.x_and(lanes, Some(parts));
        let at = self.begin_guarded(go);
        self.x_store(aligned, hi, Origin::Write(base_hi), None, &sets, &oks, Some(&keeps));
        self.end_guarded(at, &[]);
    }
}
