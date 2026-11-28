//! P10: the 16 bits of a d16 typed-buffer access move as they are (`XCL_RECOMP_D16_HALF_BITS`).
//!
//! A console d16 access takes each channel's 16 bits as they are: an integer format's value,
//! or a half float's pattern. The translator already decodes a 16-bit integer format to its
//! bits; a 16-bit *float* format still went through f32 and the f16 conversion, which keeps
//! every finite value and infinity and changes signal NaNs. Upstream fxceless/victory-fork PR
//! 927 moves all 65,536 patterns as integers for the same reason (`emit/halves.rs`,
//! `places_read_as_halves`); the face network keeps its halves in 16-bit views.
//!
//!
//! xsemu: from fxceless/victory-fork d4b786f (PR 927, `gpu/half-views`)
//! `crates/victory-shader/src/emit/halves.rs`,
//! `crates/victory-gpu/src/descriptor/host.rs` (`half_view`), GPL-3.0. Behaviour, not code.
//!
//! These tests run the translated program in `xcl-dxil`'s interpreter and compare all 65,536
//! patterns through both directions. For each of 16 chunks, a lane stores its pattern to title
//! memory with a raw `buffer_store_dword`, loads it back with `buffer_load_format_d16_x`
//! through a V# and stores the loaded register to a results region; then it stores the
//! pattern with `buffer_store_format_d16_x` through the same kind of V#, loads it back raw and
//! stores that. Lane offsets and the 16 patterns come from `io`'s VGPRs; the V#'s format is
//! the only difference between the cases.

use super::emit::D16_HALF_BITS;
use super::{translate, Options, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::rdna2::encode;
use xcl_dxil::interp::{self, res, Config, Input};

const IN: u32 = 0x1000;
const IN2: u32 = 0x1800;
const OUT: u32 = 0x2000;
const CHUNKS: u32 = 16;
const LANES: u32 = 32;
const STRIDE: u32 = 128;

/// GFX10 unified buffer formats: `BUF_FORMATS[11]` = 16_UINT, `[12]` = 16_SINT,
/// `[13]` = 16_FLOAT
const FMT_16_UINT: u32 = 11;
const FMT_16_SINT: u32 = 12;
const FMT_16_FLOAT: u32 = 13;
/// identity DST_SEL: SEL_X..SEL_W = 4..7
const IDENT: u32 = 4 | 5 << 3 | 6 << 6 | 7 << 9;

type Inst = (&'static str, &'static str, Vec<(&'static str, u64)>);

/// One MUBUF instruction at `off` through the descriptor quad `srsrc` (1 and 4 hold the d16
/// V#, 2, 3 and 5 the raw ones).
fn op(name: &'static str, srsrc: u64, data: Option<u64>, off: u64) -> Inst {
    let mut f = vec![("VADDR", 0), ("SRSRC", srsrc), ("SOFFSET", 128), ("OFFSET", off), ("OFFEN", 1)];
    if let Some(d) = data {
        f.insert(0, ("VDATA", d));
    }
    (name, "ENC_MUBUF", f)
}

/// For each of 16 chunks: the pattern to `IN`, a d16 load of it to `OUT`, a d16 store of it to
/// `IN2`, and a raw load of that to `OUT` (`CHUNKS +` the chunk).
fn program() -> Vec<u8> {
    let mut ops: Vec<Inst> = Vec::new();
    for k in 0..CHUNKS as u64 {
        let off = k * STRIDE as u64;
        let back = (CHUNKS as u64 + k) * STRIDE as u64;
        ops.push(op("BUFFER_STORE_DWORD", 2, Some(1 + k), off));
        ops.push(op("BUFFER_LOAD_FORMAT_D16_X", 1, Some(17), off));
        ops.push(op("BUFFER_STORE_DWORD", 3, Some(17), off));
        ops.push(op("BUFFER_STORE_FORMAT_D16_X", 4, Some(1 + k), off));
        ops.push(op("BUFFER_LOAD_DWORD", 5, Some(17), off));
        ops.push(op("BUFFER_STORE_DWORD", 3, Some(17), back));
    }
    ops.push(("S_ENDPGM", "ENC_SOPP", vec![]));
    ops.iter().flat_map(|(n, e, f)| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

/// The `io` words a run patches: five V#s (quad 1 and 4 hold `format`), the lanes' offsets and
/// the 16 patterns a lane holds in v1..v16.
fn patches(format: u32, patterns: &[u32]) -> Vec<(u64, u32)> {
    let word = |w: u64, v: u32| ((res::IDENT_IO as u64) << 32 | w, v);
    let vsharp = |q: u64, base: u32, format: u32| [word(q * 4, base), word(q * 4 + 1, 0), word(q * 4 + 2, 0x1000), word(q * 4 + 3, format << 12 | IDENT)];
    let mut p = vec![word(126, u32::MAX), word(127, u32::MAX)];
    p.extend(vsharp(1, IN, format));
    p.extend(vsharp(2, IN, 0));
    p.extend(vsharp(3, OUT, 0));
    p.extend(vsharp(4, IN2, format));
    p.extend(vsharp(5, IN2, 0));
    for lane in 0..LANES as u64 {
        let base = KERNEL_SGPRS as u64 + lane * KERNEL_VGPRS as u64;
        p.push(word(base, 4 * lane as u32));
        for k in 0..CHUNKS as u64 {
            p.push(word(base + 1 + k, patterns[(k * LANES as u64 + lane) as usize]));
        }
    }
    p
}

fn compiled(halves: bool) -> interp::Program {
    D16_HALF_BITS.with(|b| b.set(halves));
    let t = translate(&program(), &Options { wave64: false, mem_windows: 1, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = xcl_dxil::compile(&t.hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    interp::load(&dxil).unwrap()
}

/// The words the kernel stored to the results region, by `chunk * 32 + lane` (`CHUNKS +` the
/// chunk is the d16 store's readback).
fn results(run: &interp::Run) -> Vec<u32> {
    let mut out = vec![0u32; (2 * CHUNKS * LANES) as usize];
    for &(kind, _, addr, v) in &run.written {
        let Some(off) = addr.checked_sub(OUT as u64).filter(|o| *o < (2 * CHUNKS * STRIDE) as u64) else {
            continue;
        };
        let (k, lane) = ((off / STRIDE as u64) as usize, (off % STRIDE as u64 / 4) as usize);
        if kind == 0 && lane < LANES as usize {
            out[k * LANES as usize + lane] = v[0];
        }
    }
    out
}

/// How many of the 65,536 patterns a d16 load (left) and a d16 store (right) of `format` do not
/// move unchanged.
fn changed(halves: bool, format: u32) -> (usize, usize) {
    let p = compiled(halves);
    let (mut load, mut store) = (0, 0);
    for first in (0..65536u32).step_by((CHUNKS * LANES) as usize) {
        let patterns: Vec<u32> = (0..CHUNKS * LANES).map(|i| first + i).collect();
        let (run, _) = interp::run_input(&p, &Input { seed: 0, patches: patches(format, &patterns) }, &Config { force_ballots: false, ..Default::default() }, false);
        assert!(run.stop.is_none(), "{:?}", run.stop);
        let got = results(&run);
        for (i, &pat) in patterns.iter().enumerate() {
            load += usize::from(got[i] & 0xFFFF != pat);
            store += usize::from(got[CHUNKS as usize * LANES as usize + i] & 0xFFFF != pat);
        }
    }
    (load, store)
}

/// The switch is what the sweep measures: it changes the translated program.
fn hlsl(halves: bool) -> String {
    D16_HALF_BITS.with(|b| b.set(halves));
    let t = translate(&program(), &Options { wave64: false, mem_windows: 1, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

/// Every 16-bit pattern of a 16-bit float view keeps its bits; through the conversion the
/// 1,022 signal-NaN patterns (both signs, mantissa below the quiet bit) come back quieted.
#[test]
fn a_d16_access_of_a_16_bit_float_view_keeps_every_pattern() {
    assert_ne!(hlsl(false), hlsl(true), "the switch changes the program");
    assert_eq!(changed(false, FMT_16_FLOAT), (1022, 1022), "signal NaNs quieted by the f16 conversion");
    assert_eq!(changed(true, FMT_16_FLOAT), (0, 0));
    assert_eq!(changed(true, FMT_16_UINT), (0, 0));
    // an integer format's bits were already the console's, with and without the switch
    assert_eq!(changed(false, FMT_16_SINT), (0, 0));
    assert_eq!(changed(true, FMT_16_SINT), (0, 0));
}
