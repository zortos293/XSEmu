// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../../LICENSE and ../../../PROVENANCE.md), so it carries that licence.
//! Tests of the XCL binding on hand-assembled code. Nothing here runs DXIL: they check what
//! was emitted, by the instruction census and the binding's own counts.

use crate::emit::bitcode::census;
use crate::emit::lower::{Lowered, Parts, lower};
use crate::emit::{io, structure};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::xcl::{BindingScope, XclConfig, XclStats};
use crate::{ShaderError, ir, resource};

const S_ENDPGM: u32 = 0xBF81_0000;
/// ds_write_b32 v0, v1: gives v1 an effect.
const KEEP_V1: [u32; 2] = [0xD834_0000, 0x0000_0100];

/// v_mov_b32 v1, s`n`.
const fn v1_from(n: u32) -> u32 {
    0x7E02_0200 | n
}

/// s_mov_b32 s`n`, a 32-bit literal.
fn s_mov(n: u32, value: u32) -> [u32; 2] {
    [0xBE80_03FF | n << 16, value]
}

struct Emitted {
    lowered: Lowered,
    stats: XclStats,
    counts: census::Census,
}

impl Emitted {
    fn count(&self, kind: &str) -> u32 {
        self.counts.counts[census::KINDS.iter().position(|k| *k == kind).expect("a kind")]
    }
}

/// Lowers a one-thread wave32 compute program with 12 user SGPRs and 512 bytes of LDS under
/// the binding.
fn emit_with(words: &[u32], cfg: XclConfig) -> Result<Emitted, ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 15 | 12 << 1, threads: [1, 1, 1] }),
        code,
    };
    let _scope = BindingScope::enter(cfg);
    crate::xcl::clear_stats();
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })?;
    let (stats, counts) = crate::xcl::take_stats(&lowered.dxil);
    assert_ne!(stats.instructions, 0, "the lowering's counts");
    Ok(Emitted { lowered, stats, counts })
}

fn emit(words: &[u32]) -> Emitted {
    emit_with(words, XclConfig::default()).unwrap()
}

/// Without the thread limits and without memory: only what the program itself needs.
fn bare() -> XclConfig {
    XclConfig { thread_limits: false, ..XclConfig::default() }
}

#[test]
fn an_entry_sgpr_is_its_io_dword() {
    let e = emit_with(&[v1_from(3), KEEP_V1[0], KEEP_V1[1], S_ENDPGM], bare()).unwrap();
    // One load, of io, through the one handle there is: u0, and no heap.
    assert_eq!(e.count("buffer_load"), 1);
    assert_eq!(e.count("handle"), 1);
    assert_eq!(e.lowered.root_uavs, vec![0]);
    assert!(e.lowered.user_data.is_empty() && e.lowered.root_cbvs.is_empty() && e.lowered.root_srvs.is_empty());
    assert_eq!(e.stats.resolves_single + e.stats.resolves_multi, 0);
}

/// s_or_saveexec_b32 s0, -1: EXEC to every lane, the old EXEC kept in s0.
const WIDEN: u32 = 0xBE80_3DC1;

#[test]
fn the_lanes_past_the_thread_limits_run_with_exec_clear() {
    let plain = [v1_from(3), KEEP_V1[0], KEEP_V1[1], S_ENDPGM];
    let with = emit(&plain);
    let without = emit_with(&plain, bare()).unwrap();
    // One more load, of the limits. No lane returns: the store is behind the test (a block
    // and its join).
    assert_eq!(with.count("buffer_load"), without.count("buffer_load") + 1);
    assert_eq!(with.count("blocks"), without.count("blocks") + 2);
    // A program that sets EXEC to every lane first runs in every lane, as on the console.
    let wide = emit(&[WIDEN, v1_from(3), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((wide.count("guard_select"), wide.count("blocks")), (without.count("guard_select"), without.count("blocks")));
}

#[test]
fn a_mask_saved_before_exec_is_widened_holds_the_lanes_inside_the_limits() {
    use crate::ir::Reg;
    use crate::structure::{Atom, Bool};
    // s_or_saveexec_b32 s0, -1, then v_mov_b32 v1, s0: the saved mask as data.
    let words = [WIDEN, v1_from(0), KEEP_V1[0], KEEP_V1[1], S_ENDPGM];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for (cfg, limits) in [(XclConfig::default(), true), (bare(), false)] {
        let _scope = BindingScope::enter(cfg);
        let p = ir::build(&code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let saved = st.masks[&p.blocks[0].insts[0].write(Reg::S(0)).unwrap()];
        if limits {
            assert_eq!(st.bools.support(saved), [Atom::Launch]);
        } else {
            assert_eq!(saved, Bool::TRUE);
        }
        // Past the widening every lane runs: nothing is guarded.
        assert!(st.guards[0].iter().all(Option::is_none), "{:?}", st.guards);
    }
    // It translates: the lowering has the launch mask's value.
    emit(&words);
}

#[test]
fn a_scalar_load_reads_title_memory_at_its_registers() {
    // s_load_dword s8, s[4:5], 0x10
    let e = emit(&[0xF400_0202, 0xFA00_0010, v1_from(8), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((e.stats.scalar_loads, e.stats.record_loads), (1, 0));
    assert_eq!((e.stats.resolves_single, e.stats.resolves_multi), (1, 0));
    // Heap handles: the locator's view, the directory's root, both leaves (read in one
    // block, no loop), the data view.
    assert_eq!(e.count("handle"), 1 + 5);
    // Loads: the limits, the memory words, the locator's view, s4 and s5, the root, both
    // leaves, the data.
    assert_eq!(e.count("buffer_load"), 3 + 2 + 3 + 1);
    assert_eq!(e.lowered.root_uavs, vec![0]);
}

#[test]
fn a_record_load_resolves_the_page_and_the_next_with_no_loop() {
    // s_load_dwordx4 s[8:11], s[4:5], 0
    let e = emit(&[0xF408_0202, 0xFA00_0000, v1_from(11), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((e.stats.scalar_loads, e.stats.record_loads), (0, 1));
    // The page of the first dword and the page after it, resolved one after the other: no
    // loop, no phi, no block of its own.
    assert_eq!((e.stats.resolves_single, e.stats.resolves_multi), (0, 2));
    assert_eq!(e.count("handle"), 1 + 1 + 1 + 2 * 2 + 2);
    let single = emit(&[0xF400_0202, 0xFA00_0000, v1_from(8), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!(e.count("blocks"), single.count("blocks"));
    assert_eq!(e.count("phi"), single.count("phi"));
    // A read of title memory for each dword past the first, from each page, and a select.
    assert!(e.count("buffer_load") > single.count("buffer_load"));
    assert!(e.count("select") > single.count("select"));
}

#[test]
fn a_scalar_buffer_load_tests_each_dword_against_the_buffer() {
    // s_buffer_load_dword s8, s[4:7], 0x10 and s_buffer_load_dwordx4 s[8:11], s[4:7], 0x10
    let one = emit(&[0xF420_0202, 0xFA00_0010, v1_from(8), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((one.stats.buffer_accesses, one.stats.resolves_single), (1, 1));
    let four = emit(&[
        0xF428_0202,
        0xFA00_0010,
        v1_from(8),
        KEEP_V1[0],
        KEEP_V1[1],
        v1_from(11),
        KEEP_V1[0],
        KEEP_V1[1],
        S_ENDPGM,
    ]);
    assert_eq!((four.stats.buffer_accesses, four.stats.resolves_multi), (1, 2));
    // A dword out of the buffer is 0: one more select for each dword that is kept.
    let plain = emit(&[0xF400_0202, 0xFA00_0010, v1_from(8), KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert!(one.count("select") > plain.count("select"));
    // The V#'s first three words are read.
    assert_eq!(one.count("buffer_load"), plain.count("buffer_load") + 1);
}

/// buffer_load_dword v1, v0, s[4:7], 0 and buffer_store_dword v1, v0, s[4:7], 0, each
/// with idxen if `index`.
fn load_and_store(index: bool) -> Vec<u32> {
    let idxen = if index { 0x2000 } else { 0 };
    vec![0xE030_0000 | idxen, 0x8001_0100, 0xE070_0000 | idxen, 0x8001_0100, S_ENDPGM]
}

#[test]
fn buffer_loads_and_stores_go_through_the_resolver() {
    for index in [false, true] {
        let e = emit(&load_and_store(index));
        assert_eq!(e.stats.buffer_accesses, 2, "index {index}");
        assert_eq!((e.stats.resolves_single, e.stats.resolves_multi), (2, 0));
        assert_eq!(e.count("buffer_store"), 1);
        // Every view of title memory is taken from the heap by a computed index: io, the
        // locator's view and the root once each, and a data view and both leaves for each
        // resolver.
        assert_eq!(e.count("handle"), 1 + 2 + 2 * 3);
        assert_eq!(e.lowered.root_uavs, vec![0]);
    }
    // With an index the address is a 64-bit product and the index is tested.
    let (plain, indexed) = (emit(&load_and_store(false)), emit(&load_and_store(true)));
    assert!(indexed.count("total") > plain.count("total"));
}

#[test]
fn a_store_without_a_high_dword_lands_in_the_null_view() {
    // The store's view is chosen among the window, the directory's and the null view by
    // selects, never by leaving the store out: one store, no branch around it.
    let e = emit(&load_and_store(false));
    let load_only = emit(&[0xE030_0000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!(e.count("buffer_store"), 1);
    assert_eq!(load_only.count("buffer_store"), 0);
    // (the store's resolver walks the directory in its own block, as the load's does: the
    // walk is not behind a branch, so neither adds a block)
    assert_eq!(e.count("blocks"), load_only.count("blocks"));
}

#[test]
fn a_buffer_atomic_resolves_as_a_store() {
    // buffer_atomic_add v1, v0, s[4:7], 0 idxen
    let e = emit(&[0xE0C8_2000, 0x8001_0100, S_ENDPGM]);
    assert_eq!(e.count("atomic"), 1);
    assert_eq!((e.stats.buffer_accesses, e.stats.resolves_single), (1, 1));
}

#[test]
fn a_load_in_the_descriptors_format_is_decoded_through_the_tables() {
    // buffer_load_format_x v1, v0, s[4:7], 0 idxen, and the same with xyzw
    let x = emit(&[0xE000_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!(x.stats.buffer_accesses, 1);
    // The bytes may start anywhere in a dword: five dwords through two resolvers, the
    // page of the first dword and the page after it.
    assert_eq!((x.stats.resolves_single, x.stats.resolves_multi), (0, 2));
    // What the format is, and the four words of the one component that is kept.
    assert_eq!(x.count("lds_load"), 1 + 4);
    let xyzw = emit(&[0xE00C_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!(xyzw.count("lds_load"), 1 + 4);
    let all: Vec<u32> = [0xE00C_2000, 0x8001_0100].into_iter().chain((1..5).flat_map(|v| [0xD834_0000, v << 8])).chain([S_ENDPGM]).collect();
    assert_eq!(emit(&all).count("lds_load"), 1 + 4 * 4);
    // No more memory is read for four components than for one.
    assert_eq!(emit(&all).count("buffer_load"), x.count("buffer_load"));
}

#[test]
fn a_store_in_the_descriptors_format_is_whole_dwords_or_atomics() {
    // v_mov_b32 v1 to v4 from s8 to s11, and buffer_store_format_xyzw v[1:4], v0, s[4:7], 0
    // idxen
    let values = (1..5).map(|v| 0x7E00_0200 | v << 17 | (7 + v));
    let e = emit(&values.chain([0xE01C_2000, 0x8001_0100, S_ENDPGM]).collect::<Vec<u32>>());
    assert_eq!(e.stats.buffer_accesses, 1);
    // Whole dwords on a dword are stores. Anything else is five dwords, each by an and
    // and an or that leave its other bytes alone.
    assert_eq!((e.count("buffer_store"), e.count("atomic")), (4, 2 * 5));
    assert_eq!((e.stats.resolves_single, e.stats.resolves_multi), (0, 2));
    // What the format is, and three words for each of four components.
    assert_eq!(e.count("lds_load"), 1 + 3 * 4);
}

/// ds_write_b32 v0, v2: gives v2 an effect.
const KEEP_V2: [u32; 2] = [0xD834_0000, 0x0000_0200];

#[test]
fn a_load_of_halves_in_the_descriptors_format_packs_the_components() {
    let math = |e: &Emitted| e.count("math_call");
    // buffer_load_format_x and buffer_load_format_d16_x v1, v0, s[4:7], 0 idxen
    let x = emit(&[0xE000_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    let d16 = emit(&[0xE200_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    // The same decode, and the component made a half (an integer format keeps its low bits).
    assert_eq!(d16.count("lds_load"), x.count("lds_load"));
    assert_eq!(math(&d16), math(&x) + 1);
    assert_eq!((d16.stats.buffer_accesses, d16.stats.resolves_multi), (1, 2));
    // buffer_load_format_d16_xyzw v[1:2]: four components in two registers, against
    // buffer_load_format_xyzw v[1:4] with all four kept.
    let xyzw = emit(&[0xE20C_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], KEEP_V2[0], KEEP_V2[1], S_ENDPGM]);
    let all: Vec<u32> = [0xE00C_2000, 0x8001_0100].into_iter().chain((1..5).flat_map(|v| [0xD834_0000, v << 8])).chain([S_ENDPGM]).collect();
    let all = emit(&all);
    assert_eq!(xyzw.count("lds_load"), 1 + 4 * 4);
    assert_eq!(math(&xyzw), math(&all) + 4);
    // buffer_load_format_d16_hi_x v1: one component, above the register's low half.
    let hi = emit(&[0xE098_2000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((hi.count("lds_load"), math(&hi)), (d16.count("lds_load"), math(&d16)));
}

#[test]
fn a_store_of_halves_in_the_descriptors_format_widens_them_first() {
    // v_mov_b32 v1 and v2 from s8 and s9, and buffer_store_format_d16_xyzw v[1:2], v0,
    // s[4:7], 0 idxen
    let values = (1..3).map(|v| 0x7E00_0200 | v << 17 | (7 + v));
    let e = emit(&values.chain([0xE21C_2000, 0x8001_0100, S_ENDPGM]).collect::<Vec<u32>>());
    assert_eq!(e.stats.buffer_accesses, 1);
    // The bytes go as for any format store.
    assert_eq!((e.count("buffer_store"), e.count("atomic")), (4, 2 * 5));
    // What the format is, whether it is signed, and three words for each of four components.
    assert_eq!(e.count("lds_load"), 1 + 1 + 3 * 4);
    // buffer_store_format_d16_hi_x v1 and tbuffer_store_format_d16_x v1 (format 56).
    let hi = emit(&[0x7E02_0208, 0xE09C_2000, 0x8001_0100, S_ENDPGM]);
    assert_eq!(hi.count("lds_load"), 1 + 1 + 3);
    let t = emit(&[0x7E02_0208, 0xE9C4_2000, 0x8021_0100, S_ENDPGM]);
    assert_eq!(t.count("lds_load"), 1 + 1 + 3);
}

/// global_atomic_`op` v1, v[2:3], v4, off, glc, and v1 kept.
fn global_atomic(op: u32) -> Vec<u32> {
    vec![0xDC01_8000 | op << 18, 0x017D_0402, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]
}

#[test]
fn a_float_atomic_is_a_loop_of_compare_exchanges() {
    let add = emit(&global_atomic(0x32));
    // fmax: the dword read once, then exchanged in a loop, one compare-exchange in it.
    let fmax = emit(&global_atomic(0x40));
    assert_eq!(fmax.count("atomic"), 1);
    assert_eq!(fmax.count("buffer_load"), add.count("buffer_load") + 1);
    assert_eq!(fmax.count("blocks"), add.count("blocks") + 2);
    assert_eq!(fmax.count("phi"), add.count("phi") + 2);
    assert_eq!(fmax.stats.global_accesses, 1);
    // No float arithmetic: the order is taken on the bits.
    assert_eq!(fmax.count("math_call"), add.count("math_call"));
    // fmin is the same loop.
    let fmin = emit(&global_atomic(0x3f));
    assert_eq!((fmin.count("atomic"), fmin.count("blocks"), fmin.count("phi")), (1, fmax.count("blocks"), fmax.count("phi")));
    // buffer_atomic_fmax v1, v0, s[4:7], 0 idxen glc
    let buffer = emit(&[0xE100_6000, 0x8001_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    assert_eq!((buffer.count("atomic"), buffer.stats.buffer_accesses), (1, 1));
}

/// image_store v[1..], v[0:1], s[8:15] of a 2D image with `dmask`, halves if `d16`.
const fn image_store(dmask: u32, d16: bool) -> [u32; 2] {
    [0xF020_0008 | dmask << 8, 0x0002_0100 | (d16 as u32) << 31]
}

#[test]
fn an_image_store_of_some_channels_writes_the_others_as_0() {
    for format in [UINT, SINT, FLOAT] {
        let (all, all_ops) = typed(&image(image_store(0xF, false), Some(format)));
        // The same one store as of every channel, through the view of the T#'s class.
        for (dmask, d16) in [(0x1, false), (0x5, false), (0x7, true), (0x2, true)] {
            let (e, ops) = typed(&image(image_store(dmask, d16), Some(format)));
            assert_eq!((ops, e.stats.image_ops), (all_ops.clone(), 1), "format {format} dmask {dmask:#x}");
            // A float view widens the halves of the mask, an integer view keeps their bits.
            let widened = if d16 && format == FLOAT { dmask.count_ones() } else { 0 };
            assert_eq!(e.count("math_call"), all.count("math_call") + widened, "format {format} dmask {dmask:#x}");
        }
    }
    // A class the program reads: the three ways, as for every channel.
    let (_, ops) = typed(&image(image_store(0x3, false), None));
    assert_eq!(ops, typed(&image(image_store(0xF, false), None)).1);
}

#[test]
fn the_formats_are_cgbjs_and_the_notes() {
    use super::formats::{decode_entry, encode_entry, shape};
    // Against the table CGBJ keeps for the formats an instruction names.
    for f in 1..=77u32 {
        let (bits, numeric) = crate::emit::lower::buffer_format(f as u8).expect("a format");
        assert_eq!(shape(f), (bits.to_vec(), numeric as u32), "format {f}");
    }
    assert!(crate::emit::lower::buffer_format(78).is_none());
    // No format, and a number past the last.
    assert_eq!(shape(0), (vec![], 0));
    assert_eq!(shape(78), shape(77));
    assert_eq!(shape(127), (vec![32; 4], 6));
    // 8_8_8_8_UNORM: z is bits 23:16 over 255.
    assert_eq!(decode_entry(56, 2), [16 | 8 << 8, 255.0f32.to_bits(), 0xff, 0]);
    // 16_SNORM: sign-extended by 16, over 32767. Its w is 1.0.
    assert_eq!(decode_entry(8, 0), [16 << 8 | 16 << 16 | 1 << 29, 32767.0f32.to_bits(), 0xffff, 0]);
    assert_eq!(decode_entry(8, 3)[0] >> 26 & 1, 1);
    assert_eq!(decode_entry(8, 3)[3], 0x3f80_0000);
    assert_eq!(decode_entry(8, 1)[3], 0);
    // 8_8_UINT: an integer, and its w is 1.
    assert_eq!(decode_entry(18, 1)[0], 8 | 8 << 8 | 2 << 14);
    assert_eq!(decode_entry(18, 3)[3], 1);
    // 10_11_11_FLOAT: x and y are 11 bits, z is 10 bits from bit 22, made halves by 4 and 5.
    assert_eq!(decode_entry(36, 0)[0], 11 << 8 | 3 << 14 | 4 << 21 | 1 << 28);
    assert_eq!(decode_entry(36, 2)[0], 22 | 10 << 8 | 3 << 14 | 5 << 21 | 1 << 28);
    // 16_16_FLOAT and 32_32_FLOAT: a half, and the bits as they are.
    assert_eq!(decode_entry(29, 1)[0], 16 | 16 << 8 | 3 << 14);
    assert_eq!(decode_entry(64, 1)[0], 32 | 32 << 8 | 3 << 14 | 1 << 27);
    // Encoding: where the component goes, how, its mask and its norm maximum.
    assert_eq!(encode_entry(56, 1), [8 | 1 << 8, 0xff, 0xff]);
    assert_eq!(encode_entry(57, 3), [24 | 2 << 8, 0xff, 0x7f]);
    assert_eq!(encode_entry(36, 0), [7 << 8, 0x7ff, 0x7ff]);
    assert_eq!(encode_entry(36, 2), [22 | 8 << 8, 0x3ff, 0x3ff]);
    assert_eq!(encode_entry(71, 2), [32 | 6 << 8, 0xffff, 0xffff]);
    assert_eq!(encode_entry(77, 3), [96 | 5 << 8, u32::MAX, u32::MAX]);
    // 10_10_10_2 has its 2 bits lowest, 2_10_10_10 highest.
    assert_eq!(encode_entry(48, 0), [5 << 8, 3, 3]);
    assert_eq!(encode_entry(54, 3), [30 | 5 << 8, 3, 3]);
    assert_eq!(encode_entry(14, 2), [0; 3]);
}

/// A scalar load of 1, 2, 4 or 8 dwords (`log` 0 to 3) into s`data` on, from s[4:5] plus
/// `offset`, and its first dword kept.
fn load(log: u32, data: u32, offset: u32) -> [u32; 5] {
    [0xF400_0002 | log << 18 | data << 6, 0xFA00_0000 | offset, v1_from(data), KEEP_V1[0], KEEP_V1[1]]
}

fn program(loads: &[[u32; 5]]) -> Vec<u32> {
    loads.iter().flatten().copied().chain([S_ENDPGM]).collect()
}

#[test]
fn loads_of_one_base_share_a_page_group() {
    let e = emit(&program(&[load(0, 8, 0x10), load(0, 9, 0x20)]));
    assert_eq!((e.stats.page_groups, e.stats.grouped_loads, e.stats.scalar_loads), (1, 2, 2));
    // One resolver for the base's page and one for the next, one after the other.
    assert_eq!((e.stats.resolves_single, e.stats.resolves_multi), (0, 2));
    // Heap handles: the locator's view and the root once each, both leaves for each of the
    // group's two pages, and the page each load picks.
    assert_eq!(e.count("handle"), 1 + 1 + 1 + 4 + 2);
    // Loads: the limits, the memory words, the locator's view, s4 and s5, the root and both
    // leaves for each of the group's two pages, and one for each load.
    assert_eq!(e.count("buffer_load"), 3 + 2 + 2 * 3 + 2);
    // A base that is read once is resolved by its load.
    let one = emit(&program(&[load(0, 8, 0x10)]));
    assert_eq!((one.stats.page_groups, one.stats.grouped_loads, one.stats.resolves_single), (0, 0, 1));
    // The base's own dword is in the first page: nothing to pick.
    let own = emit(&program(&[load(0, 8, 0), load(0, 9, 0x20)]));
    assert_eq!(own.count("handle"), e.count("handle"));
    assert!(own.count("select") < e.count("select"));
}

/// buffer_load_dwordx2 v[1:2], v0, s[4:7], `offset`, with the lane's offset on.
fn buffer_x2(offset: u32) -> [u32; 2] {
    [0xE034_1000 | offset, 0x8004_0100]
}

#[test]
fn buffer_loads_of_one_descriptor_and_lane_offset_share_a_page_group() {
    // Two loads of one descriptor at one lane offset, 0x10 apart: one page group serves both.
    let pair: Vec<u32> = buffer_x2(0x100).into_iter().chain(buffer_x2(0x110)).chain(KEEP_V1).chain([S_ENDPGM]).collect();
    let e = emit(&pair);
    assert_eq!((e.stats.page_groups, e.stats.grouped_loads, e.stats.resolves_multi), (1, 2, 2));
    // One of them alone: no group, and its own resolution of the page and the next.
    let one: Vec<u32> = buffer_x2(0x100).into_iter().chain(KEEP_V1).chain([S_ENDPGM]).collect();
    let one = emit(&one);
    assert_eq!((one.stats.page_groups, one.stats.resolves_multi), (0, 2));
    // A single dword is served by its own resolver even when the base is shared: a group
    // walks two pages where the load walks one.
    let singles: Vec<u32> = [0xE030_1000u32, 0x8004_0100, 0xE030_1010, 0x8004_0100].into_iter().chain(KEEP_V1).chain([S_ENDPGM]).collect();
    let singles = emit(&singles);
    assert_eq!((singles.stats.page_groups, singles.stats.resolves_single), (0, 2));
    // The same descriptor at another lane offset is another group.
    let moved: Vec<u32> = buffer_x2(0x100)
        .into_iter()
        .chain([0x7E00_0206])
        .chain(buffer_x2(0x110))
        .chain(KEEP_V1)
        .chain([S_ENDPGM])
        .collect();
    let moved = emit(&moved);
    assert_eq!((moved.stats.page_groups, moved.stats.resolves_multi), (0, 4));
}

#[test]
fn a_page_group_is_one_64_kib_of_offsets() {
    // Offsets 0x10 and 0x1_0010 are in different groups, each with one load: no group.
    let apart = emit(&program(&[load(0, 8, 0x10), load(0, 9, 0x1_0010)]));
    assert_eq!((apart.stats.page_groups, apart.stats.resolves_single), (0, 2));
    let e = emit(&program(&[load(0, 8, 0x10), load(0, 9, 0x1_0010), load(0, 10, 0xfffc)]));
    assert_eq!((e.stats.page_groups, e.stats.grouped_loads, e.stats.resolves_single), (1, 2, 1));
    // A load that is not on a dword, or whose last dword could lie past the next page, is
    // not a member.
    let off = emit(&program(&[load(0, 8, 0x12), load(0, 9, 0x22)]));
    assert_eq!((off.stats.page_groups, off.stats.resolves_single), (0, 2));
    let long = emit(&program(&[load(2, 8, 0xfff4), load(2, 12, 0xfff4)]));
    assert_eq!((long.stats.page_groups, long.stats.resolves_multi), (0, 4));
    let fits = emit(&program(&[load(2, 8, 0xfff0), load(2, 12, 0xfff0)]));
    assert_eq!((fits.stats.page_groups, fits.stats.grouped_loads, fits.stats.resolves_multi), (1, 2, 2));
}

#[test]
fn a_grouped_record_is_read_four_dwords_at_a_time() {
    // (without the thread limits, whose launch mask guards the LDS stores that keep the
    // dwords: that changes the count of handles, and this test counts reads)
    let emit = |words: &[u32]| emit_with(words, bare()).unwrap();
    let two = emit(&program(&[load(0, 8, 0x10), load(0, 9, 0x20)]));
    // (the record's first and last dwords are kept)
    let mut words = program(&[load(0, 8, 0x10), load(3, 9, 0x20)]);
    words.splice(words.len() - 1..words.len() - 1, [v1_from(16), KEEP_V1[0], KEEP_V1[1]]);
    let e = emit(&words);
    assert_eq!((e.stats.page_groups, e.stats.grouped_loads, e.stats.record_loads), (1, 2, 1));
    // Eight dwords: two reads of four, and one of the next page's first three dwords for
    // those of a four that lie past the page's end.
    assert_eq!(e.count("buffer_load"), two.count("buffer_load") + 2);
    // (a pick for the second four, and the next page's own handle)
    assert_eq!(e.count("handle"), two.count("handle") + 2);
    assert_eq!(e.count("blocks"), two.count("blocks"));
}

#[test]
fn a_group_reads_the_next_page_once_unless_the_program_is_coherent() {
    let emit = |words: &[u32]| emit_with(words, bare()).unwrap();
    let mut words = program(&[load(3, 8, 0x10), load(3, 16, 0x40)]);
    words.splice(words.len() - 1..words.len() - 1, [v1_from(23), KEEP_V1[0], KEEP_V1[1], v1_from(15), KEEP_V1[0], KEEP_V1[1]]);
    let plain = emit(&words);
    assert_eq!((plain.stats.page_groups, plain.stats.grouped_loads), (1, 2));
    // s_dcache_inv first: scalar loads must not take an older value, so each member reads
    // the next page's first dwords where it stands.
    let coherent = emit(&[&[0xF480_0000, 0][..], &words].concat());
    assert_eq!((coherent.stats.page_groups, coherent.stats.grouped_loads), (1, 2));
    assert_eq!(coherent.count("buffer_load"), plain.count("buffer_load") + 1);
}

#[test]
fn a_scalar_buffer_load_has_its_own_page_group() {
    // s_buffer_load_dword s8 and s9, s[4:7], 0x10 and 0x20: the same registers as a
    // pointer are another base, because the high dword is cut to 16 bits.
    let buffer = |data: u32, offset: u32| {
        let mut l = load(0, data, offset);
        l[0] |= 8 << 18;
        l
    };
    let e = emit(&program(&[buffer(8, 0x10), buffer(9, 0x20), load(0, 10, 0x30)]));
    assert_eq!((e.stats.page_groups, e.stats.grouped_loads, e.stats.resolves_single), (1, 2, 1));
    assert_eq!((e.stats.buffer_accesses, e.stats.scalar_loads), (2, 1));
}

/// image_sample_lz v[1:4], v[0:1], s[8:15] (a 2D image), s[4:7], after `before`.
fn sample(before: &[u32]) -> Vec<u32> {
    let mut words = before.to_vec();
    words.extend([0xF09C_0F08, 0x0022_0100, KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    words
}

#[test]
fn a_sampler_of_constants_is_static_and_any_other_comes_from_the_heap() {
    let constants: Vec<u32> = (4..8).flat_map(|n| s_mov(n, 0x1000 + n)).collect();
    let e = emit_with(&sample(&constants), XclConfig { sampler_base: 5, ..XclConfig::default() }).unwrap();
    assert_eq!(e.lowered.samplers, vec![[0x1004, 0x1005, 0x1006, 0x1007]]);
    assert_eq!((e.stats.static_samplers, e.stats.heap_samplers, e.stats.image_ops), (1, 0, 1));
    // The image from the view heap, the sampler by register.
    let heap = emit(&sample(&[]));
    assert_eq!((heap.stats.static_samplers, heap.stats.heap_samplers, heap.stats.image_ops), (0, 1, 1));
    assert!(heap.lowered.samplers.is_empty());
    assert_eq!(heap.count("handle"), e.count("handle"));
}

#[test]
fn a_sampler_word_made_by_reversing_a_constant_is_a_constant() {
    // s_brev_b32 s6, 40 (inline) and s_brev_b32 s7, 0x1 (literal): the console compiler makes
    // S# words without an inline form this way. xsemu's translator takes them as known, so
    // the sampler is static, not the heap slot in a last word the host never wrote.
    let mut words: Vec<u32> = [s_mov(4, 0x92), s_mov(5, 0x00C0_0000)].concat();
    words.extend([0xBE80_0BA8 | 6 << 16, 0xBE80_0BFF | 7 << 16, 1]);
    let e = emit(&sample(&words));
    assert_eq!(e.lowered.samplers, vec![[0x92, 0x00C0_0000, 0x1400_0000, 0x8000_0000]]);
    assert_eq!((e.stats.static_samplers, e.stats.heap_samplers), (1, 0));
    // s_not_b32 s7, 0: xsemu's translator does not follow it, so neither does the binding.
    let mut words: Vec<u32> = [s_mov(4, 0x92), s_mov(5, 0x00C0_0000)].concat();
    words.extend([0xBE80_0BA8 | 6 << 16, 0xBE80_0780 | 7 << 16]);
    let e = emit(&sample(&words));
    assert_eq!((e.stats.static_samplers, e.stats.heap_samplers), (0, 1));
}

#[test]
fn ray_tests_are_refused_under_the_binding() {
    // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[4:7]
    let words = [0xF19C_9F07, 0x0001_0D01, 0x0504_0302, 0x0908_0706, 0x000C_0B0A, S_ENDPGM];
    let r = emit_with(&words, XclConfig::default());
    assert!(matches!(r, Err(ShaderError::Unsupported { what: "ray tests under the XCL binding", .. })), "{:?}", r.err());
}

// ---- the numeric class of an image ----

/// image_load v[1:4], v[0:1], s[8:15] (2D, four channels).
const IMAGE_LOAD: [u32; 2] = [0xF000_0F08, 0x0002_0100];
/// image_store v[1:4], v[0:1], s[8:15] (2D, four channels).
const IMAGE_STORE: [u32; 2] = [0xF020_0F08, 0x0002_0100];
/// image_gather4_lz v[1:4], v[0:1], s[8:15], s[4:7] (2D, the first channel).
const IMAGE_GATHER: [u32; 2] = [0xF11C_0108, 0x0022_0100];
/// The d16 form of an image instruction.
const fn d16(image: [u32; 2]) -> [u32; 2] {
    [image[0], image[1] | 1 << 31]
}

/// v_mov_b32 v`k`, s`n`.
const fn v_from(k: u32, n: u32) -> u32 {
    0x7E00_0200 | k << 17 | n
}

/// `image` after v1..v4 are set from s0..s3, with a T# whose second word names `format` (the
/// first word is the heap index 3), or one the program reads from io when `format` is None.
/// v1 is kept.
fn image(image: [u32; 2], format: Option<u32>) -> Vec<u32> {
    let mut words: Vec<u32> = (1..5).map(|k| v_from(k, k - 1)).collect();
    if let Some(format) = format {
        words.extend(s_mov(8, 3));
        words.extend(s_mov(9, format << 20));
    }
    words.extend(image);
    words.extend([KEEP_V1[0], KEEP_V1[1], S_ENDPGM]);
    words
}

/// What [`emit`] gives, and the image intrinsics the binding emitted with their views'
/// component types.
fn typed(words: &[u32]) -> (Emitted, Vec<(String, u32)>) {
    super::TYPED.with(|t| t.borrow_mut().clear());
    let e = emit_with(words, bare()).unwrap();
    (e, super::TYPED.with(|t| std::mem::take(&mut *t.borrow_mut())))
}

fn op(name: &str, component: u32) -> Vec<(String, u32)> {
    vec![(name.to_string(), component)]
}

/// Formats of the buffer table: 32_UINT, 32_SINT, 32_FLOAT, and one past the table.
const UINT: u32 = 20;
const SINT: u32 = 21;
const FLOAT: u32 = 22;
const PAST: u32 = 100;

#[test]
fn a_uint_image_store_keeps_the_registers_bits() {
    let (uint, ops) = typed(&image(IMAGE_STORE, Some(UINT)));
    assert_eq!(ops, op("textureStore.i32", 0x405));
    // The registers' values go to the store as they are: no float conversion at all.
    assert_eq!((uint.count("bitcast"), uint.count("math_call")), (0, 0));
    let (sint, ops) = typed(&image(IMAGE_STORE, Some(SINT)));
    assert_eq!(ops, op("textureStore.i32", 0x404));
    assert_eq!(sint.count("bitcast"), 0);
    // d16: each half's 16 bits, not a half widened to a float.
    let (half, ops) = typed(&image(d16(IMAGE_STORE), Some(UINT)));
    assert_eq!(ops, op("textureStore.i32", 0x405));
    assert_eq!(half.count("math_call"), 0);
}

#[test]
fn a_sint_image_load_reads_a_view_of_signed_integers() {
    // The view's type is what makes the host sign-extend a narrow sint texel; the texel
    // reaches the register as the view gives it.
    let (sint, ops) = typed(&image(IMAGE_LOAD, Some(SINT)));
    assert_eq!(ops, op("textureLoad.i32", 0x404));
    assert_eq!((sint.count("bitcast"), sint.count("math_call")), (0, 0));
    let (_, ops) = typed(&image(IMAGE_LOAD, Some(UINT)));
    assert_eq!(ops, op("textureLoad.i32", 0x405));
    // d16 from an integer view: the low 16 bits, not a float rounded to a half.
    let (half, ops) = typed(&image(d16(IMAGE_LOAD), Some(SINT)));
    assert_eq!(ops, op("textureLoad.i32", 0x404));
    assert_eq!(half.count("math_call"), 0);
    let (_, ops) = typed(&image(IMAGE_GATHER, Some(SINT)));
    assert_eq!(ops, op("textureGather.i32", 0x404));
}

#[test]
fn a_float_image_stays_float() {
    for format in [FLOAT, PAST, 0] {
        let (store, ops) = typed(&image(IMAGE_STORE, Some(format)));
        assert_eq!(ops, op("textureStore.f32", 0x409), "format {format}");
        assert!(store.count("bitcast") > 0);
        let (_, ops) = typed(&image(IMAGE_LOAD, Some(format)));
        assert_eq!(ops, op("textureLoad.f32", 0x409));
        let (_, ops) = typed(&image(IMAGE_GATHER, Some(format)));
        assert_eq!(ops, op("textureGather.f32", 0x409));
    }
    // d16 from a float view: a float rounded to a half.
    assert!(typed(&image(d16(IMAGE_LOAD), Some(FLOAT))).0.count("math_call") > 0);
    // Sampling is float whatever the format, as in xsemu's translator.
    let mut words = image([0xF09C_0F08, 0x0022_0100], Some(UINT));
    words.splice(0..0, (4..8).flat_map(|n| s_mov(n, 0x1000 + n)));
    let (_, ops) = typed(&words);
    assert!(ops.is_empty());
}

#[test]
fn a_class_the_program_reads_is_chosen_when_it_runs() {
    let fixed = typed(&image(IMAGE_STORE, Some(UINT))).0;
    for (inst, name) in [(IMAGE_STORE, "textureStore"), (IMAGE_LOAD, "textureLoad"), (IMAGE_GATHER, "textureGather")] {
        let (e, ops) = typed(&image(inst, None));
        let i = format!("{name}.i32");
        assert_eq!(ops, vec![(i.clone(), 0x405), (i, 0x404), (format!("{name}.f32"), 0x409)]);
        assert_eq!((e.count("texture_read") + e.count("buffer_store"), e.stats.image_ops), (3, 1), "{name}");
    }
    // Two tests of the class and three ways that meet again.
    let (store, _) = typed(&image(IMAGE_STORE, None));
    assert_eq!(store.count("blocks"), fixed.count("blocks") + 5);
}

#[test]
fn the_class_table_is_the_translators() {
    use super::Class;
    let table = Class::table();
    for format in 0..512u32 {
        let bits = table.get(format as usize >> 4).copied().unwrap_or(0);
        let class = match format {
            5 | 11 | 18 | 20 | 27 | 34 | 41 | 48 | 54 | 60 | 62 | 69 | 72 | 75 => Class::Uint,
            6 | 12 | 19 | 21 | 28 | 35 | 42 | 49 | 55 | 61 | 63 | 70 | 73 | 76 => Class::Sint,
            _ => Class::Float,
        };
        assert_eq!((Class::of(format), bits >> (format % 16 * 2) & 3), (class, class as u32), "format {format}");
    }
}

#[test]
fn a_grouped_x8_or_x16_load_reads_each_four_dwords_16_bytes_on() {
    // s_load_dwordx8 and x16 at s[4:5] + offset, beside an s_load_dword of the same base
    // that makes the group. Dwords 4 to 15 are read at the offset plus 16, 32 and 48 bytes.
    // An offset near the page's end puts the later pieces in the next page for any base
    // whose low bits are not 0: the piece's byte decides which page it reads.
    for (log, n, offset) in [(3, 8, 0x40), (4, 16, 0x40), (3, 8, 0xffe0), (4, 16, 0xffc0)] {
        super::PIECES.with(|p| p.borrow_mut().clear());
        let e = emit_with(&program(&[load(log, 8, offset), load(0, 30, 4)]), bare()).unwrap();
        assert_eq!((e.stats.page_groups, e.stats.grouped_loads), (1, 2), "x{n} at {offset:#x}");
        let mut want: Vec<(u32, u32)> = (0..n).step_by(4).map(|first| (offset + 4 * first, (n - first).min(4))).collect();
        want.push((4, 1));
        assert_eq!(super::PIECES.with(|p| std::mem::take(&mut *p.borrow_mut())), want, "x{n} at {offset:#x}");
    }
}

