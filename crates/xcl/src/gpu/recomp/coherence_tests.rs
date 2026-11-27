use super::{translate, CsInterface, Mode, Options};
use crate::gpu::rdna2::encode;

type Inst = (&'static str, &'static str, &'static [(&'static str, u64)]);

const PLAIN: Inst = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SOFFSET", 125)]);
const FLAGGED: Inst = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SOFFSET", 125), ("OFFSET", 4), ("GLC", 1)]);
const ATOMIC: Inst = ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 0), ("DATA", 2), ("SADDR", 2), ("GLC", 1)]);

fn program(insts: &[Inst]) -> Vec<u8> {
    insts.iter().copied().chain([("S_ENDPGM", "ENC_SOPP", &[][..])])
        .flat_map(|(name, enc, fields)| encode(name, enc, fields, None).unwrap_or_else(|e| panic!("{name}: {e}")))
        .collect()
}

fn options(mem_windows: u32, wave64: bool, split_wave64: bool, legacy_compute_wave64: bool) -> Options {
    let threads = if wave64 { 64 } else { 32 };
    let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 };
    Options {
        mode: Mode::Compute(cs), mem_windows, wave64, split_wave64, legacy_compute_wave64,
        compute_per_access_coherence: true, guard_trip_counters: false, ..Default::default()
    }
}

fn hlsl(insts: &[Inst], opt: &Options) -> String {
    plain_loads_hlsl(insts, opt, false)
}

/// The per-access option's tests are of that option alone: the loads `emit::plain_load_limit` takes
/// off the coherent view have their own test (`plain`: on).
fn plain_loads_hlsl(insts: &[Inst], opt: &Options, plain: bool) -> String {
    super::emit::PLAIN_LOADS.set(Some(plain));
    let t = translate(&program(insts), opt);
    super::emit::PLAIN_LOADS.set(None);
    let t = t.unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

fn legacy_hlsl(insts: &[Inst], opt: &Options) -> String {
    hlsl(insts, &Options { compute_per_access_coherence: false, ..opt.clone() })
}

#[test]
fn compute_per_access_coherence_caches_only_before_first_flagged_read() {
    // the text this test reads is the earlier one (`XCL_RECOMP_NO_COMPACT_ACCESS`); the compact accessors have
    // their own tests (`compact_access_tests`)
    super::emit::NO_COMPACT_ACCESS.set(true);
    let wide = ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SOFFSET", 125), ("OFFSET", 16)][..]);
    let wait = ("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)][..]);
    for windows in [1, 4] {
        for (wave64, split, legacy) in [(false, false, false), (true, false, false), (true, true, false), (true, true, true)] {
            let opt = options(windows, wave64, split, legacy);
            let h = hlsl(&[PLAIN, wide, wait, FLAGGED, wait, PLAIN, wide], &opt);
            let (prelude, body) = h.split_once("void main(").unwrap();
            let call = body.lines().filter_map(|line| {
                let (_, tail) = line.split_once(" = xcl_m")?;
                Some(tail.split_once('(')?.0)
            }).collect::<Vec<_>>();
            let grouped = windows > 1 && !(split && !legacy);
            let after = if grouped { ["LoadG", "Load4G"] } else { ["Load", "Load4"] };
            assert_eq!(call, ["_cachedLoad", "_cachedLoad4", "Load", after[0], after[1]]);
            let view = if windows == 1 { "xcl_mem_views" } else { "xcl_mem_views + xcl_w" };
            if grouped {
                assert!(prelude.contains(&format!("xcl_mLoadG(uint a, uint64_t g, uint4 r) {{ globallycoherent RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            }
            assert!(prelude.contains(&format!("xcl_m_cachedLoad(uint a) {{ RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            assert!(prelude.contains(&format!("xcl_mLoad(uint a) {{ globallycoherent RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            assert_eq!(hlsl(&[FLAGGED, PLAIN], &opt), legacy_hlsl(&[FLAGGED, PLAIN], &opt));
        }
    }
}

#[test]
fn compute_per_access_coherence_preserves_whole_shader_fallbacks() {
    let opt = options(4, false, false, false);
    let baseline = hlsl(&[PLAIN], &opt);
    assert!(!baseline.contains("globallycoherent"));
    assert_eq!(baseline, legacy_hlsl(&[PLAIN], &opt));
    let cases: &[Inst] = &[
        ("S_DCACHE_INV", "ENC_SMEM", &[]),
        ("S_GL1_INV", "ENC_SMEM", &[]),
        ("BUFFER_GL0_INV", "ENC_MUBUF", &[]),
        ("BUFFER_GL1_INV", "ENC_MUBUF", &[]),
        ("IMAGE_LOAD", "ENC_MIMG", &[("DMASK", 1), ("DIM", 1)]),
        ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 2)]),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 125)]),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 4)]),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 125), ("SBASE", 6)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 0), ("SOFFSET", 125)]),
    ];
    for &inst in cases {
        let code = [PLAIN, FLAGGED, inst, PLAIN];
        let h = hlsl(&code, &opt);
        assert!(h.contains("globallycoherent"));
        assert!(!h.contains("_cachedLoad"), "{inst:?}");
        assert!(h == legacy_hlsl(&code, &opt), "{inst:?}");
    }
    let kernel = Options { mode: Mode::Kernel, ..opt };
    let h = hlsl(&[PLAIN, ATOMIC], &kernel);
    assert!(!h.contains("globallycoherent"));
    assert_eq!(h, legacy_hlsl(&[PLAIN, ATOMIC], &kernel));
}

#[test]
fn compute_per_access_coherence_keeps_writes_atomics_and_barriers() {
    let store = ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 2), ("SADDR", 0)][..]);
    let byte_store = ("BUFFER_STORE_BYTE", "ENC_MUBUF", &[("VDATA", 2), ("SOFFSET", 128)][..]);
    let barrier = ("S_BARRIER", "ENC_SOPP", &[][..]);
    let cas = ("FLAT_ATOMIC_INC_X2", "ENC_FLAT", &[("VDST", 5), ("DATA", 2), ("GLC", 1)][..]);
    for windows in [1, 4] {
        let opt = options(windows, true, true, false);
        for inst in [ATOMIC, store, byte_store, barrier, cas] {
            let code = [PLAIN, inst, PLAIN, FLAGGED, PLAIN];
            let h = hlsl(&code, &opt);
            assert!(!h.contains("_cachedLoad"), "{}", inst.0);
            assert!(h == legacy_hlsl(&code, &opt), "{}", inst.0);
        }
    }
}

#[test]
fn compute_per_access_coherence_does_not_cache_branching_or_polling() {
    let opt = options(4, false, false, false);
    let branch = ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)][..]);
    let cmp = ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 128)][..]);
    let back = ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffc)][..]);
    for code in [
        &[PLAIN, branch, FLAGGED, PLAIN][..],
        &[PLAIN, FLAGGED, cmp, back, PLAIN][..],
    ] {
        let h = hlsl(code, &opt);
        assert!(!h.contains("_cachedLoad"));
        assert!(h == legacy_hlsl(code, &opt));
    }
}

/// The scalar loads a coherent program runs before its first memory write, atomic, flagged access,
/// barrier or loop read through the view that is not coherent (`emit::plain_load_limit`), with their
/// page groups; every later load and every write keeps the coherent one. A program that
/// invalidates caches itself keeps it everywhere, and a program that is not coherent has neither.
#[test]
fn plain_loads_end_at_the_first_write_flag_barrier_or_loop() {
    // through a group's pages, then with them folded (`emit::group_fold`): the same loads plain
    // with a base's pages kept across blocks or not (`emit::keep_pages`), and with a four-address
    // load that is alone on its base in a group of its own or not (`emit::lone_loads`)
    for (fold, kept, lone) in [(false, false, false), (true, false, false), (false, true, false), (true, true, false), (false, false, true), (true, true, true)] {
        super::emit::NO_GROUP_FOLD.set(!fold);
        super::emit::NO_KEPT_PAGES.set(!kept);
        super::emit::NO_LONE_LOADS.set(!lone);
        plain_loads_end(fold, kept, lone);
    }
    super::emit::NO_KEPT_PAGES.set(false);
    super::emit::NO_LONE_LOADS.set(false);
}

fn plain_loads_end(fold: bool, kept: bool, lone: bool) {
    let second = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 125), ("OFFSET", 8)][..]);
    let wide = ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SOFFSET", 125), ("OFFSET", 16)][..]);
    let buf = ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("SOFFSET", 125)][..]);
    let buf1 = ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 4), ("SOFFSET", 125), ("OFFSET", 16)][..]);
    let store = ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 2), ("SADDR", 0)][..]);
    let barrier = ("S_BARRIER", "ENC_SOPP", &[][..]);
    let cmp = ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 128)][..]);
    let back = ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffc)][..]);
    let inv = ("S_DCACHE_INV", "ENC_SMEM", &[][..]);
    for (wave64, split, legacy) in [(false, false, false), (true, true, false), (true, true, true)] {
        let opt = Options { compute_per_access_coherence: false, ..options(4, wave64, split, legacy) };
        // the loads of the program in their order, as the accessors they call
        let loads = |code: &[Inst], plain: bool| -> (String, Vec<String>) {
            let h = plain_loads_hlsl(code, &opt, plain);
            let body = h.split_once("void main(").unwrap().1;
            let mut calls = Vec::new();
            for line in body.lines() {
                let mut rest = line;
                while let Some(i) = ["xcl_m_cachedLoad", "xcl_mLoad", "xcl_loadx4"].iter().filter_map(|k| rest.find(k)).min() {
                    let tail = &rest[i..];
                    let n = tail.find('(').unwrap();
                    calls.push(tail[..n].to_string());
                    rest = &tail[n..];
                }
            }
            (h, calls)
        };
        // the packed form has no page groups
        let g = if split && !legacy { "" } else if fold { "S" } else { "G" };
        let c = |n: &str| format!("xcl_m_cachedLoad{n}{g}");
        let m = |n: &str| format!("xcl_mLoad{n}{g}");
        // the four-address load of each view
        let (x4c, x4) = if g == "S" { ("xcl_m_cachedLoad4X".to_string(), "xcl_mLoad4X".to_string()) } else { (format!("xcl_loadx4{g}_cached"), format!("xcl_loadx4{g}")) };
        let cases: [(&[Inst], Vec<String>); 5] = [
            // (the last load is alone on its base: its own accessor where a group needs a second load)
            (&[PLAIN, second, wide, buf, buf1, ATOMIC, PLAIN, buf], vec![c(""), c(""), c("4"), x4c.clone(), c(""), m(""), if lone && !g.is_empty() { "xcl_loadx4".to_string() } else { x4.clone() }]),
            (&[PLAIN, wide, store, PLAIN, ATOMIC], vec![c(""), c("4"), m("")]),
            (&[PLAIN, wide, FLAGGED, PLAIN, ATOMIC], vec![c(""), c("4"), m(""), m("")]),
            (&[PLAIN, wide, barrier, PLAIN, ATOMIC], vec![c(""), c("4"), m("")]),
            // the loop's head is its second load: the first runs once, before it. The three are
            // in three blocks: each alone, or through the pages kept for their base
            (&[PLAIN, second, cmp, back, PLAIN, ATOMIC], if kept { vec![c(""), m(""), m("")] } else { vec!["xcl_m_cachedLoad".into(), "xcl_mLoad".into(), "xcl_mLoad".into()] }),
        ];
        for (code, want) in cases {
            let (h, calls) = loads(code, true);
            assert_eq!(calls, want, "{code:?}");
            assert!(h.contains("globallycoherent "));
            let (off, calls) = loads(code, false);
            assert!(!off.contains("_cached") && calls.iter().all(|c| c.starts_with("xcl_mLoad") || c == &x4 || lone && c == "xcl_loadx4"), "{calls:?}");
        }
        for code in [&[PLAIN, wide, inv, ATOMIC][..], &[ATOMIC, PLAIN], &[PLAIN, wide, store]] {
            let (h, _) = loads(code, true);
            assert!(!h.contains("_cached"), "{code:?}");
            assert_eq!(h, loads(code, false).0);
            assert_eq!(h.contains("globallycoherent "), code.contains(&ATOMIC));
        }
    }
}
