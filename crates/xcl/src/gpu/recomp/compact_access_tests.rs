//! The compact accessors (`emit::compact_accessors`) against the earlier text
//! (`XCL_RECOMP_NO_COMPACT_ACCESS`). Both whole accessor blocks, as the translator writes them
//! for four windows, are compiled by the direct backend (`xcl_dxil`) and run by its interpreter
//! on the same generated memory, and must load, store and combine the same words at the same
//! views and offsets: with and without a page directory, on window and directory addresses,
//! absent, unmapped, mapped and invalid entries, addresses past 48 bits and past the windows, a
//! window base of 0 and one past 16 bits, and every dword of a 2, 3 and 4 dword access on
//! either side of a page end and of the 32-bit end. The interpreter's heap is generated, so the
//! miss view is a buffer like any other there: a wrong view or offset shows even for a miss.
//! A wrong page, offset, direct test or trip count in the compact text must show.
//!
//! One thing the generated heap can do that the host's cannot: a generated root can name the
//! miss view or a mapped view as a leaf, so a store rewrites the entry of its own page. The
//! earlier text resolves both pages of a store that crosses before its first dword and the
//! compact text resolves each dword as it stores, so the two differ there, and only there. The
//! host's leaves are buffers of the directory that no view of memory covers, so the programs
//! that store run with the directory apart from the views (`apart`).

use xcl_dxil::interp::{self, Config, Verdict};

/// One state of the accessors' variables from two generated words: no directory or one at a
/// generated slot, a window that fits the address or any, an origin that is null, the address's,
/// or anything, and an address at a page end, at the 32-bit end, at a page start, or anywhere.
/// Returns the low word of an address a few dwords around the effective address.
const SETUP: &str = r#"
ByteAddressBuffer xin : register(t0);
RWByteAddressBuffer xout : register(u0);
uint edge(uint x, uint k) {
    uint s = (k >> 28) & 7u;
    return select(s == 0u, 0u, select(s == 1u, x & 15u, select(s == 2u, 65534u + (x & 3u), select(s == 3u, xcl_uma_base + (x & 7u),
        select(s == 4u, 0xfffffff0u + (x & 15u), select(s == 5u, 1u, x))))));
}
uint setup(uint4 a, uint4 b, uint c) {
    uint u = (a.x >> 4) & 3u;
    xcl_uma_base = select(u == 0u, 0u, select(u == 3u, 65533u + (a.x & 3u), 16u + (a.x & 3u)));
    xcl_windows = 1u + ((a.x >> 2) & 3u);
    xcl_mem_views = 100u;
    xcl_page_table = select((c & 3u) == 0u, 0u, 0x80000u | (a.y & 0x3fffcu) | 4u);
    uint w = select((c & 4u) != 0u, (b.y >> 30) % xcl_windows, edge(b.x ^ a.w, b.z) & 7u);
    xcl_w = w;
    xcl_wv = select((a.x & 64u) == 0u, w, (b.y >> 27) & 7u);
    uint k = (c >> 3) & 3u;
    xcl_origin = select(k == 0u, 0u, select(k == 1u, xcl_uma_base + w, select(k == 2u, 0u, edge(a.z, b.x))));
    uint hi = select(k == 0u, 0u, select(k == 1u, xcl_uma_base + w + (b.w & 1u), select(k == 2u, b.w & 1u, edge(b.w, a.y))));
    uint s = (c >> 5) & 3u;
    uint lo = select(s == 0u, (b.z & 0xffff0000u) | (65520u + (b.z & 15u)), select(s == 1u, 0xfffffff0u + (b.z & 15u), select(s == 2u, (b.z & 0xffff0000u) | (b.z & 3u), b.z)));
    xcl_ea = select(and(k == 0u, (b.x & 1u) == 0u), 0ull, ((uint64_t)hi << 32) | (uint64_t)lo);
    return lo + select((a.z & 1u) == 0u, 0u, (a.z >> 1) & 12u) - select((a.z & 32u) == 0u, 0u, (a.z >> 6) & 12u);
}
"#;

/// A generated leaf word as a page table entry of each kind in turn: absent, unmapped, a mapped
/// page of a view, or the word itself (nearly always invalid). A generated 0 stays absent.
const ENTRY_KINDS: &str = r#"
uint xcl_test_pte(uint x) {
    uint m = x * 0x9e3779b1u, k = m >> 30;
    return select(k == 0u, 0u, select(k == 1u, 0x80000000u, select(k == 2u, 0x80000000u | (((m >> 8) & 127u) << 20) | ((m & 255u) + 1u), x)));
}
"#;

/// The slots of the directory and of the views kept apart: a leaf in the upper half of the
/// slots the root's are in, a mapped view below it.
const APART: &str = r#"
uint2 xcl_test_leaves(uint2 l) { return select(l == 0u, 0u, 0x80000u | (l & 0x3ffffu)); }
uint xcl_test_view(uint pte) { return pte & ~0x80000u; }
"#;

const LOADS: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    uint p = setup(a, b, i ^ (i >> 6));
    uint m = a.w ^ b.y;
    bool4 q = bool4((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) == 0u, (m & 24u) != 0u);
    uint4 xa = select((m & 96u) == 0u, uint4(p, b.x, 0u, a.z), select(q, p + uint4(0u, 4u, 8u, 12u), 0u));
    uint o = i * 112u;
    xout.Store4(o, xcl_mLoad4(p));
    xout.Store4(o + 16u, uint4(xcl_mLoad3(p + 4u), xcl_mLoad(p)));
    xout.Store4(o + 32u, uint4(xcl_mLoad2(p + 8u), xcl_vLoad2(p + 12u)));
    xout.Store4(o + 48u, xcl_vLoad4(p + 4u));
    xout.Store4(o + 64u, uint4(xcl_vLoad3(p), xcl_vLoad(p + 8u)));
    xout.Store4(o + 80u, xcl_loadx4(xa));
    xout.Store4(o + 96u, uint4(xcl_resolve_read(p), xcl_resolve_va(((uint64_t)b.w << 32) | (uint64_t)p)));
}
"#;

/// One thread: two lanes storing to one generated word would race in the interpreter. Each store
/// has pages of its own (the final words are what is compared, and a later store over an earlier
/// one would hide it), and each in turn has the pages the setup aims at an end.
const STORES: &str = r#"
[numthreads(1, 1, 1)]
void main() {
    for (uint k = 0u; k < 256u; k++) {
        uint4 a = xin.Load4(k * 32u), b = xin.Load4(k * 32u + 16u);
        uint p = setup(a, b, (k >> 3) * 5u + (a.w >> 25));
        xcl_mStore4(p + ((k + 0u) & 7u) * 0x100000u, a ^ b);
        xcl_mStore3(p + ((k + 1u) & 7u) * 0x100000u, b.xyz);
        xcl_mStore2(p + ((k + 2u) & 7u) * 0x100000u, a.zw);
        xcl_mStore(p + ((k + 3u) & 7u) * 0x100000u, k);
        xcl_vStore4(p + ((k + 4u) & 7u) * 0x100000u, b);
        xcl_vStore3(p + ((k + 5u) & 7u) * 0x100000u, a.yzw);
        xcl_vStore2(p + ((k + 6u) & 7u) * 0x100000u, b.xy);
        xcl_vStore(p + ((k + 7u) & 7u) * 0x100000u, ~k);
    }
}
"#;

const ATOMICS: &str = r#"
[numthreads(1, 1, 1)]
void main() {
    for (uint k = 0u; k < 96u; k++) {
        uint4 a = xin.Load4(k * 32u), b = xin.Load4(k * 32u + 16u);
        uint p = setup(a, b, k * 5u + (a.w >> 25));
        uint64_t x = ((uint64_t)a.x << 32) | (uint64_t)b.x, y = ((uint64_t)b.z << 32) | (uint64_t)a.z;
        uint o0, o1, o2; int o3; uint64_t q0, q1, q2; int64_t q3;
        xcl_mInterlockedAdd(p, a.y, o0);
        xcl_mInterlockedOr(p + 4u, b.y);
        xcl_mInterlockedCompareExchange(p + 8u, a.z, b.z, o1);
        xcl_vInterlockedExchange(p, a.w, o2);
        xcl_vInterlockedMax(p + 4u, (int)b.w, o3);
        xcl_mInterlockedAdd64(p, x, q0);
        xcl_mInterlockedCompareExchange64(p + 8u, x, y, q1);
        xcl_vInterlockedXor64(p + 4u, y, q2);
        xcl_vInterlockedMin64(p, (int64_t)x, q3);
        xout.Store4(k * 48u, uint4(o0, o1, o2, (uint)o3));
        xout.Store4(k * 48u + 16u, uint4((uint)q0, (uint)(q0 >> 32), (uint)q1, (uint)(q1 >> 32)));
        xout.Store4(k * 48u + 32u, uint4((uint)q2, (uint)(q2 >> 32), (uint)q3, (uint)((uint64_t)q3 >> 32)));
    }
}
"#;

/// A group's loads: the effective address of each at one of eight offsets from the group's
/// anchor, in its page and the next, as the translator's loads of one base are; four addresses
/// with lanes whose result the load drops (address 0).
const GROUPS: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    setup(a, b, i ^ (i >> 7));
    uint p = (uint)xcl_ea & ~3u;
    uint64_t e0 = xcl_ea;
    uint64_t g = xcl_operand_va(p);
    uint4 r = xcl_group_pages(g);
    uint m = a.w ^ b.y;
    uint o = i * 640u;
    [unroll] for (uint k = 0u; k < 8u; k++) {
        uint d = select(k == 7u, 65504u, select(k == 6u, 4096u, select(k == 5u, 65280u, 4u * k * (k + 1u))));
        xcl_ea = e0 + d;
        bool ok = ((m >> k) & 1u) != 0u;
        bool4 q = bool4(((m >> (8u + k)) & 1u) != 0u, ((m >> (16u + k)) & 1u) != 0u, ((m >> (24u + k)) & 1u) == 0u, true);
        xout.Store4(o + 80u * k, xcl_mLoad4G(p + d, g, r));
        xout.Store4(o + 80u * k + 16u, xcl_loadx4G(select(q, (p + d) + uint4(0u, 4u, 8u, 12u), 0u), g, r));
        xout.Store4(o + 80u * k + 32u, uint4(xcl_mLoadG(select(ok, p + d + 4u, 0u), g, r), xcl_mLoad2G(p + d + 8u, g, r), xcl_vLoadG(p + d, g, r)));
        xout.Store4(o + 80u * k + 48u, uint4(xcl_mLoad3G(p + d + 4u, g, r), d));
        xout.Store4(o + 80u * k + 64u, xcl_vLoad4G(p + d + 8u, g, r));
    }
}
"#;

/// The loads of a coherent program on the view that is not coherent: the cached loads and the
/// plain copies of the group and four-address loads.
const CACHED: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    uint p = setup(a, b, i ^ (i >> 6));
    uint64_t g = xcl_operand_va(p);
    uint4 r = xcl_group_pages(g);
    uint o = i * 80u;
    xout.Store4(o, xcl_m_cachedLoad4(p));
    xout.Store4(o + 16u, uint4(xcl_m_cachedLoad2(p + 8u), xcl_m_cachedLoad(p + 4u), xcl_mLoad(p)));
#ifdef XCL_TEST_PLAIN_COPIES
    xout.Store4(o + 32u, xcl_m_cachedLoad4G(p + 4u, g, r));
    xout.Store4(o + 48u, uint4(xcl_m_cachedLoad2G(p + 8u, g, r), xcl_m_cachedLoadG(p, g, r), 0u));
    xout.Store4(o + 64u, xcl_loadx4_cached(p + uint4(0u, 4u, 65536u, 12u)) ^ xcl_loadx4G_cached(p + uint4(0u, 4u, 8u, 12u), g, r));
#endif
}
"#;

/// The accessor block of one form, its window count a variable the setup sets.
fn block(compact: bool, groups: bool, coherence: u8) -> String {
    let all = super::emit::mem_accessors_in(4, groups, compact, coherence);
    let head = "static const uint xcl_windows = 4u;";
    assert!(all.starts_with(head));
    all.replacen(head, "static uint xcl_windows = 4u;", 1)
}

/// `block` reading its leaves through `xcl_test_pte`, so entries of every kind are common.
fn every_kind(block: &str) -> String {
    let wrapped = block
        .replace("return leaf.Load((lo >> 16) * 4u);", "return xcl_test_pte(leaf.Load((lo >> 16) * 4u));")
        .replace("pte = leaf0.Load((lo >> 16) * 4u);", "pte = xcl_test_pte(leaf0.Load((lo >> 16) * 4u));")
        .replace("pte = leaf1.Load((lo >> 16) * 4u);", "pte = xcl_test_pte(leaf1.Load((lo >> 16) * 4u));");
    assert!(wrapped != block);
    wrapped
}

/// `block` with its leaves and its mapped views in slots of their own (`APART`), so that no store
/// reaches the directory.
fn apart(block: &str) -> String {
    let mut wrapped = block.to_string();
    for root in ["root.Load2(va.y * 8u)", "root.Load2(h * 8u)"] {
        wrapped = wrapped.replace(root, &format!("xcl_test_leaves({root})"));
    }
    for leaf in ["leaf.Load((lo >> 16) * 4u)", "leaf0.Load((lo >> 16) * 4u)", "leaf1.Load((lo >> 16) * 4u)"] {
        wrapped = wrapped.replace(leaf, &format!("xcl_test_view({leaf})"));
    }
    assert_eq!(wrapped.matches("xcl_test_leaves(").count(), 1);
    assert!(matches!(wrapped.matches("xcl_test_view(").count(), 1 | 2));
    wrapped
}

fn program(block: &str, driver: &str) -> interp::Program {
    let src = format!("static uint xcl_mem_views = 0u, xcl_uma_base = 0u, xcl_w = 0u, xcl_wv = 0u;\n{ENTRY_KINDS}\n{APART}\n{block}\n{SETUP}\n{driver}");
    interp::load(&xcl_dxil::compile(&src, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{src}"))).unwrap()
}

fn verdict(earlier: &str, compact: &str, driver: &str) -> Verdict {
    interp::diff(&program(earlier, driver), &program(compact, driver), 0..24, &Config::default()).0
}

#[test]
fn compact_accessors_load_store_and_combine_what_the_earlier_text_does() {
    let (earlier, compact) = (block(false, false, 0), block(true, false, 0));
    assert!(earlier.contains("bool xcl_direct(") && earlier.contains("uint2 xcl_lookup_gpu_pte(") && !earlier.contains("xcl_at("));
    assert!(compact.contains("uint2 xcl_at(") && !compact.contains("xcl_direct(") && !compact.contains("xcl_lookup_gpu_pte("));
    assert_eq!(verdict(&earlier, &compact, LOADS), Verdict::Equal, "loads, the directory anywhere");
    assert_eq!(verdict(&every_kind(&earlier), &every_kind(&compact), LOADS), Verdict::Equal, "loads, the directory anywhere, entries of every kind");
    for (name, driver) in [("loads", LOADS), ("stores", STORES), ("atomics", ATOMICS)] {
        assert_eq!(verdict(&apart(&earlier), &apart(&compact), driver), Verdict::Equal, "{name}");
        assert_eq!(verdict(&apart(&every_kind(&earlier)), &apart(&every_kind(&compact)), driver), Verdict::Equal, "{name}, entries of every kind");
    }
}

#[test]
fn compact_group_loads_read_what_the_earlier_text_does() {
    // the earlier text here is that of `accessor_shrink`, whose four-address group load the compact one follows
    let (earlier, compact) = (block(false, true, 0), block(true, true, 0));
    assert!(earlier.contains("xcl_group_at(v3, g, r)") && compact.contains("uint2 p = xcl_group_at(u, g, r);"));
    assert_eq!(verdict(&earlier, &compact, GROUPS), Verdict::Equal);
    assert_eq!(verdict(&every_kind(&earlier), &every_kind(&compact), GROUPS), Verdict::Equal, "entries of every kind");
    assert_eq!(verdict(&earlier, &compact, LOADS), Verdict::Equal, "the loads without a group, beside the group's");
}

/// The views of a coherent program: every accessor's view is coherent where the earlier one's
/// was, the cached loads and the plain copies are the same text without the qualifier, and all
/// of them read what the earlier ones read.
#[test]
fn compact_accessors_keep_the_coherence_of_each_view() {
    for coherence in 1..=3u8 {
        let (earlier, compact) = (block(false, true, coherence), block(true, true, coherence));
        let driver = if coherence == 3 { format!("#define XCL_TEST_PLAIN_COPIES 1\n{CACHED}") } else { CACHED.to_string() };
        let driver = if coherence == 1 { LOADS } else { driver.as_str() };
        assert_eq!(verdict(&earlier, &compact, driver), Verdict::Equal, "coherence {coherence}");
        assert_eq!(verdict(&earlier, &compact, GROUPS), Verdict::Equal, "coherence {coherence}, groups");
        for line in compact.lines().filter(|l| l.contains("RWByteAddressBuffer b = ")) {
            let cached = line.contains("_cached");
            assert_eq!(line.matches("RWByteAddressBuffer b = ").count(), 1, "one memory view an accessor: {line}");
            assert_eq!(line.contains("globallycoherent RWByteAddressBuffer b = "), !cached, "{line}");
        }
        let named = |text: &str, head: &str| text.lines().find(|l| l.starts_with(head)).unwrap_or_else(|| panic!("{head}")).to_string();
        let mut pairs = if coherence == 1 { vec![] } else { vec![("uint xcl_mLoad(", "uint xcl_m_cachedLoad("), ("uint2 xcl_mLoad2(", "uint2 xcl_m_cachedLoad2("), ("uint4 xcl_mLoad4(", "uint4 xcl_m_cachedLoad4(")] };
        if coherence == 3 {
            pairs.extend([("uint xcl_mLoadG(", "uint xcl_m_cachedLoadG("), ("uint4 xcl_mLoad4G(", "uint4 xcl_m_cachedLoad4G("), ("uint4 xcl_loadx4(", "uint4 xcl_loadx4_cached("), ("uint4 xcl_loadx4G(", "uint4 xcl_loadx4G_cached(")]);
        }
        for (coherent, plain) in pairs {
            assert_eq!(named(&compact, coherent).replacen(&coherent[coherent.find(' ').unwrap()..], &plain[plain.find(' ').unwrap()..], 1).replace("globallycoherent ", ""), named(&compact, plain));
        }
        assert_eq!(compact.contains("_cached"), coherence != 1);
    }
}

/// One resolution and one load of each kind an accessor: what the compact form is for.
#[test]
fn a_compact_accessor_resolves_once_and_has_one_view() {
    let compact = block(true, true, 0);
    let accessors: Vec<&str> = compact.lines().filter(|l| [" xcl_m", " xcl_v", " xcl_loadx4"].iter().any(|p| l.split('(').next().is_some_and(|h| h.contains(p)))).collect();
    // 34 methods and overloads and 4 group loads a family, and the two four-address loads
    assert_eq!(accessors.len(), 2 * (34 + 4) + 2);
    for a in accessors {
        assert_eq!(a.matches("ResourceDescriptorHeap[").count(), 1, "{a}");
        assert_eq!(a.matches("xcl_resolve_va(").count() + a.matches("xcl_at(").count() + a.matches("xcl_group_at(").count(), 1, "{a}");
        let several = ["Load2", "Load3", "Load4", "Store2", "Store3", "Store4", "loadx4"].iter().any(|n| a.contains(n));
        assert_eq!(a.matches("[loop] for (").count(), usize::from(several), "{a}");
        assert!(!a.contains(" ? ") && !a.contains("[noinline]"), "{a}");
    }
    // the walk is in the resolution alone, behind its one branch
    assert_eq!(compact.matches("root.Load2(").count(), 1);
    assert_eq!(compact.matches("if (and(ok, xcl_page_table != 0u))").count(), 1);
}

#[test]
fn a_wrong_page_offset_direct_test_or_trip_count_is_caught() {
    let (earlier, compact) = (apart(&every_kind(&block(false, true, 0))), apart(&every_kind(&block(true, true, 0))));
    let references = [LOADS, STORES, GROUPS].map(|d| program(&earlier, d));
    let mut missed = Vec::new();
    for (from, to) in [
        // the resolution
        ("uint h = select(hi == 0u, xcl_uma_base, hi);", "uint h = hi;"),
        ("!or(xcl_origin > 65535u, or(", "!or(false, or("),
        ("or((xcl_origin == 0u) != (hi == 0u), h > 65535u)", "or((xcl_origin == 0u) != (hi == 0u), false)"),
        ("or((xcl_origin == 0u) != (hi == 0u), h > 65535u)", "or(false, h > 65535u)"),
        ("if (and(pte == 0u, leaves.y != 0u))", "if (leaves.y != 0u)"),
        ("if (leaves.x != 0u) {", "if (false) {"),
        ("(pte & 0xf8000000u) == 0x80000000u", "(pte & 0xf0000000u) == 0x80000000u"),
        ("bool mapped = and((pte & 0xf8000000u) == 0x80000000u, v != 0u);", "bool mapped = (pte & 0xf8000000u) == 0x80000000u;"),
        ("and(pte == 0u, w < xcl_windows)", "and(pte == 0u, w <= xcl_windows)"),
        ("and(pte == 0u, w < xcl_windows)", "w < xcl_windows"),
        ("select(mapped, v, xcl_mem_views + w), xcl_mem_views + 4u)", "select(mapped, v, xcl_mem_views + w), xcl_mem_views + 3u)"),
        ("((pte >> 20) & 127u) << 16", "((pte >> 20) & 63u) << 16"),
        ("| (lo & 65535u), 0u));", "| (lo & 65535u), lo & 65535u));"),
        // the direct test
        ("and(xcl_page_table == 0u, xcl_origin <= 65535u)", "and(true, xcl_origin <= 65535u)"),
        ("select(xcl_origin == 0u, and(hi == 0u, w == 0u), hi == xcl_uma_base + w)", "select(xcl_origin == 0u, hi == 0u, hi == xcl_uma_base + w)"),
        // several dwords
        ("xcl_operand_va((a | 65535u) + 2u)", "xcl_operand_va((a | 65535u))"),
        ("select((ak >> 16) == (a >> 16), va, vq)", "select((ak >> 16) != (a >> 16), va, vq)"),
        ("and((a & 65535u) > 65520u, !and(c, a <= 0xfffffff0u))", "and((a & 65535u) > 65524u, !and(c, a <= 0xfffffff0u))"),
        ("and((a & 65535u) > 65524u, !and(c, a <= 0xfffffff4u))", "and((a & 65535u) > 65524u, !and(c, a <= 0xfffffffcu))"),
        ("n = select(x, 4u, 1u);", "n = select(x, 3u, 1u);"),
        ("bool f = and(c, ak <= select(x, 0xfffffffcu, 0xfffffff8u));", "bool f = and(c, ak <= select(x, 0xfffffff8u, 0xfffffff8u));"),
        // (of two dwords at the 32-bit end only the first is in the window, so this one needs three)
        ("uint y = select(f, ak, p.y); if (x) { b.Store(y, v[k]); } else { b.Store3(y, v); }", "uint y = select(f, a, p.y); if (x) { b.Store(y, v[k]); } else { b.Store3(y, v); }"),
        ("if (x) { b.Store(y, v[k]); } else { b.Store3(y, v); }", "if (x) { b.Store(y, v[0]); } else { b.Store3(y, v); }"),
        ("o = select(k == uint3(0u, 1u, 2u), uint3(t, t, t), o);", "o = select(k == uint3(0u, 2u, 1u), uint3(t, t, t), o);"),
        // four addresses
        ("uint n = select(same, 1u, 4u);", "uint n = select(same, 1u, 3u);"),
        ("uint4 q = (p.y & 0xffff0000u) | (a & 65535u);", "uint4 q = (p.y & 0xffff0000u) | (a.x & 65535u);"),
        ("bool f = and(xcl_window(u, xcl_w), ak <= 0xfffffffcu); p = uint2(select(f, xcl_mem_views + xcl_w, p.x), select(f, ak, p.y));", "bool f = false;"),
        ("uint2 p = xcl_at(u, xcl_w, 0xfffffffcu);", "uint2 p = xcl_at(u, xcl_w, 0xfffffff0u);"),
        // a group's loads
        ("bool d = and(xcl_window(va, xcl_w), a <= 0xfffffffcu);", "bool d = and(xcl_window(va, xcl_w), a <= 0xfffffff0u);"),
        ("uint2 p = xcl_group_at(u, g, r); bool f = and(c, ak <= select(x, 0xfffffffcu, 0xfffffff0u));", "uint2 p = xcl_group_at(va0, g, r); bool f = and(c, ak <= select(x, 0xfffffffcu, 0xfffffff0u));"),
    ] {
        assert!(compact.contains(from), "{from}");
        let mutant = compact.replacen(from, to, 1).replace("xcl_group_at(va0, g, r)", "xcl_group_at(xcl_operand_va(a), g, r)");
        let caught = references.iter().zip([LOADS, STORES, GROUPS]).any(|(reference, driver)| {
            matches!(interp::diff(reference, &program(&mutant, driver), 0..24, &Config::default()).0, Verdict::Different(_))
        });
        if !caught {
            missed.push(to);
        }
    }
    assert!(missed.is_empty(), "mutants not caught: {missed:#?}");
}

/// A group's loads through the group's pages (`xcl_group_at`) and through its folded pages
/// (`emit::group_fold`): the loads the translator writes for one group, at eight offsets from the
/// anchor, in its page and the next. A result that is not taken (a masked address) is left out
/// on both sides: the fold reads it where it lies, the earlier text at address 0.
const FOLDED: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    setup(a, b, i ^ (i >> 7));
    uint p = (uint)xcl_ea & ~3u;
    uint64_t e0 = xcl_ea;
    uint64_t g = xcl_operand_va(p);
    uint4 r = xcl_group_pages(g);
    uint gs = (uint)g & 65535u, gz = (uint)g >> 16;
    uint4 rm = xcl_mFold(g, r), rv = xcl_vFold(g, r);
    uint m = a.w ^ b.y;
    uint o = i * 896u;
    [unroll] for (uint k = 0u; k < 8u; k++) {
        uint d = select(k == 7u, 65504u, select(k == 6u, 4096u, select(k == 5u, 65280u, 4u * k * (k + 1u))));
        xcl_ea = e0 + d;
        bool ok = ((m >> k) & 1u) != 0u;
        bool4 q = or(((m >> (8u + k)) & 1u) != 0u, bool4(((m >> (16u + k)) & 1u) != 0u, ((m >> (24u + k)) & 1u) == 0u, ((m >> (20u + k)) & 1u) == 0u, ((m >> (12u + k)) & 1u) == 0u));
        bool4 q2 = q.xyyy, q3 = q.xyzz;
        uint c = o + 112u * k;
#ifdef XCL_TEST_FOLDED
        xout.Store4(c, LOAD4S(gs + d, rm));
        xout.Store4(c + 16u, select(q, LOAD4X(gs + d, gz, rm, q), 0u));
        xout.Store4(c + 32u, uint4(select(ok, LOADS(gs + d + 4u, rm), 0u), LOAD2S(gs + d + 8u, rm), xcl_vLoadS(gs + d, rv)));
        xout.Store4(c + 48u, uint4(xcl_mLoad3S(gs + d + 4u, rm), d));
        xout.Store4(c + 64u, xcl_vLoad4S(gs + d + 8u, rv));
        xout.Store4(c + 80u, uint4(select(q2, LOAD2X(gs + d + 4u, gz, rm, q2), 0u).xy, xcl_vLoad2S(gs + d + 16u, rv)));
        xout.Store4(c + 96u, uint4(select(q3, xcl_mLoad3X(gs + d + 8u, gz, rm, q3), 0u).xyz, xcl_vLoad3S(gs + d + 12u, rv).z));
#else
        xout.Store4(c, LOAD4G(p + d, g, r));
        xout.Store4(c + 16u, select(q, LOADX4(select(q, (p + d) + uint4(0u, 4u, 8u, 12u), 0u), g, r), 0u));
        xout.Store4(c + 32u, uint4(select(ok, LOADG(select(ok, p + d + 4u, 0u), g, r), 0u), LOAD2G(p + d + 8u, g, r), xcl_vLoadG(p + d, g, r)));
        xout.Store4(c + 48u, uint4(xcl_mLoad3G(p + d + 4u, g, r), d));
        xout.Store4(c + 64u, xcl_vLoad4G(p + d + 8u, g, r));
        xout.Store4(c + 80u, uint4(select(q2, LOADX4(select(q2, (p + d + 4u) + uint4(0u, 4u, 4u, 4u), 0u), g, r), 0u).xy, xcl_vLoad2G(p + d + 16u, g, r)));
        xout.Store4(c + 96u, uint4(select(q3, xcl_loadx4G(select(q3, (p + d + 8u) + uint4(0u, 4u, 8u, 8u), 0u), g, r), 0u).xyz, xcl_vLoad3G(p + d + 12u, g, r).z));
#endif
    }
}
"#;

/// `FOLDED` on the coherent views (the plain names) or on the plain copies of a coherent
/// program, through the group's pages or the folded ones.
fn folded_driver(folded: bool, cached: bool) -> String {
    let (load, four) = if cached { ("xcl_m_cachedLoad", "xcl_loadx4G_cached") } else { ("xcl_mLoad", "xcl_loadx4G") };
    format!("{}{}", if folded { "#define XCL_TEST_FOLDED 1\n" } else { "" }, FOLDED.replace("LOADX4", four).replace("LOAD", load))
}

#[test]
fn folded_group_loads_read_what_the_loads_through_the_group_read() {
    for (coherence, cached) in [(4, false), (5, false), (7, false), (7, true)] {
        let text = block(true, true, coherence);
        assert!(text.contains("uint4 xcl_mFold(") && text.contains("uint4 xcl_vFold(") && text.contains("uint4 xcl_mLoad4X(") && !text.contains("xcl_vLoad4X("));
        assert_eq!(text.contains("uint4 xcl_m_cachedLoad4X(") && text.contains("uint xcl_m_cachedLoadS("), coherence == 7);
        for text in [text.clone(), every_kind(&text)] {
            let (through, folded) = (program(&text, &folded_driver(false, cached)), program(&text, &folded_driver(true, cached)));
            assert_eq!(interp::diff(&through, &folded, 0..24, &Config::default()).0, Verdict::Equal, "coherence {coherence}, plain copies {cached}");
        }
        // a folded load: one view, no resolution, no direct test and no address in 64 bits
        for line in text.lines().filter(|l| ["S(uint s, uint4 r)", "X(uint s, uint z, uint4 r, bool4 m)"].iter().any(|p| l.contains(p))) {
            assert_eq!(line.matches("ResourceDescriptorHeap[").count(), 1, "{line}");
            assert!(!["xcl_resolve_va(", "xcl_at(", "xcl_group_at(", "xcl_window(", "xcl_operand_va(", "uint64_t", " ? ", "[noinline]"].iter().any(|p| line.contains(p)), "{line}");
            assert_eq!(line.contains("globallycoherent "), coherence != 4 && !line.contains("_cached"), "{line}");
        }
    }
    // without the fold the text is that of the compact accessors alone
    assert!(!block(true, true, 0).contains("Fold(") && !block(true, true, 3).contains("S(uint s"));
}

#[test]
fn a_wrong_fold_page_offset_or_miss_is_caught() {
    let text = every_kind(&block(true, true, 4));
    let reference = program(&text, &folded_driver(false, false));
    let mut missed = Vec::new();
    for (from, to) in [
        // the fold
        ("bool d0 = xcl_window(s, xcl_w), d1 = xcl_window(s + 0x10000ull, xcl_w);", "bool d0 = xcl_window(s, xcl_w), d1 = d0;"),
        ("bool d0 = xcl_window(s, xcl_wv), d1 = xcl_window(s + 0x10000ull, xcl_wv);", "bool d0 = xcl_window(s, xcl_w), d1 = xcl_window(s + 0x10000ull, xcl_w);"),
        ("select(d0, (uint)s + 1u, r.y)", "select(d0, (uint)s, r.y)"),
        ("select(d1, (uint)s + 0x10001u, r.w)", "select(d1, (uint)s + 1u, r.w)"),
        ("select(d1, v, r.z)", "r.z"),
        ("return uint4(select(d0, v, r.x)", "return uint4(r.x"),
        // one dword
        ("uint xcl_mLoadS(uint s, uint4 r) { bool q = s > 65535u; uint y = select(q, r.w, r.y);", "uint xcl_mLoadS(uint s, uint4 r) { bool q = s > 65535u; uint y = r.y;"),
        ("return b.Load(select(y == 0u, 0u, y - 1u + (s & 65535u))); }", "return b.Load(select(y == 0u, 0u, y + (s & 65535u))); }"),
        ("return b.Load(select(y == 0u, 0u, y - 1u + (s & 65535u))); }", "return b.Load(y - 1u + (s & 65535u)); }"),
        // the dwords of one address
        ("uint4 xcl_mLoad4S(uint s, uint4 r) { uint4 o = uint4(0u, 0u, 0u, 0u); bool x = (s & 65535u) > 65520u;", "uint4 xcl_mLoad4S(uint s, uint4 r) { uint4 o = uint4(0u, 0u, 0u, 0u); bool x = (s & 65535u) > 65524u;"),
        ("uint4 xcl_mLoad4S(uint s, uint4 r) { uint4 o = uint4(0u, 0u, 0u, 0u); bool x = (s & 65535u) > 65520u; uint n = select(x, 4u, 1u);", "uint4 xcl_mLoad4S(uint s, uint4 r) { uint4 o = uint4(0u, 0u, 0u, 0u); bool x = (s & 65535u) > 65520u; uint n = select(x, 3u, 1u);"),
        ("else { o = b.Load4(select(y == 0u, 0u, y - 1u + (sk & 65535u))); }", "else { o = b.Load4(select(y == 0u, 0u, y - 1u) + (sk & 65535u)); }"),
        ("uint2 xcl_mLoad2S(uint s, uint4 r) { uint2 o = uint2(0u, 0u); bool x = (s & 65535u) > 65528u;", "uint2 xcl_mLoad2S(uint s, uint4 r) { uint2 o = uint2(0u, 0u); bool x = (s & 65535u) > 65520u;"),
        ("uint sk = s + 4u * k; bool q = sk > 65535u; uint y = select(q, r.w, r.y);", "uint sk = s + 4u * k; bool q = s > 65535u; uint y = select(q, r.w, r.y);"),
        // the dwords of a four-address load
        ("bool c = (s & 65535u) > 65520u; bool x = or(c, !all(m));", "bool c = (s & 65535u) > 65520u; bool x = c;"),
        ("bool c = (s & 65535u) > 65520u; bool x = or(c, !all(m));", "bool c = (s & 65535u) > 65520u; bool x = !all(m);"),
        ("bool e = or(!x, all(or(!m, ((z + ((s + uint4(0u, 4u, 8u, 12u)) >> 16)) & 65535u) == 0u)));", "bool e = !x;"),
        ("bool e = or(!x, all(or(!m, ((z + ((s + uint4(0u, 4u, 8u, 12u)) >> 16)) & 65535u) == 0u)));", "bool e = true;"),
        ("bool e = or(!x, all(or(!m, ((z + ((s + uint4(0u, 4u, 8u, 12u)) >> 16)) & 65535u) == 0u)));", "bool e = or(!x, all(or(!m, ((s + uint4(0u, 4u, 8u, 12u)) >> 16) == 0u)));"),
        ("bool c = (s & 65535u) > 65528u; bool x = or(c, !all(m));", "bool c = (s & 65535u) > 65528u; bool x = c;"),
        ("bool x = or(c, !all(m)); bool e = or(!x, all(or(!m, ((z + ((s + uint4(0u, 4u, 8u, 8u)) >> 16)) & 65535u) == 0u))); uint n = select(x, 3u, 1u);", "bool x = or(c, !all(m)); bool e = or(!x, all(or(!m, ((z + ((s + uint4(0u, 4u, 8u, 8u)) >> 16)) & 65535u) == 0u))); uint n = select(x, 2u, 1u);"),
    ] {
        assert!(text.contains(from), "{from}");
        let mutant = program(&text.replacen(from, to, 1), &folded_driver(true, false));
        if !matches!(interp::diff(&reference, &mutant, 0..24, &Config::default()).0, Verdict::Different(_)) {
            missed.push(to);
        }
    }
    assert!(missed.is_empty(), "mutants not caught: {missed:#?}");
}

/// `XCL_RECOMP_NO_COMPACT_ACCESS` is the earlier text byte for byte: the blocks of the forms the
/// switch covers, by the hash of the text the tree had before the compact form.
#[test]
fn the_switch_gives_the_earlier_text_byte_for_byte() {
    let hash = |windows, groups, coherence| crate::gpu::shader::fnv64(super::emit::mem_accessors_in(windows, groups, false, coherence).as_bytes());
    let got: Vec<u64> = [(4, false, 0), (4, true, 0), (4, true, 1), (4, true, 2), (4, true, 3), (2, true, 1)].iter().map(|&(w, g, c)| hash(w, g, c)).collect();
    assert_eq!(got, EARLIER_TEXT, "{got:#x?}");
    // one window has no directory and no compact form
    for (groups, coherence) in [(false, 0), (true, 1)] {
        assert_eq!(super::emit::mem_accessors_in(1, groups, true, coherence), super::emit::mem_accessors_in(1, groups, false, coherence));
    }
}

const EARLIER_TEXT: [u64; 6] = [0x2bc74ded4507e388, 0x06b7b424dbb31f3c, 0x871f517f6718eea6, 0x007207e53a00f83f, 0x7925fc6e0bbdfb18, 0x8facbdf8b36dbc30];
