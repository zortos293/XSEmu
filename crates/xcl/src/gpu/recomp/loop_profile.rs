use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::OnceLock;

use super::{emit, rdna2::Decoded};

pub const LOOP_PROFILE_SITES: usize = 32;
pub const LOOP_PROFILE_WORDS: usize = 10;

pub fn loop_profile_enabled() -> bool {
    selection().is_some()
}

pub(super) fn selection() -> Option<&'static str> {
    static SELECTION: OnceLock<Option<String>> = OnceLock::new();
    SELECTION.get_or_init(|| std::env::var("XCL_LOOP_PROFILE").ok().map(|v| v.trim().to_owned()).filter(|v| v == "1" || v.split(',').all(|p| u64::from_str_radix(p.trim(), 16).is_ok()))).as_deref()
}

pub(super) fn enabled_for(blob: &[u8]) -> bool {
    selection().is_some_and(|v| v == "1" || {
        let hash = crate::gpu::shader::fnv64(blob);
        v.split(',').any(|p| u64::from_str_radix(p.trim(), 16) == Ok(hash))
    })
}

pub(super) fn static_cap() -> u32 {
    static CAP: OnceLock<u32> = OnceLock::new();
    *CAP.get_or_init(|| std::env::var("XCL_STATIC_LOOP_CAP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

pub(super) struct Sites {
    profile: bool,
    cap: u32,
    bounds: BTreeMap<(u64, u64), u32>,
    ids: RefCell<BTreeMap<(u64, u64), usize>>,
}

impl Sites {
    pub(super) fn new(insts: &[Decoded], profile: bool) -> Self {
        let cap = static_cap();
        let bounds = if profile || cap != 0 {
            super::loop_bounds(insts).filter_map(|(a, b, _, n)| Some(((a, b), u32::try_from(n?).ok()?))).collect()
        } else {
            BTreeMap::new()
        };
        Self { profile, cap, bounds, ids: RefCell::default() }
    }

    pub(super) fn probe(&self, a: u64, b: u64) -> Probe {
        let id = if self.profile {
            let mut ids = self.ids.borrow_mut();
            let next = ids.len();
            let id = *ids.entry((a, b)).or_insert(next);
            (id < LOOP_PROFILE_SITES).then_some(id)
        } else {
            None
        };
        let bound = self.bounds.get(&(a, b)).copied().unwrap_or(0);
        let cap = if self.cap != 0 && bound > self.cap { self.cap } else { 0 };
        Probe { id, pc: a, bound, cap }
    }
}

pub(super) struct Probe {
    id: Option<usize>,
    pc: u64,
    bound: u32,
    cap: u32,
}

impl Probe {
    fn active(&self) -> bool {
        self.id.is_some() || self.cap != 0
    }

    pub(super) fn before(&self, body: &mut String, pad: &str) {
        if self.active() {
            let _ = writeln!(body, "{pad}{{ uint xcl_lp_n = 0u; uint xcl_lp_exit = 0u;");
        }
    }

    pub(super) fn guard(&self, body: &mut String, pad: &str, em: &emit::Em) {
        if !self.active() {
            let _ = writeln!(body, "{pad}    {}", super::loop_guard_line(em));
            return;
        }
        // the tests and the count of `LOOP_GUARD_WAVE` / `LOOP_GUARD`, as the program's other loops have them
        let (test, end, spend) = if matches!(em.loop_budget, super::LoopBudget::Wave(_)) { ("WaveActiveAnyTrue(", ")", "xcl_budget -= min(xcl_budget, 1u);") } else { ("", "", "xcl_budget--;") };
        let _ = writeln!(body, "{pad}    if ({test}xcl_budget == 0u{end}) {{ xcl_lp_exit = 1u; xcl_trip(0u); break; }}");
        if self.cap != 0 {
            let _ = writeln!(body, "{pad}    if ({test}xcl_lp_n >= {}u{end}) {{ xcl_lp_exit = 2u; xcl_trip(0u); break; }}", self.cap);
        }
        let _ = writeln!(body, "{pad}    {spend} xcl_lp_n++;");
        if let Some(id) = self.id {
            let _ = writeln!(body, "{pad}    if ((xcl_lp_n & (xcl_lp_n - 1u)) == 0u) xcl_lp_pulse({id}u, {}u, xcl_lp_n);", self.pc + 1);
        }
        if em.waits() {
            let _ = writeln!(body, "{pad}    xcl_tick();");
        }
    }

    pub(super) fn after(&self, body: &mut String, pad: &str) {
        if let Some(id) = self.id {
            let _ = writeln!(body, "{pad}if (xcl_lp_n != 0u || xcl_lp_exit != 0u) xcl_lp_done({id}u, {}u, xcl_lp_n, xcl_lp_exit, {}u);", self.pc + 1, self.bound);
        }
        if self.active() {
            let _ = writeln!(body, "{pad}}}");
        }
    }
}

pub(super) fn prelude(h: &mut String, packed: bool) {
    let _ = writeln!(h, "uint xcl_lp_base(uint slot) {{ return io.Load({}u) + 16u + slot * {}u; }}", super::IO_TRIP_AT * 4, LOOP_PROFILE_WORDS * 4);
    let _ = writeln!(h, "#define XCL_LP_HEAP ResourceDescriptorHeap[io.Load({}u) + io.Load({}u)]", super::IO_MEM_VIEWS * 4, super::IO_TRIP_WINDOW * 4);
    let weight = if packed { "(uint)(WaveGetLaneIndex() < xcl_valid_lanes) + (uint)(WaveGetLaneIndex() + 32u < xcl_valid_lanes)" } else { "1u" };
    let _ = writeln!(h, "uint xcl_lp_weight() {{ return {weight}; }}");
    h.push_str(PRELUDE);
}

const PRELUDE: &str = r#"
void xcl_lp_pulse(uint slot, uint pc, uint n) {
    if (WaveIsFirstLane() && io.Load(460u) != 0u) {
        RWByteAddressBuffer stats = XCL_LP_HEAP;
        uint addr = xcl_lp_base(slot);
        uint old;
        stats.InterlockedMax(addr, pc, old);
        stats.InterlockedMax(addr + 4u, n, old);
    }
}

void xcl_lp_done(uint slot, uint pc, uint n, uint reason, uint bound) {
    uint weight = xcl_lp_weight();
    uint lanes = WaveActiveSum(weight);
    uint max_n = WaveActiveMax(n);
    uint low16 = WaveActiveSum((n & 65535u) * weight);
    uint high16 = WaveActiveSum((n >> 16u) * weight);
    uint budget = WaveActiveSum(select(reason == 1u, weight, 0u));
    uint capped = WaveActiveSum(select(reason == 2u, weight, 0u));
    uint limit = WaveActiveSum(select(reason == 0u && bound != 0u && n >= bound, weight, 0u));
    uint early = WaveActiveSum(select(reason == 0u && bound != 0u && n < bound, weight, 0u));
    if (WaveIsFirstLane() && io.Load(460u) != 0u) {
        RWByteAddressBuffer stats = XCL_LP_HEAP;
        uint addr = xcl_lp_base(slot);
        uint old;
        stats.InterlockedMax(addr, pc, old);
        stats.InterlockedMax(addr + 8u, max_n, old);
        uint low = low16 + (high16 << 16u);
        uint high = (high16 >> 16u) + (uint)(low < low16);
        stats.InterlockedAdd(addr + 12u, low, old);
        uint carry = (uint)(old + low < old);
        stats.InterlockedAdd(addr + 16u, high + carry, old);
        stats.InterlockedAdd(addr + 20u, lanes, old);
        stats.InterlockedAdd(addr + 24u, early, old);
        stats.InterlockedAdd(addr + 28u, limit, old);
        stats.InterlockedAdd(addr + 32u, budget, old);
        stats.InterlockedAdd(addr + 36u, capped, old);
    }
}
"#;
