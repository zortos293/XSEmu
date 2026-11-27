//! `XCL_DSV_TRACE=1`: which PC texture the title's depth views, depth draws, copies and clears of
//! a console address land on. `XCL_DSV_TRACE_ADDRS` (comma-separated hex) names addresses that
//! are always reported. Every line starts with "DSV trace:".

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DSV_TRACE").is_ok_and(|v| v.trim() == "1"))
}

fn watched() -> &'static [usize] {
    static ADDRS: OnceLock<Vec<usize>> = OnceLock::new();
    ADDRS.get_or_init(|| {
        std::env::var("XCL_DSV_TRACE_ADDRS").unwrap_or_default().split(',')
            .filter_map(|a| usize::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect()
    })
}

fn is_depth(format: i32) -> bool {
    matches!(format, 19..=22 | 39..=41 | 44..=47 | 53..=55)
}

/// Whether a watched address is in `[va, va + bytes)`.
fn hits(va: Option<usize>, bytes: u64) -> bool {
    va.is_some_and(|va| watched().iter().any(|&a| a >= va && ((a - va) as u64) < bytes.max(1)))
}

fn address(res: usize) -> Option<usize> {
    super::real::title_address(res).or_else(|| super::title_textures::address(res))
}

#[derive(Clone, Copy)]
struct Texture { va: Option<usize>, buffer: bool, format: i32, size: (u64, u32, u16), mips: u16, flags: i32 }

impl Texture {
    unsafe fn of(res: usize) -> Option<Self> {
        let d = ID3D12Resource::from_raw_borrowed(&(res as *mut c_void))?.GetDesc();
        Some(Texture {
            va: address(res), buffer: d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER, format: d.Format.0,
            size: (d.Width, d.Height, d.DepthOrArraySize), mips: d.MipLevels, flags: d.Flags.0,
        })
    }
    /// at most the bytes of the title's memory it is over
    fn bytes(&self) -> u64 {
        if self.buffer { self.size.0 } else { self.size.0 * u64::from(self.size.1) * u64::from(self.size.2) * 8 }
    }
    fn wanted(&self) -> bool {
        is_depth(self.format) || hits(self.va, self.bytes())
    }
    fn describe(&self) -> String {
        let (w, h, a) = self.size;
        format!("at {:x?} format {} {w}x{h}x{a}, {} mips, flags {:#x}", self.va, self.format, self.mips, self.flags)
    }
}

#[derive(Default)]
struct Counts { binds: u64, draws: u64, all_binds: u64, all_draws: u64, writing: u64, not_writing: u64, unknown: u64 }

#[derive(Default)]
struct State {
    /// DSV handle → the PC resource it is a view of
    handles: HashMap<usize, usize>,
    /// command list → the PC resource of its DSV
    bound: HashMap<usize, usize>,
    /// PC resource of a DSV → what it is, its binds and the draws while it is bound
    dsvs: HashMap<usize, (Texture, Counts)>,
    /// console address → the PC resource the last view made by address there got
    views: HashMap<usize, usize>,
    /// (what, destination, source) of a logged view, transfer or clear → how many so far
    seen: HashMap<(&'static str, usize, usize), u64>,
    /// graphics pipeline → whether it writes depth
    psos: HashMap<usize, bool>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static LINES: AtomicU64 = AtomicU64::new(0);
static MEMORY_LINES: AtomicU64 = AtomicU64::new(0);
static CLEAR_LINES: AtomicU64 = AtomicU64::new(0);
static VIEW_LINES: AtomicU64 = AtomicU64::new(0);

fn state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    f(STATE.lock().unwrap().get_or_insert_with(State::default))
}

/// The count of `key` so far when it is logged: the 1st and each power of two, `max` lines in all.
fn logged(key: (&'static str, usize, usize), lines: &AtomicU64, max: u64) -> Option<u64> {
    let n = state(|s| {
        let n = s.seen.entry(key).or_default();
        *n += 1;
        *n
    });
    (n.is_power_of_two() && lines.fetch_add(1, Ordering::Relaxed) < max).then_some(n)
}

/// The live placements at `va` (`texture_alias`) other than `except`, and whether one is a depth buffer.
fn placed_at(va: usize, except: usize) -> (bool, String) {
    let placed: Vec<_> = super::texture_alias::placed_at(va).into_iter().filter(|p| p.0 != except).collect();
    let depth = placed.iter().any(|&(_, format, _, _, flags)| is_depth(format) || flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0);
    let text = placed.iter()
        .map(|&(r, format, (w, h, a), mips, flags)| format!("{r:#x} format {format} {w}x{h}x{a}, {mips} mips, flags {flags:#x}"))
        .collect::<Vec<_>>().join("; ");
    (depth, text)
}

/// The physical pages (64 KiB) the title's memory [va, va + bytes) shows (`kernelx::backing`).
fn pages(va: usize, bytes: u64) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = va & !0xffff;
    while (at as u64) < va as u64 + bytes {
        if let Some(b) = crate::hle::kernelx::backing(at) {
            if b.page != 0 {
                out.push(b.page);
            }
        }
        at += 0x1_0000;
    }
    out
}

/// A DSV at `cpu` on `res` (`format`: the view's): once per PC resource and address, what it is,
/// what else the title placed at that address and what views made by address there get.
pub(super) unsafe fn created(res: *mut c_void, cpu: usize, format: i32) {
    let Some(t) = Texture::of(res as usize) else {
        return state(|s| {
            s.handles.remove(&cpu);
        });
    };
    let r = res as usize;
    let fresh = state(|s| {
        s.handles.insert(cpu, r);
        let fresh = s.dsvs.get(&r).is_none_or(|(o, _)| (o.va, o.format, o.size) != (t.va, t.format, t.size));
        if fresh {
            s.dsvs.insert(r, (t, Counts::default()));
        }
        fresh
    });
    if !fresh {
        return;
    }
    let (_, others) = t.va.map(|va| placed_at(va, r)).unwrap_or_default();
    let sampled = t.va.map(|va| super::title_textures::placed_over(va, 1)).unwrap_or_default();
    let view = t.va.and_then(|va| state(|s| s.views.get(&va).copied()));
    crate::xlog_info!(
        "DSV trace: DSV {cpu:#x} on {r:#x} {} (view format {format}); also placed there: [{others}]; sampled textures over it: {sampled:?}; views made by address there get {view:x?}",
        t.describe()
    );
}

/// The DSV handle `dsv` is bound on `list` (None: no DSV).
pub(super) fn bind(list: usize, dsv: Option<usize>) {
    state(|s| {
        let Some(r) = dsv.and_then(|h| s.handles.get(&h).copied()) else {
            s.bound.remove(&list);
            return;
        };
        s.bound.insert(list, r);
        if let Some((_, c)) = s.dsvs.get_mut(&r) {
            c.binds += 1;
            c.all_binds += 1;
        }
    });
}

/// A graphics pipeline `pso` is made: whether it writes depth (enabled, mask ALL, a test that can pass).
pub(super) fn pso(pso: usize, writes: bool) {
    if on() {
        state(|s| {
            s.psos.insert(pso, writes);
        });
    }
}

/// A draw or indirect draw is recorded on `list`.
pub(super) fn draw(list: usize) {
    let pso = super::list_x::draw_gate_pso(list as *mut c_void);
    state(|s| {
        let Some(&r) = s.bound.get(&list) else { return };
        let writes = s.psos.get(&pso).copied();
        if let Some((_, c)) = s.dsvs.get_mut(&r) {
            c.draws += 1;
            c.all_draws += 1;
            match writes {
                Some(true) => c.writing += 1,
                Some(false) => c.not_writing += 1,
                None => c.unknown += 1,
            }
        }
    });
}

/// (every 300 frames) The binds and draws of each DSV resource drawn to since the last time or at
/// a watched address; what is placed at each watched address and what views made by address there get.
pub(super) fn log_frame() {
    if !on() {
        return;
    }
    let lines = state(|s| {
        let mut lines = Vec::new();
        for (&r, (t, c)) in &mut s.dsvs {
            if c.draws != 0 || t.va.is_some_and(|va| watched().contains(&va)) {
                lines.push(format!(
                    "DSV trace: DSV resource {r:#x} {}: {} binds, {} draws in the last 300 frames ({} / {} in all); of those draws {} by a pipeline that writes depth, {} by one that does not, {} by a pipeline not seen made",
                    t.describe(), c.binds, c.draws, c.all_binds, c.all_draws, c.writing, c.not_writing, c.unknown
                ));
            }
            (c.binds, c.draws, c.writing, c.not_writing, c.unknown) = (0, 0, 0, 0, 0);
        }
        lines
    });
    for line in lines {
        crate::xlog_info!("{line}");
    }
    for &va in watched() {
        let (_, placed) = placed_at(va, 0);
        let view = state(|s| s.views.get(&va).copied());
        crate::xlog_info!("DSV trace: watched {va:#x}: placed there [{placed}]; views made by address there get {view:x?}");
    }
    let spans: Vec<(usize, Vec<usize>)> = watched().iter().map(|&va| {
        let bytes = super::texture_alias::placed_at(va).first().map_or(0x1_0000, |&(_, _, (w, h, a), _, _)| w * u64::from(h) * u64::from(a) * 4);
        (va, pages(va, bytes))
    }).collect();
    for (i, (va, p)) in spans.iter().enumerate() {
        let shared: Vec<String> = spans[i + 1..].iter().map(|(other, q)| format!("{} with {other:#x}", p.iter().filter(|x| q.contains(x)).count())).collect();
        crate::xlog_info!("DSV trace: physical pages at {va:#x}: {} of 64 KiB, first {:x?}; shared: [{}]", p.len(), p.first(), shared.join(", "));
    }
}

/// A view made by address (`kind`) at `va` got `res` (`real::resource_at`): kept for the DSV
/// lines, and logged when it is of a depth format, a depth buffer is placed there, or `va` is watched.
pub(super) unsafe fn view_at(kind: &'static str, va: usize, res: *mut c_void) {
    if !on() {
        return;
    }
    state(|s| {
        s.views.insert(va, res as usize);
    });
    let t = Texture::of(res as usize);
    let (depth, others) = placed_at(va, res as usize);
    if !depth && !t.is_some_and(|t| t.wanted()) && !watched().contains(&va) {
        return;
    }
    let Some(n) = logged((kind, va, res as usize), &VIEW_LINES, 100) else { return };
    crate::xlog_info!(
        "DSV trace: {kind} #{n} made by address at {va:#x} gets {res:p} {}; also placed there: [{others}]",
        t.map_or_else(|| "(no resource)".to_string(), |t| t.describe())
    );
}

/// A copy or resolve (`kind`) of `src` into `dst`, logged when either is of a depth format or over
/// a watched address.
pub(super) unsafe fn transfer(kind: &'static str, dst: usize, src: usize) {
    if !on() {
        return;
    }
    let (d, s) = (Texture::of(dst), Texture::of(src));
    if !d.is_some_and(|t| t.wanted()) && !s.is_some_and(|t| t.wanted()) {
        return;
    }
    let Some(n) = logged((kind, dst, src), &LINES, 400) else { return };
    let describe = |t: Option<Texture>| t.map_or_else(|| "(no resource)".to_string(), |t| t.describe());
    crate::xlog_info!("DSV trace: {kind} #{n}: {src:#x} {} → {dst:#x} {}", describe(s), describe(d));
}

/// A copy of `bytes` bytes between title addresses (`kind`), logged when either range holds a
/// watched address.
pub(super) fn memory(kind: &'static str, dst: Option<usize>, src: Option<usize>, bytes: u64) {
    if !on() {
        return;
    }
    if !hits(dst, bytes) && !hits(src, bytes) {
        return;
    }
    let Some(n) = logged((kind, dst.unwrap_or(0), src.unwrap_or(0)), &MEMORY_LINES, 400) else { return };
    crate::xlog_info!("DSV trace: {kind} #{n}: {src:x?} → {dst:x?}, {bytes} bytes");
}

/// ClearDepthStencilView of the DSV `handle`.
pub(super) fn clear(handle: usize, flags: u32) {
    if !on() {
        return;
    }
    let Some((r, t)) = state(|s| s.handles.get(&handle).and_then(|r| Some((*r, s.dsvs.get(r)?.0)))) else { return };
    let Some(n) = logged(("ClearDepthStencilView", r, flags as usize), &CLEAR_LINES, 400) else { return };
    crate::xlog_info!("DSV trace: ClearDepthStencilView #{n} of DSV {handle:#x} on {r:#x} {}, flags {flags:#x} (1 depth, 2 stencil)", t.describe());
}

/// A UAV of `res` (`rd`) made a null view – the PC's depth buffers take none, and it has no shadow
/// texture (`depth_uav`) – when `res` is a depth buffer or of a depth format.
pub(super) fn null_uav(res: usize, rd: &D3D12_RESOURCE_DESC, view: Option<i32>) {
    if !is_depth(rd.Format.0) && rd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0 {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 64 {
        crate::xlog_info!(
            "DSV trace: a UAV of {res:#x} at {:x?} (format {} {}x{}, flags {:#x}, view format {view:?}) is a null view",
            address(res), rd.Format.0, rd.Width, rd.Height, rd.Flags.0
        );
    }
}
