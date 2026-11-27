//! `XCL_DUMP_TARGETS_AFTER_S=<seconds>` (off by default): once, a whole frame the title presents
//! after that many seconds is recorded, and at the present that ends it the images its last
//! passes wrote are copied out, described and saved. For each draw, dispatch, copy and clear the
//! frame submits, the recording keeps the images it writes (render targets, UAV textures, copy
//! and clear destinations), the viewport and scissor it ran with, and the images it samples with
//! the size and format their descriptors give. Recording starts at one present and the frame
//! dumped is the one between the next two, so lists the title recorded a frame ahead are in it.
//!
//! Dumped: the presented planes, the last `XCL_DUMP_TARGETS_COUNT` (32) images written, and the
//! images those passes sample, with the copies bounded by `XCL_DUMP_TARGETS_MAX_MB` (512). Each
//! is logged ("target dump:") with its PC description and row pitch, the console's description
//! and layout (row pitch, tile mode) beside it, what differs between the two, the other dumped
//! images at the same title address, its last writers and readers, and written to the title's
//! logs folder as `targets_<frame>/<n>.raw` (the copy's rows as the GPU laid them out) and
//! `<n>.bmp` for the formats it can show. The copies are made in whatever state the title left
//! each image in, without a transition.
//!
//! A frame of fewer than `XCL_DUMP_TARGETS_MIN_PASSES` (100) passes (a loading screen, a menu, a
//! present of the interface plane alone) is passed over for the next one, for as long as it
//! takes (0: the first frame is dumped, whatever it holds).
//!
//! `XCL_DUMP_TARGETS_EACH=<W>x<H>[@<format>][,…]`: images of those PC sizes (and, with `@`, that
//! DXGI format number alone) are also copied after every pass of the dumped frame that writes
//! them (after a UAV barrier), so a target several passes write in turn is seen at each step: at
//! most `XCL_DUMP_TARGETS_EACH_MAX` (48) copies, counted against the same `XCL_DUMP_TARGETS_MAX_MB`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// An image a pass samples: its PC resource and what its descriptor says.
#[derive(Clone, Debug)]
pub(super) struct Read {
    pub resource: usize,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub swizzle: u32,
    pub base_level: u32,
    pub slot: String,
}

/// One pass of the frame as recorded on its list.
#[derive(Clone, Debug, Default)]
pub(super) struct Op {
    pub kind: &'static str,
    pub pso: usize,
    pub shaders: String,
    pub writes: Vec<usize>,
    pub reads: Vec<Read>,
    /// the first viewport (x, y, width, height) and how many were set
    pub viewport: Option<([f32; 4], usize)>,
    /// the first scissor (left, top, right, bottom) and how many were set
    pub scissor: Option<([i32; 4], usize)>,
}

fn after() -> Option<f64> {
    static AFTER: OnceLock<Option<f64>> = OnceLock::new();
    *AFTER.get_or_init(|| std::env::var("XCL_DUMP_TARGETS_AFTER_S").ok().and_then(|v| v.trim().parse().ok()).filter(|s: &f64| *s >= 0.0))
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

pub(super) fn configured() -> bool {
    after().is_some()
}

/// 0 waiting, 1 recording ahead, 2 recording the frame, 3 done
static STATE: AtomicU8 = AtomicU8::new(0);

pub(super) fn recording() -> bool {
    matches!(STATE.load(Ordering::Relaxed), 1 | 2)
}

#[derive(Default)]
struct Recorded {
    lists: HashMap<usize, Vec<Op>>,
    /// (queue, pass) in submission order, from the frame's first present on
    frame: Vec<(usize, Op)>,
    /// every image a recorded pass named, held until the dump is written
    held: HashMap<usize, ID3D12Resource>,
    /// `XCL_DUMP_TARGETS_EACH` copies on lists not submitted yet, by list
    list_snaps: HashMap<usize, Vec<Snap>>,
    /// `XCL_DUMP_TARGETS_EACH` copies of lists the frame submitted, in submission order
    snaps: Vec<Snap>,
    snap_bytes: u64,
    /// `XCL_DUMP_TARGETS_EACH` copies refused for `XCL_DUMP_TARGETS_EACH_MAX` or the budget
    snaps_refused: usize,
    /// copies dropped as their list was neither submitted nor reset within a frame
    snaps_stale: usize,
    /// frame boundaries passed while recording
    gen: u32,
}

impl Recorded {
    /// A frame boundary that does not dump: the frame's copies go, and so do the ones on lists
    /// left unsubmitted for a whole frame (they would hold `XCL_DUMP_TARGETS_EACH_MAX` forever).
    fn next_frame(&mut self) {
        self.frame.clear();
        park(std::mem::take(&mut self.snaps));
        self.gen += 1;
        let gen = self.gen;
        let mut stale = Vec::new();
        for snaps in self.list_snaps.values_mut() {
            let (keep, old): (Vec<Snap>, Vec<Snap>) = std::mem::take(snaps).into_iter().partition(|s| s.gen + 1 >= gen);
            *snaps = keep;
            stale.extend(old);
        }
        self.list_snaps.retain(|_, s| !s.is_empty());
        self.snaps_stale += stale.len();
        park(stale);
        self.snap_bytes = self.list_snaps.values().flatten().map(|s| s.bytes).sum();
    }
}

/// Readback buffers a copy may still be writing, and the images copied from, by the allocator
/// of the list holding the copy: a buffer goes only when that allocator is reset or released,
/// which D3D12 allows only once the GPU has finished every list recorded from it.
struct Parked(Vec<(usize, ID3D12Resource)>);

// only touched under the lock
unsafe impl Send for Parked {}

static PARKED: Mutex<Parked> = Mutex::new(Parked(Vec::new()));

fn park(snaps: impl IntoIterator<Item = Snap>) {
    let mut p = PARKED.lock().unwrap();
    for s in snaps {
        p.0.push((s.allocator, s.copy.0));
        p.0.push((s.allocator, s.src));
    }
}

/// `allocator` was reset or released: the GPU is done with the copies its lists recorded.
pub(super) fn allocator_done(allocator: usize) {
    let done: Vec<_> = {
        let mut p = PARKED.lock().unwrap();
        if p.0.is_empty() {
            return;
        }
        let (done, kept) = std::mem::take(&mut p.0).into_iter().partition(|e| e.0 == allocator);
        p.0 = kept;
        done
    };
    drop(done);
}

/// An image copied right after a pass that wrote it (`XCL_DUMP_TARGETS_EACH`).
struct Snap {
    res: usize,
    what: String,
    bytes: u64,
    /// the frame boundary count when it was made
    gen: u32,
    /// the allocator of the list the copy is recorded on
    allocator: usize,
    /// the image copied, held while the copy may run
    src: ID3D12Resource,
    copy: (ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, DXGI_FORMAT),
}

/// `XCL_DUMP_TARGETS_EACH`: the PC sizes (width, height, DXGI format if named) copied after each
/// pass writing them.
fn each() -> &'static [(u64, u32, Option<i32>)] {
    static EACH: OnceLock<Vec<(u64, u32, Option<i32>)>> = OnceLock::new();
    EACH.get_or_init(|| sizes(&std::env::var("XCL_DUMP_TARGETS_EACH").unwrap_or_default()))
}

fn sizes(list: &str) -> Vec<(u64, u32, Option<i32>)> {
    list.split(',').filter_map(|s| {
        let (size, format) = match s.split_once('@') {
            Some((size, f)) => (size, Some(f.trim().parse().ok()?)),
            None => (s, None),
        };
        let (w, h) = size.trim().split_once(['x', 'X'])?;
        Some((w.trim().parse().ok()?, h.trim().parse().ok()?, format))
    }).collect()
}

/// The pass just recorded on `list` wrote `writes`: those of an `XCL_DUMP_TARGETS_EACH` size
/// are copied now, behind a UAV barrier, and kept with `what` (the pass) if the list is
/// submitted in the dumped frame.
pub(super) unsafe fn after_pass(list: *mut c_void, allocator: usize, writes: &[usize], what: impl FnOnce() -> String) {
    if each().is_empty() || !recording() || writes.is_empty() {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    if allocator == 0 {
        return;
    }
    // a bundle holds neither copies nor barriers, and is never submitted on its own
    if l.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return;
    }
    let (max, budget) = (setting("XCL_DUMP_TARGETS_EACH_MAX", 48), (setting("XCL_DUMP_TARGETS_MAX_MB", 512) as u64) << 20);
    let mut what = Some(what);
    let mut label = String::new();
    let mut barrier = false;
    for &res in writes {
        let p = res as *mut c_void;
        let Some(resource) = ID3D12Resource::from_raw_borrowed(&p) else { continue };
        let d = resource.GetDesc();
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 || !each().iter().any(|&(w, h, f)| (w, h) == (d.Width, d.Height) && f.is_none_or(|f| f == d.Format.0)) {
            continue;
        }
        let mut g = RECORDED.lock().unwrap();
        let Some(r) = g.as_mut() else { return };
        let taken = r.snaps.len() + r.list_snaps.values().map(Vec::len).sum::<usize>();
        let bytes = d.Width * d.Height as u64 * bytes_per_element(d.Format).unwrap_or(16) as u64;
        if taken >= max || r.snap_bytes + bytes > budget {
            r.snaps_refused += 1;
            return;
        }
        if !barrier {
            l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }]);
            barrier = true;
        }
        let Some((buffer, fp)) = super::present::readback_copy(l, resource, &d) else { continue };
        if let Some(f) = what.take() {
            label = f();
        }
        r.snap_bytes += bytes;
        r.held.entry(res).or_insert_with(|| resource.clone());
        let pass = r.lists.get(&(list as usize)).map_or(0, Vec::len) + 1;
        r.list_snaps.entry(list as usize).or_default().push(Snap { res, what: format!("{label}, pass {pass} of its list"), bytes, gen: r.gen, allocator, src: resource.clone(), copy: (buffer, fp, d.Format) });
    }
}

// the held resources are only touched under the lock and released on the writer thread
unsafe impl Send for Recorded {}

static RECORDED: Mutex<Option<Recorded>> = Mutex::new(None);

pub(super) fn record(list: usize, op: Op) {
    if !recording() {
        return;
    }
    let mut g = RECORDED.lock().unwrap();
    let Some(r) = g.as_mut() else { return };
    for res in op.writes.iter().copied().chain(op.reads.iter().map(|x| x.resource)) {
        if res != 0 && r.held.len() < 8192 && !r.held.contains_key(&res) {
            if let Some(resource) = unsafe { ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)) } {
                r.held.insert(res, resource.clone());
            }
        }
    }
    let ops = r.lists.entry(list).or_default();
    if ops.len() < 1 << 16 {
        ops.push(op);
    }
}

/// A copy, resolve or clear into `res`.
pub(super) fn record_write(list: usize, res: usize, kind: &'static str) {
    if recording() && res != 0 {
        record(list, Op { kind, writes: vec![res], ..Default::default() });
    }
}

pub(super) fn reset(list: usize) {
    if recording() {
        if let Some(r) = RECORDED.lock().unwrap().as_mut() {
            r.lists.remove(&list);
            if let Some(snaps) = r.list_snaps.remove(&list) {
                park(snaps);
            }
        }
    }
}

pub(super) fn submitted(queue: usize, lists: &[*mut c_void]) {
    if !recording() {
        return;
    }
    let q = super::list_x::queue_id(queue);
    let mut g = RECORDED.lock().unwrap();
    let Some(r) = g.as_mut() else { return };
    for list in lists {
        if let Some(ops) = r.lists.remove(&(*list as usize)) {
            if r.frame.len() < 1 << 18 {
                r.frame.extend(ops.into_iter().map(|op| (q, op)));
            }
        }
        if let Some(snaps) = r.list_snaps.remove(&(*list as usize)) {
            r.snaps.extend(snaps);
        }
    }
}

/// A title present that XCL composes: whether it ends the recorded frame (dump now).
pub(super) fn boundary() -> bool {
    let Some(after) = after() else { return false };
    match STATE.load(Ordering::Relaxed) {
        0 if crate::log::uptime() >= after => {
            *RECORDED.lock().unwrap() = Some(Recorded::default());
            STATE.store(1, Ordering::Relaxed);
            crate::xlog_info!("target dump: recording from present {} at {:.1} s (XCL_DUMP_TARGETS_AFTER_S={after})", super::present::presents(), crate::log::uptime());
            false
        }
        1 => {
            if let Some(r) = RECORDED.lock().unwrap().as_mut() {
                r.next_frame();
            }
            STATE.store(2, Ordering::Relaxed);
            false
        }
        2 => {
            static LIGHT: AtomicU32 = AtomicU32::new(0);
            let min = setting("XCL_DUMP_TARGETS_MIN_PASSES", 100);
            let mut g = RECORDED.lock().unwrap();
            if let Some(r) = g.as_mut().filter(|r| r.frame.len() < min) {
                let n = LIGHT.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    crate::xlog_info!("target dump: present {} ends a frame of {} passes, under XCL_DUMP_TARGETS_MIN_PASSES={min}; waiting for a fuller one ({n} passed over)", super::present::presents(), r.frame.len());
                }
                r.next_frame();
                return false;
            }
            STATE.store(3, Ordering::Relaxed);
            true
        }
        _ => false,
    }
}

/// What to dump, in order: the planes, then the images the frame's last passes wrote (most
/// recent first) up to `count`, then the images those passes sample. (resource, why)
fn select(frame: &[(usize, Op)], planes: &[usize], count: usize) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = planes.iter().enumerate().map(|(k, &r)| (r, format!("plane {k}"))).collect();
    let mut first_pass = frame.len();
    let mut written = 0;
    for (i, (_, op)) in frame.iter().enumerate().rev() {
        if written >= count {
            break;
        }
        for &res in &op.writes {
            if written < count && !out.iter().any(|o| o.0 == res) {
                out.push((res, format!("written {} passes before the present", frame.len() - i)));
                written += 1;
                first_pass = i;
            }
        }
    }
    for (i, (_, op)) in frame.iter().enumerate().skip(first_pass) {
        for read in &op.reads {
            if out.len() < planes.len() + 2 * count && !out.iter().any(|o| o.0 == read.resource) {
                out.push((read.resource, format!("sampled {} passes before the present", frame.len() - i)));
            }
        }
    }
    out
}

/// The format and texel size an image's copy is shown as: the copy's own (a depth-stencil
/// resource's plane 0 is R32 or R24X8, not the resource's 8-byte format), when every row's texels
/// lie inside the row pitch and the copy holds every row; None for formats `texel_rgb` cannot
/// show and for block-compressed ones.
fn shown_as(copy: DXGI_FORMAT, resource: DXGI_FORMAT, w: usize, h: usize, pitch: usize, len: usize) -> Option<(DXGI_FORMAT, usize)> {
    let f = if copy.0 != 0 { copy } else { resource };
    let b = bytes_per_element(f)? as usize;
    (b * w <= pitch && pitch * h <= len && len >= b && texel_rgb(f, &vec![0u8; b]).is_some()).then_some((f, b))
}

fn bytes_per_element(f: DXGI_FORMAT) -> Option<u32> {
    Some(match f.0 {
        1..=4 => 16,
        5..=8 => 12,
        9..=22 => 8,
        23..=47 | 67 | 87..=93 => 4,
        48..=59 | 85 | 86 | 115 => 2,
        60..=65 => 1,
        _ => return None,
    })
}

fn half(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let v = match e {
        0 => m as f32 / 1024.0 / 16384.0,
        31 => if m == 0 { f32::INFINITY } else { 0.0 },
        _ => f32::from_bits((e + 112) << 23 | m << 13),
    };
    if s == 1 { -v } else { v }
}

fn tone(v: f32) -> u8 {
    let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
    ((v / (1.0 + v)).sqrt() * 255.0) as u8
}

/// One texel as 8-bit RGB for viewing (HDR tone-mapped; one channel as grey); None: not shown.
fn texel_rgb(f: DXGI_FORMAT, t: &[u8]) -> Option<[u8; 3]> {
    let u16_at = |k: usize| u16::from_le_bytes([t[2 * k], t[2 * k + 1]]);
    let f32_at = |k: usize| f32::from_le_bytes(t[4 * k..4 * k + 4].try_into().unwrap());
    let u32_0 = || u32::from_le_bytes(t[0..4].try_into().unwrap());
    Some(match f.0 {
        2 | 1 => [tone(f32_at(0)), tone(f32_at(1)), tone(f32_at(2))],
        9 | 10 => [tone(half(u16_at(0))), tone(half(u16_at(1))), tone(half(u16_at(2)))],
        11 => [(u16_at(0) >> 8) as u8, (u16_at(1) >> 8) as u8, (u16_at(2) >> 8) as u8],
        16 | 15 => [tone(f32_at(0)), tone(f32_at(1)), 0],
        33 | 34 => [tone(half(u16_at(0))), tone(half(u16_at(1))), 0],
        35 => [(u16_at(0) >> 8) as u8, (u16_at(1) >> 8) as u8, 0],
        39..=41 => [tone(f32_at(0)); 3],
        44..=46 => [((u32_0() & 0xff_ffff) >> 16) as u8; 3],
        19..=21 => [tone(f32_at(0)); 3],
        54 => [tone(half(u16_at(0))); 3],
        53 | 55 | 56 => [(u16_at(0) >> 8) as u8; 3],
        48 | 49 => [t[0], t[1], 0],
        60 | 61 => [t[0]; 3],
        27 => [t[0], t[1], t[2]],
        23 => [(u32_0() >> 2 & 0xff) as u8, (u32_0() >> 12 & 0xff) as u8, (u32_0() >> 22 & 0xff) as u8],
        _ if t.len() == 4 => return super::present::rgb(f, u32_0()),
        _ => return None,
    })
}

fn bmp(width: usize, height: usize, mut pixel: impl FnMut(usize, usize) -> [u8; 3]) -> Vec<u8> {
    let row = (width * 3).div_ceil(4) * 4;
    let mut out = Vec::with_capacity(54 + row * height);
    out.extend_from_slice(b"BM");
    for v in [(54 + row * height) as u32, 0, 54, 40, width as u32, height as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in (0..height).rev() {
        let start = out.len();
        for x in 0..width {
            let [r, g, b] = pixel(x, y);
            out.extend_from_slice(&[b, g, r]);
        }
        out.resize(start + row, 0);
    }
    out
}

/// The console's description of `res` (from its creation) and how the console lays it out.
unsafe fn console_side(res: &ID3D12Resource, host: &D3D12_RESOURCE_DESC, host_row: u64) -> (String, Option<u64>) {
    let Ok(g) = crate::gpu::texture::image_bridge::GuestImage::get(res) else { return ("no console description (not a title placement)".into(), None) };
    let d = &g.desc;
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let w16 = |o: usize| u16::from_le_bytes(d[o..o + 2].try_into().unwrap());
    let (width, height, format, layout, flags) = (u64::from_le_bytes(d[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x2C), u(0x30));
    let mut s = format!("console at {:#x}: {width}x{height} format {format} array {} mips {} layout {layout:#x} flags {flags:#x}", g.va, w16(0x1C), w16(0x1E));
    let mut differs = Vec::new();
    if width != host.Width || height != host.Height {
        differs.push(format!("size {width}x{height} vs PC {}x{}", host.Width, host.Height));
    }
    if format != host.Format.0 as u32 {
        differs.push(format!("format {format} vs PC {}", host.Format.0));
    }
    match crate::hle::xg::Texture::new(d) {
        Ok(t) => match t.texels() {
            Some(sf) => {
                let level = &sf.chain.levels[0];
                let row = level.pitch as u64 * sf.bpe as u64;
                let _ = write!(s, "; console layout: mode {} {} B a texel, level 0 {}x{} laid out {} texels wide ({row} B a row), {} B in all", sf.mode, sf.bpe, level.width, level.height, level.pitch, t.size());
                if let Some(pc) = bytes_per_element(host.Format).filter(|&b| b != sf.bpe) {
                    differs.push(format!("{} B a texel vs PC {pc}", sf.bpe));
                }
                if row != host_row {
                    differs.push(format!("console row {row} B vs PC row pitch {host_row} B"));
                }
            }
            None => s.push_str("; console layout: no texel plane"),
        },
        Err(hr) => {
            let _ = write!(s, "; console layout: none ({:#x})", hr);
        }
    }
    if !differs.is_empty() {
        let _ = write!(s, "; DIFFERS: {}", differs.join(", "));
    }
    (s, Some(g.va))
}

fn pass(frame: &[(usize, Op)], i: usize) -> String {
    let (q, op) = &frame[i];
    let mut s = format!("pass -{} q{q} {}", frame.len() - i, op.kind);
    if op.pso != 0 {
        let _ = write!(s, " pso {:#x} {}", op.pso, op.shaders);
    }
    if let Some((v, n)) = op.viewport {
        let _ = write!(s, ", viewport ({}, {}) {}x{}{}", v[0], v[1], v[2], v[3], if n > 1 { format!(" (+{} more)", n - 1) } else { String::new() });
    }
    if let Some((r, n)) = op.scissor {
        let _ = write!(s, ", scissor ({}, {})..({}, {}){}", r[0], r[1], r[2], r[3], if n > 1 { format!(" (+{} more)", n - 1) } else { String::new() });
    }
    s
}

/// The log lines for one dumped image: who wrote it and who sampled it, with the size they
/// sampled it at (the last three of each).
fn usage(frame: &[(usize, Op)], res: usize) -> Vec<String> {
    let mut out = Vec::new();
    let writers: Vec<usize> = frame.iter().enumerate().rev().filter(|(_, (_, op))| op.writes.contains(&res)).map(|(i, _)| i).take(3).collect();
    let total = frame.iter().filter(|(_, op)| op.writes.contains(&res)).count();
    for &i in &writers {
        out.push(format!("written by {}", pass(frame, i)));
    }
    if total > writers.len() {
        out.push(format!("({} earlier writes in the frame)", total - writers.len()));
    }
    let readers: Vec<(usize, &Read)> = frame.iter().enumerate().rev().filter_map(|(i, (_, op))| op.reads.iter().find(|r| r.resource == res).map(|r| (i, r))).take(3).collect();
    for (i, r) in readers {
        out.push(format!("sampled as {}x{} gfx10-fmt {} tile {} mip {} ({}) by {}", r.width, r.height, r.format, r.swizzle, r.base_level, r.slot, pass(frame, i)));
    }
    if writers.is_empty() && !frame.iter().any(|(_, op)| op.reads.iter().any(|r| r.resource == res)) {
        out.push("no recorded pass of the frame wrote or sampled it".into());
    }
    out
}

struct Item {
    n: usize,
    copy: Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, DXGI_FORMAT)>,
    /// a copy after a pass: its list's allocator and the image, so the buffer is parked, not
    /// dropped, once written (`park`)
    after_pass: Option<(usize, ID3D12Resource)>,
}

pub(super) struct Pending {
    frame: u64,
    items: Vec<Item>,
    _held: HashMap<usize, ID3D12Resource>,
    /// the presenter's fence passed and the files are written: what it holds may go
    done: bool,
}

/// A dump dropped before its presenter fence passed (the present failed after `capture`, or the
/// copies did not finish in 60 s) keeps its buffers and images for good: a copy may still run.
impl Drop for Pending {
    fn drop(&mut self) {
        if !self.done {
            std::mem::forget(std::mem::take(&mut self.items));
            std::mem::forget(std::mem::take(&mut self._held));
        }
    }
}

unsafe impl Send for Pending {}

/// The present that ends the recorded frame: describes what `select` picks and records its
/// copies on `list` (the presenter's, before it is closed). None when nothing was recorded.
pub(super) unsafe fn capture(list: &ID3D12GraphicsCommandList, planes: &[ID3D12Resource]) -> Option<Pending> {
    let recorded = RECORDED.lock().unwrap().take()?;
    let frame_no = super::present::presents();
    let plane_ids: Vec<usize> = planes.iter().map(|p| p.as_raw() as usize).collect();
    let picked = select(&recorded.frame, &plane_ids, setting("XCL_DUMP_TARGETS_COUNT", 32).clamp(1, 256));
    let budget = (setting("XCL_DUMP_TARGETS_MAX_MB", 512) as u64) << 20;
    crate::xlog_info!(
        "target dump: frame {frame_no} at {:.1} s: {} passes recorded, {} images to dump, copies bounded at {} MiB, files in targets_{frame_no}",
        crate::log::uptime(), recorded.frame.len(), picked.len(), budget >> 20
    );
    let mut used = 0u64;
    let mut addresses: Vec<(u64, usize)> = Vec::new();
    let mut items = Vec::new();
    for (n, (res, why)) in picked.into_iter().enumerate() {
        let resource = planes.iter().find(|p| p.as_raw() as usize == res).or_else(|| recorded.held.get(&res)).cloned();
        let Some(resource) = resource else {
            crate::xlog_info!("target dump: #{n} {res:#x} ({why}): not held, not described");
            continue;
        };
        let d = resource.GetDesc();
        let dev: Option<ID3D12Device> = { let mut dv = None; resource.GetDevice(&mut dv).ok().and(dv) };
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        if let Some(dev) = &dev {
            dev.GetCopyableFootprints(&d, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        }
        let (console, va) = console_side(&resource, &d, fp.Footprint.RowPitch as u64);
        let same: Vec<String> = va.map_or_else(Vec::new, |va| addresses.iter().filter(|(a, _)| *a == va).map(|(_, k)| format!("#{k}")).collect());
        if let Some(va) = va {
            addresses.push((va, n));
        }
        crate::xlog_info!(
            "target dump: #{n} {res:#x} ({why}): PC {}x{} format {} mips {} array {} flags {:#x}, row pitch {} B for {} B of texels; {console}{}",
            d.Width, d.Height, d.Format.0, d.MipLevels, d.DepthOrArraySize, d.Flags.0, fp.Footprint.RowPitch,
            bytes_per_element(d.Format).map_or(0, |b| b as u64 * d.Width),
            if same.is_empty() { String::new() } else { format!("; same title address as {}", same.join(", ")) }
        );
        for line in usage(&recorded.frame, res) {
            crate::xlog_info!("target dump: #{n} {line}");
        }
        let copyable = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count == 1 && total != u64::MAX && total != 0;
        let copy = if !copyable {
            crate::xlog_info!("target dump: #{n} not copied (dimension {}, {} samples)", d.Dimension.0, d.SampleDesc.Count);
            None
        } else if used + total > budget {
            crate::xlog_info!("target dump: #{n} not copied: {} MiB more would pass XCL_DUMP_TARGETS_MAX_MB", total.div_ceil(1 << 20));
            None
        } else {
            used += total;
            super::present::readback_copy(list, &resource, &d).map(|(b, f)| (b, f, d.Format))
        };
        items.push(Item { n, copy, after_pass: None });
    }
    crate::xlog_info!(
        "target dump: {} copies after passes kept, {} made on lists the frame did not submit, {} dropped as their lists stayed unsubmitted a whole frame, {} refused for XCL_DUMP_TARGETS_EACH_MAX or the budget",
        recorded.snaps.len(), recorded.list_snaps.values().map(Vec::len).sum::<usize>(), recorded.snaps_stale, recorded.snaps_refused
    );
    park(recorded.list_snaps.into_values().flatten());
    let first = items.last().map_or(0, |i| i.n + 1);
    let mut step: HashMap<usize, usize> = HashMap::new();
    for (k, s) in recorded.snaps.into_iter().enumerate() {
        let n = first + k;
        let fp = s.copy.1.Footprint;
        let at = step.entry(s.res).or_insert(0);
        *at += 1;
        crate::xlog_info!(
            "target dump: #{n} {:#x} after pass (copy {} of it in the frame): {}: PC {}x{} format {}, row pitch {} B",
            s.res, at, s.what, fp.Width, fp.Height, s.copy.2.0, fp.RowPitch
        );
        items.push(Item { n, copy: Some(s.copy), after_pass: Some((s.allocator, s.src)) });
    }
    Some(Pending { frame: frame_no, items, _held: recorded.held, done: false })
}

/// Writes the copies once `fence` reaches `value` (the presenter's signal after them), on a
/// thread of its own.
pub(super) fn write_after(p: Pending, fence: ID3D12Fence, value: u64) {
    struct Job(Pending, ID3D12Fence);
    unsafe impl Send for Job {}
    let job = Job(p, fence);
    let _ = std::thread::Builder::new().name("xcl-target-dump".into()).spawn(move || {
        let Job(mut p, fence) = job;
        let start = std::time::Instant::now();
        while unsafe { fence.GetCompletedValue() } < value {
            if start.elapsed() > std::time::Duration::from_secs(60) {
                crate::xlog_warn!("target dump: frame {}: the copies did not finish in 60 s; nothing written, and the copies' buffers are kept for good", p.frame);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        write_items(&p);
        let mut parked = PARKED.lock().unwrap();
        for item in std::mem::take(&mut p.items) {
            if let (Some((buffer, ..)), Some((allocator, src))) = (item.copy, item.after_pass) {
                parked.0.push((allocator, buffer));
                parked.0.push((allocator, src));
            }
        }
        p.done = true;
    });
}

/// The files of a dump whose presenter fence has passed.
fn write_items(p: &Pending) {
    let Some(boot) = crate::state::boot() else { return };
    let dir = boot.logs_dir().join(format!("targets_{}", p.frame));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::xlog_warn!("target dump: {}: {e}", dir.display());
        return;
    }
    for item in &p.items {
        let Some((buffer, fp, format)) = &item.copy else { continue };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if unsafe { buffer.Map(0, None, Some(&mut ptr)) }.is_err() || ptr.is_null() {
            continue;
        }
        let (w, h, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
        let len = ((pitch * h) as u64).min(unsafe { buffer.GetDesc() }.Width.saturating_sub(fp.Offset)) as usize;
        let rows = unsafe { std::slice::from_raw_parts((ptr as *const u8).add(fp.Offset as usize), len) };
        let _ = std::fs::write(dir.join(format!("{:02}.raw", item.n)), rows);
        let zero = rows.iter().all(|&b| b == 0);
        let shown = shown_as(fp.Footprint.Format, *format, w, h, pitch, len);
        if let Some((f, b)) = shown {
            let _ = std::fs::write(dir.join(format!("{:02}.bmp", item.n)), bmp(w, h, |x, y| texel_rgb(f, &rows[y * pitch + x * b..y * pitch + x * b + b]).unwrap_or([0; 3])));
        }
        unsafe { buffer.Unmap(0, None) };
        crate::xlog_info!(
            "target dump: #{} written: {:02}.raw ({w}x{h}, row pitch {pitch} B, {len} B{}){}",
            item.n, item.n, if zero { ", all zero" } else { "" },
            if shown.is_some() { format!(", {:02}.bmp", item.n) } else { " (format not shown as an image)".into() }
        );
    }
    crate::xlog_info!("target dump: frame {} written to {}", p.frame, dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(kind: &'static str, writes: &[usize], reads: &[usize]) -> (usize, Op) {
        let reads = reads.iter().map(|&resource| Read { resource, width: 960, height: 540, format: 10, swizzle: 27, base_level: 0, slot: "t0".into() }).collect();
        (1, Op { kind, pso: 0x77, shaders: "ps 1".into(), writes: writes.to_vec(), reads, viewport: Some(([0.0, 0.0, 960.0, 540.0], 1)), scissor: Some(([0, 0, 960, 540], 1)) })
    }

    /// The planes first, then the most recent writes up to the count, then what those passes
    /// sample; each image once.
    #[test]
    fn the_dump_takes_the_planes_the_last_writes_and_their_inputs() {
        let frame = [op("draw", &[0xA], &[]), op("dispatch", &[0xB], &[0xA]), op("draw", &[0xC], &[0xB, 0xD]), op("draw", &[0xF], &[0xC, 0xA])];
        let picked: Vec<usize> = select(&frame, &[0xF], 2).into_iter().map(|p| p.0).collect();
        assert_eq!(picked, [0xF, 0xC, 0xB, 0xA, 0xD]);
        let why = select(&frame, &[], 1);
        assert_eq!(why[0], (0xF, "written 1 passes before the present".into()));
        assert_eq!(why[1].0, 0xC, "what the last pass samples");
    }

    /// A pass is named with its viewport and scissor, and a reader with the size it samples at.
    #[test]
    fn usage_names_writers_with_their_viewport_and_readers_with_their_sample_size() {
        let frame = [op("draw", &[0xA], &[]), op("draw", &[0xF], &[0xA])];
        let lines = usage(&frame, 0xA);
        assert_eq!(lines[0], "written by pass -2 q1 draw pso 0x77 ps 1, viewport (0, 0) 960x540, scissor (0, 0)..(960, 540)");
        assert!(lines[1].starts_with("sampled as 960x540 gfx10-fmt 10 tile 27 mip 0 (t0) by pass -1 q1 draw"));
        assert_eq!(usage(&frame, 0xE), ["no recorded pass of the frame wrote or sampled it"]);
    }

    #[test]
    fn a_depth_copy_is_shown_by_its_plane_format() {
        assert_eq!(shown_as(DXGI_FORMAT(39), DXGI_FORMAT(19), 1280, 720, 5120, 5120 * 720), Some((DXGI_FORMAT(39), 4)));
        assert_eq!(shown_as(DXGI_FORMAT(0), DXGI_FORMAT(19), 1280, 720, 5120, 5120 * 720), None, "8-byte texels do not fit a 5120 B row");
        assert_eq!(shown_as(DXGI_FORMAT(71), DXGI_FORMAT(71), 1024, 256, 2048, 2048 * 64), None, "block-compressed");
        assert_eq!(shown_as(DXGI_FORMAT(28), DXGI_FORMAT(28), 64, 64, 256, 256 * 63), None, "a copy short of its rows");
    }

    #[test]
    fn each_takes_a_list_of_sizes() {
        assert_eq!(sizes("1280x720, 640X360,bad,1920x"), [(1280, 720, None), (640, 360, None)]);
        assert_eq!(sizes("1280x720@26,512x512@x"), [(1280, 720, Some(26))]);
        assert!(sizes("").is_empty());
    }

    #[test]
    fn texels_are_shown_for_float_and_unorm_formats() {
        let one = 0x3c00u16.to_le_bytes();
        assert_eq!(texel_rgb(DXGI_FORMAT_R16G16B16A16_FLOAT, &[one, one, [0, 0], one].concat()), Some([180, 180, 0]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R8G8B8A8_UNORM, &[1, 2, 3, 4]), Some([1, 2, 3]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R32_FLOAT, &1.0f32.to_le_bytes()), Some([180; 3]));
        assert_eq!(texel_rgb(DXGI_FORMAT_BC1_UNORM, &[0; 8]), None);
        assert_eq!(half(0xc000), -2.0);
        let image = bmp(3, 2, |x, y| [x as u8, y as u8, 9]);
        assert_eq!((image.len(), &image[54..57]), (54 + 12 * 2, &[9u8, 1, 0][..]), "rows bottom-up, padded to 4 bytes, BGR");
    }
}
