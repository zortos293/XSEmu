// SPDX-License-Identifier: GPL-3.0-or-later
//
// GPL-3 PROVENANCE. read_back (the footprints with their row-end padding, the copy of every
// subresource into one READBACK buffer) and land (keeping the title bytes the subresources do
// not cover, tiling each mapped subresource into the console layout) are adapted from CGBJ
// (github.com/cputroll/cgbj-reupload at f4b48ad766ddbec1a1867022cd42f7645b0dd55e,
// crates/victory-gpu/src/texback.rs, functions record and land), which takes them from
// fxceless/victory-fork f5c03f3b (pull request 999: Geckomacabre 895c0b8f, 04df4350).
// Copyright (C) the VIctory contributors (Marvisak/VIctory and its forks).
// This file is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3 of
// the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY;
// see <https://www.gnu.org/licenses/gpl-3.0.html>.
//
//! Native images the GPU wrote, written back to the guest's physical pages when guest memory is
//! read (ZOR-603). A console title reads what its GPU stored into a texture straight from memory
//! (a water height map that boats float on); here the native resource holds those texels and
//! the title's pages keep what was there before.
//!
//! Authority: a submission wrote an image when one of its lists recorded a write into it: a
//! render target or writable depth bound for a draw, dispatch, ExecuteIndirect or bundle, a
//! clear, a render pass, a copy, resolve or UAV clear into it, or shader work while it allows
//! unordered access and is in a UAV state (states tracked from the creation state through every
//! submitted transition, as for uploads). A list with an attachment that cannot be told takes
//! the state rule instead: a subresource in COMMON or a write state counts. Then the image's
//! pages are captured
//! (kernelx::snapshot_queue_sources: the physical pages behind its address at that submission,
//! never resolved again) and its address and every other address showing those pages are
//! guarded (no access). A CPU access there, or a bulk read of XCL (memmap::copy_title), writes
//! it back first: a private queue waits (GPU-side) on the fence signalled after the last
//! submission that touched the image, copies it into a read-back buffer, and the texels are
//! tiled through the console layout into the captured pages through their own section views.
//!
//! Off unless XCL_IMAGE_WRITEBACK=1. What is refused stays play81 behaviour and is counted
//! ("D3D12.X: image writeback: <class> (N so far)"): compression metadata planes (CMASK,
//! HTILE), multisampled and converted formats, the packed D24 depth plane, enhanced-barrier
//! (layout) images, shapes the layout does not match, pages with a queue-ordered remap pending.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::core::{Interface, Owned};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::CreateEventW;
use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE};

use crate::gpu::texture::image_bridge::{classes, GuestImage};
use crate::hle::kernelx::{self, QueuePageSource, PHYS_PAGE};
use crate::hle::xg;

#[derive(Clone, Copy)]
enum Count {
    Registered,
    Superseded,
    Layout,
    NoLayout,
    Shape,
    Metadata,
    Msaa,
    Converted,
    Packing,
    PlaneUsage,
    Format,
    PlaneLeft,
    Guarded,
    Source,
    RemapPending,
    Flushed,
    Released,
    NotRetired,
    Failed,
    Hole,
    Racing,
    Unchanged,
    Untracked,
    Own,
    NotGuarded,
}

const NAMES: [&str; 25] = [
    "images registered",
    "images superseded by a newer placement",
    "refused: enhanced-barrier layout image",
    "refused: no console layout",
    "refused: native shape differs from the console layout",
    "refused: compression metadata plane (CMASK/HTILE, play81 behaviour)",
    "refused: multisampled",
    "refused: converted format",
    "refused: packed D24 depth plane",
    "refused: plane usage",
    "refused: format not identical to the native one",
    "images with a plane left on play81 behaviour",
    "writes guarded",
    "write not captured: page source unknown",
    "write not captured: queue-ordered remap pending",
    "images written back",
    "write-back skipped: image released",
    "write-back skipped: writer not retired in time",
    "write-back failed",
    "pages skipped: hole",
    "write-back raced a newer write",
    "write-backs with no texel changed since the last: the CPU's bytes kept",
    "submissions with an untracked attachment: images in a write state taken as written",
    "guards left open without a write-back: XCL's own thread",
    "writes not guarded: the image is over the guard size or count budget",
];

static COUNTS: [AtomicU64; 25] = [const { AtomicU64::new(0) }; 25];

#[cfg(test)]
thread_local! {
    /// what this thread counted: the tests compare deltas while three other tests submit in the
    /// same process and the product's counters are shared
    static TEST_COUNTS: std::cell::RefCell<[u64; 23]> = const { std::cell::RefCell::new([0; 23]) };
    /// submissions this thread reported, so a test can tell whether the queue's own wrapper did it
    static REPORTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn count(c: Count) {
    let n = COUNTS[c as usize].fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(test)]
    TEST_COUNTS.with(|t| t.borrow_mut()[c as usize] += 1);
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: image writeback: {} ({n} so far)", NAMES[c as usize]);
    }
}

fn refused(e: classes::Unsupported) -> Count {
    use classes::Unsupported::*;
    match e {
        InvalidSampleCount | Msaa => Count::Msaa,
        Cmask | Htile => Count::Metadata,
        ConvertedFormat => Count::Converted,
        DepthStencilPacking => Count::Packing,
        PlaneUsage => Count::PlaneUsage,
        Format => Count::Format,
    }
}

/// 0 unknown, 1 off, 2 on
static ENABLED: AtomicU8 = AtomicU8::new(0);

fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("XCL_IMAGE_WRITEBACK").is_ok_and(|v| v.trim() == "1");
            ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            if on {
                crate::xlog_info!("D3D12.X: image writeback: on (XCL_IMAGE_WRITEBACK=1)");
            }
            on
        }
        v => v == 2,
    }
}

/// A native plane written back and the console plane it lands in.
struct Part {
    plane: u32,
    guest: u32,
}

struct Clock {
    fence: ID3D12Fence,
    value: AtomicU64,
}

struct Image {
    va: usize,
    size: usize,
    tex: Arc<xg::Texture>,
    parts: Arc<Vec<Part>>,
    mips: u32,
    layers: u32,
    states: Vec<u32>,
    last: Option<(Arc<Clock>, u64)>,
    dirty: Option<Arc<Vec<QueuePageSource>>>,
    generation: u64,
    /// the CPU has read it back: guarded whatever its size from then on
    read: bool,
    guards: Vec<(usize, usize, u32)>,
    /// digest of every subresource as last landed (subresources order); empty: never landed
    landed: Arc<Vec<u64>>,
    /// native subresources wholly overwritten by a tracked write since the last landing
    full: BTreeSet<u32>,
    /// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS
    uav: bool,
    simultaneous: bool,
}

// D3D12 objects are free-threaded
unsafe impl Send for Image {}
unsafe impl Send for Clock {}
unsafe impl Sync for Clock {}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static IMAGES: Mutex<BTreeMap<usize, Image>> = Mutex::new(BTreeMap::new());
/// guarded range start -> (end, image); the ranges never overlap
static GUARDED: RwLock<BTreeMap<usize, (usize, usize)>> = RwLock::new(BTreeMap::new());
static LIVE: AtomicUsize = AtomicUsize::new(0);
static CLOCKS: Mutex<BTreeMap<usize, Arc<Clock>>> = Mutex::new(BTreeMap::new());
static FLUSH: Mutex<BTreeMap<usize, Context>> = Mutex::new(BTreeMap::new());

thread_local! {
    /// This thread is making a write-back: a page it touches must not start another one.
    static IN_FLUSH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// This thread is inside submitted, where the images lock is held: a page it touches must
    /// not be flushed from here, because a flush wants that lock.
    static IN_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Guest bytes of images guarded at most in all (XCL_WB_GUARD_BUDGET_MB, 64 MiB)
static GUARDED_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Native images guarded at most in all (XCL_WB_GUARD_MAX, 4096)
static GUARDED_IMAGES: AtomicUsize = AtomicUsize::new(0);
/// Submissions that took 250 ms or more (counted, logged)
static SLOW_SUBMITS: AtomicU64 = AtomicU64::new(0);

fn env_usize(name: &str, default: usize) -> usize {
    static VALUES: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    if let Some(&v) = VALUES.lock().unwrap().get(name) {
        return v;
    }
    let v = std::env::var(name).ok().and_then(|t| t.trim().parse::<usize>().ok()).unwrap_or(default);
    VALUES.lock().unwrap().insert(name.to_string(), v);
    v
}

/// Guest bytes one image may take to be guarded (XCL_WB_GUARD_MAX_MB, 4 MiB). A title reads a
/// small feedback texture back far more often than a material plane, and a guarded page the
/// title never touches costs a protection change for nothing.
fn guard_max_bytes() -> usize {
    env_usize("XCL_WB_GUARD_MAX_MB", 4) << 20
}

/// Guest bytes guarded in all (XCL_WB_GUARD_BUDGET_MB, 64 MiB). Over it, further images keep
/// play81 behaviour until the guarded ones are read and lifted.
fn guard_budget_bytes() -> usize {
    env_usize("XCL_WB_GUARD_BUDGET_MB", 64) << 20
}

/// Images guarded in all (XCL_WB_GUARD_MAX, 4096)
fn guard_max_images() -> usize {
    env_usize("XCL_WB_GUARD_MAX", 4096)
}

/// This thread is inside submitted: no page it touches may start a write-back (a flush wants the
/// images lock, which submitted holds), and a submission that takes long is logged.
struct Submission {
    began: std::time::Instant,
}

impl Submission {
    fn enter() -> Self {
        IN_SUBMIT.with(|c| c.set(true));
        Submission { began: std::time::Instant::now() }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        IN_SUBMIT.with(|c| c.set(false));
        let ms = self.began.elapsed().as_millis();
        if ms >= 250 {
            let n = SLOW_SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                crate::xlog_info!(
                    "D3D12.X: image writeback: a submission took {ms} ms with {} images guarded ({} slow submissions)",
                    GUARDED_IMAGES.load(Ordering::Relaxed),
                    n
                );
            }
        }
    }
}

/// Leaves the guarded pages at [at, end) open without writing them back, because this thread
/// holds a lock a write-back wants (its own code inside a submission, or a write-back already).
/// The image is guarded again at its next capture.
unsafe fn open_pages(at: usize, end: usize) -> bool {
    let mut g = GUARDED.write().unwrap_or_else(|e| e.into_inner());
    let hit: Vec<(usize, usize)> = g.range(..end).rev().take_while(|(_, &(e, _))| e > at).map(|(&s, &(e, _))| (s, e)).collect();
    if hit.is_empty() {
        return false;
    }
    for (s, e) in hit {
        let mut old = 0;
        VirtualProtect(s as *const c_void, e - s, PAGE_READWRITE, &mut old);
        g.remove(&s);
        LIVE.fetch_sub(1, Ordering::Release);
        count(Count::Own);
    }
    true
}

const COMMON: u32 = 0;
const COPY_SOURCE: u32 = 0x800;
/// RENDER_TARGET, UNORDERED_ACCESS, DEPTH_WRITE, STREAM_OUT, COPY_DEST, RESOLVE_DEST and the
/// video write states; COMMON counts too (a copy promotes a texture from it)
const WRITES: u32 = 0x4 | 0x8 | 0x10 | 0x100 | 0x400 | 0x1000 | 0x2_0000 | 0x8_0000 | 0x80_0000;
const ALL: u32 = u32::MAX;
const WAIT_MS: u32 = 2_000;

fn writable(state: u32) -> bool {
    state == COMMON || state & WRITES != 0
}

fn planes_of(format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> u32 {
    if matches!(format.0, 19..=22 | 44..=47) {
        2
    } else {
        1
    }
}

/// Whether native image res is written back (for title_textures::note_transitions).
pub(super) fn tracks(res: usize) -> bool {
    ACTIVE.load(Ordering::Relaxed) && IMAGES.lock().unwrap().contains_key(&res)
}

pub(super) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Native image native was made for image (tagged): written back from here on when the console
/// layout and the native resource hold the same texels.
pub(super) unsafe fn register(native: *mut c_void, image: &GuestImage) {
    if !enabled() {
        return;
    }
    let key = native as usize;
    if image.initial[0] != 0 {
        forget(key);
        count(Count::Layout);
        return;
    }
    let Some(res) = ID3D12Resource::from_raw_borrowed(&native) else { return };
    let d = &image.desc;
    let word = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let (format, samples, flags) = (word(0x20), word(0x24), word(0x30));
    let nd = res.GetDesc();
    let Ok(tex) = xg::Texture::new(d) else {
        count(Count::NoLayout);
        return;
    };
    let size = tex.size();
    if size == 0 || size == u64::MAX || tex.has_packed_volume_mips() {
        count(Count::NoLayout);
        return;
    }
    let three_d = nd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let layers = if three_d { 1 } else { u32::from(nd.DepthOrArraySize) };
    let mut parts = Vec::new();
    let mut left = false;
    for (i, p) in tex.planes.iter().enumerate() {
        match &p.kind {
            xg::PlaneKind::Meta { .. } => {
                count(Count::Metadata);
                return;
            }
            xg::PlaneKind::Blank => {}
            xg::PlaneKind::Surface(s) => match classes::classify(format, nd.Format.0 as u32, flags, samples, p.usage) {
                Ok(plane) => {
                    let slices = if three_d { s.chain.layers() } else { s.array as usize };
                    let shape = nd.SampleDesc.Count == 1
                        && s.chain.levels.len() == usize::from(nd.MipLevels)
                        && if three_d { s.depth == u32::from(nd.DepthOrArraySize) } else { s.depth == 0 && s.array == layers }
                        && plane < planes_of(nd.Format)
                        && p.offset as usize + s.chain.slice_bytes * slices <= size as usize;
                    if shape {
                        parts.push(Part { plane, guest: i as u32 });
                    } else {
                        count(Count::Shape);
                        left = true;
                    }
                }
                Err(e) => {
                    count(refused(e));
                    left = true;
                }
            },
        }
    }
    if parts.is_empty() {
        return;
    }
    if left {
        count(Count::PlaneLeft);
    }
    let mut state = image.initial[1] & super::real::PC_RESOURCE_STATES;
    if flags & 2 != 0 {
        state = super::depth_uav::pc_state(state);
    }
    let subs = u32::from(nd.MipLevels) * layers * planes_of(nd.Format);
    let va = image.va as usize;
    let end = va + size as usize;
    let mut g = IMAGES.lock().unwrap();
    let superseded: Vec<usize> = g.iter().filter(|(&k, i)| k != key && i.va < end && va < i.va + i.size).map(|(&k, _)| k).collect();
    for k in superseded {
        if let Some(mut old) = g.remove(&k) {
            unguard(&mut old);
            count(Count::Superseded);
        }
    }
    let made = Image {
        va,
        size: size as usize,
        tex: Arc::new(tex),
        parts: Arc::new(parts),
        mips: u32::from(nd.MipLevels),
        layers,
        states: vec![state; subs as usize],
        last: None,
        dirty: None,
        generation: 0,
        read: false,
        guards: Vec::new(),
        landed: Arc::new(Vec::new()),
        full: BTreeSet::new(),
        uav: nd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0,
        simultaneous: nd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0 != 0,
    };
    if let Some(mut old) = g.insert(key, made) {
        unguard(&mut old);
    }
    ACTIVE.store(true, Ordering::Relaxed);
    count(Count::Registered);
}

/// Native image res is released.
pub(super) fn forget(res: usize) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(mut img) = IMAGES.lock().unwrap().remove(&res) {
        unsafe { unguard(&mut img) };
    }
}

fn apply(states: &mut [u32], sub: u32, after: u32) {
    if sub == ALL {
        states.fill(after);
    } else if let Some(s) = states.get_mut(sub as usize) {
        *s = after;
    }
}

unsafe fn clock(queue: &ID3D12CommandQueue) -> Option<Arc<Clock>> {
    let mut g = CLOCKS.lock().unwrap();
    if let Some(c) = g.get(&(queue.as_raw() as usize)) {
        return Some(Arc::clone(c));
    }
    let mut dev: Option<ID3D12Device> = None;
    queue.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let c = Arc::new(Clock { fence, value: AtomicU64::new(0) });
    g.insert(queue.as_raw() as usize, Arc::clone(&c));
    Some(c)
}

/// What a PC list's recorded calls write, from its recording to its reset: images written
/// (native resource -> its subresources wholly overwritten), the render targets bound now,
/// whether it ran shader work (a UAV image in a UAV state may be written by it), and whether an
/// attachment could not be told (the state rule then decides).
#[derive(Default)]
struct ListWrites {
    images: BTreeMap<usize, BTreeSet<u32>>,
    bound: Vec<usize>,
    bound_noted: bool,
    work: bool,
    untracked: bool,
}

static LISTS: Mutex<BTreeMap<usize, ListWrites>> = Mutex::new(BTreeMap::new());
/// the attachment sink could not be installed: every submission takes the state rule
static NO_ATTACHMENTS: AtomicBool = AtomicBool::new(false);

/// At device creation: with write-back on, attachments (render-target and depth binds, clears,
/// render passes) are reported per list from here on.
pub(super) fn init() {
    if enabled() && super::real::image_views::install(attachments).is_err() {
        NO_ATTACHMENTS.store(true, Ordering::Relaxed);
        crate::xlog_warn!("D3D12.X: image writeback: attachment events taken already; every submission takes the state rule");
    }
}

/// Whether write-back is on (for installing the list hooks).
pub(super) fn on() -> bool {
    enabled()
}

fn with_list(list: usize, f: impl FnOnce(&mut ListWrites)) {
    if ACTIVE.load(Ordering::Relaxed) {
        f(LISTS.lock().unwrap().entry(list).or_default());
    }
}

fn whole_subs(view: &super::real::image_views::View, plane: u32) -> impl Iterator<Item = u32> + '_ {
    view.subs(plane).filter(|_| view.covers_full_subresources())
}

fn attachments(list: usize, event: Result<super::real::image_views::Event, super::real::image_views::Error>) {
    use super::real::image_views::{Clear, Event, Store};
    with_list(list, |w| match event {
        Ok(Event::Bind { colors, depth }) => {
            let depth = depth.filter(|d| !(d.slice.read_only_depth && (d.slice.read_only_stencil || !d.has_stencil())));
            w.bound = colors.iter().chain(&depth).map(|v| v.resource.as_raw() as usize).collect();
            w.bound_noted = false;
        }
        Ok(Event::Clear { view, value, rects }) => {
            let e = w.images.entry(view.resource.as_raw() as usize).or_default();
            if rects.is_empty() {
                match value {
                    Clear::Color(_) => e.extend(whole_subs(&view, view.slice.plane)),
                    Clear::DepthStencil { depth, stencil } => {
                        if depth.is_some() {
                            e.extend(whole_subs(&view, 0));
                        }
                        if stencil.is_some() && view.has_stencil() {
                            e.extend(whole_subs(&view, 1));
                        }
                    }
                }
            }
        }
        Ok(Event::Begin(pass)) => {
            for a in &pass.attachments {
                if !a.read_only {
                    w.images.entry(a.view.resource.as_raw() as usize).or_default();
                }
                if let Store::Resolve(r) = &a.store {
                    w.images.entry(r.dst.as_raw() as usize).or_default();
                }
            }
        }
        Ok(Event::End) => w.bound.clear(),
        Err(_) => w.untracked = true,
    });
}

/// list recorded a draw, dispatch, ExecuteIndirect or bundle: the bound attachments are written
/// and a UAV image in a UAV state may be.
pub(super) fn note_work(list: *mut c_void) {
    with_list(list as usize, |w| {
        w.work = true;
        if !w.bound_noted {
            w.bound_noted = true;
            for &r in &w.bound {
                w.images.entry(r).or_default();
            }
        }
    });
}

/// What a destination write covers of the destination.
pub(super) enum Covers {
    Part,
    Whole,
    Sub(u32),
}

/// list recorded a write (copy, resolve, UAV clear) into resource res.
pub(super) unsafe fn note_write(list: *mut c_void, res: *mut c_void, covers: Covers) {
    if res.is_null() || !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let subs = match covers {
        Covers::Part => Vec::new(),
        Covers::Sub(k) => vec![k],
        Covers::Whole => match ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()) {
            Some(d) => {
                let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(d.DepthOrArraySize) };
                (0..u32::from(d.MipLevels) * layers * planes_of(d.Format)).collect()
            }
            None => Vec::new(),
        },
    };
    with_list(list as usize, |w| w.images.entry(res as usize).or_default().extend(subs));
}

/// The extent of subresource sub of a resource described by d.
fn extent(d: &D3D12_RESOURCE_DESC, sub: u32) -> (u64, u64, u64) {
    let mip = sub % u32::from(d.MipLevels.max(1));
    let depth = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { (u64::from(d.DepthOrArraySize) >> mip).max(1) } else { 1 };
    ((d.Width >> mip).max(1), (u64::from(d.Height) >> mip).max(1), depth)
}

/// CopyTextureRegion into dst: the subresource is wholly overwritten when the copy starts at its
/// origin and its extent reaches the subresource's.
pub(super) unsafe fn note_copy_region(list: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, area: *const D3D12_BOX) {
    let (Some(d), Some(s)) = (dst.as_ref(), src.as_ref()) else { return };
    let Some(res) = d.pResource.as_ref() else { return };
    let raw = res.as_raw();
    if d.Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX {
        return note_write(list, raw, Covers::Part);
    }
    let sub = d.Anonymous.SubresourceIndex;
    let copied = if let Some(b) = area.as_ref() {
        Some((u64::from(b.right.saturating_sub(b.left)), u64::from(b.bottom.saturating_sub(b.top)), u64::from(b.back.saturating_sub(b.front))))
    } else if s.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        let f = s.Anonymous.PlacedFootprint.Footprint;
        Some((u64::from(f.Width), u64::from(f.Height), u64::from(f.Depth)))
    } else {
        s.pResource.as_ref().map(|r| extent(&r.GetDesc(), s.Anonymous.SubresourceIndex))
    };
    let (w, h, dd) = extent(&res.GetDesc(), sub);
    let whole = x == 0 && y == 0 && z == 0 && copied.is_some_and(|(cw, ch, cd)| cw >= w && ch >= h && cd >= dd);
    note_write(list, raw, if whole { Covers::Sub(sub) } else { Covers::Part });
}

/// PC list list starts a new recording: what it wrote goes.
pub(super) fn list_reset(list: usize) {
    if ACTIVE.load(Ordering::Relaxed) {
        LISTS.lock().unwrap().remove(&list);
    }
}

fn uav_state(img: &Image, state: u32) -> bool {
    state & 0x8 != 0 || (state == COMMON && img.simultaneous)
}

/// The PC lists went to queue, in this order: the images they wrote are captured and guarded,
/// and every image they wrote or transitioned is ordered after them.
#[cfg(test)]
pub(super) fn reports() -> u64 {
    REPORTS.with(|r| r.get())
}

pub(super) unsafe fn submitted(queue: *mut c_void, lists: &[*mut c_void]) {
    if !ACTIVE.load(Ordering::Relaxed) || lists.is_empty() || super::title_textures::is_own_work() {
        return;
    }
    #[cfg(test)]
    REPORTS.with(|r| r.set(r.get() + 1));
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let _submission = Submission::enter();
    let notes = super::title_textures::list_transitions(lists);
    let mut writes: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    let (mut work, mut untracked) = (false, NO_ATTACHMENTS.load(Ordering::Relaxed));
    {
        let l = LISTS.lock().unwrap();
        for list in lists {
            if let Some(w) = l.get(&(*list as usize)) {
                work |= w.work;
                untracked |= w.untracked;
                for (&k, subs) in &w.images {
                    writes.entry(k).or_default().extend(subs);
                }
            }
        }
    }
    if untracked {
        count(Count::Untracked);
    }
    let mut g = IMAGES.lock().unwrap();
    let mut touched = Vec::new();
    for (&key, img) in g.iter_mut() {
        let mut by_state = img.states.iter().any(|&s| writable(s));
        let mut uav = img.states.iter().any(|&s| uav_state(img, s));
        let mut seen = false;
        for &(res, sub, after) in &notes {
            if res == key {
                seen = true;
                apply(&mut img.states, sub, after.0 as u32);
                by_state |= writable(after.0 as u32);
                uav |= uav_state(img, after.0 as u32);
            }
        }
        let full = writes.remove(&key);
        let written = full.is_some() || (work && img.uav && uav) || (untracked && by_state);
        if written || seen {
            touched.push((key, written, full.unwrap_or_default()));
        }
    }
    if touched.is_empty() {
        return;
    }
    let Some(clock) = clock(q) else {
        count(Count::Failed);
        return;
    };
    let value = clock.value.fetch_add(1, Ordering::SeqCst) + 1;
    if q.Signal(&clock.fence, value).is_err() {
        count(Count::Failed);
        return;
    }
    for (key, written, full) in touched {
        let img = g.get_mut(&key).unwrap();
        img.last = Some((Arc::clone(&clock), value));
        if written {
            img.full.extend(full);
            capture(key, img);
        }
    }
}

/// img may have been written by the submission just made: its pages as they are at its first
/// such submission since it was last written back, and a guard on every address showing them.
unsafe fn capture(key: usize, img: &mut Image) {
    // already captured and guarded since its last write-back: the same pages
    if img.dirty.is_some() && !img.guards.is_empty() {
        img.generation += 1;
        return;
    }
    if super::physical_directory::remap_pending(img.va as u64, img.size as u64) {
        count(Count::RemapPending);
        img.dirty = None;
        unguard(img);
        return;
    }
    let first = img.va & !(PHYS_PAGE - 1);
    let slots: Vec<usize> = (first..img.va + img.size).step_by(PHYS_PAGE).collect();
    match kernelx::snapshot_queue_sources(&slots) {
        Ok(sources) => img.dirty = Some(Arc::new(sources)),
        Err(_) => {
            count(Count::Source);
            img.dirty = None;
            unguard(img);
            return;
        }
    }
    img.generation += 1;
    if img.guards.is_empty() {
        let over = img.size > guard_max_bytes()
            || GUARDED_IMAGES.load(Ordering::Relaxed) >= guard_max_images()
            || GUARDED_BYTES.load(Ordering::Relaxed).saturating_add(img.size) > guard_budget_bytes();
        if over && !img.read {
            count(Count::NotGuarded);
        } else {
            guard(key, img);
        }
    }
}

unsafe fn guard(key: usize, img: &mut Image) {
    kernelx::lazy::install_handler();
    let mut ranges = vec![(img.va, img.size)];
    ranges.extend(kernelx::alias_ranges(&ranges));
    for &(at, len) in &ranges {
        kernelx::lazy::ensure(at, len);
        let mut x = at & !0xFFF;
        let end = (at + len + 0xFFF) & !0xFFF;
        while x < end {
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            if VirtualQuery(x as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
                break;
            }
            let next = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(end);
            if next <= x {
                break;
            }
            let mut old = 0;
            if m.State == MEM_COMMIT && m.Protect != 0 && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
                && VirtualProtect(x as *const c_void, next - x, PAGE_NOACCESS, &mut old) != 0
            {
                img.guards.push((x, next - x, old));
            }
            x = next;
        }
    }
    if img.guards.is_empty() {
        return;
    }
    let mut g = GUARDED.write().unwrap_or_else(|e| e.into_inner());
    for &(at, len, _) in &img.guards {
        g.insert(at, (at + len, key));
    }
    LIVE.fetch_add(img.guards.len(), Ordering::Release);
    GUARDED_IMAGES.fetch_add(1, Ordering::Relaxed);
    GUARDED_BYTES.fetch_add(img.size, Ordering::Relaxed);
    count(Count::Guarded);
}

unsafe fn unguard(img: &mut Image) {
    if img.guards.is_empty() {
        return;
    }
    let mut g = GUARDED.write().unwrap_or_else(|e| e.into_inner());
    for (at, len, old) in img.guards.drain(..) {
        let mut was = 0;
        VirtualProtect(at as *const c_void, len, old, &mut was);
        g.remove(&at);
        LIVE.fetch_sub(1, Ordering::Release);
    }
    GUARDED_IMAGES.fetch_sub(1, Ordering::Relaxed);
    GUARDED_BYTES.fetch_sub(img.size, Ordering::Relaxed);
}

/// The images guarded in [at, end).
fn guarded_in(at: usize, end: usize) -> Vec<usize> {
    let g = GUARDED.read().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<usize> = g.range(..end).rev().take_while(|(_, &(e, _))| e > at).map(|(_, &(_, k))| k).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// An access violation at at: a guarded image there is written back first and its guard goes,
/// so the access is made again.
pub(crate) unsafe fn fault(at: usize) -> bool {
    if LIVE.load(Ordering::Acquire) == 0 {
        return false;
    }
    if IN_FLUSH.with(|c| c.get()) || IN_SUBMIT.with(|c| c.get()) {
        return open_pages(at, at + 1);
    }
    let images = guarded_in(at, at + 1);
    if images.is_empty() {
        return false;
    }
    IN_FLUSH.with(|c| c.set(true));
    for key in images {
        flush(key);
    }
    IN_FLUSH.with(|c| c.set(false));
    true
}

/// XCL reads [at, at + len) of the title's memory itself (memmap::copy_title): images guarded
/// there are written back first.
pub(crate) fn before_read(at: usize, len: usize) {
    if LIVE.load(Ordering::Acquire) == 0 || len == 0 {
        return;
    }
    if IN_FLUSH.with(|c| c.get()) || IN_SUBMIT.with(|c| c.get()) {
        unsafe { open_pages(at, at.saturating_add(len)) };
        return;
    }
    for key in guarded_in(at, at.saturating_add(len)) {
        unsafe { flush(key) };
    }
}

struct Snapshot {
    va: usize,
    size: usize,
    tex: Arc<xg::Texture>,
    parts: Arc<Vec<Part>>,
    mips: u32,
    layers: u32,
    states: Vec<u32>,
    last: Option<(Arc<Clock>, u64)>,
    sources: Arc<Vec<QueuePageSource>>,
    generation: u64,
    landed: Arc<Vec<u64>>,
    full: BTreeSet<u32>,
}

/// Writes image key back into its captured pages, again while a newer write raced it (four
/// times at most), and lifts its guard.
unsafe fn flush(key: usize) {
    let began = std::time::Instant::now();
    flush_inner(key);
    let ms = began.elapsed().as_millis();
    if ms >= 100 {
        crate::xlog_info!("D3D12.X: image writeback: a write-back took {ms} ms (image {key:#x}, {} guards open)", LIVE.load(Ordering::Acquire));
    }
}

unsafe fn flush_inner(key: usize) {
    let mut contexts = FLUSH.lock().unwrap_or_else(|e| e.into_inner());
    for _ in 0..4 {
        let snap = {
            let mut g = IMAGES.lock().unwrap();
            let Some(img) = g.get_mut(&key) else { return };
            img.read = true;
            if img.guards.is_empty() {
                return;
            }
            let Some(sources) = img.dirty.clone() else {
                unguard(img);
                return;
            };
            Snapshot {
                va: img.va,
                size: img.size,
                tex: Arc::clone(&img.tex),
                parts: Arc::clone(&img.parts),
                mips: img.mips,
                layers: img.layers,
                states: img.states.clone(),
                last: img.last.clone(),
                sources,
                generation: img.generation,
                landed: Arc::clone(&img.landed),
                full: img.full.clone(),
            }
        };
        let landed = land(&mut contexts, key, &snap);
        let mut g = IMAGES.lock().unwrap();
        let Some(img) = g.get_mut(&key) else { return };
        let done = landed.is_none() || img.generation == snap.generation;
        if let Some((digests, wrote)) = landed {
            count(if wrote { Count::Flushed } else { Count::Unchanged });
            img.landed = Arc::new(digests);
            img.full.retain(|k| !snap.full.contains(k));
        }
        if done {
            img.dirty = None;
            unguard(img);
            return;
        }
    }
    count(Count::Racing);
    if let Some(img) = IMAGES.lock().unwrap().get_mut(&key) {
        unguard(img);
    }
}

/// XCL's queue for write-backs on a device, with its fence.
struct Context {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    event: Owned<windows::Win32::Foundation::HANDLE>,
}

unsafe impl Send for Context {}

unsafe fn context<'a>(contexts: &'a mut BTreeMap<usize, Context>, dev: &ID3D12Device) -> Option<&'a mut Context> {
    let key = dev.as_raw() as usize;
    if !contexts.contains_key(&key) {
        let queue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).ok()?;
        let fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        let event = Owned::new(CreateEventW(None, false, false, None).ok()?);
        contexts.insert(key, Context { queue, fence, value: 0, event });
    }
    contexts.get_mut(&key)
}

/// The subresources written back: (native subresource, part, mip, layer).
fn subresources(s: &Snapshot) -> Vec<(u32, usize, u32, u32)> {
    let mut out = Vec::new();
    for (k, p) in s.parts.iter().enumerate() {
        for layer in 0..s.layers {
            for mip in 0..s.mips {
                out.push((mip + s.mips * (layer + s.layers * p.plane), k, mip, layer));
            }
        }
    }
    out
}

/// A subresource's linear texels, to tell whether the GPU changed them since the last landing.
fn digest(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h = bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(K);
    }
    for &b in words.remainder() {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }
    h
}

/// Reads image key back after the last submission that wrote it and lands, in the captured
/// pages, the subresources a tracked write wholly overwrote and those whose texels differ from
/// what it last landed: one the GPU left alone keeps the bytes the CPU wrote over it since. None: nothing was read back; else
/// every subresource's digest and whether any was landed.
unsafe fn land(contexts: &mut BTreeMap<usize, Context>, key: usize, s: &Snapshot) -> Option<(Vec<u64>, bool)> {
    let Some(res) = super::real::reference(key) else {
        count(Count::Released);
        return None;
    };
    let linear = read_back(contexts, &res, s)?;
    let first = s.va & !(PHYS_PAGE - 1);
    let views: Vec<Option<kernelx::PhysicalSectionView>> = s
        .sources
        .iter()
        .map(|src| match src {
            QueuePageSource::Physical(p) => kernelx::pin_physical_view_run(std::slice::from_ref(p)),
            _ => None,
        })
        .collect();
    let span = |i: usize| {
        let page = first + i * PHYS_PAGE;
        (page, page.max(s.va), (page + PHYS_PAGE).min(s.va + s.size))
    };
    // the title bytes the subresources do not cover are kept
    let mut tiled = vec![0u8; s.size];
    for (i, src) in s.sources.iter().enumerate() {
        let (page, lo, hi) = span(i);
        let out = &mut tiled[lo - s.va..hi - s.va];
        match (src, &views[i]) {
            (QueuePageSource::Physical(_), Some(v)) => std::ptr::copy_nonoverlapping((v.base() + (lo - page)) as *const u8, out.as_mut_ptr(), out.len()),
            (QueuePageSource::GuestRange { .. }, _) => with_access(lo, hi - lo, || std::ptr::copy_nonoverlapping(lo as *const u8, out.as_mut_ptr(), out.len())),
            (QueuePageSource::Physical(_), None) => {
                count(Count::Failed);
                return None;
            }
            (QueuePageSource::Hole, _) => {}
        }
    }
    let mut digests = Vec::new();
    let mut wrote = false;
    for (i, (sub, part, mip, layer)) in subresources(s).into_iter().enumerate() {
        let (surface, offset) = s.tex.plane_surface(s.parts[part].guest)?;
        let (data, row, rows, depth) = &linear[&sub];
        let level = surface.chain.levels[mip as usize];
        if level.width * surface.bpe as usize != *row || level.height != *rows as usize || (surface.depth > 0 && level.depth != *depth as usize) {
            count(Count::Shape);
            return None;
        }
        let d = digest(data);
        digests.push(d);
        if !s.full.contains(&sub) && s.landed.get(i) == Some(&d) {
            continue;
        }
        wrote = true;
        let offset = offset as usize;
        if surface.depth > 0 {
            let len = surface.chain.slice_bytes * surface.chain.layers();
            surface.chain.tile_volume(tiled.get_mut(offset..offset + len)?, mip as usize, data);
        } else {
            let at = offset + layer as usize * surface.chain.slice_bytes;
            surface.chain.tile_level(tiled.get_mut(at..at + surface.chain.slice_bytes)?, mip as usize, layer as usize, data);
        }
    }
    if !wrote {
        return Some((digests, false));
    }
    for (i, src) in s.sources.iter().enumerate() {
        let (page, lo, hi) = span(i);
        let from = &tiled[lo - s.va..hi - s.va];
        match (src, &views[i]) {
            (QueuePageSource::Physical(_), Some(v)) => std::ptr::copy_nonoverlapping(from.as_ptr(), (v.base() + (lo - page)) as *mut u8, from.len()),
            (QueuePageSource::GuestRange { .. }, _) => with_access(lo, hi - lo, || std::ptr::copy_nonoverlapping(from.as_ptr(), lo as *mut u8, from.len())),
            _ => count(Count::Hole),
        }
    }
    kernelx::note_written(s.va, s.size);
    Some((digests, true))
}

/// f with [at, at + len) readable and writable (a guarded page of private memory: there is no
/// other view of it).
unsafe fn with_access(at: usize, len: usize, f: impl FnOnce()) {
    let lo = at & !0xFFF;
    let hi = (at + len + 0xFFF) & !0xFFF;
    let mut old = 0;
    let opened = VirtualProtect(lo as *const c_void, hi - lo, PAGE_READWRITE, &mut old) != 0;
    f();
    if opened {
        let mut was = 0;
        VirtualProtect(lo as *const c_void, hi - lo, old, &mut was);
    }
}

/// native subresource -> (linear rows, row bytes, rows, depth)
type Linear = BTreeMap<u32, (Vec<u8>, usize, u32, u32)>;

/// Every written-back subresource of res as linear rows, copied on XCL's queue once the last
/// submission that touched it is done, in the states that submission left.
unsafe fn read_back(contexts: &mut BTreeMap<usize, Context>, res: &ID3D12Resource, s: &Snapshot) -> Option<Linear> {
    let mut dev: Option<ID3D12Device> = None;
    res.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let desc = res.GetDesc();
    let n = s.states.len();
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut sizes = vec![0u64; n];
    let mut total = 0u64;
    dev.GetCopyableFootprints(&desc, 0, n as u32, 0, Some(footprints.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(sizes.as_mut_ptr()), Some(&mut total));
    // the last footprint's advertised extent can omit row-end padding; the mapped buffer must
    // include it (CGBJ texback.rs)
    for (f, &r) in footprints.iter().zip(&rows) {
        let pitched = u64::from(f.Footprint.RowPitch).checked_mul(u64::from(r))?.checked_mul(u64::from(f.Footprint.Depth))?;
        total = total.max(f.Offset.checked_add(pitched)?);
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?;
    let buffer = buffer?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    let barrier = |sub: u32, before: u32, after: u32| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(res),
                Subresource: sub,
                StateBefore: D3D12_RESOURCE_STATES(before as i32),
                StateAfter: D3D12_RESOURCE_STATES(after as i32),
            }),
        },
    };
    let subs = subresources(s);
    let moved: Vec<u32> = subs.iter().map(|t| t.0).filter(|&k| s.states[k as usize] & COPY_SOURCE == 0).collect();
    let there: Vec<_> = moved.iter().map(|&k| barrier(k, s.states[k as usize], COPY_SOURCE)).collect();
    let back: Vec<_> = moved.iter().map(|&k| barrier(k, COPY_SOURCE, s.states[k as usize])).collect();
    if !there.is_empty() {
        list.ResourceBarrier(&there);
    }
    for &(k, ..) in &subs {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprints[k as usize] },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(res),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    if !back.is_empty() {
        list.ResourceBarrier(&back);
    }
    list.Close().ok()?;
    let ctx = context(contexts, &dev)?;
    if let Some((clock, value)) = &s.last {
        ctx.queue.Wait(&clock.fence, *value).ok()?;
    }
    let lists = [Some(list.cast::<ID3D12CommandList>().ok()?)];
    super::title_textures::own_work(|| ctx.queue.ExecuteCommandLists(&lists));
    ctx.value += 1;
    ctx.queue.Signal(&ctx.fence, ctx.value).ok()?;
    if crate::gpu::fence::wait(&ctx.fence, ctx.value, &dev, Some(*ctx.event), WAIT_MS).is_err() {
        count(Count::NotRetired);
        // the GPU still runs the copy: what it uses stays alive
        std::mem::forget((buffer, alloc, list, lists));
        return None;
    }
    let mut data = std::ptr::null_mut();
    if buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
        count(Count::Failed);
        return None;
    }
    let mut out = Linear::new();
    for &(k, ..) in &subs {
        let f = footprints[k as usize];
        let pitch = f.Footprint.RowPitch as usize;
        let row = (sizes[k as usize] as usize).min(pitch);
        let (r, depth) = (rows[k as usize], f.Footprint.Depth);
        let mut packed = Vec::with_capacity(row * r as usize * depth as usize);
        for z in 0..depth as usize {
            for y in 0..r as usize {
                let at = f.Offset as usize + (z * r as usize + y) * pitch;
                packed.extend_from_slice(std::slice::from_raw_parts(data.cast::<u8>().add(at), row));
            }
        }
        out.insert(k, (packed, row, r, depth));
    }
    buffer.Unmap(0, None);
    Some(out)
}

#[cfg(test)]
fn counted(name: &str) -> u64 {
    NAMES.iter().position(|n| *n == name).map_or(0, |i| TEST_COUNTS.with(|t| t.borrow()[i]))
}

#[cfg(test)]
fn force_on() {
    ENABLED.store(2, Ordering::Relaxed);
}

#[cfg(test)]
mod tests;
