use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

const HEAPS: usize = 256;
const VIEWS: usize = 8192;
const CLEARS: usize = 1024;

#[derive(Debug, Clone, Copy)]
// Recorded with each depth clear; only some of these fields are reported yet.
#[allow(dead_code)]
pub(super) struct DepthClear {
    pub(super) depth: f32,
    pub(super) uptime: f64,
    pub(super) list: usize,
    pub(super) sequence: u64,
}

pub(super) struct DsvView {
    pub(super) resource: ID3D12Resource,
    pub(super) mip: u32,
    pub(super) first_array_slice: u32,
    pub(super) array_size: u32,
    pub(super) format: DXGI_FORMAT,
    pub(super) read_only_depth: bool,
    pub(super) last_clear: Option<DepthClear>,
}

#[derive(Clone, Copy)]
struct View {
    owner: usize,
    generation: u64,
    mip: u32,
    first: u32,
    count: u32,
    format: DXGI_FORMAT,
    read_only_depth: bool,
    read_only_stencil: bool,
    mips: u32,
    slices: u32,
}

impl View {
    fn planes(self) -> u32 {
        if matches!(self.format.0, 20 | 45) { 2 } else { 1 }
    }

    fn range(self, plane: u32, planes: u32) -> Option<super::depth_state::Range> {
        super::depth_state::Range::texture(
            D3D12_BARRIER_SUBRESOURCE_RANGE {
                IndexOrFirstMipLevel: self.mip, NumMipLevels: 1,
                FirstArraySlice: self.first, NumArraySlices: self.count,
                FirstPlane: plane, NumPlanes: planes,
            },
            self.mips, self.slices, self.planes(),
        )
    }
}

struct Heap {
    owner: usize,
    generation: u64,
    end: usize,
    step: usize,
}

#[derive(Default)]
struct State {
    sequence: u64,
    heaps: BTreeMap<usize, Heap>,
    views: HashMap<usize, View>,
    clears: HashMap<(usize, u32, u32), DepthClear>,
}

impl State {
    fn next(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn contains(&self, handle: usize) -> bool {
        self.heaps.range(..=handle).next_back().is_some_and(|(&first, heap)| {
            handle < heap.end && (handle - first) % heap.step == 0
        })
    }

    fn put(&mut self, handle: usize, view: Option<View>) {
        self.views.remove(&handle);
        if let Some(mut view) = view.filter(|_| self.contains(handle) && self.views.len() < VIEWS) {
            view.generation = self.next();
            self.views.insert(handle, view);
        }
    }
}

static TABLE: Mutex<Option<State>> = Mutex::new(None);

pub(super) fn enabled() -> bool {
    selected_ps().is_some()
}

pub(super) fn selected_ps() -> Option<u64> {
    static HASH: OnceLock<Option<u64>> = OnceLock::new();
    *HASH.get_or_init(|| {
        if std::env::var("XCL_DEPTH_CAPTURE").as_deref() != Ok("1") || !super::gpu_timing::enabled() { return None }
        let value = std::env::var("XCL_DEPTH_CAPTURE_PS").ok()?;
        u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok().filter(|&v| v != 0)
    })
}

pub(super) fn record_heap(owner: usize, first: usize, count: u32, step: u32) {
    let Some(end) = (count as usize).checked_mul(step as usize).and_then(|n| first.checked_add(n)) else { return };
    if first == 0 || first == end {
        return;
    }
    let mut table = TABLE.lock().unwrap();
    let state = table.get_or_insert_with(State::default);
    state.views.retain(|&handle, _| handle < first || handle >= end);
    if state.heaps.len() < HEAPS || state.heaps.contains_key(&first) {
        let generation = state.next();
        state.heaps.insert(first, Heap { owner, generation, end, step: step as usize });
    }
}

pub(super) fn heap_generation(owner: usize) -> Option<(usize, u64)> {
    if !enabled() {
        return None;
    }
    TABLE.lock().unwrap().as_ref()?.heaps.iter().find_map(|(&first, heap)| {
        (heap.owner == owner).then_some((first, heap.generation))
    })
}

pub(super) fn forget_heap(first: usize, generation: u64) {
    let mut table = TABLE.lock().unwrap();
    let Some(state) = table.as_mut() else { return };
    if let Some(heap) = state.heaps.get(&first).filter(|heap| heap.generation == generation) {
        let end = heap.end;
        state.heaps.remove(&first);
        state.views.retain(|&handle, _| handle < first || handle >= end);
    }
}

pub(super) fn forget_resource(owner: usize) {
    if !enabled() {
        return;
    }
    if let Some(state) = TABLE.lock().unwrap().as_mut() {
        state.views.retain(|_, view| view.owner != owner);
        state.clears.retain(|&(resource, _, _), _| resource != owner);
    }
}

pub(super) unsafe fn record_view(handle: usize, raw: *mut c_void, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) {
    use windows::core::Interface;
    let view = (|| {
        if raw.is_null() || !super::real::is_adopted(raw) {
            return None;
        }
        let rd = if desc.is_none() || super::depth_lineage::enabled() {
            Some(ID3D12Resource::from_raw_borrowed(&raw)?.GetDesc())
        } else {
            None
        };
        let (mip, first, count, format, flags) = if let Some(d) = desc {
            let (mip, first, count) = match d.ViewDimension {
                D3D12_DSV_DIMENSION_TEXTURE1D => (d.Anonymous.Texture1D.MipSlice, 0, 1),
                D3D12_DSV_DIMENSION_TEXTURE2D => (d.Anonymous.Texture2D.MipSlice, 0, 1),
                D3D12_DSV_DIMENSION_TEXTURE1DARRAY => {
                    let v = d.Anonymous.Texture1DArray;
                    (v.MipSlice, v.FirstArraySlice, v.ArraySize)
                }
                D3D12_DSV_DIMENSION_TEXTURE2DARRAY => {
                    let v = d.Anonymous.Texture2DArray;
                    (v.MipSlice, v.FirstArraySlice, v.ArraySize)
                }
                D3D12_DSV_DIMENSION_TEXTURE2DMS => (0, 0, 1),
                D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY => {
                    let v = d.Anonymous.Texture2DMSArray;
                    (0, v.FirstArraySlice, v.ArraySize)
                }
                _ => return None,
            };
            (mip, first, count, d.Format, d.Flags)
        } else {
            let d = rd?;
            (0, 0, d.DepthOrArraySize as u32, d.Format, D3D12_DSV_FLAG_NONE)
        };
        super::device_x::watch_depth_first_view(raw as usize, "DSV");
        (count != 0).then_some(View {
            owner: raw as usize,
            generation: 0,
            mip,
            first,
            count,
            format,
            read_only_depth: flags.0 & D3D12_DSV_FLAG_READ_ONLY_DEPTH.0 != 0,
            read_only_stencil: flags.0 & D3D12_DSV_FLAG_READ_ONLY_STENCIL.0 != 0,
            mips: rd.map_or(0, |d| u32::from(d.MipLevels)),
            slices: rd.map_or(0, |d| u32::from(d.DepthOrArraySize)),
        })
    })();
    if let Some(state) = TABLE.lock().unwrap().as_mut() {
        state.put(handle, view);
    }
}

pub(super) unsafe fn copy_ranges(
    step: usize,
    dst: (&[u64], *const u32),
    src: (&[u64], *const u32),
) {
    let mut table = TABLE.lock().unwrap();
    let Some(state) = table.as_mut() else { return };
    let counts = |sizes: *const u32, i: usize| if sizes.is_null() { 1 } else { *sizes.add(i) as usize };
    let mut sources = src.0.iter().enumerate().flat_map(|(i, &first)| {
        (0..counts(src.1, i)).map(move |n| (first as usize).saturating_add(n.saturating_mul(step)))
    });
    let mut remaining = VIEWS;
    for (i, &first) in dst.0.iter().enumerate() {
        let count = counts(dst.1, i);
        if count > remaining {
            state.views.clear();
            return;
        }
        remaining -= count;
        for n in 0..count {
            let view = sources.next().and_then(|handle| state.views.get(&handle).copied());
            state.put((first as usize).saturating_add(n.saturating_mul(step)), view);
        }
    }
}

pub(super) fn note_bind(list: usize, handle: usize, kind: &'static str) -> Option<u64> {
    if !enabled() {
        return None;
    }
    let view = *TABLE.lock().unwrap().as_ref()?.views.get(&handle)?;
    if super::depth_lineage::enabled() {
        if let Some(range) = view.range(0, view.planes()) {
            super::depth_lineage::note(view.owner, list, range, kind, || format!(
                "handle={handle:#x} generation={} format={} read_only_depth={} read_only_stencil={}",
                view.generation, view.format.0, view.read_only_depth, view.read_only_stencil,
            ));
        }
    }
    Some(view.generation)
}

pub(super) fn note_depth_draw(list: usize, handle: usize, generation: u64, pso: usize, call: &'static str, predicated: bool) {
    let view = TABLE.lock().unwrap().as_ref().and_then(|state| state.views.get(&handle)).copied();
    let Some(view) = view.filter(|v| v.generation == generation && !v.read_only_depth) else { return };
    if let Some(range) = view.range(0, 1) {
        super::depth_lineage::note(view.owner, list, range, "dsv_depth_write_capable_draw", || format!(
            "call={call} handle={handle:#x} generation={generation} pso={pso:#x} predicated={predicated}",
        ));
    }
}

pub(super) fn lookup(handle: usize, binding_generation: u64) -> Option<DsvView> {
    if !enabled() {
        return None;
    }
    let table = TABLE.lock().unwrap();
    let state = table.as_ref()?;
    let view = state.views.get(&handle).filter(|v| v.generation == binding_generation)?;
    let resource = unsafe { super::real::try_reference_depth_capture(view.owner)? };
    let last_clear = if view.count == 1 { state.clears.get(&(view.owner, view.mip, view.first)).copied() } else { None };
    Some(DsvView {
        resource,
        mip: view.mip,
        first_array_slice: view.first,
        array_size: view.count,
        format: view.format,
        read_only_depth: view.read_only_depth,
        last_clear,
    })
}

pub(super) fn note_clear(list: *mut c_void, handle: usize, flags: u32, depth: f32, rects: u32) {
    if !enabled() {
        return;
    }
    if super::depth_lineage::enabled() {
        let view = TABLE.lock().unwrap().as_ref().and_then(|state| state.views.get(&handle)).copied();
        if let Some(view) = view {
            for (plane, flag, read_only) in [
                (0, D3D12_CLEAR_FLAG_DEPTH, view.read_only_depth),
                (1, D3D12_CLEAR_FLAG_STENCIL, view.read_only_stencil),
            ] {
                if flags & flag.0 as u32 != 0 {
                    if let Some(range) = view.range(plane, 1) {
                        super::depth_lineage::note(view.owner, list as usize, range, "dsv_clear_recorded", || format!(
                            "handle={handle:#x} generation={} flags={flags:#x} plane={plane} depth={:?} rects={rects} read_only={read_only}",
                            view.generation, (plane == 0).then_some(depth),
                        ));
                    }
                }
            }
        }
    }
    if flags & D3D12_CLEAR_FLAG_DEPTH.0 as u32 == 0 {
        return;
    }
    let mut table = TABLE.lock().unwrap();
    let Some(state) = table.as_mut() else { return };
    let Some(view) = state.views.get(&handle).copied() else {
        if state.contains(handle) {
            state.clears.clear();
        }
        return;
    };
    if rects != 0 || view.count != 1 || view.read_only_depth {
        state.clears.retain(|&(owner, mip, slice), _| {
            !(owner == view.owner && mip == view.mip && slice >= view.first && slice - view.first < view.count)
        });
        return;
    }
    if state.clears.len() >= CLEARS {
        state.clears.clear();
    }
    let clear = DepthClear { depth, uptime: crate::log::uptime(), list: list as usize, sequence: state.next() };
    state.clears.insert((view.owner, view.mip, view.first), clear);
}
