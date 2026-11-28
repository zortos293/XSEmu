use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use super::super::{console_heap, depth_lineage_queue as queue, dispatch_timing, present, real, title_textures, vram};
use super::VIEWS;

struct Config {
    shader: u64,
    slots: Vec<(usize, u32)>,
    arm: PathBuf,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let value = std::env::var("XCL_SOURCE_DEPTH_WRITER").ok()?;
        let parse = || {
            let (shader, slots) = value.split_once('=')?;
            let shader = u64::from_str_radix(shader.trim_start_matches("0x"), 16).ok()?;
            let slots = slots.split(',').map(|slot| {
                let (r, o) = slot.trim().strip_prefix('s')?.split_once("+0x")?;
                let reg = r.parse::<usize>().ok()?;
                let offset = u32::from_str_radix(o, 16).ok()?;
                (reg < 32 && offset % console_heap::SLOT == 0).then_some((reg, offset))
            }).collect::<Option<Vec<_>>>()?;
            let arm = std::env::var_os("XCL_DEPTH_CAPTURE_ARM_FILE")?.into();
            (shader != 0 && !slots.is_empty() && slots.len() <= 16).then_some(Config { shader, slots, arm })
        };
        let c = parse();
        if c.is_none() { crate::xlog_warn!("source writer probe: invalid writer slots or missing arm path"); }
        c
    }).as_ref()
}

pub(crate) fn enabled() -> bool { config().is_some() }

pub(crate) fn identity(at: u32, index: u32, owner: usize, resource: usize, stamp: u64) -> String {
    let native = console_heap::native_uav_details_at(at, index).map(|u| (u.resource, u.view, u.backing, u.volume));
    let backing = if resource == 0 { None } else { vram::arena_lineage(resource).or_else(|| vram::writer_tiles(resource)) };
    let title = real::title_address(resource).or_else(|| title_textures::address(resource));
    let current = VIEWS.lock().unwrap().as_ref().and_then(|v| v.observed.get(&owner).copied());
    if current != Some(stamp) || real::resource_release_pending(owner) {
        return " owner_released_during_probe".into()
    }
    format!(" first_observed_lifetime={stamp} created_native_uav_at_index_plus_1={native:x?} title_va={title:x?} backing={}", backing.as_deref().unwrap_or("untracked placement"))
}

pub(crate) struct Snapshot {
    lease: Option<super::super::material_gate_capture::Lease>,
    list: usize,
    generation: u64,
    epoch: u64,
    pso: usize,
    hash: u64,
    views: String,
}

pub(crate) fn snapshot(list: usize, pso: usize, io: &[u32]) -> Option<Snapshot> {
    let c = config()?;
    if !queue::enabled() || dispatch_timing::profile_shader(pso) != Some(c.shader) { return None }
    static COUNT: AtomicU32 = AtomicU32::new(0);
    if COUNT.load(Ordering::Relaxed) >= 24 || !c.arm.is_file() { return None }
    let generation = queue::recording(list);
    if generation == 0 { return None }
    COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| (n < 24).then_some(n + 1)).ok()?;
    let (_, epoch) = present::counters();
    let views = super::describe_slots(io, &c.slots, true).1;
    Some(Snapshot { list, generation, epoch, pso, hash: c.shader, views, lease: None })
}

type Lease = super::super::material_gate_capture::Lease;
static LEASES: std::sync::Mutex<Vec<(usize, u64, Lease, usize)>> = std::sync::Mutex::new(Vec::new());
static HAS_LEASES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn pending() -> bool { HAS_LEASES.load(Ordering::Acquire) }

pub(crate) fn entered(lists: &[*mut std::ffi::c_void], live: &std::collections::HashMap<usize, u64>) {
    if !pending() { return }
    let mut leases = LEASES.lock().unwrap();
    for (list, generation, _, count) in leases.iter_mut() {
        if live.get(list) == Some(generation) {
            *count += lists.iter().filter(|&&raw| raw as usize == *list).count();
        }
    }
}

pub(crate) fn retired(list: usize, generation: u64, q: usize, seq: u64, submit: u64) {
    if !HAS_LEASES.load(Ordering::Acquire) { return }
    let mut leases = LEASES.lock().unwrap();
    let mut i = 0;
    while i < leases.len() {
        if (leases[i].0, leases[i].1) != (list, generation) { i += 1; continue; }
        let (_, _, lease, count) = leases.swap_remove(i);
        if count == 1 {
            unsafe { lease.writer_retired(list, generation, q, seq, submit); }
        } else { super::super::material_gate_capture::refusal("writer list replayed or not seen entering native submit"); }
    }
    HAS_LEASES.store(!leases.is_empty(), Ordering::Release);
}

impl Snapshot {
    pub(crate) fn prepare(&mut self, io: &[u32]) -> Option<u64> {
        use super::super::{depth_lineage, list_x, material_gate_capture};
        use windows::Win32::Graphics::Direct3D12::{D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        if !list_x::depth_copy_unpredicated(self.list) || !matches!(list_x::tracked_list_type(self.list as *mut _), Some(D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
            return None;
        }
        let resolve = |&(r, offset): &(usize, u32)| {
            let (at, words, key) = console_heap::material_input(*io.get(r)?, offset)?;
            let g = VIEWS.lock().unwrap();
            let views = g.as_ref()?;
            let entry = *views.by_key.get(&key?)?;
            let v = entry.view;
            let sub = v.subresources;
            if entry.words != words || v.resource == 0 || v.owner != v.resource || real::resource_release_pending(v.owner)
                || v.dimension != 4 || v.mapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING || v.min_lod != 0.0
                || (sub.mip, sub.mips, sub.first_slice, sub.slices, sub.plane) != (0,1,0,1,0) { return None }
            Some((at, words[0], crate::gpu::desc::image(&words)?, v, *views.observed.get(&v.owner)?))
        };
        let [src, dst] = super::inputs() else { return None };
        let (_, si, image, s, src_stamp) = resolve(src)?;
        let (at, di, out, d, dst_stamp) = resolve(dst)?;
        let u = console_heap::native_uav_details_at(at, di)?;
        if !matches!(s.requested, Some(super::GuestView::Srv(21))) || (s.resource_format, s.view_format, image.format) != (19,21,0)
            || !matches!(d.requested, Some(super::GuestView::Uav(41))) || (d.resource_format, d.view_format, out.format) != (41,41,22)
            || s.owner == d.owner || u.resource != d.owner || u.view != (41,4) || (image.width,image.height) != (out.width,out.height) { return None }
        super::prepass::select(s.owner, src_stamp);
        static ISSUED: AtomicU32 = AtomicU32::new(0);
        let ticket = ISSUED.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| (n < 6).then_some(n+1)).ok()?;
        if ticket < 3 { depth_lineage::report_writer_input(s.owner, self.list, self.generation); }
        let (lease, va) = material_gate_capture::Lease::new(self.generation, di.checked_add(1)?, out.width, out.height, || {
            (Some(si), format!("epoch={} cs={:016x} input_owner={:#x}/{} native=21 plane=0 gfx10={} output_owner={:#x}/{} native=41 plane=0 gfx10={}; {}", self.epoch, self.hash, s.owner, src_stamp, image.format, d.owner, dst_stamp, out.format, self.views))
        })?;
        self.lease = Some(lease);
        Some(va)
    }

    pub(crate) fn recorded(self, grid: [u32; 3]) {
        let Self { list, generation, epoch, pso, hash, views, lease } = self;
        if queue::recording(list) != generation {
            crate::xlog_warn!("source writer probe: recording changed before native dispatch join");
            return
        }
        queue::watch_selected(list, generation);
        if let Some(lease) = lease {
            let mut leases = LEASES.lock().unwrap();
            leases.push((list, generation, lease, 0));
            HAS_LEASES.store(true, Ordering::Release);
        }
        crate::xlog_info!("source writer probe: epoch={epoch} list={list:#x} generation={generation} pso={pso:#x} cs={hash:016x} grid={grid:?} native_dispatch_recorded; store operands are candidate destinations, not executed stores; {views}");
    }
}
