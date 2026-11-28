use super::super::{
    depth_capture_views, depth_lineage_queue as queue, indirect, list_x, markers, present, real,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const LIMIT: usize = 96;
struct Ticket {
    list: usize,
    generation: u64,
    submits: usize,
    reset: bool,
    retired: bool,
}
static TICKETS: Mutex<Vec<Ticket>> = Mutex::new(Vec::new());

pub(super) fn limit() -> usize {
    if records::enabled() {
        256
    } else {
        LIMIT
    }
}

fn ticket(list: usize, generation: u64) -> Option<u32> {
    let mut slots = TICKETS.lock().unwrap();
    let next = Ticket {
        list,
        generation,
        submits: 0,
        reset: false,
        retired: false,
    };
    if let Some(i) = slots
        .iter()
        .position(|t| t.list == 0 || (t.reset && t.retired && t.submits == 1))
    {
        slots[i] = next;
        return Some(i as u32);
    }
    if slots.len() >= limit() {
        reject("all private tickets in flight, replayed, or awaiting native Reset");
        return None;
    }
    slots.push(next);
    Some(slots.len() as u32 - 1)
}

pub(crate) fn native_reset(list: usize) {
    if records::enabled() {
        for t in TICKETS
            .lock()
            .unwrap()
            .iter_mut()
            .filter(|t| t.list == list)
        {
            t.reset = true;
        }
    }
}
const STRIDE: usize = 160;
static SELECTED: Mutex<Option<(usize, u64)>> = Mutex::new(None);
static N: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static SEEN_QUERY: LazyLock<Mutex<HashMap<usize, u64>>> = LazyLock::new(Default::default);
static UNKNOWN: AtomicBool = AtomicBool::new(false);
static ADMIT: LazyLock<Mutex<(u64, HashMap<usize, usize>, usize)>> =
    LazyLock::new(Default::default);
static PENDING: Mutex<Vec<(Probe, usize)>> = Mutex::new(Vec::new());
static POOL: OnceLock<Option<Pool>> = OnceLock::new();

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DEPTH_PREPASS_QUERY").as_deref() == Ok("1"))
}

pub(crate) fn select(owner: usize, lifetime: u64) {
    if enabled() {
        *SELECTED.lock().unwrap() = Some((owner, lifetime));
    }
}

pub(crate) fn query_seen(list: usize) {
    if !enabled() {
        return;
    }
    let gen_id = queue::recording(list);
    let mut map = SEEN_QUERY.lock().unwrap();
    if map.len() >= 4096 && !map.contains_key(&list) {
        UNKNOWN.store(true, Ordering::Release);
        return;
    }
    map.insert(list, gen_id);
}

struct Pool {
    occ: ID3D12QueryHeap,
    stats: Option<ID3D12QueryHeap>,
    buf: ID3D12Resource,
    cpu: usize,
    device: usize,
}

unsafe fn pool(list: &ID3D12GraphicsCommandList) -> Option<&'static Pool> {
    let mut device: Option<ID3D12Device> = None;
    list.GetDevice(&mut device).ok()?;
    let device = device?;
    if device.as_raw() != real::device_raw() {
        return None;
    }
    let pool = POOL
        .get_or_init(|| {
            let heap = |ty| {
                let mut h: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: ty,
                            Count: limit() as u32,
                            NodeMask: 0,
                        },
                        &mut h,
                    )
                    .ok()?;
                h
            };
            let occ = heap(D3D12_QUERY_HEAP_TYPE_OCCLUSION)?;
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let stats = (device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS9,
                    (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS9).cast(),
                    std::mem::size_of_val(&options) as u32,
                )
                .is_ok()
                && options.MeshShaderPipelineStatsSupported.as_bool())
            .then(|| heap(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1))
            .flatten();
            let (b, _, cpu) = markers::readback_buffer((limit() * STRIDE) as u64)?;
            if cpu == 0 {
                return None;
            }
            std::ptr::write_bytes(cpu as *mut u8, 0xff, limit() * STRIDE);
            Some(Pool {
                occ,
                stats,
                buf: ID3D12Resource::from_raw(b as *mut c_void),
                cpu,
                device: device.as_raw() as usize,
            })
        })
        .as_ref()?;
    (pool.device == device.as_raw() as usize).then_some(pool)
}

type Query = unsafe extern "system" fn(*mut c_void, *mut c_void, D3D12_QUERY_TYPE, u32);
type Resolve = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    D3D12_QUERY_TYPE,
    u32,
    u32,
    *mut c_void,
    u64,
);

pub(crate) mod records;

pub(crate) struct Probe {
    owner: (usize, u64),
    dsv: (usize, u64),
    exports: Option<records::Lease>,
    slot: u32,
    list: usize,
    generation: u64,
    offsets: [u64; 3],
    scratch: Option<ID3D12Resource>,
    desc: String,
}

fn reject(why: &'static str) {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) < 6 {
        crate::xlog_info!("prepass query: unavailable: {why}");
    }
}

pub(crate) unsafe fn prepare(
    list: &ID3D12GraphicsCommandList,
    pso: usize,
    tp: &real::TwoPassPso,
    scratch: usize,
    offsets: [u64; 3],
    sig: usize,
    k: u64,
    max: u32,
    guest: (Option<u64>, Option<u64>),
    views: [u64; 2],
    planned: bool,
    staged_io: &[u32],
) -> Option<Probe> {
    let qualified = records::enabled();
    if !enabled()
        || (!qualified && N.load(Ordering::Relaxed) >= LIMIT)
        || (qualified && records::done())
    {
        return None;
    }
    let (owner, stamp) = (*SELECTED.lock().unwrap())?;
    let raw = list.as_raw();
    let generation = queue::recording(raw as usize);
    if generation == 0
        || UNKNOWN.load(Ordering::Acquire)
        || !real::is_adopted(raw)
        || !super::super::gpu_timing::query_scope_open(raw as usize, generation)
        || SEEN_QUERY
            .lock()
            .unwrap()
            .get(&(raw as usize))
            .is_some_and(|&g| g == 0 || g == generation)
    {
        reject("native query/pass scope unavailable on this recording");
        return None;
    }
    let (handle, view_gen, bounds) = list_x::prepass_dsv(raw as usize, pso, planned)?;
    let v = depth_capture_views::lookup(handle, view_gen)?;
    if v.resource.as_raw() as usize != owner
        || v.read_only_depth
        || v.mip != 0
        || v.first_array_slice != 0
        || v.array_size != 1
        || v.format.0 != 20
        || super::VIEWS.lock().unwrap().as_ref()?.observed.get(&owner) != Some(&stamp)
    {
        return None;
    }
    let fixed = tp.prepass_fixed.as_ref()?;
    let variant = if qualified {
        Some(tp.export_observer()?)
    } else {
        None
    };
    let (_, epoch) = present::counters();
    {
        let mut a = ADMIT.lock().unwrap();
        if epoch > a.0 {
            *a = (epoch, HashMap::new(), 0);
        }
        if epoch != a.0 || a.2 >= 20 || a.1.len() >= 256 && !a.1.contains_key(&pso) {
            return None;
        }
        let n = a.1.entry(pso).or_default();
        *n += 1;
        if !qualified && !matches!(*n, 1 | 8 | 32 | 128 | 256) {
            return None;
        }
        a.2 += 1;
    }
    let pool = pool(list).or_else(|| {
        reject("private query heap/readback unsupported");
        None
    })?;
    let sr = scratch as *mut c_void;
    let sr = ID3D12Resource::from_raw_borrowed(&sr)?;
    let rd = sr.GetDesc();
    if rd.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || offsets
            .iter()
            .zip([12, 12, 4])
            .any(|(&o, n)| o.checked_add(n).is_none_or(|end| end > rd.Width))
    {
        return None;
    }
    let slot = if qualified {
        ticket(raw as usize, generation)?
    } else {
        N.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            (n < LIMIT).then_some(n + 1)
        })
        .ok()? as u32
    };
    let desc = format!(
        "owner={owner:#x}/{stamp} sub=0 dsv={handle:#x}/{view_gen} pso={pso:#x} epoch={epoch} sig={sig:#x} command={k}/{max} guest_arg_and_count={guest:x?} mesh_split={} depth_bounds={bounds:?} last_recorded_clear={:?} shaders={:016x?} {fixed}; {}",
        tp.mesh_groups_per_record,
        v.last_clear,
        real::draw_gate_shaders(pso),
        list_x::state_summary(raw)
    );
    let exports = match variant {
        Some(variant) => {
            let Some(lease) = records::Lease::bind(
                raw, variant, pso, tp.mesh_rs, views, &tp.record, slot, generation, staged_io,
            ) else {
                TICKETS.lock().unwrap()[slot as usize].list = 0;
                reject("no private mailbox or unambiguous viewport");
                return None;
            };
            std::ptr::write_bytes((pool.cpu + slot as usize * STRIDE) as *mut u8, 0xff, STRIDE);
            Some(lease)
        }
        None => None,
    };
    Some(Probe {
        owner: (owner, stamp),
        dsv: (handle, view_gen),
        exports,
        slot,
        list: raw as usize,
        generation,
        offsets,
        scratch: Some(sr.clone()),
        desc,
    })
}

impl Probe {
    pub(crate) unsafe fn start(
        mut self,
        list: &ID3D12GraphicsCommandList,
        pso: usize,
    ) -> Option<Self> {
        let raw = list.as_raw();
        let (handle, vg, _) = list_x::prepass_dsv(raw as usize, pso, false)?;
        if (handle, vg) != self.dsv
            || *SELECTED.lock().unwrap() != Some(self.owner)
            || queue::recording(raw as usize) != self.generation
            || !super::super::gpu_timing::query_scope_open(raw as usize, self.generation)
        {
            return None;
        }
        let pool = POOL.get()?.as_ref()?;
        if let Some(lease) = &self.exports {
            lease.mesh(raw);
        }
        let begin: Query = std::mem::transmute(real::orig_fn(raw, 52));
        let slot = self.slot;
        if let Some(stats) = &pool.stats {
            begin(
                raw,
                stats.as_raw(),
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
                slot,
            );
        }
        begin(raw, pool.occ.as_raw(), D3D12_QUERY_TYPE_OCCLUSION, slot);
        if records::producer() {
            self.desc.push_str(&format!(
                " actual_mesh_state={}",
                list_x::state_summary(raw)
            ));
        }
        Some(self)
    }

    pub(crate) unsafe fn compute(&self, raw: *mut c_void, tp: &real::TwoPassPso, restore: bool) {
        if let Some(lease) = &self.exports {
            lease.compute(raw, tp, restore);
        }
    }
    pub(crate) unsafe fn finish(mut self, list: &ID3D12GraphicsCommandList) {
        let pool = POOL.get().unwrap().as_ref().unwrap();
        let raw = list.as_raw();
        if let Some(lease) = &self.exports {
            lease.restore(raw);
        }
        let at = self.slot as u64 * STRIDE as u64;
        let end: Query = std::mem::transmute(real::orig_fn(raw, 53));
        let resolve: Resolve = std::mem::transmute(real::orig_fn(raw, 54));
        end(
            raw,
            pool.occ.as_raw(),
            D3D12_QUERY_TYPE_OCCLUSION,
            self.slot,
        );
        if let Some(stats) = &pool.stats {
            end(
                raw,
                stats.as_raw(),
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
                self.slot,
            );
            resolve(
                raw,
                stats.as_raw(),
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
                self.slot,
                1,
                pool.buf.as_raw(),
                at,
            );
        }
        resolve(
            raw,
            pool.occ.as_raw(),
            D3D12_QUERY_TYPE_OCCLUSION,
            self.slot,
            1,
            pool.buf.as_raw(),
            at + 112,
        );
        let source = self.scratch.take().unwrap();
        let scratch = source.as_raw() as usize;
        list.ResourceBarrier(&[indirect::transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for ((off, bytes), src) in [(120, 12), (132, 12), (144, 4)]
            .into_iter()
            .zip(self.offsets)
        {
            list.CopyBufferRegion(&pool.buf, at + off, &source, src, bytes);
        }
        list.ResourceBarrier(&[indirect::transition(
            scratch,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
        queue::watch_selected(self.list, self.generation);
        let mut slots = PENDING.lock().unwrap();
        slots.push((self, 0));
        ACTIVE.store(true, Ordering::Release);
    }
}

pub(crate) fn pending() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

pub(crate) fn entered(lists: &[*mut c_void], live: &HashMap<usize, u64>) {
    if !pending() {
        return;
    }
    if records::enabled() {
        let mut tickets = TICKETS.lock().unwrap();
        for t in tickets.iter_mut().filter(|t| !t.reset && t.list != 0) {
            t.submits += lists.iter().filter(|&&l| l as usize == t.list).count();
        }
    }
    for (p, n) in PENDING.lock().unwrap().iter_mut() {
        if live.get(&p.list) == Some(&p.generation) {
            *n += lists.iter().filter(|&&l| l as usize == p.list).count();
        }
    }
}

pub(crate) fn retired(list: usize, generation: u64, q: usize, seq: u64, submit: u64) {
    let mut pending = PENDING.lock().unwrap();
    let mut i = 0;
    while i < pending.len() {
        if (pending[i].0.list, pending[i].0.generation) != (list, generation) {
            i += 1;
            continue;
        }
        let (p, n) = pending.swap_remove(i);
        let pool = POOL.get().unwrap().as_ref().unwrap();
        if n != 1
            || (p.exports.is_some()
                && !TICKETS
                    .lock()
                    .unwrap()
                    .get(p.slot as usize)
                    .is_some_and(|t| {
                        t.list == list && t.generation == generation && t.submits == 1
                    }))
        {
            reject("list replayed or not submitted exactly once");
            continue;
        }
        let at = pool.cpu + p.slot as usize * STRIDE;
        let occ = unsafe { ((at + 112) as *const u64).read_volatile() };
        let args = unsafe { ((at + 120) as *const [u32; 7]).read_volatile() };
        if occ == u64::MAX || args.contains(&u32::MAX) {
            reject("query or private argument copies not resolved");
            continue;
        }
        let stats = pool.stats.as_ref().and_then(|_| {
            let words = unsafe { (at as *const [u64; 14]).read_volatile() };
            (!words.contains(&u64::MAX)).then_some(words)
        });
        if let Some(lease) = p.exports {
            lease.report(
                list,
                generation,
                q,
                seq,
                submit,
                p.slot,
                &args,
                occ,
                stats,
                p.desc.clone(),
            );
            let mut tickets = TICKETS.lock().unwrap();
            if let Some(t) = tickets
                .get_mut(p.slot as usize)
                .filter(|t| t.list == list && t.generation == generation && t.submits == 1)
            {
                t.retired = true;
            }
        }
        crate::xlog_info!(
            "prepass query: retired list={list:#x} generation={generation} q={q} seq={seq} submit={submit} slot={} actual_compute_grid={:?} actual_mesh_grid={:?} sanitised_count={} passed_depth_stencil_samples={occ} pipeline_stats1[IAvertices,IAprimitives,VS,GS,GSprim,clip_in,clip_out,PS,HS,DS,CS,AS,MS,MSprim]={stats:?}; {}",
            p.slot,
            &args[0..3],
            &args[3..6],
            args[6],
            p.desc
        );
    }
    ACTIVE.store(!pending.is_empty(), Ordering::Release);
}
