use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub(super) use super::gpu_timing_stats::Kind;
use super::depth_lineage_queue as lineage;
use super::gpu_timing_stats::{self as stats, Batch, Span};
use super::gpu_timing_waits as waits;

const SLOTS: usize = 512;
const MAX_QUEUES: usize = 64;
const MAX_FLIGHTS: usize = 512;
const PAIRS_PER_PAGE: usize = 1024;
const MAX_LIST_PAGES: usize = 32;
const MAX_SPILL_PAGES: usize = 512;
const QUERIES: u32 = 2 * PAIRS_PER_PAGE as u32;
const BYTES: u64 = QUERIES as u64 * 8;
const QUEUE_TOKEN: GUID = GUID::from_u128(0x93ac32d8_2167_43b1_8172_4c95de854c03);
const LIST_TOKEN: GUID = GUID::from_u128(0xb7d3b648_e514_405e_a76c_20c7603271fa);

static UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
static NO_QUEUE_CLOCK: AtomicU64 = AtomicU64::new(0);
static UNAVAILABLE: AtomicU64 = AtomicU64::new(0);
static CAPACITY_DROPS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
static WARP_TIMING_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
thread_local! {
    static WARP_TEST: Cell<bool> = const { Cell::new(false) };
    static WARP_CLASSES: Cell<bool> = const { Cell::new(true) };
    static WARP_STRIDE: Cell<u32> = const { Cell::new(1) };
    static WARP_DISPATCH_MIN_GROUPS: Cell<Option<u64>> = const { Cell::new(None) };
    static WARP_FAIL_MARKER: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if WARP_TEST.with(Cell::get) {
        return true;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_GPU_LIST_TIMING").is_ok_and(|s| s.trim() == "1"))
}

pub(super) fn classes_enabled() -> bool {
    #[cfg(test)]
    if WARP_TEST.with(Cell::get) {
        return WARP_CLASSES.with(Cell::get);
    }
    static CLASSES: OnceLock<bool> = OnceLock::new();
    enabled()
        && *CLASSES.get_or_init(|| {
            std::env::var("XCL_GPU_LIST_TIMING_CLASSES").is_ok_and(|s| s.trim() == "1")
        })
}

fn stride() -> u32 {
    #[cfg(test)]
    if WARP_TEST.with(Cell::get) {
        return WARP_STRIDE.with(Cell::get);
    }
    static STRIDE: OnceLock<u32> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("XCL_GPU_LIST_TIMING_EVERY")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|&n| n != 0)
            .unwrap_or(32)
    })
}

fn dispatch_min_groups() -> Option<u64> {
    #[cfg(test)]
    if WARP_TEST.with(Cell::get) {
        return WARP_DISPATCH_MIN_GROUPS.with(Cell::get);
    }
    static MIN_GROUPS: OnceLock<Option<u64>> = OnceLock::new();
    *MIN_GROUPS.get_or_init(|| {
        std::env::var("XCL_GPU_LIST_TIMING_DISPATCH_MIN_GROUPS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|&n| n != 0)
    })
}

fn dispatch_qpc_selection() -> Option<&'static (usize, Vec<u64>)> {
    static SELECT: OnceLock<Option<(usize, Vec<u64>)>> = OnceLock::new();
    SELECT.get_or_init(|| {
        let value = std::env::var("XCL_GPU_LIST_TIMING_QPC").ok()?;
        let (queue, shaders) = value.split_once(':')?;
        let queue = queue.trim().parse::<usize>().ok()?;
        let shaders = shaders.split(',').map(|s| u64::from_str_radix(s.trim(), 16)).collect::<Result<Vec<_>, _>>().ok()?;
        (queue != 0).then_some((queue, shaders))
    }).as_ref()
}

#[derive(Clone, Copy)]
struct Op {
    kind: Kind,
    pso: usize,
    shaders: Option<(u64, u64)>,
    grid: Option<[u32; 3]>,
    sampled: bool,
}

#[derive(Default)]
struct Recording {
    id: usize,
    depth_capture: Option<Arc<super::depth_capture::Capture>>,
    pipeline: usize,
    owned: bool,
    started: bool,
    sealed: bool,
    pass: Option<D3D12_RENDER_PASS_FLAGS>,
    suspended: bool,
    quarantined: bool,
    in_flight: u32,
    readable_flight: Option<usize>,
    ops: Arc<Vec<Op>>,
    spills: Arc<Vec<Arc<SlotGpu>>>,
    capture_full: bool,
    dropped: u64,
}

struct SlotGpu {
    device: ID3D12Device,
    ty: D3D12_QUERY_HEAP_TYPE,
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    cpu: usize,
}

#[derive(Clone, Copy)]
struct QueueInfo {
    id: usize,
    ty: i32,
    hz: u64,
}

#[derive(Clone)]
struct Queue {
    info: QueueInfo,
    device: Arc<ID3D12Device>,
}

struct Slot {
    gpu: Arc<SlotGpu>,
    read_gate: Arc<Mutex<()>>,
    recording: Option<Recording>,
}

struct Readback {
    base: Arc<SlotGpu>,
    spills: Arc<Vec<Arc<SlotGpu>>>,
    ops: Arc<Vec<Op>>,
    dropped: u64,
}

#[derive(Clone)]
struct Marker {
    device: Arc<ID3D12Device>,
    fence: Arc<ID3D12Fence>,
    value: u64,
}

struct Submission {
    frame: u64,
    queue: QueueInfo,
    seq: u64,
    lineage_id: u64,
    uncovered: bool,
    recordings: Vec<(usize, usize)>,
    folded_drops: u64,
    dispatch_clock: Option<DispatchClock>,
}

struct DispatchClock {
    gpu_ref: u64,
    qpc_ref: u64,
    qpc_hz: i64,
    raw_queue: usize,
    fence: usize,
    fence_value: u64,
    signal_qpc: i64,
    signal_tid: u32,
}

struct Flight {
    submission: Submission,
    marker: Marker,
    covered: Vec<Submission>,
}

struct Parked {
    queue: Arc<ID3D12CommandQueue>,
    submissions: Vec<Submission>,
}

impl std::ops::Deref for Flight {
    type Target = Submission;

    fn deref(&self) -> &Submission {
        &self.submission
    }
}

#[derive(Default)]
struct State {
    frame: u64,
    ops_seen: u32,
    next_list: usize,
    next_queue: usize,
    next_flight: usize,
    lists: HashMap<usize, usize>,
    slots: Vec<Slot>,
    opening_slots: usize,
    free: Vec<usize>,
    available_spills: Vec<Arc<SlotGpu>>,
    quarantined_spills: Vec<Arc<SlotGpu>>,
    faulted_devices: HashSet<usize>,
    spill_count: usize,
    queues: HashMap<usize, Queue>,
    registering: HashSet<usize>,
    flights: HashMap<usize, Flight>,
    parked: HashMap<usize, Parked>,
    available_markers: Vec<Marker>,
    quarantined_markers: Vec<Marker>,
    marker_count: usize,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

thread_local! {
    static IN_OP: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static NATIVE_QUEUE: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

struct OpGuard;

impl Drop for OpGuard {
    fn drop(&mut self) {
        IN_OP.with(|active| {
            active.borrow_mut().pop();
        });
    }
}

struct NativeQueue(Option<(usize, usize)>);

impl NativeQueue {
    fn enter(raw: *mut c_void, id: usize) -> Self {
        Self(NATIVE_QUEUE.with(|queue| queue.replace(Some((raw as usize, id)))))
    }
}

impl Drop for NativeQueue {
    fn drop(&mut self) {
        NATIVE_QUEUE.with(|queue| queue.set(self.0));
    }
}

pub(super) unsafe fn native<F: Copy>(object: *mut c_void, slot: usize) -> F {
    let addr = if super::real::is_adopted(object) {
        super::real::orig_fn(object, slot)
    } else {
        *(*object.cast::<*const usize>()).add(slot)
    };
    std::mem::transmute_copy(&addr)
}

type EndQuery = unsafe extern "system" fn(*mut c_void, *mut c_void, D3D12_QUERY_TYPE, u32);
type ResolveQuery = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    D3D12_QUERY_TYPE,
    u32,
    u32,
    *mut c_void,
    u64,
);

impl SlotGpu {
    unsafe fn new(device: ID3D12Device, ty: D3D12_QUERY_HEAP_TYPE) -> Option<Self> {
        let mut heap = None;
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: ty,
                    Count: QUERIES,
                    NodeMask: 0,
                },
                &mut heap,
            )
            .ok()?;
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BYTES,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        super::title_textures::own_work(|| {
            device.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
        })
        .ok()?;
        let readback = readback?;
        let mut cpu = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut cpu)).ok()?;
        if cpu.is_null() {
            return None;
        }
        Some(Self {
            device,
            ty,
            heap: heap?,
            readback,
            cpu: cpu as usize,
        })
    }
}

impl Slot {
    fn count(&self) -> u64 {
        self.recording
            .as_ref()
            .map_or(0, |r| r.ops.len() as u64 + 1 + r.dropped)
    }

    fn start(&mut self) -> Option<Arc<SlotGpu>> {
        let r = self.recording.as_mut().unwrap();
        (!std::mem::replace(&mut r.started, true)).then(|| self.gpu.clone())
    }

    fn timestamps(&self) -> u64 {
        self.recording
            .as_ref()
            .filter(|r| r.sealed)
            .map_or(0, |r| 2 * (r.ops.len() as u64 + 1))
    }

    fn snapshot(&self) -> Option<Readback> {
        let r = self
            .recording
            .as_ref()
            .filter(|r| r.sealed && !r.quarantined)?;
        Some(Readback {
            base: self.gpu.clone(),
            spills: r.spills.clone(),
            ops: r.ops.clone(),
            dropped: r.dropped,
        })
    }

    #[cfg(test)]
    unsafe fn collect(&self, batch: &mut Batch) {
        batch.timestamps += self.timestamps();
        if let Some(readback) = self.snapshot() {
            readback.collect(batch);
        } else {
            batch.dropped += self.count();
        }
    }
}

impl Readback {
    unsafe fn collect(&self, batch: &mut Batch) {
        batch.dropped += self.dropped;
        if batch.hz == 0 {
            batch.dropped += self.ops.len() as u64 + 1;
            return;
        }
        batch.spans.reserve(self.ops.len() + 1);
        let list = Op {
            kind: Kind::List,
            pso: 0,
            shaders: None,
            grid: None,
            sampled: false,
        };
        for (i, op) in std::iter::once(&list).chain(self.ops.iter()).enumerate() {
            let page = i / PAIRS_PER_PAGE;
            let pair = i % PAIRS_PER_PAGE;
            let gpu = if page == 0 {
                &self.base
            } else {
                &self.spills[page - 1]
            };
            let cpu = gpu.cpu as *const u64;
            let begin = cpu.add(2 * pair).read_volatile();
            let end = cpu.add(2 * pair + 1).read_volatile();
            if end < begin {
                batch.dropped += 1;
            } else {
                batch.spans.push(Span {
                    kind: op.kind,
                    pso: op.pso,
                    shaders: op.shaders,
                    grid: op.grid,
                    sampled: op.sampled,
                    begin,
                    end,
                });
            }
        }
    }
}

impl QueueInfo {
    fn batch(self, frame: u64) -> Batch {
        Batch {
            frame,
            queue: self.id,
            ty: self.ty,
            hz: self.hz,
            spans: Vec::new(),
            dropped: 0,
            timestamps: 0,
        }
    }
}

impl State {
    fn release_if_idle(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        if slot
            .recording
            .as_ref()
            .is_some_and(|r| !r.owned && r.in_flight == 0 && !r.quarantined)
        {
            let recording = slot.recording.take().unwrap();
            self.available_spills
                .extend(recording.spills.iter().cloned());
            self.free.push(index);
        }
    }

    fn forget(&mut self, list: usize) {
        if let Some(index) = self.lists.remove(&list) {
            let slot = &mut self.slots[index];
            slot.recording.as_mut().unwrap().owned = false;
            self.release_if_idle(index);
        }
    }

    fn retire(
        &mut self,
        index: usize,
        generation: usize,
        readable: Option<usize>,
        quarantine: bool,
    ) {
        let slot = &mut self.slots[index];
        let Some(r) = slot
            .recording
            .as_mut()
            .filter(|r| r.id == generation && r.in_flight != 0)
        else {
            debug_assert!(false, "GPU timing retirement without its recording");
            return;
        };
        if quarantine && !r.quarantined {
            self.faulted_devices
                .insert(slot.gpu.device.as_raw() as usize);
            r.quarantined = true;
            let spills = std::mem::take(&mut r.spills);
            self.spill_count -= spills.len();
            self.quarantined_spills.extend(spills.iter().cloned());
        }
        if quarantine || r.readable_flight == readable {
            r.readable_flight = None;
        }
        r.in_flight -= 1;
        self.release_if_idle(index);
    }

    fn take_marker(&mut self, device: &ID3D12Device) -> Option<Marker> {
        if let Some(index) = self
            .available_markers
            .iter()
            .position(|m| m.device.as_raw() == device.as_raw() && m.value < u64::MAX - 1)
        {
            let mut marker = self.available_markers.swap_remove(index);
            marker.value += 1;
            return Some(marker);
        }
        self.marker_count += 1;
        None
    }
}

unsafe fn get_queue(raw: *mut c_void) -> Option<Queue> {
    if let Some((queue, id)) = NATIVE_QUEUE.with(Cell::get) {
        if queue != raw as usize || id == 0 {
            return None;
        }
        return STATE
            .lock()
            .unwrap()
            .queues
            .get(&queue)
            .filter(|q| q.info.id == id)
            .cloned();
    }
    if let Some(id) = queue_token(raw) {
        return STATE
            .lock()
            .unwrap()
            .queues
            .get(&(raw as usize))
            .filter(|q| q.info.id == id)
            .cloned();
    }
    let key = raw as usize;
    let id = {
        let mut state = STATE.lock().unwrap();
        let id = state.next_queue.checked_add(1)?;
        if state.queues.len() + state.registering.len() >= MAX_QUEUES
            || !state.registering.insert(key)
        {
            return None;
        }
        state.next_queue = id;
        id
    };
    let made = (|| {
        if let Some(id) = queue_token(raw) {
            return STATE
                .lock()
                .unwrap()
                .queues
                .get(&key)
                .filter(|q| q.info.id == id)
                .cloned();
        }
        let object = ID3D12CommandQueue::from_raw_borrowed(&raw)?;
        let mut dev: Option<ID3D12Device> = None;
        object.GetDevice(&mut dev).ok()?;
        let device = Arc::new(dev?);
        let mut desc = D3D12_COMMAND_QUEUE_DESC::default();
        type GetDesc = unsafe extern "system" fn(*mut c_void, *mut D3D12_COMMAND_QUEUE_DESC);
        native::<GetDesc>(raw, 18)(raw, &mut desc);
        let mut hz = 0;
        type Frequency = unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT;
        if native::<Frequency>(raw, 16)(raw, &mut hz).is_err() {
            hz = 0;
        }
        if hz == 0 {
            NO_QUEUE_CLOCK.fetch_add(1, Ordering::Relaxed);
        }
        object
            .SetPrivateData(
                &QUEUE_TOKEN,
                std::mem::size_of_val(&id) as u32,
                Some((&id as *const usize).cast()),
            )
            .ok()?;
        let q = Queue {
            info: QueueInfo {
                id,
                ty: desc.Type.0,
                hz,
            },
            device,
        };
        let previous = STATE.lock().unwrap().queues.insert(key, q.clone());
        drop(previous);
        Some(q)
    })();
    STATE.lock().unwrap().registering.remove(&key);
    made
}

pub(super) unsafe fn queue_created(raw: *mut c_void) {
    if enabled() {
        if queue_token(raw).is_none() {
            let previous = STATE.lock().unwrap().queues.remove(&(raw as usize));
            drop(previous);
        }
        let _ = get_queue(raw);
    }
}

unsafe fn poll(queue: Option<usize>) {
    let snapshots = {
        let state = STATE.lock().unwrap();
        state
            .flights
            .iter()
            .filter(|(_, flight)| {
                queue.is_none_or(|id| flight.queue.id == id || flight.queue.id == 0)
            })
            .map(|(&id, flight)| (id, flight.marker.clone()))
            .collect::<Vec<_>>()
    };
    for (id, marker) in snapshots {
        let completed = marker.fence.GetCompletedValue();
        if completed < marker.value {
            continue;
        }
        let flight = STATE.lock().unwrap().flights.remove(&id);
        let Some(flight) = flight else { continue };
        let valid = completed == marker.value;
        if valid {
            let fence = marker.fence.as_raw() as usize;
            lineage::retired(flight.lineage_id, id, fence, marker.value);
            for covered in &flight.covered {
                lineage::retired(covered.lineage_id, id, fence, marker.value);
            }
        }
        finish(flight.submission, Some(id), valid);
        for covered in flight.covered {
            finish(covered, None, valid);
        }
        {
            let mut state = STATE.lock().unwrap();
            if !valid {
                state.quarantined_markers.push(flight.marker);
                state.marker_count -= 1;
            } else if state.marker_count > MAX_FLIGHTS {
                state.marker_count -= 1;
            } else {
                state.available_markers.push(flight.marker);
            }
        }
    }
}

unsafe fn finish(submission: Submission, readable: Option<usize>, valid: bool) {
    let mut batch = submission.queue.batch(submission.frame);
    batch.dropped = submission.folded_drops;
    for (index, generation) in submission.recordings {
        let gate = (valid && readable.is_some())
            .then(|| STATE.lock().unwrap().slots[index].read_gate.clone());
        let _read = gate.as_ref().map(|gate| gate.lock().unwrap());
        let (snapshot, depth, count, timestamps) = {
            let state = STATE.lock().unwrap();
            let slot = &state.slots[index];
            let Some(recording) = slot.recording.as_ref().filter(|r| r.id == generation) else {
                debug_assert!(false, "GPU timing readback without its recording");
                continue;
            };
            let snapshot = if valid
                && readable.is_some()
                && recording.readable_flight == readable
                && recording.in_flight == 1
            {
                slot.snapshot()
            } else {
                None
            };
            (
                snapshot,
                recording.depth_capture.as_ref().filter(|c| c.first_flight() == readable && readable.is_some()).cloned(),
                slot.count(),
                if valid { slot.timestamps() } else { 0 },
            )
        };
        batch.timestamps += timestamps;
        if let Some(depth) = depth {
            if snapshot.is_some() {
                depth.retired(generation, submission.queue.id, submission.seq);
            } else {
                crate::xlog_warn!("depth capture: skipped unreadable or replayed recording {generation}, q{} seq {}", submission.queue.id, submission.seq);
            }
        }
        if let Some(snapshot) = snapshot {
            snapshot.collect(&mut batch);
        } else {
            batch.dropped += count;
        }
        STATE
            .lock()
            .unwrap()
            .retire(index, generation, readable, !valid);
    }
    if valid {
        let bounds = if submission.uncovered || batch.dropped != 0 || batch.hz == 0 {
            None
        } else {
            batch
                .spans
                .iter()
                .filter(|span| span.kind == Kind::List)
                .map(|span| (span.begin, span.end))
                .reduce(|(begin, end), (next_begin, next_end)| {
                    (begin.min(next_begin), end.max(next_end))
                })
        };
        waits::completed(batch.queue, submission.seq, batch.hz, bounds);
    }
    if let Some((clock, (_, selected))) = submission.dispatch_clock.as_ref().zip(dispatch_qpc_selection()) {
        let DispatchClock { gpu_ref, qpc_ref, qpc_hz, raw_queue, fence, fence_value, signal_qpc, signal_tid } = clock;
        let (queue, hz, seq) = (batch.queue, batch.hz, submission.seq);
        for span in &batch.spans {
            if span.kind != Kind::Dispatch || !span.sampled || span.begin == 0 {
                continue;
            }
            let (Some((hash, 0)), Some([x, y, z])) = (span.shaders, span.grid) else { continue };
            if selected.contains(&hash) {
                crate::xlog_info!("GPU dispatch clock: CS {hash:016x} groups {x}x{y}x{z}: gpu_start={} gpu_end={} gpu_ref={gpu_ref} qpc_ref={qpc_ref} gpu_hz={hz} qpc_hz={qpc_hz} timing_queue={queue} queue={raw_queue:#x} submit_seq={seq} submission_fence={fence:#x} fence_value={fence_value} signal_qpc={signal_qpc} signal_tid={signal_tid}", span.begin, span.end);
            }
        }
    }
    stats::record(batch);
}

unsafe fn allocate_slot(device: ID3D12Device, ty: D3D12_QUERY_HEAP_TYPE) -> Option<Arc<SlotGpu>> {
    if device.GetNodeCount() != 1 {
        UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if ty == D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS3).cast(),
                std::mem::size_of_val(&options) as u32,
            )
            .is_err()
        {
            UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if !options.CopyQueueTimestampQueriesSupported.as_bool() {
            UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    }
    SlotGpu::new(device, ty).map(Arc::new).or_else(|| {
        UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
        None
    })
}

unsafe fn allocate_spill(base: &SlotGpu) -> Option<Arc<SlotGpu>> {
    {
        let mut state = STATE.lock().unwrap();
        if state
            .faulted_devices
            .contains(&(base.device.as_raw() as usize))
        {
            return None;
        }
        if let Some(i) = state
            .available_spills
            .iter()
            .rposition(|page| page.device.as_raw() == base.device.as_raw() && page.ty == base.ty)
        {
            return Some(state.available_spills.swap_remove(i));
        }
        if state.spill_count == MAX_SPILL_PAGES {
            return None;
        }
        state.spill_count += 1;
    }
    let made = SlotGpu::new(base.device.clone(), base.ty).map(Arc::new);
    if made.is_none() {
        UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
        STATE.lock().unwrap().spill_count -= 1;
    }
    made
}

pub(super) unsafe fn opened(raw: *mut c_void, pipeline: usize) {
    let lineage_generation = lineage::opened(raw as usize);
    if !enabled() || raw.is_null() {
        return;
    }
    STATE.lock().unwrap().forget(raw as usize);
    if super::title_textures::is_own_work() {
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&raw) else {
        return;
    };
    let list_type = list.GetType();
    let ty = match list_type {
        D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE => {
            D3D12_QUERY_HEAP_TYPE_TIMESTAMP
        }
        D3D12_COMMAND_LIST_TYPE_COPY => D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP,
        D3D12_COMMAND_LIST_TYPE_BUNDLE => return,
        _ => {
            UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    let mut device: Option<ID3D12Device> = None;
    if list.GetDevice(&mut device).is_err() {
        UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Some(device) = device else { return };
    enum Claim {
        Ready(usize),
        Grow,
    }
    let claim = {
        let mut state = STATE.lock().unwrap();
        if state.faulted_devices.contains(&(device.as_raw() as usize)) {
            return;
        }
        if let Some(i) = state.free.iter().rposition(|&i| {
            let slot = &state.slots[i].gpu;
            slot.device.as_raw() == device.as_raw() && slot.ty == ty
        }) {
            Claim::Ready(state.free.swap_remove(i))
        } else if state.slots.len() + state.opening_slots < SLOTS {
            state.opening_slots += 1;
            Claim::Grow
        } else {
            CAPACITY_DROPS.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    let index = match claim {
        Claim::Ready(i) => i,
        Claim::Grow => {
            let gpu = allocate_slot(device, ty);
            let mut state = STATE.lock().unwrap();
            state.opening_slots -= 1;
            let Some(gpu) = gpu else { return };
            let i = state.slots.len();
            state.slots.push(Slot {
                gpu,
                read_gate: Arc::new(Mutex::new(())),
                recording: None,
            });
            i
        }
    };
    let id = {
        let mut state = STATE.lock().unwrap();
        let device = state.slots[index].gpu.device.as_raw() as usize;
        if state.faulted_devices.contains(&device) {
            state.free.push(index);
            return;
        }
        let Some(id) = lineage_generation.or_else(|| {
            (!lineage::enabled()).then(|| state.next_list.checked_add(1)).flatten()
        }) else {
            state.free.push(index);
            return;
        };
        state.next_list = state.next_list.max(id);
        id
    };
    if list
        .SetPrivateData(
            &LIST_TOKEN,
            std::mem::size_of_val(&id) as u32,
            Some((&id as *const usize).cast()),
        )
        .is_err()
    {
        STATE.lock().unwrap().free.push(index);
        return;
    }
    let start = {
        let mut state = STATE.lock().unwrap();
        state.lists.insert(raw as usize, index);
        let slot = &mut state.slots[index];
        slot.recording = Some(Recording {
            id,
            pipeline,
            owned: true,
            ..Default::default()
        });
        (list_type != D3D12_COMMAND_LIST_TYPE_DIRECT)
            .then(|| slot.start())
            .flatten()
    };
    if let Some(gpu) = start {
        native::<EndQuery>(raw, 53)(raw, gpu.heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
}

pub(super) unsafe fn render_pass(list: *mut c_void, flags: Option<D3D12_RENDER_PASS_FLAGS>) {
    if !enabled() {
        return;
    }
    let start = {
        let mut state = STATE.lock().unwrap();
        let Some(&index) = state.lists.get(&(list as usize)) else {
            return;
        };
        let slot = &mut state.slots[index];
        let r = slot.recording.as_mut().unwrap();
        let suspends = |flags: D3D12_RENDER_PASS_FLAGS| {
            flags.0 & D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS.0 != 0
        };
        match flags {
            Some(flags) => {
                r.pass = Some(flags);
                (!r.sealed && !r.suspended && !suspends(flags))
                    .then(|| slot.start())
                    .flatten()
            }
            None => {
                r.suspended |= r.pass.take().is_some_and(suspends);
                None
            }
        }
    };
    if let Some(gpu) = start {
        native::<EndQuery>(list, 53)(list, gpu.heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
}

pub(super) unsafe fn before_close(raw: *mut c_void) {
    if !enabled() {
        return;
    }
    let (pages, pairs) = {
        let mut state = STATE.lock().unwrap();
        let Some(&index) = state.lists.get(&(raw as usize)) else {
            return;
        };
        let slot = &mut state.slots[index];
        let r = slot.recording.as_mut().unwrap();
        if !r.started || r.sealed || r.pass.is_some() || r.suspended {
            return;
        }
        r.sealed = true;
        (
            std::iter::once(&slot.gpu)
                .chain(r.spills.iter())
                .cloned()
                .collect::<Vec<_>>(),
            r.ops.len() + 1,
        )
    };
    native::<EndQuery>(raw, 53)(raw, pages[0].heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    let resolve: ResolveQuery = native(raw, 54);
    for (i, gpu) in pages.into_iter().enumerate() {
        let pairs = pairs.saturating_sub(i * PAIRS_PER_PAGE).min(PAIRS_PER_PAGE);
        if pairs == 0 {
            break;
        }
        resolve(
            raw,
            gpu.heap.as_raw(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            0,
            2 * pairs as u32,
            gpu.readback.as_raw(),
            0,
        );
    }
}

pub(super) fn forget(list: usize) {
    lineage::forget(list, None);
    if enabled() {
        STATE.lock().unwrap().forget(list);
    }
}

pub(super) unsafe fn attach_depth(raw: *mut c_void, capture: Arc<super::depth_capture::Capture>) -> Result<usize, String> {
    let (id, start) = {
        let mut state = STATE.lock().unwrap();
        let index = *state.lists.get(&(raw as usize)).ok_or("no timed list generation")?;
        let slot = &mut state.slots[index];
        let r = slot.recording.as_mut().ok_or("no active recording")?;
        if !r.owned || r.sealed || r.quarantined || r.in_flight != 0 || r.pass.is_some() || r.suspended || r.depth_capture.is_some() {
            return Err("recording is not open outside a render pass".into());
        }
        r.depth_capture = Some(capture);
        (r.id, slot.start())
    };
    if let Some(gpu) = start {
        native::<EndQuery>(raw, 53)(raw, gpu.heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    Ok(id)
}

pub(super) fn query_scope_open(list: usize, generation: u64) -> bool {
    let state = STATE.lock().unwrap();
    state.lists.get(&list).and_then(|i| state.slots.get(*i)).is_some_and(|s|
        !state.faulted_devices.contains(&(s.gpu.device.as_raw() as usize)) && s.recording.as_ref().is_some_and(|r|
            r.id as u64 == generation && r.owned && !r.sealed && !r.quarantined && r.in_flight == 0 && r.pass.is_none() && !r.suspended))
}

pub(super) fn pipeline(list: *mut c_void, pso: usize) {
    if !enabled() {
        return;
    }
    let mut state = STATE.lock().unwrap();
    if let Some(&index) = state.lists.get(&(list as usize)) {
        state.slots[index].recording.as_mut().unwrap().pipeline = pso;
    }
}

unsafe fn get_token(raw: *mut c_void, guid: &GUID) -> Option<usize> {
    let object = ID3D12Object::from_raw_borrowed(&raw)?;
    let mut id = 0usize;
    let mut bytes = std::mem::size_of_val(&id) as u32;
    object
        .GetPrivateData(guid, &mut bytes, Some((&mut id as *mut usize).cast()))
        .ok()?;
    (bytes == std::mem::size_of_val(&id) as u32 && id != 0).then_some(id)
}

pub(super) unsafe fn list_token(raw: *mut c_void) -> Option<(usize, usize)> {
    let generation = lineage::recording(raw as usize);
    if generation != 0 {
        return Some((raw as usize, generation as usize));
    }
    if !enabled() {
        return None;
    }
    get_token(raw, &LIST_TOKEN).map(|id| (raw as usize, id))
}

pub(super) fn list_released(token: Option<(usize, usize)>) {
    let Some((list, id)) = token else { return };
    lineage::forget(list, Some(id as u64));
    let mut state = STATE.lock().unwrap();
    if state.lists.get(&list).is_some_and(|&index| {
        state.slots[index]
            .recording
            .as_ref()
            .is_some_and(|r| r.id == id)
    }) {
        state.forget(list);
    }
}

pub(super) unsafe fn queue_token(raw: *mut c_void) -> Option<usize> {
    if !enabled() {
        return None;
    }
    get_token(raw, &QUEUE_TOKEN)
}

pub(super) fn queue_released(id: Option<usize>) {
    let Some(id) = id else { return };
    let closed = {
        let mut state = STATE.lock().unwrap();
        let key = state
            .queues
            .iter()
            .find_map(|(&key, q)| (q.info.id == id).then_some(key));
        key.and_then(|key| state.queues.remove(&key))
    };
    drop(closed);
}

pub(super) unsafe fn op<R>(
    raw: *mut c_void,
    kind: Kind,
    pso: impl FnOnce() -> usize,
    f: impl FnOnce() -> R,
) -> R {
    capture(raw, kind, Some(pso), None, f)
}

pub(super) unsafe fn op_grid<R>(
    raw: *mut c_void,
    kind: Kind,
    pso: impl FnOnce() -> usize,
    grid: [u32; 3],
    f: impl FnOnce() -> R,
) -> R {
    capture(raw, kind, Some(pso), Some(grid), f)
}

pub(super) unsafe fn class<R>(raw: *mut c_void, kind: Kind, f: impl FnOnce() -> R) -> R {
    capture(raw, kind, None::<fn() -> usize>, None, f)
}

unsafe fn capture<R>(
    raw: *mut c_void,
    kind: Kind,
    pso: Option<impl FnOnce() -> usize>,
    grid: Option<[u32; 3]>,
    f: impl FnOnce() -> R,
) -> R {
    if !enabled()
        || IN_OP.with(|active| {
            let mut active = active.borrow_mut();
            if active.contains(&(raw as usize)) {
                true
            } else {
                active.push(raw as usize);
                false
            }
        })
    {
        return f();
    }
    let _guard = OpGuard;
    let full = classes_enabled();
    let detailed_dispatch = full
        && kind == Kind::Dispatch
        && pso.is_some()
        && grid
            .zip(dispatch_min_groups())
            .is_some_and(|([x, y, z], threshold)| {
                (u64::from(x) * u64::from(y)).saturating_mul(u64::from(z)) >= threshold
            });
    let (start, sample) = {
        let mut state = STATE.lock().unwrap();
        let State {
            lists,
            slots,
            ops_seen,
            faulted_devices,
            ..
        } = &mut *state;
        match lists.get(&(raw as usize)) {
            Some(&index) => {
                let slot = &mut slots[index];
                let r = slot.recording.as_ref().unwrap();
                if r.sealed
                    || r.suspended
                    || r.quarantined
                    || faulted_devices.contains(&(slot.gpu.device.as_raw() as usize))
                {
                    (None, None)
                } else {
                    let start = slot.start();
                    let r = slot.recording.as_mut().unwrap();
                    let sampled = if pso.is_some() {
                        *ops_seen += 1;
                        if *ops_seen >= stride() {
                            *ops_seen = 0;
                            true
                        } else {
                            detailed_dispatch
                        }
                    } else {
                        false
                    };
                    let pair = r.ops.len() + 1;
                    let page = pair / PAIRS_PER_PAGE;
                    let sample = if !full && !sampled {
                        None
                    } else if r.capture_full || page == MAX_LIST_PAGES {
                        r.capture_full = true;
                        r.dropped = r.dropped.saturating_add(1);
                        CAPACITY_DROPS.fetch_add(1, Ordering::Relaxed);
                        None
                    } else {
                        let gpu = if page == 0 {
                            Some(slot.gpu.clone())
                        } else {
                            r.spills.get(page - 1).cloned()
                        };
                        let grow = gpu.is_none().then(|| slot.gpu.clone());
                        Some((
                            index,
                            r.id,
                            gpu,
                            grow,
                            2 * (pair % PAIRS_PER_PAGE) as u32,
                            r.pipeline,
                            sampled,
                        ))
                    };
                    (start, sample)
                }
            }
            None => (None, None),
        }
    };
    if let Some(gpu) = start {
        native::<EndQuery>(raw, 53)(raw, gpu.heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    let Some((index, generation, gpu, grow, query, pipeline, sampled)) = sample else {
        return f();
    };
    let gpu = if let Some(gpu) = gpu {
        gpu
    } else {
        let new = allocate_spill(grow.as_deref().unwrap());
        let mut state = STATE.lock().unwrap();
        let current = state.slots[index]
            .recording
            .as_mut()
            .filter(|r| r.id == generation && r.owned);
        match (current, new) {
            (Some(r), Some(new)) => {
                Arc::make_mut(&mut r.spills).push(new.clone());
                new
            }
            (Some(r), None) => {
                r.capture_full = true;
                r.dropped = r.dropped.saturating_add(1);
                CAPACITY_DROPS.fetch_add(1, Ordering::Relaxed);
                drop(state);
                return f();
            }
            (None, new) => {
                state.available_spills.extend(new);
                drop(state);
                return f();
            }
        }
    };
    let end: EndQuery = native(raw, 53);
    let (pso, shaders) = if sampled {
        let pso = pso.unwrap()();
        let pso = if pso == 0 { pipeline } else { pso };
        (pso, stats::shaders(pso))
    } else {
        (0, None)
    };
    end(raw, gpu.heap.as_raw(), D3D12_QUERY_TYPE_TIMESTAMP, query);
    let value = f();
    end(
        raw,
        gpu.heap.as_raw(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        query + 1,
    );
    let mut state = STATE.lock().unwrap();
    if let Some(r) = state.slots[index]
        .recording
        .as_mut()
        .filter(|r| r.id == generation)
    {
        Arc::make_mut(&mut r.ops).push(Op {
            kind,
            pso,
            shaders,
            grid,
            sampled,
        });
    }
    value
}

fn observed_submit(
    raw: *mut c_void,
    queue: usize,
    frame: u64,
    lists: &[*mut c_void],
    f: impl FnOnce(),
) -> (u64, u64) {
    let seq = waits::submitted(queue, frame);
    let lineage_id = lineage::enter(raw as usize, queue, seq, lists);
    {
        let _native = NativeQueue::enter(raw, queue);
        f();
    }
    waits::returned(queue, seq);
    lineage::returned(lineage_id);
    (seq, lineage_id)
}

unsafe fn marker_for(device: &ID3D12Device) -> Option<Marker> {
    #[cfg(test)]
    if WARP_FAIL_MARKER.with(Cell::get) {
        return None;
    }
    let available = STATE.lock().unwrap().take_marker(device);
    if available.is_some() {
        return available;
    }
    let Some(fence) = super::real::create_residency_fence(device).ok() else {
        STATE.lock().unwrap().marker_count -= 1;
        return None;
    };
    Some(Marker {
        device: Arc::new(device.clone()),
        fence: Arc::new(fence),
        value: 1,
    })
}

fn park(queue: &Arc<ID3D12CommandQueue>, mut submission: Submission, flight: Option<usize>) {
    let raw = queue.as_raw() as usize;
    let mut state = STATE.lock().unwrap();
    let existing: HashSet<_> = state
        .parked
        .get(&raw)
        .into_iter()
        .flat_map(|parked| {
            parked
                .submissions
                .iter()
                .flat_map(|s| s.recordings.iter().copied())
        })
        .collect();
    submission.recordings.retain(|&(index, generation)| {
        let slot = &mut state.slots[index];
        let Some(r) = slot.recording.as_mut().filter(|r| r.id == generation) else {
            debug_assert!(false, "GPU timing parked submission without its recording");
            return false;
        };
        if r.readable_flight == flight {
            r.readable_flight = None;
        }
        if existing.contains(&(index, generation)) {
            submission.folded_drops += slot.count();
            state.retire(index, generation, None, false);
            return false;
        }
        true
    });
    if submission.recordings.is_empty() {
        drop(state);
        if submission.folded_drops != 0 {
            let mut batch = submission.queue.batch(submission.frame);
            batch.dropped = submission.folded_drops;
            stats::record(batch);
        }
    } else {
        state
            .parked
            .entry(raw)
            .or_insert_with(|| Parked {
                queue: queue.clone(),
                submissions: Vec::new(),
            })
            .submissions
            .push(submission);
    }
}

unsafe fn arm(
    raw: *mut c_void,
    id: usize,
    mut submission: Submission,
    covered: Vec<Submission>,
    marker: Marker,
) {
    type Signal = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
    let armed = {
        let _native = NativeQueue::enter(raw, submission.queue.id);
        if submission.queue.hz != 0 && dispatch_qpc_selection().is_some_and(|(q, _)| *q == submission.queue.id) {
            use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
            let (mut gpu_ref, mut qpc_ref, mut qpc_hz, mut signal_qpc) = (0, 0, 0, 0);
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                if QueryPerformanceFrequency(&mut qpc_hz) != 0
                    && qpc_hz > 0
                    && q.GetClockCalibration(&mut gpu_ref, &mut qpc_ref).is_ok()
                    && QueryPerformanceCounter(&mut signal_qpc) != 0
                {
                    submission.dispatch_clock = Some(DispatchClock {
                        gpu_ref, qpc_ref, qpc_hz, raw_queue: raw as usize,
                        fence: marker.fence.as_raw() as usize, fence_value: marker.value,
                        signal_qpc, signal_tid: windows_sys::Win32::System::Threading::GetCurrentThreadId(),
                    });
                }
            }
        }
        native::<Signal>(raw, 14)(raw, marker.fence.as_raw(), marker.value).is_ok()
    };
    if armed {
        STATE.lock().unwrap().flights.insert(
            id,
            Flight {
                submission,
                marker,
                covered,
            },
        );
    } else {
        {
            let mut state = STATE.lock().unwrap();
            state
                .faulted_devices
                .insert(marker.device.as_raw() as usize);
            state.marker_count -= 1;
            state.quarantined_markers.push(marker);
        }
        finish(submission, Some(id), false);
        for covered in covered {
            finish(covered, None, false);
        }
    }
}

unsafe fn retry_parked() {
    let (raw, parked, base, id) = {
        let mut state = STATE.lock().unwrap();
        if state.parked.is_empty() {
            return;
        }
        let raw = *state
            .parked
            .keys()
            .nth(state.frame as usize % state.parked.len())
            .unwrap();
        let parked = state.parked.remove(&raw).unwrap();
        let index = parked.submissions[0].recordings[0].0;
        let base = state.slots[index].gpu.clone();
        state.next_flight = state
            .next_flight
            .checked_add(1)
            .expect("GPU timing submission ID exhausted");
        (raw, parked, base, state.next_flight)
    };
    let Parked { queue, submissions } = parked;
    if let Some(marker) = marker_for(&base.device) {
        let mut submissions = submissions.into_iter();
        arm(
            raw as *mut c_void,
            id,
            submissions.next().unwrap(),
            submissions.collect(),
            marker,
        );
    } else if base.device.GetDeviceRemovedReason().is_err() {
        STATE
            .lock()
            .unwrap()
            .faulted_devices
            .insert(base.device.as_raw() as usize);
        for submission in submissions {
            finish(submission, None, false);
        }
    } else {
        for submission in submissions {
            park(&queue, submission, None);
        }
    }
}

pub(super) unsafe fn submit(raw: *mut c_void, lists: &[*mut c_void], f: impl FnOnce()) {
    if raw.is_null() || lists.is_empty() {
        return f();
    }
    if !enabled() {
        let lineage_id = lineage::enter(raw as usize, 0, 0, lists);
        f();
        lineage::returned(lineage_id);
        return;
    }
    let queue = get_queue(raw);
    let info = queue.as_ref().map_or(
        QueueInfo {
            id: 0,
            ty: -1,
            hz: 0,
        },
        |q| q.info,
    );
    let poll_all = {
        let state = STATE.lock().unwrap();
        state.marker_count >= MAX_FLIGHTS && state.available_markers.is_empty()
            || lists.iter().any(|&list| {
                state
                    .lists
                    .get(&(list as usize))
                    .is_some_and(|&i| state.slots[i].recording.as_ref().unwrap().in_flight != 0)
            })
    };
    poll((!poll_all).then_some(info.id));
    let (flight_id, batch, recordings, gates, base) = {
        let mut state = STATE.lock().unwrap();
        let mut batch = info.batch(state.frame);
        let mut started: Vec<_> = lists
            .iter()
            .filter_map(|&list| {
                let &i = state.lists.get(&(list as usize))?;
                state.slots[i]
                    .recording
                    .as_ref()
                    .is_some_and(|r| r.started)
                    .then_some(i)
            })
            .collect();
        started.sort_unstable();
        started.dedup();
        let gates = started
            .iter()
            .map(|&index| state.slots[index].read_gate.clone())
            .collect::<Vec<_>>();
        let mut recordings = Vec::new();
        let id = state
            .next_flight
            .checked_add(1)
            .expect("GPU timing submission ID exhausted");
        for index in started {
            let r = state.slots[index].recording.as_mut().unwrap();
            if r.quarantined {
                r.readable_flight = None;
                continue;
            }
            r.readable_flight = (r.in_flight == 0).then_some(id);
            if let Some(capture) = &r.depth_capture {
                capture.submitted(id);
            }
            r.in_flight += 1;
            recordings.push((index, r.id));
        }
        batch.dropped = (lists.len() - recordings.len()) as u64;
        let base = recordings
            .first()
            .or_else(|| {
                state
                    .parked
                    .get(&(raw as usize))?
                    .submissions
                    .first()?
                    .recordings
                    .first()
            })
            .map(|&(index, _)| state.slots[index].gpu.clone());
        if base.is_some() {
            state.next_flight = id;
        }
        (id, batch, recordings, gates, base)
    };
    let marker = base
        .as_ref()
        .and_then(|base| marker_for(queue.as_ref().map_or(&base.device, |q| q.device.as_ref())));
    for gate in gates {
        drop(gate.lock().unwrap());
    }
    let (seq, lineage_id) = observed_submit(raw, info.id, batch.frame, lists, f);
    let Some(base) = base else {
        stats::record(batch);
        return;
    };
    let submission = Submission {
        frame: batch.frame,
        queue: info,
        seq,
        lineage_id,
        uncovered: batch.dropped != 0,
        recordings,
        folded_drops: 0,
        dispatch_clock: None,
    };
    let Some(marker) = marker else {
        UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
        let removed = base.device.GetDeviceRemovedReason().is_err();
        if removed {
            let mut state = STATE.lock().unwrap();
            state.faulted_devices.insert(base.device.as_raw() as usize);
            let parked = state.parked.remove(&(raw as usize));
            drop(state);
            finish(submission, Some(flight_id), false);
            if let Some(parked) = parked {
                for submission in parked.submissions {
                    finish(submission, None, false);
                }
            }
        } else if !submission.recordings.is_empty() {
            let queue = Arc::new(ID3D12CommandQueue::from_raw_borrowed(&raw).unwrap().clone());
            park(&queue, submission, Some(flight_id));
        }
        stats::record(batch);
        return;
    };
    let parked = STATE.lock().unwrap().parked.remove(&(raw as usize));
    let (queue_hold, covered) =
        parked.map_or((None, Vec::new()), |p| (Some(p.queue), p.submissions));
    arm(raw, flight_id, submission, covered, marker);
    drop(queue_hold);
    stats::record(batch);
}

pub(super) fn boundary() {
    if !enabled() {
        return;
    }
    unsafe {
        poll(None);
        retry_parked();
    }
    let frame = {
        let mut state = STATE.lock().unwrap();
        state.frame = state.frame.saturating_add(1);
        state.frame
    };
    stats::boundary(frame);
    if frame % 120 == 1 {
        let capacity = CAPACITY_DROPS.swap(0, Ordering::Relaxed);
        let unsupported = UNSUPPORTED.swap(0, Ordering::Relaxed);
        let unavailable = UNAVAILABLE.swap(0, Ordering::Relaxed);
        let no_clock = NO_QUEUE_CLOCK.swap(0, Ordering::Relaxed);
        if capacity != 0 || unsupported != 0 || unavailable != 0 || no_clock != 0 {
            crate::xlog_info!("GPU timing capture: capacity drops={capacity} unsupported recordings={unsupported} queues without timestamps={no_clock} resource failures={unavailable}");
        }
    }
}

pub(super) fn epoch() -> u64 {
    if !enabled() {
        return 0;
    }
    STATE.lock().unwrap().frame
}

unsafe fn checked_queue(raw: *mut c_void) -> Option<(u64, QueueInfo)> {
    if !enabled() || raw.is_null() {
        return None;
    }
    let id = match NATIVE_QUEUE.with(Cell::get) {
        Some((queue, id)) if queue == raw as usize && id != 0 => id,
        Some(_) => return None,
        None => queue_token(raw)?,
    };
    let state = STATE.lock().unwrap();
    state
        .queues
        .get(&(raw as usize))
        .filter(|queue| queue.info.id == id)
        .map(|queue| (state.frame, queue.info))
}

pub(super) unsafe fn queue_stamp(raw: *mut c_void) -> Option<(u64, usize)> {
    checked_queue(raw).map(|(epoch, queue)| (epoch, queue.id))
}

pub(super) unsafe fn wait(raw: *mut c_void) -> Option<(u64, usize)> {
    let (epoch, queue) = checked_queue(raw)?;
    stats::wait(epoch, queue.id, queue.ty);
    Some((epoch, queue.id))
}

#[cfg(test)]
#[path = "gpu_timing_reentrant_tests.rs"]
mod reentrant_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};

    unsafe fn fence_of(id: usize, index: usize) -> (Arc<ID3D12Fence>, u64) {
        let state = STATE.lock().unwrap();
        let flight = state
            .flights
            .values()
            .find(|f| f.queue.id == id && f.recordings.iter().any(|r| r.0 == index))
            .unwrap();
        (flight.marker.fence.clone(), flight.marker.value)
    }

    struct Unblock(ID3D12Fence);

    impl Drop for Unblock {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(1);
            }
        }
    }

    #[test]
    fn warp_resolves_fourteen_thousand_class_scopes_without_drops() {
        let _serial = WARP_TIMING_TEST_LOCK.lock().unwrap();
        WARP_TEST.with(|on| on.set(true));
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP adapter");
            let mut dev = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev)
                .expect("WARP D3D12 device");
            let dev: ID3D12Device = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: ty,
                    ..Default::default()
                })
                .unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let raw = list.as_raw();
            opened(raw, 0);
            let index = STATE.lock().unwrap().lists[&(raw as usize)];
            let token = list_token(raw);
            for _ in 0..14_500 {
                class(raw, Kind::Transfer, || ());
            }
            op_grid(raw, Kind::Dispatch, || 0x1234, [7, 5, 3], || ());
            before_close(raw);
            list.Close().unwrap();
            let execute = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            submit(queue.as_raw(), &[raw], || {
                queue.ExecuteCommandLists(&execute)
            });
            let queue_id = queue_token(queue.as_raw()).unwrap();
            let (fence, value) = fence_of(queue_id, index);
            crate::gpu::fence::wait(&fence, value, &dev, None, 5_000).unwrap();
            let pages = {
                let state = STATE.lock().unwrap();
                let slot = &state.slots[index];
                let r = slot.recording.as_ref().unwrap();
                assert_eq!(
                    r.spills.len() + 1,
                    (r.ops.len() + 1).div_ceil(PAIRS_PER_PAGE)
                );
                let info = QueueInfo {
                    id: 1,
                    ty: ty.0,
                    hz: 1,
                };
                let mut batch = info.batch(state.frame);
                slot.collect(&mut batch);
                assert_eq!(batch.dropped, 0);
                assert_eq!(batch.timestamps, 2 * 14_502);
                assert_eq!(batch.spans.len(), 14_502);
                let list = &batch.spans[0];
                assert_eq!(list.kind, Kind::List);
                assert_ne!(list.begin, 0);
                for span in &batch.spans[1..=14_500] {
                    assert_eq!(span.kind, Kind::Transfer);
                    assert!(!span.sampled);
                    assert!(list.begin <= span.begin && span.end <= list.end);
                }
                let dispatch = batch.spans.last().unwrap();
                assert_eq!(
                    (dispatch.kind, dispatch.pso, dispatch.grid),
                    (Kind::Dispatch, 0x1234, Some([7, 5, 3]))
                );
                assert!(dispatch.sampled);
                r.spills.iter().map(Arc::as_ptr).collect::<Vec<_>>()
            };
            poll(Some(queue_id));
            assert_eq!(
                STATE.lock().unwrap().slots[index]
                    .recording
                    .as_ref()
                    .unwrap()
                    .in_flight,
                0
            );
            drop(execute);
            drop(list);
            list_released(token);
            assert!(STATE.lock().unwrap().slots[index].recording.is_none());

            alloc.Reset().unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let raw = list.as_raw();
            opened(raw, 0);
            let token = list_token(raw);
            for _ in 0..PAIRS_PER_PAGE {
                class(raw, Kind::Transfer, || ());
            }
            {
                let state = STATE.lock().unwrap();
                let index = state.lists[&(raw as usize)];
                let spills = &state.slots[index].recording.as_ref().unwrap().spills;
                assert_eq!(spills.len(), 1);
                assert!(pages.contains(&Arc::as_ptr(&spills[0])));
            }
            drop(list);
            list_released(token);
            drop(queue);
            queue_released(Some(queue_id));
        }
        WARP_TEST.with(|on| on.set(false));
    }

    #[test]
    fn warp_default_captures_list_and_rank_only_and_full_mode_stops_at_cap() {
        let _serial = WARP_TIMING_TEST_LOCK.lock().unwrap();
        WARP_TEST.with(|on| on.set(true));
        WARP_CLASSES.with(|on| on.set(false));
        WARP_STRIDE.with(|n| n.set(4));
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev: ID3D12Device = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let raw = list.as_raw();
            opened(raw, 0);
            let token = list_token(raw);
            let index = STATE.lock().unwrap().lists[&(raw as usize)];
            for _ in 0..4 {
                class(raw, Kind::Transfer, || ());
            }
            {
                let state = STATE.lock().unwrap();
                let r = state.slots[index].recording.as_ref().unwrap();
                assert!(r.started && r.ops.is_empty());
            }
            let pso_calls = Cell::new(0);
            for _ in 0..4 {
                op(
                    raw,
                    Kind::Draw,
                    || {
                        pso_calls.set(pso_calls.get() + 1);
                        37
                    },
                    || (),
                );
            }
            assert_eq!(pso_calls.get(), 1);
            {
                let mut state = STATE.lock().unwrap();
                let r = state.slots[index].recording.as_mut().unwrap();
                assert_eq!(r.ops.len(), 1);
                assert!(r.ops[0].sampled);
                let captured = r.ops[0];
                Arc::make_mut(&mut r.ops).resize(MAX_LIST_PAGES * PAIRS_PER_PAGE - 1, captured);
            }
            WARP_CLASSES.with(|on| on.set(true));
            class(raw, Kind::Clear, || ());
            {
                let state = STATE.lock().unwrap();
                let r = state.slots[index].recording.as_ref().unwrap();
                assert!(r.capture_full);
                assert_eq!(r.dropped, 1);
                assert_eq!(r.ops.len(), MAX_LIST_PAGES * PAIRS_PER_PAGE - 1);
            }
            drop(list);
            list_released(token);
        }
        WARP_STRIDE.with(|n| n.set(1));
        WARP_TEST.with(|on| on.set(false));
    }

    #[test]
    fn warp_class_detail_ranks_dispatches_above_the_grid_threshold() {
        let _serial = WARP_TIMING_TEST_LOCK.lock().unwrap();
        WARP_TEST.with(|on| on.set(true));
        WARP_STRIDE.with(|n| n.set(u32::MAX));
        WARP_DISPATCH_MIN_GROUPS.with(|n| n.set(Some(128)));
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev: ID3D12Device = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let raw = list.as_raw();
            opened(raw, 0);
            let token = list_token(raw);
            let index = STATE.lock().unwrap().lists[&(raw as usize)];
            let small = [31, 2, 2];
            let big = [32, 2, 2];
            op_grid(
                raw,
                Kind::Dispatch,
                || panic!("below the threshold"),
                small,
                || (),
            );
            op_grid(raw, Kind::Dispatch, || 0x1234, big, || ());
            op_grid(
                raw,
                Kind::MeshCompute,
                || panic!("not a dispatch"),
                big,
                || (),
            );
            WARP_CLASSES.with(|on| on.set(false));
            op_grid(
                raw,
                Kind::Dispatch,
                || panic!("classes are disabled"),
                big,
                || (),
            );
            before_close(raw);
            list.Close().unwrap();
            {
                let state = STATE.lock().unwrap();
                let slot = &state.slots[index];
                let r = slot.recording.as_ref().unwrap();
                assert_eq!(r.ops.len(), 3);
                assert!(!r.ops[0].sampled && !r.ops[2].sampled);
                assert_eq!(
                    (r.ops[1].sampled, r.ops[1].pso, r.ops[1].grid),
                    (true, 0x1234, Some(big))
                );
                assert_eq!(slot.timestamps(), 8);
            }
            drop(list);
            list_released(token);
        }
        WARP_DISPATCH_MIN_GROUPS.with(|n| n.set(None));
        WARP_CLASSES.with(|on| on.set(true));
        WARP_STRIDE.with(|n| n.set(1));
        WARP_TEST.with(|on| on.set(false));
    }

    #[test]
    fn warp_boundary_retires_parked_work_without_another_submit() {
        let _serial = WARP_TIMING_TEST_LOCK.lock().unwrap();
        WARP_TEST.with(|on| on.set(true));
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev: ID3D12Device = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: ty,
                    ..Default::default()
                })
                .unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let unblock = Unblock(gate.clone());
            queue.Wait(&gate, 1).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let raw = list.as_raw();
            opened(raw, 0);
            let token = list_token(raw);
            let index = STATE.lock().unwrap().lists[&(raw as usize)];
            class(raw, Kind::Transfer, || ());
            before_close(raw);
            list.Close().unwrap();
            let executable = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            WARP_FAIL_MARKER.with(|fail| fail.set(true));
            submit(queue.as_raw(), &[raw], || {
                queue.ExecuteCommandLists(&executable)
            });
            WARP_FAIL_MARKER.with(|fail| fail.set(false));
            let qraw = queue.as_raw() as usize;
            let id = queue_token(queue.as_raw()).unwrap();
            {
                let state = STATE.lock().unwrap();
                assert!(state.parked.contains_key(&qraw));
                let r = state.slots[index].recording.as_ref().unwrap();
                assert_eq!(r.in_flight, 1);
                assert!(!r.quarantined);
            }
            boundary();
            let (fence, value) = fence_of(id, index);
            assert!(fence.GetCompletedValue() < value);
            drop(executable);
            drop(list);
            list_released(token);
            assert!(STATE.lock().unwrap().slots[index].recording.is_some());
            drop(unblock);
            crate::gpu::fence::wait(&fence, value, &dev, None, 5_000).unwrap();
            poll(Some(id));
            let state = STATE.lock().unwrap();
            assert!(!state.parked.contains_key(&qraw));
            assert!(state.slots[index].recording.is_none());
            drop(state);
            drop(queue);
            queue_released(Some(id));
        }
        WARP_TEST.with(|on| on.set(false));
    }

    #[test]
    fn warp_timestamps_survive_replay_and_reset() {
        let _serial = WARP_TIMING_TEST_LOCK.lock().unwrap();
        WARP_TEST.with(|on| on.set(true));
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP adapter");
            let mut dev = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev)
                .expect("WARP D3D12 device");
            let dev: ID3D12Device = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: ty,
                ..Default::default()
            };
            let first: ID3D12CommandQueue = dev.CreateCommandQueue(&desc).unwrap();
            let replay: ID3D12CommandQueue = dev.CreateCommandQueue(&desc).unwrap();
            let first_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let second_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, ty, &first_alloc, None).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let executable = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let raw = list.as_raw();
            opened(raw, 0x1234);
            let old_token = list_token(raw);
            assert!(old_token.is_some());
            let index = STATE.lock().unwrap().lists[&(raw as usize)];
            let called = Cell::new(false);
            op(
                raw,
                Kind::Draw,
                || 0,
                || {
                    op(
                        raw,
                        Kind::Indirect,
                        || panic!("nested timing sampled"),
                        || {
                            called.set(true);
                            list.ResourceBarrier(&[]);
                        },
                    );
                },
            );
            assert!(called.get());
            before_close(raw);
            list.Close().unwrap();
            submit(first.as_raw(), &[raw], || {
                first.ExecuteCommandLists(&executable)
            });
            let first_id = queue_token(first.as_raw()).unwrap();
            let (fence, value) = fence_of(first_id, index);
            crate::gpu::fence::wait(&fence, value, &dev, None, 5_000).unwrap();

            let unblock = Unblock(gate.clone());
            replay.Wait(&gate, 1).unwrap();
            submit(replay.as_raw(), &[raw], || {
                replay.ExecuteCommandLists(&executable)
            });
            let replay_id = queue_token(replay.as_raw()).unwrap();
            let (fence, value) = fence_of(replay_id, index);
            assert_ne!(first_id, replay_id);
            assert!(!STATE
                .lock()
                .unwrap()
                .flights
                .values()
                .any(|f| f.queue.id == first_id));

            list.Reset(&second_alloc, None).unwrap();
            opened(raw, 0);
            let new_token = list_token(raw);
            assert!(new_token.is_some() && new_token != old_token);
            list_released(old_token);
            let new_index = {
                let state = STATE.lock().unwrap();
                let new_index = state.lists[&(raw as usize)];
                assert_ne!(index, new_index);
                let old = state.slots[index].recording.as_ref().unwrap();
                assert!(!old.owned);
                assert_eq!(old.in_flight, 1);
                new_index
            };
            drop(unblock);
            crate::gpu::fence::wait(&fence, value, &dev, None, 5_000).unwrap();
            {
                let state = STATE.lock().unwrap();
                let q = &state.queues[&(replay.as_raw() as usize)];
                let mut batch = q.info.batch(state.frame);
                state.slots[index].collect(&mut batch);
                assert_eq!(batch.dropped, 0);
                assert_eq!(batch.spans.len(), 2);
                let whole = &batch.spans[0];
                let draw = &batch.spans[1];
                assert_eq!(
                    (whole.kind, draw.kind, draw.pso),
                    (Kind::List, Kind::Draw, 0x1234)
                );
                assert!(whole.begin != 0 && whole.begin <= draw.begin && draw.end <= whole.end);
            }
            boundary();
            {
                let state = STATE.lock().unwrap();
                assert!(state.slots[index].recording.is_none() && state.free.contains(&index));
                assert!(!state.flights.values().any(|f| f.queue.id == replay_id));
            }
            before_close(raw);
            list.Close().unwrap();
            drop(executable);
            drop(list);
            list_released(new_token);
            assert!(STATE.lock().unwrap().slots[new_index].recording.is_none());
            drop(first);
            drop(replay);
            queue_released(Some(first_id));
            queue_released(Some(replay_id));
            let state = STATE.lock().unwrap();
            assert!(!state
                .queues
                .values()
                .any(|q| q.info.id == first_id || q.info.id == replay_id));
        }
        WARP_TEST.with(|on| on.set(false));
    }
}
