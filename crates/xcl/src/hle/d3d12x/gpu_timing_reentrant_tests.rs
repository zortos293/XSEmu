use super::super::{list_x, real};
use super::*;
use std::process::Command;
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};

const FIRST: u32 = 0x3f78_2211;
const SECOND: u32 = 0x6cb4_4433;

struct Timing(bool);

impl Timing {
    fn enable() -> Self {
        Self(WARP_TEST.with(|flag| flag.replace(true)))
    }
}

impl Drop for Timing {
    fn drop(&mut self) {
        WARP_TEST.with(|flag| flag.set(self.0));
    }
}

fn with_watchdog(case: impl FnOnce()) {
    const CHILD: &str = "_XCL_WARP_TIMING_TEST_CHILD";
    const TIMEOUT: Duration = Duration::from_secs(30);
    let thread = std::thread::current();
    let name = thread.name().expect("libtest name");
    if std::env::var(CHILD).as_deref() == Ok(name) {
        let _lock = WARP_TIMING_TEST_LOCK.lock().unwrap();
        case();
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([name, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, name)
        .env_remove("XCL_GPU_LIST_TIMING")
        .spawn()
        .expect("start isolated WARP timing test");
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "{name} failed in child process: {status}");
                return;
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(25)),
            result => {
                let kill = child.kill();
                let reap = kill.is_ok().then(|| child.wait());
                panic!(
                    "{name} did not complete within {TIMEOUT:?} (poll: {result:?}, kill: {kill:?}, wait: {reap:?})"
                );
            }
        }
    }
}

struct Gate(ID3D12Fence);

impl Drop for Gate {
    fn drop(&mut self) {
        unsafe {
            let _ = self.0.Signal(1);
        }
    }
}

unsafe fn warp() -> ID3D12Device {
    let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
    let adapter: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP adapter");
    let mut dev = None;
    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
    dev.expect("native device")
}

unsafe fn adopted_queue(dev: &ID3D12Device) -> ID3D12CommandQueue {
    let queue: ID3D12CommandQueue = dev
        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })
        .expect("native DIRECT queue");
    real::readopt(queue.as_raw(), real::Kind::CommandQueue);
    assert!(real::is_adopted(queue.as_raw()));
    queue
}

unsafe fn buffer(
    dev: &ID3D12Device,
    heap: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut result = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES {
            Type: heap,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 12,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        state,
        None,
        &mut result,
    )
    .expect("marker buffer");
    result.expect("native buffer")
}

struct Markers {
    upload: ID3D12Resource,
    readback: ID3D12Resource,
}

impl Markers {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        let upload = buffer(
            dev,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let readback = buffer(
            dev,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        let mut ptr = std::ptr::null_mut();
        upload.Map(0, Some(&empty), Some(&mut ptr)).unwrap();
        ptr.cast::<[u32; 3]>().write([FIRST, SECOND, 0]);
        upload.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 12 }));
        Self { upload, readback }
    }

    unsafe fn copy(&self, list: &ID3D12GraphicsCommandList, dst: u64, src: u64) {
        list.CopyBufferRegion(&self.readback, dst, &self.upload, src, 4);
    }

    unsafe fn read(&self) -> [u32; 3] {
        let mut ptr = std::ptr::null_mut();
        self.readback
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 12 }), Some(&mut ptr))
            .unwrap();
        let values = ptr.cast::<[u32; 3]>().read();
        self.readback
            .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        values
    }
}

struct Profiled {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
    recording: Captured,
}

#[derive(Clone, Copy)]
struct Captured {
    index: usize,
    generation: usize,
    label: usize,
}

impl Captured {
    unsafe fn of(raw: *mut c_void, label: usize) -> Self {
        let (_, generation) = list_token(raw).expect("the real list must own a timing recording");
        let index = STATE.lock().unwrap().lists[&(raw as usize)];
        Self {
            index,
            generation,
            label,
        }
    }
}

impl Profiled {
    unsafe fn new(
        dev: &ID3D12Device,
        label: usize,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> Self {
        let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        let list: ID3D12GraphicsCommandList =
            dev.CreateCommandList(0, ty, &allocator, None).unwrap();
        let raw = list.as_raw();
        opened(raw, label);
        let recording = Captured::of(raw, label);
        op(raw, Kind::Indirect, || 0, || record(&list));
        before_close(raw);
        list.Close().unwrap();
        Self {
            list,
            _allocator: allocator,
            recording,
        }
    }

    unsafe fn execute(&self, queue: &ID3D12CommandQueue) {
        let raw = queue.as_raw();
        type Execute = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
        let execute: Execute = std::mem::transmute(real::orig_fn(raw, 10));
        execute(raw, 1, &self.list.as_raw());
    }
}

impl Drop for Profiled {
    fn drop(&mut self) {
        list_released(Some((
            self.list.as_raw() as usize,
            self.recording.generation,
        )));
    }
}

unsafe fn enqueue_drain(dev: &ID3D12Device, queue: &ID3D12CommandQueue) -> ID3D12Fence {
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    let raw = queue.as_raw();
    type Signal = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
    let signal: Signal = std::mem::transmute(real::orig_fn(raw, 14));
    signal(raw, fence.as_raw(), 1).ok().unwrap();
    fence
}

unsafe fn verify_spans(queue: &ID3D12CommandQueue, lists: &[Captured]) {
    let raw = queue.as_raw();
    let id = queue_token(raw).expect("native queue must be tracked");
    type Frequency = unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT;
    let frequency: Frequency = std::mem::transmute(real::orig_fn(raw, 16));
    let mut hz = 0;
    frequency(raw, &mut hz).ok().unwrap();
    assert_ne!(hz, 0);
    {
        let state = STATE.lock().unwrap();
        let mut batch = Batch {
            frame: state.frame,
            queue: id,
            ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
            hz,
            spans: Vec::new(),
            dropped: 0,
            timestamps: 0,
        };
        for list in lists {
            let slot = &state.slots[list.index];
            assert_eq!(slot.recording.as_ref().unwrap().id, list.generation);
            let start = batch.spans.len();
            slot.collect(&mut batch);
            assert_eq!(batch.dropped, 0);
            let [whole, sample] = &batch.spans[start..] else {
                panic!("expected one list span and one operation span");
            };
            assert_eq!(
                (whole.kind, sample.kind, sample.pso),
                (Kind::List, Kind::Indirect, list.label)
            );
            assert!(
                whole.begin != 0
                    && whole.begin <= sample.begin
                    && sample.begin <= sample.end
                    && sample.end <= whole.end
            );
        }
    }
    boundary();
    assert_eq!(
        stats::captures_for_test(id),
        (lists.len() * 2, lists.len() as u64 * 4, 0),
        "production poll must publish both spans and all four timestamps for each recording",
    );
}

unsafe fn verify_in_flight(queue: &ID3D12CommandQueue, lists: &[Captured]) {
    let id = queue_token(queue.as_raw()).expect("tracked queue");
    let state = STATE.lock().unwrap();
    let flights: Vec<_> = state
        .flights
        .values()
        .filter(|flight| flight.queue.id == id)
        .collect();
    assert_eq!(flights.len(), lists.len());
    for list in lists {
        assert!(
            flights
                .iter()
                .any(|flight| flight.recordings == [(list.index, list.generation)])
        );
    }
}

#[test]
fn warp_adopted_queue_wait_reenters_profiled_submit() {
    with_watchdog(|| unsafe {
        let _timing = Timing::enable();
        let dev = warp();
        let queue = adopted_queue(&dev);
        let raw = queue.as_raw();
        queue_created(raw);
        let id = queue_token(raw).expect("queue creation must register timing metadata");
        let markers = Markers::new(&dev);
        let list = Profiled::new(&dev, 0x7101, |list| {
            markers.copy(list, 0, 0);
            markers.copy(list, 4, 8);
            markers.copy(list, 8, 8);
        });
        let gate = Gate(dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap());

        let first_frame = STATE.lock().unwrap().frame;
        assert_eq!(stats::waits_for_test(first_frame, id), 0);
        submit(raw, &[list.list.as_raw()], || {
            queue.Wait(&gate.0, 1).unwrap();
            list.execute(&queue);
        });
        let last_frame = STATE.lock().unwrap().frame;
        assert_eq!(
            (first_frame..=last_frame)
                .map(|frame| stats::waits_for_test(frame, id))
                .sum::<u64>(),
            1,
            "the adopted PC Wait must re-enter the timing hook exactly once",
        );
        let drain = enqueue_drain(&dev, &queue);
        assert_eq!(gate.0.GetCompletedValue(), 0);
        assert_eq!(
            drain.GetCompletedValue(),
            0,
            "native Wait must still gate GPU progress"
        );
        verify_in_flight(&queue, &[list.recording]);
        gate.0.Signal(1).unwrap();
        crate::gpu::fence::wait(&drain, 1, &dev, None, 5_000).unwrap();
        assert_eq!(markers.read(), [FIRST, 0, 0]);
        verify_spans(&queue, &[list.recording]);
    });
}

#[test]
fn warp_nested_profiled_submissions_execute_inner_then_outer() {
    with_watchdog(|| unsafe {
        let _timing = Timing::enable();
        let dev = warp();
        let queue = adopted_queue(&dev);
        let raw = queue.as_raw();
        let markers = Markers::new(&dev);
        let inner = Profiled::new(&dev, 0x7201, |list| {
            markers.copy(list, 0, 0);
            markers.copy(list, 4, 0);
        });
        let outer = Profiled::new(&dev, 0x7202, |list| {
            markers.copy(list, 0, 4);
            markers.copy(list, 8, 4);
        });
        assert_ne!(inner.recording.generation, outer.recording.generation);
        let gate = Gate(dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap());

        queue.Wait(&gate.0, 1).unwrap();
        submit(raw, &[outer.list.as_raw()], || {
            real::execute_mapped(raw, &[inner.list.as_raw()]);
            outer.execute(&queue);
        });
        let drain = enqueue_drain(&dev, &queue);
        assert_eq!(
            drain.GetCompletedValue(),
            0,
            "native Wait must gate both executes"
        );
        let captures = [inner.recording, outer.recording];
        verify_in_flight(&queue, &captures);
        gate.0.Signal(1).unwrap();
        crate::gpu::fence::wait(&drain, 1, &dev, None, 5_000).unwrap();
        assert_eq!(
            markers.read(),
            [SECOND, FIRST, SECOND],
            "both native executes must arrive, inner first"
        );
        verify_spans(&queue, &captures);
    });
}

#[test]
fn warp_kickoff_reenters_profiled_submit_and_reopens_list() {
    with_watchdog(|| unsafe {
        let _timing = Timing::enable();
        let dev = warp();
        let queue = adopted_queue(&dev);
        let q = queue.as_raw();
        let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
        list_x::on_queue_create(q as usize, dev.as_raw() as usize, ty);
        let markers = Markers::new(&dev);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
        let raw = list.as_raw();
        real::readopt(raw, real::Kind::CommandList);
        list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
        opened(raw, 0);
        let kickoff = Captured::of(raw, 0x7301);
        op(
            raw,
            Kind::Indirect,
            || kickoff.label,
            || {
                markers.copy(&list, 0, 0);
                markers.copy(&list, 4, 0);
            },
        );
        let outer = Profiled::new(&dev, 0x7302, |list| {
            markers.copy(list, 0, 4);
            markers.copy(list, 8, 4);
        });
        let gate = Gate(dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap());
        queue.Wait(&gate.0, 1).unwrap();
        submit(q, &[outer.list.as_raw()], || {
            list_x::kickoff_x(raw);
            outer.execute(&queue);
        });
        let reopened = Captured::of(raw, 0);
        assert_ne!(
            reopened.generation, kickoff.generation,
            "KickoffX must reopen the native list"
        );
        assert_ne!(
            reopened.index, kickoff.index,
            "reopen must not reuse an in-flight slot"
        );
        list.Close().expect("close the list reopened by KickoffX");
        let drain = enqueue_drain(&dev, &queue);
        assert_eq!(
            drain.GetCompletedValue(),
            0,
            "native Wait must gate KickoffX"
        );
        let captures = [kickoff, outer.recording];
        verify_in_flight(&queue, &captures);
        gate.0.Signal(1).unwrap();
        crate::gpu::fence::wait(&drain, 1, &dev, None, 5_000).unwrap();
        assert_eq!(
            markers.read(),
            [SECOND, FIRST, SECOND],
            "KickoffX must execute before the outer list"
        );
        verify_spans(&queue, &captures);
    });
}

#[test]
fn warp_adopted_native_transfers_and_barriers_are_classified() {
    with_watchdog(|| unsafe {
        let dev = warp();
        let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        assert!(
            !enabled(),
            "the first list must be adopted with timing disabled"
        );
        let first: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
        let first_raw = first.as_raw();
        let runtime_vtable = first_raw.cast::<usize>().read();
        real::readopt(first_raw, real::Kind::CommandList);
        let adopted_vtable = first_raw.cast::<usize>().read();
        first.Close().unwrap();
        drop(first);

        let _timing = Timing::enable();
        let queue = adopted_queue(&dev);
        let markers = Markers::new(&dev);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
        let raw = list.as_raw();
        assert_eq!(raw.cast::<usize>().read(), runtime_vtable);
        real::readopt(raw, real::Kind::CommandList);
        assert_eq!(raw.cast::<usize>().read(), adopted_vtable);
        opened(raw, 0);
        let token = list_token(raw);
        markers.copy(&list, 0, 0);
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }]);
        markers.copy(&list, 4, 4);
        markers.copy(&list, 8, 8);
        list.Close().unwrap();
        submit(queue.as_raw(), &[raw], || {
            type Execute = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let f: Execute = std::mem::transmute(real::orig_fn(queue.as_raw(), 10));
            f(queue.as_raw(), 1, &raw);
        });
        let drain = enqueue_drain(&dev, &queue);
        crate::gpu::fence::wait(&drain, 1, &dev, None, 5_000).unwrap();
        assert_eq!(markers.read(), [FIRST, SECOND, 0]);
        {
            let state = STATE.lock().unwrap();
            let q = &state.queues[&(queue.as_raw() as usize)];
            let mut batch = q.info.batch(state.frame);
            state.slots[state.lists[&(raw as usize)]].collect(&mut batch);
            assert_eq!(batch.dropped, 0);
            assert_eq!(
                batch.spans.iter().map(|span| span.kind).collect::<Vec<_>>(),
                [
                    Kind::List,
                    Kind::Transfer,
                    Kind::Barrier,
                    Kind::Transfer,
                    Kind::Transfer
                ]
            );
        }
        boundary();
        drop(list);
        list_released(token);
    });
}
