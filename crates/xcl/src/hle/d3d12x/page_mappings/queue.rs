//! Queue order of the GPU physical directory. Each CopyPageMappingsBatchX version and each
//! submission whose lists carry a directory locator gets four calls on its own native queue:
//! Signal(ready, n), Wait(grant, n), the work, Signal(done, n). The `xcl-pte-order` worker grants
//! n only once ready n has completed on the GPU: a map publishes its immutable version first, a
//! reader binds its locators to the version current at that point. Work this cannot route runs
//! as it did before the directory existed, and every such path is counted (`super::fallback`).

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::{physical_directory::{self, recordings::{self, Captured}}, real};
use super::{fallback, Fallback, Update};
use crate::util::{HRESULT, E_FAIL, S_OK};

struct Lane {
    device: ID3D12Device,
    ready: ID3D12Fence,
    grant: ID3D12Fence,
    done: ID3D12Fence,
    next: AtomicU64,
    issue: Mutex<()>,
    missing_tail: Mutex<Option<u64>>,
    repaired: Condvar,
}
static LANES: LazyLock<Mutex<HashMap<usize, Arc<Lane>>>> = LazyLock::new(Default::default);

/// `XCL_PHYSICAL_DIRECTORY=0` keeps the directory (and so every queue lane) off.
pub(crate) fn available() -> bool {
    std::env::var("XCL_PHYSICAL_DIRECTORY").map_or(true, |v| v != "0")
}

/// A native queue was created: it gets its lane while the directory is active. A queue without
/// one keeps the submissions it had (counted).
pub(crate) unsafe fn register(raw: *mut c_void, device: &ID3D12Device) -> Result<(), HRESULT> {
    if physical_directory::current_topology().is_none() {
        return Ok(());
    }
    let fence = || real::create_residency_fence(device).ok();
    let lane = (|| {
        start_worker().ok()?;
        Some(Arc::new(Lane {
            device: device.clone(),
            ready: fence()?,
            grant: fence()?,
            done: fence()?,
            next: AtomicU64::new(0),
            issue: Mutex::new(()),
            missing_tail: Mutex::new(None),
            repaired: Condvar::new(),
        }))
    })();
    match lane {
        Some(lane) => {
            LANES.lock().unwrap().insert(raw as usize, lane);
            Ok(())
        }
        None => {
            fallback(Fallback::LaneMissing, "fences or worker could not be created");
            Err(E_FAIL)
        }
    }
}

pub(crate) fn released(raw: usize) {
    LANES.lock().unwrap().remove(&raw);
}

fn lane(raw: *mut c_void) -> Option<Arc<Lane>> {
    LANES.lock().unwrap().get(&(raw as usize)).cloned()
}

thread_local! {static ENTERED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };}
struct Entered;
impl Drop for Entered {
    fn drop(&mut self) {
        ENTERED.with_borrow_mut(|v| {
            v.pop();
        });
    }
}

fn entered<R>(key: usize, call: impl FnOnce() -> R) -> R {
    ENTERED.with_borrow_mut(|v| v.push(key));
    let _scope = Entered;
    call()
}

/// `call` with the lane's four calls issued together, after a missing done marker was repaired.
fn issued<R>(key: usize, lane: &Lane, call: impl FnOnce() -> R) -> R {
    if ENTERED.with_borrow(|v| v.contains(&key)) {
        return call();
    }
    let _issue = loop {
        let lock = lane.issue.lock().unwrap();
        let missing = lane.missing_tail.lock().unwrap();
        if missing.is_none() {
            break lock;
        }
        drop(lock);
        drop(lane.repaired.wait_while(missing, |m| m.is_some()).unwrap());
    };
    entered(key, call)
}

unsafe fn q_fence(raw: *mut c_void, slot: usize, f: &ID3D12Fence, value: u64) -> HRESULT {
    let method: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(real::original_or_native(raw, slot).0);
    method(raw, f.as_raw(), value)
}

enum Action {
    Map(Option<Update>),
    Read(Arc<Captured>, Option<Arc<physical_directory::Topology>>),
}

struct Flight {
    queue: ID3D12CommandQueue,
    lane: Arc<Lane>,
    point: u64,
    marked: bool,
    action: Action,
}

static WORK: LazyLock<(Mutex<VecDeque<Flight>>, Condvar)> = LazyLock::new(Default::default);

fn start_worker() -> Result<(), HRESULT> {
    static START: Mutex<bool> = Mutex::new(false);
    let mut started = START.lock().unwrap();
    if !*started {
        std::thread::Builder::new().name("xcl-pte-order".into()).spawn(worker).map_err(|_| E_FAIL)?;
        *started = true;
    }
    Ok(())
}

/// The directory version of a CopyPageMappingsBatchX on `raw`, after the queue's earlier work.
/// The guest call has already made its CPU views: a version that cannot be ordered is dropped
/// and counted, never a failure of the call.
pub(super) unsafe fn publish(raw: *mut c_void, update: Update) {
    let Some(l) = lane(raw) else { return fallback(Fallback::Unordered, "queue has no lane") };
    let hr = issued(raw as usize, &l, || enqueue(raw, &l, Action::Map(Some(update)), None));
    if hr < 0 {
        fallback(Fallback::PublishFailed, "native queue refused the ready/grant calls");
    }
}

/// Execution-time captures of lists handed to a queue later (a list may be reset before).
// The recordings holder has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Recordings {
    by_list: HashMap<usize, Arc<Captured>>,
    _submitted: Vec<ID3D12CommandList>,
}

impl Recordings {
    // The recordings holder has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn hold(mapped: &[*mut c_void], original: &[*mut c_void]) -> Arc<Self> {
        let by_list = mapped.iter().map(|&p| (p as usize, Arc::new(recordings::capture(&[p])))).collect();
        let _submitted = original.iter().filter_map(|p| ID3D12CommandList::from_raw_borrowed(p).cloned()).collect();
        Arc::new(Self { by_list, _submitted })
    }
}

/// Native Execute of `lists` on `raw`, ordered by the lane when they carry locators.
// The recordings holder has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) unsafe fn reader(raw: *mut c_void, lists: &[*mut c_void], held: Option<&Recordings>) {
    let members = match held {
        Some(h) => Captured::for_execution(&lists.iter().filter_map(|p| h.by_list.get(&(*p as usize))).map(Arc::as_ref).collect::<Vec<_>>()),
        None => recordings::capture(lists),
    };
    route(raw, members, || {
        let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) = std::mem::transmute(real::original_or_native(raw, 10).0);
        exec(raw, lists.len() as u32, lists.as_ptr());
    });
}

/// `submit` (a title submission of `lists` on `raw`, with all that goes with it): ordered by
/// the lane when the lists carry locators, else exactly as before.
pub(crate) unsafe fn submit(raw: *mut c_void, lists: &[*mut c_void], submit: impl FnOnce()) {
    route(raw, recordings::capture(lists), submit);
}

unsafe fn route(raw: *mut c_void, members: Captured, submit: impl FnOnce()) {
    if members.is_empty() {
        return submit();
    }
    let Some(l) = lane(raw) else {
        fallback(Fallback::UnroutedReader, "queue has no lane");
        return submit();
    };
    let mut submit = Some(submit);
    let hr = issued(raw as usize, &l, || enqueue(raw, &l, Action::Read(Arc::new(members), None), Some(&mut || submit.take().unwrap()())));
    if let Some(submit) = submit.take() {
        fallback(Fallback::UnroutedReader, if hr < 0 { "native queue refused the ready/grant calls" } else { "not enqueued" });
        submit();
    }
}

/// The four calls; within `issued`. A map's are fallible (its version is dropped); a reader's
/// are retried until they are made or the device is removed.
unsafe fn enqueue(raw: *mut c_void, l: &Arc<Lane>, action: Action, work: Option<&mut dyn FnMut()>) -> HRESULT {
    let Some(queue) = ID3D12CommandQueue::from_raw_borrowed(&raw) else { return E_FAIL };
    let fallible = matches!(action, Action::Map(_));
    let n = l.next.fetch_add(1, Ordering::Relaxed) + 1;
    for (slot, fence) in [(14, &l.ready), (15, &l.grant)] {
        loop {
            let hr = q_fence(raw, slot, fence, n);
            if hr >= 0 {
                break;
            }
            if fallible || l.device.GetDeviceRemovedReason().is_err() {
                return hr;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    if let Some(work) = work {
        work();
    }
    let marked = q_fence(raw, 14, &l.done, n) >= 0;
    if !marked {
        *l.missing_tail.lock().unwrap() = Some(n);
    }
    WORK.0.lock().unwrap().push_back(Flight { queue: queue.clone(), lane: l.clone(), point: n, marked, action });
    WORK.1.notify_one();
    S_OK
}

unsafe fn worker_step(waiting: &mut VecDeque<Flight>, reading: &mut Vec<Flight>) {
    for m in waiting.iter_mut().filter(|m| !m.marked) {
        if let Ok(_lock) = m.lane.issue.try_lock() {
            let raw = m.queue.as_raw();
            let armed = entered(raw as usize, || q_fence(raw, 14, &m.lane.done, m.point) >= 0);
            if armed || m.lane.done.GetCompletedValue() == u64::MAX {
                m.lane.missing_tail.lock().unwrap().take();
                m.lane.repaired.notify_all();
                m.marked = true;
            }
        }
    }
    reading.retain(|r| r.lane.done.GetCompletedValue() < r.point);
    let mut i = 0;
    while i < waiting.len() {
        let m = &mut waiting[i];
        if !m.marked {
            i += 1;
            continue;
        }
        let ready = m.lane.ready.GetCompletedValue();
        if ready == u64::MAX {
            waiting.remove(i);
            continue;
        }
        if ready < m.point {
            i += 1;
            continue;
        }
        match &mut m.action {
            Action::Map(update) => {
                if let Some(u) = update.take() {
                    u.version.publish_at_ready();
                }
            }
            Action::Read(members, selection) => {
                if selection.is_none() {
                    let Some(now) = physical_directory::current_topology() else {
                        i += 1;
                        continue;
                    };
                    members.publish_ready(&now);
                    *selection = Some(now);
                }
                std::sync::atomic::fence(Ordering::SeqCst);
            }
        }
        let sig: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT = std::mem::transmute(real::original_or_native(m.lane.grant.as_raw(), 10).0);
        if sig(m.lane.grant.as_raw(), m.point) < 0 {
            i += 1;
            continue;
        }
        let f = waiting.remove(i).unwrap();
        reading.push(f);
    }
}

fn worker() {
    let mut waiting = VecDeque::new();
    let mut reading = Vec::new();
    loop {
        let mut g = WORK.0.lock().unwrap();
        while g.is_empty() {
            if waiting.is_empty() && reading.is_empty() {
                g = WORK.1.wait(g).unwrap();
            } else {
                g = WORK.1.wait_timeout(g, Duration::from_millis(1)).unwrap().0;
                break;
            }
        }
        waiting.append(&mut g);
        drop(g);
        unsafe { worker_step(&mut waiting, &mut reading) };
    }
}
