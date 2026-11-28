//! `XCL_RT_PAGE_LRU=1`: a bounded cache for the pages XCL places for the GPU's view of a pool
//! page (`copy_batch`'s `claim`).
//!
//! A copied GPU view is 64 KiB committed in the title's address space (`Held::claim`), and the
//! title's ray passes read thousands of them: the ray bins, ray counts and screen-probe lists are
//! pool pages the title maps at GPU addresses that cannot hold a view of the page itself. Nothing
//! gave that memory back – a view the title unmaps leaves its slot private, with no mapping for
//! `unmap_at` to decommit, so the page stays committed until the title happens to map a physical
//! page there. On NVIDIA, where fewer addresses hold a view of the page itself, that grows with
//! every page the title streams.
//!
//! With the switch on such a page is a cache entry: up to `BUDGET` of them are kept for the next
//! view at that address, and a page no view shows any more gives its memory back
//! (`Held::release`) once the fence of the submission that last used it has completed – never
//! while the GPU may still be reading it. Pages are released on the thread that maps new ones:
//! `copy_batch` already holds the physical pages there, so a claim and a release never interleave.
//! `XCL_RT_PAGE_LRU=0`, or unset, keeps today's behaviour (the memory held).
//!
//! The census, one line every 2000 submissions and one when a run of evictions ends:
//! `D3D12.X: ray page census: {pages} pages resident ({mib} MiB), {evictions} evictions, {waits}
//! fence waits, {released} pages released`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::real;
use crate::hle::kernelx::Held;

const PAGE: usize = 0x1_0000;
/// the cache's bound: 512 MiB of placed pages
const BUDGET: usize = 512 << 20;
const PAGES: usize = BUDGET / PAGE;
/// a census line every this many submissions
const CENSUS_EVERY: u64 = 2000;
/// no page's submission is older than this while `flights` holds fewer (pruned above it)
const FLIGHTS_MAX: usize = 4096;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_RT_PAGE_LRU").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: ray page census on (XCL_RT_PAGE_LRU=1): the pages XCL places for the GPU's view of a pool page are a {} MiB cache, given back only after the fence of the submission that last used them (XCL_RT_PAGE_LRU=0: held, as before)", BUDGET >> 20);
        }
        on
    })
}

/// A page XCL placed for a copied GPU view.
struct Place {
    /// the submission that last used it, as its fence (0: not in a look a submission was seen for)
    used: u64,
    /// the view at its address now: never given back
    live: bool,
}

struct QueueFence {
    fence: Arc<ID3D12Fence>,
    value: u64,
}

#[derive(Default)]
struct Cache {
    pages: HashMap<usize, Place>,
    /// submission serial → (queue, the fence value it signalled)
    flights: BTreeMap<u64, (usize, u64)>,
    queues: HashMap<usize, QueueFence>,
    submissions: u64,
    released: u64,
    evictions: u64,
    waits: u64,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
static SERIAL: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// the pages of this thread's last look, for the submission it is made for
    static PENDING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// A page was claimed as a copied GPU view (`Held::claim`).
pub(super) fn placed(at: usize) {
    if !on() {
        return;
    }
    let mut g = CACHE.lock().unwrap();
    g.get_or_insert_with(Cache::default).pages.insert(at, Place { used: 0, live: true });
}

/// No view shows the page at `at` any more: it may be given back once its submission is through.
pub(super) fn gone(at: usize) {
    if !on() {
        return;
    }
    if let Some(p) = CACHE.lock().unwrap().as_mut().and_then(|c| c.pages.get_mut(&at)) {
        p.live = false;
    }
}

/// The copied views this look covers: the pages a submission's look looks at. The submission they
/// are made for gives them their fence when it is issued (`submitted`).
pub(super) fn look(used: &[usize]) {
    if !on() {
        return;
    }
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        p.clear();
        p.extend_from_slice(used);
    });
}

/// A title submission was issued on `queue`: the pages of the look before it are ordered after
/// this submission's fence.
pub(super) unsafe fn submitted(queue: *mut c_void) {
    if !on() {
        return;
    }
    let used = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    let mut g = CACHE.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if !used.is_empty() {
        if let Some(flight) = signal(c, queue) {
            let serial = SERIAL.fetch_add(1, Ordering::Relaxed) + 1;
            c.flights.insert(serial, flight);
            for at in used {
                if let Some(p) = c.pages.get_mut(&at) {
                    p.used = serial;
                }
            }
            let Cache { flights, pages, .. } = c;
            if flights.len() > FLIGHTS_MAX {
                let oldest = pages.values().map(|p| p.used).filter(|&s| s != 0).min().unwrap_or(u64::MAX);
                let keep = flights.split_off(&oldest);
                *flights = keep;
            }
        }
    }
    c.submissions += 1;
    if c.submissions % CENSUS_EVERY == 0 {
        census(c);
    }
}

/// `queue`'s own fence signalled one more: (its key, the value).
unsafe fn signal(c: &mut Cache, queue: *mut c_void) -> Option<(usize, u64)> {
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    let key = queue as usize;
    if !c.queues.contains_key(&key) {
        let mut device = None;
        q.GetDevice(&mut device).ok()?;
        let fence = real::create_residency_fence(device.as_ref()?).ok()?;
        c.queues.insert(key, QueueFence { fence: Arc::new(fence), value: 0 });
    }
    let f = c.queues.get_mut(&key)?;
    f.value += 1;
    let value = f.value;
    q.Signal(&*f.fence, value).ok().map(|()| (key, value))
}

/// The submission of `serial` is through. 0 – the page was not in any look a submission was seen
/// for – and a serial no submission named: not through, so the page keeps its memory.
fn through(c: &Cache, serial: u64) -> bool {
    let Some(&(q, v)) = c.flights.get(&serial) else { return false };
    c.queues.get(&q).is_some_and(|f| unsafe { f.fence.GetCompletedValue() } >= v)
}

fn census(c: &Cache) {
    crate::xlog_info!(
        "D3D12.X: ray page census: {} pages resident ({} MiB of placed GPU view pages), {} evictions, {} fence waits, {} pages released",
        c.pages.len(),
        c.pages.len() * PAGE >> 20,
        c.evictions,
        c.waits,
        c.released
    );
}

/// The cache is over its bound: the pages no view shows, the longest unused first, give their
/// memory back – those whose submission is through (the others wait for it, counted). Called with
/// the physical pages held (`copy_batch`), so no claim can interleave with a release.
pub(super) unsafe fn evict(held: &mut Held) {
    if !on() {
        return;
    }
    let mut g = CACHE.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    let over = c.pages.len().saturating_sub(PAGES);
    if over == 0 {
        return;
    }
    let mut idle: Vec<(u64, usize)> = c.pages.iter().filter(|(_, p)| !p.live).map(|(&at, p)| (p.used, at)).collect();
    idle.sort_unstable();
    let (mut waits, mut released) = (0u64, 0u64);
    for (used, at) in idle {
        if released as usize >= over {
            break;
        }
        if !through(c, used) {
            waits += 1;
            continue;
        }
        if !held.release(at) {
            continue;
        }
        c.pages.remove(&at);
        released += 1;
    }
    c.evictions += released;
    c.waits += waits;
    c.released += released;
    if released != 0 {
        census(c);
    }
}
