use super::{buffer, view, ReadBuffer, Topology, BACKEND};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{
    atomic::{fence, AtomicU64, Ordering},
    Arc, LazyLock, Mutex, OnceLock, Weak,
};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12GraphicsCommandList, D3D12_COMMAND_LIST_TYPE_BUNDLE, D3D12_HEAP_TYPE_UPLOAD, D3D12_RANGE,
};

/// Work that runs with locator 0: its translated packets read no topology and take the shader's
/// unified-memory path, as before the physical directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Uncovered {
    /// a bundle with translated packets or native leases executed without a site recipe
    Bundle,
    /// a bundle site recipe that failed or returned no independent private bundle
    BundleSite,
    /// a recording whose locator could not be made
    NoLocator,
    /// an execution of a recording that is still in flight with another topology
    Replay,
    /// a second publish of one execution with another topology
    Regrant,
}

static UNCOVERED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

pub(in super::super) fn uncovered(class: Uncovered) {
    let n = UNCOVERED[class as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: physical directory: {class:?} ran with locator 0 (unified-memory path) {n} times");
    }
}

#[cfg(test)]
pub(in super::super) fn uncovered_count(class: Uncovered) -> u64 {
    UNCOVERED[class as usize].load(Ordering::Relaxed)
}

struct Locator {
    cell: ReadBuffer,
    mapped: usize,
    flight: Mutex<Flight>,
}

#[derive(Default)]
struct Flight {
    visible: u32,
    users: Vec<Weak<OnceLock<Arc<Topology>>>>,
}

impl Locator {
    unsafe fn new() -> Result<Self, &'static str> {
        let b = BACKEND.get().ok_or("physical directory not initialized")?;
        let resource =
            buffer(&b.dev, 4, D3D12_HEAP_TYPE_UPLOAD).ok_or("locator allocation failed")?;
        let cell = view(
            &b.dev,
            resource,
            Default::default(),
            super::super::console_heap::Holder::Locators,
        )
        .ok_or("locator SRV unavailable")?;
        let mut ptr = std::ptr::null_mut();
        cell.resource
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))
            .map_err(|_| "locator map failed")?;
        let Some(ptr) = std::ptr::NonNull::new(ptr.cast::<u32>()) else {
            cell.resource
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            return Err("locator map returned null");
        };
        ptr.as_ptr().write_volatile(0);
        Ok(Self {
            cell,
            mapped: ptr.as_ptr() as usize,
            flight: Mutex::new(Flight::default()),
        })
    }
}

impl Drop for Locator {
    fn drop(&mut self) {
        unsafe {
            self.cell
                .resource
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 4 }));
        }
    }
}

/// Locators no recording or execution holds any more, reset to 0, for the next recordings.
static POOL: Mutex<Vec<Locator>> = Mutex::new(Vec::new());

/// A recording's locator: back to the pool, reset to 0, once the recording and every
/// execution that captured it are gone.
struct Lease(Option<Locator>);

impl Lease {
    fn take() -> Result<Self, &'static str> {
        let pooled = POOL.lock().unwrap().pop();
        Ok(Self(Some(match pooled {
            Some(l) => l,
            None => unsafe { Locator::new()? },
        })))
    }
}

impl std::ops::Deref for Lease {
    type Target = Locator;
    fn deref(&self) -> &Locator {
        self.0.as_ref().unwrap()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let Some(l) = self.0.take() else { return };
        *l.flight.lock().unwrap() = Flight::default();
        unsafe { (l.mapped as *mut u32).write_volatile(0) };
        POOL.lock().unwrap().push(l);
    }
}

#[derive(Default)]
struct Recording {
    own: OnceLock<Result<Arc<Lease>, &'static str>>,
    native: Mutex<Vec<Arc<dyn Send + Sync>>>,
    /// private bundle instances this recording executes, one per ExecuteBundle site
    bundles: Mutex<Vec<Arc<Recording>>>,
    recipe: OnceLock<Arc<BundleRecipe>>,
}

impl Recording {
    /// Its packets read a locator or native leases, so executing it natively would read none.
    fn managed(&self) -> bool {
        self.own.get().is_some() || !self.native.lock().unwrap().is_empty()
    }
}

static OPEN: LazyLock<Mutex<HashMap<usize, Arc<Recording>>>> = LazyLock::new(Default::default);

fn recording(pc: usize) -> Arc<Recording> {
    Arc::clone(OPEN.lock().unwrap().entry(pc).or_default())
}

pub(in super::super) fn ended(pc: usize) {
    let old = OPEN.lock().unwrap().remove(&pc);
    drop(old);
}

pub(in super::super) fn release(pc: usize, f: impl FnOnce() -> u32) -> u32 {
    let before = OPEN.lock().unwrap().get(&pc).cloned();
    let count = f();
    if count == 0 {
        if let Some(before) = before {
            let mut open = OPEN.lock().unwrap();
            if open.get(&pc).is_some_and(|r| Arc::ptr_eq(r, &before)) {
                open.remove(&pc);
            }
        }
    }
    count
}

/// io[`IO_PHYSICAL_TABLE`] for a packet recorded on the native list `pc`: its recording's
/// locator, or 0 (the unified-memory path) where there is none. A bundle's locator is published
/// only through a private instance from its site recipe (`bundle_site`); one executed natively
/// keeps 0.
pub(in super::super) fn packet_index(pc: usize) -> u32 {
    match recording(pc).own.get_or_init(|| Lease::take().map(Arc::new)) {
        Ok(l) => l.cell.view.0,
        Err(_) => {
            uncovered(Uncovered::NoLocator);
            0
        }
    }
}

/// Retain native packet dependencies for this recording and each captured execution of it.
pub(in super::super) fn retain_native(
    pc: usize,
    lease: Arc<dyn Send + Sync>,
) -> Result<(), &'static str> {
    if pc == 0 {
        return Err("native resource lease needs a recording");
    }
    recording(pc).native.lock().unwrap().push(lease);
    Ok(())
}

/// A complete, immutable recipe; each call emits one independent instance for the parent list
/// it is given.
pub(in super::super) type BundleRecipe =
    dyn Fn(usize) -> Result<PrivateBundle, String> + Send + Sync;

pub(in super::super) fn install_bundle_recipe(
    bundle_pc: usize,
    recipe: Arc<BundleRecipe>,
) -> Result<(), &'static str> {
    if bundle_pc == 0 {
        return Err("bundle recipe needs a recording");
    }
    recording(bundle_pc)
        .recipe
        .set(recipe)
        .map_err(|_| "bundle recipe already installed")
}

static PRIVATE: LazyLock<Mutex<HashSet<usize>>> = LazyLock::new(Default::default);

struct Site {
    list: ID3D12GraphicsCommandList,
}

impl Drop for Site {
    fn drop(&mut self) {
        PRIVATE
            .lock()
            .unwrap()
            .remove(&(self.list.as_raw() as usize));
    }
}

/// One closed native BUNDLE a site recipe made for one ExecuteBundle: its own allocator,
/// scratch and locator, kept by `lease` and the parent's executions until they retire.
pub(in super::super) struct PrivateBundle {
    site: Arc<Site>,
    generation: Arc<Recording>,
    lease: Arc<dyn Send + Sync>,
}

impl PrivateBundle {
    pub(in super::super) fn new(
        list: ID3D12GraphicsCommandList,
        lease: Arc<dyn Send + Sync>,
    ) -> Result<Self, &'static str> {
        if unsafe { list.GetType() } != D3D12_COMMAND_LIST_TYPE_BUNDLE {
            return Err("site recipe did not return a native BUNDLE");
        }
        let pc = list.as_raw() as usize;
        let generation = recording(pc);
        if !generation.bundles.lock().unwrap().is_empty() {
            return Err("a bundle cannot execute another bundle");
        }
        if !PRIVATE.lock().unwrap().insert(pc) {
            return Err("site recipe reused a live private bundle");
        }
        Ok(Self {
            site: Arc::new(Site { list }),
            generation,
            lease,
        })
    }
}

/// The bundle `parent` executes at this ExecuteBundle: a private instance from `bundle`'s
/// site recipe, owned by `parent`'s recording and published with it. None: execute `bundle`
/// itself, natively, at locator 0 (a bundle without translated packets, or counted: without
/// a recipe, or whose recipe failed). Never fails the call or marks `parent`.
pub(in super::super) fn bundle_site(parent: usize, bundle: usize) -> Option<ID3D12GraphicsCommandList> {
    let child = OPEN.lock().unwrap().get(&bundle).cloned()?;
    if !child.managed() {
        return None;
    }
    let Some(factory) = child.recipe.get().cloned() else {
        uncovered(Uncovered::Bundle);
        return None;
    };
    let site = || -> Result<PrivateBundle, String> {
        if parent == bundle {
            return Err("recursive ExecuteBundle".into());
        }
        let private = factory(parent)?;
        let pc = private.site.list.as_raw() as usize;
        if pc == bundle || pc == parent || Arc::ptr_eq(&child, &private.generation) {
            return Err("site recipe returned its source or parent recording".into());
        }
        if child.own.get().is_some() && !private.generation.own.get().is_some_and(Result::is_ok) {
            return Err("translated bundle recipe did not create a private locator".into());
        }
        Ok(private)
    };
    let private = match site() {
        Ok(p) => p,
        Err(why) => {
            uncovered(Uncovered::BundleSite);
            crate::xlog_info!("D3D12.X: physical directory: bundle {bundle:#x} runs natively at locator 0: {why}");
            return None;
        }
    };
    let pc = private.site.list.as_raw() as usize;
    let list = private.site.list.clone();
    private.generation.native.lock().unwrap().push(private.lease);
    let parent_gen = recording(parent);
    parent_gen.native.lock().unwrap().push(private.site);
    parent_gen.bundles.lock().unwrap().push(private.generation);
    ended(pc);
    Some(list)
}

// Recording conflicts and native ids are not consumed yet; kept for the next fix.
#[allow(dead_code)]
pub(in super::super) struct Captured {
    direct: Vec<Arc<Recording>>,
    members: Vec<Arc<Lease>>,
    native_ids: Vec<usize>,
    granted: Arc<OnceLock<Arc<Topology>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Recording conflicts and native ids are not consumed yet; kept for the next fix.
#[allow(dead_code)]
pub(in super::super) enum Conflict {
    DirectReplay,
    SharedScratch,
}

pub(in super::super) fn capture(lists: &[*mut c_void]) -> Captured {
    let direct = {
        let open = OPEN.lock().unwrap();
        lists
            .iter()
            .filter_map(|&l| open.get(&(l as usize)).cloned())
            .collect::<Vec<_>>()
    };
    let mut members = Vec::new();
    let mut native_ids = Vec::new();
    for r in &direct {
        let bundles = r.bundles.lock().unwrap();
        for g in std::iter::once(r).chain(bundles.iter()) {
            members.extend(g.own.get().and_then(|l| l.as_ref().ok()).cloned());
            native_ids.extend(
                g.native
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|x| Arc::as_ptr(x) as *const () as usize),
            );
        }
    }
    Captured::new(direct, members, native_ids)
}

impl Captured {
    fn new(
        mut direct: Vec<Arc<Recording>>,
        mut members: Vec<Arc<Lease>>,
        mut native_ids: Vec<usize>,
    ) -> Self {
        direct.sort_unstable_by_key(Arc::as_ptr);
        direct.dedup_by(|a, b| Arc::ptr_eq(a, b));
        members.sort_unstable_by_key(|l| l.cell.view.0);
        members.dedup_by_key(|l| l.cell.view.0);
        native_ids.sort_unstable();
        native_ids.dedup();
        Self {
            direct,
            members,
            native_ids,
            granted: Default::default(),
        }
    }

    /// A fresh grant for the exact native chunk; captures in pending software queues are not grants.
    // Recording conflicts and native ids are not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    pub(in super::super) fn for_execution(parts: &[&Captured]) -> Self {
        Self::new(
            parts.iter().flat_map(|p| p.direct.iter().cloned()).collect(),
            parts.iter().flat_map(|p| p.members.iter().cloned()).collect(),
            parts.iter().flat_map(|p| p.native_ids.iter().copied()).collect(),
        )
    }

    pub(in super::super) fn is_empty(&self) -> bool {
        self.direct.is_empty()
    }

    // Recording conflicts and native ids are not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    pub(in super::super) fn overlaps(&self, other: &Self) -> bool {
        self.direct
            .iter()
            .any(|a| other.direct.iter().any(|b| Arc::ptr_eq(a, b)))
    }

    // Recording conflicts and native ids are not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    pub(in super::super) fn pre_enqueue_conflict(&self, other: &Self) -> Option<Conflict> {
        if self.overlaps(other) {
            return Some(Conflict::DirectReplay);
        }
        self.native_ids
            .iter()
            .any(|id| other.native_ids.binary_search(id).is_ok())
            .then_some(Conflict::SharedScratch)
    }

    /// Make `topology` what this execution's locators select, before it is submitted. Never
    /// fails or waits: a locator still in flight with another topology goes to 0 (counted), so
    /// every execution sharing it takes the unified-memory path until they have all retired.
    /// True when every locator selects `topology`.
    pub(in super::super) fn publish_ready(&self, topology: &Arc<Topology>) -> bool {
        let mut flights: Vec<_> = self
            .members
            .iter()
            .map(|l| l.flight.lock().unwrap())
            .collect();
        let regrant = !Arc::ptr_eq(self.granted.get_or_init(|| Arc::clone(topology)), topology);
        if regrant {
            uncovered(Uncovered::Regrant);
        }
        let index = topology.index();
        let mut all = !regrant;
        for (f, l) in flights.iter_mut().zip(&self.members) {
            f.users.retain(|u| u.strong_count() != 0);
            let value = if regrant || (!f.users.is_empty() && f.visible != index) {
                0
            } else {
                index
            };
            if value != index && !regrant {
                uncovered(Uncovered::Replay);
                all = false;
            }
            if f.visible != value {
                unsafe { (l.mapped as *mut u32).write_volatile(value) };
                f.visible = value;
            }
            if !f.users.iter().any(|u| std::ptr::eq(u.as_ptr(), Arc::as_ptr(&self.granted))) {
                f.users.push(Arc::downgrade(&self.granted));
            }
        }
        fence(Ordering::SeqCst);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_keeps_the_native_recording_not_the_pointer() {
        let key = Box::new(0u8);
        let pc = &*key as *const u8 as usize;
        let members = [pc as *mut c_void];
        recording(pc);
        let original = capture(&members);
        assert!(original.overlaps(&capture(&members)));
        assert_eq!(release(pc, || 1), 1);
        assert!(original.overlaps(&capture(&members)));
        assert_eq!(
            release(pc, || {
                ended(pc);
                recording(pc);
                0
            }),
            0
        );
        let next = capture(&members);
        assert!(!original.overlaps(&next));
        assert!(next.overlaps(&capture(&members)));
        ended(pc);
        assert!(!next.overlaps(&capture(&members)));
    }

    #[test]
    fn native_lease_survives_reset_until_its_execution_retires() {
        let key = 0u8;
        let pc = &key as *const u8 as usize;
        let lease = Arc::new(9u8);
        let weak = Arc::downgrade(&lease);
        retain_native(pc, lease).unwrap();
        let pending = capture(&[pc as *mut c_void]);
        let running = Captured::for_execution(&[&pending, &pending]);
        assert_eq!(running.direct.len(), 1);
        ended(pc);
        drop(pending);
        assert!(weak.upgrade().is_some());
        drop(running);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn preflight_distinguishes_shared_scratch_from_replay() {
        let keys = [0u8; 2];
        let [a, b] = std::array::from_fn(|i| &keys[i] as *const u8 as usize);
        let scratch = Arc::new(1u8);
        retain_native(a, scratch.clone()).unwrap();
        retain_native(b, scratch).unwrap();
        let first = capture(&[a as *mut c_void]);
        assert_eq!(first.pre_enqueue_conflict(&first), Some(Conflict::DirectReplay));
        assert_eq!(
            first.pre_enqueue_conflict(&capture(&[b as *mut c_void])),
            Some(Conflict::SharedScratch)
        );
        for p in [a, b] {
            ended(p);
        }
    }
}

#[cfg(test)] pub(in super::super) fn test_locator_buffer(pc:usize)->windows::Win32::Graphics::Direct3D12::ID3D12Resource {recording(pc).own.get().unwrap().as_ref().unwrap().cell.resource.clone()}
