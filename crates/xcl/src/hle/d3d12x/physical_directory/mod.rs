pub(super) mod recordings;
mod ownership;
#[cfg(test)]
mod tests;

use super::{console_heap, physical_import};
use crate::hle::kernelx::PhysicalPageRef;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
use windows::core::Interface;

pub(super) use ownership::Plane;

const LEAF_WORDS: usize = 65536;
const ROOT_WORDS: usize = 2 * 65536;

struct Descriptor(u32, console_heap::Holder);
impl Drop for Descriptor {
    fn drop(&mut self) {
        console_heap::release_held(self.0, 1, self.1);
    }
}

struct ReadBuffer {
    view: Descriptor,
    resource: ID3D12Resource,
    _placed: crate::gpu::rebar::Placed,
}

/// A raw view of physical pages: of one page's own buffer, or of the buffer over a whole
/// imported run, which every bound page of the run shares.
struct RawView {
    descriptor: Descriptor,
    _run: Option<physical_import::ImportedRun>,
}

pub(super) struct PageBinding {
    view: Arc<RawView>,
    /// the page's bytes into the view
    offset: usize,
    run: Arc<physical_import::ImportedPage>,
    page: PhysicalPageRef,
}

impl PageBinding {
    pub(super) fn token(&self) -> usize {
        self.page.id()
    }
    pub(super) fn resource(&self) -> &ID3D12Resource {
        self.run.resource()
    }
    pub(super) fn value(&self) -> u32 {
        ownership::mapped_pte(self.view.descriptor.0, self.offset)
            .expect("validated page descriptor")
    }
}

pub(super) type Mapping = ownership::Target<PageBinding>;
#[derive(Clone)]
pub(super) struct Change {
    pub guest: u64,
    pub plane: Plane,
    pub target: Mapping,
}
struct Leaf {
    view: ReadBuffer,
    pages: BTreeMap<u16, Mapping>,
}

pub(super) struct Topology {
    root: ReadBuffer,
    leaves: BTreeMap<(u16, Plane), Arc<Leaf>>,
}

pub(super) enum PageLookup {
    Absent,
    Unmapped,
    Mapped {
        binding: Arc<PageBinding>,
        offset: u32,
    },
}

impl Topology {
    pub(super) fn lookup(&self, guest: u64) -> Result<PageLookup, &'static str> {
        if guest >> 48 != 0 {
            return Err("non-48-bit GPU address");
        }
        let hi = (guest >> 32) as u16;
        let page = (guest >> 16) as u16;
        for plane in [Plane::Gpu, Plane::Default] {
            match self
                .leaves
                .get(&(hi, plane))
                .and_then(|leaf| leaf.pages.get(&page))
            {
                Some(Mapping::Page(binding)) => {
                    return Ok(PageLookup::Mapped {
                        binding: binding.clone(),
                        offset: guest as u16 as u32,
                    });
                }
                Some(Mapping::Unmapped) => return Ok(PageLookup::Unmapped),
                Some(Mapping::Inherit) | None => (),
            }
        }
        Ok(PageLookup::Absent)
    }
    pub(super) fn index(&self) -> u32 {
        self.root.view.0
    }
}

struct Backend {
    dev: ID3D12Device,
    head: Mutex<Arc<Topology>>,
    imports: Mutex<HashMap<usize, Weak<PageBinding>>>,
    /// each imported run's view, while a page of the run is bound
    runs: Mutex<HashMap<usize, Weak<RawView>>>,
}

static BACKEND: OnceLock<Backend> = OnceLock::new();
unsafe fn buffer(
    device: &ID3D12Device,
    bytes: usize,
    ty: D3D12_HEAP_TYPE,
) -> Option<ID3D12Resource> {
    let mut out = None;
    device
        .CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: ty,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(bytes),
            if ty == D3D12_HEAP_TYPE_UPLOAD {
                D3D12_RESOURCE_STATE_GENERIC_READ
            } else if ty == D3D12_HEAP_TYPE_READBACK {
                D3D12_RESOURCE_STATE_COPY_DEST
            } else {
                D3D12_RESOURCE_STATE_COMMON
            },
            None,
            &mut out,
        )
        .ok()?;
    out
}

fn buffer_desc(bytes: usize) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

unsafe fn view(
    device: &ID3D12Device,
    resource: ID3D12Resource,
    placed: crate::gpu::rebar::Placed,
    holder: console_heap::Holder,
) -> Option<ReadBuffer> {
    let words = u32::try_from(resource.GetDesc().Width / 4).ok()?;
    let (index, cpu) = console_heap::reserve_single(holder)?;
    let guard = Descriptor(index, holder);
    if index == 0 || index >= 1 << 20 {
        return None;
    }
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: words,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    device.CreateShaderResourceView(&resource, Some(&desc), cpu);
    Some(ReadBuffer {
        view: guard,
        resource,
        _placed: placed,
    })
}

struct PrivateUpload {
    buffer: ReadBuffer,
    address: usize,
    words: usize,
}
impl PrivateUpload {
    unsafe fn reserve(
        device: &ID3D12Device,
        words: usize,
        holder: console_heap::Holder,
    ) -> Option<Self> {
        // a root or a leaf: written once here, then only read, by every translated shader
        // that takes an address through the directory
        let (resource, placed) = crate::gpu::rebar::upload_buffer(
            device,
            &buffer_desc(words * 4),
            crate::gpu::rebar::Class::Directory,
        )
        .ok()?;
        let b = view(device, resource, placed, holder)?;
        let mut p = std::ptr::null_mut();
        b.resource
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut p))
            .ok()?;
        if p.is_null() {
            b.resource.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            return None;
        }
        Some(Self {
            buffer: b,
            address: p as usize,
            words,
        })
    }
    fn seal(self, entries: impl IntoIterator<Item = (usize, u32)>) -> ReadBuffer {
        let dest = self.address as *mut u32;
        unsafe {
            std::ptr::write_bytes(dest, 0, self.words);
            for (i, v) in entries {
                assert!(i < self.words);
                dest.add(i).write(v);
            }
            self.buffer.resource.Unmap(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: 4 * self.words,
                }),
            );
        }
        self.buffer
    }
}

pub(super) unsafe fn init(device: &ID3D12Device) -> bool {
    if let Some(b) = BACKEND.get() {
        return b.dev.as_raw() == device.as_raw();
    }
    if !console_heap::shared_heap(device) {
        return false;
    }
    let Some(root) = PrivateUpload::reserve(device, ROOT_WORDS, console_heap::Holder::Roots)
    else {
        return false;
    };
    let _ = BACKEND.set(Backend {
        dev: device.clone(),
        head: Mutex::new(Arc::new(Topology {
            root: root.seal([]),
            leaves: BTreeMap::new(),
        })),
        imports: Default::default(),
        runs: Default::default(),
    });
    BACKEND
        .get()
        .is_some_and(|b| b.dev.as_raw() == device.as_raw())
}

/// The physical directory is one per process and its tests are not: each one maps pages, publishes a
/// topology and then reads the index it landed on, so two of them running at once on `cargo test`'s
/// four threads read each other's indexes. Every test that touches it holds this for its whole run.
/// A test that panics poisons the lock; the next one takes it anyway, because what it guards is order,
/// not the directory's integrity.
#[cfg(test)]
pub(crate) fn test_serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(super) fn current_topology() -> Option<Arc<Topology>> {
    Some(Arc::clone(&BACKEND.get()?.head.lock().unwrap()))
}

pub(super) fn root_index() -> u32 {
    current_topology().map_or(0, |t| t.index())
}
pub(super) fn bind(page: &PhysicalPageRef) -> Result<Arc<PageBinding>, &'static str> {
    let b = BACKEND.get().ok_or("physical directory unavailable")?;
    let mut g = b.imports.lock().unwrap();
    if let Some(p) = g.get(&page.id()).and_then(Weak::upgrade) {
        return Ok(p);
    }
    let run = physical_import::resolve_page(&b.dev, page).ok_or("physical section not imported")?;
    let bound = |view: Arc<RawView>, offset: usize| {
        let bound = Arc::new(PageBinding {
            view,
            offset,
            run: Arc::clone(&run),
            page: page.clone(),
        });
        crate::gpu::physical_binding::BindingIndex::shared().remember(
            bound.view.descriptor.0,
            offset as u32,
            crate::gpu::physical_binding::PageKey::Physical(page.id() as u64),
            &bound,
        )?;
        Ok::<_, &'static str>(bound)
    };
    let shared = if run_views() {
        run_view(b, &run).and_then(|(view, offset)| bound(view, offset).ok())
    } else {
        None
    };
    let bound = match shared {
        Some(bound) => bound,
        None => {
            let descriptor = unsafe { raw_view(&b.dev, run.resource(), 65536) }?;
            let own = RawView {
                descriptor,
                _run: None,
            };
            bound(Arc::new(own), 0)?
        }
    };
    g.insert(page.id(), Arc::downgrade(&bound));
    if g.len().is_power_of_two() {
        g.retain(|_, w| w.strong_count() != 0);
    }
    Ok(bound)
}

/// The bound pages of an imported run (8 MiB at most) share one raw view of the run's buffer,
/// each named by the view's index and its page in the view (`ownership::mapped_pte`), where a
/// view of its own for every 64 KiB page took a descriptor of the shared heap each
/// (`XCL_DESCRIPTOR_SEGMENT_VIEWS=0`: a view per page, as before).
fn run_views() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DESCRIPTOR_SEGMENT_VIEWS").map_or(true, |v| v != "0"))
}

/// A raw view of the first `bytes` of `resource` in a single descriptor of the shared heap.
unsafe fn raw_view(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    bytes: usize,
) -> Result<Descriptor, &'static str> {
    let words = u32::try_from(bytes / 4).map_err(|_| "physical view too large")?;
    let (index, cpu) = console_heap::reserve_single(console_heap::Holder::Pages)
        .ok_or("physical descriptor space exhausted")?;
    let guard = Descriptor(index, console_heap::Holder::Pages);
    ownership::mapped_pte(index, 0).ok_or("descriptor index not encodable")?;
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: words,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    device.CreateUnorderedAccessView(resource, None, Some(&desc), cpu);
    Ok(guard)
}

/// The view of the run `page` is in, made if no bound page of the run has one, and the page's
/// bytes into it. None: the run has no buffer, the page is not encodable in it, or no
/// descriptor is left: the page then gets a view of its own.
fn run_view(b: &Backend, page: &physical_import::ImportedPage) -> Option<(Arc<RawView>, usize)> {
    let (run, offset) = physical_import::resolve_run(&b.dev, page)?;
    ownership::mapped_pte(1, offset)?;
    let mut runs = b.runs.lock().unwrap();
    if let Some(view) = runs.get(&run.id()).and_then(Weak::upgrade) {
        return Some((view, offset));
    }
    let descriptor = unsafe { raw_view(&b.dev, run.resource(), run.len()) }.ok()?;
    let id = run.id();
    let view = Arc::new(RawView {
        descriptor,
        _run: Some(run),
    });
    runs.insert(id, Arc::downgrade(&view));
    if runs.len().is_power_of_two() {
        runs.retain(|_, w| w.strong_count() != 0);
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::xlog_info!("D3D12.X: physical directory: the bound pages of an imported run share one raw view (128 pages at most), a page table entry names the view and the page in it (XCL_DESCRIPTOR_SEGMENT_VIEWS=0: a view for every 64 KiB page)"));
    Some((view, offset))
}

static PENDING: Mutex<BTreeMap<u32, u32>> = Mutex::new(BTreeMap::new());
static PENDING_PAGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PENDING_HITS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

struct Pending(Vec<u32>);
impl Pending {
    fn new(pages: Vec<u32>) -> Self {
        let mut p = PENDING.lock().unwrap();
        for &v in &pages {
            *p.entry(v).or_default() += 1;
        }
        PENDING_PAGES.store(p.len(), std::sync::atomic::Ordering::Release);
        Self(pages)
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        let mut p = PENDING.lock().unwrap();
        for v in &self.0 {
            if let std::collections::btree_map::Entry::Occupied(mut e) = p.entry(*v) {
                *e.get_mut() -= 1;
                if *e.get() == 0 {
                    e.remove();
                }
            }
        }
        PENDING_PAGES.store(p.len(), std::sync::atomic::Ordering::Release);
    }
}

/// A native texture consumer that still reads or writes the play81 CPU view at a GPU address
/// (compressed planes and other unconverted classes) reports each access here. It counts the
/// accesses that touch a page whose queue-ordered remap is accepted but not yet published:
/// those see the CPU-view timing, not the queue's, a known gap rather than coverage.
pub(super) fn note_native_texture_access(va: u64, len: u64, write: bool) {
    if len == 0 || PENDING_PAGES.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return;
    }
    let (Ok(lo), Ok(hi)) = (
        u32::try_from(va >> 16),
        u32::try_from(va.saturating_add(len - 1) >> 16),
    ) else {
        return;
    };
    if PENDING.lock().unwrap().range(lo..=hi).next().is_none() {
        return;
    }
    let n = PENDING_HITS[usize::from(write)].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_warn!(
            "physical directory: native texture {} touched a GPU page whose queue-ordered remap is still pending ({n} so far; known gap: unconverted native texture consumers keep play81 CPU-view timing)",
            if write { "write" } else { "read" }
        );
    }
}

/// Whether a page of [va, va + len) has a queue-ordered remap accepted but not yet published.
pub(super) fn remap_pending(va: u64, len: u64) -> bool {
    if len == 0 || PENDING_PAGES.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return false;
    }
    let (Ok(lo), Ok(hi)) = (u32::try_from(va >> 16), u32::try_from(va.saturating_add(len - 1) >> 16)) else {
        return true;
    };
    PENDING.lock().unwrap().range(lo..=hi).next().is_some()
}

/// (reads, writes) counted by `note_native_texture_access`.
pub(super) fn pending_native_texture_hits() -> (u64, u64) {
    (
        PENDING_HITS[0].load(std::sync::atomic::Ordering::Relaxed),
        PENDING_HITS[1].load(std::sync::atomic::Ordering::Relaxed),
    )
}

struct PreparedLeaf {
    upload: PrivateUpload,
    changes: BTreeMap<u16, Mapping>,
}
pub(super) struct PreparedVersion {
    backend: &'static Backend,
    root: PrivateUpload,
    changed: BTreeMap<(u16, Plane), PreparedLeaf>,
    pending: Pending,
}

pub(super) fn prepare_version(changes: Vec<Change>) -> Result<PreparedVersion, &'static str> {
    let b = BACKEND.get().ok_or("physical directory unavailable")?;
    let mut changed: BTreeMap<_, PreparedLeaf> = BTreeMap::new();
    let mut pages = Vec::with_capacity(changes.len());
    for change in changes {
        let key = ownership::Cell::at(change.plane, change.guest)
            .ok_or("unaligned or non-48-bit GPU mapping")?;
        let leaf = match changed.entry(((key.vpn >> 16) as u16, key.plane)) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                let upload = unsafe {
                    PrivateUpload::reserve(&b.dev, LEAF_WORDS, console_heap::Holder::Leaves)
                }
                .ok_or("cannot reserve immutable leaf")?;
                e.insert(PreparedLeaf {
                    upload,
                    changes: BTreeMap::new(),
                })
            }
        };
        pages.push(key.vpn);
        leaf.changes.insert(key.vpn as u16, change.target);
    }
    let root =
        unsafe { PrivateUpload::reserve(&b.dev, ROOT_WORDS, console_heap::Holder::Roots) }
            .ok_or("cannot reserve immutable root")?;
    Ok(PreparedVersion {
        backend: b,
        root,
        changed,
        pending: Pending::new(pages),
    })
}

impl PreparedVersion {
    pub(super) fn publish_at_ready(self) {
        let Self {
            backend,
            root,
            changed,
            pending,
        } = self;
        let mut head = backend.head.lock().unwrap();
        let mut leaves = head.leaves.clone();
        for (key, delta) in changed {
            let mut pages = leaves
                .get(&key)
                .map_or_else(BTreeMap::new, |l| l.pages.clone());
            for (page, target) in delta.changes {
                match target {
                    Mapping::Inherit => {
                        pages.remove(&page);
                    }
                    value => {
                        pages.insert(page, value);
                    }
                }
            }
            if pages.is_empty() {
                leaves.remove(&key);
                continue;
            }
            let view = delta.upload.seal(pages.iter().map(|(&i, m)| {
                (
                    usize::from(i),
                    match m {
                        Mapping::Inherit => 0,
                        Mapping::Unmapped => ownership::UNMAPPED,
                        Mapping::Page(p) => p.value(),
                    },
                )
            }));
            leaves.insert(key, Arc::new(Leaf { view, pages }));
        }
        let root = root.seal(leaves.iter().map(|(&(hi, plane), l)| {
            (
                2 * usize::from(hi) + usize::from(plane == Plane::Default),
                l.view.view.0,
            )
        }));
        *head = Arc::new(Topology { root, leaves });
        drop(head);
        drop(pending);
    }
}

impl Topology {
    /// (diagnostics, `memdiag`) Each GPU page this version maps to a page: its address and the
    /// page's binding.
    pub(super) fn diag_bound(&self) -> impl Iterator<Item = (u64, &PageBinding)> + '_ {
        self.leaves.iter().flat_map(|(&(hi, _), leaf)| {
            leaf.pages.iter().filter_map(move |(&page, m)| match m {
                Mapping::Page(b) => Some(((u64::from(hi) << 32) | (u64::from(page) << 16), &**b)),
                _ => None,
            })
        })
    }
}
