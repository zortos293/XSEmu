//! The D3D12 pipeline library (`ID3D12PipelineLibrary`): every translated pipeline a warm-up or
//! the game builds is stored under its driver blob's name (`driver_cache`, which names the host
//! description and its DXIL), and each build site loads from the library before the blob cache.
//! (Behaviour after VIctory/CGBJ `translate --cache --pipelines` and its pipeline library: a
//! library made over a read-only mapping of its file, the file held while the library lives.)
//!
//! Files, "shards": `library-<writer>.<generation>.bin` in `<driver cache>\library\`. The folder
//! belongs to the driver (`driver_cache::dir`, which carries the adapter and the driver version),
//! never to a translation: a build that emits the same DXIL asks for the same entry and hits.
//! A warm-up child's writer is its warm-up peer slot (`w00` …), the game's
//! `game`: no two live processes write one writer's files. A writer's newest shard holds
//! everything it stored: a process makes its writer's library over the writer's newest shard,
//! mapped read-only, adds its new pipelines to it, and reads every other shard of the folder
//! read-only. Loads read only the shards on disk at the start, each through its own read-only
//! library; new pipelines go to the writer's library, which nothing loads from, so a save never
//! holds up a load. The writer's library is made over its newest shard while that is under
//! `SEGMENT` bytes, else made empty: a save rewrites at most a segment plus what is new, and a
//! writer has about one shard per segment. A save, at most once per `SAVE_INTERVAL` and at the
//! process's end, serializes the writer's library on its own thread into a mapped temporary file
//! (no lock held, stores held in a list meanwhile), renames it to the writer's next generation
//! (never over a file), reads it back, and removes the generation the library was made over,
//! which the new one holds in full; the writer then adds to the new shard, or to a new empty
//! library once the new shard is a full segment.
//! A shard the driver refuses as another driver's or adapter's, or as damaged, is renamed aside
//! (`.invalid`); one that cannot be read now is left as it is; neither, nor any other writer's
//! shard, is ever removed.
//!
//! Only the game keeps a library (a warm-up process makes none). It is on by default;
//! `XCL_PSO_LIBRARY=0` turns it off. Each entry is named by its key, a hash of the DXIL and the
//! exact description it was built from (every shader's bytes, the root signature's static
//! samplers and all state, `real::library_key`), and of nothing of the build that made it. A
//! shard is read only with its key list (`library-<writer>.<generation>.keys`), written in the
//! same format; a load asks the driver only for a key a shard lists. The driver then compares the
//! description and refuses an entry of another one, and it refuses a library of another driver or
//! adapter, so a pipeline is never taken under a description it was not built from.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, RwLockWriteGuard, TryLockError};
use std::time::{Duration, Instant};

use windows::core::{IUnknown, IUnknown_Vtbl, Interface, GUID, HRESULT, PCWSTR};
use windows::Win32::Foundation::{D3D12_ERROR_ADAPTER_NOT_FOUND, D3D12_ERROR_DRIVER_VERSION_MISMATCH, E_INVALIDARG, E_NOINTERFACE, E_POINTER, S_OK};
use windows::Win32::Graphics::Direct3D12::{ID3D12PipelineLibrary, ID3D12PipelineLibrary1, ID3D12PipelineState, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_PIPELINE_STATE_STREAM_DESC};

/// the least time between two saves of a library before the process's end
pub(super) const SAVE_INTERVAL: Duration = Duration::from_secs(300);
/// the fewest new pipelines a save before the process's end waits for
pub(super) const SAVE_MIN: usize = 256;
/// the size under which a writer's newest shard is added to (rewritten whole by a save); a
/// larger one stays as it is and new pipelines go to a new shard
pub(super) const SEGMENT: usize = 32 << 20;
/// what the save at the process's end waits for the locks at most
pub(super) const EXIT_WAIT: Duration = Duration::from_millis(200);

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static STORED: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static STALE: AtomicU64 = AtomicU64::new(0);
static LOAD_REFUSED: AtomicU64 = AtomicU64::new(0);
static SAVES: AtomicU64 = AtomicU64::new(0);
static SAVED_BYTES: AtomicU64 = AtomicU64::new(0);
static SAVE_MS_MAX: AtomicU64 = AtomicU64::new(0);
static WAIT_US_MAX: AtomicU64 = AtomicU64::new(0);

/// The run's library hits, misses, pipelines stored, shards refused and saves, and the longest a
/// load or store waited for the library's locks, for the caches line.
pub(super) fn summary() -> String {
    let n = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "pipeline library {} hits, {} misses, {} stored, {} shards refused, {} shards skipped without a key list this build reads, {} loads refused, {} saves ({} MB, longest {} ms), longest wait {:.1} ms",
        n(&HITS), n(&MISSES), n(&STORED), n(&REFUSED), n(&STALE), n(&LOAD_REFUSED), n(&SAVES), n(&SAVED_BYTES) >> 20, n(&SAVE_MS_MAX), n(&WAIT_US_MAX) as f64 / 1000.0
    )
}

/// Records how long a load or store waited for a library lock since `since`.
fn waited(since: Instant) {
    WAIT_US_MAX.fetch_max(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Whether a library with `unsaved` new pipelines, `since` after its last save (or its start),
/// is saved now, before the process's end.
pub(super) fn save_due(unsaved: usize, since: Duration) -> bool {
    unsaved >= SAVE_MIN && since >= SAVE_INTERVAL
}

/// Whether a writer's shard of `len` bytes is added to, rather than left as it is.
pub(super) fn grows(len: usize) -> bool {
    len < SEGMENT
}

/// Whether this process keeps a pipeline library: on unless `XCL_PSO_LIBRARY=0`.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PSO_LIBRARY").map_or(true, |v| v.trim() != "0"))
}

/// The key of a pipeline: a hash of the driver blob's name and of the description the pipeline is
/// built from, and of nothing of the build or the translation that made it. `tag` says which kind
/// of description `description` is; the blob's name carries the DXIL, its flags, and the root
/// signature's static samplers and node mask when they are not empty (`driver_cache`).
pub(super) fn key(blob_name: &str, description: &[u8], tag: u8) -> u64 {
    let mut bytes = Vec::with_capacity(1 + blob_name.len() + description.len());
    bytes.push(tag);
    bytes.extend_from_slice(blob_name.as_bytes());
    bytes.extend_from_slice(description);
    crate::gpu::shader::fnv64(&bytes)
}

/// The name of the entry of `key`: the key alone, so a build that emits the same DXIL under the
/// same description asks the driver for the same entry.
pub(super) fn entry_name(key: u64) -> Vec<u16> {
    format!("{key:016x}").encode_utf16().chain(Some(0)).collect()
}

/// The key list of shard `shard` (`library-<writer>.<generation>.keys`).
pub(super) fn keys_path(shard: &Path) -> PathBuf {
    shard.with_extension("keys")
}

const KEYS_MAGIC: &[u8; 8] = b"XCLKEYS2";
/// the key list's format, in every list; a list of another format is not read
const KEYS_FORMAT: u64 = 2;

/// Writes the key list of a shard at `path` (a new file, never over one): the format and the
/// keys, sorted.
pub(super) fn write_keys(path: &Path, keys: &[u64]) -> std::io::Result<()> {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut bytes = Vec::with_capacity(24 + sorted.len() * 8);
    bytes.extend_from_slice(KEYS_MAGIC);
    bytes.extend_from_slice(&KEYS_FORMAT.to_le_bytes());
    bytes.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for k in sorted {
        bytes.extend_from_slice(&k.to_le_bytes());
    }
    let tmp = path.with_extension(format!("keys.{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let written = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).and_then(|mut f| {
        use std::io::Write;
        f.write_all(&bytes)?;
        f.sync_all()
    });
    let result = written.and_then(|()| rename_new(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// The sorted keys of the key list at `path`; an error for a missing or damaged list or one of
/// another format.
pub(super) fn read_keys(path: &Path) -> Result<Vec<u64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("no key list ({e})"))?;
    let word = |at: usize| bytes.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    if bytes.get(..8) != Some(&KEYS_MAGIC[..]) {
        return Err("a damaged key list".into());
    }
    let (Some(format), Some(n)) = (word(8), word(16)) else { return Err("a damaged key list".into()) };
    if format != KEYS_FORMAT {
        return Err(format!("a key list of format {format}, this build reads {KEYS_FORMAT}"));
    }
    if bytes.len() as u64 != 24 + n.saturating_mul(8) {
        return Err("a damaged key list".into());
    }
    let keys: Vec<u64> = (0..n as usize).filter_map(|i| word(24 + i * 8)).collect();
    if keys.windows(2).any(|w| w[0] >= w[1]) {
        return Err("a damaged key list".into());
    }
    Ok(keys)
}

static WRITER: OnceLock<String> = OnceLock::new();

/// This process writes as warm-up peer slot `slot` (before its first build).
pub(super) fn set_warm_slot(slot: usize) {
    let _ = WRITER.set(format!("w{slot:02}"));
}

fn writer() -> &'static str {
    WRITER.get_or_init(|| "game".into())
}

pub(super) fn shard_name(writer: &str, generation: u32) -> String {
    format!("library-{writer}.{generation}.bin")
}

/// (writer, generation) of a shard's file name; None for any other file.
pub(super) fn parse(name: &str) -> Option<(&str, u32)> {
    let (writer, generation) = name.strip_prefix("library-")?.strip_suffix(".bin")?.rsplit_once('.')?;
    if writer.is_empty() || !writer.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    Some((writer, generation.parse().ok()?))
}

/// The shards among a folder's `names` for writer `me` (each writer's library is saved whole,
/// as its newest generation).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Found {
    /// `me`'s newest generation, which its library is made over
    pub(super) own: Option<u32>,
    /// every other shard, read-only: other writers' generations and `me`'s older ones
    pub(super) others: Vec<String>,
    /// how many of `others` are `me`'s older generations
    pub(super) older: usize,
    /// the generation `me` saves next
    pub(super) next: u32,
}

pub(super) fn shards<'a>(me: &str, names: impl IntoIterator<Item = &'a str>) -> Found {
    let mut found: Vec<(&str, u32)> = names.into_iter().filter_map(parse).collect();
    found.sort_unstable();
    let own = found.iter().filter(|(w, _)| *w == me).map(|(_, g)| *g).max();
    let older = found.iter().filter(|(w, g)| *w == me && Some(*g) != own).count();
    let others = found.into_iter().filter(|&(w, g)| !(w == me && Some(g) == own)).map(|(w, g)| shard_name(w, g)).collect();
    Found { own, others, older, next: own.map_or(0, |g| g + 1) }
}

/// Why a shard is not read.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    /// the driver refuses it (another driver or adapter, a damaged file): set aside for good
    Invalid(String),
    /// it cannot be read now (the file busy, no memory): left as it is
    Unreadable(String),
    /// no key list of the format this build reads: never read by this build, left as it is
    Unkeyed(String),
}

/// Whether the driver's answer to a library blob means the blob is no use to it.
pub(super) fn invalid(hr: HRESULT) -> bool {
    hr == D3D12_ERROR_DRIVER_VERSION_MISMATCH || hr == D3D12_ERROR_ADAPTER_NOT_FOUND || hr == E_INVALIDARG
}

/// A refused shard: an invalid one renamed aside (`.invalid`, never written over), an unreadable
/// one left as it is.
pub(super) fn set_aside(path: &Path, refusal: &Refusal) {
    if let Refusal::Invalid(_) = refusal {
        let _ = rename_new(path, &path.with_extension("bin.invalid"));
    }
}

fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

/// Renames `from` to `to` unless `to` exists: a shard is never written over.
pub(super) fn rename_new(from: &Path, to: &Path) -> std::io::Result<()> {
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};
    if unsafe { MoveFileExW(wide(from).as_ptr(), wide(to).as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// The temporary file process `pid` saves writer `me`'s library into: no other process's.
pub(super) fn temp_name(me: &str, pid: u32) -> String {
    format!("library-{me}.{pid}.tmp")
}

/// Saves writer `me`'s library: `write` fills this process's temporary file (made new, never
/// another's), renamed (`rename`) to generation `generation` or, when that fails (the name is
/// taken), to a later one. The generation written; this process's temporary file is gone either
/// way.
pub(super) fn save_shard(dir: &Path, me: &str, generation: u32, write: impl FnOnce(&Path) -> std::io::Result<()>, rename: impl Fn(&Path, &Path) -> std::io::Result<()>) -> std::io::Result<u32> {
    let tmp = dir.join(temp_name(me, std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let mut result = write(&tmp).map(|()| generation);
    if result.is_ok() {
        result = (generation..generation + 8)
            .map(|g| rename(&tmp, &dir.join(shard_name(me, g))).map(|()| g))
            .find(Result::is_ok)
            .unwrap_or_else(|| Err(std::io::Error::other(format!("{} could not be renamed to a shard", tmp.display()))));
    }
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// A file mapped read-only, with its handles, for as long as a library made over it lives.
pub(super) struct Mapped {
    view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    mapping: windows_sys::Win32::Foundation::HANDLE,
    file: windows_sys::Win32::Foundation::HANDLE,
}

unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    pub(super) fn open(path: &Path) -> std::io::Result<Mapped> {
        use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::Storage::FileSystem::{CreateFileW, GetFileSizeEx, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_DELETE, FILE_SHARE_READ, OPEN_EXISTING};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, MapViewOfFile, FILE_MAP_READ, PAGE_READONLY};
        unsafe {
            let file = CreateFileW(wide(path).as_ptr(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, std::ptr::null(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, std::ptr::null_mut());
            if file == INVALID_HANDLE_VALUE {
                return Err(std::io::Error::last_os_error());
            }
            let mut len = 0i64;
            if GetFileSizeEx(file, &mut len) == 0 || len <= 0 {
                let e = if len == 0 { std::io::Error::other("an empty file") } else { std::io::Error::last_os_error() };
                CloseHandle(file);
                return Err(e);
            }
            let mapping = CreateFileMappingW(file, std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null());
            if mapping.is_null() {
                let e = std::io::Error::last_os_error();
                CloseHandle(file);
                return Err(e);
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            if view.Value.is_null() {
                let e = std::io::Error::last_os_error();
                CloseHandle(mapping);
                CloseHandle(file);
                return Err(e);
            }
            Ok(Mapped { view, len: len as usize, mapping, file })
        }
    }

    pub(super) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.view.Value as *const u8, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        unsafe {
            windows_sys::Win32::System::Memory::UnmapViewOfFile(self.view);
            CloseHandle(self.mapping);
            CloseHandle(self.file);
        }
    }
}

/// A new file of `len` bytes at `path` (an error if one is there), held unshared while `fill`
/// fills it through a mapped view (no copy of it in the heap), flushed to disk.
pub(super) fn write_mapped(path: &Path, len: usize, fill: impl FnOnce(*mut c_void) -> bool) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FlushFileBuffers, CREATE_NEW, FILE_ATTRIBUTE_NORMAL};
    use windows_sys::Win32::System::Memory::{CreateFileMappingW, FlushViewOfFile, MapViewOfFile, UnmapViewOfFile, FILE_MAP_WRITE, PAGE_READWRITE};
    if len == 0 {
        return Err(std::io::Error::other("nothing to write"));
    }
    unsafe {
        let file = CreateFileW(wide(path).as_ptr(), GENERIC_READ | GENERIC_WRITE, 0, std::ptr::null(), CREATE_NEW, FILE_ATTRIBUTE_NORMAL, std::ptr::null_mut());
        if file == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        let size = len as u64;
        let mapping = CreateFileMappingW(file, std::ptr::null(), PAGE_READWRITE, (size >> 32) as u32, size as u32, std::ptr::null());
        if mapping.is_null() {
            let e = std::io::Error::last_os_error();
            CloseHandle(file);
            return Err(e);
        }
        let view = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, len);
        let result = if view.Value.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            let filled = fill(view.Value);
            let flushed = FlushViewOfFile(view.Value, len) != 0;
            UnmapViewOfFile(view);
            match (filled, flushed) {
                (false, _) => Err(std::io::Error::other("the library did not serialize")),
                (true, false) => Err(std::io::Error::last_os_error()),
                (true, true) => Ok(()),
            }
        };
        CloseHandle(mapping);
        let synced = result.is_ok() && FlushFileBuffers(file) != 0;
        let result = result.and_then(|()| if synced { Ok(()) } else { Err(std::io::Error::last_os_error()) });
        CloseHandle(file);
        result
    }
}

/// `take` until it gives a guard, until `deadline` at most (None: no limit).
pub(super) fn within<G>(deadline: Option<Instant>, mut take: impl FnMut() -> Option<G>) -> Option<G> {
    loop {
        if let Some(guard) = take() {
            return Some(guard);
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn try_mutex<T>(m: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match m.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

fn try_write<T>(l: &RwLock<T>) -> Option<RwLockWriteGuard<'_, T>> {
    match l.try_write() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a pipeline is built from, as its library entry must match it (D3D12 checks it).
#[derive(Clone, Copy)]
pub(super) enum Desc<'a> {
    Compute(&'a D3D12_COMPUTE_PIPELINE_STATE_DESC),
    Graphics(&'a D3D12_GRAPHICS_PIPELINE_STATE_DESC),
    /// a pipeline state stream
    Stream(&'a [u8]),
}

/// The private data GUID a library's mapped file is attached to the library under.
const KEEP_FILE: GUID = GUID::from_u128(0xcf7e258b79494951bb6cc0db2399cb32);

/// A COM object owning `T` until its last `Release`.
#[repr(C)]
struct Keeper<T> {
    vtable: *const IUnknown_Vtbl,
    refs: AtomicU32,
    value: T,
}

impl<T: Send + Sync + 'static> Keeper<T> {
    const VTABLE: IUnknown_Vtbl = IUnknown_Vtbl { QueryInterface: Self::query, AddRef: Self::add_ref, Release: Self::release };

    unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        if out.is_null() {
            return E_POINTER;
        }
        if iid.is_null() || *iid != IUnknown::IID {
            *out = std::ptr::null_mut();
            return E_NOINTERFACE;
        }
        Self::add_ref(this);
        *out = this;
        S_OK
    }

    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        (*this.cast::<Self>()).refs.fetch_add(1, Ordering::Relaxed) + 1
    }

    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        let left = (*this.cast::<Self>()).refs.fetch_sub(1, Ordering::Release) - 1;
        if left == 0 {
            std::sync::atomic::fence(Ordering::Acquire);
            drop(Box::from_raw(this.cast::<Self>()));
        }
        left
    }
}

/// `value` owned by a COM object, dropped at its last `Release`.
pub(super) fn keeper<T: Send + Sync + 'static>(value: T) -> IUnknown {
    let keeper = Box::new(Keeper { vtable: &Keeper::<T>::VTABLE, refs: AtomicU32::new(1), value });
    unsafe { IUnknown::from_raw(Box::into_raw(keeper).cast()) }
}

/// A `keeper`: its count is atomic and what it owns is `Send + Sync`.
struct Kept(#[allow(dead_code)] IUnknown);

unsafe impl Send for Kept {}
unsafe impl Sync for Kept {}

struct Shard {
    library: ID3D12PipelineLibrary,
    library1: Option<ID3D12PipelineLibrary1>,
    /// the keys its list names, sorted: the only ones a load asks it for
    keys: Vec<u64>,
    /// the mapped file when the library did not take it as private data: dropped after the library
    _file: Option<Kept>,
}

impl Shard {
    /// A shard of `library`, made over `file`: the file is attached to the library object, so it
    /// is unmapped at the object's last release, whoever holds it.
    fn new(library: ID3D12PipelineLibrary, file: Option<Mapped>) -> Shard {
        let file = file.map(keeper).filter(|file| unsafe { library.SetPrivateDataInterface(&KEEP_FILE, file) }.is_err()).map(Kept);
        Shard { library1: library.cast().ok(), library, keys: Vec::new(), _file: file }
    }

    /// The shard of the file at `path`, its library made over the mapped file, when its key list
    /// is one this build reads.
    fn open(path: &Path, create: &Create) -> Result<Shard, Refusal> {
        let keys = read_keys(&keys_path(path)).map_err(Refusal::Unkeyed)?;
        let file = Mapped::open(path).map_err(|e| Refusal::Unreadable(e.to_string()))?;
        match create(file.bytes()) {
            Ok(library) => Ok(Shard { keys, ..Shard::new(library, Some(file)) }),
            Err(hr) if invalid(hr) => Err(Refusal::Invalid(format!("{hr:?}"))),
            Err(hr) => Err(Refusal::Unreadable(format!("{hr:?}"))),
        }
    }

    unsafe fn load(&self, name: PCWSTR, desc: &Desc, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
        let raw = self.library.as_raw();
        match desc {
            Desc::Compute(d) => (self.library.vtable().LoadComputePipeline)(raw, name, *d, riid, ppv),
            Desc::Graphics(d) => (self.library.vtable().LoadGraphicsPipeline)(raw, name, *d, riid, ppv),
            Desc::Stream(s) => match &self.library1 {
                Some(l) => {
                    let d = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: s.len(), pPipelineStateSubobjectStream: s.as_ptr() as *mut c_void };
                    (l.vtable().LoadPipeline)(l.as_raw(), name, &d, riid, ppv)
                }
                None => E_INVALIDARG,
            },
        }
    }

    unsafe fn store(&self, name: &[u16], pso: *mut c_void) -> bool {
        (self.library.vtable().StorePipeline)(self.library.as_raw(), PCWSTR(name.as_ptr()), pso).is_ok()
    }
}

/// The device's `CreatePipelineLibrary` over a blob (empty: a new library).
pub(super) type Create = Box<dyn Fn(&[u8]) -> Result<ID3D12PipelineLibrary, HRESULT> + Send + Sync>;

/// This process's writer's library, which new pipelines are stored in and nothing loads from: the
/// writer's newest shard when that is under a segment, and what this process added since.
struct Own {
    shard: Arc<Shard>,
    /// the generation the library was made over: a save holds all of it, so the save removes it
    over: Option<u32>,
    /// the keys the library holds: its key list when saved
    keys: Mutex<Vec<u64>>,
}

impl Own {
    fn new(shard: Shard, over: Option<u32>) -> Own {
        Own { keys: Mutex::new(shard.keys.clone()), shard: Arc::new(shard), over }
    }

    /// Stores `pso` under key `key` and lists the key (false: one of that key is there).
    unsafe fn add(&self, key: u64, pso: *mut c_void) -> bool {
        let added = self.shard.store(&entry_name(key), pso);
        if added {
            lock(&self.keys).push(key);
        }
        added
    }
}

/// One device's libraries: every shard on disk at the start, read-only, and the writer's own.
pub(super) struct Libraries {
    /// what loads read: never stored in nor serialized, so no save holds up a load
    shards: Vec<Arc<Shard>>,
    /// held shared by stores; held alone by a save only to start holding stores in `pending` and
    /// to hand the writer's library on, never across a serialize or a file's IO
    own: RwLock<Own>,
    /// during a save (Some), the pipelines stored meanwhile, stored in its successor after it
    pending: Mutex<Option<Vec<(u64, ID3D12PipelineState)>>>,
    dir: PathBuf,
    create: Create,
    /// the generation the next save writes; held through a save
    next: Mutex<u32>,
    unsaved: AtomicUsize,
    /// pipelines stored that no successful save holds yet
    dirty: AtomicBool,
    saving: AtomicBool,
    born: Instant,
    /// when the last save started, in milliseconds after `born`
    last_save_ms: AtomicU64,
    /// loads of one name one at a time: the one call D3D12 leaves to the caller to synchronize
    names: [Mutex<()>; 64],
}

/// Removes writer `me`'s generation `over` from `dir` once a save holds all of it: the one
/// generation the writer's library was made over, never another writer's or a refused file. The
/// names removed.
pub(super) fn remove_superseded(dir: &Path, me: &str, over: Option<u32>) -> Vec<String> {
    over.map(|g| shard_name(me, g)).filter(|name| match std::fs::remove_file(dir.join(name)) {
        Ok(()) => true,
        Err(e) => {
            crate::xlog_warn!("D3D12.X: pipeline library: {name} not removed ({e}): it stays read-only");
            false
        }
    }).into_iter().collect()
}

impl Libraries {
    /// The pipeline of key `key` (`real::library_key`: its exact description) as `desc` builds it,
    /// from a shard on disk at the start whose key list names it; None (counted a miss) when none
    /// does, or the driver refuses it there.
    pub(super) unsafe fn load(&self, key: u64, desc: Desc, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
        let wide = entry_name(key);
        let asked = Instant::now();
        let _one = lock(&self.names[key as usize % 64]);
        waited(asked);
        for shard in self.shards.iter().filter(|shard| shard.keys.binary_search(&key).is_ok()) {
            for iid in [riid, &ID3D12PipelineState::IID as *const GUID] {
                *ppv = std::ptr::null_mut();
                let hr = shard.load(PCWSTR(wide.as_ptr()), &desc, iid, ppv);
                if hr.is_ok() && !(*ppv).is_null() {
                    HITS.fetch_add(1, Ordering::Relaxed);
                    return Some(hr);
                }
                if hr != E_NOINTERFACE {
                    LOAD_REFUSED.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
        }
        *ppv = std::ptr::null_mut();
        MISSES.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Adds pipeline `pso`, of key `key`, to the writer's library, or during a save to the list
    /// stored in its successor (false: not added, one of that key is there). A save starts on its
    /// own thread once `save_due`.
    pub(super) unsafe fn store(self: &Arc<Self>, key: u64, pso: *mut c_void) -> bool {
        let asked = Instant::now();
        let stored = {
            let own = self.own.read().unwrap_or_else(|e| e.into_inner());
            let held = {
                let mut pending = lock(&self.pending);
                waited(asked);
                pending.as_mut().map(|later| ID3D12PipelineState::from_raw_borrowed(&pso).map(|p| later.push((key, p.clone()))).is_some())
            };
            match held {
                Some(held) => held,
                None => own.add(key, pso),
            }
        };
        if !stored {
            return false;
        }
        STORED.fetch_add(1, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Release);
        let unsaved = self.unsaved.fetch_add(1, Ordering::Relaxed) + 1;
        let since = self.born.elapsed().saturating_sub(Duration::from_millis(self.last_save_ms.load(Ordering::Relaxed)));
        if save_due(unsaved, since) && !self.saving.swap(true, Ordering::AcqRel) {
            let me = self.clone();
            let spawned = std::thread::Builder::new().name("xcl-pso-library-save".into()).spawn(move || {
                me.save(None);
                me.saving.store(false, Ordering::Release);
            });
            if spawned.is_err() {
                self.saving.store(false, Ordering::Release);
            }
        }
        true
    }

    /// Saves the writer's library, everything it holds, as the writer's next generation. The
    /// library is serialized with no lock held, straight into the mapped file, with stores held in
    /// a list meanwhile; nothing loads from it. The new file is read back, the writer's library
    /// handed on (made over the new file while that is under a segment, else made empty), the
    /// list stored in it, and the generation it was made over removed. A save that fails keeps
    /// the library as it was. False: `deadline` passed before the locks were free.
    pub(super) fn save(&self, deadline: Option<Instant>) -> bool {
        let Some(mut next) = within(deadline, || try_mutex(&self.next)) else { return false };
        let (old, over, new, keys) = {
            let Some(own) = within(deadline, || try_write(&self.own)) else { return false };
            if !self.dirty.swap(false, Ordering::AcqRel) {
                return true;
            }
            *lock(&self.pending) = Some(Vec::new());
            let keys = lock(&own.keys).clone();
            (own.shard.clone(), own.over, self.unsaved.swap(0, Ordering::Relaxed), keys)
        };
        let now = self.born.elapsed();
        let since = now.saturating_sub(Duration::from_millis(self.last_save_ms.swap(now.as_millis() as u64, Ordering::Relaxed)));
        let started = Instant::now();
        let (raw, vtable) = (old.library.as_raw(), old.library.vtable());
        let len = unsafe { (vtable.GetSerializedSize)(raw) };
        let written = save_shard(&self.dir, writer(), *next, |tmp| write_mapped(tmp, len, |view| unsafe { (vtable.Serialize)(raw, view, len).is_ok() }), rename_new).and_then(|generation| {
            *next = generation + 1;
            write_keys(&keys_path(&self.dir.join(shard_name(writer(), generation))), &keys).map(|()| generation)
        });
        let saved = match written {
            Ok(generation) => {
                let path = self.dir.join(shard_name(writer(), generation));
                if grows(len) || over.is_some() {
                    Shard::open(&path, &self.create)
                        .map_err(|why| crate::xlog_warn!("D3D12.X: pipeline library: {} saved but not read back ({why:?}): it is kept, and the next save holds its pipelines too", path.display()))
                        .ok()
                        .map(|shard| (generation, Some(shard)))
                } else {
                    Some((generation, None))
                }
            }
            Err(e) => {
                self.dirty.store(true, Ordering::Release);
                self.unsaved.fetch_add(new, Ordering::Relaxed);
                crate::xlog_warn!("D3D12.X: pipeline library: {} not saved: {e}", self.dir.display());
                None
            }
        };
        let took = started.elapsed();
        let successor = saved.and_then(|(generation, shard)| {
            let own = match (grows(len), shard) {
                (true, Some(shard)) => Some(Own::new(shard, Some(generation))),
                (_, shard) => (self.create)(&[]).ok().map(|library| Own::new(Shard::new(library, None), None)).or_else(|| shard.map(|shard| Own::new(shard, Some(generation)))),
            };
            Some((generation, own?))
        });
        let (later, handed) = {
            let mut own = self.own.write().unwrap_or_else(|e| e.into_inner());
            let handed = successor.map(|(generation, successor)| {
                let full = successor.over.is_none();
                *own = successor;
                (generation, full)
            });
            (lock(&self.pending).take().unwrap_or_default(), handed)
        };
        {
            let own = self.own.read().unwrap_or_else(|e| e.into_inner());
            for (key, pso) in later {
                unsafe { own.add(key, pso.as_raw()) };
            }
        }
        let Some((generation, full)) = handed else { return true };
        drop(old);
        let removed = remove_superseded(&self.dir, writer(), over);
        let ms = took.as_millis() as u64;
        SAVES.fetch_add(1, Ordering::Relaxed);
        SAVED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        SAVE_MS_MAX.fetch_max(ms, Ordering::Relaxed);
        let from = over.map_or("none".to_string(), |g| g.to_string());
        let then = if full { "a full segment, new pipelines go to a new shard" } else { "new pipelines are added to it" };
        crate::xlog_info!(
            "D3D12.X: pipeline library: {} {}: generation {from} -> {generation} ({len} B serialized); {new} new, {} s after the last save, written and read back in {ms} ms off the title's threads; {then}; removed {}",
            self.dir.display(), writer(), since.as_secs(), if removed.is_empty() { "nothing".to_string() } else { removed.join(", ") }
        );
        true
    }
}

/// Every device's libraries, made at its first use (None: it makes none).
static DEVICES: Mutex<Vec<(usize, Option<Arc<Libraries>>)>> = Mutex::new(Vec::new());

/// Device `device`'s libraries in folder `dir`, opened at its first use outside the table's lock
/// (one that raced it is taken instead): `create` makes a library over a blob, `refused` hears of
/// each shard not read. None: the device makes no library.
pub(super) fn of(device: usize, dir: impl FnOnce() -> PathBuf, create: Create, refused: impl Fn()) -> Option<Arc<Libraries>> {
    if writer() != "game" {
        return None;
    }
    if let Some((_, libraries)) = lock(&DEVICES).iter().find(|(d, _)| *d == device) {
        return libraries.clone();
    }
    let libraries = open(dir(), create, &refused).map_err(|e| crate::xlog_warn!("D3D12.X: pipeline library: none for this device: {e}")).ok().map(Arc::new);
    let mut devices = lock(&DEVICES);
    if let Some((_, raced)) = devices.iter().find(|(d, _)| *d == device) {
        return raced.clone();
    }
    devices.push((device, libraries.clone()));
    libraries
}

fn open(dir: PathBuf, create: Create, refused: &dyn Fn()) -> Result<Libraries, String> {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let names: Vec<String> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect();
    let found = shards(writer(), names.iter().map(String::as_str));
    let read = |name: &str| {
        let path = dir.join(name);
        Shard::open(&path, &create).map_err(|why| {
            if let Refusal::Unkeyed(_) = why {
                STALE.fetch_add(1, Ordering::Relaxed);
                return;
            }
            REFUSED.fetch_add(1, Ordering::Relaxed);
            refused();
            set_aside(&path, &why);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!("D3D12.X: pipeline library: {} not read ({why:?}): its pipelines are built again", path.display());
            }
        }).ok()
    };
    let mut shards: Vec<Arc<Shard>> = found.others.iter().filter_map(|name| read(name)).map(Arc::new).collect();
    let newest = found.own.and_then(|g| Some((g, read(&shard_name(writer(), g))?)));
    let grown = newest.as_ref().is_some_and(|(g, _)| std::fs::metadata(dir.join(shard_name(writer(), *g))).is_ok_and(|m| grows(m.len() as usize)));
    let base = newest.as_ref().filter(|_| grown).and_then(|(g, _)| Some(Own::new(read(&shard_name(writer(), *g))?, Some(*g))));
    let read_others = shards.len();
    shards.extend(newest.map(|(_, shard)| Arc::new(shard)));
    let own = match base {
        Some(own) => own,
        None => Own::new(Shard::new(create(&[]).map_err(|hr| format!("an empty library: {hr:?}"))?, None), None),
    };
    if found.older > 0 {
        crate::xlog_info!("D3D12.X: pipeline library: {} older generations of {} kept read-only (left by an earlier save, never removed)", found.older, writer());
    }
    crate::xlog_info!("D3D12.X: pipeline library: {} as {} over generation {}: {} of {} other shards read; {} skipped without a key list this build reads (kept as they are)", dir.display(), writer(), own.over.map_or("none".to_string(), |g| g.to_string()), read_others, found.others.len(), STALE.load(Ordering::Relaxed));
    if let (Some(g), false) = (found.own, grown) {
        crate::xlog_info!("D3D12.X: pipeline library: {}'s generation {g} is a full segment or unread: kept as it is, new pipelines go to a new shard", writer());
    }
    Ok(Libraries { shards, own: RwLock::new(own), pending: Mutex::new(None), dir, create, next: Mutex::new(found.next), unsaved: AtomicUsize::new(0), dirty: AtomicBool::new(false), saving: AtomicBool::new(false), born: Instant::now(), last_save_ms: AtomicU64::new(0), names: std::array::from_fn(|_| Mutex::new(())) })
}

/// Saves every device's new pipelines: at a clean end without a limit (a warm-up child's, a
/// warm-up's), at the process's end within `deadline` (a library whose locks are busy is skipped;
/// the saves every `SAVE_INTERVAL` keep what it loses to that).
pub(super) fn save_all(deadline: Option<Instant>) {
    let Some(devices) = within(deadline, || try_mutex(&DEVICES)) else {
        crate::xlog_warn!("D3D12.X: pipeline library: not saved at the end: the libraries were busy");
        return;
    };
    let libraries: Vec<Arc<Libraries>> = devices.iter().filter_map(|(_, l)| l.clone()).collect();
    drop(devices);
    for l in libraries {
        if !l.save(deadline) {
            crate::xlog_warn!("D3D12.X: pipeline library: {} not saved at the end: it was busy", l.dir.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(what: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xcl-pso-library-{what}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn every_shard_is_read_and_a_writer_saves_after_its_last_generation() {
        assert_eq!(shard_name("w03", 7), "library-w03.7.bin");
        assert_eq!(parse("library-w03.7.bin"), Some(("w03", 7)));
        assert_eq!(parse("library-game.12.bin"), Some(("game", 12)));
        for other in ["library-w03.tmp", "library-w03.7.bin.invalid", "library-.1.bin", "library-w-3.1.bin", "library-w03.x.bin", "0123-1f-0.pso"] {
            assert_eq!(parse(other), None, "{other}");
        }
        let names = ["library-w00.3.bin", "library-w00.5.bin", "library-w01.2.bin", "library-game.9.bin", "library-w00.4.bin", "library-w00.6.bin.invalid", "library-w01.tmp"];
        let found = shards("w00", names);
        assert_eq!(found.own, Some(5), "the library is made over the newest generation");
        assert_eq!(found.others, ["library-game.9.bin", "library-w00.3.bin", "library-w00.4.bin", "library-w01.2.bin"], "older generations stay, read-only");
        assert_eq!((found.older, found.next), (2, 6));
        assert_eq!(shards("w02", names), Found { own: None, others: ["library-game.9.bin", "library-w00.3.bin", "library-w00.4.bin", "library-w00.5.bin", "library-w01.2.bin"].map(String::from).to_vec(), older: 0, next: 0 }, "a writer's first shard");
    }

    #[test]
    fn a_save_takes_the_next_free_generation_and_never_writes_over_a_shard() {
        let dir = scratch("save");
        let write = |what: &'static [u8]| move |tmp: &Path| std::fs::write(tmp, what);
        assert_eq!(save_shard(&dir, "w01", 3, write(b"one"), rename_new).unwrap(), 3);
        assert_eq!(save_shard(&dir, "w01", 3, write(b"two"), rename_new).unwrap(), 4, "generation 3 is taken");
        assert_eq!(std::fs::read(dir.join("library-w01.3.bin")).unwrap(), b"one", "not written over");
        assert_eq!(std::fs::read(dir.join("library-w01.4.bin")).unwrap(), b"two");
        assert!(save_shard(&dir, "w01", 5, write(b"three"), |_, _| Err(std::io::Error::other("no"))).is_err());
        assert!(save_shard(&dir, "w01", 5, |_| Err(std::io::Error::other("full disk")), rename_new).is_err());
        assert!(!dir.join(temp_name("w01", std::process::id())).exists(), "no temporary file is left");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_a_shard_the_driver_refuses_is_set_aside_and_none_is_deleted() {
        assert!(invalid(D3D12_ERROR_DRIVER_VERSION_MISMATCH) && invalid(D3D12_ERROR_ADAPTER_NOT_FOUND) && invalid(E_INVALIDARG));
        assert!(!invalid(windows::Win32::Foundation::E_OUTOFMEMORY) && !invalid(windows::Win32::Foundation::E_ACCESSDENIED));
        let dir = scratch("refused");
        let (busy, stale) = (dir.join("library-w00.1.bin"), dir.join("library-w00.2.bin"));
        std::fs::write(&busy, "one").unwrap();
        std::fs::write(&stale, "two").unwrap();
        set_aside(&busy, &Refusal::Unreadable("a sharing violation".into()));
        assert_eq!(std::fs::read_to_string(&busy).unwrap(), "one", "left exactly as it is");
        set_aside(&stale, &Refusal::Invalid("D3D12_ERROR_DRIVER_VERSION_MISMATCH".into()));
        assert!(!stale.exists());
        assert_eq!(std::fs::read_to_string(dir.join("library-w00.2.bin.invalid")).unwrap(), "two", "renamed aside, not deleted");
        std::fs::write(&stale, "two again").unwrap();
        set_aside(&stale, &Refusal::Invalid("E_INVALIDARG".into()));
        assert_eq!(std::fs::read_to_string(dir.join("library-w00.2.bin.invalid")).unwrap(), "two", "an earlier one set aside is not written over");
        assert!(stale.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_save_uses_its_own_temporary_file_and_never_another_process_s() {
        let dir = scratch("temp");
        let other = dir.join(temp_name("w01", std::process::id() ^ 1));
        std::fs::write(&other, b"another process").unwrap();
        assert_eq!(parse(&temp_name("w01", 77)), None, "a temporary file is no shard");
        let mine = dir.join(temp_name("w01", std::process::id()));
        std::fs::write(&mine, b"stale").unwrap();
        assert!(write_mapped(&mine, 4, |_| true).is_err(), "a file already there is never taken over");
        assert_eq!(save_shard(&dir, "w01", 0, |tmp| write_mapped(tmp, 4, |view| unsafe { std::ptr::write_bytes(view.cast::<u8>(), 7, 4); true }), rename_new).unwrap(), 0);
        assert_eq!(std::fs::read(dir.join("library-w01.0.bin")).unwrap(), [7; 4]);
        assert!(save_shard(&dir, "w01", 1, |_| Err(std::io::Error::other("full disk")), rename_new).is_err());
        assert_eq!(std::fs::read(&other).unwrap(), b"another process", "another process's file is never touched");
        assert!(!mine.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_mapped_file_lives_until_the_last_release_of_its_keeper() {
        static DROPS: AtomicU32 = AtomicU32::new(0);
        struct Counted;
        impl Drop for Counted {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let keep = keeper(Counted);
        let held = keep.clone();
        let same: IUnknown = keep.cast().unwrap();
        assert!(keep.cast::<ID3D12PipelineLibrary>().is_err(), "only IUnknown");
        drop(keep);
        drop(same);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0, "a reference is still held");
        drop(held);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1, "dropped once, at the last release");

        let dir = scratch("keeper");
        let path = dir.join("library-w00.2.bin");
        std::fs::write(&path, [9u8; 64]).unwrap();
        let mapped = Mapped::open(&path).unwrap();
        let (view, len) = (mapped.bytes().as_ptr(), mapped.bytes().len());
        let keep = keeper(mapped);
        let removed = !remove_superseded(&dir, "w00", Some(2)).is_empty();
        assert!(unsafe { std::slice::from_raw_parts(view, len) }.iter().all(|&b| b == 9), "still mapped while referenced");
        drop(keep);
        assert_eq!(path.exists(), !removed, "a removed file goes at the last release (delete-pending until then), a kept one stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_save_removes_only_the_generation_its_library_was_made_over() {
        let dir = scratch("superseded");
        for name in ["library-w00.1.bin", "library-w00.2.bin", "library-w00.2.bin.invalid", "library-w01.2.bin", "library-w00.3.bin"] {
            std::fs::write(dir.join(name), name).unwrap();
        }
        assert_eq!(remove_superseded(&dir, "w00", Some(2)), ["library-w00.2.bin"]);
        assert!(remove_superseded(&dir, "w00", None).is_empty(), "a library made empty holds no generation");
        assert!(remove_superseded(&dir, "w00", Some(2)).is_empty(), "already gone");
        let mut left: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        left.sort();
        assert_eq!(left, ["library-w00.1.bin", "library-w00.2.bin.invalid", "library-w00.3.bin", "library-w01.2.bin"], "older, refused, newer and other writers' files are kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_library_is_written_through_a_mapping_and_read_back_mapped() {
        let dir = scratch("map");
        let path = dir.join("library-w00.0.bin");
        write_mapped(&path, 4096, |view| {
            unsafe { std::ptr::write_bytes(view.cast::<u8>(), 0x5a, 4096) };
            true
        }).unwrap();
        let mapped = Mapped::open(&path).unwrap();
        assert_eq!(mapped.bytes().len(), 4096);
        assert!(mapped.bytes().iter().all(|&b| b == 0x5a));
        assert!(write_mapped(&dir.join("refused.bin"), 16, |_| false).is_err(), "a failed serialization is an error");
        assert!(write_mapped(&dir.join("none.bin"), 0, |_| true).is_err());
        std::fs::write(dir.join("empty.bin"), b"").unwrap();
        assert!(Mapped::open(&dir.join("empty.bin")).is_err(), "an empty file is no library");
        drop(mapped);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_end_waits_for_a_busy_lock_a_bounded_time() {
        let busy = Mutex::new(());
        let held = busy.lock().unwrap();
        let t0 = Instant::now();
        assert!(within(Some(t0 + EXIT_WAIT), || try_mutex(&busy)).is_none());
        assert!(t0.elapsed() >= EXIT_WAIT && t0.elapsed() < EXIT_WAIT * 3, "{:?}", t0.elapsed());
        drop(held);
        assert!(within(Some(Instant::now()), || try_mutex(&busy)).is_some(), "a free lock is taken at once");
        let rw = RwLock::new(());
        let reader = rw.read().unwrap();
        assert!(within(Some(Instant::now() + Duration::from_millis(20)), || try_write(&rw)).is_none());
        drop(reader);
        assert!(within(None, || try_write(&rw)).is_some());
    }

    mod fake {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;
        use windows::core::{implement, Ref, Result, GUID, PCWSTR, IUnknown};
        use windows::Win32::Foundation::{E_INVALIDARG, E_NOTIMPL};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;

        /// A stand-in for the driver's library: a serialize that takes `slow_ms`, a size of `len`.
        #[implement(ID3D12PipelineLibrary)]
        pub(super) struct Library {
            pub(super) len: Arc<AtomicUsize>,
            pub(super) slow_ms: Arc<AtomicU64>,
            /// every load is a hit (a driver that hands anything out)
            pub(super) hand_out: bool,
        }

        impl ID3D12Object_Impl for Library_Impl {
            fn GetPrivateData(&self, _: *const GUID, _: *mut u32, _: *mut std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetPrivateData(&self, _: *const GUID, _: u32, _: *const std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetPrivateDataInterface(&self, _: *const GUID, _: Ref<IUnknown>) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetName(&self, _: &PCWSTR) -> Result<()> { Ok(()) }
        }
        impl ID3D12DeviceChild_Impl for Library_Impl {
            fn GetDevice(&self, _: *const GUID, _: *mut *mut std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
        }
        impl ID3D12PipelineLibrary_Impl for Library_Impl {
            fn StorePipeline(&self, _: &PCWSTR, _: Ref<ID3D12PipelineState>) -> Result<()> { Ok(()) }
            fn LoadGraphicsPipeline(&self, _: &PCWSTR, _: *const D3D12_GRAPHICS_PIPELINE_STATE_DESC, _: *const GUID, _: *mut *mut std::ffi::c_void) -> Result<()> { Err(E_INVALIDARG.into()) }
            fn LoadComputePipeline(&self, _: &PCWSTR, _: *const D3D12_COMPUTE_PIPELINE_STATE_DESC, _: *const GUID, out: *mut *mut std::ffi::c_void) -> Result<()> {
                if !self.hand_out {
                    return Err(E_INVALIDARG.into());
                }
                let pso: ID3D12PipelineState = Pso.into();
                unsafe { *out = windows::core::Interface::into_raw(pso) };
                Ok(())
            }
            fn GetSerializedSize(&self) -> usize { self.len.load(Ordering::SeqCst) }
            fn Serialize(&self, data: *mut std::ffi::c_void, len: usize) -> Result<()> {
                std::thread::sleep(Duration::from_millis(self.slow_ms.load(Ordering::SeqCst)));
                unsafe { std::ptr::write_bytes(data.cast::<u8>(), 0x11, len) };
                Ok(())
            }
        }

        #[implement(ID3D12PipelineState)]
        pub(super) struct Pso;

        impl ID3D12Object_Impl for Pso_Impl {
            fn GetPrivateData(&self, _: *const GUID, _: *mut u32, _: *mut std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetPrivateData(&self, _: *const GUID, _: u32, _: *const std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetPrivateDataInterface(&self, _: *const GUID, _: Ref<IUnknown>) -> Result<()> { Err(E_NOTIMPL.into()) }
            fn SetName(&self, _: &PCWSTR) -> Result<()> { Ok(()) }
        }
        impl ID3D12DeviceChild_Impl for Pso_Impl {
            fn GetDevice(&self, _: *const GUID, _: *mut *mut std::ffi::c_void) -> Result<()> { Err(E_NOTIMPL.into()) }
        }
        impl ID3D12Pageable_Impl for Pso_Impl {}
        impl ID3D12PipelineState_Impl for Pso_Impl {
            fn GetCachedBlob(&self) -> Result<ID3DBlob> { Err(E_NOTIMPL.into()) }
        }
    }

    #[test]
    fn saves_wait_for_time_and_new_pipelines_and_a_full_segment_stays_as_it_is() {
        assert!(!save_due(SAVE_MIN, SAVE_INTERVAL - Duration::from_secs(1)), "not twice within the interval");
        assert!(!save_due(SAVE_MIN - 1, SAVE_INTERVAL * 10), "not for a few new pipelines");
        assert!(save_due(SAVE_MIN, SAVE_INTERVAL));
        assert!(grows(SEGMENT - 1) && !grows(SEGMENT));
    }

    #[test]
    fn a_save_holds_up_no_load_or_store_and_hands_the_writer_s_library_on() {
        let dir = scratch("nonblocking");
        let (len, slow_ms) = (Arc::new(AtomicUsize::new(4096)), Arc::new(std::sync::atomic::AtomicU64::new(1500)));
        let (l, s) = (len.clone(), slow_ms.clone());
        let create: Create = Box::new(move |_: &[u8]| Ok(fake::Library { len: l.clone(), slow_ms: s.clone(), hand_out: false }.into()));
        let libraries = Arc::new(open(dir.clone(), create, &|| {}).unwrap());
        let pso: ID3D12PipelineState = fake::Pso.into();
        for i in 0..10 {
            assert!(unsafe { libraries.store(i, pso.as_raw()) });
        }
        let saver = {
            let libraries = libraries.clone();
            std::thread::spawn(move || {
                let started = Instant::now();
                assert!(libraries.save(None));
                started.elapsed()
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        let (mut store_worst, mut load_worst, mut n) = (Duration::ZERO, Duration::ZERO, 0u64);
        let until = Instant::now() + Duration::from_millis(1000);
        while Instant::now() < until {
            let t = Instant::now();
            assert!(unsafe { libraries.store(1000 + n, pso.as_raw()) });
            store_worst = store_worst.max(t.elapsed());
            let (t, mut out) = (Instant::now(), std::ptr::null_mut());
            assert!(unsafe { libraries.load(u64::MAX, Desc::Stream(&[]), &ID3D12PipelineState::IID, &mut out) }.is_none());
            load_worst = load_worst.max(t.elapsed());
            n += 1;
        }
        let took = saver.join().unwrap();
        println!("save {took:?} (serialize 1500 ms); during it {n} stores and loads, the slowest store {store_worst:?}, the slowest load {load_worst:?}");
        assert!(took >= Duration::from_millis(1500), "the serialize ran while the stores and loads did");
        assert!(store_worst < Duration::from_millis(50) && load_worst < Duration::from_millis(50), "store {store_worst:?}, load {load_worst:?}");
        let name = |g| dir.join(shard_name(writer(), g));
        assert!(name(0).exists() && keys_path(&name(0)).exists(), "a shard and its key list");
        assert!(libraries.pending.lock().unwrap().is_none(), "the stores held during the save went to its successor");
        assert_eq!(libraries.own.read().unwrap().over, Some(0), "a shard under a segment is added to");

        slow_ms.store(0, Ordering::SeqCst);
        assert!(libraries.save(None), "the pipelines stored during the first save");
        assert!(name(1).exists() && !name(0).exists(), "the generation added to is removed once its successor is read back");
        assert!(libraries.save(None));
        assert!(!name(2).exists(), "nothing new: no save");
        assert!(unsafe { libraries.store(1 << 40, pso.as_raw()) });
        len.store(SEGMENT, Ordering::SeqCst);
        assert!(libraries.save(None));
        assert!(name(2).exists() && !name(1).exists());
        assert_eq!(libraries.own.read().unwrap().over, None, "after a full segment new pipelines go to a new library");
        len.store(4096, Ordering::SeqCst);
        assert!(unsafe { libraries.store(1 << 41, pso.as_raw()) });
        assert!(libraries.save(None));
        assert!(name(2).exists() && name(3).exists(), "a full segment is never rewritten nor removed");
        drop(libraries);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A driver's library that hands out a pipeline for every key, so a miss can only come from
    /// this library's own key list.
    fn hand_out() -> Create {
        Box::new(|_: &[u8]| Ok(fake::Library { len: Arc::new(AtomicUsize::new(64)), slow_ms: Arc::new(AtomicU64::new(0)), hand_out: true }.into()))
    }

    /// Whether the libraries answer a load of `key` with that driver, releasing what they hand out.
    fn hit(libraries: &Libraries, key: u64, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC) -> bool {
        let mut out = std::ptr::null_mut();
        let hit = unsafe { libraries.load(key, Desc::Compute(desc), &ID3D12PipelineState::IID, &mut out) }.is_some();
        if hit {
            drop(unsafe { ID3D12PipelineState::from_raw(out) });
        }
        hit
    }

    fn compute_desc(cs: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
        D3D12_COMPUTE_PIPELINE_STATE_DESC { CS: windows::Win32::Graphics::Direct3D12::D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() }, ..Default::default() }
    }

    #[test]
    fn a_library_one_run_saves_is_read_by_the_next_with_no_build_in_any_name() {
        let dir = scratch("next-run");
        let pso: ID3D12PipelineState = fake::Pso.into();
        let cs = [7u8; 16];
        let desc = compute_desc(&cs);
        let blob = "0123456789abcdef-10-0.pso";
        let entry = key(blob, &cs, b'c');
        {
            let run = Arc::new(open(dir.clone(), hand_out(), &|| {}).unwrap());
            assert!(unsafe { run.store(entry, pso.as_raw()) });
            assert!(run.save(None));
        }
        let shard = dir.join(shard_name(writer(), 0));
        let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok()).collect();
        names.sort();
        assert_eq!(names, [shard_name(writer(), 0), keys_path(&shard).file_name().unwrap().to_str().unwrap().to_string()], "a shard and its key list, and nothing of a build");
        let list = std::fs::read(keys_path(&shard)).unwrap();
        assert_eq!(list.get(..8), Some(&KEYS_MAGIC[..]));
        assert_eq!(list.get(8..16), Some(&KEYS_FORMAT.to_le_bytes()[..]), "the list says its format, not a translation");
        assert_eq!(read_keys(&keys_path(&shard)).unwrap(), [entry]);
        assert_eq!(String::from_utf16(&entry_name(entry)[..16]).unwrap(), format!("{entry:016x}"), "an entry is named by its key alone");
        let next = open(dir.clone(), hand_out(), &|| {}).unwrap();
        assert!(hit(&next, entry, &desc), "the next run reads the entry, with no build identity to change");
        assert!(!hit(&next, entry ^ 1, &desc), "a key no list names is a miss, though the driver hands out anything");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_changed_dxil_or_a_changed_description_is_another_key_and_does_not_hit() {
        let dir = scratch("changed");
        let pso: ID3D12PipelineState = fake::Pso.into();
        let cs = [7u8; 16];
        let desc = compute_desc(&cs);
        let entry = key("0123456789abcdef-10-0.pso", &cs, b'c');
        {
            let run = Arc::new(open(dir.clone(), hand_out(), &|| {}).unwrap());
            assert!(unsafe { run.store(entry, pso.as_raw()) });
            assert!(run.save(None));
        }
        let next = open(dir.clone(), hand_out(), &|| {}).unwrap();
        assert!(hit(&next, entry, &desc));
        let mut changed = cs;
        changed[3] ^= 1;
        assert_ne!(key_of("0123456789abcdef-10-0.pso", &changed), entry, "a changed DXIL is another key");
        assert!(!hit(&next, key_of("0123456789abcdef-10-0.pso", &changed), &desc), "and misses");
        assert_ne!(key_of("another-name-10-0.pso", &cs), entry, "a changed blob name (the root signature's samplers, the flags) is another key");
        assert!(!hit(&next, key_of("another-name-10-0.pso", &cs), &desc), "and misses");
        assert_ne!(key("0123456789abcdef-10-0.pso", &cs, b'g'), entry, "another kind of description is another key");
        assert_eq!(key_of("0123456789abcdef-10-0.pso", &cs), entry, "the same DXIL and description are the same key, whichever build asks");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn key_of(blob: &str, cs: &[u8]) -> u64 {
        key(blob, cs, b'c')
    }

    #[test]
    fn a_key_its_list_lacks_or_a_list_of_another_format_is_a_miss() {
        let dir = scratch("keys");
        let pso: ID3D12PipelineState = fake::Pso.into();
        let cs = [0u8; 16];
        let desc = compute_desc(&cs);
        {
            let writer = Arc::new(open(dir.clone(), hand_out(), &|| {}).unwrap());
            assert!(unsafe { writer.store(42, pso.as_raw()) });
            assert!(writer.save(None));
        }
        let shard = dir.join(shard_name(writer(), 0));
        assert_eq!(read_keys(&keys_path(&shard)).unwrap(), [42]);
        let mut earlier = std::fs::read(keys_path(&shard)).unwrap();
        earlier[..8].copy_from_slice(b"XCLKEYS1");
        std::fs::write(keys_path(&shard), &earlier).unwrap();
        assert!(read_keys(&keys_path(&shard)).is_err(), "a list of an earlier format is not read");
        std::fs::write(keys_path(&shard), &std::fs::read(keys_path(&shard)).unwrap()[..8]).unwrap();
        assert!(read_keys(&keys_path(&shard)).is_err(), "nor is a damaged one");
        std::fs::write(keys_path(&shard), &{
            let mut list = Vec::new();
            list.extend_from_slice(KEYS_MAGIC);
            list.extend_from_slice(&KEYS_FORMAT.to_le_bytes());
            list.extend_from_slice(&2u64.to_le_bytes());
            list.extend_from_slice(&42u64.to_le_bytes());
            list.extend_from_slice(&41u64.to_le_bytes());
            list
        }).unwrap();
        assert!(read_keys(&keys_path(&shard)).is_err(), "nor an unsorted one");
        std::fs::write(keys_path(&shard), &{
            let mut list = Vec::new();
            list.extend_from_slice(KEYS_MAGIC);
            list.extend_from_slice(&KEYS_FORMAT.to_le_bytes());
            list.extend_from_slice(&1u64.to_le_bytes());
            list.extend_from_slice(&42u64.to_le_bytes());
            list
        }).unwrap();
        let same = open(dir.clone(), hand_out(), &|| {}).unwrap();
        assert_eq!(read_keys(&keys_path(&shard)).unwrap(), [42]);
        assert!(hit(&same, 42, &desc));
        assert!(!hit(&same, 43, &desc), "a key no list names is a miss without asking the driver");
        std::fs::write(dir.join("library-w05.0.bin"), [1u8; 64]).unwrap();
        let unkeyed = STALE.load(Ordering::Relaxed);
        drop(open(dir.clone(), hand_out(), &|| {}).unwrap());
        assert_eq!(STALE.load(Ordering::Relaxed), unkeyed + 1, "a shard without a key list is not read");
        assert!(shard.exists() && dir.join("library-w05.0.bin").exists(), "and nothing is removed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
