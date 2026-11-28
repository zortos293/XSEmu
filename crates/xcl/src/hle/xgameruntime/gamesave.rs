//! XGameSave – containers of named blobs (public GDK `XGameSave.h`), stored under
//! `<data>/gamesave/<configuration id>/<container>/`.
//! A container's committed snapshot holds `b/<blob>` and `display` (display name).
//! `b/~xclsave.current` selects one of two snapshots under the container's `b/` directory.
//! Containers without a pointer read the original files directly from the container root.
//! Offline: no cloud sync. Async variants use XAsync.

use std::ffi::{c_void, CString, OsStr};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockWriteGuard, TryLockError};
use std::time::{Duration, Instant};

use super::xasync::*;
use crate::util::{from_cstr_ptr, E_FAIL, E_INVALIDARG, HRESULT, S_OK};

/// Prefix for atomic-write temporary files placed alongside blobs/display files.
/// `enc()` always percent-encodes `~` as `%7E`, so no encoded blob name can ever start with this.
const TMP_PREFIX: &str = "~xcltmp.";
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);
const CURRENT: &str = "~xclsave.current";
const SNAPSHOTS: [&str; 2] = ["~xclsave.0", "~xclsave.1"];
static SAVE_FILES: RwLock<()> = RwLock::new(());
/// Async commits and deletes submitted and not yet done (`settle_for_exit`); a sync one holds
/// `SAVE_FILES` for writing while it runs.
static QUEUED: AtomicUsize = AtomicUsize::new(0);

/// One async commit or delete in `QUEUED`, from its submit until its job has run or is dropped.
struct Queued;

impl Queued {
    fn new() -> Self {
        QUEUED.fetch_add(1, Ordering::AcqRel);
        Queued
    }
}

impl Drop for Queued {
    fn drop(&mut self) {
        QUEUED.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `XCL_EXIT_SAVE_WAIT_MS` (default 5000, 0: off): how long XCL's own end of the process (its
/// window closed, the launcher's stop) waits for the title's save commits and deletes.
fn exit_wait() -> Duration {
    let ms = std::env::var("XCL_EXIT_SAVE_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(5000u64);
    Duration::from_millis(ms)
}

/// Before XCL ends the process on purpose: the save commits and deletes under way or queued
/// finish (up to `XCL_EXIT_SAVE_WAIT_MS`), then the save files stay locked so that none starts
/// while the process ends – a title thread that tries waits there until it is ended. A commit
/// cut short leaves the last committed snapshot in place, so what this keeps is the save the
/// title asked for last. Never blocks past the wait: the lock is only tried.
pub fn settle_for_exit() {
    let wait = exit_wait();
    if wait.is_zero() {
        return;
    }
    let start = Instant::now();
    let held = settle(&SAVE_FILES, &QUEUED, wait);
    let queued = QUEUED.load(Ordering::Acquire);
    if held.is_none() || queued > 0 {
        crate::xlog_warn!(
            "XGameSave: the process ends with a save {} after {} ms ({queued} async commits or deletes not run)",
            if held.is_some() { "queued" } else { "commit or delete under way" },
            start.elapsed().as_millis()
        );
    } else if start.elapsed() >= Duration::from_millis(5) {
        crate::xlog_info!("XGameSave: the process ends after the title's saves finished ({} ms)", start.elapsed().as_millis());
    }
    std::mem::forget(held);
}

/// `files` locked for writing once `queued` is 0 and no writer holds it, or at `wait` if
/// it is free then (None: a writer still held it).
fn settle<'a>(files: &'a RwLock<()>, queued: &AtomicUsize, wait: Duration) -> Option<RwLockWriteGuard<'a, ()>> {
    let start = Instant::now();
    loop {
        let timed_out = start.elapsed() >= wait;
        if queued.load(Ordering::Acquire) == 0 || timed_out {
            match files.try_write() {
                Ok(g) => return Some(g),
                Err(TryLockError::Poisoned(e)) => return Some(e.into_inner()),
                Err(TryLockError::WouldBlock) => {}
            }
        }
        if timed_out {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

pub const E_GS_INVALID_CONTAINER_NAME: HRESULT = 0x8083_0001u32 as i32;
pub const E_GS_OUT_OF_LOCAL_STORAGE: HRESULT = 0x8083_0003u32 as i32;
pub const E_GS_PROVIDED_BUFFER_TOO_SMALL: HRESULT = 0x8083_0007u32 as i32;
pub const E_GS_BLOB_NOT_FOUND: HRESULT = 0x8083_0008u32 as i32;
pub const E_GS_CONTAINER_NOT_FOUND: HRESULT = 0x8083_0010u32 as i32;
/// quota reported to the title (the GDK default)
pub const QUOTA: i64 = 256 << 20;

const MAGIC_PROVIDER: u32 = 0x5053_4758; // "XGSP"
const MAGIC_CONTAINER: u32 = 0x4353_4758;
const MAGIC_UPDATE: u32 = 0x5553_4758;

pub struct Provider {
    magic: u32,
    root: PathBuf,
}

pub struct Container {
    magic: u32,
    name: String,
    dir: PathBuf,
}

pub struct Update {
    magic: u32,
    dir: PathBuf,
    display: Option<String>,
    /// blob name → new contents (None: delete)
    ops: Vec<(String, Option<Vec<u8>>)>,
}

#[repr(C)]
pub struct ContainerInfo {
    pub name: *const u8,
    pub display_name: *const u8,
    pub blob_count: u32,
    pub total_size: u64,
    pub last_modified_time: i64,
    pub needs_sync: bool,
}

#[repr(C)]
pub struct BlobInfo {
    pub name: *const u8,
    pub size: u32,
}

#[repr(C)]
pub struct Blob {
    pub info: BlobInfo,
    pub data: *mut u8,
}

type ContainerCb = unsafe extern "system" fn(*const ContainerInfo, *mut c_void) -> bool;
type BlobCb = unsafe extern "system" fn(*const BlobInfo, *mut c_void) -> bool;

/// File-name form of a container / blob name: unusual characters as %XX.
fn enc(name: &str) -> String {
    let mut out: String = name
        .bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"._- ".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect();
    // Names Windows does not keep as given get one more character escaped: a device name (with
    // any extension) opens the device, "." and ".." are the folder itself and its parent, and a
    // trailing dot or space is dropped.
    let base = name.split('.').next().unwrap_or("").trim_end_matches(' ').to_ascii_uppercase();
    let device = matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (base.len() == 4 && (base.starts_with("COM") || base.starts_with("LPT")) && base.as_bytes()[3].is_ascii_digit());
    if !name.is_empty() && (device || name.bytes().all(|b| b == b'.')) {
        out.replace_range(..1, &format!("%{:02X}", name.as_bytes()[0]));
    }
    if out.ends_with('.') || out.ends_with(' ') {
        let last = out.pop().unwrap();
        out.push_str(&format!("%{:02X}", last as u8));
    }
    out
}

fn dec(file: &str) -> String {
    let b = file.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(hi), Some(lo)) = ((b[i + 1] as char).to_digit(16), (b[i + 2] as char).to_digit(16)) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn current_snapshot(dir: &Path) -> io::Result<PathBuf> {
    let storage = dir.join("b");
    let name = match fs::read(storage.join(CURRENT)) {
        Ok(marker) => SNAPSHOTS
            .into_iter()
            .find(|name| name.as_bytes() == marker)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "invalid save snapshot"))?,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(dir.to_path_buf()),
        Err(e) => return Err(e),
    };
    Ok(storage.join(name))
}

fn internal_file(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with(TMP_PREFIX) || name.starts_with("~xclsave.")
}

fn blobs_of(dir: &Path) -> Vec<(String, PathBuf, u64)> {
    let mut v: Vec<(String, PathBuf, u64)> = std::fs::read_dir(dir.join("b"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()) && !internal_file(&e.file_name()))
        .map(|e| (dec(&e.file_name().to_string_lossy()), e.path(), e.metadata().map(|m| m.len()).unwrap_or(0)))
        .collect();
    v.sort();
    v
}

fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.as_bytes()
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix.as_bytes()))
}

#[cfg(windows)]
fn verbatim_wide_path(path: &Path) -> io::Result<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Component, Prefix};

    let absolute = std::path::absolute(path)?;
    let (prefix, skip) = match absolute.components().next() {
        Some(Component::Prefix(p)) => match p.kind() {
            Prefix::Disk(_) => (r"\\?\", 0),
            Prefix::UNC(..) => (r"\\?\UNC\", 2),
            _ => ("", 0),
        },
        _ => return Err(io::Error::new(ErrorKind::InvalidInput, "missing absolute Windows path prefix")),
    };
    Ok(prefix.encode_utf16().chain(absolute.as_os_str().encode_wide().skip(skip)).chain(std::iter::once(0)).collect())
}

fn replace_file(tmp: &Path, dest: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH};
        let src = verbatim_wide_path(tmp)?;
        let dst = verbatim_wide_path(dest)?;
        // SAFETY: `src` and `dst` are valid NUL-terminated UTF-16 paths that outlive the call.
        let ok = unsafe { MoveFileExW(src.as_ptr(), dst.as_ptr(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) };
        if ok != 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
    #[cfg(not(windows))]
    {
        std::fs::rename(tmp, dest)
    }
}

/// Write `data` to a unique sibling temp file, flush to disk, and atomically replace `dest`.
fn write_atomic(dest: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let parent = dest
        .parent()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing parent directory"))?;
    let id = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!("{TMP_PREFIX}{}.{id}", std::process::id()));
    let res = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        drop(f);
        replace_file(&tmp, dest)?;
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

unsafe fn provider<'a>(p: *mut Provider) -> Option<&'a Provider> {
    p.as_ref().filter(|p| p.magic == MAGIC_PROVIDER)
}

unsafe fn container<'a>(c: *mut Container) -> Option<&'a Container> {
    c.as_ref().filter(|c| c.magic == MAGIC_CONTAINER)
}

fn open_provider(config: Option<String>) -> *mut Provider {
    let id: String = enc(&config.unwrap_or_else(|| "default".into()));
    let root = crate::state::data_dir().join("gamesave").join(if id.is_empty() { "default".into() } else { id });
    let _ = std::fs::create_dir_all(&root);
    Box::into_raw(Box::new(Provider { magic: MAGIC_PROVIDER, root }))
}

pub unsafe extern "system" fn XGameSaveInitializeProvider(_user: *mut c_void, config: *const u8, _sync_on_demand: bool, out: *mut *mut Provider) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = open_provider(from_cstr_ptr(config));
    S_OK
}

static INIT_ID: u8 = 0;

pub unsafe extern "system" fn XGameSaveInitializeProviderAsync(_user: *mut c_void, config: *const u8, _sync: bool, b: *mut XAsyncBlock) -> HRESULT {
    let config = from_cstr_ptr(config);
    begin_with(b, &INIT_ID as *const u8 as usize, move || Ok((open_provider(config) as usize).to_le_bytes().to_vec()))
}

pub unsafe extern "system" fn XGameSaveInitializeProviderResult(b: *mut XAsyncBlock, out: *mut *mut Provider) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let mut p = 0usize;
    let hr = XAsyncGetResult(b, &INIT_ID as *const u8 as *const c_void, 8, &mut p as *mut usize as *mut c_void, std::ptr::null_mut());
    if hr >= 0 {
        *out = p as *mut Provider;
    }
    hr
}

pub unsafe extern "system" fn XGameSaveCloseProvider(p: *mut Provider) {
    if provider(p).is_some() {
        (*p).magic = 0;
        drop(Box::from_raw(p));
    }
}

pub unsafe extern "system" fn XGameSaveGetRemainingQuota(p: *mut Provider, quota: *mut i64) -> HRESULT {
    let Some(pr) = provider(p) else { return E_INVALIDARG };
    if quota.is_null() {
        return E_INVALIDARG;
    }
    let used: u64 = {
        let _files = SAVE_FILES.read().unwrap_or_else(|e| e.into_inner());
        fs::read_dir(&pr.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| current_snapshot(&e.path()).ok())
            .flat_map(|snapshot| blobs_of(&snapshot))
            .map(|b| b.2)
            .sum()
    };
    *quota = (QUOTA - used as i64).max(0);
    static LAST: std::sync::Mutex<Option<i64>> = std::sync::Mutex::new(None);
    if let Ok(mut last) = LAST.lock() {
        if *last != Some(*quota) {
            *last = Some(*quota);
            crate::xlog_info!("XGameSaveGetRemainingQuota: {} KiB left of {} MiB", *quota >> 10, QUOTA >> 20);
        }
    }
    S_OK
}

static QUOTA_ID: u8 = 0;

pub unsafe extern "system" fn XGameSaveGetRemainingQuotaAsync(p: *mut Provider, b: *mut XAsyncBlock) -> HRESULT {
    let mut q = 0i64;
    let hr = XGameSaveGetRemainingQuota(p, &mut q);
    begin_with(b, &QUOTA_ID as *const u8 as usize, move || if hr < 0 { Err(hr) } else { Ok(q.to_le_bytes().to_vec()) })
}

pub unsafe extern "system" fn XGameSaveGetRemainingQuotaResult(b: *mut XAsyncBlock, quota: *mut i64) -> HRESULT {
    XAsyncGetResult(b, &QUOTA_ID as *const u8 as *const c_void, 8, quota as *mut c_void, std::ptr::null_mut())
}

fn valid_container_name_length(name: &str) -> bool {
    !name.is_empty() && name.len() <= 256
}

unsafe fn delete_path(p: *mut Provider, name: *const u8) -> Result<PathBuf, HRESULT> {
    let pr = provider(p).ok_or(E_INVALIDARG)?;
    let name = from_cstr_ptr(name).ok_or(E_INVALIDARG)?;
    if !valid_container_name_length(&name) {
        return Err(E_GS_INVALID_CONTAINER_NAME);
    }
    std::path::absolute(pr.root.join(enc(&name))).map_err(|_| E_FAIL)
}

fn delete_container(dir: &Path) {
    let _files = SAVE_FILES.write().unwrap_or_else(|e| e.into_inner());
    let _ = std::fs::remove_dir_all(dir);
}

pub unsafe extern "system" fn XGameSaveDeleteContainer(p: *mut Provider, name: *const u8) -> HRESULT {
    match delete_path(p, name) {
        Ok(dir) => {
            delete_container(&dir);
            S_OK
        }
        Err(hr) => hr,
    }
}

static DELETE_ID: u8 = 0;

pub unsafe extern "system" fn XGameSaveDeleteContainerAsync(p: *mut Provider, name: *const u8, b: *mut XAsyncBlock) -> HRESULT {
    if b.is_null() {
        return E_INVALIDARG;
    }
    let dir = delete_path(p, name);
    let queued = Queued::new();
    begin_with(b, &DELETE_ID as *const u8 as usize, move || {
        let _queued = queued;
        delete_container(&dir?);
        Ok(Vec::new())
    })
}

pub unsafe extern "system" fn XGameSaveDeleteContainerResult(b: *mut XAsyncBlock) -> HRESULT {
    XAsyncGetStatus(b, false)
}

pub unsafe extern "system" fn XGameSaveCreateContainer(p: *mut Provider, name: *const u8, out: *mut *mut Container) -> HRESULT {
    let (Some(pr), Some(n)) = (provider(p), from_cstr_ptr(name)) else { return E_INVALIDARG };
    if out.is_null() {
        return E_INVALIDARG;
    }
    if !valid_container_name_length(&n) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    *out = Box::into_raw(Box::new(Container { magic: MAGIC_CONTAINER, dir: pr.root.join(enc(&n)), name: n }));
    S_OK
}

pub unsafe extern "system" fn XGameSaveCloseContainer(c: *mut Container) {
    if container(c).is_some() {
        (*c).magic = 0;
        drop(Box::from_raw(c));
    }
}

/// Call `cb` for every stored container whose name starts with `prefix`.
unsafe fn each_container(pr: &Provider, prefix: &str, exact: bool, ctx: *mut c_void, cb: ContainerCb) -> usize {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&pr.root).into_iter().flatten().flatten().filter(|e| e.path().is_dir()).map(|e| e.path()).collect();
    dirs.sort();
    let mut n = 0;
    for d in dirs {
        let name = dec(&d.file_name().unwrap_or_default().to_string_lossy());
        if exact {
            if !name.eq_ignore_ascii_case(prefix) {
                continue;
            }
        } else if !starts_with_ignore_ascii_case(&name, prefix) {
            continue;
        }
        let (blobs, display, modified) = {
            let _files = SAVE_FILES.read().unwrap_or_else(|e| e.into_inner());
            let Ok(snapshot) = current_snapshot(&d) else { continue };
            let blobs = blobs_of(&snapshot);
            if blobs.is_empty() && !snapshot.join("display").exists() {
                continue; // created but never written
            }
            let display = fs::read_to_string(snapshot.join("display")).unwrap_or_else(|_| name.clone());
            let modified = fs::metadata(snapshot)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|t| t.as_secs() as i64)
                .unwrap_or(0);
            (blobs, display, modified)
        };
        let (cn, cd) = (CString::new(name).unwrap_or_default(), CString::new(display).unwrap_or_default());
        let info = ContainerInfo {
            name: cn.as_ptr() as *const u8,
            display_name: cd.as_ptr() as *const u8,
            blob_count: blobs.len() as u32,
            total_size: blobs.iter().map(|b| b.2).sum(),
            last_modified_time: modified,
            needs_sync: false,
        };
        n += 1;
        if !cb(&info, ctx) {
            break;
        }
    }
    n
}

pub unsafe extern "system" fn XGameSaveGetContainerInfo(p: *mut Provider, name: *const u8, ctx: *mut c_void, cb: Option<ContainerCb>) -> HRESULT {
    let (Some(pr), Some(n), Some(cb)) = (provider(p), from_cstr_ptr(name), cb) else { return E_INVALIDARG };
    if each_container(pr, &n, true, ctx, cb) == 0 {
        return E_GS_CONTAINER_NOT_FOUND;
    }
    S_OK
}

pub unsafe extern "system" fn XGameSaveEnumerateContainerInfo(p: *mut Provider, ctx: *mut c_void, cb: Option<ContainerCb>) -> HRESULT {
    let (Some(pr), Some(cb)) = (provider(p), cb) else { return E_INVALIDARG };
    each_container(pr, "", false, ctx, cb);
    S_OK
}

pub unsafe extern "system" fn XGameSaveEnumerateContainerInfoByName(p: *mut Provider, prefix: *const u8, ctx: *mut c_void, cb: Option<ContainerCb>) -> HRESULT {
    let (Some(pr), Some(cb)) = (provider(p), cb) else { return E_INVALIDARG };
    each_container(pr, &from_cstr_ptr(prefix).unwrap_or_default(), false, ctx, cb);
    S_OK
}

unsafe fn each_blob(c: &Container, prefix: &str, ctx: *mut c_void, cb: BlobCb) {
    let blobs = {
        let _files = SAVE_FILES.read().unwrap_or_else(|e| e.into_inner());
        let Ok(snapshot) = current_snapshot(&c.dir) else { return };
        blobs_of(&snapshot)
    };
    for (name, _, size) in blobs {
        if !starts_with_ignore_ascii_case(&name, prefix) {
            continue;
        }
        let cn = CString::new(name).unwrap_or_default();
        if !cb(&BlobInfo { name: cn.as_ptr() as *const u8, size: size as u32 }, ctx) {
            break;
        }
    }
}

pub unsafe extern "system" fn XGameSaveEnumerateBlobInfo(c: *mut Container, ctx: *mut c_void, cb: Option<BlobCb>) -> HRESULT {
    let (Some(c), Some(cb)) = (container(c), cb) else { return E_INVALIDARG };
    each_blob(c, "", ctx, cb);
    S_OK
}

pub unsafe extern "system" fn XGameSaveEnumerateBlobInfoByName(c: *mut Container, prefix: *const u8, ctx: *mut c_void, cb: Option<BlobCb>) -> HRESULT {
    let (Some(c), Some(cb)) = (container(c), cb) else { return E_INVALIDARG };
    each_blob(c, &from_cstr_ptr(prefix).unwrap_or_default(), ctx, cb);
    S_OK
}

/// Blobs `names` (all when empty) laid out as the API returns them: `Blob` array, then the
/// names and data; pointers stored as offsets from the start (fixed up by the caller).
fn read_layout(dir: &Path, names: &[String]) -> Result<(Vec<u8>, u32), HRESULT> {
    let _files = SAVE_FILES.read().unwrap_or_else(|e| e.into_inner());
    let all = blobs_of(&current_snapshot(dir).map_err(|_| E_FAIL)?);
    let picked: Vec<(String, PathBuf)> = if names.is_empty() {
        all.into_iter().map(|b| (b.0, b.1)).collect()
    } else {
        names
            .iter()
            .map(|n| {
                all.iter()
                    .find(|b| &b.0 == n)
                    .or_else(|| all.iter().find(|b| b.0.eq_ignore_ascii_case(n)))
                    .map(|b| (b.0.clone(), b.1.clone()))
                    .ok_or(E_GS_BLOB_NOT_FOUND)
            })
            .collect::<Result<_, _>>()?
    };
    let head = picked.len() * std::mem::size_of::<Blob>();
    let mut out = vec![0u8; head];
    for (k, (name, path)) in picked.iter().enumerate() {
        let data = std::fs::read(path).map_err(|_| E_GS_BLOB_NOT_FOUND)?;
        let name_at = out.len();
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        let data_at = out.len();
        out.extend_from_slice(&data);
        let e = k * std::mem::size_of::<Blob>();
        out[e..e + 8].copy_from_slice(&(name_at as u64).to_le_bytes());
        out[e + 8..e + 12].copy_from_slice(&(data.len() as u32).to_le_bytes());
        out[e + 16..e + 24].copy_from_slice(&(data_at as u64).to_le_bytes());
    }
    Ok((out, picked.len() as u32))
}

unsafe fn names_of(names: *const *const u8, count: u32) -> Result<Vec<String>, HRESULT> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if names.is_null() {
        return Err(E_INVALIDARG);
    }
    (0..count as usize).map(|i| from_cstr_ptr(*names.add(i)).ok_or(E_INVALIDARG)).collect()
}

/// Turn the offsets of a copied layout into pointers.
unsafe fn fix_up(buf: *mut Blob, count: u32) {
    for k in 0..count as usize {
        let b = &mut *buf.add(k);
        b.info.name = (buf as *const u8).add(b.info.name as usize);
        b.data = (buf as *mut u8).add(b.data as usize);
    }
}

pub unsafe extern "system" fn XGameSaveReadBlobData(c: *mut Container, names: *const *const u8, count: *mut u32, size: usize, out: *mut Blob) -> HRESULT {
    let Some(c) = container(c) else { return E_INVALIDARG };
    if count.is_null() {
        return E_INVALIDARG;
    }
    let requested = match names_of(names, *count) {
        Ok(names) => names,
        Err(e) => return e,
    };
    let (bytes, n) = match read_layout(&c.dir, &requested) {
        Ok(x) => x,
        Err(e) => return e,
    };
    if out.is_null() || size < bytes.len() {
        return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
    fix_up(out, n);
    *count = n;
    S_OK
}

static READ_ID: u8 = 0;

pub unsafe extern "system" fn XGameSaveReadBlobDataAsync(c: *mut Container, names: *const *const u8, count: u32, b: *mut XAsyncBlock) -> HRESULT {
    let Some(c) = container(c) else { return E_INVALIDARG };
    let Ok(list) = names_of(names, count) else { return E_INVALIDARG };
    let dir = c.dir.clone();
    begin_with(b, &READ_ID as *const u8 as usize, move || {
        let (mut bytes, n) = read_layout(&dir, &list)?;
        bytes.splice(0..0, n.to_le_bytes()); // count travels in front, dropped by the Result
        Ok(bytes)
    })
}

pub unsafe extern "system" fn XGameSaveReadBlobDataResult(b: *mut XAsyncBlock, size: usize, out: *mut Blob, count: *mut u32) -> HRESULT {
    let mut need = 0usize;
    let hr = XAsyncGetResultSize(b, &mut need);
    if hr < 0 {
        return hr;
    }
    if need < std::mem::size_of::<u32>() {
        return E_FAIL;
    }
    let body_size = need - std::mem::size_of::<u32>();
    if out.is_null() || size < body_size {
        // Do not consume the one-shot XAsync result until the caller can
        // receive it; the documented insufficient-buffer retry must succeed.
        return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    }
    let mut tmp = vec![0u8; need];
    let hr = XAsyncGetResult(b, &READ_ID as *const u8 as *const c_void, need, tmp.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
    if hr < 0 {
        return hr;
    }
    let n = u32::from_le_bytes(tmp[..4].try_into().unwrap());
    let body = &tmp[4..];
    debug_assert_eq!(body.len(), body_size);
    std::ptr::copy_nonoverlapping(body.as_ptr(), out as *mut u8, body.len());
    fix_up(out, n);
    if !count.is_null() {
        *count = n;
    }
    S_OK
}

pub unsafe extern "system" fn XGameSaveCreateUpdate(c: *mut Container, display: *const u8, out: *mut *mut Update) -> HRESULT {
    let Some(c) = container(c) else { return E_INVALIDARG };
    if out.is_null() {
        return E_INVALIDARG;
    }
    let _ = &c.name;
    *out = Box::into_raw(Box::new(Update { magic: MAGIC_UPDATE, dir: c.dir.clone(), display: from_cstr_ptr(display), ops: Vec::new() }));
    S_OK
}

pub unsafe extern "system" fn XGameSaveCloseUpdate(u: *mut Update) {
    if u.as_ref().is_some_and(|u| u.magic == MAGIC_UPDATE) {
        (*u).magic = 0;
        drop(Box::from_raw(u));
    }
}

unsafe fn update<'a>(u: *mut Update) -> Option<&'a mut Update> {
    u.as_mut().filter(|u| u.magic == MAGIC_UPDATE)
}

pub unsafe extern "system" fn XGameSaveSubmitBlobWrite(u: *mut Update, name: *const u8, data: *const u8, size: usize) -> HRESULT {
    let (Some(u), Some(n)) = (update(u), from_cstr_ptr(name)) else { return E_INVALIDARG };
    if data.is_null() && size > 0 {
        return E_INVALIDARG;
    }
    let bytes = if size == 0 { Vec::new() } else { std::slice::from_raw_parts(data, size).to_vec() };
    u.ops.push((n, Some(bytes)));
    S_OK
}

pub unsafe extern "system" fn XGameSaveSubmitBlobDelete(u: *mut Update, name: *const u8) -> HRESULT {
    let (Some(u), Some(n)) = (update(u), from_cstr_ptr(name)) else { return E_INVALIDARG };
    u.ops.push((n, None));
    S_OK
}

fn ignore_missing(result: io::Result<()>) -> io::Result<()> {
    result.or_else(|e| if e.kind() == ErrorKind::NotFound { Ok(()) } else { Err(e) })
}

fn copy_save_file(mut src: fs::File, dest: &Path) -> io::Result<()> {
    let mut dest = fs::File::create(dest)?;
    io::copy(&mut src, &mut dest)?;
    dest.sync_all()
}

fn commit_update(dir: &Path, display: Option<&str>, ops: &[(String, Option<Vec<u8>>)]) -> io::Result<()> {
    let _files = SAVE_FILES.write().unwrap_or_else(|e| e.into_inner());
    let current = current_snapshot(dir)?;
    let marker = if current.ends_with(SNAPSHOTS[0]) { SNAPSHOTS[1] } else { SNAPSHOTS[0] };
    let storage = dir.join("b");
    let stage = storage.join(marker);
    let staged_blobs = stage.join("b");
    ignore_missing(fs::remove_dir_all(&stage))?;
    let mut legacy_files = Vec::new();
    let result = (|| {
        fs::create_dir_all(&staged_blobs)?;
        let source_blobs = current.join("b");
        match fs::read_dir(source_blobs) {
            Ok(entries) => {
                for e in entries {
                    let e = e?;
                    let name = e.file_name();
                    if internal_file(&name) {
                        continue;
                    }
                    let ty = e.file_type()?;
                    let dest = staged_blobs.join(name);
                    if ty.is_file() {
                        copy_save_file(fs::File::open(e.path())?, &dest)?;
                        if current == dir {
                            legacy_files.push(e.path());
                        }
                    } else if ty.is_dir() {
                        fs::create_dir(dest)?;
                    }
                }
            }
            Err(e) if e.kind() == ErrorKind::NotFound && current == dir => {}
            Err(e) => return Err(e),
        }
        let staged_display = stage.join("display");
        let current_display = current.join("display");
        if let Some(display) = display {
            write_atomic(&staged_display, display.as_bytes())?;
        } else {
            match fs::File::open(&current_display) {
                Ok(src) => copy_save_file(src, &staged_display)?,
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        for (name, data) in ops {
            let path = staged_blobs.join(enc(name));
            match data {
                Some(data) => write_atomic(&path, data)?,
                None => ignore_missing(fs::remove_file(path))?,
            }
        }
        write_atomic(&storage.join(CURRENT), marker.as_bytes())
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(stage);
    } else if current == dir {
        for file in legacy_files {
            let _ = fs::remove_file(file);
        }
        let _ = fs::remove_file(dir.join("display"));
    } else {
        let _ = fs::remove_dir_all(current);
    }
    result
}

fn update_error(error: io::Error) -> HRESULT {
    use windows_sys::Win32::Foundation::{ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL};

    if error.kind() == ErrorKind::StorageFull
        || error.raw_os_error().is_some_and(|code| matches!(code as u32, ERROR_DISK_FULL | ERROR_HANDLE_DISK_FULL))
    {
        E_GS_OUT_OF_LOCAL_STORAGE
    } else {
        E_FAIL
    }
}

fn apply(dir: &Path, display: Option<&str>, ops: &[(String, Option<Vec<u8>>)]) -> HRESULT {
    commit_update(dir, display, ops).map_or_else(update_error, |()| S_OK)
}

pub unsafe extern "system" fn XGameSaveSubmitUpdate(u: *mut Update) -> HRESULT {
    let Some(u) = update(u) else { return E_INVALIDARG };
    apply(&u.dir, u.display.as_deref(), &u.ops)
}

static SUBMIT_ID: u8 = 0;

pub unsafe extern "system" fn XGameSaveSubmitUpdateAsync(u: *mut Update, b: *mut XAsyncBlock) -> HRESULT {
    let Some(u) = update(u) else { return E_INVALIDARG };
    let (dir, display, ops) = (u.dir.clone(), u.display.clone(), u.ops.clone());
    let queued = Queued::new();
    begin_with(b, &SUBMIT_ID as *const u8 as usize, move || {
        let _queued = queued;
        match apply(&dir, display.as_deref(), &ops) {
            S_OK => Ok(Vec::new()),
            e => Err(e),
        }
    })
}

pub unsafe extern "system" fn XGameSaveSubmitUpdateResult(b: *mut XAsyncBlock) -> HRESULT {
    XAsyncGetResult(b, &SUBMIT_ID as *const u8 as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::taskqueue::{
        XTaskQueueCloseHandle, XTaskQueueCreate, XTaskQueueDispatch, XTaskQueueTerminate,
        MODE_MANUAL, PORT_WORK,
    };
    use crate::util::{E_ABORT, E_PENDING};

    const DELETE_TEST_NAME: &str = "slot/1";

    #[test]
    fn save_update_reports_full_host_storage() {
        use windows_sys::Win32::Foundation::{ERROR_ACCESS_DENIED, ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL};

        for code in [ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL] {
            assert_eq!(update_error(io::Error::from_raw_os_error(code as i32)), E_GS_OUT_OF_LOCAL_STORAGE);
        }
        assert_eq!(update_error(ErrorKind::StorageFull.into()), E_GS_OUT_OF_LOCAL_STORAGE);
        assert_eq!(update_error(io::Error::from_raw_os_error(ERROR_ACCESS_DENIED as i32)), E_FAIL);
    }

    struct DeleteFixture {
        root: PathBuf,
        provider: *mut Provider,
        block: Box<XAsyncBlock>,
    }

    impl DeleteFixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!("xcl-gamesave-delete-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            let mut fixture = Self {
                provider: Box::into_raw(Box::new(Provider { magic: MAGIC_PROVIDER, root: root.clone() })),
                root,
                block: Box::new(unsafe { std::mem::zeroed() }),
            };
            unsafe {
                assert_eq!(XTaskQueueCreate(MODE_MANUAL, MODE_MANUAL, &mut fixture.block.queue), S_OK);
            }
            std::fs::create_dir_all(fixture.container_dir().join("b")).unwrap();
            std::fs::write(fixture.blob_path(), b"synthetic save").unwrap();
            fixture
        }

        fn container_dir(&self) -> PathBuf {
            self.root.join(enc(DELETE_TEST_NAME))
        }

        fn blob_path(&self) -> PathBuf {
            self.container_dir().join("b").join("data")
        }
    }

    impl Drop for DeleteFixture {
        fn drop(&mut self) {
            unsafe {
                let queue = self.block.queue;
                if !queue.is_null() {
                    // Drain pending work before freeing the block, including after an assertion fails.
                    let _ = XTaskQueueTerminate(queue, true, std::ptr::null_mut(), 0);
                    XTaskQueueCloseHandle(queue);
                }
                XGameSaveCloseProvider(self.provider);
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn exit_waits_for_a_commit_under_way_then_holds_the_save_files() {
        let files = std::sync::Arc::new(RwLock::new(()));
        let (queued, (tx, rx)) = (AtomicUsize::new(0), std::sync::mpsc::channel());
        let writer = {
            let files = files.clone();
            std::thread::spawn(move || {
                let _commit = files.write().unwrap();
                tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(200));
            })
        };
        rx.recv().unwrap();
        let start = Instant::now();
        let held = settle(&files, &queued, Duration::from_secs(5));
        assert!(held.is_some());
        assert!(start.elapsed() >= Duration::from_millis(150), "{:?}", start.elapsed());
        assert!(matches!(files.try_read(), Err(TryLockError::WouldBlock)), "no new commit or read starts");
        drop(held);
        writer.join().unwrap();
    }

    #[test]
    fn exit_waits_for_queued_async_saves() {
        let (files, queued) = (std::sync::Arc::new(RwLock::new(())), std::sync::Arc::new(AtomicUsize::new(1)));
        let job = {
            let queued = queued.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                queued.store(0, Ordering::Release);
            })
        };
        let start = Instant::now();
        assert!(settle(&files, &queued, Duration::from_secs(5)).is_some());
        assert!(start.elapsed() >= Duration::from_millis(150), "{:?}", start.elapsed());
        job.join().unwrap();
    }

    #[test]
    fn exit_wait_is_bounded() {
        let files = std::sync::Arc::new(RwLock::new(()));
        let start = Instant::now();
        assert!(settle(&files, &AtomicUsize::new(1), Duration::from_millis(100)).is_some(), "a job never dispatched: the files are still fenced");
        assert!(start.elapsed() < Duration::from_secs(2));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let writer = {
            let files = files.clone();
            std::thread::spawn(move || {
                let _stuck = files.write().unwrap();
                held_tx.send(()).unwrap();
                let _ = rx.recv();
            })
        };
        held_rx.recv().unwrap();
        let start = Instant::now();
        assert!(settle(&files, &AtomicUsize::new(0), Duration::from_millis(100)).is_none());
        assert!(start.elapsed() < Duration::from_secs(2));
        drop(tx);
        writer.join().unwrap();
    }

    #[test]
    fn async_saves_count_as_queued_until_run_or_dropped() {
        let _g = ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let before = QUEUED.load(Ordering::Acquire);
        let mut fixture = DeleteFixture::new();
        unsafe {
            let name = CString::new(DELETE_TEST_NAME).unwrap();
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), &mut *fixture.block), S_OK);
            assert_eq!(QUEUED.load(Ordering::Acquire), before + 1);
            assert!(XTaskQueueDispatch(fixture.block.queue, PORT_WORK, 0));
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), S_OK);
            assert_eq!(QUEUED.load(Ordering::Acquire), before);
            std::fs::create_dir_all(fixture.container_dir().join("b")).unwrap();
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), &mut *fixture.block), S_OK);
            assert_eq!(QUEUED.load(Ordering::Acquire), before + 1);
            assert_eq!(XTaskQueueTerminate(fixture.block.queue, true, std::ptr::null_mut(), 0), S_OK);
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), E_ABORT);
        }
        assert_eq!(QUEUED.load(Ordering::Acquire), before, "a job its queue dropped no longer counts");
    }

    #[test]
    fn delete_container_async_runs_on_work_dispatch_with_owned_inputs() {
        let _g = ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut fixture = DeleteFixture::new();
        unsafe {
            let name = CString::new(DELETE_TEST_NAME).unwrap();
            let sibling = fixture.root.join("another-slot");
            std::fs::create_dir(&sibling).unwrap();
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), &mut *fixture.block), S_OK);
            drop(name);
            XGameSaveCloseProvider(fixture.provider);
            fixture.provider = std::ptr::null_mut();

            assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), E_PENDING);
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), E_PENDING);
            assert_eq!(std::fs::read(fixture.blob_path()).unwrap().as_slice(), b"synthetic save");
            assert!(XTaskQueueDispatch(fixture.block.queue, PORT_WORK, 0));
            assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), S_OK);
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), S_OK);
            assert!(sibling.is_dir());
        }
        assert!(!fixture.container_dir().exists());
        assert!(fixture.root.is_dir());
    }

    #[test]
    fn delete_container_async_terminated_queue_preserves_save() {
        let _g = ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut fixture = DeleteFixture::new();
        unsafe {
            let name = CString::new(DELETE_TEST_NAME).unwrap();
            let queue = fixture.block.queue;
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), &mut *fixture.block), S_OK);
            assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), E_PENDING);
            assert_eq!(XTaskQueueTerminate(queue, true, std::ptr::null_mut(), 0), S_OK);
            assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), E_ABORT);
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), E_ABORT);
            assert_eq!(std::fs::read(fixture.blob_path()).unwrap().as_slice(), b"synthetic save");

            // A new request on the same queue fails at scheduling.
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), &mut *fixture.block), S_OK);
            assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), E_ABORT);
            assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), E_ABORT);
            assert!(!XTaskQueueDispatch(queue, PORT_WORK, 0));
        }
        assert_eq!(std::fs::read(fixture.blob_path()).unwrap().as_slice(), b"synthetic save");
    }

    #[test]
    fn delete_container_invalid_arguments_preserve_provider_root() {
        let _g = ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut fixture = DeleteFixture::new();
        unsafe {
            let name = CString::new(DELETE_TEST_NAME).unwrap();
            let too_long = CString::new("a".repeat(257)).unwrap();
            assert_eq!(XGameSaveDeleteContainerAsync(fixture.provider, name.as_ptr().cast(), std::ptr::null_mut()), E_INVALIDARG);
            for (provider, name, expected) in [
                (std::ptr::null_mut(), name.as_ptr().cast(), E_INVALIDARG),
                (fixture.provider, std::ptr::null(), E_INVALIDARG),
                (fixture.provider, b"\0".as_ptr(), E_GS_INVALID_CONTAINER_NAME),
                (fixture.provider, too_long.as_ptr().cast(), E_GS_INVALID_CONTAINER_NAME),
            ] {
                assert_eq!(XGameSaveDeleteContainer(provider, name), expected);
                assert_eq!(XGameSaveDeleteContainerAsync(provider, name, &mut *fixture.block), S_OK);
                assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), E_PENDING);
                assert!(XTaskQueueDispatch(fixture.block.queue, PORT_WORK, 0));
                assert_eq!(XAsyncGetStatus(&mut *fixture.block, false), expected);
                assert_eq!(XGameSaveDeleteContainerResult(&mut *fixture.block), expected);
                assert_eq!(std::fs::read(fixture.blob_path()).unwrap().as_slice(), b"synthetic save");
            }
        }
        assert!(fixture.root.is_dir());
    }

    unsafe extern "system" fn count_containers(i: *const ContainerInfo, ctx: *mut c_void) -> bool {
        let v = &mut *(ctx as *mut Vec<(String, String, u32, u64)>);
        let i = &*i;
        v.push((from_cstr_ptr(i.name).unwrap(), from_cstr_ptr(i.display_name).unwrap(), i.blob_count, i.total_size));
        true
    }

    #[test]
    fn failed_updates_preserve_display_and_all_blobs() {
        let _g = ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut fixture = DeleteFixture::new();
        let dir = fixture.container_dir();
        fs::write(dir.join("display"), b"Original").unwrap();
        fs::write(dir.join("b").join("other"), b"keep").unwrap();
        fs::create_dir(dir.join("b").join("blocked")).unwrap();
        let (mut c, mut u) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            assert_eq!(XGameSaveCreateContainer(fixture.provider, b"slot/1\0".as_ptr(), &mut c), S_OK);
            for asynchronous in [false, true] {
                if asynchronous {
                    assert_eq!(XGameSaveCreateUpdate(c, std::ptr::null(), &mut u), S_OK);
                    assert_eq!(XGameSaveSubmitUpdate(u), S_OK);
                    XGameSaveCloseUpdate(u);
                }
                assert_eq!(XGameSaveCreateUpdate(c, b"Changed\0".as_ptr(), &mut u), S_OK);
                assert_eq!(XGameSaveSubmitBlobWrite(u, b"data\0".as_ptr(), b"changed".as_ptr(), 7), S_OK);
                assert_eq!(XGameSaveSubmitBlobDelete(u, b"other\0".as_ptr()), S_OK);
                assert_eq!(XGameSaveSubmitBlobWrite(u, b"blocked\0".as_ptr(), b"invalid".as_ptr(), 7), S_OK);
                if asynchronous {
                    assert_eq!(XGameSaveSubmitUpdateAsync(u, &mut *fixture.block), S_OK);
                    XGameSaveCloseUpdate(u);
                    assert!(XTaskQueueDispatch(fixture.block.queue, PORT_WORK, 0));
                    assert_eq!(XGameSaveSubmitUpdateResult(&mut *fixture.block), E_FAIL);
                } else {
                    assert_eq!(XGameSaveSubmitUpdate(u), E_FAIL);
                    XGameSaveCloseUpdate(u);
                }

                let mut list: Vec<(String, String, u32, u64)> = Vec::new();
                assert_eq!(
                    XGameSaveEnumerateContainerInfo(fixture.provider, std::ptr::addr_of_mut!(list).cast(), Some(count_containers)),
                    S_OK
                );
                assert_eq!(list, [("slot/1".into(), "Original".into(), 2, 18)]);
                let mut n = 0;
                let mut buf = [0u64; 16];
                let blobs = buf.as_mut_ptr().cast::<Blob>();
                assert_eq!(XGameSaveReadBlobData(c, std::ptr::null(), &mut n, std::mem::size_of_val(&buf), blobs), S_OK);
                assert_eq!(n, 2);
                for (i, (name, data)) in [("data", &b"synthetic save"[..]), ("other", &b"keep"[..])].into_iter().enumerate() {
                    let blob = &*blobs.add(i);
                    assert_eq!(from_cstr_ptr(blob.info.name).as_deref(), Some(name));
                    assert_eq!(std::slice::from_raw_parts(blob.data, blob.info.size as usize), data);
                }
            }
            XGameSaveCloseContainer(c);
        }
    }

    #[test]
    fn read_blob_names_require_a_complete_counted_array() {
        // Another test temporarily routes the process-global XAsync entry points
        // through the system runtime. Keep this invalid-block status check on the
        // local implementation, like every other test that exercises XAsync.
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unsafe {
            let mut provider = std::ptr::null_mut();
            assert_eq!(
                XGameSaveInitializeProvider(
                    std::ptr::null_mut(),
                    b"xcl-read-name-validation\0".as_ptr(),
                    false,
                    &mut provider,
                ),
                S_OK
            );
            let _ = XGameSaveDeleteContainer(provider, b"slot\0".as_ptr());
            let mut container = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(provider, b"slot\0".as_ptr(), &mut container), S_OK);
            let mut update = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(container, std::ptr::null(), &mut update), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(update, b"data\0".as_ptr(), b"secret".as_ptr(), 6), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(update), S_OK);
            XGameSaveCloseUpdate(update);

            let mut count = 1u32;
            let mut buffer = vec![0u64; 32];
            assert_eq!(
                XGameSaveReadBlobData(container, std::ptr::null(), &mut count, 256, buffer.as_mut_ptr() as *mut Blob),
                E_INVALIDARG
            );
            assert_eq!(count, 1, "invalid filter changed the caller's count");
            let null_name = [std::ptr::null()];
            assert_eq!(
                XGameSaveReadBlobData(container, null_name.as_ptr(), &mut count, 256, buffer.as_mut_ptr() as *mut Blob),
                E_INVALIDARG
            );

            let mut async_block: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveReadBlobDataAsync(container, std::ptr::null(), 1, &mut async_block), E_INVALIDARG);
            assert_eq!(XAsyncGetStatus(&mut async_block, false), E_INVALIDARG, "invalid async filter started an operation");
            assert_eq!(XGameSaveReadBlobDataAsync(container, null_name.as_ptr(), 1, &mut async_block), E_INVALIDARG);

            // The documented empty filter remains the read-all form.
            count = 0;
            assert_eq!(
                XGameSaveReadBlobData(container, std::ptr::null(), &mut count, 256, buffer.as_mut_ptr() as *mut Blob),
                S_OK
            );
            assert_eq!(count, 1);

            XGameSaveCloseContainer(container);
            assert_eq!(XGameSaveDeleteContainer(provider, b"slot\0".as_ptr()), S_OK);
            XGameSaveCloseProvider(provider);
        }
    }

    unsafe extern "system" fn blob_names(i: *const BlobInfo, ctx: *mut c_void) -> bool {
        (*(ctx as *mut Vec<(String, u32)>)).push((from_cstr_ptr((*i).name).unwrap(), (*i).size));
        true
    }

    #[test]
    fn save_updates_replace_files_past_max_path() {
        let nonce = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("xcl-gamesave-long-{}-{nonce}", std::process::id()));
        let dir = root.join("a".repeat(150)).join("b".repeat(150));
        let first = apply(&dir, Some("Long"), &[("data".into(), Some(b"old".to_vec()))]);
        let second = apply(&dir, None, &[("data".into(), Some(b"new".to_vec()))]);
        let contents = current_snapshot(&dir).and_then(|p| fs::read(p.join("b").join("data")));
        let _ = fs::remove_dir_all(&root);
        assert_eq!(first, S_OK);
        assert_eq!(second, S_OK);
        assert_eq!(contents.unwrap(), b"new");
    }

    #[cfg(windows)]
    #[test]
    fn move_file_paths_are_extended_length() {
        for (path, expected) in [
            (r"C:\save\blob", r"\\?\C:\save\blob"),
            (r"\\server\share\blob", r"\\?\UNC\server\share\blob"),
            (r"\\?\C:\save\blob", r"\\?\C:\save\blob"),
        ] {
            assert_eq!(
                verbatim_wide_path(Path::new(path)).unwrap(),
                expected.encode_utf16().chain(std::iter::once(0)).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn decode_names_with_malformed_percent_escapes() {
        for name in ["%aé", "%é", "%+1", "%A", "%", "%GG"] {
            assert_eq!(dec(name), name);
        }
        assert_eq!(dec("%2F%2f"), "//");
        assert_eq!(dec("%C3%A9"), "é");
    }

    /// Names Windows does not keep as file names (device names, a trailing dot or space,
    /// "." / "..") come back as they were written.
    #[test]
    fn names_windows_would_change_round_trip() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-names-{}\0", std::process::id());
            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProvider(std::ptr::null_mut(), cfg.as_ptr(), false, &mut p), S_OK);
            for container_name in ["aux", "..", "slot."] {
                let cname = format!("{container_name}\0");
                let mut c = std::ptr::null_mut();
                assert_eq!(XGameSaveCreateContainer(p, cname.as_ptr(), &mut c), S_OK);
                let mut u = std::ptr::null_mut();
                assert_eq!(XGameSaveCreateUpdate(c, std::ptr::null(), &mut u), S_OK);
                let blobs: [&[u8]; 5] = [b"aux\0", b"CON.dat\0", b"save.\0", b"save \0", b"nul .txt\0"];
                for (i, name) in blobs.iter().enumerate() {
                    assert_eq!(XGameSaveSubmitBlobWrite(u, name.as_ptr(), b"0123".as_ptr(), i + 1), S_OK);
                }
                assert_eq!(XGameSaveSubmitUpdate(u), S_OK, "{container_name}");
                XGameSaveCloseUpdate(u);
                let mut names: Vec<(String, u32)> = Vec::new();
                assert_eq!(XGameSaveEnumerateBlobInfo(c, &mut names as *mut _ as *mut c_void, Some(blob_names)), S_OK);
                names.sort();
                let want: Vec<(String, u32)> = vec![("CON.dat".into(), 2), ("aux".into(), 1), ("nul .txt".into(), 5), ("save ".into(), 4), ("save.".into(), 3)];
                assert_eq!(names, want, "{container_name}");
                XGameSaveCloseContainer(c);
            }
            let mut list: Vec<(String, String, u32, u64)> = Vec::new();
            assert_eq!(XGameSaveEnumerateContainerInfo(p, &mut list as *mut _ as *mut c_void, Some(count_containers)), S_OK);
            let mut got: Vec<String> = list.into_iter().map(|c| c.0).collect();
            got.sort();
            assert_eq!(got, ["..", "aux", "slot."]);
            for name in [&b"aux\0"[..], b"..\0", b"slot.\0", b".\0"] {
                assert_eq!(XGameSaveDeleteContainer(p, name.as_ptr()), S_OK);
            }
            // deleting "." / ".." deleted only those containers, not the provider's folder or its parent
            assert!((*p).root.is_dir());
            XGameSaveCloseProvider(p);
        }
    }

    #[test]
    fn containers_blobs_round_trip() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-{}\0", std::process::id());
            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProvider(std::ptr::null_mut(), cfg.as_ptr(), false, &mut p), S_OK);
            let mut c = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(p, b"slot/1\0".as_ptr(), &mut c), S_OK);
            let mut u = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, b"Save 1\0".as_ptr(), &mut u), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"data\0".as_ptr(), b"hello".as_ptr(), 5), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"meta\0".as_ptr(), b"xy".as_ptr(), 2), S_OK);
            // async submit
            let mut blk: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveSubmitUpdateAsync(u, &mut blk), S_OK);
            assert_eq!(XAsyncGetStatus(&mut blk, true), S_OK);
            assert_eq!(XGameSaveSubmitUpdateResult(&mut blk), S_OK);
            XGameSaveCloseUpdate(u);
            let mut list: Vec<(String, String, u32, u64)> = Vec::new();
            assert_eq!(XGameSaveEnumerateContainerInfo(p, &mut list as *mut _ as *mut c_void, Some(count_containers)), S_OK);
            assert_eq!(list, vec![("slot/1".to_string(), "Save 1".to_string(), 2, 7)]);
            // read everything (sync): too small first
            let mut n = 0u32;
            let mut buf = vec![0u64; 32];
            assert_eq!(XGameSaveReadBlobData(c, std::ptr::null(), &mut n, 8, buf.as_mut_ptr() as *mut Blob), E_GS_PROVIDED_BUFFER_TOO_SMALL);
            n = 0;
            assert_eq!(XGameSaveReadBlobData(c, std::ptr::null(), &mut n, 256, buf.as_mut_ptr() as *mut Blob), S_OK);
            assert_eq!(n, 2);
            let b0 = &*(buf.as_ptr() as *const Blob);
            assert_eq!((from_cstr_ptr(b0.info.name).unwrap(), b0.info.size), ("data".to_string(), 5));
            assert_eq!(std::slice::from_raw_parts(b0.data, 5), b"hello");
            // read one by name (async), then delete it
            let names = [b"meta\0".as_ptr()];
            let mut blk: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveReadBlobDataAsync(c, names.as_ptr(), 1, &mut blk), S_OK);
            assert_eq!(XAsyncGetStatus(&mut blk, true), S_OK);
            let mut buf2 = vec![0u64; 16];
            let mut n2 = u32::MAX;
            assert_eq!(
                XGameSaveReadBlobDataResult(&mut blk, 1, buf2.as_mut_ptr() as *mut Blob, &mut n2),
                E_GS_PROVIDED_BUFFER_TOO_SMALL
            );
            assert_eq!(n2, u32::MAX, "failed result changed the output count");
            assert_eq!(XGameSaveReadBlobDataResult(&mut blk, 128, buf2.as_mut_ptr() as *mut Blob, &mut n2), S_OK);
            let b1 = &*(buf2.as_ptr() as *const Blob);
            assert_eq!((n2, std::slice::from_raw_parts(b1.data, 2)), (1, &b"xy"[..]));
            let mut u = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, std::ptr::null(), &mut u), S_OK);
            assert_eq!(XGameSaveSubmitBlobDelete(u, b"meta\0".as_ptr()), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u), S_OK);
            XGameSaveCloseUpdate(u);
            let names = [b"meta\0".as_ptr()];
            let mut n3 = 1u32;
            assert_eq!(XGameSaveReadBlobData(c, names.as_ptr(), &mut n3, 256, buf.as_mut_ptr() as *mut Blob), E_GS_BLOB_NOT_FOUND);
            let mut q = 0i64;
            assert_eq!(XGameSaveGetRemainingQuota(p, &mut q), S_OK);
            assert_eq!(q, QUOTA - 5);
            XGameSaveCloseContainer(c);
            assert_eq!(XGameSaveDeleteContainer(p, b"slot/1\0".as_ptr()), S_OK);
            assert_eq!(XGameSaveGetContainerInfo(p, b"slot/1\0".as_ptr(), std::ptr::null_mut(), Some(count_containers)), E_GS_CONTAINER_NOT_FOUND);
            let root = (*p).root.clone();
            XGameSaveCloseProvider(p);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn atomic_writes_preserve_xcltmp_blobs_and_skip_leftover_temps() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-atomic-{}\0", std::process::id());
            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProvider(std::ptr::null_mut(), cfg.as_ptr(), false, &mut p), S_OK);
            let root = (*p).root.clone();
            let mut c = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(p, b"slot\0".as_ptr(), &mut c), S_OK);

            // Write a `.xcltmp` blob, a `~xcltmp.*` blob, and a sibling `.dat` blob.
            let mut u = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, b"Atomic\0".as_ptr(), &mut u), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"save.xcltmp\0".as_ptr(), b"orig-xcltmp".as_ptr(), 11), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"~xcltmp.1.0\0".as_ptr(), b"orig-tilde".as_ptr(), 10), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"save.dat\0".as_ptr(), b"dat-v1".as_ptr(), 6), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u), S_OK);
            XGameSaveCloseUpdate(u);

            // Overwriting `save.dat` must never clobber `save.xcltmp`, and overwriting `save.xcltmp` must succeed.
            let mut u = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, std::ptr::null(), &mut u), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"save.dat\0".as_ptr(), b"dat-v2".as_ptr(), 6), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"save.xcltmp\0".as_ptr(), b"next-xcltmp".as_ptr(), 11), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u), S_OK);
            XGameSaveCloseUpdate(u);

            // A simulated crash-leftover temp file in `b/` must never be enumerated as a blob.
            let blobs_dir = (*c).dir.join("b");
            let leftover = blobs_dir.join(format!("{TMP_PREFIX}99999.7"));
            std::fs::write(&leftover, b"partial").unwrap();

            let mut listed: Vec<(String, u32)> = Vec::new();
            assert_eq!(XGameSaveEnumerateBlobInfo(c, &mut listed as *mut _ as *mut c_void, Some(blob_names)), S_OK);
            assert_eq!(
                listed,
                vec![
                    ("save.dat".to_string(), 6),
                    ("save.xcltmp".to_string(), 11),
                    ("~xcltmp.1.0".to_string(), 10),
                ]
            );

            // Verify all three blobs retained their exact contents.
            let names = [b"save.dat\0".as_ptr(), b"save.xcltmp\0".as_ptr(), b"~xcltmp.1.0\0".as_ptr()];
            let mut count = 3u32;
            let mut buf = vec![0u64; 64];
            assert_eq!(XGameSaveReadBlobData(c, names.as_ptr(), &mut count, 512, buf.as_mut_ptr() as *mut Blob), S_OK);
            assert_eq!(count, 3);
            let b0 = &*(buf.as_ptr() as *const Blob);
            let b1 = &*(buf.as_ptr() as *const Blob).add(1);
            let b2 = &*(buf.as_ptr() as *const Blob).add(2);
            assert_eq!(std::slice::from_raw_parts(b0.data, b0.info.size as usize), b"dat-v2");
            assert_eq!(std::slice::from_raw_parts(b1.data, b1.info.size as usize), b"next-xcltmp");
            assert_eq!(std::slice::from_raw_parts(b2.data, b2.info.size as usize), b"orig-tilde");

            // Failed atomic replace (dest is an existing directory) removes its temp file.
            let _ = std::fs::remove_file(&leftover);
            let blocker = blobs_dir.join("dir_blocker");
            std::fs::create_dir_all(&blocker).unwrap();
            assert!(write_atomic(&blocker, b"cannot-replace-dir").is_err());
            let remaining_tmps: Vec<_> = std::fs::read_dir(&blobs_dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with(TMP_PREFIX))
                .collect();
            assert!(remaining_tmps.is_empty(), "temp file leaked on failed write_atomic: {remaining_tmps:?}");

            XGameSaveCloseContainer(c);
            XGameSaveCloseProvider(p);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn case_insensitive_blob_and_container_matching_on_ntfs() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-case-{}\0", std::process::id());
            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProvider(std::ptr::null_mut(), cfg.as_ptr(), false, &mut p), S_OK);
            let root = (*p).root.clone();

            // Write container "SlotMixed" and blob "Meta"
            let mut c1 = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(p, b"SlotMixed\0".as_ptr(), &mut c1), S_OK);
            let mut u1 = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c1, b"Display 1\0".as_ptr(), &mut u1), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u1, b"Meta\0".as_ptr(), b"old".as_ptr(), 3), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u1), S_OK);
            XGameSaveCloseUpdate(u1);
            XGameSaveCloseContainer(c1);

            // Re-open with lowercase name, overwrite "meta", then query/read using different casings.
            let mut c2 = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(p, b"slotmixed\0".as_ptr(), &mut c2), S_OK);
            let mut u2 = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c2, b"Display 2\0".as_ptr(), &mut u2), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u2, b"meta\0".as_ptr(), b"fresh".as_ptr(), 5), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u2), S_OK);
            XGameSaveCloseUpdate(u2);

            for query_name in [b"meta\0".as_ptr(), b"Meta\0".as_ptr(), b"META\0".as_ptr()] {
                let names = [query_name];
                let mut count = 1u32;
                let mut buf = vec![0u64; 32];
                assert_eq!(XGameSaveReadBlobData(c2, names.as_ptr(), &mut count, 256, buf.as_mut_ptr() as *mut Blob), S_OK);
                assert_eq!(count, 1);
                let b0 = &*(buf.as_ptr() as *const Blob);
                assert_eq!(std::slice::from_raw_parts(b0.data, b0.info.size as usize), b"fresh");
            }

            let mut prefixed_blobs: Vec<(String, u32)> = Vec::new();
            assert_eq!(
                XGameSaveEnumerateBlobInfoByName(c2, b"me\0".as_ptr(), &mut prefixed_blobs as *mut _ as *mut c_void, Some(blob_names)),
                S_OK
            );
            assert_eq!(prefixed_blobs.len(), 1);
            assert!(prefixed_blobs[0].0.eq_ignore_ascii_case("meta"));

            let mut exact_c: Vec<(String, String, u32, u64)> = Vec::new();
            assert_eq!(
                XGameSaveGetContainerInfo(p, b"slotmixed\0".as_ptr(), &mut exact_c as *mut _ as *mut c_void, Some(count_containers)),
                S_OK
            );
            assert_eq!(exact_c.len(), 1);
            assert_eq!(exact_c[0].1, "Display 2");

            let mut prefix_c: Vec<(String, String, u32, u64)> = Vec::new();
            assert_eq!(
                XGameSaveEnumerateContainerInfoByName(p, b"slot\0".as_ptr(), &mut prefix_c as *mut _ as *mut c_void, Some(count_containers)),
                S_OK
            );
            assert_eq!(prefix_c.len(), 1);

            XGameSaveCloseContainer(c2);
            XGameSaveCloseProvider(p);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn initialize_provider_result_rejects_null_out_without_consuming_async() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-null-out-{}\0", std::process::id());
            let mut blk: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveInitializeProviderAsync(std::ptr::null_mut(), cfg.as_ptr(), false, &mut blk), S_OK);
            assert_eq!(XAsyncGetStatus(&mut blk, true), S_OK);

            // Passing null `out` must fail with E_INVALIDARG and preserve the completed async result.
            assert_eq!(XGameSaveInitializeProviderResult(&mut blk, std::ptr::null_mut()), E_INVALIDARG);

            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProviderResult(&mut blk, &mut p), S_OK);
            assert!(!p.is_null());
            let root = (*p).root.clone();
            XGameSaveCloseProvider(p);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn delete_container_rejects_empty_name_without_wiping_provider_root() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cfg = format!("xcl-test-empty-del-{}\0", std::process::id());
            let mut p = std::ptr::null_mut();
            assert_eq!(XGameSaveInitializeProvider(std::ptr::null_mut(), cfg.as_ptr(), false, &mut p), S_OK);
            let root = (*p).root.clone();
            let mut c = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateContainer(p, b"slot1\0".as_ptr(), &mut c), S_OK);
            let mut u = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, b"Save 1\0".as_ptr(), &mut u), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(u, b"data\0".as_ptr(), b"keepme".as_ptr(), 6), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(u), S_OK);
            XGameSaveCloseUpdate(u);

            // Empty and overlong names must be rejected without touching the provider root or any container.
            assert_eq!(XGameSaveDeleteContainer(p, b"\0".as_ptr()), E_GS_INVALID_CONTAINER_NAME);
            let overlong = format!("{}\0", "a".repeat(257));
            assert_eq!(XGameSaveDeleteContainer(p, overlong.as_ptr()), E_GS_INVALID_CONTAINER_NAME);

            let mut blk: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveDeleteContainerAsync(p, b"\0".as_ptr(), &mut blk), S_OK);
            assert_eq!(XAsyncGetStatus(&mut blk, true), E_GS_INVALID_CONTAINER_NAME);
            assert_eq!(XGameSaveDeleteContainerResult(&mut blk), E_GS_INVALID_CONTAINER_NAME);

            assert!(root.is_dir(), "provider root must still exist");
            let mut list: Vec<(String, String, u32, u64)> = Vec::new();
            assert_eq!(XGameSaveEnumerateContainerInfo(p, &mut list as *mut _ as *mut c_void, Some(count_containers)), S_OK);
            assert_eq!(list, vec![("slot1".to_string(), "Save 1".to_string(), 1, 6)]);

            let names = [b"data\0".as_ptr()];
            let mut n = 1u32;
            let mut buf = vec![0u64; 32];
            assert_eq!(XGameSaveReadBlobData(c, names.as_ptr(), &mut n, 256, buf.as_mut_ptr() as *mut Blob), S_OK);
            assert_eq!(n, 1);
            let b0 = &*(buf.as_ptr() as *const Blob);
            assert_eq!(std::slice::from_raw_parts(b0.data, 6), b"keepme");

            XGameSaveCloseContainer(c);
            XGameSaveCloseProvider(p);
            let _ = std::fs::remove_dir_all(root);
        }
    }

}
