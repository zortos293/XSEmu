use std::ffi::c_void;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::mem::{size_of, size_of_val, MaybeUninit};
use std::os::windows::fs::FileExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows_sys::Win32::Foundation::{
    CloseHandle, DuplicateHandle, LocalFree, DUPLICATE_SAME_ACCESS, HANDLE, INVALID_HANDLE_VALUE,
    WAIT_OBJECT_0,
};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::Storage::FileSystem::*;
use windows_sys::Win32::System::Diagnostics::Debug::*;
use windows_sys::Win32::System::Memory::{
    VirtualQueryEx, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED,
    MEM_PRIVATE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32EnumProcessModules, K32GetMappedFileNameW, K32GetModuleFileNameExW,
};
use windows_sys::Win32::System::SystemServices::FILE_PERSISTENT_ACLS;
use windows_sys::Win32::System::Threading::{
    CreateEventW, GetCurrentProcess, GetCurrentProcessId, GetCurrentThread, GetCurrentThreadId,
    OpenProcessToken, OpenThread, ResetEvent, SetEvent, WaitForSingleObject, INFINITE,
    THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION,
};

const AV: u32 = 0xC000_0005;
const LIMIT: u64 = 16 * 1024;
pub(super) const DUMP_MS: u32 = 12_000;
const OWNER_WAIT_MS: u32 = 15_000;
const IDLE: usize = 0;
const IN_FLIGHT: usize = 1;
const DETACHED: usize = 2;
const SIGNALING: usize = 3;
const DONE: usize = 4;
const STATE_MASK: usize = 7;
static OWNER: AtomicU32 = AtomicU32::new(0);
static FAULT_TID: AtomicU32 = AtomicU32::new(0);
static FAULT_IP: AtomicUsize = AtomicUsize::new(0);
static REPORTER: OnceLock<Reporter> = OnceLock::new();
static EXE: OnceLock<PathBuf> = OnceLock::new();
static EXE_BASE: AtomicUsize = AtomicUsize::new(0);

#[repr(align(16))]
struct AlignedContext(CONTEXT);

struct Reporter {
    request: OwnedHandle,
    done: OwnedHandle,
    busy: AtomicUsize,
    job: Mutex<Option<(usize, Job)>>,
    ok: AtomicBool,
}

struct Job {
    tid: u32,
    rec: EXCEPTION_RECORD,
    ctx: Option<CONTEXT>,
    title: Option<(OwnedHandle, u32)>,
    summary: bool,
}

unsafe impl Send for Job {}

#[link(name = "advapi32")]
extern "system" {
    fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl: *const u16,
        revision: u32,
        sd: *mut *mut c_void,
        size: *mut u32,
    ) -> i32;
    fn ConvertSidToStringSidW(sid: *mut c_void, text: *mut *mut u16) -> i32;
}

fn local(path: &Path) -> bool {
    use std::path::{Component, Prefix};
    let Some(Component::Prefix(prefix)) = path.components().next() else {
        return false;
    };
    let drive = match prefix.kind() {
        Prefix::Disk(drive) | Prefix::VerbatimDisk(drive) => drive,
        _ => return false,
    };
    path.is_absolute()
        && unsafe { GetDriveTypeW([drive as u16, b':' as u16, b'\\' as u16, 0].as_ptr()) } == 3
}

fn verified_dir(path: &Path) -> Option<PathBuf> {
    if !local(path) {
        return None;
    }
    let resolved = path.canonicalize().ok()?;
    let input = path.to_str()?;
    let actual = resolved.to_str()?;
    (local(&resolved)
        && input
            .trim_start_matches(r"\\?\")
            .eq_ignore_ascii_case(actual.trim_start_matches(r"\\?\")))
    .then_some(resolved)
}

fn sddl() -> Option<Vec<u16>> {
    use windows_sys::Win32::Security::{GetTokenInformation, TokenUser, TOKEN_QUERY, TOKEN_USER};
    unsafe {
        let mut token = null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return None;
        }
        let token = OwnedHandle::from_raw_handle(token);
        let mut buffer = [0usize; 64];
        let mut len = 0;
        if GetTokenInformation(
            token.as_raw_handle(),
            TokenUser,
            buffer.as_mut_ptr().cast(),
            size_of_val(&buffer) as u32,
            &mut len,
        ) == 0
        {
            return None;
        }
        let user = &*buffer.as_ptr().cast::<TOKEN_USER>();
        let mut sid = null_mut();
        if ConvertSidToStringSidW(user.User.Sid, &mut sid) == 0 || sid.is_null() {
            return None;
        }
        let n = (0..256).take_while(|&i| *sid.add(i) != 0).count();
        let sid_text = String::from_utf16_lossy(std::slice::from_raw_parts(sid, n));
        LocalFree(sid.cast());
        Some(crate::util::wide(&format!(
            "D:P(A;;GA;;;{sid_text})(A;;GA;;;SY)"
        )))
    }
}

fn wide_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain([0]).collect()
}

fn open_private(path: &Path, sddl: &[u16]) -> Option<File> {
    let name = wide_path(path);
    unsafe {
        let mut sd = null_mut();
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut sd,
            null_mut(),
        ) == 0
        {
            return None;
        }
        let sa = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: sd,
            bInheritHandle: 0,
        };
        let handle = CreateFileW(
            name.as_ptr(),
            0xC000_0000,
            FILE_SHARE_READ | FILE_SHARE_DELETE,
            &sa,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        );
        LocalFree(sd);
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let file = File::from_raw_handle(handle);
        let mut flags = 0;
        let mut actual = [0u16; 1024];
        let len = GetFinalPathNameByHandleW(
            handle,
            actual.as_mut_ptr(),
            actual.len() as u32,
            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS,
        ) as usize;
        if len == 0
            || len >= actual.len()
            || GetVolumeInformationByHandleW(
                handle,
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                &mut flags,
                null_mut(),
                0,
            ) == 0
            || flags & FILE_PERSISTENT_ACLS == 0
        {
            return None;
        }
        let resolved = String::from_utf16_lossy(&actual[..len]);
        (local(Path::new(&resolved))
            && path
                .to_str()
                .is_some_and(|name| name.eq_ignore_ascii_case(&resolved)))
        .then_some(file)
    }
}

pub(super) fn init() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = EXE.set(exe);
    }
    EXE_BASE.store(
        unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(null()) } as usize,
        Ordering::Release,
    );
    let event = || unsafe {
        let h = CreateEventW(null(), 0, 0, null());
        (!h.is_null()).then(|| OwnedHandle::from_raw_handle(h))
    };
    let (Some(request), Some(done)) = (event(), event()) else {
        return;
    };
    if REPORTER
        .set(Reporter {
            request,
            done,
            busy: AtomicUsize::new(IDLE),
            job: Mutex::new(None),
            ok: AtomicBool::new(false),
        })
        .is_err()
    {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("xcl-crash-reporter".into())
        .spawn(crash_worker);
}

fn crash_worker() {
    let Some(reporter) = REPORTER.get() else {
        return;
    };
    let (dir, dump_dir, sddl) = crate::vfs_hooks::with_host_io(|| {
        let dir = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .and_then(|p| verified_dir(&p))
            .and_then(|mut dir| {
                for part in ["XCL", "crashes"] {
                    dir.push(part);
                    if !dir.is_dir() {
                        std::fs::create_dir(&dir).ok()?;
                    }
                    dir = verified_dir(&dir)?;
                }
                Some(dir)
            });
        let dump_dir = std::env::var_os("XCL_FULL_DUMP")
            .map(PathBuf::from)
            .and_then(|p| verified_dir(&p));
        (dir, dump_dir, sddl())
    });
    unsafe {
        while WaitForSingleObject(reporter.request.as_raw_handle(), INFINITE) == WAIT_OBJECT_0 {
            let Some((ticket, job)) = reporter.job.lock().ok().and_then(|mut job| job.take())
            else {
                continue;
            };
            let ok = report(job, dir.as_deref(), dump_dir.as_deref(), sddl.as_deref());
            reporter.ok.store(ok, Ordering::Release);
            if reporter
                .busy
                .compare_exchange(
                    ticket | IN_FLIGHT,
                    ticket | SIGNALING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                SetEvent(reporter.done.as_raw_handle());
                let _ = reporter.busy.compare_exchange(
                    ticket | SIGNALING,
                    ticket | DONE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            let _ = reporter.busy.compare_exchange(
                ticket | DETACHED,
                ticket | IDLE,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

fn read<T: Copy>(process: HANDLE, at: usize) -> Option<T> {
    let mut dst = MaybeUninit::<T>::uninit();
    let mut n = 0;
    if at == 0 || at.checked_add(size_of::<T>()).is_none() {
        return None;
    }
    unsafe {
        (ReadProcessMemory(
            process,
            at as *const c_void,
            dst.as_mut_ptr().cast(),
            size_of::<T>(),
            &mut n,
        ) != 0
            && n == size_of::<T>())
        .then(|| dst.assume_init())
    }
}

fn vm(process: HANDLE, at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut m = MEMORY_BASIC_INFORMATION::default();
    (unsafe {
        VirtualQueryEx(
            process,
            at as *const c_void,
            &mut m,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    } != 0)
        .then_some(m)
}

pub(super) fn instruction_bytes(ip: usize) -> Option<[u8; 16]> {
    let process = unsafe { GetCurrentProcess() };
    let m = vm(process, ip)?;
    if m.State != MEM_COMMIT
        || m.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0
        || ip.checked_add(16)? > (m.BaseAddress as usize).saturating_add(m.RegionSize)
    {
        return None;
    }
    read(process, ip)
}

fn printable(s: &str, limit: usize) -> String {
    if s.contains(['\\', '/']) {
        return "unknown".into();
    }
    s.chars()
        .take(limit)
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

fn pe_header(process: HANDLE, base: usize) -> Option<[u8; 192]> {
    let dos = read::<[u8; 64]>(process, base)?;
    let offset = word(&dos, 60) as usize;
    if &dos[..2] != b"MZ" || !(64..=65536).contains(&offset) {
        return None;
    }
    let pe = read::<[u8; 192]>(process, base.checked_add(offset)?)?;
    (&pe[..4] == b"PE\0\0"
        && &pe[24..26] == b"\x0b\x02"
        && u16::from_le_bytes([pe[20], pe[21]]) >= 168)
        .then_some(pe)
}

fn word(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn image(process: HANDLE, base: usize) -> Option<(u32, u32, Option<[u8; 20]>)> {
    let pe = pe_header(process, base)?;
    let (stamp, size) = (word(&pe, 8), word(&pe, 80));
    let cv = (|| {
        let rva = word(&pe, 184);
        let len = word(&pe, 188);
        if word(&pe, 132) < 7 || rva == 0 || rva.checked_add(len)? > size {
            return None;
        }
        for i in 0..(len as usize / 28).min(16) {
            let entry = read::<[u8; 28]>(process, base.checked_add(rva as usize + i * 28)?)?;
            let cv = word(&entry, 20);
            if word(&entry, 12) == 2
                && word(&entry, 16) >= 24
                && cv != 0
                && cv.checked_add(24)? <= size
            {
                let rsds = read::<[u8; 24]>(process, base.checked_add(cv as usize)?)?;
                if &rsds[..4] == b"RSDS" {
                    return rsds[4..].try_into().ok();
                }
            }
        }
        None
    })();
    Some((stamp, size, cv))
}

unsafe extern "system" fn unwind_base(process: HANDLE, pc: u64) -> u64 {
    let Some(m) = vm(process, pc as usize).filter(|m| m.Type == MEM_PRIVATE || m.Type == MEM_IMAGE)
    else {
        return 0;
    };
    let base = m.AllocationBase as usize;
    let Some(pe) = pe_header(process, base) else {
        return 0;
    };
    if pc
        .checked_sub(base as u64)
        .is_some_and(|rva| rva < word(&pe, 80) as u64)
    {
        base as u64
    } else {
        0
    }
}

unsafe extern "system" fn unwind_entry(process: HANDLE, pc: u64) -> *mut c_void {
    let found = (|| {
        let base = unwind_base(process, pc) as usize;
        let pe = pe_header(process, base)?;
        let (at, len, size) = (word(&pe, 160), word(&pe, 164), word(&pe, 80));
        if word(&pe, 132) < 4 || at == 0 || at.checked_add(len)? > size || len % 12 != 0 {
            return None;
        }
        let rva = u32::try_from(pc.checked_sub(base as u64)?).ok()?;
        let (mut lo, mut hi) = (0, len as usize / 12);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry = read::<[u32; 3]>(process, base.checked_add(at as usize + mid * 12)?)?;
            if rva < entry[0] {
                hi = mid;
            } else if rva >= entry[1] {
                lo = mid + 1;
            } else {
                return (entry[1] <= size && entry[2] != 0 && entry[2] < size).then_some(entry);
            }
        }
        None
    })();
    thread_local! {
        static ENTRY: std::cell::UnsafeCell<[u32; 3]> = const { std::cell::UnsafeCell::new([0; 3]) };
    }
    found.map_or(null_mut(), |entry| {
        ENTRY.with(|slot| {
            slot.get().write(entry);
            slot.get().cast()
        })
    })
}

fn identity(process: HANDLE, base: usize) -> String {
    let Some((stamp, size, cv)) = image(process, base) else {
        return "PE identity unknown".into();
    };
    let cv = cv.map_or_else(
        || "PDB identity unknown".into(),
        |cv| {
            let guid = cv[..16]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            let age = u32::from_le_bytes(cv[16..].try_into().unwrap());
            format!("CodeView GUID (raw little-endian bytes) {guid} age {age}")
        },
    );
    format!("PE timestamp {stamp:#010x} image-size {size:#x}; {cv}")
}

fn location(process: HANDLE, ip: usize, live: bool) -> (String, Option<usize>) {
    if live {
        if let Some(base) = crate::modules::image_base_for_addr(ip) {
            return (crate::modules::describe_address(ip), Some(base));
        }
    }
    let Some(m) = vm(process, ip).filter(|m| m.Type == MEM_IMAGE || m.Type == MEM_PRIVATE) else {
        return ("module and RVA unknown".into(), None);
    };
    let base = m.AllocationBase as usize;
    if m.Type == MEM_PRIVATE {
        let rva = ip.saturating_sub(base);
        return match pe_header(process, base) {
            Some(pe) if rva < word(&pe, 80) as usize => {
                (format!("mapped image (name unknown)+{rva:#x}"), Some(base))
            }
            _ => ("module and RVA unknown".into(), None),
        };
    }
    let mut name = [0u16; 512];
    let n = unsafe {
        K32GetMappedFileNameW(
            process,
            ip as *const _,
            name.as_mut_ptr(),
            name.len() as u32,
        )
    } as usize;
    let name = if n == 0 || n >= name.len() {
        "unknown".into()
    } else {
        let path = String::from_utf16_lossy(&name[..n]);
        printable(path.rsplit('\\').next().unwrap_or("unknown"), 100)
    };
    (format!("{name}+{:#x}", ip.saturating_sub(base)), Some(base))
}

fn context(process: HANDLE, ptr: usize) -> Option<(EXCEPTION_RECORD, Option<CONTEXT>)> {
    let p = read::<EXCEPTION_POINTERS>(process, ptr)?;
    let rec = read::<EXCEPTION_RECORD>(process, p.ExceptionRecord as usize)?;
    (rec.ExceptionCode as u32 == AV).then(|| (rec, read(process, p.ContextRecord as usize)))
}

pub(super) fn started() -> bool {
    OWNER.load(Ordering::Acquire) != 0
}

pub(super) fn exe() -> Option<&'static Path> {
    EXE.get().map(PathBuf::as_path)
}

struct Incident;

impl Incident {
    fn claim(thread: u32, fault_tid: u32, ip: usize) -> Result<Self, u32> {
        OWNER.compare_exchange(0, thread, Ordering::AcqRel, Ordering::Acquire)?;
        FAULT_IP.store(ip, Ordering::Relaxed);
        FAULT_TID.store(fault_tid, Ordering::Release);
        Ok(Self)
    }
}

impl Drop for Incident {
    fn drop(&mut self) {
        FAULT_TID.store(0, Ordering::Release);
        OWNER.store(0, Ordering::Release);
    }
}

pub(super) unsafe fn unhandled(
    info: *const EXCEPTION_POINTERS,
    forward: unsafe fn(*const EXCEPTION_POINTERS) -> i32,
) -> Option<i32> {
    let caller = GetCurrentThreadId();
    if OWNER.load(Ordering::Acquire) == caller {
        return Some(0);
    }
    let (rec, ctx) = context(GetCurrentProcess(), info as usize)?;
    let _incident = match Incident::claim(caller, caller, rec.ExceptionAddress as usize) {
        Ok(incident) => incident,
        Err(owner) if owner == caller => return Some(0),
        Err(_) => return Some(forward(info)),
    };
    submit(Job {
        tid: caller,
        rec,
        ctx,
        title: None,
        summary: true,
    });
    Some(forward(info))
}

pub(super) unsafe fn title_dump(file: HANDLE, ty: u32, tid: u32, pointers: usize) -> Option<i32> {
    let process = GetCurrentProcess();
    let (rec, ctx) = context(process, pointers)?;
    if tid == 0 {
        return None;
    }
    let caller = GetCurrentThreadId();
    let ip = rec.ExceptionAddress as usize;
    let incident = match Incident::claim(caller, tid, ip) {
        Ok(incident) => Some(incident),
        Err(_)
            if FAULT_TID.load(Ordering::Acquire) == tid
                && FAULT_IP.load(Ordering::Relaxed) == ip =>
        {
            None
        }
        Err(_) => return Some(0),
    };
    let mut owned = null_mut();
    if file.is_null()
        || DuplicateHandle(
            process,
            file,
            process,
            &mut owned,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        ) == 0
    {
        return Some(0);
    }
    let title = Some((OwnedHandle::from_raw_handle(owned), ty));
    Some(i32::from(submit(Job {
        tid,
        rec,
        ctx,
        title,
        summary: incident.is_some(),
    })))
}

unsafe fn submit(job: Job) -> bool {
    let Some(reporter) = REPORTER.get() else {
        return false;
    };
    let idle = reporter.busy.load(Ordering::Acquire);
    let ticket = idle.wrapping_add(STATE_MASK + 1);
    if idle & STATE_MASK != IDLE
        || reporter
            .busy
            .compare_exchange(
                idle,
                ticket | IN_FLIGHT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
    {
        return false;
    }
    ResetEvent(reporter.done.as_raw_handle());
    let Ok(mut slot) = reporter.job.try_lock() else {
        reporter.busy.store(ticket | IDLE, Ordering::Release);
        return false;
    };
    *slot = Some((ticket, job));
    drop(slot);
    if SetEvent(reporter.request.as_raw_handle()) == 0 {
        reporter.busy.store(ticket | IDLE, Ordering::Release);
        return false;
    }
    if WaitForSingleObject(reporter.done.as_raw_handle(), OWNER_WAIT_MS) == WAIT_OBJECT_0 {
        let ok = reporter.ok.load(Ordering::Acquire);
        reporter.busy.store(ticket | IDLE, Ordering::Release);
        return ok;
    }
    let _ = reporter
        .busy
        .try_update(Ordering::AcqRel, Ordering::Acquire, |state| {
            (state & !STATE_MASK == ticket).then_some(
                ticket
                    | if state & STATE_MASK == DONE {
                        IDLE
                    } else {
                        DETACHED
                    },
            )
        });
    false
}

unsafe fn report(
    job: Job,
    dir: Option<&Path>,
    dump_dir: Option<&Path>,
    sddl: Option<&[u16]>,
) -> bool {
    let Job {
        tid,
        mut rec,
        ctx,
        title,
        summary,
    } = job;
    rec.ExceptionRecord = null_mut();
    let mut ctx = ctx.map(AlignedContext);
    let pointers = EXCEPTION_POINTERS {
        ExceptionRecord: &mut rec,
        ContextRecord: ctx.as_mut().map_or(null_mut(), |c| &mut c.0),
    };
    let ptr = &pointers as *const EXCEPTION_POINTERS as usize;
    let observed = if title.is_some() {
        "title requested a dump for an access violation"
    } else {
        "unhandled access violation"
    };
    let file = if summary {
        match (dir, sddl) {
            (Some(dir), Some(sddl)) => crate::vfs_hooks::with_host_io(|| {
                let final_path = dir.join("last-access-violation.txt");
                let temp = dir.join("pending-access-violation.txt");
                let _ = std::fs::remove_file(&temp);
                let file = open_private(&temp, sddl)?;
                let ctx = ctx.as_ref().map(|c| &c.0);
                if !write_minimal(&file, tid, &rec, ctx, observed) {
                    return None;
                }
                let from = wide_path(&temp);
                let to = wide_path(&final_path);
                MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_REPLACE_EXISTING);
                capture(&file, GetCurrentProcess(), tid, &rec, ctx);
                Some(file)
            }),
            _ => None,
        }
    } else {
        None
    };
    if summary {
        crate::log::access_violation_notice(tid, title.is_none());
    }
    if let Some(file) = file.as_ref().filter(|_| rec.NumberParameters >= 2) {
        let details = crate::alias::crash_ownership(rec.ExceptionInformation[1])
            .unwrap_or_else(|| "unavailable or no record".into());
        let mut details = format!("Guest ownership snapshot (best effort):\n{details}\n");
        let mut out = file;
        if let Ok(end) = out.seek(SeekFrom::End(0)) {
            let cap = (LIMIT / 2).saturating_sub(end) as usize;
            details.truncate(details.floor_char_boundary(cap.min(details.len())));
            let _ = out.write_all(details.as_bytes());
        }
    }
    let full = if title.is_none() {
        dump_dir.zip(sddl).and_then(|(dir, sddl)| {
            let path = dir.join(format!("xcl-small-{}.dmp", GetCurrentProcessId()));
            crate::vfs_hooks::with_host_io(|| open_private(&path, sddl))
        })
    } else {
        None
    };
    let (dump, ty) = title.as_ref().map_or_else(
        || {
            (
                full.as_ref().map_or(null_mut(), |f| f.as_raw_handle()),
                0x40 | 0x800 | 0x1000,
            )
        },
        |(handle, ty)| (handle.as_raw_handle(), *ty),
    );
    if dump.is_null() && file.is_none() {
        return false;
    }
    let handle = file.as_ref().map(|f| f.as_raw_handle());
    let result = super::dump_by_helper(
        dump,
        ty,
        tid,
        ptr,
        if dump.is_null() {
            SYMBOL_WAIT_MS
        } else {
            DUMP_MS
        },
        handle,
    );
    if let Some(mut file) = file {
        if file.seek(SeekFrom::End(0)).is_ok_and(|n| n <= LIMIT - 128) {
            let _ = writeln!(
                file,
                "{} helper: {}",
                if dump.is_null() {
                    "Local unwind/symbol"
                } else {
                    "Local minidump/unwind/symbol"
                },
                result.as_ref().err().map_or("finished", String::as_str)
            );
        }
    }
    result.is_ok()
}

fn access_type(value: usize) -> &'static str {
    match value {
        0 => "read",
        1 => "write",
        8 => "execute",
        _ => "unknown",
    }
}

fn write_minimal(
    file: &File,
    tid: u32,
    rec: &EXCEPTION_RECORD,
    ctx: Option<&CONTEXT>,
    observed: &str,
) -> bool {
    let mut buffer = [0u8; 512];
    let mut record = std::io::Cursor::new(buffer.as_mut_slice());
    let _ = writeln!(
        record,
        "XCL {} | {}\nFailure: {observed} ({AV:#010x}); cause not established\nMinimum record: thread {tid}; fault instruction {:#x}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        rec.ExceptionAddress as usize,
    );
    if rec.NumberParameters >= 2 {
        let op = access_type(rec.ExceptionInformation[0]);
        let dest = rec.ExceptionInformation[1];
        let _ = writeln!(
            record,
            "Windows access: {op}; reported destination {dest:#x}"
        );
    } else {
        let _ = writeln!(record, "Windows access and destination: unknown");
    }
    if let Some(c) = ctx {
        let _ = writeln!(
            record,
            "Original context: rip={:#x} rsp={:#x}",
            c.Rip, c.Rsp
        );
    }
    let _ = writeln!(
        record,
        "Module, symbols and stack are unknown unless captured below.\n"
    );
    let len = record.position() as usize;
    file.seek_write(&buffer[..len], 0).ok() == Some(len)
}

unsafe fn capture(
    file: &File,
    process: HANDLE,
    tid: u32,
    rec: &EXCEPTION_RECORD,
    ctx: Option<&CONTEXT>,
) {
    use std::fmt::Write;
    let mut s = format!(
        "Extended details captured at {:.3}s\n",
        crate::log::uptime()
    );
    let _ = writeln!(
        s,
        "Emulator build: {}",
        identity(process, EXE_BASE.load(Ordering::Acquire))
    );
    let thread = if tid == GetCurrentThreadId() {
        GetCurrentThread()
    } else {
        OpenThread(THREAD_QUERY_INFORMATION, 0, tid)
    };
    let name = if thread.is_null() {
        String::new()
    } else {
        super::thread_name(thread)
    };
    if tid != GetCurrentThreadId() && !thread.is_null() {
        CloseHandle(thread);
    }
    let _ = writeln!(
        s,
        "Faulting thread: {tid} name={}",
        if name.is_empty() {
            "unknown".into()
        } else {
            printable(&name, 80)
        }
    );
    let ip = rec.ExceptionAddress as usize;
    let (loc, base) = location(process, ip, true);
    let _ = writeln!(
        s,
        "Faulting instruction: {ip:#x} {loc}; {}",
        base.map_or_else(
            || "module build unknown".into(),
            |base| identity(process, base)
        )
    );
    let _ = writeln!(
        s,
        "Fault source file/function/line: unknown unless resolved by the local helper below"
    );
    if let Some(bytes) = instruction_bytes(ip) {
        let _ = writeln!(s, "Instruction bytes (local): {}", crate::insn::hex(&bytes));
    }
    if rec.NumberParameters >= 2 {
        let op = access_type(rec.ExceptionInformation[0]);
        let dest = rec.ExceptionInformation[1];
        let _ = writeln!(
            s,
            "Access: {op} (Windows parameter {}); destination: {dest:#x}",
            rec.ExceptionInformation[0]
        );
        if let Some(m) = vm(process, dest) {
            let state = match m.State {
                MEM_COMMIT => "committed",
                MEM_RESERVE => "reserved",
                MEM_FREE => "free",
                _ => "unknown",
            };
            let ty = match m.Type {
                MEM_IMAGE => "image",
                MEM_MAPPED => "mapped",
                MEM_PRIVATE => "private",
                _ => "none/unknown",
            };
            let _ = writeln!(s, "Destination VM: {state} ({:#x}) protect {:#x} allocation-protect {:#x} {ty} ({:#x}) region {:#x}+{:#x} allocation {:#x}", m.State, m.Protect, m.AllocationProtect, m.Type, m.BaseAddress as usize, m.RegionSize, m.AllocationBase as usize);
        } else {
            let _ = writeln!(s, "Destination VM: unknown");
        }
        if dest == usize::MAX {
            if let Some(c) = ctx {
                let hint = super::not_addresses(c);
                if !hint.is_empty() {
                    let _ = writeln!(s, "Destination therefore unconfirmed{hint}");
                }
            }
        }
    } else {
        let _ = writeln!(
            s,
            "Access and destination: unknown (missing exception parameters)"
        );
    }
    if let Some(c) = ctx {
        let _ = writeln!(s, "Original exception context: rip={:#x} rsp={:#x} rbp={:#x} flags={:#x}\n  rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x}\n  r8={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
            c.Rip, c.Rsp, c.Rbp, c.EFlags, c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13, c.R14, c.R15);
        let _ = writeln!(
            s,
            "Fault-context stack (fallback: bounded frame-pointer walk; may be incomplete):"
        );
        let _ = writeln!(
            s,
            "  #0 {:#x} {}",
            c.Rip,
            location(process, c.Rip as usize, true).0
        );
        let mut bp = c.Rbp as usize;
        if let Some(stack) = vm(process, c.Rsp as usize)
            .filter(|m| m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0)
        {
            let limit = (stack.BaseAddress as usize)
                .saturating_add(stack.RegionSize)
                .min((c.Rsp as usize).saturating_add(256 * 1024));
            for i in 1..8 {
                if bp & 7 != 0
                    || bp < c.Rsp as usize
                    || bp.checked_add(16).is_none_or(|end| end > limit)
                {
                    break;
                }
                let Some([prev, ret]) = read::<[usize; 2]>(process, bp) else {
                    break;
                };
                let (loc, base) = location(process, ret, true);
                if base.is_none() {
                    break;
                }
                let _ = writeln!(s, "  #{i} {ret:#x} {loc}");
                if prev <= bp {
                    break;
                }
                bp = prev;
            }
        }
    } else {
        let _ = writeln!(s, "Original registers and stack: unavailable");
    }
    let now = Instant::now();
    for sample in crate::live_stall::snapshot()
        .into_iter()
        .filter(|s| s.stage.blocking())
        .take(4)
    {
        let _ = writeln!(
            s,
            "Coincident stall, not a cause: {} {:.3}s (thread {})",
            sample.stage.label(),
            now.saturating_duration_since(sample.since).as_secs_f32(),
            sample.tid
        );
    }
    let mut out = file;
    if out.seek(SeekFrom::End(0)).is_ok() {
        let _ = out.write_all(s.as_bytes());
    }
}

const SYMBOL_MS: u128 = 4_000;
pub(super) const SYMBOL_WAIT_MS: u32 = 8_000;

fn append(mut file: &File, text: &str) -> bool {
    let Ok(end) = file.seek(SeekFrom::End(0)) else {
        return false;
    };
    let room = LIMIT.saturating_sub(128).saturating_sub(end) as usize;
    let cut = text.floor_char_boundary(room.min(text.len()));
    file.write_all(&text.as_bytes()[..cut]).is_ok() && cut == text.len()
}

/// The helper's part of a report: first the unwind from the original context, each frame
/// written as soon as it is walked and without any symbol lookup; then \`between\` (the
/// minidump, if one was asked for); then an optional symbol pass with its own budget. A helper
/// stopped part way keeps every line it has written.
pub(super) unsafe fn enrich(
    process: HANDLE,
    tid: u32,
    pointers: usize,
    summary: HANDLE,
    between: impl FnOnce() -> bool,
) -> bool {
    let file = File::from_raw_handle(summary);
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().and_then(verified_dir))
        .map(|d| d.to_string_lossy().into_owned())
        .filter(|d| !d.contains([';', '*']));
    SymSetOptions(
        SYMOPT_EXACT_SYMBOLS
            | SYMOPT_DEFERRED_LOADS
            | SYMOPT_LOAD_LINES
            | SYMOPT_NO_PROMPTS
            | SYMOPT_IGNORE_NT_SYMPATH
            | SYMOPT_NO_IMAGE_SEARCH
            | SYMOPT_IGNORE_IMAGEDIR
            | SYMOPT_IGNORE_CVREC
            | SYMOPT_SECURE
            | SYMOPT_DISABLE_SRVSTAR_ON_STARTUP
            | SYMOPT_DISABLE_SYMSRV_AUTODETECT
            | SYMOPT_FAIL_CRITICAL_ERRORS,
    );
    let wide_dir = dir.as_deref().map(crate::util::wide);
    let initialized =
        SymInitializeW(process, wide_dir.as_ref().map_or(null(), |d| d.as_ptr()), 0) != 0;
    let (pc, frames) = unwind(process, tid, pointers, &file, initialized);
    let ok = between();
    if initialized {
        match (dir.is_some(), pc) {
            (true, Some(pc)) => symbols(process, pc, &frames, &file),
            (false, Some(_)) => {
                append(&file, "Symbol pass: skipped (the executable's folder is not a verified local folder)\n");
            }
            _ => {}
        }
        let _ = SymCleanup(process);
    }
    ok
}

unsafe fn unwind(
    process: HANDLE,
    tid: u32,
    pointers: usize,
    file: &File,
    initialized: bool,
) -> (Option<u64>, Vec<u64>) {
    let mut frames = Vec::new();
    let Some((rec, Some(ctx))) = context(process, pointers) else {
        append(
            file,
            "\nFault-context unwind: the original context could not be read by the helper\n",
        );
        return (None, frames);
    };
    let pc = Some(rec.ExceptionAddress as u64);
    let mut out = file;
    if out
        .seek(SeekFrom::End(0))
        .map_or(true, |end| end > LIMIT / 2)
    {
        append(
            file,
            "\nFault-context unwind: skipped (report already full)\n",
        );
        return (pc, frames);
    }
    if !initialized {
        append(
            file,
            "\nFault-context unwind: skipped (DbgHelp could not be initialized)\n",
        );
        return (pc, frames);
    }
    let thread = OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT, 0, tid);
    if thread.is_null() {
        append(
            file,
            "\nFault-context unwind: skipped (the faulting thread could not be opened)\n",
        );
        return (pc, frames);
    }
    let mut ctx = AlignedContext(ctx);
    let flat = |offset| ADDRESS64 {
        Offset: offset,
        Segment: 0,
        Mode: AddrModeFlat,
    };
    let mut frame = STACKFRAME64 {
        AddrPC: flat(ctx.0.Rip),
        AddrStack: flat(ctx.0.Rsp),
        AddrFrame: flat(ctx.0.Rbp),
        ..STACKFRAME64::default()
    };
    let mut prev = (0, 0);
    let initial_sp = ctx.0.Rsp;
    append(file, "\nFault-context stack (Windows unwind from the original context, no symbols, written per frame):\n");
    let mut end = "frame limit (12) reached";
    for i in 0..12 {
        let (ip, sp) = (frame.AddrPC.Offset, frame.AddrStack.Offset);
        if ip == 0 || (ip, sp) == prev || sp < prev.1 || sp.saturating_sub(initial_sp) > 256 * 1024
        {
            end = "no further valid frame";
            break;
        }
        prev = (ip, sp);
        frames.push(ip);
        if !append(
            file,
            &format!(
                "  #{i} {ip:#x} {}\n",
                location(process, ip as usize, false).0
            ),
        ) {
            end = "report full";
            break;
        }
        if unwind_base(process, ip) == 0 {
            end = "frame outside a known image";
            break;
        }
        if StackWalk64(
            0x8664,
            process,
            thread,
            &mut frame,
            (&mut ctx.0 as *mut CONTEXT).cast(),
            None,
            Some(unwind_entry),
            Some(unwind_base),
            None,
        ) == 0
        {
            end = "Windows unwind stopped";
            break;
        }
    }
    CloseHandle(thread);
    append(
        file,
        &format!("Unwind pass ended: {end} after {} frame(s)\n", frames.len()),
    );
    (pc, frames)
}

unsafe fn symbols(process: HANDLE, pc: u64, frames: &[u64], file: &File) {
    let mut modules = [null_mut(); 128];
    let mut bytes = 0;
    if K32EnumProcessModules(
        process,
        modules.as_mut_ptr(),
        size_of::<HANDLE>() as u32 * modules.len() as u32,
        &mut bytes,
    ) != 0
    {
        for &m in modules
            .iter()
            .take((bytes as usize / size_of::<HANDLE>()).min(modules.len()))
        {
            let mut path = [0u16; 1024];
            let n =
                K32GetModuleFileNameExW(process, m, path.as_mut_ptr(), path.len() as u32) as usize;
            if n == 0 || n >= path.len() {
                continue;
            }
            let name = String::from_utf16_lossy(&path[..n]);
            if !local(Path::new(&name)) {
                continue;
            }
            SymLoadModuleExW(
                process,
                null_mut(),
                path.as_ptr(),
                null(),
                m as u64,
                0,
                null(),
                0,
            );
        }
    }
    let start = Instant::now();
    append(
        file,
        &format!(
            "Symbol pass (optional, matching local PDBs only, {} s budget):\n",
            SYMBOL_MS / 1000
        ),
    );
    let mut end = "done";
    for (label, ip) in std::iter::once(("fault".to_string(), pc)).chain(
        frames
            .iter()
            .enumerate()
            .map(|(i, &ip)| (format!("#{i}"), ip)),
    ) {
        if start.elapsed().as_millis() > SYMBOL_MS {
            end = "budget used";
            break;
        }
        if !append(file, &format!("  {label} {}\n", symbol(process, ip))) {
            end = "report full";
            break;
        }
    }
    append(
        file,
        &format!(
            "Symbol pass ended: {end} after {:.1} s\n",
            start.elapsed().as_secs_f32()
        ),
    );
}

unsafe fn symbol(process: HANDLE, ip: u64) -> String {
    #[repr(C)]
    struct Symbol {
        h: SYMBOL_INFOW,
        tail: [u16; 256],
    }
    let mut buf: Symbol = std::mem::zeroed();
    buf.h.SizeOfStruct = size_of::<SYMBOL_INFOW>() as u32;
    buf.h.MaxNameLen = 256;
    let mut disp = 0;
    if SymFromAddrW(process, ip, &mut disp, &mut buf.h) == 0 {
        return "function/file/line unknown".into();
    }
    let mut m = IMAGEHLP_MODULE64 {
        SizeOfStruct: size_of::<IMAGEHLP_MODULE64>() as u32,
        ..IMAGEHLP_MODULE64::default()
    };
    if SymGetModuleInfo64(process, ip, &mut m) == 0
        || (m.SymType != SymPdb && m.SymType != SymDia)
        || m.PdbUnmatched != 0
    {
        return "function/file/line unknown".into();
    }
    let Some((_, _, Some(cv))) = image(process, m.BaseOfImage as usize) else {
        return "function/file/line unknown".into();
    };
    let key = m.PdbSig70;
    if cv[..4] != key.data1.to_le_bytes()
        || cv[4..6] != key.data2.to_le_bytes()
        || cv[6..8] != key.data3.to_le_bytes()
        || cv[8..16] != key.data4
        || cv[16..] != m.PdbAge.to_le_bytes()
    {
        return "function/file/line unknown".into();
    }
    let name = std::slice::from_raw_parts(
        std::ptr::addr_of!(buf.h.Name).cast::<u16>(),
        (buf.h.NameLen as usize).min(256),
    );
    let name = printable(&String::from_utf16_lossy(name), 128);
    let mut line = IMAGEHLP_LINEW64 {
        SizeOfStruct: size_of::<IMAGEHLP_LINEW64>() as u32,
        ..IMAGEHLP_LINEW64::default()
    };
    let mut line_disp = 0;
    if SymGetLineFromAddrW64(process, ip, &mut line_disp, &mut line) == 0 || line.FileName.is_null()
    {
        return format!("{name} + {disp:#x}; file/line unknown");
    }
    let n = (0..1024)
        .take_while(|&i| *line.FileName.add(i) != 0)
        .count();
    let path = String::from_utf16_lossy(std::slice::from_raw_parts(line.FileName, n));
    let basename = path.rsplit(['\\', '/']).next().unwrap_or("unknown");
    format!(
        "{name} + {disp:#x}; {}:{}",
        printable(basename, 100),
        line.LineNumber
    )
}
