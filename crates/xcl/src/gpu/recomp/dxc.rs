//! In-process HLSL → DXIL compilation with DXC.
//!
//! `dxcompiler.dll` is not part of Windows, so it is loaded at run time from
//! `%XCL_DXC_DIR%`, the folder of xcl.exe, or the Windows SDK with the newest DXC
//! (`Windows Kits\10\bin\<ver>\x64`). `dxil.dll` from the same folder is loaded first so
//! the compiler can sign the DXIL (unsigned DXIL is rejected by D3D12).
//!
//! Which DXC it is matters ([`Toolchain::outdated`]), so the one that was loaded is logged and
//! kept for `report.json` ([`toolchain`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::{Interface, GUID, HRESULT, PCWSTR};
use windows::Win32::Graphics::Direct3D::Dxc::{DxcBuffer, IDxcBlob, IDxcCompiler3, IDxcResult, IDxcUtils, IDxcValidator, CLSID_DxcCompiler, CLSID_DxcUtils, CLSID_DxcValidator, DXC_CP_ACP, DXC_CP_UTF8};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;
/// DxcCreateInstance2: the same with the `IMalloc` the instance allocates through.
type CreateInstance2 = unsafe extern "system" fn(*mut c_void, *const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

/// A dotted version, `10.0.26100.0` or `1.9.2602.17`, as four numbers (missing ones are 0).
fn parse_version(s: &str) -> Option<[u32; 4]> {
    let mut v = [0u32; 4];
    let mut parts = s.split('.');
    for x in v.iter_mut() {
        match parts.next() {
            Some(p) => *x = p.parse().ok()?,
            None => break,
        }
    }
    parts.next().is_none().then_some(v)
}

/// The version resource of a file.
fn file_version(path: &Path) -> Option<[u32; 4]> {
    use windows_sys::Win32::Storage::FileSystem::{GetFileVersionInfoSizeW, GetFileVersionInfoW, VerQueryValueW, VS_FIXEDFILEINFO};
    let name = crate::util::wide(&path.to_string_lossy());
    unsafe {
        let n = GetFileVersionInfoSizeW(name.as_ptr(), std::ptr::null_mut());
        if n == 0 {
            return None;
        }
        let mut data = vec![0u8; n as usize];
        if GetFileVersionInfoW(name.as_ptr(), 0, n, data.as_mut_ptr() as *mut c_void) == 0 {
            return None;
        }
        let (mut p, mut len) = (std::ptr::null_mut::<c_void>(), 0u32);
        if VerQueryValueW(data.as_ptr() as *const c_void, crate::util::wide("\\").as_ptr(), &mut p, &mut len) == 0 || (len as usize) < std::mem::size_of::<VS_FIXEDFILEINFO>() {
            return None;
        }
        let f = std::ptr::read_unaligned(p as *const VS_FIXEDFILEINFO);
        Some([f.dwFileVersionMS >> 16, f.dwFileVersionMS & 0xFFFF, f.dwFileVersionLS >> 16, f.dwFileVersionLS & 0xFFFF])
    }
}

/// The DXC release a `dxcompiler.dll` with this version resource is, for comparing folders:
/// `1.7.2308.16` is itself; the old SDKs' compilers carry the number of their Windows build
/// (`10.0.19041.x`, DXC 1.5 and before), so those, and files that say nothing, come last.
fn dxc_release(file: Option<[u32; 4]>) -> [u32; 4] {
    match file {
        Some(v) if v[0] < 10 => v,
        _ => [0; 4],
    }
}

/// Windows SDK folders (name, version resource of the `dxcompiler.dll` in its `x64`), the one
/// to try first in front: the newest DXC, and between equal ones the newest SDK. The names are
/// compared as versions (`10.0.100000.0` is newer than `10.0.26100.0`), other names last.
fn order_sdk_folders(mut v: Vec<(String, Option<[u32; 4]>)>) -> Vec<String> {
    v.sort_by(|a, b| (dxc_release(b.1), parse_version(&b.0), &b.0).cmp(&(dxc_release(a.1), parse_version(&a.0), &a.0)));
    v.into_iter().map(|(name, _)| name).collect()
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("XCL_DXC_DIR") {
        v.push(PathBuf::from(d));
    }
    if let Some(d) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        v.push(d);
    }
    for root in [r"C:\Program Files (x86)\Windows Kits\10\bin", r"C:\Program Files\Windows Kits\10\bin"] {
        if let Ok(rd) = std::fs::read_dir(root) {
            let vers = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir());
            let vers = vers.map(|p| (p.file_name().unwrap_or_default().to_string_lossy().into_owned(), file_version(&p.join("x64").join("dxcompiler.dll"))));
            for ver in order_sdk_folders(vers.collect()) {
                v.push(Path::new(root).join(ver).join("x64"));
            }
        }
    }
    v
}

/// One of DXC's two libraries, as it identifies itself.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Library {
    /// the file's version resource: `1.9.2602.17` is the GitHub release 1.9.2602. The Windows
    /// SDK's `dxil.dll` says `101.7.2308.24`, and SDKs before 10.0.22000 the number of their
    /// Windows build
    pub file_version: String,
    /// `IDxcVersionInfo::GetVersion`: the newest DXIL the library knows, which is not always
    /// its release (1.8.2505 says 1.9). 0 and 0: the library did not say
    pub major: u32,
    pub minor: u32,
    /// `IDxcVersionInfo2::GetCommitInfo`: the commit DXC was built from
    pub commit: String,
}

impl Library {
    /// The file's version when it is a DXC release number.
    fn release(&self) -> Option<[u32; 4]> {
        parse_version(&self.file_version).filter(|v| (1..10).contains(&v[0]))
    }

    /// The release, (1, 9) for 1.9.2602: from the file, else what the library said.
    pub fn version(&self) -> Option<(u32, u32)> {
        match self.release() {
            Some(v) => Some((v[0], v[1])),
            None => ((self.major, self.minor) != (0, 0)).then_some((self.major, self.minor)),
        }
    }
}

impl std::fmt::Display for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.release(), self.version(), self.file_version.is_empty()) {
            (Some(_), ..) => write!(f, "{}", self.file_version)?,
            (None, Some((major, minor)), true) => write!(f, "{major}.{minor}")?,
            (None, Some((major, minor)), false) => write!(f, "{major}.{minor} (file {})", self.file_version)?,
            (None, None, false) => write!(f, "(file {})", self.file_version)?,
            (None, None, true) => write!(f, "of unknown version")?,
        }
        if !self.commit.is_empty() {
            write!(f, " ({})", self.commit)?;
        }
        Ok(())
    }
}

/// The DXC this process compiles with (`dxc` in `report.json`).
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Toolchain {
    /// the folder the libraries came from; empty while DXC is not loaded
    pub dir: String,
    pub dxcompiler: Library,
    /// the signer from the same folder; empty when the folder has none
    pub dxil: Library,
}

/// The oldest DXC release XCL recommends (see [`Toolchain::outdated`]).
pub const RECOMMENDED: (u32, u32) = (1, 8);

impl Toolchain {
    /// Whether the compiler is older than [`RECOMMENDED`]. Measured with 1.7.2308 (what Windows
    /// SDK 10.0.26100.4654 installs; other revisions of that SDK have 1.8.2502) against
    /// 1.9.2602 on an RTX 5070 Ti, synthetic translated compute programs (issue #156): DXIL
    /// 2.1–2.7 times the size, pipeline creation in the driver 1.4–2.2 times as long. 1.7 also
    /// rejects `[WaveSize]` on the entries of a library, which 1.8 accepts; the sizes of 1.8's
    /// DXIL were not measured.
    pub fn outdated(&self) -> bool {
        self.dxcompiler.version().is_some_and(|v| v < RECOMMENDED)
    }
}

/// Why an outdated DXC is worth replacing, and how.
pub const OUTDATED_COST: &str = "DXC 1.7 made DXIL 2–2.7 times the size of DXC 1.9's from translated compute programs, and an NVIDIA driver took 1.4–2.2 times as long to build pipelines from it";
pub const OUTDATED_FIX: &str = r"download DXC 1.8 or newer from github.com/microsoft/DirectXShaderCompiler/releases and set XCL_DXC_DIR to its bin\x64 folder";

/// Ask a DXC library for its version. Nothing here fails the load: what the library does not
/// answer stays empty.
unsafe fn library(create: CreateInstance, class: &GUID, file: &Path) -> Library {
    use windows::Win32::Graphics::Direct3D::Dxc::{IDxcVersionInfo, IDxcVersionInfo2};
    let mut lib = Library { file_version: file_version(file).map(|v| format!("{}.{}.{}.{}", v[0], v[1], v[2], v[3])).unwrap_or_default(), ..Default::default() };
    let mut p = std::ptr::null_mut();
    if create(class, &IDxcVersionInfo::IID, &mut p).is_err() || p.is_null() {
        return lib;
    }
    let info = IDxcVersionInfo::from_raw(p);
    let _ = info.GetVersion(&mut lib.major, &mut lib.minor);
    if let Ok(info2) = info.cast::<IDxcVersionInfo2>() {
        let (mut count, mut hash) = (0u32, std::ptr::null_mut::<i8>());
        if info2.GetCommitInfo(&mut count, &mut hash).is_ok() && !hash.is_null() {
            lib.commit = std::ffi::CStr::from_ptr(hash).to_string_lossy().into_owned();
            windows::Win32::System::Com::CoTaskMemFree(Some(hash as *const c_void));
        }
    }
    lib
}

static LIB: OnceLock<Result<(PathBuf, usize), String>> = OnceLock::new();
static TOOLCHAIN: OnceLock<Toolchain> = OnceLock::new();

/// The DXC in use; empty while none is loaded (asking does not load it).
pub fn toolchain() -> Toolchain {
    TOOLCHAIN.get().cloned().unwrap_or_default()
}

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn versions_parse_and_compare_as_numbers() {
        assert_eq!(parse_version("10.0.26100.0"), Some([10, 0, 26100, 0]));
        assert_eq!(parse_version("1.9.2602.17"), Some([1, 9, 2602, 17]));
        assert_eq!(parse_version("1.8"), Some([1, 8, 0, 0]));
        for bad in ["", "x64", "1.8.", "1.x", "1.2.3.4.5", "-1.0"] {
            assert_eq!(parse_version(bad), None, "{bad:?}");
        }
        assert!(parse_version("10.0.100000.0") > parse_version("10.0.26100.0"));
        assert!(parse_version("1.10") > parse_version("1.9.2602.17"));
    }

    #[test]
    fn the_sdk_with_the_newest_dxc_is_tried_first() {
        let sdk = |name: &str, dxc: Option<[u32; 4]>| (name.to_string(), dxc);
        let order = order_sdk_folders(vec![
            sdk("arm64", None),
            sdk("10.0.19041.0", Some([10, 0, 19041, 5609])), // DXC 1.5 under Windows' own number
            sdk("10.0.22621.0", Some([1, 6, 2112, 16])),
            sdk("10.0.26100.0", Some([1, 7, 2308, 16])),
            sdk("10.0.100000.0", Some([1, 7, 2308, 16])), // sorts below 10.0.2… as a name
            sdk("10.0.22000.0", Some([1, 8, 2505, 32])), // a newer DXC copied into an older SDK
            sdk("10.0.17134.0", None),
            sdk("x64", None),
        ]);
        assert_eq!(order, ["10.0.22000.0", "10.0.100000.0", "10.0.26100.0", "10.0.22621.0", "10.0.19041.0", "10.0.17134.0", "x64", "arm64"]);
    }

    #[test]
    fn dxc_below_1_8_is_outdated() {
        let lib = |major, minor, file: &str| Library { major, minor, file_version: file.into(), commit: String::new() };
        let outdated = |l: Library| Toolchain { dir: r"C:\dxc".into(), dxcompiler: l, dxil: Library::default() }.outdated();
        // the four compilers of the note in the pull request, as they answer
        assert!(outdated(lib(1, 7, "1.7.2308.16")));
        assert!(outdated(lib(1, 5, "10.0.19041.5609")));
        assert!(!outdated(lib(1, 9, "1.8.2505.32")), "the release counts, not the DXIL version it knows");
        assert!(!outdated(lib(1, 9, "1.9.2602.17")));
        assert!(outdated(lib(1, 9, "1.7.2308.16")));
        // a file that says nothing of the release: what the library says; neither: no verdict
        assert!(outdated(lib(1, 6, "")));
        assert!(!outdated(lib(1, 10, "")));
        assert!(!outdated(lib(2, 0, "101.0.1.2")));
        assert!(outdated(lib(0, 0, "1.6.2112.16")));
        assert!(!outdated(lib(0, 0, "10.0.19041.5609")));
        assert!(!outdated(lib(0, 0, "")));
        assert!(!Toolchain::default().outdated());

        assert_eq!(lib(1, 9, "1.8.2505.32").to_string(), "1.8.2505.32");
        assert_eq!(lib(1, 5, "10.0.19041.5609").to_string(), "1.5 (file 10.0.19041.5609)");
        assert_eq!(lib(1, 7, "101.7.2308.24").to_string(), "1.7 (file 101.7.2308.24)");
        assert_eq!(Library { commit: "21d28f72".into(), ..lib(1, 9, "1.9.2602.17") }.to_string(), "1.9.2602.17 (21d28f72)");
        assert_eq!(lib(1, 9, "").to_string(), "1.9");
        assert_eq!(lib(0, 0, "10.0.1.2").to_string(), "(file 10.0.1.2)");
    }

    /// The DXC of this PC, when it has one: what `report.json` says of it.
    #[test]
    fn the_loaded_dxc_is_identified() {
        let Ok(dir) = location() else {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        };
        let t = toolchain();
        assert_eq!(Path::new(&t.dir), dir);
        let major = t.dxcompiler.major;
        assert!(major >= 1, "IDxcVersionInfo of dxcompiler.dll: {t:?}");
        assert!(!t.dxcompiler.file_version.is_empty(), "{t:?}");
        assert_eq!(Some(t.outdated()), t.dxcompiler.version().map(|v| v < RECOMMENDED));
        if dir.join("dxil.dll").is_file() {
            assert!(t.dxil.version().is_some(), "{t:?}");
        }
        // as the `dxc` of a report, and a report from before the field reads as "not loaded"
        let report = crate::report::Report { dxc: t.clone(), ..Default::default() };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["dxc"]["dir"], t.dir);
        assert_eq!(json["dxc"]["dxcompiler"]["major"], major);
        let mut old = json.clone();
        old.as_object_mut().unwrap().remove("dxc");
        assert_eq!(serde_json::from_value::<crate::report::Report>(old).unwrap().dxc, Toolchain::default());
        println!("DXC: dxcompiler {}, dxil {}, from {}", t.dxcompiler, t.dxil, t.dir);
    }
}

/// Folder with dxcompiler.dll (+ dxil.dll) that will be used.
pub fn location() -> Result<&'static Path, String> {
    load().map(|(p, _)| p.as_path())
}

fn load() -> Result<&'static (PathBuf, usize), String> {
    LIB.get_or_init(|| crate::vfs_hooks::with_host_io(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        for dir in candidates() {
            let dxc = dir.join("dxcompiler.dll");
            if !dxc.is_file() {
                continue;
            }
            unsafe {
                let dxil = dir.join("dxil.dll");
                let signer = if dxil.is_file() { LoadLibraryW(crate::util::wide(&dxil.to_string_lossy()).as_ptr()) } else { std::ptr::null_mut() };
                let h = LoadLibraryW(crate::util::wide(&dxc.to_string_lossy()).as_ptr());
                if h.is_null() {
                    continue;
                }
                if let Some(f) = GetProcAddress(h, b"DxcCreateInstance\0".as_ptr()) {
                    let mut t = Toolchain { dir: dir.display().to_string(), ..Default::default() };
                    t.dxcompiler = library(std::mem::transmute::<usize, CreateInstance>(f as usize), &CLSID_DxcCompiler, &dxc);
                    if let Some(g) = if signer.is_null() { None } else { GetProcAddress(signer, b"DxcCreateInstance\0".as_ptr()) } {
                        t.dxil = library(std::mem::transmute::<usize, CreateInstance>(g as usize), &CLSID_DxcValidator, &dxil);
                        let _ = VALIDATOR_CREATE.set(g as usize);
                    }
                    let signer = if signer.is_null() { "no dxil.dll".to_string() } else { format!("dxil {}", t.dxil) };
                    crate::xlog_info!("DXC: dxcompiler {}, {signer}, from {}", t.dxcompiler, t.dir);
                    if t.outdated() {
                        crate::xlog_warn!("DXC {} is older than {}.{}: {OUTDATED_COST}; {OUTDATED_FIX}", t.dxcompiler, RECOMMENDED.0, RECOMMENDED.1);
                    }
                    let _ = TOOLCHAIN.set(t);
                    return Ok((dir, f as usize));
                }
            }
        }
        Err("dxcompiler.dll not found (set XCL_DXC_DIR or install the Windows SDK)".into())
    }))
    .as_ref()
    .map_err(|e| e.clone())
}

/// The PC compiler's DxcCreateInstance (for titles that use the console's `dxcompiler_xs.dll`).
pub fn create_instance() -> Option<usize> {
    load().ok().map(|&(_, f)| f)
}

fn create_fn() -> Result<CreateInstance, String> {
    let &(_, f) = load()?;
    Ok(unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

/// The DxcCreateInstance of the dxil.dll that was loaded beside dxcompiler.dll, when there is one.
static VALIDATOR_CREATE: OnceLock<usize> = OnceLock::new();

thread_local! {
    static COMPILER: RefCell<Option<IDxcCompiler3>> = const { RefCell::new(None) };
    static VALIDATOR: RefCell<Option<(IDxcValidator, IDxcUtils)>> = const { RefCell::new(None) };
}

fn validator() -> Result<(IDxcValidator, IDxcUtils), String> {
    VALIDATOR.with(|v| {
        if let Some(x) = v.borrow().as_ref() {
            return Ok(x.clone());
        }
        let utils_create = create_fn()?;
        let validator_create = *VALIDATOR_CREATE.get().ok_or("no dxil.dll beside dxcompiler.dll to validate with")?;
        let validator_create = unsafe { std::mem::transmute::<usize, CreateInstance>(validator_create) };
        let (mut pv, mut pu) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr = unsafe { validator_create(&CLSID_DxcValidator, &IDxcValidator::IID, &mut pv) };
        if hr.is_err() || pv.is_null() {
            return Err(format!("dxil.dll DxcCreateInstance: {hr:?}"));
        }
        let validator = unsafe { IDxcValidator::from_raw(pv) };
        let hr = unsafe { utils_create(&CLSID_DxcUtils, &IDxcUtils::IID, &mut pu) };
        if hr.is_err() || pu.is_null() {
            return Err(format!("DxcCreateInstance(DxcUtils): {hr:?}"));
        }
        let x = (validator, unsafe { IDxcUtils::from_raw(pu) });
        *v.borrow_mut() = Some(x.clone());
        Ok(x)
    })
}

/// What dxil.dll's validator says of a container: what DXC runs on every container it makes,
/// here run on the direct backend's before anything uses them.
fn validate(dxil: &[u8]) -> Result<(), String> {
    let (validator, utils) = validator()?;
    let size = u32::try_from(dxil.len()).map_err(|_| "container larger than 4 GiB".to_string())?;
    unsafe {
        let blob = utils.CreateBlobFromPinned(dxil.as_ptr() as *const c_void, size, DXC_CP_ACP).map_err(|e| format!("CreateBlobFromPinned: {e}"))?;
        let blob: IDxcBlob = blob.cast().map_err(|e| format!("{e}"))?;
        let r = validator.Validate(&blob, 0).map_err(|e| format!("Validate: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_ok() {
            return Ok(());
        }
        let msg = r
            .GetErrorBuffer()
            .ok()
            .map(|b| {
                let s = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
                String::from_utf8_lossy(s).trim_end_matches('\0').trim().to_string()
            })
            .unwrap_or_default();
        Err(if msg.is_empty() { format!("{status:?}") } else { msg })
    }
}

/// The DxcCreateInstance2 of the dxcompiler.dll in use (None: it has none).
fn create2_fn() -> Option<CreateInstance2> {
    static F: OnceLock<Option<usize>> = OnceLock::new();
    let f = (*F.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleExW, GetProcAddress, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT};
        let &(_, create) = load().ok()?;
        unsafe {
            let mut module = std::mem::zeroed();
            if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, create as *const u16, &mut module) == 0 {
                return None;
            }
            GetProcAddress(module, b"DxcCreateInstance2\0".as_ptr()).map(|g| g as usize)
        }
    }))?;
    Some(unsafe { std::mem::transmute::<usize, CreateInstance2>(f) })
}

/// A compile worker's DXC allocates in a heap of the worker's own (`crate::dxc_heap`), which
/// is destroyed when the worker has been idle: for how long. None: DXC allocates in the
/// process heap, as it did (`XCL_DXC_PRIVATE_HEAP=0`, or a dxcompiler.dll without
/// DxcCreateInstance2). `XCL_DXC_HEAP_IDLE_S`: the idle seconds (10); `XCL_DXC_HEAP_CHECK=1`:
/// every block is looked up in its heap before it is freed (counts blocks that are another
/// allocator's). A heap has no limit: DXC does not survive an allocation that fails.
fn own_heap() -> Option<std::time::Duration> {
    static SET: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *SET.get_or_init(|| {
        if std::env::var("XCL_DXC_PRIVATE_HEAP").is_ok_and(|v| v == "0") {
            return None;
        }
        if create2_fn().is_none() {
            crate::xlog_info!("DXC: this dxcompiler.dll has no DxcCreateInstance2: its memory stays in the process heap");
            return None;
        }
        let idle = std::env::var("XCL_DXC_HEAP_IDLE_S").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(10).max(1);
        let check = std::env::var("XCL_DXC_HEAP_CHECK").is_ok_and(|v| v == "1");
        crate::dxc_heap::check(check);
        crate::xlog_info!(
            "DXC: each compile worker's compiler allocates in a heap of its own, destroyed when the worker has been idle for {idle} s and DXC left nothing in it (XCL_DXC_HEAP_IDLE_S; XCL_DXC_PRIVATE_HEAP=0: the process heap, as before){}",
            if check { "; every block is looked up before it is freed (XCL_DXC_HEAP_CHECK)" } else { "" }
        );
        Some(std::time::Duration::from_secs(idle))
    })
}

thread_local! {
    /// whether this thread is a compile worker of the pool, and whether its heap stays for
    /// good (DXC left something in it: `worker_idle`)
    static WORKER: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((false, false)) };
    /// a compile worker's heap, from its first compile until it has been idle
    static HEAP: RefCell<Option<crate::dxc_heap::WorkerHeap>> = const { RefCell::new(None) };
}

/// How long a compile worker waits for a job before it gives its heap up (None: it has none
/// to give up).
fn worker_wait() -> Option<std::time::Duration> {
    if WORKER.get().1 || !HEAP.with(|h| h.borrow().is_some()) {
        return None;
    }
    own_heap()
}

/// A compile worker has been idle: its compiler is released and its heap destroyed, which
/// gives back all that DXC took for its compiles. A heap DXC left something in stays, for
/// this worker's further compiles.
fn worker_idle() {
    static ENDED: AtomicU64 = AtomicU64::new(0);
    COMPILER.with(|c| c.borrow_mut().take());
    let Some(heap) = HEAP.with(|h| h.borrow_mut().take()) else { return };
    match heap.end() {
        Ok(peak) => {
            let n = ENDED.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                let (made, ended, bytes) = crate::dxc_heap::counts();
                crate::xlog_info!("DXC: an idle compile worker's heap is destroyed: DXC had {} MB allocated in it at most ({n} so far; {} heaps alive with {} MB allocated)", peak >> 20, made.saturating_sub(ended), bytes >> 20);
            }
        }
        Err(heap) => {
            let (blocks, bytes) = heap.live();
            crate::xlog_warn!(
                "DXC: an idle compile worker's heap stays: with its compiler released DXC still has {blocks} blocks ({} KB) in it and {} references to its allocator, so this worker keeps it ({} blocks of another allocator were handed to the heaps: counted with XCL_DXC_HEAP_CHECK=1)",
                bytes >> 10,
                heap.held(),
                crate::dxc_heap::foreign()
            );
            HEAP.with(|h| *h.borrow_mut() = Some(heap));
            WORKER.set((true, true));
        }
    }
}

fn compiler() -> Result<IDxcCompiler3, String> {
    COMPILER.with(|c| {
        if let Some(x) = c.borrow().as_ref() {
            return Ok(x.clone());
        }
        let create = create_fn()?;
        let mut p = std::ptr::null_mut();
        // (a pool worker: through the allocator of its own heap)
        let own = if WORKER.get().0 { own_heap() } else { None };
        let malloc = own.and_then(|_| {
            HEAP.with(|h| {
                let mut h = h.borrow_mut();
                if h.is_none() {
                    *h = crate::dxc_heap::WorkerHeap::new();
                }
                h.as_ref().map(|heap| heap.imalloc())
            })
        });
        let hr = match (malloc, create2_fn()) {
            (Some(malloc), Some(create2)) => unsafe { create2(malloc, &CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut p) },
            _ => unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut p) },
        };
        if hr.is_err() || p.is_null() {
            return Err(format!("DxcCreateInstance: {hr:?}"));
        }
        let x = unsafe { IDxcCompiler3::from_raw(p) };
        *c.borrow_mut() = Some(x.clone());
        Ok(x)
    })
}

/// Stack of the thread DXC runs on: its LLVM passes recurse along long dependency chains of
/// big translated programs (2 MB overflowed on a 7 000-instruction retail PS); the caller may
/// be a game thread with a small stack. Reserved, committed only as used.
const DXC_STACK: usize = 64 << 20;

struct Job {
    src: String,
    entry: String,
    profile: String,
    extra: Vec<String>,
    reply: std::sync::mpsc::Sender<Result<Vec<u8>, String>>,
}

thread_local! {
    static PROGRAM: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run `f` with `label` (a console program's hash) naming every shader this thread has compiled
/// in it: the direct backend's fallback lines carry it, so a log says which program DXC got.
pub fn for_program<R>(label: &str, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRAM.with(|p| *p.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(PROGRAM.with(|p| p.replace(Some(label.to_string()))));
    f()
}

pub fn compile_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let requested = std::env::var("XCL_DXC_THREADS").ok().and_then(|v| v.parse().ok());
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let available = (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some((m.ullAvailPhys, m.ullAvailPageFile));
        compile_thread_limit(cores, requested, available, RESERVE_GB.load(std::sync::atomic::Ordering::Relaxed))
    })
}

fn compile_thread_limit(cores: usize, requested: Option<usize>, available: Option<(u64, u64)>, reserve_gb: u64) -> usize {
    if let Some(n) = requested {
        return n.clamp(1, 64);
    }
    let memory_limit = available.map_or(1, |(physical, commit)| {
        (physical.min(commit).saturating_sub(reserve_gb.saturating_mul(1 << 30)) >> 30).clamp(1, 64) as usize
    });
    cores.saturating_sub(1).clamp(1, 64).min(memory_limit)
}

#[cfg(test)]
mod worker_tests {
    use super::compile_thread_limit;

    #[test]
    fn default_workers_fit_physical_memory_and_commit() {
        for (physical, commit, want) in [(20, 40, 14), (12, 40, 6), (8, 40, 2), (40, 8, 2), (6, 40, 1), (0, 0, 1)] {
            assert_eq!(compile_thread_limit(32, None, Some((physical << 30, commit << 30)), 6), want);
        }
        assert_eq!(compile_thread_limit(32, None, Some(((8 << 30) - 1, 40 << 30)), 6), 1);
        assert_eq!(compile_thread_limit(32, None, Some((20 << 30, 40 << 30)), 2), 18);
    }

    #[test]
    fn default_workers_leave_a_cpu_free_and_handle_missing_memory() {
        let plenty = Some((128 << 30, 128 << 30));
        assert_eq!(compile_thread_limit(16, None, plenty, 6), 15);
        assert_eq!(compile_thread_limit(1, None, plenty, 6), 1);
        assert_eq!(compile_thread_limit(128, None, plenty, 6), 64);
        assert_eq!(compile_thread_limit(32, None, None, 6), 1);
    }

    #[test]
    fn explicit_workers_keep_the_existing_override_and_bounds() {
        for (requested, want) in [(0, 1), (4, 4), (64, 64), (128, 64)] {
            assert_eq!(compile_thread_limit(1, Some(requested), Some((0, 0)), 6), want);
            assert_eq!(compile_thread_limit(1, Some(requested), None, 6), want);
        }
    }
}

/// GB of commit `wait_for_memory` keeps free besides the compiles: 6 for a running title (its
/// own allocations), less without one (`set_memory_reserve`: `xcl shaders --precompile`).
static RESERVE_GB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(6);

/// The commit `wait_for_memory` keeps free besides the compiles (no title running: less).
pub fn set_memory_reserve(gb: u64) {
    RESERVE_GB.store(gb, std::sync::atomic::Ordering::Relaxed);
}

/// Compiles running on the DXC workers (`wait_for_memory` lets them in).
static WORKING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A compile let in by `wait_for_memory`, until it is dropped.
pub struct MemorySlot;

impl Drop for MemorySlot {
    fn drop(&mut self) {
        WORKING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Before a compile: with 6 GB of commit left or more, plus 1 GB for every compile already
/// running (each can still grow to ~0.7 GB), at once; with less, one compile at a
/// time – it goes when no other is running (and after a minute in any case). Each compile used
/// to wait the minute while commit was short: with XDefiant's own 10 GB committed, 4 GB were
/// left, and its last cold-cache shaders came a few a minute.
pub fn wait_for_memory() -> MemorySlot {
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let start = std::time::Instant::now();
    loop {
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let need = (RESERVE_GB.load(Ordering::Relaxed) << 30) + ((WORKING.load(Ordering::Acquire) as u64) << 30);
        let plenty = unsafe { GlobalMemoryStatusEx(&mut m) } == 0 || m.ullAvailPageFile >= need;
        if plenty || start.elapsed().as_secs() >= 60 {
            if start.elapsed().as_secs() >= 2 {
                crate::xlog_info!("DXC: a compile waited {:.1} s for memory ({} MB of page file left)", start.elapsed().as_secs_f32(), m.ullAvailPageFile >> 20);
            }
            WORKING.fetch_add(1, Ordering::AcqRel);
            return MemorySlot;
        }
        if WORKING.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            if start.elapsed().as_secs() >= 2 {
                crate::xlog_info!("DXC: a compile waited {:.1} s for memory, then ran alone ({} MB of page file left)", start.elapsed().as_secs_f32(), m.ullAvailPageFile >> 20);
            }
            return MemorySlot;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// DXC workers (`compile_threads`, each with its own compiler and a `DXC_STACK` stack), started
/// on first use; callers wait for their job.
/// The compile threads' work: a title thread's compile (it waits in its own call, a pipeline it
/// creates now) goes before the background preparation of pipelines for later
/// (`pso_async::preparing`) – in one queue, thousands of those held the title's few behind them
/// and its loading stood still while a cold cache recompiled.
struct Queue {
    title: std::collections::VecDeque<Job>,
    background: std::collections::VecDeque<Job>,
}

static QUEUE: std::sync::Mutex<Queue> = std::sync::Mutex::new(Queue { title: std::collections::VecDeque::new(), background: std::collections::VecDeque::new() });
static QUEUED: std::sync::Condvar = std::sync::Condvar::new();

/// The next job. `idle`: how long to wait for one at most (None: until there is one); None is
/// returned when none came.
fn next_job(idle: Option<std::time::Duration>) -> Option<Job> {
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    let start = std::time::Instant::now();
    loop {
        if let Some(j) = q.title.pop_front().or_else(|| q.background.pop_front()) {
            return Some(j);
        }
        q = match idle {
            None => QUEUED.wait(q).unwrap_or_else(|e| e.into_inner()),
            Some(idle) => {
                let left = idle.checked_sub(start.elapsed()).filter(|left| !left.is_zero())?;
                QUEUED.wait_timeout(q, left).unwrap_or_else(|e| e.into_inner()).0
            }
        };
    }
}

fn pool() -> bool {
    static POOL: OnceLock<bool> = OnceLock::new();
    *POOL.get_or_init(|| {
        let mut started = false;
        for k in 0..compile_threads() {
            started |= std::thread::Builder::new().name(format!("xcl-dxc-{k}")).stack_size(DXC_STACK).spawn(move || loop {
                WORKER.set((true, WORKER.get().1));
                let Some(job) = next_job(worker_wait()) else {
                    worker_idle();
                    continue;
                };
                let _slot = wait_for_memory();
                let extra: Vec<&str> = job.extra.iter().map(|s| s.as_str()).collect();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_with_dxc(&job.src, &job.entry, &job.profile, &extra)))
                    .unwrap_or_else(|_| Err("DXC: panic in the compile thread".into()));
                let _ = job.reply.send(r);
            }).is_ok();
        }
        started
    })
}

/// Compile HLSL to signed DXIL. `profile` e.g. "cs_6_6"; `extra` are dxc arguments.
pub fn compile(src: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    submit(src, entry, profile, extra, false)
}

static DXC_RETRIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Compile with DXC whatever `XCL_SHADER_BACKEND` says, and put the result in the DXIL cache in
/// place of the direct backend's container: for a program whose container the driver refused,
/// so the retry and every later run get DXC's. Logged and counted (`stage_summary`). Caches
/// that keep DXIL elsewhere (in memory, by program) are the caller's to refresh.
pub fn compile_dxc_only(src: &str, entry: &str, profile: &str) -> Result<Vec<u8>, String> {
    let n = DXC_RETRIES.fetch_add(1, Ordering::Relaxed) + 1;
    let program = PROGRAM.with(|p| p.borrow().clone()).map_or(String::new(), |p| format!(" of {p}"));
    crate::xlog_warn!("[SHADER] DXC recompiles a {profile} program{program} in place of the direct DXIL backend's container ({} B HLSL; {n} so far)", src.len());
    let dxil = submit(src, entry, profile, &[], true)?;
    let dir = normalized_cache_dir(&cache_dir());
    let file = dir.join(format!("{:016x}.dxil", cache_key(src, entry, profile)));
    let _process_guard = process_cache_lock(&file)?;
    if std::fs::create_dir_all(&dir).is_ok() {
        let tmp = file.with_extension(format!("tmp{}", std::process::id()));
        // rename does not replace an existing file on Windows
        let _ = std::fs::remove_file(&file);
        if std::fs::write(&tmp, &dxil).is_ok() && std::fs::rename(&tmp, &file).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(dxil)
}

/// The direct backend first, on the caller's thread (it takes milliseconds to a second, and
/// in the DXC queue it waited behind DXC's minutes-long compiles), then DXC on its workers for
/// what the direct backend did not make. DXC arguments (an optimisation level, a define) ask
/// for DXC itself, and so does `dxc_only`.
fn submit(src: &str, entry: &str, profile: &str, extra: &[&str], dxc_only: bool) -> Result<Vec<u8>, String> {
    // the direct backend inlines every call, so a program with outlined functions
    // (`emit::outlined`) is DXC's, which keeps them; with only the direct backend it is inlined
    let outlined = backend() == Backend::Direct && extra.is_empty() && !dxc_only && src.contains("[noinline] ");
    if outlined {
        static OUTLINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = OUTLINED.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            let program = PROGRAM.with(|p| p.borrow().clone()).map_or(String::new(), |p| format!(" of {p}"));
            crate::xlog_info!("[SHADER] DXC compiles a {profile} program{program} with outlined memory functions ({} B HLSL; {n} so far)", src.len());
        }
    }
    if backend() != Backend::Dxc && extra.is_empty() && !dxc_only && !outlined {
        let direct = {
            let _slot = wait_for_memory();
            std::thread::scope(|s| match std::thread::Builder::new().name("xcl-dxil".into()).stack_size(DXC_STACK).spawn_scoped(s, || compile_direct(src, entry, profile)) {
                Ok(t) => t.join().unwrap_or_else(|_| Err("error: panic in the backend thread".into())),
                Err(e) => Err(format!("error: no thread for the backend ({e})")),
            })
        };
        match direct {
            Ok(dxil) => return Ok(dxil),
            Err(why) if backend() == Backend::DirectOnly => return Err(format!("xcl-dxil: {why}")),
            Err(why) => {
                let c = direct_counts();
                // every fallback is logged: a warm-up's log is the proof of what the backend covers
                let program = PROGRAM.with(|p| p.borrow().clone()).map_or(String::new(), |p| format!(" of {p}"));
                crate::xlog_warn!(
                    "[SHADER] direct DXIL backend: {why}; DXC compiles this {profile} program{program} (fallback {}: {} unsupported, {} failed, {} invalid; {} direct)",
                    c.fallbacks(),
                    c.unsupported,
                    c.failed,
                    c.invalid,
                    c.compiled
                );
            }
        }
    }
    let (reply, rx) = std::sync::mpsc::channel();
    let job = Job { src: src.into(), entry: entry.into(), profile: profile.into(), extra: extra.iter().map(|s| s.to_string()).collect(), reply };
    if !pool() {
        return Err("DXC: no compile threads".to_string());
    }
    {
        let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        if crate::hle::d3d12x::pso_async::preparing() {
            q.background.push_back(job);
        } else if crate::hle::d3d12x::pso_async::first_build() {
            q.title.push_front(job);
        } else {
            q.title.push_back(job);
        }
    }
    QUEUED.notify_one();
    rx.recv().map_err(|_| "DXC: the compile thread exited".to_string())?
}

/// `XCL_DXC_OPT=0|1|2|3`: the level of every compile that names none.
fn opt_override() -> Option<&'static str> {
    static O: OnceLock<Option<&'static str>> = OnceLock::new();
    *O.get_or_init(|| match std::env::var("XCL_DXC_OPT").ok()?.trim() {
        "0" => Some("-O0"),
        "1" => Some("-O1"),
        "2" => Some("-O2"),
        "3" => Some("-O3"),
        other => {
            crate::xlog_warn!("XCL_DXC_OPT={other:?} is not 0, 1, 2 or 3: ignored");
            None
        }
    })
}

/// The level DXC compiles at when the caller names none: `XCL_DXC_OPT`, else -O1 whatever the
/// size (a retail game's 12.8 MB FSR program: 6.5 MB of DXIL at -O1 in 8 min and 2.2 GB; -O0
/// makes about twice the DXIL, more than the driver can compile within memory).
/// DXC's optimiser was superlinear on translated programs full of short-circuit `?:` / `&&` /
/// `if` per instruction (a 1589-instruction console CS: ~6 min and > 30 GB at -O3); with
/// branch-free emission (select / and / or) it is fast: that shader 0.3 s, a 15 720-instruction
/// one 6 s (585 KB DXIL vs 1.6 MB at -O0, which the driver could not compile).
/// -O1 against -O3 on 40 of a retail game's programs across their sizes: DXC 115.7 s
/// against 116.5 s, the same DXIL (5416 KB against 5402 KB); -O0 70.6 s, but twice the DXIL.
fn opt_level() -> &'static str {
    opt_override().unwrap_or("-O1")
}

/// What chooses the level (`XCL_DXC_OPT`, or the default), for caches that keep DXIL by
/// something other than its HLSL (`pso::translated`).
pub fn opt_policy() -> &'static str {
    opt_override().unwrap_or("default")
}

/// IEEE strictness: without it every float op in the DXIL is `fast`, and DXC reassociates the
/// console's separately rounded adds and subs (`(b + c) - a` as `(b - a) + c`) wherever it
/// can prove two lane selects share a condition. No measurable cost in DXC time.
const STRICT: &str = "-Gis";

/// No source locations in the IR: only DXC's late diagnostics use them, the DXIL's code is the
/// same (its disassembly differs only in the shader hash), and DXC is 5 – 13% faster on big
/// translated programs.
const NO_LOC: &str = "-fdisable-loc-tracking";

/// Which compiler turns translated HLSL into DXIL. `XCL_SHADER_BACKEND=dxil` (the default) is
/// the direct backend (`xcl_dxil`: linear time, no DXC): dxil.dll validates each container it
/// makes before anything uses it, and DXC compiles what the backend does not support, fails on or
/// makes invalid, each such program logged and counted. `hlsl` is DXC for every program;
/// `dxil-only` (tests) has no fallback, so a run shows what the direct backend alone does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Dxc,
    Direct,
    DirectOnly,
}

/// The direct DXIL backend makes the programs it can (`XCL_SHADER_BACKEND` is not "hlsl"): a
/// program without outlined functions takes it milliseconds to a second, whatever its size.
pub fn direct_backend() -> bool {
    backend() != Backend::Dxc
}

fn backend() -> Backend {
    static B: OnceLock<Backend> = OnceLock::new();
    *B.get_or_init(|| match std::env::var("XCL_SHADER_BACKEND").unwrap_or_default().trim() {
        "" | "dxil" => Backend::Direct,
        "dxil-only" => Backend::DirectOnly,
        "hlsl" => Backend::Dxc,
        other => {
            crate::xlog_warn!("XCL_SHADER_BACKEND={other:?} is not dxil, hlsl or dxil-only: using dxil");
            Backend::Direct
        }
    })
}

/// What the direct DXIL backend did with the programs of this process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCounts {
    /// compiled, and dxil.dll found the container valid
    pub compiled: usize,
    /// a construct the backend does not handle: DXC compiled it
    pub unsupported: usize,
    /// an error or a panic in the backend: DXC compiled it
    pub failed: usize,
    /// dxil.dll rejected the container, or could not be asked: DXC compiled it
    pub invalid: usize,
}

impl DirectCounts {
    pub fn fallbacks(&self) -> usize {
        self.unsupported + self.failed + self.invalid
    }
}

static DIRECT: [std::sync::atomic::AtomicUsize; 4] = [const { std::sync::atomic::AtomicUsize::new(0) }; 4];

pub fn direct_counts() -> DirectCounts {
    let n = |k: usize| DIRECT[k].load(Ordering::Relaxed);
    DirectCounts { compiled: n(0), unsupported: n(1), failed: n(2), invalid: n(3) }
}

/// The direct backend on one program: its validated container, or why DXC has to compile it (a
/// panic in the backend is an error like any other). A container dxil.dll does not pass is never
/// returned. `XCL_DXIL_BACKEND_LOG=<file>`: one line per program (FNV-1a 64 of the source,
/// profile, outcome, seconds compiling, source bytes, seconds validating).
fn compile_direct(src: &str, entry: &str, profile: &str) -> Result<Vec<u8>, String> {
    let t0 = std::time::Instant::now();
    let compiled = std::panic::catch_unwind(|| xcl_dxil::compile(src, entry, profile));
    let secs = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let r = match compiled {
        Ok(Ok(dxil)) => validate(&dxil).map(|()| dxil).map_err(|e| (3, format!("invalid: {e}"))),
        Ok(Err(xcl_dxil::Error::Unsupported(what))) => Err((1, format!("unsupported: {what}"))),
        Ok(Err(e)) => Err((2, format!("error: {e}"))),
        Err(_) => Err((2, "error: panic in the backend".to_string())),
    };
    let validate_secs = t1.elapsed().as_secs_f64();
    DIRECT[r.as_ref().map_or_else(|(slot, _)| *slot, |_| 0)].fetch_add(1, Ordering::Relaxed);
    if r.is_ok() && std::env::var_os("XCL_DXIL_BACKEND_LOG").is_none() {
        return r.map_err(|(_, why)| why);
    }
    // what names the program here is its HLSL: the hash the fallback warning and the log carry
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in src.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let outcome = r.as_ref().map_or_else(|(_, why)| why.as_str(), |_| "direct");
    if let Some(log) = std::env::var_os("XCL_DXIL_BACKEND_LOG") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
            let _ = f.write_all(format!("{h:016x}\t{profile}\t{}\t{secs:.4}\t{}\t{validate_secs:.4}\n", outcome.replace(['\n', '\t'], " "), src.len()).as_bytes());
        }
    }
    r.map_err(|(_, why)| format!("{} (HLSL {h:016x}, {} bytes)", why.lines().next().unwrap_or_default(), src.len()))
}

fn compile_with_dxc(src: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    let c = compiler()?;
    let mut args: Vec<Vec<u16>> = ["-E", entry, "-T", profile, "-HV", "2021", STRICT, NO_LOC].iter().map(|a| crate::util::wide(a)).collect();
    if !extra.iter().any(|a| a.starts_with("-O")) {
        args.push(crate::util::wide(opt_level()));
    }
    args.extend(extra.iter().map(|a| crate::util::wide(a)));
    let ptrs: Vec<PCWSTR> = args.iter().map(|a| PCWSTR(a.as_ptr())).collect();
    let buf = DxcBuffer { Ptr: src.as_ptr() as *const c_void, Size: src.len(), Encoding: DXC_CP_UTF8.0 };
    unsafe {
        let r: IDxcResult = c.Compile(&buf, Some(&ptrs), None).map_err(|e| format!("Compile: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_err() {
            let msg = r
                .GetErrorBuffer()
                .ok()
                .map(|b| {
                    let s = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
                    String::from_utf8_lossy(s).trim_end_matches('\0').to_string()
                })
                .unwrap_or_default();
            return Err(format!("dxc: {msg}"));
        }
        let blob: IDxcBlob = r.GetResult().map_err(|e| format!("{e}"))?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec())
    }
}

pub(super) fn toolchain_identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        // DXIL of the direct backend and of DXC never share a cache entry, nor do two builds of
        // the direct backend (its id is a hash of its sources), nor its validated containers and
        // those of builds that did not validate them
        let direct = if backend() == Backend::Dxc { String::new() } else { format!("xcl-dxil {} validated", xcl_dxil::BUILD_ID) };
        let Ok(dir) = location() else {
            return format!("unavailable{direct}");
        };
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for byte in direct.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for name in ["dxcompiler.dll", "dxil.dll"] {
            for byte in name.bytes().chain([0]) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            let Ok(metadata) = crate::vfs_hooks::with_host_io(|| std::fs::metadata(dir.join(name))) else {
                h ^= 0xff;
                h = h.wrapping_mul(0x100_0000_01b3);
                continue;
            };
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |duration| duration.as_nanos());
            for byte in metadata.len().to_le_bytes().into_iter().chain(modified.to_le_bytes()) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        format!("{h:016x}")
    })
}

/// FNV-1a 64 of the inputs that determine the DXIL.
fn cache_key_for_toolchain(src: &str, entry: &str, profile: &str, opt: &str, toolchain: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in [env!("CARGO_PKG_VERSION"), toolchain, profile, entry, opt, STRICT, NO_LOC, src] {
        for b in part.bytes().chain([0u8]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn cache_key(src: &str, entry: &str, profile: &str) -> u64 {
    cache_key_for_toolchain(src, entry, profile, opt_level(), toolchain_identity())
}

/// Folder of the DXIL cache: `XCL_DXIL_CACHE`, else `%LOCALAPPDATA%\xcl\dxil`.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("XCL_DXIL_CACHE") {
        return PathBuf::from(d);
    }
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    base.join("xcl").join("dxil")
}

/// DXIL cache statistics for this process.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CacheStats {
    /// Immediate cache hits in memory or on disk, including cached-only requests.
    pub hits: u64,
    /// Initial misses, excluding cached-only requests.
    pub misses: u64,
    /// Misses filled by another caller; included in `misses`.
    pub coalesced_hits: u64,
    /// Cached-only misses; no compilation attempted.
    pub deferred_misses: u64,
    /// Compilation attempts, including failures.
    pub compilations: u64,
    pub compilation_failures: u64,
    /// First 20 distinct compiler errors.
    pub compilation_errors: Vec<String>,
}

#[derive(Default)]
struct CacheCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    coalesced_hits: AtomicU64,
    deferred_misses: AtomicU64,
    compilations: AtomicU64,
    compilation_failures: AtomicU64,
    compilation_errors: Mutex<Vec<String>>,
}

impl CacheCounters {
    fn snapshot(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            coalesced_hits: self.coalesced_hits.load(Ordering::Relaxed),
            deferred_misses: self.deferred_misses.load(Ordering::Relaxed),
            compilations: self.compilations.load(Ordering::Relaxed),
            compilation_failures: self.compilation_failures.load(Ordering::Relaxed),
            compilation_errors: self.compilation_errors.lock().unwrap().clone(),
        }
    }

    fn failure(&self, error: String) -> String {
        self.compilation_failures.fetch_add(1, Ordering::Relaxed);
        let mut errors = self.compilation_errors.lock().unwrap();
        if errors.len() < 20 && !errors.contains(&error) {
            errors.push(error.clone());
        }
        error
    }
}

fn cache_counters() -> &'static CacheCounters {
    static COUNTERS: OnceLock<CacheCounters> = OnceLock::new();
    COUNTERS.get_or_init(CacheCounters::default)
}

pub fn cache_stats() -> CacheStats {
    cache_counters().snapshot()
}

pub(super) fn note_memory_hit() {
    cache_counters().hits.fetch_add(1, Ordering::Relaxed);
}

/// What boot spent per stage: translations taken from disk and made, DXC compile time
/// (summed over threads), pipelines the driver built and its time on them.
static TRANSLATED_HITS: AtomicU64 = AtomicU64::new(0);
static TRANSLATIONS: AtomicU64 = AtomicU64::new(0);
static DXC_US: AtomicU64 = AtomicU64::new(0);
static DRIVER_BUILDS: AtomicU64 = AtomicU64::new(0);
static DRIVER_US: AtomicU64 = AtomicU64::new(0);

pub(super) fn note_translation(from_disk: bool) {
    (if from_disk { &TRANSLATED_HITS } else { &TRANSLATIONS }).fetch_add(1, Ordering::Relaxed);
}

/// Wave64 compute programs translated for a GPU with 32-lane waves: those every lane of which
/// computes on its own (`lane_independent`), and the others.
static LANE_INDEPENDENT: AtomicU64 = AtomicU64::new(0);
static LANE_WAVE_FORM: AtomicU64 = AtomicU64::new(0);

pub(super) fn note_lane_census(qualifies: bool) {
    (if qualifies { &LANE_INDEPENDENT } else { &LANE_WAVE_FORM }).fetch_add(1, Ordering::Relaxed);
}

/// (qualifying, other) programs of `note_lane_census`.
pub fn lane_census() -> (u64, u64) {
    (LANE_INDEPENDENT.load(Ordering::Relaxed), LANE_WAVE_FORM.load(Ordering::Relaxed))
}

/// A pipeline the driver built (or refused) in `t`.
pub fn note_driver(t: std::time::Duration) {
    DRIVER_BUILDS.fetch_add(1, Ordering::Relaxed);
    DRIVER_US.fetch_add(t.as_micros() as u64, Ordering::Relaxed);
}

/// One line: each stage's cache hits and misses and the time its misses took.
pub fn stage_summary() -> String {
    let c = cache_counters();
    let s = |us: &AtomicU64| us.load(Ordering::Relaxed) as f64 / 1e6;
    let direct = match (backend(), direct_counts()) {
        (Backend::Dxc, _) => String::new(),
        (_, c) => format!(
            "; direct DXIL backend {} compiled and validated, {} left to DXC ({} unsupported, {} failed, {} invalid), {} recompiled by DXC after the driver refused them",
            c.compiled,
            c.fallbacks(),
            c.unsupported,
            c.failed,
            c.invalid,
            DXC_RETRIES.load(Ordering::Relaxed)
        ),
    };
    // always there and without a digit of its own: `warm_children::merge_summaries` sums the
    // numbers of lines that read the same
    let (yes, no) = lane_census();
    let lanes = format!("; lane-independent compute {yes}, wave form {no}");
    format!(
        "translation {} from disk, {} made; DXC {} hits, {} compiled in {:.0} thread-s; driver {} pipelines in {:.0} thread-s{direct}{lanes}",
        TRANSLATED_HITS.load(Ordering::Relaxed),
        TRANSLATIONS.load(Ordering::Relaxed),
        c.hits.load(Ordering::Relaxed),
        c.compilations.load(Ordering::Relaxed),
        s(&DXC_US),
        DRIVER_BUILDS.load(Ordering::Relaxed),
        s(&DRIVER_US)
    )
}

/// `compile` through a disk cache in `dir` (big translated programs take seconds in DXC;
/// the same console shader comes back every run). Returns (DXIL, from cache).
pub fn compile_cached_in(dir: &Path, src: &str, entry: &str, profile: &str) -> Result<(Vec<u8>, bool), String> {
    compile_cached_in_with(dir, src, entry, profile, || compile(src, entry, profile, &[]))
}

type CacheFlight = (PathBuf, u64);

fn cache_flight(key: CacheFlight) -> Arc<Mutex<()>> {
    static FLIGHTS: OnceLock<Mutex<HashMap<CacheFlight, Weak<Mutex<()>>>>> = OnceLock::new();
    let mut flights = FLIGHTS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    flights.retain(|_, v| v.strong_count() != 0);
    if let Some(flight) = flights.get(&key).and_then(Weak::upgrade) {
        return flight;
    }
    let flight = Arc::new(Mutex::new(()));
    flights.insert(key, Arc::downgrade(&flight));
    flight
}

struct ProcessCacheLock {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

impl Drop for ProcessCacheLock {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::System::Threading::ReleaseMutex(self.handle);
            windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

fn cache_mutex_name(file: &Path) -> String {
    let absolute = file
        .parent()
        .and_then(|parent| file.file_name().map(|name| normalized_cache_dir(parent).join(name)))
        .unwrap_or_else(|| std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf()));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in absolute.to_string_lossy().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!(r"Local\XclDxilCache-{h:016x}")
}

pub(crate) fn normalized_cache_dir(dir: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(dir) {
        return canonical;
    }
    let absolute = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    for ancestor in absolute.ancestors().skip(1) {
        if let Ok(canonical) = std::fs::canonicalize(ancestor) {
            if let Ok(remainder) = absolute.strip_prefix(ancestor) {
                return canonical.join(remainder);
            }
        }
    }
    absolute
}

fn process_cache_lock(file: &Path) -> Result<ProcessCacheLock, String> {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_ABANDONED, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{CreateMutexW, WaitForSingleObject, INFINITE};
    let name = crate::util::wide(&cache_mutex_name(file));
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() {
        return Err("DXIL cache: cannot create the cross-process mutex".into());
    }
    let wait = unsafe { WaitForSingleObject(handle, INFINITE) };
    if wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED {
        unsafe { CloseHandle(handle) };
        return Err(format!("DXIL cache: waiting for the cross-process mutex failed ({wait:#x})"));
    }
    Ok(ProcessCacheLock { handle })
}

fn cached_file(file: &Path) -> Option<Vec<u8>> {
    if let Ok(b) = std::fs::read(&file) {
        if valid_dxil_container(&b) {
            return Some(b);
        }
    }
    None
}

/// Cheap validation before handing a cached blob to D3D12. DXIL uses the DXBC
/// container header: magic, hash, version, total size, chunk count and offsets.
fn valid_dxil_container(blob: &[u8]) -> bool {
    let rd = |at: usize| {
        blob.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    if blob.len() < 36 || blob.get(..4) != Some(b"DXBC") || rd(20) != Some(1) || rd(24) != Some(blob.len()) {
        return false;
    }
    let Some(count) = rd(28) else { return false };
    if count == 0 || count > 64 {
        return false;
    }
    let Some(table_end) = 32usize.checked_add(count * 4) else { return false };
    if table_end > blob.len() {
        return false;
    }
    let mut ranges = Vec::with_capacity(count);
    let mut has_dxil = false;
    for i in 0..count {
        let Some(off) = rd(32 + i * 4) else { return false };
        if off < table_end || off & 3 != 0 {
            return false;
        }
        let Some(size) = rd(off + 4) else { return false };
        let Some(end) = off.checked_add(8).and_then(|v| v.checked_add(size)) else { return false };
        if end > blob.len() || ranges.iter().any(|&(a, b)| off < b && a < end) {
            return false;
        }
        if blob.get(off..off + 4) == Some(b"DXIL") {
            // DxilProgramHeader is six dwords before the bitcode payload.
            if size < 24 || blob.get(off + 16..off + 20) != Some(b"DXIL") {
                return false;
            }
            has_dxil = true;
        }
        ranges.push((off, end));
    }
    has_dxil
}

fn compile_cached_in_with<F>(dir: &Path, src: &str, entry: &str, profile: &str, compile_one: F) -> Result<(Vec<u8>, bool), String>
where
    F: FnOnce() -> Result<Vec<u8>, String>,
{
    compile_cached_in_observed(dir, src, entry, profile, None, cache_counters(), compile_one)
}

fn compile_cached_in_observed<F>(
    dir: &Path,
    src: &str,
    entry: &str,
    profile: &str,
    opt: Option<&str>,
    counters: &CacheCounters,
    compile_one: F,
) -> Result<(Vec<u8>, bool), String>
where
    F: FnOnce() -> Result<Vec<u8>, String>,
{
    let opt = opt.unwrap_or_else(|| opt_level());
    let key = cache_key_for_toolchain(src, entry, profile, opt, toolchain_identity());
    let dir = normalized_cache_dir(dir);
    let file = dir.join(format!("{key:016x}.dxil"));
    if let Some(b) = cached_file(&file) {
        counters.hits.fetch_add(1, Ordering::Relaxed);
        return Ok((b, true));
    }
    // before the locks: a cached-only attempt (non-blocking pipeline creation) must not wait
    // for a worker compiling the same shader
    if CACHED_ONLY.with(|c| c.get()) {
        counters.deferred_misses.fetch_add(1, Ordering::Relaxed);
        MISSED.with(|m| m.set(true));
        return Err("DXIL not compiled yet".into());
    }
    counters.misses.fetch_add(1, Ordering::Relaxed);
    let flight = cache_flight((dir.clone(), key));
    let _guard = flight.lock().unwrap_or_else(|e| e.into_inner());
    let _process_guard = process_cache_lock(&file)?;
    // Another caller may have populated the cache while this one waited.
    if let Some(b) = cached_file(&file) {
        counters.coalesced_hits.fetch_add(1, Ordering::Relaxed);
        return Ok((b, true));
    }
    // A corrupt target must be removed before the atomic rename on Windows;
    // rename does not replace an existing file there.
    if file.is_file() {
        let _ = std::fs::remove_file(&file);
    }
    counters.compilations.fetch_add(1, Ordering::Relaxed);
    let b = {
        let _active = Compiling::start();
        let t0 = std::time::Instant::now();
        let b = compile_one();
        DXC_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        if t0.elapsed().as_secs() >= 5 {
            let t = std::thread::current();
            crate::xlog_info!("DXC: {profile} {opt} {key:016x} ({} B HLSL) took {:.1} s, asked by thread {}", src.len(), t0.elapsed().as_secs_f32(), t.name().map_or_else(|| format!("{:?}", t.id()), str::to_string));
        }
        b
    }
    .and_then(|b| {
        if valid_dxil_container(&b) {
            Ok(b)
        } else {
            Err("DXIL cache: compiler returned an invalid DXIL container".into())
        }
    })
    .map_err(|e| {
        counters.failure(format!(
            "DXIL cache {} ({profile}, entry {entry:?}, {} B HLSL): {e}",
            file.display(),
            src.len()
        ))
    })?;
    if std::fs::create_dir_all(&dir).is_ok() {
        store_dxil(&dir, &file, &b);
    }
    Ok((b, false))
}

/// Whether DXIL files with the same bytes share them (`store_dxil`): `XCL_DXIL_DEDUP=0`, each
/// file its own, as before.
fn dedup() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DXIL_DEDUP").map_or(true, |v| v.trim() != "0"))
}

/// The file in cache folder `dir` that keeps DXIL `b` by its bytes (`store_dxil`).
fn content_file(dir: &Path, b: &[u8]) -> PathBuf {
    dir.join("content").join(format!("{:016x}-{:x}.dxil", crate::gpu::shader::fnv64(b), b.len()))
}

/// Keep DXIL `b` as cache file `file` in folder `dir`, under another name first: another thread
/// or process may be reading. A file kept already with the same bytes (`content_file`) gives
/// them: the new file is another name of it (a hard link), so a module that many programs
/// compile to is on the disk once – of a retail game's 181,133 cache files 69,871 were copies of
/// others, 3.9 of 30 GB. A first one is kept under its bytes' name as well. Readers see a file as
/// before, and an old build reads them all. Where a link cannot be made (another file system, a
/// file with as many names as NTFS allows) the file gets its own bytes.
fn store_dxil(dir: &Path, file: &Path, b: &[u8]) {
    let tmp = file.with_extension(format!("tmp{}", std::process::id()));
    let mut done = false;
    if dedup() {
        let content = content_file(dir, b);
        if std::fs::read(&content).is_ok_and(|kept| kept == b) {
            done = std::fs::hard_link(&content, &tmp).is_ok();
            if done {
                DEDUPED.fetch_add(1, Ordering::Relaxed);
            }
        } else if std::fs::write(&tmp, b).is_ok() {
            done = true;
            if !content.exists() && content.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok()) {
                let _ = std::fs::hard_link(&tmp, &content);
            }
        }
    }
    if !done && std::fs::write(&tmp, b).is_err() {
        return;
    }
    if std::fs::rename(&tmp, file).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// DXIL files this process kept as another name of one with the same bytes (`store_dxil`).
static DEDUPED: AtomicU64 = AtomicU64::new(0);

/// (DXIL files kept as a hard link of the same bytes, this process.)
pub fn deduplicated() -> u64 {
    DEDUPED.load(Ordering::Relaxed)
}

/// DXC compiles of programs the disk cache did not have: (running now, finished) – the XCL
/// window shows them (`present::show_compiling`), whatever path the pipeline took (built in
/// the background, or in the title's call when it could not get a stand-in).
static COMPILES: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

struct Compiling;

impl Compiling {
    fn start() -> Compiling {
        COMPILES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Compiling
    }
}

impl Drop for Compiling {
    fn drop(&mut self) {
        COMPILES[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        COMPILES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// (DXC compiles running now, finished so far) – cache misses only.
pub fn compile_counts() -> (usize, usize) {
    (COMPILES[0].load(std::sync::atomic::Ordering::Relaxed), COMPILES[1].load(std::sync::atomic::Ordering::Relaxed))
}

thread_local! {
    static CACHED_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MISSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` without compiling: a program missing from the disk cache fails at once instead
/// (non-blocking pipeline creation). Returns `f`'s result and whether something was missing.
pub fn cached_only<R>(f: impl FnOnce() -> R) -> (R, bool) {
    CACHED_ONLY.with(|c| c.set(true));
    MISSED.with(|m| m.set(false));
    let r = f();
    CACHED_ONLY.with(|c| c.set(false));
    (r, MISSED.with(|m| m.replace(false)))
}

/// Inside `cached_only`: what is being made is deferred as if its DXIL were missing (the
/// driver's compile of a big program is the slow part then). False outside `cached_only`.
pub fn defer() -> bool {
    let inside = CACHED_ONLY.with(|c| c.get());
    if inside {
        MISSED.with(|m| m.set(true));
    }
    inside
}

/// Inside `cached_only`: something was missing so far (what `cached_only` will return).
pub fn missed() -> bool {
    MISSED.with(|m| m.get())
}

/// Inside `cached_only`.
pub fn in_cached_only() -> bool {
    CACHED_ONLY.with(|c| c.get())
}

pub fn compile_cached(src: &str, entry: &str, profile: &str) -> Result<(Vec<u8>, bool), String> {
    compile_cached_in(&cache_dir(), src, entry, profile)
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    /// Two programs that compile to the same DXIL share its bytes on disk; one with other bytes
    /// gets its own; each reads back as it was kept.
    #[test]
    fn dxil_with_the_same_bytes_is_kept_once() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-dedup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (one, other) = (b"DXBC one module".to_vec(), b"DXBC another module".to_vec());
        let (a, b, c) = (dir.join("0000000000000001.dxil"), dir.join("0000000000000002.dxil"), dir.join("0000000000000003.dxil"));
        store_dxil(&dir, &a, &one);
        let before = deduplicated();
        store_dxil(&dir, &b, &one);
        store_dxil(&dir, &c, &other);
        assert_eq!(deduplicated() - before, 1, "the second file of the same bytes is a link");
        assert_eq!((std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), std::fs::read(&c).unwrap()), (one.clone(), one.clone(), other.clone()));
        assert_eq!(std::fs::read(content_file(&dir, &one)).unwrap(), one);
        assert_eq!(std::fs::read_dir(dir.join("content")).unwrap().count(), 2);
        // (a file of the content's name holding other bytes is not linked to)
        let wrong = b"DXBC same hash, other bytes".to_vec();
        std::fs::write(content_file(&dir, &wrong), b"something else").unwrap();
        let d = dir.join("0000000000000004.dxil");
        store_dxil(&dir, &d, &wrong);
        assert_eq!(std::fs::read(&d).unwrap(), wrong);
        // (a key file deleted, as a broken one is, leaves the bytes of the others)
        std::fs::remove_file(&a).unwrap();
        assert_eq!(std::fs::read(&b).unwrap(), one);
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().contains(".tmp")), "no file under another name is left");
        let _ = std::fs::remove_dir_all(&dir);
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn synthetic_dxil(marker: u8) -> Vec<u8> {
        let mut b = vec![0u8; 68];
        b[..4].copy_from_slice(b"DXBC");
        b[20..24].copy_from_slice(&1u32.to_le_bytes());
        b[24..28].copy_from_slice(&68u32.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        b[32..36].copy_from_slice(&36u32.to_le_bytes());
        b[36..40].copy_from_slice(b"DXIL");
        b[40..44].copy_from_slice(&24u32.to_le_bytes());
        b[52..56].copy_from_slice(b"DXIL"); // inner DxilBitcodeHeader magic
        b[67] = marker;
        b
    }

    #[test]
    fn cache_statistics_separate_deferred_misses_failures_and_hits() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-stats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let counters = CacheCounters::default();
        let src = "cache diagnostics";
        let request = |compile_one: fn() -> Result<Vec<u8>, String>| {
            compile_cached_in_observed(&dir, src, "shade", "ps_6_6", None, &counters, compile_one)
        };

        let (r, missed) = cached_only(|| request(|| panic!("a cached-only miss must not compile")));
        assert!(r.is_err() && missed);
        assert_eq!(
            counters.snapshot(),
            CacheStats {
                deferred_misses: 1,
                ..Default::default()
            }
        );

        let error = request(|| Err("synthetic DXC diagnostic\nline 2".into())).unwrap_err();
        assert!(
            error.contains(&format!("{:016x}.dxil", cache_key(src, "shade", "ps_6_6"))),
            "{error}"
        );
        assert!(
            error.contains("ps_6_6, entry \"shade\", 17 B HLSL"),
            "{error}"
        );
        assert!(
            error.ends_with("synthetic DXC diagnostic\nline 2"),
            "{error}"
        );
        let invalid = request(|| Ok(b"invalid DXIL".to_vec())).unwrap_err();
        assert!(invalid.contains("invalid DXIL container"), "{invalid}");

        let (dxil, hit) = request(|| Ok(synthetic_dxil(6))).unwrap();
        assert!(!hit);
        assert_eq!(dxil, synthetic_dxil(6));
        let (_, hit) = request(|| panic!("a cache hit must not compile")).unwrap();
        assert!(hit);
        let (r, missed) = cached_only(|| request(|| panic!("a cached-only hit must not compile")));
        assert!(r.unwrap().1 && !missed);

        assert_eq!(
            counters.snapshot(),
            CacheStats {
                hits: 2,
                misses: 3,
                deferred_misses: 1,
                compilations: 3,
                compilation_failures: 2,
                compilation_errors: vec![error, invalid],
                ..Default::default()
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compilation_error_samples_are_bounded_and_distinct() {
        let counters = CacheCounters::default();
        for _ in 0..25 {
            counters.failure("same shader failed".into());
        }
        assert_eq!(
            counters.snapshot().compilation_errors,
            ["same shader failed"]
        );
        for k in 0..25 {
            counters.failure(format!("another shader {k}"));
        }
        let stats = counters.snapshot();
        assert_eq!(stats.compilation_failures, 50);
        assert_eq!(stats.compilation_errors.len(), 20);
        assert_eq!(stats.compilation_errors[0], "same shader failed");
        assert_eq!(stats.compilation_errors[19], "another shader 18");
    }

    #[test]
    fn dxil_cache_round_trip() {
        if location().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("xcl-dxil-test-{}", std::process::id()));
        let src = "[numthreads(1, 1, 1)] void main() {}";
        let (a, hit) = compile_cached_in(&dir, src, "main", "cs_6_6").unwrap();
        assert!(!hit);
        let (b, hit) = compile_cached_in(&dir, src, "main", "cs_6_6").unwrap();
        assert!(hit && a == b);
        assert_ne!(cache_key(src, "main", "cs_6_6"), cache_key(src, "main", "cs_6_7"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The direct backend's container goes through dxil.dll before use; one dxil.dll rejects is
    /// never returned, and DXC's compiles that program instead.
    #[test]
    fn the_validator_passes_direct_containers_and_rejects_damaged_ones() {
        if location().is_err() || VALIDATOR_CREATE.get().is_none() {
            crate::gpu::recomp::runner::skip_test("no DXC with dxil.dll");
            return;
        }
        let src = "RWByteAddressBuffer o : register(u0);\n[numthreads(64, 1, 1)] void main(uint3 t : SV_DispatchThreadID) { o.Store(t.x * 4u, t.x * 3u + 1u); }\n";
        let direct = xcl_dxil::compile(src, "main", "cs_6_6").unwrap();
        validate(&direct).unwrap();
        validate(&compile_with_dxc(src, "main", "cs_6_6", &[]).unwrap()).unwrap();
        let dxil = (0..direct.len() - 4).find(|&i| &direct[i..i + 4] == b"DXIL").unwrap();
        let mut damaged = direct.clone();
        let at = dxil + (damaged.len() - dxil) / 2;
        damaged[at] ^= 0x5a;
        assert!(validate(&damaged).is_err());
        assert!(validate(&direct[..direct.len() / 2]).is_err());
        if backend() == Backend::Direct {
            let before = direct_counts();
            assert_eq!(compile(src, "main", "cs_6_6", &[]).unwrap(), direct);
            assert!(direct_counts().compiled > before.compiled);
        }
    }

    #[test]
    fn cache_key_tracks_the_dxc_toolchain() {
        let a = cache_key_for_toolchain("source", "main", "cs_6_6", "-O1", "compiler-a/signer-a");
        assert_eq!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "-O1", "compiler-a/signer-a"));
        assert_ne!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "-O1", "compiler-b/signer-a"));
        assert_ne!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "-O1", "compiler-a/signer-b"));
    }

    /// DXIL made at one optimisation level is not what a compile at another finds.
    #[test]
    fn cache_key_tracks_the_optimisation_level() {
        let key = |opt| cache_key_for_toolchain("source", "main", "cs_6_6", opt, "compiler/signer");
        assert_ne!(key("-O1"), key("-O3"));
        assert_ne!(key("-O1"), key("-O0"));
        assert_eq!(key("-O1"), key("-O1"));
        if opt_override().is_none() {
            assert_eq!(opt_level(), "-O1");
            assert_eq!(cache_key("source", "main", "cs_6_6"), cache_key_for_toolchain("source", "main", "cs_6_6", "-O1", toolchain_identity()));
        }
    }

    #[test]
    fn validates_dxil_cache_structure_and_rebuilds_corruption() {
        let valid = synthetic_dxil(7);
        assert!(valid_dxil_container(&valid));
        assert!(!valid_dxil_container(&valid[..valid.len() - 1]));
        let mut bad = valid.clone();
        bad[0] = b'X';
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[24..28].copy_from_slice(&999u32.to_le_bytes());
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[32..36].copy_from_slice(&35u32.to_le_bytes());
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[36..40].copy_from_slice(b"STAT");
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[52..56].fill(0);
        assert!(!valid_dxil_container(&bad));

        let mut overlap = vec![0u8; 80];
        overlap[..4].copy_from_slice(b"DXBC");
        overlap[20..24].copy_from_slice(&1u32.to_le_bytes());
        overlap[24..28].copy_from_slice(&80u32.to_le_bytes());
        overlap[28..32].copy_from_slice(&2u32.to_le_bytes());
        overlap[32..36].copy_from_slice(&40u32.to_le_bytes());
        overlap[36..40].copy_from_slice(&40u32.to_le_bytes());
        overlap[40..44].copy_from_slice(b"DXIL");
        overlap[44..48].copy_from_slice(&24u32.to_le_bytes());
        overlap[56..60].copy_from_slice(b"DXIL");
        assert!(!valid_dxil_container(&overlap));

        let dir = std::env::temp_dir().join(format!("xcl-dxil-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = cache_key("corrupt", "main", "cs_6_6");
        let file = dir.join(format!("{key:016x}.dxil"));
        std::fs::write(&file, b"truncated cache entry").unwrap();
        let calls = AtomicUsize::new(0);
        let (rebuilt, hit) = compile_cached_in_with(&dir, "corrupt", "main", "cs_6_6", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(synthetic_dxil(8))
        })
        .unwrap();
        assert!(!hit);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(rebuilt, synthetic_dxil(8));
        let (_, hit) = compile_cached_in_with(&dir, "corrupt", "main", "cs_6_6", || panic!("valid rebuilt entry was not reused")).unwrap();
        assert!(hit);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_invalid_fresh_dxil_without_poisoning_the_cache() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-fresh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let key = cache_key("fresh validation", "main", "cs_6_6");
        let file = dir.join(format!("{key:016x}.dxil"));

        let error = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || Ok(b"not DXIL".to_vec())).unwrap_err();
        assert!(error.contains("invalid DXIL container"));
        assert!(!file.exists());

        let valid = synthetic_dxil(11);
        let (rebuilt, hit) = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || Ok(valid.clone())).unwrap();
        assert!(!hit);
        assert_eq!(rebuilt, valid);
        let (_, hit) = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || panic!("valid retry was not cached")).unwrap();
        assert!(hit);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_cache_misses_compile_once() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-flight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let calls = AtomicUsize::new(0);
        let hits = AtomicUsize::new(0);
        let counters = CacheCounters::default();
        std::thread::scope(|s| {
            let mut threads = Vec::new();
            for _ in 0..8 {
                threads.push(s.spawn(|| {
                    let (dxil, hit) = compile_cached_in_observed(&dir, "same source", "main", "cs_6_6", None, &counters, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(25));
                        Ok(synthetic_dxil(4))
                    })
                    .unwrap();
                    assert_eq!(dxil, synthetic_dxil(4));
                    hits.fetch_add(hit as usize, Ordering::SeqCst);
                }));
            }
            for thread in threads {
                thread.join().unwrap();
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(hits.load(Ordering::SeqCst), 7);
        let stats = counters.snapshot();
        assert_eq!(stats.compilations, 1);
        assert_eq!(stats.compilation_failures, 0);
        assert_eq!(stats.hits + stats.coalesced_hits, 7);
        assert_eq!(stats.misses, 1 + stats.coalesced_hits);
        assert_eq!(stats.hits + stats.misses, 8);
        assert_eq!(stats.deferred_misses, 0);

        // A cached-only attempt (non-blocking pipeline creation on the title's thread) does not
        // wait for a worker compiling the same shader: it misses at once.
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                compile_cached_in_with(&dir, "slow", "main", "cs_6_6", || {
                    std::thread::sleep(Duration::from_millis(400));
                    Ok(synthetic_dxil(5))
                })
            });
            std::thread::sleep(Duration::from_millis(50));
            let t0 = std::time::Instant::now();
            let (r, missed) = cached_only(|| compile_cached_in_with(&dir, "slow", "main", "cs_6_6", || Ok(synthetic_dxil(5))));
            assert!(r.is_err() && missed, "cached-only: a miss");
            assert!(t0.elapsed() < Duration::from_millis(200), "cached-only waited {:?} for the compile", t0.elapsed());
            worker.join().unwrap().unwrap();
        });

        // A failed leader releases the flight; the next caller can retry.
        let failed = compile_cached_in_with(&dir, "retry", "main", "cs_6_6", || Err("synthetic failure".into()));
        assert!(failed.is_err());
        let (dxil, hit) = compile_cached_in_with(&dir, "retry", "main", "cs_6_6", || Ok(synthetic_dxil(9))).unwrap();
        assert!(!hit);
        assert_eq!(dxil, synthetic_dxil(9));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_cache_keys_do_not_block_each_other() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-parallel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            let a = s.spawn(|| compile_cached_in_with(&dir, "source a", "main", "cs_6_6", || {
                barrier.wait();
                Ok(synthetic_dxil(1))
            }));
            let b = s.spawn(|| compile_cached_in_with(&dir, "source b", "main", "cs_6_6", || {
                barrier.wait();
                Ok(synthetic_dxil(2))
            }));
            assert_eq!(a.join().unwrap().unwrap().0, synthetic_dxil(1));
            assert_eq!(b.join().unwrap().unwrap().0, synthetic_dxil(2));
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_aliases_share_the_same_cache_flight() {
        let root = std::env::temp_dir().join(format!("xcl-dxil-alias-{}", std::process::id()));
        let dir = root.join("cache");
        let child = dir.join("child");
        let alias = child.join("..");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&child).unwrap();
        assert_eq!(normalized_cache_dir(&dir), normalized_cache_dir(&alias));
        assert_eq!(normalized_cache_dir(&root.join("future")), normalized_cache_dir(&child.join("..").join("..").join("future")));
        let key = cache_key("aliased source", "main", "cs_6_6");
        assert_eq!(
            cache_mutex_name(&dir.join(format!("{key:016x}.dxil"))),
            cache_mutex_name(&alias.join(format!("{key:016x}.dxil")))
        );

        let start = std::sync::Barrier::new(2);
        let calls = AtomicUsize::new(0);
        let hits = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                start.wait();
                compile_cached_in_with(&dir, "aliased source", "main", "cs_6_6", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    Ok(synthetic_dxil(12))
                })
                .unwrap()
            });
            let second = scope.spawn(|| {
                start.wait();
                compile_cached_in_with(&alias, "aliased source", "main", "cs_6_6", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    Ok(synthetic_dxil(12))
                })
                .unwrap()
            });
            for result in [first.join().unwrap(), second.join().unwrap()] {
                assert_eq!(result.0, synthetic_dxil(12));
                hits.fetch_add(result.1 as usize, Ordering::SeqCst);
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn process_cache_mutex_serializes_the_same_file() {
        let file = std::env::temp_dir().join(format!("xcl-dxil-mutex-{}-{}.dxil", std::process::id(), cache_key("mutex", "main", "cs_6_6")));
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (second_tx, second_rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let first_file = file.clone();
            s.spawn(move || {
                let _guard = process_cache_lock(&first_file).unwrap();
                held_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            held_rx.recv().unwrap();
            s.spawn(move || {
                let _guard = process_cache_lock(&file).unwrap();
                second_tx.send(()).unwrap();
            });
            assert!(second_rx.recv_timeout(Duration::from_millis(30)).is_err());
            release_tx.send(()).unwrap();
            second_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        });
    }
}
