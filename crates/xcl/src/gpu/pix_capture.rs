use std::num::NonZeroU32;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use windows::core::{s, HRESULT, PCWSTR};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS};

type CaptureNextFrame = unsafe extern "system" fn(PCWSTR, u32) -> HRESULT;
type SetGlobalTargetWindow = unsafe extern "system" fn(HWND);

struct Capture {
    at_s: Option<f64>,
    at_present: Option<u64>,
    world: bool,
    frames: NonZeroU32,
    path: PathBuf,
    set_window: SetGlobalTargetWindow,
    next_frame: CaptureNextFrame,
    requested: AtomicBool,
}

static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();

pub(crate) fn load_before_d3d12() {
    CAPTURE.get_or_init(|| {
        if !std::env::var_os("XCL_PIX_CAPTURE").is_some_and(|value| value == "1") {
            return None;
        }
        match load() {
            Ok(capture) => Some(capture),
            Err(error) => {
                crate::xlog_warn!("PIX: GPU capture disabled: {error}");
                None
            }
        }
    });
}

fn load() -> Result<Capture, String> {
    // `XCL_PIX_CAPTURE_AT_S` (seconds), or `XCL_PIX_CAPTURE_AT` (a present number, or `world`: the
    // world entry the draw census sees)
    let at_s: Option<f64> = match std::env::var("XCL_PIX_CAPTURE_AT_S") {
        Ok(value) => {
            let s = value.parse().ok().filter(|v: &f64| v.is_finite() && *v >= 0.0);
            Some(s.ok_or("XCL_PIX_CAPTURE_AT_S must be a finite, nonnegative number")?)
        }
        Err(_) => None,
    };
    let (at_present, world) = match std::env::var("XCL_PIX_CAPTURE_AT") {
        Ok(value) if value.trim().eq_ignore_ascii_case("world") => (None, true),
        Ok(value) => (Some(value.trim().parse::<u64>().map_err(|_| "XCL_PIX_CAPTURE_AT must be a present number or `world`")?), false),
        Err(_) => (None, false),
    };
    if at_s.is_none() && at_present.is_none() && !world {
        return Err("nothing to trigger on: set XCL_PIX_CAPTURE_AT (a present number or `world`) or XCL_PIX_CAPTURE_AT_S".into());
    }
    let frames = match std::env::var("XCL_PIX_CAPTURE_FRAMES") {
        Ok(value) => value
            .parse::<NonZeroU32>()
            .map_err(|_| "XCL_PIX_CAPTURE_FRAMES must be a positive integer")?,
        Err(std::env::VarError::NotPresent) => NonZeroU32::MIN,
        Err(_) => return Err("XCL_PIX_CAPTURE_FRAMES must be a positive integer".into()),
    };
    let tag = match std::env::var("XCL_PIX_CAPTURE_TAG") {
        Ok(tag) if !tag.is_empty() && tag.len() <= 64 && tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') => tag,
        Err(std::env::VarError::NotPresent) => format!("xcl-{}", std::process::id()),
        _ => return Err("XCL_PIX_CAPTURE_TAG must be 1–64 ASCII letters, digits, hyphens, or underscores".into()),
    };
    let dll_name = "WinPixGpuCapturer.dll";
    let dll = match std::env::var_os("XCL_PIX_DLL").map(PathBuf::from) {
        Some(path) if path.is_file() => path,
        Some(path) => return Err(format!("XCL_PIX_DLL {} is not a file", path.display())),
        None => {
            let root = std::env::var_os("ProgramFiles")
                .map(PathBuf::from)
                .filter(|dir| dir.is_absolute())
                .ok_or("ProgramFiles is not an absolute path")?
                .join("Microsoft PIX");
            let missing_dll = || format!("{dll_name} not found under {}; install Microsoft PIX", root.display());
            std::fs::read_dir(&root)
                .map_err(|_| missing_dll())?
                .filter_map(Result::ok)
                .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
                .map(|entry| entry.path().join(dll_name))
                .filter(|path| path.is_file())
                .max()
                .ok_or_else(missing_dll)?
        }
    };

    let out = std::env::var_os("XCL_PIX_OUT").map(PathBuf::from);
    let path = match out {
        Some(path) if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("wpix")) => path,
        Some(dir) => dir.join(format!("{tag}.wpix")),
        None => {
            let dir = Path::new(r"D:\xsemu-capy-work-archive\pix").to_path_buf();
            dir.join(format!("{tag}.wpix"))
        }
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    }
    if path.exists() {
        return Err(format!("{} already exists; choose a new XCL_PIX_CAPTURE_TAG", path.display()));
    }

    let wide: Vec<u16> = dll.as_os_str().encode_wide().chain([0]).collect();
    let module = unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
        .map_err(|error| format!("cannot load {}: {error}", dll.display()))?;
    let (set_window, next_frame) = unsafe {
        let set_window = GetProcAddress(module, s!("SetGlobalTargetWindow")).ok_or("PIXSetTargetWindow export unavailable")?;
        let next_frame = GetProcAddress(module, s!("CaptureNextFrame")).ok_or("PIXGpuCaptureNextFrames export unavailable")?;
        (
            std::mem::transmute::<_, SetGlobalTargetWindow>(set_window),
            std::mem::transmute::<_, CaptureNextFrame>(next_frame),
        )
    };
    match (at_s, at_present, world) {
        (Some(s), _, _) => crate::xlog_info!("PIX: loaded {}; will request {frames} frame(s) at {s:.3} s to {}", dll.display(), path.display()),
        (_, Some(n), _) => crate::xlog_info!("PIX: loaded {}; will request {frames} frame(s) at present {n} to {}", dll.display(), path.display()),
        _ => crate::xlog_info!("PIX: loaded {}; will request {frames} frame(s) at the world entry to {}", dll.display(), path.display()),
    }
    Ok(Capture {
        at_s,
        at_present,
        world,
        frames,
        path,
        set_window,
        next_frame,
        requested: AtomicBool::new(false),
    })
}

pub(crate) fn before_present(hwnd: HWND) {
    let Some(Some(capture)) = CAPTURE.get() else { return };
    if capture.requested.load(Ordering::Relaxed) {
        return;
    }
    let now = crate::log::uptime();
    let frames = crate::hle::d3d12x::present::counters().1;
    let due = match (capture.at_s, capture.at_present) {
        (Some(s), _) => now >= s,
        (_, Some(n)) => frames >= n,
        _ => capture.world && WORLD.load(Ordering::Relaxed),
    };
    if !due || capture.requested.swap(true, Ordering::Relaxed) {
        return;
    }
    let wide: Vec<u16> = capture.path.as_os_str().encode_wide().chain([0]).collect();
    crate::xlog_info!("PIX: capture of {} frame(s) to {} started (present {frames}, {now:.3} s)", capture.frames, capture.path.display());
    let result = unsafe {
        (capture.set_window)(hwnd);
        (capture.next_frame)(PCWSTR(wide.as_ptr()), capture.frames.get())
    };
    match result.ok() {
        Ok(()) => crate::xlog_info!("PIX: capture of {} frame(s) to {} requested; PIX writes it when the capture completes", capture.frames, capture.path.display()),
        Err(error) => crate::xlog_warn!("PIX: GPU capture request failed at {now:.3} s: {error}"),
    }
}

/// The draw census saw the world entry (`XCL_PIX_CAPTURE_AT=world`): capture at the next present.
static WORLD: AtomicBool = AtomicBool::new(false);

pub(crate) fn world_entry() {
    if WORLD.swap(true, Ordering::Relaxed) {
        return;
    }
    if matches!(CAPTURE.get(), Some(Some(capture)) if capture.world) {
        crate::xlog_info!("PIX: world entry seen; a capture follows at the next present");
    }
}
