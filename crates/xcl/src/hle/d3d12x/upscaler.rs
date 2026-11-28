//! Which upscaler writes the FSR4 chain's output (`XCL_UPSCALER=blit|dlss`).
//!
//! `blit` (the default) is the stand-in's own bilinear pass, exactly as before. `dlss` asks a shim
//! DLL for a DLSS Super Resolution evaluation of the same three images the chain's first pass reads:
//! the rendered colour, the depth buffer and the motion vectors (see [`plan`]). The shim is loaded at
//! run time, never linked: `XCL_DLSS_DIR` names the folder, `<exe dir>\dlss` is the default, and the
//! DLLs in it (NVIDIA's Streamline plus our shim, fetched by `scripts/fetch-dlss.ps1` and never
//! committed) are the only thing that makes the path live. Absent, the bilinear pass runs and one log
//! line says why, so a build with the switch on and no DLLs behaves exactly like a build without it.
//!
//! The shim's C ABI, which `docs/dlss-shim.md` also states for whoever builds it:
//!
//! ```c
//! // 0 on success; anything else is a reason for the caller to fall back to the bilinear pass.
//! int  xsemu_dlss_init(void* device, int mode);
//! int  xsemu_dlss_evaluate(void* cmd_list, void* colour, void* depth, void* motion, void* output,
//!                          float jitter_x, float jitter_y, int reset);
//! void xsemu_dlss_shutdown(void);
//! ```
//!
//! `mode` is 0 quality, 1 balanced, 2 performance, 3 ultra performance (`XCL_DLSS_MODE`). The
//! pointers are the D3D12 objects the stand-in already holds; the sizes, formats and what Streamline
//! wants beyond them (its own state, tags and the per-frame `sl::Constants`) are the shim's to read
//! from the resources' descriptors.
//!
//! Nothing here runs unless `XCL_UPSCALER=dlss`: the choice, the folder and the DLL are read once.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::PCWSTR;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

/// Which upscaler the run asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Choice {
    Blit,
    Dlss,
}

/// `XCL_UPSCALER`, read once. An unknown value is `blit` with one line saying so.
pub(super) fn choice() -> Choice {
    static C: OnceLock<Choice> = OnceLock::new();
    *C.get_or_init(|| {
        let raw = std::env::var("XCL_UPSCALER").unwrap_or_default();
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "blit" => Choice::Blit,
            "dlss" => {
                crate::xlog_info!(
                    "D3D12.X: upscaler: DLSS (XCL_UPSCALER=dlss) at {} mode, from {}",
                    mode_name(mode()),
                    shim_dir().display()
                );
                Choice::Dlss
            }
            other => {
                crate::xlog_warn!("D3D12.X: upscaler: XCL_UPSCALER={other} is neither blit nor dlss; blitting");
                Choice::Blit
            }
        }
    })
}

/// `XCL_DLSS_MODE`: 0 quality (the default), 1 balanced, 2 performance, 3 ultra performance.
pub(super) fn mode() -> i32 {
    static M: OnceLock<i32> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("XCL_DLSS_MODE").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "balanced" | "1" => 1,
        "performance" | "2" => 2,
        "ultra-performance" | "ultra" | "3" => 3,
        "" | "quality" | "0" => 0,
        other => {
            crate::xlog_warn!("D3D12.X: upscaler: XCL_DLSS_MODE={other} is not a DLSS mode; quality");
            0
        }
    })
}

fn mode_name(mode: i32) -> &'static str {
    match mode {
        1 => "balanced",
        2 => "performance",
        3 => "ultra performance",
        _ => "quality",
    }
}

/// The DLSS mode a ratio implies: DLSS pins the internal resolution of each mode, so a mode the
/// title's own buffers do not have the ratio for is not a preference to honour but a wrong input.
/// DLSS's ratios are 1.5 (quality), 1.7 (balanced), 2.0 (performance), 3.0 (ultra performance).
fn mode_for_ratio(out_w: u32, in_w: u32) -> i32 {
    if in_w == 0 {
        return 0;
    }
    let ratio = out_w as f32 / in_w as f32;
    let modes = [(0, 1.5f32), (1, 1.7), (2, 2.0), (3, 3.0)];
    modes.iter().min_by(|a, b| (a.1 - ratio).abs().total_cmp(&(b.1 - ratio).abs())).map(|m| m.0).unwrap_or(0)
}

/// Where the shim and Streamline's DLLs are: `XCL_DLSS_DIR`, else `<exe dir>\dlss`.
fn shim_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("XCL_DLSS_DIR") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return std::path::PathBuf::from(dir);
        }
    }
    let exe = std::env::current_exe().unwrap_or_default();
    exe.parent().unwrap_or_else(|| std::path::Path::new(".")).join("dlss")
}

/// The three images the chain's first pass reads at the rendered resolution, in the order DLSS wants
/// them. A depth view is a typeless 32-bit-plus-8 depth format or a plain depth format; motion
/// vectors are a two-channel float image at the rendered size. Measured on the title's own FSR4
/// chain: `1280x720` colour `R16G16B16A16_FLOAT`, depth `R32G8X24_TYPELESS`, motion `R16G16_FLOAT`.
pub(super) struct Plan {
    pub colour: usize,
    pub depth: usize,
    pub motion: usize,
}

fn depth_format(format: i32) -> bool {
    // DXGI_FORMAT_R32G8X24_TYPELESS, D32_FLOAT_S8X24_UINT, D32_FLOAT, D24_UNORM_S8_UINT, D16_UNORM
    matches!(format, 19 | 20 | 40 | 45 | 55)
}

fn motion_format(format: i32) -> bool {
    // DXGI_FORMAT_R16G16_FLOAT, R32G32_FLOAT
    matches!(format, 34 | 16)
}

/// Of what the chain's first pass read: the depth and motion images at the colour input's size, or
/// why they could not be told. Both must be there: DLSS without either is a blurred guess.
pub(super) fn plan(images: &[super::fsr4_standin::Image], colour: &super::fsr4_standin::Image) -> Result<Plan, String> {
    let at = |wanted: fn(i32) -> bool| {
        images
            .iter()
            .find(|i| !i.uav && i.flat && i.width == colour.width && i.height == colour.height && wanted(i.format))
    };
    let depth = at(depth_format).ok_or_else(|| format!("no depth image at {}x{} among {}", colour.width, colour.height, describe(images)))?;
    let motion = at(motion_format).ok_or_else(|| format!("no motion-vector image at {}x{} among {}", colour.width, colour.height, describe(images)))?;
    Ok(Plan { colour: colour.resource, depth: depth.resource, motion: motion.resource })
}

fn describe(images: &[super::fsr4_standin::Image]) -> String {
    images
        .iter()
        .map(|i| format!("{}x{} fmt{}{}", i.width, i.height, i.format, if i.uav { " (uav)" } else { "" }))
        .collect::<Vec<_>>()
        .join(", ")
}

type InitFn = unsafe extern "system" fn(*mut c_void, i32) -> i32;
type EvaluateFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, f32, f32, i32) -> i32;

struct Shim {
    evaluate: EvaluateFn,
}

/// The loaded shim, `None` with the reason logged once when it is not there.
fn shim() -> Option<&'static Shim> {
    static SHIM: OnceLock<Option<Shim>> = OnceLock::new();
    SHIM.get_or_init(|| {
        let dir = shim_dir();
        let path = dir.join("xsemu-dlss-shim.dll");
        if !path.is_file() {
            crate::xlog_info!(
                "D3D12.X: upscaler: {} not found, so XCL_UPSCALER=dlss falls back to the bilinear pass (fetch the DLLs with scripts/fetch-dlss.ps1)",
                path.display()
            );
            return None;
        }
        let wide: Vec<u16> = path.to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
        let module = match unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) } {
            Ok(m) => m,
            Err(e) => {
                crate::xlog_warn!("D3D12.X: upscaler: {} did not load: {e}", path.display());
                return None;
            }
        };
        unsafe {
            let init = GetProcAddress(module, windows::core::PCSTR(b"xsemu_dlss_init\0".as_ptr()));
            let evaluate = GetProcAddress(module, windows::core::PCSTR(b"xsemu_dlss_evaluate\0".as_ptr()));
            let (Some(init), Some(evaluate)) = (init, evaluate) else {
                crate::xlog_warn!("D3D12.X: upscaler: {} exports no xsemu_dlss_init/xsemu_dlss_evaluate; falling back to the bilinear pass", path.display());
                return None;
            };
            let init: InitFn = std::mem::transmute(init);
            let evaluate: EvaluateFn = std::mem::transmute(evaluate);
            // One device per run; the resolution is the title's to change, and a failed init is final.
            if init(super::real::device_raw(), mode()) != 0 {
                crate::xlog_warn!("D3D12.X: upscaler: the DLSS shim refused the device; falling back to the bilinear pass");
                return None;
            }
            crate::xlog_info!("D3D12.X: upscaler: DLSS shim loaded from {}", path.display());
            Some(Shim { evaluate })
        }
    })
    .as_ref()
}

/// Runs one DLSS evaluation for `plan` into `output`. `Ok` means the shim wrote the picture and the
/// caller must not blit; `Err` is the one-line reason to blit instead.
pub(super) unsafe fn upscale(cmd_list: *mut c_void, plan: &Plan, output: usize, in_w: u32, out_w: u32) -> Result<(), String> {
    let Some(shim) = shim() else { return Err(String::from("the shim is not there")) };
    let pinned = mode_for_ratio(out_w, in_w);
    if pinned != mode() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            crate::xlog_info!(
                "D3D12.X: upscaler: the title renders {in_w} into {out_w} ({:.2}x), which is DLSS {}; XCL_DLSS_MODE asked for {} and the title's ratio wins",
                out_w as f32 / in_w.max(1) as f32,
                mode_name(pinned),
                mode_name(mode())
            );
        });
    }
    // The title's own jitter is what DLSS needs, and reading it out of the chain's constants is
    // patch 106; until then the shim is refused rather than fed a guess, and the bilinear pass runs.
    let Some((jx, jy, reset)) = super::fsr4_standin::jitter() else {
        return Err(String::from("the title's jitter is not read yet (patch 106)"));
    };
    let rc = (shim.evaluate)(
        cmd_list,
        plan.colour as *mut c_void,
        plan.depth as *mut c_void,
        plan.motion as *mut c_void,
        output as *mut c_void,
        jx,
        jy,
        reset,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("the shim returned {rc}"))
    }
}
