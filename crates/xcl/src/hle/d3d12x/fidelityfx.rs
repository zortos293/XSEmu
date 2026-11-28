//! The title's per-frame camera jitter, read from its own upscaler dispatch.
//!
//! DLSS and FSR both need the exact sub-pixel offset the frame was rendered with: the upscaler
//! reprojects with it, and an evaluation that assumes zero jitter produces a blurred picture.
//! The title hands that offset to its FSR provider in `ffxDispatchDescUpscale.jitterOffset`, so
//! the value is on the console's own ABI and needs no guesswork.
//!
//! The provider DLL is mapped by our loader, not by Windows, so the title reaches `ffxDispatch`
//! through the export table this crate built. With `XCL_UPSCALER=dlss` that export is detoured
//! once, on the module that exports it: the descriptor is read for `jitterOffset` and `reset`,
//! then the call goes to the provider unchanged. With the default `blit` nothing is detoured.
//!
//! Layout of `ffxDispatchDescUpscale` (FSR 4.0.2 `ffx_upscale.h` + `ffx_api_types.h`, x64):
//! `ffxApiHeader {u32 type; void* pNext}` at 0, six `FfxApiResource` (48 bytes each) from 24,
//! `jitterOffset` at 360, `renderSize` at 376, `upscaleSize` at 384, `reset` at 408, total 432.
//! The offsets were produced by compiling the vendor headers with `x86_64-w64-mingw32-gcc` and
//! reading `offsetof` out of the generated code, not transcribed by hand.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use retour::RawDetour;

use super::upscaler::{self, Choice};

/// `FFX_API_DISPATCH_DESC_TYPE_UPSCALE`: the descriptor type this module can read.
const DESC_TYPE_UPSCALE: u32 = 0x0001_0001;
/// `offsetof(ffxDispatchDescUpscale, jitterOffset)` – two `f32`, in unit pixel space.
const JITTER_OFFSET: usize = 360;
/// `offsetof(ffxDispatchDescUpscale, renderSize)` – two `u32`.
const RENDER_SIZE_OFFSET: usize = 376;
/// `offsetof(ffxDispatchDescUpscale, upscaleSize)` – two `u32`.
const UPSCALE_SIZE_OFFSET: usize = 384;
/// `offsetof(ffxDispatchDescUpscale, reset)` – one `bool`.
const RESET_OFFSET: usize = 408;
/// `sizeof(ffxDispatchDescUpscale)`.
const DESC_SIZE: usize = 432;

/// The last `(x, y)` seen, packed as two `f32` bit patterns; `u64::MAX` means "no dispatch yet".
static JITTER: AtomicU64 = AtomicU64::new(u64::MAX);
/// The last `reset` flag, as `0` or `1`.
static RESET: AtomicU32 = AtomicU32::new(0);
/// How many upscale dispatches have been read.
static SEEN: AtomicU32 = AtomicU32::new(0);
/// The provider's own `ffxDispatch`, from the detour's trampoline.
static ORIG: AtomicUsize = AtomicUsize::new(0);

type DispatchFn = extern "C" fn(*mut usize, *const u32) -> u32;

/// The title's jitter for the frame it is about to upscale: `(x, y, reset)` in DLSS's convention,
/// both in unit pixel space. `None` until the title has dispatched an upscale.
pub(super) fn jitter() -> Option<(f32, f32, i32)> {
    if SEEN.load(Ordering::Acquire) == 0 {
        return None;
    }
    let packed = JITTER.load(Ordering::Acquire);
    let x = f32::from_bits(packed as u32);
    let y = f32::from_bits((packed >> 32) as u32);
    Some((x, y, RESET.load(Ordering::Acquire) as i32))
}

/// Read the descriptor, then run the provider's dispatch. Anything that is not an upscale
/// dispatch is passed through untouched.
unsafe extern "C" fn dispatch(ctx: *mut usize, desc: *const u32) -> u32 {
    let orig: DispatchFn = unsafe { std::mem::transmute(ORIG.load(Ordering::Acquire)) };
    if !desc.is_null() && unsafe { desc.read() } == DESC_TYPE_UPSCALE {
        let p = desc as *const u8;
        // SAFETY: checked type, and the vendor struct is DESC_SIZE bytes; unaligned reads because
        // only the ABI's field offsets are guaranteed, not the caller's alignment of the struct.
        let (x, y, reset, render, upscale) = unsafe {
            (
                (p.add(JITTER_OFFSET) as *const f32).read_unaligned(),
                (p.add(JITTER_OFFSET + 4) as *const f32).read_unaligned(),
                p.add(RESET_OFFSET).read() != 0,
                (
                    (p.add(RENDER_SIZE_OFFSET) as *const u32).read_unaligned(),
                    (p.add(RENDER_SIZE_OFFSET + 4) as *const u32).read_unaligned(),
                ),
                (
                    (p.add(UPSCALE_SIZE_OFFSET) as *const u32).read_unaligned(),
                    (p.add(UPSCALE_SIZE_OFFSET + 4) as *const u32).read_unaligned(),
                ),
            )
        };
        JITTER.store((x.to_bits() as u64) | ((y.to_bits() as u64) << 32), Ordering::Release);
        RESET.store(reset as u32, Ordering::Release);
        let n = SEEN.fetch_add(1, Ordering::AcqRel) + 1;
        if n <= 3 || n % 600 == 0 {
            crate::xlog_info!(
                "D3D12.X: upscaler: ffxDispatch #{n}: {}x{} -> {}x{}, jitter ({x:.3}, {y:.3}), reset {reset}",
                render.0,
                render.1,
                upscale.0,
                upscale.1
            );
        }
    }
    orig(ctx, desc)
}

/// Detour `ffxDispatch` on a freshly mapped console module that exports it.
///
/// Only with `XCL_UPSCALER=dlss`: the blit path never reads the title's jitter, so the default
/// run keeps the provider's code exactly as the title shipped it.
pub(crate) fn install_if_ffx(name: &str, base: usize, exports: &crate::pe::OwnedExports) {
    if !matches!(upscaler::choice(), Choice::Dlss) {
        return;
    }
    let Some(crate::pe::ExportTarget::Rva(rva)) = exports.by_name.get("ffxDispatch") else {
        // A forwarder or no export at all: the title's provider is reached differently, and the
        // upscaler keeps blitting instead of using a jitter it cannot read.
        if exports.by_name.contains_key("ffxDispatch") {
            crate::xlog_warn!("D3D12.X: upscaler: {name} forwards ffxDispatch; its jitter stays unread");
        }
        return;
    };
    if ORIG.load(Ordering::Acquire) != 0 {
        return;
    }
    let target = (base + *rva as usize) as *const ();
    // SAFETY: `target` is code inside the mapped image, executable for as long as the module lives,
    // and `dispatch` has the provider's own signature.
    let detour = match unsafe { RawDetour::new(target, dispatch as *const ()) } {
        Ok(d) => d,
        Err(e) => {
            crate::xlog_warn!("D3D12.X: upscaler: hooking {name}!ffxDispatch failed ({e}); the jitter stays unread");
            return;
        }
    };
    ORIG.store(detour.trampoline() as *const () as usize, Ordering::Release);
    // SAFETY: `target` stays valid for the life of the process and nothing else writes those bytes.
    match unsafe { detour.enable() } {
        Ok(()) => {
            // Leaked on purpose: the detour has to outlive every dispatch, which for a provider
            // loaded once is the process.
            Box::leak(Box::new(detour));
            crate::xlog_info!(
                "D3D12.X: upscaler: reading the title's jitter from {name}!ffxDispatch ({} byte descriptor)",
                DESC_SIZE
            );
        }
        Err(e) => {
            ORIG.store(0, Ordering::Release);
            crate::xlog_warn!("D3D12.X: upscaler: enabling the {name}!ffxDispatch hook failed ({e})");
        }
    }
}
