# The DLSS upscaler path (`XCL_UPSCALER`)

The title's own FSR4 chain cannot run on this driver (its `FFX_FSR4_PASS_*` pipelines fail to build), so
`hle/d3d12x/fsr4_standin.rs` takes its place and writes the chain's output with one bilinear pass. This
page is the second, optional upscaler for that same slot: NVIDIA DLSS Super Resolution through
Streamline, loaded at run time.

## Switches

| variable | values | meaning |
| --- | --- | --- |
| `XCL_UPSCALER` | `blit` (default), `dlss` | which upscaler writes the picture |
| `XCL_DLSS_MODE` | `quality` (default), `balanced`, `performance`, `ultra-performance` | the DLSS mode |
| `XCL_DLSS_DIR` | a folder | where the DLLs are; default `<exe dir>\dlss` |

`blit` is byte-for-byte the behaviour before this existed. `dlss` falls back to `blit`, with one line in
the log, when anything is missing — there is no configuration in which asking for DLSS makes the frame
worse than not asking.

## What it needs in the folder

* NVIDIA's Streamline runtime: `sl.interposer.dll`, `sl.common.dll`, `sl.dlss.dll` and the DLSS model
  data Streamline ships beside them.
* `xsemu-dlss-shim.dll` — our thin C++ shim (below). Streamline's API is C++; the emulator is Rust, and
  a shim of a few hundred lines is the smallest honest way across.

`scripts/fetch-dlss.ps1` downloads Streamline into that folder and builds the shim with the MSVC
toolchain already on the PC. **No NVIDIA binary is ever committed**: the folder is git-ignored, and the
script is what makes a build able to use DLSS.

## The shim's ABI

```c
// 0 on success; any other value means "not usable, blit instead".
int  xsemu_dlss_init(void* device, int mode);              // ID3D12Device*, mode = the table above
int  xsemu_dlss_evaluate(void* cmd_list,                   // ID3D12GraphicsCommandList*
                         void* colour,                     // ID3D12Resource*, rendered frame
                         void* depth,                      // ID3D12Resource*, depth
                         void* motion,                     // ID3D12Resource*, motion vectors
                         void* output,                     // ID3D12Resource*, the upscaled picture
                         float jitter_x, float jitter_y,   // the title's jitter, pixels
                         int reset);                        // 1: drop the history this frame
void xsemu_dlss_shutdown(void);
```

The shim reads sizes, formats and flags from the resources' descriptors; it owns Streamline's
`sl::setD3DDevice`, its plugin setup, the per-frame `sl::Constants` and the `sl::ViewportHandle`. It must
create no device of its own and must not submit: the evaluation is recorded on the command list it is
given, between the title's own submissions.

## What it must do per frame

Streamline's SR surface is small — the shim needs nine exports (`slInit`, `slIsFeatureSupported`,
`slSetD3DDevice`, `slGetFeatureFunction`, `slGetNewFrameToken`, `slSetConstants`, `slEvaluateFeature`,
`slFreeResources`, `slShutdown`) plus the DLSS feature's own functions. Per frame:

1. `slSetConstants` with the frame's constants: `jitterOffset` (the title's jitter, **pixels**, and the
   projection matrices must carry **no** jitter), `mvecScale`, `cameraViewToClip`/`clipToCameraView`,
   `depthInverted`, `cameraMotionIncluded`, `reset`.
2. `slEvaluateFeature(sl::kFeatureDLSS, ...)` on the command list.

`slSetTag` is not part of the SR path. Motion vectors are current→previous, in pixels, origin top-left,
+Y down; `mvecScale` is `1/renderRes` for pixel-space vectors, which for this title's FFX vectors (already
in render-size units) collapses to `(1, 1)`.

## The mode is the title's, not ours

DLSS pins the internal resolution of each mode: quality 1.5×, balanced 1.7×, performance 2.0×, ultra
performance 3.0×. The title renders 1280×720 and its chain writes 1920×1080 — 1.5×, i.e. Quality — so
`XCL_DLSS_MODE` is a *request*: the module clamps it to the ratio the title's buffers actually have and
logs the override once. Asking for performance on a 1.5× pair would hand Streamline an invalid pair.

## What is not done yet

The title's jitter (and its reset flag, which the FFX dispatch carries as its own byte) has to be read
out of the title's FidelityFX dispatch, and the production tree has no hook there yet — it is patch 106,
a new `hle/d3d12x/fidelityfx.rs` on the dispatch path: until it exists
`fsr4_standin::jitter()` answers "not known", the shim is refused and the bilinear pass runs. The
reasoning is in that function: an unjittered DLSS evaluation is a blurred picture, not a faster one, so
a guess is worse than the fallback.

## Why the hook is here and not in the present path

`present.rs` copies planes onto the host window; by then the title has already composited its HUD into
plane 0's resolution and the internal render is gone. The FSR4 chain's last pass is the only place where
the rendered frame, the depth and the motion vectors are all still separate and at the internal
resolution — which is exactly the set DLSS wants, and the resolution ratio the title itself chose
(measured: 1280x720 to 1920x1080).
