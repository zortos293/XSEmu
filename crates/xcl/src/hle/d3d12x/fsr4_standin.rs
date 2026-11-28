//! A stand-in for the FSR4 upscaler (`XCL_FSR4_STANDIN`): AMD's library names its pipelines
//! `FFX_FSR4_PASS_<n>`, the first of them reads the frame the title rendered, the ones between
//! are the layers of a network that only touch a memory buffer, and the last writes the upscaled
//! picture the title goes on with. When one of them cannot be built (its DXIL is larger than the
//! driver can take in this memory) the chain cannot finish: held (`pass_chain`) it writes
//! nothing, and run without the missing layers it writes a picture cut and shifted at the
//! network's working size. Engaged, the stand-in runs none of the chain's dispatches; at the last
//! pass it writes that pass's colour targets from the colour the first pass read, with one
//! bilinear pass of XCL's own over the title's own descriptors.
//!
//! `XCL_FSR4_STANDIN` unset: engaged while a pass of the chain has failed to build (or was left
//! out). `1`: always engaged. `0`: never. The names are the library's, not the title's.
//!
//! Engaged or forced, the chain's pipelines are not built either (`drop_builds`): forced, each is
//! left out when the title names it; engaged by a failure, the ones still to build are left out
//! then, and each one named after that. The title keeps its pipeline objects.
//!
//! What is the colour input and what the output is told by what the passes bind, not by AMD's
//! slot names (only the console code of the passes is at hand): the outputs are the largest
//! colour images the last pass may write (the picture, and a history of its size if it keeps
//! one); the input is the largest colour image the first pass reads that the last does not
//! write and that is smaller than the output – one of the output's size only when the first pass
//! reads no smaller one (the chain's own histories are of the output's size, and one of them is
//! not always among what the last pass shows as written). The whole input image is taken as
//! rendered (no dynamic-resolution rectangle), and nothing is accumulated over frames.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const PREFIX: &str = "FFX_FSR4_PASS_";

const SOURCE: &str = r#"
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState lin : register(s0);
[RootSignature("DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), StaticSampler(s0, filter = FILTER_MIN_MAG_LINEAR_MIP_POINT, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    dst.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) {
        return;
    }
    dst[id.xy] = src.SampleLevel(lin, (float2(id.xy) + 0.5) / float2(w, h), 0);
}
"#;

/// An image the compute root arguments on a list name (`bind::compute_images`).
#[derive(Clone, Debug)]
pub(super) struct Image {
    /// its PC resource
    pub resource: usize,
    /// its view's place in the shared view heap; its unordered-access view is the next one
    pub index: u32,
    /// the pass may write it: it has an unordered-access view there
    pub uav: bool,
    pub width: u32,
    pub height: u32,
    /// a plain 2D image
    pub flat: bool,
    /// its PC format (DXGI_FORMAT); 0: not known
    pub format: i32,
}

impl Image {
    fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// `XCL_FSR4_STANDIN`: Some(true) always, Some(false) never, None while a pass has failed.
fn setting() -> Option<bool> {
    static S: OnceLock<Option<bool>> = OnceLock::new();
    *S.get_or_init(|| match std::env::var("XCL_FSR4_STANDIN").ok().as_deref().map(str::trim) {
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    })
}

/// The stand-in may engage in this run: what each view slot names and which slots hold an
/// unordered-access view is kept from the start (`console_heap::note_resource`,
/// `note_native_uav`), whatever the traces are set to – `bind::compute_images` tells the
/// chain's input and outputs by it, and finds no image without it.
pub(super) fn wants_views() -> bool {
    setting() != Some(false)
}

/// The pass number in a pipeline's name, if it is one of the chain's.
fn pass_of(name: &str) -> Option<u32> {
    name.strip_prefix(PREFIX)?.parse().ok()
}

/// A format a float shader can sample or write as colour with three channels or more
/// (DXGI_FORMAT; typeless ones are shown through the title's own typed views).
fn colour(format: i32) -> bool {
    matches!(format, 1 | 2 | 9 | 10 | 11 | 13 | 23 | 24 | 26 | 27 | 28 | 29 | 31 | 87 | 88 | 90 | 91 | 93)
}

/// Of what the first pass read and the last may write: the input (its place in `read`) and the
/// outputs (their places in `written`), or what could not be told.
fn pick(read: &[Image], written: &[Image]) -> Result<(usize, Vec<usize>), String> {
    let target = |i: &&Image| i.uav && i.flat && colour(i.format);
    let most = written.iter().filter(target).map(Image::area).max().ok_or_else(|| format!("no output: the last pass may write {}", list(written.iter().filter(|i| i.uav))))?;
    let mut outputs: Vec<usize> = Vec::new();
    for (k, i) in written.iter().enumerate() {
        if target(&i) && i.area() == most && !outputs.iter().any(|&o| written[o].resource == i.resource) {
            outputs.push(k);
        }
    }
    let source = |i: &&Image| !i.uav && i.flat && colour(i.format) && i.area() <= most && !written.iter().any(|w| w.uav && w.resource == i.resource);
    // (the rendered frame is smaller than the picture unless the title renders at the picture's size)
    let smaller = read.iter().filter(source).map(Image::area).filter(|a| *a < most).max();
    let largest = smaller.or_else(|| read.iter().filter(source).map(Image::area).max()).ok_or_else(|| format!("no input: the first pass read {}", list(read.iter().filter(|i| !i.uav))))?;
    let input = read.iter().position(|i| source(&i) && i.area() == largest).expect("the largest is one of them");
    Ok((input, outputs))
}

fn list<'a>(images: impl Iterator<Item = &'a Image>) -> String {
    // (each resource once: a root table may name it more than once)
    let mut seen: Vec<usize> = Vec::new();
    let v: Vec<String> = images
        .filter(|i| !seen.contains(&i.resource) && {
            seen.push(i.resource);
            true
        })
        .map(|i| format!("{}x{} format {}{} ({:#x})", i.width, i.height, i.format, if i.flat { "" } else { " (not a plain 2D image)" }, i.resource))
        .collect();
    if v.is_empty() {
        "no image".into()
    } else {
        v.join(", ")
    }
}

#[derive(Default)]
struct State {
    /// the chain's pipelines the title has made: their pass numbers
    passes: HashMap<usize, u32>,
    /// by list: what the chain's first pass read there
    read: HashMap<usize, Vec<Image>>,
    /// engaged at the last dispatch of the chain (logged when it changes)
    engaged: bool,
    /// the input and outputs last logged (resources)
    shown: Option<(usize, Vec<usize>)>,
    /// builds of the chain's pipelines given up (`drop_builds`): (no worker had started, one had)
    dropped: (u32, u32),
    /// `dropped` as last logged
    dropped_shown: (u32, u32),
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static ANY: AtomicBool = AtomicBool::new(false);

/// The title named pipeline `pso` `name` (SetName).
pub(super) fn named(pso: usize, name: &str) {
    let Some(n) = pass_of(name) else { return };
    let leave_out = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        s.passes.insert(pso, n);
        setting() == Some(true) || (setting().is_none() && s.engaged)
    };
    ANY.store(true, Ordering::Release);
    if leave_out {
        drop_builds(&[(pso, n)]);
    }
}

/// `pso` is a pipeline of the chain.
pub(super) fn member(pso: usize) -> bool {
    ANY.load(Ordering::Acquire) && STATE.lock().unwrap().as_ref().is_some_and(|s| s.passes.contains_key(&pso))
}

/// The stand-in takes the chain's place: the builds of these pipelines of it (stand-in, pass
/// number) are given up (`pso_async::drop_build`). One no worker had started is never
/// translated or compiled; one a worker had started gets no driver compile from here on, and a
/// helper process compiling it is ended. The title's pipeline objects stay as they are.
fn drop_builds(passes: &[(usize, u32)]) {
    let (mut fresh, mut started) = (0, 0);
    for &(pso, n) in passes {
        let Some(had_started) = super::pso_async::drop_build(pso) else { continue };
        super::real::note_untranslated(pso, format!("{PREFIX}{n} is left out unbuilt: the FSR4 stand-in takes the chain's place (XCL_FSR4_STANDIN=0 builds it)"));
        if had_started {
            started += 1;
        } else {
            fresh += 1;
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            crate::xlog_info!("D3D12.X: FSR4 stand-in: the pipelines of {PREFIX}* are left out unbuilt while it takes the chain's place (first: {PREFIX}{n}, {pso:#x}; XCL_FSR4_STANDIN=0 builds and runs them)");
        });
    }
    if fresh + started != 0 {
        if let Some(s) = STATE.lock().unwrap().as_mut() {
            s.dropped.0 += fresh;
            s.dropped.1 += started;
        }
    }
}

/// One line with the builds dropped so far, when the count has changed (at the chain's last pass).
fn show_dropped(frame: u64) {
    let (fresh, started) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        if s.dropped == s.dropped_shown {
            return;
        }
        s.dropped_shown = s.dropped;
        s.dropped
    };
    crate::xlog_info!(
        "D3D12.X: FSR4 stand-in: {} builds of {PREFIX}* dropped by frame {frame}: {fresh} no worker had started (never translated or compiled), {started} a worker had started (no driver compile of theirs starts from here, a helper process compiling one is ended)",
        fresh + started
    );
}

/// The jitter the title rendered the frame with, and whether the upscaler must reset its history:
/// `(x, y, reset)` in the convention DLSS takes, read from the title's own upscale dispatch by
/// `fidelityfx`. Until the title has dispatched an upscale the answer is "not known", and
/// `upscaler` blits instead of guessing - an unjittered DLSS evaluation is a blurred picture, not
/// a faster one.
pub(super) fn jitter() -> Option<(f32, f32, i32)> {
    super::fidelityfx::jitter()
}

/// The title released pipeline `pso` for good.
pub(super) fn gone(pso: usize) {
    if !ANY.load(Ordering::Acquire) {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.passes.remove(&pso);
    }
}

/// A dispatch on `list` with the title's pipeline `pso`: true when the stand-in takes it – a
/// pass of the chain while the stand-in is engaged. The dispatch is then not to run: the first
/// pass's is remembered for what it read, and at the last pass's the stand-in's own pass is
/// recorded on `list`.
pub(super) unsafe fn takes(list: *mut c_void, pso: usize) -> bool {
    if !ANY.load(Ordering::Acquire) || setting() == Some(false) {
        return false;
    }
    let forced = setting() == Some(true);
    let frame = super::present::counters().0;
    let mut to_drop: Vec<(usize, u32)> = Vec::new();
    let (n, first, last) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return false };
        let Some(&n) = s.passes.get(&pso) else { return false };
        let mut failed: Vec<u32> = if forced { Vec::new() } else { s.passes.iter().filter(|(p, _)| super::pso_async::build_state(**p) == super::pass_chain::Build::Failed).map(|(_, n)| *n).collect() };
        failed.sort_unstable();
        let engaged = forced || !failed.is_empty();
        if engaged != s.engaged {
            s.engaged = engaged;
            if engaged {
                to_drop = s.passes.iter().map(|(p, n)| (*p, *n)).collect();
                let why = if forced { "XCL_FSR4_STANDIN=1".to_string() } else { format!("{} failed to build or was left out", failed.iter().map(|n| format!("{PREFIX}{n}")).collect::<Vec<_>>().join(", ")) };
                crate::xlog_info!("D3D12.X: FSR4 stand-in engaged at frame {frame} ({why}): no {PREFIX}* dispatch runs, and the last pass's colour targets get a bilinear upscale of the colour the first pass reads (XCL_FSR4_STANDIN=0 turns it off)");
            } else {
                crate::xlog_info!("D3D12.X: FSR4 stand-in off from frame {frame}: no pass of {PREFIX}* is failed any more, the chain runs");
            }
        }
        if !engaged {
            s.read.remove(&(list as usize));
            return false;
        }
        (n, s.passes.values().copied().min().unwrap_or(n), s.passes.values().copied().max().unwrap_or(n))
    };
    drop_builds(&to_drop);
    if n == first {
        let read = super::bind::compute_images(list);
        let mut g = STATE.lock().unwrap();
        if let Some(s) = g.as_mut() {
            // (a list that never reaches the last pass leaves its entry behind)
            if s.read.len() >= 64 {
                s.read.clear();
            }
            s.read.insert(list as usize, read);
        }
    }
    if n == last {
        let read = STATE.lock().unwrap().as_mut().and_then(|s| s.read.remove(&(list as usize)));
        upscale(list, read, frame);
        show_dropped(frame);
    }
    true
}

/// At the chain's last pass on `list`: the stand-in's pass from the input to each output.
unsafe fn upscale(list: *mut c_void, read: Option<Vec<Image>>, frame: u64) {
    let written = super::bind::compute_images(list);
    let Some(read) = read else {
        return trouble(frame, "no input: the chain's first pass was not dispatched on this list before its last");
    };
    let (input, outputs) = match pick(&read, &written) {
        Ok(p) => p,
        Err(e) => return trouble(frame, &e),
    };
    let input = &read[input];
    let outputs: Vec<&Image> = outputs.into_iter().map(|k| &written[k]).collect();
    // `XCL_UPSCALER=dlss`: the shim writes the picture and there is nothing to blit. Every reason it
    // cannot - no DLLs, no jitter yet, a refusal - is one line the first time and then the bilinear
    // pass, which is what a build without the switch does.
    if super::upscaler::choice() == super::upscaler::Choice::Dlss {
        let picture = outputs.iter().find(|o| o.format == input.format).copied().or_else(|| outputs.first().copied());
        let attempt = match picture {
            None => Err(String::from("no output to write")),
            Some(out) => match super::upscaler::plan(&read, input) {
                Err(why) => Err(why),
                Ok(plan) => super::upscaler::upscale(list, &plan, out.resource, input.width, out.width),
            },
        };
        match attempt {
            Ok(()) => return,
            Err(why) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    crate::xlog_info!("D3D12.X: upscaler: XCL_UPSCALER=dlss falls back to the bilinear pass: {why}");
                });
            }
        }
    }
    let Some((rs, pso)) = pipeline() else { return };
    let (rs, pso) = (rs as *mut c_void, pso as *mut c_void);
    let (Some(rs), Some(pso), Some(l)) = (ID3D12RootSignature::from_raw_borrowed(&rs), ID3D12PipelineState::from_raw_borrowed(&pso), ID3D12GraphicsCommandList::from_raw_borrowed(&list)) else { return };
    let (Some(src), Some(dst)) = (super::console_heap::shared_gpu(input.index), outputs.iter().map(|o| super::console_heap::shared_gpu(o.index + 1)).collect::<Option<Vec<u64>>>()) else {
        return trouble(frame, "the views are not in the shared view heap");
    };
    super::list_x::ensure_view_heap(list);
    // (every translated dispatch and draw binds its own root signature and pipeline: nothing to put back)
    l.SetComputeRootSignature(rs);
    l.SetPipelineState(pso);
    l.SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: src });
    for (o, at) in outputs.iter().zip(dst) {
        l.SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: at });
        l.Dispatch(o.width.max(1).div_ceil(8), o.height.max(1).div_ceil(8), 1);
    }
    let now = (input.resource, outputs.iter().map(|o| o.resource).collect::<Vec<_>>());
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if s.shown.as_ref() != Some(&now) {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            super::census::trigger("the FSR4 stand-in");
            crate::xlog_info!(
                "D3D12.X: FSR4 stand-in: from frame {frame} the input is {}x{} format {} (view {}, resource {:#x}; of what the first pass read: {}), written to {} (views {:?})",
                input.width, input.height, input.format, input.index, input.resource, self::list(read.iter().filter(|i| !i.uav)), self::list(outputs.iter().copied()), outputs.iter().map(|o| o.index + 1).collect::<Vec<_>>()
            );
        }
        s.shown = Some(now);
    }
}

/// The input or an output could not be told at `frame`: nothing is written (the first 16
/// times and every 256th after are logged).
fn trouble(frame: u64, what: &str) {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n % 256 == 0 {
        crate::xlog_warn!("D3D12.X: FSR4 stand-in: nothing written at frame {frame} ({} so far): {what}", n + 1);
    }
}

/// XCL's upscale pass: (root signature, pipeline), raw; None when it could not be made (logged).
fn pipeline() -> Option<(usize, usize)> {
    static P: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *P.get_or_init(|| unsafe {
        let code = super::real::compile(SOURCE, "cs_5_1");
        let raw = super::real::device_raw();
        let made = (|| -> Result<(usize, usize), String> {
            if code.is_empty() {
                return Err("its HLSL did not compile".into());
            }
            let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
            super::title_textures::own_work(|| {
                let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &code).map_err(|e| format!("root signature: {e}"))?;
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(|e| format!("pipeline: {e}"))?;
                if super::gpu_timing::enabled() {
                    super::dispatch_timing::note(pso.as_raw() as usize, &code);
                }
                Ok((rs.into_raw() as usize, pso.into_raw() as usize))
            })
        })();
        match made {
            Ok(p) => Some(p),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: no FSR4 stand-in pass ({e}): the chain's dispatches are skipped and nothing is written in their place");
                None
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(resource: usize, uav: bool, width: u32, height: u32, format: i32) -> Image {
        Image { resource, index: resource as u32 * 2, uav, width, height, flat: true, format }
    }

    #[test]
    fn only_the_numbered_passes_are_the_chain() {
        assert_eq!(pass_of("FFX_FSR4_PASS_0"), Some(0));
        assert_eq!(pass_of("FFX_FSR4_PASS_26"), Some(26));
        assert_eq!(pass_of("FFX_FSR4_RCAS_PASS"), None);
        assert_eq!(pass_of("FFX_FSR4_SPD_AUTOEXPOSURE_PASS"), None);
        assert_eq!(pass_of("CS_DepthUpscale"), None);
    }

    #[test]
    fn the_rendered_frame_goes_to_the_picture_and_its_history() {
        // the first pass: colour 1280x720 (R11G11B10), depth (R32), motion (R16G16), history 1920x1080, its own target
        let read = [image(1, false, 1280, 720, 26), image(2, false, 1280, 720, 41), image(3, false, 1280, 720, 34), image(4, false, 1920, 1080, 10), image(5, true, 1920, 1080, 10)];
        // the last pass: what the first wrote, the picture, the history, a state of the network's size
        let written = [image(5, false, 1920, 1080, 10), image(6, true, 1920, 1080, 10), image(4, true, 1920, 1080, 10), image(7, true, 960, 540, 10), image(6, true, 1920, 1080, 10)];
        assert_eq!(pick(&read, &written), Ok((0, vec![1, 2])));
    }

    #[test]
    fn what_cannot_be_told_is_said() {
        let colour_in = [image(1, false, 1280, 720, 26)];
        let depth_only = [image(2, false, 1280, 720, 41)];
        let out = [image(6, true, 1920, 1080, 10)];
        assert!(pick(&colour_in, &[image(6, true, 1920, 1080, 41)]).unwrap_err().starts_with("no output"));
        assert!(pick(&depth_only, &out).unwrap_err().starts_with("no input"));
        // an input larger than the output is not the rendered frame
        assert!(pick(&[image(1, false, 3840, 2160, 10)], &out).unwrap_err().starts_with("no input"));
        let mut array = image(1, false, 1280, 720, 26);
        array.flat = false;
        assert!(pick(&[array], &out).unwrap_err().contains("not a plain 2D image"));
    }

    #[test]
    fn an_image_of_the_pictures_size_is_the_input_only_when_no_smaller_one_is_read() {
        let out = [image(6, true, 1920, 1080, 10)];
        // a history of the picture's size the last pass does not show as written, beside the
        // rendered frame and a smaller image of the title's
        let read = [image(8, false, 1920, 1080, 10), image(1, false, 1280, 720, 10), image(9, false, 640, 360, 10)];
        assert_eq!(pick(&read, &out), Ok((1, vec![0])));
        // rendered at the picture's size
        assert_eq!(pick(&[image(2, false, 1920, 1080, 41), image(8, false, 1920, 1080, 10)], &out), Ok((1, vec![0])));
    }

    #[test]
    fn a_resource_is_listed_once() {
        let twice = [image(1, false, 1280, 720, 10), image(1, false, 1280, 720, 10), image(2, false, 1920, 1080, 28)];
        assert_eq!(list(twice.iter()), "1280x720 format 10 (0x1), 1920x1080 format 28 (0x2)");
    }
}
