//! Ray tracing off (`XCL_RAYTRACING`; off unless it is 1, on or a tier – the user, 2026-11-26
//! 21:43Z: "You should just turn off ray tracing for now"). Every title gets the general layer:
//! - `CheckFeatureSupport(D3D12_OPTIONS5)` reports `RaytracingTier` 0 (`Config::raytracing_tier_override`),
//!   so a title that asks takes its own non-ray-traced path;
//! - `CreateStateObject` is refused (E_NOTIMPL, as on a GPU without ray tracing);
//! - `DispatchRays`, `SetPipelineState1` and acceleration-structure builds and copies on a command list
//!   are skipped; the postbuild information a build or `EmitRaytracingAccelerationStructurePostbuildInfo`
//!   asks for is still written when the list runs (a `HEADER_BYTES` structure, no bottom levels),
//!   so a title that reads it back for compaction or serialization sizes gets sane values;
//! - the console's acceleration-structure builder (every compute pipeline whose CS lies in
//!   XboxBvhBuilder.dll's image: the DLL ships its stages' programs and passes them as they lie to
//!   CreateComputePipelineState, which every other compute pipeline goes through too; or one the
//!   DLL's own code makes – play-94 keyed
//!   the builder on the console slot, 62, and left all 2,771 compute pipelines out, FFX_FSR4's too) is one unit: off, none of its stages is built or
//!   dispatched; on, no stage dispatches until every stage is built (all or none – play-92 ran some
//!   stages while others were skipped, and the live ones wrote nodes over title objects);
//! - console compute pipelines that traverse acceleration structures (`image_bvh(64)_intersect_ray`
//!   in their native code) are never translated or built, and their dispatches are skipped like
//!   those of a failed build.
//! `XCL_RT_GATE` picks what happens to those pipelines and to the title layer's
//! (`gpu::recomp::rt_gate_miss`): "miss", the default, builds and runs them with every
//! intersection a miss (`gpu::recomp::rt_all_miss`: no structure exists, the builder being off);
//! "hold" leaves them out as described here. The builder is left out in both. "probes" is "miss"
//! with the title layer's ray activation and ray-hit resolve left out unbuilt
//! (`title_gta6::probes_gate_pass`), the selection cool_fox lights its world with on NVIDIA.
//! Fences are never touched: the skipped calls are recorded work, and the queues signal as before.
//! The title layer (`title_gta6::rt_family`) adds the title's own ray passes that do not trace
//! themselves. VIctory forks' `--no-rt` (DexNom RayTime: builder, ray bins, activation, traversal
//! and resolve left out) is the same selection; GTA VI never asks for the ray-tracing tier
//! (play-92 logs no `OPTIONS5` query), so its non-RT path cannot be chosen through a capability.
//! Ray-tracing pipelines - those whose shaders traverse acceleration structures and the title's
//! ray passes, under either gate - can be built in a queue of their own, `pso_sched::RT_AT_ONCE`
//! of them at a time, after every other build (`XCL_RT_COMPILE_QUEUE=1`; off by default).

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::recomp::traces_rays;

/// Ray tracing is off for this run (logged once).
pub fn off() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let off = !crate::state::config().raytracing_on();
        if off {
            if crate::gpu::recomp::rt_all_miss() {
                crate::xlog_info!(
                    "D3D12.X: ray tracing off (XCL_RAYTRACING=1 restores it): no ray-tracing tier reported, state objects refused, DispatchRays and acceleration-structure builds and copies skipped, the acceleration-structure builder left out unbuilt; pipelines that traverse acceleration structures and the title's ray passes are built and run with every ray a miss (XCL_RT_GATE=hold leaves them out unbuilt instead)"
                );
            } else {
                crate::xlog_info!(
                    "D3D12.X: ray tracing off (XCL_RAYTRACING=1 restores it): no ray-tracing tier reported, state objects refused, DispatchRays and acceleration-structure builds and copies skipped, the acceleration-structure builder and pipelines that traverse acceleration structures left out unbuilt (XCL_RT_GATE=hold)"
                );
            }
        }
        off
    })
}

static PIPELINES: AtomicUsize = AtomicUsize::new(0);
static DISPATCHES: AtomicUsize = AtomicUsize::new(0);
static LIST_CALLS: AtomicUsize = AtomicUsize::new(0);

const BUILDER_DLL: &str = "xboxbvhbuilder.dll";
const BUILDER_STAGE: &str = "ray tracing off: a stage of the acceleration-structure builder (XboxBvhBuilder.dll) left out unbuilt";
const TRAVERSAL: &str = "ray tracing off: a pipeline that traverses acceleration structures left out unbuilt";
const TITLE_PASS: &str = "ray tracing off: one of the title's ray passes left out unbuilt";
const PROBES_PASS: &str = "ray tracing off (XCL_RT_GATE=probes): the title's ray activation or ray-hit resolve left out unbuilt";

/// `XCL_RT_GATE=probes`: every ray a miss as with "miss" (the translation is the same, so the
/// cache stays warm), but the title's ray activation and ray-hit resolve are left out unbuilt
/// (`title_gta6::probes_gate_pass`). Unset or any other value: as before.
fn gate_probes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = gate_is_probes(std::env::var("XCL_RT_GATE").ok().as_deref());
        if on {
            crate::xlog_info!("D3D12.X: ray tracing off: XCL_RT_GATE=probes: every ray a miss, and the title's CS_ScreenProbes_ActivateRays and CS_RaytracingResolve left out unbuilt (XCL_RT_GATE=miss runs them, XCL_TITLE_RT_SKIP=0 too)");
        }
        on
    })
}

fn gate_is_probes(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("probes"))
}

/// Why a console shader is left out, and its family for the log (`first_of_family`).
fn classify(container: &[u8]) -> Option<(&'static str, String)> {
    let miss = crate::gpu::recomp::rt_all_miss();
    classify_as(traces_rays(container), container, miss, miss && gate_probes())
}

/// `classify` with the gate as given: with every ray a miss (`miss`) nothing is left out, the
/// traversals and the title's ray passes are built and run (the first of each kind is logged).
#[cfg(test)]
fn classify_gated(container: &[u8], miss: bool) -> Option<(&'static str, String)> {
    classify_as(traces_rays(container), container, miss, false)
}

/// `classify_gated` of a container that traverses acceleration structures or not (`traces`).
/// `probes`: with every ray a miss, the title's ray activation and ray-hit resolve are left out
/// (`gate_probes`).
fn classify_as(traces: bool, container: &[u8], miss: bool, probes: bool) -> Option<(&'static str, String)> {
    if miss {
        if let Some(pass) = probes.then(|| super::title_gta6::probes_gate_pass(container)).flatten() {
            return Some((PROBES_PASS, format!("title ray pass {} (XCL_RT_GATE=probes)", String::from_utf8_lossy(pass))));
        }
        if traces {
            first_miss("a pipeline that traverses acceleration structures", container);
        } else if let Some(pass) = super::title_gta6::ray_pass(container) {
            first_miss(&format!("title ray pass {}", String::from_utf8_lossy(pass)), container);
        }
        return None;
    }
    if traces {
        return Some((TRAVERSAL, "traversal (image_bvh*_intersect_ray in its native code)".into()));
    }
    let pass = super::title_gta6::rt_family(container)?;
    Some((TITLE_PASS, format!("title ray pass {}", String::from_utf8_lossy(pass))))
}

/// The first pipeline of each family built with every ray a miss is logged with its shader.
fn first_miss(family: &str, container: &[u8]) {
    static LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    if LOGGED.lock().unwrap().get_or_insert_with(HashSet::new).insert(family.to_owned()) {
        crate::xlog_info!("D3D12.X: ray tracing off: {family} is built and runs with every ray a miss (first: CS {:016x}; XCL_RT_GATE=hold leaves it out)", crate::gpu::shader::fnv64(container));
    }
}

/// The first pipeline left out of each family is logged with its shader, so a false positive shows.
fn first_of_family(family: &str, shader: u64) {
    static LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    if LOGGED.lock().unwrap().get_or_insert_with(HashSet::new).insert(family.to_owned()) {
        crate::xlog_info!("D3D12.X: ray tracing off: left out {family} (first: CS {shader:016x}; XCL_RAYTRACING=1 keeps it)");
    }
}

/// Why a pipeline of these console shaders is left out unbuilt, when ray tracing is off.
/// `builder`: a stage of the acceleration-structure builder, its CS in XboxBvhBuilder.dll's image or
/// the DLL on the call stack (`building_builder`).
pub fn left_out<'a>(compute: bool, builder: bool, shaders: impl Iterator<Item = &'a [u8]>) -> Option<&'static str> {
    if !compute || !off() {
        return None;
    }
    if builder {
        PIPELINES.fetch_add(1, Ordering::Relaxed);
        first_of_family("the acceleration-structure builder (CS in XboxBvhBuilder.dll's image)", shaders.map(crate::gpu::shader::fnv64).next().unwrap_or(0));
        return Some(BUILDER_STAGE);
    }
    // (by shader: ~2,700 compute shaders against ~51,000 pipelines in the first 30 s)
    static SEEN: Mutex<Option<HashMap<u64, Option<&'static str>>>> = Mutex::new(None);
    let mut why = None;
    for s in shaders {
        let id = crate::gpu::shader::fnv64(s);
        let cached = SEEN.lock().unwrap().as_ref().and_then(|m| m.get(&id).copied());
        let found = cached.unwrap_or_else(|| {
            let found = classify(s).map(|(why, family)| {
                first_of_family(&family, id);
                why
            });
            SEEN.lock().unwrap().get_or_insert_with(HashMap::new).insert(id, found);
            found
        });
        why = why.or(found);
    }
    if why.is_some() {
        PIPELINES.fetch_add(1, Ordering::Relaxed);
    }
    why
}

/// A pipeline whose console compute shaders traverse acceleration structures (`traces_rays`) or
/// are one of the title's ray passes (`title_gta6::miss_ray_pass`): the ray-tracing class, whatever
/// the ray-tracing setting – built and run with every ray a miss (`miss_pass`), tracing rays with
/// ray tracing on. Its builds go in a queue of their own, `pso_sched::RT_AT_ONCE` at a time
/// (`pso_async::rt_pass_last`, `pso_sched::rt_compile_queue`).
pub(super) fn rt_pipeline<'a>(compute: bool, shaders: impl Iterator<Item = &'a [u8]>) -> bool {
    if !compute {
        return false;
    }
    static SEEN: Mutex<Option<HashMap<u64, bool>>> = Mutex::new(None);
    shaders.into_iter().any(|s| {
        let id = crate::gpu::shader::fnv64(s);
        if let Some(&found) = SEEN.lock().unwrap().as_ref().and_then(|m| m.get(&id)) {
            return found;
        }
        let found = traces_rays(s) || super::title_gta6::miss_ray_pass(s).is_some();
        SEEN.lock().unwrap().get_or_insert_with(HashMap::new).insert(id, found);
        found
    })
}

/// Ray tracing off with every ray a miss (`XCL_RT_GATE=miss`): a pipeline of these console shaders
/// traverses acceleration structures or is one of the title's ray passes (`rt_pipeline`), so it is
/// built and runs, but behind every other build (`pso_async::rt_pass_last`;
/// `XCL_RT_PASS_LAST=0`: in its place). Its output with every ray a miss is never what a world
/// pass waits for: play-101 queued ~1,070 of them (999 CS_RaytracingResolve permutations) beside
/// the pipelines the frame drew with, and play-128 built hundreds of them beside the frame
/// (`pso_sched::RT_AT_ONCE` of them at once).
pub fn miss_pass<'a>(compute: bool, shaders: impl Iterator<Item = &'a [u8]>) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var("XCL_RT_PASS_LAST").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: ray tracing off: ray passes built with every ray a miss are built in their place (XCL_RT_PASS_LAST=0)");
        }
        on
    });
    if !on || !off() || !crate::gpu::recomp::rt_all_miss() {
        return false;
    }
    let found = rt_pipeline(compute, shaders);
    if found {
        let n = MISS_PASSES.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || (n.is_power_of_two() && n >= 64) {
            crate::xlog_info!("D3D12.X: ray tracing off: {n} ray-pass pipelines built with every ray a miss go behind every other build (XCL_RT_PASS_LAST=0 builds them in their place)");
        }
    }
    found
}

static MISS_PASSES: AtomicUsize = AtomicUsize::new(0);

/// A draw or dispatch with a pipeline left out (`left_out`) was skipped.
pub fn note_dispatch() {
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

thread_local! {
    static CREATING_BUILDER: Cell<bool> = const { Cell::new(false) };
}

/// The pipeline of compute pipeline description `desc` is a stage of the acceleration-structure
/// builder: its CS lies in XboxBvhBuilder.dll's image, or the DLL's code makes it.
pub(super) unsafe fn builder_desc(desc: *const c_void) -> bool {
    let cs = desc as usize + 8;
    (!desc.is_null() && crate::gpu::shader::readable(cs, 8) && builder_code(*(cs as *const usize))) || called_from_builder()
}

fn builder_code(code: usize) -> bool {
    code != 0 && crate::modules::find_mapped_by_addr(code).is_some_and(|m| m.name.eq_ignore_ascii_case(BUILDER_DLL))
}

fn called_from_builder() -> bool {
    let Some(dll) = crate::modules::find_mapped_by_name(BUILDER_DLL) else { return false };
    let mut frames = [std::ptr::null_mut::<c_void>(); 32];
    let n = unsafe { windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(0, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) };
    frames[..n as usize].iter().any(|&f| dll.contains(f as usize))
}

/// `create` makes a pipeline, a stage of the acceleration-structure builder when `builder` (`builder_desc`).
pub(super) fn creating_builder<R>(builder: bool, create: impl FnOnce() -> R) -> R {
    CREATING_BUILDER.with(|c| c.set(builder));
    let r = create();
    CREATING_BUILDER.with(|c| c.set(false));
    r
}

/// This thread makes a stage of the acceleration-structure builder (`creating_builder`).
pub(super) fn building_builder() -> bool {
    CREATING_BUILDER.with(Cell::get)
}

#[derive(Default)]
struct Builder {
    /// each stage's pipeline as made (a stand-in while it builds in the background)
    stages: Vec<usize>,
    /// every pointer a draw or dispatch may name a stage by: as made, as the title holds it
    /// (console-shaped), and its real pipeline once built
    ids: HashSet<usize>,
}

static BUILDER: Mutex<Option<Builder>> = Mutex::new(None);
static ANY_BUILDER: AtomicBool = AtomicBool::new(false);

/// A stage of the builder was made: `made` by the device, `title` as the title holds it.
pub(super) fn note_builder_stage(made: usize, title: usize) {
    let mut b = BUILDER.lock().unwrap();
    let b = b.get_or_insert_with(Builder::default);
    b.stages.push(made);
    b.ids.extend([made, title]);
    ANY_BUILDER.store(true, Ordering::Release);
}

/// `title` is how the title holds pipeline `made` (console-shaped): one more id if `made` is a stage.
pub(super) fn note_builder_alias(made: usize, title: usize) {
    if let Some(b) = BUILDER.lock().unwrap().as_mut().filter(|b| b.ids.contains(&made)) {
        b.ids.insert(title);
    }
}

/// A stage's real pipeline (`pso_async::mark_builder`).
pub(super) fn note_builder_identity(pso: usize) {
    BUILDER.lock().unwrap().get_or_insert_with(Builder::default).ids.insert(pso);
}

/// A draw or dispatch with `pso` is held back: it is a stage of the builder, and the builder is off
/// or one of its stages is not built.
pub(super) fn builder_held(pso: usize) -> bool {
    if !ANY_BUILDER.load(Ordering::Acquire) || !gate_closed(off(), pso) {
        return false;
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    true
}

/// `builder_held` with ray tracing `off` as given: on, the stages not built yet are put first.
pub(super) fn gate_closed(off: bool, pso: usize) -> bool {
    let stages = {
        let b = BUILDER.lock().unwrap();
        match b.as_ref() {
            Some(b) if b.ids.contains(&pso) => b.stages.clone(),
            _ => return false,
        }
    };
    if off {
        return true;
    }
    let waiting: Vec<usize> = stages.iter().copied().filter(|&s| !super::pso_async::ready(s)).collect();
    for &s in &waiting {
        super::pso_async::build_first(s);
    }
    let waiting = waiting.len();
    if waiting != 0 {
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: acceleration-structure builder held: {waiting} of its {} stages not built – none dispatches until all are (#{n})", stages.len());
        }
    }
    waiting != 0
}

/// For the 2-second pipeline status line: what ray tracing off left out (None: on).
pub fn status() -> Option<String> {
    static LAST: AtomicUsize = AtomicUsize::new(0);
    if !off() {
        return None;
    }
    let now = DISPATCHES.load(Ordering::Relaxed);
    let since = now - LAST.swap(now, Ordering::Relaxed);
    Some(format!(
        "; ray tracing off: {} pipelines left out, {since} of their draws/dispatches skipped in the last 2 s ({now} so far), {} DXR list calls skipped",
        PIPELINES.load(Ordering::Relaxed),
        LIST_CALLS.load(Ordering::Relaxed)
    ))
}

/// What the stand-in structure reports: an empty structure (the console layout's header only).
const STAND_IN_BYTES: u32 = crate::gpu::bvh::HEADER_BYTES as u32;

/// The 32-bit writes (GPU address, value) that answer `count` postbuild queries of `info_type` at
/// `dest`: compacted, current and visualization sizes `STAND_IN_BYTES`; serialization the same
/// size and no bottom-level pointers. An unknown type writes nothing.
fn postbuild_writes(info_type: i32, dest: u64, count: usize) -> Vec<(u64, u32)> {
    let entry: &[u32] = match D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE(info_type) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE => &[STAND_IN_BYTES, 0],
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION => &[STAND_IN_BYTES, 0, 0, 0],
        _ => &[],
    };
    (0..count)
        .flat_map(|k| entry.iter().enumerate().map(move |(i, v)| (dest + (k * entry.len() * 4 + i * 4) as u64, *v)))
        .collect()
}

unsafe fn write_postbuild(list: *mut c_void, writes: &[(u64, u32)]) {
    if writes.is_empty() {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList2::from_raw_borrowed(&list) else { return };
    let params: Vec<_> = writes.iter().map(|&(Dest, Value)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest, Value }).collect();
    l.WriteBufferImmediate(params.len() as u32, params.as_ptr(), None);
}

fn skipped() {
    LIST_CALLS.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "system" fn build(
    this: *mut c_void,
    _desc: *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    count: u32,
    infos: *const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC,
) {
    skipped();
    if count == 0 || infos.is_null() {
        return;
    }
    let writes: Vec<_> = std::slice::from_raw_parts(infos, count as usize).iter().flat_map(|i| postbuild_writes(i.InfoType.0, i.DestBuffer, 1)).collect();
    write_postbuild(this, &writes);
}

unsafe extern "system" fn emit_postbuild(this: *mut c_void, desc: *const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC, sources: u32, _src: *const u64) {
    skipped();
    if let Some(d) = desc.as_ref() {
        write_postbuild(this, &postbuild_writes(d.InfoType.0, d.DestBuffer, sources as usize));
    }
}

unsafe extern "system" fn copy(_this: *mut c_void, _dest: u64, _src: u64, _mode: i32) {
    skipped();
}

unsafe extern "system" fn dispatch_rays(_this: *mut c_void, _desc: *const D3D12_DISPATCH_RAYS_DESC) {
    skipped();
}

unsafe extern "system" fn set_state_object(_this: *mut c_void, _state: *mut c_void) {
    skipped();
}

/// Command list methods replaced while ray tracing is off.
pub(super) fn list_overrides(v: &mut Vec<(&'static str, usize)>) {
    if off() {
        v.extend_from_slice(&[
            ("BuildRaytracingAccelerationStructure", build as usize),
            ("EmitRaytracingAccelerationStructurePostbuildInfo", emit_postbuild as usize),
            ("CopyRaytracingAccelerationStructure", copy as usize),
            ("DispatchRays", dispatch_rays as usize),
            ("SetPipelineState1", set_state_object as usize),
        ]);
    }
}

/// `CreateStateObject` while ray tracing is off: refused, nothing made.
pub(super) unsafe extern "system" fn create_state_object(_this: *mut c_void, _desc: usize, _riid: *const windows::core::GUID, ppv: *mut *mut c_void) -> windows::core::HRESULT {
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: ray tracing off: CreateStateObject refused (E_NOTIMPL)");
    }
    windows::core::HRESULT(0x8000_4001u32 as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postbuild_queries_get_an_empty_structure_s_sizes() {
        let compacted = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE.0;
        assert_eq!(postbuild_writes(compacted, 0x1000, 1), [(0x1000, 256), (0x1004, 0)]);
        assert_eq!(postbuild_writes(compacted, 0x1000, 2), [(0x1000, 256), (0x1004, 0), (0x1008, 256), (0x100c, 0)]);
        let current = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE.0;
        assert_eq!(postbuild_writes(current, 0x40, 1), [(0x40, 256), (0x44, 0)]);
        let serial = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION.0;
        assert_eq!(postbuild_writes(serial, 0x2000, 2), [(0x2000, 256), (0x2004, 0), (0x2008, 0), (0x200c, 0), (0x2010, 256), (0x2014, 0), (0x2018, 0), (0x201c, 0)]);
        assert!(postbuild_writes(99, 0x2000, 1).is_empty());
        assert!(postbuild_writes(compacted, 0x2000, 0).is_empty());
    }

    #[test]
    fn a_bvh_intersection_in_native_code_is_a_traversal() {
        use crate::gpu::recomp::code_traces_rays;
        use crate::gpu::rdna2::encode;
        let mimg = |op: &str| encode(op, "ENC_MIMG", &[("VDATA", 100), ("VADDR", 12), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
        let end = encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap();
        for op in ["IMAGE_BVH_INTERSECT_RAY", "IMAGE_BVH64_INTERSECT_RAY"] {
            assert!(code_traces_rays(&[mimg(op), end.clone()].concat()), "{op}");
        }
        assert!(!code_traces_rays(&[mimg("IMAGE_SAMPLE"), end.clone()].concat()));
        assert!(!code_traces_rays(&end));
        assert_eq!(classify(b"CS_Tree_BatchCull_c47eb2b0_Wrapped"), None);
    }

    #[test]
    fn with_every_ray_a_miss_no_traversal_or_ray_pass_is_left_out() {
        let traversal: &[u8] = b"CS_Tree_BatchCull_c47eb2b0_Wrapped";
        assert_eq!(classify_as(true, traversal, false, false).map(|c| c.0), Some(TRAVERSAL));
        assert_eq!(classify_as(true, traversal, true, false), None);
        assert_eq!(classify_as(true, traversal, true, true), None);
        assert_eq!(classify_gated(b"CS_RaytracingResolve_0", true), None);
        assert_eq!(classify_gated(b"CS_Tree_BatchCull_c47eb2b0_Wrapped", true), None);
        assert_eq!(classify_gated(b"CS_Tree_BatchCull_c47eb2b0_Wrapped", false), None);
    }

    #[test]
    fn the_probes_gate_is_its_own_value() {
        for probes in [Some("probes"), Some(" PROBES "), Some("Probes")] {
            assert!(gate_is_probes(probes), "{probes:?}");
        }
        for other in [None, Some(""), Some("miss"), Some("hold"), Some("probe"), Some("0")] {
            assert!(!gate_is_probes(other), "{other:?}");
        }
        assert_eq!(classify_as(false, b"CS_ScreenProbes_ActivateRays_d60186e6_Wrapped", true, false), None);
        assert_eq!(classify_as(false, b"CS_ScreenProbes_TemporalResolve_0", true, true), None);
    }

    #[test]
    fn the_builder_is_left_out_whole_and_only_compute_pipelines_are_judged() {
        let other: &[u8] = b"CS_Tree_BatchCull";
        assert_eq!(left_out(false, true, [other].into_iter()), None);
        assert_eq!(left_out(true, true, [other].into_iter()).is_some(), off(), "every stage, whatever its shader");
        assert_eq!(left_out(true, false, [other].into_iter()), None);
        assert!(!building_builder());
        assert!(creating_builder(true, building_builder));
        assert!(!creating_builder(false, building_builder));
        assert!(!building_builder());
    }

    #[test]
    fn ffx_fsr4_pipelines_are_not_classified_as_ray_tracing() {
        use std::sync::{atomic::AtomicBool, Arc};
        let fsr4: &[u8] = b"\x00FFX_FSR4_PASS_0\x00FFX_FSR4_RCAS_PASS\x00FFX_FSR4_SPD_AUTOEXPOSURE_PASS\x00";
        assert_eq!(classify(fsr4), None);
        assert_eq!(left_out(true, false, [fsr4].into_iter()), None);
        let dll = vec![0u8; 0x1000];
        let base = dll.as_ptr() as usize;
        crate::modules::register_mapped(Arc::new(crate::modules::MappedModule {
            name: BUILDER_DLL.into(), path: BUILDER_DLL.into(), base, size: dll.len(), entry_rva: 0, is_dll: true,
            exports: Default::default(), tls_callbacks: Mutex::new(Vec::new()), no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true), stack_reserve: 0,
        }));
        let desc = |cs: usize, len: usize| [0usize, cs, len, 0, 0, 0, 0];
        let (stage, other) = (desc(base + 0x100, 0x40), desc(fsr4.as_ptr() as usize, fsr4.len()));
        let (stage_is, other_is) = unsafe { (builder_desc(stage.as_ptr().cast()), builder_desc(other.as_ptr().cast())) };
        crate::modules::unregister_mapped(base);
        assert!(stage_is, "a CS in the builder DLL's image is a builder stage");
        assert!(!other_is, "any other CS is not, whichever slot made it");
        assert!(!unsafe { builder_desc(std::ptr::null()) });
    }

    /// `XCL_RT_DUMP=<shader dump dir>`: every compute shader of a dump through the classifiers,
    /// the title's ray passes included; what would be left out is listed.
    #[test]
    #[ignore]
    fn rt_dump() {
        let dir = std::env::var("XCL_RT_DUMP").expect("XCL_RT_DUMP");
        let (mut n, mut out) = (0, Vec::new());
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(id) = name.strip_suffix("_cs.bin") else { continue };
            let blob = std::fs::read(e.path()).unwrap();
            n += 1;
            let pass = super::super::title_gta6::ray_pass(&blob).map(|p| String::from_utf8_lossy(p).into_owned());
            if traces_rays(&blob) || pass.is_some() {
                out.push(format!("{id} traversal {} pass {}", traces_rays(&blob), pass.unwrap_or_default()));
            }
        }
        out.sort();
        println!("rt_dump: {n} compute shaders, {} left out\n{}", out.len(), out.join("\n"));
    }
}
