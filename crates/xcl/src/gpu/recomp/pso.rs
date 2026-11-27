//! Console shaders in pipeline states: a shader blob a title passes to Create*PipelineState
//! (DXBC container with XDAT/XBCH/FSSM, no usable DXIL) becomes DXIL through the recompiler,
//! and the pipeline gets XCL's shader ABI root signature instead of the title's one:
//! root UAVs u0 `io` (initial SGPRs = user data), u1 `res`, plus the directly indexed
//! descriptor heaps (images / samplers, and the raw views of the memory windows: `mem`). Mapping the title's root arguments
//! onto that ABI at draw time is the next step (needs the console runtime's user-data layout).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

use super::{dxc, interface, translate, Mode, Options};
use crate::gpu::xbox;

pub mod material_gate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Pixel,
    Compute,
}

/// A console (xbox-native) shader container?
pub fn is_console(blob: &[u8]) -> bool {
    xbox::is_xbox(blob) || fspx(blob).is_some()
}

/// A compute program in the console driver's own container (an `ACCC` part holding `FSPX`),
/// what AMD FidelityFX (FSR4) creates its passes from on the console: no DXIL, the RDNA2
/// code with its thread geometry and COMPUTE_PGM_RSRC2.
pub struct Fspx<'a> {
    pub threads: [u32; 3],
    pub rsrc1: u32,
    pub rsrc2: u32,
    /// RDNA2 code, from its first instruction to the last `s_endpgm`
    pub code: &'a [u8],
}

/// `FSPX`: 'FSPX', version, entry count, size, then 16-byte entries (type, offset, size, 0)
/// from +0x10, offsets from the start of FSPX. Type 0 (0x1C bytes): numthreads as u16 x3 at
/// +0x0C, RSRC1 at +0x14, RSRC2 at +0x18; type 1: a 0x14-byte hash; type 6: the code behind
/// a 0x24-byte header (the v_nop hash prologue); type 7: resource declarations (32-byte
/// records). Read from FSR4's pipelines (the console upscaler's "NewBE" path).
pub fn fspx(blob: &[u8]) -> Option<Fspx<'_>> {
    let parts = crate::gpu::shader::container_parts(blob)?;
    let p = parts.iter().find(|p| p.fourcc == "ACCC")?;
    let part = blob.get(p.offset..p.offset + p.size)?;
    let u32_at = |b: &[u8], o: usize| b.get(o..o + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
    if part.get(..4)? != b"FSPX" {
        return None;
    }
    let n = u32_at(part, 8)? as usize;
    let entry = |ty: u32| -> Option<&[u8]> {
        (0..n.min(32)).find_map(|e| {
            let at = 0x10 + 16 * e;
            (u32_at(part, at)? == ty).then(|| {
                let (o, s) = (u32_at(part, at + 4)? as usize, u32_at(part, at + 8)? as usize);
                part.get(o..o.checked_add(s)?)
            })?
        })
    };
    let head = entry(0)?;
    let u16_at = |o: usize| head.get(o..o + 2).map(|x| u16::from_le_bytes(x.try_into().unwrap()) as u32);
    let threads = [u16_at(0x0C)?, u16_at(0x0E)?, u16_at(0x10)?];
    if threads.iter().any(|&t| t == 0 || t > 1024) || threads.iter().product::<u32>() > 1024 {
        return None;
    }
    let code = entry(6)?.get(0x24..)?;
    // up to the last s_endpgm (the driver pads with s_code_end)
    let end = (0..code.len() / 4).rev().find(|&i| u32_at(code, 4 * i) == Some(0xBF81_0000))?;
    Some(Fspx { threads, rsrc1: u32_at(head, 0x14)?, rsrc2: u32_at(head, 0x18)?, code: &code[..4 * (end + 1)] })
}

/// Memory windows of the title's unified memory (1 without it).
pub fn mem_windows() -> u32 {
    crate::gpu::uma::get().map_or(1, |u| u.shader_windows() as u32)
}

/// A static sampler of a translated program: (S#, comparison) – see `Translation`.
pub type StaticSampler = ([u32; 4], bool);

/// A console shader as PC DXIL and the static samplers it needs in its root signature.
#[derive(Clone)]
pub struct Compiled {
    pub dxil: Vec<u8>,
    pub samplers: Vec<StaticSampler>,
}

/// Translate a console shader for `stage` to signed DXIL (DXC, disk-cached); its static
/// samplers take registers from `sampler_base` (space 1).
/// Fails when the blob is not a console shader or the program is not fully translated.
pub fn console_to_dxil(blob: &[u8], stage: Stage, sampler_base: u32) -> Result<Compiled, String> {
    console_to_dxil_for(blob, stage, sampler_base, false)
}

/// `console_to_dxil` for a GPU that runs only 32-lane waves when `split_wave64`
/// (`wave32_only_for_device`): a wave64 compute program gets the split form
/// (`Options::split_wave64`) instead of `[WaveSize(64)]`, which such a driver rejects.
/// Finished shaders are also kept in memory (`ShaderCache`): pipelines share their shaders.
pub fn console_to_dxil_for(blob: &[u8], stage: Stage, sampler_base: u32, split_wave64: bool) -> Result<Compiled, String> {
    console_to_dxil_in(blob, stage, sampler_base, split_wave64, mem_windows())
}

/// Every translation for `stage` in the disk cache of this XCL build (`translated`): what
/// `xcl warm-up` builds the driver's blobs of.
pub fn translated_on_disk(stage: Stage) -> Vec<Compiled> {
    translated::all(&dxc::cache_dir(), stage)
}

/// The translation era of this XCL build: its translations' disk cache identity (0: that cache is
/// off), what `pso_manifest` notes a pipeline was made in.
pub fn translation_era() -> u64 {
    translated::identity()
}

/// `console_to_dxil_for` with the title's memory windows given (`xcl shaders --precompile`
/// fills the caches without a title: the same keys as at its run).
pub fn console_to_dxil_in(blob: &[u8], stage: Stage, sampler_base: u32, split_wave64: bool, windows: u32) -> Result<Compiled, String> {
    let hash = crate::gpu::shader::fnv64(blob);
    let legacy = super::legacy_compute_wave64() || (stage == Stage::Compute && split_wave64 && legacy_compute_selected(hash));
    // the direct DXIL backend's fallback lines name the program (`dxc::for_program`)
    let label = format!("{stage:?} {hash:016x}");
    let compile_one = || dxc::for_program(&label, || compile_console_shader(blob, stage, sampler_base, windows, split_wave64, legacy));
    if !cache_allowed(stage, |name| std::env::var_os(name).is_some()) {
        return compile_one();
    }
    static CACHE: OnceLock<Mutex<ShaderCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(ShaderCache::new(64 << 20)));
    let key = ShaderKey {
        hash,
        stage,
        sampler_base,
        windows,
        split_wave64,
        cache_dir: dxc::cache_dir(),
    };
    let profile = stage == Stage::Compute && super::loop_profile::enabled_for(blob);
    let from_disk = || translated::cached_in(&dxc::cache_dir(), translated_identity(blob, stage, legacy, profile), blob, stage, sampler_base, windows, split_wave64, compile_one);
    cached_shader(cache, key, blob, from_disk).map(|shader| shader.as_ref().clone())
}

/// The translated cache's identity for `blob`: a program that may traverse acceleration structures
/// is another translation with every ray a miss (`rt_all_miss`), and a compute program another one
/// without the dispatch thread limits in its prologue (`XCL_DISPATCH_THREAD_DIMENSIONS=0`) or the
/// group spread (`XCL_DISPATCH_SPREAD=0`); every
/// other program keeps the build's identity, so neither switch makes the other stages cold.
fn translated_identity(blob: &[u8], stage: Stage, legacy: bool, profile: bool) -> u64 {
    let ident = translated::identity_for_layout(legacy, profile);
    let ident = tagged_identity(ident, stage == Stage::Compute && !super::dispatch_thread_dimensions(), "no-thread-limits");
    let ident = tagged_identity(ident, stage == Stage::Compute && !super::dispatch_spread(), "no-group-spread");
    rt_miss_identity(ident, super::rt_all_miss() && super::may_trace_rays(blob))
}

pub(super) fn rt_miss_identity(ident: u64, miss: bool) -> u64 {
    tagged_identity(ident, miss, "rt-all-miss")
}

/// `ident` with `tag` when `on`; a disabled cache (0) stays disabled.
pub(super) fn tagged_identity(ident: u64, on: bool, tag: &str) -> u64 {
    if ident != 0 && on {
        translated::identity_for(&format!("{ident:016x}"), tag)
    } else {
        ident
    }
}

fn legacy_compute_selected(hash: u64) -> bool {
    static SHADERS: OnceLock<Vec<u64>> = OnceLock::new();
    SHADERS.get_or_init(|| {
        std::env::var("XCL_COMPUTE_WAVE64_LEGACY_ONLY").unwrap_or_default().split(',')
            .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()
    }).contains(&hash)
}

/// Translations on disk: a console program's (DXIL, static samplers) by its bytes, so a run
/// does not translate again what an earlier run of the same XCL build translated. The DXIL
/// cache is keyed by the generated HLSL, so it needs the translation first: a retail game's
/// ~2 800 programs (611 of more than 10 000 instructions) took its loading threads 0.1 - 0.3 s
/// each at every start. Keyed by two hashes and the length of the bytes, the stage and the
/// translation parameters; translator sources and the DXC toolchain identify compatible builds.
/// `XCL_TRANSLATED_CACHE=0` turns it off.
mod translated {
    use super::{Compiled, Stage};
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const MAGIC: &[u8; 8] = b"XCLTRAN2";

    pub(super) fn identity_for(source: &str, toolchain: &str) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for part in ["xcl-translated-cache-v2", source, toolchain] {
            for byte in part.bytes().chain([0]) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h | 1
    }

    /// Translation semantics and the DXC DLLs, independent of link metadata and build profile,
    /// and the adapter's float compare form (`nan_guards`). Zero means the disk cache is disabled.
    pub(super) fn identity() -> u64 {
        identity_for_layout(super::super::legacy_compute_wave64(), false)
    }

    pub(super) fn identity_for_layout(legacy: bool, profile: bool) -> u64 {
        static I: OnceLock<u64> = OnceLock::new();
        let base = *I.get_or_init(|| {
            if std::env::var("XCL_TRANSLATED_CACHE").is_ok_and(|v| v == "0") {
                return 0;
            }
            identity_for(env!("XCL_TRANSLATOR_SOURCE_ID"), &format!("{}\0{}", super::dxc::toolchain_identity(), super::dxc::opt_policy()))
        });
        if base == 0 {
            return 0;
        }
        // the loop guard's other forms (`XCL_RECOMP_NO_LOOP_REGIONS`, `XCL_LOOP_GUARD_REGIONS`) are other HLSL
        let base = if super::super::wave_wide_loop_guard() {
            identity_for(&format!("{base:016x}"), "no-loop-regions")
        } else if super::super::loop_regions() {
            identity_for(&format!("{base:016x}"), "loop-guard-regions")
        } else {
            base
        };
        let base = if super::super::compute_per_access_coherence() {
            identity_for(&format!("{base:016x}"), "compute-per-access-coherence")
        } else {
            base
        };
        let base = if super::super::bvh_node_fix() {
            identity_for(&format!("{base:016x}"), "bvh-node-fix")
        } else {
            base
        };
        // xsemu: from xcl-dev/xcl nvidia-dev 67a08dbe crates/xcl/src/gpu/recomp/pso.rs
        // `XCL_X_LDS_ORDER=0`: the split wave64 form without its LDS order
        let base = if super::super::emit::lds_order_wanted() { base } else { identity_for(&format!("{base:016x}"), "xcl-x-lds-order-0") };
        let nan = if super::nan_guards() { "nan-guards" } else { "" };
        let nan = &["XCL_RECOMP_NO_PLAIN_LOADS", "XCL_RECOMP_PLAIN_MEMORY"].iter().filter(|v| std::env::var_os(v).is_some()).fold(nan.to_string(), |s, v| format!("{s}\0{v}"));
        let guard = super::super::loop_guard();
        let trip_counters = crate::config::guard_trips_enabled();
        let static_cap = super::super::loop_profile::static_cap();
        // the A/B switches of the structured form, the lane-guard proofs, the quad reads, the page groups, their accessors, the compact accessors, the load loop, the primitive index width and the lane-independent form change the HLSL
        let nan = ["XCL_RECOMP_NO_CFG", "XCL_RECOMP_NO_EXEC_PROOF", "XCL_RECOMP_NO_QUAD_READS", "XCL_RECOMP_NO_PAGE_GROUPS", "XCL_RECOMP_NO_ACCESSOR_SHRINK", "XCL_RECOMP_NO_COMPACT_ACCESS", "XCL_RECOMP_NO_LOAD_LOOP", "XCL_RECOMP_NO_GROUP_FOLD", "XCL_RECOMP_NO_KEPT_PAGES", "XCL_RECOMP_NO_KEPT_PAGES_OUTLINED", "XCL_RECOMP_NO_LONE_LOADS", "XCL_RECOMP_NO_OUTLINE", "XCL_RECOMP_NO_PRIM9", "XCL_RECOMP_NO_LANE_INDEPENDENT"].iter().filter(|v| std::env::var_os(v).is_some()).fold(nan.to_string(), |s, v| format!("{s}\0{v}"));
        let nan = std::env::var("XCL_RECOMP_OUTLINE_MIN_INSTS").map_or(nan.clone(), |n| format!("{nan}\0outline-min-insts={n}"));
        let nan = ["XCL_RECOMP_OUTLINE_RESULTS", "XCL_RECOMP_OUTLINE_ATOMICS", "XCL_RECOMP_OUTLINE_NESTED"].iter().fold(nan, |s, v| std::env::var(v).map_or(s.clone(), |x| format!("{s}\0{v}={x}")));
        let nan = format!("{nan}\0outline-above={}", super::outline_budget());
        if legacy {
            return identity_for(&format!("{base:016x}"), &format!("{nan}\0loop-guard={guard}\0trip-counters={trip_counters}\0loop-profile={profile}\0static-loop-cap={static_cap}\0legacy-compute-wave64"));
        }
        let (ratio, giant) = super::packed_dxil_limits();
        identity_for(&format!("{base:016x}"), &format!("{nan}\0loop-guard={guard}\0trip-counters={trip_counters}\0loop-profile={profile}\0static-loop-cap={static_cap}\0compute-wave64-pack-max-insts={}\0packed-dxil-ratio={ratio}\0packed-dxil-giant={giant}", super::super::compute_wave64_pack_max_insts()))
    }

    /// a second hash of the bytes, independent of `fnv64`'s
    fn hash2(b: &[u8]) -> u64 {
        b.iter().fold(0x9E37_79B9_7F4A_7C15u64, |h, &x| (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29))
    }

    pub(super) fn file(dir: &Path, ident: u64, blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split: bool) -> PathBuf {
        dir.join("translated")
            .join("v2")
            .join(format!("{ident:016x}"))
            .join(format!("{:016x}_{stage:?}_{sampler_base}_{windows}_{}.bin", crate::gpu::shader::fnv64(blob), split as u8))
    }

    fn header(ident: u64, blob: &[u8]) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        for v in [ident, blob.len() as u64, hash2(blob)] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h
    }

    pub(super) fn load_in(path: &Path, ident: u64, blob: &[u8]) -> Option<Compiled> {
        let b = std::fs::read(path).ok()?;
        body(b.strip_prefix(header(ident, blob).as_slice())?)
    }

    /// A file of build `ident`, whatever program it was made from.
    fn load_any(path: &Path, ident: u64) -> Option<Compiled> {
        let b = std::fs::read(path).ok()?;
        let (id, rest) = b.strip_prefix(MAGIC.as_slice())?.split_first_chunk::<8>()?;
        (u64::from_le_bytes(*id) == ident).then_some(())?;
        body(rest.get(16..)?)
    }

    /// Every translation for `stage` this XCL build (`identity`) keeps in `dir`.
    pub(super) fn all_in(dir: &Path, ident: u64, stage: Stage) -> Vec<Compiled> {
        let tag = format!("_{stage:?}_");
        let Ok(files) = std::fs::read_dir(dir.join("translated").join("v2").join(format!("{ident:016x}"))) else { return Vec::new() };
        files
            .flatten()
            .map(|f| f.path())
            .filter(|p| p.extension().is_some_and(|x| x == "bin") && p.file_name().is_some_and(|n| n.to_string_lossy().contains(&tag)))
            .filter_map(|p| load_any(&p, ident))
            .collect()
    }

    pub(super) fn all(dir: &Path, stage: Stage) -> Vec<Compiled> {
        match identity() {
            0 => Vec::new(),
            ident => all_in(dir, ident, stage),
        }
    }

    fn body(mut r: &[u8]) -> Option<Compiled> {
        let u32_ = |r: &mut &[u8]| -> Option<u32> {
            let (v, rest) = r.split_first_chunk::<4>()?;
            *r = rest;
            Some(u32::from_le_bytes(*v))
        };
        let n = u32_(&mut r)? as usize;
        let mut samplers = Vec::with_capacity(n.min(64));
        for _ in 0..n {
            let s = [u32_(&mut r)?, u32_(&mut r)?, u32_(&mut r)?, u32_(&mut r)?];
            let (&cmp, rest) = r.split_first()?;
            r = rest;
            samplers.push((s, cmp != 0));
        }
        let len = u32_(&mut r)? as usize;
        (r.len() == len && len > 0).then(|| Compiled { dxil: r.to_vec(), samplers })
    }

    pub(super) fn store_in(path: &Path, ident: u64, blob: &[u8], c: &Compiled) {
        let mut b = header(ident, blob);
        b.extend_from_slice(&(c.samplers.len() as u32).to_le_bytes());
        for (s, cmp) in &c.samplers {
            s.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
            b.push(*cmp as u8);
        }
        b.extend_from_slice(&(c.dxil.len() as u32).to_le_bytes());
        b.extend_from_slice(&c.dxil);
        let Some(dir) = path.parent() else { return };
        let tmp = dir.join(format!("{}.{}.{:?}.tmp", path.file_name().unwrap_or_default().to_string_lossy(), std::process::id(), std::thread::current().id()));
        // written whole, then renamed: a reader never sees half a file
        if std::fs::create_dir_all(dir).is_ok() && std::fs::write(&tmp, &b).is_ok() && std::fs::rename(&tmp, path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    pub(super) fn cached_in(
        dir: &Path,
        ident: u64,
        blob: &[u8],
        stage: Stage,
        sampler_base: u32,
        windows: u32,
        split: bool,
        translate: impl FnOnce() -> Result<Compiled, String>,
    ) -> Result<Compiled, String> {
        if ident == 0 {
            return translate();
        }
        let path = file(dir, ident, blob, stage, sampler_base, windows, split);
        if let Some(c) = load_in(&path, ident, blob) {
            super::dxc::note_memory_hit();
            super::dxc::note_translation(true);
            return Ok(c);
        }
        super::dxc::note_translation(false);
        let c = translate()?;
        store_in(&path, ident, blob, &c);
        Ok(c)
    }

}

// build.rs keys the translated-shader cache on this file's code (comments and log lines
// aside), so any change here that alters the cached result re-keys it on its own.
fn packed_dxil_limits() -> (usize, usize) {
    static LIMITS: OnceLock<(usize, usize)> = OnceLock::new();
    *LIMITS.get_or_init(|| {
        let limit = |name, default| std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default);
        (
            limit("XCL_COMPUTE_WAVE64_PACK_MAX_DXIL_RATIO", 10),
            limit("XCL_COMPUTE_WAVE64_PACK_GIANT_DXIL_BYTES", crate::hle::d3d12x::real::GIANT_DXIL),
        )
    })
}

#[cfg(test)]
thread_local! {
    /// tests: `outline_budget` on this thread
    pub(super) static OUTLINE_BUDGET: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// The outline budget: bytes of DXIL a compute program may be without calls. A program
/// `emit::outline_from` names is built without calls first, by the direct backend, and outlined
/// only when that DXIL is over the budget (calls are what a driver has been seen to trip over;
/// the size is what it cannot build). `XCL_RECOMP_OUTLINE_ABOVE_KB` in KiB, 1000 unless set; 0:
/// no measuring, every program `emit::outline_from` names is outlined (the rule before this one).
/// Nothing is measured either when DXC makes every program (`XCL_SHADER_BACKEND=hlsl`): the form
/// without calls of a big program takes it minutes.
pub(super) fn outline_budget() -> usize {
    #[cfg(test)]
    if let Some(b) = OUTLINE_BUDGET.get() {
        return b;
    }
    std::env::var("XCL_RECOMP_OUTLINE_ABOVE_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1000).saturating_mul(1024)
}

/// A compute program in the wave64 form of `options` under the outline budget: without calls
/// when that DXIL is within `budget` bytes or `emit::outline_from` does not name the program,
/// else outlined (true). `budget` 0: as `emit::outline_from` says, nothing measured (false).
fn within_outline_budget(blob: &[u8], program: &super::DecodedProgram<'_>, options: &Options, profile: &str, budget: usize) -> Result<(Compiled, bool), String> {
    let measured = budget != 0;
    let t = check_console_translation(blob, super::emit_program(program, &Options { no_outline: measured, ..options.clone() })?)?;
    let dxil = dxc::compile_cached(&t.hlsl, "main", profile)?.0;
    if !(measured && t.outline_candidate && dxil.len() > budget) {
        return Ok((Compiled { dxil, samplers: t.static_samplers }, false));
    }
    let t = check_console_translation(blob, super::emit_program(program, options)?)?;
    let dxil = dxc::compile_cached(&t.hlsl, "main", profile)?.0;
    Ok((Compiled { dxil, samplers: t.static_samplers }, true))
}

fn packed_dxil_fallback(packed: usize, legacy: usize, ratio: usize, giant: usize) -> Option<&'static str> {
    if giant != 0 && packed >= giant && legacy < giant {
        Some("legacy-wave64: crossed driver giant threshold")
    } else if ratio != 0 && packed > legacy.saturating_mul(ratio) {
        Some("legacy-wave64: exceeded DXIL growth ratio")
    } else {
        None
    }
}

pub(super) fn compile_console_shader(blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split_wave64: bool, legacy: bool) -> Result<Compiled, String> {
    with_console_code(blob, stage, sampler_base, windows, split_wave64, |code, mut options, profile| {
        options.legacy_compute_wave64 = legacy;
        let (ratio, giant) = packed_dxil_limits();
        if stage == Stage::Compute && options.wave64 && options.split_wave64 && !options.legacy_compute_wave64 && (ratio != 0 || giant != 0) && dxc::defer() {
            return Err("packed DXIL selection is not cached yet".into());
        }
        // a compute program is made without calls first: that DXIL's size decides (`outline_budget`)
        let budget = if stage == Stage::Compute && options.mem_windows > 1 && dxc::direct_backend() { outline_budget() } else { 0 };
        let program = super::decode_program(code, &options)?;
        let mut t = check_console_translation(blob, super::emit_program(&program, &Options { no_outline: budget != 0, ..options.clone() })?)?;
        if let Some([r, g, b]) = (stage == Stage::Pixel).then(|| debug_color(blob)).flatten() {
            t.hlsl = t.hlsl.replacen("    return xo;\n}", &format!("    xo.t0 = float4({r:.3}, {g:.3}, {b:.3}, 1.0);\n    return xo;\n}}"), 1);
        }
        // `XCL_DEBUG_LOD0=hash,…`: those pixel shaders sample at LOD 0 instead of with implicit
        // derivatives – tells derivative (helper lane) trouble apart from the rest
        if stage == Stage::Pixel && std::env::var("XCL_DEBUG_LOD0").is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == crate::gpu::shader::fnv64(blob))) {
            t.hlsl = sample_at_lod0(&t.hlsl);
        }
        // `XCL_DEBUG_PS_SUB=hash:from=>to||from=>to…` (one pixel shader): every `from` in its HLSL
        // becomes `to`
        if let Some(subs) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_SUB").ok()).flatten().and_then(|v| {
            let (h, rest) = v.split_once(':')?;
            (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| rest.to_string())
        }) {
            for sub in subs.split("||") {
                let Some((from, to)) = sub.split_once("=>") else { continue };
                let n = t.hlsl.matches(from).count();
                crate::xlog_info!("[SHADER] XCL_DEBUG_PS_SUB: {n} occurrence(s) of {from:?} replaced");
                t.hlsl = t.hlsl.replace(from, to);
            }
        }
        // `XCL_DEBUG_PS_EXPR=hash:expression` (one pixel shader): its target 0 gets that HLSL
        // expression, e.g. `float4(xi.a1.rgb, 1)` – an input or a register made visible
        if let Some(expr) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_EXPR").ok()).flatten().and_then(|v| {
            let (h, e) = v.split_once(':')?;
            (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| e.to_string())
        }) {
            t.hlsl = t.hlsl.replacen("    if (!xcl_live) discard;\n    return xo;\n}", &format!("    xo.t0 = {expr};\n    return xo;\n}}"), 1);
        }
        let dxil = dxc::compile_cached(&t.hlsl, "main", profile)?.0;
        let mut selected = Compiled { dxil, samplers: t.static_samplers };
        let mut selected_legacy = options.legacy_compute_wave64;
        let (mut compare_forms, mut outlined) = (t.packed_compute, false);
        if budget != 0 && t.outline_candidate && selected.dxil.len() > budget {
            // over the budget without calls: the legacy wave64 form without calls when that one
            // fits, else the outlined program
            let shader = crate::gpu::shader::fnv64(blob);
            let legacy = if compare_forms { Some(within_outline_budget(blob, &program, &Options { legacy_compute_wave64: true, ..options.clone() }, profile, budget)?) } else { None };
            if let Some((legacy, false)) = legacy {
                crate::xlog_info!("[SHADER] compute {shader:016x}: legacy-wave64: no calls within the outline budget ({} B packed, {} B legacy DXIL without calls; budget {budget} B)", selected.dxil.len(), legacy.dxil.len());
                (selected, selected_legacy, compare_forms) = (legacy, true, false);
            } else {
                let o = check_console_translation(blob, super::emit_program(&program, &options)?)?;
                let dxil = dxc::compile_cached(&o.hlsl, "main", profile)?.0;
                crate::xlog_info!("[SHADER] compute {shader:016x}: outlined, {} B of DXIL ({} B without calls, over the outline budget of {budget} B)", dxil.len(), selected.dxil.len());
                (selected, outlined) = (Compiled { dxil, samplers: o.static_samplers }, true);
            }
        }
        if compare_forms && (ratio != 0 || (giant != 0 && selected.dxil.len() >= giant)) {
            let emit_legacy = || {
                let options = Options { legacy_compute_wave64: true, ..options.clone() };
                within_outline_budget(blob, &program, &options, profile, budget).map(|(legacy, _)| legacy)
            };
            let legacy = if cache_allowed(stage, |name| std::env::var_os(name).is_some()) {
                translated::cached_in(&dxc::cache_dir(), translated_identity(blob, stage, true, false), blob, stage, sampler_base, windows, split_wave64, emit_legacy)
            } else {
                emit_legacy()
            };
            let shader = crate::gpu::shader::fnv64(blob);
            match legacy {
                Ok(legacy) => {
                    let fallback = packed_dxil_fallback(selected.dxil.len(), legacy.dxil.len(), ratio, giant);
                    crate::xlog_info!(
                        "[SHADER] compute {shader:016x}: {} ({} B packed, {} B legacy DXIL; ratio {ratio}x, giant line {giant} B)",
                        fallback.unwrap_or("packed-wave64"), selected.dxil.len(), legacy.dxil.len()
                    );
                    if fallback.is_some() {
                        selected = legacy;
                        selected_legacy = true;
                    }
                }
                Err(e) if dxc::in_cached_only() => return Err(e),
                Err(e) => crate::xlog_warn!("[SHADER] compute {shader:016x}: packed-wave64 ({} B DXIL); legacy comparison unavailable: {e}", selected.dxil.len()),
            }
        }
        if stage == Stage::Compute && super::loop_profile::enabled_for(blob) {
            let options = Options { loop_profile: true, legacy_compute_wave64: selected_legacy, no_outline: budget != 0 && !outlined, ..options };
            let instrumented = check_console_translation(blob, super::emit_program(&program, &options)?)?;
            let dxil = dxc::compile_cached(&instrumented.hlsl, "main", profile)?.0;
            selected = Compiled { dxil, samplers: instrumented.static_samplers };
        }
        Ok(selected)
    })
}

/// Every `X_T.Sample(args)` of generated HLSL as `X_T.SampleLevel(args, 0.0f)`.
fn sample_at_lod0(hlsl: &str) -> String {
    let mut out = String::with_capacity(hlsl.len());
    let mut rest = hlsl;
    while let Some(at) = rest.find("_T.Sample(") {
        let open = at + "_T.Sample(".len();
        out.push_str(&rest[..at]);
        out.push_str("_T.SampleLevel(");
        let mut depth = 1;
        let mut end = open;
        for (k, ch) in rest[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&rest[open..end]);
        out.push_str(", 0.0f)");
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `XCL_DEBUG_MAGENTA=hash[:rrggbb],…` (dump hashes, hex; magenta without a colour): those
/// pixel shaders write that opaque colour to target 0 wherever they run – where a draw lands,
/// told apart from what it shades (a colour per shader: which one drew a pixel).
fn debug_color(blob: &[u8]) -> Option<[f32; 3]> {
    static S: std::sync::OnceLock<Vec<(u64, [f32; 3])>> = std::sync::OnceLock::new();
    let wanted = S.get_or_init(|| {
        std::env::var("XCL_DEBUG_MAGENTA")
            .unwrap_or_default()
            .split(',')
            .filter_map(|item| {
                let item = item.trim();
                let (h, c) = item.split_once(':').unwrap_or((item, "ff00ff"));
                let c = u32::from_str_radix(c, 16).ok()?;
                let ch = |shift: u32| ((c >> shift) & 0xFF) as f32 / 255.0;
                Some((u64::from_str_radix(h, 16).ok()?, [ch(16), ch(8), ch(0)]))
            })
            .collect()
    });
    let h = crate::gpu::shader::fnv64(blob);
    wanted.iter().find(|(w, _)| *w == h).map(|(_, c)| *c)
}

/// The HLSL (and its DXC profile) `console_to_dxil` compiles.
pub fn console_to_hlsl(blob: &[u8], stage: Stage) -> Result<(String, &'static str), String> {
    console_translation(blob, stage, 0, mem_windows(), false, false).map(|(t, p)| (t.hlsl, p))
}

#[derive(Clone, PartialEq, Eq)]
struct ShaderKey {
    hash: u64,
    stage: Stage,
    sampler_base: u32,
    windows: u32,
    split_wave64: bool,
    cache_dir: PathBuf,
}

struct ShaderEntry {
    key: ShaderKey,
    blob: Vec<u8>,
    shader: Arc<Compiled>,
}

impl ShaderEntry {
    fn bytes(&self) -> usize {
        self.blob.len()
            + self.key.cache_dir.capacity()
            + self.shader.dxil.capacity()
            + self.shader.samplers.capacity() * std::mem::size_of::<StaticSampler>()
    }
}

struct ShaderCache {
    entries: VecDeque<ShaderEntry>,
    bytes: usize,
    limit: usize,
}

impl ShaderCache {
    fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }

    fn get(&mut self, key: &ShaderKey, blob: &[u8]) -> Option<Arc<Compiled>> {
        let at = self
            .entries
            .iter()
            .rposition(|e| &e.key == key && e.blob == blob)?;
        let entry = self.entries.remove(at)?;
        let shader = entry.shader.clone();
        self.entries.push_back(entry);
        Some(shader)
    }

    fn insert(&mut self, key: ShaderKey, blob: &[u8], shader: Arc<Compiled>) {
        let bytes = blob.len()
            + key.cache_dir.capacity()
            + shader.dxil.capacity()
            + shader.samplers.capacity() * std::mem::size_of::<StaticSampler>();
        if bytes > self.limit {
            return;
        }
        while self.bytes + bytes > self.limit || self.entries.len() >= 128 {
            let Some(old) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= old.bytes();
        }
        self.entries.push_back(ShaderEntry {
            key,
            blob: blob.to_vec(),
            shader,
        });
        self.bytes += bytes;
    }
}

fn cached_shader(
    cache: &Mutex<ShaderCache>,
    key: ShaderKey,
    blob: &[u8],
    compile_one: impl FnOnce() -> Result<Compiled, String>,
) -> Result<Arc<Compiled>, String> {
    if let Some(shader) = cache.lock().unwrap().get(&key, blob) {
        dxc::note_memory_hit();
        return Ok(shader);
    }
    // Keep compilation outside the cache lock, including cached-only probes.
    let shader = Arc::new(compile_one()?);
    let mut cache = cache.lock().unwrap();
    if let Some(existing) = cache.get(&key, blob) {
        return Ok(existing);
    }
    cache.insert(key, blob, shader.clone());
    Ok(shader)
}

fn cache_allowed(stage: Stage, present: impl Fn(&str) -> bool) -> bool {
    !["XCL_RECOMP_NO_STRUCTURED", "XCL_RECOMP_WHY", "XCL_X64_ALL_MASKS", "XCL_WAVE64_SPIN", "XCL_HLSL_COMMENTS", "XCL_DS_VALIDATE"]
        .iter()
        .any(|name| present(name))
        && (stage != Stage::Pixel
            || ![
                "XCL_DEBUG_MAGENTA",
                "XCL_DEBUG_LOD0",
                "XCL_DEBUG_PS_SUB",
                "XCL_DEBUG_PS_EXPR",
            ]
            .iter()
            .any(|name| present(name)))
}

/// User SGPRs of a console compute shader (COMPUTE_PGM_RSRC2 in FSPX or FSSM):
/// s0…s3 are the driver's, so what is above 4 is the root arguments its compiler passed in
/// registers. None: not a console compute shader.
pub fn compute_user_sgprs(blob: &[u8]) -> Option<u32> {
    compute_interface(blob).map(|cs| cs.user_sgprs)
}

/// The dispatch ABI of a console compute shader: its user SGPRs and the console's group shape
/// (`numthreads`). None: not a console compute shader.
pub fn compute_interface(blob: &[u8]) -> Option<super::CsInterface> {
    if let Some(f) = fspx(blob) {
        return Some(super::CsInterface::from_rsrc2(f.threads, f.rsrc2));
    }
    let sh = xbox::parse(blob)?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6))?;
    match interface(blob, &sh, f)? {
        Mode::Compute(cs) => Some(cs),
        _ => None,
    }
}

/// Once per translation: the loops its loop budget (`XCL_LOOP_GUARD`, `super::loop_budget`) may cut
/// short in a working program.
fn note_long_loops(blob: &[u8], t: &super::Translation) {
    let Some(budget) = t.loop_budget.filter(|_| !t.long_loops.is_empty()) else {
        return;
    };
    let budget = match budget {
        super::LoopBudget::Regions { region, total } => format!("{region} iterations per lane in a loop region, {total} in all"),
        one => format!("{} iterations per lane", one.of_a_loop()),
    };
    let more = if t.long_loops.len() > 4 { format!("; and {} more", t.long_loops.len() - 4) } else { String::new() };
    crate::xlog_info!(
        "[SHADER] {:016x}: {} loop(s) may run longer than the loop guard ({budget}, XCL_LOOP_GUARD): {}{more}",
        crate::gpu::shader::fnv64(blob),
        t.long_loops.len(),
        t.long_loops[..t.long_loops.len().min(4)].join("; ")
    );
}

fn console_translation(blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split_wave64: bool, force_legacy_compute: bool) -> Result<(super::Translation, &'static str), String> {
    with_console_code(blob, stage, sampler_base, windows, split_wave64, |code, mut options, profile| {
        options.legacy_compute_wave64 |= force_legacy_compute;
        Ok((check_console_translation(blob, translate(code, &options)?)?, profile))
    })
}

fn with_console_code<R>(blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split_wave64: bool, run: impl FnOnce(&[u8], Options, &'static str) -> Result<R, String>) -> Result<R, String> {
    let guard_trip_counters = stage != Stage::Compute || crate::config::guard_trips_enabled();
    let options = Options { split_wave64, static_sampler_base: sampler_base, mem_windows: windows, nan_guards: nan_guards(), guard_trip_counters, ..Default::default() };
    if let Some(f) = fspx(blob) {
        if stage != Stage::Compute {
            return Err("an FSPX (driver) container in a graphics slot".into());
        }
        let cs = super::CsInterface::from_rsrc2(f.threads, f.rsrc2);
        return run(f.code, Options { mode: Mode::Compute(cs), wave64: super::detect_wave64(f.code), ..options }, "cs_6_6");
    }
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    // the hardware variant for this pipeline slot: VS (not the ES variant used with GS), PS
    let want = match stage {
        Stage::Vertex => 4,
        Stage::Pixel => 5,
        Stage::Compute => 6,
    };
    let f = sh
        .fssm
        .iter()
        .find(|f| f.stages.first().is_some_and(|s| s.hw == want))
        .ok_or_else(|| format!("no hardware variant {}", xbox::hw_stage_name(want)))?;
    let mode = interface(blob, &sh, f).ok_or("unknown shader interface")?;
    let profile = match (&mode, stage) {
        (Mode::Vertex(_), Stage::Vertex) => "vs_6_6",
        (Mode::Pixel(_), Stage::Pixel) => "ps_6_6",
        (Mode::Compute(_), Stage::Compute) => "cs_6_6",
        _ => return Err("the shader stage does not match its slot in the PSO".into()),
    };
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    run(code, Options { mode, wave64: super::fssm_wave64(f, code), code_data: super::code_data(f, p), ..options }, profile)
}

fn check_console_translation(blob: &[u8], t: super::Translation) -> Result<super::Translation, String> {
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    note_long_loops(blob, &t);
    Ok(t)
}

/// A console geometry stage as XCL runs it, in two passes: the console program as a compute
/// shader writes one record per thread group (vertices + primitives, `record`) into `res`,
/// then a pass-through mesh shader rasterises the records with the pixel shader.
pub struct TwoPass {
    /// compute pass (cs_6_6 DXIL)
    pub cs: Vec<u8>,
    /// pass-through mesh shader (ms_6_6 DXIL)
    pub ms: Vec<u8>,
    pub record: super::MsInterface,
    /// mesh groups of the pass-through per record (> 1: the record is over D3D12's mesh
    /// output limit and goes out in runs of triangles)
    pub mesh_groups_per_record: u32,
    /// static samplers of the compute pass
    pub samplers: Vec<StaticSampler>,
}

fn finish_two_pass(t: super::Translation, record: super::MsInterface, ps: Option<&[u8]>) -> Result<TwoPass, String> {
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    let cs = dxc::compile_cached(&t.hlsl, "main", "cs_6_6")?.0;
    let ps_inputs = ps.map(|p| crate::gpu::dxbc::signatures(p).0);
    let ms = dxc::compile_cached(&super::emit::mesh_passthrough_for(&record, ps_inputs.as_deref()), "main", "ms_6_6")?.0;
    let mesh_groups_per_record = super::emit::passthrough_split(&record, super::emit::passthrough_elements(&record, ps_inputs.as_deref()));
    Ok(TwoPass { cs, ms, record, mesh_groups_per_record, samplers: t.static_samplers })
}

/// The mesh-shader (NGG) program of a console MS container; the pass-through carries what
/// the pixel shader `ps` (its container: console or PC) reads.
pub fn console_mesh(blob: &[u8], ps: Option<&[u8]>, split_wave64: bool) -> Result<TwoPass, String> {
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 7)).ok_or("no NGG hardware variant")?;
    let mode = interface(blob, &sh, f).ok_or("unknown MS interface (e.g. lines / points)")?;
    let Mode::Mesh(ms) = &mode else { return Err("the container is not an MS".into()) };
    let record = ms.clone();
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    let t = super::translate_mesh(code, ms.clone(), super::fssm_wave64(f, code), split_wave64, mem_windows())?;
    note_long_loops(blob, &t);
    finish_two_pass(t, record, ps)
}

/// A console mesh (NGG) program as a host mesh shader (`super::MeshNative`).
pub struct NativeMesh {
    /// ms_6_6 DXIL
    pub ms: Vec<u8>,
    /// static samplers of the mesh shader (registers from 0; the pixel shader's follow)
    pub samplers: Vec<StaticSampler>,
}

/// The mesh-shader program of a console MS container as a host mesh shader whose vertices carry
/// what the pixel shader `ps` reads. `split_wave64`: `mesh_native_split` of the adapter. Err: the
/// program cannot be one (`emit::Em::mesh_native_blocker`, an incomplete translation, DXC) – the
/// caller falls back to `console_mesh`.
pub fn console_mesh_native(blob: &[u8], ps: Option<&[u8]>, split_wave64: bool) -> Result<NativeMesh, String> {
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 7)).ok_or("no NGG hardware variant")?;
    let mode = interface(blob, &sh, f).ok_or("unknown MS interface (e.g. lines / points)")?;
    if !matches!(mode, Mode::Mesh(_)) {
        return Err("the container is not an MS".into());
    }
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    let native = super::MeshNative {
        ps_inputs: ps.map(|p| crate::gpu::dxbc::signatures(p).0),
        lds_bytes: crate::gpu::dxbc::psv(blob).map_or(0, |p| p.ms_groupshared),
    };
    let t = translate(code, &Options { mode, wave64: super::fssm_wave64(f, code), split_wave64, mem_windows: mem_windows(), nan_guards: nan_guards(), mesh_native: Some(native), code_data: super::code_data(f, p), ..Default::default() })?;
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    note_long_loops(blob, &t);
    Ok(NativeMesh { ms: dxc::compile_cached(&t.hlsl, "main", "ms_6_6")?.0, samplers: t.static_samplers })
}

/// Whether console mesh programs can be host mesh shaders on an adapter with this wave range, and
/// then the `split_wave64` flag of their translation. HLSL has no `WaveSize` for mesh shaders, so
/// the translation must know the wave size the driver will use: only a range of exactly 32 lanes
/// says so (NVIDIA) – a wave64 program then packs both halves into one warp. Anything else (AMD's
/// 32..64, Intel's 8..32, WARP's 4) keeps the two-pass form, whose compute pass sets `WaveSize`.
pub fn mesh_native_waves(wave_min: u32, wave_max: u32) -> Option<bool> {
    (wave_min == 32 && wave_max == 32).then_some(true)
}

/// `XCL_MESH=two-pass`: console mesh programs in two passes (compute + pass-through) also where
/// they could be host mesh shaders – to compare the two forms, and to test the fallback.
pub fn mesh_two_pass_forced() -> bool {
    std::env::var("XCL_MESH").is_ok_and(|v| v.eq_ignore_ascii_case("two-pass"))
}

/// `mesh_native_waves` for the device (None also when it cannot be asked, has no mesh shaders, or
/// `XCL_MESH=two-pass`).
pub fn mesh_native_split(device: &ID3D12Device) -> Option<bool> {
    if mesh_two_pass_forced() {
        return None;
    }
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32) }.ok()?;
    if o7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
        return None;
    }
    wave_range(device).and_then(|(min, max)| mesh_native_waves(min, max))
}

/// Whether the adapter runs only 32-lane waves (its wave range lacks 64): wave64 compute,
/// mesh and geometry programs then need the split form.
/// PCI vendor of the adapter shaders are translated for (0 until the device exists).
static ADAPTER_VENDOR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_adapter_vendor(vendor: u32) {
    ADAPTER_VENDOR.store(vendor, std::sync::atomic::Ordering::Relaxed);
}

/// `Options::nan_guards`: unless the adapter is known not to be AMD's (its driver gave a
/// later ordered compare of the same operands the unordered result, `emit::float_compare`).
pub fn nan_guards() -> bool {
    matches!(ADAPTER_VENDOR.load(std::sync::atomic::Ordering::Relaxed), 0 | 0x1002)
}

pub fn wave32_only(wave_min: u32, wave_max: u32) -> bool {
    !(wave_min <= 64 && wave_max >= 64)
}

/// `XCL_SPLIT_WAVE64=1`: the split wave64 form also where the GPU has 64-lane waves – to check
/// the path of GPUs with 32-lane waves only (NVIDIA) on an RDNA GPU, which runs both sizes.
pub fn split_wave64_forced() -> bool {
    std::env::var("XCL_SPLIT_WAVE64").is_ok_and(|v| v == "1")
}

/// Whether a wave64 VS / PS can run in 32-lane waves on the adapter: D3D12 has no `WaveSize`
/// for graphics shaders, so the driver picks, and one whose range includes 32 may pick 32 –
/// an RDNA2 driver (RX 6600 XT, issue #42) runs pixel shaders in waves of 32 although it
/// reports 32–64. Such a VS / PS needs the `upper_lanes` form (`Options::split_wave64`), which
/// checks the wave size at run time, so a driver that picks 64 lanes runs it as before.
pub fn graphics_wave32(wave_min: u32, wave_max: u32) -> bool {
    wave32_only(wave_min, wave_max) || wave_min <= 32
}

/// The device's wave lane range; `None` when it cannot be asked.
fn wave_range(device: &ID3D12Device) -> Option<(u32, u32)> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            &mut options as *mut _ as _,
            std::mem::size_of_val(&options) as u32,
        )
    }
    .ok()
    .map(|_| (options.WaveLaneCountMin, options.WaveLaneCountMax))
}

pub fn wave32_only_for_device(device: &ID3D12Device) -> bool {
    split_wave64_forced() || wave_range(device).is_none_or(|(min, max)| wave32_only(min, max))
}

/// `graphics_wave32` for the device: the split flag for a wave64 VS / PS. Unknown: assume so.
pub fn graphics_wave32_for_device(device: &ID3D12Device) -> bool {
    split_wave64_forced() || wave_range(device).is_none_or(|(min, max)| graphics_wave32(min, max))
}

/// A legacy GS pipeline: the ES variant of a console VS + the GS as one program.
pub fn console_geometry(vs: &[u8], gs: &[u8], ps: Option<&[u8]>, split_wave64: bool) -> Result<TwoPass, String> {
    let (code, g) = super::geometry_program(vs, gs)?;
    let record = g.record();
    let translation = super::translate_geometry(&code, g, split_wave64, mem_windows())?;
    note_long_loops(gs, &translation);
    finish_two_pass(translation, record, ps)
}

/// The geometry shader (gs_6_6 DXIL) that draws a console rect list's rectangles behind the
/// console VS `vs` (its container): `emit::rect_list_gs` for the VS's output signature, which
/// is what the translated VS's output struct is declared from.
pub fn rect_list_gs(vs: &[u8]) -> Result<Vec<u8>, String> {
    let outputs = crate::gpu::dxbc::signatures(vs).1;
    if outputs.is_empty() {
        return Err("the VS has no output signature".into());
    }
    Ok(dxc::compile_cached(&super::emit::rect_list_gs(&outputs), "main", "gs_6_6")?.0)
}

/// XCL shader ABI root signature (see the module docs); `graphics` allows an input layout.
pub fn abi_root_signature(dev: &ID3D12Device, graphics: bool) -> Result<ID3D12RootSignature, String> {
    abi_root_signature_with(dev, graphics, &[])
}

/// A static sampler (register `k`, space 1) from its S#; static samplers have only three
/// border colours (others: transparent black).
pub fn static_sampler_desc(k: u32, (w, comparison): &StaticSampler) -> D3D12_STATIC_SAMPLER_DESC {
    let d = crate::gpu::desc::sampler_d3d12(w, *comparison);
    let border = match d.BorderColor {
        [0.0, 0.0, 0.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        [1.0, 1.0, 1.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        _ => D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
    };
    D3D12_STATIC_SAMPLER_DESC {
        Filter: d.Filter,
        AddressU: d.AddressU,
        AddressV: d.AddressV,
        AddressW: d.AddressW,
        MipLODBias: d.MipLODBias,
        MaxAnisotropy: d.MaxAnisotropy,
        ComparisonFunc: d.ComparisonFunc,
        BorderColor: border,
        MinLOD: d.MinLOD,
        MaxLOD: d.MaxLOD,
        ShaderRegister: k,
        RegisterSpace: 1,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// The ABI root signature with the pipeline's static samplers (registers 0.., space 1).
pub fn abi_root_signature_with(dev: &ID3D12Device, graphics: bool, samplers: &[StaticSampler]) -> Result<ID3D12RootSignature, String> {
    let statics: Vec<D3D12_STATIC_SAMPLER_DESC> = samplers.iter().enumerate().map(|(k, s)| static_sampler_desc(k as u32, s)).collect();
    unsafe {
        // u0 io, u1 res; the memory windows are raw views in the descriptor heap (`IO_MEM_VIEWS`)
        let params: Vec<D3D12_ROOT_PARAMETER1> = (0..2)
            .map(|i| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: i, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            })
            .collect();
        let mut flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        if graphics {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: statics.len() as u32,
                    pStaticSamplers: if statics.is_empty() { std::ptr::null() } else { statics.as_ptr() },
                    Flags: flags,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| format!("{e}"))?;
        let blob = blob.ok_or("root signature")?;
        dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))
    }
}

#[cfg(test)]
mod shader_cache_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn packed_dxil_checks_the_strict_ratio_and_giant_boundary_independently() {
        let giant = crate::hle::d3d12x::real::GIANT_DXIL;
        assert!(packed_dxil_fallback(giant, giant - 1, 10, giant).is_some());
        assert!(packed_dxil_fallback(giant, giant, 10, giant).is_none());
        assert!(packed_dxil_fallback(100, 10, 10, giant).is_none());
        assert!(packed_dxil_fallback(101, 10, 10, giant).is_some());
        assert!(packed_dxil_fallback(101, 10, 0, giant).is_none());
        assert!(packed_dxil_fallback(giant, giant - 1, 10, 0).is_none());
        assert!(packed_dxil_fallback(usize::MAX, usize::MAX / 2, 10, giant).is_none());
    }

    #[test]
    fn translated_cache_separates_shader_sources_toolchains_and_optimization() {
        let dir = Path::new("cache");
        let path = |source, toolchain, opt| {
            let ident = translated::identity_for(source, &format!("{toolchain}\0{opt}"));
            translated::file(dir, ident, b"console", Stage::Compute, 0, 1, false)
        };
        let base = path("source-a", "dxc-a", "default");
        assert_eq!(base, path("source-a", "dxc-a", "default"));
        assert_ne!(base, path("source-b", "dxc-a", "default"));
        assert_ne!(base, path("source-a", "dxc-b", "default"));
        assert_ne!(base, path("source-a", "dxc-a", "-O0"));
    }

    #[test]
    fn wave_emitter_overrides_bypass_translated_cache() {
        for option in ["XCL_X64_ALL_MASKS", "XCL_WAVE64_SPIN"] {
            assert!(!cache_allowed(Stage::Compute, |name| name == option));
        }
    }

    #[test]
    fn translator_fingerprint_covers_the_dxil_backend() {
        assert_eq!(env!("XCL_DXIL_SOURCE_ID"), xcl_dxil::BUILD_ID);
    }

    #[test]
    fn translated_identity_tracks_sources_and_dxc() {
        let built_source = env!("XCL_TRANSLATOR_SOURCE_ID");
        assert_eq!(built_source.len(), 16);
        assert!(u64::from_str_radix(built_source, 16).is_ok());
        let a = translated::identity_for("source-a", "dxc-a");
        assert_ne!(a, 0);
        assert_eq!(a, translated::identity_for("source-a", "dxc-a"));
        assert_ne!(a, translated::identity_for("source-b", "dxc-a"));
        assert_ne!(a, translated::identity_for("source-a", "dxc-b"));
    }

    /// The loop guard's forms (`XCL_RECOMP_NO_LOOP_REGIONS`, `XCL_LOOP_GUARD_REGIONS`) are other HLSL:
    /// each has its own translations on disk.
    #[test]
    fn translated_identity_tracks_the_loop_guards_form() {
        let lane = translated::identity_for_layout(false, false);
        super::super::NO_LOOP_REGIONS.set(true);
        let wave = translated::identity_for_layout(false, false);
        super::super::LOOP_REGIONS.set(true);
        let wave_too = translated::identity_for_layout(false, false);
        super::super::NO_LOOP_REGIONS.set(false);
        let regions = translated::identity_for_layout(false, false);
        super::super::LOOP_REGIONS.set(false);
        assert_ne!(lane, 0, "the disk cache is on in tests");
        assert!(lane != wave && lane != regions && wave != regions, "{lane:x} {wave:x} {regions:x}");
        assert_eq!(wave, wave_too, "the base form is one form, whatever the regions switch says");
    }

    #[test]
    fn translated_namespace_tracks_every_request_input() {
        let dir = PathBuf::from("cache");
        let base = translated::file(&dir, 42, b"console", Stage::Compute, 3, 2, false);
        let paths = [
            base.clone(),
            translated::file(&dir, 43, b"console", Stage::Compute, 3, 2, false),
            translated::file(&dir, 42, b"consolf", Stage::Compute, 3, 2, false),
            translated::file(&dir, 42, b"console", Stage::Vertex, 3, 2, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 4, 2, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 3, 3, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 3, 2, true),
        ];
        assert_eq!(paths.iter().collect::<std::collections::HashSet<_>>().len(), paths.len());
        assert!(base.starts_with(dir.join("translated").join("v2").join("000000000000002a")));
    }

    /// A translation on disk comes back for the same bytes and semantic identity only: other
    /// bytes of the same length, another translator/toolchain or a cut file are misses.
    #[test]
    fn translations_on_disk_need_the_same_bytes_and_identity() {
        let dir = std::env::temp_dir().join(format!("xcl-translated-{}", std::process::id()));
        let path = dir.join("t.bin");
        let c = Compiled { dxil: vec![1, 2, 3, 4, 5], samplers: vec![([7, 8, 9, 10], true), ([1, 2, 3, 4], false)] };
        translated::store_in(&path, 42, b"console", &c);
        let back = translated::load_in(&path, 42, b"console").expect("stored");
        assert_eq!((back.dxil, back.samplers), (c.dxil.clone(), c.samplers.clone()));
        assert!(translated::load_in(&path, 43, b"console").is_none(), "another identity");
        assert!(translated::load_in(&path, 42, b"consolf").is_none(), "other bytes");
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 1]).unwrap();
        assert!(translated::load_in(&path, 42, b"console").is_none(), "a cut file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `xcl warm-up` finds every translation of one stage and build, whatever its bytes were.
    #[test]
    fn the_translations_of_a_stage_and_build_are_listed() {
        let dir = std::env::temp_dir().join(format!("xcl-translated-all-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (blob, stage, ident, first) in [(&b"cs one"[..], Stage::Compute, 42, 1), (b"cs two", Stage::Compute, 42, 2), (b"ps", Stage::Pixel, 42, 3), (b"old cs", Stage::Compute, 43, 4)] {
            translated::store_in(&translated::file(&dir, ident, blob, stage, 0, 1, false), ident, blob, &compiled(first));
        }
        let mut found: Vec<u8> = translated::all_in(&dir, 42, Stage::Compute).iter().map(|c| c.dxil[0]).collect();
        found.sort();
        assert_eq!(found, [1, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn translated_v2_ignores_valid_entries_outside_its_namespace() {
        let dir = std::env::temp_dir().join(format!("xcl-translated-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let blob = b"console";
        let ident = 42;
        let unnamespaced = dir.join("translated").join(format!("{:016x}_Compute_0_1_0.bin", crate::gpu::shader::fnv64(blob)));
        // Use a valid v2 payload so the miss proves the namespace, rather than header parsing,
        // kept the old root-level location out of consideration.
        let old = compiled(3);
        translated::store_in(&unnamespaced, ident, blob, &old);

        let mut calls = 0;
        let first = translated::cached_in(&dir, ident, blob, Stage::Compute, 0, 1, false, || {
            calls += 1;
            Ok(compiled(7))
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(first.dxil[0], 7);
        assert!(unnamespaced.exists());
        assert!(translated::file(&dir, ident, blob, Stage::Compute, 0, 1, false).exists());

        let second = translated::cached_in(&dir, ident, blob, Stage::Compute, 0, 1, false, || panic!("v2 entry was not reused")).unwrap();
        assert_eq!(second.dxil, first.dxil);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn packed_selection_cache_skips_both_layouts_and_reuses_legacy_under_new_rules() {
        let dir = std::env::temp_dir().join(format!("xcl-packed-selection-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let blob = b"console";
        let stage = Stage::Compute;
        let (legacy_id, first_policy, next_policy) = (21, 22, 23);
        let first = translated::cached_in(&dir, first_policy, blob, stage, 3, 4, true, || {
            let legacy = translated::cached_in(&dir, legacy_id, blob, stage, 3, 4, true, || Ok(compiled(7)))?;
            assert_eq!(legacy.dxil, vec![7; 16]);
            Ok(compiled(8))
        }).unwrap();
        assert_eq!(first.dxil, vec![8; 16]);
        let again = translated::cached_in(&dir, first_policy, blob, stage, 3, 4, true, || panic!("both layouts must be skipped")).unwrap();
        assert_eq!(again.dxil, first.dxil);
        assert_eq!(again.samplers, first.samplers);
        let with_new_rules = translated::cached_in(&dir, next_policy, blob, stage, 3, 4, true, || {
            translated::cached_in(&dir, legacy_id, blob, stage, 3, 4, true, || panic!("reuse the exact legacy bytes"))
        }).unwrap();
        assert_eq!(with_new_rules.dxil, vec![7; 16]);
        assert_eq!(with_new_rules.samplers, vec![([7; 4], false)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn key(hash: u64) -> ShaderKey {
        ShaderKey {
            hash,
            stage: Stage::Compute,
            sampler_base: 0,
            windows: 1,
            split_wave64: false,
            cache_dir: "cache".into(),
        }
    }

    fn compiled(marker: u8) -> Compiled {
        Compiled {
            dxil: vec![marker; 16],
            samplers: vec![([marker as u32; 4], false)],
        }
    }

    #[test]
    fn shader_cache_skips_repeated_translation_and_disk_lookups() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let mut calls = 0;
        let first = cached_shader(&cache, key(1), b"a", || {
            calls += 1;
            Ok(compiled(7))
        })
        .unwrap();
        for _ in 0..100 {
            let next = cached_shader(&cache, key(1), b"a", || {
                panic!("repeated shader reached the compiler path")
            })
            .unwrap();
            assert!(Arc::ptr_eq(&first, &next));
            assert_eq!(next.dxil, first.dxil);
            assert_eq!(next.samplers, first.samplers);
        }
        assert_eq!(calls, 1);
        let mut owned = first.as_ref().clone();
        owned.dxil[0] = 99;
        owned.samplers.clear();
        assert_eq!(first.dxil[0], 7);
        assert_eq!(first.samplers.len(), 1);
    }

    #[test]
    fn shader_cache_checks_context_and_blob_bytes() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let original = key(7);
        let a = cached_shader(&cache, original.clone(), b"a", || Ok(compiled(1))).unwrap();
        for changed in [
            ShaderKey {
                stage: Stage::Vertex,
                ..original.clone()
            },
            ShaderKey {
                sampler_base: 1,
                ..original.clone()
            },
            ShaderKey {
                windows: 2,
                ..original.clone()
            },
            ShaderKey {
                cache_dir: "other cache".into(),
                ..original.clone()
            },
        ] {
            let b = cached_shader(&cache, changed, b"a", || Ok(compiled(2))).unwrap();
            assert!(!Arc::ptr_eq(&a, &b));
            assert_eq!(b.dxil[0], 2);
        }
        let b = cached_shader(&cache, original.clone(), b"b", || Ok(compiled(3))).unwrap();
        assert_eq!(b.dxil[0], 3);
        let a_again =
            cached_shader(&cache, original, b"a", || panic!("original entry lost")).unwrap();
        assert!(Arc::ptr_eq(&a, &a_again));
    }

    #[test]
    fn shader_cache_retries_failed_and_deferred_compiles() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        assert!(cached_shader(&cache, key(1), b"a", || Err("compile failed".into())).is_err());
        let dir = std::env::temp_dir().join(format!("xcl-shader-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (result, missed) = dxc::cached_only(|| {
            cached_shader(&cache, key(1), b"a", || {
                let dxil = dxc::compile_cached_in(&dir, "probe", "main", "cs_6_6")?.0;
                Ok(Compiled {
                    dxil,
                    samplers: Vec::new(),
                })
            })
        });
        assert!(result.is_err() && missed);
        assert!(cache.lock().unwrap().entries.is_empty());
        let t = cached_shader(&cache, key(1), b"a", || Ok(compiled(9))).unwrap();
        assert_eq!(t.dxil[0], 9);
    }

    #[test]
    fn shader_cache_evicts_old_entries_and_skips_oversized_shaders() {
        let t = Arc::new(compiled(1));
        let limit = 2 * ShaderEntry {
            key: key(1),
            blob: b"a".to_vec(),
            shader: t.clone(),
        }
        .bytes();
        let mut cache = ShaderCache::new(limit);
        cache.insert(key(1), b"a", t.clone());
        cache.insert(key(2), b"b", t.clone());
        assert!(cache.get(&key(1), b"a").is_some());
        cache.insert(key(3), b"c", t.clone());
        assert!(cache.get(&key(2), b"b").is_none());
        assert!(cache.get(&key(1), b"a").is_some());
        assert!(cache.get(&key(3), b"c").is_some());
        cache.insert(
            key(4),
            b"d",
            Arc::new(Compiled {
                dxil: vec![0; limit + 1],
                samplers: Vec::new(),
            }),
        );
        assert!(cache.get(&key(4), b"d").is_none());
        assert_eq!(cache.bytes, limit);
        assert_eq!(cache.entries.len(), 2);
        let mut cache = ShaderCache::new(1 << 20);
        for i in 0..129 {
            cache.insert(key(i), b"a", t.clone());
        }
        assert_eq!(cache.entries.len(), 128);
        assert!(cache.get(&key(0), b"a").is_none());
    }

    #[test]
    fn shader_cache_bypasses_debug_overrides() {
        for name in ["XCL_RECOMP_NO_STRUCTURED", "XCL_RECOMP_WHY", "XCL_X64_ALL_MASKS", "XCL_WAVE64_SPIN"] {
            assert!(!cache_allowed(Stage::Compute, |v| v == name));
            assert!(!cache_allowed(Stage::Vertex, |v| v == name));
            assert!(!cache_allowed(Stage::Pixel, |v| v == name));
        }
        for name in [
            "XCL_DEBUG_MAGENTA",
            "XCL_DEBUG_LOD0",
            "XCL_DEBUG_PS_SUB",
            "XCL_DEBUG_PS_EXPR",
        ] {
            assert!(!cache_allowed(Stage::Pixel, |v| v == name));
            assert!(cache_allowed(Stage::Compute, |v| v == name));
        }
        assert!(cache_allowed(Stage::Pixel, |_| false));
    }

    #[test]
    fn shader_cache_probe_does_not_wait_for_an_in_flight_compile() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let dir = std::env::temp_dir().join(format!("xcl-shader-pending-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let worker = scope.spawn(move || {
                cached_shader(cache, key(1), b"a", || {
                    started_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    Ok(compiled(1))
                })
            });
            started_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            let probe = scope.spawn(move || {
                let (result, missed) = dxc::cached_only(|| {
                    cached_shader(cache, key(1), b"a", || {
                        let dxil =
                            dxc::compile_cached_in(&dir, "pending probe", "main", "cs_6_6")?.0;
                        Ok(Compiled {
                            dxil,
                            samplers: Vec::new(),
                        })
                    })
                });
                probe_tx.send(result.is_err() && missed).unwrap();
            });
            let result = probe_rx.recv_timeout(std::time::Duration::from_secs(2));
            release_tx.send(()).unwrap();
            worker.join().unwrap().unwrap();
            probe.join().unwrap();
            assert!(result.unwrap());
        });
    }

    #[test]
    fn shader_cache_does_not_serialize_different_compiles() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let mut threads = Vec::new();
            for hash in [1, 2] {
                let cache = &cache;
                let barrier = &barrier;
                threads.push(scope.spawn(move || {
                    cached_shader(cache, key(hash), b"a", || {
                        barrier.wait();
                        Ok(compiled(1))
                    })
                }));
            }
            for thread in threads {
                assert_eq!(thread.join().unwrap().unwrap().dxil[0], 1);
            }
        });
    }
}

#[cfg(test)]
mod compile_time_bench {
    use super::*;

    #[test]
    #[ignore]
    fn repeated_pso_shader_compile() {
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        let mut code = Vec::new();
        for _ in 0..256 {
            code.extend_from_slice(&CODE[..8]);
        }
        code.extend_from_slice(&CODE[8..]);
        let mut f = fssm(&code, &[(6, 0)]);
        for offset in [0x13C, 0x140, 0x144] {
            f[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        let blob = sample(5, &[f]);
        let first = console_to_dxil(&blob, Stage::Compute, 0).unwrap();
        let mut batches = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            for _ in 0..100 {
                let next = console_to_dxil(&blob, Stage::Compute, 0).unwrap();
                assert_eq!(next.dxil, first.dxil);
                assert_eq!(next.samplers, first.samplers);
            }
            batches.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("100 repeated PSO shader requests, warm DXIL cache, ms: {batches:?}");
    }
}


#[cfg(test)]
mod shader_bench {
    //! `XCL_BENCH_SHADERS=<folder of dumped console shaders (*_cs.bin, *_ps.bin, *_vs.bin)>`:
    //! where a pipeline's time goes (translation, DXC at each optimisation level, the driver)
    //! and how DXC scales over threads.
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    /// (result, seconds, peak private bytes above those at the start)
    fn measured<R: Send>(f: impl FnOnce() -> R + Send) -> (R, f64, u64) {
        let stop = AtomicBool::new(false);
        let base = crate::util::process_commit_bytes();
        let peak = AtomicU64::new(base);
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Relaxed) {
                    peak.fetch_max(crate::util::process_commit_bytes(), Relaxed);
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            let t0 = Instant::now();
            let r = f();
            let secs = t0.elapsed().as_secs_f64();
            stop.store(true, Relaxed);
            (r, secs, peak.load(Relaxed).saturating_sub(base))
        })
    }

    fn device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let (adapter, _) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()?;
            crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).ok()
        }
    }

    struct Shared<T>(T);
    unsafe impl<T> Send for Shared<T> {}

    fn compute_pso(dev: &ID3D12Device, rs: &ID3D12RootSignature, dxil: &[u8]) -> (bool, f64, u64) {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
            ..Default::default()
        };
        let d = Shared((&desc, dev));
        let r = measured(move || {
            let d = d;
            unsafe { d.0 .1.CreateComputePipelineState::<ID3D12PipelineState>(d.0 .0) }.is_ok()
        });
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
        r
    }

    fn median(mut v: Vec<f64>) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// Cold driver time of the console compute programs in `XCL_BENCH_SHADERS` that
    /// `XCL_BENCH_ONLY` names, for their translation ("emit") and each `<name>.<variant>.hlsl`
    /// in `XCL_BENCH_VARIANTS`. Every run compiles HLSL salted with a load neither compiler can
    /// fold, so the driver never has the DXIL cached; `XCL_BENCH_RUNS` runs (3), interleaved
    /// over the variants, and the median. `XCL_BENCH_WINDOWS=n` adds the translation with `n`
    /// memory windows (the others have 4) as variant "w<n>".
    #[test]
    #[ignore]
    fn driver_time_per_variant() {
        let (Some(dir), Some(dev)) = (std::env::var_os("XCL_BENCH_SHADERS"), device()) else {
            println!("XCL_BENCH_SHADERS not set, or no device");
            return;
        };
        let runs: usize = std::env::var("XCL_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let variants = std::env::var_os("XCL_BENCH_VARIANTS").map(PathBuf::from);
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as u32;
        for name in std::env::var("XCL_BENCH_ONLY").unwrap_or_default().split(',').filter(|n| !n.is_empty()) {
            let blob = std::fs::read(PathBuf::from(&dir).join(format!("{name}.bin"))).unwrap();
            let (t, profile) = console_translation(&blob, Stage::Compute, 0, 4, false, false).unwrap();
            let rs = abi_root_signature_with(&dev, true, &t.static_samplers).unwrap();
            let mut sources = vec![("emit".to_string(), t.hlsl.clone())];
            if let Some(n) = std::env::var("XCL_BENCH_WINDOWS").ok().and_then(|v| v.parse::<u32>().ok()) {
                sources.push((format!("w{n}"), console_translation(&blob, Stage::Compute, 0, n, false, false).unwrap().0.hlsl));
            }
            if let Some(v) = &variants {
                let mut found: Vec<_> = std::fs::read_dir(v).unwrap().flatten().filter_map(|e| {
                    let file = e.file_name().to_string_lossy().into_owned();
                    let variant = file.strip_prefix(&format!("{name}."))?.strip_suffix(".hlsl")?.to_string();
                    Some((variant, std::fs::read_to_string(e.path()).unwrap()))
                }).collect();
                found.sort();
                sources.extend(found);
            }
            let mut dxil = vec![Vec::new(); sources.len()];
            let mut dxc_secs = vec![Vec::new(); sources.len()];
            for (k, (_, hlsl)) in sources.iter().enumerate() {
                let main = hlsl.find("void main(").unwrap();
                let body = main + hlsl[main..].find("\n{\n").unwrap() + 3;
                for run in 0..runs {
                    let salt = seed.wrapping_mul(16).wrapping_add(run as u32);
                    let salted = format!("{}    if (io.Load(4092u) == {salt}u) io.Store(4088u, {salt}u);\n{}", &hlsl[..body], &hlsl[body..]);
                    let (r, secs, _) = measured(|| dxc::compile(&salted, "main", profile, &[]));
                    dxil[k].push(r.unwrap());
                    dxc_secs[k].push(secs);
                }
            }
            let mut drv = vec![Vec::new(); sources.len()];
            let mut peak = vec![0u64; sources.len()];
            for run in 0..runs {
                for k in 0..sources.len() {
                    let (ok, secs, p) = compute_pso(&dev, &rs, &dxil[k][run]);
                    assert!(ok, "{name}.{}: CreateComputePipelineState failed", sources[k].0);
                    drv[k].push(secs);
                    peak[k] = peak[k].max(p);
                }
            }
            for (k, (variant, hlsl)) in sources.iter().enumerate() {
                println!(
                    "{name}.{variant}: HLSL {} KB, DXC {:.2} s, DXIL {} KB, driver {:.2} s {:.1?}, +{} MB",
                    hlsl.len() >> 10,
                    median(dxc_secs[k].clone()),
                    dxil[k][0].len() >> 10,
                    median(drv[k].clone()),
                    drv[k],
                    peak[k] >> 20
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn where_pipeline_time_goes() {
        let Some(dir) = std::env::var_os("XCL_BENCH_SHADERS") else {
            println!("XCL_BENCH_SHADERS not set");
            return;
        };
        let n: usize = std::env::var("XCL_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
        let max_kb: u64 = std::env::var("XCL_BENCH_MAX_KB").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX >> 10);
        let only = std::env::var("XCL_BENCH_ONLY").unwrap_or_default();
        let mut files: Vec<(u64, PathBuf, Stage)> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let stage = if name.ends_with("_cs.bin") {
                    Stage::Compute
                } else if name.ends_with("_ps.bin") {
                    Stage::Pixel
                } else if name.ends_with("_vs.bin") {
                    Stage::Vertex
                } else {
                    return None;
                };
                let len = e.metadata().ok()?.len();
                let wanted = only.is_empty() || only.split(',').any(|o| name.starts_with(o));
                (len <= max_kb << 10 && wanted).then(|| (len, e.path(), stage))
            })
            .collect();
        files.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        let take = n.min(files.len());
        let picked: Vec<_> = (0..take).map(|k| files[k * files.len() / take].clone()).collect();
        println!("{} console shaders in the folder, {} picked across their sizes", files.len(), picked.len());
        let dev = device();
        println!("driver timing: {}", if dev.is_some() { "on" } else { "no device" });
        // `XCL_BENCH_LEVELS=0,1`: the two DXC levels to compare (default 3,1)
        let lv: Vec<String> = std::env::var("XCL_BENCH_LEVELS").unwrap_or_else(|_| "3,1".into()).split(',').map(|l| format!("-O{}", l.trim())).collect();
        assert_eq!(lv.len(), 2, "XCL_BENCH_LEVELS: two levels, such as 0,1");
        let flags: Vec<[&str; 1]> = lv.iter().map(|l| [l.as_str()]).collect();
        let levels: [(&str, &[&str]); 2] = [(&lv[0], &flags[0]), (&lv[1], &flags[1])];
        let mut rows: Vec<(&'static str, String)> = Vec::new();
        let mut tr = Vec::new();
        let mut dx = vec![Vec::new(); 2];
        let (mut dx_sum, mut dxil_sum, mut peak_max) = ([0f64; 2], [0usize; 2], [0u64; 2]);
        // the driver at each level: (first creation, the same again) of compute programs new to
        // it at both levels
        let mut drv: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
        let mut not_cold = 0;
        for (i, (_, path, stage)) in picked.iter().enumerate() {
            let blob = std::fs::read(path).unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let t0 = Instant::now();
            let t = console_translation(&blob, *stage, 0, 4, std::env::var_os("XCL_BENCH_SPLIT").is_some(), false);
            let secs_t = t0.elapsed().as_secs_f64();
            let Ok((t, profile)) = t else { continue };
            tr.push(secs_t);
            // `XCL_BENCH_HLSL=<dir>`: write each translation there and compile nothing
            if let Some(d) = std::env::var_os("XCL_BENCH_HLSL") {
                std::fs::write(PathBuf::from(d).join(format!("{name}.hlsl")), &t.hlsl).unwrap();
                continue;
            }
            let mut line = format!("{name}: HLSL {} KB, translate {secs_t:.3} s", t.hlsl.len() >> 10);
            let mut dxil = [Vec::new(), Vec::new()];
            for (k, (label, extra)) in levels.iter().enumerate() {
                let (r, secs, peak) = measured(|| dxc::compile(&t.hlsl, "main", profile, extra));
                let Ok(d) = r else {
                    line += &format!(", DXC {label} failed");
                    continue;
                };
                dx[k].push(secs);
                dx_sum[k] += secs;
                dxil_sum[k] += d.len();
                peak_max[k] = peak_max[k].max(peak);
                line += &format!(", DXC {label} {secs:.2} s {} KB +{} MB", d.len() >> 10, peak >> 20);
                dxil[k] = d;
            }
            if let (Some(dev), Stage::Compute, false) = (&dev, stage, dxil.iter().any(Vec::is_empty)) {
                if let Ok(rs) = abi_root_signature_with(dev, true, &t.static_samplers) {
                    // which level the driver sees first alternates
                    let mut m = [(false, 0f64, 0u64, 0f64); 2];
                    for k in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let (ok, first, peak) = compute_pso(dev, &rs, &dxil[k]);
                        let (_, again, _) = compute_pso(dev, &rs, &dxil[k]);
                        m[k] = (ok, first, peak, again);
                    }
                    // new to the driver: its first creation took well over the cached one
                    let cold = |k: usize| m[k].0 && m[k].1 > 0.02 && m[k].1 > 5.0 * m[k].3;
                    if cold(0) && cold(1) {
                        for k in 0..2 {
                            drv[k].push((m[k].1, m[k].3));
                        }
                    } else {
                        not_cold += 1;
                    }
                    for k in 0..2 {
                        line += &format!(", driver {} {:.3} s +{} MB (again {:.3} s)", levels[k].0, m[k].1, m[k].2 >> 20, m[k].3);
                    }
                }
            }
            println!("{line}");
            rows.push((profile, t.hlsl));
        }
        println!("== {} translated", rows.len());
        println!("translation: sum {:.1} s, median {:.3} s", tr.iter().sum::<f64>(), median(tr.clone()));
        for (k, (label, _)) in levels.iter().enumerate() {
            println!("DXC {label}: sum {:.1} s, median {:.3} s, DXIL {} KB in all, peak commit of one compile +{} MB", dx_sum[k], median(dx[k].clone()), dxil_sum[k] >> 10, peak_max[k] >> 20);
        }
        for (k, (label, _)) in levels.iter().enumerate() {
            let first: Vec<f64> = drv[k].iter().map(|d| d.0).collect();
            println!(
                "driver, {label} DXIL, {} compute programs new to it at both levels: sum {:.1} s, median {:.3} s; from its cache median {:.3} s",
                first.len(),
                first.iter().sum::<f64>(),
                median(first.clone()),
                median(drv[k].iter().map(|d| d.1).collect())
            );
        }
        println!("compute programs left out of the driver sums (cached at a level already, or refused): {not_cold}");
        if std::env::var_os("XCL_BENCH_SCALING").is_none() {
            return;
        }
        for threads in [4usize, 8, 16, dxc::compile_threads()] {
            if threads > dxc::compile_threads() {
                continue;
            }
            let next = std::sync::atomic::AtomicUsize::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Relaxed);
                        let Some((profile, hlsl)) = rows.get(i) else { break };
                        let _ = dxc::compile(hlsl, "main", profile, &[]);
                    });
                }
            });
            let wall = t0.elapsed().as_secs_f64();
            println!("DXC default flags, {threads} at once: {wall:.1} s for {} programs ({:.2}/s; one at a time at -O1 {:.1} s)", rows.len(), rows.len() as f64 / wall, dx_sum[1]);
        }
    }

    /// `XCL_BENCH_SHADERS` as above: each console VS there (the first `XCL_BENCH_MAX`), translated
    /// with `XCL_BENCH_WINDOWS` memory windows (default 4), in a pipeline with `rect_list_gs`
    /// behind it as `translated_graphics_pso` makes a rect list's. The driver must take each one
    /// it takes without the GS. `XCL_BENCH_DEBUG=1`: the debug layer's word on the first refusals.
    #[test]
    #[ignore]
    fn rect_list_gs_pipelines_behind_translated_vs() {
        let Some(dir) = std::env::var_os("XCL_BENCH_SHADERS") else {
            println!("XCL_BENCH_SHADERS not set");
            return;
        };
        if std::env::var_os("XCL_BENCH_DEBUG").is_some() {
            let mut d: Option<ID3D12Debug> = None;
            if unsafe { D3D12GetDebugInterface(&mut d) }.is_ok() {
                if let Some(d) = d {
                    unsafe { d.EnableDebugLayer() };
                }
            }
        }
        let Some(dev) = device() else { return };
        let windows: u32 = std::env::var("XCL_BENCH_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let max: usize = std::env::var("XCL_BENCH_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with("_vs.bin")).collect();
        paths.sort();
        paths.truncate(max);
        let info = windows::core::Interface::cast::<ID3D12InfoQueue>(&dev).ok();
        let create = |rs: &ID3D12RootSignature, vs: &[u8], gs: Option<&[u8]>| -> windows::core::Result<ID3D12PipelineState> {
            let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
            let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                VS: code(vs),
                GS: gs.map_or_else(D3D12_SHADER_BYTECODE::default, code),
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RasterizerState.DepthClipEnable = true.into();
            let r = unsafe { dev.CreateGraphicsPipelineState(&pd) };
            drop(std::mem::ManuallyDrop::into_inner(pd.pRootSignature));
            r
        };
        let say = |what: String| {
            println!("{what}");
            let Some(q) = &info else { return };
            for k in 0..unsafe { q.GetNumStoredMessages() } {
                let mut len = 0usize;
                let _ = unsafe { q.GetMessage(k, None, &mut len) };
                let mut buf = vec![0u64; len.div_ceil(8)];
                let m = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                if unsafe { q.GetMessage(k, Some(m), &mut len) }.is_ok() {
                    println!("  {}", unsafe { std::ffi::CStr::from_ptr((*m).pDescription as *const std::ffi::c_char) }.to_string_lossy());
                }
            }
            unsafe { q.ClearStoredMessages() };
        };
        let (mut tried, mut skipped, mut plain_refused, mut refused) = (0usize, 0usize, 0usize, Vec::new());
        for path in &paths {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let vs = std::fs::read(path).unwrap();
            let (Ok(c), Ok(gs)) = (console_to_dxil_in(&vs, Stage::Vertex, 0, std::env::var_os("XCL_BENCH_SPLIT").is_some(), windows), rect_list_gs(&vs)) else {
                skipped += 1;
                continue;
            };
            let Ok(rs) = abi_root_signature_with(&dev, true, &c.samplers) else {
                skipped += 1;
                continue;
            };
            tried += 1;
            if let Err(e) = create(&rs, &c.dxil, None) {
                if plain_refused < 3 {
                    say(format!("{name}: refused without the GS: {e}"));
                }
                plain_refused += 1;
                continue;
            }
            if let Err(e) = create(&rs, &c.dxil, Some(&gs)) {
                if refused.len() < 3 {
                    say(format!("{name}: refused with the rect list GS: {e}"));
                }
                refused.push(name);
            }
        }
        println!("{} VS, {windows} windows: {tried} tried, {skipped} not translated, {plain_refused} refused without the GS, {} refused only with it", paths.len(), refused.len());
        assert!(refused.is_empty(), "refused with the rect list GS: {:?}", &refused[..refused.len().min(20)]);
    }

    /// `XCL_BENCH_PAIRS=<file of "vs ps" console hashes>` with `XCL_BENCH_SHADERS`: each pair
    /// translated as `translate_graphics_stages` does (PS samplers after the VS's), in a pipeline
    /// with and without the rect list GS. `XCL_BENCH_DEBUG=1`: the debug layer's word.
    #[test]
    #[ignore]
    fn rect_list_gs_pipelines_with_their_pixel_shaders() {
        let (Some(dir), Some(pairs)) = (std::env::var_os("XCL_BENCH_SHADERS"), std::env::var_os("XCL_BENCH_PAIRS")) else {
            println!("XCL_BENCH_SHADERS / XCL_BENCH_PAIRS not set");
            return;
        };
        if std::env::var_os("XCL_BENCH_DEBUG").is_some() {
            let mut d: Option<ID3D12Debug> = None;
            if unsafe { D3D12GetDebugInterface(&mut d) }.is_ok() {
                if let Some(d) = d {
                    unsafe { d.EnableDebugLayer() };
                }
            }
        }
        let Some(dev) = device() else { return };
        let windows: u32 = std::env::var("XCL_BENCH_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let split = std::env::var_os("XCL_BENCH_SPLIT").is_some();
        let info = windows::core::Interface::cast::<ID3D12InfoQueue>(&dev).ok();
        let say = |what: String| {
            println!("{what}");
            let Some(q) = &info else { return };
            for k in 0..unsafe { q.GetNumStoredMessages() } {
                let mut len = 0usize;
                let _ = unsafe { q.GetMessage(k, None, &mut len) };
                let mut buf = vec![0u64; len.div_ceil(8)];
                let m = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                if unsafe { q.GetMessage(k, Some(m), &mut len) }.is_ok() {
                    println!("  {}", unsafe { std::ffi::CStr::from_ptr((*m).pDescription as *const std::ffi::c_char) }.to_string_lossy());
                }
            }
            unsafe { q.ClearStoredMessages() };
        };
        let dir = PathBuf::from(dir);
        let (mut tried, mut missing, mut plain, mut only_gs) = (0usize, 0usize, 0usize, 0usize);
        for line in std::fs::read_to_string(pairs).unwrap().lines() {
            let mut it = line.split_whitespace();
            let (Some(v), Some(p)) = (it.next(), it.next()) else { continue };
            let (Ok(vs), Ok(ps)) = (std::fs::read(dir.join(format!("{v}_vs.bin"))), std::fs::read(dir.join(format!("{p}_ps.bin")))) else {
                missing += 1;
                continue;
            };
            let Ok(vc) = console_to_dxil_in(&vs, Stage::Vertex, 0, split, windows) else { missing += 1; continue };
            let Ok(pc) = console_to_dxil_in(&ps, Stage::Pixel, vc.samplers.len() as u32, split, windows) else { missing += 1; continue };
            let Ok(gs) = rect_list_gs(&vs) else { missing += 1; continue };
            let mut samplers = vc.samplers.clone();
            samplers.extend(pc.samplers.iter().cloned());
            let Ok(rs) = abi_root_signature_with(&dev, true, &samplers) else { missing += 1; continue };
            tried += 1;
            let create = |gs: Option<&[u8]>| -> windows::core::Result<ID3D12PipelineState> {
                let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
                let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    VS: code(&vc.dxil),
                    PS: code(&pc.dxil),
                    GS: gs.map_or_else(D3D12_SHADER_BYTECODE::default, code),
                    SampleMask: u32::MAX,
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                pd.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
                pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pd.RasterizerState.DepthClipEnable = true.into();
                let r = unsafe { dev.CreateGraphicsPipelineState(&pd) };
                drop(std::mem::ManuallyDrop::into_inner(pd.pRootSignature));
                r
            };
            if let Err(e) = create(None) {
                if plain < 3 {
                    say(format!("VS {v} PS {p}: refused without the GS: {e}"));
                }
                plain += 1;
                continue;
            }
            if let Err(e) = create(Some(&gs)) {
                if only_gs < 5 {
                    say(format!("VS {v} PS {p}: refused with the rect list GS: {e}"));
                }
                only_gs += 1;
            }
        }
        println!("pairs: {tried} tried, {missing} not found or not translated, {plain} refused without the GS, {only_gs} refused only with it");
    }
}
