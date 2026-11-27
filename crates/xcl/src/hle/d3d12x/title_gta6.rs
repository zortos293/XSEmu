//! GTA VI (TitleId 7952987A) title patches. The user, 2026-11-26 21:23Z: title-specific patches
//! are allowed while a general fix is out of reach. Each one is gated on the title id, has its own
//! off switch, and logs when it engages; `XCL_TITLE_PATCHES=0` turns all of them off.
//!
//! - Present chain first (`XCL_TITLE_PRESENT_FIRST=0`): the pipelines the title names `FFX_*`
//!   (FidelityFX: the 27 FSR4 passes, RCAS, SPD auto-exposure; play-91b and play-92 name no other
//!   pipeline) are built when named, before every other build, as compiles a bind waits for, and
//!   their DXC work goes first in the title lane. play-92 had 2 of the 29 built at 451 s, behind
//!   1,547 pending ones. General fix: rank builds by what feeds the present (the frame's last
//!   passes before Present), for every title.
//! - Early shaders (`XCL_TITLE_EARLY=0`): pipelines with one of the title's opening-scene and
//!   decoration-bake shaders are built when made, the same way. General fix: the manifest's
//!   record of what earlier runs drew first, which a cold cache does not have yet.
//! - Ray passes (`XCL_TITLE_RT_SKIP=0`), the title layer of ray tracing off (`raytracing`; nothing
//!   while `XCL_RAYTRACING=1`): the title's ray passes that do not traverse a structure themselves –
//!   ray-hit resolve (1,000 of the dumped compute shaders), screen-probe ray and probe activation
//!   (the three compute giants play-92 could never build, up to 16.7 MB of DXIL), ray binning,
//!   budgeting and sorting, ray-traced visibility and shadow passes – are left out unbuilt. The
//!   general layer catches the tracing passes and the BVH builder. General fix: inline ray tracing
//!   the driver can compile in bounded memory, then ray tracing on. With `XCL_RT_GATE=probes`
//!   (every ray a miss otherwise, as "miss") only the ray activation and the ray-hit resolve are
//!   left out (`probes_gate_pass`), as cool_fox does: its f4164f86 records that with
//!   `CS_ScreenProbes_ActivateRays` running and no rays behind it "the whole picture lost light,
//!   shadows and reflections"; the probe activation and the probes' temporal resolve run. Off
//!   with `XCL_TITLE_RT_SKIP=0` or any other `XCL_RT_GATE`.
//! - Watchdog (`XCL_TITLE_GTA6_WATCHDOG=0`): the title's watchdog thread ends the process (a store
//!   to address 0) when its main thread's heartbeat has stood for 30 one-second ticks. play-95
//!   ended so at 310 s: the main thread had waited 30.8 s for a frame XCL held up. While the
//!   heartbeat has stood for 10 s, the title's own count of long tasks is raised by one (the title
//!   raises it around its long tasks; the watchdog then skips the heartbeat check), and it is
//!   given back when the heartbeat moves or has stood for 300 s: a real hang still ends the run,
//!   30 s later. The count is found through the watchdog loop's own instructions (`WATCHDOG`), and
//!   nothing is written when they, the counters' layout or their values are not as build 1.0.2.1
//!   has them. This is the one patch that writes the title's memory (`Image`). General fix: no
//!   frame takes over 30 s (a bind's promotion off the backlog lock, admission that does not
//!   starve background builds).
//! - Stand-in indirect light (`XCL_GTA6_AMBIENT`, off unless it holds a level; `title_gta6_ambient`
//!   has the patch and its other switches): with ray tracing off the title's screen probes leave
//!   indoor surfaces black, and the diffuse light is raised to an estimate taken from the frame
//!   right after the title's program that resolves it. General fix: inline ray tracing the driver
//!   can compile in bounded memory, then ray tracing on.
//!
//! The ray-hit resolve and ray activation names and the early shader list come from codec_d's
//! cool_fox (the CGBJ re-upload), commit f4b48ad766ddbec1a1867022cd42f7645b0dd55e: crates/
//! victory-gpu/src/pipeline/make.rs:53-101 and crates/victory-gpu/src/draw.rs:244-262; the ray
//! binning names follow DexNom RayTime's --no-rt selection (crates/victory-gpu/src/pipeline/
//! ray_tracing.rs). That code is licensed under the GNU General Public License version 3; this
//! file carries that material under the same licence. cool_fox names shaders by the first 16 hex
//! digits of their container's BLAKE3; the ids below are XCL's FNV-1a ids of the same containers,
//! matched by BLAKE3 over this title's shader dump (all 15 found). Not for a public or shared build.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;

const TITLE_ID: &str = "7952987A";

#[derive(Clone, Copy)]
pub(super) enum Patch {
    PresentFirst,
    Early,
    RtSkip,
    Watchdog,
    Ambient,
}

const PATCHES: usize = 5;

impl Patch {
    pub(super) const fn switch(self) -> &'static str {
        match self {
            Patch::PresentFirst => "XCL_TITLE_PRESENT_FIRST",
            Patch::Early => "XCL_TITLE_EARLY",
            Patch::RtSkip => "XCL_TITLE_RT_SKIP",
            Patch::Watchdog => "XCL_TITLE_GTA6_WATCHDOG",
            Patch::Ambient => "XCL_GTA6_AMBIENT",
        }
    }

    pub(super) const fn what(self) -> &'static str {
        match self {
            Patch::PresentFirst => "present chain first",
            Patch::Early => "early shaders",
            Patch::RtSkip => "ray passes left out",
            Patch::Watchdog => "watchdog held to 300 s",
            Patch::Ambient => "stand-in indirect light",
        }
    }

    /// Whether the patch is on with its switch holding `value`: on unless that is 0, but the
    /// stand-in for the indirect light changes the picture and is on only when its switch holds a
    /// level (`title_gta6_ambient::set`).
    fn set(self, value: Option<&str>) -> bool {
        match self {
            Patch::Ambient => super::title_gta6_ambient::set(value),
            _ => switch_on(value),
        }
    }

    /// What its switch holds when the patch is off, for the log.
    const fn off(self) -> &'static str {
        match self {
            Patch::Ambient => " unset or 0; a share of the estimate such as 0.8 turns it on",
            _ => "=0",
        }
    }
}

/// cool_fox's six opening-scene producers (compute), its seven decoration-bake pixel shaders, and
/// the bake's two geometry shaders (vertex f0e5138d…, mesh 2bdc6171…), as XCL ids.
const EARLY: [u64; 15] = [
    0x0ec4_be5d_d682_1455,
    0xb177_5db5_ce97_5d48,
    0xd979_b1c9_1334_81ab,
    0xbd96_8d88_fabf_544f,
    0x9dc8_5ad3_97cb_affe,
    0x1a28_fd84_7859_90f1,
    0xa156_79f5_e553_1811,
    0xaf40_7345_2a99_3d01,
    0x890a_c9be_a44d_cd5f,
    0xd85c_f42f_d3ed_4138,
    0x81c5_72f7_7529_17e2,
    0x9ffd_a50d_9934_e444,
    0x71f3_141f_4bef_a427,
    0xf0e5_138d_b15d_6d83,
    0x2bdc_6171_bf33_7731,
];

/// The title's ray passes by the name in their container (a prefix of the shader's name): only
/// those that trace acceleration structures, read their hits, or start their rays. The BVH
/// traces are caught by their code too (`raytracing`); the probe activations were never built in
/// any run (giants of 19 to 21 KB of console code). Not here, so they run as before ray tracing
/// was off (play-93b ran them): the ray-request, binning, sorting, budgeting, clearing and
/// classification passes and the screen probes' voxel and fallback tracing, temporal resolve and
/// upscale. play-96 left them out too; a skipped dispatch also marks the buffers it binds as
/// unwritten, so the indirect draws that read them are refused.
const RAY_PASSES: [&[u8]; 5] = [
    b"CS_RaytracingResolve",
    b"CS_ScreenProbes_ActivateRays",
    b"CS_ScreenProbes_ActivateProbes",
    b"CS_ScreenProbes_TraceRays_RayBinning_BVH",
    b"CS_ScreenProbes_TraceRays_BVH",
];

fn switch_on(value: Option<&str>) -> bool {
    value.map_or(true, |v| v.trim() != "0")
}

fn is_title(id: &str) -> bool {
    let id = id.trim();
    id.strip_prefix("0x").or_else(|| id.strip_prefix("0X")).unwrap_or(id).eq_ignore_ascii_case(TITLE_ID)
}

fn title() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !is_title(&crate::state::config().title_id) {
            return false;
        }
        let on = switch_on(std::env::var("XCL_TITLE_PATCHES").ok().as_deref());
        crate::xlog_info!("D3D12.X: GTA VI title patches {}", if on { "on (XCL_TITLE_PATCHES=0 turns them off)" } else { "off (XCL_TITLE_PATCHES=0)" });
        on
    })
}

pub(super) fn on(patch: Patch) -> bool {
    static ON: [OnceLock<bool>; PATCHES] = [const { OnceLock::new() }; PATCHES];
    *ON[patch as usize].get_or_init(|| {
        let on = title() && patch.set(std::env::var(patch.switch()).ok().as_deref());
        if title() && !on {
            crate::xlog_info!("D3D12.X: title patch {} off ({}{})", patch.what(), patch.switch(), patch.off());
        }
        on
    })
}

/// Logged the first time, then at powers of two.
fn engaged(patch: Patch, detail: impl FnOnce() -> String) {
    static SEEN: [AtomicUsize; PATCHES] = [const { AtomicUsize::new(0) }; PATCHES];
    let n = SEEN[patch as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: title patch {} ({}=0 turns it off): {} (#{n})", patch.what(), patch.switch(), detail());
    }
}

/// The two ray passes cool_fox leaves out with rays off (DoubleGhosty/cool_fox bd8b4f52,
/// crates/victory-gpu/src/pipeline/make.rs:79-104, GPL-3.0-or-later; idle_without_rays in
/// f4164f86 crates/victory-gpu/src/pipeline.rs): the screen probes' ray activation and the
/// ray-hit resolve. The probe activation, tracing (every ray a miss) and temporal resolve run.
const PROBE_GATE_PASSES: [&[u8]; 2] = [b"CS_ScreenProbes_ActivateRays", b"CS_RaytracingResolve"];

/// Ray tracing off with `XCL_RT_GATE=probes` (`raytracing::classify_as`): a compute shader of
/// `PROBE_GATE_PASSES`, left out unbuilt. Title-specific (by the pass names in GTA VI's
/// containers, only with the title's id; `XCL_TITLE_RT_SKIP=0` turns it off). General fix: inline
/// ray tracing the driver can compile in bounded memory, then ray tracing on.
pub fn probes_gate_pass(container: &[u8]) -> Option<&'static [u8]> {
    let name = PROBE_GATE_PASSES.iter().copied().find(|name| container.windows(name.len()).any(|w| w == *name))?;
    if !title() || !on(Patch::RtSkip) {
        return None;
    }
    engaged(Patch::RtSkip, || format!("{} left out unbuilt (XCL_RT_GATE=probes)", String::from_utf8_lossy(name)));
    Some(name)
}

pub(super) fn ray_pass(container: &[u8]) -> Option<&'static [u8]> {
    RAY_PASSES.iter().copied().find(|name| container.windows(name.len()).any(|w| w == *name))
}

/// Ray tracing off with every ray a miss (`raytracing::miss_pass`): a compute shader of the title's
/// ray passes, built behind every other pipeline. Title-specific (by the pass names in GTA VI's
/// containers, only with the title's id; `XCL_RT_PASS_LAST=0` turns it off with the general part):
/// a general fix would order builds by what the frame's visible output reads.
pub fn miss_ray_pass(container: &[u8]) -> Option<&'static [u8]> {
    if !title() {
        return None;
    }
    ray_pass(container)
}

/// Ray tracing off (`raytracing::left_out`): a compute shader of the title's ray passes.
pub fn rt_family(container: &[u8]) -> Option<&'static [u8]> {
    let name = ray_pass(container)?;
    if !title() || !on(Patch::RtSkip) {
        return None;
    }
    engaged(Patch::RtSkip, || format!("{} left out unbuilt", String::from_utf8_lossy(name)));
    Some(name)
}

/// The early shader (`EARLY`) among a pipeline's shaders, if one is.
pub fn early_shader<'a>(shaders: impl Iterator<Item = &'a [u8]>) -> Option<u64> {
    if !title() || !on(Patch::Early) {
        return None;
    }
    shaders.map(crate::gpu::shader::fnv64).find(|id| EARLY.contains(id))
}

/// The title named pipeline `pso` (SetName): the present chain is built now, first.
pub fn named(pso: usize, name: &str) {
    if name.starts_with("FFX_") && title() && on(Patch::PresentFirst) && super::pso_async::build_first(pso) {
        engaged(Patch::PresentFirst, || format!("'{name}' ({pso:#x}) built now, before every other build"));
    }
}

/// `pso` has an early shader (`early_shader`): built now, first.
pub fn build_early(pso: usize, shader: u64) {
    if super::pso_async::build_first(pso) {
        engaged(Patch::Early, || format!("pipeline {pso:#x} with shader {shader:016x} built now, before every other build"));
    }
}

/// The title's main image as the loader mapped it (`modules::main_module`): what a patch that
/// reads or writes the title's own memory goes through. Nothing is read or written outside the
/// image, or where Windows does not report committed memory of the kind asked for.
#[derive(Clone, Copy)]
struct Image {
    base: usize,
    size: usize,
}

#[derive(Clone, Copy)]
enum Access {
    Code,
    Data,
}

impl Image {
    fn main() -> Option<Image> {
        crate::modules::main_module().map(|m| Image { base: m.base, size: m.size })
    }

    /// The end of the memory region that holds `at`, no further than the image's end, and whether
    /// the region is committed code and committed writable data (None: `at` is outside the image).
    fn region(self, at: usize) -> Option<(usize, bool, bool)> {
        use windows_sys::Win32::System::Memory::{
            VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READWRITE, PAGE_WRITECOPY,
        };
        let end = self.base.saturating_add(self.size);
        if at < self.base || at >= end {
            return None;
        }
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const std::ffi::c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
            return None;
        }
        let usable = m.State == MEM_COMMIT && m.Protect & PAGE_GUARD == 0;
        let kind = m.Protect & 0xff;
        let code = usable && [PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY].contains(&kind);
        let data = usable && [PAGE_READWRITE, PAGE_WRITECOPY, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY].contains(&kind);
        Some(((m.BaseAddress as usize).saturating_add(m.RegionSize).min(end), code, data))
    }

    /// The bytes from `at` to the end of its region when the region allows `access`, else 0.
    fn span(self, at: usize, access: Access) -> usize {
        match (self.region(at), access) {
            (Some((end, true, _)), Access::Code) | (Some((end, _, true)), Access::Data) => end.saturating_sub(at),
            _ => 0,
        }
    }

    /// `len` bytes of the title's code at `at`, when all of them are there.
    fn code(self, at: usize, len: usize) -> Option<&'static [u8]> {
        (self.span(at, Access::Code) >= len).then(|| unsafe { std::slice::from_raw_parts(at as *const u8, len) })
    }

    /// Where the title's code holds `pattern` (one that lies across two regions is not found).
    fn find(self, pattern: &[i16]) -> Vec<usize> {
        let (mut at, mut found) = (self.base, Vec::new());
        while let Some((end, code, _)) = self.region(at) {
            if end <= at {
                break;
            }
            if code {
                let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, end - at) };
                found.extend(matches(bytes, pattern).map(|i| at + i));
            }
            at = end;
        }
        found
    }
}

/// Any byte, in a pattern of code bytes.
const ANY: i16 = -1;

fn holds(bytes: &[u8], pattern: &[i16]) -> bool {
    bytes.len() >= pattern.len() && pattern.iter().zip(bytes).all(|(p, b)| *p == ANY || *p == i16::from(*b))
}

fn matches<'a>(bytes: &'a [u8], pattern: &'a [i16]) -> impl Iterator<Item = usize> + 'a {
    bytes.windows(pattern.len()).enumerate().filter(move |(_, w)| holds(w, pattern)).map(|(i, _)| i)
}

/// What an instruction at `at + ..` addresses through the RIP-relative displacement that is its
/// last four bytes, `bytes[disp..disp + 4]` (`bytes` start at `at`).
fn rip_target(at: usize, bytes: &[u8], disp: usize) -> usize {
    let rel = i32::from_le_bytes([bytes[disp], bytes[disp + 1], bytes[disp + 2], bytes[disp + 3]]);
    (at + disp + 4).wrapping_add_signed(rel as isize)
}

/// Where build 1.0.2.1 has `WATCHDOG` in its image; the title's code is searched when it is not there.
const WATCHDOG_RVA: usize = 0x2215b50;

/// The title's watchdog loop where a tick finds the watchdog paused and then asks for long tasks:
/// `mov cl, [paused]; test cl, 1; je +17; mov [heartbeat], rax; mov ebp, 30; mov r15, rax; jmp ..;
/// mov eax, [long tasks]; test eax, eax`. The 30 is its count of one-second ticks.
const WATCHDOG: [i16; 36] = [
    0x8a, 0x0d, ANY, ANY, ANY, ANY, 0xf6, 0xc1, 0x01, 0x74, 0x11, 0x48, 0x89, 0x05, ANY, ANY, ANY, ANY, 0xbd, 0x1e, 0x00, 0x00, 0x00, 0x49, 0x89, 0xc7, 0xeb, ANY, 0x8b, 0x05, ANY, ANY,
    ANY, ANY, 0x85, 0xc0,
];

/// The title's watchdog state: a byte that pauses it, its count of long tasks under way (the title
/// raises it around them with `lock inc` and `lock dec`; while it is not 0 the watchdog skips its
/// heartbeat check and restarts its 30 ticks), and the main thread's heartbeat (a time stamp).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Watchdog {
    paused: usize,
    long_tasks: usize,
    heartbeat: usize,
}

/// What the `WATCHDOG` in `bytes` at `at` addresses, when the three lie as build 1.0.2.1 lays them
/// out: the count four bytes after the pause byte, the heartbeat four after the count.
fn watchdog_at(at: usize, bytes: &[u8]) -> Option<Watchdog> {
    if !holds(bytes, &WATCHDOG) {
        return None;
    }
    let w = Watchdog { paused: rip_target(at, bytes, 2), heartbeat: rip_target(at, bytes, 14), long_tasks: rip_target(at, bytes, 30) };
    (w.heartbeat % 8 == 0 && w.long_tasks.checked_add(4) == Some(w.heartbeat) && w.paused.checked_add(4) == Some(w.long_tasks)).then_some(w)
}

/// A count of long tasks above this is not the title's count: nothing is written.
const LONG_TASKS_MOST: u32 = 64;

/// The title's watchdog state in `image`, and whether its loop was where build 1.0.2.1 has it; or
/// why nothing is written.
fn watchdog(image: Image) -> Result<(Watchdog, bool), String> {
    let known = image.base.saturating_add(WATCHDOG_RVA);
    let there = image.code(known, WATCHDOG.len()).is_some_and(|bytes| holds(bytes, &WATCHDOG));
    let sites = if there { vec![known] } else { image.find(&WATCHDOG) };
    let found: Vec<Watchdog> = sites.iter().filter_map(|&at| watchdog_at(at, image.code(at, WATCHDOG.len())?)).collect();
    let Some(&w) = found.first() else {
        return Err(format!("the watchdog loop of build 1.0.2.1 is not in the title's code ({} places hold its bytes, none with the counters laid out as expected)", sites.len()));
    };
    if found.len() != sites.len() || found.iter().any(|other| *other != w) {
        return Err(format!("{} places in the title's code hold the watchdog loop's bytes and they do not address the same counters", sites.len()));
    }
    if image.span(w.paused, Access::Data) < 16 {
        return Err(format!("the counters at {:#x} (image base {:#x}) are not committed writable memory of the image", w.paused, image.base));
    }
    let (paused, long_tasks) = unsafe { (AtomicU8::from_ptr(w.paused as *mut u8).load(Ordering::Relaxed), AtomicU32::from_ptr(w.long_tasks as *mut u32).load(Ordering::Relaxed)) };
    if paused > 1 || long_tasks > LONG_TASKS_MOST {
        return Err(format!("the pause byte holds {paused:#x} and the count of long tasks {long_tasks:#x}: not what the title keeps there"));
    }
    Ok((w, there))
}

const WATCHDOG_POLL: std::time::Duration = std::time::Duration::from_secs(1);
/// The heartbeat has stood this long: the count of long tasks is raised (the title's watchdog ends
/// the process at 30 s).
const WATCHDOG_HOLD_AFTER: std::time::Duration = std::time::Duration::from_secs(10);
/// The heartbeat has stood this long: the count is given back, and the title's watchdog ends the
/// process 30 s later.
const WATCHDOG_LIMIT: std::time::Duration = std::time::Duration::from_secs(300);

/// The loader is done with the title's main image (mapped, its imports resolved, its sections under
/// their protections, so its code reads as code): the watchdog patch watches the heartbeat from
/// its own thread from now on.
pub fn image_ready() {
    let _ = std::thread::Builder::new().name("xcl-title-watchdog".into()).spawn(watchdog_guard);
}

/// Raises the title's count of long tasks while its heartbeat has stood for `WATCHDOG_HOLD_AFTER`
/// to `WATCHDOG_LIMIT` (a heartbeat of 0 is one the title has not started), and gives it back
/// otherwise.
fn watchdog_guard() {
    if !on(Patch::Watchdog) {
        return;
    }
    let Some(image) = Image::main() else { return };
    let (w, there) = match watchdog(image) {
        Ok(found) => found,
        Err(why) => {
            crate::xlog_warn!("D3D12.X: title patch {} not applied, nothing is written ({}=0 turns it off): {why}", Patch::Watchdog.what(), Patch::Watchdog.switch());
            return;
        }
    };
    crate::xlog_info!(
        "D3D12.X: title patch {} watches the title's heartbeat at image+{:#x}; its count of long tasks is at image+{:#x} (watchdog loop {})",
        Patch::Watchdog.what(),
        w.heartbeat - image.base,
        w.long_tasks - image.base,
        if there { "where build 1.0.2.1 has it" } else { "found by its bytes, not where build 1.0.2.1 has it" }
    );
    let (count, beat) = unsafe { (AtomicU32::from_ptr(w.long_tasks as *mut u32), AtomicU64::from_ptr(w.heartbeat as *mut u64)) };
    let (mut last, mut since, mut held) = (beat.load(Ordering::Relaxed), std::time::Instant::now(), false);
    loop {
        std::thread::sleep(WATCHDOG_POLL);
        let now = beat.load(Ordering::Relaxed);
        if now != last || now == 0 {
            (last, since) = (now, std::time::Instant::now());
        }
        let stood = since.elapsed();
        let hold = stood >= WATCHDOG_HOLD_AFTER && stood < WATCHDOG_LIMIT;
        if hold == held {
            continue;
        }
        if image.span(w.long_tasks, Access::Data) < 4 || count.load(Ordering::Relaxed) > LONG_TASKS_MOST {
            crate::xlog_warn!(
                "D3D12.X: title patch {} stops, nothing more is written: the count of long tasks at image+{:#x} is no longer writable or holds {:#x}",
                Patch::Watchdog.what(),
                w.long_tasks - image.base,
                count.load(Ordering::Relaxed)
            );
            return;
        }
        if hold {
            let n = count.fetch_add(1, Ordering::SeqCst) + 1;
            engaged(Patch::Watchdog, || {
                format!(
                    "the main thread's heartbeat has stood for {} s: the title's count of long tasks raised to {n}, its watchdog waits until the heartbeat has stood for {} s",
                    stood.as_secs(),
                    WATCHDOG_LIMIT.as_secs()
                )
            });
        } else {
            let mut n = count.load(Ordering::SeqCst);
            while n != 0 {
                match count.compare_exchange_weak(n, n - 1, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(now) => n = now,
                }
            }
            if stood >= WATCHDOG_LIMIT {
                crate::xlog_warn!(
                    "D3D12.X: title patch {}: the main thread's heartbeat has stood for {} s – the count of long tasks is given back, and the title's own watchdog ends the run in 30 s",
                    Patch::Watchdog.what(),
                    stood.as_secs()
                );
            }
        }
        held = hold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_title_id_and_the_switches_parse() {
        assert!(is_title("7952987A"));
        assert!(is_title("0x7952987a"));
        assert!(!is_title(""));
        assert!(!is_title("7952987B"));
        assert!(switch_on(None));
        assert!(switch_on(Some("1")));
        assert!(!switch_on(Some(" 0 ")));
    }

    #[test]
    fn the_title_s_ray_passes_are_found_by_name() {
        for name in [
            &b"\x00\x01..CS_RaytracingResolve_1234abcd_Wrapped\x00"[..],
            b"xxCS_ScreenProbes_ActivateRays_d60186e6_Wrapped",
            b"CS_ScreenProbes_ActivateProbes_9083c094_Wrapped",
            b"CS_ScreenProbes_ActivateProbesTarget_534f271f_Wrapped",
            b"CS_ScreenProbes_TraceRays_RayBinning_BVH_Shared_ShaderBinning_WithFar",
        ] {
            assert!(ray_pass(name).is_some(), "{}", String::from_utf8_lossy(name));
        }
        for name in [
            &b"CS_ScreenProbes_TraceRays_Voxels"[..],
            b"CS_ScreenProbes_TraceRays_Fallback",
            b"CS_ScreenProbes_TemporalResolve",
            b"CS_TraceShadowRays_ScreenSpaceOnlyFullRes",
            b"CS_GatherRaymarchedLocalLightsClassified27_0f05597d_Wrapped",
            b"CS_Tree_BatchCullReflect",
            b"CS_GenerateRayBinOffsets_01234567_Wrapped",
            b"CS_RayBudgeting_Cull_RayBinning_",
            b"CS_ScreenProbes_EvaluateRayRequests_RayBinning_19e9e6fd_Wrapped",
            b"CS_ScreenProbes_GenerateRayBins_",
            b"CS_SortRayTiles",
            b"CS_WriteRTClassificationResults",
            b"CS_ClearRayCounts",
            b"CS_RayPrePass",
        ] {
            assert_eq!(ray_pass(name), None, "{}", String::from_utf8_lossy(name));
        }
    }

    #[test]
    fn the_early_list_holds_fifteen_distinct_ids() {
        let distinct: std::collections::HashSet<_> = EARLY.iter().collect();
        assert_eq!(distinct.len(), 15);
    }

    #[test]
    fn the_watchdog_loop_names_its_counters() {
        let first: [u8; 36] = [
            0x8a, 0x0d, 0xda, 0x07, 0xfa, 0x04, 0xf6, 0xc1, 0x01, 0x74, 0x11, 0x48, 0x89, 0x05, 0xd6, 0x07, 0xfa, 0x04, 0xbd, 0x1e, 0x00, 0x00, 0x00, 0x49, 0x89, 0xc7, 0xeb, 0x8a, 0x8b, 0x05, 0xc2,
            0x07, 0xfa, 0x04, 0x85, 0xc0,
        ];
        let second: [u8; 36] = [
            0x8a, 0x0d, 0x36, 0x06, 0xfa, 0x04, 0xf6, 0xc1, 0x01, 0x74, 0x11, 0x48, 0x89, 0x05, 0x32, 0x06, 0xfa, 0x04, 0xbd, 0x1e, 0x00, 0x00, 0x00, 0x49, 0x89, 0xc7, 0xeb, 0x8c, 0x8b, 0x05, 0x1e,
            0x06, 0xfa, 0x04, 0x85, 0xc0,
        ];
        let w = Watchdog { paused: 0x1471b6330, long_tasks: 0x1471b6334, heartbeat: 0x1471b6338 };
        assert_eq!(watchdog_at(0x142215b50, &first), Some(w));
        assert_eq!(watchdog_at(0x142215cf4, &second), Some(w));
        assert_eq!(watchdog_at(0x152215b50, &first), Some(Watchdog { paused: 0x1571b6330, long_tasks: 0x1571b6334, heartbeat: 0x1571b6338 }), "a relocated image");
        let mut sixty = first;
        sixty[19] = 0x3c;
        assert_eq!(watchdog_at(0x142215b50, &sixty), None, "another count of ticks");
        let mut elsewhere = first;
        elsewhere[30] = 0xc6;
        assert_eq!(watchdog_at(0x142215b50, &elsewhere), None, "a count that is not four bytes before the heartbeat");
        assert_eq!(watchdog_at(0x142215b50, &first[..35]), None);
        let mut code = vec![0xcc_u8; 96];
        code[7..43].copy_from_slice(&first);
        code[50..86].copy_from_slice(&second);
        assert_eq!(matches(&code, &WATCHDOG).collect::<Vec<_>>(), [7, 50]);
    }
}
