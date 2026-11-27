//! A second compute translator: crates/xcl-cgbj under its XCL binding (CAPA-86, M1).
//!
//! Off unless `XCL_SHADER_EMITTER` asks for it. Then real.rs tries [`compute`] before today's
//! translator, and a program CGBJ refuses goes to today's path. This folder is outside the
//! translator fingerprint (crates/xcl/build.rs), so changing it keeps the HLSL caches warm.
//!
//! Switches (read once):
//! - `XCL_SHADER_EMITTER`: `xcl` (default), `cgbj`, or `cs:cgbj,ps:xcl,vs:xcl,ms:xcl`. Only `cs` is honoured.
//! - `XCL_SHADER_EMITTER_ONLY` / `XCL_SHADER_EMITTER_SKIP`: blob hashes (16 hex digits, as logged).
//! - `XCL_CGBJ_REFUSED`: `xcl` (default) or `skip` (refused programs stay untranslated, A/B only).
//! - `XCL_CGBJ_WAVE`: `exact` (default) or `approx`.
//! - `XCL_CGBJ_VALIDATE`: `1` (default) or `0`.
//! - `XCL_TRANSLATED_CACHE=0`: no disk cache (the memory map stays).

mod cache;
mod switches;
mod validate;

use crate::gpu::recomp::pso::Compiled;
use cache::{Entry, Key};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use switches::{Emitter, Refused, Switches};
use xcl_cgbj::xcl::{XclConfig, BINDING_VERSION};

/// What [`compute`] came to.
pub enum Outcome {
    Compiled(Compiled),
    /// Not CGBJ's program: today's translator takes it.
    Fallback,
    /// Inside the cached-only probe with nothing cached: `dxc::defer` was called.
    Deferred,
    /// Refused with `XCL_CGBJ_REFUSED=skip`: it stays untranslated.
    Skipped,
}

fn switches() -> Option<&'static Switches> {
    static S: OnceLock<Option<Switches>> = OnceLock::new();
    S.get_or_init(|| match switches::parse(|name| std::env::var(name).ok()) {
        Ok(s) => {
            for stage in &s.unavailable {
                crate::xlog_warn!("XCL_SHADER_EMITTER asks for cgbj for {stage}: not available yet, xcl translates it");
            }
            (s.compute == Emitter::Cgbj).then(|| {
                crate::xlog_info!(
                    "cgbj: compute programs go to xcl-cgbj first (binding v{BINDING_VERSION}, wave {}, validate {}, refused -> {}{}{})",
                    if s.wave_exact { "exact" } else { "approx" },
                    s.validate as u8,
                    if s.refused == Refused::Xcl { "xcl" } else { "skip" },
                    s.only.as_ref().map_or_else(String::new, |o| format!(", only {} programs", o.len())),
                    if s.skip.is_empty() { String::new() } else { format!(", never {} programs", s.skip.len()) }
                );
                s
            })
        }
        Err(e) => {
            crate::xlog_warn!("cgbj: {e}; xcl translates every program");
            None
        }
    })
    .as_ref()
}

/// Host facts, set once before the first translation.
struct Host {
    hw_wave: u32,
    /// Hash of the translator's fingerprint, the binding and the validator; with the per-call
    /// facts it makes the cache identity.
    base: u64,
    /// The disk cache's folder; `None` with `XCL_TRANSLATED_CACHE=0`.
    dir: Option<PathBuf>,
}

fn host(s: &Switches, hw_wave: u32) -> &'static Host {
    static H: OnceLock<Host> = OnceLock::new();
    H.get_or_init(|| {
        xcl_cgbj::emit::set_host_max_wave(hw_wave);
        // Off: the emulator does not know whether the GPU is RDNA (pso.rs keeps only the PCI
        // vendor, privately), and off keeps the console's float results on every GPU.
        xcl_cgbj::emit::set_host_rdna(false);
        xcl_cgbj::xcl::set_wave_exact(s.wave_exact);
        let fingerprint = xcl_cgbj::emit::translation_fingerprint();
        let validator = if s.validate {
            let _ = crate::gpu::recomp::dxc::location();
            serde_json::to_string(&crate::gpu::recomp::dxc::toolchain()).unwrap_or_default()
        } else {
            "no validation".into()
        };
        let base = cache::identity(&[b"xcl-cgbj-cache-v1", &fingerprint, &BINDING_VERSION.to_le_bytes(), validator.as_bytes()]);
        let disk = std::env::var("XCL_TRANSLATED_CACHE").map_or(true, |v| v != "0");
        Host { hw_wave, base, dir: disk.then(crate::gpu::recomp::dxc::cache_dir) }
    })
}

static TRIED: AtomicU64 = AtomicU64::new(0);
static PRODUCED: AtomicU64 = AtomicU64::new(0);
static FROM_CACHE: AtomicU64 = AtomicU64::new(0);
static INVALID: AtomicU64 = AtomicU64::new(0);
static DXIL_BYTES: AtomicU64 = AtomicU64::new(0);
static TRANSLATE_MS: AtomicU64 = AtomicU64::new(0);
static REFUSED: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Counts a refusal; true the first time this reason is seen.
fn count_refusal(reason: &str) -> bool {
    let Ok(mut m) = REFUSED.lock() else { return false };
    let n = m.entry(reason.to_string()).or_default();
    *n += 1;
    *n == 1
}

/// Translate the console compute program `blob` with CGBJ, or say why not. `hw_wave` is 32
/// when the device runs only 32-lane waves (real.rs `wave32_only`), else 64. With the
/// switches off this returns `Fallback` at once: no translation, no file, no log.
pub fn compute(blob: &[u8], sampler_base: u32, hw_wave: u32) -> Outcome {
    let Some(s) = switches() else { return Outcome::Fallback };
    let fnv = crate::gpu::shader::fnv64(blob);
    if !s.wants(fnv) {
        return Outcome::Fallback;
    }
    let host = host(s, hw_wave);
    if hw_wave != host.hw_wave {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| crate::xlog_warn!("cgbj: a device with {hw_wave}-lane waves after one with {}: xcl translates its programs", host.hw_wave));
        return Outcome::Fallback;
    }
    let refused_outcome = if s.refused == Refused::Xcl { Outcome::Fallback } else { Outcome::Skipped };
    let windows = crate::gpu::recomp::pso::mem_windows();
    let thread_limits = crate::gpu::recomp::dispatch_thread_dimensions();
    let ident = cache::identity(&[&host.base.to_le_bytes(), &windows.to_le_bytes(), &[thread_limits as u8], &hw_wave.to_le_bytes()]);
    let key = Key::new(ident, fnv, blob, sampler_base);
    let cached = cache::remembered(&key).or_else(|| {
        let e = cache::load(host.dir.as_deref()?, &key)?;
        cache::remember(key, &e);
        Some(e)
    });
    match cached {
        Some(e) => {
            TRIED.fetch_add(1, Relaxed);
            FROM_CACHE.fetch_add(1, Relaxed);
            return match e {
                Entry::Good(c) => Outcome::Compiled(c),
                Entry::Refused(reason) => {
                    count_refusal(&reason);
                    refused_outcome
                }
            };
        }
        None if crate::gpu::recomp::dxc::in_cached_only() => {
            crate::gpu::recomp::dxc::defer();
            return Outcome::Deferred;
        }
        None => {}
    }
    TRIED.fetch_add(1, Relaxed);
    if let Err(why) = s.validate.then(validate::ready).unwrap_or(Ok(())) {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| crate::xlog_warn!("cgbj: no validator ({why}); xcl translates compute programs until there is one"));
        count_refusal("validator unavailable");
        return refused_outcome;
    }
    let cfg = XclConfig { mem_windows: windows, thread_limits, sampler_base };
    let keep = |e: &Entry| {
        cache::remember(key, e);
        if let Some(dir) = &host.dir {
            cache::store(dir, &key, e);
        }
    };
    let started = std::time::Instant::now();
    match translate(blob, &cfg, s.validate) {
        Ok((c, form)) => {
            let ms = started.elapsed().as_millis() as u64;
            PRODUCED.fetch_add(1, Relaxed);
            DXIL_BYTES.fetch_add(c.dxil.len() as u64, Relaxed);
            TRANSLATE_MS.fetch_add(ms, Relaxed);
            crate::xlog_info!(
                "[SHADER] compute {fnv:016x}: cgbj, {} B DXIL, {ms} ms, {} static samplers{}",
                c.dxil.len(),
                c.samplers.len(),
                form.map_or_else(String::new, |(label, kind)| format!(", form {label} ({})", kind.name()))
            );
            keep(&Entry::Good(c.clone()));
            Outcome::Compiled(c)
        }
        Err(Refusal::Transient(why)) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| crate::xlog_warn!("cgbj: {why}; programs go to xcl until it works"));
            count_refusal("validator unavailable");
            refused_outcome
        }
        Err(Refusal::Kept { reason, detail }) => {
            if count_refusal(&reason) {
                crate::xlog_info!("[SHADER] compute {fnv:016x}: cgbj refused ({detail}), first of its kind; {}", if s.refused == Refused::Xcl { "xcl translates it" } else { "left untranslated" });
            }
            keep(&Entry::Refused(reason));
            refused_outcome
        }
    }
}

enum Refusal {
    /// Cached: the same build refuses the same program again.
    Kept { reason: String, detail: String },
    /// Not the program's fault (no validator): not cached.
    Transient(String),
}

/// CGBJ's translation on a thread of its own with a 64 MiB stack (as dxc.rs runs its direct
/// backend), then the checks a container needs before it is used.
fn translate(blob: &[u8], cfg: &XclConfig, validate: bool) -> Result<(Compiled, Option<(String, xcl_cgbj::xcl::FormKind)>), Refusal> {
    let kept = |reason: &str, detail: String| Refusal::Kept { reason: reason.to_string(), detail };
    let run = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new().name("xcl-cgbj".into()).stack_size(64 << 20).spawn_scoped(scope, || xcl_cgbj::xcl::translate_compute(blob, cfg));
        worker.map(|w| w.join())
    });
    let shader = match run {
        Err(e) => return Err(Refusal::Transient(format!("no translation thread ({e})"))),
        Ok(Err(_)) => return Err(kept("panic", "the translator panicked".into())),
        Ok(Ok(Err(e))) => {
            use xcl_cgbj::ShaderError as E;
            let reason = match &e {
                E::Unsupported { what, .. } => what.to_string(),
                E::Isa { reason, .. } => reason.to_string(),
                E::Malformed(m) => m.to_string(),
                other => other.to_string(),
            };
            return Err(kept(&reason, e.to_string()));
        }
        Ok(Ok(Ok(shader))) => shader,
    };
    if let Some(console) = crate::gpu::recomp::pso::compute_interface(blob).map(|i| i.threads) {
        let packed = shader.form.as_ref().is_some_and(|(label, _)| label == xcl_cgbj::xcl::PACKED_FORM);
        if shader.threads != if packed { xcl_cgbj::xcl::packed_threads(console) } else { console } {
            return Err(kept("numthreads differ from the console's", format!("numthreads {:?}, console {console:?}", shader.threads)));
        }
    }
    if validate {
        match validate::check(&shader.dxil) {
            Ok(()) => {}
            Err(validate::Failure::Unavailable(why)) => return Err(Refusal::Transient(why)),
            Err(validate::Failure::Invalid(msg)) => {
                let n = INVALID.fetch_add(1, Relaxed);
                if n < 8 {
                    let head: Vec<&str> = msg.lines().take(6).collect();
                    crate::xlog_warn!("cgbj: dxil.dll rejects a container ({} B): {}", shader.dxil.len(), head.join(" | "));
                }
                return Err(kept("invalid", "dxil.dll rejects the container".into()));
            }
        }
    }
    Ok((Compiled { dxil: shader.dxil, samplers: shader.samplers }, shader.form))
}

/// One line for the caches log; `None` while CGBJ is off.
pub fn summary() -> Option<String> {
    switches()?;
    let refused = REFUSED.lock().map(|m| m.clone()).unwrap_or_default();
    let total: u64 = refused.values().sum();
    let mut top: Vec<_> = refused.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top: Vec<String> = top.iter().take(4).map(|(r, n)| format!("{r} {n}")).collect();
    Some(format!(
        "cgbj {} tried, {} made ({:.1} MB DXIL in {:.1} s), {} from cache, {total} refused ({} invalid{}{})",
        TRIED.load(Relaxed),
        PRODUCED.load(Relaxed),
        DXIL_BYTES.load(Relaxed) as f64 / (1u64 << 20) as f64,
        TRANSLATE_MS.load(Relaxed) as f64 / 1000.0,
        FROM_CACHE.load(Relaxed),
        INVALID.load(Relaxed),
        if top.is_empty() { "" } else { "; " },
        top.join(", ")
    ))
}
