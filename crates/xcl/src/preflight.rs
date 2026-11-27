//! What the PC has of what a run needs, checked by `xcl run` before the game starts: DXC, the
//! system XGameRuntime and the GPU the game will get (Shader Model 6.6, mesh shaders, wave
//! width). Without one of them the game still starts, but draws nothing or part of the
//! picture, so each missing piece is printed with what to do, and kept in `report.json`.

use serde::{Deserialize, Serialize};

/// How much a missing piece costs the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Ok,
    /// runs, with a known limit
    Note,
    /// part of the game is missing (no geometry, no system services, software rendering)
    Warning,
    /// the game draws nothing
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub level: Level,
    pub what: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

/// The GPU the game gets, as D3D12 reports it.
#[derive(Debug, Clone, Default)]
pub struct Gpu {
    pub name: String,
    pub warp: bool,
    /// read from a device (`device_caps`): without it only the name and `warp` are known
    pub probed: bool,
    /// (6, 6) = Shader Model 6.6
    pub shader_model: (u8, u8),
    /// the message for a missing SM 6.6 (it depends on the Agility SDK state)
    pub no_sm66: String,
    pub mesh: bool,
    pub wave_min: u32,
    pub wave_max: u32,
}

/// Everything the checks look at; `checks` turns it into the list.
#[derive(Debug, Clone)]
pub struct Found {
    pub dxc: Result<String, String>,
    /// the DXC in that folder, for its version
    pub dxc_toolchain: crate::gpu::recomp::dxc::Toolchain,
    pub xgameruntime: bool,
    pub gpu: Result<Gpu, String>,
}

fn check(level: Level, what: impl Into<String>, fix: Option<&str>) -> Check {
    Check { level, what: what.into(), fix: fix.map(str::to_string) }
}

/// The checks of the GPU alone (`checks`; the host's, `log_device_checks`).
fn gpu_checks(gpu: &Result<Gpu, String>) -> Vec<Check> {
    let mut v = Vec::new();
    match gpu {
        Err(e) => v.push(check(Level::Error, format!("no D3D12 device: {e}"), Some("update the GPU driver"))),
        Ok(g) => {
            if g.warp {
                v.push(check(Level::Warning, "GPU: WARP (software renderer) – no hardware D3D12 device was usable", Some("update the GPU driver; d3d12_adapter in xcl.toml or XCL_ADAPTER picks the GPU")));
            } else {
                v.push(check(Level::Ok, format!("GPU: {}", g.name), None));
            }
            if !g.probed {
                v.push(check(Level::Note, "shader model, mesh shaders and wave width: checked by the game process on its own device, in xcl.log (XCL_PREFLIGHT_DEVICE=1: here, with a device of the launcher's own)", None));
                return v;
            }
            if g.shader_model < (6, 6) {
                v.push(check(Level::Error, format!("{} (has {}.{}): shaders cannot run", g.no_sm66, g.shader_model.0, g.shader_model.1), Some("update the GPU driver; on Windows 10, put the D3D12 Agility SDK's D3D12Core.dll in a D3D12 folder next to XSEmu.exe")));
            }
            if !g.mesh {
                v.push(check(Level::Warning, "no mesh shaders: the game's geometry and NGG programs are not drawn", Some("a GPU with D3D12 mesh shaders is needed (NVIDIA RTX 20 / GTX 16 or newer, AMD RX 6000 or newer, Intel Arc)")));
            }
            let wave64 = g.wave_min <= 64 && g.wave_max >= 64;
            let wave32 = g.wave_min <= 32 && g.wave_max >= 32;
            if !wave64 && wave32 {
                v.push(check(Level::Note, format!("waves of {}–{} lanes: the console's 64-lane programs run as two 32-lane halves (slower; vertex / pixel shaders that read lanes 32–63 can differ)", g.wave_min, g.wave_max), None));
            } else if !wave64 && !wave32 && g.wave_max > 0 {
                v.push(check(Level::Error, format!("waves of {}–{} lanes: neither 64 nor 32, the console's programs cannot run", g.wave_min, g.wave_max), None));
            }
        }
    }
    v
}

pub fn checks(f: &Found) -> Vec<Check> {
    let mut v = gpu_checks(&f.gpu);
    match &f.dxc {
        Ok(dir) => {
            use crate::gpu::recomp::dxc;
            let t = &f.dxc_toolchain;
            v.push(check(Level::Ok, format!("DXC {}: {dir}", t.dxcompiler), None));
            if t.outdated() {
                v.push(check(Level::Note, format!("DXC is older than {}.{}: {}", dxc::RECOMMENDED.0, dxc::RECOMMENDED.1, dxc::OUTDATED_COST), Some(dxc::OUTDATED_FIX)));
            }
        }
        Err(_) => v.push(check(Level::Error, "no DXC (dxcompiler.dll + dxil.dll): no shader can be compiled, the game draws nothing", Some("install the Windows SDK, or put dxcompiler.dll and dxil.dll next to XSEmu.exe (or in %XCL_DXC_DIR%)"))),
    }
    if !f.xgameruntime {
        v.push(check(Level::Warning, "no System32\\xgameruntime.dll: dialogs, text entry, game invites and game streaming are missing (XCL's own task queues, async and networking are used)", Some("install Gaming Services (Microsoft Store)")));
    }
    v
}

/// Whether `xcl run` makes a D3D12 device of its own for the GPU checks (`XCL_PREFLIGHT_DEVICE=1`;
/// off by default). Making it maps NVIDIA's shader cache into the launcher, which lives for the
/// whole run, so what it holds is paid beside the host's for the whole session: play-118 on 102q2
/// fell ~1 GB/s from its first second there and was stopped before the host started. Off, the
/// launcher names the adapter from DXGI alone and the host checks its own device
/// (`log_device_checks`).
pub fn device_probe() -> bool {
    std::env::var("XCL_PREFLIGHT_DEVICE").is_ok_and(|v| v.trim() == "1")
}

/// The adapter `setting` (`d3d12_adapter`, overridden by `XCL_ADAPTER`) picks – the same
/// choice and the same fallback to WARP as the D3D12 backend's device. Without `device_probe`
/// only its name, from DXGI; with it, a device made, read (`device_caps`), released and trimmed.
fn probe_gpu(setting: &str) -> Result<Gpu, String> {
    use crate::gpu::adapter::{self, Selector};
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI: {e}"))?;
        let sel = Selector::from_env_or(setting);
        let mut list = if sel != Selector::Warp { adapter::candidates(&factory, &sel) } else { Vec::new() };
        if list.is_empty() && sel != Selector::Warp {
            list = adapter::candidates(&factory, &Selector::Auto);
        }
        if !device_probe() {
            return Ok(match list.first() {
                Some((_, desc)) => Gpu { name: format!("{} ({} MiB VRAM)", adapter::name(desc), desc.DedicatedVideoMemory >> 20), ..Gpu::default() },
                None => Gpu { name: "WARP (software renderer)".into(), warp: true, ..Gpu::default() },
            });
        }
        crate::hle::d3d12x::real::game_trim_device(true);
        let mut found: Option<(ID3D12Device, String)> = None;
        for (a, desc) in list {
            if let Ok(dev) = crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0) {
                found = Some((dev, format!("{} ({} MiB VRAM)", adapter::name(&desc), desc.DedicatedVideoMemory >> 20)));
                break;
            }
        }
        let warp = found.is_none();
        let (dev, name) = match found {
            Some(x) => x,
            None => {
                let w = adapter::software_adapter(&factory)?;
                let d = crate::gpu::agility::create_device(&w, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("WARP: {e}"))?;
                (d, "WARP (software renderer)".into())
            }
        };
        crate::hle::d3d12x::real::game_trim_device(false);
        let gpu = device_caps(&dev, name, warp);
        drop(dev);
        crate::hle::d3d12x::real::trim_mapped_files();
        Ok(gpu)
    }
}

/// What D3D12 reports of `dev` (named `name`, WARP with `warp`): the GPU the checks look at.
pub(crate) fn device_caps(dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device, name: String, warp: bool) -> Gpu {
    use windows::Win32::Graphics::Direct3D12::*;
    unsafe {
        // the highest model asked for that the runtime knows; a lower one comes back if that is all
        let mut shader_model = (0, 0);
        for m in [D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_0] {
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: m };
            if dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32).is_ok() {
                let x = sm.HighestShaderModel.0 as u32;
                shader_model = ((x >> 4) as u8, (x & 15) as u8);
                break;
            }
        }
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let _ = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut o1 as *mut _ as _, std::mem::size_of_val(&o1) as u32);
        let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let mesh = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32).is_ok()
            && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
        let no_sm66 = crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state());
        Gpu { name, warp, probed: true, shader_model, no_sm66, mesh, wave_min: o1.WaveLaneCountMin, wave_max: o1.WaveLaneCountMax }
    }
}

/// The host's device (`dev`, `name`) checked as `xcl run` would have with `device_probe`: once,
/// each GPU check in the log, those not OK as warnings (on the console too).
pub(crate) fn log_device_checks(dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device, name: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if device_probe() {
        return;
    }
    ONCE.call_once(|| {
        for c in gpu_checks(&Ok(device_caps(dev, name.to_string(), name.starts_with("WARP")))) {
            let fix = c.fix.map_or(String::new(), |f| format!(" → {f}"));
            match c.level {
                Level::Ok => crate::xlog_info!("D3D12: GPU check on the game's device (XCL_PREFLIGHT_DEVICE=0): {}", c.what),
                _ => crate::xlog_warn!("D3D12: GPU check on the game's device (XCL_PREFLIGHT_DEVICE=0): {}{fix}", c.what),
            }
        }
    });
}

fn xgameruntime_present() -> bool {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    std::path::Path::new(&root).join("System32").join("xgameruntime.dll").is_file()
}

/// Check this PC for a run with the adapter `setting`.
pub fn run(setting: &str) -> Vec<Check> {
    let found = Found {
        dxc: crate::gpu::recomp::dxc::location().map(|p| p.display().to_string()),
        dxc_toolchain: crate::gpu::recomp::dxc::toolchain(),
        xgameruntime: xgameruntime_present(),
        gpu: probe_gpu(setting),
    };
    checks(&found)
}

/// The checks as console lines: OK ones first as `  label`, the rest as WARNING / ERROR with
/// what to do.
pub fn print(checks: &[Check]) {
    for c in checks {
        let tag = match c.level {
            Level::Ok => "  ",
            Level::Note => "  NOTE: ",
            Level::Warning => "  WARNING: ",
            Level::Error => "  ERROR: ",
        };
        println!("{tag}{}", c.what);
        if let Some(fix) = &c.fix {
            println!("           → {fix}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Gpu {
        Gpu { name: "Radeon RX 9070 (16304 MiB VRAM)".into(), warp: false, probed: true, shader_model: (6, 6), no_sm66: "x: no SM 6.6".into(), mesh: true, wave_min: 32, wave_max: 64 }
    }

    fn found(gpu: Gpu) -> Found {
        Found { dxc: Ok(r"C:\dxc".into()), dxc_toolchain: dxc(1, 9), xgameruntime: true, gpu: Ok(gpu) }
    }

    fn dxc(major: u32, minor: u32) -> crate::gpu::recomp::dxc::Toolchain {
        let dxcompiler = crate::gpu::recomp::dxc::Library { major, minor, ..Default::default() };
        crate::gpu::recomp::dxc::Toolchain { dir: r"C:\dxc".into(), dxcompiler, ..Default::default() }
    }

    fn levels(f: &Found) -> Vec<Level> {
        checks(f).into_iter().map(|c| c.level).collect()
    }

    #[test]
    fn a_complete_pc_has_nothing_to_report() {
        assert_eq!(levels(&found(gpu())), [Level::Ok, Level::Ok]);
    }

    #[test]
    fn each_missing_piece_is_reported_with_a_fix() {
        let mut f = found(Gpu { shader_model: (6, 5), mesh: false, ..gpu() });
        f.dxc = Err("dxcompiler.dll not found".into());
        f.xgameruntime = false;
        let c = checks(&f);
        assert_eq!(c.iter().map(|c| c.level).collect::<Vec<_>>(), [Level::Ok, Level::Error, Level::Warning, Level::Error, Level::Warning]);
        assert!(c[1].what.contains("has 6.5"), "{:?}", c[1]);
        // XCL's own task queues and networking stand in for the PC runtime's: not missing
        let missing = c[4].what.split(" missing").next().unwrap();
        assert!(missing.contains("dialogs") && !missing.contains("task queues") && !missing.contains("networking"), "{:?}", c[4]);
        assert!(c.iter().filter(|c| c.level != Level::Ok).all(|c| c.fix.is_some()), "{c:?}");
    }

    #[test]
    fn wave_width() {
        // NVIDIA: 32 only – the split form, a note
        let c = checks(&found(Gpu { wave_min: 32, wave_max: 32, ..gpu() }));
        assert_eq!(c[1].level, Level::Note);
        assert!(c[1].what.contains("two 32-lane halves"), "{:?}", c[1]);
        // 64 in range: nothing; 8–16 (neither size): an error
        assert_eq!(levels(&found(Gpu { wave_min: 64, wave_max: 64, ..gpu() })), [Level::Ok, Level::Ok]);
        assert_eq!(levels(&found(Gpu { wave_min: 8, wave_max: 16, ..gpu() }))[1], Level::Error);
    }

    #[test]
    fn an_old_dxc_is_a_note_with_a_fix() {
        let c = checks(&Found { dxc_toolchain: dxc(1, 7), ..found(gpu()) });
        assert_eq!(c.iter().map(|c| c.level).collect::<Vec<_>>(), [Level::Ok, Level::Ok, Level::Note]);
        assert_eq!(c[1].what, r"DXC 1.7: C:\dxc");
        assert!(c[2].what.contains("older than 1.8") && c[2].fix.as_deref().is_some_and(|f| f.contains("XCL_DXC_DIR")), "{:?}", c[2]);
        assert_eq!(levels(&Found { dxc_toolchain: dxc(1, 8), ..found(gpu()) }), [Level::Ok, Level::Ok]);
    }

    /// `XCL_PREFLIGHT_DEVICE` off: the launcher has the adapter's name only, and says where the rest is checked.
    #[test]
    fn an_adapter_not_probed_leaves_its_checks_to_the_game() {
        let c = checks(&found(Gpu { name: "RTX".into(), ..Gpu::default() }));
        assert_eq!(c.iter().map(|c| c.level).collect::<Vec<_>>(), [Level::Ok, Level::Note, Level::Ok]);
        assert!(c[1].what.contains("XCL_PREFLIGHT_DEVICE=1"), "{:?}", c[1]);
        assert_eq!(levels(&found(Gpu { name: "WARP".into(), warp: true, ..Gpu::default() }))[..2], [Level::Warning, Level::Note]);
    }

    #[test]
    fn warp_and_no_device() {
        assert_eq!(levels(&found(Gpu { warp: true, ..gpu() }))[0], Level::Warning);
        let f = Found { gpu: Err("DXGI: E_FAIL".into()), ..found(gpu()) };
        assert_eq!(checks(&f)[0].level, Level::Error);
    }

    #[test]
    fn checks_round_trip_through_the_report() {
        let c = checks(&Found { dxc: Err("x".into()), ..found(gpu()) });
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"level\":\"error\""), "{json}");
        assert_eq!(serde_json::from_str::<Vec<Check>>(&json).unwrap(), c);
    }
}
