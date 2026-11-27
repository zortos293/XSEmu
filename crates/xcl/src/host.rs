//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
}

/// The console's waits time out to the millisecond; Windows' default 15.6 ms timer tick turns a
/// title's 3–4 ms waits (an audio mixer's) into ~10 ms ones, and its audio plays at half speed.
/// The process asks for a fine timer (`XCL_TIMER_MS`, default 1; 0 keeps the default tick) and
/// keeps it when minimised (Windows 11 ignores a hidden process's request otherwise).
fn fine_timer() {
    use windows_sys::Win32::System::Threading as thr;
    let ms = std::env::var("XCL_TIMER_MS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
    if ms == 0 {
        return;
    }
    let state = thr::PROCESS_POWER_THROTTLING_STATE {
        Version: thr::PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: thr::PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    let throttling = unsafe {
        thr::SetProcessInformation(thr::GetCurrentProcess(), thr::ProcessPowerThrottling, &state as *const _ as *const std::ffi::c_void, std::mem::size_of_val(&state) as u32)
    };
    let period = unsafe { timeBeginPeriod(ms) };
    crate::xlog_info!("timer: {ms} ms resolution requested ({}), timer requests kept when hidden: {}", if period == 0 { "granted" } else { "refused" }, throttling != 0);
}

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            crate::control::exit(0x7C1_0001, "error", &format!("cannot read {}: {e}", boot_path.display()));
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::xlog_console!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_console!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());
    crate::hle::d3d12x::real::start_game_trim("the host", false);
    crate::gpu::pix_capture::load_before_d3d12();

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    crate::hosting::start();
    crate::control::start();
    fine_timer();
    crate::hle::d3d12x::console_status::start();
    crate::memdiag::start();
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail, crate::hle::d3d12x::present_order::dropped(), crate::hle::d3d12x::bind::indirect_refusals()]
            };
            let mut last = snap();
            for tick in 1u64.. {
                std::thread::sleep(std::time::Duration::from_secs(2));
                if tick % 15 == 0 {
                    let (live, idle, again) = crate::hle::d3d12x::pso_async::live();
                    crate::xlog_info!(
                        "caches: {}; {}; pipelines live {live} (released unbound {idle}, built again {again}), commit {} MB; {}{}{}",
                        crate::gpu::recomp::dxc::stage_summary(),
                        crate::hle::d3d12x::real::driver_compiles_summary(),
                        crate::util::process_commit_bytes() >> 20,
                        crate::hle::d3d12x::bind::indirect_calls(),
                        crate::hle::d3d12x::real::pipeline_library_summary().map_or_else(String::new, |s| format!("; {s}")),
                        crate::gpu::cgbj::summary().map_or_else(String::new, |s| format!("; {s}"))
                    );
                    crate::xlog_info!("commit: {}", commit_breakdown());
                    note_resident_memory();
                    crate::xlog_info!("{}", crate::hle::d3d12x::bind::mesh_counts());
                    crate::hle::d3d12x::bind::log_loop_profiles();
                }
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents ({} dropped as older than the frame shown), {} submits, {} draws, {} dispatches, {} indirect calls refused as reading unwritten arguments, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed, {} entries waiting for Submit",
                    d[0], d[11], d[1], d[2], d[3], d[12], d[4], d[5], d[6], d[7], d[8], d[9], d[10], crate::hle::dstorage::waiting_for_submit()
                );
                if let Some(refused) = crate::hle::d3d12x::bind::indirect_refusal_detail() {
                    crate::xlog_info!("indirect calls refused (2 s), by what they wait for: {refused}");
                }
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = crate::util::process_commit_bytes();
                let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let unbacked = crate::hle::d3d12x::vram::unbacked_bytes();
                let (slab_used, slabs) = crate::hle::d3d12x::vram::slab_bytes();
                let mb = |b: u64| b >> 20;
                // non-local video memory by owner: the coherent window is system memory D3D12
                // opened as a heap (the title's own heaps; all of it counted, used or not)
                let (window, window_used, pool) = crate::gpu::uma::made()
                    .map_or((0, 0, (0, 0)), |u| (u.size() as u64, u.used() as u64, u.pool_bytes()));
                let (window, other_non_local) = crate::gpu::uma::non_local_split(non_local, window);
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB; {} MB of texture tiles without texels have none); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped); small-texture slabs {} KB ({} KB in use); non-local is the coherent window {} MB ({} MB in use, XCL's pool {} MB of that with {} MB handed out) and {} MB of upload and readback memory",
                    mb(private),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(unbacked as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64),
                    slabs >> 10,
                    slab_used >> 10,
                    mb(window),
                    mb(window_used),
                    pool.0 >> 20,
                    pool.1 >> 20,
                    mb(other_non_local)
                );
                if let Some(aliases) = crate::hle::d3d12x::texture_alias::summary() {
                    crate::xlog_info!("{aliases}");
                }
                last = now;
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    let shader_cache = crate::gpu::recomp::dxc::cache_dir();
    crate::vfs::init(
        crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts)
            .own_folder(&shader_cache)
            .own_folder(&crate::gpu::recomp::dxc::normalized_cache_dir(&shader_cache)),
    );
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::wait_handles::install() {
        crate::xlog_error!("XCL_TRACE_HANDLES: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    // the D3D12 device before any title code runs: only now may the coherent window become a
    // driver heap at its address (`hle::d3d12x::uma_driver`)
    crate::hle::d3d12x::backend_before_guest();
    HOST_MODE.store(true, Ordering::Release);

    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::hle::d3d12x::present::flush_frame_dumps();
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            crate::control::exit(0x7C1_0002, "error", &format!("loading the game failed: {e}"));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::control::exit(code as u32, "title_exit", "");
    crate::hle::d3d12x::present::flush_frame_dumps();
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

fn note_resident_memory() {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION, PROCESS_MEMORY_COUNTERS_EX2};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut process = PROCESS_MEMORY_COUNTERS_EX2 { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32, ..Default::default() };
    let have_process = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), std::ptr::from_mut(&mut process).cast(), process.cb) } != 0;
    let mut system = PERFORMANCE_INFORMATION::default();
    let have_system = unsafe { K32GetPerformanceInfo(&mut system, std::mem::size_of::<PERFORMANCE_INFORMATION>() as u32) } != 0;
    let process_status = if have_process {
        format!("XCL working set {} MB ({} MB private, {} MB shareable)", process.WorkingSetSize >> 20, process.PrivateWorkingSetSize >> 20,
            process.WorkingSetSize.saturating_sub(process.PrivateWorkingSetSize) >> 20)
    } else {
        "XCL private working set unavailable".into()
    };
    let system_status = if have_system {
        let mb = |pages: usize| pages.saturating_mul(system.PageSize) >> 20;
        format!("physical {} of {} MB available, system cache {} MB, kernel {} MB, system commit {} of {} MB",
            mb(system.PhysicalAvailable), mb(system.PhysicalTotal), mb(system.SystemCache), mb(system.KernelTotal), mb(system.CommitTotal), mb(system.CommitLimit))
    } else {
        "system memory counters unavailable".into()
    };
    crate::xlog_info!("RAM: {process_status}; {system_status}; in-process compute blobs {} MB",
        crate::hle::d3d12x::real::resident_compute_blob_bytes() >> 20);
}

/// The process's committed memory by owner, from a walk of its address space: private commit in
/// XCL's unified memory and sparse windows, the title's watched graphics reservations and
/// physical-page reservations, the rest of what the title allocated (`memmap`), and what nobody
/// records (heaps: XCL's, the driver's, DXC's, the title's) by allocation size.
fn commit_breakdown() -> String {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE};
    let uma = crate::gpu::uma::get();
    let (mut sparse, mut unified, mut watched, mut physical, mut title, mut mapped, mut image) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut untracked: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    // write-combined private memory nobody here allocates: what WDDM commits in the process for
    // each video-memory allocation (heaps, committed resources, upload buffers) – by allocation
    let mut write_combined: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    let mut at = 0x1_0000usize;
    while at < 0x7FFF_FFFF_0000 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const std::ffi::c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.RegionSize == 0 {
            break;
        }
        let (base, size) = (m.BaseAddress as usize, m.RegionSize as u64);
        at = base + m.RegionSize;
        if m.State != MEM_COMMIT {
            continue;
        }
        match m.Type {
            // (the coherent window as a driver heap, `uma_driver`, is a mapping)
            MEM_MAPPED if uma.is_some_and(|u| u.is_committed(base)) => unified += size,
            MEM_MAPPED => mapped += size,
            MEM_IMAGE => image += size,
            MEM_PRIVATE if uma.is_some_and(|u| u.is_sparse(base)) => sparse += size,
            MEM_PRIVATE if uma.is_some_and(|u| u.contains(base)) => unified += size,
            MEM_PRIVATE if crate::hle::kernelx::in_graphics_reservation(base) => watched += size,
            MEM_PRIVATE if crate::hle::kernelx::in_reservation(base) => physical += size,
            MEM_PRIVATE if crate::memmap::reservation(base).is_some() => title += size,
            _ => {
                *untracked.entry(m.AllocationBase as usize).or_default() += size;
                if m.Type == MEM_PRIVATE && m.Protect & 0x400 != 0 {
                    *write_combined.entry(m.AllocationBase as usize).or_default() += size;
                }
            }
        }
    }
    let (mut large, mut medium, mut small, mut n_large, mut n_medium) = (0u64, 0u64, 0u64, 0, 0);
    for &b in untracked.values() {
        if b >= 64 << 20 {
            large += b;
            n_large += 1;
        } else if b >= 1 << 20 {
            medium += b;
            n_medium += 1;
        } else {
            small += b;
        }
    }
    let mb = |b: u64| b >> 20;
    let pool = 64u64 << 20;
    format!(
        "private {} MB in sparse windows, {} MB in unified memory, {} MB in watched graphics reservations, {} MB in physical-page reservations, {} MB other title allocations; untracked {} MB ({} allocations of 64 MB+: {} MB, {} of 1-64 MB: {} MB, smaller: {} MB; the title's XMemAlloc heap memory {} MB of it; write-combined – the commit behind video memory – {} MB of it in {} allocations, {} of them of 64 MB); mapped {} MB, images {} MB",
        mb(sparse), mb(unified), mb(watched), mb(physical), mb(title),
        mb(large + medium + small), n_large, mb(large), n_medium, mb(medium), mb(small), mb(crate::hle::kernelx::xmem_heap_bytes() as u64),
        mb(write_combined.values().sum()), write_combined.len(), write_combined.values().filter(|&&b| b == pool).count(), mb(mapped), mb(image)
    )
}
