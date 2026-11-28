//! xcl.exe – command line front-end and (in host mode) the process the game runs in.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Our own TLS callback: ntdll calls it for every thread start/exit in the process.
/// In host mode it initialises the game's static TLS for the new thread and forwards
/// the notification to the game's TLS callbacks / DllMains.
unsafe extern "system" fn xcl_tls_callback(_h: *mut c_void, reason: u32, _r: *mut c_void) {
    if xcl::host::HOST_MODE.load(Ordering::Acquire) {
        xcl::loader::on_tls_event(reason);
    }
}

#[used]
#[link_section = ".CRT$XLY"]
static XCL_TLS_CALLBACK: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = xcl_tls_callback;

const HELP: &str = "\
XCL – Xbox Series X|S → Windows compatibility layer

Usage:
  xcl run [--host-hwnd <hex>] [--control-pipe <name>] <game folder | file.exe> [-- game arguments]
      Runs the game. The game folder is only read; writes, game saves and logs go to
      %LOCALAPPDATA%\\XCL\\<title>\\ (or %XCL_DATA_ROOT%\\<title>\\).
      --host-hwnd: the game window is a child of that window (a launcher's), 16:9 inside it.
      --control-pipe: JSON lines with the launcher on \\\\.\\pipe\\<name> (docs/launcher-protocol.md).
  xcl inspect <file.exe|file.dll> [--json]
      Analysis without running: imports and what XCL has for them (HLE / Windows / stub).
  xcl disasm <file> [--offset N] [--length N] [--all] [--hex] [--fssm FILE]
      Disassembler for console GPU code (RDNA2) – e.g. for shaders saved in <data>\\shaders\\.
      Console containers (XDAT/XBCH/FSSM) are unpacked; --fssm saves the unpacked FSSM.
  xcl shaders <file | folder | package.rpf> [--top N] [--json] [--recomp] [--compile N]
      Shader corpus statistics: formats, stages, RDNA2 instruction histogram (priorities
      for the recompiler). RPF9 packages are scanned for shader records.
      --recomp: translates every RDNA2 program → HLSL and shows what is missing;
      --compile N: additionally compiles N complete translations with DXC.
  xcl rpf <package.rpf> [--top N] [--find NAME] [--extract NAME --out DIR]
      Lists or extracts readable inline records. Archives larger than 1 GB use a
      bounded-memory header probe; --probe requests it explicitly and --full overrides it.
      Use --probe only for header summaries, without --find, --extract, --out or --full.
  xcl warm-up [--full] [--progress-json]
      Builds the pipelines earlier runs made, so a game needs fewer driver compiles while it
      loads: first those an earlier run bound, then every compute pipeline and the driver's
      blob of every translated compute program (giants over 1 MB of DXIL left out), then the
      rest largest first for 20 minutes (XCL_WARMUP_BUDGET_S), leaving out those no run of the
      last 3 made. --full: everything, giants and stale ones too, with no time limit, going on
      where an earlier --full pass stopped.
      --progress-json: progress as JSON lines on stdout, {\"cmd\":\"cancel\"} or EOF on stdin
      cancels and keeps what is built (docs/launcher-protocol.md).
  xcl warm-up --mesh-only [--recent-used N]
      Builds only missing driver blobs for the mesh pipelines kept in the manifest.
  xcl warm-up --blocking [--budget-s N] [--targets FILE] [--max-keys N] [--fresh] [--in-process]
                         [--pool-children N] [--giant-children N] [--recycle-keys N] [--recycle-mb N]
                         [--key-timeout-s N]
      Replays every captured descriptor that held a bind or first use (the first N with
      --max-keys) on one worker per logical core: translation, DXC and driver compiles run at
      once within the memory governor; a waiting giant keeps smaller compiles from taking its
      memory. Keys with a measured wait go first, longest first, then --targets keys listed
      without one (a line with the key alone) in file order. Keys made ready in an earlier pass
      of the same translation and driver are skipped; --fresh replays them.
      The compiles run in child processes of xcl.exe, one governor across them: pool children
      (--pool-children, default 2 from 8 logical cores) share the cores and are replaced after
      --recycle-keys keys (default 400) or --recycle-mb MB of private commit (default 8192);
      a predicted giant (43,691 B of console shaders or more, or slow before) gets a child of
      its own, at most --giant-children (1–96, default half the logical cores) at once, each started
      when its expected peak fits in the RAM available less 1 GB. A key running past --key-timeout-s
      (default 1800) is taken for a hang and its child killed. Each key in flight in a child
      that crashed or hung is tried again alone; one that ends its child again is listed in
      blocking-bad.txt and left out of later passes until --fresh. --in-process (or
      XCL_WARM_CHILDREN=0): every compile in this process.
      Default budget: 120 seconds. Ctrl+C stops before the next driver call; an active call may finish.
      With --targets, reads exact manifest keys and observed wait times (key in hex, milliseconds).
  xcl warm-up --giants-only --floor-mb N
      Builds, one at a time, the driver blob of each giant compute program (over 1 MB of DXIL,
      no blob kept yet) that the normal 6 GB floor leaves no room for now, keeping N MB
      available instead (N at least 1024). Nothing else should run meanwhile.
  xcl --version
";

/// Other xcl.exe and game processes running now, as "name pid".
fn other_xcl_processes() -> Vec<String> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS};
    let me = std::process::id();
    let mut found = Vec::new();
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE {
            return found;
        }
        let mut e: PROCESSENTRY32W = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut ok = Process32FirstW(snap, &mut e) != 0;
        while ok {
            let len = e.szExeFile.iter().position(|&c| c == 0).unwrap_or(e.szExeFile.len());
            let name = String::from_utf16_lossy(&e.szExeFile[..len]);
            if e.th32ProcessID != me && (name.eq_ignore_ascii_case("xcl.exe") || name.eq_ignore_ascii_case("game_scarlett_master.exe")) {
                found.push(format!("{name} {}", e.th32ProcessID));
            }
            ok = Process32NextW(snap, &mut e) != 0;
        }
        CloseHandle(snap);
    }
    found
}

fn warm_up_mesh(recent_used: Option<std::num::NonZeroU32>) -> i32 {
    let others = other_xcl_processes();
    if !others.is_empty() {
        eprintln!("xcl warm-up: refusing to start – another XCL process is running ({}); one XCL process tree at a time: stop it or wait for it to end", others.join(", "));
        return 3;
    }
    match xcl::hle::d3d12x::real::warm_up_mesh(recent_used) {
        Ok(w) => {
            println!(
                "warm-up --mesh-only: {} pipelines, {} already cached, {} warmed, {} failed, {} left for the memory limits; {:.2} s on {} threads",
                w.kept + w.built + w.failed + w.too_big, w.kept, w.built, w.failed, w.too_big, w.took.as_secs_f64(), w.threads
            );
            (w.failed + w.too_big != 0) as i32
        }
        Err(e) => {
            eprintln!("xcl warm-up --mesh-only: {e}");
            2
        }
    }
}

/// `xcl warm-up --giants-only --floor-mb N`: `real::warm_up_giants`, each giant's outcome on the console.
fn warm_up_giants(floor_mb: u64) -> i32 {
    let others = other_xcl_processes();
    if !others.is_empty() {
        eprintln!("xcl warm-up: refusing to start – another XCL process is running ({}); one XCL process tree at a time: stop it or wait for it to end", others.join(", "));
        return 3;
    }
    let started = std::time::Instant::now();
    match xcl::hle::d3d12x::real::warm_up_giants(floor_mb << 20) {
        Ok((giants, peak_commit)) => {
            for g in &giants {
                println!("warm-up giant: DXIL {:016x} ({} B): {} ({:.0} s)", g.hash, g.len, g.outcome, g.took.as_secs_f64());
            }
            println!(
                "warm-up --giants-only: {} giants without a driver blob, floor {floor_mb} MB; {:.0} s, peak commit {} MB",
                giants.len(),
                started.elapsed().as_secs_f64(),
                peak_commit >> 20
            );
            giants.iter().any(|g| !g.outcome.starts_with("built") && !g.outcome.starts_with("fits")) as i32
        }
        Err(e) => {
            eprintln!("xcl warm-up: no D3D12 device: {e}");
            2
        }
    }
}

/// `--progress-json`: the stdout the launcher reads, for the JSON lines alone; the process's
/// stdout handle is pointed at stderr, so whatever else XCL prints goes there.
struct JsonOut(std::sync::Mutex<std::fs::File>);

impl JsonOut {
    fn take_stdout() -> JsonOut {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
        unsafe {
            let stdout = GetStdHandle(STD_OUTPUT_HANDLE);
            SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
            JsonOut(std::sync::Mutex::new(std::fs::File::from_raw_handle(stdout)))
        }
    }

    fn emit(&self, e: &xcl::warm_up_json::Event) {
        use std::io::Write;
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(out, "{}", xcl::warm_up_json::line(e));
        let _ = out.flush();
    }
}

/// `xcl warm-up`: `real::warm_up`, with its progress and result on the console: with
/// `XCL_PROGRESS` on (the default) a line at most every second with the blobs built, the DXIL
/// they came from, the driver compiles in flight and the RAM free. `--progress-json`: the
/// launcher's JSON lines on stdout instead (docs/launcher-protocol.md), the console's text on
/// stderr, and `{"cmd":"cancel"}` or the end of stdin cancels (exit code 4).
fn warm_up(args: xcl::warm_up_json::Args) -> i32 {
    use xcl::hle::d3d12x::real::WarmStatus;
    use xcl::warm_up_json::{phase_id, Event, Rate};
    let full = args.full;
    let out = args.progress_json.then(JsonOut::take_stdout);
    let others = other_xcl_processes();
    if !others.is_empty() {
        eprintln!("xcl warm-up: refusing to start – another XCL process is running ({}); one XCL process tree at a time: stop it or wait for it to end", others.join(", "));
        if let Some(out) = &out {
            out.emit(&Event::Error { message: format!("another XCL process is running ({})", others.join(", ")) });
        }
        return 3;
    }
    if out.is_some() {
        let _ = std::thread::Builder::new().name("xcl-warm-up-stdin".into()).spawn(|| {
            let why = xcl::warm_up_json::wait_for_cancel(std::io::stdin().lock());
            eprintln!("xcl warm-up: cancel ({why:?}): no more builds start, those running finish");
            xcl::hle::d3d12x::real::cancel_warm_up();
        });
    }
    let reported = std::sync::Mutex::new((Rate::new(std::time::Instant::now()), None::<WarmStatus>));
    let status = |s: &WarmStatus| {
        let Some(out) = &out else { return };
        let mut reported = reported.lock().unwrap_or_else(|e| e.into_inner());
        let (rate, last) = &mut *reported;
        if last.is_none() {
            out.emit(&Event::WarmupStart { total: s.pipelines, full, threads: s.threads });
        }
        *last = Some(*s);
        let phase = phase_id(s.phase);
        let (keys_per_s, eta_s) = rate.at(phase, s.done, s.total, std::time::Instant::now());
        out.emit(&Event::WarmupProgress {
            done: s.done,
            total: s.total,
            keys_per_s,
            eta_s,
            ram_free_mb: xcl::hle::d3d12x::real::available_physical() >> 20,
            giants_compiling: s.giants,
            in_flight: s.in_flight,
            failed: s.failed,
            phase,
            paused_for_ram: s.short_of_ram,
        });
    };
    let dxil = std::sync::atomic::AtomicU64::new(0);
    let last = std::sync::Mutex::new(None::<std::time::Instant>);
    let monitor = xcl::progress_window::Monitor::start("warm-up", "driver blobs", 0, None);
    let report = |done: usize, total: usize, len: usize, took: std::time::Duration, hash: u64| {
        let dxil = dxil.fetch_add(len as u64, Ordering::Relaxed) + len as u64;
        xcl::progress::warm_up(done, total);
        monitor.set_total(total);
        monitor.advance(done, 0);
        if monitor.console() && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            return;
        }
        if !xcl::progress::enabled() {
            if done % 25 == 0 || took.as_secs() >= 30 || done == total {
                println!("  {done}/{total} (last: {hash:016x}, {len} B, {:.1} s)", took.as_secs_f32());
            }
            return;
        }
        let mut last = last.lock().unwrap();
        if last.is_some_and(|t| t.elapsed().as_secs() < 1) && done != total {
            return;
        }
        *last = Some(std::time::Instant::now());
        let s = xcl::progress::snapshot();
        println!(
            "  warm-up {done}/{total} ({}%), {} MB DXIL built, {} in flight ({} large), {} MB RAM free (last: {hash:016x}, {len} B, {:.1} s)",
            done * 100 / total.max(1),
            dxil >> 20,
            s.driver.pending,
            s.large,
            xcl::hle::d3d12x::real::available_physical() >> 20,
            took.as_secs_f32()
        );
    };
    let result = xcl::hle::d3d12x::real::warm_up(full, report, &status);
    drop(monitor);
    match result {
        Ok(w) => {
            println!(
                "warm-up{}: {} pipelines earlier runs made ({} kept entries dropped), {} last made in this translation, {} stale left out (none made by the last {} of {} runs){}; {} compute programs, {} driver blobs kept already, {} too big for the memory limits; {} stale blobs deleted ({} MB); {:.0} s on {} threads, peak commit {} MB",
                if w.full { " --full" } else { "" },
                w.pipelines,
                w.dropped,
                w.this_era,
                w.stale_left,
                xcl::hle::d3d12x::pso_manifest::STALE_AFTER_RUNS,
                w.runs,
                match (w.full, w.complete) {
                    (false, _) => String::new(),
                    (true, true) => format!(", {} done by an earlier pass, this pass complete", w.resumed),
                    (true, false) => format!(", {} done by an earlier pass, this pass stopped part way", w.resumed),
                },
                w.programs,
                w.kept,
                w.too_big(),
                w.pruned.0,
                w.pruned.1 >> 20,
                w.took.as_secs_f32(),
                w.threads,
                w.peak_commit >> 20
            );
            for p in &w.phases {
                println!(
                    "warm-up phase {}: {} of {} built, {} failed, {} too big for the memory limits (after retries), {} giants left for later, {} not started within the budget; {:.0} s",
                    p.name,
                    p.built,
                    p.of,
                    p.failed,
                    p.too_big,
                    p.giants_left,
                    p.not_started,
                    p.took.as_secs_f32()
                );
            }
            println!("warm-up shaders: {}", xcl::gpu::recomp::dxc::stage_summary());
            let code = if w.cancelled { 4 } else { (w.failed() != 0) as i32 };
            if let Some(out) = &out {
                let last = reported.lock().unwrap_or_else(|e| e.into_inner()).1;
                if let (Some(last), Some(p)) = (last, w.phases.last()) {
                    status(&WarmStatus { phase: p.name, done: p.of.saturating_sub(p.not_started), total: p.of, failed: w.failed(), in_flight: 0, giants: 0, short_of_ram: false, ..last });
                }
                out.emit(&Event::WarmupDone {
                    built: w.phases.iter().map(|p| p.built).sum(),
                    failed: w.failed(),
                    too_big: w.too_big(),
                    took_s: (w.took.as_secs_f64() * 10.0).round() / 10.0,
                    complete: w.phases.iter().all(|p| p.not_started == 0),
                    cancelled: w.cancelled,
                    code,
                });
            }
            code
        }
        Err(e) => {
            eprintln!("xcl warm-up: no D3D12 device: {e}");
            if let Some(out) = &out {
                out.emit(&Event::Error { message: format!("no D3D12 device: {e}") });
            }
            2
        }
    }
}

/// Ctrl+C or Ctrl+Break stops a blocking warm-up (its children included) before its next driver call.
unsafe extern "system" fn cancel_blocking_warm_up(event: u32) -> i32 {
    use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};
    if !matches!(event, CTRL_C_EVENT | CTRL_BREAK_EVENT) {
        return 0;
    }
    xcl::hle::d3d12x::real::cancel_blocking_warm_up();
    1
}

/// `value` as a number within `range`.
fn in_range(value: &str, range: std::ops::RangeInclusive<u64>) -> Option<u64> {
    value.parse::<u64>().ok().filter(|n| range.contains(n))
}

fn warm_up_blocking(args: &[String]) -> i32 {
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
    let (mut budget, mut targets, mut max_keys, mut fresh) = (None, None, None, false);
    let mut in_process = std::env::var("XCL_WARM_CHILDREN").is_ok_and(|v| v.trim() == "0");
    let mut children = xcl::hle::d3d12x::warm_children::ChildOptions::default();
    let (mut pool, mut giants, mut recycle_keys, mut recycle_mb, mut key_timeout) = (None, None, None, None, None);
    let mut args = args.iter();
    let invalid = || {
        eprintln!("usage: xcl warm-up --blocking [--budget-s N (1–86400)] [--targets FILE] [--max-keys N (1–1000000)] [--fresh] [--in-process] [--pool-children N (1–16)] [--giant-children N (1–96)] [--recycle-keys N (1–1000000)] [--recycle-mb N (512–1048576)] [--key-timeout-s N (10–86400)]");
        1
    };
    while let Some(flag) = args.next() {
        if flag == "--fresh" && !fresh {
            fresh = true;
            continue;
        }
        if flag == "--in-process" {
            in_process = true;
            continue;
        }
        let numbered = match flag.as_str() {
            "--pool-children" => Some((&mut pool, 1..=16)),
            "--giant-children" => Some((&mut giants, 1..=96)),
            "--recycle-keys" => Some((&mut recycle_keys, 1..=1_000_000)),
            "--recycle-mb" => Some((&mut recycle_mb, 512..=1_048_576)),
            "--key-timeout-s" => Some((&mut key_timeout, 10..=86_400)),
            _ => None,
        };
        if let Some((slot, range)) = numbered {
            match args.next().and_then(|value| in_range(value, range)) {
                Some(n) if slot.is_none() => *slot = Some(n),
                _ => return invalid(),
            }
            continue;
        }
        match (flag.as_str(), args.next()) {
            ("--budget-s", Some(value)) if budget.is_none() => {
                budget = in_range(value, 1..=86400);
                if budget.is_none() {
                    return invalid();
                }
            }
            ("--targets", Some(value)) if targets.is_none() => targets = Some(std::path::Path::new(value)),
            ("--max-keys", Some(value)) if max_keys.is_none() => {
                max_keys = in_range(value, 1..=1_000_000).map(|n| n as usize);
                if max_keys.is_none() {
                    return invalid();
                }
            }
            _ => return invalid(),
        }
    }
    children.pool_children = pool.map(|n| n as usize);
    children.giant_children = giants.map(|n| n as usize).or(children.giant_children);
    children.recycle_keys = recycle_keys.map_or(children.recycle_keys, |n| n as usize);
    children.recycle_mb = recycle_mb.unwrap_or(children.recycle_mb);
    children.key_timeout_s = key_timeout.unwrap_or(children.key_timeout_s);
    if !other_xcl_processes().is_empty() {
        eprintln!("xcl warm-up --blocking: another XCL process or game is running");
        return 3;
    }
    if unsafe { SetConsoleCtrlHandler(Some(cancel_blocking_warm_up), 1) } == 0 {
        eprintln!("xcl warm-up --blocking: could not install Ctrl+C handling");
        return 2;
    }
    match xcl::hle::d3d12x::real::warm_up_blocking(targets, std::time::Duration::from_secs(budget.unwrap_or(120)), max_keys, fresh, (!in_process).then_some(children)) {
        Ok(cancelled) => if cancelled { 130 } else { 0 },
        Err(e) => {
            eprintln!("xcl warm-up --blocking: {e}");
            2
        }
    }
}

/// `xcl warm-up --blocking-child --threads N --budget-s N [--leave-giants]`: one child process of `xcl warm-up --blocking`
/// (keys on stdin, a line per key on stdout); started by it, not by hand.
fn warm_up_blocking_child(args: &[String]) -> i32 {
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
    let (threads, budget, leave_giants) = match args {
        [t, threads, b, budget, rest @ ..] if t == "--threads" && b == "--budget-s" && rest.iter().all(|flag| flag == "--leave-giants") && rest.len() <= 1 => {
            (in_range(threads, 1..=256), in_range(budget, 1..=86400), !rest.is_empty())
        }
        _ => (None, None, false),
    };
    let (Some(threads), Some(budget)) = (threads, budget) else {
        eprintln!("usage: xcl warm-up --blocking-child --threads N (1–256) --budget-s N (1–86400) [--leave-giants], started by xcl warm-up --blocking");
        return 1;
    };
    if unsafe { SetConsoleCtrlHandler(Some(cancel_blocking_warm_up), 1) } == 0 {
        eprintln!("xcl warm-up --blocking-child: could not install Ctrl+C handling");
        return 2;
    }
    match xcl::hle::d3d12x::real::warm_up_blocking_child(threads as usize, std::time::Duration::from_secs(budget), leave_giants) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("xcl warm-up --blocking-child: {e}");
            2
        }
    }
}

fn main() {
    // keep the TLS callback from being discarded by the linker
    unsafe { std::ptr::read_volatile(&XCL_TLS_CALLBACK) };

    if let Some(boot) = std::env::var_os("XCL_HOST") {
        std::env::remove_var("XCL_HOST");
        std::process::exit(xcl::host::host_main(&PathBuf::from(boot)));
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "run") {
        xcl::hle::d3d12x::real::start_game_trim("the launcher", true);
    }
    let code = match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => match xcl::hosting::parse_run_args(&args[1..]) {
            Ok(run) => {
                run.forward();
                xcl::launcher::run(&PathBuf::from(&run.target), &run.game_args)
            }
            Err(e) => {
                eprintln!("xcl run: {e}");
                2
            }
        },
        Some("inspect") if args.len() >= 2 => {
            let json = args.iter().any(|a| a == "--json");
            xcl::inspect::run(&PathBuf::from(&args[1]), json)
        }
        Some("disasm") if args.len() >= 2 => xcl::disasm::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("shaders") if args.len() >= 2 => xcl::shaders::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("rpf") if args.len() >= 2 => xcl::rpf::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("minidump") => xcl::eh::write_minidump_of(&args[1..]),
        Some("warm-up") if args.get(1).is_some_and(|a| a == "--blocking") => warm_up_blocking(&args[2..]),
        Some("warm-up") if args.get(1).is_some_and(|a| a == "--blocking-child") => warm_up_blocking_child(&args[2..]),
        Some("compile-helper") if args.len() == 2 => xcl::hle::d3d12x::real::giant_helper_child(std::path::Path::new(&args[1])),
        Some("warm-up") if args.iter().any(|a| a == "--mesh-only") => {
            match &args[1..] {
                [mesh] if mesh == "--mesh-only" => warm_up_mesh(None),
                [mesh, scope, n] if mesh == "--mesh-only" && scope == "--recent-used" && n.parse::<std::num::NonZeroU32>().is_ok() => {
                    warm_up_mesh(n.parse().ok())
                }
                _ => {
                    eprintln!("xcl warm-up --mesh-only accepts only an optional --recent-used N (N >= 1)");
                    2
                }
            }
        }
        Some("warm-up") if args.iter().any(|a| a == "--giants-only") => {
            let floor = args.iter().position(|a| a == "--floor-mb").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<u64>().ok());
            match floor {
                Some(mb) if mb >= 1024 => warm_up_giants(mb),
                _ => {
                    eprintln!("xcl warm-up --giants-only needs --floor-mb N with N at least 1024");
                    2
                }
            }
        }
        Some("warm-up") => warm_up(xcl::warm_up_json::warm_up_args(&args[1..])),
        Some("--version") | Some("-V") => {
            println!("XSEmu {}", env!("CARGO_PKG_VERSION"));
            0
        }
        None if started_by_double_click() => run_picked_game(),
        _ => {
            print!("{HELP}");
            if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
                0
            } else {
                2
            }
        }
    };
    std::process::exit(code);
}

/// xcl.exe opened from Explorer: the console is its own (from a terminal, the shell shares it).
fn started_by_double_click() -> bool {
    let mut ids = [0u32; 4];
    unsafe { windows_sys::Win32::System::Console::GetConsoleProcessList(ids.as_mut_ptr(), ids.len() as u32) == 1 }
}

/// No command line: a small menu (choose a game / run the last one again), then the game runs
/// in this console. On an error the console stays open until Enter, so the message can be read.
fn run_picked_game() -> i32 {
    let last_file = xcl::launcher::data_root().join("last-game.txt");
    let last = std::fs::read_to_string(&last_file).ok().map(|s| PathBuf::from(s.trim())).filter(|p| p.exists());
    let target = {
        let _ui = Ui::enter();
        let picked = match menu(last.as_deref()) {
            Choice::Again => last.clone(),
            Choice::Cancel => None,
            Choice::Choose | Choice::NoMenu => pick_game_file(),
        };
        // MicrosoftGame.config stands for its folder: XCL picks the exe the config names
        picked.map(|file| match file.file_name() {
            Some(n) if n.eq_ignore_ascii_case("MicrosoftGame.config") => file.parent().map(PathBuf::from).unwrap_or(file),
            _ => file,
        })
    };
    let Some(target) = target else { return 0 };
    let _ = std::fs::create_dir_all(xcl::launcher::data_root());
    let _ = std::fs::write(&last_file, target.to_string_lossy().as_bytes());
    println!("xcl run \"{}\"", target.display());
    let code = xcl::launcher::run(&target, &[]);
    if code != 0 {
        println!("\nXCL ended with code {code}. Press Enter to close.");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    code
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `name` of system DLL `dll`, looked up when the menu needs it: xcl.exe does not import the
/// dialog libraries (the game runs in this executable, and must not get them).
unsafe fn system_proc(dll: &str, name: &str) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let lib = LoadLibraryW(wide(dll).as_ptr());
    if lib.is_null() {
        return None;
    }
    GetProcAddress(lib, format!("{name}\0").as_ptr()).map(|f| f as usize)
}

/// While the menu is open: COM for the file dialog, and the common controls 6 (the task
/// dialog, the current look) through an activation context of its own – not a manifest of
/// xcl.exe, which would apply to the game too.
struct Ui {
    actctx: Option<(windows_sys::Win32::Foundation::HANDLE, usize)>,
}

impl Ui {
    fn enter() -> Ui {
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{ActivateActCtx, CreateActCtxW, ACTCTXW};
        const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency><dependentAssembly>
    <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
  </dependentAssembly></dependency>
</assembly>
"#;
        unsafe {
            windows_sys::Win32::System::Com::CoInitializeEx(std::ptr::null(), windows_sys::Win32::System::Com::COINIT_APARTMENTTHREADED as u32);
        }
        let path = std::env::temp_dir().join("xcl-menu.manifest");
        let actctx = std::fs::write(&path, MANIFEST).ok().and_then(|()| unsafe {
            let source = wide(&path.to_string_lossy());
            let mut c: ACTCTXW = std::mem::zeroed();
            c.cbSize = std::mem::size_of::<ACTCTXW>() as u32;
            c.lpSource = source.as_ptr();
            let h = CreateActCtxW(&c);
            if h == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                return None;
            }
            let mut cookie = 0usize;
            (ActivateActCtx(h, &mut cookie) != 0).then_some((h, cookie))
        });
        Ui { actctx }
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{DeactivateActCtx, ReleaseActCtx};
        unsafe {
            if let Some((h, cookie)) = self.actctx {
                DeactivateActCtx(0, cookie);
                ReleaseActCtx(h);
            }
            windows_sys::Win32::System::Com::CoUninitialize();
        }
    }
}

enum Choice {
    Choose,
    Again,
    Cancel,
    /// no task dialog on this system: straight to the file dialog
    NoMenu,
}

/// The menu: a task dialog with "Choose a game…" and, after a first run, "Run again".
fn menu(last: Option<&std::path::Path>) -> Choice {
    use windows_sys::Win32::UI::Controls::*;
    const CHOOSE: i32 = 100;
    const AGAIN: i32 = 101;
    type TaskDialogIndirect = unsafe extern "system" fn(*const TASKDIALOGCONFIG, *mut i32, *mut i32, *mut i32) -> i32;
    let Some(f) = (unsafe { system_proc("comctl32.dll", "TaskDialogIndirect") }) else { return Choice::NoMenu };
    let title = wide("XCL");
    let heading = wide("Run an Xbox Series game");
    let content = wide("XCL runs the game from its unpacked files. Choose the game's exe or its MicrosoftGame.config.");
    let again = last.map(|p| wide(&format!("Run again\n{}", p.display())));
    let choose = wide("Choose a game…\nThe game's exe or MicrosoftGame.config");
    let mut buttons = Vec::new();
    if let Some(a) = &again {
        buttons.push(TASKDIALOG_BUTTON { nButtonID: AGAIN, pszButtonText: a.as_ptr() });
    }
    buttons.push(TASKDIALOG_BUTTON { nButtonID: CHOOSE, pszButtonText: choose.as_ptr() });
    let mut c: TASKDIALOGCONFIG = unsafe { std::mem::zeroed() };
    c.cbSize = std::mem::size_of::<TASKDIALOGCONFIG>() as u32;
    c.hwndParent = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    c.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    c.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    c.pszWindowTitle = title.as_ptr();
    c.pszMainInstruction = heading.as_ptr();
    c.pszContent = content.as_ptr();
    c.cButtons = buttons.len() as u32;
    c.pButtons = buttons.as_ptr();
    c.nDefaultButton = if again.is_some() { AGAIN } else { CHOOSE };
    let mut pressed = 0i32;
    let hr = unsafe { std::mem::transmute::<usize, TaskDialogIndirect>(f)(&c, &mut pressed, std::ptr::null_mut(), std::ptr::null_mut()) };
    match (hr, pressed) {
        (h, _) if h < 0 => Choice::NoMenu,
        (_, CHOOSE) => Choice::Choose,
        (_, AGAIN) => Choice::Again,
        _ => Choice::Cancel,
    }
}

/// The standard Windows open dialog (None: cancelled).
fn pick_game_file() -> Option<PathBuf> {
    use windows_sys::Win32::UI::Controls::Dialogs::*;
    type GetOpenFileName = unsafe extern "system" fn(*mut OPENFILENAMEW) -> i32;
    let f = unsafe { system_proc("comdlg32.dll", "GetOpenFileNameW") }?;
    // pairs of (description, pattern), each ending in NUL, the list in one more
    let filter: Vec<u16> = "Xbox game (*.exe, MicrosoftGame.config)\0*.exe;MicrosoftGame.config\0All files\0*.*\0\0".encode_utf16().collect();
    let title = wide("XCL – choose a game");
    let mut file = vec![0u16; 32768];
    let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
    ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
    ofn.hwndOwner = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    ofn.lpstrFilter = filter.as_ptr();
    ofn.lpstrFile = file.as_mut_ptr();
    ofn.nMaxFile = file.len() as u32;
    ofn.lpstrTitle = title.as_ptr();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if unsafe { std::mem::transmute::<usize, GetOpenFileName>(f)(&mut ofn) } == 0 {
        return None;
    }
    let n = file.iter().position(|&c| c == 0).unwrap_or(file.len());
    Some(PathBuf::from(String::from_utf16_lossy(&file[..n])))
}
