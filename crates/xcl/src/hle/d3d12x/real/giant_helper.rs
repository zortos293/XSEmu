//! Cold giant compute compiles in a helper process (`XCL_GIANT_HELPER=0`: in the game's own).
//!
//! A cold giant's driver compile took the game's process up by 1.5 to 7.8 GB, and loading the
//! same pipeline from its driver blob 0.1 to 0.3 GB (play-45: all 9 giants of a separate process
//! loaded in the game in 0.5 to 1.6 s). So the game hands the compile to `xcl compile-helper`: a
//! child with its own device that compiles the DXIL with the same root signature, flags and node
//! mask and writes the driver's blob where `driver_cache` looks; the game then makes the pipeline
//! from that blob. The child does no GPU work: it compiles one pipeline and exits.
//!
//! The child runs in a job object whose memory limit is what the governor reserved for it, so it
//! cannot take the game past its reservation; while it runs, the memory available is watched and
//! the child is ended when it falls under `KILL_BELOW` (the run's watchdog stops at 1 GB). An
//! ended or outgrown compile is queued again at its measured peak (job accounting), and one that
//! needs more than the PC's memory less 1 GB is refused as never fitting, its peak logged.
//!
//! The first try reserves `FIRST_PER_BYTE` per byte of DXIL plus the child's own `BASELINE`,
//! not the curve's guard for unmeasured sizes (3,380 B per byte from 4 MiB): the job limit is what
//! protects the floor, and a try that outgrows it says what the compile really takes.

use super::*;
use std::path::Path;
use std::time::{Duration, Instant};

/// committed by a helper before its compile: its own runtime and device (`warm_children`'s
/// `CHILD_BASELINE`)
const BASELINE: u64 = 1 << 30;
/// the first try's reservation for each byte of DXIL: measured peaks were 3.09 MB of DXIL to
/// 3,034 MiB (~1,030 B per byte) and 3.5 MB to 4,468 MiB (~1,340)
const FIRST_PER_BYTE: u64 = 1400;
/// tries before a helper's compile counts as failed
const TRIES: usize = 4;
/// the memory available under which a running helper is ended and its compile queued again
const KILL_BELOW: u64 = URGENT_FLOOR + (256 << 20);
const MAGIC: &[u8; 8] = b"XCLGH001";

/// exit codes of `xcl compile-helper`
const EXIT_REQUEST: i32 = 2;
const EXIT_NO_DEVICE: i32 = 3;
const EXIT_REJECTED: i32 = 4;
const EXIT_OUT_OF_MEMORY: i32 = 5;
const EXIT_NO_BLOB: i32 = 6;
/// the code a helper ended for the memory gets (`TerminateJobObject`)
const EXIT_KILLED: u32 = 0xC0DE_0001;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_GIANT_HELPER").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: cold giant compute compiles run in the game's process (XCL_GIANT_HELPER=0)");
        }
        on
    })
}

/// a helper could not be started at all this run (spawn, job object): the game compiles itself
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);
/// the largest measured peak of a helper's compile of each DXIL (`fnv64`), ended or not
static PEAKS: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);
static HELPERS: AtomicU64 = AtomicU64::new(0);

/// What became of a cold giant given to `build`.
pub(super) enum End {
    /// its driver blob is on disk: the game makes the pipeline from it
    Blob,
    /// not built: put off for memory, or refused or failed and noted
    Refused,
    /// no helper: the game compiles it itself
    InProcess,
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Done { peak: u64, took: Duration },
    OutgrewLimit { peak: u64 },
    Killed { peak: u64, available: u64 },
    Unavailable(String),
    Failed(String),
}

/// The first try's reservation: the measured peak of an earlier try and a quarter, or
/// `FIRST_PER_BYTE`, each with the child's `BASELINE`.
fn first_reserve(dxil_len: usize, measured: Option<u64>) -> u64 {
    let curve = (dxil_len as u64).saturating_mul(FIRST_PER_BYTE).saturating_add(BASELINE);
    measured.map_or(curve, |peak| curve.max(peak.saturating_add(peak / 4)))
}

/// The reservation after a try that was ended (`killed`: the peak so far is a lower bound) or
/// outgrew its limit: half as much again at least.
fn next_reserve(reserve: u64, peak: u64, killed: bool) -> u64 {
    let measured = peak.saturating_add(peak / 4);
    if killed {
        reserve.max(measured)
    } else {
        reserve.saturating_add(reserve / 2).max(measured)
    }
}

fn encode(dxil: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], flags: i32, node_mask: u32, out: &Path) -> Vec<u8> {
    let out = out.to_string_lossy();
    let mut b = Vec::with_capacity(dxil.len() + out.len() + 64);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&flags.to_le_bytes());
    b.extend_from_slice(&node_mask.to_le_bytes());
    b.extend_from_slice(&(samplers.len() as u32).to_le_bytes());
    for (words, flag) in samplers {
        for w in words {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b.push(*flag as u8);
    }
    b.extend_from_slice(&(out.len() as u32).to_le_bytes());
    b.extend_from_slice(out.as_bytes());
    b.extend_from_slice(&(dxil.len() as u32).to_le_bytes());
    b.extend_from_slice(dxil);
    b
}

struct Request {
    flags: i32,
    node_mask: u32,
    samplers: Vec<crate::gpu::recomp::pso::StaticSampler>,
    out: std::path::PathBuf,
    dxil: Vec<u8>,
}

fn decode(b: &[u8]) -> Option<Request> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            if self.0.len() < n {
                return None;
            }
            let (head, rest) = self.0.split_at(n);
            self.0 = rest;
            Some(head)
        }
        fn u32(&mut self) -> Option<u32> {
            self.take(4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        }
    }
    let mut r = Reader(b);
    if r.take(8)? != MAGIC {
        return None;
    }
    let flags = r.u32()? as i32;
    let node_mask = r.u32()?;
    let count = r.u32()? as usize;
    let mut samplers = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let words = [r.u32()?, r.u32()?, r.u32()?, r.u32()?];
        samplers.push((words, r.take(1)?[0] != 0));
    }
    let out_len = r.u32()? as usize;
    let out = std::path::PathBuf::from(String::from_utf8(r.take(out_len)?.to_vec()).ok()?);
    let dxil_len = r.u32()? as usize;
    let dxil = r.take(dxil_len)?.to_vec();
    r.0.is_empty().then_some(Request { flags, node_mask, samplers, out, dxil })
}

/// Build the cold giant `dxil` (its root signature from `samplers`, `flags`, `node_mask`)
/// in a helper, its blob to `out` (`driver_cache::compute_path`), within the memory the
/// governor gives it (`driver_slot`, reserved at `HELPER_COST`).
pub(super) fn build(dxil: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], flags: i32, node_mask: u32, out: &Path, what: &dyn Fn() -> String) -> End {
    if UNAVAILABLE.load(Ordering::Relaxed) {
        return End::InProcess;
    }
    let id = crate::gpu::shader::fnv64(dxil);
    let curve = compile_cost(dxil.len()).0;
    let physical = total_physical();
    let mut reserve = first_reserve(dxil.len(), PEAKS.lock().unwrap().as_ref().and_then(|m| m.get(&id).copied()));
    // (a try ended because the memory available fell is not counted: it waits for the memory
    // again, as any compile that does not fit now)
    let (mut attempt, mut dips) = (1, 0u32);
    while attempt <= TRIES {
        if physical != 0 && reserve.saturating_add(URGENT_FLOOR) > physical {
            let why = format!(
                "cannot build on this PC: {} ({} KB of DXIL) needs more than {} MiB in its helper, {} MiB of physical memory less the {} MiB floor",
                what(), dxil.len() >> 10, reserve >> 20, physical >> 20, URGENT_FLOOR >> 20
            );
            COMPILE_FAILURES.fetch_add(1, Ordering::Relaxed);
            CANNOT_BUILD.fetch_add(1, Ordering::Relaxed);
            crate::xlog_warn!("D3D12.X: giant helper: {why}");
            note_translation_error(why);
            return End::Refused;
        }
        HELPER_COST.with(|c| c.set(Some(reserve)));
        let slot = driver_slot(dxil.len(), false, false, what);
        HELPER_COST.with(|c| c.set(None));
        let Ok(slot) = slot else { return End::Refused };
        let n = HELPERS.fetch_add(1, Ordering::Relaxed) + 1;
        crate::xlog_info!(
            "D3D12.X: giant helper #{n}: {} ({} KB of DXIL) compiles in a helper process within {} MiB (try {attempt} of {TRIES}; the curve's estimate {} MiB)",
            what(), dxil.len() >> 10, reserve >> 20, curve >> 20
        );
        let outcome = run(dxil, samplers, flags, node_mask, out, reserve, Duration::from_secs(super::super::pso_sched::compile_secs(dxil.len()).saturating_mul(4).max(600)));
        drop(slot);
        if let Outcome::Done { peak, .. } | Outcome::OutgrewLimit { peak } | Outcome::Killed { peak, .. } = outcome {
            let mut peaks = PEAKS.lock().unwrap();
            let kept = peaks.get_or_insert_with(HashMap::new).entry(id).or_insert(0);
            *kept = (*kept).max(peak);
        }
        match outcome {
            Outcome::Done { peak, took } => {
                crate::xlog_info!(
                    "D3D12.X: giant helper #{n}: {} built in {:.1} s, peak {} MiB of its {} MiB limit ({} MiB the child's own); the game makes it from the driver blob",
                    what(), took.as_secs_f64(), peak >> 20, reserve >> 20, BASELINE >> 20
                );
                return End::Blob;
            }
            Outcome::OutgrewLimit { peak } => {
                let next = next_reserve(reserve, peak, false);
                crate::xlog_info!("D3D12.X: giant helper #{n}: {} outgrew its {} MiB limit (peak {} MiB): queued again at {} MiB", what(), reserve >> 20, peak >> 20, next >> 20);
                reserve = next;
                attempt += 1;
            }
            Outcome::Killed { peak, available } => {
                let next = next_reserve(reserve, peak, true);
                dips += 1;
                crate::xlog_info!(
                    "D3D12.X: giant helper #{n}: {} ended at {} MiB: the memory available fell to {} MiB (under {} MiB); queued again at {} MiB, still try {attempt} of {TRIES} (ended for memory {dips} times, not counted)",
                    what(), peak >> 20, available >> 20, KILL_BELOW >> 20, next >> 20
                );
                reserve = next;
            }
            Outcome::Unavailable(why) => {
                UNAVAILABLE.store(true, Ordering::Relaxed);
                crate::xlog_warn!("D3D12.X: giant helper: no helper process ({why}): cold giants compile in the game's process for the rest of the run");
                return End::InProcess;
            }
            Outcome::Failed(why) => {
                note_translation_error(format!("{}: the giant helper failed: {why}", what()));
                crate::xlog_warn!("D3D12.X: giant helper #{n}: {} failed: {why} – not built, its binds skip", what());
                return End::Refused;
            }
        }
    }
    note_translation_error(format!("{}: the giant helper did not build it in {TRIES} tries", what()));
    crate::xlog_warn!("D3D12.X: giant helper: {} not built in {TRIES} tries (last limit {} MiB) – its binds skip", what(), reserve >> 20);
    End::Refused
}

/// A job object that limits its process's committed memory to `limit` and ends it with the game.
struct Job(windows_sys::Win32::Foundation::HANDLE);

impl Job {
    fn new(limit: u64) -> std::io::Result<Job> {
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        };
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let job = Job(job);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            limits.ProcessMemoryLimit = limit as usize;
            if SetInformationJobObject(job.0, JobObjectExtendedLimitInformation, (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), std::mem::size_of_val(&limits) as u32) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(job)
        }
    }

    fn assign(&self, child: &std::process::Child) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        if unsafe { windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(self.0, child.as_raw_handle()) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// the most memory its process committed
    fn peak(&self) -> u64 {
        use windows_sys::Win32::System::JobObjects::{JobObjectExtendedLimitInformation, QueryInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION};
        unsafe {
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            let ok = QueryInformationJobObject(self.0, JobObjectExtendedLimitInformation, (&mut info as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), std::mem::size_of_val(&info) as u32, std::ptr::null_mut());
            if ok == 0 { 0 } else { info.PeakProcessMemoryUsed as u64 }
        }
    }

    fn end(&self) {
        unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0, EXIT_KILLED) };
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

fn run(dxil: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], flags: i32, node_mask: u32, out: &Path, limit: u64, timeout: Duration) -> Outcome {
    use std::os::windows::process::CommandExt;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = crate::gpu::recomp::dxc::cache_dir().join("helper");
    let request = dir.join(format!("{}-{}.req", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&request, encode(dxil, samplers, flags, node_mask, out))) {
        return Outcome::Unavailable(format!("cannot write its request {}: {e}", request.display()));
    }
    struct Gone<'a>(&'a Path);
    impl Drop for Gone<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }
    let _gone = Gone(&request);
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => return Outcome::Unavailable(format!("no path to xcl.exe: {e}")),
    };
    let job = match Job::new(limit) {
        Ok(job) => job,
        Err(e) => return Outcome::Unavailable(format!("no job object: {e}")),
    };
    let mut child = match std::process::Command::new(&exe)
        .arg("compile-helper")
        .arg(&request)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .creation_flags(windows_sys::Win32::System::Threading::CREATE_SUSPENDED | windows_sys::Win32::System::Threading::CREATE_NO_WINDOW)
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return Outcome::Unavailable(format!("cannot start {}: {e}", exe.display())),
    };
    if let Err(e) = job.assign(&child).and_then(|_| super::super::warm_children::resume(child.id())) {
        let _ = child.kill();
        let _ = child.wait();
        return Outcome::Unavailable(format!("cannot limit or start the helper: {e}"));
    }
    let started = Instant::now();
    let (mut killed, mut late, mut left_out) = (None, false, false);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => {
                job.end();
                let _ = child.wait();
                return Outcome::Failed(format!("lost the helper: {e}"));
            }
        }
        let available = available_physical();
        if killed.is_none() && available != 0 && available < KILL_BELOW {
            killed = Some(available);
            job.end();
        } else if !late && started.elapsed() > timeout {
            late = true;
            job.end();
        } else if !left_out && super::super::pso_async::building_left_out() {
            left_out = true;
            job.end();
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    let (peak, code) = (job.peak(), status.code().unwrap_or(-1));
    if left_out {
        return Outcome::Failed(format!("its pipeline was left out while the helper compiled it: the helper was ended after {:.1} s (peak {} MiB)", started.elapsed().as_secs_f64(), peak >> 20));
    }
    if let Some(available) = killed {
        return Outcome::Killed { peak, available };
    }
    if late {
        return Outcome::Failed(format!("the helper took longer than {} s and was ended (peak {} MiB)", timeout.as_secs(), peak >> 20));
    }
    match code {
        0 if driver_cache::kept(out) => Outcome::Done { peak, took: started.elapsed() },
        0 => Outcome::Failed("the helper left no driver blob".into()),
        EXIT_NO_DEVICE => Outcome::Unavailable("the helper could not make a device".into()),
        EXIT_REQUEST => Outcome::Unavailable("the helper could not read its request".into()),
        _ if code == EXIT_OUT_OF_MEMORY || peak.saturating_add(64 << 20) >= limit => Outcome::OutgrewLimit { peak },
        EXIT_REJECTED => Outcome::Failed("the driver rejected the pipeline in the helper".into()),
        _ => Outcome::Failed(format!("the helper ended with {:#x} (peak {} MiB of {} MiB)", code as u32, peak >> 20, limit >> 20)),
    }
}

/// `xcl compile-helper <request>`: one cold giant's driver compile for the game (`build`), its
/// blob written where `driver_cache` looks. No GPU work: a device, one pipeline, its blob.
pub fn child(request: &Path) -> i32 {
    WARM_UP_PROCESS.store(true, Ordering::Relaxed);
    let Some(req) = std::fs::read(request).ok().and_then(|b| decode(&b)) else {
        eprintln!("xcl compile-helper: cannot read the request {}", request.display());
        return EXIT_REQUEST;
    };
    if let Err(e) = init() {
        eprintln!("xcl compile-helper: no device: {e}");
        return EXIT_NO_DEVICE;
    }
    unsafe {
        let dev = device_raw();
        let Some(rs) = abi_rs_for(dev, &req.samplers) else {
            eprintln!("xcl compile-helper: no root signature: {}", take_last_note().unwrap_or_default());
            return EXIT_REJECTED;
        };
        let mut desc: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::mem::zeroed();
        desc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
        desc.CS = bytecode(&req.dxil);
        desc.NodeMask = req.node_mask;
        desc.Flags = D3D12_PIPELINE_STATE_FLAGS(req.flags);
        let f: PsoFn = std::mem::transmute(orig_fn(dev, slot_of(dev, "CreateComputePipelineState")));
        let iid = pc_iid(Kind::PipelineState);
        let mut pso = std::ptr::null_mut();
        let started = Instant::now();
        let hr = finish_pso(dev, f, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &iid, &mut pso);
        if hr < 0 || pso.is_null() {
            eprintln!("xcl compile-helper: the driver rejected the pipeline ({hr:#x})");
            return if hr == windows::Win32::Foundation::E_OUTOFMEMORY.0 { EXIT_OUT_OF_MEMORY } else { EXIT_REJECTED };
        }
        let Some(blob) = pso_blob(pso) else {
            eprintln!("xcl compile-helper: the driver gave no blob");
            return EXIT_NO_BLOB;
        };
        if !driver_cache::write(&req.out, &blob) {
            eprintln!("xcl compile-helper: cannot write {}", req.out.display());
            return EXIT_NO_BLOB;
        }
        eprintln!("xcl compile-helper: {} B of DXIL built in {} ms, {} KB blob", req.dxil.len(), started.elapsed().as_millis(), blob.len() >> 10);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_reads_back_as_written() {
        let samplers = [([1, 2, 3, 4], true), ([5, 6, 7, 8], false)];
        let out = Path::new("C:\\cache\\abc-10-0.pso");
        let b = encode(&[9u8; 37], &samplers, 0x4, 1, out);
        let r = decode(&b).expect("decoded");
        assert_eq!((r.flags, r.node_mask, r.samplers.as_slice(), r.out.as_path(), r.dxil.as_slice()), (0x4, 1, &samplers[..], out, &[9u8; 37][..]));
        assert!(decode(&b[..b.len() - 1]).is_none(), "cut short");
        let mut longer = b.clone();
        longer.push(0);
        assert!(decode(&longer).is_none(), "trailing bytes");
    }

    #[test]
    fn the_first_try_reserves_far_less_than_the_guard_and_a_measured_peak_raises_it() {
        const MIB: u64 = 1 << 20;
        let fsr4_18 = 7_698 << 10;
        let first = first_reserve(fsr4_18, None);
        assert_eq!(first >> 20, (7_698 * 1024 * 1400 >> 20) + 1024);
        assert!(first < 25_411 * MIB / 2, "the guard's estimate was 25,411 MiB");
        assert_eq!(first_reserve(fsr4_18, Some(16_000 * MIB)), 20_000 * MIB, "a measured peak and a quarter");
        assert_eq!(first_reserve(fsr4_18, Some(MIB)), first, "never under the first try");
    }

    #[test]
    fn a_try_ended_or_outgrown_is_queued_again_higher() {
        const MIB: u64 = 1 << 20;
        assert_eq!(next_reserve(8_000 * MIB, 8_000 * MIB, false), 12_000 * MIB, "outgrew: half as much again");
        assert_eq!(next_reserve(8_000 * MIB, 10_000 * MIB, false), 12_500 * MIB, "or the peak and a quarter");
        assert_eq!(next_reserve(8_000 * MIB, 3_000 * MIB, true), 8_000 * MIB, "ended for the game's memory: the same, waiting for room");
        assert_eq!(next_reserve(8_000 * MIB, 7_200 * MIB, true), 9_000 * MIB);
    }
}
