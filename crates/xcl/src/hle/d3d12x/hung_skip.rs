//! The hung-pipeline guard (`XCL_HUNG_SKIP=0` turns it off): the compute pipelines a GPU hang
//! caught in flight, written to `%LOCALAPPDATA%\xcl\hung-pipelines.txt` at the removal and skipped
//! at the next start of the same translator. A title that hangs on one of its dispatches then gets
//! past it at the next launch instead of hanging the same way again, and the log names every
//! dispatch the guard skipped (the draw gate counts them, `bind::shader_skip`).
//!
//! The record is keyed by the translator (`recomp::pso::translator_id`): another translation may no
//! longer hang, so a record of a different translator is ignored and removed rather than skipping
//! pipelines a fix has already cured. Only a hang is recorded (`DXGI_ERROR_DEVICE_HUNG`): a removal
//! for another reason says nothing about the pipelines that happened to be in flight.
//!
//! What the GPU had in flight comes from the submit watch (`submit_watch::record_hung`), so the
//! guard needs the watch on, which it is by default.

use std::path::PathBuf;
use std::sync::OnceLock;

/// The format of the file: a line this module does not know is read as a pipeline hash.
const VERSION: u32 = 1;

/// At most this many pipelines are recorded and skipped: a hang is one or a few dispatches, and a
/// removal that names thousands of them (a queue that never got to run its own work) must not turn
/// into a title whose whole compute pass is skipped.
const MOST: usize = 64;

/// `DXGI_ERROR_DEVICE_HUNG`: the removal the guard records.
const DEVICE_HUNG: u32 = 0x887A_0006;

/// `%LOCALAPPDATA%\xcl\hung-pipelines.txt`: what one run leaves the next. Global rather than under
/// the title's own directory, which a hang in the title's first seconds can leave unwritten.
fn path() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .filter(|v| !v.is_empty())
        .map(|dir| PathBuf::from(dir).join("xcl").join("hung-pipelines.txt"))
}

/// Whether the guard records a hang and skips what it recorded (`XCL_HUNG_SKIP=0`: neither).
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_HUNG_SKIP").is_ok_and(|v| v.trim() == "0"))
}

/// The compute pipeline hashes the draw gate skips in this run: those a previous run of the same
/// translator recorded at a hang. Read once, at the first bind the gate looks at.
pub(super) fn hashes() -> &'static [u64] {
    static HASHES: OnceLock<Vec<u64>> = OnceLock::new();
    HASHES.get_or_init(|| {
        if !enabled() {
            return Vec::new();
        }
        let Some(path) = path() else {
            crate::xlog_warn!("hung skip: no %LOCALAPPDATA%, so no record of a previous hang can be read or written");
            return Vec::new();
        };
        let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
        let record = Record::parse(&text);
        let translator = crate::gpu::recomp::pso::translator_id();
        if record.version != VERSION {
            crate::xlog_warn!(
                "hung skip: {} is version {} of the file, this build reads {VERSION}: ignored and removed",
                path.display(),
                record.version
            );
            let _ = std::fs::remove_file(&path);
            return Vec::new();
        }
        if record.translator != translator {
            crate::xlog_info!(
                "hung skip: {} was written at a hang by translator {:#018x}, not this build's {translator:#018x}: ignored and removed (another translation may no longer hang)",
                path.display(),
                record.translator
            );
            let _ = std::fs::remove_file(&path);
            return Vec::new();
        }
        if record.hashes.is_empty() {
            return Vec::new();
        }
        let named: Vec<String> = record.hashes.iter().map(|&h| format!("{h:016x}")).collect();
        crate::xlog_info!(
            "hung skip: {} compute pipelines that were in flight at a GPU hang are skipped this run (translator {translator:#018x}, {}): {}; XCL_HUNG_SKIP=0 turns it off",
            record.hashes.len(),
            path.display(),
            named.join(" ")
        );
        record.hashes
    })
}

/// The file's contents: its version, the translator that wrote it, and the pipeline hashes.
#[derive(Default, PartialEq, Debug)]
struct Record {
    version: u32,
    translator: u64,
    hashes: Vec<u64>,
}

impl Record {
    fn parse(text: &str) -> Record {
        let mut record = Record::default();
        let mut version = false;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(v) = line.strip_prefix("version ") {
                record.version = v.trim().parse().unwrap_or(0);
                version = true;
                continue;
            }
            if let Some(v) = line.strip_prefix("translator ") {
                record.translator = hex(v.trim()).unwrap_or(0);
                continue;
            }
            let Some(hash) = line.split_whitespace().next().and_then(hex) else { continue };
            if hash != 0 && !record.hashes.contains(&hash) && record.hashes.len() < MOST {
                record.hashes.push(hash);
            }
        }
        if !version {
            record.version = 0;
        }
        record
    }

    fn render(&self) -> String {
        let mut s = String::from("# xcl hung pipelines: what was in flight at a GPU hang. A start of the same\n# translator skips these dispatches (XCL_HUNG_SKIP=0 turns that off); the file is\n# ignored and removed when the translator differs.\n");
        s.push_str(&format!("version {}\n", self.version));
        s.push_str(&format!("translator {:#018x}\n", self.translator));
        for entry in &self.hashes {
            s.push_str(&format!("{entry:016x}\n"));
        }
        s
    }
}

fn hex(word: &str) -> Option<u64> {
    u64::from_str_radix(word.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()
}

/// The device was removed for `reason`: record the compute pipelines the GPU had in flight
/// (`dispatches`: the title's pipeline and thread groups, as the submit watch keeps them), for the
/// next start of the same translator to skip. Runs at the removal, before the process is stopped, so
/// it writes the file before anything else about the process goes away.
pub(super) fn record_at_removal(reason: u32, dispatches: &[(usize, Option<[u32; 3]>)]) {
    if !enabled() {
        return;
    }
    if reason != DEVICE_HUNG {
        crate::xlog_info!(
            "hung skip: the device was removed for {reason:#010x}, not a hang ({DEVICE_HUNG:#010x}): the pipelines in flight are not recorded"
        );
        return;
    }
    let Some(path) = path() else {
        crate::xlog_warn!("hung skip: no %LOCALAPPDATA%: the pipeline this hang caught in flight cannot be recorded for the next start");
        return;
    };
    let mut counts: Vec<(u64, u32)> = Vec::new();
    let mut unnamed = 0usize;
    for &(title, _) in dispatches {
        let hash = super::pso_async::built_real(title).and_then(super::dispatch_timing::profile_shader);
        match hash {
            Some(hash) => match counts.iter_mut().find(|(h, _)| *h == hash) {
                Some((_, n)) => *n += 1,
                None => counts.push((hash, 1)),
            },
            None => unnamed += 1,
        }
    }
    counts.sort_by_key(|&(hash, n)| (std::cmp::Reverse(n), hash));
    counts.truncate(MOST);
    if counts.is_empty() {
        crate::xlog_info!(
            "hung skip: no compute dispatch was in flight at the hang ({} dispatches recorded, none of their pipelines built): nothing recorded",
            dispatches.len()
        );
        return;
    }
    let record = Record { version: VERSION, translator: crate::gpu::recomp::pso::translator_id(), hashes: counts.iter().map(|&(hash, _)| hash).collect() };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let text = record.render();
    let written = std::fs::write(&path, &text).is_ok();
    if written {
        crate::xlog_info!(
            "hung skip: the GPU hung with {} compute pipelines in flight ({} dispatches{}); wrote them to {} for the next start of this translator to skip: {}",
            counts.len(),
            dispatches.len(),
            if unnamed == 0 { String::new() } else { format!(", {unnamed} whose pipeline was not built") },
            path.display(),
            counts.iter().map(|(hash, n)| format!("{hash:016x} x{n}")).collect::<Vec<_>>().join(" ")
        );
    } else {
        crate::xlog_warn!("hung skip: {} could not be written: the pipeline this hang caught in flight is not recorded", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_record_reads_back_the_same() {
        let record = Record { version: VERSION, translator: 0x1234_5678_9abc_def1, hashes: vec![0x150c_a447_ebf5_7311, 0xffff_0000_0000_0001] };
        assert_eq!(Record::parse(&record.render()), record);
    }

    #[test]
    fn a_line_the_record_does_not_use_is_read_where_it_holds_a_hash() {
        let text = "# a comment\nversion 1\ntranslator 0x0000000000000001\n150ca447ebf57311 # CS of 12x1x1 groups\n0x150ca447ebf57311\n150ca447ebf57311\nnonsense\n";
        let record = Record::parse(text);
        assert_eq!(record.version, 1);
        assert_eq!(record.translator, 1);
        assert_eq!(record.hashes, vec![0x150c_a447_ebf5_7311], "a hash is kept once and only once");
    }

    #[test]
    fn a_file_without_a_version_or_a_translator_is_not_this_modules() {
        let record = Record::parse("150ca447ebf57311\n");
        assert_eq!(record.version, 0);
        assert_eq!(record.translator, 0);
        assert_eq!(record.hashes, vec![0x150c_a447_ebf5_7311]);
    }

    #[test]
    fn at_most_the_first_pipelines_of_a_record_are_skipped() {
        let text: String = (0..MOST + 8).map(|n| format!("{:016x}\n", n as u64 + 1)).collect();
        assert_eq!(Record::parse(&text).hashes.len(), MOST);
    }
}
