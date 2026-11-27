//! Every pipeline a title made, kept between runs (`<DXIL cache>\pipelines\`), so `xcl warm-up`
//! builds all of them before a run, not only the ones some run happened to bind: a pipeline the
//! warm-up never saw costs its first bind a driver compile (a retail game's big compute programs:
//! 15 – 180 s each), and binds waiting that long tripped the game's own GPU watchdog.
//!
//! A pipeline is kept as the title described it (`Record`): the description with its pointers
//! taken out, the console shaders it names (each with where the description points at it; the
//! bytes kept once, under their hash, in `shaders\`) and its input layout. Its key is a hash of
//! all that, and a key kept already is not written again. The translation is not kept: the
//! warm-up translates the shaders with the XCL it runs, so after a translator change it builds
//! every pipeline from the new translation. `XCL_PSO_MANIFEST=0`: nothing is kept.
//!
//! Each game run also notes every kept pipeline it made, once, with the run and the translation
//! era it was made in (`created.bin`; the runs counted in `runs`): a pipeline no run made in the
//! last `STALE_AFTER_RUNS` is left out of the budgeted warm-up (`xcl warm-up --full` still builds
//! it). `XCL_PSO_RUN_COUNTS=0`: this run's pipelines are noted, but the run is not counted (a
//! short check would otherwise age everything it never reached).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// bytes of a `D3D12_INPUT_ELEMENT_DESC`
pub const ELEMENT: usize = 32;
const VERSION: u8 = 1;
/// a record whose description is in `pso_canon`'s one form
const CANONICAL: u8 = 2;
/// runs after which a pipeline none of them made is stale
pub const STALE_AFTER_RUNS: u32 = 3;
/// bytes of a `created.bin` (and `bound.bin`) entry: the key, the translation era, the run
const CREATED: usize = 20;

/// How the pipeline was made (which of XCL's builds makes it again).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Compute = 0,
    Graphics = 1,
    /// a pipeline state stream with a mesh shader
    MeshStream = 2,
    /// a pipeline state stream with a vertex shader
    VertexStream = 3,
}

impl Kind {
    fn from_u8(v: u8) -> Option<Kind> {
        [Kind::Compute, Kind::Graphics, Kind::MeshStream, Kind::VertexStream].into_iter().find(|k| *k as u8 == v)
    }
}

/// One pipeline as the title described it.
#[derive(Clone, PartialEq, Debug)]
pub struct Record {
    pub kind: Kind,
    /// the description (a desc structure or a stream's bytes), its pointers 0
    pub desc: Vec<u8>,
    /// each console shader, with the offset in `desc` of the pointer to it
    pub shaders: Vec<(u32, Arc<Vec<u8>>)>,
    /// the input layout: the offset in `desc` of the pointer to its elements, and each element
    /// (its name pointer 0) with its semantic name
    pub layout: Option<(u32, Vec<(Vec<u8>, [u8; ELEMENT])>)>,
    /// the description is in `pso_canon`'s one form (kept in the version, so in the key)
    pub canonical: bool,
}

impl Record {
    /// What is kept of it: its shaders as their hash and length.
    fn encode(&self) -> Vec<u8> {
        let mut out = vec![if self.canonical { CANONICAL } else { VERSION }, self.kind as u8];
        out.extend_from_slice(&(self.desc.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.desc);
        out.extend_from_slice(&(self.shaders.len() as u32).to_le_bytes());
        for (at, bytes) in &self.shaders {
            out.extend_from_slice(&at.to_le_bytes());
            out.extend_from_slice(&crate::gpu::shader::fnv64(bytes).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        }
        match &self.layout {
            None => out.push(0),
            Some((at, elems)) => {
                out.push(1);
                out.extend_from_slice(&at.to_le_bytes());
                out.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for (name, e) in elems {
                    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                    out.extend_from_slice(name);
                    out.extend_from_slice(e);
                }
            }
        }
        out
    }

    /// Its console shaders' bytes together (the warm-up builds the largest first).
    pub fn size(&self) -> usize {
        self.shaders.iter().map(|(_, s)| s.len()).sum()
    }

    /// What it is kept by.
    pub fn key(&self) -> u64 {
        key_of(&self.encode())
    }
}

/// The stage of the shader record `r` points at from `at` (for the log).
pub fn stage_at(r: &Record, at: u32) -> &'static str {
    use std::mem::offset_of;
    use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC as G;
    match r.kind {
        Kind::Compute => "CS",
        Kind::Graphics => [(offset_of!(G, VS), "VS"), (offset_of!(G, PS), "PS"), (offset_of!(G, DS), "DS"), (offset_of!(G, HS), "HS"), (offset_of!(G, GS), "GS")]
            .into_iter()
            .find(|o| o.0 as u32 == at)
            .map_or("?", |o| o.1),
        // a stream's subobject type is 8 bytes before its payload
        Kind::MeshStream | Kind::VertexStream => match (at as usize).checked_sub(8).and_then(|i| r.desc.get(i..i + 4)).map(|b| u32::from_le_bytes(b.try_into().unwrap())) {
            Some(1) => "VS",
            Some(2) => "PS",
            Some(3) => "DS",
            Some(4) => "HS",
            Some(5) => "GS",
            Some(6) => "CS",
            Some(24) => "AS",
            Some(25) => "MS",
            _ => "?",
        },
    }
}

fn key_of(encoded: &[u8]) -> u64 {
    crate::gpu::shader::fnv64(encoded) | 1
}

pub(super) fn shaders_key(hashes: impl Iterator<Item = u64>) -> u64 {
    hashes.fold(0xcbf2_9ce4_8422_2325u64, |h, hash| (h ^ hash).wrapping_mul(0x100_0000_01b3)) | 1
}

fn encoded_shaders_key(e: &[u8]) -> Option<u64> {
    if !matches!(e.first(), Some(&VERSION | &CANONICAL)) {
        return None;
    }
    let word = |p: usize| Some(u32::from_le_bytes(e.get(p..p + 4)?.try_into().ok()?) as usize);
    let p = 6usize.checked_add(word(2)?)?;
    let bytes = word(p)?.checked_mul(16)?;
    let n = p.checked_add(4)?;
    let hashes = e.get(n..n.checked_add(bytes)?)?;
    Some(shaders_key(hashes.chunks_exact(16).map(|s| u64::from_le_bytes(s[4..12].try_into().unwrap()))))
}

/// A kept record (None: not one this XCL reads, or a shader it names is missing).
fn decode(b: &[u8], shader: &mut impl FnMut(u64, u32) -> Option<Arc<Vec<u8>>>) -> Option<Record> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(at..at.checked_add(n)?)?;
        at += n;
        Some(s)
    };
    let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    let head = take(2)?;
    if head[0] != VERSION && head[0] != CANONICAL {
        return None;
    }
    let canonical = head[0] == CANONICAL;
    let kind = Kind::from_u8(head[1])?;
    let n = u32_(take(4)?) as usize;
    let desc = take(n)?.to_vec();
    let n = u32_(take(4)?) as usize;
    let mut shaders = Vec::with_capacity(n.min(16));
    for _ in 0..n {
        let off = u32_(take(4)?);
        let hash = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let len = u32_(take(4)?);
        shaders.push((off, shader(hash, len)?));
    }
    let layout = match take(1)?[0] {
        0 => None,
        _ => {
            let off = u32_(take(4)?);
            let n = u32_(take(4)?) as usize;
            let mut elems = Vec::with_capacity(n.min(64));
            for _ in 0..n {
                let len = u16::from_le_bytes(take(2)?.try_into().unwrap()) as usize;
                let name = take(len)?.to_vec();
                elems.push((name, take(ELEMENT)?.try_into().unwrap()));
            }
            Some((off, elems))
        }
    };
    (at == b.len()).then_some(Record { kind, desc, shaders, layout, canonical })
}

/// The kept entries' bytes: each a length and an encoded record (a torn last entry is left out).
fn entries(file: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(len) = file.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize) {
        let Some(e) = file.get(at + 4..at + 4 + len) else { break };
        out.push(e);
        at += 4 + len;
    }
    out
}

/// The keys of the kept entries (`entries`, `key_of`), the file read an entry at a time: after a
/// hundred runs it holds 954,785 entries in 718 MB, four fifths of it zeros, and read whole it was
/// a game run's highest Rust heap (749 MB at its start, 134 MB live). `XCL_PSO_MANIFEST_STREAM=0`:
/// read whole, as before. The entries a key list covers (`kept_keys`) are not read: only its
/// keys, and the entries added since.
fn keys_in(path: &Path) -> HashSet<u64> {
    use std::io::{Read, Seek};
    let Ok(file) = std::fs::File::open(path) else { return HashSet::new() };
    let len = file.metadata().map_or(0, |m| m.len());
    let listed = keys_listed();
    let t0 = std::time::Instant::now();
    let (mut keys, start) = listed.then(|| kept_keys(path, &file, len)).flatten().unwrap_or_default();
    let from_list = keys.len();
    let mut end = start;
    if (&file).seek(std::io::SeekFrom::Start(start)).is_ok() {
        let mut left = len - start;
        let mut reader = std::io::BufReader::with_capacity(1 << 20, &file);
        let (mut len, mut entry) = ([0u8; 4], Vec::new());
        while left >= 4 && reader.read_exact(&mut len).is_ok() {
            let n = u32::from_le_bytes(len) as u64;
            left -= 4;
            if n > left {
                break;
            }
            entry.resize(n as usize, 0);
            if reader.read_exact(&mut entry).is_err() {
                break;
            }
            left -= n;
            end += 4 + n;
            keys.insert(key_of(&entry));
        }
    }
    if listed && end > start {
        keep_keys(path, &file, &keys, end);
    }
    static SAID: std::sync::Once = std::sync::Once::new();
    if listed && len > 0 {
        SAID.call_once(|| {
            crate::xlog_info!(
                "pipeline list: {} pipelines kept, {from_list} of them from pipelines.keys and {} MB of entries read in {} ms (XCL_PSO_MANIFEST_KEYS=0: the whole list read)",
                keys.len(),
                (end - start) >> 20,
                t0.elapsed().as_millis()
            )
        });
    }
    keys
}

/// `pipelines.keys` beside the list: the keys of its entries up to a length of the list, so that
/// a run reads them (8 bytes a pipeline: 7 MB and 40 ms for 959,791, against 688 MB and 855 ms
/// for the whole list with the file in memory) and only the entries added since. Made again when
/// the list grew; one that does not match the list's bytes before the length it covers (the list
/// rewritten or cut) is not used. Old builds do not read it. `XCL_PSO_MANIFEST_KEYS=0`: not read
/// or written.
const KEYS_MAGIC: &[u8; 8] = b"XCLKEYS1";
/// bytes of the list, before the length a key list covers, that the key list is checked against
const KEYS_CHECKED: u64 = 64 << 10;

fn keys_listed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PSO_MANIFEST_KEYS").map_or(true, |v| v.trim() != "0"))
}

fn keys_path(list: &Path) -> PathBuf {
    list.with_extension("keys")
}

/// What a key list covering `end` bytes of `file` is checked by: a hash of the `KEYS_CHECKED`
/// bytes before `end`, and `end` (None: they cannot be read).
fn list_check(mut file: &std::fs::File, end: u64) -> Option<u64> {
    use std::io::{Read, Seek};
    let from = end.saturating_sub(KEYS_CHECKED);
    file.seek(std::io::SeekFrom::Start(from)).ok()?;
    let mut b = vec![0u8; (end - from) as usize];
    file.read_exact(&mut b).ok()?;
    Some(crate::gpu::shader::fnv64(&b) ^ end)
}

/// The keys of list `path`'s key list (`file` the list, `len` its length) and the length of the
/// list they cover. None: there is none, or it is not one of this list.
fn kept_keys(path: &Path, file: &std::fs::File, len: u64) -> Option<(HashSet<u64>, u64)> {
    let b = std::fs::read(keys_path(path)).ok()?;
    let word = |k: usize| Some(u64::from_le_bytes(b.get(8 + 8 * k..16 + 8 * k)?.try_into().ok()?));
    if b.get(..8) != Some(&KEYS_MAGIC[..]) {
        return None;
    }
    let (covered, check, count) = (word(0)?, word(1)?, word(2)?);
    if covered > len || (b.len() as u64 - 32) != count.checked_mul(8)? || list_check(file, covered)? != check {
        return None;
    }
    Some((b[32..].chunks_exact(8).map(|k| u64::from_le_bytes(k.try_into().unwrap())).collect(), covered))
}

/// Write the key list of list `path`: `keys`, those of its first `end` bytes.
fn keep_keys(path: &Path, file: &std::fs::File, keys: &HashSet<u64>, end: u64) {
    let Some(check) = list_check(file, end) else { return };
    let mut b = Vec::with_capacity(32 + 8 * keys.len());
    b.extend_from_slice(KEYS_MAGIC);
    for w in [end, check, keys.len() as u64] {
        b.extend_from_slice(&w.to_le_bytes());
    }
    keys.iter().for_each(|k| b.extend_from_slice(&k.to_le_bytes()));
    let to = keys_path(path);
    let tmp = to.with_extension(format!("keys.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, &b).is_err() || std::fs::rename(&tmp, &to).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The list a warm-up child indexes instead of its folder's own (`Store::index`, `write_subset`).
pub(super) const WARM_LIST: &str = "XCL_WARM_LIST";

/// The entries of `keys` from the list in folder `dir`, written to `to` in the list's own form
/// (each a length and an encoded record, a key's first entry only), the list read an entry at a
/// time; how many. A warm-up's children index it (`WARM_LIST`) instead of the whole list: in the
/// 102n test (2026-11-27) each of 8 children read the 722 MB list whole, was replaced at its 2 GB
/// of private commit after a few keys, and its successor read it again; the PC fell to 61 MB.
pub(super) fn write_subset(dir: &Path, keys: &HashSet<u64>, to: &Path) -> std::io::Result<usize> {
    use std::io::{Read, Write};
    let file = std::fs::File::open(dir.join("pipelines.bin"))?;
    let mut left = file.metadata()?.len();
    let mut file = std::io::BufReader::with_capacity(1 << 20, file);
    let mut out = std::io::BufWriter::new(std::fs::File::create(to)?);
    let (mut len, mut entry, mut seen) = ([0u8; 4], Vec::new(), HashSet::new());
    while left >= 4 && file.read_exact(&mut len).is_ok() {
        let n = u32::from_le_bytes(len) as u64;
        left -= 4;
        if n > left {
            break;
        }
        entry.resize(n as usize, 0);
        file.read_exact(&mut entry)?;
        left -= n;
        let key = key_of(&entry);
        if keys.contains(&key) && seen.insert(key) {
            out.write_all(&len)?;
            out.write_all(&entry)?;
        }
    }
    out.flush()?;
    Ok(seen.len())
}

/// `Store::index`: the list's bytes and each key's first entry in them (offset, length).
pub(super) struct Index {
    dir: PathBuf,
    file: Vec<u8>,
    at: HashMap<u64, (usize, usize)>,
}

impl Index {
    /// Pipeline `key` (None: not kept, not one this XCL reads, or a shader it names is missing).
    pub(super) fn get(&self, key: u64) -> Option<Record> {
        let &(at, len) = self.at.get(&key)?;
        decode(self.file.get(at..at + len)?, &mut |hash, len| read_shader(&self.dir, hash, len))
    }
}

/// Shader `hash` of `len` bytes, as kept (None: missing or not that shader).
fn read_shader(dir: &Path, hash: u64, len: u32) -> Option<Arc<Vec<u8>>> {
    std::fs::read(shader_path(dir, hash, len)).ok().filter(|b| b.len() == len as usize && crate::gpu::shader::fnv64(b) == hash).map(Arc::new)
}

fn shader_path(dir: &Path, hash: u64, len: u32) -> PathBuf {
    dir.join("shaders").join(format!("{hash:016x}-{len:x}.bin"))
}

/// The list in folder `dir`.
pub struct Store {
    dir: PathBuf,
    /// the keys kept (in the file or written by this run)
    seen: Mutex<HashSet<u64>>,
}

/// When the game last made each kept pipeline (`Store::creations`).
pub struct Creations {
    /// runs counted
    pub runs: u32,
    /// each key's last run and translation era
    last: HashMap<u64, (u32, u64)>,
}

impl Creations {
    /// No run of the last `STALE_AFTER_RUNS` made pipeline `key` (one never noted counts as made
    /// before the first run counted).
    pub fn stale(&self, key: u64) -> bool {
        self.runs >= self.last.get(&key).map_or(0, |l| l.0) + STALE_AFTER_RUNS
    }

    /// The translation era pipeline `key` was last made in.
    pub fn era(&self, key: u64) -> Option<u64> {
        self.last.get(&key).map(|l| l.1)
    }
}

/// `xcl warm-up --full`'s record of the pipelines it went through in one translation era
/// (`Store::progress`): a pass stopped part way goes on from there.
pub struct Progress {
    file: Mutex<std::fs::File>,
    /// keys gone through by an earlier pass
    pub done: HashSet<u64>,
}

impl Progress {
    /// Pipeline `key` was gone through.
    pub fn note(&self, key: u64) {
        let _ = self.file.lock().unwrap().write_all(&key.to_le_bytes());
    }
}

/// What `Store::load` found.
pub struct Loaded {
    pub records: Vec<Record>,
    /// entries this XCL does not read or whose shaders are missing
    pub broken: usize,
    /// entries kept twice (two runs that wrote at once)
    pub repeated: usize,
}

impl Store {
    pub fn open(dir: PathBuf) -> Store {
        let seen = if std::env::var("XCL_PSO_MANIFEST_STREAM").is_ok_and(|v| v.trim() == "0") {
            let file = std::fs::read(dir.join("pipelines.bin")).unwrap_or_default();
            entries(&file).into_iter().map(key_of).collect()
        } else {
            keys_in(&dir.join("pipelines.bin"))
        };
        Store { dir, seen: Mutex::new(seen) }
    }

    /// Pipelines kept.
    pub fn len(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    /// `r`'s key, and `r` encoded if it is new (then it is to be written: `write`).
    fn note(&self, r: &Record) -> (u64, Option<Vec<u8>>) {
        let encoded = r.encode();
        let key = key_of(&encoded);
        (key, self.seen.lock().unwrap().insert(key).then_some(encoded))
    }

    /// Keep `r` (encoded: `note`): its shaders not kept yet, then its entry.
    fn write(&self, r: &Record, encoded: &[u8]) -> std::io::Result<()> {
        std::fs::create_dir_all(self.dir.join("shaders"))?;
        for (_, bytes) in &r.shaders {
            let path = shader_path(&self.dir, crate::gpu::shader::fnv64(bytes), bytes.len() as u32);
            if !path.is_file() {
                let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
                std::fs::write(&tmp, bytes.as_slice())?;
                std::fs::rename(&tmp, &path)?;
            }
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("pipelines.bin"))?;
        let mut entry = (encoded.len() as u32).to_le_bytes().to_vec();
        entry.extend_from_slice(encoded);
        f.write_all(&entry)
    }

    /// Keep `r` now, unless it is kept already (true: written).
    pub fn add(&self, r: &Record) -> std::io::Result<bool> {
        match self.note(r).1 {
            Some(encoded) => self.write(r, &encoded).map(|_| true),
            None => Ok(false),
        }
    }

    /// Every pipeline kept, each once.
    pub fn load(&self) -> Loaded {
        Self::load_filtered(&self.dir, None, |_, _| true)
    }

    pub fn load_kind(dir: &Path, kind: Kind, mut include: impl FnMut(u64, Option<u64>) -> bool) -> Loaded {
        Self::load_filtered(dir, Some(kind), |key, e| include(key, encoded_shaders_key(e)))
    }

    pub(super) fn load_keys(dir: &Path, keys: &HashSet<u64>) -> Loaded {
        Self::load_filtered(dir, None, |key, _| keys.contains(&key))
    }

    /// Each kept pipeline's entry by key, the list read once: a warm-up child decodes the keys it
    /// is given from it (`Index::get`) instead of reading the list again for each batch.
    /// A warm-up child reads the pass's own list instead when the parent wrote one (`WARM_LIST`).
    pub(super) fn index(dir: &Path) -> Index {
        let list = std::env::var_os(WARM_LIST).map_or_else(|| dir.join("pipelines.bin"), PathBuf::from);
        Self::index_at(dir, &list)
    }

    /// `index` of the list at `list`, its shaders in folder `dir`.
    pub(super) fn index_at(dir: &Path, list: &Path) -> Index {
        let file = std::fs::read(list).unwrap_or_default();
        let base = file.as_ptr() as usize;
        let mut at = HashMap::new();
        for e in entries(&file) {
            at.entry(key_of(e)).or_insert((e.as_ptr() as usize - base, e.len()));
        }
        Index { dir: dir.to_path_buf(), file, at }
    }

    pub(super) fn latest_run(dir: &Path) -> u32 {
        Self::runs_in(dir)
    }

    fn load_filtered(dir: &Path, kind: Option<Kind>, mut include: impl FnMut(u64, &[u8]) -> bool) -> Loaded {
        let file = std::fs::read(dir.join("pipelines.bin")).unwrap_or_default();
        let mut shaders: HashMap<(u64, u32), Option<Arc<Vec<u8>>>> = HashMap::new();
        let mut read = |hash: u64, len: u32| {
            shaders
                .entry((hash, len))
                .or_insert_with(|| read_shader(dir, hash, len))
                .clone()
        };
        let (mut seen, mut out) = (HashSet::new(), Loaded { records: Vec::new(), broken: 0, repeated: 0 });
        for e in entries(&file) {
            if kind.is_some_and(|kind| e.get(1) != Some(&(kind as u8))) {
                continue;
            }
            let key = key_of(e);
            if !seen.insert(key) {
                out.repeated += 1;
            } else if include(key, e) {
                if let Some(r) = decode(e, &mut read) {
                    out.records.push(r);
                } else {
                    out.broken += 1;
                }
            }
        }
        out
    }

    /// The file again with only `records`, each once (broken and repeated entries left out).
    pub fn rewrite(&self, records: &[Record]) -> std::io::Result<()> {
        let mut file = Vec::new();
        for r in records {
            let encoded = r.encode();
            file.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            file.extend_from_slice(&encoded);
        }
        let path = self.dir.join("pipelines.bin");
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, file)?;
        std::fs::rename(&tmp, &path)?;
        *self.seen.lock().unwrap() = records.iter().map(|r| key_of(&r.encode())).collect();
        Ok(())
    }

    fn runs(&self) -> u32 {
        Self::runs_in(&self.dir)
    }

    fn runs_in(dir: &Path) -> u32 {
        std::fs::read_to_string(dir.join("runs")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
    }

    pub(super) fn recently_created(dir: &Path, count: std::num::NonZeroU32) -> std::io::Result<(std::ops::RangeInclusive<u32>, HashSet<u64>)> {
        Self::recently_in(dir, "created.bin", count)
    }

    /// The pipelines the last `count` runs bound (`note_bound`), each once, in the order of its
    /// first bind in them (`bound.bin` is kept in bind order).
    pub(super) fn bound_in_order(dir: &Path, count: std::num::NonZeroU32) -> Vec<u64> {
        let last = Self::runs_in(dir);
        let runs = last.saturating_sub(count.get() - 1).max(1)..=last;
        let file = std::fs::read(dir.join("bound.bin")).unwrap_or_default();
        let mut seen = HashSet::new();
        file.chunks_exact(CREATED)
            .filter(|e| runs.contains(&u32::from_le_bytes(e[16..].try_into().unwrap())))
            .map(|e| u64::from_le_bytes(e[..8].try_into().unwrap()))
            .filter(|&key| seen.insert(key))
            .collect()
    }

    /// The pipelines the last `count` runs bound (`note_bound`); none before `bound.bin` was kept.
    pub(super) fn recently_bound(dir: &Path, count: std::num::NonZeroU32) -> HashSet<u64> {
        Self::recently_in(dir, "bound.bin", count).map_or_else(|_| HashSet::new(), |(_, keys)| keys)
    }

    fn recently_in(dir: &Path, name: &str, count: std::num::NonZeroU32) -> std::io::Result<(std::ops::RangeInclusive<u32>, HashSet<u64>)> {
        let last = Self::runs_in(dir);
        let runs = last.saturating_sub(count.get() - 1).max(1)..=last;
        let file = std::fs::read(dir.join(name))?;
        let keys = file.chunks_exact(CREATED)
            .filter(|e| runs.contains(&u32::from_le_bytes(e[16..].try_into().unwrap())))
            .map(|e| u64::from_le_bytes(e[..8].try_into().unwrap()))
            .collect();
        Ok((runs, keys))
    }

    /// A game run begins: the number its pipelines are noted with (`made`), a new one when it
    /// `counts`, else the last counted.
    pub fn begin_run(&self, counts: bool) -> std::io::Result<u32> {
        let run = self.runs() + counts as u32;
        if counts {
            std::fs::create_dir_all(&self.dir)?;
            let path = self.dir.join("runs");
            let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
            std::fs::write(&tmp, run.to_string())?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(run)
    }

    /// Run `run` made pipeline `key` in translation era `era` (`file`: `created.bin`, opened by
    /// the first).
    pub fn made(&self, file: &mut Option<std::fs::File>, key: u64, era: u64, run: u32) -> std::io::Result<()> {
        self.note_in("created.bin", file, key, era, run)
    }

    /// Run `run` bound pipeline `key` (`file`: `bound.bin`, opened by the first).
    fn bound(&self, file: &mut Option<std::fs::File>, key: u64, era: u64, run: u32) -> std::io::Result<()> {
        self.note_in("bound.bin", file, key, era, run)
    }

    fn note_in(&self, name: &str, file: &mut Option<std::fs::File>, key: u64, era: u64, run: u32) -> std::io::Result<()> {
        if file.is_none() {
            std::fs::create_dir_all(&self.dir)?;
            *file = Some(std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join(name))?);
        }
        file.as_mut().unwrap().write_all(&created_entry(key, era, run))
    }

    /// When the game last made each pipeline (a torn last entry is left out).
    pub fn creations(&self) -> Creations {
        let file = std::fs::read(self.dir.join("created.bin")).unwrap_or_default();
        let mut last: HashMap<u64, (u32, u64)> = HashMap::new();
        for e in file.chunks_exact(CREATED) {
            let key = u64::from_le_bytes(e[..8].try_into().unwrap());
            let era = u64::from_le_bytes(e[8..16].try_into().unwrap());
            let run = u32::from_le_bytes(e[16..].try_into().unwrap());
            let l = last.entry(key).or_insert((run, era));
            if run >= l.0 {
                *l = (run, era);
            }
        }
        Creations { runs: self.runs(), last }
    }

    /// `created.bin` again with only the last of each of `keys` (the pipelines kept).
    pub fn compact_creations(&self, c: &Creations, keys: &HashSet<u64>) -> std::io::Result<()> {
        let mut file = Vec::new();
        for (key, (run, era)) in &c.last {
            if keys.contains(key) {
                file.extend_from_slice(&created_entry(*key, *era, *run));
            }
        }
        let path = self.dir.join("created.bin");
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, file)?;
        std::fs::rename(&tmp, &path)
    }

    /// `xcl warm-up --full`'s progress in translation era `era`: what a pass of this era went
    /// through already; a record of another era is started again.
    pub fn progress(&self, era: u64) -> std::io::Result<Progress> {
        self.progress_in("full-progress.bin", era)
    }

    /// `xcl warm-up --blocking`'s record of the keys it made ready in translation era `era` with
    /// adapter and driver `driver`; true: a record of another driver was dropped (the driver
    /// refuses the blobs of another). A record kept before the driver was noted is taken as this
    /// driver's.
    pub(super) fn blocking_progress(&self, era: u64, driver: &str) -> std::io::Result<(Progress, bool)> {
        std::fs::create_dir_all(&self.dir)?;
        let noted = self.dir.join("blocking-progress.driver");
        let changed = std::fs::read_to_string(&noted).is_ok_and(|kept| kept != driver);
        if changed {
            self.blocking_progress_clear();
        }
        std::fs::write(&noted, driver)?;
        Ok((self.progress_in("blocking-progress.bin", era)?, changed))
    }

    /// `xcl warm-up --blocking --fresh`: the next pass replays every key.
    pub(super) fn blocking_progress_clear(&self) {
        let _ = std::fs::remove_file(self.dir.join("blocking-progress.bin"));
    }

    fn progress_in(&self, name: &str, era: u64) -> std::io::Result<Progress> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(name);
        let mut bytes = std::fs::read(&path).unwrap_or_default();
        if bytes.get(..8) != Some(&era.to_le_bytes()[..]) {
            bytes = era.to_le_bytes().to_vec();
        }
        bytes.truncate(8 + (bytes.len() - 8) / 8 * 8);
        std::fs::write(&path, &bytes)?;
        let done = bytes[8..].chunks_exact(8).map(|k| u64::from_le_bytes(k.try_into().unwrap())).collect();
        let file = std::fs::OpenOptions::new().append(true).open(&path)?;
        Ok(Progress { file: Mutex::new(file), done })
    }

    /// A `--full` pass went through everything: the next starts from the beginning.
    pub fn progress_complete(&self) {
        let _ = std::fs::remove_file(self.dir.join("full-progress.bin"));
    }
}

fn created_entry(key: u64, era: u64, run: u32) -> [u8; CREATED] {
    let mut e = [0u8; CREATED];
    e[..8].copy_from_slice(&key.to_le_bytes());
    e[8..16].copy_from_slice(&era.to_le_bytes());
    e[16..].copy_from_slice(&run.to_le_bytes());
    e
}

/// The folder of the list: beside the DXIL cache, not in a translator's folder of it.
pub fn dir() -> PathBuf {
    crate::gpu::recomp::dxc::cache_dir().join("pipelines")
}

pub(super) fn blocking_path() -> PathBuf {
    dir().join("blocking-v2.txt")
}

pub(super) const BLOCKING_WAIT_MS: usize = 250;

/// The keys of a blocking list, each once: those with a measured wait of `BLOCKING_WAIT_MS` or
/// more, longest first, then those a target list (`min_run` None) names without a wait (the key
/// alone) in the order listed. The game's own list always has a wait: a line without one is torn.
pub(super) fn blocking_order(text: &str, min_run: Option<u32>) -> Vec<u64> {
    let mut waits = HashMap::<u64, usize>::new();
    let mut unmeasured = Vec::new();
    let mut listed = HashSet::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        if let Some(min_run) = min_run {
            if !fields.next().and_then(|s| s.parse::<u32>().ok()).is_some_and(|run| run >= min_run) {
                continue;
            }
        }
        let Some(key) = fields.next().and_then(|v| u64::from_str_radix(v, 16).ok()).filter(|&key| key != 0) else { continue };
        let ms = fields.next();
        if fields.next().is_some() {
            continue;
        }
        match ms.map(str::parse::<usize>) {
            None => {
                if min_run.is_none() && listed.insert(key) {
                    unmeasured.push(key);
                }
            }
            Some(Ok(ms)) if ms >= BLOCKING_WAIT_MS => {
                let old = waits.entry(key).or_default();
                *old = (*old).max(ms);
            }
            Some(_) => {}
        }
    }
    let mut ordered: Vec<_> = waits.iter().map(|(&key, &ms)| (key, ms)).collect();
    ordered.sort_unstable_by_key(|&(key, ms)| (std::cmp::Reverse(ms), key));
    ordered.into_iter().map(|(key, _)| key).chain(unmeasured.into_iter().filter(|key| !waits.contains_key(key))).collect()
}

enum ToWrite {
    /// a pipeline to keep, encoded
    Keep(Record, Vec<u8>),
    /// the key of a pipeline this run made for the first time
    Made(u64),
    /// the key of a pipeline this run bound for the first time
    Bound(u64),
    Blocked(u64, usize),
}

struct Global {
    store: Store,
    tx: Mutex<mpsc::Sender<ToWrite>>,
    /// the keys this run made
    made: Mutex<HashSet<u64>>,
    /// the keys this run bound
    bound: Mutex<HashSet<u64>>,
}

/// This process's list and the thread that writes to it (None: `XCL_PSO_MANIFEST=0`, or a test).
fn global() -> Option<&'static Global> {
    static G: OnceLock<Option<Global>> = OnceLock::new();
    G.get_or_init(|| {
        if cfg!(test) || std::env::var("XCL_PSO_MANIFEST").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        let store = Store::open(dir());
        let counts = !std::env::var("XCL_PSO_RUN_COUNTS").is_ok_and(|v| v.trim() == "0");
        let run = store.begin_run(counts).unwrap_or(0);
        let era = crate::gpu::recomp::pso::translation_era();
        crate::xlog_info!(
            "D3D12.X: {} pipelines earlier runs made are kept for the warm-up; new ones are added (run {run}{}, translation {era:016x})",
            store.len(),
            if counts { "" } else { ", not counted" }
        );
        let (tx, rx) = mpsc::channel::<ToWrite>();
        let _ = std::thread::Builder::new().name("xcl-pso-manifest".into()).spawn(move || {
            let Some(g) = global() else { return };
            let (mut failed, mut created, mut bound, mut blocking) = (false, None, None, None);
            let mut slowest = HashMap::new();
            for w in rx {
                let r = match w {
                    ToWrite::Keep(r, encoded) => g.store.write(&r, &encoded),
                    ToWrite::Made(key) => g.store.made(&mut created, key, era, run),
                    ToWrite::Bound(key) => g.store.bound(&mut bound, key, era, run),
                    ToWrite::Blocked(key, ms) => {
                        let previous = slowest.entry(key).or_insert(0);
                        if ms <= *previous {
                            continue;
                        }
                        let result = (|| {
                            if blocking.is_none() {
                                blocking = Some(std::fs::OpenOptions::new().create(true).append(true).open(blocking_path())?);
                            }
                            writeln!(blocking.as_mut().unwrap(), "{run} {key:016x} {ms}")
                        })();
                        if result.is_ok() {
                            *previous = ms;
                        }
                        result
                    }
                };
                if let Err(e) = r {
                    if !std::mem::replace(&mut failed, true) {
                        crate::xlog_warn!("D3D12.X: a pipeline could not be kept for the warm-up ({e})");
                    }
                }
            }
        });
        Some(Global { store, tx: Mutex::new(tx), made: Mutex::new(HashSet::new()), bound: Mutex::new(HashSet::new()) })
    })
    .as_ref()
}

/// The title made pipeline `r`: kept for the warm-up unless it is already, and noted as made by
/// this run (written on a thread of its own; the title's thread only hashes it).
pub fn record(r: Record) -> u64 {
    let Some(g) = global() else { return 0 };
    let (key, encoded) = g.store.note(&r);
    let made = g.made.lock().unwrap().insert(key);
    let tx = g.tx.lock().unwrap();
    if let Some(encoded) = encoded {
        let _ = tx.send(ToWrite::Keep(r, encoded));
    }
    if made {
        let _ = tx.send(ToWrite::Made(key));
    }
    key
}

/// The title bound pipeline `key` (a `record` key): the next runs and warm-ups build it ahead
/// of the ones no run bound (`pso_sched::Backlog::drawn_first`, `blocking_targets`).
pub(super) fn note_bound(key: u64) {
    let Some(g) = global() else { return };
    if key != 0 && g.bound.lock().unwrap().insert(key) {
        let _ = g.tx.lock().unwrap().send(ToWrite::Bound(key));
    }
}

/// Where pipeline `key` comes in the order the runs of the last `STALE_AFTER_RUNS` first bound
/// their pipelines (`Store::bound_in_order`; 0: the first), for the order the game builds the ones
/// earlier runs used in (`pso_async::first_use`).
pub(super) fn bound_order() -> &'static HashMap<u64, usize> {
    static ORDER: OnceLock<HashMap<u64, usize>> = OnceLock::new();
    ORDER.get_or_init(|| {
        if cfg!(test) {
            return HashMap::new();
        }
        let runs = std::num::NonZeroU32::new(STALE_AFTER_RUNS + 1).unwrap();
        Store::bound_in_order(&dir(), runs).into_iter().enumerate().map(|(rank, key)| (key, rank)).collect()
    })
}

/// An earlier run of the last `STALE_AFTER_RUNS` bound pipeline `key`.
pub(super) fn bound_earlier(key: u64) -> bool {
    static EARLIER: OnceLock<HashSet<u64>> = OnceLock::new();
    key != 0
        && EARLIER
            .get_or_init(|| {
                if cfg!(test) {
                    return HashSet::new();
                }
                let dir = dir();
                // (the runs before this one: `global` counted this one already when it was opened)
                Store::recently_bound(&dir, std::num::NonZeroU32::new(STALE_AFTER_RUNS + 1).unwrap())
            })
            .contains(&key)
}

pub(super) fn note_blocked(key: u64, ms: usize) {
    if key != 0 && ms >= BLOCKING_WAIT_MS {
        if let Some(g) = global() {
            let _ = g.tx.lock().unwrap().send(ToWrite::Blocked(key, ms));
        }
    }
}

// Adapted from CGBJ (a VIctory fork) at f4b48ad766ddbec1a1867022cd42f7645b0dd55e,
// crates/victory-gpu/src/pipeline/library.rs (`SLOW_FILE`, `SLOW_MS`, `slow`, `note_time`):
// keyed by our manifest key, and read by the warm-up's order as well.
// Copyright (C) the VIctory/CGBJ contributors. GPL-3.0-or-later.

/// A pipeline whose build took this long is a slow one (`slow_ms`).
pub const SLOW_MS: u64 = 1000;

/// How long each pipeline that was slow to build took at most, in ms: `slow-pipelines.txt` in
/// the list's folder, a key and its time a line (a key may be there more than once).
fn slow_times() -> &'static Mutex<(HashMap<u64, u64>, Option<std::fs::File>)> {
    static S: OnceLock<Mutex<(HashMap<u64, u64>, Option<std::fs::File>)>> = OnceLock::new();
    S.get_or_init(|| {
        if cfg!(test) {
            return Mutex::new((HashMap::new(), None));
        }
        let path = dir().join("slow-pipelines.txt");
        let times = parse_slow(&std::fs::read_to_string(&path).unwrap_or_default());
        let file = std::fs::create_dir_all(dir()).and_then(|_| std::fs::OpenOptions::new().create(true).append(true).open(&path)).ok();
        Mutex::new((times, file))
    })
}

fn parse_slow(text: &str) -> HashMap<u64, u64> {
    let mut times = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(key), Some(ms), None) = (fields.next(), fields.next(), fields.next()) else { continue };
        let (Ok(key), Ok(ms)) = (u64::from_str_radix(key, 16), ms.parse::<u64>()) else { continue };
        if key != 0 && ms >= SLOW_MS {
            let t = times.entry(key).or_insert(0);
            *t = (*t).max(ms);
        }
    }
    times
}

/// The longest an earlier build of pipeline `key` took, if that was `SLOW_MS` or more.
pub fn slow_ms(key: u64) -> Option<u64> {
    if key == 0 { None } else { slow_times().lock().unwrap().0.get(&key).copied() }
}

pub fn slow(key: u64) -> bool {
    slow_ms(key).is_some()
}

/// Building pipeline `key` took `ms`, waits for memory left out: kept when slow and the
/// longest yet.
pub fn note_build_time(key: u64, ms: u64) {
    if key == 0 || ms < SLOW_MS {
        return;
    }
    let mut s = slow_times().lock().unwrap();
    let known = s.0.entry(key).or_insert(0);
    if ms > *known {
        *known = ms;
        if let Some(f) = s.1.as_mut() {
            let _ = writeln!(f, "{key:016x} {ms}");
        }
    }
}

/// What `xcl warm-up --blocking` builds without `--targets` (`blocking_targets`).
pub(super) struct Targets {
    pub keys: Vec<u64>,
    /// of `keys`, those with a measured wait (first)
    pub waited: usize,
    /// of `keys`, those the last runs bound, after the waited ones: with them the first tier
    pub drawn: usize,
    /// of `keys`, the other pipelines of the shaders runs used, after the first tier
    pub variants: usize,
    /// kept pipelines no run drew with: left for their first bind in a run, or `warm-up --full`
    pub left: usize,
}

/// The pipelines kept in `dir` that runs drew with, as `xcl warm-up --blocking` builds them. The
/// first tier: those whose bind a run of `min_run` on waited for, the longest wait first
/// (`blocking_order`), then the exact ones the last `bound_runs` runs bound (`note_bound`), in
/// the order of their first bind. Then, with `variants`, the other pipelines of the shaders a run
/// used or skipped a draw for (`used`: shader keys, `shaders_key`), the slowest measured first.
/// The rest are counted, not built. The last six runs (play-109 to play-114) bound 2,133
/// pipelines, while the shaders they used have 77,690: a pass of all of them first spent its hour
/// on pipelines no run had bound.
pub(super) fn blocking_targets(dir: &Path, text: &str, min_run: u32, used: &HashSet<u64>, bound_runs: std::num::NonZeroU32, variants: bool) -> Targets {
    let mut keys = blocking_order(text, Some(min_run));
    let waited = keys.len();
    let mut seen: HashSet<u64> = keys.iter().copied().collect();
    let file = std::fs::read(dir.join("pipelines.bin")).unwrap_or_default();
    let (mut kept, mut by_shader) = (HashSet::new(), Vec::new());
    for e in entries(&file) {
        let key = key_of(e);
        if kept.insert(key) && encoded_shaders_key(e).is_some_and(|k| used.contains(&k)) {
            by_shader.push(key);
        }
    }
    drop(file);
    let bound: Vec<u64> = Store::bound_in_order(dir, bound_runs).into_iter().filter(|key| kept.contains(key) && seen.insert(*key)).collect();
    let drawn = bound.len();
    keys.extend(bound);
    let mut others: Vec<u64> = by_shader.into_iter().filter(|key| seen.insert(*key)).collect();
    others.sort_by_key(|&key| std::cmp::Reverse(slow_ms(key).unwrap_or(0)));
    let n = if variants { others.len() } else { 0 };
    keys.extend(others.into_iter().take(n));
    let left = kept.len() - keys.iter().filter(|key| kept.contains(key)).count();
    Targets { keys, waited, drawn, variants: n, left }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build times: the longest of each key, from `SLOW_MS` on; other lines are passed over.
    #[test]
    fn slow_times_keep_the_longest_from_the_threshold() {
        let t = parse_slow(&format!("a 1200\na 3000\na 2000\nb {}\n0 5000\nc x\nd 1500 extra\nzz 9000\n", SLOW_MS - 1));
        assert_eq!(t.len(), 1);
        assert_eq!(t[&0xa], 3000);
    }

    /// The warm-up's own list: waited-for ones first, then what runs bound or used, and the
    /// ones no run drew with counted, not built.
    #[test]
    fn blocking_targets_are_waits_then_what_runs_drew_with() {
        let dir = temp_dir("targets");
        let waited = sample(Kind::Compute, b"waited", b"W");
        let bound = sample(Kind::Graphics, b"bound", b"B");
        let used = sample(Kind::Graphics, b"used", b"U");
        let neither = sample(Kind::Graphics, b"neither", b"N");
        let store = Store::open(dir.clone());
        for r in [&waited, &bound, &used, &neither] {
            store.add(r).unwrap();
        }
        let run = store.begin_run(true).unwrap();
        store.bound(&mut None, bound.key(), 0, run).unwrap();
        let used_key = shaders_key(used.shaders.iter().map(|(_, s)| crate::gpu::shader::fnv64(s)));
        let three = std::num::NonZeroU32::new(3).unwrap();
        let t = blocking_targets(&dir, &format!("{run} {:x} 900\n", waited.key()), run, &[used_key].into_iter().collect(), three, true);
        assert_eq!(t.keys, [waited.key(), bound.key(), used.key()]);
        assert_eq!((t.waited, t.drawn, t.variants, t.left), (1, 1, 1, 1));
        let t = blocking_targets(&dir, &format!("{run} {:x} 900\n", waited.key()), run, &[used_key].into_iter().collect(), three, false);
        assert_eq!(t.keys, [waited.key(), bound.key()], "XCL_WARM_VARIANTS=0: the first tier alone");
        assert_eq!((t.waited, t.drawn, t.variants, t.left), (1, 1, 0, 2));
        // (the bound ones in the order of their first bind, each once)
        store.bound(&mut None, neither.key(), 0, run).unwrap();
        store.bound(&mut None, bound.key(), 0, run).unwrap();
        assert_eq!(Store::bound_in_order(&dir, three), [bound.key(), neither.key()]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xcl-pso-manifest-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample(kind: Kind, shader: &[u8], name: &[u8]) -> Record {
        let mut e = [0u8; ELEMENT];
        e[8] = 3;
        Record {
            kind,
            desc: (0u8..64).collect(),
            shaders: vec![(8, Arc::new(shader.to_vec())), (24, Arc::new(b"shared shader".to_vec()))],
            layout: Some((40, vec![(name.to_vec(), e)])),
            canonical: false,
        }
    }

    #[test]
    fn the_canonical_form_is_part_of_the_key_and_kept() {
        let dir = temp_dir("canonical");
        let raw = sample(Kind::MeshStream, b"mesh shader", b"POSITION");
        let canonical = Record { canonical: true, ..raw.clone() };
        assert_ne!(raw.key(), canonical.key());
        let store = Store::open(dir.clone());
        for r in [&raw, &canonical] {
            assert!(store.add(r).unwrap());
        }
        let loaded = Store::open(dir.clone()).load();
        assert_eq!(loaded.records.len(), 2);
        assert!(loaded.records.contains(&raw) && loaded.records.contains(&canonical));
        assert!(loaded.records.iter().all(|r| [raw.key(), canonical.key()].contains(&r.key())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_index_gives_each_kept_record_as_loading_it_does() {
        let dir = temp_dir("index");
        let a = sample(Kind::Graphics, b"vertex shader", b"POSITION");
        let b = sample(Kind::Compute, b"compute shader", b"C");
        let store = Store::open(dir.clone());
        for r in [&a, &b] {
            store.add(r).unwrap();
        }
        let index = Store::index(&dir);
        assert_eq!(index.get(a.key()), Some(a.clone()));
        assert_eq!(index.get(b.key()), Some(b.clone()));
        assert_eq!(Store::load_keys(&dir, &[a.key(), b.key()].into_iter().collect()).records.len(), 2);
        assert_eq!(index.get(a.key() ^ 2), None);
        let subset = dir.join("subset.bin");
        assert_eq!(write_subset(&dir, &[b.key()].into_iter().collect(), &subset).unwrap(), 1);
        let small = Store::index_at(&dir, &subset);
        assert_eq!((small.get(a.key()), small.get(b.key())), (None, Some(b.clone())), "a warm-up's own list holds its keys alone");
        std::fs::remove_file(shader_path(&dir, crate::gpu::shader::fnv64(b"compute shader"), b"compute shader".len() as u32)).unwrap();
        assert_eq!(index.get(b.key()), None, "a shader it names is missing");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn blocking_replay_selects_exact_records_not_other_descriptors_of_the_same_shader() {
        let dir = temp_dir("blocking");
        let a = sample(Kind::Graphics, b"vertex shader", b"POSITION");
        let b = sample(Kind::Graphics, b"vertex shader", b"TEXCOORD");
        let c = sample(Kind::MeshStream, b"mesh shader", b"MESH");
        let store = Store::open(dir.clone());
        for record in [&a, &b, &c] {
            store.add(record).unwrap();
        }
        let order = blocking_order(&format!("{:x} 400\n{:x} 700\n{:x} 900\n{:x} 249\ninvalid\n{:x} 1000 extra\n", a.key(), c.key(), a.key(), b.key(), b.key()), None);
        assert_eq!(order, [a.key(), c.key()]);
        let loaded = Store::load_keys(&dir, &order.into_iter().collect());
        assert_eq!(loaded.records.len(), 2);
        assert!(loaded.records.contains(&a) && loaded.records.contains(&c));
        assert_eq!(blocking_order("4 1 9999\n5 2 500\n6 3 800\n", Some(5)), [3, 2]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A target list may name keys without a wait: they follow the measured ones, in the order
    /// listed, each once, and a key also listed with a wait goes with the measured ones.
    #[test]
    fn blocking_order_takes_keys_without_a_wait_after_the_measured_ones() {
        assert_eq!(blocking_order("c\n5 300\na\nc\n7 900\n5\nb\n0\nzz\n9 100\n", None), [7, 5, 0xc, 0xa, 0xb]);
        assert_eq!(blocking_order("4 1\n6 3\n6 2 500\n", Some(5)), [2], "the game's list: a line without a wait is torn");
    }

    /// Pipelines are kept once each, with their shaders, and read back as they were made.
    #[test]
    fn the_keys_read_an_entry_at_a_time_are_the_whole_files() {
        let dir = temp_dir("keys-in");
        let path = dir.join("pipelines.bin");
        assert!(keys_in(&path).is_empty(), "no file");
        let mut file = Vec::new();
        for (shader, name) in [(&b"one"[..], &b"A"[..]), (b"two", b"B"), (b"one", b"A")] {
            let e = sample(Kind::Graphics, shader, name).encode();
            file.extend_from_slice(&(e.len() as u32).to_le_bytes());
            file.extend_from_slice(&e);
        }
        let whole = |file: &[u8]| entries(file).into_iter().map(key_of).collect::<HashSet<u64>>();
        // (a torn last entry, one whose length runs past the end, and a torn length)
        for tail in [&[][..], &[9, 0, 0, 0, 1, 2][..], &[0xff, 0xff, 0xff, 0xff, 1][..], &[7, 0][..]] {
            let mut torn = file.clone();
            torn.extend_from_slice(tail);
            std::fs::write(&path, &torn).unwrap();
            assert_eq!(keys_in(&path), whole(&torn));
            assert_eq!(keys_in(&path).len(), 2);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A key list spares reading the entries it covers, follows the list as it grows, and is
    /// not used once the list's bytes are not the ones it was made of.
    #[test]
    fn a_key_list_spares_reading_the_entries_it_covers() {
        let dir = temp_dir("key-list");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pipelines.bin");
        let entry = |shader: &[u8], name: &[u8]| {
            let e = sample(Kind::Graphics, shader, name).encode();
            let mut b = (e.len() as u32).to_le_bytes().to_vec();
            b.extend_from_slice(&e);
            b
        };
        let whole = |file: &[u8]| entries(file).into_iter().map(key_of).collect::<HashSet<u64>>();
        let mut file = [entry(b"one", b"A"), entry(b"two", b"B")].concat();
        std::fs::write(&path, &file).unwrap();
        assert_eq!(keys_in(&path), whole(&file));
        let list = |path: &Path| kept_keys(path, &std::fs::File::open(path).unwrap(), std::fs::metadata(path).unwrap().len());
        assert_eq!(list(&path), Some((whole(&file), file.len() as u64)), "a key list of the whole list");
        // (the list grows, a torn entry after it: the list covers the entries, not the tear)
        let grown = file.len() as u64;
        file.extend(entry(b"three", b"C"));
        let covered = file.len() as u64;
        file.extend_from_slice(&[9, 0, 0, 0, 1]);
        std::fs::write(&path, &file).unwrap();
        assert_eq!(keys_in(&path), whole(&file));
        assert_eq!(keys_in(&path).len(), 3);
        assert_eq!(list(&path).map(|l| l.1), Some(covered));
        // (its keys are taken as they are: a key only the list holds comes back)
        let mut fake = std::fs::read(keys_path(&path)).unwrap();
        fake[24..32].copy_from_slice(&4u64.to_le_bytes());
        fake.extend_from_slice(&0x1234_5678u64.to_le_bytes());
        std::fs::write(keys_path(&path), &fake).unwrap();
        assert!(keys_in(&path).contains(&0x1234_5678), "the keys come from the key list");
        // (the list rewritten with other bytes, or cut short: the key list is not used)
        let rewritten = [entry(b"four", b"D"), entry(b"five", b"E"), entry(b"six", b"F")].concat();
        std::fs::write(&path, &rewritten).unwrap();
        assert_eq!(keys_in(&path), whole(&rewritten));
        std::fs::write(&path, &file[..grown as usize]).unwrap();
        assert_eq!(keys_in(&path), whole(&file[..grown as usize]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pipelines_are_kept_once_and_read_back_whole() {
        let dir = temp_dir("roundtrip");
        let (a, b) = (sample(Kind::Graphics, b"vertex shader a", b"POSITION"), sample(Kind::VertexStream, b"vertex shader b", b"TEXCOORD"));
        let compute = Record { kind: Kind::Compute, desc: vec![7; 40], shaders: vec![(8, Arc::new(vec![1, 2, 3]))], layout: None, canonical: false };
        let store = Store::open(dir.clone());
        assert!(store.add(&a).unwrap());
        assert!(store.add(&b).unwrap());
        assert!(!store.add(&a.clone()).unwrap(), "kept already");
        assert!(store.add(&compute).unwrap());
        let again = Store::open(dir.clone());
        assert_eq!(again.len(), 3);
        assert!(!again.add(&b).unwrap(), "kept by an earlier run");
        let loaded = again.load();
        assert_eq!(loaded.records, vec![a.clone(), b, compute]);
        assert_eq!((loaded.broken, loaded.repeated), (0, 0));
        assert_eq!(std::fs::read_dir(dir.join("shaders")).unwrap().count(), 4, "a shader two pipelines name is kept once");
        assert_eq!(a.size(), b"vertex shader a".len() + b"shared shader".len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kind_and_key_filters_run_before_shader_reads() {
        let dir = temp_dir("kind-filter");
        let store = Store::open(dir.clone());
        let cold = sample(Kind::MeshStream, b"cold mesh", b"POSITION");
        let kept = sample(Kind::MeshStream, b"missing mesh", b"POSITION");
        let other = sample(Kind::Graphics, b"missing vertex", b"TEXCOORD");
        for r in [&cold, &kept, &other] {
            assert!(store.add(r).unwrap());
        }
        for bytes in [b"missing mesh".as_slice(), b"missing vertex".as_slice()] {
            std::fs::remove_file(shader_path(&dir, crate::gpu::shader::fnv64(bytes), bytes.len() as u32)).unwrap();
        }
        let mut inspected = Vec::new();
        let loaded = Store::load_kind(&dir, Kind::MeshStream, |key, shaders| {
            let r = [&cold, &kept].into_iter().find(|r| r.key() == key).unwrap();
            assert_eq!(shaders, Some(shaders_key(r.shaders.iter().map(|(_, s)| crate::gpu::shader::fnv64(s)))));
            inspected.push(key);
            key != kept.key()
        });
        assert_eq!(inspected, [cold.key(), kept.key()]);
        assert_eq!(loaded.records, [cold]);
        assert_eq!(loaded.broken, 0);
        assert_eq!(store.load().broken, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recent_creations_include_only_counted_runs_in_range() {
        let dir = temp_dir("recent-creations");
        let store = Store::open(dir.clone());
        let mut file = None;
        for key in [1, 2, 3] {
            let run = store.begin_run(true).unwrap();
            store.made(&mut file, key, 0x1234, run).unwrap();
        }
        drop(file);
        let (runs, keys) = Store::recently_created(&dir, std::num::NonZeroU32::new(2).unwrap()).unwrap();
        assert_eq!(runs, 2..=3);
        assert_eq!(keys, HashSet::from([2, 3]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A torn last entry, an entry kept twice and one whose shader is gone are left out, and a
    /// rewrite keeps only what loads.
    #[test]
    fn torn_repeated_and_broken_entries_are_left_out() {
        let dir = temp_dir("damage");
        let (a, b) = (sample(Kind::Graphics, b"shader a", b"A"), sample(Kind::Graphics, b"shader b", b"B"));
        let store = Store::open(dir.clone());
        store.add(&a).unwrap();
        store.add(&b).unwrap();
        let file = dir.join("pipelines.bin");
        let mut bytes = std::fs::read(&file).unwrap();
        let first = entries(&bytes)[0].len() + 4;
        let copy = bytes[..first].to_vec();
        bytes.extend_from_slice(&copy);
        bytes.extend_from_slice(&[9, 0, 0, 0, 1, 2]);
        std::fs::write(&file, &bytes).unwrap();
        std::fs::remove_file(shader_path(&dir, crate::gpu::shader::fnv64(b"shader b"), 8)).unwrap();
        let store = Store::open(dir.clone());
        let loaded = store.load();
        assert_eq!(loaded.records, vec![a.clone()]);
        assert_eq!((loaded.broken, loaded.repeated), (1, 1));
        store.rewrite(&loaded.records).unwrap();
        let loaded = Store::open(dir.clone()).load();
        assert_eq!((loaded.records, loaded.broken, loaded.repeated), (vec![a], 0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pipeline no counted run of the last `STALE_AFTER_RUNS` made is stale; a run not counted
    /// keeps what it made fresh without aging the rest, and the era of the last making is kept.
    #[test]
    fn a_pipeline_no_recent_run_made_is_stale() {
        let dir = temp_dir("stale");
        let store = Store::open(dir.clone());
        let (a, b, never) = (sample(Kind::Graphics, b"a", b"A").key(), sample(Kind::Graphics, b"b", b"B").key(), 7u64);
        let mut file = None;
        let run = store.begin_run(true).unwrap();
        assert_eq!(run, 1);
        store.made(&mut file, a, 10, run).unwrap();
        store.made(&mut file, b, 10, run).unwrap();
        let c = store.creations();
        assert!(!c.stale(a) && !c.stale(never), "noted before the runs began to count");
        for _ in 0..2 {
            let run = store.begin_run(true).unwrap();
            store.made(&mut file, b, 11, run).unwrap();
        }
        let c = store.creations();
        assert_eq!(c.runs, 3);
        assert!(!c.stale(a), "made in the oldest of the last three");
        assert!(c.stale(never), "never made in three runs");
        assert_eq!((c.era(a), c.era(b), c.era(never)), (Some(10), Some(11), None));
        let check = store.begin_run(false).unwrap();
        assert_eq!(check, 3, "a run not counted notes with the last counted");
        store.made(&mut file, b, 12, check).unwrap();
        let run = store.begin_run(true).unwrap();
        store.made(&mut file, b, 12, run).unwrap();
        let c = store.creations();
        assert_eq!(c.runs, 4);
        assert!(c.stale(a) && !c.stale(b));
        assert_eq!(c.era(b), Some(12));
        drop(file);
        std::fs::OpenOptions::new().append(true).open(dir.join("created.bin")).unwrap().write_all(&[1, 2, 3]).unwrap();
        store.compact_creations(&store.creations(), &[a].into_iter().collect()).unwrap();
        assert_eq!(std::fs::metadata(dir.join("created.bin")).unwrap().len(), CREATED as u64, "the last of each kept key, a torn entry gone");
        let c = store.creations();
        assert!(c.stale(a) && c.stale(b), "b is no longer kept");
        assert_eq!(c.era(a), Some(10));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `--full` pass stopped part way goes on where it left off in the same translation era, and
    /// starts again in another or once it went through everything.
    #[test]
    fn a_blocking_pass_starts_again_on_another_driver() {
        let dir = temp_dir("blocking-progress");
        let store = Store::open(dir.clone());
        store.progress_in("blocking-progress.bin", 5).unwrap().note(1);
        let (p, changed) = store.blocking_progress(5, "10de-2684-a").unwrap();
        assert!(!changed && p.done == [1].into_iter().collect(), "a record kept before the driver was noted is kept");
        p.note(2);
        drop(p);
        let (p, changed) = store.blocking_progress(5, "10de-2684-a").unwrap();
        assert!(!changed && p.done == [1, 2].into_iter().collect());
        drop(p);
        let (p, changed) = store.blocking_progress(5, "10de-2684-b").unwrap();
        assert!(changed && p.done.is_empty(), "another driver starts again");
        drop(p);
        assert!(!store.blocking_progress(5, "10de-2684-b").unwrap().1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_full_pass_goes_on_where_it_stopped_in_its_era() {
        let dir = temp_dir("progress");
        let store = Store::open(dir.clone());
        let p = store.progress(5).unwrap();
        assert!(p.done.is_empty());
        p.note(1);
        p.note(2);
        drop(p);
        std::fs::OpenOptions::new().append(true).open(dir.join("full-progress.bin")).unwrap().write_all(&[9, 9, 9]).unwrap();
        let p = store.progress(5).unwrap();
        assert_eq!(p.done, [1, 2].into_iter().collect());
        p.note(3);
        drop(p);
        assert_eq!(store.progress(5).unwrap().done, [1, 2, 3].into_iter().collect(), "a torn key is dropped, not misread");
        assert!(store.progress(6).unwrap().done.is_empty(), "another era starts again");
        assert!(store.progress(5).unwrap().done.is_empty(), "the record of era 5 was replaced");
        store.progress(5).unwrap().note(4);
        store.progress_complete();
        assert!(store.progress(5).unwrap().done.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
