//! CGBJ's own translation cache: a small map in memory in front of files under
//! `{dxc::cache_dir()}/cgbj/{identity}/`. Refusals are kept too, with their reason, so a warm
//! run does not translate a refused program again.

use crate::gpu::recomp::pso::Compiled;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"XCLCGBJ1";
const GOOD: u8 = 1;
const REFUSED: u8 = 2;
/// The memory map is emptied when its DXIL would pass this.
const MEMORY_BUDGET: usize = 64 << 20;

/// What a translation came to.
#[derive(Clone)]
pub enum Entry {
    Good(Compiled),
    /// The short reason (no code offsets), as counted.
    Refused(String),
}

/// FNV-1a over `parts`, each closed by a zero byte. Never 0.
pub fn identity(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &byte in part.iter().chain(&[0]) {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h | 1
}

/// A second hash of the bytes, independent of `fnv64`'s.
pub fn hash2(b: &[u8]) -> u64 {
    b.iter().fold(0x9E37_79B9_7F4A_7C15u64, |h, &x| (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29))
}

/// Which program under which identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub ident: u64,
    pub fnv: u64,
    pub hash2: u64,
    pub len: usize,
    pub sampler_base: u32,
}

impl Key {
    pub fn new(ident: u64, fnv: u64, blob: &[u8], sampler_base: u32) -> Key {
        Key { ident, fnv, hash2: hash2(blob), len: blob.len(), sampler_base }
    }

    pub fn file(&self, dir: &Path) -> PathBuf {
        dir.join("cgbj").join(format!("{:016x}", self.ident)).join(format!("{:016x}_{}.bin", self.fnv, self.sampler_base))
    }

    fn header(&self) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        for v in [self.ident, self.len as u64, self.hash2] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h
    }
}

pub fn encode(key: &Key, e: &Entry) -> Vec<u8> {
    let mut b = key.header();
    match e {
        Entry::Good(c) => {
            b.push(GOOD);
            b.extend_from_slice(&(c.samplers.len() as u32).to_le_bytes());
            for (s, cmp) in &c.samplers {
                s.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
                b.push(*cmp as u8);
            }
            b.extend_from_slice(&(c.dxil.len() as u32).to_le_bytes());
            b.extend_from_slice(&c.dxil);
        }
        Entry::Refused(reason) => {
            b.push(REFUSED);
            b.extend_from_slice(reason.as_bytes());
        }
    }
    b
}

/// The entry in a file's bytes, if the file is whole and made for `key`.
pub fn decode(key: &Key, b: &[u8]) -> Option<Entry> {
    let (&tag, mut r) = b.strip_prefix(key.header().as_slice())?.split_first()?;
    if tag == REFUSED {
        return Some(Entry::Refused(String::from_utf8(r.to_vec()).ok()?));
    }
    (tag == GOOD).then_some(())?;
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
    (r.len() == len && len > 0).then(|| Entry::Good(Compiled { dxil: r.to_vec(), samplers }))
}

pub fn load(dir: &Path, key: &Key) -> Option<Entry> {
    decode(key, &std::fs::read(key.file(dir)).ok()?)
}

/// Written whole under a temp name, then renamed, as pso.rs does: a reader never sees half a file.
pub fn store(dir: &Path, key: &Key, e: &Entry) {
    let path = key.file(dir);
    let Some(parent) = path.parent() else { return };
    let tmp = parent.join(format!("{}.{}.{:?}.tmp", path.file_name().unwrap_or_default().to_string_lossy(), std::process::id(), std::thread::current().id()));
    if std::fs::create_dir_all(parent).is_ok() && std::fs::write(&tmp, encode(key, e)).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

static MEMORY: Mutex<Option<(HashMap<Key, Entry>, usize)>> = Mutex::new(None);

pub fn remembered(key: &Key) -> Option<Entry> {
    MEMORY.lock().ok()?.as_ref()?.0.get(key).cloned()
}

pub fn remember(key: Key, e: &Entry) {
    let Ok(mut m) = MEMORY.lock() else { return };
    let (map, bytes) = m.get_or_insert_with(Default::default);
    let size = match e {
        Entry::Good(c) => c.dxil.len(),
        Entry::Refused(r) => r.len(),
    };
    if *bytes + size > MEMORY_BUDGET {
        map.clear();
        *bytes = 0;
    }
    if map.insert(key, e.clone()).is_none() {
        *bytes += size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of an entry, to compare (Compiled has no PartialEq).
    fn parts(e: Option<Entry>) -> Option<(u8, Vec<u8>, Vec<([u32; 4], bool)>)> {
        e.map(|e| match e {
            Entry::Good(c) => (GOOD, c.dxil, c.samplers),
            Entry::Refused(r) => (REFUSED, r.into_bytes(), vec![]),
        })
    }

    fn key() -> Key {
        Key::new(identity(&[b"a", b"b"]), 7, b"blob", 0)
    }

    #[test]
    fn round_trips() {
        let good = Entry::Good(Compiled { dxil: vec![1, 2, 3], samplers: vec![([1, 2, 3, 4], true), ([5, 6, 7, 8], false)] });
        let refused = Entry::Refused("ray tests under the XCL binding".into());
        for e in [good, refused] {
            assert_eq!(parts(decode(&key(), &encode(&key(), &e))), parts(Some(e)));
        }
    }

    #[test]
    fn another_program_or_build_or_a_cut_file_is_not_read() {
        let e = Entry::Good(Compiled { dxil: vec![9; 10], samplers: vec![] });
        let b = encode(&key(), &e);
        assert!(decode(&Key::new(key().ident, 7, b"bloc", 0), &b).is_none());
        assert!(decode(&Key { ident: key().ident ^ 2, ..key() }, &b).is_none());
        assert!(decode(&key(), &b[..b.len() - 1]).is_none());
        assert!(decode(&key(), &b).is_some());
        let empty = encode(&key(), &Entry::Good(Compiled { dxil: vec![], samplers: vec![] }));
        assert!(decode(&key(), &empty).is_none());
    }

    #[test]
    fn identity_tells_parts_apart() {
        assert_ne!(identity(&[b"ab", b"c"]), identity(&[b"a", b"bc"]));
        assert_ne!(identity(&[]), 0);
    }

    #[test]
    fn file_names() {
        let k = Key { ident: 0x10, fnv: 0xabc, hash2: 0, len: 0, sampler_base: 3 };
        let p = k.file(Path::new("c"));
        assert!(p.ends_with(Path::new("cgbj").join("0000000000000010").join("0000000000000abc_3.bin")));
    }
}
