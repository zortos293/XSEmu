use super::{HEAPS, SLOT};
use crate::hle::{
    d3d12x::{external_buffers, page_mappings, title_textures},
    kernelx,
};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};

fn config() -> Option<(u32, u32)> {
    static CFG: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *CFG.get_or_init(|| {
        if !super::material_gate_views::prepass::records::producer() {
            return None;
        }
        let setting = std::env::var("XCL_MESH_VSHARP").ok()?;
        let (offset, bound) = setting.split_once(',')?;
        let (offset, bound) = (offset.parse().ok()?, bound.parse().ok()?);
        (offset % 4 == 0 && (1..=4096).contains(&bound)).then_some((offset, bound))
    })
}

pub(crate) fn enabled() -> bool {
    config().is_some()
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Writer {
    Other,
    // Written by the vsharp trace; its report does not read these fields yet.
    #[allow(dead_code)]
    Api {
        kind: &'static str,
        input: u64,
        first: u64,
        count: u32,
        stride: u32,
        raw: bool,
    },
    Copy(usize),
}

#[derive(Clone, Copy, Debug)]
// Written by the vsharp trace; its report does not read these fields yet.
#[allow(dead_code)]
struct Event {
    first_slot: usize,
    first_serial: u64,
    source: Writer,
    hops: u32,
    last: Writer,
    serial: u64,
}

#[derive(Default)]
struct State {
    live: HashMap<usize, ([u32; 4], Event)>,
    last_api: HashMap<[u32; 4], Event>,
    serial: u64,
    dropped: u64,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(Default::default);

pub(crate) fn written(at: usize, w: &[u32], writer: Writer) {
    let Some((_, bound)) = config() else { return };
    let mut s = STATE.lock().unwrap();
    let Some(words) = w
        .get(..4)
        .and_then(|w| <[u32; 4]>::try_from(w).ok())
        .filter(|w| w[2] == bound && w[3] >> 28 < 8 && w[1] >> 16 & 0x3fff == 0)
    else {
        s.live.remove(&at);
        return;
    };
    s.serial += 1;
    let serial = s.serial;
    let parent = match writer {
        Writer::Copy(src) => s.live.get(&src).filter(|v| v.0 == words).map(|v| v.1),
        _ => None,
    };
    let event = match parent {
        Some(p) => Event {
            hops: p.hops.saturating_add(1),
            last: writer,
            serial,
            ..p
        },
        None => Event {
            first_slot: at,
            first_serial: serial,
            source: writer,
            hops: 0,
            last: writer,
            serial,
        },
    };
    if s.live.len() < 32768 || s.live.contains_key(&at) {
        s.live.insert(at, (words, event));
    } else {
        s.dropped += 1;
    }
    if matches!(writer, Writer::Api { .. }) {
        if s.last_api.len() < 16384 || s.last_api.contains_key(&words) {
            s.last_api.insert(words, event);
        } else {
            s.dropped += 1;
        }
    }
}

pub(crate) fn forget(range: std::ops::Range<usize>) {
    if enabled() {
        STATE
            .lock()
            .unwrap()
            .live
            .retain(|at, _| !range.contains(at));
    }
}

fn heap_identity(at: usize) -> Option<(usize, u64, usize, usize, bool)> {
    let heaps = HEAPS.lock().unwrap();
    heaps.iter().find_map(|h| {
        let off = at.checked_sub(h.shadow)?;
        (!h.sampler && off % SLOT as usize == 0 && off / (SLOT as usize) < h.n as usize)
            .then_some((h.obj, h.stamp, h.shadow, off / SLOT as usize, h.visible))
    })
}

fn ownership(at: usize, words: [u32; 4]) -> String {
    let s = STATE.lock().unwrap();
    let exact = s.live.get(&at).filter(|w| w.0 == words);
    let equivalent = s.last_api.get(&words);
    format!(
        "last_exact_slot_hle_write={exact:x?} same_words_last_api_any_slot_history_only={equivalent:x?} dropped={} serial={}",
        s.dropped, s.serial
    )
}

fn region(at: usize, n: usize) -> Option<(usize, usize, u32, u32)> {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut m = MEMORY_BASIC_INFORMATION::default();
    if unsafe { VirtualQuery(at as *const _, &mut m, std::mem::size_of_val(&m)) } == 0
        || at.checked_add(n)? > (m.BaseAddress as usize).checked_add(m.RegionSize)?
    {
        return None;
    }
    Some((
        m.AllocationBase as usize,
        m.BaseAddress as usize,
        m.State,
        m.Protect,
    ))
}

fn range_facts(at: usize, n: usize, counterpart: bool) -> String {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    };
    let end = at + n - 1;
    let mbi = region(at, n);
    let page = kernelx::backing(at);
    let last = if at >> 16 == end >> 16 {
        page
    } else {
        kernelx::backing(end)
    };
    let view = page_mappings::alias_facts(at, page.and_then(|p| (p.page != 0).then_some(p.page)));
    let owner = page
        .and_then(|p| (p.page != 0).then(|| kernelx::page_data(p.page)))
        .flatten();
    let cpu_bytes = if counterpart
        && mbi.is_some_and(|(_, _, s, p)| {
            s == MEM_COMMIT && matches!(p, PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY)
        }) {
        crate::memmap::title_bytes(at, n).map(|b| {
            (
                b.iter().filter(|&&x| x != 0).count(),
                b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
                    (h ^ x as u64).wrapping_mul(0x100000001b3)
                }),
            )
        })
    } else {
        None
    };
    format!(
        "at={at:#x} n={n} os_region={mbi:x?} physical_first_last={page:x?}/{last:x?} page_owner={owner:x?} queue_alias_first={:x?}/{:x?} external_full={:x?} placed_textures={:x?} cpu_snapshot_nonzero_bytes_fnv={cpu_bytes:x?}",
        view.here,
        view.aliases.iter().take(8).collect::<Vec<_>>(),
        external_buffers::view_owners(at, n),
        title_textures::placed_over(at, n)
    )
}

fn facts(w: [u32; 4]) -> Option<String> {
    let (_, bound) = config()?;
    if w[2] != bound || w[1] >> 16 & 0x3fff != 0 || w[3] >> 28 >= 8 {
        return None;
    }
    let at = ((w[1] as usize & 65535) << 32) | w[0] as usize;
    let n = bound as usize;
    let last = at.checked_add(n - 1)?;
    let gens = || (kernelx::gpu_view_map_gen(), external_buffers::generation());
    let before = gens();
    let cpu = kernelx::mirrors::source(at).filter(|c| {
        c.checked_add(n - 1)
            .is_some_and(|end| kernelx::mirrors::source(last) == Some(end))
    });
    let gpu = range_facts(at, n, false);
    let cpu = cpu.map(|a| range_facts(a, n, true));
    Some(format!(
        "gens_before_after={before:?}/{:?} guest=[{gpu}] paired_cpu={cpu:?}",
        gens()
    ))
}

pub(crate) struct Snapshot {
    reg: u32,
    off: u32,
    slot: u32,
    words: Option<[u32; 4]>,
    host: String,
    backing: Option<String>,
}

pub(crate) fn capture(io: &[u32]) -> Option<Snapshot> {
    let (off, _) = config()?;
    let reg = *io.get(crate::gpu::recomp::KERNEL_SGPRS + 9)?;
    let slot = reg.checked_add(off)?;
    let u = crate::gpu::uma::get()?;
    let at = u.base().checked_add(slot as usize)?;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    };
    let heap = heap_identity(at);
    let readable = u.in_committed_window(at, 16)
        || u.window_of(at) == Some(0)
            && region(at, 16).is_some_and(|(_, _, s, p)| {
                s == MEM_COMMIT && matches!(p, PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY)
            });
    let words = readable
        .then(|| crate::memmap::title_words(at, 4))
        .flatten()
        .and_then(|v| v.try_into().ok());
    let ledger = words.map(|w| ownership(at, w));
    let heap_end = heap_identity(at);
    let host = format!(
        "heap_obj_stamp_base_slot_visible_before_after={heap:x?}/{heap_end:x?} ledger={ledger:?}"
    );
    let backing = words.and_then(facts);
    Some(Snapshot {
        reg,
        off,
        slot,
        words,
        host,
        backing,
    })
}

pub(crate) fn report(s: Option<&Snapshot>, key: &str, row: &[u32; 16], ptr: &[u32], window: u32) {
    if !enabled() {
        return;
    }
    let gpu: [u32; 4] = ptr[5..9].try_into().unwrap();
    let exact = s.is_some_and(|s| {
        s.reg == row[8]
            && s.slot == ptr[2]
            && (ptr[3] == 0 || ptr[3] == row[10])
            && ptr[4] == 0
            && s.words == Some(gpu)
    });
    let at_report = facts(gpu);
    crate::xlog_info!(
        "mesh V# provenance: paired {key} exact_recorded_slot_and_words={exact} actual_native_heap_index={:?} selected_window={window} stage=report_after_retirement facts={at_report:?}",
        row[9].checked_add(window)
    );
    if let Some(s) = s {
        crate::xlog_info!(
            "mesh V# provenance: recorded {key} root_s17={:#x} table_offset={} slot={:#x} words={:08x?} {} stage=recording facts={:?}",
            s.reg,
            s.off,
            s.slot,
            s.words,
            s.host,
            s.backing
        );
    }
}
