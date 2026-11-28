//! Residency priorities for the video memory XCL places the title's textures in.
//!
//! D3D12 pages video memory out under budget pressure by priority: memory at
//! `D3D12_RESIDENCY_PRIORITY_HIGH` is kept, `_LOW` goes first, and what went is copied back from
//! system memory when it is used again (`ID3D12Device1::SetResidencyPriority`; D3D12 devices have
//! it since Windows 10 1607, and this crate's `windows` 0.62 declares it). XCL places the title's
//! textures in 64 MiB page heaps (`vram::Pages`), in arenas, and in 1 MiB slab heaps, and without
//! a priority every one of them is NORMAL: a pool no page holds any more is as likely to stay
//! resident as the frame's 1080p render target (1920x1080 RGBA8 is 7.9 MiB, and its next write
//! waits for whatever was paged out). `residency` (`XCL_VRAM_EVICT`) evicts after the budget is
//! over; a priority says which memory to page out first, before it comes to that.
//!
//! `XCL_RESIDENCY=1` sets one priority per heap:
//! - HIGH for a page pool holding a page of a GPU-written texture (a render target, depth buffer
//!   or UAV texture the next frame writes), and for an arena of one: paging it out stalls the
//!   next frame and copies it back;
//! - NORMAL for a pool of sampled textures only, an arena of sampled textures, and a slab of
//!   small sampled textures: a frame reads them, and the copy that went is re-read from the
//!   title's memory;
//! - LOW for a pool no page holds: cache, and the memory to give back first (the heaps
//!   `XCL_PLACED_CACHE_MB` gives back are the same ones).
//! Unset or `XCL_RESIDENCY=0`: no call at all, and D3D12's default (NORMAL) stands. The first
//! priority set logs the switch, a device without `ID3D12Device1` logs once, and the first four
//! failures log once each.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Whether `XCL_RESIDENCY=1`: a priority is set on the heaps XCL places textures in (`0` or
/// unset: no call, D3D12's NORMAL stands).
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_RESIDENCY").is_ok_and(|v| v.trim() == "1"))
}

/// What a heap's video memory is worth keeping resident.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Priority {
    /// cache: no page of a placed texture holds it, or its textures are cold
    Low,
    /// a frame reads it (sampled textures): pageable back in at the cost of a copy
    Normal,
    /// a frame writes it (render targets, depth buffers, UAV textures): paging it out stalls
    High,
}

impl Priority {
    fn raw(self) -> D3D12_RESIDENCY_PRIORITY {
        match self {
            Priority::Low => D3D12_RESIDENCY_PRIORITY_LOW,
            Priority::Normal => D3D12_RESIDENCY_PRIORITY_NORMAL,
            Priority::High => D3D12_RESIDENCY_PRIORITY_HIGH,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Priority::Low => "low",
            Priority::Normal => "normal",
            Priority::High => "high",
        }
    }
}

/// The priority of a page pool from what it holds (`vram::Pool`): nothing on it is cache (`Low`),
/// a GPU-written page is the frame's target (`High`), sampled pages only are `Normal`. A pool
/// that held a target and now holds sampled textures keeps `High`: the target may come back at
/// that address within `vram::IDLE` and find its contents.
pub(super) fn pool(held: bool, gpu: bool) -> Priority {
    match (held, gpu) {
        (false, _) => Priority::Low,
        (true, true) => Priority::High,
        (true, false) => Priority::Normal,
    }
}

/// `ID3D12Device1` of each device, by device raw pointer (`None`: the device has no
/// `ID3D12Device1`, its priority is left alone).
static DEVICES: Mutex<Option<HashMap<usize, Option<ID3D12Device1>>>> = Mutex::new(None);

fn device1(dev: &ID3D12Device) -> Option<ID3D12Device1> {
    let mut g = DEVICES.lock().unwrap();
    let devices = g.get_or_insert_with(HashMap::new);
    let cached = devices.entry(dev.as_raw() as usize).or_insert_with(|| {
        let asked = dev.cast::<ID3D12Device1>().ok();
        if asked.is_none() {
            static WARNED: AtomicU64 = AtomicU64::new(0);
            if WARNED.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_warn!("D3D12.X: residency: the device has no ID3D12Device1 (SetResidencyPriority): no priorities are set (XCL_RESIDENCY=1)");
            }
        }
        asked
    });
    cached.clone()
}

/// heap raw pointer → the priority last set on it, so a call only goes out on a change.
static SET: Mutex<Option<HashMap<usize, i32>>> = Mutex::new(None);
/// above this many, the NORMAL entries are dropped (a redundant call is cheaper than the map)
const PRUNE_AT: usize = 1 << 13;

/// Sets `heap`'s residency priority, unless it already is: one call per change.
pub(super) fn set(dev: &ID3D12Device, heap: &ID3D12Heap, priority: Priority) {
    if !enabled() {
        return;
    }
    let raw = heap.as_raw() as usize;
    let want = priority.raw().0;
    if SET.lock().unwrap().as_ref().and_then(|m| m.get(&raw)).is_some_and(|&have| have == want) {
        return;
    }
    let Some(device) = device1(dev) else { return };
    // the vtable's call takes the bare pageable pointers (no reference of ours is taken)
    let objects = [heap.as_raw()];
    let priorities = [priority.raw()];
    let hr = unsafe { (device.vtable().SetResidencyPriority)(device.as_raw(), 1, objects.as_ptr(), priorities.as_ptr()) };
    if hr.is_err() {
        static FAILED: AtomicU64 = AtomicU64::new(0);
        if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: residency: SetResidencyPriority({}) on heap {raw:#x} failed: {hr:?}", priority.name());
        }
        return;
    }
    let mut g = SET.lock().unwrap();
    let set = g.get_or_insert_with(HashMap::new);
    if set.len() >= PRUNE_AT {
        let normal = D3D12_RESIDENCY_PRIORITY_NORMAL.0;
        set.retain(|_, &mut have| have != normal);
    }
    set.insert(raw, want);
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        crate::xlog_info!("D3D12.X: residency: XCL_RESIDENCY=1: the page pools, arenas and slabs XCL places the title's textures in get a priority (HIGH while a GPU-written page is on them, NORMAL for sampled ones, LOW when none is; XCL_RESIDENCY=0 sets none)");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three priorities are the D3D12 values, and they order as D3D12 pages memory out: LOW
    /// (0x5000_0000) goes first, HIGH (0xa001_0000, negative as an `i32`) last.
    #[test]
    fn the_priorities_are_the_d3d12_values_in_eviction_order() {
        assert_eq!(Priority::Low.raw(), D3D12_RESIDENCY_PRIORITY_LOW);
        assert_eq!(Priority::Normal.raw(), D3D12_RESIDENCY_PRIORITY_NORMAL);
        assert_eq!(Priority::High.raw(), D3D12_RESIDENCY_PRIORITY_HIGH);
        let unsigned = |p: Priority| p.raw().0 as u32;
        assert_eq!([Priority::Low, Priority::Normal, Priority::High].map(unsigned), [0x5000_0000, 0x7800_0000, 0xa001_0000]);
        assert!(unsigned(Priority::Low) < unsigned(Priority::Normal) && unsigned(Priority::Normal) < unsigned(Priority::High));
        // and the three sit inside D3D12's range, above its MINIMUM and below its MAXIMUM
        assert!(unsigned(Priority::Low) > D3D12_RESIDENCY_PRIORITY_MINIMUM.0 as u32);
        assert!(unsigned(Priority::High) < D3D12_RESIDENCY_PRIORITY_MAXIMUM.0 as u32);
        assert_eq!([Priority::Low.name(), Priority::Normal.name(), Priority::High.name()], ["low", "normal", "high"]);
    }

    /// A pool with nothing on it is cache; one with a GPU-written page is the frame's target and
    /// keeps HIGH even after sampled textures share it.
    #[test]
    fn a_pool_is_cache_only_while_no_page_holds_it() {
        assert_eq!(pool(false, false), Priority::Low);
        assert_eq!(pool(false, true), Priority::Low, "nothing held is cache, whatever the flag says");
        assert_eq!(pool(true, false), Priority::Normal);
        assert_eq!(pool(true, true), Priority::High);
    }
}
