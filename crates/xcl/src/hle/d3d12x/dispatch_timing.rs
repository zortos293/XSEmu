use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const SAMPLES: u32 = 256;

fn selection() -> &'static [u64] {
    static SELECT: OnceLock<Vec<u64>> = OnceLock::new();
    SELECT.get_or_init(|| {
        std::env::var("XCL_GPU_DISPATCH_TIMING").unwrap_or_default().split(',')
            .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()
    })
}

static SHADERS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

fn capture_selection() -> Option<(u64, u32)> {
    static CAPTURE: OnceLock<Option<(u64, u32)>> = OnceLock::new();
    *CAPTURE.get_or_init(|| {
        let value = std::env::var("XCL_GPU_DISPATCH_CAPTURE").ok()?;
        let (hash, count) = value.trim().split_once(':').unwrap_or((value.trim(), "1"));
        let hash = u64::from_str_radix(hash.trim().trim_start_matches("0x"), 16).ok()?;
        let count = count.trim().parse::<u32>().ok()?.min(8);
        (count != 0).then_some((hash, count))
    })
}

fn loop_profile_matches(hash: u64) -> bool {
    if !crate::gpu::recomp::loop_profile_enabled() {
        return false;
    }
    static SELECT: OnceLock<Option<Vec<u64>>> = OnceLock::new();
    SELECT.get_or_init(|| {
        let value = std::env::var("XCL_LOOP_PROFILE").unwrap_or_default();
        (value.trim() != "1").then(|| {
            value.split(',').filter_map(|p| u64::from_str_radix(p.trim(), 16).ok()).collect()
        })
    }).as_ref().is_none_or(|matches| matches.contains(&hash))
}

pub(super) fn note(pso: usize, shader: &[u8]) {
    let selected = selection();
    let capture = capture_selection().map(|(hash, _)| hash);
    let profile = super::gpu_timing::enabled() || super::bind::draw_gate_hashes();
    if selected.is_empty() && capture.is_none() && !profile && !crate::gpu::recomp::loop_profile_enabled() {
        return;
    }
    let hash = crate::gpu::shader::fnv64(shader);
    if profile || selected.contains(&hash) || capture == Some(hash) || loop_profile_matches(hash) {
        SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, hash);
    }
}

pub(super) fn forget(pso: usize) {
    if !selection().is_empty() || capture_selection().is_some() || super::gpu_timing::enabled() || super::bind::draw_gate_hashes() || crate::gpu::recomp::loop_profile_enabled() {
        if let Some(h) = SHADERS.lock().unwrap().as_mut() {
            h.remove(&pso);
        }
    }
}

pub(super) fn profile_shader(pso: usize) -> Option<u64> {
    SHADERS.lock().unwrap().as_ref()?.get(&pso).copied()
}

pub(super) fn loop_shader(pso: usize) -> Option<u64> {
    if !crate::gpu::recomp::loop_profile_enabled() {
        return None;
    }
    profile_shader(pso).filter(|&hash| loop_profile_matches(hash))
}

struct Queries {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    cpu: usize,
    next: u32,
}

unsafe impl Send for Queries {}

static QUERIES: OnceLock<Option<Mutex<Queries>>> = OnceLock::new();

pub(super) fn shader(pso: usize) -> Option<u64> {
    if selection().is_empty() {
        return None;
    }
    SHADERS.lock().unwrap().as_ref()?.get(&pso).copied().filter(|h| selection().contains(h))
}

pub(super) fn take_capture(pso: usize) -> Option<(u64, u32)> {
    static TAKEN: AtomicU32 = AtomicU32::new(0);
    let (hash, limit) = capture_selection()?;
    if TAKEN.load(Ordering::Relaxed) >= limit || SHADERS.lock().unwrap().as_ref()?.get(&pso) != Some(&hash) {
        return None;
    }
    let index = TAKEN.try_update(Ordering::Relaxed, Ordering::Relaxed, |i| (i < limit).then_some(i + 1)).ok()?;
    Some((hash, index + 1))
}

pub(super) unsafe fn dispatch(list: &ID3D12GraphicsCommandList, pso: usize, [x, y, z]: [u32; 3]) {
    let scope = shader(pso).and_then(|shader| {
        let pool = QUERIES.get_or_init(|| {
            let mut dev: Option<ID3D12Device> = None;
            list.GetDevice(&mut dev).ok()?;
            let mut heap = None;
            dev?.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: 2 * SAMPLES,
                NodeMask: 0,
            }, &mut heap).ok()?;
            let (readback, _, cpu) = super::markers::readback_buffer(SAMPLES as u64 * 16)?;
            Some(Mutex::new(Queries { heap: heap?, readback: ID3D12Resource::from_raw(readback as _), cpu, next: 0 }))
        }).as_ref()?;
        let (heap, readback, cpu, query) = {
            let mut p = pool.lock().unwrap();
            if p.next >= 2 * SAMPLES {
                return None;
            }
            let query = p.next;
            p.next += 2;
            (p.heap.clone(), p.readback.clone(), p.cpu, query)
        };
        let offset = query as u64 * 8;
        let at = super::markers::position(list.as_raw() as usize);
        let write = super::cpu_writes::Write::DispatchTiming {
            shader, at, groups: [x, y, z], src: cpu + offset as usize, _buf: readback.clone(),
        };
        if !super::list_x::note_dispatch_timing(list.as_raw() as usize, readback.as_raw() as usize, write) {
            return None;
        }
        Some((heap, readback, query, offset))
    });
    if let Some((heap, _, query, _)) = &scope {
        list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, *query);
    }
    list.Dispatch(x, y, z);
    if let Some((heap, readback, query, offset)) = scope {
        list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, query + 1);
        list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, query, 2, &readback, offset);
    }
}
