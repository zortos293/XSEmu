use std::mem::size_of;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::gpu::recomp::pso::material_gate::{BYTES, MAGIC};

const SLOT_BYTES: usize = 256;
const SLOTS: usize = 256;
const REPORT_WORDS: usize = BYTES / size_of::<u32>();

static ARENA: OnceLock<Option<usize>> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);
static REFUSALS: AtomicUsize = AtomicUsize::new(0);

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_MATERIAL_GATE_PROBE").as_deref() == Ok("1"))
}

pub(super) fn refusal(reason: &str) {
    if REFUSALS.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_warn!("material gate probe: {reason}");
    }
}

pub(super) struct Lease {
    cpu: usize,
    generation: u64,
    header: [u32; 4],
    source_index: u32,
    description: String,
}

impl Lease {
    pub(super) unsafe fn writer_retired(&self, list: usize, generation: u64, q: usize, seq: u64, submit: u64) {
        let w = (self.cpu as *const [u32; SLOT_BYTES / 4]).read_volatile();
        if generation != self.generation || w[..4] != self.header || w[17] != self.source_index || w[4] != 1 {
            return refusal("writer retirement header/generation or unique group-zero heartbeat does not match");
        }
        crate::xlog_info!("writer depth probe: retired list={list:#x} generation={generation} q={q} seq={seq} submit={submit} heartbeat=1 src_idx={} dst_uav={} dimensions={}x{} input=[active,index,dimensions,finite_uv,unit_uv,interior,gather_ordered_nonzero,gather_nan,grid_taps,grid_ordered_nonzero,grid_nan,uv_load_taps,uv_load_ordered_nonzero,uv_load_nan,interior_different,grid_uv_same_base] {:?} center_input=[hits,uv(2),gather(4),grid(4),uv_load(4)] {:08x?} store_sites=[active,index_and_bounds,ordered_nonzero] first4={:?} last4={:?} store_nan={} center_output_hits={} center_output_bits={:08x}; {}", w[17], w[1], w[2], w[3], &w[18..34], &w[34..49], &w[5..17], &w[49..61], w[61], w[62], w[63], self.description);
    }

    pub(super) fn new(
        generation: u64,
        index: u32,
        width: u32,
        height: u32,
        describe: impl FnOnce() -> (Option<u32>, String),
    ) -> Option<(Self, u64)> {
        let Some((cpu, gpu)) = ARENA
            .get_or_init(|| {
                let u = crate::gpu::uma::get()?;
                let p = u.alloc(SLOTS * SLOT_BYTES, SLOT_BYTES)? as usize;
                u.in_committed_window(p, SLOTS * SLOT_BYTES).then_some(p)
            })
            .and_then(|cpu| Some((cpu, super::real::uma_va_of(cpu)?)))
        else {
            refusal("no private coherent UMA arena");
            return None;
        };
        let Ok(slot) = NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            (n < SLOTS).then_some(n + 1)
        }) else {
            refusal("private UMA arena exhausted; slots are never reused by replayable lists");
            return None;
        };
        let offset = slot * SLOT_BYTES;
        let cpu = cpu + offset;
        let header = [MAGIC, index, width, height];
        let mut initial = [0u32; SLOT_BYTES / size_of::<u32>()];
        initial[..header.len()].copy_from_slice(&header);
        let (source_index, description) = describe();
        let source_index = source_index.unwrap_or(u32::MAX);
        initial[17] = source_index;
        unsafe { (cpu as *mut [u32; SLOT_BYTES / size_of::<u32>()]).write(initial) };
        Some((
            Self {
                cpu,
                generation,
                header,
                source_index,
                description,
            },
            gpu + offset as u64,
        ))
    }

    pub(super) unsafe fn report_retired(
        &self,
        generation: usize,
        q: usize,
        seq: u64,
        dimensions: [usize; 2],
        context: &str,
    ) {
        let refuse = |reason| {
            refusal(&format!(
                "retired generation={generation} q={q} seq={seq}: {reason}; {context}"
            ))
        };
        if generation as u64 != self.generation
            || dimensions != [self.header[2] as usize, self.header[3] as usize]
        {
            refuse("twin does not match the prepared generation or dimensions");
            return;
        }
        let words = (self.cpu as *const [u32; REPORT_WORDS]).read_volatile();
        if words[..4] != self.header || words[17] != self.source_index {
            refuse("counter header no longer matches its dispatch");
            return;
        }
        match words[4] {
            1 => {}
            0 => return refuse("no material gate heartbeat"),
            _ => return refuse("multiple material gate heartbeats (replayed dispatch)"),
        }
        crate::xlog_info!(
            "material gate probe: retired generation={generation} q={q} seq={seq} native_index={} width={} height={} stages=[positive_pre,active_post,store_in_bounds] q0={:?} q1={:?} q2={:?} q3={:?}; {context}; {}",
            self.header[1],
            self.header[2],
            self.header[3],
            &words[5..8],
            &words[8..11],
            &words[11..14],
            &words[14..17],
            self.description,
        );
        if std::env::var_os("XCL_MATERIAL_GATE_SOURCE_PC").is_some() {
            crate::xlog_info!("source-depth probe: retired generation={generation} q={q} seq={seq} src_idx={} counters=[active,index_match,dimensions_match,finite_uv,unit_uv,interior_unambiguous,raw_gather_nonzero_ordered,raw_gather_nan,grid_taps,grid_nonzero_ordered,grid_nan,uv_load_taps,uv_load_nonzero_ordered,uv_load_nan,unambiguous_different_taps,grid_uv_same_base] {:?} center=[hits,uv_bits(2),gather(4),grid_load(4),uv_load(4)] {:08x?}", words[17], &words[18..34], &words[34..49]);
        }
    }
}
