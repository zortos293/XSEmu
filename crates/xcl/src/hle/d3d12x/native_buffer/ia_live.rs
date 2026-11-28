//! Draw-use IA for translated graphics: each live vertex / index view the title bound is
//! gathered on the GPU, in command order, from its owning console buffer into native scratch
//! retained by the recording, and that scratch is bound for this one draw only.

use super::native_buffer::{self, fixed::Lease, live, recipes::{Address, Binding}};
use super::{device_of, pc, set_index_buffer, transition, tracked_list_type, warn, LISTS};
use crate::hle::d3d12x::{bind, console_heap, guard, layout, physical_directory::recordings, pso_async, real};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const PC_IA_SET_VERTEX_BUFFERS: usize = 44;

/// Native scopes a helper dispatch must not be recorded in, as the list's native calls left them.
#[derive(Default)]
pub(super) struct Hazards {
    queries: HashSet<(usize, i32, u32)>,
    render: bool,
    suspending: bool,
    so: u32,
    raw_predicate: bool,
    opaque: bool,
}

impl Hazards {
    fn check(&self) -> Result<(), &'static str> {
        if self.render {
            return Err("active or suspended render pass");
        }
        if !self.queries.is_empty() {
            return Err("active query");
        }
        if self.so != 0 {
            return Err("active stream-output target");
        }
        if self.raw_predicate {
            return Err("raw native predication");
        }
        if self.opaque {
            return Err("state inherited from an executed bundle");
        }
        Ok(())
    }
}

static OBSERVED: AtomicBool = AtomicBool::new(false);

fn observe(list: *mut c_void, f: impl FnOnce(&mut Hazards)) {
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        f(&mut s.helper);
    }
}

pub(super) extern "system" fn query_begin(list: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    observe(list, |h| {
        h.queries.insert((heap, ty.0, index));
    });
}

pub(super) extern "system" fn query_end(list: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    observe(list, |h| {
        h.queries.remove(&(heap, ty.0, index));
    });
}

extern "system" fn pass_begin(list: *mut c_void, flags: D3D12_RENDER_PASS_FLAGS) {
    observe(list, |h| {
        h.render = true;
        h.suspending = flags.contains(D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
    });
}

extern "system" fn pass_end(list: *mut c_void) {
    observe(list, |h| h.render = h.suspending);
}

unsafe extern "system" fn so_targets(list: *mut c_void, first: u32, count: u32, views: *const D3D12_STREAM_OUTPUT_BUFFER_VIEW) {
    observe(list, |h| {
        let Some(end) = first.checked_add(count).filter(|&e| e <= 4) else {
            h.opaque = true;
            return;
        };
        for slot in first..end {
            let bound = !views.is_null() && (*views.add((slot - first) as usize)).BufferLocation != 0;
            if bound {
                h.so |= 1 << slot;
            } else {
                h.so &= !(1 << slot);
            }
        }
    });
}

extern "system" fn native_predicate(list: *mut c_void, resource: usize) {
    observe(list, |h| h.raw_predicate = resource != 0);
}

extern "system" fn bundle(list: *mut c_void) {
    observe(list, |h| h.opaque = true);
}

/// Wrap the adopted command list class's native scope setters (slots[i] = PC method i) so
/// their calls are observed first and then run unchanged. Without this no scope is proven.
pub(crate) fn observe_slots(iface: &layout::Iface, slots: &mut [usize]) {
    for (name, target) in [
        ("BeginQuery", query_begin as usize),
        ("EndQuery", query_end as usize),
        ("BeginRenderPass", pass_begin as usize),
        ("EndRenderPass", pass_end as usize),
        ("SOSetTargets", so_targets as usize),
        ("SetPredication", native_predicate as usize),
        ("ExecuteBundle", bundle as usize),
    ] {
        let Some(i) = layout::slot(iface, name).filter(|&i| i < slots.len()) else { continue };
        let mut jump = vec![0x48, 0xB8];
        jump.extend_from_slice(&(slots[i] as u64).to_le_bytes());
        jump.extend_from_slice(&[0xFF, 0xE0]);
        // BeginRenderPass: its fifth argument (flags) as the observer's second, mov edx, [rsp + 0xB0]
        let body: &[u8] = if name == "BeginRenderPass" { &[0x8B, 0x94, 0x24, 0xB0, 0, 0, 0] } else { &[] };
        slots[i] = guard::call_preserving_arguments(body, target, &jump);
    }
    OBSERVED.store(true, Ordering::Release);
}

fn scope(raw: *mut c_void) -> Result<live::Scope, &'static str> {
    if !OBSERVED.load(Ordering::Acquire) || !unsafe { real::is_adopted(raw) } {
        return Err("command-list scope is not observed");
    }
    let lists = LISTS.lock().unwrap();
    let s = lists.as_ref().and_then(|m| m.get(&(raw as usize))).ok_or("no command-list state")?;
    match s.list_type {
        Some(D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE) => {}
        Some(D3D12_COMMAND_LIST_TYPE_BUNDLE) => return Err("bundle: helper storage would be shared by every execution"),
        _ => return Err("unsupported command-list type"),
    }
    s.helper.check()?;
    if s.occlusion.active.is_some() {
        return Err("active occlusion query");
    }
    if s.bindings.predication.is_some() {
        return Err("raw resource predication");
    }
    let predicate = match s.predication {
        None => None,
        Some(op) if s.pred_in_state => Some((s.pred_buf.clone().ok_or("predicate buffer missing")?, 0, op)),
        Some(_) => return Err("predicate buffer state unknown"),
    };
    let heaps = s
        .bindings
        .heaps
        .iter()
        .map(|&p| unsafe { ID3D12DescriptorHeap::from_raw_borrowed(&(p as *mut c_void)).cloned() }.ok_or("unknown descriptor heap"))
        .collect::<Result<_, _>>()?;
    Ok(live::Scope::new(heaps, predicate))
}

/// The proof a helper may run on the adopted raw list now (None: a live render pass, query,
/// stream output, raw predication, bundle inheritance, or an unobserved list).
pub(crate) fn tracked_scope(raw: *mut c_void) -> Option<live::Scope> {
    scope(raw).ok()
}


struct Gather {
    read: native_buffer::Read,
    plan: live::Pooled,
    data: live::Pooled,
    binding: Binding,
}

/// What a translated draw does with its IA.
pub(crate) enum Disposition {
    /// not a translated graphics draw (or one the draw path skips): IA as before
    Untranslated,
    /// gathered views bound for this draw; dropping it binds the title's views again
    // The IA gather packet type is recorded but not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    Gathered(DrawIa),
    /// a translated draw whose IA could not be gathered safely here, and why
    // The IA gather packet type is recorded but not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    Refused(String),
}

impl Disposition {
    pub(crate) fn gathered(&self) -> bool {
        matches!(self, Self::Gathered(_))
    }
}

pub(crate) struct DrawIa {
    list: *mut c_void,
    vertex: Vec<(u32, Option<D3D12_VERTEX_BUFFER_VIEW>)>,
    index: Option<Option<D3D12_INDEX_BUFFER_VIEW>>,
}

impl Drop for DrawIa {
    fn drop(&mut self) {
        unsafe {
            for (slot, view) in &self.vertex {
                pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(self.list, PC_IA_SET_VERTEX_BUFFERS)(
                    self.list,
                    *slot,
                    1,
                    view.as_ref().map_or(std::ptr::null(), |v| v as *const _),
                );
            }
            if let Some(view) = &self.index {
                set_index_buffer(self.list, view.as_ref());
            }
        }
    }
}

/// Before a draw on list (an indexed one with indexed; DrawIndexedX's own immediate index
/// view), and before its translated pipeline and roots are bound: the helper dispatches change
/// the pipeline and compute roots, and only the heaps and predication are put back here.
pub(crate) unsafe fn prepare(list: *mut c_void, indexed: bool, immediate: Option<D3D12_INDEX_BUFFER_VIEW>) -> Disposition {
    match gather(list, indexed, immediate) {
        Ok(Some(ia)) => Disposition::Gathered(ia),
        Ok(None) => Disposition::Untranslated,
        Err(why) => {
            static N: AtomicU64 = AtomicU64::new(0);
            static REFUSED: AtomicU64 = AtomicU64::new(0);
            warn(&N, "draw-use IA gather", format_args!("refused: {why}"));
            let n = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: draw-use IA gather refused, play81 title-buffer path kept: {n} so far");
            }
            Disposition::Refused(why)
        }
    }
}

unsafe fn gather(list: *mut c_void, indexed: bool, immediate: Option<D3D12_INDEX_BUFFER_VIEW>) -> Result<Option<DrawIa>, String> {
    let title_pso = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).map_or(0, |s| s.bound_pso);
    let pso = pso_async::resolve_wait(title_pso);
    if bind::shader_skip(title_pso) || bind::shader_skip(pso) || real::two_pass_of(pso).is_some() || real::is_native_mesh(pso) || !real::is_translated_graphics(pso) {
        return Ok(None);
    }
    let proof = scope(list)?;
    if tracked_list_type(list) != Some(D3D12_COMMAND_LIST_TYPE_DIRECT) {
        return Err("IA needs a DIRECT list".into());
    }
    let mut recipe = super::ia_recipe(list, indexed && immediate.is_none());
    if let Some(v) = immediate {
        recipe.push(Address { original: v.BufferLocation, binding: Binding::Index { bytes: v.SizeInBytes, format: v.Format.0 } });
    }
    let (shadow_vertex, shadow_index) = {
        let lists = LISTS.lock().unwrap();
        let s = lists.as_ref().and_then(|m| m.get(&(list as usize))).ok_or("lost command-list state")?;
        if !indexed && s.bindings.rect_list && !real::has_rect_list_gs(pso) {
            return Err("rect list without the rect list GS is expanded on the CPU".into());
        }
        (s.bindings.vertex_buffers.clone(), s.index_buffer)
    };
    let locator = match recordings::packet_index(list as usize) { 0 => return Err("no recording locator (unified-memory path)".into()), l => l };
    let mut owners = Vec::<Lease>::new();
    let mut reads = Vec::new();
    for a in recipe {
        let bytes = a.known_bytes().ok_or("not an IA binding")?;
        if a.original == 0 || bytes == 0 {
            if immediate.is_some() && matches!(a.binding, Binding::Index { .. }) {
                return Err("empty DrawIndexedX index view".into());
            }
            continue;
        }
        let (resource, base, _) =
            real::owned_console_buffer_at(a.original as usize, bytes as usize).ok_or("IA view has no unique owning console buffer")?;
        let native = real::buffer_delta(resource.as_raw() as usize)
            .checked_add(a.original - base as u64)
            .ok_or("IA view offset overflows")?;
        let read = native_buffer::Copy::read(a.original, bytes, locator, |page| {
            let (meta, lease) = live::fallback(&resource, a.original, native, page)?;
            owners.push(lease);
            Some(meta)
        })
        .ok_or("IA view page has no pinned fallback")?;
        owners.push(Arc::new(resource));
        reads.push((read, a.binding));
    }
    let mut ia = DrawIa { list, vertex: Vec::new(), index: None };
    if reads.is_empty() {
        return Ok(Some(ia));
    }
    let dev = device_of(list).ok_or("no device for the IA gather")?;
    let views = console_heap::global_heap().ok_or("no shared view heap for the IA gather")?;
    let pipeline = native_buffer::shared_pipeline(&dev)?;
    let mut gathers = Vec::with_capacity(reads.len());
    for (read, binding) in reads {
        let (plan_bytes, data_bytes) = read.storage_bytes();
        gathers.push(Gather { plan: live::scratch(&dev, plan_bytes)?, data: live::scratch(&dev, data_bytes)?, read, binding });
    }
    let gathers = Arc::new(gathers);
    recordings::retain_native(list as usize, Arc::new((owners, gathers.clone(), pipeline.clone(), views.clone())))?;
    let com = ID3D12GraphicsCommandList::from_raw_borrowed(&list).ok_or("null IA list")?;
    // the helper's SetPipelineState runs XCL's own hook, which takes it for the title's pipeline
    let shadow = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).map(|s| (s.bound_pso, s.using_fallback_pso, s.pso_dirty));
    let recorded = live::with_helper(com, &proof, || {
        for g in gathers.iter() {
            pipeline.record_read(com, &views, &g.read, &g.plan, &g.data)?;
        }
        Ok(())
    });
    if let (Some((bound, fallback, dirty)), Some(s)) = (shadow, LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize)))) {
        (s.bound_pso, s.using_fallback_pso, s.pso_dirty) = (bound, fallback, dirty);
    }
    recorded?;
    for g in gathers.iter() {
        let va = g.data.GetGPUVirtualAddress();
        match g.binding {
            Binding::Vertex { slot, bytes, stride } => {
                transition(list, g.data.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: va, SizeInBytes: bytes, StrideInBytes: stride };
                pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(list, PC_IA_SET_VERTEX_BUFFERS)(list, slot, 1, &view);
                ia.vertex.push((slot, shadow_vertex.get(&slot).copied()));
            }
            Binding::Index { bytes, format } => {
                transition(list, g.data.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDEX_BUFFER);
                set_index_buffer(list, Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: va, SizeInBytes: bytes, Format: DXGI_FORMAT(format) }));
                ia.index = Some(shadow_index);
            }
            _ => return Err("not an IA binding".into()),
        }
    }
    Ok(Some(ia))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn native_scope_setters_block_helpers_until_closed() {
        let list = 0x7E57_1A00usize as *mut c_void;
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(list as usize, Default::default());
        let check = || LISTS.lock().unwrap().as_ref().unwrap()[&(list as usize)].helper.check();
        assert_eq!(check(), Ok(()));
        let stats = D3D12_QUERY_TYPE_PIPELINE_STATISTICS;
        query_begin(list, 0x40, stats, 3);
        assert_eq!(check(), Err("active query"));
        query_end(list, 0x40, D3D12_QUERY_TYPE_OCCLUSION, 3);
        assert_eq!(check(), Err("active query"), "another query type is another query");
        query_end(list, 0x40, stats, 3);
        assert_eq!(check(), Ok(()));
        pass_begin(list, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
        assert_eq!(check(), Err("active or suspended render pass"));
        pass_end(list);
        assert_eq!(check(), Err("active or suspended render pass"), "a suspended pass resumes later in this list");
        pass_begin(list, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
        pass_end(list);
        assert_eq!(check(), Ok(()));
        let views = [D3D12_STREAM_OUTPUT_BUFFER_VIEW { BufferLocation: 0x1000, SizeInBytes: 64, BufferFilledSizeLocation: 0x2000 }; 2];
        unsafe { so_targets(list, 2, 2, views.as_ptr()) };
        assert_eq!(check(), Err("active stream-output target"));
        unsafe { so_targets(list, 2, 1, std::ptr::null()) };
        assert_eq!(check(), Err("active stream-output target"), "slot 3 is still bound");
        unsafe { so_targets(list, 3, 1, std::ptr::null()) };
        assert_eq!(check(), Ok(()));
        native_predicate(list, 0x5000);
        assert_eq!(check(), Err("raw native predication"));
        native_predicate(list, 0);
        assert_eq!(check(), Ok(()));
        bundle(list);
        assert_eq!(check(), Err("state inherited from an executed bundle"));
        LISTS.lock().unwrap().as_mut().unwrap().remove(&(list as usize));
    }
}

