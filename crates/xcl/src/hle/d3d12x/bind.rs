//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param, UavBinding};
use super::gpu_timing::{self, Kind as Timed};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the title's compute root signature that `layout` is of
    rs: usize,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    pso: usize,
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    /// (`XCL_GPU_MARKERS`) the user SGPRs `bind_abi` last built here: their first 16 words,
    /// the shader's own register count and the argument bytes passed in registers
    last_io: (Vec<u32>, Option<u32>, u32),
    /// the next dispatch's thread limits (io[`IO_THREAD_LIMIT`]: a `DispatchX` with thread
    /// dimensions), taken by the next compute `abi_views` so every other one gets 0
    limits: [u32; 3],
    /// the next dispatch's group spread (io[`IO_GROUP_SPREAD`]: one past 65,535 groups on an axis),
    /// taken by the next compute `abi_views` as `limits` is
    spread: [u32; 2],
    /// the descriptor tables bound in this recording (`recordings_live`)
    tables: super::console_heap::Tables,
}

static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}

/// ExecuteIndirect calls refused as reading what a skipped dispatch should have written
/// (`indirect::check`), so far (`XCL_ACTIVITY`)
static INDIRECT_REFUSED: AtomicU64 = AtomicU64::new(0);

pub fn indirect_refusals() -> u64 {
    INDIRECT_REFUSED.load(Ordering::Relaxed)
}

/// The indirect calls refused since the last call, by what they waited for: the 6 ranges and
/// producers refused most, each with its pipeline's key and why it has not run. None when none
/// was refused.
pub fn indirect_refusal_detail() -> Option<String> {
    let refused = super::indirect::take_refusals();
    if refused.is_empty() {
        return None;
    }
    let shown: Vec<String> = refused.iter().take(6).map(|&((s, e, p), n)| {
        let (key, _) = super::pso_async::key_of(p);
        let state = if super::pso_async::is_pending(p) {
            "still building".to_string()
        } else if let Some(why) = super::real::why_untranslated(p) {
            why
        } else if super::pso_async::pending_or_built(p) {
            "built since".into()
        } else {
            "its build failed or was left out".into()
        };
        format!("{n} reading {s:#x}..{e:#x} ({} B) wait for pipeline {p:#x} (key {key:016x}): {state}", e - s)
    }).collect();
    let more = refused.len().saturating_sub(shown.len());
    Some(format!("{}{}", shown.join("; "), if more > 0 { format!("; and {more} more ranges") } else { String::new() }))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

/// `XCL_SKIP_FAILED_DRAWS` (default 1): a draw whose pipeline has stand-in shaders
/// (`real::is_standin_pso`: not built yet, or its translation or build failed) is not recorded
/// on the PC list; 0: the stand-in draws (nothing visible).
pub(super) fn skip_failed_draws() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SKIP_FAILED_DRAWS").map_or(true, |v| v.trim() != "0"))
}

/// `XCL_SKIP_FAILED_DISPATCHES` (default 1): as `skip_failed_draws` for dispatches. A dispatch
/// whose pipeline has no translated shader has nothing to run either way; 0 keeps a dispatch
/// waiting for a pipeline still being built (`pso_async::async_pso`).
pub(super) fn skip_failed_dispatches() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SKIP_FAILED_DISPATCHES").map_or(true, |v| v.trim() != "0"))
}

/// draws and dispatches skipped for a pipeline that failed, by the title's pipeline
static FAILED_SKIPS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// A `what` ("draw", "dispatch", ...) with the title's pipeline `title_pso` (`real` as it
/// resolved) was skipped: counted, and logged by its pipeline key at the 1st, 2nd, 4th ... skip
/// of a pipeline that failed (one still being built is counted by `pso_async::note_pending_skip`).
pub(super) fn note_failed_skip(title_pso: usize, real: usize, what: &str) {
    if super::pso_async::pending_or_built(title_pso) {
        super::pso_async::note_pending_skip();
        return;
    }
    let n = {
        let mut skips = FAILED_SKIPS.lock().unwrap();
        let n = skips.get_or_insert_with(HashMap::new).entry(title_pso).or_insert(0);
        *n += 1;
        *n
    };
    if n.is_power_of_two() {
        let (key, shaders) = super::pso_async::key_of(title_pso);
        let why = super::real::why_untranslated(real).or_else(|| super::real::why_untranslated(title_pso))
            .unwrap_or_else(|| "its pipeline has stand-in shaders (the driver refused it, or no translation)".into());
        crate::xlog_warn!(
            "D3D12.X: {what} skipped (#{n} of pipeline {title_pso:#x}, key {key:016x}{}{shaders}): {why}",
            if shaders.is_empty() { "" } else { ", " }
        );
    }
}

/// Whether `list` is a bundle: recorded once, so its binds wait for pipelines still being
/// built (`pso_async::waiting_if`).
fn in_bundle(list: *mut c_void) -> bool {
    super::list_x::tracked_list_type(list) == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
}

pub(super) fn forget_failed_skips(pso: usize) {
    if let Some(m) = FAILED_SKIPS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
}

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        if let Some(u) = crate::gpu::uma::get().filter(|u| u.has_sparse()) {
            crate::gpu::uma::note_coherent_need(u.peak());
        }
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(STATES.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous allocator's until that one is reset, and pipeline
/// and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    super::present::reset_plane_writes(list as usize);
    let (old, blocks) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.rs = 0;
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.tables.clear();
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
    super::indirect::recording_done(list as usize, old);
    super::markers::recording_ended(list as usize);
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.rs = 0;
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    super::present::reset_plane_writes(to as usize);
    let (layout, args, rs, layout_gfx, args_gfx, rs_gfx, pso) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.rs, s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.pso)
    });
    let (old, blocks) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.layout = layout;
        s.args = args;
        s.rs = rs;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.pso = pso;
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
    super::indirect::recording_done(to as usize, old);
    super::markers::recording_ended(to as usize);
}

/// A call whose pipeline is still being built, which would be skipped for it
/// (`pso_async::deferrable`), is put off to the submit instead (`list_x::defer`): true when it
/// was. What a dispatch writes is noted unwritten, as a skipped one's is (an indirect call
/// reading it is skipped too), until it runs at the submit.
pub(super) unsafe fn put_off(this: *mut c_void, what: super::list_x::Deferred) -> bool {
    let pso = with(this, |s| s.pso);
    if in_bundle(this) || (what.draws() && !super::pso_async::defer_draws()) || !super::pso_async::deferrable(pso) || !super::list_x::defer(this, what) {
        return false;
    }
    if !what.draws() {
        note_skipped_writer(this, pso, pso);
    }
    super::pso_async::need_by_submit(pso);
    true
}

/// Whether `list` has a compute and a graphics root signature bound (`list_x::record_put_off`).
pub(super) fn roots_bound(list: *mut c_void) -> (bool, bool) {
    with(list, |s| (s.rs != 0 || s.layout.is_some(), s.rs_gfx != 0 || s.layout_gfx.is_some()))
}

/// A call put off to the submit that could not be recorded there: counted as the skip it was.
pub(super) fn deferred_skipped(draws: bool) {
    count(if draws { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    super::present::reset_plane_writes(list as usize);
    if let Some(s) = STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        super::indirect::recording_done(list as usize, s.allocator);
        console_list::hold_until_reset(s.allocator, s.uma);
    }
    super::markers::recording_ended(list as usize);
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

pub(super) unsafe fn execute_indirect_packet(this:*mut c_void,allocator:usize,title_pso:usize,root:usize,layout:Option<Arc<Layout>>,arguments:&mut Vec<u8>,
    sig:usize,max:u32,args:usize,argument_offset:u64,count:usize,count_offset:u64,scope:&super::list_x::native_buffer::live::Scope)->Result<(),String> {
    let layout=layout.ok_or("compute packet has no root layout")?;
    let recipe=super::indirect::IndirectRecipe::retain(sig,max,args,argument_offset,count,count_offset).ok_or("unknown compute indirect layout")?;
    let pso=super::pso_async::resolve_for_compute_use(title_pso).map_err(|_|"compute pipeline not available")?;
    if !super::real::is_translated_compute(pso){return Err("packet is not a translated compute pipeline".into())}
    with(this,|s|{
        s.pso=title_pso;s.allocator=allocator;s.rs=root;s.layout=Some(layout.clone());s.args.clone_from(arguments);
    });
    translated_indirect(this,pso,root,&layout,arguments,&recipe,scope)?;
    with(this,|s|s.args.clone_from(arguments));
    Ok(())
}

/// ExecuteIndirect of dispatches on `this` with the translated compute pipeline `pso` (resolved)
/// and the title's compute root signature `root`, whose packed arguments `arguments` (of
/// `layout`, `layout_of(root)`) are the records' base: each record's own root constants and root
/// descriptors (guest addresses, as the console's shaders read them) go into a row of io and
/// argument table made on the GPU from the arguments and count `recipe` reads, gathered through
/// the recording's page directory after the writes before them; then the call on XCL's ABI.
/// `scope` is the list's state the helper passes must give back (heaps, predication). Ok: the
/// call is recorded and the argument words the signature sets are 0 in `arguments` (D3D12's
/// state after ExecuteIndirect). Err: nothing of the title's ran and `arguments` is untouched.
pub(super) unsafe fn translated_indirect(this: *mut c_void, pso: usize, root: usize, layout: &Layout, arguments: &mut [u8], recipe: &super::indirect::IndirectRecipe, scope: &super::list_x::native_buffer::live::Scope) -> Result<(), String> {
    use crate::gpu::recomp::{IO_CARRY_AT, IO_MEM_VIEWS, IO_PHYSICAL_TABLE, IO_TRIP_AT, IO_TRIP_WINDOW, IO_UMA_BASE_HI, KERNEL_SGPRS};
    use super::list_x::native_buffer::{fixed::Lease, live, translated};
    if !super::real::is_translated_compute(pso) || super::real::is_translated_graphics(pso) {
        return Err("not a translated compute pipeline".into());
    }
    if !console_list::layout_of(root).is_some_and(|l| std::ptr::eq(&*l, layout)) {
        return Err("the root arguments are not of the guest root signature".into());
    }
    let (root_raw, pso_raw) = (root as *mut c_void, pso as *mut c_void);
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).ok_or("no command list")?;
    let guest_root = ID3D12RootSignature::from_raw_borrowed(&root_raw).ok_or("no guest root signature")?;
    let pipeline = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("no pipeline")?;
    let abi = super::real::pso_root_signature(pso)
        .or_else(|| super::real::abi_rs_cached().map(|p| p as usize))
        .and_then(|r| ID3D12RootSignature::from_raw_borrowed(&(r as *mut c_void)).cloned())
        .ok_or("no XCL ABI root signature")?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    super::real::shader_window_base(&super::real::uma_window_vas())?;
    let mut io = [0u32; KERNEL_SGPRS];
    io[IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    io[IO_MEM_VIEWS] = super::real::mem_views().ok_or("memory windows have no views")?;
    io[IO_PHYSICAL_TABLE] = match super::physical_directory::recordings::packet_index(this as usize) { 0 => return Err("no recording locator (unified-memory path)".into()), l => l };
    io[IO_CARRY_AT] = carry_watch();
    (io[IO_TRIP_WINDOW], io[IO_TRIP_AT]) = trip_watch(this, with(this, |s| s.pso), pso, false);
    let mut owners: Vec<Lease> = Vec::new();
    let (mut rows, reads) = recipe
        .prebuilt_translated(pso, guest_root, arguments, &io, |_| false, |op, page| {
            let (meta, lease) = live::fallback(&op.resource, op.guest_address?, op.native_offset, page)?;
            owners.push(lease);
            Some(meta)
        })
        .ok_or("no proven rows for the signature or no GPU gather of its arguments and count")?;
    rows.spread = indirect_spread(with(this, |s| s.pso), pso);
    if recipe.max == 0 {
        rows.reset_arguments(arguments);
        return Ok(());
    }
    let take = |n: usize| with(this, |s| s.arena.take(u, n.max(16), 256, &mut s.uma)).map(|p| p as usize);
    let res_at = take(256).ok_or("unified memory is full")?;
    std::ptr::write_bytes(res_at as *mut u8, 0, 256);
    let res = super::real::uma_va_of(res_at).ok_or("res is not open on the GPU")?;
    translated::execute(list, &rows, recipe.max, reads, owners, take, &abi, pipeline, res, scope, false, || {})?;
    rows.reset_arguments(arguments);
    Ok(())
}

/// `translated_indirect` for a title's ExecuteIndirect on `this` (offsets as the title gave them),
/// with the list's kept compute root arguments, which keep D3D12's state after it.
unsafe fn lower_indirect(this: *mut c_void, pso: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) -> Result<(), String> {
    let scope = super::list_x::tracked_scope(this).ok_or("the list's queries, predication or heaps are not known to be restorable")?;
    let recipe = super::indirect::IndirectRecipe::retain(sig, max, args, args_off, count_buf, count_off).ok_or("no retained layout or buffers for the call")?;
    let (root, layout, mut arguments) = with(this, |s| (s.rs, s.layout.clone(), s.args.clone()));
    let layout = layout.ok_or("no compute root signature")?;
    translated_indirect(this, pso, root, &layout, &mut arguments, &recipe, &scope)?;
    with(this, |s| {
        if s.rs == root {
            s.args = arguments;
        }
    });
    Ok(())
}

/// A title's ExecuteIndirect of draws with the translated graphics pipeline `pso` on `this`
/// (offsets as the title gave them): its arguments and count gathered through the recording's
/// page directory, one row of io and argument table per record from the list's graphics root
/// arguments, the draws capped as the sanitiser caps them, then the call on XCL's ABI after
/// `before_call`. The bound vertex and index views are gathered as for a draw (`ia_live`); views
/// it refuses stay as they are. Ok: recorded, and the graphics root words the signature sets are
/// 0 (D3D12's state after it). Err: nothing of the title's ran.
#[allow(clippy::too_many_arguments)]
unsafe fn lower_draw_indirect(this: *mut c_void, pso: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64, before_call: impl FnOnce()) -> Result<(), String> {
    use super::list_x::native_buffer::{fixed::Lease, live, translated};
    if !super::real::is_translated_graphics(pso) || super::real::two_pass_of(pso).is_some() || super::real::is_native_mesh(pso) {
        return Err("not a translated graphics pipeline".into());
    }
    super::list_x::ensure_view_heap(this);
    let scope = super::list_x::tracked_scope(this).ok_or("the list's queries, predication or heaps are not known to be restorable")?;
    let recipe = super::indirect::IndirectRecipe::retain(sig, max, args, args_off, count_buf, count_off).ok_or("no retained layout or buffers for the call")?;
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).ok_or("no command list")?;
    let root = with(this, |s| s.rs_gfx);
    let root_raw = root as *mut c_void;
    let guest_root = ID3D12RootSignature::from_raw_borrowed(&root_raw).ok_or("no graphics root signature")?;
    let (real, views, layout, mut arguments, io) = abi_views(this, pso, RootStage::Graphics)?;
    if !layout.zip(console_list::layout_of(root)).is_some_and(|(kept, guest)| Arc::ptr_eq(&kept, &guest)) {
        return Err("the graphics root arguments are not of the guest root signature".into());
    }
    let base: &[u32; crate::gpu::recomp::KERNEL_SGPRS] = io.as_slice().try_into().map_err(|_| "graphics io is not one kernel block")?;
    if base[crate::gpu::recomp::IO_PHYSICAL_TABLE] == 0 {
        return Err("no recording locator (unified-memory path)".into());
    }
    let mut owners: Vec<Lease> = Vec::new();
    let (rows, reads) = recipe
        .prebuilt_translated(real, guest_root, &arguments, base, |_| false, |op, page| {
            let (meta, lease) = live::fallback(&op.resource, op.guest_address?, op.native_offset, page)?;
            owners.push(lease);
            Some(meta)
        })
        .ok_or("no proven rows for the signature or no GPU gather of its arguments and count")?;
    if recipe.max != 0 {
        let abi = super::real::pso_root_signature(real)
            .or_else(|| super::real::abi_rs_cached().map(|p| p as usize))
            .and_then(|r| ID3D12RootSignature::from_raw_borrowed(&(r as *mut c_void)).cloned())
            .ok_or("no XCL ABI root signature")?;
        let real_raw = real as *mut c_void;
        let pipeline = ID3D12PipelineState::from_raw_borrowed(&real_raw).ok_or("no pipeline")?;
        let u = crate::gpu::uma::get().ok_or("no unified memory")?;
        let take = |n: usize| with(this, |s| s.arena.take(u, n.max(16), 256, &mut s.uma)).map(|p| p as usize);
        let ia = super::list_x::ia_live::prepare(this, rows.action == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, None);
        translated::execute(list, &rows, recipe.max, reads, owners, take, &abi, pipeline, views[1], &scope, true, before_call)?;
        drop(ia);
    }
    rows.reset_arguments(&mut arguments);
    with(this, |s| {
        if s.rs_gfx == root {
            s.args_gfx = arguments;
        }
    });
    Ok(())
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// The descriptor table at console address `addr` is bound as parameter `i`: the heap pages
/// it reaches by the root signature are in use (`console_heap::table_bound`; no root signature:
/// to the heap's end), and the recording keeps the table for as long as it can run
/// (`recordings_live`) – from the start, so that a recording older than the following of tables
/// is known as well.
fn table_bound(list: *mut c_void, gfx: bool, i: u32, addr: u32) {
    let keeping = super::console_heap::keeping();
    if !keeping && !super::console_heap::tables_followed() {
        return;
    }
    let slots = with(list, |s| {
        let slots = (if gfx { &s.layout_gfx } else { &s.layout }).as_ref().and_then(|l| l.tables.get(i as usize).copied()).unwrap_or(u32::MAX);
        if keeping {
            s.tables.note(addr, slots);
        }
        slots
    });
    super::console_heap::table_bound(addr, slots);
}

/// Lists go to a queue: once a frame (`console_heap::recordings_due`), the pages that the
/// tables of every recording alive reach – the title's lists and bundles here, the console
/// lists' (`console_list::tables_live`) – are in use (`console_heap::recordings_live`).
pub(super) fn recordings_live() {
    if !super::console_heap::recordings_due() {
        return;
    }
    let (mut all, mut recordings) = console_list::tables_live();
    if let Some(m) = STATES.lock().unwrap().as_ref() {
        for s in m.values().filter(|s| !s.tables.is_empty()) {
            all.join(&s.tables);
            recordings += 1;
        }
    }
    super::console_heap::recordings_live(&all, recordings);
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    // the same root signature again keeps the root arguments (D3D12: only a change of root
    // signature leaves them undefined)
    if with(this, |s| s.rs_gfx == rs as usize && s.layout_gfx.is_some()) {
        return;
    }
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    table_bound(this, true, i, handle as u32);
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

pub(super) fn draw_gate_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DRAW_GATE_TRACE").is_ok_and(|v| v == "1"))
}

pub(super) fn ignore_predication() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_IGNORE_PREDICATION").is_ok_and(|v| v == "1"))
}

fn parse_shader_skips(value: &str) -> Vec<u64> {
    value.split(',').filter_map(|word| {
        let word = word.trim();
        u64::from_str_radix(word.strip_prefix("0x").or_else(|| word.strip_prefix("0X")).unwrap_or(word), 16)
            .ok().filter(|&hash| hash != 0)
    }).collect()
}

fn shader_skips() -> &'static [u64] {
    static HASHES: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    HASHES.get_or_init(|| parse_shader_skips(&std::env::var("XCL_SKIP_SHADERS").unwrap_or_default()))
}

struct LiveShaderSkips {
    phase: u64,
    hits: Vec<(u64, u64)>,
}

static LIVE_SHADER_SKIPS: Mutex<LiveShaderSkips> = Mutex::new(LiveShaderSkips { phase: 0, hits: Vec::new() });
static LIVE_SHADER_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn live_shader_file_configured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let Some(path) = std::env::var_os("XCL_SKIP_SHADERS_FILE").filter(|v| !v.is_empty()) else {
            return false;
        };
        match std::thread::Builder::new().name("xcl-shader-skip-file".into()).spawn(move || {
            watch_shader_skip_file(std::path::Path::new(&path));
        }) {
            Ok(_) => true,
            Err(e) => {
                crate::xlog_warn!("D3D12.X: shader skip file watcher unavailable: {:?}", e.kind());
                false
            }
        }
    })
}

fn read_shader_skip_file(path: &std::path::Path) -> std::io::Result<Vec<u64>> {
    use std::io::{Error, ErrorKind, Read};
    let mut input = String::new();
    match std::fs::File::open(path) {
        Ok(file) => file.take(4097).read_to_string(&mut input)?,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    if input.len() > 4096 {
        return Err(Error::new(ErrorKind::InvalidData, "shader skip file exceeds 4096 bytes"));
    }
    let mut hashes = Vec::new();
    for word in input.split(|c: char| c == ',' || c.is_whitespace()).filter(|word| !word.is_empty()) {
        let hash = u64::from_str_radix(word.strip_prefix("0x").or_else(|| word.strip_prefix("0X")).unwrap_or(word), 16)
            .ok().filter(|&h| h != 0).ok_or_else(|| Error::new(ErrorKind::InvalidData, "invalid shader hash"))?;
        hashes.push(hash);
    }
    hashes.sort_unstable();
    hashes.dedup();
    Ok(hashes)
}

fn watch_shader_skip_file(path: &std::path::Path) {
    let mut last_error = None;
    loop {
        match read_shader_skip_file(path) {
            Ok(hashes) => {
                last_error = None;
                let mut live = LIVE_SHADER_SKIPS.lock().unwrap();
                if !live.hits.iter().map(|&(hash, _)| hash).eq(hashes.iter().copied()) {
                    let (frame, title) = super::present::counters();
                    let previous = live.hits.iter().map(|(hash, hits)| format!("{hash:016x}={hits}")).collect::<Vec<_>>().join(", ");
                    crate::xlog_info!("D3D12.X: shader skip file phase {} -> {} at {:.3}s, composite frame {frame}, title epoch {title}; previous hits [{previous}]; active {:016x?}",
                        live.phase, live.phase + 1, crate::log::uptime(), hashes);
                    live.phase += 1;
                    live.hits = hashes.into_iter().map(|hash| (hash, 0)).collect();
                    LIVE_SHADER_ACTIVE.store(!live.hits.is_empty(), Ordering::Release);
                }
            }
            Err(e) => {
                let error = e.kind();
                if last_error != Some(error) {
                    crate::xlog_warn!("D3D12.X: shader skip file ignored ({error:?}); keeping prior selection");
                    last_error = Some(error);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

pub(super) fn draw_gate_hashes() -> bool {
    live_shader_file_configured() || draw_gate_trace() || !shader_skips().is_empty()
}

pub(super) fn draw_gate_calls() -> bool {
    draw_gate_trace() || !shader_skips().is_empty() || LIVE_SHADER_ACTIVE.load(Ordering::Acquire)
}

pub(super) fn shader_skip(pso: usize) -> bool {
    if super::raytracing::builder_held(pso) {
        return true;
    }
    let skips = shader_skips();
    let live = LIVE_SHADER_ACTIVE.load(Ordering::Acquire);
    if skips.is_empty() && !live {
        return false;
    }
    let pso = super::pso_async::resolve_if_ready(pso);
    let stages = super::real::draw_gate_shaders(pso).unwrap_or_default();
    let cs = super::dispatch_timing::profile_shader(pso);
    let mut skipped = false;
    if live {
        let mut live = LIVE_SHADER_SKIPS.lock().unwrap();
        let phase = live.phase;
        for (hash, hits) in &mut live.hits {
            if stages.contains(hash) || cs == Some(*hash) {
                *hits = hits.saturating_add(1);
                if hits.is_power_of_two() {
                    let (frame, title) = super::present::counters();
                    crate::xlog_info!("D3D12.X: draw gate skipped shader {hash:016x}, pipeline {pso:#x}, file phase {phase} hits {hits}, composite frame {frame}, title epoch {title}");
                }
                skipped = true;
            }
        }
    }
    if let Some(hash) = stages.into_iter().chain(cs).find(|hash| skips.contains(hash)) {
        static SEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap();
        if !seen.contains(&hash) {
            seen.push(hash);
            crate::xlog_info!("D3D12.X: draw gate skipped shader {hash:016x}, pipeline {pso:#x}");
        }
        skipped = true;
    }
    skipped
}

pub(super) fn indirect_max(requested: u32) -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    requested.min(*LIMIT.get_or_init(|| std::env::var("XCL_INDIRECT_MAX_COUNT").ok()
        .and_then(|v| v.trim().parse().ok()).filter(|&n| n != 0).unwrap_or(u32::MAX)))
}

fn draw_gate_event() -> Option<(u64, u64, usize)> {
    if !draw_gate_trace() {
        return None;
    }
    let (frame, slot) = super::present::draw_gate_capture(12)?;
    static EVENTS: Mutex<(Option<u64>, u64)> = Mutex::new((None, 0));
    let mut events = EVENTS.lock().unwrap();
    if events.0 != Some(frame) {
        *events = (Some(frame), 0);
    }
    events.1 += 1;
    if events.1 == 32_769 {
        crate::xlog_warn!("D3D12.X: draw gate title epoch {frame}: trace truncated after 32768 events");
    }
    if events.1 > 32_768 {
        return None;
    }
    Some((frame, events.1, slot as usize))
}

pub(super) fn draw_gate_call(this: *mut c_void, pso: usize, name: &str, args: std::fmt::Arguments<'_>, skipped: bool) {
    let Some((frame, event, _)) = draw_gate_event() else { return };
    let predicate = super::list_x::draw_gate_predicate(this);
    let writer = predicate.and_then(|p| p.title).map(|va| writer_of(va).0);
    let resolved = super::pso_async::resolve_if_ready(pso);
    let shaders = super::real::draw_gate_shaders(resolved);
    let cs = super::dispatch_timing::profile_shader(resolved);
    crate::xlog_info!("D3D12.X: draw gate title epoch {frame}/{event} list {:#x} {name} {args}; pso {pso:#x} stages {shaders:016x?} cs {cs:016x?}; active predicate {predicate:x?} writer candidate {writer:?}; shader match at record point {skipped}", this as usize);
}

pub(super) fn draw_gate_predication(this: *mut c_void, name: &str, args: std::fmt::Arguments<'_>, title: Option<u64>) {
    let Some((frame, event, _)) = draw_gate_event() else { return };
    let words = title.and_then(|va| crate::memmap::title_words(va as usize, 2));
    let writer = title.map(|va| writer_of(va).0);
    let predicate = super::list_x::draw_gate_predicate(this);
    crate::xlog_info!("D3D12.X: draw gate title epoch {frame}/{event} list {:#x} {name} {args}; CPU predicate words {words:08x?}; writer candidate {writer:?}; active predicate {predicate:x?}; ignore {}", this as usize, ignore_predication());
}

#[allow(clippy::too_many_arguments)]
pub(super) fn draw_gate_indirect(this: *mut c_void, pso: usize, sig: usize, requested: u32, max: u32, args: usize, args_off: u64, count: usize, count_off: u64, skipped: bool) -> Option<(u64, u64, usize)> {
    let (frame, event, slot) = draw_gate_event()?;
    let (a, c) = (resource_title(args, args_off), resource_title(count, count_off));
    let layout = super::indirect::signature(sig);
    let words = a.zip(layout).and_then(|(va, (_, stride, _))| crate::memmap::title_words(va as usize, (stride as usize / 4).min(8)));
    let observed = c.and_then(|va| crate::memmap::title_words(va as usize, 1)).and_then(|w| w.first().copied());
    let command_limit = (count == 0).then_some(max);
    let writers = (a.map(|va| writer_of(va).0), c.map(|va| writer_of(va).0));
    let predicate = super::list_x::draw_gate_predicate(this);
    let pred_writer = predicate.and_then(|p| p.title).map(|va| writer_of(va).0);
    let resolved = super::pso_async::resolve_if_ready(pso);
    let shaders = super::real::draw_gate_shaders(resolved);
    let cs = super::dispatch_timing::profile_shader(resolved);
    crate::xlog_info!("D3D12.X: draw gate title epoch {frame}/{event} list {:#x} ExecuteIndirect sig {sig:#x} layout {layout:?} requested {requested} max {max} count without buffer {command_limit:?}; pso {pso:#x} stages {shaders:016x?} cs {cs:016x?}; args {args:#x}+{args_off:#x} title {a:x?} CPU first record {words:08x?}; count {count:#x}+{count_off:#x} title {c:x?} CPU count {observed:?}; arg/count writer candidates {writers:?}; active predicate {predicate:x?} writer candidate {pred_writer:?}; shader match at record point {skipped}", this as usize);
    Some((frame, event, slot))
}

pub(super) fn resource_title(resource: usize, offset: u64) -> Option<u64> {
    if resource == 0 {
        return None;
    }
    (super::real::title_address(resource)? as u64).checked_add(offset)
}

#[test]
fn draw_gate_shader_list_accepts_only_nonzero_hex() {
    assert_eq!(parse_shader_skips(" 0x123, ABC, nope, , 0, 0Xf"), vec![0x123, 0xabc, 0xf]);
}

pub(super) unsafe fn capture_selected_depth(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, visible: usize, event: Option<(u64, u64, usize)>, pc: bool) {
    if crate::gpu::recomp::pso::material_gate::writer_mode() { return; }
    let Some(ps) = super::depth_capture_views::selected_ps() else { return };
    if super::real::pso_shaders(visible).map(|(_, h)| h) != Some(ps) { return }
    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |index: u32, why: &str| {
        let bit = 1u32 << (index + if pc { 8 } else { 0 });
        if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            crate::xlog_info!("depth capture: selected PS {ps:016x} path={} list={this:p} native_pso={pso:#x} visible_pso={visible:#x}: {why}", if pc { "PC indirect" } else { "console indirect" });
        }
    };
    log(0, "reached draw");
    if !super::depth_capture::due() { return log(1, "capture already claimed or disabled") }
    if pso != visible || !super::real::capture_depth_equal_pso(pso) { return log(2, "native EQUAL PSO not proven") }
    static ARM: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let path = ARM.get_or_init(|| std::env::var_os("XCL_DEPTH_CAPTURE_ARM_FILE").map(Into::into));
    let Some(path) = path.as_ref() else { return log(3, "arm path unset") };
    if !path.is_file() { return log(4, "waiting for arm file") }
    log(5, "armed; checking DSV");
    let (frame, epoch) = super::present::counters();
    let state = super::list_x::state_summary(this);
    let Some((handle, view_generation)) = super::list_x::capture_depth_binding(this) else {
        return note_depth_skip("no unpredicated tracked DSV binding", &state)
    };
    let Some(view) = super::depth_capture_views::lookup(handle, view_generation) else {
        return note_depth_skip("DSV binding changed or owner unavailable", &state)
    };
    if view.array_size != 1 { return note_depth_skip("multi-slice view", &state) }
    let d = view.resource.GetDesc();
    let Some(sub) = view.first_array_slice.checked_mul(u32::from(d.MipLevels)).and_then(|s| s.checked_add(view.mip)) else { return };
    let resource = view.resource.as_raw() as usize;
    super::depth_uav::select_twin_probe(resource, view.mip, view.first_array_slice, sub);
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT};
    super::depth_uav::report_slot146(this as usize, resource, sub);
    let stencil_sub = matches!(d.Format, DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
        .then(|| u32::from(d.MipLevels).checked_mul(u32::from(d.DepthOrArraySize)).and_then(|stride| sub.checked_add(stride))).flatten();
    super::depth_lineage::report(resource, sub, stencil_sub, this as usize);
    let twin = super::depth_uav::capture_twin(resource);
    let backing = super::real::title_address(resource).or_else(|| super::title_textures::address(resource));
    let context = format!("recording at present_count {frame}, title_epoch {epoch}, gate_event {event:?}, list {this:p} PS {ps:016x} EQUAL PSO {pso:#x} DSV {handle:#x} generation {view_generation} depth {resource:#x} view_format {} resource_format {} title_backing {backing:x?} mip {} slice {} read_only={} sub {sub} twin(resource,format,exact,ever_written) {twin:x?} recorded_clear {:?}; {state}", view.format.0, d.Format.0, view.mip, view.first_array_slice, view.read_only_depth, view.last_clear);
    let Some(before) = super::depth_state::proven(this as usize, resource, sub) else {
        return note_depth_skip("no proven copy-safe depth state on this recording", &context)
    };
    let before = match before {
        super::depth_state::State::Legacy(v) => super::depth_capture::Before::Legacy(v.0 as u32),
        super::depth_state::State::Enhanced { layout, access, sync } => super::depth_capture::Before::Enhanced {
            layout: layout.0, access: access.0 as u32, sync: sync.0 as u32,
        },
    };
    if let Err(e) = super::depth_capture::record(list, &view.resource, view.format, view.mip, view.first_array_slice, 0, before, view.last_clear.map(|v| v.depth), context) {
        note_depth_skip(&e, &state);
    }
}

fn note_depth_skip(reason: &str, context: &str) {
    static MESSAGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = MESSAGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 5 || n.is_power_of_two() {
        crate::xlog_warn!("depth capture: not recorded: {reason}; {context}");
    }
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    if super::indirect::dispatches(sig) && chain_held(this, with(this, |s| s.pso)) {
        return;
    }
    // (only a signature of the one call: root arguments or buffer views it set would be missing
    // from the list after it)
    let one_call = super::indirect::signature(sig).is_some_and(|(last, _, single)| single && [super::indirect::DRAW, super::indirect::DRAW_INDEXED, super::indirect::DISPATCH].contains(&last));
    if one_call && put_off(this, super::list_x::Deferred::Indirect(sig, max, args, args_off, count_buf, count_off)) {
        return;
    }
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let title_pso = with(this, |s| s.pso);
    let kind = super::indirect::note_call(sig);
    let requested = max;
    let max = indirect_max(requested);
    let skipped = shader_skip(title_pso);
    let event = draw_gate_indirect(this, title_pso, sig, requested, max, args, args_off, count_buf, count_off, skipped);
    let dispatches = super::indirect::dispatches(sig);
    let bundle = in_bundle(this);
    let pso = if dispatches {
        match super::pso_async::waiting_if(bundle, || super::pso_async::resolve_for_compute_use(title_pso)) {
            Ok(pso) => pso,
            Err(skip) => return note_pending_compute_skip(this, title_pso, "ExecuteIndirect(Dispatch)", skip),
        }
    } else {
        super::pso_async::waiting_if(bundle, || super::pso_async::resolve_wait(title_pso))
    };
    let failed = super::real::is_standin_pso(pso) && if dispatches { skip_failed_dispatches() } else { skip_failed_draws() };
    if failed {
        note_failed_skip(title_pso, pso, if dispatches { "ExecuteIndirect(Dispatch)" } else { "ExecuteIndirect" });
    }
    if skipped || shader_skip(pso) || failed {
        if dispatches {
            note_skipped_writer(this, pso, title_pso);
        }
        count(if dispatches { DISPATCHES_SKIPPED } else { DRAWS_SKIPPED });
        return;
    }
    note_named_use(title_pso, pso, "indirect call");
    // a mesh signature needs a mesh pipeline and a mesh pipeline a mesh signature: the PC call
    // with any other pair is invalid (a PC pipeline or a stand-in is the title's own pair)
    let mesh_sig = super::indirect::dispatches_mesh(sig);
    let two_pass = super::real::two_pass_of(pso);
    if two_pass.as_ref().is_some_and(|tp| tp.kind != "MS") {
        mesh_skipped(this, pso, "an ExecuteIndirect", GS_TWO_PASS);
        return;
    }
    let mesh_pso = two_pass.is_some() || super::real::is_native_mesh(pso);
    if mesh_sig != mesh_pso && (mesh_pso || super::real::is_translated_graphics(pso) || super::real::is_translated_compute(pso)) {
        mesh_skipped(this, pso, "an ExecuteIndirect", if mesh_sig { "its pipeline has no mesh shader" } else { "its signature does not dispatch mesh work and its pipeline is a mesh pipeline" });
        return;
    }
    note_indirect(pso, sig, max, args, args_off, count_buf, count_off);
    // console addresses of the arguments and count: an indirect call reading what a skipped
    // dispatch should have written is skipped too
    let (args_title, count_title) = (resource_title(args, args_off), resource_title(count_buf, count_off));
    if let Err(why) = super::indirect::check(sig, max, args_title, count_title) {
        INDIRECT_REFUSED.fetch_add(1, Ordering::Relaxed);
        count(if dispatches { DISPATCHES_SKIPPED } else { DRAWS_SKIPPED });
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect skipped (pipeline {pso:#x}, at most {max} commands): {why}");
        }
        return;
    }
    let (gfx, compute) = (super::real::is_translated_graphics(pso) || two_pass.is_some(), super::real::is_translated_compute(pso));
    let (title_args_off, title_count_off) = (args_off, count_off);
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    reach(this);
    let execute = |safe: Option<(usize, u64, u64)>| {
        if super::depth_lineage::enabled() && max != 0
            && super::indirect::signature(sig).is_some_and(|(kind, _, _)| matches!(kind, 0 | 1))
        {
            super::list_x::note_lineage_depth_draw(this, "console_indirect_draw");
        }
        super::residency::note_list_resources(this as usize, [args, count_buf]);
        super::residency::note_list_untracked(this as usize);
        gpu_timing::op(this, Timed::Indirect, || title_pso, || match safe {
            Some((scratch, records, n)) => pc(this, sig, max, scratch, records, scratch, n),
            None => fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off)),
        })
    };
    if !gfx && !compute {
        // The stand-in computes and draws nothing, and the list may hold XCL's ABI root
        // signature from a translated call before it – the PC call with that pair is invalid
        // (debug layer 953, "the currently set Root Signature doesn't match the currently set
        // Pipeline State Object"; a retail title's world then hung the GPU, 0x887a0006).
        if let Some(skip) = indirect_skip(dispatches, title_pso) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let log = N.fetch_add(1, Ordering::Relaxed) < 8;
            match skip {
                IndirectSkip::Untranslated => {
                    note_skipped_writer(this, pso, title_pso);
                    count(DISPATCHES_SKIPPED);
                    if log {
                        let why = super::real::why_untranslated(pso).unwrap_or_else(|| "no reason kept (not made through XCL's compute pipeline path)".into());
                        crate::xlog_warn!("D3D12.X: ExecuteIndirect of dispatches with a pipeline that has no translated console shader ({pso:#x}) skipped: {why}");
                    }
                }
                IndirectSkip::Building => {
                    count(DRAWS_SKIPPED);
                    if log {
                        crate::xlog_warn!("D3D12.X: ExecuteIndirect of draws with a pipeline still being built after the draw backstop (XCL_BIND_BACKSTOP_MS) skipped");
                    }
                }
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
            list.SetGraphicsRootSignature(r);
        }
        execute(None);
        mark(this, kind);
        return;
    }
    // compute records on XCL's ABI, root arguments of their own included: rows of io built on the
    // GPU from arguments gathered through the recording's page directory
    if compute && args_title.is_some() {
        match lower_indirect(this, pso, sig, max, args, title_args_off, count_buf, title_count_off) {
            Ok(()) => {
                super::residency::note_list_untracked(this as usize);
                count(DISPATCHES);
                mark(this, kind);
                note_dispatch_binding(this, title_pso, None);
                note_built_writes(this, title_pso);
                return;
            }
            Err(why) => {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_info!("D3D12.X: ExecuteIndirect of translated dispatches (pipeline {pso:#x}) not lowered to rows: {why}");
                }
            }
        }
    }
    if gfx && !mesh_sig && two_pass.is_none() && args_title.is_some() {
        let depth = || {
            if let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) {
                capture_selected_depth(this, list, pso, pso, event, false);
            }
        };
        match lower_draw_indirect(this, pso, sig, max, args, title_args_off, count_buf, title_count_off, depth) {
            Ok(()) => {
                super::residency::note_list_resources(this as usize, [args, count_buf]);
                super::residency::note_list_untracked(this as usize);
                count(DRAWS);
                mark(this, kind);
                return;
            }
            Err(why) => {
                static N: AtomicU64 = AtomicU64::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: ExecuteIndirect of translated draws (pipeline {pso:#x}) kept on the sanitiser path: {why}; {n} so far");
                }
            }
        }
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        if compute {
            note_skipped_writer(this, pso, title_pso);
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    // the arguments sanitised on the GPU first: XCL's pass binds its own compute pipeline, so it
    // goes before the call's own bindings
    // `XCL_GPU_MARKERS`: the first compute calls after any skip keep their words for the log
    let capture = compute && super::markers::enabled() && capture_now();
    let timed_capture = event.filter(|&(epoch, _, slot)| {
        (count_buf == 0 || count_title.is_some())
            && super::present::draw_gate_capture(0) == Some((epoch, slot as u64))
            && super::list_x::draw_gate_predicate(this).is_none()
    });
    if let Some(tp) = two_pass {
        // in two passes: each command's prepass and mesh call read the sanitised copy
        let plan = two_pass_indirect_plan(&tp, requested);
        let safe = args_title.and_then(|a| super::indirect::sanitise(this as usize, list, sig, max, a, count_title, false, Some(plan)));
        match two_pass_indirect(this, list, pso, &tp, sig, max, safe, plan, pc, (args_title, count_title)) {
            Ok(()) => {
                capture_draw_gate_indirect(this, list, timed_capture, safe, args_title, count_title, pso, false);
                count(DRAWS);
                mesh_ran(MESH_TWO_PASS);
                mark(this, "ExecuteIndirect(mesh dispatch in two passes)");
            }
            Err(e) => {
                count(DRAWS_SKIPPED);
                mesh_skipped(this, pso, "an ExecuteIndirect", &e);
            }
        }
        return;
    }
    let raw = timed_capture.is_some() && indirect_world_trace() && dispatches && max == 1
        && super::indirect::signature(sig).is_some_and(|(_, stride, single)| stride == 12 && single);
    let safe = args_title.and_then(|a| super::indirect::sanitise(this as usize, list, sig, max, a, count_title, capture || raw, None));
    let call = if mesh_sig { "indirect_mesh" } else if gfx { "indirect_draw" } else { "indirect_dispatch" };
    let stage = if mesh_sig { RootStage::Mesh } else if gfx { RootStage::Graphics } else { RootStage::Compute };
    match bind_abi(this, list, pso, stage, call) {
        Ok(_) => {
            if mesh_sig {
                mesh_ran(MESH_NATIVE);
            }
            count(if gfx { DRAWS } else { DISPATCHES });
            let position = capture.then(|| super::markers::position(this as usize)).flatten();
            if gfx && !mesh_sig {
                capture_selected_depth(this, list, pso, pso, event, false);
            }
            execute(safe);
            mark(this, kind);
            capture_draw_gate_indirect(this, list, timed_capture.filter(|_| !capture), safe, args_title, count_title, pso, raw);
            if compute {
                note_dispatch_binding(this, title_pso, None);
                note_built_writes(this, title_pso);
            }
            if capture {
                let (writer, writer_at) = args_title.map_or(("unknown".into(), None), writer_of);
                let what = format!(
                    "{} on list {:#x}, at most {max} records at {} (count at {}); the arguments' writer: {writer}",
                    super::markers::pipeline(title_pso),
                    this as usize,
                    args_title.map_or("an address outside title memory".into(), |a| format!("{a:#x}")),
                    count_title.map_or("none".into(), |c| format!("{c:#x}")),
                );
                super::indirect::capture(list, safe, what, position, writer_at, true);
            }
        }
        Err(e) => {
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            if compute {
                note_skipped_writer(this, pso, title_pso);
            }
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

pub(super) fn indirect_world_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_INDIRECT_PRODUCER_TRACE").is_ok_and(|v| v == "1"))
        && super::present::draw_gate_capture(2).is_some()
}

pub(super) fn indirect_world_event(list: usize, what: std::fmt::Arguments<'_>) {
    static EVENTS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    if let Some((epoch, slot)) = super::present::draw_gate_capture(2) {
        let id = EVENTS[slot as usize].fetch_add(1, Ordering::Relaxed);
        if id < 30_000 {
            crate::xlog_info!("indirect lineage {slot}/#{id} epoch {epoch} list {list:#x} {what}");
        } else if id == 30_000 {
            crate::xlog_warn!("indirect lineage slot {slot} truncated after 30000 events");
        }
    }
}

pub(super) fn indirect_world_submit(queue: usize, lists: &[*mut c_void]) {
    if !indirect_world_trace() {
        return;
    }
    let q = super::list_x::queue_id(queue);
    let seq = super::list_x::queue_seq_now(queue);
    for (index, &list) in lists.iter().enumerate() {
        let marker = super::markers::position(list as usize);
        indirect_world_event(list as usize, format_args!("submit q{q} after sync seq {seq} batch list #{index} recording marker {marker:?}"));
    }
}

fn capture_writers_now() -> bool {
    draw_gate_trace() && super::present::writer_trace_active()
}

pub(super) fn dispatch_snapshot_enabled() -> bool {
    super::markers::enabled() || capture_writers_now()
}

#[allow(clippy::too_many_arguments)]
unsafe fn capture_draw_gate_indirect(this: *mut c_void, list: &ID3D12GraphicsCommandList, event: Option<(u64, u64, usize)>, safe: Option<(usize, u64, u64)>, args: Option<u64>, count: Option<u64>, pso: usize, raw: bool) {
    let (Some((frame, event, slot)), Some(safe)) = (event, safe) else { return };
    static SLOTS: Mutex<[u8; 16]> = Mutex::new([0; 16]);
    let mut slots = SLOTS.lock().unwrap();
    if slots[slot] >= 16 || !super::indirect::capture_wanted() {
        return;
    }
    slots[slot] += 1;
    drop(slots);
    let (writer, at) = args.map_or(("unknown".into(), None), writer_of);
    let what = format!("draw gate title epoch {frame}/{event} list {:#x} pipeline {pso:#x} args {args:x?} count {count:x?}; arguments' binding candidate {writer}", this as usize);
    let consumer = super::markers::position(this as usize).map(|(s, g, op)| (s, g, op.saturating_sub(1)));
    if indirect_world_trace() {
        crate::xlog_info!("indirect lineage consumer {frame}/{event} list {:#x} marker {consumer:?} args {args:x?} count {count:x?}", this as usize);
    }
    super::indirect::capture(list, Some(safe), what, consumer, at, raw);
}

/// XCL's own buffer for `fresh_indirect_arguments` one per device: COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<HashMap<usize, ID3D12Resource>>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t == D3D12_COMMAND_LIST_TYPE_DIRECT || t == D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let key = dev.as_raw() as usize;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    let buffers = g.get_or_insert_with(HashMap::new);
    if let Some(b) = buffers.get(&key) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    buffers.insert(key, b.clone());
    Some(b)
}

/// Why an ExecuteIndirect cannot run its untranslated pipeline `pso` (what `resolve_wait` gave
/// for the title's `title_pso`).
#[derive(Debug, PartialEq)]
enum IndirectSkip {
    /// dispatches: `resolve_wait` waited until the build ended, as for Dispatch, and it gave no
    /// translation (failed or refused) – skipped as Dispatch skips it, and what the pipeline
    /// would have written is unwritten (`note_skipped_writer`)
    Untranslated,
    /// draws: the pipeline is still being built after the draw backstop – skipped as a draw is
    Building,
}

/// None: the call runs (a PC pipeline, with the title's root signature, as for a draw).
fn indirect_skip(dispatches: bool, title_pso: usize) -> Option<IndirectSkip> {
    if dispatches {
        return Some(IndirectSkip::Untranslated);
    }
    super::pso_async::is_pending(title_pso).then_some(IndirectSkip::Building)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize {
            return false;
        }
        // (the vertices as a copy: the title can unmap them under the read, `memmap::title_bytes`)
        let Some(vertices) = crate::memmap::title_bytes(at, len) else { return false };
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| vertices.as_ptr().add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = vertices.as_ptr().add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    gpu_timing::op(this, Timed::Draw, || with(this, |s| s.pso), || list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance));
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    if put_off(this, super::list_x::Deferred::Draw(vertices, instances, first_vertex, first_instance)) {
        return;
    }
    let pso = with(this, |s| s.pso);
    let skipped = shader_skip(pso);
    draw_gate_call(this, pso, "DrawInstanced", format_args!("vertices {vertices} instances {instances} first_vertex {first_vertex} first_instance {first_instance}"), skipped);
    if skipped {
        count(DRAWS_SKIPPED);
        return;
    }
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    reach(this);
    let ia = super::list_x::ia_live::prepare(this, false, None);
    if let Some(list) = graphics_bound(this) {
        super::residency::note_list_untracked(this as usize);
        if !draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            if !ia.gathered() { super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances }); }
            gpu_timing::op(this, Timed::Draw, || with(this, |s| s.pso), || list.DrawInstanced(vertices, instances, first_vertex, first_instance));
        }
        mark(this, "draw");
    }
}

/// `XCL_GPU_MARKERS`: an op is about to be recorded on `this` – before its bindings and what
/// else XCL records for it (`markers::before_op`).
fn reach(this: *mut c_void) {
    if super::markers::enabled() {
        unsafe { super::markers::before_op(this) }
    }
}

/// `XCL_GPU_MARKERS`: the op just recorded on `this`, with its bound pipeline.
fn mark(this: *mut c_void, kind: &'static str) {
    if super::present::writer_trace_active() {
        trace_plane_write(this, kind);
    }
    if super::target_dump::recording() {
        note_target_dump(this, kind);
    }
    if super::submit_watch::on() {
        super::submit_watch::op(this as usize, kind, with(this, |s| s.pso));
    }
    if super::markers::enabled() {
        unsafe { super::markers::after_op(this, kind, with(this, |s| s.pso)) }
        // (a dispatch's arguments are kept with what they bound: `note_dispatch_snapshot`)
        if kind != "dispatch" && kind != "ExecuteIndirect(dispatch)" {
            if let Some((layout, args)) = with(this, |s| s.layout_gfx.clone().map(|l| (l, s.args_gfx.clone()))) {
                super::hang::note_graphics_op(this as usize, layout, args);
            }
        }
    }
}

/// `XCL_DUMP_TARGETS_AFTER_S`: the op just recorded on `list` – what it writes (render targets,
/// UAV textures), what it samples and at what size, its viewport and scissor.
fn note_target_dump(list: *mut c_void, kind: &'static str) {
    let compute = matches!(kind, "dispatch" | "ExecuteIndirect(dispatch)");
    let (pso, layout, args) = with(list, |s| {
        let (layout, args) = if compute { (&s.layout, &s.args) } else { (&s.layout_gfx, &s.args_gfx) };
        (s.pso, layout.clone(), args.clone())
    });
    let mut images = Vec::new();
    if let Some(layout) = layout.as_ref() {
        bound_resources(layout, &args, Some(&mut images));
    }
    let mut writes = if compute { Vec::new() } else { super::list_x::bound_color_resources(list) };
    let mut reads = Vec::new();
    for i in &images {
        let Some(res) = i.resource else { continue };
        if super::console_heap::native_uav_at(i.table.wrapping_add(32 * i.slot), i.srv) == Some(res) {
            if !writes.contains(&res) {
                writes.push(res);
            }
            continue;
        }
        reads.push(super::target_dump::Read {
            resource: res, width: i.view.width, height: i.view.height, format: i.view.format, swizzle: i.view.swizzle_mode,
            base_level: i.view.base_level, slot: format!("p{} table {:#x} slot {}", i.param, i.table, i.slot),
        });
    }
    let shaders = if compute {
        super::dispatch_timing::profile_shader(pso).map_or_else(String::new, |cs| format!("cs {cs:016x}"))
    } else {
        super::gpu_timing_stats::shaders(pso).map_or_else(String::new, |(vs, ps)| format!("vs {vs:016x} ps {ps:016x}"))
    };
    let (viewport, scissor) = if compute { (None, None) } else { super::list_x::viewport_scissor(list as usize) };
    unsafe { super::target_dump::after_pass(list, with(list, |s| s.allocator), &writes, || format!("{kind} of pipeline {pso:#x}{}{shaders}", if shaders.is_empty() { "" } else { " " })) };
    super::target_dump::record(list as usize, super::target_dump::Op { kind, pso, shaders, writes, reads, viewport, scissor });
}

fn trace_plane_write(list: *mut c_void, kind: &'static str) {
    let compute = matches!(kind, "dispatch" | "ExecuteIndirect(dispatch)");
    let deep = super::present::lineage_depth() > 2;
    let targets = if compute {
        Vec::new()
    } else {
        super::list_x::bound_color_resources(list).into_iter().filter(|&res| super::present::is_writer_target(res)).collect::<Vec<_>>()
    };
    if !compute && targets.is_empty() {
        return;
    }
    let (pso, layout, args) = with(list, |s| {
        let (layout, args) = if compute { (&s.layout, &s.args) } else { (&s.layout_gfx, &s.args_gfx) };
        (s.pso, layout.clone(), args.clone())
    });
    let mut images = Vec::new();
    let mut selected = Vec::new();
    let reads_present_plane = !compute && targets.iter().any(|&res| super::present::is_presented(res));
    if let Some(layout) = layout.as_ref() {
        if compute || reads_present_plane {
            bound_resources(layout, &args, Some(&mut images));
        }
        if !compute {
            selected = selected_present_inputs(pso, layout, &args, reads_present_plane);
        }
    }
    let mut targets = targets;
    if compute {
        for image in &images {
            if let Some(res) = image.resource {
                let addr = image.table.wrapping_add(32 * image.slot);
                let writable = deep && super::console_heap::native_uav_at(addr, image.srv) == Some(res);
                if (writable || super::present::is_known_writer_target(res)) && !targets.contains(&res) && targets.len() < 16 {
                    targets.push(res);
                }
            }
        }
    }
    if targets.is_empty() {
        return;
    }
    let reads = selected.iter().filter_map(|(r, _)| *r).collect::<Vec<_>>();
    let inputs = selected.into_iter().map(|(_, label)| label).chain(images.iter().take(12).map(|i| {
        let backing = i.resource.and_then(super::residency::describe_backing).unwrap_or_else(|| "untracked/unresolved".into());
        format!("p{} table {:#x}/{} idx {} {:?} {}x{} gfx10-fmt {} tile {} mirror {:x?} {backing}", i.param, i.table, i.slot, i.srv, i.view.ty, i.view.width, i.view.height, i.view.format, i.view.swizzle_mode, i.resource)
    })).collect::<Vec<_>>();
    let candidate_args = if deep && !compute && reads.is_empty() {
        layout.map(|layout| LineageArgs { layout, args: args.into() })
    } else {
        None
    };
    for res in targets {
        super::present::record_plane_write_with_reads(list as usize, res, if compute { "compute binding candidate" } else { kind }, pso, inputs.clone(), reads.clone(), candidate_args.clone());
    }
}

#[derive(Clone)]
pub(super) struct LineageArgs {
    layout: Arc<Layout>,
    args: Arc<[u8]>,
}

pub(super) fn lineage_candidates(binding: &LineageArgs) -> Vec<(usize, String)> {
    let mut images = Vec::new();
    bound_resources(&binding.layout, &binding.args, Some(&mut images));
    let mut seen = std::collections::HashSet::new();
    images.into_iter().filter_map(|i| {
        let resource = i.resource.filter(|&r| seen.insert(r))?;
        let addr = i.table.wrapping_add(32 * i.slot);
        let native = super::console_heap::native_srv_at(addr, i.srv);
        Some((resource, format!("candidate descriptor at {addr:#x}, idx {}, {:?}, {}x{}, gfx10-fmt {}, tile {}, mip {}; PC SRV (owner, format, dimension, resource format, layout, flags) {native:x?}", i.srv, i.view.ty, i.view.width, i.view.height, i.view.format, i.view.swizzle_mode, i.view.base_level)))
    }).take(64).collect()
}

fn parse_input_selector(selector: &str) -> Option<(usize, u32)> {
    let (reg, offset) = selector.trim().strip_prefix('s')?.split_once('+')?;
    let reg = reg.parse::<usize>().ok()?;
    let offset = match offset.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None => offset.parse().ok()?,
    };
    (reg < 32 && offset % 32 == 0).then_some((reg, offset))
}

fn note_image_probes(list: usize, pso: usize, io: &[u32]) {
    type Selection = Vec<(u64, Vec<(usize, u32)>)>;
    static PROBES: std::sync::OnceLock<Selection> = std::sync::OnceLock::new();
    let probes = PROBES.get_or_init(|| {
        std::env::var("XCL_GPU_IMAGE_PROBES").unwrap_or_default().split(';').filter_map(|group| {
            let (hash, fields) = group.trim().split_once('=')?;
            let hash = u64::from_str_radix(hash, 16).ok()?;
            let selectors = fields.split(',').filter_map(parse_input_selector).take(16).collect::<Vec<_>>();
            (!selectors.is_empty()).then_some((hash, selectors))
        }).take(16).collect()
    });
    if probes.is_empty() {
        return;
    }
    let Some((epoch, capture)) = super::present::draw_gate_capture(2) else { return };
    let Some(hash) = super::dispatch_timing::profile_shader(pso) else { return };
    let Some((probe, (_, selectors))) = probes.iter().enumerate().find(|(_, (target, _))| *target == hash) else { return };
    let frame = super::present::presents();
    static BUDGET: Mutex<[[u8; 16]; 16]> = Mutex::new([[0; 16]; 16]);
    let mut quota = BUDGET.lock().unwrap();
    let count = &mut quota[capture as usize][probe];
    if *count >= 8 {
        return;
    }
    *count += 1;
    drop(quota);
    for &(reg, offset) in selectors {
        let slot = io.get(reg).filter(|&&r| r != 0).and_then(|r| r.checked_add(offset));
        let word = slot.and_then(super::console_heap::words_at);
        let desc = word.as_ref().and_then(crate::gpu::desc::image);
        let (owner, srv, uav) = slot.zip(word).filter(|_| desc.is_some()).map_or((None, None, None), |(at, w)| {
            (super::console_heap::image_resource_at(at, w[0]), super::console_heap::native_srv_at(at, w[0]),
             super::console_heap::native_uav_details_at(at, w[0]))
        });
        let uav = uav.map(|v| (v.resource, v.view, v.backing, v.volume));
        let backing = owner.and_then(|r| super::real::title_address(r).or_else(|| super::title_textures::address(r)));
        let tiles = owner.and_then(super::vram::tile_backing);
        crate::xlog_info!("image probe: frame {frame} epoch {epoch} list {list:#x} CS {hash:016x} s{reg}+{offset:#x} slot {slot:x?} T# {desc:?} title {backing:x?} tiles {tiles:?} owner {owner:x?} native SRV {srv:x?} native UAV {uav:x?}");
    }
}

type ScalarSelector = (usize, u32, u32, u32);

fn parse_scalar_selector(selector: &str) -> Option<ScalarSelector> {
    let (descriptor, rest) = selector.split_once(':')?;
    let (offset, count) = rest.split_once(':')?;
    let (reg, descriptor) = parse_input_selector(descriptor)?;
    let offset = offset.strip_prefix("0x").map_or_else(|| offset.parse().ok(), |s| u32::from_str_radix(s, 16).ok())?;
    let count: u32 = count.parse().ok()?;
    (offset % 4 == 0 && (1..=8).contains(&count)).then_some((reg, descriptor, offset, count))
}

fn note_scalar_probes(list: usize, pso: usize, gfx: bool, io: &[u32]) {
    type Selection = Option<(u64, Vec<ScalarSelector>)>;
    static PROBES: std::sync::OnceLock<[Selection; 2]> = std::sync::OnceLock::new();
    let probes = PROBES.get_or_init(|| {
        let selection = |shader, descriptors| {
            let input = std::env::var(shader).ok()?;
            let hash = u64::from_str_radix(input.trim().trim_start_matches("0x"), 16).ok()?;
            let values = std::env::var(descriptors).ok()?.split(',').filter_map(parse_scalar_selector).take(8).collect::<Vec<_>>();
            (!values.is_empty()).then_some((hash, values))
        };
        [selection("XCL_PRESENT_SCALAR_CS", "XCL_PRESENT_CS_SCALARS"), selection("XCL_PRESENT_SCALAR_PS", "XCL_PRESENT_PS_SCALARS")]
    });
    let Some((shader, selectors)) = &probes[usize::from(gfx)] else { return };
    if !super::present::writer_trace_active() {
        return;
    }
    let active = if gfx { super::gpu_timing_stats::shaders(pso).map(|(_, ps)| ps) } else { super::dispatch_timing::profile_shader(pso) };
    if active != Some(*shader) {
        return;
    }
    let probes = selectors.iter().map(|&(reg, descriptor, offset, count)| {
        let addr = io.get(reg).filter(|&&v| v != 0).and_then(|v| v.checked_add(descriptor));
        let words = addr.and_then(super::console_heap::words_at);
        let va = words.and_then(|w| {
            let first = buffer_dword_address(&w, offset)?;
            let last = buffer_dword_address(&w, offset.checked_add(4 * (count - 1))?)?;
            (first.is_multiple_of(4) && last.checked_sub(first) == Some(4 * (count as u64 - 1))).then_some(first)
        });
        super::present::ScalarProbe { selector: (reg, descriptor, offset), words: count, va }
    }).collect();
    super::present::record_scalar_probes(list, *shader, probes);
}

#[test]
fn scalar_probe_checks_descriptor_and_dword_alignment_and_bounds() {
    assert_eq!(parse_scalar_selector("s8+0x80:0x10:4"), Some((8, 128, 16, 4)));
    for bad in ["s8+0x1:0:4", "s8+0x80:1:4", "s8+0x80:0:9", "s8+0x80:0:0"] {
        assert!(parse_scalar_selector(bad).is_none());
    }
}

fn selected_present_inputs(pso: usize, layout: &Layout, args: &[u8], present_plane: bool) -> Vec<(Option<usize>, String)> {
    type InputSelection = (Vec<(usize, u32)>, Option<u64>);
    static SELECTORS: std::sync::OnceLock<[InputSelection; 2]> = std::sync::OnceLock::new();
    let selections = SELECTORS.get_or_init(|| {
        let selection = |desc, ps| {
            let selectors = std::env::var(desc).unwrap_or_default()
                .split(',').filter_map(parse_input_selector).take(24).collect();
            let shader = std::env::var(ps).ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok());
            (selectors, shader)
        };
        [selection("XCL_PRESENT_INPUT_DESCS", "XCL_PRESENT_INPUT_PS"),
         selection("XCL_PRESENT_UPSTREAM_DESCS", "XCL_PRESENT_UPSTREAM_PS")]
    });
    let (selectors, shader) = &selections[usize::from(!present_plane)];
    if selectors.is_empty() || (!present_plane && shader.is_none())
        || shader.is_some_and(|wanted| super::gpu_timing_stats::shaders(pso).map(|(_, ps)| ps) != Some(wanted))
    {
        return Vec::new();
    }
    let io = console_list::user_data_gfx(layout, args, 0);
    selectors.iter().map(|&(reg, offset)| {
        let label = format!("selected s{reg}+{offset:#x}");
        let Some(base) = io.get(reg).copied().filter(|&v| v != 0) else { return (None, format!("{label}: unset graphics SGPR")); };
        let Some(at) = base.checked_add(offset) else { return (None, format!("{label}: address overflow")); };
        let Some(words) = super::console_heap::words_at(at) else { return (None, format!("{label}: {at:#x} is outside a tracked console heap")); };
        let Some(view) = crate::gpu::desc::image(&words) else { return (None, format!("{label}: {at:#x} holds a null/nonimage descriptor")); };
        let resource = super::console_heap::image_resource_at(at, words[0]);
        let native = super::console_heap::native_srv_at(at, words[0]);
        let (title, tiles, backing) = resource.map_or((None, None, None), |r| {
            super::present::watch_plane_input(r);
            (super::real::title_address(r).or_else(|| super::title_textures::address(r)), super::vram::tile_backing(r), super::residency::describe_backing(r))
        });
        let dxgi = crate::gpu::desc::dxgi_format(view.format).map(|(fmt, _)| fmt.0);
        (resource, format!("{label}: {at:#x} idx {} {:?} {}x{} gfx10-fmt {} mapped DXGI {dxgi:?} tile {} mip {} array {} mirror {resource:x?} title {title:x?} backing {backing:?} tiles {tiles:x?} PC SRV (owner, format, dimension, resource format, layout, flags) {native:x?}", words[0], view.ty, view.width, view.height, view.format, view.swizzle_mode, view.base_level, view.base_array))
    }).collect()
}

#[test]
fn present_input_selector_checks_register_and_descriptor_alignment() {
    assert_eq!(parse_input_selector("s6+0x20"), Some((6, 32)));
    assert_eq!(parse_input_selector("s5+256"), Some((5, 256)));
    assert_eq!(parse_input_selector("s32+0x20"), None);
    assert_eq!(parse_input_selector("s8+0x1"), None);
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    if put_off(this, super::list_x::Deferred::DrawIndexed(indices, instances, first_index, base_vertex, first_instance)) {
        return;
    }
    let ia = super::list_x::ia_live::prepare(this, true, None);
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex, first_instance) {
        if !ia.gathered() { super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances }); }
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) -> Option<ID3D12GraphicsCommandList> {
    let pso = with(this, |s| s.pso);
    let skipped = shader_skip(pso);
    draw_gate_call(this, pso, "DrawIndexedInstanced", format_args!("indices {indices} instances {instances} first_index {first_index} base_vertex {base_vertex} first_instance {first_instance}"), skipped);
    if skipped {
        count(DRAWS_SKIPPED);
        return None;
    }
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    reach(this);
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    super::residency::note_list_untracked(list.as_raw() as usize);
    let raw = list.as_raw();
    gpu_timing::op(raw, Timed::Draw, || with(raw, |s| s.pso), || list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance));
    mark(list.as_raw(), "indexed draw");
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a pipeline still being built in the background is waited for (its stand-in draws nothing)
    let real = super::pso_async::waiting_if(in_bundle(this), || super::pso_async::resolve_wait(pso));
    if shader_skip(real) {
        count(DRAWS_SKIPPED);
        return None;
    }
    if skip_failed_draws() && super::real::is_standin_pso(real) {
        note_failed_skip(pso, real, "draw");
        count(DRAWS_SKIPPED);
        return None;
    }
    let two_pass = super::real::two_pass_of(real);
    if two_pass.is_some() || super::real::is_native_mesh(real) {
        // the PC draw with a mesh pipeline is invalid; mesh pipelines draw through DispatchMesh
        mesh_skipped(this, real, "a draw", if two_pass.is_some_and(|tp| tp.kind != "MS") { GS_TWO_PASS } else { "its pipeline is a mesh pipeline" });
        return None;
    }
    if !super::real::is_translated_graphics(real) {
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve_wait(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        return Some(list);
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    let call = if DRAW_ARGS.with(|d| d.get().0) { "draw_indexed" } else { "draw" };
    match bind_abi(this, &list, pso, RootStage::Graphics, call) {
        Ok(_) => {
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = (va != 0).then(|| crate::memmap::title_words(va, buffer_words)).flatten().unwrap_or_default();
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    if with(this, |s| s.rs == rs as usize && s.layout.is_some()) {
        return;
    }
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
        s.rs = rs as usize;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    table_bound(this, false, i, handle as u32);
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    with(this, |s| s.pso = pso as usize);
    super::real::bind_pipeline(this, pso);
}

/// What each dispatch skipped for want of a translated pipeline may have written, by its
/// pipeline: its root descriptors (console addresses, 16 MB assumed) and the buffer views of its
/// descriptor tables (console address, size, PC resource, slot) – checked against the arguments
/// of indirect calls after it (`note_indirect`).
static SKIPPED_WRITERS: Mutex<Vec<(usize, Vec<(u64, u64, usize, String)>)>> = Mutex::new(Vec::new());

/// A dispatch of `pso` (the title's `title_pso`) on `this` was skipped: the memory its UAVs
/// cover is kept, for the log and as unwritten (`indirect::skipped_producer`).
fn note_skipped_writer(this: *mut c_void, pso: usize, title_pso: usize) {
    let trace = indirect_world_trace();
    let (bound, writes) = with(this, |s| {
        s.layout.as_ref().map_or_else(Default::default, |l| (if trace { bound_writes(l, &s.args) } else { Vec::new() }, uav_writes(l, &s.args, console_descriptor)))
    });
    if trace {
        let cs = super::dispatch_timing::profile_shader(pso);
        indirect_world_event(this as usize, format_args!("skipped dispatch pso {pso:#x} cs {cs:016x?} buffer bindings {:x?} UAV writes {:x?}", &bound[..bound.len().min(24)], &writes[..writes.len().min(24)]));
    }
    record_skipped_writes(pso, title_pso, writes);
}

/// A dispatch of the title's `title_pso` ran on `this` with a translation: what that pipeline
/// writes is written again, and so is what its UAVs cover, whichever skip kept it.
fn note_built_writes(this: *mut c_void, title_pso: usize) {
    super::indirect::produced(title_pso);
    if super::indirect::nothing_unwritten() {
        return;
    }
    let writes = with(this, |s| s.layout.as_ref().map_or_else(Vec::new, |l| uav_writes(l, &s.args, console_descriptor)));
    let ranges = write_ranges(&writes);
    super::indirect::bound(title_pso, &ranges);
    super::indirect::written(&ranges);
}

/// The console words of the descriptor at `addr` and the PC resource behind it (0: none known).
fn console_descriptor(addr: u32) -> Option<([u32; 8], usize)> {
    let words = super::console_heap::words_at(addr)?;
    Some((words, crate::gpu::uma::get().and_then(|u| super::console_heap::resource_at(u.base() + addr as usize)).unwrap_or(0)))
}

/// The buffer views a call bound by `layout` and `args` can write: its root UAVs (16 MB assumed,
/// as `bound_writes`) and the buffer descriptors in its tables' UAV ranges, the first 64 slots
/// (console address, bytes, PC resource, where). Texture UAVs hold no indirect arguments.
fn uav_writes(layout: &Layout, args: &[u8], descriptor: impl Fn(u32) -> Option<([u32; 8], usize)>) -> Vec<(u64, u64, usize, String)> {
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut out = Vec::new();
    let mut slots = 64u32;
    for binding in &layout.uavs {
        match *binding {
            UavBinding::Root { param, at } => {
                let va = word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32;
                if va != 0 {
                    out.push((va, 16 << 20, 0, format!("p{param} root UAV")));
                }
            }
            UavBinding::Table { param, at, first, count } => {
                let table = word(at as usize);
                let n = count.min(slots);
                slots -= n;
                if table == 0 {
                    continue;
                }
                for slot in first..first.saturating_add(n) {
                    let Some((w, resource)) = slot.checked_mul(32).and_then(|off| table.checked_add(off)).and_then(&descriptor) else { break };
                    let (base, stride, records) = (w[0] as u64 | ((w[1] & 0xffff) as u64) << 32, ((w[1] >> 16) & 0x3fff) as u64, w[2] as u64);
                    if w[3] >> 30 != 0 || base == 0 || records == 0 {
                        continue;
                    }
                    out.push((base, records * stride.max(1), resource, format!("p{param} UAV table {table:#x} slot {slot}")));
                }
            }
        }
    }
    out
}

/// The console ranges (start, bytes) `writes` keep as unwritten or write: a root view's assumed
/// 16 MB as 64 KB.
fn write_ranges(writes: &[(u64, u64, usize, String)]) -> Vec<(u64, u64)> {
    writes.iter().map(|&(base, size, _, _)| (base, if size == 16 << 20 { 64 << 10 } else { size })).collect()
}

pub(super) fn record_skipped_writes(pso: usize, title_pso: usize, writes: Vec<(u64, u64, usize, String)>) {
    super::indirect::skipped_producer(title_pso, &write_ranges(&writes));
    let mut w = SKIPPED_WRITERS.lock().unwrap();
    if w.len() < 256 && !w.iter().any(|e| e.0 == pso) {
        w.push((pso, writes));
    }
}

static PENDING_COMPUTE_SKIPS: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

fn note_pending_compute_skip(this: *mut c_void, title_pso: usize, call: &str, skip: super::pso_async::PendingComputeSkip) {
    if matches!(skip, super::pso_async::PendingComputeSkip::ImageOnly) {
        count(DISPATCHES_SKIPPED);
        return;
    }
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    note_pending_compute_skip_with_args(title_pso, call, skip, layout.as_deref(), &args);
}

pub(super) fn note_pending_compute_skip_with_args(title_pso: usize, call: &str, skip: super::pso_async::PendingComputeSkip, layout: Option<&Layout>, args: &[u8]) {
    count(DISPATCHES_SKIPPED);
    let super::pso_async::PendingComputeSkip::Deadline { shader, waited_ms } = skip else { return };
    let writes = layout.map_or_else(Vec::new, |l| uav_writes(l, args, console_descriptor));
    record_skipped_writes(title_pso, title_pso, writes);
    let n = {
        let mut counts = PENDING_COMPUTE_SKIPS.lock().unwrap();
        let n = counts.get_or_insert_with(HashMap::new).entry(shader).or_default();
        *n += 1;
        *n
    };
    if n == 1 {
        let uavs = layout.map_or_else(Vec::new, |l| describe_uavs(l, args, super::console_heap::words_at));
        crate::xlog_warn!(
            "D3D12.X: {call} skipped while shader {shader:016x} was pending after {} ms (pipeline {title_pso:#x}); UAV bindings: {}",
            waited_ms,
            if uavs.is_empty() { "none captured (bindless and indirect root changes may not be visible)".to_string() } else { uavs.join("; ") }
        );
    } else if n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: shader {shader:016x}: {n} compute calls skipped while its pipeline was pending");
    }
}

fn describe_uavs(layout: &Layout, args: &[u8], words_at: impl Fn(u32) -> Option<[u32; 8]>) -> Vec<String> {
    let word = |at| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut out = Vec::new();
    const MAX: usize = 32;
    for binding in &layout.uavs {
        if out.len() >= MAX {
            break;
        }
        match *binding {
            UavBinding::Root { param, at } => {
                let at = at as usize;
                let va = word(at) as u64 | (word(at + 4) as u64) << 32;
                out.push(format!("p{param} root UAV {va:#x}"));
            }
            UavBinding::Table { param, at, first, count } => {
                let table = word(at as usize);
                for slot in first..first.saturating_add(count.min((MAX - out.len()) as u32)) {
                    let words = slot.checked_mul(32).and_then(|off| table.checked_add(off)).and_then(&words_at);
                    let words = words.map_or_else(|| "descriptor unavailable".to_string(), |w| format!("{w:08x?}"));
                    out.push(format!("p{param} UAV table {table:#x} slot {slot}: {words}"));
                }
            }
        }
    }
    if out.len() == MAX {
        out.push("capture limited to 32 declared UAV slots".into());
    }
    out
}

/// Whether indirect calls are captured now (`XCL_GPU_MARKERS`): fewer than the most so far, and
/// console code has had a dispatch, draw or indirect call skipped, or a dispatch is waiting for
/// its compute pipeline.
fn capture_now() -> bool {
    let skipped = COUNTS[DISPATCHES_SKIPPED].load(Ordering::Relaxed) + COUNTS[DRAWS_SKIPPED].load(Ordering::Relaxed) > 0;
    (skipped || super::pso_async::compute_bound_pending()) && super::indirect::capture_wanted()
}

/// mesh calls of console code (DispatchMesh, mesh ExecuteIndirect): run as a host mesh shader,
/// run in two passes, skipped
static MESH_CALLS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
const MESH_NATIVE: usize = 0;
const MESH_TWO_PASS: usize = 1;
const MESH_SKIPPED: usize = 2;

/// why a call with a console geometry-shader pipeline (`real::TwoPassPso::kind` "GS") does not
/// run: its two passes (the vertex and geometry programs as compute, then the pass-through)
/// are made at pipeline creation, and no draw runs them yet
const GS_TWO_PASS: &str = "its pipeline has a console geometry shader in two passes, which draws do not run yet";

fn mesh_ran(how: usize) {
    MESH_CALLS[how].fetch_add(1, Ordering::Relaxed);
}

/// A draw, mesh dispatch or indirect call (`what`) on the list `this` with the pipeline `pso` does
/// not run: `why`.
fn mesh_skipped(this: *mut c_void, pso: usize, what: &str, why: &str) {
    if super::markers::enabled() {
        // what it was bound to may never be written: kept for the snapshot (`describe_dispatch`)
        let bound = with(this, |s| s.layout_gfx.as_ref().map_or_else(Vec::new, |l| bound_writes(l, &s.args_gfx)));
        let mut r = REFUSED_BOUND.lock().unwrap();
        for (base, size, ..) in bound {
            if r.len() < 8192 && !r.iter().any(|&(b, _, p)| b == base && p == pso) {
                r.push((base, size, pso));
            }
        }
    }
    let n = MESH_CALLS[MESH_SKIPPED].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} with mesh work skipped (pipeline {pso:#x}): {why} (#{n})");
    }
}

/// The mesh pipelines made so far and the mesh calls run, native (as host mesh shaders) and
/// two-pass: a line of its own every 30 s, in the GPU snapshot and at the end.
pub fn mesh_counts() -> String {
    let c = |k: usize| MESH_CALLS[k].load(Ordering::Relaxed);
    let (native, two_pass) = super::real::mesh_pipeline_counts();
    format!("mesh pipelines: {native} native, {two_pass} two-pass; mesh calls: {} native, {} two-pass, {} skipped", c(MESH_NATIVE), c(MESH_TWO_PASS), c(MESH_SKIPPED))
}

/// The indirect calls so far by type and how mesh calls ran, for the 30 s line.
pub fn indirect_calls() -> String {
    let c = |k: usize| MESH_CALLS[k].load(Ordering::Relaxed);
    let (native, two_pass) = super::real::mesh_pipeline_counts();
    let trips: String = trips(TRIP_PIPELINES_TOLD).iter().filter(|t| t.starts_with("guard tripwire")).map(|t| format!("; {t}")).collect();
    let pending = PENDING_COMPUTE_SKIPS.lock().unwrap().as_ref().map_or_else(String::new, |counts| {
        let mut top: Vec<_> = counts.iter().collect();
        top.sort_unstable_by_key(|(&shader, &count)| (std::cmp::Reverse(count), shader));
        let top = top.into_iter().take(8).map(|(&shader, &count)| format!("{shader:016x}={count}")).collect::<Vec<_>>().join(", ");
        format!("; {} compute calls skipped while pending across {} shaders (highest counts: {top})", counts.values().sum::<u64>(), counts.len())
    });
    format!(
        "ExecuteIndirect {}; mesh pipelines {native} as host mesh shaders, {two_pass} in two passes; mesh calls {} as host mesh shaders, {} in two passes, {} skipped{trips}{pending}",
        super::indirect::calls_summary(),
        c(MESH_NATIVE),
        c(MESH_TWO_PASS),
        c(MESH_SKIPPED)
    )
}

/// PC ID3D12GraphicsCommandList6::DispatchMesh of the PC list behind `this`.
unsafe fn pc_dispatch_mesh(this: *mut c_void) -> Option<unsafe extern "system" fn(*mut c_void, u32, u32, u32)> {
    let slot = super::layout::slot(super::layout::find("ID3D12GraphicsCommandList6")?, "DispatchMesh")?;
    let f = super::real::orig_fn(this, slot);
    (f != 0).then(|| std::mem::transmute(f))
}

/// DispatchMesh(x, y, z) – #127 of the older console layout, #129 of the newer: the bound mesh
/// pipeline's console program runs as a host mesh shader (one PC DispatchMesh with XCL's ABI
/// bound, as a draw with a translated pipeline), or in two passes (`two_pass_direct`). A console
/// program numbers its groups on one axis (it finds its group in one register), so y and z fold
/// into x in both forms.
pub(super) unsafe extern "system" fn dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    super::depth_uav::note_op(this);
    let title_pso = with(this, |s| s.pso);
    let pso = super::pso_async::waiting_if(in_bundle(this), || super::pso_async::resolve_wait(title_pso));
    let failed = skip_failed_draws() && super::real::is_standin_pso(pso);
    let skipped = shader_skip(pso) || failed;
    draw_gate_call(this, title_pso, "DispatchMesh", format_args!("groups {x} {y} {z}"), skipped);
    if failed {
        note_failed_skip(title_pso, pso, "DispatchMesh");
    }
    if skipped {
        count(DRAWS_SKIPPED);
        return;
    }
    note_named_use(title_pso, pso, "mesh dispatch");
    let (Some(list), Some(pc)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), pc_dispatch_mesh(this)) else {
        mesh_skipped(this, pso, "a DispatchMesh", "the PC command list has no DispatchMesh");
        return;
    };
    reach(this);
    if let Some(tp) = super::real::two_pass_of(pso) {
        if tp.kind != "MS" {
            count(DRAWS_SKIPPED);
            mesh_skipped(this, pso, "a DispatchMesh", GS_TWO_PASS);
            return;
        }
        match two_pass_direct(this, list, pso, &tp, x, y, z, pc) {
            Ok(()) => {
                count(DRAWS);
                mesh_ran(MESH_TWO_PASS);
                mark(this, "mesh dispatch in two passes");
            }
            Err(e) => {
                count(DRAWS_SKIPPED);
                mesh_skipped(this, pso, "a DispatchMesh", &e);
            }
        }
        return;
    }
    if !super::real::is_translated_graphics(pso) {
        // a PC pipeline (also a stand-in): the title's root signature, as for a draw
        // (`graphics_bound`)
        count(DRAWS_PC);
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        super::residency::note_list_untracked(this as usize);
        gpu_timing::op(this, Timed::Draw, || title_pso, || pc(this, x, y, z));
        mark(this, "mesh dispatch with a PC pipeline");
        return;
    }
    if !super::real::is_native_mesh(pso) {
        count(DRAWS_SKIPPED);
        mesh_skipped(this, pso, "a DispatchMesh", "its pipeline has no mesh shader");
        return;
    }
    let groups = (x as u64).saturating_mul(y as u64).saturating_mul(z as u64);
    if groups > 65535 {
        count(DRAWS_SKIPPED);
        mesh_skipped(this, pso, "a DispatchMesh", &format!("{groups} groups are over what one host DispatchMesh takes on an axis"));
        return;
    }
    match bind_abi(this, list, pso, RootStage::Mesh, "mesh_dispatch") {
        Ok(_) => {
            count(DRAWS);
            mesh_ran(MESH_NATIVE);
            super::residency::note_list_untracked(this as usize);
            if groups != 0 {
                super::list_x::note_lineage_depth_draw(this, "mesh_dispatch");
            }
            gpu_timing::op(this, Timed::Draw, || title_pso, || pc(this, groups as u32, 1, 1));
            mark(this, "mesh dispatch");
        }
        Err(e) => {
            count(DRAWS_SKIPPED);
            mesh_skipped(this, pso, "a DispatchMesh", &e);
        }
    }
}

/// DispatchMeshX(x, flags, indirect arguments, payload) – #98 of the older console layout, #100
/// of the newer: the direct form is DispatchMesh(x, 1, 1). With indirect arguments the call is
/// skipped: the layout of what the console reads there is not known (no title seen uses the
/// form; its indirect mesh work goes through ExecuteIndirect). A payload is not passed on –
/// a console mesh program without an amplification shader declares none.
pub(super) unsafe extern "system" fn dispatch_mesh_x(this: *mut c_void, x: u32, _flags: u64, indirect: u64, payload: u64) {
    super::depth_uav::note_op(this);
    if indirect != 0 {
        draw_gate_call(this, with(this, |s| s.pso), "DispatchMeshX", format_args!("groups {x} indirect {indirect:#x} (unsupported)"), false);
        count(DRAWS_SKIPPED);
        mesh_skipped(this, with(this, |s| s.pso), "a DispatchMeshX", "it takes its group count from memory");
        return;
    }
    if payload != 0 {
        static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !TOLD.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("D3D12.X: a DispatchMeshX passes a payload, which the mesh program does not get");
        }
    }
    dispatch_mesh(this, x, 1, 1);
}

/// record bytes one mesh dispatch in two passes may take from unified memory
const TWO_PASS_BYTES: u64 = 256 << 20;
/// XCL's record memory for mesh calls in two passes (`two_pass_scratch`): 32 MiB, or
/// `XCL_TWO_PASS_SCRATCH_MB` (32 to 1024). An indirect call's commands share it, each up to the
/// thread groups its part holds (`two_pass_indirect_plan`); the sanitiser cuts the rest.
fn two_pass_scratch_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let mb = std::env::var("XCL_TWO_PASS_SCRATCH_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(32, |mb| mb.clamp(32, 1024));
        if mb != 32 {
            crate::xlog_info!("D3D12.X: record memory for mesh calls in two passes: {mb} MiB (XCL_TWO_PASS_SCRATCH_MB)");
        }
        mb << 20
    })
}
/// commands one indirect mesh call in two passes may hold
const TWO_PASS_INDIRECT_COMMANDS: u32 = 64;

/// The GPU address of XCL's record memory for mesh calls in two passes: one buffer of
/// `two_pass_scratch_bytes()` every such call writes (its prepass) and reads (its mesh call) from
/// the start. The GPU runs a queue's calls in their order, and a prepass leaves no word of an
/// earlier call for its mesh call to read (it empties its thread's primitive and its group's
/// counts first, `emit`'s mesh entry), so the calls need no memory of their own – an indirect
/// call's group counts are the GPU's, and thousands of such calls a frame could not each take
/// their worst case. (Mesh calls of two queues at once would share it; console titles record
/// their graphics work for one.)
unsafe fn two_pass_scratch() -> Result<u64, String> {
    static MADE: Mutex<Option<(usize, u64)>> = Mutex::new(None);
    let mut g = MADE.lock().unwrap();
    if let Some((_, va)) = *g {
        return Ok(va);
    }
    let raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: two_pass_scratch_bytes(),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    super::title_textures::own_work(|| dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut r)).map_err(|e| format!("no record memory for mesh calls in two passes: {e}"))?;
    let r = r.ok_or("no record memory for mesh calls in two passes")?;
    let va = r.GetGPUVirtualAddress();
    *g = Some((r.into_raw() as usize, va));
    Ok(va)
}

/// Record memory for a mesh dispatch of `bytes` of records: XCL's (`two_pass_scratch`), or for
/// more than that holds the list's unified memory.
unsafe fn two_pass_records(this: *mut c_void, bytes: u64) -> Result<u64, String> {
    if bytes <= two_pass_scratch_bytes() {
        return two_pass_scratch();
    }
    let p = allocate_upload_memory(this, bytes as usize, 256);
    if p.is_null() {
        return Err(format!("no unified memory for {} MB of mesh records", bytes >> 20));
    }
    super::real::uma_va_of(p as usize).ok_or_else(|| "the mesh records are outside the memory windows".into())
}

fn uav_barrier_all() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }
}

/// A mesh dispatch with a two-pass pipeline (`real::TwoPassPso`): the console program as a
/// compute pass writes one record per group, then the title's mesh pipeline (the pass-through)
/// draws the records. The console numbers mesh groups in x only, so y and z fold into x.
unsafe fn two_pass_direct(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, tp: &super::real::TwoPassPso, x: u32, y: u32, z: u32, pc: unsafe extern "system" fn(*mut c_void, u32, u32, u32)) -> Result<(), String> {
    let groups = (x as u64).saturating_mul(y as u64).saturating_mul(z as u64);
    if groups == 0 {
        return Ok(());
    }
    let split = tp.mesh_groups_per_record.max(1) as u64;
    let bytes = groups.saturating_mul(tp.record.record_dwords() as u64 * 4);
    if groups > 65535 || groups * split > 65535 || bytes > TWO_PASS_BYTES {
        return Err(format!("{groups} groups in two passes are over what one prepass holds ({} MB of records)", bytes >> 20));
    }
    let (real, [io, _], layout, args, _) = abi_views(this, pso, RootStage::Mesh)?;
    let res = two_pass_records(this, bytes)?;
    bind_views(this, list, tp.compute_rs, tp.compute, [io, res], false)?;
    note_lineage_uavs(this, (tp.compute, real), "mesh_two_pass_prepass", layout.as_deref(), &args);
    super::residency::note_list_untracked(this as usize);
    gpu_timing::op_grid(this, Timed::MeshCompute, || tp.compute, [groups as u32, 1, 1], || list.Dispatch(groups as u32, 1, 1));
    list.ResourceBarrier(&[uav_barrier_all()]);
    bind_views(this, list, tp.mesh_rs, real, [io, res], true)?;
    note_lineage_uavs(this, (real, real), "mesh_two_pass_draw", layout.as_deref(), &args);
    super::list_x::note_lineage_depth_draw(this, "mesh_two_pass_draw");
    gpu_timing::op_grid(this, Timed::MeshDraw, || pso, [(groups * split) as u32, 1, 1], || pc(this, (groups * split) as u32, 1, 1));
    Ok(())
}

/// How an indirect mesh call of `max` commands runs in two passes: (thread groups a command may
/// run – its share of the call's record memory –, mesh groups per record).
fn two_pass_indirect_plan(tp: &super::real::TwoPassPso, max: u32) -> (u32, u32) {
    let split = tp.mesh_groups_per_record.max(1);
    let record = tp.record.record_dwords() as u64 * 4;
    let groups = two_pass_scratch_bytes() / max.max(1) as u64 / record.max(1);
    ((groups as u32).clamp(1, 65535 / split), split)
}

/// Indirect mesh calls in two passes: (calls checked, commands over their room, thread groups
/// past it, the most groups one command asked for).
static TWO_PASS_CUTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Counts the commands of an indirect mesh call in two passes whose thread groups are more than
/// `room`, the most the sanitiser lets one run (it cuts the rest). The counts are read from the
/// title's memory as the call is recorded: a count the GPU writes is the one its last run left
/// there, usually the previous frame's, so this is an estimate of the cut.
fn note_two_pass_cut(args: u64, count: Option<u64>, stride: u64, max: u32, room: u32) {
    let commands = match count {
        Some(at) => match crate::memmap::title_words(at as usize, 1) {
            Some(w) => w[0].min(max),
            None => return,
        },
        None => max,
    };
    let (mut cut, mut past) = (0u64, 0u64);
    for k in 0..commands as u64 {
        let Some(w) = crate::memmap::title_words((args + k * stride) as usize, 3) else { return };
        let groups = w[0] as u64 * w[1] as u64 * w[2] as u64;
        TWO_PASS_CUTS[3].fetch_max(groups, Ordering::Relaxed);
        if groups > room as u64 {
            cut += 1;
            past += groups - room as u64;
        }
    }
    let calls = TWO_PASS_CUTS[0].fetch_add(1, Ordering::Relaxed) + 1;
    let before = TWO_PASS_CUTS[1].fetch_add(cut, Ordering::Relaxed);
    let past = TWO_PASS_CUTS[2].fetch_add(past, Ordering::Relaxed) + past;
    let after = before + cut;
    let crossed = cut != 0 && (before == 0 || after.ilog2() > before.ilog2());
    if crossed || (calls.is_power_of_two() && calls >= 64) {
        crate::xlog_info!(
            "D3D12.X: indirect mesh calls in two passes: {after} commands over their room of {room} thread groups so far ({past} groups cut; the most one command asked for {}; {calls} calls checked, counts as the title's memory held them when recorded)",
            TWO_PASS_CUTS[3].load(Ordering::Relaxed)
        );
    }
}

/// An indirect mesh call with a two-pass pipeline. The group counts are the GPU's, so the record
/// memory cannot follow them: each command gets room for `plan.0` groups (the sanitiser clamps
/// its count to that) at its own place – the res view moves – and runs as its own prepass
/// (XCL's dispatch signature over the sanitised copy) and mesh call (the copy for the mesh pass,
/// its counts times the pass-through's split). Commands past the GPU's count are empty there.
#[allow(clippy::too_many_arguments)]
unsafe fn two_pass_indirect(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    pso: usize,
    tp: &super::real::TwoPassPso,
    sig: usize,
    max: u32,
    safe: Option<(usize, u64, u64)>,
    plan: (u32, u32),
    pc: unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64),
    guest: (Option<u64>, Option<u64>),
) -> Result<(), String> {
    let (scratch, records, count_at) = safe.ok_or("its arguments could not be sanitised (two passes need the sanitised copy)")?;
    if max == 0 || max > TWO_PASS_INDIRECT_COMMANDS {
        return Err(format!("{max} commands in two passes (at most {TWO_PASS_INDIRECT_COMMANDS})"));
    }
    let stride = super::indirect::stride_of(sig).ok_or("unknown command signature")? as u64;
    let dispatch_sig = super::indirect::dispatch_signature(stride as u32).ok_or("no dispatch signature for the prepass")?;
    let per_command = plan.0 as u64 * tp.record.record_dwords() as u64 * 4;
    static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !TOLD.swap(true, Ordering::Relaxed) {
        crate::xlog_warn!("D3D12.X: an indirect mesh call runs in two passes: each of its {max} commands draws at most {} thread groups ({} KB of records); more are cut", plan.0, per_command >> 10);
    }
    if let Some(args_at) = guest.0 {
        note_two_pass_cut(args_at, guest.1, stride, max, plan.0);
    }
    let (real, [io, _], layout, args, staged_io) = abi_views(this, pso, RootStage::Mesh)?;
    let res = two_pass_scratch()?;
    super::residency::note_list_untracked(this as usize);
    // the mesh pass's records follow the count in the sanitised copy
    let mesh_records = count_at + 4;
    for k in 0..max as u64 {
        let views = [io, res + k * per_command];
        let paired=super::material_gate_views::prepass::records::producer();
        let prepare=|planned|super::material_gate_views::prepass::prepare(list, real, tp, scratch, [records + k * stride, mesh_records + k * stride, count_at], sig, k, max, (guest.0.and_then(|a| a.checked_add(k * stride)), guest.1), views,planned,&staged_io);
        bind_views(this, list, tp.compute_rs, tp.compute, views, false)?;
        if k == 0 {
            note_lineage_uavs(this, (tp.compute, real), "indirect_mesh_two_pass_prepass", layout.as_deref(), &args);
        }
        let prepared=if paired {prepare(true)} else {None};
        gpu_timing::op(this, Timed::MeshCompute, || tp.compute, || {
            if let Some(probe)=&prepared {probe.compute(this,tp,false);}
            pc(this, dispatch_sig, 1, scratch, records + k * stride, 0, 0);
            if let Some(probe)=&prepared {probe.compute(this,tp,true);}
        });
        list.ResourceBarrier(&[uav_barrier_all()]);
        bind_views(this, list, tp.mesh_rs, real, views, true)?;
        if k == 0 {
            note_lineage_uavs(this, (real, real), "indirect_mesh_two_pass_draw", layout.as_deref(), &args);
            super::list_x::note_lineage_depth_draw(this, "indirect_mesh_two_pass_draw");
        }
        gpu_timing::op(this, Timed::MeshDraw, || pso, || {
            let probe=if paired {prepared} else {prepare(false)}.and_then(|probe|probe.start(list,real));
            pc(this, sig, 1, scratch, mesh_records + k * stride, 0, 0);
            if let Some(probe) = probe { probe.finish(list); }
        });
    }
    Ok(())
}

/// (`XCL_GPU_MARKERS`) The root arguments of the dispatch that is op `op` (from 0) of the
/// recording at marker `slot` / `generation`, as they were bound (`hang::describe`).
pub(super) fn dispatch_args(slot: usize, generation: u32, op: usize) -> Option<(Arc<Layout>, Vec<u8>)> {
    let d = DISPATCH_BINDINGS.lock().unwrap();
    d.iter().rev().find(|b| b.at == Some((slot, generation, op))).map(|b| (b.layout.clone(), b.args.clone()))
}

/// (`XCL_GPU_MARKERS`) A translated dispatch as it was bound.
struct BoundDispatch {
    title_pso: usize,
    layout: Arc<Layout>,
    args: Vec<u8>,
    /// its place in its recording: (marker slot, generation, op from 0)
    at: Option<(usize, u32, usize)>,
    /// its thread groups (none: an indirect dispatch)
    groups: Option<[u32; 3]>,
    /// the buffers its arguments bound then (`bound_writes`)
    buffers: Vec<(u64, u64, usize, String)>,
    images: Vec<BoundImage>,
    /// its user SGPRs as `bind_abi` built them (`State::last_io`)
    io: (Vec<u32>, Option<u32>, u32),
}

struct BoundImage {
    table: u32,
    param: usize,
    slot: u32,
    srv: u32,
    resource: Option<usize>,
    view: crate::gpu::desc::Image,
}

/// (`XCL_GPU_MARKERS`) the latest translated dispatches – what `writer_of` and
/// `describe_dispatch` search
static DISPATCH_BINDINGS: Mutex<std::collections::VecDeque<BoundDispatch>> = Mutex::new(std::collections::VecDeque::new());

/// (`XCL_GPU_MARKERS`) buffers bound to skipped mesh calls: (console address, bytes, pipeline)
static REFUSED_BOUND: Mutex<Vec<(u64, u64, usize)>> = Mutex::new(Vec::new());

/// The bytes of a bound buffer that count as its memory: a root descriptor's size is not known.
fn span(size: u64) -> u64 {
    if size == 16 << 20 {
        64 << 10
    } else {
        size
    }
}

fn marker_buffer_probes() -> &'static [(usize, u32, u32)] {
    static PROBES: std::sync::OnceLock<Vec<(usize, u32, u32)>> = std::sync::OnceLock::new();
    PROBES.get_or_init(|| {
        let parse = |s: &str| {
            let s = s.trim();
            match s.strip_prefix("0x") {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => s.parse().ok(),
            }
        };
        std::env::var("XCL_GPU_MARKER_BUFFER").unwrap_or_default().split(',').filter_map(|entry| {
            let [param, slot, offset]: [&str; 3] = entry.split(':').collect::<Vec<_>>().try_into().ok()?;
            Some((parse(param)? as usize, parse(slot)?, parse(offset)?))
        }).collect()
    })
}

fn buffer_dword_address(words: &[u32; 8], offset: u32) -> Option<u64> {
    if words[3] >> 30 != 0 {
        return None;
    }
    let desc = crate::gpu::desc::buffer(words[..4].try_into().unwrap());
    let bytes = desc.num_records.saturating_mul(desc.stride.max(1));
    (desc.address != 0 && offset.checked_add(4).is_some_and(|end| end <= bytes))
        .then(|| (desc.address + offset as u64) & !3)
}

fn dispatch_backing(resource: Option<usize>, address: Option<u64>) -> String {
    let mut kind = Vec::new();
    if let Some((local, non_local, unmapped)) = resource.and_then(super::vram::tile_backing) {
        kind.push(format!("tiled {local} local / {non_local} non-local / {unmapped} unmapped tiles"));
    }
    if let Some(s) = resource.and_then(super::residency::describe_backing) {
        kind.push(s);
    }
    if address.is_some_and(|va| super::real::uma_window_buffer(va as usize, 1).is_some()) {
        kind.push("address in imported host UMA window".into());
    }
    if kind.is_empty() { "backing unknown".into() } else { kind.join("; ") }
}

fn note_dispatch_binding(this: *mut c_void, title_pso: usize, groups: Option<[u32; 3]>) {
    if !dispatch_snapshot_enabled() {
        return;
    }
    let Some((layout, args, io)) = with(this, |s| s.layout.clone().map(|l| (l, s.args.clone(), std::mem::take(&mut s.last_io)))) else { return };
    note_dispatch_snapshot(this as usize, title_pso, groups, layout, args, io);
}

pub(super) fn note_dispatch_snapshot(list: usize, title_pso: usize, groups: Option<[u32; 3]>, layout: Arc<Layout>, args: Vec<u8>, io: (Vec<u32>, Option<u32>, u32)) {
    let at = super::markers::position(list).map(|(s, g, op)| (s, g, op.saturating_sub(1)));
    let mut images = Vec::new();
    let buffers = bound_resources(&layout, &args, Some(&mut images));
    if indirect_world_trace() {
        let cs = super::dispatch_timing::profile_shader(super::pso_async::resolve_if_ready(title_pso));
        let buffers = &buffers[..buffers.len().min(24)];
        let images: Vec<_> = images.iter().take(12).map(|i| (i.param, i.slot, i.resource, i.view.width, i.view.height, i.view.format, i.view.base_level)).collect();
        indirect_world_event(list, format_args!("dispatch cs {cs:016x?} marker {at:?} groups {groups:?} SGPR {:08x?} buffer bindings {buffers:x?} image bindings {images:x?}", io.0));
    }
    let mut d = DISPATCH_BINDINGS.lock().unwrap();
    if d.len() >= 16384 {
        d.pop_front();
    }
    d.push_back(BoundDispatch { title_pso, layout, args, at, groups, buffers, images, io });
}

/// (`XCL_GPU_MARKERS`) What the dispatch that is op `op` (from 0) of the recording at marker
/// `slot` / `generation` was bound to, for the snapshot of a GPU that hangs in it: its thread
/// groups, its root arguments, and for each buffer bound what memory holds there now, whether a
/// skipped dispatch or a skipped mesh call was bound to it too (it may never have been
/// written), and the latest dispatch before it that was.
pub(super) fn describe_dispatch(slot: usize, generation: u32, op: usize) -> Vec<String> {
    let d = DISPATCH_BINDINGS.lock().unwrap();
    let Some(i) = d.iter().rposition(|b| b.at == Some((slot, generation, op))) else {
        return vec![format!("not among the last {} dispatches kept", d.len())];
    };
    let b = &d[i];
    let word = |at: usize| b.args.get(at..at + 4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).unwrap_or(0);
    let hex = |w: &[u32]| w.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" ");
    let mut out = vec![format!("{}; thread groups {}", super::markers::pipeline(b.title_pso), b.groups.map_or("from an argument buffer".into(), |g| format!("{} x {} x {}", g[0], g[1], g[2])))];
    if let Some([local, non_local]) = super::real::video_memory_info() {
        out.push(format!("DXGI budget now: local {} / {} MiB, non-local {} / {} MiB",
            local.CurrentUsage >> 20, local.Budget >> 20, non_local.CurrentUsage >> 20, non_local.Budget >> 20));
    }
    // the registers the shader was given: s0:s1 the argument table, from s4 the first `inline`
    // bytes of the packed arguments (`console_list::user_data_for`)
    let (io, user_sgprs, inline) = &b.io;
    out.push(format!(
        "io words 0-15 as bound: {}; the shader has {} user registers, {inline} of the {} argument bytes are in registers from s4 (the root signature's rule: {}), the rest in the table at s0:s1",
        hex(io),
        user_sgprs.map_or("an unknown number of".into(), |n| n.to_string()),
        b.layout.size,
        b.layout.inline
    ));
    let place = |at: u32, bytes: u32| format!("bytes {at}..{} {}", at + bytes, if at + bytes <= *inline { format!("in s{}", 4 + at / 4) } else { "in the table".into() });
    out.push(format!(
        "root signature layout: {}",
        b.layout.params.iter().enumerate().map(|(k, p)| match *p {
            Param::Constants { at, count } => format!("p{k} {count} constants, {}", place(at, 4 * count)),
            Param::Descriptor { at } => format!("p{k} descriptor, {}", place(at, 8)),
            Param::Table { at } => format!("p{k} table, {}", place(at, 4)),
        }).collect::<Vec<_>>().join("; ")
    ));
    for (k, p) in b.layout.params.iter().enumerate() {
        out.push(match *p {
            Param::Constants { at, count } => format!("p{k} constants: {}", hex(&(0..count.min(64) as usize).map(|n| word(at as usize + 4 * n)).collect::<Vec<_>>())),
            Param::Descriptor { at } => format!("p{k} root descriptor: {:#x}", word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32),
            Param::Table { at } => format!("p{k} table at {:#x}", word(at as usize)),
        });
    }
    for &(param, slot, offset) in marker_buffer_probes() {
        let Some(Param::Table { at }) = b.layout.params.get(param) else { continue };
        let label = format!("p{param} slot {slot} buffer dword +{offset:#x}");
        let Some(words) = slot.checked_mul(32).and_then(|n| word(*at as usize).checked_add(n)).and_then(super::console_heap::words_at) else {
            out.push(format!("{label}: descriptor unavailable"));
            continue;
        };
        let value = match buffer_dword_address(&words, offset) {
            Some(va) if crate::gpu::shader::readable(va as usize, 4) => {
                let value = unsafe { (va as *const u32).read_unaligned() };
                format!("at {va:#x} now {value:#x} ({value})")
            }
            Some(va) => format!("at {va:#x} not readable by the CPU"),
            None => "not an in-bounds buffer dword".into(),
        };
        let backing = dispatch_backing(None, buffer_dword_address(&words, offset));
        out.push(format!("{label}: descriptor {}; {value}; {backing}", hex(&words)));
    }
    if let Some(t) = TRIPS.lock().unwrap().as_ref().filter(|t| !t.pieces.is_empty()) {
        if let Some(i) = (0..TRIP_RING).find(|&i| t.slots[i].0 != 0 && t.slots[i].1 == b.at) {
            out.push(format!("guard tripwire now: {}", trip_text(&t.words(i).map(|w| w as u64))));
            out.extend(t.loop_lines(i));
        }
    }
    for BoundImage { table, param, slot, srv, resource, view } in &b.images {
        let owner = resource.map_or_else(|| "owner untracked".into(), |r| format!("resource {r:#x}"));
        let backing = dispatch_backing(*resource, None);
        out.push(format!(
            "p{param} table {table:#x} slot {slot} image {:?} {}x{}x{}, GFX10 format {}, {} view mips, SRV index {srv}, companion UAV index {}, {owner}; {backing}",
            view.ty, view.width, view.height, view.depth, view.format, view.last_level + 1, srv.saturating_add(1)
        ));
    }
    let overlaps = |base: u64, size: u64, other: u64, other_size: u64| base < other + span(other_size) && other < base + span(size);
    let refused = REFUSED_BOUND.lock().unwrap();
    for (base, size, res, what) in b.buffers.iter().take(64) {
        let now = words8(*base).map_or("not readable by the CPU".into(), |w| hex(&w));
        let skipped = super::indirect::unwritten_overlapping(*base, span(*size)).map_or(String::new(), |(s, e, p)| format!("; NEVER WRITTEN? a skipped dispatch was bound to {s:#x}..{e:#x} ({})", super::markers::pipeline(p)));
        let mesh = refused.iter().find(|&&(rb, rs, _)| overlaps(*base, *size, rb, rs)).map_or(String::new(), |&(_, _, p)| format!("; a skipped mesh call was bound to it too (pipeline {p:#x})"));
        let before = d.iter().take(i).rev().enumerate().find(|(_, e)| e.buffers.iter().any(|&(eb, es, ..)| overlaps(*base, *size, eb, es))).map_or("; no dispatch kept before it was bound to it".to_string(), |(n, e)| {
            format!("; bound before by {}, {} dispatches earlier{}", super::markers::pipeline(e.title_pso), n + 1, e.at.map_or(String::new(), |(s, _, o)| format!(" (op {} of slot {s})", o + 1)))
        });
        let backing = dispatch_backing((*res != 0).then_some(*res), Some(*base));
        out.push(format!("{what}: {base:#x}+{:#x}{}: now {now}; {backing}{skipped}{mesh}{before}", span(*size), if *res == 0 { String::new() } else { format!(" (resource {res:#x})") }));
    }
    out
}

/// (`config::guard_trips_enabled`) The guard tripwire's counters (`recomp::IO_TRIP_AT`): four words of
/// committed unified memory per slot, one for each of `recomp::TRIP_KINDS` – the lanes of a translated shader in which
/// that guard gave up. Every translated dispatch gets the next slot of a ring; a graphics
/// pipeline keeps one slot for all its draws.
struct Trips {
    /// the counters' memory: pieces of `TRIP_PIECE` slots (no memory: no tripwire)
    pieces: Vec<usize>,
    stride: usize,
    /// per slot: the title pipeline it counts for, the place in its recording (dispatches),
    /// the counts already added to `totals`, and the profiled compute shader, if any
    slots: Vec<(usize, Option<(usize, u32, usize)>, [u32; 4], Option<u64>)>,
    /// dispatches given a slot so far
    next: usize,
    /// the graphics pipelines' slots (after the ring)
    gfx: HashMap<usize, usize>,
    /// by title pipeline: lanes by kind, and the dispatches (or looks at a graphics slot) with any
    totals: HashMap<usize, ([u64; 4], u64)>,
    told: u32,
    /// the first dispatches a guard gave up in, to be described (`describe_tripped`)
    pending: Vec<(usize, u32, usize)>,
}

const TRIP_RING: usize = 16384;
/// the pipelines with the most lanes that the 30 s line names (`indirect_calls`)
const TRIP_PIPELINES_TOLD: usize = 4;
const TRIP_GFX: usize = 4096;
const TRIP_PIECE: usize = 4096;
static TRIPS: Mutex<Option<Trips>> = Mutex::new(None);

fn trip_text(c: &[u64; 4]) -> String {
    let t: Vec<String> = crate::gpu::recomp::TRIP_KINDS.iter().zip(c).filter(|(_, n)| **n != 0).map(|(k, n)| format!("{k} {n}")).collect();
    if t.is_empty() {
        "nothing".into()
    } else {
        t.join(", ")
    }
}

impl Trips {
    fn over(pieces: Vec<usize>) -> Trips {
        Trips { pieces, stride: 16, slots: vec![(0, None, [0; 4], None); TRIP_RING + TRIP_GFX], next: 0, gfx: HashMap::new(), totals: HashMap::new(), told: 0, pending: Vec::new() }
    }

    /// where slot `i`'s four words are
    fn at(&self, i: usize) -> usize {
        self.pieces[i / TRIP_PIECE] + self.stride * (i % TRIP_PIECE)
    }

    fn words(&self, i: usize) -> [u32; 4] {
        std::array::from_fn(|k| unsafe { std::ptr::read_volatile((self.at(i) + 4 * k) as *const u32) })
    }

    fn loop_lines(&self, i: usize) -> Vec<String> {
        use crate::gpu::recomp::{LOOP_PROFILE_SITES, LOOP_PROFILE_WORDS};
        if self.stride == 16 || i >= TRIP_RING {
            return Vec::new();
        }
        (0..LOOP_PROFILE_SITES).filter_map(|site| {
            let base = self.at(i) + 16 + site * LOOP_PROFILE_WORDS * 4;
            let [pc, peak_seen, max, low, high, lanes, early, bound, budget, cap] =
                std::array::from_fn(|offset| unsafe { ((base + offset * 4) as *const u32).read_volatile() });
            let pc = pc.checked_sub(1)?;
            let total = ((high as u64) << 32) | low as u64;
            let avg = if lanes == 0 { "none finished".to_string() } else { format!("{:.2}", total as f64 / lanes as f64) };
            let natural_unknown = lanes.saturating_sub(early.saturating_add(bound).saturating_add(budget).saturating_add(cap));
            Some(format!("loop profile +{pc:#x}: avg {avg}, max completed {max}, live milestone {peak_seen}; completed lane visits {lanes}, below static estimate {early}, reached static estimate {bound}, natural unclassified {natural_unknown}, shared budget {budget}, static cap {cap}"))
        }).collect()
    }

    /// What slot `i` counted since it was last looked at: into the totals, and a line for the
    /// first ones.
    fn harvest(&mut self, i: usize) {
        let now = self.words(i);
        let (pso, at, seen, _) = self.slots[i];
        if now == seen {
            return;
        }
        let delta: [u64; 4] = std::array::from_fn(|k| now[k].wrapping_sub(seen[k]) as u64);
        self.slots[i].2 = now;
        let t = self.totals.entry(pso).or_default();
        for k in 0..4 {
            t.0[k] += delta[k];
        }
        t.1 += 1;
        self.told += 1;
        if let Some(at) = at.filter(|_| self.told <= 8) {
            self.pending.push(at);
        }
        if self.told <= 64 || self.told.is_power_of_two() {
            let place = at.map_or(String::new(), |(s, _, op)| format!(" (op {} of slot {s})", op + 1));
            crate::xlog_warn!(
                "guard tripwire: {} in {} with {}{place} (#{})",
                trip_text(&delta),
                if i < TRIP_RING { "a dispatch" } else { "draws" },
                super::markers::pipeline(pso),
                self.told
            );
        }
    }

    /// The slot for the next dispatch (the oldest of the ring, harvested and cleared) or for
    /// graphics pipeline `title_pso`.
    fn take(&mut self, title_pso: usize, at: Option<(usize, u32, usize)>, gfx: bool, shader: Option<u64>) -> usize {
        if gfx {
            let n = TRIP_RING + self.gfx.len().min(TRIP_GFX - 1);
            let i = *self.gfx.entry(title_pso).or_insert(n);
            if self.slots[i].0 == 0 {
                self.slots[i].0 = title_pso;
            }
            return i;
        }
        let i = self.next % TRIP_RING;
        self.next += 1;
        self.harvest(i);
        unsafe { std::ptr::write_bytes(self.at(i) as *mut u8, 0, self.stride) };
        self.slots[i] = (title_pso, at, [0; 4], shader);
        i
    }

    /// Every slot looked at, then: the lanes by kind in all, and the `most` pipelines with the
    /// most lanes.
    fn lines(&mut self, most: usize) -> Vec<String> {
        for i in 0..self.slots.len() {
            if self.slots[i].0 != 0 {
                self.harvest(i);
            }
        }
        let mut v: Vec<(usize, [u64; 4], u64)> = self.totals.iter().map(|(&p, &(lanes, hits))| (p, lanes, hits)).collect();
        v.sort_by_key(|&(_, lanes, _)| std::cmp::Reverse(lanes.iter().sum::<u64>()));
        let all: [u64; 4] = std::array::from_fn(|k| v.iter().map(|e| e.1[k]).sum());
        let mut out = vec![format!("guard tripwire: {} in {} pipelines ({} dispatches and {} graphics pipelines watched)", trip_text(&all), v.len(), self.next, self.gfx.len())];
        for (p, lanes, hits) in v.into_iter().take(most) {
            out.push(format!("guard tripwire: {} in {hits} {} with {}", trip_text(&lanes), if self.gfx.contains_key(&p) { "looks at draws" } else { "dispatches" }, super::markers::pipeline(p)));
        }
        out.extend(self.profile_lines(most));
        out
    }

    fn profile_lines(&self, most: usize) -> Vec<String> {
        let mut out = Vec::new();
        if self.stride == 16 || most == 0 {
            return out;
        }
        let mut profiled = std::collections::HashSet::new();
        for seq in (self.next.saturating_sub(TRIP_RING)..self.next).rev() {
            let i = seq % TRIP_RING;
            let (pso, at, _, shader) = self.slots[i];
            let Some(hash) = shader else { continue };
            if profiled.contains(&(pso, hash)) {
                continue;
            }
            let lines = self.loop_lines(i);
            if lines.is_empty() {
                continue;
            }
            profiled.insert((pso, hash));
            let place = at.map_or(String::new(), |(s, _, op)| format!(" (op {} of slot {s})", op + 1));
            out.extend(lines.into_iter().map(|line| format!("CS {hash:016x} pipeline {pso:#x}{place}: {line}")));
            if profiled.len() == most {
                break;
            }
        }
        out
    }
}

/// io[`IO_TRIP_WINDOW`] and io[`IO_TRIP_AT`] for the next dispatch or draw
/// of `list` with `title_pso`: the memory window of its four counters and their offset in it;
/// offset 0: no tripwire.
pub(super) fn trip_watch(list: *mut c_void, title_pso: usize, real_pso: usize, gfx: bool) -> (u32, u32) {
    let markers = super::markers::enabled();
    let shader = if gfx { None } else { super::dispatch_timing::loop_shader(real_pso) };
    if !markers && shader.is_none() && !crate::config::guard_trips_enabled() {
        return (0, 0);
    }
    let Some(u) = crate::gpu::uma::get() else { return (0, 0) };
    let at = if markers && !gfx { super::markers::position(list as usize) } else { None };
    let mut g = TRIPS.lock().unwrap();
    let t = g.get_or_insert_with(|| {
        // from XCL's own pool, as the lists' argument tables are (the coherent window, which
        // the CPU reads; the title holds the rest of it)
        let (mut arena, mut blocks) = (crate::gpu::uma::Arena::default(), Vec::new());
        let stride = 16 + if crate::gpu::recomp::loop_profile_enabled() {
            crate::gpu::recomp::LOOP_PROFILE_SITES * crate::gpu::recomp::LOOP_PROFILE_WORDS * 4
        } else { 0 };
        let pieces: Option<Vec<usize>> = (0..(TRIP_RING + TRIP_GFX) / TRIP_PIECE)
            .map(|_| {
                let p = arena.take(u, stride * TRIP_PIECE, 16, &mut blocks).filter(|&p| u.is_committed(p as usize))?;
                unsafe { std::ptr::write_bytes(p, 0, stride * TRIP_PIECE) };
                Some(p as usize)
            })
            .collect();
        match &pieces {
            Some(p) => crate::xlog_info!("guard tripwire: counters for {TRIP_RING} dispatches and {TRIP_GFX} graphics pipelines, the first at {:#x}", p[0]),
            None => crate::xlog_warn!("guard tripwire: no room in committed unified memory for its counters – off"),
        }
        let mut trips = Trips::over(pieces.unwrap_or_default());
        trips.stride = stride;
        trips
    });
    if t.pieces.is_empty() {
        return (0, 0);
    }
    let i = t.take(title_pso, at, gfx, shader);
    let a = t.at(i);
    let pending = std::mem::take(&mut t.pending);
    drop(g);
    describe_tripped(pending);
    u.window_of(a).map_or((0, 0), |k| (k as u32, (a - u.window(k).0) as u32))
}

/// What the first dispatches a guard gave up in were bound to (`describe_dispatch`: their
/// registers, root arguments and buffers) – the lock on the counters is not held here.
fn describe_tripped(pending: Vec<(usize, u32, usize)>) {
    for (slot, generation, op) in pending {
        for line in describe_dispatch(slot, generation, op) {
            crate::xlog_warn!("guard tripwire: slot {slot} op {}: {line}", op + 1);
        }
    }
}

/// (`XCL_GPU_MARKERS`) The guards that gave up so far (`Trips::lines`); nothing without a
/// tripwire.
pub(super) fn trips(most: usize) -> Vec<String> {
    let (lines, pending) = TRIPS.lock().unwrap().as_mut().filter(|t| !t.pieces.is_empty()).map_or_else(Default::default, |t| (t.lines(most), std::mem::take(&mut t.pending)));
    describe_tripped(pending);
    lines
}

/// (`XCL_GPU_MARKERS`) The dispatches of the recordings `open` (marker slot, generation) in
/// which a guard has given up by now: (slot, op from 0, the lanes by kind), from the counters as
/// they are. A dispatch whose counters the ring has given to a later one is not among them.
pub(super) fn trips_in(open: &[(usize, u32)]) -> Vec<(usize, usize, String)> {
    let g = TRIPS.lock().unwrap();
    let Some(t) = g.as_ref().filter(|t| !t.pieces.is_empty()) else { return Vec::new() };
    let mut v: Vec<(usize, usize, String)> = (0..TRIP_RING)
        .filter_map(|i| {
            let (pso, at, _, _) = t.slots[i];
            let (slot, _, op) = at.filter(|&(s, generation, _)| pso != 0 && open.contains(&(s, generation)))?;
            let w = t.words(i);
            w.iter().any(|&n| n != 0).then(|| (slot, op, trip_text(&w.map(|n| n as u64))))
        })
        .collect();
    v.sort();
    v
}

/// (`XCL_GPU_MARKERS`) The lanes that gave up in the draws of graphics pipeline `title_pso`,
/// all of them so far, by kind (None: its draws are not watched).
pub(super) fn trips_of_draws(title_pso: usize) -> Option<String> {
    let g = TRIPS.lock().unwrap();
    let t = g.as_ref().filter(|t| !t.pieces.is_empty())?;
    t.gfx.get(&title_pso).map(|&i| trip_text(&t.words(i).map(|n| n as u64)))
}

/// The process ends: the guards that gave up since the last 30 s line would be lost.
pub fn log_trips() {
    for line in trips(24) {
        crate::xlog_info!("at the end: {line}");
    }
}

pub fn log_loop_profiles() {
    if !crate::gpu::recomp::loop_profile_enabled() {
        return;
    }
    let lines = TRIPS.lock().unwrap().as_ref().filter(|t| !t.pieces.is_empty()).map_or_else(Vec::new, |t| t.profile_lines(24));
    for line in lines {
        crate::xlog_info!("{line}");
    }
}

/// The latest translated dispatch whose bound buffers (as bound then) cover console address
/// `va`, as text, and its place in its recording.
fn writer_of(va: u64) -> (String, Option<(usize, u32, usize)>) {
    let d = DISPATCH_BINDINGS.lock().unwrap();
    for b in d.iter().rev() {
        if let Some((base, size, _, what)) = b.buffers.iter().find(|&&(base, size, _, _)| va >= base && va < base + size) {
            let cs = super::dispatch_timing::profile_shader(super::pso_async::resolve_if_ready(b.title_pso));
            return (format!("{} CS {cs:016x?} through {what} ({base:#x}+{size:#x})", super::markers::pipeline(b.title_pso)), b.at);
        }
    }
    if let Some((s, e, p)) = super::indirect::unwritten_at(va) {
        return (format!("none of the last {} dispatches; unwritten since pipeline {p:#x} was skipped ({s:#x}..{e:#x})", d.len()), None);
    }
    (format!("none of the last {} dispatches binds it", d.len()), None)
}

/// The buffers root arguments `args` of `layout` bind (console address, bytes, PC resource,
/// which argument): the memory a dispatch with them may write.
pub(super) fn bound_writes(layout: &Layout, args: &[u8]) -> Vec<(u64, u64, usize, String)> {
    bound_resources(layout, args, None)
}

fn bound_resources(
    layout: &Layout,
    args: &[u8],
    mut images: Option<&mut Vec<BoundImage>>,
) -> Vec<(u64, u64, usize, String)> {
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut out = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Descriptor { at } => {
                let va = word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32;
                if va != 0 {
                    out.push((va, 16 << 20, 0, format!("p{i} root descriptor")));
                }
            }
            Param::Table { at } => {
                let t = word(at as usize);
                for k in 0..64u32 {
                    let addr = t.wrapping_add(32 * k);
                    let Some(w) = super::console_heap::words_at(addr) else { break };
                    let resource = || crate::gpu::uma::get().and_then(|u| super::console_heap::resource_at(u.base() + addr as usize)).unwrap_or(0);
                    if w[3] >> 30 != 0 {
                        if let Some(images) = images.as_deref_mut() {
                            if let Some(view) = crate::gpu::desc::image(&w) {
                                images.push(BoundImage { table: t, param: i, slot: k, srv: w[0], resource: super::console_heap::image_resource_at(addr, w[0]), view });
                            }
                        }
                        continue;
                    }
                    let (base, stride, records) = (w[0] as u64 | ((w[1] & 0xffff) as u64) << 32, ((w[1] >> 16) & 0x3fff) as u64, w[2] as u64);
                    if base == 0 || records == 0 {
                        continue;
                    }
                    out.push((base, records * stride.max(1), resource(), format!("p{i} table {t:#x} slot {k}")));
                }
            }
            _ => {}
        }
    }
    out
}

/// The arguments and counts of the first indirect calls after a skipped dispatch (console
/// addresses), read again when the device is removed (`indirect_args_at_removal`).
static INDIRECT_SEEN: Mutex<Vec<(u64, Option<u64>)>> = Mutex::new(Vec::new());

/// The first 8 words at a console address, when the CPU can read it.
fn words8(va: u64) -> Option<[u32; 8]> {
    if va == 0 || !crate::gpu::shader::readable(va as usize, 32) {
        return None;
    }
    let mut w = [0u32; 8];
    for (k, x) in w.iter_mut().enumerate() {
        *x = unsafe { ((va as usize + 4 * k) as *const u32).read_unaligned() };
    }
    Some(w)
}

/// An indirect call after a dispatch was skipped: its most commands, where its arguments and
/// count are, what they hold now, and which skipped dispatch's buffers cover them (logged, the
/// first 64).
unsafe fn note_indirect(pso: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    let writers = SKIPPED_WRITERS.lock().unwrap();
    if writers.is_empty() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 64 {
        return;
    }
    let va = |r: usize, off: u64| {
        (r != 0).then(|| super::real::title_address(r).map_or_else(|| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map_or(0, |r| r.GetGPUVirtualAddress()), |a| a as u64) + off)
    };
    let (args_va, count_va) = (va(args, args_off).unwrap_or(0), va(count_buf, count_off));
    let covers = |res: usize, at: u64| {
        writers.iter().find_map(|(p, ws)| {
            ws.iter()
                .find(|&&(base, size, r, _)| (r != 0 && r == res) || (at >= base && at < base + size))
                .map(|(base, size, r, what)| format!("{p:#x} ({what}: {base:#x}, {size} B{})", if *r == res && res != 0 { ", the same resource" } else { "" }))
        })
    };
    let producer = match (covers(args, args_va), count_va.and_then(|v| covers(count_buf, v))) {
        (None, None) => format!("no skipped dispatch's root descriptors or table buffers cover them ({} skipped pipelines)", writers.len()),
        (a, c) => format!("arguments written by skipped {}, count by skipped {}", a.unwrap_or_else(|| "none".into()), c.unwrap_or_else(|| "none".into())),
    };
    {
        let mut seen = INDIRECT_SEEN.lock().unwrap();
        if seen.len() < 16 {
            seen.push((args_va, count_va));
        }
    }
    crate::xlog_warn!(
        "D3D12.X: ExecuteIndirect after a skipped dispatch (pipeline {pso:#x}, translated graphics {}, compute {}; signature {sig:#x}): at most {max} commands, arguments at {args_va:#x} (now {:x?}), count {}; {producer}",
        super::real::is_translated_graphics(pso),
        super::real::is_translated_compute(pso),
        words8(args_va),
        count_va.map_or_else(|| "none (max is the count)".to_string(), |v| format!("at {v:#x} (now {:x?})", words8(v).map(|w| w[0])))
    );
}

/// When the device is removed: what the arguments of the first indirect calls after a skipped
/// dispatch hold then (written by the GPU by that time, where the CPU can read them).
pub(super) fn indirect_args_at_removal() {
    for (args, count) in INDIRECT_SEEN.lock().unwrap().iter() {
        crate::xlog_error!(
            "D3D12.X: at the removal, ExecuteIndirect arguments at {args:#x}: {:x?}{}",
            words8(*args),
            count.map_or(String::new(), |c| format!(", count at {c:#x}: {:x?}", words8(c).map(|w| w[0])))
        );
    }
}

/// The first dispatch or indirect call of a traced named pipeline (`XCL_TRACE_NAMED_PSO`).
fn note_named_use(title_pso: usize, pso: usize, what: &str) {
    let Some(name) = super::real::traced_pso(title_pso) else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if seen.len() >= 4096 || seen.contains(&title_pso) {
        return;
    }
    seen.push(title_pso);
    let state = if super::real::is_translated_compute(pso) || super::real::is_translated_graphics(pso) {
        "translated, run".to_string()
    } else {
        format!("no translation, skipped: {}", super::real::why_untranslated(pso).unwrap_or_else(|| "no reason kept".into()))
    };
    crate::xlog_info!("named pipeline '{name}': first {what} with {pso:#x} (title's {title_pso:#x}): {state}");
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
/// A dispatch of the title's `title_pso` (translated as `pso`) with more than 65,535 groups on an
/// axis: the console runs it, D3D12 allows 65,535 (D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION).
/// Spread over x and y (`spread_launch`) when `XCL_DISPATCH_SPREAD` is on and the program can be;
/// otherwise passed on as it is. Some((groups launched, io[`IO_GROUP_SPREAD`])) when spread. Logged
/// at the 1st, 2nd, 4th ... call of a pipeline, so a run shows whether the title's lists reach the
/// limit.
fn note_over_limit(title_pso: usize, pso: usize, call: &str, groups: [u32; 3]) -> Option<([u32; 3], [u32; 2])> {
    static SEEN: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
    let n = {
        let mut g = SEEN.lock().unwrap();
        let n = g.get_or_insert_with(HashMap::new).entry(title_pso).or_insert(0);
        *n += 1;
        *n
    };
    let on = crate::gpu::recomp::dispatch_spread();
    let able = on && super::real::spreads_groups(pso);
    let spread = if able { spread_launch(groups) } else { None };
    if spread.is_some() {
        DIRECT_SPREAD.fetch_add(1, Ordering::Relaxed);
    }
    if n.is_power_of_two() {
        let (key, _) = super::pso_async::key_of(title_pso);
        let how = match spread {
            Some((l, _)) => format!("spread over x and y as {}x{}x{} groups (XCL_DISPATCH_SPREAD; {} spread so far)", l[0], l[1], l[2], DIRECT_SPREAD.load(Ordering::Relaxed)),
            None if !on => "passed on as it is (XCL_DISPATCH_SPREAD=0)".into(),
            None if !able => "passed on as it is: its program cannot be spread (not XCL's own translation)".into(),
            None => "passed on as it is: z is past 65,535 or x * y past 65,535 rows of 65,535".into(),
        };
        crate::xlog_warn!("D3D12.X: {call} of pipeline {title_pso:#x} (key {key:016x}) with {}x{}x{} groups, past D3D12's 65,535 on an axis (#{n} of this pipeline); {how}", groups[0], groups[1], groups[2]);
    }
    spread
}

/// direct dispatches spread over x and y so far (`note_over_limit`)
static DIRECT_SPREAD: AtomicU64 = AtomicU64::new(0);

/// What a dispatch of `groups` with more than 65,535 on an axis launches spread over x and y
/// (`recomp::IO_GROUP_SPREAD`): its groups, `GROUP_SPREAD_WIDTH` x ceil(x * y / that) x z, and the
/// words for io[`IO_GROUP_SPREAD`]. No groups when an axis is 0. None: it cannot be spread (z past
/// 65,535, or x * y past 65,535 rows).
fn spread_launch([x, y, z]: [u32; 3]) -> Option<([u32; 3], [u32; 2])> {
    let width = u64::from(crate::gpu::recomp::GROUP_SPREAD_WIDTH);
    let xy = u64::from(x) * u64::from(y);
    if xy == 0 || z == 0 {
        return Some(([0; 3], [0; 2]));
    }
    if u64::from(z) > width || xy > width * width {
        return None;
    }
    Some(([width as u32, xy.div_ceil(width) as u32, z], [x, y]))
}

/// Per title compute pipeline of a lowered ExecuteIndirect (`indirect_spread`): its slot of
/// `OVER_WORDS` words in `OVER_BLOCK` bytes of unified memory, and the count logged last as a
/// power-of-two rank.
static INDIRECT_OVER: Mutex<Option<(usize, HashMap<usize, (usize, u32)>)>> = Mutex::new(None);
const OVER_BLOCK: usize = 64 << 10;
const OVER_WORDS: usize = 8;

/// For the rows of a lowered ExecuteIndirect of `title_pso` run with the translated `pso`: the
/// byte offset of io[`IO_GROUP_SPREAD`] in a row when its dispatches past 65,535 groups on an axis
/// are spread (0: capped at 65,535, as with `XCL_DISPATCH_SPREAD=0`), and the GPU address of the
/// pipeline's counters (0: none), which XCL's LimitDispatch pass keeps: the records past 65,535 on
/// an axis, the largest x, y and z asked, the records of them left capped. What the GPU counted by
/// now is logged each time its count passes a power of two.
fn indirect_spread(title_pso: usize, pso: usize) -> (u32, u64) {
    let on = crate::gpu::recomp::dispatch_spread();
    let able = on && super::real::spreads_groups(pso);
    let at = if able { (crate::gpu::recomp::IO_GROUP_SPREAD[0] * 4) as u32 } else { 0 };
    let mut g = INDIRECT_OVER.lock().unwrap();
    let (block, slots) = g.get_or_insert_with(|| {
        let block = crate::gpu::uma::get().and_then(|u| u.alloc_in(0, OVER_BLOCK, 256)).map_or(0, |p| p as usize);
        if block != 0 {
            unsafe { std::ptr::write_bytes(block as *mut u8, 0, OVER_BLOCK) };
        }
        (block, HashMap::new())
    });
    let next = slots.len();
    if *block == 0 || (!slots.contains_key(&title_pso) && (next + 1) * OVER_WORDS * 4 > OVER_BLOCK) {
        return (at, 0);
    }
    let base = *block;
    let (slot, logged) = slots.entry(title_pso).or_insert((base + next * OVER_WORDS * 4, 0));
    let words: [u32; 5] = std::array::from_fn(|k| unsafe { std::ptr::read_volatile((*slot as *const u32).add(k)) });
    if words[0] != 0 && words[0].ilog2() + 1 > *logged {
        *logged = words[0].ilog2() + 1;
        let (key, _) = super::pso_async::key_of(title_pso);
        let how = if able {
            format!("spread over x and y (XCL_DISPATCH_SPREAD), {} of them left capped at 65,535 (z past 65,535 or x * y past 65,535 rows)", words[4])
        } else if !on {
            "capped at 65,535 on each axis (XCL_DISPATCH_SPREAD=0)".into()
        } else {
            "capped at 65,535 on each axis: its program cannot be spread (not XCL's own translation)".into()
        };
        crate::xlog_warn!("D3D12.X: ExecuteIndirect of pipeline {title_pso:#x} (key {key:016x}): {} dispatch records past D3D12's 65,535 groups on an axis so far, the largest asked per axis {}x{}x{}; {how}", words[0], words[1], words[2], words[3]);
    }
    (at, super::real::uma_va_of(*slot).unwrap_or(0))
}

pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    dispatch_flags(this, x, y, z, None);
}

/// `DispatchX` flag USE_THREAD_DIMENSIONS: x, y, z are thread counts, not group counts; the
/// console launches whole groups and leaves the lanes past each count out of EXEC (0x8000, the
/// wave32 bit of 0x8020, needs nothing here: the pipeline's program has its wave size)
const DISPATCH_THREAD_DIMENSIONS: u32 = 0x20;

/// What a dispatch of `dims` with `DispatchX` `flags` launches on a pipeline whose console
/// group shape is `threads`: its groups, and the thread limits for io[`IO_THREAD_LIMIT`] (0: that
/// axis ends on a whole group). With thread dimensions an axis of n threads is ceil(n / numthreads)
/// groups. None: it cannot be run as the console would (thread dimensions, group shape not
/// known). `enabled` false (`XCL_DISPATCH_THREAD_DIMENSIONS=0`): the flags are ignored.
/// Adapted from ivall/xcl a3e820d8 (PR #2610).
fn dispatch_groups(dims: [u32; 3], flags: u32, threads: Option<[u32; 3]>, enabled: bool) -> Option<([u32; 3], [u32; 3])> {
    if !enabled || flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    let threads = threads?.map(|t| t.max(1));
    let groups = std::array::from_fn(|k| dims[k].div_ceil(threads[k]));
    let limits = std::array::from_fn(|k| if dims[k] % threads[k] != 0 { dims[k] } else { 0 });
    Some((groups, limits))
}

/// Once per distinct `DispatchX` flags value: the counts, the pipeline's console group shape,
/// what is launched, and whether thread dimensions are honoured (`XCL_DISPATCH_THREAD_DIMENSIONS`).
fn note_dispatch_x_flags(flags: u32, dims: [u32; 3], threads: Option<[u32; 3]>, launched: Option<([u32; 3], [u32; 3])>) {
    static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    {
        let mut seen = SEEN.lock().unwrap();
        if seen.contains(&flags) {
            return;
        }
        seen.push(flags);
    }
    let on = if crate::gpu::recomp::dispatch_thread_dimensions() { "on" } else { "off (XCL_DISPATCH_THREAD_DIMENSIONS=0)" };
    let launched = match launched {
        Some((groups, limits)) => format!("launched as {groups:?} groups, thread limits {limits:?}"),
        None => "skipped: the pipeline's group shape is not known".into(),
    };
    crate::xlog_info!("D3D12.X: first DispatchX with flags {flags:#x}: counts {dims:?}, numthreads {threads:?}, {launched} (thread dimensions {on})");
}

/// The images the compute root arguments on `list` name, for `fsr4_standin`: those with a PC
/// resource behind their view.
pub(super) fn compute_images(list: *mut c_void) -> Vec<super::fsr4_standin::Image> {
    let (layout, args) = with(list, |s| (s.layout.clone(), s.args.clone()));
    let mut images = Vec::new();
    if let Some(layout) = layout.as_ref() {
        bound_resources(layout, &args, Some(&mut images));
    }
    images
        .iter()
        .filter_map(|i| {
            let resource = i.resource?;
            let p = resource as *mut c_void;
            let format = unsafe { ID3D12Resource::from_raw_borrowed(&p) }.map_or(0, |r| unsafe { r.GetDesc() }.Format.0);
            Some(super::fsr4_standin::Image {
                resource,
                index: i.srv,
                uav: super::console_heap::native_uav_at(i.table.wrapping_add(32 * i.slot), i.srv) == Some(resource),
                width: i.view.width,
                height: i.view.height,
                flat: i.view.ty == crate::gpu::desc::ImageType::D2,
                format,
            })
        })
        .collect()
}

/// The layout of the compute root signature on `list` and the root arguments kept for it, for
/// `title_gta6_ambient`: what the dispatch just recorded there was bound with.
pub(super) fn compute_arguments(list: *mut c_void) -> Option<(Arc<Layout>, Vec<u8>)> {
    with(list, |s| s.layout.clone().map(|layout| (layout, s.args.clone())))
}

/// A dispatch with the title's `title_pso`, a pass of a chain not all built (`pass_chain`): skipped,
/// counted as one whose pipeline is still building, and what it writes marked unwritten. One the
/// FSR4 stand-in takes (`fsr4_standin`) is skipped the same way, and is not one that is still building.
fn chain_held(this: *mut c_void, title_pso: usize) -> bool {
    if unsafe { super::fsr4_standin::takes(this, title_pso) } {
        note_skipped_writer(this, title_pso, title_pso);
        count(DISPATCHES_SKIPPED);
        return true;
    }
    if super::pass_chain::may_dispatch(title_pso) {
        return false;
    }
    note_skipped_writer(this, title_pso, title_pso);
    super::pso_async::note_pending_skip();
    count(DISPATCHES_SKIPPED);
    true
}

/// `Dispatch` (`flags` None) and `DispatchX(x, y, z, flags)`.
pub(super) unsafe fn dispatch_flags(this: *mut c_void, x: u32, y: u32, z: u32, flags: Option<u32>) {
    let title_pso = with(this, |s| s.pso);
    if chain_held(this, title_pso) {
        return;
    }
    if put_off(this, super::list_x::Deferred::Dispatch(x, y, z, flags)) {
        return;
    }
    let pso = match super::pso_async::waiting_if(in_bundle(this), || super::pso_async::resolve_for_compute_use(title_pso)) {
        Ok(pso) => pso,
        Err(skip) => return note_pending_compute_skip(this, title_pso, "Dispatch", skip),
    };
    if shader_skip(pso) {
        note_skipped_writer(this, pso, title_pso);
        count(DISPATCHES_SKIPPED);
        return;
    }
    note_named_use(title_pso, pso, "dispatch");
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        note_skipped_writer(this, pso, title_pso);
        note_failed_skip(title_pso, pso, "dispatch");
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            let why = super::real::why_untranslated(pso).unwrap_or_else(|| "no reason kept (not made through XCL's compute pipeline path)".into());
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped: {why}");
        }
        count(DISPATCHES_SKIPPED);
        return;
    }
    let threads = super::real::pso_threads(pso);
    let launched = dispatch_groups([x, y, z], flags.unwrap_or(0), threads, crate::gpu::recomp::dispatch_thread_dimensions());
    if let Some(flags) = flags {
        note_dispatch_x_flags(flags, [x, y, z], threads, launched);
    }
    let Some((groups, limits)) = launched else {
        note_skipped_writer(this, pso, title_pso);
        count(DISPATCHES_SKIPPED);
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: DispatchX of {x}x{y}x{z} threads (flags {:#x}) with a pipeline whose group shape is not known ({pso:#x}) skipped", flags.unwrap_or(0));
        }
        return;
    };
    let [x, y, z] = groups;
    reach(this);
    let spread = if x.max(y).max(z) > 65535 { note_over_limit(title_pso, pso, "Dispatch", [x, y, z]) } else { None };
    let ([lx, ly, lz], words) = spread.unwrap_or(([x, y, z], [0; 2]));
    with(this, |s| (s.limits, s.spread) = (limits, words));
    let done = translated_dispatch(this, list, pso, lx, ly, lz);
    with(this, |s| (s.limits, s.spread) = ([0; 3], [0; 2]));
    match done {
        Ok(()) => {
            count(DISPATCHES);
            mark(this, "dispatch");
            super::submit_watch::groups(this as usize, [x, y, z]);
            note_dispatch_binding(this, title_pso, Some([x, y, z]));
            note_built_writes(this, title_pso);
            super::title_gta6_ambient::after_dispatch(this, title_pso);
        }
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if let Some(w) = (va != 0).then(|| crate::memmap::title_words(va, n)).flatten() {
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    super::depth_twin_probe::begin();
    let (restore, writer) = bind_abi(this, list, pso, RootStage::Compute, "dispatch")?;
    if let Some((hash, index)) = super::dispatch_timing::take_capture(pso) {
        capture_dispatch(this, pso, hash, index, [x, y, z]);
    }
    super::residency::note_list_untracked(this as usize);
    gpu_timing::op_grid(this, Timed::Dispatch, || with(this, |s| s.pso), [x, y, z], || {
        super::dispatch_timing::dispatch(list, pso, [x, y, z]);
        if let Some(writer) = writer { writer.recorded([x, y, z]); }
    });
    if let Some((original, res)) = restore {
        set_gate_state(this, original, res);
    }
    super::depth_twin_probe::dispatched(this as usize, [x, y, z]);
    Ok(())
}

fn capture_dispatch(this: *mut c_void, pso: usize, hash: u64, index: u32, [x, y, z]: [u32; 3]) {
    let Some((layout, args)) = with(this, |s| s.layout.clone().map(|layout| (layout, s.args.clone()))) else { return };
    let inline = (console_list::inline_for(&layout, super::real::pso_user_sgprs(pso)) as usize).min(args.len());
    let params = layout.params.iter().enumerate().map(|(i, p)| match *p {
        Param::Constants { at, count } => format!("p{i} {count} constants, bytes {at}..{}", at + 4 * count),
        Param::Descriptor { at } => format!("p{i} descriptor, bytes {at}..{}", at + 8),
        Param::Table { at } => format!("p{i} table, bytes {at}..{}", at + 4),
    }).collect::<Vec<_>>().join("; ");
    crate::xlog_info!("GPU dispatch capture {index}: CS {hash:016x}; thread groups {x} x {y} x {z}");
    crate::xlog_info!("GPU dispatch capture {index}: inline argument bytes: {inline}");
    crate::xlog_info!("GPU dispatch capture {index}: root signature layout: {params}");
    for line in super::hang::arguments(&layout, &args, false) {
        crate::xlog_info!("GPU dispatch capture {index}: {line}");
    }
}

/// `XCL_DS_VALIDATE`: io[`IO_CARRY_AT`], the window 0 offset of a word translated shaders set
/// when a lane's address leaves the window of its scalar base (`emit::Em::check_carry`), and a
/// warning the first time one did.
pub(super) fn carry_watch() -> u32 {
    static WORD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let p = *WORD.get_or_init(|| {
        if !std::env::var("XCL_DS_VALIDATE").is_ok_and(|v| v != "0") {
            return 0;
        }
        let Some(p) = crate::gpu::uma::get().and_then(|u| u.alloc_in(0, 16, 16)) else { return 0 };
        unsafe { std::ptr::write_volatile(p as *mut u32, 0) };
        p as usize
    });
    if p == 0 {
        return 0;
    }
    if unsafe { std::ptr::read_volatile(p as *const u32) } != 0 && !TOLD.swap(true, Ordering::Relaxed) {
        crate::xlog_warn!("XCL_DS_VALIDATE: a translated shader addressed memory past the 4 GiB window of its scalar base (outside any allocation): the access wrapped inside that window");
    }
    p as u32
}

unsafe fn set_gate_state(raw: *mut c_void, pipeline: usize, result_va: u64) {
    let set_pso = std::mem::offset_of!(ID3D12GraphicsCommandList_Vtbl, SetPipelineState) / size_of::<usize>();
    let set_uav = std::mem::offset_of!(ID3D12GraphicsCommandList_Vtbl, SetComputeRootUnorderedAccessView) / size_of::<usize>();
    gpu_timing::native::<unsafe extern "system" fn(*mut c_void, u32, u64)>(raw, set_uav)(raw, 1, result_va);
    gpu_timing::native::<unsafe extern "system" fn(*mut c_void, usize)>(raw, set_pso)(raw, pipeline);
}

#[derive(Clone, Copy)]
enum RootStage { Compute, Graphics, Mesh }

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, the shared descriptor heap (the memory windows' views, `real::mem_views`);
/// then the pipeline.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, stage: RootStage, call: &'static str) -> Result<(Option<(usize, u64)>, Option<super::material_gate_views::writer::Snapshot>), String> {
    let gfx = !matches!(stage, RootStage::Compute);
    let (real, views, layout, args, io) = abi_views(this, pso, stage)?;
    super::depth_uav::before_op(this, gfx, layout.as_deref(), &args);
    // (a graphics pipeline may be taken away for the code budget between the caller's look at it
    // and this one: its stand-in has the title's root signature, not the ABI's bound below)
    if gfx && super::real::is_standin_pso(real) {
        return Err("its pipeline is not built (taken away for the code budget, or its build failed)".into());
    }
    let rs = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
    bind_views(this, list, rs, real, views, gfx)?;
    let mut writer = (call == "dispatch").then(|| super::material_gate_views::writer::snapshot(this as usize, real, &io)).flatten();
    note_lineage_uavs(this, (real, real), call, layout.as_deref(), &args);
    if let Some(snapshot) = writer.as_mut().filter(|_| super::material_gate_capture::enabled() && crate::gpu::recomp::pso::material_gate::writer_mode()) {
        if let Some(variant) = super::real::material_gate_variant(real) {
            if let Some(result_va) = snapshot.prepare(&io) {
                set_gate_state(this, variant, result_va);
                return Ok((Some((real, views[1])), writer));
            }
        }
    }
    if call == "dispatch" && super::material_gate_capture::enabled() && !crate::gpu::recomp::pso::material_gate::writer_mode() && super::depth_twin_probe::gate_bound(this as usize) {
        if let Some(variant) = super::real::material_gate_variant(real) {
            if let Some(result_va) = super::depth_twin_probe::prepare_counters(this as usize, || super::material_gate_views::describe(&io)) {
                if super::material_gate_views::writer::enabled() {
                    super::depth_lineage_queue::watch_selected(this as usize, super::depth_lineage_queue::recording(this as usize));
                }
                set_gate_state(this, variant, result_va);
                return Ok((Some((real, views[1])), writer));
            }
        }
    }
    Ok((None, writer))
}

fn note_lineage_uavs(list: *mut c_void, (pso, shader_pso): (usize, usize), call: &'static str, layout: Option<&Layout>, args: &[u8]) {
    let Some(layout) = layout.filter(|l| !l.uavs.is_empty()) else { return };
    if !super::depth_lineage::enabled() || !super::console_heap::has_lineage_uavs() {
        return;
    }
    let mut remaining = 128u32;
    let mut hashes = None;
    for binding in &layout.uavs {
        let UavBinding::Table { param, at, first, count } = *binding else { continue };
        let at = at as usize;
        let Some(bytes) = args.get(at..at + 4) else { continue };
        let table = u32::from_le_bytes(bytes.try_into().unwrap());
        if table == 0 || table % super::console_heap::SLOT != 0 {
            continue;
        }
        let count = count.min(remaining);
        remaining -= count;
        for (slot, index, uav) in super::console_heap::lineage_uavs_in_table(table, first, count) {
            let (graphics, compute) = *hashes.get_or_insert_with(|| (
                super::real::draw_gate_shaders(shader_pso),
                super::dispatch_timing::profile_shader(shader_pso),
            ));
            let kind = match (uav.shadow, uav.plane) {
                (Some(_), _) => "typed_uav_shadow_binding",
                (None, 1) => "typed_uav_null_stencil_binding",
                (None, _) => "typed_uav_null_binding",
            };
            if call == "dispatch" && uav.plane == 0 {
                super::depth_twin_probe::binding(list as usize, uav.owner, uav.shadow, uav.native_format, index + 1, uav.range, compute);
            }
            super::depth_lineage::note_key(uav.owner, list as usize, uav.range, kind, shader_pso, || format!(
                "call={call} pso={pso:#x} source_pso={shader_pso:#x} graphics_source_hashes={graphics:016x?} compute_source_hash={compute:016x?} root={param} table={table:#x} slot={slot} native_index={} shadow={:x?} requested_format={} shadow_format={} plane={}",
                index + 1, uav.shadow, uav.requested_format, uav.native_format, uav.plane,
            ));
        }
        if remaining == 0 {
            break;
        }
    }
}

/// The ABI's root signature `rs`, its two root views (io, res) and the pipeline `pipeline` on
/// `list`, as compute or graphics state.
unsafe fn bind_views(this: *mut c_void, list: &ID3D12GraphicsCommandList, rs: usize, pipeline: usize, views: [u64; 2], gfx: bool) -> Result<(), String> {
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
    super::list_x::ensure_view_heap(this);
    if gfx {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.into_iter().enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.into_iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    }
    super::shader_code::used(this as usize, pipeline);
    let pipeline = pipeline as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&pipeline).ok_or("pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

/// What `bind_abi` binds for the pipeline `pso`: the pipeline (a build in the background waited
/// for) and the GPU addresses of its io (the user SGPRs, the argument table written to unified
/// memory) and of a small res.
unsafe fn abi_views(this: *mut c_void, pso: usize, stage: RootStage) -> Result<(usize, [u64; 2], Option<Arc<Layout>>, Vec<u8>, Vec<u32>), String> {
    let gfx = !matches!(stage, RootStage::Compute);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = super::real::shader_window_base(&windows)?;
    let (layout, args) = with(this, |s| if gfx { (s.layout_gfx.clone(), s.args_gfx.clone()) } else { (s.layout.clone(), s.args.clone()) });
    let limits = if gfx { [0; 3] } else { with(this, |s| std::mem::take(&mut s.limits)) };
    let spread = if gfx { [0; 2] } else { with(this, |s| std::mem::take(&mut s.spread)) };
    // 32-bit console addresses: window 0; pieces of the list's arena (not a 64 KiB grain each)
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
    let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
    let table_at = copy(&args[inline..])?;
    let mut io = match layout.as_ref() {
        Some(l) if gfx => console_list::user_data_gfx(l, &args, table_at as u64),
        Some(l) => console_list::user_data_for(l, user_sgprs, &args, table_at as u64),
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    if matches!(stage, RootStage::Mesh) {
        let ms_inline = layout.as_ref().map_or(0, |l| l.inline_ms as usize).min(args.len());
        let ms_table = if ms_inline == inline { table_at } else { copy(&args[ms_inline..])? };
        io[crate::gpu::recomp::IO_NGG_USER_DISPLACEMENT] = crate::gpu::recomp::NGG_USER_DISPLACEMENT as u32;
        io.extend(console_list::ngg_user_data(ms_inline as u32, &args, ms_table as u64));
        io.resize(crate::gpu::recomp::KERNEL_SGPRS + 24, 0);
    }
    if !gfx && dispatch_snapshot_enabled() {
        with(this, |s| s.last_io = (io[..16].to_vec(), user_sgprs, inline as u32));
    }
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    io[crate::gpu::recomp::IO_MEM_VIEWS] = super::real::mem_views().ok_or("memory windows have no views")?;
    io[crate::gpu::recomp::IO_PHYSICAL_TABLE] = super::physical_directory::recordings::packet_index(this as usize);
    io[crate::gpu::recomp::IO_CARRY_AT] = carry_watch();
    for (k, l) in limits.into_iter().enumerate() {
        io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = l;
    }
    for (k, w) in spread.into_iter().enumerate() {
        io[crate::gpu::recomp::IO_GROUP_SPREAD[k]] = w;
    }
    (io[crate::gpu::recomp::IO_TRIP_WINDOW], io[crate::gpu::recomp::IO_TRIP_AT]) = trip_watch(this, with(this, |s| s.pso), real, gfx);
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let res_at = copy(&[0u8; 256])?;
    note_scalar_probes(this as usize, real, gfx, &io);
    if !gfx {
        note_image_probes(this as usize, real, &io);
    }
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    Ok((real, [va(io_at), va(res_at)], layout, args, io))
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

/// The pipeline `list` has bound (as the title made it; 0: none).
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// `DispatchX` with thread dimensions (flag 0x20; 0x8020 with the wave32 bit): the console
    /// launches ceil(count / numthreads) groups an axis and leaves the lanes past a count that
    /// ends inside a group out; without the flag the counts are groups. With the flag and a
    /// group shape not known nothing is launched (the counts as groups was the bug), and with
    /// `XCL_DISPATCH_THREAD_DIMENSIONS=0` the flags are ignored as before.
    #[test]
    fn dispatch_x_with_thread_dimensions_launches_whole_groups() {
        let line = Some([64, 1, 1]);
        // 640 threads in groups of 64: 10 groups, the last one whole
        assert_eq!(dispatch_groups([640, 1, 1], 0x20, line, true), Some(([10, 1, 1], [0, 0, 0])));
        // 100 threads: 2 groups, and the lanes past 100 to leave out
        assert_eq!(dispatch_groups([100, 1, 1], 0x20, line, true), Some(([2, 1, 1], [100, 0, 0])));
        // one thread past a whole group, one short of one
        assert_eq!(dispatch_groups([65, 1, 1], 0x20, line, true), Some(([2, 1, 1], [65, 0, 0])));
        assert_eq!(dispatch_groups([63, 1, 1], 0x20, line, true), Some(([1, 1, 1], [63, 0, 0])));
        // the wave32 bit (0x8000) changes nothing here
        assert_eq!(dispatch_groups([100, 1, 1], 0x8020, line, true), Some(([2, 1, 1], [100, 0, 0])));
        // a 2-D and a 3-D group shape: each axis by its own size
        assert_eq!(dispatch_groups([20, 16, 3], 0x20, Some([8, 8, 1]), true), Some(([3, 2, 3], [20, 0, 0])));
        assert_eq!(dispatch_groups([16, 9, 7], 0x8020, Some([4, 4, 4]), true), Some(([4, 3, 2], [0, 9, 7])));
        // no threads: no groups (a 0 limit is "no limit", but nothing is launched on that axis)
        assert_eq!(dispatch_groups([0, 1, 1], 0x20, line, true), Some(([0, 1, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups([100, 0, 5], 0x20, Some([8, 8, 1]), true), Some(([13, 0, 5], [100, 0, 0])));
        // a group shape of 0 on an axis counts as 1 (no division by zero)
        assert_eq!(dispatch_groups([5, 1, 1], 0x20, Some([0, 1, 1]), true), Some(([5, 1, 1], [0, 0, 0])));
        // the largest counts: no overflow in the group count
        assert_eq!(dispatch_groups([u32::MAX, 1, 1], 0x20, line, true), Some(([u32::MAX / 64 + 1, 1, 1], [u32::MAX, 0, 0])));
        // group counts (Dispatch, DispatchX without the flag) are launched as they are
        assert_eq!(dispatch_groups([7, 2, 1], 0, line, true), Some(([7, 2, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups([7, 2, 1], 0x8000, line, true), Some(([7, 2, 1], [0, 0, 0])));
        // thread dimensions on a pipeline whose group shape is not known: not launched
        assert_eq!(dispatch_groups([100, 1, 1], 0x20, None, true), None);
        assert_eq!(dispatch_groups([100, 1, 1], 0, None, true), Some(([100, 1, 1], [0, 0, 0])));
        // switched off: the counts are groups whatever the flags
        assert_eq!(dispatch_groups([100, 1, 1], 0x20, line, false), Some(([100, 1, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups([100, 1, 1], 0x8020, None, false), Some(([100, 1, 1], [0, 0, 0])));
    }

    /// A dispatch past 65,535 groups on an axis, spread over x and y: decoded as the compute
    /// prologue decodes io[IO_GROUP_SPREAD], the launched groups run each console group of the
    /// x-y plane once and the rest leave; one that cannot be spread is None.
    #[test]
    fn a_dispatch_past_the_axis_limit_runs_each_console_group_once_spread() {
        for dims in [[129_889, 1, 1], [85_000, 3, 2], [2, 70_000, 1], [115_200, 1, 1]] {
            let (launched, words) = spread_launch(dims).unwrap();
            assert!(launched.iter().all(|&n| n <= 65535), "{dims:?}: {launched:?}");
            assert_eq!((words, launched[2]), ([dims[0], dims[1]], dims[2]));
            let width = crate::gpu::recomp::GROUP_SPREAD_WIDTH;
            let mut seen = vec![false; (dims[0] * dims[1]) as usize];
            for gy in 0..launched[1] {
                for gx in 0..launched[0] {
                    let n = gx + width * gy;
                    if n >= words[0] * words[1] {
                        continue;
                    }
                    let (x, y) = (n % words[0], n / words[0]);
                    let at = (y * dims[0] + x) as usize;
                    assert!(!seen[at], "{dims:?}: group {x},{y} twice");
                    seen[at] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "{dims:?}: a console group never ran");
        }
        assert_eq!(spread_launch([100_000, 0, 1]), Some(([0; 3], [0; 2])));
        assert_eq!(spread_launch([100_000, 1, 70_000]), None);
        assert_eq!(spread_launch([u32::MAX, u32::MAX, 1]), None);
    }

    #[test]
    fn pending_skip_logs_only_declared_uav_slots() {
        let words = [
            2u32, 2, 24, 0, 0, 0,
            4, 0, 48, 0, 0, 60,
            0, 0, 0, 2, 68,
            0, 3, 0, 0, 0, 0,
            1, 1, 0, 0, 0, u32::MAX,
        ];
        let blob: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let layout = console_list::layout_from_blob(&blob).unwrap();
        let args: Vec<_> = [0x9000u32, 1, 0x40].into_iter().flat_map(u32::to_le_bytes).collect();
        let uavs = describe_uavs(&layout, &args, |addr| {
            assert_eq!(addr, 0x40 + 3 * 32, "the preceding SRV range is not captured");
            Some([0x42; 8])
        });
        assert_eq!(uavs.len(), 2);
        assert_eq!(uavs[0], "p0 root UAV 0x100009000");
        assert!(uavs[1].starts_with("p1 UAV table 0x40 slot 3: "));
    }

    #[test]
    fn marker_buffer_probe_checks_descriptor_type_and_bounds() {
        let mut words = [0; 8];
        words[..4].copy_from_slice(&crate::gpu::desc::cbv_words(0x7_1234_0000, 0x100));
        assert_eq!(buffer_dword_address(&words, 0xfc), Some(0x7_1234_00fc));
        assert_eq!(buffer_dword_address(&words, 0x100), None);
        assert_eq!(buffer_dword_address(&words, u32::MAX), None);
        assert_eq!(buffer_dword_address(&[0; 8], 0), None);
        let image = crate::gpu::desc::xcl_image_for_view(1, 22, crate::gpu::desc::ImageType::D3, 4, 4, 4, 1);
        assert_eq!(buffer_dword_address(&image, 0), None);
    }

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect of dispatches waits for its compute pipeline as Dispatch does, past the
    /// draw backstop, and runs it once built; one whose build failed is skipped as Dispatch skips
    /// it. An ExecuteIndirect of draws follows the draw backstop: skipped while its pipeline is
    /// still being built after it, and a failed one goes to the PC call (its stand-in draws
    /// nothing). The gate `execute_indirect` uses after `resolve_wait`.
    #[test]
    fn execute_indirect_waits_for_a_compute_pipeline_and_a_draw_follows_the_backstop() {
        let (compute, compute_failed, draw, draw_failed) = (0x7E57_6011usize, 0x7E57_6012usize, 0x7E57_6013usize, 0x7E57_6014usize);
        let gate = std::sync::Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let held = |g: std::sync::Arc<(Mutex<bool>, std::sync::Condvar)>, real: usize| -> pso_async::Job {
            Box::new(move || {
                if pso_async::preparing() {
                    return 0;
                }
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                real
            })
        };
        pso_async::submit_lazy(compute, true, held(gate.clone(), 0x1241), 0);
        pso_async::submit_lazy(draw, false, held(gate.clone(), 0x1243), 0);
        pso_async::submit_lazy(compute_failed, true, Box::new(|| 0), 0);
        pso_async::submit_lazy(draw_failed, false, Box::new(|| 0), 0);
        real::register_translated_compute_for_test(0x1241);

        pso_async::BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(100)))));
        let t0 = std::time::Instant::now();
        assert_eq!(pso_async::resolve_wait(draw), draw, "a draw is not waited for past the backstop");
        assert!(t0.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(indirect_skip(false, draw), Some(IndirectSkip::Building));

        let waiter = std::thread::spawn(move || {
            pso_async::BACKSTOP.with(|b| b.set(Some(Some(std::time::Duration::from_millis(100)))));
            pso_async::resolve_wait(compute)
        });
        std::thread::sleep(std::time::Duration::from_millis(400));
        assert!(!waiter.is_finished(), "dispatches wait past the backstop");
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let built = waiter.join().unwrap();
        assert_eq!(built, 0x1241);
        assert!(real::is_translated_compute(built), "built: it runs translated, never skipped");

        assert_eq!(pso_async::resolve_wait(compute_failed), compute_failed);
        assert!(!real::is_translated_compute(compute_failed));
        assert_eq!(indirect_skip(true, compute_failed), Some(IndirectSkip::Untranslated), "a failed build is skipped as Dispatch skips it");
        assert_eq!(pso_async::resolve_wait(draw_failed), draw_failed);
        assert_eq!(indirect_skip(false, draw_failed), None, "a failed draw pipeline goes to the PC call");

        real::forget_pso_for_test(0x1241);
        for p in [compute, compute_failed, draw, draw_failed] {
            pso_async::released(p);
        }
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Setting the root signature already set keeps the root arguments (D3D12 leaves them
    /// undefined only when the root signature changes); another one starts from zero.
    #[test]
    fn the_same_root_signature_again_keeps_the_root_arguments() {
        let list = 0x7100_0501usize as *mut c_void;
        let (rs, other) = (0x7100_0510usize, 0x7100_0520usize);
        let blob = test_rts0_blob();
        console_list::note_root_signature(rs, &blob);
        console_list::note_root_signature(other, &blob);
        reset(list, 0x7100_0600, 0);
        let holds = |gfx: bool, v: u32| with(list, |s| if gfx { &s.args_gfx } else { &s.args }.windows(4).any(|w| w == v.to_le_bytes()));
        unsafe {
            set_compute_root_signature(list, rs as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
            set_compute_root_signature(list, rs as *mut c_void);
            set_graphics_root_signature(list, rs as *mut c_void);
        }
        assert!(holds(false, 0xDEAD_BEEF) && holds(true, 0xCAFE_BABE));
        unsafe {
            set_compute_root_signature(list, other as *mut c_void);
            set_graphics_root_signature(list, other as *mut c_void);
        }
        assert!(!holds(false, 0xDEAD_BEEF) && !holds(true, 0xCAFE_BABE));
        assert!(with(list, |s| !s.args.is_empty() && s.args.iter().chain(&s.args_gfx).all(|&b| b == 0)));
        remove_state_for_test(list);
    }

    /// Indirect calls with a mesh signature are known as mesh work, and every skipped mesh call
    /// is counted for the 30 s line.
    #[test]
    fn mesh_signatures_are_known_and_skipped_mesh_calls_counted() {
        let (mesh, dispatch) = (0x7100_0701usize, 0x7100_0702usize);
        super::super::indirect::note_signature(mesh, super::super::indirect::DISPATCH_MESH, 12, true);
        super::super::indirect::note_signature(dispatch, super::super::indirect::DISPATCH, 12, true);
        assert!(super::super::indirect::dispatches_mesh(mesh) && !super::super::indirect::dispatches_mesh(dispatch));
        assert_eq!(super::super::indirect::stride_of(mesh), Some(12));
        let before = MESH_CALLS[MESH_SKIPPED].load(Ordering::Relaxed);
        mesh_skipped(0x7100_0711usize as *mut c_void, 0x7100_0710, "a test call", "a test");
        assert_eq!(MESH_CALLS[MESH_SKIPPED].load(Ordering::Relaxed), before + 1);
        assert!(indirect_calls().contains("as host mesh shaders"));
    }

    /// The tripwire's readback: a dispatch's slot is counted once into its pipeline's totals,
    /// the ring clears a slot before the next dispatch takes it, and a graphics pipeline keeps
    /// one slot whose counts add up.
    #[test]
    fn tripped_guards_are_counted_by_pipeline_and_dispatch() {
        let mut words = vec![0u32; 4 * (TRIP_RING + TRIP_GFX)];
        let mut t = Trips::over((0..(TRIP_RING + TRIP_GFX) / TRIP_PIECE).map(|p| words.as_mut_ptr() as usize + 16 * TRIP_PIECE * p).collect());
        let set = |t: &Trips, i: usize, k: usize, n: u32| unsafe { std::ptr::write_volatile((t.at(i) + 4 * k) as *mut u32, n) };
        let (cs, vs) = (0x7100_0910usize, 0x7100_0920usize);
        let first = t.take(cs, Some((990_002, 3, 0)), false, None);
        assert_eq!((first, t.take(cs, Some((990_002, 3, 1)), false, None)), (0, 1));
        set(&t, 0, 0, 640);
        set(&t, 0, 3, 2);
        let lines = t.lines(8);
        assert!(lines[0].contains("loop guard 640, compare-exchange retries 2 in 1 pipelines (2 dispatches"), "{lines:?}");
        assert!(lines[1].contains("in 1 dispatches with pipeline 0x71000910"), "{lines:?}");
        assert_eq!(t.lines(8), lines, "a slot already counted is not counted again");

        let g = t.take(vs, None, true, None);
        assert_eq!((g, t.take(vs, None, true, None)), (TRIP_RING, TRIP_RING));
        set(&t, g, 1, 5);
        t.lines(0);
        set(&t, g, 1, 9);
        assert!(t.lines(8).iter().any(|l| l.contains("wave64 meet 9 in 2 looks at draws with pipeline 0x71000920")), "{:?}", t.lines(8));

        // the ring comes round: the slot is cleared for its next dispatch, its counts are kept
        t.next = TRIP_RING;
        assert_eq!(t.take(0x7100_0930, None, false, None), 0);
        assert_eq!(t.words(0), [0; 4]);
        assert_eq!(t.totals[&cs], ([640, 0, 0, 2], 1));
    }

    /// The snapshot's account of a dispatch: its groups, its root arguments, and for each buffer
    /// it was bound to whether a skipped dispatch or a skipped mesh call was bound to it too
    /// and which dispatch before it was.
    #[test]
    fn a_dispatch_is_described_with_what_its_buffers_were_bound_to_before() {
        let rs = 0x7100_0810usize;
        console_list::note_root_signature(rs, &test_rts0_blob());
        let layout = console_list::layout_of(rs).unwrap();
        let (slot, skipped_pso, mesh_pso) = (990_001usize, 0x7100_0820usize, 0x7100_0830usize);
        let (a, b, c) = (0x7_1000_0000u64, 0x7_2000_0000u64, 0x7_3000_0000u64);
        let bound = |pso: usize, op: usize, buffers: &[u64]| BoundDispatch {
            title_pso: pso,
            layout: layout.clone(),
            args: vec![0xEF, 0xBE, 0xAD, 0xDE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            at: Some((slot, 7, op)),
            groups: Some([8, 4, 1]),
            buffers: buffers.iter().map(|&base| (base, 0x100, 0, "p1 table 0x40 slot 0".to_string())).collect(),
            images: vec![BoundImage {
                table: 0x40,
                param: 1,
                slot: 5,
                srv: 10,
                resource: Some(0x1234),
                view: crate::gpu::desc::image(&crate::gpu::desc::xcl_image_for_view(10, 22, crate::gpu::desc::ImageType::D3, 8, 4, 2, 1)).unwrap(),
            }],
            io: (vec![0x5000, 7, 0, 0, 0xDEAD_BEEF, 0, 0, 0], Some(8), 16),
        };
        {
            let mut d = DISPATCH_BINDINGS.lock().unwrap();
            d.push_back(bound(0x7100_0840, 0, &[a]));
            d.push_back(bound(0x7100_0850, 1, &[a, b, c]));
        }
        super::super::indirect::skipped_producer(skipped_pso, &[(b + 0x80, 0x40)]);
        REFUSED_BOUND.lock().unwrap().push((c, 0x100, mesh_pso));

        let lines = describe_dispatch(slot, 7, 1);
        assert!(lines.iter().any(|s| s.contains("p1 table 0x40 slot 5 image D3 8x4x2") && s.contains("SRV index 10, companion UAV index 11, resource 0x1234")), "{lines:?}");
        assert!(lines[0].contains("thread groups 8 x 4 x 1"), "{lines:?}");
        assert!(lines[1].contains("io words 0-15 as bound: 5000 7 0 0 deadbeef 0 0 0; the shader has 8 user registers, 16 of the 16 argument bytes are in registers"), "{lines:?}");
        assert!(lines[2].contains("root signature layout: p0 4 constants, bytes 0..16 in s4"), "{lines:?}");
        assert!(lines[3].contains("p0 constants: deadbeef 0 0 0"), "{lines:?}");
        let of = |base: u64| lines.iter().find(|l| l.contains(&format!("{base:#x}+0x100"))).unwrap_or_else(|| panic!("{lines:?}"));
        assert!(of(a).contains("bound before by") && of(a).contains("1 dispatches earlier (op 1 of slot 990001)") && !of(a).contains("NEVER WRITTEN"), "{}", of(a));
        assert!(of(b).contains("NEVER WRITTEN? a skipped dispatch was bound to") && of(b).contains("no dispatch kept before"), "{}", of(b));
        assert!(of(c).contains("a skipped mesh call was bound to it too (pipeline 0x71000830)"), "{}", of(c));
        assert_eq!(describe_dispatch(slot, 8, 1).len(), 1, "another generation of the slot is another recording");

        super::super::indirect::produced(skipped_pso);
        DISPATCH_BINDINGS.lock().unwrap().retain(|e| e.at.is_none_or(|at| at.0 != slot));
        REFUSED_BOUND.lock().unwrap().retain(|e| e.2 != mesh_pso);
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// `XCL_DEFER_TO_SUBMIT=0` keeps the skip: a dispatch whose pipeline is still being built is
    /// skipped and counted at once, nothing reaches the PC list and nothing waits. A list that is
    /// not a title list has nowhere to put it off to: skipped the same way with it on.
    #[test]
    fn with_defer_to_submit_off_a_dispatch_whose_pipeline_is_building_is_skipped() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;
        let standin = 0x7200_0D01usize;
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let g = gate.clone();
        pso_async::submit(standin, true, Box::new(move || {
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0
        }));
        reset(list_ptr, 0x7200_0100, standin);
        pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(true));
        let mut skips = Vec::new();
        for on in [false, true] {
            pso_async::set_defer_for_test(Some(on));
            let before = counts_snapshot();
            let t = std::time::Instant::now();
            unsafe { dispatch(list_ptr, 1, 1, 1) };
            let after = counts_snapshot();
            skips.push((after[DISPATCHES_SKIPPED] - before[DISPATCHES_SKIPPED], after[DISPATCHES] - before[DISPATCHES], t.elapsed()));
        }
        pso_async::set_defer_for_test(None);
        pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(false));
        let pending = pso_async::is_pending(standin);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let t = std::time::Instant::now();
        while pso_async::is_pending(standin) && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let _ = pso_async::released(standin);
        forget(list_ptr);
        assert!(pending, "the dispatch waited for the build");
        for (skipped, ran, took) in skips {
            assert_eq!((skipped, ran), (1, 0));
            assert!(took < std::time::Duration::from_millis(500), "the dispatch waited {took:?}");
        }
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        let pending = 0x7200_0004usize;
        let finished = 0x7200_0005usize;
        let sig_compute = 0x7200_00F1usize;
        let shader = 0x7200_4321u64;
        let (finish, gate) = std::sync::mpsc::channel();
        let gate = Mutex::new(gate);
        pso_async::submit_lazy_described(pending, true, Box::new(move || {
            let _ = gate.lock().unwrap().recv();
            finished
        }), 0, format!("CS {shader:016x}"), 0, 0);
        super::super::indirect::note_signature(sig_compute, super::super::indirect::DISPATCH, 12, true);
        reset(list_ptr, 0x7200_0100, pending);
        pso_async::COMPUTE_SKIP_LIMIT.with(|limit| {
            limit.set(Some(std::time::Duration::ZERO));
            assert_eq!(pso_async::resolve_for_bind(pending), pending, "state bind must not block");
            unsafe {
                dispatch(list_ptr, 1, 1, 1);
                execute_indirect(list_ptr, sig_compute, 1, 0, 0, 0, 0);
            }
            limit.set(None);
        });
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1);
        assert_eq!(PENDING_COMPUTE_SKIPS.lock().unwrap().as_ref().unwrap().get(&shader), Some(&2));
        finish.send(()).unwrap();
        assert_eq!(pso_async::resolve_wait(pending), finished);
        assert_eq!(pso_async::released(pending), finished);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
    }

    /// The stand-in for the indirect light (`title_gta6_ambient`) follows a dispatch that was recorded
    /// and no other: one skipped because its pipeline failed to build, or is still building, is
    /// not followed, and nothing of the stand-in reaches the list.
    #[test]
    fn a_skipped_dispatch_is_not_followed_by_the_stand_in_for_the_indirect_light() {
        use super::super::title_gta6_ambient as ambient;
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;
        let name = b"CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped";

        // a pipeline whose build failed: its dispatch is skipped
        let failed = 0x7200_0E01usize;
        pso_async::submit(failed, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(failed), failed);
        ambient::made(failed, name);
        reset(list_ptr, 0x7200_0100, failed);
        let before = counts_snapshot();
        unsafe { dispatch(list_ptr, 160, 90, 1) };
        let after = counts_snapshot();
        // (the counts are every list's: other tests skip dispatches at the same time)
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(ambient::followed_for_test(failed), 0, "a dispatch that was not recorded is not followed");

        // a pipeline still building, with nowhere to put its dispatch off to: skipped too
        let pending = 0x7200_0E02usize;
        let (finish, gate) = std::sync::mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        pso_async::submit(pending, true, Box::new(move || {
            let _ = gate.lock().unwrap().recv();
            0
        }));
        ambient::made(pending, name);
        reset(list_ptr, 0x7200_0100, pending);
        pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(true));
        let before = counts_snapshot();
        unsafe { dispatch(list_ptr, 160, 90, 1) };
        let after = counts_snapshot();
        pso_async::SKIP_UNTIL_BUILT.with(|s| s.set(false));
        // (the counts are every list's: other tests skip dispatches at the same time)
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(ambient::followed_for_test(pending), 0, "nor one whose pipeline is still building");
        finish.send(()).unwrap();
        assert_eq!(pso_async::resolve_wait(pending), pending);

        // what the recorded dispatch's arm does: the pipeline's role is found and the stand-in asked
        // (with its switch unset, and no root signature on this list, it records nothing)
        unsafe { ambient::after_dispatch(list_ptr, failed) };
        assert_eq!(ambient::followed_for_test(failed), 1);
        ambient::gone(failed);
        unsafe { ambient::after_dispatch(list_ptr, failed) };
        assert_eq!(ambient::followed_for_test(failed), 1, "a released pipeline has no role");
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);

        ambient::gone(pending);
        forget(list_ptr);
        let _ = pso_async::released(failed);
        let _ = pso_async::released(pending);
    }

    unsafe extern "system" fn fake_dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
        let f = &*(this as *const FakeAdoptedList);
        f.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        f.set_compute_uav_calls.store((x as usize) << 32 | (y as usize) << 16 | z as usize, Ordering::Relaxed);
    }

    /// How mesh calls are routed: DispatchMesh with a PC pipeline is the PC call with the
    /// title's counts; a pipeline with no mesh shader, a draw with a mesh pipeline and an
    /// indirect call whose signature and pipeline disagree are skipped and counted, and never
    /// reach the PC list; DispatchMeshX from memory is skipped.
    #[test]
    fn mesh_calls_reach_the_pc_list_only_with_a_mesh_pipeline() {
        let slot = super::super::layout::slot(super::super::layout::find("ID3D12GraphicsCommandList6").unwrap(), "DispatchMesh").unwrap();
        let mut orig_vt = [0usize; 96];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        orig_vt[slot] = fake_dispatch_mesh as usize;
        let mut ext_vt = vec![0usize; 97];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            // SAFETY: as in `async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect`
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let skipped = || MESH_CALLS[MESH_SKIPPED].load(Ordering::Relaxed);
        let pc_calls = || (fake.dispatch_calls.load(Ordering::Relaxed), fake.execute_indirect_calls.load(Ordering::Relaxed));
        let (pc_pso, plain, native) = (0x7300_0001usize, 0x7300_0002usize, 0x7300_0003usize);
        real::register_translated_graphics_for_test(plain);
        real::register_translated_graphics_for_test(native);
        real::register_native_mesh_for_test(native);
        let (mesh_sig, draw_sig) = (0x7300_00F1usize, 0x7300_00F2usize);
        super::super::indirect::note_signature(mesh_sig, super::super::indirect::DISPATCH_MESH, 12, true);
        super::super::indirect::note_signature(draw_sig, super::super::indirect::DRAW, 16, true);

        // a PC pipeline: the PC DispatchMesh with the title's three counts
        reset(list, 0x7300_0100, pc_pso);
        unsafe { dispatch_mesh(list, 5, 2, 3) };
        assert_eq!(pc_calls(), (1, 0));
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 5 << 32 | 2 << 16 | 3);

        // a translated pipeline with no mesh shader: skipped
        reset(list, 0x7300_0100, plain);
        let before = skipped();
        unsafe { dispatch_mesh(list, 5, 1, 1) };
        assert!(skipped() > before);
        assert_eq!(pc_calls(), (1, 0));
        // … and with a mesh signature
        let before = skipped();
        unsafe { execute_indirect(list, mesh_sig, 1, 0, 0, 0, 0) };
        assert!(skipped() > before);
        assert_eq!(pc_calls(), (1, 0));

        // a mesh pipeline: a draw and an indirect draw are skipped
        reset(list, 0x7300_0100, native);
        let before = skipped();
        assert!(unsafe { graphics_bound(list) }.is_none());
        unsafe { execute_indirect(list, draw_sig, 1, 0, 0, 0, 0) };
        assert!(skipped() >= before + 2);
        assert_eq!(pc_calls(), (1, 0));
        // more groups than one axis takes (y and z fold into x), and DispatchMeshX with its
        // counts in memory
        let before = skipped();
        unsafe { dispatch_mesh(list, 65535, 2, 1) };
        unsafe { dispatch_mesh_x(list, 1, 0, 0x1000, 0) };
        assert!(skipped() >= before + 2);
        assert_eq!(pc_calls(), (1, 0));
        assert!(indirect_calls().contains("skipped"));

        forget(list);
        real::forget_pso_for_test(plain);
        real::forget_pso_for_test(native);
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    /// play-93b's pattern: a skipped culling pass binds the shared argument buffer read-only (a
    /// root SRV, an SRV range), and among its UAVs the title's view of all of its memory, a root
    /// UAV and its own output. Only its root UAV and its output are kept as unwritten: indirect
    /// calls reading the shared buffer or elsewhere in memory run, one reading its output is
    /// refused until a built pass with a UAV over it runs, and the view of all memory neither
    /// marks nor clears.
    #[test]
    fn a_skipped_dispatch_keeps_only_what_its_uavs_can_write() {
        const SHARED: u64 = 0x4_20b1_0000;
        const OUT: u64 = 0x5_1234_0000;
        const ROOT_OUT: u64 = 0x5_2000_0000;
        const WHOLE: u64 = 0x3d00_0000;
        const WHOLE_BYTES: u64 = 0xf665_d000_000 - WHOLE;
        let words = [
            2u32, 3, 24, 0, 0, 0,
            3, 0, 60, 4, 0, 72, 0, 0, 84,
            0, 0, 0, 1, 0, 0,
            2, 92,
            0, 3, 0, 0, 0, 0,
            1, 2, 0, 0, 0, u32::MAX,
        ];
        let blob: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let layout = console_list::layout_from_blob(&blob).unwrap();
        let args = |root_srv: u64, root_uav: u64, table: u32| -> Vec<u8> { [root_srv.to_le_bytes(), root_uav.to_le_bytes()].concat().into_iter().chain(table.to_le_bytes()).collect() };
        let buf = |base: u64, bytes: u64, stride: u64| ([base as u32, (base >> 32) as u32 & 0xffff | (stride as u32) << 16, (bytes / stride) as u32, 0, 0, 0, 0, 0], 0usize);
        let empty = ([0u32; 8], 0usize);
        let (producer, sig) = (0x7E57_9300usize, 0x7E57_9302usize);

        let writes = uav_writes(&layout, &args(SHARED + 0x200, ROOT_OUT, 0x40), |addr| {
            assert!(addr >= 0x40 + 3 * 32, "the SRV range is not read: {addr:#x}");
            match (addr - 0x40) / 32 {
                3 => Some(buf(WHOLE, WHOLE_BYTES, 0x1000)),
                4 => Some(buf(OUT, 0x1000, 4)),
                _ => None,
            }
        });
        assert_eq!(writes.iter().map(|w| (w.0, w.1)).collect::<Vec<_>>(), [(ROOT_OUT, 16 << 20), (WHOLE, WHOLE_BYTES), (OUT, 0x1000)]);
        record_skipped_writes(producer, producer, writes);
        super::super::indirect::note_signature(sig, super::super::indirect::DISPATCH, 12, true);
        let check = |at: u64| super::super::indirect::check(sig, 1, Some(at), None);
        assert!(check(SHARED + 0x200).is_ok(), "a buffer it only reads");
        assert!(check(0x6_0000_0000).is_ok(), "elsewhere in the view of all memory");
        assert!(check(OUT + 0x10).unwrap_err().contains(&format!("{producer:#x}")));
        assert!(check(ROOT_OUT + 0x10).is_err());

        let all = uav_writes(&layout, &args(0, 0, 0x40), |addr| Some(if (addr - 0x40) / 32 == 3 { buf(WHOLE, WHOLE_BYTES, 0x1000) } else { empty }));
        super::super::indirect::written(&write_ranges(&all));
        assert!(check(OUT + 0x10).is_err(), "a write through the view of all memory cannot be placed");
        let culled = uav_writes(&layout, &args(0, 0, 0x40), |addr| Some(if (addr - 0x40) / 32 == 4 { buf(OUT, 0x1000, 4) } else { empty }));
        super::super::indirect::written(&write_ranges(&culled));
        assert!(check(OUT + 0x10).is_ok(), "the built culling pass wrote its output");
        assert!(check(ROOT_OUT + 0x10).is_err(), "the root UAV it did not cover");
        super::super::indirect::produced(producer);
        assert!(check(ROOT_OUT + 0x10).is_ok());
    }

    /// play-96's pattern: skipped passes each bind their own output as a root UAV and the title's
    /// 174 MiB pool of indirect arguments through slot 185 of a global table. While fewer than
    /// SHARED_BINDERS pipelines bind the pool it is kept as unwritten like any buffer; from then
    /// on it is a shared pool: its marks go, later skips do not mark it and a dispatch that runs
    /// does not clear through it, while each pass's own output stays kept until it runs.
    #[test]
    fn a_pool_many_pipelines_bind_through_a_global_table_is_neither_kept_nor_cleared() {
        use super::super::indirect::{self as ind, SHARED_BINDERS};
        const POOL: u64 = 0x6_70b1_0000;
        const POOL_BYTES: u64 = 0xae0_0000;
        const OUT: u64 = 0x6_9000_0000;
        const TABLE: u32 = 0x4000;
        let words = [
            2u32, 2, 24, 0, 0, 0,
            4, 0, 48, 0, 0, 60,
            1, 0, 0,
            1, 68,
            1, 1, 0, 0, 0, 185,
        ];
        let blob: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let layout = console_list::layout_from_blob(&blob).unwrap();
        let args = |out: u64| -> Vec<u8> { out.to_le_bytes().into_iter().chain(TABLE.to_le_bytes()).collect() };
        let pool = |addr: u32| {
            assert_eq!(addr, TABLE + 185 * 32, "only slot 185 of the global table is a UAV");
            Some(([POOL as u32, (POOL >> 32) as u32 & 0xffff | 4 << 16, (POOL_BYTES / 4) as u32, 0, 0, 0, 0, 0], 0usize))
        };
        let sig = 0x7E57_9600usize;
        ind::note_signature(sig, ind::DISPATCH, 12, true);
        let check = |at: u64| ind::check(sig, 1, Some(at), None);
        let producer = |k: usize| 0x7E57_9610usize + k;
        let out = |k: usize| OUT + 0x10_0000 * k as u64;

        for k in 0..SHARED_BINDERS - 1 {
            let writes = uav_writes(&layout, &args(out(k)), pool);
            assert_eq!(writes.iter().map(|w| (w.0, w.1)).collect::<Vec<_>>(), [(out(k), 16 << 20), (POOL, POOL_BYTES)]);
            record_skipped_writes(producer(k), producer(k), writes);
        }
        assert!(check(POOL + 0x5000).is_err(), "bound by fewer than SHARED_BINDERS pipelines: kept like any buffer");
        record_skipped_writes(producer(SHARED_BINDERS - 1), producer(SHARED_BINDERS - 1), uav_writes(&layout, &args(out(SHARED_BINDERS - 1)), pool));
        assert!(check(POOL + 0x5000).is_ok(), "a shared pool is not kept");
        assert!(check(POOL).is_ok() && check(POOL + POOL_BYTES - 12).is_ok());
        for k in 0..SHARED_BINDERS {
            assert!(check(out(k) + 0x10).unwrap_err().contains(&format!("{:#x}", producer(k))), "pass {k}'s own output is still kept");
        }
        record_skipped_writes(producer(0), producer(0), uav_writes(&layout, &args(out(0)), pool));
        assert!(check(POOL + 0x5000).is_ok(), "a later skip does not mark the pool again");

        ind::written(&write_ranges(&uav_writes(&layout, &args(0x6_f000_0000), pool)));
        assert!(check(out(1) + 0x10).is_err(), "a write through the pool cannot be placed");
        for k in 0..SHARED_BINDERS {
            ind::produced(producer(k));
        }
        assert!(check(out(1) + 0x10).is_ok());
    }
}
