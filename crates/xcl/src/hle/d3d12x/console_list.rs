//! Console-shaped command lists and pipeline states.
//!
//! D3D12.X code compiled into a title writes command records straight into the command list
//! object (HANDOFF: "D3D12.X command list format"): +0x18 write pointer, +0x28 / +0x30
//! bounds, +0x344 + 4·i root parameter headers (filled by SetComputeRootSignature), and it
//! copies 28 bytes from +0x10 of the pipeline state object into the stream. So the title gets
//! XCL objects with that layout; the records are turned into PC commands at `close`:
//! each Dispatch gets the root arguments as the console passes them (`user_data`: the ones
//! that fit in user SGPRs s4…s15, the rest in a table in unified memory that s[0:1] points at)
//! and runs the translated pipeline on XCL's shader ABI.
//!
//! Compute only, and opt-in (`XCL_CONSOLE_LISTS=1`) until the graphics records are known.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::core::GUID;

use crate::util::{HRESULT, S_OK};

const DLL: &str = "d3d12_xs.dll";
use super::list_x::texture_ops;
/// stream markers of our own (bits 31:30 = 10: neither a parameter header nor a dispatch)
const PIPE_MARKER: u32 = 0x8C0F_E002;
const RS_MARKER: u32 = 0x8C0F_E003;
const BARRIER_MARKER: u32 = 0x8C0F_E005;
const HEAPS_MARKER: u32 = 0x8C0F_E006;
const TRANSITIONS_MARKER: u32 = 0x8C0F_E007;
const COPY_BUFFER_MARKER: u32 = 0x8C0F_E008;
const CLEAR_UAV_MARKER: u32 = 0x8C0F_E009;
const COPY_TEXTURE_MARKER: u32 = 0x8C0F_E00A;
const INDIRECT_MARKER: u32 = 0x8C0F_E00B;
const OWNED_COPY_TEXTURE_MARKER: u32 = 0x8C0F_E00C;
/// root parameter header: 0xC0000000 | i << 10 (bits 31:30 set: the inline code leaves its
/// dirty flag alone); the inline code adds the constant offset · 4 (bits 9:2) and
/// (count − 1) << 16
const PARAM_BASE: u32 = 0xC000_0000;
const DISPATCH: u32 = 0x4002_C400;
const LIST_SLOTS: usize = 600;
const PSO_SLOTS: usize = 64;
const CHUNK: usize = 1 << 20;
/// room past the visible end: inline writers check `write < end` and then write a record
const SLACK: usize = 4096;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("XCL_CONSOLE_LISTS").is_some())
}

// ---------------------------------------------------------------------------------------
// root signature layouts (where each parameter sits in the root argument table)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// `count` 32-bit constants at byte `at`
    Constants { at: u32, count: u32 },
    /// root CBV / SRV / UAV: 64-bit GPU address at `at`
    Descriptor { at: u32 },
    /// descriptor table: 32-bit GPU address of its first descriptor (console shaders load
    /// it with the high half 0; amd_fidelityfx_upscaler_xbox.dll writes 1 word)
    Table { at: u32 },
}

/// Where a root signature's arguments go: all of them packed in order (`at`, `size` bytes);
/// the first `inline` bytes of that are passed in user SGPRs (compute from s4, graphics
/// stages from s8: `inline_gfx` bytes), the rest is the table.
#[derive(Debug, Default)]
pub struct Layout {
    pub params: Vec<Param>,
    pub size: u32,
    pub inline: u32,
    pub inline_gfx: u32,
    pub inline_ms: u32,
    pub(super) uavs: Vec<UavBinding>,
    /// for each parameter that is a descriptor table, its slots: where the last of its ranges
    /// ends (u32::MAX: a range without end, or ranges not read); 0 for another parameter
    pub(super) tables: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UavBinding {
    Root { param: usize, at: u32 },
    Table { param: usize, at: u32, first: u32, count: u32 },
}

/// User SGPRs the console's compiler gives root arguments: compute s4…s15 (s0…s3 are the
/// driver's), graphics stages (VS, PS) s8…s31 (s0…s7: the driver's, s[4:5] of a VS the
/// fetch shader) – both measured on its output (`tests/console_shaders.rs`).
const INLINE_SGPRS: u32 = 12;
const INLINE_SGPRS_GFX: u32 = 24;
const FIRST_SGPR: usize = 4;
const FIRST_SGPR_GFX: usize = 8;

/// Layout from a serialized root signature: a DXBC container with RTS0 or a bare RTS0.
pub fn layout_from_blob(blob: &[u8]) -> Option<Layout> {
    let rts0 = match crate::gpu::shader::container_parts(blob) {
        Some(parts) => {
            let p = parts.into_iter().find(|p| p.fourcc == "RTS0")?;
            blob.get(p.offset..p.offset + p.size)?.to_vec()
        }
        None => blob.to_vec(),
    };
    let r = |o: usize| rts0.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (n, poff) = (r(4)? as usize, r(8)? as usize);
    let range_words = match r(0) {
        Some(1) => Some(5usize),
        Some(2 | 3) => Some(6usize),
        _ => None,
    };
    let mut l = Layout::default();
    // the parameters go into user SGPRs in order while each fits whole; from the first one
    // that does not, the rest go to the table (packed from its start) – as the console's
    // compiler expects them (HANDOFF: root argument binding)
    let (mut spilled, mut spilled_gfx, mut spilled_ms) = (false, false, false);
    for k in 0..n {
        let (ty, pay) = (r(poff + 12 * k)?, r(poff + 12 * k + 8)? as usize);
        let p = match ty {
            1 => Param::Constants { at: l.size, count: r(pay + 8)? },
            2..=4 => Param::Descriptor { at: l.size },
            _ => Param::Table { at: l.size },
        };
        let mut slots = if matches!(p, Param::Table { .. }) { u32::MAX } else { 0 };
        if ty == 4 {
            l.uavs.push(UavBinding::Root { param: k, at: l.size });
        } else if let (0, Some(words), Some(count), Some(off)) = (ty, range_words, r(pay), r(pay + 4)) {
            let mut append = 0u32;
            slots = 0;
            for i in 0..count as usize {
                let at = off as usize + i * words * 4;
                let (Some(kind), Some(count), Some(first)) = (r(at), r(at + 4), r(at + (words - 1) * 4)) else {
                    slots = u32::MAX;
                    break;
                };
                let first = if first == u32::MAX { append } else { first };
                append = first.saturating_add(count);
                slots = slots.max(append);
                if kind == 1 && count != 0 {
                    l.uavs.push(UavBinding::Table { param: k, at: l.size, first, count });
                }
            }
        }
        l.tables.push(slots);
        let bytes = match p {
            Param::Constants { count, .. } => 4 * count,
            Param::Table { .. } => 4,
            Param::Descriptor { .. } => 8,
        };
        spilled |= l.inline + bytes > 4 * INLINE_SGPRS;
        if !spilled {
            l.inline += bytes;
        }
        spilled_gfx |= l.inline_gfx + bytes > 4 * INLINE_SGPRS_GFX;
        if !spilled_gfx {
            l.inline_gfx += bytes;
        }
        spilled_ms |= l.inline_ms + bytes > 16 * 4;
        if !spilled_ms {
            l.inline_ms += bytes;
        }
        l.size += bytes;
        l.params.push(p);
    }
    Some(l)
}

/// The user SGPRs of a dispatch: s[0:1] = `table_at` (where the bytes of `args` past the
/// inline part are), from s4 the inline part of `args` (all arguments packed by `l`).
pub fn user_data(l: &Layout, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR, l.inline, args, table_at)
}

/// Bytes of a compute dispatch's arguments that go in user SGPRs: the layout's rule, but
/// never more than the shader itself has registers for. `user_sgprs` is the shader's own
/// count (`real::pso_user_sgprs`; None = not known, the rule alone). The console's newer
/// compiler follows the rule; XboxBvhBuilder.dll's shaders were built with four user SGPRs
/// whatever their root signature (s4 is the group id) and take every argument from the
/// table, also where the first ones would fit. Parameters stay whole: the inline part ends
/// where the last parameter that fits the shader's registers ends.
pub fn inline_for(l: &Layout, user_sgprs: Option<u32>) -> u32 {
    let Some(user) = user_sgprs else { return l.inline };
    let has = 4 * user.saturating_sub(FIRST_SGPR as u32);
    if l.inline <= has {
        return l.inline;
    }
    let end = |p: &Param| match *p {
        Param::Constants { at, count } => at + 4 * count,
        Param::Descriptor { at } => at + 8,
        Param::Table { at } => at + 4,
    };
    l.params.iter().map(end).take_while(|&e| e <= has).last().unwrap_or(0)
}

/// `user_data` for a shader with `user_sgprs` user SGPRs (`inline_for`).
pub fn user_data_for(l: &Layout, user_sgprs: Option<u32>, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR, inline_for(l, user_sgprs), args, table_at)
}

/// `user_data` of a draw: the vertex and pixel shader's user SGPRs (inline from s8).
pub fn user_data_gfx(l: &Layout, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR_GFX, l.inline_gfx, args, table_at)
}

pub(super) fn ngg_user_data(inline: u32, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR_GFX, inline, args, table_at)
}

fn sgprs(first: usize, inline: u32, args: &[u8], table_at: u64) -> Vec<u32> {
    let mut s = vec![0u32; first];
    s[0] = table_at as u32;
    s[1] = (table_at >> 32) as u32;
    let inline = &args[..(inline as usize).min(args.len())];
    s.extend(inline.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }));
    s
}

static LAYOUTS: Mutex<Option<HashMap<usize, Arc<Layout>>>> = Mutex::new(None);

/// A root signature was created from `blob`: remember its argument layout.
pub fn note_root_signature(rs: usize, blob: &[u8]) {
    if let Some(l) = layout_from_blob(blob) {
        LAYOUTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(rs, Arc::new(l));
    }
}

pub(super) fn layout_of(rs: usize) -> Option<Arc<Layout>> {
    LAYOUTS.lock().unwrap().as_ref()?.get(&rs).cloned()
}

// ---------------------------------------------------------------------------------------
// pipeline states
// ---------------------------------------------------------------------------------------

/// +0x10..+0x2B is the pipeline packet the inline code copies into the stream.
#[repr(C)]
struct ConsolePso {
    vtable: *const usize,
    refs: AtomicU32,
    _pad: u32,
    packet: [u32; 4], // marker, 0, self (64-bit)
    flags: u64,       // bit 20 is copied to list +0x141
    tail: u32,
    _pad2: u32,
    pc: usize,
    translated: bool,
    /// root signature of the translated pipeline (static samplers; 0 = the plain ABI one)
    rs: usize,
}

fn pc_vtable_call(obj: usize, slot: usize) -> usize {
    unsafe { *(*(obj as *const *const usize)).add(slot) }
}

unsafe extern "system" fn pso_qi(this: *mut c_void, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    *ppv = this;
    pso_addref(this);
    S_OK
}
unsafe extern "system" fn pso_addref(this: *mut c_void) -> u32 {
    (*(this as *mut ConsolePso)).refs.fetch_add(1, Ordering::AcqRel) + 1
}
unsafe extern "system" fn pso_release(this: *mut c_void) -> u32 {
    let p = this as *mut ConsolePso;
    let n = (*p).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        let release: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(pc_vtable_call((*p).pc, 2));
        release((*p).pc);
        drop(Box::from_raw(p));
    }
    n
}
/// ID3D12Object / DeviceChild / PipelineState methods 3..=8 go to the PC object
macro_rules! pso_forward {
    ($($name:ident = $slot:literal),*) => {$(
        unsafe extern "system" fn $name(this: *mut c_void, a: usize, b: usize, c: usize) -> HRESULT {
            let pc = (*(this as *mut ConsolePso)).pc;
            let f: unsafe extern "system" fn(usize, usize, usize, usize) -> HRESULT = std::mem::transmute(pc_vtable_call(pc, $slot));
            f(pc, a, b, c)
        }
    )*};
}
pso_forward!(pso_3 = 3, pso_4 = 4, pso_5 = 5, pso_6 = 6, pso_7 = 7, pso_8 = 8);

fn pso_vtable() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        let mut v: Vec<usize> = (0..PSO_SLOTS).map(|i| crate::stubs::make_stub_info(DLL, &format!("ID3D12PipelineState(console)::#{i}"), 0).0).collect();
        let known: [usize; 9] = [pso_qi as usize, pso_addref as usize, pso_release as usize, pso_3 as usize, pso_4 as usize, pso_5 as usize, pso_6 as usize, pso_7 as usize, pso_8 as usize];
        v[..9].copy_from_slice(&known);
        Box::leak(v.into_boxed_slice()).as_ptr() as usize
    })
}

/// Wrap a PC pipeline state (its reference moves into the wrapper) for console code.
pub unsafe fn wrap_pso(pc: *mut c_void, translated: bool, rs: usize) -> *mut c_void {
    let p = Box::into_raw(Box::new(ConsolePso {
        vtable: pso_vtable() as *const usize,
        refs: AtomicU32::new(1),
        _pad: 0,
        packet: [PIPE_MARKER, 0, 0, 0],
        flags: 0,
        tail: 0,
        _pad2: 0,
        pc: pc as usize,
        translated,
        rs,
    }));
    (*p).packet[2] = p as usize as u32;
    (*p).packet[3] = (p as usize >> 32) as u32;
    p as *mut c_void
}

// ---------------------------------------------------------------------------------------
// command lists
// ---------------------------------------------------------------------------------------

const F_WRITE: usize = 0x18;
const F_DISPATCH_END: usize = 0x28;
const F_END: usize = 0x30;
const F_HEADERS: usize = 0x344;
const MAX_PARAMS: usize = 64;
const OBJECT_BYTES: usize = 0x1000;

struct ListState {
    indirect: Vec<Arc<PacketIndirect>>,
    pc: usize,
    /// allocator the current recording uses (its Reset frees `uma`)
    allocator: usize,
    refs: u32,
    /// record chunks: (start, bytes used – filled in when the next chunk starts / at close)
    chunks: Vec<(Box<[u8]>, usize)>,
    texture_operations: Vec<Result<texture_ops::Operation,String>>,
    /// unified-memory blocks owned by the list (upload memory, argument tables, io)
    uma: Vec<usize>,
    /// where the next draw's argument table and io go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
}

struct PacketIndirect {
    signature: ID3D12CommandSignature,
    max: u32,
    arguments: ID3D12Resource,
    argument_offset: u64,
    count: Option<ID3D12Resource>,
    count_offset: u64,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);
/// The descriptor tables each list's last translation bound (`close`): its PC list holds their
/// pages' places until the list is closed again or is gone (`tables_live`).
static TABLES: Mutex<Option<HashMap<usize, super::console_heap::Tables>>> = Mutex::new(None);

/// The tables of every console list that has a translation, and how many of them bound any.
pub fn tables_live() -> (super::console_heap::Tables, usize) {
    let (mut all, mut lists) = (super::console_heap::Tables::default(), 0);
    if let Some(m) = TABLES.lock().unwrap().as_ref() {
        for t in m.values().filter(|t| !t.is_empty()) {
            all.join(t);
            lists += 1;
        }
    }
    (all, lists)
}
/// unified-memory blocks of finished recordings, freed when their allocator is reset
static PENDING: Mutex<Option<HashMap<usize, Vec<usize>>>> = Mutex::new(None);

/// Unified-memory blocks a recording on `allocator` uses: freed when it is reset (no
/// allocator known: kept until the next reset of any – never freed early).
pub(super) fn hold_until_reset(allocator: usize, blocks: Vec<usize>) {
    if !blocks.is_empty() {
        PENDING.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().extend(blocks);
    }
}

/// Unified-memory blocks recordings hold until their allocators reset: (allocators, blocks).
pub fn pending_blocks() -> (usize, usize) {
    PENDING.lock().unwrap().as_ref().map_or((0, 0), |m| (m.len(), m.values().map(|v| v.len()).sum()))
}

#[cfg(test)]
pub(crate) fn take_pending_for_test(allocator: usize) -> Vec<usize> {
    PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)).unwrap_or_default()
}

/// An allocator was reset: the GPU no longer uses what its recordings put in unified memory.
pub fn allocator_reset(allocator: *mut c_void) {
    super::indirect::allocator_reset(allocator as usize);
    let blocks = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&(allocator as usize)));
    if let Some(b) = blocks {
        unsafe { free_uma(&b) };
    }
}

fn with_list<R>(obj: usize, f: impl FnOnce(&mut ListState) -> R) -> Option<R> {
    LISTS.lock().unwrap().as_mut()?.get_mut(&obj).map(f)
}

unsafe fn field(obj: usize, at: usize) -> *mut usize {
    (obj + at) as *mut usize
}

/// start a fresh record chunk (the previous one keeps what was written)
unsafe fn new_chunk(obj: usize, st: &mut ListState, record_bytes: usize) {
    if let Some(last) = st.chunks.last_mut() {
        last.1 = *field(obj, F_WRITE) - last.0.as_ptr() as usize;
    }
    let size = CHUNK.max(record_bytes + SLACK);
    let c = vec![0u8; size].into_boxed_slice();
    let start = c.as_ptr() as usize;
    st.chunks.push((c, 0));
    *field(obj, F_WRITE) = start;
    *field(obj, F_END) = start + size - SLACK;
    *field(obj, F_DISPATCH_END) = start + size - SLACK;
}

/// append a record (methods: the same encodings the inline code writes)
unsafe fn push(obj: usize, words: &[u32]) {
    let bytes = std::mem::size_of_val(words);
    let w = *field(obj, F_WRITE);
    let limit = *field(obj, F_END) + SLACK / 2;
    if bytes > limit.saturating_sub(w) {
        if with_list(obj, |st| new_chunk(obj, st, bytes)).is_none() {
            return;
        }
    }
    let w = *field(obj, F_WRITE);
    std::ptr::copy_nonoverlapping(words.as_ptr(), w as *mut u32, words.len());
    *field(obj, F_WRITE) = w + bytes;
}

fn header(i: usize) -> u32 {
    PARAM_BASE | (i as u32) << 10
}

unsafe extern "system" fn cl_qi(this: *mut c_void, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    *ppv = this;
    cl_addref(this);
    S_OK
}
unsafe extern "system" fn cl_addref(this: *mut c_void) -> u32 {
    with_list(this as usize, |st| {
        st.refs += 1;
        st.refs
    })
    .unwrap_or(1)
}
unsafe extern "system" fn cl_release(this: *mut c_void) -> u32 {
    let obj = this as usize;
    let left = with_list(obj, |st| {
        st.refs -= 1;
        st.refs
    })
    .unwrap_or(0);
    if left == 0 {
        if let Some(m) = TABLES.lock().unwrap().as_mut() {
            m.remove(&obj);
        }
        if let Some(st) = LISTS.lock().unwrap().as_mut().and_then(|m| m.remove(&obj)) {
            free_uma(&st.uma);
            super::markers::recording_ended(st.pc);
            let timing = super::gpu_timing::list_token(st.pc as *mut c_void);
            if !super::real::is_adopted(st.pc as *mut c_void){super::bind::forget(st.pc as *mut c_void);}
            let release: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(pc_vtable_call(st.pc, 2));
            if super::physical_directory::recordings::release(st.pc, || release(st.pc)) == 0 {
                super::gpu_timing::list_released(timing);
                super::residency::list_reset(st.pc);
            }
        }
        drop(Box::from_raw(std::slice::from_raw_parts_mut(obj as *mut u8, OBJECT_BYTES)));
    }
    left
}
/// ID3D12Object / DeviceChild / CommandList methods 3..=8 go to the PC list
macro_rules! list_forward {
    ($($name:ident = $slot:literal),*) => {$(
        unsafe extern "system" fn $name(this: *mut c_void, a: usize, b: usize, c: usize) -> usize {
            let Some(pc) = with_list(this as usize, |st| st.pc) else { return 0 };
            let f: unsafe extern "system" fn(usize, usize, usize, usize) -> usize = std::mem::transmute(pc_vtable_call(pc, $slot));
            f(pc, a, b, c)
        }
    )*};
}
list_forward!(cl_3 = 3, cl_4 = 4, cl_5 = 5, cl_6 = 6, cl_7 = 7, cl_8 = 8);

unsafe extern "system" fn cl_close(this: *mut c_void) -> HRESULT {
    match close(this) {
        Ok(_) => S_OK,
        Err(e) => {
            crate::xlog_error!("console list: Close – {e}");
            crate::util::E_FAIL
        }
    }
}

unsafe extern "system" fn cl_reset(this: *mut c_void, allocator: *mut c_void, pso: *mut c_void) -> HRESULT {
    let obj = this as usize;
    let Some((pc, old_alloc, blocks)) = with_list(obj, |st| {
        st.chunks.clear();
        st.arena = Default::default();
        (st.pc, std::mem::replace(&mut st.allocator, allocator as usize), std::mem::take(&mut st.uma))
    }) else {
        return crate::util::E_FAIL;
    };
    // what the previous recording left in unified memory lives until its allocator resets
    if !blocks.is_empty() {
        PENDING.lock().unwrap().get_or_insert_with(HashMap::new).entry(old_alloc).or_default().extend(blocks);
    }
    with_list(obj, |st| new_chunk(obj, st, 0));
    let f: unsafe extern "system" fn(usize, *mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(pc_vtable_call(pc, 10));
    let hr = f(pc, allocator, pc_pso(pso));
    if hr >= 0 {
        super::physical_directory::recordings::ended(pc);
        if super::gpu_timing::enabled() && !super::real::is_adopted(pc as *mut c_void) {
            super::gpu_timing::opened(pc as *mut c_void, pc_pso(pso) as usize);
        }
        super::residency::list_reset(pc);
        if !super::real::is_adopted(pc as *mut c_void){super::bind::reset(pc as *mut c_void,allocator as usize,pc_pso(pso) as usize);}
        with_list(obj,|st|st.indirect.clear());
        super::markers::recording_ended(pc);
    }
    hr
}

unsafe extern "system" fn cl_dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    push(this as usize, &[DISPATCH, x, y, z]);
}

unsafe extern "system" fn cl_execute_indirect(this:*mut c_void,sig:*mut c_void,max:u32,args:*mut c_void,argument_offset:u64,count:*mut c_void,count_offset:u64) {
    let index=(||{
        let packet=PacketIndirect {
            signature:ID3D12CommandSignature::from_raw_borrowed(&sig)?.clone(), max,
            arguments:ID3D12Resource::from_raw_borrowed(&args)?.clone(), argument_offset,
            count:ID3D12Resource::from_raw_borrowed(&count).cloned(), count_offset,
        };
        with_list(this as usize,|st|{
            let index=u32::try_from(st.indirect.len()).ok()?;
            st.indirect.push(Arc::new(packet));
            Some(index)
        })?
    })().unwrap_or(u32::MAX);
    push(this as usize,&[INDIRECT_MARKER,index]);
}
unsafe extern "system" fn cl_set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    if pso.is_null() || *(pso as *const usize) != pso_vtable() {
        crate::xlog_warn!("console list: SetPipelineState with an object not from XCL ({pso:?}) – skipping");
        return;
    }
    let words: [u32; 7] = std::ptr::read((pso as usize + 0x10) as *const [u32; 7]);
    push(this as usize, &words);
}
unsafe extern "system" fn cl_set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let obj = this as usize;
    let n = layout_of(rs as usize).map(|l| l.params.len()).unwrap_or(0);
    if n == 0 {
        crate::xlog_warn!("console list: SetComputeRootSignature – unknown layout ({rs:?})");
    }
    for i in 0..MAX_PARAMS {
        *((obj + F_HEADERS + 4 * i) as *mut u32) = header(i);
    }
    push(obj, &[RS_MARKER, 0, rs as usize as u32, (rs as usize >> 32) as u32]);
}
unsafe extern "system" fn cl_set_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    push(this as usize, &[header(i as usize) + 4 * offset, value]);
}
unsafe extern "system" fn cl_set_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if n == 0 {
        return;
    }
    let mut w = vec![header(i as usize) + 4 * offset + ((n - 1) << 16)];
    w.extend_from_slice(std::slice::from_raw_parts(data, n as usize));
    push(this as usize, &w);
}
unsafe extern "system" fn cl_set_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    push(this as usize, &[header(i as usize), va as u32, (va >> 32) as u32]);
}
/// #32 SetDescriptorHeaps(n, heaps)
unsafe extern "system" fn cl_set_descriptor_heaps(this: *mut c_void, n: u32, heaps: *const usize) {
    let mut w = vec![HEAPS_MARKER, n];
    for k in 0..n as usize {
        let h = *heaps.add(k);
        w.extend([h as u32, (h >> 32) as u32]);
    }
    push(this as usize, &w);
}
/// #26 ResourceBarrier(n, barriers): D3D12_RESOURCE_BARRIER, 32 bytes each
unsafe extern "system" fn cl_resource_barrier(this: *mut c_void, n: u32, barriers: *const [u32; 8]) {
    let mut w = vec![TRANSITIONS_MARKER, n];
    for k in 0..n as usize {
        w.extend_from_slice(&*barriers.add(k));
    }
    push(this as usize, &w);
}
/// #15 CopyBufferRegion(dst, dst offset, src, src offset, bytes)
unsafe extern "system" fn cl_copy_buffer_region(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    let mut w = vec![COPY_BUFFER_MARKER];
    for v in [dst as u64, dst_off, src as u64, src_off, n] {
        w.extend([v as u32, (v >> 32) as u32]);
    }
    push(this as usize, &w);
}
/// #53 / #54 ClearUnorderedAccessViewUint / Float(gpu handle, cpu handle, resource, values[4],
/// rect count, rects) – amd_fidelityfx_upscaler_xbox.dll picks #54 for float targets
unsafe fn clear_uav(this: *mut c_void, float: bool, gpu: u64, cpu: u64, res: usize, values: *const u32, rects: u32) {
    if rects != 0 {
        crate::xlog_warn!("console list: ClearUnorderedAccessView with {rects} rectangles – clearing the whole view");
    }
    let mut w = vec![CLEAR_UAV_MARKER, float as u32];
    for v in [gpu, cpu, res as u64] {
        w.extend([v as u32, (v >> 32) as u32]);
    }
    w.extend_from_slice(std::slice::from_raw_parts(values, 4));
    push(this as usize, &w);
}
unsafe extern "system" fn cl_clear_uav_uint(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const u32, rects: u32, _r: *const c_void) {
    clear_uav(this, false, gpu, cpu, res, values, rects);
}
unsafe extern "system" fn cl_clear_uav_float(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const u32, rects: u32, _r: *const c_void) {
    clear_uav(this, true, gpu, cpu, res, values, rects);
}
/// #85 CopyTextureRegion(dst location, x, y, z, src location, box) – 48-byte
/// D3D12_TEXTURE_COPY_LOCATION, 24-byte D3D12_BOX
unsafe extern "system" fn cl_copy_texture_region(this: *mut c_void, dst: *const [u32; 12], x: u32, y: u32, z: u32, src: *const [u32; 12], bx: *const [u32; 6]) {
    if texture_ops::installed() {
        let aligned = |p: *const [u32;12]| (!p.is_null()).then(|| std::ptr::read_unaligned(p.cast::<[u64;6]>()));
        let (d,s) = (aligned(dst), aligned(src));
        let op = texture_ops::Operation::copy(d.as_ref().map_or(std::ptr::null(), |l| l.as_ptr().cast()), [x,y,z],
            s.as_ref().map_or(std::ptr::null(), |l| l.as_ptr().cast()), bx.cast(), 0);
        let index = with_list(this as usize, |st| {
            let i = u32::try_from(st.texture_operations.len()).expect("too many console texture operations");
            st.texture_operations.push(op);
            i
        }).expect("missing console recording");
        push(this as usize, &[OWNED_COPY_TEXTURE_MARKER, index]);
        return;
    }
    let mut w = vec![COPY_TEXTURE_MARKER];
    w.extend_from_slice(&*dst);
    w.extend([x, y, z]);
    w.extend_from_slice(&*src);
    w.push(!bx.is_null() as u32);
    w.extend_from_slice(&bx.as_ref().copied().unwrap_or([0; 6]));
    push(this as usize, &w);
}
unsafe extern "system" fn cl_barrier(this: *mut c_void, n: u32, barriers: *const c_void) {
    if super::depth_state::enabled() {
        if let Some(pc) = with_list(this as usize, |s| s.pc) {
            super::depth_uav::observe_slot146(pc, n, barriers, false);
        }
    }
    push(this as usize, &[BARRIER_MARKER]);
}
unsafe extern "system" fn cl_allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else { return std::ptr::null_mut() };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with_list(this as usize, |st| st.arena.take(u, size, align, &mut st.uma)).flatten() {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with_list(this as usize, |st| st.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with_list(this as usize, |st| st.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

fn list_vtable() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        let mut v: Vec<usize> = (0..LIST_SLOTS).map(|i| crate::stubs::make_stub_info(DLL, &format!("ID3D12GraphicsCommandList(console)::#{i}"), 0).0).collect();
        // console slots found on XboxBvhBuilder.dll (HANDOFF)
        for (slot, f) in [
            (0, cl_qi as usize),
            (1, cl_addref as usize),
            (2, cl_release as usize),
            (3, cl_3 as usize),
            (4, cl_4 as usize),
            (5, cl_5 as usize),
            (6, cl_6 as usize),
            (7, cl_7 as usize),
            (8, cl_8 as usize),
            (9, cl_close as usize),
            (10, cl_reset as usize),
            (14, cl_dispatch as usize),
            (15, cl_copy_buffer_region as usize),
            (25, cl_set_pipeline_state as usize),
            (26, cl_resource_barrier as usize),
            (32, cl_set_descriptor_heaps as usize),
            (53, cl_clear_uav_uint as usize),
            (54, cl_clear_uav_float as usize),
            (59, cl_execute_indirect as usize),
            (85, cl_copy_texture_region as usize),
            (33, cl_set_compute_root_signature as usize),
            (37, cl_set_root_constant as usize),
            (39, cl_set_root_constants as usize),
            (41, cl_set_root_descriptor as usize),
            (43, cl_set_root_descriptor as usize),
            (45, cl_set_root_descriptor as usize),
            (146, cl_barrier as usize),
            (148, cl_allocate_upload_memory as usize),
        ] {
            v[slot] = f;
        }
        Box::leak(v.into_boxed_slice()).as_ptr() as usize
    })
}

/// A console-shaped command list recording into `pc` (an open PC list; its reference moves
/// into the console list).
pub unsafe fn create(pc: *mut c_void) -> *mut c_void {
    create_with_allocator(pc, std::ptr::null_mut())
}

/// `create` for a list recording from `allocator` (its Reset frees the list's unified memory).
pub unsafe fn create_with_allocator(pc: *mut c_void, allocator: *mut c_void) -> *mut c_void {
    super::physical_directory::recordings::ended(pc as usize);
    let obj = Box::into_raw(vec![0u8; OBJECT_BYTES].into_boxed_slice()) as *mut u8 as usize;
    *field(obj, 0) = list_vtable();
    let mut st = ListState { pc: pc as usize, allocator: allocator as usize, refs: 1, chunks: Vec::new(), uma: Vec::new(), arena: Default::default(), indirect: Vec::new(), texture_operations: Vec::new() };
    new_chunk(obj, &mut st, 0);
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(obj, st);
    obj as *mut c_void
}

/// The PC list behind a console list (for ExecuteCommandLists).
pub fn pc_list(obj: *mut c_void) -> Option<*mut c_void> {
    if obj.is_null() || unsafe { *(obj as *const usize) } != list_vtable() {
        return None;
    }
    with_list(obj as usize, |st| st.pc as *mut c_void)
}

/// The PC pipeline state behind a console-shaped one (others unchanged).
pub fn pc_pso(pso: *mut c_void) -> *mut c_void {
    if pso.is_null() || unsafe { *(pso as *const usize) } != pso_vtable() {
        return pso;
    }
    unsafe { (*(pso as *const ConsolePso)).pc as *mut c_void }
}

unsafe fn free_uma(blocks: &[usize]) {
    if let Some(u) = crate::gpu::uma::get() {
        for &b in blocks {
            if u.is_sparse(b) {
                super::sparse::free(b);
            } else {
                u.free(b);
            }
        }
    }
}

unsafe fn uma_copy(st: &mut ListState, bytes: &[u8]) -> Option<usize> {
    let u = crate::gpu::uma::get()?;
    // 32-bit console addresses: window 0; a piece of the list's arena
    let p = st.arena.take(u, bytes.len().max(16), 256, &mut st.uma)?;
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
    Some(p as usize)
}

/// The recorded words so far (diagnostics: what console code wrote).
pub fn recorded_words(obj: *mut c_void) -> Vec<u32> {
    let o = obj as usize;
    with_list(o, |st| {
        let mut out = Vec::new();
        let n = st.chunks.len();
        for (k, (c, used)) in st.chunks.iter().enumerate() {
            let used = if k + 1 == n { unsafe { *field(o, F_WRITE) - c.as_ptr() as usize } } else { *used };
            out.extend(c[..used].chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
        }
        out
    })
    .unwrap_or_default()
}

/// What `close` produced (for tests and logs).
#[derive(Debug, Default, Clone)]
pub struct Translation {
    pub dispatches: u32,
    pub skipped_dispatches: u32,
    pub barriers: u32,
    pub clears: u32,
    pub copies: u32,
    /// records whose resources / descriptors were not XCL's (skipped)
    pub skipped: u32,
    pub unknown: Option<(u32, usize)>,
}

/// Turn the recorded console commands into commands on the PC list and close it.
pub unsafe fn close(obj: *mut c_void) -> Result<Translation, String> {
    let obj = obj as usize;
    let mut guard = LISTS.lock().unwrap();
    let st = guard.as_mut().and_then(|m| m.get_mut(&obj)).ok_or("not an XCL console list")?;
    if let Some(last) = st.chunks.last_mut() {
        last.1 = *field(obj, F_WRITE) - last.0.as_ptr() as usize;
    }
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&(st.pc as *mut c_void)).ok_or("no PC list")?.clone();
    // created with the first translated pipeline (without one every dispatch is skipped)
    let rs = super::real::abi_rs_cached().and_then(|p| ID3D12RootSignature::from_raw_borrowed(&p).cloned());
    let u = crate::gpu::uma::get().ok_or("no UMA")?;
    // the whole recording is replayed into the PC list here: its transitions of textures kept
    // in step are these
    super::title_textures::list_reset(st.pc);
    let windows = super::real::uma_window_vas();
    let uma_va = super::real::shader_window_base(&windows)?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(uma_va);
    let mut out = Translation::default();
    let mut layout: Option<Arc<Layout>> = None;
    let mut title_rs = 0usize;
    let mut table: Vec<u8> = Vec::new();
    let mut pso: Option<*const ConsolePso> = None;
    // translated shaders reach memory through the shared heap (`real::mem_views`): bound before
    // the first dispatch when the title binds none
    let mut heaps_bound = false;
    let mut bound_heaps = Vec::new();
    if super::console_heap::keeping() {
        TABLES.lock().unwrap().get_or_insert_with(HashMap::new).entry(obj).or_default().clear();
    }
    let texture_operations = std::mem::take(&mut st.texture_operations);
    let chunks: Vec<Vec<u32>> = st.chunks.iter().map(|(c, used)| c[..*used].chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()).collect();
    'chunks: for (chunk, words) in chunks.iter().enumerate() {
        let mut at = 0usize;
        while at < words.len() {
            let w0 = words[at];
            let get = |k: usize| words.get(at + k).copied().unwrap_or(0);
            if w0 == PIPE_MARKER {
                pso = Some((get(2) as usize | (get(3) as usize) << 32) as *const ConsolePso);
                at += 7;
            } else if w0 == RS_MARKER {
                let r = get(2) as usize | (get(3) as usize) << 32;
                if r != title_rs || layout.is_none() {
                    layout = layout_of(r);
                    table = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
                    title_rs = r;
                }
                at += 4;
            } else if w0 == HEAPS_MARKER {
                let n = get(1) as usize;
                let named: Vec<usize> = (0..n).map(|k| get(2 + 2 * k) as usize | (get(3 + 2 * k) as usize) << 32).collect();
                let heaps = super::list_x::pc_heaps(n as u32, named.as_ptr());
                if super::present::trace_frame() {
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::xlog_info!("frame trace: SetDescriptorHeaps {:x?}", (0..n).map(|k| get(2 + 2 * k) as usize | (get(3 + 2 * k) as usize) << 32).collect::<Vec<_>>());
                    }
                }
                list.SetDescriptorHeaps(&heaps);
                bound_heaps=heaps.into_iter().flatten().collect();
                heaps_bound = true;
                at += 2 + 2 * n;
            } else if w0 == TRANSITIONS_MARKER {
                let n = get(1) as usize;
                let words = &words[(at + 2).min(words.len())..(at + 2 + 8 * n).min(words.len())];
                if words.len() == 8 * n && n > 0 {
                    // 4-byte aligned in the stream: copy out; the resource pointers are
                    // borrowed (ManuallyDrop fields, nothing released)
                    let mut b: Vec<u64> = vec![0; 4 * n];
                    std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, b.as_mut_ptr() as *mut u8, 32 * n);
                    if super::title_textures::any() {
                        // each barrier: type | flags, resource, subresource | before, after
                        let t: Vec<(usize, u32, D3D12_RESOURCE_STATES)> =
                            b.chunks_exact(4).filter(|c| c[0] as u32 == 0).map(|c| (c[1] as usize, c[2] as u32, D3D12_RESOURCE_STATES(c[3] as u32 as i32))).collect();
                        super::title_textures::note_transitions(st.pc, &t);
                    }
                    super::residency::note_list_resources(st.pc, b.chunks_exact(4).flat_map(|c| match c[0] as u32 {
                        0 | 2 => [c[1] as usize, 0],
                        1 => [c[1] as usize, c[2] as usize],
                        _ => [0, 0],
                    }));
                    if super::residency::eviction_enabled() && b.chunks_exact(4).any(|c| c[0] as u32 == 1 && (c[1] == 0 || c[2] == 0)) {
                        super::residency::note_list_untracked(st.pc);
                    }
                    list.ResourceBarrier(std::slice::from_raw_parts(b.as_ptr() as *const D3D12_RESOURCE_BARRIER, n));
                }
                out.barriers += n as u32;
                at += 2 + 8 * n;
            } else if w0 == COPY_BUFFER_MARKER {
                let q = |k: usize| get(1 + 2 * k) as u64 | (get(2 + 2 * k) as u64) << 32;
                let (dst, src) = (q(0) as *mut c_void, q(2) as *mut c_void);
                if let (Some(d), Some(s)) = (ID3D12Resource::from_raw_borrowed(&dst), ID3D12Resource::from_raw_borrowed(&src)) {
                    super::residency::note_list_resources(st.pc, [dst as usize, src as usize]);
                    list.CopyBufferRegion(d, q(1), s, q(3), q(4));
                    out.copies += 1;
                } else {
                    out.skipped += 1;
                }
                at += 11;
            } else if w0 == CLEAR_UAV_MARKER {
                let q = |k: usize| get(2 + 2 * k) as u64 | (get(3 + 2 * k) as u64) << 32;
                let values = [get(8), get(9), get(10), get(11)];
                let res = q(2) as *mut c_void;
                let (gpu, cpu) = (super::console_heap::slot(q(0)), super::console_heap::slot(q(1)));
                match (gpu, cpu, ID3D12Resource::from_raw_borrowed(&res)) {
                    (Some(g), Some(c), Some(r)) if g.pc_gpu_uav != 0 => {
                        super::residency::note_list_resources(st.pc, [res as usize]);
                        let (gh, ch) = (D3D12_GPU_DESCRIPTOR_HANDLE { ptr: g.pc_gpu_uav }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: c.pc_uav as usize });
                        if get(1) != 0 {
                            let f = values.map(f32::from_bits);
                            list.ClearUnorderedAccessViewFloat(gh, ch, r, &f, &[]);
                        } else {
                            list.ClearUnorderedAccessViewUint(gh, ch, r, &values, &[]);
                        }
                        out.clears += 1;
                    }
                    _ => out.skipped += 1,
                }
                at += 12;
            } else if w0 == OWNED_COPY_TEXTURE_MARKER {
                let idx = *words.get(at+1).ok_or("missing owned texture copy index")? as usize;
                let op = texture_operations.get(idx).ok_or("invalid owned texture copy index")?.as_ref().map_err(Clone::clone)?;
                if !texture_ops::intercept(st.pc as *mut c_void, texture_ops::RecordingPoint::Console { chunk, dword: at }, || Ok(op.clone()))? {
                    op.unaffected_console_copy(&list)?;
                }
                out.copies += 1;
                at += 2;
            } else if w0 == COPY_TEXTURE_MARKER {
                if texture_ops::installed() { return Err("console texture copy predates owned operation capture".into()); }
                // aligned copies of the two 48-byte locations and the box (borrowed pointers)
                let loc = |from: usize| -> [u64; 6] {
                    let mut b = [0u64; 6];
                    let src = &words[(at + from).min(words.len())..(at + from + 12).min(words.len())];
                    std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, b.as_mut_ptr() as *mut u8, 4 * src.len());
                    b
                };
                let (dst, src) = (loc(1), loc(16));
                let bx: [u32; 6] = [get(29), get(30), get(31), get(32), get(33), get(34)];
                let (pd, ps) = (dst[0] as *mut c_void, src[0] as *mut c_void);
                if ID3D12Resource::from_raw_borrowed(&pd).is_some() && ID3D12Resource::from_raw_borrowed(&ps).is_some() {
                    super::residency::note_list_resources(st.pc, [pd as usize, ps as usize]);
                    super::vram::note_gpu_write(pd as usize);
                    list.CopyTextureRegion(
                        dst.as_ptr() as *const D3D12_TEXTURE_COPY_LOCATION,
                        get(13),
                        get(14),
                        get(15),
                        src.as_ptr() as *const D3D12_TEXTURE_COPY_LOCATION,
                        if get(28) != 0 { Some(bx.as_ptr() as *const D3D12_BOX) } else { None },
                    );
                    out.copies += 1;
                } else {
                    out.skipped += 1;
                }
                at += 35;
            } else if w0 == INDIRECT_MARKER {
                let packet=st.indirect.get(*words.get(at+1).ok_or("truncated ExecuteIndirect packet")? as usize).ok_or("invalid ExecuteIndirect packet")?.clone();
                let pso=pso.ok_or("ExecuteIndirect has no pipeline")?;
                if !(*pso).translated || !super::indirect::dispatches(packet.signature.as_raw() as usize) {
                    return Err("the compute packet recorder cannot replay this ExecuteIndirect signature and pipeline".into());
                }
                super::physical_directory::recordings::retain_native(st.pc,packet.clone())?;
                let entry=super::list_x::native_buffer::live::Scope::new(bound_heaps.clone(),None);
                super::bind::execute_indirect_packet(st.pc as *mut c_void,st.allocator,(*pso).pc,title_rs,layout.clone(),&mut table,
                    packet.signature.as_raw() as usize,packet.max,packet.arguments.as_raw() as usize,packet.argument_offset,
                    packet.count.as_ref().map_or(0,|r|r.as_raw() as usize),packet.count_offset,&entry)?;
                at+=2;
            } else if w0 == BARRIER_MARKER {
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
                }]);
                super::depth_uav::note_slot146_global(st.pc);
                out.barriers += 1;
                at += 1;
            } else if w0 == DISPATCH || w0 & !4 == 0x4002_E800 {
                let (x, y, z) = (get(1), get(2), get(3));
                at += 4;
                match pso {
                    Some(p) if (*p).translated => {
                        // a pipeline still being built in the background is waited for; if
                        // the background translation failed, skip instead of binding XCL's
                        // ABI root signature onto the stand-in
                        let real = match super::pso_async::resolve_for_compute_use((*p).pc) {
                            Ok(real) => real,
                            Err(skip) => {
                                super::bind::note_pending_compute_skip_with_args((*p).pc, "console-list Dispatch", skip, layout.as_deref(), &table);
                                out.skipped_dispatches += 1;
                                continue;
                            }
                        };
                        if super::bind::shader_skip(real) {
                            if let Some(layout) = &layout {
                                super::bind::record_skipped_writes(real, (*p).pc, super::bind::bound_writes(layout, &table));
                            }
                            out.skipped_dispatches += 1;
                            continue;
                        }
                        if !super::real::is_translated_compute(real) {
                            out.skipped_dispatches += 1;
                            continue;
                        }
                        let user_sgprs = super::real::pso_user_sgprs(real);
                        let inline = layout.as_ref().map_or(0, |l| inline_for(l, user_sgprs) as usize).min(table.len());
                        let table_at = uma_copy(st, &table[inline..]).ok_or("UMA: argument table")?;
                        let mut io = match layout.as_ref() {
                            Some(l) => user_data_for(l, user_sgprs, &table, table_at as u64),
                            None => vec![table_at as u32, (table_at >> 32) as u32],
                        };
                        io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
                        io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
                        io[crate::gpu::recomp::IO_MEM_VIEWS] = super::real::mem_views().ok_or("memory windows have no views")?;
                        io[crate::gpu::recomp::IO_PHYSICAL_TABLE] = super::physical_directory::recordings::packet_index(st.pc);
                        io[crate::gpu::recomp::IO_CARRY_AT] = super::bind::carry_watch();
                        let raw_list = st.pc as *mut c_void;
                        let title_pso = (*p).pc;
                        super::markers::before_op(raw_list);
                        let (window, offset) = super::bind::trip_watch(raw_list, title_pso, real, false);
                        io[crate::gpu::recomp::IO_TRIP_WINDOW] = window;
                        io[crate::gpu::recomp::IO_TRIP_AT] = offset;
                        let io_at = uma_copy(st, bytemuck_u32(&io)).ok_or("UMA: io")?;
                        let res_at = uma_copy(st, &[0u8; 256]).ok_or("UMA: res")?;
                        // the pipeline's own root signature when it has static samplers
                        let own = super::real::pso_root_signature(real).or(((*p).rs != 0).then_some((*p).rs)).map(|r| r as *mut c_void);
                        match own.as_ref().and_then(|r| ID3D12RootSignature::from_raw_borrowed(r)) {
                            Some(r) => list.SetComputeRootSignature(r),
                            None => list.SetComputeRootSignature(rs.as_ref().ok_or("no XCL ABI root signature")?),
                        }
                        list.SetComputeRootUnorderedAccessView(0, va(io_at));
                        list.SetComputeRootUnorderedAccessView(1, va(res_at));
                        if !heaps_bound {
                            let h=super::console_heap::global_heap();
                            list.SetDescriptorHeaps(&[h.clone()]);
                            bound_heaps=h.into_iter().collect();
                            heaps_bound = true;
                        }
                        let raw = real as *mut c_void;
                        let pc = ID3D12PipelineState::from_raw_borrowed(&raw).ok_or("PSO")?;
                        list.SetPipelineState(pc);
                        super::shader_code::used(st.pc, real);
                        super::residency::note_list_untracked(st.pc);
                        super::gpu_timing::op_grid(raw_list, super::gpu_timing::Kind::Dispatch, || title_pso, [x, y, z], || list.Dispatch(x, y, z));
                        super::markers::after_op(raw_list, "dispatch", title_pso);
                        if super::bind::dispatch_snapshot_enabled() {
                            if let Some(layout) = layout.clone() {
                                super::bind::note_dispatch_snapshot(st.pc, title_pso, Some([x, y, z]), layout, table.clone(), (io[..16].to_vec(), user_sgprs, inline as u32));
                            }
                        }
                        out.dispatches += 1;
                    }
                    _ => out.skipped_dispatches += 1,
                }
            } else if w0 & 0xC000_0000 == PARAM_BASE {
                let i = ((w0 >> 10) & 0x3F) as usize;
                let off = ((w0 & 0x3FF) >> 2) as usize;
                let n = ((w0 >> 16) & 0x3FFF) as usize + 1;
                match layout.as_ref().and_then(|l| l.params.get(i).copied()) {
                    Some(Param::Constants { at: o, .. }) => {
                        for k in 0..n {
                            let d = (o as usize) + 4 * (off + k);
                            if let Some(t) = table.get_mut(d..d + 4) {
                                t.copy_from_slice(&get(1 + k).to_le_bytes());
                            }
                        }
                        at += 1 + n;
                    }
                    Some(Param::Table { at: o }) => {
                        let o = o as usize;
                        let slots = layout.as_ref().and_then(|l| l.tables.get(i).copied()).unwrap_or(u32::MAX);
                        if let Some(kept) = TABLES.lock().unwrap().as_mut().and_then(|m| m.get_mut(&obj)) {
                            kept.note(get(1), slots);
                        }
                        super::console_heap::table_bound(get(1), slots);
                        if let Some(t) = table.get_mut(o..o + 4) {
                            t.copy_from_slice(&get(1).to_le_bytes());
                        }
                        at += 2;
                    }
                    Some(Param::Descriptor { at: o }) => {
                        let o = o as usize;
                        if let Some(t) = table.get_mut(o..o + 8) {
                            t[..4].copy_from_slice(&get(1).to_le_bytes());
                            t[4..].copy_from_slice(&get(2).to_le_bytes());
                        }
                        at += 3;
                    }
                    None => {
                        out.unknown = Some((w0, at));
                        break 'chunks;
                    }
                }
            } else {
                out.unknown = Some((w0, at));
                break 'chunks;
            }
        }
    }
    drop(guard);
    if std::env::var_os("XCL_CONSOLE_LISTS_LOG").is_some() {
        crate::xlog_info!(
            "console list: {} dispatches, {} skipped (PSO without a translation), {} barriers, {} clears, {} copies, {} skipped records",
            out.dispatches,
            out.skipped_dispatches,
            out.barriers,
            out.clears,
            out.copies,
            out.skipped
        );
    }
    if let Some((w, at)) = out.unknown {
        crate::xlog_warn!("console list: unknown record {w:#010x} (word {at}) – rest skipped");
    }
    super::gpu_timing::before_close(list.as_raw());
    list.Close().map_err(|e| format!("Close: {e}"))?;
    Ok(out)
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_packs_parameters_in_order() {
        // RTS0 v2: consts b0 x19 | UAV u0 | UAV u1 | SRV t0
        let mut r = vec![2u32, 4, 24, 0, 0, 0];
        let pay = 24 + 4 * 12;
        r.extend([1, 0, pay as u32, 4, 0, pay as u32 + 12, 4, 0, pay as u32 + 24, 3, 0, pay as u32 + 36]);
        r.extend([0, 0, 19, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
        let bytes: Vec<u8> = r.iter().flat_map(|w| w.to_le_bytes()).collect();
        let l = layout_from_blob(&bytes).unwrap();
        assert_eq!(l.params, vec![Param::Constants { at: 0, count: 19 }, Param::Descriptor { at: 76 }, Param::Descriptor { at: 84 }, Param::Descriptor { at: 92 }]);
        assert_eq!(l.size, 100);
        assert_eq!(l.inline, 0, "19 constants do not fit in user SGPRs: everything in the table");
    }

    /// The shader's own count of user SGPRs caps what goes inline: a shader built with four
    /// (XboxBvhBuilder.dll's, whatever the root signature) takes every argument from the table.
    #[test]
    fn a_shader_with_four_user_sgprs_takes_every_argument_from_the_table() {
        // two root descriptors and 16 constants: by the rule alone the descriptors are inline
        let l = layout_from_blob(&rts0(&[(4, 0), (4, 0), (1, 16)])).unwrap();
        assert_eq!((l.size, l.inline), (80, 16));
        let args: Vec<u8> = (0..80u8).collect();
        assert_eq!(inline_for(&l, None), 16, "count not known: the rule");
        assert_eq!(inline_for(&l, Some(8)), 16, "four registers of arguments: the rule");
        assert_eq!(inline_for(&l, Some(20)), 16, "more registers than the rule uses: the rule");
        assert_eq!(inline_for(&l, Some(6)), 8, "two registers of arguments: one descriptor");
        assert_eq!(inline_for(&l, Some(7)), 8, "three: still one descriptor, a parameter stays whole");
        assert_eq!(inline_for(&l, Some(4)), 0, "s4 is not an argument: everything in the table");
        assert_eq!(inline_for(&l, Some(0)), 0, "no user SGPR at all");
        assert_eq!(user_data_for(&l, Some(4), &args, 0x7_0000_0100), vec![0x100, 7, 0, 0]);
        assert_eq!(user_data_for(&l, Some(8), &args, 0x7_0000_0100), user_data(&l, &args, 0x7_0000_0100));
    }

    /// A table's slots: where the last of its ranges ends, in descriptors from its start – a
    /// range at offset u32::MAX starts where the one before ends, a count of u32::MAX has no
    /// end. (RTS0 v2: three constants, then the table of `ranges`: kind, count, offset.)
    #[test]
    fn a_tables_slots_are_where_its_ranges_end() {
        let tables = |ranges: &[[u32; 3]]| {
            let mut r = vec![2u32, 2, 24, 0, 0, 0, 1, 0, 48, 0, 0, 60, 0, 0, 3, ranges.len() as u32, 68];
            for &[kind, count, offset] in ranges {
                r.extend([kind, count, 0, 0, 0, offset]);
            }
            layout_from_blob(&r.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap().tables
        };
        assert_eq!(tables(&[[0, 8, 0], [1, 4, u32::MAX]]), [0, 12]);
        assert_eq!(tables(&[[0, 8, 0], [1, 4, u32::MAX], [2, 2, 100]]), [0, 102]);
        assert_eq!(tables(&[[2, 2, 100], [0, 8, 0]]), [0, 102], "the range that ends last, wherever it is listed");
        assert_eq!(tables(&[[0, 16, 4], [0, u32::MAX, u32::MAX]]), [0, u32::MAX]);
        assert_eq!(tables(&[]), [0, 0]);
        let mut cut = vec![2u32, 1, 24, 0, 0, 0, 0, 0, 36, 2, 44, 0, 8, 0, 0, 0, 0];
        cut.truncate(16);
        assert_eq!(layout_from_blob(&cut.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap().tables, [u32::MAX], "a range past the blob's end");
    }

    /// RTS0 v2 of parameters (type: 0 table, 1 constants, 2–4 CBV / SRV / UAV; constant count)
    fn rts0(params: &[(u32, u32)]) -> Vec<u8> {
        let n = params.len() as u32;
        let mut r = vec![2u32, n, 24, 0, 0, 0];
        let pay = 24 + 12 * n;
        for (k, &(ty, _)) in params.iter().enumerate() {
            r.extend([ty, 0, pay + 12 * k as u32]);
        }
        for &(_, count) in params {
            r.extend([0, 0, count]);
        }
        r.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// What the console's compiler does (its output on these root signatures): parameters in
    /// user SGPRs s4… while each fits whole, the rest in the table from its start.
    #[test]
    fn arguments_that_fit_go_in_user_sgprs() {
        let inline = |p: &[(u32, u32)]| layout_from_blob(&rts0(p)).unwrap().inline;
        assert_eq!(inline(&[(4, 0)]), 8);
        assert_eq!(inline(&[(1, 2), (4, 0)]), 16);
        assert_eq!(inline(&[(0, 0)]), 4);
        assert_eq!(inline(&[(4, 0); 6]), 48);
        assert_eq!(inline(&[(1, 10), (4, 0)]), 48);
        assert_eq!(inline(&[(1, 11), (4, 0)]), 44, "the UAV goes to the table");
        assert_eq!(inline(&[(4, 0), (1, 11)]), 8, "the constants go to the table, whole");
        assert_eq!(inline(&[(1, 20), (4, 0)]), 0, "once one goes to the table, the rest follow");
        let l = layout_from_blob(&rts0(&[(1, 1), (4, 0), (1, 11)])).unwrap();
        let args: Vec<u8> = [7u32, 0x1000, 4].iter().flat_map(|w| w.to_le_bytes()).chain([0u8; 44]).collect();
        assert_eq!(user_data(&l, &args, 0x4_0000_0100), vec![0x100, 4, 0, 0, 7, 0x1000, 4]);
    }

    /// The same for a vertex / pixel shader: from s8, 24 SGPRs (the console compiler's output).
    #[test]
    fn graphics_arguments_go_in_user_sgprs_from_s8() {
        let inline = |p: &[(u32, u32)]| layout_from_blob(&rts0(p)).unwrap().inline_gfx;
        assert_eq!(inline(&[(1, 2), (2, 0), (0, 0)]), 20);
        assert_eq!(inline(&[(1, 22), (2, 0)]), 96);
        assert_eq!(inline(&[(1, 23), (2, 0)]), 92, "the CBV goes to the table");
        assert_eq!(inline(&[(1, 25), (2, 0)]), 0);
        let l = layout_from_blob(&rts0(&[(1, 2), (2, 0), (0, 0)])).unwrap();
        let args: Vec<u8> = [1u32, 2, 0x5000, 4, 0x80].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(user_data_gfx(&l, &args, 0x200), vec![0x200, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0x5000, 4, 0x80]);
    }
}
