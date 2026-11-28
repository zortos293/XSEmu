//! ExecuteIndirect's arguments made safe to run. A console compute pass that XCL skips (its
//! pipeline failed or cannot be built) leaves the arguments it should have written as whatever
//! the memory held, and an indirect call turns that into oversized work: a dispatch of millions
//! of groups, or an argument the GPU reads past, until the PC removes the device. Two guards:
//!
//! - **Producers** (`XCL_INDIRECT_GUARD`, default 1; 0: off). The buffers a skipped dispatch
//!   could have written – the buffer views its root signature declares as UAVs (root UAVs, the
//!   UAV ranges of its descriptor tables; console addresses) – are kept as unwritten until that
//!   pipeline dispatches again with a translation, or a dispatch that runs has a UAV over them
//!   (`written`). Views it only reads are not kept. A view larger than `MAX_TRACKED` spans the
//!   title's memory rather than one buffer: a write through it cannot be placed, so it neither
//!   marks nor clears. Nor does a view of a shared pool – the same view a UAV of
//!   `SHARED_BINDERS` or more pipelines covers (a title's pool of indirect arguments, bound
//!   through a global table): where one pass writes in it cannot be placed either. An indirect
//!   call whose arguments or count lie in a kept range is skipped
//!   (`check`); its own bindings are not kept, so a refusal does not spread.
//! - **Sanitiser.** Before an indirect call with a translated pipeline runs, a compute pass of
//!   XCL's copies its active records (`min(max count, the GPU's count)`) into a device-local
//!   scratch buffer, the title's argument and count buffers untouched: each dispatch axis capped
//!   at 65,535 (D3D12's limit), a mesh dispatch over D3D12's 2^22 groups emptied, and records past
//!   the call's budget emptied – the running total saturates, never wraps. The indirect call then
//!   reads the scratch copy and its count.
//!
//! The budgets (`GROUP_BUDGET`, `DRAW_BUDGET`) are XCL's bounds on one indirect call, not
//! hardware limits: legitimate work beyond them is cut.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// D3D12_INDIRECT_ARGUMENT_TYPE values the sanitiser knows
pub(super) const DRAW: u32 = 0;
pub(super) const DRAW_INDEXED: u32 = 1;
pub(super) const DISPATCH: u32 = 2;
pub(super) const DISPATCH_MESH: u32 = 10;

/// thread groups one indirect call may run in all (XCL's bound)
pub(super) const GROUP_BUDGET: u32 = 1 << 22;
/// vertex instances (vertices or indices times instances) one indirect call may draw (XCL's bound)
pub(super) const DRAW_BUDGET: u32 = 1 << 28;
/// D3D12_MS_DISPATCH_MAX_THREAD_GROUPS_PER_GRID
const MESH_GROUPS: u32 = 1 << 22;
/// one scratch buffer; an indirect call needing more runs unsanitised
const SCRATCH_BYTES: u64 = 1 << 20;

const SOURCE: &str = r#"
cbuffer Call : register(b0) { uint kind; uint stride; uint records; uint has_count; uint group_budget; uint draw_budget; uint mesh_groups; uint raw_bytes; uint two_pass_groups; uint two_pass_split; };
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer cnt : register(u1);
RWByteAddressBuffer dst : register(u2);
[RootSignature("RootConstants(num32BitConstants=10, b0), UAV(u0), UAV(u1), UAV(u2)")]
[numthreads(1, 1, 1)]
void main() {
    uint n = records;
    if (has_count != 0) {
        n = min(n, cnt.Load(0));
    }
    uint groups_left = group_budget;
    uint draws_left = draw_budget;
    for (uint i = 0; i < n; i++) {
        uint at = i * stride;
        for (uint o = 0; o < stride; o += 4) {
            dst.Store(at + o, src.Load(at + o));
        }
        if (kind <= 1) {
            uint c = src.Load(at);
            uint inst = src.Load(at + 4);
            bool fits = true;
            if (c != 0 && inst != 0) {
                fits = inst <= draws_left / c;
            }
            if (fits) {
                if (c != 0 && inst != 0) {
                    draws_left -= c * inst;
                }
            } else {
                dst.Store2(at, uint2(0, 0));
            }
        } else {
            uint3 g = min(src.Load3(at), uint3(65535, 65535, 65535));
            if (two_pass_groups != 0) {
                // a mesh call in two passes: its groups on one axis (the console numbers mesh
                // groups in x only), cut to the record memory its command has
                uint gxy = g.x * g.y;
                uint t = 0;
                if (gxy != 0 && g.z != 0) {
                    t = two_pass_groups;
                    if (g.z <= two_pass_groups / gxy) {
                        t = min(gxy * g.z, two_pass_groups);
                    }
                }
                g = t != 0 ? uint3(t, 1, 1) : uint3(0, 0, 0);
            }
            uint total = 0;
            bool fits = true;
            if (g.x != 0 && g.y != 0 && g.z != 0) {
                if (g.y > groups_left / g.x) {
                    fits = false;
                } else {
                    uint xy = g.x * g.y;
                    if (g.z > groups_left / xy) {
                        fits = false;
                    } else {
                        total = xy * g.z;
                    }
                }
            }
            if (kind == 10 && total > mesh_groups) {
                fits = false;
            }
            if (fits) {
                groups_left -= total;
            } else {
                g = uint3(0, 0, 0);
            }
            dst.Store3(at, g);
            if (two_pass_groups != 0) {
                // the same record for the mesh pass after the count: groups times the
                // pass-through's mesh groups per record
                uint at2 = records * stride + 4 + raw_bytes + at;
                for (uint o2 = 12; o2 < stride; o2 += 4) {
                    dst.Store(at2 + o2, src.Load(at + o2));
                }
                dst.Store3(at2, uint3(g.x * two_pass_split, g.y, g.z));
            }
        }
    }
    // records past the active ones are empty, for a caller that runs them one by one
    for (uint j = n; j < records; j++) {
        for (uint o3 = 0; o3 < stride; o3 += 4) {
            dst.Store(j * stride + o3, 0);
            if (two_pass_groups != 0) {
                dst.Store(records * stride + 4 + raw_bytes + j * stride + o3, 0);
            }
        }
    }
    dst.Store(records * stride, n);
    for (uint o = 0; o < raw_bytes; o += 4) {
        dst.Store(records * stride + 4 + o, src.Load(o));
    }
}
"#;

/// The argument layout of a command signature: (its last argument's type, byte stride, whether
/// it is the only argument).
static SIGNATURES: Mutex<Option<HashMap<usize, (u32, u32, bool)>>> = Mutex::new(None);

const RECORD_LAYOUT_ID: windows::core::GUID = windows::core::GUID::from_u128(0xaf05f94d_19cf_4cc1_863d_55231cb0a7c6);
const RECORD_ROOT_ID: windows::core::GUID = windows::core::GUID::from_u128(0x93669287_011b_47dc_a778_85fde225daaa);

pub(super) unsafe fn store_record_layout(raw: *mut c_void, root: usize, stride: u32, arguments: &[[u32; 4]]) {
    let Some(signature) = ID3D12CommandSignature::from_raw_borrowed(&raw) else {
        return;
    };
    let mut words = Vec::with_capacity(1 + arguments.len() * 4);
    words.push(stride);
    words.extend_from_slice(arguments.as_flattened());
    let _ = signature.SetPrivateData(&RECORD_LAYOUT_ID, (words.len() * 4) as u32, Some(words.as_ptr().cast()));
    if let Some(rs) = ID3D12RootSignature::from_raw_borrowed(&(root as *mut c_void)) {
        let _ = signature.SetPrivateDataInterface(&RECORD_ROOT_ID, rs);
    }
}

unsafe fn record_arguments(signature: &ID3D12CommandSignature) -> Option<(u32, Vec<[u32; 4]>)> {
    let mut words = [0u32; 257];
    let mut len = std::mem::size_of_val(&words) as u32;
    signature.GetPrivateData(&RECORD_LAYOUT_ID, &mut len, Some(words.as_mut_ptr().cast())).ok()?;
    if len < 20 || len > 1028 || (len - 4) % 16 != 0 {
        return None;
    }
    let (args, _) = words[1..len as usize / 4].as_chunks::<4>();
    Some((words[0], args.to_vec()))
}

// The prebuilt indirect-action path has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) unsafe fn plain_action(signature: &ID3D12CommandSignature) -> Option<(u32, u32)> {
    let (stride, args) = record_arguments(signature)?;
    if args.len() != 1 || !matches!(args[0][0], 0..=2 | 10) {
        return None;
    }
    super::list_x::native_buffer::recipes::Layout::parse(stride, &args).ok()?;
    Some((args[0][0], stride))
}

pub(crate) struct IndirectRecipe {
    pub(crate) signature: ID3D12CommandSignature,
    pub(crate) layout: super::list_x::native_buffer::recipes::Layout,
    pub(crate) max: u32,
    pub(crate) arguments: super::list_x::native_buffer::Operand,
    pub(crate) count: Option<super::list_x::native_buffer::Operand>,
}

impl IndirectRecipe {
    pub(crate) unsafe fn translated_rows(
        &self,
        native_pso: usize,
        guest_root: &ID3D12RootSignature,
        base_args: &[u8],
        io: &[u32; 128],
    ) -> Option<super::list_x::native_buffer::translated::Rows> {
        use windows::core::IUnknown;
        let gfx = super::real::is_translated_graphics(native_pso);
        if gfx == super::real::is_translated_compute(native_pso) || super::real::two_pass_of(native_pso).is_some() {
            return None;
        }
        let layout = super::console_list::layout_of(guest_root.as_raw() as usize)?;
        let (stride, args) = record_arguments(&self.signature)?;
        if args.len() != 1 {
            let (mut bytes, mut raw) = (std::mem::size_of::<usize>() as u32, std::ptr::null_mut::<c_void>());
            self.signature
                .GetPrivateData(&RECORD_ROOT_ID, &mut bytes, Some((&mut raw as *mut *mut c_void).cast()))
                .ok()?;
            if raw.is_null() {
                return None;
            }
            let root = IUnknown::from_raw(raw);
            if root.cast::<IUnknown>().ok()? != guest_root.cast::<IUnknown>().ok()? {
                return None;
            }
        }
        super::list_x::native_buffer::translated::Rows::prepare(stride, &args, &layout, gfx, super::real::pso_user_sgprs(native_pso), base_args, io)
    }
    // The prebuilt indirect-action path has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn prebuilt_reads(
        &self,
        locator: u32,
        is_native: impl Fn(&super::list_x::native_buffer::Operand) -> bool,
        fallback: impl FnMut(&super::list_x::native_buffer::Operand, u32) -> Option<super::list_x::native_buffer::FallbackPage>,
    ) -> Option<IndirectReads> {
        if self.layout.embeds_addresses() {
            return None;
        }
        self.read_inputs(locator, is_native, fallback)
    }

    pub(crate) unsafe fn prebuilt_translated(
        &self,
        native_pso: usize,
        root: &ID3D12RootSignature,
        base_args: &[u8],
        io: &[u32; 128],
        is_native: impl Fn(&super::list_x::native_buffer::Operand) -> bool,
        fallback: impl FnMut(&super::list_x::native_buffer::Operand, u32) -> Option<super::list_x::native_buffer::FallbackPage>,
    ) -> Option<(super::list_x::native_buffer::translated::Rows, IndirectReads)> {
        let generated = self.translated_rows(native_pso, root, base_args, io)?;
        let inputs = self.read_inputs(io[crate::gpu::recomp::IO_PHYSICAL_TABLE], is_native, fallback)?;
        Some((generated, inputs))
    }

    fn read_inputs(
        &self,
        locator: u32,
        is_native: impl Fn(&super::list_x::native_buffer::Operand) -> bool,
        mut fallback: impl FnMut(&super::list_x::native_buffer::Operand, u32) -> Option<super::list_x::native_buffer::FallbackPage>,
    ) -> Option<IndirectReads> {
        use super::list_x::native_buffer::{Copy, Operand};
        let mut operand = |op: &Operand, bytes: u64| -> Option<IndirectInput> {
            match op.guest_address {
                Some(address) => {
                    let d = unsafe { op.resource.GetDesc() };
                    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || op.native_offset.checked_add(bytes)? > d.Width {
                        return None;
                    }
                    if let Some((base, len)) = super::real::console_buffer(op.resource.as_raw() as usize) {
                        if address.checked_sub(base as u64)?.checked_add(bytes)? > len as u64 { return None; }
                    }
                    Some(IndirectInput::Gather(Copy::read(address, bytes, locator, |page| fallback(op, page))?))
                }
                None if is_native(op) => {
                    let end = op.native_offset.checked_add(bytes)?;
                    let d = unsafe { op.resource.GetDesc() };
                    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || end > d.Width {
                        return None;
                    }
                    Some(IndirectInput::Native(op.resource.clone(), op.native_offset))
                }
                None => None,
            }
        };
        Some(IndirectReads {
            arguments: operand(&self.arguments, self.layout.active_bytes(self.max, None)? as u64)?,
            count: match self.count.as_ref() {
                Some(c) => Some(operand(c, 4)?),
                None => None,
            },
        })
    }
    pub(crate) unsafe fn retain(sig: usize, max: u32, args: usize, args_off: u64, count: usize, count_off: u64) -> Option<Self> {
        use super::list_x::native_buffer::Operand;
        if args_off & 3 != 0 || (count != 0 && count_off & 3 != 0) {
            return None;
        }
        let arguments = Operand::retain(args, args_off)?;
        let count = if count == 0 { None } else { Some(Operand::retain(count, count_off)?) };
        let signature = ID3D12CommandSignature::from_raw_borrowed(&(sig as *mut c_void))?.clone();
        let (stride, records) = record_arguments(&signature)?;
        let layout = super::list_x::native_buffer::recipes::Layout::parse(stride, &records).ok()?;
        Some(Self { signature, layout, max, arguments, count })
    }
}

pub(crate) enum IndirectInput {
    // The native indirect input is carried but not consumed yet; kept for the next fix.
    #[allow(dead_code)]
    Native(ID3D12Resource, u64),
    Gather(super::list_x::native_buffer::Read),
}

pub(crate) struct IndirectReads {
    pub(crate) arguments: IndirectInput,
    pub(crate) count: Option<IndirectInput>,
}

pub(super) fn note_signature(sig: usize, last: u32, stride: u32, single: bool) {
    SIGNATURES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, (last, stride, single));
}

/// Whether `sig` dispatches compute work (its last argument is a Dispatch).
pub(super) fn dispatches(sig: usize) -> bool {
    signature(sig).is_some_and(|s| s.0 == DISPATCH)
}

/// Whether `sig` dispatches mesh work.
pub(super) fn dispatches_mesh(sig: usize) -> bool {
    signature(sig).is_some_and(|s| s.0 == DISPATCH_MESH)
}

/// The byte stride of `sig`'s records.
pub(super) fn stride_of(sig: usize) -> Option<u32> {
    signature(sig).map(|s| s.1)
}

/// XCL's command signature of one Dispatch with records `stride` bytes apart (raw; kept): the
/// prepass of an indirect mesh call in two passes reads the mesh call's sanitised records.
pub(super) unsafe fn dispatch_signature(stride: u32) -> Option<usize> {
    static MADE: Mutex<Option<HashMap<u32, usize>>> = Mutex::new(None);
    let mut g = MADE.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&s) = map.get(&stride) {
        return Some(s);
    }
    let raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw)?;
    let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
    let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: stride.max(12), NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
    let mut sig: Option<ID3D12CommandSignature> = None;
    super::title_textures::own_work(|| dev.CreateCommandSignature(&desc, None, &mut sig)).ok()?;
    let s = sig?.into_raw() as usize;
    map.insert(stride, s);
    Some(s)
}

/// indirect calls by their signature's last argument: dispatch, draw, indexed draw, mesh, other
static CALLS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// An indirect call with `sig` was made.
pub(super) fn note_call(sig: usize) -> &'static str {
    let (k, label) = match signature(sig).map(|s| s.0) {
        Some(DISPATCH) => (0, "ExecuteIndirect(dispatch)"),
        Some(DRAW) => (1, "ExecuteIndirect(draw)"),
        Some(DRAW_INDEXED) => (2, "ExecuteIndirect(indexed draw)"),
        Some(DISPATCH_MESH) => (3, "ExecuteIndirect(mesh dispatch)"),
        _ => (4, "ExecuteIndirect(unknown signature)"),
    };
    CALLS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    label
}

/// The indirect calls so far by type, for the 30 s line.
pub(super) fn calls_summary() -> String {
    let c: Vec<u64> = CALLS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    format!("{} dispatch, {} draw, {} indexed draw, {} mesh, {} other", c[0], c[1], c[2], c[3], c[4])
}

pub(super) fn signature(sig: usize) -> Option<(u32, u32, bool)> {
    SIGNATURES.lock().unwrap().as_ref()?.get(&sig).copied()
}

/// The largest view kept as unwritten. The title's real buffers stay well below it (its shared
/// indirect-argument buffer is 174 MiB); its views of all of its memory are larger (play-85w…93b:
/// 0x3d000000..0xf665d000000, about 15 TiB).
pub(super) const MAX_TRACKED: u64 = 1 << 30;

/// Distinct pipelines with a UAV over one view that make it a shared pool. A pass's own output
/// has one writer or a few (a culling chain's); a pool every pass takes its part of is bound by
/// many (play-96: one 174 MiB view through slot 185 of a global table, at least 10 pipelines).
pub(super) const SHARED_BINDERS: usize = 8;

/// Console address ranges skipped dispatches were bound to write, by the title's pipeline.
#[derive(Default)]
pub(super) struct Unwritten {
    ranges: Vec<(u64, u64, usize)>,
    /// views (start, bytes) and the distinct pipelines seen with a UAV over each, up to
    /// `SHARED_BINDERS`
    binders: BTreeMap<(u64, u64), Vec<usize>>,
    /// shared pools (start, end): views bound by `SHARED_BINDERS` pipelines; a view inside one
    /// is part of the pool, whatever its own start and size
    pools: Vec<(u64, u64)>,
}

impl Unwritten {
    /// `producer` has UAVs over `ranges` (start, bytes). Returns the views this made shared
    /// pools; what was kept as unwritten on them goes.
    pub(super) fn bound(&mut self, producer: usize, ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut pooled = Vec::new();
        for &(start, bytes) in ranges.iter().filter(|r| r.1 > 0 && r.1 <= MAX_TRACKED).take(64) {
            if self.in_pool(start, bytes) || self.binders.len() >= 4096 && !self.binders.contains_key(&(start, bytes)) {
                continue;
            }
            let b = self.binders.entry((start, bytes)).or_default();
            if b.len() < SHARED_BINDERS && !b.contains(&producer) {
                b.push(producer);
                if b.len() == SHARED_BINDERS {
                    pooled.push((start, bytes));
                }
            }
        }
        for &(start, bytes) in &pooled {
            let end = start.saturating_add(bytes);
            if self.pools.len() < 4096 {
                self.pools.push((start, end));
            }
            self.ranges.retain(|&(s, e, _)| s < start || e > end);
        }
        pooled
    }

    /// Whether the view (start, bytes) lies inside a shared pool.
    fn in_pool(&self, start: u64, bytes: u64) -> bool {
        let end = start.saturating_add(bytes);
        self.pools.iter().any(|&(s, e)| s <= start && end <= e)
    }

    /// `producer` was skipped while bound to `ranges` (start, bytes): they replace its earlier ones.
    pub(super) fn mark(&mut self, producer: usize, ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let pooled = self.bound(producer, ranges);
        self.ranges.retain(|r| r.2 != producer);
        for &(start, bytes) in ranges.iter().filter(|r| r.1 > 0 && r.1 <= MAX_TRACKED).take(64) {
            if self.in_pool(start, bytes) {
                continue;
            }
            if self.ranges.len() >= 4096 {
                break;
            }
            self.ranges.push((start, start.saturating_add(bytes), producer));
        }
        pooled
    }

    /// `producer` ran with a translation: what it writes is written again.
    pub(super) fn written_by(&mut self, producer: usize) {
        self.ranges.retain(|r| r.2 != producer);
    }

    /// A dispatch ran with UAVs over `ranges` (start, bytes): what they cover is written, whoever
    /// skipped it.
    pub(super) fn written(&mut self, ranges: &[(u64, u64)]) {
        let placed: Vec<_> = ranges.iter().copied().filter(|r| r.1 > 0 && r.1 <= MAX_TRACKED && !self.in_pool(r.0, r.1)).collect();
        for (start, bytes) in placed {
            let end = start.saturating_add(bytes);
            if !self.ranges.iter().any(|&(s, e, _)| start < e && s < end) {
                continue;
            }
            let mut kept = Vec::with_capacity(self.ranges.len() + 1);
            for &(s, e, p) in &self.ranges {
                if start >= e || s >= end {
                    kept.push((s, e, p));
                    continue;
                }
                if s < start {
                    kept.push((s, start, p));
                }
                if end < e {
                    kept.push((end, e, p));
                }
            }
            self.ranges = kept;
        }
    }

    /// The first unwritten range overlapping `[start, start + bytes)`: (start, end, producer).
    pub(super) fn overlapping(&self, start: u64, bytes: u64) -> Option<(u64, u64, usize)> {
        let end = start.saturating_add(bytes.max(1));
        self.ranges.iter().copied().find(|&(s, e, _)| start < e && s < end)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

static UNWRITTEN: Mutex<Unwritten> = Mutex::new(Unwritten { ranges: Vec::new(), binders: BTreeMap::new(), pools: Vec::new() });

/// Indirect calls refused since the last `take_refusals`, by the unwritten range they read
/// (start, end) and the skipped pipeline it waits for.
static REFUSED: Mutex<BTreeMap<(u64, u64, usize), u64>> = Mutex::new(BTreeMap::new());

/// The refusals since the last call, most first: ((start, end, producer), calls).
pub(super) fn take_refusals() -> Vec<((u64, u64, usize), u64)> {
    let mut v: Vec<_> = std::mem::take(&mut *REFUSED.lock().unwrap()).into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// `XCL_INDIRECT_PRODUCER_FIRST=0`: a refused call leaves its producer's build where it is.
fn producer_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_INDIRECT_PRODUCER_FIRST").map_or(true, |v| v.trim() != "0"))
}

fn log_pooled(pooled: &[(u64, u64)]) {
    for &(start, bytes) in pooled {
        crate::xlog_info!("D3D12.X: indirect guard: {start:#x}..{:#x} is bound as a UAV by {SHARED_BINDERS} pipelines – a shared pool, neither kept as unwritten nor cleared", start.saturating_add(bytes));
    }
}

fn guard_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_INDIRECT_GUARD").map_or(true, |v| v.trim() != "0"))
}

/// A dispatch of the title's `pso` was skipped while its UAVs covered `ranges` (console start,
/// bytes).
pub(super) fn skipped_producer(pso: usize, ranges: &[(u64, u64)]) {
    if guard_on() {
        let pooled = UNWRITTEN.lock().unwrap().mark(pso, ranges);
        log_pooled(&pooled);
    }
}

/// A dispatch of the title's `pso` that ran had UAVs over `ranges` (console start, bytes): it
/// counts towards their views being shared pools.
pub(super) fn bound(pso: usize, ranges: &[(u64, u64)]) {
    if guard_on() {
        let pooled = UNWRITTEN.lock().unwrap().bound(pso, ranges);
        log_pooled(&pooled);
    }
}

/// Whether no range is kept as unwritten.
pub(super) fn nothing_unwritten() -> bool {
    UNWRITTEN.lock().unwrap().is_empty()
}

/// A dispatch that ran had UAVs over `ranges` (console start, bytes).
pub(super) fn written(ranges: &[(u64, u64)]) {
    let mut u = UNWRITTEN.lock().unwrap();
    if !u.is_empty() {
        u.written(ranges);
    }
}

/// The unwritten range holding console address `va`, and the skipped pipeline it waits for.
pub(super) fn unwritten_at(va: u64) -> Option<(u64, u64, usize)> {
    UNWRITTEN.lock().unwrap().overlapping(va, 4)
}

/// The first unwritten range overlapping `bytes` at console address `start`, and the skipped
/// pipeline it waits for.
pub(super) fn unwritten_overlapping(start: u64, bytes: u64) -> Option<(u64, u64, usize)> {
    UNWRITTEN.lock().unwrap().overlapping(start, bytes)
}

/// A dispatch of the title's `pso` ran with a translation.
pub(super) fn produced(pso: usize) {
    let mut u = UNWRITTEN.lock().unwrap();
    if !u.is_empty() {
        u.written_by(pso);
    }
}

/// Whether an indirect call with `sig` reading `max` records at console address `args` (and its
/// count at `count`) may run: Err names the skipped producer of what it reads. A refusal is
/// counted by range and producer (`take_refusals`), and a producer still building is built
/// before every other pipeline (`pso_async::build_first`).
pub(super) fn check(sig: usize, max: u32, args: Option<u64>, count: Option<u64>) -> Result<(), String> {
    let hit = {
        let u = UNWRITTEN.lock().unwrap();
        if u.is_empty() {
            return Ok(());
        }
        let stride = signature(sig).map_or(16, |s| s.1.max(4)) as u64;
        let hit = |what: &'static str, at: u64, bytes: u64| u.overlapping(at, bytes).map(|r| (what, at, r));
        args.and_then(|a| hit("arguments", a, stride * max.max(1) as u64)).or_else(|| count.and_then(|c| hit("count", c, 4)))
    };
    let Some((what, at, (s, e, p))) = hit else { return Ok(()) };
    *REFUSED.lock().unwrap().entry((s, e, p)).or_insert(0) += 1;
    if producer_first() && super::pso_async::build_first(p) {
        crate::xlog_info!("D3D12.X: indirect guard: pipeline {p:#x} builds before every other one – an indirect call reads {s:#x}..{e:#x}, which it writes");
    }
    Err(format!("its {what} at {at:#x} lie in {s:#x}..{e:#x}, unwritten since a dispatch of pipeline {p:#x} was skipped"))
}

/// XCL's sanitiser: (root signature, pipeline), raw; None when it could not be made (logged).
fn pipeline() -> Option<(usize, usize)> {
    static P: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    *P.get_or_init(|| unsafe {
        let code = super::real::compile(SOURCE, "cs_5_1");
        let raw = super::real::device_raw();
        let made = (|| -> Result<(usize, usize), String> {
            if code.is_empty() {
                return Err("its HLSL did not compile".into());
            }
            let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
            super::title_textures::own_work(|| {
                let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &code).map_err(|e| format!("root signature: {e}"))?;
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(|e| format!("pipeline: {e}"))?;
                if super::gpu_timing::enabled() {
                    super::dispatch_timing::note(pso.as_raw() as usize, &code);
                }
                Ok((rs.into_raw() as usize, pso.into_raw() as usize))
            })
        })();
        match made {
            Ok(p) => Some(p),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: no ExecuteIndirect argument sanitiser ({e}): indirect calls run with the title's arguments as they are");
                None
            }
        }
    })
}

/// Records on `list` the sanitiser's pass over `records` records of PC argument type `kind`,
/// `stride` bytes apart, from GPU address `src` (bounded by the 32-bit count at `count`) into
/// `dst`, which gets the records and, after them, the count of active ones, then the first
/// `raw_bytes` of `src` as they were. `src`, `count` and `dst` are read and written as UAVs;
/// ordering them is the caller's. `two_pass` (a mesh call run in two passes; groups a command may
/// run, mesh groups per record): each record's groups go on its x axis, cut to that many, and
/// after the count (and the raw bytes) follow the records again with x times the second number.
/// Records past the active ones are zero in both.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn record(list: &ID3D12GraphicsCommandList, kind: u32, stride: u32, records: u32, src: u64, count: Option<u64>, dst: u64, raw_bytes: u32, two_pass: Option<(u32, u32)>) -> bool {
    let Some((rs, pso)) = pipeline() else { return false };
    let (rs, pso) = (rs as *mut c_void, pso as *mut c_void);
    let (Some(rs), Some(pso)) = (ID3D12RootSignature::from_raw_borrowed(&rs), ID3D12PipelineState::from_raw_borrowed(&pso)) else { return false };
    let (two_pass_groups, two_pass_split) = two_pass.filter(|_| kind == DISPATCH_MESH).unwrap_or((0, 0));
    let consts = [kind, stride, records, count.is_some() as u32, GROUP_BUDGET, DRAW_BUDGET, MESH_GROUPS, raw_bytes, two_pass_groups, two_pass_split];
    list.SetComputeRootSignature(rs);
    list.SetComputeRoot32BitConstants(0, consts.len() as u32, consts.as_ptr() as *const c_void, 0);
    list.SetComputeRootUnorderedAccessView(1, src);
    list.SetComputeRootUnorderedAccessView(2, count.unwrap_or(src));
    list.SetComputeRootUnorderedAccessView(3, dst);
    list.SetPipelineState(pso);
    list.Dispatch(1, 1, 1);
    true
}

/// A scratch buffer of a recording: (resource, GPU address, bytes used, in INDIRECT_ARGUMENT).
struct Scratch {
    res: usize,
    va: u64,
    used: u64,
    indirect: bool,
}

/// scratch buffers of the recordings, by list
static RECORDING: Mutex<Option<HashMap<usize, Vec<Scratch>>>> = Mutex::new(None);
/// scratch buffers of finished recordings, by allocator: free once it is reset
static HELD: Mutex<Option<HashMap<usize, Vec<usize>>>> = Mutex::new(None);
/// scratch buffers the GPU no longer uses
static FREE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `list`'s recording on `allocator` ended (reset, split or gone): its scratch buffers are free
/// once `allocator` is reset.
pub(super) fn recording_done(list: usize, allocator: usize) {
    let Some(v) = RECORDING.lock().unwrap().as_mut().and_then(|m| m.remove(&list)) else { return };
    HELD.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().extend(v.into_iter().map(|s| s.res));
}

/// `allocator` was reset: the GPU no longer uses its recordings' scratch buffers.
pub(super) fn allocator_reset(allocator: usize) {
    if let Some(v) = HELD.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        FREE.lock().unwrap().extend(v);
    }
}

/// A new or free scratch buffer (resource, GPU address), in COMMON.
unsafe fn scratch_buffer() -> Option<(usize, u64)> {
    let res = match FREE.lock().unwrap().pop() {
        Some(r) => r,
        None => {
            let raw = super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw)?;
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: SCRATCH_BYTES,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            super::title_textures::own_work(|| dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r)).ok()?;
            r?.into_raw() as usize
        }
    };
    let p = res as *mut c_void;
    Some((res, ID3D12Resource::from_raw_borrowed(&p)?.GetGPUVirtualAddress()))
}

pub(super) fn transition(res: usize, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    let mut b = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, ..Default::default() };
    b.Anonymous.Transition = std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: unsafe { std::mem::transmute_copy(&res) },
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: before,
        StateAfter: after,
    });
    b
}

/// Sanitises an indirect call's arguments on `list` (the PC list of console list `this`): `max`
/// records of `sig` at console address `args`, bounded by the count at console address `count`.
/// Some((resource, offset of the records, offset of the count)) for the indirect call to read
/// instead; None: not sanitised (a signature with several arguments or of another type, memory
/// outside the unified-memory windows, more than a scratch buffer holds), the call runs as before.
/// `capture`: the first `RAW_BYTES` of the title's arguments are kept after the count, for
/// `capture`. `two_pass`: `record`'s – the mesh pass's records follow the count.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn sanitise(this: usize, list: &ID3D12GraphicsCommandList, sig: usize, max: u32, args: u64, count: Option<u64>, capture: bool, two_pass: Option<(u32, u32)>) -> Option<(usize, u64, u64)> {
    let (kind, stride, single) = signature(sig)?;
    if !single || !matches!(kind, DRAW | DRAW_INDEXED | DISPATCH | DISPATCH_MESH) || max == 0 {
        return None;
    }
    let bytes = stride as u64 * max as u64;
    let raw_bytes = if capture { bytes.min(RAW_BYTES) as u32 } else { 0 };
    let two_pass = two_pass.filter(|_| kind == DISPATCH_MESH);
    let copies = if two_pass.is_some() { 2 } else { 1 };
    if copies * bytes + 4 + raw_bytes as u64 > SCRATCH_BYTES {
        return None;
    }
    super::real::uma_window_buffer(args as usize, bytes as usize)?;
    let src = super::real::uma_va_of(args as usize)?;
    let count_va = match count {
        Some(c) => {
            super::real::uma_window_buffer(c as usize, 4)?;
            Some(super::real::uma_va_of(c as usize)?)
        }
        None => None,
    };
    pipeline()?;
    // a piece of the recording's scratch: 256-byte aligned, in UNORDERED_ACCESS for the pass
    let need = (copies * bytes + 4 + raw_bytes as u64 + 255) & !255;
    let (res, va, off, was_indirect) = {
        let mut g = RECORDING.lock().unwrap();
        let v = g.get_or_insert_with(HashMap::new).entry(this).or_default();
        if !v.last().is_some_and(|s| s.used + need <= SCRATCH_BYTES) {
            drop(g);
            let (res, va) = scratch_buffer()?;
            g = RECORDING.lock().unwrap();
            g.get_or_insert_with(HashMap::new).entry(this).or_default().push(Scratch { res, va, used: 0, indirect: false });
        }
        let s = g.as_mut()?.get_mut(&this)?.last_mut()?;
        let off = s.used;
        s.used += need;
        let was = std::mem::replace(&mut s.indirect, true);
        (s.res, s.va, off, was)
    };
    // the title's writers before the pass (a UAV barrier for all resources: they write through
    // XCL's memory windows); a scratch buffer an earlier call of this recording read moves back
    let mut before = vec![D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }];
    if was_indirect {
        before.push(transition(res, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
    list.ResourceBarrier(&before);
    if !record(list, kind, stride, max, src, count_va, va + off, raw_bytes, two_pass) {
        return None;
    }
    // first use in a recording: COMMON was promoted to UNORDERED_ACCESS by the pass
    list.ResourceBarrier(&[transition(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: ExecuteIndirect arguments sanitised on the GPU before they run (first: type {kind}, {max} records of {stride} B at {args:#x}{})", count.map_or(String::new(), |c| format!(", count at {c:#x}")));
    }
    Some((res, off, off + bytes))
}

/// title argument bytes a capture keeps (`XCL_GPU_MARKERS`)
const RAW_BYTES: u64 = 256;
/// indirect calls captured at most, and the readback bytes of each: the sanitised records (up
/// to `RAW_BYTES`), the count, the title's raw bytes, a done word last
const CAPTURES: usize = 64;
const CAPTURE_BYTES: u64 = 1024;

/// what was captured, its readback slot (None: nothing was copied), whether it is logged, and
/// the call's place in its recording (`markers::position`)
struct Captured {
    what: String,
    slot: Option<usize>,
    logged: bool,
    at: Option<(usize, u32, usize)>,
    writer_at: Option<(usize, u32, usize)>,
    raw_bytes: u64,
}

static CAPTURED: Mutex<Vec<Captured>> = Mutex::new(Vec::new());

impl Captured {
    fn call(&self) -> String {
        let place = |at: Option<(usize, u32, usize)>| at.map_or("not marked".into(), |(s, g, op)| format!("op {} of its recording, {}", op + 1, super::markers::submissions_of(s, g)));
        format!("{}; the writer's dispatch: {}", place(self.at), place(self.writer_at))
    }
}

/// Whether another indirect call may be captured.
pub(super) fn capture_wanted() -> bool {
    CAPTURED.lock().unwrap().len() < CAPTURES
}

fn capture_buffer() -> Option<(usize, u64, usize)> {
    static B: std::sync::OnceLock<Option<(usize, u64, usize)>> = std::sync::OnceLock::new();
    *B.get_or_init(|| unsafe { super::markers::readback_buffer(CAPTURES as u64 * CAPTURE_BYTES) })
}

/// After an indirect call ran on `list` from the sanitised copy `safe` (`sanitise` with capture):
/// its words copied on the GPU to a readback slot, logged with `what` once the copy has retired.
/// None: not sanitised – logged at once, nothing to copy.
pub(super) unsafe fn capture(list: &ID3D12GraphicsCommandList, safe: Option<(usize, u64, u64)>, what: String, at: Option<(usize, u32, usize)>, writer_at: Option<(usize, u32, usize)>, raw: bool) {
    let k = {
        let mut c = CAPTURED.lock().unwrap();
        if c.len() >= CAPTURES {
            return;
        }
        let k = c.len();
        let copy = safe.is_some() && capture_buffer().is_some();
        if !copy {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect capture {k}: {what}; not sanitised, so not copied");
        }
        let raw_bytes = if raw { safe.map_or(0, |(_, off, count_off)| (count_off - off).min(RAW_BYTES)) } else { 0 };
        c.push(Captured { what, slot: copy.then_some(k), logged: !copy, at, writer_at, raw_bytes });
        if !copy {
            return;
        }
        k
    };
    let (Some((res, off, count_off)), Some((cap, cap_va, _))) = (safe, capture_buffer()) else { return };
    let records = (count_off - off).min(RAW_BYTES);
    let (scratch, at) = (res as *mut c_void, k as u64 * CAPTURE_BYTES);
    let cap = cap as *mut c_void;
    let (Some(dst), Some(src)) = (ID3D12Resource::from_raw_borrowed(&cap), ID3D12Resource::from_raw_borrowed(&scratch)) else { return };
    list.ResourceBarrier(&[transition(res, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    list.CopyBufferRegion(dst, at, src, off, records);
    list.CopyBufferRegion(dst, at + RAW_BYTES, src, count_off, 4 + if raw { records } else { 0 });
    list.ResourceBarrier(&[transition(res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
    let raw = list.as_raw();
    if let Some(l2) = ID3D12GraphicsCommandList2::from_raw_borrowed(&raw) {
        let p = [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: cap_va + at + CAPTURE_BYTES - 4, Value: 0xC0DE_0000 | (k as u32 + 1) }];
        l2.WriteBufferImmediate(1, p.as_ptr(), Some([D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT].as_ptr()));
    }
    static LOGGER: std::sync::Once = std::sync::Once::new();
    LOGGER.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-ei-capture".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if log_captures(false) {
                break;
            }
        });
    });
}

/// Logs the captures whose copies retired (`all`: the rest too, as not retired); true when every
/// capture there will be is logged.
pub(super) fn log_captures(all: bool) -> bool {
    let Some((_, _, cpu)) = capture_buffer() else { return true };
    let mut c = CAPTURED.lock().unwrap();
    for e in c.iter_mut().filter(|e| !e.logged) {
        let Some(k) = e.slot else { continue };
        let words = |at: u64, n: u64| -> Vec<u32> { (0..n).map(|i| unsafe { ((cpu as u64 + k as u64 * CAPTURE_BYTES + at + 4 * i) as *const u32).read_volatile() }).collect() };
        let done = words(CAPTURE_BYTES - 4, 1)[0] == 0xC0DE_0000 | (k as u32 + 1);
        if !done && !all {
            continue;
        }
        e.logged = true;
        let (what, call) = (&e.what, e.call());
        if !done {
            crate::xlog_error!("D3D12.X: ExecuteIndirect capture {k}: {what}; the call: {call}; its copy did not retire (the list may not have been submitted)");
            continue;
        }
        let trim = |w: Vec<u32>| { let n = w.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1); format!("{:x?}{}", &w[..n], if n < w.len() { " (zeros after)" } else { "" }) };
        crate::xlog_info!(
            "D3D12.X: ExecuteIndirect capture {k}: {what}; the call: {call}; sanitised records {} and count {}; the title's words {}",
            trim(words(0, RAW_BYTES / 4)),
            words(RAW_BYTES, 1)[0],
            if e.raw_bytes != 0 { format!("{:x?}", words(RAW_BYTES + 4, e.raw_bytes / 4)) } else { "not captured".into() }
        );
    }
    c.len() >= CAPTURES && c.iter().all(|e| e.logged)
}

#[cfg(test)]
#[path = "indirect_tests.rs"]
mod tests;
