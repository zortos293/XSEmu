use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT};
use crate::gpu::texture::image_bridge::{GuestImage,ImageUse};

mod pass;
pub(crate) use pass::*;

pub(crate) type Error = &'static str;
pub(crate) type Sink = fn(usize, Result<Event, Error>);
static SINK: OnceLock<Sink> = OnceLock::new();

pub(crate) fn install(sink: Sink) -> Result<(), Sink> {
    SINK.set(sink)
}

pub(super) fn enabled() -> bool { SINK.get().is_some() }
pub(super) fn record(list: *mut c_void, make: impl FnOnce() -> Result<Event, Error>) {
    if let Some(sink) = SINK.get() { sink(list as usize, make()); }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind { Rtv, Dsv }

impl Kind {
    pub(super) fn of(ty: u32) -> Option<Self> {
        match D3D12_DESCRIPTOR_HEAP_TYPE(ty as i32) {
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV => Some(Self::Rtv),
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV => Some(Self::Dsv),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Slice {
    pub mip: u32, pub first: u32, pub count: u32, pub plane: u32,
    pub w_slices: Option<(u32, u32, u32)>,
    pub format: DXGI_FORMAT, pub read_only_depth: bool, pub read_only_stencil: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct View {
    pub resource: ID3D12Resource, pub slice: Slice, pub generation: u64,
    pub mips: u32, pub layers: u32,
}

impl View {
    pub(crate) fn subs(&self, plane: u32) -> impl Iterator<Item=u32> + '_ {
        let s = self.slice;
        (s.first..s.first+s.count).map(move |layer| s.mip + self.mips * (layer + self.layers * plane))
    }
    pub(crate) fn covers_full_subresources(&self) -> bool {
        self.slice.w_slices.is_none_or(|(first,count,total)| first == 0 && count == total)
    }
    pub(crate) fn has_stencil(&self) -> bool { matches!(self.slice.format.0, 20 | 45) }
}

#[derive(Clone, Copy)]
struct Entry { owner: usize, slice: Slice, mips: u32, layers: u32, generation: u64 }
struct Heap { owner: usize, device: usize, kind: Kind, step: usize, end: usize, generation: u64 }
#[derive(Default)]
struct Table {
    next: u64, heaps: BTreeMap<usize, Heap>, slots: BTreeMap<usize, Entry>, twins: HashMap<usize,usize>,
}

impl Table {
    fn generation(&mut self) -> u64 {
        self.next = self.next.checked_add(1).expect("descriptor generation exhausted");
        self.next
    }
    fn heap(&self, h: usize, kind: Kind) -> Option<&Heap> {
        self.heaps.range(..=h).next_back().and_then(|(first,heap)|
            (heap.kind == kind && h < heap.end && (h-first)%heap.step == 0).then_some(heap))
    }
    fn remove_heap(&mut self, first: usize) {
        if let Some(heap) = self.heaps.remove(&first) {
            self.slots.retain(|&h,_| h < first || h >= heap.end);
        }
    }
    fn put(&mut self, device: usize, h: usize, kind: Kind, e: Option<Entry>) {
        self.slots.remove(&h);
        if let Some(mut e) = e.filter(|_| self.heap(h,kind).is_some_and(|heap| heap.device == device)) {
            e.generation = self.generation();
            self.slots.insert(h,e);
        }
    }
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(Default::default);

pub(super) fn heap(owner: usize, device: usize, kind: Kind, first: usize, step: usize, count: u32) {
    if !enabled() || first == 0 || step == 0 || count == 0 { return }
    let Some(end) = step.checked_mul(count as usize).and_then(|n| first.checked_add(n)) else {return};
    let mut t = TABLE.lock().unwrap();
    let overlaps: Vec<_> = t.heaps.range(..end).filter_map(|(&f,h)| (first<h.end).then_some(f)).collect();
    for f in overlaps { t.remove_heap(f) }
    let generation = t.generation();
    t.heaps.insert(first,Heap {owner,device,kind,step,end,generation});
}

pub(super) fn heap_id(owner: usize) -> Option<(usize,u64)> {
    if !enabled() { return None }
    TABLE.lock().unwrap().heaps.iter().find_map(|(&f,h)| (h.owner==owner).then_some((f,h.generation)))
}
pub(super) fn forget_heap((first,id): (usize,u64)) {
    let mut t = TABLE.lock().unwrap();
    if t.heaps.get(&first).is_some_and(|h| h.generation == id) { t.remove_heap(first); }
}
pub(super) fn forget_resource(owner: usize) {
    if !enabled() { return }
    let mut t = TABLE.lock().unwrap();
    t.slots.retain(|_,e| e.owner != owner);
    t.twins.retain(|&native, original| native != owner && *original != owner);
}
pub(super) fn note_twin(native: usize, original: usize) {
    if enabled() { TABLE.lock().unwrap().twins.insert(native,original); }
}
pub(crate) fn shadow_origin(actual_native: &ID3D12Resource) -> Result<Option<ID3D12Resource>,Error> {
    let t = TABLE.lock().unwrap();
    let Some(&origin) = t.twins.get(&(actual_native.as_raw() as usize)) else {return Ok(None)};
    unsafe { super::try_reference_depth_capture(origin) }.map(Some).ok_or("depth shadow owner retiring")
}

pub(crate) struct Operand {
    pub native:ID3D12Resource, pub tag_owner:ID3D12Resource, pub image:GuestImage,
    pub sub:u32, pub guest_plane:u32, pub is_twin:bool,
}

pub(crate) unsafe fn operand(actual:&ID3D12Resource, role:ImageUse, sub:u32)->Result<Operand,Error>{
    let origin=shadow_origin(actual)?;
    let rd=actual.GetDesc();
    let image=GuestImage::get(actual).map_err(|_|"no guest image creation tag")?;
    let is_twin=origin.is_some();
    let owner=origin.unwrap_or_else(||actual.clone());
    if is_twin {
        let od=owner.GetDesc();
        if !matches!(role,ImageUse::UnorderedAccess|ImageUse::Sample)
            || !matches!(od.Format.0,39|53) || od.Format!=rd.Format
            || od.Dimension!=rd.Dimension || od.Width!=rd.Width || od.Height!=rd.Height
            || od.DepthOrArraySize!=rd.DepthOrArraySize || od.MipLevels!=rd.MipLevels
            || od.SampleDesc.Count!=1 || rd.SampleDesc.Count!=1
            || sub>=u32::from(rd.MipLevels)*u32::from(rd.DepthOrArraySize) {
            return Err("converted, multisampled or mismatched depth twin");
        }
        let o=GuestImage::get(&owner).map_err(|_|"depth twin origin has no guest image tag")?;
        if image.va!=o.va || image.desc!=o.desc || image.initial!=o.initial {
            return Err("depth twin guest provenance differs from origin");
        }
        if image.host!=crate::gpu::texture::image_bridge::ownership::HostBacking::PRIVATE {
            return Err("depth twin tag is not private host backing");
        }
    }
    if !image.permits(role){return Err("guest image role not permitted")}
    let guest_plane=image.plane_for_sub(&rd,sub).map_err(|_|"guest plane not proven")?;
    Ok(Operand {native:actual.clone(),tag_owner:owner,image,sub,guest_plane,is_twin})
}

enum Desc<'a> { Rtv(Option<&'a D3D12_RENDER_TARGET_VIEW_DESC>), Dsv(Option<&'a D3D12_DEPTH_STENCIL_VIEW_DESC>) }

unsafe fn describe(rd: D3D12_RESOURCE_DESC, desc: Desc<'_>) -> Option<Slice> {
    let mut s = Slice { mip:0,first:0,count:1,plane:0,format:rd.Format,w_slices:None, read_only_depth:false,read_only_stencil:false };
    let volume = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let layers = if volume {1} else {u32::from(rd.DepthOrArraySize)};
    let end = |first:u32, count:u32, total:u32| {
        let count = if count == u32::MAX {total.checked_sub(first)?} else {count};
        (count != 0 && first.checked_add(count)? <= total).then_some(count)
    };
    if rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER { return None }
    match desc {
        Desc::Rtv(None) | Desc::Dsv(None) => {
            s.count = layers;
            if volume {s.w_slices=Some((0,rd.DepthOrArraySize.into(),rd.DepthOrArraySize.into()))}
        }
        Desc::Dsv(Some(d)) => {
            if d.Format.0 != 0 { s.format=d.Format; }
            s.read_only_depth = d.Flags.contains(D3D12_DSV_FLAG_READ_ONLY_DEPTH);
            s.read_only_stencil = d.Flags.contains(D3D12_DSV_FLAG_READ_ONLY_STENCIL);
            (s.mip,s.first,s.count) = match d.ViewDimension {
                D3D12_DSV_DIMENSION_TEXTURE1D => (d.Anonymous.Texture1D.MipSlice,0,1),
                D3D12_DSV_DIMENSION_TEXTURE2D => (d.Anonymous.Texture2D.MipSlice,0,1),
                D3D12_DSV_DIMENSION_TEXTURE1DARRAY => {let a=d.Anonymous.Texture1DArray;(a.MipSlice,a.FirstArraySlice,a.ArraySize)}
                D3D12_DSV_DIMENSION_TEXTURE2DARRAY => {let a=d.Anonymous.Texture2DArray;(a.MipSlice,a.FirstArraySlice,a.ArraySize)}
                D3D12_DSV_DIMENSION_TEXTURE2DMS => (0,0,1),
                D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY => {let a=d.Anonymous.Texture2DMSArray;(0,a.FirstArraySlice,a.ArraySize)}
                _ => return None,
            };
        }
        Desc::Rtv(Some(d)) => {
            if d.Format.0 != 0 { s.format=d.Format; }
            (s.mip,s.first,s.count,s.plane) = match d.ViewDimension {
                D3D12_RTV_DIMENSION_TEXTURE1D => (d.Anonymous.Texture1D.MipSlice,0,1,0),
                D3D12_RTV_DIMENSION_TEXTURE2D => {let a=d.Anonymous.Texture2D;(a.MipSlice,0,1,a.PlaneSlice)}
                D3D12_RTV_DIMENSION_TEXTURE1DARRAY => {let a=d.Anonymous.Texture1DArray;(a.MipSlice,a.FirstArraySlice,a.ArraySize,0)}
                D3D12_RTV_DIMENSION_TEXTURE2DARRAY => {let a=d.Anonymous.Texture2DArray;(a.MipSlice,a.FirstArraySlice,a.ArraySize,a.PlaneSlice)}
                D3D12_RTV_DIMENSION_TEXTURE2DMS => (0,0,1,0),
                D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY => {let a=d.Anonymous.Texture2DMSArray;(0,a.FirstArraySlice,a.ArraySize,0)}
                D3D12_RTV_DIMENSION_TEXTURE3D if volume => {
                    let a=d.Anonymous.Texture3D;
                    let total=u32::from(rd.DepthOrArraySize).checked_shr(a.MipSlice)?.max(1);
                    s.w_slices=Some((a.FirstWSlice,end(a.FirstWSlice,a.WSize,total)?,total));
                    (a.MipSlice,0,1,0)
                }
                _ => return None,
            };
        }
    }
    s.count=end(s.first,s.count,layers)?;
    if s.plane>2 || s.mip>=u32::from(rd.MipLevels) {return None}
    Some(s)
}

unsafe fn put_view(device:usize, h:usize, raw:*mut c_void, kind:Kind, d:Desc<'_>) {
    if !enabled() {return}
    let e = (|| {
        if raw.is_null() || !super::is_adopted(raw) {return None}
        let rd = ID3D12Resource::from_raw_borrowed(&raw)?.GetDesc();
        let slice=describe(rd,d)?;
        Some(Entry {owner:raw as usize,slice,mips:rd.MipLevels.into(),layers:if rd.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE3D {1} else {rd.DepthOrArraySize.into()},generation:0})
    })();
    TABLE.lock().unwrap().put(device,h,kind,e);
}

pub(super) unsafe fn rtv(device:usize,h:usize,r:*mut c_void,d:Option<&D3D12_RENDER_TARGET_VIEW_DESC>){put_view(device,h,r,Kind::Rtv,Desc::Rtv(d))}
pub(super) unsafe fn dsv(device:usize,h:usize,r:*mut c_void,d:Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>){put_view(device,h,r,Kind::Dsv,Desc::Dsv(d))}

pub(super) unsafe fn copies(device:usize,kind:Kind,nd:u32,dst:*const u64,ds:*const u32,ns:u32,src:*const u64,ss:*const u32) {
    if !enabled() || nd == 0 || dst.is_null() {return}
    let mut t=TABLE.lock().unwrap();
    let ranges=|n:u32, starts:*const u64,sizes:*const u32| -> Vec<(usize,u32)> {
        if n==0 || starts.is_null() {return Vec::new()}
        (0..n as usize).map(|i|(*starts.add(i) as usize,if sizes.is_null(){1}else{*sizes.add(i)})).collect()
    };
    let expands=|ranges:Vec<(usize,u32)>| {
        ranges.into_iter().flat_map(|(first,n)| {
            let heap=t.heap(first,kind).filter(|h|h.device==device);
            (0..n as usize).map(move |i|heap.and_then(|h| first.checked_add(i.checked_mul(h.step)?).filter(|&v|v<h.end)))
        }).collect::<Vec<_>>()
    };
    let d=expands(ranges(nd,dst,ds));
    let s=expands(ranges(ns,src,ss));
    let values:Vec<_>=s.iter().map(|h|h.and_then(|i|t.slots.get(&i).copied())).collect();
    for (i,h) in d.into_iter().enumerate() {
        if let Some(h)=h {t.put(device,h,kind,values.get(i).copied().flatten());}
    }
}

pub(crate) fn snapshot(h:usize,kind:Kind)->Result<View,Error>{
    let t=TABLE.lock().unwrap();
    t.heap(h,kind).ok_or("untracked attachment heap")?;
    let e=t.slots.get(&h).ok_or("untracked or null attachment descriptor")?;
    let resource=unsafe {super::try_reference_depth_capture(e.owner)}.ok_or("attachment retiring")?;
    Ok(View {resource,slice:e.slice,generation:e.generation,mips:e.mips,layers:e.layers})
}

#[derive(Debug)]
pub(crate) enum Event {
    Bind { colors:Vec<View>, depth:Option<View> },
    Clear { view:View, value:Clear, rects:Vec<RECT> },
    Begin(Pass),
    End,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Clear { Color([f32;4]), DepthStencil {depth:Option<f32>, stencil:Option<u8>} }

pub(super) unsafe fn bind(n:u32,handles:*const usize,single:i32,ds:*const usize)->Result<Event,Error>{
    if n>8 || (n>0 && handles.is_null()) {return Err("invalid render target count")}
    let mut colors=Vec::new();
    for i in 0..n as usize {
        let handle=if single==0 {*handles.add(i)} else {
            let t=TABLE.lock().unwrap();
            let h=t.heap(*handles,Kind::Rtv).ok_or("unknown contiguous RTV heap")?;
            handles.read().checked_add(i.checked_mul(h.step).ok_or("RTV increment")?).filter(|&n|n<h.end).ok_or("RTV outside heap")?
        };
        colors.push(snapshot(handle,Kind::Rtv)?);
    }
    Ok(Event::Bind {colors,depth:ds.as_ref().map(|v|snapshot(*v,Kind::Dsv)).transpose()?})
}

pub(super) unsafe fn clear(h:usize,value:Clear,n:u32,p:*const RECT)->Result<Event,Error> {
    let kind=if matches!(value,Clear::Color(_)) {Kind::Rtv} else {Kind::Dsv};
    let view=snapshot(h,kind)?;
    if let Clear::DepthStencil {depth,stencil}=value {
        if (depth.is_none() && stencil.is_none()) || (depth.is_some() && view.slice.read_only_depth)
            || (stencil.is_some() && (!view.has_stencil() || view.slice.read_only_stencil)) {return Err("clear of absent or read-only aspect")}
    }
    let rects=if n==0 {vec![]} else if p.is_null(){return Err("missing clear rectangles")} else {std::slice::from_raw_parts(p,n as usize).to_vec()};
    Ok(Event::Clear {view,value,rects})
}

pub(super) unsafe fn begin(n:u32,rt:*const D3D12_RENDER_PASS_RENDER_TARGET_DESC,ds:*const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC,f:D3D12_RENDER_PASS_FLAGS)->Result<Event,Error>{
    pass::decode(n,rt,ds,f).map(Event::Begin)
}

pub(super) unsafe extern "system" fn pc_create_dsv(this:*mut c_void,res:*mut c_void,desc:*const D3D12_DEPTH_STENCIL_VIEW_DESC,h:usize){
    let f:unsafe extern "system" fn(*mut c_void,*mut c_void,*const D3D12_DEPTH_STENCIL_VIEW_DESC,usize)=std::mem::transmute(super::orig_fn(this,21));
    f(this,res,desc,h);
    dsv(this as usize,h,res,desc.as_ref());
}
pub(super) unsafe fn copied(this:*mut c_void,ty:u32,nd:u32,dst:*const u64,ds:*const u32,ns:u32,src:*const u64,ss:*const u32){
    if let Some(kind)=Kind::of(ty) {copies(this as usize,kind,nd,dst,ds,ns,src,ss)}
}
pub(super) unsafe extern "system" fn pc_copy(this:*mut c_void,nd:u32,dst:*const u64,ds:*const u32,ns:u32,src:*const u64,ss:*const u32,ty:u32){
    let f:unsafe extern "system" fn(*mut c_void,u32,*const u64,*const u32,u32,*const u64,*const u32,u32)=std::mem::transmute(super::orig_fn(this,23));
    f(this,nd,dst,ds,ns,src,ss,ty);
    copied(this,ty,nd,dst,ds,ns,src,ss);
}
pub(super) unsafe extern "system" fn pc_copy_simple(this:*mut c_void,n:u32,dst:u64,src:u64,ty:u32){
    let f:unsafe extern "system" fn(*mut c_void,u32,u64,u64,u32)=std::mem::transmute(super::orig_fn(this,24));
    f(this,n,dst,src,ty);
    copied(this,ty,1,&dst,&n,1,&src,&n);
}
unsafe fn record_dsv_clear(this:*mut c_void,h:usize,flags:u32,d:f32,s:u8,n:u32,p:*const RECT){
    super::super::dsv_trace::clear(h,flags);
    record(this,||{
        if flags & !3 != 0 {return Err("unknown DSV clear flags")}
        clear(h,Clear::DepthStencil{depth:(flags&1!=0).then_some(d),stencil:(flags&2!=0).then_some(s)},n,p)
    });
}
pub(super) unsafe extern "system" fn clear_dsv_pc(this:*mut c_void,h:usize,flags:u32,d:f32,s:u8,n:u32,p:*const RECT){
    record_dsv_clear(this,h,flags,d,s,n,p);
    super::super::depth_uav::pc_clear_depth_stencil(this,h,flags,d,s,n,p);
}
pub(super) unsafe extern "system" fn clear_dsv_guest(this:*mut c_void,h:usize,flags:u32,d:f32,s:u8,n:u32,p:*const RECT){
    record_dsv_clear(this,h,flags,d,s,n,p);
    super::super::list_x::clear_depth_stencil_x(this,h,flags,d,s,n,p);
}
