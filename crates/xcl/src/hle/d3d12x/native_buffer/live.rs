use super::FallbackPage;
use crate::hle::d3d12x::real;
use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub(crate) fn fallback(resource:&ID3D12Resource,guest_base:u64,native_base:u64,page:u32)->Option<(FallbackPage,Arc<dyn Send+Sync>)>{
    let raw=resource.as_raw() as usize;
    let (base,len)=real::owned_console_buffer(raw)?;
    let delta=real::buffer_delta(raw);
    if (base as u64).checked_add(native_base.checked_sub(delta)?)?!=guest_base {return None}
    let page_start=u64::from(page)<<16;
    if page_start.checked_add(65536)?<=base as u64 || page_start >= base.checked_add(len)? as u64 {return None}
    Some((uma_page(page)?,Arc::new(resource.clone())))
}

/// A 64 KiB title page read or written in unified memory, through its window's view.
pub(crate) fn uma_page(page:u32)->Option<FallbackPage>{
    let page_start=u64::from(page)<<16;
    let window=crate::gpu::uma::get()?.window_of(page_start as usize)?;
    let (_,offset)=real::uma_window_buffer(page_start as usize,65536)?;
    let view=real::mem_views()?.checked_add(window as u32)?;
    Some(FallbackPage::Raw{view,page_base:u32::try_from(offset).ok()?})
}

#[derive(Clone)]
pub(crate) struct Scope {
    heaps:Vec<ID3D12DescriptorHeap>,
    predicate:Option<(ID3D12Resource,u64,D3D12_PREDICATION_OP)>,
}

impl Scope {
    pub(crate) fn new(heaps:Vec<ID3D12DescriptorHeap>,predicate:Option<(ID3D12Resource,u64,D3D12_PREDICATION_OP)>)->Self {
        Self{heaps,predicate}
    }
    // live::Live::packet has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn packet()->Self { Self::new(Vec::new(),None) }
    pub(crate) fn predicated(&self)->bool { self.predicate.is_some() }
}

pub(crate) unsafe fn set_pipeline(list:&ID3D12GraphicsCommandList,pso:&ID3D12PipelineState){
    let raw=list.as_raw();
    let f:unsafe extern "system" fn(*mut c_void,*mut c_void)=std::mem::transmute(real::original_or_native(raw,25).0);
    f(raw,pso.as_raw());
}

pub(crate) unsafe fn dispatch(list:&ID3D12GraphicsCommandList,x:u32,y:u32,z:u32){
    let raw=list.as_raw();
    let f:unsafe extern "system" fn(*mut c_void,u32,u32,u32)=std::mem::transmute(real::original_or_native(raw,14).0);
    f(raw,x,y,z);
}

pub(crate) unsafe fn with_helper<T>(list:&ID3D12GraphicsCommandList,proof:&Scope,emit:impl FnOnce()->Result<T,String>)->Result<T,String> {
    let raw=list.as_raw();
    let pred:unsafe extern "system" fn(*mut c_void,*mut c_void,u64,D3D12_PREDICATION_OP)=std::mem::transmute(real::original_or_native(raw,55).0);
    let set:unsafe extern "system" fn(*mut c_void,u32,*const *mut c_void)=std::mem::transmute(real::original_or_native(raw,28).0);
    if proof.predicate.is_some(){pred(raw,std::ptr::null_mut(),0,D3D12_PREDICATION_OP_EQUAL_ZERO);}
    let result=emit();
    let default=if proof.heaps.is_empty(){crate::hle::d3d12x::console_heap::global_heap()}else{None};
    let heaps:Vec<_>=proof.heaps.iter().chain(default.iter()).map(Interface::as_raw).collect();
    set(raw,heaps.len() as u32,heaps.as_ptr());
    if let Some((p,offset,op))=&proof.predicate{pred(raw,p.as_raw(),*offset,*op);}
    result
}

const POOL_KEEP_PER_CLASS: usize = 32;

static POOL: Mutex<Option<(ID3D12Device, HashMap<u64, Vec<ID3D12Resource>>)>> = Mutex::new(None);

/// Helper UAV scratch of a power-of-two size class, back in the pool once the recording that retains
/// it (and every captured execution of it) retires. Buffers start and decay to COMMON, so a
/// reused one starts each execution in COMMON and the gather's UAV writes promote it.
pub(crate) struct Pooled {
    resource: Option<ID3D12Resource>,
    class: u64,
    dev: ID3D12Device,
}

impl std::ops::Deref for Pooled {
    type Target = ID3D12Resource;
    fn deref(&self) -> &ID3D12Resource {
        self.resource.as_ref().unwrap()
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        let mut pool = POOL.lock().unwrap();
        let Some((dev, free)) = pool.as_mut() else { return };
        let kept = free.entry(self.class).or_default();
        if *dev == self.dev && kept.len() < POOL_KEEP_PER_CLASS {
            kept.extend(self.resource.take());
        }
    }
}

pub(crate) unsafe fn scratch(dev: &ID3D12Device, bytes: u64) -> Result<Pooled, String> {
    let class = bytes.max(1 << 16).checked_next_power_of_two().ok_or("native scratch size overflows")?;
    let reused = {
        let mut pool = POOL.lock().unwrap();
        if pool.as_ref().is_none_or(|(d, _)| d != dev) {
            *pool = Some((dev.clone(), HashMap::new()));
        }
        pool.as_mut().unwrap().1.get_mut(&class).and_then(Vec::pop)
    };
    if let Some(resource) = reused {
        return Ok(Pooled { resource: Some(resource), class, dev: dev.clone() });
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: class,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut out = None;
    crate::hle::d3d12x::title_textures::own_work(|| dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out))
        .map_err(|e| e.to_string())?;
    Ok(Pooled { resource: Some(out.ok_or("no native scratch")?), class, dev: dev.clone() })
}
