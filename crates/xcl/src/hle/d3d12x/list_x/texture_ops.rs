use std::{ffi::c_void, mem::ManuallyDrop, sync::OnceLock};
use windows::{core::Interface, Win32::{Foundation::RECT, Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT}}};

mod lifecycle;

#[derive(Clone, Copy, Debug)]
pub(crate) enum RecordingPoint {
    Adopted,
    // List-X texture-op scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    Console { chunk: usize, dword: usize },
}

#[derive(Clone, Copy)]
// List-X texture-op scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Site {
    pub list: usize,
    pub point: RecordingPoint,
    pub kind: D3D12_COMMAND_LIST_TYPE,
}

/// A subresource index is already native and includes its plane. A footprint
/// keeps both offsets: the console's byte offset and the native buffer placement.
#[derive(Clone)]
pub(crate) enum Layout {
    Subresource(u32),
    Footprint { guest: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, native: D3D12_PLACED_SUBRESOURCE_FOOTPRINT },
}

#[derive(Clone)]
pub(crate) struct Location {
    pub resource: ID3D12Resource,
    pub layout: Layout,
}

impl Location {
    unsafe fn capture(l: &D3D12_TEXTURE_COPY_LOCATION) -> Result<Self, String> {
        let resource = l.pResource.as_ref().ok_or("texture copy without a resource")?.clone();
        let layout = match l.Type {
            D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX => Layout::Subresource(l.Anonymous.SubresourceIndex),
            D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT => {
                let guest = l.Anonymous.PlacedFootprint;
                let mut native = guest;
                native.Offset = guest.Offset.checked_add(super::real::buffer_delta(resource.as_raw() as usize)).ok_or("texture copy placement offset overflow")?;
                Layout::Footprint { guest, native }
            }
            _ => return Err("unknown texture copy location kind".into()),
        };
        Ok(Self { resource, layout })
    }

    fn with_location<T>(&self, adjust: bool, f: impl FnOnce(&D3D12_TEXTURE_COPY_LOCATION) -> T) -> T {
        let (ty, data) = match self.layout {
            Layout::Subresource(i) => (D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i }),
            Layout::Footprint { guest, native } => (D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: if adjust { native } else { guest } }),
        };
        let mut loc = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(self.resource.clone())), Type: ty, Anonymous: data };
        let result = f(&loc);
        unsafe { ManuallyDrop::drop(&mut loc.pResource); }
        result
    }

    /// Native plane dimensions, including planar subsampling; not the parent
    /// texture's mip-zero shape. Used when no source box/rect was specified.
    // List-X texture-op scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn extent(&self) -> Result<[u32; 3], String> {
        let f = match self.layout {
            Layout::Footprint { native, .. } => native.Footprint,
            Layout::Subresource(i) => {
                let mut d = self.resource.GetDesc();
                d.SampleDesc.Count = 1;
                d.SampleDesc.Quality = 0;
                d.Alignment = 0;
                let mut dev: Option<ID3D12Device> = None;
                self.resource.GetDevice(&mut dev).map_err(|e| e.to_string())?;
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut bytes = 0;
                dev.ok_or("no resource device")?.GetCopyableFootprints(&d, i, 1, 0, Some(&mut fp), None, None, Some(&mut bytes));
                if bytes == u64::MAX || bytes == 0 { return Err("invalid native subresource footprint".into()); }
                if d.MipLevels == 0 { return Err("unknown mip level count".into()); }
                let mip = i % u32::from(d.MipLevels);
                fp.Footprint.Width = fp.Footprint.Width.min(u32::try_from((d.Width >> mip).max(1)).map_err(|_| "texture width overflow")?);
                fp.Footprint.Height = fp.Footprint.Height.min((d.Height >> mip).max(1));
                fp.Footprint
            }
        };
        Ok([f.Width, f.Height, f.Depth])
    }
}

#[derive(Clone)]
pub(crate) enum Operation {
    Copy { dst: Location, origin: [u32; 3], src: Location, source_box: Option<D3D12_BOX>, flags: u32 },
    // List-X texture-op scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    Resolve { dst: Location, src: Location, format: DXGI_FORMAT, region: Option<(u32, u32, Option<RECT>, D3D12_RESOLVE_MODE)> },
}

impl Operation {
    pub(crate) unsafe fn copy(dst: *const D3D12_TEXTURE_COPY_LOCATION, origin: [u32; 3], src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX, flags: u32) -> Result<Self, String> {
        Ok(Self::Copy { dst: Location::capture(dst.as_ref().ok_or("missing destination location")?)?, origin,
            src: Location::capture(src.as_ref().ok_or("missing source location")?)?, source_box: bx.as_ref().copied(), flags })
    }

    pub(super) unsafe fn resolve(dst: usize, dst_sub: u32, src: usize, src_sub: u32, format: i32, region: Option<(u32,u32,Option<RECT>,D3D12_RESOLVE_MODE)>) -> Result<Self, String> {
        let sub = |address, index| -> Result<Location, String> {
            Ok(Location { resource: ID3D12Resource::from_raw_borrowed(&(address as *mut c_void)).ok_or("missing resolve resource")?.clone(), layout: Layout::Subresource(index) })
        };
        Ok(Self::Resolve { dst: sub(dst, dst_sub)?, src: sub(src, src_sub)?, format: DXGI_FORMAT(format), region })
    }

    fn permits(&self, kind: D3D12_COMMAND_LIST_TYPE) -> bool {
        match self {
            Self::Copy { flags: 0, .. } => matches!(kind, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE | D3D12_COMMAND_LIST_TYPE_COPY),
            Self::Resolve { .. } => kind == D3D12_COMMAND_LIST_TYPE_DIRECT,
            _ => false,
        }
    }

    // List-X texture-op scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn regions(&self) -> Result<(D3D12_BOX,D3D12_BOX),String> {
        let (src, origin, specified) = match self {
            Self::Copy { src, origin, source_box, .. } => (src, *origin, *source_box),
            Self::Resolve { src, region, .. } => {
                let (x,y,r) = region.as_ref().map_or((0,0,None), |(x,y,r,_)| (*x,*y,*r));
                let b = if let Some(r) = r {
                    Some(D3D12_BOX { left: r.left.try_into().map_err(|_| "negative resolve rect")?, top: r.top.try_into().map_err(|_| "negative resolve rect")?,
                        right: r.right.try_into().map_err(|_| "negative resolve rect")?, bottom: r.bottom.try_into().map_err(|_| "negative resolve rect")?, front: 0, back: 1 })
                } else { None };
                (src, [x,y,0], b)
            }
        };
        let b = if let Some(b) = specified { b } else {
            let [right,bottom,back] = src.extent()?;
            D3D12_BOX { left: 0, top: 0, front: 0, right, bottom, back }
        };
        let [left, top, front] = origin;
        let end = |start:u32, finish:u32, at:u32| finish.checked_sub(start).and_then(|n| at.checked_add(n)).ok_or("invalid texture region");
        Ok((b, D3D12_BOX { left, top, front, right: end(b.left,b.right,left)?, bottom: end(b.top,b.bottom,top)?, back: end(b.front,b.back,front)? }))
    }

    unsafe fn copy_into(&self, list: &ID3D12GraphicsCommandList, adjust: bool) -> Result<(),String> {
        let Self::Copy { dst, origin: [x,y,z], src, source_box, .. } = self else { return Err("not a texture copy".into()) };
        dst.with_location(adjust, |d| src.with_location(adjust, |s| list.CopyTextureRegion(d, *x, *y, *z, s, source_box.as_ref().map(|b| b as *const _))));
        Ok(())
    }

    pub(crate) unsafe fn unaffected_console_copy(&self, list: &ID3D12GraphicsCommandList) -> Result<(),String> {
        let Self::Copy {dst, src, ..} = self else { return Err("not a console texture copy".into()) };
        super::super::residency::note_list_resources(list.as_raw() as usize, [dst.resource.as_raw() as usize, src.resource.as_raw() as usize]);
        super::super::vram::note_gpu_write(dst.resource.as_raw() as usize);
        self.copy_into(list, false)
    }

    /// Replay only on a fresh, unpatched native list at the certified cut:
    /// outside a pass, unpredicated, both endpoints already in their exact
    /// required states. The scheduler must retain the operation through fence completion.
    // List-X texture-op scaffolding has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn issue(&self, list: &ID3D12GraphicsCommandList) -> Result<(), String> {
        if super::real::is_adopted(list.as_raw()) || !self.permits(list.GetType()) {
            return Err("unsupported native texture replay list or copy flags".into());
        }
        match self {
            Self::Copy { .. } => self.copy_into(list, true)?,
            Self::Resolve { dst, src, format, region } => {
                let (Layout::Subresource(d), Layout::Subresource(s)) = (&dst.layout, &src.layout) else { return Err("resolve requires two subresources".into()) };
                if let Some((x,y,rect,mode)) = region {
                    list.cast::<ID3D12GraphicsCommandList1>().map_err(|e| e.to_string())?.ResolveSubresourceRegion(&dst.resource, *d, *x, *y,
                        &src.resource, *s, rect.as_ref().map(|r| r as *const _), *format, *mode);
                } else {
                    list.ResolveSubresource(&dst.resource, *d, &src.resource, *s, *format);
                }
            }
        }
        Ok(())
    }
}

/// The implementor owns list generations and queue ordering. Unaffected must
/// remain true for every submission of this recording, not just today's PTE.
/// defer must commit the exact prefix/operation/suffix cut or return Err;
/// it must exclude active/suspended render passes, split barriers, queries,
/// and predication (including state set outside adopted hooks). It cannot
/// reenter console translation while handling a Console recording point.
pub(crate) unsafe trait OperationCuts: Sync {
    fn proven_unaffected(&self, site: Site, operation: &Operation) -> bool;
    fn defer(&self, site: Site, operation: Operation) -> Result<(), String>;
}

static CUTS: OnceLock<&'static dyn OperationCuts> = OnceLock::new();

// List-X texture-op scaffolding has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn install(cuts: &'static dyn OperationCuts) -> bool { CUTS.set(cuts).is_ok() }
pub(crate) fn installed() -> bool { CUTS.get().is_some() }

pub(crate) unsafe fn intercept(list: *mut c_void, point: RecordingPoint, op: impl FnOnce() -> Result<Operation,String>) -> Result<bool,String> {
    let Some(cuts) = CUTS.get() else { return Ok(false) };
    let op = op()?;
    let kind = match point {
        RecordingPoint::Adopted => super::list_type(list),
        RecordingPoint::Console { .. } => ID3D12GraphicsCommandList::from_raw_borrowed(&list).ok_or("missing console translation list")?.GetType(),
    };
    let site = Site { list: list as usize, point, kind };
    if cuts.proven_unaffected(site, &op) { return Ok(false); }
    if !op.permits(kind) { return Err("affected native operation requires an unsupported queue or copy flags".into()); }
    let blocked = super::LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&site.list)).is_some_and(|s|
        s.bindings.predication.is_some() || s.predication.is_some() || s.draw_predicate.is_some()
            || s.occlusion.active.is_some() || s.bindings.lineage_pass_dsv.is_some());
    if blocked { return Err("affected native texture operation under a predicate, query, or render pass".into()); }
    cuts.defer(site, op)?;
    Ok(true)
}
