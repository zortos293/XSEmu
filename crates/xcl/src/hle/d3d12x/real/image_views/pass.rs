use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Aspect { Color, Depth, Stencil }
#[derive(Debug)]
// Render-pass conversion is not read by a consumer yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Load { Preserve, Discard, Clear(Clear) }
#[derive(Debug)]
pub(crate) enum Store { Preserve, Discard, Resolve(Resolve) }
#[derive(Debug)]
// Render-pass conversion is not read by a consumer yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Resolve {
    pub dst: ID3D12Resource, pub format: DXGI_FORMAT, pub mode: D3D12_RESOLVE_MODE,
    pub preserve_source: bool, pub regions: Vec<Region>,
}
#[derive(Debug)]
// Render-pass conversion is not read by a consumer yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Region {
    pub params: D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS,
    pub full_destination: bool,
}
#[derive(Debug)]
// Render-pass conversion is not read by a consumer yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Attachment {
    pub view: View, pub aspect: Aspect, pub plane: u32, pub read_only: bool,
    pub load: Load, pub store: Store,
}
#[derive(Debug)]
pub(crate) struct Pass {pub attachments:Vec<Attachment>}

fn rectangle(dst_width:u64,dst_height:u32,src_width:u64,src_height:u32,p:D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS)->Result<bool,Error>{
    let r=p.SrcRect;
    if r.left<0 || r.top<0 || r.right<=r.left || r.bottom<=r.top || r.right as u64>src_width || r.bottom as u32>src_height {
        return Err("unsupported or empty resolve rectangle");
    }
    let w=(r.right-r.left) as u64; let h=(r.bottom-r.top) as u64;
    if u64::from(p.DstX)+w>dst_width || u64::from(p.DstY)+h>u64::from(dst_height) {return Err("resolve outside destination")}
    Ok(p.DstX==0 && p.DstY==0 && w==dst_width && h==u64::from(dst_height))
}

unsafe fn resolve(v:&View,plane:u32,d:&D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_PARAMETERS)->Result<Resolve,Error>{
    let src=d.pSrcResource.as_ref().ok_or("missing resolve source")?;
    let dst=d.pDstResource.as_ref().ok_or("missing resolve destination")?;
    if src.as_raw()!=v.resource.as_raw() || d.SubresourceCount==0 || d.pSubresourceParameters.is_null(){
        return Err("resolve source differs from attachment or regions absent");
    }
    if !matches!(d.ResolveMode,D3D12_RESOLVE_MODE_MIN|D3D12_RESOLVE_MODE_MAX|D3D12_RESOLVE_MODE_AVERAGE){
        return Err("special resolve mode requires separate semantics");
    }
    let a=src.GetDesc();
    let b=dst.GetDesc();
    if a.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || b.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        a.SampleDesc.Count<=1 || b.SampleDesc.Count!=1 || b.MipLevels==0 || b.DepthOrArraySize==0 {
        return Err("unsupported resolve resource shape");
    }
    let dest_plane_stride=u32::from(b.MipLevels)*u32::from(b.DepthOrArraySize);
    let mut info=D3D12_FEATURE_DATA_FORMAT_INFO {Format:b.Format,PlaneCount:0};
    let mut device:Option<ID3D12Device>=None;
    dst.GetDevice(&mut device).map_err(|_|"cannot query resolve destination device")?;
    device.ok_or("missing resolve destination device")?.CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO,
        (&mut info as *mut D3D12_FEATURE_DATA_FORMAT_INFO).cast(),std::mem::size_of_val(&info) as u32)
        .map_err(|_|"unknown resolve destination plane count")?;
    if info.PlaneCount>1 && !matches!(b.Format.0,19|20|44|45) {return Err("unknown resolve plane extent")}
    let mut regions=Vec::new();
    for &p in std::slice::from_raw_parts(d.pSubresourceParameters,d.SubresourceCount as usize) {
        if !v.subs(plane).any(|s|s==p.SrcSubresource) {return Err("resolve source outside view aspect")}
        let dst_plane=p.DstSubresource/dest_plane_stride;
        if dst_plane>=u32::from(info.PlaneCount) {return Err("resolve destination subresource out of range")}
        let mip=p.DstSubresource%u32::from(b.MipLevels);
        let full_destination=rectangle((b.Width>>mip).max(1),(b.Height>>mip).max(1),(a.Width>>v.slice.mip).max(1),
            (a.Height>>v.slice.mip).max(1),p)?;
        regions.push(Region {params:p,full_destination});
    }
    Ok(Resolve {dst:dst.clone(),format:d.Format,mode:d.ResolveMode,preserve_source:d.PreserveResolveSource.as_bool(),regions})
}

unsafe fn attachment(h:usize,aspect:Aspect,start:&D3D12_RENDER_PASS_BEGINNING_ACCESS,end:&D3D12_RENDER_PASS_ENDING_ACCESS,flags:D3D12_RENDER_PASS_FLAGS)->Result<Option<Attachment>,Error>{
    if start.Type==D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS && end.Type==D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS {
        return Ok(None)
    }
    let view=snapshot(h,if aspect==Aspect::Color {Kind::Rtv} else {Kind::Dsv})?;
    let (plane,read_only)=match aspect {
        Aspect::Color=>(view.slice.plane,false),
        Aspect::Depth=>(0,view.slice.read_only_depth || flags.contains(D3D12_RENDER_PASS_FLAG_BIND_READ_ONLY_DEPTH)),
        Aspect::Stencil if view.has_stencil()=>(1,view.slice.read_only_stencil || flags.contains(D3D12_RENDER_PASS_FLAG_BIND_READ_ONLY_STENCIL)),
        Aspect::Stencil=>return Err("pass uses nonexistent stencil aspect"),
    };
    let load=match start.Type {
        D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE=>Load::Preserve,
        D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_DISCARD if !read_only=>Load::Discard,
        D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR if !read_only=>{
            let c=start.Anonymous.Clear.ClearValue;
            if c.Format!=view.slice.format {return Err("clear format differs from native view")}
            Load::Clear(match aspect {
                Aspect::Color=>Clear::Color(c.Anonymous.Color),
                Aspect::Depth=>Clear::DepthStencil {depth:Some(c.Anonymous.DepthStencil.Depth),stencil:None},
                Aspect::Stencil=>Clear::DepthStencil {depth:None,stencil:Some(c.Anonymous.DepthStencil.Stencil)},
            })
        }
        _=>return Err("unsupported beginning access or read-only clear/discard"),
    };
    let store=match end.Type {
        D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE=>Store::Preserve,
        D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_DISCARD if !read_only=>Store::Discard,
        D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE=>Store::Resolve(resolve(&view,plane,&end.Anonymous.Resolve)?),
        _=>return Err("unsupported ending access or read-only discard"),
    };
    Ok(Some(Attachment {view,aspect,plane,read_only,load,store}))
}

pub(super) unsafe fn decode(n:u32,rt:*const D3D12_RENDER_PASS_RENDER_TARGET_DESC,ds:*const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC,flags:D3D12_RENDER_PASS_FLAGS)->Result<Pass,Error>{
    let supported=D3D12_RENDER_PASS_FLAG_BIND_READ_ONLY_DEPTH | D3D12_RENDER_PASS_FLAG_BIND_READ_ONLY_STENCIL;
    if flags.0 & !supported.0 != 0 {return Err("pass continuation, UAV writes or unknown pass flags")}
    if n>8 || (n!=0 && rt.is_null()) {return Err("invalid pass target array")}
    let mut attachments=Vec::new();
    for i in 0..n as usize {
        let d=&*rt.add(i);
        attachments.extend(attachment(d.cpuDescriptor.ptr,Aspect::Color,&d.BeginningAccess,&d.EndingAccess,flags)?);
    }
    if let Some(d)=ds.as_ref(){
        attachments.extend(attachment(d.cpuDescriptor.ptr,Aspect::Depth,&d.DepthBeginningAccess,&d.DepthEndingAccess,flags)?);
        attachments.extend(attachment(d.cpuDescriptor.ptr,Aspect::Stencil,&d.StencilBeginningAccess,&d.StencilEndingAccess,flags)?);
    }
    Ok(Pass {attachments})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn resolve_coverage_is_about_the_destination_not_the_source() {
        let mut p=D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS {SrcRect:RECT {left:2,top:3,right:12,bottom:9},..Default::default()};
        assert_eq!(rectangle(10,6,16,16,p),Ok(true));
        assert_eq!(rectangle(11,6,16,16,p),Ok(false));
        p.DstX=2;
        assert!(rectangle(11,6,16,16,p).is_err());
        assert!(rectangle(20,6,11,16,p).is_err());
    }
}
