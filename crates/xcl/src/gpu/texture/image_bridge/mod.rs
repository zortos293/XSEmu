use crate::gpu::tiling::Layout;
use crate::hle::xg;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, Interface};

pub(crate) mod classes;
pub(crate) mod commands;
pub(crate) mod ownership;
pub(crate) mod lifecycle;
mod shaders;
mod stencil;

// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
pub(super) const PAGE: u64 = 1 << 16;

const IMAGE_PLACEMENT: GUID = GUID::from_u128(0xec184eaa_aecd_4cd4_9712_73827aae2ce7);

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct GuestImage {
    pub va: u64,
    pub desc: [u8; 0x38],
    pub host: ownership::HostBacking,
    pub initial: [u32; 2],
}

#[derive(Clone, Copy)]
// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum ImageUse { Sample, RenderTarget, DepthStencil, UnorderedAccess, Copy, Resolve }

impl GuestImage {
    // The image-bridge plan has no caller until the title path is wired; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn plane_for_sub(self, native:&D3D12_RESOURCE_DESC, sub:u32) -> Result<u32,Error> {
        if native.SampleDesc.Count!=1 { return Err(Error::Class(classes::Unsupported::Msaa)); }
        let mips=u32::from(native.MipLevels);
        let layers=if native.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE3D {1}else{u32::from(native.DepthOrArraySize)};
        let stride=mips.checked_mul(layers).filter(|&n| n!=0).ok_or(Error::Shape)?;
        let pc_plane=sub/stride;
        let t=xg::Texture::new(&self.desc).map_err(|_|Error::Layout)?;
        let guest=u32::from_le_bytes(self.desc[32..36].try_into().unwrap());
        let flags=u32::from_le_bytes(self.desc[48..52].try_into().unwrap());
        let host=native.Format.0 as u32;
        let mut found=None;
        for (i,p) in t.planes.iter().enumerate() {
            if t.plane_surface(i as u32).is_none(){continue}
            let mapped=stencil::native_plane(guest,host,flags,1,p.usage)
                .or_else(||classes::transfer(guest,host,flags,1,p.usage).ok().map(|c|c.0));
            if mapped==Some(pc_plane) {
                if found.replace(i as u32).is_some(){return Err(Error::Layout)}
            }
        }
        found.ok_or(Error::Class(classes::Unsupported::PlaneUsage))
    }
    // The image-bridge plan has no caller until the title path is wired; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn permits(&self, role:ImageUse) -> bool {
        let flags = u32::from_le_bytes(self.desc[48..52].try_into().unwrap());
        match role {
            ImageUse::Sample => flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 as u32 == 0,
            ImageUse::RenderTarget => flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 as u32 != 0,
            ImageUse::DepthStencil => flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 as u32 != 0,
            ImageUse::UnorderedAccess => flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32 != 0,
            ImageUse::Copy | ImageUse::Resolve => true,
        }
    }
    pub(crate) unsafe fn tag(self, resource: *mut std::ffi::c_void) -> windows::core::Result<()> {
        let r:ID3D12Resource = windows::core::IUnknown::from_raw_borrowed(&resource).ok_or(windows::Win32::Foundation::E_POINTER)?.cast()?;
        r.SetPrivateData(&IMAGE_PLACEMENT, std::mem::size_of::<Self>() as u32, Some((&self as *const Self).cast()))
    }

    pub(crate) unsafe fn get(r: &ID3D12Resource) -> windows::core::Result<Self> {
        let mut out = Self { va: 0, desc: [0; 0x38], host: ownership::HostBacking::TILED, initial:[0;2] };
        let mut len = std::mem::size_of::<Self>() as u32;
        r.GetPrivateData(&IMAGE_PLACEMENT, &mut len, Some((&mut out as *mut Self).cast()))?;
        if len != std::mem::size_of::<Self>() as u32 {
            return Err(windows::Win32::Foundation::E_INVALIDARG.into());
        }
        Ok(out)
    }
}

// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Subresource {
    pub sub: u32,
    pub footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub words: [u32; 68],
}

// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Plan {
    pub host: ownership::HostBacking,
    pub page_base: u64,
    pub page_count: u32,
    pub footprint_bytes: u64,
    pub subs: Vec<Subresource>,
}

#[derive(Debug)]
// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Error {
    Class(classes::Unsupported),
    Layout,
    Shape,
    Address,
    Footprint,
}

// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
fn equation(layout: &Layout) -> [u32; 48] {
    let mut basis = [0; 48];
    for (axis, table) in basis.chunks_exact_mut(16).enumerate() {
        for (bit, contribution) in table.iter_mut().enumerate() {
            let mut xyz = [0; 3];
            xyz[axis] = 1 << bit;
            let offset = layout.offset_z(xyz[0], xyz[1], xyz[2], layout.block_w);
            *contribution = ((offset ^ layout.base_xor) & (layout.block_bytes - 1)) as u32;
        }
    }
    basis
}

// The image-bridge plan has no caller until the title path is wired; kept for the next fix.
#[allow(dead_code)]
fn words(surface: &xg::Surface, mip: usize, slice: u32, plane_at: u64, page_delta: u64,
         offset: u64, pitch: u32, rows: u32, row_bytes: u64, depth: u32, conversion: u32, basis: &[u32; 48]) -> Result<[u32; 68], Error> {
    let c = &surface.chain;
    let l = &c.layout;
    let m = &c.levels[mip];
    let narrow = |n: usize| u32::try_from(n).map_err(|_| Error::Address);
    let pc_bpe = match (conversion, surface.bpe) { (0,b) => b, (1 | 2,2) => 4, (3,4) => 8, _ => return Err(Error::Footprint) };
    if m.depth != depth as usize || m.height != rows as usize || m.width as u64 * u64::from(pc_bpe) != row_bytes
        || u64::from(pitch) < row_bytes || surface.bpe != 1 << l.elem_log2 {
        return Err(Error::Footprint);
    }
    let base = page_delta.checked_add(plane_at).and_then(|a| a.checked_add(u64::from(slice) * c.slice_bytes as u64))
        .and_then(|a| a.checked_add(m.offset as u64)).ok_or(Error::Address)?;
    let mut w = [0; 68];
    w[..20].copy_from_slice(&[
        narrow(m.width)?, rows, depth, l.elem_log2,
        narrow(m.origin.0)?, narrow(m.origin.1)?, l.block_w.ilog2(), l.block_h.ilog2(),
        l.block_d.ilog2(), l.block_bytes.ilog2(), narrow(m.pitch.div_ceil(l.block_w))?, narrow(l.base_xor)?,
        narrow(c.slice_bytes)?, slice, (base % PAGE) as u32, u32::try_from(base / PAGE).map_err(|_| Error::Address)?,
        u32::try_from(offset).map_err(|_| Error::Address)?, pitch, pitch.checked_mul(rows).ok_or(Error::Footprint)?, conversion,
    ]);
    w[20..].copy_from_slice(basis);
    Ok(w)
}

impl Plan {
    // The image-bridge plan has no caller until the title path is wired; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) unsafe fn expanded_plane(dev: &ID3D12Device, native: &ID3D12Resource, image: GuestImage, plane: u32) -> Result<Self, Error> {
        let GuestImage { va, desc, host, .. } = image;
        let rd = native.GetDesc();
        let word = |i: usize| u32::from_le_bytes(desc[i..i+4].try_into().unwrap());
        let half = |i: usize| u16::from_le_bytes(desc[i..i+2].try_into().unwrap());
        if rd.Dimension.0 as u32 != word(0) || rd.Width != u64::from_le_bytes(desc[16..24].try_into().unwrap())
            || rd.Height != word(24) || rd.DepthOrArraySize != half(28) || rd.SampleDesc.Count != word(36) {
            return Err(Error::Shape);
        }
        if rd.SampleDesc.Count != 1 {
            return Err(Error::Class(if rd.SampleDesc.Count == 0 { classes::Unsupported::InvalidSampleCount } else { classes::Unsupported::Msaa }));
        }
        let texture = xg::Texture::new(&desc).map_err(|_| Error::Layout)?;
        let usage = texture.planes.get(plane as usize).ok_or(Error::Layout)?.usage;
        let stencil = stencil::native_plane(word(32),rd.Format.0 as u32,word(48),rd.SampleDesc.Count,usage);
        let (pc_plane, conversion) = match stencil {
            Some(plane) => (plane,0),
            None => classes::transfer(word(32), rd.Format.0 as u32, word(48), rd.SampleDesc.Count, usage).map_err(Error::Class)?,
        };
        let (surface, plane_at) = texture.plane_surface(plane).ok_or(Error::Layout)?;
        if stencil.is_some() && (surface.bpe != 1 || surface.side != 1) {
            return Err(Error::Layout);
        }
        let size = texture.size();
        let page_base = va & !(PAGE - 1);
        let delta = va - page_base;
        if va == 0 || va % u64::from(surface.bpe) != 0 || size == 0 || va.checked_add(size).is_none_or(|end| end > 1 << 48)
            || delta + size > u64::from(u32::MAX) {
            return Err(Error::Address);
        }
        if u32::from(rd.MipLevels) != texture.mips || surface.chain.levels.len() != texture.mips as usize {
            return Err(Error::Shape);
        }
        let count = surface.array.checked_mul(texture.mips).ok_or(Error::Shape)?;
        let first = count.checked_mul(pc_plane).ok_or(Error::Shape)?;
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut rows = vec![0u32; fps.len()];
        let mut sizes = vec![0u64; fps.len()];
        let mut bytes = 0;
        dev.GetCopyableFootprints(&rd, first, count, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(sizes.as_mut_ptr()), Some(&mut bytes));
        let bytes = bytes.checked_next_multiple_of(4).filter(|&b| b > 0 && b <= u64::from(u32::MAX)).ok_or(Error::Footprint)?;
        let basis = equation(&surface.chain.layout);
        let mut subs = Vec::with_capacity(fps.len());
        for (i, fp) in fps.into_iter().enumerate() {
            let m = i % texture.mips as usize;
            let slice = i as u32 / texture.mips;
            let w = words(surface, m, slice, plane_at, delta, fp.Offset, fp.Footprint.RowPitch, rows[i], sizes[i], fp.Footprint.Depth, conversion, &basis)?;
            subs.push(Subresource { sub: first + i as u32, footprint: fp, words: w });
        }
        Ok(Self { host, page_base, page_count: (delta + size).div_ceil(PAGE) as u32, footprint_bytes: bytes, subs })
    }
}
