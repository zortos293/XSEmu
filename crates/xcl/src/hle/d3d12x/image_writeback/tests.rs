// SPDX-License-Identifier: GPL-3.0-or-later
// (part of image_writeback.rs; see its provenance notice)

use super::*;
use crate::gpu::texture::image_bridge::ownership::HostBacking;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

fn console_desc(w: u64, h: u32, array: u16, mips: u16, format: u32, samples: u32, flags: u32) -> [u8; 0x38] {
    let mut d = [0u8; 0x38];
    d[0..4].copy_from_slice(&3u32.to_le_bytes());
    d[0x10..0x18].copy_from_slice(&w.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
    d[0x1C..0x1E].copy_from_slice(&array.to_le_bytes());
    d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
    d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
    d[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
    d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
    d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    d
}

unsafe fn texture(dev: &ID3D12Device, w: u64, h: u32, array: u16, mips: u16) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: w,
        Height: h,
        DepthOrArraySize: array,
        MipLevels: mips,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
    r.unwrap()
}

/// The GPU writes every subresource of a texture placed over physical pages (a copy of distinct
/// texels); a CPU read at its address and a read at another address of the same pages then see
/// those texels in the console layout, and so does a view of the pages themselves.
#[test]
fn gpu_written_texels_reach_the_physical_pages_for_cpu_and_alias_reads() {
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no D3D12 device");
            return;
        }
        let dev = dev.unwrap();
        force_on();
        let (w, h, array, mips) = (128u64, 64u32, 2u16, 3u16);
        let desc = console_desc(w, h, array, mips, 28, 1, 0);
        let tex = xg::Texture::new(&desc).unwrap();
        let size = tex.size() as usize;
        let pages = size.div_ceil(PHYS_PAGE);
        let (at, alias, ids) = kernelx::test_alias_views(pages).expect("physical pages at two addresses");
        std::ptr::write_bytes(at as *mut u8, 0xEE, pages * PHYS_PAGE);
        assert_eq!(*(alias as *const u8), 0xEE, "one memory at both addresses");
        let native = texture(&dev, w, h, array, mips);
        super::register(native.as_raw(), &GuestImage { va: at as u64, desc, host: HostBacking::PRIVATE, initial: [0, 0] });
        assert!(tracks(native.as_raw() as usize));

        // what the GPU writes: subresource k's texel i is (k, i, i >> 8, 0x5A)
        let n = u32::from(mips) * u32::from(array);
        let nd = native.GetDesc();
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0u64;
        dev.GetCopyableFootprints(&nd, 0, n, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        let bdesc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut up: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).unwrap();
        let up = up.unwrap();
        let mut p = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).unwrap();
        let surface = tex.texels().unwrap();
        let mut want = vec![0xEEu8; size];
        for k in 0..n as usize {
            let (mip, slice) = (k % mips as usize, k / mips as usize);
            let l = surface.chain.levels[mip];
            let texels: Vec<u8> = (0..l.width * l.height).flat_map(|i| [k as u8, i as u8, (i >> 8) as u8, 0x5A]).collect();
            for y in 0..l.height {
                std::ptr::copy_nonoverlapping(texels[y * l.width * 4..].as_ptr(), (p as *mut u8).add(fps[k].Offset as usize + y * fps[k].Footprint.RowPitch as usize), l.width * 4);
            }
            surface.chain.tile_level(&mut want[slice * surface.chain.slice_bytes..][..surface.chain.slice_bytes], mip, slice, &texels);
        }
        up.Unmap(0, None);

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let transition = |before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&native),
                    Subresource: ALL,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        for k in 0..n {
            let to = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&native), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k } };
            let from = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fps[k as usize] } };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            super::note_copy_region(list.as_raw(), &to, 0, 0, 0, &from, std::ptr::null());
        }
        list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
        list.Close().unwrap();
        let raw = list.as_raw();
        super::super::title_textures::note_transitions(raw as usize, &[(native.as_raw() as usize, ALL, D3D12_RESOURCE_STATE_COPY_DEST), (native.as_raw() as usize, ALL, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        super::submitted(queue.as_raw(), &[raw]);
        super::super::title_textures::list_reset(raw as usize);
        let flushed = counted("images written back");

        // a CPU read at the texture's address: the guard writes it back first
        let mut seen = vec![0u8; size];
        std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
        assert_eq!(counted("images written back"), flushed + 1);
        assert!(seen == want, "CPU read at the texture's address: {} of {size} bytes differ", seen.iter().zip(&want).filter(|(a, b)| a != b).count());
        // the raw alias and the physical pages themselves
        std::ptr::copy_nonoverlapping(alias as *const u8, seen.as_mut_ptr(), size);
        assert!(seen == want, "read at the alias differs");
        let QueuePageSource::Physical(page) = kernelx::snapshot_queue_sources(&[at]).unwrap().remove(0) else { panic!("a physical page") };
        let view = kernelx::pin_physical_view_run(std::slice::from_ref(&page)).unwrap();
        let n0 = PHYS_PAGE.min(size);
        assert!(std::slice::from_raw_parts(view.base() as *const u8, n0) == &want[..n0], "the physical page differs");
        // bytes past the texture on its last page are left
        assert_eq!(*((at + size) as *const u8), 0xEE);

        // a submission that leaves it only read changes nothing: no guard, no write-back
        let alloc2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let idle: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc2, None).unwrap();
        idle.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(idle.cast().unwrap())]);
        super::submitted(queue.as_raw(), &[idle.as_raw()]);
        std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
        assert_eq!(counted("images written back"), flushed + 1);

        super::forget(native.as_raw() as usize);
        assert!(!tracks(native.as_raw() as usize));
        drop(view);
        kernelx::test_free_alias_views(at, alias, ids);
    }
}

/// Multisampled images and converted formats are refused and counted, never registered.
#[test]
fn refused_classes_are_counted_and_left_alone() {
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no D3D12 device");
            return;
        }
        let dev = dev.unwrap();
        force_on();
        let native = texture(&dev, 64, 64, 1, 1);
        let converted = counted("refused: converted format");
        super::register(native.as_raw(), &GuestImage { va: 0x7000_0000, desc: console_desc(64, 64, 1, 1, 86, 1, 1), host: HostBacking::PRIVATE, initial: [0, 0] });
        let layout = counted("refused: enhanced-barrier layout image");
        super::register(native.as_raw(), &GuestImage { va: 0x7000_0000, desc: console_desc(64, 64, 1, 1, 28, 1, 0), host: HostBacking::PRIVATE, initial: [1, 0] });
        assert!(!tracks(native.as_raw() as usize));
        assert!(counted("refused: converted format") > converted || counted("refused: format not identical to the native one") > 0);
        assert_eq!(counted("refused: enhanced-barrier layout image"), layout + 1);
    }
}

/// Records and submits a copy of distinct texels (seeded by seed) into every subresource of
/// native, which ends the list in COMMON as a title's decayed texture does; the texels as the
/// console layout holds them over a background of fill.
unsafe fn gpu_write(dev: &ID3D12Device, queue: &ID3D12CommandQueue, native: &ID3D12Resource, tex: &xg::Texture, n: u32, mips: u16, seed: u8, fill: u8) -> Vec<u8> {
    let nd = native.GetDesc();
    let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut total = 0u64;
    dev.GetCopyableFootprints(&nd, 0, n, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut total));
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
    let bdesc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut up: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).unwrap();
    let up = up.unwrap();
    let mut p = std::ptr::null_mut();
    up.Map(0, None, Some(&mut p)).unwrap();
    let surface = tex.texels().unwrap();
    let mut want = vec![fill; tex.size() as usize];
    for k in 0..n as usize {
        let (mip, slice) = (k % mips as usize, k / mips as usize);
        let l = surface.chain.levels[mip];
        let texels: Vec<u8> = (0..l.width * l.height).flat_map(|i| [k as u8 ^ seed, i as u8, (i >> 8) as u8, seed]).collect();
        for y in 0..l.height {
            std::ptr::copy_nonoverlapping(texels[y * l.width * 4..].as_ptr(), (p as *mut u8).add(fps[k].Offset as usize + y * fps[k].Footprint.RowPitch as usize), l.width * 4);
        }
        surface.chain.tile_level(&mut want[slice * surface.chain.slice_bytes..][..surface.chain.slice_bytes], mip, slice, &texels);
    }
    up.Unmap(0, None);
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
    let transition = |before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(native),
                Subresource: ALL,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
    for k in 0..n {
        let to = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(native), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k } };
        let from = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fps[k as usize] } };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        super::note_copy_region(list.as_raw(), &to, 0, 0, 0, &from, std::ptr::null());
    }
    list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
    list.Close().unwrap();
    let raw = list.as_raw();
    super::super::title_textures::note_transitions(raw as usize, &[(native.as_raw() as usize, ALL, D3D12_RESOURCE_STATE_COPY_DEST), (native.as_raw() as usize, ALL, D3D12_RESOURCE_STATE_COMMON)]);
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    super::submitted(queue.as_raw(), &[raw]);
    super::super::title_textures::list_reset(raw as usize);
    // the upload buffer, allocator and list stay alive until the write-back has waited on them
    std::mem::forget((up, alloc, list));
    want
}

/// What the title sees (play-85: the account picker never left): the GPU writes a texture that
/// ends its list in COMMON, the CPU reads it (written back) and then reuses the memory; a later
/// submission that never touches the texture must leave the CPU's bytes alone, and a later GPU
/// write must still reach the CPU.
#[test]
fn cpu_writes_survive_submissions_that_leave_a_written_back_image_alone() {
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no D3D12 device");
            return;
        }
        let dev = dev.unwrap();
        force_on();
        let (w, h, array, mips) = (128u64, 64u32, 2u16, 3u16);
        let desc = console_desc(w, h, array, mips, 28, 1, 0);
        let tex = xg::Texture::new(&desc).unwrap();
        let size = tex.size() as usize;
        let pages = size.div_ceil(PHYS_PAGE);
        let (at, alias, ids) = kernelx::test_alias_views(pages).expect("physical pages at two addresses");
        std::ptr::write_bytes(at as *mut u8, 0xEE, pages * PHYS_PAGE);
        let native = texture(&dev, w, h, array, mips);
        super::register(native.as_raw(), &GuestImage { va: at as u64, desc, host: HostBacking::PRIVATE, initial: [0, 0] });
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let n = u32::from(mips) * u32::from(array);
        let read = || {
            let mut seen = vec![0u8; size];
            std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
            seen
        };
        let differ = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();

        let want = gpu_write(&dev, &queue, &native, &tex, n, mips, 0x5A, 0xEE);
        assert!(read() == want, "the GPU's texels reach the CPU");

        // the CPU reuses the memory, then submits work that never touches the texture
        std::ptr::write_bytes(at as *mut u8, 0x11, size);
        let (guarded, flushed) = (counted("writes guarded"), counted("images written back"));
        for _ in 0..3 {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let idle: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            idle.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(idle.cast().unwrap())]);
            super::submitted(queue.as_raw(), &[idle.as_raw()]);
            let seen = read();
            assert!(seen.iter().all(|&b| b == 0x11), "the CPU's bytes after a submission that left the texture alone: {} of {size} bytes reverted", seen.iter().filter(|&&b| b != 0x11).count());
            std::mem::forget((alloc, idle));
        }
        assert!(*(alias as *const u8) == 0x11, "the alias shows the CPU's bytes");
        assert_eq!((counted("writes guarded"), counted("images written back")), (guarded, flushed), "lists that write nothing guard and write back nothing");

        // the GPU writes the same texels again: a whole-subresource copy lands over the CPU's bytes
        let same = gpu_write(&dev, &queue, &native, &tex, n, mips, 0x5A, 0x11);
        let seen = read();
        assert!(seen == same, "an identical GPU rewrite reaches the CPU: {} of {size} bytes differ", differ(&seen, &same));

        // a new GPU write still lands, over the CPU's bytes
        std::ptr::write_bytes(at as *mut u8, 0x11, size);
        let again = gpu_write(&dev, &queue, &native, &tex, n, mips, 0xA5, 0x11);
        let seen = read();
        assert!(seen == again, "a later GPU write reaches the CPU: {} of {size} bytes differ", differ(&seen, &again));

        super::forget(native.as_raw() as usize);
        kernelx::test_free_alias_views(at, alias, ids);
    }
}

unsafe fn texture_with(dev: &ID3D12Device, w: u64, h: u32, array: u16, mips: u16, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: w,
        Height: h,
        DepthOrArraySize: array,
        MipLevels: mips,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: flags,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap();
    r.unwrap()
}

/// A registered 128x64 image (two layers, three mips) over physical pages of its own, in the
/// given console state.
unsafe fn tracked_image(dev: &ID3D12Device, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> (ID3D12Resource, usize, usize, usize, Vec<usize>) {
    let desc = console_desc(128, 64, 2, 3, 28, 1, 0);
    let size = xg::Texture::new(&desc).unwrap().size() as usize;
    let pages = size.div_ceil(PHYS_PAGE);
    let (at, alias, ids) = kernelx::test_alias_views(pages).expect("physical pages at two addresses");
    std::ptr::write_bytes(at as *mut u8, 0x22, pages * PHYS_PAGE);
    let native = texture_with(dev, 128, 64, 2, 3, flags, state);
    super::register(native.as_raw(), &GuestImage { va: at as u64, desc, host: HostBacking::PRIVATE, initial: [0, state.0 as u32] });
    assert!(tracks(native.as_raw() as usize));
    (native, at, size, alias, ids)
}

unsafe fn submit_empty(dev: &ID3D12Device, queue: &ID3D12CommandQueue, note: impl FnOnce(*mut c_void)) {
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
    list.Close().unwrap();
    note(list.as_raw());
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    super::submitted(queue.as_raw(), &[list.as_raw()]);
    super::super::title_textures::list_reset(list.as_raw() as usize);
    std::mem::forget((alloc, list));
}

/// Which submissions guard an image: a render target bound and drawn with, a UAV image in a UAV
/// state with shader work; not a bind without a draw, a UAV image without shader work, or a list
/// that writes nothing while the image sits in a write state. A list with an untracked
/// attachment takes the state rule, and the write-back then keeps the CPU's bytes when the
/// texels did not change.
#[test]
fn only_lists_that_write_an_image_guard_it() {
    use super::super::real::image_views::{Event, Slice, View};
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            crate::gpu::recomp::runner::skip_test("no D3D12 device");
            return;
        }
        let dev = dev.unwrap();
        force_on();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let guarded = || counted("writes guarded");

        // a render target, left in RENDER_TARGET
        let (rt, at, size, alias, ids) = tracked_image(&dev, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET);
        let view = || View {
            resource: rt.clone(),
            slice: Slice { mip: 0, first: 0, count: 2, plane: 0, w_slices: None, format: DXGI_FORMAT_R8G8B8A8_UNORM, read_only_depth: false, read_only_stencil: false },
            generation: 0,
            mips: 3,
            layers: 2,
        };
        let g0 = guarded();
        submit_empty(&dev, &queue, |_| {});
        assert_eq!(guarded(), g0, "a list that writes nothing leaves a render target in RENDER_TARGET alone");
        submit_empty(&dev, &queue, |l| super::attachments(l as usize, Ok(Event::Bind { colors: vec![view()], depth: None })));
        assert_eq!(guarded(), g0, "binding without drawing writes nothing");
        submit_empty(&dev, &queue, |l| {
            super::attachments(l as usize, Ok(Event::Bind { colors: vec![view()], depth: None }));
            super::note_work(l);
        });
        assert_eq!(guarded(), g0 + 1, "a draw with it bound writes it");
        let mut seen = vec![0u8; size];
        std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
        super::forget(rt.as_raw() as usize);
        kernelx::test_free_alias_views(at, alias, ids);

        // a UAV image in UNORDERED_ACCESS
        let (uav, at, _, alias, ids) = tracked_image(&dev, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let g1 = guarded();
        submit_empty(&dev, &queue, |_| {});
        assert_eq!(guarded(), g1, "no shader work: the UAV image is not written");
        submit_empty(&dev, &queue, |l| super::note_work(l));
        assert_eq!(guarded(), g1 + 1, "shader work may write a UAV image in a UAV state");
        super::forget(uav.as_raw() as usize);
        kernelx::test_free_alias_views(at, alias, ids);

        // an untracked attachment: the state rule; nothing changed, so the CPU's bytes stay
        let (plain, at, size, alias, ids) = tracked_image(&dev, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
        let (g2, untracked) = (guarded(), counted("submissions with an untracked attachment: images in a write state taken as written"));
        submit_empty(&dev, &queue, |l| super::attachments(l as usize, Err("untracked or null attachment descriptor")));
        assert_eq!(guarded(), g2 + 1, "an untracked attachment takes the state rule");
        assert_eq!(counted("submissions with an untracked attachment: images in a write state taken as written"), untracked + 1);
        let mut seen = vec![0u8; size];
        std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
        std::ptr::write_bytes(at as *mut u8, 0x33, size);
        let unchanged = counted("write-backs with no texel changed since the last: the CPU's bytes kept");
        submit_empty(&dev, &queue, |l| super::attachments(l as usize, Err("untracked or null attachment descriptor")));
        std::ptr::copy_nonoverlapping(at as *const u8, seen.as_mut_ptr(), size);
        assert!(seen.iter().all(|&b| b == 0x33), "the state rule's write-back of unchanged texels keeps the CPU's bytes");
        assert_eq!(counted("write-backs with no texel changed since the last: the CPU's bytes kept"), unchanged + 1);
        super::forget(plain.as_raw() as usize);
        kernelx::test_free_alias_views(at, alias, ids);
    }
}
