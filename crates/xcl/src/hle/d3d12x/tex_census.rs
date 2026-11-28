//! Video memory XCL's textures hold, by the console format they come from (`XCL_TEX_CENSUS`).
//!
//! A run's video memory cannot be attributed today: `memdiag` says how much the device uses
//! (play-125: 6,902 MB local + 8,155 MB non-local, against a Series S budget of 8 GB), and how
//! much of the title's memory is imported (3,855 MB in 657 segments) and bound in the physical
//! directory (2,283 MB), but not which formats those bytes are. So a format that costs its
//! console size and one that costs twice it look the same, and a search for memory to save
//! cannot tell where to look. This counts every console texture XCL makes a PC resource for,
//! at creation, by (kind, console format, PC format), with the texels each holds on the console
//! and on the PC.
//!
//! The formats are the ones `device_x::pc_resource_desc_for` gives a PC format other than the
//! console's:
//! * R9G9B9E5_SHAREDEXP (67) render targets and UAV textures, which the PC renders to and
//!   writes in neither way: R16G16B16A16_FLOAT, 8 bytes a texel against the console's 4
//!   (`castable`, `XCL_RGB9E5_CASTABLE=1`: R11G11B10_FLOAT, 4, when the driver's relaxed format
//!   casting takes it).
//! * B5G6R5_UNORM (86) and B5G5R5A1_UNORM (115) render targets and UAV textures, which a PC
//!   driver refuses with unordered access: R8G8B8A8, 4 bytes a texel against 2.
//! * R10G10B10_7E3_A2_FLOAT / _6E4_A2_FLOAT (116, 117): R16G16B16A16_FLOAT, 8 bytes a texel
//!   against 4. `title_textures::register` refuses these as sampled copies (the title's bytes
//!   are not the resource's texels), so one is video memory that is never filled.
//! Everything else, block-compressed formats (169–182: 8 or 16 bytes a 4×4 block) included, is
//! the console's own size: a BC7 texture is 16 bytes a block on both sides, so a sampled BC7
//! texture costs its console bytes once in the title's memory and once in video memory and
//! nothing more.
//!
//! Counted are textures that were made, never counted down (XCL sees no release of a resource
//! it did not make); the line says "since start" for that reason. A sampled texture uploaded
//! again after a CPU write is not counted again (one entry per `title_textures` registration),
//! but a render target made again for the same address every frame counts every time — compare
//! the classes with each other, and compare the sampled copies with the bound pages, not the
//! total with `memdiag`'s live video-memory figure.
//!
//! The texel bytes exclude the tiling padding of the console's layout library and the driver's
//! allocation padding: a few per cent either way. Multisampled textures are counted at their
//! sample count. `XCL_TEX_CENSUS=0` turns the counting off (the line then does not appear).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Which route made the texture: a sampled copy kept in step with the title's memory
/// (`title_textures`), or a GPU-owned render target, depth buffer or UAV texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Sampled,
    Target,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Sampled => "sampled",
            Kind::Target => "targets",
        }
    }
}

/// The textures of one (kind, console format, PC format), in texel bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Pair {
    pub count: u64,
    /// the console's texels of them
    pub console: u64,
    /// the PC's texels of them (the resource XCL made)
    pub host: u64,
}

static ON: AtomicBool = AtomicBool::new(false);
static READY: OnceLock<()> = OnceLock::new();
static TEXTURES: AtomicU64 = AtomicU64::new(0);
static PAIRS: Mutex<BTreeMap<(Kind, u32, u32), Pair>> = Mutex::new(BTreeMap::new());

/// Whether the census counts (false only with `XCL_TEX_CENSUS=0`).
fn on() -> bool {
    READY.get_or_init(|| {
        let on = std::env::var("XCL_TEX_CENSUS").map_or(true, |v| v.trim() != "0");
        ON.store(on, Ordering::Relaxed);
        if on {
            crate::xlog_info!("XCL_TEX_CENSUS: video-memory textures are counted by console and PC format for memdiag (XCL_TEX_CENSUS=0: off)");
        }
    });
    ON.load(Ordering::Relaxed)
}

/// Bytes of one element (a pixel, or a 4×4 block for BCn) and pixels per element side, from the
/// console layout library's own table (`xg::element`). None: a format XCL has no layout for
/// (buffers, the console's D16_UNORM_S8_UINT family 118–120, unknown numbers).
pub(super) fn element(format: u32) -> Option<(u64, u64)> {
    let (log2, side) = crate::hle::xg::element(format)?;
    Some((1u64 << log2, side as u64))
}

fn buffer() -> u32 {
    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32
}

fn one_dim() -> u32 {
    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE1D.0 as u32
}

fn three_dim() -> u32 {
    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE3D.0 as u32
}

/// The texels of a D3D12_RESOURCE_DESC as XCL lays them out: every mip level (halved, rounded up
/// to whole elements), `DepthOrArraySize` slices or layers deep (a 3D texture's mips halve its
/// depth), at the format's element size and side.
pub(super) fn texel_bytes(desc: &[u8; 0x38]) -> Option<u64> {
    let dimension = u32::from_le_bytes(desc[0..4].try_into().unwrap());
    if dimension == buffer() {
        return None;
    }
    let (element_bytes, side) = element(u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap()))?;
    let width = u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap());
    // a 1D texture has no height (D3D12 reads none), and its depth is an array
    let height = if dimension == one_dim() { 1 } else { u32::from_le_bytes(desc[0x18..0x1c].try_into().unwrap()) as u64 };
    let slices = u16::from_le_bytes(desc[0x1c..0x1e].try_into().unwrap()) as u64;
    let mips = u16::from_le_bytes(desc[0x1e..0x20].try_into().unwrap()).max(1) as u64;
    let samples = u32::from_le_bytes(desc[0x24..0x28].try_into().unwrap()).max(1) as u64;
    if width == 0 || slices == 0 || height == 0 {
        return None;
    }
    let three_d = dimension == three_dim();
    let mut total = 0u64;
    for mip in 0..mips {
        let w = (width >> mip).max(1).div_ceil(side);
        let h = (height >> mip).max(1).div_ceil(side);
        let d = if three_d { (slices >> mip).max(1) } else { slices };
        total = total.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element_bytes)?)?;
    }
    Some(total.saturating_mul(samples))
}

/// Count one texture XCL made from the console's `console` desc: its PC resource is described by
/// `pc` (`device_x::pc_resource_desc` of the same desc). `console_bytes` is the console layout
/// library's own size when the caller has it (`xg::Texture::size`).
pub(super) fn note(console: &[u8; 0x38], pc: &[u8; 0x38], console_bytes: Option<u64>) {
    if !on() {
        return;
    }
    let flags = u32::from_le_bytes(console[0x30..0x34].try_into().unwrap());
    let kind = if super::vram::gpu_owned(flags) { Kind::Target } else { Kind::Sampled };
    let (console_format, pc_format) = (u32::from_le_bytes(console[0x20..0x24].try_into().unwrap()), u32::from_le_bytes(pc[0x20..0x24].try_into().unwrap()));
    let Some(host) = texel_bytes(pc) else { return };
    // the console's own layout library gives the console's size exactly (`xg::Texture::size`);
    // the callers that have it pass it, the rest fall back to the same texel arithmetic
    let console = console_bytes.filter(|&b| b != 0 && b != u64::MAX).unwrap_or_else(|| texel_bytes(console).unwrap_or(host));
    let mut pairs = PAIRS.lock().unwrap_or_else(|e| e.into_inner());
    let e = pairs.entry((kind, console_format, pc_format)).or_default();
    e.count += 1;
    e.console = e.console.saturating_add(console);
    e.host = e.host.saturating_add(host);
    drop(pairs);
    TEXTURES.fetch_add(1, Ordering::Relaxed);
}

/// Whether a format is one of the block-compressed ones (BC1–BC7): XCL keeps those at their
/// console size, 8 or 16 bytes a 4×4 block.
pub(super) fn block_compressed(format: u32) -> bool {
    matches!(format, 70..=84 | 94..=99)
}

fn mb(bytes: u64) -> u64 {
    bytes >> 20
}

/// The classes of `pairs`, by texel bytes: the sizes that could not shrink, the ones that are
/// larger than the console's, and by how much.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Totals {
    pub count: u64,
    pub console: u64,
    pub host: u64,
    /// of `host`, textures in a block-compressed format (a BC7 texel is 1 byte, both sides)
    pub compressed: u64,
    /// of `host` and `console`, textures whose PC format is not the console's (the widened
    /// classes above)
    pub widened: u64,
    pub widened_console: u64,
    pub sampled: Pair,
    pub targets: Pair,
}

pub(super) fn totals(pairs: &BTreeMap<(Kind, u32, u32), Pair>) -> Totals {
    let mut t = Totals::default();
    for (&(kind, console_format, pc_format), p) in pairs {
        t.count += p.count;
        t.console = t.console.saturating_add(p.console);
        t.host = t.host.saturating_add(p.host);
        if block_compressed(console_format) {
            t.compressed = t.compressed.saturating_add(p.host);
        }
        if console_format != pc_format {
            t.widened = t.widened.saturating_add(p.host);
            t.widened_console = t.widened_console.saturating_add(p.console);
        }
        let kind_total = match kind {
            Kind::Sampled => &mut t.sampled,
            Kind::Target => &mut t.targets,
        };
        kind_total.count += p.count;
        kind_total.console = kind_total.console.saturating_add(p.console);
        kind_total.host = kind_total.host.saturating_add(p.host);
    }
    t
}

/// The memdiag line: the totals by kind and class, then the formats that hold the most.
pub(super) fn summary(pairs: &BTreeMap<(Kind, u32, u32), Pair>) -> String {
    let t = totals(pairs);
    let mut top: Vec<(&(Kind, u32, u32), &Pair)> = pairs.iter().filter(|(_, p)| p.host != 0).collect();
    top.sort_by_key(|(_, p)| std::cmp::Reverse(p.host));
    let top: Vec<String> = top
        .iter()
        .take(6)
        .map(|((kind, console, pc), p)| {
            let pair = if console == pc { format!("{console}") } else { format!("{console}→{pc}") };
            format!("{} {} {} textures {} MB console/{} MB PC", kind.name(), pair, p.count, mb(p.console), mb(p.host))
        })
        .collect();
    let sampled = format!("{} textures {} MB", t.sampled.count, mb(t.sampled.host));
    let targets = format!("{} textures {} MB", t.targets.count, mb(t.targets.host));
    let ratio = if t.widened_console == 0 { "no widening".to_string() } else { format!("{:.1}x", t.widened as f64 / t.widened_console as f64) };
    let saving = t.widened.saturating_sub(t.widened_console);
    let top = if top.is_empty() { "none".to_string() } else { top.join(", ") };
    format!(
        "texture census since start: {} textures holding {} MB of video memory ({} MB of console texels); sampled {sampled} ({} MB of console texels), targets {targets} ({} MB); block-compressed {} MB, at their console size; widened formats {} MB of console texels are {} MB on the PC ({ratio}), {} MB that a console-format route would not spend; top: {top}",
        t.count,
        mb(t.host),
        mb(t.console),
        mb(t.sampled.console),
        mb(t.targets.console),
        mb(t.compressed),
        mb(t.widened_console),
        mb(t.widened),
        mb(saving),
    )
}

/// The line for `memdiag`'s periodic report (None: off, or nothing counted yet).
pub fn line() -> Option<String> {
    if !on() || TEXTURES.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let pairs = PAIRS.lock().unwrap_or_else(|e| e.into_inner());
    Some(summary(&pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXTURE2D: u32 = 3;
    const TEXTURE3D: u32 = 4;
    const BUFFER: u32 = 1;

    /// A D3D12_RESOURCE_DESC of a texture: dimension, width, height, depth-or-array, mips,
    /// format, samples.
    fn desc(dimension: u32, width: u64, height: u32, depth_or_array: u16, mips: u16, format: u32, samples: u32) -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&dimension.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
        d[0x18..0x1c].copy_from_slice(&height.to_le_bytes());
        d[0x1c..0x1e].copy_from_slice(&depth_or_array.to_le_bytes());
        d[0x1e..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
        d
    }

    fn count(pairs: &mut BTreeMap<(Kind, u32, u32), Pair>, kind: Kind, console: &[u8; 0x38], pc: &[u8; 0x38], n: u64) {
        let format = |d: &[u8; 0x38]| u32::from_le_bytes(d[0x20..0x24].try_into().unwrap());
        let e = pairs.entry((kind, format(console), format(pc))).or_default();
        e.count += n;
        e.console += n * texel_bytes(console).unwrap();
        e.host += n * texel_bytes(pc).unwrap();
    }

    #[test]
    fn a_block_compressed_copy_is_one_byte_a_texel_on_both_sides() {
        let bc7 = desc(TEXTURE2D, 1024, 1024, 1, 1, 98, 1);
        assert_eq!(texel_bytes(&bc7), Some(1 << 20), "16 B a 4×4 block");
        let bc1 = desc(TEXTURE2D, 1024, 1024, 1, 1, 71, 1);
        assert_eq!(texel_bytes(&bc1), Some(1 << 19), "8 B a 4×4 block");
        assert_eq!(texel_bytes(&desc(BUFFER, 1 << 20, 1, 1, 1, 0, 1)), None, "not a texture");
        assert_eq!(texel_bytes(&desc(TEXTURE2D, 256, 256, 1, 1, 118, 1)), None, "no layout table for 118");
    }

    #[test]
    fn a_shared_exponent_target_is_counted_at_twice_its_console_texels() {
        let console = desc(TEXTURE2D, 1280, 720, 1, 1, 67, 1);
        let pc = desc(TEXTURE2D, 1280, 720, 1, 1, 10, 1);
        assert_eq!(texel_bytes(&console), Some(1280 * 720 * 4));
        assert_eq!(texel_bytes(&pc), Some(1280 * 720 * 8));
        let mut pairs = BTreeMap::new();
        count(&mut pairs, Kind::Target, &console, &pc, 2);
        let t = totals(&pairs);
        assert_eq!((t.count, t.console, t.host), (2, 2 * 1280 * 720 * 4, 2 * 1280 * 720 * 8));
        assert_eq!((t.widened_console, t.widened), (2 * 1280 * 720 * 4, 2 * 1280 * 720 * 8));
        assert_eq!(t.compressed, 0);
        let line = summary(&pairs);
        assert!(line.contains("2.0x"), "the widening shows: {line}");
        assert!(line.contains("67→10"), "the formats show: {line}");
        assert!(line.contains("7 MB that a console-format route would not spend"), "the bytes it costs show: {line}");
    }

    #[test]
    fn mips_layers_and_a_volume_count_every_subresource() {
        let mips = desc(TEXTURE2D, 64, 64, 1, 7, 28, 1);
        assert_eq!(texel_bytes(&mips), Some((4096 + 1024 + 256 + 64 + 16 + 4 + 1) * 4), "the chain to 1×1");
        let layers = desc(TEXTURE2D, 64, 64, 2, 7, 28, 1);
        assert_eq!(texel_bytes(&layers), Some(2 * (4096 + 1024 + 256 + 64 + 16 + 4 + 1) * 4), "two slices");
        let volume = desc(TEXTURE3D, 64, 64, 64, 3, 28, 1);
        assert_eq!(texel_bytes(&volume), Some((64 * 64 * 64 + 32 * 32 * 32 + 16 * 16 * 16) * 4), "a 3D texture's mips halve its depth");
        let msaa = desc(TEXTURE2D, 16, 16, 1, 1, 28, 4);
        assert_eq!(texel_bytes(&msaa), Some(16 * 16 * 4 * 4), "at its sample count");
    }

    #[test]
    fn the_real_console_to_pc_mapping_is_the_two_times_the_census_reports() {
        let console = desc(TEXTURE2D, 1280, 720, 1, 1, 67, 1);
        let mut target = console;
        target[0x30..0x34].copy_from_slice(&5u32.to_le_bytes());
        let pc = crate::hle::d3d12x::device_x::pc_resource_desc_for(&target, false);
        assert_eq!(u32::from_le_bytes(pc[0x20..0x24].try_into().unwrap()), 10, "R9G9B9E5 without castable is R16G16B16A16_FLOAT");
        assert_eq!(texel_bytes(&pc), Some(2 * texel_bytes(&console).unwrap()));
        let mut small = desc(TEXTURE2D, 2560, 1440, 1, 1, 115, 1);
        small[0x30..0x34].copy_from_slice(&5u32.to_le_bytes());
        let pc = crate::hle::d3d12x::device_x::pc_resource_desc_for(&small, false);
        assert_eq!(u32::from_le_bytes(pc[0x20..0x24].try_into().unwrap()), 28, "B5G5R5A1 with unordered access is R8G8B8A8");
        assert_eq!(texel_bytes(&pc), Some(2 * texel_bytes(&small).unwrap()));
        let bc7 = desc(TEXTURE2D, 1024, 1024, 1, 1, 98, 1);
        let pc = crate::hle::d3d12x::device_x::pc_resource_desc(&bc7);
        assert_eq!(u32::from_le_bytes(pc[0x20..0x24].try_into().unwrap()), 98, "BC7 stays BC7");
        assert_eq!(texel_bytes(&pc), texel_bytes(&bc7), "at its console size");
    }

    #[test]
    fn the_line_separates_the_classes_the_audit_asks_about() {
        let mut pairs = BTreeMap::new();
        let bc7 = (desc(TEXTURE2D, 256, 256, 1, 1, 98, 1), 64 << 10);
        count(&mut pairs, Kind::Sampled, &bc7.0, &bc7.0, (812 << 20) / bc7.1);
        count(&mut pairs, Kind::Target, &desc(TEXTURE2D, 1280, 720, 1, 1, 67, 1), &desc(TEXTURE2D, 1280, 720, 1, 1, 10, 1), 7);
        count(&mut pairs, Kind::Target, &desc(TEXTURE2D, 2560, 1440, 1, 1, 86, 1), &desc(TEXTURE2D, 2560, 1440, 1, 1, 28, 1), 1);
        let t = totals(&pairs);
        assert_eq!(t.compressed, t.sampled.host, "the sampled copies are the block-compressed class");
        assert_eq!((t.widened_console, t.widened), (7 * 1280 * 720 * 4 + 2560 * 1440 * 2, 7 * 1280 * 720 * 8 + 2560 * 1440 * 4));
        assert_eq!(t.count, t.sampled.count + t.targets.count);
        let line = summary(&pairs);
        assert!(line.contains("block-compressed 812 MB"), "{line}");
        assert!(line.contains("2.0x"), "{line}");
        assert!(line.contains("31 MB that a console-format route would not spend"), "half of the 63 MB the two widenings hold: {line}");
        assert!(line.contains("67→10") && line.contains("86→28") && line.contains("98 "), "{line}");
    }
}
