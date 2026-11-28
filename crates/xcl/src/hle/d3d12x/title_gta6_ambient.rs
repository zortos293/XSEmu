//! GTA VI (TitleId 7952987A) title patch: a stand-in for the indirect light the title traces rays
//! for (`XCL_GTA6_AMBIENT`; `title_gta6` has the rules for title patches). Off unless the switch
//! holds a level: a run that does not set it records nothing more and keeps nothing more.
//!
//! With ray tracing off (`raytracing`) the title's rays find nothing. Its program that resolves the
//! diffuse light of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the sky's light
//! outdoors, nothing indoors (walls are black where no lamp or sun reaches) and, in blocks of the
//! picture's tiles, what the probes' memory held. Right after that program's dispatch the light is
//! raised to an estimate taken from the frame (bounded by a multiple of it with
//! `XCL_GTA6_AMBIENT_CAP`): the light the visible surfaces send out (the title keeps it for its own
//! screen traces, `CS_CacheScreenSpaceRadiance`,
//! the frame before) is averaged in 64x36 cells, and every point gets the mean of the cells weighted
//! by the share of its surroundings they cover, as a directional value (first-order harmonics at
//! 160x90 points, taken between them by depth). What is not on screen is assumed to send out what
//! is, with a slow mean of the last seconds so that turning into a dark corner is not black at
//! once; there is no occlusion but the shade of creases. This is an estimate for the eye, not the
//! title's lighting: rendering bugs are judged with it off.
//!
//! - `XCL_GTA6_AMBIENT`: unset, `0` or anything that is no level: off. A number: that share of the
//!   estimate (1: all of it; cool_fox's launcher runs 0.8). `constant:<level>`: a flat light of that
//!   level, shaded in creases, as beloooo's fork has it.
//! - `XCL_GTA6_AMBIENT_AIR`: the air of a room is lit by the same probes and holds the sky's light
//!   indoors too without rays (a brown veil at sunset). A number bounds what the air scatters by that
//!   multiple of the light around (cool_fox's launcher runs 1.5); unset or `0`: the air is left as
//!   the title has it. Nothing with the flat light.
//! - `XCL_GTA6_AMBIENT_SHADE=<strength>,<radius in metres>`: the shade of creases (1 and 0.6 unless
//!   set; a strength of `0` turns it off).
//! - `XCL_GTA6_AMBIENT_CAP=<multiple>`: the title's own light is cut to that multiple of the
//!   estimate (cool_fox runs 4). Unset or `0`: it is kept whole. The cut feeds on itself: what the
//!   title leaves is the light of the next frame's surfaces, so a dimmer estimate cuts more, and
//!   play-108 faded to black in 20 minutes with 4.
//! - `XCL_GTA6_AMBIENT_FLOOR=<share>`: the estimate is at least that share of the slow mean (0
//!   unless set: no floor).
//! - `XCL_GTA6_AMBIENT_FALL=<share>`: how fast the slow mean follows a darker frame, a frame's share
//!   (0.04 unless set, as fast as it follows a brighter one, as cool_fox has it; 0: it only rises).
//!   A floor on a mean that only rises feeds on itself the other way: what the raise adds is the
//!   light of the next frame's surfaces, the mean can only take it in, and play-110 went white a
//!   minute after the stand-in engaged with a floor of 0.5 and a fall of 0.
//!   `XCL_GTA6_AMBIENT_CAP=4` is cool_fox's light.
//! - `XCL_GTA6_AMBIENT_BOUND=<multiple>`: the estimate is at most that multiple of the mean light
//!   the title's own program left in the frame, before the raise (1 unless set; `0`: unbounded).
//!   The title's light follows its exposure, so the stand-in cannot outshine the frame it is put in.
//! - `XCL_GTA6_AMBIENT_FALLBACK=<share>`: the estimate above is only as good as the light it is
//!   taken from, and with rays off the probes can leave no light at all over most of the picture
//!   (play-126: zero over 92.5% of it), so the estimate is zero there and the raise adds nothing.
//!   While the estimate reads nothing in more than that share of its cells, or the light the title's
//!   own program left reads zero over more than that share of the picture, the estimate is raised,
//!   where it is darker, to the mean of the light the picture's lit part still sends out (its sky and
//!   its sunlit surfaces, damped as the cells are so a glint does not set it), flat, at the same
//!   `level` of it the estimate is taken at (0.8 of it), shaded in creases like the flat light, and
//!   not bounded by `XCL_GTA6_AMBIENT_BOUND`, which reads that same dead light. 0.5 unless set (more
//!   than half; `0` is today's behaviour, the estimate from the probe chain and nothing else; a share
//!   of 1, or more, never engages). One line per occasion is logged, with both shares that engaged
//!   it (they are read back from the GPU: see `copy_report`). A picture with no light anywhere reads
//!   zero over all of itself and leaves nothing to take the level from: it takes
//!   `XCL_GTA6_AMBIENT=constant:<level>` instead. General fix: inline ray tracing the driver can
//!   compile in bounded memory, then ray tracing on, and the probes carry the light themselves.
//!
//! None of these changes a translated shader, so none is in the translation cache's identity:
//! the passes here are XCL's own programs, compiled like its other helpers (`real::compile`).
//!
//! The patch follows three of the title's compute programs by the name in their container
//! (`role`) and takes their images from fixed places in their root tables (`LIGHT`, `DEPTH`,
//! `RADIANCE`, `AIR`): both are this title's, as the build cool_fox runs lays them out. Before anything is
//! recorded the views found there are checked (`view_fits`, `accept`). The places are those of
//! cool_fox's game build, and this build's root signatures can lay the tables out otherwise (in
//! play-105 the radiance's place had no unordered-access range), so when the view at a place is not
//! of the kind, format and size the pass needs, the program's root tables are searched for the one
//! view that is (`find_image`, `find_depth`; `XCL_GTA6_AMBIENT_SCAN=0`: the fixed places only), and
//! where it was found is logged once and kept. No view, or more than one where the pass writes, is
//! refused and logged with what was found. General fix: inline ray tracing the
//! driver can compile in bounded memory, then ray tracing on; the title's own probes then find
//! their light.
//!
//! The passes run on the list of the title's dispatch, right after it and only when it was
//! recorded (`bind::dispatch_flags`; a dispatch put off to the submit is recorded there through the
//! same function, so its stand-in follows it there). They bind XCL's own root signature, root
//! arguments and pipelines as compute state and leave them bound: every translated dispatch binds
//! its root signature, its two root views and its pipeline again (`bind::bind_views`), the title
//! names a graphics pipeline again before it draws, and the shared view heap stays the one bound,
//! so there is nothing to put back. No resource changes state: the title's dispatch has just used
//! the same views the same way (the image through its unordered-access view, the depth through its
//! shader view, both in compute), XCL keeps no record of the states the title's barriers leave, and
//! the two buffers of the patch never leave unordered access.
//!
//! This file is ported from cool_fox (the CGBJ re-upload; a fork of Marvisak/VIctory by cool_fox
//! and codec_d), commit f4b48ad766ddbec1a1867022cd42f7645b0dd55e:
//! - crates/victory-gpu/src/ambient.rs:1-530: the description above, the roles, the places of the
//!   views, the settings and their parsing, the HLSL of the six passes (`SOURCE`, copied; each
//!   entry is named `main` here and `mean` has its own part, because XCL's helper compiler takes one
//!   entry name), the buffers' sizes, the constants and the order of the passes (`record`);
//! - crates/victory-gpu/src/draw.rs:3159-3227: the hook after a dispatch and the check of format
//!   and size (`after_dispatch`, `accept`);
//! - crates/victory-gpu/src/pipeline.rs:66,118: the role of a program, told when it is made (`made`).
//! That a stand-in can be put in these images is beloooo's find (VIctory-Belobug-AMD 06031237,
//! 4f45ab88, cd9f78ce), as cool_fox's file says. Not for a public or shared build (`title_gta6`).
//!
//! SPDX-License-Identifier: GPL-3.0-or-later
//!
//! Copyright (C) the authors of cool_fox and VIctory (see the repositories named above).
//!
//! This program is free software: you can redistribute it and/or modify it under the terms of the
//! GNU General Public License as published by the Free Software Foundation, either version 3 of
//! the License, or (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
//! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
//! the GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License along with this program. If
//! not, see <https://www.gnu.org/licenses/>.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS,
    DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC,
};

use super::console_list::{Layout, Param, UavBinding};
use super::title_gta6::Patch;
use crate::gpu::desc::ImageType;

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program that writes that volume
    /// before it is summed up along the view).
    Air,
}

impl Role {
    /// The name in the program's container.
    const fn name(self) -> &'static str {
        match self {
            Role::Resolve => "CS_ScreenProbes_TemporalResolve",
            Role::Radiance => "CS_CacheScreenSpaceRadiance",
            Role::Air => "CS_AccumulateLocalLightInscatteringVolume",
        }
    }

    /// Where the image the stand-in writes after the program is.
    const fn image(self) -> Place {
        match self {
            Role::Resolve => LIGHT,
            Role::Radiance => RADIANCE,
            Role::Air => AIR,
        }
    }

    /// The format of that image's view: R32_UINT holding R9G9B9E5 for the diffuse light,
    /// R11G11B10_FLOAT for the other two.
    const fn format(self) -> i32 {
        match self {
            Role::Resolve => DXGI_FORMAT_R32_UINT.0,
            Role::Radiance | Role::Air => DXGI_FORMAT_R11G11B10_FLOAT.0,
        }
    }
}

/// The program's part in this, by its name in the container.
pub(super) fn role(container: &[u8]) -> Option<Role> {
    [Role::Resolve, Role::Radiance, Role::Air].into_iter().find(|r| container.windows(r.name().len()).any(|w| w == r.name().as_bytes()))
}

/// A view's place in a dispatch's root arguments: the root parameter (a descriptor table) and the
/// place in that table, in dwords (a console descriptor is 8 dwords).
type Place = (usize, u32);

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers) and reads the scene's
/// depth, where the other program writes the light of visible surfaces, and where the light the
/// air scatters is.
const LIGHT: Place = (1, 176);
const DEPTH: Place = (0, 464);
const RADIANCE: Place = (1, 16);
const AIR: Place = (1, 32);

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn number(text: &str) -> Option<f32> {
    text.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0)
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

/// Whether `XCL_GTA6_AMBIENT` holding `value` turns the patch on (`title_gta6::on`): unlike the other
/// title patches it is off unless its switch holds a level.
pub(super) fn set(value: Option<&str>) -> bool {
    value.is_some_and(|v| parse(v) != Set::Off)
}

/// The most the air scatters, as a multiple of the light around (0: what the title has).
fn parse_air(set: &str) -> f32 {
    number(set).unwrap_or(0.0)
}

/// A share of 0 or more that `set` names, `default` when it names none.
fn parse_share(set: &str, default: f32) -> f32 {
    set.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(default)
}

/// Strength and radius of the creases' shade: what `set` names of the two, 1 and 0.6 m otherwise.
fn parse_shade(set: &str) -> [f32; 2] {
    let mut shade = [1.0, 0.6];
    for (to, v) in shade.iter_mut().zip(set.split(',')) {
        if let Some(v) = v.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0) {
            *to = v;
        }
    }
    shade
}

struct Settings {
    light: Set,
    air: f32,
    shade: [f32; 2],
    /// The most of the title's own light that is kept, as a multiple of the estimate (0: all).
    cap: f32,
    /// The least the estimate is, as a share of the slow mean.
    lowest: f32,
    /// How fast the slow mean follows a darker frame.
    fall: f32,
    /// The most the estimate is, as a multiple of the mean of the title's own light (0: no bound).
    bound: f32,
    /// The share of the estimate's cells, or of the picture, that has to read no light before the
    /// estimate is raised to the level the picture's lit part sends out (0: no fallback).
    fallback: f32,
}

fn settings() -> &'static Settings {
    static SET: OnceLock<Settings> = OnceLock::new();
    SET.get_or_init(|| {
        let var = |name: &str| std::env::var(name).unwrap_or_default();
        Settings {
            light: parse(&var(Patch::Ambient.switch())),
            air: parse_air(&var("XCL_GTA6_AMBIENT_AIR")),
            shade: parse_shade(&var("XCL_GTA6_AMBIENT_SHADE")),
            cap: parse_share(&var("XCL_GTA6_AMBIENT_CAP"), 0.0),
            lowest: parse_share(&var("XCL_GTA6_AMBIENT_FLOOR"), 0.0),
            fall: parse_share(&var("XCL_GTA6_AMBIENT_FALL"), FOLLOW).min(1.0),
            bound: parse_share(&var("XCL_GTA6_AMBIENT_BOUND"), 1.0),
            fallback: parse_share(&var("XCL_GTA6_AMBIENT_FALLBACK"), FALLBACK),
        }
    })
}

/// Whether the stand-in is on for this run: the title is GTA VI, the switch holds a level, and
/// ray tracing is off (with ray tracing on the title's rays find their light, and the stand-in
/// would only cover it). False costs the callers one load each.
pub(super) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !super::title_gta6::on(Patch::Ambient) {
            return false;
        }
        let (what, switch) = (Patch::Ambient.what(), Patch::Ambient.switch());
        if !super::raytracing::off() {
            crate::xlog_info!("D3D12.X: title patch {what} does nothing in this run: ray tracing is on (XCL_RAYTRACING), and the title's own rays light its probes");
            return false;
        }
        let s = settings();
        let light = match s.light {
            Set::Estimate(level) => format!("{level} of the estimate taken from the frame"),
            Set::Constant(level) => format!("a flat light of {level}"),
            Set::Off => return false,
        };
        let air = if s.air > 0.0 && matches!(s.light, Set::Estimate(_)) { format!("the air scatters at most {} times the light around", s.air) } else { "the air is left as the title has it (XCL_GTA6_AMBIENT_AIR)".into() };
        let cap = if s.cap > 0.0 { format!("the title's own light cut to {} times the estimate", s.cap) } else { "the title's own light kept whole".into() };
        let fallback = if s.fallback > 0.0 { format!("the estimate raised to the level of the light the picture's lit part sends out when it reads nothing in more than {} of its cells, or the title's own light reads zero over more than {} of the picture", s.fallback, s.fallback) } else { "the estimate the probe chain's alone, never raised (XCL_GTA6_AMBIENT_FALLBACK)".into() };
        crate::xlog_info!(
            "D3D12.X: title patch {what} on ({switch}=0 turns it off): {light}, creases shaded with strength {} over {} m (XCL_GTA6_AMBIENT_SHADE), {air}, {cap} (XCL_GTA6_AMBIENT_CAP), the estimate at least {} of the slow mean (XCL_GTA6_AMBIENT_FLOOR), which follows a darker frame at {} a frame (XCL_GTA6_AMBIENT_FALL), and at most {} times the mean of the title's own light (XCL_GTA6_AMBIENT_BOUND), {fallback} (XCL_GTA6_AMBIENT_FALLBACK); it engages at the first dispatch of {}",
            s.shade[0],
            s.shade[1],
            s.lowest,
            s.fall,
            s.bound,
            Role::Resolve.name()
        );
        true
    })
}

/// The title's compute pipelines that have a role, by the pipeline the title holds.
static ROLES: Mutex<Option<HashMap<usize, Role>>> = Mutex::new(None);
/// A pipeline with a role has been made: until then a dispatch asks nothing more.
static ANY: AtomicBool = AtomicBool::new(false);

/// The title made compute pipeline `pso` from the shader `container` (only asked while the stand-in
/// is `wanted`): its role is kept, and that of a pipeline that was at this address before is not.
pub(super) fn made(pso: usize, container: &[u8]) {
    let role = role(container);
    if role.is_none() && !ANY.load(Ordering::Acquire) {
        return;
    }
    let mut roles = ROLES.lock().unwrap();
    let roles = roles.get_or_insert_with(HashMap::new);
    match role {
        Some(role) => {
            roles.insert(pso, role);
            ANY.store(true, Ordering::Release);
            // (a run's log then tells a program the title never made from one never dispatched)
            static SEEN: [AtomicUsize; ROLES_N] = [const { AtomicUsize::new(0) }; ROLES_N];
            let n = SEEN[role as usize].fetch_add(1, Ordering::Relaxed) + 1;
            if n == 1 || n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: title patch {} follows the dispatches of pipeline {pso:#x}, the title's {} (#{n})", Patch::Ambient.what(), role.name());
            }
        }
        None => {
            roles.remove(&pso);
        }
    }
}

/// The title released pipeline `pso` for good.
pub(super) fn gone(pso: usize) {
    if !ANY.load(Ordering::Acquire) {
        return;
    }
    if let Some(roles) = ROLES.lock().unwrap().as_mut() {
        roles.remove(&pso);
    }
}

fn role_of(pso: usize) -> Option<Role> {
    if !ANY.load(Ordering::Acquire) {
        return None;
    }
    ROLES.lock().unwrap().as_ref()?.get(&pso).copied()
}

/// The console address of the descriptor at `place` of the root arguments `args` laid out by
/// `layout`, or why there is none.
fn descriptor_at(layout: &Layout, args: &[u8], (param, dwords): Place) -> Result<u32, String> {
    let Some(&Param::Table { at }) = layout.params.get(param) else {
        return Err(format!("root parameter {param} is not a descriptor table"));
    };
    let at = at as usize;
    let table = args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).filter(|&t| t != 0).ok_or_else(|| format!("root parameter {param} has no table"))?;
    dwords.checked_mul(4).and_then(|bytes| table.checked_add(bytes)).ok_or_else(|| format!("table {table:#x} of root parameter {param} ends before dword {dwords}"))
}

/// Whether the root signature has an unordered-access range over `place`: the title's program
/// writes through the view there, so the PC view beside the shader view is one of that image.
fn declares_uav(layout: &Layout, (param, dwords): Place) -> bool {
    let slot = dwords * 4 / super::console_heap::SLOT;
    layout.uavs.iter().any(|u| matches!(*u, UavBinding::Table { param: p, first, count, .. } if p == param && slot >= first && slot - first < count))
}

fn place_text((param, dwords): Place) -> String {
    format!("root parameter {param}, dword {dwords}")
}

/// Whether the view the console descriptor `view` describes is of the kind the pass after `role`
/// takes at that place: the image it writes (`image`: a volume for the air, a plain 2D image
/// otherwise, in the role's format), or the scene's depth (a plain 2D image). Asked before the PC
/// resource behind the view is looked at.
fn view_fits(role: Role, image: bool, view: &crate::gpu::desc::Image) -> Result<(), String> {
    let kind = if image && role == Role::Air { ImageType::D3 } else { ImageType::D2 };
    if view.ty != kind {
        return Err(format!("a {:?} view, not a {kind:?} one", view.ty));
    }
    let format = crate::gpu::desc::dxgi_format(view.format).map_or(0, |(f, _)| f.0);
    if image && format != role.format() {
        return Err(format!("a view of format {format} (console format {}), not {}", view.format, role.format()));
    }
    Ok(())
}

/// A view of the title's and the PC resource behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Viewed {
    /// the shader view's place in the shared view heap; the unordered-access view is the next
    index: u32,
    /// the size the view shows (of its first mip)
    view_size: [u32; 2],
    /// the PC resource: its format, size, depth or slices, and D3D12_RESOURCE_DIMENSION
    format: i32,
    width: u64,
    height: u32,
    slices: u16,
    dimension: i32,
}

/// Whether the pass after `role` may write `image`, reading `depth` (the image again for the roles
/// that read none): the size of the image and its slices, or what is not as the pass needs it. As
/// cool_fox checks it: the resource in the role's format (a typeless one of its family counts, the
/// view names the format), a volume for the air and a 2D texture otherwise, the depth as large as
/// the light; and the view of the whole first mip, which the pass covers.
fn accept(role: Role, image: &Viewed, depth: &Viewed) -> Result<([u32; 2], u32), String> {
    let volume = image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0;
    let flat = image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D.0;
    if volume != (role == Role::Air) || !(volume || flat) {
        return Err(format!("the image is a resource of dimension {}, not a {}", image.dimension, if role == Role::Air { "volume" } else { "2D texture" }));
    }
    let typeless = if role.format() == DXGI_FORMAT_R32_UINT.0 { DXGI_FORMAT_R32_TYPELESS.0 } else { role.format() };
    if image.format != role.format() && image.format != typeless {
        return Err(format!("the image is a resource of format {}, not {}", image.format, role.format()));
    }
    let size = [image.width as u32, image.height];
    if image.width == 0 || image.width > 16384 || image.height == 0 || image.view_size != size {
        return Err(format!("the image is {}x{} and its view shows {}x{}", image.width, image.height, image.view_size[0], image.view_size[1]));
    }
    if role == Role::Resolve && (depth.dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 || (depth.width, depth.height) != (image.width, image.height)) {
        return Err(format!("the depth is {}x{} (dimension {}, format {}) and the light {}x{}", depth.width, depth.height, depth.dimension, depth.format, image.width, image.height));
    }
    Ok((size, u32::from(image.slices)))
}

/// The view at `place` of the root arguments and its PC resource, when it is of the kind the pass
/// after `role` takes there; or what is there instead.
unsafe fn viewed(layout: &Layout, args: &[u8], role: Role, place: Place, image: bool) -> Result<Viewed, String> {
    let fail = |why: String| format!("{} ({}): {why}", if image { "the image" } else { "the depth" }, place_text(place));
    let at = descriptor_at(layout, args, place).map_err(fail)?;
    let words = super::console_heap::words_at(at).ok_or_else(|| fail(format!("{at:#x} is in no console view heap")))?;
    // (the pass writes the image through the view beside its shader view: one the title's program
    // writes through, or one XCL made for the view because its resource takes unordered access)
    if image && !declares_uav(layout, place) && super::console_heap::native_uav_at(at, words[0]).is_none() {
        return Err(fail("the root signature has no unordered-access range there, and the view there has no unordered-access view".into()));
    }
    let view = crate::gpu::desc::image(&words).ok_or_else(|| fail(format!("{at:#x} holds no image view")))?;
    view_fits(role, image, &view).map_err(fail)?;
    let resource = super::console_heap::image_resource_at(at, words[0]).ok_or_else(|| fail(format!("no PC resource is known behind view {} at {at:#x}", words[0])))?;
    // (the view may outlive its resource: one that is gone no longer has XCL's vtable)
    let raw = resource as *mut c_void;
    if !super::real::is_adopted(raw) {
        return Err(fail(format!("the resource {resource:#x} behind view {} is not a live resource of the title's", words[0])));
    }
    let d = ID3D12Resource::from_raw_borrowed(&raw).ok_or_else(|| fail("no PC resource".into()))?.GetDesc();
    Ok(Viewed { index: words[0], view_size: [view.width, view.height], format: d.Format.0, width: d.Width, height: d.Height, slices: d.DepthOrArraySize, dimension: d.Dimension.0 })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=16, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define OWN_AT (2 * CELLS + 1)
#define SKY_AT (2 * CELLS + 2)
#define DEAD_AT (2 * CELLS + 3)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; float lowest;
    float fall; float bound; float fallback; float unused2;
};
Texture2D<float> depth : register(t0);
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are, at OWN_AT the mean the title's own program
// left, at SKY_AT the level and share of the light its lit part sends out,
// and at DEAD_AT the share the estimate reads nothing in.
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean and a direction.
RWStructuredBuffer<float4> points : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}

#ifdef OWN
RWTexture2D<uint> light : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    float sum = 0;
    float3 lit_sum = 0;
    uint lit = 0;
    for (uint y = 0; y < CY; y++) for (uint x = 0; x < CX; x++) {
        float3 c = unpack(light[uint2((float2(x, y) + 0.5) * float2(width, height) / float2(CX, CY))]);
        float l = dot(c, LUMA);
        sum += l;
        // The lit part, damped as the cells are: a lamp or a glint counts, but not for everything.
        lit_sum += c / max(1.0, l / 20.0);
        lit += (l > 0.0) ? 1u : 0u;
    }
    cells[OWN_AT] = float4(sum / CELLS, 0.0, 0.0, 1.0);
    cells[SKY_AT] = float4(lit_sum / max(lit, 1u), float(lit) / CELLS);
}
#endif

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        uint3 bits = asuint(radiance[p]);
        float3 c = any((bits & 0x7f800000u) == 0x7f800000u) ? 0.0 : max(asfloat(bits), 0.0);
        // Lamps and glints count, but not for everything.
        sum += c / max(1.0, dot(c, LUMA) / 20.0);
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
#endif

#ifdef MEAN
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    float3 sum = 0;
    uint dead = 0;
    for (uint i = 0; i < CELLS; i++) {
        float3 c = cells[i].rgb;
        sum += c;
        dead += (dot(c, LUMA) > 0.0) ? 0u : 1u;
    }
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    float rate = dot(now, LUMA) < dot(was.rgb, LUMA) ? fall : follow;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, rate) : now, 1.0);
    // The share of the cells the estimate reads nothing in: with rays off, most of them.
    cells[DEAD_AT] = float4(float(dead) / CELLS, 0.0, 0.0, 1.0);
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    cells[CELLS + 1 + at.y * CX + at.x] = float4(p, p.z * p.z);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    float3 low = max(light, lowest * cells[CELLS].rgb);
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, low.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, low.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, low.b);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is).
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    // How open it is (as in beloooo's cd9f78ce).
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(round(o)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    // Between the four points around, those at its own depth first; which
    // four is dithered a little.
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * points[at0];
        sg += w * points[at0 + 1];
        sb += w * points[at0 + 2];
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = max(found, lowest * cells[CELLS].rgb);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level * open;
    if (bound > 0.0) {
        found *= min(1.0, bound * cells[OWN_AT].x / max(dot(found, LUMA), 1e-6));
    }
    // The probes are dead over most of the picture (or the light they resolved with them): the
    // estimate is nothing (and the bound above cut it to that nothing), so the picture takes the
    // level of the light its own lit part still sends out, flat, at the share of it the estimate is
    // taken at. The bound is not asked again: the mean it reads is that same dead light.
    if (fallback > 0.0 && (cells[DEAD_AT].x > fallback || cells[SKY_AT].w < 1.0 - fallback)) found = max(found, cells[SKY_AT].rgb * (level * open));
    float3 was = unpack(light[p]);
    // What the program left counts as far as it can be true: the sky's light
    // outdoors, not the blocks of other memory.
    if (most > 0.0) {
        float top = max(dot(found, LUMA) * most, 1e-6);
        float has = dot(was, LUMA);
        was *= min(1.0, top / max(has, 1e-6));
    }
    light[p] = pack(max(was, found));
}
#endif
"#;

/// The six passes, by the part of `SOURCE` each is.
const PARTS: [&str; 7] = ["KEEP", "MEAN", "PLACE", "GATHER", "RAISE", "AIR", "OWN"];

/// The source of one pass: `SOURCE` with that part's entry.
fn source(part: &str) -> String {
    format!("#define {part}\n{SOURCE}")
}

/// Cells across and down, points across and down (`SOURCE`'s CX, CY, PX, PY), and the bytes of the
/// cells' buffer (the light of each, the slow mean, where each is, the mean light the title's own
/// program left, the level and share of the light the picture's own lit part sends out, and the
/// share the estimate reads nothing in) and of the points' (three float4 a point).
const CELLS: [u32; 2] = [64, 36];
const POINTS: [u32; 2] = [160, 90];
const CELL_BYTES: u64 = (2 * (CELLS[0] * CELLS[1]) as u64 + 4) * 16;
const POINT_BYTES: u64 = (POINTS[0] * POINTS[1]) as u64 * 48;
/// Where the cells' buffer holds that level and share (`SKY_AT`), from which the log copies out the
/// whole tail of the buffer: `SKY_AT` and `DEAD_AT` are the last two slots.
const SKY_BYTES: u64 = (2 * (CELLS[0] * CELLS[1]) as u64 + 2) * 16;

/// The distance of the near plane in metres and the tangent of half the view's height, which the
/// passes take the place of a texel from; and how fast the slow mean follows a brighter frame (a
/// frame's share).
const NEAR: f32 = 0.1;
const TAN_Y: f32 = 0.45;
const FOLLOW: f32 = 0.04;

/// The share of the estimate's cells, or of the picture, that `XCL_GTA6_AMBIENT_FALLBACK` waits
/// for unless it names one: more than half of either reading no light at all.
const FALLBACK: f32 = 0.5;

/// XCL's objects for the passes, raw (never released): the root signature, the pipelines in the
/// order of `PARTS`, and the two buffers (the cells' own resource too, for the log's copy out).
struct Passes {
    rs: usize,
    pipelines: [usize; 7],
    cells: u64,
    cells_res: usize,
    points: u64,
}

const KEEP: usize = 0;
const MEAN: usize = 1;
const PLACE: usize = 2;
const GATHER: usize = 3;
const RAISE: usize = 4;
const CLEAR_AIR: usize = 5;
const OWN: usize = 6;

/// The passes, made at the first dispatch that takes them; None when they could not be (logged).
fn passes() -> Option<&'static Passes> {
    static P: OnceLock<Option<Passes>> = OnceLock::new();
    P.get_or_init(|| unsafe {
        let made = (|| -> Result<Passes, String> {
            let raw = super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw).ok_or("no device")?;
            let code: Vec<Vec<u8>> = PARTS.iter().map(|part| super::real::compile(&source(part), "cs_5_1")).collect();
            if let Some(k) = code.iter().position(Vec::is_empty) {
                return Err(format!("the HLSL of its {} pass did not compile", PARTS[k]));
            }
            super::title_textures::own_work(|| {
                let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &code[0]).map_err(|e| format!("root signature: {e}"))?;
                let mut pipelines = [0usize; 7];
                for (k, code) in code.iter().enumerate() {
                    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&rs),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() },
                        ..Default::default()
                    };
                    let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(|e| format!("the pipeline of its {} pass: {e}", PARTS[k]))?;
                    if super::gpu_timing::enabled() {
                        super::dispatch_timing::note(pso.as_raw() as usize, code);
                    }
                    pipelines[k] = pso.into_raw() as usize;
                }
                // (a new committed buffer is zero: the slow mean starts unset, `was.a` 0)
                let buffer = |bytes: u64| -> Result<(u64, usize), String> {
                    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                    let desc = D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: bytes,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    };
                    let mut r: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut r).map_err(|e| format!("a buffer of {bytes} bytes: {e}"))?;
                    let r = r.ok_or("a buffer")?;
                    let va = r.GetGPUVirtualAddress();
                    Ok((va, r.into_raw() as usize))
                };
                let (cells, cells_res) = buffer(CELL_BYTES)?;
                Ok(Passes { rs: rs.into_raw() as usize, pipelines, cells, cells_res, points: buffer(POINT_BYTES)?.0 })
            })
        })();
        match made {
            Ok(p) => Some(p),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: title patch {} does nothing in this run, its passes could not be made ({e}): the title's light is left as it is", Patch::Ambient.what());
                None
            }
        }
    })
    .as_ref()
}

/// Whether the light of a frame has been kept: the estimate has something to begin with.
static KEPT: AtomicBool = AtomicBool::new(false);
/// Whether the points have been gathered once. The air's bound reads them, and against points
/// never written (the resolve's views refused, say) it would take the far air's light away, which
/// cool_fox, where the resolve is always followed, does not have to ask.
static GATHERED: AtomicBool = AtomicBool::new(false);

/// The constants of the passes (`SOURCE`'s `Set`) for an image of `size`.
fn constants(level: f32, constant: f32, s: &Settings, size: [u32; 2]) -> [u32; 16] {
    let f = f32::to_bits;
    [f(level), f(constant), f(s.shade[0]), f(s.shade[1]), f(NEAR), f(TAN_Y), size[0], size[1], f(FOLLOW), f(s.cap), f(s.air), f(s.lowest), f(s.fall), f(s.bound), f(s.fallback), 0]
}

/// The passes after `role` and the groups each is dispatched with (a pipeline of `PARTS` each), for
/// an image of `size` with `slices`: the cells and points in the thread groups of their programs, the
/// image in groups of 8x8 (the air's volume 8x8x2). The flat light needs no cells or points.
fn schedule(role: Role, flat: bool, size: [u32; 2], slices: u32) -> Vec<(usize, [u32; 3])> {
    let image = |z: u32| [size[0].div_ceil(8), size[1].div_ceil(8), z];
    match role {
        Role::Radiance => vec![(KEEP, [CELLS[0] / 8, CELLS[1] / 4, 1]), (MEAN, [1, 1, 1])],
        Role::Air => vec![(CLEAR_AIR, image(slices.div_ceil(2)))],
        Role::Resolve if flat => vec![(RAISE, image(1))],
        Role::Resolve => vec![(PLACE, [CELLS[0] / 8, CELLS[1] / 4, 1]), (GATHER, [POINTS[0] / 8, POINTS[1] / 2, 1]), (OWN, [1, 1, 1]), (RAISE, image(1))],
    }
}

/// Records `passes` on `list`, with `image` and `depth` as the handles of the title's views in the shared
/// view heap. A barrier for all unordered access goes before the first and after each: a pass
/// reads what the title's program and the pass before it wrote. With `report`, the light's own
/// report is copied out for the log after the last pass (`copy_report`).
unsafe fn record(list: *mut c_void, p: &Passes, words: &[u32; 16], image: u64, depth: u64, passes: &[(usize, [u32; 3])], report: bool) -> bool {
    let (rs, Some(l)) = (p.rs as *mut c_void, ID3D12GraphicsCommandList::from_raw_borrowed(&list)) else { return false };
    let Some(rs) = ID3D12RootSignature::from_raw_borrowed(&rs) else { return false };
    let barrier = || [D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }];
    super::list_x::ensure_view_heap(list);
    l.ResourceBarrier(&barrier());
    l.SetComputeRootSignature(rs);
    l.SetComputeRoot32BitConstants(0, words.len() as u32, words.as_ptr() as *const c_void, 0);
    l.SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: depth });
    l.SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: image });
    l.SetComputeRootUnorderedAccessView(3, p.cells);
    l.SetComputeRootUnorderedAccessView(4, p.points);
    for &(pass, [x, y, z]) in passes {
        let pso = p.pipelines[pass] as *mut c_void;
        let Some(pso) = ID3D12PipelineState::from_raw_borrowed(&pso) else { return false };
        l.SetPipelineState(pso);
        l.Dispatch(x, y, z);
        l.ResourceBarrier(&barrier());
    }
    if report {
        copy_report(list, l, p);
    }
    true
}

/// The light's own report (`SKY_AT` and `DEAD_AT`: the level of the picture's lit part, the share
/// that is lit, and the share the estimate reads nothing in) in a readback buffer the CPU reads for
/// the log, and the sequence number of the copy that carries it.
struct Readback {
    /// the readback resource itself, for the copy
    readback: usize,
    /// its GPU address, where the sequence number goes
    va: u64,
    /// its CPU address, mapped for good
    cpu: usize,
    seq: AtomicU32,
}

/// The report's readback buffer, made at the first fallback the stand-in records; `None` when the
/// device would not make one (the log then says nothing about the share).
fn readback() -> Option<&'static Readback> {
    static BUFFER: OnceLock<Option<Readback>> = OnceLock::new();
    BUFFER.get_or_init(|| unsafe { super::markers::readback_buffer(64) }.map(|(readback, va, cpu)| Readback { readback, va, cpu, seq: AtomicU32::new(0) })).as_ref()
}

/// Copies `SKY_AT` and `DEAD_AT` of the cells' buffer (`p`) into the report's readback at 4, after
/// the passes that wrote them, and then writes the copy's sequence number at its start. A list's
/// commands run in order on their queue, so a CPU that reads that sequence reads a copy the GPU has
/// finished: the number at 0 is what `note_fallback` waits for, never a fence.
unsafe fn copy_report(list: *mut c_void, l: &ID3D12GraphicsCommandList, p: &Passes) {
    let Some(r) = readback() else { return };
    let (dst_raw, src_raw) = (r.readback as *mut c_void, p.cells_res as *mut c_void);
    let (Some(dst), Some(src), Some(l2)) = (
        ID3D12Resource::from_raw_borrowed(&dst_raw),
        ID3D12Resource::from_raw_borrowed(&src_raw),
        ID3D12GraphicsCommandList2::from_raw_borrowed(&list),
    ) else {
        return;
    };
    l.ResourceBarrier(&[super::indirect::transition(p.cells_res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    l.CopyBufferRegion(dst, 4, src, SKY_BYTES, 32);
    l.ResourceBarrier(&[super::indirect::transition(p.cells_res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    let seq = r.seq.fetch_add(1, Ordering::Relaxed) + 1;
    let params = [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: r.va, Value: seq }];
    l2.WriteBufferImmediate(1, params.as_ptr(), None);
}

/// The shares the fallback waits for, from the report the GPU last finished writing: logged the
/// first time it engages and again each time the light comes back to it, one line an occasion, with
/// the share that engaged it. The report lags the frame it was measured in by a frame or two (it is
/// read when its copy has landed), and a pair read while its copy is in flight can name another
/// frame's share: this is a log, it decides nothing.
fn note_fallback(s: &Settings) {
    static SEEN: AtomicU32 = AtomicU32::new(0);
    static ENGAGED: AtomicBool = AtomicBool::new(false);
    let Some(r) = readback() else { return };
    // (the sequence is written after the copy it follows: a different one is a copy that landed)
    let seq = unsafe { (r.cpu as *const u32).read_volatile() };
    if seq == 0 || seq <= SEEN.load(Ordering::Relaxed) {
        return;
    }
    SEEN.store(seq, Ordering::Relaxed);
    // (4 bytes of sequence, then `SKY_AT` and `DEAD_AT`: the lit part's level and share, then the
    // share the estimate reads nothing in)
    let slot = unsafe { ((r.cpu + 4) as *const [f32; 8]).read_volatile() };
    let dead = slot[4];
    let engaged = dead > s.fallback || slot[3] < 1.0 - s.fallback;
    let was = ENGAGED.swap(engaged, Ordering::Relaxed);
    if engaged && !was {
        let (what, switch, frame) = (Patch::Ambient.what(), Patch::Ambient.switch(), super::present::counters().0);
        crate::xlog_info!(
            "D3D12.X: title patch {what} ({switch}=0 turns it off) falls back at frame {frame}: the estimate reads nothing in {:.1}% of its cells and the title's own light is zero over {:.1}% of the picture (XCL_GTA6_AMBIENT_FALLBACK={}), so the estimate is the level of the light its lit part sends out, {:.3} {:.3} {:.3}, at the share of it the stand-in adds",
            dead * 100.0,
            (1.0 - slot[3]) * 100.0,
            s.fallback,
            slot[0],
            slot[1],
            slot[2]
        );
    }
}

/// A dispatch of the title's pipeline `title_pso` was recorded on `list` just now: when the pipeline has
/// a role, the stand-in's passes for it follow on `list`, or the reason they do not is logged.
/// Whether the root tables are searched for a view that is not at its fixed place
/// (`XCL_GTA6_AMBIENT_SCAN`, on unless `0`).
fn scan() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GTA6_AMBIENT_SCAN").map_or(true, |v| v.trim() != "0"))
}

/// The most slots one search reads.
const SCAN_SLOTS: usize = 4096;
/// Frames between two searches that found nothing.
const SCAN_EVERY: u64 = 60;

/// Every slot of the unordered-access ranges of the root signature's tables, in order: where a
/// program's images are looked for.
fn uav_places(layout: &Layout) -> Vec<Place> {
    let dwords = super::console_heap::SLOT / 4;
    layout
        .uavs
        .iter()
        .filter_map(|u| match *u {
            UavBinding::Table { param, first, count, .. } => Some((param, first, count)),
            UavBinding::Root { .. } => None,
        })
        .flat_map(|(param, first, count)| (first..first.saturating_add(count)).map(move |slot| (param, slot.saturating_mul(dwords))))
        .take(SCAN_SLOTS)
        .collect()
}

/// Every slot of the root signature's descriptor tables (of a table whose end is not known, the
/// first 1024): where the depth is looked for.
fn table_places(layout: &Layout) -> Vec<Place> {
    let dwords = super::console_heap::SLOT / 4;
    layout
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, Param::Table { .. }))
        .flat_map(|(param, _)| {
            let end = layout.tables.get(param).copied().unwrap_or(0);
            (0..if end == u32::MAX { 1024 } else { end }).map(move |slot| (param, slot.saturating_mul(dwords)))
        })
        .take(SCAN_SLOTS)
        .collect()
}

/// Whether a resource of `format` holds a depth: a depth format, or a typeless format a depth
/// buffer is made in.
fn depth_like(format: i32) -> bool {
    [DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM]
        .iter()
        .any(|f| f.0 == format)
}

fn places_text(found: &[(Place, Viewed)]) -> String {
    found.iter().map(|(p, v)| format!("{} ({}x{}, format {}, view {})", place_text(*p), v.width, v.height, v.format, v.index)).collect::<Vec<_>>().join(", ")
}

/// The image of the views a search found that are of the kind, format and size the pass after
/// `role` needs: the one there is; of several, the largest for the light of visible surfaces (the
/// pass only reads it), and none for the images a pass writes (a guess could write the wrong one).
fn choose(role: Role, found: &[(Place, Viewed)]) -> Result<(Place, Viewed), String> {
    match found {
        [] => Err("no view there is of that kind, format and size".into()),
        [one] => Ok(*one),
        _ if role == Role::Radiance => Ok(*found.iter().rev().max_by_key(|(_, v)| v.width * u64::from(v.height)).unwrap()),
        _ => Err(format!("{} views there are of that kind, format and size, and the pass would write one: {}", found.len(), places_text(found))),
    }
}

/// The depth of the views a search found that are as large as the light: the one held in a depth
/// format, or else the one 32-bit float image.
fn choose_depth(found: &[(Place, Viewed)]) -> Result<(Place, Viewed), String> {
    let depth: Vec<_> = found.iter().copied().filter(|(_, v)| depth_like(v.format)).collect();
    let floats: Vec<_> = found.iter().copied().filter(|(_, v)| v.format == DXGI_FORMAT_R32_FLOAT.0).collect();
    match (depth.as_slice(), floats.as_slice()) {
        ([one], _) | ([], [one]) => Ok(*one),
        ([], []) => Err("no 2D view there is of a depth as large as the light".into()),
        ([], several) => Err(format!("{} 32-bit float images there are as large as the light: {}", several.len(), places_text(several))),
        (several, _) => Err(format!("{} depths there are as large as the light: {}", several.len(), places_text(several))),
    }
}

/// Where the views were found away from their fixed places (the image of each role, then the
/// depth), and the frame of the last search of each that found nothing.
static FOUND: Mutex<[Option<Place>; ROLES_N + 1]> = Mutex::new([None; ROLES_N + 1]);
static SEARCHED: Mutex<[Option<u64>; ROLES_N + 1]> = Mutex::new([None; ROLES_N + 1]);

/// A view the pass after `role` takes (`which`: its index in `FOUND`): at the place it was found the
/// last time or at its fixed place `fixed`, when `fits` takes the view there; else, with `scan`, of
/// the views at `places` that `fits` takes, the one `choose` picks (searched again at most every
/// `SCAN_EVERY` frames while none is found). Where it was found is logged once.
fn find(
    role: Role,
    which: usize,
    what: &str,
    fixed: Place,
    places: impl FnOnce() -> Vec<Place>,
    fits: impl Fn(Place) -> Result<Viewed, String>,
    choose: impl FnOnce(&[(Place, Viewed)]) -> Result<(Place, Viewed), String>,
) -> Result<(Place, Viewed), String> {
    let last = FOUND.lock().unwrap()[which];
    if let Some(place) = last {
        if let Ok(v) = fits(place) {
            return Ok((place, v));
        }
    }
    let why = match fits(fixed) {
        Ok(v) => return Ok((fixed, v)),
        Err(why) => why,
    };
    if !scan() {
        return Err(why);
    }
    let frame = super::present::counters().0;
    if SEARCHED.lock().unwrap()[which].is_some_and(|f| frame < f + SCAN_EVERY) {
        return Err(format!("{why}; the last search of the root tables found none (XCL_GTA6_AMBIENT_SCAN; searched again every {SCAN_EVERY} frames)"));
    }
    let places = places();
    let found: Vec<_> = places.iter().filter(|&&p| p != fixed).filter_map(|&p| fits(p).ok().map(|v| (p, v))).collect();
    match choose(&found) {
        Ok((place, v)) => {
            SEARCHED.lock().unwrap()[which] = None;
            if last != Some(place) {
                FOUND.lock().unwrap()[which] = Some(place);
                crate::xlog_info!(
                    "D3D12.X: title patch {} finds {what} for the pass after {} at {} ({}x{}, format {}, view {}), not at cool_fox's {}: {why}",
                    Patch::Ambient.what(),
                    role.name(),
                    place_text(place),
                    v.width,
                    v.height,
                    v.format,
                    v.index,
                    place_text(fixed)
                );
            }
            Ok((place, v))
        }
        Err(e) => {
            SEARCHED.lock().unwrap()[which] = Some(frame);
            Err(format!("{why}; of the {} slots searched (XCL_GTA6_AMBIENT_SCAN): {e}", places.len()))
        }
    }
}

/// The image the pass after `role` writes (reads, for the light of visible surfaces) and where it is:
/// searched in the unordered-access ranges of the root signature first, then in every slot of its
/// tables (a view there that has an unordered-access view of its own).
unsafe fn find_image(layout: &Layout, args: &[u8], role: Role) -> Result<(Place, Viewed), String> {
    let fits = |place| unsafe { viewed(layout, args, role, place, true) }.and_then(|v| accept(role, &v, &v).map(|_| v));
    let uavs: std::collections::HashSet<Place> = uav_places(layout).into_iter().collect();
    let places = || {
        let mut all = uav_places(layout);
        all.extend(table_places(layout).into_iter().filter(|p| !uavs.contains(p)));
        all
    };
    let pick = |found: &[(Place, Viewed)]| {
        let inside: Vec<_> = found.iter().copied().filter(|(p, _)| uavs.contains(p)).collect();
        choose(role, if inside.is_empty() { found } else { &inside })
    };
    find(role, role as usize, "the image", role.image(), places, fits, pick)
}

/// The scene's depth the resolve's pass reads beside `image`, and where it is.
unsafe fn find_depth(layout: &Layout, args: &[u8], role: Role, image: &Viewed) -> Result<(Place, Viewed), String> {
    let fits = |place| unsafe { viewed(layout, args, role, place, false) }.and_then(|d| if d.index == image.index { Err("the light itself".into()) } else { accept(role, image, &d).map(|_| d) });
    find(role, ROLES_N, "the depth", DEPTH, || table_places(layout), fits, choose_depth)
}

pub(super) unsafe fn after_dispatch(list: *mut c_void, title_pso: usize) {
    let Some(role) = role_of(title_pso) else { return };
    #[cfg(test)]
    FOLLOWED.lock().unwrap().push(title_pso);
    let s = settings();
    let (level, constant) = match s.light {
        Set::Off => return,
        Set::Estimate(level) => (level, 0.0),
        Set::Constant(level) => (0.0, level),
    };
    // (the fallback is the resolve chain's: that is where the title's own light is read, and its
    // report is what the log copies out)
    let report = role == Role::Resolve && constant == 0.0 && s.fallback > 0.0;
    if report {
        note_fallback(s);
    }
    // (the flat light reads neither the cells nor the points, so nothing is kept for it, and the
    // air's bound is a multiple of the estimate)
    if (role != Role::Resolve && constant > 0.0) || (role == Role::Air && s.air == 0.0) {
        return;
    }
    if role != Role::Radiance && constant == 0.0 && !KEPT.load(Ordering::Relaxed) {
        return waits(role, Role::Radiance);
    }
    if role == Role::Air && !GATHERED.load(Ordering::Relaxed) {
        return waits(role, Role::Resolve);
    }
    let Some((layout, args)) = super::bind::compute_arguments(list) else {
        return refused(role, "the list has no compute root signature".into());
    };
    let (image_place, image) = match find_image(&layout, &args, role) {
        Ok(found) => found,
        Err(why) => return refused(role, why),
    };
    // (the passes that read no depth get the image's view there, as cool_fox binds them)
    let depth = match role {
        Role::Resolve => match find_depth(&layout, &args, role, &image) {
            Ok((_, v)) => Some(v),
            Err(why) => return refused(role, why),
        },
        Role::Radiance | Role::Air => None,
    };
    let (size, slices) = match accept(role, &image, depth.as_ref().unwrap_or(&image)) {
        Ok(found) => found,
        Err(why) => return refused(role, why),
    };
    let (Some(image_at), Some(depth_at)) = (super::console_heap::shared_gpu(image.index + 1), super::console_heap::shared_gpu(depth.map_or(image.index + 1, |d| d.index))) else {
        return refused(role, "its views are not in the shared view heap".into());
    };
    let Some(p) = passes() else { return };
    let words = constants(level, constant, s, size);
    if !record(list, p, &words, image_at, depth_at, &schedule(role, constant > 0.0, size, slices), report) {
        return refused(role, "the list or the passes' objects are gone".into());
    }
    match role {
        Role::Radiance => KEPT.store(true, Ordering::Relaxed),
        Role::Resolve if constant == 0.0 => GATHERED.store(true, Ordering::Relaxed),
        _ => {}
    }
    engaged(role, || {
        let what = match role {
            Role::Radiance => "the light of visible surfaces kept in 64x36 cells from".to_string(),
            Role::Air => format!("the light the air scatters bounded to {} times the light around in", s.air),
            Role::Resolve if constant > 0.0 => format!("the diffuse light set to a flat {constant} in"),
            Role::Resolve => format!("the diffuse light raised to {level} of the estimate in"),
        };
        let depth = depth.map_or(String::new(), |d| format!(", depth view {} of a {}x{} resource of format {}", d.index, d.width, d.height, d.format));
        format!("{what} a {}x{}x{slices} image of format {} (view {} at {}){depth}", size[0], size[1], image.format, image.index + 1, place_text(image_place))
    });
}

/// The pipelines with a role whose dispatches the stand-in was asked to follow, one entry each.
#[cfg(test)]
static FOLLOWED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

#[cfg(test)]
pub(super) fn followed_for_test(pso: usize) -> usize {
    FOLLOWED.lock().unwrap().iter().filter(|&&p| p == pso).count()
}

const ROLES_N: usize = 3;

/// The stand-in's passes for `role` were recorded: logged the first time, then at powers of two.
fn engaged(role: Role, detail: impl FnOnce() -> String) {
    static SEEN: [AtomicUsize; ROLES_N] = [const { AtomicUsize::new(0) }; ROLES_N];
    let n = SEEN[role as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_power_of_two() {
        let (what, switch, frame) = (Patch::Ambient.what(), Patch::Ambient.switch(), super::present::counters().0);
        crate::xlog_info!("D3D12.X: title patch {what} ({switch}=0 turns it off) engaged after {} at frame {frame}: {} (#{n})", role.name(), detail());
    }
}

/// Nothing is recorded after a dispatch of `role` because of `why`: logged the first time, when the
/// reason is another (the first 16 times), then at powers of two.
fn refused(role: Role, why: String) {
    static SEEN: [AtomicUsize; ROLES_N] = [const { AtomicUsize::new(0) }; ROLES_N];
    static LAST: Mutex<[(String, u32); ROLES_N]> = Mutex::new([const { (String::new(), 0) }; ROLES_N]);
    let n = SEEN[role as usize].fetch_add(1, Ordering::Relaxed) + 1;
    let changed = {
        let mut last = LAST.lock().unwrap();
        let (said, changes) = &mut last[role as usize];
        let changed = *said != why && *changes < 16;
        if changed {
            *said = why.clone();
            *changes += 1;
        }
        changed
    };
    if changed || n.is_power_of_two() {
        let (what, switch, frame) = (Patch::Ambient.what(), Patch::Ambient.switch(), super::present::counters().0);
        crate::xlog_warn!("D3D12.X: title patch {what} ({switch}=0 turns it off) refuses after {} at frame {frame}, nothing is recorded: {why} (#{n})", role.name());
    }
}

/// Nothing is recorded after a dispatch of `role` yet: the pass after `needs` has not run, so the
/// estimate has no frame's light to begin with, or the air's bound no points (logged the first
/// time, then at powers of two: a run in which that program never runs or is always refused
/// shows here).
fn waits(role: Role, needs: Role) {
    static SEEN: [AtomicUsize; ROLES_N] = [const { AtomicUsize::new(0) }; ROLES_N];
    let n = SEEN[role as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let (what, frame) = (Patch::Ambient.what(), super::present::counters().0);
        crate::xlog_info!("D3D12.X: title patch {what} waits after {} at frame {frame}, nothing is recorded: no dispatch of {} has been followed by its pass yet (#{n})", role.name(), needs.name());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::desc;

    #[test]
    fn the_switches_parse() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse(" 0.0 "), Set::Off);
        assert_eq!(parse("-1"), Set::Off);
        assert_eq!(parse("inf"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse(" 1 "), Set::Estimate(1.0));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant: 2"), Set::Constant(2.0));
        assert_eq!(parse("constant:nan"), Set::Off);
        assert_eq!(parse("constant:0"), Set::Off);
        assert_eq!(parse("constant"), Set::Off);
        // unset is off, unlike the other title patches' switches
        assert!(!set(None));
        assert!(!set(Some("0")));
        assert!(!set(Some("")));
        assert!(set(Some("0.8")));
        assert!(set(Some("constant:0.3")));
        assert_eq!(parse_air(""), 0.0);
        assert_eq!(parse_air("0"), 0.0);
        assert_eq!(parse_air("-2"), 0.0);
        assert_eq!(parse_air(" 1.5 "), 1.5);
        assert_eq!(parse_shade(""), [1.0, 0.6]);
        assert_eq!(parse_shade("0.5"), [0.5, 0.6]);
        assert_eq!(parse_shade("0, 2"), [0.0, 2.0]);
        assert_eq!(parse_shade("x,0.3"), [1.0, 0.3]);
        assert_eq!(parse_shade("-1,nan"), [1.0, 0.6]);
        // the cap and the floor are off, and the slow mean falls as fast as it rises, unless set
        assert_eq!(parse_share("", 0.0), 0.0);
        assert_eq!(parse_share("4", 0.0), 4.0);
        assert_eq!(parse_share("", 0.5), 0.5);
        assert_eq!(parse_share("0", 0.5), 0.0);
        assert_eq!(parse_share("-1", 0.5), 0.5);
        assert_eq!(parse_share("nan", 0.5), 0.5);
        assert_eq!(parse_share(" 0.04 ", 0.0), 0.04);
        // the fallback waits for more than half of the picture unless its switch names a share
        assert_eq!(parse_share("", FALLBACK), 0.5);
        assert_eq!(parse_share("0.9", FALLBACK), 0.9);
        assert_eq!(parse_share("0", FALLBACK), 0.0, "0 is today's behaviour");
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"CS_TemporalResolve"), None);
        assert_eq!(role(b"CS_ScreenProbes_TraceRays_Voxels"), None);
        assert_eq!(role(b""), None);
        // the resolve is not one of the ray passes ray tracing off leaves out: it runs, so there is a dispatch to follow
        assert_eq!(super::super::title_gta6::ray_pass(Role::Resolve.name().as_bytes()), None);
    }

    #[test]
    fn a_pipeline_s_role_lasts_as_long_as_the_pipeline() {
        let (resolve, other) = (0x7a6b_0010usize, 0x7a6b_0020usize);
        assert_eq!(role_of(resolve), None);
        made(resolve, b"xxCS_ScreenProbes_TemporalResolve_a911b49c_Wrapped");
        made(other, b"CS_SortRayTiles");
        assert_eq!(role_of(resolve), Some(Role::Resolve));
        assert_eq!(role_of(other), None);
        gone(resolve);
        assert_eq!(role_of(resolve), None, "a later pipeline at the same address has no role");
        made(resolve, b"CS_CacheScreenSpaceRadiance_eed2fc35");
        assert_eq!(role_of(resolve), Some(Role::Radiance));
        made(resolve, b"CS_SortRayTiles");
        assert_eq!(role_of(resolve), None, "nor one made there without a release seen");
    }

    /// Two tables as the title's programs have them: the unordered-access views in the second.
    fn layout() -> Layout {
        Layout { params: vec![Param::Table { at: 0 }, Param::Table { at: 4 }, Param::Constants { at: 8, count: 2 }], size: 16, uavs: vec![UavBinding::Table { param: 1, at: 4, first: 0, count: 24 }], ..Default::default() }
    }

    fn args(tables: [u32; 2]) -> Vec<u8> {
        [tables[0], tables[1], 7, 9].into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn the_views_are_where_cool_fox_takes_them() {
        assert_eq!((LIGHT, DEPTH, RADIANCE, AIR), ((1, 176), (0, 464), (1, 16), (1, 32)));
        assert_eq!((Role::Resolve.image(), Role::Radiance.image(), Role::Air.image()), (LIGHT, RADIANCE, AIR));
        // a console descriptor is 8 dwords: each place is a whole slot of its table
        for (place, slot) in [(LIGHT, 22), (DEPTH, 58), (RADIANCE, 2), (AIR, 4)] {
            assert_eq!(place.1 * 4 % super::super::console_heap::SLOT, 0);
            assert_eq!(place.1 * 4 / super::super::console_heap::SLOT, slot);
        }
        let (l, a) = (layout(), args([0x0010_0000, 0x0020_0000]));
        assert_eq!(descriptor_at(&l, &a, LIGHT), Ok(0x0020_0000 + 22 * 32));
        assert_eq!(descriptor_at(&l, &a, DEPTH), Ok(0x0010_0000 + 58 * 32));
        assert_eq!(descriptor_at(&l, &a, RADIANCE), Ok(0x0020_0000 + 2 * 32));
        assert_eq!(descriptor_at(&l, &a, AIR), Ok(0x0020_0000 + 4 * 32));
        assert!(descriptor_at(&l, &a, (2, 16)).unwrap_err().contains("not a descriptor table"));
        assert!(descriptor_at(&l, &a, (9, 16)).unwrap_err().contains("not a descriptor table"));
        assert!(descriptor_at(&l, &args([0x0010_0000, 0]), LIGHT).unwrap_err().contains("no table"));
        assert!(descriptor_at(&l, &a[..4], LIGHT).unwrap_err().contains("no table"), "arguments that end before the table");
        assert!(descriptor_at(&l, &args([0, u32::MAX - 31]), LIGHT).is_err(), "a table that ends before the place");
        // the image is written through an unordered-access view: the root signature has a range there
        assert!(declares_uav(&l, LIGHT) && declares_uav(&l, RADIANCE) && declares_uav(&l, AIR));
        assert!(!declares_uav(&l, DEPTH));
        assert!(!declares_uav(&l, (1, 24 * 8)), "the slot after the range");
        let none = Layout { uavs: Vec::new(), ..layout() };
        assert!(!declares_uav(&none, LIGHT));
    }

    #[test]
    fn the_searches_read_every_unordered_access_slot_and_every_table_slot() {
        let mut l = layout();
        l.uavs.insert(0, UavBinding::Root { param: 2, at: 8 });
        l.uavs.push(UavBinding::Table { param: 0, at: 0, first: 60, count: 2 });
        let uavs = uav_places(&l);
        assert_eq!(uavs.len(), 26, "the root view is no table slot");
        assert_eq!((uavs[0], uavs[2], uavs[23], uavs[24], uavs[25]), ((1, 0), RADIANCE, (1, 23 * 8), (0, 60 * 8), (0, 61 * 8)));
        assert!(uavs.contains(&LIGHT) && uavs.contains(&AIR) && !uavs.contains(&DEPTH));
        l.tables = vec![64, u32::MAX, 0];
        let all = table_places(&l);
        assert_eq!(all.len(), 64 + 1024, "a table without a known end: its first 1024 slots");
        assert!(all.contains(&DEPTH) && all.contains(&LIGHT) && all.iter().all(|p| p.0 < 2));
        l.tables = vec![u32::MAX, u32::MAX, 0];
        assert_eq!(table_places(&l).len(), 2048);
        let wide = Layout { uavs: vec![UavBinding::Table { param: 1, at: 4, first: 0, count: u32::MAX }], ..layout() };
        assert_eq!(uav_places(&wide).len(), SCAN_SLOTS, "a search reads at most SCAN_SLOTS");
    }

    #[test]
    fn a_search_takes_the_one_view_and_guesses_only_where_the_pass_reads() {
        let (uint, float) = (DXGI_FORMAT_R32_UINT.0, DXGI_FORMAT_R11G11B10_FLOAT.0);
        let flat = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        let small = ((1, 8), Viewed { index: 50, ..viewed(float, 640, 360, 1, flat) });
        let large = ((1, 40), Viewed { index: 52, ..viewed(float, 1280, 720, 1, flat) });
        assert!(choose(Role::Radiance, &[]).unwrap_err().contains("no view"));
        assert_eq!(choose(Role::Resolve, &[small]), Ok(small));
        assert_eq!(choose(Role::Radiance, &[small, large]), Ok(large), "the light of visible surfaces is only read: the largest");
        let two = choose(Role::Resolve, &[small, large]).unwrap_err();
        assert!(two.contains("2 views") && two.contains("root parameter 1, dword 8") && two.contains("root parameter 1, dword 40"), "{two}");
        assert!(choose(Role::Air, &[small, large]).is_err(), "nor the air's");
        let depth = ((0, 464), Viewed { index: 60, ..viewed(DXGI_FORMAT_R32_TYPELESS.0, 1280, 720, 1, flat) });
        let r32f = ((0, 8), Viewed { index: 62, ..viewed(DXGI_FORMAT_R32_FLOAT.0, 1280, 720, 1, flat) });
        let other = ((0, 16), Viewed { index: 64, ..viewed(uint, 1280, 720, 1, flat) });
        assert_eq!(choose_depth(&[r32f, depth, other]), Ok(depth), "a depth format first");
        assert_eq!(choose_depth(&[r32f, other]), Ok(r32f), "else the one 32-bit float image");
        assert!(choose_depth(&[other]).unwrap_err().contains("no 2D view"));
        assert!(choose_depth(&[depth, ((0, 24), Viewed { index: 66, ..depth.1 })]).unwrap_err().contains("2 depths"));
        assert!(choose_depth(&[r32f, ((0, 32), Viewed { index: 68, ..r32f.1 })]).unwrap_err().contains("2 32-bit float"));
        assert!(depth_like(DXGI_FORMAT_R24G8_TYPELESS.0) && depth_like(DXGI_FORMAT_D32_FLOAT.0) && !depth_like(DXGI_FORMAT_R32_FLOAT.0) && !depth_like(uint));
    }

    fn view(format: i32, ty: ImageType, width: u32, height: u32, depth: u32) -> desc::Image {
        let format = desc::gfx10_format(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format)).unwrap();
        desc::image(&desc::xcl_image_for_view(40, format, ty, width, height, depth, 1)).unwrap()
    }

    #[test]
    fn a_view_of_another_kind_or_format_is_refused() {
        let (uint, float, r32f) = (DXGI_FORMAT_R32_UINT.0, DXGI_FORMAT_R11G11B10_FLOAT.0, 41);
        assert_eq!(view_fits(Role::Resolve, true, &view(uint, ImageType::D2, 1280, 720, 1)), Ok(()));
        assert_eq!(view_fits(Role::Radiance, true, &view(float, ImageType::D2, 1280, 720, 1)), Ok(()));
        assert_eq!(view_fits(Role::Air, true, &view(float, ImageType::D3, 160, 90, 64)), Ok(()));
        // the depth: any format, a plain 2D image
        assert_eq!(view_fits(Role::Resolve, false, &view(r32f, ImageType::D2, 1280, 720, 1)), Ok(()));
        assert!(view_fits(Role::Resolve, false, &view(r32f, ImageType::D2Array, 1280, 720, 2)).unwrap_err().contains("D2Array"));
        // the light as floats, the other images as packed integers
        assert!(view_fits(Role::Resolve, true, &view(float, ImageType::D2, 1280, 720, 1)).unwrap_err().contains("format 26"));
        assert!(view_fits(Role::Resolve, true, &view(10, ImageType::D2, 1280, 720, 1)).unwrap_err().contains("format 10"));
        assert!(view_fits(Role::Radiance, true, &view(uint, ImageType::D2, 1280, 720, 1)).unwrap_err().contains("format 42"));
        // a volume where a plain image is written, and the other way round
        assert!(view_fits(Role::Radiance, true, &view(float, ImageType::D3, 160, 90, 64)).is_err());
        assert!(view_fits(Role::Air, true, &view(float, ImageType::D2, 160, 90, 1)).is_err());
        assert!(view_fits(Role::Resolve, true, &view(uint, ImageType::D2Array, 1280, 720, 2)).is_err());
    }

    fn viewed(format: i32, width: u64, height: u32, slices: u16, dimension: D3D12_RESOURCE_DIMENSION) -> Viewed {
        Viewed { index: 40, view_size: [width as u32, height], format, width, height, slices, dimension: dimension.0 }
    }

    #[test]
    fn an_image_of_another_format_or_size_is_refused() {
        let (uint, float) = (DXGI_FORMAT_R32_UINT.0, DXGI_FORMAT_R11G11B10_FLOAT.0);
        let (flat, volume) = (D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_DIMENSION_TEXTURE3D);
        let light = viewed(uint, 1280, 720, 1, flat);
        let depth = viewed(39, 1280, 720, 1, flat);
        assert_eq!(accept(Role::Resolve, &light, &depth), Ok(([1280, 720], 1)));
        // a typeless resource of the family, shown as integers by its view
        assert_eq!(accept(Role::Resolve, &viewed(DXGI_FORMAT_R32_TYPELESS.0, 1280, 720, 1, flat), &depth), Ok(([1280, 720], 1)));
        // XCL makes a console R9G9B9E5 target R16G16B16A16_FLOAT: not what the pass packs for
        assert!(accept(Role::Resolve, &viewed(10, 1280, 720, 1, flat), &depth).unwrap_err().contains("format 10"));
        assert!(accept(Role::Resolve, &viewed(float, 1280, 720, 1, flat), &depth).unwrap_err().contains("format 26"));
        // the depth of another size (a half-size light, a full-size depth), or not a 2D texture
        assert!(accept(Role::Resolve, &viewed(uint, 640, 360, 1, flat), &depth).unwrap_err().contains("1280x720"));
        assert!(accept(Role::Resolve, &light, &viewed(39, 1280, 720, 4, volume)).is_err());
        // a view of another mip of the light
        assert!(accept(Role::Resolve, &Viewed { view_size: [640, 360], ..light }, &depth).unwrap_err().contains("640x360"));
        // the radiance and the air read no depth: the image stands in for it
        let radiance = viewed(float, 1280, 720, 1, flat);
        assert_eq!(accept(Role::Radiance, &radiance, &radiance), Ok(([1280, 720], 1)));
        assert!(accept(Role::Radiance, &viewed(uint, 1280, 720, 1, flat), &radiance).is_err());
        assert!(accept(Role::Radiance, &viewed(float, 160, 90, 64, volume), &radiance).unwrap_err().contains("not a 2D texture"));
        let air = viewed(float, 160, 90, 64, volume);
        assert_eq!(accept(Role::Air, &air, &air), Ok(([160, 90], 64)));
        assert!(accept(Role::Air, &radiance, &radiance).unwrap_err().contains("not a volume"));
        assert!(accept(Role::Air, &viewed(float, 160, 90, 64, D3D12_RESOURCE_DIMENSION_BUFFER), &air).is_err());
        assert!(accept(Role::Resolve, &viewed(uint, 0, 720, 1, flat), &depth).is_err());
        assert!(accept(Role::Resolve, &viewed(uint, 1 << 20, 720, 1, flat), &viewed(39, 1 << 20, 720, 1, flat)).is_err());
    }

    #[test]
    fn the_light_kept_drops_nan_and_infinity_by_their_bits() {
        assert!(!SOURCE.contains("isnan") && !SOURCE.contains("isinf"));
        assert!(SOURCE.contains("any((bits & 0x7f800000u) == 0x7f800000u) ? 0.0"));
        let dropped = |v: f32| v.to_bits() & 0x7f80_0000 == 0x7f80_0000;
        assert!([f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7f80_0001)].into_iter().all(dropped));
        assert!(![0.0, -0.0, 1.0, f32::MAX, f32::MIN_POSITIVE, 1e-40].into_iter().any(dropped));
    }

    #[test]
    fn the_passes_cover_the_cells_the_points_and_the_image() {
        // the cells in groups of 8x4 threads, the points in groups of 8x2, the image in 8x8
        assert_eq!(schedule(Role::Radiance, false, [1280, 720], 1), [(KEEP, [8, 9, 1]), (MEAN, [1, 1, 1])]);
        assert_eq!(schedule(Role::Resolve, false, [1280, 720], 1), [(PLACE, [8, 9, 1]), (GATHER, [20, 45, 1]), (OWN, [1, 1, 1]), (RAISE, [160, 90, 1])]);
        assert_eq!(schedule(Role::Resolve, false, [1281, 721], 1)[3], (RAISE, [161, 91, 1]));
        assert_eq!(schedule(Role::Resolve, true, [1280, 720], 1), [(RAISE, [160, 90, 1])]);
        assert_eq!(schedule(Role::Air, false, [160, 90], 65), [(CLEAR_AIR, [20, 12, 33])]);
        assert_eq!((CELL_BYTES, POINT_BYTES), ((2 * 2304 + 4) * 16, 160 * 90 * 48));
        assert_eq!(SKY_BYTES, (2 * 2304 + 2) * 16, "the fallback's own slots, after the title's own light");
        assert!(SOURCE.contains("#define OWN_AT (2 * CELLS + 1)
"), "the title's own light sits after the cells' places");
        assert!(SOURCE.contains("#define SKY_AT (2 * CELLS + 2)
"), "the lit part's level and share sit after it");
        assert!(SOURCE.contains("#define DEAD_AT (2 * CELLS + 3)
"), "the share the estimate reads nothing in sits after those");
        // the fallback takes the picture's own level only above the share it waits for, and only
        // while it is asked for
        assert!(SOURCE.contains("if (fallback > 0.0 && (cells[DEAD_AT].x > fallback || cells[SKY_AT].w < 1.0 - fallback))"), "the fallback waits for either share that reads nothing");
        assert!(SOURCE.contains("lit += (l > 0.0) ? 1u : 0u;"), "the share that is lit is counted, not the share that is dark");
        assert!(SOURCE.contains("dead += (dot(c, LUMA) > 0.0) ? 0u : 1u;"), "and the cells the estimate reads nothing in");
        assert_eq!(PARTS[KEEP], "KEEP");
        assert_eq!(PARTS[MEAN], "MEAN");
        assert_eq!(PARTS[PLACE], "PLACE");
        assert_eq!(PARTS[GATHER], "GATHER");
        assert_eq!(PARTS[RAISE], "RAISE");
        assert_eq!(PARTS[CLEAR_AIR], "AIR");
        assert_eq!(PARTS[OWN], "OWN");
        let s = Settings { light: Set::Estimate(0.8), air: 1.5, shade: [1.0, 0.6], cap: 4.0, lowest: 0.5, fall: 0.04, bound: 1.0, fallback: 0.5 };
        let words = constants(0.8, 0.0, &s, [1280, 720]);
        assert_eq!(words, [0.8f32.to_bits(), 0, 1.0f32.to_bits(), 0.6f32.to_bits(), 0.1f32.to_bits(), 0.45f32.to_bits(), 1280, 720, 0.04f32.to_bits(), 4.0f32.to_bits(), 1.5f32.to_bits(), 0.5f32.to_bits(), 0.04f32.to_bits(), 1.0f32.to_bits(), 0.5f32.to_bits(), 0]);
        // the sizes the programs are written for are the ones the buffers and the groups are
        for (name, value) in [("CX", CELLS[0]), ("CY", CELLS[1]), ("PX", POINTS[0]), ("PY", POINTS[1])] {
            assert!(SOURCE.contains(&format!("#define {name} {value}\n")), "{name}");
        }
        assert!(SOURCE.contains("num32BitConstants=16"));
    }

    /// The raise pass's unpack and pack (`SOURCE`), step for step in Rust, so that the packing
    /// can be held against the format's definition without a GPU. The HLSL itself is not run here.
    fn unpack(v: u32) -> [f32; 3] {
        let scale = ((v >> 27) as f32 - 24.0).exp2();
        [(v & 511) as f32 * scale, ((v >> 9) & 511) as f32 * scale, ((v >> 18) & 511) as f32 * scale]
    }

    fn pack(c: [f32; 3]) -> u32 {
        let c = c.map(|x| x.clamp(0.0, 65408.0));
        let top = c[0].max(c[1]).max(c[2]);
        let mut e = (top.max(1e-9).log2().floor() as i32 + 16).clamp(0, 31);
        let mut scale = ((24 - e) as f32).exp2();
        if top * scale >= 511.5 {
            e = (e + 1).min(31);
            scale = ((24 - e) as f32).exp2();
        }
        let m = c.map(|x| ((x * scale + 0.5) as u32).min(511));
        m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
    }

    /// R9G9B9E5 as EXT_texture_shared_exponent defines the conversion (9 mantissa bits, bias 15).
    fn reference(c: [f32; 3]) -> u32 {
        let max = (511.0 / 512.0) * 65536.0f64;
        let c = c.map(|x| (x as f64).clamp(0.0, max));
        let top = c[0].max(c[1]).max(c[2]);
        let mut e = (top.log2().floor() as i32).max(-16) + 16;
        let mut denom = 2f64.powi(e - 24);
        if (top / denom + 0.5).floor() as u32 == 512 {
            denom *= 2.0;
            e += 1;
        }
        let m = c.map(|x| (x / denom + 0.5).floor() as u32);
        m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
    }

    #[test]
    fn the_light_is_packed_as_r9g9b9e5() {
        // known values: a mantissa of 256 is a half, of 511 the most; the exponent is biased by 15
        assert_eq!(pack([1.0, 1.0, 1.0]), 0x8402_0100);
        assert_eq!(unpack(0x8402_0100), [1.0, 1.0, 1.0]);
        assert_eq!(pack([0.0, 0.0, 0.0]), 0);
        assert_eq!(unpack(0), [0.0, 0.0, 0.0]);
        assert_eq!(pack([1.0, 0.5, 0.25]), 0x8101_0100);
        assert_eq!(unpack(0x8101_0100), [1.0, 0.5, 0.25]);
        assert_eq!(pack([0.5, 0.5, 0.5]), 0x7c02_0100);
        assert_eq!(pack([65408.0, 0.0, 0.0]), 0xf800_01ff);
        assert_eq!(unpack(0xffff_ffff), [65408.0, 65408.0, 65408.0]);
        // more than the format holds, and less than nothing
        assert_eq!(pack([1.0e9, 1.0e9, 1.0e9]), 0xffff_ffff);
        assert_eq!(pack([-1.0, 2.0, -0.5]), pack([0.0, 2.0, 0.0]));
        // a mantissa that rounds up to 512 takes the next exponent
        assert_eq!(pack([0.99999, 0.0, 0.0]), pack([1.0, 0.0, 0.0]));
        assert_eq!(unpack(pack([1.999, 1.0, 0.0])), [2.0, 1.0, 0.0]);
        // against the definition, over the range the light has and its ends
        let mut seed = 0x9e37_79b9u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for k in 0..20_000 {
            let scale = 2f32.powi((next() % 36) as i32 - 18);
            let c = [next(), next(), next()].map(|v| (v >> 8) as f32 / (1 << 24) as f32 * scale);
            let c = if k % 7 == 0 { [c[0], 0.0, c[2] * 1.0e-3] } else { c };
            let packed = pack(c);
            assert_eq!(packed, reference(c), "{c:?}");
            // what is read back is within half a step of the largest channel's scale
            let step = ((packed >> 27) as f32 - 24.0).exp2();
            for (was, now) in c.iter().zip(unpack(packed)) {
                assert!((was.min(65408.0) - now).abs() <= step * 0.5001, "{c:?}: {was} read back as {now}");
            }
        }
    }

    /// The six passes compile with the system's compiler, each with the root signature the passes
    /// are bound with. Where the compiler has no cs_5_1 at all (Wine's own d3dcompiler_47, without
    /// Microsoft's beside the test) nothing is told.
    #[test]
    fn the_passes_compile() {
        if super::super::real::compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_1").is_empty() {
            return println!("SKIPPED the_passes_compile: this d3dcompiler_47 compiles no cs_5_1 program");
        }
        for part in PARTS {
            let code = super::super::real::compile(&source(part), "cs_5_1");
            assert!(!code.is_empty(), "the {part} pass compiles");
            assert!(code.windows(4).any(|w| w == b"RTS0"), "the {part} pass carries the root signature");
        }
    }
}
