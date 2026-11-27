// SPDX-License-Identifier: GPL-3.0-or-later
// From CGBJ (cgbj-reupload) commit f4b48ad, crates/victory-shader/src/emit/waterfall_tests.rs. GPL-3.0-or-later, see
// crates/xcl-cgbj/LICENSE and PROVENANCE.md. Unmodified below these three lines.
//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
pub(super) fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

#[test]
fn immediate_sampler_chosen_by_control_flow() {
    let s_mov_lit = |dst: u8, lit: u32| [0xBE80_03FF | (dst as u32) << 16, lit];
    let s_mov_zero = |dst: u8| 0xBE80_0380 | (dst as u32) << 16;
    let s_load_x8 = |dst: u8, base: u8| [0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1), 125 << 25];
    let s_load_x4_off = |dst: u8, base: u8, off: u32| [0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1), 125 << 25 | off];
    // image_sample_lz v4, v[2:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    let image_sample_lz: [u32; 2] = [
        0x3C << 26 | 0x21 << 18 | 1 << 8 | 1 << 3, // dmask=1, dim=2D(1)
        (16u32 >> 2) << 21 | (8u32 >> 2) << 16 | 4 << 8 | 2,
    ];
    // buffer_store_dword v4, off, s[20:23], 0
    let buf_store: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | (20u32 >> 2) << 16 | 4 << 8];

    // Two distinct valid immediate sampler descriptors (different wrap/filter bits).
    let samp_a: [u32; 4] = [0x0000_0092, 0x00C0_0000, 0x1850_0000, 0];
    let samp_b: [u32; 4] = [0x0000_0092, 0x00C0_0000, 0x1850_0001, 0];

    for (case_name, alt) in [("same_sampler", samp_a), ("different_samplers", samp_b)] {
        let mut words = Vec::new();
        // s[2:3] = view table base (s0, 0).
        words.push(0xBE80_0300 | (2 << 16)); // s_mov_b32 s2, s0
        words.push(s_mov_zero(3));
        words.extend_from_slice(&s_load_x8(8, 2)); // s[8:15] = texture SRV (offset 0)
        words.extend_from_slice(&s_load_x4_off(20, 2, 64)); // s[20:23] = raw UAV (offset 16 dwords = 64B)
        // v2 = 0.5, v3 = 0.5 (sample coordinates).
        words.push(0x7E00_02F0 | (2 << 17)); // v_mov_b32 v2, 0.5
        words.push(0x7E00_02F0 | (3 << 17)); // v_mov_b32 v3, 0.5
        // s_cmp_eq_u32 s1, 0; s_cbranch_scc0 else_arm
        words.push(0xBF00_8001); // s_cmp_eq_u32 s1, 0 (const 128)
        let br_else = words.len();
        words.push(0); // placeholder s_cbranch_scc0
        // then arm: load samp_a into s[16:19]
        for (i, &w) in samp_a.iter().enumerate() {
            words.extend_from_slice(&s_mov_lit(16 + i as u8, w));
        }
        let br_join = words.len();
        words.push(0); // placeholder s_branch
        let else_pc = words.len();
        // else arm: load alt into s[16:19]
        for (i, &w) in alt.iter().enumerate() {
            words.extend_from_slice(&s_mov_lit(16 + i as u8, w));
        }
        let join_pc = words.len();
        words[br_else] = 0xBF84_0000 | ((else_pc - br_else - 1) as u32 & 0xFFFF);
        words[br_join] = 0xBF82_0000 | ((join_pc - br_join - 1) as u32 & 0xFFFF);
        words.extend_from_slice(&image_sample_lz);
        words.extend_from_slice(&buf_store);
        words.push(0xBF81_0000); // s_endpgm

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 1,
                    slot: 1,
                },
            ],
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 8,
                    user_data: 0,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 16,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };

        let p = ir::build(&native.code, false, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap_or_else(|e| panic!("{case_name}: lower failed: {e}"));

        let static_samplers = lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect();
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers,
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: Vec::new(),
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }

        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{case_name}/{label}: compute_pipeline failed: {e}"));
            for branch in [0u32, 1u32] {
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Table, 0) => 0,
                        (RootType::Constants, 1) => branch,
                        other => panic!("unexpected user data {other:?}"),
                    };
                }
                let texels = [[1.0f32, 0.0, 0.0, 1.0]];
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[
                                View::Texture(1, 1, &texels),
                                View::Texture(1, 1, &texels),
                                View::RawUav(1),
                            ],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{case_name}/{label}: dispatch failed: {e}"));
                let got = f32::from_bits(out[0][0]);
                assert_eq!(got, 1.0, "{case_name}/{label} branch={branch}");
            }
        }
    }
}
