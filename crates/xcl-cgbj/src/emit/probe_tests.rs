// SPDX-License-Identifier: GPL-3.0-or-later
// From CGBJ (cgbj-reupload) commit f4b48ad, crates/victory-shader/src/emit/probe_tests.rs. GPL-3.0-or-later, see
// crates/xcl-cgbj/LICENSE and PROVENANCE.md. Unmodified below these three lines.
//! Not a test of the translator: what the host's compiler makes of DXIL.
//! `VICTORY_PROBE=<folder>` and `--ignored probe`: each `.dxil` there (a
//! compute program with a constant buffer `b0` of four dwords and a raw
//! buffer, slot 0 of the view heap) is run on the card. `sem*` programs are
//! given the bits of 0.0, infinity, -0.0 and a NaN and what they store is
//! printed; of the others the time is (several runs of 4096 groups).

use super::HostShader;
use super::host::{Dispatch, Host, Root, View};
use crate::native::{HwStage, RootType};

#[test]
#[ignore = "a tool: VICTORY_PROBE=<folder>"]
fn probe() {
    let Some(folder) = std::env::var_os("VICTORY_PROBE") else { return };
    let host = Host::new(false).expect("a card");
    let mut files: Vec<_> = std::fs::read_dir(folder)
        .expect("the folder")
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "dxil"))
        .collect();
    files.sort();
    for file in files {
        let name = file.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: std::fs::read(&file).expect("the file"),
            user_data: (0..4)
                .map(|n| super::UserData { index: n, slot: 0, root: RootType::Constants, dword: n })
                .collect(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(made) => made,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let run = |constants: [u32; 4], groups: u32| {
            let began = std::time::Instant::now();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(constants.to_vec()))],
                        views: &[View::RawUav(64)],
                        samplers: &[],
                        groups: [groups, 1, 1],
                    },
                )
                .expect("a dispatch");
            (out[0].clone(), began.elapsed().as_secs_f64() * 1e3)
        };
        if name.starts_with("sem") {
            let (out, _) = run([0, 0x7F80_0000, 0x8000_0000, 0x7FC0_0000], 1);
            for (n, x) in out.iter().enumerate().take(40) {
                println!("{name} {n:>2}: {x:08x} {}", f32::from_bits(*x));
            }
            continue;
        }
        let constants = [0.37f32.to_bits(), 0.11f32.to_bits(), 16384, 0];
        run(constants, 64);
        let mut times: Vec<f64> = (0..9).map(|_| run(constants, 4096).1).collect();
        times.sort_by(f64::total_cmp);
        println!("{name}: {:.2} ms (least {:.2})", times[times.len() / 2], times[0]);
    }
}
