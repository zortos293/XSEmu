//! How many driver pipelines a console mesh pair costs.
//!
//! A pair is one console mesh program and the pixel shader it draws with. As a host mesh
//! shader the pair is one pipeline: the program and the pixel shader in one pipeline state
//! stream (`gpu::recomp::pso::console_mesh_native`, `real::native_mesh_stream`). In the
//! two-pass fallback it is two: the program as a compute pass, then a pass-through mesh shader
//! with the pixel shader (`gpu::recomp::pso::console_mesh`, `real::finish_two_pass`). See
//! `docs/mesh-shaders.md` for which hosts take which form.
//!
//! Every entry below is one `CreatePipelineState` or `CreateComputePipelineState` call on the
//! device. The build lines print the count `mesh_pipelines` gives, and `mesh_counts` prints the
//! compiles that actually reached the driver, so a run reports both what a pair should cost and
//! what it did cost.

/// The form a console mesh pair takes on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshForm {
    /// The console program as the host's mesh shader: one pipeline.
    OnePass,
    /// The console program as a compute pass, then a pass-through mesh pass: two pipelines.
    TwoPass,
}

/// The record observer of `real::record_variant`, which the two-pass path builds on demand.
/// It exists for the depth prepass probe: `XCL_DEPTH_PREPASS_QUERY=1` with
/// `XCL_MESH_EXPORT_CLASS=raw-ps`, plus `XCL_MESH_PRODUCER_SITES` for its paired program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordProbe {
    Off,
    /// The observer's own mesh pipeline.
    Raster,
    /// The observer's mesh pipeline and its paired compute program.
    Paired,
}

/// One driver pipeline of a mesh pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshPso {
    /// The title's mesh pipeline: the console program (one-pass) or the pass-through mesh
    /// shader (two-pass), with the pair's pixel shader.
    Raster,
    /// The console program as a compute pass (two-pass only).
    Compute,
    /// The record observer's own mesh pipeline.
    ProbeRaster,
    /// The record observer's paired compute program.
    ProbeCompute,
}

/// The pipelines a pair builds, in order. The record observer belongs to the two-pass path,
/// which is where `real::finish_two_pass` builds it; a pair that runs as a host mesh shader has
/// none.
pub fn mesh_pipelines(form: MeshForm, probe: RecordProbe) -> &'static [MeshPso] {
    use RecordProbe::*;
    const ONE: &[MeshPso] = &[MeshPso::Raster];
    const TWO: &[MeshPso] = &[MeshPso::Raster, MeshPso::Compute];
    const PROBED: &[MeshPso] = &[MeshPso::Raster, MeshPso::Compute, MeshPso::ProbeRaster];
    const PAIRED: &[MeshPso] = &[MeshPso::Raster, MeshPso::Compute, MeshPso::ProbeRaster, MeshPso::ProbeCompute];
    match (form, probe) {
        (MeshForm::OnePass, _) => ONE,
        (MeshForm::TwoPass, Off) => TWO,
        (MeshForm::TwoPass, Raster) => PROBED,
        (MeshForm::TwoPass, Paired) => PAIRED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_costs_one_pipeline_as_a_host_mesh_shader_and_two_in_two_passes() {
        assert_eq!(mesh_pipelines(MeshForm::OnePass, RecordProbe::Off), [MeshPso::Raster]);
        assert_eq!(mesh_pipelines(MeshForm::TwoPass, RecordProbe::Off), [MeshPso::Raster, MeshPso::Compute]);
        assert_eq!(mesh_pipelines(MeshForm::OnePass, RecordProbe::Off).len(), 1);
        assert_eq!(mesh_pipelines(MeshForm::TwoPass, RecordProbe::Off).len(), 2);
    }

    #[test]
    fn the_record_observer_adds_one_pipeline_and_its_paired_program_one_more() {
        assert_eq!(mesh_pipelines(MeshForm::TwoPass, RecordProbe::Raster).len(), 3);
        assert_eq!(mesh_pipelines(MeshForm::TwoPass, RecordProbe::Paired).len(), 4);
    }

    #[test]
    fn a_host_mesh_shader_pair_never_builds_the_record_observer() {
        assert_eq!(mesh_pipelines(MeshForm::OnePass, RecordProbe::Raster), [MeshPso::Raster]);
        assert_eq!(mesh_pipelines(MeshForm::OnePass, RecordProbe::Paired), [MeshPso::Raster]);
    }
}
