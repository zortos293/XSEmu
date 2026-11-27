// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../../LICENSE and ../../PROVENANCE.md), so it carries that licence.
//! The pipeline of a translation under the XCL binding ([`crate::xcl::translate_compute`]).
//!
//! It is CGBJ's ([`super::translate`]) with three differences. The code is taken as the
//! hardware runs it: a spilled user-data table is memory like any other, so its loads are
//! not rewritten. The descriptor tracer may fail ([`tracked`]). And the result must bind
//! nothing but io and static samplers.

use std::cell::Cell;

use super::{io, lower, structure};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, StageIo};
use crate::resource::Resources;
use crate::xcl::XclShader;
use crate::{ShaderError, ir};

const fn refused(what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset: 0, what }
}

/// The refusal of a wave64 program no form of which has 32-lane waves.
pub(super) const NO_WAVE32: &str = "no wave32 lowering for this wave64 program";

/// The refusal of a wave64 program that only translates with each host wave on its own.
pub(super) const LOCAL_WAVES: &str = "a wave64 program as two independent host waves under the XCL binding";

/// Whether the forms that run each host wave on its own ([`lower::with_local_waves`]) may be
/// kept. Under the binding they are off: on the console the 64 lanes are one wave, so a lane
/// sees what another lane stored earlier in program order, and two host waves without a
/// barrier between them do not. `VICTORY_XCL_LOCAL_WAVES=1` turns them back on (a VICTORY_
/// name, so the translation fingerprint takes its value and cached containers follow it).
pub(super) fn local_waves() -> bool {
    crate::xcl::binding().is_none() || std::env::var_os("VICTORY_XCL_LOCAL_WAVES").is_some_and(|v| v == "1")
}

/// What the retries of a program in its other wave forms were refused for
/// ([`super::emit`]), for the refusal that is reported when no form translates. CGBJ reports
/// the first attempt's, which says where every lane of the wave was needed. A retry that
/// got past that place and was refused for something no form helps (an instruction that is
/// not covered) says more: the last such refusal is reported in its place.
#[derive(Default)]
pub(super) struct Refusals {
    other: Cell<Option<(usize, &'static str)>>,
}

impl Refusals {
    /// Whether a refusal is of the wave's form: another form of the program may not have it.
    fn of_the_form(what: &str) -> bool {
        lower::needs_every_lane(what) || what == lower::LANE_MASK || what == NO_WAVE32
    }

    /// Notes the result of a retry.
    pub(super) fn retried<T>(&self, r: &Result<T, ShaderError>) {
        if let Err(ShaderError::Unsupported { offset, what }) = r
            && !Self::of_the_form(what)
        {
            self.other.set(Some((*offset, *what)));
        }
    }

    /// `r`, or the refusal to report in its place.
    pub(super) fn reported<T>(&self, r: Result<T, ShaderError>) -> Result<T, ShaderError> {
        match (r, self.other.get()) {
            (Err(ShaderError::Unsupported { what, .. }), Some((offset, other))) if Self::of_the_form(what) => Err(ShaderError::Unsupported { offset, what: other }),
            (r, _) => r,
        }
    }
}

/// The refusal of a packed-form container at or over the size cap ([`packed_cap`]). The words
/// name the default cap.
pub(super) const PACKED_TOO_BIG: &str = "packed wave64 output over 1 MiB";

/// The size in bytes at which a packed-form container is refused, from the value of
/// `VICTORY_XCL_PACKED_MAX_BYTES`: 1 MiB when it is unset or not a number, 0 for no cap. The
/// form lowers each vector instruction twice, and the driver compiles CGBJ's DXIL slowly (about
/// 100 s per MB on NVIDIA), so a bigger program goes to xsemu's translator. A VICTORY_ name, so
/// the translation fingerprint takes its value.
fn packed_cap(setting: Option<&str>) -> usize {
    setting.and_then(|v| v.trim().parse().ok()).unwrap_or(1 << 20)
}

/// Whether a first attempt refused for `what` goes through the retries for the packed form: a lane
/// mask read as data, under the binding. (A refusal of one host wave's form is retried anyway.)
pub(super) fn packed_after(what: &str) -> bool {
    crate::xcl::binding().is_some() && what == lower::LANE_MASK
}

/// The packed form of `p` (lower/packed.rs), under the binding on a host of 32-lane waves: every
/// branch on a lane mask is the console wave's ([`structure::structure_wide`]), and each host lane
/// runs two console lanes. `None` where the form does not apply.
pub(super) fn packed(c: &Container, native: &crate::native::NativeProgram, p: &ir::Program) -> Option<Result<(super::HostShader, Option<u32>), ShaderError>> {
    let cfg = crate::xcl::binding()?;
    let StageIo::Cs(cs) = &native.io else { return None };
    if native.stage != HwStage::Cs || !native.wave64 || lower::host_max_wave() != 32 {
        return None;
    }
    let lanes = cs.threads.iter().product::<u32>();
    Some((|| {
        // A console lane past the group runs with its EXEC bit clear, by the launch mask.
        if lanes % 64 != 0 && !cfg.thread_limits {
            return Err(refused("a group that ends inside a console wave, packed without the thread limits"));
        }
        let st = structure::structure(p)?;
        let wide = (0..p.blocks.len()).filter(|&b| st.wave_any.get(b).copied().flatten().is_some()).collect();
        let st = structure::structure_wide(p, &wide)?;
        let io = io::raise(p, native)?;
        let res = crate::resource::track(p, &native.layout, native.stage, Some(&st))?;
        let out = lower::with_packed(|| super::emit_program(c, native, p, &st, &io, &res))?;
        let cap = packed_cap(std::env::var("VICTORY_XCL_PACKED_MAX_BYTES").ok().as_deref());
        if cap != 0 && out.0.dxil.len() >= cap {
            return Err(refused(PACKED_TOO_BIG));
        }
        Ok(out)
    })())
}

/// What the descriptor tracer gave for `p`, as the translation on this thread takes it.
/// Under the binding no descriptor is traced, so a failure of the tracer is not one of the
/// translation. Its results are still needed for addresses in the code (`s_getpc_b64`).
pub(crate) fn tracked(p: &ir::Program, r: Result<Resources, ShaderError>) -> Result<Resources, ShaderError> {
    if r.is_ok() || crate::xcl::binding().is_none() {
        return r;
    }
    if p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_getpc_b64") {
        return Err(refused("code addresses without the tracer"));
    }
    Ok(Resources::default())
}

/// The compute program of a container, under the binding the caller selected.
pub(crate) fn translate(container: &[u8]) -> Result<XclShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
        .ok_or(refused("a container without a vertex, pixel, compute or mesh program"))?;
    let StageIo::Cs(cs) = &native.io else {
        return Err(refused("the XCL binding translates compute only"));
    };
    if native.stage != HwStage::Cs {
        return Err(refused("the XCL binding translates compute only"));
    }
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p);
    let res = crate::resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    crate::xcl::clear_stats();
    let shader = super::emit(&c, &native, &p, &st, &io, &res)?;
    let (stats, _) = crate::xcl::take_stats(&shader.dxil);
    // The host binds io and the static samplers, and nothing else by register.
    if !shader.user_data.is_empty()
        || !shader.root_cbvs.is_empty()
        || !shader.root_srvs.is_empty()
        || shader.root_uavs.iter().any(|&slot| slot != 0)
    {
        return Err(refused("a word or a view of CGBJ's host under the XCL binding"));
    }
    let samplers = shader
        .samplers
        .iter()
        .zip(&shader.static_samplers)
        .map(|(&words, s)| (words, s.filter & 0x80 != 0))
        .collect();
    let form = crate::xcl::take_form();
    let threads = match &form {
        Some((label, _)) if label == crate::xcl::PACKED_FORM => crate::xcl::packed_threads(cs.threads),
        _ => cs.threads,
    };
    Ok(XclShader {
        dxil: shader.dxil,
        samplers,
        threads,
        form,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_WAVES, NO_WAVE32, PACKED_TOO_BIG, Refusals, local_waves, packed_cap};
    use crate::ShaderError;
    use crate::emit::lower;

    fn refused(offset: usize, what: &'static str) -> Result<(), ShaderError> {
        Err(ShaderError::Unsupported { offset, what })
    }

    #[test]
    fn a_retry_refused_for_something_no_form_helps_is_what_is_reported() {
        let first = || refused(0x218, lower::DIVERGENT_MASK_EXCHANGE);
        // No retry, or retries stopped where every lane was needed: the first refusal.
        let r = Refusals::default();
        assert_eq!(r.reported(first()), first());
        r.retried(&refused(0x300, lower::MASK_ELSEWHERE));
        r.retried(&refused(0x400, lower::LANE_MASK));
        r.retried(&Ok::<(), ShaderError>(()));
        assert_eq!(r.reported(first()), first());
        // A retry that got past it: its refusal, and the last such one.
        r.retried(&refused(0x500, "buffer_load_format_d16_x"));
        r.retried(&refused(0x600, "ray tests under the XCL binding"));
        r.retried(&refused(0x700, lower::DIVERGENT_MASK_EXCHANGE));
        assert_eq!(r.reported(first()), refused(0x600, "ray tests under the XCL binding"));
        assert_eq!(r.reported(refused(0, NO_WAVE32)), refused(0x600, "ray tests under the XCL binding"));
        // A first refusal that is not of the wave's form stays, and so does a result.
        assert_eq!(r.reported(refused(0x20, "s_memtime")), refused(0x20, "s_memtime"));
        assert_eq!(r.reported(Ok::<u32, ShaderError>(7)), Ok(7));
        assert_eq!(r.reported(Err::<(), _>(ShaderError::NotDxbc)), Err(ShaderError::NotDxbc));
    }

    #[test]
    fn each_host_wave_on_its_own_is_refused_under_the_binding_unless_asked_for() {
        let asked = std::env::var_os("VICTORY_XCL_LOCAL_WAVES").is_some_and(|v| v == "1");
        assert!(local_waves());
        {
            let _scope = crate::xcl::BindingScope::enter(crate::xcl::XclConfig::default());
            assert_eq!(local_waves(), asked);
        }
        assert!(local_waves());
        // The refusal is not one of the wave's form, so it is reported in place of the first.
        let r = Refusals::default();
        r.retried(&refused(0, LOCAL_WAVES));
        assert_eq!(r.reported(refused(0x218, lower::DIVERGENT_MASK_EXCHANGE)), refused(0, LOCAL_WAVES));
    }

    /// M1d: the packed form's size cap is 1 MiB unless the switch names another, and 0 is none. A
    /// packed container refused for its size is the refusal reported, so the host sees why.
    #[test]
    fn a_packed_container_at_the_size_cap_is_refused_and_reported() {
        assert_eq!(packed_cap(None), 1 << 20);
        assert_eq!(packed_cap(Some("0")), 0);
        assert_eq!(packed_cap(Some("4096")), 4096);
        assert_eq!(packed_cap(Some("lots")), 1 << 20);
        let r = Refusals::default();
        r.retried(&refused(0, PACKED_TOO_BIG));
        assert_eq!(r.reported(refused(0x218, lower::DIVERGENT_MASK_EXCHANGE)), refused(0, PACKED_TOO_BIG));
    }
}
