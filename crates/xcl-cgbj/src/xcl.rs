// SPDX-License-Identifier: GPL-3.0-or-later
// Written for xsemu (CAPA-86, ZOR-632). Not CGBJ code, but part of crates/xcl-cgbj, which is CGBJ's
// shader translator (GPL-3.0-or-later, see ../LICENSE and ../PROVENANCE.md), so it carries that licence.
//! What xsemu adds to the translator: policy its host sets before translating, and facts it
//! reads back after. The call sites in CGBJ's files are marked `xsemu:`.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

static WAVE_EXACT: AtomicBool = AtomicBool::new(false);

/// Which forms a console wave64 compute program may take on a host of 32-lane waves.
///
/// Off (the default) is CGBJ's own behaviour. On, only the forms that keep the console's
/// results are tried: no half-wave acting as a wave of its own outside the allocation idiom
/// (as `VICTORY_WAVE_LOCAL=idiom`), and no wave-wide branch where a console wave is two host
/// waves, which is the form that puts group barriers inside branches and loops. A wave-wide
/// branch in a program whose waves are the host's has no barrier and stays allowed.
/// A program no allowed form covers is refused, for the host to translate another way.
/// Set it once, before any translation; it is part of [`crate::emit::translation_fingerprint`].
pub fn set_wave_exact(on: bool) {
    WAVE_EXACT.store(on, Ordering::Relaxed);
}

/// See [`set_wave_exact`].
pub fn wave_exact() -> bool {
    WAVE_EXACT.load(Ordering::Relaxed)
}

/// What a wave form does to the console's results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormKind {
    /// The console's results (the allocation idiom hands out the same slots in another order).
    Exact,
    /// Each half of a console wave acts as a wave of its own: a wave-wide result is the half's.
    Approximate,
    /// The console's results, by group barriers inside branches or loops.
    Wide,
}

impl FormKind {
    pub fn name(self) -> &'static str {
        match self {
            FormKind::Exact => "exact",
            FormKind::Approximate => "approx",
            FormKind::Wide => "wide",
        }
    }
}

/// The kind of the form with this label (the labels are those of `emit::emit`).
pub fn form_kind(label: &str) -> FormKind {
    if label.contains("APPROXIMATE") {
        FormKind::Approximate
    } else if label.starts_with("wave-wide branches") && !label.contains("one host wave") {
        FormKind::Wide
    } else {
        FormKind::Exact
    }
}

thread_local! {
    /// The form the last translation on this thread was kept in, when it was not the first one tried.
    static FORM: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) fn clear_form() {
    FORM.with(|f| *f.borrow_mut() = None);
}

pub(crate) fn note_form(label: &str) {
    FORM.with(|f| *f.borrow_mut() = Some(label.to_string()));
}

/// The form the last successful translation on this thread took, and its kind; `None` when
/// it translated as it stood. Forgets it.
pub fn take_form() -> Option<(String, FormKind)> {
    FORM.with(|f| f.borrow_mut().take()).map(|label| {
        let kind = form_kind(&label);
        (label, kind)
    })
}

/// Bumped on any change to the code emitted under the XCL binding. The host keys its caches on it.
pub const BINDING_VERSION: u32 = 5;

/// The label ([`take_form`]) of the packed form: a console wave64 program on 32-lane waves with
/// each host lane holding two console lanes. Its numthreads are [`packed_threads`].
pub const PACKED_FORM: &str = "packed wave64";

/// The numthreads of the packed form for a console thread group of `console`: 32 host threads per
/// console wave of 64, in one dimension. The host dispatches the console's group counts.
pub fn packed_threads(console: [u32; 3]) -> [u32; 3] {
    [console.iter().product::<u32>().div_ceil(64) * 32, 1, 1]
}

/// What the host tells a translation under the XCL binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XclConfig {
    /// Title memory windows in the view heap. The game has 4. 1 is refused.
    pub mem_windows: u32,
    /// Honour the dispatch thread limits in io[119..121].
    pub thread_limits: bool,
    /// First static sampler register in space 1. 0 for compute.
    pub sampler_base: u32,
}

impl Default for XclConfig {
    fn default() -> Self {
        XclConfig { mem_windows: 4, thread_limits: true, sampler_base: 0 }
    }
}

/// Counts of what a translation under the XCL binding emitted, for the census.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XclStats {
    /// Instructions in the function, after the clean-up passes.
    pub instructions: u32,
    /// Resolver instances for an access of one dword.
    pub resolves_single: u32,
    /// Resolver instances for an access of several dwords (the next-page one counts too).
    pub resolves_multi: u32,
    /// s_load_dwordx4 and wider (descriptor-sized records).
    pub record_loads: u32,
    /// s_load_dword and s_load_dwordx2.
    pub scalar_loads: u32,
    /// s_buffer_load_* and buffer_* instructions.
    pub buffer_accesses: u32,
    /// global_* and flat_* instructions.
    pub global_accesses: u32,
    /// Image instructions.
    pub image_ops: u32,
    /// Static samplers declared.
    pub static_samplers: u32,
    /// Image instructions that take their sampler from the sampler heap.
    pub heap_samplers: u32,
    /// Page groups: bases whose two pages are resolved once for several loads.
    pub page_groups: u32,
    /// Scalar loads that read through a page group, without a resolver of their own.
    pub grouped_loads: u32,
}

/// A compute program translated under the XCL binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XclShader {
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// Static samplers in register order: the S# words and whether it compares.
    pub samplers: Vec<([u32; 4], bool)>,
    /// numthreads as declared: the console's, or [`packed_threads`] of them in the packed form.
    pub threads: [u32; 3],
    /// The wave form the translation took, as [`take_form`] gives it.
    pub form: Option<(String, FormKind)>,
    pub stats: XclStats,
}

thread_local! {
    /// The XCL binding's settings while a translation under it runs on this thread.
    static BINDING: std::cell::Cell<Option<XclConfig>> = const { std::cell::Cell::new(None) };
}

/// The XCL binding's settings, when the translation running on this thread is under it.
/// The whole translation runs on the calling thread, so a thread-local is enough.
pub(crate) fn binding() -> Option<XclConfig> {
    BINDING.with(|b| b.get())
}

use crate::emit::census::Census;

thread_local! {
    /// The instruction counts of the module written last on this thread.
    static WRITTEN: std::cell::RefCell<Option<Census>> = const { std::cell::RefCell::new(None) };
    /// What each lowering under the binding emitted since [`clear_stats`], by its container:
    /// a program may be lowered in several forms before one is kept.
    static STATS: std::cell::RefCell<Vec<(u64, XclStats, Census)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A container's length and an FNV-1a hash of its bytes, folded into one number.
fn digest(dxil: &[u8]) -> u64 {
    let hash = dxil.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    hash ^ (dxil.len() as u64).rotate_left(40)
}

pub(crate) fn note_census(census: Census) {
    WRITTEN.with(|w| *w.borrow_mut() = Some(census));
}

pub(crate) fn note_stats(dxil: &[u8], mut stats: XclStats) {
    let census = WRITTEN.with(|w| w.borrow_mut().take()).unwrap_or_default();
    stats.instructions = census.counts[0];
    STATS.with(|s| s.borrow_mut().push((digest(dxil), stats, census)));
}

pub(crate) fn clear_stats() {
    STATS.with(|s| s.borrow_mut().clear());
}

/// The statistics and the instruction counts of the lowering that gave `dxil`, and forgets
/// the others.
pub(crate) fn take_stats(dxil: &[u8]) -> (XclStats, Census) {
    let (all, d) = (STATS.with(|s| s.take()), digest(dxil));
    all.into_iter().rev().find(|s| s.0 == d).map(|s| (s.1, s.2)).unwrap_or_default()
}


/// Selects the XCL binding on this thread until it is dropped, also on a panic.
pub(crate) struct BindingScope(Option<XclConfig>);

impl BindingScope {
    pub(crate) fn enter(cfg: XclConfig) -> Self {
        BindingScope(BINDING.with(|b| b.replace(Some(cfg))))
    }
}

impl Drop for BindingScope {
    fn drop(&mut self) {
        BINDING.with(|b| b.set(self.0));
    }
}

/// Translates the compute program of a console shader container under the XCL binding:
/// every memory instruction works on register values and reaches title memory through the
/// emulator's windows and page directory, as xsemu's own translator does.
///
/// The caller sets the host facts once before (`emit::set_host_max_wave`,
/// `emit::set_host_rdna`, [`set_wave_exact`]). The view heap limit and raw view settings
/// are not used under the binding.
pub fn translate_compute(container: &[u8], cfg: &XclConfig) -> Result<XclShader, crate::ShaderError> {
    if cfg.mem_windows != 4 {
        return Err(crate::ShaderError::Unsupported { offset: 0, what: "the XCL binding with other than 4 memory windows" });
    }
    let _scope = BindingScope::enter(*cfg);
    crate::emit::xcl::translate(container)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms_are_classified_by_what_they_do_to_results() {
        assert_eq!(form_kind("without early-outs"), FormKind::Exact);
        assert_eq!(form_kind("without skips, scalars followed"), FormKind::Exact);
        assert_eq!(form_kind("local waves (idiom), without early-outs"), FormKind::Exact);
        assert_eq!(form_kind("local waves (APPROXIMATE), without skips"), FormKind::Approximate);
        assert_eq!(form_kind("local waves (APPROXIMATE, any mask)"), FormKind::Approximate);
        assert_eq!(form_kind("wave-wide branches"), FormKind::Wide);
        assert_eq!(form_kind("wave-wide branches, one host wave"), FormKind::Exact);
        assert_eq!(form_kind("after jump threading"), FormKind::Exact);
    }

    #[test]
    fn the_binding_is_selected_for_one_call_only() {
        assert_eq!(binding(), None);
        {
            let _scope = BindingScope::enter(XclConfig::default());
            assert_eq!(binding(), Some(XclConfig::default()));
            // (a panic inside the scope leaves it too)
            let r = std::panic::catch_unwind(|| {
                let _inner = BindingScope::enter(XclConfig { sampler_base: 3, ..XclConfig::default() });
                panic!("inside");
            });
            assert!(r.is_err());
            assert_eq!(binding(), Some(XclConfig::default()));
        }
        assert_eq!(binding(), None);
    }

    #[test]
    fn one_memory_window_is_refused() {
        let cfg = XclConfig { mem_windows: 1, ..XclConfig::default() };
        assert!(matches!(translate_compute(&[], &cfg), Err(crate::ShaderError::Unsupported { .. })));
        assert_eq!(translate_compute(&[], &XclConfig::default()), Err(crate::ShaderError::NotDxbc));
    }

    #[test]
    fn a_noted_form_is_taken_once() {
        clear_form();
        assert_eq!(take_form(), None);
        note_form("wave-wide branches");
        assert_eq!(take_form(), Some(("wave-wide branches".to_string(), FormKind::Wide)));
        assert_eq!(take_form(), None);
    }
}
