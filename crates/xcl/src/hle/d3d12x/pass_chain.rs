//! Pass chains: the pipelines a library names with one prefix (`FFX_FSR4_PASS_` by default) run
//! as one chain, and no dispatch of a chain runs until every pipeline of it the title has created
//! is built (`XCL_CHAIN_HOLD=0`: each runs as soon as its own pipeline is built;
//! `XCL_PASS_CHAINS`: the prefixes, comma-separated). Such passes spin-wait on flags their
//! predecessors write, so a pass that runs while a predecessor is still building waits its loop
//! budget out on every lane: play-102 ran FFX_FSR4_PASS_14 and 16 from warm blobs while 12, 18
//! and 22 were still in DXC, the frame held the GPU for 60.2 s, and the device was removed
//! (0x887a0005) at 423.8 s. play-101, where 14, 16 and 18 never ran, kept its device for 848 s.
//! The name is the library's, not the title's, so the prefix holds for any title. While a
//! chain is held its dispatches are skipped and counted as a pending pipeline's are. A member
//! whose build failed keeps the rest held: what it should have written is never written, so the
//! passes after it would spin as on a half-built chain.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Where one pipeline of a chain is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Build {
    Built,
    Building,
    Failed,
}

/// What a chain's dispatches do now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Run,
    /// some are still building
    Hold,
    /// one failed to build
    HoldFailed,
}

fn verdict(states: impl Iterator<Item = Build>) -> Verdict {
    states.fold(Verdict::Run, |v, s| match (v, s) {
        (Verdict::HoldFailed, _) | (_, Build::Failed) => Verdict::HoldFailed,
        (_, Build::Building) => Verdict::Hold,
        (v, Build::Built) => v,
    })
}

fn prefixes() -> &'static [String] {
    static P: OnceLock<Vec<String>> = OnceLock::new();
    P.get_or_init(|| {
        if std::env::var("XCL_CHAIN_HOLD").is_ok_and(|v| v.trim() == "0") {
            crate::xlog_info!("D3D12.X: pass chains off: each pass runs once its own pipeline is built (XCL_CHAIN_HOLD=0)");
            return Vec::new();
        }
        let list = std::env::var("XCL_PASS_CHAINS").unwrap_or_else(|_| "FFX_FSR4_PASS_".into());
        list.split(',').map(str::trim).filter(|p| !p.is_empty()).map(String::from).collect()
    })
}

#[derive(Default)]
struct Chains {
    /// each member's chain
    member: HashMap<usize, usize>,
    /// (prefix, members, last verdict logged with the members not built then)
    chains: Vec<(String, Vec<(usize, String)>, Option<(Verdict, Vec<usize>)>)>,
}

static CHAINS: Mutex<Option<Chains>> = Mutex::new(None);
static ANY: AtomicBool = AtomicBool::new(false);

/// The title named pipeline `pso` `name` (SetName): a member of the chain of its prefix, if any.
pub(super) fn named(pso: usize, name: &str) {
    let Some(prefix) = prefixes().iter().find(|p| name.starts_with(p.as_str())) else { return };
    let mut g = CHAINS.lock().unwrap();
    let c = g.get_or_insert_with(Chains::default);
    forget(c, pso);
    let at = c.chains.iter().position(|(p, ..)| p == prefix).unwrap_or_else(|| {
        c.chains.push((prefix.clone(), Vec::new(), None));
        c.chains.len() - 1
    });
    c.chains[at].1.push((pso, name.to_owned()));
    c.member.insert(pso, at);
    ANY.store(true, Ordering::Release);
}

/// Whether the title's pipeline `pso` is a member of a chain.
pub(super) fn member(pso: usize) -> bool {
    ANY.load(Ordering::Acquire) && CHAINS.lock().unwrap().as_ref().is_some_and(|c| c.member.contains_key(&pso))
}

/// The title released pipeline `pso` for good: it leaves its chain.
pub(super) fn gone(pso: usize) {
    if !ANY.load(Ordering::Acquire) {
        return;
    }
    if let Some(c) = CHAINS.lock().unwrap().as_mut() {
        forget(c, pso);
    }
}

fn forget(c: &mut Chains, pso: usize) {
    if let Some(at) = c.member.remove(&pso) {
        c.chains[at].1.retain(|(p, _)| *p != pso);
    }
}

/// Whether a dispatch with the title's pipeline `pso` runs now: always, unless it is a member of
/// a chain with a member still building or failed (logged when the chain is released or held).
pub(super) fn may_dispatch(pso: usize) -> bool {
    if !ANY.load(Ordering::Acquire) {
        return true;
    }
    may_dispatch_with(pso, super::pso_async::build_state, || super::present::counters().0)
}

fn may_dispatch_with(pso: usize, state: impl Fn(usize) -> Build, frame: impl Fn() -> u64) -> bool {
    let mut g = CHAINS.lock().unwrap();
    let Some(c) = g.as_mut() else { return true };
    let Some(&at) = c.member.get(&pso) else { return true };
    let (prefix, members, logged) = &mut c.chains[at];
    let states: Vec<Build> = members.iter().map(|(p, _)| state(*p)).collect();
    let now = verdict(states.iter().copied());
    let waiting: Vec<usize> = members.iter().zip(&states).filter(|(_, s)| **s != Build::Built).map(|((p, _), _)| *p).collect();
    // (logged again whenever the members it waits for change: a pass built and taken away again
    // holds the chain again)
    if logged.as_ref().is_none_or(|(v, w)| *v != now || *w != waiting) {
        let built = states.len() - waiting.len();
        let total = states.len();
        let frame = frame();
        let names = |of: Build| members.iter().zip(&states).filter(|(_, s)| **s == of).map(|((_, n), _)| n.as_str()).collect::<Vec<_>>().join(", ");
        match now {
            Verdict::Run => crate::xlog_info!("D3D12.X: pass chain {prefix}*: {built} of {total} built, dispatching from frame {frame} (XCL_CHAIN_HOLD=0: each pass as soon as it is built)"),
            Verdict::Hold if logged.is_none() => crate::xlog_info!("D3D12.X: pass chain {prefix}*: held until every pass is built ({built} of {total} built, frame {frame}; building: {})", names(Build::Building)),
            Verdict::Hold => crate::xlog_info!("D3D12.X: pass chain {prefix}*: held again at frame {frame}: {built} of {total} built; building: {}", names(Build::Building)),
            Verdict::HoldFailed => crate::xlog_warn!(
                "D3D12.X: pass chain {prefix}*: held while a member's build has failed ({}); {built} of {total} built, frame {frame}; its dispatches are skipped (XCL_CHAIN_HOLD=0 runs the built ones)",
                names(Build::Failed)
            ),
        }
    }
    *logged = Some((now, waiting));
    now == Verdict::Run
}

#[cfg(test)]
mod tests {
    use super::*;

    /// play-102: 14 and 16 built from warm blobs while 12, 18 and 22 were in DXC; no pass of the
    /// chain runs until every one is built, and a failed one holds the rest.
    #[test]
    fn no_pass_of_a_chain_runs_until_every_pass_is_built() {
        use Build::*;
        assert_eq!(verdict([Built, Built].into_iter()), Verdict::Run);
        assert_eq!(verdict([Built, Building, Built].into_iter()), Verdict::Hold);
        assert_eq!(verdict([Building, Failed, Built].into_iter()), Verdict::HoldFailed);
        assert_eq!(verdict(std::iter::empty()), Verdict::Run);
        let states: std::cell::RefCell<HashMap<usize, Build>> = Default::default();
        let set = |pso, s| drop(states.borrow_mut().insert(pso, s));
        let may = |pso| may_dispatch_with(pso, |p| states.borrow().get(&p).copied().unwrap_or(Built), || 7);
        for (pso, name) in [(0x140, "FFX_FSR4_PASS_14"), (0x160, "FFX_FSR4_PASS_16"), (0x120, "FFX_FSR4_PASS_12"), (0x900, "FFX_OTHER_1")] {
            named(pso, name);
        }
        set(0x120, Building);
        assert!(!may(0x140) && !may(0x160), "12 still in DXC: 14 and 16 wait");
        assert!(may(0x900), "a pipeline of no chain runs");
        assert!(may(0x777), "nor does an unnamed one wait");
        set(0x120, Built);
        assert!(may(0x140) && may(0x160) && may(0x120));
        named(0x180, "FFX_FSR4_PASS_18");
        set(0x180, Building);
        assert!(!may(0x140), "a pass made later holds the chain again");
        set(0x180, Failed);
        assert!(!may(0x160), "a failed pass holds the rest");
        gone(0x180);
        assert!(may(0x160), "released for good, it leaves the chain");
    }
}
