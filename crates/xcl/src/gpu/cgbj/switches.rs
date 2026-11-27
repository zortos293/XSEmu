//! The switches of the CGBJ compute emitter. Parsing is pure: `parse` takes a lookup so tests
//! can feed it values without touching the environment.

/// Which translator a shader stage asks for (`XCL_SHADER_EMITTER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emitter {
    Xcl,
    Cgbj,
}

/// What happens to a program CGBJ refuses (`XCL_CGBJ_REFUSED`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// Today's translator takes it.
    Xcl,
    /// It stays untranslated, so the skip rule drops its dispatches. For A/B runs only.
    Skip,
}

/// The stages `XCL_SHADER_EMITTER` names. Only `cs` is honoured for now.
pub const STAGES: [&str; 4] = ["cs", "ps", "vs", "ms"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Switches {
    pub compute: Emitter,
    /// Stages other than compute that asked for CGBJ, which they cannot have yet.
    pub unavailable: Vec<&'static str>,
    /// `XCL_SHADER_EMITTER_ONLY`: CGBJ only for these blob hashes.
    pub only: Option<Vec<u64>>,
    /// `XCL_SHADER_EMITTER_SKIP`: never CGBJ for these blob hashes. Wins over `only`.
    pub skip: Vec<u64>,
    pub refused: Refused,
    /// `XCL_CGBJ_WAVE`: exact (the default) or approx.
    pub wave_exact: bool,
    /// `XCL_CGBJ_VALIDATE`: dxil.dll checks every new container (the default).
    pub validate: bool,
}

impl Switches {
    pub const OFF: Switches = Switches {
        compute: Emitter::Xcl,
        unavailable: Vec::new(),
        only: None,
        skip: Vec::new(),
        refused: Refused::Xcl,
        wave_exact: true,
        validate: true,
    };

    /// Whether the compute program with this blob hash goes to CGBJ.
    pub fn wants(&self, hash: u64) -> bool {
        self.compute == Emitter::Cgbj && !self.skip.contains(&hash) && self.only.as_ref().is_none_or(|only| only.contains(&hash))
    }
}

/// Every switch from `get` (an environment lookup). With no stage asking for CGBJ the
/// other switches are not read. An error names the switch; the caller keeps CGBJ off then.
pub fn parse(get: impl Fn(&str) -> Option<String>) -> Result<Switches, String> {
    let stages = match get("XCL_SHADER_EMITTER") {
        Some(v) => parse_emitter(&v).map_err(|e| format!("XCL_SHADER_EMITTER={v:?}: {e}"))?,
        None => [Emitter::Xcl; 4],
    };
    if !stages.contains(&Emitter::Cgbj) {
        return Ok(Switches::OFF);
    }
    let hashes = |name: &str| get(name).map(|v| parse_hashes(&v).map_err(|e| format!("{name}={v:?}: {e}"))).transpose();
    let choice = |name: &str, choices: [&str; 2]| match get(name) {
        None => Ok(true),
        Some(v) => match v.trim() {
            x if x.eq_ignore_ascii_case(choices[0]) => Ok(true),
            x if x.eq_ignore_ascii_case(choices[1]) => Ok(false),
            _ => Err(format!("{name}={v:?}: expected {} or {}", choices[0], choices[1])),
        },
    };
    Ok(Switches {
        compute: stages[0],
        unavailable: STAGES.iter().zip(stages).skip(1).filter(|(_, e)| *e == Emitter::Cgbj).map(|(s, _)| *s).collect(),
        only: hashes("XCL_SHADER_EMITTER_ONLY")?.filter(|h| !h.is_empty()),
        skip: hashes("XCL_SHADER_EMITTER_SKIP")?.unwrap_or_default(),
        refused: if choice("XCL_CGBJ_REFUSED", ["xcl", "skip"])? { Refused::Xcl } else { Refused::Skip },
        wave_exact: choice("XCL_CGBJ_WAVE", ["exact", "approx"])?,
        validate: choice("XCL_CGBJ_VALIDATE", ["1", "0"])?,
    })
}

/// `xcl`, `cgbj`, or a list like `cs:cgbj,ps:xcl`, in the order of [`STAGES`]. A stage the
/// list does not name stays on xcl.
pub fn parse_emitter(v: &str) -> Result<[Emitter; 4], String> {
    let emitter = |e: &str| match e.trim() {
        x if x.eq_ignore_ascii_case("xcl") => Ok(Emitter::Xcl),
        x if x.eq_ignore_ascii_case("cgbj") => Ok(Emitter::Cgbj),
        x => Err(format!("unknown emitter {x:?} (xcl or cgbj)")),
    };
    if !v.contains(':') {
        return Ok([emitter(v)?; 4]);
    }
    let mut out = [Emitter::Xcl; 4];
    for item in v.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (stage, e) = item.split_once(':').ok_or_else(|| format!("{item:?} is not stage:emitter"))?;
        let k = STAGES.iter().position(|s| s.eq_ignore_ascii_case(stage.trim())).ok_or_else(|| format!("unknown stage {:?} (cs, ps, vs or ms)", stage.trim()))?;
        out[k] = emitter(e)?;
    }
    Ok(out)
}

/// Comma-separated blob hashes as the logs print them: the fnv64 of the blob, 16 hex digits
/// (fewer are read as leading zeros, an `0x` prefix is allowed).
pub fn parse_hashes(v: &str) -> Result<Vec<u64>, String> {
    v.split(',')
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .map(|h| {
            let digits = h.strip_prefix("0x").or_else(|| h.strip_prefix("0X")).unwrap_or(h);
            if digits.is_empty() || digits.len() > 16 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("{h:?} is not a hash of up to 16 hex digits"));
            }
            u64::from_str_radix(digits, 16).map_err(|e| format!("{h:?}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn with(pairs: &[(&str, &str)]) -> Result<Switches, String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        parse(|name| map.get(name).cloned())
    }

    #[test]
    fn unset_is_off_and_reads_nothing_else() {
        assert_eq!(with(&[]), Ok(Switches::OFF));
        assert_eq!(with(&[("XCL_SHADER_EMITTER", "xcl"), ("XCL_CGBJ_WAVE", "bogus")]), Ok(Switches::OFF));
        assert!(!Switches::OFF.wants(1));
    }

    #[test]
    fn emitter_forms() {
        use Emitter::{Cgbj, Xcl};
        assert_eq!(parse_emitter("cgbj"), Ok([Cgbj; 4]));
        assert_eq!(parse_emitter(" XCL "), Ok([Xcl; 4]));
        assert_eq!(parse_emitter("cs:cgbj,ps:xcl,vs:xcl,ms:xcl"), Ok([Cgbj, Xcl, Xcl, Xcl]));
        assert_eq!(parse_emitter("ps:cgbj"), Ok([Xcl, Cgbj, Xcl, Xcl]));
        assert!(parse_emitter("cs:cgbj,gs:cgbj").is_err());
        assert!(parse_emitter("cs=cgbj").is_err());
        assert!(parse_emitter("cs:dxc").is_err());
        assert!(parse_emitter("cs").is_err());
    }

    #[test]
    fn other_stages_are_noted_as_unavailable() {
        let s = with(&[("XCL_SHADER_EMITTER", "cgbj")]).unwrap();
        assert_eq!(s.compute, Emitter::Cgbj);
        assert_eq!(s.unavailable, vec!["ps", "vs", "ms"]);
        let s = with(&[("XCL_SHADER_EMITTER", "cs:xcl,vs:cgbj")]).unwrap();
        assert_eq!(s.compute, Emitter::Xcl);
        assert_eq!(s.unavailable, vec!["vs"]);
        assert!(!s.wants(1));
    }

    #[test]
    fn defaults_when_on() {
        let s = with(&[("XCL_SHADER_EMITTER", "cs:cgbj")]).unwrap();
        assert_eq!((s.refused, s.wave_exact, s.validate, s.only.clone(), s.skip.clone()), (Refused::Xcl, true, true, None, vec![]));
        assert!(s.wants(0x1234));
    }

    #[test]
    fn choices() {
        let s = with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_CGBJ_REFUSED", "skip"), ("XCL_CGBJ_WAVE", "approx"), ("XCL_CGBJ_VALIDATE", "0")]).unwrap();
        assert_eq!((s.refused, s.wave_exact, s.validate), (Refused::Skip, false, false));
        assert!(with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_CGBJ_REFUSED", "drop")]).unwrap_err().contains("XCL_CGBJ_REFUSED"));
        assert!(with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_CGBJ_VALIDATE", "yes")]).is_err());
    }

    #[test]
    fn hash_lists() {
        assert_eq!(parse_hashes("00000000000000ff, 0xA,,"), Ok(vec![0xff, 0xa]));
        assert_eq!(parse_hashes(""), Ok(vec![]));
        assert!(parse_hashes("00000000000000fff").is_err());
        assert!(parse_hashes("xyz").is_err());
        assert!(parse_hashes("0x").is_err());
    }

    #[test]
    fn only_and_skip() {
        let s = with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_SHADER_EMITTER_ONLY", "1,2"), ("XCL_SHADER_EMITTER_SKIP", "2")]).unwrap();
        assert!(s.wants(1));
        assert!(!s.wants(2));
        assert!(!s.wants(3));
        let s = with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_SHADER_EMITTER_ONLY", " ")]).unwrap();
        assert_eq!(s.only, None);
        assert!(with(&[("XCL_SHADER_EMITTER", "cgbj"), ("XCL_SHADER_EMITTER_SKIP", "nope")]).unwrap_err().contains("XCL_SHADER_EMITTER_SKIP"));
    }
}
