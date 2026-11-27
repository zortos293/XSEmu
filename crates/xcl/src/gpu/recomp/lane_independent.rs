//! Which wave64 compute programs never look at another lane, so that each console lane can run
//! as its own host thread (the lane-independent form: no `WaveSize`, no ballots, no exchange).
//!
//! Console code is compiled from HLSL, where a thread only meets another through wave
//! intrinsics, group-shared memory and barriers; the wave-wide machinery in the machine code
//! (EXEC, mask registers, branches when no lane is left) is the compiler's way to run 64 threads
//! at once. VIctory (Marvisak/VIctory 63bcca4, `victory-shader/src/emit/lower.rs` 555-606) takes
//! that as given for every program without a cross-lane instruction or a mask read as a number.
//! Here it is proved per program, and a program the proof does not cover keeps the wave form.
//!
//! The claim: lane L computes the same in its console wave W as in a wave of 64 copies of
//! itself, which is what the lane-independent form runs (a ballot of a wave of copies is all
//! lanes or none, a lane read gives the reader's own value). Three things carry it:
//!
//! * typing ([`Kind`]): every scalar register holds a number, which only comes from numbers
//!   (constants, user data, memory no thread is said to write, scalar arithmetic) and is
//!   therefore the same in both waves,
//!   or a lane mask, which only comes from EXEC, vector compares and carries, the constants 0
//!   and -1 and bitwise operations on those, and whose bit L is the same in both. A mask read
//!   as a number, a number used as a mask and every cross-lane instruction end the proof;
//! * `v_readfirstlane_b32` only of a register every lane the wave started with holds the same
//!   number in (written from numbers where EXEC provably holds all of them, `lane_guards`);
//! * a branch on a wave-wide mask test (`s_cbranch_execz`, VCCZ, the SCC of a mask operation or
//!   of a mask compared with 0 or -1) can go one way in W and the other in the wave of copies
//!   only when other lanes decide it; lane L's bit of the tested mask is then known. From
//!   there W runs code the copies skip until it reaches where they went ([`Analysis::solo`]):
//!   that code must keep lane L out of EXEC wherever it writes a vector register or memory,
//!   must reach that place on every path, and must leave every scalar register that is read
//!   afterwards as it was (a mask: bit L as it was).
//!
//! Not covered, so on the wave form whatever they do: LDS and `s_barrier` (the console
//! compiler drops barriers for LDS shared inside one wave, and the proof does not follow memory),
//! indexed scalar registers, indirect jumps, sampling with implicit derivatives.
//!
//! Taken as given, not proved: a lane does not read from memory what another lane of its wave
//! stored earlier in the program. A wave keeps that order (its lanes run instruction by
//! instruction) and separate threads do not; in the HLSL the code was compiled from it is a
//! data race, as it is between two waves on the console. Where the code itself says the
//! memory is shared the proof ends: a lane's load flagged GLC or DLC that the program's own
//! stores or atomics can come before (`Rule::SharedLoad`).

use std::collections::{BTreeSet, HashMap};

use super::lane_guards::{self, mask_step, on_cycles, Bdd, Half, Word, ATOMS_PER_INST};
use super::x64_masks::{graph, sgpr_use, GuardExits, Regs, SgprUse, ALL};
use super::{Mode, Options};
use crate::gpu::rdna2::{self, Decoded, Variant};

/// Why a program keeps the wave form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Rule {
    NotWave64Compute,
    Instrumented,
    IndirectJump,
    CrossLane,
    Lds,
    Barrier,
    Unclassified,
    IndexedRegisters,
    MaskAsData,
    DataAsMask,
    ConstantLaneSet,
    MixedRegister,
    MaskTestAsData,
    SharedScalar,
    SharedLoad,
    OpaqueBranch,
    FirstLane,
    SoloVector,
    SoloScalar,
    SoloEnd,
    ProofSize,
}

impl Rule {
    pub(super) fn text(self) -> &'static str {
        match self {
            Rule::NotWave64Compute => "not a wave64 compute program",
            Rule::Instrumented => "an instrumented translation (loop profile, material gate)",
            Rule::IndirectJump => "an indirect jump",
            Rule::CrossLane => "reads or writes another lane",
            Rule::Lds => "LDS",
            Rule::Barrier => "s_barrier",
            Rule::Unclassified => "an instruction the proof does not cover",
            Rule::IndexedRegisters => "scalar registers addressed through m0",
            Rule::MaskAsData => "a lane mask read as a number",
            Rule::DataAsMask => "a number used as a lane mask",
            Rule::ConstantLaneSet => "a constant lane set other than all lanes or none",
            Rule::MixedRegister => "a register that holds a lane mask on one path and a number on another",
            Rule::MaskTestAsData => "a wave-wide mask test read as a number",
            Rule::SharedScalar => "a scalar load of memory another thread may be writing (GLC / DLC)",
            Rule::SharedLoad => "a load of memory another thread may be writing (GLC / DLC) after the program's own stores or atomics",
            Rule::OpaqueBranch => "a branch on a lane set the proof cannot follow",
            Rule::FirstLane => "v_readfirstlane_b32 of a value that is not the same in every lane",
            Rule::SoloVector => "the wave runs a vector instruction with this lane on where its own thread has left",
            Rule::SoloScalar => "the wave changes a scalar register that is read later where a lane's own thread has left",
            Rule::SoloEnd => "the wave may end or stay where a lane's own thread has left",
            Rule::ProofSize => "the lane predicates outgrew the proof",
        }
    }
}

/// The first instruction the proof fails at (None: the program as a whole) and the rule.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Reject {
    pub rule: Rule,
    pub pc: Option<u64>,
    pub inst: String,
    pub detail: String,
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pc) = self.pc {
            write!(f, "+{pc:#x} {}: ", self.inst)?;
        }
        f.write_str(self.rule.text())?;
        if !self.detail.is_empty() {
            write!(f, " ({})", self.detail)?;
        }
        Ok(())
    }
}

/// What the proof of a qualifying program covered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Proof {
    /// branches on a wave-wide mask test, each with its skipped code checked
    pub mask_branches: usize,
    pub first_lane_reads: usize,
}

#[cfg(test)]
thread_local! {
    /// tests: every program qualifies on this thread (what the form computes for a program the
    /// proof refuses)
    pub(super) static ASSUME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What a scalar register holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// a number, the same in the console wave and in the wave of copies
    Data,
    /// the low / high dword of a lane mask
    Lo,
    Hi,
    /// the low / high dword of a 64-bit 0 or -1: a number, and the mask of no lane or all
    CLo,
    CHi,
    /// a mask on one path, a number on another: nothing may read it
    Mixed,
}

fn join_kind(a: Kind, b: Kind) -> Kind {
    use Kind::*;
    match (a, b) {
        _ if a == b => a,
        (Lo, CLo) | (CLo, Lo) => Lo,
        (Hi, CHi) | (CHi, Hi) => Hi,
        (Data, CLo | CHi) | (CLo | CHi, Data) | (CLo, CHi) | (CHi, CLo) => Data,
        _ => Mixed,
    }
}

/// A wave-wide test of a mask: what SCC says about it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Test {
    /// SCC = some lane is set
    Any,
    /// SCC = no lane is set
    None,
    /// SCC = every lane is set
    All,
    /// SCC = some lane is clear
    NotAll,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scc {
    /// from numbers: the same in both waves
    Data,
    /// a test of the mask in s[reg : reg + 1], which still holds the tested value
    Test(u8, Test),
    /// depends on other lanes in a way the proof does not follow
    Opaque,
}

#[derive(Clone, PartialEq)]
struct State {
    s: [Kind; 128],
    scc: Scc,
    /// VGPRs every lane the wave started with holds the same number in
    uniform: [u128; 2],
}

impl State {
    fn entry() -> State {
        let mut s = [Kind::Data; 128];
        s[126] = Kind::Lo;
        s[127] = Kind::Hi;
        State { s, scc: Scc::Data, uniform: [0; 2] }
    }

    /// Whether anything changed.
    fn join(&mut self, o: &State) -> bool {
        let mut changed = false;
        for (a, &b) in self.s.iter_mut().zip(&o.s) {
            let j = join_kind(*a, b);
            changed |= j != *a;
            *a = j;
        }
        let scc = if self.scc == o.scc { self.scc } else { Scc::Opaque };
        changed |= scc != self.scc;
        self.scc = scc;
        for (a, &b) in self.uniform.iter_mut().zip(&o.uniform) {
            changed |= *a & b != *a;
            *a &= b;
        }
        changed
    }

    fn is_uniform(&self, v: u64) -> bool {
        v < 256 && self.uniform[(v >> 7) as usize] >> (v & 127) & 1 != 0
    }

    fn set_uniform(&mut self, v: u64, on: bool) {
        if v < 256 {
            let bit = 1u128 << (v & 127);
            let w = &mut self.uniform[(v >> 7) as usize];
            *w = if on { *w | bit } else { *w & !bit };
        }
    }
}

/// A 64-bit scalar source as a mask operation sees it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pair {
    Mask,
    /// 0 or -1
    Const,
    /// a number in registers
    Data,
    /// a number in the instruction (an inline constant other than 0 and -1, a literal)
    Number,
    Bad(Rule),
}

fn regs(mut m: Regs) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (m != 0).then(|| {
            let r = m.trailing_zeros() as usize;
            m &= m - 1;
            r
        })
    })
}

/// A 64-bit operation that is exact lane by lane: s[dst] = a op b with operands or the result
/// inverted, a copy, or a saveexec (`b` = EXEC, the result into EXEC and its old (`wrexec`: new)
/// value into `dst`). The operations `lane_guards::mask_step` follows exactly.
struct MaskOp {
    copy: bool,
    a: u64,
    b: Option<u64>,
    dst: Option<u64>,
    saveexec: bool,
    wrexec: bool,
}

fn mask_op(d: &Decoded) -> Option<MaskOp> {
    let name = d.inst.name;
    if !name.ends_with("_B64") || !matches!(d.enc.name, "ENC_SOP1" | "ENC_SOP2") {
        return None;
    }
    let field = |f: &str| rdna2::field_value(d.enc, &d.words, f);
    let word = name.strip_prefix("S_")?.split('_').next()?;
    let logic = matches!(word, "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "NAND" | "NOR" | "XNOR");
    let dst = field("SDST").filter(|&c| c != 125);
    let a = field("SSRC0")?;
    if dst == Some(127) {
        return None;
    }
    let saveexec = name.contains("SAVEEXEC") || name.contains("WREXEC");
    if saveexec {
        return (logic || matches!(word, "ANDN1" | "ORN1")).then_some(MaskOp { copy: false, a, b: Some(126), dst, saveexec, wrexec: name.contains("WREXEC") });
    }
    let (copy, b) = match (d.enc.name, name) {
        ("ENC_SOP1", "S_MOV_B64") => (true, None),
        ("ENC_SOP1", "S_NOT_B64") => (false, None),
        ("ENC_SOP2", _) if logic && name.len() == word.len() + 6 => (false, Some(field("SSRC1")?)),
        _ => return None,
    };
    Some(MaskOp { copy, a, b, dst, saveexec, wrexec: false })
}

fn is_vector(d: &Decoded) -> bool {
    d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP") || d.enc.name.starts_with("ENC_FLAT") || matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG")
}

/// Vector instructions whose scalar result is a lane mask: the lane's condition or carry in its
/// bit, 0 for a lane outside EXEC (`emit::Em::write_mask`).
fn writes_mask(name: &str) -> bool {
    name.starts_with("V_CMP") || name.contains("_CO_") || name.starts_with("V_MAD_U64_U32") || name.starts_with("V_MAD_I64_I32") || name.starts_with("V_DIV_SCALE")
}

/// Operations that write their one destination register whole, from their listed sources and
/// nothing else: no bit of the old destination stays (the accumulators, the 16-bit and packed
/// operations, a swap), no other register is written (the relative moves), nothing comes from
/// the lane or its neighbours. Where every lane runs one on the same numbers, every lane gets
/// the same number. What is not listed is not trusted with that.
fn whole_result(d: &Decoded) -> bool {
    d.variant == Variant::Plain
        && matches!(d.enc.name, "ENC_VOP1" | "ENC_VOP2" | "ENC_VOP3")
        && matches!(
            d.inst.name,
            "V_MOV_B32" | "V_NOT_B32" | "V_BFREV_B32" | "V_FFBH_U32" | "V_FFBL_B32" | "V_FFBH_I32" | "V_BCNT_U32_B32"
                | "V_CVT_F32_I32" | "V_CVT_F32_U32" | "V_CVT_U32_F32" | "V_CVT_I32_F32" | "V_CVT_RPI_I32_F32" | "V_CVT_FLR_I32_F32" | "V_CVT_OFF_F32_I4"
                | "V_CVT_F32_UBYTE0" | "V_CVT_F32_UBYTE1" | "V_CVT_F32_UBYTE2" | "V_CVT_F32_UBYTE3"
                | "V_FRACT_F32" | "V_TRUNC_F32" | "V_CEIL_F32" | "V_RNDNE_F32" | "V_FLOOR_F32" | "V_EXP_F32" | "V_LOG_F32" | "V_RCP_F32" | "V_RCP_IFLAG_F32" | "V_RSQ_F32" | "V_SQRT_F32" | "V_SIN_F32" | "V_COS_F32"
                | "V_ADD_F32" | "V_SUB_F32" | "V_SUBREV_F32" | "V_MUL_F32" | "V_MUL_LEGACY_F32" | "V_MIN_F32" | "V_MAX_F32" | "V_LDEXP_F32"
                | "V_FMA_F32" | "V_FMA_LEGACY_F32" | "V_FMAMK_F32" | "V_FMAAK_F32"
                | "V_MUL_I32_I24" | "V_MUL_HI_I32_I24" | "V_MUL_U32_U24" | "V_MUL_HI_U32_U24" | "V_MUL_LO_U32" | "V_MUL_HI_U32" | "V_MUL_HI_I32" | "V_MAD_I32_I24" | "V_MAD_U32_U24"
                | "V_MIN_I32" | "V_MAX_I32" | "V_MIN_U32" | "V_MAX_U32" | "V_MIN3_F32" | "V_MIN3_I32" | "V_MIN3_U32" | "V_MAX3_F32" | "V_MAX3_I32" | "V_MAX3_U32" | "V_MED3_F32" | "V_MED3_I32" | "V_MED3_U32"
                | "V_LSHRREV_B32" | "V_ASHRREV_I32" | "V_LSHLREV_B32" | "V_AND_B32" | "V_OR_B32" | "V_XOR_B32" | "V_XNOR_B32" | "V_XOR3_B32" | "V_OR3_B32" | "V_AND_OR_B32" | "V_LSHL_OR_B32"
                | "V_ADD_NC_U32" | "V_SUB_NC_U32" | "V_SUBREV_NC_U32" | "V_ADD_NC_I32" | "V_SUB_NC_I32" | "V_ADD3_U32" | "V_XAD_U32" | "V_LSHL_ADD_U32" | "V_ADD_LSHL_U32"
                | "V_BFE_U32" | "V_BFE_I32" | "V_BFI_B32" | "V_BFM_B32" | "V_ALIGNBIT_B32" | "V_ALIGNBYTE_B32" | "V_LERP_U8" | "V_PERM_B32"
        )
}

fn cross_lane(d: &Decoded) -> bool {
    let name = d.inst.name;
    matches!(d.variant, Variant::Dpp16 | Variant::Dpp8)
        || name.starts_with("DS_SWIZZLE")
        || name.contains("PERMUTE")
        || name.starts_with("V_MBCNT")
        // (v_perm_b32 picks bytes inside one lane)
        || matches!(name, "V_READLANE_B32" | "V_WRITELANE_B32" | "V_PERMLANE16_B32" | "V_PERMLANEX16_B32")
}

/// Image instructions that take their level of detail from the neighbouring lanes.
fn implicit_derivatives(name: &str) -> bool {
    (name.starts_with("IMAGE_SAMPLE") || name.starts_with("IMAGE_GATHER4") || name == "IMAGE_GET_LOD") && !name.split('_').any(|p| matches!(p, "L" | "LZ" | "D"))
}

/// The scalar source codes of an instruction's explicit operands (SDWA: from its extension).
fn source_codes(d: &Decoded) -> Vec<u64> {
    let field = |f: &str| rdna2::field_value(d.enc, &d.words, f);
    let mut out = Vec::new();
    if d.variant == Variant::Sdwa {
        let ext = d.words.get(1).copied().unwrap_or(0);
        if ext >> 23 & 1 != 0 {
            out.push(u64::from(ext & 0xff));
        }
        if ext >> 31 != 0 {
            out.extend(field("VSRC1"));
        }
        return out;
    }
    for o in d.ienc.operands {
        if let Some(f) = o.field.filter(|f| o.input && matches!(*f, "SSRC0" | "SSRC1" | "SRC0" | "SRC1" | "SRC2")) {
            out.extend(field(f));
        }
    }
    out
}

fn scc_operand(d: &Decoded, input: bool) -> bool {
    d.ienc.operands.iter().any(|o| o.ty == "OPR_SSRC_SPECIAL_SCC" && if input { o.input } else { o.output })
}

fn reads_scc(d: &Decoded) -> bool {
    scc_operand(d, true) || source_codes(d).contains(&253)
}

type Fail = (Rule, String);

/// Where a branch on a wave-wide mask test can part the two waves: the tested mask, lane L's
/// bit of it when they part, and whether the console wave is then the one that takes the branch.
struct Parting {
    reg: u8,
    own: bool,
    wave_taken: bool,
}

struct Analysis<'a> {
    insts: &'a [Decoded],
    succ: Vec<Vec<usize>>,
    /// instructions that start a stretch of straight-line code
    leader: Vec<bool>,
    /// instructions that can run more than once
    cyclic: Vec<bool>,
    /// where EXEC provably holds every lane the wave started with
    full: BTreeSet<u64>,
    /// `predicates`
    words: HashMap<usize, Vec<Word>>,
    /// the first atom that names a register word's value: 128 where two waves part (`solo`),
    /// then 128 for each instruction paths meet at (`predicates`)
    names: u32,
}

impl Analysis<'_> {
    fn pair(&self, st: &State, code: u64) -> Pair {
        match code {
            125 | 128 | 193 => Pair::Const,
            251..=253 => Pair::Bad(Rule::MaskTestAsData),
            0..=126 => match (st.s[code as usize], st.s[code as usize + 1]) {
                (Kind::Mixed, _) | (_, Kind::Mixed) => Pair::Bad(Rule::MixedRegister),
                (Kind::CLo, Kind::CHi) => Pair::Const,
                (Kind::Lo | Kind::CLo, Kind::Hi | Kind::CHi) => Pair::Mask,
                (Kind::Lo | Kind::Hi, _) | (_, Kind::Lo | Kind::Hi) => Pair::Bad(Rule::MaskAsData),
                _ => Pair::Data,
            },
            127 => Pair::Bad(Rule::MaskAsData),
            _ => Pair::Number,
        }
    }

    fn data(st: &State, r: usize) -> Result<(), Fail> {
        match st.s[r] {
            Kind::Data | Kind::CLo | Kind::CHi => Ok(()),
            Kind::Lo | Kind::Hi => Err((Rule::MaskAsData, reg_name(r))),
            Kind::Mixed => Err((Rule::MixedRegister, reg_name(r))),
        }
    }

    /// The state after instruction `i`, and the first rule it breaks. The state is one to go
    /// on from either way.
    fn step(&self, i: usize, st: &mut State) -> Option<Fail> {
        let d = &self.insts[i];
        let name = d.inst.name;
        if cross_lane(d) {
            return Some((Rule::CrossLane, String::new()));
        }
        if super::emit::NOOPS.iter().any(|n| n.eq_ignore_ascii_case(name)) {
            return None;
        }
        let u = sgpr_use(d);
        if u.maybe == ALL {
            return Some((Rule::IndexedRegisters, String::new()));
        }
        // a test of a mask that is written again says nothing about its new value
        if let Scc::Test(reg, _) = st.scc {
            if u.maybe & (3 << reg) != 0 {
                st.scc = Scc::Opaque;
            }
        }
        match d.enc.name {
            "ENC_SOPP" => match name {
                // a conditional branch's test is `parting`'s
                "S_ENDPGM" | "S_BRANCH" | "S_ROUND_MODE" | "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" | "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" | "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => None,
                "S_BARRIER" => Some((Rule::Barrier, String::new())),
                _ => Some((Rule::Unclassified, String::new())),
            },
            "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" | "ENC_SMEM" => self.scalar(d, &u, st).err(),
            "ENC_DS" => Some((Rule::Lds, String::new())),
            _ if is_vector(d) => self.vector(d, &u, st).err(),
            _ => Some((Rule::Unclassified, String::new())),
        }
    }

    fn scalar(&self, d: &Decoded, u: &SgprUse, st: &mut State) -> Result<(), Fail> {
        let name = d.inst.name;
        if let Some(m) = mask_op(d) {
            return self.mask_logic(&m, st);
        }
        let field = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        if matches!(name, "S_CMP_EQ_U64" | "S_CMP_LG_U64") {
            let (a, b) = (field("SSRC0"), field("SSRC1"));
            let eq = name == "S_CMP_EQ_U64";
            st.scc = match (self.pair(st, a), self.pair(st, b)) {
                (Pair::Bad(rule), _) | (_, Pair::Bad(rule)) => return Err((rule, String::new())),
                (Pair::Mask, _) if b == 128 => Scc::Test(a as u8, if eq { Test::None } else { Test::Any }),
                (Pair::Mask, _) if b == 193 => Scc::Test(a as u8, if eq { Test::All } else { Test::NotAll }),
                (_, Pair::Mask) if a == 128 => Scc::Test(b as u8, if eq { Test::None } else { Test::Any }),
                (_, Pair::Mask) if a == 193 => Scc::Test(b as u8, if eq { Test::All } else { Test::NotAll }),
                (Pair::Mask, _) | (_, Pair::Mask) => return Err((Rule::MaskAsData, "compared as a number".into())),
                _ => Scc::Data,
            };
            return Ok(());
        }
        // a jump the decoder resolved (`rdna2::resolve_far_jumps`) is a branch
        if d.far_target.is_some() {
            return Ok(());
        }
        let refused = ["S_SETPC", "S_SWAPPC", "S_RFE", "S_CALL", "S_GETREG", "S_SETREG", "S_SUBVECTOR", "S_SET_GPR_IDX", "S_SETVSKIP", "S_WQM", "S_QUADMASK"];
        if refused.iter().any(|p| name.starts_with(p)) || (d.enc.name == "ENC_SMEM" && !name.starts_with("S_LOAD_DWORD") && !name.starts_with("S_BUFFER_LOAD_DWORD")) {
            return Err((Rule::Unclassified, String::new()));
        }
        // a number from numbers; what a wave loads once from memory that is being written, its
        // lanes' threads would each load at their own time
        let mut result = Ok(());
        if d.enc.name == "ENC_SMEM" && ["GLC", "DLC"].iter().any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(1)) {
            result = Err((Rule::SharedScalar, String::new()));
        }
        for r in regs(u.whole | u.lane) {
            result = result.and(Self::data(st, r));
        }
        if reads_scc(d) && st.scc != Scc::Data {
            result = result.and(Err((Rule::MaskTestAsData, String::new())));
        }
        for r in regs(u.maybe) {
            if r >= 126 {
                result = result.and(Err((Rule::DataAsMask, "into exec".into())));
            } else {
                st.s[r] = Kind::Data;
            }
        }
        if scc_operand(d, false) {
            st.scc = Scc::Data;
        }
        result
    }

    fn mask_logic(&self, m: &MaskOp, st: &mut State) -> Result<(), Fail> {
        let a = self.pair(st, m.a);
        let b = m.b.map(|b| self.pair(st, b));
        let mut result = Ok(());
        let sources = [Some(a), b];
        let sources = || sources.iter().flatten().copied();
        if let Some(rule) = sources().find_map(|p| if let Pair::Bad(rule) = p { Some(rule) } else { None }) {
            result = Err((rule, String::new()));
        }
        let mask = sources().any(|p| p == Pair::Mask);
        if mask && result.is_ok() {
            if sources().any(|p| p == Pair::Number) {
                result = Err((Rule::ConstantLaneSet, String::new()));
            } else if sources().any(|p| p == Pair::Data) {
                result = Err((Rule::DataAsMask, String::new()));
            }
        }
        // the result: a mask from a mask, 0 or -1 from those alone, else a number
        let kinds = if mask {
            (Kind::Lo, Kind::Hi)
        } else if sources().all(|p| p == Pair::Const) {
            (Kind::CLo, Kind::CHi)
        } else {
            (Kind::Data, Kind::Data)
        };
        let write = |st: &mut State, dst: u64, kinds: (Kind, Kind), result: &mut Result<(), Fail>| {
            if dst == 126 && kinds.0 == Kind::Data && result.is_ok() {
                *result = Err(if sources().any(|p| p == Pair::Number) { (Rule::ConstantLaneSet, String::new()) } else { (Rule::DataAsMask, "into exec".into()) });
                return;
            }
            st.s[dst as usize] = kinds.0;
            st.s[dst as usize + 1] = kinds.1;
        };
        let tested = if m.saveexec {
            let old = (st.s[126], st.s[127]);
            write(st, 126, kinds, &mut result);
            if let Some(dst) = m.dst.filter(|&dst| dst != 126) {
                write(st, dst, if m.wrexec { kinds } else { old }, &mut result);
            }
            Some(126)
        } else {
            if let Some(dst) = m.dst {
                write(st, dst, kinds, &mut result);
            }
            m.dst
        };
        // SCC: the result is not 0 (a copy leaves SCC alone)
        if !m.copy {
            st.scc = match (mask, tested) {
                (false, _) => Scc::Data,
                (true, Some(reg)) => Scc::Test(reg as u8, Test::Any),
                (true, None) => Scc::Opaque,
            };
        }
        result
    }

    fn vector(&self, d: &Decoded, u: &SgprUse, st: &mut State) -> Result<(), Fail> {
        let name = d.inst.name;
        if d.enc.name == "ENC_MIMG" && implicit_derivatives(name) {
            return Err((Rule::CrossLane, "implicit derivatives".into()));
        }
        // a buffer or flat access redirected to LDS
        if d.enc.name != "ENC_MIMG" && rdna2::field_value(d.enc, &d.words, "LDS") == Some(1) {
            return Err((Rule::Lds, String::new()));
        }
        let mut result = Ok(());
        // v_div_fmas takes the lane's bit of VCC (the scale v_div_scale asked for)
        let fmas = name.starts_with("V_DIV_FMAS");
        for r in regs(if fmas { u.whole & !super::x64_masks::VCC } else { u.whole }) {
            result = result.and(Self::data(st, r));
        }
        // a mask read lane by lane (the selector of v_cndmask_b32, a carry in): VCC, or the
        // VOP3 form's third source
        if fmas || super::x64_masks::reads_mask_per_lane(name) {
            let code = if !fmas && d.enc.field("SRC2").is_some() { rdna2::field_value(d.enc, &d.words, "SRC2").unwrap_or(0) } else { 106 };
            result = result.and(match self.pair(st, code) {
                Pair::Mask | Pair::Const => Ok(()),
                Pair::Bad(rule) => Err((rule, String::new())),
                Pair::Number => Err((Rule::ConstantLaneSet, String::new())),
                Pair::Data => Err((Rule::DataAsMask, reg_name(code as usize))),
            });
        } else if u.lane != 0 {
            return Err((Rule::Unclassified, "its mask operand".into()));
        }
        if reads_scc(d) && st.scc != Scc::Data {
            result = result.and(Err((Rule::MaskTestAsData, String::new())));
        }
        let (vr, vw) = rdna2::vgpr_uses(d);
        let mut every_lane = false;
        if name == "V_READFIRSTLANE_B32" {
            if vr.is_empty() || !vr.iter().all(|&v| st.is_uniform(v)) {
                result = result.and(Err((Rule::FirstLane, vr.first().map_or(String::new(), |v| format!("v{v}")))));
            }
            for r in regs(u.maybe) {
                if r >= 126 {
                    result = result.and(Err((Rule::DataAsMask, "into exec".into())));
                } else {
                    st.s[r] = Kind::Data;
                }
            }
        } else if u.maybe != 0 {
            let w: Vec<usize> = regs(u.maybe).collect();
            if !writes_mask(name) || w.len() != 2 || w[1] != w[0] + 1 || u.written != u.maybe {
                return Err((Rule::Unclassified, "its scalar result".into()));
            }
            st.s[w[0]] = Kind::Lo;
            st.s[w[1]] = Kind::Hi;
        } else {
            // the same number in every lane: from numbers, where EXEC holds every lane
            every_lane = whole_result(d) && u.lane == 0 && vw.len() == 1 && self.full.contains(&d.pc) && vr.iter().all(|&v| st.is_uniform(v)) && result.is_ok();
        }
        if name.starts_with("V_MOVREL") || name.starts_with("V_SWAPREL") {
            // v[dst + m0], v[src + m0]: any register
            st.uniform = [0; 2];
        } else if name == "V_SWAP_B32" && self.full.contains(&d.pc) && vw.len() == 2 {
            // every lane swaps: each register is what the other was
            let (a, b) = (st.is_uniform(vw[0]), st.is_uniform(vw[1]));
            st.set_uniform(vw[0], b);
            st.set_uniform(vw[1], a);
            return result;
        }
        // memory and image loads write up to four registers from the first, and one or two
        // more with TFE / LWE
        let flag = |f: &str| rdna2::field_value(d.enc, &d.words, f) == Some(1);
        let span = match d.enc.name {
            "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG" if flag("TFE") || flag("LWE") => 6,
            e if e.starts_with("ENC_M") || e.starts_with("ENC_FLAT") => 4,
            _ => 1,
        };
        for &v in &vw {
            for k in 0..span {
                st.set_uniform(v + k, every_lane);
            }
        }
        result
    }

    /// Whether the conditional branch `i` tests a mask wave-wide, and how the waves can part
    /// there. None: it tests a number (both waves go the same way) or is no such branch.
    fn parting(&self, i: usize, st: &State) -> Result<Option<Parting>, Fail> {
        let name = self.insts[i].inst.name;
        // (the tested mask, what the test says, branch taken when it holds)
        let (reg, test, taken_when) = match name {
            "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => match st.scc {
                Scc::Data => return Ok(None),
                Scc::Opaque => return Err((Rule::OpaqueBranch, String::new())),
                Scc::Test(reg, test) => (reg, test, name == "S_CBRANCH_SCC1"),
            },
            "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" | "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => {
                let reg = if name.contains("VCC") { 106 } else { 126 };
                match self.pair(st, reg) {
                    Pair::Mask => (reg as u8, Test::Any, name.ends_with("NZ")),
                    Pair::Bad(rule) => return Err((rule, String::new())),
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        // they part when other lanes decide the test: (lane L's bit, what the console wave's
        // test then says)
        let (own, wave_holds) = match test {
            Test::Any => (false, true),
            Test::None => (false, false),
            Test::All => (true, false),
            Test::NotAll => (true, true),
        };
        Ok(Some(Parting { reg, own, wave_taken: wave_holds == taken_when }))
    }

    /// Scalar registers (and SCC) that may be read from each instruction on before they are
    /// written. Every instruction that is not a scalar one reads EXEC.
    fn live_in(&self) -> Vec<(Regs, bool)> {
        let n = self.insts.len();
        let uses: Vec<(Regs, Regs, bool, bool)> = self
            .insts
            .iter()
            .map(|d| {
                let u = sgpr_use(d);
                let (r, w) = (reads_scc(d), scc_operand(d, false));
                if u.maybe == ALL {
                    return (ALL, 0, true, false);
                }
                let exec = if matches!(d.enc.name, "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" | "ENC_SOPP" | "ENC_SMEM") { 0 } else { super::x64_masks::EXEC };
                (u.whole | u.lane | (u.maybe & !u.written) | exec, u.written, r, w && !r)
            })
            .collect();
        let mut live = vec![(0 as Regs, false); n];
        loop {
            let mut changed = false;
            for i in (0..n).rev() {
                let (mut out, mut out_scc) = (0, false);
                for &j in &self.succ[i] {
                    out |= live[j].0;
                    out_scc |= live[j].1;
                }
                let (gen, kill, gen_scc, kill_scc) = uses[i];
                let inn = (gen | (out & !kill), gen_scc || (out_scc && !kill_scc));
                changed |= inn != live[i];
                live[i] = inn;
            }
            if !changed {
                return live;
            }
        }
    }

    /// `lane_guards`' lane predicates of every scalar register word before each stretch of
    /// straight-line code, for the console wave from its entry EXEC. Where paths meet, a word
    /// they do not agree on is not unknown as there but an atom of its own, the word's value
    /// at that place: a loop's EXEC is then one atom throughout an iteration, and what the
    /// loop makes of it stays a predicate of it. A word made of a place's atoms on arriving
    /// there is from an earlier time round, so it never agrees.
    fn predicates(&self, bdd: &mut Bdd) -> HashMap<usize, Vec<Word>> {
        let mut entry = vec![Word::Unknown; 128];
        entry[126] = Word::Known(1, Half::Lo);
        entry[127] = Word::Known(1, Half::Hi);
        let mut state = HashMap::from([(0, entry)]);
        let mut work = BTreeSet::from([0usize]);
        while let Some(l) = work.pop_first() {
            let mut s = state[&l].clone();
            let mut i = l;
            loop {
                mask_step(bdd, &self.insts[i], i as u32, true, self.cyclic[i], &mut s);
                match self.succ[i][..] {
                    [j] if j == i + 1 && !self.leader[j] => i = j,
                    _ => break,
                }
            }
            for &j in &self.succ[i] {
                let Some(t) = state.get_mut(&j) else {
                    state.insert(j, s.clone());
                    work.insert(j);
                    continue;
                };
                let base = self.names + 128 * (1 + j as u32);
                let mut changed = false;
                for (w, (a, b)) in t.iter_mut().zip(&s).enumerate() {
                    let named = Word::Known(bdd.var(base + w as u32), Half::Any);
                    let stale = matches!(*b, Word::Known(q, _) if bdd.uses_range(q, base, base + 128));
                    if *a != named && (stale || *a != *b) {
                        *a = named;
                        changed = true;
                    }
                }
                if changed {
                    work.insert(j);
                }
            }
        }
        state
    }

    fn words_before(&self, bdd: &mut Bdd, i: usize) -> Option<Vec<Word>> {
        let mut l = i;
        while !self.leader[l] {
            l -= 1;
        }
        let mut s = self.words.get(&l)?.clone();
        for j in l..i {
            mask_step(bdd, &self.insts[j], j as u32, true, self.cyclic[j], &mut s);
        }
        Some(s)
    }

    /// The console wave alone from the branch `i` where the waves part (`p`) to where the
    /// wave of copies went: lane L stays out of every vector write and memory access, the wave
    /// gets there on every path, and what is read afterwards is as the copies have it.
    ///
    /// Every scalar register word is a predicate of the lane (`lane_guards`); what is known of
    /// lane L is its bit of the tested mask (`given`, per half of the wave L may be in). A word
    /// the wave may make again on its way (a loop) cannot be told from its new value by its
    /// atoms, so it goes by a name for its value at the parting, and its atoms leave `given`.
    fn solo(&self, bdd: &mut Bdd, i: usize, p: &Parting, before: &[Option<State>], live: &[(Regs, bool)]) -> Result<(), (usize, Fail)> {
        let (taken, next) = match self.succ[i][..] {
            [t, n] => (t, n),
            _ => return Err((i, (Rule::SoloEnd, "no code after the branch".into()))),
        };
        let (wave, join) = if p.wave_taken { (taken, next) } else { (next, taken) };
        if wave == join {
            return Ok(());
        }
        let size = |j: usize| (j, (Rule::ProofSize, String::new()));
        let kinds = &before[i].as_ref().unwrap().s;
        // what the wave can run alone, and what of it more than once
        let mut region = BTreeSet::new();
        let mut stack = vec![wave];
        while let Some(j) = stack.pop() {
            if j != join && region.insert(j) {
                stack.extend(&self.succ[j]);
            }
        }
        let again: BTreeSet<u32> = region.iter().filter(|&&j| self.cyclic[j]).map(|&j| j as u32).collect();
        let name = |bdd: &mut Bdd, w: usize| Word::Known(bdd.var(self.names + w as u32), Half::Any);
        let mut s = self.words_before(bdd, i).ok_or(size(i))?;
        for (w, word) in s.iter_mut().enumerate() {
            if *word == Word::Unknown {
                *word = name(bdd, w);
            }
        }
        let mut given = [1u32; 2];
        for (k, g) in given.iter_mut().enumerate() {
            let Word::Known(t, _) = s[p.reg as usize + k] else { return Err(size(i)) };
            *g = if p.own { t } else { bdd.not(t).ok_or(size(i))? };
        }
        let half = |kind: Kind| match kind {
            Kind::Lo | Kind::CLo => Some(0),
            Kind::Hi | Kind::CHi => Some(1),
            _ => None,
        };
        for w in 0..128 {
            let Word::Known(q, h) = s[w] else { continue };
            if let Some(k) = half(kinds[w]) {
                // lane L's bit of this mask is settled by the given one
                if bdd.apply(0, q, given[k]).ok_or(size(i))? == 0 {
                    s[w] = Word::Known(0, h);
                    continue;
                }
                let not = bdd.not(q).ok_or(size(i))?;
                if bdd.apply(0, not, given[k]).ok_or(size(i))? == 0 {
                    s[w] = Word::Known(1, h);
                    continue;
                }
            }
            if uses(bdd, q, &again) {
                s[w] = name(bdd, w);
            }
        }
        let mut memo = HashMap::new();
        for g in &mut given {
            *g = exists(bdd, *g, &again, &mut memo).ok_or(size(i))?;
        }
        let init = s.clone();
        // the copies end where they went: so may the wave, wherever it does
        let join_ends = self.succ[join].is_empty();
        let mut ended = false;
        let mut at_join: HashMap<usize, Vec<Word>> = HashMap::new();
        let mut at: HashMap<usize, Vec<Word>> = HashMap::from([(wave, s)]);
        let mut work = BTreeSet::from([wave]);
        let (mut written, mut scc_written) = (0 as Regs, false);
        while let Some(l) = work.pop_first() {
            let mut s = at[&l].clone();
            let mut j = l;
            loop {
                let d = &self.insts[j];
                if self.succ[j].is_empty() {
                    if !join_ends {
                        return Err((j, (Rule::SoloEnd, String::new())));
                    }
                    ended = true;
                    break;
                }
                // a compare writes a mask only, v_readfirstlane_b32 a scalar register
                if is_vector(d) && !d.inst.name.starts_with("V_CMP") && d.inst.name != "V_READFIRSTLANE_B32" {
                    for (k, &g) in given.iter().enumerate() {
                        let on = match s[126 + k] {
                            Word::Known(e, _) => bdd.apply(0, e, g).ok_or(size(j))? != 0,
                            Word::Unknown => true,
                        };
                        if on {
                            return Err((j, (Rule::SoloVector, String::new())));
                        }
                    }
                }
                written |= sgpr_use(d).maybe;
                scc_written |= scc_operand(d, false);
                mask_step(bdd, d, j as u32, true, self.cyclic[j], &mut s);
                match self.succ[j][..] {
                    [k] if k == j + 1 && k != join && !self.leader[k] => j = k,
                    _ => {
                        for &k in &self.succ[j] {
                            if k == join {
                                meet(&mut at_join, join, &s);
                            } else if meet(&mut at, k, &s) {
                                work.insert(k);
                            }
                        }
                        break;
                    }
                }
            }
        }
        let Some(end) = at_join.get(&join) else {
            return if ended { Ok(()) } else { Err((i, (Rule::SoloEnd, "never back".into()))) };
        };
        let (live_regs, live_scc) = live[join];
        let kinds = &before[join].as_ref().unwrap().s;
        for w in regs(written & live_regs) {
            let same = match (half(kinds[w]), end[w], init[w]) {
                (Some(k), Word::Known(a, _), Word::Known(b, _)) => {
                    let differs = bdd.apply(2, a, b).ok_or(size(i))?;
                    bdd.apply(0, differs, given[k]).ok_or(size(i))? == 0
                }
                _ => false,
            };
            if !same {
                return Err((i, (Rule::SoloScalar, reg_name(w))));
            }
        }
        if scc_written && live_scc {
            return Err((i, (Rule::SoloScalar, "scc".into())));
        }
        Ok(())
    }
}

/// `lane_guards`' meet: a word that differs where paths meet is unknown. Whether anything
/// changed.
fn meet(state: &mut HashMap<usize, Vec<Word>>, j: usize, s: &[Word]) -> bool {
    let Some(t) = state.get_mut(&j) else {
        state.insert(j, s.to_vec());
        return true;
    };
    let mut changed = false;
    for (a, b) in t.iter_mut().zip(s) {
        if *a != *b && *a != Word::Unknown {
            *a = Word::Unknown;
            changed = true;
        }
    }
    changed
}

/// Whether predicate `a` tests an atom of one of the instructions `of`.
fn uses(bdd: &Bdd, a: u32, of: &BTreeSet<u32>) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![a];
    while let Some(n) = stack.pop() {
        if n < 2 || !seen.insert(n) {
            continue;
        }
        let (v, lo, hi) = bdd.nodes[n as usize];
        if of.contains(&(v / ATOMS_PER_INST)) {
            return true;
        }
        stack.extend([lo, hi]);
    }
    false
}

/// `a` for some value of every atom of the instructions `of` (what it still says once those
/// are made again). None: past the BDD's node limit.
fn exists(bdd: &mut Bdd, a: u32, of: &BTreeSet<u32>, memo: &mut HashMap<u32, u32>) -> Option<u32> {
    if a < 2 {
        return Some(a);
    }
    if let Some(&r) = memo.get(&a) {
        return Some(r);
    }
    let (v, lo, hi) = bdd.nodes[a as usize];
    let (lo, hi) = (exists(bdd, lo, of, memo)?, exists(bdd, hi, of, memo)?);
    let r = if of.contains(&(v / ATOMS_PER_INST)) { bdd.apply(1, lo, hi)? } else { bdd.mk(v, lo, hi) };
    memo.insert(a, r);
    Some(r)
}

fn reg_name(r: usize) -> String {
    match r {
        106 | 107 => "vcc".into(),
        124 => "m0".into(),
        126 | 127 => "exec".into(),
        _ => format!("s{r}"),
    }
}

/// Whether every lane of the wave64 compute program `insts` computes on its own, so that the
/// lane-independent form computes what the console does (the module's claim); else the first
/// instruction, by address, that the proof fails at.
pub(super) fn classify(insts: &[Decoded], opt: &Options) -> Result<Proof, Reject> {
    let whole = |rule| Reject { rule, pc: None, inst: String::new(), detail: String::new() };
    if !matches!(opt.mode, Mode::Compute(_)) || !opt.wave64 || opt.mesh_native.is_some() {
        return Err(whole(Rule::NotWave64Compute));
    }
    if opt.loop_profile || super::pso::material_gate::current().is_some() {
        return Err(whole(Rule::Instrumented));
    }
    #[cfg(test)]
    if ASSUME.get() {
        return Ok(Proof::default());
    }
    prove(insts)
}

/// `classify` for the program alone, whatever its translation is for: what an instrumented
/// translation's program gets without the instrument.
pub(super) fn prove(insts: &[Decoded]) -> Result<Proof, Reject> {
    let whole = |rule| Reject { rule, pc: None, inst: String::new(), detail: String::new() };
    let at = |i: usize, (rule, detail): Fail| Reject { rule, pc: Some(insts[i].pc), inst: rdna2::mnemonic(&insts[i]), detail };
    let (succ, _) = graph(insts, &GuardExits::new());
    if let Some(i) = succ.iter().position(Option::is_none) {
        return Err(at(i, (Rule::IndirectJump, String::new())));
    }
    let succ: Vec<Vec<usize>> = succ.into_iter().map(Option::unwrap).collect();
    let n = insts.len();
    let mut preds = vec![0usize; n];
    for s in &succ {
        for &j in s {
            preds[j] += 1;
        }
    }
    if n > 1 << 24 {
        return Err(whole(Rule::ProofSize));
    }
    let leader = (0..n).map(|j| j == 0 || preds[j] != 1 || succ[j - 1][..] != [j]).collect();
    let cyclic = on_cycles(&succ);
    let full = lane_guards::full_exec(insts, true, true, &GuardExits::new());
    let mut a = Analysis { insts, succ, leader, cyclic, full, words: HashMap::new(), names: (n as u32 + 1) * ATOMS_PER_INST };
    // what every register holds before each instruction, over all paths
    let mut before: Vec<Option<State>> = vec![None; n];
    let mut work = BTreeSet::new();
    if n > 0 {
        before[0] = Some(State::entry());
        work.insert(0);
    }
    while let Some(i) = work.pop_first() {
        let mut st = before[i].clone().unwrap();
        let _ = a.step(i, &mut st);
        for &j in &a.succ[i] {
            let changed = match &mut before[j] {
                Some(t) => t.join(&st),
                slot => {
                    *slot = Some(st.clone());
                    true
                }
            };
            if changed {
                work.insert(j);
            }
        }
    }
    let mut proof = Proof::default();
    for i in 0..n {
        let Some(st) = &before[i] else { continue };
        if let Some(fail) = a.step(i, &mut st.clone()) {
            return Err(at(i, fail));
        }
        proof.first_lane_reads += (insts[i].inst.name == "V_READFIRSTLANE_B32") as usize;
    }
    // memory flagged as written by other threads, loaded where the program has stored or run an
    // atomic before: a lane may be reading what another lane of its wave wrote, which a wave
    // orders and threads do not
    let own = |d: &Decoded, what: &[&str]| is_vector(d) && d.enc.name != "ENC_FLAT_SCRATCH" && what.iter().any(|w| d.inst.name.contains(w));
    let mut after_store = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&i| before[i].is_some() && own(&insts[i], &["STORE", "ATOMIC"])).flat_map(|i| a.succ[i].iter().copied()).collect();
    while let Some(j) = stack.pop() {
        if !std::mem::replace(&mut after_store[j], true) {
            stack.extend(&a.succ[j]);
        }
    }
    let shared = |d: &Decoded| ["GLC", "DLC"].iter().any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(1));
    if let Some(i) = (0..n).find(|&i| before[i].is_some() && after_store[i] && own(&insts[i], &["LOAD"]) && shared(&insts[i])) {
        return Err(at(i, (Rule::SharedLoad, String::new())));
    }
    // the branches on wave-wide mask tests, and what the wave runs past each alone
    let mut bdd = Bdd::new();
    if n > 0 {
        a.words = a.predicates(&mut bdd);
    }
    let live = a.live_in();
    for i in 0..n {
        let Some(st) = &before[i] else { continue };
        if let Some(p) = a.parting(i, st).map_err(|f| at(i, f))? {
            a.solo(&mut bdd, i, &p, &before, &live).map_err(|(j, (rule, detail))| {
                let detail = if j == i { detail } else { format!("+{:#x} {}{}{detail}", insts[j].pc, rdna2::mnemonic(&insts[j]), if detail.is_empty() { "" } else { ": " }) };
                at(i, (rule, detail))
            })?;
            proof.mask_branches += 1;
        }
    }
    Ok(proof)
}
