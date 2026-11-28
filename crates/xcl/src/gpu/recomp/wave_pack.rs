pub(super) fn lane_code(code: &str, upper: bool) -> String {
    let mut out = String::with_capacity(code.len());
    let bytes = code.as_bytes();
    let mut i = 0;
    let mut text = 0;
    while i < bytes.len() {
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            i += 1;
            continue;
        }
        out.push_str(&code[text..i]);
        let begin = i;
        i += 1;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let token = &code[begin..i];
        if let Some(v) = token.strip_prefix("xcl_lower_") {
            out.push_str(v);
        } else {
            let lane_local = token.as_bytes().get(1).is_some_and(u8::is_ascii_digit)
                && (token.starts_with('t') || token.starts_with('v'))
                || token.starts_with("xcl_ov") || token.starts_with("scr_")
                || matches!(token, "xcl_on" | "xcl_op" | "xcl_slot" | "xcl_thread" | "xcl_scratch");
            if upper && lane_local {
                out.push_str("hi_");
            }
            out.push_str(token);
        }
        text = i;
    }
    out.push_str(&code[text..]);
    out
}

pub(super) const EXEC_LO: &str = "and(xcl_slot < xcl_valid_lanes, ((exec_lo >> xcl_slot) & 1u) != 0u)";
pub(super) const EXEC_HI: &str = "and(hi_xcl_slot < xcl_valid_lanes, ((exec_hi >> xcl_slot) & 1u) != 0u)";

pub(super) const PRELUDE: &str = r#"
uint xcl_pack_half(uint value, uint lane) {
    uint v = WaveReadLaneAt(value, lane & 31u);
    return select(lane < xcl_valid_lanes, v, 0u);
}

uint xcl_pack_read(uint lo, uint hi, uint lane) {
    uint l = WaveReadLaneAt(lo, lane & 31u);
    uint h = WaveReadLaneAt(hi, lane & 31u);
    return select(lane < xcl_valid_lanes, select(lane < 32u, l, h), 0u);
}

uint xcl_pack_scatter(uint lo_addr, uint hi_addr, uint lo_data, uint hi_data, uint exec_lo, uint exec_hi, uint offset, uint lane) {
    uint value = 0u;
    [loop] for (uint i = 0u; i < 64u; i++) {
        uint addr = xcl_pack_read(lo_addr, hi_addr, i) + offset;
        uint data = xcl_pack_read(lo_data, hi_data, i);
        if (i < xcl_valid_lanes && lanebit_at(exec_lo, exec_hi, i) && ((addr >> 2u) & 63u) == lane) value = data;
    }
    return value;
}
"#;

pub(super) const LDS: &str = r#"
uint xcl_mesh_lds_load(uint addr) {
    uint word = addr >> 2u;
    if (word < XCL_LDS_WORDS) return xcl_lds[word];
    return 0u;
}

void xcl_mesh_lds_store(uint addr, uint value) {
    uint word = addr >> 2u;
    if (word < XCL_LDS_WORDS) xcl_lds[word] = value;
}
"#;
