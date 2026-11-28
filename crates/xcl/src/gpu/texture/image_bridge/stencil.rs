// stencil::native_plane has no caller until the bridge is wired; kept for the next fix.
#[allow(dead_code)]
pub(super) fn native_plane(guest: u32, host: u32, flags: u32, samples: u32, usage: u32) -> Option<u32> {
    let stencil_usage = if flags & 2 == 0 { 6 } else { 8 };
    (guest == 118 && host == 44 && samples == 1 && usage == stencil_usage).then_some(1)
}

#[cfg(test)]
mod tests {
    use super::native_plane;

    #[test]
    fn only_the_present_stencil_payload_maps_to_native_plane_one() {
        for (flags, stencil) in [(0, 6), (2, 8)] {
            for usage in 0..=9 {
                assert_eq!(native_plane(118, 44, flags, 1, usage), (usage == stencil).then_some(1));
            }
            for (guest, host, samples) in [(119, 44, 1), (120, 44, 1), (118, 45, 1), (118, 47, 1), (118, 44, 0), (118, 44, 4)] {
                assert_eq!(native_plane(guest, host, flags, samples, stencil), None);
            }
        }
    }
}
