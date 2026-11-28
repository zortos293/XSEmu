#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Unsupported {
    InvalidSampleCount,
    Msaa,
    Cmask,
    Htile,
    ConvertedFormat,
    DepthStencilPacking,
    PlaneUsage,
    Format,
}

fn color_family(format: u32) -> Option<u32> {
    Some(match format {
        1..=4 => 1,
        9..=14 => 9,
        15..=18 => 15,
        23..=25 => 23,
        27..=32 => 27,
        33..=38 => 33,
        39 | 41..=43 => 39,
        48..=52 => 48,
        53 | 54 | 56..=59 => 53,
        60..=64 => 60,
        70..=72 => 70,
        73..=75 => 73,
        76..=78 => 76,
        79..=81 => 79,
        82..=84 => 82,
        87 | 90 | 91 => 90,
        94..=96 => 94,
        97..=99 => 97,
        26 | 65 | 67 | 85 | 86 | 89 | 115 => format,
        _ => return None,
    })
}

/// Classifies the expanded payload of an XG surface against the actual native
/// resource format, not its view. The caller must establish authoritative
/// ownership and match the plane's block dimensions, row bytes, rows and depth
/// to GetCopyableFootprints. This does not classify compression state or padding.
pub(crate) fn classify(guest_format: u32, host_format: u32, guest_flags: u32, samples: u32, xg_usage: u32) -> Result<u32, Unsupported> {
    use Unsupported::*;

    match samples {
        0 => return Err(InvalidSampleCount),
        1 => {}
        _ => return Err(Msaa),
    }
    match xg_usage {
        2 => return Err(Cmask),
        4 => return Err(Htile),
        1 | 5..=8 => {}
        _ => return Err(PlaneUsage),
    }
    if matches!(guest_format, 116..=120) || (matches!(guest_format, 67 | 86 | 115) && guest_flags & 5 != 0) {
        return Err(ConvertedFormat);
    }

    let ds = guest_flags & 2 != 0;
    if matches!(guest_format, 19..=22 | 44..=47) {
        let plane = match (ds, xg_usage) {
            (false, 5) | (true, 7) => 0,
            (false, 6) | (true, 8) => 1,
            _ => return Err(PlaneUsage),
        };
        return match (guest_format, host_format, plane) {
            (19 | 20, 19 | 20, _) | (44 | 45, 44 | 45, 1) => Ok(plane),
            (44 | 45, 44 | 45, 0) => Err(DepthStencilPacking),
            _ => Err(Format),
        };
    }
    if ds || matches!(guest_format, 40 | 55) {
        if !(xg_usage == 1 || (ds && xg_usage == 7)) {
            return Err(PlaneUsage);
        }
        return match (guest_format, host_format) {
            (39 | 40, 39 | 40) | (53 | 55, 53 | 55) => Ok(0),
            _ => Err(Format),
        };
    }
    if xg_usage != 1 {
        return Err(PlaneUsage);
    }
    let family = color_family(guest_format).ok_or(Format)?;
    if color_family(host_format) == Some(family) && (guest_format == host_format || guest_format == family || host_format == family) {
        Ok(0)
    } else {
        Err(Format)
    }
}

// classes::transfer has no caller until the bridge is wired; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn transfer(guest_format: u32, host_format: u32, guest_flags: u32, samples: u32, xg_usage: u32) -> Result<(u32, u32), Unsupported> {
    if samples == 1 && xg_usage == 1 && guest_flags & 2 == 0 && guest_flags & 5 != 0 {
        match (guest_format, host_format) {
            (86, 28) => return Ok((0, 1)),
            (115, 28) => return Ok((0, 2)),
            (67, 10) => return Ok((0, 3)),
            _ => {}
        }
    }
    classify(guest_format, host_format, guest_flags, samples, xg_usage).map(|plane| (plane, 0))
}

#[cfg(test)]
mod tests {
    use super::{classify, Unsupported::*};

    #[test]
    fn widened_targets_require_the_expanded_single_sample_color_plane() {
        for (format, widened, conversion) in [(86, 28, 1), (115, 28, 2), (67, 10, 3)] {
            for flags in [1, 4, 5] {
                assert_eq!(super::transfer(format, widened, flags, 1, 1), Ok((0, conversion)));
                for (host, f, samples, usage) in [(27, flags, 1, 1), (29, flags, 1, 1), (widened, flags | 2, 1, 1),
                    (widened, flags, 2, 1), (widened, flags, 1, 2), (widened, flags, 1, 7), (widened, 0, 1, 1)] {
                    assert!(super::transfer(format, host, f, samples, usage).is_err());
                }
            }
            assert_eq!(super::transfer(format, format, 0, 1, 1), Ok((0, 0)));
        }
    }

    #[test]
    fn plain_and_block_formats_require_identity_or_a_typeless_endpoint() {
        for (typeless, typed) in [
            (1, 2),
            (9, 14),
            (15, 16),
            (23, 24),
            (27, 29),
            (33, 37),
            (39, 41),
            (48, 50),
            (53, 54),
            (60, 64),
            (70, 71),
            (73, 74),
            (76, 78),
            (79, 81),
            (82, 83),
            (90, 87),
            (90, 91),
            (94, 96),
            (97, 99),
        ] {
            for (g, h) in [(typed, typed), (typeless, typeless), (typed, typeless), (typeless, typed)] {
                assert_eq!(classify(g, h, 0, 1, 1), Ok(0), "{g} -> {h}");
            }
        }
        for (g, h) in [(28, 29), (87, 28), (87, 88), (90, 92), (71, 80), (79, 70), (39, 40), (60, 65), (115, 86)] {
            assert_eq!(classify(g, h, 0, 1, 1), Err(Format), "{g} -> {h}");
        }
    }

    #[test]
    fn translations_in_the_guest_resource_descriptor_are_not_reinterprets() {
        for (g, h) in [(67, 10), (86, 28), (115, 28)] {
            assert_eq!(classify(g, g, 0, 1, 1), Ok(0));
            for flags in [1, 4, 5] {
                assert_eq!(classify(g, h, flags, 1, 1), Err(ConvertedFormat));
                assert_eq!(classify(g, g, flags, 1, 1), Err(ConvertedFormat));
            }
        }
        for g in 116..=120 {
            assert_eq!(classify(g, if g <= 117 { 10 } else { 44 }, 2, 1, 7), Err(ConvertedFormat));
        }
    }

    #[test]
    fn expanded_depth_and_stencil_payloads_use_d3d_plane_not_xg_index() {
        for (flags, depth, stencil) in [(2, 7, 8), (0, 5, 6)] {
            assert_eq!(classify(20, 19, flags, 1, depth), Ok(0));
            assert_eq!(classify(20, 19, flags, 1, stencil), Ok(1));
            assert_eq!(classify(45, 44, flags, 1, depth), Err(DepthStencilPacking));
            assert_eq!(classify(45, 44, flags, 1, stencil), Ok(1));
        }
        for (g, h) in [(40, 39), (55, 53)] {
            for usage in [1, 7] {
                assert_eq!(classify(g, h, 2, 1, usage), Ok(0));
            }
            assert_eq!(classify(g, h, 2, 1, 8), Err(PlaneUsage));
        }
        assert_eq!(classify(45, 44, 2, 1, 5), Err(PlaneUsage));
        assert_eq!(classify(20, 19, 0, 1, 8), Err(PlaneUsage));
        assert_eq!(classify(28, 28, 0, 1, 7), Err(PlaneUsage));
        assert_eq!(classify(21, 21, 0, 1, 5), Err(Format));
        assert_eq!(classify(20, 44, 2, 1, 7), Err(Format));
        assert_eq!(classify(55, 56, 2, 1, 7), Err(Format));
    }

    #[test]
    fn unmodeled_samples_metadata_planes_and_formats_fail_closed() {
        assert_eq!(classify(28, 28, 1, 0, 1), Err(InvalidSampleCount));
        for count in [2, 4, 8, 16, u32::MAX] {
            assert_eq!(classify(28, 28, 1, count, 1), Err(Msaa));
        }
        assert_eq!(classify(28, 28, 1, 1, 2), Err(Cmask));
        assert_eq!(classify(40, 39, 2, 1, 4), Err(Htile));
        for usage in [0, 3, 9, u32::MAX] {
            assert_eq!(classify(28, 28, 0, 1, usage), Err(PlaneUsage));
        }
        for g in [0, 5, 66, 68, 69, 88, 92, 93, 100, 103, 111, 121, u32::MAX] {
            assert_eq!(classify(g, g, 0, 1, 1), Err(Format));
        }
    }
}
