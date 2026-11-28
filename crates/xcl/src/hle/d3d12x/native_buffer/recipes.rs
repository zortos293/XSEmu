#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Binding {
    Vertex { slot: u32, bytes: u32, stride: u32 },
    Index { bytes: u32, format: i32 },
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    RootCbv(u32),
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    RootSrv(u32),
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    RootUav(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Access {
    Read,
    Write,
}

// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Run<L> {
    pub(crate) identity: usize,
    pub(crate) native_base: u64,
    pub(crate) native_offset: u64,
    pub(crate) bytes: u64,
    pub(crate) writable: bool,
    pub(crate) lease: L,
}

// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) fn direct_runs<L>(
    cover_bytes: u64,
    access: Access,
    runs: impl IntoIterator<Item = Run<L>>,
) -> Option<Resolved<Vec<L>>> {
    let mut runs = runs.into_iter();
    let first = runs.next()?;
    let (identity, base, offset) = (first.identity, first.native_base, first.native_offset);
    if identity == 0 || base == 0 || cover_bytes == 0 {
        return None;
    }
    let mut covered = 0u64;
    let mut leases = Vec::new();
    let native = base.checked_add(offset)?;
    native.checked_add(cover_bytes)?;
    for r in std::iter::once(first).chain(runs) {
        if r.bytes == 0
            || r.identity != identity
            || r.native_base != base
            || r.native_offset != offset.checked_add(covered)?
            || (access == Access::Write && !r.writable)
        {
            return None;
        }
        covered = covered.checked_add(r.bytes)?;
        leases.push(r.lease);
        if covered >= cover_bytes {
            return Some(Resolved::Canonical {
                native,
                lease: leases,
            });
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Address {
    pub(crate) original: u64,
    pub(crate) binding: Binding,
}

impl Address {
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn access(self) -> Access {
        if matches!(self.binding, Binding::RootUav(_)) {
            Access::Write
        } else {
            Access::Read
        }
    }

    pub(crate) fn known_bytes(self) -> Option<u64> {
        match self.binding {
            Binding::Vertex { bytes, .. } | Binding::Index { bytes, .. } => Some(u64::from(bytes)),
            _ => None,
        }
    }
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn resolve<L>(
        self,
        resolver: &mut impl FnMut(Self) -> Option<Resolved<L>>,
    ) -> Option<(u64, Option<L>)> {
        if self.original == 0 {
            return Some((0, None));
        }
        let (native, lease) = resolver(self)?.binding(self.binding)?;
        Some((native, Some(lease)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayoutError {
    Overflow,
    Invalid,
    Unsupported(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldKind {
    Vertex(u32),
    Index,
    Cbv(u32),
    Srv(u32),
    Uav(u32),
}

#[derive(Clone, Debug)]
// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Layout {
    fields: Vec<(usize, FieldKind)>,
    stride: usize,
}

impl Layout {
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn embeds_addresses(&self) -> bool {
        !self.fields.is_empty()
    }
    pub(crate) fn parse(stride: u32, args: &[[u32; 4]]) -> Result<Self, LayoutError> {
        if stride == 0 || stride % 4 != 0 || args.is_empty() {
            return Err(LayoutError::Invalid);
        }
        let mut offset = 0usize;
        let mut fields = Vec::new();
        for (i, &[kind, parameter, _, words]) in args.iter().enumerate() {
            let last = i + 1 == args.len();
            let action = matches!(kind, 0 | 1 | 2 | 10);
            if kind == 9 {
                return Err(LayoutError::Unsupported(kind));
            }
            if last != action {
                return Err(LayoutError::Invalid);
            }
            let (width, field) = match kind {
                0 => (16, None),
                1 => (20, None),
                2 | 10 => (12, None),
                3 => (16, Some(FieldKind::Vertex(parameter))),
                4 => (16, Some(FieldKind::Index)),
                5 => (
                    (words as usize)
                        .checked_mul(4)
                        .ok_or(LayoutError::Overflow)?,
                    None,
                ),
                6 => (8, Some(FieldKind::Cbv(parameter))),
                7 => (8, Some(FieldKind::Srv(parameter))),
                8 => (8, Some(FieldKind::Uav(parameter))),
                11 => (0, None),
                _ => return Err(LayoutError::Unsupported(kind)),
            };
            if let Some(field) = field {
                fields.push((offset, field));
            }
            offset = offset.checked_add(width).ok_or(LayoutError::Overflow)?;
        }
        if offset > stride as usize {
            return Err(LayoutError::Invalid);
        }
        Ok(Self {
            fields,
            stride: stride as usize,
        })
    }

    pub(crate) fn active_bytes(&self, max: u32, count: Option<u32>) -> Option<usize> {
        let n = max.min(count.unwrap_or(max)) as usize;
        n.checked_mul(self.stride)
    }

    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn records<'a>(
        &self,
        max: u32,
        count: Option<u32>,
        after_writers: &'a [u8],
    ) -> Option<impl Iterator<Item = &'a [u8]>> {
        let needed = self.active_bytes(max, count)?;
        Some(after_writers.get(..needed)?.chunks(self.stride))
    }

    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn materialize<L>(
        &self,
        source: &[u8],
        mut resolve: impl FnMut(Address) -> Option<Resolved<L>>,
    ) -> Option<Materialized<L>> {
        if source.len() != self.stride {
            return None;
        }
        let mut record = source.to_vec();
        let mut leases = Vec::with_capacity(self.fields.len());
        for &(at, kind) in &self.fields {
            let word =
                |o: usize| u32::from_le_bytes(record[at + o..at + o + 4].try_into().unwrap());
            let binding = match kind {
                FieldKind::Vertex(slot) => Binding::Vertex {
                    slot,
                    bytes: word(8),
                    stride: word(12),
                },
                FieldKind::Index => Binding::Index {
                    bytes: word(8),
                    format: word(12) as i32,
                },
                FieldKind::Cbv(p) => Binding::RootCbv(p),
                FieldKind::Srv(p) => Binding::RootSrv(p),
                FieldKind::Uav(p) => Binding::RootUav(p),
            };
            let original = u64::from_le_bytes(record[at..at + 8].try_into().unwrap());
            let (native, lease) = Address { original, binding }.resolve(&mut resolve)?;
            record[at..at + 8].copy_from_slice(&native.to_le_bytes());
            leases.extend(lease);
        }
        Some(Materialized { record, leases })
    }
}

// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Resolved<L> {
    Canonical { native: u64, lease: L },
    Gathered { native: u64, bytes: u32, lease: L },
}

impl<L> Resolved<L> {
    // Native-buffer recipe resolution has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn binding(self, binding: Binding) -> Option<(u64, L)> {
        let (native, lease) = match (self, binding) {
            (Self::Canonical { native, lease }, _) => (native, lease),
            (
                Self::Gathered {
                    native,
                    bytes,
                    lease,
                },
                Binding::Vertex {
                    bytes: requested, ..
                }
                | Binding::Index {
                    bytes: requested, ..
                },
            ) if bytes >= requested => (native, lease),
            _ => return None,
        };
        (native != 0 && (!matches!(binding, Binding::RootCbv(_)) || native & 255 == 0))
            .then_some((native, lease))
    }
}
// Native-buffer recipe resolution has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Materialized<L> {
    pub(crate) record: Vec<u8>,
    pub(crate) leases: Vec<L>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_indirect_uses_signature_order_not_va_alignment_and_keeps_padding() {
        let layout = Layout::parse(
            96,
            &[
                [5, 4, 1, 1],
                [3, 2, 0, 0],
                [4, 0, 0, 0],
                [6, 5, 0, 0],
                [7, 6, 0, 0],
                [8, 7, 0, 0],
                [11, 1, 0, 0],
                [1, 0, 0, 0],
            ],
        )
        .unwrap();
        let mut packed = [0x55; 192];
        let positions = [4, 20, 36, 44, 52];
        for (i, &at) in positions.iter().enumerate() {
            packed[at..at + 8].copy_from_slice(&(0xfff0_0000_0000u64 + i as u64).to_le_bytes());
        }
        packed[12..16].copy_from_slice(&512u32.to_le_bytes());
        packed[16..20].copy_from_slice(&16u32.to_le_bytes());
        packed[28..32].copy_from_slice(&128u32.to_le_bytes());
        packed[32..36].copy_from_slice(&42u32.to_le_bytes());
        assert_eq!(layout.active_bytes(20, Some(1)), Some(96));
        assert_eq!(layout.records(20, Some(0), &[]).unwrap().count(), 0);
        assert!(layout.records(20, Some(2), &packed[..191]).is_none());
        let mut found = Vec::new();
        let rec = layout
            .records(20, Some(1), &packed)
            .unwrap()
            .next()
            .unwrap();
        let result = layout
            .materialize(rec, |addr| {
                found.push(addr.binding);
                Some(Resolved::Canonical {
                    native: (found.len() as u64) * 256,
                    lease: addr.original,
                })
            })
            .unwrap();
        assert_eq!(
            found,
            [
                Binding::Vertex {
                    slot: 2,
                    bytes: 512,
                    stride: 16
                },
                Binding::Index {
                    bytes: 128,
                    format: 42
                },
                Binding::RootCbv(5),
                Binding::RootSrv(6),
                Binding::RootUav(7)
            ]
        );
        assert_eq!(result.leases.len(), 5);
        for i in 0..96 {
            if !positions.iter().any(|&at| (at..at + 8).contains(&i)) {
                assert_eq!(result.record[i], packed[i]);
            }
        }
        assert_eq!(
            u64::from_le_bytes(result.record[36..44].try_into().unwrap()),
            768
        );
    }

    #[test]
    fn refuses_unowned_embedded_address_without_partial_rewrite_or_native_pointer_fallback() {
        let layout = Layout::parse(32, &[[3, 0, 0, 0], [0, 0, 0, 0]]).unwrap();
        let mut data = [0; 32];
        data[..8].copy_from_slice(&8u64.to_le_bytes());
        data[8..12].copy_from_slice(&4096u32.to_le_bytes());
        assert!(layout
            .materialize(&data, |_| None::<Resolved<()>>)
            .is_none());
        assert!(layout
            .materialize(&data, |_| Some(Resolved::Gathered {
                native: 256,
                bytes: 4000,
                lease: ()
            }))
            .is_none());
        let rec = layout
            .materialize(&data, |_| {
                Some(Resolved::Gathered {
                    native: 256,
                    bytes: 4096,
                    lease: (),
                })
            })
            .unwrap();
        assert_eq!(u64::from_le_bytes(rec.record[..8].try_into().unwrap()), 256);
        assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), 8);
        for (kind, binding) in [
            (6, Binding::RootCbv(0)),
            (7, Binding::RootSrv(0)),
            (8, Binding::RootUav(0)),
        ] {
            let lay = Layout::parse(20, &[[kind, 0, 0, 0], [2, 0, 0, 0]]).unwrap();
            assert!(lay
                .materialize(&data[..20], |a| {
                    assert_eq!(a.binding, binding);
                    Some(Resolved::Gathered {
                        native: 256,
                        bytes: u32::MAX,
                        lease: (),
                    })
                })
                .is_none());
        }
        assert!(Layout::parse(8, &[[9, 0, 0, 0]]).is_err());
        assert!(Layout::parse(24, &[[5, 0, 0, 5], [0, 0, 0, 0]]).is_err());
    }

    #[test]
    fn direct_views_require_one_owned_run_not_adjacent_looking_native_addresses() {
        let run = |id, off, writable| Run {
            identity: id,
            native_base: 0x10000,
            native_offset: off,
            bytes: 65536,
            writable,
            lease: off,
        };
        assert!(direct_runs(
            65537,
            Access::Read,
            [run(1, 0, false), run(2, 65536, false)]
        )
        .is_none());
        assert!(direct_runs(
            65537,
            Access::Write,
            [run(1, 0, true), run(1, 65536, false)]
        )
        .is_none());
        assert!(direct_runs(65537, Access::Read, [run(1, 0, false), run(1, 0, false)]).is_none());
        let v = direct_runs(65537, Access::Write, [run(1, 0, true), run(1, 65536, true)]).unwrap();
        assert_eq!(
            v.binding(Binding::RootUav(0)),
            Some((65536, vec![0, 65536]))
        );
    }
}
