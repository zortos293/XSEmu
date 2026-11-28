use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, LazyLock, Mutex};

#[cfg(test)]
use crate::gpu::physical_binding::BindingIndex;
use crate::gpu::physical_binding::{PageKey, PinnedPage};

// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
const PAGE: u64 = 65536;
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum Location {
    Physical { uav: u32, offset: u32 },
    Uma { uav: u32, offset: u32 },
    Native { guest_page: u64 },
}

// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct CapturedTopology {
    id: u64,
    first: u32,
    cells: Vec<Option<Arc<PinnedPage>>>,
}

#[derive(Clone, Copy, Debug)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) enum PayloadEvidence {
    MetadataFreeRawAtUse,
    MetadataFreeFullTypedOverwriteFromBinary16,
    ExpandedTexelsAfterOrderedDecompression,
    LogicalNativeContentsOnly,
    GuestVisibleCompressionEncoding,
    Unknown,
}

impl CapturedTopology {
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(super) fn resolve(
        id: u64,
        first: u32,
        records: &[[u32; 4]],
        mut resolve: impl FnMut(u32, Location) -> Result<Arc<PinnedPage>>,
    ) -> Result<Self> {
        if id == 0 || records.is_empty() || u64::from(first) + records.len() as u64 > 1 << 32 {
            return Err("invalid completed GPU capture");
        }
        let mut cells = Vec::with_capacity(records.len());
        for (i, &[uav, offset, _, kind]) in records.iter().enumerate() {
            let vpn = first + i as u32;
            let location = match kind {
                0 if uav == 0 && offset == 0 => {
                    cells.push(None);
                    continue;
                }
                1 if (1..1 << 20).contains(&uav) && offset % 65536 == 0 && offset < 128 * 65536 => {
                    Location::Physical { uav, offset }
                }
                2 if offset == vpn << 16 => Location::Uma { uav, offset },
                3 if uav == 0 && offset == 0 => Location::Native {
                    guest_page: u64::from(vpn) << 16,
                },
                _ => return Err("unrecognized captured owner"),
            };
            cells.push(Some(resolve(vpn, location)?));
        }
        Ok(Self { id, first, cells })
    }

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn capture_id(&self) -> u64 {
        self.id
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn first_vpn(&self) -> u32 {
        self.first
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.cells.len()
    }

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    fn span(&self, request: &mut Request, mut lo: u64, hi: u64) -> Result<()> {
        if lo >= hi || hi > self.cells.len() as u64 * PAGE {
            return Err("image or raw range outside capture");
        }
        while lo < hi {
            let i = lo / PAGE;
            let next = hi.min((i + 1) * PAGE);
            if let Some(page) = &self.cells[i as usize] {
                if !page.readable || (request.write && !page.writable) {
                    return Err("captured backing lacks required access");
                }
                request.claims.push(Claim {
                    page: Arc::clone(page),
                    vpn: self.first + i as u32,
                    lo: (lo % PAGE) as u32,
                    hi: (next - i * PAGE) as u32,
                });
            }
            lo = next;
        }
        Ok(())
    }

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn raw_request(&self, byte_offset: u64, bytes: u64, write: bool) -> Result<Request> {
        let mut r = Request {
            image: None,
            host: None,
            write,
            claims: Vec::new(),
        };
        self.span(
            &mut r,
            byte_offset,
            byte_offset.checked_add(bytes).ok_or("raw range overflow")?,
        )?;
        Ok(r)
    }

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(super) fn image_request(
        &self,
        native: usize,
        sub: u32,
        w: &[u32; 68],
        write: bool,
        evidence: PayloadEvidence,
        host: HostBacking,
    ) -> Result<Request> {
        match evidence {
            PayloadEvidence::MetadataFreeRawAtUse if write && w[19] == 3 => {
                return Err("shared-exponent writeback needs a full typed overwrite")
            }
            PayloadEvidence::MetadataFreeRawAtUse
            | PayloadEvidence::MetadataFreeFullTypedOverwriteFromBinary16 => {}
            PayloadEvidence::ExpandedTexelsAfterOrderedDecompression if !write => {}
            PayloadEvidence::ExpandedTexelsAfterOrderedDecompression => {
                return Err("native writes require guest-valid metadata regeneration")
            }
            PayloadEvidence::LogicalNativeContentsOnly => {
                return Err("native logical contents do not establish guest physical bytes")
            }
            PayloadEvidence::GuestVisibleCompressionEncoding => {
                return Err("compressed guest representation is not implemented")
            }
            PayloadEvidence::Unknown => {
                return Err("no evidence of guest payload representation at this use")
            }
        }
        if native == 0
            || w[..3].contains(&0)
            || w[6..10].iter().any(|&n| n > 16)
            || w[10] == 0
            || w[14] >= PAGE as u32
        {
            return Err("invalid native image layout");
        }
        let mut r = Request {
            image: Some((self.id, native, sub)),
            host: Some(host),
            write,
            claims: Vec::new(),
        };
        let (bw, bh, bd, block) = (1u64 << w[6], 1u64 << w[7], 1u64 << w[8], 1u64 << w[9]);
        let x0 = u64::from(w[4]) / bw;
        let x1 = (u64::from(w[4]) + u64::from(w[0])).div_ceil(bw);
        let y0 = u64::from(w[5]) / bh;
        let y1 = (u64::from(w[5]) + u64::from(w[1])).div_ceil(bh);
        let base = u64::from(w[15]) * PAGE + u64::from(w[14]);
        for layer in 0..u64::from(w[2]).div_ceil(bd) {
            let at = base
                .checked_add(layer * u64::from(w[12]))
                .ok_or("layer address overflow")?;
            if x1 - x0 == u64::from(w[10]) {
                let row = |y: u64| {
                    y.checked_mul(u64::from(w[10]))
                        .and_then(|n| n.checked_add(x0))
                        .and_then(|n| n.checked_mul(block))
                        .and_then(|n| n.checked_add(at))
                        .ok_or("tile address overflow")
                };
                self.span(&mut r, row(y0)?, row(y1)?)?;
                continue;
            }
            for y in y0..y1 {
                let row = y
                    .checked_mul(u64::from(w[10]))
                    .ok_or("row address overflow")?;
                let address = |x| {
                    row.checked_add(x)
                        .and_then(|n| n.checked_mul(block))
                        .and_then(|n| at.checked_add(n))
                        .ok_or("tile address overflow")
                };
                self.span(&mut r, address(x0)?, address(x1)?)?;
            }
        }
        r.compact();
        Ok(r)
    }
}

#[derive(Clone)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
struct Claim {
    page: Arc<PinnedPage>,
    vpn: u32,
    lo: u32,
    hi: u32,
}

// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Request {
    image: Option<(u64, usize, u32)>,
    host: Option<HostBacking>,
    write: bool,
    claims: Vec<Claim>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct HostBacking {
    kind: u64,
    key: u64,
    start: u64,
    end: u64,
}

impl HostBacking {
    pub(crate) const PRIVATE: Self = Self {
        kind: 0,
        key: 0,
        start: 0,
        end: 0,
    };
    pub(crate) const TILED: Self = Self {
        kind: 1,
        key: 0,
        start: 0,
        end: u64::MAX,
    };
    pub(crate) fn arena(key: u64, start: usize, end: usize) -> Self {
        Self {
            kind: 2,
            key,
            start: start as u64,
            end: end as u64,
        }
    }
    pub(crate) fn slab(heap: usize, start: usize, end: usize) -> Self {
        Self {
            kind: 3,
            key: heap as u64,
            start: start as u64,
            end: end as u64,
        }
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    fn overlaps(self, b: Self) -> bool {
        self.kind != 0
            && self.kind == b.kind
            && self.key == b.key
            && self.start < b.end
            && b.start < self.end
    }
}

impl Request {
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    fn compact(&mut self) {
        self.claims.sort_unstable_by_key(|c| (c.vpn, c.lo));
        self.claims.dedup_by(|b, a| {
            if a.vpn == b.vpn && b.lo <= a.hi {
                a.hi = a.hi.max(b.hi);
                true
            } else {
                false
            }
        });
    }
}

#[derive(Clone, Copy)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
struct Held {
    lo: u32,
    hi: u32,
    vpn: u32,
    write: bool,
    image: Option<(u64, usize)>,
    id: u64,
}

impl Held {
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    fn conflicts(self, other: Self) -> bool {
        (self.write || other.write) && self.lo < other.hi && other.lo < self.hi
    }
}

// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
struct Lease {
    keys: Vec<PageKey>,
    native: BTreeSet<usize>,
    hosts: Vec<HostBacking>,
    _pins: Vec<Arc<PinnedPage>>,
}

#[derive(Default)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
struct State {
    next: u64,
    by_page: HashMap<PageKey, Vec<Held>>,
    native: BTreeSet<usize>,
    live: HashMap<u64, Lease>,
}

#[derive(Clone)]
// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct Ownership(Arc<Mutex<State>>);

// Image-bridge page ownership is not consulted yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct WritePermit {
    book: Ownership,
    id: u64,
    subs: BTreeSet<(u64, usize, u32)>,
    read_subs: BTreeSet<(u64, usize, u32)>,
}

impl Ownership {
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn shared() -> &'static Self {
        static BOOK: LazyLock<Ownership> = LazyLock::new(|| Ownership(Arc::default()));
        &BOOK
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn admit(&self, requests: &[Request]) -> Result<WritePermit> {
        self.try_admit(requests)?.ok_or("native or physical bytes are still in use")
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn try_admit(&self, requests: &[Request]) -> Result<Option<WritePermit>> {
        if requests.is_empty() {
            return Err("no declared accesses");
        }
        let mut planned: HashMap<PageKey, Vec<Held>> = HashMap::new();
        let mut pins = HashMap::new();
        let mut subs = BTreeSet::new();
        let mut read_subs = BTreeSet::new();
        let mut native = BTreeSet::new();
        let mut hosts: HashMap<usize, HostBacking> = HashMap::new();
        for r in requests {
            if let Some(i) = r.image {
                read_subs.insert(i);
                native.insert(i.1);
                let host = r.host.ok_or("native backing identity required")?;
                if hosts.iter().any(|(&n, &h)| {
                    if n == i.1 {
                        h != host
                    } else {
                        h.overlaps(host)
                    }
                }) {
                    return Err("different native images require the same host bytes");
                }
                hosts.insert(i.1, host);
            }
            if let Some(i) = r.image.filter(|_| r.write) {
                subs.insert(i);
            }
            for c in &r.claims {
                let h = Held {
                    lo: c.lo,
                    hi: c.hi,
                    vpn: c.vpn,
                    write: r.write,
                    image: r.image.map(|(id, n, _)| (id, n)),
                    id: 0,
                };
                let spans = planned.entry(c.page.key).or_default();
                if spans.iter().any(|&s| {
                    h.conflicts(s) && (h.image.is_none() || h.image != s.image || h.vpn != s.vpn)
                }) {
                    return Err("simultaneous physical aliases need one canonical producer");
                }
                spans.push(h);
                pins.entry(Arc::as_ptr(&c.page) as usize)
                    .or_insert_with(|| Arc::clone(&c.page));
            }
        }
        let mut state = self.0.lock().unwrap();
        if !state.native.is_disjoint(&native) {
            return Ok(None);
        }
        if state.live.values().any(|lease| {
            lease
                .hosts
                .iter()
                .any(|&h| hosts.values().any(|&b| h.overlaps(b)))
        }) {
            return Ok(None);
        }
        for (key, spans) in &planned {
            if let Some(held) = state.by_page.get(key) {
                if held.iter().any(|&s| spans.iter().any(|h| h.conflicts(s))) {
                    return Ok(None);
                }
            }
        }
        let id = state
            .next
            .checked_add(1)
            .ok_or("ownership sequence exhausted")?;
        state.next = id;
        state.native.extend(native.iter().copied());
        let keys = planned.keys().copied().collect();
        for (key, mut spans) in planned {
            for h in &mut spans {
                h.id = id;
            }
            state.by_page.entry(key).or_default().extend(spans);
        }
        state.live.insert(
            id,
            Lease {
                keys,
                native,
                hosts: hosts.into_values().filter(|h| h.kind != 0).collect(),
                _pins: pins.into_values().collect(),
            },
        );
        Ok(Some(WritePermit {
            book: self.clone(),
            id,
            subs,
            read_subs,
        }))
    }
}

impl WritePermit {
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn can_read(&self, capture_id: u64, native_id: usize, native_sub: u32) -> bool {
        self.read_subs
            .contains(&(capture_id, native_id, native_sub))
    }
    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn allows(&self, capture_id: u64, native_id: usize, native_sub: u32) -> bool {
        self.subs.contains(&(capture_id, native_id, native_sub))
    }

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    fn release(self) {
        let mut state = self.book.0.lock().unwrap();
        let lease = state.live.remove(&self.id).expect("live ownership permit");
        for n in &lease.native {
            state.native.remove(n);
        }
        for key in &lease.keys {
            let held = state.by_page.get_mut(key).unwrap();
            held.retain(|h| h.id != self.id);
            if held.is_empty() {
                state.by_page.remove(key);
            }
        }
        drop(state);
        drop(lease);
    }
}

#[cfg(windows)]
pub(crate) use retirement::Pending;
#[cfg(windows)]
mod retirement {
    use super::WritePermit;
    use windows::Win32::Graphics::Direct3D12::*;

    // Image-bridge page ownership is not consulted yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) struct Pending {
        uses: Option<WritePermit>,
        fences: Vec<ID3D12Fence>,
    }

    impl WritePermit {
        /// Each queue must have received its last work for this acquisition.
        // Image-bridge page ownership is not consulted yet; kept for the next fix.
        #[allow(dead_code)]
        pub(crate) unsafe fn signal_after_last_use(
            self,
            queues: &[ID3D12CommandQueue],
        ) -> Result<Pending, String> {
            if queues.is_empty() {
                return Err("no consumer queue for physical ownership".into());
            }
            let mut fences = Vec::with_capacity(queues.len());
            for q in queues {
                let mut device: Option<ID3D12Device> = None;
                q.GetDevice(&mut device).map_err(|e| e.to_string())?;
                let fence: ID3D12Fence = device
                    .ok_or("no device")?
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|e| e.to_string())?;
                q.Signal(&fence, 1).map_err(|e| e.to_string())?;
                fences.push(fence);
            }
            Ok(Pending {
                uses: Some(self),
                fences,
            })
        }
    }

    impl Pending {
        // Image-bridge page ownership is not consulted yet; kept for the next fix.
        #[allow(dead_code)]
        pub(crate) fn retire_if_complete(&mut self) -> Result<bool, &'static str> {
            for fence in &self.fences {
                match unsafe { fence.GetCompletedValue() } {
                    u64::MAX => return Err("consumer device removed"),
                    0 => return Ok(false),
                    _ => {}
                }
            }
            if let Some(permit) = self.uses.take() {
                permit.release();
            }
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PayloadEvidence::MetadataFreeRawAtUse as Raw;
    fn book() -> Ownership {
        Ownership(Arc::default())
    }
    fn captured(id: u64, pfns: &[u64], read: bool, write: bool) -> CapturedTopology {
        let records: Vec<_> = (0..pfns.len() as u32).map(|i| [10 + i, 0, 0, 1]).collect();
        CapturedTopology::resolve(id, 0, &records, |i, _| {
            Ok(Arc::new(PinnedPage {
                key: PageKey::Physical(pfns[i as usize]),
                readable: read,
                writable: write,
                backing: Arc::new(()),
            }))
        })
        .unwrap()
    }
    fn words(page: u32, subpage: u32, blocks: u32) -> [u32; 68] {
        let mut w = [0; 68];
        w[0] = blocks * 256;
        w[1] = 1;
        w[2] = 1;
        w[6] = 8;
        w[9] = 8;
        w[10] = blocks;
        w[15] = page;
        w[14] = subpage;
        w
    }
    #[test]
    fn exact_capture_and_subresource_grants_conflict_by_pfn_not_uav_or_va() {
        let book = book();
        let a = captured(7, &[13, 19], true, true);
        let b = captured(8, &[19, 99], true, true);
        let w = a
            .image_request(10, 3, &words(1, 512, 1), true, Raw, HostBacking::PRIVATE)
            .unwrap();
        let p = book.admit(&[w]).unwrap();
        assert!(
            p.allows(7, 10, 3) && !p.allows(8, 10, 3) && !p.allows(7, 11, 3) && !p.allows(7, 10, 4)
        );
        assert!(book
            .admit(&[b.raw_request(511, 2, false).unwrap()])
            .is_err());
        book.admit(&[b.raw_request(768, 1, true).unwrap()])
            .unwrap()
            .release();
        p.release();
        book.admit(&[b.raw_request(512, 256, true).unwrap()])
            .unwrap()
            .release();
    }
    #[test]
    fn duplicate_physical_page_in_image_rejects_writer_but_not_readers() {
        let a = captured(77, &[3, 3], true, true);
        assert!(book()
            .admit(&[a
                .image_request(5, 2, &words(0, 0, 512), true, Raw, HostBacking::PRIVATE)
                .unwrap()])
            .is_err());
        let p = book()
            .admit(&[a
                .image_request(5, 2, &words(0, 0, 512), false, Raw, HostBacking::PRIVATE)
                .unwrap()])
            .unwrap();
        assert!(!p.allows(77, 5, 2));
        assert!(p.can_read(77, 5, 2));
        p.release();
    }
    #[test]
    fn two_logical_reads_cannot_materialize_into_one_native_image_concurrently() {
        let a = captured(1, &[7], true, false);
        let b = book();
        let req = |native, sub| {
            a.image_request(
                native,
                sub,
                &words(0, 0, 1),
                false,
                Raw,
                HostBacking::PRIVATE,
            )
            .unwrap()
        };
        let first = b.admit(&[req(42, 0)]).unwrap();
        assert!(b.admit(&[req(42, 1)]).is_err());
        b.admit(&[req(43, 0)]).unwrap().release();
        first.release();
        b.admit(&[req(42, 1)]).unwrap().release();
    }
    #[test]
    fn batch_is_atomic_and_packed_mips_share_only_the_same_native_capture() {
        let a = captured(3, &[5], true, true);
        let r = |native, sub| {
            a.image_request(
                native,
                sub,
                &words(0, 0, 1),
                true,
                Raw,
                HostBacking::PRIVATE,
            )
            .unwrap()
        };
        let book = book();
        assert!(book.admit(&[r(90, 0), r(91, 1)]).is_err());
        assert!(book
            .admit(&[r(90, 0), a.raw_request(0, 5, false).unwrap()])
            .is_err());
        let p = book.admit(&[r(90, 0), r(90, 1)]).unwrap();
        assert!(p.allows(3, 90, 0) && p.allows(3, 90, 1));
        p.release();
    }
    #[test]
    fn rights_come_from_the_actual_pinned_owner() {
        let ro = captured(1, &[7], true, false);
        assert!(ro
            .image_request(9, 0, &words(0, 0, 1), true, Raw, HostBacking::PRIVATE)
            .is_err());
        let denied = captured(2, &[8], false, true);
        assert!(denied.raw_request(0, 1, false).is_err());
        assert!(ro.raw_request(65535, 2, false).is_err());
        assert!(CapturedTopology::resolve(1, 0, &[[0, 1, 0, 9]], |_, _| Err("unbound")).is_err());
        assert!(CapturedTopology::resolve(1, 0, &[[10, 0, 0, 1]], |_, _| Err("unbound")).is_err());
    }
    #[test]
    fn losing_ticket_never_releases_an_inflight_claim() {
        let a = captured(1, &[3], true, true);
        let s = book();
        drop(s.admit(&[a.raw_request(0, 128, true).unwrap()]).unwrap());
        assert!(s.admit(&[a.raw_request(1, 1, false).unwrap()]).is_err());
    }

    #[test]
    fn native_heap_alias_and_shared_exponent_granularity_are_checked_independently() {
        let a = captured(1, &[5, 9], true, true);
        let b = book();
        let r = |native, pg, host| {
            a.image_request(native, 0, &words(pg, 0, 1), false, Raw, host)
                .unwrap()
        };
        assert!(b
            .admit(&[r(4, 0, HostBacking::TILED), r(8, 1, HostBacking::TILED)])
            .is_err());
        let p = b
            .admit(&[r(4, 0, HostBacking::arena(2, 4096, 8192))])
            .unwrap();
        assert!(b
            .admit(&[r(8, 1, HostBacking::arena(2, 6144, 8192))])
            .is_err());
        b.admit(&[r(8, 1, HostBacking::arena(2, 8192, 12288))])
            .unwrap()
            .release();
        b.admit(&[r(8, 1, HostBacking::slab(2, 4096, 8192))])
            .unwrap()
            .release();
        p.release();
        let mut w = words(0, 0, 1);
        w[19] = 3;
        assert!(a
            .image_request(8, 0, &w, false, Raw, HostBacking::PRIVATE)
            .is_ok());
        assert!(a
            .image_request(8, 0, &w, true, Raw, HostBacking::PRIVATE)
            .is_err());
        assert!(a
            .image_request(
                8,
                0,
                &w,
                true,
                PayloadEvidence::MetadataFreeFullTypedOverwriteFromBinary16,
                HostBacking::PRIVATE
            )
            .is_ok());
    }

    #[test]
    fn logical_clear_and_compressed_guest_bytes_never_certify_raw_access() {
        use PayloadEvidence::*;
        let a = captured(4, &[9], true, true);
        for e in [
            Unknown,
            LogicalNativeContentsOnly,
            GuestVisibleCompressionEncoding,
        ] {
            assert!(a
                .image_request(42, 0, &words(0, 0, 1), false, e, HostBacking::PRIVATE)
                .is_err());
        }
        assert!(a
            .image_request(
                42,
                0,
                &words(0, 0, 1),
                false,
                ExpandedTexelsAfterOrderedDecompression,
                HostBacking::PRIVATE
            )
            .is_ok());
        assert!(a
            .image_request(
                42,
                0,
                &words(0, 0, 1),
                true,
                ExpandedTexelsAfterOrderedDecompression,
                HostBacking::PRIVATE
            )
            .is_err());
    }

    #[test]
    fn captured_descriptor_reverse_lookup_uses_retained_physical_token_not_current_va() {
        let index = BindingIndex::default();
        let old = Arc::new(());
        let new = Arc::new(());
        index.remember(5, 0, PageKey::Physical(100), &old).unwrap();
        index.remember(6, 0, PageKey::Physical(200), &new).unwrap();
        let a =
            CapturedTopology::resolve(93, 0, &[[5, 0, 0, 1], [6, 0, 0, 1]], |_, loc| match loc {
                Location::Physical { uav, offset } => index.pin(uav, offset, true, true),
                _ => Err("unexpected fallback"),
            })
            .unwrap();
        drop(old);
        assert!(index.remember(5, 0, PageKey::Physical(200), &new).is_err());
        let b = book();
        let p = b
            .admit(&[a
                .image_request(8, 4, &words(0, 0, 1), true, Raw, HostBacking::PRIVATE)
                .unwrap()])
            .unwrap();
        drop(a);
        assert!(index.remember(5, 0, PageKey::Physical(200), &new).is_err());
        p.release();
        index.remember(5, 0, PageKey::Physical(200), &new).unwrap();
    }
}
