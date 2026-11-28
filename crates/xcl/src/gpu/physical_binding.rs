use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PageKey {
    Physical(u64),
    // Allocation-keyed page pinning has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    Allocation { generation: u64, page: u64 },
}

// Allocation-keyed page pinning has no caller yet; kept for the next fix.
#[allow(dead_code)]
pub(crate) struct PinnedPage {
    pub key: PageKey,
    pub readable: bool,
    pub writable: bool,
    pub backing: Arc<dyn Send + Sync>,
}

#[derive(Default)]
pub(crate) struct BindingIndex(Mutex<HashMap<(u32, u32), (PageKey, Weak<dyn Send + Sync>)>>);

impl BindingIndex {
    pub(crate) fn shared() -> &'static Self {
        static INDEX: LazyLock<BindingIndex> = LazyLock::new(|| BindingIndex(Mutex::default()));
        &INDEX
    }

    pub(crate) fn remember<T: Send + Sync + 'static>(
        &self,
        uav: u32,
        offset: u32,
        key: PageKey,
        object: &Arc<T>,
    ) -> Result<()> {
        if uav >= 1 << 20 || offset % 65536 != 0 {
            return Err("invalid immutable backing location");
        }
        let raw: Arc<dyn Send + Sync> = object.clone();
        let pin = Arc::downgrade(&raw);
        let mut index = self.0.lock().unwrap();
        if let Some((old, live)) = index.get(&(uav, offset)) {
            if live.strong_count() != 0 && (old != &key || !Weak::ptr_eq(live, &pin)) {
                return Err("live raw view changed owner");
            }
        }
        index.insert((uav, offset), (key, pin));
        Ok(())
    }

    // Allocation-keyed page pinning has no caller yet; kept for the next fix.
    #[allow(dead_code)]
    pub(crate) fn pin(
        &self,
        uav: u32,
        offset: u32,
        readable: bool,
        writable: bool,
    ) -> Result<Arc<PinnedPage>> {
        let index = self.0.lock().unwrap();
        let (key, pin) = index
            .get(&(uav, offset))
            .ok_or("captured GPU page not registered")?;
        Ok(Arc::new(PinnedPage {
            key: *key,
            readable,
            writable,
            backing: pin.upgrade().ok_or("captured backing expired")?,
        }))
    }
}
