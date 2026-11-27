//! dxil.dll's IDxcValidator for CGBJ's containers. dxc.rs has its own, but it is private
//! and inside the translator fingerprint, so this one is separate.

use crate::gpu::recomp::dxc;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::OnceLock;
use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{IDxcBlob, IDxcUtils, IDxcValidator, CLSID_DxcUtils, CLSID_DxcValidator, DXC_CP_ACP};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

pub enum Failure {
    /// No validator could be made. Not the container's fault, so not cached.
    Unavailable(String),
    /// The validator's message.
    Invalid(String),
}

/// dxil.dll's DxcCreateInstance, from the folder dxc.rs loaded it from.
fn validator_create() -> Result<CreateInstance, String> {
    static CREATE: OnceLock<Result<usize, String>> = OnceLock::new();
    let f = CREATE.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        let dll = dxc::location()?.join("dxil.dll");
        if !dll.is_file() {
            return Err("no dxil.dll beside dxcompiler.dll".into());
        }
        crate::vfs_hooks::with_host_io(|| unsafe {
            let h = LoadLibraryW(crate::util::wide(&dll.to_string_lossy()).as_ptr());
            if h.is_null() {
                return Err(format!("{} does not load", dll.display()));
            }
            GetProcAddress(h, b"DxcCreateInstance\0".as_ptr()).map(|f| f as usize).ok_or_else(|| "dxil.dll has no DxcCreateInstance".to_string())
        })
    });
    f.clone().map(|f| unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

thread_local! {
    static VALIDATOR: RefCell<Option<(IDxcValidator, IDxcUtils)>> = const { RefCell::new(None) };
}

fn validator() -> Result<(IDxcValidator, IDxcUtils), String> {
    VALIDATOR.with(|v| {
        if let Some(x) = v.borrow().as_ref() {
            return Ok(x.clone());
        }
        let validator_create = validator_create()?;
        let utils_create = unsafe { std::mem::transmute::<usize, CreateInstance>(dxc::create_instance().ok_or("dxcompiler.dll not loaded")?) };
        let (mut pv, mut pu) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr = unsafe { validator_create(&CLSID_DxcValidator, &IDxcValidator::IID, &mut pv) };
        if hr.is_err() || pv.is_null() {
            return Err(format!("dxil.dll DxcCreateInstance: {hr:?}"));
        }
        let validator = unsafe { IDxcValidator::from_raw(pv) };
        let hr = unsafe { utils_create(&CLSID_DxcUtils, &IDxcUtils::IID, &mut pu) };
        if hr.is_err() || pu.is_null() {
            return Err(format!("DxcCreateInstance(DxcUtils): {hr:?}"));
        }
        let x = (validator, unsafe { IDxcUtils::from_raw(pu) });
        *v.borrow_mut() = Some(x.clone());
        Ok(x)
    })
}

/// Whether a validator can be made on this thread, before anything is translated for it.
pub fn ready() -> Result<(), String> {
    validator().map(|_| ())
}

/// What dxil.dll's validator says of a container.
pub fn check(dxil: &[u8]) -> Result<(), Failure> {
    let (validator, utils) = validator().map_err(Failure::Unavailable)?;
    let size = u32::try_from(dxil.len()).map_err(|_| Failure::Invalid("container larger than 4 GiB".into()))?;
    let unavailable = |e: windows::core::Error| Failure::Unavailable(format!("{e}"));
    unsafe {
        let blob = utils.CreateBlobFromPinned(dxil.as_ptr() as *const c_void, size, DXC_CP_ACP).map_err(unavailable)?;
        let blob: IDxcBlob = blob.cast().map_err(unavailable)?;
        let r = validator.Validate(&blob, 0).map_err(|e| Failure::Invalid(format!("Validate: {e}")))?;
        let status = r.GetStatus().map_err(unavailable)?;
        if status.is_ok() {
            return Ok(());
        }
        let msg = r
            .GetErrorBuffer()
            .ok()
            .map(|b| {
                let s = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
                String::from_utf8_lossy(s).trim_end_matches('\0').trim().to_string()
            })
            .unwrap_or_default();
        Err(Failure::Invalid(if msg.is_empty() { format!("{status:?}") } else { msg }))
    }
}
