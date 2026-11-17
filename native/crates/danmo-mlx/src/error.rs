use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Once;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{operation}: {message}")]
pub struct Error {
    operation: &'static str,
    message: String,
}

impl Error {
    pub fn new(operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            operation,
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

extern "C" fn error_handler(message: *const c_char, _data: *mut c_void) {
    if message.is_null() {
        return;
    }
    let message = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    let _ = LAST_ERROR.try_with(|slot| {
        *slot.borrow_mut() = Some(message);
    });
}

pub(crate) fn initialize() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        danmo_mlx_sys::mlx_set_error_handler(Some(error_handler), std::ptr::null_mut(), None);
    });
}

pub(crate) fn clear_error() {
    let _ = LAST_ERROR.try_with(|slot| {
        slot.borrow_mut().take();
    });
}

pub(crate) fn set_error(message: String) {
    let _ = LAST_ERROR.try_with(|slot| {
        *slot.borrow_mut() = Some(message);
    });
}

fn take_error() -> Option<String> {
    LAST_ERROR
        .try_with(|slot| slot.borrow_mut().take())
        .ok()
        .flatten()
}

pub(crate) fn check(status: i32, operation: &'static str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::new(
            operation,
            take_error().unwrap_or_else(|| format!("mlx-c returned status {status}")),
        ))
    }
}

pub(crate) fn check_infallible<T>(operation: &'static str, value: T) -> Result<T> {
    match take_error() {
        Some(message) => Err(Error::new(operation, message)),
        None => Ok(value),
    }
}
