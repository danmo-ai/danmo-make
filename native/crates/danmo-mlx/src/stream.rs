use std::marker::PhantomData;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::device::Device;
use crate::error::{check, check_infallible, clear_error, initialize, Result};

pub struct Stream {
    pub(crate) raw: sys::mlx_stream,
    _thread_affine: PhantomData<Rc<()>>,
}

impl Stream {
    pub fn new(device: &Device) -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_stream_new_device(device.raw) };
        check_infallible("mlx_stream_new_device", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn cpu() -> Result<Self> {
        Self::new(&Device::cpu()?)
    }

    pub fn gpu() -> Result<Self> {
        Self::new(&Device::gpu()?)
    }

    pub fn synchronize(&self) -> Result<()> {
        clear_error();
        check(unsafe { sys::mlx_synchronize(self.raw) }, "mlx_synchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_stream_free(self.raw);
        }
        clear_error();
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_stream_new() };
        let mut output = check_infallible("mlx_stream_new", raw)
            .map(|raw| Self {
                raw,
                _thread_affine: PhantomData,
            })
            .expect("create MLX stream clone");
        check(
            unsafe { sys::mlx_stream_set(&mut output.raw, self.raw) },
            "mlx_stream_set",
        )
        .expect("copy MLX stream handle");
        output
    }
}
