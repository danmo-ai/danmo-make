use std::marker::PhantomData;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::error::{check, check_infallible, clear_error, initialize, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Gpu,
}

pub struct Device {
    pub(crate) raw: sys::mlx_device,
    _thread_affine: PhantomData<Rc<()>>,
}

impl Device {
    pub fn cpu() -> Result<Self> {
        Self::new(DeviceType::Cpu, 0)
    }

    pub fn gpu() -> Result<Self> {
        Self::new(DeviceType::Gpu, 0)
    }

    pub fn new(kind: DeviceType, index: i32) -> Result<Self> {
        initialize();
        clear_error();
        let kind = match kind {
            DeviceType::Cpu => sys::mlx_device_type__MLX_CPU,
            DeviceType::Gpu => sys::mlx_device_type__MLX_GPU,
        };
        let raw = unsafe { sys::mlx_device_new_type(kind, index) };
        check_infallible("mlx_device_new_type", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn is_available(&self) -> Result<bool> {
        clear_error();
        let mut available = false;
        check(
            unsafe { sys::mlx_device_is_available(&mut available, self.raw) },
            "mlx_device_is_available",
        )?;
        Ok(available)
    }

    pub fn kind(&self) -> Result<DeviceType> {
        clear_error();
        let mut kind = sys::mlx_device_type__MLX_CPU;
        check(
            unsafe { sys::mlx_device_get_type(&mut kind, self.raw) },
            "mlx_device_get_type",
        )?;
        match kind {
            value if value == sys::mlx_device_type__MLX_CPU => Ok(DeviceType::Cpu),
            value if value == sys::mlx_device_type__MLX_GPU => Ok(DeviceType::Gpu),
            _ => Err(crate::Error::new(
                "mlx_device_get_type",
                format!("unknown device type {kind}"),
            )),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_device_free(self.raw);
        }
        clear_error();
    }
}
