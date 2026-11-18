//! One long-lived MLX executor for this process.
//!
//! Families load resident weights through it. They do not spawn their own
//! device thread. Arrays stay on this thread; callers only pass paths and
//! plain results.

use std::sync::OnceLock;

use danmo_mlx::{DeviceType, Error, Executor, Result};

static EXECUTOR: OnceLock<Executor> = OnceLock::new();

pub fn executor() -> Result<&'static Executor> {
    if let Some(executor) = EXECUTOR.get() {
        return Ok(executor);
    }
    let created = Executor::spawn(device_type())?;
    match EXECUTOR.set(created) {
        Ok(()) => {}
        Err(extra) => drop(extra),
    }
    EXECUTOR
        .get()
        .ok_or_else(|| Error::new("MLX executor", "process executor is missing after spawn"))
}

pub fn set_budget(bytes: usize) -> Result<()> {
    executor()?.set_budget(bytes)
}

fn device_type() -> DeviceType {
    if cfg!(any(feature = "mlx-metal", feature = "mlx-cuda")) {
        DeviceType::Gpu
    } else {
        DeviceType::Cpu
    }
}
