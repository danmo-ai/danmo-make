use danmo_mlx_sys as sys;

use crate::error::{check, clear_error, initialize, Result};

pub fn active_memory() -> Result<usize> {
    initialize();
    let mut bytes = 0usize;
    clear_error();
    check(
        unsafe { sys::mlx_get_active_memory(&mut bytes) },
        "mlx_get_active_memory",
    )?;
    Ok(bytes)
}

pub fn peak_memory() -> Result<usize> {
    initialize();
    let mut bytes = 0usize;
    clear_error();
    check(
        unsafe { sys::mlx_get_peak_memory(&mut bytes) },
        "mlx_get_peak_memory",
    )?;
    Ok(bytes)
}

pub fn reset_peak_memory() -> Result<()> {
    initialize();
    clear_error();
    check(
        unsafe { sys::mlx_reset_peak_memory() },
        "mlx_reset_peak_memory",
    )
}

/// Release MLX's cached free buffers back to the allocator (`mx.clear_cache`).
/// Resident arrays are untouched; only the reusable-buffer pool is dropped.
pub fn clear_cache() -> Result<()> {
    initialize();
    clear_error();
    check(unsafe { sys::mlx_clear_cache() }, "mlx_clear_cache")
}
