//! Build-time and run-time capability report for the MLX backend.
//!
//! The build decides whether MLX compiled its M5 tensor-unit (NAX) kernels; the
//! process decides whether this machine can run them. Both answers are needed
//! before a caller picks a kernel path, and both are visible: a missing NAX path
//! is a performance cliff, not a correctness bug, so it is reported rather than
//! silently absorbed.

use std::ffi::CString;
use std::os::raw::c_char;

use danmo_mlx_sys as sys;

use crate::device::Device;
use crate::error::{check_infallible, clear_error, initialize, take_error, Error, Result};

const NAX_DEPLOYMENT_TARGET: &str = "26.2";

/// The macOS deployment target MLX was configured with at build time.
///
/// `unknown` when the artifact does not target macOS (CUDA, or a CPU-only
/// build). `danmo-mlx-sys` reports the value as `links` metadata and the
/// `danmo-mlx` build script forwards it as `rustc-env` (`DEP_MLXC_*` is only
/// visible inside build scripts).
pub const DEPLOYMENT_TARGET: &str = match option_env!("DANMO_MLX_DEPLOYMENT_TARGET") {
    Some(target) => target,
    None => "unknown",
};

/// The macOS deployment target MLX was configured with at build time.
pub fn deployment_target() -> &'static str {
    DEPLOYMENT_TARGET
}

/// Whether the linked MLX was built with NAX kernels.
///
/// `false` means this artifact cannot drive the M5 tensor units at all,
/// whatever the host machine supports. The build script warns when it falls
/// back to a pre-26.2 target.
pub fn nax_built() -> bool {
    matches!(option_env!("DANMO_MLX_NAX_BUILT"), Some("ON"))
}

/// GPU architecture string reported by MLX, for example `applegpu_g16s`.
pub fn gpu_architecture() -> Result<String> {
    let device = Device::gpu()?;
    Ok(device_info_string(&device, "architecture")?.unwrap_or_default())
}

/// Human-readable GPU name, for example `Apple M5 Max`.
pub fn gpu_name() -> Result<String> {
    let device = Device::gpu()?;
    Ok(device_info_string(&device, "device_name")?.unwrap_or_default())
}

fn device_info_string(device: &Device, key: &str) -> Result<Option<String>> {
    initialize();
    clear_error();
    let mut info = check_infallible("mlx_device_info_new", unsafe { sys::mlx_device_info_new() })?;
    if info.ctx.is_null() {
        return Err(Error::new(
            "mlx_device_info_new",
            "MLX returned an empty device info",
        ));
    }
    clear_error();
    let status = unsafe { sys::mlx_device_info_get(&mut info, device.raw) };
    if status != 0 {
        let message =
            take_error().unwrap_or_else(|| format!("mlx_device_info_get returned status {status}"));
        clear_error();
        unsafe {
            let _ = sys::mlx_device_info_free(info);
        }
        return Err(Error::new("mlx_device_info_get", message));
    }
    let key = CString::new(key)
        .map_err(|_| Error::new("mlx_device_info_get_string", "key has a null byte"))?;
    clear_error();
    let mut value: *const c_char = std::ptr::null();
    let status = unsafe { sys::mlx_device_info_get_string(&mut value, info, key.as_ptr()) };
    let outcome = match status {
        // 2 is "key not found or wrong type": an absence, not a failure.
        2 => Ok(None),
        0 if !value.is_null() => Ok(Some(
            unsafe { std::ffi::CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        )),
        _ => Err(Error::new(
            "mlx_device_info_get_string",
            take_error().unwrap_or_else(|| format!("returned status {status}")),
        )),
    };
    clear_error();
    unsafe {
        let _ = sys::mlx_device_info_free(info);
    }
    clear_error();
    outcome
}

/// Parsed GPU generation from the architecture string.
///
/// MLX reads the two digits before the trailing part class: `applegpu_g16s` is
/// generation 16, `applegpu_g17p` is 17.
pub fn gpu_generation(architecture: &str) -> Option<u32> {
    let mut digits: Vec<char> = Vec::new();
    for character in architecture.chars().rev().skip(1) {
        if character.is_ascii_digit() {
            digits.push(character);
        } else {
            break;
        }
    }
    if digits.is_empty() {
        return None;
    }
    digits.iter().rev().collect::<String>().parse().ok()
}

/// Whether this machine can run NAX kernels: the artifact carries them, the OS
/// is new enough, and the GPU generation qualifies.
pub fn nax_available() -> Result<bool> {
    if !nax_built() {
        return Ok(false);
    }
    if !os_supports_nax() {
        return Ok(false);
    }
    let architecture = gpu_architecture()?;
    let generation = match gpu_generation(&architecture) {
        Some(generation) => generation,
        // Unknown architecture: report unavailable instead of guessing, so the
        // caller keeps its correct generic path.
        None => return Ok(false),
    };
    let floor = if architecture.ends_with('p') { 18 } else { 17 };
    Ok(generation >= floor)
}

fn os_supports_nax() -> bool {
    match host_macos_version() {
        Some(version) => version_at_least(&version, NAX_DEPLOYMENT_TARGET),
        None => false,
    }
}

fn host_macos_version() -> Option<String> {
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version.is_empty() {
        None
    } else {
        Some(version)
    }
}

fn version_at_least(actual: &str, minimum: &str) -> bool {
    let parse = |value: &str| -> Vec<u32> {
        value
            .split('.')
            .map(|part| part.parse::<u32>().unwrap_or(0))
            .collect()
    };
    let actual = parse(actual);
    let minimum = parse(minimum);
    for index in 0..minimum.len().max(actual.len()) {
        let left = actual.get(index).copied().unwrap_or(0);
        let right = minimum.get(index).copied().unwrap_or(0);
        if left != right {
            return left > right;
        }
    }
    true
}

/// One-line summary for the health payload and the `[backend]` task log line.
pub fn summary() -> String {
    let architecture = gpu_architecture().unwrap_or_default();
    let available = nax_available().unwrap_or(false);
    format!(
        "mlx deployment_target={DEPLOYMENT_TARGET} nax_built={} nax_available={available} architecture={architecture}",
        nax_built(),
    )
}

/// Process-wide cached [`summary`]. Computing it probes the GPU and reads the
/// host OS version, so health checks and per-task log lines must share one
/// evaluation instead of paying it per call.
pub fn cached_summary() -> &'static str {
    static SUMMARY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SUMMARY.get_or_init(summary).as_str()
}

/// Fail when a caller requires NAX and this artifact or machine cannot supply
/// it. Used at startup, never per task.
pub fn require_nax() -> Result<()> {
    if nax_available()? {
        return Ok(());
    }
    let reason = if !nax_built() {
        format!(
            "this build has no NAX kernels (deployment target {DEPLOYMENT_TARGET}, needs {NAX_DEPLOYMENT_TARGET})"
        )
    } else if !os_supports_nax() {
        "the running macOS is older than 26.2".to_string()
    } else {
        "the GPU generation is below the NAX floor".to_string()
    };
    Err(Error::new("nax", reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_parses_applegpu_names() {
        assert_eq!(gpu_generation("applegpu_g16s"), Some(16));
        assert_eq!(gpu_generation("applegpu_g17p"), Some(17));
        assert_eq!(gpu_generation("applegpu_g13g"), Some(13));
        assert_eq!(gpu_generation("applegpu"), None);
    }

    #[test]
    fn version_compare_handles_uneven_lengths() {
        assert!(version_at_least("27.2", "26.2"));
        assert!(version_at_least("26.2", "26.2"));
        assert!(!version_at_least("26.1", "26.2"));
        assert!(version_at_least("26.2.1", "26.2"));
    }

    #[test]
    fn summary_reports_the_build_flags() {
        let summary = summary();
        assert!(summary.contains("nax_built="));
        assert!(summary.contains("deployment_target="));
    }

    #[test]
    fn cached_summary_is_stable_and_complete() {
        let first = cached_summary();
        let second = cached_summary();
        assert!(std::ptr::eq(first, second));
        assert!(first.contains("nax_available="));
    }
}
