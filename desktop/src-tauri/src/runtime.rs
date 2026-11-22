//! Desktop runtime status helpers.
//!
//! The former Linux MLX **Python thin runtime** (portable CPython + uvicorn
//! `backend.main:app` + `runtime_bootstrap.py`) was removed. Desktop always
//! starts the `danqing-api` sidecar. These APIs remain so the shell / setup UI
//! can report that thin install is retired instead of spawning Python.

use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager};

#[derive(Clone, Serialize)]
pub struct RuntimeStatusDto {
    pub mode: String,
    pub ready: bool,
    pub thin: bool,
    pub mirror: String,
    pub mirrors: Vec<String>,
    pub data_dir: String,
    pub log_path: String,
    pub detail: Value,
    pub message: String,
}

/// Control plane root — ``~/.danmo-make`` (aligned with danmo-work ``~/.danmo-work``).
/// Override with ``DANQING_USER_DATA_DIR``. Holds pointer, app settings, logs.
pub fn server_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(raw) = std::env::var("DANQING_USER_DATA_DIR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let dir = PathBuf::from(trimmed);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let _ = std::fs::create_dir_all(dir.join("logs"));
            let _ = std::fs::create_dir_all(dir.join("config"));
            return Ok(dir);
        }
    }
    let home = app
        .path()
        .home_dir()
        .map_err(|e| format!("failed to resolve home dir: {e}"))?;
    let dir = home.join(".danmo-make");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let _ = std::fs::create_dir_all(dir.join("logs"));
    let _ = std::fs::create_dir_all(dir.join("config"));
    Ok(dir)
}

pub fn runtime_status(app: &AppHandle) -> RuntimeStatusDto {
    RuntimeStatusDto {
        mode: "sidecar".into(),
        ready: true,
        thin: false,
        mirror: "n/a".into(),
        mirrors: vec![],
        data_dir: server_data_dir(app)
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        log_path: String::new(),
        detail: Value::Null,
        message: "Python thin runtime removed — desktop uses danqing-api sidecar only".into(),
    }
}

pub fn set_mirror(_app: &AppHandle, _mirror: &str) -> Result<(), String> {
    Err("Python thin desktop runtime was removed; pip mirror setup no longer applies".into())
}

pub fn cancel_install() {}

pub fn start_install(
    _app: AppHandle,
    _mode: String,
    _mirror: Option<String>,
) -> Result<(), String> {
    Err(
        "Python thin desktop runtime was removed; use a danqing-api sidecar pack instead"
            .into(),
    )
}
