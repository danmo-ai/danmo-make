//! CUDA thin-runtime helpers for the desktop shell.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};

static INSTALL_CANCEL: AtomicBool = AtomicBool::new(false);
static INSTALL_RUNNING: AtomicBool = AtomicBool::new(false);

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
/// Override with ``DANQING_USER_DATA_DIR``. Holds pointer, app settings, logs, runtime-venv.
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

fn resource_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(res) = app.path().resource_dir() {
        out.push(res.clone());
        if let Some(parent) = res.parent() {
            out.push(parent.to_path_buf());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.to_path_buf());
        }
    }
    out
}

pub fn thin_runtime_root(app: &AppHandle) -> Option<PathBuf> {
    for base in resource_candidates(app) {
        let runtime = base.join("runtime");
        let py_unix = runtime.join("python").join("bin").join("python3");
        let py_win = runtime.join("python").join("python.exe");
        let app_backend = runtime.join("app").join("backend");
        if (py_unix.is_file() || py_win.is_file()) && app_backend.is_dir() {
            return Some(runtime);
        }
    }
    None
}

pub fn is_thin_runtime(app: &AppHandle) -> bool {
    thin_runtime_root(app).is_some()
}

fn portable_python(runtime: &Path) -> Result<PathBuf, String> {
    let win = runtime.join("python").join("python.exe");
    if win.is_file() {
        return Ok(win);
    }
    let unix = runtime.join("python").join("bin").join("python3");
    if unix.is_file() {
        return Ok(unix);
    }
    let unix2 = runtime.join("python").join("bin").join("python");
    if unix2.is_file() {
        return Ok(unix2);
    }
    Err(format!("portable python missing under {}", runtime.display()))
}

fn bootstrap_script(runtime: &Path) -> Result<PathBuf, String> {
    let p = runtime.join("app").join("scripts").join("runtime_bootstrap.py");
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("missing {}", p.display()))
    }
}

fn env_json_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(server_data_dir(app)?.join("runtime-env.json"))
}

pub fn env_ready(app: &AppHandle) -> bool {
    let Ok(path) = env_json_path(app) else {
        return false;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    v.get("ready").and_then(|x| x.as_bool()).unwrap_or(false)
}

fn run_bootstrap_json(
    app: &AppHandle,
    extra_args: &[&str],
) -> Result<Value, String> {
    let runtime = thin_runtime_root(app).ok_or_else(|| "thin runtime not found".to_string())?;
    let py = portable_python(&runtime)?;
    let script = bootstrap_script(&runtime)?;
    let data = server_data_dir(app)?;
    let app_root = runtime.join("app");
    let portable = runtime.join("python");

    let mut cmd = Command::new(&py);
    cmd.arg(&script)
        .arg("--json")
        .arg("--data-dir")
        .arg(&data)
        .arg("--app-root")
        .arg(&app_root)
        .arg("--portable-python")
        .arg(&portable);
    for a in extra_args {
        cmd.arg(a);
    }
    let out = cmd.output().map_err(|e| format!("spawn bootstrap: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if !out.status.success() && stdout.is_empty() {
        return Err(format!(
            "bootstrap failed ({})",
            if stderr.is_empty() {
                out.status.to_string()
            } else {
                stderr
            }
        ));
    }
    // status may exit 1 when not ready but still print JSON
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    serde_json::from_str(json_line).map_err(|e| {
        format!("parse bootstrap json: {e}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    })
}

pub fn runtime_status(app: &AppHandle) -> RuntimeStatusDto {
    let thin = is_thin_runtime(app);
    if !thin {
        return RuntimeStatusDto {
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
            message: "MLX/sidecar build — runtime bootstrap not applicable".into(),
        };
    }
    match run_bootstrap_json(app, &["--status"]) {
        Ok(v) => RuntimeStatusDto {
            mode: "thin".into(),
            ready: v.get("ready").and_then(|x| x.as_bool()).unwrap_or(false),
            thin: true,
            mirror: v
                .get("mirror")
                .and_then(|x| x.as_str())
                .unwrap_or("official")
                .to_string(),
            mirrors: v
                .get("mirrors")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        "official".into(),
                        "tuna".into(),
                        "aliyun".into(),
                    ]
                }),
            data_dir: v
                .get("data_dir")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            log_path: v
                .get("log_path")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            detail: v.get("detail").cloned().unwrap_or(Value::Null),
            message: String::new(),
        },
        Err(err) => RuntimeStatusDto {
            mode: "thin".into(),
            ready: false,
            thin: true,
            mirror: "official".into(),
            mirrors: vec![
                "official".into(),
                "tuna".into(),
                "aliyun".into(),
            ],
            data_dir: server_data_dir(app)
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            log_path: String::new(),
            detail: Value::Null,
            message: err,
        },
    }
}

pub fn set_mirror(app: &AppHandle, mirror: &str) -> Result<(), String> {
    let runtime = thin_runtime_root(app).ok_or_else(|| "thin runtime not found".to_string())?;
    let data = server_data_dir(app)?;
    let setup = data.join("runtime-setup.json");
    let _ = runtime;
    let body = serde_json::json!({
        "mirror": mirror,
        "updated_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
    std::fs::write(&setup, format!("{}\n", body)).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn cancel_install() {
    INSTALL_CANCEL.store(true, Ordering::SeqCst);
}

pub fn start_install(
    app: AppHandle,
    mode: String,
    mirror: Option<String>,
) -> Result<(), String> {
    if !is_thin_runtime(&app) {
        return Err("Runtime install is only available on CUDA thin desktop builds".into());
    }
    if INSTALL_RUNNING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Runtime install already in progress".into());
    }
    INSTALL_CANCEL.store(false, Ordering::SeqCst);

    let mode = match mode.as_str() {
        "bootstrap" | "repair" | "reinstall" => mode,
        _ => {
            INSTALL_RUNNING.store(false, Ordering::SeqCst);
            return Err(format!("unknown mode {mode}"));
        }
    };

    thread::spawn(move || {
        let result = run_install_thread(&app, &mode, mirror.as_deref());
        INSTALL_RUNNING.store(false, Ordering::SeqCst);
        let payload = match result {
            Ok(()) => serde_json::json!({"ok": true, "phase": "done"}),
            Err(err) => serde_json::json!({"ok": false, "phase": "error", "message": err}),
        };
        let _ = app.emit("runtime-setup://progress", payload);
    });
    Ok(())
}

fn run_install_thread(app: &AppHandle, mode: &str, mirror: Option<&str>) -> Result<(), String> {
    let runtime = thin_runtime_root(app).ok_or_else(|| "thin runtime not found".to_string())?;
    let py = portable_python(&runtime)?;
    let script = bootstrap_script(&runtime)?;
    let data = server_data_dir(app)?;
    let app_root = runtime.join("app");
    let portable = runtime.join("python");
    let log_dir = data.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join("runtime-setup.log");

    if let Some(m) = mirror {
        set_mirror(app, m)?;
    }

    let mut args: Vec<String> = vec![
        script.display().to_string(),
        "--data-dir".into(),
        data.display().to_string(),
        "--app-root".into(),
        app_root.display().to_string(),
        "--portable-python".into(),
        portable.display().to_string(),
    ];
    match mode {
        "repair" => args.push("--repair".into()),
        "reinstall" => {
            args.push("--reinstall".into());
            args.push("--yes".into());
        }
        _ => {}
    }
    if let Some(m) = mirror {
        args.push("--mirror".into());
        args.push(m.to_string());
    }

    let _ = app.emit(
        "runtime-setup://progress",
        serde_json::json!({"phase": "start", "mode": mode, "message": "starting"}),
    );

    let mut child = Command::new(&py)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn install: {e}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_out = app.clone();
    let app_err = app.clone();
    let log_out = log_path.clone();
    let log_err = log_path.clone();

    let t_out = thread::spawn(move || pipe_lines(stdout, &app_out, &log_out, "stdout"));
    let t_err = thread::spawn(move || pipe_lines(stderr, &app_err, &log_err, "stderr"));

    loop {
        if INSTALL_CANCEL.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = t_out.join();
            let _ = t_err.join();
            return Err("Runtime install cancelled".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = t_out.join();
                let _ = t_err.join();
                if status.success() {
                    return Ok(());
                }
                return Err(format!(
                    "Runtime install failed ({status}). See {}",
                    log_path.display()
                ));
            }
            Ok(None) => thread::sleep(Duration::from_millis(200)),
            Err(e) => {
                let _ = child.kill();
                return Err(format!("wait install: {e}"));
            }
        }
    }
}

fn pipe_lines<T: std::io::Read + Send + 'static>(
    stream: Option<T>,
    app: &AppHandle,
    log_path: &Path,
    _label: &str,
) {
    let Some(stream) = stream else { return };
    let reader = BufReader::new(stream);
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .ok();
    for line in reader.lines().flatten() {
        if let Some(f) = log_file.as_mut() {
            let _ = writeln!(f, "{line}");
        }
        let phase = if line.contains("torch") {
            "torch"
        } else if line.contains("pip") {
            "pip"
        } else {
            "progress"
        };
        let _ = app.emit(
            "runtime-setup://progress",
            serde_json::json!({"phase": phase, "message": line}),
        );
    }
}

pub fn venv_python(app: &AppHandle) -> Result<PathBuf, String> {
    let data = server_data_dir(app)?;
    let win = data.join("runtime-venv").join("Scripts").join("python.exe");
    if win.is_file() {
        return Ok(win);
    }
    let unix = data.join("runtime-venv").join("bin").join("python3");
    if unix.is_file() {
        return Ok(unix);
    }
    let unix2 = data.join("runtime-venv").join("bin").join("python");
    if unix2.is_file() {
        return Ok(unix2);
    }
    // Fall back to runtime-env.json
    if let Ok(text) = std::fs::read_to_string(data.join("runtime-env.json")) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            if let Some(p) = v.get("python").and_then(|x| x.as_str()) {
                let path = PathBuf::from(p);
                if path.is_file() {
                    return Ok(path);
                }
            }
        }
    }
    Err("runtime venv python not found — run setup first".into())
}

pub fn app_root(app: &AppHandle) -> Result<PathBuf, String> {
    let runtime = thin_runtime_root(app).ok_or_else(|| "thin runtime not found".to_string())?;
    Ok(runtime.join("app"))
}

