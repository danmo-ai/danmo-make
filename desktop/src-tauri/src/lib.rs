mod runtime;

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, Url};

const API_PID_FILE: &str = "api.pid";
const API_PORT_FILE: &str = "api.port";

/// Owns the desktop API child (PyInstaller sidecar or thin uvicorn).
/// Rust's `Child` drop does **not** kill the OS process — terminate explicitly
/// on stop / ExitRequested / Exit / Drop (aligned with danmo-work).
pub struct ApiProcess {
    child: Mutex<Option<Child>>,
    /// Control plane (``~/.danmo-make``) where pid/port/logs live.
    data_dir: Mutex<Option<PathBuf>>,
    /// Idempotent app-exit shutdown (restart paths reset via `take()` only).
    exiting: AtomicBool,
}

impl ApiProcess {
    fn new() -> Self {
        Self {
            child: Mutex::new(None),
            data_dir: Mutex::new(None),
            exiting: AtomicBool::new(false),
        }
    }

    fn shutdown(&self) {
        let mut guard = match self.child.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(mut child) = guard.take() {
            terminate_child(&mut child);
        }
        let data_dir = match self.data_dir.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        if let Some(dir) = data_dir {
            reclaim_pidfile(&dir);
        }
        eprintln!("[sidecar] shutdown complete");
    }
}

impl Drop for ApiProcess {
    fn drop(&mut self) {
        if self
            .exiting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.shutdown();
        }
    }
}

fn sidecar_exe(app: &AppHandle) -> Result<PathBuf, String> {
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let base = res.join("danqing-api");
    #[cfg(windows)]
    let candidates = [base.join("danqing-api.exe"), base.join("danqing-api")];
    #[cfg(not(windows))]
    let candidates = [base.join("danqing-api")];
    for exe in candidates {
        if exe.is_file() {
            return Ok(exe);
        }
    }
    Err(format!(
        "Sidecar not found under {} (run pack-*-sidecar / pack-*-desktop for this platform)",
        base.display()
    ))
}

#[cfg(not(debug_assertions))]
fn wait_for_main_window(app: &AppHandle) -> Result<(), String> {
    for _ in 0..100 {
        if app.get_webview_window("main").is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err("Timed out waiting for main window".to_string())
}

fn log_tail(path: &Path, lines: usize) -> String {
    let log_tail = fs::read_to_string(path).unwrap_or_default();
    log_tail
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

fn wait_for_health(port: u16, child: &mut Child, log_path: &Path) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/api/system/health");
    for _ in 0..120 {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "API process exited with status {status} before health check passed.\nLog: {}\n---\n{}",
                log_path.display(),
                log_tail(log_path, 12)
            ));
        }
        match ureq::get(&url).call() {
            Ok(resp) if (200..300).contains(&resp.status()) => return Ok(()),
            _ => thread::sleep(Duration::from_millis(500)),
        }
    }
    Err(format!(
        "Timed out waiting for API at {url}\nLog: {}\n---\n{}",
        log_path.display(),
        log_tail(log_path, 12)
    ))
}

/// Stop a PID (and its process group on Unix). Used for pidfile orphans.
fn stop_pid(pid: u32) {
    if pid == 0 {
        return;
    }
    #[cfg(unix)]
    {
        // Negative PID = process group (when spawned with process_group(0)).
        let pg = format!("-{pid}");
        let _ = Command::new("kill").args(["-TERM", &pg]).status();
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
        thread::sleep(Duration::from_millis(200));
        let _ = Command::new("kill").args(["-KILL", &pg]).status();
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }
}

/// Graceful-then-force terminate of a spawned API Child (process group on Unix).
fn terminate_child(child: &mut Child) {
    let pid = child.id();
    eprintln!("[sidecar] terminating api pid={pid}");
    #[cfg(unix)]
    {
        let pg = format!("-{pid}");
        let _ = Command::new("kill").args(["-TERM", &pg]).status();
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
        thread::sleep(Duration::from_millis(250));
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!("[sidecar] api pid={pid} exited ({status})");
            }
            _ => {
                let _ = Command::new("kill").args(["-KILL", &pg]).status();
                let _ = child.kill();
                let _ = child.wait();
                eprintln!("[sidecar] api pid={pid} force-killed");
            }
        }
        // Belt-and-suspenders: group may still hold helpers that left the Child wait set.
        stop_pid(pid);
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
        let _ = child.wait();
        eprintln!("[sidecar] api pid={pid} taskkilled");
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// PIDs still listening on a TCP port (best-effort; mirrors danmo-work reclaim).
fn listeners_on_port(port: u16) -> Vec<u32> {
    #[cfg(unix)]
    {
        let output = Command::new("lsof")
            .args([
                "-nP",
                &format!("-iTCP:{port}"),
                "-sTCP:LISTEN",
                "-t",
            ])
            .output();
        let Ok(out) = output else {
            return Vec::new();
        };
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect()
    }
    #[cfg(windows)]
    {
        let output = Command::new("netstat").args(["-ano"]).output();
        let Ok(out) = output else {
            return Vec::new();
        };
        let needle = format!(":{port}");
        let mut pids = Vec::new();
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let lower = line.to_ascii_lowercase();
            if !(lower.contains(&needle) && lower.contains("listen")) {
                continue;
            }
            if let Some(pid) = line.split_whitespace().last().and_then(|s| s.parse().ok()) {
                if pid > 0 && !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
        }
        pids
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = port;
        Vec::new()
    }
}

/// Kill pidfile process + anything still listening on the recorded API port.
fn reclaim_pidfile(data_dir: &Path) {
    let pid_path = data_dir.join(API_PID_FILE);
    let port_path = data_dir.join(API_PORT_FILE);
    if let Ok(raw) = fs::read_to_string(&pid_path) {
        if let Ok(pid) = raw.trim().parse::<u32>() {
            eprintln!("[sidecar] stopping previous api pid={pid}");
            stop_pid(pid);
        }
    }
    if let Ok(raw) = fs::read_to_string(&port_path) {
        if let Ok(port) = raw.trim().parse::<u16>() {
            for pid in listeners_on_port(port) {
                eprintln!("[sidecar] reclaiming 127.0.0.1:{port} from pid={pid}");
                stop_pid(pid);
            }
        }
    }
    let _ = fs::remove_file(&pid_path);
    let _ = fs::remove_file(&port_path);
}

/// Stop managed child + any leftover pidfile process from a previous crash.
fn reclaim_stale_api(app: &AppHandle, data_dir: &Path) {
    if let Some(api) = app.try_state::<ApiProcess>() {
        let mut guard = match api.child.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(mut child) = guard.take() {
            terminate_child(&mut child);
        }
    }
    reclaim_pidfile(data_dir);
    // Give the OS a moment to release ports / file locks.
    thread::sleep(Duration::from_millis(300));
}

fn write_pid_files(data_dir: &Path, pid: u32, port: u16) {
    let _ = fs::write(data_dir.join(API_PID_FILE), pid.to_string());
    let _ = fs::write(data_dir.join(API_PORT_FILE), port.to_string());
}

fn clear_pid_files(data_dir: &Path) {
    let _ = fs::remove_file(data_dir.join(API_PID_FILE));
    let _ = fs::remove_file(data_dir.join(API_PORT_FILE));
}

fn apply_process_group(cmd: &mut Command) {
    // Own process group on Unix so Exit/Drop can SIGTERM/SIGKILL the whole tree
    // (uvicorn/PyInstaller helpers; killing only the parent leaves orphans).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let _ = cmd;
}

fn register_child(app: &AppHandle, data_dir: PathBuf, child: Child) -> Result<(), String> {
    let api = app.state::<ApiProcess>();
    *api.child
        .lock()
        .map_err(|_| "api process lock poisoned".to_string())? = Some(child);
    *api.data_dir
        .lock()
        .map_err(|_| "api process lock poisoned".to_string())? = Some(data_dir);
    Ok(())
}

fn stop_api_process(app: &AppHandle) -> Result<(), String> {
    let api = app.state::<ApiProcess>();
    let mut g = api
        .child
        .lock()
        .map_err(|_| "api process lock poisoned".to_string())?;
    if let Some(mut c) = g.take() {
        terminate_child(&mut c);
    }
    drop(g);
    let data_dir = api
        .data_dir
        .lock()
        .map_err(|_| "api process lock poisoned".to_string())?
        .clone();
    if let Some(dir) = data_dir {
        reclaim_pidfile(&dir);
    } else if let Ok(dir) = runtime::server_data_dir(app) {
        // Spawn may have failed after writing a pidfile.
        reclaim_pidfile(&dir);
    }
    Ok(())
}

/// Called on ExitRequested / Exit — Child drop alone never kills the OS process.
fn shutdown_api_on_exit(app: &AppHandle) {
    if let Some(api) = app.try_state::<ApiProcess>() {
        if api
            .exiting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        api.shutdown();
        return;
    }
    // Manage() never ran, but a pidfile may still exist from a prior launch.
    if let Ok(dir) = runtime::server_data_dir(app) {
        reclaim_pidfile(&dir);
    }
}

fn open_log(log_path: &Path) -> Result<(File, File), String> {
    let stdout = File::options()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open {}: {e}", log_path.display()))?;
    let stderr = File::options()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open {}: {e}", log_path.display()))?;
    Ok((stdout, stderr))
}

fn prepare_api_log(user_dir: &Path) -> Result<PathBuf, String> {
    let log_dir = user_dir.join("logs");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join("sidecar.log");
    let mut log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| format!("open {}: {e}", log_path.display()))?;
    let _ = writeln!(log_file, "--- sidecar spawn ---");
    Ok(log_path)
}

fn fail_if_exited_immediately(
    child: &mut Child,
    data_dir: &Path,
    log_path: &Path,
) -> Result<(), String> {
    // Fail fast if the process exits immediately (common under App Translocation /
    // missing mlx[cuda] deps / broken venv).
    thread::sleep(Duration::from_millis(400));
    match child.try_wait() {
        Ok(Some(status)) => {
            clear_pid_files(data_dir);
            Err(format!(
                "API process exited immediately ({status}). log tail:\n{}",
                log_tail(log_path, 40)
            ))
        }
        Ok(None) => Ok(()),
        Err(e) => Err(format!("API wait failed: {e}")),
    }
}

fn start_thin_api(app: &AppHandle) -> Result<u16, String> {
    let user_dir = runtime::server_data_dir(app)?;
    reclaim_stale_api(app, &user_dir);

    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);

    let log_path = prepare_api_log(&user_dir)?;
    let _ = writeln!(
        &mut OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| e.to_string())?,
        "Starting thin runtime API on port {port}…"
    );

    let py = runtime::venv_python(app)?;
    let app_root = runtime::app_root(app)?;
    let mut cmd = Command::new(&py);
    cmd.current_dir(&app_root)
        .arg("-m")
        .arg("uvicorn")
        .arg("backend.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("DANQING_HTTP_HOST", "127.0.0.1")
        .env("DANQING_HTTP_PORT", port.to_string())
        .env("DANQING_USER_DATA_DIR", user_dir.as_os_str())
        .env("PYTHONPATH", app_root.as_os_str());
    apply_process_group(&mut cmd);

    let (stdout, stderr) = open_log(&log_path)?;
    let mut child = cmd
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", py.display()))?;

    write_pid_files(&user_dir, child.id(), port);
    fail_if_exited_immediately(&mut child, &user_dir, &log_path)?;
    wait_for_health(port, &mut child, &log_path)?;
    eprintln!(
        "[sidecar] thin api spawned pid={} on 127.0.0.1:{port}",
        child.id()
    );
    register_child(app, user_dir, child)?;
    Ok(port)
}

fn start_sidecar(app: &AppHandle) -> Result<u16, String> {
    let user_dir = runtime::server_data_dir(app)?;
    reclaim_stale_api(app, &user_dir);

    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);

    let log_path = prepare_api_log(&user_dir)?;
    let _ = writeln!(
        &mut OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| e.to_string())?,
        "Starting danqing-api on port {port}…"
    );

    let exe = sidecar_exe(app)?;
    let cwd = exe
        .parent()
        .ok_or_else(|| format!("sidecar has no parent dir: {}", exe.display()))?;

    let mut cmd = Command::new(&exe);
    cmd.current_dir(cwd)
        .env("DANQING_HTTP_HOST", "127.0.0.1")
        .env("DANQING_HTTP_PORT", port.to_string())
        .env("DANQING_USER_DATA_DIR", user_dir.as_os_str());
    #[cfg(target_os = "macos")]
    {
        let mlx_lib = cwd.join("_internal").join("mlx").join("lib");
        if mlx_lib.is_dir() {
            cmd.env("DYLD_LIBRARY_PATH", mlx_lib.as_os_str());
        }
    }
    apply_process_group(&mut cmd);

    let (stdout, stderr) = open_log(&log_path)?;
    let mut child = cmd
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", exe.display()))?;

    write_pid_files(&user_dir, child.id(), port);
    fail_if_exited_immediately(&mut child, &user_dir, &log_path)?;
    wait_for_health(port, &mut child, &log_path)?;
    eprintln!(
        "[sidecar] danqing-api spawned pid={} on 127.0.0.1:{port} (cwd={})",
        child.id(),
        cwd.display()
    );
    register_child(app, user_dir, child)?;
    Ok(port)
}

fn navigate_main(app: &AppHandle, port: u16) -> Result<(), String> {
    let win = app
        .get_webview_window("main")
        .ok_or_else(|| "missing webview window 'main'".to_string())?;
    let target = Url::parse(&format!("http://127.0.0.1:{port}/")).map_err(|e| e.to_string())?;
    win.navigate(target).map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    apply_macos_shell(app);
    Ok(())
}

fn navigate_runtime_setup(app: &AppHandle) -> Result<(), String> {
    let win = app
        .get_webview_window("main")
        .ok_or_else(|| "missing webview window 'main'".to_string())?;
    // loader/frontendDist ships runtime-setup.html next to index.html
    win.eval("window.location.replace('runtime-setup.html');")
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(debug_assertions))]
fn bootstrap_production(app: &AppHandle) -> Result<(), String> {
    wait_for_main_window(app)?;
    if runtime::is_thin_runtime(app) {
        if !runtime::env_ready(app) {
            return navigate_runtime_setup(app);
        }
        let port = start_thin_api(app)?;
        return navigate_main(app, port);
    }
    let port = start_sidecar(app)?;
    navigate_main(app, port)
}

#[cfg(not(debug_assertions))]
fn spawn_production_bootstrap(app: &AppHandle) {
    static BOOTSTRAP_STARTED: AtomicBool = AtomicBool::new(false);
    if BOOTSTRAP_STARTED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    let handle = app.clone();
    thread::spawn(move || {
        if let Err(err) = bootstrap_production(&handle) {
            eprintln!("Danmo Make desktop bootstrap failed: {err}");
            let app = handle.clone();
            let _ = handle.run_on_main_thread(move || {
                if let Some(win) = app.get_webview_window("main") {
                    let html = format!(
                        "<html><body style=\"font-family:system-ui;background:#1a1a2e;color:#eaeaea;padding:2rem\"><h2>Failed to start API</h2><pre style=\"white-space:pre-wrap;opacity:0.9\">{}</pre></body></html>",
                        html_escape(&err)
                    );
                    let data_url = format!("data:text/html;charset=utf-8,{}", pct_encode(&html));
                    if let Ok(url) = Url::parse(&data_url) {
                        let _ = win.navigate(url);
                    }
                }
            });
        }
    });
}

#[cfg(not(debug_assertions))]
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(not(debug_assertions))]
fn pct_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn apply_macos_shell(app: &AppHandle) {
    let Some(win) = app.get_webview_window("main") else {
        return;
    };
    use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial};
    if let Err(err) = apply_vibrancy(&win, NSVisualEffectMaterial::UnderWindowBackground, None, None)
    {
        eprintln!("macOS window vibrancy failed: {err}");
    }
    if let Err(err) = win.set_theme(Some(tauri::Theme::Dark)) {
        eprintln!("macOS window theme failed: {err}");
    }
    let _ = win.eval(r#"document.documentElement.classList.add('dq-tauri-macos');"#);
}

#[tauri::command]
fn runtime_status(app: AppHandle) -> runtime::RuntimeStatusDto {
    runtime::runtime_status(&app)
}

#[tauri::command]
fn runtime_set_mirror(app: AppHandle, mirror: String) -> Result<(), String> {
    runtime::set_mirror(&app, &mirror)
}

#[tauri::command]
fn runtime_install_start(app: AppHandle, mode: String, mirror: Option<String>) -> Result<(), String> {
    // Stop API before repair/reinstall so files are not locked
    if mode == "repair" || mode == "reinstall" {
        let _ = stop_api_process(&app);
    }
    runtime::start_install(app, mode, mirror)
}

#[tauri::command]
fn runtime_install_cancel() -> Result<(), String> {
    runtime::cancel_install();
    Ok(())
}

#[tauri::command]
fn runtime_stop_api(app: AppHandle) -> Result<(), String> {
    stop_api_process(&app)
}

#[tauri::command]
fn runtime_start_api(app: AppHandle) -> Result<u16, String> {
    // Always reclaim previous child/pidfile before (re)start.
    if runtime::is_thin_runtime(&app) {
        if !runtime::env_ready(&app) {
            return Err("Runtime not ready".into());
        }
        let port = start_thin_api(&app)?;
        navigate_main(&app, port)?;
        Ok(port)
    } else {
        let port = start_sidecar(&app)?;
        navigate_main(&app, port)?;
        Ok(port)
    }
}

#[tauri::command]
fn runtime_open_setup(app: AppHandle) -> Result<(), String> {
    let _ = stop_api_process(&app);
    navigate_runtime_setup(&app)
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(ApiProcess::new())
        .invoke_handler(tauri::generate_handler![
            runtime_status,
            runtime_set_mirror,
            runtime_install_start,
            runtime_install_cancel,
            runtime_stop_api,
            runtime_start_api,
            runtime_open_setup,
        ])
        .setup(|app| {
            #[cfg(target_os = "macos")]
            apply_macos_shell(app.handle());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        #[cfg(not(debug_assertions))]
        if matches!(event, RunEvent::Ready) {
            spawn_production_bootstrap(app_handle);
        }

        // ExitRequested fires first (window close / Cmd+Q); Exit is last chance.
        // Without this, the API sidecar keeps listening forever.
        match event {
            RunEvent::ExitRequested { .. } | RunEvent::Exit => {
                shutdown_api_on_exit(app_handle);
            }
            _ => {}
        }
    });
}
