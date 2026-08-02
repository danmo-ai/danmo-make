use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, Url};

pub struct ApiProcess(pub Mutex<Option<Child>>);

static BOOTSTRAP_STARTED: AtomicBool = AtomicBool::new(false);

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

fn wait_for_main_window(app: &AppHandle) -> Result<(), String> {
    for _ in 0..100 {
        if app.get_webview_window("main").is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err("Timed out waiting for main window".to_string())
}

fn wait_for_health(port: u16, child: &mut Child, log_path: &Path) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/api/system/health");
    for _ in 0..120 {
        if let Ok(Some(status)) = child.try_wait() {
            let log_tail = std::fs::read_to_string(log_path).unwrap_or_default();
            let tail: String = log_tail.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            return Err(format!(
                "API process exited with status {status} before health check passed.\nLog: {}\n---\n{tail}",
                log_path.display()
            ));
        }
        match ureq::get(&url).call() {
            Ok(resp) if (200..300).contains(&resp.status()) => return Ok(()),
            _ => thread::sleep(Duration::from_millis(500)),
        }
    }
    let log_tail = std::fs::read_to_string(log_path).unwrap_or_default();
    Err(format!(
        "Timed out waiting for API at {url}\nLog: {}\n---\n{}",
        log_path.display(),
        log_tail.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")
    ))
}

fn start_sidecar(app: &AppHandle) -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);

    let user_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("server-data");
    std::fs::create_dir_all(&user_dir).map_err(|e| e.to_string())?;
    let log_dir = user_dir.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join("sidecar.log");
    let mut log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| format!("open {}: {e}", log_path.display()))?;
    let _ = writeln!(log_file, "Starting danqing-api on port {port}…");

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
    let mut child = cmd
        .stdout(Stdio::from(
            File::options()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| e.to_string())?,
        ))
        .stderr(Stdio::from(
            File::options()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| e.to_string())?,
        ))
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", exe.display()))?;

    wait_for_health(port, &mut child, &log_path)?;

    let api = app.state::<ApiProcess>();
    *api.0.lock().map_err(|_| "api process lock poisoned".to_string())? = Some(child);

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

fn bootstrap_production(app: &AppHandle) -> Result<(), String> {
    wait_for_main_window(app)?;
    let port = start_sidecar(app)?;
    navigate_main(app, port)
}

fn spawn_production_bootstrap(app: &AppHandle) {
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
                    let data_url = format!(
                        "data:text/html;charset=utf-8,{}",
                        pct_encode(&html)
                    );
                    if let Ok(url) = Url::parse(&data_url) {
                        let _ = win.navigate(url);
                    }
                }
            });
        }
    });
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

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
    if let Err(err) = apply_vibrancy(&win, NSVisualEffectMaterial::UnderWindowBackground, None, None) {
        eprintln!("macOS window vibrancy failed: {err}");
    }
    // Match default product theme flash (`dq-mac` / dark) until the webview syncs.
    if let Err(err) = win.set_theme(Some(tauri::Theme::Dark)) {
        eprintln!("macOS window theme failed: {err}");
    }
    let _ = win.eval(
        r#"document.documentElement.classList.add('dq-tauri-macos');"#,
    );
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(ApiProcess(Mutex::new(None)))
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

        if matches!(event, RunEvent::Exit) {
            if let Some(api) = app_handle.try_state::<ApiProcess>() {
                if let Ok(mut g) = api.0.lock() {
                    if let Some(mut c) = g.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                }
            }
        }
    });
}
