#!/usr/bin/env python3
"""Package thin CUDA desktop / server archives (no embedded torch)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402
import stage_cuda_runtime as stage  # noqa: E402

_RUN_SH = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export DANQING_USER_DATA_DIR="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}"
mkdir -p "$DANQING_USER_DATA_DIR"/{models,outputs,db,config,logs}
export DANQING_HTTP_HOST="${DANQING_HTTP_HOST:-0.0.0.0}"
export DANQING_HTTP_PORT="${DANQING_HTTP_PORT:-7800}"
export DANQING_APP_ROOT="$ROOT/runtime/app"
export DANQING_PORTABLE_PYTHON="$ROOT/runtime/python"
export PYTHONPATH="$DANQING_APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BOOTSTRAP_PY="$DANQING_APP_ROOT/scripts/runtime_bootstrap.py"
PY_BOOT="$DANQING_PORTABLE_PYTHON/bin/python3"
if [[ ! -x "$PY_BOOT" ]]; then
  PY_BOOT="$DANQING_PORTABLE_PYTHON/bin/python"
fi

mode=""
for arg in "$@"; do
  case "$arg" in
    --repair-runtime) mode=repair ;;
    --reinstall-runtime) mode=reinstall ;;
    --status-runtime) mode=status ;;
  esac
done

if [[ "$mode" == "status" ]]; then
  exec "$PY_BOOT" "$BOOTSTRAP_PY" --status --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON"
fi

need_setup=1
if "$PY_BOOT" "$BOOTSTRAP_PY" --status --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON" >/dev/null 2>&1; then
  need_setup=0
fi

if [[ "$mode" == "repair" || "$mode" == "reinstall" ]]; then
  need_setup=1
fi

if [[ "$need_setup" -eq 1 ]]; then
  if [[ "${DANQING_RUNTIME_SKIP_AUTO_SETUP:-0}" == "1" && -z "$mode" ]]; then
    echo "Runtime not ready. Run: $0 --repair-runtime" >&2
    echo "Or unset DANQING_RUNTIME_SKIP_AUTO_SETUP to auto-install on start." >&2
    exit 1
  fi
  echo "==> Installing CUDA Python runtime (progress on stderr)…" >&2
  extra=()
  if [[ "$mode" == "repair" ]]; then
    extra+=(--repair)
  elif [[ "$mode" == "reinstall" ]]; then
    extra+=(--reinstall --yes)
  fi
  "$PY_BOOT" "$BOOTSTRAP_PY" --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON" \\
    "${extra[@]}"
fi

VENV_PY="$DANQING_USER_DATA_DIR/runtime-venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python at $VENV_PY" >&2
  exit 1
fi
cd "$DANQING_APP_ROOT"
exec "$VENV_PY" -m uvicorn backend.main:app --host "$DANQING_HTTP_HOST" --port "$DANQING_HTTP_PORT"
"""

_RUN_BAT = r"""@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if not defined DANQING_USER_DATA_DIR set "DANQING_USER_DATA_DIR=%USERPROFILE%\.danmo-make"
if not exist "%DANQING_USER_DATA_DIR%\models" mkdir "%DANQING_USER_DATA_DIR%\models"
if not exist "%DANQING_USER_DATA_DIR%\outputs" mkdir "%DANQING_USER_DATA_DIR%\outputs"
if not exist "%DANQING_USER_DATA_DIR%\db" mkdir "%DANQING_USER_DATA_DIR%\db"
if not exist "%DANQING_USER_DATA_DIR%\config" mkdir "%DANQING_USER_DATA_DIR%\config"
if not exist "%DANQING_USER_DATA_DIR%\logs" mkdir "%DANQING_USER_DATA_DIR%\logs"
if not defined DANQING_HTTP_HOST set "DANQING_HTTP_HOST=0.0.0.0"
if not defined DANQING_HTTP_PORT set "DANQING_HTTP_PORT=7800"
set "DANQING_APP_ROOT=%ROOT%runtime\app"
set "DANQING_PORTABLE_PYTHON=%ROOT%runtime\python"
set "PYTHONPATH=%DANQING_APP_ROOT%;%PYTHONPATH%"
set "BOOTSTRAP_PY=%DANQING_APP_ROOT%\scripts\runtime_bootstrap.py"
set "PY_BOOT=%DANQING_PORTABLE_PYTHON%\python.exe"

set "MODE="
if /I "%~1"=="--repair-runtime" set "MODE=repair"
if /I "%~1"=="--reinstall-runtime" set "MODE=reinstall"
if /I "%~1"=="--status-runtime" set "MODE=status"

if /I "%MODE%"=="status" (
  "%PY_BOOT%" "%BOOTSTRAP_PY%" --status --data-dir "%DANQING_USER_DATA_DIR%" --app-root "%DANQING_APP_ROOT%" --portable-python "%DANQING_PORTABLE_PYTHON%"
  exit /b %ERRORLEVEL%
)

"%PY_BOOT%" "%BOOTSTRAP_PY%" --status --data-dir "%DANQING_USER_DATA_DIR%" --app-root "%DANQING_APP_ROOT%" --portable-python "%DANQING_PORTABLE_PYTHON%" >NUL 2>&1
set "NEED_SETUP=%ERRORLEVEL%"
if /I "%MODE%"=="repair" set "NEED_SETUP=1"
if /I "%MODE%"=="reinstall" set "NEED_SETUP=1"

if not "%NEED_SETUP%"=="0" (
  if "%DANQING_RUNTIME_SKIP_AUTO_SETUP%"=="1" if "%MODE%"=="" (
    echo Runtime not ready. Run: run.bat --repair-runtime
    exit /b 1
  )
  echo Installing CUDA Python runtime...
  if /I "%MODE%"=="repair" (
    "%PY_BOOT%" "%BOOTSTRAP_PY%" --repair --data-dir "%DANQING_USER_DATA_DIR%" --app-root "%DANQING_APP_ROOT%" --portable-python "%DANQING_PORTABLE_PYTHON%"
  ) else if /I "%MODE%"=="reinstall" (
    "%PY_BOOT%" "%BOOTSTRAP_PY%" --reinstall --yes --data-dir "%DANQING_USER_DATA_DIR%" --app-root "%DANQING_APP_ROOT%" --portable-python "%DANQING_PORTABLE_PYTHON%"
  ) else (
    "%PY_BOOT%" "%BOOTSTRAP_PY%" --data-dir "%DANQING_USER_DATA_DIR%" --app-root "%DANQING_APP_ROOT%" --portable-python "%DANQING_PORTABLE_PYTHON%"
  )
  if errorlevel 1 exit /b 1
)

set "VENV_PY=%DANQING_USER_DATA_DIR%\runtime-venv\Scripts\python.exe"
cd /d "%DANQING_APP_ROOT%"
"%VENV_PY%" -m uvicorn backend.main:app --host "%DANQING_HTTP_HOST%" --port "%DANQING_HTTP_PORT%"
endlocal
"""

_SERVER_README = """Danmo Make — CUDA server (thin bundle)
======================================

This archive does NOT include PyTorch. On first start, run.sh / run.bat downloads
and installs CUDA Python wheels into $DANQING_USER_DATA_DIR/runtime-venv with
console progress.

Contents:
  runtime/python/   Portable CPython
  runtime/app/      Application code + web UI
  run.sh | run.bat  Start API (auto bootstrap if needed)

Requirements:
  - x86_64 Linux or Windows 10/11
  - NVIDIA driver (CUDA GPU required; no CPU fallback)
  - Network on first run (~1.5GB+ download)

Quick start:
  export DANQING_USER_DATA_DIR=$HOME/.danmo-make   # optional
  export DANQING_PIP_MIRROR=tuna                   # optional: official|tuna|aliyun
  ./run.sh

Repair / reinstall:
  ./run.sh --repair-runtime
  ./run.sh --reinstall-runtime
  ./run.sh --status-runtime

Environment:
  DANQING_USER_DATA_DIR           Writable data root
  DANQING_HTTP_HOST / _PORT       Bind (default 0.0.0.0:7800)
  DANQING_PIP_MIRROR              official|tuna|aliyun
  DANQING_RUNTIME_SKIP_AUTO_SETUP Set to 1 to refuse start when not ready
  DANQING_RUNTIME_QUIET           Set to 1 to reduce console progress (still logs)

Open http://127.0.0.1:7800 — API docs at /docs
"""

_DESKTOP_README = """Danmo Make — Windows CUDA desktop (thin portable)
=================================================

This zip does NOT include PyTorch. On first launch the app opens a setup wizard
that downloads CUDA wheels into your app data directory (visible progress).

Contents:
  danqing-desktop.exe   Tauri shell
  runtime/              Portable Python + application code

Requirements:
  - Windows 10/11 x64 + WebView2
  - NVIDIA driver (CUDA GPU required)
  - Network on first run

Quick start:
  Unzip to a short path (e.g. C:\\DanmoMake) and run danqing-desktop.exe

Repair / reinstall: Settings → Runtime environment

Optional mirror: set DANQING_PIP_MIRROR=tuna before launch (or pick in UI).
"""


def _release_version(explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lstrip("v")
    env = os.environ.get("RELEASE_VERSION", "").strip()
    if env:
        return env.lstrip("v")
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=op.PROJECT_ROOT,
            text=True,
        ).strip()
        return out.lstrip("v")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "dev"


def _ensure_runtime(platform: str) -> Path:
    runtime = op.OUT_ROOT / "runtime"
    if not (runtime / "python").is_dir() or not (runtime / "app" / "backend").is_dir():
        stage.stage_runtime(platform=platform, dest=runtime)
    return runtime


def package_linux_server(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    runtime = _ensure_runtime("linux-x86_64")
    dist_root = op.OUT_ROOT / "dist"
    dist_root.mkdir(parents=True, exist_ok=True)
    bundle_name = f"danmo-make-linux-cuda-x86_64-{ver}"
    staging = dist_root / bundle_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(runtime, staging / "runtime")
    run_sh = staging / "run.sh"
    run_sh.write_text(_RUN_SH, encoding="utf-8")
    run_sh.chmod(0o755)
    (staging / "README.txt").write_text(_SERVER_README, encoding="utf-8")
    # Convenience CLI wrapper
    setup = staging / "danqing-runtime-setup"
    setup.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ROOT="$(cd "$(dirname "$0")" && pwd)"\n'
        'export DANQING_USER_DATA_DIR="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}"\n'
        'exec "$ROOT/runtime/python/bin/python3" "$ROOT/runtime/app/scripts/runtime_bootstrap.py" \\\n'
        '  --data-dir "$DANQING_USER_DATA_DIR" --app-root "$ROOT/runtime/app" \\\n'
        '  --portable-python "$ROOT/runtime/python" "$@"\n',
        encoding="utf-8",
    )
    setup.chmod(0o755)

    archive = dist_root / f"{bundle_name}.tar.gz"
    if archive.exists():
        archive.unlink()
    subprocess.run(
        ["tar", "-C", str(dist_root), "-czf", str(archive), bundle_name],
        check=True,
    )
    print("Release archive:", archive)
    return archive


def package_windows_server(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    runtime = _ensure_runtime("windows-x86_64")
    dist_root = op.OUT_ROOT / "dist"
    dist_root.mkdir(parents=True, exist_ok=True)
    bundle_name = f"danmo-make-windows-cuda-x86_64-{ver}"
    staging = dist_root / bundle_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(runtime, staging / "runtime")
    (staging / "run.bat").write_text(_RUN_BAT, encoding="utf-8", newline="\r\n")
    (staging / "README.txt").write_text(_SERVER_README, encoding="utf-8", newline="\r\n")
    archive = dist_root / f"{bundle_name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(staging)))
    print("Release archive:", archive)
    return archive


def _find_windows_shell_exe() -> Path:
    target = op.DESKTOP_CARGO_TARGET / "x86_64-pc-windows-msvc" / "release"
    exe = target / "danqing-desktop.exe"
    if exe.is_file():
        return exe
    fallback = op.DESKTOP_CARGO_TARGET / "release" / "danqing-desktop.exe"
    if fallback.is_file():
        return fallback
    raise SystemExit(
        f"Missing danqing-desktop.exe under {target}.\n"
        "Run: python scripts/tauri_build.py --platform windows"
    )


def package_windows_desktop(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    exe = _find_windows_shell_exe()
    runtime = _ensure_runtime("windows-x86_64")
    # Prefer Tauri-staged runtime if present
    staged = op.PROJECT_ROOT / "desktop" / "src-tauri" / "runtime"
    if (staged / "app" / "backend").is_dir():
        runtime = staged

    zip_dir = op.DESKTOP_BUNDLE_DIR / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    staging = op.DESKTOP_BUNDLE_DIR / "portable-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(exe, staging / "danqing-desktop.exe")
    for dll in exe.parent.glob("*.dll"):
        shutil.copy2(dll, staging / dll.name)
    shutil.copytree(runtime, staging / "runtime")
    (staging / "README.txt").write_text(_DESKTOP_README, encoding="utf-8", newline="\r\n")

    archive = zip_dir / f"DanmoMake_{ver}_x64-portable.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(staging)))
    shutil.rmtree(staging, ignore_errors=True)
    size_mb = archive.stat().st_size / (1024**2)
    print(f"Portable Windows desktop zip -> {archive} ({size_mb:.1f} MiB)")
    return archive


def main() -> int:
    p = argparse.ArgumentParser(description="Package thin CUDA archives")
    p.add_argument(
        "--product",
        choices=("linux-server", "windows-server", "windows-desktop"),
        required=True,
    )
    p.add_argument("--version", default=None)
    args = p.parse_args()
    if args.product == "linux-server":
        package_linux_server(version=args.version)
    elif args.product == "windows-server":
        package_windows_server(version=args.version)
    else:
        package_windows_desktop(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
