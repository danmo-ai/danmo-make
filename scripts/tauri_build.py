#!/usr/bin/env python3
"""Tauri release builds — macOS (MLX Metal), Linux (MLX / mlx[cuda]). Windows unsupported."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402
import prepare_tauri_resources as prep  # noqa: E402


def _python() -> str:
    venv_py = op.PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3")
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _npm() -> str:
    """Resolve npm for subprocess (Windows needs npm.cmd; bare 'npm' is not CreateProcess-able)."""
    if sys.platform == "win32":
        found = shutil.which("npm.cmd") or shutil.which("npm")
        if not found:
            raise RuntimeError("npm not found on PATH (install Node.js / run actions/setup-node)")
        return found
    found = shutil.which("npm")
    if not found:
        raise RuntimeError("npm not found on PATH")
    return found


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or op.PROJECT_ROOT, env=merged, check=True)


def _maybe_set_desktop_version() -> None:
    script = op.PROJECT_ROOT / "scripts" / "set_desktop_version.py"
    ver = os.environ.get("DANQING_DESKTOP_VERSION", "").strip()
    if ver:
        _run([_python(), str(script), ver])
        return
    try:
        subprocess.check_call(
            ["git", "describe", "--exact-match", "--tags", "HEAD"],
            cwd=op.PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("==> Desktop version from git tag")
        _run([_python(), str(script)])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def build_macos() -> None:
    if sys.platform != "darwin":
        raise SystemExit("macOS desktop build must run on Darwin.")
    import platform

    if platform.machine().lower() != "arm64":
        raise SystemExit("Danmo Make macOS desktop requires Apple Silicon (arm64).")

    bash = op.PROJECT_ROOT / "scripts" / "tauri_build_macos.sh"
    _run(["bash", str(bash)])


def build_linux() -> None:
    if sys.platform != "linux":
        raise SystemExit("Linux desktop build must run on Linux.")
    import platform

    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("Danmo Make Linux desktop currently supports x86_64 only.")

    cargo_target = os.environ.get("CARGO_TARGET_DIR", str(op.DESKTOP_CARGO_TARGET))
    os.environ["CARGO_TARGET_DIR"] = cargo_target
    op.DESKTOP_CARGO_TARGET.mkdir(parents=True, exist_ok=True)

    _maybe_set_desktop_version()
    prep.prepare()

    _run(["rustup", "target", "add", "x86_64-unknown-linux-gnu"])

    desktop = op.PROJECT_ROOT / "desktop"
    npm = _npm()
    _run([npm, "install"], cwd=desktop)
    # AppImage + deb (matches tauri.linux.conf.json / danmo-work)
    _run(
        [
            npm,
            "exec",
            "tauri",
            "build",
            "--",
            "--target",
            "x86_64-unknown-linux-gnu",
            "-b",
            "appimage",
            "-b",
            "deb",
        ],
        cwd=desktop,
    )

    _run([_python(), str(op.PROJECT_ROOT / "scripts" / "stage_desktop_bundle.py")])


def build_windows(*, thin_runtime: bool = True) -> None:
    _ = thin_runtime
    raise SystemExit("Windows temporarily unsupported")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tauri desktop release build")
    parser.add_argument(
        "--platform",
        choices=("macos", "linux", "windows"),
        required=True,
        help="Target desktop platform",
    )
    parser.add_argument(
        "--thin-runtime",
        action="store_true",
        default=True,
        help="Linux: stage thin MLX runtime (default)",
    )
    parser.add_argument(
        "--legacy-sidecar",
        action="store_true",
        help="Unused (Windows unsupported)",
    )
    args = parser.parse_args()
    if args.platform == "macos":
        build_macos()
    elif args.platform == "linux":
        build_linux()
    else:
        build_windows(thin_runtime=not args.legacy_sidecar)


if __name__ == "__main__":
    main()
