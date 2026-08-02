#!/usr/bin/env python3
"""Shared CUDA runtime bootstrap (desktop Tauri + headless server).

Creates/repairs a venv under ``data_dir/runtime-venv``, installs torch + app
requirements with visible progress, verifies CUDA, writes ``runtime-env.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ProgressCb = Callable[[dict], None]

ENV_JSON_NAME = "runtime-env.json"
SETUP_JSON_NAME = "runtime-setup.json"
VENV_DIR_NAME = "runtime-venv"
LOG_NAME = "runtime-setup.log"

MIRRORS: dict[str, dict[str, str]] = {
    "official": {
        "torch_index": "https://download.pytorch.org/whl/cu124",
        "pip_index": "https://pypi.org/simple",
    },
    "tuna": {
        "torch_index": "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124",
        "pip_index": "https://pypi.tuna.tsinghua.edu.cn/simple",
    },
    "aliyun": {
        "torch_index": "https://mirrors.aliyun.com/pytorch-wheels/cu124",
        "pip_index": "https://mirrors.aliyun.com/pypi/simple",
    },
}

_MIN_FREE_BYTES = 8 * 1024**3  # 8 GiB soft check


class BootstrapError(RuntimeError):
    """Fail-loud bootstrap failure."""


@dataclass
class BootstrapPaths:
    data_dir: Path
    app_root: Path
    portable_python: Path
    venv_dir: Path
    env_json: Path
    setup_json: Path
    log_path: Path

    @classmethod
    def resolve(
        cls,
        *,
        data_dir: Path,
        app_root: Path,
        portable_python: Path,
    ) -> "BootstrapPaths":
        data = data_dir.expanduser().resolve()
        data.mkdir(parents=True, exist_ok=True)
        logs = data / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data,
            app_root=app_root.expanduser().resolve(),
            portable_python=portable_python.expanduser().resolve(),
            venv_dir=data / VENV_DIR_NAME,
            env_json=data / ENV_JSON_NAME,
            setup_json=data / SETUP_JSON_NAME,
            log_path=logs / LOG_NAME,
        )


def _emit(cb: ProgressCb | None, **payload: object) -> None:
    if cb:
        cb(dict(payload))


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def requirements_hash(app_root: Path) -> str:
    parts: list[str] = []
    for name in ("requirements-torch-cuda.txt", "requirements-cuda.txt", "requirements.txt"):
        p = app_root / name
        if p.is_file():
            parts.append(f"{name}:{_file_sha256(p)}")
    if not parts:
        raise BootstrapError(f"No requirements files under {app_root}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def portable_python_exe(portable_root: Path) -> Path:
    if sys.platform == "win32" or (portable_root / "python.exe").is_file():
        exe = portable_root / "python.exe"
    else:
        exe = portable_root / "bin" / "python3"
        if not exe.is_file():
            exe = portable_root / "bin" / "python"
    if not exe.is_file():
        raise BootstrapError(f"Portable Python not found under {portable_root}")
    return exe


def venv_python(venv_dir: Path) -> Path:
    if (venv_dir / "Scripts" / "python.exe").is_file():
        return venv_dir / "Scripts" / "python.exe"
    for name in ("python3", "python"):
        p = venv_dir / "bin" / name
        if p.is_file():
            return p
    raise BootstrapError(f"venv interpreter missing under {venv_dir}")


def load_mirror_preference(paths: BootstrapPaths) -> str:
    env = os.environ.get("DANQING_PIP_MIRROR", "").strip().lower()
    if env in MIRRORS:
        return env
    if paths.setup_json.is_file():
        try:
            data = json.loads(paths.setup_json.read_text(encoding="utf-8"))
            mid = str(data.get("mirror", "")).strip().lower()
            if mid in MIRRORS:
                return mid
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return "official"


def save_mirror_preference(paths: BootstrapPaths, mirror: str) -> None:
    if mirror not in MIRRORS:
        raise BootstrapError(f"Unknown mirror {mirror!r}; choose from {sorted(MIRRORS)}")
    payload = {"mirror": mirror, "updated_at": int(time.time())}
    paths.setup_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_env_status(paths: BootstrapPaths) -> dict:
    ready = False
    detail: dict = {}
    if paths.env_json.is_file():
        try:
            detail = json.loads(paths.env_json.read_text(encoding="utf-8"))
            ready = bool(detail.get("ready"))
        except (OSError, json.JSONDecodeError):
            detail = {"error": f"corrupt {paths.env_json.name}"}
            ready = False
    req_hash = None
    try:
        req_hash = requirements_hash(paths.app_root)
    except BootstrapError as exc:
        detail.setdefault("requirements_error", str(exc))
    if ready and req_hash and detail.get("requirements_hash") != req_hash:
        ready = False
        detail["stale"] = True
        detail["expected_requirements_hash"] = req_hash
    mirror = load_mirror_preference(paths)
    return {
        "ready": ready,
        "mirror": mirror,
        "mirrors": sorted(MIRRORS),
        "data_dir": str(paths.data_dir),
        "venv_dir": str(paths.venv_dir),
        "env_json": str(paths.env_json),
        "log_path": str(paths.log_path),
        "app_root": str(paths.app_root),
        "portable_python": str(paths.portable_python),
        "requirements_hash": req_hash,
        "detail": detail,
    }


def _disk_free(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _ensure_disk(paths: BootstrapPaths, cb: ProgressCb | None) -> None:
    free = _disk_free(paths.data_dir)
    _emit(cb, phase="check", message=f"free_disk_bytes={free}", free_disk_bytes=free)
    if free < _MIN_FREE_BYTES:
        raise BootstrapError(
            f"Not enough free disk space under {paths.data_dir} "
            f"({free / 1024**3:.1f} GiB free; need >= {_MIN_FREE_BYTES / 1024**3:.0f} GiB)."
        )


def _run_logged(
    cmd: list[str],
    *,
    log_path: Path,
    cb: ProgressCb | None,
    phase: str,
    env: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
) -> None:
    _append_log(log_path, f"$ {' '.join(cmd)}")
    _emit(cb, phase=phase, message=" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if cancel and cancel.is_set():
            proc.kill()
            raise BootstrapError("Runtime install cancelled")
        line = line.rstrip()
        _append_log(log_path, line)
        _emit(cb, phase=phase, message=line)
    code = proc.wait()
    if code != 0:
        raise BootstrapError(f"Command failed ({code}): {' '.join(cmd)}")


def _create_venv(paths: BootstrapPaths, cb: ProgressCb | None, cancel: threading.Event | None) -> Path:
    py = portable_python_exe(paths.portable_python)
    if paths.venv_dir.exists():
        _emit(cb, phase="venv", message="reusing existing venv")
    else:
        _emit(cb, phase="venv", message=f"create venv with {py}")
        _run_logged(
            [str(py), "-m", "venv", str(paths.venv_dir)],
            log_path=paths.log_path,
            cb=cb,
            phase="venv",
            cancel=cancel,
        )
    return venv_python(paths.venv_dir)


def _pip_base(venv_py: Path, mirror: str) -> list[str]:
    indexes = MIRRORS[mirror]
    return [
        str(venv_py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--index-url",
        indexes["pip_index"],
    ]


def _install_torch(
    venv_py: Path,
    paths: BootstrapPaths,
    mirror: str,
    cb: ProgressCb | None,
    cancel: threading.Event | None,
) -> None:
    torch_req = paths.app_root / "requirements-torch-cuda.txt"
    if not torch_req.is_file():
        raise BootstrapError(f"Missing {torch_req}")
    torch_index = os.environ.get("DANQING_TORCH_INDEX_URL", "").strip() or MIRRORS[mirror]["torch_index"]
    _emit(cb, phase="torch", message=f"index={torch_index}")
    cmd = [
        str(venv_py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--index-url",
        torch_index,
        "-r",
        str(torch_req),
    ]
    _run_logged(cmd, log_path=paths.log_path, cb=cb, phase="torch", cancel=cancel)


def _install_app_reqs(
    venv_py: Path,
    paths: BootstrapPaths,
    mirror: str,
    cb: ProgressCb | None,
    cancel: threading.Event | None,
) -> None:
    cuda_req = paths.app_root / "requirements-cuda.txt"
    if not cuda_req.is_file():
        raise BootstrapError(f"Missing {cuda_req}")
    cmd = _pip_base(venv_py, mirror) + ["-r", str(cuda_req)]
    _run_logged(cmd, log_path=paths.log_path, cb=cb, phase="pip", cancel=cancel)


def _verify_cuda(venv_py: Path, paths: BootstrapPaths, cb: ProgressCb | None) -> dict:
    code = (
        "import json,torch;"
        "ok=bool(torch.cuda.is_available());"
        "info={"
        "'torch': getattr(torch,'__version__',None),"
        "'cuda_available': ok,"
        "'cuda_version': getattr(getattr(torch,'version',None),'cuda',None),"
        "'device_count': int(torch.cuda.device_count()) if ok else 0,"
        "'device_name': torch.cuda.get_device_name(0) if ok else None,"
        "};"
        "print(json.dumps(info))"
    )
    _emit(cb, phase="verify", message="import torch; torch.cuda.is_available()")
    proc = subprocess.run(
        [str(venv_py), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    _append_log(paths.log_path, proc.stdout or "")
    _append_log(paths.log_path, proc.stderr or "")
    if proc.returncode != 0:
        raise BootstrapError(
            "Failed to import torch in runtime venv.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    line = (proc.stdout or "").strip().splitlines()[-1]
    info = json.loads(line)
    if not info.get("cuda_available"):
        raise BootstrapError(
            "torch.cuda.is_available() is False. Install a compatible NVIDIA driver "
            "and ensure this machine has a CUDA GPU. Refusing CPU fallback."
        )
    _emit(cb, phase="verify", message=f"cuda ok: {info.get('device_name')}", detail=info)
    return info


def _write_env_json(paths: BootstrapPaths, *, mirror: str, torch_info: dict, mode: str) -> None:
    payload = {
        "ready": True,
        "mode": mode,
        "mirror": mirror,
        "python": str(venv_python(paths.venv_dir)),
        "venv_dir": str(paths.venv_dir),
        "app_root": str(paths.app_root),
        "requirements_hash": requirements_hash(paths.app_root),
        "torch": torch_info,
        "updated_at": int(time.time()),
    }
    paths.env_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_ready_marker(paths: BootstrapPaths) -> None:
    if paths.env_json.is_file():
        try:
            data = json.loads(paths.env_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        data["ready"] = False
        data["cleared_at"] = int(time.time())
        paths.env_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def wipe_venv(paths: BootstrapPaths, cb: ProgressCb | None = None) -> None:
    clear_ready_marker(paths)
    if paths.venv_dir.exists():
        _emit(cb, phase="wipe", message=f"remove {paths.venv_dir}")
        shutil.rmtree(paths.venv_dir)


def run_install(
    paths: BootstrapPaths,
    *,
    mode: str = "bootstrap",
    mirror: str | None = None,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    if mode not in ("bootstrap", "repair", "reinstall"):
        raise BootstrapError(f"Unknown mode {mode!r}")
    # Truncate log for this run
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("", encoding="utf-8")
    _append_log(paths.log_path, f"=== runtime bootstrap mode={mode} ===")

    mid = (mirror or load_mirror_preference(paths)).strip().lower()
    if mid not in MIRRORS:
        raise BootstrapError(f"Unknown mirror {mid!r}")
    save_mirror_preference(paths, mid)

    _emit(progress, phase="start", mode=mode, mirror=mid, message="starting")
    _ensure_disk(paths, progress)

    if mode == "reinstall":
        wipe_venv(paths, progress)

    venv_py = _create_venv(paths, progress, cancel)
    # Always refresh pip tooling
    _run_logged(
        [str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        log_path=paths.log_path,
        cb=progress,
        phase="pip",
        cancel=cancel,
    )
    _install_torch(venv_py, paths, mid, progress, cancel)
    _install_app_reqs(venv_py, paths, mid, progress, cancel)
    torch_info = _verify_cuda(venv_py, paths, progress)
    _write_env_json(paths, mirror=mid, torch_info=torch_info, mode=mode)
    status = read_env_status(paths)
    _emit(progress, phase="done", message="runtime ready", status=status)
    return status


def console_progress(quiet: bool = False) -> ProgressCb:
    def _cb(payload: dict) -> None:
        if quiet:
            return
        phase = payload.get("phase", "")
        msg = payload.get("message", "")
        print(f"[{phase}] {msg}", file=sys.stderr, flush=True)

    return _cb


def default_paths_from_env() -> BootstrapPaths:
    data = Path(os.environ.get("DANQING_USER_DATA_DIR") or Path.home() / "danqing-data")
    app_root = Path(os.environ.get("DANQING_APP_ROOT") or Path.cwd())
    portable = Path(os.environ.get("DANQING_PORTABLE_PYTHON") or (app_root / "runtime" / "python"))
    return BootstrapPaths.resolve(data_dir=data, app_root=app_root, portable_python=portable)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Danmo Make CUDA runtime bootstrap")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--app-root", type=Path, default=None)
    parser.add_argument("--portable-python", type=Path, default=None)
    parser.add_argument("--mirror", choices=sorted(MIRRORS), default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Required for --reinstall")
    parser.add_argument("--json", action="store_true", help="Print status/result as JSON")
    args = parser.parse_args(argv)

    base = default_paths_from_env()
    paths = BootstrapPaths.resolve(
        data_dir=args.data_dir or base.data_dir,
        app_root=args.app_root or base.app_root,
        portable_python=args.portable_python or base.portable_python,
    )

    quiet = os.environ.get("DANQING_RUNTIME_QUIET", "").strip() in ("1", "true", "yes")

    if args.status and not (args.repair or args.reinstall):
        status = read_env_status(paths)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"ready={status['ready']} mirror={status['mirror']}")
            print(f"data_dir={status['data_dir']}")
            print(f"log={status['log_path']}")
            if status.get("detail"):
                print(json.dumps(status["detail"], indent=2))
        return 0 if status["ready"] else 1

    mode = "bootstrap"
    if args.reinstall:
        if not args.yes:
            raise SystemExit("--reinstall requires --yes (destructive: deletes runtime-venv)")
        mode = "reinstall"
    elif args.repair:
        mode = "repair"

    try:
        result = run_install(
            paths,
            mode=mode,
            mirror=args.mirror,
            progress=console_progress(quiet=quiet),
        )
    except BootstrapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"See log: {paths.log_path}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Runtime environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
