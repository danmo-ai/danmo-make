#!/usr/bin/env python3
"""Download astral python-build-standalone into out/runtime/python/."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402

# Pinned release — bump intentionally when upgrading CPython.
PBS_TAG = "20250409"
PBS_VERSION = "3.11.12"
PBS_BASE = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_TAG}"

_ARTIFACTS = {
    "linux-x86_64": (
        f"cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "python",
    ),
    "windows-x86_64": (
        f"cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "python",
    ),
}


def _platform_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    if sys.platform == "win32":
        return "windows-x86_64"
    raise SystemExit(
        f"Unsupported host platform {sys.platform!r} for portable CPython "
        "(use --platform linux-x86_64|windows-x86_64)."
    )


def _download(url: str, dest: Path, *, progress: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "danmo-make-fetch-python/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress and total:
                    pct = 100.0 * done / total
                    print(f"\r  download {done}/{total} ({pct:.1f}%)", end="", flush=True)
        if progress:
            print()


def fetch(*, platform: str | None = None, dest: Path | None = None, force: bool = False) -> Path:
    key = _platform_key(platform)
    if key not in _ARTIFACTS:
        raise SystemExit(f"Unknown platform {key!r}; choose from {sorted(_ARTIFACTS)}")
    name, inner_root = _ARTIFACTS[key]
    url = f"{PBS_BASE}/{name}"
    out = dest or (op.OUT_ROOT / "runtime" / "python")
    marker = out / ".pbs-version"
    expected = f"{PBS_TAG}:{PBS_VERSION}:{key}"
    if out.is_dir() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected and not force:
        print(f"Portable Python already present: {out}")
        return out

    cache = op.OUT_ROOT / "runtime" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name
    if force or not archive.is_file():
        print(f"==> Fetch {url}")
        _download(url, archive)

    staging = Path(tempfile.mkdtemp(prefix="dq-pbs-"))
    try:
        print(f"==> Extract {archive.name}")
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(staging)
        # install_only layout: <staging>/python/...
        src = staging / inner_root
        if not src.is_dir():
            candidates = [p for p in staging.iterdir() if p.is_dir()]
            if len(candidates) == 1:
                src = candidates[0]
            else:
                raise SystemExit(f"Unexpected archive layout under {staging}")
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, out)
        marker.write_text(expected + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Sanity
    if key.startswith("windows"):
        exe = out / "python.exe"
    else:
        exe = out / "bin" / "python3"
        if not exe.is_file():
            exe = out / "bin" / "python"
    if not exe.is_file():
        raise SystemExit(f"Portable Python missing interpreter at {exe}")
    print(f"Portable Python ready: {out} ({exe.name})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch portable CPython for CUDA thin bundles")
    p.add_argument("--platform", choices=sorted(_ARTIFACTS), default=None)
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    fetch(platform=args.platform, dest=args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
