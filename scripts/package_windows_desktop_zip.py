#!/usr/bin/env python3
"""Package Windows Tauri desktop + thin CUDA runtime as a portable ``.zip``.

Default is thin (no torch in zip). Legacy sidecar zip: use ``--legacy-sidecar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import package_cuda_thin as thin  # noqa: E402
import package_windows_desktop_zip_legacy as legacy  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Package Windows CUDA desktop portable zip")
    p.add_argument("--version", help="X.Y.Z (default: RELEASE_VERSION or git describe)")
    p.add_argument(
        "--legacy-sidecar",
        action="store_true",
        help="Bundle PyInstaller danqing-api (large offline)",
    )
    args = p.parse_args()
    if args.legacy_sidecar:
        legacy.package(version=args.version)
    else:
        thin.package_windows_desktop(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
