#!/usr/bin/env python3
"""Stage ``out/sidecar/danqing-api`` OR produce thin Linux CUDA server ``.tar.gz``.

Default: thin bundle (portable Python + app, first-run bootstrap).
Legacy PyInstaller onedir: ``--legacy-sidecar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import package_cuda_thin as thin  # noqa: E402
import package_linux_cuda_release_legacy as legacy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Linux CUDA server tar.gz")
    parser.add_argument("--version", help="Release version (default: RELEASE_VERSION or git describe)")
    parser.add_argument(
        "--legacy-sidecar",
        action="store_true",
        help="Use PyInstaller onedir sidecar (large offline bundle)",
    )
    args = parser.parse_args()
    if args.legacy_sidecar:
        legacy.package(version=args.version)
    else:
        thin.package_linux_server(version=args.version)


if __name__ == "__main__":
    main()
