#!/usr/bin/env python3
"""Package Windows CUDA server zip (thin by default; ``--legacy-sidecar`` for PyInstaller)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import package_cuda_thin as thin  # noqa: E402
import package_windows_cuda_release_legacy as legacy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Windows CUDA server zip")
    parser.add_argument("--version", help="Release version")
    parser.add_argument("--legacy-sidecar", action="store_true")
    args = parser.parse_args()
    if args.legacy_sidecar:
        legacy.package(version=args.version)
    else:
        thin.package_windows_server(version=args.version)


if __name__ == "__main__":
    main()
