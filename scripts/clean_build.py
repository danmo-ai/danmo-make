#!/usr/bin/env python3
"""Remove unified ``out/`` and staged Tauri sidecar resources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Danmo Make build outputs")
    parser.add_argument(
        "--keep-frontend",
        action="store_true",
        help="Only remove sidecar/desktop/pyinstaller under out/, keep out/frontend/dist",
    )
    args = parser.parse_args()
    removed = op.clean_build_artifacts(include_frontend=not args.keep_frontend)
    if not removed:
        print("Nothing to clean.")
        return
    print("Removed:")
    for path in removed:
        try:
            rel = path.relative_to(op.PROJECT_ROOT)
        except ValueError:
            rel = path
        print(f"  {rel}")


if __name__ == "__main__":
    main()
