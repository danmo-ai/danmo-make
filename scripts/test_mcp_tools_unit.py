#!/usr/bin/env python3
"""Smoke: Danmo Make MCP tool registration (no GPU / no live API)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp.server import create_mcp


EXPECTED = {
    "health",
    "list_models",
    "get_model",
    "upload_asset",
    "get_asset",
    "get_task",
    "wait_task",
    "cancel_task",
    "diagnose_task",
    "generate_image",
    "edit_image",
    "upscale_image",
    "generate_video",
    "edit_video",
    "upscale_video",
    "generate_audio",
    "edit_audio",
}


def main() -> int:
    mcp = create_mcp()
    names = set(mcp._tool_manager._tools.keys())
    missing = EXPECTED - names
    extra = names - EXPECTED
    if missing:
        print("MISSING tools:", sorted(missing), file=sys.stderr)
        return 1
    if extra:
        print("UNEXPECTED tools:", sorted(extra), file=sys.stderr)
        return 1
    print(f"ok: {len(names)} MCP tools registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
