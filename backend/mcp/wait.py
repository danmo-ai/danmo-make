"""Poll task status until terminal or timeout."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.mcp.bridge import MakeAPIBridge

_TERMINAL = frozenset({"completed", "failed", "cancelled"})


async def wait_task(
    bridge: MakeAPIBridge,
    task_id: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval: float = 1.5,
) -> dict[str, Any]:
    """Poll ``GET /api/tasks/{id}`` until terminal status or timeout (fail loud)."""
    tid = (task_id or "").strip()
    if not tid:
        raise RuntimeError("wait_task: task_id is required")
    deadline = asyncio.get_event_loop().time() + max(1.0, float(timeout_seconds))
    last: dict[str, Any] | None = None
    while True:
        raw = await bridge.get(f"/api/tasks/{tid}")
        if not isinstance(raw, dict):
            raise RuntimeError(f"wait_task: unexpected response for {tid}: {raw!r}")
        last = raw
        status = str(raw.get("status") or "")
        if status in _TERMINAL:
            return raw
        if asyncio.get_event_loop().time() >= deadline:
            return {
                "status": "timeout",
                "task_id": tid,
                "message": f"wait_task timed out after {timeout_seconds}s",
                "last": last,
            }
        await asyncio.sleep(poll_interval)
