#!/usr/bin/env python3
"""Fetch Danmo Make task metadata + diagnostic bundle for AI/human troubleshooting."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

TASK_ID_RE = re.compile(r"^tsk_[0-9a-f]{24}$")
GRAPH_STEP_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def _base_url(arg_base: str | None) -> str:
    if arg_base:
        return arg_base.rstrip("/")
    host = os.environ.get("DANQING_HTTP_HOST", "127.0.0.1")
    port = os.environ.get("DANQING_HTTP_PORT", "7800")
    return f"http://{host}:{port}"


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_graph_steps(logs: list[dict]) -> list[dict]:
    steps: list[dict] = []
    for row in logs:
        msg = str(row.get("message") or "").strip()
        m = GRAPH_STEP_RE.match(msg)
        if not m:
            continue
        steps.append(
            {
                "node": m.group(1),
                "detail": (m.group(2) or "").strip(),
                "level": row.get("level"),
                "time": row.get("time"),
            }
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a Danmo Make generation task by ID")
    parser.add_argument("task_id", help="Task ID, e.g. tsk_abcd…")
    parser.add_argument("--base", default=None, help="API base URL (default http://HOST:PORT)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Skip GET /diagnostic; assemble bundle from task + logs only",
    )
    args = parser.parse_args()

    task_id = args.task_id.strip()
    if not TASK_ID_RE.match(task_id):
        print(f"Invalid task_id format: {task_id!r} (expected tsk_ + 24 hex)", file=sys.stderr)
        return 2

    base = _base_url(args.base)
    try:
        diagnostic = None
        if not args.legacy:
            try:
                diagnostic = _get(f"{base}/api/tasks/{task_id}/diagnostic")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise

        task = _get(f"{base}/api/tasks/{task_id}")
        logs_payload = _get(f"{base}/api/tasks/{task_id}/logs?limit=2000")
        health = None
        try:
            health = _get(f"{base}/api/system/health")
        except urllib.error.HTTPError:
            pass
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {e.reason}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Cannot reach API at {base}: {e.reason}", file=sys.stderr)
        return 1

    items = logs_payload.get("items") if isinstance(logs_payload, dict) else logs_payload
    if not isinstance(items, list):
        items = []

    if diagnostic is not None:
        bundle = {
            "source": "diagnostic_api",
            "api_base": base,
            **diagnostic,
        }
    else:
        graph_steps = _parse_graph_steps(items)
        errors = [r for r in items if str(r.get("level", "")).lower() == "error"]
        bundle = {
            "source": "legacy_logs",
            "task_id": task_id,
            "api_base": base,
            "task": task,
            "log_count": len(items),
            "graph_steps": graph_steps,
            "errors": errors[-10:],
            "last_graph_node": graph_steps[-1]["node"] if graph_steps else None,
            "health": health,
            "hints": {
                "work_dir": f"outputs/work/{task_id}",
                "logs_api": f"{base}/api/tasks/{task_id}/logs",
                "diagnostic_api": f"{base}/api/tasks/{task_id}/diagnostic",
            },
        }

    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0

    status = (diagnostic or task).get("status", task.get("status", "?"))
    kind = (diagnostic or task).get("kind", task.get("kind", "?"))
    model = (
        (diagnostic or {}).get("model_id")
        or task.get("model_id")
        or (task.get("model") or {}).get("id")
        or "?"
    )
    err = (
        ((diagnostic or {}).get("failure") or {}).get("detail")
        or task.get("error")
        or task.get("error_message")
        or ""
    )

    print(f"Task: {task_id}")
    print(f"Status: {status}  Kind: {kind}  Model: {model}")
    if bundle.get("source") == "diagnostic_api" and diagnostic:
        fail = diagnostic.get("failure") or {}
        if fail.get("code"):
            print(f"Failure code: {fail.get('code')}  Span: {fail.get('span_name')}")
        if fail.get("hints"):
            print("Hints:")
            for h in fail["hints"]:
                print(f"  - {h}")
    if err:
        print(f"Error: {err}")
    print(f"Logs: {len(items)} lines")
    if diagnostic and diagnostic.get("graph"):
        nodes = diagnostic["graph"].get("nodes") or []
        if nodes:
            print("\nPipeline graph:")
            for n in nodes:
                if n.get("status") != "pending":
                    print(f"  {n.get('id')}: {n.get('status')}")
    elif bundle.get("graph_steps"):
        print("\nPipeline nodes (legacy log parse):")
        for s in bundle["graph_steps"]:
            detail = f" — {s['detail']}" if s.get("detail") else ""
            print(f"  [{s['node']}]{detail}")
    if health:
        backends = health.get("backends") if isinstance(health, dict) else None
        if backends:
            print(f"\nHealth backends: {backends}")
    print(f"\nWork dir: outputs/work/{task_id}")
    print(f"Diagnostic: {base}/api/tasks/{task_id}/diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
