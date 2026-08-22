#!/usr/bin/env python3
"""Run MiniMax-H3 smoke benchmark case (512², 22f, 8 steps) and record wall time."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "tests" / "benchmark" / "cases" / "minimax_h3_smoke.yaml"
OUT_DIR = ROOT / "tests" / "benchmark" / "outputs" / "h3"
DEFAULT_OUT = OUT_DIR / "minimax_h3_smoke.mp4"
REPORT_PATH = OUT_DIR / "minimax_h3_smoke_report.json"


def _load_case() -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    if not CASE_PATH.is_file():
        raise RuntimeError(f"Missing benchmark case: {CASE_PATH}")
    data = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid case YAML in {CASE_PATH}")
    return data


def main() -> int:
    case = _load_case()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(os.environ.get("H3_SMOKE_OUTPUT", str(DEFAULT_OUT)))
    cli = ROOT / "bin" / "danqing-video-generate"
    py = ROOT / ".venv" / "bin" / "python3"
    if not cli.is_file():
        raise RuntimeError(f"CLI not found: {cli}")

    cmd = [
        str(py if py.is_file() else sys.executable),
        str(cli),
        "--model",
        str(case.get("model", "minimax-h3-fl2va")),
        "--prompt",
        str(case["prompt"]),
        "--width",
        str(case.get("width", 512)),
        "--height",
        str(case.get("height", 512)),
        "--num-frames",
        str(case.get("num_frames", 22)),
        "--steps",
        str(case.get("steps", 8)),
        "--seed",
        str(case.get("seed", 42)),
        "--output",
        str(out_path),
    ]
    version = case.get("version")
    if version:
        cmd.extend(["--version", str(version)])

    print("Running:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.perf_counter() - t0

    report = {
        "case_id": case.get("id", "minimax-h3-smoke"),
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "wall_sec": round(elapsed, 3),
        "output_path": str(out_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": case.get("notes", ""),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
