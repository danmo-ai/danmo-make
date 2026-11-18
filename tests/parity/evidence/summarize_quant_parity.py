"""Condense the published-quant (mflux mlx-q4/q8) parity reports into a
committable evidence summary. Full fixtures live under gitignored out/.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FIXTURES = {
    "q4-16": "parity-zimage-q4-16.json",
    "q4-768x1280": "parity-zimage-q4-768x1280.json",
    "q8-16": "parity-zimage-q8-16.json",
    "q8-768x1280": "parity-zimage-q8-768x1280.json",
}
MEMORY = {
    "q4-768x1280": "parity-zimage-q4-768-memory.json",
    "q8-768x1280": "parity-zimage-q8-768-memory.json",
}


def main():
    out = {"date": "2026-11-18", "bundles": {}}
    for label, name in FIXTURES.items():
        data = (ROOT / "out" / name).read_bytes()
        payload = json.loads(data)
        entry = {
            "fixture_sha256": hashlib.sha256(data).hexdigest(),
            "width": payload["width"],
            "height": payload["height"],
            "steps": payload["steps"],
            "quantization_bits": payload["quantization_bits"],
            "quantization_group_size": payload.get("quantization_group_size"),
            "teacache": payload["teacache"],
            "cache_counts": payload.get("cache_counts"),
            "main_timings": payload.get("timings"),
            "main_memory": payload.get("memory"),
        }
        out["bundles"][label] = entry
    for label, name in MEMORY.items():
        data = (ROOT / "out" / name).read_bytes()
        report = json.loads(data)
        report["report_sha256"] = hashlib.sha256(data).hexdigest()
        report["peak_ratio"] = round(report["peak_memory"] / report["main_peak_memory"], 4)
        report["active_ratio"] = round(report["active_memory"] / report["main_active_memory"], 4)
        out["bundles"][label]["rust_memory"] = report
    target = Path(__file__).parent / "zimage-published-quant-parity-2026-11-18.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
