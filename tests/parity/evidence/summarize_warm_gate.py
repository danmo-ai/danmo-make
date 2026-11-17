"""Condense the formal warm-gate reports into a committable evidence summary.

The full fixtures live under gitignored out/; this keeps the p50 numbers,
per-pass timings, memory metrics, and full-fixture SHA-256 in the repo.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = [
    "parity-zimage-native-default-teacache-rust-warm.json",
    "parity-zimage-native-default-teacache-main-warm.json",
]


def main():
    out = {}
    for name in NAMES:
        data = (ROOT / "out" / name).read_bytes()
        payload = json.loads(data)
        summary = {
            key: payload[key]
            for key in ("width", "height", "steps", "teacache", "native_dtype", "warm_p50", "memory")
            if key in payload
        }
        summary["passes"] = [
            {key: value for key, value in entry.items() if key != "steps"}
            for entry in payload["passes"]
        ]
        if "rust_to_main_ratio" in payload:
            for key in ("rust_to_main_ratio", "peak_memory", "steady_active_memory", "dit_weight_bytes"):
                summary[key] = payload[key]
        else:
            summary["cache_counts"] = payload.get("cache_counts")
        summary["fixture_sha256"] = hashlib.sha256(data).hexdigest()
        out[name] = summary
    target = Path(__file__).parent / "zimage-native-default-teacache-warm-gate-2026-11-18.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
