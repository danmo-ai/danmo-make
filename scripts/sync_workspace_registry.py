#!/usr/bin/env python3
"""Copy factory models_registry.json into the effective workspace config/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.catalog.migrate_v2 import migrate_v2_to_v3
from backend.catalog.schema_v3 import SCHEMA_VERSION_V3
from backend.utils.config_paths import resolve_default_config_root
from backend.utils.user_home import ensure_control_plane
from backend.utils.workspace import resolve_workspace_root

DEFAULT_REGISTRY = ROOT / "default_config" / "models_registry.json"


def _prepare_registry_payload(src: Path) -> dict:
    data = json.loads(src.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 2)) >= SCHEMA_VERSION_V3:
        return data
    migrated, report = migrate_v2_to_v3(data)
    for line in report:
        print(f"  migrate: {line}")
    return migrated


def main() -> int:
    if not DEFAULT_REGISTRY.is_file():
        print(f"Missing factory registry: {DEFAULT_REGISTRY}", file=sys.stderr)
        return 1
    default_cfg = resolve_default_config_root(bootstrap_root=ROOT, bundle_root=None)
    control = ensure_control_plane()
    workspace = resolve_workspace_root(
        ROOT, control_plane=control, legacy_default_config=default_cfg
    )
    dst_dir = workspace / "config"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "models_registry.json"

    factory_ver = int(json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8")).get("schema_version", 2))
    if factory_ver >= SCHEMA_VERSION_V3:
        shutil.copy2(DEFAULT_REGISTRY, dst)
        print(f"Synced {DEFAULT_REGISTRY.name} -> {dst}")
        return 0

    payload = _prepare_registry_payload(DEFAULT_REGISTRY)
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Synced + migrated v2→v3 {DEFAULT_REGISTRY.name} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
