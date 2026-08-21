#!/usr/bin/env python3
"""Scaffold a Shape-A (DiT + ImagePipeline) image family directory."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"


def _snake(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.strip("_").lower()


def _pascal(s: str) -> str:
    return "".join(part.capitalize() for part in _snake(s).split("_") if part)


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip existing {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, help="Registry family id (e.g. my_family)")
    ap.add_argument("--class", dest="class_name", default="", help="Transformer class name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    family = _snake(args.family)
    cls_name = args.class_name or _pascal(family) + "Transformer"
    family_dir = FAMILIES / family
    if family_dir.exists() and not args.force:
        print(f"FAIL: {family_dir} already exists (use --force)", file=sys.stderr)
        return 1

    _write(
        family_dir / "transformer.py",
        f'''"""{cls_name} — public stem; MLX in ``transformer_mlx``."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class {cls_name}(DelegatingDiTStem):
    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import {cls_name} as _MLX

        super().__init__(config, ctx, mlx_cls=_MLX)
''',
        force=args.force,
    )
    _write(
        family_dir / "transformer_mlx.py",
        f'''"""{cls_name} MLX implementation."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class {cls_name}(TransformerBase):
    def __init__(self, config: Any, ctx: Any):
        super().__init__(config, ctx)
        self._param_map: dict[str, Any] = {{}}

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Transform checkpoint keys to match ``_param_map``."""
        from backend.engine.families.{family}.weights import remap_{family}_weights
        return remap_{family}_weights(weights)

    def forward(self, latents, t, *, txt_embeds=None, **kwargs):
        raise NotImplementedError("{cls_name}.forward is not implemented yet")

    def parameters(self):
        return self._param_map
''',
        force=args.force,
    )
    _write(
        family_dir / "weights.py",
        f'''"""Weight remap for {family}."""
from __future__ import annotations


def remap_{family}_weights(weights: dict) -> dict:
    remapped = {{}}
    for key, tensor in weights.items():
        remapped[key] = tensor
    return remapped
''',
        force=args.force,
    )
    build_fn = f"build_{family}_plugin"
    register_fn = f"register_{family}_plugin"
    _write(
        family_dir / "plugin.py",
        f'''"""{cls_name} v3 ``FamilyPlugin`` factory."""

from __future__ import annotations

from pathlib import Path

from backend.engine.config.model_configs import get_config_class
from backend.engine.families._image_backbone import ImagePluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family


def {build_fn}(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("{family}")()
    spec = family_spec_from_config("{family}", config, media="image")
    return FamilyPlugin(
        family_id="{family}",
        spec=spec,
        backbone=ImagePluginBackbone(spec),
    )


def {register_fn}() -> None:
    register_family("{family}", {build_fn})
''',
        force=args.force,
    )

    print("\nRegistry snippets (manual):")
    print(f'  _TRANSFORMER["{family}"] = ("backend.engine.families.{family}.transformer", "{cls_name}")')
    print(f'  # Weight remap is internalized via sanitize() on the TransformerBase subclass')
    print(f'  bundle_manifest.FAMILY_BUNDLE_CONTRACTS: add "{family}" required components')
    # Count root-level stem units + sub-packages
    root_units = len(
        {
            p.stem.removesuffix("_mlx")
            for p in family_dir.glob("*.py")
            if p.name != "__init__.py"
        }
    )
    sub_pkgs = len(
        [d for d in family_dir.iterdir() if d.is_dir() and d.name not in ("__pycache__", "data")]
    )
    units = root_units + sub_pkgs
    if units > 8:
        print(
            f"\nWARN: {family} has {units} logical units (budget 8); "
            f"extract sub-packages or consolidate stems.",
            file=sys.stderr,
        )
    print(f"\nNext: python scripts/register_image_family.py --family {family}")
    print("See docs/engine_architecture.md §6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
