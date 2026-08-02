"""Lazy G1 asset helpers: depth from scene reference image."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def depth_asset_from_source_image(
    source_path: Path,
    *,
    asset_store: Any,
    depth_bundle_root: Path,
    width: int = 1024,
    height: int = 1024,
    source_asset_id: str = "",
    group_id: str | None = None,
    on_log: Any = None,
) -> str:
    """Run depth-pro on *source_path* and store grayscale PNG as asset."""
    from backend.engine.platform import active_backends

    if not source_path.is_file():
        raise RuntimeError(f"source image not found: {source_path}")
    if not depth_bundle_root.is_dir():
        raise RuntimeError(
            f"depth-pro bundle not found at {depth_bundle_root}; install from Models → Tools"
        )

    pil = Image.open(source_path).convert("RGB")
    backends = active_backends()
    if "mlx" in backends:
        from backend.engine.families.flux1.depth_encode_mlx import estimate_depth_rgb01_mlx

        rgb = estimate_depth_rgb01_mlx(
            pil,
            width=width,
            height=height,
            depth_bundle_root=depth_bundle_root,
            on_log=on_log,
        )
    elif "cuda" in backends:
        from backend.engine.families.flux1.depth_encode_cuda import estimate_depth_rgb01_cuda

        rgb = estimate_depth_rgb01_cuda(
            pil,
            width=width,
            height=height,
            depth_bundle_root=depth_bundle_root,
            on_log=on_log,
        )
    else:
        raise RuntimeError("depth-pro requires mlx or cuda backend")

    import numpy as np

    gray = (rgb[..., 0] * 255.0).clip(0, 255).astype("uint8")
    out_path = source_path.parent / f"{source_path.stem}_depth.png"
    Image.fromarray(gray, mode="L").save(out_path)
    from backend.engine.group_utils import resolve_group_id_from_asset

    gid = group_id
    if not gid and source_asset_id:
        gid = resolve_group_id_from_asset(asset_store, source_asset_id)
    asset_id = asset_store.create_from_file(
        out_path,
        kind="image",
        mime_type="image/png",
        source_task_id="",
        metadata={"long_video_phase": "scene_grounding_depth"},
        source_action="tool",
        parent_asset_id=source_asset_id or None,
        relation_type="derived" if source_asset_id else None,
        group_id=gid,
    )
    return str(asset_id)
