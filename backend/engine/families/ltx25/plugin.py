"""LTX-2.5 video ``FamilyPlugin`` factory (MLX only)."""

from __future__ import annotations

from pathlib import Path

from backend.engine.config.model_configs import get_config_class
from backend.engine.families._video_backbone import VideoPluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family


def build_ltx25_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("ltx25")()
    spec = family_spec_from_config("ltx25", config, media="video")
    return FamilyPlugin(family_id="ltx25", spec=spec, backbone=VideoPluginBackbone(spec))


def register_ltx25_plugin() -> None:
    register_family("ltx25", build_ltx25_plugin)
