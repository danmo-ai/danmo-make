"""Scene grounding bundle (G0–G2): layout + optional panorama/depth asset refs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CameraCandidate:
    zone_id: str
    azimuth: float = 0.0
    elevation: float = 0.0
    fov: float = 60.0
    description: str = ""


@dataclass
class SceneGroundingBundle:
    scene_key: str
    location: str = ""
    spatial_layout_json: dict[str, Any] = field(default_factory=dict)
    panorama_prompt: str = ""
    panorama_asset_id: str = ""
    depth_asset_id: str = ""
    selected_camera_zone_id: str = ""
    camera_candidates: list[CameraCandidate] = field(default_factory=list)
    layout_version: int = 1


def build_camera_candidates(spatial_layout: dict[str, Any] | None) -> list[CameraCandidate]:
    layout = spatial_layout or {}
    zones = layout.get("camera_zones") or []
    out: list[CameraCandidate] = []
    if isinstance(zones, list):
        for i, z in enumerate(zones):
            if not isinstance(z, dict):
                continue
            out.append(
                CameraCandidate(
                    zone_id=str(z.get("id") or f"zone_{i}"),
                    azimuth=float(z.get("azimuth", 0) or 0),
                    elevation=float(z.get("elevation", 0) or 0),
                    fov=float(z.get("fov", 60) or 60),
                    description=str(z.get("description") or ""),
                )
            )
    if not out:
        out.append(CameraCandidate(zone_id="default_wide", description="wide establishing"))
    return out


def pick_camera_zone(
    candidates: list[CameraCandidate],
    *,
    preferred_zone_id: str = "",
    visibility: str = "full_face",
) -> CameraCandidate:
    preferred = (preferred_zone_id or "").strip()
    if preferred:
        for c in candidates:
            if c.zone_id == preferred:
                return c
    vis = str(visibility or "full_face")
    if vis in ("invisible", "silhouette"):
        for c in candidates:
            desc = c.description.lower()
            if "wide" in desc or "door" in desc or "entry" in desc or "门" in c.description:
                return c
    return candidates[0]


def select_camera_zone_with_vlm(
    candidates: list[CameraCandidate],
    *,
    shot_description: str,
    score_fn: Callable[[CameraCandidate, str], float] | None = None,
    preferred_zone_id: str = "",
    visibility: str = "full_face",
) -> CameraCandidate:
    """G2: pick best camera zone; uses VLM callback when provided, else heuristics."""
    if score_fn and candidates:
        ranked = sorted(
            candidates,
            key=lambda c: score_fn(c, shot_description),
            reverse=True,
        )
        if ranked:
            return ranked[0]
    return pick_camera_zone(
        candidates,
        preferred_zone_id=preferred_zone_id,
        visibility=visibility,
    )


def build_grounding_bundle(
    scene_key: str,
    *,
    location: str = "",
    spatial_layout: dict[str, Any] | None = None,
    environment_text: str = "",
) -> SceneGroundingBundle:
    layout = dict(spatial_layout or {})
    candidates = build_camera_candidates(layout)
    picked = pick_camera_zone(candidates, preferred_zone_id="", visibility="full_face")
    prompt_parts = [environment_text.strip(), location.strip(), str(layout.get("dimensions", ""))]
    prompt = "，".join(p for p in prompt_parts if p) or location or scene_key
    return SceneGroundingBundle(
        scene_key=scene_key,
        location=location,
        spatial_layout_json=layout,
        panorama_prompt=f"360 environment reference, {prompt}, cinematic, no people",
        selected_camera_zone_id=picked.zone_id,
        camera_candidates=candidates,
    )


def grounding_bundle_to_scene_fields(bundle: SceneGroundingBundle) -> dict[str, Any]:
    return {
        "spatial_layout_json": bundle.spatial_layout_json,
        "grounding_panorama_asset_id": bundle.panorama_asset_id,
        "grounding_depth_asset_id": bundle.depth_asset_id,
    }


def keyframe_grounding_metadata(
    shot: dict[str, Any],
    scene: dict[str, Any] | None,
) -> dict[str, str]:
    """Metadata for ImagePipeline keyframe when first_frame_strategy uses grounding."""
    layout_raw = (scene or {}).get("spatial_layout_json") or {}
    layout = layout_raw if isinstance(layout_raw, dict) else {}
    candidates = build_camera_candidates(layout)
    picked = pick_camera_zone(
        candidates,
        preferred_zone_id=str(shot.get("camera_zone_id") or ""),
        visibility=str(shot.get("first_frame_visibility") or "full_face"),
    )
    meta: dict[str, str] = {
        "long_video_first_frame_strategy": str(shot.get("first_frame_strategy") or "t2i_from_grounding"),
        "long_video_scene_grounding_camera_zone_id": picked.zone_id,
    }
    pano = (scene or {}).get("grounding_panorama_asset_id") or ""
    depth = (scene or {}).get("grounding_depth_asset_id") or ""
    if pano:
        meta["long_video_scene_grounding_panorama_asset_id"] = str(pano)
    if depth:
        meta["long_video_scene_grounding_depth_asset_id"] = str(depth)
    req = str(shot.get("first_frame_requirement") or "").strip()
    if req:
        meta["long_video_first_frame_requirement"] = req
    return meta


def apply_grounding_to_scene_dtos(
    scene_dtos: list[dict[str, Any]],
    layouts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in scene_dtos:
        copy = dict(row)
        layout = copy.get("spatial_layout_json") or {}
        if isinstance(layout, dict) and layout.get("scene_key"):
            key = str(layout["scene_key"])
        else:
            name = str(copy.get("name") or "scene")
            key = name.replace(" ", "_")[:48]
            layout = layouts.get(key, layout if isinstance(layout, dict) else {})
        env = ""
        looks = copy.get("looks") or []
        if looks and isinstance(looks[0], dict):
            env = str(looks[0].get("environment") or looks[0].get("body") or "")
        bundle = build_grounding_bundle(
            key,
            location=str(copy.get("name") or ""),
            spatial_layout=layout if isinstance(layout, dict) else None,
            environment_text=env,
        )
        copy["spatial_layout_json"] = bundle.spatial_layout_json
        copy.setdefault("grounding_panorama_asset_id", bundle.panorama_asset_id)
        copy.setdefault("grounding_depth_asset_id", bundle.depth_asset_id)
        out.append(copy)
    return out
