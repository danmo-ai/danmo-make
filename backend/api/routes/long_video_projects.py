"""REST API for long-video workbench projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_long_video_activity_store, get_long_video_project_store
from backend.core.contracts import LongVideoProjectActivityDTO, LongVideoProjectActivityListResponse
from backend.persistence.long_video_activity_store import LongVideoActivityStore
from backend.persistence.long_video_project_store import LongVideoProjectStore

router = APIRouter(prefix="/api/long-video", tags=["long-video"])


class LongVideoProjectCreateRequest(BaseModel):
    title: str = ""
    state: Optional[dict[str, Any]] = None


class LongVideoProjectUpdateRequest(BaseModel):
    title: Optional[str] = None
    state: Optional[dict[str, Any]] = None


class SceneGroundingDepthRequest(BaseModel):
    source_asset_id: str
    width: int = 1024
    height: int = 1024


@router.get("/projects")
async def list_long_video_projects(
    limit: int = 100,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    return {"items": store.list_projects(limit=limit)}


@router.post("/projects", status_code=201)
async def create_long_video_project(
    body: LongVideoProjectCreateRequest,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    return store.create_project(title=body.title, state=body.state)


@router.get("/projects/{project_id}")
async def get_long_video_project(
    project_id: str,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    row = store.get_project(project_id)
    if not row:
        raise HTTPException(status_code=404, detail="long video project not found")
    return row


@router.put("/projects/{project_id}")
async def update_long_video_project(
    project_id: str,
    body: LongVideoProjectUpdateRequest,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    row = store.update_project(project_id, title=body.title, state=body.state)
    if not row:
        raise HTTPException(status_code=404, detail="long video project not found")
    return row


@router.delete("/projects/{project_id}", status_code=204)
async def delete_long_video_project(
    project_id: str,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="long video project not found")


@router.get("/projects/{project_id}/activity")
async def list_long_video_project_activity(
    project_id: str,
    limit: int = 200,
    offset: int = 0,
    category: str | None = None,
    phase: str | None = None,
    event_type: str | None = None,
    parse_run_id: str | None = None,
    task_id: str | None = None,
    shot_id: str | None = None,
    project_store: LongVideoProjectStore = Depends(get_long_video_project_store),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not project_store.get_project(project_id):
        raise HTTPException(status_code=404, detail="long video project not found")
    items = activity_store.list_events(
        project_id,
        limit=limit,
        offset=offset,
        category=category,
        phase=phase,
        event_type=event_type,
        parse_run_id=parse_run_id,
        task_id=task_id,
        shot_id=shot_id,
    )
    total = activity_store.count_events(
        project_id,
        category=category,
        event_type=event_type,
        parse_run_id=parse_run_id,
    )
    return LongVideoProjectActivityListResponse(
        items=[LongVideoProjectActivityDTO(**row) for row in items],
        total=total,
    )


@router.get("/projects/{project_id}/activity/parse-runs/{parse_run_id}")
async def get_long_video_parse_run(
    project_id: str,
    parse_run_id: str,
    project_store: LongVideoProjectStore = Depends(get_long_video_project_store),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not project_store.get_project(project_id):
        raise HTTPException(status_code=404, detail="long video project not found")
    row = activity_store.get_parse_run(project_id, parse_run_id)
    if not row:
        raise HTTPException(status_code=404, detail="parse run not found")
    return row


@router.post("/scene-grounding/depth-from-asset")
async def scene_grounding_depth_from_asset(
    body: SceneGroundingDepthRequest,
):
    """Lazy G1: depth map from scene reference / panorama asset (depth-pro)."""
    from backend.api.deps import get_asset_store, get_model_registry
    from backend.long_video.scene_grounding_assets import depth_asset_from_source_image
    from backend.engine.families.flux1.structural import resolve_depth_pro_bundle_root

    asset_store = get_asset_store()
    registry = get_model_registry()
    source_id = (body.source_asset_id or "").strip()
    if not source_id:
        raise HTTPException(status_code=400, detail="source_asset_id is required")
    try:
        src_path = asset_store.get_file_path(source_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    project_root = Path(__file__).resolve().parents[3]
    try:
        depth_root = resolve_depth_pro_bundle_root(registry, project_root)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        depth_id = depth_asset_from_source_image(
            src_path,
            asset_store=asset_store,
            depth_bundle_root=depth_root,
            width=int(body.width),
            height=int(body.height),
            source_asset_id=source_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"depth_asset_id": depth_id, "panorama_asset_id": source_id}
