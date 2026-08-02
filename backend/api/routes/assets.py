"""CRUD /api/assets — plan assets.py"""

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_asset_store
from backend.persistence.asset_store import SQLiteAssetStore

router = APIRouter(prefix="/api/assets", tags=["assets"])


class AssetReconcileRequest(BaseModel):
    """磁盘对账：主文件路径在磁盘上不存在时视为孤儿（可仅报告或从 DB 删除行）。"""

    dry_run: bool = Field(
        True,
        description="为 true 时只返回 missing 列表，不修改数据库；为 false 时删除对应 assets 行",
    )
    purge_legacy_step_previews: bool = Field(
        False,
        description="同时删除误入库的去噪步进预览（metadata.preview=true）；与 dry_run 无关，会真实删库",
    )


class AssetGroupCreateRequest(BaseModel):
    title: str = ""
    kind: str = "mixed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetGroupUpdateRequest(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] | None = None


class AssetSetGroupRequest(BaseModel):
    group_id: str | None = None


@router.post("", status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    tid = "upl_" + uuid.uuid4().hex[:12]
    tmp = store.files_root / f"_tmp_{tid}{suffix}"
    try:
        content = await file.read()
        tmp.write_bytes(content)
        mime = file.content_type or "application/octet-stream"
        kind = "image" if mime.startswith("image/") else "file"
        aid = store.create_from_file(
            tmp,
            kind=kind,
            mime_type=mime,
            source_task_id="",
            metadata={"original_name": file.filename},
            source_action="upload",
        )
        tmp.unlink(missing_ok=True)
        return {"id": aid, "kind": kind}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, str(e)) from e


class BatchDeleteRequest(BaseModel):
    asset_ids: list[str] = Field(..., description="要批量删除的资产 ID 列表")


@router.get("")
async def list_assets(
    kind: str | None = Query(None),
    source_task_id: str | None = Query(None, description="仅列出该任务产出的资产"),
    source_action: str | None = Query(None, description="按 source_action 精确筛选"),
    source_action_in: list[str] | None = Query(None, description="按 source_action 多值筛选（逗号分隔）"),
    group_id: str | None = Query(None, description="仅列出指定分组的资产"),
    exclude_grouped: bool = Query(False, description="排除已归属分组的资产"),
    created_after: str | None = Query(None, description="ISO8601：created_at 下界（含）"),
    created_before: str | None = Query(None, description="ISO8601：created_at 上界（含）"),
    model: str | None = Query(None, description="按 metadata.model 筛选"),
    model_in: list[str] | None = Query(None, description="按 metadata.model 多值筛选（逗号分隔）"),
    search: str | None = Query(None, description="搜索 metadata 中的关键词"),
    search_fields: list[str] | None = Query(None, description="指定搜索字段（如 title,prompt），默认 title,prompt"),
    exclude_upload_refs: bool = Query(False, description="排除创作页上传的参考图/蒙版（source_task_id 为空且 source_action 为 upload）"),
    exclude_step_previews: bool = Query(
        True,
        description="排除去噪步进预览（512px 临时图，不应出现在作品库）",
    ),
    sort_by: str = Query("created_at", description="排序字段：created_at|name|width|height"),
    sort_order: str = Query("desc", description="排序方向：asc|desc"),
    limit: int = Query(40, ge=1, le=500),
    offset: int = Query(0, ge=0),
    parent_asset_id: str | None = Query(None, description="仅列出指定父资产的直接子资产"),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    return {
        "items": store.list_assets(
            kind=kind,
            source_task_id=source_task_id,
            source_action=source_action,
            source_action_in=source_action_in,
            parent_asset_id=parent_asset_id,
            group_id=group_id,
            exclude_grouped=exclude_grouped,
            created_after=created_after,
            created_before=created_before,
            model=model,
            model_in=model_in,
            search=search,
            search_fields=search_fields,
            exclude_upload_refs=exclude_upload_refs,
            exclude_step_previews=exclude_step_previews,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ),
        "limit": limit,
        "offset": offset,
    }


# ------------------------------------------------------------------------------
# Asset groups — must be defined before /{asset_id} catch-all routes
# ------------------------------------------------------------------------------

@router.get("/groups")
async def list_asset_groups(
    kind: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    groups = store.list_groups(kind=kind, limit=limit, offset=offset)
    for g in groups:
        g["asset_count"] = store.count_group_assets(g["id"], kind=kind)
        g["preview_assets"] = store.list_group_preview_assets(g["id"], kind=kind, limit=4)
    return {"items": groups, "limit": limit, "offset": offset}


@router.post("/groups", status_code=201)
async def create_asset_group(
    body: AssetGroupCreateRequest,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    gid = "grp_" + uuid.uuid4().hex[:20]
    return store.ensure_group(
        gid,
        title=body.title,
        kind=body.kind,
        metadata=body.metadata,
    )


@router.get("/groups/{group_id}")
async def get_asset_group(
    group_id: str,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    group = store.get_group(group_id)
    if not group:
        raise HTTPException(404, "asset group not found")
    group["asset_count"] = store.count_group_assets(group_id)
    group["preview_assets"] = store.list_group_preview_assets(group_id, limit=4)
    return group


@router.get("/groups/{group_id}/assets")
async def list_group_assets(
    group_id: str,
    kind: str | None = Query(None),
    limit: int = Query(40, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """List assets belonging to a group (optionally filtered by media kind)."""
    if not store.get_group(group_id):
        raise HTTPException(404, "asset group not found")
    return {
        "items": store.list_assets(
            group_id=group_id,
            kind=kind,
            exclude_upload_refs=True,
            exclude_step_previews=True,
            sort_by="created_at",
            sort_order="desc",
            limit=limit,
            offset=offset,
        ),
        "limit": limit,
        "offset": offset,
    }


@router.put("/groups/{group_id}")
async def update_asset_group(
    group_id: str,
    body: AssetGroupUpdateRequest,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    group = store.update_group(
        group_id,
        title=body.title,
        metadata=body.metadata,
    )
    if not group:
        raise HTTPException(404, "asset group not found")
    return group


@router.delete("/groups/{group_id}", status_code=204)
async def delete_asset_group(
    group_id: str,
    unlink_assets: bool = Query(False, description="为 true 时保留资产并移除分组归属"),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    if not store.delete_group(group_id, unlink_assets=unlink_assets):
        raise HTTPException(404, "asset group not found")


@router.get("/{asset_id}")
async def get_asset(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    """Single asset metadata (path, mime, lineage refs) — used by MCP / agents."""
    row = store.get_asset_record(asset_id)
    if not row:
        raise HTTPException(404, "asset not found")
    return row


@router.get("/{asset_id}/lineage")
async def get_asset_lineage(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    """查询资产生成谱系：当前节点 + 祖先链 + 后代树。"""
    try:
        return store.get_lineage(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/reconcile")
async def reconcile_assets_disk(
    body: AssetReconcileRequest | None = Body(default=None),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """比对 DB 登记的主文件路径与磁盘；默认 dry_run 不删库。须声明 `dry_run: false` 才会移除孤儿行。"""
    opts = body or AssetReconcileRequest()
    out = store.reconcile_disk_vs_db(dry_run=opts.dry_run)
    if opts.purge_legacy_step_previews:
        out["purged_step_previews"] = store.purge_all_generation_step_previews()
    return out


@router.get("/{asset_id}/thumbnail")
async def get_asset_thumbnail(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    tp = store.ensure_image_thumbnail(asset_id) or store.get_thumbnail_path(asset_id)
    if tp and tp.exists():
        suf = tp.suffix.lower()
        media = "image/png" if suf == ".png" else "image/webp" if suf == ".webp" else "image/jpeg"
        return FileResponse(str(tp), media_type=media)
    try:
        main = store.get_file_path(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "asset not found") from e
    if not main.exists():
        raise HTTPException(404, "file missing")
    suf = main.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        media = "image/png" if suf == ".png" else "image/jpeg" if suf in (".jpg", ".jpeg") else "image/webp"
        return FileResponse(str(main), media_type=media)
    raise HTTPException(404, "no thumbnail")


_AUDIO_MEDIA = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
}


@router.head("/{asset_id}/file")
@router.get("/{asset_id}/file")
async def get_asset_file(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    try:
        p = store.get_file_path(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "asset not found") from e
    if not p.exists():
        raise HTTPException(404, "file missing")
    media = _AUDIO_MEDIA.get(p.suffix.lower())
    if media:
        return FileResponse(str(p), media_type=media)
    return FileResponse(str(p))


@router.post("/batch-delete")
async def batch_delete_assets(
    body: BatchDeleteRequest,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    return store.delete_batch(body.asset_ids)


@router.delete("/{asset_id}")
async def delete_asset(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    return {"ok": store.delete(asset_id)}


@router.patch("/{asset_id}/group")
async def set_asset_group(
    asset_id: str,
    body: AssetSetGroupRequest,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    if not store.set_asset_group(asset_id, body.group_id):
        raise HTTPException(404, "asset not found")
    return {"ok": True}
