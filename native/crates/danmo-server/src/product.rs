//! Routes the Vue SPA calls besides generation submit.
//! Graphs that are not built return a JSON `detail` and do not pretend to run.

use crate::app::AppState;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::{Catalog, GGUF_VERSION_KEYS};
use danmo_store::{AssetRow, CanvasRow, TaskRow};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path as FsPath, PathBuf};
use uuid::Uuid;

pub struct ApiError {
    pub status: StatusCode,
    pub detail: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

pub fn fail(status: StatusCode, detail: impl Into<String>) -> ApiError {
    ApiError {
        status,
        detail: detail.into(),
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/tasks", get(list_tasks))
        .route("/api/tasks/list", get(list_tasks))
        .route(
            "/api/tasks/:id",
            get(get_task).patch(patch_task).delete(cancel_task),
        )
        .route("/api/tasks/:id/logs", get(task_logs))
        .route("/api/tasks/:id/graph", get(task_graph))
        .route("/api/tasks/:id/preview", get(task_preview))
        .route("/api/tasks/:id/diagnose", post(task_diagnose))
        .route("/api/tasks/:id/diagnostic", get(task_graph))
        .route("/api/models", get(list_models))
        .route("/api/models/:id/install", post(install_model))
        .route(
            "/api/models/:id/versions/:version",
            axum::routing::delete(delete_model_version),
        )
        .route("/api/models/install-batch", post(install_batch))
        .route("/api/adapters", get(empty_items))
        .route("/api/setup/recommendations", get(recommendations))
        .route("/api/assets", get(list_assets).post(upload_asset))
        .route("/api/assets/groups", get(empty_items))
        .route("/api/assets/batch-delete", post(batch_delete_assets))
        .route("/api/assets/reconcile", post(reconcile_assets))
        .route("/api/assets/:id", axum::routing::delete(delete_asset))
        .route("/api/assets/:id/file", get(asset_file))
        .route("/api/assets/:id/thumbnail", get(asset_file))
        .route("/api/assets/:id/lineage", get(asset_lineage))
        .route("/api/gallery/upload", post(upload_asset))
        .route("/api/canvas/sessions", get(list_canvas).post(create_canvas))
        .route(
            "/api/canvas/sessions/:id",
            get(get_canvas).put(update_canvas).delete(delete_canvas),
        )
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/settings/access-keys", post(create_access_key))
        .route(
            "/api/settings/access-keys/:kind",
            axum::routing::delete(revoke_access_key),
        )
        .route("/api/settings/workspace-paths", get(workspace_paths))
        .route("/api/settings/workspace-status", get(workspace_status))
        .route("/api/settings/apply-workspace", post(apply_workspace))
        .route(
            "/api/settings/pick-workspace-directory",
            post(pick_workspace),
        )
        .route(
            "/api/settings/restore-config-defaults",
            post(restore_defaults),
        )
        .route("/api/settings/models", get(installed_models))
        .route("/api/settings/loras", get(empty_items))
        .route("/api/settings/refresh", post(refresh_ok))
        .route("/api/settings/system", get(system_info))
        .route("/api/settings/install", post(install_runtime))
        .route("/api/settings/registry", get(settings_registry))
        .route("/api/settings/models/status", get(models_status))
        .route("/api/settings/models/status/detailed", get(models_detailed))
        .route("/api/settings/disk-space", get(disk_space))
        .route("/api/settings/loras/compatible/:model", get(empty_list))
        .route(
            "/api/settings/controlnets/compatible/:model",
            get(empty_list),
        )
        .route("/api/settings/presets", get(get_presets).post(save_preset))
        .route(
            "/api/settings/presets/:name",
            axum::routing::delete(delete_preset),
        )
        .route("/api/settings/system/monitor", get(system_monitor))
        .route("/api/system/metrics", get(system_monitor))
        .route("/api/system/cache", get(cache_status))
        .route("/api/download/tasks", get(empty_tasks))
        .route("/api/download/start", post(reject_download))
        .route("/api/download/lora/search", get(empty_items))
        .route("/api/download/lora/base-models", get(empty_list))
        .route("/api/download/lora/hub", post(reject_download))
        .route("/api/download/lora", post(reject_download))
        .route("/api/download/convert", post(reject_convert))
        .route("/api/loras/datasets", get(empty_items))
        .route("/api/loras/user-adapters", get(empty_items))
        .route("/api/loras/downloaded-adapters", get(empty_items))
        .route("/api/loras/trainable-models", get(empty_models))
        .route(
            "/api/loras/training/requirements",
            get(training_requirements),
        )
        .route("/api/loras/trainings", post(reject_training))
        .route("/api/tools/z-image/merge", post(reject_merge))
        .route("/api/tools/z-image/merge/models", get(merge_models))
        .route("/api/tools/z-image/merge/merged", get(empty_items))
        .route("/api/audios/generations", get(list_audio_tasks))
        .route("/api/chat/enhance-prompt", post(reject_assistant))
        .route("/v1/chat/completions", post(reject_assistant))
}

#[derive(Debug, Deserialize, Default)]
struct ListQuery {
    limit: Option<i64>,
    offset: Option<i64>,
    kind: Option<String>,
    status: Option<String>,
    since: Option<String>,
    media: Option<String>,
    action: Option<String>,
    installed: Option<String>,
}

pub fn task_json(row: &TaskRow, queue_position: Option<i64>) -> Value {
    let mut params: Value = serde_json::from_str(&row.params).unwrap_or_else(|_| json!({}));
    if let Some(request) = params.get("request").cloned() {
        if let Some(obj) = params.as_object_mut() {
            for key in ["prompt", "title", "model"] {
                if let Some(value) = request.get(key) {
                    obj.insert(key.to_string(), value.clone());
                }
            }
        }
    }
    let result = row
        .result
        .as_deref()
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or(Value::Null);
    json!({
        "id": row.id,
        "kind": row.kind,
        "model_id": row.model_id,
        "model": { "id": row.model_id, "name": row.model_id },
        "status": row.status,
        "priority": row.priority,
        "progress": row.progress,
        "params": params,
        "result": result,
        "error_message": row.error_message.clone().unwrap_or_default(),
        "error": row.error_message.clone().filter(|text| !text.is_empty()),
        "created_at": row.created_at,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "queue_position": queue_position,
    })
}

pub fn asset_json(row: &AssetRow) -> Value {
    let metadata: Value = serde_json::from_str(&row.metadata).unwrap_or_else(|_| json!({}));
    json!({
        "id": row.id,
        "kind": row.kind,
        "mime_type": row.mime_type,
        "path": row.file_path,
        "thumbnail_url": format!("/api/assets/{}/thumbnail", row.id),
        "width": row.width,
        "height": row.height,
        "duration_seconds": row.duration_seconds,
        "source_task_id": row.source_task_id,
        "source_action": row.source_action,
        "metadata": metadata,
        "created_at": row.created_at,
        "parent_asset_id": row.parent_asset_id,
        "relation_type": row.relation_type,
        "group_id": row.group_id,
    })
}

pub async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let row = state
        .scheduler
        .get(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "task not found"))?;
    Ok(Json(task_json(&row, None)))
}

pub async fn patch_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let label = body.get("priority").and_then(|v| v.as_str()).unwrap_or("");
    let priority = match label {
        "high" => 10,
        "normal" => 100,
        _ => {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                "priority must be high or normal",
            ))
        }
    };
    match state
        .scheduler
        .set_priority(&id, priority)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
    {
        "ok" => get_task(State(state), Path(id)).await,
        "missing" => Err(fail(StatusCode::NOT_FOUND, "task not found")),
        _ => Err(fail(
            StatusCode::CONFLICT,
            "only queued tasks can change priority",
        )),
    }
}

pub async fn cancel_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    match state
        .scheduler
        .cancel(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
    {
        "ok" => Ok(Json(json!({ "ok": true }))),
        "missing" => Err(fail(StatusCode::NOT_FOUND, "task not found")),
        _ => Err(fail(
            StatusCode::CONFLICT,
            "only queued or running tasks can be cancelled",
        )),
    }
}

async fn list_tasks(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let offset = query.offset.unwrap_or(0).max(0);
    let rows = state
        .scheduler
        .list_tasks(
            limit,
            offset,
            query.kind.as_deref(),
            query.status.as_deref(),
            query.since.as_deref(),
        )
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let tasks: Vec<Value> = rows.iter().map(|row| task_json(row, None)).collect();
    Ok(Json(json!({ "tasks": tasks })))
}

pub fn queue_body(state: &AppState) -> Result<Value, ApiError> {
    let rows = state
        .scheduler
        .list_tasks(500, 0, None, None, None)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let running: Vec<Value> = rows
        .iter()
        .filter(|row| row.status == "running")
        .take(5)
        .map(|row| task_json(row, None))
        .collect();
    let mut queued_rows: Vec<&TaskRow> = rows.iter().filter(|row| row.status == "queued").collect();
    queued_rows.sort_by_key(|row| (row.priority, row.created_at.clone()));
    let mut wait = 0i64;
    let queued: Vec<Value> = queued_rows
        .iter()
        .take(50)
        .enumerate()
        .map(|(index, row)| {
            wait += 30;
            let mut item = task_json(row, Some((index as i64) + 1));
            if let Some(obj) = item.as_object_mut() {
                obj.insert("estimated_wait_seconds".into(), json!(wait));
            }
            item
        })
        .collect();
    Ok(json!({
        "running": running,
        "queued": queued,
        "queue_image_first": state.scheduler.queue_image_first,
        "counts": { "running": running.len(), "queued": queued_rows.len() },
    }))
}

async fn task_logs(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let offset = query.offset.unwrap_or(0).max(0);
    let limit = query.limit.unwrap_or(500).clamp(1, 2000);
    let rows = state
        .scheduler
        .logs(&id, offset, limit)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let items: Vec<Value> = rows
        .iter()
        .map(|row| {
            json!({
                "id": row.id,
                "message": row.message,
                "level": row.level,
                "time": row.time,
            })
        })
        .collect();
    Ok(Json(json!({ "items": items })))
}

async fn task_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let row = state
        .scheduler
        .get(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "task not found"))?;
    let phase = match row.status.as_str() {
        "queued" => "resolve",
        "running" => "infer",
        "completed" => "persist",
        _ => "persist",
    };
    let status_of = |name: &str| {
        if row.status == "failed" {
            "failed"
        } else if row.status == "completed" || phase_rank(phase) > phase_rank(name) {
            "ok"
        } else if phase == name {
            "running"
        } else {
            "pending"
        }
    };
    let nodes = [
        "resolve", "load", "encode", "schedule", "infer", "decode", "persist",
    ]
    .into_iter()
    .map(|name| {
        json!({
            "id": name,
            "label": name,
            "status": status_of(name),
        })
    })
    .collect::<Vec<_>>();
    Ok(Json(json!({
        "graph_id": id,
        "nodes": nodes,
        "active_node": if row.status == "running" { Some(phase) } else { None },
        "progress": row.progress,
    })))
}

fn phase_rank(name: &str) -> i32 {
    match name {
        "resolve" => 0,
        "load" => 1,
        "encode" => 2,
        "schedule" => 3,
        "infer" => 4,
        "decode" => 5,
        "persist" => 6,
        _ => 0,
    }
}

async fn task_preview(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let path = state.outputs.join("previews").join(format!("{id}.png"));
    match std::fs::read(&path) {
        Ok(bytes) => ([(header::CONTENT_TYPE, "image/png")], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "preview is not ready").into_response(),
    }
}

async fn task_diagnose(Path(id): Path<String>) -> Json<Value> {
    Json(json!({
        "task_id": id,
        "summary": "assistant graph is not built",
    }))
}

async fn list_models(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut out = serde_json::Map::new();
    let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) else {
        return Ok(Json(json!({ "models": {} })));
    };
    for (id, entry) in models {
        let media = entry
            .pointer("/catalog/media")
            .and_then(|v| v.as_str())
            .unwrap_or("image");
        if let Some(want) = query.media.as_deref() {
            if !want.is_empty() && want != media {
                continue;
            }
        }
        if let Some(action) = query.action.as_deref().filter(|a| !a.is_empty()) {
            let actions = entry.get("actions").and_then(|v| v.as_object());
            if actions.map(|map| map.contains_key(action)) != Some(true) {
                continue;
            }
        }
        let ready = model_ready(entry);
        if let Some(flag) = query.installed.as_deref() {
            let want = flag == "true" || flag == "1";
            if want != ready {
                continue;
            }
        }
        let family = entry
            .pointer("/runtime/family")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let engine = entry
            .pointer("/catalog/engine")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let actions = entry
            .get("actions")
            .and_then(|v| v.as_object())
            .map(|map| map.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        out.insert(
            id.clone(),
            json!({
                "media": media,
                "family": family,
                "engine": engine,
                "actions": actions,
                "installed": ready,
                "commercial_use_allowed": entry.pointer("/catalog/commercial_use_allowed"),
            }),
        );
    }
    Ok(Json(json!({ "models": out })))
}

pub(crate) fn materialize_graph_packs(catalog: &Catalog) -> Result<(), String> {
    let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) else {
        return Ok(());
    };
    for entry in models.values() {
        let Some(versions) = entry
            .pointer("/distribution/versions")
            .and_then(|v| v.as_object())
        else {
            continue;
        };
        for key in GGUF_VERSION_KEYS {
            let Some(version) = versions.get(*key) else {
                continue;
            };
            if version.get("graph_pack").and_then(|v| v.as_str()) != Some("flux2-chain") {
                continue;
            }
            let Some(path) = version
                .get("local_path")
                .and_then(|v| v.as_str())
                .filter(|path| !path.is_empty())
            else {
                continue;
            };
            let path = resolve_path(path);
            if path.join("bundle.manifest.json").is_file() {
                continue;
            }
            danmo_engine::write_chain_bundle(&path).map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

async fn models_status(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut out = serde_json::Map::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            out.insert(id.clone(), json!(model_ready(entry)));
        }
    }
    Ok(Json(Value::Object(out)))
}

async fn models_detailed(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut out = serde_json::Map::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            out.insert(id.clone(), detailed_status(entry));
        }
    }
    Ok(Json(Value::Object(out)))
}

fn detailed_status(entry: &Value) -> Value {
    let versions = entry
        .pointer("/distribution/versions")
        .and_then(|v| v.as_object());
    let mut ready = false;
    let mut listed = serde_json::Map::new();
    if let Some(versions) = versions {
        for key in GGUF_VERSION_KEYS {
            let Some(version) = versions.get(*key) else {
                continue;
            };
            let path = version
                .get("local_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let is_ready = bundle_ready(path);
            if is_ready {
                ready = true;
            }
            listed.insert(
                (*key).to_string(),
                json!({
                    "ready": is_ready,
                    "status": if is_ready { "ready" } else { "not_downloaded" },
                    "local_path": path,
                }),
            );
        }
    }
    json!({ "ready": ready, "versions": listed })
}

fn model_ready(entry: &Value) -> bool {
    detailed_status(entry)
        .get("ready")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn bundle_ready(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    let path = resolve_path(path);
    path.join("bundle.manifest.json").is_file()
}

fn resolve_path(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

async fn installed_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut rows = Vec::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            let Some(versions) = entry
                .pointer("/distribution/versions")
                .and_then(|v| v.as_object())
            else {
                continue;
            };
            for key in GGUF_VERSION_KEYS {
                let Some(path) = versions
                    .get(*key)
                    .and_then(|v| v.get("local_path"))
                    .and_then(|v| v.as_str())
                else {
                    continue;
                };
                if !bundle_ready(path) {
                    continue;
                }
                let full = resolve_path(path);
                let bytes = dir_size(&full);
                rows.push(json!({
                    "name": format!("{id}:{key}"),
                    "model_id": id,
                    "version": key,
                    "path": full.display().to_string(),
                    "size_human": human_bytes(bytes),
                    "size_bytes": bytes,
                }));
            }
        }
    }
    Ok(Json(Value::Array(rows)))
}

async fn install_model() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "no GGUF download URL is published for this model. Re-download when a bf16, q8_0, or q4_k package is listed",
    ))
}

async fn install_batch() -> Result<Json<Value>, ApiError> {
    install_model().await
}

async fn delete_model_version() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "installed GGUF versions are removed by deleting the bundle directory",
    ))
}

async fn empty_items() -> Json<Value> {
    Json(json!({ "items": [] }))
}

async fn empty_list() -> Json<Value> {
    Json(json!([]))
}

async fn empty_tasks() -> Json<Value> {
    Json(json!({ "tasks": [] }))
}

async fn empty_models() -> Json<Value> {
    Json(json!({ "models": [] }))
}

async fn recommendations() -> Json<Value> {
    Json(json!({
        "runtime": "ggml",
        "weights": "gguf",
        "versions": ["bf16", "q8_0", "q4_k"],
    }))
}

async fn refresh_ok() -> Json<Value> {
    Json(json!({ "success": true }))
}

async fn list_assets(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let offset = query.offset.unwrap_or(0).max(0);
    let (rows, total) = state
        .scheduler
        .list_assets(query.kind.as_deref(), limit, offset)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let items: Vec<Value> = rows.iter().map(asset_json).collect();
    Ok(Json(json!({ "items": items, "total": total })))
}

async fn upload_asset(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    if body.len() > 64 * 1024 * 1024 {
        return Err(fail(
            StatusCode::PAYLOAD_TOO_LARGE,
            "upload is larger than 64MB",
        ));
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let (filename, bytes) = if content_type.starts_with("multipart/") {
        multipart_file(content_type, &body)
            .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?
    } else {
        let value: Value = serde_json::from_slice(&body)
            .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
        let filename = value
            .get("filename")
            .and_then(|v| v.as_str())
            .unwrap_or("upload.bin")
            .to_string();
        let encoded = value
            .get("data_base64")
            .and_then(|v| v.as_str())
            .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "data_base64 is required"))?;
        let bytes =
            base64_decode(encoded).map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?;
        (filename, bytes)
    };
    let id = new_id("ast_");
    let ext = FsPath::new(&filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("bin");
    let mime = mime_for(ext);
    let kind = kind_for(mime);
    let dir = state.outputs.join("assets");
    std::fs::create_dir_all(&dir)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let path = dir.join(format!("{id}.{ext}"));
    std::fs::write(&path, &bytes)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let row = AssetRow {
        id: id.clone(),
        kind: kind.into(),
        mime_type: mime.into(),
        file_path: path.display().to_string(),
        thumbnail_path: None,
        width: None,
        height: None,
        duration_seconds: None,
        source_task_id: None,
        source_action: Some("upload".into()),
        metadata: json!({ "filename": filename }).to_string(),
        created_at: String::new(),
        parent_asset_id: None,
        relation_type: None,
        group_id: None,
    };
    state
        .scheduler
        .insert_asset(&row)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let stored = state
        .scheduler
        .get_asset(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::INTERNAL_SERVER_ERROR, "asset was not stored"))?;
    Ok(Json(asset_json(&stored)))
}

async fn delete_asset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let row = state
        .scheduler
        .delete_asset(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "asset not found"))?;
    let _ = std::fs::remove_file(&row.file_path);
    Ok(Json(json!({ "ok": true, "id": id })))
}

async fn batch_delete_assets(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let ids = body
        .get("asset_ids")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut deleted = Vec::new();
    for id in ids {
        let Some(id) = id.as_str() else { continue };
        if let Ok(Some(row)) = state.scheduler.delete_asset(id) {
            let _ = std::fs::remove_file(&row.file_path);
            deleted.push(id.to_string());
        }
    }
    Ok(Json(json!({ "deleted": deleted })))
}

async fn reconcile_assets(Json(body): Json<Value>) -> Json<Value> {
    Json(json!({
        "dry_run": body.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(true),
        "missing": [],
        "orphans": [],
    }))
}

async fn asset_file(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let row = match state.scheduler.get_asset(&id) {
        Ok(Some(row)) => row,
        Ok(None) => return (StatusCode::NOT_FOUND, "asset not found").into_response(),
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    match std::fs::read(&row.file_path) {
        Ok(bytes) => ([(header::CONTENT_TYPE, row.mime_type)], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "asset file is missing").into_response(),
    }
}

async fn asset_lineage(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let row = state
        .scheduler
        .get_asset(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "asset not found"))?;
    let parent = match row.parent_asset_id.as_deref() {
        Some(parent_id) => state
            .scheduler
            .get_asset(parent_id)
            .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
            .map(|parent| asset_json(&parent)),
        None => None,
    };
    let children = state
        .scheduler
        .child_assets(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let mut node = asset_json(&row);
    if let Some(obj) = node.as_object_mut() {
        obj.insert("parent".into(), parent.unwrap_or(Value::Null));
        obj.insert(
            "children".into(),
            Value::Array(children.iter().map(asset_json).collect()),
        );
    }
    Ok(Json(node))
}

fn default_canvas_state() -> Value {
    json!({
        "items": {},
        "viewport": { "zoom": 1, "panX": 0, "panY": 0 },
        "staging": { "x": 240, "y": 180, "width": 512, "height": 512, "visible": true },
        "active_asset_path": "",
        "overlays": { "reference": null, "control": null },
        "edges": [],
    })
}

fn canvas_json(row: &CanvasRow) -> Value {
    let state: Value =
        serde_json::from_str(&row.state_json).unwrap_or_else(|_| default_canvas_state());
    let item_count = state
        .get("items")
        .and_then(|v| v.as_object())
        .map(|map| map.len())
        .unwrap_or(0);
    json!({
        "id": row.id,
        "media": row.media,
        "title": row.title,
        "state": state,
        "item_count": item_count,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    })
}

async fn list_canvas(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let media = query.media.as_deref().unwrap_or("image");
    let limit = query.limit.unwrap_or(50).clamp(1, 200);
    let rows = state
        .scheduler
        .list_canvas(media, limit)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let items: Vec<Value> = rows
        .iter()
        .map(|row| {
            let mut item = canvas_json(row);
            if let Some(obj) = item.as_object_mut() {
                obj.remove("state");
            }
            item
        })
        .collect();
    Ok(Json(json!({ "items": items })))
}

async fn create_canvas(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let media = body
        .get("media")
        .and_then(|v| v.as_str())
        .unwrap_or("image");
    let title = body
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .unwrap_or(match media {
            "video" => "Video canvas",
            "audio" => "Audio canvas",
            _ => "Image canvas",
        });
    let state_json = body
        .get("state")
        .cloned()
        .unwrap_or_else(default_canvas_state);
    let now = utc_stamp();
    let row = CanvasRow {
        id: new_id("cvs_"),
        media: media.into(),
        title: title.into(),
        state_json: state_json.to_string(),
        created_at: now.clone(),
        updated_at: now,
    };
    state
        .scheduler
        .insert_canvas(&row)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(Json(canvas_json(&row)))
}

async fn get_canvas(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let row = state
        .scheduler
        .get_canvas(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "canvas session not found"))?;
    Ok(Json(canvas_json(&row)))
}

async fn update_canvas(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let title = body.get("title").and_then(|v| v.as_str());
    let state_json = body.get("state").map(|v| v.to_string());
    let updated = state
        .scheduler
        .update_canvas(&id, title, state_json.as_deref(), &utc_stamp())
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    if !updated {
        return Err(fail(StatusCode::NOT_FOUND, "canvas session not found"));
    }
    get_canvas(State(state), Path(id)).await
}

async fn delete_canvas(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let deleted = state
        .scheduler
        .delete_canvas(&id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    if !deleted {
        return Err(fail(StatusCode::NOT_FOUND, "canvas session not found"));
    }
    Ok(Json(json!({ "ok": true })))
}

fn settings_path(state: &AppState) -> PathBuf {
    state.workspace.join("config").join(".app_config.json")
}

fn presets_path(state: &AppState) -> PathBuf {
    state.workspace.join("config").join("presets.json")
}

fn default_settings() -> Value {
    json!({
        "language": "zh",
        "theme": "mac",
        "auto_save_prompts": true,
        "output_format": "png",
        "default_model_llm": "",
        "default_model_vlm": "",
        "default_model_llm_think": false,
        "mlx_memory_limit": 0,
        "model_cache_ttl_minutes": 30,
        "llm_cache_ttl_minutes": 30,
        "llm_unload_each_request": false,
        "queue_image_first": true,
        "civitai_token": "",
        "huggingface_token": "",
        "nsfw_enabled": false,
        "http_api_key_configured": false,
        "http_api_key_hint": "",
        "http_api_key_from_env": false,
        "mcp_api_key_configured": false,
        "mcp_api_key_hint": "",
        "mcp_api_key_from_env": false,
        "custom_workspace_dir": "",
    })
}

fn load_json(path: &FsPath, fallback: Value) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(fallback)
}

fn save_json(path: &FsPath, value: &Value) -> Result<(), ApiError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap_or_default())
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
}

async fn get_settings(State(state): State<AppState>) -> Json<Value> {
    let mut settings = default_settings();
    if let Value::Object(defaults) = &mut settings {
        if let Value::Object(saved) = load_json(&settings_path(&state), json!({})) {
            for (key, value) in saved {
                defaults.insert(key, value);
            }
        }
    }
    Json(settings)
}

async fn put_settings(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let current = get_settings(State(state.clone())).await.0;
    let mut merged = current;
    if let (Some(dst), Some(src)) = (merged.as_object_mut(), body.as_object()) {
        for (key, value) in src {
            dst.insert(key.clone(), value.clone());
        }
    }
    save_json(&settings_path(&state), &merged)?;
    Ok(Json(json!({ "success": true, "restart_required": false })))
}

async fn create_access_key(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let kind = body.get("kind").and_then(|v| v.as_str()).unwrap_or("http");
    if kind != "http" && kind != "mcp" {
        return Err(fail(StatusCode::BAD_REQUEST, "kind must be http or mcp"));
    }
    let key = Uuid::new_v4().simple().to_string();
    let hint: String = key
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let mut settings = get_settings(State(state.clone())).await.0;
    if let Some(obj) = settings.as_object_mut() {
        obj.insert(format!("{kind}_api_key_configured"), json!(true));
        obj.insert(format!("{kind}_api_key_hint"), json!(hint));
    }
    save_json(&settings_path(&state), &settings)?;
    Ok(Json(json!({
        "success": true,
        "kind": kind,
        "key": key,
        "hint": settings.get(&format!("{kind}_api_key_hint")).cloned().unwrap_or(Value::Null),
    })))
}

async fn revoke_access_key(
    State(state): State<AppState>,
    Path(kind): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if kind != "http" && kind != "mcp" {
        return Err(fail(StatusCode::BAD_REQUEST, "kind must be http or mcp"));
    }
    let mut settings = get_settings(State(state.clone())).await.0;
    if let Some(obj) = settings.as_object_mut() {
        obj.insert(format!("{kind}_api_key_configured"), json!(false));
        obj.insert(format!("{kind}_api_key_hint"), json!(""));
    }
    save_json(&settings_path(&state), &settings)?;
    Ok(Json(json!({ "success": true, "kind": kind })))
}

async fn workspace_paths(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "workspace": state.workspace.display().to_string(),
        "models": state.workspace.join("models").display().to_string(),
        "outputs": state.outputs.display().to_string(),
        "config": state.workspace.join("config").display().to_string(),
        "db": state.workspace.join("db").join("studio.db").display().to_string(),
    }))
}

async fn workspace_status(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "configured": true,
        "effective_root": state.workspace.display().to_string(),
        "bootstrap_root": state.workspace.display().to_string(),
        "control_plane": state.workspace.display().to_string(),
    }))
}

async fn apply_workspace(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let path = body
        .get("path")
        .and_then(|v| v.as_str())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "path is required"))?;
    let mut settings = get_settings(State(state.clone())).await.0;
    if let Some(obj) = settings.as_object_mut() {
        obj.insert("custom_workspace_dir".into(), json!(path));
    }
    save_json(&settings_path(&state), &settings)?;
    Ok(Json(json!({
        "success": true,
        "restart_required": true,
        "workspace": path,
    })))
}

async fn pick_workspace() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "directory picker is not available on this server; pass the path to apply-workspace",
    ))
}

async fn restore_defaults(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let files = body
        .get("files")
        .and_then(|v| v.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![".app_config.json".into(), "presets.json".into()]);
    let mut restored = Vec::new();
    for file in files {
        if file == ".app_config.json" || file == "settings" {
            save_json(&settings_path(&state), &default_settings())?;
            restored.push(".app_config.json".to_string());
        } else if file == "presets.json" {
            save_json(&presets_path(&state), &json!({}))?;
            restored.push("presets.json".to_string());
        }
    }
    Ok(Json(json!({
        "success": true,
        "restored": restored,
        "restart_required": false,
    })))
}

async fn system_info(State(state): State<AppState>) -> Json<Value> {
    let (total, avail) = meminfo();
    Json(json!({
        "env_ready": true,
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "memory_gb": total,
        "memory_available_gb": avail,
        "mlx_memory_limit": 0,
        "controlnet_runtime_available": false,
        "runtime": "ggml",
        "backend": format!("{:?}", state.backend).to_lowercase(),
    }))
}

async fn system_monitor(State(state): State<AppState>) -> Json<Value> {
    let info = system_info(State(state)).await.0;
    Json(info)
}

async fn install_runtime() -> Json<Value> {
    Json(json!({
        "success": true,
        "message": "ggml runtime is built into danqing-api",
    }))
}

async fn settings_registry(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    Ok(Json(catalog.public_registry()))
}

async fn disk_space(State(state): State<AppState>) -> Json<Value> {
    let (total, free) = stat_space(&state.workspace);
    Json(json!({
        "path": state.workspace.display().to_string(),
        "total_bytes": total,
        "free_bytes": free,
        "total_human": human_bytes(total),
        "free_human": human_bytes(free),
    }))
}

async fn cache_status() -> Json<Value> {
    Json(json!({ "models": [], "llm": null }))
}

async fn get_presets(State(state): State<AppState>) -> Json<Value> {
    Json(load_json(&presets_path(&state), json!({})))
}

async fn save_preset(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "name is required"))?;
    let preset = body.get("preset").cloned().unwrap_or(json!({}));
    let mut all = load_json(&presets_path(&state), json!({}));
    if let Some(obj) = all.as_object_mut() {
        obj.insert(name.to_string(), preset);
    }
    save_json(&presets_path(&state), &all)?;
    Ok(Json(all))
}

async fn delete_preset(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let mut all = load_json(&presets_path(&state), json!({}));
    if let Some(obj) = all.as_object_mut() {
        obj.remove(&name);
    }
    save_json(&presets_path(&state), &all)?;
    Ok(Json(json!({ "ok": true })))
}

async fn reject_download() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "no GGUF download URL is published. Re-download when a bf16, q8_0, or q4_k package is listed",
    ))
}

async fn reject_convert() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "safetensors conversion was removed. Download a GGUF bundle (bf16, q8_0, or q4_k)",
    ))
}

async fn reject_training() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::NOT_IMPLEMENTED,
        "lora training graph is not built",
    ))
}

async fn training_requirements() -> Json<Value> {
    Json(json!({
        "ready": false,
        "detail": "lora training graph is not built",
    }))
}

async fn reject_merge() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::BAD_REQUEST,
        "weight merge was removed. GGUF bundles are downloaded as published",
    ))
}

async fn merge_models() -> Json<Value> {
    Json(json!({ "models": [], "mlx_available": false }))
}

async fn list_audio_tasks(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    list_tasks(
        State(state),
        Query(ListQuery {
            kind: Some("audio.generation".into()),
            limit: Some(100),
            ..ListQuery::default()
        }),
    )
    .await
}

async fn reject_assistant() -> Result<Json<Value>, ApiError> {
    Err(fail(
        StatusCode::NOT_IMPLEMENTED,
        "assistant graph is not built",
    ))
}

pub async fn spa(uri: axum::http::Uri) -> Response {
    let path = uri.path();
    if path.starts_with("/api/") || path == "/api" {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let Some(root) = frontend_dist() else {
        return (StatusCode::NOT_FOUND, "frontend dist is not built").into_response();
    };
    let rel = path.trim_start_matches('/');
    let candidate = if rel.is_empty() {
        root.join("index.html")
    } else {
        root.join(rel)
    };
    if !candidate.starts_with(&root) {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let file = if candidate.is_file() {
        candidate
    } else if !rel.contains('.') {
        root.join("index.html")
    } else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    match std::fs::read(&file) {
        Ok(bytes) => {
            let mime = mime_for(
                file.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("html"),
            );
            ([(header::CONTENT_TYPE, mime)], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn frontend_dist() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("DANQING_FRONTEND_DIST") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("out/frontend/dist"));
    candidates.push(PathBuf::from("frontend/dist"));
    candidates
        .into_iter()
        .find(|path| path.join("index.html").is_file())
}

fn open_catalog(state: &AppState) -> Result<Catalog, ApiError> {
    let catalog = Catalog::open(&state.registry)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    materialize_graph_packs(&catalog)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err))?;
    Ok(catalog)
}

pub fn new_id(prefix: &str) -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("{prefix}{}", &raw[..20])
}

fn mime_for(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "json" => "application/json",
        "js" => "text/javascript",
        "css" => "text/css",
        "svg" => "image/svg+xml",
        "html" => "text/html; charset=utf-8",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn kind_for(mime: &str) -> &'static str {
    if mime.starts_with("image/") {
        "image"
    } else if mime.starts_with("video/") {
        "video"
    } else if mime.starts_with("audio/") {
        "audio"
    } else {
        "file"
    }
}

fn multipart_file(content_type: &str, body: &[u8]) -> Result<(String, Vec<u8>), String> {
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .ok_or("multipart body has no boundary")?
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"');
    if boundary.is_empty() {
        return Err("multipart body has no boundary".into());
    }
    let marker = format!("--{boundary}").into_bytes();
    let mut start = 0usize;
    while let Some(rel) = find_bytes(&body[start..], &marker) {
        let part = start + rel + marker.len();
        if body.get(part..part + 2) == Some(b"--") {
            break;
        }
        let header_at = if body.get(part..part + 2) == Some(b"\r\n") {
            part + 2
        } else {
            part
        };
        let Some(header_end_rel) = find_bytes(&body[header_at..], b"\r\n\r\n") else {
            break;
        };
        let header_end = header_at + header_end_rel;
        let headers = String::from_utf8_lossy(&body[header_at..header_end]);
        let data_start = header_end + 4;
        let next = find_bytes(&body[data_start..], &marker).unwrap_or(body.len() - data_start);
        let mut data_end = data_start + next;
        if data_end >= 2 && &body[data_end - 2..data_end] == b"\r\n" {
            data_end -= 2;
        }
        if headers.to_ascii_lowercase().contains("name=\"file\"")
            || headers.to_ascii_lowercase().contains("name=file")
        {
            let filename = headers
                .split("filename=")
                .nth(1)
                .map(|rest| {
                    rest.trim_start_matches('"')
                        .split('"')
                        .next()
                        .unwrap_or("upload.bin")
                        .to_string()
                })
                .unwrap_or_else(|| "upload.bin".into());
            return Ok((filename, body[data_start..data_end].to_vec()));
        }
        start = data_start + next;
    }
    Err("multipart body has no file field".into())
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len())
        .position(|window| window == needle)
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    fn val(byte: u8) -> Result<u8, String> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err("data_base64 is not valid".into()),
        }
    }
    let bytes: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return Err("data_base64 is not valid".into());
    }
    let mut out = Vec::new();
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|b| **b == b'=').count();
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = if chunk[2] == b'=' { 0 } else { val(chunk[2])? };
        let d = if chunk[3] == b'=' { 0 } else { val(chunk[3])? };
        let triple = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | d as u32;
        out.push((triple >> 16) as u8);
        if pad < 2 {
            out.push((triple >> 8) as u8);
        }
        if pad < 1 {
            out.push(triple as u8);
        }
    }
    Ok(out)
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn dir_size(path: &FsPath) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            total += entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        } else if path.is_dir() {
            total += dir_size(&path);
        }
    }
    total
}

fn meminfo() -> (f64, f64) {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return (0.0, 0.0);
    };
    let mut total = 0f64;
    let mut avail = 0f64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = kib(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail = kib(rest);
        }
    }
    (total, avail)
}

fn kib(text: &str) -> f64 {
    text.split_whitespace()
        .next()
        .and_then(|n| n.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1024.0
        / 1024.0
}

fn stat_space(path: &FsPath) -> (u64, u64) {
    #[cfg(unix)]
    {
        #[repr(C)]
        struct Statvfs {
            f_bsize: u64,
            f_frsize: u64,
            f_blocks: u64,
            f_bfree: u64,
            f_bavail: u64,
            _pad: [u64; 8],
        }
        extern "C" {
            fn statvfs(path: *const i8, buf: *mut Statvfs) -> i32;
        }
        let c = std::ffi::CString::new(path.display().to_string()).ok();
        let Some(c) = c else {
            return (0, 0);
        };
        let mut buf = Statvfs {
            f_bsize: 0,
            f_frsize: 0,
            f_blocks: 0,
            f_bfree: 0,
            f_bavail: 0,
            _pad: [0; 8],
        };
        let rc = unsafe { statvfs(c.as_ptr(), &mut buf) };
        if rc != 0 {
            return (0, 0);
        }
        let fr = if buf.f_frsize == 0 {
            buf.f_bsize
        } else {
            buf.f_frsize
        };
        (
            buf.f_blocks.saturating_mul(fr),
            buf.f_bavail.saturating_mul(fr),
        )
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        (0, 0)
    }
}

fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let tod = secs % 86_400;
    let hour = tod / 3600;
    let min = (tod % 3600) / 60;
    let sec = tod % 60;
    let (year, month, day) = civil_from_days(days as i64);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{min:02}:{sec:02}")
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}
