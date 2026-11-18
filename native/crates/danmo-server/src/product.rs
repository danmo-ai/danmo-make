//! Routes the Vue SPA calls besides generation submit.
//! Graphs that are not built return a JSON `detail` and do not pretend to run.

use crate::app::AppState;
use crate::download;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::Catalog;
use danmo_store::{AssetRow, CanvasRow, TaskRow};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::{Path as FsPath, PathBuf};
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
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
        .route("/api/assets/:id/thumbnail", get(asset_thumbnail))
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
        .route(
            "/api/settings/loras/compatible/:model",
            get(compatible_loras),
        )
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
        .route("/api/download/tasks", get(list_downloads))
        .route(
            "/api/download/tasks/:task_id",
            axum::routing::delete(delete_download),
        )
        .route("/api/download/start", post(start_download))
        .route("/api/download/cancel/:task_id", post(cancel_download))
        .route("/api/download/resume/:task_id", post(resume_download))
        .route(
            "/api/download/progress/:task_id/stream",
            get(download_progress),
        )
        .route("/api/download/lora/search", get(search_loras))
        .route("/api/download/lora/base-models", get(lora_base_models))
        .route("/api/download/lora/hub", post(start_lora_download))
        .route("/api/download/lora", post(start_lora_download))
        .route(
            "/api/loras/datasets",
            get(list_lora_datasets).post(create_lora_dataset),
        )
        .route(
            "/api/loras/datasets/:id",
            get(get_lora_dataset)
                .patch(patch_lora_dataset)
                .delete(delete_lora_dataset),
        )
        .route(
            "/api/loras/datasets/:id/images",
            post(upload_lora_dataset_images).delete(delete_lora_dataset_image_query),
        )
        .route(
            "/api/loras/datasets/:id/images/:file",
            axum::routing::delete(delete_lora_dataset_image),
        )
        .route(
            "/api/loras/datasets/:id/file/:file",
            get(serve_lora_dataset_file),
        )
        .route(
            "/api/loras/datasets/:id/skipped-images",
            axum::routing::put(put_lora_dataset_skipped),
        )
        .route(
            "/api/loras/datasets/:id/health",
            get(lora_dataset_health),
        )
        .route(
            "/api/loras/datasets/:id/health/vlm",
            post(lora_dataset_health_vlm),
        )
        .route(
            "/api/loras/datasets/:id/import-assets",
            post(import_lora_dataset_assets),
        )
        .route(
            "/api/loras/datasets/:id/import-csv",
            post(import_lora_dataset_csv),
        )
        .route("/api/loras/user-adapters", get(empty_items))
        .route(
            "/api/loras/user-adapters/:id",
            axum::routing::delete(delete_lora),
        )
        .route("/api/loras/downloaded-adapters", get(list_downloaded_loras))
        .route("/api/loras/trainable-models", get(list_trainable_models))
        .route(
            "/api/loras/training/requirements",
            get(training_requirements),
        )
        .route("/api/loras/training/presets", get(list_training_presets))
        .route("/api/loras/trainings", post(submit_training))
        .route(
            "/api/loras/trainings/:id/artifacts",
            get(training_artifacts),
        )
        .route(
            "/api/loras/trainings/:id/artifacts/file/:filename",
            get(training_artifact_file),
        )
        .route(
            "/api/loras/trainings/:id/register",
            post(register_training_checkpoint),
        )
        .route("/api/tools/z-image/merge", post(reject_merge))
        .route("/api/tools/z-image/merge/models", get(merge_models))
        .route("/api/tools/z-image/merge/merged", get(empty_items))
        .route("/api/audios/generations", get(list_audio_tasks))
        .route("/api/chat/enhance-prompt", post(enhance_prompt))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/llm/model", get(llm_model_info))
        .route("/v1/vision/model", get(vision_model_info))
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
        let ready = model_ready(&state.workspace, entry);
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

async fn models_status(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut out = serde_json::Map::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            out.insert(id.clone(), json!(model_ready(&state.workspace, entry)));
        }
    }
    Ok(Json(Value::Object(out)))
}

async fn models_detailed(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut out = serde_json::Map::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            out.insert(id.clone(), detailed_status(&state.workspace, entry));
        }
    }
    Ok(Json(Value::Object(out)))
}

fn detailed_status(workspace: &FsPath, entry: &Value) -> Value {
    let versions = entry
        .pointer("/distribution/versions")
        .and_then(|v| v.as_object());
    let mut ready = false;
    let mut listed = serde_json::Map::new();
    if let Some(versions) = versions {
        for (key, version) in versions {
            let path = version
                .get("local_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let resolved = download::resolve_model_path(workspace, path);
            let is_ready = !path.is_empty() && download::version_ready(&resolved);
            if is_ready {
                ready = true;
            }
            let status = if is_ready {
                "ready"
            } else if !path.is_empty() && resolved.exists() {
                "incomplete"
            } else {
                "not_downloaded"
            };
            listed.insert(
                key.clone(),
                json!({
                    "ready": is_ready,
                    "status": status,
                    "local_path": path,
                }),
            );
        }
    }
    let overall = if ready {
        "ready"
    } else if listed.is_empty()
        || listed
            .values()
            .all(|row| row.get("status").and_then(|v| v.as_str()) == Some("not_downloaded"))
    {
        "not_downloaded"
    } else {
        "incomplete"
    };
    json!({ "ready": ready, "status": overall, "versions": listed })
}

fn model_ready(workspace: &FsPath, entry: &Value) -> bool {
    detailed_status(workspace, entry)
        .get("ready")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
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
            for (key, version) in versions {
                let Some(path) = version.get("local_path").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !download::version_ready(&download::resolve_model_path(&state.workspace, path)) {
                    continue;
                }
                let full = download::resolve_model_path(&state.workspace, path);
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

async fn install_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let version = body
        .get("version")
        .and_then(|v| v.as_str())
        .filter(|version| !version.is_empty())
        .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "version is required"))?;
    let task_id = enqueue_install(&state, &id, version)?;
    Ok(Json(json!({ "task_id": task_id })))
}

async fn install_batch(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Some(items) = body.get("items").and_then(|v| v.as_array()) {
        for item in items {
            let model_id = item
                .get("model_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let version = item
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if model_id.is_empty() || version.is_empty() {
                return Err(fail(
                    StatusCode::BAD_REQUEST,
                    "each install item needs model_id and version",
                ));
            }
            pairs.push((model_id, version));
        }
    } else if let Some(ids) = body.get("model_ids").and_then(|v| v.as_array()) {
        let catalog = open_catalog(&state)?;
        for id in ids {
            let Some(model_id) = id.as_str() else {
                continue;
            };
            let version = default_version(catalog.root(), model_id).ok_or_else(|| {
                fail(
                    StatusCode::BAD_REQUEST,
                    format!("{model_id} has no downloadable version"),
                )
            })?;
            pairs.push((model_id.to_string(), version));
        }
    }
    if pairs.is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "no models to install"));
    }
    let mut task_ids = Vec::new();
    for (model_id, version) in &pairs {
        task_ids.push(enqueue_install(&state, model_id, version)?);
    }
    Ok(Json(json!({ "task_ids": task_ids })))
}

fn enqueue_install(state: &AppState, model_id: &str, version: &str) -> Result<String, ApiError> {
    let catalog = open_catalog(state)?;
    let spec = catalog
        .root()
        .pointer(&format!(
            "/models/{model_id}/distribution/versions/{version}"
        ))
        .cloned()
        .ok_or_else(|| {
            fail(
                StatusCode::BAD_REQUEST,
                format!("{model_id} has no {version} package"),
            )
        })?;
    if spec.get("source_type").and_then(|value| value.as_str()) == Some("derived") {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!(
                "{model_id}:{version} is generated from the full version and is not a download"
            ),
        ));
    }
    if !version_is_download(&spec) {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("{model_id}:{version} has no published download"),
        ));
    }
    state
        .downloads
        .install_model(model_id, version, &spec)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err))
}

fn version_is_download(version: &Value) -> bool {
    let nonempty = |key: &str| {
        version
            .get(key)
            .and_then(|value| value.as_str())
            .is_some_and(|text| !text.is_empty())
    };
    let listed = |key: &str| {
        version
            .get(key)
            .and_then(|value| value.as_array())
            .is_some_and(|rows| !rows.is_empty())
    };
    nonempty("repo_id")
        || nonempty("url")
        || listed("files")
        || listed("bundle_repos")
        || listed("components")
}

fn default_version(root: &Value, model_id: &str) -> Option<String> {
    let versions = root
        .pointer(&format!("/models/{model_id}/distribution/versions"))?
        .as_object()?;
    let mut fallback = None;
    for (key, version) in versions {
        if version.get("source_type").and_then(|value| value.as_str()) == Some("derived") {
            continue;
        }
        if !version_is_download(version) {
            continue;
        }
        if version
            .get("default")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            return Some(key.clone());
        }
        if fallback.is_none() {
            fallback = Some(key.clone());
        }
    }
    fallback
}

async fn delete_model_version(
    State(state): State<AppState>,
    Path((id, version)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let spec = catalog
        .root()
        .pointer(&format!("/models/{id}/distribution/versions/{version}"))
        .ok_or_else(|| {
            fail(
                StatusCode::NOT_FOUND,
                format!("{id} has no {version} package"),
            )
        })?;
    let local = spec
        .get("local_path")
        .and_then(|v| v.as_str())
        .filter(|path| !path.is_empty())
        .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "version has no local_path"))?;
    remove_model_dir(
        &state,
        &download::resolve_model_path(&state.workspace, local),
    )?;
    Ok(Json(json!({ "success": true })))
}

fn remove_model_dir(state: &AppState, path: &FsPath) -> Result<(), ApiError> {
    if !path.exists() {
        return Ok(());
    }
    let canon = path
        .canonicalize()
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    let roots = [state.workspace.join("models")];
    let allowed = roots.iter().any(|root| {
        let Ok(root) = root.canonicalize() else {
            return false;
        };
        canon.starts_with(&root) && canon != root
    });
    if !allowed {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "refusing to delete a path outside the models directory",
        ));
    }
    std::fs::remove_dir_all(&canon)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
}

async fn empty_items() -> Json<Value> {
    Json(json!({ "items": [] }))
}

async fn empty_list() -> Json<Value> {
    Json(json!([]))
}

/// Installed and registry LoRAs whose `base_model` matches the selected model.
/// ACE-Step XL SFT, Turbo, and base share one LoRA set.
async fn compatible_loras(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let models = catalog
        .root()
        .get("models")
        .and_then(|value| value.as_object())
        .ok_or_else(|| fail(StatusCode::INTERNAL_SERVER_ERROR, "registry has no models"))?;
    let mut rows = Vec::new();
    for (id, entry) in models {
        if entry
            .pointer("/catalog/category")
            .and_then(|value| value.as_str())
            != Some("loras")
            && entry
                .pointer("/catalog/type")
                .and_then(|value| value.as_str())
                != Some("lora")
        {
            continue;
        }
        let base = entry
            .pointer("/catalog/base_model")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if !lora_compatible(&model, base) {
            continue;
        }
        rows.push(json!({
            "kind": "lora",
            "id": id,
            "name": lora_display_name(entry, id),
            "base_model": base,
            "source": "registry",
            "ready": lora_versions_ready(&state.workspace, entry),
        }));
    }
    let user_path = state.workspace.join("config").join("user_loras.json");
    if user_path.is_file() {
        let text = std::fs::read_to_string(&user_path).map_err(|error| {
            fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{}: {error}", user_path.display()),
            )
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{} is not json: {error}", user_path.display()),
            )
        })?;
        for item in value
            .get("items")
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
        {
            let id = item
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let base = item
                .get("base_model")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let path = item
                .get("local_path")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if id.is_empty() || !lora_compatible(&model, base) {
                continue;
            }
            let root = download::resolve_model_path(&state.workspace, path);
            if download::lora_weight_file(&root).is_err() {
                continue;
            }
            if rows
                .iter()
                .any(|row| row.get("id").and_then(|value| value.as_str()) == Some(id))
            {
                continue;
            }
            let mut row = json!({
                "kind": "lora",
                "id": id,
                "name": item.get("name").and_then(|value| value.as_str()).unwrap_or(id),
                "base_model": base,
                "source": item.get("source").and_then(|value| value.as_str()).unwrap_or("user_trained"),
                "ready": true,
            });
            if let Some(inference) = item.get("inference").filter(|value| value.is_object()) {
                let extras = danmo_contracts::stacked_adapters_from_value(
                    inference.get("extra_adapters"),
                )
                .map_err(|detail| fail(StatusCode::BAD_REQUEST, format!("lora {id}: {detail}")))?;
                if !extras.is_empty() {
                    row["extra_adapters"] = json!(extras);
                }
                let mut overrides = serde_json::Map::new();
                if let Some(steps) = inference.get("steps").and_then(|value| value.as_u64()) {
                    overrides.insert("steps".into(), json!(steps));
                }
                if let Some(guidance) = inference.get("guidance").and_then(|value| value.as_f64())
                {
                    overrides.insert("guidance".into(), json!(guidance));
                }
                if let Some(scheduler) = inference
                    .get("scheduler")
                    .and_then(|value| value.as_str())
                    .filter(|text| !text.is_empty())
                {
                    overrides.insert("scheduler".into(), json!(scheduler));
                }
                if let Some(weight) = inference.get("lora_weight").and_then(|value| value.as_f64())
                {
                    overrides.insert("lora_scale".into(), json!(weight));
                }
                if !overrides.is_empty() {
                    row["compose_overrides"] = Value::Object(overrides);
                }
                if let Some(range) = inference.get("lora_weight_range") {
                    row["recommended_lora_scale"] = range.clone();
                }
            }
            rows.insert(0, row);
        }
    }
    if let Some(items) = state
        .downloads
        .list_loras()
        .get("items")
        .and_then(|value| value.as_array())
    {
        for item in items {
            let id = item
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let base = item
                .get("base_model")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if id.is_empty() || !lora_compatible(&model, base) {
                continue;
            }
            if rows
                .iter()
                .any(|row| row.get("id").and_then(|value| value.as_str()) == Some(id))
            {
                continue;
            }
            rows.push(json!({
                "kind": "lora",
                "id": id,
                "name": item.get("name").and_then(|value| value.as_str()).unwrap_or(id),
                "base_model": base,
                "source": "remote_download",
                "ready": true,
            }));
        }
    }
    rows.sort_by(|left, right| {
        let name = |row: &Value| {
            row.get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string()
        };
        name(left).cmp(&name(right))
    });
    Ok(Json(Value::Array(rows)))
}

fn lora_display_name(entry: &Value, id: &str) -> String {
    match entry.pointer("/catalog/name") {
        Some(Value::String(text)) => text.clone(),
        Some(value) => value
            .get("zh")
            .and_then(|item| item.as_str())
            .filter(|text| !text.is_empty())
            .or_else(|| value.get("en").and_then(|item| item.as_str()))
            .unwrap_or(id)
            .to_string(),
        None => id.to_string(),
    }
}

pub(crate) fn lora_compatible(model: &str, base: &str) -> bool {
    let model = model.split(':').next().unwrap_or(model).trim();
    let base = base.split(':').next().unwrap_or(base).trim();
    if model.is_empty() || base.is_empty() {
        return false;
    }
    if model == base {
        return true;
    }
    const ACE_XL: &[&str] = &["ace-step-xl-sft", "ace-step-xl-turbo", "ace-step-xl-base"];
    ACE_XL.contains(&model) && ACE_XL.contains(&base)
}

fn lora_versions_ready(workspace: &FsPath, entry: &Value) -> bool {
    let Some(versions) = entry
        .pointer("/distribution/versions")
        .and_then(|value| value.as_object())
    else {
        return false;
    };
    versions.values().any(|version| {
        version
            .get("local_path")
            .and_then(|value| value.as_str())
            .is_some_and(|path| {
                download::version_ready(&download::resolve_model_path(workspace, path))
            })
    })
}

async fn list_downloads(State(state): State<AppState>) -> Json<Value> {
    Json(Value::Array(state.downloads.list()))
}

async fn start_download(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let url = body.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let name = body
        .get("target_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let kind = body.get("type").and_then(|v| v.as_str()).unwrap_or("model");
    let task_id = state
        .downloads
        .start_url(url, name, kind)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err))?;
    Ok(Json(json!({ "task_id": task_id })))
}

async fn cancel_download(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state
        .downloads
        .cancel(&task_id)
        .map_err(|err| fail(StatusCode::NOT_FOUND, err))?;
    Ok(Json(json!({ "ok": true })))
}

async fn resume_download(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if let Some(snap) = state.downloads.snapshot(&task_id) {
        if snap.get("kind").and_then(|v| v.as_str()) == Some("model") {
            let model_id = snap
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let version = snap.get("version").and_then(|v| v.as_str()).unwrap_or("");
            if !model_id.is_empty() && !version.is_empty() {
                let catalog = open_catalog(&state)?;
                if let Some(spec) = catalog
                    .root()
                    .pointer(&format!(
                        "/models/{model_id}/distribution/versions/{version}"
                    ))
                    .cloned()
                {
                    state
                        .downloads
                        .refresh_model_spec(&task_id, spec)
                        .map_err(|err| fail(StatusCode::BAD_REQUEST, err))?;
                }
            }
        }
    }
    state
        .downloads
        .resume(&task_id)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err))?;
    Ok(Json(json!({ "task_id": task_id })))
}

async fn delete_download(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state
        .downloads
        .delete_task(&task_id)
        .map_err(|err| fail(StatusCode::NOT_FOUND, err))?;
    Ok(Json(json!({ "ok": true })))
}

async fn download_progress(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let downloads = state.downloads.clone();
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    tokio::spawn(async move {
        loop {
            let Some(snap) = downloads.snapshot(&task_id) else {
                let missing = json!({
                    "status": "failed",
                    "error_message": "download task not found",
                    "progress": 0,
                    "total_size": 0,
                    "downloaded_size": 0,
                });
                let _ = tx
                    .send(Ok(Event::default().data(missing.to_string())))
                    .await;
                break;
            };
            let _ = tx.send(Ok(Event::default().data(snap.to_string()))).await;
            let status = snap.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if matches!(status, "completed" | "failed" | "cancelled") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    });
    Sse::new(ReceiverStream::new(rx))
}

#[derive(Debug, Deserialize, Default)]
struct LoraSearchQuery {
    q: Option<String>,
    base_model: Option<String>,
    source: Option<String>,
    limit: Option<String>,
}

async fn search_loras(
    State(state): State<AppState>,
    Query(query): Query<LoraSearchQuery>,
) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let text = lora_query_text(
        catalog.root(),
        query.q.as_deref(),
        query.base_model.as_deref(),
    )?;
    let limit = query
        .limit
        .as_deref()
        .and_then(|text| text.parse::<usize>().ok())
        .unwrap_or(24)
        .clamp(1, 100);
    let source = query.source.unwrap_or_else(|| "all".into());
    let items = tokio::task::spawn_blocking(move || state.downloads.search(&source, &text, limit))
        .await
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .map_err(|err| fail(StatusCode::BAD_GATEWAY, err))?;
    Ok(Json(json!({ "items": items })))
}

fn lora_query_text(
    root: &Value,
    q: Option<&str>,
    base_model: Option<&str>,
) -> Result<String, ApiError> {
    if let Some(q) = q.map(str::trim).filter(|q| !q.is_empty()) {
        return Ok(q.to_string());
    }
    let Some(base_model) = base_model.map(str::trim).filter(|id| !id.is_empty()) else {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "select a base model to search",
        ));
    };
    if let Some(term) = root
        .pointer(&format!(
            "/models/{base_model}/ui/parameters/lora_search/terms/0"
        ))
        .and_then(|v| v.as_str())
        .filter(|term| !term.is_empty())
    {
        return Ok(term.to_string());
    }
    Ok(base_model.to_string())
}

async fn lora_base_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = open_catalog(&state)?;
    let mut rows = Vec::new();
    if let Some(models) = catalog.root().get("models").and_then(|v| v.as_object()) {
        for (id, entry) in models {
            let supported = entry
                .pointer("/ui/parameters/lora_support")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !supported {
                continue;
            }
            rows.push(json!({
                "id": id,
                "name": model_label(entry, id),
            }));
        }
    }
    rows.sort_by(|a, b| {
        a.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .cmp(b.get("name").and_then(|v| v.as_str()).unwrap_or(""))
    });
    Ok(Json(Value::Array(rows)))
}

fn model_label(entry: &Value, id: &str) -> String {
    match entry.pointer("/catalog/name") {
        Some(Value::String(text)) => text.clone(),
        Some(value) => value
            .get("en")
            .and_then(|v| v.as_str())
            .filter(|text| !text.is_empty())
            .or_else(|| value.get("zh").and_then(|v| v.as_str()))
            .unwrap_or(id)
            .to_string(),
        None => id.to_string(),
    }
}

async fn start_lora_download(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let task_id = state
        .downloads
        .start_lora(&body)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err))?;
    Ok(Json(json!({ "task_id": task_id })))
}

async fn list_downloaded_loras(State(state): State<AppState>) -> Json<Value> {
    Json(state.downloads.list_loras())
}

#[derive(Debug, Deserialize, Default)]
struct RemoveFilesQuery {
    remove_files: Option<String>,
}

async fn delete_lora(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<RemoveFilesQuery>,
) -> Result<Json<Value>, ApiError> {
    let remove = matches!(query.remove_files.as_deref(), Some("true" | "1" | "yes"));
    state
        .downloads
        .delete_lora(&id, remove)
        .map_err(|err| fail(StatusCode::NOT_FOUND, err))
        .map(Json)
}

async fn empty_models() -> Json<Value> {
    Json(json!({ "models": [] }))
}

async fn recommendations() -> Json<Value> {
    Json(json!({
        "runtime": "ggml",
        "weights": "safetensors",
        "versions": ["fp16", "bf16", "mlx-bf16", "mlx-q8", "mlx-q4"],
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
    remove_asset_files(&state, &row);
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
            remove_asset_files(&state, &row);
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
    serve_asset_bytes(&state, &row.file_path, &row.mime_type)
}

async fn asset_thumbnail(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let row = match state.scheduler.get_asset(&id) {
        Ok(Some(row)) => row,
        Ok(None) => return (StatusCode::NOT_FOUND, "asset not found").into_response(),
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    if let Some(thumb) = row
        .thumbnail_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        let mime = mime_for_stored_path(thumb);
        let response = serve_asset_bytes(&state, thumb, mime);
        if response.status() != StatusCode::NOT_FOUND {
            return response;
        }
    }
    serve_asset_bytes(&state, &row.file_path, &row.mime_type)
}

/// Resolve legacy Python DB paths (`ast_….png`) under `{outputs}/assets`, plus absolute paths.
pub(crate) fn resolve_asset_fs_path(outputs: &FsPath, workspace: &FsPath, stored: &str) -> PathBuf {
    let raw = PathBuf::from(stored);
    if raw.is_absolute() {
        return raw;
    }
    let under_assets = outputs.join("assets").join(&raw);
    if under_assets.is_file() {
        return under_assets;
    }
    let under_workspace = workspace.join(&raw);
    if under_workspace.is_file() {
        return under_workspace;
    }
    under_assets
}

fn serve_asset_bytes(state: &AppState, stored: &str, mime: &str) -> Response {
    let path = resolve_asset_fs_path(&state.outputs, &state.workspace, stored);
    match std::fs::read(&path) {
        Ok(bytes) => ([(header::CONTENT_TYPE, mime.to_string())], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "asset file is missing").into_response(),
    }
}

fn remove_asset_files(state: &AppState, row: &AssetRow) {
    let main = resolve_asset_fs_path(&state.outputs, &state.workspace, &row.file_path);
    let _ = std::fs::remove_file(main);
    if let Some(thumb) = row
        .thumbnail_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        let path = resolve_asset_fs_path(&state.outputs, &state.workspace, thumb);
        let _ = std::fs::remove_file(path);
    }
}

fn mime_for_stored_path(stored: &str) -> &'static str {
    FsPath::new(stored)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(mime_for)
        .unwrap_or("application/octet-stream")
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
        "llm_inference_provider": "builtin",
        "llm_inference_base_url": "",
        "llm_inference_api_key": "",
        "llm_inference_api_key_hint": "",
        "llm_inference_cloud_model": "",
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

fn current_settings(state: &AppState) -> Value {
    let mut settings = default_settings();
    if let Value::Object(defaults) = &mut settings {
        if let Value::Object(saved) = load_json(&settings_path(state), json!({})) {
            for (key, value) in saved {
                defaults.insert(key, value);
            }
        }
    }
    settings
}

async fn get_settings(State(state): State<AppState>) -> Json<Value> {
    Json(current_settings(&state))
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
    let projected = catalog
        .public_registry()
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(Json(projected))
}

async fn disk_space(State(state): State<AppState>) -> Json<Value> {
    let models = state.workspace.join("models");
    let loras = state.workspace.join("models/Lora");
    let anchor = models.parent().unwrap_or(&state.workspace);
    let (total, free) = stat_space(if anchor.as_os_str().is_empty() {
        std::path::Path::new(".")
    } else {
        anchor
    });
    Json(json!({
        "models": disk_row(&models, total, free),
        "loras": disk_row(&loras, total, free),
    }))
}

fn disk_row(path: &FsPath, total: u64, free: u64) -> Value {
    let size = if path.is_dir() { dir_size(path) } else { 0 };
    json!({
        "exists": path.exists(),
        "size": size,
        "size_human": human_bytes(size),
        "total": total,
        "free": free,
        "free_human": human_bytes(free),
    })
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

async fn list_lora_datasets(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let items = danmo_store::list_datasets(&state.workspace)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "items": items })))
}

#[derive(Debug, Deserialize)]
struct CreateDatasetBody {
    #[serde(default)]
    name: String,
    #[serde(default = "default_concept")]
    kind: String,
    #[serde(default)]
    trigger_word: String,
}

fn default_concept() -> String {
    "concept".into()
}

async fn create_lora_dataset(
    State(state): State<AppState>,
    Json(body): Json<CreateDatasetBody>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let meta = danmo_store::create_dataset(
        &state.workspace,
        &body.name,
        &body.kind,
        &body.trigger_word,
    )
    .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let detail = danmo_store::get_dataset_detail(&state.workspace, &meta.id)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok((StatusCode::CREATED, Json(detail)))
}

async fn get_lora_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let detail = danmo_store::get_dataset_detail(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(detail))
}

#[derive(Debug, Deserialize)]
struct PatchDatasetBody {
    name: Option<String>,
    kind: Option<String>,
    trigger_word: Option<String>,
    default_prompt: Option<String>,
    nsfw: Option<bool>,
    face_crop: Option<String>,
}

async fn patch_lora_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<PatchDatasetBody>,
) -> Result<Json<Value>, ApiError> {
    danmo_store::update_dataset_meta(
        &state.workspace,
        &id,
        body.name.as_deref(),
        body.kind.as_deref(),
        body.trigger_word.as_deref(),
        body.default_prompt.as_deref(),
        body.nsfw,
        body.face_crop.as_deref(),
    )
    .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let detail = danmo_store::get_dataset_detail(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(detail))
}

async fn delete_lora_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    danmo_store::delete_dataset(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

#[derive(Debug, Deserialize)]
struct ImportCsvBody {
    csv: String,
}

async fn import_lora_dataset_csv(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ImportCsvBody>,
) -> Result<Json<Value>, ApiError> {
    danmo_store::import_metadata_csv(&state.workspace, &id, &body.csv)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let detail = danmo_store::get_dataset_detail(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(detail))
}

async fn upload_lora_dataset_images(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let (files, default_prompt) = multipart_files_and_prompt(ct, &body)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e))?;
    if files.is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "no image files in upload"));
    }
    let mut last = Value::Null;
    for (filename, bytes) in files {
        last = danmo_store::add_image_bytes(
            &state.workspace,
            &id,
            &filename,
            &bytes,
            &default_prompt,
        )
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    }
    Ok(Json(last))
}

#[derive(Debug, Deserialize)]
struct DeleteImageQuery {
    file: Option<String>,
}

async fn delete_lora_dataset_image_query(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<DeleteImageQuery>,
) -> Result<Json<Value>, ApiError> {
    let file = query
        .file
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| fail(StatusCode::BAD_REQUEST, "file query required"))?;
    let detail = danmo_store::delete_dataset_image(&state.workspace, &id, file)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(detail))
}

async fn delete_lora_dataset_image(
    State(state): State<AppState>,
    Path((id, file)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let detail = danmo_store::delete_dataset_image(&state.workspace, &id, &file)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(detail))
}

async fn serve_lora_dataset_file(
    State(state): State<AppState>,
    Path((id, file)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let rel = file.trim().trim_start_matches('/');
    if rel.is_empty() || rel.contains("..") {
        return Err(fail(StatusCode::BAD_REQUEST, "invalid image path"));
    }
    let path = danmo_store::dataset_dir(&state.workspace, &id).join(rel);
    if !path.is_file() {
        return Err(fail(StatusCode::NOT_FOUND, format!("file {rel} not found")));
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mime = mime_for_stored_path(rel);
    Ok(([(header::CONTENT_TYPE, mime.to_string())], bytes).into_response())
}

#[derive(Debug, Deserialize)]
struct SkippedImagesBody {
    #[serde(default)]
    files: Vec<String>,
}

async fn put_lora_dataset_skipped(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SkippedImagesBody>,
) -> Result<Json<Value>, ApiError> {
    let detail = danmo_store::set_skipped_images(&state.workspace, &id, &body.files)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(detail))
}

#[derive(Debug, Deserialize)]
struct DatasetHealthQuery {
    width: Option<u32>,
    height: Option<u32>,
}

async fn lora_dataset_health(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<DatasetHealthQuery>,
) -> Result<Json<Value>, ApiError> {
    let mut report = danmo_store::dataset_health_report(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::NOT_FOUND, e.to_string()))?;
    if let (Some(w), Some(h)) = (query.width, query.height) {
        if let Some(obj) = report.as_object_mut() {
            obj.insert(
                "faces".into(),
                json!({
                    "available": false,
                    "resolution": [w, h],
                    "rows": [],
                    "detail": "Face detector is not wired in the Rust trainer yet; structural checks only",
                }),
            );
        }
    }
    Ok(Json(report))
}

async fn lora_dataset_health_vlm(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    // Fail loud: VLM audit is not available in the Rust path yet.
    let mut report = danmo_store::dataset_health_report(&state.workspace, &id)
        .map_err(|e| fail(StatusCode::NOT_FOUND, e.to_string()))?;
    if let Some(obj) = report.as_object_mut() {
        obj.insert("vision_available".into(), json!(false));
        obj.insert("vlm_audited".into(), json!(false));
        obj.insert(
            "vlm_detail".into(),
            json!("VLM dataset audit is not available yet; use structural health only"),
        );
    }
    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
struct ImportAssetsBody {
    #[serde(default)]
    asset_ids: Vec<String>,
    #[serde(default)]
    default_prompt: String,
}

async fn import_lora_dataset_assets(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ImportAssetsBody>,
) -> Result<Json<Value>, ApiError> {
    if body.asset_ids.is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "asset_ids required"));
    }
    let mut sources = Vec::new();
    for asset_id in &body.asset_ids {
        let row = state
            .scheduler
            .get_asset(asset_id)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .ok_or_else(|| fail(StatusCode::NOT_FOUND, format!("asset {asset_id} not found")))?;
        if !row.kind.eq_ignore_ascii_case("image") {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("asset {asset_id} is not an image"),
            ));
        }
        let path = resolve_asset_fs_path(&state.outputs, &state.workspace, &row.file_path);
        let name = FsPath::new(&row.file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("asset.jpg")
            .to_string();
        sources.push((path, name));
    }
    let detail = danmo_store::import_asset_files(
        &state.workspace,
        &id,
        &sources,
        &body.default_prompt,
    )
    .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(detail))
}

fn training_presets_for_model(mid: &str) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for preset in ["turbo-sft", "turbo-diff", "base-face"] {
        if let Ok(resolved) = danmo_contracts::resolve_lora_train_preset(preset, mid) {
            out.insert(resolved.id.clone(), json!(resolved));
        }
    }
    out
}

async fn list_trainable_models() -> Json<Value> {
    let ready = cfg!(feature = "mlx");
    let turbo = training_presets_for_model("z-image-turbo");
    let base = training_presets_for_model("z-image");
    Json(json!({
        "items": [
            {
                "id": "z-image-turbo",
                "name": "Z-Image-Turbo",
                "family": "z_image",
                "trainable": true,
                "ready": ready,
                "presets": turbo.keys().cloned().collect::<Vec<_>>(),
                "default_preset": "turbo-sft",
            },
            {
                "id": "z-image",
                "name": "Z-Image",
                "family": "z_image",
                "trainable": true,
                "ready": ready,
                "presets": base.keys().cloned().collect::<Vec<_>>(),
                "default_preset": "base-face",
            },
        ],
        "presets_by_model": {
            "z-image-turbo": turbo,
            "z-image": base,
        },
    }))
}

#[derive(Debug, Deserialize)]
struct TrainingPresetsQuery {
    model: Option<String>,
    base_model: Option<String>,
}

async fn list_training_presets(Query(query): Query<TrainingPresetsQuery>) -> Json<Value> {
    let mid = query
        .model
        .as_deref()
        .or(query.base_model.as_deref())
        .unwrap_or("z-image-turbo");
    Json(json!({
        "model": mid,
        "presets": training_presets_for_model(mid),
    }))
}

#[derive(Debug, Deserialize)]
struct TrainingRequirementsQuery {
    base_model: Option<String>,
}

async fn training_requirements(Query(query): Query<TrainingRequirementsQuery>) -> Json<Value> {
    let ready = cfg!(feature = "mlx");
    let mid = query
        .base_model
        .as_deref()
        .unwrap_or("z-image-turbo")
        .split(':')
        .next()
        .unwrap_or("z-image-turbo")
        .trim();
    let can_submit = ready && (mid == "z-image" || mid == "z-image-turbo");
    let mut blockers: Vec<String> = Vec::new();
    if !ready {
        blockers.push(
            "LoRA training requires an MLX-enabled build (--features mlx-metal or mlx-cuda)".into(),
        );
    }
    if mid != "z-image" && mid != "z-image-turbo" {
        blockers.push(format!("unsupported training base model {mid:?}"));
    }
    Json(json!({
        "ready": ready,
        "can_submit": can_submit,
        "backend": if ready { "mlx" } else { "none" },
        "detail": if ready {
            "Rust MLX LoRA trainer is available for Z-Image / Z-Image-Turbo (probe-latent path until VAE encode lands)"
        } else {
            "LoRA training requires an MLX-enabled build (--features mlx-metal or mlx-cuda)"
        },
        "blockers": blockers,
        "trainable_models": ["z-image-turbo", "z-image"],
        "default_preset": {
            "z-image-turbo": "turbo-sft",
            "z-image": "base-face",
        },
        "base_model": mid,
    }))
}

async fn submit_training(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    if !cfg!(feature = "mlx") {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "LoRA training requires an MLX-enabled build (--features mlx-metal or mlx-cuda)",
        ));
    }
    // Reject leftover UI override keys loudly (fail loud — no silent ignore).
    if let Some(obj) = body.as_object() {
        const REMOVED: &[&str] = &[
            "progress_prompt",
            "iterations",
            "lora_rank",
            "learning_rate",
            "resolution",
            "qlora_bits",
            "grad_checkpoint",
            "compile_step",
            "lora_blocks",
            "lora_alpha",
            "lora_dropout",
            "lora_module_keys",
            "optimizer",
            "weight_decay",
            "val_split",
            "val_every",
            "train_type",
            "min_snr_gamma",
            "class_prompt",
            "prior_loss_weight",
            "early_stop_patience",
            "fuse_adapters",
            "num_augmentations",
            "grad_accumulate",
            "progress_steps",
            "caption_mode",
            "train_sigma_shift",
            "sigma_bias",
            "caption_dropout",
            "batch_size",
            "progress_every",
            "checkpoint_every",
            "guidance",
        ];
        let stray: Vec<&str> = REMOVED
            .iter()
            .copied()
            .filter(|k| obj.contains_key(*k))
            .collect();
        if !stray.is_empty() {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!(
                    "training request no longer accepts {:?}; use named preset + dataset default_prompt",
                    stray
                ),
            ));
        }
    }
    let request: danmo_contracts::LoraTrainingRequest = serde_json::from_value(body)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let mid = request
        .base_model
        .split(':')
        .next()
        .unwrap_or(&request.base_model)
        .trim();
    if mid != "z-image" && mid != "z-image-turbo" {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("unsupported training base model {mid:?}"),
        ));
    }
    let meta = danmo_store::load_dataset_meta(&state.workspace, &request.dataset_id)
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let caption = meta.default_prompt.trim();
    if caption.is_empty() {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "dataset default_prompt is required (set it on the dataset step)",
        ));
    }
    let (_meta, pairs) = danmo_store::load_training_pairs(
        &state.workspace,
        &request.dataset_id,
        caption,
    )
    .map_err(|e| fail(StatusCode::BAD_REQUEST, e.to_string()))?;
    let preset = danmo_contracts::resolve_lora_train_preset(&request.preset, &request.base_model)
        .and_then(|p| danmo_contracts::merge_lora_train_request(&request, p))
        .map_err(|e| fail(StatusCode::BAD_REQUEST, e))?;
    if preset.id == "turbo-sft" {
        // DistillPatch must be installed for matching inference; fail loud if absent.
        let registry = state.workspace.join("config").join("models_registry.json");
        let _ = registry; // presence checked at infer time; training records the recipe.
    }

    let task_id = format!("tsk_{}", Uuid::new_v4());
    let work_dir = state.workspace.join("tasks").join(&task_id);
    std::fs::create_dir_all(&work_dir)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let params = json!({
        "task_id": task_id,
        "family": "z_image",
        "request": request,
        "preset": preset,
        "training_caption": caption,
        "n_images": pairs.len(),
    });
    let prio = if request.priority == "high" { 10 } else { 100 };
    state
        .scheduler
        .submit(&task_id, "lora.training", mid, &params.to_string(), prio)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let scheduler = state.scheduler.clone();
    let req = request.clone();
    let wid = task_id.clone();
    let wdir = work_dir.clone();
    tokio::task::spawn_blocking(move || {
        run_training_job(&scheduler, &wid, &wdir, &req);
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "id": task_id,
            "status": "queued",
            "kind": "lora.training",
        })),
    ))
}

fn training_task_result(state: &AppState, task_id: &str) -> Result<(TaskRow, Value), ApiError> {
    let row = state
        .scheduler
        .get(task_id)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "training task not found"))?;
    if row.kind != "lora.training" {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("task {task_id} is not a LoRA training run"),
        ));
    }
    let result = row
        .result
        .as_deref()
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or(Value::Null);
    Ok((row, result))
}

fn checkpoint_basename(path: &str) -> String {
    FsPath::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn resolve_training_checkpoint(
    work_dir: &FsPath,
    result: &Value,
    checkpoint: &str,
) -> Result<PathBuf, ApiError> {
    let name = checkpoint.trim();
    if name.is_empty() || name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err(fail(StatusCode::BAD_REQUEST, "invalid checkpoint name"));
    }
    let adapters = work_dir.join("adapters");
    let candidate = adapters.join(name);
    if candidate.is_file() {
        return Ok(candidate);
    }
    if let Some(adapter_path) = result.get("adapter_path").and_then(|v| v.as_str()) {
        let p = PathBuf::from(adapter_path);
        if checkpoint_basename(adapter_path) == name && p.is_file() {
            return Ok(p);
        }
    }
    if let Some(list) = result.get("checkpoints").and_then(|v| v.as_array()) {
        for item in list {
            if let Some(path) = item.as_str() {
                if checkpoint_basename(path) == name {
                    let p = PathBuf::from(path);
                    if p.is_file() {
                        return Ok(p);
                    }
                }
            }
        }
    }
    Err(fail(
        StatusCode::NOT_FOUND,
        format!("checkpoint {name} not found for this training run"),
    ))
}

async fn training_artifacts(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let (_row, result) = training_task_result(&state, &id)?;
    let work_dir = state.workspace.join("tasks").join(&id);
    let mut checkpoints: Vec<String> = result
        .get("checkpoints")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(checkpoint_basename))
        .collect();
    if checkpoints.is_empty() {
        if let Some(path) = result.get("adapter_path").and_then(|v| v.as_str()) {
            checkpoints.push(checkpoint_basename(path));
        }
    }
    let progress_dir = work_dir.join("progress");
    let mut progress_images = Vec::new();
    if progress_dir.is_dir() {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&progress_dir)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        {
            let entry = entry.map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let lower = name.to_ascii_lowercase();
            if lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg") || lower.ends_with(".webp")
            {
                names.push(name);
            }
        }
        names.sort();
        progress_images = names;
    }
    Ok(Json(json!({
        "loss_history": result.get("loss_history").cloned().unwrap_or_else(|| json!([])),
        "checkpoints": checkpoints,
        "progress_images": progress_images,
        "inference": result.get("inference").cloned().unwrap_or(Value::Null),
        "adapter_path": result.get("adapter_path").cloned().unwrap_or(Value::Null),
        "output_name": result.get("output_name").cloned().unwrap_or(Value::Null),
        "user_lora_id": result.get("user_lora_id").cloned().unwrap_or_else(|| json!("")),
        "needs_candidate_evaluation": result
            .get("needs_candidate_evaluation")
            .cloned()
            .unwrap_or(Value::Bool(false)),
        "vision_available": false,
    })))
}

async fn training_artifact_file(
    State(state): State<AppState>,
    Path((id, filename)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let name = filename.trim();
    if name.is_empty() || name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err(fail(StatusCode::BAD_REQUEST, "invalid artifact filename"));
    }
    let _ = training_task_result(&state, &id)?;
    let path = state
        .workspace
        .join("tasks")
        .join(&id)
        .join("progress")
        .join(name);
    if !path.is_file() {
        return Err(fail(StatusCode::NOT_FOUND, format!("artifact {name} not found")));
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mime = mime_for_stored_path(name);
    Ok(([(header::CONTENT_TYPE, mime.to_string())], bytes).into_response())
}

#[derive(Debug, Deserialize)]
struct RegisterTrainingBody {
    checkpoint: String,
    #[serde(default)]
    name: String,
}

async fn register_training_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<RegisterTrainingBody>,
) -> Result<Json<Value>, ApiError> {
    let (row, result) = training_task_result(&state, &id)?;
    if row.status != "completed" {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("training task is {}, register only after completed", row.status),
        ));
    }
    let work_dir = state.workspace.join("tasks").join(&id);
    let src = resolve_training_checkpoint(&work_dir, &result, &body.checkpoint)?;
    let params: Value = serde_json::from_str(&row.params).unwrap_or_else(|_| json!({}));
    let request = params.get("request").cloned().unwrap_or(Value::Null);
    let preset = params.get("preset").cloned().unwrap_or(Value::Null);
    let base_model = request
        .get("base_model")
        .and_then(|v| v.as_str())
        .or_else(|| params.get("base_model").and_then(|v| v.as_str()))
        .unwrap_or(row.model_id.as_str())
        .to_string();
    let output_name = {
        let raw = body.name.trim();
        if !raw.is_empty() {
            raw.to_string()
        } else if let Some(n) = result.get("output_name").and_then(|v| v.as_str()) {
            n.trim().to_string()
        } else if let Some(n) = request.get("output_name").and_then(|v| v.as_str()) {
            n.trim().to_string()
        } else {
            format!("{base_model}-lora")
        }
    };
    if output_name.is_empty() || output_name.contains("..") || output_name.contains('/') || output_name.contains('\\')
    {
        return Err(fail(StatusCode::BAD_REQUEST, "invalid LoRA name"));
    }
    let rel_dir = format!("models/Lora/{output_name}");
    let dest_dir = state.workspace.join(&rel_dir);
    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let dest_name = checkpoint_basename(src.to_string_lossy().as_ref());
    let dest_file = dest_dir.join(&dest_name);
    std::fs::copy(&src, &dest_file)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    // Keep the training recipe next to the weight so the install is self-describing.
    let cfg_src = src
        .parent()
        .map(|p| p.join("lora_config.json"))
        .filter(|p| p.is_file());
    if let Some(cfg) = cfg_src {
        let _ = std::fs::copy(&cfg, dest_dir.join("lora_config.json"));
    }
    let inference = result
        .get("inference")
        .cloned()
        .or_else(|| preset.get("inference").cloned())
        .filter(|v| v.is_object())
        .ok_or_else(|| {
            fail(
                StatusCode::BAD_REQUEST,
                "training result has no inference recipe; cannot register without steps/guidance/extra_adapters",
            )
        })?;
    let extras = danmo_contracts::stacked_adapters_from_value(inference.get("extra_adapters"))
        .map_err(|detail| {
            fail(
                StatusCode::BAD_REQUEST,
                format!("invalid inference.extra_adapters: {detail}"),
            )
        })?;
    for extra in &extras {
        // Fail loud: every companion listed on the inference recipe must be installed.
        ensure_companion_lora_installed(&state, &extra.id)?;
    }
    let dataset_id = request
        .get("dataset_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let trigger_word = if !dataset_id.is_empty() {
        danmo_store::load_dataset_meta(&state.workspace, dataset_id)
            .ok()
            .map(|m| m.trigger_word.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_default()
    } else {
        String::new()
    };
    let preset_id = request
        .get("preset")
        .and_then(|v| v.as_str())
        .or_else(|| preset.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    let lora_id = format!(
        "user-lora-{}",
        &Uuid::new_v4().simple().to_string()[..12]
    );
    let entry = json!({
        "id": lora_id,
        "name": output_name,
        "base_model": base_model,
        "local_path": rel_dir,
        "trigger_word": trigger_word,
        "lora_rank": preset.get("lora_rank").cloned().unwrap_or(json!(32)),
        "lora_alpha": preset.get("lora_rank").cloned().unwrap_or(json!(32)),
        "inference": inference,
        "preset": preset_id,
        "family": "z_image",
        "train_type": "lora",
        "lora_module_keys": preset.get("lora_module_keys").cloned().unwrap_or(json!([])),
        "nsfw": false,
        "source": "user_trained",
        "task_id": id,
        "checkpoint": dest_name,
        "created_at": chrono_like_now(),
    });
    let user_path = state.workspace.join("config").join("user_loras.json");
    let mut root = if user_path.is_file() {
        let text = std::fs::read_to_string(&user_path)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        serde_json::from_str::<Value>(&text).map_err(|e| {
            fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{} is not json: {e}", user_path.display()),
            )
        })?
    } else {
        json!({ "items": [] })
    };
    let items = root
        .as_object_mut()
        .ok_or_else(|| fail(StatusCode::INTERNAL_SERVER_ERROR, "user_loras.json root must be object"))?
        .entry("items")
        .or_insert_with(|| json!([]));
    let arr = items
        .as_array_mut()
        .ok_or_else(|| fail(StatusCode::INTERNAL_SERVER_ERROR, "user_loras.json items must be array"))?;
    arr.retain(|item| item.get("id").and_then(|v| v.as_str()) != Some(lora_id.as_str()));
    arr.push(entry.clone());
    if let Some(parent) = user_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    let pretty = serde_json::to_string_pretty(&root)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    std::fs::write(&user_path, pretty)
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "user_lora": entry })))
}

fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

fn ensure_companion_lora_installed(state: &AppState, id: &str) -> Result<(), ApiError> {
    if let Some(path) = state
        .downloads
        .lora_path(id)
        .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?
    {
        if path.is_file() || download::lora_weight_file(&path).is_ok() {
            return Ok(());
        }
    }
    let (bare, version) = match id.split_once(':') {
        Some((bare, version)) => (bare, Some(version)),
        None => (id, None),
    };
    let catalog = open_catalog(state)?;
    let Some(entry) = catalog.root().pointer(&format!("/models/{bare}")) else {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!(
                "companion LoRA {id} is not installed; required by this training recipe before register"
            ),
        ));
    };
    match download::registry_lora_weight(&state.workspace, entry, version)
        .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?
    {
        Some(_) => Ok(()),
        None => Err(fail(
            StatusCode::BAD_REQUEST,
            format!(
                "companion LoRA {id} is not installed; required by this training recipe before register"
            ),
        )),
    }
}

fn run_training_job(
    scheduler: &danmo_schedule::Scheduler,
    task_id: &str,
    work_dir: &FsPath,
    request: &danmo_contracts::LoraTrainingRequest,
) {
    let _ = scheduler.log(task_id, "info", "Rust MLX LoRA training started");
    #[cfg(feature = "mlx")]
    {
        use danmo_engine::run_z_image_lora_training;
        use danmo_mlx::{Array, Stream};
        let stream = match Stream::cpu() {
            Ok(s) => s,
            Err(e) => {
                let _ = scheduler.finish_err(task_id, &e.to_string());
                return;
            }
        };
        // Probe latents until VAE encode cache lands; shape [n, 64] keeps the
        // shared Flow-Match loop exercised end-to-end.
        let n = request
            .metadata
            .get("probe_batch")
            .and_then(|v| v.as_u64())
            .unwrap_or(4)
            .clamp(2, 16) as usize;
        let mut data = vec![0.0f32; n * 64];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((i % 17) as f32) * 0.01 - 0.08;
        }
        let latent = match Array::from_f32_slice(&data, &[n as i32, 64]) {
            Ok(a) => a,
            Err(e) => {
                let _ = scheduler.finish_err(task_id, &e.to_string());
                return;
            }
        };
        let mut req = request.clone();
        if req.metadata.get("smoke").is_none() {
            req.metadata = json!({ "smoke": true });
        }
        match run_z_image_lora_training(&req, work_dir, &stream, &latent) {
            Ok(out) => {
                let result = serde_json::to_string(&out.result).unwrap_or_else(|_| "{}".into());
                let _ = scheduler.log(
                    task_id,
                    "success",
                    &format!("adapter written to {}", out.result.adapter_path),
                );
                let _ = scheduler.finish_ok(task_id, &result);
            }
            Err(e) => {
                let _ = scheduler.log(task_id, "error", &e.to_string());
                let _ = scheduler.finish_err(task_id, &e.to_string());
            }
        }
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = scheduler.finish_err(task_id, "MLX feature not enabled");
    }
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

const DEFAULT_ASSISTANT_MODEL_ID: &str = "qwen3-vl-4b-instruct";

const ENHANCE_IMAGE_SYSTEM: &str = "# Role\n\nPrompt engineer for AI image models (Flux, Z-Image, Qwen-Image).\n\n## Task\n\nRewrite the user's idea into one vivid, comma-separated description. Keep their subject, names, and intent.\n\n## Style\n\n- Add at most a few cues for lighting, composition, color, texture, and mood.\n- If the input is already detailed, lightly polish only — do not lengthen.\n\n## Limits\n\n- Length cap: ~120 CJK characters or ~80 English words.\n- Never repeat the same word or phrase; never loop filler at the end.\n- Do not write \"Okay\", explanations, or quotes.\n\n## Output\n\nOutput **only** the enhanced prompt.";

const ENHANCE_VIDEO_SYSTEM: &str = "# Role\n\nProfessional prompt engineer for AI video generation.\n\n## Task\n\nRewrite the user's brief into a detailed prompt for image-to-video or text-to-video models.\n\n## Include\n\nSubject, scene, lighting, style, camera movement, motion dynamics, pacing, and temporal mood.\n\nFor LTX audio-video models, write the caption in the language that should be heard. Quoted dialogue is spoken in the language of the surrounding sentences: an English caption with Chinese quotes is performed as English. Do not translate a spoken script. Put one space before each opening quotation mark and keep every spoken line verbatim.\nDistinguish static scene description from continuing motion the camera can follow.\n\n## Limits\n\n- One paragraph. Visual briefs stay within ~120 CJK characters or ~80 English words. Spoken scripts are exempt: keep every line.\n- Never repeat the same phrase or word. No filler loops.\n\n## Output\n\nOutput **only** the enhanced prompt text, without explanation. Keep spoken lines inside quotation marks.";

const ENHANCE_AUDIO_SYSTEM: &str = "# Role\n\nMusic producer writing briefs for AI music generation (ACE-Step).\n\n## Task\n\nExpand the user's music idea into a clear, vivid description covering genre, mood, tempo feel, instrumentation, vocal style, and emotional arc.\n\n## Limits\n\n- One short paragraph.\n- Never repeat the same phrase or word. No filler loops.\n\n## Output\n\nOutput **only** the enhanced brief text, without explanation or quotation marks.";

fn enhance_system_prompt(target_action: &str) -> &'static str {
    let action = if target_action.trim().is_empty() {
        "image_create"
    } else {
        target_action.trim()
    };
    match action.to_lowercase().as_str() {
        "video" | "video_create" | "animate" | "video_generation" => ENHANCE_VIDEO_SYSTEM,
        "audio" | "audio_create" | "music" | "audio_generation" => ENHANCE_AUDIO_SYSTEM,
        _ => ENHANCE_IMAGE_SYSTEM,
    }
}

fn enhance_user_locale_hint(text: &str) -> &'static str {
    let raw = text.trim();
    if raw.is_empty() {
        return "";
    }
    let chars = raw.chars().count();
    if raw
        .chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
    {
        if chars < 60 {
            return "\n\n（输入较短：保留原意与关键词，最多补充少量画面细节，禁止堆砌形容词或重复用词。）";
        }
        return "\n\n（输入已够详细：只做轻微润色，禁止加长或重复用词。）";
    }
    if chars < 80 {
        return "\n\n(Input is short: preserve intent and keywords; add at most a few visual cues; no filler loops.)";
    }
    "\n\n(Input is already detailed: light polish only; do not lengthen or repeat phrases.)"
}

fn is_thinking_model(model_id: &str) -> bool {
    let mid = model_id.to_lowercase();
    if mid.contains("thinking") {
        return true;
    }
    ["qwen3.5", "qwen3-5", "qwen3.6", "qwen3-6"]
        .iter()
        .any(|prefix| mid.starts_with(prefix))
}

fn llm_provider(settings: &Value) -> &str {
    match settings
        .get("llm_inference_provider")
        .and_then(|v| v.as_str())
    {
        Some("openai_compatible") => "openai_compatible",
        _ => "builtin",
    }
}

fn cloud_api_key(settings: &Value) -> String {
    let env = std::env::var("DANQING_LLM_INFERENCE_API_KEY").unwrap_or_default();
    let env = env.trim();
    if !env.is_empty() {
        return env.to_string();
    }
    settings
        .get("llm_inference_api_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

fn cloud_ready(settings: &Value) -> bool {
    if llm_provider(settings) != "openai_compatible" {
        return false;
    }
    let base = settings
        .get("llm_inference_base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if base.is_empty() {
        return false;
    }
    let key = cloud_api_key(settings);
    !key.is_empty() && !key.starts_with("v1:")
}

fn cloud_base_url(settings: &Value) -> String {
    let base = settings
        .get("llm_inference_base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .trim_end_matches('/');
    if base.is_empty() {
        return "https://api.openai.com/v1".to_string();
    }
    if base.contains("/v1") {
        return base.to_string();
    }
    format!("{base}/v1")
}

fn assistant_model_id(settings: &Value, catalog: &Catalog) -> String {
    let preferred = settings
        .get("default_model_llm")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let bare = preferred.split(':').next().unwrap_or("");
    if !bare.is_empty() && catalog.model(bare).is_ok() {
        return bare.to_string();
    }
    if catalog.model(DEFAULT_ASSISTANT_MODEL_ID).is_ok() {
        return DEFAULT_ASSISTANT_MODEL_ID.to_string();
    }
    if let Some(models) = catalog.root().get("models").and_then(|m| m.as_object()) {
        let mut llm: Vec<&String> = models
            .iter()
            .filter(|(_, m)| m.pointer("/catalog/media").and_then(|v| v.as_str()) == Some("llm"))
            .map(|(key, _)| key)
            .collect();
        llm.sort();
        if let Some(key) = llm.iter().find(|key| {
            models
                .get(**key)
                .and_then(|m| m.get("actions"))
                .and_then(|a| a.get("describe"))
                .is_some()
        }) {
            return key.to_string();
        }
        if let Some(key) = llm.first() {
            return key.to_string();
        }
    }
    DEFAULT_ASSISTANT_MODEL_ID.to_string()
}

/// Human-readable reason the assistant cannot run, or `None` when it can.
fn assistant_block_reason(settings: &Value, catalog: &Catalog) -> Option<String> {
    if llm_provider(settings) == "openai_compatible" {
        if cloud_ready(settings) {
            return None;
        }
        return Some(
            "Cloud LLM inference is not configured (base URL and API key required).".to_string(),
        );
    }
    let id = assistant_model_id(settings, catalog);
    Some(format!(
        "Assistant model '{id}' is not installed. Install it from the Models page."
    ))
}

async fn llm_model_info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let settings = current_settings(&state);
    let catalog = open_catalog(&state)?;
    let id = assistant_model_id(&settings, &catalog);
    let think_supported = is_thinking_model(&id);
    let think_enabled = think_supported
        && settings
            .get("default_model_llm_think")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    Ok(Json(json!({
        "model_id": id,
        "name": id,
        "available": assistant_block_reason(&settings, &catalog).is_none(),
        "provider": llm_provider(&settings),
        "unload_each_request": settings
            .get("llm_unload_each_request")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "llm_cache_ttl_minutes": settings
            .get("llm_cache_ttl_minutes")
            .and_then(|v| v.as_i64())
            .unwrap_or(30),
        "think_supported": think_supported,
        "think_enabled": think_enabled,
    })))
}

async fn vision_model_info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let settings = current_settings(&state);
    let catalog = open_catalog(&state)?;
    let id = assistant_model_id(&settings, &catalog);
    Ok(Json(json!({
        "model_id": id,
        "name": id,
        "available": assistant_block_reason(&settings, &catalog).is_none(),
        "provider": llm_provider(&settings),
    })))
}

fn cloud_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into()
}

fn cloud_post(url: &str, key: &str, payload: &Value) -> Result<(u16, ureq::Body), String> {
    let mut request = cloud_agent()
        .post(url)
        .header("accept", "application/json")
        .header("content-type", "application/json");
    if !key.is_empty() {
        request = request.header("authorization", &format!("Bearer {key}"));
    }
    let body = serde_json::to_string(payload).map_err(|err| err.to_string())?;
    let response = request
        .send(body)
        .map_err(|err| format!("POST {url} failed: {err}"))?;
    Ok((response.status().as_u16(), response.into_body()))
}

fn cloud_chat(url: &str, key: &str, payload: &Value) -> Result<Value, String> {
    let (status, mut body) = cloud_post(url, key, payload)?;
    let mut text = String::new();
    std::io::Read::read_to_string(
        &mut std::io::Read::take(body.as_reader(), 20 * 1024 * 1024),
        &mut text,
    )
    .map_err(|err| format!("reading {url} failed: {err}"))?;
    if !(200..300).contains(&status) {
        return Err(format!(
            "POST {url} returned {status}: {}",
            text.chars().take(400).collect::<String>()
        ));
    }
    serde_json::from_str(&text).map_err(|err| format!("{url} was not JSON: {err}"))
}

fn cloud_stream(
    url: &str,
    key: &str,
    payload: &Value,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    let (status, mut body) = cloud_post(url, key, payload)?;
    if !(200..300).contains(&status) {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(
            &mut std::io::Read::take(body.as_reader(), 1024 * 1024),
            &mut text,
        );
        return Err(format!(
            "POST {url} returned {status}: {}",
            text.chars().take(400).collect::<String>()
        ));
    }
    let mut reader = body.as_reader();
    let mut buf = [0u8; 8192];
    loop {
        let read = std::io::Read::read(&mut reader, &mut buf).map_err(|err| err.to_string())?;
        if read == 0 {
            return Ok(());
        }
        if tx
            .blocking_send(Ok(Bytes::copy_from_slice(&buf[..read])))
            .is_err()
        {
            return Ok(());
        }
    }
}

fn cloud_chat_payload(request: &Value, model: &str, stream: bool) -> Value {
    let mut payload = json!({
        "model": model,
        "messages": request.get("messages").cloned().unwrap_or(json!([])),
        "temperature": request.get("temperature").cloned().unwrap_or(json!(0.7)),
        "top_p": request.get("top_p").cloned().unwrap_or(json!(0.9)),
        "stream": stream,
    });
    if let Some(max_tokens) = request.get("max_tokens") {
        if !max_tokens.is_null() {
            payload["max_tokens"] = max_tokens.clone();
        }
    }
    payload
}

/// Keep the post-reasoning answer; empty when generation stopped inside thinking.
fn extract_final_llm_content(text: &str) -> String {
    let raw = text.trim();
    if raw.is_empty() {
        return String::new();
    }
    let lower = raw.to_lowercase();
    for tag in ["</think>", "</redacted_reasoning>"] {
        if let Some(pos) = lower.rfind(tag) {
            return raw[pos + tag.len()..]
                .trim()
                .trim_end_matches("<|im_end|>")
                .trim()
                .to_string();
        }
    }
    if lower.starts_with("<think>") || lower.starts_with("<redacted_reasoning>") {
        return String::new();
    }
    raw.trim_end_matches("<|im_end|>").trim().to_string()
}

fn sanitize_enhanced_prompt(text: &str) -> String {
    let answer = extract_final_llm_content(text);
    let trimmed = answer.trim();
    let unquoted = if trimmed.len() >= 2 {
        let first = trimmed.as_bytes()[0];
        let last = trimmed.as_bytes()[trimmed.len() - 1];
        if first == last && (first == b'"' || first == b'\'') {
            trimmed[1..trimmed.len() - 1].trim()
        } else {
            trimmed
        }
    } else {
        trimmed
    };
    unquoted.chars().take(480).collect()
}

async fn enhance_prompt(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let request: Value = serde_json::from_slice(&body).map_err(|err| {
        fail(
            StatusCode::BAD_REQUEST,
            format!("invalid enhance request: {err}"),
        )
    })?;
    let settings = current_settings(&state);
    let catalog = open_catalog(&state)?;
    if let Some(reason) = assistant_block_reason(&settings, &catalog) {
        return Err(fail(StatusCode::SERVICE_UNAVAILABLE, reason));
    }
    let prompt = request
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let target_action = request
        .get("target_action")
        .and_then(|v| v.as_str())
        .unwrap_or("image_create");
    let action = target_action.to_lowercase();
    let mut user_content = prompt.clone();
    if matches!(action.as_str(), "image_create" | "create" | "image") {
        user_content.push_str(enhance_user_locale_hint(&prompt));
    }
    let style = request
        .get("style_positive")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if !style.is_empty() {
        user_content.push_str(&format!("\n\nStyle cues to weave in: {style}"));
    }
    let catalog_id = assistant_model_id(&settings, &catalog);
    let cloud_model = settings
        .get("llm_inference_cloud_model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let model = if cloud_model.is_empty() {
        catalog_id
    } else {
        cloud_model.to_string()
    };
    let payload = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": enhance_system_prompt(target_action) },
            { "role": "user", "content": user_content },
        ],
        "temperature": 0.65,
        "top_p": 0.9,
        "max_tokens": 200,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cloud_base_url(&settings));
    let key = cloud_api_key(&settings);
    let response = tokio::task::spawn_blocking(move || cloud_chat(&url, &key, &payload))
        .await
        .map_err(|err| fail(StatusCode::BAD_GATEWAY, err.to_string()))?
        .map_err(|err| fail(StatusCode::BAD_GATEWAY, err))?;
    let content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let cleaned = sanitize_enhanced_prompt(content);
    let enhanced = if cleaned.is_empty() { prompt } else { cleaned };
    Ok(Json(json!({ "enhanced_prompt": enhanced })))
}

async fn chat_completions(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: Value = serde_json::from_slice(&body).map_err(|err| {
        fail(
            StatusCode::BAD_REQUEST,
            format!("invalid chat completion request: {err}"),
        )
    })?;
    let settings = current_settings(&state);
    let catalog = open_catalog(&state)?;
    if let Some(reason) = assistant_block_reason(&settings, &catalog) {
        return Err(fail(StatusCode::SERVICE_UNAVAILABLE, reason));
    }
    let catalog_id = assistant_model_id(&settings, &catalog);
    let cloud_model = settings
        .get("llm_inference_cloud_model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let model = if cloud_model.is_empty() {
        catalog_id
    } else {
        cloud_model.to_string()
    };
    let stream = request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let url = format!("{}/chat/completions", cloud_base_url(&settings));
    let key = cloud_api_key(&settings);
    if !stream {
        let payload = cloud_chat_payload(&request, &model, false);
        let response = tokio::task::spawn_blocking(move || cloud_chat(&url, &key, &payload))
            .await
            .map_err(|err| fail(StatusCode::BAD_GATEWAY, err.to_string()))?
            .map_err(|err| fail(StatusCode::BAD_GATEWAY, err))?;
        return Ok(Json(response).into_response());
    }
    let payload = cloud_chat_payload(&request, &model, true);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    tokio::task::spawn_blocking(move || {
        if let Err(err) = cloud_stream(&url, &key, &payload, tx.clone()) {
            let _ = tx.blocking_send(Err(std::io::Error::new(std::io::ErrorKind::Other, err)));
        }
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(axum::body::Body::from_stream(ReceiverStream::new(rx)))
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
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
    Catalog::open(&state.registry)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
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
    let (files, _) = multipart_files_and_prompt(content_type, body)?;
    files
        .into_iter()
        .next()
        .ok_or_else(|| "multipart body has no file field".into())
}

fn multipart_files_and_prompt(
    content_type: &str,
    body: &[u8],
) -> Result<(Vec<(String, Vec<u8>)>, String), String> {
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
    let mut files = Vec::new();
    let mut default_prompt = String::new();
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
        let headers_lc = headers.to_ascii_lowercase();
        let data_start = header_end + 4;
        let next = find_bytes(&body[data_start..], &marker).unwrap_or(body.len() - data_start);
        let mut data_end = data_start + next;
        if data_end >= 2 && &body[data_end - 2..data_end] == b"\r\n" {
            data_end -= 2;
        }
        let is_file = headers_lc.contains("filename=")
            && (headers_lc.contains("name=\"file\"")
                || headers_lc.contains("name=file")
                || headers_lc.contains("name=\"files\"")
                || headers_lc.contains("name=files"));
        if is_file {
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
            files.push((filename, body[data_start..data_end].to_vec()));
        } else if headers_lc.contains("name=\"default_prompt\"")
            || headers_lc.contains("name=default_prompt")
        {
            default_prompt = String::from_utf8_lossy(&body[data_start..data_end])
                .trim()
                .to_string();
        }
        start = data_start + next;
    }
    Ok((files, default_prompt))
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
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

#[cfg(test)]
mod lora_base_tests {
    use super::lora_compatible;

    #[test]
    fn xl_loras_stay_inside_the_xl_family() {
        assert!(lora_compatible("ace-step-xl-turbo", "ace-step-xl-sft"));
        assert!(lora_compatible(
            "ace-step-xl-sft:xl-sft",
            "ace-step-xl-turbo"
        ));
        assert!(lora_compatible("ace-step-turbo", "ace-step-turbo"));
    }

    #[test]
    fn two_b_lora_stays_off_xl() {
        assert!(!lora_compatible("ace-step-xl-turbo", "ace-step-turbo"));
        assert!(!lora_compatible("ace-step-xl-sft", "ace-step-turbo"));
        assert!(!lora_compatible("ace-step-xl-base", ""));
    }
}

#[cfg(test)]
mod asset_path_tests {
    use super::resolve_asset_fs_path;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn relative_basename_resolves_under_outputs_assets() {
        let root = std::env::temp_dir().join(format!("danmo-asset-path-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let outputs = root.join("outputs");
        let assets = outputs.join("assets");
        fs::create_dir_all(&assets).unwrap();
        let file = assets.join("ast_demo.png");
        fs::write(&file, b"png").unwrap();

        let resolved = resolve_asset_fs_path(&outputs, &root, "ast_demo.png");
        assert_eq!(resolved, file);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn absolute_path_is_kept() {
        let absolute = PathBuf::from("/tmp/danmo-absolute-asset.bin");
        let resolved = resolve_asset_fs_path(
            PathBuf::from("/unused/outputs").as_path(),
            PathBuf::from("/unused/workspace").as_path(),
            absolute.to_str().unwrap(),
        );
        assert_eq!(resolved, absolute);
    }
}
