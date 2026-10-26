use crate::product;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::Catalog;
use danmo_contracts::{
    AudioGenerationRequest, ImageEditRequest, ImageGenerationRequest, ImageUpscaleRequest,
    TaskAccepted, TaskKind, VideoGenerationRequest,
};
use danmo_engine::{audio_create, image_create, upscale, video_create, BackendKind};
use danmo_schedule::Scheduler;
use danmo_store::AssetRow;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::{Path as FsPath, PathBuf};
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) scheduler: Scheduler,
    pub(crate) backend: BackendKind,
    pub(crate) registry: PathBuf,
    pub(crate) workspace: PathBuf,
    pub(crate) outputs: PathBuf,
}

pub fn router(db: &std::path::Path, registry: &std::path::Path) -> Router {
    let backend = BackendKind::from_env().unwrap_or(BackendKind::Cpu);
    let workspace = workspace_of(db);
    let outputs = std::env::var("DANQING_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("outputs"));
    let scheduler = Scheduler::open(db, true).expect("open task db");
    let state = AppState {
        scheduler,
        backend,
        registry: registry.to_path_buf(),
        workspace,
        outputs,
    };
    spawn_worker(state.clone());
    Router::new()
        .route("/api/system/health", get(health))
        .route("/api/registry", get(registry_route))
        .route("/api/queue", get(queue))
        .route("/api/tasks/:id/stream", get(stream_task))
        .route("/api/images/generations", post(image_generations))
        .route("/api/images/edits", post(image_edits))
        .route("/api/images/upscales", post(image_upscales))
        .route("/api/videos/generations", post(video_generations))
        .route("/api/videos/edits", post(video_edits))
        .route("/api/videos/upscales", post(video_upscales))
        .route("/api/videos/avatars", post(video_avatars))
        .route("/api/videos/avatars/script", post(video_avatar_script))
        .route("/api/audios/generations", post(audio_generations))
        .route("/api/audios/edits", post(audio_edits))
        .merge(product::routes())
        .fallback(product::spa)
        .with_state(state)
}

fn workspace_of(db: &FsPath) -> PathBuf {
    if let Ok(path) = std::env::var("DANQING_WORKSPACE") {
        return PathBuf::from(path);
    }
    match db.parent() {
        Some(parent) if parent.file_name().and_then(|name| name.to_str()) == Some("db") => parent
            .parent()
            .map(|path| path.to_path_buf())
            .unwrap_or_else(|| parent.to_path_buf()),
        Some(parent) => parent.to_path_buf(),
        None => PathBuf::from("."),
    }
}

fn spawn_worker(state: AppState) {
    std::thread::spawn(move || loop {
        let next = state.scheduler.dequeue().ok().flatten();
        let Some(task) = next else {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        let params: Value = serde_json::from_str(&task.params).unwrap_or(json!({}));
        let family = params.get("family").and_then(|v| v.as_str()).unwrap_or("");
        let outcome = run_task(&state, &task.kind, family, &task.model_id, &params);
        match outcome {
            Ok(result) => {
                let _ = state.scheduler.finish_ok(&task.id, &result.to_string());
            }
            Err(err) => {
                let _ = state.scheduler.finish_err(&task.id, &err);
            }
        }
    });
}

fn run_task(
    state: &AppState,
    kind: &str,
    family: &str,
    model: &str,
    params: &Value,
) -> Result<Value, String> {
    let needs_bundle = matches!(
        kind,
        "image.generation"
            | "image.edit"
            | "image.upscale"
            | "video.generation"
            | "video.edit"
            | "video.upscale"
            | "video.avatar"
            | "audio.generation"
            | "audio.edit"
    );
    let bundle = if needs_bundle {
        Some(gguf_bundle(&state.registry, model)?)
    } else {
        None
    };
    match kind {
        "image.generation" => {
            let request: ImageGenerationRequest =
                serde_json::from_value(params.get("request").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            let task_id = params
                .get("task_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let out = image_create(&request, family, &bundle, state.backend, |step, steps| {
                let _ = state.scheduler.progress(
                    &task_id,
                    step as f64 / steps as f64,
                    &format!("step {step}/{steps}"),
                );
            })
            .map_err(|e| e.to_string())?;
            let asset_ids = persist_preview(state, &task_id, model, params, &out)?;
            Ok(json!({
                "family": out.family,
                "steps": out.steps,
                "graph_computes": out.graph_computes,
                "width": out.width,
                "height": out.height,
                "latent_len": out.latents.len(),
                "asset_ids": asset_ids,
            }))
        }
        "image.edit" | "image.upscale" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            upscale_or_edit(kind, family, &bundle)
        }
        "video.generation" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            video_create(family, &bundle)
                .map(|_| json!({"ok": true}))
                .map_err(|e| e.to_string())
        }
        "audio.generation" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            audio_create(family, &bundle)
                .map(|_| json!({"ok": true}))
                .map_err(|e| e.to_string())
        }
        "video.edit" | "video.upscale" | "video.avatar" | "audio.edit" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            danmo_engine::assert_gguf_bundle(&bundle).map_err(|e| e.to_string())?;
            Err(format!("{family} graph is not built"))
        }
        other => Err(format!("unsupported task kind {other}")),
    }
}

fn persist_preview(
    state: &AppState,
    task_id: &str,
    model: &str,
    params: &Value,
    out: &danmo_engine::RunOutput,
) -> Result<Vec<String>, String> {
    if out.preview_png.is_empty() {
        return Ok(Vec::new());
    }
    let preview_dir = state.outputs.join("previews");
    std::fs::create_dir_all(&preview_dir).map_err(|err| err.to_string())?;
    std::fs::write(preview_dir.join(format!("{task_id}.png")), &out.preview_png)
        .map_err(|err| err.to_string())?;
    let id = product::new_id("ast_");
    let asset_dir = state.outputs.join("assets");
    std::fs::create_dir_all(&asset_dir).map_err(|err| err.to_string())?;
    let path = asset_dir.join(format!("{id}.png"));
    std::fs::write(&path, &out.preview_png).map_err(|err| err.to_string())?;
    let (width, height) = png_size(&out.preview_png);
    let request = params.get("request").cloned().unwrap_or(json!({}));
    let metadata = json!({
        "prompt": request.get("prompt").and_then(|v| v.as_str()).unwrap_or(""),
        "title": request.get("title").and_then(|v| v.as_str()).unwrap_or(""),
        "model": model,
        "requested_size": request.get("size").and_then(|v| v.as_str()).unwrap_or(""),
        "decode": "decoder.conv.weight",
    });
    state
        .scheduler
        .insert_asset(&AssetRow {
            id: id.clone(),
            kind: "image".into(),
            mime_type: "image/png".into(),
            file_path: path.display().to_string(),
            thumbnail_path: None,
            width: Some(width),
            height: Some(height),
            duration_seconds: None,
            source_task_id: Some(task_id.to_string()),
            source_action: Some("create".into()),
            metadata: metadata.to_string(),
            created_at: String::new(),
            parent_asset_id: None,
            relation_type: None,
            group_id: None,
        })
        .map_err(|err| err.to_string())?;
    Ok(vec![id])
}

fn png_size(png: &[u8]) -> (i64, i64) {
    if png.len() < 24 {
        return (0, 0);
    }
    let width = u32::from_be_bytes(png[16..20].try_into().unwrap_or([0, 0, 0, 0]));
    let height = u32::from_be_bytes(png[20..24].try_into().unwrap_or([0, 0, 0, 0]));
    (width as i64, height as i64)
}

fn gguf_bundle(registry: &std::path::Path, model: &str) -> Result<PathBuf, String> {
    let catalog = Catalog::open(registry).map_err(|e| e.to_string())?;
    product::materialize_graph_packs(&catalog)?;
    let path = PathBuf::from(catalog.gguf_local_path(model).map_err(|e| e.to_string())?);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path))
    }
}

fn upscale_or_edit(kind: &str, family: &str, bundle: &std::path::Path) -> Result<Value, String> {
    if kind == "image.upscale" {
        upscale(family, bundle)
            .map(|_| json!({"ok": true}))
            .map_err(|e| e.to_string())
    } else {
        let _ = danmo_engine::assert_gguf_bundle(bundle).map_err(|e| e.to_string())?;
        Err(format!("{family} edit graph is not built"))
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "runtime": "ggml",
        "backend": format!("{:?}", state.backend).to_lowercase(),
    }))
}

async fn registry_route(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let text = std::fs::read_to_string(&state.registry).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("registry unreadable: {e}"),
        )
    })?;
    let catalog = Catalog::from_value(
        serde_json::from_str(&text).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
    );
    Ok(Json(catalog.public_registry()))
}

async fn queue(State(state): State<AppState>) -> Result<Json<Value>, product::ApiError> {
    product::queue_body(&state).map(Json)
}

async fn stream_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        loop {
            let row = state.scheduler.get(&id).ok().flatten();
            let Some(row) = row else {
                let _ = tx
                    .send(Ok(Event::default().event("done").data("missing")))
                    .await;
                break;
            };
            let progress = json!({
                "progress": row.progress,
                "message": "denoise",
            });
            let _ = tx
                .send(Ok(Event::default()
                    .event("progress")
                    .data(progress.to_string())))
                .await;
            let status = json!({
                "status": row.status,
                "progress": row.progress,
            });
            let _ = tx
                .send(Ok(Event::default()
                    .event("status")
                    .data(status.to_string())))
                .await;
            if row.status == "completed" || row.status == "failed" || row.status == "cancelled" {
                let payload = row
                    .result
                    .as_deref()
                    .and_then(|text| serde_json::from_str::<Value>(text).ok())
                    .unwrap_or(Value::Null);
                let _ = tx
                    .send(Ok(Event::default()
                        .event("result")
                        .data(payload.to_string())))
                    .await;
                let done = json!({ "status": row.status });
                let _ = tx
                    .send(Ok(Event::default().event("done").data(done.to_string())))
                    .await;
                break;
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    });
    Sse::new(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn wait_terminal(app: &Router, id: &str) -> Value {
        wait_terminal_for(app, id, 50, 40).await
    }

    async fn wait_terminal_for(app: &Router, id: &str, tries: u32, pause_ms: u64) -> Value {
        let mut last = json!({});
        for _ in 0..tries {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/api/tasks/{id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            last = serde_json::from_slice(&bytes).unwrap();
            let status = last["status"].as_str().unwrap_or("");
            if status == "completed" || status == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(pause_ms)).await;
        }
        last
    }

    #[tokio::test]
    async fn flux2_generation_loads_gguf_bundle() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-ok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let bundle = dir.join("flux2-klein-4b-bf16");
        danmo_engine::write_smoke_bundle(&bundle, 4).unwrap();
        let registry = dir.join("registry.json");
        std::fs::write(
            &registry,
            serde_json::json!({
                "models": {
                    "flux2-klein-4b": {
                        "catalog": { "media": "image" },
                        "runtime": { "family": "flux2", "backends": ["ggml"] },
                        "distribution": {
                            "versions": {
                                "bf16": { "local_path": bundle, "default": true }
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let app = router(&dir.join("studio.db"), &registry);
        let body = serde_json::json!({
            "model": "flux2-klein-4b",
            "prompt": "a cat",
            "size": "64x64",
            "steps": 2
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let accepted: Value = serde_json::from_slice(&bytes).unwrap();
        let id = accepted["id"].as_str().unwrap();
        let task = wait_terminal(&app, id).await;
        assert_eq!(task["status"], "completed", "task {task}");
        let ids = task["result"]["asset_ids"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(ids.len(), 1, "task {task}");
        let assets = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/assets")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(assets.status(), StatusCode::OK);
        let bytes = assets.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(listed["items"][0]["id"], ids[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn flux2_klein_chain_pack_writes_a_64px_png() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-old-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json");
        let app = router(&dir.join("studio.db"), &registry);
        let body = serde_json::json!({
            "model": "flux2-klein-4b",
            "prompt": "a cat",
            "size": "64x64",
            "steps": 2
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let accepted: Value = serde_json::from_slice(&bytes).unwrap();
        let id = accepted["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 120, 250).await;
        assert_eq!(task["status"], "completed", "task {task}");
        let ids = task["result"]["asset_ids"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(ids.len(), 1, "task {task}");
        let asset_id = ids[0].as_str().unwrap();
        let file = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/assets/{asset_id}/file"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(file.status(), StatusCode::OK);
        let png = file.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
        assert_eq!((width, height), (64, 64));

        let registry_body = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/registry")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = registry_body
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let model = &listed["models"]["flux2-klein-4b"];
        assert_eq!(model["media"], "image");
        assert_eq!(
            model["parameters"]["resolution_presets"]["default"],
            "64x64"
        );
        let options = model["parameters"]["resolution_presets"]["options"]
            .as_array()
            .unwrap();
        assert!(options.iter().any(|v| v == "64x64"));
        assert!(options.iter().any(|v| v == "1024x1024"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn spa_reads_tasks_canvas_and_settings() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-spa-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json");
        let app = router(&dir.join("studio.db"), &registry);

        let settings = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/settings")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(settings.status(), StatusCode::OK);
        let bytes = settings.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["language"], "zh");
        assert_eq!(body["queue_image_first"], true);

        let system = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/settings/system")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(system.status(), StatusCode::OK);
        let bytes = system.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["env_ready"], true);
        assert_eq!(body["runtime"], "ggml");

        let queue = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/queue")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = queue.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body["running"].is_array());
        assert!(body["queued"].is_array());

        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/canvas/sessions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"media":"image","title":"desk"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let bytes = created.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        let id = body["id"].as_str().unwrap();
        assert_eq!(body["title"], "desk");

        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/canvas/sessions?media=image")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = listed.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["items"][0]["id"], id);

        let convert = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/download/convert")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(convert.status(), StatusCode::BAD_REQUEST);
        let bytes = convert.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            body["detail"].as_str().unwrap_or("").contains("GGUF"),
            "{body}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn accept(
    state: &AppState,
    kind: TaskKind,
    model: &str,
    request: &impl serde::Serialize,
    priority: &str,
) -> Result<Json<TaskAccepted>, (StatusCode, String)> {
    let catalog =
        Catalog::open(&state.registry).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let entry = catalog
        .model(model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let id = Uuid::new_v4().to_string();
    let params = json!({
        "task_id": id,
        "family": entry.family,
        "request": request,
    });
    let prio = if priority == "high" { 10 } else { 100 };
    state
        .scheduler
        .submit(&id, kind.as_str(), &entry.id, &params.to_string(), prio)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(TaskAccepted {
        id,
        status: "queued".into(),
        kind: kind.as_str().into(),
    }))
}

async fn image_generations(
    State(state): State<AppState>,
    Json(body): Json<ImageGenerationRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::ImageGeneration,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn image_edits(
    State(state): State<AppState>,
    Json(body): Json<ImageEditRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::ImageEdit,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn image_upscales(
    State(state): State<AppState>,
    Json(body): Json<ImageUpscaleRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::ImageUpscale,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn video_generations(
    State(state): State<AppState>,
    Json(body): Json<VideoGenerationRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::VideoGeneration,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn video_edits(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    accept_value(&state, "video.edit", &body)
}

async fn video_upscales(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    accept_value(&state, "video.upscale", &body)
}

async fn video_avatars(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    accept_value(&state, "video.avatar", &body)
}

async fn video_avatar_script(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    accept_value(&state, "video.avatar", &body)
}

async fn audio_edits(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    accept_value(&state, "audio.edit", &body)
}

fn accept_value(
    state: &AppState,
    kind: &str,
    body: &Value,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if model.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "model is required".into()));
    }
    let priority = body
        .get("priority")
        .and_then(|v| v.as_str())
        .unwrap_or("normal");
    let catalog =
        Catalog::open(&state.registry).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let entry = catalog
        .model(&model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let id = Uuid::new_v4().to_string();
    let params = json!({
        "task_id": id,
        "family": entry.family,
        "request": body,
    });
    let prio = if priority == "high" { 10 } else { 100 };
    state
        .scheduler
        .submit(&id, kind, &entry.id, &params.to_string(), prio)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok((
        StatusCode::ACCEPTED,
        Json(TaskAccepted {
            id,
            status: "queued".into(),
            kind: kind.into(),
        }),
    ))
}

async fn audio_generations(
    State(state): State<AppState>,
    Json(body): Json<AudioGenerationRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::AudioGeneration,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}
