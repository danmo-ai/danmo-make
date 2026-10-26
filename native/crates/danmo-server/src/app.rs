use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::Catalog;
use danmo_contracts::{
    AudioGenerationRequest, ImageEditRequest, ImageGenerationRequest, ImageUpscaleRequest,
    TaskAccepted, TaskKind, VideoGenerationRequest,
};
use danmo_engine::{audio_create, image_create, upscale, video_create, BackendKind};
use danmo_schedule::Scheduler;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    scheduler: Scheduler,
    backend: BackendKind,
    registry: PathBuf,
}

pub fn router(db: &std::path::Path, registry: &std::path::Path) -> Router {
    let backend = BackendKind::from_env().unwrap_or(BackendKind::Cpu);
    let scheduler = Scheduler::open(db, true).expect("open task db");
    let state = AppState {
        scheduler,
        backend,
        registry: registry.to_path_buf(),
    };
    spawn_worker(state.clone());
    Router::new()
        .route("/api/system/health", get(health))
        .route("/api/registry", get(registry_route))
        .route("/api/queue", get(queue))
        .route("/api/tasks", get(list_tasks))
        .route("/api/tasks/:id", get(get_task))
        .route("/api/tasks/:id/stream", get(stream_task))
        .route("/api/images/generations", post(image_generations))
        .route("/api/images/edits", post(image_edits))
        .route("/api/images/upscales", post(image_upscales))
        .route("/api/videos/generations", post(video_generations))
        .route("/api/audios/generations", post(audio_generations))
        .with_state(state)
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
            | "audio.generation"
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
            let out = image_create(&request, family, &bundle, state.backend, |step, steps| {
                let _ = state.scheduler.progress(
                    params.get("task_id").and_then(|v| v.as_str()).unwrap_or(""),
                    step as f64 / steps as f64,
                    &format!("step {step}/{steps}"),
                );
            })
            .map_err(|e| e.to_string())?;
            Ok(json!({
                "family": out.family,
                "steps": out.steps,
                "graph_computes": out.graph_computes,
                "width": out.width,
                "height": out.height,
                "latent_len": out.latents.len(),
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
        other => Err(format!("unsupported task kind {other}")),
    }
}

fn gguf_bundle(registry: &std::path::Path, model: &str) -> Result<PathBuf, String> {
    let catalog = Catalog::open(registry).map_err(|e| e.to_string())?;
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

async fn queue(State(state): State<AppState>) -> Json<Value> {
    let _ = &state;
    Json(json!({ "queue_image_first": state.scheduler.queue_image_first }))
}

async fn list_tasks() -> Json<Value> {
    Json(json!({ "tasks": [] }))
}

async fn get_task(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.scheduler.get(&id) {
        Ok(Some(row)) => (
            StatusCode::OK,
            Json(json!({
                "id": row.id,
                "kind": row.kind,
                "model": row.model_id,
                "status": row.status,
                "progress": row.progress,
                "error": row.error_message,
                "result": row.result,
            })),
        )
            .into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "task not found").into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
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
            let _ = tx
                .send(Ok(Event::default()
                    .event("progress")
                    .data(row.progress.to_string())))
                .await;
            if row.status == "succeeded" || row.status == "failed" || row.status == "cancelled" {
                let payload = row
                    .result
                    .unwrap_or_else(|| row.error_message.unwrap_or_default());
                let _ = tx
                    .send(Ok(Event::default().event("result").data(payload)))
                    .await;
                let _ = tx
                    .send(Ok(Event::default().event("done").data(row.status)))
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
        let mut last = json!({});
        for _ in 0..50 {
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
            if status == "succeeded" || status == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(40)).await;
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
        assert_eq!(task["status"], "succeeded", "task {task}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn legacy_registry_weights_fail_the_task() {
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
        let task = wait_terminal(&app, id).await;
        assert_eq!(task["status"], "failed", "task {task}");
        let err = task["error"].as_str().unwrap_or("");
        assert!(err.contains("Re-download"), "{err}");
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
