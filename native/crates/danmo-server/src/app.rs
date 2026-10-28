use crate::download::{self, Downloads};
use crate::product::{self, fail, ApiError};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::Catalog;
use danmo_contracts::{
    AudioGenerationRequest, ImageGenerationRequest, TaskAccepted, TaskKind, VideoGenerationRequest,
};
use danmo_engine::{audio_create, image_create, spec, upscale, video_create, BackendKind};
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
    pub(crate) downloads: Downloads,
}

pub fn router(db: &std::path::Path, registry: &std::path::Path) -> Router {
    let backend = BackendKind::from_env().unwrap_or(BackendKind::Cpu);
    let workspace = workspace_of(db);
    let outputs = std::env::var("DANQING_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("outputs"));
    let scheduler = Scheduler::open(db, true).expect("open task db");
    let downloads = Downloads::open(&workspace);
    let state = AppState {
        scheduler,
        backend,
        registry: registry.to_path_buf(),
        workspace,
        outputs,
        downloads,
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
        return absolute_dir(PathBuf::from(path));
    }
    let raw = match db.parent() {
        Some(parent) if parent.file_name().and_then(|name| name.to_str()) == Some("db") => parent
            .parent()
            .map(|path| path.to_path_buf())
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from(".")),
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    absolute_dir(raw)
}

fn absolute_dir(path: PathBuf) -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if path.as_os_str().is_empty() || path == PathBuf::from(".") {
        cwd
    } else if path.is_absolute() {
        path
    } else {
        cwd.join(path)
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
        Some(gguf_bundle(state, model)?)
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
            let count = request.n.max(1);
            let mut asset_ids = Vec::new();
            let mut last = None;
            let mut computes = 0u32;
            for index in 0..count {
                let mut one = request.clone();
                if count > 1 {
                    one.seed = Some(request.seed.unwrap_or(0) + index as i64);
                }
                let out = image_create(&one, family, &bundle, state.backend, |step, steps| {
                    let done = index as f64 * steps as f64 + step as f64;
                    let all = count as f64 * steps as f64;
                    let _ = state.scheduler.progress(
                        &task_id,
                        done / all,
                        &format!("step {step}/{steps}"),
                    );
                })
                .map_err(|e| e.to_string())?;
                computes += out.graph_computes;
                asset_ids.extend(persist_preview(state, &task_id, model, params, &out)?);
                last = Some(out);
            }
            let out = last.ok_or_else(|| "n must be at least 1".to_string())?;
            Ok(json!({
                "family": out.family,
                "steps": out.steps,
                "graph_computes": computes,
                "width": out.width,
                "height": out.height,
                "latent_len": out.latents.len(),
                "asset_ids": asset_ids,
                "primary_asset_id": asset_ids.first(),
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
            let request: AudioGenerationRequest =
                serde_json::from_value(params.get("request").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let lyrics = if request.lyrics.trim().is_empty() {
                "[inst]"
            } else {
                request.lyrics.as_str()
            };
            let vocal = if request.vocal_language.trim().is_empty() {
                "en"
            } else {
                request.vocal_language.as_str()
            };
            let out = audio_create(
                family,
                &bundle,
                &danmo_engine::AudioCreateRequest {
                    prompt: &request.prompt,
                    lyrics,
                    duration_sec: request.duration.unwrap_or(30.0),
                    seed: request.seed,
                    guidance: request.guidance.unwrap_or(1.0),
                    steps: request.steps,
                    vocal_language: vocal,
                },
            )
            .map_err(|e| e.to_string())?;
            let task_id = params
                .get("task_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let asset_ids = persist_audio_wav(state, task_id, model, params, &out)?;
            let alignment: Value = serde_json::from_str(&out.lyrics_alignment_json)
                .unwrap_or_else(|_| json!({"lyrics_alignment": "unknown"}));
            Ok(json!({
                "family": out.family,
                "steps": out.steps,
                "graph_computes": out.graph_computes,
                "frames": out.frames,
                "sample_rate": out.sample_rate,
                "lyrics_effective": out.lyrics_effective,
                "lyrics_alignment": alignment.get("lyrics_alignment").cloned().unwrap_or(json!("unknown")),
                "lyrics_segments": alignment.get("lyrics_segments").cloned().unwrap_or(json!([])),
                "asset_ids": asset_ids,
                "primary_asset_id": asset_ids.first(),
            }))
        }
        "audio.edit" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            danmo_engine::assert_gguf_bundle(&bundle).map_err(|e| e.to_string())?;
            if family == "ace_step" {
                return Err(
                    "ACE-Step cover needs VAE encode + FSQ roundtrip (acestep.cpp ops_encode_src)"
                        .into(),
                );
            }
            Err(format!("{family} graph is not built"))
        }
        "video.edit" | "video.upscale" | "video.avatar" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            danmo_engine::assert_gguf_bundle(&bundle).map_err(|e| e.to_string())?;
            Err(format!("{family} graph is not built"))
        }
        other => Err(format!("unsupported task kind {other}")),
    }
}

fn persist_audio_wav(
    state: &AppState,
    task_id: &str,
    model: &str,
    params: &Value,
    out: &danmo_engine::AudioRunOutput,
) -> Result<Vec<String>, String> {
    if out.wav.is_empty() {
        return Ok(Vec::new());
    }
    let id = product::new_id("ast_");
    let asset_dir = state.outputs.join("assets");
    std::fs::create_dir_all(&asset_dir).map_err(|err| err.to_string())?;
    let path = asset_dir.join(format!("{id}.wav"));
    std::fs::write(&path, &out.wav).map_err(|err| err.to_string())?;
    // Main-compatible sidecars: `<stem>_lyrics.txt` + `<stem>.lrc`.
    if let Some(txt) = &out.lyrics_txt {
        let side = asset_dir.join(format!("{id}_lyrics.txt"));
        std::fs::write(&side, txt).map_err(|err| err.to_string())?;
    }
    if let Some(lrc) = &out.lyrics_lrc {
        let side = asset_dir.join(format!("{id}.lrc"));
        std::fs::write(&side, lrc).map_err(|err| err.to_string())?;
    }
    let request = params.get("request").cloned().unwrap_or(json!({}));
    let duration_seconds = out.frames as f64 / 25.0;
    let alignment: Value = serde_json::from_str(&out.lyrics_alignment_json)
        .unwrap_or_else(|_| json!({"lyrics_alignment": "unknown"}));
    let metadata = json!({
        "prompt": request.get("prompt").and_then(|v| v.as_str()).unwrap_or(""),
        "lyrics": request.get("lyrics").and_then(|v| v.as_str()).unwrap_or(""),
        "lyrics_effective": out.lyrics_effective,
        "lyrics_input": request.get("lyrics").and_then(|v| v.as_str()).unwrap_or(""),
        "model": model,
        "sample_rate": out.sample_rate,
        "frames": out.frames,
        "steps": out.steps,
        "lyrics_alignment": alignment.get("lyrics_alignment").cloned().unwrap_or(json!("unknown")),
        "lyrics_segments": alignment.get("lyrics_segments").cloned().unwrap_or(json!([])),
        "lyrics_sidecar": out.lyrics_txt.is_some(),
        "lrc_sidecar": out.lyrics_lrc.is_some(),
    });
    state
        .scheduler
        .insert_asset(&AssetRow {
            id: id.clone(),
            kind: "audio".into(),
            mime_type: "audio/wav".into(),
            file_path: path.display().to_string(),
            thumbnail_path: None,
            width: None,
            height: None,
            duration_seconds: Some(duration_seconds),
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

fn parse_step_message(message: &str) -> (u32, u32) {
    let Some(rest) = message.strip_prefix("step ") else {
        return (0, 0);
    };
    let Some((step, total)) = rest.split_once('/') else {
        return (0, 0);
    };
    (
        step.trim().parse().unwrap_or(0),
        total.trim().parse().unwrap_or(0),
    )
}

fn png_size(png: &[u8]) -> (i64, i64) {
    if png.len() < 24 {
        return (0, 0);
    }
    let width = u32::from_be_bytes(png[16..20].try_into().unwrap_or([0, 0, 0, 0]));
    let height = u32::from_be_bytes(png[20..24].try_into().unwrap_or([0, 0, 0, 0]));
    (width as i64, height as i64)
}

fn gguf_bundle(state: &AppState, model: &str) -> Result<PathBuf, String> {
    let catalog = Catalog::open(&state.registry).map_err(|e| e.to_string())?;
    let path = catalog.gguf_local_path(model).map_err(|e| e.to_string())?;
    Ok(download::resolve_model_path(&state.workspace, &path))
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
        let mut last_log_id = 0i64;
        loop {
            let row = state.scheduler.get(&id).ok().flatten();
            let Some(row) = row else {
                let _ = tx
                    .send(Ok(Event::default().event("done").data("missing")))
                    .await;
                break;
            };
            let logs = state.scheduler.logs(&id, 0, 50).unwrap_or_default();
            if let Some(log) = logs.last() {
                if log.id != last_log_id {
                    last_log_id = log.id;
                    let payload = json!({
                        "id": log.id,
                        "message": log.message,
                        "level": log.level,
                        "time": log.time,
                    });
                    let _ = tx
                        .send(Ok(Event::default().event("log").data(payload.to_string())))
                        .await;
                }
            }
            let message = logs
                .last()
                .map(|log| log.message.clone())
                .unwrap_or_else(|| "denoise".into());
            let (step, total) = parse_step_message(&message);
            let progress = json!({
                "progress": row.progress,
                "message": "denoise",
                "step": step,
                "total": total,
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
    use std::io::{Read, Write};
    use tower::ServiceExt;

    fn factory_registry() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json")
    }

    /// Weight key in, registry install directory out.
    fn published_weight(model: &str, weight: &str) -> (String, String) {
        let catalog = Catalog::open(&factory_registry()).expect("factory registry");
        let id = format!("{model}:{weight}");
        let path = catalog
            .gguf_local_path(&id)
            .unwrap_or_else(|err| panic!("{id}: {err}"));
        (id, path)
    }

    fn vue_generate_body(model: &str, prompt: &str) -> Value {
        json!({
            "model": model,
            "title": "chain",
            "prompt": prompt,
            "negative_prompt": "",
            "size": "64x64",
            "n": 1,
            "steps": 2,
            "guidance": 1.0,
            "seed": 1,
            "adapters": [],
            "metadata": { "source": "create" },
            "priority": "normal"
        })
    }

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
        let (model, local_path) = published_weight("flux2-klein-4b", "q4_k");
        let bundle = dir.join(&local_path);
        danmo_engine::write_smoke_bundle(&bundle, 4).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let body = serde_json::json!({
            "model": model,
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
    async fn image_create_rejects_unbuilt_inference_options() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-reject-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let (model, _) = published_weight("z-image", "q4_k");
        let mut body = vue_generate_body(&model, "a cat");
        body["teacache_mode"] = json!("auto");
        body["lemica_mode"] = json!("none");
        body["adapters"] = json!([{ "id": "portrait", "weight": 0.8 }]);
        let missing = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(missing.0, StatusCode::BAD_REQUEST);
        assert!(
            missing.1["detail"]
                .as_str()
                .unwrap()
                .contains("lora portrait is not installed"),
            "{}",
            missing.1
        );
        body["model"] = json!("fibo");
        let refused = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(refused.0, StatusCode::NOT_IMPLEMENTED);
        assert!(
            refused.1["detail"]
                .as_str()
                .unwrap()
                .contains("lora inference"),
            "{}",
            refused.1
        );
        let config = dir.join("config");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(
            config.join("user_loras.json"),
            r#"{"items":[{"id":"user-lora-abc","base_model":"z-image-turbo","local_path":"missing.safetensors","inference":{"scheme":"sft","model":"z-image-turbo"}}]}"#,
        )
        .unwrap();
        body["model"] = json!("z-image-turbo");
        body["adapters"] = json!([{ "id": "user-lora-abc", "weight": 0.8 }]);
        let sft = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(sft.0, StatusCode::NOT_IMPLEMENTED);
        assert!(
            sft.1["detail"]
                .as_str()
                .unwrap()
                .contains("Turbo SFT DistillPatch is not built"),
            "{}",
            sft.1
        );
        body["model"] = json!(model);
        body["adapters"] = json!([]);
        body["negative_prompt"] = json!("blurry");
        let accepted_negative = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(
            accepted_negative.0,
            StatusCode::ACCEPTED,
            "{}",
            accepted_negative.1
        );
        body["negative_prompt"] = json!("");
        body["scheduler"] = json!("flow_match_euler");
        body["scheduler_shift"] = json!(6);
        let accepted = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(accepted.0, StatusCode::ACCEPTED, "{}", accepted.1);
        body["scheduler"] = json!("flow_match_euler_discrete");
        body["use_empirical_mu"] = json!(true);
        let empirical = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(empirical.0, StatusCode::ACCEPTED, "{}", empirical.1);
        let edit = post_json(&app, "/api/images/edits", &body).await;
        assert_eq!(edit.0, StatusCode::NOT_IMPLEMENTED);
        assert!(
            edit.1["detail"].as_str().unwrap().contains("edit graph"),
            "{}",
            edit.1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn post_json(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let payload = serde_json::from_slice(&bytes)
            .unwrap_or(json!({ "raw": String::from_utf8_lossy(&bytes).to_string() }));
        (status, payload)
    }

    #[tokio::test]
    async fn flux2_klein_q4_uses_the_download_directory() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-q4-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("flux2-klein-4b", "q4_k");
        let weight = model.split(':').nth(1).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        assert!(!dir.join(&local_path).exists());
        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(vue_generate_body(&model, "一只猫").to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::ACCEPTED);
        let bytes = missing.into_body().collect().await.unwrap().to_bytes();
        let accepted: Value = serde_json::from_slice(&bytes).unwrap();
        let failed = wait_terminal(&app, accepted["id"].as_str().unwrap()).await;
        assert_eq!(failed["status"], "failed", "{failed}");
        let err = failed["error"].as_str().unwrap_or("");
        assert!(
            err.contains(&local_path),
            "generation did not use the registry path for {model}: {err}"
        );
        assert_eq!(failed["model_id"], model);

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
        assert_eq!(model["recommended"], true);
        assert_eq!(
            model["parameters"]["resolution_presets"]["default"],
            "512x512"
        );
        let options = model["parameters"]["resolution_presets"]["options"]
            .as_array()
            .unwrap();
        assert!(options.iter().any(|v| v == "64x64"));
        assert!(options.iter().any(|v| v == "1024x1024"));
        let q4 = &model["versions"][weight];
        assert_eq!(q4["default"], true);
        assert_eq!(q4["local_path"], local_path);
        let patterns = q4["allow_patterns"].as_array().expect("allow_patterns");
        assert!(
            patterns
                .iter()
                .any(|name| name.as_str().unwrap_or("").ends_with(".gguf")),
            "{patterns:?}"
        );
        assert!(q4.get("graph_pack").is_none());
        assert!(model["versions"]["bf16"].get("graph_pack").is_none());
        let klein9 = &listed["models"]["flux2-klein-9b"]["versions"];
        assert_eq!(klein9["bf16"]["size"], "34.7GB");
        assert_eq!(klein9["q8_0"]["size"], "26.5GB");
        assert_eq!(klein9["q4_k"]["size"], "22.5GB");
        assert_eq!(
            klein9["q4_k"]["components"][0]["repo_id"],
            "black-forest-labs/FLUX.2-klein-9B"
        );
        let turbo = &listed["models"]["z-image-turbo"]["versions"];
        assert_eq!(turbo["bf16"]["size"], "20.5GB");
        assert_eq!(turbo["q8_0"]["size"], "15.5GB");
        assert_eq!(turbo["q4_k"]["size"], "13.2GB");
        assert_eq!(
            turbo["q4_k"]["components"][0]["repo_id"],
            "Tongyi-MAI/Z-Image-Turbo"
        );
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
        let app = router(&dir.join("studio.db"), &factory_registry());

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
        assert_eq!(convert.status(), StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn download_page_fetches_a_gguf_over_http() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let payload = b"gguf-demo-payload";
        std::thread::spawn(move || {
            for _ in 0..4 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 1];
                let mut header = Vec::new();
                loop {
                    if stream.read(&mut buf).ok() != Some(1) {
                        break;
                    }
                    header.push(buf[0]);
                    if header.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let body = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(body.as_bytes());
                let _ = stream.write_all(payload);
            }
        });

        let dir = std::env::temp_dir().join(format!("danmo-dl-route-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bundle = dir.join("models/demo-http");
        let registry = dir.join("registry.json");
        let spec = json!({
            "models": {
                "demo-gguf": {
                    "catalog": {
                        "media": "image",
                        "name": { "en": "Demo GGUF" }
                    },
                    "ui": { "parameters": { "lora_support": true } },
                    "distribution": {
                        "versions": {
                            "bf16": {
                                "local_path": "models/demo-http",
                                "url": format!("http://127.0.0.1:{port}/demo.gguf"),
                                "source": "huggingface"
                            },
                            "q4_k": {
                                "local_path": "models/demo-http/q4",
                                "url": format!("http://127.0.0.1:{port}/model.safetensors"),
                                "source": "huggingface"
                            }
                        }
                    }
                }
            }
        });
        std::fs::write(&registry, spec.to_string()).unwrap();
        let app = router(&dir.join("studio.db"), &registry);

        let disk = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/settings/disk-space")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = disk.into_body().collect().await.unwrap().to_bytes();
        let disk: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(disk["models"]["free_human"].is_string(), "{disk}");

        let bases = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/download/lora/base-models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = bases.into_body().collect().await.unwrap().to_bytes();
        let bases: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(bases
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["id"] == "demo-gguf"));

        let started = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/models/demo-gguf/install")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"version":"bf16"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(started.status(), StatusCode::OK);
        let bytes = started.into_body().collect().await.unwrap().to_bytes();
        let started: Value = serde_json::from_slice(&bytes).unwrap();
        let task_id = started["task_id"].as_str().unwrap().to_string();

        let mut status = String::new();
        let mut error = String::new();
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let listed = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/download/tasks")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = listed.into_body().collect().await.unwrap().to_bytes();
            let tasks: Value = serde_json::from_slice(&bytes).unwrap();
            let task = tasks
                .as_array()
                .unwrap()
                .iter()
                .find(|task| task["id"] == task_id)
                .unwrap();
            status = task["status"].as_str().unwrap_or("").to_string();
            error = task["error_message"].as_str().unwrap_or("").to_string();
            if matches!(status.as_str(), "completed" | "failed" | "cancelled") {
                break;
            }
        }
        assert_eq!(status, "completed", "{error}");
        assert_eq!(std::fs::read(bundle.join("demo.gguf")).unwrap(), payload);

        let detailed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/settings/models/status/detailed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = detailed.into_body().collect().await.unwrap().to_bytes();
        let detailed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(detailed["demo-gguf"]["versions"]["bf16"]["ready"], true);

        let stream = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/download/progress/{task_id}/stream"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stream.status(), StatusCode::OK);
        let bytes = stream.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("completed"), "{text}");

        let removed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/models/demo-gguf/versions/bf16")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(removed.status(), StatusCode::OK, "{removed:?}");
        assert!(!bundle.join("demo.gguf").exists());

        let refused = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/models/demo-gguf/install")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"version":"q4_k"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = refused.into_body().collect().await.unwrap().to_bytes();
        let refused: Value = serde_json::from_slice(&bytes).unwrap();
        let refused_id = refused["task_id"].as_str().unwrap();
        let mut refused_error = String::new();
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let listed = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/download/tasks")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = listed.into_body().collect().await.unwrap().to_bytes();
            let tasks: Value = serde_json::from_slice(&bytes).unwrap();
            let task = tasks
                .as_array()
                .unwrap()
                .iter()
                .find(|task| task["id"] == refused_id)
                .unwrap();
            if task["status"] == "failed" {
                refused_error = task["error_message"].as_str().unwrap_or("").to_string();
                break;
            }
            if task["status"] == "completed" {
                panic!("safetensors URL was downloaded: {task}");
            }
        }
        assert!(refused_error.contains("GGUF"), "{refused_error}");
        let _ = std::fs::remove_dir_all(&bundle);
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
        .submit(
            &id,
            kind.as_str(),
            &stored_model_id(model, &entry.id),
            &params.to_string(),
            prio,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(TaskAccepted {
        id,
        status: "queued".into(),
        kind: kind.as_str().into(),
    }))
}

fn stored_model_id(requested: &str, bare: &str) -> String {
    match requested.split_once(':') {
        Some((id, version))
            if id == bare && danmo_catalog::GGUF_VERSION_KEYS.contains(&version) =>
        {
            requested.to_string()
        }
        _ => bare.to_string(),
    }
}

async fn image_generations(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let request = prepare_image_request(&state, body)?;
    let accepted = accept(
        &state,
        TaskKind::ImageGeneration,
        &request.model,
        &request,
        &request.priority,
    )
    .map_err(|(status, detail)| fail(status, detail))?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn image_edits(
    State(_state): State<AppState>,
    Json(_body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    Err(fail(StatusCode::NOT_IMPLEMENTED, "edit graph is not built"))
}

async fn image_upscales(
    State(_state): State<AppState>,
    Json(_body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    Err(fail(
        StatusCode::NOT_IMPLEMENTED,
        "upscale graph is not built",
    ))
}

fn prepare_image_request(
    state: &AppState,
    mut body: Value,
) -> Result<ImageGenerationRequest, ApiError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if model.is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "model is required"));
    }
    let catalog = Catalog::open(&state.registry)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    let entry = catalog
        .model(&model)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    reject_unbuilt_inference(&entry.family, &body)?;
    bind_installed_loras(state, &model, &mut body)?;
    fill_registry_defaults(&catalog, &entry.id, &mut body);
    let request: ImageGenerationRequest = serde_json::from_value(body)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    if request.n == 0 {
        return Err(fail(StatusCode::BAD_REQUEST, "n must be at least 1"));
    }
    Ok(request)
}

fn reject_unbuilt_inference(family: &str, body: &Value) -> Result<(), ApiError> {
    let adapters = body.get("adapters").and_then(|v| v.as_array());
    let family_spec = spec(family);
    if adapters.is_some_and(|rows| !rows.is_empty())
        && !family_spec.is_some_and(|row| row.lora_inference)
    {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "lora inference graph is not built",
        ));
    }
    if body
        .get("structural_guide")
        .is_some_and(|value| !value.is_null())
    {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "controlnet graph is not built",
        ));
    }
    if body
        .get("latent_refine")
        .is_some_and(|value| !value.is_null())
    {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "latent refine graph is not built",
        ));
    }
    let teacache = text_field(body, "teacache_mode").unwrap_or("");
    if !matches!(teacache, "" | "none" | "auto") {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "teacache graph is not built",
        ));
    }
    let lemica = text_field(body, "lemica_mode").unwrap_or("");
    if !matches!(lemica, "" | "none" | "off") {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            "lemica graph is not built",
        ));
    }
    if let Some(scheduler) = text_field(body, "scheduler") {
        let accepted = family_spec.is_some_and(|row| row.accepts_scheduler(scheduler));
        if !accepted {
            return Err(fail(
                StatusCode::NOT_IMPLEMENTED,
                format!("scheduler {scheduler} is not built"),
            ));
        }
    }
    Ok(())
}

fn bind_installed_loras(state: &AppState, model: &str, body: &mut Value) -> Result<(), ApiError> {
    let Some(rows) = body.get_mut("adapters").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    if rows.is_empty() {
        return Ok(());
    }
    let bare = model.split(':').next().unwrap_or(model);
    for row in rows.iter_mut() {
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if id.is_empty() {
            return Err(fail(StatusCode::BAD_REQUEST, "lora id is required"));
        }
        if bare == "z-image-turbo" {
            match state.downloads.turbo_sft(&id) {
                Ok(true) => {
                    return Err(fail(
                        StatusCode::NOT_IMPLEMENTED,
                        "Turbo SFT DistillPatch is not built",
                    ));
                }
                Ok(false) => {}
                Err(detail) => return Err(fail(StatusCode::BAD_REQUEST, detail)),
            }
        }
        let path = state
            .downloads
            .lora_path(&id)
            .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?;
        let Some(path) = path else {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("lora {id} is not installed"),
            ));
        };
        let Some(obj) = row.as_object_mut() else {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                "lora adapter is not an object",
            ));
        };
        obj.insert(
            "path".to_string(),
            json!(path.to_string_lossy().to_string()),
        );
    }
    Ok(())
}

fn fill_registry_defaults(catalog: &Catalog, id: &str, body: &mut Value) {
    let Some(params) = catalog
        .root()
        .pointer(&format!("/models/{id}/ui/parameters"))
        .and_then(|v| v.as_object())
    else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    for key in ["steps", "guidance", "scheduler", "scheduler_shift"] {
        let present = obj.get(key).is_some_and(|value| !value.is_null());
        if present {
            continue;
        }
        let Some(spec) = params.get(key) else {
            continue;
        };
        let default = spec.get("default").unwrap_or(spec);
        if default.is_null() || default.is_object() || default.is_array() {
            continue;
        }
        obj.insert((*key).to_string(), default.clone());
    }
}

fn text_field<'a>(body: &'a Value, key: &str) -> Option<&'a str> {
    body.get(key).and_then(|v| v.as_str())
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
