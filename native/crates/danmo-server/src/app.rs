use crate::download::{self, Downloads};
use crate::product::{self, fail, ApiError};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use danmo_catalog::Catalog;
use danmo_contracts::{
    AudioEditRequest, AudioGenerationRequest, AudioLmRequest, ImageEditRequest,
    ImageGenerationRequest, ImageUpscaleRequest, TaskAccepted, TaskKind, VideoEditRequest,
    VideoGenerationRequest, VideoUpscaleRequest,
};
use danmo_engine::{spec, BackendKind};
use danmo_schedule::Scheduler;
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
    pub(crate) backend_selection: danmo_engine::BackendSelection,
    pub(crate) registry: PathBuf,
    pub(crate) workspace: PathBuf,
    pub(crate) outputs: PathBuf,
    pub(crate) downloads: Downloads,
}

pub fn try_router(db: &std::path::Path, registry: &std::path::Path) -> Result<Router, String> {
    let backend_selection = danmo_engine::select_from_env().map_err(|error| error.to_string())?;
    let backend = backend_selection.backend;
    tracing::info!(
        "ggml backend {} selected: {}{}",
        backend.name(),
        backend_selection.reason,
        backend_selection
            .probe
            .as_ref()
            .map(|probe| format!("; {}", probe.summary()))
            .unwrap_or_default()
    );
    let workspace = workspace_of(db);
    let outputs = std::env::var("DANQING_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("outputs"));
    let scheduler = Scheduler::open(db, true).map_err(|error| error.to_string())?;
    let downloads = Downloads::open(&workspace);
    let state = AppState {
        scheduler,
        backend,
        backend_selection,
        registry: registry.to_path_buf(),
        workspace,
        outputs,
        downloads,
    };
    spawn_worker(state.clone());
    Ok(Router::new()
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
        .route("/api/audios/lm", post(audio_lm_route))
        .merge(product::routes())
        .fallback(product::spa)
        .with_state(state))
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
        let next = match state.scheduler.dequeue() {
            Ok(next) => next,
            Err(error) => {
                tracing::error!("task dequeue failed: {error}");
                std::thread::sleep(Duration::from_millis(250));
                continue;
            }
        };
        let Some(task) = next else {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        let params: Value = match serde_json::from_str(&task.params) {
            Ok(params) => params,
            Err(error) => {
                let message = format!("invalid task params: {error}");
                if let Err(finish_error) = state.scheduler.finish_err(&task.id, &message) {
                    tracing::error!("task finish failed: {finish_error}");
                }
                continue;
            }
        };
        let cancellation = state.scheduler.start_cancellable(&task.id);
        let outcome = run_task(
            &state,
            &task.kind,
            &task.model_id,
            &params,
            cancellation.clone(),
        );
        state.scheduler.finish_cancellable(&task.id);
        let outcome = if cancellation.is_cancelled() {
            state.scheduler.finish_cancelled(&task.id)
        } else {
            match outcome {
                Ok(result) => state.scheduler.finish_ok(&task.id, &result.to_string()),
                Err(error) => state.scheduler.finish_err(&task.id, &error),
            }
        };
        if let Err(error) = outcome {
            tracing::error!("task finish failed: {error}");
        }
    });
}

fn run_task(
    state: &AppState,
    kind: &str,
    model: &str,
    params: &Value,
    cancellation: danmo_engine::CancellationToken,
) -> Result<Value, String> {
    crate::engine_host::execute(state, kind, model, params, cancellation)
        .map_err(|error| error.to_string())
}

#[cfg(any())]
fn run_task_legacy(
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
            | "audio.lm"
    );
    let task_id = params.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
    let note = |level: &str, message: String| {
        if !task_id.is_empty() {
            let _ = state.scheduler.log(task_id, level, &message);
        }
    };
    let bundle = if needs_bundle {
        let resolved = gguf_bundle(state, model)?;
        note("info", format!("bundle {}", resolved.display()));
        Some(resolved)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let outcome = match kind {
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
            // Prefer catalog LM card when DiT has no companions/lm (fail later if still missing).
            let lm_override = if request.lm_expansion.trim().eq_ignore_ascii_case("off") {
                None
            } else {
                gguf_bundle(state, "ace-step-5hz-lm").ok()
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
                    bpm: request.bpm,
                    keyscale: request.key_scale.as_str(),
                    timesignature: request.time_signature.as_str(),
                    lm_expansion: request.lm_expansion.as_str(),
                    lm_bundle: lm_override.as_deref(),
                    audio_code_indices: request.audio_code_indices.as_deref(),
                    timbre_latents: request.timbre_latents.as_deref(),
                },
                state.backend,
                |phase, step, total| {
                    let task_id = params.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
                    if task_id.is_empty() || total == 0 {
                        return;
                    }
                    // LM expansion owns the first quarter of the bar, denoise the rest.
                    let frac = step as f64 / total as f64;
                    let progress = if phase == "lm" {
                        frac * 0.25
                    } else {
                        0.25 + frac * 0.75
                    };
                    // One row per token would flood the task log; sample it.
                    let sample = phase != "lm" || step % 8 == 0 || step + 1 == total;
                    if !sample {
                        let _ = state.scheduler.set_progress(task_id, progress);
                        return;
                    }
                    let _ = state.scheduler.progress(
                        task_id,
                        progress,
                        &format!("{phase} {step}/{total}"),
                    );
                },
            )
            .map_err(|e| e.to_string())?;
            let task_id = params.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
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
            let request: AudioEditRequest =
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
            let (pcm, _sr) = resolve_edit_source_audio(state, &request)?;
            let out = audio_edit(
                family,
                &bundle,
                state.backend,
                &danmo_engine::AudioEditRequest {
                    prompt: &request.prompt,
                    lyrics,
                    seed: request.seed,
                    guidance: request.guidance.unwrap_or(1.0),
                    steps: request.steps,
                    vocal_language: vocal,
                    task: &request.task,
                    source_audio: pcm.as_deref(),
                    source_latents: request.source_latents.as_deref(),
                    audio_cover_strength: request.audio_cover_strength.unwrap_or(1.0),
                    cover_noise_strength: request.cover_noise_strength.unwrap_or(0.0),
                    region_start_sec: request.region_start_sec,
                    region_end_sec: request.region_end_sec,
                    repaint_injection_ratio: request.repaint_injection_ratio.unwrap_or(1.0),
                    repaint_crossfade_frames: request.repaint_crossfade_frames.unwrap_or(8),
                    timbre_from_source: request.timbre_from_source.unwrap_or(false),
                    track: request.track.as_deref().unwrap_or(""),
                },
            )
            .map_err(|e| e.to_string())?;
            let task_id = params.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
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
                "task": request.task,
            }))
        }
        "audio.lm" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            let request: AudioLmRequest =
                serde_json::from_value(params.get("request").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let vocal = if request.vocal_language.trim().is_empty() {
                "en"
            } else {
                request.vocal_language.as_str()
            };
            let out = audio_lm(
                family,
                &bundle,
                state.backend,
                &danmo_engine::AudioLmRequest {
                    mode: &request.mode,
                    prompt: &request.prompt,
                    lyrics: &request.lyrics,
                    duration: request.duration.unwrap_or(0.0),
                    bpm: request.bpm.unwrap_or(0),
                    keyscale: &request.keyscale,
                    timesignature: &request.timesignature,
                    vocal_language: vocal,
                    max_new_tokens: request.max_new_tokens.unwrap_or(256) as usize,
                    temperature: request.temperature.unwrap_or(0.0),
                    top_p: request.top_p.unwrap_or(1.0),
                    top_k: request.top_k.unwrap_or(0),
                    seed: request.seed.unwrap_or(0),
                    use_fsm: request.use_fsm,
                    cfg_scale: request.cfg_scale.unwrap_or(1.0),
                    negative_prompt: request.negative_prompt.as_str(),
                },
            )
            .map_err(|e| e.to_string())?;
            Ok(json!({
                "family": out.family,
                "mode": out.mode,
                "prompt_tokens": out.prompt_tokens,
                "generated_ids": out.generated_ids,
                "text": out.text,
                "caption": out.caption,
                "lyrics": out.lyrics,
                "bpm": out.bpm,
                "duration": out.duration,
                "keyscale": out.keyscale,
                "timesignature": out.timesignature,
                "language": out.language,
                "audio_code_indices": out.audio_code_indices,
            }))
        }
        "video.edit" | "video.upscale" | "video.avatar" => {
            let bundle = bundle.ok_or_else(|| "missing GGUF bundle".to_string())?;
            danmo_engine::assert_gguf_bundle(&bundle).map_err(|e| e.to_string())?;
            Err(format!("{family} graph is not built"))
        }
        other => Err(format!("unsupported task kind {other}")),
    };
    let elapsed = started.elapsed().as_secs_f64();
    match &outcome {
        Ok(_) => note("info", format!("done in {elapsed:.1}s")),
        Err(err) => note("error", format!("failed after {elapsed:.1}s: {err}")),
    }
    outcome
}

#[cfg(any())]
fn resolve_edit_source_audio(
    state: &AppState,
    request: &AudioEditRequest,
) -> Result<(Option<Vec<f32>>, u32), String> {
    if request
        .source_latents
        .as_ref()
        .is_some_and(|v| !v.is_empty())
    {
        return Ok((None, 48_000));
    }
    let bytes = if let Some(path) = request.source_audio.as_ref().filter(|p| !p.is_empty()) {
        let p = FsPath::new(path);
        let resolved = if p.is_absolute() {
            p.to_path_buf()
        } else {
            state.workspace.join(p)
        };
        std::fs::read(&resolved)
            .map_err(|err| format!("ACE-Step cover source_audio {}: {err}", resolved.display()))?
    } else if let Some(b64) = request
        .source_audio_base64
        .as_ref()
        .filter(|p| !p.is_empty())
    {
        product::base64_decode(b64)
            .map_err(|err| format!("ACE-Step source_audio_base64 is invalid: {err}"))?
    } else {
        return Err(
            "ACE-Step cover/repaint needs source_audio, source_audio_base64, or source_latents"
                .into(),
        );
    };
    let (pcm, sr) = read_audio_wav_pcm16(&bytes).map_err(|e| e.to_string())?;
    if sr != 48_000 {
        return Err(format!(
            "ACE-Step cover source sample rate {sr} Hz must be 48000 (resampling is not built)"
        ));
    }
    Ok((Some(pcm), sr))
}

#[cfg(any())]
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

#[cfg(any())]
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

#[cfg(any())]
fn png_size(png: &[u8]) -> (i64, i64) {
    if png.len() < 24 {
        return (0, 0);
    }
    let width = u32::from_be_bytes(png[16..20].try_into().unwrap_or([0, 0, 0, 0]));
    let height = u32::from_be_bytes(png[20..24].try_into().unwrap_or([0, 0, 0, 0]));
    (width as i64, height as i64)
}

/// Resolve a model id to its installed bundle. `id:version` pins the tier.
/// A bare id picks the first **installed** tier, so a registry `default` the
/// user never downloaded cannot win over one they did.
pub(crate) fn gguf_bundle(state: &AppState, model: &str) -> Result<PathBuf, String> {
    let catalog = Catalog::open(&state.registry).map_err(|e| e.to_string())?;
    let (bare, requested) = match model.split_once(':') {
        Some((bare, version)) if !bare.is_empty() && !version.is_empty() => (bare, Some(version)),
        _ => (model, None),
    };
    if let Some(version) = requested {
        if danmo_catalog::GGUF_VERSION_KEYS.contains(&version) {
            let path = catalog.gguf_local_path(model).map_err(|e| e.to_string())?;
            return Ok(download::resolve_model_path(&state.workspace, &path));
        }
        return safetensors_bundle(&state.workspace, catalog.root(), bare, version);
    }
    if let Some(version) = installed_version(&state.workspace, catalog.root(), bare) {
        let path = catalog
            .gguf_local_path(&format!("{bare}:{version}"))
            .map_err(|e| e.to_string())?;
        return Ok(download::resolve_model_path(&state.workspace, &path));
    }
    if let Some(version) = installed_safetensors_version(&state.workspace, catalog.root(), bare) {
        return safetensors_bundle(&state.workspace, catalog.root(), bare, &version);
    }
    let path = catalog.gguf_local_path(model).map_err(|e| e.to_string())?;
    Ok(download::resolve_model_path(&state.workspace, &path))
}

fn safetensors_bundle(
    workspace: &std::path::Path,
    root: &serde_json::Value,
    model: &str,
    version: &str,
) -> Result<PathBuf, String> {
    let path = version_local_path(root, model, version)?;
    let resolved = download::resolve_model_path(workspace, &path);
    if danmo_engine::is_safetensors_image_bundle(&resolved)
        || danmo_engine::is_ace_step_safetensors_bundle(&resolved)
    {
        return Ok(resolved);
    }
    Err(format!(
        "{model}:{version} is not an installed safetensors bundle at {}",
        resolved.display()
    ))
}

fn version_local_path(
    root: &serde_json::Value,
    model: &str,
    version: &str,
) -> Result<String, String> {
    root.pointer(&format!(
        "/models/{model}/distribution/versions/{version}/local_path"
    ))
    .and_then(|value| value.as_str())
    .filter(|path| !path.is_empty())
    .map(|path| path.to_string())
    .ok_or_else(|| format!("{model}:{version} has no local_path"))
}

fn installed_safetensors_version(
    workspace: &std::path::Path,
    root: &serde_json::Value,
    model_id: &str,
) -> Option<String> {
    let versions = root
        .pointer(&format!("/models/{model_id}/distribution/versions"))?
        .as_object()?;
    let ready = |key: &str| {
        let path = versions
            .get(key)
            .and_then(|version| version.get("local_path"))
            .and_then(|value| value.as_str())
            .filter(|path| !path.is_empty())?;
        {
            let resolved = download::resolve_model_path(workspace, path);
            (danmo_engine::is_safetensors_image_bundle(&resolved)
                || danmo_engine::is_ace_step_safetensors_bundle(&resolved))
            .then_some(key.to_string())
        }
    };
    for key in danmo_catalog::SAFETENSORS_VERSION_KEYS {
        if versions
            .get(*key)
            .and_then(|version| version.get("default"))
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            if let Some(key) = ready(key) {
                return Some(key);
            }
        }
    }
    danmo_catalog::SAFETENSORS_VERSION_KEYS
        .iter()
        .find_map(|key| ready(key))
}

/// First tier whose directory is a ready GGUF bundle, in bf16 / q8_0 / q4_k order.
pub(crate) fn installed_version(
    workspace: &std::path::Path,
    root: &serde_json::Value,
    model_id: &str,
) -> Option<String> {
    let versions = root
        .pointer(&format!("/models/{model_id}/distribution/versions"))?
        .as_object()?;
    for key in danmo_catalog::GGUF_VERSION_KEYS {
        let Some(path) = versions
            .get(*key)
            .and_then(|v| v.get("local_path"))
            .and_then(|v| v.as_str())
            .filter(|path| !path.is_empty())
        else {
            continue;
        };
        if download::version_ready(&download::resolve_model_path(workspace, path)) {
            return Some((*key).to_string());
        }
    }
    None
}

#[cfg(any())]
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
        "backend": state.backend.name(),
        "backend_selection": state.backend_selection.reason,
        "backend_probe": state.backend_selection.probe.as_ref().map(|probe| probe.summary()),
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
    let projected = catalog
        .public_registry()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(projected))
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
    use axum::body::{Body, Bytes};
    use axum::http::{header, HeaderMap, Request, StatusCode};
    use axum::response::{IntoResponse, Response};
    use http_body_util::BodyExt;
    use std::io::{Read, Write};
    use tower::ServiceExt;

    fn router(db: &std::path::Path, registry: &std::path::Path) -> Router {
        try_router(db, registry).expect("test router")
    }

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

    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHA: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHA[(n >> 18) as usize & 63] as char);
            out.push(ALPHA[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHA[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHA[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    #[tokio::test]
    async fn qwen_image_edit_runs_end_to_end() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-edit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("qwen-image-edit", "q4_k");
        let bundle = dir.join(&local_path);
        danmo_engine::write_edit_smoke_bundle(&bundle).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let rgb = vec![128u8; 48 * 32 * 3];
        let png = danmo_engine::png::encode_rgb(48, 32, &rgb);
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "source.png",
                "data_base64": base64_encode(&png),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        let source_id = upload.1["id"].as_str().unwrap().to_string();
        let edit = post_json(
            &app,
            "/api/images/edits",
            &json!({
                "model": model,
                "operation": "rewrite",
                "source_asset_id": source_id,
                "title": "smoke edit",
                "prompt": "make it blue",
                "steps": 2,
                "guidance": 1.0,
                "seed": 1,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(edit.0, StatusCode::ACCEPTED, "{}", edit.1);
        let id = edit.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 48, "task {task}");
        assert_eq!(task["result"]["height"], 32, "task {task}");
        assert_eq!(task["result"]["source_asset_id"], json!(source_id));
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
        let bytes = assets.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let edited = listed["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == ids[0])
            .cloned()
            .unwrap_or(json!({}));
        assert_eq!(edited["parent_asset_id"], json!(source_id), "{edited}");
        assert_eq!(edited["relation_type"], "edit", "{edited}");
        let unbuilt = post_json(
            &app,
            "/api/images/edits",
            &json!({
                "model": "fibo",
                "operation": "rewrite",
                "source_asset_id": source_id,
                "prompt": "make it blue",
                "steps": 2
            }),
        )
        .await;
        assert_eq!(unbuilt.0, StatusCode::NOT_IMPLEMENTED, "{}", unbuilt.1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn rewrite_edit_runs_end_to_end(model: &str, write_bundle: fn(&std::path::Path)) {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-edit-{}-{}-{}",
            model.replace(':', "-"),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight(model, "q4_k");
        let bundle = dir.join(&local_path);
        write_bundle(&bundle);
        let app = router(&dir.join("studio.db"), &factory_registry());
        let rgb = vec![128u8; 32 * 32 * 3];
        let png = danmo_engine::png::encode_rgb(32, 32, &rgb);
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "source.png",
                "data_base64": base64_encode(&png),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        let source_id = upload.1["id"].as_str().unwrap().to_string();
        let edit = post_json(
            &app,
            "/api/images/edits",
            &json!({
                "model": model,
                "operation": "rewrite",
                "source_asset_id": source_id,
                "title": "smoke edit",
                "prompt": "make it blue",
                "steps": 4,
                "guidance": 1.0,
                "seed": 1,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(edit.0, StatusCode::ACCEPTED, "{}", edit.1);
        let id = edit.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 32, "task {task}");
        assert_eq!(task["result"]["height"], 32, "task {task}");
        assert_eq!(task["result"]["source_asset_id"], json!(source_id));
        assert_eq!(task["result"]["init_timestep"], json!(2), "task {task}");
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
        let bytes = assets.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let edited = listed["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == ids[0])
            .cloned()
            .unwrap_or(json!({}));
        assert_eq!(edited["parent_asset_id"], json!(source_id), "{edited}");
        assert_eq!(edited["relation_type"], "edit", "{edited}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn z_image_edit_fails_without_the_ggml_graph() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-zedit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut registry: Value =
            serde_json::from_str(&std::fs::read_to_string(factory_registry()).unwrap()).unwrap();
        let local = "models/Image/z-image/q4_k";
        registry["models"]["z-image"]["distribution"]["versions"]["q4_k"] = json!({
            "name": { "zh": "Q4_K", "en": "Q4_K" },
            "size": "1GB",
            "local_path": local
        });
        let registry_path = dir.join("registry.json");
        std::fs::write(&registry_path, serde_json::to_string(&registry).unwrap()).unwrap();
        danmo_engine::write_z_image_edit_smoke_bundle(&dir.join(local)).unwrap();
        let app = router(&dir.join("studio.db"), &registry_path);
        let model = "z-image:q4_k";
        let rgb = vec![128u8; 32 * 32 * 3];
        let png = danmo_engine::png::encode_rgb(32, 32, &rgb);
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "source.png",
                "data_base64": base64_encode(&png),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        let source_id = upload.1["id"].as_str().unwrap().to_string();
        let edit = post_json(
            &app,
            "/api/images/edits",
            &json!({
                "model": model,
                "operation": "rewrite",
                "source_asset_id": source_id,
                "title": "smoke edit",
                "prompt": "make it blue",
                "steps": 4,
                "guidance": 1.0,
                "seed": 1,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(edit.0, StatusCode::ACCEPTED, "{}", edit.1);
        let id = edit.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 50, 40).await;
        assert_eq!(task["status"], "failed", "task {task}");
        let err = task["error"].as_str().unwrap_or("");
        assert!(err.contains("no longer has a ggml graph"), "task {task}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn flux2_edit_runs_end_to_end() {
        rewrite_edit_runs_end_to_end("flux2-klein-4b", |bundle| {
            danmo_engine::write_flux2_edit_smoke_bundle(bundle).unwrap()
        })
        .await;
    }

    #[tokio::test]
    async fn seedvr2_upscale_runs_end_to_end() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-upscale-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("seedvr2-3b", "q4_k");
        let bundle = dir.join(&local_path);
        danmo_engine::write_seedvr2_smoke_bundle(&bundle).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let rgb = vec![128u8; 32 * 32 * 3];
        let png = danmo_engine::png::encode_rgb(32, 32, &rgb);
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "source.png",
                "data_base64": base64_encode(&png),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        let source_id = upload.1["id"].as_str().unwrap().to_string();
        let upscale = post_json(
            &app,
            "/api/images/upscales",
            &json!({
                "model": model,
                "source_asset_id": source_id,
                "scale": 2,
                "seed": 1,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(upscale.0, StatusCode::ACCEPTED, "{}", upscale.1);
        let id = upscale.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 64, "task {task}");
        assert_eq!(task["result"]["height"], 64, "task {task}");
        assert_eq!(task["result"]["source_asset_id"], json!(source_id));
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
        let bytes = assets.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let upscaled = listed["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == ids[0])
            .cloned()
            .unwrap_or(json!({}));
        assert_eq!(upscaled["parent_asset_id"], json!(source_id), "{upscaled}");
        assert_eq!(upscaled["relation_type"], "upscale", "{upscaled}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn seedvr2_video_upscale_runs_end_to_end() {
        let have_ffmpeg = ["ffmpeg", "ffprobe"].iter().all(|tool| {
            std::process::Command::new(tool)
                .arg("-version")
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        });
        if !have_ffmpeg {
            eprintln!("ffmpeg/ffprobe not installed; skipping video upscale e2e");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-video-upscale-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("seedvr2-video-3b", "q4_k");
        let bundle = dir.join(&local_path);
        danmo_engine::write_seedvr2_smoke_bundle(&bundle).unwrap();
        // Five 32x32 frames at 5 fps.
        let source_mp4 = dir.join("source.mp4");
        let rendered = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=32x32:rate=5:duration=1",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
            ])
            .arg(&source_mp4)
            .output()
            .expect("spawn ffmpeg");
        assert!(
            rendered.status.success(),
            "{}",
            String::from_utf8_lossy(&rendered.stderr)
        );
        let app = router(&dir.join("studio.db"), &factory_registry());
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "source.mp4",
                "data_base64": base64_encode(&std::fs::read(&source_mp4).unwrap()),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        assert_eq!(upload.1["kind"], "video", "{}", upload.1);
        let source_id = upload.1["id"].as_str().unwrap().to_string();
        let upscale = post_json(
            &app,
            "/api/videos/upscales",
            &json!({
                "model": model,
                "source_asset_id": source_id,
                "scale": 2,
                "temporal_window": 5,
                "max_frames": 10,
                "seed": 1,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(upscale.0, StatusCode::ACCEPTED, "{}", upscale.1);
        let id = upscale.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 600, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 64, "task {task}");
        assert_eq!(task["result"]["height"], 64, "task {task}");
        assert_eq!(task["result"]["frames"], 5, "task {task}");
        assert_eq!(task["result"]["source_asset_id"], json!(source_id));
        assert_eq!(task["result"]["scale"], 2, "task {task}");
        let denoise = task["result"]["denoise"].as_f64().unwrap_or(-1.0);
        assert!((denoise - 0.3).abs() < 1e-6, "task {task}");
        assert_eq!(task["result"]["chunk_frames"], 5, "task {task}");
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
        let bytes = assets.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let upscaled = listed["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == ids[0])
            .cloned()
            .unwrap_or(json!({}));
        assert_eq!(upscaled["kind"], "video", "{upscaled}");
        assert_eq!(upscaled["mime_type"], "video/mp4", "{upscaled}");
        assert_eq!(upscaled["parent_asset_id"], json!(source_id), "{upscaled}");
        assert_eq!(upscaled["relation_type"], "upscale", "{upscaled}");
        // ffprobe the persisted mp4: 64x64, 5 fps, 5 frames.
        let asset_id = ids[0].as_str().unwrap();
        let mp4 = dir
            .join("outputs")
            .join("assets")
            .join(format!("{asset_id}.mp4"));
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate",
                "-of",
                "json",
            ])
            .arg(&mp4)
            .output()
            .expect("spawn ffprobe");
        assert!(
            probe.status.success(),
            "{}",
            String::from_utf8_lossy(&probe.stderr)
        );
        let probe_json: Value = serde_json::from_slice(&probe.stdout).unwrap();
        let stream = &probe_json["streams"][0];
        assert_eq!(stream["width"], 64, "{probe_json}");
        assert_eq!(stream["height"], 64, "{probe_json}");
        assert_eq!(stream["avg_frame_rate"], "5/1", "{probe_json}");
        assert_eq!(stream["nb_frames"], "5", "{probe_json}");
        // A non-seedvr2 family fails loud before queueing.
        let unbuilt = post_json(
            &app,
            "/api/videos/upscales",
            &json!({
                "model": "hunyuan-video-1.5-1080p-sr",
                "source_asset_id": source_id,
                "scale": 2
            }),
        )
        .await;
        assert_eq!(unbuilt.0, StatusCode::NOT_IMPLEMENTED, "{}", unbuilt.1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn wan_video_create_runs_end_to_end() {
        let have_ffmpeg = ["ffmpeg", "ffprobe"].iter().all(|tool| {
            std::process::Command::new(tool)
                .arg("-version")
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        });
        if !have_ffmpeg {
            eprintln!("ffmpeg/ffprobe not installed; skipping wan t2v e2e");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-wan-t2v-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("wan-2.2-ti2v-5b", "q4_k");
        danmo_engine::write_wan_smoke_bundle(&dir.join(&local_path)).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let create = post_json(
            &app,
            "/api/videos/generations",
            &json!({
                "model": model,
                "prompt": "hello world",
                "size": "32x32",
                "num_frames": 5,
                "fps": 4,
                "steps": 2,
                "guidance": 2.0,
                "seed": 3,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(create.0, StatusCode::ACCEPTED, "{}", create.1);
        let id = create.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 32, "task {task}");
        assert_eq!(task["result"]["height"], 32, "task {task}");
        assert_eq!(task["result"]["frames"], 5, "task {task}");
        let ids = task["result"]["asset_ids"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(ids.len(), 1, "task {task}");
        let asset_id = ids[0].as_str().unwrap();
        let mp4 = dir
            .join("outputs")
            .join("assets")
            .join(format!("{asset_id}.mp4"));
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate",
                "-of",
                "json",
            ])
            .arg(&mp4)
            .output()
            .expect("spawn ffprobe");
        assert!(
            probe.status.success(),
            "{}",
            String::from_utf8_lossy(&probe.stderr)
        );
        let probe_json: Value = serde_json::from_slice(&probe.stdout).unwrap();
        let stream = &probe_json["streams"][0];
        assert_eq!(stream["width"], 32, "{probe_json}");
        assert_eq!(stream["height"], 32, "{probe_json}");
        assert_eq!(stream["avg_frame_rate"], "4/1", "{probe_json}");
        assert_eq!(stream["nb_frames"], "5", "{probe_json}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn hunyuan_video_edit_fails_loud_until_the_graphs_land() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-hunyuan-i2v-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let (model, local_path) = published_weight("hunyuan-video-1.5-i2v-720p", "q4_k");
        let missing_source = post_json(
            &app,
            "/api/videos/edits",
            &json!({
                "model": model,
                "prompt": "a cat",
            }),
        )
        .await;
        assert_eq!(
            missing_source.0,
            StatusCode::BAD_REQUEST,
            "{}",
            missing_source.1
        );
        let detail = missing_source.1["detail"]
            .as_str()
            .or_else(|| missing_source.1["raw"].as_str())
            .unwrap_or("");
        assert!(detail.contains("source_asset_id"), "{}", missing_source.1);
        // The family is wired, so a well-formed request queues; with no
        // installed bundle the task fails loud naming the registry path.
        let create = post_json(
            &app,
            "/api/videos/edits",
            &json!({
                "model": model,
                "source_asset_id": "ast_src",
                "prompt": "a cat",
                "num_frames": 81,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(create.0, StatusCode::ACCEPTED, "{}", create.1);
        let id = create.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "failed", "task {task}");
        let err = task["error"].as_str().unwrap_or("");
        assert!(
            err.contains(&local_path),
            "hunyuan i2v did not use the registry path for {model}: {err}"
        );
        // A non-hunyuan family fails loud before queueing.
        let unbuilt = post_json(
            &app,
            "/api/videos/edits",
            &json!({
                "model": "wan-2.2-ti2v-5b",
                "source_asset_id": "ast_src",
                "prompt": "a cat",
            }),
        )
        .await;
        assert_eq!(unbuilt.0, StatusCode::NOT_IMPLEMENTED, "{}", unbuilt.1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ltx_video_create_runs_end_to_end() {
        let have_ffmpeg = ["ffmpeg", "ffprobe"].iter().all(|tool| {
            std::process::Command::new(tool)
                .arg("-version")
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        });
        if !have_ffmpeg {
            eprintln!("ffmpeg/ffprobe not installed; skipping ltx t2v e2e");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-ltx-t2v-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, local_path) = published_weight("ltx-2.3-dev", "bf16");
        danmo_engine::write_ltx_smoke_bundle(&dir.join(&local_path)).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let create = post_json(
            &app,
            "/api/videos/generations",
            &json!({
                "model": model,
                "prompt": "a cat",
                // The smoke bundle ships a toy tokenizer with a tiny alphabet.
                "negative_prompt": "b",
                "size": "64x64",
                "num_frames": 9,
                "fps": 8,
                "steps": 2,
                "guidance": 2.0,
                "seed": 3,
                "priority": "normal"
            }),
        )
        .await;
        assert_eq!(create.0, StatusCode::ACCEPTED, "{}", create.1);
        let id = create.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "completed", "task {task}");
        assert_eq!(task["result"]["width"], 64, "task {task}");
        assert_eq!(task["result"]["height"], 64, "task {task}");
        assert_eq!(task["result"]["frames"], 9, "task {task}");
        assert_eq!(task["result"]["steps"], 5, "task {task}");
        let ids = task["result"]["asset_ids"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(ids.len(), 1, "task {task}");
        let asset_id = ids[0].as_str().unwrap();
        let mp4 = dir
            .join("outputs")
            .join("assets")
            .join(format!("{asset_id}.mp4"));
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,nb_frames,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
            ])
            .arg(&mp4)
            .output()
            .expect("spawn ffprobe");
        assert!(
            probe.status.success(),
            "{}",
            String::from_utf8_lossy(&probe.stderr)
        );
        let probe_json: Value = serde_json::from_slice(&probe.stdout).unwrap();
        let streams = probe_json["streams"].as_array().unwrap();
        let video = streams
            .iter()
            .find(|s| s["codec_type"] == "video")
            .unwrap_or_else(|| panic!("no video stream: {probe_json}"));
        assert_eq!(video["width"], 64, "{probe_json}");
        assert_eq!(video["height"], 64, "{probe_json}");
        assert_eq!(video["avg_frame_rate"], "8/1", "{probe_json}");
        assert_eq!(video["nb_frames"], "9", "{probe_json}");
        // LTX generates the 48 kHz stereo track alongside the video.
        let audio = streams
            .iter()
            .find(|s| s["codec_type"] == "audio")
            .unwrap_or_else(|| panic!("no audio stream: {probe_json}"));
        assert_eq!(audio["sample_rate"], "48000", "{probe_json}");
        assert_eq!(audio["channels"], 2, "{probe_json}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn z_image_controlnet_create_fails_without_the_ggml_graph() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-control-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // The factory registry leaves the union controlnet unpublished; the
        // test copy points its q4_k tier at the smoke adapter bundle.
        let mut registry: Value =
            serde_json::from_str(&std::fs::read_to_string(factory_registry()).unwrap()).unwrap();
        let control_local = "models/ControlNet/z-image-turbo-fun-controlnet-union/q4_k";
        let base_local = "models/Image/z-image/q4_k";
        registry["models"]["z-image-turbo-fun-controlnet-union"]["distribution"]["versions"] = json!({
            "q4_k": {
                "name": { "zh": "Q4_K", "en": "Q4_K" },
                "size": "1GB",
                "local_path": control_local
            }
        });
        registry["models"]["z-image"]["distribution"]["versions"]["q4_k"] = json!({
            "name": { "zh": "Q4_K", "en": "Q4_K" },
            "size": "1GB",
            "local_path": base_local
        });
        let registry_path = dir.join("registry.json");
        std::fs::write(&registry_path, serde_json::to_string(&registry).unwrap()).unwrap();
        let model = "z-image:q4_k";
        danmo_engine::write_z_image_control_smoke_bundle(
            &dir.join(&base_local),
            &dir.join(control_local),
        )
        .unwrap();
        let app = router(&dir.join("studio.db"), &registry_path);
        let rgb = vec![200u8; 64 * 64 * 3];
        let png = danmo_engine::png::encode_rgb(64, 64, &rgb);
        let upload = post_json(
            &app,
            "/api/assets",
            &json!({
                "filename": "control.png",
                "data_base64": base64_encode(&png),
            }),
        )
        .await;
        assert_eq!(upload.0, StatusCode::OK, "{}", upload.1);
        let control_id = upload.1["id"].as_str().unwrap().to_string();
        let mut body = vue_generate_body(&model, "a cat");
        body["structural_guide"] = json!({
            "asset_id": control_id,
            "model_id": "z-image-turbo-fun-controlnet-union:q4_k",
            "type": "canny",
            "weight": 0.8
        });
        let create = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(create.0, StatusCode::ACCEPTED, "{}", create.1);
        let id = create.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "failed", "task {task}");
        let err = task["error"].as_str().unwrap_or("");
        assert!(err.contains("no longer has a ggml graph"), "task {task}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn z_image_safetensors_bundle_resolves_past_the_gguf_catalog() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-mlx-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut registry: Value =
            serde_json::from_str(&std::fs::read_to_string(factory_registry()).unwrap()).unwrap();
        let local = "models/Image/z-image-turbo-fp16";
        registry["models"]["z-image-turbo"]["distribution"]["versions"]["fp16"]["local_path"] =
            json!(local);
        let bundle = dir.join(local);
        for role in ["transformer", "text_encoder", "vae"] {
            let role_dir = bundle.join(role);
            std::fs::create_dir_all(&role_dir).unwrap();
            std::fs::write(role_dir.join("model.safetensors"), b"not a real shard").unwrap();
        }
        let registry_path = dir.join("registry.json");
        std::fs::write(&registry_path, serde_json::to_string(&registry).unwrap()).unwrap();
        let app = router(&dir.join("studio.db"), &registry_path);
        let mut body = vue_generate_body("z-image-turbo", "a cat");
        body["guidance"] = json!(0);
        let create = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(create.0, StatusCode::ACCEPTED, "{}", create.1);
        let id = create.1["id"].as_str().unwrap();
        let task = wait_terminal_for(&app, id, 300, 100).await;
        assert_eq!(task["status"], "failed", "task {task}");
        let err = task["error"].as_str().unwrap_or("");
        assert!(
            !err.contains("no GGUF"),
            "safetensors bundle was rejected by the GGUF catalog: {task}"
        );
        assert!(
            err.contains("no longer has a ggml graph")
                || err.contains("safetensors")
                || err.contains("missing tensor")
                || err.contains("embed_tokens"),
            "task {task}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn assistant_endpoints_fail_loud_without_configuration() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-assistant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let info = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/llm/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(info.status(), StatusCode::OK);
        let bytes = info.into_body().collect().await.unwrap().to_bytes();
        let info: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(info["model_id"], "qwen3.5-4b", "{info}");
        assert_eq!(info["provider"], "builtin", "{info}");
        assert_eq!(info["available"], false, "{info}");
        assert_eq!(info["think_supported"], true, "{info}");
        let vision = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/vision/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(vision.status(), StatusCode::OK);
        let bytes = vision.into_body().collect().await.unwrap().to_bytes();
        let vision: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(vision["available"], false, "{vision}");
        let enhance = post_json(
            &app,
            "/api/chat/enhance-prompt",
            &json!({ "prompt": "a cat" }),
        )
        .await;
        assert_eq!(enhance.0, StatusCode::SERVICE_UNAVAILABLE, "{}", enhance.1);
        assert!(
            enhance.1["detail"]
                .as_str()
                .unwrap()
                .contains("not wired"),
            "{}",
            enhance.1
        );
        assert!(
            !enhance.1["detail"]
                .as_str()
                .unwrap()
                .contains("api.openai.com"),
            "builtin must not fall back to cloud: {}",
            enhance.1
        );
        let assistant = post_json(
            &app,
            "/api/assistant/run",
            &json!({ "kind": "lyrics", "prompt": "dreamy synth pop" }),
        )
        .await;
        assert_eq!(assistant.0, StatusCode::SERVICE_UNAVAILABLE, "{}", assistant.1);
        let put = put_json(
            &app,
            "/api/settings",
            &json!({ "llm_inference_provider": "openai_compatible" }),
        )
        .await;
        assert_eq!(put.0, StatusCode::OK, "{}", put.1);
        let info = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/llm/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = info.into_body().collect().await.unwrap().to_bytes();
        let info: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(info["provider"], "openai_compatible", "{info}");
        assert_eq!(info["available"], false, "{info}");
        let enhance = post_json(
            &app,
            "/api/chat/enhance-prompt",
            &json!({ "prompt": "a cat" }),
        )
        .await;
        assert_eq!(enhance.0, StatusCode::SERVICE_UNAVAILABLE, "{}", enhance.1);
        assert!(
            enhance.1["detail"]
                .as_str()
                .unwrap()
                .contains("not configured"),
            "{}",
            enhance.1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn assistant_cloud_product_features_roundtrip() {
        async fn mock_chat(headers: HeaderMap, body: Bytes) -> Response {
            let auth = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if auth != "Bearer test-key" {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({ "error": "bad key" })),
                )
                    .into_response();
            }
            let request: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(request["model"], "mock-model", "{request}");
            assert_eq!(request["stream"], false, "{request}");
            let system = request["messages"][0]["content"].as_str().unwrap_or("");
            let content = if system.contains("ACE-Step lyrics") {
                "[Verse 1]\nsoft neon rain\n\n[Chorus]\nkeep the beat alive"
            } else {
                "a vivid cat portrait, soft light"
            };
            Json(json!({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": content },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9 }
            }))
            .into_response()
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mock = Router::new().route("/v1/chat/completions", post(mock_chat));
        tokio::spawn(async move {
            axum::serve(listener, mock).await.unwrap();
        });
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-cloud-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let put = put_json(
            &app,
            "/api/settings",
            &json!({
                "llm_inference_provider": "openai_compatible",
                "llm_inference_base_url": format!("http://127.0.0.1:{port}"),
                "llm_inference_api_key": "test-key",
                "llm_inference_cloud_model": "mock-model"
            }),
        )
        .await;
        assert_eq!(put.0, StatusCode::OK, "{}", put.1);
        let info = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/llm/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = info.into_body().collect().await.unwrap().to_bytes();
        let info: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(info["available"], true, "{info}");
        assert_eq!(info["provider"], "openai_compatible", "{info}");
        let enhance = post_json(
            &app,
            "/api/chat/enhance-prompt",
            &json!({ "prompt": "a cat", "target_action": "image_create" }),
        )
        .await;
        assert_eq!(enhance.0, StatusCode::OK, "{}", enhance.1);
        assert_eq!(
            enhance.1["enhanced_prompt"], "a vivid cat portrait, soft light",
            "{}",
            enhance.1
        );
        let lyrics = post_json(
            &app,
            "/api/assistant/run",
            &json!({ "kind": "lyrics", "prompt": "dreamy synth pop" }),
        )
        .await;
        assert_eq!(lyrics.0, StatusCode::OK, "{}", lyrics.1);
        assert!(
            lyrics.1["text"]
                .as_str()
                .unwrap_or("")
                .contains("[Verse 1]"),
            "{}",
            lyrics.1
        );
        assert_eq!(lyrics.1["vision_used"], false, "{}", lyrics.1);
        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "messages": [{ "role": "user", "content": "hi" }]
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            missing.status(),
            StatusCode::NOT_FOUND,
            "public OpenAI proxy must stay removed"
        );
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
        let missing_file = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(missing_file.0, StatusCode::BAD_REQUEST);
        assert!(
            missing_file.1["detail"]
                .as_str()
                .unwrap()
                .contains("lora user-lora-abc is not installed"),
            "{}",
            missing_file.1
        );
        let trained = dir.join("trained.safetensors");
        std::fs::write(&trained, b"not-a-safetensors").unwrap();
        std::fs::write(
            config.join("user_loras.json"),
            format!(
                r#"{{"items":[{{"id":"user-lora-abc","base_model":"z-image-turbo","local_path":"{}","inference":{{"scheme":"sft","model":"z-image-turbo","extra_adapters":[{{"id":"companion-lora:bf16","weight":1.0}}]}}}}]}}"#,
                trained.display()
            ),
        )
        .unwrap();
        let stacked = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(stacked.0, StatusCode::BAD_REQUEST);
        assert!(
            stacked.1["detail"]
                .as_str()
                .unwrap()
                .contains("lora companion-lora:bf16 is not installed"),
            "{}",
            stacked.1
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
        let edit = post_json(
            &app,
            "/api/images/edits",
            &json!({
                "model": "fibo",
                "operation": "rewrite",
                "source_asset_id": "asset-missing",
                "prompt": "a cat",
                "steps": 2
            }),
        )
        .await;
        assert_eq!(edit.0, StatusCode::NOT_IMPLEMENTED, "{}", edit.1);
        assert!(
            edit.1["detail"].as_str().unwrap().contains("image.edit"),
            "{}",
            edit.1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn image_create_validates_structural_guide() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-guide-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let controlnet = "z-image-turbo-fun-controlnet-union";
        let mut body = vue_generate_body("z-image-turbo", "a cat");
        body["structural_guide"] = json!({ "asset_id": "asset-control" });
        let missing_model = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(missing_model.0, StatusCode::BAD_REQUEST);
        assert!(
            missing_model.1["detail"]
                .as_str()
                .unwrap()
                .contains("structural_guide.model_id"),
            "{}",
            missing_model.1
        );
        body["structural_guide"] = json!({
            "asset_id": "asset-control",
            "model_id": "z-image-turbo"
        });
        let not_controlnet = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(not_controlnet.0, StatusCode::BAD_REQUEST);
        assert!(
            not_controlnet.1["detail"]
                .as_str()
                .unwrap()
                .contains("is not a controlnet"),
            "{}",
            not_controlnet.1
        );
        body["structural_guide"] = json!({
            "asset_id": "asset-control",
            "model_id": controlnet,
            "type": "redux"
        });
        let redux = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(redux.0, StatusCode::BAD_REQUEST);
        assert!(
            redux.1["detail"].as_str().unwrap().contains("redux"),
            "{}",
            redux.1
        );
        body["structural_guide"] = json!({
            "asset_id": "asset-control",
            "model_id": controlnet,
            "inpaint_source_asset_id": "asset-src"
        });
        let half_pair = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(half_pair.0, StatusCode::BAD_REQUEST);
        assert!(
            half_pair.1["detail"]
                .as_str()
                .unwrap()
                .contains("inpaint_mask_asset_id"),
            "{}",
            half_pair.1
        );
        body["structural_guide"] = json!({
            "asset_id": "asset-control",
            "model_id": controlnet,
            "weight": 3.0
        });
        let heavy = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(heavy.0, StatusCode::BAD_REQUEST);
        body["model"] = json!("fibo");
        body["structural_guide"] = json!({
            "asset_id": "asset-control",
            "model_id": controlnet
        });
        let wrong_family = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(
            wrong_family.0,
            StatusCode::NOT_IMPLEMENTED,
            "{}",
            wrong_family.1
        );
        assert!(
            wrong_family.1["detail"]
                .as_str()
                .unwrap()
                .contains("structural guide graph is not built"),
            "{}",
            wrong_family.1
        );
        body["model"] = json!("z-image-turbo");
        let accepted = post_json(&app, "/api/images/generations", &body).await;
        assert_eq!(accepted.0, StatusCode::ACCEPTED, "{}", accepted.1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn audio_create_binds_installed_ace_loras() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-api-ace-lora-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let body = json!({
            "model": "ace-step-xl-sft",
            "prompt": "lofi",
            "lyrics": "[inst]",
            "instrumental": true,
            "vocal_type": "female",
            "n": 2,
            "audio_format": "mp3",
            "denoise": "mild",
            "duration": 5,
            "lm_expansion": "off",
            "dcw_enabled": false,
            "dcw_mode": "pix",
            "dcw_wavelet": "haar",
            "dcw_scaler": 0.1,
            "dcw_high_scaler": 0.02,
            "adapters": [{"id": "ace-style", "weight": 0.75}]
        });
        let missing = post_json(&app, "/api/audios/generations", &body).await;
        assert_eq!(missing.0, StatusCode::BAD_REQUEST);
        assert!(missing.1["detail"]
            .as_str()
            .unwrap_or("")
            .contains("not installed"));

        let config = dir.join("config");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(
            config.join("user_loras.json"),
            r#"{"items":[{"id":"ace-style","base_model":"ace-step-xl-sft","local_path":"models/Lora/ace-style.gguf"}]}"#,
        )
        .unwrap();
        let lora = dir.join("models/Lora/ace-style.gguf");
        std::fs::create_dir_all(lora.parent().unwrap()).unwrap();
        std::fs::write(&lora, b"test").unwrap();
        let accepted = post_json(&app, "/api/audios/generations", &body).await;
        assert_eq!(accepted.0, StatusCode::ACCEPTED, "{}", accepted.1);
        let edit = json!({
            "model": "ace-step-xl-sft",
            "prompt": "lofi",
            "task": "cover-nofsq",
            "source_latents": vec![0.0; 64],
            "duration": 8,
            "n": 2,
            "audio_format": "flac",
            "denoise": "strong",
            "dcw_enabled": true,
            "dcw_mode": "double",
            "adapters": [{"id": "ace-style", "weight": 0.5}]
        });
        let accepted_edit = post_json(&app, "/api/audios/edits", &edit).await;
        assert_eq!(accepted_edit.0, StatusCode::ACCEPTED, "{}", accepted_edit.1);
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

    async fn put_json(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
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
    async fn registry_route_projects_v3_models() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-registry-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let app = router(&dir.join("studio.db"), &factory_registry());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/registry")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let listed: Value = serde_json::from_slice(&bytes).unwrap();
        let turbo = &listed["models"]["z-image-turbo"];
        assert_eq!(turbo["media"], "image");
        assert_eq!(turbo["family"], "z_image");
        assert_eq!(turbo["name"]["zh"], "Z-Image-Turbo");
        assert_eq!(turbo["parameters"]["steps"]["default"], 9);
        assert_eq!(turbo["parameters"]["supports_guidance"], false);
        assert_eq!(
            turbo["versions"]["mlx-q4"]["repo_id"],
            "OsaurusAI/Z-Image-Turbo-mflux-4bit"
        );
        assert!(turbo.get("catalog").is_none());
        assert_eq!(
            listed["_index"]["z-image-turbo"]["actions"],
            json!(["edit", "generate"])
        );
        assert!(listed["categories"]["base_models"].is_object());
        let _ = std::fs::remove_dir_all(&dir);
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
                                "url": format!("http://127.0.0.1:{port}/model.safetensors"),
                                "source": "huggingface"
                            },
                            "q4_k": {
                                "local_path": "models/demo-http/q4",
                                "url": format!("http://127.0.0.1:{port}/model.gguf"),
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
        assert_eq!(
            std::fs::read(bundle.join("model.safetensors")).unwrap(),
            payload
        );

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
        assert!(!bundle.join("model.safetensors").exists());

        let batch = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/models/install-batch")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model_ids":["demo-gguf"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch.status(), StatusCode::OK);
        let bytes = batch.into_body().collect().await.unwrap().to_bytes();
        let batch: Value = serde_json::from_slice(&bytes).unwrap();
        let batch_id = batch["task_ids"][0].as_str().unwrap().to_string();
        let mut batch_status = String::new();
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
                .find(|task| task["id"] == batch_id)
                .unwrap();
            batch_status = task["status"].as_str().unwrap_or("").to_string();
            if matches!(batch_status.as_str(), "completed" | "failed" | "cancelled") {
                break;
            }
        }
        assert_eq!(batch_status, "completed");
        assert_eq!(
            std::fs::read(bundle.join("model.safetensors")).unwrap(),
            payload
        );
        let removed_batch = app
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
        assert_eq!(removed_batch.status(), StatusCode::OK);

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
                panic!("gguf URL was downloaded: {task}");
            }
        }
        assert!(refused_error.contains("safetensors"), "{refused_error}");
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
    let action = action_for_kind(kind.as_str()).ok_or_else(|| {
        (
            StatusCode::NOT_IMPLEMENTED,
            format!("unsupported task kind {}", kind.as_str()),
        )
    })?;
    danmo_engine::ensure_available(&entry.family, action)
        .map_err(|error| (StatusCode::NOT_IMPLEMENTED, error.to_string()))?;
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

fn action_for_kind(kind: &str) -> Option<danmo_engine::Action> {
    use danmo_engine::Action;
    Some(match kind {
        "image.generation" => Action::ImageCreate,
        "image.edit" => Action::ImageEdit,
        "image.upscale" => Action::ImageUpscale,
        "video.generation" => Action::VideoCreate,
        "video.edit" => Action::VideoEdit,
        "video.upscale" => Action::VideoUpscale,
        "video.avatar" => Action::VideoAvatar,
        "audio.generation" => Action::AudioCreate,
        "audio.edit" => Action::AudioEdit,
        "audio.lm" => Action::AudioLm,
        _ => return None,
    })
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
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let request = prepare_image_edit_request(&state, body)?;
    let accepted = accept(
        &state,
        TaskKind::ImageEdit,
        &request.model,
        &request,
        &request.priority,
    )
    .map_err(|(status, detail)| fail(status, detail))?;
    Ok((StatusCode::ACCEPTED, accepted))
}

fn random_seed() -> i64 {
    (Uuid::new_v4().as_u128() % 2_147_483_647) as i64
}

fn ensure_seed(seed: &mut Option<i64>) {
    if seed.is_none() {
        *seed = Some(random_seed());
    }
}

fn prepare_image_edit_request(
    state: &AppState,
    mut body: Value,
) -> Result<ImageEditRequest, ApiError> {
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
    reject_unbuilt_inference(&entry.family, &body, false)?;
    bind_installed_loras(state, &model, &mut body)?;
    fill_registry_defaults(&catalog, &entry.id, &mut body);
    let mut request: ImageEditRequest = serde_json::from_value(body)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    if request.n == 0 {
        return Err(fail(StatusCode::BAD_REQUEST, "n must be at least 1"));
    }
    ensure_seed(&mut request.seed);
    Ok(request)
}

async fn image_upscales(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let request = prepare_image_upscale_request(&state, body)?;
    let accepted = accept(
        &state,
        TaskKind::ImageUpscale,
        &request.model,
        &request,
        &request.priority,
    )
    .map_err(|(status, detail)| fail(status, detail))?;
    Ok((StatusCode::ACCEPTED, accepted))
}

fn prepare_image_upscale_request(
    state: &AppState,
    body: Value,
) -> Result<ImageUpscaleRequest, ApiError> {
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
    catalog
        .model(&model)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    let request: ImageUpscaleRequest = serde_json::from_value(body)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    if request.source_asset_id.trim().is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "source_asset_id is required"));
    }
    Ok(request)
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
    reject_unbuilt_inference(&entry.family, &body, true)?;
    bind_installed_loras(state, &model, &mut body)?;
    fill_registry_defaults(&catalog, &entry.id, &mut body);
    let mut request: ImageGenerationRequest = serde_json::from_value(body)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    if request.n == 0 {
        return Err(fail(StatusCode::BAD_REQUEST, "n must be at least 1"));
    }
    ensure_seed(&mut request.seed);
    if let Some(guide) = &request.structural_guide {
        validate_structural_guide(&catalog, &entry, guide)?;
    }
    Ok(request)
}

fn validate_structural_guide(
    catalog: &Catalog,
    entry: &danmo_catalog::ModelEntry,
    guide: &danmo_contracts::StructuralGuide,
) -> Result<(), ApiError> {
    if entry.family != "z_image" {
        return Err(fail(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "structural guide graph is not built for family {}",
                entry.family
            ),
        ));
    }
    if guide.asset_id.trim().is_empty() {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "structural_guide.asset_id is required",
        ));
    }
    if guide.model_id.trim().is_empty() {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "structural_guide.model_id is required",
        ));
    }
    let controlnet = catalog
        .model(&guide.model_id)
        .map_err(|err| fail(StatusCode::BAD_REQUEST, err.to_string()))?;
    let category = catalog
        .root()
        .pointer(&format!("/models/{}/catalog/category", controlnet.id))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if category != "controlnets" {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!(
                "structural_guide.model_id {} is not a controlnet",
                controlnet.id
            ),
        ));
    }
    match guide.guide_type.as_str() {
        "canny" | "depth" | "pose" | "hed" | "mlsd" | "scribble" | "gray" | "auto" => {}
        "redux" => {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                "redux structural_guide is FLUX-only; use a control image",
            ));
        }
        other => {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("structural_guide.type {other} is not supported"),
            ));
        }
    }
    if !(0.0..=2.0).contains(&guide.weight) {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "structural_guide.weight must be between 0 and 2",
        ));
    }
    let has_source = guide
        .inpaint_source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let has_mask = guide
        .inpaint_mask_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    if has_source != has_mask {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            "structural_guide.inpaint_source_asset_id and inpaint_mask_asset_id must both be set for inpaint mode",
        ));
    }
    Ok(())
}

fn prepare_audio_request<T: serde::de::DeserializeOwned>(
    state: &AppState,
    mut body: Value,
) -> Result<T, ApiError> {
    let model = body
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    if model.is_empty() {
        return Err(fail(StatusCode::BAD_REQUEST, "model is required"));
    }
    let catalog = Catalog::open(&state.registry)
        .map_err(|error| fail(StatusCode::BAD_REQUEST, error.to_string()))?;
    let entry = catalog
        .model(&model)
        .map_err(|error| fail(StatusCode::BAD_REQUEST, error.to_string()))?;
    reject_unbuilt_inference(&entry.family, &body, false)?;
    bind_installed_loras(state, &model, &mut body)?;
    serde_json::from_value(body).map_err(|error| fail(StatusCode::BAD_REQUEST, error.to_string()))
}

fn reject_unbuilt_inference(family: &str, body: &Value, allow_guide: bool) -> Result<(), ApiError> {
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
    if !allow_guide
        && body
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

fn reject_foreign_lora(state: &AppState, model: &str, id: &str) -> Result<(), ApiError> {
    let bare = id.split(':').next().unwrap_or(id);
    let catalog = danmo_catalog::Catalog::open(&state.registry)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let Some(entry) = catalog.root().pointer(&format!("/models/{bare}")) else {
        return Ok(());
    };
    let lora_card = entry
        .pointer("/catalog/category")
        .and_then(|value| value.as_str())
        == Some("loras")
        || entry
            .pointer("/catalog/type")
            .and_then(|value| value.as_str())
            == Some("lora");
    if !lora_card {
        return Ok(());
    }
    let base = entry
        .pointer("/catalog/base_model")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if product::lora_compatible(model, base) {
        return Ok(());
    }
    Err(fail(
        StatusCode::BAD_REQUEST,
        format!("lora {id} is for {base}"),
    ))
}

fn installed_lora_path(state: &AppState, id: &str) -> Result<PathBuf, ApiError> {
    if let Some(path) = state
        .downloads
        .lora_path(id)
        .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?
    {
        return Ok(path);
    }
    let (bare, version) = match id.split_once(':') {
        Some((bare, version)) => (bare, Some(version)),
        None => (id, None),
    };
    let catalog = danmo_catalog::Catalog::open(&state.registry)
        .map_err(|err| fail(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let Some(entry) = catalog.root().pointer(&format!("/models/{bare}")) else {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("lora {id} is not installed"),
        ));
    };
    let lora_card = entry
        .pointer("/catalog/category")
        .and_then(|value| value.as_str())
        == Some("loras")
        || entry
            .pointer("/catalog/type")
            .and_then(|value| value.as_str())
            == Some("lora");
    if !lora_card {
        return Err(fail(
            StatusCode::BAD_REQUEST,
            format!("lora {id} is not a LoRA"),
        ));
    }
    match download::registry_lora_weight(&state.workspace, entry, version)
        .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?
    {
        Some(path) => Ok(path),
        None => Err(fail(
            StatusCode::BAD_REQUEST,
            format!("lora {id} is not installed"),
        )),
    }
}

fn bind_installed_loras(state: &AppState, model: &str, body: &mut Value) -> Result<(), ApiError> {
    let Some(rows) = body.get_mut("adapters").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    if rows.is_empty() {
        return Ok(());
    }
    let bare = model.split(':').next().unwrap_or(model);
    let mut order: Vec<(String, f32)> = Vec::new();
    for row in rows.iter() {
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if id.is_empty() {
            return Err(fail(StatusCode::BAD_REQUEST, "lora id is required"));
        }
        let weight = row.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        if !weight.is_finite() || !(0.0..=2.0).contains(&weight) {
            return Err(fail(
                StatusCode::BAD_REQUEST,
                format!("lora {id} weight is not between 0 and 2"),
            ));
        }
        order.push((id, weight));
    }
    let requested = order.clone();
    for (id, _) in &requested {
        let extras = state
            .downloads
            .inference_stack(id)
            .map_err(|detail| fail(StatusCode::BAD_REQUEST, detail))?;
        for extra in extras {
            if order.iter().any(|(have, _)| have == &extra.id) {
                continue;
            }
            order.push((extra.id, extra.weight));
        }
    }
    let mut bound = Vec::with_capacity(order.len());
    for (id, weight) in order {
        reject_foreign_lora(state, bare, &id)?;
        let path = installed_lora_path(state, &id)?;
        bound.push(json!({
            "id": id,
            "weight": weight,
            "path": path.to_string_lossy().to_string(),
        }));
    }
    *rows = bound;
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
    let mut body = body;
    ensure_seed(&mut body.seed);
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
    let mut request: VideoEditRequest = serde_json::from_value(body)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    ensure_seed(&mut request.seed);
    if request.source_asset_id.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "source_asset_id is required".into(),
        ));
    }
    let accepted = accept(
        &state,
        TaskKind::VideoEdit,
        &request.model,
        &request,
        &request.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn video_upscales(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let mut request: VideoUpscaleRequest = serde_json::from_value(body)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    ensure_seed(&mut request.seed);
    if request.source_asset_id.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "source_asset_id is required".into(),
        ));
    }
    if let Some(scale) = request.scale {
        if scale != 2 && scale != 4 {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("seedvr2 video upscale scale must be 2 or 4, got {scale}"),
            ));
        }
    }
    let accepted = accept(
        &state,
        TaskKind::VideoUpscale,
        &request.model,
        &request,
        &request.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
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
    Json(mut body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    if let Some(obj) = body.as_object_mut() {
        let task_missing = obj.get("task").map(|value| value.is_null()).unwrap_or(true);
        if task_missing && obj.get("operation").and_then(|value| value.as_str()) == Some("cover") {
            obj.insert("task".into(), json!("cover"));
        }
    }
    let mut body: AudioEditRequest = prepare_audio_request(&state, body)?;
    ensure_seed(&mut body.seed);
    let accepted = accept(
        &state,
        TaskKind::AudioEdit,
        &body.model,
        &body,
        &body.priority,
    )
    .map_err(|(status, detail)| fail(status, detail))?;
    Ok((StatusCode::ACCEPTED, accepted))
}

async fn audio_lm_route(
    State(state): State<AppState>,
    Json(body): Json<AudioLmRequest>,
) -> Result<(StatusCode, Json<TaskAccepted>), (StatusCode, String)> {
    let accepted = accept(
        &state,
        TaskKind::AudioLm,
        &body.model,
        &body,
        &body.priority,
    )?;
    Ok((StatusCode::ACCEPTED, accepted))
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
    let mut body = body.clone();
    if matches!(body.get("seed"), None | Some(Value::Null)) {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("seed".into(), json!(random_seed()));
        }
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
    let action = action_for_kind(kind).ok_or_else(|| {
        (
            StatusCode::NOT_IMPLEMENTED,
            format!("unsupported task kind {kind}"),
        )
    })?;
    danmo_engine::ensure_available(&entry.family, action)
        .map_err(|error| (StatusCode::NOT_IMPLEMENTED, error.to_string()))?;
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
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let mut body: AudioGenerationRequest = prepare_audio_request(&state, body)?;
    ensure_seed(&mut body.seed);
    let accepted = accept(
        &state,
        TaskKind::AudioGeneration,
        &body.model,
        &body,
        &body.priority,
    )
    .map_err(|(status, detail)| fail(status, detail))?;
    Ok((StatusCode::ACCEPTED, accepted))
}
