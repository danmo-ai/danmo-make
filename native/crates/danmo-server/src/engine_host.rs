use crate::app::{gguf_bundle, AppState};
use crate::product;
use danmo_contracts::{
    AudioEditRequest, AudioGenerationRequest, AudioLmRequest, ImageEditRequest,
    ImageGenerationRequest, ImageUpscaleRequest, VideoEditRequest, VideoGenerationRequest,
    VideoUpscaleRequest,
};
use danmo_engine::{
    Action, Artifact, ArtifactSink, CancellationToken, Engine, EngineAudioCreateRequest,
    EngineAudioEditRequest, EngineAudioLmRequest, EngineError, EngineRequest, ExecutionContext,
    PersistedAsset, ResolvePort, ResolvedRun, RunObserver,
};
use danmo_store::AssetRow;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct EngineHost<'a> {
    state: &'a AppState,
}

impl<'a> EngineHost<'a> {
    pub fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl ResolvePort for EngineHost<'_> {
    fn resolve(&self, model: &str, action: Action) -> Result<ResolvedRun, EngineError> {
        let catalog = danmo_catalog::Catalog::open(&self.state.registry)
            .map_err(|error| EngineError::Resolve(error.to_string()))?;
        let entry = catalog
            .model(model)
            .map_err(|error| EngineError::Resolve(error.to_string()))?;
        let bundle = gguf_bundle(self.state, model).map_err(EngineError::Resolve)?;
        Ok(ResolvedRun {
            model: model.into(),
            version: model
                .split_once(':')
                .map(|(_, version)| version.to_string()),
            family: entry.family,
            action,
            bundle,
            backend: self.state.backend,
        })
    }

    fn resolve_component(
        &self,
        _run: &ResolvedRun,
        role: &str,
        model: &str,
    ) -> Result<PathBuf, EngineError> {
        if role.trim().is_empty() {
            return Err(EngineError::Resolve("component role is required".into()));
        }
        gguf_bundle(self.state, model).map_err(EngineError::Resolve)
    }

    fn read_source_audio(&self, _run: &ResolvedRun, source: &str) -> Result<Vec<u8>, EngineError> {
        if let Some(encoded) = source.strip_prefix("base64:") {
            return product::base64_decode(encoded).map_err(EngineError::Resolve);
        }
        if let Some(asset_id) = source.strip_prefix("asset:") {
            let row = self
                .state
                .scheduler
                .get_asset(asset_id)
                .map_err(|error| EngineError::Resolve(error.to_string()))?
                .ok_or_else(|| {
                    EngineError::Resolve(format!("source asset {asset_id} is missing"))
                })?;
            let path = product::resolve_asset_fs_path(
                &self.state.outputs,
                &self.state.workspace,
                &row.file_path,
            );
            return std::fs::read(&path).map_err(|error| {
                EngineError::Resolve(format!("source audio {}: {error}", path.display()))
            });
        }
        let path = Path::new(source);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.state.workspace.join(path)
        };
        std::fs::read(&resolved).map_err(|error| {
            EngineError::Resolve(format!("source audio {}: {error}", resolved.display()))
        })
    }

    fn read_source_image(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<(Vec<u8>, u32, u32), EngineError> {
        let row = self
            .state
            .scheduler
            .get_asset(source_asset_id)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
            .ok_or_else(|| {
                EngineError::Resolve(format!("source asset {source_asset_id} is missing"))
            })?;
        let path = product::resolve_asset_fs_path(
            &self.state.outputs,
            &self.state.workspace,
            &row.file_path,
        );
        let bytes = std::fs::read(&path).map_err(|error| {
            EngineError::Resolve(format!("source image {}: {error}", path.display()))
        })?;
        if let Some((rgb, width, height)) = danmo_engine::png::decode_rgb(&bytes) {
            return Ok((rgb, width, height));
        }
        decode_image_with_ffmpeg(&path)
    }

    fn read_source_video_path(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<PathBuf, EngineError> {
        let row = self
            .state
            .scheduler
            .get_asset(source_asset_id)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
            .ok_or_else(|| {
                EngineError::Resolve(format!("source asset {source_asset_id} is missing"))
            })?;
        Ok(product::resolve_asset_fs_path(
            &self.state.outputs,
            &self.state.workspace,
            &row.file_path,
        ))
    }
}

/// Generic image decode through the ffmpeg CLI for uploads that are not the
/// stored-deflate PNGs this server writes (JPEG, compressed PNG, WebP).
fn decode_image_with_ffmpeg(path: &Path) -> Result<(Vec<u8>, u32, u32), EngineError> {
    let ffprobe = std::env::var_os("DANQING_FFPROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffprobe"));
    let probe = Command::new(&ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|error| {
            EngineError::Resolve(format!(
                "image decode needs ffprobe on PATH for {}: {error}",
                path.display()
            ))
        })?;
    if !probe.status.success() {
        return Err(EngineError::Resolve(format!(
            "ffprobe could not read {}: {}",
            path.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        )));
    }
    let text = String::from_utf8_lossy(&probe.stdout);
    let line = text.lines().next().unwrap_or("");
    let (width, height) = line
        .split_once(',')
        .and_then(|(w, h)| {
            w.trim()
                .parse::<u32>()
                .ok()
                .zip(h.trim().parse::<u32>().ok())
        })
        .filter(|(w, h)| *w > 0 && *h > 0)
        .ok_or_else(|| {
            EngineError::Resolve(format!(
                "ffprobe returned no video dimensions for {}",
                path.display()
            ))
        })?;
    let ffmpeg = std::env::var_os("DANQING_FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let decoded = Command::new(&ffmpeg)
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .map_err(|error| {
            EngineError::Resolve(format!(
                "image decode needs ffmpeg on PATH for {}: {error}",
                path.display()
            ))
        })?;
    if !decoded.status.success() {
        return Err(EngineError::Resolve(format!(
            "ffmpeg could not decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&decoded.stderr).trim()
        )));
    }
    let expect = width as usize * height as usize * 3;
    if decoded.stdout.len() != expect {
        return Err(EngineError::Resolve(format!(
            "ffmpeg decoded {} bytes for {}x{} rgb24, expected {expect}",
            decoded.stdout.len(),
            width,
            height
        )));
    }
    Ok((decoded.stdout, width, height))
}

impl ArtifactSink for EngineHost<'_> {
    fn persist(
        &self,
        run: &ResolvedRun,
        task_id: &str,
        artifacts: Vec<Artifact>,
    ) -> Result<Vec<PersistedAsset>, EngineError> {
        let mut persisted = Vec::new();
        for artifact in artifacts {
            let id = product::new_id("ast_");
            let asset_dir = self.state.outputs.join("assets");
            std::fs::create_dir_all(&asset_dir)
                .map_err(|error| EngineError::Persist(error.to_string()))?;
            let path = asset_dir.join(format!("{id}.{}", artifact.extension));
            atomic_write(&path, &artifact.bytes)?;
            if artifact.kind == "image" {
                let preview_dir = self.state.outputs.join("previews");
                std::fs::create_dir_all(&preview_dir)
                    .map_err(|error| EngineError::Persist(error.to_string()))?;
                atomic_write(&preview_dir.join(format!("{task_id}.png")), &artifact.bytes)?;
            }
            for sidecar in &artifact.sidecars {
                let sidecar_path = asset_dir.join(format!("{id}{}", sidecar.suffix));
                atomic_write(&sidecar_path, &sidecar.bytes)?;
            }
            let mut metadata = artifact.metadata.clone();
            metadata["model"] = json!(run.model);
            metadata["family"] = json!(run.family);
            self.state
                .scheduler
                .insert_asset(&AssetRow {
                    id: id.clone(),
                    kind: artifact.kind,
                    mime_type: artifact.mime,
                    file_path: path.display().to_string(),
                    thumbnail_path: None,
                    width: artifact.width,
                    height: artifact.height,
                    duration_seconds: artifact.duration_seconds,
                    source_task_id: Some(task_id.to_string()),
                    source_action: Some(artifact.source_action),
                    metadata: metadata.to_string(),
                    created_at: String::new(),
                    parent_asset_id: artifact.parent_asset_id,
                    relation_type: artifact.relation_type,
                    group_id: None,
                })
                .map_err(|error| EngineError::Persist(error.to_string()))?;
            persisted.push(PersistedAsset { id, path });
        }
        Ok(persisted)
    }
}

struct TaskObserver {
    scheduler: danmo_schedule::Scheduler,
    task_id: String,
    backend: danmo_engine::BackendKind,
    preview_dir: PathBuf,
    preview: bool,
}

impl RunObserver for TaskObserver {
    fn progress(&self, phase: &str, step: u32, total: u32) -> Result<(), EngineError> {
        let progress = if total == 0 {
            0.0
        } else {
            step as f64 / total as f64
        };
        let message = if phase.is_empty() {
            format!("[{}] {step}/{total}", self.backend.name())
        } else {
            format!("[{}] {phase} {step}/{total}", self.backend.name())
        };
        self.scheduler
            .progress(&self.task_id, progress, &message)
            .map_err(|error| EngineError::Observer(error.to_string()))
    }

    fn log(&self, level: &str, message: &str) -> Result<(), EngineError> {
        self.scheduler
            .log(
                &self.task_id,
                level,
                format!("[{}] {message}", self.backend.name()).as_str(),
            )
            .map_err(|error| EngineError::Observer(error.to_string()))
    }

    fn phase(&self, phase: &str) -> Result<(), EngineError> {
        self.log("info", &format!("phase {phase}"))
    }

    fn preview_interval_steps(&self) -> Option<u32> {
        if self.preview && !self.task_id.is_empty() {
            Some(2)
        } else {
            None
        }
    }

    fn preview_max_edge(&self) -> u32 {
        512
    }

    fn write_step_preview(&self, png: &[u8]) -> Result<(), EngineError> {
        if png.len() < 8 || &png[..8] != b"\x89PNG\r\n\x1a\n" {
            return Err(EngineError::Observer("step preview is not a PNG".into()));
        }
        std::fs::create_dir_all(&self.preview_dir)
            .map_err(|error| EngineError::Observer(error.to_string()))?;
        atomic_write(&self.preview_dir.join(format!("{}.png", self.task_id)), png)
    }
}

pub fn execute(
    state: &AppState,
    kind: &str,
    model: &str,
    params: &Value,
    cancellation: CancellationToken,
) -> Result<Value, EngineError> {
    let task_id = params.get("task_id").and_then(Value::as_str).unwrap_or("");
    let host = EngineHost::new(state);
    let request = engine_request(kind, model, params)?;
    let engine = Engine::new(&host, &host, state.backend);
    let observer = std::sync::Arc::new(TaskObserver {
        scheduler: state.scheduler.clone(),
        task_id: task_id.into(),
        backend: state.backend,
        preview_dir: state.outputs.join("previews"),
        preview: request_preview(params),
    });
    let exec = ExecutionContext::new(task_id, observer).with_cancellation(cancellation);
    engine
        .execute(request, exec)
        .map(|result| result.task_json())
}

fn request_preview(params: &Value) -> bool {
    match params.pointer("/request/preview") {
        Some(value) => value.as_bool().unwrap_or(true),
        None => true,
    }
}

fn engine_request(kind: &str, model: &str, params: &Value) -> Result<EngineRequest, EngineError> {
    let request = params.get("request").cloned().unwrap_or_else(|| json!({}));
    match kind {
        "image.generation" => serde_json::from_value::<ImageGenerationRequest>(request)
            .map(EngineRequest::ImageCreate)
            .map_err(|error| EngineError::bad_request(error.to_string())),
        "audio.generation" => {
            let request: AudioGenerationRequest = serde_json::from_value(request)
                .map_err(|error| EngineError::bad_request(error.to_string()))?;
            Ok(EngineRequest::AudioCreate(EngineAudioCreateRequest {
                model: request.model,
                prompt: request.prompt,
                instrumental: request.instrumental,
                vocal_type: request.vocal_type,
                n: request.n,
                metadata: request.metadata.clone(),
                lyrics: request.lyrics,
                duration_sec: request.duration.unwrap_or(30.0),
                seed: request.seed,
                guidance: request.guidance.unwrap_or(1.0),
                steps: request.steps,
                vocal_language: request.vocal_language,
                adapters: request.adapters,
                family_options: json!({
                    "bpm": request.bpm,
                    "keyscale": request.key_scale,
                    "timesignature": request.time_signature,
                    "lm_expansion": request.lm_expansion,
                    "audio_code_indices": request.audio_code_indices,
                    "timbre_latents": request.timbre_latents,
                    "audio_format": request.audio_format,
                    "denoise": request.denoise,
                    "dcw_enabled": request.dcw_enabled.or(metadata_bool(&request.metadata, "dcw_enabled")?),
                    "dcw_mode": request.dcw_mode.or(metadata_text(&request.metadata, "dcw_mode")?).unwrap_or_else(|| "double".into()),
                    "dcw_wavelet": request.dcw_wavelet.or(metadata_text(&request.metadata, "dcw_wavelet")?).unwrap_or_else(|| "haar".into()),
                    "dcw_scaler": request.dcw_scaler.or(metadata_float(&request.metadata, "dcw_scaler")?).unwrap_or(0.05),
                    "dcw_high_scaler": request.dcw_high_scaler.or(metadata_float(&request.metadata, "dcw_high_scaler")?).unwrap_or(0.02),
                }),
            }))
        }
        "audio.edit" => {
            let request: AudioEditRequest = serde_json::from_value(request)
                .map_err(|error| EngineError::bad_request(error.to_string()))?;
            Ok(EngineRequest::AudioEdit(EngineAudioEditRequest {
                model: request.model,
                prompt: request.prompt,
                instrumental: request.instrumental,
                vocal_type: request.vocal_type,
                n: request.n,
                metadata: request.metadata.clone(),
                duration_sec: request.duration,
                bpm: request.bpm,
                keyscale: request.key_scale,
                timesignature: request.time_signature,
                lyrics: request.lyrics,
                seed: request.seed,
                guidance: request.guidance.unwrap_or(1.0),
                steps: request.steps,
                vocal_language: request.vocal_language,
                source_audio: request.source_audio.clone().or_else(|| {
                    request
                        .source_asset_id
                        .as_ref()
                        .filter(|id| !id.trim().is_empty())
                        .map(|id| format!("asset:{id}"))
                }),
                source_audio_base64: request.source_audio_base64,
                adapters: request.adapters,
                family_options: json!({
                    "task": request.task,
                    "source_latents": request.source_latents,
                    "audio_cover_strength": request
                        .audio_cover_strength
                        .or(request.source_fidelity)
                        .unwrap_or(1.0),
                    "cover_noise_strength": request.cover_noise_strength.unwrap_or(0.0),
                    "region_start_sec": request.region_start_sec,
                    "region_end_sec": request.region_end_sec,
                    "repaint_injection_ratio": request.repaint_injection_ratio.unwrap_or(1.0),
                    "repaint_crossfade_frames": request.repaint_crossfade_frames.unwrap_or(8),
                    "timbre_from_source": request.timbre_from_source.or(metadata_bool(&request.metadata, "cover_timbre_from_source")?).unwrap_or(false),
                    "track": request.track.unwrap_or_default(),
                    "audio_format": request.audio_format,
                    "denoise": request.denoise,
                    "dcw_enabled": request.dcw_enabled.or(metadata_bool(&request.metadata, "dcw_enabled")?),
                    "dcw_mode": request.dcw_mode.or(metadata_text(&request.metadata, "dcw_mode")?).unwrap_or_else(|| "double".into()),
                    "dcw_wavelet": request.dcw_wavelet.or(metadata_text(&request.metadata, "dcw_wavelet")?).unwrap_or_else(|| "haar".into()),
                    "dcw_scaler": request.dcw_scaler.or(metadata_float(&request.metadata, "dcw_scaler")?).unwrap_or(0.05),
                    "dcw_high_scaler": request.dcw_high_scaler.or(metadata_float(&request.metadata, "dcw_high_scaler")?).unwrap_or(0.02),
                }),
            }))
        }
        "audio.lm" => {
            let request: AudioLmRequest = serde_json::from_value(request)
                .map_err(|error| EngineError::bad_request(error.to_string()))?;
            Ok(EngineRequest::AudioLm(EngineAudioLmRequest {
                model: request.model,
                mode: request.mode,
                prompt: request.prompt,
                lyrics: request.lyrics,
                family_options: json!({
                    "duration": request.duration.unwrap_or(0.0),
                    "bpm": request.bpm.unwrap_or(0),
                    "keyscale": request.keyscale,
                    "timesignature": request.timesignature,
                    "vocal_language": if request.vocal_language.trim().is_empty() { "en".into() } else { request.vocal_language },
                    "max_new_tokens": request.max_new_tokens.unwrap_or(256),
                    "temperature": request.temperature.unwrap_or(0.0),
                    "top_p": request.top_p.unwrap_or(1.0),
                    "top_k": request.top_k.unwrap_or(0),
                    "seed": request.seed.unwrap_or(0),
                    "use_fsm": request.use_fsm,
                    "cfg_scale": request.cfg_scale.unwrap_or(1.0),
                    "negative_prompt": request.negative_prompt,
                }),
            }))
        }
        "image.edit" => serde_json::from_value::<ImageEditRequest>(request)
            .map(EngineRequest::ImageEdit)
            .map_err(|error| EngineError::bad_request(error.to_string())),
        "image.upscale" => serde_json::from_value::<ImageUpscaleRequest>(request)
            .map(EngineRequest::ImageUpscale)
            .map_err(|error| EngineError::bad_request(error.to_string())),
        "video.generation" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if !matches!(family, "wan" | "ltx") {
                return unsupported(model, Action::VideoCreate);
            }
            serde_json::from_value::<VideoGenerationRequest>(request)
                .map(EngineRequest::VideoCreate)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.edit" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if family != "hunyuan" {
                return unsupported(model, Action::VideoEdit);
            }
            serde_json::from_value::<VideoEditRequest>(request)
                .map(EngineRequest::VideoEdit)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.upscale" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if family != "seedvr2" {
                return unsupported(model, Action::VideoUpscale);
            }
            serde_json::from_value::<VideoUpscaleRequest>(request)
                .map(EngineRequest::VideoUpscale)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.avatar" => unsupported(model, Action::VideoAvatar),
        other => Err(EngineError::bad_request(format!(
            "unsupported task kind {other}"
        ))),
    }
}

fn unsupported(model: &str, action: Action) -> Result<EngineRequest, EngineError> {
    Ok(EngineRequest::Unsupported {
        model: model.into(),
        action,
    })
}

fn metadata_bool(metadata: &Value, key: &str) -> Result<Option<bool>, EngineError> {
    match metadata.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::String(value)) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(Some(true)),
            "0" | "false" | "no" | "off" => Ok(Some(false)),
            _ => Err(EngineError::bad_request(format!(
                "ACE-Step metadata {key}={value:?} is not a boolean"
            ))),
        },
        Some(_) => Err(EngineError::bad_request(format!(
            "ACE-Step metadata {key} is not a boolean"
        ))),
    }
}

fn metadata_text(metadata: &Value, key: &str) -> Result<Option<String>, EngineError> {
    match metadata.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(EngineError::bad_request(format!(
            "ACE-Step metadata {key} is not a string"
        ))),
    }
}

fn metadata_float(metadata: &Value, key: &str) -> Result<Option<f32>, EngineError> {
    let value = match metadata.get(key) {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Number(value)) => value.as_f64().map(|value| value as f32),
        Some(Value::String(value)) => value.parse::<f32>().ok(),
        Some(_) => None,
    }
    .filter(|value| value.is_finite());
    match value {
        Some(value) => Ok(Some(value)),
        None => Err(EngineError::bad_request(format!(
            "ACE-Step metadata {key} is not a finite number"
        ))),
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("asset"),
        std::process::id()
    ));
    std::fs::write(&temporary, bytes).map_err(|error| EngineError::Persist(error.to_string()))?;
    std::fs::rename(&temporary, path).map_err(|error| EngineError::Persist(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_parsing_is_typed_and_fail_loud() {
        let valid = json!({"enabled": "yes", "mode": "double", "scale": "0.05"});
        assert_eq!(metadata_bool(&valid, "enabled").unwrap(), Some(true));
        assert_eq!(
            metadata_text(&valid, "mode").unwrap().as_deref(),
            Some("double")
        );
        assert_eq!(metadata_float(&valid, "scale").unwrap(), Some(0.05));
        assert!(metadata_bool(&json!({"enabled": "maybe"}), "enabled").is_err());
        assert!(metadata_text(&json!({"mode": 1}), "mode").is_err());
        assert!(metadata_float(&json!({"scale": "bad"}), "scale").is_err());
        assert!(metadata_float(&json!({"scale": null}), "scale")
            .unwrap()
            .is_none());
    }

    #[test]
    fn video_generation_routes_wan_and_ltx() {
        let params = json!({
            "family": "wan",
            "request": {"model": "wan-2.2-ti2v-5b", "prompt": "a cat", "num_frames": 81},
        });
        let request = engine_request("video.generation", "wan-2.2-ti2v-5b", &params).unwrap();
        match request {
            EngineRequest::VideoCreate(request) => {
                assert_eq!(request.prompt, "a cat");
                assert_eq!(request.num_frames, Some(81));
                assert_eq!(request.size, None);
                assert_eq!(request.fps, None);
            }
            other => panic!("wan video generation routed to {other:?}"),
        }
        let params = json!({
            "family": "ltx",
            "request": {"model": "ltx-2.3-dev", "prompt": "a cat", "num_frames": 97},
        });
        let request = engine_request("video.generation", "ltx-2.3-dev", &params).unwrap();
        match request {
            EngineRequest::VideoCreate(request) => {
                assert_eq!(request.prompt, "a cat");
                assert_eq!(request.num_frames, Some(97));
            }
            other => panic!("ltx video generation routed to {other:?}"),
        }
        let params = json!({"family": "hunyuan", "request": {"model": "hunyuan", "prompt": "x"}});
        let request = engine_request("video.generation", "hunyuan", &params).unwrap();
        assert!(matches!(
            request,
            EngineRequest::Unsupported {
                action: Action::VideoCreate,
                ..
            }
        ));
    }

    #[test]
    fn video_edit_routes_hunyuan_only() {
        let params = json!({
            "family": "hunyuan",
            "request": {
                "model": "hunyuan-video-1.5-i2v-720p",
                "source_asset_id": "ast_src",
                "prompt": "a cat",
                "num_frames": 81,
            },
        });
        let request = engine_request("video.edit", "hunyuan-video-1.5-i2v-720p", &params).unwrap();
        match request {
            EngineRequest::VideoEdit(request) => {
                assert_eq!(request.source_asset_id, "ast_src");
                assert_eq!(request.num_frames, Some(81));
                assert_eq!(request.size, None);
            }
            other => panic!("hunyuan video edit routed to {other:?}"),
        }
        let params = json!({
            "family": "wan",
            "request": {"model": "wan-2.2-ti2v-5b", "source_asset_id": "ast_src", "prompt": "x"},
        });
        let request = engine_request("video.edit", "wan-2.2-ti2v-5b", &params).unwrap();
        assert!(matches!(
            request,
            EngineRequest::Unsupported {
                action: Action::VideoEdit,
                ..
            }
        ));
    }
}
