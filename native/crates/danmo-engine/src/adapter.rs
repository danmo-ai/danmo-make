//! Mechanical task-kind → `EngineRequest` mapping shared by every host.
//!
//! The HTTP server and the CLI both accept tasks as `{kind, model, params}`
//! triples; this module is the single place that turns them into typed
//! engine requests. It contains no family-id branches: availability gates go
//! through `session::ensure_available`, and family-specific knobs travel in
//! the opaque `family_options` bag the family parses itself.

use crate::contract::{
    Action, AudioCreateRequest as EngineAudioCreateRequest,
    AudioEditRequest as EngineAudioEditRequest, AudioLmRequest as EngineAudioLmRequest,
    EngineRequest,
};
use crate::error::EngineError;
use danmo_contracts::{
    AudioEditRequest, AudioGenerationRequest, AudioLmRequest, ImageEditRequest,
    ImageGenerationRequest, ImageUpscaleRequest, VideoAvatarRequest, VideoEditRequest,
    VideoGenerationRequest, VideoUpscaleRequest,
};
use serde_json::{json, Value};

pub fn engine_request(
    kind: &str,
    model: &str,
    params: &Value,
) -> Result<EngineRequest, EngineError> {
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
                title: request.title,
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
                title: request.title,
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
            if crate::session::ensure_available(family, Action::VideoCreate).is_err() {
                return unsupported(model, Action::VideoCreate);
            }
            serde_json::from_value::<VideoGenerationRequest>(request)
                .map(EngineRequest::VideoCreate)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.edit" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if crate::session::ensure_available(family, Action::VideoEdit).is_err() {
                return unsupported(model, Action::VideoEdit);
            }
            serde_json::from_value::<VideoEditRequest>(request)
                .map(EngineRequest::VideoEdit)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.upscale" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if crate::session::ensure_available(family, Action::VideoUpscale).is_err() {
                return unsupported(model, Action::VideoUpscale);
            }
            serde_json::from_value::<VideoUpscaleRequest>(request)
                .map(EngineRequest::VideoUpscale)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
        "video.avatar" => {
            let family = params.get("family").and_then(Value::as_str).unwrap_or("");
            if crate::session::ensure_available(family, Action::VideoAvatar).is_err() {
                return unsupported(model, Action::VideoAvatar);
            }
            serde_json::from_value::<VideoAvatarRequest>(request)
                .map(EngineRequest::VideoAvatar)
                .map_err(|error| EngineError::bad_request(error.to_string()))
        }
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
                "audio metadata {key}={value:?} is not a boolean"
            ))),
        },
        Some(_) => Err(EngineError::bad_request(format!(
            "audio metadata {key} is not a boolean"
        ))),
    }
}

fn metadata_text(metadata: &Value, key: &str) -> Result<Option<String>, EngineError> {
    match metadata.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(EngineError::bad_request(format!(
            "audio metadata {key} is not a string"
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
            "audio metadata {key} is not a finite number"
        ))),
    }
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
    fn image_upscale_preserves_spatial_tile_size() {
        let params = json!({
            "request": {
                "model": "real-esrgan-x4plus",
                "source_asset_id": "ast_src",
                "scale": 4,
                "tile_size": 256
            }
        });
        let request = engine_request("image.upscale", "real-esrgan-x4plus", &params).unwrap();
        let EngineRequest::ImageUpscale(request) = request else {
            panic!("expected image upscale request");
        };
        assert_eq!(request.tile_size, Some(256));
    }

    #[test]
    fn video_generation_gates_on_engine_availability() {
        let params = json!({
            "family": "wan",
            "request": {"model": "wan-2.2-ti2v-5b", "prompt": "a cat", "num_frames": 81},
        });
        let request = engine_request("video.generation", "wan-2.2-ti2v-5b", &params).unwrap();
        assert!(matches!(request, EngineRequest::VideoCreate(_)));
        let params = json!({
            "family": "minimax_h3",
            "request": {"model": "minimax-h3-fl2va", "prompt": "x"},
        });
        let request = engine_request("video.generation", "minimax-h3-fl2va", &params).unwrap();
        assert!(matches!(
            request,
            EngineRequest::Unsupported {
                action: Action::VideoCreate,
                ..
            }
        ));
    }
}
