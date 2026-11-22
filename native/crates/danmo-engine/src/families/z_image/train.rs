//! Z-Image / Turbo LoRA training entry (Rust MLX).
//!
//! v1 trains Flow-Match SFT on cached latents through the shared train loop.
//! Full DiT forward differentiation lands by feeding LoRA A/B as Closure
//! inputs into the existing MLX blocks; the smoke path below validates loss,
//! export, and `attach_lora` round-trip with the DiffSynth schedule.

use crate::error::EngineError;
use crate::train::{
    export_lora_adapters, init_lora_params, run_lora_train_steps, TrainEvent, TrainProgress,
    TrainStepConfig, TrainStepResult,
};
use danmo_contracts::{
    merge_lora_train_request, resolve_lora_train_preset, LoraInferenceRecipe, LoraTrainingRequest,
    LoraTrainingResult,
};
use danmo_mlx::{Array, RandomKey, Stream};
use serde_json::json;
use std::path::{Path, PathBuf};

use super::mlx::{self, MlxZImageConfig};
use crate::weights::WeightDtype;

pub struct ZImageTrainOutput {
    pub result: LoraTrainingResult,
    pub work_dir: PathBuf,
}

/// Run a Z-Image family LoRA training job.
///
/// When `latent_x0` is provided (encode cache / tests), trains against that
/// batch. Otherwise fails loud — encode cache must be prepared by the caller.
pub fn run_z_image_lora_training(
    request: &LoraTrainingRequest,
    work_dir: &Path,
    stream: &Stream,
    latent_x0: &Array,
) -> Result<ZImageTrainOutput, EngineError> {
    let mid = request
        .base_model
        .split(':')
        .next()
        .unwrap_or(&request.base_model)
        .trim();
    if mid != "z-image" && mid != "z-image-turbo" {
        return Err(EngineError::message(format!(
            "Z-Image trainer only supports z-image|z-image-turbo, got {mid:?}"
        )));
    }
    let preset = resolve_lora_train_preset(&request.preset, &request.base_model)
        .map_err(EngineError::message)?;
    let preset = merge_lora_train_request(request, preset).map_err(EngineError::message)?;

    std::fs::create_dir_all(work_dir).map_err(|e| EngineError::message(e.to_string()))?;
    let adapter_dir = work_dir.join("adapters");
    std::fs::create_dir_all(&adapter_dir).map_err(|e| EngineError::message(e.to_string()))?;

    // Probe linear matching latent channel width for the smoke / unit path.
    // Family DiT wiring replaces this with real module dims from the bundle.
    let shape = latent_x0.shape();
    if shape.len() < 2 {
        return Err(EngineError::message(
            "latent_x0 must be at least rank-2 [batch, features]",
        ));
    }
    let feat = shape[shape.len() - 1];
    let base = identity_weight(feat, stream)?;
    let seed = request.seed.unwrap_or(42) as u64;
    let rng = RandomKey::new(seed).map_err(|e| EngineError::message(e.to_string()))?;
    let smoke = request
        .metadata
        .get("smoke")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Smoke / probe latents are tiny; clamp rank to feature width.
    let rank = if smoke {
        (preset.lora_rank as i32).min(feat).min(4).max(1)
    } else {
        preset.lora_rank as i32
    };
    let alpha = rank as f32;
    let mut params = vec![
        init_lora_params("proj", feat, feat, rank, alpha, &rng, stream)
            .map_err(|e| EngineError::message(e.to_string()))?,
    ];

    let iterations = if smoke {
        preset.iterations.min(32)
    } else {
        preset.iterations
    };
    let learning_rate = if smoke { 5e-2 } else { preset.learning_rate };

    let shift = if preset.id == "turbo-sft" {
        3.0
    } else {
        preset.train_sigma_shift
    };
    let fixed_sigma = if smoke { Some(0.5) } else { None };
    let trained: TrainStepResult = run_lora_train_steps(
        &base,
        &mut params,
        latent_x0,
        &TrainStepConfig {
            iterations,
            learning_rate,
            weight_decay: 0.01,
            seed,
            shift,
            schedule_steps: 1000,
            fixed_sigma,
        },
        stream,
    )?;

    let final_path = adapter_dir.join("final_adapters.safetensors");
    export_lora_adapters(&final_path, &params, stream)?;

    let config_path = adapter_dir.join("lora_config.json");
    let config = json!({
        "base_model": mid,
        "preset": preset.id,
        "lora_rank": preset.lora_rank,
        "lora_blocks": preset.lora_blocks,
        "lora_module_keys": preset.lora_module_keys,
        "inference": preset.inference,
        "train_sigma_shift": shift,
        "sigma_bias": preset.sigma_bias,
    });
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config).map_err(|e| EngineError::message(e.to_string()))?,
    )
    .map_err(|e| EngineError::message(e.to_string()))?;

    let loss_history: Vec<serde_json::Value> = trained
        .loss_history
        .iter()
        .enumerate()
        .map(|(i, loss)| json!({ "step": i + 1, "loss": loss }))
        .collect();

    let output_name = if request.output_name.trim().is_empty() {
        format!("{mid}-{}", request.dataset_id)
    } else {
        request.output_name.clone()
    };

    Ok(ZImageTrainOutput {
        result: LoraTrainingResult {
            adapter_path: final_path.display().to_string(),
            user_lora_id: String::new(),
            output_name,
            loss_history,
            needs_candidate_evaluation: true,
            checkpoints: vec![final_path.display().to_string()],
            inference: Some(preset.inference),
        },
        work_dir: work_dir.to_path_buf(),
    })
}

/// Full-blood training entry: real VAE latents, Qwen3 captions, and
/// Flow-Match SFT through the actual DiT on the MLX executor. Fails loud on
/// anything the pipeline cannot honor (face detection, grad accumulation,
/// unknown module keys) — no silent downgrade to the probe path.
pub fn run_z_image_full_training(
    request: &LoraTrainingRequest,
    bundle: &Path,
    pairs: &[(PathBuf, String)],
    resume_adapter: Option<&Path>,
    work_dir: &Path,
    progress: TrainProgress,
) -> Result<LoraTrainingResult, EngineError> {
    let mid = request
        .base_model
        .split(':')
        .next()
        .unwrap_or(&request.base_model)
        .trim();
    if mid != "z-image" && mid != "z-image-turbo" {
        return Err(EngineError::message(format!(
            "Z-Image trainer only supports z-image|z-image-turbo, got {mid:?}"
        )));
    }
    let preset = resolve_lora_train_preset(&request.preset, &request.base_model)
        .map_err(EngineError::message)?;
    let mut preset = merge_lora_train_request(request, preset).map_err(EngineError::message)?;
    if preset.grad_accumulate != 1 {
        return Err(EngineError::message(format!(
            "grad_accumulate {} is not wired; set the recipe to 1",
            preset.grad_accumulate
        )));
    }
    match request.face_crop.as_deref().map(str::trim) {
        None | Some("") | Some("off") => {}
        Some("auto") => progress(TrainEvent::Phase(
            "face_crop=auto: face detection is not wired; using center crop".into(),
        )),
        Some("on") => {
            return Err(EngineError::message(
                "face_crop=on needs face detection, which is not wired; \
                 recrop the dataset or set face_crop=off",
            ))
        }
        Some(other) => {
            return Err(EngineError::message(format!(
                "unknown face_crop value {other:?}"
            )))
        }
    }
    // Diagnostic override for short E2E runs; always logged, never silent.
    if let Some(iterations) = request.metadata.get("iterations").and_then(|v| v.as_u64()) {
        if !(1..=100_000).contains(&iterations) {
            return Err(EngineError::message(format!(
                "metadata.iterations {iterations} is outside 1..=100000"
            )));
        }
        progress(TrainEvent::Phase(format!(
            "metadata override: iterations {} -> {iterations}",
            preset.iterations
        )));
        preset.iterations = iterations as u32;
    }
    let [width, height] = preset.resolution;
    let align = 2 * MlxZImageConfig::published().vae_scale as u32;
    if width % align != 0 || height % align != 0 {
        return Err(EngineError::message(format!(
            "resolution {width}x{height} is not divisible by {align}"
        )));
    }
    if pairs.is_empty() {
        return Err(EngineError::message("training dataset is empty"));
    }

    progress(TrainEvent::Phase(format!(
        "preprocessing {} images at {width}x{height}",
        pairs.len()
    )));
    let mut images = Vec::with_capacity(pairs.len());
    let mut caption_ids = Vec::with_capacity(pairs.len());
    let config = MlxZImageConfig::published();
    for (path, caption) in pairs {
        images.push(preprocess_image(path, width, height)?);
        caption_ids.push(
            config
                .token_ids(bundle, caption)
                .map_err(|e| EngineError::message(e.to_string()))?,
        );
    }
    // Caption dropout uses the ChatML scaffold with empty user content
    // (DiffSynth encodes "" through the same template); generation's
    // non-empty prompt guard stays untouched.
    let tokenizer = crate::common::qwen3::load_tokenizer(bundle)
        .map_err(|e| EngineError::message(e.to_string()))?;
    let empty_ids = crate::common::qwen3::tokenize_chat_empty(&tokenizer, config.vocab, true)
        .map_err(|e| EngineError::message(e.to_string()))?;

    let dit_dir = bundle.join("transformer");
    let text_dir = bundle.join("text_encoder");
    let vae_dir = bundle.join("vae");
    for (dir, role) in [
        (&dit_dir, "transformer"),
        (&text_dir, "text encoder"),
        (&vae_dir, "vae"),
    ] {
        if !dir.is_dir() {
            return Err(EngineError::message(format!(
                "bundle {} is missing the {role} component",
                bundle.display()
            )));
        }
    }
    progress(TrainEvent::Phase(
        "loading transformer, text encoder, vae".into(),
    ));
    mlx::resident::load_transformer(&dit_dir, config, WeightDtype::Native, None)
        .map_err(|e| EngineError::message(e.to_string()))?;
    mlx::resident::load_text_encoder(&text_dir).map_err(|e| EngineError::message(e.to_string()))?;
    mlx::resident::load_vae(&vae_dir).map_err(|e| EngineError::message(e.to_string()))?;
    let job = mlx::train::TrainJob {
        request: request.clone(),
        preset,
        caption_ids,
        empty_ids,
        images,
        width,
        height,
        resume_adapter: resume_adapter.map(Path::to_path_buf),
        work_dir: work_dir.to_path_buf(),
        dit_key: mlx::resident::cache_key(&dit_dir)
            .map_err(|e| EngineError::message(e.to_string()))?,
        text_key: mlx::resident::cache_key(&text_dir)
            .map_err(|e| EngineError::message(e.to_string()))?,
        vae_key: mlx::resident::cache_key(&vae_dir)
            .map_err(|e| EngineError::message(e.to_string()))?,
    };
    crate::runtime::mlx_executor()
        .map_err(|e| EngineError::message(e.to_string()))?
        .execute("z_image lora train", move |_, stream, cache, _| {
            Ok(mlx::train::train_on_executor(job, stream, cache, progress))
        })
        .map_err(|e| EngineError::message(e.to_string()))?
}

/// Resize so the shorter side covers the target, then center-crop to the
/// recipe resolution (DiffSynth bucket behavior for a fixed bucket).
fn preprocess_image(path: &Path, width: u32, height: u32) -> Result<Vec<f32>, EngineError> {
    let (rgb, w, h) = crate::common::image::decode_file_rgb(path).map_err(EngineError::message)?;
    let scale = (width as f64 / w as f64).max(height as f64 / h as f64);
    let nw = ((w as f64 * scale).ceil() as u32).max(width);
    let nh = ((h as f64 * scale).ceil() as u32).max(height);
    let rgb = if nw == w && nh == h {
        rgb
    } else {
        crate::common::image::resize_lanczos(&rgb, w, h, nw, nh)
            .map_err(|e| EngineError::message(e.to_string()))?
    };
    let (dx, dy) = ((nw - width) / 2, (nh - height) / 2);
    let (nwu, wu) = (nw as usize, width as usize);
    let mut cropped = vec![0u8; wu * height as usize * 3];
    for row in 0..height as usize {
        let src = ((row + dy as usize) * nwu + dx as usize) * 3;
        let dst = row * wu * 3;
        cropped[dst..dst + wu * 3].copy_from_slice(&rgb[src..src + wu * 3]);
    }
    crate::common::image::rgb_to_n11_planar(&cropped, width, height)
        .map_err(|e| EngineError::message(e.to_string()))
}

fn identity_weight(dim: i32, stream: &Stream) -> Result<Array, EngineError> {
    let mut data = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim as usize {
        data[i * dim as usize + i] = 1.0;
    }
    let w = Array::from_f32_slice(&data, &[dim, dim])
        .map_err(|e| EngineError::message(e.to_string()))?;
    let _ = stream;
    Ok(w)
}

#[cfg(all(test, feature = "mlx"))]
mod tests {
    use super::*;
    use danmo_mlx::Stream;

    #[test]
    fn z_image_turbo_sft_smoke_train() {
        let stream = Stream::cpu().unwrap();
        let x0 =
            Array::from_f32_slice(&[0.2, -0.1, 0.4, 0.05, -0.3, 0.15, 0.0, 0.25], &[4, 2]).unwrap();
        let dir = std::env::temp_dir().join(format!("zimg-train-{}", std::process::id()));
        let req = LoraTrainingRequest {
            base_model: "z-image-turbo".into(),
            dataset_id: "ds_test".into(),
            preset: "turbo-sft".into(),
            output_name: "smoke".into(),
            auto_register: false,
            priority: "normal".into(),
            seed: Some(3),
            face_crop: None,
            resume_task_id: String::new(),
            resume_checkpoint: String::new(),
            metadata: json!({ "smoke": true }),
        };
        let out = run_z_image_lora_training(&req, &dir, &stream, &x0).unwrap();
        assert!(Path::new(&out.result.adapter_path).is_file());
        let recipe = out.result.inference.as_ref().unwrap();
        assert_eq!(recipe.scheme, "sft");
        assert_eq!(recipe.extra_adapters.len(), 1);
        assert_eq!(recipe.extra_adapters[0].weight, 1.0);
        let losses: Vec<f64> = out
            .result
            .loss_history
            .iter()
            .map(|row| row["loss"].as_f64().unwrap())
            .collect();
        let n = losses.len();
        let first = losses[..n / 4].iter().sum::<f64>() / (n / 4) as f64;
        let last = losses[3 * n / 4..].iter().sum::<f64>() / (n / 4) as f64;
        assert!(
            last < first * 0.95,
            "mean loss did not drop: {first} → {last}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
