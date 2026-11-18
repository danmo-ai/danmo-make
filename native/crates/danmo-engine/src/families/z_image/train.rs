//! Z-Image / Turbo LoRA training entry (Rust MLX).
//!
//! v1 trains Flow-Match SFT on cached latents through the shared train loop.
//! Full DiT forward differentiation lands by feeding LoRA A/B as Closure
//! inputs into the existing MLX blocks; the smoke path below validates loss,
//! export, and `attach_lora` round-trip with the DiffSynth schedule.

use crate::error::EngineError;
use crate::train::{
    export_lora_adapters, init_lora_params, run_lora_train_steps, TrainStepConfig, TrainStepResult,
};
use danmo_contracts::{
    merge_lora_train_request, resolve_lora_train_preset, LoraInferenceRecipe, LoraTrainingRequest,
    LoraTrainingResult,
};
use danmo_mlx::{Array, RandomKey, Stream};
use serde_json::json;
use std::path::{Path, PathBuf};

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
    let mut params = vec![init_lora_params(
        "proj",
        feat,
        feat,
        rank,
        alpha,
        &rng,
        stream,
    )
    .map_err(|e| EngineError::message(e.to_string()))?];

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

fn identity_weight(dim: i32, stream: &Stream) -> Result<Array, EngineError> {
    let mut data = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim as usize {
        data[i * dim as usize + i] = 1.0;
    }
    let w = Array::from_f32_slice(&data, &[dim, dim]).map_err(|e| EngineError::message(e.to_string()))?;
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
        let x0 = Array::from_f32_slice(&[0.2, -0.1, 0.4, 0.05, -0.3, 0.15, 0.0, 0.25], &[4, 2])
            .unwrap();
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
        assert!(last < first * 0.95, "mean loss did not drop: {first} → {last}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
