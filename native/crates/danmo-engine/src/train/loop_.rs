//! Generic LoRA train loop: Closure + value_and_grad + AdamW.

use super::export::LoraTrainParam;
use super::loss::{flow_match_mse, sample_noisy_latent};
use super::schedule::{sample_discrete_sigma, FlowMatchTrainSchedule};
use crate::error::EngineError;
use danmo_mlx::{
    adamw_update_all, AdamWConfig, AdamWState, Array, Closure, Dtype, RandomKey, Stream,
    ValueAndGrad,
};

pub struct TrainStepConfig {
    pub iterations: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub shift: f32,
    pub schedule_steps: usize,
    /// When set, every step uses this sigma (unit tests / diagnostics).
    pub fixed_sigma: Option<f32>,
}

pub struct TrainStepResult {
    pub loss_history: Vec<f32>,
    pub final_loss: f32,
}

/// Train LoRA residuals on a single dense linear `W` with Flow-Match SFT.
///
/// Forward: `pred = (x_t @ W^T) + scale * (x_t @ A^T @ B^T)` matching
/// `LinearWeight` residual layout. Used as the rust MLX trainer smoke path
/// and as the differentiable core for family runners.
pub fn run_lora_train_steps(
    base_weight: &Array,
    params: &mut [LoraTrainParam],
    x0_batch: &Array,
    config: &TrainStepConfig,
    stream: &Stream,
) -> Result<TrainStepResult, EngineError> {
    if params.is_empty() {
        return Err(EngineError::message("run_lora_train_steps: empty params"));
    }
    let schedule = FlowMatchTrainSchedule::z_image(config.schedule_steps, config.shift);
    let mut rng = RandomKey::new(config.seed).map_err(|e| EngineError::message(e.to_string()))?;
    let adam = AdamWConfig {
        learning_rate: config.learning_rate,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: config.weight_decay,
    };
    let mut states: Vec<AdamWState> = Vec::new();
    for p in params.iter() {
        states.push(
            AdamWState::zeros_like(&p.a, stream)
                .map_err(|e| EngineError::message(e.to_string()))?,
        );
        states.push(
            AdamWState::zeros_like(&p.b, stream)
                .map_err(|e| EngineError::message(e.to_string()))?,
        );
    }

    let mut loss_history = Vec::new();
    let mut last_loss = f32::INFINITY;

    for step in 0..config.iterations {
        let (key_eps, key_rest) = rng
            .split(stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        rng = key_rest;
        let shape = x0_batch.shape();
        let eps = key_eps
            .normal(&shape, Dtype::Float32, stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        let (sigma, weight) = if let Some(fixed) = config.fixed_sigma {
            (fixed.clamp(1e-3, 1.0 - 1e-3), 1.0)
        } else {
            let (s, w, _) = sample_discrete_sigma(&schedule, host_unit(step));
            (s, w)
        };
        let x_t = sample_noisy_latent(x0_batch, sigma, &eps, stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        let weight_arr =
            Array::scalar_f32(weight).map_err(|e| EngineError::message(e.to_string()))?;

        // Flatten trainable: [a0,b0,a1,b1,...]
        let mut flat: Vec<Array> = Vec::new();
        for p in params.iter() {
            flat.push(p.a.clone());
            flat.push(p.b.clone());
        }
        let n_train = flat.len();
        let argnums: Vec<i32> = (0..n_train as i32).collect();

        // Closure inputs: trainable..., x_t, eps, x0, base_w, loss_weight, scales...
        let mut inputs = flat;
        inputs.push(x_t);
        inputs.push(eps);
        inputs.push(x0_batch.clone());
        inputs.push(base_weight.clone());
        inputs.push(weight_arr);
        for p in params.iter() {
            inputs
                .push(Array::scalar_f32(p.scale).map_err(|e| EngineError::message(e.to_string()))?);
        }

        let n_modules = params.len();
        let stream_for_closure = Stream::cpu().map_err(|e| EngineError::message(e.to_string()))?;
        let closure = Closure::new(stream_for_closure, move |ins, stream| {
            let train = &ins[..n_train];
            let x_t = &ins[n_train];
            let eps = &ins[n_train + 1];
            let x0 = &ins[n_train + 2];
            let base = &ins[n_train + 3];
            let loss_w = &ins[n_train + 4];
            let scales = &ins[n_train + 5..];
            let mut pred = Array::linear(x_t, base, None, stream)?;
            for i in 0..n_modules {
                let a = &train[i * 2];
                let b = &train[i * 2 + 1];
                let scale = &scales[i];
                // residual: scale * ((x @ A^T) @ B^T)
                let a_t = a.matrix_transpose(stream)?;
                let mid = Array::matmul(x_t, &a_t, stream)?;
                let b_t = b.matrix_transpose(stream)?;
                let delta = Array::matmul(&mid, &b_t, stream)?;
                let scaled = Array::multiply(&delta, scale, stream)?;
                pred = Array::add(&pred, &scaled, stream)?;
            }
            let loss = flow_match_mse(&pred, x0, eps, stream)?;
            Array::multiply(&loss, loss_w, stream)
        })
        .map_err(|e| EngineError::message(e.to_string()))?;

        let vag = ValueAndGrad::new(closure, &argnums)
            .map_err(|e| EngineError::message(e.to_string()))?;
        let (values, grads) = vag
            .apply(&inputs)
            .map_err(|e| EngineError::message(e.to_string()))?;
        if values.is_empty() || grads.len() != n_train {
            return Err(EngineError::message(format!(
                "value_and_grad returned {} values / {} grads (expected 1 / {n_train})",
                values.len(),
                grads.len()
            )));
        }
        Array::eval_all(&[&values[0]]).map_err(|e| EngineError::message(e.to_string()))?;
        let loss_v = values[0]
            .to_vec_f32(stream)
            .map_err(|e| EngineError::message(e.to_string()))?[0];
        loss_history.push(loss_v);
        last_loss = loss_v;

        let mut train_params: Vec<Array> = Vec::new();
        for p in params.iter() {
            train_params.push(p.a.clone());
            train_params.push(p.b.clone());
        }
        adamw_update_all(&mut train_params, &grads, &mut states, &adam, stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        for (i, p) in params.iter_mut().enumerate() {
            p.a = train_params[i * 2].clone();
            p.b = train_params[i * 2 + 1].clone();
        }
    }

    Ok(TrainStepResult {
        loss_history,
        final_loss: last_loss,
    })
}

fn host_unit(step: u32) -> f32 {
    let x = ((step.wrapping_mul(1103515245).wrapping_add(12345)) % 10_000) as f32;
    x / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::export::{export_lora_adapters, init_lora_params};
    use crate::weights::{LoraFile, MlxWeights};
    use std::collections::BTreeMap;
    use std::env;

    #[test]
    fn fake_latent_lora_train_exports_attachable_adapter() {
        let stream = Stream::cpu().unwrap();
        let base = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let rng = RandomKey::new(7).unwrap();
        let mut params = vec![init_lora_params("proj", 2, 2, 4, 4.0, &rng, &stream).unwrap()];
        let x0 =
            Array::from_f32_slice(&[0.2, -0.1, 0.4, 0.05, -0.3, 0.15, 0.0, 0.25], &[4, 2]).unwrap();
        let result = run_lora_train_steps(
            &base,
            &mut params,
            &x0,
            &TrainStepConfig {
                iterations: 24,
                learning_rate: 5e-2,
                weight_decay: 0.0,
                seed: 11,
                shift: 3.0,
                schedule_steps: 64,
                fixed_sigma: Some(0.5),
            },
            &stream,
        )
        .unwrap();
        assert!(result.loss_history.iter().all(|l| l.is_finite()));
        let b = params[0].b.to_vec_f32(&stream).unwrap();
        assert!(b.iter().any(|v| v.abs() > 1e-6), "LoRA B stayed zero");

        let dir = env::temp_dir().join(format!("danmo-lora-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapters.safetensors");
        export_lora_adapters(&path, &params, &stream).unwrap();

        let mut tensors = BTreeMap::new();
        tensors.insert("proj.weight".into(), base);
        let weights = MlxWeights::from_dense(tensors);
        let adapted = weights
            .attach_lora(
                &[LoraFile {
                    id: "train".into(),
                    path: path.clone(),
                    weight: 1.0,
                }],
                &stream,
            )
            .unwrap();
        let _ = adapted.linear("proj.weight").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
