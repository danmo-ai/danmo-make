//! LoRA parameter init / safetensors export (inference `attach_lora` key layout).

use crate::error::EngineError;
use crate::weights::{write_safetensors, SafeTensor};
use danmo_mlx::{Array, Dtype, RandomKey, Result as MlxResult, Stream};
use std::path::Path;

pub struct LoraTrainParam {
    pub module: String,
    pub a: Array,
    pub b: Array,
    pub scale: f32,
    pub in_dim: i32,
    pub out_dim: i32,
    pub rank: i32,
}

/// Initialize LoRA A/B for one linear `[out, in]` (A=`[rank,in]`, B=`[out,rank]`).
pub fn init_lora_params(
    module: &str,
    out_dim: i32,
    in_dim: i32,
    rank: i32,
    alpha: f32,
    rng: &RandomKey,
    stream: &Stream,
) -> MlxResult<LoraTrainParam> {
    let rank = rank.max(1);
    let a = rng.normal(&[rank, in_dim], Dtype::Float32, stream)?;
    // Scale init like common LoRA: A ~ N(0, 1/sqrt(in)), B zeros.
    let scale_a = Array::scalar_f32(1.0 / (in_dim as f32).sqrt())?;
    let a = Array::multiply(&a, &scale_a, stream)?;
    let b = Array::from_f32_slice(&vec![0.0; (out_dim * rank) as usize], &[out_dim, rank])?;
    Ok(LoraTrainParam {
        module: module.to_string(),
        a,
        b,
        scale: alpha / rank as f32,
        in_dim,
        out_dim,
        rank,
    })
}

pub fn export_lora_adapters(
    path: &Path,
    params: &[LoraTrainParam],
    stream: &Stream,
) -> Result<(), EngineError> {
    if params.is_empty() {
        return Err(EngineError::message(
            "export_lora_adapters: no LoRA parameters",
        ));
    }
    let mut tensors = Vec::new();
    let mut owned: Vec<(String, Vec<u8>, Vec<u64>)> = Vec::new();
    for param in params {
        param
            .a
            .eval()
            .map_err(|e| EngineError::message(e.to_string()))?;
        param
            .b
            .eval()
            .map_err(|e| EngineError::message(e.to_string()))?;
        let a_vals = param
            .a
            .to_vec_f32(stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        let b_vals = param
            .b
            .to_vec_f32(stream)
            .map_err(|e| EngineError::message(e.to_string()))?;
        let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_name = format!("{}.lora_A.weight", param.module);
        let b_name = format!("{}.lora_B.weight", param.module);
        let a_shape = vec![param.rank as u64, param.in_dim as u64];
        let b_shape = vec![param.out_dim as u64, param.rank as u64];
        owned.push((a_name, a_bytes, a_shape));
        owned.push((b_name, b_bytes, b_shape));
        let alpha = (param.scale * param.rank as f32).to_le_bytes().to_vec();
        owned.push((format!("{}.alpha", param.module), alpha, vec![]));
    }
    for (name, data, shape) in &owned {
        tensors.push(SafeTensor {
            name: name.as_str(),
            dtype: "F32",
            shape: shape.as_slice(),
            data: data.as_slice(),
        });
    }
    write_safetensors(path, &tensors).map_err(|e| EngineError::message(e.to_string()))
}
