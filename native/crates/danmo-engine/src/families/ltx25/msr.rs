//! LTX 2.5 multi-subject reference (MSR) LoRA helpers.
//!
//! Slot embeddings are Fourier-MLP tensors stored alongside linear LoRA
//! factors in the same safetensors file. Linear factors attach through the
//! shared weight layer; slot embeddings are added to each reference token
//! block after patchify.

use std::path::Path;

use danmo_mlx::{Array, Dtype, Result, Stream};

use crate::weights::{LoraRoute, MlxWeights, WeightDtype, WeightError};

const IMAGE_SLOT_PREFIXES: &[&str] = &[
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
];

const AUDIO_SLOT_PREFIXES: &[&str] = &[
    "diffusion_model.reference_audio_slot_embedding.",
    "reference_audio_slot_embedding.",
];

/// Published MSR-AVref defaults for absolute audio time windows.
pub const AUDIO_SLOT_DURATION_SECONDS: f32 = 5.0;
pub const AUDIO_SLOT_END_MARGIN_SECONDS: f32 = 0.04;

/// Remap Diffusers / ComfyUI MSR LoRA module stems onto this family's graph.
pub fn route_lora_module(module: &str) -> std::result::Result<LoraRoute, WeightError> {
    if module.contains("reference_slot_embedding")
        || module.contains("reference_audio_slot_embedding")
    {
        return Ok(LoraRoute::Skip);
    }
    // Published Comfy / Diffusers MSR packs use `.ff.net.2` (no `.proj`) for the
    // out projection; some dumps still include `.net.2.proj`. Map both.
    let mapped = module
        .replace(".audio_ff.net.0.proj", ".audio_ff.proj_in")
        .replace(".audio_ff.net.2.proj", ".audio_ff.proj_out")
        .replace(".audio_ff.net.2", ".audio_ff.proj_out")
        .replace(".ff.net.0.proj", ".ff.proj_in")
        .replace(".ff.net.2.proj", ".ff.proj_out")
        .replace(".ff.net.2", ".ff.proj_out")
        .replace(".to_out.0", ".to_out");
    Ok(LoraRoute::One(mapped))
}

#[derive(Clone)]
pub struct SlotEmbedding {
    frequencies: Vec<f32>,
    weight0: Array,
    bias0: Array,
    weight2: Array,
    bias2: Array,
    dim: usize,
}

impl SlotEmbedding {
    pub fn load_image_slots(
        path: &Path,
        stream: &Stream,
    ) -> std::result::Result<Self, WeightError> {
        load_slots(path, IMAGE_SLOT_PREFIXES, "image", stream)
    }

    pub fn load_audio_slots(
        path: &Path,
        stream: &Stream,
    ) -> std::result::Result<Self, WeightError> {
        load_slots(path, AUDIO_SLOT_PREFIXES, "audio", stream)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Fourier-MLP embedding for 1-based slot id (pic1 → 1).
    pub fn embed(&self, slot_id: usize, stream: &Stream) -> Result<Array> {
        if slot_id == 0 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 MSR",
                "slot id must be 1-based",
            ));
        }
        let scaled = slot_id as f32 / 16.0;
        let mut features = Vec::with_capacity(1 + 2 * self.frequencies.len());
        features.push(scaled);
        for &frequency in &self.frequencies {
            let phase = scaled * frequency;
            features.push(phase.sin());
        }
        for &frequency in &self.frequencies {
            let phase = scaled * frequency;
            features.push(phase.cos());
        }
        let features = Array::from_f32_slice(&features, &[1, features.len() as i32])?;
        // Weights are stored [out, in]; matmul wants x @ W^T.
        let hidden = Array::add(
            &Array::matmul(&features, &self.weight0.matrix_transpose(stream)?, stream)?,
            &self.bias0,
            stream,
        )?
        .silu(stream)?;
        Array::add(
            &Array::matmul(&hidden, &self.weight2.matrix_transpose(stream)?, stream)?,
            &self.bias2,
            stream,
        )
    }

    /// Broadcast-add the slot embedding onto `[B,T,C]` reference tokens.
    pub fn apply_to_tokens(
        &self,
        tokens: &Array,
        slot_id: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = tokens.shape();
        if shape.len() != 3 || shape[2] as usize != self.dim {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 MSR",
                format!(
                    "reference tokens {:?} do not match slot embedding dim {}",
                    shape, self.dim
                ),
            ));
        }
        let embedding = self
            .embed(slot_id, stream)?
            .astype(tokens.dtype(), stream)?
            .reshape(&[1, 1, self.dim as i32], stream)?;
        Array::add(tokens, &embedding, stream)
    }
}

fn load_slots(
    path: &Path,
    prefixes: &[&str],
    label: &str,
    stream: &Stream,
) -> std::result::Result<SlotEmbedding, WeightError> {
    // Safetensors Load ops cannot eval on Metal; the shared weight layer
    // materializes on CPU, then casts onto the caller's stream for matmul.
    let cpu = Stream::cpu().map_err(mlx)?;
    let tensors = crate::weights::load_mlx_file(path, WeightDtype::Native, None).map_err(mlx)?;
    let frequencies = required_f32(&tensors, prefixes, "frequencies", &cpu)?;
    let weight0 = required_array(&tensors, prefixes, "net.0.weight")?;
    let bias0 = required_array(&tensors, prefixes, "net.0.bias")?;
    let weight2 = required_array(&tensors, prefixes, "net.2.weight")?;
    let bias2 = required_array(&tensors, prefixes, "net.2.bias")?;
    let freq_count = frequencies.len();
    let expected_features = 1 + 2 * freq_count;
    let w0_shape = weight0.shape();
    if w0_shape != [w0_shape[0], expected_features as i32] {
        return Err(WeightError::NotBuilt(format!(
            "MSR {label} slot net.0.weight shape {w0_shape:?} does not match 1+2*{freq_count} features"
        )));
    }
    let dim = bias2.shape().first().copied().unwrap_or(0) as usize;
    if dim == 0 {
        return Err(WeightError::NotBuilt(format!(
            "MSR {label} slot embedding output dimension is empty"
        )));
    }
    if weight2.shape() != [dim as i32, w0_shape[0]] {
        return Err(WeightError::NotBuilt(format!(
            "MSR {label} slot net.2.weight shape {:?} does not match hidden {}",
            weight2.shape(),
            w0_shape[0]
        )));
    }
    let _ = stream;
    Ok(SlotEmbedding {
        frequencies,
        weight0: weight0.astype(Dtype::Float32, &cpu).map_err(mlx)?,
        bias0: bias0.astype(Dtype::Float32, &cpu).map_err(mlx)?,
        weight2: weight2.astype(Dtype::Float32, &cpu).map_err(mlx)?,
        bias2: bias2.astype(Dtype::Float32, &cpu).map_err(mlx)?,
        dim,
    })
}

fn required_array(
    tensors: &MlxWeights,
    prefixes: &[&str],
    suffix: &str,
) -> std::result::Result<Array, WeightError> {
    for prefix in prefixes {
        let key = format!("{prefix}{suffix}");
        if tensors.has(&key) {
            return tensors.tensor(&key).cloned().map_err(mlx);
        }
    }
    Err(WeightError::NotBuilt(format!(
        "MSR LoRA is missing reference_slot_embedding.{suffix}"
    )))
}

fn required_f32(
    tensors: &MlxWeights,
    prefixes: &[&str],
    suffix: &str,
    stream: &Stream,
) -> std::result::Result<Vec<f32>, WeightError> {
    let array = required_array(tensors, prefixes, suffix)?;
    array
        .astype(Dtype::Float32, stream)
        .and_then(|array| array.to_vec_f32(stream))
        .map_err(mlx)
}

fn mlx(error: danmo_mlx::Error) -> WeightError {
    WeightError::NotBuilt(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::route_lora_module;
    use crate::weights::LoraRoute;

    #[test]
    fn slot_tensors_are_skipped_and_ff_names_remap() {
        assert!(matches!(
            route_lora_module("reference_slot_embedding.frequencies").unwrap(),
            LoraRoute::Skip
        ));
        assert!(matches!(
            route_lora_module("diffusion_model.reference_audio_slot_embedding.net.0").unwrap(),
            LoraRoute::Skip
        ));
        match route_lora_module("transformer_blocks.0.ff.net.0.proj").unwrap() {
            LoraRoute::One(name) => assert_eq!(name, "transformer_blocks.0.ff.proj_in"),
            other => panic!("unexpected {other:?}"),
        }
        match route_lora_module("transformer_blocks.0.ff.net.2").unwrap() {
            LoraRoute::One(name) => assert_eq!(name, "transformer_blocks.0.ff.proj_out"),
            other => panic!("unexpected {other:?}"),
        }
        match route_lora_module("transformer_blocks.0.audio_ff.net.2").unwrap() {
            LoraRoute::One(name) => assert_eq!(name, "transformer_blocks.0.audio_ff.proj_out"),
            other => panic!("unexpected {other:?}"),
        }
        match route_lora_module("transformer_blocks.0.attn1.to_out.0").unwrap() {
            LoraRoute::One(name) => assert_eq!(name, "transformer_blocks.0.attn1.to_out"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
