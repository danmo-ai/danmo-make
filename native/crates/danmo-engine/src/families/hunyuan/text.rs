//! HunyuanVideo 1.5 prompt conditioning on the shared Qwen2.5-VL text trunk
//! (`common::qwen25vl`). main `_NUM_HIDDEN_LAYERS_TO_SKIP = 2`: the output is
//! `hidden_states[-3]` — the layer-(layers - 3) output with **no final norm** —
//! and the first `CROP_START` tokens of the rendered chat template are dropped
//! (main `prompt_template_crop_start = 108`, `mllm_max_length = 1000`).

use crate::common::qwen25vl::{self, Qwen25VlConfig, TextForward, TowerOutput};
use crate::weights::WeightError;

/// main `_HUNYUAN_SYSTEM_MESSAGE`.
pub const SYSTEM_MESSAGE: &str = "You are a helpful assistant. Describe the video by detailing the following aspects: 1. The main content and theme of the video. 2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. 3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. 4. background environment, light, style and atmosphere. 5. camera angles, movements, and transitions used in the video.";
/// main `prompt_template_crop_start`.
pub const CROP_START: usize = 108;
/// main `mllm_max_length`.
pub const MLLM_MAX_LENGTH: usize = 1000;
/// main pads to `mllm_max_length + crop_start` before the tower.
pub const MAX_LENGTH: usize = MLLM_MAX_LENGTH + CROP_START;
/// main `_NUM_HIDDEN_LAYERS_TO_SKIP`: the embedding is `hidden_states[-3]`.
pub const NUM_HIDDEN_LAYERS_TO_SKIP: usize = 2;

fn output_stage(config: &Qwen25VlConfig) -> Result<TowerOutput, WeightError> {
    let layers = config.layers as usize;
    if layers <= NUM_HIDDEN_LAYERS_TO_SKIP {
        return Err(WeightError::NotBuilt(format!(
            "Qwen2.5-VL tower has {layers} layers; hidden_states[-{}] does not exist",
            NUM_HIDDEN_LAYERS_TO_SKIP + 1
        )));
    }
    Ok(TowerOutput::AfterLayer(
        layers - 1 - NUM_HIDDEN_LAYERS_TO_SKIP,
    ))
}

/// Token ids → `(hidden, len)`: `hidden` is `[hidden_dim, len]` with the
/// hidden dim on `ne0`, `len` the token count after the 108-token crop.
pub fn encode(
    ctx: &crate::runtime::GgmlContext,
    weights: &crate::weights::GgufWeights,
    config: Qwen25VlConfig,
    ids: &[i32],
) -> Result<(Vec<f32>, usize), WeightError> {
    if ids.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    if ids.len() > MAX_LENGTH {
        return Err(WeightError::NotBuilt(format!(
            "hunyuan prompt tokenized to {} ids past the {MAX_LENGTH} cap",
            ids.len()
        )));
    }
    if !qwen25vl::layout_matches(weights, config) {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL text encoder layout does not match the tower config".into(),
        ));
    }
    let output = output_stage(&config)?;
    let seq = ids.len() as i64;
    let mut tower = TextForward::build(ctx, config, seq, output)?;
    qwen25vl::load_tower(&mut tower, weights, config, output)?;
    let hidden = tower.compute(ids, &qwen25vl::causal_mask(seq))?;
    let trimmed = qwen25vl::trim_prefix(&hidden, config.hidden as usize, ids.len(), CROP_START)?;
    Ok((trimmed, ids.len() - CROP_START))
}

#[cfg(test)]
mod tests {
    use super::{
        encode, output_stage, CROP_START, MAX_LENGTH, MLLM_MAX_LENGTH, NUM_HIDDEN_LAYERS_TO_SKIP,
        SYSTEM_MESSAGE,
    };
    use crate::common::qwen25vl::{self, Qwen25VlConfig, TowerOutput};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_safetensors, GgufWeights, SafeTensor};
    use std::path::Path;

    /// Three-layer smoke tower: skipping 2 leaves layer 0.
    fn smoke() -> Qwen25VlConfig {
        Qwen25VlConfig {
            vocab: 258,
            hidden: 8,
            layers: 3,
            heads: 4,
            kv_heads: 2,
            head_dim: 2,
            intermediate: 8,
            rope_theta: 1_000_000.0,
            mrope: [1, 0, 0],
        }
    }

    fn tap(name: &str, index: usize) -> f32 {
        let mut hash = 2166136261u32;
        for byte in name.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
        }
        hash = hash.wrapping_mul(16777619) ^ index as u32;
        ((hash % 7) as f32 - 3.0) * 0.05
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Write the tower as safetensors (PyTorch dims). Layer 0 is the identity
    /// (zero attention and MLP weights and biases, unit RMS norms); layers 1+
    /// and the final norm are random, so the output proves which layers ran.
    fn write_tower(dir: &Path, config: Qwen25VlConfig) -> std::path::PathBuf {
        let mut tensors = Vec::new();
        let mut owned = Vec::new();
        for (name, ne0, ne1) in qwen25vl::text_layout(config) {
            let layer = name
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|index| index.parse::<i64>().ok());
            let data: Vec<f32> = (0..(ne0 * ne1) as usize)
                .map(|i| match layer {
                    Some(0) if name.ends_with("layernorm.weight") => 1.0,
                    Some(0) => 0.0,
                    _ if name == "model.norm.weight" => 3.0,
                    _ => tap(&name, i),
                })
                .collect();
            owned.push((name, [ne1 as u64, ne0 as u64], f32_bytes(&data)));
        }
        for (name, shape, bytes) in &owned {
            let shape: &[u64] = if shape[1] == 1 { &shape[..1] } else { shape };
            tensors.push(SafeTensor {
                name,
                dtype: "F32",
                shape,
                data: bytes,
            });
        }
        let path = dir.join("model.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        path
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-hy-text-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn constants_match_main() {
        assert_eq!(CROP_START, 108);
        assert_eq!(MLLM_MAX_LENGTH, 1000);
        assert_eq!(MAX_LENGTH, 1108);
        assert_eq!(NUM_HIDDEN_LAYERS_TO_SKIP, 2);
        assert!(SYSTEM_MESSAGE.starts_with("You are a helpful assistant. Describe the video"));
        assert!(SYSTEM_MESSAGE.contains("camera angles, movements, and transitions"));
    }

    #[test]
    fn output_stage_is_the_third_to_last_layer() {
        let published = Qwen25VlConfig::published();
        assert_eq!(
            output_stage(&published).unwrap(),
            TowerOutput::AfterLayer(25)
        );
        let one_layer = Qwen25VlConfig::smoke();
        assert!(output_stage(&one_layer).is_err());
    }

    #[test]
    fn encode_crops_the_system_prefix() {
        let dir = temp_dir("crop");
        let config = smoke();
        let path = write_tower(&dir, config);
        let weights = GgufWeights::open_parts(&[path]).unwrap();
        // 110 tokens: crop 108 leaves the last two.
        let ids: Vec<i32> = (0..110).map(|i| i % 6).collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (hidden, len) = encode(&ctx, &weights, config, &ids).unwrap();
        assert_eq!(len, 2);
        assert_eq!(hidden.len(), 2 * config.hidden as usize);
        // Layer 0 is the identity, so the hidden state is the embedding rows.
        let embed: Vec<f32> = (0..(config.hidden * config.vocab) as usize)
            .map(|i| tap("model.embed_tokens.weight", i))
            .collect();
        for (token_index, id) in ids[108..110].iter().enumerate() {
            for h in 0..config.hidden as usize {
                let want = embed[h + config.hidden as usize * (*id as usize)];
                let got = hidden[h + config.hidden as usize * token_index];
                assert!((got - want).abs() < 1e-5, "token {token_index} dim {h}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
