//! Flux2 reads the shared Qwen3 tower at layers 9, 18, and 27.
//! A shorter tower concatenates every layer when `hidden * layers` equals the
//! joint width. Thinking is off, so a template that mentions `enable_thinking`
//! appends the empty think block.

use super::weights::Flux2Config;
use crate::common::qwen3::{encoder_layout, is_rms, Loaded, Qwen3Config};
use crate::runtime::GgmlContext;
use crate::weights::{write_f32_tensors, F32Tensor, WeightError};
use std::path::Path;

pub fn encode_prompt(
    ctx: &GgmlContext,
    paths: &[std::path::PathBuf],
    tokenizer_json: &str,
    prompt: &str,
    joint: i64,
) -> Result<(Vec<f32>, i64), WeightError> {
    let loaded = Loaded::open(paths)?;
    let config = loaded.config();
    let selected = select_layers(config.layers, config.hidden, joint)?;
    loaded.encode(ctx, tokenizer_json, prompt, &selected, false, joint)
}

fn select_layers(layers: i64, hidden: i64, joint: i64) -> Result<Vec<i64>, WeightError> {
    if layers >= 27 && hidden.saturating_mul(3) == joint {
        return Ok(vec![9, 18, 27]);
    }
    if hidden.saturating_mul(layers) == joint {
        return Ok((1..=layers).collect());
    }
    Err(WeightError::NotBuilt(format!(
        "text encoder hidden {hidden} x {layers} layers is not joint {joint}"
    )))
}

pub fn write_qwen3_bundle(dir: &Path, dit: Flux2Config) -> Result<(), WeightError> {
    let enc = Qwen3Config::smoke();
    if dit.joint_attention_dim != enc.hidden * enc.layers {
        return Err(WeightError::NotBuilt(format!(
            "joint {} does not match qwen3 smoke hidden {} x {} layers",
            dit.joint_attention_dim, enc.hidden, enc.layers
        )));
    }
    super::write_config_bundle(dir, dit)?;
    write_encoder(dir, enc)?;
    let tokenizer = r#"{"vocab":{"a":0,"cat":1,"dog":2}}
"#;
    std::fs::write(dir.join("tokenizer.json"), tokenizer).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    Ok(())
}

/// Local pack the product can generate with: byte tokenizer, Qwen3 tower, tiny DiT, 1×1 VAE.
pub fn write_chain_bundle(dir: &Path) -> Result<(), WeightError> {
    let mut dit = Flux2Config::from_dim(8);
    dit.joint_attention_dim = 16;
    super::write_config_bundle(dir, dit)?;
    super::rewrite_chain_transformer(dir, dit)?;
    let mut enc = Qwen3Config::smoke();
    enc.vocab = 256;
    if dit.joint_attention_dim != enc.hidden * enc.layers {
        return Err(WeightError::NotBuilt(format!(
            "joint {} does not match qwen3 hidden {} x {} layers",
            dit.joint_attention_dim, enc.hidden, enc.layers
        )));
    }
    write_encoder(dir, enc)?;
    std::fs::write(dir.join("tokenizer.json"), "{\"encoding\":\"bytes\"}\n").map_err(|err| {
        WeightError::Open {
            path: dir.join("tokenizer.json").display().to_string(),
            detail: err.to_string(),
        }
    })?;
    let manifest = r#"{
  "format": "gguf",
  "quant": "bf16",
  "max_edge": 128,
  "pack_rev": 2,
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#;
    std::fs::write(dir.join("bundle.manifest.json"), manifest).map_err(|err| {
        WeightError::Open {
            path: dir.join("bundle.manifest.json").display().to_string(),
            detail: err.to_string(),
        }
    })?;
    Ok(())
}

fn write_encoder(dir: &Path, enc: Qwen3Config) -> Result<(), WeightError> {
    let owned: Vec<(String, i64, i64, Vec<f32>)> = encoder_layout(enc)
        .into_iter()
        .map(|(name, ne0, ne1)| {
            let data = if name == "embed_tokens.weight" {
                embed_rows(ne0, ne1)
            } else if is_rms(&name) {
                vec![1.0; (ne0 as usize) * (ne1 as usize)]
            } else {
                diagonal(ne0, ne1)
            };
            (name, ne0, ne1, data)
        })
        .collect();
    let tensors: Vec<F32Tensor<'_>> = owned
        .iter()
        .map(|(name, ne0, ne1, data)| F32Tensor {
            name,
            ne0: *ne0,
            ne1: *ne1,
            data,
        })
        .collect();
    write_f32_tensors(&dir.join("text_encoder.gguf"), &tensors)
}

fn embed_rows(hidden: i64, vocab: i64) -> Vec<f32> {
    let mut data = vec![0f32; (hidden as usize) * (vocab as usize)];
    let hidden = hidden as usize;
    for token in 0..vocab {
        let row = (token as usize) % hidden;
        let scale = 0.25 + ((token % 17) as f32) * 0.05;
        data[row + (token as usize) * hidden] = scale;
    }
    data
}

fn diagonal(ne0: i64, ne1: i64) -> Vec<f32> {
    let mut data = vec![0f32; (ne0 as usize) * (ne1 as usize)];
    let span = ne0.min(ne1) as usize;
    for i in 0..span {
        data[i + i * ne0 as usize] = 0.05;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::{encode_prompt, write_chain_bundle, write_qwen3_bundle};
    use crate::families::Flux2Config;
    use crate::runtime::{BackendKind, GgmlContext};

    #[test]
    fn cat_and_dog_prompts_embed_differently() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-qwen3-emb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut dit = Flux2Config::from_dim(8);
        dit.joint_attention_dim = 16;
        write_qwen3_bundle(&dir, dit).unwrap();
        let tokenizer = std::fs::read_to_string(dir.join("tokenizer.json")).unwrap();
        let paths = vec![dir.join("text_encoder.gguf")];
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (cat, seq) = encode_prompt(&ctx, &paths, &tokenizer, "a cat", 16).unwrap();
        let (dog, _) = encode_prompt(&ctx, &paths, &tokenizer, "a dog", 16).unwrap();
        assert_eq!(seq, 2);
        assert_eq!(cat.len(), 32);
        assert!(cat.iter().any(|v| v.abs() > 1e-4), "cat embed {cat:?}");
        assert_ne!(cat, dog);
        let err = encode_prompt(&ctx, &paths, &tokenizer, "a missing", 16).unwrap_err();
        assert!(err.to_string().contains("no id for missing"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn byte_prompts_embed_differently() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-bytes-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_chain_bundle(&dir).unwrap();
        let tokenizer = std::fs::read_to_string(dir.join("tokenizer.json")).unwrap();
        let paths = vec![dir.join("text_encoder.gguf")];
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (zh, seq) = encode_prompt(&ctx, &paths, &tokenizer, "一只猫", 16).unwrap();
        let (en, _) = encode_prompt(&ctx, &paths, &tokenizer, "a dog", 16).unwrap();
        assert!(seq > 1, "seq {seq}");
        assert_ne!(zh.len(), en.len());
        assert!(zh.iter().any(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
