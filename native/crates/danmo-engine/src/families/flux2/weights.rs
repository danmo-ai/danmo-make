//! Flux2 weight remap. Same key rules as `backend/engine/families/flux2/weights.py`.

use std::collections::BTreeMap;

/// Map diffusers Flux.2 keys onto the names stored in the ggml graph.
pub fn remap_flux2_weights<T>(weights: BTreeMap<String, T>) -> BTreeMap<String, T> {
    let mut remapped = BTreeMap::new();
    for (key, tensor) in weights {
        let new_key = key
            .replace(".to_out.0.", ".to_out.")
            .replace(".to_add_out.0.", ".to_add_out.")
            .replace(
                "time_guidance_embed.timestep_embedder.",
                "time_guidance_embed.",
            );
        remapped.insert(new_key, tensor);
    }
    remapped
}

fn lora_module(key: &str) -> String {
    for suffix in [
        ".lora_A",
        ".lora_B",
        ".lora_down",
        ".lora_up",
        "lora_A",
        "lora_B",
        "lora_down",
        "lora_up",
    ] {
        if let Some(idx) = key.find(suffix) {
            return key[..idx].trim_end_matches('.').to_string();
        }
    }
    key.to_string()
}

/// Module-name half of `remap_flux2_lora_keys`. Tensor splits stay with the caller
/// once both A and B matrices are present; this function only rewrites the target name.
pub fn remap_flux2_lora_module(module: &str) -> Option<String> {
    let mut name = module.replace("diffusion_model.", "");
    if let Some(rest) = name.strip_prefix("double_blocks.") {
        if let Some((block, tail)) = rest.split_once('.') {
            name = match tail {
                "img_attn.proj" => format!("transformer_blocks.{block}.attn.to_out"),
                "txt_attn.proj" => format!("transformer_blocks.{block}.attn.to_add_out"),
                "img_mlp.0" => format!("transformer_blocks.{block}.ff.linear_in"),
                "img_mlp.2" => format!("transformer_blocks.{block}.ff.linear_out"),
                "txt_mlp.0" => format!("transformer_blocks.{block}.ff_context.linear_in"),
                "txt_mlp.2" => format!("transformer_blocks.{block}.ff_context.linear_out"),
                _ => name,
            };
        }
    }
    if let Some(rest) = name.strip_prefix("single_blocks.") {
        if let Some((block, tail)) = rest.split_once('.') {
            if tail == "linear1" {
                name = format!("single_transformer_blocks.{block}.attn.to_qkv_mlp_proj");
            }
        }
    }
    if name.contains("modulation") && name.contains("single") {
        name = "single_stream_modulation.linear".into();
    } else if name.contains("img_mod") {
        name = "double_stream_modulation_img.linear".into();
    } else if name.contains("txt_mod") {
        name = "double_stream_modulation_txt.linear".into();
    }
    if name.contains("norm") && !name.contains("modulation") {
        return None;
    }
    Some(name)
}

pub fn load_safetensors_f32(path: &std::path::Path) -> Result<BTreeMap<String, Vec<f32>>, String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|err| err.to_string())?;
    let mut weights = BTreeMap::new();
    for name in tensors.names() {
        let tensor = tensors.tensor(name).map_err(|err| err.to_string())?;
        if tensor.dtype() != safetensors::Dtype::F32 {
            return Err(format!(
                "{name} is {:?}; flux2 load expects F32",
                tensor.dtype()
            ));
        }
        let mut values = Vec::with_capacity(tensor.data().len() / 4);
        for chunk in tensor.data().chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        weights.insert(name.to_string(), values);
    }
    Ok(remap_flux2_weights(weights))
}

pub fn remap_flux2_lora_target(key: &str) -> Option<String> {
    remap_flux2_lora_module(&lora_module(key))
}

#[derive(Debug, Clone, Copy)]
pub struct Flux2Config {
    pub in_channels: i64,
    pub inner_dim: i64,
    pub num_heads: i64,
    pub attn_head_dim: i64,
    pub num_layers: i64,
    pub num_single_layers: i64,
    pub joint_attention_dim: i64,
    pub requires_sigma_shift: bool,
    pub vae_scale: u32,
}

impl Flux2Config {
    /// 4B Klein defaults from `Flux2Config`, scaled down only when a caller passes overrides.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            inner_dim: 3072,
            num_heads: 24,
            attn_head_dim: 128,
            num_layers: 8,
            num_single_layers: 24,
            joint_attention_dim: 12288,
            requires_sigma_shift: true,
            vae_scale: 16,
        }
    }

    /// Tiny config so the denoise graph can execute on CPU without a weight bundle.
    pub fn smoke() -> Self {
        Self {
            in_channels: 4,
            inner_dim: 4,
            num_heads: 1,
            attn_head_dim: 4,
            num_layers: 1,
            num_single_layers: 1,
            joint_attention_dim: 4,
            requires_sigma_shift: true,
            vae_scale: 16,
        }
    }
}
