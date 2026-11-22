use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct ErnieConfig {
    pub hidden: i32,
    pub layers: i32,
    pub heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub text_dim: i32,
    pub patch: i32,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_axes: [i32; 3],
}

impl ErnieConfig {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let value = read_json(&directory.join("config.json"))?;
        if value.get("_class_name").and_then(|v| v.as_str()) != Some("ErnieImageTransformer2DModel")
        {
            return Err("ERNIE transformer config is not ErnieImageTransformer2DModel".into());
        }
        let hidden = int(&value, "hidden_size")?;
        let heads = int(&value, "num_attention_heads")?;
        let head_dim = int(&value, "head_dim").unwrap_or(hidden / heads);
        let rope_axes = value
            .get("rope_axes_dim")
            .and_then(|v| v.as_array())
            .filter(|v| v.len() == 3)
            .map(|v| {
                [
                    v[0].as_i64().unwrap_or(0) as i32,
                    v[1].as_i64().unwrap_or(0) as i32,
                    v[2].as_i64().unwrap_or(0) as i32,
                ]
            })
            .ok_or_else(|| "ERNIE rope_axes_dim must have three entries".to_string())?;
        if hidden < 1 || heads < 1 || head_dim < 1 || rope_axes.iter().sum::<i32>() != head_dim {
            return Err("ERNIE transformer head and RoPE dimensions are inconsistent".into());
        }
        Ok(Self {
            hidden,
            layers: int(&value, "num_layers")?,
            heads,
            head_dim,
            intermediate: int(&value, "ffn_hidden_size")?,
            in_channels: int(&value, "in_channels")?,
            out_channels: int(&value, "out_channels")?,
            text_dim: int(&value, "text_in_dim")?,
            patch: int(&value, "patch_size")?,
            eps: number(&value, "eps", 1e-6)?,
            rope_theta: number(&value, "rope_theta", 256.0)?,
            rope_axes,
        })
    }

    #[cfg(test)]
    pub fn smoke() -> Self {
        Self {
            hidden: 8,
            layers: 2,
            heads: 1,
            head_dim: 8,
            intermediate: 16,
            in_channels: 4,
            out_channels: 4,
            text_dim: 6,
            patch: 1,
            eps: 1e-6,
            rope_theta: 256.0,
            rope_axes: [4, 2, 2],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MistralConfig {
    pub vocab: i32,
    pub hidden: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_positions: i32,
    pub attention_scale: f32,
    pub llama_4_scaling_beta: f32,
    pub eos: i32,
}

impl MistralConfig {
    pub fn load_text_encoder(directory: &Path) -> Result<Self, String> {
        let value = read_json(&directory.join("config.json"))?;
        let text = value
            .get("text_config")
            .ok_or_else(|| "Mistral3 text_config is missing".to_string())?;
        Self::from_value(text)
    }

    pub fn load_pe(directory: &Path) -> Result<Self, String> {
        let value = read_json(&directory.join("config.json"))?;
        Self::from_value(&value)
    }

    fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        let hidden = int(value, "hidden_size")?;
        let heads = int(value, "num_attention_heads")?;
        let kv_heads = int(value, "num_key_value_heads")?;
        let head_dim = int(value, "head_dim").unwrap_or(hidden / heads);
        let rope = value
            .get("rope_parameters")
            .ok_or_else(|| "Ministral3 rope_parameters are missing".to_string())?;
        if rope.get("rope_type").and_then(|v| v.as_str()) != Some("yarn") {
            return Err("Ministral3 requires YaRN rope_parameters".into());
        }
        let factor = number(rope, "factor", 1.0)?;
        let mscale = number(rope, "mscale", 1.0)?;
        let mscale_all = number(rope, "mscale_all_dim", 1.0)?;
        let scale = |factor: f32, mscale: f32| {
            if factor <= 1.0 {
                1.0
            } else {
                0.1 * mscale * factor.ln() + 1.0
            }
        };
        let attention_scale = scale(factor, mscale) / scale(factor, mscale_all);
        if hidden < 1 || heads < 1 || kv_heads < 1 || heads % kv_heads != 0 || head_dim % 2 != 0 {
            return Err("Ministral3 attention dimensions are invalid".into());
        }
        Ok(Self {
            vocab: int(value, "vocab_size")?,
            hidden,
            layers: int(value, "num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            intermediate: int(value, "intermediate_size")?,
            rms_eps: number(value, "rms_norm_eps", 1e-5)?,
            rope_theta: number(rope, "rope_theta", 1_000_000.0)?,
            rope_factor: factor,
            beta_fast: number(rope, "beta_fast", 32.0)?,
            beta_slow: number(rope, "beta_slow", 1.0)?,
            original_positions: int(rope, "original_max_position_embeddings")?,
            attention_scale,
            llama_4_scaling_beta: number(rope, "llama_4_scaling_beta", 0.1)?,
            eos: int(value, "eos_token_id").unwrap_or(2),
        })
    }

    #[cfg(test)]
    pub fn smoke() -> Self {
        Self {
            vocab: 16,
            hidden: 8,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_factor: 2.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_positions: 16,
            attention_scale: 1.0,
            llama_4_scaling_beta: 0.1,
            eos: 2,
        }
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

fn int(value: &serde_json::Value, key: &str) -> Result<i32, String> {
    value
        .get(key)
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| format!("config {key} is missing or invalid"))
}

fn number(value: &serde_json::Value, key: &str, default: f32) -> Result<f32, String> {
    match value.get(key) {
        Some(value) => value
            .as_f64()
            .map(|v| v as f32)
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("config {key} is invalid")),
        None => Ok(default),
    }
}
