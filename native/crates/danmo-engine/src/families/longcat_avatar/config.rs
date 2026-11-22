//! LongCat-Video-Avatar 1.5 bundle and DiT configuration. Values are read
//! from the installed component `config.json` files; nothing is hardcoded to
//! a specific quantization variant.

use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub caption_channels: usize,
    pub adaln_tembed_dim: usize,
    pub frequency_embedding_size: usize,
    pub patch_size: [usize; 3],
    pub text_tokens_zero_pad: bool,
    pub audio_window: usize,
    pub audio_block: usize,
    pub audio_channel: usize,
    pub intermediate_dim: usize,
    pub output_dim: usize,
    pub context_tokens: usize,
    pub vae_scale: usize,
    pub audio_prenorm: bool,
    pub quantization: Option<danmo_mlx::AffineQuantization>,
}

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub z_dim: usize,
    pub base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BundleConfig {
    pub dit: DitConfig,
    pub vae: VaeConfig,
}

impl DitConfig {
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let integer = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let patch = value
            .get("patch_size")
            .and_then(Value::as_array)
            .map(|items| {
                let mut out = [1usize, 2, 2];
                for (index, item) in items.iter().take(3).enumerate() {
                    out[index] = item.as_u64().unwrap_or(out[index] as u64) as usize;
                }
                out
            })
            .unwrap_or([1, 2, 2]);
        if patch[0] != 1 {
            return Err(format!(
                "longcat_avatar DiT patch_size {patch:?} has a temporal patch != 1"
            ));
        }
        let quantization = value.get("quantization").and_then(|quant| {
            let bits = quant.get("bits")?.as_u64()? as i32;
            let group_size = quant
                .get("group_size")
                .and_then(Value::as_u64)
                .unwrap_or(64) as i32;
            Some(danmo_mlx::AffineQuantization { bits, group_size })
        });
        Ok(Self {
            in_channels: integer("in_channels", 16),
            out_channels: integer("out_channels", 16),
            hidden_size: integer("hidden_size", 4096),
            depth: integer("depth", 48),
            num_heads: integer("num_heads", 32),
            caption_channels: integer("caption_channels", 4096),
            adaln_tembed_dim: integer("adaln_tembed_dim", 512),
            frequency_embedding_size: integer("frequency_embedding_size", 256),
            patch_size: patch,
            text_tokens_zero_pad: value
                .get("text_tokens_zero_pad")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            audio_window: integer("audio_window", 5),
            audio_block: integer("audio_block", 5),
            audio_channel: integer("audio_channel", 1280),
            intermediate_dim: integer("intermediate_dim", 512),
            output_dim: integer("output_dim", 768),
            context_tokens: integer("context_tokens", 32),
            vae_scale: integer("vae_scale", 4),
            audio_prenorm: value
                .get("audio_prenorm")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            quantization,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

impl VaeConfig {
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let floats = |key: &str| -> Result<Vec<f32>, String> {
            let items = value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("longcat_avatar VAE config missing {key}"))?;
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(
                    item.as_f64()
                        .ok_or_else(|| format!("longcat_avatar VAE {key} entry is not a float"))?
                        as f32,
                );
            }
            Ok(out)
        };
        let usize_list = |key: &str| -> Result<Vec<usize>, String> {
            let items = value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("longcat_avatar VAE config missing {key}"))?;
            Ok(items
                .iter()
                .map(|item| item.as_u64().unwrap_or(1) as usize)
                .collect())
        };
        let bool_list = |key: &str| -> Result<Vec<bool>, String> {
            let items = value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("longcat_avatar VAE config missing {key}"))?;
            Ok(items
                .iter()
                .map(|item| item.as_bool().unwrap_or(false))
                .collect())
        };
        let z_dim = value.get("z_dim").and_then(Value::as_u64).unwrap_or(16) as usize;
        let mean = floats("latents_mean")?;
        let std = floats("latents_std")?;
        if mean.len() != z_dim || std.len() != z_dim {
            return Err(format!(
                "longcat_avatar VAE latents_mean/std must have {z_dim} entries"
            ));
        }
        Ok(Self {
            z_dim,
            base_dim: value.get("base_dim").and_then(Value::as_u64).unwrap_or(96) as usize,
            dim_mult: usize_list("dim_mult")?,
            num_res_blocks: value
                .get("num_res_blocks")
                .and_then(Value::as_u64)
                .unwrap_or(2) as usize,
            temperal_downsample: bool_list("temperal_downsample")?,
            latents_mean: mean,
            latents_std: std,
        })
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

impl BundleConfig {
    /// Read the DiT and VAE `config.json` files from an installed avatar
    /// bundle. Missing components fail loud with their expected paths.
    pub fn load(bundle_root: &Path) -> Result<Self, String> {
        for dir in ["dit", "vae", "text_encoder", "audio_encoder", "tokenizer"] {
            let path = bundle_root.join(dir);
            if !path.is_dir() {
                return Err(format!(
                    "longcat_avatar bundle is missing {}/ under {}",
                    dir,
                    bundle_root.display()
                ));
            }
        }
        let dit = DitConfig::from_json(&read_json(&bundle_root.join("dit/config.json"))?)?;
        let vae = VaeConfig::from_json(&read_json(&bundle_root.join("vae/config.json"))?)?;
        for file in [
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model.safetensors.index.json",
            "audio_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            let path = bundle_root.join(file);
            if !path.is_file() {
                return Err(format!(
                    "longcat_avatar bundle is missing {}",
                    path.display()
                ));
            }
        }
        Ok(Self { dit, vae })
    }
}
