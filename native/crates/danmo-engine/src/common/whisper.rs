//! Neutral Whisper-large-v3 encoder-only frontend.
//!
//! Produces the stack of all encoder hidden states (input embedding + one per
//! transformer layer) that audio-conditioned video families consume. Grouping,
//! pooling, and projection stay with the family; this module only runs the
//! encoder on a mel spectrogram.
//!
//! Weight keys follow HF `WhisperEncoder` (Meituan ships them verbatim):
//! `conv1.weight` `[d_model, 3, n_mel]`, `conv2.weight`
//! `[d_model, 3, d_model]` stride 2, `embed_positions.weight`,
//! `layers.{i}.self_attn_{layer_norm,layer_norm,layer_norm}` …
//! `self_attn.{q_proj,k_proj(no bias),v_proj,out_proj}`,
//! `final_layer_norm`, `fc1`, `fc2`, plus `layer_norm`.

use danmo_mlx::{Array, Error, Result, Stream};
use serde_json::Value;
use std::path::Path;

use crate::weights::{load_mlx_file, MlxWeights, WeightDtype};

#[derive(Debug, Clone, Copy)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub mel_bins: usize,
    pub max_source_positions: usize,
}

impl WhisperConfig {
    /// Parse an HF `config.json` (`WhisperConfig`). Missing fields fail loud.
    pub fn from_json(config: &Value) -> Result<Self> {
        let integer = |key: &str| -> Result<usize> {
            config
                .get(key)
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| Error::new("Whisper config", format!("missing {key}")))
        };
        Ok(Self {
            d_model: integer("d_model")?,
            layers: integer("encoder_layers")?,
            heads: integer("encoder_attention_heads")?,
            ffn: integer("encoder_ffn_dim")?,
            mel_bins: integer("num_mel_bins")?,
            max_source_positions: integer("max_source_positions")?,
        })
    }
}

pub struct WhisperEncoder {
    config: WhisperConfig,
    weights: MlxWeights,
}

impl WhisperEncoder {
    /// Load `audio_encoder/{config.json,model.safetensors}` from an installed
    /// component directory. The tensor dtype follows the checkpoint.
    pub fn open(component_dir: &Path) -> Result<Self> {
        let config_path = component_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|error| {
            Error::new(
                "Whisper config",
                format!("{}: {error}", config_path.display()),
            )
        })?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| Error::new("Whisper config", error.to_string()))?;
        let config = WhisperConfig::from_json(&value)?;
        let weights_path = component_dir.join("model.safetensors");
        if !weights_path.is_file() {
            return Err(Error::new(
                "Whisper weights",
                format!("{} is missing", weights_path.display()),
            ));
        }
        let weights = load_mlx_file(&weights_path, WeightDtype::Native, None)?;
        let encoder = Self { config, weights };
        encoder.assert_topology()?;
        Ok(encoder)
    }

    /// Strict shape check so a swapped component fails at load, not mid-run.
    fn assert_topology(&self) -> Result<()> {
        let d = self.config.d_model as i32;
        let mel = self.config.mel_bins as i32;
        let positions = self.weights.tensor("embed_positions.weight")?.shape();
        if positions != [self.config.max_source_positions as i32, d] {
            return Err(Error::new(
                "Whisper topology",
                format!(
                    "embed_positions is {positions:?}, expected [{}, {d}]",
                    self.config.max_source_positions
                ),
            ));
        }
        let conv1 = self.weights.tensor("conv1.weight")?.shape();
        if conv1 != [d, 3, mel] {
            return Err(Error::new(
                "Whisper topology",
                format!("conv1.weight is {conv1:?}, expected [{d}, 3, {mel}]"),
            ));
        }
        for layer in 0..self.config.layers {
            let prefix = format!("layers.{layer}");
            for suffix in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.out_proj.weight",
                "fc1.weight",
                "fc2.weight",
            ] {
                let name = format!("{prefix}.{suffix}");
                if !self.weights.has(&name) {
                    return Err(Error::new(
                        "Whisper topology",
                        format!("missing tensor {name}"),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn config(&self) -> WhisperConfig {
        self.config
    }

    /// Encode a mel spectrogram `[B, mel_bins, T_mel]` into every encoder
    /// hidden state, in HF order: index 0 is the pre-transformer embedding
    /// (conv stem + positions), indices 1..=layers are each block output
    /// (pre final LayerNorm).
    pub fn encode_all_hidden_states(
        &self,
        mel_features: &Array,
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let shape = mel_features.shape();
        if shape.len() != 3 || shape[1] != self.config.mel_bins as i32 {
            return Err(Error::new(
                "Whisper encode",
                format!(
                    "mel must be [B, {}, T_mel], got {shape:?}",
                    self.config.mel_bins
                ),
            ));
        }
        // channels-second [B, C, T] -> channels-last [B, T, C] for conv1d.
        let t_mel = mel_features.shape()[2];
        let mut x = mel_features.transpose_axes(&[0, 2, 1], stream)?;
        x = self.conv1d("conv1", &x, 1, stream)?;
        x = crate::common::mlx_ops::gelu(&x, stream)?;
        x = self.conv1d("conv2", &x, 2, stream)?;
        x = crate::common::mlx_ops::gelu(&x, stream)?;
        // conv2 (stride 2, pad 1) yields ceil(T_mel/2) frames; the 50 Hz
        // audio grid is exactly floor(T_mel/2) — drop the stride tail.
        let t_audio = t_mel / 2;
        let t_enc = x.shape()[1];
        if t_enc < t_audio {
            return Err(Error::new(
                "Whisper encode",
                format!(
                    "conv stem produced {t_enc} frames from T_mel={t_mel}; expected at least T_mel/2={t_audio}"
                ),
            ));
        }
        x = x.slice(
            &[0, 0, 0],
            &[x.shape()[0], t_audio, x.shape()[2]],
            &[1, 1, 1],
            stream,
        )?;
        let positions = self.weights.tensor("embed_positions.weight")?.slice(
            &[0, 0],
            &[t_audio, self.config.d_model as i32],
            &[1, 1],
            stream,
        )?;
        x = Array::add(&x, &positions, stream)?;
        let mut hidden = vec![x.clone()];
        for layer in 0..self.config.layers {
            x = self.layer(layer, &x, stream)?;
            hidden.push(x.clone());
        }
        Ok(hidden)
    }

    /// `weight` is PyTorch `[out, kernel, in]`; conv1d consumes
    /// channels-last `[out, kernel, in]` directly on `[B, T, in]` input.
    /// Symmetric pad 1 (both conv1 and conv2 have kernel 3), matching HF.
    fn conv1d(&self, prefix: &str, input: &Array, stride: i32, stream: &Stream) -> Result<Array> {
        let padded = input.pad(&[1], &[1], &[1], 0.0, "constant", stream)?;
        let mut output = Array::conv1d(
            &padded,
            self.weights.tensor(&format!("{prefix}.weight"))?,
            stride,
            0,
            1,
            1,
            stream,
        )?;
        let bias_name = format!("{prefix}.bias");
        if self.weights.has(&bias_name) {
            output = Array::add(&output, self.weights.tensor(&bias_name)?, stream)?;
        }
        Ok(output)
    }

    fn layer(&self, index: usize, input: &Array, stream: &Stream) -> Result<Array> {
        let prefix = format!("layers.{index}");
        let normed = self.layer_norm(&format!("{prefix}.self_attn_layer_norm"), input, stream)?;
        let attended = self.attention(&prefix, &normed, stream)?;
        let hidden = Array::add(input, &attended, stream)?;
        let normed = self.layer_norm(&format!("{prefix}.final_layer_norm"), &hidden, stream)?;
        let mid = self
            .linear(&format!("{prefix}.fc1"), &normed, stream)?
            .pipe(|x| crate::common::mlx_ops::gelu(&x, stream))?;
        let out = self.linear(&format!("{prefix}.fc2"), &mid, stream)?;
        Array::add(&hidden, &out, stream)
    }

    fn layer_norm(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let weight = self.weights.tensor(&format!("{prefix}.weight"))?;
        let bias = self.weights.tensor(&format!("{prefix}.bias"))?;
        let normed = input.layer_norm(1e-5, stream)?;
        let scaled = Array::multiply(&normed, weight, stream)?;
        Array::add(&scaled, bias, stream)
    }

    fn linear(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let weight = self.weights.linear(&format!("{prefix}.weight"))?;
        let bias_name = format!("{prefix}.bias");
        let bias = if self.weights.has(&bias_name) {
            Some(self.weights.tensor(&bias_name)?)
        } else {
            None
        };
        weight.apply(input, bias, stream)
    }

    fn attention(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input.shape();
        let (batch, sequence) = (shape[0], shape[1]);
        let heads = self.config.heads as i32;
        let head_dim = (self.config.d_model / self.config.heads) as i32;
        let split = |x: &Array| -> Result<Array> {
            x.reshape(&[batch, sequence, heads, head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)
        };
        let q = split(&self.linear(&format!("{prefix}.self_attn.q_proj"), input, stream)?)?;
        let k = split(&self.linear(&format!("{prefix}.self_attn.k_proj"), input, stream)?)?;
        let v = split(&self.linear(&format!("{prefix}.self_attn.v_proj"), input, stream)?)?;
        let output = Array::scaled_dot_product_attention(
            &q.astype(danmo_mlx::Dtype::Float32, stream)?,
            &k.astype(danmo_mlx::Dtype::Float32, stream)?,
            &v.astype(danmo_mlx::Dtype::Float32, stream)?,
            1.0 / (head_dim as f32).sqrt(),
            "",
            None,
            stream,
        )?
        .astype(input.dtype(), stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, self.config.d_model as i32], stream)?;
        self.linear(&format!("{prefix}.self_attn.out_proj"), &output, stream)
    }
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> Result<T>) -> Result<T> {
        f(self)
    }
}
impl Pipe for Array {}
