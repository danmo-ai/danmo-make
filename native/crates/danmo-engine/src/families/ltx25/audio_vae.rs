//! LTX 2.5 causal audio VAE (encode log-mel → latent, decode latent → log-mel).

use std::path::Path;

use danmo_mlx::{Array, Dtype, Result, Stream};
use serde_json::Value;

use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

pub struct AudioDecoder {
    weights: MlxWeights,
    ch: i32,
    out_channels: i32,
    ch_mult: Vec<i32>,
    res_blocks: usize,
    attention_resolutions: Vec<i32>,
    resolution: i32,
    z_channels: i32,
    norm_type: String,
    causality: String,
    mid_attention: bool,
    mel_bins: i32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl AudioDecoder {
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        let weights = load_mlx_file(
            &bundle.join("audio_vae.safetensors"),
            WeightDtype::Native,
            None,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let dd = config
            .get("ddconfig")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        let ch = integer(&dd, "ch", 128) as i32;
        let ch_mult = integers(&dd, "ch_mult", &[1, 2, 4, 8]);
        let base = ch * ch_mult.last().copied().unwrap_or(8);
        let resolution = integer(&dd, "resolution", 256) as i32;
        let z_channels = integer(&dd, "z_channels", 8) as i32;
        let mid_attention = boolean(&dd, "mid_block_add_attention", true);
        let expected = weights
            .tensor("audio_vae.decoder.conv_in.conv.weight")
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?
            .shape();
        if expected != [base, 3, 3, z_channels] {
            return Err(WeightError::NotBuilt(format!(
                "audio VAE config/weight mismatch: expected conv_in [{base},3,3,{z_channels}], got {expected:?}"
            )));
        }
        let _ = resolution;
        let _ = mid_attention;
        let mean = tensor_f32(&weights, "audio_vae.per_channel_statistics.mean-of-means")?;
        let std = tensor_f32(&weights, "audio_vae.per_channel_statistics.std-of-means")?;
        Ok(Self {
            weights,
            ch,
            out_channels: integer(&dd, "out_ch", 2) as i32,
            ch_mult,
            res_blocks: integer(&dd, "num_res_blocks", 2) as usize,
            attention_resolutions: integers(&dd, "attn_resolutions", &[8, 16, 32]),
            resolution,
            z_channels,
            norm_type: string(&dd, "norm_type", "pixel"),
            causality: string(&dd, "causality_axis", "height"),
            mid_attention,
            mel_bins: integer(&dd, "mel_bins", integer(config, "mel_bins", 64)) as i32,
            mean,
            std,
        })
    }

    /// `[B,8,T,16]` latent to `[B,2,4T-3,mel_bins]` log-mel.
    pub fn decode(&self, latent: &Array, stream: &Stream) -> Result<Array> {
        let shape = latent.shape();
        let (channels, time) = (shape[1], shape[2]);
        let patched =
            super::transformer::patchify_audio(&latent.astype(Dtype::BFloat16, stream)?, stream)?;
        let mean = Array::from_f32_slice(&self.mean, &[1, 1, self.mean.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let std = Array::from_f32_slice(&self.std, &[1, 1, self.std.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let patched = Array::add(&Array::multiply(&patched, &std, stream)?, &mean, stream)?;
        let mut x = super::transformer::unpatchify_audio(&patched, channels, stream)?;
        x = self.conv("audio_vae.decoder.conv_in", &x, 3, 1, 1, stream)?;
        x = self.resnet("audio_vae.decoder.mid.block_1", &x, stream)?;
        if self.mid_attention {
            x = self.attention("audio_vae.decoder.mid.attn_1", &x, stream)?;
        }
        x = self.resnet("audio_vae.decoder.mid.block_2", &x, stream)?;
        let mut block_in = self.ch * self.ch_mult.last().copied().unwrap_or(8);
        let mut current_resolution = self.resolution / 2_i32.pow(self.ch_mult.len() as u32 - 1);
        for level in (0..self.ch_mult.len()).rev() {
            let block_out = self.ch * self.ch_mult[level];
            for block in 0..self.res_blocks + 1 {
                let prefix = format!("audio_vae.decoder.up.{level}.block.{block}");
                x = self.resnet(&prefix, &x, stream)?;
                block_in = block_out;
                if self.attention_resolutions.contains(&current_resolution) {
                    x = self.attention(
                        &format!("audio_vae.decoder.up.{level}.attn.{block}"),
                        &x,
                        stream,
                    )?;
                }
            }
            if level != 0 {
                x = x.repeat_axis(2, 2, stream)?.repeat_axis(2, 3, stream)?;
                x = self.conv(
                    &format!("audio_vae.decoder.up.{level}.upsample.conv"),
                    &x,
                    3,
                    1,
                    1,
                    stream,
                )?;
                x = match self.causality.as_str() {
                    "height" => crop(&x, 2, 1, x.shape()[2], stream)?,
                    "width" | "width_compatibility" => crop(&x, 3, 1, x.shape()[3], stream)?,
                    _ => x,
                };
                current_resolution *= 2;
            }
        }
        x = self
            .norm("audio_vae.decoder.norm_out", &x, block_in, stream)?
            .silu(stream)?;
        x = self.conv("audio_vae.decoder.conv_out", &x, 3, 1, 1, stream)?;
        let target_time = if self.causality == "none" {
            time * 4
        } else {
            time * 4 - 3
        };
        let current_time = x.shape()[2];
        let current_freq = x.shape()[3];
        x = crop(&x, 2, 0, current_time.min(target_time), stream)?;
        x = crop(&x, 3, 0, current_freq.min(self.mel_bins), stream)?;
        let pad_time = target_time - x.shape()[2];
        let pad_freq = self.mel_bins - x.shape()[3];
        if pad_time > 0 || pad_freq > 0 {
            x = x.pad(
                &[2, 3],
                &[0, 0],
                &[pad_time, pad_freq],
                0.0,
                "constant",
                stream,
            )?;
        }
        Ok(x)
    }

    fn resnet(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let input_channels = input.shape()[1];
        let output_channels = self
            .weights
            .tensor(&format!("{prefix}.conv1.conv.weight"))?
            .shape()
            .first()
            .copied()
            .unwrap_or(input_channels);
        let mut hidden = self.norm(&format!("{prefix}.norm1"), input, input_channels, stream)?;
        hidden = self.conv(
            &format!("{prefix}.conv1"),
            &hidden.silu(stream)?,
            3,
            1,
            1,
            stream,
        )?;
        hidden = self.norm(&format!("{prefix}.norm2"), &hidden, output_channels, stream)?;
        hidden = self.conv(
            &format!("{prefix}.conv2"),
            &hidden.silu(stream)?,
            3,
            1,
            1,
            stream,
        )?;
        let shortcut = if input_channels != output_channels {
            self.conv(&format!("{prefix}.nin_shortcut"), input, 1, 1, 1, stream)?
        } else {
            input.clone()
        };
        Array::add(&shortcut, &hidden, stream)
    }

    fn attention(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input.shape();
        let (b, c, t, f) = (shape[0], shape[1], shape[2], shape[3]);
        let normed = self.norm(&format!("{prefix}.norm"), input, c, stream)?;
        let project = |name: &str| -> Result<Array> {
            self.conv(&format!("{prefix}.{name}"), &normed, 1, 1, 1, stream)?
                .reshape(&[b, c, t * f], stream)?
                .transpose_axes(&[0, 2, 1], stream)
        };
        let q = project("q")?.expand_dims(1, stream)?;
        let k = project("k")?.expand_dims(1, stream)?;
        let v = project("v")?.expand_dims(1, stream)?;
        let output = Array::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (c as f32).sqrt(),
            "array",
            None,
            stream,
        )?
        .reshape(&[b, t * f, c], stream)?
        .transpose_axes(&[0, 2, 1], stream)?
        .reshape(&[b, c, t, f], stream)?;
        let output = self.conv(&format!("{prefix}.proj_out"), &output, 1, 1, 1, stream)?;
        Array::add(input, &output, stream)
    }

    fn norm(&self, prefix: &str, input: &Array, channels: i32, stream: &Stream) -> Result<Array> {
        if self.norm_type == "group" {
            audio_group_norm(&self.weights, prefix, input, channels, stream)
        } else {
            audio_pixel_norm(input, stream)
        }
    }

    fn conv(
        &self,
        prefix: &str,
        input: &Array,
        kernel: i32,
        stride: i32,
        dilation: i32,
        stream: &Stream,
    ) -> Result<Array> {
        audio_conv2d(
            &self.weights,
            prefix,
            input,
            kernel,
            stride,
            dilation,
            &self.causality,
            stream,
        )
    }
}

/// Causal audio VAE encoder: log-mel `[B,2,T,64]` → normalized tokens `[B,T',128]`.
pub struct AudioEncoder {
    weights: MlxWeights,
    ch: i32,
    ch_mult: Vec<i32>,
    res_blocks: usize,
    causality: String,
    double_z: bool,
    z_channels: i32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl AudioEncoder {
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        let weights = load_mlx_file(
            &bundle.join("audio_vae.safetensors"),
            WeightDtype::Native,
            None,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let dd = config
            .get("ddconfig")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        let ch = integer(&dd, "ch", 128) as i32;
        let ch_mult = integers(&dd, "ch_mult", &[1, 2, 4]);
        let z_channels = integer(&dd, "z_channels", 8) as i32;
        let double_z = boolean(&dd, "double_z", true);
        let expected_out = if double_z { 2 * z_channels } else { z_channels };
        let expected = weights
            .tensor("audio_vae.encoder.conv_out.conv.weight")
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?
            .shape();
        let block_in = ch * ch_mult.last().copied().unwrap_or(4);
        if expected != [expected_out, 3, 3, block_in] {
            return Err(WeightError::NotBuilt(format!(
                "audio VAE encoder conv_out shape mismatch: expected [{expected_out},3,3,{block_in}], got {expected:?}"
            )));
        }
        let mean = tensor_f32(&weights, "audio_vae.per_channel_statistics.mean-of-means")?;
        let std = tensor_f32(&weights, "audio_vae.per_channel_statistics.std-of-means")?;
        Ok(Self {
            weights,
            ch,
            ch_mult,
            res_blocks: integer(&dd, "num_res_blocks", 2) as usize,
            causality: string(&dd, "causality_axis", "height"),
            double_z,
            z_channels,
            mean,
            std,
        })
    }

    /// Encode planar log-mel `[B,2,T,mel]` into DiT audio tokens `[B,T',128]`.
    pub fn encode_tokens(&self, mel: &Array, stream: &Stream) -> Result<Array> {
        let latent = self.encode_latent(mel, stream)?;
        self.normalize_to_tokens(&latent, stream)
    }

    /// CNN encode only: `[B,2,T,mel]` → `[B,8,T',16]` (Gaussian mode).
    pub fn encode_latent(&self, mel: &Array, stream: &Stream) -> Result<Array> {
        let shape = mel.shape();
        if shape.len() != 4 || shape[1] != 2 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 audio VAE",
                format!("expected log-mel [B,2,T,mel], got {shape:?}"),
            ));
        }
        let mut x = mel.astype(Dtype::BFloat16, stream)?;
        x = self.conv("audio_vae.encoder.conv_in", &x, 3, 1, 1, stream)?;
        for level in 0..self.ch_mult.len() {
            for block in 0..self.res_blocks {
                let prefix = format!("audio_vae.encoder.down.{level}.block.{block}");
                x = self.resnet(&prefix, &x, stream)?;
            }
            if level + 1 != self.ch_mult.len() {
                x = self.downsample(
                    &format!("audio_vae.encoder.down.{level}.downsample"),
                    &x,
                    stream,
                )?;
            }
        }
        x = self.resnet("audio_vae.encoder.mid.block_1", &x, stream)?;
        x = self.resnet("audio_vae.encoder.mid.block_2", &x, stream)?;
        let channels = x.shape()[1];
        x = audio_pixel_norm(&x, stream)?.silu(stream)?;
        x = self.conv("audio_vae.encoder.conv_out", &x, 3, 1, 1, stream)?;
        let _ = channels;
        if self.double_z {
            let total = x.shape()[1];
            if total != 2 * self.z_channels {
                return Err(danmo_mlx::Error::new(
                    "LTX 2.5 audio VAE",
                    format!(
                        "double_z encoder out channels {total} != {}",
                        2 * self.z_channels
                    ),
                ));
            }
            // DiagonalGaussian mode = mean half.
            x = x.slice(
                &[0, 0, 0, 0],
                &[x.shape()[0], self.z_channels, x.shape()[2], x.shape()[3]],
                &[1, 1, 1, 1],
                stream,
            )?;
        }
        Ok(x)
    }

    fn normalize_to_tokens(&self, latent: &Array, stream: &Stream) -> Result<Array> {
        let patched =
            super::transformer::patchify_audio(&latent.astype(Dtype::BFloat16, stream)?, stream)?;
        let mean = Array::from_f32_slice(&self.mean, &[1, 1, self.mean.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let std = Array::from_f32_slice(&self.std, &[1, 1, self.std.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        Array::divide(&Array::subtract(&patched, &mean, stream)?, &std, stream)
    }

    fn downsample(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        // Causal height downsample: pad (left=0,right=1,top=2,bottom=0), stride 2.
        let padded = input.pad(&[2, 3], &[2, 0], &[0, 1], 0.0, "constant", stream)?;
        let output = Array::conv2d(
            &padded.transpose_axes(&[0, 2, 3, 1], stream)?,
            self.weights.tensor(&format!("{prefix}.conv.weight"))?,
            [2, 2],
            [0, 0],
            [1, 1],
            1,
            stream,
        )?;
        let bias_name = format!("{prefix}.conv.bias");
        let output = if self.weights.has(&bias_name) {
            Array::add(&output, self.weights.tensor(&bias_name)?, stream)?
        } else {
            output
        };
        output.transpose_axes(&[0, 3, 1, 2], stream)
    }

    fn resnet(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let input_channels = input.shape()[1];
        let output_channels = self
            .weights
            .tensor(&format!("{prefix}.conv1.conv.weight"))?
            .shape()
            .first()
            .copied()
            .unwrap_or(input_channels);
        let mut hidden = audio_pixel_norm(input, stream)?;
        hidden = self.conv(
            &format!("{prefix}.conv1"),
            &hidden.silu(stream)?,
            3,
            1,
            1,
            stream,
        )?;
        hidden = audio_pixel_norm(&hidden, stream)?;
        hidden = self.conv(
            &format!("{prefix}.conv2"),
            &hidden.silu(stream)?,
            3,
            1,
            1,
            stream,
        )?;
        let shortcut = if input_channels != output_channels {
            self.conv(&format!("{prefix}.nin_shortcut"), input, 1, 1, 1, stream)?
        } else {
            input.clone()
        };
        Array::add(&shortcut, &hidden, stream)
    }

    fn conv(
        &self,
        prefix: &str,
        input: &Array,
        kernel: i32,
        stride: i32,
        dilation: i32,
        stream: &Stream,
    ) -> Result<Array> {
        audio_conv2d(
            &self.weights,
            prefix,
            input,
            kernel,
            stride,
            dilation,
            &self.causality,
            stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn audio_conv2d(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    kernel: i32,
    stride: i32,
    dilation: i32,
    causality: &str,
    stream: &Stream,
) -> Result<Array> {
    let pad = (kernel - 1) * dilation;
    let (left, right, top, bottom) = match causality {
        "none" => (pad / 2, pad - pad / 2, pad / 2, pad - pad / 2),
        "width" | "width_compatibility" => (pad, 0, pad / 2, pad - pad / 2),
        "height" => (pad / 2, pad - pad / 2, pad, 0),
        other => {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 audio VAE",
                format!("invalid causality {other}"),
            ))
        }
    };
    let padded = input.pad(
        &[2, 3],
        &[top, left],
        &[bottom, right],
        0.0,
        "constant",
        stream,
    )?;
    let output = Array::conv2d(
        &padded.transpose_axes(&[0, 2, 3, 1], stream)?,
        weights.tensor(&format!("{prefix}.conv.weight"))?,
        [stride, stride],
        [0, 0],
        [dilation, dilation],
        1,
        stream,
    )?;
    let bias_name = format!("{prefix}.conv.bias");
    let output = if weights.has(&bias_name) {
        Array::add(&output, weights.tensor(&bias_name)?, stream)?
    } else {
        output
    };
    output.transpose_axes(&[0, 3, 1, 2], stream)
}

fn audio_group_norm(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    channels: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, c, t, f) = (shape[0], shape[1], shape[2], shape[3]);
    let grouped = input
        .reshape(&[b, 32, c / 32, t, f], stream)?
        .astype(Dtype::Float32, stream)?;
    let mean = grouped.mean_axes(&[2, 3, 4], true, stream)?;
    let centered = Array::subtract(&grouped, &mean, stream)?;
    let variance =
        Array::multiply(&centered, &centered, stream)?.mean_axes(&[2, 3, 4], true, stream)?;
    let denominator = Array::add(&variance, &Array::scalar_f32(1e-6)?, stream)?.sqrt(stream)?;
    let normalized = Array::divide(&centered, &denominator, stream)?
        .reshape(&[b, c, t, f], stream)?
        .astype(input.dtype(), stream)?;
    let weight = weights
        .tensor(&format!("{prefix}.weight"))?
        .reshape(&[1, channels, 1, 1], stream)?;
    let bias = weights
        .tensor(&format!("{prefix}.bias"))?
        .reshape(&[1, channels, 1, 1], stream)?;
    Array::add(
        &Array::multiply(&normalized, &weight, stream)?,
        &bias,
        stream,
    )
}

fn audio_pixel_norm(input: &Array, stream: &Stream) -> Result<Array> {
    let square = Array::multiply(input, input, stream)?.astype(Dtype::Float32, stream)?;
    let mean = square.mean_axes(&[1], true, stream)?;
    let denominator = Array::add(&mean, &Array::scalar_f32(1e-6)?, stream)?
        .sqrt(stream)?
        .astype(input.dtype(), stream)?;
    Array::divide(input, &denominator, stream)
}

fn crop(input: &Array, axis: usize, start: i32, stop: i32, stream: &Stream) -> Result<Array> {
    let mut starts = vec![0; input.shape().len()];
    let mut stops = input.shape();
    starts[axis] = start;
    stops[axis] = stop;
    input.slice(&starts, &stops, &vec![1; starts.len()], stream)
}

fn tensor_f32(weights: &MlxWeights, name: &str) -> std::result::Result<Vec<f32>, WeightError> {
    let stream = Stream::cpu().map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    weights
        .tensor(name)
        .and_then(|tensor| tensor.astype(Dtype::Float32, &stream))
        .and_then(|tensor| tensor.to_vec_f32(&stream))
        .map_err(|error| WeightError::NotBuilt(error.to_string()))
}
fn integer(value: &Value, key: &str, default: u64) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(default)
}
fn boolean(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}
fn string(value: &Value, key: &str, default: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}
fn integers(value: &Value, key: &str, default: &[i32]) -> Vec<i32> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_i64)
                .map(|v| v as i32)
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn real_audio_decoder_produces_finite_log_mel() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let decoder = super::AudioDecoder::load(&root, &config.audio_vae).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let fixture_path = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_REFERENCE").expect("DANQING_TEST_LTX25_REFERENCE"),
        );
        let fixture =
            crate::weights::load_mlx_file(&fixture_path, crate::weights::WeightDtype::Native, None)
                .unwrap()
                .into_dense()
                .unwrap();
        let latent = fixture["audio_decoder_input"]
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let output = decoder.decode(&latent, &stream).unwrap();
        assert_eq!(output.shape(), &[1, 2, 13, 64]);
        let values = output.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        let expected = fixture["audio_decoder_output"].to_vec_f32(&stream).unwrap();
        let (max_abs, mean_abs) = values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold((0.0f32, 0.0f32), |(max, sum), value| {
                (max.max(value), sum + value / values.len() as f32)
            });
        println!("LTX25 audio decoder: max_abs={max_abs}, mean_abs={mean_abs}");
        assert!(max_abs < 0.1 && mean_abs < 0.02);
    }

    #[test]
    #[ignore]
    fn real_audio_encoder_emits_finite_tokens() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let encoder = super::AudioEncoder::load(&root, &config.audio_vae).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let wave = vec![0.01f32; 16_000 * 2];
        let (planar, time) = crate::families::ltx25::mel::stereo_log_mel(&wave).unwrap();
        let mel = danmo_mlx::Array::from_f32_slice(&planar, &[1, 2, time as i32, 64])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let tokens = encoder.encode_tokens(&mel, &stream).unwrap();
        assert_eq!(tokens.shape()[0], 1);
        assert_eq!(tokens.shape()[2], 128);
        assert!(tokens.shape()[1] > 0);
        assert!(tokens
            .astype(danmo_mlx::Dtype::Float32, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
