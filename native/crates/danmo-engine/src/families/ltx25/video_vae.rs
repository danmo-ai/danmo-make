//! Config-driven LTX 2.5 video VAE encoder and decoder.

use std::path::Path;

use danmo_mlx::{Array, Dtype, RandomKey, Result, Stream};
use serde_json::Value;

use super::conv::{
    conv3d, depth_to_space_3d, group_norm, patchify_spatial, space_to_depth_downsample,
    unpatchify_spatial,
};
use crate::common::mlx_ops::{scalar, slice_last, timestep_embedding};
use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

#[derive(Clone, Debug)]
enum DownBlock {
    ResStage {
        index: usize,
        channels: i32,
        layers: usize,
    },
    SpaceToDepth {
        index: usize,
        in_channels: i32,
        out_channels: i32,
        stride: [i32; 3],
    },
}

#[derive(Clone, Debug)]
enum UpBlock {
    ResStage {
        index: usize,
        channels: i32,
        layers: usize,
        inject_noise: bool,
        timestep_conditioning: bool,
    },
    Resample {
        index: usize,
        input_channels: i32,
        output_channels: i32,
        inject_noise: bool,
    },
    DepthToSpace {
        index: usize,
        spatial: i32,
        temporal: i32,
    },
    Attention {
        index: usize,
        channels: i32,
    },
}

pub struct VideoEncoder {
    weights: MlxWeights,
    blocks: Vec<DownBlock>,
    spatial_padding: String,
    patch: i32,
    latent_log_var: String,
    mean: Vec<f32>,
    std: Vec<f32>,
}

pub struct VideoDecoder {
    weights: MlxWeights,
    blocks: Vec<UpBlock>,
    causal: bool,
    timestep_conditioning: bool,
    spatial_padding: String,
    patch: i32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl VideoEncoder {
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        let weights = load_mlx_file(
            &bundle.join("video_vae.safetensors"),
            WeightDtype::Native,
            None,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let blocks_spec = config
            .get("encoder_blocks")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::NotBuilt("video VAE encoder_blocks missing".into()))?;
        let mut feature_channels = integer(config, "encoder_base_channels", 128) as i32;
        let mut blocks = Vec::new();
        for (index, entry) in blocks_spec.iter().enumerate() {
            let (name, spec) = block_spec(entry)?;
            match name.as_str() {
                "res_x" | "attn_res_x" => {
                    blocks.push(DownBlock::ResStage {
                        index,
                        channels: feature_channels,
                        layers: integer_value(spec, "num_layers", 1) as usize,
                    });
                }
                "compress_space_res" | "compress_time_res" | "compress_all_res" => {
                    let multiplier = integer_value(spec, "multiplier", 2) as i32;
                    let out_channels = feature_channels * multiplier;
                    let stride = match name.as_str() {
                        "compress_space_res" => [1, 2, 2],
                        "compress_time_res" => [2, 1, 1],
                        _ => [2, 2, 2],
                    };
                    blocks.push(DownBlock::SpaceToDepth {
                        index,
                        in_channels: feature_channels,
                        out_channels,
                        stride,
                    });
                    feature_channels = out_channels;
                }
                other => {
                    return Err(WeightError::NotBuilt(format!(
                        "unsupported LTX 2.5 video encoder block {other}"
                    )));
                }
            }
        }
        let mean = numbers(config, "latents_mean");
        let std = numbers(config, "latents_std");
        let mean = if mean.is_empty() {
            tensor_f32(&weights, "per_channel_statistics.mean-of-means")?
        } else {
            mean
        };
        let std = if std.is_empty() {
            tensor_f32(&weights, "per_channel_statistics.std-of-means")?
        } else {
            std
        };
        Ok(Self {
            weights,
            blocks,
            spatial_padding: string(config, "spatial_padding_mode", "zeros"),
            patch: integer(config, "patch_size", 4) as i32,
            latent_log_var: string(config, "latent_log_var", "uniform"),
            mean,
            std,
        })
    }

    /// Encode `[B,3,F,H,W]` pixels to normalized `[B,128,F',H/32,W/32]` latents.
    /// `F` must be `1 + 8k` (e.g. 1, 9, 17, 25, 33).
    pub fn encode(&self, pixels: &Array, stream: &Stream) -> Result<Array> {
        let shape = pixels.shape();
        if shape.len() != 5 || shape[1] != 3 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 video encoder",
                format!("expected [B,3,F,H,W], got {shape:?}"),
            ));
        }
        let frames = shape[2];
        if frames < 1 || (frames - 1) % 8 != 0 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 video encoder",
                format!("frame count {frames} must be 1 + 8k"),
            ));
        }
        if shape[3] % self.patch != 0 || shape[4] % self.patch != 0 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 video encoder",
                format!(
                    "spatial size {}x{} must be divisible by patch {}",
                    shape[3], shape[4], self.patch
                ),
            ));
        }
        let mut x = pixels
            .astype(Dtype::BFloat16, stream)?
            .transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        x = patchify_spatial(&x, self.patch, stream)?;
        x = conv3d(
            &self.weights,
            "encoder.conv_in.conv",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            true,
            &self.spatial_padding,
            stream,
        )?;
        // Dummy key: encoder resblocks never inject noise.
        let key = RandomKey::new(0)?;
        for block in &self.blocks {
            match *block {
                DownBlock::ResStage {
                    index,
                    channels,
                    layers,
                } => {
                    let prefix = format!("encoder.down_blocks.{index}");
                    for layer in 0..layers {
                        x = resnet(
                            &self.weights,
                            &format!("{prefix}.res_blocks.{layer}"),
                            &x,
                            channels,
                            channels,
                            None,
                            false,
                            true,
                            &self.spatial_padding,
                            &key,
                            stream,
                        )?;
                    }
                }
                DownBlock::SpaceToDepth {
                    index,
                    in_channels,
                    out_channels,
                    stride,
                } => {
                    x = space_to_depth_downsample(
                        &self.weights,
                        &format!("encoder.down_blocks.{index}.conv"),
                        &x,
                        in_channels,
                        out_channels,
                        stride,
                        &self.spatial_padding,
                        stream,
                    )?;
                }
            }
            x.eval()?;
        }
        x = x.rms_norm(None, 1e-8, stream)?.silu(stream)?;
        x = conv3d(
            &self.weights,
            "encoder.conv_out.conv",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            true,
            &self.spatial_padding,
            stream,
        )?;
        let channels = x.shape()[4];
        let means = match self.latent_log_var.as_str() {
            "uniform" | "constant" => {
                if channels < 2 {
                    return Err(danmo_mlx::Error::new(
                        "LTX 2.5 video encoder",
                        format!("uniform/constant log-var expects >=2 channels, got {channels}"),
                    ));
                }
                x.slice(
                    &[0, 0, 0, 0, 0],
                    &[
                        x.shape()[0],
                        x.shape()[1],
                        x.shape()[2],
                        x.shape()[3],
                        channels - 1,
                    ],
                    &[1, 1, 1, 1, 1],
                    stream,
                )?
            }
            "per_channel" => {
                if channels % 2 != 0 {
                    return Err(danmo_mlx::Error::new(
                        "LTX 2.5 video encoder",
                        format!("per_channel log-var expects even channels, got {channels}"),
                    ));
                }
                let half = channels / 2;
                x.slice(
                    &[0, 0, 0, 0, 0],
                    &[x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3], half],
                    &[1, 1, 1, 1, 1],
                    stream,
                )?
            }
            "none" => x,
            other => {
                return Err(danmo_mlx::Error::new(
                    "LTX 2.5 video encoder",
                    format!("unsupported latent_log_var {other}"),
                ));
            }
        };
        let mean = Array::from_f32_slice(&self.mean, &[1, 1, 1, 1, self.mean.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let std = Array::from_f32_slice(&self.std, &[1, 1, 1, 1, self.std.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let normalized = Array::divide(&Array::subtract(&means, &mean, stream)?, &std, stream)?;
        normalized.transpose_axes(&[0, 4, 1, 2, 3], stream)
    }
}

impl VideoDecoder {
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        let weights = load_mlx_file(
            &bundle.join("video_vae.safetensors"),
            WeightDtype::Native,
            None,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let blocks_spec = config
            .get("decoder_blocks")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::NotBuilt("video VAE decoder_blocks missing".into()))?;
        let base = integer(config, "decoder_base_channels", 128) as i32;
        let mut feature_channels = base;
        for entry in blocks_spec {
            let (name, spec) = block_spec(entry)?;
            match name.as_str() {
                "compress_time" | "compress_space" | "compress_all" => {
                    feature_channels *= integer_value(spec, "multiplier", 1) as i32;
                }
                "res_x_y" => {
                    feature_channels *= integer_value(spec, "multiplier", 2) as i32;
                }
                _ => {}
            }
        }
        let timestep_conditioning = boolean(config, "timestep_conditioning", true);
        let mut blocks = Vec::new();
        for (index, entry) in blocks_spec.iter().rev().enumerate() {
            let (name, spec) = block_spec(entry)?;
            match name.as_str() {
                "res_x" | "attn_res_x" => {
                    blocks.push(UpBlock::ResStage {
                        index,
                        channels: feature_channels,
                        layers: integer_value(spec, "num_layers", 1) as usize,
                        inject_noise: boolean_value(spec, "inject_noise", false),
                        timestep_conditioning,
                    });
                }
                "res_x_y" => {
                    let output_channels =
                        feature_channels / integer_value(spec, "multiplier", 2) as i32;
                    blocks.push(UpBlock::Resample {
                        index,
                        input_channels: feature_channels,
                        output_channels,
                        inject_noise: boolean_value(spec, "inject_noise", false),
                    });
                    feature_channels = output_channels;
                }
                "compress_time" | "compress_space" | "compress_all" => {
                    let reduction = integer_value(spec, "multiplier", 1) as i32;
                    let (spatial, temporal) = match name.as_str() {
                        "compress_time" => (1, 2),
                        "compress_space" => (2, 1),
                        _ => (2, 2),
                    };
                    blocks.push(UpBlock::DepthToSpace {
                        index,
                        spatial,
                        temporal,
                    });
                    feature_channels /= reduction;
                }
                "attn" => blocks.push(UpBlock::Attention {
                    index,
                    channels: feature_channels,
                }),
                other => {
                    return Err(WeightError::NotBuilt(format!(
                        "unsupported LTX 2.5 video decoder block {other}"
                    )));
                }
            }
        }
        let mean = numbers(config, "latents_mean");
        let std = numbers(config, "latents_std");
        let mean = if mean.is_empty() {
            tensor_f32(&weights, "per_channel_statistics.mean-of-means")?
        } else {
            mean
        };
        let std = if std.is_empty() {
            tensor_f32(&weights, "per_channel_statistics.std-of-means")?
        } else {
            std
        };
        Ok(Self {
            weights,
            blocks,
            causal: boolean(config, "causal_decoder", false),
            timestep_conditioning,
            spatial_padding: string(config, "spatial_padding_mode", "reflect"),
            patch: integer(config, "patch_size", 4) as i32,
            mean,
            std,
        })
    }

    /// Decode `[B,C,F,H,W]` latent to `[B,3,8(F-1)+1,32H,32W]` pixels.
    pub fn decode(&self, latent: &Array, seed: u64, stream: &Stream) -> Result<Array> {
        let batch = latent.shape()[0];
        let key = RandomKey::new(seed)?;
        let mut x = latent.astype(Dtype::BFloat16, stream)?;
        if self.timestep_conditioning {
            let noise = key.normal(&x.shape(), Dtype::BFloat16, stream)?;
            x = Array::add(
                &Array::multiply(&noise, &scalar(0.025)?, stream)?,
                &Array::multiply(&x, &scalar(0.975)?, stream)?,
                stream,
            )?;
        }
        x = x.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        let mean = Array::from_f32_slice(&self.mean, &[1, 1, 1, 1, self.mean.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        let std = Array::from_f32_slice(&self.std, &[1, 1, 1, 1, self.std.len() as i32])?
            .astype(Dtype::BFloat16, stream)?;
        x = Array::add(&Array::multiply(&x, &std, stream)?, &mean, stream)?;
        x = self.conv("decoder.conv_in", &x, stream)?;
        let scaled_timestep = Array::from_f32_slice(&vec![50.0; batch as usize], &[batch])?
            .astype(Dtype::BFloat16, stream)?;

        for block in &self.blocks {
            match *block {
                UpBlock::ResStage {
                    index,
                    channels,
                    layers,
                    inject_noise,
                    timestep_conditioning,
                } => {
                    let prefix = format!("decoder.up_blocks.{index}");
                    let embedded = if timestep_conditioning {
                        Some(time_embedder(
                            &self.weights,
                            &format!("{prefix}.time_embedder"),
                            &scaled_timestep,
                            channels * 4,
                            stream,
                        )?)
                    } else {
                        None
                    };
                    for layer in 0..layers {
                        x = resnet(
                            &self.weights,
                            &format!("{prefix}.res_blocks.{layer}"),
                            &x,
                            channels,
                            channels,
                            embedded.as_ref(),
                            inject_noise,
                            self.causal,
                            &self.spatial_padding,
                            &key,
                            stream,
                        )?;
                    }
                }
                UpBlock::Resample {
                    index,
                    input_channels,
                    output_channels,
                    inject_noise,
                } => {
                    x = resnet(
                        &self.weights,
                        &format!("decoder.up_blocks.{index}"),
                        &x,
                        input_channels,
                        output_channels,
                        None,
                        inject_noise,
                        self.causal,
                        &self.spatial_padding,
                        &key,
                        stream,
                    )?;
                }
                UpBlock::DepthToSpace {
                    index,
                    spatial,
                    temporal,
                } => {
                    x = self.conv(&format!("decoder.up_blocks.{index}.conv"), &x, stream)?;
                    x = depth_to_space_3d(&x, spatial, temporal, stream)?;
                    if temporal > 1 {
                        let shape = x.shape();
                        x = x.slice(
                            &[0, 1, 0, 0, 0],
                            &[shape[0], shape[1], shape[2], shape[3], shape[4]],
                            &[1, 1, 1, 1, 1],
                            stream,
                        )?;
                    }
                }
                UpBlock::Attention { index, channels } => {
                    x = attention(
                        &self.weights,
                        &format!("decoder.up_blocks.{index}"),
                        &x,
                        channels,
                        stream,
                    )?;
                }
            }
            x.eval()?;
        }
        x = x.rms_norm(None, 1e-8, stream)?;
        if self.timestep_conditioning {
            let channels = x.shape()[4];
            let embedded = time_embedder(
                &self.weights,
                "decoder.last_time_embedder",
                &scaled_timestep,
                channels * 2,
                stream,
            )?;
            let table = self.weights.tensor("decoder.last_scale_shift_table")?;
            let values = Array::add(
                &embedded.reshape(&[batch, 2, channels], stream)?,
                table,
                stream,
            )?;
            let shift = values
                .slice(&[0, 0, 0], &[batch, 1, channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, channels], stream)?;
            let scale = values
                .slice(&[0, 1, 0], &[batch, 2, channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, channels], stream)?;
            x = affine(&x, &scale, &shift, stream)?;
        }
        x = self.conv("decoder.conv_out", &x.silu(stream)?, stream)?;
        x = unpatchify_spatial(&x, self.patch, stream)?;
        x.transpose_axes(&[0, 4, 1, 2, 3], stream)
    }

    fn conv(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        conv3d(
            &self.weights,
            &format!("{prefix}.conv"),
            input,
            [3, 3, 3],
            [1, 1, 1],
            self.causal,
            &self.spatial_padding,
            stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn resnet(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    input_channels: i32,
    output_channels: i32,
    timestep: Option<&Array>,
    inject_noise: bool,
    causal: bool,
    spatial_padding: &str,
    key: &RandomKey,
    stream: &Stream,
) -> Result<Array> {
    let batch = input.shape()[0];
    let mut hidden = input.rms_norm(None, 1e-8, stream)?;
    let table = timestep
        .map(|embedded| {
            Array::add(
                &embedded.reshape(&[batch, 4, input_channels], stream)?,
                weights.tensor(&format!("{prefix}.scale_shift_table"))?,
                stream,
            )
        })
        .transpose()?;
    if let Some(table) = &table {
        hidden = affine(
            &hidden,
            &table
                .slice(&[0, 1, 0], &[batch, 2, input_channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, input_channels], stream)?,
            &table
                .slice(&[0, 0, 0], &[batch, 1, input_channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, input_channels], stream)?,
            stream,
        )?;
    }
    hidden = conv3d(
        weights,
        &format!("{prefix}.conv1.conv"),
        &hidden.silu(stream)?,
        [3, 3, 3],
        [1, 1, 1],
        causal,
        spatial_padding,
        stream,
    )?;
    if inject_noise {
        hidden = spatial_noise(
            weights,
            &format!("{prefix}.per_channel_scale1"),
            &hidden,
            key,
            stream,
        )?;
    }
    hidden = hidden.rms_norm(None, 1e-8, stream)?;
    if let Some(table) = &table {
        hidden = affine(
            &hidden,
            &table
                .slice(&[0, 3, 0], &[batch, 4, input_channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, input_channels], stream)?,
            &table
                .slice(&[0, 2, 0], &[batch, 3, input_channels], &[1, 1, 1], stream)?
                .reshape(&[batch, 1, 1, 1, input_channels], stream)?,
            stream,
        )?;
    }
    hidden = conv3d(
        weights,
        &format!("{prefix}.conv2.conv"),
        &hidden.silu(stream)?,
        [3, 3, 3],
        [1, 1, 1],
        causal,
        spatial_padding,
        stream,
    )?;
    if inject_noise {
        hidden = spatial_noise(
            weights,
            &format!("{prefix}.per_channel_scale2"),
            &hidden,
            key,
            stream,
        )?;
    }
    let mut shortcut = input.clone();
    if input_channels != output_channels {
        shortcut = group_norm(
            weights,
            &format!("{prefix}.norm3"),
            &shortcut,
            1,
            1e-6,
            stream,
        )?;
        shortcut = conv3d(
            weights,
            &format!("{prefix}.conv_shortcut.conv"),
            &shortcut,
            [1, 1, 1],
            [1, 1, 1],
            causal,
            spatial_padding,
            stream,
        )?;
    }
    Array::add(&shortcut, &hidden, stream)
}

fn spatial_noise(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    key: &RandomKey,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let channels = shape[4];
    let noise = key.normal(&[1, shape[2], shape[3], 1], input.dtype(), stream)?;
    let scale = weights
        .tensor(name)?
        .reshape(&[1, 1, 1, channels], stream)?;
    Array::add(input, &Array::multiply(&noise, &scale, stream)?, stream)
}

fn attention(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    channels: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let gamma = weights
        .tensor(&format!("{prefix}.norm.gamma"))?
        .reshape(&[channels], stream)?;
    let normed = Array::multiply(
        &input.rms_norm(None, 1e-8, stream)?,
        &Array::multiply(&gamma, &scalar((channels as f32).sqrt())?, stream)?,
        stream,
    )?;
    let qkv = conv3d(
        weights,
        &format!("{prefix}.to_qkv.conv"),
        &normed,
        [1, 1, 1],
        [1, 1, 1],
        false,
        "constant",
        stream,
    )?
    .reshape(&[b * d, h * w, channels * 3], stream)?;
    let q = slice_last(&qkv, 0, channels, stream)?.expand_dims(1, stream)?;
    let k = slice_last(&qkv, channels, channels * 2, stream)?.expand_dims(1, stream)?;
    let v = slice_last(&qkv, channels * 2, channels * 3, stream)?.expand_dims(1, stream)?;
    let output = Array::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (channels as f32).sqrt(),
        "array",
        None,
        stream,
    )?
    .reshape(&[b, d, h, w, channels], stream)?;
    let projected = conv3d(
        weights,
        &format!("{prefix}.proj.conv"),
        &output,
        [1, 1, 1],
        [1, 1, 1],
        false,
        "constant",
        stream,
    )?;
    Array::add(input, &projected, stream)
}

fn time_embedder(
    weights: &MlxWeights,
    prefix: &str,
    timestep: &Array,
    output_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    let projected =
        timestep_embedding(timestep, 256, 10_000.0)?.astype(timestep.dtype(), stream)?;
    let hidden = linear(weights, &format!("{prefix}.linear_1"), &projected, stream)?;
    let output = linear(
        weights,
        &format!("{prefix}.linear_2"),
        &hidden.silu(stream)?,
        stream,
    )?;
    if output.shape()[1] != output_dim {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 VAE timestep",
            "embedder output mismatch",
        ));
    }
    Ok(output)
}

fn linear(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let output = Array::matmul(
        input,
        &weights
            .tensor(&format!("{prefix}.weight"))?
            .transpose_axes(&[1, 0], stream)?,
        stream,
    )?;
    Array::add(&output, weights.tensor(&format!("{prefix}.bias"))?, stream)
}

fn affine(input: &Array, scale: &Array, shift: &Array, stream: &Stream) -> Result<Array> {
    Array::add(
        &Array::multiply(input, &Array::add(&scalar(1.0)?, scale, stream)?, stream)?,
        shift,
        stream,
    )
}

fn block_spec(value: &Value) -> std::result::Result<(String, &Value), WeightError> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| WeightError::NotBuilt(format!("invalid VAE block {value}")))?;
    let name = pair[0]
        .as_str()
        .ok_or_else(|| WeightError::NotBuilt(format!("invalid VAE block name {value}")))?;
    Ok((name.to_string(), &pair[1]))
}

fn integer_value(value: &Value, key: &str, default: u64) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(default)
}
fn boolean_value(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}
fn integer(value: &Value, key: &str, default: u64) -> u64 {
    integer_value(value, key, default)
}
fn boolean(value: &Value, key: &str, default: bool) -> bool {
    boolean_value(value, key, default)
}
fn string(value: &Value, key: &str, default: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}
fn numbers(value: &Value, key: &str) -> Vec<f32> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect()
        })
        .unwrap_or_default()
}

pub fn latent_statistics(bundle: &Path) -> std::result::Result<(Vec<f32>, Vec<f32>), WeightError> {
    let weights = load_mlx_file(
        &bundle.join("video_vae.safetensors"),
        WeightDtype::Native,
        None,
    )
    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    Ok((
        tensor_f32(&weights, "per_channel_statistics.mean-of-means")?,
        tensor_f32(&weights, "per_channel_statistics.std-of-means")?,
    ))
}

fn tensor_f32(weights: &MlxWeights, name: &str) -> std::result::Result<Vec<f32>, WeightError> {
    let stream = Stream::cpu().map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    weights
        .tensor(name)
        .and_then(|tensor| tensor.astype(Dtype::Float32, &stream))
        .and_then(|tensor| tensor.to_vec_f32(&stream))
        .map_err(|error| WeightError::NotBuilt(error.to_string()))
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn real_video_decoder_has_expected_geometry_and_finite_pixels() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let decoder = super::VideoDecoder::load(&root, &config.vae).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let fixture_path = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_REFERENCE").expect("DANQING_TEST_LTX25_REFERENCE"),
        );
        let fixture =
            crate::weights::load_mlx_file(&fixture_path, crate::weights::WeightDtype::Native, None)
                .unwrap()
                .into_dense()
                .unwrap();
        let latent = fixture["video_decoder_input"]
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let output = decoder.decode(&latent, 7, &stream).unwrap();
        assert_eq!(output.shape(), &[1, 3, 1, 64, 64]);
        let values = output.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        let expected = fixture["video_decoder_output"].to_vec_f32(&stream).unwrap();
        let (max_abs, mean_abs) = values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold((0.0f32, 0.0f32), |(max, sum), value| {
                (max.max(value), sum + value / values.len() as f32)
            });
        println!("LTX25 video decoder: max_abs={max_abs}, mean_abs={mean_abs}");
        assert!(max_abs < 1e-6 && mean_abs < 1e-6);
    }

    #[test]
    #[ignore]
    fn real_video_encoder_compresses_still_clip_to_latent_grid() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let encoder = super::VideoEncoder::load(&root, &config.vae).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        // 25 frames @ 64x64 → latent 4 x 2 x 2 after 8× temporal / 32× spatial.
        let pixels = danmo_mlx::RandomKey::new(3)
            .unwrap()
            .normal(&[1, 3, 25, 64, 64], danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let latent = encoder.encode(&pixels, &stream).unwrap();
        assert_eq!(latent.shape(), &[1, 128, 4, 2, 2]);
        let values = latent.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
