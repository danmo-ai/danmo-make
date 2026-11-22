//! LTX 2.5 stage-1 → stage-2 spatial latent upsampler.
//! Channel-first model interfaces are kept in this family; MLX convolutions
//! use channel-last tensors internally.

use std::path::Path;

use danmo_mlx::{Array, Result, Stream};
use serde_json::Value;

use super::conv::{conv3d_zero_temporal, group_norm};
use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

pub struct LatentUpsampler {
    weights: MlxWeights,
    blocks: usize,
    channels: i32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl LatentUpsampler {
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        if config
            .get("rational_resampler")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || config
                .get("temporal_upsample")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || !config
                .get("spatial_upsample")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        {
            return Err(WeightError::NotBuilt(
                "LTX 2.5 upsampler requires spatial-only convolutional x2 resampling".into(),
            ));
        }
        let weights = load_mlx_file(
            &bundle.join("upsampler.safetensors"),
            WeightDtype::Native,
            None,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let blocks = config
            .get("num_blocks_per_stage")
            .and_then(Value::as_u64)
            .unwrap_or(4) as usize;
        let channels = config
            .get("in_channels")
            .and_then(Value::as_u64)
            .unwrap_or(128) as i32;
        let (mean, std) = super::video_vae::latent_statistics(bundle)?;
        if mean.len() != channels as usize || std.len() != mean.len() {
            return Err(WeightError::NotBuilt(format!(
                "LTX 2.5 upsampler statistics have {}/{} channels, expected {channels}",
                mean.len(),
                std.len()
            )));
        }
        Ok(Self {
            weights,
            blocks,
            channels,
            mean,
            std,
        })
    }

    /// `[B,C,F,H,W]` normalized latent → `[B,C,F,2H,2W]` normalized latent.
    pub fn upsample(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let dtype = input.dtype();
        let mean = Array::from_f32_slice(&self.mean, &[1, self.channels, 1, 1, 1])?;
        let std = Array::from_f32_slice(&self.std, &[1, self.channels, 1, 1, 1])?;
        let denormalized = Array::add(&Array::multiply(input, &std, stream)?, &mean, stream)?
            .astype(danmo_mlx::Dtype::Float32, stream)?;
        let mut x = conv_cf(&self.weights, "initial_conv", &denormalized, stream)?;
        x = norm_cf(&self.weights, "initial_norm", &x, stream)?.silu(stream)?;
        for block in 0..self.blocks {
            x = residual(&self.weights, &format!("res_blocks.{block}"), &x, stream)?;
        }

        let shape = x.shape();
        let (b, c, f, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let folded = x
            .transpose_axes(&[0, 2, 3, 4, 1], stream)?
            .reshape(&[b * f, h, w, c], stream)?;
        let mut upsampled = Array::conv2d(
            &folded,
            self.weights.tensor("upsampler.0.weight")?,
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            stream,
        )?;
        upsampled = Array::add(&upsampled, self.weights.tensor("upsampler.0.bias")?, stream)?;
        let total = upsampled.shape()[3];
        let upsampled = upsampled
            .reshape(&[b * f, h, w, total / 4, 2, 2], stream)?
            .transpose_axes(&[0, 1, 4, 2, 5, 3], stream)?
            .reshape(&[b, f, h * 2, w * 2, total / 4], stream)?
            .transpose_axes(&[0, 4, 1, 2, 3], stream)?;
        x = upsampled;
        for block in 0..self.blocks {
            x = residual(
                &self.weights,
                &format!("post_upsample_res_blocks.{block}"),
                &x,
                stream,
            )?;
        }
        x = conv_cf(&self.weights, "final_conv", &x, stream)?;
        Array::divide(&Array::subtract(&x, &mean, stream)?, &std, stream)?.astype(dtype, stream)
    }
}

fn conv_cf(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    input
        .transpose_axes(&[0, 2, 3, 4, 1], stream)
        .and_then(|input| {
            conv3d_zero_temporal(weights, name, &input, [3, 3, 3], [1, 1, 1], "zeros", stream)
        })?
        .transpose_axes(&[0, 4, 1, 2, 3], stream)
}

fn norm_cf(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    input
        .transpose_axes(&[0, 2, 3, 4, 1], stream)
        .and_then(|input| group_norm(weights, name, &input, 32, 1e-5, stream))?
        .transpose_axes(&[0, 4, 1, 2, 3], stream)
}

fn residual(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let x = conv_cf(weights, &format!("{prefix}.conv1"), input, stream)?;
    let x = norm_cf(weights, &format!("{prefix}.norm1"), &x, stream)?.silu(stream)?;
    let x = conv_cf(weights, &format!("{prefix}.conv2"), &x, stream)?;
    let x = norm_cf(weights, &format!("{prefix}.norm2"), &x, stream)?;
    Array::add(&x, input, stream)?.silu(stream)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn real_upsampler_shape_and_finite_values() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("set DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let model = super::LatentUpsampler::load(&root, &config.upsampler).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let fixture_path = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_REFERENCE").expect("DANQING_TEST_LTX25_REFERENCE"),
        );
        let fixture =
            crate::weights::load_mlx_file(&fixture_path, crate::weights::WeightDtype::Native, None)
                .unwrap()
                .into_dense()
                .unwrap();
        let input = fixture["upsampler_input"].clone();
        let output = model.upsample(&input, &stream).unwrap();
        assert_eq!(output.shape(), [1, 128, 1, 4, 4]);
        let values = output.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        let expected = fixture["upsampler_output"].to_vec_f32(&stream).unwrap();
        let (max_abs, mean_abs) = values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold((0.0f32, 0.0f32), |(max, sum), value| {
                (max.max(value), sum + value / values.len() as f32)
            });
        println!("LTX25 upsampler: max_abs={max_abs}, mean_abs={mean_abs}");
        assert!(max_abs < 1e-4 && mean_abs < 1e-4);
    }
}
