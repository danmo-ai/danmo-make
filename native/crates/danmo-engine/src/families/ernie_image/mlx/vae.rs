use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use crate::common::vae::VaeSpec;
use crate::weights::MlxWeights;

const BN_EPS: f32 = 1e-5;

pub fn decode(
    weights: &MlxWeights,
    latents: &[f32],
    grid_h: i32,
    grid_w: i32,
    width: u32,
    height: u32,
    stream: &Stream,
) -> Result<Vec<u8>> {
    let latent = prepare_latent(weights, latents, grid_h, grid_w, stream)?;
    let decoded = crate::common::vae::mlx::decode(weights, &spec(), &latent, stream)?;
    let decoded = crate::common::vae::mlx::decode(weights, &spec(), &latent, stream)?;
    decoded.eval()?;
    if decoded.shape() != vec![1, 3, height as i32, width as i32] {
        return Err(Error::new(
            "ERNIE VAE",
            format!(
                "decoded shape {:?} is not [1, 3, {height}, {width}]",
                decoded.shape()
            ),
        ));
    }
    let values = decoded.to_vec_f32(stream)?;
    let plane = width as usize * height as usize;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            for channel in 0..3 {
                let sample = values[channel * plane + y * width as usize + x];
                rgb[(y * width as usize + x) * 3 + channel] =
                    ((sample + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(rgb)
}

fn prepare_latent(
    weights: &MlxWeights,
    latents: &[f32],
    grid_h: i32,
    grid_w: i32,
    stream: &Stream,
) -> Result<Array> {
    let expected = (128 * grid_h * grid_w) as usize;
    if latents.len() != expected {
        return Err(Error::new(
            "ERNIE VAE",
            format!("latent has {} values, expected {expected}", latents.len()),
        ));
    }
    let mean = weights.tensor("bn.running_mean")?.to_vec_f32(stream)?;
    let var = weights.tensor("bn.running_var")?.to_vec_f32(stream)?;
    if mean.len() != 128
        || var.len() != 128
        || mean.iter().any(|v| !v.is_finite())
        || var.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(Error::new(
            "ERNIE VAE",
            "batch norm statistics must contain 128 finite values and non-negative variance",
        ));
    }
    let std: Vec<f32> = var
        .into_iter()
        .map(|value| (value + BN_EPS).sqrt())
        .collect();
    let latent = Array::from_f32_slice(latents, &[1, 128, grid_h, grid_w])?
        .astype(Dtype::BFloat16, stream)?;
    let mean = Array::from_f32_slice(&mean, &[1, 128, 1, 1])?.astype(latent.dtype(), stream)?;
    let std = Array::from_f32_slice(&std, &[1, 128, 1, 1])?.astype(latent.dtype(), stream)?;
    let latent = Array::add(&Array::multiply(&latent, &std, stream)?, &mean, stream)?;
    let latent = latent
        .reshape(&[1, 32, 2, 2, grid_h, grid_w], stream)?
        .transpose_axes(&[0, 1, 4, 2, 5, 3], stream)?
        .reshape(&[1, 32, grid_h * 2, grid_w * 2], stream)?;
    let input = latent.transpose_axes(&[0, 2, 3, 1], stream)?;
    let kernel = weights
        .tensor("post_quant_conv.weight")?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let latent = Array::conv2d(&input, &kernel, [1, 1], [0, 0], [1, 1], 1, stream)?;
    Array::add(&latent, weights.tensor("post_quant_conv.bias")?, stream)?
        .transpose_axes(&[0, 3, 1, 2], stream)
}

fn spec() -> VaeSpec {
    VaeSpec {
        latent_channels: 32,
        block_out: vec![128, 256, 512, 512],
        resnets: 3,
        groups: 32,
        scaling: 1.0,
        shift: 0.0,
        attention: true,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use danmo_mlx::{Array, Stream};

    use super::{prepare_latent, BN_EPS};
    use crate::weights::MlxWeights;

    #[test]
    fn unpatch_matches_the_official_channel_order() {
        let stream = Stream::cpu().unwrap();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "bn.running_mean".into(),
            Array::from_f32_slice(&vec![0.0; 128], &[128]).unwrap(),
        );
        tensors.insert(
            "bn.running_var".into(),
            Array::from_f32_slice(&vec![1.0 - BN_EPS; 128], &[128]).unwrap(),
        );
        let mut kernel = vec![0.0; 32 * 32];
        for channel in 0..32 {
            kernel[channel * 32 + channel] = 1.0;
        }
        tensors.insert(
            "post_quant_conv.weight".into(),
            Array::from_f32_slice(&kernel, &[32, 32, 1, 1]).unwrap(),
        );
        tensors.insert(
            "post_quant_conv.bias".into(),
            Array::from_f32_slice(&vec![0.0; 32], &[32]).unwrap(),
        );
        let weights = MlxWeights::from_dense(tensors);
        let source: Vec<f32> = (0..256).map(|value| value as f32).collect();
        let actual = prepare_latent(&weights, &source, 1, 2, &stream).unwrap();
        assert_eq!(actual.shape(), vec![1, 32, 2, 4]);
        let actual = actual.to_vec_f32(&stream).unwrap();
        for channel in 0..32 {
            for ph in 0..2 {
                for pw in 0..2 {
                    for px in 0..2 {
                        let output_index = channel * 8 + ph * 4 + px * 2 + pw;
                        let source_index = (channel * 4 + ph * 2 + pw) * 2 + px;
                        assert!((actual[output_index] - source[source_index]).abs() < 1e-4);
                    }
                }
            }
        }
    }
}
