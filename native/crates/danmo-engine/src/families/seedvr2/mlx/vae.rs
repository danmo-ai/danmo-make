//! SeedVR2 causal 3D VAE on MLX, including temporal resampling and per-frame
//! spatial attention. Published OIDHW kernels are transposed once to OKHWDI.

use std::collections::BTreeMap;
use std::path::Path;

use danmo_mlx::{Array, Dtype, Result, Stream};

use super::super::vae::{VaeSpec, decode_frames_out, encode_frames_out};
use crate::weights::{MlxWeights, WeightDtype, WeightError, load_mlx_file};

const NORM_EPS: f32 = 1e-6;

pub struct Vae {
    weights: MlxWeights,
    spec: VaeSpec,
}

#[derive(Default)]
pub struct VaeStreamCache {
    layers: BTreeMap<String, Array>,
    pub enabled: bool,
}

impl VaeStreamCache {
    pub fn reset(&mut self) {
        self.layers.clear();
    }
}

impl Vae {
    pub fn load(path: &Path, stream: &Stream) -> std::result::Result<Self, WeightError> {
        let mut weights = load_mlx_file(path, WeightDtype::Native, None)
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let spec = infer_spec(&weights)?;
        let conv_names: Vec<String> = weights
            .names()
            .filter(|name| name.ends_with(".weight"))
            .filter(|name| {
                weights
                    .tensor(name)
                    .map(|tensor| tensor.shape().len() == 5)
                    .unwrap_or(false)
            })
            .map(str::to_string)
            .collect();
        for name in conv_names {
            let kernel = weights
                .tensor(&name)
                .and_then(|tensor| tensor.transpose_axes(&[0, 2, 3, 4, 1], stream))
                .and_then(|tensor| tensor.contiguous(stream))
                .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
            kernel
                .eval()
                .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
            weights = weights.with_tensor_production(name, kernel);
        }
        Ok(Self { weights, spec })
    }

    pub fn spec(&self) -> &VaeSpec {
        &self.spec
    }

    pub fn nbytes(&self) -> usize {
        self.weights.nbytes()
    }

    pub fn encode_frames_out(&self, frames: i64) -> i64 {
        encode_frames_out(&self.spec, frames)
    }

    pub fn decode_frames_out(&self, frames: i64) -> i64 {
        decode_frames_out(&self.spec, frames)
    }

    /// Encode `[1,3,T,H,W]` pixels in `[-1,1]` to `[1,16,D,H/8,W/8]`.
    pub fn encode_tile(&self, pixels: &Array, stream: &Stream) -> Result<Array> {
        self.encode_tile_with_cache(pixels, stream, None)
    }

    pub fn encode_tile_with_cache(
        &self,
        pixels: &Array,
        stream: &Stream,
        mut cache: Option<&mut VaeStreamCache>,
    ) -> Result<Array> {
        expect_video(pixels, 3, "SeedVR2 encoder")?;
        let mut x = pixels.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        x = conv3d_cached(
            &self.weights,
            "encoder.conv_in",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            stream,
            cache.as_deref_mut(),
        )?;
        let blocks = self.spec.block_out.len();
        for (index, _) in self.spec.block_out.iter().enumerate() {
            for layer in 0..self.spec.enc_resnets {
                x = resnet_cached(
                    &self.weights,
                    &format!("encoder.down_blocks.{index}.resnets.{layer}"),
                    &x,
                    self.spec.groups,
                    stream,
                    cache.as_deref_mut(),
                )?;
                x.eval()?;
            }
            if index + 1 != blocks {
                let temporal = index >= blocks - self.spec.temporal_down - 1;
                x = x.pad(&[2, 3], &[0, 0], &[1, 1], 0.0, "constant", stream)?;
                x = conv3d_cached(
                    &self.weights,
                    &format!("encoder.down_blocks.{index}.downsamplers.0.conv"),
                    &x,
                    if temporal { [3, 3, 3] } else { [1, 3, 3] },
                    if temporal { [2, 2, 2] } else { [1, 2, 2] },
                    [0, 0, 0],
                    stream,
                    cache.as_deref_mut(),
                )?;
                x.eval()?;
            }
        }
        x = mid_block_cached(
            &self.weights,
            "encoder.mid_block",
            &x,
            self.spec.groups,
            stream,
            cache.as_deref_mut(),
        )?;
        x = group_norm(
            &self.weights,
            "encoder.conv_norm_out",
            &x,
            self.spec.groups,
            stream,
        )?
        .silu(stream)?;
        x = conv3d_cached(
            &self.weights,
            "encoder.conv_out",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            stream,
            cache.as_deref_mut(),
        )?;
        let shape = x.shape();
        let mean = x.slice(
            &[0, 0, 0, 0, 0],
            &[
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                self.spec.latent as i32,
            ],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        Array::multiply(&mean, &Array::scalar_f32(self.spec.scaling)?, stream)?
            .transpose_axes(&[0, 4, 1, 2, 3], stream)
    }

    /// Decode `[1,16,D,H,W]` scaled latents to `[1,3,F,8H,8W]` pixels.
    pub fn decode_tile(&self, latent: &Array, stream: &Stream) -> Result<Array> {
        self.decode_tile_with_cache(latent, stream, None)
    }

    pub fn decode_tile_with_cache(
        &self,
        latent: &Array,
        stream: &Stream,
        mut cache: Option<&mut VaeStreamCache>,
    ) -> Result<Array> {
        expect_video(latent, self.spec.latent as i32, "SeedVR2 decoder")?;
        let z = Array::divide(latent, &Array::scalar_f32(self.spec.scaling)?, stream)?;
        let mut x = z.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        x = conv3d_cached(
            &self.weights,
            "decoder.conv_in",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            stream,
            cache.as_deref_mut(),
        )?;
        x = mid_block_cached(
            &self.weights,
            "decoder.mid_block",
            &x,
            self.spec.groups,
            stream,
            cache.as_deref_mut(),
        )?;
        let blocks = self.spec.block_out.len();
        for index in 0..blocks {
            for layer in 0..self.spec.dec_resnets {
                x = resnet_cached(
                    &self.weights,
                    &format!("decoder.up_blocks.{index}.resnets.{layer}"),
                    &x,
                    self.spec.groups,
                    stream,
                    cache.as_deref_mut(),
                )?;
                x.eval()?;
            }
            if index + 1 != blocks {
                let temporal = index < self.spec.temporal_up;
                let tf = if temporal { 2 } else { 1 };
                x = conv3d_cached(
                    &self.weights,
                    &format!("decoder.up_blocks.{index}.upsamplers.0.upscale_conv"),
                    &x,
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    stream,
                    cache.as_deref_mut(),
                )?;
                x = pixel_shuffle(&x, tf, stream)?;
                x = conv3d_cached(
                    &self.weights,
                    &format!("decoder.up_blocks.{index}.upsamplers.0.conv"),
                    &x,
                    [3, 3, 3],
                    [1, 1, 1],
                    [0, 1, 1],
                    stream,
                    cache.as_deref_mut(),
                )?;
                x.eval()?;
            }
        }
        x = group_norm(
            &self.weights,
            "decoder.conv_norm_out",
            &x,
            self.spec.groups,
            stream,
        )?
        .silu(stream)?;
        conv3d_cached(
            &self.weights,
            "decoder.conv_out",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            stream,
            cache,
        )?
        .transpose_axes(&[0, 4, 1, 2, 3], stream)
    }
}

fn infer_spec(weights: &MlxWeights) -> std::result::Result<VaeSpec, WeightError> {
    let names: Vec<String> = weights.names().map(str::to_string).collect();
    let mut blocks = 0usize;
    while names
        .iter()
        .any(|name| name.starts_with(&format!("encoder.down_blocks.{blocks}.")))
    {
        blocks += 1;
    }
    if blocks == 0 {
        return Err(WeightError::NotBuilt(
            "seedvr2 MLX VAE has no encoder.down_blocks".into(),
        ));
    }
    let shape = |name: &str| -> std::result::Result<Vec<i32>, WeightError> {
        weights
            .tensor(name)
            .map(|tensor| tensor.shape())
            .map_err(|error| WeightError::NotBuilt(error.to_string()))
    };
    let mut block_out = Vec::with_capacity(blocks);
    for index in 0..blocks {
        block_out
            .push(shape(&format!("encoder.down_blocks.{index}.resnets.0.conv1.bias"))?[0] as i64);
    }
    let count = |prefix: &str| {
        let mut count = 0usize;
        while names.contains(&format!("{prefix}.resnets.{count}.norm1.weight")) {
            count += 1;
        }
        count
    };
    let enc_resnets = count("encoder.down_blocks.0");
    let dec_resnets = count(&format!("decoder.up_blocks.{}", blocks - 1));
    if enc_resnets == 0 || dec_resnets == 0 {
        return Err(WeightError::NotBuilt(
            "seedvr2 MLX VAE has no resnet blocks".into(),
        ));
    }
    let latent = shape("encoder.conv_out.bias")?[0] as i64 / 2;
    let temporal_down = (0..blocks - 1)
        .filter(|index| {
            shape(&format!(
                "encoder.down_blocks.{index}.downsamplers.0.conv.weight"
            ))
            .map(|shape| shape[2] == 3)
            .unwrap_or(false)
        })
        .count();
    let temporal_up = (0..blocks - 1)
        .filter(|index| {
            let channels = block_out[blocks - 1 - index];
            shape(&format!(
                "decoder.up_blocks.{index}.upsamplers.0.upscale_conv.bias"
            ))
            .map(|shape| shape[0] as i64 == channels * 8)
            .unwrap_or(false)
        })
        .count();
    let mut common = 0i64;
    for name in names.iter().filter(|name| {
        name.ends_with(".bias")
            && (name.contains("norm1")
                || name.contains("norm2")
                || name.contains("conv_norm_out")
                || name.contains("group_norm"))
    }) {
        let channels = shape(name)?[0] as i64;
        common = if common == 0 {
            channels
        } else {
            let (mut a, mut b) = (common, channels);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
    }
    let mut groups = common.min(32);
    while groups > 1 && common % groups != 0 {
        groups -= 1;
    }
    if groups < 1 {
        return Err(WeightError::NotBuilt(
            "seedvr2 MLX VAE has no valid GroupNorm divisor".into(),
        ));
    }
    Ok(VaeSpec {
        block_out,
        enc_resnets,
        dec_resnets,
        latent,
        temporal_down,
        temporal_up,
        groups: groups as i32,
        scaling: 0.9152,
    })
}

fn expect_video(input: &Array, channels: i32, what: &'static str) -> Result<()> {
    let shape = input.shape();
    if shape.len() != 5 || shape[0] != 1 || shape[1] != channels {
        return Err(danmo_mlx::Error::new(
            what,
            format!("expected [1,{channels},T,H,W], got {shape:?}"),
        ));
    }
    if shape[2] < 1 || shape[3] < 1 || shape[4] < 1 {
        return Err(danmo_mlx::Error::new(
            what,
            format!("empty video shape {shape:?}"),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv3d(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    kernel: [i32; 3],
    stride: [i32; 3],
    padding: [i32; 3],
    stream: &Stream,
) -> Result<Array> {
    conv3d_cached(weights, prefix, input, kernel, stride, padding, stream, None)
}

#[allow(clippy::too_many_arguments)]
fn conv3d_cached(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    kernel: [i32; 3],
    stride: [i32; 3],
    padding: [i32; 3],
    stream: &Stream,
    mut cache: Option<&mut VaeStreamCache>,
) -> Result<Array> {
    let original = input.clone();
    let mut x = input.clone();
    if kernel[0] > 1 {
        let cached = cache
            .as_deref_mut()
            .filter(|cache| cache.enabled)
            .and_then(|cache| cache.layers.get(prefix).cloned());
        if let Some(tail) = cached {
            x = Array::concatenate_axis(&[tail, x], 1, stream)?;
        } else {
            let shape = x.shape();
            let first = x.slice(
                &[0, 0, 0, 0, 0],
                &[shape[0], 1, shape[2], shape[3], shape[4]],
                &[1, 1, 1, 1, 1],
                stream,
            )?;
            let pad = first.repeat_axis(kernel[0] - 1, 1, stream)?;
            x = Array::concatenate_axis(&[pad, x], 1, stream)?;
        }
    }
    if padding[1] > 0 || padding[2] > 0 {
        x = x.pad(
            &[2, 3],
            &[padding[1], padding[2]],
            &[padding[1], padding[2]],
            0.0,
            "constant",
            stream,
        )?;
    }
    let output = Array::conv3d(
        &x,
        weights.tensor(&format!("{prefix}.weight"))?,
        stride,
        [0, 0, 0],
        [1, 1, 1],
        1,
        stream,
    )?;
    let output = Array::add(&output, weights.tensor(&format!("{prefix}.bias"))?, stream)?;
    if kernel[0] > 1 {
        if let Some(cache) = cache.as_deref_mut().filter(|cache| cache.enabled) {
            let shape = original.shape();
            let tail = original.slice(
                &[0, shape[1] - (kernel[0] - 1), 0, 0, 0],
                &[shape[0], shape[1], shape[2], shape[3], shape[4]],
                &[1, 1, 1, 1, 1],
                stream,
            )?;
            cache.layers.insert(prefix.to_string(), tail);
        }
    }
    Ok(output)
}

fn group_norm(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w, c) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let grouped = input
        .reshape(&[b, d * h * w, groups, c / groups], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[b, groups, d * h * w * (c / groups)], stream)?
        .layer_norm(NORM_EPS, stream)?;
    let normalized = grouped
        .reshape(&[b, groups, d * h * w, c / groups], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[b, d, h, w, c], stream)?
        .astype(Dtype::BFloat16, stream)?;
    let scaled = Array::multiply(
        &normalized,
        weights.tensor(&format!("{prefix}.weight"))?,
        stream,
    )?;
    Array::add(&scaled, weights.tensor(&format!("{prefix}.bias"))?, stream)
}

fn group_norm_tokens(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, tokens, channels) = (shape[0], shape[1], shape[2]);
    let grouped = input
        .reshape(&[batch, tokens, groups, channels / groups], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, groups, tokens * (channels / groups)], stream)?
        .layer_norm(NORM_EPS, stream)?;
    let normalized = grouped
        .reshape(&[batch, groups, tokens, channels / groups], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, tokens, channels], stream)?
        .astype(Dtype::BFloat16, stream)?;
    let scaled = Array::multiply(
        &normalized,
        weights.tensor(&format!("{prefix}.weight"))?,
        stream,
    )?;
    Array::add(&scaled, weights.tensor(&format!("{prefix}.bias"))?, stream)
}

fn linear(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    weights.linear(&format!("{prefix}.weight"))?.apply(
        input,
        Some(weights.tensor(&format!("{prefix}.bias"))?),
        stream,
    )
}

fn attention(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w, c) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let tokens = input.reshape(&[b * d, h * w, c], stream)?;
    let normed = group_norm_tokens(
        weights,
        &format!("{prefix}.group_norm"),
        &tokens,
        groups,
        stream,
    )?;
    let q = linear(weights, &format!("{prefix}.to_q"), &normed, stream)?.expand_dims(1, stream)?;
    let k = linear(weights, &format!("{prefix}.to_k"), &normed, stream)?.expand_dims(1, stream)?;
    let v = linear(weights, &format!("{prefix}.to_v"), &normed, stream)?.expand_dims(1, stream)?;
    let output =
        Array::scaled_dot_product_attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), "", None, stream)?
            .squeeze_axis(1, stream)?;
    let output = linear(weights, &format!("{prefix}.to_out.0"), &output, stream)?
        .reshape(&[b, d, h, w, c], stream)?;
    Array::add(input, &output, stream)
}

fn resnet(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
) -> Result<Array> {
    resnet_cached(weights, prefix, input, groups, stream, None)
}

fn resnet_cached(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
    mut cache: Option<&mut VaeStreamCache>,
) -> Result<Array> {
    let mut hidden =
        group_norm(weights, &format!("{prefix}.norm1"), input, groups, stream)?.silu(stream)?;
    hidden = conv3d_cached(
        weights,
        &format!("{prefix}.conv1"),
        &hidden,
        [3, 3, 3],
        [1, 1, 1],
        [0, 1, 1],
        stream,
        cache.as_deref_mut(),
    )?;
    hidden =
        group_norm(weights, &format!("{prefix}.norm2"), &hidden, groups, stream)?.silu(stream)?;
    hidden = conv3d_cached(
        weights,
        &format!("{prefix}.conv2"),
        &hidden,
        [3, 3, 3],
        [1, 1, 1],
        [0, 1, 1],
        stream,
        cache.as_deref_mut(),
    )?;
    let shortcut = format!("{prefix}.conv_shortcut.weight");
    let residual = if weights.has(&shortcut) {
        conv3d_cached(
            weights,
            &format!("{prefix}.conv_shortcut"),
            input,
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            stream,
            cache,
        )?
    } else {
        input.clone()
    };
    Array::add(&residual, &hidden, stream)
}

fn mid_block_cached(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    groups: i32,
    stream: &Stream,
    mut cache: Option<&mut VaeStreamCache>,
) -> Result<Array> {
    let mut x = resnet_cached(
        weights,
        &format!("{prefix}.resnets.0"),
        input,
        groups,
        stream,
        cache.as_deref_mut(),
    )?;
    x = attention(
        weights,
        &format!("{prefix}.attentions.0"),
        &x,
        groups,
        stream,
    )?;
    resnet_cached(
        weights,
        &format!("{prefix}.resnets.1"),
        &x,
        groups,
        stream,
        cache,
    )
}

fn pixel_shuffle(input: &Array, temporal_factor: i32, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w, total) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let channels = total / (4 * temporal_factor);
    let mut output = input
        .reshape(&[b, d, h, w, 2, 2, temporal_factor, channels], stream)?
        .transpose_axes(&[0, 1, 6, 2, 4, 3, 5, 7], stream)?
        .reshape(&[b, d * temporal_factor, h * 2, w * 2, channels], stream)?;
    if d == 1 && temporal_factor > 1 {
        output = output.slice(
            &[0, 0, 0, 0, 0],
            &[b, 1, h * 2, w * 2, channels],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
    }
    Ok(output)
}

#[cfg(all(test, feature = "mlx-metal"))]
mod tests {
    #[test]
    #[ignore = "loads the installed VAE and a main stage fixture"]
    fn real_vae_encoder_stages_match_main() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DANQING_TEST_SEEDVR2_VAE_STAGES")
                    .expect("DANQING_TEST_SEEDVR2_VAE_STAGES"),
            )
            .unwrap(),
        )
        .unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let model = super::Vae::load(&bundle.join("ema_vae_fp16.safetensors"), &stream).unwrap();
        let (w, h, frames) = (16i64, 16i64, 5i64);
        let pixels: Vec<f32> = (0..3 * frames * w * h)
            .map(|index| (index as f32 * 0.017).sin() * 0.5)
            .collect();
        let pixels = danmo_mlx::Array::from_f32_slice(&pixels, &[1, 3, 5, 16, 16])
            .unwrap()
            .transpose_axes(&[0, 2, 3, 4, 1], &stream)
            .unwrap();
        let mut x = super::conv3d(
            &model.weights,
            "encoder.conv_in",
            &pixels,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        let mut stages = vec![("conv_in", x.clone())];
        let mut probe = super::group_norm(
            &model.weights,
            "encoder.down_blocks.0.resnets.0.norm1",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("resnet_norm_1", probe.clone()));
        probe = super::conv3d(
            &model.weights,
            "encoder.down_blocks.0.resnets.0.conv1",
            &probe.silu(&stream).unwrap(),
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        stages.push(("resnet_conv_1", probe.clone()));
        probe = super::group_norm(
            &model.weights,
            "encoder.down_blocks.0.resnets.0.norm2",
            &probe,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("resnet_norm_2", probe.clone()));
        probe = super::conv3d(
            &model.weights,
            "encoder.down_blocks.0.resnets.0.conv2",
            &probe.silu(&stream).unwrap(),
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        stages.push(("resnet_conv_2", probe));
        for index in 0..model.spec.block_out.len() {
            for layer in 0..model.spec.enc_resnets {
                x = super::resnet(
                    &model.weights,
                    &format!("encoder.down_blocks.{index}.resnets.{layer}"),
                    &x,
                    model.spec.groups,
                    &stream,
                )
                .unwrap();
                stages.push((
                    Box::leak(format!("down_{index}_resnet_{layer}").into_boxed_str()) as &str,
                    x.clone(),
                ));
            }
            if index + 1 != model.spec.block_out.len() {
                let temporal = index >= model.spec.block_out.len() - model.spec.temporal_down - 1;
                x = x
                    .pad(&[2, 3], &[0, 0], &[1, 1], 0.0, "constant", &stream)
                    .unwrap();
                x = super::conv3d(
                    &model.weights,
                    &format!("encoder.down_blocks.{index}.downsamplers.0.conv"),
                    &x,
                    if temporal { [3, 3, 3] } else { [1, 3, 3] },
                    if temporal { [2, 2, 2] } else { [1, 2, 2] },
                    [0, 0, 0],
                    &stream,
                )
                .unwrap();
                stages.push((
                    Box::leak(format!("down_{index}_downsampler").into_boxed_str()) as &str,
                    x.clone(),
                ));
            }
            stages.push((
                Box::leak(format!("down_{index}").into_boxed_str()) as &str,
                x.clone(),
            ));
        }
        x = super::resnet(
            &model.weights,
            "encoder.mid_block.resnets.0",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_resnet_0", x.clone()));
        x = super::attention(
            &model.weights,
            "encoder.mid_block.attentions.0",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_attention", x.clone()));
        x = super::resnet(
            &model.weights,
            "encoder.mid_block.resnets.1",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_resnet_1", x.clone()));
        x = super::group_norm(&model.weights, "encoder.conv_norm_out", &x, 32, &stream).unwrap();
        stages.push(("norm_out", x.clone()));
        x = x.silu(&stream).unwrap();
        x = super::conv3d(
            &model.weights,
            "encoder.conv_out",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        stages.push(("conv_out", x));
        for (name, actual) in stages {
            let actual = if name.starts_with("resnet_norm_") {
                actual
            } else {
                actual.transpose_axes(&[0, 4, 1, 2, 3], &stream).unwrap()
            }
            .to_vec_f32(&stream)
            .unwrap();
            let expected: Vec<f32> = reference[name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(actual.len(), expected.len(), "{name} length");
            let mut mse = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut energy = 0.0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let difference = actual - expected;
                mse += f64::from(difference * difference);
                max_abs = max_abs.max(difference.abs());
                energy += f64::from(expected * expected);
            }
            mse /= actual.len() as f64;
            let rel_l2 = (mse * actual.len() as f64 / energy.max(1e-20)).sqrt();
            eprintln!("vae {name}: mse={mse:.9} max_abs={max_abs:.6} rel_l2={rel_l2:.6}");
            assert!(rel_l2 < 0.01, "{name} relative L2 {rel_l2} exceeds 1%");
        }
    }

    #[test]
    #[ignore = "loads the installed VAE and a main decoder fixture"]
    fn real_vae_decoder_stages_match_main() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DANQING_TEST_SEEDVR2_VAE_DECODER_STAGES")
                    .expect("DANQING_TEST_SEEDVR2_VAE_DECODER_STAGES"),
            )
            .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            reference[name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let model = super::Vae::load(&bundle.join("ema_vae_fp16.safetensors"), &stream).unwrap();
        let z = danmo_mlx::Array::from_f32_slice(&values("restored"), &[1, 16, 2, 2, 2]).unwrap();
        let z =
            danmo_mlx::Array::divide(&z, &danmo_mlx::Array::scalar_f32(0.9152).unwrap(), &stream)
                .unwrap()
                .transpose_axes(&[0, 2, 3, 4, 1], &stream)
                .unwrap();
        let mut x = super::conv3d(
            &model.weights,
            "decoder.conv_in",
            &z,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        let mut stages = vec![("conv_in", x.clone())];
        x = super::resnet(
            &model.weights,
            "decoder.mid_block.resnets.0",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_resnet_0", x.clone()));
        x = super::attention(
            &model.weights,
            "decoder.mid_block.attentions.0",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_attention", x.clone()));
        x = super::resnet(
            &model.weights,
            "decoder.mid_block.resnets.1",
            &x,
            32,
            &stream,
        )
        .unwrap();
        stages.push(("mid_resnet_1", x.clone()));
        for index in 0..model.spec.block_out.len() {
            for layer in 0..model.spec.dec_resnets {
                x = super::resnet(
                    &model.weights,
                    &format!("decoder.up_blocks.{index}.resnets.{layer}"),
                    &x,
                    32,
                    &stream,
                )
                .unwrap();
                stages.push((
                    Box::leak(format!("up_{index}_resnet_{layer}").into_boxed_str()) as &str,
                    x.clone(),
                ));
            }
            if index + 1 != model.spec.block_out.len() {
                let temporal = index < model.spec.temporal_up;
                let tf = if temporal { 2 } else { 1 };
                x = super::conv3d(
                    &model.weights,
                    &format!("decoder.up_blocks.{index}.upsamplers.0.upscale_conv"),
                    &x,
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    &stream,
                )
                .unwrap();
                stages.push((
                    Box::leak(format!("up_{index}_upscale_conv").into_boxed_str()) as &str,
                    x.clone(),
                ));
                x = super::pixel_shuffle(&x, tf, &stream).unwrap();
                stages.push((
                    Box::leak(format!("up_{index}_pixel_shuffle").into_boxed_str()) as &str,
                    x.clone(),
                ));
                x = super::conv3d(
                    &model.weights,
                    &format!("decoder.up_blocks.{index}.upsamplers.0.conv"),
                    &x,
                    [3, 3, 3],
                    [1, 1, 1],
                    [0, 1, 1],
                    &stream,
                )
                .unwrap();
                stages.push((
                    Box::leak(format!("up_{index}_upsampler").into_boxed_str()) as &str,
                    x.clone(),
                ));
            }
        }
        x = super::group_norm(&model.weights, "decoder.conv_norm_out", &x, 32, &stream).unwrap();
        stages.push(("norm_out", x.clone()));
        x = x.silu(&stream).unwrap();
        x = super::conv3d(
            &model.weights,
            "decoder.conv_out",
            &x,
            [3, 3, 3],
            [1, 1, 1],
            [0, 1, 1],
            &stream,
        )
        .unwrap();
        stages.push(("conv_out", x));
        for (name, actual) in stages {
            let actual = actual
                .transpose_axes(&[0, 4, 1, 2, 3], &stream)
                .unwrap()
                .to_vec_f32(&stream)
                .unwrap();
            let expected = values(name);
            assert_eq!(actual.len(), expected.len(), "{name} length");
            let mut mse = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut energy = 0.0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let difference = actual - expected;
                mse += f64::from(difference * difference);
                max_abs = max_abs.max(difference.abs());
                energy += f64::from(expected * expected);
            }
            mse /= actual.len() as f64;
            let rel_l2 = (mse * actual.len() as f64 / energy.max(1e-20)).sqrt();
            eprintln!("decoder {name}: mse={mse:.9} max_abs={max_abs:.6} rel_l2={rel_l2:.6}");
            assert!(rel_l2 < 0.01, "{name} relative L2 {rel_l2} exceeds 1%");
        }
    }
}
