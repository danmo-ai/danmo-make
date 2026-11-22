//! Qwen-Image 3D causal VAE decode (create: single frame).
//!
//! Names and math follow danmo-make `families/qwen/vae/*_mlx.py`. The
//! checkpoint stores conv kernels channel-last (`[O, kT, kH, kW, I]` for 3D,
//! `[O, kH, kW, I]` for 2D), so convs read them directly. Spatial norms are
//! channel L2 (`x / max(||x||_c, eps) * sqrt(C) * weight`), not GroupNorm.
//!
//! The `upsample3d` blocks carry `time_conv` weights that main's image path
//! never calls (temporal upsampling is for video); the single-frame decode
//! skips them exactly like main.

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use crate::weights::MlxWeights;

const NORM_EPS: f32 = 1e-12;

// diffusers AutoencoderKLQwenImage latents_mean / latents_std.
const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

/// Packed DiT latent `[1, grid_h*grid_w, 64]` → `[1, 16, 1, 2*grid_h,
/// 2*grid_w]` (main `qwen_unpack_latents_nchw` plus the frame axis).
pub(super) fn unpack_latent(
    latents: &[f32],
    grid_h: i32,
    grid_w: i32,
    stream: &Stream,
) -> Result<Array> {
    let expected = (64 * grid_h * grid_w) as usize;
    if latents.len() != expected {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("latent has {} values, expected {expected}", latents.len()),
        ));
    }
    let packed = Array::from_f32_slice(latents, &[1, grid_h * grid_w, 64])?;
    let unpacked = packed
        .reshape(&[1, grid_h, grid_w, 16, 2, 2], stream)?
        .transpose_axes(&[0, 3, 1, 4, 2, 5], stream)?
        .reshape(&[1, 16, grid_h * 2, grid_w * 2], stream)?
        .expand_dims(2, stream)?;
    Ok(unpacked)
}

/// `[1, 16, H_lat, W_lat]` → packed DiT sequence `[1, (H/2)*(W/2), 64]`.
pub(super) fn pack_latent(latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 16 || shape[2] % 2 != 0 || shape[3] % 2 != 0
    {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("pack input {shape:?} is not [1, 16, even H, even W]"),
        ));
    }
    let (height, width) = (shape[2], shape[3]);
    let (grid_h, grid_w) = (height / 2, width / 2);
    latent
        .reshape(&[1, 16, grid_h, 2, grid_w, 2], stream)?
        .transpose_axes(&[0, 2, 4, 1, 3, 5], stream)?
        .reshape(&[1, grid_h * grid_w, 64], stream)
}

/// Encode one RGB frame in `[-1, 1]` and pack it for the DiT.
/// `image` is NCHW `[1, 3, H, W]` with `H` and `W` divisible by 16.
/// Returns `(packed [1, grid_h*grid_w, 64] as f32, grid_h, grid_w)`.
pub(super) fn encode_packed(
    weights: &MlxWeights,
    image: &Array,
    stream: &Stream,
) -> Result<(Vec<f32>, i32, i32)> {
    let shape = image.shape();
    if shape.len() != 4
        || shape[0] != 1
        || shape[1] != 3
        || shape[2] % 16 != 0
        || shape[3] % 16 != 0
    {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("encode image {shape:?} is not [1, 3, H, W] divisible by 16"),
        ));
    }
    let (height, width) = (shape[2], shape[3]);
    let frame = image.expand_dims(2, stream)?;
    let mut x = causal_conv3d(weights, "encoder.conv_in.conv3d", &frame, 1, stream)?;
    for stage in 0..4 {
        let prefix = format!("encoder.down_blocks.{stage}");
        x = down_block(weights, &prefix, &x, stage < 3, stream)?;
    }
    x = res_block(weights, "encoder.mid_block.resnets.0", &x, stream)?;
    x = attention_block(weights, "encoder.mid_block.attentions.0", &x, stream)?;
    x = res_block(weights, "encoder.mid_block.resnets.1", &x, stream)?;
    x = channel_norm(weights, "encoder.norm_out", &x, stream)?;
    x = x.silu(stream)?;
    x = causal_conv3d(weights, "encoder.conv_out.conv3d", &x, 1, stream)?;
    x = causal_conv3d(weights, "quant_conv.conv3d", &x, 0, stream)?;
    let encoded = x.shape();
    if encoded.len() != 5 || encoded[1] < 16 || encoded[2] != 1 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("quant conv shape {encoded:?} is not [1, C>=16, 1, H, W]"),
        ));
    }
    let x = x.slice(
        &[0, 0, 0, 0, 0],
        &[1, 16, 1, encoded[3], encoded[4]],
        &[1, 1, 1, 1, 1],
        stream,
    )?;
    let dtype = x.dtype();
    let mean = Array::from_f32_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1])?.astype(dtype, stream)?;
    let std = Array::from_f32_slice(&LATENTS_STD, &[1, 16, 1, 1, 1])?.astype(dtype, stream)?;
    let x = Array::divide(&Array::subtract(&x, &mean, stream)?, &std, stream)?;
    let planar = x.squeeze_axis(2, stream)?;
    let packed = pack_latent(&planar, stream)?;
    packed.eval()?;
    let grid_h = planar.shape()[2] / 2;
    let grid_w = planar.shape()[3] / 2;
    if packed.shape() != vec![1, grid_h * grid_w, 64] {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!(
                "packed shape {:?} is not [1, {}, 64]",
                packed.shape(),
                grid_h * grid_w
            ),
        ));
    }
    if grid_h != height / 16 || grid_w != width / 16 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!(
                "encoded grid {grid_h}x{grid_w} is not {}x{}",
                height / 16,
                width / 16
            ),
        ));
    }
    Ok((packed.to_vec_f32(stream)?, grid_h, grid_w))
}

/// Decode one frame. Input `[1, 16, 1, H_lat, W_lat]` normalized latent;
/// output `[1, 3, H_lat*8, W_lat*8]` in `[-1, 1]`.
pub(super) fn decode_frame(weights: &MlxWeights, latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 5 || shape[0] != 1 || shape[1] != 16 || shape[2] != 1 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("latent shape {shape:?} is not [1, 16, 1, H, W]"),
        ));
    }
    let dtype = latent.dtype();
    let mean = Array::from_f32_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1])?.astype(dtype, stream)?;
    let std = Array::from_f32_slice(&LATENTS_STD, &[1, 16, 1, 1, 1])?.astype(dtype, stream)?;
    let mut x = Array::add(&Array::multiply(latent, &std, stream)?, &mean, stream)?;
    x = causal_conv3d(weights, "post_quant_conv.conv3d", &x, 0, stream)?;
    x = causal_conv3d(weights, "decoder.conv_in.conv3d", &x, 1, stream)?;
    x = res_block(weights, "decoder.mid_block.resnets.0", &x, stream)?;
    x = attention_block(weights, "decoder.mid_block.attentions.0", &x, stream)?;
    x = res_block(weights, "decoder.mid_block.resnets.1", &x, stream)?;
    // Each upsample conv already emits the next block's width:
    // 384→192, 384→192, 192→96. up_block3 stays at 96.
    x = up_block(weights, "decoder.up_block0", &x, true, stream)?;
    x = up_block(weights, "decoder.up_block1", &x, true, stream)?;
    x = up_block(weights, "decoder.up_block2", &x, true, stream)?;
    x = up_block(weights, "decoder.up_block3", &x, false, stream)?;
    x = channel_norm(weights, "decoder.norm_out", &x, stream)?;
    x = x.silu(stream)?;
    x = causal_conv3d(weights, "decoder.conv_out.conv3d", &x, 1, stream)?;
    let shape = x.shape();
    if shape.len() != 5 || shape[1] != 3 || shape[2] != 1 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("decoded shape {shape:?} is not [1, 3, 1, H, W]"),
        ));
    }
    x.squeeze_axis(2, stream)
}

/// Causal conv: zero-pad `2*pad` frames before, symmetric spatial pad.
/// `pad` is 1 for 3x3x3 kernels and 0 for 1x1x1.
fn causal_conv3d(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    pad: i32,
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    if pad > 0 {
        x = x.pad(&[2], &[2 * pad], &[0], 0.0, "constant", stream)?;
        x = x.pad(&[3, 4], &[pad, pad], &[pad, pad], 0.0, "constant", stream)?;
    }
    let weight = weights.tensor(&format!("{name}.weight"))?;
    let shape = weight.shape();
    if shape.len() != 5 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{name} kernel {shape:?} is not [O, kT, kH, kW, I]"),
        ));
    }
    let x = x.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
    let out = Array::conv3d(&x, weight, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, stream)?;
    let out = out.transpose_axes(&[0, 4, 1, 2, 3], stream)?;
    let bias_name = format!("{name}.bias");
    if weights.has(&bias_name) {
        let bias = weights.tensor(&bias_name)?;
        let channels = bias.shape()[0];
        let bias = bias.reshape(&[1, channels, 1, 1, 1], stream)?;
        return Array::add(&out, &bias, stream);
    }
    Ok(out)
}

/// Channel L2 norm over NCTHW: `x / max(sqrt(sum_c x^2), eps) * sqrt(C) * w`.
fn channel_norm(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    if shape.len() != 5 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{name} input {shape:?} is not NCTHW"),
        ));
    }
    let channels = shape[1];
    let dtype = input.dtype();
    let x = input.astype(Dtype::Float32, stream)?;
    let square = Array::multiply(&x, &x, stream)?;
    // sum over channels = mean * C. MLX has no grouped conv3d, so the
    // reduction runs through mean_axes instead of a depthwise 1x1x1 conv.
    let sum = Array::multiply(
        &square.mean_axes(&[1], true, stream)?,
        &Array::from_f32_slice(&[channels as f32], &[])?,
        stream,
    )?;
    let norm = sum.sqrt(stream)?;
    let denom = Array::maximum(&norm, &Array::from_f32_slice(&[NORM_EPS], &[])?, stream)?;
    let normalized = Array::divide(&x, &denom, stream)?;
    let scale = Array::from_f32_slice(&[(channels as f32).sqrt()], &[])?;
    let weight = weights.tensor(&format!("{name}.weight"))?;
    if weight.shape() != vec![channels] {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{name}.weight {:?} is not [{channels}]", weight.shape()),
        ));
    }
    let weight = weight
        .astype(Dtype::Float32, stream)?
        .reshape(&[1, channels, 1, 1, 1], stream)?;
    let out = Array::multiply(
        &Array::multiply(&normalized, &scale, stream)?,
        &weight,
        stream,
    )?;
    out.astype(dtype, stream)
}

fn res_block(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let mut residual = input.clone();
    let mut x = channel_norm(weights, &format!("{prefix}.norm1"), input, stream)?;
    x = x.silu(stream)?;
    x = causal_conv3d(weights, &format!("{prefix}.conv1.conv3d"), &x, 1, stream)?;
    x = channel_norm(weights, &format!("{prefix}.norm2"), &x, stream)?;
    x = x.silu(stream)?;
    x = causal_conv3d(weights, &format!("{prefix}.conv2.conv3d"), &x, 1, stream)?;
    let skip = format!("{prefix}.skip_conv.conv3d.weight");
    if weights.has(&skip) {
        residual = causal_conv3d(
            weights,
            &format!("{prefix}.skip_conv.conv3d"),
            &residual,
            0,
            stream,
        )?;
    }
    Array::add(&x, &residual, stream)
}

/// Per-frame spatial self-attention with 1x1 conv projections (single head,
/// head dim = channels). Matches main `QwenImageAttentionBlock3D`.
fn attention_block(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, channels, time, height, width) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let frames = batch * time;
    let area = height * width;
    let x = input
        .transpose_axes(&[0, 2, 1, 3, 4], stream)?
        .reshape(&[frames, channels, height, width], stream)?
        .expand_dims(2, stream)?;
    let x = channel_norm(weights, &format!("{prefix}.norm"), &x, stream)?;
    let x = x
        .squeeze_axis(2, stream)?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let qkv = conv2d(weights, &format!("{prefix}.to_qkv"), &x, stream)?;
    let qkv = qkv
        .transpose_axes(&[0, 3, 1, 2], stream)?
        .reshape(&[frames, 1, channels * 3, area], stream)?
        .transpose_axes(&[0, 1, 3, 2], stream)?;
    let parts = qkv.split_sections(&[channels, channels * 2], -1, stream)?;
    let (q, k, v) = (&parts[0], &parts[1], &parts[2]);
    let attended = Array::scaled_dot_product_attention(
        &q.astype(Dtype::Float32, stream)?,
        &k.astype(Dtype::Float32, stream)?,
        &v.astype(Dtype::Float32, stream)?,
        1.0 / (channels as f32).sqrt(),
        "",
        None,
        stream,
    )?
    .astype(q.dtype(), stream)?;
    let attended = attended
        .squeeze_axis(1, stream)?
        .transpose_axes(&[0, 2, 1], stream)?
        .reshape(&[frames, channels, height, width], stream)?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let projected = conv2d(weights, &format!("{prefix}.proj"), &attended, stream)?;
    let projected = projected
        .transpose_axes(&[0, 3, 1, 2], stream)?
        .reshape(&[batch, time, channels, height, width], stream)?
        .transpose_axes(&[0, 2, 1, 3, 4], stream)?;
    Array::add(&projected, input, stream)
}

fn conv2d(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let weight = weights.tensor(&format!("{name}.weight"))?;
    let shape = weight.shape();
    if shape.len() != 4 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{name} kernel {shape:?} is not [O, kH, kW, I]"),
        ));
    }
    let kernel = [shape[1], shape[2]];
    let pad = [(kernel[0] - 1) / 2, (kernel[1] - 1) / 2];
    let out = Array::conv2d(input, weight, [1, 1], pad, [1, 1], 1, stream)?;
    let bias_name = format!("{name}.bias");
    if weights.has(&bias_name) {
        return Array::add(&out, weights.tensor(&bias_name)?, stream);
    }
    Ok(out)
}

fn down_block(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    downsample: bool,
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    let mut index = 0u32;
    while weights.has(&format!("{prefix}.resnets.{index}.norm1.weight")) {
        x = res_block(weights, &format!("{prefix}.resnets.{index}"), &x, stream)?;
        index += 1;
    }
    if index == 0 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{prefix} has no resnets"),
        ));
    }
    if downsample {
        x = spatial_downsample(weights, &format!("{prefix}.downsamplers.0"), &x, stream)?;
    }
    Ok(x)
}

/// Image-path downsample: pad one pixel on the bottom and right, then a
/// stride-2 3x3. `time_conv` stays on disk and is not applied.
fn spatial_downsample(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, channels, time, height, width) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let x = input
        .transpose_axes(&[0, 2, 1, 3, 4], stream)?
        .reshape(&[batch * time, channels, height, width], stream)?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let x = x.pad(&[1, 2], &[0, 0], &[1, 1], 0.0, "constant", stream)?;
    let weight = weights.tensor(&format!("{prefix}.resample_conv.weight"))?;
    let kernel = weight.shape();
    if kernel.len() != 4 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{prefix} resample kernel {kernel:?} is not [O, 3, 3, I]"),
        ));
    }
    let out = Array::conv2d(&x, weight, [2, 2], [0, 0], [1, 1], 1, stream)?;
    let bias_name = format!("{prefix}.resample_conv.bias");
    let out = if weights.has(&bias_name) {
        Array::add(&out, weights.tensor(&bias_name)?, stream)?
    } else {
        out
    };
    let new_h = out.shape()[1];
    let new_w = out.shape()[2];
    let new_c = out.shape()[3];
    out.transpose_axes(&[0, 3, 1, 2], stream)?
        .reshape(&[batch, time, new_c, new_h, new_w], stream)?
        .transpose_axes(&[0, 2, 1, 3, 4], stream)
}

/// Three resnets, then optional nearest-2x + 3x3 `resample_conv`.
/// `time_conv` stays on disk for video and is not applied on this image path.
fn up_block(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    upsample: bool,
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    let mut index = 0u32;
    while weights.has(&format!("{prefix}.resnets.{index}.norm1.weight")) {
        x = res_block(weights, &format!("{prefix}.resnets.{index}"), &x, stream)?;
        index += 1;
    }
    if index == 0 {
        return Err(Error::new(
            "Qwen-Image VAE",
            format!("{prefix} has no resnets"),
        ));
    }
    if upsample {
        x = spatial_upsample(weights, &format!("{prefix}.upsamplers.0"), &x, stream)?;
    }
    Ok(x)
}

fn spatial_upsample(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, channels, time, height, width) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let x = input
        .transpose_axes(&[0, 2, 1, 3, 4], stream)?
        .reshape(&[batch * time, channels, height, width], stream)?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let x = x.repeat_axis(2, 1, stream)?.repeat_axis(2, 2, stream)?;
    let x = conv2d(weights, &format!("{prefix}.resample_conv"), &x, stream)?;
    let new_channels = x.shape()[3];
    let x = x
        .transpose_axes(&[0, 3, 1, 2], stream)?
        .reshape(&[batch, time, new_channels, height * 2, width * 2], stream)?
        .transpose_axes(&[0, 2, 1, 3, 4], stream)?;
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::{unpack_latent, LATENTS_MEAN, LATENTS_STD};
    use danmo_mlx::Stream;

    #[test]
    fn unpack_is_the_inverse_of_the_diffusers_pack() {
        let stream = Stream::cpu().unwrap();
        // Build a known [1,16,4,6] latent, pack it host-side with the
        // diffusers formula, then unpack through the graph and compare.
        let (h, w) = (4i32, 6i32);
        let (gh, gw) = (h / 2, w / 2);
        let mut latent = vec![0f32; (16 * h * w) as usize];
        for (index, value) in latent.iter_mut().enumerate() {
            *value = (index % 97) as f32 * 0.01 - 0.4;
        }
        let mut packed = vec![0f32; (64 * gh * gw) as usize];
        for gy in 0..gh as usize {
            for gx in 0..gw as usize {
                for c in 0..16usize {
                    for py in 0..2usize {
                        for px in 0..2usize {
                            let src =
                                c * (h * w) as usize + (gy * 2 + py) * w as usize + gx * 2 + px;
                            let dst = ((gy * gw as usize + gx) * 16 + c) * 4 + py * 2 + px;
                            packed[dst] = latent[src];
                        }
                    }
                }
            }
        }
        let unpacked = unpack_latent(&packed, gh, gw, &stream).unwrap();
        unpacked.eval().unwrap();
        assert_eq!(unpacked.shape(), vec![1, 16, 1, h, w]);
        let got = unpacked.to_vec_f32(&stream).unwrap();
        assert_eq!(got, latent);
    }

    #[test]
    fn latent_stats_are_sixteen_channels() {
        assert_eq!(LATENTS_MEAN.len(), 16);
        assert_eq!(LATENTS_STD.len(), 16);
        assert!(LATENTS_STD.iter().all(|value| *value > 0.0));
    }
}
