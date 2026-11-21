//! Flux2 latent pack and unpack.
//!
//! Packed DiT latents are 128 channels at image/16. Decode applies the Flux2
//! batch norm (`x * sqrt(var + 1e-4) + mean`), unpatches each group of 4
//! channels into a 2×2 block (32 channels at image/8), applies scaling 1 and
//! shift 0, then the 1×1 `post_quant_conv`. Encode is the inverse of that
//! front end: `quant_conv` on the encoder `conv_out`, the mean half, an even
//! crop, 2×2 patchify, then `(z - mean) / sqrt(var + 1e-4)`.

use crate::weights::WeightError;

const PACKED_CHANNELS: i64 = 128;
const LATENT_CHANNELS: i64 = 32;
const BN_EPS: f32 = 1e-4;
const SCALING: f32 = 1.0;
const SHIFT: f32 = 0.0;

/// NCHW `conv_out` `[2 * latent, H, W]` to the token-major packed DiT latent
/// `[H/2 * W/2, 128]`. `quant_conv` is 1×1 with weight index `ic + oc * in`.
pub(super) fn pack_from_conv_out(
    conv_out: &[f32],
    latent_h: i64,
    latent_w: i64,
    quant_weight: &[f32],
    quant_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
) -> Result<Vec<f32>, WeightError> {
    let doubled = (LATENT_CHANNELS * 2) as usize;
    if latent_h < 2 || latent_w < 2 {
        return Err(WeightError::NotBuilt(format!(
            "flux2 edit latent grid {latent_w}x{latent_h} is too small to patchify"
        )));
    }
    let plane = (latent_h * latent_w) as usize;
    if conv_out.len() != doubled * plane {
        return Err(WeightError::Shape {
            name: "encoder.conv_out".into(),
            expected: vec![doubled as i64, latent_h, latent_w],
            got: vec![conv_out.len() as i64],
        });
    }
    if quant_bias.len() != doubled || quant_weight.len() != doubled * doubled {
        return Err(WeightError::Shape {
            name: "quant_conv.weight".into(),
            expected: vec![doubled as i64, doubled as i64],
            got: vec![quant_weight.len() as i64, quant_bias.len() as i64],
        });
    }
    if bn_mean.len() != PACKED_CHANNELS as usize || bn_var.len() != PACKED_CHANNELS as usize {
        return Err(WeightError::Shape {
            name: "bn.running_mean".into(),
            expected: vec![PACKED_CHANNELS],
            got: vec![bn_mean.len() as i64, bn_var.len() as i64],
        });
    }
    let mut mean = vec![0f32; plane * LATENT_CHANNELS as usize];
    for oc in 0..LATENT_CHANNELS as usize {
        for index in 0..plane {
            let mut acc = quant_bias[oc];
            for ic in 0..doubled {
                acc += conv_out[index + ic * plane] * quant_weight[ic + oc * doubled];
            }
            let scaled = (acc - SHIFT) * SCALING;
            if !scaled.is_finite() {
                return Err(WeightError::NotBuilt(
                    "flux2 quant_conv produced a non-finite sample".into(),
                ));
            }
            mean[index + oc * plane] = scaled;
        }
    }
    let height = latent_h & !1;
    let width = latent_w & !1;
    let packed_w = width / 2;
    let packed_h = height / 2;
    let packed_c = PACKED_CHANNELS as usize;
    let mut packed = vec![0f32; packed_c * (packed_w * packed_h) as usize];
    for y in 0..height {
        for x in 0..width {
            let px = x / 2;
            let py = y / 2;
            let pw = x % 2;
            let ph = y % 2;
            for channel in 0..LATENT_CHANNELS {
                let pc = (channel * 4 + ph * 2 + pw) as usize;
                let value = mean[(x + y * latent_w) as usize + channel as usize * plane];
                let var = bn_var[pc];
                if !var.is_finite() || var < 0.0 {
                    return Err(WeightError::NotBuilt(format!(
                        "bn.running_var[{pc}] is not a finite non-negative value"
                    )));
                }
                let normed = (value - bn_mean[pc]) / (var + BN_EPS).sqrt();
                if !normed.is_finite() {
                    return Err(WeightError::NotBuilt(
                        "flux2 batch-norm encode produced a non-finite value".into(),
                    ));
                }
                let pixel = (px + py * packed_w) as usize;
                packed[pixel * packed_c + pc] = normed;
            }
        }
    }
    Ok(packed)
}

/// Channel-fast packed latent `[C, W*H]` to decoder layout `[W*2, H*2, C/4]`.
/// Channel `c_out * 4 + ph * 2 + pw` at `(x, y)` lands at `(x * 2 + pw, y * 2 + ph)`.
fn unpatch(
    packed: &[f32],
    mean: &[f32],
    var: &[f32],
    width: i64,
    height: i64,
    channels: i64,
) -> Result<Vec<f32>, WeightError> {
    if width < 1 || height < 1 {
        return Err(WeightError::NotBuilt("flux2 packed grid is empty".into()));
    }
    if channels < 4 || channels % 4 != 0 {
        return Err(WeightError::NotBuilt(format!(
            "flux2 packed channels {channels} are not a multiple of 4"
        )));
    }
    if mean.len() as i64 != channels || var.len() as i64 != channels {
        return Err(WeightError::Shape {
            name: "bn.running_mean".into(),
            expected: vec![channels],
            got: vec![mean.len() as i64, var.len() as i64],
        });
    }
    let pixels = width * height;
    if packed.len() as i64 != channels * pixels {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![channels, pixels],
            got: vec![packed.len() as i64],
        });
    }
    let mut std = Vec::with_capacity(channels as usize);
    for (index, value) in var.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(WeightError::NotBuilt(format!(
                "bn.running_var[{index}] is not a finite non-negative value"
            )));
        }
        let scale = (*value + BN_EPS).sqrt();
        if !scale.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "bn.running_var[{index}] scale is not finite"
            )));
        }
        std.push(scale);
    }
    for (index, value) in mean.iter().enumerate() {
        if !value.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "bn.running_mean[{index}] is not finite"
            )));
        }
    }
    let out_c = channels / 4;
    let out_w = width * 2;
    let out_h = height * 2;
    let mut out = vec![0f32; (out_c * out_w * out_h) as usize];
    for y in 0..height {
        for x in 0..width {
            for c_out in 0..out_c {
                for ph in 0..2i64 {
                    for pw in 0..2i64 {
                        let pc = c_out * 4 + ph * 2 + pw;
                        let src = (pc + (x + y * width) * channels) as usize;
                        let sample = packed[src] * std[pc as usize] + mean[pc as usize];
                        if !sample.is_finite() {
                            return Err(WeightError::NotBuilt(
                                "flux2 batch-norm latent is not finite".into(),
                            ));
                        }
                        let ox = x * 2 + pw;
                        let oy = y * 2 + ph;
                        let dst = (ox + oy * out_w + c_out * out_w * out_h) as usize;
                        out[dst] = sample;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Packed DiT latent to the 32-channel NCHW tensor the shared KL decoder reads.
/// `packed` is `[pixel, 128]` with the channel index fastest, matching both
/// the DiT token and `pack_from_conv_out`.
pub(super) fn kl_latent_from_packed(
    packed: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
    grid_w: i64,
    grid_h: i64,
) -> Result<Vec<f32>, WeightError> {
    let unpacked = unpatch(packed, mean, var, grid_w, grid_h, PACKED_CHANNELS)?;
    let scaled = scale_shift(&unpacked, SCALING, SHIFT)?;
    let width = (grid_w * 2) as usize;
    let height = (grid_h * 2) as usize;
    conv1x1_nchw(&scaled, width, height, weight, bias)
}

fn conv1x1_nchw(
    input: &[f32],
    width: usize,
    height: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, WeightError> {
    let channels = bias.len();
    if channels == 0 || weight.len() != channels * channels {
        return Err(WeightError::Shape {
            name: "post_quant_conv.weight".into(),
            expected: vec![channels as i64, channels as i64],
            got: vec![weight.len() as i64, bias.len() as i64],
        });
    }
    if input.len() != width * height * channels {
        return Err(WeightError::Shape {
            name: "post_quant latent".into(),
            expected: vec![width as i64, height as i64, channels as i64],
            got: vec![input.len() as i64],
        });
    }
    let mut out = vec![0.0f32; input.len()];
    for y in 0..height {
        for x in 0..width {
            for oc in 0..channels {
                let mut acc = bias[oc];
                for ic in 0..channels {
                    acc += input[x + y * width + ic * width * height] * weight[ic + oc * channels];
                }
                if !acc.is_finite() {
                    return Err(WeightError::NotBuilt(
                        "flux2 post_quant_conv produced a non-finite sample".into(),
                    ));
                }
                out[x + y * width + oc * width * height] = acc;
            }
        }
    }
    Ok(out)
}

fn scale_shift(samples: &[f32], scaling: f32, shift: f32) -> Result<Vec<f32>, WeightError> {
    if !scaling.is_finite() || scaling == 0.0 || !shift.is_finite() {
        return Err(WeightError::NotBuilt(format!(
            "flux2 vae scaling {scaling} shift {shift} is not usable"
        )));
    }
    let mut out = Vec::with_capacity(samples.len());
    for value in samples {
        let scaled = value / scaling + shift;
        if !scaled.is_finite() {
            return Err(WeightError::NotBuilt(
                "flux2 vae scale produced a non-finite sample".into(),
            ));
        }
        out.push(scaled);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{pack_from_conv_out, unpatch, BN_EPS};

    #[test]
    fn unpatch_follows_the_flux2_channel_groups() {
        let channels = 4i64;
        let mean = [1.0, 2.0, 3.0, 4.0];
        let var = [0.0, 0.0, 0.0, 0.0];
        let packed = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let got = unpatch(&packed, &mean, &var, 2, 1, channels).unwrap();
        let std = BN_EPS.sqrt();
        assert_eq!(got.len(), 4 * 2);
        let at = |x: i64, y: i64| got[(x + y * 4) as usize];
        assert!((at(0, 0) - (10.0 * std + 1.0)).abs() < 1e-6);
        assert!((at(1, 0) - (20.0 * std + 2.0)).abs() < 1e-6);
        assert!((at(0, 1) - (30.0 * std + 3.0)).abs() < 1e-6);
        assert!((at(1, 1) - (40.0 * std + 4.0)).abs() < 1e-6);
        assert!((at(2, 0) - (50.0 * std + 1.0)).abs() < 1e-6);
        assert!((at(3, 0) - (60.0 * std + 2.0)).abs() < 1e-6);
        assert!((at(2, 1) - (70.0 * std + 3.0)).abs() < 1e-6);
        assert!((at(3, 1) - (80.0 * std + 4.0)).abs() < 1e-6);
    }

    #[test]
    fn negative_batch_norm_variance_fails() {
        let mut var = [0.0f32; 4];
        var[2] = -1.0;
        let err = unpatch(&[0.0; 4], &[0.0; 4], &var, 1, 1, 4).unwrap_err();
        assert!(err.to_string().contains("bn.running_var[2]"), "{err}");
    }

    #[test]
    fn pack_puts_the_mean_channel_on_the_even_patch() {
        let doubled = 64usize;
        let plane = 4usize;
        let mut conv_out = vec![0f32; doubled * plane];
        conv_out[0] = 0.25;
        let mut weight = vec![0f32; doubled * doubled];
        for channel in 0..doubled {
            weight[channel + channel * doubled] = 1.0;
        }
        let bias = vec![0f32; doubled];
        let bn_mean = vec![0f32; 128];
        let bn_var = vec![0f32; 128];
        let packed =
            pack_from_conv_out(&conv_out, 2, 2, &weight, &bias, &bn_mean, &bn_var).unwrap();
        assert_eq!(packed.len(), 128);
        let expect = 0.25 / BN_EPS.sqrt();
        assert!((packed[0] - expect).abs() < 1e-4, "{}", packed[0]);
        assert!(packed[1..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn post_quant_matches_the_channel_dot_product() {
        let mut weight = vec![0.0f32; 32 * 32];
        for channel in 0..32 {
            weight[channel + channel * 32] = 0.5;
        }
        let bias = vec![0.05f32; 32];
        let packed = vec![0.25f32; 128];
        let got =
            super::kl_latent_from_packed(&packed, &[0.0; 128], &[0.0; 128], &weight, &bias, 1, 1)
                .unwrap();
        let latent = unpatch(&packed, &[0.0; 128], &[0.0; 128], 1, 1, 128).unwrap();
        let expect = host_conv1x1(&latent, 2, 2, &weight, &bias);
        let max = got
            .iter()
            .zip(&expect)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-5, "kl path max abs {max}");
    }

    fn host_conv1x1(
        input: &[f32],
        width: usize,
        height: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let channels = bias.len();
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                for oc in 0..channels {
                    let mut acc = bias[oc];
                    for ic in 0..channels {
                        acc +=
                            input[x + y * width + ic * width * height] * weight[ic + oc * channels];
                    }
                    out[x + y * width + oc * width * height] = acc;
                }
            }
        }
        out
    }
}
