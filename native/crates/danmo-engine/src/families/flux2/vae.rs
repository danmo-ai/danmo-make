//! Flux2 pixel decode.
//!
//! Packed DiT latents are 128 channels at image/16. This family applies the
//! Flux2 batch norm (`x * sqrt(var + 1e-4) + mean`), unpatches each group of 4
//! channels into a 2×2 block (32 channels at image/8), applies scaling 1 and
//! shift 0, then the 1×1 `post_quant_conv`. The shared AutoencoderKL decoder
//! runs after that with scaling 1 and shift 0, so the scale is not applied twice.
//! A file whose only tensor is `decoder.conv.weight` stays the 1×1 preview.

use crate::common::png::encode_rgb;
use crate::common::vae::{self, VaeSpec};
use crate::runtime::{preview_conv, shape, GgmlContext, Tensor};
use crate::weights::{GgufWeights, WeightError};

const PACKED_CHANNELS: i64 = 128;
const LATENT_CHANNELS: i64 = 32;
const BN_EPS: f32 = 1e-4;
const SCALING: f32 = 1.0;
const SHIFT: f32 = 0.0;

fn decoder_spec() -> VaeSpec {
    VaeSpec {
        latent_channels: LATENT_CHANNELS,
        block_out: vec![128, 256, 512, 512],
        resnets: 3,
        groups: 32,
        scaling: 1.0,
        shift: 0.0,
        attention: true,
    }
}

pub(super) fn decode(
    ctx: &GgmlContext,
    paths: &[std::path::PathBuf],
    latent: &[f32],
    in_ch: i64,
    grid_w: i64,
    grid_h: i64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, WeightError> {
    let vae = GgufWeights::open_parts(paths)?;
    let names: Vec<String> = vae.names().map(str::to_string).collect();
    if names.len() == 1 && names.first().map(String::as_str) == Some("decoder.conv.weight") {
        return decode_preview(ctx, &vae, latent, in_ch, grid_w, grid_h, width, height);
    }
    if !vae.contains("decoder.conv_in.weight") {
        return Err(WeightError::NotBuilt("vae graph is not built".into()));
    }
    if in_ch != PACKED_CHANNELS {
        return Err(WeightError::NotBuilt(format!(
            "flux2 packed latent has {in_ch} channels, the decoder reads {PACKED_CHANNELS}"
        )));
    }
    let mean = vae.f32_widened("bn.running_mean", &[PACKED_CHANNELS])?;
    let var = vae.f32_widened("bn.running_var", &[PACKED_CHANNELS])?;
    let weight = vae.f32_widened(
        "post_quant_conv.weight",
        &[1, 1, LATENT_CHANNELS, LATENT_CHANNELS],
    )?;
    let bias = vae.f32_widened("post_quant_conv.bias", &[LATENT_CHANNELS])?;
    let unpacked = unpatch(&latent, &mean, &var, grid_w, grid_h, PACKED_CHANNELS)?;
    let scaled = scale_shift(&unpacked, SCALING, SHIFT)?;
    let latent_w = grid_w * 2;
    let latent_h = grid_h * 2;
    let quantized = post_quant(ctx, &scaled, latent_w, latent_h, &weight, &bias)?;
    let spec = decoder_spec();
    let samples = vae::decode_samples(
        ctx,
        &vae,
        &quantized,
        &spec,
        LATENT_CHANNELS,
        latent_w,
        latent_h,
        width,
        height,
    )?;
    let rgb = vae::rgb_bytes(&samples, width, height);
    Ok(encode_rgb(width, height, &rgb))
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

fn post_quant(
    ctx: &GgmlContext,
    latent: &[f32],
    width: i64,
    height: i64,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, WeightError> {
    let channels = LATENT_CHANNELS;
    let expect = (width * height * channels) as usize;
    if latent.len() != expect {
        return Err(WeightError::Shape {
            name: "post_quant latent".into(),
            expected: vec![width, height, channels],
            got: vec![latent.len() as i64],
        });
    }
    if weight.len() != (channels * channels) as usize || bias.len() != channels as usize {
        return Err(WeightError::Shape {
            name: "post_quant_conv.weight".into(),
            expected: vec![1, 1, channels, channels],
            got: vec![weight.len() as i64, bias.len() as i64],
        });
    }
    let mut builder = ctx.builder(8 * 1024 * 1024)?;
    let x = builder.f32_3d("latent", width, height, channels, true);
    let x = nn(builder.cont(builder.reshape_4d(x, width, height, channels, 1)))?;
    let kernel = builder.f32_4d("post_quant_conv.weight", 1, 1, channels, channels, true);
    let y = nn(builder.conv2d_direct(kernel, x, 1, 1, 0, 0, 1, 1))?;
    let y = nn(builder.cont(y))?;
    let y = expect_shape(y, [width, height, channels, 1], "post_quant_conv")?;
    let bias_t = builder.f32_1d("post_quant_conv.bias", channels, true);
    let bias_t = nn(builder.cont(builder.reshape_4d(bias_t, 1, 1, channels, 1)))?;
    let y = nn(builder.add(y, nn(builder.repeat(bias_t, y))?))?;
    let y = nn(builder.cont(y))?;
    let y = expect_shape(y, [width, height, channels, 1], "post_quant_conv")?;
    let mut graph = builder.finish(y)?;
    graph.set_f32("latent", latent)?;
    graph.set_f32("post_quant_conv.weight", weight)?;
    graph.set_f32("post_quant_conv.bias", bias)?;
    let out = graph.compute()?;
    if out.len() != expect {
        return Err(WeightError::Shape {
            name: "post_quant_conv".into(),
            expected: vec![width, height, channels],
            got: vec![out.len() as i64],
        });
    }
    if out.iter().any(|value| !value.is_finite()) {
        return Err(WeightError::NotBuilt(
            "post_quant_conv produced a non-finite sample".into(),
        ));
    }
    Ok(out)
}

fn decode_preview(
    ctx: &GgmlContext,
    vae: &GgufWeights,
    latent: &[f32],
    in_ch: i64,
    grid_w: i64,
    grid_h: i64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, WeightError> {
    let weight = vae.f32_widened("decoder.conv.weight", &[in_ch, 3])?;
    let seq = grid_w * grid_h;
    if latent.len() as i64 != in_ch * seq {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![in_ch, seq],
            got: vec![latent.len() as i64],
        });
    }
    let planar = preview_conv(ctx, latent, &weight, in_ch, grid_w, grid_h)?;
    let rgb = planar_rgb(&planar, grid_w as u32, grid_h as u32);
    let rgb = nearest_expand(&rgb, grid_w as u32, grid_h as u32, width, height);
    Ok(encode_rgb(width, height, &rgb))
}

fn planar_rgb(planar: &[f32], grid_w: u32, grid_h: u32) -> Vec<u8> {
    let width = grid_w as usize;
    let height = grid_h as usize;
    let plane = width * height;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let v = planar
                    .get(x + y * width + channel * plane)
                    .copied()
                    .unwrap_or(0.0);
                rgb[(y * width + x) * 3 + channel] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    rgb
}

fn nearest_expand(src: &[u8], grid_w: u32, grid_h: u32, width: u32, height: u32) -> Vec<u8> {
    if grid_w == width && grid_h == height {
        return src.to_vec();
    }
    let mut out = vec![0u8; (width as usize) * (height as usize) * 3];
    for y in 0..height {
        let sy = ((y as u64) * (grid_h as u64) / (height as u64)) as usize;
        for x in 0..width {
            let sx = ((x as u64) * (grid_w as u64) / (width as u64)) as usize;
            let from = (sy * grid_w as usize + sx) * 3;
            let to = (y as usize * width as usize + x as usize) * 3;
            out[to..to + 3].copy_from_slice(&src[from..from + 3]);
        }
    }
    out
}

fn expect_shape(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, WeightError> {
    let got = shape(tensor);
    if got != ne {
        return Err(WeightError::NotBuilt(format!(
            "{what} is {got:?}, the graph expects {ne:?}"
        )));
    }
    Ok(tensor)
}

fn nn(tensor: Tensor) -> Result<Tensor, WeightError> {
    if tensor.is_null() {
        Err(WeightError::NotBuilt(
            "post_quant_conv graph op returned null".into(),
        ))
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, decoder_spec, unpatch, BN_EPS};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd};
    use std::path::PathBuf;

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
    fn preview_file_stays_a_png() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("preview");
        let path = dir.join("vae.gguf");
        write_f32_nd(
            &path,
            &[F32Nd {
                name: "decoder.conv.weight",
                ne: [2, 3, 1, 1],
                data: &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }],
        )
        .unwrap();
        let png = decode(&ctx, &[path], &[0.2, 0.4], 2, 1, 1, 2, 2).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_vae_files_fail_loud() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("fail");
        let unknown = dir.join("unknown.gguf");
        write_f32_nd(
            &unknown,
            &[F32Nd {
                name: "encoder.conv_in.weight",
                ne: [1, 1, 1, 1],
                data: &[0.0],
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[unknown], &[0.0], 128, 1, 1, 16, 16).unwrap_err();
        assert!(err.to_string().contains("vae graph is not built"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn packed_channel_count_must_be_128() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("channels");
        let path = dir.join("vae.gguf");
        write_f32_nd(
            &path,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [1, 1, 1, 1],
                data: &[0.0],
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[path], &[0.0; 4], 4, 1, 1, 16, 16).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("4 channels") && message.contains("128"),
            "{message}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_batch_norm_fails() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("bn");
        let path = dir.join("vae.gguf");
        write_f32_nd(
            &path,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [1, 1, 1, 1],
                data: &[0.0],
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[path], &[0.0; 128], 128, 1, 1, 16, 16).unwrap_err();
        assert!(err.to_string().contains("bn.running_mean"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unpack_reaches_the_shared_decoder() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("reach");
        let path = front_end_file(&dir);
        let err = decode(&ctx, &[path], &[0.1; 128], 128, 1, 1, 1, 1).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("16x16") && message.contains("1x1"),
            "{message}"
        );
        let spec = decoder_spec();
        assert_eq!(spec.scaling, 1.0);
        assert_eq!(spec.shift, 0.0);
        assert_eq!(spec.latent_channels, 32);
        assert_eq!(spec.resnets, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn post_quant_matches_the_channel_dot_product() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("quant");
        let path = front_end_file(&dir);
        let vae = crate::weights::GgufWeights::open(&path).unwrap();
        let weight = vae
            .f32_exact("post_quant_conv.weight", &[1, 1, 32, 32])
            .unwrap();
        let bias = vae.f32_exact("post_quant_conv.bias", &[32]).unwrap();
        let latent = unpatch(&[0.25; 128], &[0.0; 128], &[0.0; 128], 1, 1, 128).unwrap();
        let got = super::post_quant(&ctx, &latent, 2, 2, &weight, &bias).unwrap();
        let expect = host_conv1x1(&latent, 2, 2, &weight, &bias);
        let max = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-5, "max abs {max}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn front_end_file(dir: &PathBuf) -> PathBuf {
        let path = dir.join("vae.gguf");
        let mean = vec![0.1f32; 128];
        let var = vec![0.25f32; 128];
        let mut weight = vec![0.0f32; 32 * 32];
        for channel in 0..32 {
            weight[channel + channel * 32] = 0.5;
        }
        let bias = vec![0.05f32; 32];
        write_f32_nd(
            &path,
            &[
                F32Nd {
                    name: "decoder.conv_in.weight",
                    ne: [1, 1, 1, 1],
                    data: &[0.0],
                },
                F32Nd {
                    name: "bn.running_mean",
                    ne: [128, 1, 1, 1],
                    data: &mean,
                },
                F32Nd {
                    name: "bn.running_var",
                    ne: [128, 1, 1, 1],
                    data: &var,
                },
                F32Nd {
                    name: "post_quant_conv.weight",
                    ne: [1, 1, 32, 32],
                    data: &weight,
                },
                F32Nd {
                    name: "post_quant_conv.bias",
                    ne: [32, 1, 1, 1],
                    data: &bias,
                },
            ],
        )
        .unwrap();
        path
    }

    fn host_conv1x1(
        input: &[f32],
        width: usize,
        height: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let channels = bias.len();
        let mut out = vec![0.0f32; width * height * channels];
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

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("danmo-flux2-vae-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
