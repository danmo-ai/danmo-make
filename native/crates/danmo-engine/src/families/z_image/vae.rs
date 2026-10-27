//! Z-Image pixel decode.
//!
//! The full graph is the shared AutoencoderKL decoder with this family's spec:
//! 16 latent channels, blocks 128/256/512/512, 32 groups, scaling 0.3611,
//! shift 0.1159. A file whose only tensor is `decoder.conv.weight` stays the
//! 1×1 preview. Anything else fails.

use crate::common::png::encode_rgb;
use crate::common::vae::{self, VaeSpec};
use crate::runtime::{preview_conv, GgmlContext};
use crate::weights::{GgufWeights, WeightError};

fn ae_spec() -> VaeSpec {
    VaeSpec {
        latent_channels: 16,
        block_out: vec![128, 256, 512, 512],
        resnets: 3,
        groups: 32,
        scaling: 0.3611,
        shift: 0.1159,
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
    let samples = vae::decode_samples(
        ctx,
        &vae,
        latent,
        &ae_spec(),
        in_ch,
        grid_w,
        grid_h,
        width,
        height,
    )?;
    let rgb = vae::rgb_bytes(&samples, width, height);
    Ok(encode_rgb(width, height, &rgb))
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
    let planar = repack_channel_fast(latent, grid_w, grid_h, in_ch);
    let decoded = preview_conv(ctx, &planar, &weight, in_ch, grid_w, grid_h)?;
    let rgb = preview_rgb(&decoded, grid_w as u32, grid_h as u32);
    let rgb = nearest_expand(&rgb, grid_w as u32, grid_h as u32, width, height);
    Ok(crate::common::png::encode_rgb(width, height, &rgb))
}

/// `[W, H, C]` (x fastest) to `[C, W*H]` (channel fastest) for `preview_conv`.
fn repack_channel_fast(src: &[f32], width: i64, height: i64, channels: i64) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    let channels = channels as usize;
    let mut out = vec![0f32; src.len()];
    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let from = x + y * width + channel * width * height;
                let to = channel + (x + y * width) * channels;
                out[to] = src[from];
            }
        }
    }
    out
}

fn preview_rgb(planar: &[f32], grid_w: u32, grid_h: u32) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::decode;
    use crate::common::vae::parameter_list;
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd};
    use std::path::PathBuf;

    #[test]
    fn flux_decoder_names_follow_diffusers() {
        let params = parameter_list(&super::ae_spec()).unwrap();
        let names: Vec<&str> = params.iter().map(|param| param.name.as_str()).collect();
        assert!(names.contains(&"decoder.conv_in.weight"));
        let conv_in = params
            .iter()
            .find(|param| param.name == "decoder.conv_in.weight")
            .unwrap();
        assert_eq!(conv_in.shape, vec![3, 3, 16, 512]);
        let shortcut = params
            .iter()
            .find(|param| param.name == "decoder.up_blocks.2.resnets.0.conv_shortcut.weight")
            .unwrap();
        assert_eq!(shortcut.shape, vec![1, 1, 512, 256]);
        assert!(names.contains(&"decoder.up_blocks.3.resnets.0.conv_shortcut.weight"));
        assert!(!names
            .iter()
            .any(|name| name.contains("up_blocks.0") && name.contains("shortcut")));
        assert!(!names
            .iter()
            .any(|name| name.contains("up_blocks.1") && name.contains("shortcut")));
        assert!(!names
            .iter()
            .any(|name| name.contains("up_blocks.3.upsamplers")));
        assert!(names.contains(&"decoder.up_blocks.2.upsamplers.0.conv.weight"));
        assert!(names.contains(&"decoder.up_blocks.0.resnets.2.conv2.weight"));
        assert!(!names.iter().any(|name| name.contains("resnets.3")));
        assert!(names.contains(&"decoder.mid_block.attentions.0.to_out.0.weight"));
        let query = params
            .iter()
            .find(|param| param.name == "decoder.mid_block.attentions.0.to_q.weight")
            .unwrap();
        assert_eq!(query.shape, vec![512, 512]);
        let norm = params
            .iter()
            .find(|param| param.name == "decoder.conv_norm_out.weight")
            .unwrap();
        assert_eq!(norm.shape, vec![128]);
        let out = params
            .iter()
            .find(|param| param.name == "decoder.conv_out.weight")
            .unwrap();
        assert_eq!(out.shape, vec![3, 3, 128, 3]);
    }

    #[test]
    fn other_vae_files_fail_loud() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let dir = scratch("vae-fail");
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
        let err = decode(&ctx, &[unknown], &[0.0], 1, 1, 1, 1, 1).unwrap_err();
        assert!(err.to_string().contains("vae graph is not built"), "{err}");

        let mixed = dir.join("mixed.gguf");
        write_f32_nd(
            &mixed,
            &[
                F32Nd {
                    name: "decoder.conv.weight",
                    ne: [1, 3, 1, 1],
                    data: &[0.0, 0.0, 0.0],
                },
                F32Nd {
                    name: "decoder.extra",
                    ne: [1, 1, 1, 1],
                    data: &[0.0],
                },
            ],
        )
        .unwrap();
        let err = decode(&ctx, &[mixed], &[0.0], 1, 1, 1, 1, 1).unwrap_err();
        assert!(err.to_string().contains("vae graph is not built"), "{err}");

        let size = dir.join("size.gguf");
        write_f32_nd(
            &size,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [1, 1, 1, 1],
                data: &[0.0],
            }],
        )
        .unwrap();
        let latent = vec![0.0f32; 16 * 2 * 2];
        let err = decode(&ctx, &[size], &latent, 16, 2, 2, 64, 64).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("16x16") && message.contains("64x64"),
            "{message}"
        );

        let channels = dir.join("channels.gguf");
        write_f32_nd(
            &channels,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [1, 1, 1, 1],
                data: &[0.0],
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[channels], &[0.0; 4], 4, 1, 1, 8, 8).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("4 channels") && message.contains("16"),
            "{message}"
        );

        let shape = dir.join("shape.gguf");
        let bad = vec![0.0f32; 3 * 3 * 2 * 4];
        write_f32_nd(
            &shape,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [3, 3, 2, 4],
                data: &bad,
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[shape], &[0.0; 16], 16, 1, 1, 8, 8).unwrap_err();
        assert!(err.to_string().contains("shape"), "{err}");

        let missing = dir.join("missing.gguf");
        let conv_in = vec![0.0f32; 3 * 3 * 16 * 512];
        write_f32_nd(
            &missing,
            &[F32Nd {
                name: "decoder.conv_in.weight",
                ne: [3, 3, 16, 512],
                data: &conv_in,
            }],
        )
        .unwrap();
        let err = decode(&ctx, &[missing], &[0.0; 16], 16, 1, 1, 8, 8).unwrap_err();
        assert!(err.to_string().contains("decoder.conv_in.bias"), "{err}");

        let preview = dir.join("preview.gguf");
        write_f32_nd(
            &preview,
            &[F32Nd {
                name: "decoder.conv.weight",
                ne: [2, 3, 1, 1],
                data: &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }],
        )
        .unwrap();
        let png = decode(&ctx, &[preview], &[0.4, 0.2], 2, 1, 1, 2, 2).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
        assert_eq!((width, height), (2, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
