//! ACE-Step AutoencoderOobleck encoder (audio → latent) via ggml.
//!
//! Topology matches `/tmp/acestep-cpp/vae-enc.h`: conv1(2→128) → five downsample
//! blocks (3×ResUnit + snake + strided conv) → snake → conv2(2048→128).
//! Downsample 2×4×4×6×10 = **1920×**. Output 128ch = mean[64] + scale[64];
//! deterministic encode returns mean only as time-major `[T_latent, 64]`.

use super::vae::{
    conv1d_bias, declare_res_unit, f32_row_to_f16, fuse_weight_norm, res_unit,
    res_unit_placeholder, snake, AceVaeConfig, ResUnitW, AUDIO_UPSAMPLE, LATENT_CHANNELS,
    OUT_CHANNELS,
};
use crate::runtime::{shape, GraphBuilder, GgmlContext, ReusedGraph, Tensor, TYPE_F16, TYPE_F32};
use crate::weights::{GgufWeights, WeightError};

const ENC_IN_CH: [i64; 5] = [128, 128, 256, 512, 1024];
const ENC_OUT_CH: [i64; 5] = [128, 256, 512, 1024, 2048];
const ENC_STRIDES: [i64; 5] = [2, 4, 4, 6, 10];
const RES_DILATIONS: [i32; 3] = [1, 3, 9];
const GRAPH_MEM: usize = 512 * 1024 * 1024;
const MEAN_CHANNELS: i64 = 64;
const RAW_OUT_CHANNELS: i64 = 128;

#[derive(Debug, Clone)]
pub struct EncWeightSpec {
    pub name: String,
    pub ne: [i64; 4],
    pub ggml_type: u32,
}

/// Unfused GGUF names (weight-norm + raw snake) for layout checks.
pub fn weight_layout(_config: AceVaeConfig) -> Vec<EncWeightSpec> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<EncWeightSpec>, name: &str, ne: [i64; 4]| {
        out.push(EncWeightSpec {
            name: name.to_string(),
            ne,
            ggml_type: TYPE_F32,
        });
    };
    // weight_g is stored as [1,1,C,1] in published VAE GGUF (channel on ne[2]).
    push(&mut out, "encoder.conv1.weight_g", [1, 1, 128, 1]);
    push(&mut out, "encoder.conv1.weight_v", [7, 2, 128, 1]);
    push(&mut out, "encoder.conv1.bias", [128, 1, 1, 1]);
    for i in 0..5 {
        let ic = ENC_IN_CH[i];
        let oc = ENC_OUT_CH[i];
        let k = ENC_STRIDES[i] * 2;
        let p = format!("encoder.block.{i}");
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            push(&mut out, &format!("{ru}.snake1.alpha"), [1, ic, 1, 1]);
            push(&mut out, &format!("{ru}.snake1.beta"), [1, ic, 1, 1]);
            push(&mut out, &format!("{ru}.conv1.weight_g"), [1, 1, ic, 1]);
            push(&mut out, &format!("{ru}.conv1.weight_v"), [7, ic, ic, 1]);
            push(&mut out, &format!("{ru}.conv1.bias"), [ic, 1, 1, 1]);
            push(&mut out, &format!("{ru}.snake2.alpha"), [1, ic, 1, 1]);
            push(&mut out, &format!("{ru}.snake2.beta"), [1, ic, 1, 1]);
            push(&mut out, &format!("{ru}.conv2.weight_g"), [1, 1, ic, 1]);
            push(&mut out, &format!("{ru}.conv2.weight_v"), [1, ic, ic, 1]);
            push(&mut out, &format!("{ru}.conv2.bias"), [ic, 1, 1, 1]);
        }
        push(&mut out, &format!("{p}.snake1.alpha"), [1, ic, 1, 1]);
        push(&mut out, &format!("{p}.snake1.beta"), [1, ic, 1, 1]);
        push(&mut out, &format!("{p}.conv1.weight_g"), [1, 1, oc, 1]);
        push(&mut out, &format!("{p}.conv1.weight_v"), [k, ic, oc, 1]);
        push(&mut out, &format!("{p}.conv1.bias"), [oc, 1, 1, 1]);
    }
    push(&mut out, "encoder.snake1.alpha", [1, 2048, 1, 1]);
    push(&mut out, "encoder.snake1.beta", [1, 2048, 1, 1]);
    push(&mut out, "encoder.conv2.weight_g", [1, 1, 128, 1]);
    push(&mut out, "encoder.conv2.weight_v", [3, 2048, 128, 1]);
    push(&mut out, "encoder.conv2.bias", [128, 1, 1, 1]);
    out
}

/// Fused graph parameter names after weight-norm / snake prep.
pub fn graph_weight_layout(_config: AceVaeConfig) -> Vec<EncWeightSpec> {
    let mut out = Vec::new();
    out.push(EncWeightSpec {
        name: "encoder.conv1.weight".into(),
        ne: [7, 2, 128, 1],
        ggml_type: TYPE_F16,
    });
    out.push(EncWeightSpec {
        name: "encoder.conv1.bias".into(),
        ne: [128, 1, 1, 1],
        ggml_type: TYPE_F32,
    });
    for i in 0..5 {
        let ic = ENC_IN_CH[i];
        let oc = ENC_OUT_CH[i];
        let k = ENC_STRIDES[i] * 2;
        let p = format!("encoder.block.{i}");
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            for stem in ["snake1", "snake2"] {
                out.push(EncWeightSpec {
                    name: format!("{ru}.{stem}.exp_a"),
                    ne: [1, ic, 1, 1],
                    ggml_type: TYPE_F32,
                });
                out.push(EncWeightSpec {
                    name: format!("{ru}.{stem}.inv_b"),
                    ne: [1, ic, 1, 1],
                    ggml_type: TYPE_F32,
                });
            }
            out.push(EncWeightSpec {
                name: format!("{ru}.conv1.weight"),
                ne: [7, ic, ic, 1],
                ggml_type: TYPE_F16,
            });
            out.push(EncWeightSpec {
                name: format!("{ru}.conv1.bias"),
                ne: [ic, 1, 1, 1],
                ggml_type: TYPE_F32,
            });
            out.push(EncWeightSpec {
                name: format!("{ru}.conv2.weight"),
                ne: [1, ic, ic, 1],
                ggml_type: TYPE_F16,
            });
            out.push(EncWeightSpec {
                name: format!("{ru}.conv2.bias"),
                ne: [ic, 1, 1, 1],
                ggml_type: TYPE_F32,
            });
        }
        out.push(EncWeightSpec {
            name: format!("{p}.snake1.exp_a"),
            ne: [1, ic, 1, 1],
            ggml_type: TYPE_F32,
        });
        out.push(EncWeightSpec {
            name: format!("{p}.snake1.inv_b"),
            ne: [1, ic, 1, 1],
            ggml_type: TYPE_F32,
        });
        out.push(EncWeightSpec {
            name: format!("{p}.conv1.weight"),
            ne: [k, ic, oc, 1],
            ggml_type: TYPE_F16,
        });
        out.push(EncWeightSpec {
            name: format!("{p}.conv1.bias"),
            ne: [oc, 1, 1, 1],
            ggml_type: TYPE_F32,
        });
    }
    out.push(EncWeightSpec {
        name: "encoder.snake1.exp_a".into(),
        ne: [1, 2048, 1, 1],
        ggml_type: TYPE_F32,
    });
    out.push(EncWeightSpec {
        name: "encoder.snake1.inv_b".into(),
        ne: [1, 2048, 1, 1],
        ggml_type: TYPE_F32,
    });
    out.push(EncWeightSpec {
        name: "encoder.conv2.weight".into(),
        ne: [3, 2048, 128, 1],
        ggml_type: TYPE_F16,
    });
    out.push(EncWeightSpec {
        name: "encoder.conv2.bias".into(),
        ne: [128, 1, 1, 1],
        ggml_type: TYPE_F32,
    });
    out
}

struct EncBlockW {
    ru: [ResUnitW; 3],
    sa: Tensor,
    sb: Tensor,
    dw: Tensor,
    db: Tensor,
    stride: i32,
    padding: i32,
}

struct EncWeights {
    c1w: Tensor,
    c1b: Tensor,
    blocks: [EncBlockW; 5],
    sa: Tensor,
    sb: Tensor,
    c2w: Tensor,
    c2b: Tensor,
}

/// Encode interleaved stereo `[T_audio*2]` → time-major mean latents `[T_latent, 64]`.
pub fn encode(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    interleaved: &[f32],
    t_audio: i64,
) -> Result<Vec<f32>, WeightError> {
    let config = AceVaeConfig::published();
    if t_audio < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE encode needs at least one audio sample".into(),
        ));
    }
    if interleaved.len() as i64 != t_audio * OUT_CHANNELS {
        return Err(WeightError::Shape {
            name: "audio".into(),
            expected: vec![t_audio, OUT_CHANNELS],
            got: vec![interleaved.len() as i64],
        });
    }
    let mut session = AceVaeEncSession::build(ctx, config, t_audio)?;
    load_weights(&mut session, weights, config)?;
    session.run(interleaved, t_audio)
}

/// Tiled encode for long audio (acestep.cpp `vae_enc_encode_tiled`).
pub fn encode_tiled(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    interleaved: &[f32],
    t_audio: i64,
    chunk_latent: i64,
    overlap_latent: i64,
) -> Result<Vec<f32>, WeightError> {
    if t_audio < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE encode needs at least one audio sample".into(),
        ));
    }
    if interleaved.len() as i64 != t_audio * OUT_CHANNELS {
        return Err(WeightError::Shape {
            name: "audio".into(),
            expected: vec![t_audio, OUT_CHANNELS],
            got: vec![interleaved.len() as i64],
        });
    }
    let audio_chunk = chunk_latent * AUDIO_UPSAMPLE;
    let mut audio_overlap = overlap_latent * AUDIO_UPSAMPLE;
    while audio_chunk - 2 * audio_overlap <= 0 && audio_overlap > 0 {
        audio_overlap /= 2;
    }
    if t_audio <= audio_chunk {
        return encode(ctx, weights, interleaved, t_audio);
    }
    let audio_stride = audio_chunk - 2 * audio_overlap;
    if audio_stride <= 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE tiled encode stride is not positive".into(),
        ));
    }
    let config = AceVaeConfig::published();
    let mut out = Vec::new();
    let mut downsample = 0f32;
    let mut core_start = 0i64;
    let mut tile_idx = 0i64;
    while core_start < t_audio {
        let core_end = (core_start + audio_stride).min(t_audio);
        let win_start = (core_start - audio_overlap).max(0);
        let win_end = (core_end + audio_overlap).min(t_audio);
        let win_len = win_end - win_start;
        let slice = &interleaved[(win_start * OUT_CHANNELS) as usize..(win_end * OUT_CHANNELS) as usize];
        let mut session = AceVaeEncSession::build(ctx, config, win_len)?;
        load_weights(&mut session, weights, config)?;
        let tile = session.run_raw_mean(slice, win_len)?;
        let tile_t = tile.len() as i64 / MEAN_CHANNELS;
        if tile_idx == 0 {
            downsample = tile_t as f32 / win_len as f32;
        }
        let added_start = core_start - win_start;
        let trim_start = (added_start as f32 * downsample).round() as i64;
        let added_end = win_end - core_end;
        let trim_end = (added_end as f32 * downsample).round() as i64;
        let end_idx = if trim_end > 0 { tile_t - trim_end } else { tile_t };
        let core_len = end_idx - trim_start;
        if core_len > 0 {
            let start = (trim_start * MEAN_CHANNELS) as usize;
            let end = (end_idx * MEAN_CHANNELS) as usize;
            out.extend_from_slice(&tile[start..end]);
        }
        tile_idx += 1;
        if core_end >= t_audio {
            break;
        }
        core_start += audio_stride;
    }
    Ok(out)
}

pub(crate) struct AceVaeEncSession {
    graph: ReusedGraph,
    t_audio: i64,
}

impl AceVaeEncSession {
    pub(crate) fn build(
        ctx: &GgmlContext,
        config: AceVaeConfig,
        t_audio: i64,
    ) -> Result<Self, WeightError> {
        let mut builder = ctx
            .builder(GRAPH_MEM)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let audio_in = builder.f32_2d("vae_enc.audio", t_audio, OUT_CHANNELS, true);
        let w = declare_weights(&mut builder, config)?;
        let output = build_encode_graph(&builder, &w, audio_in);
        let graph = builder
            .finish(output)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        Ok(Self { graph, t_audio })
    }

    fn run(&mut self, interleaved: &[f32], t_audio: i64) -> Result<Vec<f32>, WeightError> {
        self.run_raw_mean(interleaved, t_audio)
    }

    fn run_raw_mean(&mut self, interleaved: &[f32], t_audio: i64) -> Result<Vec<f32>, WeightError> {
        if t_audio != self.t_audio {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE enc graph was built for T={} but encode asked for T={t_audio}",
                self.t_audio
            )));
        }
        let t = t_audio as usize;
        let mut packed = vec![0f32; t * OUT_CHANNELS as usize];
        for tt in 0..t {
            packed[tt] = interleaved[tt * 2];
            packed[t + tt] = interleaved[tt * 2 + 1];
        }
        self.graph
            .set_f32("vae_enc.audio", &packed)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        self.graph
            .compute_device()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let raw = self
            .graph
            .peek_output()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let [t_lat, ch, ..] = shape(self.graph.output());
        if ch != RAW_OUT_CHANNELS {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE enc output channels {ch} != {RAW_OUT_CHANNELS}"
            )));
        }
        if raw.len() as i64 != t_lat * ch {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE enc readback {} floats != {t_lat}×{ch}",
                raw.len()
            )));
        }
        // Channel-contiguous → time-major mean (ch 0..63).
        let t_usize = t_lat as usize;
        let mut mean = vec![0f32; t_usize * MEAN_CHANNELS as usize];
        for tt in 0..t_usize {
            for cc in 0..MEAN_CHANNELS as usize {
                mean[tt * MEAN_CHANNELS as usize + cc] = raw[cc * t_usize + tt];
            }
        }
        Ok(mean)
    }

    fn set_f32(&mut self, name: &str, data: &[f32]) -> Result<(), WeightError> {
        self.graph
            .set_f32(name, data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))
    }

    fn set_f16(&mut self, name: &str, data: &[f32]) -> Result<(), WeightError> {
        let mut bytes = vec![0u8; data.len() * 2];
        f32_row_to_f16(data, &mut bytes);
        self.graph
            .set_bytes(name, &bytes)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))
    }
}

fn declare_weights(b: &mut GraphBuilder, config: AceVaeConfig) -> Result<EncWeights, WeightError> {
    if config.latent_channels != LATENT_CHANNELS {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE encoder only supports 64-dim latents".into(),
        ));
    }
    let c1w = b.tensor_3d("encoder.conv1.weight", TYPE_F16, 7, 2, 128, true);
    let c1b = b.tensor_1d("encoder.conv1.bias", TYPE_F32, 128, true);
    let mut blocks: [EncBlockW; 5] = [
        enc_block_placeholder(),
        enc_block_placeholder(),
        enc_block_placeholder(),
        enc_block_placeholder(),
        enc_block_placeholder(),
    ];
    for i in 0..5 {
        let ic = ENC_IN_CH[i];
        let oc = ENC_OUT_CH[i];
        let stride = ENC_STRIDES[i] as i32;
        let padding = (stride + 1) / 2;
        let p = format!("encoder.block.{i}");
        let mut ru = [
            res_unit_placeholder(),
            res_unit_placeholder(),
            res_unit_placeholder(),
        ];
        for (r_idx, dil) in RES_DILATIONS.iter().enumerate() {
            let r = r_idx + 1;
            ru[r_idx] = declare_res_unit(b, &format!("{p}.res_unit{r}"), ic, *dil)?;
        }
        let sa = b.tensor_2d(&format!("{p}.snake1.exp_a"), TYPE_F32, 1, ic, true);
        let sb = b.tensor_2d(&format!("{p}.snake1.inv_b"), TYPE_F32, 1, ic, true);
        let dw = b.tensor_3d(
            &format!("{p}.conv1.weight"),
            TYPE_F16,
            ENC_STRIDES[i] * 2,
            ic,
            oc,
            true,
        );
        let db = b.tensor_1d(&format!("{p}.conv1.bias"), TYPE_F32, oc, true);
        blocks[i] = EncBlockW {
            ru,
            sa,
            sb,
            dw,
            db,
            stride,
            padding,
        };
    }
    let sa = b.tensor_2d("encoder.snake1.exp_a", TYPE_F32, 1, 2048, true);
    let sb = b.tensor_2d("encoder.snake1.inv_b", TYPE_F32, 1, 2048, true);
    let c2w = b.tensor_3d("encoder.conv2.weight", TYPE_F16, 3, 2048, 128, true);
    let c2b = b.tensor_1d("encoder.conv2.bias", TYPE_F32, 128, true);
    Ok(EncWeights {
        c1w,
        c1b,
        blocks,
        sa,
        sb,
        c2w,
        c2b,
    })
}

fn enc_block_placeholder() -> EncBlockW {
    EncBlockW {
        ru: [
            res_unit_placeholder(),
            res_unit_placeholder(),
            res_unit_placeholder(),
        ],
        sa: std::ptr::null_mut(),
        sb: std::ptr::null_mut(),
        dw: std::ptr::null_mut(),
        db: std::ptr::null_mut(),
        stride: 0,
        padding: 0,
    }
}

fn build_encode_graph(b: &GraphBuilder, w: &EncWeights, audio: Tensor) -> Tensor {
    let mut x = conv1d_bias(b, w.c1w, Some(w.c1b), audio, 1, 3, 1);
    for blk in &w.blocks {
        for ru in &blk.ru {
            x = res_unit(b, ru, x);
        }
        x = snake(b, x, blk.sa, blk.sb);
        x = conv1d_bias(b, blk.dw, Some(blk.db), x, blk.stride, blk.padding, 1);
    }
    x = snake(b, x, w.sa, w.sb);
    conv1d_bias(b, w.c2w, Some(w.c2b), x, 1, 1, 1)
}

fn load_weights(
    session: &mut AceVaeEncSession,
    weights: &GgufWeights,
    config: AceVaeConfig,
) -> Result<(), WeightError> {
    let _ = config;
    load_conv1d_fused(session, weights, "encoder.conv1")?;
    load_bias(session, weights, "encoder.conv1.bias", 128)?;
    for i in 0..5 {
        let ic = ENC_IN_CH[i];
        let oc = ENC_OUT_CH[i];
        let p = format!("encoder.block.{i}");
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            load_snake_pair(session, weights, &format!("{ru}.snake1"), ic)?;
            load_conv1d_fused(session, weights, &format!("{ru}.conv1"))?;
            load_bias(session, weights, &format!("{ru}.conv1.bias"), ic)?;
            load_snake_pair(session, weights, &format!("{ru}.snake2"), ic)?;
            load_conv1d_fused(session, weights, &format!("{ru}.conv2"))?;
            load_bias(session, weights, &format!("{ru}.conv2.bias"), ic)?;
        }
        load_snake_pair(session, weights, &format!("{p}.snake1"), ic)?;
        load_conv1d_fused(session, weights, &format!("{p}.conv1"))?;
        load_bias(session, weights, &format!("{p}.conv1.bias"), oc)?;
    }
    load_snake_pair(session, weights, "encoder.snake1", 2048)?;
    load_conv1d_fused(session, weights, "encoder.conv2")?;
    load_bias(session, weights, "encoder.conv2.bias", 128)?;
    Ok(())
}

fn load_bias(
    session: &mut AceVaeEncSession,
    weights: &GgufWeights,
    name: &str,
    len: i64,
) -> Result<(), WeightError> {
    let data = weights.f32_widened(name, &[len])?;
    session.set_f32(name, &data)
}

fn load_snake_pair(
    session: &mut AceVaeEncSession,
    weights: &GgufWeights,
    prefix: &str,
    channels: i64,
) -> Result<(), WeightError> {
    let exp_name = format!("{prefix}.exp_a");
    let inv_name = format!("{prefix}.inv_b");
    if weights.contains(&exp_name) && weights.contains(&inv_name) {
        let ne_a = weights.shape(&exp_name)?;
        let ne_b = weights.shape(&inv_name)?;
        let exp_a = weights.f32_widened(&exp_name, &[ne_a[0], ne_a[1], ne_a[2]])?;
        let inv_b = weights.f32_widened(&inv_name, &[ne_b[0], ne_b[1], ne_b[2]])?;
        session.set_f32(&exp_name, &exp_a)?;
        return session.set_f32(&inv_name, &inv_b);
    }
    let alpha_name = format!("{prefix}.alpha");
    let beta_name = format!("{prefix}.beta");
    let alpha = weights.f32_widened(&alpha_name, &[1, channels, 1])?;
    let beta = weights.f32_widened(&beta_name, &[1, channels, 1])?;
    let mut exp_a = Vec::with_capacity(channels as usize);
    let mut inv_b = Vec::with_capacity(channels as usize);
    for i in 0..channels as usize {
        let a = alpha.get(i).copied().unwrap_or(alpha[0]);
        let b = beta.get(i).copied().unwrap_or(beta[0]);
        if !a.is_finite() || !b.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "{prefix} snake alpha/beta is not finite"
            )));
        }
        exp_a.push(a.exp());
        inv_b.push((b.exp()).recip());
    }
    session.set_f32(&format!("{prefix}.exp_a"), &exp_a)?;
    session.set_f32(&format!("{prefix}.inv_b"), &inv_b)
}

fn load_conv1d_fused(
    session: &mut AceVaeEncSession,
    weights: &GgufWeights,
    prefix: &str,
) -> Result<(), WeightError> {
    let fused = fuse_weight_norm(weights, prefix, false)?;
    session.set_f16(&format!("{prefix}.weight"), &fused)
}

/// Zero fused weights + neutral snake for layout-only smoke runs.
pub fn load_smoke_zeros(session: &mut AceVaeEncSession) -> Result<(), WeightError> {
    for spec in graph_weight_layout(AceVaeConfig::smoke()) {
        let n = spec.ne[0] * spec.ne[1] * spec.ne[2] * spec.ne[3].max(1);
        if spec.ggml_type == TYPE_F16 {
            session.set_f16(&spec.name, &vec![0f32; n as usize])?;
        } else {
            let fill = if spec.name.ends_with(".exp_a") || spec.name.ends_with(".inv_b") {
                vec![1f32; n as usize]
            } else {
                vec![0f32; n as usize]
            };
            session.set_f32(&spec.name, &fill)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        encode, graph_weight_layout, load_smoke_zeros, weight_layout, AceVaeEncSession,
    };
    use crate::families::ace_step::vae::{AceVaeConfig, AUDIO_UPSAMPLE, LATENT_CHANNELS};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ace-vae-enc-{name}-{}-{}",
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

    #[test]
    fn published_layout_lists_encoder_blocks() {
        let rows = weight_layout(AceVaeConfig::published());
        assert!(rows.iter().any(|r| r.name == "encoder.conv1.weight_v"));
        assert!(rows
            .iter()
            .any(|r| r.name == "encoder.block.4.res_unit3.conv2.weight_v"));
        assert_eq!(graph_weight_layout(AceVaeConfig::published()).len(), 146);
    }

    #[test]
    fn smoke_encode_zero_audio_mean_latent_length() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let t_audio = AUDIO_UPSAMPLE;
        let mut session = AceVaeEncSession::build(&ctx, AceVaeConfig::smoke(), t_audio).unwrap();
        load_smoke_zeros(&mut session).unwrap();
        let audio = vec![0f32; (t_audio * 2) as usize];
        let latent = session.run(&audio, t_audio).unwrap();
        assert_eq!(latent.len() % LATENT_CHANNELS as usize, 0);
        assert!(latent.iter().all(|v| v.is_finite()));
        // With exact 1920 samples, expect one latent frame.
        assert_eq!(latent.len(), LATENT_CHANNELS as usize);
    }

    #[test]
    fn smoke_gguf_fused_weights_encode() {
        let dir = scratch("fused");
        let config = AceVaeConfig::smoke();
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = graph_weight_layout(config)
            .into_iter()
            .map(|row| {
                let n = (row.ne[0] * row.ne[1] * row.ne[2] * row.ne[3].max(1)) as usize;
                let data = if row.name.ends_with(".exp_a") || row.name.ends_with(".inv_b") {
                    vec![1f32; n]
                } else {
                    vec![0f32; n]
                };
                (row.name, row.ne, data)
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        write_f32_nd(&dir.join("vae-enc-smoke.gguf"), &tensors).unwrap();
        let weights = GgufWeights::open(&dir.join("vae-enc-smoke.gguf")).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let audio = vec![0f32; (AUDIO_UPSAMPLE * 2) as usize];
        let latent = encode(&ctx, &weights, &audio, AUDIO_UPSAMPLE).unwrap();
        assert_eq!(latent.len(), LATENT_CHANNELS as usize);
        assert!(latent.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
