//! ACE-Step condition encoder from acestep.cpp `cond-enc.h`.
//!
//! Packs `lyric (8L) ‖ optional timbre[0] ‖ text_proj` into `[2048, S_total]`.
//! Timbre: Linear(64→2048)+bias → optional CLS → 4L Qwen → norm → take frame[0].
//! Create without a reference uses a single silence latent frame (acestep.cpp
//! `ops_encode_timbre`). Cover with `timbre_from_source` feeds
//! [`sample_timbre_reference_latents`] (750 = 3×250 @ 25 Hz).

use crate::common::ops::non_finite;
use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

pub const COND_HIDDEN: i64 = 2048;
pub const TEXT_HIDDEN: i64 = 1024;
pub const LYRIC_LAYERS: i64 = 8;
pub const TIMBRE_LAYERS: i64 = 4;
pub const TIMBRE_IN: i64 = 64;
pub const TIMBRE_REFERENCE_FRAMES: i64 = 750;
pub const TIMBRE_SEGMENT_FRAMES: i64 = 250;
const LYRIC_WINDOW: i64 = 128;
const TIMBRE_WINDOW: i64 = 128;

/// Project text + lyric (+ optional timbre), pack lyric ‖ timbre[0]? ‖ text_proj.
pub fn encode(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    text_hidden: &[f32],
    s_text: i64,
    lyric_embed: &[f32],
    s_lyric: i64,
    timbre_feats: Option<&[f32]>,
) -> Result<(Vec<f32>, i64), WeightError> {
    if s_text < 1 || text_hidden.len() as i64 != TEXT_HIDDEN * s_text {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step text hidden length {} is not {TEXT_HIDDEN}×{s_text}",
            text_hidden.len()
        )));
    }
    if s_lyric < 1 || lyric_embed.len() as i64 != TEXT_HIDDEN * s_lyric {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step lyric embed length {} is not {TEXT_HIDDEN}×{s_lyric}",
            lyric_embed.len()
        )));
    }
    let (s_ref, timbre_slice) = match timbre_feats {
        None => (0i64, &[][..]),
        Some(feats) => {
            if feats.is_empty() {
                return Err(WeightError::NotBuilt(
                    "ACE-Step timbre feats must be non-empty when provided".into(),
                ));
            }
            if feats.len() as i64 % TIMBRE_IN != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step timbre feats length {} is not a multiple of {TIMBRE_IN}",
                    feats.len()
                )));
            }
            let s_ref = feats.len() as i64 / TIMBRE_IN;
            (s_ref, feats)
        }
    };
    let use_cls = s_ref > 0 && use_timbre_cls(dit);
    if s_ref > 0 && use_cls {
        // Fail loud: XL product path requires the CLS token (MLX parity).
        load_timbre_cls(dit)?;
    }
    check_cond_layout(dit, s_ref > 0)?;
    let mut types = HashMap::new();
    for (name, _, _) in cond_weight_layout(s_ref > 0) {
        types.insert(name.clone(), dit.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut graph = CondForward::build(ctx, s_text, s_lyric, s_ref, use_cls, &types)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    for (name, ne0, ne1) in cond_weight_layout(s_ref > 0) {
        let data = dit.f32_widened(&name, &[ne0, ne1])?;
        graph
            .load_f32(&name, &data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    if use_cls {
        let cls = load_timbre_cls(dit)?;
        graph
            .load_f32("encoder.timbre_encoder.special_token", &cls)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    let packed = graph
        .compute(text_hidden, lyric_embed, if s_ref > 0 { Some(timbre_slice) } else { None })
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    if let Some(detail) = non_finite("ACE-Step cond hidden", &packed) {
        return Err(WeightError::NotBuilt(detail));
    }
    let s_timbre_out = if s_ref > 0 { 1 } else { 0 };
    let s_total = s_lyric + s_timbre_out + s_text;
    if packed.len() as i64 != COND_HIDDEN * s_total {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cond packed length {} is not {COND_HIDDEN}×{s_total}",
            packed.len()
        )));
    }
    Ok((packed, s_total))
}

/// Upstream `process_reference_audio` / MLX `sample_timbre_reference_latents`:
/// 30 s = front/middle/back 10 s segments at 25 Hz (`[T, 64]` time-major).
pub fn sample_timbre_reference_latents(
    latents: &[f32],
    frames: i64,
    channels: i64,
    seed: i64,
) -> Result<Vec<f32>, WeightError> {
    if channels != TIMBRE_IN {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step timbre latents expect {TIMBRE_IN} channels, got {channels}"
        )));
    }
    if frames < 1 || latents.len() as i64 != frames * channels {
        return Err(WeightError::NotBuilt(
            "ACE-Step timbre reference latents must be a non-empty multiple of 64".into(),
        ));
    }
    let mut tiled = latents.to_vec();
    let mut total = frames;
    if total < TIMBRE_REFERENCE_FRAMES {
        let reps = (TIMBRE_REFERENCE_FRAMES + total - 1) / total;
        tiled = Vec::with_capacity((reps * total * channels) as usize);
        for _ in 0..reps {
            tiled.extend_from_slice(latents);
        }
        total = tiled.len() as i64 / channels;
    }
    let mut state = (seed as u64).wrapping_add(0xA5E57E17_u64);
    let seg = total / 3;
    let starts = [
        rand_start(&mut state, seg, TIMBRE_SEGMENT_FRAMES),
        seg + rand_start(&mut state, seg, TIMBRE_SEGMENT_FRAMES),
        2 * seg
            + rand_start(
                &mut state,
                (total - 2 * seg).max(0),
                TIMBRE_SEGMENT_FRAMES,
            ),
    ];
    let mut out = Vec::with_capacity((TIMBRE_REFERENCE_FRAMES * channels) as usize);
    for s in starts {
        let off = (s * channels) as usize;
        let end = ((s + TIMBRE_SEGMENT_FRAMES) * channels) as usize;
        if end > tiled.len() {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step timbre sample window [{s}, {}) exceeds tiled frames {total}",
                s + TIMBRE_SEGMENT_FRAMES
            )));
        }
        out.extend_from_slice(&tiled[off..end]);
    }
    Ok(out)
}

fn rand_start(state: &mut u64, region: i64, window: i64) -> i64 {
    let span = (region - window).max(0) + 1;
    if span <= 1 {
        return 0;
    }
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) % span as u64) as i64
}

pub fn use_timbre_cls(dit: &GgufWeights) -> bool {
    dit.contains("encoder.timbre_encoder.special_token")
}

fn load_timbre_cls(dit: &GgufWeights) -> Result<Vec<f32>, WeightError> {
    for ne in [
        vec![COND_HIDDEN],
        vec![COND_HIDDEN, 1],
        vec![COND_HIDDEN, 1, 1],
    ] {
        if let Ok(values) = dit.f32_widened("encoder.timbre_encoder.special_token", &ne) {
            if values.len() as i64 == COND_HIDDEN {
                return Ok(values);
            }
        }
    }
    Err(WeightError::NotBuilt(
        "encoder.timbre_encoder.special_token does not match condition_dim 2048".into(),
    ))
}

pub fn check_cond_layout(dit: &GgufWeights, with_timbre: bool) -> Result<(), WeightError> {
    for (name, ne0, ne1) in cond_weight_layout(with_timbre) {
        let got = dit.shape(&name)?;
        if got[0] != ne0 || got[1] != ne1 {
            return Err(WeightError::Shape {
                name,
                expected: vec![ne0, ne1],
                got: got.to_vec(),
            });
        }
    }
    Ok(())
}

fn cond_weight_layout(with_timbre: bool) -> Vec<(String, i64, i64)> {
    let h = COND_HIDDEN;
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    // ggml mul_mat: weight is [out, in].
    push(
        "encoder.text_projector.weight".into(),
        COND_HIDDEN,
        TEXT_HIDDEN,
    );
    push(
        "encoder.lyric_encoder.embed_tokens.weight".into(),
        h,
        TEXT_HIDDEN,
    );
    push("encoder.lyric_encoder.embed_tokens.bias".into(), h, 1);
    push("encoder.lyric_encoder.norm.weight".into(), h, 1);
    let heads = 16i64;
    let kv = 8i64;
    let head = 128i64;
    let ff = 6144i64; // acestep.cpp qwen3_lyric_config / qwen3_timbre_config
    let q = heads * head;
    let k = kv * head;
    for layer in 0..LYRIC_LAYERS {
        let p = format!("encoder.lyric_encoder.layers.{layer}");
        push(format!("{p}.input_layernorm.weight"), h, 1);
        push(format!("{p}.self_attn.q_proj.weight"), q, h);
        push(format!("{p}.self_attn.k_proj.weight"), k, h);
        push(format!("{p}.self_attn.v_proj.weight"), k, h);
        push(format!("{p}.self_attn.q_norm.weight"), head, 1);
        push(format!("{p}.self_attn.k_norm.weight"), head, 1);
        push(format!("{p}.self_attn.o_proj.weight"), h, q);
        push(format!("{p}.post_attention_layernorm.weight"), h, 1);
        push(format!("{p}.mlp.gate_proj.weight"), ff, h);
        push(format!("{p}.mlp.up_proj.weight"), ff, h);
        push(format!("{p}.mlp.down_proj.weight"), h, ff);
    }
    if with_timbre {
        push(
            "encoder.timbre_encoder.embed_tokens.weight".into(),
            h,
            TIMBRE_IN,
        );
        push("encoder.timbre_encoder.embed_tokens.bias".into(), h, 1);
        push("encoder.timbre_encoder.norm.weight".into(), h, 1);
        for layer in 0..TIMBRE_LAYERS {
            let p = format!("encoder.timbre_encoder.layers.{layer}");
            push(format!("{p}.input_layernorm.weight"), h, 1);
            push(format!("{p}.self_attn.q_proj.weight"), q, h);
            push(format!("{p}.self_attn.k_proj.weight"), k, h);
            push(format!("{p}.self_attn.v_proj.weight"), k, h);
            push(format!("{p}.self_attn.q_norm.weight"), head, 1);
            push(format!("{p}.self_attn.k_norm.weight"), head, 1);
            push(format!("{p}.self_attn.o_proj.weight"), h, q);
            push(format!("{p}.post_attention_layernorm.weight"), h, 1);
            push(format!("{p}.mlp.gate_proj.weight"), ff, h);
            push(format!("{p}.mlp.up_proj.weight"), ff, h);
            push(format!("{p}.mlp.down_proj.weight"), h, ff);
        }
    }
    out
}

struct CondForward {
    graph: ReusedGraph,
    s_lyric: i64,
    s_timbre: i64,
}

impl CondForward {
    fn build(
        ctx: &GgmlContext,
        s_text: i64,
        s_lyric: i64,
        s_ref: i64,
        use_cls: bool,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let s_timbre = if s_ref > 0 {
            s_ref + if use_cls { 1 } else { 0 }
        } else {
            0
        };
        // Sliding masks dominate for long timbre refs (750² f32 ≈ 2.25M).
        let mem = 128 * 1024 * 1024
            + if s_timbre > 0 {
                (s_timbre * s_timbre * 8).max(32 * 1024 * 1024) as usize
            } else {
                0
            };
        let mut b = ctx.builder(mem)?;
        let with_timbre = s_ref > 0;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in cond_weight_layout(with_timbre) {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        if use_cls {
            // CLS is a graph input (GGUF may store [2048], [2048,1], or [2048,1,1]).
            let cls = b.tensor_2d(
                "encoder.timbre_encoder.special_token",
                TYPE_F32,
                COND_HIDDEN,
                1,
                true,
            );
            if cls.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert("encoder.timbre_encoder.special_token".into(), cls);
        }
        let text_in = nn(b.f32_2d("text_in", TEXT_HIDDEN, s_text, true))?;
        let lyric_in = nn(b.f32_2d("lyric_in", TEXT_HIDDEN, s_lyric, true))?;
        let lyric_pos = nn(b.i32_1d("lyric_pos", s_lyric, true))?;
        let lyric_mask = nn(b.f32_2d("lyric_slide_mask", s_lyric, s_lyric, true))?;
        let text_proj = nn(b.project(
            wt(&weights, "encoder.text_projector.weight")?,
            text_in,
        ))?;
        let mut lyric = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "encoder.lyric_encoder.embed_tokens.weight")?,
                lyric_in,
            ))?,
            wt(&weights, "encoder.lyric_encoder.embed_tokens.bias")?,
        )?;
        for layer in 0..LYRIC_LAYERS {
            let mask = if layer % 2 == 0 {
                lyric_mask
            } else {
                std::ptr::null_mut()
            };
            lyric = qwen_layer(
                &b,
                &weights,
                "encoder.lyric_encoder",
                layer,
                lyric,
                lyric_pos,
                mask,
                s_lyric,
            )?;
        }
        let lyric = rms(
            &b,
            lyric,
            wt(&weights, "encoder.lyric_encoder.norm.weight")?,
            1e-6,
        )?;

        let packed = if with_timbre {
            let timbre_in = nn(b.f32_2d("timbre_in", TIMBRE_IN, s_ref, true))?;
            let timbre_pos = nn(b.i32_1d("timbre_pos", s_timbre, true))?;
            let timbre_mask = nn(b.f32_2d("timbre_slide_mask", s_timbre, s_timbre, true))?;
            let mut timbre = add_bias(
                &b,
                nn(b.project(
                    wt(&weights, "encoder.timbre_encoder.embed_tokens.weight")?,
                    timbre_in,
                ))?,
                wt(&weights, "encoder.timbre_encoder.embed_tokens.bias")?,
            )?;
            if use_cls {
                let cls = nn(b.reshape_2d(
                    wt(&weights, "encoder.timbre_encoder.special_token")?,
                    COND_HIDDEN,
                    1,
                ))?;
                timbre = nn(b.cont(b.concat(cls, timbre, 1)))?;
            }
            for layer in 0..TIMBRE_LAYERS {
                let mask = if layer % 2 == 0 {
                    timbre_mask
                } else {
                    std::ptr::null_mut()
                };
                timbre = qwen_layer(
                    &b,
                    &weights,
                    "encoder.timbre_encoder",
                    layer,
                    timbre,
                    timbre_pos,
                    mask,
                    s_timbre,
                )?;
            }
            let timbre = rms(
                &b,
                timbre,
                wt(&weights, "encoder.timbre_encoder.norm.weight")?,
                1e-6,
            )?;
            // Take first position (CLS when present, else first audio frame).
            let stride = (COND_HIDDEN as usize) * std::mem::size_of::<f32>();
            let timbre0 = nn(b.cont(b.view_2d(timbre, COND_HIDDEN, 1, stride, 0)))?;
            let mid = nn(b.cont(b.concat(lyric, timbre0, 1)))?;
            nn(b.cont(b.concat(mid, text_proj, 1)))?
        } else {
            nn(b.cont(b.concat(lyric, text_proj, 1)))?
        };
        let graph = b.finish(packed)?;
        let _ = s_text;
        Ok(Self {
            graph,
            s_lyric,
            s_timbre,
        })
    }

    fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    fn compute(
        &mut self,
        text: &[f32],
        lyric: &[f32],
        timbre: Option<&[f32]>,
    ) -> Result<Vec<f32>, RuntimeError> {
        self.graph.set_f32("text_in", text)?;
        self.graph.set_f32("lyric_in", lyric)?;
        let positions: Vec<i32> = (0..self.s_lyric as i32).collect();
        self.graph.set_i32("lyric_pos", &positions)?;
        self.graph
            .set_f32("lyric_slide_mask", &slide_mask(self.s_lyric, LYRIC_WINDOW))?;
        if let Some(feats) = timbre {
            self.graph.set_f32("timbre_in", feats)?;
            let tpos: Vec<i32> = (0..self.s_timbre as i32).collect();
            self.graph.set_i32("timbre_pos", &tpos)?;
            self.graph.set_f32(
                "timbre_slide_mask",
                &slide_mask(self.s_timbre, TIMBRE_WINDOW),
            )?;
        }
        self.graph.compute()
    }
}

fn slide_mask(seq: i64, window: i64) -> Vec<f32> {
    let mut mask = vec![0f32; (seq * seq) as usize];
    for i in 0..seq {
        for j in 0..seq {
            let d = (i - j).abs();
            if d > window {
                mask[(j + i * seq) as usize] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

fn qwen_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask_f32: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("{prefix}.layers.{layer}");
    let heads = 16i64;
    let kv = 8i64;
    let head = 128i64;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.input_layernorm.weight"))?,
        1e-6,
    )?;
    let q = nn(b.project(wt(weights, &format!("{p}.self_attn.q_proj.weight"))?, normed))?;
    let k = nn(b.project(wt(weights, &format!("{p}.self_attn.k_proj.weight"))?, normed))?;
    let v = nn(b.project(wt(weights, &format!("{p}.self_attn.v_proj.weight"))?, normed))?;
    let q = nn(b.reshape_3d(q, head, heads, seq))?;
    let k = nn(b.reshape_3d(k, head, kv, seq))?;
    let v = nn(b.reshape_3d(v, head, kv, seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, 1e-6))?,
        wt(weights, &format!("{p}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, 1e-6))?,
        wt(weights, &format!("{p}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, 1_000_000.0)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, 1_000_000.0)))?;
    let attn = flash_attend(b, q, k, v, mask_f32)?;
    let attn = nn(b.reshape_2d(attn, heads * head, seq))?;
    let attn = nn(b.project(wt(weights, &format!("{p}.self_attn.o_proj.weight"))?, attn))?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
        1e-6,
    )?;
    let gate = nn(b.silu(nn(b.project(
        wt(weights, &format!("{p}.mlp.gate_proj.weight"))?,
        normed,
    ))?))?;
    let up = nn(b.project(wt(weights, &format!("{p}.mlp.up_proj.weight"))?, normed))?;
    let ff = nn(b.mul(gate, up))?;
    let ff = nn(b.project(wt(weights, &format!("{p}.mlp.down_proj.weight"))?, ff))?;
    nn(b.add(hidden, ff))
}

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor, eps: f32) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, eps))?, weight))
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|t| !t.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("cond graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        sample_timbre_reference_latents, TIMBRE_IN, TIMBRE_REFERENCE_FRAMES, TIMBRE_SEGMENT_FRAMES,
    };

    #[test]
    fn sample_timbre_pads_short_and_emits_750() {
        let frames = 100i64;
        let lat: Vec<f32> = (0..(frames * TIMBRE_IN) as usize)
            .map(|i| i as f32)
            .collect();
        let out = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 7).unwrap();
        assert_eq!(out.len() as i64, TIMBRE_REFERENCE_FRAMES * TIMBRE_IN);
        // Three windows of 250 contiguous (post-tile) frames.
        assert_eq!(out.len() / TIMBRE_IN as usize, (3 * TIMBRE_SEGMENT_FRAMES) as usize);
    }

    #[test]
    fn sample_timbre_rejects_empty() {
        assert!(sample_timbre_reference_latents(&[], 0, TIMBRE_IN, 0).is_err());
        assert!(sample_timbre_reference_latents(&[0.0; 64], 1, 32, 0).is_err());
    }

    #[test]
    fn sample_timbre_is_seed_deterministic() {
        let frames = 900i64;
        let lat = vec![1.0f32; (frames * TIMBRE_IN) as usize];
        let a = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 42).unwrap();
        let b = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 42).unwrap();
        let c = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 43).unwrap();
        assert_eq!(a, b);
        // Different seeds should usually differ; force a detectable change by
        // encoding starts into values (flat latents stay equal — use indexed).
        let lat: Vec<f32> = (0..(frames * TIMBRE_IN) as usize)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let a = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 1).unwrap();
        let b = sample_timbre_reference_latents(&lat, frames, TIMBRE_IN, 99).unwrap();
        assert_ne!(a, b);
    }
}
