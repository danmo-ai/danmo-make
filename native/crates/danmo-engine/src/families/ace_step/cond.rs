//! ACE-Step condition encoder from acestep.cpp `cond-enc.h`.
//!
//! Packs `lyric (8L) + optional timbre[0] + text_proj` into `[2048, S_total]`.
//! Timbre is optional (text2music create uses silence and skips this path).

use crate::common::ops::non_finite;
use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

pub const COND_HIDDEN: i64 = 2048;
pub const TEXT_HIDDEN: i64 = 1024;
pub const LYRIC_LAYERS: i64 = 8;
const LYRIC_WINDOW: i64 = 128;

/// Project text + run the lyric tower, then pack lyric ‖ text_proj.
pub fn encode(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    text_hidden: &[f32],
    s_text: i64,
    lyric_embed: &[f32],
    s_lyric: i64,
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
    check_cond_layout(dit)?;
    let mut types = HashMap::new();
    for (name, _, _) in cond_weight_layout() {
        types.insert(name.clone(), dit.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut graph = CondForward::build(ctx, s_text, s_lyric, &types)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    for (name, ne0, ne1) in cond_weight_layout() {
        let data = dit.f32_widened(&name, &[ne0, ne1])?;
        graph
            .load_f32(&name, &data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    let packed = graph
        .compute(text_hidden, lyric_embed)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    if let Some(detail) = non_finite("ACE-Step cond hidden", &packed) {
        return Err(WeightError::NotBuilt(detail));
    }
    let s_total = s_lyric + s_text;
    if packed.len() as i64 != COND_HIDDEN * s_total {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cond packed length {} is not {COND_HIDDEN}×{s_total}",
            packed.len()
        )));
    }
    Ok((packed, s_total))
}

pub fn check_cond_layout(dit: &GgufWeights) -> Result<(), WeightError> {
    for (name, ne0, ne1) in cond_weight_layout() {
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

fn cond_weight_layout() -> Vec<(String, i64, i64)> {
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
    let ff = 6144i64; // acestep.cpp qwen3_lyric_config
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
    out
}

struct CondForward {
    graph: ReusedGraph,
    s_text: i64,
    s_lyric: i64,
}

impl CondForward {
    fn build(
        ctx: &GgmlContext,
        s_text: i64,
        s_lyric: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(128 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in cond_weight_layout() {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
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
            lyric = lyric_layer(&b, &weights, layer, lyric, lyric_pos, mask, s_lyric)?;
        }
        let lyric = rms(
            &b,
            lyric,
            wt(&weights, "encoder.lyric_encoder.norm.weight")?,
            1e-6,
        )?;
        let packed = nn(b.cont(b.concat(lyric, text_proj, 1)))?;
        let graph = b.finish(packed)?;
        Ok(Self {
            graph,
            s_text,
            s_lyric,
        })
    }

    fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    fn compute(&mut self, text: &[f32], lyric: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        self.graph.set_f32("text_in", text)?;
        self.graph.set_f32("lyric_in", lyric)?;
        let positions: Vec<i32> = (0..self.s_lyric as i32).collect();
        self.graph.set_i32("lyric_pos", &positions)?;
        let mut mask = vec![0f32; (self.s_lyric * self.s_lyric) as usize];
        for i in 0..self.s_lyric {
            for j in 0..self.s_lyric {
                let d = (i - j).abs();
                if d > LYRIC_WINDOW {
                    mask[(j + i * self.s_lyric) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        self.graph.set_f32("lyric_slide_mask", &mask)?;
        self.graph.compute()
    }
}

fn lyric_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask_f32: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("encoder.lyric_encoder.layers.{layer}");
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
