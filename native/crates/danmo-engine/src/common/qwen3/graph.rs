//! Qwen3 decoder stack. The caller chooses which 1-based hidden states to keep.

use crate::runtime::{flash_attend, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use std::collections::HashMap;

const QWEN_RMS_EPS: f32 = 1e-6;

pub(crate) struct QwenTower {
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub rope_theta: f32,
}

/// Append the tower and return the concatenated selected states.
pub(crate) fn build_qwen3(
    b: &mut GraphBuilder,
    tower: QwenTower,
    prefix: &str,
    selected: &[i64],
    seq: i64,
    layout: &[(String, i64, i64)],
    types: &HashMap<String, u32>,
) -> Result<Tensor, RuntimeError> {
    if seq < 1 {
        return Err(RuntimeError::Message("text sequence is empty".into()));
    }
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in layout {
        let ggml_type = types.get(name).copied().unwrap_or(TYPE_F32);
        let tensor = b.tensor_2d(name, ggml_type, *ne0, *ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name.clone(), tensor);
    }
    let token_ids = nn(b.i32_1d("token_ids", seq, true))?;
    let pos = nn(b.i32_1d("positions", seq, true))?;
    let mask = nn(b.f32_2d("attn_mask", seq, seq, true))?;
    let mut hidden = nn(b.get_rows(
        wt(&weights, &format!("{prefix}embed_tokens.weight"))?,
        token_ids,
    ))?;
    let last = *selected.iter().max().unwrap_or(&0);
    let mut captured = Vec::new();
    for layer in 0..last {
        hidden = qwen_layer(b, &weights, &tower, prefix, layer, hidden, pos, mask, seq)?;
        let index = layer + 1;
        if selected.contains(&index) {
            captured.push(hidden);
        }
    }
    let mut out = *captured
        .first()
        .ok_or_else(|| RuntimeError::Message("text encoder captured no hidden states".into()))?;
    for extra in captured.iter().skip(1) {
        out = nn(b.concat(out, *extra, 0))?;
    }
    Ok(out)
}

fn qwen_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    tower: &QwenTower,
    prefix: &str,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("{prefix}layers.{layer}");
    let normed = qwen_rms(
        b,
        hidden,
        wt(weights, &format!("{p}.input_layernorm.weight"))?,
    )?;
    let attn = qwen_attention(b, weights, tower, &p, normed, pos, mask, seq)?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = qwen_rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
    )?;
    let mid = nn(b.mul(
        nn(b.silu(linear(
            b,
            normed,
            wt(weights, &format!("{p}.mlp.gate_proj.weight"))?,
        )))?,
        nn(linear(
            b,
            normed,
            wt(weights, &format!("{p}.mlp.up_proj.weight"))?,
        ))?,
    ))?;
    let mlp = nn(linear(
        b,
        mid,
        wt(weights, &format!("{p}.mlp.down_proj.weight"))?,
    ))?;
    nn(b.add(hidden, mlp))
}

fn qwen_attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    tower: &QwenTower,
    prefix: &str,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let head = tower.head_dim;
    let q = qwen_rms_heads(
        b,
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
        head,
        tower.heads,
        seq,
    )?;
    let k = qwen_rms_heads(
        b,
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
        head,
        tower.kv_heads,
        seq,
    )?;
    let v = nn(b.cont(b.reshape_3d(
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
        ))?,
        head,
        tower.kv_heads,
        seq,
    )))?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, tower.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, tower.rope_theta)))?;
    let mixed = flash_attend(b, q, k, v, mask)?;
    // Qwen3 query width is heads * head_dim. It is wider than the residual.
    let mixed = nn(b.reshape_2d(mixed, tower.heads * tower.head_dim, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
    ))
}

fn qwen_rms(b: &GraphBuilder, x: Tensor, weight: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, QWEN_RMS_EPS))?, weight))
}

fn qwen_rms_heads(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let x = nn(b.cont(b.reshape_3d(x, head, heads, seq)))?;
    nn(b.mul(nn(b.rms_norm(x, QWEN_RMS_EPS))?, weight))
}

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.mul_mat(w, x)
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}
