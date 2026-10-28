//! Qwen2.5-VL text trunk used by Qwen-Image create.
//!
//! The file names are the Hugging Face names (`model.embed_tokens.weight`,
//! `model.layers.*.self_attn.q_proj`). Main remaps those onto `encoder.*`.
//! This graph reads the file names. Vision tensors are not in the layout:
//! create drops `visual` the way main's `_nested_without_encoder_visual` does.
//!
//! Create positions are one index on all three MRoPE axes, so the section
//! picker matches half-split RoPE. `rope_neox` with theta `1_000_000` is that
//! case. A vision grid, where the axes differ, is not this path. ggml's MRoPE
//! keeps one axis on both halves of a pair; the diffusers chunk of the doubled
//! table can put different axes on the two halves. Those two agree only when
//! the three positions are equal, which is the case `hf_mrope_cos` locks.

use crate::runtime::{flash_attend, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use std::collections::HashMap;

pub const DROP_IDX: usize = 34;
pub const MAX_LENGTH: usize = 1058;
const RMS_EPS: f32 = 1e-6;

/// Hardcoded Qwen-Image create template from main `QwenImageTextEncoder`.
pub const PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug, Clone, Copy)]
pub struct Qwen25VlConfig {
    pub vocab: i64,
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub intermediate: i64,
    pub rope_theta: f32,
    pub mrope: [usize; 3],
}

impl Qwen25VlConfig {
    /// Qwen2.5-VL 7B counts from main `QwenEncoderLayer` defaults.
    pub fn published() -> Self {
        Self {
            vocab: 152064,
            hidden: 3584,
            layers: 28,
            heads: 28,
            kv_heads: 4,
            head_dim: 128,
            intermediate: 18944,
            rope_theta: 1_000_000.0,
            mrope: [16, 24, 24],
        }
    }

    /// One layer, GQA, head 2. The published MRoPE sections do not fit this head.
    pub fn smoke() -> Self {
        Self {
            vocab: 16,
            hidden: 8,
            layers: 1,
            heads: 4,
            kv_heads: 2,
            head_dim: 2,
            intermediate: 8,
            rope_theta: 1_000_000.0,
            mrope: [1, 0, 0],
        }
    }
}

/// Diffusers `_apply_multimodal_rotary_pos_emb` cosine row for one token.
/// `sections` are the published pair counts. The doubled chunks index the
/// concatenated frequency table.
pub fn hf_mrope_cos(
    axis_pos: [i32; 3],
    head_dim: usize,
    sections: [usize; 3],
    theta: f32,
) -> Vec<f32> {
    let half = head_dim / 2;
    let doubled = [sections[0] * 2, sections[1] * 2, sections[2] * 2];
    let mut inv = vec![0f32; half];
    for i in 0..half {
        inv[i] = (f64::from(theta).powf(-(i as f64) / (half as f64))) as f32;
    }
    let mut rows = Vec::with_capacity(3);
    for axis in 0..3 {
        let mut emb = Vec::with_capacity(head_dim);
        for i in 0..half {
            let freq = (axis_pos[axis] as f32) * inv[i];
            emb.push(freq.cos());
        }
        let freq_cos = emb.clone();
        emb.extend_from_slice(&freq_cos);
        rows.push(emb);
    }
    let mut out = Vec::with_capacity(head_dim);
    let mut start = 0usize;
    for (chunk, size) in doubled.iter().copied().enumerate() {
        let axis = chunk % 3;
        out.extend_from_slice(&rows[axis][start..start + size]);
        start += size;
    }
    out
}

/// ggml MRoPE assigns one axis to both halves of pair `i`. Section sizes are pair counts.
pub fn ggml_mrope_axis(pair: usize, sections: [usize; 3]) -> usize {
    let sum = sections[0] + sections[1] + sections[2];
    let sector = pair % sum.max(1);
    if sector < sections[0] {
        0
    } else if sector < sections[0] + sections[1] {
        1
    } else {
        2
    }
}

/// Drop the system-prefix tokens from a `[hidden, seq]` row-major-in-hidden buffer.
/// `valid` is the unpadded length. The prefix is the published `drop_idx` of 34.
pub fn trim_system_prefix(
    hidden: &[f32],
    width: usize,
    valid: usize,
) -> Result<Vec<f32>, RuntimeError> {
    if width == 0 || hidden.len() % width != 0 {
        return Err(RuntimeError::Message(
            "text hidden width does not divide the buffer".into(),
        ));
    }
    let seq = hidden.len() / width;
    if valid > seq {
        return Err(RuntimeError::Message(format!(
            "text valid length {valid} is past the sequence {seq}"
        )));
    }
    if valid <= DROP_IDX {
        return Err(RuntimeError::Message(format!(
            "prompt is shorter than the {DROP_IDX}-token system prefix"
        )));
    }
    let mut out = Vec::with_capacity(width * (valid - DROP_IDX));
    for token in DROP_IDX..valid {
        let start = token * width;
        out.extend_from_slice(&hidden[start..start + width]);
    }
    Ok(out)
}

pub fn text_layout(config: Qwen25VlConfig) -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    out.push((
        "model.embed_tokens.weight".into(),
        config.hidden,
        config.vocab,
    ));
    out.push(("model.norm.weight".into(), config.hidden, 1));
    let q_out = config.heads * config.head_dim;
    let kv_out = config.kv_heads * config.head_dim;
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        out.push((format!("{p}.input_layernorm.weight"), config.hidden, 1));
        out.push((
            format!("{p}.post_attention_layernorm.weight"),
            config.hidden,
            1,
        ));
        out.push((format!("{p}.self_attn.q_proj.weight"), config.hidden, q_out));
        out.push((format!("{p}.self_attn.q_proj.bias"), q_out, 1));
        out.push((
            format!("{p}.self_attn.k_proj.weight"),
            config.hidden,
            kv_out,
        ));
        out.push((format!("{p}.self_attn.k_proj.bias"), kv_out, 1));
        out.push((
            format!("{p}.self_attn.v_proj.weight"),
            config.hidden,
            kv_out,
        ));
        out.push((format!("{p}.self_attn.v_proj.bias"), kv_out, 1));
        out.push((format!("{p}.self_attn.o_proj.weight"), q_out, config.hidden));
        out.push((
            format!("{p}.mlp.gate_proj.weight"),
            config.hidden,
            config.intermediate,
        ));
        out.push((
            format!("{p}.mlp.up_proj.weight"),
            config.hidden,
            config.intermediate,
        ));
        out.push((
            format!("{p}.mlp.down_proj.weight"),
            config.intermediate,
            config.hidden,
        ));
    }
    out
}

pub struct TextForward {
    graph: crate::runtime::ReusedGraph,
    seq: i64,
    hidden: i64,
}

impl TextForward {
    pub fn build(
        ctx: &crate::runtime::GgmlContext,
        config: Qwen25VlConfig,
        seq: i64,
    ) -> Result<Self, RuntimeError> {
        if seq < 1 {
            return Err(RuntimeError::Message("text sequence is empty".into()));
        }
        if config.heads < 1
            || config.kv_heads < 1
            || config.head_dim < 2
            || config.head_dim % 2 != 0
            || config.heads * config.head_dim != config.hidden
            || config.heads % config.kv_heads != 0
        {
            return Err(RuntimeError::Message(
                "qwen2.5-vl head geometry does not match the hidden width".into(),
            ));
        }
        let mut builder = ctx.builder(32 * 1024 * 1024)?;
        let output = build_tower(&mut builder, config, seq)?;
        let graph = builder.finish(output)?;
        Ok(Self {
            graph,
            seq,
            hidden: config.hidden,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    /// Causal mask, positions `0..seq`. Padding is a `-inf` the caller writes into `mask`.
    pub fn compute(&mut self, ids: &[i32], mask: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        if ids.len() as i64 != self.seq || mask.len() as i64 != self.seq * self.seq {
            return Err(RuntimeError::Message(format!(
                "text ids {} mask {}, graph sequence is {}",
                ids.len(),
                mask.len(),
                self.seq
            )));
        }
        let positions: Vec<i32> = (0..self.seq).map(|i| i as i32).collect();
        self.graph.set_i32("token_ids", ids)?;
        self.graph.set_i32("positions", &positions)?;
        self.graph.set_f32("attn_mask", mask)?;
        let out = self.graph.compute()?;
        if out.len() != (self.hidden * self.seq) as usize {
            return Err(RuntimeError::Message(format!(
                "text tower returned {} values, expected {} by {}",
                out.len(),
                self.hidden,
                self.seq
            )));
        }
        Ok(out)
    }
}

fn build_tower(
    b: &mut GraphBuilder,
    config: Qwen25VlConfig,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in text_layout(config) {
        let tensor = b.tensor_2d(&name, TYPE_F32, ne0, ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    let token_ids = nn(b.i32_1d("token_ids", seq, true))?;
    let pos = nn(b.i32_1d("positions", seq, true))?;
    let mask = nn(b.f32_2d("attn_mask", seq, seq, true))?;
    let mut hidden = nn(b.get_rows(wt(&weights, "model.embed_tokens.weight")?, token_ids))?;
    for layer in 0..config.layers {
        hidden = block(b, &weights, config, layer, hidden, pos, mask, seq)?;
    }
    rms(b, hidden, wt(&weights, "model.norm.weight")?)
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: Qwen25VlConfig,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("model.layers.{layer}");
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.input_layernorm.weight"))?,
    )?;
    let attn = attention(b, weights, config, &p, normed, pos, mask, seq)?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
    )?;
    let gate = nn(b.silu(linear(
        b,
        normed,
        wt(weights, &format!("{p}.mlp.gate_proj.weight"))?,
    )))?;
    let up = linear(b, normed, wt(weights, &format!("{p}.mlp.up_proj.weight"))?);
    let mid = nn(b.mul(gate, up))?;
    let mlp = linear(b, mid, wt(weights, &format!("{p}.mlp.down_proj.weight"))?);
    nn(b.add(hidden, mlp))
}

fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: Qwen25VlConfig,
    prefix: &str,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let head = config.head_dim;
    let q = biased_heads(
        b,
        hidden,
        wt(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
        wt(weights, &format!("{prefix}.self_attn.q_proj.bias"))?,
        head,
        config.heads,
        seq,
    )?;
    let k = biased_heads(
        b,
        hidden,
        wt(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
        wt(weights, &format!("{prefix}.self_attn.k_proj.bias"))?,
        head,
        config.kv_heads,
        seq,
    )?;
    let v = biased_heads(
        b,
        hidden,
        wt(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
        wt(weights, &format!("{prefix}.self_attn.v_proj.bias"))?,
        head,
        config.kv_heads,
        seq,
    )?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, config.rope_theta)))?;
    let mixed = flash_attend(b, q, k, v, mask)?;
    let mixed = nn(b.reshape_2d(mixed, config.heads * head, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
    ))
}

fn biased_heads(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let projected = linear(b, x, weight);
    let y = nn(b.add(projected, nn(b.repeat(bias, projected))?))?;
    nn(b.cont(b.reshape_3d(y, head, heads, seq)))
}

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.mul_mat(w, x)
}

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
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

#[cfg(test)]
mod tests {
    use super::{
        ggml_mrope_axis, hf_mrope_cos, text_layout, trim_system_prefix, Qwen25VlConfig,
        TextForward, DROP_IDX, MAX_LENGTH, PROMPT_TEMPLATE,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use std::collections::HashMap;

    fn tap(name: &str, index: usize) -> f32 {
        let mut hash = 2166136261u32;
        for byte in name.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
        }
        hash = hash.wrapping_mul(16777619) ^ index as u32;
        ((hash % 7) as f32 - 3.0) * 0.05
    }

    fn at(data: &[f32], width: usize, row: usize, col: usize) -> f32 {
        data[row + width * col]
    }

    #[test]
    fn published_template_drops_thirty_four_and_skips_vision() {
        let config = Qwen25VlConfig::published();
        assert_eq!(config.layers, 28);
        assert_eq!(config.kv_heads, 4);
        assert_eq!(config.intermediate, 18944);
        assert_eq!(config.mrope, [16, 24, 24]);
        assert_eq!(
            (config.mrope[0] + config.mrope[1] + config.mrope[2]) * 2,
            config.head_dim as usize
        );
        assert_eq!(DROP_IDX, 34);
        assert_eq!(MAX_LENGTH, 1058);
        assert!(PROMPT_TEMPLATE.contains(
            "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
        ));
        assert!(PROMPT_TEMPLATE.contains("<|im_start|>user\n{}<|im_end|>"));
        let names: Vec<_> = text_layout(config)
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(names
            .iter()
            .any(|name| name == "model.layers.0.self_attn.q_proj.bias"));
        assert!(names.iter().any(|name| name == "model.norm.weight"));
        assert!(names.iter().all(|name| !name.contains("q_norm")
            && !name.contains("visual")
            && !name.starts_with("encoder.")));
        let hidden: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let trimmed = trim_system_prefix(&hidden, 1, 36).unwrap();
        assert_eq!(trimmed, vec![34.0, 35.0]);
        let err = trim_system_prefix(&hidden, 1, 34).unwrap_err();
        assert!(err.to_string().contains("34-token system prefix"), "{err}");
    }

    #[test]
    fn equal_axes_match_half_split_and_unequal_axes_split_the_halves() {
        let sections = [16usize, 24, 24];
        let head = 128;
        let row = hf_mrope_cos([3, 3, 3], head, sections, 1_000_000.0);
        let half = head / 2;
        for i in 0..half {
            let freq = 3.0 * (1_000_000f64.powf(-(i as f64) / (half as f64))) as f32;
            assert!((row[i] - freq.cos()).abs() < 1e-5, "{}", row[i]);
            assert!(
                (row[i + half] - freq.cos()).abs() < 1e-5,
                "{}",
                row[i + half]
            );
        }
        let split = hf_mrope_cos([0, 1, 2], head, sections, 10_000.0);
        assert!(
            (split[0] - split[half]).abs() > 0.1,
            "halves {} {}",
            split[0],
            split[half]
        );
        assert_eq!(ggml_mrope_axis(0, sections), 0);
        assert_eq!(ggml_mrope_axis(16, sections), 1);
        assert_eq!(ggml_mrope_axis(40, sections), 2);
    }

    #[test]
    fn smoke_tower_matches_the_host_block() {
        let config = Qwen25VlConfig::smoke();
        let seq = 4i64;
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut forward = TextForward::build(&ctx, config, seq).unwrap();
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in text_layout(config) {
            let data: Vec<f32> = (0..(ne0 * ne1) as usize).map(|i| tap(&name, i)).collect();
            forward.load_f32(&name, &data).unwrap();
            weights.insert(name, data);
        }
        let ids = [1i32, 3, 0, 2];
        let mut mask = vec![0f32; (seq * seq) as usize];
        for q in 0..seq {
            for k in 0..seq {
                if k > q {
                    mask[(k + q * seq) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let got = forward.compute(&ids, &mask).unwrap();
        let expect = host_tower(&weights, config, &ids, &mask);
        assert_eq!(got.len(), expect.len());
        let mut worst = 0f32;
        for (got, want) in got.iter().zip(&expect) {
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-3, "text tower max abs {worst}");
    }

    fn host_tower(
        weights: &HashMap<String, Vec<f32>>,
        config: Qwen25VlConfig,
        ids: &[i32],
        mask: &[f32],
    ) -> Vec<f32> {
        let seq = ids.len();
        let hidden = config.hidden as usize;
        let embed = &weights["model.embed_tokens.weight"];
        let mut x = vec![0f32; hidden * seq];
        for (token, id) in ids.iter().enumerate() {
            for h in 0..hidden {
                x[h + hidden * token] = embed[h + hidden * (*id as usize)];
            }
        }
        for layer in 0..config.layers {
            let p = format!("model.layers.{layer}");
            x = host_block(weights, config, &p, &x, seq, mask);
        }
        host_rms(&x, hidden, seq, &weights["model.norm.weight"])
    }

    fn host_block(
        weights: &HashMap<String, Vec<f32>>,
        config: Qwen25VlConfig,
        prefix: &str,
        x: &[f32],
        seq: usize,
        mask: &[f32],
    ) -> Vec<f32> {
        let hidden = config.hidden as usize;
        let normed = host_rms(
            x,
            hidden,
            seq,
            &weights[&format!("{prefix}.input_layernorm.weight")],
        );
        let attn = host_attn(weights, config, prefix, &normed, seq, mask);
        let mut added = vec![0f32; x.len()];
        for i in 0..x.len() {
            added[i] = x[i] + attn[i];
        }
        let normed = host_rms(
            &added,
            hidden,
            seq,
            &weights[&format!("{prefix}.post_attention_layernorm.weight")],
        );
        let gate = host_linear(
            &normed,
            hidden,
            &weights[&format!("{prefix}.mlp.gate_proj.weight")],
            config.intermediate as usize,
            seq,
            None,
        );
        let up = host_linear(
            &normed,
            hidden,
            &weights[&format!("{prefix}.mlp.up_proj.weight")],
            config.intermediate as usize,
            seq,
            None,
        );
        let mut mid = vec![0f32; gate.len()];
        for i in 0..gate.len() {
            mid[i] = silu(gate[i]) * up[i];
        }
        let mlp = host_linear(
            &mid,
            config.intermediate as usize,
            &weights[&format!("{prefix}.mlp.down_proj.weight")],
            hidden,
            seq,
            None,
        );
        for i in 0..added.len() {
            added[i] += mlp[i];
        }
        added
    }

    fn host_attn(
        weights: &HashMap<String, Vec<f32>>,
        config: Qwen25VlConfig,
        prefix: &str,
        x: &[f32],
        seq: usize,
        mask: &[f32],
    ) -> Vec<f32> {
        let hidden = config.hidden as usize;
        let head = config.head_dim as usize;
        let heads = config.heads as usize;
        let kv = config.kv_heads as usize;
        let q = host_linear(
            x,
            hidden,
            &weights[&format!("{prefix}.self_attn.q_proj.weight")],
            heads * head,
            seq,
            Some(&weights[&format!("{prefix}.self_attn.q_proj.bias")]),
        );
        let k = host_linear(
            x,
            hidden,
            &weights[&format!("{prefix}.self_attn.k_proj.weight")],
            kv * head,
            seq,
            Some(&weights[&format!("{prefix}.self_attn.k_proj.bias")]),
        );
        let v = host_linear(
            x,
            hidden,
            &weights[&format!("{prefix}.self_attn.v_proj.weight")],
            kv * head,
            seq,
            Some(&weights[&format!("{prefix}.self_attn.v_proj.bias")]),
        );
        let q = host_rope(&q, head, heads, seq, config.rope_theta);
        let k = host_rope(&k, head, kv, seq, config.rope_theta);
        let scale = 1.0 / (head as f32).sqrt();
        let repeat = heads / kv;
        let mut out = vec![0f32; heads * head * seq];
        for h in 0..heads {
            let kv_h = h / repeat;
            for q_tok in 0..seq {
                let mut scores = vec![0f32; seq];
                let mut max_score = f32::NEG_INFINITY;
                for k_tok in 0..seq {
                    let allowed = mask[k_tok + q_tok * seq].is_finite();
                    if !allowed {
                        scores[k_tok] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0f32;
                    for d in 0..head {
                        dot +=
                            at(&q, head, d, h + heads * q_tok) * at(&k, head, d, kv_h + kv * k_tok);
                    }
                    scores[k_tok] = dot * scale;
                    max_score = max_score.max(scores[k_tok]);
                }
                let mut denom = 0f32;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - max_score).exp();
                        denom += *score;
                    } else {
                        *score = 0.0;
                    }
                }
                for k_tok in 0..seq {
                    let weight = scores[k_tok] / denom;
                    for d in 0..head {
                        let dest = d + head * (h + heads * q_tok);
                        out[dest] += weight * at(&v, head, d, kv_h + kv * k_tok);
                    }
                }
            }
        }
        host_linear(
            &out,
            heads * head,
            &weights[&format!("{prefix}.self_attn.o_proj.weight")],
            hidden,
            seq,
            None,
        )
    }

    fn host_rope(x: &[f32], head: usize, heads: usize, seq: usize, theta: f32) -> Vec<f32> {
        let mut out = x.to_vec();
        let theta_scale = theta.powf(-2.0 / head as f32);
        for token in 0..seq {
            let mut angle = token as f32;
            let mut cos = vec![0f32; head / 2];
            let mut sin = vec![0f32; head / 2];
            for i in 0..head / 2 {
                cos[i] = angle.cos();
                sin[i] = angle.sin();
                angle *= theta_scale;
            }
            for h in 0..heads {
                for i in 0..head / 2 {
                    let i0 = i + head * (h + heads * token);
                    let i1 = i0 + head / 2;
                    let x0 = out[i0];
                    let x1 = out[i1];
                    out[i0] = x0 * cos[i] - x1 * sin[i];
                    out[i1] = x0 * sin[i] + x1 * cos[i];
                }
            }
        }
        out
    }

    fn host_linear(
        x: &[f32],
        input: usize,
        weight: &[f32],
        output: usize,
        seq: usize,
        bias: Option<&Vec<f32>>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; output * seq];
        for token in 0..seq {
            for o in 0..output {
                let mut acc = bias.map(|row| row[o]).unwrap_or(0.0);
                for i in 0..input {
                    acc += weight[i + input * o] * x[i + input * token];
                }
                y[o + output * token] = acc;
            }
        }
        y
    }

    fn host_rms(x: &[f32], width: usize, seq: usize, weight: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for token in 0..seq {
            let mut mean = 0f32;
            for h in 0..width {
                let value = x[h + width * token];
                mean += value * value;
            }
            mean /= width as f32;
            let scale = 1.0 / (mean + 1.0e-6).sqrt();
            for h in 0..width {
                y[h + width * token] = x[h + width * token] * scale * weight[h];
            }
        }
        y
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}
