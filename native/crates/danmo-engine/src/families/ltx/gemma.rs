//! LTX 2.3 Gemma-3-12B text encoder: SentencePiece-BPE tokenizer and the
//! decoder tower that emits the per-layer state stack the LTX connectors
//! consume.
//!
//! Semantics are a port of main's `text_encoder_mlx.py` with the tower math
//! cross-checked against the reference `llm.hpp` GEMMA3_12B arch:
//! - tokenizer: HF `tokenizer.json` BPE with `byte_fallback`; the normalizer
//!   replaces every space with `▁` and the literal-space split then keeps the
//!   whole string one piece, so plain text is normalized and BPE-merged
//!   directly; added tokens split out of the text; `<bos>` (2) is prepended,
//!   no EOS; over-length prompts keep the LAST `MAX_LEN` tokens (which can
//!   drop the BOS, same as main); the sequence is left-padded with `<pad>` 0
//!   to `MAX_LEN` (1024).
//! - tower (llama.cpp GGUF names): `token_embd * sqrt(dim)`; 48 blocks of
//!   rms(`attn_norm`) -> attention (per-head RMS `attn_q/k_norm`, NEOX RoPE on
//!   the full 256 head dim, local theta 1e4 / global theta 1e6 with the linear
//!   context scale 1/8 on every 6th layer, causal + left-pad mask) ->
//!   `attn_output` -> rms(`post_attention_norm`) -> residual; rms(`ffn_norm`)
//!   -> `gelu_tanh(gate) * up` -> `down` -> rms(`post_ffw_norm`) -> residual.
//!   main passes one combined mask to every layer, which equals the published
//!   1024 sliding window at the 1024-token cap.
//! - states: the embedding output plus every block output, each per-token RMS
//!   normalized (no weight), concatenated to `[dim * (depth + 1), seq]` and
//!   multiplied by the valid-token mask. The connectors project this stack.
//!
//! Fully padded query rows are finite by construction: the additive mask uses
//! `-3e4` (F16-safe) instead of main's `-1e9`, which is identical after
//! softmax for valid rows and keeps the discarded pad rows finite.

use crate::runtime::{
    flash_attend, GgmlContext, ReusedGraph, RuntimeError, Tensor, TYPE_BF16, TYPE_F16, TYPE_F32,
};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// main caps the prompt at 1024 tokens (`LTX2_GEMMA_MAX_LENGTH` default).
pub const MAX_LEN: usize = 1024;
const RMS_EPS: f32 = 1e-6;
const PAD_ID: i32 = 0;
const BOS_ID: i32 = 2;
/// Five sliding layers then one global. At the 1024-token cap the sliding
/// window equals full causal, so the mask is shared; only the RoPE theta and
/// the linear context scale differ.
const GLOBAL_PERIOD: i64 = 6;
const LOCAL_ROPE_THETA: f32 = 10_000.0;
const GLOBAL_ROPE_THETA: f32 = 1_000_000.0;
const GLOBAL_FREQ_SCALE: f32 = 1.0 / 8.0;
/// F16-safe additive mask sentinel (see the module docs).
const MASK_NEG: f32 = -3e4;

// ---------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------

pub struct GemmaTokenizer {
    vocab: HashMap<String, i32>,
    ranks: HashMap<(String, String), usize>,
    specials: Vec<(String, i32)>,
}

impl GemmaTokenizer {
    pub fn from_json(json: &str) -> Result<Self, WeightError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
        let model = value
            .get("model")
            .ok_or_else(|| WeightError::NotBuilt("tokenizer has no model".into()))?;
        if model.get("type").and_then(Value::as_str) != Some("BPE") {
            return Err(WeightError::NotBuilt(
                "gemma tokenizer model is not BPE".into(),
            ));
        }
        if !model
            .get("byte_fallback")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(WeightError::NotBuilt(
                "gemma tokenizer without byte_fallback is not supported".into(),
            ));
        }
        let mut vocab = HashMap::new();
        let entries = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| WeightError::NotBuilt("tokenizer has no vocab".into()))?;
        for (token, id) in entries {
            let Some(id) = id.as_i64() else {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer vocab id for {token} is not an integer"
                )));
            };
            vocab.insert(token.clone(), id as i32);
        }
        let merges = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::NotBuilt("tokenizer has no merges".into()))?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let Some(text) = merge.as_str() else {
                return Err(WeightError::NotBuilt(
                    "tokenizer merge is not a string".into(),
                ));
            };
            let Some((left, right)) = text.split_once(' ') else {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer merge {text:?} has no separator"
                )));
            };
            ranks.insert((left.to_string(), right.to_string()), rank);
        }
        let mut specials = Vec::new();
        if let Some(added) = value.get("added_tokens").and_then(Value::as_array) {
            for token in added {
                let content = token.get("content").and_then(Value::as_str);
                let id = token.get("id").and_then(Value::as_i64);
                let (Some(content), Some(id)) = (content, id) else {
                    return Err(WeightError::NotBuilt(
                        "tokenizer added token is missing content or id".into(),
                    ));
                };
                specials.push((content.to_string(), id as i32));
            }
        }
        specials.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));
        Ok(Self {
            vocab,
            ranks,
            specials,
        })
    }

    fn bpe(&self, plain: &str) -> Result<Vec<i32>, WeightError> {
        let normalized: String = plain
            .chars()
            .map(|c| if c == ' ' { '▁' } else { c })
            .collect();
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        while symbols.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_index = None;
            for index in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .ranks
                    .get(&(symbols[index].clone(), symbols[index + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(index);
                    }
                }
            }
            let Some(index) = best_index else {
                break;
            };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols.splice(index..index + 2, std::iter::once(merged));
        }
        let mut ids = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            if let Some(id) = self.vocab.get(&symbol) {
                ids.push(*id);
                continue;
            }
            // byte_fallback: emit one <0xXX> token per UTF-8 byte.
            for byte in symbol.bytes() {
                let token = format!("<0x{byte:02X}>");
                let Some(id) = self.vocab.get(&token) else {
                    return Err(WeightError::NotBuilt(format!(
                        "gemma tokenizer has no byte fallback {token} for {symbol:?}"
                    )));
                };
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    /// Tokenize into the padded `(ids, mask)` pair main feeds the tower:
    /// `<bos>` plus the token ids, truncated to the last `MAX_LEN`, left
    /// padded with `<pad>` to `MAX_LEN`.
    pub fn encode(&self, raw: &str) -> Result<(Vec<i32>, Vec<f32>), WeightError> {
        let text = raw.trim();
        let mut ids = vec![BOS_ID];
        let mut index = 0;
        while index < text.len() {
            let found = self
                .specials
                .iter()
                .find(|(token, _)| !token.is_empty() && text[index..].starts_with(token.as_str()));
            if let Some((token, id)) = found {
                ids.push(*id);
                index += token.len();
                continue;
            }
            let step = text[index..].chars().next().map_or(1, char::len_utf8);
            let mut end = index + step;
            while end < text.len() {
                if self
                    .specials
                    .iter()
                    .any(|(token, _)| !token.is_empty() && text[end..].starts_with(token.as_str()))
                {
                    break;
                }
                end += text[end..].chars().next().map_or(1, char::len_utf8);
            }
            ids.extend(self.bpe(&text[index..end])?);
            index = end;
        }
        if ids.len() > MAX_LEN {
            ids = ids[ids.len() - MAX_LEN..].to_vec();
        }
        let pad = MAX_LEN - ids.len();
        let mut mask = vec![0f32; pad];
        mask.extend(std::iter::repeat(1f32).take(ids.len()));
        let mut padded = vec![PAD_ID; pad];
        padded.extend(ids);
        Ok((padded, mask))
    }
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct GemmaConfig {
    pub dim: i64,
    pub depth: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub ffn: i64,
    pub vocab: i64,
}

impl GemmaConfig {
    pub fn smoke() -> Self {
        // Depth 6 exercises one global layer (index 5).
        Self {
            dim: 8,
            depth: 6,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            ffn: 16,
            vocab: 64,
        }
    }

    /// The number of stacked states the connectors consume: the embedding
    /// output plus every block output.
    pub fn states(&self) -> i64 {
        self.depth + 1
    }

    pub fn infer(weights: &GgufWeights) -> Result<Self, WeightError> {
        let embed = weights.shape("token_embd.weight")?;
        let dim = embed[0];
        let vocab = embed[1];
        let mut depth = 0i64;
        while weights.contains(&format!("blk.{depth}.attn_q.weight")) {
            depth += 1;
        }
        if depth < 1 {
            return Err(WeightError::MissingTensor {
                path: "text_encoder".into(),
                name: "blk.0.attn_q.weight".into(),
            });
        }
        if !weights.contains("output_norm.weight") {
            return Err(WeightError::MissingTensor {
                path: "text_encoder".into(),
                name: "output_norm.weight".into(),
            });
        }
        let q_norm = weights.shape("blk.0.attn_q_norm.weight")?;
        let head_dim = q_norm[0];
        if head_dim < 1 {
            return Err(WeightError::NotBuilt("gemma head dim is zero".into()));
        }
        let q = weights.shape("blk.0.attn_q.weight")?;
        let k = weights.shape("blk.0.attn_k.weight")?;
        let o = weights.shape("blk.0.attn_output.weight")?;
        if q[0] != dim || k[0] != dim || q[1] % head_dim != 0 || k[1] % head_dim != 0 {
            return Err(WeightError::NotBuilt(format!(
                "gemma attention shape mismatch: q {q:?} k {k:?} dim {dim} head_dim {head_dim}"
            )));
        }
        let heads = q[1] / head_dim;
        let kv_heads = k[1] / head_dim;
        if heads < 1 || kv_heads < 1 || heads % kv_heads != 0 {
            return Err(WeightError::NotBuilt(format!(
                "gemma heads {heads} are not a multiple of kv heads {kv_heads}"
            )));
        }
        if o[0] != heads * head_dim || o[1] != dim {
            return Err(WeightError::NotBuilt(format!(
                "gemma attn_output shape {o:?} does not fold heads {heads} x head_dim {head_dim} into dim {dim}"
            )));
        }
        let gate = weights.shape("blk.0.ffn_gate.weight")?;
        let down = weights.shape("blk.0.ffn_down.weight")?;
        if gate[0] != dim || down[0] != gate[1] || down[1] != dim {
            return Err(WeightError::NotBuilt(format!(
                "gemma ffn shape mismatch: gate {gate:?} down {down:?} dim {dim}"
            )));
        }
        Ok(Self {
            dim,
            depth,
            heads,
            kv_heads,
            head_dim,
            ffn: gate[1],
            vocab,
        })
    }
}

/// `(name, ne0, ne1)` for every tower weight, in load order.
pub(crate) fn layout(config: &GemmaConfig) -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    out.push(("token_embd.weight".into(), config.dim, config.vocab));
    let q_width = config.heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    for i in 0..config.depth {
        let p = format!("blk.{i}");
        out.push((format!("{p}.attn_norm.weight"), config.dim, 1));
        out.push((format!("{p}.attn_q.weight"), config.dim, q_width));
        out.push((format!("{p}.attn_k.weight"), config.dim, kv_width));
        out.push((format!("{p}.attn_v.weight"), config.dim, kv_width));
        out.push((format!("{p}.attn_output.weight"), q_width, config.dim));
        out.push((format!("{p}.attn_q_norm.weight"), config.head_dim, 1));
        out.push((format!("{p}.attn_k_norm.weight"), config.head_dim, 1));
        out.push((format!("{p}.post_attention_norm.weight"), config.dim, 1));
        out.push((format!("{p}.ffn_norm.weight"), config.dim, 1));
        out.push((format!("{p}.ffn_gate.weight"), config.dim, config.ffn));
        out.push((format!("{p}.ffn_up.weight"), config.dim, config.ffn));
        out.push((format!("{p}.ffn_down.weight"), config.ffn, config.dim));
        out.push((format!("{p}.post_ffw_norm.weight"), config.dim, 1));
    }
    out.push(("output_norm.weight".into(), config.dim, 1));
    out
}

/// Norms are stored F32 and widened on load; everything else is a matmul.
fn is_f32_weight(name: &str) -> bool {
    name.ends_with("norm.weight")
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

struct GemmaGraph {
    graph: ReusedGraph,
    seq: i64,
}

impl GemmaGraph {
    fn build(
        ctx: &GgmlContext,
        config: &GemmaConfig,
        seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(64 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in layout(config) {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let token_ids = nn(b.i32_1d("token_ids", seq, true))?;
        let pos = nn(b.i32_1d("positions", seq, true))?;
        let mask = nn(b.f32_2d("attn_mask", seq, seq, true))?;
        let pad_mask = nn(b.f32_2d("pad_mask", 1, seq, true))?;
        let dim = config.dim;
        let head = config.head_dim;
        let mut hidden = nn(b.get_rows(wt(&weights, "token_embd.weight")?, token_ids))?;
        if types.get("token_embd.weight").copied() != Some(TYPE_F32) {
            hidden = nn(b.cast_f32(hidden))?;
        }
        hidden = nn(b.scale(hidden, (dim as f32).sqrt()))?;
        let mut states = vec![hidden];
        for layer in 0..config.depth {
            hidden = gemma_layer(&b, &weights, config, layer, hidden, pos, mask, seq)?;
            states.push(hidden);
        }
        let mut stacked = nn(b.rms_norm(
            *states
                .first()
                .ok_or_else(|| RuntimeError::Message("gemma captured no states".into()))?,
            RMS_EPS,
        ))?;
        for state in states.iter().skip(1) {
            let normed = nn(b.rms_norm(*state, RMS_EPS))?;
            stacked = nn(b.concat(stacked, normed, 0))?;
        }
        let out = nn(b.mul(stacked, pad_mask))?;
        let graph = b.finish(out)?;
        Ok(Self { graph, seq })
    }

    fn compute(
        &mut self,
        ids: &[i32],
        pos: &[i32],
        mask: &[f32],
        pad_mask: &[f32],
    ) -> Result<Vec<f32>, RuntimeError> {
        if ids.len() as i64 != self.seq {
            return Err(RuntimeError::Message(format!(
                "token ids has {} values, graph expects {}",
                ids.len(),
                self.seq
            )));
        }
        self.graph.set_i32("token_ids", ids)?;
        self.graph.set_i32("positions", pos)?;
        self.graph.set_f32("attn_mask", mask)?;
        self.graph.set_f32("pad_mask", pad_mask)?;
        self.graph.compute()
    }
}

#[allow(clippy::too_many_arguments)]
fn gemma_layer(
    b: &crate::runtime::GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &GemmaConfig,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("blk.{layer}");
    let dim = config.dim;
    let head = config.head_dim;
    let heads = config.heads;
    let kv_heads = config.kv_heads;
    let (theta, freq_scale) = if (layer + 1) % GLOBAL_PERIOD == 0 {
        (GLOBAL_ROPE_THETA, GLOBAL_FREQ_SCALE)
    } else {
        (LOCAL_ROPE_THETA, 1.0)
    };
    let normed = nn(b.mul(
        nn(b.rms_norm(hidden, RMS_EPS))?,
        wt(weights, &format!("{p}.attn_norm.weight"))?,
    ))?;
    let q = nn(b.mul_mat(wt(weights, &format!("{p}.attn_q.weight"))?, normed))?;
    let q = nn(b.cont(nn(b.reshape_3d(q, head, heads, seq))?))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, RMS_EPS))?,
        wt(weights, &format!("{p}.attn_q_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox_fs(q, pos, head as i32, theta, freq_scale)))?;
    let k = nn(b.mul_mat(wt(weights, &format!("{p}.attn_k.weight"))?, normed))?;
    let k = nn(b.cont(nn(b.reshape_3d(k, head, kv_heads, seq))?))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, RMS_EPS))?,
        wt(weights, &format!("{p}.attn_k_norm.weight"))?,
    ))?;
    let k = nn(b.cont(b.rope_neox_fs(k, pos, head as i32, theta, freq_scale)))?;
    let v = nn(b.mul_mat(wt(weights, &format!("{p}.attn_v.weight"))?, normed))?;
    let v = nn(b.cont(nn(b.reshape_3d(v, head, kv_heads, seq))?))?;
    let mixed = flash_attend(b, q, k, v, mask)?;
    let mixed = nn(b.reshape_2d(mixed, heads * head, seq))?;
    let attn = nn(b.mul_mat(wt(weights, &format!("{p}.attn_output.weight"))?, mixed))?;
    let attn = nn(b.mul(
        nn(b.rms_norm(attn, RMS_EPS))?,
        wt(weights, &format!("{p}.post_attention_norm.weight"))?,
    ))?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = nn(b.mul(
        nn(b.rms_norm(hidden, RMS_EPS))?,
        wt(weights, &format!("{p}.ffn_norm.weight"))?,
    ))?;
    let gate = super::gelu_tanh(
        b,
        nn(b.mul_mat(wt(weights, &format!("{p}.ffn_gate.weight"))?, normed))?,
    )?;
    let up = nn(b.mul_mat(wt(weights, &format!("{p}.ffn_up.weight"))?, normed))?;
    let ffn = nn(b.mul_mat(
        wt(weights, &format!("{p}.ffn_down.weight"))?,
        nn(b.mul(gate, up))?,
    ))?;
    let ffn = nn(b.mul(
        nn(b.rms_norm(ffn, RMS_EPS))?,
        wt(weights, &format!("{p}.post_ffw_norm.weight"))?,
    ))?;
    let _ = dim;
    nn(b.add(hidden, ffn))
}

/// Causal + left-pad additive mask: `mask[k + q * seq]` is 0 for keys a query
/// may attend and `MASK_NEG` otherwise. Pad-region queries are fully masked
/// and produce the finite garbage the pad mask discards after the stack.
pub(crate) fn attention_mask(valid: &[f32]) -> Vec<f32> {
    let seq = valid.len();
    let mut mask = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if k > q || valid[k] == 0.0 {
                mask[k + q * seq] = MASK_NEG;
            }
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

/// The masked state stack one prompt produces: `[dim * states, seq]` flat in
/// ggml order plus the valid-token mask the connectors reuse.
pub struct GemmaConditioning {
    pub stacked: Vec<f32>,
    pub mask: Vec<f32>,
    pub seq: i64,
    pub states: i64,
    pub dim: i64,
}

pub struct GemmaEncoder {
    weights: std::sync::Arc<GgufWeights>,
    config: GemmaConfig,
}

impl GemmaEncoder {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = GemmaConfig::infer(&weights)?;
        Ok(Self { weights, config })
    }

    pub fn config(&self) -> GemmaConfig {
        self.config
    }

    /// Tower forward on already-tokenized input: `[dim * states, seq]`.
    pub fn encode_ids(
        &self,
        ctx: &GgmlContext,
        ids: &[i32],
        valid_mask: &[f32],
    ) -> Result<Vec<f32>, WeightError> {
        let seq = ids.len() as i64;
        if seq < 1 || valid_mask.len() as i64 != seq {
            return Err(WeightError::NotBuilt(format!(
                "gemma ids {} / mask {} length mismatch",
                ids.len(),
                valid_mask.len()
            )));
        }
        for id in ids {
            if *id < 0 || *id as i64 >= self.config.vocab {
                return Err(WeightError::NotBuilt(format!(
                    "token id {id} is outside the gemma vocab {}",
                    self.config.vocab
                )));
            }
        }
        let layout = layout(&self.config);
        let mut types = HashMap::new();
        for (name, ne0, _) in &layout {
            let ggml_type = if is_f32_weight(name) {
                TYPE_F32
            } else {
                let ggml_type = self.weights.ggml_type_of(name)?;
                if name == "token_embd.weight" {
                    if ggml_type != TYPE_F32 && ggml_type != TYPE_F16 && ggml_type != TYPE_BF16 {
                        return Err(WeightError::NotBuilt(format!(
                            "token_embd.weight is ggml type {ggml_type}. The embedding table must be F32, F16, or BF16"
                        )));
                    }
                } else {
                    if !matmul_type(ggml_type) {
                        return Err(WeightError::Type {
                            name: name.clone(),
                            ggml_type,
                        });
                    }
                    let block = type_block(ggml_type);
                    if block > 1 && ne0 % block != 0 {
                        return Err(WeightError::NotBuilt(format!(
                            "{name} length {ne0} is not divisible by ggml block {block}"
                        )));
                    }
                }
                ggml_type
            };
            types.insert(name.clone(), ggml_type);
        }
        let mut graph = GemmaGraph::build(ctx, &self.config, seq, &types)?;
        for (name, ne0, ne1) in &layout {
            if is_f32_weight(name) {
                let values = self.weights.f32_widened(name, &[*ne0, *ne1])?;
                graph.graph.set_f32(name, &values)?;
            } else {
                let raw = self.weights.raw_exact(name, &[*ne0, *ne1])?;
                graph.graph.set_bytes(name, &raw)?;
            }
        }
        let pos: Vec<i32> = (0..seq as i32).collect();
        let mask = attention_mask(valid_mask);
        let out = graph.compute(ids, &pos, &mask, valid_mask)?;
        if let Some(detail) = crate::common::ops::non_finite("gemma states", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }

    /// Tokenize + tower. The output is the masked stack at `MAX_LEN`.
    pub fn encode(
        &self,
        ctx: &GgmlContext,
        tokenizer_json: &str,
        prompt: &str,
    ) -> Result<GemmaConditioning, WeightError> {
        let tokenizer = GemmaTokenizer::from_json(tokenizer_json)?;
        let (ids, mask) = tokenizer.encode(prompt)?;
        let stacked = self.encode_ids(ctx, &ids, &mask)?;
        Ok(GemmaConditioning {
            stacked,
            mask,
            seq: MAX_LEN as i64,
            states: self.config.states(),
            dim: self.config.dim,
        })
    }
}

// ---------------------------------------------------------------------------
// smoke writer (graph-vs-host tests and the future session smoke bundle)
// ---------------------------------------------------------------------------

/// Deterministic smoke weights: `dim * dim * 13 * depth + vocab` values.
pub fn smoke_store(config: &GemmaConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 7u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let mut fill = |count: usize| (0..count).map(|_| next()).collect::<Vec<f32>>();
    let mut store = Vec::new();
    store.push((
        "token_embd.weight".to_string(),
        fill((config.dim * config.vocab) as usize),
    ));
    let q_width = (config.heads * config.head_dim) as usize;
    let kv_width = (config.kv_heads * config.head_dim) as usize;
    let dim = config.dim as usize;
    let ffn = config.ffn as usize;
    for i in 0..config.depth {
        let p = format!("blk.{i}");
        store.push((format!("{p}.attn_norm.weight"), fill(dim)));
        store.push((format!("{p}.attn_q.weight"), fill(dim * q_width)));
        store.push((format!("{p}.attn_k.weight"), fill(dim * kv_width)));
        store.push((format!("{p}.attn_v.weight"), fill(dim * kv_width)));
        store.push((format!("{p}.attn_output.weight"), fill(q_width * dim)));
        store.push((
            format!("{p}.attn_q_norm.weight"),
            fill(config.head_dim as usize),
        ));
        store.push((
            format!("{p}.attn_k_norm.weight"),
            fill(config.head_dim as usize),
        ));
        store.push((format!("{p}.post_attention_norm.weight"), fill(dim)));
        store.push((format!("{p}.ffn_norm.weight"), fill(dim)));
        store.push((format!("{p}.ffn_gate.weight"), fill(dim * ffn)));
        store.push((format!("{p}.ffn_up.weight"), fill(dim * ffn)));
        store.push((format!("{p}.ffn_down.weight"), fill(ffn * dim)));
        store.push((format!("{p}.post_ffw_norm.weight"), fill(dim)));
    }
    store.push(("output_norm.weight".to_string(), fill(dim)));
    store
}

pub fn write_smoke(dir: &Path) -> Result<PathBuf, WeightError> {
    write_smoke_with(dir, &GemmaConfig::smoke())
}

/// Write the smoke Gemma GGUF for an explicit config (session smoke bundle
/// needs 48 blocks so the stacked state count matches the connectors).
pub fn write_smoke_with(dir: &Path, config: &GemmaConfig) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = *config;
    let store = smoke_store(&config);
    let dims: HashMap<String, (i64, i64)> = layout(&config)
        .into_iter()
        .map(|(name, ne0, ne1)| (name, (ne0, ne1)))
        .collect();
    let bytes: Vec<Vec<u8>> = store
        .iter()
        .map(|(_, values)| bytemuck_f32(values))
        .collect();
    let tensors: Vec<crate::weights::TypedTensor<'_>> = store
        .iter()
        .zip(&bytes)
        .map(|((name, _), data)| {
            let (ne0, ne1) = dims[name];
            crate::weights::TypedTensor {
                name,
                ne0,
                ne1,
                ggml_type: TYPE_F32,
                data,
            }
        })
        .collect();
    let path = dir.join("gemma_smoke.gguf");
    crate::weights::write_typed_tensors(&path, &tensors)?;
    Ok(path)
}

fn bytemuck_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The toy tokenizer the unit tests encode with: SentencePiece semantics at
/// hand-computable scale, including byte fallback and an added token.
pub fn smoke_tokenizer_json() -> String {
    let mut vocab = HashMap::new();
    let entries = [
        ("<pad>", 0),
        ("<eos>", 1),
        ("<bos>", 2),
        ("<unk>", 3),
        ("▁", 4),
        ("a", 5),
        ("b", 6),
        ("c", 7),
        ("t", 8),
        ("▁a", 9),
        ("▁b", 10),
        ("ca", 11),
        ("cat", 12),
        ("▁cat", 13),
        ("▁a▁cat", 14),
        ("<0xC3>", 15),
        ("<0xA9>", 16),
        ("<0xE2>", 17),
        ("<0x98>", 18),
        ("<0x80>", 19),
        ("ab", 20),
        ("▁ab", 21),
    ];
    for (token, id) in entries {
        vocab.insert(token, id);
    }
    let merges = [
        "▁ a",
        "c a",
        "ca t",
        "▁ cat",
        "▁a ▁cat",
        "a b",
        "▁a b",
        "▁ b",
    ];
    let body = serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": true},
            {"id": 1, "content": "<eos>", "special": true},
            {"id": 2, "content": "<bos>", "special": true},
            {"id": 3, "content": "<unk>", "special": true},
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "model": {
            "type": "BPE",
            "byte_fallback": true,
            "vocab": vocab,
            "merges": merges,
        },
    });
    body.to_string()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;

    fn smoke_tokenizer() -> GemmaTokenizer {
        GemmaTokenizer::from_json(&smoke_tokenizer_json()).unwrap()
    }

    #[test]
    fn tokenizer_merges_like_sentencepiece() {
        let tok = smoke_tokenizer();
        // The normalizer only replaces spaces: "a cat" -> "a▁cat" (no
        // leading ▁ on the first word), merging to ["a", "▁cat"].
        let (ids, mask) = tok.encode("a cat").unwrap();
        assert_eq!(&ids[MAX_LEN - 3..], &[2, 5, 13]);
        assert_eq!(&mask[MAX_LEN - 3..], &[1.0, 1.0, 1.0]);
        assert!(mask[..MAX_LEN - 3].iter().all(|v| *v == 0.0));
        assert!(ids[..MAX_LEN - 3].iter().all(|id| *id == PAD_ID));
        let (ids, _) = tok.encode("b").unwrap();
        assert_eq!(&ids[MAX_LEN - 2..], &[2, 6]); // <bos> b
                                                  // "ab": the "a b" merge (rank 5) applies directly.
        let (ids, _) = tok.encode("ab").unwrap();
        assert_eq!(&ids[MAX_LEN - 2..], &[2, 20]);
    }

    #[test]
    fn tokenizer_falls_back_to_bytes_for_unknown_chars() {
        let tok = smoke_tokenizer();
        // 'é' = 0xC3 0xA9 — not in vocab, so one fallback token per byte.
        let (ids, _) = tok.encode("é").unwrap();
        assert_eq!(&ids[MAX_LEN - 3..], &[2, 15, 16]);
    }

    #[test]
    fn tokenizer_splits_added_tokens_out() {
        let tok = smoke_tokenizer();
        let (ids, _) = tok.encode("a<eos>cat").unwrap();
        // "a" (no leading space), <eos>, then "cat" merged through "c a"/"ca t".
        assert_eq!(&ids[MAX_LEN - 4..], &[2, 5, 1, 12]);
    }

    #[test]
    fn tokenizer_truncates_to_the_last_max_len_tokens() {
        let tok = smoke_tokenizer();
        let long = "cat ".repeat(MAX_LEN + 10);
        let (ids, mask) = tok.encode(&long).unwrap();
        assert_eq!(ids.len(), MAX_LEN);
        assert!(mask.iter().all(|v| *v == 1.0));
        // The BOS is truncated away, matching main's `tokens[-max_length:]`.
        assert_ne!(ids[0], BOS_ID);
    }

    #[test]
    fn tokenizer_rejects_non_bpe_models() {
        let err = match GemmaTokenizer::from_json(
            r#"{"model": {"type": "Unigram", "vocab": [], "byte_fallback": true}}"#,
        ) {
            Err(err) => err,
            Ok(_) => panic!("accepted a unigram tokenizer"),
        };
        assert!(err.to_string().contains("not BPE"), "{err}");
    }

    // ----- graph vs host reference -----

    fn at<'a>(store: &'a [(String, Vec<f32>)], name: &str) -> &'a [f32] {
        store
            .iter()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
            .as_slice()
    }

    fn gelu_tanh_host(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    fn rms_host(x: &[f32], w: Option<&[f32]>, dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            let mut sum = 0f32;
            for d in 0..dim {
                sum += x[d + dim * s] * x[d + dim * s];
            }
            let scale = 1.0 / (sum / dim as f32 + RMS_EPS).sqrt();
            for d in 0..dim {
                out[d + dim * s] = x[d + dim * s] * scale * w.map_or(1.0, |w| w[d]);
            }
        }
        out
    }

    /// `w` is ggml `[in, out]`; `x` is `[in, seq]`. Returns `[out, seq]`.
    fn linear_host(w: &[f32], x: &[f32], inp: usize, out: usize, seq: usize) -> Vec<f32> {
        let mut y = vec![0f32; out * seq];
        for s in 0..seq {
            for o in 0..out {
                let mut acc = 0f32;
                for i in 0..inp {
                    acc += w[i + inp * o] * x[i + inp * s];
                }
                y[o + out * s] = acc;
            }
        }
        y
    }

    /// NEOX half-split rotation on `[head_dim, heads, seq]` rows: the head
    /// dim is fastest, so the position of column `h + heads * s` is `s`.
    fn rope_host(
        x: &[f32],
        head_dim: usize,
        heads: usize,
        seq: usize,
        theta: f32,
        freq_scale: f32,
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = head_dim / 2;
        for col in 0..x.len() / head_dim {
            let s = col / heads;
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = s as f32 * freq_scale * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x1 = x[i + head_dim * col];
                let x2 = x[i + half + head_dim * col];
                out[i + head_dim * col] = x1 * cos - x2 * sin;
                out[i + half + head_dim * col] = x1 * sin + x2 * cos;
            }
        }
        out
    }

    fn host_forward(
        store: &[(String, Vec<f32>)],
        config: &GemmaConfig,
        ids: &[i32],
        valid_mask: &[f32],
    ) -> Vec<f32> {
        let dim = config.dim as usize;
        let heads = config.heads as usize;
        let kv_heads = config.kv_heads as usize;
        let hd = config.head_dim as usize;
        let ffn = config.ffn as usize;
        let seq = ids.len();
        let embd = at(store, "token_embd.weight");
        let scale = (dim as f32).sqrt();
        let mut hidden = vec![0f32; dim * seq];
        for (s, id) in ids.iter().enumerate() {
            for d in 0..dim {
                hidden[d + dim * s] = embd[d + dim * (*id as usize)] * scale;
            }
        }
        let mut states = vec![hidden.clone()];
        let attn_mask = attention_mask(valid_mask);
        for layer in 0..config.depth {
            let p = format!("blk.{layer}");
            let (theta, freq_scale) = if (layer + 1) % GLOBAL_PERIOD == 0 {
                (GLOBAL_ROPE_THETA, GLOBAL_FREQ_SCALE)
            } else {
                (LOCAL_ROPE_THETA, 1.0)
            };
            let normed = rms_host(
                &hidden,
                Some(at(store, &format!("{p}.attn_norm.weight"))),
                dim,
                seq,
            );
            let q = linear_host(
                at(store, &format!("{p}.attn_q.weight")),
                &normed,
                dim,
                heads * hd,
                seq,
            );
            let k = linear_host(
                at(store, &format!("{p}.attn_k.weight")),
                &normed,
                dim,
                kv_heads * hd,
                seq,
            );
            let v = linear_host(
                at(store, &format!("{p}.attn_v.weight")),
                &normed,
                dim,
                kv_heads * hd,
                seq,
            );
            let q = rms_host(
                &q,
                Some(at(store, &format!("{p}.attn_q_norm.weight"))),
                hd,
                heads * seq,
            );
            let k = rms_host(
                &k,
                Some(at(store, &format!("{p}.attn_k_norm.weight"))),
                hd,
                kv_heads * seq,
            );
            let q = rope_host(&q, hd, heads, seq, theta, freq_scale);
            let k = rope_host(&k, hd, kv_heads, seq, theta, freq_scale);
            let scale = 1.0 / (hd as f32).sqrt();
            // GQA: kv head index = q head / (heads / kv_heads).
            let group = heads / kv_heads;
            let mut mixed = vec![0f32; heads * hd * seq];
            for h in 0..heads {
                let kvh = h / group;
                for qp in 0..seq {
                    let mut scores = vec![0f32; seq];
                    for kp in 0..seq {
                        let mut acc = 0f32;
                        for d in 0..hd {
                            acc += q[d + hd * (h + heads * qp)] * k[d + hd * (kvh + kv_heads * kp)];
                        }
                        scores[kp] = acc * scale + attn_mask[kp + qp * seq];
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for kp in 0..seq {
                            acc += scores[kp] / sum * v[d + hd * (kvh + kv_heads * kp)];
                        }
                        mixed[d + hd * (h + heads * qp)] = acc;
                    }
                }
            }
            let attn = linear_host(
                at(store, &format!("{p}.attn_output.weight")),
                &mixed,
                heads * hd,
                dim,
                seq,
            );
            let attn = rms_host(
                &attn,
                Some(at(store, &format!("{p}.post_attention_norm.weight"))),
                dim,
                seq,
            );
            for i in 0..hidden.len() {
                hidden[i] += attn[i];
            }
            let normed = rms_host(
                &hidden,
                Some(at(store, &format!("{p}.ffn_norm.weight"))),
                dim,
                seq,
            );
            let gate = linear_host(
                at(store, &format!("{p}.ffn_gate.weight")),
                &normed,
                dim,
                ffn,
                seq,
            );
            let up = linear_host(
                at(store, &format!("{p}.ffn_up.weight")),
                &normed,
                dim,
                ffn,
                seq,
            );
            let mid: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| gelu_tanh_host(*g) * u)
                .collect();
            let ffn_out = linear_host(
                at(store, &format!("{p}.ffn_down.weight")),
                &mid,
                ffn,
                dim,
                seq,
            );
            let ffn_out = rms_host(
                &ffn_out,
                Some(at(store, &format!("{p}.post_ffw_norm.weight"))),
                dim,
                seq,
            );
            for i in 0..hidden.len() {
                hidden[i] += ffn_out[i];
            }
            states.push(hidden.clone());
        }
        let mut stacked = vec![0f32; dim * states.len() * seq];
        // Flatten as the graph's `[dim * states, seq]` tensor: per token, the
        // state vectors are contiguous along the channel dim.
        for s in 0..seq {
            for (st, state) in states.iter().enumerate() {
                let normed = rms_host(state, None, dim, seq);
                for d in 0..dim {
                    stacked[st * dim + d + dim * states.len() * s] =
                        normed[d + dim * s] * valid_mask[s];
                }
            }
        }
        stacked
    }

    #[test]
    fn graph_matches_the_host_reference() {
        let config = GemmaConfig::smoke();
        let dir = std::env::temp_dir().join(format!(
            "danmo-gemma-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let encoder = GemmaEncoder::open(&[path]).unwrap();
        assert_eq!(encoder.config().depth, config.depth);
        // 12 valid tokens at the tail of a 32-row sequence (small stand-in for
        // the 1024-row production shape).
        let seq = 32usize;
        let mut ids = vec![PAD_ID; seq];
        let mut valid = vec![0f32; seq];
        let content = [BOS_ID, 9, 13, 5, 6, 20, 12, 14, 21, 7, 8, 10];
        ids[seq - content.len()..].copy_from_slice(&content);
        for v in valid[seq - content.len()..].iter_mut() {
            *v = 1.0;
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let graph_out = encoder.encode_ids(&ctx, &ids, &valid).unwrap();
        let store = smoke_store(&config);
        let host_out = host_forward(&store, &config, &ids, &valid);
        assert_eq!(graph_out.len(), host_out.len());
        let mut max_diff = 0f32;
        for (g, h) in graph_out.iter().zip(&host_out) {
            max_diff = max_diff.max((g - h).abs());
        }
        assert!(max_diff < 1e-3, "graph vs host max diff {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn infer_fails_loud_on_a_missing_block() {
        let dir = std::env::temp_dir().join(format!("danmo-gemma-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.gguf");
        crate::weights::write_f32_gguf(&path, "dummy.weight", 1, 1, &[0.0]).unwrap();
        let err = match GemmaEncoder::open(&[path]) {
            Err(err) => err,
            Ok(_) => panic!("accepted an empty gguf"),
        };
        assert!(err.to_string().contains("token_embd.weight"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fidelity check against the published `unsloth/gemma-3-12b-it`
    /// tokenizer.json. Run on demand:
    /// `LTX_GEMMA_TOKENIZER=/path/to/tokenizer.json cargo test -p danmo-engine -- --ignored gemma_real`
    #[test]
    #[ignore]
    fn gemma_real_tokenizer_matches_hf() {
        let path = std::env::var("LTX_GEMMA_TOKENIZER")
            .unwrap_or_else(|_| "/tmp/gemma_tokenizer.json".to_string());
        let json = std::fs::read_to_string(path).expect("tokenizer.json not found");
        let tok = GemmaTokenizer::from_json(&json).unwrap();
        let cases: &[(&str, &[i32])] = &[
            ("a cat", &[2, 236746, 5866]),
            (
                "A cinematic shot of a cat sitting on a windowsill, golden hour light, shallow depth of field",
                &[
                    2, 236776, 60420, 5719, 529, 496, 5866, 8116, 580, 496, 191646, 236764, 13935,
                    6468, 2214, 236764, 28855, 8744, 529, 2135,
                ],
            ),
            ("你好，世界", &[2, 144626, 236900, 12811]),
            ("Hello  world!  123", &[2, 9259, 138, 12392, 236888, 138, 236770, 236778, 236800]),
            ("café ☕ résumé", &[2, 123125, 236859, 236743, 244360, 118515]),
        ];
        for (text, expected) in cases {
            let (ids, mask) = tok.encode(text).unwrap();
            let real: Vec<i32> = ids
                .iter()
                .zip(&mask)
                .filter(|(_, m)| **m == 1.0)
                .map(|(id, _)| *id)
                .collect();
            assert_eq!(&real, expected, "tokenizing {text:?}");
        }
    }
}
