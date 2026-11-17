//! Wan umT5-XXL text encoder: text cleaning, unigram tokenizer, and the
//! encoder graph.
//!
//! Semantics are a port of main's `wan_text_encoder_mlx.py`:
//! - clean: `html.unescape` twice (numeric entities plus a common named
//!   subset; the full HTML5 table is not ported), collapse all whitespace
//!   runs to one space, strip. `ftfy` is not a dependency, same as main when
//!   the import is absent.
//! - tokenize: HF `tokenizer.json` unigram model with the Metaspace
//!   pre-tokenizer (`▁`, prefix space), Viterbi best-path segmentation
//!   (sd.cpp `t5_unigram_tokenizer.cpp` port: `unk_score = min_score - 10`),
//!   added tokens split out of the text, EOS (`</s>` = 1) appended, content
//!   truncated to `TEXT_LEN - 1` so the final sequence is at most 512.
//! - tower: RMS norm (eps 1e-6) -> unscaled bidirectional attention with a
//!   per-layer relative position bias (32 buckets, bidirectional, max
//!   distance 128, `rel = k - q`) -> residual; RMS -> `gelu_erf(gate) * up`
//!   -> down -> residual. No `1/sqrt(d)` attention scale; softmax in F32.
//! - output: hidden states at the real token length; the caller zero-pads to
//!   `TEXT_LEN` rows, which matches main's "trim then pad" exactly because
//!   the padded key columns never mix into the real rows.
//!
//! GGUF tensor names follow the llama.cpp T5 layout that the published
//! city96 umT5 GGUFs use: `token_embd.weight`, `enc.blk.{i}.attn_{q,k,v,o}`,
//! `attn_norm`, `attn_rel_b` ([heads, buckets]), `ffn_norm`,
//! `ffn_{gate,up,down}`, `enc.output_norm.weight`.

use crate::runtime::{
    GgmlContext, ReusedGraph, RuntimeError, Tensor, TYPE_BF16, TYPE_F16, TYPE_F32,
};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Wan text length: prompts are padded to 512 tokens.
pub const TEXT_LEN: usize = 512;
const RMS_EPS: f32 = 1e-6;
const EOS_ID: i32 = 1;
const MAX_DISTANCE: i64 = 128;

// ---------------------------------------------------------------------------
// text cleaning
// ---------------------------------------------------------------------------

fn named_entity(name: &str) -> Option<char> {
    let c = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{a0}',
        "copy" => '\u{a9}',
        "reg" => '\u{ae}',
        "trade" => '\u{2122}',
        "hellip" => '\u{2026}',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "laquo" => '\u{ab}',
        "raquo" => '\u{bb}',
        "times" => '\u{d7}',
        "divide" => '\u{f7}',
        "deg" => '\u{b0}',
        "plusmn" => '\u{b1}',
        "micro" => '\u{b5}',
        "para" => '\u{b6}',
        "sect" => '\u{a7}',
        "middot" => '\u{b7}',
        "bull" => '\u{2022}',
        "dagger" => '\u{2020}',
        "Dagger" => '\u{2021}',
        "permil" => '\u{2030}',
        "prime" => '\u{2032}',
        "Prime" => '\u{2033}',
        "euro" => '\u{20ac}',
        "pound" => '\u{a3}',
        "yen" => '\u{a5}',
        "cent" => '\u{a2}',
        "curren" => '\u{a4}',
        "brvbar" => '\u{a6}',
        "uml" => '\u{a8}',
        "ordf" => '\u{aa}',
        "ordm" => '\u{ba}',
        "iexcl" => '\u{a1}',
        "iquest" => '\u{bf}',
        "szlig" => '\u{df}',
        _ => return None,
    };
    Some(c)
}

/// One `html.unescape` pass: numeric entities plus the named subset above.
/// Unknown entities stay as-is, matching Python's `html.unescape`.
fn html_unescape_once(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let Some(end) = after.find(';') else {
            out.push('&');
            rest = after;
            continue;
        };
        let entity = &after[..end];
        let decoded = if let Some(num) = entity
            .strip_prefix("#x")
            .or_else(|| entity.strip_prefix("#X"))
        {
            u32::from_str_radix(num, 16).ok().and_then(char::from_u32)
        } else if let Some(num) = entity.strip_prefix('#') {
            num.parse::<u32>().ok().and_then(char::from_u32)
        } else {
            named_entity(entity)
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &after[end + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// main `_basic_clean` (without the optional ftfy) + `_whitespace_clean`.
pub fn clean_text(text: &str) -> String {
    let unescaped = html_unescape_once(&html_unescape_once(text));
    let mut out = String::with_capacity(unescaped.len());
    let mut pending_space = false;
    for c in unescaped.chars() {
        if c.is_whitespace() {
            pending_space = true;
        } else {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(c);
            pending_space = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// unigram tokenizer (sd.cpp t5_unigram port)
// ---------------------------------------------------------------------------

const UNK_PENALTY: f32 = 10.0;

pub struct Umt5Tokenizer {
    /// Vocab pieces by id, with their unigram scores.
    pieces: Vec<(String, f32)>,
    /// `(piece, id)` sorted by piece bytes for binary search.
    sorted: Vec<(Box<[u8]>, i32)>,
    min_score: f32,
    unk_id: i32,
    /// Added tokens, longest content first, split out before the model runs.
    specials: Vec<(String, i32)>,
    replacement: String,
    add_prefix_space: bool,
}

impl Umt5Tokenizer {
    pub fn from_json(json: &str) -> Result<Self, WeightError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
        let model = value
            .get("model")
            .ok_or_else(|| WeightError::NotBuilt("tokenizer.json has no model".into()))?;
        if model.get("type").and_then(Value::as_str) != Some("Unigram") {
            return Err(WeightError::NotBuilt(
                "umT5 expects a Unigram tokenizer model".into(),
            ));
        }
        let vocab = model
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::NotBuilt("tokenizer model has no vocab".into()))?;
        let unk_id = model.get("unk_id").and_then(Value::as_i64).unwrap_or(3) as i32;
        let mut pieces = Vec::with_capacity(vocab.len());
        let mut min_score = f32::MAX;
        for entry in vocab {
            let piece = entry
                .get(0)
                .and_then(Value::as_str)
                .ok_or_else(|| WeightError::NotBuilt("vocab entry is not [piece, score]".into()))?;
            let score =
                entry.get(1).and_then(Value::as_f64).ok_or_else(|| {
                    WeightError::NotBuilt("vocab entry is not [piece, score]".into())
                })? as f32;
            min_score = min_score.min(score);
            pieces.push((piece.to_string(), score));
        }
        let mut sorted: Vec<(Box<[u8]>, i32)> = pieces
            .iter()
            .enumerate()
            .map(|(id, (piece, _))| (piece.as_bytes().into(), id as i32))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let (replacement, add_prefix_space) = match value.get("pre_tokenizer") {
            Some(pre) => {
                if pre.get("type").and_then(Value::as_str) != Some("Metaspace") {
                    return Err(WeightError::NotBuilt(
                        "umT5 expects the Metaspace pre-tokenizer".into(),
                    ));
                }
                (
                    pre.get("replacement")
                        .and_then(Value::as_str)
                        .unwrap_or("\u{2581}")
                        .to_string(),
                    pre.get("add_prefix_space")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                )
            }
            None => ("\u{2581}".to_string(), true),
        };
        let mut specials = Vec::new();
        if let Some(added) = value.get("added_tokens").and_then(Value::as_array) {
            for token in added {
                let Some(id) = token.get("id").and_then(Value::as_i64) else {
                    continue;
                };
                let Some(content) = token.get("content").and_then(Value::as_str) else {
                    continue;
                };
                specials.push((content.to_string(), id as i32));
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Self {
            pieces,
            sorted,
            min_score,
            unk_id,
            specials,
            replacement,
            add_prefix_space,
        })
    }

    fn find(&self, piece: &[u8]) -> Option<i32> {
        self.sorted
            .binary_search_by(|(key, _)| key.as_ref().cmp(piece))
            .ok()
            .map(|index| self.sorted[index].1)
    }

    fn has_prefix(&self, prefix: &[u8]) -> bool {
        let index = self
            .sorted
            .partition_point(|(key, _)| key.as_ref() < prefix);
        self.sorted
            .get(index)
            .is_some_and(|(key, _)| key.starts_with(prefix))
    }

    /// Viterbi best-path segmentation over byte positions.
    fn viterbi(&self, text: &str) -> Vec<i32> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }
        let unk_score = self.min_score - UNK_PENALTY;
        // best[end] = (score, start, id)
        let mut best: Vec<Option<(f32, usize, i32)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, -1));
        let mut start = 0;
        while start < n {
            let base = best[start].map(|node| node.0).unwrap_or(0.0);
            let mblen = utf8_len(bytes[start]).min(n - start);
            let mut has_single = false;
            let mut len = 1;
            while start + len <= n {
                let piece = &bytes[start..start + len];
                match self.find(piece) {
                    Some(id) => {
                        let cand = base + self.pieces[id as usize].1;
                        let target = &mut best[start + len];
                        if target.map_or(true, |node| cand > node.0) {
                            *target = Some((cand, start, id));
                        }
                        if len == mblen {
                            has_single = true;
                        }
                    }
                    None => {
                        if !self.has_prefix(piece) {
                            break;
                        }
                    }
                }
                len += 1;
            }
            if !has_single {
                let cand = base + unk_score;
                let target = &mut best[start + mblen];
                if target.map_or(true, |node| cand > node.0) {
                    *target = Some((cand, start, self.unk_id));
                }
            }
            start += mblen;
        }
        let mut ids = Vec::new();
        let mut end = n;
        while end > 0 {
            let Some((_, start, id)) = best[end] else {
                break;
            };
            ids.push(id);
            end = start;
        }
        ids.reverse();
        ids
    }

    fn metaspace(&self, chunk: &str) -> String {
        let mut out = String::new();
        if self.add_prefix_space {
            out.push_str(&self.replacement);
        }
        for (index, part) in chunk.split(' ').enumerate() {
            if index > 0 {
                out.push_str(&self.replacement);
            }
            out.push_str(part);
        }
        out
    }

    /// Clean, split added tokens, segment, append EOS, truncate to `TEXT_LEN`.
    pub fn encode(&self, raw: &str) -> Vec<i32> {
        let text = clean_text(raw);
        let mut ids = Vec::new();
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let mut hit: Option<(usize, usize, i32)> = None;
            for (content, id) in &self.specials {
                if let Some(at) = rest.find(content.as_str()) {
                    let better = match hit {
                        None => true,
                        Some((best_at, best_len, _)) => {
                            at < best_at || (at == best_at && content.len() > best_len)
                        }
                    };
                    if better {
                        hit = Some((at, content.len(), *id));
                    }
                }
            }
            let (chunk, special, tail) = match hit {
                Some((0, len, id)) => ("", Some(id), &rest[len..]),
                Some((at, len, id)) => (&rest[..at], Some(id), &rest[at + len..]),
                None => (rest, None, ""),
            };
            if !chunk.is_empty() {
                ids.extend(self.viterbi(&self.metaspace(chunk)));
            }
            if let Some(id) = special {
                ids.push(id);
            }
            rest = tail;
        }
        ids.truncate(TEXT_LEN - 1);
        ids.push(EOS_ID);
        ids
    }
}

fn utf8_len(first: u8) -> usize {
    match first >> 4 {
        0..=11 => 1,
        12 | 13 => 2,
        14 => 3,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Umt5Config {
    pub dim: i64,
    pub depth: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub buckets: i64,
    pub ffn: i64,
    pub vocab: i64,
}

impl Umt5Config {
    pub fn smoke() -> Self {
        Self {
            dim: 8,
            depth: 2,
            heads: 2,
            head_dim: 4,
            buckets: 32,
            ffn: 16,
            vocab: 32,
        }
    }

    pub fn infer(weights: &GgufWeights) -> Result<Self, WeightError> {
        let embed = weights.shape("token_embd.weight")?;
        let dim = embed[0];
        let vocab = embed[1];
        let mut depth = 0i64;
        while weights.contains(&format!("enc.blk.{depth}.attn_q.weight")) {
            depth += 1;
        }
        if depth < 1 {
            return Err(WeightError::MissingTensor {
                path: "text_encoder".into(),
                name: "enc.blk.0.attn_q.weight".into(),
            });
        }
        let rel = weights.shape("enc.blk.0.attn_rel_b.weight")?;
        let heads = rel[0];
        let buckets = rel[1];
        if heads < 1 || buckets < 2 || buckets % 2 != 0 {
            return Err(WeightError::Shape {
                name: "enc.blk.0.attn_rel_b.weight".into(),
                expected: vec![dim, 32],
                got: rel.to_vec(),
            });
        }
        if dim % heads != 0 {
            return Err(WeightError::NotBuilt(format!(
                "umT5 dim {dim} is not divisible by heads {heads}"
            )));
        }
        let q = weights.shape("enc.blk.0.attn_q.weight")?;
        let o = weights.shape("enc.blk.0.attn_o.weight")?;
        if q[0] != dim || q[1] != dim || o[0] != dim || o[1] != dim {
            return Err(WeightError::NotBuilt(format!(
                "umT5 attention width is not the residual dim: q {q:?} o {o:?} dim {dim}"
            )));
        }
        let gate = weights.shape("enc.blk.0.ffn_gate.weight")?;
        let down = weights.shape("enc.blk.0.ffn_down.weight")?;
        if gate[0] != dim || down[0] != gate[1] || down[1] != dim {
            return Err(WeightError::NotBuilt(format!(
                "umT5 ffn shape mismatch: gate {gate:?} down {down:?} dim {dim}"
            )));
        }
        Ok(Self {
            dim,
            depth,
            heads,
            head_dim: dim / heads,
            buckets,
            ffn: gate[1],
            vocab,
        })
    }
}

/// `(name, ne0, ne1)` for every encoder weight, in load order.
pub(crate) fn layout(config: &Umt5Config) -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    out.push(("token_embd.weight".into(), config.dim, config.vocab));
    for i in 0..config.depth {
        let p = format!("enc.blk.{i}");
        out.push((format!("{p}.attn_norm.weight"), config.dim, 1));
        out.push((format!("{p}.attn_q.weight"), config.dim, config.dim));
        out.push((format!("{p}.attn_k.weight"), config.dim, config.dim));
        out.push((format!("{p}.attn_v.weight"), config.dim, config.dim));
        out.push((format!("{p}.attn_o.weight"), config.dim, config.dim));
        out.push((
            format!("{p}.attn_rel_b.weight"),
            config.heads,
            config.buckets,
        ));
        out.push((format!("{p}.ffn_norm.weight"), config.dim, 1));
        out.push((format!("{p}.ffn_gate.weight"), config.dim, config.ffn));
        out.push((format!("{p}.ffn_up.weight"), config.dim, config.ffn));
        out.push((format!("{p}.ffn_down.weight"), config.ffn, config.dim));
    }
    out.push(("enc.output_norm.weight".into(), config.dim, 1));
    out
}

/// Norms and the relative-position table stay F32 (widened on load).
fn is_f32_weight(name: &str) -> bool {
    name.ends_with("norm.weight") || name.ends_with("attn_rel_b.weight")
}

// ---------------------------------------------------------------------------
// relative position bias
// ---------------------------------------------------------------------------

/// T5 bidirectional bucket for `rel = k - q` (memory - context).
fn rel_bucket(q: i64, k: i64, buckets: i64) -> i32 {
    let rel = k - q;
    let half = buckets / 2;
    let mut bucket = if rel > 0 { half } else { 0 };
    let abs = rel.abs();
    let max_exact = half / 2;
    bucket += if abs < max_exact {
        abs
    } else {
        let scale = (MAX_DISTANCE as f32 / max_exact as f32).ln();
        let large = max_exact
            + (((abs as f32 / max_exact as f32).ln() / scale) * (half - max_exact) as f32) as i64;
        large.min(half - 1)
    };
    bucket as i32
}

/// Row index into `[heads, buckets]` for every (k, q): `idx[k + seq * q]`.
pub(crate) fn rel_pos_index(seq: i64, buckets: i64) -> Vec<i32> {
    let mut idx = Vec::with_capacity((seq * seq) as usize);
    for q in 0..seq {
        for k in 0..seq {
            idx.push(rel_bucket(q, k, buckets));
        }
    }
    idx
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

struct Umt5Graph {
    graph: ReusedGraph,
    seq: i64,
}

impl Umt5Graph {
    fn build(
        ctx: &GgmlContext,
        config: &Umt5Config,
        seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        if seq < 1 {
            return Err(RuntimeError::Message("text sequence is empty".into()));
        }
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
        let rel_idx = nn(b.i32_1d("rel_pos_idx", seq * seq, true))?;
        let dim = config.dim;
        let heads = config.heads;
        let head = config.head_dim;
        let mut hidden = nn(b.get_rows(wt(&weights, "token_embd.weight")?, token_ids))?;
        if types.get("token_embd.weight").copied() != Some(TYPE_F32) {
            hidden = nn(b.cast_f32(hidden))?;
        }
        for layer in 0..config.depth {
            let p = format!("enc.blk.{layer}");
            let normed = nn(b.mul(
                nn(b.rms_norm(hidden, RMS_EPS))?,
                wt(&weights, &format!("{p}.attn_norm.weight"))?,
            ))?;
            let q = nn(b.mul_mat(wt(&weights, &format!("{p}.attn_q.weight"))?, normed))?;
            let k = nn(b.mul_mat(wt(&weights, &format!("{p}.attn_k.weight"))?, normed))?;
            let v = nn(b.mul_mat(wt(&weights, &format!("{p}.attn_v.weight"))?, normed))?;
            // [head, heads, seq] -> [head, seq, heads]
            let q = nn(b.cont(nn(b.permute(
                nn(b.reshape_3d(q, head, heads, seq))?,
                0,
                2,
                1,
                3,
            ))?))?;
            let k = nn(b.cont(nn(b.permute(
                nn(b.reshape_3d(k, head, heads, seq))?,
                0,
                2,
                1,
                3,
            ))?))?;
            let v = nn(b.cont(nn(b.permute(
                nn(b.reshape_3d(v, head, heads, seq))?,
                0,
                2,
                1,
                3,
            ))?))?;
            // Unscaled QK^T: [lk, lq, heads].
            let mut scores = nn(b.mul_mat(k, q))?;
            // Per-layer relative bias: gather [heads, buckets] rows into
            // [heads, lk*lq], reshape [heads, lk, lq], permute [lk, lq, heads].
            // ggml permute moves input dim k to output dim axes[k].
            let bias = nn(b.get_rows(wt(&weights, &format!("{p}.attn_rel_b.weight"))?, rel_idx))?;
            let bias = nn(b.reshape_3d(bias, heads, seq, seq))?;
            let bias = nn(b.cont(nn(b.permute(bias, 2, 0, 1, 3))?))?;
            scores = nn(b.add(scores, bias))?;
            let attn = nn(b.soft_max(scores))?;
            // [head, lk, heads] -> [lk, head, heads]
            let v = nn(b.cont(nn(b.permute(v, 1, 0, 2, 3))?))?;
            let mixed = nn(b.mul_mat(v, attn))?; // [head, lq, heads]
            let mixed = nn(b.cont(nn(b.permute(mixed, 0, 2, 1, 3))?))?; // [head, heads, lq]
            let mixed = nn(b.reshape_2d(mixed, dim, seq))?;
            let attn_out = nn(b.mul_mat(wt(&weights, &format!("{p}.attn_o.weight"))?, mixed))?;
            hidden = nn(b.add(hidden, attn_out))?;
            let normed = nn(b.mul(
                nn(b.rms_norm(hidden, RMS_EPS))?,
                wt(&weights, &format!("{p}.ffn_norm.weight"))?,
            ))?;
            let gate = nn(b.gelu_erf(nn(
                b.mul_mat(wt(&weights, &format!("{p}.ffn_gate.weight"))?, normed)
            )?))?;
            let up = nn(b.mul_mat(wt(&weights, &format!("{p}.ffn_up.weight"))?, normed))?;
            let ffn = nn(b.mul_mat(
                wt(&weights, &format!("{p}.ffn_down.weight"))?,
                nn(b.mul(gate, up))?,
            ))?;
            hidden = nn(b.add(hidden, ffn))?;
        }
        let out = nn(b.mul(
            nn(b.rms_norm(hidden, RMS_EPS))?,
            wt(&weights, "enc.output_norm.weight")?,
        ))?;
        let graph = b.finish(out)?;
        Ok(Self { graph, seq })
    }

    fn compute(&mut self, ids: &[i32], rel_idx: &[i32]) -> Result<Vec<f32>, RuntimeError> {
        if ids.len() as i64 != self.seq {
            return Err(RuntimeError::Message(format!(
                "token ids has {} values, graph expects {}",
                ids.len(),
                self.seq
            )));
        }
        self.graph.set_i32("token_ids", ids)?;
        self.graph.set_i32("rel_pos_idx", rel_idx)?;
        self.graph.compute()
    }
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

pub struct Umt5Encoder {
    weights: Arc<GgufWeights>,
    paths: Vec<PathBuf>,
    config: Umt5Config,
}

impl Umt5Encoder {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = Umt5Config::infer(&weights)?;
        Ok(Self {
            weights,
            paths: paths.to_vec(),
            config,
        })
    }

    pub fn config(&self) -> Umt5Config {
        self.config
    }

    /// Hidden states at the real token length: `[dim, seq]` flat, ggml order.
    pub fn encode_ids(&self, ctx: &GgmlContext, ids: &[i32]) -> Result<Vec<f32>, WeightError> {
        if ids.is_empty() {
            return Err(WeightError::NotBuilt("prompt has no tokens".into()));
        }
        for id in ids {
            if *id < 0 || *id as i64 >= self.config.vocab {
                return Err(WeightError::NotBuilt(format!(
                    "token id {id} is outside the umT5 vocab {}",
                    self.config.vocab
                )));
            }
        }
        let seq = ids.len() as i64;
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
        let mut graph = Umt5Graph::build(ctx, &self.config, seq, &types)?;
        for (name, ne0, ne1) in &layout {
            let raw = if is_f32_weight(name) {
                let values = self.weights.f32_widened(name, &[*ne0, *ne1])?;
                let mut raw = Vec::with_capacity(values.len() * 4);
                for value in values {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
                raw
            } else {
                self.weights.raw_exact(name, &[*ne0, *ne1])?
            };
            graph.graph.set_bytes(name, &raw)?;
        }
        let rel_idx = rel_pos_index(seq, self.config.buckets);
        let out = graph.compute(ids, &rel_idx)?;
        if let Some(detail) = crate::common::ops::non_finite("text embedding", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }

    /// Clean + tokenize + encode. Returns the hidden states zero-padded to
    /// `TEXT_LEN` rows (`[dim, TEXT_LEN]` flat) and the real token count.
    pub fn encode(
        &self,
        ctx: &GgmlContext,
        tokenizer_json: &str,
        prompt: &str,
    ) -> Result<(Vec<f32>, i64), WeightError> {
        let tokenizer = Umt5Tokenizer::from_json(tokenizer_json)?;
        let ids = tokenizer.encode(prompt);
        let real_len = ids.len() as i64;
        let dim = self.config.dim;
        let values = crate::common::conditioning::get_or_compute(
            &self.paths,
            ctx.kind(),
            &[],
            dim,
            &ids,
            || {
                let hidden = self.encode_ids(ctx, &ids)?;
                let mut padded = vec![0f32; (dim as usize) * TEXT_LEN];
                padded[..hidden.len()].copy_from_slice(&hidden);
                Ok(padded)
            },
        )?;
        Ok((values, real_len))
    }
}

// ---------------------------------------------------------------------------
// smoke writer (session smoke bundle)
// ---------------------------------------------------------------------------

/// The toy unigram tokenizer.json paired with the smoke encoder.
pub fn smoke_tokenizer_json() -> String {
    let vocab = [
        ("<pad>", 0.0),
        ("</s>", 0.0),
        ("<s>", 0.0),
        ("<unk>", 0.0),
        ("\u{2581}", -5.0),
        ("\u{2581}hello", -0.1),
        ("hello", -3.0),
        ("\u{2581}world", -0.2),
        ("world", -1.0),
        ("\u{2581}a", -1.5),
        ("a", -2.0),
        ("b", -1.0),
        ("\u{2581}b", -1.0),
    ];
    let entries: Vec<String> = vocab
        .iter()
        .map(|(piece, score)| format!("[{piece:?}, {score}]"))
        .collect();
    format!(
        r#"{{"version":"1.0","added_tokens":[
            {{"id":0,"content":"<pad>","special":true}},
            {{"id":1,"content":"</s>","special":true}},
            {{"id":2,"content":"<s>","special":true}},
            {{"id":3,"content":"<unk>","special":true}}],
        "normalizer":{{"type":"Replace","pattern":{{"Regex":" {{2,}}"}},"content":" "}},
        "pre_tokenizer":{{"type":"Metaspace","replacement":"▁","add_prefix_space":true}},
        "model":{{"type":"Unigram","unk_id":3,"vocab":[{}]}}}}"#,
        entries.join(",")
    )
}

/// Deterministic smoke values at the `Umt5Config::smoke` layout.
pub fn smoke_store() -> Vec<(String, Vec<f32>)> {
    let config = Umt5Config::smoke();
    let mut store = Vec::new();
    let mut seed = 1.0f32;
    for (name, ne0, ne1) in layout(&config) {
        let len = (ne0 * ne1) as usize;
        let scale = if name.ends_with("attn_rel_b.weight") {
            0.02
        } else {
            0.05
        };
        let data: Vec<f32> = if name.ends_with("norm.weight") {
            (0..len).map(|i| 0.9 + 0.05 * (i % 5) as f32).collect()
        } else {
            (0..len)
                .map(|i| (((i as i32 * 7 + seed as i32 * 13) % 17) as f32 - 8.0) * scale)
                .collect()
        };
        seed += 1.0;
        store.push((name, data));
    }
    store
}

/// Write the smoke umT5 GGUF and return its path.
pub fn write_smoke(dir: &Path) -> Result<PathBuf, WeightError> {
    use crate::weights::{write_f32_nd, F32Nd};
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let store = smoke_store();
    let tensors: Vec<F32Nd> = store
        .iter()
        .zip(layout(&Umt5Config::smoke()))
        .map(|((name, data), (_, ne0, ne1))| F32Nd {
            name: name.as_str(),
            ne: [ne0, ne1, 1, 1],
            data: data.as_slice(),
        })
        .collect();
    let path = dir.join("umt5_smoke.gguf");
    write_f32_nd(&path, &tensors)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;
    use std::path::Path;

    // ----- text cleaning -----

    #[test]
    fn clean_text_unescapes_twice_and_collapses_whitespace() {
        assert_eq!(
            clean_text("Tom &amp;amp; Jerry\n\t  spike"),
            "Tom & Jerry spike"
        );
        assert_eq!(clean_text("  a&#44; b&#x21;  "), "a, b!");
        assert_eq!(clean_text("&unknown; stays"), "&unknown; stays");
        assert_eq!(clean_text("   "), "");
    }

    // ----- tokenizer -----

    fn tokenizer_json() -> String {
        smoke_tokenizer_json()
    }

    #[test]
    fn tokenizer_picks_the_best_path_and_appends_eos() {
        let tok = Umt5Tokenizer::from_json(&tokenizer_json()).unwrap();
        assert_eq!(tok.encode("hello world"), vec![5, 7, EOS_ID]);
        assert_eq!(tok.encode("   "), vec![EOS_ID]);
    }

    #[test]
    fn tokenizer_falls_back_to_unk_per_char() {
        let tok = Umt5Tokenizer::from_json(&tokenizer_json()).unwrap();
        // "x" is not in the vocab: [▁, <unk>, EOS].
        assert_eq!(tok.encode("x"), vec![4, 3, EOS_ID]);
    }

    #[test]
    fn tokenizer_splits_added_tokens() {
        let tok = Umt5Tokenizer::from_json(&tokenizer_json()).unwrap();
        assert_eq!(tok.encode("hello</s>world"), vec![5, 1, 7, EOS_ID]);
    }

    #[test]
    fn tokenizer_truncates_content_to_text_len_minus_one() {
        let tok = Umt5Tokenizer::from_json(&tokenizer_json()).unwrap();
        let long = "b".repeat(600);
        let ids = tok.encode(&long);
        assert_eq!(ids.len(), TEXT_LEN);
        assert_eq!(ids[TEXT_LEN - 1], EOS_ID);
    }

    // ----- graph vs host reference -----

    fn write_smoke(dir: &Path) -> Umt5Encoder {
        Umt5Encoder::open(&[super::write_smoke(dir).unwrap()]).unwrap()
    }

    fn at<'a>(store: &'a [(String, Vec<f32>)], name: &str) -> &'a [f32] {
        store
            .iter()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
            .as_slice()
    }

    fn erf_approx(x: f32) -> f32 {
        // Abramowitz-Stegun 7.1.26, |error| <= 1.5e-7.
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        sign * y
    }

    fn gelu_erf_host(x: f32) -> f32 {
        0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
    }

    fn rms_host(x: &[f32], w: &[f32], dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            let mut sum = 0f32;
            for d in 0..dim {
                sum += x[d + dim * s] * x[d + dim * s];
            }
            let scale = 1.0 / (sum / dim as f32 + RMS_EPS).sqrt();
            for d in 0..dim {
                out[d + dim * s] = x[d + dim * s] * scale * w[d];
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

    fn host_forward(store: &[(String, Vec<f32>)], config: &Umt5Config, ids: &[i32]) -> Vec<f32> {
        let dim = config.dim as usize;
        let heads = config.heads as usize;
        let hd = config.head_dim as usize;
        let ffn = config.ffn as usize;
        let buckets = config.buckets as i64;
        let seq = ids.len();
        let embd = at(store, "token_embd.weight");
        let mut hidden = vec![0f32; dim * seq];
        for (s, id) in ids.iter().enumerate() {
            for d in 0..dim {
                hidden[d + dim * s] = embd[d + dim * (*id as usize)];
            }
        }
        for layer in 0..config.depth {
            let p = format!("enc.blk.{layer}");
            let normed = rms_host(
                &hidden,
                at(store, &format!("{p}.attn_norm.weight")),
                dim,
                seq,
            );
            let q = linear_host(
                at(store, &format!("{p}.attn_q.weight")),
                &normed,
                dim,
                dim,
                seq,
            );
            let k = linear_host(
                at(store, &format!("{p}.attn_k.weight")),
                &normed,
                dim,
                dim,
                seq,
            );
            let v = linear_host(
                at(store, &format!("{p}.attn_v.weight")),
                &normed,
                dim,
                dim,
                seq,
            );
            let rel_b = at(store, &format!("{p}.attn_rel_b.weight"));
            // scores[(k, q, h)] then softmax over k.
            let mut attn = vec![0f32; seq * seq * heads];
            for h in 0..heads {
                for qpos in 0..seq {
                    for kpos in 0..seq {
                        let mut acc = 0f32;
                        for d in 0..hd {
                            acc += q[d + hd * h + dim * qpos] * k[d + hd * h + dim * kpos];
                        }
                        let bucket = rel_bucket(qpos as i64, kpos as i64, buckets) as usize;
                        acc += rel_b[h + heads * bucket];
                        attn[kpos + seq * (qpos + seq * h)] = acc;
                    }
                    let mut max = f32::NEG_INFINITY;
                    for kpos in 0..seq {
                        max = max.max(attn[kpos + seq * (qpos + seq * h)]);
                    }
                    let mut sum = 0f32;
                    for kpos in 0..seq {
                        let e = (attn[kpos + seq * (qpos + seq * h)] - max).exp();
                        attn[kpos + seq * (qpos + seq * h)] = e;
                        sum += e;
                    }
                    for kpos in 0..seq {
                        attn[kpos + seq * (qpos + seq * h)] /= sum;
                    }
                }
            }
            let mut mixed = vec![0f32; dim * seq];
            for h in 0..heads {
                for qpos in 0..seq {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for kpos in 0..seq {
                            acc += attn[kpos + seq * (qpos + seq * h)] * v[d + hd * h + dim * kpos];
                        }
                        mixed[d + hd * h + dim * qpos] = acc;
                    }
                }
            }
            let attn_out = linear_host(
                at(store, &format!("{p}.attn_o.weight")),
                &mixed,
                dim,
                dim,
                seq,
            );
            for (h2, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h2 += a;
            }
            let normed = rms_host(
                &hidden,
                at(store, &format!("{p}.ffn_norm.weight")),
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
                .zip(up.iter())
                .map(|(g, u)| gelu_erf_host(*g) * u)
                .collect();
            let ffn_out = linear_host(
                at(store, &format!("{p}.ffn_down.weight")),
                &mid,
                ffn,
                dim,
                seq,
            );
            for (h2, f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h2 += f;
            }
        }
        rms_host(&hidden, at(store, "enc.output_norm.weight"), dim, seq)
    }

    #[test]
    fn infer_reads_the_smoke_layout() {
        let dir = std::env::temp_dir().join(format!("danmo-umt5-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let encoder = write_smoke(&dir);
        let config = encoder.config();
        assert_eq!(config.dim, 8);
        assert_eq!(config.depth, 2);
        assert_eq!(config.heads, 2);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.buckets, 32);
        assert_eq!(config.ffn, 16);
        assert_eq!(config.vocab, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_umt5_matches_the_host_reference() {
        let dir = std::env::temp_dir().join(format!("danmo-umt5-fwd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = smoke_store();
        let encoder = write_smoke(&dir);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let ids = [4i32, 5, 7, 9, 1];
        let got = encoder.encode_ids(&ctx, &ids).unwrap();
        let want = host_forward(&store, &Umt5Config::smoke(), &ids);
        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w).abs());
        }
        assert!(worst < 1e-4, "graph vs host max diff {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encode_pads_to_text_len() {
        let dir = std::env::temp_dir().join(format!("danmo-umt5-pad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let encoder = write_smoke(&dir);
        // vocab is 32 in the smoke file; the tokenizer json uses ids 0..=12.
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (padded, real_len) = encoder
            .encode(&ctx, &tokenizer_json(), "hello world")
            .unwrap();
        assert_eq!(real_len, 3);
        let dim = Umt5Config::smoke().dim as usize;
        assert_eq!(padded.len(), dim * TEXT_LEN);
        assert!(padded[dim * 3..].iter().all(|v| *v == 0.0));
        assert!(padded[..dim * 3].iter().any(|v| *v != 0.0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
