//! LongCat-Avatar umT5-XXL text encoder (MLX).
//!
//! Semantics match HF `UMT5EncoderModel` and the behavior baseline: per-block
//! relative position bias (32 buckets, bidirectional, max distance 128),
//! unscaled attention with an F32 softmax, gated GELU FFN (`gelu_tanh`),
//! RMSNorm, and no biases on projections. The tokenizer is the bundle's
//! unigram `tokenizer.json` with the Metaspace pre-tokenizer; prompts are
//! truncated to 511 content tokens plus `</s>` and padded to 512 rows.
//!
//! Checkpoint keys are the converted MLX layout shipped in the avatar bundle:
//! `token_embedding.weight`, `norm.weight`,
//! `blocks.{i}.attn.{q,k,v,o}.weight`, `blocks.{i}.{norm1,norm2}.weight`,
//! `blocks.{i}.ffn.{gate_proj,fc1,fc2}.weight`,
//! `blocks.{i}.pos_embedding.embedding.weight`.

use danmo_mlx::{Array, Dtype, Error, LinearWeight, Result, Stream};
use serde_json::Value;
use std::path::Path;

use crate::weights::{load_mlx_directory, MlxWeights, WeightDtype};

pub const TEXT_LEN: usize = 512;
const EOS_ID: i32 = 1;
const UNK_PENALTY: f32 = 10.0;
const MAX_DISTANCE: f32 = 128.0;

#[derive(Debug, Clone, Copy)]
pub struct Umt5Config {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub buckets: usize,
    pub vocab: usize,
}

impl Umt5Config {
    pub fn from_json(value: &Value) -> Result<Self> {
        let integer = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| Error::new("umT5 config", format!("missing {key}")))
        };
        Ok(Self {
            dim: integer("d_model")?,
            layers: integer("num_layers")?,
            heads: integer("num_heads")?,
            ffn: integer("d_ff")?,
            buckets: integer("relative_attention_num_buckets")?,
            vocab: integer("vocab_size")?,
        })
    }
}

pub struct Umt5Encoder {
    config: Umt5Config,
    weights: MlxWeights,
    tokenizer: Umt5Tokenizer,
}

pub struct PromptEncoding {
    /// `[1, 1, 512, dim]` — what the DiT `y_embedder` consumes.
    pub embeds: Array,
    /// `[1, 1, 1, 512]` keep mask.
    pub mask: Array,
    /// Valid (non-pad) token count.
    pub valid: usize,
}

impl Umt5Encoder {
    pub fn open(component_dir: &Path, tokenizer_dir: &Path) -> Result<Self> {
        let config_path = component_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|error| {
            Error::new("umT5 config", format!("{}: {error}", config_path.display()))
        })?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| Error::new("umT5 config", error.to_string()))?;
        let config = Umt5Config::from_json(&value)?;
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        let tokenizer_text = std::fs::read_to_string(&tokenizer_path).map_err(|error| {
            Error::new(
                "umT5 tokenizer",
                format!("{}: {error}", tokenizer_path.display()),
            )
        })?;
        let tokenizer = Umt5Tokenizer::from_json(&tokenizer_text)?;
        let weights = load_mlx_directory(component_dir, WeightDtype::Native, None)?;
        let encoder = Self {
            config,
            weights,
            tokenizer,
        };
        encoder.assert_topology()?;
        Ok(encoder)
    }

    fn assert_topology(&self) -> Result<()> {
        let embedding = self.weights.tensor("token_embedding.weight")?.shape();
        if embedding != [self.config.vocab as i32, self.config.dim as i32] {
            return Err(Error::new(
                "umT5 topology",
                format!(
                    "token_embedding.weight is {embedding:?}, expected [{}, {}]",
                    self.config.vocab, self.config.dim
                ),
            ));
        }
        let bias_table = self
            .weights
            .tensor("blocks.0.pos_embedding.embedding.weight")?
            .shape();
        if bias_table != [self.config.buckets as i32, self.config.heads as i32] {
            return Err(Error::new(
                "umT5 topology",
                format!(
                    "pos_embedding.embedding.weight is {bias_table:?}, expected [{}, {}]",
                    self.config.buckets, self.config.heads
                ),
            ));
        }
        for layer in 0..self.config.layers {
            let prefix = format!("blocks.{layer}");
            for suffix in [
                "attn.q.weight",
                "attn.k.weight",
                "attn.v.weight",
                "attn.o.weight",
                "norm1.weight",
                "norm2.weight",
                "ffn.gate_proj.weight",
                "ffn.fc1.weight",
                "ffn.fc2.weight",
                "pos_embedding.embedding.weight",
            ] {
                let name = format!("{prefix}.{suffix}");
                if !self.weights.has(&name) {
                    return Err(Error::new("umT5 topology", format!("missing {name}")));
                }
            }
        }
        Ok(())
    }

    /// Tokenize and encode one prompt; output is `[1, 1, 512, dim]` embeds and
    /// the `[1, 1, 1, 512]` keep mask the DiT packs by. An empty prompt is a
    /// real call with a single EOS token, never a fabricated zero tensor.
    pub fn encode_prompt(&self, prompt: &str, stream: &Stream) -> Result<PromptEncoding> {
        let ids = self.tokenizer.encode(prompt);
        let valid = ids.len();
        let tokens = Array::from_i32_slice(&ids, &[1, valid as i32])?;
        let table = self.weights.tensor("token_embedding.weight")?;
        let mut hidden = table.take_axis(&tokens, 0, stream)?;
        // [1, valid, dim] -> [1, 512, dim] zero-padded, then [1, 1, 512, dim].
        if valid < TEXT_LEN {
            let pad = Array::from_f32_slice(&[0.0], &[])?
                .astype(hidden.dtype(), stream)?
                .broadcast_to(
                    &[1, (TEXT_LEN - valid) as i32, self.config.dim as i32],
                    stream,
                )?;
            hidden = Array::concatenate_axis(&[hidden, pad], 1, stream)?;
        }
        let sequence = TEXT_LEN as i32;
        for layer in 0..self.config.layers {
            hidden = self.layer(layer, &hidden, valid, stream)?;
        }
        let norm = self.weights.tensor("norm.weight")?;
        hidden = hidden.rms_norm(Some(norm), 1e-6, stream)?;
        let embeds = hidden.reshape(&[1, 1, sequence, self.config.dim as i32], stream)?;
        let mut mask = vec![0f32; TEXT_LEN];
        for slot in mask.iter_mut().take(valid) {
            *slot = 1.0;
        }
        let mask = Array::from_f32_slice(&mask, &[1, 1, 1, sequence])?;
        Ok(PromptEncoding {
            embeds,
            mask,
            valid,
        })
    }

    fn layer(&self, index: usize, input: &Array, valid: usize, stream: &Stream) -> Result<Array> {
        let prefix = format!("blocks.{index}");
        let norm = self.weights.tensor(&format!("{prefix}.norm1.weight"))?;
        let normed = input.rms_norm(Some(norm), 1e-6, stream)?;
        let attended = self.attention(&prefix, &normed, valid, stream)?;
        let hidden = Array::add(input, &attended, stream)?;
        let norm = self.weights.tensor(&format!("{prefix}.norm2.weight"))?;
        let normed = hidden.rms_norm(Some(norm), 1e-6, stream)?;
        let gate = crate::common::mlx_ops::gelu_tanh(
            &self.linear(&format!("{prefix}.ffn.gate_proj"), &normed, stream)?,
            stream,
        )?;
        let up = self.linear(&format!("{prefix}.ffn.fc1"), &normed, stream)?;
        let mid = Array::multiply(&gate, &up, stream)?;
        let out = self.linear(&format!("{prefix}.ffn.fc2"), &mid, stream)?;
        Array::add(&hidden, &out, stream)
    }

    fn linear(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let weight: &LinearWeight = self.weights.linear(&format!("{prefix}.weight"))?;
        weight.apply(input, None, stream)
    }

    fn attention(
        &self,
        prefix: &str,
        input: &Array,
        valid: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let dim = self.config.dim as i32;
        let heads = self.config.heads as i32;
        let head_dim = dim / heads;
        let sequence = TEXT_LEN as i32;
        let split = |x: &Array| -> Result<Array> {
            x.reshape(&[1, sequence, heads, head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)
        };
        let q = split(&self.linear(&format!("{prefix}.attn.q"), input, stream)?)?;
        let k = split(&self.linear(&format!("{prefix}.attn.k"), input, stream)?)?;
        let v = split(&self.linear(&format!("{prefix}.attn.v"), input, stream)?)?;
        // T5 has no 1/sqrt(d) scale; the SDPA kernel gets scale=1 and the
        // per-block position bias as the additive mask (F32 softmax inside).
        let bias = self.position_bias(prefix, valid, stream)?;
        let out = Array::scaled_dot_product_attention(
            &q.astype(Dtype::Float32, stream)?,
            &k.astype(Dtype::Float32, stream)?,
            &v.astype(Dtype::Float32, stream)?,
            1.0,
            "",
            Some(&bias),
            stream,
        )?
        .astype(input.dtype(), stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[1, sequence, dim], stream)?;
        self.linear(&format!("{prefix}.attn.o"), &out, stream)
    }

    /// Per-block bidirectional relative position bias `[1, heads, 512, 512]`
    /// with pad columns zeroed past `valid` (masked by the caller's zero rows:
    /// padded key rows contribute a large negative through the zero-bias rows
    /// of real queries only, matching main's trim-then-pad convention).
    fn position_bias(&self, prefix: &str, valid: usize, stream: &Stream) -> Result<Array> {
        let buckets = self.config.buckets;
        let heads = self.config.heads;
        let table = self
            .weights
            .tensor(&format!("{prefix}.pos_embedding.embedding.weight"))?;
        // table is [buckets, heads] (HF nn.Embedding(num_buckets, heads)).
        // Gather rows by bucket index -> [seq*seq, heads] -> [heads, seq, seq].
        let mut index = Vec::with_capacity(TEXT_LEN * TEXT_LEN);
        for q in 0..TEXT_LEN {
            for k in 0..TEXT_LEN {
                index.push(rel_bucket(q as i64, k as i64, buckets as i64));
            }
        }
        let index = Array::from_i32_slice(&index, &[(TEXT_LEN * TEXT_LEN) as i32])?;
        let flat = table.take_axis(&index, 0, stream)?; // [seq*seq, heads]
        let mut bias = flat
            .reshape(&[TEXT_LEN as i32, TEXT_LEN as i32, heads as i32], stream)?
            .transpose_axes(&[2, 0, 1], stream)?;
        if valid < TEXT_LEN {
            // Zero the pad key columns so pad rows cannot leak signal; pad
            // query rows stay garbage but are discarded downstream via the
            // attention mask.
            let mut keep = vec![f32::NEG_INFINITY; TEXT_LEN];
            for slot in keep.iter_mut().take(valid) {
                *slot = 0.0;
            }
            let keep = Array::from_f32_slice(&keep, &[1, 1, TEXT_LEN as i32])?
                .astype(bias.dtype(), stream)?;
            bias = Array::add(&bias, &keep, stream)?;
        }
        bias.expand_dims(0, stream)
    }
}

/// T5 bidirectional bucket for `rel = k - q` (mirrors HF).
fn rel_bucket(q: i64, k: i64, buckets: i64) -> i32 {
    let rel = k - q;
    let half = buckets / 2;
    let mut bucket = if rel > 0 { half } else { 0 };
    let abs = rel.unsigned_abs() as f32;
    let max_exact = half / 2;
    bucket += if (abs as i64) < max_exact {
        abs as i64
    } else {
        let scale = (MAX_DISTANCE / max_exact as f32).ln();
        let large =
            max_exact + ((abs / max_exact as f32).ln() / scale * (half - max_exact) as f32) as i64;
        large.min(half - 1)
    };
    bucket as i32
}

// ---------------------------------------------------------------------------
// unigram tokenizer (same algorithm as main's Metaspace + Viterbi port)
// ---------------------------------------------------------------------------

pub struct Umt5Tokenizer {
    pieces: Vec<(String, f32)>,
    sorted: Vec<(Box<[u8]>, i32)>,
    min_score: f32,
    unk_id: i32,
    specials: Vec<(String, i32)>,
    replacement: String,
    add_prefix_space: bool,
}

impl Umt5Tokenizer {
    pub fn from_json(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| Error::new("umT5 tokenizer", error.to_string()))?;
        let model = value
            .get("model")
            .ok_or_else(|| Error::new("umT5 tokenizer", "missing model"))?;
        if model.get("type").and_then(Value::as_str) != Some("Unigram") {
            return Err(Error::new(
                "umT5 tokenizer",
                "expected a Unigram model in tokenizer.json",
            ));
        }
        let vocab = model
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("umT5 tokenizer", "model has no vocab"))?;
        let unk_id = model.get("unk_id").and_then(Value::as_i64).unwrap_or(3) as i32;
        let mut pieces = Vec::with_capacity(vocab.len());
        let mut min_score = f32::MAX;
        for entry in vocab {
            let piece = entry
                .get(0)
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("umT5 tokenizer", "vocab entry is not [piece, score]"))?;
            let score =
                entry.get(1).and_then(Value::as_f64).ok_or_else(|| {
                    Error::new("umT5 tokenizer", "vocab entry is not [piece, score]")
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
                    return Err(Error::new(
                        "umT5 tokenizer",
                        "expected the Metaspace pre-tokenizer",
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
                let (Some(id), Some(content)) = (
                    token.get("id").and_then(Value::as_i64),
                    token.get("content").and_then(Value::as_str),
                ) else {
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

    fn viterbi(&self, text: &str) -> Vec<i32> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }
        let unk_score = self.min_score - UNK_PENALTY;
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
                        if target.is_none_or(|node| cand > node.0) {
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
                if target.is_none_or(|node| cand > node.0) {
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

    /// Split added tokens, Viterbi-segment the rest, append `</s>`, truncate
    /// content to `TEXT_LEN - 1` so the final row count stays within 512.
    pub fn encode(&self, raw: &str) -> Vec<i32> {
        let mut ids = Vec::new();
        let mut rest = raw;
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

#[cfg(test)]
mod tests {
    use super::{rel_bucket, Umt5Tokenizer};

    #[test]
    fn rel_bucket_matches_t5_bidirectional() {
        assert_eq!(rel_bucket(0, 0, 32), 0);
        assert_eq!(rel_bucket(0, 1, 32), 17);
        assert_eq!(rel_bucket(1, 0, 32), 1);
        assert_eq!(rel_bucket(0, 200, 32), 31);
    }

    #[test]
    fn tokenizer_smoke_parses_unigram() {
        let json = r#"{
            "model": {"type": "Unigram", "unk_id": 3, "vocab": [
                ["<pad>", 0.0], ["</s>", 0.0], ["<s>", 0.0], ["<unk>", 0.0],
                ["▁a", -0.1], ["▁red", -0.2], ["▁apple", -0.15], ["▁re", -1.0], ["d", -0.3]
            ]},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
        }"#;
        let tokenizer = Umt5Tokenizer::from_json(json).unwrap();
        let ids = tokenizer.encode("a red apple");
        assert_eq!(ids.last(), Some(&1));
        assert_eq!(ids[..ids.len() - 1], [4, 5, 6]);
    }
}
