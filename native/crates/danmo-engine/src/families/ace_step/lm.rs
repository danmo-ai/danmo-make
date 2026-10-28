//! ACE-Step 5Hz Qwen3 causal LM (acestep.cpp `qwen3-lm.h` forward).
//!
//! Prefill can run the full sequence with a causal mask. Greedy decode prefers
//! a fixed-topology `set_rows` KV cache (acestep.cpp); host-side concat past ‖
//! new is the fallback when `max_seq` is tiny.

use crate::common::ops::non_finite;
use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::Path;

const RMS_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AceLmConfig {
    pub vocab: i64,
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub intermediate: i64,
    pub rope_theta: f32,
}

impl AceLmConfig {
    /// Tiny ranks for CPU layout / greedy smoke (not a published card).
    pub fn smoke() -> Self {
        Self {
            vocab: 32,
            hidden: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            rope_theta: 1_000_000.0,
        }
    }
}

pub fn weight_layout(config: AceLmConfig) -> Vec<(String, i64, i64)> {
    // ggml mul_mat: weight is [ne0=in, ne1=out] (same as common qwen3 / fsq).
    let h = config.hidden;
    let q = config.heads * config.head_dim;
    let k = config.kv_heads * config.head_dim;
    let ff = config.intermediate;
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push("model.embed_tokens.weight".into(), h, config.vocab);
    push("model.norm.weight".into(), h, 1);
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        push(format!("{p}.input_layernorm.weight"), h, 1);
        push(format!("{p}.self_attn.q_proj.weight"), h, q);
        push(format!("{p}.self_attn.k_proj.weight"), h, k);
        push(format!("{p}.self_attn.v_proj.weight"), h, k);
        push(format!("{p}.self_attn.q_norm.weight"), config.head_dim, 1);
        push(format!("{p}.self_attn.k_norm.weight"), config.head_dim, 1);
        push(format!("{p}.self_attn.o_proj.weight"), q, h);
        push(format!("{p}.post_attention_layernorm.weight"), h, 1);
        push(format!("{p}.mlp.gate_proj.weight"), h, ff);
        push(format!("{p}.mlp.up_proj.weight"), h, ff);
        push(format!("{p}.mlp.down_proj.weight"), ff, h);
    }
    out
}

pub fn open_lm_gguf(bundle: &Path) -> Result<(GgufWeights, AceLmConfig), WeightError> {
    let path = find_lm_gguf(bundle)?;
    let weights = GgufWeights::open(&path)?;
    let config = infer_config(&weights)?;
    for (name, ne0, ne1) in weight_layout(config) {
        let got = weights.shape(&name)?;
        if got[0] != ne0 || got[1] != ne1 {
            return Err(WeightError::Shape {
                name,
                expected: vec![ne0, ne1],
                got: got.to_vec(),
            });
        }
    }
    Ok((weights, config))
}

fn find_lm_gguf(bundle: &Path) -> Result<std::path::PathBuf, WeightError> {
    if bundle.is_file() {
        return if bundle.extension().and_then(|e| e.to_str()) == Some("gguf") {
            Ok(bundle.to_path_buf())
        } else {
            Err(WeightError::NotBuilt(format!(
                "ACE-Step LM path {} is not a GGUF",
                bundle.display()
            )))
        };
    }
    if !bundle.is_dir() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM bundle {} is missing",
            bundle.display()
        )));
    }
    // Prefer dedicated LM companion dirs (product install layout).
    for sub in ["lm", "companions/lm", "5hz-lm"] {
        let dir = bundle.join(sub);
        if dir.is_dir() {
            if let Ok(path) = collect_lm_gguf(&dir) {
                return Ok(path);
            }
        }
    }
    collect_lm_gguf(bundle)
}

fn collect_lm_gguf(dir: &Path) -> Result<std::path::PathBuf, WeightError> {
    let mut named = Vec::new();
    let mut any = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })? {
        let path = entry
            .map_err(|err| WeightError::Open {
                path: dir.display().to_string(),
                detail: err.to_string(),
            })?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        any.push(path.clone());
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if name.contains("lm") || name.contains("qwen") {
            named.push(path);
        }
    }
    let mut matches = if named.is_empty() { any } else { named };
    match matches.len() {
        0 => Err(WeightError::NotBuilt(format!(
            "ACE-Step 5Hz LM GGUF is missing under {}",
            dir.display()
        ))),
        1 => Ok(matches.remove(0)),
        _ => Err(WeightError::NotBuilt(format!(
            "ACE-Step LM bundle {} has multiple GGUF files; keep a single LM card",
            dir.display()
        ))),
    }
}

fn infer_config(weights: &GgufWeights) -> Result<AceLmConfig, WeightError> {
    // Weights are [ne0=in, ne1=out]: q_proj [hidden, heads*head_dim].
    let embed = weights.shape("model.embed_tokens.weight")?;
    let q = weights.shape("model.layers.0.self_attn.q_proj.weight")?;
    let k = weights.shape("model.layers.0.self_attn.k_proj.weight")?;
    let qn = weights.shape("model.layers.0.self_attn.q_norm.weight")?;
    let gate = weights.shape("model.layers.0.mlp.gate_proj.weight")?;
    let hidden = embed[0];
    let vocab = embed[1];
    let head_dim = qn[0];
    if q[0] != hidden || k[0] != hidden || head_dim < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM GGUF layout could not be inferred".into(),
        ));
    }
    let heads = q[1] / head_dim;
    let kv_heads = k[1] / head_dim;
    let intermediate = gate[1];
    let mut layers = 0i64;
    while weights.contains(&format!("model.layers.{layers}.self_attn.q_proj.weight")) {
        layers += 1;
    }
    if layers < 1 || vocab < 1 || heads < 1 || kv_heads < 1 || intermediate < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM GGUF layout could not be inferred".into(),
        ));
    }
    Ok(AceLmConfig {
        vocab,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        rope_theta: 1_000_000.0,
    })
}

/// Prefill `token_ids` → vocab logits for the last position.
pub fn prefill_logits(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    token_ids: &[i32],
) -> Result<Vec<f32>, WeightError> {
    if token_ids.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM prefill needs at least one token".into(),
        ));
    }
    for &id in token_ids {
        if id < 0 || id as i64 >= config.vocab {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM token id {id} is outside vocab {}",
                config.vocab
            )));
        }
    }
    let mut types = HashMap::new();
    for (name, _, _) in weight_layout(config) {
        types.insert(name.clone(), weights.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut graph = LmPrefill::build(ctx, config, token_ids.len() as i64, &types)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    for (name, ne0, ne1) in weight_layout(config) {
        let data = weights.f32_widened(&name, &[ne0, ne1])?;
        graph
            .load_f32(&name, &data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    let logits = graph
        .compute(token_ids)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    if logits.len() as i64 != config.vocab {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM logits length {} is not vocab {}",
            logits.len(),
            config.vocab
        )));
    }
    if let Some(detail) = non_finite("ACE-Step LM logits", &logits) {
        return Err(WeightError::NotBuilt(detail));
    }
    Ok(logits)
}

/// Greedy decode with fixed-topology `set_rows` KV (acestep.cpp). Host-side
/// concat remains available as [`greedy_generate_host_kv`] for debugging.
pub fn greedy_generate(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    prompt: &[i32],
    max_new: usize,
    stop_ids: &[i32],
) -> Result<Vec<i32>, WeightError> {
    let max_seq = (prompt.len() + max_new).max(1) as i64;
    greedy_generate_set_rows(ctx, weights, config, prompt, max_new, stop_ids, max_seq)
}

/// Fixed-topology decode: one graph, padded KV via `ggml_set_rows`.
pub fn greedy_generate_set_rows(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    prompt: &[i32],
    max_new: usize,
    stop_ids: &[i32],
    max_seq: i64,
) -> Result<Vec<i32>, WeightError> {
    if max_seq < (prompt.len() + max_new) as i64 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM max_seq {max_seq} is shorter than prompt+new {}",
            prompt.len() + max_new
        )));
    }
    let mut session = LmSetRowsSession::open(ctx, weights, config, max_seq)?;
    let mut logits = session.prefill(prompt)?;
    let mut seq = prompt.to_vec();
    for _ in 0..max_new {
        let next = argmax(&logits);
        seq.push(next);
        if stop_ids.contains(&next) {
            break;
        }
        logits = session.decode(next)?;
    }
    Ok(seq)
}

/// Fixed-topology KV session (acestep.cpp single-graph replay).
pub struct LmSetRowsSession {
    config: AceLmConfig,
    max_seq: i64,
    graph: ReusedGraph,
    pos: i64,
}

impl LmSetRowsSession {
    pub fn open(
        ctx: &GgmlContext,
        weights: &GgufWeights,
        config: AceLmConfig,
        max_seq: i64,
    ) -> Result<Self, WeightError> {
        if max_seq < 1 {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM set_rows max_seq must be >= 1".into(),
            ));
        }
        let mut types = HashMap::new();
        for (name, _, _) in weight_layout(config) {
            types.insert(name.clone(), weights.ggml_type_of(&name).unwrap_or(TYPE_F32));
        }
        let graph = LmSetRows::build(ctx, config, max_seq, &types)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        for (name, ne0, ne1) in weight_layout(config) {
            let data = weights.f32_widened(&name, &[ne0, ne1])?;
            graph
                .set_f32(&name, &data)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        }
        let cache_elems = (config.head_dim * max_seq * config.kv_heads) as usize;
        let zeros = vec![0f32; cache_elems];
        for layer in 0..config.layers {
            graph
                .set_f32(&format!("cache_k.{layer}"), &zeros)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            graph
                .set_f32(&format!("cache_v.{layer}"), &zeros)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        }
        Ok(Self {
            config,
            max_seq,
            graph,
            pos: 0,
        })
    }

    pub fn config(&self) -> AceLmConfig {
        self.config
    }

    pub fn pos(&self) -> i64 {
        self.pos
    }

    /// Feed prompt tokens; returns logits after the last prompt token.
    pub fn prefill(&mut self, prompt: &[i32]) -> Result<Vec<f32>, WeightError> {
        if prompt.is_empty() {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM set_rows prefill needs a non-empty prompt".into(),
            ));
        }
        for &id in prompt {
            if id < 0 || id as i64 >= self.config.vocab {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step LM token id {id} is outside vocab {}",
                    self.config.vocab
                )));
            }
        }
        if self.pos != 0 {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM set_rows session was already primed".into(),
            ));
        }
        let mut logits = Vec::new();
        for &tok in prompt {
            logits = self.step(tok)?;
        }
        Ok(logits)
    }

    /// Append one token and return next-step logits.
    pub fn decode(&mut self, token: i32) -> Result<Vec<f32>, WeightError> {
        if token < 0 || token as i64 >= self.config.vocab {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM token id {token} is outside vocab {}",
                self.config.vocab
            )));
        }
        self.step(token)
    }

    fn step(&mut self, token: i32) -> Result<Vec<f32>, WeightError> {
        if self.pos >= self.max_seq {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM set_rows KV exceeded max_seq".into(),
            ));
        }
        let pos = self.pos;
        self.graph
            .set_i32("token_ids", &[token])
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        self.graph
            .set_i32("positions", &[pos as i32])
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        self.graph
            .set_i64("kv_rows", &[pos])
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let mut mask = vec![f32::NEG_INFINITY; self.max_seq as usize];
        for j in 0..=pos as usize {
            if j < mask.len() {
                mask[j] = 0.0;
            }
        }
        self.graph
            .set_f32("attn_mask", &mask)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let logits = self
            .graph
            .compute()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        if logits.len() as i64 != self.config.vocab {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM set_rows logits length {} is not vocab {}",
                logits.len(),
                self.config.vocab
            )));
        }
        if let Some(detail) = non_finite("ACE-Step LM logits", &logits) {
            return Err(WeightError::NotBuilt(detail));
        }
        self.pos += 1;
        Ok(logits)
    }
}

struct LmSetRows;

impl LmSetRows {
    fn build(
        ctx: &GgmlContext,
        config: AceLmConfig,
        max_seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<ReusedGraph, RuntimeError> {
        let mem = 96 * 1024 * 1024
            + (max_seq as usize)
                * (config.layers as usize)
                * (config.head_dim * config.kv_heads) as usize
                * 16;
        let mut b = ctx.builder(mem)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let token_ids = nn(b.i32_1d("token_ids", 1, true))?;
        let pos = nn(b.i32_1d("positions", 1, true))?;
        let kv_rows = nn(b.i64_1d("kv_rows", 1, true))?;
        let mask = nn(b.f32_2d("attn_mask", max_seq, 1, true))?;
        let mut hidden = nn(b.get_rows(
            wt(&weights, "model.embed_tokens.weight")?,
            token_ids,
        ))?;
        for layer in 0..config.layers {
            hidden = lm_set_rows_layer(
                &mut b, &weights, config, layer, hidden, pos, kv_rows, mask, max_seq,
            )?;
        }
        hidden = rms(&b, hidden, wt(&weights, "model.norm.weight")?)?;
        let logits = nn(b.project(wt(&weights, "model.embed_tokens.weight")?, hidden))?;
        b.finish(logits)
    }
}

fn lm_set_rows_layer(
    b: &mut GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceLmConfig,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    kv_rows: Tensor,
    mask: Tensor,
    max_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("model.layers.{layer}");
    let head = config.head_dim;
    let d = head;
    let nkv = config.kv_heads;
    let nh = config.heads;
    let normed = rms(b, hidden, wt(weights, &format!("{p}.input_layernorm.weight"))?)?;
    let q = nn(b.project(wt(weights, &format!("{p}.self_attn.q_proj.weight"))?, normed))?;
    let k = nn(b.project(wt(weights, &format!("{p}.self_attn.k_proj.weight"))?, normed))?;
    let v = nn(b.project(wt(weights, &format!("{p}.self_attn.v_proj.weight"))?, normed))?;
    let q = nn(b.reshape_3d(q, d, nh, 1))?;
    let k = nn(b.reshape_3d(k, d, nkv, 1))?;
    let v = nn(b.reshape_3d(v, d, nkv, 1))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, pos, d as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, d as i32, config.rope_theta)))?;
    // Cache layout [D, max_seq, Nkv] matches flash_attn_ext (acestep.cpp).
    let q = nn(b.cont(b.permute(q, 0, 2, 1, 3)))?; // [D, 1, Nh]
    let k = nn(b.cont(b.permute(k, 0, 2, 1, 3)))?; // [D, 1, Nkv]
    let v = nn(b.cont(b.permute(v, 0, 2, 1, 3)))?;
    let cache_k = nn(b.tensor_3d(
        &format!("cache_k.{layer}"),
        TYPE_F32,
        d,
        max_seq,
        nkv,
        true,
    ))?;
    let cache_v = nn(b.tensor_3d(
        &format!("cache_v.{layer}"),
        TYPE_F32,
        d,
        max_seq,
        nkv,
        true,
    ))?;
    let cache_k = nn(b.set_rows(cache_k, k, kv_rows))?;
    let cache_v = nn(b.set_rows(cache_v, v, kv_rows))?;
    let nb1 = (d as usize) * std::mem::size_of::<f32>();
    let nb2 = (d as usize) * (max_seq as usize) * std::mem::size_of::<f32>();
    let k_full = nn(b.view_3d(cache_k, d, max_seq, nkv, nb1, nb2, 0))?;
    let v_full = nn(b.view_3d(cache_v, d, max_seq, nkv, nb1, nb2, 0))?;
    let mask_f16 = nn(b.cont(b.cast_f16(mask)))?;
    let scale = 1.0 / (d as f32).sqrt();
    let attn = nn(b.flash_attn(q, k_full, v_full, mask_f16, scale))?;
    let attn = nn(b.reshape_2d(attn, nh * d, 1))?;
    let attn = nn(b.project(wt(weights, &format!("{p}.self_attn.o_proj.weight"))?, attn))?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
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

/// Host-side concat KV (fallback / older path).
pub fn greedy_generate_host_kv(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    prompt: &[i32],
    max_new: usize,
    stop_ids: &[i32],
) -> Result<Vec<i32>, WeightError> {
    if prompt.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM generate needs a non-empty prompt".into(),
        ));
    }
    for &id in prompt {
        if id < 0 || id as i64 >= config.vocab {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM token id {id} is outside vocab {}",
                config.vocab
            )));
        }
    }
    let mut types = HashMap::new();
    for (name, _, _) in weight_layout(config) {
        types.insert(name.clone(), weights.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut session = LmKvSession {
        config,
        types,
        cache_k: vec![Vec::new(); config.layers as usize],
        cache_v: vec![Vec::new(); config.layers as usize],
        seq: 0,
        graphs: HashMap::new(),
    };
    let mut logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        logits = session.step(ctx, weights, tok, i as i32)?;
    }
    let mut seq = prompt.to_vec();
    for _ in 0..max_new {
        let next = argmax(&logits);
        seq.push(next);
        if stop_ids.contains(&next) {
            break;
        }
        logits = session.step(ctx, weights, next, session.seq as i32)?;
    }
    Ok(seq)
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as i32
}

fn kv_row(config: AceLmConfig) -> usize {
    (config.head_dim * config.kv_heads) as usize
}

/// Host KV + per-`past_len` decode graphs (concat past ‖ new; set_rows later).
struct LmKvSession {
    config: AceLmConfig,
    types: HashMap<String, u32>,
    cache_k: Vec<Vec<f32>>,
    cache_v: Vec<Vec<f32>>,
    seq: i64,
    graphs: HashMap<i64, ReusedGraph>,
}

impl LmKvSession {
    fn step(
        &mut self,
        ctx: &GgmlContext,
        weights: &GgufWeights,
        token: i32,
        position: i32,
    ) -> Result<Vec<f32>, WeightError> {
        let past = self.seq;
        if !self.graphs.contains_key(&past) {
            let graph = LmDecode::build(ctx, self.config, past, &self.types)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            for (name, ne0, ne1) in weight_layout(self.config) {
                let data = weights.f32_widened(&name, &[ne0, ne1])?;
                graph
                    .set_f32(&name, &data)
                    .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            }
            self.graphs.insert(past, graph);
        }
        let graph = self
            .graphs
            .get_mut(&past)
            .ok_or_else(|| WeightError::NotBuilt("ACE-Step LM decode graph missing".into()))?;
        graph
            .set_i32("token_ids", &[token])
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        graph
            .set_i32("positions", &[position])
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        if past > 0 {
            for layer in 0..self.config.layers {
                let k_name = format!("past_k.{layer}");
                let v_name = format!("past_v.{layer}");
                graph
                    .set_f32(&k_name, &self.cache_k[layer as usize])
                    .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
                graph
                    .set_f32(&v_name, &self.cache_v[layer as usize])
                    .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            }
        }
        let packed = graph
            .compute()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let vocab = self.config.vocab as usize;
        let row = kv_row(self.config);
        let need = vocab + (self.config.layers as usize) * 2 * row;
        if packed.len() != need {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM decode packed length {} is not {need}",
                packed.len()
            )));
        }
        if let Some(detail) = non_finite("ACE-Step LM logits", &packed[..vocab]) {
            return Err(WeightError::NotBuilt(detail));
        }
        let mut offset = vocab;
        for layer in 0..self.config.layers as usize {
            self.cache_k[layer].extend_from_slice(&packed[offset..offset + row]);
            offset += row;
            self.cache_v[layer].extend_from_slice(&packed[offset..offset + row]);
            offset += row;
        }
        self.seq += 1;
        Ok(packed[..vocab].to_vec())
    }
}

struct LmDecode;

impl LmDecode {
    fn build(
        ctx: &GgmlContext,
        config: AceLmConfig,
        past: i64,
        types: &HashMap<String, u32>,
    ) -> Result<ReusedGraph, RuntimeError> {
        let kv_len = past + 1;
        let mem = 64 * 1024 * 1024
            + ((kv_len * kv_len * 8) as usize).max(0)
            + (past.max(0) as usize) * (config.layers as usize) * kv_row(config) * 8;
        let mut b = ctx.builder(mem)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let token_ids = nn(b.i32_1d("token_ids", 1, true))?;
        let pos = nn(b.i32_1d("positions", 1, true))?;
        let mut hidden = nn(b.get_rows(
            wt(&weights, "model.embed_tokens.weight")?,
            token_ids,
        ))?;
        let mut new_ks = Vec::new();
        let mut new_vs = Vec::new();
        for layer in 0..config.layers {
            let past_k = if past > 0 {
                Some(nn(b.tensor_3d(
                    &format!("past_k.{layer}"),
                    TYPE_F32,
                    config.head_dim,
                    config.kv_heads,
                    past,
                    true,
                ))?)
            } else {
                None
            };
            let past_v = if past > 0 {
                Some(nn(b.tensor_3d(
                    &format!("past_v.{layer}"),
                    TYPE_F32,
                    config.head_dim,
                    config.kv_heads,
                    past,
                    true,
                ))?)
            } else {
                None
            };
            let (h, nk, nv) =
                lm_decode_layer(&b, &weights, config, layer, hidden, pos, past_k, past_v)?;
            hidden = h;
            new_ks.push(nk);
            new_vs.push(nv);
        }
        hidden = rms(&b, hidden, wt(&weights, "model.norm.weight")?)?;
        let logits = nn(b.project(wt(&weights, "model.embed_tokens.weight")?, hidden))?;
        // Pack logits ‖ new_k ‖ new_v per layer for host cache append.
        let mut packed = nn(b.cont(b.reshape_2d(logits, config.vocab, 1)))?;
        let row = config.head_dim * config.kv_heads;
        for (nk, nv) in new_ks.into_iter().zip(new_vs) {
            let nk = nn(b.cont(b.reshape_2d(nk, row, 1)))?;
            let nv = nn(b.cont(b.reshape_2d(nv, row, 1)))?;
            packed = nn(b.cont(b.concat(packed, nk, 0)))?;
            packed = nn(b.cont(b.concat(packed, nv, 0)))?;
        }
        b.finish(packed)
    }
}

fn lm_decode_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceLmConfig,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    past_k: Option<Tensor>,
    past_v: Option<Tensor>,
) -> Result<(Tensor, Tensor, Tensor), RuntimeError> {
    let p = format!("model.layers.{layer}");
    let head = config.head_dim;
    let normed = rms(b, hidden, wt(weights, &format!("{p}.input_layernorm.weight"))?)?;
    let q = nn(b.project(wt(weights, &format!("{p}.self_attn.q_proj.weight"))?, normed))?;
    let k = nn(b.project(wt(weights, &format!("{p}.self_attn.k_proj.weight"))?, normed))?;
    let v = nn(b.project(wt(weights, &format!("{p}.self_attn.v_proj.weight"))?, normed))?;
    let q = nn(b.reshape_3d(q, head, config.heads, 1))?;
    let k = nn(b.reshape_3d(k, head, config.kv_heads, 1))?;
    let v = nn(b.reshape_3d(v, head, config.kv_heads, 1))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, config.rope_theta)))?;
    let new_k = k;
    let new_v = v;
    let (k_full, v_full) = match (past_k, past_v) {
        (Some(past_k), Some(past_v)) => (
            nn(b.cont(b.concat(past_k, new_k, 2)))?,
            nn(b.cont(b.concat(past_v, new_v, 2)))?,
        ),
        _ => (new_k, new_v),
    };
    let attn = flash_attend(b, q, k_full, v_full, std::ptr::null_mut())?;
    let attn = nn(b.reshape_2d(attn, config.heads * head, 1))?;
    let attn = nn(b.project(wt(weights, &format!("{p}.self_attn.o_proj.weight"))?, attn))?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
    )?;
    let gate = nn(b.silu(nn(b.project(
        wt(weights, &format!("{p}.mlp.gate_proj.weight"))?,
        normed,
    ))?))?;
    let up = nn(b.project(wt(weights, &format!("{p}.mlp.up_proj.weight"))?, normed))?;
    let ff = nn(b.mul(gate, up))?;
    let ff = nn(b.project(wt(weights, &format!("{p}.mlp.down_proj.weight"))?, ff))?;
    let hidden = nn(b.add(hidden, ff))?;
    Ok((hidden, new_k, new_v))
}

struct LmPrefill {
    graph: ReusedGraph,
    seq: i64,
}

impl LmPrefill {
    fn build(
        ctx: &GgmlContext,
        config: AceLmConfig,
        seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mem = 64 * 1024 * 1024 + (seq * seq * 8).max(0) as usize;
        let mut b = ctx.builder(mem)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
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
        let mut hidden = nn(b.get_rows(
            wt(&weights, "model.embed_tokens.weight")?,
            token_ids,
        ))?;
        for layer in 0..config.layers {
            hidden = lm_layer(&b, &weights, config, layer, hidden, pos, mask, seq)?;
        }
        hidden = rms(&b, hidden, wt(&weights, "model.norm.weight")?)?;
        // Last token hidden [H] (acestep.cpp qw3lm_forward view_1d).
        let last = if seq > 1 {
            let offset = ((seq - 1) as usize)
                * (config.hidden as usize)
                * std::mem::size_of::<f32>();
            nn(b.cont(b.view_1d(hidden, config.hidden, offset)))?
        } else {
            hidden
        };
        // Tied lm_head: logits[V] = embed @ last  (embed is [H, V]).
        let logits = nn(b.project(wt(&weights, "model.embed_tokens.weight")?, last))?;
        let graph = b.finish(logits)?;
        Ok(Self { graph, seq })
    }

    fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    fn compute(&mut self, ids: &[i32]) -> Result<Vec<f32>, RuntimeError> {
        if ids.len() as i64 != self.seq {
            return Err(RuntimeError::Message(format!(
                "LM token ids {} != graph seq {}",
                ids.len(),
                self.seq
            )));
        }
        self.graph.set_i32("token_ids", ids)?;
        let positions: Vec<i32> = (0..self.seq as i32).collect();
        self.graph.set_i32("positions", &positions)?;
        let mut mask = vec![0f32; (self.seq * self.seq) as usize];
        for q in 0..self.seq {
            for k in 0..self.seq {
                if k > q {
                    mask[(k + q * self.seq) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        self.graph.set_f32("attn_mask", &mask)?;
        self.graph.compute()
    }
}

fn lm_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceLmConfig,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("model.layers.{layer}");
    let head = config.head_dim;
    let normed = rms(b, hidden, wt(weights, &format!("{p}.input_layernorm.weight"))?)?;
    let q = nn(b.project(wt(weights, &format!("{p}.self_attn.q_proj.weight"))?, normed))?;
    let k = nn(b.project(wt(weights, &format!("{p}.self_attn.k_proj.weight"))?, normed))?;
    let v = nn(b.project(wt(weights, &format!("{p}.self_attn.v_proj.weight"))?, normed))?;
    let q = nn(b.reshape_3d(q, head, config.heads, seq))?;
    let k = nn(b.reshape_3d(k, head, config.kv_heads, seq))?;
    let v = nn(b.reshape_3d(v, head, config.kv_heads, seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, RMS_EPS))?,
        wt(weights, &format!("{p}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, config.rope_theta)))?;
    let attn = flash_attend(b, q, k, v, mask)?;
    let attn = nn(b.reshape_2d(attn, config.heads * head, seq))?;
    let attn = nn(b.project(wt(weights, &format!("{p}.self_attn.o_proj.weight"))?, attn))?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
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

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|t| !t.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("LM graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

/// Write a zero-weight smoke LM GGUF under `dir` (test helper).
#[cfg(test)]
pub fn write_smoke_lm(dir: &Path, config: AceLmConfig) -> Result<(), WeightError> {
    use crate::weights::{write_f32_nd, F32Nd};
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let owned: Vec<(String, [i64; 4], Vec<f32>)> = weight_layout(config)
        .into_iter()
        .map(|(name, ne0, ne1)| (name, [ne0, ne1, 1, 1], vec![0f32; (ne0 * ne1) as usize]))
        .collect();
    let tensors: Vec<F32Nd<'_>> = owned
        .iter()
        .map(|(name, ne, data)| F32Nd {
            name,
            ne: *ne,
            data,
        })
        .collect();
    write_f32_nd(&dir.join("acestep-lm-smoke.gguf"), &tensors)
}

#[cfg(test)]
mod tests {
    use super::{
        greedy_generate, greedy_generate_host_kv, greedy_generate_set_rows, prefill_logits,
        weight_layout, write_smoke_lm, AceLmConfig,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::GgufWeights;

    #[test]
    fn smoke_layout_lists_tied_embed_head() {
        let layout = weight_layout(AceLmConfig::smoke());
        assert!(layout
            .iter()
            .any(|(n, _, _)| n == "model.embed_tokens.weight"));
        assert!(layout.iter().any(|(n, _, _)| n == "model.norm.weight"));
        assert!(layout
            .iter()
            .any(|(n, _, _)| n == "model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn smoke_prefill_and_greedy_are_finite() {
        let dir = std::env::temp_dir().join(format!(
            "ace-lm-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = AceLmConfig::smoke();
        write_smoke_lm(&dir, config).unwrap();
        let weights = GgufWeights::open(&dir.join("acestep-lm-smoke.gguf")).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let prompt = vec![1i32, 2, 3];
        let logits = prefill_logits(&ctx, &weights, config, &prompt).unwrap();
        assert_eq!(logits.len() as i64, config.vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        let out = greedy_generate(&ctx, &weights, config, &prompt, 4, &[0]).unwrap();
        assert!(out.len() >= prompt.len());
        assert!(out.len() <= prompt.len() + 4);
        let via_rows =
            greedy_generate_set_rows(&ctx, &weights, config, &prompt, 4, &[0], 8).unwrap();
        assert_eq!(via_rows.len(), out.len());
        let via_host = greedy_generate_host_kv(&ctx, &weights, config, &prompt, 4, &[0]).unwrap();
        assert!(via_host.len() >= prompt.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
