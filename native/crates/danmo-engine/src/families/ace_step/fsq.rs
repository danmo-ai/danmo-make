//! ACE-Step FSQ tokenizer + detokenizer (acestep.cpp `fsq-tok.h` / `fsq-detok.h`).
//!
//! Roundtrip: VAE latents `[T_25Hz, 64]` → 5Hz codes → reconstructed `[T', 64]`.
//! Quant codec uses HOT-Step soft-clamp (`fsq_quant`). Weights live in the DiT GGUF
//! under `tokenizer.*` and `detokenizer.*`.

use super::fsq_quant::{decode_index, encode_index, FSQ_NDIMS, POOL_WINDOW};
use super::vae::LATENT_CHANNELS;
use crate::common::ops::non_finite;
use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

const HIDDEN: i64 = 2048;
const HEADS: i64 = 16;
const KV_HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;
const FF: i64 = 6144;
const LAYERS: i64 = 2;
const GRAPH_MEM: usize = 128 * 1024 * 1024;

/// Tokenize 25Hz VAE latents → 5Hz FSQ codes (pad with silence to ×5).
pub fn tokenize(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    latents: &[f32],
    silence_latent: &[f32],
) -> Result<Vec<i32>, WeightError> {
    let t_25 = latents.len() as i64 / LATENT_CHANNELS;
    if t_25 < 1 || latents.len() as i64 != t_25 * LATENT_CHANNELS {
        return Err(WeightError::NotBuilt(
            "ACE-Step FSQ tokenize needs a non-empty multiple of 64 latents".into(),
        ));
    }
    if silence_latent.len() < LATENT_CHANNELS as usize {
        return Err(WeightError::NotBuilt(
            "ACE-Step FSQ tokenize needs silence_latent frames for padding".into(),
        ));
    }
    check_tok_layout(dit)?;
    let pad = ((POOL_WINDOW - (t_25 % POOL_WINDOW)) % POOL_WINDOW) as usize;
    let t_padded = t_25 as usize + pad;
    let t_5 = t_padded / POOL_WINDOW as usize;
    let mut input = vec![0f32; t_padded * LATENT_CHANNELS as usize];
    input[..latents.len()].copy_from_slice(latents);
    for i in 0..pad {
        let dst = (t_25 as usize + i) * LATENT_CHANNELS as usize;
        input[dst..dst + LATENT_CHANNELS as usize]
            .copy_from_slice(&silence_latent[..LATENT_CHANNELS as usize]);
    }

    let mut types = HashMap::new();
    for (name, _, _) in tok_weight_layout() {
        types.insert(name.clone(), dit.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut session = TokSession::build(ctx, &types).map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    for (name, ne0, ne1) in tok_weight_layout() {
        let data = dit.f32_widened(&name, &[ne0, ne1])?;
        session
            .load_f32(&name, &data)
            .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    }

    let mut codes = Vec::with_capacity(t_5);
    for g in 0..t_5 {
        let start = g * POOL_WINDOW as usize * LATENT_CHANNELS as usize;
        let end = start + POOL_WINDOW as usize * LATENT_CHANNELS as usize;
        let raw = session
            .forward(&input[start..end])
            .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
        if raw.len() != FSQ_NDIMS {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step FSQ tok out length {} != {FSQ_NDIMS}",
                raw.len()
            )));
        }
        if let Some(detail) = non_finite("ACE-Step FSQ tok", &raw) {
            return Err(WeightError::NotBuilt(detail));
        }
        let mut vals = [0f32; FSQ_NDIMS];
        vals.copy_from_slice(&raw);
        codes.push(encode_index(&vals));
    }
    Ok(codes)
}

/// Detokenize 5Hz codes → 25Hz VAE latents `[T_5Hz*5, 64]`.
pub fn detokenize(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    codes: &[i32],
) -> Result<Vec<f32>, WeightError> {
    if codes.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step FSQ detokenize needs at least one code".into(),
        ));
    }
    check_detok_layout(dit)?;
    let mut types = HashMap::new();
    for (name, _, _) in detok_weight_layout() {
        types.insert(name.clone(), dit.ggml_type_of(&name).unwrap_or(TYPE_F32));
    }
    let mut session =
        DetokSession::build(ctx, &types).map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    for (name, ne0, ne1) in detok_weight_layout() {
        let data = dit.f32_widened(&name, &[ne0, ne1])?;
        session
            .load_f32(&name, &data)
            .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    }

    let mut out = vec![0f32; codes.len() * POOL_WINDOW as usize * LATENT_CHANNELS as usize];
    for (g, &code) in codes.iter().enumerate() {
        let fsq = decode_index(code);
        let frames = session
            .forward(&fsq)
            .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
        if frames.len() != (POOL_WINDOW * LATENT_CHANNELS) as usize {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step FSQ detok out length {} != {}",
                frames.len(),
                POOL_WINDOW * LATENT_CHANNELS
            )));
        }
        if let Some(detail) = non_finite("ACE-Step FSQ detok", &frames) {
            return Err(WeightError::NotBuilt(detail));
        }
        let dst = g * POOL_WINDOW as usize * LATENT_CHANNELS as usize;
        out[dst..dst + frames.len()].copy_from_slice(&frames);
    }
    Ok(out)
}

/// Cover FSQ roundtrip: tokenize → detokenize, copy min(T_rt, T_cover) back.
pub fn roundtrip(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    cover_latents: &[f32],
    silence_latent: &[f32],
) -> Result<Vec<f32>, WeightError> {
    let codes = tokenize(ctx, dit, cover_latents, silence_latent)?;
    let rt = detokenize(ctx, dit, &codes)?;
    let t_cover = cover_latents.len() / LATENT_CHANNELS as usize;
    let t_rt = rt.len() / LATENT_CHANNELS as usize;
    let copy_t = t_rt.min(t_cover);
    let mut out = cover_latents.to_vec();
    let n = copy_t * LATENT_CHANNELS as usize;
    out[..n].copy_from_slice(&rt[..n]);
    Ok(out)
}

pub fn check_tok_layout(dit: &GgufWeights) -> Result<(), WeightError> {
    for (name, ne0, ne1) in tok_weight_layout() {
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

pub fn check_detok_layout(dit: &GgufWeights) -> Result<(), WeightError> {
    for (name, ne0, ne1) in detok_weight_layout() {
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

/// ggml `mul_mat` weights use `[ne0=in, ne1=out]` (acestep.cpp Linear comments).
fn tok_weight_layout() -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    out.push((
        "tokenizer.audio_acoustic_proj.weight".into(),
        LATENT_CHANNELS,
        HIDDEN,
    ));
    out.push(("tokenizer.audio_acoustic_proj.bias".into(), HIDDEN, 1));
    out.push((
        "tokenizer.attention_pooler.embed_tokens.weight".into(),
        HIDDEN,
        HIDDEN,
    ));
    out.push((
        "tokenizer.attention_pooler.embed_tokens.bias".into(),
        HIDDEN,
        1,
    ));
    out.push(("tokenizer.attention_pooler.special_token".into(), HIDDEN, 1));
    out.push(("tokenizer.attention_pooler.norm.weight".into(), HIDDEN, 1));
    push_qwen_layers(&mut out, "tokenizer.attention_pooler");
    out.push((
        "tokenizer.quantizer.project_in.weight".into(),
        HIDDEN,
        FSQ_NDIMS as i64,
    ));
    out.push((
        "tokenizer.quantizer.project_in.bias".into(),
        FSQ_NDIMS as i64,
        1,
    ));
    out
}

fn detok_weight_layout() -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    out.push((
        "tokenizer.quantizer.project_out.weight".into(),
        FSQ_NDIMS as i64,
        HIDDEN,
    ));
    out.push(("tokenizer.quantizer.project_out.bias".into(), HIDDEN, 1));
    out.push(("detokenizer.embed_tokens.weight".into(), HIDDEN, HIDDEN));
    out.push(("detokenizer.embed_tokens.bias".into(), HIDDEN, 1));
    // special_tokens [2048, 5]
    out.push(("detokenizer.special_tokens".into(), HIDDEN, POOL_WINDOW));
    out.push(("detokenizer.norm.weight".into(), HIDDEN, 1));
    push_qwen_layers(&mut out, "detokenizer");
    out.push(("detokenizer.proj_out.weight".into(), HIDDEN, LATENT_CHANNELS));
    out.push(("detokenizer.proj_out.bias".into(), LATENT_CHANNELS, 1));
    out
}

fn push_qwen_layers(out: &mut Vec<(String, i64, i64)>, prefix: &str) {
    let q = HEADS * HEAD_DIM;
    let k = KV_HEADS * HEAD_DIM;
    for layer in 0..LAYERS {
        let p = format!("{prefix}.layers.{layer}");
        out.push((format!("{p}.input_layernorm.weight"), HIDDEN, 1));
        // Linear weights: [in, out]
        out.push((format!("{p}.self_attn.q_proj.weight"), HIDDEN, q));
        out.push((format!("{p}.self_attn.k_proj.weight"), HIDDEN, k));
        out.push((format!("{p}.self_attn.v_proj.weight"), HIDDEN, k));
        out.push((format!("{p}.self_attn.q_norm.weight"), HEAD_DIM, 1));
        out.push((format!("{p}.self_attn.k_norm.weight"), HEAD_DIM, 1));
        out.push((format!("{p}.self_attn.o_proj.weight"), q, HIDDEN));
        out.push((format!("{p}.post_attention_layernorm.weight"), HIDDEN, 1));
        out.push((format!("{p}.mlp.gate_proj.weight"), HIDDEN, FF));
        out.push((format!("{p}.mlp.up_proj.weight"), HIDDEN, FF));
        out.push((format!("{p}.mlp.down_proj.weight"), FF, HIDDEN));
    }
}

struct TokSession {
    graph: ReusedGraph,
}

impl TokSession {
    fn build(ctx: &GgmlContext, types: &HashMap<String, u32>) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in tok_weight_layout() {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let t = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if t.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, t);
        }
        let tok_in = nn(b.f32_2d("tok_in", LATENT_CHANNELS, POOL_WINDOW, true))?;
        let pos = nn(b.i32_1d("tok_pos", POOL_WINDOW + 1, true))?;
        let projected = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "tokenizer.audio_acoustic_proj.weight")?,
                tok_in,
            ))?,
            wt(&weights, "tokenizer.audio_acoustic_proj.bias")?,
        )?;
        let embedded = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "tokenizer.attention_pooler.embed_tokens.weight")?,
                projected,
            ))?,
            wt(&weights, "tokenizer.attention_pooler.embed_tokens.bias")?,
        )?;
        let cls = nn(b.reshape_2d(
            wt(&weights, "tokenizer.attention_pooler.special_token")?,
            HIDDEN,
            1,
        ))?;
        let mut hidden = nn(b.cont(b.concat(cls, embedded, 1)))?;
        let seq = POOL_WINDOW + 1;
        for layer in 0..LAYERS {
            hidden = qwen_layer(
                &b,
                &weights,
                "tokenizer.attention_pooler",
                layer,
                hidden,
                pos,
                seq,
            )?;
        }
        hidden = rms(
            &b,
            hidden,
            wt(&weights, "tokenizer.attention_pooler.norm.weight")?,
        )?;
        // CLS = column 0: view [H, 1]
        let cls_out = nn(b.view_2d(hidden, HIDDEN, 1, (HIDDEN as usize) * 4, 0))?;
        let fsq = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "tokenizer.quantizer.project_in.weight")?,
                cls_out,
            ))?,
            wt(&weights, "tokenizer.quantizer.project_in.bias")?,
        )?;
        let graph = b.finish(fsq)?;
        Ok(Self { graph })
    }

    fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    fn forward(&mut self, group: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        // ggml [ne0=64, ne1=5]: contiguous columns match time-major [5, 64].
        self.graph.set_f32("tok_in", group)?;
        let positions: Vec<i32> = (0..=POOL_WINDOW as i32).collect();
        self.graph.set_i32("tok_pos", &positions)?;
        self.graph.compute()
    }
}

struct DetokSession {
    graph: ReusedGraph,
}

impl DetokSession {
    fn build(ctx: &GgmlContext, types: &HashMap<String, u32>) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in detok_weight_layout() {
            let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
            let t = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if t.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, t);
        }
        let fsq_in = nn(b.f32_1d("fsq_in", FSQ_NDIMS as i64, true))?;
        let pos = nn(b.i32_1d("detok_pos", POOL_WINDOW, true))?;
        let quantized = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "tokenizer.quantizer.project_out.weight")?,
                fsq_in,
            ))?,
            wt(&weights, "tokenizer.quantizer.project_out.bias")?,
        )?;
        let embedded = add_bias(
            &b,
            nn(b.project(
                wt(&weights, "detokenizer.embed_tokens.weight")?,
                nn(b.reshape_2d(quantized, HIDDEN, 1))?,
            ))?,
            wt(&weights, "detokenizer.embed_tokens.bias")?,
        )?;
        let special = nn(b.reshape_2d(
            wt(&weights, "detokenizer.special_tokens")?,
            HIDDEN,
            POOL_WINDOW,
        ))?;
        let mut hidden = nn(b.add(nn(b.repeat(embedded, special))?, special))?;
        for layer in 0..LAYERS {
            hidden = qwen_layer(&b, &weights, "detokenizer", layer, hidden, pos, POOL_WINDOW)?;
        }
        hidden = rms(&b, hidden, wt(&weights, "detokenizer.norm.weight")?)?;
        let output = add_bias(
            &b,
            nn(b.project(wt(&weights, "detokenizer.proj_out.weight")?, hidden))?,
            wt(&weights, "detokenizer.proj_out.bias")?,
        )?;
        let graph = b.finish(output)?;
        Ok(Self { graph })
    }

    fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    fn forward(&mut self, fsq: &[f32; FSQ_NDIMS]) -> Result<Vec<f32>, RuntimeError> {
        self.graph.set_f32("fsq_in", fsq)?;
        let positions: Vec<i32> = (0..POOL_WINDOW as i32).collect();
        self.graph.set_i32("detok_pos", &positions)?;
        // Output [64, 5] channel-contiguous = time-major [5, 64] already.
        self.graph.compute()
    }
}

fn qwen_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    layer: i64,
    hidden: Tensor,
    pos: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let p = format!("{prefix}.layers.{layer}");
    let normed = rms(b, hidden, wt(weights, &format!("{p}.input_layernorm.weight"))?)?;
    let q = nn(b.project(wt(weights, &format!("{p}.self_attn.q_proj.weight"))?, normed))?;
    let k = nn(b.project(wt(weights, &format!("{p}.self_attn.k_proj.weight"))?, normed))?;
    let v = nn(b.project(wt(weights, &format!("{p}.self_attn.v_proj.weight"))?, normed))?;
    let q = nn(b.reshape_3d(q, HEAD_DIM, HEADS, seq))?;
    let k = nn(b.reshape_3d(k, HEAD_DIM, KV_HEADS, seq))?;
    let v = nn(b.reshape_3d(v, HEAD_DIM, KV_HEADS, seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, 1e-6))?,
        wt(weights, &format!("{p}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, 1e-6))?,
        wt(weights, &format!("{p}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, pos, HEAD_DIM as i32, 1_000_000.0)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, HEAD_DIM as i32, 1_000_000.0)))?;
    let attn = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let attn = nn(b.reshape_2d(attn, HEADS * HEAD_DIM, seq))?;
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
    nn(b.mul(nn(b.rms_norm(x, 1e-6))?, weight))
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|t| !t.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("FSQ graph is missing {name}")))
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
        check_detok_layout, check_tok_layout, detok_weight_layout, detokenize, tok_weight_layout,
        tokenize, TokSession, DetokSession,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};
    use std::collections::HashMap;
    use std::path::Path;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ace-fsq-{name}-{}-{}",
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

    fn write_fsq_bundle(dir: &Path) {
        let mut owned = Vec::new();
        for (name, ne0, ne1) in tok_weight_layout()
            .into_iter()
            .chain(detok_weight_layout())
        {
            let n = (ne0 * ne1) as usize;
            let data = if name.contains("norm") {
                vec![1.0f32; n]
            } else {
                vec![0.0f32; n]
            };
            owned.push((name, [ne0, ne1, 1, 1], data));
        }
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        write_f32_nd(&dir.join("dit-fsq.gguf"), &tensors).unwrap();
    }

    #[test]
    fn layouts_list_tokenizer_and_detokenizer() {
        let tok = tok_weight_layout();
        assert!(tok
            .iter()
            .any(|(n, _, _)| n == "tokenizer.audio_acoustic_proj.weight"));
        assert!(tok
            .iter()
            .any(|(n, _, _)| n == "tokenizer.quantizer.project_in.weight"));
        let detok = detok_weight_layout();
        assert!(detok
            .iter()
            .any(|(n, _, _)| n == "detokenizer.proj_out.weight"));
        assert!(detok
            .iter()
            .any(|(n, _, _)| n == "tokenizer.quantizer.project_out.weight"));
    }

    #[test]
    fn smoke_tok_and_detok_zero_weights_finite() {
        let dir = scratch("smoke");
        write_fsq_bundle(&dir);
        let dit = GgufWeights::open(&dir.join("dit-fsq.gguf")).unwrap();
        check_tok_layout(&dit).unwrap();
        check_detok_layout(&dit).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let latents = vec![0f32; 5 * 64];
        let silence = vec![0f32; 64];
        let codes = tokenize(&ctx, &dit, &latents, &silence).unwrap();
        assert_eq!(codes.len(), 1);
        let rt = detokenize(&ctx, &dit, &codes).unwrap();
        assert_eq!(rt.len(), 5 * 64);
        assert!(rt.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_sessions_build_with_empty_types() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let types = HashMap::new();
        let mut tok = TokSession::build(&ctx, &types).unwrap();
        for (name, ne0, ne1) in tok_weight_layout() {
            let fill = if name.contains("norm") {
                vec![1.0f32; (ne0 * ne1) as usize]
            } else {
                vec![0.0f32; (ne0 * ne1) as usize]
            };
            tok.load_f32(&name, &fill).unwrap();
        }
        let out = tok.forward(&vec![0f32; 5 * 64]).unwrap();
        assert_eq!(out.len(), 6);
        let mut detok = DetokSession::build(&ctx, &types).unwrap();
        for (name, ne0, ne1) in detok_weight_layout() {
            let fill = if name.contains("norm") {
                vec![1.0f32; (ne0 * ne1) as usize]
            } else {
                vec![0.0f32; (ne0 * ne1) as usize]
            };
            detok.load_f32(&name, &fill).unwrap();
        }
        let frames = detok.forward(&[0.0; 6]).unwrap();
        assert_eq!(frames.len(), 5 * 64);
    }
}
