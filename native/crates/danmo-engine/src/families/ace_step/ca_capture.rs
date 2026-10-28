//! DiT cross-attention score capture for lyric DTW (MLX `capture_cross_attention`).
//!
//! Early-exits after `max(layers_config)`, emits softmax(QKᵀ) for every head of
//! layers `0..=max`, then host-slices configured heads → `[n_sel, enc_seq, seq]`.

use super::graph::{graph_weight_layout, nn, wt};
use super::transformer::{pack_input, remap_proj_in, remap_proj_out};
use super::weights::{AceStepConfig, TIME_FREQ_DIM};
use crate::common::ops::sinusoidal_timestep;
use crate::runtime::{
    flash_attend, shape, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::{BTreeMap, HashMap};

/// Published XL SFT/Turbo `lyric_alignment_layers_config` (HF acestep-v15-xl-sft).
pub fn xl_layers_config() -> BTreeMap<i64, Vec<i64>> {
    BTreeMap::from([
        (3, vec![18, 27]),
        (4, vec![22]),
        (5, vec![5, 6, 7]),
        (6, vec![2, 12, 13]),
        (7, vec![20, 21]),
    ])
}

pub fn smoke_layers_config() -> BTreeMap<i64, Vec<i64>> {
    BTreeMap::from([(0, vec![0])])
}

pub fn default_layers_config(config: AceStepConfig) -> Option<BTreeMap<i64, Vec<i64>>> {
    if config == AceStepConfig::xl(false) || config == AceStepConfig::xl(true) {
        Some(xl_layers_config())
    } else if config == AceStepConfig::smoke() {
        Some(smoke_layers_config())
    } else {
        None
    }
}

pub fn parse_layers_config(raw: &serde_json::Value) -> Option<BTreeMap<i64, Vec<i64>>> {
    let obj = raw.as_object()?;
    let mut out = BTreeMap::new();
    for (k, v) in obj {
        let layer: i64 = k.parse().ok()?;
        let heads = v
            .as_array()?
            .iter()
            .map(|h| h.as_i64().or_else(|| h.as_u64().map(|u| u as i64)))
            .collect::<Option<Vec<_>>>()?;
        if !heads.is_empty() {
            out.insert(layer, heads);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub struct CaCapture {
    graph: ReusedGraph,
    config: AceStepConfig,
    frames: i64,
    enc_seq: i64,
    seq: i64,
    /// Layers `0..=max` that emitted scores (contiguous).
    captured_layers: Vec<i64>,
}

impl CaCapture {
    pub fn build(
        ctx: &GgmlContext,
        config: AceStepConfig,
        frames: i64,
        enc_seq: i64,
        layers_config: &BTreeMap<i64, Vec<i64>>,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        if layers_config.is_empty() {
            return Err(RuntimeError::Message(
                "ACE-Step CA capture needs at least one (layer, head)".into(),
            ));
        }
        let last = *layers_config.keys().max().unwrap();
        if last < 0 || last >= config.layers {
            return Err(RuntimeError::Message(format!(
                "ACE-Step lyric alignment layer {last} out of range (DiT has {} layers)",
                config.layers
            )));
        }
        for (layer, heads) in layers_config {
            for &h in heads {
                if h < 0 || h >= config.heads {
                    return Err(RuntimeError::Message(format!(
                        "ACE-Step lyric alignment head {h} out of range on layer {layer}"
                    )));
                }
            }
        }
        let mem = if config == AceStepConfig::smoke() {
            64 * 1024 * 1024
        } else {
            2 * 1024 * 1024 * 1024
        };
        let mut builder = ctx.builder(mem)?;
        let (output, seq, captured_layers) =
            build_ca_capture(&mut builder, config, frames, enc_seq, last, types)?;
        let graph = builder.finish(output)?;
        Ok(Self {
            graph,
            config,
            frames,
            enc_seq,
            seq,
            captured_layers,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn forward(
        &mut self,
        xt: &[f32],
        context: &[f32],
        enc_hidden: &[f32],
        t: f32,
    ) -> Result<Vec<f32>, RuntimeError> {
        let positions: Vec<i32> = (0..self.seq as i32).collect();
        self.graph.set_i32("positions", &positions)?;
        self.graph.set_f32("enc_hidden", enc_hidden)?;
        self.graph.set_f32(
            "temb_freq_t",
            &sinusoidal_timestep(t * 1000.0, TIME_FREQ_DIM as usize),
        )?;
        // Capture pass uses t_r = t (MLX single timestep).
        self.graph.set_f32(
            "temb_freq_r",
            &sinusoidal_timestep(0.0, TIME_FREQ_DIM as usize),
        )?;
        let packed = pack_input(context, xt, self.config, self.frames)?;
        self.graph.set_f32("input_latents", &packed)?;
        self.graph.compute()
    }

    /// Raw graph output → selected heads `[n_sel, enc_seq, seq]`.
    pub fn select_heads(
        &self,
        raw: &[f32],
        layers_config: &BTreeMap<i64, Vec<i64>>,
    ) -> Result<Vec<f32>, RuntimeError> {
        let nh = self.config.heads as usize;
        let enc = self.enc_seq as usize;
        let seq = self.seq as usize;
        let per_head = enc * seq;
        let n_layers = self.captured_layers.len();
        let expect = n_layers * nh * per_head;
        if raw.len() != expect {
            return Err(RuntimeError::Message(format!(
                "ACE-Step CA raw length {} != {expect}",
                raw.len()
            )));
        }
        // Graph stores [enc*seq, n_layers*heads] → column lh has head (lh % nh) of layer lh/nh.
        let mut out = Vec::new();
        for (li, &layer) in self.captured_layers.iter().enumerate() {
            let Some(heads) = layers_config.get(&layer) else {
                continue;
            };
            for &h in heads {
                let col = li * nh + h as usize;
                let off = col * per_head;
                out.extend_from_slice(&raw[off..off + per_head]);
            }
        }
        if out.is_empty() {
            return Err(RuntimeError::Message(
                "ACE-Step CA select_heads produced no heads".into(),
            ));
        }
        Ok(out)
    }

    pub fn n_selected_heads(&self, layers_config: &BTreeMap<i64, Vec<i64>>) -> usize {
        layers_config
            .iter()
            .filter(|(l, _)| self.captured_layers.contains(l))
            .map(|(_, h)| h.len())
            .sum()
    }

    pub fn seq(&self) -> i64 {
        self.seq
    }

    pub fn enc_seq(&self) -> i64 {
        self.enc_seq
    }
}

pub fn load_capture_weights(
    capture: &mut CaCapture,
    weights: &GgufWeights,
    config: AceStepConfig,
) -> Result<(), WeightError> {
    for (name, ne0, ne1) in graph_weight_layout(config) {
        if name == "decoder.proj_in.1.weight" {
            let raw = weights.f32_exact(
                &name,
                &[config.patch, config.in_channels, config.hidden],
            )?;
            let remapped = remap_proj_in(&raw, config.patch, config.in_channels, config.hidden);
            capture
                .load_f32(&name, &remapped)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            continue;
        }
        if name == "decoder.proj_out.1.weight" {
            // Capture early-exits before proj_out; still bind so the weight table matches.
            let raw = weights.f32_exact(
                &name,
                &[config.patch, config.out_channels, config.hidden],
            )?;
            let remapped = remap_proj_out(&raw, config.patch, config.out_channels, config.hidden);
            capture
                .load_f32(&name, &remapped)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            continue;
        }
        let data = weights.f32_exact(&name, &[ne0, ne1])?;
        capture
            .load_f32(&name, &data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    Ok(())
}

/// Re-noise latents at `t=1/steps`, capture selected CA heads.
pub fn capture_selected_attention(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    config: AceStepConfig,
    latents: &[f32],
    context: &[f32],
    enc_hidden: &[f32],
    frames: i64,
    enc_seq: i64,
    steps: u32,
    seed: i64,
    layers_config: &BTreeMap<i64, Vec<i64>>,
) -> Result<(Vec<f32>, usize /*n_heads*/), WeightError> {
    if steps == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step CA capture needs positive infer_steps".into(),
        ));
    }
    let mut types = HashMap::new();
    for (name, _, _) in graph_weight_layout(config) {
        if let Ok(t) = dit.ggml_type_of(&name) {
            types.insert(name, t);
        }
    }
    let mut capture = CaCapture::build(ctx, config, frames, enc_seq, layers_config, &types)
        .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    load_capture_weights(&mut capture, dit, config)?;

    let t_last = 1.0 / steps as f32;
    let noise = crate::common::ops::normal_latent(seed, latents.len());
    let mut xt = vec![0f32; latents.len()];
    for i in 0..latents.len() {
        xt[i] = t_last * noise[i] + (1.0 - t_last) * latents[i];
    }
    let raw = capture
        .forward(&xt, context, enc_hidden, t_last)
        .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    let selected = capture
        .select_heads(&raw, layers_config)
        .map_err(|e| WeightError::NotBuilt(e.to_string()))?;
    let n_sel = capture.n_selected_heads(layers_config);
    Ok((selected, n_sel))
}

fn build_ca_capture(
    b: &mut GraphBuilder,
    config: AceStepConfig,
    frames: i64,
    enc_seq: i64,
    last_layer: i64,
    types: &HashMap<String, u32>,
) -> Result<(Tensor, i64, Vec<i64>), RuntimeError> {
    if frames < 1 || frames % config.patch != 0 {
        return Err(RuntimeError::Message(format!(
            "ACE-Step frames {frames} must be a positive multiple of patch {}",
            config.patch
        )));
    }
    let seq = frames / config.patch;
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in graph_weight_layout(config) {
        let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
        let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    let input = nn(b.f32_2d("input_latents", config.in_channels, frames, true))?;
    let enc_hidden = nn(b.f32_2d("enc_hidden", config.condition_dim, enc_seq, true))?;
    let temb_freq_t = nn(b.f32_2d("temb_freq_t", TIME_FREQ_DIM, 1, true))?;
    let temb_freq_r = nn(b.f32_2d("temb_freq_r", TIME_FREQ_DIM, 1, true))?;
    let positions = nn(b.i32_1d("positions", seq, true))?;

    let (temb_t, tproj_t) = temb(b, &weights, "decoder.time_embed", temb_freq_t)?;
    let (temb_r, tproj_r) = temb(b, &weights, "decoder.time_embed_r", temb_freq_r)?;
    let _temb = nn(b.add(temb_t, temb_r))?;
    let tproj = nn(b.add(tproj_t, tproj_r))?;
    let patched = nn(b.reshape_2d(input, config.in_channels * config.patch, seq))?;
    let mut hidden = add_bias(
        b,
        nn(linear(b, patched, wt(&weights, "decoder.proj_in.1.weight")?))?,
        wt(&weights, "decoder.proj_in.1.bias")?,
    )?;
    let enc = add_bias(
        b,
        nn(linear(
            b,
            enc_hidden,
            wt(&weights, "decoder.condition_embedder.weight")?,
        ))?,
        wt(&weights, "decoder.condition_embedder.bias")?,
    )?;

    let mut score_layers: Vec<Tensor> = Vec::new();
    let mut captured_layers = Vec::new();
    for layer in 0..=last_layer {
        let (next, scores) = layer_with_scores(
            b, &weights, config, layer, hidden, tproj, enc, positions, seq, enc_seq,
        )?;
        hidden = next;
        score_layers.push(scores);
        captured_layers.push(layer);
    }

    let mut out = score_layers[0];
    for s in score_layers.into_iter().skip(1) {
        out = nn(b.concat(out, s, 2))?;
    }
    let n_layers = captured_layers.len() as i64;
    // [enc, seq, n_layers*heads] → [enc*seq, n_layers*heads]
    let flat = nn(b.reshape_2d(out, enc_seq * seq, n_layers * config.heads))?;
    Ok((flat, seq, captured_layers))
}

fn layer_with_scores(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    layer: i64,
    hidden: Tensor,
    tproj: Tensor,
    enc: Tensor,
    positions: Tensor,
    seq: i64,
    enc_seq: i64,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let h = config.hidden;
    let p = format!("decoder.layers.{layer}");
    let ss = nn(b.reshape_2d(wt(weights, &format!("{p}.scale_shift_table"))?, h * 6, 1))?;
    let adaln_pack = nn(b.add(ss, tproj))?;
    let shift_sa = chunk(b, adaln_pack, 0, h)?;
    let scale_sa = chunk(b, adaln_pack, h, h)?;
    let gate_sa = chunk(b, adaln_pack, h * 2, h)?;
    let shift_ffn = chunk(b, adaln_pack, h * 3, h)?;
    let scale_ffn = chunk(b, adaln_pack, h * 4, h)?;
    let gate_ffn = chunk(b, adaln_pack, h * 5, h)?;

    let norm_sa = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.self_attn_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let norm_sa = adaln(b, norm_sa, scale_sa, shift_sa)?;
    let sa = self_attn(b, weights, config, &p, norm_sa, positions, seq)?;
    let hidden = gate_add(b, hidden, sa, gate_sa)?;

    let norm_ca = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.cross_attn_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let scores = cross_attn_scores(b, weights, config, &p, norm_ca, enc, seq, enc_seq)?;
    let ca = cross_attn(b, weights, config, &p, norm_ca, enc, seq, enc_seq)?;
    let hidden = nn(b.add(hidden, ca))?;

    let norm_ffn = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.mlp_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let norm_ffn = adaln(b, norm_ffn, scale_ffn, shift_ffn)?;
    let ffn = mlp(b, weights, &p, norm_ffn)?;
    Ok((gate_add(b, hidden, ffn, gate_ffn)?, scores))
}

fn cross_attn_scores(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    prefix: &str,
    x: Tensor,
    enc: Tensor,
    seq: i64,
    enc_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let d = config.head_dim;
    let nh = config.heads;
    let nkv = config.kv_heads;
    let q = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.cross_attn.q_proj.weight"))?,
    ))?;
    let k = nn(linear(
        b,
        enc,
        wt(weights, &format!("{prefix}.cross_attn.k_proj.weight"))?,
    ))?;
    let q = nn(b.reshape_3d(q, d, nh, seq))?;
    let k = nn(b.reshape_3d(k, d, nkv, enc_seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.permute(q, 0, 2, 1, 3)))?;
    let k = nn(b.cont(b.permute(k, 0, 2, 1, 3)))?;
    let scale = 1.0 / (d as f32).sqrt();
    let scores = nn(b.soft_max_ext(b.mul_mat(k, q), std::ptr::null_mut(), scale))?;
    let sh = shape(scores);
    if sh[0] != enc_seq || sh[1] != seq {
        return Err(RuntimeError::Message(format!(
            "ACE-Step CA scores shape {:?}, expected [{enc_seq}, {seq}, {nh}]",
            &sh[..3.min(sh.len())]
        )));
    }
    Ok(scores)
}

fn temb(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    stem: &str,
    freq: Tensor,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let h = add_bias(
        b,
        nn(linear(
            b,
            freq,
            wt(weights, &format!("{stem}.linear_1.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.linear_1.bias"))?,
    )?;
    let temb = add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(h))?,
            wt(weights, &format!("{stem}.linear_2.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.linear_2.bias"))?,
    )?;
    let tproj = add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(temb))?,
            wt(weights, &format!("{stem}.time_proj.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.time_proj.bias"))?,
    )?;
    Ok((temb, tproj))
}

fn self_attn(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    prefix: &str,
    x: Tensor,
    positions: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let d = config.head_dim;
    let nh = config.heads;
    let nkv = config.kv_heads;
    let q = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
    ))?;
    let k = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
    ))?;
    let v = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
    ))?;
    let q = nn(b.reshape_3d(q, d, nh, seq))?;
    let k = nn(b.reshape_3d(k, d, nkv, seq))?;
    let v = nn(b.reshape_3d(v, d, nkv, seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, positions, d as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, positions, d as i32, config.rope_theta)))?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, nh * d, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
    ))
}

fn cross_attn(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    prefix: &str,
    x: Tensor,
    enc: Tensor,
    seq: i64,
    enc_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let d = config.head_dim;
    let nh = config.heads;
    let nkv = config.kv_heads;
    let q = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.cross_attn.q_proj.weight"))?,
    ))?;
    let k = nn(linear(
        b,
        enc,
        wt(weights, &format!("{prefix}.cross_attn.k_proj.weight"))?,
    ))?;
    let v = nn(linear(
        b,
        enc,
        wt(weights, &format!("{prefix}.cross_attn.v_proj.weight"))?,
    ))?;
    let q = nn(b.reshape_3d(q, d, nh, seq))?;
    let k = nn(b.reshape_3d(k, d, nkv, enc_seq))?;
    let v = nn(b.reshape_3d(v, d, nkv, enc_seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.k_norm.weight"))?,
    ))?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, nh * d, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.cross_attn.o_proj.weight"))?,
    ))
}

fn mlp(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.silu(nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.mlp.gate_proj.weight"))?,
    ))?))?;
    let up = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.mlp.up_proj.weight"))?,
    ))?;
    let hidden = nn(b.mul(gate, up))?;
    nn(linear(
        b,
        hidden,
        wt(weights, &format!("{prefix}.mlp.down_proj.weight"))?,
    ))
}

fn adaln(b: &GraphBuilder, norm: Tensor, scale: Tensor, shift: Tensor) -> Result<Tensor, RuntimeError> {
    let scale = nn(b.repeat(scale, norm))?;
    let shift = nn(b.repeat(shift, norm))?;
    let scaled = nn(b.add(norm, nn(b.mul(norm, scale))?))?;
    nn(b.add(scaled, shift))
}

fn gate_add(
    b: &GraphBuilder,
    residual: Tensor,
    delta: Tensor,
    gate: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.repeat(gate, delta))?;
    nn(b.add(residual, nn(b.mul(delta, gate))?))
}

fn chunk(b: &GraphBuilder, packed: Tensor, offset: i64, rows: i64) -> Result<Tensor, RuntimeError> {
    let dims = shape(packed);
    let total = dims[0];
    let view = b.view_2d(
        packed,
        rows,
        1,
        (total as usize) * std::mem::size_of::<f32>(),
        (offset as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
}

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor, eps: f32) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, eps))?, weight))
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.project(w, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::ace_step::graph::graph_weight_layout;
    use crate::runtime::BackendKind;

    #[test]
    fn xl_layers_config_matches_published() {
        let cfg = xl_layers_config();
        assert_eq!(cfg.get(&3).unwrap(), &vec![18, 27]);
        assert_eq!(cfg.get(&7).unwrap(), &vec![20, 21]);
    }

    #[test]
    fn smoke_capture_scores_are_finite() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let config = AceStepConfig::smoke();
        let frames = config.patch * 2;
        let enc_seq = 4i64;
        let layers = smoke_layers_config();
        let mut capture =
            CaCapture::build(&ctx, config, frames, enc_seq, &layers, &HashMap::new()).unwrap();
        for (name, ne0, ne1) in graph_weight_layout(config) {
            let n = (ne0 * ne1) as usize;
            let data = if name.contains("norm") || name.contains("scale_shift") {
                vec![1.0f32; n]
            } else if name.ends_with(".bias") {
                vec![0.0f32; n]
            } else {
                let mut v = vec![0.0f32; n];
                let dim = ne0.min(ne1) as usize;
                for i in 0..dim {
                    v[i * ne0 as usize + i] = 0.05;
                }
                v
            };
            capture.load_f32(&name, &data).unwrap();
        }
        let xt = vec![0.1f32; (config.out_channels * frames) as usize];
        let context = vec![0.0f32; ((config.in_channels - config.out_channels) * frames) as usize];
        let enc = vec![0.05f32; (config.condition_dim * enc_seq) as usize];
        let raw = capture.forward(&xt, &context, &enc, 0.25).unwrap();
        assert!(raw.iter().all(|v| v.is_finite()), "non-finite CA scores");
        let selected = capture.select_heads(&raw, &layers).unwrap();
        assert_eq!(selected.len(), (enc_seq * (frames / config.patch)) as usize);
        assert!(selected.iter().all(|v| v.is_finite()));
        // Softmax rows should be non-negative.
        assert!(selected.iter().all(|&v| v >= 0.0));
    }
}
