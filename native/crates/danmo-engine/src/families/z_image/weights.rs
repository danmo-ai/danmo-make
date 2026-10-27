//! GGUF names are the unsloth / SD.cpp names. Nothing here renames a key.

#[cfg(test)]
use super::geom::ffn_hidden;
use super::geom::{axes_for_head, FREQ_DIM, MID_DIM, PATCH, ROPE_THETA, T_SCALE, VAE_SCALE};
use crate::weights::{GgufWeights, WeightError};

#[derive(Debug, Clone, Copy)]
pub struct ZImageConfig {
    pub dim: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub axes: [i64; 3],
    pub layers: i64,
    pub refiners: i64,
    pub in_channels: i64,
    pub cap_dim: i64,
    pub ffn: i64,
    pub mid: i64,
    pub freq: i64,
    pub adaln: i64,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub vae_scale: i64,
    pub patch: i64,
    pub requires_sigma_shift: bool,
}

impl ZImageConfig {
    pub fn token_dim(self) -> i64 {
        self.patch * self.patch * self.in_channels
    }

    /// Smoke graph: one refiner pair, one main layer, head 8.
    #[cfg(test)]
    pub fn smoke() -> Self {
        let dim = 8;
        let head = 8;
        Self {
            dim,
            heads: 1,
            head_dim: head,
            axes: axes_for_head(head).expect("smoke head"),
            layers: 1,
            refiners: 1,
            in_channels: 4,
            cap_dim: 8,
            ffn: ffn_hidden(dim),
            mid: MID_DIM,
            freq: FREQ_DIM,
            adaln: dim.min(256),
            rope_theta: ROPE_THETA,
            t_scale: T_SCALE,
            vae_scale: VAE_SCALE,
            patch: PATCH,
            requires_sigma_shift: true,
        }
    }
}

pub fn weight_layout(config: ZImageConfig) -> Vec<(String, i64, i64)> {
    let mut names = Vec::new();
    let token = config.token_dim();
    names.push(("x_embedder.weight".into(), token, config.dim));
    names.push(("x_embedder.bias".into(), config.dim, 1));
    names.push(("t_embedder.mlp.0.weight".into(), config.freq, config.mid));
    names.push(("t_embedder.mlp.0.bias".into(), config.mid, 1));
    names.push(("t_embedder.mlp.2.weight".into(), config.mid, config.adaln));
    names.push(("t_embedder.mlp.2.bias".into(), config.adaln, 1));
    names.push(("cap_embedder.0.weight".into(), config.cap_dim, 1));
    names.push(("cap_embedder.1.weight".into(), config.cap_dim, config.dim));
    names.push(("cap_embedder.1.bias".into(), config.dim, 1));
    names.push(("x_pad_token".into(), config.dim, 1));
    names.push(("cap_pad_token".into(), config.dim, 1));
    for i in 0..config.refiners {
        push_block(&mut names, &format!("noise_refiner.{i}"), config, true);
        push_block(&mut names, &format!("context_refiner.{i}"), config, false);
    }
    for i in 0..config.layers {
        push_block(&mut names, &format!("layers.{i}"), config, true);
    }
    names.push(("final_layer.linear.weight".into(), config.dim, token));
    names.push(("final_layer.linear.bias".into(), token, 1));
    names.push((
        "final_layer.adaLN_modulation.1.weight".into(),
        config.adaln,
        config.dim,
    ));
    names.push(("final_layer.adaLN_modulation.1.bias".into(), config.dim, 1));
    names
}

fn push_block(
    names: &mut Vec<(String, i64, i64)>,
    prefix: &str,
    config: ZImageConfig,
    modulated: bool,
) {
    let dim = config.dim;
    let head = config.head_dim;
    names.push((format!("{prefix}.attention.qkv.weight"), dim, dim * 3));
    names.push((format!("{prefix}.attention.out.weight"), dim, dim));
    names.push((format!("{prefix}.attention.q_norm.weight"), head, 1));
    names.push((format!("{prefix}.attention.k_norm.weight"), head, 1));
    names.push((format!("{prefix}.feed_forward.w1.weight"), dim, config.ffn));
    names.push((format!("{prefix}.feed_forward.w2.weight"), config.ffn, dim));
    names.push((format!("{prefix}.feed_forward.w3.weight"), dim, config.ffn));
    for leaf in [
        "attention_norm1",
        "attention_norm2",
        "ffn_norm1",
        "ffn_norm2",
    ] {
        names.push((format!("{prefix}.{leaf}.weight"), dim, 1));
    }
    if modulated {
        names.push((
            format!("{prefix}.adaLN_modulation.0.weight"),
            config.adaln,
            dim * 4,
        ));
        names.push((format!("{prefix}.adaLN_modulation.0.bias"), dim * 4, 1));
    }
}

pub fn config_from_weights(weights: &GgufWeights) -> Result<ZImageConfig, WeightError> {
    let x = weights.shape("x_embedder.weight")?;
    let token_in = x[0];
    let dim = x[1];
    if token_in % (PATCH * PATCH) != 0 || dim < 1 {
        return Err(WeightError::Shape {
            name: "x_embedder.weight".into(),
            expected: vec![PATCH * PATCH, dim],
            got: vec![token_in, dim],
        });
    }
    let in_channels = token_in / (PATCH * PATCH);
    let head = weights.shape("layers.0.attention.q_norm.weight")?[0];
    if head < 1 || dim % head != 0 {
        return Err(WeightError::Shape {
            name: "layers.0.attention.q_norm.weight".into(),
            expected: vec![head, 1],
            got: vec![head, dim],
        });
    }
    let qkv = weights.shape("layers.0.attention.qkv.weight")?;
    if qkv[0] != dim || qkv[1] != dim * 3 {
        return Err(WeightError::NotBuilt(format!(
            "layers.0.attention.qkv.weight is {:?} ; the graph reads fused q,k,v [dim, 3*dim]",
            qkv
        )));
    }
    let cap_dim = weights.shape("cap_embedder.0.weight")?[0];
    let freq = weights.shape("t_embedder.mlp.0.weight")?;
    let proj = weights.shape("t_embedder.mlp.2.weight")?;
    if freq[0] != FREQ_DIM || freq[1] != MID_DIM || proj[0] != freq[1] {
        return Err(WeightError::NotBuilt(format!(
            "t_embedder is {:?} -> {:?} ; expected freq {FREQ_DIM}, mid {MID_DIM}",
            freq, proj
        )));
    }
    let adaln = proj[1];
    if adaln != dim.min(256) {
        return Err(WeightError::NotBuilt(format!(
            "t_embedder output {adaln} is not min(dim, 256) = {}",
            dim.min(256)
        )));
    }
    let ffn = weights.shape("layers.0.feed_forward.w1.weight")?[1];
    let layers = count_prefix(weights, "layers");
    let noise = count_prefix(weights, "noise_refiner");
    let context = count_prefix(weights, "context_refiner");
    if layers < 1 || noise < 1 || noise != context {
        return Err(WeightError::NotBuilt(format!(
            "z-image has {layers} layers, {noise} noise refiners, {context} context refiners"
        )));
    }
    let axes = axes_for_head(head).map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    let final_out = weights.shape("final_layer.linear.weight")?;
    if final_out[0] != dim || final_out[1] != token_in {
        return Err(WeightError::Shape {
            name: "final_layer.linear.weight".into(),
            expected: vec![dim, token_in],
            got: vec![final_out[0], final_out[1]],
        });
    }
    Ok(ZImageConfig {
        dim,
        heads: dim / head,
        head_dim: head,
        axes,
        layers,
        refiners: noise,
        in_channels,
        cap_dim,
        ffn,
        mid: freq[1],
        freq: freq[0],
        adaln,
        rope_theta: ROPE_THETA,
        t_scale: T_SCALE,
        vae_scale: VAE_SCALE,
        patch: PATCH,
        requires_sigma_shift: true,
    })
}

fn count_prefix(weights: &GgufWeights, prefix: &str) -> i64 {
    let mut n = 0i64;
    loop {
        let name = format!("{prefix}.{n}.attention.qkv.weight");
        if !weights.contains(&name) {
            break;
        }
        n += 1;
    }
    n
}

pub fn vector_param(name: &str) -> bool {
    name.contains("norm")
        || name.ends_with("bias")
        || name.ends_with("pad_token")
        || name == "cap_embedder.0.weight"
}

#[cfg(test)]
pub fn smoke_values(name: &str, ne0: i64, ne1: i64) -> Vec<f32> {
    let n = (ne0 * ne1) as usize;
    if name.contains("norm") || name == "cap_embedder.0.weight" {
        return vec![1.0; n];
    }
    if name.ends_with("bias") || name.ends_with("pad_token") {
        return vec![0.01; n];
    }
    (0..n).map(|i| (((i % 5) as f32) - 2.0) * 0.01).collect()
}
