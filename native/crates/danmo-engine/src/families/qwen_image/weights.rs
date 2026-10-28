//! Bare diffusers names. The file name is the graph parameter name.
//!
//! Unsloth Qwen-Image-2512 stores these names without `model.diffusion_model.`.
//! SD.cpp adds that prefix for its own blocks. This graph does not.
//! Main's `remap_qwen_transformer_weights` maps these names onto MLX
//! `mlp_in` / `img_mod_linear`. Those MLX names are not read.

use crate::runtime::RuntimeError;
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};

pub const LN_EPS: f32 = 1e-6;
pub const PUBLISHED_PATCH: i64 = 2;

#[derive(Debug, Clone, Copy)]
pub struct QwenImageConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub inner_dim: i64,
    pub num_heads: i64,
    pub attn_head_dim: i64,
    pub axes: [i64; 3],
    pub num_layers: i64,
    pub joint_attention_dim: i64,
    pub freq_dim: i64,
    pub patch: i64,
    pub rope_theta: f32,
    pub vae_scale: u32,
}

impl QwenImageConfig {
    /// Published `QwenImageTransformer2DModel` counts from main `QwenImageConfig`
    /// and SD.cpp `QwenImageParams`.
    pub fn published() -> Self {
        Self {
            in_channels: 64,
            out_channels: 16,
            inner_dim: 3072,
            num_heads: 24,
            attn_head_dim: 128,
            axes: [16, 56, 56],
            num_layers: 60,
            joint_attention_dim: 3584,
            freq_dim: 256,
            patch: PUBLISHED_PATCH,
            rope_theta: 10000.0,
            vae_scale: 16,
        }
    }

    /// One block, head 8, patch 1 so `proj_out` is square.
    pub fn smoke() -> Self {
        Self {
            in_channels: 8,
            out_channels: 8,
            inner_dim: 8,
            num_heads: 1,
            attn_head_dim: 8,
            axes: [2, 2, 4],
            num_layers: 1,
            joint_attention_dim: 8,
            freq_dim: 8,
            patch: 1,
            rope_theta: 10000.0,
            vae_scale: 16,
        }
    }

    pub fn proj_out_dim(self) -> i64 {
        self.patch * self.patch * self.out_channels
    }

    pub fn mlp_hidden(self) -> i64 {
        self.inner_dim * 4
    }
}

/// Head 128 is the published split. Head 8 is the smoke graph.
pub fn axes_for_head(head: i64) -> Result<[i64; 3], RuntimeError> {
    match head {
        128 => Ok([16, 56, 56]),
        8 => Ok([2, 2, 4]),
        other => Err(RuntimeError::Message(format!(
            "qwen-image rope axes are 16+56+56 for head 128; head {other} has no axis table"
        ))),
    }
}

/// `(name, ne0, ne1)`. ne0 is the ggml input axis. Biases and RMS weights are `[n, 1]`.
pub fn weight_layout(config: QwenImageConfig) -> Vec<(String, i64, i64)> {
    let dim = config.inner_dim;
    let head = config.attn_head_dim;
    let hidden = config.mlp_hidden();
    let joint = config.joint_attention_dim;
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push(
        "time_text_embed.timestep_embedder.linear_1.weight".into(),
        config.freq_dim,
        dim,
    );
    push(
        "time_text_embed.timestep_embedder.linear_1.bias".into(),
        dim,
        1,
    );
    push(
        "time_text_embed.timestep_embedder.linear_2.weight".into(),
        dim,
        dim,
    );
    push(
        "time_text_embed.timestep_embedder.linear_2.bias".into(),
        dim,
        1,
    );
    push("txt_norm.weight".into(), joint, 1);
    push("img_in.weight".into(), config.in_channels, dim);
    push("img_in.bias".into(), dim, 1);
    push("txt_in.weight".into(), joint, dim);
    push("txt_in.bias".into(), dim, 1);
    for i in 0..config.num_layers {
        let p = format!("transformer_blocks.{i}");
        push(format!("{p}.img_mod.1.weight"), dim, dim * 6);
        push(format!("{p}.img_mod.1.bias"), dim * 6, 1);
        push(format!("{p}.txt_mod.1.weight"), dim, dim * 6);
        push(format!("{p}.txt_mod.1.bias"), dim * 6, 1);
        for stem in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_out.0",
            "attn.to_add_out",
        ] {
            push(format!("{p}.{stem}.weight"), dim, dim);
            push(format!("{p}.{stem}.bias"), dim, 1);
        }
        for stem in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            push(format!("{p}.attn.{stem}.weight"), head, 1);
        }
        for side in ["img_mlp", "txt_mlp"] {
            push(format!("{p}.{side}.net.0.proj.weight"), dim, hidden);
            push(format!("{p}.{side}.net.0.proj.bias"), hidden, 1);
            push(format!("{p}.{side}.net.2.weight"), hidden, dim);
            push(format!("{p}.{side}.net.2.bias"), dim, 1);
        }
    }
    push("norm_out.linear.weight".into(), dim, dim * 2);
    push("norm_out.linear.bias".into(), dim * 2, 1);
    push("proj_out.weight".into(), dim, config.proj_out_dim());
    push("proj_out.bias".into(), config.proj_out_dim(), 1);
    out
}

pub fn stored_f32(name: &str) -> bool {
    name.ends_with(".bias")
        || name.ends_with("txt_norm.weight")
        || name.ends_with("norm_q.weight")
        || name.ends_with("norm_k.weight")
        || name.ends_with("norm_added_q.weight")
        || name.ends_with("norm_added_k.weight")
}

/// Patch stays 2. A file whose `proj_out` width is not `4 * out_channels` fails.
pub fn config_from_weights(weights: &GgufWeights) -> Result<QwenImageConfig, WeightError> {
    let img = weights.shape("img_in.weight")?;
    let context = weights.shape("txt_in.weight")?;
    let freq = weights.shape("time_text_embed.timestep_embedder.linear_1.weight")?;
    let norm = weights.shape("transformer_blocks.0.attn.norm_q.weight")?;
    let proj = weights.shape("proj_out.weight")?;
    let in_channels = img[0];
    let dim = img[1];
    if context[1] != dim || freq[1] != dim || proj[0] != dim {
        return Err(WeightError::Shape {
            name: "img_in.weight".into(),
            expected: vec![in_channels, dim],
            got: vec![context[0], context[1], freq[0], freq[1], proj[0], proj[1]],
        });
    }
    let head = norm[0];
    if head < 1 || norm[1] != 1 || dim % head != 0 {
        return Err(WeightError::Shape {
            name: "transformer_blocks.0.attn.norm_q.weight".into(),
            expected: vec![head.max(1), 1],
            got: vec![norm[0], norm[1], dim],
        });
    }
    let axes = axes_for_head(head).map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    let patch = PUBLISHED_PATCH;
    if proj[1] % (patch * patch) != 0 {
        return Err(WeightError::Shape {
            name: "proj_out.weight".into(),
            expected: vec![dim, patch * patch],
            got: vec![proj[0], proj[1]],
        });
    }
    let layers = count_blocks(weights);
    if layers < 1 {
        return Err(WeightError::MissingTensor {
            path: "transformer".into(),
            name: "transformer_blocks.0.img_mod.1.weight".into(),
        });
    }
    Ok(QwenImageConfig {
        in_channels,
        out_channels: proj[1] / (patch * patch),
        inner_dim: dim,
        num_heads: dim / head,
        attn_head_dim: head,
        axes,
        num_layers: layers,
        joint_attention_dim: context[0],
        freq_dim: freq[0],
        patch,
        rope_theta: 10000.0,
        vae_scale: 16,
    })
}

pub fn check_layout(weights: &GgufWeights, config: QwenImageConfig) -> Result<(), WeightError> {
    for (name, ne0, ne1) in weight_layout(config) {
        let ggml_type = weights.ggml_type_of(&name)?;
        check_stored_type(&name, ggml_type, ne0)?;
        let got = weights.shape(&name)?;
        if got[0] != ne0 || got[1] != ne1 || got[2] != 1 || got[3] != 1 {
            return Err(WeightError::Shape {
                name,
                expected: vec![ne0, ne1, 1, 1],
                got: got.to_vec(),
            });
        }
    }
    Ok(())
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if stored_f32(name) {
        if ggml_type != crate::runtime::TYPE_F32 {
            return Err(WeightError::NotBuilt(format!(
                "{name} is ggml type {ggml_type}. Norm and bias weights stay F32"
            )));
        }
        return Ok(());
    }
    if !matmul_type(ggml_type) {
        return Err(WeightError::Type {
            name: name.to_string(),
            ggml_type,
        });
    }
    let block = type_block(ggml_type);
    if block > 1 && (ne0 % block != 0) {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}

fn count_blocks(weights: &GgufWeights) -> i64 {
    let mut n = 0i64;
    loop {
        let name = format!("transformer_blocks.{n}.img_mod.1.weight");
        if !weights.contains(&name) {
            break;
        }
        n += 1;
    }
    n
}
