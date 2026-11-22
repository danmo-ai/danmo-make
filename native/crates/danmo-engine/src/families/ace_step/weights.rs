//! ACE-Step 1.5 DiT weight table. Names match acestep.cpp `dit.h`.
//!
//! Product path is XL (`hidden=2560`, 32 layers). The 2B shape
//! (`hidden=2048`, 24 layers) is refused. Timestep MLP width is 256
//! (cos-then-sin of `t * 1000`). `proj_in.1` / `proj_out.1` stay the
//! Conv1d GGUF ranks; acestep.cpp pre-permutes them at load time.

use crate::weights::{GgufWeights, WeightError};

/// Frequency table width into `time_embed.linear_1` (acestep.cpp / main).
pub const TIME_FREQ_DIM: i64 = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AceStepConfig {
    pub hidden: i64,
    pub intermediate: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub patch: i64,
    pub sliding_window: i64,
    /// Qwen3-Embedding width into `condition_embedder`.
    pub condition_dim: i64,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub is_turbo: bool,
}

impl AceStepConfig {
    /// XL SFT / Turbo from main `AceStepConfig` and the product cards.
    pub fn xl(is_turbo: bool) -> Self {
        Self {
            hidden: 2560,
            intermediate: 9728,
            layers: 32,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            in_channels: 192,
            out_channels: 64,
            patch: 2,
            sliding_window: 128,
            condition_dim: 2048,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            is_turbo,
        }
    }

    /// One layer, tiny ranks for CPU layout tests. Not a published card.
    pub fn smoke() -> Self {
        Self {
            hidden: 8,
            intermediate: 16,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            in_channels: 4,
            out_channels: 2,
            patch: 2,
            sliding_window: 4,
            condition_dim: 8,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            is_turbo: false,
        }
    }

    /// ACE-Step 1.5 2B counts. Product create refuses this shape.
    pub fn base_2b() -> Self {
        Self {
            hidden: 2048,
            intermediate: 8192,
            layers: 24,
            heads: 16,
            kv_heads: 8,
            head_dim: 128,
            in_channels: 192,
            out_channels: 64,
            patch: 2,
            sliding_window: 128,
            condition_dim: 2048,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            is_turbo: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NamedShape {
    pub name: String,
    pub ne: [i64; 4],
}

/// Required DiT tensors for the separate-projection GGUF layout.
/// Fused QKV / gate+up rows are optional at load time in acestep.cpp; this
/// table lists the unfused names the converter writes.
pub fn weight_layout(config: AceStepConfig) -> Vec<NamedShape> {
    let h = config.hidden;
    let ff = config.intermediate;
    let q = config.heads * config.head_dim;
    let kv = config.kv_heads * config.head_dim;
    let mut out = Vec::new();
    let push2 = |out: &mut Vec<NamedShape>, name: String, ne0: i64, ne1: i64| {
        out.push(NamedShape {
            name,
            ne: [ne0, ne1, 1, 1],
        });
    };
    let push1 = |out: &mut Vec<NamedShape>, name: String, ne0: i64| {
        out.push(NamedShape {
            name,
            ne: [ne0, 1, 1, 1],
        });
    };
    for stem in ["decoder.time_embed", "decoder.time_embed_r"] {
        push2(
            &mut out,
            format!("{stem}.linear_1.weight"),
            TIME_FREQ_DIM,
            h,
        );
        push1(&mut out, format!("{stem}.linear_1.bias"), h);
        push2(&mut out, format!("{stem}.linear_2.weight"), h, h);
        push1(&mut out, format!("{stem}.linear_2.bias"), h);
        push2(&mut out, format!("{stem}.time_proj.weight"), h, h * 6);
        push1(&mut out, format!("{stem}.time_proj.bias"), h * 6);
    }
    // ggml stores Conv1d as [P, in_ch, H] (acestep.cpp `dit_load_proj_in_w`).
    out.push(NamedShape {
        name: "decoder.proj_in.1.weight".into(),
        ne: [config.patch, config.in_channels, h, 1],
    });
    push1(&mut out, "decoder.proj_in.1.bias".into(), h);
    push2(
        &mut out,
        "decoder.condition_embedder.weight".into(),
        config.condition_dim,
        h,
    );
    push1(&mut out, "decoder.condition_embedder.bias".into(), h);
    for layer in 0..config.layers {
        let p = format!("decoder.layers.{layer}");
        push1(&mut out, format!("{p}.self_attn_norm.weight"), h);
        push2(&mut out, format!("{p}.self_attn.q_proj.weight"), h, q);
        push2(&mut out, format!("{p}.self_attn.k_proj.weight"), h, kv);
        push2(&mut out, format!("{p}.self_attn.v_proj.weight"), h, kv);
        push1(
            &mut out,
            format!("{p}.self_attn.q_norm.weight"),
            config.head_dim,
        );
        push1(
            &mut out,
            format!("{p}.self_attn.k_norm.weight"),
            config.head_dim,
        );
        push2(&mut out, format!("{p}.self_attn.o_proj.weight"), q, h);
        push1(&mut out, format!("{p}.cross_attn_norm.weight"), h);
        push2(&mut out, format!("{p}.cross_attn.q_proj.weight"), h, q);
        push2(&mut out, format!("{p}.cross_attn.k_proj.weight"), h, kv);
        push2(&mut out, format!("{p}.cross_attn.v_proj.weight"), h, kv);
        push1(
            &mut out,
            format!("{p}.cross_attn.q_norm.weight"),
            config.head_dim,
        );
        push1(
            &mut out,
            format!("{p}.cross_attn.k_norm.weight"),
            config.head_dim,
        );
        push2(&mut out, format!("{p}.cross_attn.o_proj.weight"), q, h);
        push1(&mut out, format!("{p}.mlp_norm.weight"), h);
        push2(&mut out, format!("{p}.mlp.gate_proj.weight"), h, ff);
        push2(&mut out, format!("{p}.mlp.up_proj.weight"), h, ff);
        push2(&mut out, format!("{p}.mlp.down_proj.weight"), ff, h);
        // ggml layout [hidden, 6] after load; stored file may be [1,6,hidden].
        push2(&mut out, format!("{p}.scale_shift_table"), h, 6);
    }
    push1(&mut out, "decoder.norm_out.weight".into(), h);
    push2(&mut out, "decoder.scale_shift_table".into(), h, 2);
    out.push(NamedShape {
        name: "decoder.proj_out.1.weight".into(),
        ne: [config.patch, config.out_channels, h, 1],
    });
    push1(
        &mut out,
        "decoder.proj_out.1.bias".into(),
        config.out_channels,
    );
    out
}

pub fn layout_matches(weights: &GgufWeights, config: AceStepConfig) -> bool {
    for row in weight_layout(config) {
        let Ok(got) = weights.shape(&row.name) else {
            return false;
        };
        if got != row.ne {
            return false;
        }
    }
    true
}

pub fn config_from_weights(weights: &GgufWeights) -> Result<AceStepConfig, WeightError> {
    // SFT and Turbo share the DiT tensor table; `is_turbo` is a sampler policy.
    for candidate in [AceStepConfig::xl(false), AceStepConfig::smoke()] {
        if layout_matches(weights, candidate) {
            return Ok(candidate);
        }
    }
    if layout_matches(weights, AceStepConfig::base_2b()) {
        return Err(WeightError::NotBuilt(
            "ACE-Step 2B DiT is not the product XL path (need hidden 2560 and 32 layers)".into(),
        ));
    }
    Err(WeightError::NotBuilt(
        "ACE-Step DiT layout is not the published XL or smoke tower".into(),
    ))
}

pub fn check_layout(weights: &GgufWeights, config: AceStepConfig) -> Result<(), WeightError> {
    for row in weight_layout(config) {
        let got = weights.shape(&row.name)?;
        if got != row.ne {
            return Err(WeightError::Shape {
                name: row.name,
                expected: row.ne.to_vec(),
                got: got.to_vec(),
            });
        }
    }
    Ok(())
}
