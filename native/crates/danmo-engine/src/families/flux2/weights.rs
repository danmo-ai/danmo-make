//! Flux2 tensor names. GGUF keys are these names; nothing is remapped.

#[derive(Debug, Clone, Copy)]
pub struct Flux2Config {
    pub in_channels: i64,
    pub inner_dim: i64,
    pub num_heads: i64,
    pub attn_head_dim: i64,
    pub num_layers: i64,
    pub num_single_layers: i64,
    pub joint_attention_dim: i64,
    pub freq_dim: i64,
    pub mlp_ratio: i64,
    pub requires_sigma_shift: bool,
    pub vae_scale: u32,
    pub rope_theta: f32,
}

impl Flux2Config {
    /// 4B Klein defaults. Head dim 128 uses four RoPE axes of 32.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            inner_dim: 3072,
            num_heads: 24,
            attn_head_dim: 128,
            num_layers: 8,
            num_single_layers: 24,
            joint_attention_dim: 12288,
            freq_dim: 256,
            mlp_ratio: 3,
            requires_sigma_shift: true,
            vae_scale: 16,
            rope_theta: 2000.0,
        }
    }

    /// One image token stream at dim 4, one double block and one single block.
    pub fn smoke() -> Self {
        Self::from_dim(4)
    }

    /// `heads` splits `dim`. Head dim is `dim / heads` and must divide evenly.
    pub fn with_heads(dim: i64, heads: i64) -> Self {
        let mut config = Self::from_dim(dim);
        config.num_heads = heads;
        config.attn_head_dim = dim / heads.max(1);
        config
    }

    pub fn from_dim(dim: i64) -> Self {
        Self {
            in_channels: dim,
            inner_dim: dim,
            num_heads: 1,
            attn_head_dim: dim,
            num_layers: 1,
            num_single_layers: 1,
            joint_attention_dim: dim,
            freq_dim: dim,
            mlp_ratio: 3,
            requires_sigma_shift: true,
            vae_scale: 16,
            rope_theta: 2000.0,
        }
    }

    pub fn hidden(&self) -> i64 {
        self.inner_dim * self.mlp_ratio
    }
}

/// `(name, ne0, ne1)` for every DiT weight the graph reads. ne0 is the ggml input axis.
pub fn weight_layout(config: Flux2Config) -> Vec<(String, i64, i64)> {
    let dim = config.inner_dim;
    let head = config.attn_head_dim;
    let hidden = config.hidden();
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push(
        "time_guidance_embed.linear_1.weight".into(),
        config.freq_dim,
        dim,
    );
    push("time_guidance_embed.linear_2.weight".into(), dim, dim);
    push("x_embedder.weight".into(), config.in_channels, dim);
    push(
        "context_embedder.weight".into(),
        config.joint_attention_dim,
        dim,
    );
    push(
        "double_stream_modulation_img.linear.weight".into(),
        dim,
        dim * 6,
    );
    push(
        "double_stream_modulation_txt.linear.weight".into(),
        dim,
        dim * 6,
    );
    push(
        "single_stream_modulation.linear.weight".into(),
        dim,
        dim * 3,
    );
    for i in 0..config.num_layers {
        let p = format!("transformer_blocks.{i}");
        for name in [
            "attn.to_q.weight",
            "attn.to_k.weight",
            "attn.to_v.weight",
            "attn.to_out.weight",
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
            "attn.to_add_out.weight",
        ] {
            push(format!("{p}.{name}"), dim, dim);
        }
        for name in [
            "attn.norm_q.weight",
            "attn.norm_k.weight",
            "attn.norm_added_q.weight",
            "attn.norm_added_k.weight",
        ] {
            push(format!("{p}.{name}"), head, 1);
        }
        push(format!("{p}.ff.linear_in.weight"), dim, hidden * 2);
        push(format!("{p}.ff.linear_out.weight"), hidden, dim);
        push(format!("{p}.ff_context.linear_in.weight"), dim, hidden * 2);
        push(format!("{p}.ff_context.linear_out.weight"), hidden, dim);
    }
    for i in 0..config.num_single_layers {
        let p = format!("single_transformer_blocks.{i}.attn");
        push(
            format!("{p}.to_qkv_mlp_proj.weight"),
            dim,
            dim * 3 + hidden * 2,
        );
        push(format!("{p}.norm_q.weight"), head, 1);
        push(format!("{p}.norm_k.weight"), head, 1);
        push(format!("{p}.to_out.weight"), dim + hidden, dim);
    }
    push("norm_out.linear.weight".into(), dim, dim * 2);
    push("proj_out.weight".into(), dim, config.in_channels);
    out
}

pub fn smoke_values(name: &str, ne0: i64, ne1: i64) -> Vec<f32> {
    let n = (ne0 as usize).saturating_mul(ne1 as usize);
    if name.contains(".norm_") {
        return vec![1.0; n];
    }
    let mut data = vec![0f32; n];
    let span = ne0.min(ne1) as usize;
    for i in 0..span {
        data[i + i * ne0 as usize] = 0.05;
    }
    data
}
