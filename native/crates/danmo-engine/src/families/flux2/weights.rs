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
    /// 4B Klein counts read from the published GGUF: 5 double blocks, 20 single
    /// blocks, joint width 7680. A loaded file still counts its own blocks.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            inner_dim: 3072,
            num_heads: 24,
            attn_head_dim: 128,
            num_layers: 5,
            num_single_layers: 20,
            joint_attention_dim: 7680,
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

/// `(name, ne0, ne1)` for every DiT weight the graph reads. Names are the
/// published GGUF names. ne0 is the ggml input axis. Q, K, and V are one matrix.
pub fn weight_layout(config: Flux2Config) -> Vec<(String, i64, i64)> {
    let dim = config.inner_dim;
    let head = config.attn_head_dim;
    let hidden = config.hidden();
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push("time_in.in_layer.weight".into(), config.freq_dim, dim);
    push("time_in.out_layer.weight".into(), dim, dim);
    push("img_in.weight".into(), config.in_channels, dim);
    push("txt_in.weight".into(), config.joint_attention_dim, dim);
    push(
        "double_stream_modulation_img.lin.weight".into(),
        dim,
        dim * 6,
    );
    push(
        "double_stream_modulation_txt.lin.weight".into(),
        dim,
        dim * 6,
    );
    push("single_stream_modulation.lin.weight".into(), dim, dim * 3);
    for i in 0..config.num_layers {
        let p = format!("double_blocks.{i}");
        push(format!("{p}.img_attn.qkv.weight"), dim, dim * 3);
        push(format!("{p}.img_attn.proj.weight"), dim, dim);
        push(format!("{p}.txt_attn.qkv.weight"), dim, dim * 3);
        push(format!("{p}.txt_attn.proj.weight"), dim, dim);
        for side in ["img_attn", "txt_attn"] {
            push(format!("{p}.{side}.norm.query_norm.scale"), head, 1);
            push(format!("{p}.{side}.norm.key_norm.scale"), head, 1);
        }
        push(format!("{p}.img_mlp.0.weight"), dim, hidden * 2);
        push(format!("{p}.img_mlp.2.weight"), hidden, dim);
        push(format!("{p}.txt_mlp.0.weight"), dim, hidden * 2);
        push(format!("{p}.txt_mlp.2.weight"), hidden, dim);
    }
    for i in 0..config.num_single_layers {
        let p = format!("single_blocks.{i}");
        push(format!("{p}.linear1.weight"), dim, dim * 3 + hidden * 2);
        push(format!("{p}.linear2.weight"), dim + hidden, dim);
        push(format!("{p}.norm.query_norm.scale"), head, 1);
        push(format!("{p}.norm.key_norm.scale"), head, 1);
    }
    push("final_layer.adaLN_modulation.1.weight".into(), dim, dim * 2);
    push("final_layer.linear.weight".into(), dim, config.in_channels);
    out
}

pub(super) fn stored_f32(name: &str) -> bool {
    name.ends_with(".scale") || name.contains(".norm_")
}

pub fn smoke_values(name: &str, ne0: i64, ne1: i64) -> Vec<f32> {
    let n = (ne0 as usize).saturating_mul(ne1 as usize);
    if stored_f32(name) {
        return vec![1.0; n];
    }
    let mut data = vec![0f32; n];
    paint_diag(&mut data, ne0, ne1, 0, ne0.min(ne1) as usize, 0.05);
    data
}

/// Smoke diagonals, plus a joint-stream gate and a text-to-image path.
///
/// The plain diagonal fills only the shift slice of AdaLN, so the attention
/// gate stays zero and the prompt never reaches the latent.
pub fn chain_dit_values(name: &str, ne0: i64, ne1: i64) -> Vec<f32> {
    let mut data = smoke_values(name, ne0, ne1);
    if name.starts_with("time_in.") {
        paint_diag(&mut data, ne0, ne1, 0, ne0.min(ne1) as usize, 2.0);
    }
    if name == "double_stream_modulation_img.lin.weight" {
        let dim = ne0 as usize;
        paint_diag(&mut data, ne0, ne1, dim * 2, dim, 1.0);
    }
    if name.ends_with("txt_attn.qkv.weight") {
        let dim = ne0 as usize;
        paint_diag(&mut data, ne0, ne1, dim * 2, dim, 1.0);
    }
    if name == "img_in.weight"
        || name == "txt_in.weight"
        || name == "final_layer.linear.weight"
        || name.ends_with("img_attn.proj.weight")
    {
        paint_diag(&mut data, ne0, ne1, 0, ne0.min(ne1) as usize, 1.0);
    }
    data
}

fn paint_diag(data: &mut [f32], ne0: i64, ne1: i64, out_at: usize, count: usize, gain: f32) {
    let ne0 = ne0 as usize;
    let ne1 = ne1 as usize;
    for i in 0..count {
        let out = out_at + i;
        if i < ne0 && out < ne1 {
            data[i + out * ne0] = gain;
        }
    }
}
