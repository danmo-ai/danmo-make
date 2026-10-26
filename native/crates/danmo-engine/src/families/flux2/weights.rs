//! Flux2 shape. Weight files are GGUF; this module does not remap keys.

#[derive(Debug, Clone, Copy)]
pub struct Flux2Config {
    pub in_channels: i64,
    pub inner_dim: i64,
    pub num_heads: i64,
    pub attn_head_dim: i64,
    pub num_layers: i64,
    pub num_single_layers: i64,
    pub joint_attention_dim: i64,
    pub requires_sigma_shift: bool,
    pub vae_scale: u32,
}

impl Flux2Config {
    /// 4B Klein defaults from the registry family config.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            inner_dim: 3072,
            num_heads: 24,
            attn_head_dim: 128,
            num_layers: 8,
            num_single_layers: 24,
            joint_attention_dim: 12288,
            requires_sigma_shift: true,
            vae_scale: 16,
        }
    }

    /// Tiny config so the denoise graph can execute on CPU with a smoke GGUF.
    pub fn smoke() -> Self {
        Self {
            in_channels: 4,
            inner_dim: 4,
            num_heads: 1,
            attn_head_dim: 4,
            num_layers: 1,
            num_single_layers: 1,
            joint_attention_dim: 4,
            requires_sigma_shift: true,
            vae_scale: 16,
        }
    }
}
