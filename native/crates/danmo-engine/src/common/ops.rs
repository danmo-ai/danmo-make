//! Shared ops used by every Shape A denoise loop. Math lives here; pipelines only call it.

use crate::runtime::{GgmlContext, GraphBuilder, ReusedGraph, RuntimeError};
use danmo_ggml_sys::ggml_tensor;

pub const RMS_EPS: f32 = 1e-5;

/// y = W x, with W stored ggml-style `[in, out]` (ne0 = in, ne1 = out) and x `[in, seq]`.
pub fn linear(b: &GraphBuilder<'_>, x: *mut ggml_tensor, w: *mut ggml_tensor) -> *mut ggml_tensor {
    b.mul_mat(w, x)
}

pub fn rms_norm(b: &GraphBuilder<'_>, x: *mut ggml_tensor) -> *mut ggml_tensor {
    b.rms_norm(x, RMS_EPS)
}

/// SwiGLU-style split is family-specific. This is the activation half used by Flux2 FFN.
pub fn silu(b: &GraphBuilder<'_>, x: *mut ggml_tensor) -> *mut ggml_tensor {
    b.silu(x)
}

/// Standard classifier-free guidance: uncond + scale * (cond - uncond).
pub fn cfg_combine(uncond: &[f32], cond: &[f32], guidance: f32) -> Vec<f32> {
    uncond
        .iter()
        .zip(cond.iter())
        .map(|(u, c)| u + guidance * (c - u))
        .collect()
}

/// Flow-match linear schedule from `LinearScheduler.set_timesteps`.
pub fn linear_sigmas(steps: u32, image_width: u32, image_height: u32, shift: bool) -> Vec<f32> {
    let steps = steps.max(1);
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| {
            let start = 1.0;
            let end = 1.0 / steps as f32;
            if steps == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / (steps as f32 - 1.0)
            }
        })
        .collect();
    if shift {
        let sigma_max_shift = 1.15;
        let sigma_base_shift = 0.5;
        let sigma_max_seq_len = 4096.0;
        let sigma_base_seq_len = 256.0;
        let m = (sigma_max_shift - sigma_base_shift) / (sigma_max_seq_len - sigma_base_seq_len);
        let b = sigma_base_shift - m * sigma_base_seq_len;
        let mu = m * (image_width as f32) * (image_height as f32) / 256.0 + b;
        let exp_mu = mu.exp();
        for s in &mut sigmas {
            *s = exp_mu / (exp_mu + (1.0 / *s - 1.0));
        }
    }
    sigmas.push(0.0);
    sigmas
}

/// Euler step on a flow trajectory: x <- x + (sigma_next - sigma) * v.
pub fn euler_step(latents: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    let dt = sigma_next - sigma;
    for (x, v) in latents.iter_mut().zip(velocity.iter()) {
        *x += dt * v;
    }
}

/// One reused graph: velocity = silu(rms_norm(x)) projected by W, plus a timestep bias.
/// Shape A families call this primitive; flux2 stacks it into the denoise loop.
pub fn build_velocity_block(ctx: &GgmlContext, dim: i64) -> Result<ReusedGraph, RuntimeError> {
    let mut b = ctx.builder(16 * 1024 * 1024)?;
    let x = b.f32_2d("latent", dim, 1, true);
    let w = b.f32_2d("velocity.weight", dim, dim, true);
    let temb = b.f32_2d("temb", dim, 1, true);
    let h = rms_norm(&b, x);
    let h = silu(&b, h);
    let h = linear(&b, h, w);
    let y = b.add(h, temb);
    b.finish(y)
}

/// Sinusoidal timestep projection used by Flux2TimestepEmbeddings (host side, fed as a graph input).
pub fn sinusoidal_timestep(t: f32, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    let half = dim / 2;
    for i in 0..half {
        let freq = (10000f32).powf(-(i as f32) / half.max(1) as f32);
        let arg = t * freq;
        out[i] = arg.sin();
        if i + half < dim {
            out[i + half] = arg.cos();
        }
    }
    out
}
