//! Shared schedule. CFG and Euler run inside the ggml graph (`runtime/extend`).

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
