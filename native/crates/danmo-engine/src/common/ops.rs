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

/// Z-Image base static shift. Matches `FlowMatchEulerScheduler` when
/// `use_dynamic_shifting` is off: `linspace(1, 0, steps)`, then
/// `shift * t / (1 + (shift - 1) * t)`, then a terminal 0.
pub fn flow_match_shift_sigmas(steps: u32, shift: f32) -> Vec<f32> {
    let steps = steps.max(1);
    let mut sigmas = Vec::with_capacity(steps as usize + 1);
    for index in 0..steps {
        let t = if steps == 1 {
            1.0
        } else {
            1.0 - (index as f32) / (steps as f32 - 1.0)
        };
        sigmas.push(shift * t / (1.0 + (shift - 1.0) * t));
    }
    sigmas.push(0.0);
    sigmas
}

/// Deterministic N(0, 1) latent. Same seed and length always match.
pub fn normal_latent(seed: i64, len: usize) -> Vec<f32> {
    let mut state = (seed as u64).wrapping_add(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1 = (next_unit(&mut state) as f32).clamp(1e-6, 1.0);
        let u2 = next_unit(&mut state) as f32;
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        out.push(radius * theta.cos());
        if out.len() < len {
            out.push(radius * theta.sin());
        }
    }
    out
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::flow_match_shift_sigmas;

    #[test]
    fn static_shift_matches_the_z_image_base_schedule() {
        let sigmas = flow_match_shift_sigmas(4, 6.0);
        let expect = [1.0, 12.0 / 13.0, 0.75, 0.0, 0.0];
        assert_eq!(sigmas.len(), expect.len());
        for (got, want) in sigmas.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        assert_eq!(flow_match_shift_sigmas(1, 6.0), vec![1.0, 0.0]);
    }
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
