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
    use super::{empirical_mu, empirical_mu_sigmas, flow_match_shift_sigmas, flux_timestep_embed};

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

    #[test]
    fn empirical_mu_matches_flow_match_euler() {
        let mu = empirical_mu(16, 4);
        assert!((mu - 1.9452164).abs() < 1e-5, "{mu}");
        let small = empirical_mu_sigmas(4, 16);
        let expect = [1.0, 0.95451534, 0.8749241, 0.6998543, 0.0];
        assert_eq!(small.len(), expect.len());
        for (got, want) in small.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }

        let wide = empirical_mu(5000, 2);
        assert!((wide - 1.3030167).abs() < 1e-5, "{wide}");
        let long = empirical_mu_sigmas(2, 5000);
        let expect = [1.0, 0.78634226, 0.0];
        assert_eq!(long.len(), expect.len());
        for (got, want) in long.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn flux_timestep_is_cosine_then_sine_at_sigma_times_1000() {
        let emb = flux_timestep_embed(0.5, 4);
        let expect = [-0.88384926, 0.2836622, -0.4677718, -0.9589243];
        assert_eq!(emb.len(), expect.len());
        for (got, want) in emb.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }
}

/// `FlowMatchEulerScheduler._compute_empirical_mu`.
/// `image_seq_len` is `(width / vae_scale) * (height / vae_scale)`.
pub fn empirical_mu(image_seq_len: u32, steps: u32) -> f32 {
    let seq = f64::from(image_seq_len);
    let steps = f64::from(steps.max(1));
    let a1 = 8.73809524e-05;
    let b1 = 1.89833333;
    let a2 = 0.00016927;
    let b2 = 0.45666666;
    let mu = if seq > 4300.0 {
        a2 * seq + b2
    } else {
        let m_200 = a2 * seq + b2;
        let m_10 = a1 * seq + b1;
        let slope = (m_200 - m_10) / 190.0;
        let intercept = m_200 - 200.0 * slope;
        slope * steps + intercept
    };
    mu as f32
}

/// Empirical-mu flow schedule: `linspace(1, 1/steps, steps)`, shifted by
/// [`empirical_mu`], then a terminal 0.
pub fn empirical_mu_sigmas(steps: u32, image_seq_len: u32) -> Vec<f32> {
    let exp_mu = f64::from(empirical_mu(image_seq_len, steps)).exp();
    let mut sigmas = linear_sigmas(steps, 0, 0, false);
    let last = sigmas.len().saturating_sub(1);
    for sigma in &mut sigmas[..last] {
        let t = f64::from(*sigma);
        *sigma = (exp_mu / (exp_mu + (1.0 / t - 1.0))) as f32;
    }
    sigmas
}

/// Fail-loud summary when a host vector contains NaN or Inf.
pub fn non_finite(label: &str, data: &[f32]) -> Option<String> {
    let mut bad = 0usize;
    let mut finite = 0usize;
    let mut absmax = 0f32;
    for value in data {
        if value.is_finite() {
            finite += 1;
            absmax = absmax.max(value.abs());
        } else {
            bad += 1;
        }
    }
    if bad == 0 {
        return None;
    }
    Some(format!(
        "{label} has {bad} non-finite values out of {} ({finite} finite, abs max {absmax})",
        data.len()
    ))
}

/// Cosine then sine of `t * 10000^(-i/(dim/2))`. Shared by Flux2, Z-Image, and
/// Qwen-Image. Callers scale `t` (sigma, sigma*1000, or `(1-sigma)*t_scale`).
pub fn sinusoidal_timestep(t: f32, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    let half = dim / 2;
    if half == 0 {
        return out;
    }
    for i in 0..half {
        let freq = (-10000f32.ln() * (i as f32) / (half as f32)).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// Flux2 timestep embedding. `dim` is `time_in.in_layer`'s input
/// width. Sigmas at or below 1 are scaled by 1000, then cosine occupies the
/// first half and sine the second (`flip_sin_to_cos`).
pub fn flux_timestep_embed(sigma: f32, dim: usize) -> Vec<f32> {
    let t = if sigma <= 1.0 { sigma * 1000.0 } else { sigma };
    sinusoidal_timestep(t, dim)
}
