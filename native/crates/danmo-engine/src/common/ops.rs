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
    use super::{
        empirical_mu, empirical_mu_sigmas, flow_match_shift_sigmas, scaled_sinusoidal_timestep,
    };

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
        let emb = scaled_sinusoidal_timestep(0.5, 4);
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
pub fn scaled_sinusoidal_timestep(sigma: f32, dim: usize) -> Vec<f32> {
    let t = if sigma <= 1.0 { sigma * 1000.0 } else { sigma };
    sinusoidal_timestep(t, dim)
}

// ---- Wan flow UniPC (`wan_flow_unipc`) ----

/// `_wan_flow_sigmas`: linspace over the unshifted train range, static shift,
/// terminal 0. Values are cast to F32 like the reference.
pub fn wan_flow_sigmas(steps: u32, shift: f32) -> Vec<f32> {
    let steps = steps.max(1) as usize;
    let train = 1000f64;
    let sigma_max = 1.0 - 1.0 / train;
    let mut sigmas = Vec::with_capacity(steps + 1);
    for i in 0..steps {
        let s = sigma_max - (i as f64) * sigma_max / (steps as f64);
        let shifted = shift as f64 * s / (1.0 + (shift as f64 - 1.0) * s);
        sigmas.push(shifted as f32);
    }
    sigmas.push(0.0);
    sigmas
}

/// Wan 2.x flow UniPC, predictor/corrector bh2, order ≤ 2 — a faithful port of
/// main `WanFlowUniPCScheduler` (including its update-the-x0-cache-before-the-
/// corrector ordering, which zeroes the corrector's own d1 term). Host-side;
/// the model output and sample are flat latent vectors.
pub struct WanFlowUniPc {
    sigmas: Vec<f32>,
    order: usize,
    num_steps: usize,
    step_index: usize,
    x0_prev: Option<Vec<f32>>,
    x0_cur: Option<Vec<f32>>,
    last_sample: Option<Vec<f32>>,
    lower_order_nums: usize,
}

fn flow_lambda(sigma: f32) -> f64 {
    ((1.0 - sigma as f64) / sigma as f64).ln()
}

impl WanFlowUniPc {
    pub fn new(steps: u32, shift: f32) -> Self {
        Self {
            sigmas: wan_flow_sigmas(steps, shift),
            order: 2,
            num_steps: steps.max(1) as usize,
            step_index: 0,
            x0_prev: None,
            x0_cur: None,
            last_sample: None,
            lower_order_nums: 0,
        }
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Timestep fed to the DiT: `trunc(sigma * 1000)` like the reference.
    pub fn timestep(&self, index: usize) -> f32 {
        (self.sigmas[index] * 1000.0).trunc()
    }

    /// One step: `model_output` is the CFG-combined velocity at `sample`.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>, String> {
        let i = self.step_index;
        if i >= self.num_steps || model_output.len() != sample.len() {
            return Err(format!(
                "wan unipc step {i} of {} with v {} sample {}",
                self.num_steps,
                model_output.len(),
                sample.len()
            ));
        }
        let sigma = self.sigmas[i];
        let x0: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(s, v)| s - sigma * v)
            .collect();
        self.x0_prev = self.x0_cur.take();
        self.x0_cur = Some(x0.clone());
        let this_order = self
            .order
            .min(self.num_steps - i)
            .min(self.lower_order_nums + 1);
        let sigma_t = self.sigmas[i + 1];
        if self.lower_order_nums < self.order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        if sigma_t == 0.0 {
            return Ok(x0);
        }
        let mut sample_vec = sample.to_vec();
        if i > 0 {
            if let Some(last) = self.last_sample.take() {
                sample_vec = self.corrector(&x0, &last, &sample_vec, this_order);
            }
        }
        self.last_sample = Some(sample_vec.clone());
        Ok(self.predictor(&x0, &sample_vec, this_order))
    }

    fn predictor(&self, x0: &[f32], sample: &[f32], order: usize) -> Vec<f32> {
        let i = self.step_index - 1;
        let sigma_s0 = self.sigmas[i] as f64;
        let sigma_t = self.sigmas[i + 1] as f64;
        let h = flow_lambda(self.sigmas[i + 1]) - flow_lambda(self.sigmas[i]);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let alpha_t = 1.0 - sigma_t;
        let ratio = (sigma_t / sigma_s0) as f32;
        let coeff = (alpha_t * h_phi_1) as f32;
        let mut x_t: Vec<f32> = sample
            .iter()
            .zip(x0)
            .map(|(s, m)| ratio * s - coeff * m)
            .collect();
        if order >= 2 {
            if let (Some(mk), true) = (&self.x0_prev, i >= 1) {
                let rk = (flow_lambda(self.sigmas[i - 1]) - flow_lambda(self.sigmas[i])) / h;
                if rk.is_finite() {
                    let c = (alpha_t * h_phi_1 * 0.5 / rk) as f32;
                    for (x, (mkv, m0)) in x_t.iter_mut().zip(mk.iter().zip(x0)) {
                        *x -= c * (mkv - m0);
                    }
                }
            }
        }
        x_t
    }

    fn corrector(&self, x0: &[f32], last_sample: &[f32], sample: &[f32], order: usize) -> Vec<f32> {
        let i = self.step_index - 1;
        let sigma_s0 = self.sigmas[i - 1] as f64;
        let sigma_t = self.sigmas[i] as f64;
        let h = flow_lambda(self.sigmas[i]) - flow_lambda(self.sigmas[i - 1]);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;
        let alpha_t = 1.0 - sigma_t;
        let ratio = (sigma_t / sigma_s0) as f32;
        let coeff = (alpha_t * h_phi_1) as f32;
        // x_t_ uses m0 = x0 (cache updated first); d1_t = x0 - m0 = 0 like main.
        let x_t: Vec<f32> = last_sample
            .iter()
            .zip(x0)
            .map(|(s, m)| ratio * s - coeff * m)
            .collect();
        let _ = sample;
        if order < 2 || i < 2 {
            return x_t;
        }
        let Some(mk) = &self.x0_prev else {
            return x_t;
        };
        let rk = (flow_lambda(self.sigmas[i - 2]) - flow_lambda(self.sigmas[i - 1])) / h;
        if !rk.is_finite() {
            return x_t;
        }
        // Solve [[1, 1], [rk, 1]] @ rhos = [b1, b2] (np.linalg.solve, 2x2).
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let b1 = h_phi_k / b_h;
        h_phi_k = h_phi_k / hh - 0.5;
        let b2 = h_phi_k * 2.0 / b_h;
        let det = 1.0 - rk;
        if det.abs() < 1e-12 {
            return x_t;
        }
        let r0 = (b1 - b2) / det;
        let scale = (alpha_t * b_h * r0 / rk) as f32;
        x_t.iter()
            .zip(mk.iter().zip(x0))
            .map(|(x, (mkv, m0))| x - scale * (mkv - m0))
            .collect()
    }
}

#[cfg(test)]
mod wan_unipc_tests {
    use super::{wan_flow_sigmas, WanFlowUniPc};

    #[test]
    fn wan_flow_sigmas_match_main() {
        let sigmas = wan_flow_sigmas(4, 5.0);
        let expect = [0.99979985, 0.93726546, 0.83305538, 0.62468731, 0.0];
        assert_eq!(sigmas.len(), 5);
        for (got, want) in sigmas.iter().zip(expect) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// Golden values from main `WanFlowUniPCScheduler` (shift 5, 4 steps,
    /// bh2 order 2) driven by the fixed vectors below.
    #[test]
    fn unipc_bh2_matches_main_step_by_step() {
        let latent = [
            1.2301533575e-03f32,
            2.9874553751e-01,
            -2.7413785536e-01,
            -8.9059183876e-01,
            -4.5467078517e-01,
            -9.9164655500e-01,
        ];
        let v = [
            [
                6.0143602597e-02,
                1.3402152456e+00,
                -4.9220651855e-01,
                -6.2047489982e-01,
                4.8984205019e-01,
                3.5688700816e-01,
            ],
            [
                1.0541424900e-01,
                -9.3046804471e-01,
                -2.9251822463e-02,
                6.9530319446e-01,
                -1.3442145473e+00,
                -4.5761576104e-01,
            ],
            [
                -1.9012227398e+00,
                -1.2895377398e+00,
                -1.8417350378e+00,
                -2.3509113107e-01,
                -1.2674464814e+00,
                2.7126435882e-01,
            ],
            [
                1.5675108662e-01,
                -1.8693094463e-01,
                -2.5167597108e+00,
                -5.3869289585e-01,
                -4.8500945401e-02,
                1.1330898600e-01,
            ],
        ];
        let expect = [
            [
                -0.00253089,
                0.21493599,
                -0.24335802,
                -0.85179082,
                -0.48530276,
                -1.01396427,
            ],
            [
                -0.01632000,
                0.45253716,
                -0.26898324,
                -1.00574244,
                -0.23162794,
                -0.91582905,
            ],
            [
                0.72894999,
                0.79640526,
                0.42776048,
                -0.80210157,
                0.02932185,
                -1.09471572,
            ],
            [
                0.63102957,
                0.91317865,
                1.99994834,
                -0.46558695,
                0.05961977,
                -1.16549840,
            ],
        ];
        let mut solver = WanFlowUniPc::new(4, 5.0);
        let mut x = latent.to_vec();
        let want_ts = [999.0f32, 937.0, 833.0, 624.0];
        for i in 0..4 {
            assert_eq!(solver.timestep(i), want_ts[i]);
            x = solver.step(&v[i], &x).unwrap();
            for (got, want) in x.iter().zip(expect[i]) {
                assert!((got - want).abs() < 1e-4, "step {i}: {got} vs {want}");
            }
        }
    }
}
