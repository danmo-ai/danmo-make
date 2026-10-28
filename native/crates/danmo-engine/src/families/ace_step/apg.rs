//! Adaptive Projected Guidance (acestep.cpp `dit-sampler.h` / Python `apg_guidance.py`).
//!
//! Layout is time-major `[T, Oc]` (contiguous channels per frame).

const DEFAULT_MOMENTUM: f64 = -0.75;
const DEFAULT_NORM_THRESHOLD: f32 = 2.5;

#[derive(Debug, Clone)]
pub struct ApgMomentum {
    momentum: f64,
    running_average: Vec<f64>,
    initialized: bool,
}

impl ApgMomentum {
    pub fn new() -> Self {
        Self {
            momentum: DEFAULT_MOMENTUM,
            running_average: Vec::new(),
            initialized: false,
        }
    }

    fn update(&mut self, values: &[f64]) {
        if !self.initialized {
            self.running_average = values.to_vec();
            self.initialized = true;
            return;
        }
        for (avg, value) in self.running_average.iter_mut().zip(values.iter()) {
            *avg = *value + self.momentum * *avg;
        }
    }
}

impl Default for ApgMomentum {
    fn default() -> Self {
        Self::new()
    }
}

/// `result = pred_cond + (scale - 1) * orth(diff_momentum, pred_cond)`.
pub fn apg_forward(
    pred_cond: &[f32],
    pred_uncond: &[f32],
    guidance_scale: f32,
    mbuf: &mut ApgMomentum,
    out_channels: i64,
    frames: i64,
) -> Result<Vec<f32>, String> {
    apg_forward_with_threshold(
        pred_cond,
        pred_uncond,
        guidance_scale,
        mbuf,
        out_channels,
        frames,
        DEFAULT_NORM_THRESHOLD,
    )
}

pub fn apg_forward_with_threshold(
    pred_cond: &[f32],
    pred_uncond: &[f32],
    guidance_scale: f32,
    mbuf: &mut ApgMomentum,
    out_channels: i64,
    frames: i64,
    norm_threshold: f32,
) -> Result<Vec<f32>, String> {
    let n = (out_channels * frames) as usize;
    if pred_cond.len() != n || pred_uncond.len() != n {
        return Err(format!(
            "APG lengths cond {} uncond {}, expected {n}",
            pred_cond.len(),
            pred_uncond.len()
        ));
    }
    if !guidance_scale.is_finite() {
        return Err("APG guidance scale is not finite".into());
    }
    let oc = out_channels as usize;
    let t_len = frames as usize;

    let mut diff: Vec<f64> = pred_cond
        .iter()
        .zip(pred_uncond.iter())
        .map(|(c, u)| f64::from(*c) - f64::from(*u))
        .collect();
    mbuf.update(&diff);
    diff.copy_from_slice(&mbuf.running_average);

    if norm_threshold > 0.0 {
        let thr = f64::from(norm_threshold);
        for c in 0..oc {
            let mut norm2 = 0.0;
            for t in 0..t_len {
                let v = diff[t * oc + c];
                norm2 += v * v;
            }
            let norm = norm2.max(0.0).sqrt();
            let scale = if norm > 1e-60 {
                (1.0f64).min(thr / norm)
            } else {
                1.0
            };
            if scale < 1.0 {
                for t in 0..t_len {
                    diff[t * oc + c] *= scale;
                }
            }
        }
    }

    let pred_cond_d: Vec<f64> = pred_cond.iter().map(|v| f64::from(*v)).collect();
    let mut par = vec![0f64; n];
    let mut orth = vec![0f64; n];
    apg_project(&diff, &pred_cond_d, &mut par, &mut orth, oc, t_len);

    let w = f64::from(guidance_scale) - 1.0;
    let mut result = vec![0f32; n];
    for i in 0..n {
        result[i] = (pred_cond_d[i] + w * orth[i]) as f32;
    }
    Ok(result)
}

fn apg_project(v0: &[f64], v1: &[f64], out_par: &mut [f64], out_orth: &mut [f64], oc: usize, t_len: usize) {
    for c in 0..oc {
        let mut norm2 = 0.0;
        for t in 0..t_len {
            let v = v1[t * oc + c];
            norm2 += v * v;
        }
        let inv_norm = if norm2 > 1e-60 {
            1.0 / norm2.sqrt()
        } else {
            0.0
        };
        let mut dot = 0.0;
        for t in 0..t_len {
            dot += v0[t * oc + c] * (v1[t * oc + c] * inv_norm);
        }
        for t in 0..t_len {
            let idx = t * oc + c;
            let v1n = v1[idx] * inv_norm;
            out_par[idx] = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

/// Broadcast `null_condition_emb` `[condition_dim]` across `enc_seq` tokens.
pub fn broadcast_null_enc(null_emb: &[f32], enc_seq: i64) -> Result<Vec<f32>, String> {
    if null_emb.is_empty() {
        return Err("null_condition_emb is empty".into());
    }
    if enc_seq < 1 {
        return Err("enc_seq must be >= 1 for null condition broadcast".into());
    }
    let h = null_emb.len();
    let mut out = vec![0f32; h * enc_seq as usize];
    for s in 0..enc_seq as usize {
        out[s * h..(s + 1) * h].copy_from_slice(null_emb);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{apg_forward_with_threshold, broadcast_null_enc, ApgMomentum};

    #[test]
    fn guidance_one_returns_cond_when_diff_is_zero() {
        let cond = vec![1.0f32, 2.0, 3.0, 4.0];
        let uncond = cond.clone();
        let mut mbuf = ApgMomentum::new();
        let out = apg_forward_with_threshold(&cond, &uncond, 1.0, &mut mbuf, 2, 2, 2.5).unwrap();
        assert_eq!(out, cond);
    }

    #[test]
    fn guidance_two_adds_orthogonal_diff() {
        // cond = [1,0, 0,1], uncond = [0,0, 0,0] → diff = cond.
        // Projecting diff onto cond leaves orthogonal = 0 after first step's
        // momentum equals diff; result = cond + 1 * 0 = cond when parallel.
        // Use orthogonal uncond: uncond along a different direction per channel.
        let cond = vec![1.0f32, 0.0, 1.0, 0.0]; // T=2, Oc=2: ch0=[1,1], ch1=[0,0]
        let uncond = vec![0.0f32, 1.0, 0.0, 1.0]; // ch0=[0,0], ch1=[1,1]
        let mut mbuf = ApgMomentum::new();
        let out = apg_forward_with_threshold(&cond, &uncond, 2.0, &mut mbuf, 2, 2, 0.0).unwrap();
        // diff = cond-uncond = [1,-1, 1,-1]
        // ch0: v1=[1,1], diff=[1,1] → parallel to cond, orth≈0 for ch0
        // ch1: v1=[0,0], orth = diff = [-1,-1]; result ch1 = 0 + 1*(-1) = -1
        assert!((out[0] - 1.0).abs() < 1e-5, "{out:?}");
        assert!((out[1] + 1.0).abs() < 1e-5, "{out:?}");
        assert!((out[2] - 1.0).abs() < 1e-5, "{out:?}");
        assert!((out[3] + 1.0).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn null_enc_broadcast_repeats_the_vector() {
        let emb = vec![0.5f32, -0.25, 1.0];
        let out = broadcast_null_enc(&emb, 2).unwrap();
        assert_eq!(out, vec![0.5, -0.25, 1.0, 0.5, -0.25, 1.0]);
    }
}
