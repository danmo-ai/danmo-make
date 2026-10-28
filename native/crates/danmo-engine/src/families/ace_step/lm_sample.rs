//! ACE-Step LM token sampling (acestep.cpp `sampling.h` top-k / top-p).
//!
//! Temperature → softmax → optional top-k → optional top-p (nucleus) →
//! multinomial. `temperature <= 0` is greedy argmax (fail-loud on empty/NaN).

use crate::weights::WeightError;

/// Sample one token id from `logits` (length = vocab).
pub fn sample_top_k_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: i32,
    rng: &mut impl RngCore,
) -> Result<i32, WeightError> {
    if logits.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM sample needs non-empty logits".into(),
        ));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM temperature {temperature} is invalid"
        )));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM top_p {top_p} must be in (0, 1]"
        )));
    }
    if top_k < 0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM top_k {top_k} must be >= 0 (0 = disabled)"
        )));
    }

    // Greedy when temperature is 0 or top_k is 1.
    if temperature == 0.0 || top_k == 1 {
        return Ok(argmax(logits)?);
    }

    let mut scored: Vec<(i32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i as i32, v / temperature))
        .collect();
    if scored.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM logits have no finite values to sample".into(),
        ));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 {
        let k = (top_k as usize).min(scored.len());
        scored.truncate(k);
    }

    // Softmax on the surviving candidates.
    let max_l = scored
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, l) in scored.iter_mut() {
        *l = (*l - max_l).exp();
        sum += *l;
    }
    if !(sum > 0.0) || !sum.is_finite() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM softmax sum is not finite".into(),
        ));
    }
    for (_, p) in scored.iter_mut() {
        *p /= sum;
    }

    // Nucleus (top-p): keep smallest prefix whose cumulative mass >= top_p.
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cut = scored.len();
        for (i, (_, p)) in scored.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                cut = i + 1;
                break;
            }
        }
        scored.truncate(cut.max(1));
        let renorm: f32 = scored.iter().map(|(_, p)| *p).sum();
        if renorm > 0.0 {
            for (_, p) in scored.iter_mut() {
                *p /= renorm;
            }
        }
    }

    let u = rng.next_f32();
    let mut cum = 0.0f32;
    for &(id, p) in &scored {
        cum += p;
        if u <= cum {
            return Ok(id);
        }
    }
    Ok(scored.last().map(|(id, _)| *id).unwrap_or(0))
}

/// CFG blend: `uncond + scale * (cond - uncond)` (acestep.cpp phase-2).
/// Writes into `cond` in place. `scale <= 1` is a no-op.
pub fn cfg_blend(cond: &mut [f32], uncond: &[f32], scale: f32) -> Result<(), WeightError> {
    if cond.len() != uncond.len() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM CFG logit lengths {} vs {}",
            cond.len(),
            uncond.len()
        )));
    }
    if !scale.is_finite() || scale < 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM CFG scale {scale} is invalid"
        )));
    }
    if scale <= 1.0 {
        return Ok(());
    }
    for (c, &u) in cond.iter_mut().zip(uncond.iter()) {
        if !c.is_finite() || !u.is_finite() {
            continue;
        }
        *c = u + scale * (*c - u);
    }
    Ok(())
}

pub fn argmax(logits: &[f32]) -> Result<i32, WeightError> {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut any = false;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_v = v;
            best_i = i;
            any = true;
        }
    }
    if !any {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM argmax found no finite logit".into(),
        ));
    }
    Ok(best_i as i32)
}

/// Minimal RNG for sampling (xorshift64*). Seed 0 is remapped.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
        }
    }
}

pub trait RngCore {
    fn next_u64(&mut self) -> u64;
    fn next_f32(&mut self) -> f32 {
        let x = self.next_u64() >> 40; // 24 bits
        (x as f32) / ((1u32 << 24) as f32)
    }
}

impl RngCore for XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_argmax() {
        let logits = vec![0.1, 3.0, 0.2, -1.0];
        let mut rng = XorShift64::new(1);
        let id = sample_top_k_p(&logits, 0.0, 1.0, 0, &mut rng).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = vec![1.0, 5.0, 2.0];
        let mut rng = XorShift64::new(2);
        let id = sample_top_k_p(&logits, 1.0, 1.0, 1, &mut rng).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn samples_from_peaked_distribution() {
        let logits = vec![-10.0f32, 20.0, -10.0, -10.0];
        let mut rng = XorShift64::new(42);
        let mut counts = [0u32; 4];
        for _ in 0..64 {
            let id = sample_top_k_p(&logits, 0.5, 0.9, 0, &mut rng).unwrap();
            counts[id as usize] += 1;
        }
        assert!(counts[1] > 50, "{counts:?}");
    }

    #[test]
    fn rejects_bad_top_p() {
        let logits = vec![1.0f32];
        let mut rng = XorShift64::new(1);
        assert!(sample_top_k_p(&logits, 1.0, 0.0, 0, &mut rng).is_err());
    }

    #[test]
    fn cfg_blend_steers_toward_cond() {
        let mut cond = vec![2.0f32, 0.0];
        let uncond = vec![0.0f32, 2.0];
        cfg_blend(&mut cond, &uncond, 2.0).unwrap();
        // u + 2*(c-u) = 0 + 2*(2-0) = 4 ;  2 + 2*(0-2) = -2
        assert!((cond[0] - 4.0).abs() < 1e-5);
        assert!((cond[1] + 2.0).abs() < 1e-5);
        cfg_blend(&mut cond, &uncond, 1.0).unwrap(); // no-op
        assert!((cond[0] - 4.0).abs() < 1e-5);
    }
}
