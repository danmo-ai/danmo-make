use crate::weights::WeightError;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct PmiAssessment {
    pub score: f32,
    pub mean_logprob: f32,
}

impl PmiAssessment {
    pub fn from_mean_logprob(mean_logprob: f32) -> Option<Self> {
        if !mean_logprob.is_finite() {
            return None;
        }
        Some(Self {
            score: ((mean_logprob + 4.0) / 4.0).clamp(0.0, 1.0),
            mean_logprob,
        })
    }

    pub fn quality_bonus(&self) -> f32 {
        if self.score >= 0.75 {
            4.0
        } else if self.score >= 0.55 {
            1.5
        } else if self.score < 0.35 {
            -6.0
        } else {
            0.0
        }
    }

    pub fn as_metadata(&self) -> Value {
        json!({
            "enabled": true,
            "score": self.score,
            "mean_logprob": self.mean_logprob,
            "quality_bonus": self.quality_bonus(),
            "note": "ok",
        })
    }
}

pub fn pmi_scoring_enabled() -> bool {
    std::env::var("ACESTEP_PMI_SCORING")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub score: f32,
    pub grade: String,
    pub warnings: Vec<String>,
    pub hum_ratio: f32,
    pub mains_acf: f32,
    pub latent_cos: f32,
    pub latent_diff: f32,
    pub lm_expanded: bool,
    pub pmi_bonus: f32,
    pub attempts: u32,
}

impl QualityAssessment {
    pub fn as_metadata(&self) -> Value {
        json!({
            "quality_score": (self.score * 10.0).round() / 10.0,
            "quality_grade": self.grade,
            "quality_warnings": self.warnings,
            "hum_ratio": self.hum_ratio,
            "mains_acf": self.mains_acf,
            "latent_cos": self.latent_cos,
            "latent_diff_mean": self.latent_diff,
            "lm_expanded": self.lm_expanded,
            "pmi_bonus": self.pmi_bonus,
            "quality_attempts": self.attempts,
        })
    }

    pub fn log_message(&self) -> Option<String> {
        if self.grade == "good" && self.warnings.is_empty() && self.attempts <= 1 {
            return None;
        }
        Some(format!(
            "ACE-Step quality {:.0}/100 ({}); signals={}; attempts={}",
            self.score,
            self.grade,
            if self.warnings.is_empty() {
                "none".to_string()
            } else {
                self.warnings.join(",")
            },
            self.attempts
        ))
    }
}

pub fn assess_generation_quality(
    hum_ratio: f32,
    mains_acf: f32,
    latent_cos: f32,
    latent_diff: f32,
    lm_expanded: bool,
    near_silence: bool,
    pmi_bonus: f32,
    attempts: u32,
) -> QualityAssessment {
    let mut score = 78.0;
    let mut warnings = Vec::new();
    if near_silence || (latent_cos >= 0.95 && latent_diff < 0.10) {
        score -= 45.0;
        warnings.push("latent_near_silence".into());
    }
    if hum_ratio > 0.35 {
        score -= 18.0;
        warnings.push("high_hum_ratio".into());
    } else if hum_ratio > 0.22 {
        score -= 8.0;
        warnings.push("moderate_hum_ratio".into());
    }
    if mains_acf > 0.45 {
        score -= 12.0;
        warnings.push("mains_hum".into());
    } else if mains_acf > 0.32 {
        score -= 5.0;
    }
    if lm_expanded {
        score += 2.0;
    }
    if pmi_bonus != 0.0 {
        score += pmi_bonus;
        if pmi_bonus < 0.0 {
            warnings.push("low_lm_pmi".into());
        }
    }
    let score = score.clamp(0.0, 100.0);
    let grade = if score >= 72.0 {
        "good"
    } else if score >= 55.0 {
        "fair"
    } else {
        "poor"
    }
    .into();
    QualityAssessment {
        score,
        grade,
        warnings,
        hum_ratio,
        mains_acf,
        latent_cos,
        latent_diff,
        lm_expanded,
        pmi_bonus,
        attempts,
    }
}

pub fn assess_output(
    latents: &[f32],
    reference: &[f32],
    normalized_audio: &[f32],
    sample_rate: u32,
    lm_expanded: bool,
    pmi_bonus: f32,
    attempts: u32,
) -> Result<QualityAssessment, WeightError> {
    let latent_cos = latent_cosine_similarity(latents, reference)?;
    let latent_diff = latent_diff_mean(latents, reference)?;
    let near_silence = latent_cos >= 0.95 && latent_diff < 0.10;
    let hum_ratio = estimate_hum_ratio(normalized_audio, sample_rate);
    let mains_acf = estimate_mains_correlation(normalized_audio, sample_rate, 50.0);
    Ok(assess_generation_quality(
        hum_ratio,
        mains_acf,
        latent_cos,
        latent_diff,
        lm_expanded,
        near_silence,
        pmi_bonus,
        attempts,
    ))
}

pub fn latent_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, WeightError> {
    if a.len() != b.len() || a.is_empty() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step latent cosine lengths {} and {} are invalid",
            a.len(),
            b.len()
        )));
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&left, &right) in a.iter().zip(b) {
        dot += f64::from(left) * f64::from(right);
        norm_a += f64::from(left) * f64::from(left);
        norm_b += f64::from(right) * f64::from(right);
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return Ok(1.0);
    }
    Ok((dot / denom) as f32)
}

pub fn latent_diff_mean(a: &[f32], b: &[f32]) -> Result<f32, WeightError> {
    if a.len() != b.len() || a.is_empty() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step latent diff lengths {} and {} are invalid",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / a.len() as f32)
}

pub fn latents_collapsed_to_silence(
    latents: &[f32],
    reference: &[f32],
) -> Result<(bool, f32, f32), WeightError> {
    let cos = latent_cosine_similarity(latents, reference)?;
    let diff = latent_diff_mean(latents, reference)?;
    Ok((cos >= 0.95 && diff < 0.10, cos, diff))
}

pub fn diffusion_retry_seed(base_seed: i64, attempt: u32) -> i64 {
    if attempt == 0 {
        return base_seed & 0x7fff_ffff;
    }
    let mut mixed = (base_seed as u64) ^ (0x9e37_79b9u64.wrapping_mul(u64::from(attempt + 1)));
    mixed = mixed.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
    (mixed as i64).max(1)
}

pub fn normalize_planar_stereo(audio: &[f32], frames: i64) -> Result<Vec<f32>, WeightError> {
    let frames = frames as usize;
    if frames < 1 || audio.len() != frames * 2 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step planar stereo length {} is not 2×{frames}",
            audio.len()
        )));
    }
    let mut out = audio.to_vec();
    for channel in 0..2 {
        let mean = out[channel * frames..(channel + 1) * frames]
            .iter()
            .sum::<f32>()
            / frames as f32;
        for sample in &mut out[channel * frames..(channel + 1) * frames] {
            *sample -= mean;
        }
    }
    let peak = out.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let rms = (out.iter().map(|value| value * value).sum::<f32>() / out.len() as f32).sqrt();
    if peak < 1e-5 || rms < 1e-6 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step VAE decode produced near-silent audio (peak={peak:.2e}, rms={rms:.2e})"
        )));
    }
    if peak > 1.0 {
        for sample in &mut out {
            *sample /= peak;
        }
    }
    let target = 10.0f32.powf(-1.0 / 20.0);
    let peak = out.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    if peak > 1e-8 {
        let scale = target / peak;
        for sample in &mut out {
            *sample *= scale;
        }
    }
    Ok(out)
}

pub fn estimate_hum_ratio(audio: &[f32], sample_rate: u32) -> f32 {
    let sample_rate = sample_rate as usize;
    if sample_rate == 0 {
        return 0.0;
    }
    let mono = mono(audio);
    let len = mono.len().min(sample_rate * 5);
    if len < sample_rate {
        return 0.0;
    }
    let size = len.next_power_of_two();
    let mut spectrum: Vec<Complex> = mono[..len]
        .iter()
        .map(|value| Complex {
            re: *value,
            im: 0.0,
        })
        .collect();
    spectrum.resize(size, Complex::default());
    fft(&mut spectrum);
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    for (index, value) in spectrum.iter().take(size / 2 + 1).enumerate() {
        let frequency = index as f64 * sample_rate as f64 / size as f64;
        let power = f64::from(value.re * value.re + value.im * value.im);
        if (50.0..120.0).contains(&frequency) {
            low += power;
        } else if (200.0..4000.0).contains(&frequency) {
            mid += power;
        }
    }
    (low / (mid + 1e-12)) as f32
}

pub fn estimate_mains_correlation(audio: &[f32], sample_rate: u32, mains_hz: f32) -> f32 {
    if sample_rate == 0 || !mains_hz.is_finite() || mains_hz <= 0.0 {
        return 0.0;
    }
    let mono = mono(audio);
    let len = mono.len().min(sample_rate as usize * 10);
    if len < sample_rate as usize {
        return 0.0;
    }
    let lag = (sample_rate as f32 / mains_hz).round() as usize;
    if lag >= len {
        return 0.0;
    }
    let energy = mono[..len]
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if energy < 1e-12 {
        return 0.0;
    }
    let correlation = mono[lag..len]
        .iter()
        .zip(&mono[..len - lag])
        .map(|(right, left)| f64::from(*right) * f64::from(*left))
        .sum::<f64>();
    (correlation / energy) as f32
}

fn mono(audio: &[f32]) -> Vec<f32> {
    if audio.len() % 2 != 0 {
        return audio.to_vec();
    }
    let frames = audio.len() / 2;
    (0..frames)
        .map(|index| (audio[index] + audio[frames + index]) * 0.5)
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

fn fft(values: &mut [Complex]) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    let mut target = 0;
    for index in 1..n {
        let mut bit = n >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target ^= bit;
        if index < target {
            values.swap(index, target);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * std::f32::consts::PI / len as f32;
        let root = Complex {
            re: angle.cos(),
            im: angle.sin(),
        };
        for start in (0..n).step_by(len) {
            let mut factor = Complex { re: 1.0, im: 0.0 };
            for offset in 0..len / 2 {
                let even = values[start + offset];
                let odd = multiply(values[start + offset + len / 2], factor);
                values[start + offset] = Complex {
                    re: even.re + odd.re,
                    im: even.im + odd.im,
                };
                values[start + offset + len / 2] = Complex {
                    re: even.re - odd.re,
                    im: even.im - odd.im,
                };
                factor = multiply(factor, root);
            }
        }
        len *= 2;
    }
}

fn multiply(left: Complex, right: Complex) -> Complex {
    Complex {
        re: left.re * right.re - left.im * right.im,
        im: left.re * right.im + left.im * right.re,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_and_grades_match_main() {
        let good = assess_generation_quality(0.0, 0.0, 0.2, 0.8, true, false, 0.0, 1);
        assert_eq!(good.score, 80.0);
        assert_eq!(good.grade, "good");
        let poor = assess_generation_quality(0.4, 0.5, 0.99, 0.01, false, true, -6.0, 3);
        assert_eq!(poor.score, 0.0);
        assert_eq!(poor.grade, "poor");
        assert!(poor.warnings.contains(&"latent_near_silence".into()));
        assert!(poor.warnings.contains(&"low_lm_pmi".into()));
    }

    #[test]
    fn pmi_score_and_bonus_match_main() {
        let high = PmiAssessment::from_mean_logprob(-1.0).unwrap();
        assert_eq!(high.score, 0.75);
        assert_eq!(high.quality_bonus(), 4.0);
        let low = PmiAssessment::from_mean_logprob(-3.0).unwrap();
        assert_eq!(low.score, 0.25);
        assert_eq!(low.quality_bonus(), -6.0);
        assert!(PmiAssessment::from_mean_logprob(f32::NAN).is_none());
    }

    #[test]
    fn latent_signals_and_retry_seed_match_main() {
        let (collapsed, cos, diff) =
            latents_collapsed_to_silence(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert!(collapsed);
        assert_eq!(cos, 1.0);
        assert_eq!(diff, 0.0);
        assert_eq!(diffusion_retry_seed(7, 0), 7);
        assert_ne!(diffusion_retry_seed(7, 1), 7);
        assert!(diffusion_retry_seed(7, 1) > 0);
    }

    #[test]
    fn normalize_and_mains_signals_cover_fifty_hz_hum() {
        let rate = 48_000;
        let frames = rate as i64;
        let mut audio = vec![0.0; frames as usize * 2];
        for index in 0..frames as usize {
            let value = (index as f32 * 2.0 * std::f32::consts::PI * 50.0 / rate as f32).sin();
            audio[index] = value + 0.1;
            audio[frames as usize + index] = value + 0.1;
        }
        let normalized = normalize_planar_stereo(&audio, frames).unwrap();
        assert!(estimate_hum_ratio(&normalized, rate) > 0.35);
        assert!(estimate_mains_correlation(&normalized, rate, 50.0) > 0.45);
        assert!(normalize_planar_stereo(&[0.0; 2], 1).is_err());
    }
}
