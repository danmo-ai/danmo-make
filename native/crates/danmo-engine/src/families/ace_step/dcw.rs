use crate::weights::WeightError;

const INV_SQRT2: f32 = 0.7071067811865476;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcwMode {
    Low,
    High,
    Double,
    Pix,
}

impl DcwMode {
    pub fn parse(raw: &str) -> Result<Self, WeightError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "high" => Ok(Self::High),
            "double" => Ok(Self::Double),
            "pix" => Ok(Self::Pix),
            other => Err(WeightError::NotBuilt(format!(
                "ACE-Step dcw_mode '{other}' is unknown (use low, high, double, or pix)"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Double => "double",
            Self::Pix => "pix",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DcwSettings {
    pub enabled: Option<bool>,
    pub mode: DcwMode,
    pub scaler: f32,
    pub high_scaler: f32,
}

impl Default for DcwSettings {
    fn default() -> Self {
        Self {
            enabled: None,
            mode: DcwMode::Double,
            scaler: 0.05,
            high_scaler: 0.02,
        }
    }
}

impl DcwSettings {
    pub fn validate(self) -> Result<Self, WeightError> {
        for (name, value) in [
            ("dcw_scaler", self.scaler),
            ("dcw_high_scaler", self.high_scaler),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step {name} {value} must be in [0, 1]"
                )));
            }
        }
        Ok(self)
    }

    pub fn active(self, is_turbo: bool) -> bool {
        self.enabled.unwrap_or(is_turbo)
            && (self.scaler != 0.0 || (self.mode == DcwMode::Double && self.high_scaler != 0.0))
    }

    pub fn apply(
        self,
        x_next: &mut [f32],
        previous: &[f32],
        velocity: &[f32],
        t_curr: f32,
        frames: i64,
        channels: i64,
        is_turbo: bool,
    ) -> Result<(), WeightError> {
        self.validate()?;
        let expected = (frames * channels) as usize;
        if frames < 1 || channels < 1 || x_next.len() != expected {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step DCW latent length {} is not {expected}",
                x_next.len()
            )));
        }
        if previous.len() != expected || velocity.len() != expected || !t_curr.is_finite() {
            return Err(WeightError::NotBuilt(
                "ACE-Step DCW previous/velocity shapes or timestep are invalid".into(),
            ));
        }
        if !self.active(is_turbo) {
            return Ok(());
        }
        let channels = channels as usize;
        let frames = frames as usize;
        if self.mode == DcwMode::Pix {
            for index in 0..expected {
                let denoised = previous[index] - t_curr * velocity[index];
                x_next[index] += self.scaler * (x_next[index] - denoised);
            }
            return Ok(());
        }

        let pairs = frames.div_ceil(2);
        let band_len = pairs * channels;
        let mut low = vec![0.0; band_len];
        let mut high = vec![0.0; band_len];
        for pair in 0..pairs {
            for channel in 0..channels {
                let even_index = pair * 2 * channels + channel;
                let odd_index = even_index + channels;
                let even = x_next[even_index];
                let odd = if odd_index < expected {
                    x_next[odd_index]
                } else {
                    0.0
                };
                let denoised_even = previous[even_index] - t_curr * velocity[even_index];
                let denoised_odd = if odd_index < expected {
                    previous[odd_index] - t_curr * velocity[odd_index]
                } else {
                    0.0
                };
                let band = pair * channels + channel;
                let x_low = (even + odd) * INV_SQRT2;
                let x_high = (even - odd) * INV_SQRT2;
                let y_low = (denoised_even + denoised_odd) * INV_SQRT2;
                let y_high = (denoised_even - denoised_odd) * INV_SQRT2;
                low[band] = match self.mode {
                    DcwMode::Low | DcwMode::Double => {
                        x_low + t_curr * self.scaler * (x_low - y_low)
                    }
                    _ => x_low,
                };
                high[band] = match self.mode {
                    DcwMode::High => x_high + (1.0 - t_curr) * self.scaler * (x_high - y_high),
                    DcwMode::Double => {
                        x_high + (1.0 - t_curr) * self.high_scaler * (x_high - y_high)
                    }
                    _ => x_high,
                };
            }
        }
        for pair in 0..pairs {
            for channel in 0..channels {
                let band = pair * channels + channel;
                let even = (low[band] + high[band]) * INV_SQRT2;
                let odd = (low[band] - high[band]) * INV_SQRT2;
                let index = pair * 2 * channels + channel;
                x_next[index] = even;
                if index + channels < expected {
                    x_next[index + channels] = odd;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(mode: DcwMode) -> DcwSettings {
        DcwSettings {
            enabled: Some(true),
            mode,
            scaler: 0.05,
            high_scaler: 0.02,
        }
    }

    #[test]
    fn pix_pushes_the_next_latent_away_from_denoised() {
        let previous = [1.0, 2.0];
        let velocity = [0.25, -0.5];
        let mut next = [0.9, 1.9];
        settings(DcwMode::Pix)
            .apply(&mut next, &previous, &velocity, 0.5, 2, 1, false)
            .unwrap();
        assert!((next[0] - 0.90125).abs() < 1e-6);
        assert!((next[1] - 1.8825).abs() < 1e-6);
    }

    #[test]
    fn haar_modes_are_finite_and_odd_frames_round_trip() {
        for mode in [DcwMode::Low, DcwMode::High, DcwMode::Double] {
            let previous = [0.2, -0.4, 0.6, 0.8, -1.0, 0.5];
            let velocity = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6];
            let original = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
            let mut next = original;
            settings(mode)
                .apply(&mut next, &previous, &velocity, 0.75, 3, 2, false)
                .unwrap();
            assert!(next.iter().all(|value| value.is_finite()));
            assert_ne!(next, original);
        }
    }

    #[test]
    fn turbo_default_and_zero_scalers_match_main() {
        let turbo = DcwSettings {
            enabled: None,
            ..settings(DcwMode::Double)
        };
        assert!(turbo.active(true));
        assert!(!turbo.active(false));
        let zero = DcwSettings {
            scaler: 0.0,
            high_scaler: 0.0,
            ..turbo
        };
        assert!(!zero.active(true));
        assert!(DcwMode::parse("pix").unwrap() == DcwMode::Pix);
        assert!(DcwMode::parse("other").is_err());
    }
}
