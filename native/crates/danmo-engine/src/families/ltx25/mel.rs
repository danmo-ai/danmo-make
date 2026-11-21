//! Host log-mel spectrogram for LTX 2.5 audio VAE encode.
//!
//! Matches the published preprocessor: 16 kHz stereo, Hann STFT (n_fft=1024,
//! hop=160, center/reflect), Slaney mel filterbank, log(clamp(mel, 1e-5)).

use danmo_mlx::Error;

const TARGET_RATE: u32 = 16_000;
const N_FFT: usize = 1024;
const HOP: usize = 160;
const N_MELS: usize = 64;
const F_MAX: f32 = 8_000.0;

/// Resample interleaved stereo PCM to 16 kHz stereo `[L0,R0,L1,R1,…]`.
pub fn resample_stereo_interleaved(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, Error> {
    if sample_rate == 0 {
        return Err(Error::new("LTX 2.5 mel", "sample rate must be positive"));
    }
    if samples.len() % 2 != 0 {
        return Err(Error::new("LTX 2.5 mel", "stereo PCM length must be even"));
    }
    if sample_rate == TARGET_RATE {
        return Ok(samples.to_vec());
    }
    let frames = samples.len() / 2;
    if frames == 0 {
        return Ok(Vec::new());
    }
    let out_frames = ((frames as u64 * TARGET_RATE as u64) / sample_rate as u64).max(1) as usize;
    let mut out = vec![0.0f32; out_frames * 2];
    let scale = (frames - 1) as f64 / (out_frames.max(1) - 1).max(1) as f64;
    for (index, chunk) in out.chunks_exact_mut(2).enumerate() {
        let pos = index as f64 * scale;
        let left = pos.floor() as usize;
        let right = (left + 1).min(frames - 1);
        let frac = (pos - left as f64) as f32;
        for channel in 0..2 {
            let a = samples[left * 2 + channel];
            let b = samples[right * 2 + channel];
            chunk[channel] = a + (b - a) * frac;
        }
    }
    Ok(out)
}

/// Interleaved stereo → planar `[L…; R…]` then log-mel `[2, time, 64]`.
pub fn stereo_log_mel(interleaved_16k: &[f32]) -> Result<(Vec<f32>, usize), Error> {
    if interleaved_16k.len() % 2 != 0 {
        return Err(Error::new("LTX 2.5 mel", "stereo PCM length must be even"));
    }
    let frames = interleaved_16k.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for pair in interleaved_16k.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    let mel_left = channel_log_mel(&left)?;
    let mel_right = channel_log_mel(&right)?;
    if mel_left.len() != mel_right.len() {
        return Err(Error::new(
            "LTX 2.5 mel",
            "stereo mel channel lengths diverge",
        ));
    }
    let time = mel_left.len() / N_MELS;
    let mut planar = Vec::with_capacity(mel_left.len() * 2);
    planar.extend_from_slice(&mel_left);
    planar.extend_from_slice(&mel_right);
    Ok((planar, time))
}

fn channel_log_mel(wave: &[f32]) -> Result<Vec<f32>, Error> {
    let magnitude = stft_magnitude(wave)?;
    let frames = magnitude.len() / (N_FFT / 2 + 1);
    let filterbank = slaney_mel_filterbank();
    let mut mel = vec![0.0f32; frames * N_MELS];
    let bins = N_FFT / 2 + 1;
    for frame in 0..frames {
        let spectrum = &magnitude[frame * bins..(frame + 1) * bins];
        for mel_bin in 0..N_MELS {
            let mut acc = 0.0f32;
            let row = &filterbank[mel_bin * bins..(mel_bin + 1) * bins];
            for (weight, sample) in row.iter().zip(spectrum.iter()) {
                acc += weight * sample;
            }
            mel[frame * N_MELS + mel_bin] = acc.max(1e-5).ln();
        }
    }
    Ok(mel)
}

fn stft_magnitude(wave: &[f32]) -> Result<Vec<f32>, Error> {
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; wave.len() + 2 * pad];
    // reflect pad (center=True)
    for i in 0..pad {
        let src = (pad - i).min(wave.len().saturating_sub(1));
        padded[i] = wave.get(src).copied().unwrap_or(0.0);
    }
    padded[pad..pad + wave.len()].copy_from_slice(wave);
    for i in 0..pad {
        let src = wave.len().saturating_sub(2).saturating_sub(i);
        padded[pad + wave.len() + i] = wave.get(src).copied().unwrap_or(0.0);
    }
    let window = hann_window(N_FFT);
    let frames = if padded.len() < N_FFT {
        0
    } else {
        1 + (padded.len() - N_FFT) / HOP
    };
    let bins = N_FFT / 2 + 1;
    let mut out = vec![0.0f32; frames * bins];
    let mut frame = vec![0.0f32; N_FFT];
    let mut spectrum = vec![0.0f32; N_FFT];
    for index in 0..frames {
        let start = index * HOP;
        for (dst, (sample, weight)) in frame
            .iter_mut()
            .zip(padded[start..start + N_FFT].iter().zip(window.iter()))
        {
            *dst = sample * weight;
        }
        radix2_fft(&frame, &mut spectrum);
        let dest = &mut out[index * bins..(index + 1) * bins];
        for bin in 0..bins {
            let re = spectrum[bin];
            let im = if bin == 0 || bin == bins - 1 {
                0.0
            } else {
                spectrum[N_FFT - bin]
            };
            // power=1 → magnitude
            dest[bin] = (re * re + im * im).sqrt();
        }
    }
    Ok(out)
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = std::f32::consts::TAU * i as f32 / n as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// In-place-ish radix-2 real→complex via packed real FFT (Cooley–Tukey).
/// `time` length N; `freq` length N with Re[0..N/2], Im[1..N/2-1] mirrored.
fn radix2_fft(time: &[f32], freq: &mut [f32]) {
    let n = time.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(freq.len(), n);
    // bit-reverse copy
    let mut j = 0usize;
    for i in 0..n {
        freq[i] = 0.0;
    }
    let mut real = time.to_vec();
    let mut imag = vec![0.0f32; n];
    for i in 0..n {
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -std::f32::consts::TAU / len as f32;
        let (w_re_step, w_im_step) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for offset in 0..half {
                let u_re = real[start + offset];
                let u_im = imag[start + offset];
                let v_re = real[start + offset + half] * w_re - imag[start + offset + half] * w_im;
                let v_im = real[start + offset + half] * w_im + imag[start + offset + half] * w_re;
                real[start + offset] = u_re + v_re;
                imag[start + offset] = u_im + v_im;
                real[start + offset + half] = u_re - v_re;
                imag[start + offset + half] = u_im - v_im;
                let next_re = w_re * w_re_step - w_im * w_im_step;
                w_im = w_re * w_im_step + w_im * w_re_step;
                w_re = next_re;
            }
        }
        len *= 2;
    }
    // pack: for magnitude we only need re/im per bin; store re in freq[0..],
    // and use freq[N-k] as im for k via the caller convention above.
    for i in 0..n {
        freq[i] = real[i];
    }
    // stash imag in a way magnitude can recover: overwrite upper half imag
    // Caller reads re=freq[bin], im=freq[N-bin] for bin in 1..N/2-1.
    // So put imag[k] into freq[N-k] for k=1..N/2-1, but that destroys re of
    // mirrored bins which we don't need for real signals' magnitude (bins
    // N/2+1.. are conjugates). Safe for our magnitude loop.
    for k in 1..n / 2 {
        freq[n - k] = imag[k];
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn slaney_mel_filterbank() -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let f_min = 0.0f32;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(F_MAX);
    let mut mel_points = Vec::with_capacity(N_MELS + 2);
    for i in 0..N_MELS + 2 {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|i| i as f32 * TARGET_RATE as f32 / N_FFT as f32)
        .collect();
    let mut weights = vec![0.0f32; N_MELS * bins];
    for mel_bin in 0..N_MELS {
        let left = mel_points[mel_bin];
        let center = mel_points[mel_bin + 1];
        let right = mel_points[mel_bin + 2];
        let row = &mut weights[mel_bin * bins..(mel_bin + 1) * bins];
        for (bin, freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / (center - left);
            let upper = (right - freq) / (right - center);
            row[bin] = lower.max(0.0).min(upper.max(0.0));
        }
        // Slaney norm: peak-normalized then area-normalized by bandwidth
        let enorm = 2.0 / (right - left);
        for sample in row.iter_mut() {
            *sample *= enorm;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::{channel_log_mel, resample_stereo_interleaved, stereo_log_mel, N_MELS};

    #[test]
    fn silence_produces_finite_mel_frames() {
        let wave = vec![0.0f32; 16_000];
        let mel = channel_log_mel(&wave).unwrap();
        assert_eq!(mel.len() % N_MELS, 0);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stereo_resample_and_mel_shapes() {
        let src: Vec<f32> = (0..48_000 * 2)
            .map(|i| ((i / 2) as f32 * 0.001).sin())
            .collect();
        let resampled = resample_stereo_interleaved(&src, 48_000).unwrap();
        assert_eq!(resampled.len() / 2, 16_000);
        let (planar, time) = stereo_log_mel(&resampled).unwrap();
        assert_eq!(planar.len(), 2 * time * N_MELS);
        assert!(time > 50);
    }
}
