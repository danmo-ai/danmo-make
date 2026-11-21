//! LongCat-Avatar audio conditioning: Whisper mel → pooled feature groups →
//! per-frame windowed tokens `[B, T, 5, 1280]`.
//!
//! The Whisper encoder (neutral `common::whisper`) emits the 33 hidden states
//! (embedding + 32 layers) at 50 Hz. This family groups them into 5 pooled
//! bands, resamples 50 Hz → video fps with linear interpolation, and builds
//! the 5-frame sliding windows the DiT's `audio_proj` consumes.
//!
//! Mel extraction follows `WhisperFeatureExtractor`: 16 kHz mono, a centered
//! 400-sample Hann STFT / 160 hop with reflect padding, Slaney mel filters,
//! log10 power, an 80 dB dynamic-range clamp, and `(x + 4) / 4` scaling.

use danmo_mlx::{Array, Error, Result, Stream};

pub const MEL_BINS: usize = 128;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const SAMPLE_RATE: u32 = 16_000;
const ENC_FPS: f32 = 50.0;
const AUDIO_WINDOW: usize = 5;
const ENCODER_CHUNK: usize = 3000;
const WHISPER_LAYERS: usize = 32;

/// Mean-pool the 33 hidden states into 5 bands: layers 0-7, 8-15, 16-23,
/// 24-31, and the final layer 32. Returns `[B, T_enc, 5, d_model]`.
pub fn group_pool_hidden_states(hidden: &[Array], stream: &Stream) -> Result<Array> {
    if hidden.len() != (WHISPER_LAYERS + 1) as usize {
        return Err(Error::new(
            "longcat_avatar audio",
            format!("Whisper hidden states: expected 33 layers, got {}", hidden.len()),
        ));
    }
    let mut bands: Vec<Array> = Vec::with_capacity(5);
    for band in 0..4 {
        let stacked = Array::stack(&hidden[band * 8..band * 8 + 8], 0, stream)?;
        bands.push(stacked.mean_axes(&[0], false, stream)?);
    }
    bands.push(hidden[WHISPER_LAYERS].clone());
    Array::stack(&bands, 2, stream)
}

/// Chunked encode: Whisper runs on ≤3000-frame mel chunks; bands concat on T.
pub fn encode_audio_to_groups(
    encoder: &crate::common::whisper::WhisperEncoder,
    mel: &Array,
    stream: &Stream,
) -> Result<Array> {
    let t_mel = mel.shape()[2] as usize;
    if t_mel == 0 {
        return Err(Error::new(
            "longcat_avatar audio",
            "mel spectrogram has zero frames",
        ));
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < t_mel {
        let stop = (start + ENCODER_CHUNK).min(t_mel);
        let chunk = mel.slice(
            &[0, 0, start as i32],
            &[mel.shape()[0], mel.shape()[1], stop as i32],
            &[1, 1, 1],
            stream,
        )?;
        let hidden = encoder.encode_all_hidden_states(&chunk, stream)?;
        chunks.push(group_pool_hidden_states(&hidden, stream)?);
        start = stop;
    }
    if chunks.len() == 1 {
        Ok(chunks.remove(0))
    } else {
        Array::concatenate_axis(&chunks, 1, stream)
    }
}

/// `[B, T_enc, 5, d]` groups → `[B, T_video, W=5, 5, d]` windowed per-frame
/// features at `fps` with linear resampling from `ENC_FPS`.
pub fn build_avatar_audio_embeddings(
    groups: &Array,
    fps: u32,
    stream: &Stream,
) -> Result<Array> {
    let shape = groups.shape();
    if shape.len() != 4 {
        return Err(Error::new(
            "longcat_avatar audio",
            format!("expected [B, T_enc, 5, d] groups, got {shape:?}"),
        ));
    }
    let (b, t_audio, g, d) = (shape[0], shape[1] as usize, shape[2], shape[3]);
    let flat = groups.reshape(&[b, t_audio as i32, g * d], stream)?;
    // int(t_enc / enc_fps * fps) rounds toward zero on float division —
    // keep the exact float arithmetic of the baseline (no `.round()`).
    let target_len = ((t_audio as f32) / ENC_FPS * fps as f32) as usize;
    debug_assert!(target_len > 0, "audio too short for one video frame");
    if target_len == 0 {
        return Err(Error::new(
            "longcat_avatar audio",
            "audio is too short to produce one video frame of features",
        ));
    }
    let resampled = linear_interpolate(&flat, target_len, stream)?;
    let per_frame = resampled.reshape(&[b, target_len as i32, g, d], stream)?;
    let half = (AUDIO_WINDOW / 2) as i32;
    let first = per_frame.slice(&[0, 0, 0, 0], &[b, 1, g, d], &[1, 1, 1, 1], stream)?;
    let last = per_frame.slice(
        &[0, (target_len - 1) as i32, 0, 0],
        &[b, target_len as i32, g, d],
        &[1, 1, 1, 1],
        stream,
    )?;
    let mut padded_parts = Vec::new();
    for _ in 0..half {
        padded_parts.push(first.clone());
    }
    padded_parts.push(per_frame.clone());
    for _ in 0..half {
        padded_parts.push(last.clone());
    }
    let padded = Array::concatenate_axis(&padded_parts, 1, stream)?;
    let mut windows = Vec::with_capacity(AUDIO_WINDOW);
    for offset in 0..AUDIO_WINDOW as i32 {
        windows.push(padded.slice(
            &[0, offset, 0, 0],
            &[b, offset + target_len as i32, g, d],
            &[1, 1, 1, 1],
            stream,
        )?);
    }
    let stacked = Array::stack(&windows, 2, stream)?;
    let out_shape = stacked.shape();
    if out_shape[1] != target_len as i32 || out_shape[2] != AUDIO_WINDOW as i32 {
        return Err(Error::new(
            "longcat_avatar audio",
            format!(
                "windowed audio is {out_shape:?}, expected [B, {target_len}, {AUDIO_WINDOW}, ...]"
            ),
        ));
    }
    Ok(stacked)
}

/// Trim or last-frame-pad `[B, T, W, 5, d]` audio to `required` video frames.
pub fn align_audio_frames(audio: &Array, required: usize, stream: &Stream) -> Result<Array> {
    let shape = audio.shape();
    let have = shape[1] as usize;
    let out = if have == required {
        audio.clone()
    } else if have > required {
        let mut stop = shape.to_vec();
        stop[1] = required as i32;
        audio.slice(&vec![0; shape.len()], &stop, &vec![1; shape.len()], stream)?
    } else {
        let shortfall = required - have;
        let last = audio.slice(
            &[0, (have - 1) as i32, 0, 0, 0],
            &shape,
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let mut parts = vec![audio.clone()];
        for _ in 0..shortfall {
            parts.push(last.clone());
        }
        Array::concatenate_axis(&parts, 1, stream)?
    };
    let got = out.shape()[1] as usize;
    if got != required {
        return Err(Error::new(
            "longcat_avatar audio",
            format!("aligned audio has {got} frames, required {required} (had {have})"),
        ));
    }
    Ok(out)
}

/// Linear interpolation along axis 1 (`[B, T, D]` → `[B, out, D]`), matching
/// the baseline's `linear_interpolate_features`.
fn linear_interpolate(features: &Array, out_len: usize, stream: &Stream) -> Result<Array> {
    let shape = features.shape();
    let t = shape[1] as usize;
    if out_len == t {
        return Ok(features.clone());
    }
    let mut gathered = Vec::with_capacity(out_len);
    let width = shape[2];
    for index in 0..out_len {
        let source = index as f32 * (t - 1) as f32 / (out_len - 1).max(1) as f32;
        let lo = source.floor().min((t - 1) as f32) as i32;
        let hi = (lo + 1).min(t as i32 - 1);
        let frac = source - lo as f32;
        // Single-row slices: start..start+1 on the time axis.
        let lo_row = features.slice(&[0, lo, 0], &[shape[0], lo + 1, width], &[1, 1, 1], stream)?;
        let hi_row = features.slice(&[0, hi, 0], &[shape[0], hi + 1, width], &[1, 1, 1], stream)?;
        let row = Array::add(
            &Array::multiply(&lo_row, &Array::scalar_f32(1.0 - frac)?, stream)?,
            &Array::multiply(&hi_row, &Array::scalar_f32(frac)?, stream)?,
            stream,
        )?;
        gathered.push(row);
    }
    Array::concatenate_axis(&gathered, 1, stream)
}

// ---------------------------------------------------------------------------
// Whisper mel (128-bin log-mel, 16 kHz mono)
// ---------------------------------------------------------------------------

/// Compute the Whisper log-mel spectrogram `[1, 128, T]` from mono f32 PCM.
pub fn whisper_log_mel(samples: &[f32], _stream: &Stream) -> Result<Array> {
    if samples.len() < HOP {
        return Err(Error::new(
            "longcat_avatar audio",
            "voice track is too short for one Whisper frame",
        ));
    }
    let filters = mel_filters();
    let frames = samples.len() / HOP;
    let window = hann_window();
    let bins = N_FFT / 2 + 1;
    let mut mel = vec![0f32; MEL_BINS * frames];
    let mut frame_buf = vec![0f32; N_FFT];
    for frame in 0..frames {
        for tap in 0..N_FFT {
            let index = (frame * HOP + tap) as isize - (N_FFT / 2) as isize;
            frame_buf[tap] = samples[reflect_index(index, samples.len())] * window[tap];
        }
        let power = rfft_power(&frame_buf, bins);
        for (bin, p) in power.iter().enumerate() {
            for band in 0..MEL_BINS {
                mel[band * frames + frame] += filters[band * bins + bin] * p;
            }
        }
    }
    let mut max_value = f32::NEG_INFINITY;
    for value in &mut mel {
        *value = value.max(1e-10).log10();
        max_value = max_value.max(*value);
    }
    let floor = max_value - 8.0;
    for value in &mut mel {
        *value = (value.max(floor) + 4.0) / 4.0;
    }
    Array::from_f32_slice(&mel, &[1, MEL_BINS as i32, frames as i32])
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len - 2;
    let reflected = index.rem_euclid(period as isize) as usize;
    if reflected < len {
        reflected
    } else {
        period - reflected
    }
}

fn hann_window() -> Vec<f32> {
    (0..N_FFT)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / N_FFT as f32).cos())
        .collect()
}

fn rfft_power(frame: &[f32], bins: usize) -> Vec<f32> {
    let mut power = Vec::with_capacity(bins);
    for k in 0..bins {
        let (mut re, mut im) = (0f32, 0f32);
        for (n, sample) in frame.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / frame.len() as f32;
            re += sample * angle.cos();
            im += sample * angle.sin();
        }
        power.push(re * re + im * im);
    }
    power
}

/// Slaney-scale, area-normalized 128-band filterbank used by Whisper.
fn mel_filters() -> Vec<f32> {
    fn hz_to_mel(hz: f64) -> f64 {
        if hz >= 1000.0 {
            15.0 + (hz / 1000.0).ln() * (27.0 / 6.4f64.ln())
        } else {
            3.0 * hz / 200.0
        }
    }
    fn mel_to_hz(mel: f64) -> f64 {
        if mel >= 15.0 {
            1000.0 * ((6.4f64.ln() / 27.0) * (mel - 15.0)).exp()
        } else {
            200.0 * mel / 3.0
        }
    }

    let bins = N_FFT / 2 + 1;
    let max_mel = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let filter_freqs: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| mel_to_hz(max_mel * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let mut filters = vec![0f32; MEL_BINS * bins];
    for band in 0..MEL_BINS {
        let (lo, center, hi) = (
            filter_freqs[band],
            filter_freqs[band + 1],
            filter_freqs[band + 2],
        );
        let normalization = 2.0 / (hi - lo);
        for bin in 0..bins {
            let freq = bin as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            let up = (freq - lo) / (center - lo);
            let down = (hi - freq) / (hi - center);
            filters[band * bins + bin] = (up.min(down).max(0.0) * normalization) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::{mel_filters, reflect_index, whisper_log_mel, N_FFT};

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filters();
        assert_eq!(filters.len(), 128 * (N_FFT / 2 + 1));
        assert!(filters.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(filters.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn log_mel_is_finite() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let samples = vec![0.01f32; 1600];
        let mel = whisper_log_mel(&samples, &stream).unwrap();
        assert_eq!(mel.shape()[1], 128);
        assert_eq!(mel.shape()[2], 10);
        let values = mel.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reflect_padding_does_not_repeat_the_edge_sample() {
        assert_eq!(reflect_index(-1, 1600), 1);
        assert_eq!(reflect_index(-200, 1600), 200);
        assert_eq!(reflect_index(1600, 1600), 1598);
        assert_eq!(reflect_index(1799, 1600), 1399);
    }
}
