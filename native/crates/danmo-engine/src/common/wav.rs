//! Minimal PCM16 WAV container helpers for generated interleaved audio.

use std::io;
use std::path::Path;

/// Read a PCM16 WAV into interleaved `f32` samples in `[-1, 1]` plus sample rate.
///
/// Mono files are duplicated to stereo. Non-PCM containers fail loud.
pub fn read_pcm16_interleaved(wav: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if wav.len() < 44 {
        return Err("WAV is shorter than a RIFF header".into());
    }
    if &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("WAV is not a RIFF/WAVE file".into());
    }
    let mut offset = 12usize;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut data: Option<&[u8]> = None;
    while offset + 8 <= wav.len() {
        let id = &wav[offset..offset + 4];
        let size = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body = offset + 8;
        let next = body + size + (size % 2);
        if body + size > wav.len() {
            return Err("WAV chunk overruns file".into());
        }
        if id == b"fmt " {
            if size < 16 {
                return Err("WAV fmt chunk is too short".into());
            }
            let format = u16::from_le_bytes(wav[body..body + 2].try_into().unwrap());
            if format != 1 {
                return Err(format!("WAV format {format} is not PCM"));
            }
            channels = u16::from_le_bytes(wav[body + 2..body + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(wav[body + 4..body + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(wav[body + 14..body + 16].try_into().unwrap());
            if bits != 16 {
                return Err(format!("WAV bits {bits} is not PCM16"));
            }
            if channels != 1 && channels != 2 {
                return Err(format!("WAV channels {channels} must be 1 or 2"));
            }
        } else if id == b"data" {
            data = Some(&wav[body..body + size]);
        }
        offset = next;
    }
    let Some(pcm) = data else {
        return Err("WAV has no data chunk".into());
    };
    if sample_rate == 0 || channels == 0 {
        return Err("WAV is missing a fmt chunk".into());
    }
    if pcm.len() % 2 != 0 {
        return Err("WAV PCM16 data length is odd".into());
    }
    let samples = pcm.len() / 2;
    let mut out = Vec::with_capacity(if channels == 1 { samples * 2 } else { samples });
    if channels == 1 {
        for i in 0..samples {
            let s = i16::from_le_bytes(pcm[i * 2..i * 2 + 2].try_into().unwrap()) as f32 / 32768.0;
            out.push(s);
            out.push(s);
        }
    } else {
        for i in 0..samples {
            let s = i16::from_le_bytes(pcm[i * 2..i * 2 + 2].try_into().unwrap()) as f32 / 32768.0;
            out.push(s);
        }
    }
    Ok((out, sample_rate))
}

/// Resample mono PCM with a 32-tap Blackman-windowed sinc low-pass filter.
pub fn resample_mono(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, String> {
    if source_rate == 0 || target_rate == 0 {
        return Err("audio sample rates must be positive".into());
    }
    if samples.is_empty() || source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    const HALF_TAPS: i32 = 16;
    let ratio = source_rate as f64 / target_rate as f64;
    let cutoff = (target_rate as f64 / source_rate as f64).min(1.0);
    let output_len = ((samples.len() as u64 * target_rate as u64 + source_rate as u64 / 2)
        / source_rate as u64) as usize;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = index as f64 * ratio;
        let left = position.floor() as i32;
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for offset in -HALF_TAPS + 1..=HALF_TAPS {
            let source_index = left + offset;
            let distance = position - source_index as f64;
            let window = 0.42
                + 0.5 * (std::f64::consts::PI * distance / HALF_TAPS as f64).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * distance / HALF_TAPS as f64).cos();
            let x = distance * cutoff;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let weight = cutoff * sinc * window;
            let sample_index = source_index.clamp(0, samples.len() as i32 - 1) as usize;
            sum += samples[sample_index] as f64 * weight;
            weight_sum += weight;
        }
        if weight_sum.abs() > 1e-12 {
            output.push((sum / weight_sum) as f32);
        } else {
            output.push(0.0);
        }
    }
    Ok(output)
}

pub fn write_pcm16(
    path: &Path,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
) -> io::Result<()> {
    if channels == 0 || samples.len() % channels as usize != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "interleaved PCM samples must have a nonzero, exact channel count",
        ));
    }
    let data_len = u32::try_from(samples.len().saturating_mul(2)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "PCM WAV exceeds the RIFF size limit",
        )
    })?;
    let riff_len = 36u32.checked_add(data_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "PCM WAV exceeds the RIFF size limit",
        )
    })?;
    let mut bytes = Vec::with_capacity(riff_len as usize + 8);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_len.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate
        .saturating_mul(u32::from(channels))
        .saturating_mul(2);
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&(channels.saturating_mul(2)).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    #[test]
    fn pcm16_waveform_layout() {
        let path = std::env::temp_dir().join(format!("danmo-wav-{}.wav", std::process::id()));
        super::write_pcm16(&path, &[0.0, 1.0, -1.0, 0.5], 2, 48_000).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        assert_eq!(bytes.len(), 52);
    }

    #[test]
    fn resample_mono_preserves_passband_waveform() {
        let source: Vec<f32> = (0..48_000)
            .map(|index| (2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 48_000.0).sin())
            .collect();
        let output = super::resample_mono(&source, 48_000, 16_000).unwrap();
        assert_eq!(output.len(), 16_000);
        let mut error = 0.0f32;
        for (index, value) in output.iter().enumerate().skip(32).take(output.len() - 64) {
            let expected = (2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 16_000.0).sin();
            error += (value - expected).abs();
        }
        assert!(error / ((output.len() - 64) as f32) < 0.01);
        assert!(super::resample_mono(&source, 0, 16_000).is_err());
    }
}
