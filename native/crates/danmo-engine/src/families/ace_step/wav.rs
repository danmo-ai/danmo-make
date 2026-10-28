//! 48 kHz stereo WAV writer/reader for ACE-Step decode / cover encode.

use crate::weights::WeightError;

pub const SAMPLE_RATE: u32 = 48_000;

/// Planar `[L0..Ln, R0..Rn]` → little-endian PCM16 WAV.
pub fn write_planar_stereo_pcm16(left_right: &[f32], sample_rate: u32) -> Result<Vec<u8>, WeightError> {
    if left_right.len() % 2 != 0 {
        return Err(WeightError::NotBuilt(format!(
            "stereo planar buffer length {} is not even",
            left_right.len()
        )));
    }
    let n = left_right.len() / 2;
    let mut pcm = Vec::with_capacity(n * 2 * 2);
    for i in 0..n {
        let l = left_right[i].clamp(-1.0, 1.0);
        let r = left_right[n + i].clamp(-1.0, 1.0);
        pcm.extend_from_slice(&(quantize(l)).to_le_bytes());
        pcm.extend_from_slice(&(quantize(r)).to_le_bytes());
    }
    Ok(wrap_wav(&pcm, sample_rate, 2))
}

/// Interleaved `[L0, R0, L1, R1, …]` → planar then WAV.
pub fn write_interleaved_stereo_pcm16(
    interleaved: &[f32],
    sample_rate: u32,
) -> Result<Vec<u8>, WeightError> {
    if interleaved.len() % 2 != 0 {
        return Err(WeightError::NotBuilt(format!(
            "stereo interleaved buffer length {} is not even",
            interleaved.len()
        )));
    }
    let n = interleaved.len() / 2;
    let mut planar = vec![0f32; interleaved.len()];
    for i in 0..n {
        planar[i] = interleaved[i * 2];
        planar[n + i] = interleaved[i * 2 + 1];
    }
    write_planar_stereo_pcm16(&planar, sample_rate)
}

fn quantize(x: f32) -> i16 {
    (x * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Read PCM16 WAV → interleaved stereo f32. Mono is duplicated to stereo.
/// Resampling is not built: sample rate must already be 48 kHz (or caller accepts mismatch).
pub fn read_pcm16_interleaved(wav: &[u8]) -> Result<(Vec<f32>, u32), WeightError> {
    if wav.len() < 44 {
        return Err(WeightError::NotBuilt(
            "ACE-Step WAV is shorter than a RIFF header".into(),
        ));
    }
    if &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err(WeightError::NotBuilt(
            "ACE-Step WAV is not a RIFF/WAVE file".into(),
        ));
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
            return Err(WeightError::NotBuilt(
                "ACE-Step WAV chunk overruns file".into(),
            ));
        }
        if id == b"fmt " {
            if size < 16 {
                return Err(WeightError::NotBuilt(
                    "ACE-Step WAV fmt chunk is too short".into(),
                ));
            }
            let format = u16::from_le_bytes(wav[body..body + 2].try_into().unwrap());
            if format != 1 {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step WAV format {format} is not PCM"
                )));
            }
            channels = u16::from_le_bytes(wav[body + 2..body + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(wav[body + 4..body + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(wav[body + 14..body + 16].try_into().unwrap());
            if bits != 16 {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step WAV bits {bits} is not PCM16"
                )));
            }
            if channels != 1 && channels != 2 {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step WAV channels {channels} must be 1 or 2"
                )));
            }
        } else if id == b"data" {
            data = Some(&wav[body..body + size]);
        }
        offset = next;
    }
    let Some(pcm) = data else {
        return Err(WeightError::NotBuilt(
            "ACE-Step WAV has no data chunk".into(),
        ));
    };
    if sample_rate == 0 || channels == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step WAV is missing a fmt chunk".into(),
        ));
    }
    if pcm.len() % 2 != 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step WAV PCM16 data length is odd".into(),
        ));
    }
    let samples = pcm.len() / 2;
    let mut out = Vec::with_capacity(if channels == 1 {
        samples * 2
    } else {
        samples
    });
    if channels == 1 {
        for i in 0..samples {
            let s =
                i16::from_le_bytes(pcm[i * 2..i * 2 + 2].try_into().unwrap()) as f32 / 32768.0;
            out.push(s);
            out.push(s);
        }
    } else {
        for i in 0..samples {
            let s =
                i16::from_le_bytes(pcm[i * 2..i * 2 + 2].try_into().unwrap()) as f32 / 32768.0;
            out.push(s);
        }
    }
    Ok((out, sample_rate))
}

fn wrap_wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;
    let data_len = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::{
        read_pcm16_interleaved, write_interleaved_stereo_pcm16, write_planar_stereo_pcm16,
        SAMPLE_RATE,
    };

    #[test]
    fn wav_header_matches_pcm16_stereo() {
        let planar = vec![0.0f32; 8];
        let wav = write_planar_stereo_pcm16(&planar, SAMPLE_RATE).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(wav[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        let interleaved = [0.5f32, -0.5, 0.25, -0.25];
        let wav2 = write_interleaved_stereo_pcm16(&interleaved, SAMPLE_RATE).unwrap();
        assert!(wav2.len() > 44);
        let (roundtrip, sr) = read_pcm16_interleaved(&wav2).unwrap();
        assert_eq!(sr, SAMPLE_RATE);
        assert_eq!(roundtrip.len(), 4);
        assert!((roundtrip[0] - 0.5).abs() < 0.01);
    }
}
