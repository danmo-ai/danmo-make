//! Family-neutral audio/video latent geometry and RoPE position tables.
//!
//! Video tokens use frame → row → column order. Audio and video position
//! values are seconds for the time axis and pixel midpoints for spatial axes.
//! Tensor patchification, masks, and model execution stay with the caller.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLatentSpec {
    pub temporal_compression: u32,
    pub spatial_compression: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioLatentSpec {
    pub downsample_factor: u32,
    pub hop_length: u32,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLatentShape {
    pub frames: u32,
    pub height: u32,
    pub width: u32,
}

impl VideoLatentShape {
    pub fn token_count(self) -> Result<usize, String> {
        let frames = self.frames as usize;
        let height = self.height as usize;
        let width = self.width as usize;
        frames
            .checked_mul(height)
            .and_then(|tokens| tokens.checked_mul(width))
            .ok_or_else(|| "video latent token count overflows usize".into())
    }
}

pub fn video_latent_shape(
    num_frames: u32,
    height: u32,
    width: u32,
    spec: VideoLatentSpec,
) -> Result<VideoLatentShape, String> {
    if num_frames == 0 {
        return Err("video frame count must be positive".into());
    }
    if spec.temporal_compression == 0 || spec.spatial_compression == 0 {
        return Err("video latent compression factors must be positive".into());
    }
    if height == 0 || width == 0 {
        return Err("video pixel dimensions must be positive".into());
    }
    if height % spec.spatial_compression != 0 || width % spec.spatial_compression != 0 {
        return Err(format!(
            "video size {width}x{height} is not divisible by spatial compression {}",
            spec.spatial_compression
        ));
    }
    let temporal = u64::from(spec.temporal_compression);
    let frames = (u64::from(num_frames) + temporal - 1) / temporal;
    let frames =
        u32::try_from(frames).map_err(|_| "video latent frame count exceeds u32".to_string())?;
    let shape = VideoLatentShape {
        frames,
        height: height / spec.spatial_compression,
        width: width / spec.spatial_compression,
    };
    shape.token_count()?;
    Ok(shape)
}

pub fn audio_token_count(
    num_video_frames: u32,
    frame_rate: f32,
    spec: AudioLatentSpec,
) -> Result<usize, String> {
    validate_audio_spec(spec)?;
    if !frame_rate.is_finite() || frame_rate <= 0.0 {
        return Err(format!(
            "audio frame rate {frame_rate} must be positive and finite"
        ));
    }
    let rate = f64::from(spec.sample_rate)
        / f64::from(spec.hop_length)
        / f64::from(spec.downsample_factor);
    let count = (f64::from(num_video_frames) / f64::from(frame_rate) * rate).round();
    if !count.is_finite() || count < 0.0 || count > usize::MAX as f64 {
        return Err(format!("audio token count {count} is not representable"));
    }
    Ok(count as usize)
}

pub fn video_positions(
    shape: VideoLatentShape,
    frame_rate: f32,
    spec: VideoLatentSpec,
) -> Result<Vec<[f32; 3]>, String> {
    if !frame_rate.is_finite() || frame_rate <= 0.0 {
        return Err(format!(
            "video frame rate {frame_rate} must be positive and finite"
        ));
    }
    if spec.temporal_compression == 0 || spec.spatial_compression == 0 {
        return Err("video latent compression factors must be positive".into());
    }
    let temporal = spec.temporal_compression as f32;
    let spatial = spec.spatial_compression as f32;
    let mut positions = Vec::with_capacity(shape.token_count()?);
    for frame in 0..shape.frames {
        let index = frame as f32;
        let start = (index * temporal + 1.0 - temporal).max(0.0);
        let end = ((index + 1.0) * temporal + 1.0 - temporal).max(0.0);
        let time = (start + end) / 2.0 / frame_rate;
        for row in 0..shape.height {
            let y = row as f32 * spatial + spatial / 2.0;
            for column in 0..shape.width {
                let x = column as f32 * spatial + spatial / 2.0;
                positions.push([time, y, x]);
            }
        }
    }
    Ok(positions)
}

pub fn audio_positions(num_tokens: usize, spec: AudioLatentSpec) -> Result<Vec<f32>, String> {
    validate_audio_spec(spec)?;
    let downsample = spec.downsample_factor as f32;
    let seconds_per_hop = spec.hop_length as f32 / spec.sample_rate as f32;
    let mut positions = Vec::with_capacity(num_tokens);
    for token in 0..num_tokens {
        let index = token as f32;
        let start = (index * downsample + 1.0 - downsample).max(0.0) * seconds_per_hop;
        let end = ((index + 1.0) * downsample + 1.0 - downsample).max(0.0) * seconds_per_hop;
        positions.push((start + end) / 2.0);
    }
    Ok(positions)
}

fn validate_audio_spec(spec: AudioLatentSpec) -> Result<(), String> {
    if spec.downsample_factor == 0 || spec.hop_length == 0 || spec.sample_rate == 0 {
        return Err("audio latent downsample, hop length, and sample rate must be positive".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        audio_positions, audio_token_count, video_latent_shape, video_positions, AudioLatentSpec,
        VideoLatentSpec,
    };

    const VIDEO: VideoLatentSpec = VideoLatentSpec {
        temporal_compression: 8,
        spatial_compression: 32,
    };
    const AUDIO: AudioLatentSpec = AudioLatentSpec {
        downsample_factor: 4,
        hop_length: 160,
        sample_rate: 16_000,
    };

    #[test]
    fn geometry_uses_causal_video_frames_and_25hz_audio() {
        let shape = video_latent_shape(97, 768, 512, VIDEO).unwrap();
        assert_eq!(shape.frames, 13);
        assert_eq!((shape.height, shape.width), (24, 16));
        assert_eq!(shape.token_count().unwrap(), 4_992);
        assert_eq!(audio_token_count(97, 24.0, AUDIO).unwrap(), 101);
    }

    #[test]
    fn video_positions_use_causal_time_row_column_midpoints() {
        let shape = video_latent_shape(9, 32, 64, VIDEO).unwrap();
        let positions = video_positions(shape, 24.0, VIDEO).unwrap();
        assert_eq!(positions.len(), 4);
        assert_eq!(positions[0], [0.5 / 24.0, 16.0, 16.0]);
        assert_eq!(positions[1], [0.5 / 24.0, 16.0, 48.0]);
        assert_eq!(positions[2], [5.0 / 24.0, 16.0, 16.0]);
        assert_eq!(positions[3], [5.0 / 24.0, 16.0, 48.0]);
    }

    #[test]
    fn audio_positions_use_causal_midpoints() {
        let positions = audio_positions(3, AUDIO).unwrap();
        assert_eq!(positions.len(), 3);
        for (got, want) in positions.iter().zip([0.005, 0.03, 0.07]) {
            assert!((got - want).abs() < 1e-7, "{got} vs {want}");
        }
        let error = audio_positions(
            1,
            AudioLatentSpec {
                hop_length: 0,
                ..AUDIO
            },
        )
        .err()
        .expect("zero hop must fail");
        assert!(error.contains("hop length"), "{error}");
    }

    #[test]
    fn invalid_geometry_and_rates_fail_loud() {
        assert!(video_latent_shape(97, 768, 520, VIDEO)
            .err()
            .unwrap()
            .contains("not divisible"));
        assert!(audio_token_count(97, f32::NAN, AUDIO)
            .err()
            .unwrap()
            .contains("positive and finite"));
        assert!(
            video_positions(video_latent_shape(9, 32, 32, VIDEO).unwrap(), 0.0, VIDEO,)
                .err()
                .unwrap()
                .contains("positive and finite")
        );
    }
}
