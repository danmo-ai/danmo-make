use crate::error::EngineError;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct AudioEncoding {
    pub format: String,
    pub denoise: String,
}

impl AudioEncoding {
    pub fn new(format: &str, denoise: &str) -> Result<Self, EngineError> {
        let format = format.trim().to_ascii_lowercase();
        if !matches!(format.as_str(), "wav" | "flac" | "mp3" | "opus" | "aac") {
            return Err(EngineError::bad_request(format!(
                "ACE-Step audio_format '{format}' is not supported (use wav, flac, mp3, opus, or aac)"
            )));
        }
        let denoise = denoise.trim().to_ascii_lowercase();
        if !matches!(denoise.as_str(), "off" | "mild" | "strong") {
            return Err(EngineError::bad_request(format!(
                "ACE-Step denoise '{denoise}' is not supported (use off, mild, or strong)"
            )));
        }
        Ok(Self { format, denoise })
    }

    pub fn mime(&self) -> String {
        match self.format.as_str() {
            "mp3" => "audio/mpeg".into(),
            other => format!("audio/{other}"),
        }
    }
}

pub struct EncodedAudio {
    pub bytes: Vec<u8>,
    pub extension: String,
    pub mime: String,
}

pub fn encode_wav(
    wav: &[u8],
    sample_rate: u32,
    encoding: &AudioEncoding,
) -> Result<EncodedAudio, EngineError> {
    if encoding.format == "wav" && encoding.denoise == "off" {
        return Ok(EncodedAudio {
            bytes: wav.to_vec(),
            extension: "wav".into(),
            mime: encoding.mime(),
        });
    }
    let ffmpeg = find_ffmpeg()?;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("danmo-ace-codec-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| EngineError::Persist(error.to_string()))?;
    let input = dir.join("input.wav");
    let output = dir.join(format!("output.{}", encoding.format));
    std::fs::write(&input, wav).map_err(|error| EngineError::Persist(error.to_string()))?;
    let result = run_ffmpeg(&ffmpeg, &input, &output, sample_rate, encoding);
    let bytes = result.and_then(|()| {
        std::fs::read(&output).map_err(|error| EngineError::Persist(error.to_string()))
    });
    let _ = std::fs::remove_dir_all(&dir);
    Ok(EncodedAudio {
        bytes: bytes?,
        extension: encoding.format.clone(),
        mime: encoding.mime(),
    })
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn find_ffmpeg() -> Result<PathBuf, EngineError> {
    let path = std::env::var_os("DANQING_FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let output = Command::new(&path)
        .arg("-version")
        .output()
        .map_err(|error| {
            EngineError::bad_request(format!(
                "ACE-Step audio encoding requires ffmpeg on PATH: {error}"
            ))
        })?;
    if !output.status.success() {
        return Err(EngineError::bad_request(
            "ACE-Step audio encoding requires a working ffmpeg on PATH",
        ));
    }
    Ok(path)
}

fn run_ffmpeg(
    ffmpeg: &Path,
    input: &Path,
    output: &Path,
    sample_rate: u32,
    encoding: &AudioEncoding,
) -> Result<(), EngineError> {
    let mut command = Command::new(ffmpeg);
    command
        .arg("-nostdin")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-ar")
        .arg(sample_rate.to_string());
    match encoding.denoise.as_str() {
        "off" => {}
        "mild" => {
            command.arg("-af").arg("afftdn=nr=6:nf=-45:tn=1");
        }
        "strong" => {
            command.arg("-af").arg("afftdn=nr=10:nf=-42:tn=1");
        }
        _ => unreachable!(),
    }
    match encoding.format.as_str() {
        "wav" => {
            command.arg("-c:a").arg("pcm_s16le");
        }
        "flac" => {
            command.arg("-c:a").arg("flac");
        }
        "mp3" => {
            command.arg("-c:a").arg("libmp3lame");
        }
        "opus" => {
            command.arg("-c:a").arg("libopus");
        }
        "aac" => {
            command.arg("-c:a").arg("aac");
        }
        _ => unreachable!(),
    }
    let output_process = command
        .arg(output)
        .output()
        .map_err(|error| EngineError::Persist(error.to_string()))?;
    if !output_process.status.success() {
        return Err(EngineError::Persist(format!(
            "ffmpeg {} / {} failed: {}",
            encoding.format,
            encoding.denoise,
            String::from_utf8_lossy(&output_process.stderr).trim()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_validation_and_wav_passthrough() {
        assert!(AudioEncoding::new("bad", "off").is_err());
        assert!(AudioEncoding::new("wav", "bad").is_err());
        let encoding = AudioEncoding::new("wav", "off").unwrap();
        let encoded = encode_wav(b"RIFF", 48_000, &encoding).unwrap();
        assert_eq!(encoded.bytes, b"RIFF");
        assert_eq!(encoded.extension, "wav");
    }

    #[test]
    fn ffmpeg_transcodes_and_denoises_real_audio() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let frames = 4_800i64;
        let mut audio = vec![0.0; (frames * 2) as usize];
        for index in 0..frames as usize {
            let sample = (index as f32 * 2.0 * std::f32::consts::PI * 440.0 / 48_000.0).sin() * 0.2;
            audio[index] = sample;
            audio[frames as usize + index] = sample;
        }
        let wav = super::super::wav::write_planar_stereo_pcm16(&audio, 48_000).unwrap();
        let encoded =
            encode_wav(&wav, 48_000, &AudioEncoding::new("mp3", "mild").unwrap()).unwrap();
        assert!(!encoded.bytes.is_empty());
        assert_eq!(encoded.extension, "mp3");
        assert_eq!(encoded.mime, "audio/mpeg");
        assert_ne!(encoded.bytes, wav);
    }
}
