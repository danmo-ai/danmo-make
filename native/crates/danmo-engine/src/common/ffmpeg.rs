//! ffmpeg/ffprobe helpers for video decode, probe, and mux. Every failure is
//! loud: missing binaries, non-zero exits, and unreadable frame rates are
//! errors, never silent fallbacks.

use crate::error::EngineError;
use std::path::{Path, PathBuf};
use std::process::Command;

fn exe_stem(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn which(name: &str) -> Option<PathBuf> {
    let stem = exe_stem(name);
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(&stem);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn sibling_exe(other: &Path, want: &str) -> Option<PathBuf> {
    let candidate = other.with_file_name(exe_stem(want));
    candidate.is_file().then_some(candidate)
}

fn resolve(name: &str, env_key: &str) -> Option<PathBuf> {
    if let Ok(raw) = std::env::var(env_key) {
        let candidate = PathBuf::from(raw);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Some(found) = which(name) {
        return Some(found);
    }
    let other = if name == "ffmpeg" {
        "ffprobe"
    } else {
        "ffmpeg"
    };
    if let Some(found) = which(other) {
        if let Some(sibling) = sibling_exe(&found, name) {
            return Some(sibling);
        }
    }
    if other == "ffprobe" {
        if let Some(found) = resolve("ffprobe", "FFPROBE_PATH") {
            if let Some(sibling) = sibling_exe(&found, name) {
                return Some(sibling);
            }
        }
    }
    for root in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let candidate = Path::new(root).join(exe_stem(name));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn require_ffmpeg() -> Result<PathBuf, EngineError> {
    resolve("ffmpeg", "FFMPEG_PATH").ok_or_else(|| {
        EngineError::Resolve(
            "ffmpeg not found (checked PATH, FFMPEG_PATH, /opt/homebrew/bin, /usr/local/bin). \
             Install ffmpeg or set FFMPEG_PATH to the binary."
                .into(),
        )
    })
}

pub fn require_ffprobe() -> Result<PathBuf, EngineError> {
    resolve("ffprobe", "FFPROBE_PATH").ok_or_else(|| {
        EngineError::Resolve(
            "ffprobe not found (checked PATH, FFPROBE_PATH, sibling of ffmpeg, \
             /opt/homebrew/bin, /usr/local/bin). Install ffmpeg (ffprobe ships with it) \
             or set FFPROBE_PATH."
                .into(),
        )
    })
}

fn parse_rate(expr: &str) -> f64 {
    let text = expr.trim();
    if text.is_empty() || text == "0/0" {
        return 0.0;
    }
    if let Some((a, b)) = text.split_once('/') {
        let a: f64 = a.trim().parse().unwrap_or(0.0);
        let b: f64 = b.trim().parse().unwrap_or(0.0);
        return a / b.max(1e-9);
    }
    text.parse().unwrap_or(0.0)
}

fn run(command: &Path, args: &[String], what: &str) -> Result<std::process::Output, EngineError> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|err| EngineError::Resolve(format!("{what} failed to start: {err}")))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let err = err.trim();
        let snippet: String = err.chars().take(1200).collect();
        return Err(EngineError::Resolve(format!(
            "{what} failed (exit {}): {snippet}",
            output.status
        )));
    }
    Ok(output)
}

pub fn probe_video_fps(video: &Path) -> Result<f64, EngineError> {
    let ffprobe = require_ffprobe().map_err(|_| {
        EngineError::Resolve(format!(
            "Cannot determine video frame rate: ffprobe not found. \
             Set PATH or FFPROBE_PATH, or install ffmpeg (ffprobe ships with it). \
             Request `fps` is not used as a fallback."
        ))
    })?;
    let args = [
        "-v".into(),
        "error".into(),
        "-select_streams".into(),
        "v:0".into(),
        "-show_entries".into(),
        "stream=avg_frame_rate,r_frame_rate".into(),
        "-of".into(),
        "json".into(),
        video.display().to_string(),
    ];
    let output = run(&ffprobe, &args, "ffprobe")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let data: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        EngineError::Resolve(format!(
            "ffprobe returned invalid JSON for {}: {err}",
            video.display()
        ))
    })?;
    let stream = data
        .get("streams")
        .and_then(|streams| streams.as_array())
        .and_then(|streams| streams.first())
        .ok_or_else(|| {
            EngineError::Resolve(format!(
                "Cannot determine video frame rate: ffprobe found no video stream in {}.",
                video.display()
            ))
        })?;
    for key in ["r_frame_rate", "avg_frame_rate"] {
        let fps = stream
            .get(key)
            .and_then(|value| value.as_str())
            .map(parse_rate)
            .unwrap_or(0.0);
        if fps > 1e-3 {
            return Ok(fps);
        }
    }
    Err(EngineError::Resolve(format!(
        "Cannot determine video frame rate: ffprobe returned no usable \
         r_frame_rate/avg_frame_rate for {}.",
        video.display()
    )))
}

pub fn video_has_audio_stream(video: &Path) -> bool {
    let Ok(ffprobe) = require_ffprobe() else {
        return false;
    };
    let args = [
        "-v".into(),
        "error".into(),
        "-select_streams".into(),
        "a".into(),
        "-show_entries".into(),
        "stream=index".into(),
        "-of".into(),
        "csv=p=0".into(),
        video.display().to_string(),
    ];
    let Ok(output) = run(&ffprobe, &args, "ffprobe") else {
        return false;
    };
    !String::from_utf8_lossy(&output.stdout).trim().is_empty()
}

/// Decode up to `max_frames` frames to `out_dir/frame_%06d.png`; returns the
/// number of frames on disk.
pub fn extract_png_frames(
    video: &Path,
    out_dir: &Path,
    max_frames: u32,
) -> Result<u32, EngineError> {
    extract_frames(video, out_dir, max_frames, "png")
}

/// Decode up to `max_frames` frames to `out_dir/frame_%06d.ppm` (uncompressed
/// P6; the frame interchange the video sessions parse directly).
pub fn extract_ppm_frames(
    video: &Path,
    out_dir: &Path,
    max_frames: u32,
) -> Result<u32, EngineError> {
    extract_frames(video, out_dir, max_frames, "ppm")
}

fn extract_frames(
    video: &Path,
    out_dir: &Path,
    max_frames: u32,
    ext: &str,
) -> Result<u32, EngineError> {
    let ffmpeg = require_ffmpeg()?;
    std::fs::create_dir_all(out_dir)
        .map_err(|err| EngineError::Resolve(format!("create {}: {err}", out_dir.display())))?;
    let pattern = out_dir.join(format!("frame_%06d.{ext}"));
    let args = [
        "-y".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-i".into(),
        video.display().to_string(),
        "-fps_mode".into(),
        "passthrough".into(),
        "-vframes".into(),
        max_frames.to_string(),
        pattern.display().to_string(),
    ];
    run(&ffmpeg, &args, "ffmpeg frame extract")?;
    let mut count = 0u32;
    let entries = std::fs::read_dir(out_dir)
        .map_err(|err| EngineError::Resolve(format!("read {}: {err}", out_dir.display())))?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("frame_") && name.ends_with(&format!(".{ext}")) {
            count += 1;
        }
    }
    Ok(count)
}

fn ppm_token<'a>(bytes: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < bytes.len() && bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
    if start == *pos {
        None
    } else {
        Some(&bytes[start..*pos])
    }
}

/// Parse a binary P6 PPM (the ffmpeg image2 layout): magic, width, height,
/// maxval 255, one whitespace, then `width*height*3` RGB bytes.
pub fn decode_ppm(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut pos = 0;
    if ppm_token(bytes, &mut pos)? != b"P6" {
        return None;
    }
    let number = |pos: &mut usize| -> Option<u32> {
        std::str::from_utf8(ppm_token(bytes, pos)?)
            .ok()?
            .parse()
            .ok()
    };
    let width = number(&mut pos)?;
    let height = number(&mut pos)?;
    let maxval = number(&mut pos)?;
    if maxval != 255 || width == 0 || height == 0 {
        return None;
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    pos += 1;
    let n = width as usize * height as usize * 3;
    if bytes.len() < pos + n {
        return None;
    }
    Some((bytes[pos..pos + n].to_vec(), width, height))
}

/// Mux a printf-style PNG sequence to H.264 MP4, copying the source audio
/// track when one exists.
pub fn mux_png_sequence_to_mp4(
    png_pattern: &Path,
    source_audio_from: Option<&Path>,
    output_mp4: &Path,
    fps: f64,
) -> Result<(), EngineError> {
    let ffmpeg = require_ffmpeg()?;
    if let Some(parent) = output_mp4.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| EngineError::Resolve(format!("create {}: {err}", parent.display())))?;
    }
    let fps_s = format!("{:.6}", fps.max(1e-3));
    let fps_s = fps_s
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string();
    let mut args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-framerate".to_string(),
        fps_s,
        "-i".to_string(),
        png_pattern.display().to_string(),
    ];
    let audio = source_audio_from.filter(|path| video_has_audio_stream(path));
    if let Some(source) = audio {
        args.extend([
            "-i".to_string(),
            source.display().to_string(),
            "-map".to_string(),
            "0:v:0".to_string(),
            "-map".to_string(),
            "1:a:0".to_string(),
            "-c:v".to_string(),
            "libx264".to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
            "-crf".to_string(),
            "20".to_string(),
            "-c:a".to_string(),
            "aac".to_string(),
            "-b:a".to_string(),
            "192k".to_string(),
            "-movflags".to_string(),
            "+faststart".to_string(),
            output_mp4.display().to_string(),
        ]);
    } else {
        args.extend([
            "-c:v".to_string(),
            "libx264".to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
            "-crf".to_string(),
            "20".to_string(),
            "-movflags".to_string(),
            "+faststart".to_string(),
            output_mp4.display().to_string(),
        ]);
    }
    run(&ffmpeg, &args, "ffmpeg mux")?;
    Ok(())
}

/// Mux a PNG frame sequence and a WAV audio file into an MP4
/// (H.264 yuv420p + AAC), used by families that synthesize their own audio
/// track (LTX). Every generated frame lands in the container: the decoded
/// audio is typically a touch shorter than `frames / fps`, so `-shortest`
/// would drop the tail frame.
pub fn mux_png_sequence_wav_to_mp4(
    png_pattern: &Path,
    wav: &Path,
    output_mp4: &Path,
    fps: f64,
) -> Result<(), EngineError> {
    let ffmpeg = require_ffmpeg()?;
    if let Some(parent) = output_mp4.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| EngineError::Resolve(format!("create {}: {err}", parent.display())))?;
    }
    let fps_s = format!("{:.6}", fps.max(1e-3));
    let fps_s = fps_s
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string();
    let args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-framerate".to_string(),
        fps_s,
        "-i".to_string(),
        png_pattern.display().to_string(),
        "-i".to_string(),
        wav.display().to_string(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-map".to_string(),
        "1:a:0".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-crf".to_string(),
        "20".to_string(),
        "-c:a".to_string(),
        "aac".to_string(),
        "-b:a".to_string(),
        "192k".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
        output_mp4.display().to_string(),
    ];
    run(&ffmpeg, &args, "ffmpeg mux").map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ffmpeg-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn extract_probe_and_mux_roundtrip() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = scratch("roundtrip");
        let source = dir.join("src.mp4");
        // 8 frames of 32x32 testsrc at 8 fps, no audio.
        let args = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=8:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ];
        let status = Command::new("ffmpeg")
            .args(args)
            .arg(&source)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let fps = probe_video_fps(&source).unwrap();
        assert!((fps - 8.0).abs() < 0.5, "fps {fps}");
        assert!(!video_has_audio_stream(&source));
        let frames_dir = dir.join("frames");
        let n = extract_png_frames(&source, &frames_dir, 100).unwrap();
        assert_eq!(n, 8, "extracted {n}");
        let n = extract_png_frames(&source, &dir.join("frames2"), 3).unwrap();
        assert_eq!(n, 3, "extracted {n}");
        let out = dir.join("out.mp4");
        mux_png_sequence_to_mp4(&frames_dir.join("frame_%06d.png"), Some(&source), &out, fps)
            .unwrap();
        let out_fps = probe_video_fps(&out).unwrap();
        assert!((out_fps - 8.0).abs() < 0.5, "fps {out_fps}");
        let n = extract_png_frames(&out, &dir.join("frames3"), 100).unwrap();
        assert_eq!(n, 8, "extracted {n}");
        // PPM interchange: extract, parse the first frame back to pixels.
        let ppm_dir = dir.join("frames_ppm");
        let n = extract_ppm_frames(&source, &ppm_dir, 100).unwrap();
        assert_eq!(n, 8, "extracted {n} ppm");
        let bytes = std::fs::read(ppm_dir.join("frame_000001.ppm")).unwrap();
        let (rgb, w, h) = decode_ppm(&bytes).expect("ffmpeg ppm parses");
        assert_eq!((w, h), (32, 32));
        assert_eq!(rgb.len(), 32 * 32 * 3);
        assert!(decode_ppm(b"P3\n2 2\n255\n0 0 0").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mux_with_audio_preserves_the_video_tail_frame() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = scratch("audio-tail");
        let source = dir.join("source.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=16x16:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-frames:v",
                "5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(status.status.success());
        let frames = dir.join("frames");
        assert_eq!(extract_ppm_frames(&source, &frames, 5).unwrap(), 5);
        let output = dir.join("output.mp4");
        mux_png_sequence_to_mp4(&frames.join("frame_%06d.ppm"), Some(&source), &output, 24.0)
            .unwrap();
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
            ])
            .arg(&output)
            .output()
            .unwrap();
        let probe: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        assert_eq!(probe["streams"][0]["nb_frames"], "5");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_video_fails_loud() {
        if Command::new("ffprobe").arg("-version").output().is_err() {
            return;
        }
        let dir = scratch("missing");
        let err = probe_video_fps(&dir.join("nope.mp4")).unwrap_err();
        assert!(err.to_string().contains("ffprobe"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
