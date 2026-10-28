//! Lyrics timeline helpers from main `vocals/lyrics_alignment.py` + `lm/lm_format.py`.
//!
//! Product create starts from **structure_estimate** (even line allocation → LRC),
//! then upgrades to [`DIT_ALIGNMENT_MODE`] when DiT CA capture + DTW succeeds.

use serde_json::{json, Value};
use std::borrow::Cow;

/// Upstream / gallery marker for CA+DTW timelines.
pub const DIT_ALIGNMENT_MODE: &str = "dit_cross_attention";

#[derive(Debug, Clone, PartialEq)]
pub struct LyricsSegment {
    pub label: String,
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LyricsAlignment {
    pub mode: String,
    pub segments: Vec<LyricsSegment>,
}

impl LyricsAlignment {
    pub fn as_metadata(&self) -> Value {
        json!({
            "lyrics_alignment": self.mode,
            "lyrics_segments": self.segments.iter().map(|s| json!({
                "label": s.label,
                "start_sec": (s.start_sec * 100.0).round() / 100.0,
                "end_sec": (s.end_sec * 100.0).round() / 100.0,
                "text": truncate(&s.text, 200),
            })).collect::<Vec<_>>(),
        })
    }
}

/// Strip conditioning wrappers / `# Lyric` headers (main `normalize_lyrics_body`).
pub fn normalize_lyrics_body(text: &str) -> String {
    let mut t = text.trim().to_string();
    if t.is_empty() {
        return t;
    }
    // Drop optional "# Languages…# Lyric\n" style headers and trailing endoftext.
    if let Some(idx) = t.find("# Lyric\n") {
        t = t[idx + "# Lyric\n".len()..].to_string();
    } else if let Some(idx) = t.find("# Lyric\r\n") {
        t = t[idx + "# Lyric\r\n".len()..].to_string();
    }
    if let Some(stripped) = t.strip_suffix("<|endoftext|>") {
        t = stripped.trim_end().to_string();
    }
    t.trim().to_string()
}

/// True when lyrics request no vocals (main `is_instrumental_lyrics`).
pub fn is_instrumental_lyrics(text: &str) -> bool {
    let t = normalize_lyrics_body(text).to_ascii_lowercase();
    if t.is_empty() {
        return true;
    }
    matches!(t.as_str(), "[instrumental]" | "instrumental" | "[inst]" | "inst")
}

#[allow(dead_code)] // used when 5Hz LM expansion refuses empty vocal lyrics
pub fn vocal_lyrics_required_error() -> &'static str {
    "ACE-Step 人声曲目需要歌词：请在歌词框填写内容，或使用 Composer 中的「AI 生成歌词」。5Hz LM 仅格式化 caption/元数据，不再从描述自动生成歌词。"
}

/// Evenly allocate non-empty lyric lines across track duration by section.
pub fn estimate_lyrics_alignment(
    lyrics: &str,
    duration_sec: f32,
    instrumental: bool,
) -> LyricsAlignment {
    if instrumental {
        return LyricsAlignment {
            mode: "instrumental".into(),
            segments: Vec::new(),
        };
    }
    let text = lyrics.trim();
    if text.is_empty() {
        return LyricsAlignment {
            mode: "empty".into(),
            segments: Vec::new(),
        };
    }

    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    let mut current_label = "body".to_string();
    let mut current_lines: Vec<String> = Vec::new();

    for raw in text.split('\n') {
        let line = raw.trim_end();
        if let Some(label) = section_header(line) {
            if !current_lines.is_empty() {
                sections.push((current_label.clone(), std::mem::take(&mut current_lines)));
            }
            current_label = if label.is_empty() {
                "section".into()
            } else {
                label
            };
            continue;
        }
        let stripped = line.trim();
        if !stripped.is_empty() && !stripped.starts_with('[') {
            current_lines.push(stripped.to_string());
        }
    }
    if !current_lines.is_empty() {
        sections.push((current_label, current_lines));
    }

    let singable: Vec<_> = sections
        .into_iter()
        .filter(|(_, lines)| !lines.is_empty())
        .collect();
    if singable.is_empty() {
        return LyricsAlignment {
            mode: "conditioning_only".into(),
            segments: Vec::new(),
        };
    }
    let total_lines: usize = singable.iter().map(|(_, l)| l.len()).sum();
    if total_lines == 0 || !(duration_sec > 0.0) {
        return LyricsAlignment {
            mode: "conditioning_only".into(),
            segments: Vec::new(),
        };
    }

    let sec_duration = duration_sec / singable.len().max(1) as f32;
    let mut segments = Vec::new();
    let mut t = 0.0f32;
    for (label, lines) in singable {
        let block_end = duration_sec.min(t + sec_duration);
        let line_dur = (block_end - t) / lines.len().max(1) as f32;
        for (i, ln) in lines.into_iter().enumerate() {
            let start = t + i as f32 * line_dur;
            let end = block_end.min(start + line_dur);
            segments.push(LyricsSegment {
                label: label.clone(),
                start_sec: start,
                end_sec: end,
                text: ln,
            });
        }
        t = block_end;
    }

    LyricsAlignment {
        mode: "structure_estimate".into(),
        segments,
    }
}

/// Main `format_lrc` — `[mm:ss.xx]text` lines.
pub fn format_lrc(alignment: &LyricsAlignment) -> Option<String> {
    if alignment.segments.is_empty() {
        return None;
    }
    let mut lines = Vec::with_capacity(alignment.segments.len());
    for seg in &alignment.segments {
        let mm = (seg.start_sec / 60.0).floor() as i32;
        let ss = seg.start_sec % 60.0;
        lines.push(format!("[{mm:02}:{ss:05.2}]{}", seg.text));
    }
    lines.push(String::new());
    Some(lines.join("\n"))
}

pub fn lyrics_txt_sidecar(lyrics: &str) -> Option<String> {
    if is_instrumental_lyrics(lyrics) {
        return None;
    }
    let text = lyrics.trim();
    if text.is_empty() {
        return None;
    }
    Some(format!("{text}\n"))
}

pub fn prepare_create_lyrics<'a>(
    lyrics: &'a str,
    duration_sec: f32,
) -> Result<(Cow<'a, str>, LyricsAlignment, Option<String>, Option<String>), String> {
    let body = normalize_lyrics_body(lyrics);
    let instrumental = is_instrumental_lyrics(&body);
    // Empty after normalize with a non-empty raw request that is not instrumental
    // still runs as instrumental-style conditioning ([inst] default from server).
    let effective: Cow<'_, str> = if body.is_empty() {
        Cow::Borrowed("[inst]")
    } else {
        Cow::Owned(body)
    };
    let instrumental = instrumental || is_instrumental_lyrics(effective.as_ref());
    let alignment = estimate_lyrics_alignment(effective.as_ref(), duration_sec, instrumental);
    let lyrics_txt = lyrics_txt_sidecar(effective.as_ref());
    let lrc = format_lrc(&alignment);
    Ok((effective, alignment, lyrics_txt, lrc))
}

fn section_header(line: &str) -> Option<String> {
    let t = line.trim();
    if !(t.starts_with('[') && t.ends_with(']') && t.len() >= 2) {
        return None;
    }
    Some(t[1..t.len() - 1].trim().to_string())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instrumental_tags_are_detected() {
        assert!(is_instrumental_lyrics("[inst]"));
        assert!(is_instrumental_lyrics("[instrumental]"));
        assert!(is_instrumental_lyrics(""));
        assert!(!is_instrumental_lyrics("[verse]\nhello world"));
    }

    #[test]
    fn structure_estimate_splits_sections_evenly() {
        let lyrics = "[verse]\nline one\nline two\n[chorus]\nline three\n";
        let align = estimate_lyrics_alignment(lyrics, 30.0, false);
        assert_eq!(align.mode, "structure_estimate");
        assert_eq!(align.segments.len(), 3);
        assert_eq!(align.segments[0].label, "verse");
        assert_eq!(align.segments[2].label, "chorus");
        assert!((align.segments[0].start_sec - 0.0).abs() < 1e-3);
        assert!((align.segments.last().unwrap().end_sec - 30.0).abs() < 1e-2);
        let lrc = format_lrc(&align).unwrap();
        assert!(lrc.starts_with("[00:00.00]line one\n"));
        assert!(lrc.contains("line three"));
    }

    #[test]
    fn instrumental_alignment_has_no_lrc() {
        let align = estimate_lyrics_alignment("[inst]", 30.0, true);
        assert_eq!(align.mode, "instrumental");
        assert!(format_lrc(&align).is_none());
        assert!(lyrics_txt_sidecar("[inst]").is_none());
    }
}
