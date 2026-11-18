//! Prompt-enhance helpers shared by creative composers.

use serde_json::{json, Value};

const ENHANCE_IMAGE_SYSTEM: &str = "# Role\n\nPrompt engineer for AI image models (Flux, Z-Image, Qwen-Image).\n\n## Task\n\nRewrite the user's idea into one vivid, comma-separated description. Keep their subject, names, and intent.\n\n## Style\n\n- Add at most a few cues for lighting, composition, color, texture, and mood.\n- If the input is already detailed, lightly polish only — do not lengthen.\n\n## Limits\n\n- Length cap: ~120 CJK characters or ~80 English words.\n- Never repeat the same word or phrase; never loop filler at the end.\n- Do not write \"Okay\", explanations, or quotes.\n\n## Output\n\nOutput **only** the enhanced prompt.";

const ENHANCE_VIDEO_SYSTEM: &str = "# Role\n\nProfessional prompt engineer for AI video generation.\n\n## Task\n\nRewrite the user's brief into a detailed prompt for image-to-video or text-to-video models.\n\n## Include\n\nSubject, scene, lighting, style, camera movement, motion dynamics, pacing, and temporal mood.\n\nFor LTX audio-video models, write the caption in the language that should be heard. Quoted dialogue is spoken in the language of the surrounding sentences: an English caption with Chinese quotes is performed as English. Do not translate a spoken script. Put one space before each opening quotation mark and keep every spoken line verbatim.\nDistinguish static scene description from continuing motion the camera can follow.\n\n## Limits\n\n- One paragraph. Visual briefs stay within ~120 CJK characters or ~80 English words. Spoken scripts are exempt: keep every line.\n- Never repeat the same phrase or word. No filler loops.\n\n## Output\n\nOutput **only** the enhanced prompt text, without explanation. Keep spoken lines inside quotation marks.";

const ENHANCE_AUDIO_SYSTEM: &str = "# Role\n\nMusic producer writing briefs for AI music generation (ACE-Step).\n\n## Task\n\nExpand the user's music idea into a clear, vivid description covering genre, mood, tempo feel, instrumentation, vocal style, and emotional arc.\n\n## Limits\n\n- One short paragraph.\n- Never repeat the same phrase or word. No filler loops.\n\n## Output\n\nOutput **only** the enhanced brief text, without explanation or quotation marks.";

fn system_prompt(target_action: &str) -> &'static str {
    let action = if target_action.trim().is_empty() {
        "image_create"
    } else {
        target_action.trim()
    };
    match action.to_lowercase().as_str() {
        "video" | "video_create" | "animate" | "video_generation" => ENHANCE_VIDEO_SYSTEM,
        "audio" | "audio_create" | "music" | "audio_generation" => ENHANCE_AUDIO_SYSTEM,
        _ => ENHANCE_IMAGE_SYSTEM,
    }
}

fn user_locale_hint(text: &str) -> &'static str {
    let raw = text.trim();
    if raw.is_empty() {
        return "";
    }
    let chars = raw.chars().count();
    if raw
        .chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
    {
        if chars < 60 {
            return "\n\n（输入较短：保留原意与关键词，最多补充少量画面细节，禁止堆砌形容词或重复用词。）";
        }
        return "\n\n（输入已够详细：只做轻微润色，禁止加长或重复用词。）";
    }
    if chars < 80 {
        return "\n\n(Input is short: preserve intent and keywords; add at most a few visual cues; no filler loops.)";
    }
    "\n\n(Input is already detailed: light polish only; do not lengthen or repeat phrases.)"
}

/// Build OpenAI-style messages for an enhance-prompt call.
pub fn build_enhance_messages(
    prompt: &str,
    target_action: &str,
    style_positive: &str,
) -> Value {
    let action = target_action.to_lowercase();
    let mut user_content = prompt.to_string();
    if matches!(action.as_str(), "image_create" | "create" | "image") {
        user_content.push_str(user_locale_hint(prompt));
    }
    let style = style_positive.trim();
    if !style.is_empty() {
        user_content.push_str(&format!("\n\nStyle cues to weave in: {style}"));
    }
    json!([
        { "role": "system", "content": system_prompt(target_action) },
        { "role": "user", "content": user_content },
    ])
}

/// Keep the post-reasoning answer; empty when generation stopped inside thinking.
fn extract_final_content(text: &str) -> String {
    let raw = text.trim();
    if raw.is_empty() {
        return String::new();
    }
    let lower = raw.to_lowercase();
    for tag in ["</think>", "</redacted_reasoning>"] {
        if let Some(pos) = lower.rfind(tag) {
            return raw[pos + tag.len()..]
                .trim()
                .trim_end_matches("<|im_end|>")
                .trim()
                .to_string();
        }
    }
    if lower.starts_with("<think>") || lower.starts_with("<redacted_reasoning>") {
        return String::new();
    }
    raw.trim_end_matches("<|im_end|>").trim().to_string()
}

pub fn sanitize_enhanced_prompt(text: &str) -> String {
    let answer = extract_final_content(text);
    let trimmed = answer.trim();
    let unquoted = if trimmed.len() >= 2 {
        let first = trimmed.as_bytes()[0];
        let last = trimmed.as_bytes()[trimmed.len() - 1];
        if first == last && (first == b'"' || first == b'\'') {
            trimmed[1..trimmed.len() - 1].trim()
        } else {
            trimmed
        }
    } else {
        trimmed
    };
    unquoted.chars().take(480).collect()
}

pub fn content_from_completion(response: &Value) -> String {
    response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_quotes_and_think_blocks() {
        assert_eq!(
            sanitize_enhanced_prompt("<think>plan</think>\n\"a cat\""),
            "a cat"
        );
    }
}
