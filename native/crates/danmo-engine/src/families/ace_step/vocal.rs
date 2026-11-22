pub fn resolve_vocal_language(lyrics: &str, vocal_language: &str) -> String {
    let explicit = vocal_language.trim().to_ascii_lowercase();
    if !explicit.is_empty() && !matches!(explicit.as_str(), "auto" | "detect" | "automatic") {
        return explicit;
    }
    if lyrics_looks_chinese(lyrics) {
        "zh".into()
    } else {
        "en".into()
    }
}

pub fn lyrics_looks_chinese(lyrics: &str) -> bool {
    let cjk = lyrics
        .chars()
        .filter(|character| ('\u{4e00}'..='\u{9fff}').contains(character))
        .count();
    let latin = lyrics
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .count();
    cjk >= 4 && cjk >= latin
}

pub fn normalize_vocal_type(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "male" | "m" | "man" | "男" | "男声" | "male vocal" => "male",
        "female" | "f" | "woman" | "女" | "女声" | "female vocal" => "female",
        "chorus" | "choir" | "合唱" | "chorus vocals" => "chorus",
        "duet" | "对唱" | "duo" | "male-female" | "male female" => "duet",
        _ => "",
    }
}

pub fn apply_vocal_type_to_prompt(
    prompt: &str,
    vocal_type: &str,
    language: &str,
    instrumental: bool,
) -> (String, Option<String>) {
    let canonical = normalize_vocal_type(vocal_type);
    let base = prompt.trim();
    if canonical.is_empty() || instrumental {
        return (base.into(), None);
    }
    let zh = language.to_ascii_lowercase().starts_with("zh");
    let snippet = match (canonical, zh) {
        ("male", false) => "Clear male lead vocal, single male singer, warm masculine voice prominent in the foreground, singing the provided lyrics.",
        ("male", true) => "清晰男声主唱，单一男歌手，温暖有力的男声位于前景，按所给歌词演唱。",
        ("female", false) => "Clear female lead vocal, single female singer, bright feminine voice prominent in the foreground, singing the provided lyrics.",
        ("female", true) => "清晰女声主唱，单一女歌手，明亮女声位于前景，按所给歌词演唱。",
        ("chorus", false) => "Layered choir and group vocals with harmonies, multiple voices in chorus sections, wide vocal stack, vocals forward in the mix.",
        ("chorus", true) => "多层次合唱与群唱和声，副歌段落多人声叠唱，人声宽厚并位于前景。",
        ("duet", false) => "Male and female duet with two distinct singers, alternating lead vocals trading phrases, clear separation between voices.",
        ("duet", true) => "男女对唱，两位歌手音色分明，主唱句交替轮唱，人声清晰可分。",
        _ => return (base.into(), None),
    };
    if !base.is_empty() && prompt_already_specifies_vocals(base) {
        return (
            base.into(),
            Some(format!(
                "vocal_type={canonical} was not appended because the prompt already specifies vocals"
            )),
        );
    }
    let combined = if base.is_empty() {
        snippet.into()
    } else if zh {
        format!("{base}。{snippet}")
    } else {
        format!("{base}. {snippet}")
    };
    let mut log = format!("applied vocal_type={canonical}: {snippet}");
    if canonical == "duet" {
        log.push_str("; duet is unstable, try multiple seeds");
    } else if canonical == "chorus" {
        log.push_str("; chorus is a style hint, not guaranteed multitrack output");
    }
    (combined, Some(log))
}

fn prompt_already_specifies_vocals(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    [
        "male",
        "female",
        "masculine",
        "feminine",
        "tenor",
        "soprano",
        "baritone",
        "alto",
        "choir",
        "chorus",
        "group vocal",
        "duet",
        "duo",
        "vocal",
        "singer",
        "voice",
        "男声",
        "女声",
        "合唱",
        "对唱",
        "主唱",
        "演唱",
        "人声",
    ]
    .iter()
    .any(|keyword| lower.contains(keyword))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_detection_matches_main() {
        assert_eq!(resolve_vocal_language("你好世界", ""), "zh");
        assert_eq!(resolve_vocal_language("hello world", ""), "en");
        assert_eq!(resolve_vocal_language("你好世界", "auto"), "zh");
        assert_eq!(resolve_vocal_language("你好世界", "en"), "en");
    }

    #[test]
    fn vocal_template_is_applied_before_conditioning() {
        let (prompt, log) = apply_vocal_type_to_prompt("lofi", "female", "en", false);
        assert!(prompt.starts_with("lofi. Clear female lead vocal"));
        assert!(log.is_some());
        let (existing, _) = apply_vocal_type_to_prompt("female singer", "male", "en", false);
        assert_eq!(existing, "female singer");
        let (instrumental, _) = apply_vocal_type_to_prompt("lofi", "female", "en", true);
        assert_eq!(instrumental, "lofi");
        let (chinese, _) = apply_vocal_type_to_prompt("流行", "对唱", "zh", false);
        assert!(chinese.contains("男女对唱"));
    }
}
