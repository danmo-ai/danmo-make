//! Product assistant tasks. Prompts live here — callers pick a kind, not raw chat.

use serde_json::{json, Value};

use super::ChatRequest;

const LYRICS_SYSTEM: &str = "# ACE-Step lyrics\n\nReply with **only** a lyric script. No title, planning, markdown fences, or text before/after the script.\n\nInfer structure and language from the music description in the user message.\n\n## Vocal shape\n\n```\n[Verse 1]\n<line 1>\n<line 2>\n\n[Chorus]\n<hook line 1>\n<hook line 2>\n\n[Outro]\n<closing line>\n```\n\n## Instrumental\n\n```\n[Instrumental]\n```";

const IMAGE_TO_PROMPT_SYSTEM: &str = "# Role\n\nExpert AI art prompt engineer.\n\n## Task\n\nAnalyze the attached image and write a detailed **English** prompt suitable for text-to-image models (Flux, SDXL, etc.).\n\n## Include\n\nSubject, composition, lighting, color palette, art style, mood, camera angle, and fine details.\n\n## Output\n\nOutput **only** the prompt text — no quotes, headings, or explanation.";

const VIDEO_FRAME_TO_PROMPT_SYSTEM: &str = "# Role\n\nExpert AI video prompt engineer.\n\n## Task\n\nThe attached image is a keyframe or reference for video generation. Write a detailed **English** prompt describing the scene plus implied motion, camera movement, and temporal atmosphere suitable for image-to-video models.\n\n## Output\n\nOutput **only** the prompt text — no quotes, headings, or explanation.";

const REFERENCE_VISION_SYSTEM: &str = "# Role\n\nCreative director analyzing a reference image for an artist.\n\n## Task\n\nAnswer the user's question about the attached image. Be concise and practical.\n\n## Output\n\nOutput **only** the answer — no preamble.";

const CANVAS_DESCRIBE_SYSTEM: &str = "# Role\n\nCreative studio assistant writing short notes on canvas nodes.\n\n## Task\n\nDescribe the attached visual asset in **2–4** concise sentences for an artist's canvas board note.\n\n## Cover\n\nSubject, style, lighting, composition, and one concrete next-step suggestion.\n\n## Output\n\nOutput **only** the note text — no quotes or headings.";

const CANVAS_DESCRIBE_TEXT_SYSTEM: &str = "# Role\n\nCreative studio assistant writing short notes on canvas nodes from asset metadata.\n\n## Task\n\nWrite **2–4** concise sentences for an artist's canvas board note.\n\n## Cover\n\nSubject, style, and one concrete next-step suggestion.\n\n## Output\n\nOutput **only** the note text.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantKind {
    Lyrics,
    ImageToPrompt,
    AnalyzeReference,
    CanvasDescribe,
}

impl AssistantKind {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "lyrics" | "generate_lyrics" => Some(Self::Lyrics),
            "image_to_prompt" | "frame_to_prompt" => Some(Self::ImageToPrompt),
            "analyze_reference" | "analyze" => Some(Self::AnalyzeReference),
            "canvas_describe" | "describe" => Some(Self::CanvasDescribe),
            _ => None,
        }
    }
}

pub struct AssistantTask<'a> {
    pub kind: AssistantKind,
    pub prompt: &'a str,
    pub style: &'a str,
    pub question: &'a str,
    pub locale: &'a str,
    pub video: bool,
    pub prefer_vision: bool,
    /// `data:image/...;base64,...` when the task needs an image for the provider.
    pub image_data_url: Option<&'a str>,
    pub asset_id: &'a str,
}

pub struct PreparedAssistant {
    pub request: ChatRequest,
    pub vision_used: bool,
}

pub fn prepare(model: String, task: &AssistantTask<'_>) -> Result<PreparedAssistant, String> {
    match task.kind {
        AssistantKind::Lyrics => Ok(PreparedAssistant {
            request: ChatRequest {
                model,
                messages: lyrics_messages(task.prompt, task.style),
                temperature: 0.65,
                top_p: 0.9,
                max_tokens: Some(512),
            },
            vision_used: false,
        }),
        AssistantKind::ImageToPrompt => {
            let url = task
                .image_data_url
                .ok_or_else(|| "image_to_prompt requires an installed asset image".to_string())?;
            Ok(PreparedAssistant {
                request: ChatRequest {
                    model,
                    messages: image_to_prompt_messages(url, task.video),
                    temperature: 0.4,
                    top_p: 0.9,
                    max_tokens: Some(384),
                },
                vision_used: true,
            })
        }
        AssistantKind::AnalyzeReference => {
            let url = task
                .image_data_url
                .ok_or_else(|| "analyze_reference requires an installed asset image".to_string())?;
            let question = task.question.trim();
            if question.is_empty() {
                return Err("analyze_reference requires a question".into());
            }
            Ok(PreparedAssistant {
                request: ChatRequest {
                    model,
                    messages: analyze_messages(url, question, task.locale),
                    temperature: 0.4,
                    top_p: 0.9,
                    max_tokens: Some(384),
                },
                vision_used: true,
            })
        }
        AssistantKind::CanvasDescribe => {
            if task.prefer_vision {
                let url = task.image_data_url.ok_or_else(|| {
                    "canvas_describe with vision requires an installed asset image".to_string()
                })?;
                Ok(PreparedAssistant {
                    request: ChatRequest {
                        model,
                        messages: canvas_vision_messages(url),
                        temperature: 0.4,
                        top_p: 0.9,
                        max_tokens: Some(256),
                    },
                    vision_used: true,
                })
            } else {
                Ok(PreparedAssistant {
                    request: ChatRequest {
                        model,
                        messages: canvas_text_messages(task.asset_id),
                        temperature: 0.4,
                        top_p: 0.9,
                        max_tokens: Some(256),
                    },
                    vision_used: false,
                })
            }
        }
    }
}

fn lyrics_messages(prompt: &str, style: &str) -> Value {
    let mut parts = vec![
        "## Music description".to_string(),
        prompt.trim().to_string(),
    ];
    let style = style.trim();
    if !style.is_empty() {
        parts.push(String::new());
        parts.push("## Style".into());
        parts.push(style.to_string());
    }
    json!([
        { "role": "system", "content": LYRICS_SYSTEM },
        { "role": "user", "content": parts.join("\n") },
    ])
}

fn image_to_prompt_messages(image_data_url: &str, video: bool) -> Value {
    let system = if video {
        VIDEO_FRAME_TO_PROMPT_SYSTEM
    } else {
        IMAGE_TO_PROMPT_SYSTEM
    };
    json!([
        { "role": "system", "content": system },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Caption the attached image." },
                { "type": "image_url", "image_url": { "url": image_data_url } },
            ]
        }
    ])
}

fn analyze_messages(image_data_url: &str, question: &str, locale: &str) -> Value {
    let mut user = question.trim().to_string();
    user.push_str(&vision_locale_block(locale));
    json!([
        { "role": "system", "content": REFERENCE_VISION_SYSTEM },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": user },
                { "type": "image_url", "image_url": { "url": image_data_url } },
            ]
        }
    ])
}

fn canvas_vision_messages(image_data_url: &str) -> Value {
    json!([
        { "role": "system", "content": CANVAS_DESCRIBE_SYSTEM },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Describe the attached canvas asset." },
                { "type": "image_url", "image_url": { "url": image_data_url } },
            ]
        }
    ])
}

fn canvas_text_messages(asset_id: &str) -> Value {
    json!([
        { "role": "system", "content": CANVAS_DESCRIBE_TEXT_SYSTEM },
        {
            "role": "user",
            "content": format!("## Asset\nCanvas node asset id: {asset_id}")
        }
    ])
}

fn vision_locale_block(locale: &str) -> String {
    if locale.trim().to_lowercase().starts_with("zh") {
        "\n\n## Output language\nRespond in concise Simplified Chinese (简体中文).".into()
    } else {
        "\n\n## Output language\nRespond in concise English.".into()
    }
}
