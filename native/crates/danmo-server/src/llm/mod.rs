//! Internal LLM client for Danmo Make product features only.
//!
//! Used by prompt enhance, creative assistant tasks, and (later) training
//! dataset helpers. Providers (`builtin` / `openai_compatible`) are backends
//! for those features — this module is not a public OpenAI proxy and does not
//! re-export arbitrary cloud chat completions.

mod builtin;
mod cloud;
mod enhance;
pub mod tasks;

use serde_json::{json, Value};

pub use enhance::{build_enhance_messages, content_from_completion, sanitize_enhanced_prompt};

/// Human-readable block reason, or `None` when the selected provider can run.
pub fn block_reason(settings: &Value) -> Option<String> {
    from_settings(settings).block_reason()
}

pub fn provider_id(settings: &Value) -> &'static str {
    match settings
        .get("llm_inference_provider")
        .and_then(|v| v.as_str())
    {
        Some("openai_compatible") => "openai_compatible",
        _ => "builtin",
    }
}

/// Resolve the model id sent to the selected backend: cloud override, else catalog.
pub fn resolve_model(settings: &Value, catalog_model_id: &str) -> String {
    let cloud_model = settings
        .get("llm_inference_cloud_model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if !cloud_model.is_empty() {
        return cloud_model.to_string();
    }
    catalog_model_id.to_string()
}

pub fn from_settings(settings: &Value) -> Box<dyn LlmProvider> {
    match provider_id(settings) {
        "openai_compatible" => Box::new(cloud::OpenAiCompatibleProvider::from_settings(settings)),
        _ => Box::new(builtin::BuiltinStubProvider),
    }
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Value,
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: Option<u64>,
}

impl ChatRequest {
    pub fn to_provider_payload(&self) -> Value {
        let mut payload = json!({
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": false,
        });
        if let Some(max_tokens) = self.max_tokens {
            payload["max_tokens"] = json!(max_tokens);
        }
        payload
    }
}

#[derive(Debug)]
pub enum LlmError {
    /// Selected provider is not ready (maps to HTTP 503).
    Unavailable(String),
    /// Upstream / transport failure (maps to HTTP 502).
    Upstream(String),
}

impl LlmError {
    #[allow(dead_code)]
    pub fn detail(&self) -> &str {
        match self {
            Self::Unavailable(msg) | Self::Upstream(msg) => msg,
        }
    }
}

/// Backend for product LLM features. Not a public chat gateway.
pub trait LlmProvider: Send + Sync {
    fn block_reason(&self) -> Option<String>;
    fn complete(&self, request: &ChatRequest) -> Result<Value, LlmError>;
}
