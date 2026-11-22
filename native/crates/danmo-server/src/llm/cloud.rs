//! OpenAI-compatible HTTP backend for product LLM features.
//!
//! Used only as an internal provider for enhance / assistant / dataset helpers.
//! Danmo Make does not expose a public cloud chat proxy.

use std::time::Duration;

use serde_json::Value;

use super::{ChatRequest, LlmError, LlmProvider};

pub struct OpenAiCompatibleProvider {
    base_url: String,
    api_key: String,
}

impl OpenAiCompatibleProvider {
    pub fn from_settings(settings: &Value) -> Self {
        Self {
            base_url: base_url(settings),
            api_key: api_key(settings),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

impl LlmProvider for OpenAiCompatibleProvider {
    fn block_reason(&self) -> Option<String> {
        if self.base_url.trim().is_empty() || !key_usable(&self.api_key) {
            return Some(
                "Cloud LLM inference is not configured (base URL and API key required)."
                    .to_string(),
            );
        }
        None
    }

    fn complete(&self, request: &ChatRequest) -> Result<Value, LlmError> {
        if let Some(reason) = self.block_reason() {
            return Err(LlmError::Unavailable(reason));
        }
        let url = self.chat_url();
        let payload = request.to_provider_payload();
        post_json(&url, &self.api_key, &payload).map_err(LlmError::Upstream)
    }
}

fn api_key(settings: &Value) -> String {
    let env = std::env::var("DANQING_LLM_INFERENCE_API_KEY").unwrap_or_default();
    let env = env.trim();
    if !env.is_empty() {
        return env.to_string();
    }
    settings
        .get("llm_inference_api_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

fn key_usable(key: &str) -> bool {
    let key = key.trim();
    !key.is_empty() && !key.starts_with("v1:")
}

fn base_url(settings: &Value) -> String {
    let base = settings
        .get("llm_inference_base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .trim_end_matches('/');
    if base.is_empty() {
        return String::new();
    }
    if base.contains("/v1") {
        return base.to_string();
    }
    format!("{base}/v1")
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into()
}

fn post_json(url: &str, key: &str, payload: &Value) -> Result<Value, String> {
    let mut request = agent()
        .post(url)
        .header("accept", "application/json")
        .header("content-type", "application/json");
    if !key.is_empty() {
        request = request.header("authorization", &format!("Bearer {key}"));
    }
    let body = serde_json::to_string(payload).map_err(|err| err.to_string())?;
    let response = request
        .send(body)
        .map_err(|err| format!("POST {url} failed: {err}"))?;
    let status = response.status().as_u16();
    let mut text = String::new();
    std::io::Read::read_to_string(
        &mut std::io::Read::take(response.into_body().as_reader(), 20 * 1024 * 1024),
        &mut text,
    )
    .map_err(|err| format!("reading {url} failed: {err}"))?;
    if !(200..300).contains(&status) {
        return Err(format!(
            "POST {url} returned {status}: {}",
            text.chars().take(400).collect::<String>()
        ));
    }
    serde_json::from_str(&text).map_err(|err| format!("{url} was not JSON: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_settings_are_not_ready() {
        let provider = OpenAiCompatibleProvider::from_settings(&json!({}));
        assert!(provider.block_reason().unwrap().contains("not configured"));
    }

    #[test]
    fn ready_when_base_and_key_present() {
        let provider = OpenAiCompatibleProvider::from_settings(&json!({
            "llm_inference_base_url": "https://api.example.com/v1",
            "llm_inference_api_key": "sk-test"
        }));
        assert!(provider.block_reason().is_none());
        assert_eq!(provider.base_url, "https://api.example.com/v1");
    }

    #[test]
    fn appends_v1_when_missing() {
        let provider = OpenAiCompatibleProvider::from_settings(&json!({
            "llm_inference_base_url": "https://api.example.com",
            "llm_inference_api_key": "sk-test"
        }));
        assert_eq!(provider.base_url, "https://api.example.com/v1");
    }
}
