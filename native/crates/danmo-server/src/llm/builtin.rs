//! Local MLX assistant provider stub.
//!
//! Fail loud until the engine assistant worker is wired. Never fall back to cloud.

use serde_json::Value;

use super::{ChatRequest, LlmError, LlmProvider};

pub const LOCAL_NOT_WIRED: &str = "Local LLM assistant is not wired yet. Set Settings → Integrations to OpenAI-compatible (base URL + API key) for creative enhance/assistant, or wait for a later release.";

pub struct BuiltinStubProvider;

impl LlmProvider for BuiltinStubProvider {
    fn block_reason(&self) -> Option<String> {
        Some(LOCAL_NOT_WIRED.to_string())
    }

    fn complete(&self, _request: &ChatRequest) -> Result<Value, LlmError> {
        Err(LlmError::Unavailable(LOCAL_NOT_WIRED.to_string()))
    }
}
