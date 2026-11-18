use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("family {0} is not registered")]
    UnknownFamily(String),
    #[error("{family} does not support action {action}")]
    UnsupportedAction { family: String, action: String },
    #[error("{family} does not support option {option}")]
    UnsupportedOption { family: String, option: String },
    #[error("{family} has no ggml graph yet (shape {shape})")]
    GraphNotBuilt { family: String, shape: String },
    #[error("backend {0} is unavailable")]
    BackendUnavailable(String),
    #[error("task was cancelled")]
    Cancelled,
    #[error("run observer failed: {0}")]
    Observer(String),
    #[error("artifact persistence failed: {0}")]
    Persist(String),
    #[error("model resolution failed: {0}")]
    Resolve(String),
    #[error(transparent)]
    Weights(#[from] crate::weights::WeightError),
    #[error(transparent)]
    Runtime(#[from] crate::runtime::RuntimeError),
}

impl EngineError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub fn message(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }
}
