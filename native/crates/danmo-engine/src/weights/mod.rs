//! GGUF bundles. Tensor names in the file are graph parameter names.
//! Nothing in this module rewrites a key.

mod bundle;
mod gguf;

pub use bundle::assert_gguf_bundle;
pub use gguf::{write_f32_gguf, write_f32_tensors, F32Tensor, GgufWeights};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error(
        "{path} is not a GGUF file. Re-download the GGUF version (bf16, q8_0, or q4_k). Safetensors and MLX weights are not loaded"
    )]
    NotGguf { path: String },
    #[error(
        "{path} is an old MLX or safetensors bundle. Re-download the GGUF version (bf16, q8_0, or q4_k)"
    )]
    LegacyBundle { path: String },
    #[error("{path} is not a ready GGUF bundle: {detail}. Re-download the GGUF version (bf16, q8_0, or q4_k)")]
    Bundle { path: String, detail: String },
    #[error("failed to read GGUF {path}: {detail}")]
    Open { path: String, detail: String },
    #[error(
        "GGUF {path} has no tensor {name}. File names are graph parameter names; nothing is remapped"
    )]
    MissingTensor { path: String, name: String },
    #[error("tensor {name} shape {got:?} does not match the graph {expected:?}")]
    Shape {
        name: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
    #[error(
        "tensor {name} is ggml type {ggml_type}; this graph input expects F32. Quantized matmul is not wired for this tensor yet"
    )]
    Type { name: String, ggml_type: u32 },
    #[error("duplicate GGUF tensor {name} in {path}")]
    Duplicate { name: String, path: String },
    #[error("tensor name {name} is longer than GGML_MAX_NAME ({max})")]
    NameTooLong { name: String, max: usize },
    #[error("{0}")]
    NotBuilt(String),
    #[error(transparent)]
    Runtime(#[from] crate::runtime::RuntimeError),
}

/// LoRA rows use the base parameter name plus `.lora_a` or `.lora_b`.
/// Diffusers suffixes such as `.lora_A` are not translated.
pub fn lora_base_name(tensor: &str) -> Option<&str> {
    tensor
        .strip_suffix(".lora_a")
        .or_else(|| tensor.strip_suffix(".lora_b"))
}
