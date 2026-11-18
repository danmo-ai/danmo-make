//! GGUF bundles. Tensor names in the file are graph parameter names.
//! Nothing in this module rewrites a key.

mod bundle;
mod cache;
mod gguf;
mod lora;
#[cfg(feature = "mlx")]
mod mlx;

pub use bundle::{
    assert_gguf_bundle, assert_hunyuan_bundle, assert_ltx_bundle, assert_upscale_bundle,
    is_ace_step_safetensors_bundle, is_safetensors_image_bundle, HunyuanBundle, LtxBundle,
    UpscaleBundle,
};
pub use cache::{resident_bytes, revision};
pub(crate) use gguf::f16_to_f32;
pub use gguf::{
    f32_row_to_f16, matmul_type, quantize_q8_0, type_block, write_f32_gguf, write_f32_nd,
    write_f32_tensors, write_safetensors, write_typed_tensors, F32Nd, F32Tensor, GgufWeights,
    SafeTensor, TypedTensor,
};
pub use lora::{
    bind_lora_factors, load_lora_factors, load_lora_factors_with_default_alpha, LoraFactor,
    LoraFile,
};
#[cfg(feature = "mlx")]
pub use mlx::{load_directory as load_mlx_directory, MlxWeights, WeightDtype};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error(
        "{path} is not a weight file for this role. The DiT is GGUF (bf16, q8_0, or q4_k). The text encoder and VAE are GGUF or safetensors"
    )]
    NotGguf { path: String },
    #[error(
        "{path} is an old MLX or safetensors bundle. Re-download the GGUF version (bf16, q8_0, or q4_k)"
    )]
    LegacyBundle { path: String },
    #[error("{path} is not a ready GGUF bundle: {detail}. Re-download the GGUF version (bf16, q8_0, or q4_k)")]
    Bundle { path: String, detail: String },
    #[error("failed to read weights {path}: {detail}")]
    Open { path: String, detail: String },
    #[error(
        "weights {path} has no tensor {name}. File names are graph parameter names; nothing is remapped"
    )]
    MissingTensor { path: String, name: String },
    #[error("tensor {name} shape {got:?} does not match the graph {expected:?}")]
    Shape {
        name: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
    #[error(
        "tensor {name} is ggml type {ggml_type}. Matmul accepts F32, F16, BF16, Q8_0, Q4_K, Q5_K, and Q6_K; other types are not dequantized"
    )]
    Type { name: String, ggml_type: u32 },
    #[error("duplicate tensor {name} in {path}")]
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
