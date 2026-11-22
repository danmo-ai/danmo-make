//! Weight bundles. Product loads are safetensors/MLX; GGUF helpers remain for
//! families that still open a GGUF DiT. Tensor names in a file are graph
//! parameter names — nothing in this module rewrites a key.

mod bundle;
mod cache;
mod gguf;
mod lora;
#[cfg(feature = "mlx")]
mod mlx;
#[cfg(feature = "mlx")]
mod quantize;
#[cfg(all(test, feature = "mlx"))]
pub(crate) use quantize::fixture;

pub use bundle::{
    assert_gguf_bundle, is_ace_step_safetensors_bundle, is_controlnet_safetensors_dir,
    is_safetensors_bundle, is_safetensors_image_bundle, is_seedvr2_safetensors_bundle,
};
pub use cache::{resident_bytes, revision};
pub(crate) use gguf::f16_to_f32;
pub use gguf::{
    f32_row_to_f16, matmul_type, quantize_q8_0, type_block, write_f32_gguf, write_f32_nd,
    write_f32_tensors, write_safetensors, write_typed_tensors, F32Nd, F32Tensor, GgufWeights,
    SafeTensor, TypedTensor,
};
pub use lora::{
    bind_lora_factors, load_lora_factors, load_lora_factors_mapped,
    load_lora_factors_with_default_alpha, LoraFactor, LoraFile, LoraRoute,
};
#[cfg(feature = "mlx")]
pub use mlx::{
    load_directory as load_mlx_directory, load_file as load_mlx_file, pack_affine,
    probe_affine_quantization, read_component, read_component_meta, MlxWeights, WeightDtype,
};
#[cfg(feature = "mlx")]
pub use quantize::{
    load_recipe, manifest_value, quantize_bundle, recipe_paths, source_digest,
    verify_quantized_bundle, QuantRecipe, QuantizeReport, MANIFEST_FORMAT, MANIFEST_VERSION,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error(
        "{path} is not a weight file for this role. A GGUF DiT family expects .gguf for the transformer; companions may be GGUF or safetensors"
    )]
    NotGguf { path: String },
    #[error("{path} is not a ready weight bundle: {detail}")]
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
