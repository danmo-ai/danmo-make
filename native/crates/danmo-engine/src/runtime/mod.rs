//! ggml runtime. Families never open a device and never call ggml directly.

mod context;
mod extend;
mod graph;

use danmo_ggml_sys as g;
use thiserror::Error;

pub use context::GgmlContext;
pub(crate) use extend::euler_graph;
pub use extend::{
    flash_attend, measure_diffusion_steps, preview_conv, rope_pair, DeviceStep, DiffusionTiming,
    RopeIn,
};
pub use graph::{shape, GraphBuilder, ReusedGraph};

pub type Tensor = *mut g::ggml_tensor;
pub const TYPE_F32: u32 = g::ggml_type_GGML_TYPE_F32;
pub const TYPE_F16: u32 = g::ggml_type_GGML_TYPE_F16;
pub const TYPE_Q8_0: u32 = g::ggml_type_GGML_TYPE_Q8_0;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("ggml backend {0} was requested but this binary was not built with it")]
    BackendUnavailable(String),
    #[error("ggml init failed")]
    Init,
    #[error("ggml graph reserve failed")]
    Reserve,
    #[error("ggml graph compute failed: {0}")]
    Compute(i32),
    #[error("tensor {0} was not declared as a graph input")]
    MissingInput(String),
    #[error("tensor {name} has {expected} bytes, got {got}")]
    SizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl BackendKind {
    pub fn parse(name: &str) -> Result<Self, RuntimeError> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            other => Err(RuntimeError::BackendUnavailable(other.to_string())),
        }
    }

    pub fn from_env() -> Result<Self, RuntimeError> {
        match std::env::var("DANQING_GGML_BACKEND") {
            Ok(name) => Self::parse(name.trim()),
            Err(_) => Ok(Self::Cpu),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
        }
    }
}

/// Identity matmul used to prove gallocr reuse: y = W x, W fixed, x replaced between computes.
pub fn identity_probe(ctx: &GgmlContext) -> Result<ReusedGraph, RuntimeError> {
    let mut b = ctx.builder(8 * 1024 * 1024)?;
    let w = b.f32_2d("w", 2, 2, true);
    let x = b.f32_2d("x", 2, 1, true);
    let y = b.mul_mat(w, x);
    b.finish(y)
}
