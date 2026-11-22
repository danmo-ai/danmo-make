//! Official ACE-Step 1.5 diffusion sampler.
//!
//! The definition is ACE Studio's PyTorch `generate_audio`:
//! `acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py` and
//! `acestep/models/xl_base/modeling_acestep_v15_xl_base.py`, with APG from
//! `acestep/models/common/apg_guidance.py` (`dims=[1]`, momentum -0.75).
//! The official MLX loop (`acestep/models/mlx/dit_generate.py`) is the Apple
//! counterpart; where it diverges (APG momentum, base SDE sigma, Heun CFG),
//! this module follows the PyTorch model.

mod diffusion;
#[cfg(feature = "mlx")]
mod lm;
#[cfg(feature = "mlx")]
pub(crate) use lm::{expand as expand_lm, run as run_lm_mlx};
#[cfg(feature = "mlx")]
mod model;
#[cfg(feature = "mlx")]
pub(crate) mod session;
#[cfg(feature = "mlx")]
mod vae;

pub use diffusion::{sample_with, ModelVariant, SampleRequest};
