//! Flux2 Klein on MLX.
//!
//! The forward matches `danmo-make` `backend/engine/families/flux2/transformer_mlx.py`
//! (main): separate Q/K/V, text-then-image joint attention, scale-then-shift
//! AdaLN. Width, depth, and the text tower come from the checkpoint. A guidance
//! embedder runs only when those weights are present.

pub(crate) mod forward;
mod lora;
mod resident;
pub(crate) mod session;
mod text;
