//! Qwen-Image create on the process MLX executor.
//!
//! Component weights stay resident in `ModelCache`; encode / denoise / decode
//! run on the executor thread. The text tower is the Qwen2.5-VL decoder stack
//! (`encoder.*` keys), the DiT reads the flat mlx-community layout directly,
//! and the VAE is the 3D causal decoder (single frame for create).

mod checkpoint;
pub(crate) mod edit;
mod forward;
mod lora;
pub(crate) mod resident;
pub(crate) mod session;
mod text;
mod vae;
mod vision;
