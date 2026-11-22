//! LongCat-Video-Avatar 1.5 digital human (GenerationFacade).
//!
//! ATI2V: portrait + voice track → lip-synced video. umT5 prompt
//! conditioning, Whisper audio groups, a Wan causal 3D VAE, and the 48-layer
//! Avatar DiT run an 8-step DMD distilled schedule with 3-way disentangled
//! CFG (text/audio guidance 4.0). The script-driven path stays fail-loud
//! until a TTS family lands — the session rejects `script_text` requests.

#[cfg(feature = "mlx")]
pub mod audio;
#[cfg(feature = "mlx")]
pub mod conditioning;
pub mod config;
#[cfg(feature = "mlx")]
mod rope;
mod session;
#[cfg(feature = "mlx")]
mod transformer;
#[cfg(feature = "mlx")]
mod umt5;
#[cfg(feature = "mlx")]
mod vae;

pub use session::LongcatAvatarFamily;
