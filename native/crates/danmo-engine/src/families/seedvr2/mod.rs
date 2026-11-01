//! SeedVR2 image upscale (operation `upscale`). Single frame: the source is
//! bicubic-resized so the short edge scales by `scale`, padded to 16, encoded
//! by the 3D VAE (T=1), denoised in one Euler step at t=1000 by the MM-DiT
//! with windowed attention, decoded, color-corrected against the source, and
//! cropped back to the true scaled size.

pub mod color;
pub mod dit;
pub mod session;
pub mod tiling;
pub mod vae;

pub use session::{write_smoke_bundle, SeedVr2UpscaleFamily};
