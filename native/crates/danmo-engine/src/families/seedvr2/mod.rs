//! SeedVR2 upscale (operation `upscale`), image and video. Image: the source
//! is bicubic-resized so the short edge scales by `scale`, padded to 16,
//! encoded by the 3D VAE (T=1), denoised in one Euler step at t=1000 by the
//! MM-DiT with windowed attention, decoded, color-corrected against the
//! source, and cropped back to the true scaled size. Video: ffmpeg-extracted
//! frames run the same pipeline in `chunk_frames` chunks over the VAE/DiT
//! time axis and mux back to MP4 with the source audio track.

pub mod color;
pub mod dit;
#[cfg(feature = "mlx")]
pub mod mlx;
pub mod session;
pub mod tiling;
pub mod vae;
pub mod video_session;

pub use session::{SeedVr2UpscaleFamily, write_smoke_bundle};
pub use video_session::SeedVr2VideoUpscaleFamily;
