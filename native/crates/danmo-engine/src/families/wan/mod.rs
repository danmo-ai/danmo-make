//! Wan 2.2 video family. t2v (TI2V-5B) is wired end to end: prepare
//! validates the request against the reference geometry rules and checks the
//! GGUF bundle (DiT + umT5 text encoder + VAE + tokenizer); infer runs umT5
//! conditioning, the reserved DiT graph under WanFlowUniPC with standard CFG,
//! causal 3D VAE decode, and an ffmpeg MP4 mux. i2v / animate and the 14B
//! dual-model variants fail loud until they land.

mod dit;
mod session;
pub mod umt5;
pub mod vae;

pub use session::{write_smoke_bundle, WanVideoCreateFamily};

/// Official `wan/configs/shared_config.py` sample_neg_prompt — the reference
/// uses it when the user leaves the negative prompt empty.
pub const WAN_SAMPLE_NEG_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
