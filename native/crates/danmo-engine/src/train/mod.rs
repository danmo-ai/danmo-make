//! Shared LoRA training (Flow-Match SFT). Family runners live under `families/*/train`.

#[cfg(feature = "mlx")]
mod export;
#[cfg(feature = "mlx")]
mod loop_;
#[cfg(feature = "mlx")]
mod loss;
#[cfg(feature = "mlx")]
mod schedule;

#[cfg(feature = "mlx")]
pub use export::{export_lora_adapters, init_lora_params, LoraTrainParam};
#[cfg(feature = "mlx")]
pub use loop_::{run_lora_train_steps, TrainStepConfig, TrainStepResult};
#[cfg(feature = "mlx")]
pub use loss::{flow_match_mse, sample_noisy_latent};
#[cfg(feature = "mlx")]
pub use schedule::{
    discrete_training_weight, sample_discrete_sigma, sample_logit_normal_sigma, FlowMatchTrainSchedule,
};
