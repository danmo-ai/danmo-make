use super::transformer::QwenImageForward;
use super::weights::{check_layout, config_from_weights, weight_layout, QwenImageConfig};
use crate::contract::{ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, GgufWeights, LoraFile, WeightError};
use danmo_contracts::ImageGenerationRequest;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct QwenImageFamily;

impl DitImageFamily for QwenImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        if request.structural_guide.is_some() {
            return Err(WeightError::NotBuilt(
                "structural guide graph is not built for qwen_image".into(),
            )
            .into());
        }
        if !loras.is_empty() {
            return Err(WeightError::NotBuilt("lora inference graph is not built".into()).into());
        }
        let files = assert_gguf_bundle(&run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let config = config_from_weights(&loaded)?;
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        let scale = config.vae_scale.max(1);
        if width % scale != 0 || height % scale != 0 {
            return Err(WeightError::NotBuilt(format!(
                "size {width}x{height} is not divisible by vae scale {scale}"
            ))
            .into());
        }
        check_layout(&loaded, config)?;
        if loaded
            .names()
            .any(|name| name.contains("__index_timestep_zero__"))
        {
            return Err(WeightError::NotBuilt("zero_cond_t edit path is not built".into()).into());
        }
        if config.in_channels != 64 || config.out_channels != 16 || config.patch != 2 {
            return Err(WeightError::NotBuilt(format!(
                "qwen packed create needs in_channels 64 out_channels 16 patch 2, got {} {} {}",
                config.in_channels, config.out_channels, config.patch
            ))
            .into());
        }
        let guidance = request.guidance.unwrap_or(1.0);
        if !guidance.is_finite() {
            return Err(WeightError::NotBuilt("qwen-image guidance must be finite".into()).into());
        }
        Ok(Box::new(QwenImageSession {
            bundle: run.bundle.clone(),
            request: request.clone(),
            config,
            grid_w: (width / scale) as i64,
            grid_h: (height / scale) as i64,
            denoise: None,
            txt: None,
            txt_tokens: 0,
            uncond: None,
            sigmas: None,
        }))
    }
}

struct QwenImageSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    config: QwenImageConfig,
    grid_w: i64,
    grid_h: i64,
    denoise: Option<QwenImageForward>,
    txt: Option<Vec<f32>>,
    txt_tokens: i64,
    uncond: Option<(Vec<f32>, i64)>,
    sigmas: Option<Vec<f32>>,
}

impl DitImageSession for QwenImageSession {
    fn encode(&mut self, ctx: &GgmlContext) -> Result<(), EngineError> {
        let files = assert_gguf_bundle(&self.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let guidance = self.request.guidance.unwrap_or(1.0);
        let conditioning = super::encode_conditioning(
            ctx,
            &self.bundle,
            &files.text_encoder,
            self.config.joint_attention_dim,
            &self.request.prompt,
            &self.request.negative_prompt,
            guidance,
        )?;
        let seq_txt = conditioning.txt.len() as i64 / self.config.joint_attention_dim;
        let layout = weight_layout(self.config);
        let mut types = HashMap::new();
        for (name, _, _) in &layout {
            let ggml_type = loaded.ggml_type_of(name)?;
            types.insert(name.clone(), ggml_type);
        }
        let mut denoise =
            QwenImageForward::build(ctx, self.config, self.grid_w, self.grid_h, seq_txt, &types)?;
        for (name, ne0, ne1) in &layout {
            let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
            denoise.load_bytes(name, &raw)?;
        }
        self.txt = Some(conditioning.txt);
        self.txt_tokens = conditioning.txt_tokens;
        self.uncond = conditioning.uncond;
        self.denoise = Some(denoise);
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        let (width, height) =
            danmo_contracts::parse_size(&self.request.size).map_err(EngineError::InvalidRequest)?;
        self.sigmas = Some(super::schedule(
            self.request.steps.unwrap_or(4),
            width,
            height,
            self.config.vae_scale.max(1),
            self.request.scheduler.as_deref(),
            self.request.scheduler_shift,
            self.request.use_empirical_mu,
        )?);
        Ok(())
    }

    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError> {
        let denoise = self
            .denoise
            .as_mut()
            .ok_or_else(|| EngineError::bad_request("encode must run before infer"))?;
        let txt = self
            .txt
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("text encoding is missing"))?;
        let sigmas = self
            .sigmas
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("schedule must run before infer"))?;
        let latent = match self.request.seed {
            Some(seed) => crate::common::ops::normal_latent(seed, denoise.latent_len()),
            None => vec![0.1f32; denoise.latent_len()],
        };
        let uncond = self
            .uncond
            .as_ref()
            .map(|(values, tokens)| (values.as_slice(), *tokens));
        let mut callback_error = None;
        let result = denoise.denoise(
            &latent,
            txt,
            self.txt_tokens,
            uncond,
            sigmas,
            self.request.guidance.unwrap_or(1.0),
            |step, total| {
                if callback_error.is_none() {
                    if let Err(error) = exec.progress("denoise", step, total) {
                        callback_error = Some(error);
                    }
                }
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        let latents = result?;
        if let Some(detail) = crate::common::ops::non_finite("qwen-image latent", &latents) {
            return Err(WeightError::NotBuilt(detail).into());
        }
        Ok(latents)
    }

    fn decode(
        self: Box<Self>,
        ctx: &GgmlContext,
        latents: &[f32],
    ) -> Result<DitDecode, EngineError> {
        let files = assert_gguf_bundle(&self.bundle)?;
        let denoise = self
            .denoise
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("encode must run before decode"))?;
        let (width, height) =
            danmo_contracts::parse_size(&self.request.size).map_err(EngineError::InvalidRequest)?;
        let png = super::vae::decode_packed(
            ctx,
            &files.vae,
            latents,
            self.grid_w,
            self.grid_h,
            width,
            height,
        )?;
        Ok(DitDecode {
            preview_png: png,
            graph_computes: denoise.graph_computes(),
        })
    }
}
