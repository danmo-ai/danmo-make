use super::transformer::Flux2Denoise;
use super::weights::{stored_f32, weight_layout, Flux2Config};
use crate::contract::{ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{
    assert_gguf_bundle, load_lora_factors, matmul_type, type_block, GgufWeights, LoraFile,
    WeightError,
};
use danmo_contracts::ImageGenerationRequest;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Flux2Family;

impl DitImageFamily for Flux2Family {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        if request.structural_guide.is_some() {
            return Err(WeightError::NotBuilt(
                "structural guide graph is not built for flux2".into(),
            )
            .into());
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(WeightError::NotBuilt(
                "negative prompt is not built for this family".into(),
            )
            .into());
        }
        match request
            .scheduler
            .as_deref()
            .unwrap_or("flow_match_euler_discrete")
        {
            "flow_match_euler" | "flow_match_euler_discrete" => {}
            other => {
                return Err(WeightError::NotBuilt(format!("scheduler {other} is not built")).into())
            }
        }
        let files = assert_gguf_bundle(&run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let config = super::config_from_weights(&loaded)?;
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        let scale = config.vae_scale.max(1);
        if width % scale != 0 || height % scale != 0 {
            return Err(WeightError::NotBuilt(format!(
                "size {width}x{height} is not divisible by vae scale {scale}"
            ))
            .into());
        }
        if let Some(max_edge) = super::manifest_max_edge(&run.bundle)? {
            if width > max_edge || height > max_edge {
                return Err(WeightError::NotBuilt(format!(
                    "this bundle accepts edges up to {max_edge}, got {width}x{height}"
                ))
                .into());
            }
        }
        Ok(Box::new(Flux2Session {
            bundle: run.bundle.clone(),
            request: request.clone(),
            loras,
            config,
            grid_w: (width / scale) as i64,
            grid_h: (height / scale) as i64,
            denoise: None,
            txt: None,
            sigmas: None,
        }))
    }
}

struct Flux2Session {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    loras: Vec<LoraFile>,
    config: Flux2Config,
    grid_w: i64,
    grid_h: i64,
    denoise: Option<Flux2Denoise>,
    txt: Option<Vec<f32>>,
    sigmas: Option<Vec<f32>>,
}

impl DitImageSession for Flux2Session {
    fn encode(&mut self, ctx: &GgmlContext) -> Result<(), EngineError> {
        let files = assert_gguf_bundle(&self.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let tokenizer = crate::common::qwen3::load_tokenizer(&self.bundle)?;
        let (txt, seq_txt) = super::text_tokens(
            ctx,
            &files.text_encoder,
            &tokenizer,
            &self.request.prompt,
            self.config.joint_attention_dim,
        )?;
        let layout = weight_layout(self.config);
        let factors = load_lora_factors(&self.loras, &layout)?;
        let mut types = HashMap::new();
        let mut stored = Vec::new();
        for (name, ne0, ne1) in &layout {
            let ggml_type = loaded.ggml_type_of(name)?;
            check_stored_type(name, ggml_type, *ne0)?;
            types.insert(name.clone(), ggml_type);
            stored.push((name.clone(), loaded.raw_exact(name, &[*ne0, *ne1])?));
        }
        let mut denoise = Flux2Denoise::build_typed(
            ctx,
            self.config,
            self.grid_w,
            self.grid_h,
            seq_txt,
            &types,
            &factors,
        )?;
        for (name, raw) in &stored {
            denoise.load_bytes(name, raw)?;
        }
        for (index, factor) in factors.iter().enumerate() {
            denoise.load_f32(&format!("lora.{index}.a"), &factor.a)?;
            denoise.load_f32(&format!("lora.{index}.b"), &factor.b)?;
        }
        self.txt = Some(txt);
        self.denoise = Some(denoise);
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        self.sigmas = Some(super::schedule(
            self.request.steps.unwrap_or(4),
            self.grid_w,
            self.grid_h,
        ));
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
        let mut callback_error = None;
        let result = denoise.denoise(
            latent,
            txt,
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
        Ok(result?)
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
        let png = super::vae::decode(
            ctx,
            &files.vae,
            latents,
            self.config.in_channels,
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

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if stored_f32(name) {
        if ggml_type != crate::runtime::TYPE_F32 {
            return Err(WeightError::NotBuilt(format!(
                "{name} is ggml type {ggml_type}. Norm weights stay F32 inside a quantized GGUF"
            )));
        }
        return Ok(());
    }
    if !matmul_type(ggml_type) {
        return Err(WeightError::Type {
            name: name.to_string(),
            ggml_type,
        });
    }
    let block = type_block(ggml_type);
    if block > 1 && ne0 % block != 0 {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}
