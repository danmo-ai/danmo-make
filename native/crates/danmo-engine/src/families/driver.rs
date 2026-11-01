use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::LoraFile;
use danmo_contracts::ImageGenerationRequest;

pub struct DitDecode {
    pub preview_png: Vec<u8>,
    pub graph_computes: u32,
}

pub trait DitImageSession {
    fn encode(&mut self, ctx: &GgmlContext) -> Result<(), EngineError>;
    fn schedule(&mut self) -> Result<(), EngineError>;
    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError>;
    fn decode(
        self: Box<Self>,
        ctx: &GgmlContext,
        latents: &[f32],
    ) -> Result<DitDecode, EngineError>;
}

pub trait DitImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError>;
}

pub trait FacadeSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError>;
}

pub trait FacadeFamily {
    fn prepare(
        &self,
        ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError>;
}

pub enum FamilyDriver {
    DitImage(&'static dyn DitImageFamily),
    Facade(&'static dyn FacadeFamily),
    Job,
    Unavailable { reason: String },
}
