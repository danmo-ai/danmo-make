use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun, Shape};
use crate::error::EngineError;
use crate::families::{self, FamilyDriver};
use crate::pipeline;
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;

pub fn ensure_available(family: &str, action: crate::contract::Action) -> Result<(), EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != action.media() {
        return Err(EngineError::UnsupportedAction {
            family: family.into(),
            action: action.as_str().into(),
        });
    }
    match families::driver(family, action) {
        FamilyDriver::DitImage(_) | FamilyDriver::Facade(_) => Ok(()),
        FamilyDriver::Job => Err(EngineError::GraphNotBuilt {
            family: family.into(),
            shape: "job_pipeline".into(),
        }),
        FamilyDriver::Unavailable { reason } => Err(EngineError::GraphNotBuilt {
            family: family.into(),
            shape: reason,
        }),
    }
}

pub fn execute(
    ctx: &GgmlContext,
    run: &ResolvedRun,
    request: &EngineRequest,
    resolver: &dyn ResolvePort,
    exec: &ExecutionContext,
) -> Result<EngineOutput, EngineError> {
    let spec = families::spec(&run.family)
        .ok_or_else(|| EngineError::UnknownFamily(run.family.clone()))?;
    if spec.media != run.action.media() {
        return Err(EngineError::UnsupportedAction {
            family: run.family.clone(),
            action: run.action.as_str().into(),
        });
    }
    match families::driver(&run.family, run.action) {
        FamilyDriver::DitImage(family) => {
            let EngineRequest::ImageCreate(request) = request else {
                return Err(EngineError::bad_request("DiT driver requires image create"));
            };
            if spec.shape != Shape::DitPipeline {
                return Err(EngineError::bad_request("DiT driver shape mismatch"));
            }
            pipeline::dit::run(run, family, request, ctx, exec, resolver)
        }
        FamilyDriver::Facade(family) => {
            pipeline::facade::run(ctx, run, family, request, resolver, exec)
        }
        FamilyDriver::Job => Err(EngineError::GraphNotBuilt {
            family: run.family.clone(),
            shape: "job_pipeline".into(),
        }),
        FamilyDriver::Unavailable { reason } => Err(EngineError::GraphNotBuilt {
            family: run.family.clone(),
            shape: reason,
        }),
    }
}
