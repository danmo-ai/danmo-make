use crate::contract::{Artifact, EngineOutput, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::DitImageFamily;
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use danmo_contracts::ImageGenerationRequest;
use serde_json::json;

pub fn run(
    run: &ResolvedRun,
    family: &dyn DitImageFamily,
    request: &ImageGenerationRequest,
    ctx: &GgmlContext,
    exec: &ExecutionContext,
    resolver: &dyn ResolvePort,
) -> Result<EngineOutput, EngineError> {
    let (width, height) =
        danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
    let count = request.n.max(1);
    let mut output = EngineOutput {
        family: run.family.clone(),
        action: run.action.as_str().into(),
        steps: request.steps.unwrap_or(4),
        width: Some(width),
        height: Some(height),
        metadata: json!({
            "model": run.model,
            "prompt": request.prompt,
            "title": request.title,
            "requested_size": request.size,
            "seed": request.seed,
        }),
        ..EngineOutput::default()
    };
    for index in 0..count {
        exec.check_cancelled()?;
        let mut one = request.clone();
        if count > 1 {
            one.seed = Some(request.seed.unwrap_or(0) + index as i64);
        }
        let loras = super::request_loras(&one)?;
        exec.phase("load")?;
        let mut session = family.load(run, &one, loras, resolver)?;
        exec.phase("encode")?;
        session.encode(ctx)?;
        exec.phase("schedule")?;
        session.schedule()?;
        exec.phase("infer")?;
        let latents = session.infer(exec)?;
        exec.phase("decode")?;
        let decoded = session.decode(ctx, &latents)?;
        output.graph_computes += decoded.graph_computes;
        output.latent_len = latents.len();
        let metadata = json!({
            "prompt": one.prompt,
            "title": one.title,
            "model": run.model,
            "requested_size": one.size,
            "seed": one.seed,
            "batch_index": index,
            "batch_count": count,
        });
        output.artifacts.push(Artifact::png(
            decoded.preview_png,
            width as i64,
            height as i64,
            metadata,
        ));
    }
    Ok(output)
}
