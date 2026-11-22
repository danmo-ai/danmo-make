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
        session.encode(ctx, exec)?;
        exec.phase("schedule")?;
        session.schedule()?;
        exec.phase("infer")?;
        let latents = session.infer(exec)?;
        exec.phase("decode")?;
        let decoded = session.decode(ctx, &latents)?;
        output.graph_computes += decoded.graph_computes;
        output.latent_len = latents.len();
        let mut metadata = json!({
            "prompt": one.prompt,
            "title": one.title,
            "model": run.model,
            "requested_size": one.size,
            "seed": one.seed,
            "batch_index": index,
            "batch_count": count,
        });
        apply_request_run_meta(&mut metadata, &one);
        if let Some(values) = decoded.metadata.as_object() {
            for (key, value) in values {
                metadata[key] = value.clone();
            }
        }
        let mut artifact = Artifact::png(
            decoded.preview_png,
            width as i64,
            height as i64,
            metadata.clone(),
        );
        apply_request_lineage(&mut artifact, &mut metadata, &one);
        artifact.metadata = metadata;
        output.artifacts.push(artifact);
    }
    Ok(output)
}

/// Persist run knobs for gallery asset tip (LoRA, steps, CFG, ControlNet, …).
fn apply_request_run_meta(metadata: &mut serde_json::Value, request: &ImageGenerationRequest) {
    if let Some(steps) = request.steps {
        metadata["steps"] = json!(steps);
    }
    if let Some(guidance) = request.guidance {
        metadata["guidance"] = json!(guidance);
    }
    if let Some(scheduler) = request
        .scheduler
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        metadata["scheduler"] = json!(scheduler);
    }
    if request.scheduler_shift.is_some() {
        metadata["scheduler_shift"] = json!(request.scheduler_shift);
    }
    if let Some(use_pe) = request.use_pe {
        metadata["use_pe"] = json!(use_pe);
    }
    let negative = request.negative_prompt.trim();
    if !negative.is_empty() {
        metadata["negative_prompt"] = json!(negative);
    }
    if !request.adapters.is_empty() {
        metadata["adapters"] = json!(request
            .adapters
            .iter()
            .map(|adapter| {
                json!({
                    "id": adapter.id,
                    "weight": adapter.weight,
                })
            })
            .collect::<Vec<_>>());
    }
    if let Some(guide) = &request.structural_guide {
        metadata["structural_guide"] = json!({
            "model_id": guide.model_id,
            "type": guide.guide_type,
            "weight": guide.weight,
            "inpaint": guide.inpaint_mask_asset_id.as_ref().is_some_and(|id| !id.trim().is_empty()),
        });
    }
}

/// Canvas edges / lineage: honor `metadata.parent_asset_id` + `relation_type`
/// from the client (e.g. retouch drawer). Companion inpaint without those
/// fields falls back to the structural-guide inpaint source as parent.
fn apply_request_lineage(
    artifact: &mut Artifact,
    metadata: &mut serde_json::Value,
    request: &ImageGenerationRequest,
) {
    let meta_parent = request
        .metadata
        .get("parent_asset_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    let guide_parent = request
        .structural_guide
        .as_ref()
        .and_then(|guide| guide.inpaint_source_asset_id.as_deref())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    let parent = meta_parent.or(guide_parent);
    let Some(parent) = parent else {
        return;
    };

    let meta_relation = request
        .metadata
        .get("relation_type")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let has_inpaint_mask = request
        .structural_guide
        .as_ref()
        .and_then(|guide| guide.inpaint_mask_asset_id.as_deref())
        .is_some_and(|id| !id.trim().is_empty());
    let relation = meta_relation.unwrap_or_else(|| {
        if has_inpaint_mask {
            "retouch".into()
        } else {
            "create".into()
        }
    });

    artifact.parent_asset_id = Some(parent.clone());
    artifact.relation_type = Some(relation.clone());
    metadata["parent_asset_id"] = json!(parent);
    metadata["relation_type"] = json!(relation);
}

#[cfg(test)]
mod tests {
    use super::*;
    use danmo_contracts::{ImageGenerationRequest, StructuralGuide};

    fn base_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "z-image-turbo".into(),
            title: String::new(),
            prompt: "x".into(),
            negative_prompt: String::new(),
            size: "16x16".into(),
            n: 1,
            steps: Some(1),
            guidance: Some(0.0),
            seed: Some(1),
            scheduler: None,
            scheduler_shift: None,
            use_empirical_mu: false,
            use_pe: None,
            adapters: Vec::new(),
            structural_guide: None,
            metadata: json!({}),
            preview: true,
            priority: "normal".into(),
        }
    }

    #[test]
    fn create_lineage_uses_request_metadata() {
        let mut request = base_request();
        request.metadata = json!({
            "parent_asset_id": "ast_parent",
            "relation_type": "retouch",
        });
        let mut artifact = Artifact::png(vec![1, 2, 3], 16, 16, json!({}));
        let mut metadata = json!({});
        apply_request_lineage(&mut artifact, &mut metadata, &request);
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_parent"));
        assert_eq!(artifact.relation_type.as_deref(), Some("retouch"));
        assert_eq!(metadata["parent_asset_id"], json!("ast_parent"));
        assert_eq!(metadata["relation_type"], json!("retouch"));
    }

    #[test]
    fn create_lineage_falls_back_to_inpaint_source() {
        let mut request = base_request();
        request.structural_guide = Some(StructuralGuide {
            asset_id: "ast_ctrl".into(),
            model_id: "z-image-turbo-fun-controlnet-union:8steps".into(),
            guide_type: "auto".into(),
            weight: 0.75,
            inpaint_source_asset_id: Some("ast_src".into()),
            inpaint_mask_asset_id: Some("ast_mask".into()),
        });
        let mut artifact = Artifact::png(vec![1, 2, 3], 16, 16, json!({}));
        let mut metadata = json!({});
        apply_request_lineage(&mut artifact, &mut metadata, &request);
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_src"));
        assert_eq!(artifact.relation_type.as_deref(), Some("retouch"));
    }
}
