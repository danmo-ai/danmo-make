//! One factory row per registry family. Shape A/B/C matches `docs/engine_architecture.md`.

mod ace_step;
mod flux2;
mod qwen_image;
mod z_image;

pub use flux2::{
    weight_layout, write_chain_bundle, write_config_bundle, write_q8_bundle, write_qwen3_bundle,
    write_smoke_bundle, Flux2Config, Flux2Denoise,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    Image,
    Video,
    Audio,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Standard denoise. Entry: ImagePipeline / VideoPipeline.
    DitPipeline,
    /// Upscale / non-standard job. Entry: UpscalePipeline.
    JobPipeline,
    /// End-to-end generator. Entry: MusicPipeline or a family generator.
    GenerationFacade,
}

#[derive(Debug, Clone, Copy)]
pub struct FamilySpec {
    pub id: &'static str,
    pub media: Media,
    pub shape: Shape,
    /// True when create can attach an in-graph LoRA residual.
    pub lora_inference: bool,
    /// Scheduler names this family's create path accepts. Empty means none.
    pub schedulers: &'static [&'static str],
}

impl FamilySpec {
    pub fn accepts_scheduler(self, name: &str) -> bool {
        self.schedulers.iter().any(|s| *s == name)
    }
}

const FLOW_MATCH: &[&str] = &["flow_match_euler", "flow_match_euler_discrete"];
const FLOW_MATCH_LINEAR: &[&str] = &["linear", "flow_match_euler", "flow_match_euler_discrete"];

const FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        id: "flux1",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "flux2",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: true,
        schedulers: FLOW_MATCH,
    },
    FamilySpec {
        id: "qwen_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: FLOW_MATCH_LINEAR,
    },
    FamilySpec {
        id: "fibo",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "z_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: true,
        schedulers: FLOW_MATCH_LINEAR,
    },
    FamilySpec {
        id: "ernie_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "cogview4",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "hidream_o1",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "step1x_edit",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "boogu_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "longcat",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "seedvr2",
        media: Media::Image,
        shape: Shape::JobPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "esrgan",
        media: Media::Image,
        shape: Shape::JobPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "wan",
        media: Media::Video,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ltx",
        media: Media::Video,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ltx25",
        media: Media::Video,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "hunyuan",
        media: Media::Video,
        shape: Shape::DitPipeline,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "minimax_h3",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "longcat_avatar",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ace_step",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "diffrhythm",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "qwen3",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "qwen3_vl",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
];

pub fn spec(family: &str) -> Option<&'static FamilySpec> {
    FAMILIES.iter().find(|f| f.id == family)
}

pub fn all_specs() -> &'static [FamilySpec] {
    FAMILIES
}

pub struct DitImageRun {
    pub latents: Vec<f32>,
    pub graph_computes: u32,
    pub preview_png: Vec<u8>,
}

/// Factory row for Shape A image families. The pipeline does not match on family ids.
pub fn run_dit_image(
    family: &str,
    ctx: &crate::runtime::GgmlContext,
    bundle: &std::path::Path,
    prompt: &str,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    seed: Option<i64>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
    negative: &str,
    loras: &[crate::weights::LoraFile],
    on_progress: impl FnMut(u32, u32),
) -> Result<Option<DitImageRun>, crate::weights::WeightError> {
    match family {
        "flux2" => Ok(Some(flux2::execute(
            ctx,
            bundle,
            prompt,
            steps,
            width,
            height,
            guidance,
            seed,
            scheduler,
            negative,
            loras,
            on_progress,
        )?)),
        "qwen_image" => Ok(Some(qwen_image::execute(
            ctx,
            bundle,
            prompt,
            steps,
            width,
            height,
            guidance,
            seed,
            scheduler,
            scheduler_shift,
            empirical_mu,
            negative,
            loras,
            on_progress,
        )?)),
        "z_image" => Ok(Some(z_image::execute(
            ctx,
            bundle,
            prompt,
            steps,
            width,
            height,
            guidance,
            seed,
            scheduler,
            scheduler_shift,
            empirical_mu,
            negative,
            loras,
            on_progress,
        )?)),
        _ => Ok(None),
    }
}

/// Shape C audio entry. `Ok(None)` leaves the generic bundle check to the
/// pipeline. ACE-Step validates the DiT GGUF then refuses generate.
pub fn run_audio_facade(
    family: &str,
    bundle: &std::path::Path,
) -> Result<Option<()>, crate::weights::WeightError> {
    match family {
        "ace_step" => {
            let _ = ace_step::probe_layout(bundle)?;
            Err(crate::weights::WeightError::NotBuilt(
                "ACE-Step DiT graph is not built".into(),
            ))
        }
        _ => Ok(None),
    }
}
