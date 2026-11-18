//! One factory row per registry family. Shape A/B/C matches `docs/engine_architecture.md`.

pub(crate) mod ace_step;
mod driver;
mod flux2;
mod hunyuan;
mod ltx;
mod qwen_image;
pub mod seedvr2;
mod wan;
mod z_image;

pub use crate::contract::{Media, Shape};
pub use driver::{
    DitDecode, DitImageFamily, DitImageSession, FacadeFamily, FacadeSession, FamilyDriver,
};
pub use flux2::write_edit_smoke_bundle as write_flux2_edit_smoke_bundle;
pub use flux2::{
    weight_layout, write_chain_bundle, write_config_bundle, write_q8_bundle, write_qwen3_bundle,
    write_smoke_bundle, Flux2Config, Flux2Denoise,
};
pub use ltx::session::write_smoke_bundle as write_ltx_smoke_bundle;
pub use qwen_image::write_edit_smoke_bundle;
pub use seedvr2::write_smoke_bundle as write_seedvr2_smoke_bundle;
pub use wan::write_smoke_bundle as write_wan_smoke_bundle;
pub use z_image::write_control_smoke_bundle as write_z_image_control_smoke_bundle;
pub use z_image::write_edit_smoke_bundle as write_z_image_edit_smoke_bundle;

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
        shape: Shape::GenerationFacade,
        lora_inference: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ltx",
        media: Media::Video,
        shape: Shape::GenerationFacade,
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
        shape: Shape::GenerationFacade,
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
        lora_inference: true,
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

static FLUX2_FAMILY: flux2::Flux2Family = flux2::Flux2Family;
static FLUX2_EDIT_FAMILY: flux2::Flux2EditFamily = flux2::Flux2EditFamily;
static QWEN_IMAGE_FAMILY: qwen_image::QwenImageFamily = qwen_image::QwenImageFamily;
static QWEN_IMAGE_EDIT_FAMILY: qwen_image::QwenImageEditFamily = qwen_image::QwenImageEditFamily;
static Z_IMAGE_FAMILY: z_image::ZImageFamily = z_image::ZImageFamily;
static Z_IMAGE_EDIT_FAMILY: z_image::ZImageEditFamily = z_image::ZImageEditFamily;
static SEEDVR2_UPSCALE_FAMILY: seedvr2::SeedVr2UpscaleFamily = seedvr2::SeedVr2UpscaleFamily;
static SEEDVR2_VIDEO_UPSCALE_FAMILY: seedvr2::SeedVr2VideoUpscaleFamily =
    seedvr2::SeedVr2VideoUpscaleFamily;
static LTX_VIDEO_CREATE_FAMILY: ltx::LtxVideoCreateFamily = ltx::LtxVideoCreateFamily;
static HUNYUAN_VIDEO_EDIT_FAMILY: hunyuan::HunyuanVideoEditFamily = hunyuan::HunyuanVideoEditFamily;
static WAN_VIDEO_CREATE_FAMILY: wan::WanVideoCreateFamily = wan::WanVideoCreateFamily;
static ACE_STEP_FAMILY: ace_step::AceStepFamily = ace_step::AceStepFamily;

pub fn driver(family: &str, action: crate::contract::Action) -> FamilyDriver {
    use crate::contract::Action;
    match (family, action) {
        ("flux2", Action::ImageCreate) => FamilyDriver::DitImage(&FLUX2_FAMILY),
        ("flux2", Action::ImageEdit) => FamilyDriver::Facade(&FLUX2_EDIT_FAMILY),
        ("qwen_image", Action::ImageCreate) => FamilyDriver::DitImage(&QWEN_IMAGE_FAMILY),
        ("qwen_image", Action::ImageEdit) => FamilyDriver::Facade(&QWEN_IMAGE_EDIT_FAMILY),
        ("z_image", Action::ImageCreate) => FamilyDriver::DitImage(&Z_IMAGE_FAMILY),
        ("z_image", Action::ImageEdit) => FamilyDriver::Facade(&Z_IMAGE_EDIT_FAMILY),
        ("seedvr2", Action::ImageUpscale) => FamilyDriver::Facade(&SEEDVR2_UPSCALE_FAMILY),
        ("seedvr2", Action::VideoUpscale) => FamilyDriver::Facade(&SEEDVR2_VIDEO_UPSCALE_FAMILY),
        ("wan", Action::VideoCreate) => FamilyDriver::Facade(&WAN_VIDEO_CREATE_FAMILY),
        ("ltx", Action::VideoCreate) => FamilyDriver::Facade(&LTX_VIDEO_CREATE_FAMILY),
        ("hunyuan", Action::VideoEdit) => FamilyDriver::Facade(&HUNYUAN_VIDEO_EDIT_FAMILY),
        ("ace_step", Action::AudioCreate | Action::AudioEdit | Action::AudioLm) => {
            FamilyDriver::Facade(&ACE_STEP_FAMILY)
        }
        _ => match spec(family).map(|row| row.shape) {
            Some(Shape::JobPipeline) => FamilyDriver::Job,
            _ => FamilyDriver::Unavailable {
                reason: format!("{family} has no ggml graph for {}", action.as_str()),
            },
        },
    }
}

#[cfg(test)]
pub struct DitImageRun {
    pub latents: Vec<f32>,
    pub graph_computes: u32,
    pub preview_png: Vec<u8>,
}

/// Factory row for Shape A image families. The pipeline does not match on family ids.
#[cfg(test)]
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
        "z_image" => Err(crate::weights::WeightError::NotBuilt(
            "z_image image create no longer has a ggml graph".into(),
        )),
        _ => Ok(None),
    }
}

/// Shape C audio entry. `Ok(None)` leaves the generic bundle check to the
/// pipeline. ACE-Step proves the smoke DiT then refuses product create.
#[cfg(test)]
pub fn run_audio_facade(
    family: &str,
    bundle: &std::path::Path,
    request: &ace_step::CreateRequest<'_>,
    backend: crate::runtime::BackendKind,
    on_step: impl FnMut(u32, u32),
) -> Result<Option<ace_step::CreateOutput>, crate::weights::WeightError> {
    match family {
        "ace_step" => Ok(Some(ace_step::run_create(
            bundle, request, backend, on_step,
        )?)),
        _ => Ok(None),
    }
}

/// Shape C audio edit (cover / cover-nofsq / repaint). `Ok(None)` for families without an edit path.
#[cfg(test)]
pub fn run_audio_edit_facade(
    family: &str,
    bundle: &std::path::Path,
    request: &ace_step::CoverRequest<'_>,
    backend: crate::runtime::BackendKind,
) -> Result<Option<ace_step::CreateOutput>, crate::weights::WeightError> {
    match family {
        "ace_step" => Ok(Some(ace_step::run_cover_edit(bundle, request, backend)?)),
        _ => Ok(None),
    }
}

/// Shape C audio 5Hz LM (inspire / format / generate phase-1).
#[cfg(test)]
pub fn run_audio_lm_facade(
    family: &str,
    bundle: &std::path::Path,
    request: &ace_step::LmRequest<'_>,
    backend: crate::runtime::BackendKind,
) -> Result<Option<ace_step::LmOutput>, crate::weights::WeightError> {
    match family {
        "ace_step" => Ok(Some(ace_step::run_lm(
            bundle,
            request,
            backend,
            &mut |_, _| (),
        )?)),
        _ => Ok(None),
    }
}

#[cfg(test)]
pub(crate) fn ace_step_read_wav(
    bytes: &[u8],
) -> Result<(Vec<f32>, u32), crate::weights::WeightError> {
    ace_step::wav::read_pcm16_interleaved(bytes)
}
