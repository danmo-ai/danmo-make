//! One factory row per registry family. Shape A/B/C matches `docs/engine_architecture.md`.

pub(crate) mod ace_step;
mod driver;
mod esrgan;
mod flux2;
mod hunyuan;
mod longcat_avatar;
mod ltx25;
mod qwen_image;
pub mod seedvr2;
mod wan;
pub mod z_image;

pub use crate::contract::{Media, Shape};
pub use driver::{
    DitDecode, DitImageFamily, DitImageSession, FacadeFamily, FacadeSession, FamilyDriver,
};
pub use seedvr2::write_smoke_bundle as write_seedvr2_smoke_bundle;
pub use wan::write_smoke_bundle as write_wan_smoke_bundle;
pub use z_image::write_control_smoke_bundle as write_z_image_control_smoke_bundle;
pub use z_image::write_edit_smoke_bundle as write_z_image_edit_smoke_bundle;

/// Execute one registry-declared install hook. Family dispatch stays here;
/// downloads do not import checkpoint layouts or model implementations.
/// Callbacks are owned so the hook can run inside the MLX executor thread.
pub fn run_install_hook(
    hook_type: &str,
    bundle_root: &std::path::Path,
    hook_spec: &serde_json::Value,
    cancel: Box<dyn Fn() -> bool + Send>,
    log: Box<dyn Fn(&str) + Send>,
) -> Result<(), String> {
    if hook_type != "ltx25_ingest" {
        return Err(format!(
            "unsupported registry install hook type {hook_type}"
        ));
    }
    #[cfg(feature = "mlx")]
    {
        return ltx25::bundle::ingest_from_hook(bundle_root, hook_spec, cancel, log);
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (bundle_root, hook_spec, cancel, log);
        Err("LTX 2.5 install requires a binary built with MLX support".into())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FamilySpec {
    pub id: &'static str,
    pub media: Media,
    pub shape: Shape,
    /// True when create can attach an in-graph LoRA residual.
    pub lora_inference: bool,
    /// True when create accepts a structural guide (ControlNet-style hint).
    pub structural_guide: bool,
    /// True when the family wires a LoRA training runner.
    pub lora_training: bool,
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
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "flux2",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: true,
        structural_guide: false,
        lora_training: false,
        schedulers: FLOW_MATCH,
    },
    FamilySpec {
        id: "qwen_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: true,
        structural_guide: false,
        lora_training: false,
        schedulers: FLOW_MATCH_LINEAR,
    },
    FamilySpec {
        id: "fibo",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "z_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: true,
        structural_guide: true,
        lora_training: true,
        schedulers: FLOW_MATCH_LINEAR,
    },
    FamilySpec {
        id: "ernie_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "cogview4",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "hidream_o1",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "step1x_edit",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "boogu_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "longcat",
        media: Media::Image,
        shape: Shape::DitPipeline,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "seedvr2",
        media: Media::Image,
        // Driven by the facade runner (`SeedVr2UpscaleFamily`); no shared job
        // runner exists, so the spec shape must not claim JobPipeline.
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "esrgan",
        media: Media::Image,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "wan",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ltx25",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: true,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "hunyuan",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "minimax_h3",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "minimax_h3_image",
        media: Media::Image,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "longcat_avatar",
        media: Media::Video,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "ace_step",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
        lora_inference: true,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "diffrhythm",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "qwen3",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
        schedulers: &[],
    },
    FamilySpec {
        id: "qwen3_vl",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
        lora_inference: false,
        structural_guide: false,
        lora_training: false,
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
static ESRGAN_UPSCALE_FAMILY: esrgan::EsrganUpscaleFamily = esrgan::EsrganUpscaleFamily;
static SEEDVR2_VIDEO_UPSCALE_FAMILY: seedvr2::SeedVr2VideoUpscaleFamily =
    seedvr2::SeedVr2VideoUpscaleFamily;
static LTX25_VIDEO_CREATE_FAMILY: ltx25::Ltx25VideoCreateFamily = ltx25::Ltx25VideoCreateFamily;
static LONGCAT_AVATAR_FAMILY: longcat_avatar::LongcatAvatarFamily =
    longcat_avatar::LongcatAvatarFamily;
static HUNYUAN_VIDEO_EDIT_FAMILY: hunyuan::HunyuanVideoEditFamily = hunyuan::HunyuanVideoEditFamily;
static WAN_VIDEO_CREATE_FAMILY: wan::WanVideoCreateFamily = wan::WanVideoCreateFamily;
static ACE_STEP_FAMILY: ace_step::AceStepFamily = ace_step::AceStepFamily;

/// Family dispatch for LoRA training jobs. Same rule as [`driver`]: this
/// match is the only family-id branch for training; callers pass the family
/// resolved from the catalog, never a hardcoded id.
///
/// Full pipeline: the family runner owns bundle loading, dataset encode
/// caches, the differentiable DiT loop, and adapter export on the MLX
/// executor. `dataset` is `(image file, caption)` with the caption policy
/// already applied by the caller.
#[cfg(feature = "mlx")]
pub fn run_family_lora_training(
    family: &str,
    request: &danmo_contracts::LoraTrainingRequest,
    bundle: &std::path::Path,
    dataset: &[(std::path::PathBuf, String)],
    resume_adapter: Option<&std::path::Path>,
    work_dir: &std::path::Path,
    progress: crate::train::TrainProgress,
) -> Result<danmo_contracts::LoraTrainingResult, crate::error::EngineError> {
    match family {
        "z_image" => z_image::train::run_z_image_full_training(
            request,
            bundle,
            dataset,
            resume_adapter,
            work_dir,
            progress,
        ),
        _ => Err(crate::error::EngineError::GraphNotBuilt {
            family: family.into(),
            shape: "lora_training".into(),
        }),
    }
}

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
        ("esrgan", Action::ImageUpscale) => FamilyDriver::Facade(&ESRGAN_UPSCALE_FAMILY),
        ("seedvr2", Action::VideoUpscale) => FamilyDriver::Facade(&SEEDVR2_VIDEO_UPSCALE_FAMILY),
        ("wan", Action::VideoCreate) => FamilyDriver::Facade(&WAN_VIDEO_CREATE_FAMILY),
        ("longcat_avatar", Action::VideoAvatar) => FamilyDriver::Facade(&LONGCAT_AVATAR_FAMILY),
        ("ltx25", Action::VideoCreate | Action::VideoEdit) => {
            FamilyDriver::Facade(&LTX25_VIDEO_CREATE_FAMILY)
        }
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
        "flux2" => {
            let _ = (
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
            );
            let _ = on_progress;
            Err(crate::weights::WeightError::NotBuilt(
                "flux2 image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                    .into(),
            ))
        }
        "qwen_image" => {
            let _ = (
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
            );
            let _ = on_progress;
            Err(crate::weights::WeightError::NotBuilt(
                "qwen-image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                    .into(),
            ))
        }
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
