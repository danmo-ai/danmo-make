//! One factory row per registry family. Shape A/B/C matches `docs/engine_architecture.md`.

mod flux2;

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
}

const FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        id: "flux1",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "flux2",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "qwen_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "fibo",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "z_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "ernie_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "cogview4",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "hidream_o1",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "step1x_edit",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "boogu_image",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "longcat",
        media: Media::Image,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "seedvr2",
        media: Media::Image,
        shape: Shape::JobPipeline,
    },
    FamilySpec {
        id: "esrgan",
        media: Media::Image,
        shape: Shape::JobPipeline,
    },
    FamilySpec {
        id: "wan",
        media: Media::Video,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "ltx",
        media: Media::Video,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "ltx25",
        media: Media::Video,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "hunyuan",
        media: Media::Video,
        shape: Shape::DitPipeline,
    },
    FamilySpec {
        id: "minimax_h3",
        media: Media::Video,
        shape: Shape::GenerationFacade,
    },
    FamilySpec {
        id: "longcat_avatar",
        media: Media::Video,
        shape: Shape::GenerationFacade,
    },
    FamilySpec {
        id: "ace_step",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
    },
    FamilySpec {
        id: "diffrhythm",
        media: Media::Audio,
        shape: Shape::GenerationFacade,
    },
    FamilySpec {
        id: "qwen3",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
    },
    FamilySpec {
        id: "qwen3_vl",
        media: Media::Assistant,
        shape: Shape::GenerationFacade,
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
    bundle: &std::path::Path,
    prompt: &str,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    backend: crate::runtime::BackendKind,
    on_progress: impl FnMut(u32, u32),
) -> Result<Option<DitImageRun>, crate::weights::WeightError> {
    match family {
        "flux2" => Ok(Some(flux2::run(
            bundle,
            prompt,
            steps,
            width,
            height,
            guidance,
            backend,
            on_progress,
        )?)),
        _ => Ok(None),
    }
}
