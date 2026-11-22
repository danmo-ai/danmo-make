use super::{CoverMode, CoverRequest, CreateOutput, CreateRequest, LmMode, LmRequest};
use crate::contract::{
    Artifact, AudioCreateRequest, AudioEditRequest, AudioLmRequest, EngineOutput, EngineRequest,
    ExecutionContext, ResolvedRun, Sidecar,
};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct AceStepFamily;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CreateOptions {
    bpm: Option<i32>,
    keyscale: String,
    timesignature: String,
    lm_expansion: String,
    audio_code_indices: Option<Vec<i32>>,
    timbre_latents: Option<Vec<f32>>,
    audio_format: String,
    denoise: String,
    dcw_enabled: Option<bool>,
    dcw_mode: String,
    dcw_wavelet: String,
    dcw_scaler: f32,
    dcw_high_scaler: f32,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            bpm: None,
            keyscale: String::new(),
            timesignature: String::new(),
            lm_expansion: "auto".into(),
            audio_code_indices: None,
            timbre_latents: None,
            audio_format: "wav".into(),
            denoise: "off".into(),
            dcw_enabled: None,
            dcw_mode: "double".into(),
            dcw_wavelet: "haar".into(),
            dcw_scaler: 0.05,
            dcw_high_scaler: 0.02,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct EditOptions {
    task: String,
    source_latents: Option<Vec<f32>>,
    audio_cover_strength: f32,
    cover_noise_strength: f32,
    region_start_sec: Option<f32>,
    region_end_sec: Option<f32>,
    repaint_injection_ratio: f32,
    repaint_crossfade_frames: i64,
    timbre_from_source: bool,
    track: String,
    audio_format: String,
    denoise: String,
    dcw_enabled: Option<bool>,
    dcw_mode: String,
    dcw_wavelet: String,
    dcw_scaler: f32,
    dcw_high_scaler: f32,
}

impl Default for EditOptions {
    fn default() -> Self {
        Self {
            task: "cover-nofsq".into(),
            source_latents: None,
            audio_cover_strength: 1.0,
            cover_noise_strength: 0.0,
            region_start_sec: None,
            region_end_sec: None,
            repaint_injection_ratio: 1.0,
            repaint_crossfade_frames: 8,
            timbre_from_source: false,
            track: String::new(),
            audio_format: "wav".into(),
            denoise: "off".into(),
            dcw_enabled: None,
            dcw_mode: "double".into(),
            dcw_wavelet: "haar".into(),
            dcw_scaler: 0.05,
            dcw_high_scaler: 0.02,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LmOptions {
    duration: f32,
    bpm: i32,
    keyscale: String,
    timesignature: String,
    vocal_language: String,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u64,
    use_fsm: bool,
    cfg_scale: f32,
    negative_prompt: String,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            duration: 0.0,
            bpm: 0,
            keyscale: String::new(),
            timesignature: String::new(),
            vocal_language: "en".into(),
            max_new_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            use_fsm: false,
            cfg_scale: 1.0,
            negative_prompt: String::new(),
        }
    }
}

enum Prepared {
    Create {
        request: AudioCreateRequest,
        options: CreateOptions,
        expanded: Option<super::LmExpandOutput>,
    },
    Edit {
        request: AudioEditRequest,
        options: EditOptions,
        source_audio: Option<Vec<f32>>,
    },
    Lm {
        request: AudioLmRequest,
        options: LmOptions,
    },
}

pub struct AceStepSession {
    run: ResolvedRun,
    prepared: Prepared,
}

impl FacadeFamily for AceStepFamily {
    fn prepare(
        &self,
        ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let prepared = match request {
            EngineRequest::AudioCreate(request) => {
                let mut request = normalize_create_vocals(request, exec)?;
                if request.seed.is_none() {
                    request.seed = Some(drawn_seed());
                }
                let options: CreateOptions = parse_options(&request.family_options)?;
                let expanded = if options.lm_expansion.eq_ignore_ascii_case("off") {
                    None
                } else {
                    // The 5Hz LM is not a registry card. Its safetensors live
                    // inside the ACE bundle (acestep-5Hz-lm-*). A GGUF companion
                    // is ignored; maybe_expand fails loud when the tower is missing.
                    let lm_bundle = super::resolve_lm_bundle_for_create(&run.bundle, None).ok();
                    let mut callback_error = None;
                    exec.phase("lm")?;
                    let expanded = super::maybe_expand_for_create_with_context(
                        &run.bundle,
                        &super::LmExpandRequest {
                            mode: &options.lm_expansion,
                            caption: &request.prompt,
                            lyrics: &request.lyrics,
                            duration: request.duration_sec,
                            bpm: options.bpm.unwrap_or(0),
                            keyscale: &options.keyscale,
                            timesignature: &options.timesignature,
                            vocal_language: &request.vocal_language,
                            seed: request
                                .seed
                                .expect("create seed is drawn before LM expansion")
                                as u64,
                            lm_bundle: lm_bundle.as_deref(),
                            existing_codes: options.audio_code_indices.as_deref(),
                        },
                        ctx,
                        |token, total| {
                            if callback_error.is_none() {
                                if let Err(error) = exec.progress("lm", token as u32, total as u32)
                                {
                                    callback_error = Some(error);
                                }
                            }
                        },
                    );
                    if let Some(error) = callback_error {
                        return Err(error);
                    }
                    Some(expanded?)
                };
                Prepared::Create {
                    request,
                    options,
                    expanded,
                }
            }
            EngineRequest::AudioEdit(request) => {
                let request = normalize_edit_vocals(request, exec)?;
                let options: EditOptions = parse_options(&request.family_options)?;
                CoverMode::parse(&options.task)?;
                let source_audio = if options
                    .source_latents
                    .as_ref()
                    .is_some_and(|values| !values.is_empty())
                {
                    None
                } else {
                    let source = request
                        .source_audio
                        .as_ref()
                        .filter(|value| !value.is_empty())
                        .cloned()
                        .or_else(|| {
                            request
                                .source_audio_base64
                                .as_ref()
                                .filter(|value| !value.is_empty())
                                .map(|value| format!("base64:{value}"))
                        })
                        .ok_or_else(|| {
                            EngineError::bad_request(
                                "audio edit needs source_audio, source_audio_base64, or source_latents",
                            )
                        })?;
                    let bytes = resolver.read_source_audio(run, &source)?;
                    let decoded = super::wav::read_pcm16_interleaved(&bytes)
                        .ok()
                        .filter(|(_, sample_rate)| *sample_rate == 48_000);
                    let (pcm, _) = match decoded {
                        Some(decoded) => decoded,
                        None => {
                            let normalized =
                                crate::common::ffmpeg::decode_audio_to_pcm16_wav(&bytes)?;
                            super::wav::read_pcm16_interleaved(&normalized)?
                        }
                    };
                    Some(pcm)
                };
                Prepared::Edit {
                    request,
                    options,
                    source_audio,
                }
            }
            EngineRequest::AudioLm(request) => {
                let options: LmOptions = parse_options(&request.family_options)?;
                LmMode::parse(&request.mode)?;
                Prepared::Lm {
                    request: request.clone(),
                    options,
                }
            }
            _ => {
                return Err(EngineError::bad_request(
                    "ACE-Step received a non-audio request",
                ))
            }
        };
        Ok(Box::new(AceStepSession {
            run: run.clone(),
            prepared,
        }))
    }
}

impl FacadeSession for AceStepSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        match &mut self.prepared {
            Prepared::Create {
                request,
                options,
                expanded,
            } => {
                let expanded = expanded.take();
                let loras = crate::pipeline::lora_files(&request.adapters)?;
                let dcw = dcw_settings(
                    options.dcw_enabled,
                    &options.dcw_mode,
                    &options.dcw_wavelet,
                    options.dcw_scaler,
                    options.dcw_high_scaler,
                )?;
                let encoding = super::audio_codec::AudioEncoding::new(
                    &options.audio_format,
                    &options.denoise,
                )?;
                let batch = validate_batch(request.n)?;
                let mut outputs = Vec::with_capacity(batch);
                exec.phase("denoise")?;
                for index in 0..batch {
                    let create =
                        create_request(request, options, expanded.as_ref(), &loras, dcw, index);
                    let mut callback_error = None;
                    let out = super::run_create_with_context(
                        &self.run.bundle,
                        &create,
                        ctx,
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
                    let out = out?;
                    if let Some(message) = out.quality.log_message() {
                        exec.log("warning", &message)?;
                    }
                    let mut output = audio_output(out, &encoding)?;
                    decorate_create_output(&mut output, request, options, index);
                    outputs.push(output);
                }
                Ok(merge_audio_outputs(outputs, &request.metadata))
            }
            Prepared::Edit {
                request,
                options,
                source_audio,
            } => {
                let loras = crate::pipeline::lora_files(&request.adapters)?;
                let dcw = dcw_settings(
                    options.dcw_enabled,
                    &options.dcw_mode,
                    &options.dcw_wavelet,
                    options.dcw_scaler,
                    options.dcw_high_scaler,
                )?;
                let encoding = super::audio_codec::AudioEncoding::new(
                    &options.audio_format,
                    &options.denoise,
                )?;
                let batch = validate_batch(request.n)?;
                let mut outputs = Vec::with_capacity(batch);
                exec.phase("denoise")?;
                for index in 0..batch {
                    let mut callback_error = None;
                    let out = super::run_cover_edit_with_context(
                        &self.run.bundle,
                        &CoverRequest {
                            prompt: &request.prompt,
                            lyrics: &request.lyrics,
                            duration_sec: request.duration_sec,
                            seed: request.seed.map(|seed| seed + index as i64),
                            guidance: request.guidance,
                            steps: request.steps,
                            vocal_language: &request.vocal_language,
                            bpm: request.bpm,
                            keyscale: &request.keyscale,
                            timesignature: &request.timesignature,
                            mode: CoverMode::parse(&options.task)?,
                            source_audio: source_audio.as_deref(),
                            source_latents: options.source_latents.as_deref(),
                            audio_cover_strength: options.audio_cover_strength,
                            cover_noise_strength: options.cover_noise_strength,
                            region_start_sec: options.region_start_sec,
                            region_end_sec: options.region_end_sec,
                            repaint_injection_ratio: options.repaint_injection_ratio,
                            repaint_crossfade_frames: options.repaint_crossfade_frames,
                            timbre_from_source: options.timbre_from_source,
                            track: &options.track,
                            loras: &loras,
                            dcw,
                        },
                        ctx,
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
                    let out = out?;
                    if let Some(message) = out.quality.log_message() {
                        exec.log("warning", &message)?;
                    }
                    let mut output = audio_output(out, &encoding)?;
                    output.metadata["task"] = json!(options.task);
                    for artifact in &mut output.artifacts {
                        artifact.source_action = "edit".into();
                    }
                    decorate_edit_output(&mut output, request, options, index);
                    outputs.push(output);
                }
                Ok(merge_audio_outputs(outputs, &request.metadata))
            }
            Prepared::Lm { request, options } => {
                exec.phase("lm")?;
                let out = super::run_lm_with_context(
                    &self.run.bundle,
                    &LmRequest {
                        mode: LmMode::parse(&request.mode)?,
                        caption: &request.prompt,
                        lyrics: &request.lyrics,
                        duration: options.duration,
                        bpm: options.bpm,
                        keyscale: &options.keyscale,
                        timesignature: &options.timesignature,
                        vocal_language: &options.vocal_language,
                        max_new_tokens: options.max_new_tokens.max(1),
                        tokenizer_json: None,
                        temperature: options.temperature,
                        top_p: options.top_p,
                        top_k: options.top_k,
                        seed: options.seed,
                        use_fsm: options.use_fsm,
                        cfg_scale: options.cfg_scale,
                        negative_prompt: &options.negative_prompt,
                    },
                    ctx,
                    &mut |token, total| {
                        let _ = exec.progress("lm", token as u32, total as u32);
                    },
                )?;
                Ok(EngineOutput {
                    metadata: json!({
                        "mode": out.mode,
                        "prompt_tokens": out.prompt_tokens,
                        "generated_ids": out.generated_ids,
                        "text": out.text,
                        "caption": out.caption,
                        "lyrics": out.lyrics,
                        "bpm": out.bpm,
                        "duration": out.duration,
                        "keyscale": out.keyscale,
                        "timesignature": out.timesignature,
                        "language": out.language,
                        "audio_code_indices": out.audio_code_indices,
                        "pmi_enabled": super::quality::pmi_scoring_enabled(),
                        "pmi": out.pmi.as_ref().map(|pmi| pmi.as_metadata()).unwrap_or(Value::Null),
                    }),
                    ..EngineOutput::default()
                })
            }
        }
    }
}

fn parse_options<T: serde::de::DeserializeOwned>(value: &Value) -> Result<T, EngineError> {
    serde_json::from_value(value.clone())
        .map_err(|error| EngineError::bad_request(error.to_string()))
}

fn normalize_create_vocals(
    request: &AudioCreateRequest,
    exec: &ExecutionContext,
) -> Result<AudioCreateRequest, EngineError> {
    let (prompt, lyrics, vocal_language) = vocal_fields(
        &request.prompt,
        &request.lyrics,
        &request.vocal_language,
        &request.vocal_type,
        request.instrumental,
        true,
        exec,
    )?;
    let mut normalized = request.clone();
    normalized.prompt = prompt;
    normalized.lyrics = lyrics;
    normalized.vocal_language = vocal_language;
    Ok(normalized)
}

fn normalize_edit_vocals(
    request: &AudioEditRequest,
    exec: &ExecutionContext,
) -> Result<AudioEditRequest, EngineError> {
    let (prompt, lyrics, vocal_language) = vocal_fields(
        &request.prompt,
        &request.lyrics,
        &request.vocal_language,
        &request.vocal_type,
        request.instrumental,
        false,
        exec,
    )?;
    let mut normalized = request.clone();
    normalized.prompt = prompt;
    normalized.lyrics = lyrics;
    normalized.vocal_language = vocal_language;
    Ok(normalized)
}

fn vocal_fields(
    prompt: &str,
    lyrics: &str,
    vocal_language: &str,
    vocal_type: &str,
    instrumental: bool,
    require_lyrics: bool,
    exec: &ExecutionContext,
) -> Result<(String, String, String), EngineError> {
    let raw = lyrics.trim();
    let instrumental = instrumental
        || (super::lyrics::is_instrumental_lyrics(raw) && (!require_lyrics || !raw.is_empty()));
    if require_lyrics && !instrumental && raw.is_empty() {
        return Err(EngineError::bad_request(
            "ACE-Step vocal create needs lyrics or instrumental=true",
        ));
    }
    let lyrics = if instrumental {
        "[Instrumental]".to_string()
    } else {
        raw.to_string()
    };
    let language = super::vocal::resolve_vocal_language(&lyrics, vocal_language);
    if super::vocal::lyrics_looks_chinese(&lyrics) && language.starts_with("en") {
        exec.log(
            "warning",
            "ACE-Step lyrics look Chinese but vocal_language is en; use zh for reliable vocals",
        )?;
    }
    let (prompt, log) =
        super::vocal::apply_vocal_type_to_prompt(prompt, vocal_type, &language, instrumental);
    if let Some(log) = log {
        exec.log("info", &log)?;
    }
    Ok((prompt, lyrics, language))
}

fn dcw_settings(
    enabled: Option<bool>,
    mode: &str,
    wavelet: &str,
    scaler: f32,
    high_scaler: f32,
) -> Result<super::DcwSettings, EngineError> {
    if !wavelet.trim().eq_ignore_ascii_case("haar") {
        return Err(EngineError::bad_request(format!(
            "ACE-Step dcw_wavelet '{wavelet}' is not supported; only haar is built"
        )));
    }
    super::DcwSettings {
        enabled,
        mode: super::DcwMode::parse(mode)?,
        scaler,
        high_scaler,
    }
    .validate()
    .map_err(EngineError::from)
}

fn drawn_seed() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or(1)
        & 0x7fff_ffff
}

fn create_request<'a>(
    request: &'a AudioCreateRequest,
    options: &'a CreateOptions,
    expanded: Option<&'a super::LmExpandOutput>,
    loras: &'a [crate::weights::LoraFile],
    dcw: super::DcwSettings,
    index: usize,
) -> CreateRequest<'a> {
    let duration_sec = expanded
        .and_then(|value| value.duration)
        .map(|value| value as f32)
        .filter(|value| *value > 0.0)
        .unwrap_or(request.duration_sec);
    CreateRequest {
        prompt: expanded
            .map(|value| value.caption.as_str())
            .unwrap_or(&request.prompt),
        lyrics: expanded
            .map(|value| value.lyrics.as_str())
            .unwrap_or(&request.lyrics),
        duration_sec,
        seed: request.seed.map(|seed| seed + index as i64),
        guidance: request.guidance,
        steps: request.steps,
        vocal_language: expanded
            .map(|value| value.language.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(&request.vocal_language),
        bpm: expanded.and_then(|value| value.bpm).or(options.bpm),
        keyscale: expanded
            .map(|value| value.keyscale.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(&options.keyscale),
        timesignature: expanded
            .map(|value| value.timesignature.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(&options.timesignature),
        audio_code_indices: expanded
            .filter(|value| !value.audio_code_indices.is_empty())
            .map(|value| value.audio_code_indices.as_slice())
            .or(options.audio_code_indices.as_deref()),
        timbre_latents: options.timbre_latents.as_deref(),
        loras,
        dcw,
        lm_expanded: expanded.is_some(),
        pmi_bonus: expanded
            .and_then(|value| value.pmi.as_ref())
            .map(|pmi| pmi.quality_bonus())
            .unwrap_or(0.0),
    }
}

fn validate_batch(n: u32) -> Result<usize, EngineError> {
    if !(1..=8).contains(&n) {
        return Err(EngineError::bad_request(format!(
            "ACE-Step batch size {n} must be in 1..=8"
        )));
    }
    Ok(n as usize)
}

fn decorate_create_output(
    output: &mut EngineOutput,
    request: &AudioCreateRequest,
    options: &CreateOptions,
    index: usize,
) {
    let seed = request.seed.map(|value| value + index as i64);
    for artifact in &mut output.artifacts {
        decorate_audio_artifact(
            artifact,
            &request.title,
            &request.prompt,
            &request.lyrics,
            seed,
            request.guidance,
            request.steps,
            &request.adapters,
            &request.metadata,
            "create",
            None,
            options,
        );
    }
    sync_primary_audio_metadata(output);
}

fn decorate_edit_output(
    output: &mut EngineOutput,
    request: &AudioEditRequest,
    options: &EditOptions,
    index: usize,
) {
    let seed = request.seed.map(|value| value + index as i64);
    let relation = if options.task == "cover-nofsq" {
        "cover"
    } else {
        options.task.as_str()
    };
    for artifact in &mut output.artifacts {
        decorate_audio_artifact(
            artifact,
            &request.title,
            &request.prompt,
            &request.lyrics,
            seed,
            request.guidance,
            request.steps,
            &request.adapters,
            &request.metadata,
            relation,
            request.source_audio.as_deref(),
            options,
        );
        artifact.metadata["operation"] = json!(options.task);
        artifact.metadata["track"] = json!(options.track);
        artifact.metadata["audio_cover_strength"] = json!(options.audio_cover_strength);
        artifact.metadata["cover_noise_strength"] = json!(options.cover_noise_strength);
        artifact.metadata["timbre_from_source"] = json!(options.timbre_from_source);
        if let Some(start) = options.region_start_sec {
            artifact.metadata["region_start_sec"] = json!(start);
        }
        if let Some(end) = options.region_end_sec {
            artifact.metadata["region_end_sec"] = json!(end);
        }
        artifact.metadata["repaint_injection_ratio"] = json!(options.repaint_injection_ratio);
        artifact.metadata["repaint_crossfade_frames"] = json!(options.repaint_crossfade_frames);
    }
    sync_primary_audio_metadata(output);
}

trait AudioArtifactOptions {
    fn audio_format(&self) -> &str;
    fn denoise(&self) -> &str;
    fn dcw_enabled(&self) -> Option<bool>;
    fn dcw_mode(&self) -> &str;
    fn dcw_scaler(&self) -> f32;
    fn dcw_high_scaler(&self) -> f32;
}

impl AudioArtifactOptions for CreateOptions {
    fn audio_format(&self) -> &str {
        &self.audio_format
    }
    fn denoise(&self) -> &str {
        &self.denoise
    }
    fn dcw_enabled(&self) -> Option<bool> {
        self.dcw_enabled
    }
    fn dcw_mode(&self) -> &str {
        &self.dcw_mode
    }
    fn dcw_scaler(&self) -> f32 {
        self.dcw_scaler
    }
    fn dcw_high_scaler(&self) -> f32 {
        self.dcw_high_scaler
    }
}

impl AudioArtifactOptions for EditOptions {
    fn audio_format(&self) -> &str {
        &self.audio_format
    }
    fn denoise(&self) -> &str {
        &self.denoise
    }
    fn dcw_enabled(&self) -> Option<bool> {
        self.dcw_enabled
    }
    fn dcw_mode(&self) -> &str {
        &self.dcw_mode
    }
    fn dcw_scaler(&self) -> f32 {
        self.dcw_scaler
    }
    fn dcw_high_scaler(&self) -> f32 {
        self.dcw_high_scaler
    }
}

#[allow(clippy::too_many_arguments)]
fn decorate_audio_artifact(
    artifact: &mut Artifact,
    title: &str,
    prompt: &str,
    lyrics: &str,
    seed: Option<i64>,
    guidance: f32,
    steps: Option<u32>,
    adapters: &[danmo_contracts::ImageAdapter],
    request_metadata: &Value,
    operation: &str,
    source_audio: Option<&str>,
    options: &dyn AudioArtifactOptions,
) {
    let metadata = &mut artifact.metadata;
    metadata["title"] = json!(title);
    metadata["prompt"] = json!(prompt);
    metadata["lyrics_input"] = json!(lyrics);
    metadata["operation"] = json!(operation);
    metadata["relation_type"] = json!(operation);
    metadata["seed"] = json!(seed);
    metadata["guidance"] = json!(guidance);
    metadata["steps"] = json!(steps);
    metadata["adapters"] = json!(adapters
        .iter()
        .map(|adapter| json!({"id": adapter.id, "weight": adapter.weight}))
        .collect::<Vec<_>>());
    metadata["audio_format"] = json!(options.audio_format());
    metadata["denoise"] = json!(options.denoise());
    metadata["dcw_enabled"] = json!(options.dcw_enabled());
    metadata["dcw_mode"] = json!(options.dcw_mode());
    metadata["dcw_scaler"] = json!(options.dcw_scaler());
    metadata["dcw_high_scaler"] = json!(options.dcw_high_scaler());
    metadata["request_metadata"] = request_metadata.clone();
    if let Some(value) = request_metadata.get("canvas_session_id") {
        metadata["canvas_session_id"] = value.clone();
    }
    let parent = request_metadata
        .get("parent_asset_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            source_audio
                .and_then(|value| value.strip_prefix("asset:"))
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        });
    let relation = request_metadata
        .get("relation_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(operation)
        .to_string();
    if let Some(parent) = parent {
        artifact.parent_asset_id = Some(parent.clone());
        metadata["parent_asset_id"] = json!(parent);
    }
    artifact.relation_type = Some(relation.clone());
    metadata["relation_type"] = json!(relation);
}

fn sync_primary_audio_metadata(output: &mut EngineOutput) {
    let Some(primary) = output.artifacts.first() else {
        return;
    };
    for key in [
        "title",
        "prompt",
        "lyrics_input",
        "operation",
        "relation_type",
        "seed",
        "guidance",
        "steps",
        "adapters",
        "audio_format",
        "denoise",
    ] {
        output.metadata[key] = primary.metadata[key].clone();
    }
}

fn merge_audio_outputs(outputs: Vec<EngineOutput>, request_metadata: &Value) -> EngineOutput {
    let mut merged = EngineOutput::default();
    let mut metadata = Value::Null;
    let mut items = Vec::with_capacity(outputs.len());
    for output in outputs {
        if metadata.is_null() {
            metadata = output.metadata.clone();
        }
        items.push(output.metadata);
        merged.steps = merged.steps.max(output.steps);
        merged.graph_computes += output.graph_computes;
        merged.latent_len += output.latent_len;
        merged.sample_rate = merged.sample_rate.or(output.sample_rate);
        merged.frames = merged.frames.or(output.frames);
        merged.width = merged.width.or(output.width);
        merged.height = merged.height.or(output.height);
        merged.timings.extend(output.timings);
        merged.artifacts.extend(output.artifacts);
    }
    metadata["batch_count"] = json!(items.len());
    metadata["items"] = json!(items);
    metadata["request_metadata"] = request_metadata.clone();
    merged.metadata = metadata;
    merged
}

fn audio_output(
    out: CreateOutput,
    encoding: &super::audio_codec::AudioEncoding,
) -> Result<EngineOutput, EngineError> {
    let alignment = serde_json::to_value(out.lyrics_alignment.as_metadata())
        .map_err(|error| EngineError::bad_request(error.to_string()))?;
    let mut metadata = json!({
        "lyrics_effective": out.lyrics_effective,
        "lyrics_alignment": alignment.get("lyrics_alignment").cloned().unwrap_or(Value::Null),
        "lyrics_segments": alignment.get("lyrics_segments").cloned().unwrap_or_else(|| json!([])),
        "enc_seq": out.enc_seq,
        "dcw_enabled": out.dcw_active,
        "dcw_mode": out.dcw.mode.as_str(),
        "dcw_wavelet": "haar",
        "dcw_scaler": out.dcw.scaler,
        "dcw_high_scaler": out.dcw.high_scaler,
        "format": encoding.format,
        "denoise": encoding.denoise,
    });
    if let (Some(target), Some(quality)) = (
        metadata.as_object_mut(),
        out.quality.as_metadata().as_object(),
    ) {
        for (key, value) in quality {
            target.insert(key.clone(), value.clone());
        }
    }
    let encoded = super::audio_codec::encode_wav(&out.wav, out.sample_rate, encoding)?;
    let mut artifact = Artifact::wav(encoded.bytes, out.sample_rate, out.frames, metadata.clone());
    artifact.extension = encoded.extension;
    artifact.mime = encoded.mime;
    if let Some(txt) = out.lyrics_txt {
        artifact.sidecars.push(Sidecar {
            suffix: "_lyrics.txt".into(),
            mime: "text/plain".into(),
            bytes: txt.into_bytes(),
        });
    }
    if let Some(lrc) = out.lyrics_lrc {
        artifact.sidecars.push(Sidecar {
            suffix: ".lrc".into(),
            mime: "text/plain".into(),
            bytes: lrc.into_bytes(),
        });
    }
    artifact.metadata["lyrics_sidecar"] = json!(artifact
        .sidecars
        .iter()
        .any(|side| side.suffix == "_lyrics.txt"));
    artifact.metadata["lrc_sidecar"] =
        json!(artifact.sidecars.iter().any(|side| side.suffix == ".lrc"));
    Ok(EngineOutput {
        steps: out.steps,
        graph_computes: out.graph_computes,
        sample_rate: Some(out.sample_rate),
        frames: Some(out.frames),
        metadata,
        timings: out
            .timings
            .into_iter()
            .map(|(phase, seconds)| (phase.into(), seconds))
            .collect(),
        artifacts: vec![artifact],
        ..EngineOutput::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_limits_and_per_item_metadata_are_preserved() {
        assert_eq!(validate_batch(1).unwrap(), 1);
        assert_eq!(validate_batch(8).unwrap(), 8);
        assert!(validate_batch(0).is_err());
        assert!(validate_batch(9).is_err());
        let outputs = [72.0, 33.0].map(|score| EngineOutput {
            graph_computes: 2,
            metadata: json!({"quality_score": score}),
            artifacts: vec![Artifact::wav(
                vec![0, 1],
                48_000,
                128,
                json!({"quality_score": score}),
            )],
            ..EngineOutput::default()
        });
        let merged = merge_audio_outputs(outputs.into_iter().collect(), &json!({"source": "test"}));
        assert_eq!(merged.graph_computes, 4);
        assert_eq!(merged.artifacts.len(), 2);
        assert_eq!(merged.metadata["batch_count"], 2);
        assert_eq!(merged.metadata["items"][0]["quality_score"], 72.0);
        assert_eq!(merged.metadata["items"][1]["quality_score"], 33.0);
        assert_eq!(merged.metadata["request_metadata"]["source"], "test");
    }

    #[test]
    fn artifact_metadata_preserves_product_fields_and_edit_lineage() {
        let mut artifact = Artifact::wav(vec![0, 1], 48_000, 250, json!({}));
        decorate_audio_artifact(
            &mut artifact,
            "Remix",
            "jazz",
            "hello",
            Some(42),
            7.0,
            Some(30),
            &[],
            &json!({"canvas_session_id": "cvs_1"}),
            "repaint",
            Some("asset:ast_source"),
            &EditOptions::default(),
        );
        assert_eq!(artifact.metadata["title"], "Remix");
        assert_eq!(artifact.metadata["prompt"], "jazz");
        assert_eq!(artifact.metadata["seed"], 42);
        assert_eq!(artifact.metadata["operation"], "repaint");
        assert_eq!(artifact.metadata["canvas_session_id"], "cvs_1");
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_source"));
        assert_eq!(artifact.relation_type.as_deref(), Some("repaint"));
    }

    #[test]
    fn vocal_preparation_detects_language_and_appends_type() {
        let exec =
            ExecutionContext::new("vocal", std::sync::Arc::new(crate::contract::NoopObserver));
        let (prompt, lyrics, language) =
            vocal_fields("流行", "你好世界", "", "female", false, true, &exec).unwrap();
        assert_eq!(lyrics, "你好世界");
        assert_eq!(language, "zh");
        assert!(prompt.contains("清晰女声主唱"));
        let error = vocal_fields("lofi", "", "", "", false, true, &exec);
        assert!(error.is_err());
        let (_, instrumental, _) = vocal_fields("lofi", "", "", "", false, false, &exec).unwrap();
        assert_eq!(instrumental, "[Instrumental]");
    }
}
