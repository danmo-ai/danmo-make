//! Real-ESRGAN x4+ image upscale through the shared facade pipeline.

#[cfg(feature = "mlx")]
mod mlx;

#[cfg(feature = "mlx")]
use crate::contract::Artifact;
use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use danmo_contracts::ImageUpscaleRequest;
#[cfg(feature = "mlx")]
use serde_json::json;

const PRE_PAD: usize = 10;
const TILE_PAD: usize = 10;
const NET_SCALE: usize = 4;

pub struct EsrganUpscaleFamily;

#[derive(Clone)]
struct Prepared {
    request: ImageUpscaleRequest,
    rgb: Vec<u8>,
    source: (u32, u32),
    scale: u32,
    tile_size: usize,
}

pub struct EsrganUpscaleSession {
    run: ResolvedRun,
    prepared: Prepared,
}

impl FacadeFamily for EsrganUpscaleFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::ImageUpscale(request) = request else {
            return Err(EngineError::bad_request(
                "esrgan upscale received a non-upscale request",
            ));
        };
        let scale = request.scale.unwrap_or(4);
        if scale != 2 && scale != 4 {
            return Err(EngineError::bad_request(format!(
                "esrgan upscale scale must be 2 or 4, got {scale}"
            )));
        }
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "esrgan upscale source_asset_id is empty",
            ));
        }
        if let Some(denoise) = request.denoise {
            if !denoise.is_finite() {
                return Err(EngineError::bad_request(format!(
                    "esrgan upscale denoise {denoise} is not finite"
                )));
            }
            if denoise != 0.0 {
                exec.log(
                    "info",
                    "esrgan_upscale denoise is ignored by deterministic RRDBNet x4+",
                )?;
            }
        }
        let tile_size = request.tile_size.unwrap_or(256) as usize;
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, resolver, tile_size);
            return Err(crate::weights::WeightError::NotBuilt(
                "Real-ESRGAN x4+ requires an MLX build".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            let (rgb, source_w, source_h) =
                resolver.read_source_image(run, &request.source_asset_id)?;
            if source_w as usize <= PRE_PAD || source_h as usize <= PRE_PAD {
                return Err(EngineError::bad_request(format!(
                    "esrgan upscale source {source_w}x{source_h} is too small for the official {PRE_PAD}px reflect padding"
                )));
            }
            source_w
                .checked_mul(4)
                .and_then(|width| {
                    source_h
                        .checked_mul(4)
                        .and_then(|height| width.checked_mul(height))
                })
                .ok_or_else(|| EngineError::bad_request("esrgan output dimensions overflow u32"))?;
            exec.annotate(&[
                ("danmo.generation.source_width", source_w.to_string()),
                ("danmo.generation.source_height", source_h.to_string()),
                ("danmo.generation.width", (source_w * scale).to_string()),
                ("danmo.generation.height", (source_h * scale).to_string()),
                ("danmo.generation.scale", scale.to_string()),
                ("danmo.generation.tile_size", tile_size.to_string()),
                ("danmo.upscale.network_scale", NET_SCALE.to_string()),
                ("danmo.upscale.pre_pad", PRE_PAD.to_string()),
                ("danmo.upscale.tile_pad", TILE_PAD.to_string()),
            ])?;
            exec.log(
                "info",
                &format!(
                    "esrgan_upscale backend=mlx source={source_w}x{source_h} out={}x{} scale={scale} tile_size={tile_size}",
                    source_w * scale,
                    source_h * scale
                ),
            )?;
            Ok(Box::new(EsrganUpscaleSession {
                run: run.clone(),
                prepared: Prepared {
                    request: request.clone(),
                    rgb,
                    source: (source_w, source_h),
                    scale,
                    tile_size,
                },
            }))
        }
    }
}

impl FacadeSession for EsrganUpscaleSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(feature = "mlx")]
        {
            let run = self.run.clone();
            let prepared = self.prepared.clone();
            let exec = exec.clone();
            return crate::runtime::mlx_executor()
                .map_err(mlx_error)?
                .execute(
                    "Real-ESRGAN x4+ upscale",
                    move |_, stream, _cache, _events| {
                        Ok(infer_mlx_on_stream(&run, &prepared, &exec, stream))
                    },
                )
                .map_err(mlx_error)?;
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            Err(crate::weights::WeightError::NotBuilt(
                "Real-ESRGAN x4+ requires an MLX build".into(),
            )
            .into())
        }
    }
}

#[cfg(feature = "mlx")]
fn infer_mlx_on_stream(
    run: &ResolvedRun,
    prepared: &Prepared,
    exec: &ExecutionContext,
    stream: &danmo_mlx::Stream,
) -> Result<EngineOutput, EngineError> {
    exec.check_cancelled()?;
    exec.phase("load")?;
    let weight_file = "model.safetensors";
    let weight_path = run.bundle.join(weight_file);
    let weight_bytes = std::fs::metadata(&weight_path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    exec.annotate(&[
        ("danmo.weight.files", weight_file.into()),
        ("danmo.weight.bytes", weight_bytes.to_string()),
        ("danmo.weight.dtype", "native".into()),
    ])?;
    let model = mlx::RrdbNet::load(&run.bundle)?;
    exec.log(
        "info",
        &format!("esrgan RRDBNet weights loaded file={weight_file} bytes={weight_bytes}"),
    )?;
    let (source_w, source_h) = (prepared.source.0 as usize, prepared.source.1 as usize);
    let expected = source_w
        .checked_mul(source_h)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| EngineError::bad_request("esrgan source dimensions overflow usize"))?;
    if prepared.rgb.len() != expected {
        return Err(EngineError::bad_request(format!(
            "esrgan source RGB has {} bytes, expected {expected}",
            prepared.rgb.len()
        )));
    }
    exec.phase("preprocess")?;
    let normalized: Vec<f32> = prepared
        .rgb
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect();
    let (padded, padded_w, padded_h) =
        reflect_pad_right_bottom(&normalized, source_w, source_h, PRE_PAD)?;
    let tiles = if prepared.tile_size == 0 {
        1
    } else {
        padded_w.div_ceil(prepared.tile_size) * padded_h.div_ceil(prepared.tile_size)
    };
    exec.annotate(&[("danmo.upscale.tiles", tiles.to_string())])?;
    exec.log(
        "info",
        &format!("esrgan_upscale tiles={tiles} padded={padded_w}x{padded_h}"),
    )?;
    exec.phase("upscale")?;
    let upscaled = upscale_tiles(
        &model,
        &padded,
        padded_w,
        padded_h,
        prepared.tile_size,
        exec,
        stream,
    )?;
    exec.phase("postprocess")?;
    let native_w = source_w * NET_SCALE;
    let native_h = source_h * NET_SCALE;
    let native = crop_nhwc(&upscaled, padded_w * NET_SCALE, 0, 0, native_w, native_h)?;
    let mut native_rgb = vec![0u8; native_w * native_h * 3];
    for (output, value) in native_rgb.iter_mut().zip(native) {
        *output = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    let out_w = source_w * prepared.scale as usize;
    let out_h = source_h * prepared.scale as usize;
    let rgb = if prepared.scale as usize == NET_SCALE {
        native_rgb
    } else {
        crate::common::image::resize_lanczos(
            &native_rgb,
            native_w as u32,
            native_h as u32,
            out_w as u32,
            out_h as u32,
        )?
    };
    exec.log(
        "info",
        &format!("esrgan_upscale finished out={out_w}x{out_h}"),
    )?;
    let metadata = json!({
        "model": prepared.request.model,
        "source_asset_id": prepared.request.source_asset_id,
        "source_width": prepared.source.0,
        "source_height": prepared.source.1,
        "scale": prepared.scale,
        "network_scale": NET_SCALE,
        "tile_size": prepared.tile_size,
        "tile_pad": TILE_PAD,
        "pre_pad": PRE_PAD,
        "tiles": tiles,
        "backend": "mlx",
    });
    let png = crate::common::png::encode_rgb(out_w as u32, out_h as u32, &rgb);
    let mut artifact = Artifact::png(png, out_w as i64, out_h as i64, metadata.clone());
    artifact.source_action = "upscale".into();
    artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
    artifact.relation_type = Some("upscale".into());
    let mut full = metadata;
    full["request_metadata"] = prepared.request.metadata.clone();
    Ok(EngineOutput {
        steps: 1,
        width: Some(out_w as u32),
        height: Some(out_h as u32),
        metadata: full,
        artifacts: vec![artifact],
        ..EngineOutput::default()
    })
}

#[cfg(feature = "mlx")]
fn upscale_tiles(
    model: &mlx::RrdbNet,
    padded: &[f32],
    width: usize,
    height: usize,
    tile_size: usize,
    exec: &ExecutionContext,
    stream: &danmo_mlx::Stream,
) -> Result<Vec<f32>, EngineError> {
    use danmo_mlx::Array;

    if tile_size == 0 {
        let input = Array::from_f32_slice(padded, &[1, height as i32, width as i32, 3])
            .map_err(mlx_error)?;
        let output = model.forward(&input, stream).map_err(mlx_error)?;
        let values = output.to_vec_f32(stream).map_err(mlx_error)?;
        ensure_finite(&values)?;
        exec.progress("upscale", 1, 1)?;
        return Ok(values);
    }
    let out_w = width * NET_SCALE;
    let out_h = height * NET_SCALE;
    let mut output = vec![0.0f32; out_w * out_h * 3];
    let tiles_x = width.div_ceil(tile_size);
    let tiles_y = height.div_ceil(tile_size);
    let total = tiles_x * tiles_y;
    let mut completed = 0usize;
    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            exec.check_cancelled()?;
            let start_x = tile_x * tile_size;
            let start_y = tile_y * tile_size;
            let end_x = (start_x + tile_size).min(width);
            let end_y = (start_y + tile_size).min(height);
            let padded_start_x = start_x.saturating_sub(TILE_PAD);
            let padded_start_y = start_y.saturating_sub(TILE_PAD);
            let padded_end_x = (end_x + TILE_PAD).min(width);
            let padded_end_y = (end_y + TILE_PAD).min(height);
            let tile_w = padded_end_x - padded_start_x;
            let tile_h = padded_end_y - padded_start_y;
            let tile = crop_nhwc(
                padded,
                width,
                padded_start_x,
                padded_start_y,
                tile_w,
                tile_h,
            )?;
            let input = Array::from_f32_slice(&tile, &[1, tile_h as i32, tile_w as i32, 3])
                .map_err(mlx_error)?;
            let tile_output = model.forward(&input, stream).map_err(mlx_error)?;
            let tile_shape = tile_output.shape();
            let expected = [
                1,
                (tile_h * NET_SCALE) as i32,
                (tile_w * NET_SCALE) as i32,
                3,
            ];
            if tile_shape != expected {
                return Err(EngineError::message(format!(
                    "Real-ESRGAN tile output {tile_shape:?}, expected {expected:?}"
                )));
            }
            let tile_output = tile_output.to_vec_f32(stream).map_err(mlx_error)?;
            ensure_finite(&tile_output)?;
            let crop_x = (start_x - padded_start_x) * NET_SCALE;
            let crop_y = (start_y - padded_start_y) * NET_SCALE;
            let core_w = (end_x - start_x) * NET_SCALE;
            let core_h = (end_y - start_y) * NET_SCALE;
            let core = crop_nhwc(
                &tile_output,
                tile_w * NET_SCALE,
                crop_x,
                crop_y,
                core_w,
                core_h,
            )?;
            for row in 0..core_h {
                let source = row * core_w * 3;
                let destination = ((start_y * NET_SCALE + row) * out_w + start_x * NET_SCALE) * 3;
                output[destination..destination + core_w * 3]
                    .copy_from_slice(&core[source..source + core_w * 3]);
            }
            completed += 1;
            exec.progress("upscale", completed as u32, total as u32)?;
        }
    }
    Ok(output)
}

#[cfg(feature = "mlx")]
fn reflect_pad_right_bottom(
    input: &[f32],
    width: usize,
    height: usize,
    pad: usize,
) -> Result<(Vec<f32>, usize, usize), EngineError> {
    if width <= pad || height <= pad {
        return Err(EngineError::bad_request(format!(
            "reflect padding {pad} requires both source edges to exceed {pad}"
        )));
    }
    let out_w = width + pad;
    let out_h = height + pad;
    let mut output = vec![0.0f32; out_w * out_h * 3];
    for y in 0..out_h {
        let source_y = if y < height { y } else { 2 * height - y - 2 };
        for x in 0..out_w {
            let source_x = if x < width { x } else { 2 * width - x - 2 };
            for channel in 0..3 {
                output[(y * out_w + x) * 3 + channel] =
                    input[(source_y * width + source_x) * 3 + channel];
            }
        }
    }
    Ok((output, out_w, out_h))
}

#[cfg(feature = "mlx")]
fn crop_nhwc(
    input: &[f32],
    input_w: usize,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) -> Result<Vec<f32>, EngineError> {
    let input_h = input.len() / (input_w * 3);
    if start_x + width > input_w || start_y + height > input_h {
        return Err(EngineError::message(format!(
            "NHWC crop ({start_x},{start_y}) {width}x{height} exceeds {input_w}x{input_h}"
        )));
    }
    let mut output = vec![0.0f32; width * height * 3];
    for row in 0..height {
        let source = ((start_y + row) * input_w + start_x) * 3;
        let destination = row * width * 3;
        output[destination..destination + width * 3]
            .copy_from_slice(&input[source..source + width * 3]);
    }
    Ok(output)
}

#[cfg(feature = "mlx")]
fn ensure_finite(values: &[f32]) -> Result<(), EngineError> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EngineError::message(format!(
            "Real-ESRGAN output is non-finite at index {index}: {value}"
        )));
    }
    Ok(())
}

#[cfg(feature = "mlx")]
fn mlx_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::message(error.to_string())
}

#[cfg(all(test, feature = "mlx"))]
mod tests {
    use super::*;
    use crate::contract::{Action, NoopObserver};
    use crate::runtime::BackendKind;
    use danmo_mlx::{save_safetensors, Array};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "danmo-esrgan-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct Resolver;

    impl ResolvePort for Resolver {
        fn resolve(&self, _model: &str, _action: Action) -> Result<ResolvedRun, EngineError> {
            unreachable!()
        }

        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<PathBuf, EngineError> {
            unreachable!()
        }

        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, EngineError> {
            unreachable!()
        }

        fn read_source_image(
            &self,
            _run: &ResolvedRun,
            source: &str,
        ) -> Result<(Vec<u8>, u32, u32), EngineError> {
            assert_eq!(source, "ast_src");
            let rgb = (0..16 * 12 * 3)
                .map(|index| ((index * 29) % 251) as u8)
                .collect();
            Ok((rgb, 16, 12))
        }
    }

    fn add_conv(tensors: &mut BTreeMap<String, Array>, name: &str, inputs: i32, outputs: i32) {
        let weight: Vec<f32> = (0..outputs * 3 * 3 * inputs)
            .map(|index| (index % 23) as f32 * 0.001 - 0.011)
            .collect();
        let bias: Vec<f32> = (0..outputs).map(|index| index as f32 * 0.001).collect();
        tensors.insert(
            format!("{name}.weight"),
            Array::from_f32_slice(&weight, &[outputs, 3, 3, inputs]).unwrap(),
        );
        tensors.insert(
            format!("{name}.bias"),
            Array::from_f32_slice(&bias, &[outputs]).unwrap(),
        );
    }

    fn write_bundle(root: &Path) {
        let (feat, grow, blocks) = (2, 1, 1);
        let mut tensors = BTreeMap::new();
        add_conv(&mut tensors, "conv_first", 3, feat);
        for block in 0..blocks {
            for rdb in 1..=3 {
                for conv in 1..=5 {
                    let inputs = if conv == 5 {
                        feat + 4 * grow
                    } else {
                        feat + (conv - 1) * grow
                    };
                    let outputs = if conv == 5 { feat } else { grow };
                    add_conv(
                        &mut tensors,
                        &format!("body.{block}.rdb{rdb}.conv{conv}"),
                        inputs,
                        outputs,
                    );
                }
            }
        }
        for name in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
            add_conv(&mut tensors, name, feat, feat);
        }
        add_conv(&mut tensors, "conv_last", feat, 3);
        let tensors: Vec<(&str, &Array)> = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        save_safetensors(&root.join("model.safetensors"), &tensors, &[]).unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{"arch":"RRDBNet","netscale":4,"num_feat":2,"num_block":1,"num_grow_ch":1}"#,
        )
        .unwrap();
    }

    fn request(scale: u32, tile_size: u32) -> ImageUpscaleRequest {
        ImageUpscaleRequest {
            model: "real-esrgan-x4plus".into(),
            source_asset_id: "ast_src".into(),
            scale: Some(scale),
            tile_size: Some(tile_size),
            denoise: Some(0.3),
            seed: Some(7),
            metadata: json!({"origin": "unit"}),
            priority: "normal".into(),
        }
    }

    fn run(bundle: &Path) -> ResolvedRun {
        ResolvedRun {
            model: "real-esrgan-x4plus".into(),
            version: Some("fp16".into()),
            family: "esrgan".into(),
            action: Action::ImageUpscale,
            bundle: bundle.to_path_buf(),
            backend: BackendKind::Cpu,
        }
    }

    fn execute(bundle: &Path, scale: u32, tile_size: u32) -> EngineOutput {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("esrgan-test", std::sync::Arc::new(NoopObserver));
        let family = EsrganUpscaleFamily;
        let mut session = family
            .prepare(
                &ctx,
                &run(bundle),
                &EngineRequest::ImageUpscale(request(scale, tile_size)),
                &Resolver,
                &exec,
            )
            .unwrap();
        session.infer(&ctx, &exec).unwrap()
    }

    #[test]
    fn facade_runs_tiled_x4_and_untiled_x2_end_to_end() {
        let scratch = Scratch::new();
        write_bundle(&scratch.0);
        let x4 = execute(&scratch.0, 4, 8);
        assert_eq!((x4.width, x4.height), (Some(64), Some(48)));
        assert_eq!(x4.metadata["tiles"], json!(12));
        assert_eq!(x4.metadata["request_metadata"]["origin"], json!("unit"));
        assert_eq!(x4.artifacts.len(), 1);
        assert_eq!(x4.artifacts[0].parent_asset_id.as_deref(), Some("ast_src"));
        assert_eq!(x4.artifacts[0].relation_type.as_deref(), Some("upscale"));
        let (rgb, width, height) = crate::common::png::decode_rgb(&x4.artifacts[0].bytes).unwrap();
        assert_eq!((width, height), (64, 48));
        assert!(rgb.iter().any(|value| *value != rgb[0]));

        let x2 = execute(&scratch.0, 2, 0);
        assert_eq!((x2.width, x2.height), (Some(32), Some(24)));
        assert_eq!(x2.metadata["tiles"], json!(1));
        let (_, width, height) = crate::common::png::decode_rgb(&x2.artifacts[0].bytes).unwrap();
        assert_eq!((width, height), (32, 24));
    }

    #[test]
    fn invalid_scale_and_cancelled_task_fail_loud() {
        let scratch = Scratch::new();
        write_bundle(&scratch.0);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("esrgan-test", std::sync::Arc::new(NoopObserver));
        let error = EsrganUpscaleFamily
            .prepare(
                &ctx,
                &run(&scratch.0),
                &EngineRequest::ImageUpscale(request(3, 8)),
                &Resolver,
                &exec,
            )
            .err()
            .expect("scale 3 must fail");
        assert!(error.to_string().contains("scale"), "{error}");

        let cancellation = danmo_contracts::CancellationToken::default();
        cancellation.cancel();
        let exec = ExecutionContext::new("esrgan-cancel", std::sync::Arc::new(NoopObserver))
            .with_cancellation(cancellation);
        let mut session = EsrganUpscaleFamily
            .prepare(
                &ctx,
                &run(&scratch.0),
                &EngineRequest::ImageUpscale(request(4, 8)),
                &Resolver,
                &ExecutionContext::new("esrgan-prepare", std::sync::Arc::new(NoopObserver)),
            )
            .unwrap();
        assert!(matches!(
            session.infer(&ctx, &exec),
            Err(EngineError::Cancelled)
        ));
    }

    #[test]
    fn reflect_padding_matches_numpy_right_bottom_semantics() {
        let input: Vec<f32> = (0..18).map(|value| value as f32).collect();
        let (padded, width, height) = reflect_pad_right_bottom(&input, 3, 2, 1).unwrap();
        assert_eq!((width, height), (4, 3));
        for channel in 0..3 {
            assert_eq!(padded[(1 * 4 + 3) * 3 + channel], 12.0 + channel as f32);
            assert_eq!(padded[(2 * 4 + 1) * 3 + channel], 3.0 + channel as f32);
            assert_eq!(padded[(2 * 4 + 3) * 3 + channel], 3.0 + channel as f32);
        }
    }

    #[test]
    fn driver_routes_esrgan_upscale_to_the_facade() {
        crate::session::ensure_available("esrgan", Action::ImageUpscale).unwrap();
        assert!(crate::session::ensure_available("esrgan", Action::ImageCreate).is_err());
    }
}
