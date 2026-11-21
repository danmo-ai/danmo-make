//! Real-ESRGAN RRDBNet on MLX. Kernels use the published `[O, kH, kW, I]`
//! layout and activations are NHWC.

use std::path::Path;

use danmo_mlx::{Array, Dtype, Error, Result, Stream};
use serde::Deserialize;

use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

#[derive(Debug, Clone, Copy)]
pub struct RrdbSpec {
    pub num_feat: i32,
    pub num_block: usize,
    pub num_grow_ch: i32,
}

#[derive(Deserialize)]
struct ConfigFile {
    arch: String,
    netscale: u32,
    num_feat: i32,
    num_block: usize,
    num_grow_ch: i32,
}

pub struct RrdbNet {
    weights: MlxWeights,
    spec: RrdbSpec,
}

impl RrdbNet {
    pub fn load(bundle: &Path) -> std::result::Result<Self, WeightError> {
        let config_path = bundle.join("config.json");
        let config = std::fs::read(&config_path).map_err(|error| WeightError::Bundle {
            path: config_path.display().to_string(),
            detail: format!("read Real-ESRGAN config: {error}"),
        })?;
        let config: ConfigFile =
            serde_json::from_slice(&config).map_err(|error| WeightError::Bundle {
                path: config_path.display().to_string(),
                detail: format!("parse Real-ESRGAN config: {error}"),
            })?;
        if config.arch != "RRDBNet" || config.netscale != 4 {
            return Err(WeightError::NotBuilt(format!(
                "Real-ESRGAN x4+ requires RRDBNet netscale=4, got {} netscale={}",
                config.arch, config.netscale
            )));
        }
        let spec = RrdbSpec {
            num_feat: config.num_feat,
            num_block: config.num_block,
            num_grow_ch: config.num_grow_ch,
        };
        let weights = load_mlx_file(&bundle.join("model.safetensors"), WeightDtype::Native, None)
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        Self::from_weights(weights, spec).map_err(|error| WeightError::NotBuilt(error.to_string()))
    }

    fn from_weights(weights: MlxWeights, spec: RrdbSpec) -> Result<Self> {
        if spec.num_feat < 1 || spec.num_block < 1 || spec.num_grow_ch < 1 {
            return Err(Error::new(
                "Real-ESRGAN RRDBNet",
                format!("invalid architecture {spec:?}"),
            ));
        }
        validate_conv(&weights, "conv_first", 3, spec.num_feat)?;
        for block in 0..spec.num_block {
            for rdb in 1..=3 {
                for conv in 1..=5 {
                    let inputs = if conv == 5 {
                        spec.num_feat + 4 * spec.num_grow_ch
                    } else {
                        spec.num_feat + (conv - 1) * spec.num_grow_ch
                    };
                    let outputs = if conv == 5 {
                        spec.num_feat
                    } else {
                        spec.num_grow_ch
                    };
                    validate_conv(
                        &weights,
                        &format!("body.{block}.rdb{rdb}.conv{conv}"),
                        inputs,
                        outputs,
                    )?;
                }
            }
        }
        for name in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
            validate_conv(&weights, name, spec.num_feat, spec.num_feat)?;
        }
        validate_conv(&weights, "conv_last", spec.num_feat, 3)?;
        Ok(Self { weights, spec })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[3] != 3 {
            return Err(Error::new(
                "Real-ESRGAN RRDBNet",
                format!("input must be [1,H,W,3], got {shape:?}"),
            ));
        }
        let dtype = self.weights.tensor("conv_first.weight")?.dtype();
        let input = input.astype(dtype, stream)?;
        let first = self.conv("conv_first", &input, stream)?;
        let mut body = first.clone();
        for block in 0..self.spec.num_block {
            body = self.rrdb(block, &body, stream)?;
            body.eval()?;
        }
        let body = self.conv("conv_body", &body, stream)?;
        let mut feat = Array::add(&first, &body, stream)?;
        feat = nearest_x2(&feat, stream)?;
        feat = lrelu(&self.conv("conv_up1", &feat, stream)?, stream)?;
        feat = nearest_x2(&feat, stream)?;
        feat = lrelu(&self.conv("conv_up2", &feat, stream)?, stream)?;
        feat = lrelu(&self.conv("conv_hr", &feat, stream)?, stream)?;
        self.conv("conv_last", &feat, stream)
    }

    fn rrdb(&self, block: usize, input: &Array, stream: &Stream) -> Result<Array> {
        let mut out = self.rdb(&format!("body.{block}.rdb1"), input, stream)?;
        out = self.rdb(&format!("body.{block}.rdb2"), &out, stream)?;
        out = self.rdb(&format!("body.{block}.rdb3"), &out, stream)?;
        residual(input, &out, stream)
    }

    fn rdb(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let x1 = lrelu(
            &self.conv(&format!("{prefix}.conv1"), input, stream)?,
            stream,
        )?;
        let joined = Array::concatenate_axis(&[input.clone(), x1.clone()], -1, stream)?;
        let x2 = lrelu(
            &self.conv(&format!("{prefix}.conv2"), &joined, stream)?,
            stream,
        )?;
        let joined = Array::concatenate_axis(&[input.clone(), x1.clone(), x2.clone()], -1, stream)?;
        let x3 = lrelu(
            &self.conv(&format!("{prefix}.conv3"), &joined, stream)?,
            stream,
        )?;
        let joined = Array::concatenate_axis(
            &[input.clone(), x1.clone(), x2.clone(), x3.clone()],
            -1,
            stream,
        )?;
        let x4 = lrelu(
            &self.conv(&format!("{prefix}.conv4"), &joined, stream)?,
            stream,
        )?;
        let joined = Array::concatenate_axis(&[input.clone(), x1, x2, x3, x4], -1, stream)?;
        let x5 = self.conv(&format!("{prefix}.conv5"), &joined, stream)?;
        residual(input, &x5, stream)
    }

    fn conv(&self, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let output = Array::conv2d(
            input,
            self.weights.tensor(&format!("{name}.weight"))?,
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            stream,
        )?;
        Array::add(
            &output,
            self.weights.tensor(&format!("{name}.bias"))?,
            stream,
        )
    }
}

fn validate_conv(weights: &MlxWeights, name: &str, inputs: i32, outputs: i32) -> Result<()> {
    let weight = weights.tensor(&format!("{name}.weight"))?;
    let expected = [outputs, 3, 3, inputs];
    if weight.shape() != expected {
        return Err(Error::new(
            "Real-ESRGAN RRDBNet",
            format!(
                "{name}.weight shape {:?}, expected {expected:?} ([O,kH,kW,I])",
                weight.shape()
            ),
        ));
    }
    if !weight.dtype().is_floating() {
        return Err(Error::new(
            "Real-ESRGAN RRDBNet",
            format!("{name}.weight dtype {:?} is not floating", weight.dtype()),
        ));
    }
    let bias = weights.tensor(&format!("{name}.bias"))?;
    if bias.shape() != [outputs] || bias.dtype() != weight.dtype() {
        return Err(Error::new(
            "Real-ESRGAN RRDBNet",
            format!(
                "{name}.bias shape/dtype {:?}/{:?}, expected [{outputs}]/{:?}",
                bias.shape(),
                bias.dtype(),
                weight.dtype()
            ),
        ));
    }
    Ok(())
}

fn residual(input: &Array, residual: &Array, stream: &Stream) -> Result<Array> {
    let scale = scalar_like(0.2, residual.dtype(), stream)?;
    Array::add(input, &Array::multiply(residual, &scale, stream)?, stream)
}

fn lrelu(input: &Array, stream: &Stream) -> Result<Array> {
    let slope = scalar_like(0.2, input.dtype(), stream)?;
    Array::maximum(input, &Array::multiply(input, &slope, stream)?, stream)
}

fn scalar_like(value: f32, dtype: Dtype, stream: &Stream) -> Result<Array> {
    Array::scalar_f32(value)?.astype(dtype, stream)
}

fn nearest_x2(input: &Array, stream: &Stream) -> Result<Array> {
    input.repeat_axis(2, 1, stream)?.repeat_axis(2, 2, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Clone)]
    struct HostImage {
        w: usize,
        h: usize,
        c: usize,
        data: Vec<f32>,
    }

    fn add_conv(
        tensors: &mut BTreeMap<String, (Vec<i32>, Vec<f32>)>,
        name: &str,
        inputs: i32,
        outputs: i32,
    ) {
        for (suffix, shape) in [
            ("weight", vec![outputs, 3, 3, inputs]),
            ("bias", vec![outputs]),
        ] {
            let full_name = format!("{name}.{suffix}");
            let count = shape.iter().product::<i32>() as usize;
            let seed = full_name
                .bytes()
                .fold(0usize, |sum, byte| sum + byte as usize);
            let values = (0..count)
                .map(|index| ((index * 17 + seed) % 101) as f32 * 0.0007 - 0.035)
                .collect();
            tensors.insert(full_name, (shape, values));
        }
    }

    fn synthetic_weights(spec: RrdbSpec) -> BTreeMap<String, (Vec<i32>, Vec<f32>)> {
        let mut tensors = BTreeMap::new();
        add_conv(&mut tensors, "conv_first", 3, spec.num_feat);
        for block in 0..spec.num_block {
            for rdb in 1..=3 {
                for conv in 1..=5 {
                    let inputs = if conv == 5 {
                        spec.num_feat + 4 * spec.num_grow_ch
                    } else {
                        spec.num_feat + (conv - 1) * spec.num_grow_ch
                    };
                    let outputs = if conv == 5 {
                        spec.num_feat
                    } else {
                        spec.num_grow_ch
                    };
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
            add_conv(&mut tensors, name, spec.num_feat, spec.num_feat);
        }
        add_conv(&mut tensors, "conv_last", spec.num_feat, 3);
        tensors
    }

    fn host_conv(
        tensors: &BTreeMap<String, (Vec<i32>, Vec<f32>)>,
        name: &str,
        input: &HostImage,
    ) -> HostImage {
        let (_, weight) = &tensors[&format!("{name}.weight")];
        let (_, bias) = &tensors[&format!("{name}.bias")];
        let out_c = bias.len();
        let mut out = vec![0.0; input.w * input.h * out_c];
        for y in 0..input.h {
            for x in 0..input.w {
                for output in 0..out_c {
                    let mut value = bias[output];
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let source_y = y + ky;
                            let source_x = x + kx;
                            if source_y < 1
                                || source_y > input.h
                                || source_x < 1
                                || source_x > input.w
                            {
                                continue;
                            }
                            for channel in 0..input.c {
                                let weight_index = ((output * 3 + ky) * 3 + kx) * input.c + channel;
                                value += weight[weight_index]
                                    * input.data[((source_y - 1) * input.w + source_x - 1)
                                        * input.c
                                        + channel];
                            }
                        }
                    }
                    out[(y * input.w + x) * out_c + output] = value;
                }
            }
        }
        HostImage {
            w: input.w,
            h: input.h,
            c: out_c,
            data: out,
        }
    }

    fn host_lrelu(input: &HostImage) -> HostImage {
        let mut output = input.clone();
        for value in &mut output.data {
            *value = value.max(*value * 0.2);
        }
        output
    }

    fn host_cat(inputs: &[&HostImage]) -> HostImage {
        let first = inputs[0];
        let channels = inputs.iter().map(|input| input.c).sum();
        let mut data = vec![0.0; first.w * first.h * channels];
        for pixel in 0..first.w * first.h {
            let mut output_channel = 0;
            for input in inputs {
                for channel in 0..input.c {
                    data[pixel * channels + output_channel] = input.data[pixel * input.c + channel];
                    output_channel += 1;
                }
            }
        }
        HostImage {
            w: first.w,
            h: first.h,
            c: channels,
            data,
        }
    }

    fn host_residual(input: &HostImage, residual: &HostImage) -> HostImage {
        let mut output = input.clone();
        for (value, residual) in output.data.iter_mut().zip(&residual.data) {
            *value += residual * 0.2;
        }
        output
    }

    fn host_rdb(
        tensors: &BTreeMap<String, (Vec<i32>, Vec<f32>)>,
        prefix: &str,
        input: &HostImage,
    ) -> HostImage {
        let x1 = host_lrelu(&host_conv(tensors, &format!("{prefix}.conv1"), input));
        let x2 = host_lrelu(&host_conv(
            tensors,
            &format!("{prefix}.conv2"),
            &host_cat(&[input, &x1]),
        ));
        let x3 = host_lrelu(&host_conv(
            tensors,
            &format!("{prefix}.conv3"),
            &host_cat(&[input, &x1, &x2]),
        ));
        let x4 = host_lrelu(&host_conv(
            tensors,
            &format!("{prefix}.conv4"),
            &host_cat(&[input, &x1, &x2, &x3]),
        ));
        let x5 = host_conv(
            tensors,
            &format!("{prefix}.conv5"),
            &host_cat(&[input, &x1, &x2, &x3, &x4]),
        );
        host_residual(input, &x5)
    }

    fn host_nearest(input: &HostImage) -> HostImage {
        let mut output = HostImage {
            w: input.w * 2,
            h: input.h * 2,
            c: input.c,
            data: vec![0.0; input.w * input.h * input.c * 4],
        };
        for y in 0..output.h {
            for x in 0..output.w {
                for channel in 0..input.c {
                    output.data[(y * output.w + x) * input.c + channel] =
                        input.data[((y / 2) * input.w + x / 2) * input.c + channel];
                }
            }
        }
        output
    }

    fn host_forward(
        tensors: &BTreeMap<String, (Vec<i32>, Vec<f32>)>,
        spec: RrdbSpec,
        input: &HostImage,
    ) -> HostImage {
        let first = host_conv(tensors, "conv_first", input);
        let mut body = first.clone();
        for block in 0..spec.num_block {
            let mut out = host_rdb(tensors, &format!("body.{block}.rdb1"), &body);
            out = host_rdb(tensors, &format!("body.{block}.rdb2"), &out);
            out = host_rdb(tensors, &format!("body.{block}.rdb3"), &out);
            body = host_residual(&body, &out);
        }
        let body = host_conv(tensors, "conv_body", &body);
        let feat = HostImage {
            data: first
                .data
                .iter()
                .zip(&body.data)
                .map(|(a, b)| a + b)
                .collect(),
            ..first
        };
        let feat = host_lrelu(&host_conv(tensors, "conv_up1", &host_nearest(&feat)));
        let feat = host_lrelu(&host_conv(tensors, "conv_up2", &host_nearest(&feat)));
        let feat = host_lrelu(&host_conv(tensors, "conv_hr", &feat));
        host_conv(tensors, "conv_last", &feat)
    }

    #[test]
    fn rrdbnet_matches_independent_host_forward() {
        let spec = RrdbSpec {
            num_feat: 2,
            num_block: 1,
            num_grow_ch: 1,
        };
        let tensors = synthetic_weights(spec);
        let arrays: BTreeMap<String, Array> = tensors
            .iter()
            .map(|(name, (shape, values))| {
                (name.clone(), Array::from_f32_slice(values, shape).unwrap())
            })
            .collect();
        let model = RrdbNet::from_weights(MlxWeights::from_dense(arrays), spec).unwrap();
        let input = HostImage {
            w: 3,
            h: 2,
            c: 3,
            data: (0..18).map(|index| index as f32 * 0.07 - 0.4).collect(),
        };
        let expected = host_forward(&tensors, spec, &input);
        let stream = Stream::cpu().unwrap();
        let input_array = Array::from_f32_slice(&input.data, &[1, 2, 3, 3]).unwrap();
        let actual = model.forward(&input_array, &stream).unwrap();
        assert_eq!(actual.shape(), [1, 8, 12, 3]);
        let actual = actual.to_vec_f32(&stream).unwrap();
        let max_abs = actual
            .iter()
            .zip(&expected.data)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 2e-5, "max_abs={max_abs}");
    }

    #[test]
    fn malformed_convolution_layout_fails_before_inference() {
        let spec = RrdbSpec {
            num_feat: 2,
            num_block: 1,
            num_grow_ch: 1,
        };
        let tensors = synthetic_weights(spec);
        let arrays: BTreeMap<String, Array> = tensors
            .iter()
            .map(|(name, (shape, values))| {
                let shape = if name == "body.0.rdb1.conv1.weight" {
                    vec![1, 2, 3, 3]
                } else {
                    shape.clone()
                };
                let values = if name == "body.0.rdb1.conv1.weight" {
                    vec![0.0; 18]
                } else {
                    values.clone()
                };
                (
                    name.clone(),
                    Array::from_f32_slice(&values, &shape).unwrap(),
                )
            })
            .collect();
        let error = RrdbNet::from_weights(MlxWeights::from_dense(arrays), spec)
            .err()
            .expect("PyTorch OIHW layout must fail for the MLX OHWI contract");
        assert!(
            error.to_string().contains("body.0.rdb1.conv1.weight"),
            "{error}"
        );
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore = "set DANQING_TEST_ESRGAN_BUNDLE to the installed Real-ESRGAN x4+ directory"]
    fn installed_checkpoint_runs_on_metal_and_matches_optional_reference() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_ESRGAN_BUNDLE").expect("DANQING_TEST_ESRGAN_BUNDLE"),
        );
        let model = RrdbNet::load(&bundle).unwrap();
        let stream = Stream::gpu().unwrap();
        let values: Vec<f32> = (0..12 * 16 * 3)
            .map(|index| ((index * 37) % 255) as f32 / 255.0)
            .collect();
        let input = Array::from_f32_slice(&values, &[1, 12, 16, 3]).unwrap();
        let actual = model.forward(&input, &stream).unwrap();
        assert_eq!(actual.shape(), [1, 48, 64, 3]);
        let actual = actual.to_vec_f32(&stream).unwrap();
        assert!(actual.iter().all(|value| value.is_finite()));
        if let Ok(reference) = std::env::var("DANQING_TEST_ESRGAN_REFERENCE") {
            let reference: Vec<f32> = serde_json::from_str(
                &std::fs::read_to_string(reference).expect("read Real-ESRGAN reference"),
            )
            .expect("parse Real-ESRGAN reference");
            assert_eq!(actual.len(), reference.len());
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let mse = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).powi(2) as f64)
                .sum::<f64>()
                / actual.len() as f64;
            eprintln!("Real-ESRGAN Metal vs MLX reference max_abs={max_abs} mse={mse}");
            assert!(max_abs < 0.02, "max_abs={max_abs}");
        }
    }
}
