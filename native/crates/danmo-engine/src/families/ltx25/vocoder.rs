//! LTX 2.5 BigVGAN stereo vocoder and 16 kHz to 48 kHz bandwidth extension.

use std::path::Path;

use danmo_mlx::{Array, Dtype, Result, Stream};
use serde_json::Value;

use crate::common::mlx_ops::{scalar, slice_last};
use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

pub struct Vocoder {
    weights: MlxWeights,
    config: Value,
}

impl Vocoder {
    /// Load vocoder weights from the ingested `vocoder.safetensors` shard, or
    /// from a legacy combined `audio_vae.safetensors` that still carries the
    /// `vocoder.*` keys (pre-split workspace bundles).
    pub fn load(bundle: &Path, config: &Value) -> std::result::Result<Self, WeightError> {
        let split = bundle.join("vocoder.safetensors");
        let combined = bundle.join("audio_vae.safetensors");
        let weights = if split.is_file() {
            load_mlx_file(&split, WeightDtype::Native, None)
                .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        } else if combined.is_file() {
            let weights = load_mlx_file(&combined, WeightDtype::Native, None)
                .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
            if !weights.has("vocoder.vocoder.conv_pre.weight")
                && !weights.has("vocoder.vocoder.conv_pre.bias")
            {
                return Err(WeightError::Bundle {
                    path: combined.display().to_string(),
                    detail: "legacy audio_vae.safetensors has no vocoder.* keys; re-run LTX 2.5 ingest to produce vocoder.safetensors".into(),
                });
            }
            weights
        } else {
            return Err(WeightError::Bundle {
                path: split.display().to_string(),
                detail: "required LTX 2.5 vocoder shard is missing".into(),
            });
        };
        Ok(Self {
            weights,
            config: config.clone(),
        })
    }

    /// Decode `[B,2,T,mel]` to `[B,2,48_000*duration]` stereo waveform.
    pub fn decode(&self, mel: &Array, stream: &Stream) -> Result<Array> {
        let base = self
            .config
            .get("vocoder")
            .ok_or_else(|| danmo_mlx::Error::new("LTX 2.5 vocoder", "missing vocoder section"))?;
        let bwe = self
            .config
            .get("bwe")
            .ok_or_else(|| danmo_mlx::Error::new("LTX 2.5 vocoder", "missing bwe section"))?;
        let input_rate = integer(bwe, "input_sampling_rate", 16_000) as i32;
        let output_rate = integer(bwe, "output_sampling_rate", 48_000) as i32;
        let hop = integer(bwe, "hop_length", 80) as i32;
        if output_rate % input_rate != 0 {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 vocoder",
                "BWE output rate must be an integer multiple of input rate",
            ));
        }
        let ratio = output_rate / input_rate;
        let low = big_vgan(&self.weights, "vocoder.vocoder", base, mel, stream)?;
        let low_length = low.shape()[2];
        let output_length = low_length * ratio;
        let remainder = low_length % hop;
        let low = if remainder == 0 {
            low
        } else {
            low.pad(&[2], &[0], &[hop - remainder], 0.0, "constant", stream)?
        };
        let (batch, channels, _) = (low.shape()[0], low.shape()[1], low.shape()[2]);
        let n_fft = integer(bwe, "n_fft", 512) as i32;
        let mel_channels = integer(bwe, "num_mels", 64) as i32;
        let mel = mel_spectrogram(&self.weights, &low, n_fft, hop, mel_channels, stream)?;
        let residual = big_vgan(&self.weights, "vocoder.bwe_generator", bwe, &mel, stream)?;
        let skip = sinc_upsample(&low, ratio, stream)?;
        let output = Array::add(&residual, &skip, stream)?.clip(-1.0, 1.0, stream)?;
        let _ = batch;
        let _ = channels;
        crop_last(&output, 0, output_length, stream)
    }
}

fn big_vgan(
    weights: &MlxWeights,
    prefix: &str,
    config: &Value,
    mel: &Array,
    stream: &Stream,
) -> Result<Array> {
    let rates = integers(config, "upsample_rates", &[5, 2, 2, 2, 2, 2]);
    let up_kernels = integers(config, "upsample_kernel_sizes", &[11, 4, 4, 4, 4, 4]);
    let kernels = integers(config, "resblock_kernel_sizes", &[3, 7, 11]);
    let dilations = nested_integers(
        config,
        "resblock_dilation_sizes",
        &[&[1, 3, 5], &[1, 3, 5], &[1, 3, 5]],
    );
    let initial = integer(config, "upsample_initial_channel", 1536) as i32;
    let amp = string(config, "resblock", "AMP1") == "AMP1";
    if rates.len() != up_kernels.len() || kernels.len() != dilations.len() {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 vocoder",
            "inconsistent upsample/resblock config lengths",
        ));
    }
    let input = mel.astype(Dtype::Float32, stream)?;
    let shape = input.shape();
    let x = input
        .transpose_axes(&[0, 1, 3, 2], stream)?
        .reshape(&[shape[0], shape[1] * shape[3], shape[2]], stream)?;
    let mut x = conv1d(
        weights,
        &format!("{prefix}.conv_pre"),
        &x,
        7,
        1,
        3,
        1,
        stream,
    )?;
    for (stage, (&rate, &up_kernel)) in rates.iter().zip(&up_kernels).enumerate() {
        if !amp {
            x = leaky(&x, stream)?;
        }
        x = conv_transpose1d(
            weights,
            &format!("{prefix}.ups.{stage}"),
            &x,
            rate,
            (up_kernel - rate) / 2,
            stream,
        )?;
        let channels = initial / 2_i32.pow(stage as u32 + 1);
        let mut outputs = Vec::with_capacity(kernels.len());
        for branch in 0..kernels.len() {
            let index = stage * kernels.len() + branch;
            let block_prefix = format!("{prefix}.resblocks.{index}");
            let output = if amp {
                amp_block(
                    weights,
                    &block_prefix,
                    &x,
                    channels,
                    kernels[branch],
                    &dilations[branch],
                    stream,
                )?
            } else {
                res_block(
                    weights,
                    &block_prefix,
                    &x,
                    kernels[branch],
                    &dilations[branch],
                    stream,
                )?
            };
            outputs.push(output);
        }
        x = Array::stack(&outputs, 0, stream)?.mean_axes(&[0], false, stream)?;
        x.eval()?;
    }
    x = if amp {
        activation1d(weights, &format!("{prefix}.act_post"), &x, stream)?
    } else {
        leaky(&x, stream)?
    };
    x = conv1d(
        weights,
        &format!("{prefix}.conv_post"),
        &x,
        7,
        1,
        3,
        1,
        stream,
    )?;
    if boolean(config, "apply_final_activation", true) {
        x = if boolean(config, "use_tanh_at_final", true) {
            x.tanh(stream)?
        } else {
            x.clip(-1.0, 1.0, stream)?
        };
    }
    Ok(x)
}

fn amp_block(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    _channels: i32,
    kernel: i32,
    dilations: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    for (index, &dilation) in dilations.iter().enumerate() {
        let mut branch = activation1d(weights, &format!("{prefix}.acts1.{index}"), &x, stream)?;
        branch = conv1d(
            weights,
            &format!("{prefix}.convs1.{index}"),
            &branch,
            kernel,
            1,
            (kernel * dilation - dilation) / 2,
            dilation,
            stream,
        )?;
        branch = activation1d(weights, &format!("{prefix}.acts2.{index}"), &branch, stream)?;
        branch = conv1d(
            weights,
            &format!("{prefix}.convs2.{index}"),
            &branch,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            stream,
        )?;
        x = Array::add(&x, &branch, stream)?;
    }
    Ok(x)
}

fn res_block(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    kernel: i32,
    dilations: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    for (index, &dilation) in dilations.iter().enumerate() {
        let mut branch = conv1d(
            weights,
            &format!("{prefix}.convs1.{index}"),
            &leaky(&x, stream)?,
            kernel,
            1,
            (kernel * dilation - dilation) / 2,
            dilation,
            stream,
        )?;
        branch = conv1d(
            weights,
            &format!("{prefix}.convs2.{index}"),
            &leaky(&branch, stream)?,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            stream,
        )?;
        x = Array::add(&x, &branch, stream)?;
    }
    Ok(x)
}

fn activation1d(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let up = depthwise_filter(
        weights.tensor(&format!("{prefix}.upsample.filter"))?,
        input,
        2,
        true,
        6,
        5,
        false,
        stream,
    )?;
    let alpha = weights
        .tensor(&format!("{prefix}.act.alpha"))?
        .exp(stream)?
        .reshape(&[1, input.shape()[1], 1], stream)?;
    let beta = weights
        .tensor(&format!("{prefix}.act.beta"))?
        .exp(stream)?
        .reshape(&[1, input.shape()[1], 1], stream)?;
    let ax = Array::multiply(&up, &alpha, stream)?;
    let sine = ax.sin(stream)?;
    let squared = Array::multiply(&sine, &sine, stream)?;
    let denominator = Array::add(&beta, &scalar(1e-9)?, stream)?;
    let activated = Array::add(&up, &Array::divide(&squared, &denominator, stream)?, stream)?;
    let down = depthwise_filter(
        weights.tensor(&format!("{prefix}.downsample.lowpass.filter"))?,
        &activated,
        2,
        false,
        0,
        0,
        false,
        stream,
    )?;
    Ok(down)
}

/// Checkpoint Kaiser filter, ratio 2, depthwise up/down sampling.
fn depthwise_filter(
    filter: &Array,
    input: &Array,
    ratio: i32,
    transpose: bool,
    transpose_pad_left: i32,
    transpose_pad_right: i32,
    full_convolution: bool,
    stream: &Stream,
) -> Result<Array> {
    let channels = input.shape()[1];
    let (pad_left, pad_right) = if transpose {
        (transpose_pad_left, transpose_pad_right)
    } else {
        (5, 6)
    };
    let padded = if transpose && !full_convolution {
        zero_insert(input, ratio, stream)?
    } else {
        input.clone()
    };
    let padded = padded.pad(&[2], &[pad_left], &[pad_right], 0.0, "edge", stream)?;
    let padded = if transpose && full_convolution {
        let inserted = zero_insert(&padded, ratio, stream)?;
        let output_time = (padded.shape()[2] - 1) * ratio + 1;
        crop_last(&inserted, 0, output_time, stream)?
    } else {
        padded
    };
    let padded = if full_convolution {
        let kernel = filter.shape()[2];
        padded.pad(&[2], &[kernel - 1], &[kernel - 1], 0.0, "constant", stream)?
    } else {
        padded
    };
    let weight = filter
        .reshape(&[1, filter.shape()[2], 1], stream)?
        .repeat_axis(channels, 0, stream)?
        .astype(input.dtype(), stream)?;
    let values = padded.transpose_axes(&[0, 2, 1], stream)?;
    let output = Array::conv1d(
        &values,
        &weight,
        if transpose { 1 } else { ratio },
        0,
        1,
        channels,
        stream,
    )?;
    let output = if transpose {
        Array::multiply(&output, &scalar(ratio as f32)?, stream)?
    } else {
        output
    };
    output.transpose_axes(&[0, 2, 1], stream)
}

fn zero_insert(input: &Array, ratio: i32, stream: &Stream) -> Result<Array> {
    let (batch, channels, time) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    input
        .expand_dims(3, stream)?
        .pad(&[3], &[0], &[ratio - 1], 0.0, "constant", stream)?
        .reshape(&[batch, channels, time * ratio], stream)
}

fn mel_spectrogram(
    weights: &MlxWeights,
    input: &Array,
    n_fft: i32,
    hop: i32,
    mel_channels: i32,
    stream: &Stream,
) -> Result<Array> {
    let (batch, channels, length) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let flat = input.reshape(&[batch * channels, 1, length], stream)?;
    let padded = flat.pad(&[2], &[n_fft - hop], &[0], 0.0, "constant", stream)?;
    let basis = weights
        .tensor("vocoder.mel_stft.stft_fn.forward_basis")?
        .transpose_axes(&[0, 2, 1], stream)?;
    let spectrum = Array::conv1d(
        &padded.transpose_axes(&[0, 2, 1], stream)?,
        &basis,
        hop,
        0,
        1,
        1,
        stream,
    )?
    .transpose_axes(&[0, 2, 1], stream)?;
    let frequencies = n_fft / 2 + 1;
    let real = spectrum.slice(
        &[0, 0, 0],
        &[batch * channels, frequencies, spectrum.shape()[2]],
        &[1, 1, 1],
        stream,
    )?;
    let imag = spectrum.slice(
        &[0, frequencies, 0],
        &[batch * channels, frequencies * 2, spectrum.shape()[2]],
        &[1, 1, 1],
        stream,
    )?;
    let real2 = Array::multiply(&real, &real, stream)?;
    let imag2 = Array::multiply(&imag, &imag, stream)?;
    let magnitude = Array::add(&real2, &imag2, stream)?.sqrt(stream)?;
    let mel = Array::matmul(
        weights.tensor("vocoder.mel_stft.mel_basis")?,
        &magnitude,
        stream,
    )?;
    let log_mel = Array::maximum(&mel, &scalar(1e-5)?, stream)?.log(stream)?;
    let frames = log_mel.shape()[2];
    log_mel
        .reshape(&[batch, channels, mel_channels, frames], stream)?
        .transpose_axes(&[0, 1, 3, 2], stream)
}

fn sinc_upsample(input: &Array, ratio: i32, stream: &Stream) -> Result<Array> {
    let rolloff = 0.99f32;
    let width = (6.0f32 / rolloff).ceil() as i32;
    let kernel = 2 * width * ratio + 1;
    let mut values = Vec::with_capacity(kernel as usize);
    for index in 0..kernel {
        let time = (index as f32 / ratio as f32 - width as f32) * rolloff;
        let clamped = time.clamp(-6.0, 6.0);
        let window = (clamped * std::f32::consts::PI / 6.0 / 2.0).cos().powi(2);
        let sinc = if time.abs() < 1e-8 {
            1.0
        } else {
            (std::f32::consts::PI * time).sin() / (std::f32::consts::PI * time)
        };
        values.push(sinc * window * rolloff / ratio as f32);
    }
    let filter = Array::from_f32_slice(&values, &[1, 1, kernel])?;
    let up = depthwise_filter(&filter, input, ratio, true, width, width, true, stream)?;
    let pad_left = 2 * width * ratio;
    let pad_right = kernel - ratio;
    crop_last(&up, pad_left, up.shape()[2] - pad_right, stream)
}

#[allow(clippy::too_many_arguments)]
fn conv1d(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    _kernel: i32,
    stride: i32,
    padding: i32,
    dilation: i32,
    stream: &Stream,
) -> Result<Array> {
    let weight = weights
        .tensor(&format!("{prefix}.weight"))?
        .astype(Dtype::Float32, stream)?;
    let output = Array::conv1d(
        &input.transpose_axes(&[0, 2, 1], stream)?,
        &weight,
        stride,
        padding,
        dilation,
        1,
        stream,
    )?;
    let bias = format!("{prefix}.bias");
    let output = if weights.has(&bias) {
        Array::add(&output, weights.tensor(&bias)?, stream)?
    } else {
        output
    };
    output.transpose_axes(&[0, 2, 1], stream)
}

fn conv_transpose1d(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stride: i32,
    padding: i32,
    stream: &Stream,
) -> Result<Array> {
    let output = Array::conv_transpose1d(
        &input.transpose_axes(&[0, 2, 1], stream)?,
        weights.tensor(&format!("{prefix}.weight"))?,
        stride,
        padding,
        1,
        0,
        1,
        stream,
    )?;
    let bias = format!("{prefix}.bias");
    let output = if weights.has(&bias) {
        Array::add(&output, weights.tensor(&bias)?, stream)?
    } else {
        output
    };
    output.transpose_axes(&[0, 2, 1], stream)
}

fn leaky(input: &Array, stream: &Stream) -> Result<Array> {
    Array::maximum(
        input,
        &Array::multiply(input, &scalar(0.1)?, stream)?,
        stream,
    )
}

fn crop_last(input: &Array, start: i32, stop: i32, stream: &Stream) -> Result<Array> {
    slice_last(input, start, stop, stream)
}
fn integer(value: &Value, key: &str, default: u64) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(default)
}
fn boolean(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}
fn string(value: &Value, key: &str, default: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}
fn integers(value: &Value, key: &str, default: &[i32]) -> Vec<i32> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_i64)
                .map(|v| v as i32)
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}
fn nested_integers(value: &Value, key: &str, default: &[&[i32]]) -> Vec<Vec<i32>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(Value::as_array)
                .map(|row| {
                    row.iter()
                        .filter_map(Value::as_i64)
                        .map(|v| v as i32)
                        .collect()
                })
                .collect()
        })
        .unwrap_or_else(|| default.iter().map(|row| row.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_insert_interleaves_samples_and_zeros() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let input = danmo_mlx::Array::from_f32_slice(&[1.0, 2.0, 3.0], &[1, 1, 3]).unwrap();
        let expanded = input.expand_dims(3, &stream).unwrap();
        assert_eq!(expanded.shape(), &[1, 1, 3, 1]);
        let padded = expanded
            .pad(&[3], &[0], &[1], 0.0, "constant", &stream)
            .unwrap();
        assert_eq!(padded.shape(), &[1, 1, 3, 2]);
        assert_eq!(
            padded.to_vec_f32(&stream).unwrap(),
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
        let output = super::zero_insert(&input, 2, &stream).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6]);
        assert_eq!(
            output.to_vec_f32(&stream).unwrap(),
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    #[ignore]
    fn real_vocoder_produces_bounded_stereo_48khz_waveform() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let vocoder = super::Vocoder::load(&root, &config.vocoder).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let fixture_path = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_REFERENCE").expect("DANQING_TEST_LTX25_REFERENCE"),
        );
        let fixture =
            crate::weights::load_mlx_file(&fixture_path, crate::weights::WeightDtype::Native, None)
                .unwrap()
                .into_dense()
                .unwrap();
        let mel = fixture["vocoder_input"].clone();
        let output = vocoder.decode(&mel, &stream).unwrap();
        assert_eq!(output.shape(), &[1, 2, 1920]);
        let values = output.to_vec_f32(&stream).unwrap();
        assert!(values
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0));
        let expected = fixture["vocoder_output"].to_vec_f32(&stream).unwrap();
        let (max_abs, mean_abs) = values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold((0.0f32, 0.0f32), |(max, sum), value| {
                (max.max(value), sum + value / values.len() as f32)
            });
        println!("LTX25 vocoder: max_abs={max_abs}, mean_abs={mean_abs}");
        assert!(max_abs < 0.35 && mean_abs < 0.03);
    }
}
