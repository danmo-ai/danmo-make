//! Packaged TAEF1 decoder for denoise-step previews.
//!
//! The bytes are `taef1_decoder.safetensors` (2,464,414 bytes, sha256
//! `bb41500b…a1d4`): a 16-channel sequential decoder, three nearest 2×
//! upsamples, 64-wide 3×3 convs. Callers pass the VAE scale and shift. This
//! module does not know which family produced the latent.

use std::collections::BTreeMap;

use danmo_mlx::{Array, CacheKey, CacheRevision, Dtype, Error, ModelCache, Result, Stream, UnaryClosure};

use crate::weights::MlxWeights;

const TAEF1_BYTES: &[u8] = include_bytes!("data/taef1_decoder.safetensors");
const LATENT_CHANNELS: i32 = 16;

#[derive(Clone, Copy)]
pub struct PreviewScale {
    pub scaling: f32,
    pub shift: f32,
}

pub struct CompiledPreview {
    closure: UnaryClosure,
}

impl CompiledPreview {
    pub fn apply(&self, latent_nchw: &Array) -> Result<Array> {
        self.closure.apply(latent_nchw)
    }
}

pub fn cached<'a>(
    cache: &'a mut ModelCache,
    latent_h: i32,
    latent_w: i32,
    scale: PreviewScale,
    stream: &Stream,
) -> Result<&'a CompiledPreview> {
    let key = CacheKey {
        id: format!(
            "taesd-taef1#{latent_h}x{latent_w}#{}#{}",
            scale.scaling.to_bits(),
            scale.shift.to_bits()
        ),
        revision: CacheRevision {
            length: TAEF1_BYTES.len() as u64,
            modified_ns: 0,
            mix: 0xbb41_500b,
        },
    };
    if cache.get::<CompiledPreview>(&key)?.is_none() {
        let weights = load_weights(stream)?;
        let bytes = weights.nbytes();
        let closure = UnaryClosure::new(stream.clone(), move |latent, stream| {
            decode(&weights, scale, latent, stream)
        })?
        .compile(false)?;
        cache.insert(key.clone(), CompiledPreview { closure }, bytes)?;
    }
    cache.get::<CompiledPreview>(&key)?.ok_or_else(|| {
        Error::new(
            "TAEF1 preview",
            "compiled preview decoder is not in the executor cache",
        )
    })
}

fn decode(weights: &MlxWeights, scale: PreviewScale, latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = latent.shape();
    let batched = match shape.as_slice() {
        [1, channels, _, _] if *channels == LATENT_CHANNELS => latent.clone(),
        [channels, _, _] if *channels == LATENT_CHANNELS => {
            latent.reshape(&[1, shape[0], shape[1], shape[2]], stream)?
        }
        other => {
            return Err(Error::new(
                "TAEF1 preview",
                format!("latent shape {other:?} is not [16, H, W] or [1, 16, H, W]"),
            ));
        }
    };
    let mut z = batched.astype(Dtype::Float32, stream)?;
    if scale.scaling != 0.0 && (scale.scaling - 1.0).abs() > 1e-6 {
        let scaling = scalar(scale.scaling, stream)?;
        let shift = scalar(scale.shift, stream)?;
        z = Array::add(&Array::divide(&z, &scaling, stream)?, &shift, stream)?;
    }
    let magnitude = scalar(6.0, stream)?;
    let half = scalar(0.5, stream)?;
    z = Array::add(&Array::divide(&z, &magnitude, stream)?, &half, stream)?;
    z = clip01(&z, stream)?;
    let mut x = z.transpose_axes(&[0, 2, 3, 1], stream)?;
    x = latent_clamp(&x, stream)?;
    x = conv(weights, "1", &x, stream)?;
    for index in [3, 4, 5] {
        x = block(weights, index, &x, stream)?;
    }
    x = nearest_x2(&x, stream)?;
    x = conv(weights, "7", &x, stream)?;
    for index in [8, 9, 10] {
        x = block(weights, index, &x, stream)?;
    }
    x = nearest_x2(&x, stream)?;
    x = conv(weights, "12", &x, stream)?;
    for index in [13, 14, 15] {
        x = block(weights, index, &x, stream)?;
    }
    x = nearest_x2(&x, stream)?;
    x = conv(weights, "17", &x, stream)?;
    x = block(weights, 18, &x, stream)?;
    x = conv(weights, "19", &x, stream)?;
    clip01(&x, stream)
}

fn block(weights: &MlxWeights, index: i32, x: &Array, stream: &Stream) -> Result<Array> {
    let mut hidden = conv(weights, &format!("{index}.conv.0"), x, stream)?;
    hidden = relu(&hidden, stream)?;
    hidden = conv(weights, &format!("{index}.conv.2"), &hidden, stream)?;
    hidden = relu(&hidden, stream)?;
    hidden = conv(weights, &format!("{index}.conv.4"), &hidden, stream)?;
    relu(&Array::add(&hidden, x, stream)?, stream)
}

fn conv(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let kernel = weights.tensor(&format!("{prefix}.weight"))?;
    let mut output = Array::conv2d(input, kernel, [1, 1], [1, 1], [1, 1], 1, stream)?;
    let bias = format!("{prefix}.bias");
    if weights.has(&bias) {
        output = Array::add(&output, weights.tensor(&bias)?, stream)?;
    }
    Ok(output)
}

fn latent_clamp(x: &Array, stream: &Stream) -> Result<Array> {
    let three = scalar(3.0, stream)?;
    let scaled = Array::divide(x, &three, stream)?;
    Array::multiply(&scaled.tanh(stream)?, &three, stream)
}

fn relu(x: &Array, stream: &Stream) -> Result<Array> {
    let half = scalar(0.5, stream)?;
    Array::multiply(&Array::add(x, &x.abs(stream)?, stream)?, &half, stream)
}

fn clip01(x: &Array, stream: &Stream) -> Result<Array> {
    let one = scalar(1.0, stream)?;
    let below = relu(&Array::subtract(&one, x, stream)?, stream)?;
    relu(&Array::subtract(&one, &below, stream)?, stream)
}

fn nearest_x2(input: &Array, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let x = input.reshape(&[batch, height, 1, width, channels], stream)?;
    let x = Array::concatenate_axis(&[x.clone(), x], 2, stream)?
        .reshape(&[batch, height * 2, width, channels], stream)?;
    let x = x.reshape(&[batch, height * 2, width, 1, channels], stream)?;
    Array::concatenate_axis(&[x.clone(), x], 3, stream)?
        .reshape(&[batch, height * 2, width * 2, channels], stream)
}

fn scalar(value: f32, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[value], &[])?.astype(Dtype::Float32, stream)
}

fn load_weights(stream: &Stream) -> Result<MlxWeights> {
    let mut tensors = BTreeMap::new();
    for (name, shape, values) in read_tensors(TAEF1_BYTES)? {
        let shape_i: Vec<i32> = shape.iter().map(|dim| *dim as i32).collect();
        let mut array = Array::from_f32_slice(&values, &shape_i)?;
        if name.ends_with(".weight") && shape_i.len() == 4 {
            array = array.transpose_axes(&[0, 2, 3, 1], stream)?;
            array.eval()?;
        }
        tensors.insert(name, array);
    }
    Ok(MlxWeights::from_dense(tensors))
}

fn read_tensors(bytes: &[u8]) -> Result<Vec<(String, Vec<u64>, Vec<f32>)>> {
    if bytes.len() < 8 {
        return Err(Error::new("TAEF1 preview", "safetensors header is truncated"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize.checked_add(header_len).filter(|end| *end <= bytes.len()).ok_or_else(|| {
        Error::new("TAEF1 preview", "safetensors header extends past the file")
    })?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|error| {
        Error::new("TAEF1 preview", format!("safetensors header is not json: {error}"))
    })?;
    let object = header.as_object().ok_or_else(|| {
        Error::new("TAEF1 preview", "safetensors header is not an object")
    })?;
    let mut tensors = Vec::new();
    for (name, meta) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta.get("dtype").and_then(|value| value.as_str()).unwrap_or("");
        if dtype != "BF16" {
            return Err(Error::new(
                "TAEF1 preview",
                format!("{name} dtype {dtype} is not BF16"),
            ));
        }
        let shape: Vec<u64> = meta
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| Error::new("TAEF1 preview", format!("{name} is missing a shape")))?
            .iter()
            .map(|value| value.as_u64().unwrap_or(0))
            .collect();
        let offsets = meta
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| Error::new("TAEF1 preview", format!("{name} is missing data offsets")))?;
        let start = offsets.first().and_then(|value| value.as_u64()).unwrap_or(0) as usize + header_end;
        let end = offsets.get(1).and_then(|value| value.as_u64()).unwrap_or(0) as usize + header_end;
        if end > bytes.len() || end < start {
            return Err(Error::new(
                "TAEF1 preview",
                format!("{name} data offsets escape the file"),
            ));
        }
        let count = shape.iter().product::<u64>() as usize;
        let data = &bytes[start..end];
        if data.len() != count * 2 {
            return Err(Error::new(
                "TAEF1 preview",
                format!("{name} byte length does not match its shape"),
            ));
        }
        let values = data
            .chunks_exact(2)
            .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
            .collect();
        tensors.push((name.clone(), shape, values));
    }
    if tensors.iter().all(|(name, _, _)| name != "1.weight") {
        return Err(Error::new(
            "TAEF1 preview",
            "packaged decoder is missing 1.weight",
        ));
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::{read_tensors, TAEF1_BYTES};

    #[test]
    fn packaged_decoder_is_the_sequential_taef1() {
        assert_eq!(TAEF1_BYTES.len(), 2_464_414);
        let tensors = read_tensors(TAEF1_BYTES).unwrap();
        assert_eq!(tensors.len(), 67);
        let stem = tensors.iter().find(|(name, _, _)| name == "1.weight").unwrap();
        assert_eq!(stem.1, vec![64, 16, 3, 3]);
        assert!(tensors.iter().any(|(name, _, _)| name == "19.bias"));
        assert!(tensors.iter().all(|(name, _, _)| name != "7.bias"));
    }
}
