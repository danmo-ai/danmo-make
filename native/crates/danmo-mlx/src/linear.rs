use crate::{Array, Dtype, Error, QuantizedArray, Result, Stream};

/// Bits and group size of an MLX affine-quantized matrix. They are not stored
/// on the arrays, so they travel with the packed tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineQuantization {
    pub bits: i32,
    pub group_size: i32,
}

impl AffineQuantization {
    pub fn new(bits: i32, group_size: i32) -> Result<Self> {
        if bits != 4 && bits != 8 {
            return Err(Error::new(
                "MLX affine quantization",
                format!("bits must be 4 or 8, got {bits}"),
            ));
        }
        if ![32, 64, 128].contains(&group_size) {
            return Err(Error::new(
                "MLX affine quantization",
                format!("group size must be 32, 64, or 128, got {group_size}"),
            ));
        }
        Ok(Self { bits, group_size })
    }
}

/// Weight of a linear layer `[out, in]`. Packed affine weights stay packed:
/// `apply` runs `quantized_matmul`, which dequantizes inside the kernel.
#[derive(Clone)]
pub enum LinearWeight {
    Dense(Array),
    Affine {
        packed: QuantizedArray,
        quantization: AffineQuantization,
    },
}

impl LinearWeight {
    pub fn affine(packed: QuantizedArray, quantization: AffineQuantization) -> Result<Self> {
        let weight = packed.weight.shape();
        let scales = packed.scales.shape();
        let biases = packed.biases.shape();
        if packed.weight.dtype() != Dtype::UInt32 || weight.len() != 2 {
            return Err(Error::new(
                "MLX affine weight",
                format!(
                    "packed weight must be 2-D uint32, got {:?} {:?}",
                    packed.weight.dtype(),
                    weight
                ),
            ));
        }
        if scales != biases || scales.len() != 2 || scales[0] != weight[0] {
            return Err(Error::new(
                "MLX affine weight",
                format!("scales {scales:?} and biases {biases:?} do not match weight {weight:?}"),
            ));
        }
        if !packed.scales.dtype().is_floating() || packed.scales.dtype() != packed.biases.dtype() {
            return Err(Error::new(
                "MLX affine weight",
                "scales and biases must share one floating dtype",
            ));
        }
        let input = weight[1] as i64 * 32 / quantization.bits as i64;
        if input != scales[1] as i64 * quantization.group_size as i64 {
            return Err(Error::new(
                "MLX affine weight",
                format!(
                    "packed width {} at {} bits is {input} inputs, but {} groups of {} is {}",
                    weight[1],
                    quantization.bits,
                    scales[1],
                    quantization.group_size,
                    scales[1] as i64 * quantization.group_size as i64
                ),
            ));
        }
        Ok(Self::Affine {
            packed,
            quantization,
        })
    }

    pub fn apply(&self, input: &Array, bias: Option<&Array>, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense(weight) => Array::linear(input, weight, bias, stream),
            Self::Affine {
                packed,
                quantization,
            } => {
                let output = Array::quantized_matmul(
                    input,
                    packed,
                    true,
                    quantization.group_size,
                    quantization.bits,
                    stream,
                )?;
                match bias {
                    Some(bias) => Array::add(&output, bias, stream),
                    None => Ok(output),
                }
            }
        }
    }

    /// Dtype activations should carry into this layer.
    pub fn compute_dtype(&self) -> Dtype {
        match self {
            Self::Dense(weight) => weight.dtype(),
            Self::Affine { packed, .. } => packed.scales.dtype(),
        }
    }

    pub fn dense(&self) -> Option<&Array> {
        match self {
            Self::Dense(weight) => Some(weight),
            Self::Affine { .. } => None,
        }
    }

    pub fn quantization(&self) -> Option<AffineQuantization> {
        match self {
            Self::Dense(_) => None,
            Self::Affine { quantization, .. } => Some(*quantization),
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Dense(weight) => weight.nbytes(),
            Self::Affine { packed, .. } => packed.nbytes(),
        }
    }
}
