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

/// One low-rank residual `scale * B(Ax)` added after the base matmul.
/// `a` is `[rank, in]`, `b` is `[out, rank]`.
#[derive(Clone)]
pub struct LoraResidual {
    pub a: Array,
    pub b: Array,
    pub scale: f32,
}

/// Weight of a linear layer `[out, in]`. Packed affine weights stay packed:
/// `apply` runs `quantized_matmul`, which dequantizes inside the kernel.
///
/// The representation is private on purpose: callers can only `apply` the
/// layer, so a model cannot branch on the storage format or dequantize the
/// table back into a dense resident copy. LoRA residuals share that `apply`.
#[derive(Clone)]
pub struct LinearWeight {
    repr: Repr,
    loras: Vec<LoraResidual>,
}

#[derive(Clone)]
enum Repr {
    Dense(Array),
    Affine {
        packed: QuantizedArray,
        quantization: AffineQuantization,
    },
}

impl LinearWeight {
    pub fn dense(weight: Array) -> Self {
        Self {
            repr: Repr::Dense(weight),
            loras: Vec::new(),
        }
    }

    /// Replace the residual list. The base matrix is the same buffer.
    pub fn with_loras(&self, loras: Vec<LoraResidual>) -> Self {
        Self {
            repr: self.repr.clone(),
            loras,
        }
    }

    /// `(out, in)` of a matrix linear. Higher-rank tensors are not linears.
    pub fn out_in(&self) -> Result<Option<(i64, i64)>> {
        match &self.repr {
            Repr::Dense(weight) => {
                let shape = weight.shape();
                if shape.len() != 2 {
                    return Ok(None);
                }
                Ok(Some((shape[0] as i64, shape[1] as i64)))
            }
            Repr::Affine {
                packed,
                quantization,
            } => {
                let weight = packed.weight.shape();
                if weight.len() != 2 {
                    return Ok(None);
                }
                let inn = weight[1] as i64 * 32 / quantization.bits as i64;
                Ok(Some((weight[0] as i64, inn)))
            }
        }
    }

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
        Ok(Self {
            repr: Repr::Affine {
                packed,
                quantization,
            },
            loras: Vec::new(),
        })
    }

    pub fn apply(&self, input: &Array, bias: Option<&Array>, stream: &Stream) -> Result<Array> {
        let output = match &self.repr {
            Repr::Dense(weight) => {
                let weight = weight.matrix_transpose(stream)?;
                match bias {
                    Some(bias) => Array::addmm(bias, input, &weight, 1.0, 1.0, stream)?,
                    None => Array::matmul(input, &weight, stream)?,
                }
            }
            Repr::Affine {
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
                    Some(bias) => Array::add(&output, bias, stream)?,
                    None => output,
                }
            }
        };
        self.add_loras(input, output, stream)
    }

    fn add_loras(&self, input: &Array, output: Array, stream: &Stream) -> Result<Array> {
        let mut output = output;
        for lora in &self.loras {
            if !lora.scale.is_finite() {
                return Err(Error::new("MLX LoRA", "scale is not a finite number"));
            }
            let a = lora.a.matrix_transpose(stream)?;
            let ax = Array::matmul(input, &a, stream)?;
            let b = lora.b.matrix_transpose(stream)?;
            let delta = Array::matmul(&ax, &b, stream)?;
            let scale = Array::from_f32_slice(&[lora.scale], &[1])?;
            let delta = Array::multiply(&delta, &scale, stream)?;
            output = Array::add(&output, &delta, stream)?;
        }
        Ok(output)
    }

    /// Dtype activations should carry into this layer.
    pub fn compute_dtype(&self) -> Dtype {
        match &self.repr {
            Repr::Dense(weight) => weight.dtype(),
            Repr::Affine { packed, .. } => packed.scales.dtype(),
        }
    }

    /// Materialize a dense copy of this weight. Only for components without
    /// matmul linears (embeddings) and for explicit test/debug dequantized
    /// references — never for the production quantized matmul path.
    pub fn dequantize_embedding(&self, stream: &Stream) -> Result<Array> {
        match &self.repr {
            Repr::Dense(weight) => Ok(weight.clone()),
            Repr::Affine {
                packed,
                quantization,
            } => packed.dequantize(quantization.group_size, quantization.bits, stream),
        }
    }

    /// Resident bytes: packed codes plus scales and biases for affine layers.
    /// LoRA factors are extra buffers and are included.
    pub fn nbytes(&self) -> usize {
        let base = match &self.repr {
            Repr::Dense(weight) => weight.nbytes(),
            Repr::Affine { packed, .. } => packed.nbytes(),
        };
        base + self
            .loras
            .iter()
            .map(|lora| lora.a.nbytes() + lora.b.nbytes())
            .sum::<usize>()
    }
}
