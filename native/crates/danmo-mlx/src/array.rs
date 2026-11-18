use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::error::{check, check_infallible, clear_error, initialize, Result};
use crate::stream::Stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Float32,
    Float16,
    BFloat16,
    Int32,
    UInt32,
}

impl Dtype {
    pub(crate) fn raw(self) -> sys::mlx_dtype {
        match self {
            Self::Float32 => sys::mlx_dtype__MLX_FLOAT32,
            Self::Float16 => sys::mlx_dtype__MLX_FLOAT16,
            Self::BFloat16 => sys::mlx_dtype__MLX_BFLOAT16,
            Self::Int32 => sys::mlx_dtype__MLX_INT32,
            Self::UInt32 => sys::mlx_dtype__MLX_UINT32,
        }
    }

    pub fn is_floating(self) -> bool {
        matches!(self, Self::Float32 | Self::Float16 | Self::BFloat16)
    }
}

pub struct Array {
    pub(crate) raw: sys::mlx_array,
    _thread_affine: PhantomData<Rc<()>>,
}

impl Array {
    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Result<Self> {
        let expected = shape.iter().try_fold(1usize, |size, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| size.checked_mul(dimension))
        });
        if expected != Some(data.len()) {
            return Err(crate::Error::new(
                "Array::from_f32_slice",
                format!("shape {shape:?} does not contain {} elements", data.len()),
            ));
        }
        initialize();
        clear_error();
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                Dtype::Float32.raw(),
            )
        };
        check_infallible("mlx_array_new_data", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub(crate) fn empty() -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_array_new() };
        check_infallible("mlx_array_new", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn from_i32_slice(data: &[i32], shape: &[i32]) -> Result<Self> {
        let expected = shape.iter().try_fold(1usize, |size, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| size.checked_mul(dimension))
        });
        if expected != Some(data.len()) {
            return Err(crate::Error::new(
                "Array::from_i32_slice",
                format!("shape {shape:?} does not contain {} elements", data.len()),
            ));
        }
        initialize();
        clear_error();
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                Dtype::Int32.raw(),
            )
        };
        check_infallible("mlx_array_new_data", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dimensions = sys::mlx_array_ndim(self.raw);
            let shape = sys::mlx_array_shape(self.raw);
            if dimensions == 0 || shape.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(shape, dimensions).to_vec()
            }
        }
    }

    pub fn dtype(&self) -> Dtype {
        let raw = unsafe { sys::mlx_array_dtype(self.raw) };
        [
            Dtype::Float32,
            Dtype::Float16,
            Dtype::BFloat16,
            Dtype::Int32,
            Dtype::UInt32,
        ]
        .into_iter()
        .find(|dtype| dtype.raw() == raw)
        .unwrap_or_else(|| panic!("unsupported MLX dtype {raw}"))
    }

    pub fn nbytes(&self) -> usize {
        unsafe { sys::mlx_array_nbytes(self.raw) }
    }

    pub fn eval(&self) -> Result<()> {
        clear_error();
        check(unsafe { sys::mlx_array_eval(self.raw) }, "mlx_array_eval")
    }

    /// Evaluate several arrays in one call (`mx.eval(a, b, ...)`), sharing the
    /// lazy-graph traversal exactly like main's multi-array eval boundaries.
    pub fn eval_all(arrays: &[&Array]) -> Result<()> {
        initialize();
        let values = VectorArray::from_arrays(
            &arrays
                .iter()
                .map(|array| (*array).clone())
                .collect::<Vec<_>>(),
        )?;
        clear_error();
        check(unsafe { sys::mlx_eval(values.raw) }, "mlx_eval")
    }

    pub fn astype(&self, dtype: Dtype, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_astype(&mut output.raw, self.raw, dtype.raw(), stream.raw) },
            "mlx_astype",
        )?;
        Ok(output)
    }

    pub fn sigmoid(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_sigmoid(&mut output.raw, self.raw, stream.raw) },
            "mlx_sigmoid",
        )?;
        Ok(output)
    }

    pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_arange(&mut output.raw, start, stop, step, dtype.raw(), stream.raw) },
            "mlx_arange",
        )?;
        Ok(output)
    }

    pub fn cos(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_cos(&mut output.raw, self.raw, stream.raw) },
            "mlx_cos",
        )?;
        Ok(output)
    }

    pub fn exp(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_exp(&mut output.raw, self.raw, stream.raw) },
            "mlx_exp",
        )?;
        Ok(output)
    }

    pub fn sin(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_sin(&mut output.raw, self.raw, stream.raw) },
            "mlx_sin",
        )?;
        Ok(output)
    }

    pub fn power(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_power(&mut output.raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_power",
        )?;
        Ok(output)
    }

    pub fn divide(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_divide(&mut output.raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_divide",
        )?;
        Ok(output)
    }

    pub fn sqrt(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_sqrt(&mut output.raw, self.raw, stream.raw) },
            "mlx_sqrt",
        )?;
        Ok(output)
    }

    pub fn abs(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_abs(&mut output.raw, self.raw, stream.raw) },
            "mlx_abs",
        )?;
        Ok(output)
    }

    pub fn isfinite_all(&self, stream: &Stream) -> Result<Self> {
        let finite = Self::empty()?;
        let mut raw = finite.raw;
        clear_error();
        check(
            unsafe { sys::mlx_isfinite(&mut raw, self.raw, stream.raw) },
            "mlx_isfinite",
        )?;
        let finite = Self {
            raw,
            _thread_affine: PhantomData,
        };
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_all(&mut output.raw, finite.raw, false, stream.raw) },
            "mlx_all",
        )?;
        Ok(output)
    }

    pub fn mean(&self, keepdims: bool, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_mean(&mut output.raw, self.raw, keepdims, stream.raw) },
            "mlx_mean",
        )?;
        Ok(output)
    }

    pub fn mean_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_mean_axes(
                    &mut output.raw,
                    self.raw,
                    axes.as_ptr(),
                    axes.len(),
                    keepdims,
                    stream.raw,
                )
            },
            "mlx_mean_axes",
        )?;
        Ok(output)
    }

    pub fn conv2d(
        input: &Self,
        weight: &Self,
        stride: [i32; 2],
        padding: [i32; 2],
        dilation: [i32; 2],
        groups: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_conv2d(
                    &mut output.raw,
                    input.raw,
                    weight.raw,
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    dilation[0],
                    dilation[1],
                    groups,
                    stream.raw,
                )
            },
            "mlx_conv2d",
        )?;
        Ok(output)
    }

    pub fn tanh(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_tanh(&mut output.raw, self.raw, stream.raw) },
            "mlx_tanh",
        )?;
        Ok(output)
    }

    pub fn reshape(&self, shape: &[i32], stream: &Stream) -> Result<Self> {
        let elements = shape.iter().try_fold(1usize, |size, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| size.checked_mul(dimension))
        });
        if elements != Some(unsafe { sys::mlx_array_size(self.raw) }) {
            return Err(crate::Error::new(
                "mlx_reshape",
                format!("shape {shape:?} changes the array element count"),
            ));
        }
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_reshape(
                    &mut output.raw,
                    self.raw,
                    shape.as_ptr(),
                    shape.len(),
                    stream.raw,
                )
            },
            "mlx_reshape",
        )?;
        Ok(output)
    }

    pub fn scatter_add_axis(
        &self,
        indices: &Self,
        values: &Self,
        axis: i32,
        stream: &Stream,
    ) -> Result<Self> {
        if indices.dtype() != Dtype::Int32 {
            return Err(crate::Error::new(
                "mlx_scatter_add_axis",
                "indices must be int32",
            ));
        }
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_scatter_add_axis(
                    &mut output.raw,
                    self.raw,
                    indices.raw,
                    values.raw,
                    axis,
                    stream.raw,
                )
            },
            "mlx_scatter_add_axis",
        )?;
        Ok(output)
    }

    pub fn take_axis(&self, indices: &Self, axis: i32, stream: &Stream) -> Result<Self> {
        if indices.dtype().raw() != Dtype::Int32.raw() {
            return Err(crate::Error::new(
                "mlx_take_axis",
                format!(
                    "indices must be int32, got {:?} (raw {})",
                    indices.dtype(),
                    indices.dtype().raw()
                ),
            ));
        }
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_take_axis(&mut output.raw, self.raw, indices.raw, axis, stream.raw) },
            "mlx_take_axis",
        )?;
        Ok(output)
    }

    pub fn concatenate_axis(arrays: &[Self], axis: i32, stream: &Stream) -> Result<Self> {
        let values = VectorArray::from_arrays(arrays)?;
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_concatenate_axis(&mut output.raw, values.raw, axis, stream.raw) },
            "mlx_concatenate_axis",
        )?;
        Ok(output)
    }

    pub fn broadcast_to(&self, shape: &[i32], stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_broadcast_to(
                    &mut output.raw,
                    self.raw,
                    shape.as_ptr(),
                    shape.len(),
                    stream.raw,
                )
            },
            "mlx_broadcast_to",
        )?;
        Ok(output)
    }

    pub fn layer_norm(&self, eps: f32, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        let empty = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        clear_error();
        check(
            unsafe {
                sys::mlx_fast_layer_norm(&mut output.raw, self.raw, empty, empty, eps, stream.raw)
            },
            "mlx_fast_layer_norm",
        )?;
        Ok(output)
    }

    pub fn negative(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_negative(&mut output.raw, self.raw, stream.raw) },
            "mlx_negative",
        )?;
        Ok(output)
    }

    pub fn expand_dims(&self, axis: i32, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_expand_dims(&mut output.raw, self.raw, axis, stream.raw) },
            "mlx_expand_dims",
        )?;
        Ok(output)
    }

    pub fn repeat_axis(&self, repeats: i32, axis: i32, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_repeat_axis(&mut output.raw, self.raw, repeats, axis, stream.raw) },
            "mlx_repeat_axis",
        )?;
        Ok(output)
    }

    pub fn split_sections(&self, indices: &[i32], axis: i32, stream: &Stream) -> Result<Vec<Self>> {
        let mut values = VectorArray::new()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_split_sections(
                    &mut values.raw,
                    self.raw,
                    indices.as_ptr(),
                    indices.len(),
                    axis,
                    stream.raw,
                )
            },
            "mlx_split_sections",
        )?;
        (0..values.len()).map(|index| values.get(index)).collect()
    }

    pub fn subtract(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_subtract(&mut output.raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_subtract",
        )?;
        Ok(output)
    }

    pub fn transpose_axes(&self, axes: &[i32], stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_transpose_axes(
                    &mut output.raw,
                    self.raw,
                    axes.as_ptr(),
                    axes.len(),
                    stream.raw,
                )
            },
            "mlx_transpose_axes",
        )?;
        Ok(output)
    }

    pub fn squeeze_axis(&self, axis: i32, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_squeeze_axis(&mut output.raw, self.raw, axis, stream.raw) },
            "mlx_squeeze_axis",
        )?;
        Ok(output)
    }

    pub fn slice(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        stream: &Stream,
    ) -> Result<Self> {
        let dimensions = unsafe { sys::mlx_array_ndim(self.raw) };
        if start.len() != dimensions || stop.len() != dimensions || strides.len() != dimensions {
            return Err(crate::Error::new(
                "mlx_slice",
                "start/stop/strides must cover every array dimension",
            ));
        }
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_slice(
                    &mut output.raw,
                    self.raw,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream.raw,
                )
            },
            "mlx_slice",
        )?;
        Ok(output)
    }

    pub fn stack(arrays: &[Self], axis: i32, stream: &Stream) -> Result<Self> {
        let values = VectorArray::from_arrays(arrays)?;
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_stack_axis(&mut output.raw, values.raw, axis, stream.raw) },
            "mlx_stack_axis",
        )?;
        Ok(output)
    }

    pub fn add(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_add(&mut output.raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_add",
        )?;
        Ok(output)
    }

    pub fn multiply(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_multiply(&mut output.raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_multiply",
        )?;
        Ok(output)
    }

    pub fn silu(&self, stream: &Stream) -> Result<Self> {
        let gate = self.sigmoid(stream)?;
        Self::multiply(self, &gate, stream)
    }

    pub fn matrix_transpose(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_matrix_transpose(&mut output.raw, self.raw, stream.raw) },
            "mlx_matrix_transpose",
        )?;
        Ok(output)
    }

    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        let empty = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let weight = weight.map_or(empty, |weight| weight.raw);
        clear_error();
        check(
            unsafe { sys::mlx_fast_rms_norm(&mut output.raw, self.raw, weight, eps, stream.raw) },
            "mlx_fast_rms_norm",
        )?;
        Ok(output)
    }

    pub fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = weight.matrix_transpose(stream)?;
        let output = Self::matmul(input, &weight, stream)?;
        match bias {
            Some(bias) => Self::add(&output, bias, stream),
            None => Ok(output),
        }
    }

    pub fn addmm(
        bias: &Self,
        lhs: &Self,
        rhs: &Self,
        alpha: f32,
        beta: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        let mut raw = output.raw;
        clear_error();
        check(
            unsafe {
                sys::mlx_addmm(
                    &mut raw, bias.raw, lhs.raw, rhs.raw, alpha, beta, stream.raw,
                )
            },
            "mlx_addmm",
        )?;
        output.raw = raw;
        Ok(output)
    }

    pub fn matmul(lhs: &Self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        let mut raw = output.raw;
        clear_error();
        check(
            unsafe { sys::mlx_matmul(&mut raw, lhs.raw, rhs.raw, stream.raw) },
            "mlx_matmul",
        )?;
        output.raw = raw;
        Ok(output)
    }

    pub fn contiguous(&self, stream: &Stream) -> Result<Self> {
        let mut output = Self::empty()?;
        let mut raw = output.raw;
        clear_error();
        check(
            unsafe { sys::mlx_contiguous(&mut raw, self.raw, false, stream.raw) },
            "mlx_contiguous",
        )?;
        output.raw = raw;
        Ok(output)
    }

    pub fn to_vec_f32(&self, stream: &Stream) -> Result<Vec<f32>> {
        let contiguous = if self.dtype() == Dtype::Float32 {
            self.contiguous(stream)?
        } else {
            self.astype(Dtype::Float32, stream)?.contiguous(stream)?
        };
        contiguous.eval()?;
        let size = unsafe { sys::mlx_array_size(contiguous.raw) };
        if size == 0 {
            return Ok(Vec::new());
        }
        let data = unsafe { sys::mlx_array_data_float32(contiguous.raw) };
        if data.is_null() {
            return Err(crate::Error::new(
                "mlx_array_data_float32",
                "evaluated array returned a null data pointer",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(data, size) }.to_vec())
    }

    /// Read the raw element bytes of an evaluated array without a dtype cast.
    /// For BF16 arrays each pair of bytes is one bfloat16 value; callers can
    /// compare these bytes exactly instead of paying a GPU float32 cast.
    pub fn to_raw_bytes(&self, stream: &Stream) -> Result<Vec<u8>> {
        let contiguous = self.contiguous(stream)?;
        contiguous.eval()?;
        let bytes = contiguous.nbytes();
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let data = unsafe { sys::mlx_array_data_uint8(contiguous.raw) };
        if data.is_null() {
            return Err(crate::Error::new(
                "mlx_array_data_uint8",
                "evaluated array returned a null data pointer",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(data, bytes) }.to_vec())
    }

    pub fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask_mode: &str,
        mask: Option<&Self>,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        let mut raw = output.raw;
        let mask_mode = CString::new(mask_mode).map_err(|_| {
            crate::Error::new(
                "mlx_fast_scaled_dot_product_attention",
                "mask mode contains a null byte",
            )
        })?;
        let empty = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mask = mask.map_or(empty, |mask| mask.raw);
        clear_error();
        check(
            unsafe {
                sys::mlx_fast_scaled_dot_product_attention(
                    &mut raw,
                    queries.raw,
                    keys.raw,
                    values.raw,
                    scale,
                    mask_mode.as_ptr(),
                    mask,
                    empty,
                    false,
                    stream.raw,
                )
            },
            "mlx_fast_scaled_dot_product_attention",
        )?;
        output.raw = raw;
        Ok(output)
    }

    pub fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_conv1d(
                    &mut output.raw,
                    input.raw,
                    weight.raw,
                    stride,
                    padding,
                    dilation,
                    groups,
                    stream.raw,
                )
            },
            "mlx_conv1d",
        )?;
        Ok(output)
    }

    pub fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_conv_transpose1d(
                    &mut output.raw,
                    input.raw,
                    weight.raw,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    groups,
                    stream.raw,
                )
            },
            "mlx_conv_transpose1d",
        )?;
        Ok(output)
    }

    pub fn conv3d(
        input: &Self,
        weight: &Self,
        stride: [i32; 3],
        padding: [i32; 3],
        dilation: [i32; 3],
        groups: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_conv3d(
                    &mut output.raw,
                    input.raw,
                    weight.raw,
                    stride[0],
                    stride[1],
                    stride[2],
                    padding[0],
                    padding[1],
                    padding[2],
                    dilation[0],
                    dilation[1],
                    dilation[2],
                    groups,
                    stream.raw,
                )
            },
            "mlx_conv3d",
        )?;
        Ok(output)
    }

    pub fn conv_transpose3d(
        input: &Self,
        weight: &Self,
        stride: [i32; 3],
        padding: [i32; 3],
        output_padding: [i32; 3],
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_conv_transpose3d(
                    &mut output.raw,
                    input.raw,
                    weight.raw,
                    stride[0],
                    stride[1],
                    stride[2],
                    padding[0],
                    padding[1],
                    padding[2],
                    1,
                    1,
                    1,
                    output_padding[0],
                    output_padding[1],
                    output_padding[2],
                    1,
                    stream.raw,
                )
            },
            "mlx_conv_transpose3d",
        )?;
        Ok(output)
    }

    pub fn quantize_affine(
        &self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<QuantizedArray> {
        let mut values = VectorArray::new()?;
        let mode = CString::new("affine").unwrap();
        let group_size = sys::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let bits = sys::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        let global_scale = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        clear_error();
        check(
            unsafe {
                sys::mlx_quantize(
                    &mut values.raw,
                    self.raw,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    global_scale,
                    stream.raw,
                )
            },
            "mlx_quantize",
        )?;
        if values.len() != 3 {
            return Err(crate::Error::new(
                "mlx_quantize",
                format!("affine quantization returned {} arrays", values.len()),
            ));
        }
        Ok(QuantizedArray {
            weight: values.get(0)?,
            scales: values.get(1)?,
            biases: values.get(2)?,
        })
    }

    pub fn quantized_matmul(
        input: &Self,
        weight: &QuantizedArray,
        transpose: bool,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let mut output = Self::empty()?;
        let group_size = sys::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let bits = sys::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        let mode = CString::new("affine").unwrap();
        clear_error();
        check(
            unsafe {
                sys::mlx_quantized_matmul(
                    &mut output.raw,
                    input.raw,
                    weight.weight.raw,
                    weight.scales.raw,
                    weight.biases.raw,
                    transpose,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    stream.raw,
                )
            },
            "mlx_quantized_matmul",
        )?;
        Ok(output)
    }
}

#[derive(Clone)]
pub struct QuantizedArray {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

impl QuantizedArray {
    pub fn nbytes(&self) -> usize {
        self.weight.nbytes() + self.scales.nbytes() + self.biases.nbytes()
    }

    /// Materializes the dense matrix. Inference must use
    /// `Array::quantized_matmul`; this exists for parity references.
    pub fn dequantize(&self, group_size: i32, bits: i32, stream: &Stream) -> Result<Array> {
        let mut output = Array::empty()?;
        let mode = CString::new("affine").unwrap();
        let global_scale = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let dtype = sys::mlx_optional_dtype {
            value: self.scales.dtype().raw(),
            has_value: false,
        };
        clear_error();
        check(
            unsafe {
                sys::mlx_dequantize(
                    &mut output.raw,
                    self.weight.raw,
                    self.scales.raw,
                    self.biases.raw,
                    sys::mlx_optional_int {
                        value: group_size,
                        has_value: true,
                    },
                    sys::mlx_optional_int {
                        value: bits,
                        has_value: true,
                    },
                    mode.as_ptr(),
                    global_scale,
                    dtype,
                    stream.raw,
                )
            },
            "mlx_dequantize",
        )?;
        Ok(output)
    }
}

pub(crate) struct VectorArray {
    pub(crate) raw: sys::mlx_vector_array,
    _thread_affine: PhantomData<Rc<()>>,
}

impl VectorArray {
    pub(crate) fn new() -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_vector_array_new() };
        check_infallible("mlx_vector_array_new", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub(crate) fn from_arrays(arrays: &[Array]) -> Result<Self> {
        initialize();
        clear_error();
        let raw_values: Vec<sys::mlx_array> = arrays.iter().map(|array| array.raw).collect();
        let raw = unsafe { sys::mlx_vector_array_new_data(raw_values.as_ptr(), raw_values.len()) };
        check_infallible("mlx_vector_array_new_data", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub(crate) fn from_array(array: &Array) -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_vector_array_new_value(array.raw) };
        check_infallible("mlx_vector_array_new_value", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    fn len(&self) -> usize {
        unsafe { sys::mlx_vector_array_size(self.raw) }
    }

    pub(crate) fn get(&self, index: usize) -> Result<Array> {
        let mut output = Array::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_vector_array_get(&mut output.raw, self.raw, index) },
            "mlx_vector_array_get",
        )?;
        Ok(output)
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_vector_array_free(self.raw);
        }
        clear_error();
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut output = Self::empty().expect("create MLX array clone");
        let mut raw = output.raw;
        clear_error();
        check(
            unsafe { sys::mlx_array_set(&mut raw, self.raw) },
            "mlx_array_set",
        )
        .expect("clone MLX array");
        output.raw = raw;
        output
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_array_free(self.raw);
        }
        clear_error();
    }
}
