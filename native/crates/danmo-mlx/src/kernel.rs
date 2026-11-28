//! Run-time compiled Metal kernels (`mx.fast.metal_kernel`).
//!
//! mlx-c exposes `mx.fast.metal_kernel` but not `mx.Primitive`, so this is the
//! only custom-kernel path available through the C API: MLX writes the kernel
//! signature, compiles the Metal source once per specialization, and records a
//! `CustomKernel` primitive into the lazy graph. A true `eval_gpu` override,
//! in-place output, or custom autodiff needs a C++ bridge into MLX internals and
//! is out of scope here.
//!
//! Kernels are created on the executor thread and stay there. `MetalKernel` is
//! thread-affine for the same reason `Array` is: MLX binds a compiled library to
//! the device that compiled it.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::array::{Array, Dtype, VectorArray};
use crate::error::{check, check_infallible, clear_error, initialize, Error, Result};
use crate::stream::Stream;

/// A template argument. MLX turns these into a `template <...>` prefix on the
/// generated kernel and into the specialization part of its library name, so a
/// kernel compiled for one set of values is not reused for another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}

/// Output shape and dtype. The kernel body writes into a buffer with this exact
/// shape; MLX does not infer it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSpec {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
}

impl OutputSpec {
    pub fn new(shape: &[i32], dtype: Dtype) -> Self {
        Self {
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// A float32 output of the given shape.
    pub fn f32(shape: &[i32]) -> Self {
        Self::new(shape, Dtype::Float32)
    }

    /// An output whose shape and dtype match `array`.
    pub fn like(array: &Array) -> Self {
        Self {
            shape: array.shape(),
            dtype: array.dtype(),
        }
    }
}

/// Grid and threadgroup size for one launch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Launch {
    pub grid: (i32, i32, i32),
    pub threadgroup: (i32, i32, i32),
    pub init_value: Option<f32>,
}

impl Launch {
    pub fn new(grid: (i32, i32, i32), threadgroup: (i32, i32, i32)) -> Self {
        Self {
            grid,
            threadgroup,
            init_value: None,
        }
    }

    /// Zero the outputs before the kernel runs, for kernels that accumulate
    /// with atomics instead of overwriting.
    pub fn with_init_value(mut self, value: f32) -> Self {
        self.init_value = Some(value);
        self
    }

    fn validate(&self) -> Result<()> {
        let (gx, gy, gz) = self.grid;
        let (tx, ty, tz) = self.threadgroup;
        if gx <= 0 || gy <= 0 || gz <= 0 {
            return Err(Error::new(
                "MetalKernel::apply",
                format!("grid {gx}x{gy}x{gz} must be positive in every dimension"),
            ));
        }
        if tx <= 0 || ty <= 0 || tz <= 0 {
            return Err(Error::new(
                "MetalKernel::apply",
                format!("threadgroup {tx}x{ty}x{tz} must be positive in every dimension"),
            ));
        }
        Ok(())
    }
}

/// Arguments for one `apply` call: output buffers, launch geometry, and
/// template values.
pub struct Config {
    config: sys::mlx_fast_metal_kernel_config,
}

impl Config {
    pub fn new(outputs: &[OutputSpec], launch: Launch) -> Result<Self> {
        launch.validate()?;
        if outputs.is_empty() {
            return Err(Error::new(
                "MetalKernel::Config",
                "at least one output is required",
            ));
        }
        initialize();
        clear_error();
        let config = check_infallible("mlx_fast_metal_kernel_config_new", unsafe {
            sys::mlx_fast_metal_kernel_config_new()
        })?;
        if config.ctx.is_null() {
            return Err(Error::new(
                "mlx_fast_metal_kernel_config_new",
                "MLX returned an empty kernel config",
            ));
        }
        let this = Self { config };
        for output in outputs {
            if output.shape.iter().any(|&dimension| dimension < 0) {
                return Err(Error::new(
                    "MetalKernel::Config",
                    format!("output shape {:?} has a negative dimension", output.shape),
                ));
            }
            clear_error();
            check(
                unsafe {
                    sys::mlx_fast_metal_kernel_config_add_output_arg(
                        this.config,
                        output.shape.as_ptr(),
                        output.shape.len(),
                        output.dtype.raw(),
                    )
                },
                "mlx_fast_metal_kernel_config_add_output_arg",
            )?;
        }
        clear_error();
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_config_set_grid(
                    this.config,
                    launch.grid.0,
                    launch.grid.1,
                    launch.grid.2,
                )
            },
            "mlx_fast_metal_kernel_config_set_grid",
        )?;
        clear_error();
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_config_set_thread_group(
                    this.config,
                    launch.threadgroup.0,
                    launch.threadgroup.1,
                    launch.threadgroup.2,
                )
            },
            "mlx_fast_metal_kernel_config_set_thread_group",
        )?;
        if let Some(value) = launch.init_value {
            clear_error();
            check(
                unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(this.config, value) },
                "mlx_fast_metal_kernel_config_set_init_value",
            )?;
        }
        Ok(this)
    }

    /// Add a template argument named `T`.
    pub fn template_arg(self, arg: TemplateArg) -> Result<Self> {
        self.template_arg_named("T", arg)
    }

    pub fn template_arg_named(self, name: &str, arg: TemplateArg) -> Result<Self> {
        let name_c = CString::new(name).map_err(|_| {
            Error::new(
                "MetalKernel::Config",
                "template argument name contains a null byte",
            )
        })?;
        clear_error();
        let status = unsafe {
            match arg {
                TemplateArg::Int(value) => sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    self.config,
                    name_c.as_ptr(),
                    value,
                ),
                TemplateArg::Bool(value) => {
                    sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                        self.config,
                        name_c.as_ptr(),
                        value,
                    )
                }
                TemplateArg::Dtype(dtype) => {
                    sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        self.config,
                        name_c.as_ptr(),
                        dtype.raw(),
                    )
                }
            }
        };
        check(status, "mlx_fast_metal_kernel_config_add_template_arg")?;
        Ok(self)
    }

    /// Print the generated Metal source to stdout. Diagnostic use only: the
    /// output is thousands of lines and goes to the process stdout, not a log.
    pub fn verbose(self, verbose: bool) -> Result<Self> {
        clear_error();
        check(
            unsafe { sys::mlx_fast_metal_kernel_config_set_verbose(self.config, verbose) },
            "mlx_fast_metal_kernel_config_set_verbose",
        )?;
        Ok(self)
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            sys::mlx_fast_metal_kernel_config_free(self.config);
        }
        clear_error();
    }
}

impl std::fmt::Debug for Config {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The raw handle is opaque; identity is not useful in a log line.
        formatter.write_str("KernelConfig")
    }
}

/// A compiled Metal kernel body. Create once at model load, keep for the life of
/// the model, and apply per step.
///
/// `source` is the *inside* of the function. MLX writes the signature from
/// `input_names`, `output_names`, the input dtypes, and any [`TemplateArg`].
/// Mentioning `<input>_shape`, `<input>_strides`, or `<input>_ndim` in the body
/// makes MLX add that parameter automatically.
pub struct MetalKernel {
    raw: sys::mlx_fast_metal_kernel,
    name: String,
    output_count: usize,
    _thread_affine: PhantomData<Rc<()>>,
}

impl MetalKernel {
    /// Row-contiguous inputs are enforced, and outputs are plain buffers.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
    ) -> Result<Self> {
        Self::with_options(name, input_names, output_names, source, header, true, false)
    }

    /// `ensure_row_contiguous`: MLX copies a non-contiguous input first. Leave it
    /// on unless the kernel body handles strides itself.
    ///
    /// `atomic_outputs`: outputs are declared `atomic<T>*`; pair with
    /// [`Launch::with_init_value`].
    pub fn with_options(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(Error::new("MetalKernel::new", "kernel name is empty"));
        }
        if output_names.is_empty() {
            return Err(Error::new(
                "MetalKernel::new",
                "at least one output name is required",
            ));
        }
        if source.trim().is_empty() {
            return Err(Error::new("MetalKernel::new", "kernel source is empty"));
        }
        let name_c = CString::new(name)
            .map_err(|_| Error::new("MetalKernel::new", "name has a null byte"))?;
        let source_c = CString::new(source)
            .map_err(|_| Error::new("MetalKernel::new", "source has a null byte"))?;
        let header_c = CString::new(header)
            .map_err(|_| Error::new("MetalKernel::new", "header has a null byte"))?;
        let inputs = CStringList::new(input_names)?;
        let outputs = CStringList::new(output_names)?;

        initialize();
        clear_error();
        let raw = check_infallible("mlx_fast_metal_kernel_new", unsafe {
            sys::mlx_fast_metal_kernel_new(
                name_c.as_ptr(),
                inputs.raw,
                outputs.raw,
                source_c.as_ptr(),
                header_c.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        })?;
        if raw.ctx.is_null() {
            return Err(Error::new(
                "mlx_fast_metal_kernel_new",
                format!("Metal kernel {name} was rejected before compilation"),
            ));
        }
        Ok(Self {
            raw,
            name: name.to_string(),
            output_count: output_names.len(),
            _thread_affine: PhantomData,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Record this kernel into the graph. The returned arrays are lazy until
    /// they are evaluated like any other MLX array.
    pub fn apply(&self, inputs: &[Array], config: &Config, stream: &Stream) -> Result<Vec<Array>> {
        let inputs = VectorArray::from_arrays(inputs)?;
        let mut outputs = VectorArray::new()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_apply(
                    &mut outputs.raw,
                    self.raw,
                    inputs.raw,
                    config.config,
                    stream.raw,
                )
            },
            "mlx_fast_metal_kernel_apply",
        )?;
        let count = outputs.len();
        if count != self.output_count {
            return Err(Error::new(
                "MetalKernel::apply",
                format!(
                    "kernel {} declares {} outputs but produced {count}",
                    self.name, self.output_count
                ),
            ));
        }
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            values.push(outputs.get(index)?);
        }
        Ok(values)
    }

    /// Apply and take the single output. Fails when the kernel declares more
    /// than one output, so a mismatched spec cannot silently drop a tensor.
    pub fn apply_one(&self, inputs: &[Array], config: &Config, stream: &Stream) -> Result<Array> {
        let mut outputs = self.apply(inputs, config, stream)?;
        if outputs.len() != 1 {
            return Err(Error::new(
                "MetalKernel::apply_one",
                format!(
                    "kernel {} declares {} outputs; apply_one needs exactly one",
                    self.name,
                    outputs.len()
                ),
            ));
        }
        Ok(outputs.remove(0))
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            sys::mlx_fast_metal_kernel_free(self.raw);
        }
        clear_error();
    }
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalKernel")
            .field("name", &self.name)
            .field("outputs", &self.output_count)
            .finish()
    }
}

/// A `mlx_vector_string` that borrows the `CString`s it was built from.
struct CStringList {
    owned: Vec<CString>,
    raw: sys::mlx_vector_string,
}

impl CStringList {
    fn new(names: &[&str]) -> Result<Self> {
        let owned: Vec<CString> = names
            .iter()
            .map(|name| {
                CString::new(*name)
                    .map_err(|_| Error::new("MetalKernel::new", format!("{name} has a null byte")))
            })
            .collect::<Result<_>>()?;
        let mut pointers: Vec<*const c_char> = owned.iter().map(|name| name.as_ptr()).collect();
        clear_error();
        let raw = check_infallible("mlx_vector_string_new_data", unsafe {
            sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len())
        })?;
        // Do not read `.ctx`: bindgen may emit `mlx_vector_string` as an opaque
        // `{ _address: u8 }` blob (no `ctx` field) depending on libclang, while
        // neighbouring mlx_* handles keep an explicit pointer field. Validate
        // through the size API instead.
        let size = unsafe { sys::mlx_vector_string_size(raw) };
        if size != owned.len() {
            return Err(Error::new(
                "mlx_vector_string_new_data",
                format!(
                    "MLX string vector has {size} entries after inserting {}",
                    owned.len()
                ),
            ));
        }
        Ok(Self { owned, raw })
    }
}

impl Drop for CStringList {
    fn drop(&mut self) {
        let _ = self.owned.len();
        clear_error();
        unsafe {
            let _ = sys::mlx_vector_string_free(self.raw);
        }
        clear_error();
    }
}
