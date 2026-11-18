use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::array::VectorArray;
use crate::error::{check, check_infallible, clear_error, initialize, set_error};
use crate::{Array, Error, Result, Stream};

struct UnaryPayload<F> {
    function: F,
    stream: Stream,
}

pub struct UnaryClosure {
    raw: sys::mlx_closure,
    _thread_affine: PhantomData<Rc<()>>,
}

impl UnaryClosure {
    pub fn new<F>(stream: Stream, function: F) -> Result<Self>
    where
        F: Fn(&Array, &Stream) -> Result<Array> + 'static,
    {
        initialize();
        clear_error();
        let payload = Box::into_raw(Box::new(UnaryPayload { function, stream }));
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(call_unary::<F>),
                payload.cast(),
                Some(drop_payload::<F>),
            )
        };
        check_infallible("mlx_closure_new_func_payload", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    fn empty() -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_closure_new() };
        check_infallible("mlx_closure_new", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn compile(self, shapeless: bool) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_compile(&mut output.raw, self.raw, shapeless) },
            "mlx_compile",
        )?;
        Ok(output)
    }

    pub fn apply(&self, input: &Array) -> Result<Array> {
        let input = VectorArray::from_array(input)?;
        let mut output = VectorArray::new()?;
        clear_error();
        check(
            unsafe { sys::mlx_closure_apply(&mut output.raw, self.raw, input.raw) },
            "mlx_closure_apply",
        )?;
        let size = unsafe { sys::mlx_vector_array_size(output.raw) };
        if size != 1 {
            return Err(Error::new(
                "mlx_closure_apply",
                format!("unary closure returned {size} arrays"),
            ));
        }
        output.get(0)
    }
}

impl Drop for UnaryClosure {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_closure_free(self.raw);
        }
        clear_error();
    }
}

unsafe extern "C" fn call_unary<F>(
    result: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: Fn(&Array, &Stream) -> Result<Array> + 'static,
{
    if result.is_null() || payload.is_null() {
        set_error("unary closure received a null argument".into());
        return 1;
    }
    let payload = unsafe { &*(payload.cast::<UnaryPayload<F>>()) };
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let size = unsafe { sys::mlx_vector_array_size(input) };
        if size != 1 {
            return Err(Error::new(
                "unary closure",
                format!("expected one input, got {size}"),
            ));
        }
        let mut source = Array::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_vector_array_get(&mut source.raw, input, 0) },
            "mlx_vector_array_get",
        )?;
        let output = (payload.function)(&source, &payload.stream)?;
        clear_error();
        check(
            unsafe { sys::mlx_vector_array_set_value(result, output.raw) },
            "mlx_vector_array_set_value",
        )?;
        Ok(())
    }));
    match outcome {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error.to_string());
            1
        }
        Err(_) => {
            set_error("unary closure panicked".into());
            1
        }
    }
}

unsafe extern "C" fn drop_payload<F>(payload: *mut c_void)
where
    F: Fn(&Array, &Stream) -> Result<Array> + 'static,
{
    if payload.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(payload.cast::<UnaryPayload<F>>()) });
    }));
}

struct VariadicPayload<F> {
    function: F,
    stream: Stream,
}

/// A closure over any fixed number of arrays, the form `mlx_compile` needs for
/// real model functions: captured weights stay outside the traced inputs, and
/// only the per-call activations (latent, conditioning, timestep) are passed
/// through `apply`.
pub struct Closure {
    raw: sys::mlx_closure,
    _thread_affine: PhantomData<Rc<()>>,
}

impl Closure {
    pub(crate) fn raw(&self) -> sys::mlx_closure {
        self.raw
    }

    pub fn new<F>(stream: Stream, function: F) -> Result<Self>
    where
        F: Fn(&[Array], &Stream) -> Result<Array> + 'static,
    {
        initialize();
        clear_error();
        let payload = Box::into_raw(Box::new(VariadicPayload { function, stream }));
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(call_variadic::<F>),
                payload.cast(),
                Some(drop_variadic_payload::<F>),
            )
        };
        check_infallible("mlx_closure_new_func_payload", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    fn empty() -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_closure_new() };
        check_infallible("mlx_closure_new", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn compile(self, shapeless: bool) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_compile(&mut output.raw, self.raw, shapeless) },
            "mlx_compile",
        )?;
        Ok(output)
    }

    pub fn apply(&self, inputs: &[Array]) -> Result<Array> {
        let input = VectorArray::from_arrays(inputs)?;
        let mut output = VectorArray::new()?;
        clear_error();
        check(
            unsafe { sys::mlx_closure_apply(&mut output.raw, self.raw, input.raw) },
            "mlx_closure_apply",
        )?;
        let size = unsafe { sys::mlx_vector_array_size(output.raw) };
        if size != 1 {
            return Err(Error::new(
                "mlx_closure_apply",
                format!("closure returned {size} arrays"),
            ));
        }
        output.get(0)
    }
}

impl Drop for Closure {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_closure_free(self.raw);
        }
        clear_error();
    }
}

struct MultiPayload<F> {
    function: F,
    stream: Stream,
}

/// A closure returning several arrays (`mx.compile` of a function with a
/// tuple output). Same FFI/panic boundary as `Closure`; `apply` returns every
/// output in order.
pub struct MultiClosure {
    raw: sys::mlx_closure,
    _thread_affine: PhantomData<Rc<()>>,
}

impl MultiClosure {
    pub fn new<F>(stream: Stream, function: F) -> Result<Self>
    where
        F: Fn(&[Array], &Stream) -> Result<Vec<Array>> + 'static,
    {
        initialize();
        clear_error();
        let payload = Box::into_raw(Box::new(MultiPayload { function, stream }));
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(call_multi::<F>),
                payload.cast(),
                Some(drop_multi_payload::<F>),
            )
        };
        check_infallible("mlx_closure_new_func_payload", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    fn empty() -> Result<Self> {
        initialize();
        clear_error();
        let raw = unsafe { sys::mlx_closure_new() };
        check_infallible("mlx_closure_new", raw).map(|raw| Self {
            raw,
            _thread_affine: PhantomData,
        })
    }

    pub fn compile(self, shapeless: bool) -> Result<Self> {
        let mut output = Self::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_compile(&mut output.raw, self.raw, shapeless) },
            "mlx_compile",
        )?;
        Ok(output)
    }

    pub fn apply(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let input = VectorArray::from_arrays(inputs)?;
        let mut output = VectorArray::new()?;
        clear_error();
        check(
            unsafe { sys::mlx_closure_apply(&mut output.raw, self.raw, input.raw) },
            "mlx_closure_apply",
        )?;
        let size = unsafe { sys::mlx_vector_array_size(output.raw) };
        let mut arrays = Vec::with_capacity(size);
        for index in 0..size {
            arrays.push(output.get(index)?);
        }
        Ok(arrays)
    }
}

impl Drop for MultiClosure {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_closure_free(self.raw);
        }
        clear_error();
    }
}

unsafe extern "C" fn call_multi<F>(
    result: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: Fn(&[Array], &Stream) -> Result<Vec<Array>> + 'static,
{
    if result.is_null() || payload.is_null() {
        set_error("multi closure received a null argument".into());
        return 1;
    }
    let payload = unsafe { &*(payload.cast::<MultiPayload<F>>()) };
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let size = unsafe { sys::mlx_vector_array_size(input) };
        let mut sources = Vec::with_capacity(size);
        for index in 0..size {
            let mut source = Array::empty()?;
            clear_error();
            check(
                unsafe { sys::mlx_vector_array_get(&mut source.raw, input, index) },
                "mlx_vector_array_get",
            )?;
            sources.push(source);
        }
        let outputs = (payload.function)(&sources, &payload.stream)?;
        if outputs.is_empty() {
            return Err(Error::new("multi closure", "returned no outputs"));
        }
        let values = VectorArray::from_arrays(&outputs)?;
        clear_error();
        check(
            unsafe { sys::mlx_vector_array_set(result, values.raw) },
            "mlx_vector_array_set",
        )?;
        Ok(())
    }));
    match outcome {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error.to_string());
            1
        }
        Err(_) => {
            set_error("multi closure panicked".into());
            1
        }
    }
}

unsafe extern "C" fn drop_multi_payload<F>(payload: *mut c_void)
where
    F: Fn(&[Array], &Stream) -> Result<Vec<Array>> + 'static,
{
    if payload.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(payload.cast::<MultiPayload<F>>()) });
    }));
}

unsafe extern "C" fn call_variadic<F>(
    result: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: Fn(&[Array], &Stream) -> Result<Array> + 'static,
{
    if result.is_null() || payload.is_null() {
        set_error("closure received a null argument".into());
        return 1;
    }
    let payload = unsafe { &*(payload.cast::<VariadicPayload<F>>()) };
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let size = unsafe { sys::mlx_vector_array_size(input) };
        let mut sources = Vec::with_capacity(size);
        for index in 0..size {
            let mut source = Array::empty()?;
            clear_error();
            check(
                unsafe { sys::mlx_vector_array_get(&mut source.raw, input, index) },
                "mlx_vector_array_get",
            )?;
            sources.push(source);
        }
        let output = (payload.function)(&sources, &payload.stream)?;
        clear_error();
        check(
            unsafe { sys::mlx_vector_array_set_value(result, output.raw) },
            "mlx_vector_array_set_value",
        )?;
        Ok(())
    }));
    match outcome {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error.to_string());
            1
        }
        Err(_) => {
            set_error("closure panicked".into());
            1
        }
    }
}

unsafe extern "C" fn drop_variadic_payload<F>(payload: *mut c_void)
where
    F: Fn(&[Array], &Stream) -> Result<Array> + 'static,
{
    if payload.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(payload.cast::<VariadicPayload<F>>()) });
    }));
}
