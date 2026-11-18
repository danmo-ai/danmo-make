use std::marker::PhantomData;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::array::VectorArray;
use crate::compile::Closure;
use crate::error::{check, check_infallible, clear_error, initialize, Error, Result};
use crate::Array;

/// Differentiable transform of a [`Closure`]: returns primal outputs and
/// gradients w.r.t. the selected argument indices.
pub struct ValueAndGrad {
    raw: sys::mlx_closure_value_and_grad,
    /// Keep the source closure alive for the transform's lifetime.
    _closure: Closure,
    _thread_affine: PhantomData<Rc<()>>,
}

impl ValueAndGrad {
    /// Build `value_and_grad(fun, argnums)` for a scalar-valued closure.
    ///
    /// `argnums` indexes into the closure inputs that receive gradients.
    pub fn new(closure: Closure, argnums: &[i32]) -> Result<Self> {
        if argnums.is_empty() {
            return Err(Error::new(
                "ValueAndGrad::new",
                "argnums must contain at least one index",
            ));
        }
        initialize();
        clear_error();
        let mut raw = check_infallible(
            "mlx_closure_value_and_grad_new",
            unsafe { sys::mlx_closure_value_and_grad_new() },
        )?;
        clear_error();
        check(
            unsafe {
                sys::mlx_value_and_grad(&mut raw, closure.raw(), argnums.as_ptr(), argnums.len())
            },
            "mlx_value_and_grad",
        )?;
        Ok(Self {
            raw,
            _closure: closure,
            _thread_affine: PhantomData,
        })
    }

    /// Apply the transform. Returns `(outputs, grads)` where `grads[i]` is the
    /// gradient of `outputs[0]` w.r.t. `inputs[argnums[i]]`.
    pub fn apply(&self, inputs: &[Array]) -> Result<(Vec<Array>, Vec<Array>)> {
        let input = VectorArray::from_arrays(inputs)?;
        let mut values = VectorArray::new()?;
        let mut grads = VectorArray::new()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_closure_value_and_grad_apply(
                    &mut values.raw,
                    &mut grads.raw,
                    self.raw,
                    input.raw,
                )
            },
            "mlx_closure_value_and_grad_apply",
        )?;
        let value_count = unsafe { sys::mlx_vector_array_size(values.raw) };
        let grad_count = unsafe { sys::mlx_vector_array_size(grads.raw) };
        let mut value_out = Vec::with_capacity(value_count);
        for index in 0..value_count {
            value_out.push(values.get(index)?);
        }
        let mut grad_out = Vec::with_capacity(grad_count);
        for index in 0..grad_count {
            grad_out.push(grads.get(index)?);
        }
        Ok((value_out, grad_out))
    }
}

impl Drop for ValueAndGrad {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_closure_value_and_grad_free(self.raw);
        }
        clear_error();
    }
}
