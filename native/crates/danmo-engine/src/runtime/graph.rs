//! Graph construction. Raw ggml operators are called from this file.

use super::{RuntimeError, Tensor, TYPE_F16};
use danmo_ggml_sys as g;
use std::ffi::CString;

pub struct GraphBuilder {
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    ctx: *mut g::ggml_context,
    inputs: Vec<NamedTensor>,
}

pub struct NamedTensor {
    pub name: String,
    pub tensor: *mut g::ggml_tensor,
}

impl GraphBuilder {
    pub(crate) fn new(
        backend: g::ggml_backend_t,
        buft: g::ggml_backend_buffer_type_t,
        mem_size: usize,
    ) -> Result<Self, RuntimeError> {
        let params = g::ggml_init_params {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { g::ggml_init(params) };
        if ctx.is_null() {
            return Err(RuntimeError::Init);
        }
        Ok(Self {
            backend,
            buft,
            ctx,
            inputs: Vec::new(),
        })
    }

    pub fn f32_2d(&mut self, name: &str, ne0: i64, ne1: i64, input: bool) -> *mut g::ggml_tensor {
        self.tensor_2d(name, g::ggml_type_GGML_TYPE_F32, ne0, ne1, input)
    }

    pub fn tensor_2d(
        &mut self,
        name: &str,
        ggml_type: u32,
        ne0: i64,
        ne1: i64,
        input: bool,
    ) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_2d(self.ctx, ggml_type, ne0, ne1);
            name_tensor(t, name);
            if input {
                g::ggml_set_input(t);
                self.inputs.push(NamedTensor {
                    name: name.to_string(),
                    tensor: t,
                });
            }
            t
        }
    }

    pub fn f32_1d(&mut self, name: &str, ne0: i64, input: bool) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_1d(self.ctx, g::ggml_type_GGML_TYPE_F32, ne0);
            name_tensor(t, name);
            if input {
                g::ggml_set_input(t);
                self.inputs.push(NamedTensor {
                    name: name.to_string(),
                    tensor: t,
                });
            }
            t
        }
    }

    pub fn mul_mat(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_mul_mat(self.ctx, a, b) }
    }

    pub fn add(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_add(self.ctx, a, b) }
    }

    pub fn rms_norm(&self, a: *mut g::ggml_tensor, eps: f32) -> *mut g::ggml_tensor {
        unsafe { g::ggml_rms_norm(self.ctx, a, eps) }
    }

    pub fn silu(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_silu(self.ctx, a) }
    }

    pub fn mul(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_mul(self.ctx, a, b) }
    }

    pub fn scale(&self, a: *mut g::ggml_tensor, s: f32) -> *mut g::ggml_tensor {
        unsafe { g::ggml_scale(self.ctx, a, s) }
    }

    pub fn soft_max(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_soft_max(self.ctx, a) }
    }

    pub fn norm(&self, a: *mut g::ggml_tensor, eps: f32) -> *mut g::ggml_tensor {
        unsafe { g::ggml_norm(self.ctx, a, eps) }
    }

    pub fn swiglu(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_swiglu(self.ctx, a) }
    }

    pub fn concat(
        &self,
        a: *mut g::ggml_tensor,
        b: *mut g::ggml_tensor,
        dim: i32,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_concat(self.ctx, a, b, dim) }
    }

    pub fn transpose(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_transpose(self.ctx, a) }
    }

    pub fn cont(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_cont(self.ctx, a) }
    }

    pub fn reshape_2d(&self, a: *mut g::ggml_tensor, ne0: i64, ne1: i64) -> *mut g::ggml_tensor {
        unsafe { g::ggml_reshape_2d(self.ctx, a, ne0, ne1) }
    }

    pub fn reshape_3d(
        &self,
        a: *mut g::ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_reshape_3d(self.ctx, a, ne0, ne1, ne2) }
    }

    pub fn reshape_4d(
        &self,
        a: *mut g::ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_reshape_4d(self.ctx, a, ne0, ne1, ne2, ne3) }
    }

    pub fn permute(
        &self,
        a: *mut g::ggml_tensor,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_permute(self.ctx, a, axis0, axis1, axis2, axis3) }
    }

    pub fn sub(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_sub(self.ctx, a, b) }
    }

    pub fn view_3d(
        &self,
        a: *mut g::ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_view_3d(self.ctx, a, ne0, ne1, ne2, nb1, nb2, offset) }
    }

    pub fn f32_3d(
        &mut self,
        name: &str,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        input: bool,
    ) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_3d(self.ctx, g::ggml_type_GGML_TYPE_F32, ne0, ne1, ne2);
            name_tensor(t, name);
            if input {
                g::ggml_set_input(t);
                self.inputs.push(NamedTensor {
                    name: name.to_string(),
                    tensor: t,
                });
            }
            t
        }
    }

    pub fn view_2d(
        &self,
        a: *mut g::ggml_tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_view_2d(self.ctx, a, ne0, ne1, nb1, offset) }
    }

    pub fn get_rows(
        &self,
        table: *mut g::ggml_tensor,
        rows: *mut g::ggml_tensor,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_get_rows(self.ctx, table, rows) }
    }

    pub fn repeat(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_repeat(self.ctx, a, b) }
    }

    pub fn soft_max_ext(
        &self,
        a: *mut g::ggml_tensor,
        mask: *mut g::ggml_tensor,
        scale: f32,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_soft_max_ext(self.ctx, a, mask, scale, 0.0) }
    }

    /// Interleaved-pair RoPE. `pos` is I32 `[n_token]`. `freq_base` is 2000 for Flux2.
    pub fn rope(
        &self,
        a: *mut g::ggml_tensor,
        pos: *mut g::ggml_tensor,
        n_dims: i32,
        freq_base: f32,
    ) -> *mut g::ggml_tensor {
        unsafe {
            g::ggml_rope_ext(
                self.ctx,
                a,
                pos,
                std::ptr::null_mut(),
                n_dims,
                0,
                0,
                freq_base,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        }
    }

    /// Half-split RoPE used by Qwen3. `freq_base` is 1_000_000.
    pub fn rope_neox(
        &self,
        a: *mut g::ggml_tensor,
        pos: *mut g::ggml_tensor,
        n_dims: i32,
        freq_base: f32,
    ) -> *mut g::ggml_tensor {
        unsafe {
            g::ggml_rope_ext(
                self.ctx,
                a,
                pos,
                std::ptr::null_mut(),
                n_dims,
                g::GGML_ROPE_TYPE_NEOX as i32,
                0,
                freq_base,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        }
    }

    /// Fused attention. `q` is `[n_embd, n_batch, n_head]`, `k` and `v` are
    /// `[n_embd, n_kv, n_head_kv]` (v is not transposed). `mask` is F16 or null.
    pub fn flash_attn(&self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: f32) -> Tensor {
        unsafe { g::ggml_flash_attn_ext(self.ctx, q, k, v, mask, scale, 0.0, 0.0) }
    }

    pub fn cast_f16(&self, a: Tensor) -> Tensor {
        unsafe { g::ggml_cast(self.ctx, a, TYPE_F16) }
    }

    /// Direct 2D convolution. Kernel is `[KW, KH, IC, OC]`, data is `[W, H, C, N]`.
    /// This is ggml's direct conv, not the im2col plus matmul path.
    pub fn conv2d_direct(
        &self,
        kernel: Tensor,
        data: Tensor,
        stride0: i32,
        stride1: i32,
        pad0: i32,
        pad1: i32,
        dil0: i32,
        dil1: i32,
    ) -> Tensor {
        unsafe {
            g::ggml_conv_2d_direct(
                self.ctx, kernel, data, stride0, stride1, pad0, pad1, dil0, dil1,
            )
        }
    }

    pub fn i32_1d(&mut self, name: &str, ne0: i64, input: bool) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_1d(self.ctx, g::ggml_type_GGML_TYPE_I32, ne0);
            name_tensor(t, name);
            if input {
                g::ggml_set_input(t);
                self.inputs.push(NamedTensor {
                    name: name.to_string(),
                    tensor: t,
                });
            }
            t
        }
    }

    /// Finish the graph. `reserve` + `alloc` happen once; later steps only write inputs.
    pub fn finish(self, output: *mut g::ggml_tensor) -> Result<ReusedGraph, RuntimeError> {
        unsafe {
            g::ggml_set_output(output);
            let graph = g::ggml_new_graph_custom(self.ctx, 8192, false);
            if graph.is_null() {
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Init);
            }
            g::ggml_build_forward_expand(graph, output);
            let galloc = g::ggml_gallocr_new(self.buft);
            if galloc.is_null() {
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Init);
            }
            if !g::ggml_gallocr_reserve(galloc, graph) {
                g::ggml_gallocr_free(galloc);
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Reserve);
            }
            if !g::ggml_gallocr_alloc_graph(galloc, graph) {
                g::ggml_gallocr_free(galloc);
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Reserve);
            }
            Ok(ReusedGraph {
                ctx: self.ctx,
                graph,
                galloc,
                backend: self.backend,
                buft: self.buft,
                inputs: self.inputs,
                output,
                computes: 0,
                readbacks: 0,
            })
        }
    }
}

pub struct ReusedGraph {
    ctx: *mut g::ggml_context,
    graph: *mut g::ggml_cgraph,
    galloc: g::ggml_gallocr_t,
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    inputs: Vec<NamedTensor>,
    output: *mut g::ggml_tensor,
    computes: u32,
    readbacks: u32,
}

impl ReusedGraph {
    pub fn compute_count(&self) -> u32 {
        self.computes
    }

    pub fn host_readbacks(&self) -> u32 {
        self.readbacks
    }

    pub fn output(&self) -> Tensor {
        self.output
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor, RuntimeError> {
        self.inputs
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.tensor)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))
    }

    /// Another graph on the same ggml device. Used for the step that stays resident.
    pub fn child(&self, mem_size: usize) -> Result<GraphBuilder, RuntimeError> {
        GraphBuilder::new(self.backend, self.buft, mem_size)
    }

    pub fn set_bytes(&self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        let tensor = self
            .inputs
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.tensor)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))?;
        let nbytes = unsafe { g::ggml_nbytes(tensor) };
        if nbytes != data.len() {
            return Err(RuntimeError::SizeMismatch {
                name: name.to_string(),
                expected: nbytes,
                got: data.len(),
            });
        }
        unsafe {
            g::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, nbytes);
        }
        Ok(())
    }

    pub fn set_f32(&self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        let tensor = self
            .inputs
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.tensor)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))?;
        let nbytes = unsafe { g::ggml_nbytes(tensor) };
        let got = data.len() * std::mem::size_of::<f32>();
        if nbytes != got {
            return Err(RuntimeError::SizeMismatch {
                name: name.to_string(),
                expected: nbytes,
                got,
            });
        }
        unsafe {
            g::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, nbytes);
        }
        Ok(())
    }

    pub fn set_i32(&self, name: &str, data: &[i32]) -> Result<(), RuntimeError> {
        let tensor = self
            .inputs
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.tensor)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))?;
        let nbytes = unsafe { g::ggml_nbytes(tensor) };
        let got = data.len() * std::mem::size_of::<i32>();
        if nbytes != got {
            return Err(RuntimeError::SizeMismatch {
                name: name.to_string(),
                expected: nbytes,
                got,
            });
        }
        unsafe {
            g::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, nbytes);
        }
        Ok(())
    }

    /// Recompute the reserved graph and leave the output on the device.
    pub fn compute_device(&mut self) -> Result<(), RuntimeError> {
        let status = unsafe { g::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != g::ggml_status_GGML_STATUS_SUCCESS {
            return Err(RuntimeError::Compute(status));
        }
        self.computes += 1;
        Ok(())
    }

    /// Copy `src` onto `dst` on the ggml backend. Both tensors stay on device.
    pub fn copy(src: Tensor, dst: Tensor) -> Result<(), RuntimeError> {
        if src.is_null() || dst.is_null() {
            return Err(RuntimeError::Init);
        }
        let expected = unsafe { g::ggml_nbytes(dst) };
        let got = unsafe { g::ggml_nbytes(src) };
        if expected != got {
            return Err(RuntimeError::SizeMismatch {
                name: "tensor copy".into(),
                expected,
                got,
            });
        }
        unsafe { g::ggml_backend_tensor_copy(src, dst) };
        Ok(())
    }

    pub fn read_output(&mut self) -> Result<Vec<f32>, RuntimeError> {
        self.readbacks += 1;
        let nbytes = unsafe { g::ggml_nbytes(self.output) };
        let mut out = vec![0f32; nbytes / std::mem::size_of::<f32>()];
        unsafe {
            g::ggml_backend_tensor_get(self.output, out.as_mut_ptr().cast(), 0, nbytes);
        }
        Ok(out)
    }

    /// Recompute the reserved graph and read the output back to the host.
    pub fn compute(&mut self) -> Result<Vec<f32>, RuntimeError> {
        self.compute_device()?;
        self.read_output()
    }
}

pub fn shape(tensor: Tensor) -> [i64; 4] {
    if tensor.is_null() {
        return [0, 0, 0, 0];
    }
    unsafe {
        [
            (*tensor).ne[0],
            (*tensor).ne[1],
            (*tensor).ne[2],
            (*tensor).ne[3],
        ]
    }
}

impl Drop for ReusedGraph {
    fn drop(&mut self) {
        unsafe {
            g::ggml_gallocr_free(self.galloc);
            g::ggml_free(self.ctx);
        }
    }
}

unsafe fn name_tensor(tensor: *mut g::ggml_tensor, name: &str) {
    if tensor.is_null() {
        return;
    }
    let c = CString::new(name).unwrap_or_else(|_| CString::new("tensor").unwrap());
    g::ggml_set_name(tensor, c.as_ptr());
}
