//! Graph construction. Raw ggml operators are called from this file.

use super::{RuntimeError, Tensor, TYPE_F16};
use danmo_ggml_sys as g;
use std::collections::HashMap;
use std::ffi::CString;

struct SideResidual {
    a: Tensor,
    b: Tensor,
    scale: f32,
}

/// One reserved graph. 20480 matches SD.cpp `QWEN_IMAGE_GRAPH_SIZE`.
const GRAPH_NODES: usize = 20480;

pub struct GraphBuilder {
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    ctx: *mut g::ggml_context,
    inputs: Vec<NamedTensor>,
    sides: HashMap<usize, Vec<SideResidual>>,
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
            sides: HashMap::new(),
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
        self.tensor_1d(name, g::ggml_type_GGML_TYPE_F32, ne0, input)
    }

    pub fn tensor_1d(
        &mut self,
        name: &str,
        ggml_type: u32,
        ne0: i64,
        input: bool,
    ) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_1d(self.ctx, ggml_type, ne0);
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

    pub fn tensor_3d(
        &mut self,
        name: &str,
        ggml_type: u32,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        input: bool,
    ) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_3d(self.ctx, ggml_type, ne0, ne1, ne2);
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

    /// `Wx`, plus `scale * B(Ax)` for every factor attached to `weight`.
    /// An unattached weight is one matmul.
    pub fn project(&self, weight: Tensor, x: Tensor) -> Tensor {
        let mut y = self.mul_mat(weight, x);
        let Some(rows) = self.sides.get(&(weight as usize)) else {
            return y;
        };
        for row in rows {
            let ax = self.mul_mat(row.a, x);
            let delta = self.scale(self.mul_mat(row.b, ax), row.scale);
            y = self.add(y, delta);
        }
        y
    }

    /// Register one low-rank residual on `weight`. The factor tensors are graph inputs
    /// named `lora.{index}.a` and `lora.{index}.b`.
    pub fn attach_lora(
        &mut self,
        weight: Tensor,
        index: usize,
        in_dim: i64,
        rank: i64,
        out_dim: i64,
        scale: f32,
    ) -> Result<(), RuntimeError> {
        if weight.is_null() || in_dim < 1 || rank < 1 || out_dim < 1 || !scale.is_finite() {
            return Err(RuntimeError::Message(
                "lora factor rank or scale is empty".into(),
            ));
        }
        let a = self.f32_2d(&format!("lora.{index}.a"), in_dim, rank, true);
        let b = self.f32_2d(&format!("lora.{index}.b"), rank, out_dim, true);
        if a.is_null() || b.is_null() {
            return Err(RuntimeError::Init);
        }
        self.sides
            .entry(weight as usize)
            .or_default()
            .push(SideResidual { a, b, scale });
        Ok(())
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

    pub fn tanh(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_tanh(self.ctx, a) }
    }

    /// `ggml_gelu`: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    /// Diffusers `gelu-approximate` and the Qwen-Image MLP. Not `ggml_gelu_erf`
    /// and not `ggml_gelu_quick`.
    pub fn gelu(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_gelu(self.ctx, a) }
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

    /// Scatter rows back. Destination shape comes from `shape`; its values are not read.
    pub fn get_rows_back(
        &self,
        values: *mut g::ggml_tensor,
        rows: *mut g::ggml_tensor,
        shape: *mut g::ggml_tensor,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_get_rows_back(self.ctx, values, rows, shape) }
    }

    pub fn view_1d(&self, a: *mut g::ggml_tensor, ne0: i64, offset: usize) -> *mut g::ggml_tensor {
        unsafe { g::ggml_view_1d(self.ctx, a, ne0, offset) }
    }

    /// Shape carrier for `ggml_repeat`. Not an input; its values are never read.
    pub fn blank_f32_2d(&self, ne0: i64, ne1: i64) -> *mut g::ggml_tensor {
        unsafe { g::ggml_new_tensor_2d(self.ctx, g::ggml_type_GGML_TYPE_F32, ne0, ne1) }
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

    pub fn f32_4d(
        &mut self,
        name: &str,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        input: bool,
    ) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_4d(self.ctx, g::ggml_type_GGML_TYPE_F32, ne0, ne1, ne2, ne3);
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

    /// Group norm along channels (`ne[2]`). The affine scale and shift are a later mul and add.
    pub fn group_norm(&self, a: Tensor, n_groups: i32, eps: f32) -> Tensor {
        unsafe { g::ggml_group_norm(self.ctx, a, n_groups, eps) }
    }

    /// Nearest ×2 on width and height. Channels stay on `ne[2]`.
    pub fn upscale_nearest(&self, a: Tensor) -> Tensor {
        unsafe { g::ggml_upscale(self.ctx, a, 2, g::ggml_scale_mode_GGML_SCALE_MODE_NEAREST) }
    }

    pub fn sin(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_sin(self.ctx, a) }
    }

    /// 1D convolution. Kernel `a` is `[K, IC, OC]`, data `b` is `[T, IC]` (ggml).
    pub fn conv_1d(
        &self,
        kernel: Tensor,
        data: Tensor,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Tensor {
        unsafe { g::ggml_conv_1d(self.ctx, kernel, data, stride, padding, dilation) }
    }

    /// Scatter-add GEMM columns to a 1D signal. `a` is `[K*OC, T_in]`.
    pub fn col2im_1d(&self, columns: Tensor, stride: i32, out_channels: i32, padding: i32) -> Tensor {
        unsafe { g::ggml_col2im_1d(self.ctx, columns, stride, out_channels, padding) }
    }

    pub fn sqr(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_sqr(self.ctx, a) }
    }

    pub fn sqrt(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_sqrt(self.ctx, a) }
    }

    pub fn div(&self, a: *mut g::ggml_tensor, b: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_div(self.ctx, a, b) }
    }

    /// Sum along `ne[0]`. The result has `ne[0] == 1`.
    pub fn sum_rows(&self, a: *mut g::ggml_tensor) -> *mut g::ggml_tensor {
        unsafe { g::ggml_sum_rows(self.ctx, a) }
    }

    pub fn clamp(&self, a: *mut g::ggml_tensor, min: f32, max: f32) -> *mut g::ggml_tensor {
        unsafe { g::ggml_clamp(self.ctx, a, min, max) }
    }

    /// Asymmetric pad. Each pair is the left and right pad of `ne[0]` through `ne[3]`.
    pub fn pad_ext(
        &self,
        a: *mut g::ggml_tensor,
        lp0: i32,
        rp0: i32,
        lp1: i32,
        rp1: i32,
        lp2: i32,
        rp2: i32,
        lp3: i32,
        rp3: i32,
    ) -> *mut g::ggml_tensor {
        unsafe { g::ggml_pad_ext(self.ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3) }
    }

    /// Direct 3D convolution. Kernel is `[KW, KH, KD, IC * OC]`, data is `[W, H, D, C * N]`.
    /// `IC * OC` is packed as `ic + IC * oc`. Padding here is symmetric; causal time pad is
    /// applied with `pad_ext` before a zero-pad call.
    pub fn conv3d_direct(
        &self,
        kernel: Tensor,
        data: Tensor,
        stride0: i32,
        stride1: i32,
        stride2: i32,
        pad0: i32,
        pad1: i32,
        pad2: i32,
        dil0: i32,
        dil1: i32,
        dil2: i32,
        n_channels: i32,
        n_batch: i32,
        n_channels_out: i32,
    ) -> Tensor {
        unsafe {
            g::ggml_conv_3d_direct(
                self.ctx,
                kernel,
                data,
                stride0,
                stride1,
                stride2,
                pad0,
                pad1,
                pad2,
                dil0,
                dil1,
                dil2,
                n_channels,
                n_batch,
                n_channels_out,
            )
        }
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

    pub fn i64_1d(&mut self, name: &str, ne0: i64, input: bool) -> *mut g::ggml_tensor {
        unsafe {
            let t = g::ggml_new_tensor_1d(self.ctx, g::ggml_type_GGML_TYPE_I64, ne0);
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

    /// `ggml_set_rows`: write rows of `src` into `dst` at `row_ids` (I32/I64).
    /// Returns a view of `dst` so the write stays in the forward expand.
    pub fn set_rows(&self, dst: Tensor, src: Tensor, row_ids: Tensor) -> Tensor {
        unsafe { g::ggml_set_rows(self.ctx, dst, src, row_ids) }
    }

    /// Finish the graph. `reserve` + `alloc` happen once; later steps only write inputs.
    pub fn finish(self, output: *mut g::ggml_tensor) -> Result<ReusedGraph, RuntimeError> {
        unsafe {
            g::ggml_set_output(output);
            // Qwen-Image is 60 joint blocks. SD.cpp sizes that graph at
            // QWEN_IMAGE_GRAPH_SIZE (20480). The depth test records the node
            // count and keeps this cap above it.
            let graph = g::ggml_new_graph_custom(self.ctx, GRAPH_NODES, false);
            if graph.is_null() {
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Init);
            }
            g::ggml_build_forward_expand(graph, output);
            // Gallocr reuses an input after its last reader. The next compute
            // would then see activations where the weights were. Inputs stay
            // in their own buffer for every replay.
            let inputs_buffer = match pin_inputs(self.backend, self.buft, &self.inputs) {
                Ok(buffer) => buffer,
                Err(err) => {
                    g::ggml_free(self.ctx);
                    return Err(err);
                }
            };
            let galloc = g::ggml_gallocr_new(self.buft);
            if galloc.is_null() {
                if !inputs_buffer.is_null() {
                    g::ggml_backend_buffer_free(inputs_buffer);
                }
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Init);
            }
            if !g::ggml_gallocr_reserve(galloc, graph) {
                g::ggml_gallocr_free(galloc);
                if !inputs_buffer.is_null() {
                    g::ggml_backend_buffer_free(inputs_buffer);
                }
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Reserve);
            }
            if !g::ggml_gallocr_alloc_graph(galloc, graph) {
                g::ggml_gallocr_free(galloc);
                if !inputs_buffer.is_null() {
                    g::ggml_backend_buffer_free(inputs_buffer);
                }
                g::ggml_free(self.ctx);
                return Err(RuntimeError::Reserve);
            }
            Ok(ReusedGraph {
                ctx: self.ctx,
                graph,
                galloc,
                inputs_buffer,
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
    inputs_buffer: g::ggml_backend_buffer_t,
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    inputs: Vec<NamedTensor>,
    output: *mut g::ggml_tensor,
    computes: u32,
    readbacks: u32,
}

/// Place graph inputs in a buffer gallocr does not recycle.
fn pin_inputs(
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    inputs: &[NamedTensor],
) -> Result<g::ggml_backend_buffer_t, RuntimeError> {
    if inputs.is_empty() {
        return Ok(std::ptr::null_mut());
    }
    let align = unsafe { g::ggml_backend_buft_get_alignment(buft) }.max(1);
    let mut cursor = 0usize;
    let mut placed = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        let size = unsafe { g::ggml_backend_buft_get_alloc_size(buft, input.tensor) };
        let offset = align_up(cursor, align);
        cursor = offset.saturating_add(size);
        if cursor < offset {
            return Err(RuntimeError::Reserve);
        }
        placed.push((input.tensor, offset));
    }
    let buffer = unsafe { g::ggml_backend_alloc_buffer(backend, cursor) };
    if buffer.is_null() {
        return Err(RuntimeError::Reserve);
    }
    let base = unsafe { g::ggml_backend_buffer_get_base(buffer) };
    if base.is_null() {
        unsafe { g::ggml_backend_buffer_free(buffer) };
        return Err(RuntimeError::Reserve);
    }
    for (tensor, offset) in placed {
        let addr = unsafe { base.add(offset) };
        let status = unsafe { g::ggml_backend_tensor_alloc(buffer, tensor, addr) };
        if status != g::ggml_status_GGML_STATUS_SUCCESS {
            unsafe { g::ggml_backend_buffer_free(buffer) };
            return Err(RuntimeError::Reserve);
        }
    }
    Ok(buffer)
}

fn align_up(value: usize, align: usize) -> usize {
    let align = align.max(1);
    value.saturating_add(align - 1) / align * align
}

impl ReusedGraph {
    pub fn compute_count(&self) -> u32 {
        self.computes
    }

    pub fn node_count(&self) -> i32 {
        unsafe { g::ggml_graph_n_nodes(self.graph) }
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

    pub fn set_i64(&self, name: &str, data: &[i64]) -> Result<(), RuntimeError> {
        let tensor = self
            .inputs
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.tensor)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))?;
        let nbytes = unsafe { g::ggml_nbytes(tensor) };
        let got = data.len() * std::mem::size_of::<i64>();
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

    /// Read the graph output without counting a host readback. The device tensor stays.
    pub fn peek_output(&self) -> Result<Vec<f32>, RuntimeError> {
        if self.output.is_null() {
            return Err(RuntimeError::Init);
        }
        let ggml_type = unsafe { (*self.output).type_ };
        if ggml_type != g::ggml_type_GGML_TYPE_F32 {
            return Err(RuntimeError::Message(format!(
                "graph output is ggml type {ggml_type}, expected F32"
            )));
        }
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
            if !self.inputs_buffer.is_null() {
                g::ggml_backend_buffer_free(self.inputs_buffer);
            }
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

#[cfg(test)]
mod tests {
    use crate::runtime::{BackendKind, GgmlContext};

    /// y = Wx + scale * B(Ax). W is identity, so the residual is the only change.
    #[test]
    fn side_residual_is_base_plus_scaled_low_rank() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(8 * 1024 * 1024).unwrap();
        let w = b.f32_2d("w", 2, 2, true);
        b.attach_lora(w, 0, 2, 1, 2, 0.5).unwrap();
        let x = b.f32_2d("x", 2, 1, true);
        let y = b.project(w, x);
        let mut graph = b.finish(y).unwrap();
        graph.set_f32("w", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        graph.set_f32("lora.0.a", &[1.0, 0.0]).unwrap();
        graph.set_f32("lora.0.b", &[0.0, 2.0]).unwrap();
        graph.set_f32("x", &[3.0, 4.0]).unwrap();
        let y = graph.compute().unwrap();
        let expect = [3.0f32, 7.0];
        for (got, want) in y.iter().zip(expect) {
            assert!((got - want).abs() < 1e-4, "got {y:?}");
        }
    }
}
