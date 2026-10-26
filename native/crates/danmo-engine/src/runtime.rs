//! ggml runtime. The only module that touches ggml tensor types.

use danmo_ggml_sys as g;
use std::ffi::CString;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("ggml backend {0} was requested but this binary was not built with it")]
    BackendUnavailable(String),
    #[error("ggml init failed")]
    Init,
    #[error("ggml graph reserve failed")]
    Reserve,
    #[error("ggml graph compute failed: {0}")]
    Compute(i32),
    #[error("tensor {0} was not declared as a graph input")]
    MissingInput(String),
    #[error("tensor {name} has {expected} bytes, got {got}")]
    SizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl BackendKind {
    pub fn parse(name: &str) -> Result<Self, RuntimeError> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            other => Err(RuntimeError::BackendUnavailable(other.to_string())),
        }
    }

    pub fn from_env() -> Result<Self, RuntimeError> {
        match std::env::var("DANQING_GGML_BACKEND") {
            Ok(name) => Self::parse(name.trim()),
            Err(_) => Ok(Self::Cpu),
        }
    }
}

pub struct GgmlContext {
    backend: g::ggml_backend_t,
    buft: g::ggml_backend_buffer_type_t,
    kind: BackendKind,
}

impl GgmlContext {
    pub fn new(kind: BackendKind) -> Result<Self, RuntimeError> {
        let backend = unsafe { init_backend(kind)? };
        let buft = unsafe { g::ggml_backend_get_default_buffer_type(backend) };
        if buft.is_null() {
            unsafe { g::ggml_backend_free(backend) };
            return Err(RuntimeError::Init);
        }
        Ok(Self {
            backend,
            buft,
            kind,
        })
    }

    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    pub fn builder(&self, mem_size: usize) -> Result<GraphBuilder<'_>, RuntimeError> {
        GraphBuilder::new(self, mem_size)
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe { g::ggml_backend_free(self.backend) };
    }
}

unsafe fn init_backend(kind: BackendKind) -> Result<g::ggml_backend_t, RuntimeError> {
    let backend = match kind {
        BackendKind::Cpu => g::ggml_backend_cpu_init(),
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                g::ggml_backend_cuda_init(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(RuntimeError::BackendUnavailable("cuda".into()));
            }
        }
        BackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                g::ggml_backend_vk_init(0)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                return Err(RuntimeError::BackendUnavailable("vulkan".into()));
            }
        }
        BackendKind::Metal => {
            #[cfg(feature = "metal")]
            {
                g::ggml_backend_metal_init()
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(RuntimeError::BackendUnavailable("metal".into()));
            }
        }
    };
    if backend.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(backend)
    }
}

pub struct GraphBuilder<'a> {
    owner: &'a GgmlContext,
    ctx: *mut g::ggml_context,
    inputs: Vec<NamedTensor>,
}

pub struct NamedTensor {
    pub name: String,
    pub tensor: *mut g::ggml_tensor,
}

impl<'a> GraphBuilder<'a> {
    fn new(owner: &'a GgmlContext, mem_size: usize) -> Result<Self, RuntimeError> {
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
            owner,
            ctx,
            inputs: Vec::new(),
        })
    }

    pub fn ctx(&self) -> *mut g::ggml_context {
        self.ctx
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
            let galloc = g::ggml_gallocr_new(self.owner.buft);
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
                backend: self.owner.backend,
                inputs: self.inputs,
                output,
                computes: 0,
            })
        }
    }
}

pub struct ReusedGraph {
    ctx: *mut g::ggml_context,
    graph: *mut g::ggml_cgraph,
    galloc: g::ggml_gallocr_t,
    backend: g::ggml_backend_t,
    inputs: Vec<NamedTensor>,
    output: *mut g::ggml_tensor,
    computes: u32,
}

impl ReusedGraph {
    pub fn compute_count(&self) -> u32 {
        self.computes
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

    /// Recompute the reserved graph. The cgraph and gallocr stay in place.
    pub fn compute(&mut self) -> Result<Vec<f32>, RuntimeError> {
        let status = unsafe { g::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != g::ggml_status_GGML_STATUS_SUCCESS {
            return Err(RuntimeError::Compute(status));
        }
        self.computes += 1;
        let nbytes = unsafe { g::ggml_nbytes(self.output) };
        let mut out = vec![0f32; nbytes / std::mem::size_of::<f32>()];
        unsafe {
            g::ggml_backend_tensor_get(self.output, out.as_mut_ptr().cast(), 0, nbytes);
        }
        Ok(out)
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

/// Identity matmul used to prove gallocr reuse: y = W x, W fixed, x replaced between computes.
pub fn identity_probe(ctx: &GgmlContext) -> Result<ReusedGraph, RuntimeError> {
    let mut b = ctx.builder(8 * 1024 * 1024)?;
    let w = b.f32_2d("w", 2, 2, true);
    let x = b.f32_2d("x", 2, 1, true);
    let y = b.mul_mat(w, x);
    b.finish(y)
}
