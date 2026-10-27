//! Device session. One ggml backend, no model names.

use super::{BackendKind, GraphBuilder, RuntimeError};
use danmo_ggml_sys as g;
use std::ffi::CStr;

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

    /// ggml's name for this device (`CPU`, `Metal`, …).
    pub fn device_name(&self) -> String {
        unsafe {
            let ptr = g::ggml_backend_name(self.backend);
            if ptr.is_null() {
                return String::new();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    pub fn builder(&self, mem_size: usize) -> Result<GraphBuilder, RuntimeError> {
        GraphBuilder::new(self.backend, self.buft, mem_size)
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
