#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#if defined(DANMO_GGML_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(DANMO_GGML_VULKAN)
#include "ggml-vulkan.h"
#endif
#if defined(DANMO_GGML_METAL)
#include "ggml-metal.h"
#endif
