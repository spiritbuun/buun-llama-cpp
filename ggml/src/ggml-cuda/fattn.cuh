#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// True when the bounded f16 prefill plan (GGML_CUDA_PREFILL_KV_MIB) applies to this op: the op runs
// a per-head-group loop with scratch, so a CUDA graph holding it must run eagerly.
bool ggml_cuda_flash_attn_ext_bounded_prefill_applies(int device, const ggml_tensor * dst);

// Release one backend context's persistent Q/K/V attention scratch. The backend destructor drains
// its streams before calling this, so VMM mappings and cudaMalloc buffers are no longer in flight.
void ggml_cuda_fattn_scratch_free(ggml_backend_cuda_context & ctx);
