#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vbr.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

// TurboQuant/VBR KV-cache support — the CUDA implementation of the backend-agnostic
// interface in ggml-vbr.h (struct/semantics docs live there). libllama reaches these
// through ggml_backend_cuda_vbr_iface (resolved via GGML_VBR_BACKEND_IFACE_PROC), never
// by direct link.
GGML_BACKEND_API const struct ggml_vbr_backend_iface * ggml_backend_cuda_vbr_iface(void);

GGML_BACKEND_API void ggml_backend_cuda_kv_transcode(ggml_backend_t backend,
                                                     const struct ggml_vbr_transcode_params * params);
GGML_BACKEND_API void ggml_backend_cuda_kv_stash_capture(ggml_backend_t backend, const struct ggml_tensor * src,
                                                         void * stash_f16, int64_t n_rows, bool is_v);
GGML_BACKEND_API void ggml_backend_cuda_sync_device(int device);
GGML_BACKEND_API void ggml_backend_cuda_vbr_fence_arm(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// VMM pool (works on CUDA and ROCm — HIP maps the cuMem* driver API; *_available reports
// the device flag)
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_available(int device);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_granularity(int device);
GGML_BACKEND_API struct ggml_vbr_vmm_pool * ggml_backend_cuda_vmm_pool_init(int device, size_t va_size);
GGML_BACKEND_API void   ggml_backend_cuda_vmm_pool_free(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API void * ggml_backend_cuda_vmm_pool_base(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_pool_mapped(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_pool_map(struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_pool_unmap(struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
GGML_BACKEND_API void   ggml_backend_cuda_vmm_pool_clear(struct ggml_vbr_vmm_pool * pool);

// wrap externally-managed device memory (e.g. a VMM VA range) as a CUDA backend buffer; the buffer
// does NOT take ownership — freeing it never cudaFree's ptr.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_buffer_from_ptr(int device, void * ptr, size_t size);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// #88: boundary-time reserve of the fattn f16 dequant scratch (ggml-vbr.h vtable slot)
GGML_BACKEND_API bool ggml_backend_cuda_kv_dequant_scratch_reserve(int device, size_t k_bytes, size_t v_bytes);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
