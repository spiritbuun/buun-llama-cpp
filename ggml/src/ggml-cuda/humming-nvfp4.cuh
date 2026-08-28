#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP)

bool ggml_cuda_humming_nvfp4_enabled();
bool ggml_cuda_humming_nvfp4_supports_shape(int64_t n, int64_t k, int64_t m, int cc);

// Converts between the canonical block_nvfp4 representation and Humming's
// size-neutral [packed weights | reordered block scales] representation.
// These are upload/readback operations; the transformed tensor occupies
// exactly ggml_nbytes(tensor) bytes.
void ggml_cuda_humming_nvfp4_repack_upload(
        const void * canonical_host, void * repacked_device,
        int64_t n, int64_t k, cudaStream_t stream);
void ggml_cuda_humming_nvfp4_unrepack(
        const void * repacked_device, void * canonical_device,
        int64_t n, int64_t k, cudaStream_t stream);

void ggml_cuda_humming_nvfp4_prepare(
        const void * canonical,
        void * original_weight,
        void * repacked_weight,
        void * repacked_scale,
        int64_t n,
        int64_t k,
        cudaStream_t stream);

void ggml_cuda_humming_nvfp4_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const float * tensor_scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int sms,
        cudaStream_t stream);

#endif
