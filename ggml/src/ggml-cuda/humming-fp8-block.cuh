#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP)

bool ggml_cuda_humming_fp8_block_supports_shape(int64_t n, int64_t k, int64_t m, int cc);

void ggml_cuda_humming_fp8_block_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const float * scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int sms,
        cudaStream_t stream);

#endif
