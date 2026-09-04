#pragma once

#include "common.cuh"

bool ggml_cuda_marlin_q8_g128_enabled();
bool ggml_cuda_marlin_q8_g128_supports_shape(int64_t n, int64_t k, int64_t m, int cc);
bool ggml_cuda_marlin_q8_g128_is_repacked(const ggml_tensor * tensor);

void ggml_cuda_marlin_q8_g128_repack_upload(
    const void * canonical,
    void * storage,
    int64_t n,
    int64_t k,
    int max_shared,
    int sms,
    cudaStream_t stream);

void ggml_cuda_marlin_q8_g128_unrepack(
    const void * storage,
    void * canonical,
    int64_t n,
    int64_t k,
    cudaStream_t stream);

// BF16 rows [row0, row0 + rows) of the weight, read from the Marlin layout.
void ggml_cuda_marlin_q8_g128_dequant_bf16(
    const void * storage,
    nv_bfloat16 * dst,
    int64_t n,
    int64_t k,
    int64_t row0,
    int64_t rows,
    cudaStream_t stream);

// weight_alt/scale_alt (optional) append a second projection so the launch
// computes [weight | weight_alt] as one 2n-wide GEMM into output rows of 2n.
void ggml_cuda_marlin_q8_g128_launch(
    const nv_bfloat16 * input,
    const void * weight,
    const void * scale,
    const void * weight_alt,
    const void * scale_alt,
    nv_bfloat16 * output,
    int32_t * locks,
    int64_t n,
    int64_t k,
    int64_t m,
    int max_shared,
    int sms,
    cudaStream_t stream);
