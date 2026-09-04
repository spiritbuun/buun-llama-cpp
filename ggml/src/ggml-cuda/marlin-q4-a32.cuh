#pragma once

#include "common.cuh"

bool ggml_cuda_marlin_q4_a32_enabled();
bool ggml_cuda_marlin_q4_a32_supports_shape(int64_t n, int64_t k, int64_t m, int cc);
bool ggml_cuda_marlin_q4_a32_is_repacked(const ggml_tensor * tensor);

void ggml_cuda_marlin_q4_a32_repack_upload(
    const void * canonical,
    void * storage,
    int64_t n,
    int64_t k,
    int max_shared,
    int sms,
    cudaStream_t stream);

void ggml_cuda_marlin_q4_a32_unrepack(
    const void * storage,
    void * canonical,
    int64_t n,
    int64_t k,
    cudaStream_t stream);

void ggml_cuda_marlin_q4_a32_prepare(
    const void * canonical,
    void * raw_weight,
    void * marlin_weight,
    void * marlin_scale,
    void * marlin_zero,
    int64_t n,
    int64_t k,
    int max_shared,
    int sms,
    cudaStream_t stream);

// BF16 rows [row0, row0 + rows) of the weight, read from the Marlin layout.
void ggml_cuda_marlin_q4_a32_dequant_bf16(
    const void * storage,
    nv_bfloat16 * dst,
    int64_t n,
    int64_t k,
    int64_t row0,
    int64_t rows,
    cudaStream_t stream);

void ggml_cuda_marlin_q4_a32_launch(
    const nv_bfloat16 * input,
    const void * weight,
    const void * scale,
    const void * zero,
    void * output,          // BF16, or F32 (BF16-rounded values) when out_f32
    bool out_f32,
    int32_t * locks,
    int64_t n,
    int64_t k,
    int64_t m,
    int max_shared,
    int sms,
    cudaStream_t stream);
