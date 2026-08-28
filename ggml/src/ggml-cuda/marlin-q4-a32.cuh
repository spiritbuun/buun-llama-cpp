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
    int device,
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
    int device,
    int sms,
    cudaStream_t stream);

void ggml_cuda_marlin_q4_a32_launch(
    const nv_bfloat16 * input,
    const void * weight,
    const void * scale,
    const void * zero,
    nv_bfloat16 * output,
    int32_t * locks,
    int64_t n,
    int64_t k,
    int64_t m,
    int device,
    int sms,
    cudaStream_t stream);
