#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP)

// Ampere fast path for channel-scaled E4M3 weights. Selected model tensors are
// transformed in place while being uploaded to CUDA; buffer reads undo the
// transform so the public tensor representation remains canonical.
bool ggml_cuda_humming_fp8_enabled();
bool ggml_cuda_humming_fp8_supports_shape(int64_t n, int64_t k, int64_t m, int cc);

// Converts canonical row-major E4M3 bytes into Humming's size-neutral MMA
// layout without allocating device memory. Intended for model-upload paths.
void ggml_cuda_humming_fp8_repack_host(
        const void * src, void * dst, int64_t n, int64_t k);
void ggml_cuda_humming_fp8_unrepack_host(
        const void * src, void * dst, int64_t n, int64_t k);

void ggml_cuda_humming_fp8_repack(
        const void * src, void * dst, int64_t n, int64_t k, cudaStream_t stream);

void ggml_cuda_humming_fp8_reorder_scale(
        const nv_bfloat16 * src, nv_bfloat16 * dst, int64_t n, cudaStream_t stream);

void ggml_cuda_humming_fp8_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const nv_bfloat16 * scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int sms,
        cudaStream_t stream);

void ggml_cuda_humming_fp8_input_f32_to_bf16(
        const float * src, nv_bfloat16 * dst, int64_t n, cudaStream_t stream);

void ggml_cuda_humming_fp8_output_bf16_to_f32(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        float * dst,
        int64_t n,
        cudaStream_t stream);

void ggml_cuda_humming_fp8_swiglu_bf16(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        nv_bfloat16 * dst,
        int64_t n,
        cudaStream_t stream);

void ggml_cuda_humming_fp8_residual_add(
        const nv_bfloat16 * src,
        const float * residual,
        float * dst,
        int64_t n,
        cudaStream_t stream);

void ggml_cuda_humming_residual_rms_prepare(
        const nv_bfloat16 * src,
        const float * residual,
        const float * norm_weight,
        float * residual_out,
        float * norm_out,
        nv_bfloat16 * norm_bf16,
        int64_t ncols,
        int64_t nrows,
        float eps,
        cudaStream_t stream);

#endif
