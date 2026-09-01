#pragma once

#include "common.cuh"

bool ggml_cuda_int8_channel_supports(
    const ggml_tensor * weight,
    const ggml_tensor * input,
    const ggml_tensor * scale,
    const ggml_tensor * output,
    int cc);

bool ggml_cuda_mul_mat_int8_channel(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * weight,
    const ggml_tensor * input,
    ggml_tensor * output);

bool ggml_cuda_mul_mat_int8_channel_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * weight,
    const ggml_tensor * input,
    const ggml_tensor * scale,
    ggml_tensor * output,
    const ggml_cuda_mm_fusion_args_host * fusion);

bool ggml_cuda_mul_mat_int8_channel_swiglu(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * up,
    const ggml_tensor * gate,
    const ggml_tensor * input,
    ggml_tensor * output,
    bool retain_bf16_output);
