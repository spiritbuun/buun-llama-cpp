#pragma once

#include "common.cuh"

// Dynamic channel-FP8 with the ordinary executor's BF16 activation rounding.
// The caller must validate graph uses and storage before skipping SwiGLU.
// A declined operation launches no work and leaves the ordinary path intact.
bool ggml_cuda_mul_mat_fp8_channel_bf16(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool fuse_swiglu = false);
