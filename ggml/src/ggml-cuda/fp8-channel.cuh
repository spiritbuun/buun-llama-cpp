#pragma once

#include "common.cuh"

// fuse_swiglu requires the caller to validate the producer's uses and storage
// before eliding it. Unsupported fusion returns false without launching work.
bool ggml_cuda_mul_mat_fp8_channel_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool fuse_swiglu = false);
