#pragma once

#include "common.cuh"

bool ggml_cuda_mul_mat_fp8_channel_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
