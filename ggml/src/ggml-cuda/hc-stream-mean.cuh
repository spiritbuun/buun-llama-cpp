#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_hc_stream_mean_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * const streams[4], ggml_tensor * dst);
