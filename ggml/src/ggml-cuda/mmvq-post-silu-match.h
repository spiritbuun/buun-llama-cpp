#pragma once

#include "ggml.h"

struct ggml_cuda_mmvq_post_silu_fusion {
    ggml_tensor * mm = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * silu = nullptr;
    float factor = 1.0f;
    int nodes_to_skip = 0;
};

bool ggml_cuda_match_mmvq_post_silu(
    const ggml_cgraph * cgraph, int node_idx,
    ggml_cuda_mmvq_post_silu_fusion & fusion, bool check_memory_ranges = true);
