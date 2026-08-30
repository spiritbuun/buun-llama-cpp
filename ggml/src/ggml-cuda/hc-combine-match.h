#pragma once

#include "ggml.h"

struct ggml_cuda_hc_combine_fusion {
    const ggml_tensor * residual;
    const ggml_tensor * block;
    const ggml_tensor * repeated;
    const ggml_tensor * inject;
    ggml_tensor *       dst;
    bool                use_repeated_block;
    int                 nodes_to_skip;
};

// check_memory_ranges is false only for model-free matcher tests before backend allocation.
bool ggml_cuda_match_hc_combine(
    const ggml_cgraph * cgraph, int node_idx,
    ggml_cuda_hc_combine_fusion & fusion, bool check_memory_ranges = true);
