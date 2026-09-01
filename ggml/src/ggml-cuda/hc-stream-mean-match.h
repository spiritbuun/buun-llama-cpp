#pragma once

#include "ggml.h"

struct ggml_cuda_hc_stream_mean_fusion {
    const ggml_tensor * streams[4];
    ggml_tensor *       dst;
    int                 nodes_to_skip;
};

// check_memory_ranges is false only for model-free matcher tests before backend allocation.
bool ggml_cuda_match_hc_stream_mean(
    const ggml_cgraph * cgraph, int node_idx,
    ggml_cuda_hc_stream_mean_fusion & fusion, bool check_memory_ranges = true);
