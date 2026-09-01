#pragma once

#include "ggml.h"

struct ggml_cuda_hc_grouped_rms_fusion {
    ggml_tensor * rms = nullptr;
    ggml_tensor * reshape = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * dst = nullptr;
    int nodes_to_skip = 0;
};

bool ggml_cuda_match_hc_grouped_rms(
    const ggml_cgraph * cgraph, int node_idx,
    ggml_cuda_hc_grouped_rms_fusion & fusion, bool check_memory_ranges = true);
