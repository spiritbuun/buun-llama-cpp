#pragma once

#include "ggml.h"

enum class ggml_cuda_moe_cache_flat_hits_path {
    factor_1,
    factor_2,
};

struct ggml_cuda_moe_cache_flat_hits_selection {
    int factor;
    ggml_cuda_moe_cache_flat_hits_path path;
};

static inline ggml_cuda_moe_cache_flat_hits_selection ggml_cuda_select_moe_cache_flat_hits(
        int cc, ggml_type type, int64_t n_hits) {
    if (cc == 860 && n_hits > 0 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_1)) {
        return {2, ggml_cuda_moe_cache_flat_hits_path::factor_2};
    }
    return {1, ggml_cuda_moe_cache_flat_hits_path::factor_1};
}
