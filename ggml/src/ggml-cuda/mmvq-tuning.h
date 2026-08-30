#pragma once

enum class ggml_cuda_q8_0_mmv_path {
    baseline,
    nwarps_2,
};

enum class ggml_cuda_q8_0_mmv_fusion_kind {
    none,
    existing,
    post_only,
};

struct ggml_cuda_q8_0_mmv_selector_inputs {
    int  compute_capability;
    int  ncols_dst;
    bool has_ids;
    ggml_cuda_q8_0_mmv_fusion_kind fusion;
    bool small_k;
    bool halve_iters;
};

constexpr ggml_cuda_q8_0_mmv_path ggml_cuda_select_q8_0_mmv_path(
        ggml_cuda_q8_0_mmv_selector_inputs inputs) {
    return inputs.compute_capability == 860 && inputs.ncols_dst == 1 &&
                   !inputs.has_ids && inputs.fusion != ggml_cuda_q8_0_mmv_fusion_kind::existing &&
                   !inputs.small_k && !inputs.halve_iters
        ? ggml_cuda_q8_0_mmv_path::nwarps_2
        : ggml_cuda_q8_0_mmv_path::baseline;
}
