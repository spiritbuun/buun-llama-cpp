#include "../ggml/src/ggml-cuda/mmvq-tuning.h"

#include <cstdio>
#include <initializer_list>

static bool expect_path(
        const char * label, ggml_cuda_q8_0_mmv_selector_inputs inputs,
        ggml_cuda_q8_0_mmv_path expected) {
    const ggml_cuda_q8_0_mmv_path actual = ggml_cuda_select_q8_0_mmv_path(inputs);
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s selected %d, expected %d\n", label, (int) actual, (int) expected);
    return false;
}

int main() {
    const ggml_cuda_q8_0_mmv_selector_inputs eligible = {
        860, 1, false, ggml_cuda_q8_0_mmv_fusion_kind::none, false, false,
    };
    bool ok = expect_path("eligible", eligible, ggml_cuda_q8_0_mmv_path::nwarps_2);

    auto excluded = eligible;
    excluded.ncols_dst = 2;
    ok &= expect_path("ncols", excluded, ggml_cuda_q8_0_mmv_path::baseline);
    excluded = eligible;
    excluded.has_ids = true;
    ok &= expect_path("ids", excluded, ggml_cuda_q8_0_mmv_path::baseline);
    excluded = eligible;
    excluded.fusion = ggml_cuda_q8_0_mmv_fusion_kind::existing;
    ok &= expect_path("existing-fusion", excluded, ggml_cuda_q8_0_mmv_path::baseline);
    excluded = eligible;
    excluded.fusion = ggml_cuda_q8_0_mmv_fusion_kind::post_only;
    ok &= expect_path("post-fusion", excluded, ggml_cuda_q8_0_mmv_path::nwarps_2);
    excluded = eligible;
    excluded.small_k = true;
    ok &= expect_path("small-k", excluded, ggml_cuda_q8_0_mmv_path::baseline);
    excluded = eligible;
    excluded.halve_iters = true;
    ok &= expect_path("halve-iters", excluded, ggml_cuda_q8_0_mmv_path::baseline);

    for (const int cc : { 750, 800, 890, 1000, 0x1000000 + 0x1030,
                          0x1000000 + 0x1100, 0x1000000 + 0x1200 }) {
        excluded = eligible;
        excluded.compute_capability = cc;
        ok &= expect_path("unmeasured-architecture", excluded, ggml_cuda_q8_0_mmv_path::baseline);
    }

    if (ok) {
        std::puts("PASS: Q8_0 generic MMV selector and exclusions");
    }
    return ok ? 0 : 1;
}
