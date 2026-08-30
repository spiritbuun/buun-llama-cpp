#include "../ggml/src/ggml-cuda/moe-cache-mmv-tuning.h"

#include <cstdio>
#include <initializer_list>

static bool expect(
        const char * label, int cc, ggml_type type, int64_t hits,
        int factor, ggml_cuda_moe_cache_flat_hits_path path) {
    const auto got = ggml_cuda_select_moe_cache_flat_hits(cc, type, hits);
    if (got.factor == factor && got.path == path) {
        return true;
    }
    std::fprintf(stderr, "%s: got factor=%d path=%d, expected factor=%d path=%d\n",
        label, got.factor, (int) got.path, factor, (int) path);
    return false;
}

int main() {
    using path = ggml_cuda_moe_cache_flat_hits_path;
    bool ok = true;
    for (const ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_1}) {
        for (int64_t hits = 1; hits <= 10; ++hits) {
            ok &= expect("sm86-measured-type", 860, type, hits, 2, path::factor_2);
        }
    }
    ok &= expect("q8-unmeasured", 860, GGML_TYPE_Q8_0, 10, 1, path::factor_1);
    ok &= expect("q4_0-unmeasured", 860, GGML_TYPE_Q4_0, 10, 1, path::factor_1);
    ok &= expect("unsupported-type", 860, GGML_TYPE_F32, 10, 1, path::factor_1);
    ok &= expect("zero-hits", 860, GGML_TYPE_Q4_K, 0, 1, path::factor_1);
    for (const int cc : {750, 800, 890, 1000, 0x1000000 + 0x1030,
                         0x1000000 + 0x1100, 0x1000000 + 0x1200}) {
        ok &= expect("other-arch", cc, GGML_TYPE_Q4_K, 10, 1, path::factor_1);
    }
    if (ok) {
        std::puts("PASS: MoE cache flattened-hit selector");
    }
    return ok ? 0 : 1;
}
