#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// One token, frozen expert IDs, and equal K-split geometry: grouped and dense
// projections must use the same activation residual policy. No routing/reduction
// or model-level tolerance can hide an omitted residual here.
static bool check_policy(ggml_backend_t backend, int bits, bool per_expert_input) {
    constexpr int k = 1024, n = 512, experts = 16, used = 4;
    const int32_t selected[used] = {7, 2, 15, 0};
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, ggml_exl3_type(bits, 2), k, n, experts);
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, per_expert_input ? used : 1, 1);
    auto * svh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, experts);
    auto * suh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, experts);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1);
    auto * grouped = ggml_mul_mat_id(compute, w, x, ids);
    grouped->src[3] = svh;
    grouped->src[4] = suh;
    ggml_set_output(grouped);
    auto * graph = ggml_new_graph_custom(compute, 128, false);
    ggml_build_forward_expand(graph, grouped);
    ggml_tensor * dense[used];
    for (int i = 0; i < used; ++i) {
        auto * wi = ggml_view_2d(compute, w, k, n, w->nb[1], selected[i]*w->nb[2]);
        auto * xi = ggml_view_2d(compute, x, k, 1, x->nb[1], per_expert_input ? i*x->nb[1] : 0);
        dense[i] = ggml_mul_mat(compute, wi, xi);
        dense[i]->src[2] = ggml_view_1d(compute, svh, n, selected[i]*svh->nb[1]);
        dense[i]->src[3] = ggml_view_1d(compute, suh, k, selected[i]*suh->nb[1]);
        ggml_set_output(dense[i]);
        ggml_build_forward_expand(graph, dense[i]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer && ggml_gallocr_alloc_graph(alloc, graph));
    unsigned rng = 1234567;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng; };
    std::vector<unsigned char> weights(ggml_nbytes(w));
    for (auto & v : weights) v = next() >> 24;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    std::vector<float> input(ggml_nelements(x));
    for (size_t i = 0; i < input.size(); ++i) {
        float v = 0;
        for (int j = 0; j < 6; ++j) v += float(next() >> 8)/16777216.0f - 0.5f;
        input[i] = v*(i%127 == 0 ? 8.0f : 1.0f);
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    std::vector<ggml_fp16_t> signs(k*experts), scales(n*experts);
    for (size_t i = 0; i < signs.size(); ++i) signs[i] = ggml_fp32_to_fp16(i%3 ? 1.0f : -1.0f);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = ggml_fp32_to_fp16((i%7 ? 1.0f : -1.0f)*(0.01f + float(i%13)*0.002f));
    }
    ggml_backend_tensor_set(suh, signs.data(), 0, ggml_nbytes(suh));
    ggml_backend_tensor_set(svh, scales.data(), 0, ggml_nbytes(svh));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    std::vector<float> actual(n*used), expected(n*used), first;
    bool ok = true;
    for (int pass = 0; pass < 4; ++pass) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(grouped, actual.data(), 0, ggml_nbytes(grouped));
        for (int i = 0; i < used; ++i) {
            ggml_backend_tensor_get(dense[i], expected.data() + i*n, 0, ggml_nbytes(dense[i]));
        }
        for (float v : actual) ok &= std::isfinite(v);
        for (float v : expected) ok &= std::isfinite(v);
        ok &= memcmp(actual.data(), expected.data(), ggml_nbytes(grouped)) == 0;
        if (pass == 0) first = actual;
        else ok &= memcmp(actual.data(), first.data(), ggml_nbytes(grouped)) == 0;
    }
    ggml_gallocr_free(alloc);
    ggml_free(compute);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("K%d per_expert_input=%d: %s\n", bits, per_expert_input, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    auto * reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) return 77;
    auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    GGML_ASSERT(backend);
    bool ok = true;
    for (int bits = 1; bits <= 8; ++bits) {
        ok &= check_policy(backend, bits, false);
        ok &= check_policy(backend, bits, true);
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
