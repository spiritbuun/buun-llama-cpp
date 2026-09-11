#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Compare every row of each M=1..8 projection with that row executed alone.
// Rows deliberately have different scales/outliers: sharing an activation max
// across rows must not accidentally satisfy this test.
static bool check_batch(ggml_backend_t backend, int bits, int k, int n, bool head) {
    constexpr int max_m = 8;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({4*1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, ggml_exl3_type(bits, 2), k, n);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, max_m);
    auto * svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    auto * suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    ggml_set_name(w, head ? "output.weight" : "probe.weight");
    auto * graph = ggml_new_graph_custom(compute, 256, false);
    auto project = [&](int m, int row) {
        auto * v = ggml_view_2d(compute, x, k, m, x->nb[1], row*x->nb[1]);
        auto * y = ggml_mul_mat(compute, w, v);
        y->src[2] = svh;
        y->src[3] = suh;
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
        return y;
    };
    ggml_tensor * single[max_m], * batch[max_m];
    for (int row = 0; row < max_m; ++row) single[row] = project(1, row);
    for (int m = 1; m <= max_m; ++m) batch[m-1] = project(m, 0);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer && ggml_gallocr_alloc_graph(alloc, graph));
    unsigned rng = 1234567;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng; };
    std::vector<unsigned char> weights(ggml_nbytes(w));
    for (auto & v : weights) v = next() >> 24;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    std::vector<float> input(k*max_m);
    const float scales[max_m] = {1.0f, 0.0f, 0.001f, 100.0f, 0.1f, 10.0f, 1.0f, 0.01f};
    for (size_t i = 0; i < input.size(); ++i) {
        float v = 0;
        for (int j = 0; j < 6; ++j) v += float(next() >> 8)/16777216.0f - 0.5f;
        input[i] = v*scales[i/k]*(i%127 == 0 ? 8.0f : 1.0f);
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    std::vector<ggml_fp16_t> signs(k), output_scales(n);
    for (int i = 0; i < k; ++i) signs[i] = ggml_fp32_to_fp16(i%3 ? 1.0f : -1.0f);
    for (int i = 0; i < n; ++i) {
        output_scales[i] = ggml_fp32_to_fp16((i%7 ? 1.0f : -1.0f)*(0.01f + float(i%13)*0.002f));
    }
    ggml_backend_tensor_set(suh, signs.data(), 0, ggml_nbytes(suh));
    ggml_backend_tensor_set(svh, output_scales.data(), 0, ggml_nbytes(svh));
    std::vector<float> expected(n*max_m), actual(n*max_m), first;
    bool ok = true;
    for (int pass = 0; pass < 4; ++pass) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (int row = 0; row < max_m; ++row) {
            ggml_backend_tensor_get(single[row], expected.data() + row*n, 0, ggml_nbytes(single[row]));
        }
        for (float v : expected) ok &= std::isfinite(v);
        if (pass == 0) first = expected;
        else ok &= memcmp(first.data(), expected.data(), expected.size()*sizeof(float)) == 0;
        for (int m = 1; m <= max_m; ++m) {
            ggml_backend_tensor_get(batch[m-1], actual.data(), 0, ggml_nbytes(batch[m-1]));
            const bool same = memcmp(expected.data(), actual.data(), ggml_nbytes(batch[m-1])) == 0;
            if (!same) printf("MISMATCH K%d k=%d n=%d head=%d M=%d pass=%d\n", bits, k, n, head, m, pass);
            ok &= same;
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(compute);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("K%d k=%d n=%d head=%d: %s\n", bits, k, n, head, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    auto * reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) return 77;
    auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    GGML_ASSERT(backend);
    bool ok = true;
    for (int bits = 1; bits <= 8; ++bits) {
        ok &= check_batch(backend, bits, 5120, 640, false);  // partial N tile, many K slices
        ok &= check_batch(backend, bits, 2048, 6144, false);
        ok &= check_batch(backend, bits, 4096, 151936, true); // cap-limited split, head residual
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
