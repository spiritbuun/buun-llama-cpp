#include "../ggml/src/ggml-cuda/hc-grouped-rms-match.h"
#include "../ggml/src/ggml-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>

enum class match_case {
    valid,
    hc_one,
    wrong_gamma_shape,
    extra_reshape_consumer,
    marked_rms,
    marked_reshape,
    cleared_compute,
    wrong_edge,
    missing_reshape_view,
    strided_gamma,
    negative_eps,
    nan_eps,
    exact_rms_alias,
    partial_rms_alias,
    source_alias,
    gamma_alias,
};

static bool run_case(const char * label, match_case variant, bool expected) {
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    constexpr int64_t n_embd = 257;
    const int64_t hc = variant == match_case::hc_one ? 1 : 4;
    constexpr int64_t nt = 3;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * rms = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor * reshape = ggml_reshape_2d(ctx, rms, n_embd*hc, nt);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd*hc);
    ggml_tensor * mul = ggml_mul(ctx, reshape, gamma);
    if (variant == match_case::marked_rms) ggml_set_output(rms);
    if (variant == match_case::marked_reshape) ggml_set_output(reshape);
    ggml_tensor * output = variant == match_case::extra_reshape_consumer
        ? ggml_add(ctx, mul, reshape) : mul;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    int start = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == rms) { start = i; break; }
    }
    if (variant == match_case::wrong_gamma_shape) gamma->ne[0]--;
    if (variant == match_case::cleared_compute) rms->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    if (variant == match_case::wrong_edge) mul->src[0] = rms;
    if (variant == match_case::missing_reshape_view) reshape->view_src = nullptr;
    if (variant == match_case::strided_gamma) gamma->nb[0] = 2*sizeof(float);
    if (variant == match_case::negative_eps || variant == match_case::nan_eps) {
        const float eps = variant == match_case::negative_eps
            ? -1.0f : std::numeric_limits<float>::quiet_NaN();
        std::memcpy(rms->op_params, &eps, sizeof(eps));
    }

    const bool memory_case = variant >= match_case::exact_rms_alias;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    if (memory_case) {
        backend = ggml_backend_cpu_init();
        buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        if (buffer) {
            switch (variant) {
                case match_case::exact_rms_alias:   mul->data = rms->data; break;
                case match_case::partial_rms_alias: mul->data = (char *) rms->data + sizeof(float); break;
                case match_case::source_alias:      mul->data = x->data; break;
                case match_case::gamma_alias:       mul->data = gamma->data; break;
                default: break;
            }
        }
    }

    ggml_cuda_hc_grouped_rms_fusion match;
    const bool actual = start >= 0 && (!memory_case || buffer) &&
        ggml_cuda_match_hc_grouped_rms(graph, start, match, memory_case);
    const bool fields_ok = !actual || (match.rms == rms && match.reshape == reshape &&
        match.gamma == gamma && match.dst == mul && match.nodes_to_skip == 2);
    if (actual != expected || !fields_ok) {
        std::fprintf(stderr, "%s: expected %d got %d fields=%d\n", label, expected, actual, fields_ok);
    }
    if (buffer) ggml_backend_buffer_free(buffer);
    if (backend) ggml_backend_free(backend);
    ggml_free(ctx);
    return actual == expected && fields_ok;
}

int main() {
    bool ok = true;
    ok &= run_case("valid", match_case::valid, true);
    ok &= run_case("hc-one", match_case::hc_one, false);
    ok &= run_case("wrong-gamma-shape", match_case::wrong_gamma_shape, false);
    ok &= run_case("extra-reshape-consumer", match_case::extra_reshape_consumer, false);
    ok &= run_case("marked-rms", match_case::marked_rms, false);
    ok &= run_case("marked-reshape", match_case::marked_reshape, false);
    ok &= run_case("cleared-compute", match_case::cleared_compute, false);
    ok &= run_case("wrong-edge", match_case::wrong_edge, false);
    ok &= run_case("missing-reshape-view", match_case::missing_reshape_view, false);
    ok &= run_case("strided-gamma", match_case::strided_gamma, false);
    ok &= run_case("exact-rms-alias", match_case::exact_rms_alias, true);
    ok &= run_case("partial-rms-alias", match_case::partial_rms_alias, false);
    ok &= run_case("source-alias", match_case::source_alias, false);
    ok &= run_case("gamma-alias", match_case::gamma_alias, false);
    ok &= run_case("negative-eps", match_case::negative_eps, false);
    ok &= run_case("nan-eps", match_case::nan_eps, false);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
