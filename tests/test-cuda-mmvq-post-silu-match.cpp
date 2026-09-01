#include "../ggml/src/ggml-cuda/mmvq-post-silu-match.h"
#include "../ggml/src/ggml-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

enum class match_case {
    valid,
    nt3,
    wrong_factor,
    negative_zero_bias,
    wrong_unary,
    extra_consumer,
    extra_scale_consumer,
    marked_mm,
    marked_scale,
    cleared_compute,
    mm_view_src,
    scale_view_src,
    exact_mm_alias,
    partial_mm_alias,
    exact_scale_alias,
    partial_scale_alias,
    source0_alias,
    source1_alias,
};

static bool run_case(const char * label, match_case variant, bool expected) {
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    constexpr int64_t k = 5120;
    constexpr int64_t m = 257;
    const int64_t nt = variant == match_case::nt3 ? 3 : 1;
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, nt);
    ggml_tensor * mm = ggml_mul_mat(ctx, weights, input);
    ggml_tensor * scale = ggml_scale(ctx, mm, variant == match_case::wrong_factor ? 0.5f : 0.25f);
    if (variant == match_case::negative_zero_bias) {
        ggml_set_op_params_f32(scale, 1, -0.0f);
    }
    ggml_tensor * silu = variant == match_case::wrong_unary ? ggml_tanh(ctx, scale) : ggml_silu(ctx, scale);
    if (variant == match_case::marked_mm) {
        ggml_set_output(mm);
    }
    if (variant == match_case::marked_scale) {
        ggml_set_output(scale);
    }
    ggml_tensor * output = variant == match_case::extra_consumer ? ggml_add(ctx, silu, mm) :
        variant == match_case::extra_scale_consumer ? ggml_add(ctx, silu, scale) : silu;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    if (variant == match_case::cleared_compute) {
        mm->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    }
    if (variant == match_case::mm_view_src) {
        mm->view_src = weights;
    }
    if (variant == match_case::scale_view_src) {
        scale->view_src = mm;
    }
    int start = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == mm) {
            start = i;
            break;
        }
    }

    const bool memory_case = variant >= match_case::exact_mm_alias;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    if (memory_case) {
        backend = ggml_backend_cpu_init();
        buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        if (buffer) {
            switch (variant) {
                case match_case::exact_mm_alias:    silu->data = mm->data; break;
                case match_case::partial_mm_alias:  silu->data = (char *) mm->data + sizeof(float); break;
                case match_case::exact_scale_alias: silu->data = scale->data; break;
                case match_case::partial_scale_alias: silu->data = (char *) scale->data + sizeof(float); break;
                case match_case::source0_alias:     silu->data = weights->data; break;
                case match_case::source1_alias:     silu->data = input->data; break;
                default: break;
            }
        }
    }

    ggml_cuda_mmvq_post_silu_fusion match;
    const bool actual = start >= 0 && (!memory_case || buffer) &&
        ggml_cuda_match_mmvq_post_silu(graph, start, match, memory_case);
    const bool fields_ok = !actual ||
        (match.mm == mm && match.scale == scale && match.silu == silu &&
         match.factor == 0.25f && match.nodes_to_skip == 2);
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
    ok &= run_case("valid",              match_case::valid,              true);
    ok &= run_case("nt3",                match_case::nt3,                false);
    ok &= run_case("wrong-factor",       match_case::wrong_factor,       false);
    ok &= run_case("negative-zero-bias", match_case::negative_zero_bias, false);
    ok &= run_case("wrong-unary",        match_case::wrong_unary,        false);
    ok &= run_case("extra-consumer",     match_case::extra_consumer,     false);
    ok &= run_case("extra-scale-consumer", match_case::extra_scale_consumer, false);
    ok &= run_case("marked-mm",          match_case::marked_mm,          false);
    ok &= run_case("marked-scale",       match_case::marked_scale,       false);
    ok &= run_case("cleared-compute",     match_case::cleared_compute,    false);
    ok &= run_case("mm-view-src",         match_case::mm_view_src,        false);
    ok &= run_case("scale-view-src",      match_case::scale_view_src,     false);
    ok &= run_case("exact-mm-alias",     match_case::exact_mm_alias,     true);
    ok &= run_case("partial-mm-alias",   match_case::partial_mm_alias,   false);
    ok &= run_case("exact-scale-alias",  match_case::exact_scale_alias,  true);
    ok &= run_case("partial-scale-alias", match_case::partial_scale_alias, false);
    ok &= run_case("source0-alias",      match_case::source0_alias,      false);
    ok &= run_case("source1-alias",      match_case::source1_alias,      false);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
