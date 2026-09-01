#include "../ggml/src/ggml-cuda/hc-stream-mean-match.h"
#include "../ggml/src/ggml-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

enum class stream_mean_case {
    valid,
    flattened_base,
    swapped_first_add,
    wrong_scale,
    negative_zero_bias,
    wrong_view_offset,
    marked_add,
    marked_view,
    extra_consumer,
    exact_stream0_alias,
    partial_stream0_alias,
    exact_view1_alias,
    partial_view1_alias,
    exact_view2_alias,
    partial_view2_alias,
    exact_view3_alias,
    partial_view3_alias,
    hoisted_views,
};

static bool run_case(const char * label, stream_mean_case variant, bool expected) {
    ggml_init_params params = { 4*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    constexpr int64_t n_embd = 31;
    constexpr int64_t hc = 4;
    constexpr int64_t nt = 7;
    ggml_tensor * base = variant == stream_mean_case::flattened_base
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd*hc, nt)
        : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    const size_t stream_bytes = ggml_row_size(base->type, n_embd);
    const size_t token_stride = stream_bytes*hc;
    ggml_tensor * view[hc];
    for (int64_t c = 0; c < hc; ++c) {
        const size_t offset = variant == stream_mean_case::wrong_view_offset && c == 2
            ? stream_bytes : stream_bytes*c;
        view[c] = ggml_view_2d(ctx, base, n_embd, nt, token_stride, offset);
    }
    if (variant == stream_mean_case::marked_view) {
        ggml_set_output(view[2]);
    }
    ggml_tensor * stream0 = ggml_cont(ctx, view[0]);
    ggml_tensor * add0 = variant == stream_mean_case::swapped_first_add
        ? ggml_add(ctx, view[1], stream0) : ggml_add(ctx, stream0, view[1]);
    if (variant == stream_mean_case::marked_add) {
        ggml_set_output(add0);
    }
    ggml_tensor * add1 = ggml_add(ctx, add0, view[2]);
    ggml_tensor * add2 = ggml_add(ctx, add1, view[3]);
    ggml_tensor * scale = ggml_scale(ctx, add2,
        variant == stream_mean_case::wrong_scale ? 0.2f : 0.25f);
    if (variant == stream_mean_case::negative_zero_bias) {
        ggml_set_op_params_f32(scale, 1, -0.0f);
    }
    ggml_tensor * output = scale;
    if (variant == stream_mean_case::extra_consumer) {
        output = ggml_add(ctx, scale, add0);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    if (variant == stream_mean_case::hoisted_views) {
        ggml_tensor * reordered[] = {
            view[0], stream0, view[1], view[2], view[3], add0, add1, add2, scale,
        };
        GGML_ASSERT(graph->n_nodes == (int) (sizeof(reordered)/sizeof(reordered[0])));
        std::memcpy(graph->nodes, reordered, sizeof(reordered));
    }
    int start = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == add0) {
            start = i;
            break;
        }
    }

    ggml_cuda_hc_stream_mean_fusion match;
    const bool check_memory_ranges = variant >= stream_mean_case::exact_stream0_alias &&
        variant <= stream_mean_case::partial_view3_alias;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    if (check_memory_ranges) {
        backend = ggml_backend_cpu_init();
        buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        if (buffer) {
            if (variant == stream_mean_case::exact_stream0_alias) {
                scale->data = stream0->data;
            } else if (variant == stream_mean_case::partial_stream0_alias) {
                scale->data = (char *) stream0->data + sizeof(float);
            } else {
                const int alias_case = (int) variant - (int) stream_mean_case::exact_view1_alias;
                const int view_index = 1 + alias_case/2;
                const bool partial = alias_case % 2 != 0;
                scale->data = (char *) view[view_index]->data + (partial ? sizeof(float) : 0);
            }
        }
    }
    const bool actual = start >= 0 && (!check_memory_ranges || buffer) &&
        ggml_cuda_match_hc_stream_mean(graph, start, match, check_memory_ranges);
    const int expected_skip = variant == stream_mean_case::hoisted_views ? 3 : 5;
    const bool fields_ok = !actual ||
        (match.streams[0] == stream0 && match.streams[1] == view[1] &&
         match.streams[2] == view[2] && match.streams[3] == view[3] &&
         match.dst == scale && match.nodes_to_skip == expected_skip);
    if (actual != expected || !fields_ok) {
        std::fprintf(stderr, "%s: expected %d, got %d; skip=%d; graph:",
            label, expected, actual, actual ? match.nodes_to_skip : -1);
        for (int i = 0; i < graph->n_nodes; ++i) {
            std::fprintf(stderr, " %d:%s", i, ggml_op_name(graph->nodes[i]->op));
        }
        std::fprintf(stderr, "\n");
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
    ggml_free(ctx);
    return actual == expected && fields_ok;
}

int main() {
    bool ok = true;
    ok &= run_case("valid",             stream_mean_case::valid,             true);
    ok &= run_case("flattened-base",    stream_mean_case::flattened_base,    true);
    ok &= run_case("swapped-first-add", stream_mean_case::swapped_first_add, false);
    ok &= run_case("wrong-scale",       stream_mean_case::wrong_scale,       false);
    ok &= run_case("negative-zero-bias", stream_mean_case::negative_zero_bias, false);
    ok &= run_case("wrong-view-offset", stream_mean_case::wrong_view_offset, false);
    ok &= run_case("marked-add",        stream_mean_case::marked_add,        false);
    ok &= run_case("marked-view",       stream_mean_case::marked_view,       false);
    ok &= run_case("extra-consumer",    stream_mean_case::extra_consumer,    false);
    ok &= run_case("exact-stream0-alias", stream_mean_case::exact_stream0_alias, true);
    ok &= run_case("partial-stream0-alias", stream_mean_case::partial_stream0_alias, false);
    ok &= run_case("exact-view1-alias", stream_mean_case::exact_view1_alias, false);
    ok &= run_case("partial-view1-alias", stream_mean_case::partial_view1_alias, false);
    ok &= run_case("exact-view2-alias", stream_mean_case::exact_view2_alias, false);
    ok &= run_case("partial-view2-alias", stream_mean_case::partial_view2_alias, false);
    ok &= run_case("exact-view3-alias", stream_mean_case::exact_view3_alias, false);
    ok &= run_case("partial-view3-alias", stream_mean_case::partial_view3_alias, false);
    ok &= run_case("hoisted-views", stream_mean_case::hoisted_views, true);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
