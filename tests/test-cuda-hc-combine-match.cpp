#include "../ggml/src/ggml-cuda/hc-combine-match.h"
#include "../ggml/src/ggml-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

enum class hc_case {
    valid,
    scale_first,
    swapped_operands,
    wrong_input_scale,
    wrong_output_scale,
    wrong_unary,
    extra_consumer,
    marked_intermediate,
    marked_weights_view,
    wrong_type,
    inplace_add,
    overlapping_block,
    aliased_block_inject,
    scale_first_aliased_block_inject,
};

static bool run_case(const char * label, hc_case variant, bool expected) {
    ggml_init_params params = { 4*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "%s: failed to create context\n", label);
        return false;
    }

    constexpr int64_t n_embd = 31;
    constexpr int64_t hc = 4;
    constexpr int64_t nt = 7;
    const ggml_type type = variant == hc_case::wrong_type ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * residual = ggml_new_tensor_3d(ctx, type, n_embd, hc, nt);
    const bool aliased_block_inject =
        variant == hc_case::aliased_block_inject ||
        variant == hc_case::scale_first_aliased_block_inject;
    ggml_tensor * shared = aliased_block_inject
        ? ggml_new_tensor_1d(ctx, type, n_embd*nt) : nullptr;
    ggml_tensor * block = variant == hc_case::overlapping_block
        ? ggml_view_2d(ctx, residual, n_embd, nt, n_embd*sizeof(float), 0)
        : aliased_block_inject
            ? ggml_view_2d(ctx, shared, n_embd, nt, n_embd*sizeof(float), 0)
            : ggml_new_tensor_2d(ctx, type, n_embd, nt);
    ggml_tensor * inject = aliased_block_inject
        ? ggml_view_2d(ctx, shared, hc, nt, hc*sizeof(float), 0)
        : ggml_new_tensor_2d(ctx, type, hc, nt);
    ggml_tensor * scale_in = ggml_scale(ctx, inject,
        variant == hc_case::wrong_input_scale ? 0.5f : 1.0f/(float) hc);
    ggml_tensor * sigmoid = variant == hc_case::wrong_unary
        ? ggml_tanh(ctx, scale_in) : ggml_sigmoid(ctx, scale_in);
    ggml_tensor * scale_out = ggml_scale(ctx, sigmoid,
        variant == hc_case::wrong_output_scale ? 1.0f : 2.0f);
    if (variant == hc_case::marked_intermediate) {
        ggml_set_output(scale_out);
    }
    ggml_tensor * weights = ggml_reshape_3d(ctx, scale_out, 1, hc, nt);
    if (variant == hc_case::marked_weights_view) {
        ggml_set_output(weights);
    }
    ggml_tensor * block_view = ggml_reshape_3d(ctx, block, n_embd, 1, nt);
    ggml_tensor * repeated = ggml_repeat_4d(ctx, block_view, n_embd, hc, nt, 1);
    ggml_tensor * product = ggml_mul(ctx, repeated, weights);
    ggml_tensor * combined =
        (variant == hc_case::inplace_add || variant == hc_case::overlapping_block)
        ? ggml_add_inplace(ctx, residual, product)
        : variant == hc_case::swapped_operands
            ? ggml_add(ctx, product, residual) : ggml_add(ctx, residual, product);
    ggml_tensor * output = combined;
    if (variant == hc_case::extra_consumer) {
        ggml_tensor * extra = ggml_repeat_4d(ctx, weights, n_embd, hc, nt, 1);
        output = ggml_add(ctx, combined, extra);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    const bool scale_first = variant == hc_case::scale_first ||
        variant == hc_case::scale_first_aliased_block_inject;
    if (scale_first) {
        ggml_tensor * reordered[] = {
            scale_in, sigmoid, scale_out, weights, block_view, repeated, product, combined,
        };
        for (int i = 0; i < 8; ++i) {
            graph->nodes[i] = reordered[i];
        }
    }
    int start = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == scale_in) {
            start = i;
            break;
        }
    }
    ggml_cuda_hc_combine_fusion match;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    const bool check_memory_ranges =
        variant == hc_case::inplace_add || variant == hc_case::overlapping_block ||
        aliased_block_inject;
    if (check_memory_ranges) {
        backend = ggml_backend_cpu_init();
        buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    }
    const bool actual = start >= 0 && (!check_memory_ranges || buffer) &&
        ggml_cuda_match_hc_combine(graph, start, match, check_memory_ranges);
    const bool pointers_ok = !actual ||
        (match.residual == residual && match.block == block && match.repeated == repeated &&
         match.inject == inject && match.dst == combined && match.use_repeated_block == !scale_first);
    const int expected_skip = scale_first ? 7 : 5;
    const bool skip_ok = !actual || match.nodes_to_skip == expected_skip;
    if (actual != expected || !pointers_ok || !skip_ok) {
        std::fprintf(stderr, "%s: expected match=%d/skip=%d, got %d/skip=%d; graph:",
            label, expected, expected_skip, actual, actual ? match.nodes_to_skip : -1);
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
    return actual == expected && pointers_ok && skip_ok;
}

int main() {
    bool ok = true;
    ok &= run_case("valid",               hc_case::valid,               true);
    ok &= run_case("scale-first",         hc_case::scale_first,         true);
    ok &= run_case("swapped-operands",    hc_case::swapped_operands,    true);
    ok &= run_case("wrong-input-scale",   hc_case::wrong_input_scale,   false);
    ok &= run_case("wrong-output-scale",  hc_case::wrong_output_scale,  false);
    ok &= run_case("wrong-unary",         hc_case::wrong_unary,         false);
    ok &= run_case("extra-consumer",      hc_case::extra_consumer,      false);
    ok &= run_case("marked-intermediate", hc_case::marked_intermediate, false);
    ok &= run_case("marked-weights-view", hc_case::marked_weights_view, false);
    ok &= run_case("wrong-type",          hc_case::wrong_type,          false);
    ok &= run_case("inplace-add",         hc_case::inplace_add,         true);
    ok &= run_case("overlapping-block",   hc_case::overlapping_block,   true);
    ok &= run_case("aliased-block-inject", hc_case::aliased_block_inject, true);
    ok &= run_case("scale-first-aliased-block-inject",
        hc_case::scale_first_aliased_block_inject, false);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
