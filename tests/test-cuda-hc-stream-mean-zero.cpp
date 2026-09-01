#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (!device) {
        std::fprintf(stderr, "CUDA0 unavailable; skipping signed-zero regression\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        return EXIT_FAILURE;
    }

    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    constexpr int64_t n_embd = 31;
    constexpr int64_t hc = 4;
    constexpr int64_t nt = 7;
    ggml_tensor * base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    const size_t stream_bytes = ggml_row_size(base->type, n_embd);
    const size_t token_stride = stream_bytes*hc;
    ggml_tensor * view[hc];
    for (int64_t c = 0; c < hc; ++c) {
        view[c] = ggml_view_2d(ctx, base, n_embd, nt, token_stride, stream_bytes*c);
    }
    ggml_tensor * out = ggml_cont(ctx, view[0]);
    out = ggml_add(ctx, out, view[1]);
    out = ggml_add(ctx, out, view[2]);
    out = ggml_add(ctx, out, view[3]);
    out = ggml_scale(ctx, out, 0.25f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }

    std::vector<uint32_t> input(ggml_nelements(base), UINT32_C(0x80000000));
    ggml_backend_tensor_set(base, input.data(), 0, input.size()*sizeof(input[0]));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<uint32_t> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(output[0]));

    bool ok = status == GGML_STATUS_SUCCESS;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] != UINT32_C(0x00000000)) {
            std::fprintf(stderr, "signed-zero mismatch at %zu: 0x%08x\n", i, output[i]);
            ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
