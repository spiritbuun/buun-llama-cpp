#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int64_t k = 5120;
static constexpr int64_t m = 257;

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (!device) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return EXIT_FAILURE;

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);

    ggml_tensor * mm_fused = ggml_mul_mat(ctx, weights, input);
    ggml_tensor * scale_fused = ggml_scale(ctx, mm_fused, 0.25f);
    ggml_tensor * out_fused = ggml_silu(ctx, scale_fused);
    ggml_cgraph * graph_fused = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph_fused, out_fused);

    ggml_tensor * mm_reference = ggml_mul_mat(ctx, weights, input);
    ggml_tensor * scale_reference = ggml_scale(ctx, mm_reference, 0.25f);
    ggml_set_output(scale_reference); // Shared fusion contract forces the ordinary three-kernel path.
    ggml_tensor * out_reference = ggml_silu(ctx, scale_reference);
    ggml_cgraph * graph_reference = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph_reference, out_reference);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return EXIT_FAILURE;

    static const float scales[] = {
        0.0f, -0.0f, INFINITY, -INFINITY, NAN,
        0x1p-24f, -0x1p-24f, 65504.0f, -65504.0f,
    };
    std::vector<block_q8_0> w((size_t) (k/QK8_0)*m);
    for (block_q8_0 & block : w) {
        block.d = ggml_fp32_to_fp16(1.0f);
        std::memset(block.qs, 0, sizeof(block.qs));
    }
    for (int64_t row = 0; row < m; ++row) {
        block_q8_0 & block = w[(size_t) row*(k/QK8_0)];
        block.d = ggml_fp32_to_fp16(scales[row % (sizeof(scales)/sizeof(scales[0]))]);
        block.qs[0] = 1;
    }
    std::vector<float> x((size_t) k, 0.0f);
    x[0] = 1.0f;
    ggml_backend_tensor_set(weights, w.data(), 0, w.size()*sizeof(w[0]));
    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(x[0]));

    std::vector<uint32_t> fused((size_t) m);
    std::vector<uint32_t> reference((size_t) m);
    const ggml_status fused_status = ggml_backend_graph_compute(backend, graph_fused);
    ggml_backend_tensor_get(out_fused, fused.data(), 0, fused.size()*sizeof(fused[0]));
    const ggml_status reference_status = ggml_backend_graph_compute(backend, graph_reference);
    ggml_backend_tensor_get(out_reference, reference.data(), 0, reference.size()*sizeof(reference[0]));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (fused_status != GGML_STATUS_SUCCESS || reference_status != GGML_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (fused != reference) {
        for (size_t i = 0; i < fused.size(); ++i) {
            if (fused[i] != reference[i]) {
                std::fprintf(stderr, "special-value mismatch at %zu: fused=0x%08x reference=0x%08x\n",
                    i, fused[i], reference[i]);
                break;
            }
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
