#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (!device) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return EXIT_FAILURE;

    constexpr int64_t n_embd = 257;
    constexpr int64_t hc = 4;
    constexpr int64_t nt = 3;
    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd*hc);

    ggml_tensor * rms_fused = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor * reshape_fused = ggml_reshape_2d(ctx, rms_fused, n_embd*hc, nt);
    ggml_tensor * out_fused = ggml_mul(ctx, reshape_fused, gamma);
    ggml_cgraph * graph_fused = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph_fused, out_fused);

    ggml_tensor * rms_reference = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor * reshape_reference = ggml_reshape_2d(ctx, rms_reference, n_embd*hc, nt);
    ggml_set_output(reshape_reference); // Forces the ordinary RMS_NORM then MUL path.
    ggml_tensor * out_reference = ggml_mul(ctx, reshape_reference, gamma);
    ggml_cgraph * graph_reference = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph_reference, out_reference);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return EXIT_FAILURE;

    static const float values[] = {
        0.0f, -0.0f, 0x1p-149f, -0x1p-149f, 1.0f, -1.0f,
        0x1p-12f, -0x1p-12f, INFINITY, -INFINITY, NAN,
    };
    std::vector<float> x_data((size_t) n_embd*hc*nt);
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = values[i % (sizeof(values)/sizeof(values[0]))];
    }
    // Keep one finite row per channel/token and vary gamma by channel so an
    // accidental gamma[col] broadcast cannot pass.
    for (int64_t t = 0; t < nt; ++t) {
        for (int64_t c = 0; c < hc; ++c) {
            if ((t + c) % 2 == 0) {
                for (int64_t col = 0; col < n_embd; ++col) {
                    x_data[(size_t) ((t*hc + c)*n_embd + col)] = (float) ((col % 17) - 8)*0.125f;
                }
            }
        }
    }
    std::vector<float> gamma_data((size_t) n_embd*hc);
    for (int64_t c = 0; c < hc; ++c) {
        for (int64_t col = 0; col < n_embd; ++col) {
            gamma_data[(size_t) (c*n_embd + col)] =
                (c + 1)*0.25f + (float) (col % 7)*0.03125f;
        }
    }
    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size()*sizeof(float));
    ggml_backend_tensor_set(gamma, gamma_data.data(), 0, gamma_data.size()*sizeof(float));

    std::vector<uint32_t> fused((size_t) n_embd*hc*nt);
    std::vector<uint32_t> reference(fused.size());
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
                std::fprintf(stderr, "grouped-RMS mismatch at %zu: fused=0x%08x reference=0x%08x\n",
                    i, fused[i], reference[i]);
                break;
            }
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
