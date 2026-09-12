#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool supports_graph(ggml_backend_t backend, ggml_context * ctx) {
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor->op != GGML_OP_NONE && !ggml_backend_supports_op(backend, tensor)) {
            return false;
        }
    }
    return true;
}

static void set_tcq_data(ggml_tensor * tensor, uint8_t salt) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const size_t block_size = ggml_type_size(tensor->type);
    const ggml_fp16_t scale = ggml_fp32_to_fp16(1.0f);

    for (size_t off = 0, block = 0; off < bytes.size(); off += block_size, ++block) {
        std::memcpy(bytes.data() + off, &scale, sizeof(scale));
        for (size_t i = sizeof(scale); i < block_size; ++i) {
            bytes[off + i] = (uint8_t) (17*block + 29*i + salt);
        }
    }
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

static int test_pair(ggml_backend_t backend, ggml_type type_k, ggml_type type_v, int64_t gqa, int64_t n_decode) {
    constexpr int64_t d = 256;
    // Match the production Qwen decode geometry on RDNA. A ragged, unmasked fixture selects an
    // unsupported ncols2=1 specialization there instead of the fused path this test targets.
    constexpr int64_t kv = 256;
    constexpr int64_t n_head_kv = 4;
    // Deliberately above the <=4 fused-decode cutoff: this is the materialized reference arm.
    const int64_t n_prefill = 8*n_decode;

    const ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return EXIT_FAILURE;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_decode, n_head_kv*gqa, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k, d, kv, n_head_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v, d, kv, n_head_kv, 1);
    ggml_tensor * mask_decode = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, n_decode);

    ggml_tensor * out_decode = ggml_flash_attn_ext(ctx, q, k, v, mask_decode, 1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_decode, GGML_PREC_F32);

    ggml_tensor * q_prefill = ggml_repeat_4d(ctx, q, d, n_prefill, n_head_kv*gqa, 1);
    ggml_tensor * mask_prefill = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, n_prefill);
    ggml_tensor * out_prefill = ggml_flash_attn_ext(ctx, q_prefill, k, v, mask_prefill, 1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_prefill, GGML_PREC_F32);
    ggml_tensor * out_prefill_first = ggml_view_4d(ctx, out_prefill,
        out_decode->ne[0], out_decode->ne[1], out_decode->ne[2], 1,
        out_prefill->nb[1], out_prefill->nb[2], out_prefill->nb[3], 0);
    ggml_tensor * difference = ggml_sub(ctx, out_decode, out_prefill_first);

    if (!supports_graph(backend, ctx)) {
        std::fprintf(stderr, "%s/%s is not supported by the selected backend\n",
            ggml_type_name(type_k), ggml_type_name(type_v));
        ggml_free(ctx);
        return EXIT_FAILURE;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, difference);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return EXIT_FAILURE;
    }

    std::vector<float> q_data((size_t) ggml_nelements(q));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.5f*std::sin((float) i*0.03125f) + 0.25f*std::cos((float) i*0.0078125f);
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(q_data[0]));
    set_tcq_data(k, 11);
    set_tcq_data(v, 73);
    std::vector<ggml_fp16_t> mask_data((size_t) ggml_nelements(mask_prefill), ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(mask_decode, mask_data.data(), 0, ggml_nbytes(mask_decode));
    ggml_backend_tensor_set(mask_prefill, mask_data.data(), 0, ggml_nbytes(mask_prefill));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> result((size_t) ggml_nelements(difference));
    std::vector<float> output((size_t) ggml_nelements(out_decode));
    ggml_backend_tensor_get(difference, result.data(), 0, result.size()*sizeof(result[0]));
    ggml_backend_tensor_get(out_decode, output.data(), 0, output.size()*sizeof(output[0]));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (status != GGML_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }
    float max_output_abs = 0.0f;
    for (float value : output) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "%s/%s produced a non-finite output\n",
                ggml_type_name(type_k), ggml_type_name(type_v));
            return EXIT_FAILURE;
        }
        max_output_abs = std::max(max_output_abs, std::abs(value));
    }
    if (max_output_abs <= 1e-6f) {
        std::fprintf(stderr, "%s/%s produced only trivial output\n",
            ggml_type_name(type_k), ggml_type_name(type_v));
        return EXIT_FAILURE;
    }
    float max_abs = 0.0f;
    for (float value : result) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "%s/%s produced a non-finite difference\n",
                ggml_type_name(type_k), ggml_type_name(type_v));
            return EXIT_FAILURE;
        }
        max_abs = std::max(max_abs, std::abs(value));
    }
    if (max_abs > 5e-3f) {
        std::fprintf(stderr, "%s/%s fused-vs-materialized max difference %.9g\n",
            ggml_type_name(type_k), ggml_type_name(type_v), max_abs);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(GGML_CUDA_NAME "0");
    if (!device) {
        return 77;
    }
    const char * fused = std::getenv("GGML_TURBO_MMA_FUSED");
    if (fused && std::atoi(fused) == 0) {
        std::fprintf(stderr, "GGML_TURBO_MMA_FUSED disables the fused path this test requires\n");
        return EXIT_FAILURE;
    }
    // A runtime FUSED=1 cannot enable dispatch that was compiled out. Otherwise this
    // test can silently compare two materialized paths and miss fused-kernel bugs.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    bool turbo_fa = false;
    if (get_features) {
        for (const ggml_backend_feature * feature = get_features(reg); feature && feature->name; ++feature) {
            if (std::strcmp(feature->name, "TURBO_FA") == 0 && std::strcmp(feature->value, "1") == 0) {
                turbo_fa = true;
            }
        }
    }
    if (!turbo_fa) {
        std::fprintf(stderr, "Turbo flash-attention dispatch was not compiled into the selected backend\n");
        return EXIT_FAILURE;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        return EXIT_FAILURE;
    }

    const ggml_type pairs[][2] = {
        { GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ },
        { GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ },
        { GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ },
        { GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ },
    };
    int result = EXIT_SUCCESS;
    // Cover both head-group geometries and the full fused decode/speculative batch range.
    for (int64_t gqa : { 6, 16 }) {
        for (int64_t n_decode : { 1, 2, 3, 4 }) {
            for (const auto & pair : pairs) {
                const int pair_result = test_pair(backend, pair[0], pair[1], gqa, n_decode);
                if (pair_result != EXIT_SUCCESS) {
                    std::fprintf(stderr, "failed with GQA ratio %lld, decode batch %lld\n",
                        (long long) gqa, (long long) n_decode);
                    result = EXIT_FAILURE;
                }
            }
        }
    }

    ggml_backend_free(backend);
    return result;
}
