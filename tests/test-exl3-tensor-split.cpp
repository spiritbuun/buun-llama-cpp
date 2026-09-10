#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

struct split_config {
    int axis;
    int segments;
    bool matrix = false;
};

static ggml_backend_meta_split_state split_weight(const ggml_tensor * t, void * opaque) {
    const auto & c = *static_cast<split_config *>(opaque);
    ggml_backend_meta_split_state s{};
    s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    s.n_segments = 1;
    s.nr[0] = 1;
    int axis = c.axis;
    if (c.matrix) {
        const std::string name = t->name;
        if (name == "x" || name == "suh") {
            if (c.axis != 0 && !(name == "suh" && c.axis == 2)) return s;
            axis = c.axis == 2 ? 1 : 0;
        } else if (name == "svh") {
            if (c.axis == 0) return s;
            axis = c.axis == 2 ? 1 : 0;
        } else if (name != "w") return s;
    }
    s.axis = static_cast<ggml_backend_meta_split_axis>(axis);
    s.n_segments = c.segments < 0 ? 1 : c.segments;
    for (size_t i = 0; i < s.n_segments; ++i) {
        s.nr[i] = c.segments < 0 ? -c.segments : 1;
        s.ne[2*i] = s.ne[2*i+1] = t->ne[axis] / (2*std::abs(c.segments));
    }
    return s;
}

static bool check_bytes(ggml_backend_dev_t * devices, int axis, int segments, ggml_type type, bool async) {
    split_config config{axis, segments};
    auto * backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, split_weight, &config), nullptr);
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, type, 1024, 512, 4);
    const size_t tile_bytes = 32 * ggml_exl3_bits(type);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<unsigned char> bytes(ggml_nbytes(w));
    unsigned state = 42;
    for (auto & b : bytes) { state = state * 1664525u + 1013904223u; b = state >> 24; }
    auto set = [&](const void * data, size_t offset, size_t size) {
        if (async) ggml_backend_tensor_set_async(backend, w, data, offset, size);
        else ggml_backend_tensor_set(w, data, offset, size);
        ggml_backend_synchronize(backend);
    };
    auto get = [&](void * data, size_t offset, size_t size) {
        if (async) ggml_backend_tensor_get_async(backend, w, data, offset, size);
        else ggml_backend_tensor_get(w, data, offset, size);
        ggml_backend_synchronize(backend);
    };
    set(bytes.data(), 0, bytes.size());
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            // Deliberately crosses both packed-tile and shard boundaries.
            const size_t start = 137, count = bytes.size()/2 + 19;
            for (size_t i = start; i < start + count; ++i) bytes[i] ^= 0xa5;
            set(bytes.data()+start, start, count);
        }
        for (int rank = 0; rank < 2; ++rank) {
            auto * shard = ggml_backend_meta_buffer_simple_tensor(w, rank);
            std::vector<unsigned char> actual(ggml_nbytes(shard)), expected;
            ggml_backend_tensor_get(shard, actual.data(), 0, actual.size());
            // Independent oracle: select complete 16x16 tiles in storage order.
            for (int expert = 0; expert < 4; ++expert) {
                for (int nt = 0; nt < 512/16; ++nt) {
                    for (int kt = 0; kt < 1024/16; ++kt) {
                        const int coord = axis == 0 ? kt*16 : axis == 1 ? nt*16 : expert;
                        const int piece = int(w->ne[axis]) / (2*std::abs(segments));
                        if ((coord / piece) % 2 != rank) continue;
                        const size_t pos = ((size_t(expert)*32 + nt)*64 + kt)*tile_bytes;
                        expected.insert(expected.end(), bytes.begin()+pos, bytes.begin()+pos+tile_bytes);
                    }
                }
            }
            if (actual != expected) {
                printf("axis=%d segments=%d pass=%d rank=%d SHARD MISMATCH\n", axis, segments, pass, rank);
                ok = false;
            }
        }
        std::vector<unsigned char> back(bytes.size());
        get(back.data(), 0, back.size());
        if (back != bytes) { printf("ROUNDTRIP MISMATCH\n"); ok = false; }
        if (axis == 0) {
            std::vector<unsigned char> part(4121);
            get(part.data(), 131, part.size());
            if (!std::equal(part.begin(), part.end(), bytes.begin()+131)) { printf("PARTIAL READ MISMATCH\n"); ok = false; }
        }
        // Partial transfer contract is qualified for the new tiled column path.
        if (axis != 0) break;
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("bytes axis=%d segments=%d: %s\n", axis, segments, ok ? "PASS" : "FAIL");
    return ok;
}

struct matrix_shape {
    int k = 1024, n = 512, experts = 4, used = 2;
    bool per_expert_input = false;
};

static std::vector<float> matrix_run(ggml_backend_t backend, int axis, int tokens, ggml_type type, matrix_shape shape) {
    const bool experts = axis == 2;
    const int k = shape.k, n = shape.n, ne = experts ? shape.experts : 1;
    const int used = shape.used, input_rows = experts && shape.per_expert_input ? used : 1;
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    auto * compute = ggml_init({4*1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, type, k, n, ne);
    auto * x = experts ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, input_rows, tokens) :
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    auto * svh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, ne);
    auto * suh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, ne);
    ggml_set_name(w, "w"); ggml_set_name(x, "x");
    ggml_set_name(svh, "svh"); ggml_set_name(suh, "suh");
    ggml_tensor * ids = nullptr, * router = nullptr, * y;
    if (experts) {
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
        ggml_set_name(ids, "ids");
        y = ggml_mul_mat_id(compute, w, x, ids);
        y->src[3] = svh; y->src[4] = suh;
        router = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(router, "router");
        y = ggml_mul(compute, y, router); // same partial-sum boundary as routed MoE graphs
        y = ggml_sum_rows(compute, ggml_cont(compute, ggml_permute(compute, y, 1, 0, 2, 3)));
    } else {
        y = ggml_mul_mat(compute, w, x);
        y->src[2] = svh; y->src[3] = suh;
    }
    if (axis != 1) y = ggml_scale(compute, y, 1.0f); // consume the reduced partial result
    ggml_set_output(y);
    auto * graph = ggml_new_graph_custom(compute, 64, false);
    ggml_build_forward_expand(graph, y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) std::abort();
    std::vector<unsigned char> bytes(ggml_nbytes(w));
    unsigned state = 314;
    for (auto & b : bytes) { state = state * 1664525u + 1013904223u; b = state >> 24; }
    ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());
    std::vector<float> input(k*input_rows*tokens);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i)*0.7f)*0.2f;
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    std::vector<ggml_fp16_t> scales(std::max(k,n)*ne);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = ggml_fp32_to_fp16((i%3 == 0 ? -1 : 1)*0.125f);
    ggml_backend_tensor_set(suh, scales.data(), 0, ggml_nbytes(suh));
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = ggml_fp32_to_fp16((i%7 == 0 ? -1 : 1)*(0.5f + float(i%13)*0.125f));
    ggml_backend_tensor_set(svh, scales.data(), 0, ggml_nbytes(svh));
    if (ids) {
        const float one = 1.0f;
        ggml_backend_tensor_set(router, &one, 0, sizeof(one));
        std::vector<int32_t> selected(used*tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int u = 0; u < used; ++u) selected[used*t+u] = (t*17+u*31)%ne;
        }
        ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));
    }
    std::vector<float> result(ggml_nelements(y)), first;
    for (int pass = 0; pass < 4; ++pass) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
        ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
        if (pass == 0) first = result;
        else if (std::memcmp(first.data(), result.data(), ggml_nbytes(y))) {
            fprintf(stderr, "same-layout repeat mismatch, pass=%d\n", pass);
            std::abort();
        }
    }
    ggml_backend_synchronize(backend);
    ggml_gallocr_free(allocator); ggml_free(compute);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return result;
}

static bool check_matrix(ggml_backend_dev_t * devices, int axis, int tokens, ggml_type type, matrix_shape shape) {
    split_config config{axis, 1, true};
    auto * single = ggml_backend_dev_init(devices[0], nullptr);
    auto reference = matrix_run(single, axis, tokens, type, shape);
    ggml_backend_free(single);
    auto * meta = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, split_weight, &config), nullptr);
    auto actual = matrix_run(meta, axis, tokens, type, shape);
    ggml_backend_free(meta);
    double err2 = 0, ref2 = 0, maxerr = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) return false;
        const double err = double(actual[i]) - reference[i];
        err2 += err*err; ref2 += double(reference[i])*reference[i];
        maxerr = std::max(maxerr, std::abs(err));
    }
    const double rel = std::sqrt(err2/std::max(ref2, 1e-30));
    printf("matrix type=%s axis=%d tokens=%d rel_l2=%.9g max_abs=%.9g\n", ggml_type_name(type), axis, tokens, rel, maxerr);
    // Qualify the matrix with F32 communication (CTest sets AR1_BF16=0).
    // The separate model panel measures default BF16 communication as well.
    return rel < 5e-5;
}

int main(int argc, char ** argv) {
    auto * reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) < 2) return 77;
    ggml_backend_dev_t devices[] = {ggml_backend_reg_dev_get(reg, 0), ggml_backend_reg_dev_get(reg, 1)};
    if (argc > 1 && std::string(argv[1]) == "matrix") {
        if (argc < 4) return 2;
        matrix_shape shape;
        if (argc >= 11) shape = {std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]),
                                std::atoi(argv[9]), std::atoi(argv[10]) != 0};
        return check_matrix(devices, std::atoi(argv[2]), std::atoi(argv[3]),
                            ggml_exl3_type(argc > 4 ? std::atoi(argv[4]) : 4, argc > 5 ? std::atoi(argv[5]) : 2), shape) ? 0 : 1;
    }
    return check_bytes(devices, argc > 1 ? std::atoi(argv[1]) : 0, argc > 2 ? std::atoi(argv[2]) : 1,
                       ggml_exl3_type(argc > 3 ? std::atoi(argv[3]) : 4, argc > 4 ? std::atoi(argv[4]) : 2),
                       argc > 5 && std::string(argv[5]) == "async") ? 0 : 1;
}
