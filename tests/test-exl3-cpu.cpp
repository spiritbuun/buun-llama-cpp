#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "test-exl3-gpu-common.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

#ifdef EXL3_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

static ggml_backend_t cache_gpu = nullptr;
static bool gpu_executor = false;
static bool turing_executor = false;
static decltype(ggml_moe_cache.dispatch) real_dispatch;
static decltype(ggml_moe_cache.collect) real_collect;
static int cache_hits = 0;
static int last_cache_hits = 0;
static int dispatch_attempts = 0;
static int collect_attempts = 0;
static bool reject_dispatch = false;
static bool reject_collect = false;

static int tracked_dispatch(void * node, int type, int64_t k, int64_t n, int hits,
                            const int32_t * slots, const float * const * acts) {
    ++dispatch_attempts;
    if (reject_dispatch) return 0;
    const int ok = real_dispatch(node, type, k, n, hits, slots, acts);
    if (ok) {
        cache_hits += hits;
        last_cache_hits = hits;
    }
    return ok;
}

static int tracked_collect(void * node, int hits, float * const * rows, int64_t n) {
    ++collect_attempts;
    const int ok = real_collect(node, hits, rows, n);
    return reject_collect ? 0 : ok;
}

static float half_round(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
static float half_bits(unsigned x) { return ggml_fp16_to_fp32(ggml_fp16_t(x)); }

// Independent scalar oracle: individual stream bits, not the executor's
// packed-word extraction; double dot products, not its float accumulation.
static float codebook(unsigned x, int cb) {
    if (cb == 2) {
        x *= 0x83dcd12du;
        unsigned s = 0x6400;
        for (int i = 0; i < 4; ++i) s += (x >> (8*i)) & 255;
        return half_round(std::fma(half_bits(s), half_bits(0x1eee), half_bits(0xc931)));
    }
    x = cb == 1 ? x*0xcbac1fedu : x*89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return half_round(half_bits(x & 65535) + half_bits(x >> 16));
}

static void had(float * x, int n) {
    for (int b = 0; b < n; b += 128)
        for (int d = 1; d < 128; d *= 2)
            for (int i = 0; i < 128; i += 2*d)
                for (int j = 0; j < d; ++j) {
                    const float a = x[b+i+j], z = x[b+i+j+d];
                    x[b+i+j] = a+z;
                    x[b+i+j+d] = a-z;
                }
}

// Decodes one 16x16 trellis tile into out[col*stride + row].
static void decode_tile(const unsigned char * tile, int bits, int cb, float * out, int stride) {
    for (int t = 0; t < 256; ++t) {
        unsigned window = 0;
        for (int b = 0; b < 16; ++b) {
            const int p = ((t+257)*bits-1-b) % (256*bits);
            const int bit = (p/32)*32 + 31-p%32;
            window |= unsigned((tile[bit/8] >> (bit%8)) & 1) << b;
        }
        const int lane = t/8, j = t%8;
        const int row = (lane%4)*2 + j%2 + ((j&2) ? 8 : 0);
        const int col = lane/4 + ((j&4) ? 8 : 0);
        out[col*stride + row] = codebook(window, cb);
    }
}

// Mixed banks keep each expert's trellis unchanged. Exercise non-contiguous
// logical IDs, singleton groups, absent groups and changed routes on reuse.
static bool run_mixed(ggml_backend_t backend, int tokens, int lanes) {
    constexpr int k = 128, n = 128, topk = 3, experts = 5;
    constexpr float norm = 0.088388347648f;
    const std::vector<std::vector<int>> members{{3, 0}, {4, 1}, {2}};
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, lanes, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens);
    auto * flat = ggml_reshape_1d(ctx, ids, topk*tokens);
    ggml_tensor * result = nullptr;
    std::vector<ggml_tensor *> weights, signs, scales, maps;
    for (int g = 0; g < 3; ++g) {
        weights.push_back(ggml_new_tensor_3d(ctx, ggml_exl3_type(g+2, 2), k, n, members[g].size()));
        signs.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, members[g].size()));
        scales.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, members[g].size()));
        maps.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, experts));
        auto * local = ggml_reshape_2d(ctx, ggml_get_rows(ctx, maps.back(), flat), topk, tokens);
        auto * part = ggml_mul_mat_id(ctx, weights.back(), x, local);
        part->src[3] = scales.back();
        part->src[4] = signs.back();
        ggml_mul_mat_id_set_expert_window(part, 0, members[g].size());
        result = result ? ggml_add(ctx, result, part) : part;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    unsigned rng = 1234567;
    auto next = [&]() { rng = rng*1664525u+1013904223u; return rng; };
    std::vector<float> dense(experts*k*n), input(k*lanes*tokens);
    std::vector<ggml_fp16_t> all_signs(experts*k), all_scales(experts*n);
    for (auto & v : input) v = float(next() >> 8)/16777216.0f - 0.5f;
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    for (int g = 0; g < 3; ++g) {
        const int bits = g+2;
        std::vector<uint8_t> packed(ggml_nbytes(weights[g]));
        for (auto & v : packed) v = next() >> 24;
        std::vector<ggml_fp16_t> sg(k*members[g].size()), sc(n*members[g].size());
        std::vector<int32_t> map(experts, -1);
        for (size_t e = 0; e < members[g].size(); ++e) {
            const int id = members[g][e];
            map[id] = e;
            for (int i = 0; i < k; ++i) all_signs[id*k+i] = sg[e*k+i] = ggml_fp32_to_fp16((i+id)%3 ? 1 : -1);
            for (int i = 0; i < n; ++i) all_scales[id*n+i] = sc[e*n+i] = ggml_fp32_to_fp16(((i+id)%7 ? 1 : -1)*0.125f);
            for (int nt = 0; nt < n/16; ++nt) for (int kt = 0; kt < k/16; ++kt) {
                const size_t base = e*weights[g]->nb[2] + size_t(nt*(k/16)+kt)*32*bits;
                decode_tile(packed.data() + base, bits, 2, dense.data() + size_t(id*n+nt*16)*k + kt*16, k);
            }
        }
        ggml_backend_tensor_set(weights[g], packed.data(), 0, packed.size());
        ggml_backend_tensor_set(signs[g], sg.data(), 0, ggml_nbytes(signs[g]));
        ggml_backend_tensor_set(scales[g], sc.data(), 0, ggml_nbytes(scales[g]));
        ggml_backend_tensor_set(maps[g], map.data(), 0, ggml_nbytes(maps[g]));
    }
    bool ok = true;
    double worst = 0;
    for (int round = 0; round < 3; ++round) {
        std::vector<int32_t> routes(topk*tokens);
        const int first_routes[] = {0, 3, 1}; // the four-bit group receives no rows
        for (size_t i = 0; i < routes.size(); ++i) routes[i] = round == 0 ? first_routes[i%topk] : (i*3+round)%experts;
        ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
        std::vector<float> reference(n*topk*tokens), actual(reference.size()), xr(k);
        for (int t = 0; t < tokens; ++t) for (int e = 0; e < topk; ++e) {
            const int id = routes[t*topk+e];
            for (int i = 0; i < k; ++i) xr[i] = input[(t*lanes+e%lanes)*k+i]*half_bits(all_signs[id*k+i]);
            had(xr.data(), k);
            for (auto & v : xr) v = half_round(v*norm);
            auto * out = reference.data()+(t*topk+e)*n;
            for (int c = 0; c < n; ++c) {
                double sum = 0;
                for (int i = 0; i < k; ++i) sum += double(xr[i])*dense[(id*n+c)*k+i];
                out[c] = sum;
            }
            had(out, n);
            for (int c = 0; c < n; ++c) out[c] *= norm*half_bits(all_scales[id*n+c]);
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
        double err2 = 0, ref2 = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            ok &= std::isfinite(actual[i]);
            const double d = actual[i]-reference[i];
            err2 += d*d;
            ref2 += double(reference[i])*reference[i];
        }
        worst = std::max(worst, std::sqrt(err2/std::max(ref2, 1e-30)));
    }
    ok &= worst < (gpu_executor ? 1e-2 : 2e-5);
    printf("mixed tokens=%d lanes=%d rel_l2=%.9g %s\n", tokens, lanes, worst, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool run(ggml_backend_t backend, int bits, int cb, bool grouped, int tokens, int lanes, bool windowed = false,
                int k = 256, int n = 384, int overlap = -1, bool automatic = false, bool head = false) {
    constexpr float norm = 0.088388347648f;
    // The cache's existing pool policy requires at least 64 expert entries.
    const int experts = grouped ? (cache_gpu ? 64 : 3) : 1;
    const int topk = grouped ? 3 : 1;
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, ggml_exl3_type(bits, cb), k, n, experts);
    const int input_stride = k + (gpu_executor && !grouped ? 0 : 16);
    auto * backing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_stride, lanes, tokens);
    auto * x = ggml_view_3d(ctx, backing, k, lanes, tokens, backing->nb[1], backing->nb[2], 0);
    auto * suh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, experts);
    auto * svh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, experts);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens);
    ggml_tensor * y;
    if (grouped) {
        y = ggml_mul_mat_id(ctx, w, x, ids);
        y->src[3] = svh;
        y->src[4] = suh;
        if (windowed) ggml_mul_mat_id_set_expert_window(y, 1, experts);
    } else {
        x = ggml_view_2d(ctx, backing, k, tokens, backing->nb[2], 0);
        y = ggml_mul_mat(ctx, w, x);
        y->src[2] = svh;
        y->src[3] = suh;
    }
    ggml_set_name(w, head ? "output.weight" : "blk.0.ffn_up_exps.weight");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_supports_op(backend, y)) {
        printf("EXL3_UNSUPPORTED backend=%s bits=%d cb=%d grouped=%d\n", ggml_backend_name(backend), bits, cb, grouped);
        ggml_free(ctx);
        return false;
    }
    // Capability probes must decline malformed auxiliaries and geometry,
    // while still admitting the loader's pre-attachment probe.
    if (!gpu_executor) {
        const int aux = grouped ? 3 : 2;
        y->src[aux] = nullptr;
        GGML_ASSERT(!ggml_backend_supports_op(backend, y));
        y->src[aux+1] = nullptr;
        GGML_ASSERT(ggml_backend_supports_op(backend, y));
        y->src[aux] = svh;
        y->src[aux+1] = suh;
        svh->ne[0]--;
        GGML_ASSERT(!ggml_backend_supports_op(backend, y));
        svh->ne[0]++;
        x->type = GGML_TYPE_F16;
        GGML_ASSERT(!ggml_backend_supports_op(backend, y));
        x->type = GGML_TYPE_F32;
        w->ne[0] -= 16;
        GGML_ASSERT(!ggml_backend_supports_op(backend, y));
        w->ne[0] += 16;
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    unsigned rng = 1234567;
    auto next = [&]() { rng = rng*1664525u+1013904223u; return rng; };
    std::vector<unsigned char> packed(ggml_nbytes(w));
    for (auto & b : packed) b = next() >> 24;
    std::vector<float> input(ggml_nelements(backing));
    for (auto & v : input) v = float(next() >> 8)/16777216.0f - 0.5f;
    std::vector<ggml_fp16_t> signs(k*experts), scales(n*experts);
    for (int i = 0; i < k*experts; ++i) signs[i] = ggml_fp32_to_fp16(i%3 ? 1 : -1);
    for (int i = 0; i < n*experts; ++i) scales[i] = ggml_fp32_to_fp16((i%7 ? 1 : -1)*0.125f);
    std::vector<int32_t> routes(topk*tokens);
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = int(i%2); // repeated / unsorted IDs
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(backing, input.data(), 0, ggml_nbytes(backing));
    ggml_backend_tensor_set(suh, signs.data(), 0, ggml_nbytes(suh));
    ggml_backend_tensor_set(svh, scales.data(), 0, ggml_nbytes(svh));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    std::vector<float> weights(k*n*std::min(experts, 3));
    // Only the first two experts are routed; no need to decode unused entries.
    for (int e = 0; e < std::min(experts, 3); ++e)
        for (int nt = 0; nt < n/16; ++nt) for (int kt = 0; kt < k/16; ++kt) {
            const size_t base = e*w->nb[2] + size_t(nt*(k/16)+kt)*32*bits;
            decode_tile(packed.data() + base, bits, cb, weights.data() + size_t(e*n+nt*16)*k + kt*16, k);
        }
    std::vector<float> ref(ggml_nelements(y)), xr(k), actual(ref.size()), first;
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < topk; ++slot) {
        const int e = grouped ? routes[token*topk+slot] - (windowed ? 1 : 0) : 0;
        if (e < 0) continue; // the reference's pre-zeroed row belongs to another rank
        const float * in = input.data() + (token*lanes+slot%lanes)*input_stride;
        for (int i = 0; i < k; ++i) xr[i] = in[i]*half_bits(signs[e*k+i]);
        had(xr.data(), k);
        for (auto & v : xr) v = half_round(v*norm);
        float * out = ref.data() + (token*topk+slot)*n;
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int i = 0; i < k; ++i) sum += double(xr[i])*weights[(e*n+col)*k+i];
            out[col] = float(sum);
        }
        had(out, n);
        for (int i = 0; i < n; ++i) out[i] = out[i]*norm*half_bits(scales[e*n+i]);
    }
    bool ok = true;
    double worst = 0;
    for (int threads : {1, 4, 8, 8}) {
        if (!gpu_executor) ggml_backend_cpu_set_n_threads(backend, threads);
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
        if (first.empty()) first = actual;
        else ok &= std::memcmp(first.data(), actual.data(), ggml_nbytes(y)) == 0;
        double err2 = 0, ref2 = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            ok &= std::isfinite(actual[i]);
            const double d = actual[i]-ref[i];
            err2 += d*d;
            ref2 += double(ref[i])*ref[i];
        }
        worst = std::max(worst, std::sqrt(err2/std::max(ref2, 1e-30)));
    }
    // Standalone GPU mul1 uses quantized activations, unlike the exact CPU/cache
    // path. Its separate batch/policy gates require exact repeated execution.
    const bool quantized_activations = gpu_executor && !head &&
        (grouped || tokens <= (turing_executor ? 8 : 16));
    ok &= worst < (quantized_activations ? 1e-2 : 2e-5);
    // The cache provider accepts at most ten tokens; larger cases above still
    // exercise CPU transform sharing/fallback and exact thread-count agreement.
    if (cache_gpu && grouped && tokens <= 10 && topk*tokens <= 64) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_moe_cache_config config{};
        // Keep room for the minimum 64-entry pool at the larger MoE shapes too.
        const int cache_mib = std::max(16, int((ggml_nbytes(w) + (1 << 20) - 1) >> 20) + 8);
        GGML_ASSERT(ggml_moe_cache.query_config(automatic ? 1 : 0, automatic ? 0 : cache_mib, &config));
        if (!automatic) {
            config.reserve_bytes = 0;
            config.reserve_explicit = 1;
            config.min_expert_bytes = 1;
            config.min_expert_explicit = 1;
            config.minimum_slab_bytes = 1 << 20;
            config.min_devices = 1;
        }
        config.expert_parallel = 0;
        config.overlap_cpu_rows = overlap;
        void * backends[] = {cache_gpu, backend};
        void * session = ggml_moe_cache.session_create(backends, 2, &config);
        GGML_ASSERT(session);
        ggml_moe_cache.session_enter(session);
        real_dispatch = ggml_moe_cache.dispatch;
        real_collect = ggml_moe_cache.collect;
        ggml_moe_cache.dispatch = tracked_dispatch;
        ggml_moe_cache.collect = tracked_collect;
        cache_hits = last_cache_hits = dispatch_attempts = collect_attempts = 0;
        auto check = [&] {
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
            return std::memcmp(first.data(), actual.data(), ggml_nbytes(y)) == 0;
        };
        // Auto EXL3 must retain all resident rows on GPU; fixed overrides
        // still assign the requested share to CPU. Wait for complete residency.
        const int valid_rows = std::count_if(routes.begin(), routes.end(), [&](int32_t id) {
            return !windowed || (id >= 1 && id < 1 + experts);
        });
        const int expected_hits = valid_rows - (overlap < 0 ? 0 : std::min(overlap, valid_rows - 1));
        for (int attempt = 0; attempt < 160 && (cache_hits < 24 || last_cache_hits != expected_hits); ++attempt) {
            ok &= check();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ok &= cache_hits >= 24 && last_cache_hits == expected_hits;
        for (int threads : {1, 4, 8}) {
            ggml_backend_cpu_set_n_threads(backend, threads);
            ok &= check();
            const int before_dispatch = dispatch_attempts;
            reject_dispatch = true;
            ok &= check();
            reject_dispatch = false;
            ok &= dispatch_attempts > before_dispatch;
            const int before_collect = collect_attempts;
            reject_collect = true;
            ok &= check();
            reject_collect = false;
            ok &= collect_attempts > before_collect;
            ok &= check();
            printf("cache bits=%d cb=%d threads=%d auto=%d hits=%d dispatch_failure=%d collect_failure=%d pass=%d\n",
                bits, cb, threads, automatic, cache_hits, dispatch_attempts > before_dispatch, collect_attempts > before_collect, ok);
        }
        ggml_moe_cache.dispatch = real_dispatch;
        ggml_moe_cache.collect = real_collect;
        ggml_moe_cache.session_leave(session);
        ggml_moe_cache.session_destroy(session);
    }
    printf("bits=%d cb=%d grouped=%d tokens=%d lanes=%d window=%d rel_l2=%.9g %s\n",
           bits, cb, grouped, tokens, lanes, windowed, worst, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

#ifdef EXL3_TEST_CUDA
bool test_cuda_exl3_sm86_image();
#endif

int main(int argc, char ** argv) {
    const bool head = argc == 2 && std::strcmp(argv[1], "--gpu-head") == 0;
    gpu_executor = head || (argc == 2 && std::strcmp(argv[1], "--gpu") == 0);
#ifdef EXL3_TEST_CUDA
    if (gpu_executor) {
        cudaDeviceProp props{};
        turing_executor = cudaGetDeviceProperties(&props, 0) == cudaSuccess &&
            props.major == 7 && props.minor == 5;
    }
#endif
    if (head) {
#ifdef EXL3_TEST_CUDA
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) != cudaSuccess || props.major != 8 || props.minor != 6 ||
            !test_cuda_exl3_sm86_image()) return 77;
#else
        return 77;
#endif
    }
    if (argc == 2 && std::strcmp(argv[1], "--cache") == 0) {
        ggml_backend_load_all();
        cache_gpu = ggml_backend_init_by_name("CUDA0", nullptr);
        if (!cache_gpu) cache_gpu = ggml_backend_init_by_name("ROCm0", nullptr);
        if (!cache_gpu || !ggml_moe_cache.session_create) return 77;
    }
    ggml_backend_t backend;
    if (gpu_executor) {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name("CUDA0", nullptr);
        if (!backend) backend = ggml_backend_init_by_name("ROCm0", nullptr);
        if (!backend) return 77;
        if (!exl3_gpu_supported(backend)) {
            ggml_backend_free(backend);
            return 77;
        }
    } else {
        backend = ggml_backend_cpu_init();
    }
    GGML_ASSERT(backend);
    bool ok = true;
    if (head) {
        // Independently decode stream bits and use double dot products. Unlike
        // row-vs-batch parity, this catches consistent decode/MMA permutations.
        // K128 leaves half the K partitions idle; K640 has a partial partition.
        for (int k : {128, 384, 640}) for (int m : {1, 8, 9, 13}) {
            ok &= run(backend, 6, 2, false, m, 1, false, k, 131072, -1, false, true);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    for (int tokens : {1, 7, 700}) for (int lanes : {1, 3}) ok &= run_mixed(backend, tokens, lanes);
    for (int cb = 0; cb < 3; ++cb) for (int bits = 1; bits <= 8; ++bits) {
        ok &= run(backend, bits, cb, false, 1, 1);
        ok &= run(backend, bits, cb, false, 8, 1);
        ok &= run(backend, bits, cb, true, 2, 1);
        ok &= run(backend, bits, cb, true, 2, 3);
        ok &= run(backend, bits, cb, true, 2, 3, true);
    }
    // Shared-transform boundary and per-worker fallback, including rank-local
    // absent experts and multiple output tiles per worker at real MoE shapes.
    ok &= run(backend, 2, 2, false, 64, 1);
    ok &= run(backend, 2, 2, false, 65, 1);
    if (gpu_executor) {
        // Dense tiled prefill: admission boundary, partial token tiles, and
        // the short/long tile crossover. Compare to the independent scalar
        // oracle and require identical repeated execution, as above.
        for (int tokens : {9, 13, 16, 17, 18, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257}) {
            ok &= run(backend, 4, 2, false, tokens, 1);
        }
        ok &= run(backend, 4, 2, false, 129, 1, false, 5120, 640);
        // Wider output exercises the three-CTA SM86 GEMM variant on a 3090;
        // the narrow case above covers the two-CTA variant and partial M tiles.
        ok &= run(backend, 4, 2, false, 129, 1, false, 128, 6144);
        // Wide short batches select BM128; exercise both ragged edges.
        ok &= run(backend, 4, 2, false, 65, 1, false, 128, 8192);
        ok &= run(backend, 4, 2, false, 127, 1, false, 128, 8192);
    }
    ok &= run(backend, 2, 2, true, 21, 1);
    ok &= run(backend, 2, 2, true, 22, 3);
    ok &= run(backend, 2, 2, true, 22, 3, true);
    // Large batches group rows by expert; cover full and ragged row chunks
    // for every in-register codebook and vectorized bit width.
    for (int cb = 0; cb < 2; ++cb) for (int bits = 1; bits <= 4; ++bits) {
        ok &= run(backend, bits, cb, true, 27, 3, bits == 3);
    }
    ok &= run(backend, 2, 2, true, 2, 1, false, 2560, 640);
    ok &= run(backend, 2, 2, true, 2, 3, false, 640, 2560);
    if (cache_gpu) {
        ok &= run(backend, 2, 2, true, 1, 1, false, 2560, 640, 1);
        ok &= run(backend, 3, 0, true, 2, 3, false, 256, 384, 2);
        ok &= run(backend, 2, 2, true, 1, 1, false, 2560, 640, -1, true);
    }
    ggml_backend_free(backend);
    if (cache_gpu) ggml_backend_free(cache_gpu);
    return ok ? 0 : 1;
}
