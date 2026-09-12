#include "exl3.h"
#include "ggml-cpu-impl.h"
#include "../ggml-backend-moe-cache.h"

#include <cmath>
#include <cstring>
#include <new>

namespace {

constexpr int had_size = 128;
constexpr float had_scale = 0.088388347648f;
constexpr int cache_max_rows = 64;

// Share input transforms across output tiles for decode/small batches without
// making large prefill workspaces grow with (tokens * selected experts * K).
int64_t shared_rows(const ggml_tensor * dst) {
    const int64_t rows = dst->op == GGML_OP_MUL_MAT_ID ? ggml_nelements(dst->src[2]) : dst->src[1]->ne[1];
    return rows <= cache_max_rows ? rows : 0;
}

struct cache_state {
    void * node;
    uint64_t hit_mask;
    int n_hits;
    bool retry;
    int32_t slots[cache_max_rows];
    int32_t rows[cache_max_rows];
    const float * acts[cache_max_rows];
    float * outputs[cache_max_rows];
};

int64_t cache_rows(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_MUL_MAT_ID || ggml_mmid_window_n_local(dst) != 0) return 0;
    const int64_t rows = ggml_nelements(dst->src[2]);
    return rows <= cache_max_rows ? rows : 0;
}

float half_round(float x) {
    return GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(x));
}

float codebook_value(uint32_t state, int cb) {
    if (cb == 2) {
        const uint32_t x = state * 0x83dcd12du;
        const uint16_t sum = 0x6400 + (x & 255) + ((x >> 8) & 255) +
                             ((x >> 16) & 255) + (x >> 24);
        return half_round(std::fma(GGML_FP16_TO_FP32(sum),
            GGML_FP16_TO_FP32(ggml_fp16_t(0x1eee)), GGML_FP16_TO_FP32(ggml_fp16_t(0xc931))));
    }
    uint32_t x = cb == 1 ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return half_round(GGML_FP16_TO_FP32(ggml_fp16_t(x)) + GGML_FP16_TO_FP32(ggml_fp16_t(x >> 16)));
}

const float * codebook(int cb) {
    struct tables {
        float values[3][65536];
        tables() {
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < 65536; ++i) values[c][i] = codebook_value(i, c);
        }
    };
    static const tables lut;
    return lut.values[cb];
}

void had128(float * x) {
    for (int d = 1; d < had_size; d *= 2)
        for (int i = 0; i < had_size; i += 2*d)
            for (int j = 0; j < d; ++j) {
                const float a = x[i+j], b = x[i+j+d];
                x[i+j] = a+b;
                x[i+j+d] = a-b;
            }
}

// EXL3 advances MSB-first within little-endian words. Each weight uses the
// trailing 16-bit state of a circular 16x16 tile, including for bits=1.
template<int bits>
uint16_t window(const uint8_t * tile, int element) {
    const int pos = ((element+1)*bits + 256*bits - 16) % (256*bits);
    const int word = pos/32;
    uint32_t a, b;
    std::memcpy(&a, tile + 4*word, sizeof(a));
    std::memcpy(&b, tile + 4*((word+1) % (8*bits)), sizeof(b));
    return uint16_t(((uint64_t(a) << 32) | b) >> (48-pos%32));
}

#if defined(__AVX2__) && defined(__FMA__)
// Eight output columns at a time, retaining the scalar K accumulation order.
// Window locations depend only on bit width and the 16x16 tile permutation.
template<int bits>
struct window_indices {
    int a[2][16][8], b[2][16][8], ar[2][16][8], al[2][16][8], br[2][16][8];
    constexpr window_indices() : a{}, b{}, ar{}, al{}, br{} {
        for (int h = 0; h < 2; ++h) for (int r = 0; r < 16; ++r) for (int c = 0; c < 8; ++c) {
            const int element = (c*4+(r%8)/2)*8 + r%2 + (r/8)*2 + h*4;
            const int pos = ((element+1)*bits + 256*bits - 16) % (256*bits);
            const int offset = pos%32;
            a[h][r][c] = pos/32;
            b[h][r][c] = (pos/32+1)%(8*bits);
            ar[h][r][c] = offset <= 16 ? 16-offset : 32;
            al[h][r][c] = offset > 16 ? offset-16 : 32;
            br[h][r][c] = offset > 16 ? 48-offset : 32;
        }
    }
};

template<int bits>
void dot_tile(const uint8_t * tile, const float * x, const float * lut, float * sum) {
    static constexpr window_indices<bits> index;
    auto load = [](const int * p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); };
    for (int h = 0; h < 2; ++h) {
        __m256 acc = _mm256_loadu_ps(sum+8*h);
        for (int r = 0; r < 16; ++r) {
            const auto * words = reinterpret_cast<const int *>(tile);
            const __m256i a = _mm256_i32gather_epi32(words, load(index.a[h][r]), 4);
            const __m256i b = _mm256_i32gather_epi32(words, load(index.b[h][r]), 4);
            __m256i state = _mm256_or_si256(_mm256_srlv_epi32(a, load(index.ar[h][r])),
                                          _mm256_sllv_epi32(a, load(index.al[h][r])));
            state = _mm256_or_si256(state, _mm256_srlv_epi32(b, load(index.br[h][r])));
            state = _mm256_and_si256(state, _mm256_set1_epi32(65535));
            acc = _mm256_fmadd_ps(_mm256_i32gather_ps(lut, state, 4), _mm256_set1_ps(x[r]), acc);
        }
        _mm256_storeu_ps(sum+8*h, acc);
    }
}
#endif

template<int bits>
void dot128(const uint8_t * weights, const float * x, const float * lut,
            int64_t k, int64_t col0, float * sum) {
    for (int col = 0; col < had_size; col += 16) {
        for (int64_t kt = 0; kt < k/16; ++kt) {
            const uint8_t * tile = weights + (((col0+col)/16)*(k/16)+kt)*32*bits;
#if defined(__AVX2__) && defined(__FMA__)
            dot_tile<bits>(tile, x+kt*16, lut, sum+col);
#else
            for (int c = 0; c < 16; ++c) {
                float part = sum[col+c];
                for (int r = 0; r < 16; ++r) {
                    const int element = ((c%8)*4+(r%8)/2)*8 + r%2 + (r/8)*2 + (c/8)*4;
                    part = std::fma(lut[window<bits>(tile, element)], x[kt*16+r], part);
                }
                sum[col+c] = part;
            }
#endif
        }
    }
}

// Resolve bit width once per operation so circular-window extraction does
// not perform integer division for every decoded weight.
using dot_fn = void (*)(const uint8_t *, const float *, const float *, int64_t, int64_t, float *);
constexpr dot_fn dot_kernels[] = {dot128<1>, dot128<2>, dot128<3>, dot128<4>,
                                 dot128<5>, dot128<6>, dot128<7>, dot128<8>};

bool auxiliary_shape(const ggml_tensor * tensor, int64_t width, int64_t experts) {
    return tensor && tensor->type == GGML_TYPE_F16 && tensor->ne[0] == width &&
        tensor->ne[1] == experts && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
        ggml_is_contiguous(tensor);
}

} // namespace

bool ggml_cpu_exl3_supports(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) return false;
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    if (!w || !x || !ggml_type_is_exl3(w->type) || w->ne[0] % had_size || w->ne[1] % had_size ||
        !ggml_is_contiguous(w) || w->ne[3] != 1 || x->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) || x->ne[0] != w->ne[0] || x->ne[3] != 1 ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;

    const bool grouped = dst->op == GGML_OP_MUL_MAT_ID;
    const auto * svh = dst->src[grouped ? 3 : 2];
    const auto * suh = dst->src[grouped ? 4 : 3];
    if (grouped) {
        const auto * ids = dst->src[2];
        if (!ids || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->ne[2] != 1 || ids->ne[3] != 1 || x->ne[2] != ids->ne[1] ||
            x->ne[1] < 1 || ids->ne[0] % x->ne[1] != 0) return false;
    } else if (w->ne[2] != 1 || x->ne[2] != 1) {
        return false;
    }
    // Loader placement probes precede attachment of the scale/sign tensors.
    return (!svh && !suh) || (auxiliary_shape(svh, w->ne[1], w->ne[2]) &&
                               auxiliary_shape(suh, w->ne[0], w->ne[2]));
}

size_t ggml_cpu_exl3_work_size(const ggml_tensor * dst, int threads) {
    // Small batches share one transformed activation per row. Larger batches
    // keep one per worker; packed weights stay packed in both cases.
    const auto * w = dst->src[0];
    const size_t rows = cache_rows(dst);
    const size_t transforms = shared_rows(dst) ? shared_rows(dst) : threads;
    return transforms * size_t(w->ne[0]) * sizeof(float) +
        (rows ? GGML_PAD(sizeof(cache_state), 64) + rows*w->ne[1]*sizeof(float) : 0);
}

void ggml_cpu_exl3_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cpu_exl3_supports(dst));
    GGML_ASSERT(params->wsize >= ggml_cpu_exl3_work_size(dst, params->nth));
    const bool grouped = dst->op == GGML_OP_MUL_MAT_ID;
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const auto * ids = grouped ? dst->src[2] : nullptr;
    const auto * svh = dst->src[grouped ? 3 : 2];
    const auto * suh = dst->src[grouped ? 4 : 3];
    GGML_ASSERT(svh && suh);
    const int64_t k = w->ne[0], n = w->ne[1];
    const int64_t topk = grouped ? ids->ne[0] : 1;
    const int64_t tokens = grouped ? ids->ne[1] : x->ne[1];
    const int bits = ggml_exl3_bits(w->type);
    const auto dot = dot_kernels[bits-1];
    const float * lut = codebook(ggml_exl3_codebook(w->type));
    const int64_t shared = shared_rows(dst);
    float * transformed = static_cast<float *>(params->wdata);
    float * xh = shared ? nullptr : transformed + params->ith*k;

    auto expert_at = [&](int64_t row) {
        int32_t expert = 0;
        if (grouped) {
            std::memcpy(&expert, static_cast<const char *>(ids->data) +
                (row/topk)*ids->nb[1] + (row%topk)*ids->nb[0], sizeof(expert));
            expert = ggml_mmid_expert_index(expert, ggml_mmid_window_lo(dst), ggml_mmid_window_n_local(dst));
            GGML_ASSERT(expert >= -1 && expert < w->ne[2]);
        }
        return expert;
    };
    auto prepare = [&](int64_t row, float * transformed) {
        const int expert = expert_at(row);
        const auto * signs = reinterpret_cast<const ggml_fp16_t *>(static_cast<const char *>(suh->data) + expert*suh->nb[1]);
        const size_t offset = grouped ? (row/topk)*x->nb[2] + (row%topk % x->ne[1])*x->nb[1] : row*x->nb[1];
        const auto * input = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + offset);
        for (int64_t i = 0; i < k; ++i) transformed[i] = input[i]*GGML_FP16_TO_FP32(signs[i]);
        for (int64_t i = 0; i < k; i += had_size) had128(transformed+i);
        for (int64_t i = 0; i < k; ++i) transformed[i] = half_round(transformed[i]*had_scale);
    };
    auto finish = [&](int64_t row, int64_t col0, float * sum) {
        const auto * scales = reinterpret_cast<const ggml_fp16_t *>(static_cast<const char *>(svh->data) + expert_at(row)*svh->nb[1]);
        had128(sum);
        float * out = static_cast<float *>(dst->data) + row*n + col0;
        for (int i = 0; i < had_size; ++i) out[i] = sum[i]*had_scale*GGML_FP16_TO_FP32(scales[col0+i]);
    };

    if (shared) {
        for (int64_t row = params->ith; row < shared; row += params->nth) {
            if (expert_at(row) >= 0) prepare(row, transformed + row*k);
        }
        ggml_barrier(params->threadpool);
    }

    cache_state * cache = nullptr;
    if (cache_rows(dst)) {
        cache = reinterpret_cast<cache_state *>(transformed + shared*k);
        if (params->ith == 0) {
            new (cache) cache_state{};
            const auto buffer = w->view_src ? w->view_src->buffer : w->buffer;
            if (w->op == GGML_OP_NONE && buffer && ggml_backend_buffer_is_host(buffer) &&
                ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                ggml_moe_cache.begin && ggml_moe_cache.plan && ggml_moe_cache.dispatch &&
                ggml_moe_cache.collect && ggml_moe_cache.end) {
                cache->node = ggml_moe_cache.begin(w->name, w->data, w->nb[2], k, n, w->type, w->ne[2], tokens, tokens*topk);
            }
            if (cache->node) {
                int32_t expert_ids[cache_max_rows], slots[cache_max_rows];
                for (int64_t row = 0; row < tokens*topk; ++row) expert_ids[row] = expert_at(row);
                ggml_moe_cache.plan(cache->node, expert_ids, int(tokens*topk), slots);
                float * outputs = reinterpret_cast<float *>(reinterpret_cast<char *>(cache) + GGML_PAD(sizeof(cache_state), 64));
                for (int64_t row = 0; row < tokens*topk; ++row) {
                    if (slots[row] < 0) continue;
                    const int hit = cache->n_hits++;
                    cache->slots[hit] = slots[row];
                    cache->rows[hit] = int32_t(row);
                    cache->acts[hit] = transformed+row*k;
                    cache->outputs[hit] = outputs+hit*n;
                    cache->hit_mask |= uint64_t(1) << row;
                }
                // EXL3 cache dot products consume the same transformed inputs
                // as CPU misses. Both finish through the same output transform.
                if (!cache->n_hits || !ggml_moe_cache.dispatch(cache->node, w->type, k, n,
                        cache->n_hits, cache->slots, cache->acts)) {
                    ggml_moe_cache.end(cache->node);
                    cache->node = nullptr;
                    cache->hit_mask = 0;
                }
            }
        }
        ggml_barrier(params->threadpool);
        if (!cache->node) cache = nullptr;
    }

    // A whole Hadamard output block belongs to one worker, so neither a
    // reduction barrier nor thread-count-dependent summation is required.
    auto compute = [&](bool retry) {
        int64_t prepared_row = -1;
        for (int64_t task = params->ith; task < tokens*topk*(n/had_size); task += params->nth) {
            const int64_t col0 = (task % (n/had_size))*had_size;
            const int64_t row = task / (n/had_size);
            const bool hit = cache && (cache->hit_mask & (uint64_t(1) << row));
            if (hit != retry) continue;
            const int expert = expert_at(row);
            if (expert < 0) {
                std::memset(static_cast<float *>(dst->data) + row*n + col0, 0, had_size*sizeof(float));
                continue;
            }
            if (!shared && row != prepared_row) {
                prepare(row, xh);
                prepared_row = row;
            }

            float sum[had_size] = {};
            const auto * weights = static_cast<const uint8_t *>(w->data) + expert*w->nb[2];
            dot(weights, shared ? transformed+row*k : xh, lut, k, col0, sum);
            finish(row, col0, sum);
        }
    };
    compute(false);
    if (cache) {
        ggml_barrier(params->threadpool);
        if (params->ith == 0) {
            cache->retry = !ggml_moe_cache.collect(cache->node, cache->n_hits, cache->outputs, n);
            ggml_moe_cache.end(cache->node);
            cache->node = nullptr;
        }
        ggml_barrier(params->threadpool);
        if (cache->retry) {
            compute(true);
        } else {
            // collect copied into our workspace; each worker finishes disjoint
            // Hadamard blocks, just as on the CPU path.
            for (int64_t task = params->ith; task < cache->n_hits*(n/had_size); task += params->nth) {
                const int hit = task / (n/had_size);
                const int64_t col = (task % (n/had_size))*had_size;
                finish(cache->rows[hit], col, cache->outputs[hit]+col);
            }
        }
    }
}
