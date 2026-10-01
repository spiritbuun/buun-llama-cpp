#include "gated_delta_net.cuh"
#include "gdn-norm.cuh"
#include "gated_delta_net_fla_ptx.cuh"
#include "ggml-cuda/common.cuh"

#include <type_traits>

// RDNA3 wants 1 block/SM here: with 2 blocks/SM the compiler is forced under
// ~128 VGPR/lane and spills on this kernel. 1 block/SM raises the per-lane
// VGPR budget to ~256 and eliminates the spill entirely. CUDA keeps 2.
// See llama.cpp issue #20354 (GATED_DELTA_NET HIP underperforms on RDNA3).
// exp2 is a single SFU instruction (ex2.approx) vs expf's multiply + ex2 + range reduction
#define GDN_EXPF(x) exp2f((x) * 1.442695041f)

#if defined(GGML_USE_HIP)
#define GGML_GDN_MIN_BLOCKS_PER_SM 1
#else
#define GGML_GDN_MIN_BLOCKS_PER_SM 2
#endif

template <int S_v, bool KDA, bool keep_rs_t, bool sum_eps_t, bool indexed_state = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     ggml_cuda_gdn_norm l2_norm, const int32_t * input_rows, int64_t state_row_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // indexed_state: read the recurrent cache row input_rows[sequence] (the skipped GET_ROWS
    // source) with the cache's row stride, which may exceed D for strided caches.
    const int64_t input_sequence       = indexed_state ? input_rows[sequence] : sequence;
    const int64_t state_in_offset      = indexed_state
                                            ? (int64_t) input_sequence * state_row_stride + h_idx * S_v * S_v
                                            : sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
#if !defined(GGML_USE_HIP)
        // HIP normalizes in the preceding paired kernel and never defers it.
        // Keep this unused branch out of its recurrent token loop.
        // Deferred q/k normalization retains the graph's epsilon convention.
        if (sum_eps_t || l2_norm.eps >= 0.0f) {
            float sum_q = 0.0f;
            float sum_k = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                sum_q += q_reg[r] * q_reg[r];
                sum_k += k_reg[r] * k_reg[r];
            }
            sum_q = warp_reduce_sum<warp_size>(sum_q);
            sum_k = warp_reduce_sum<warp_size>(sum_k);
            const float scale_q = sum_eps_t ? rsqrtf(sum_q + l2_norm.eps) : l2_norm.inverse(sum_q, S_v);
            const float scale_k = sum_eps_t ? rsqrtf(sum_k + l2_norm.eps) : l2_norm.inverse(sum_k, S_v);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_reg[r] = sum_eps_t ? q_reg[r] * scale_q : l2_norm.apply(q_reg[r], scale_q);
                k_reg[r] = sum_eps_t ? k_reg[r] * scale_k : l2_norm.apply(k_reg[r], scale_k);
            }
        }

#else
        GGML_UNUSED(l2_norm);
#endif

        if constexpr (!KDA) {
            const float g_val = GDN_EXPF(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += GDN_EXPF(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = GDN_EXPF(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, ggml_cuda_gdn_norm l2_norm, cudaStream_t stream,
        const int32_t * input_rows = nullptr, int64_t state_row_stride = 0) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    // Select canonical normalization once, outside the recurrent token loop.
    const auto launch = [&](auto sum_eps_tag) {
        constexpr bool sum_eps_t = decltype(sum_eps_tag)::value;
        // input_rows (the skipped GET_ROWS source, or the narrow 3-node match) reads the state
        // through the cache rows for any S_v / keep_rs; everything else uses the plain form
#define GDN_LAUNCH(SV_) \
        do { \
            if (!KDA && input_rows != nullptr) { \
                ggml_cuda_kernel_launch(gated_delta_net_cuda<SV_, false, keep_rs_t, sum_eps_t, true>, launch_params, \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, \
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride); \
            } else { \
                ggml_cuda_kernel_launch(gated_delta_net_cuda<SV_, KDA, keep_rs_t, sum_eps_t, false>, launch_params, \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, \
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride); \
            } \
        } while (0)
        switch (S_v) {
            case 16:
                GDN_LAUNCH(16);
                break;
            case 32:
                GDN_LAUNCH(32);
                break;
            case 64:
                GDN_LAUNCH(64);
                break;
            case 128:
                GDN_LAUNCH(128);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
#undef GDN_LAUNCH
    };
#if defined(GGML_USE_HIP)
    launch(std::false_type{});
#else
    if (l2_norm.sum_eps) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
#endif
}

// same per-column math as gated_delta_net_cuda, but each warp owns NC state columns so the NC
// reduction chains interleave; multi-token, scalar-gate, S_v == 128 only (GGML_CUDA_SM86_GDN_COLS)
template <int S_v, bool keep_rs_t, int NC, bool PREFETCH, bool sum_eps_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
gated_delta_net_cuda_ilp(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         float *       state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       n_seqs,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K,
                         ggml_cuda_gdn_norm l2_norm,
                         const int32_t * input_rows, int64_t state_row_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * NC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    // input state: src[5] at sequence * D, or the recurrent cache row input_rows[sequence]
    // (row stride in floats, always D = H * S_v * S_v for the shapes matched)
    const int64_t state_in_offset = (input_rows ? (int64_t) input_rows[sequence] * state_row_stride :
                                      sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset + col0 * S_v;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;

    float s_shard[NC][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + sequence * sb3 + h_idx * sb1;

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float v_reg[NC];
    float g_raw;
    float beta_val;

    auto load_tok = [&](int t, float * kr, float * qr, float * vr, float & gr, float & br) {
        const float * q_t = q_base + t * sq2;
        const float * k_t = k_base + t * sq2;
        const float * v_t = v_base + t * sv2;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kr[r] = k_t[r * warp_size + lane];
            qr[r] = q_t[r * warp_size + lane];
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            vr[c] = v_t[c];
        }
        gr = g_base[t * sb2];
        br = b_base[t * sb2];
#if !defined(GGML_USE_HIP)
        // deferred q/k normalization, same variants as the serial kernel: applied here once per
        // token so the PREFETCH registers are already normalized when swapped in
        if (sum_eps_t || l2_norm.eps >= 0.0f) {
            float sum_q = 0.0f;
            float sum_k = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                sum_q += qr[r] * qr[r];
                sum_k += kr[r] * kr[r];
            }
            sum_q = warp_reduce_sum<warp_size>(sum_q);
            sum_k = warp_reduce_sum<warp_size>(sum_k);
            const float scale_q = sum_eps_t ? rsqrtf(sum_q + l2_norm.eps) : l2_norm.inverse(sum_q, S_v);
            const float scale_k = sum_eps_t ? rsqrtf(sum_k + l2_norm.eps) : l2_norm.inverse(sum_k, S_v);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                qr[r] = sum_eps_t ? qr[r] * scale_q : l2_norm.apply(qr[r], scale_q);
                kr[r] = sum_eps_t ? kr[r] * scale_k : l2_norm.apply(kr[r], scale_k);
            }
        }
#else
        GGML_UNUSED(l2_norm);
#endif
    };

    if constexpr (PREFETCH) {
        load_tok(0, k_reg, q_reg, v_reg, g_raw, beta_val);
    }

    for (int t = 0; t < n_tokens; t++) {
        float kn_reg[rows_per_lane];
        float qn_reg[rows_per_lane];
        float vn_reg[NC];
        float gn_raw = 0.0f;
        float bn_val = 0.0f;
        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
                load_tok(t + 1, kn_reg, qn_reg, vn_reg, gn_raw, bn_val);
            }
        } else {
            load_tok(t, k_reg, q_reg, v_reg, g_raw, beta_val);
        }

        const float g_val = GDN_EXPF(g_raw);

        // kv[col] = sum_i S[i][col] * k[i]   (same order as the serial kernel, per column)
        float kv_shard[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard[c] += s_shard[c][r] * k_reg[r];
            }
        }
        float kv_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
        }

        float attn_partial[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float delta_col = (v_reg[c] - g_val * kv_col[c]) * beta_val;
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]    = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }
        float attn_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            attn_col[c] = warp_reduce_sum<warp_size>(attn_partial[c]);
        }

        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snap = state + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < NC; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        snap[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
                    }
                }
            }
        }

        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_reg[r] = kn_reg[r];
                    q_reg[r] = qn_reg[r];
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    v_reg[c] = vn_reg[c];
                }
                g_raw    = gn_raw;
                beta_val = bn_val;
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

// default 4; 1 selects the serial kernel (with prefetch), 2/8 widen the per-warp column span
static int ggml_cuda_sm86_gdn_cols() {
    static const int cols = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_COLS");
        const int v = s ? atoi(s) : 4;
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 4;
    }();
    return cols;
}

static bool ggml_cuda_sm86_gdn_prefetch() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_PREFETCH");
        return s != nullptr && atoi(s) != 0;
    }();
    return on;
}

template <bool keep_rs_t, int NC, bool PREFETCH, bool sum_eps_t>
static void launch_gated_delta_net_ilp_inst(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, ggml_cuda_gdn_norm l2_norm,
        const int32_t * input_rows, int64_t state_row_stride, cudaStream_t stream) {
    constexpr int S_v = 128;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    GGML_ASSERT(S_v % (num_warps * NC) == 0);
    dim3 grid_dims(H, n_seqs, S_v / (num_warps * NC));
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_ilp<S_v, keep_rs_t, NC, PREFETCH, sum_eps_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride);
}

// returns false when this path does not apply; caller falls back to the serial kernel
template <bool keep_rs_t>
static bool launch_gated_delta_net_ilp(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, ggml_cuda_gdn_norm l2_norm,
        const int32_t * input_rows, int64_t state_row_stride, cudaStream_t stream) {
    const int  nc       = ggml_cuda_sm86_gdn_cols();
    const bool prefetch = ggml_cuda_sm86_gdn_prefetch();
    if (S_v != 128 || n_tokens < 2 || (nc == 1 && !prefetch)) {
        return false;
    }
    const auto launch = [&](auto sum_eps_tag) {
        constexpr bool sum_eps_t = decltype(sum_eps_tag)::value;
#define GDN_ILP_LAUNCH(NC_, PF_) \
        launch_gated_delta_net_ilp_inst<keep_rs_t, NC_, PF_, sum_eps_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride, stream)
        if (nc == 8) {
            if (prefetch) { GDN_ILP_LAUNCH(8, true); } else { GDN_ILP_LAUNCH(8, false); }
        } else if (nc == 4) {
            if (prefetch) { GDN_ILP_LAUNCH(4, true); } else { GDN_ILP_LAUNCH(4, false); }
        } else if (nc == 2) {
            if (prefetch) { GDN_ILP_LAUNCH(2, true); } else { GDN_ILP_LAUNCH(2, false); }
        } else {
            GDN_ILP_LAUNCH(1, true);
        }
#undef GDN_ILP_LAUNCH
    };
#if defined(GGML_USE_HIP)
    launch(std::false_type{});
#else
    if (l2_norm.sum_eps) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
#endif
    return true;
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

#if !defined(GGML_USE_HIP)
    const bool defer_q_l2 = ctx.gdn_deferred_l2.erase(src_q->data) != 0;
    const bool defer_k_l2 = ctx.gdn_deferred_l2.erase(src_k->data) != 0;
#else
    const bool defer_q_l2 = false;
    const bool defer_k_l2 = false;
#endif
    GGML_ASSERT(defer_q_l2 == defer_k_l2);

    ggml_cuda_gdn_norm l2_norm, k_norm;
    const ggml_tensor * packed_q = defer_q_l2 ? ggml_cuda_gdn_norm_input(src_q, l2_norm) : src_q;
    const ggml_tensor * packed_k = defer_k_l2 ? ggml_cuda_gdn_norm_input(src_k, k_norm) : src_k;
    GGML_ASSERT(packed_q && packed_k);
    GGML_ASSERT(l2_norm.eps == k_norm.eps && l2_norm.rms == k_norm.rms && l2_norm.sum_eps == k_norm.sum_eps &&
                l2_norm.post_scale == k_norm.post_scale);

    const float * q_d = (const float *) packed_q->data;
    const float * k_d = (const float *) packed_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = cache && cache->input_state ? cache->input_state : (const float *) src_state->data;
    const int32_t * input_rows = cache ? cache->input_rows : nullptr;
    int64_t state_row_stride    = S_v * S_v * H;
    // the GET_ROWS that gathered src[5] from the recurrent cache was skipped by the graph
    // evaluator (ggml_cuda_gdn_state_read_plan): read the source rows directly. Same floats,
    // same positions, so the launch computes exactly what it would have from the gathered copy.
    for (auto & e : ctx.gdn_state_reads) {
        if (e.gdn == dst) {
            GGML_ASSERT(!e.used);
            s_d              = e.base;
            input_rows       = e.rows;
            state_row_stride = e.row_stride;
            e.used           = true;
            ctx.gdn_state_read_count++;
            break;
        }
    }
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = packed_q->nb[1] / sizeof(float);
    int64_t sq2 = packed_q->nb[2] / sizeof(float);
    int64_t sq3 = packed_q->nb[3] / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr && cache->data != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!input_rows && ggml_cuda_gdn_fla_ptx_supported(cc, kda, keep_rs, S_v, H, neqk1, n_tokens, n_seqs)) {
        ggml_cuda_gdn_fla_ptx(ctx, cc, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                              n_tokens, H, neqk1, sq1, sq2, sq3, sv1, sv2, sv3,
                              nullptr,
                              l2_norm,
                              cache != nullptr ? cache->rms_weight : nullptr,
                              cache != nullptr ? cache->rms_gate : nullptr,
                              cache != nullptr ? cache->rms_gate_bf16 : false,
                              cache != nullptr ? cache->rms_output : nullptr,
                              cache != nullptr ? cache->rms_output_bf16 : false,
                              cache != nullptr ? cache->rms_output_int8 : false,
                              cache != nullptr ? cache->rms_output_scale : nullptr,
                              cache != nullptr ? cache->rms_eps : 0.0f);
        return;
    }

    // ILP kernel (multiple state columns per warp): scalar gate, S_v 128, multi-token only.
    // The FLA PTX path above already returned for the shapes it owns (n_tokens >= 512, no
    // keep_rs), so the two are mutually exclusive; keep_rs snapshots use the K-slot layout
    // (no emit_ingredients in this op) and the input state comes from src[5] or the cache
    // row (cache->input_state/input_rows), same as the serial kernel.
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows, state_row_stride);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows, state_row_stride);
        }
    } else {
        if (keep_rs) {
            if (launch_gated_delta_net_ilp<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride, stream)) {
                return;
            }
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows, state_row_stride);
        } else {
            if (launch_gated_delta_net_ilp<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, input_rows, state_row_stride, stream)) {
                return;
            }
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows, state_row_stride);
        }
    }
}

void ggml_cuda_op_gated_delta_net_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ABORT("tree GDN kernel not yet ported to upstream state layout");
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
