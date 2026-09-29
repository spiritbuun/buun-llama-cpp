#pragma once
// EXPERIMENT (exp/tc-int8-kv): fake-quant reference attention for the turbo prefill path.
// Simulates what an int8 tensor-core attention kernel would compute, in the rotated domain:
//   Q -> int8 (per-block amax), K -> int8 (per-block amax), P -> u8 with the V block scale
//   folded in (V -> int8 per-block amax), optional f16-accumulate QK.
// With every quantizer off it mimics the f16 MMA kernel (f16 Q, f32 QK acc, f16 P, f16 PV acc),
// which is the anchor. Enabled by TCKV_SIM=1; knobs:
//   TCKV_Q=8   TCKV_QB=32|64|128|256   Q int8, block size along head_dim
//   TCKV_K=8   TCKV_KB=...             K int8, block size (default 128)
//   TCKV_P=8   TCKV_VB=...             P u8 per (row, 64-key tile, V block), V int8 (default 128)
//   TCKV_QK16=1                        QK accumulated in f16 (rounded every 16 channels)
//   TCKV_EXACT=1                       fp32 everywhere where a quantizer is off

#include "common.cuh"

#define TCKV_TILE 64
#define TCKV_QG   8
#define TCKV_NT   128

struct tckv_sim_cfg {
    bool enabled;
    int  q_bits, qb, k_bits, kb, p_bits, vb;
    bool qk16, exact;
};

static const tckv_sim_cfg & tckv_sim_get_cfg() {
    static tckv_sim_cfg cfg = [] {
        auto geti = [](const char * name, int def) {
            const char * s = getenv(name);
            return s ? atoi(s) : def;
        };
        tckv_sim_cfg c;
        c.enabled = geti("TCKV_SIM", 0) != 0;
        c.q_bits  = geti("TCKV_Q", 0);
        c.qb      = geti("TCKV_QB", 32);
        c.k_bits  = geti("TCKV_K", 0);
        c.kb      = geti("TCKV_KB", 128);
        c.p_bits  = geti("TCKV_P", 0);
        c.vb      = geti("TCKV_VB", 128);
        c.qk16    = geti("TCKV_QK16", 0) != 0;
        c.exact   = geti("TCKV_EXACT", 0) != 0;
        if (c.enabled) {
            fprintf(stderr, "TCKV_SIM: Q=%d/%d K=%d/%d P=%d/%d qk16=%d exact=%d\n",
                c.q_bits, c.qb, c.k_bits, c.kb, c.p_bits, c.vb, (int) c.qk16, (int) c.exact);
        }
        return c;
    }();
    return cfg;
}

// In-place per-block amax over smem row a[0..D) into amax[0..D) (every element gets its block's amax).
static __device__ void tckv_block_amax(const float * a, float * amax, int D, int blk) {
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        amax[c] = fabsf(a[c]);
    }
    __syncthreads();
    for (int s = blk / 2; s >= 1; s /= 2) {
        for (int c = threadIdx.x; c < D; c += blockDim.x) {
            if ((c % blk) < s) {
                amax[c] = fmaxf(amax[c], amax[c + s]);
            }
        }
        __syncthreads();
    }
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        amax[c] = amax[c - (c % blk)];
    }
    __syncthreads();
}

// Per row of a dense [nrows, D] f16 buffer: K mode writes fake-quantized values, V mode (fold != null)
// writes the int8 code as float plus its per-block scale into fold[row*(D/blk) + b].
static __global__ void k_tckv_prep(const half * __restrict__ src, float * __restrict__ dst,
        float * __restrict__ fold, int D, int bits, int blk) {
    extern __shared__ float sm[];
    float * a    = sm;
    float * amax = sm + D;
    const int64_t row = blockIdx.x;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        a[c] = __half2float(src[row * D + c]);
    }
    __syncthreads();
    if (bits == 0) {
        for (int c = threadIdx.x; c < D; c += blockDim.x) {
            dst[row * D + c] = a[c];
        }
        return;
    }
    tckv_block_amax(a, amax, D, blk);
    const float qmax = (float) ((1 << (bits - 1)) - 1);
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const float s = amax[c] / qmax;
        const float q = s > 0.0f ? rintf(a[c] / s) : 0.0f;
        if (fold) {
            dst[row * D + c] = q;
            if (c % blk == 0) {
                fold[row * (D / blk) + c / blk] = s;
            }
        } else {
            dst[row * D + c] = q * s;
        }
    }
}

static __device__ __forceinline__ float tckv_h(float x) {
    return __half2float(__float2half(x));
}

// One block = TCKV_QG consecutive queries of one head. K/V are dense [n_kv, n_head_kv, D] f32
// (row = j*n_head_kv + hk), Vfold is [n_kv*n_head_kv, D/vb] when P quantization is on.
template <int D>
static __global__ void k_tckv_attend(
        const char * __restrict__ Qp, const float * __restrict__ Kq, const float * __restrict__ Vq,
        const float * __restrict__ Vfold, const char * __restrict__ maskp, float * __restrict__ dstp,
        const int n_q, const int n_head, const int n_kv, const int n_head_kv,
        const size_t qnb1, const size_t qnb2, const size_t qnb3,
        const int mne2, const int mne3, const size_t mnb1, const size_t mnb2, const size_t mnb3,
        const size_t dnb1, const size_t dnb2, const size_t dnb3,
        const float scale, const tckv_sim_cfg cfg) {
    constexpr int NC  = D / TCKV_NT;          // output channels per thread
    constexpr int NVB = 8;                    // max V blocks per row (D/vb, vb >= 32)
    __shared__ float qs[TCKV_QG][D];
    __shared__ float qamax[D];
    __shared__ float S[TCKV_QG][TCKV_TILE];
    __shared__ float W[TCKV_QG][TCKV_TILE][NVB];
    __shared__ float m_s[TCKV_QG], l_s[TCKV_QG], alpha_s[TCKV_QG];

    const int i0   = blockIdx.x * TCKV_QG;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int hk   = h / (n_head / n_head_kv);
    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int nvb  = D / cfg.vb;

    // Q rows: load, then int8 fake-quant or f16 rounding
    for (int r = 0; r < TCKV_QG; r++) {
        const int i = min(i0 + r, n_q - 1);
        const float * qrow = (const float *) (Qp + seq * qnb3 + h * qnb2 + i * qnb1);
        for (int c = tid; c < D; c += TCKV_NT) {
            qs[r][c] = qrow[c];
        }
        __syncthreads();
        if (cfg.q_bits) {
            tckv_block_amax(qs[r], qamax, D, cfg.qb);
            const float qmax = (float) ((1 << (cfg.q_bits - 1)) - 1);
            for (int c = tid; c < D; c += TCKV_NT) {
                const float s = qamax[c] / qmax;
                qs[r][c] = s > 0.0f ? rintf(qs[r][c] / s) * s : 0.0f;
            }
        } else if (!cfg.exact) {
            for (int c = tid; c < D; c += TCKV_NT) {
                qs[r][c] = tckv_h(qs[r][c] * scale) / scale;
            }
        }
        __syncthreads();
    }
    if (tid < TCKV_QG) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }

    float O[TCKV_QG][NC];
    for (int r = 0; r < TCKV_QG; r++) {
        for (int k = 0; k < NC; k++) {
            O[r][k] = 0.0f;
        }
    }
    __syncthreads();

    for (int j0 = 0; j0 < n_kv; j0 += TCKV_TILE) {
        // Scores: each warp handles TCKV_TILE/4 keys
        for (int jj = warp; jj < TCKV_TILE; jj += TCKV_NT / 32) {
            const int j = j0 + jj;
            float acc[TCKV_QG];
            if (j < n_kv) {
                const float * krow = Kq + ((int64_t) j * n_head_kv + hk) * D;
                if (cfg.qk16) {
                    // lane k computes 16-channel chunk k; chunks summed in order with f16 rounding
                    float part[TCKV_QG];
                    for (int r = 0; r < TCKV_QG; r++) {
                        part[r] = 0.0f;
                    }
                    if (lane < D / 16) {
                        for (int c = lane * 16; c < lane * 16 + 16; c++) {
                            const float kv = krow[c];
                            for (int r = 0; r < TCKV_QG; r++) {
                                part[r] += qs[r][c] * scale * kv;
                            }
                        }
                    }
                    for (int r = 0; r < TCKV_QG; r++) {
                        float a = 0.0f;
                        for (int k = 0; k < D / 16; k++) {
                            a = tckv_h(a + __shfl_sync(0xFFFFFFFF, part[r], k));
                        }
                        acc[r] = a;
                    }
                } else {
                    for (int r = 0; r < TCKV_QG; r++) {
                        acc[r] = 0.0f;
                    }
                    for (int c = lane; c < D; c += 32) {
                        const float kv = krow[c];
                        for (int r = 0; r < TCKV_QG; r++) {
                            acc[r] += qs[r][c] * kv;
                        }
                    }
                    for (int r = 0; r < TCKV_QG; r++) {
                        for (int o = 16; o >= 1; o /= 2) {
                            acc[r] += __shfl_xor_sync(0xFFFFFFFF, acc[r], o);
                        }
                        acc[r] *= scale;
                    }
                }
            }
            if (lane == 0) {
                for (int r = 0; r < TCKV_QG; r++) {
                    const int i = i0 + r;
                    float s = -INFINITY;
                    if (j < n_kv && i < n_q) {
                        const half * mrow = (const half *) (maskp + (seq % mne3) * mnb3 + (h % mne2) * mnb2 + i * mnb1);
                        s = acc[r] + (maskp ? __half2float(mrow[j]) : 0.0f);
                    }
                    S[r][jj] = s;
                }
            }
        }
        __syncthreads();

        // Online softmax per row (warp per row), P quantization
        for (int r = warp; r < TCKV_QG; r += TCKV_NT / 32) {
            float tmax = -INFINITY;
            for (int jj = lane; jj < TCKV_TILE; jj += 32) {
                tmax = fmaxf(tmax, S[r][jj]);
            }
            for (int o = 16; o >= 1; o /= 2) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, o));
            }
            const float m_old = m_s[r];
            const float m_new = fmaxf(m_old, tmax);
            const float alpha = m_new == -INFINITY ? 1.0f : expf(m_old - m_new);
            float sum = 0.0f;
            for (int jj = lane; jj < TCKV_TILE; jj += 32) {
                const float p = S[r][jj] == -INFINITY ? 0.0f : expf(S[r][jj] - m_new);
                sum += p;
                S[r][jj] = p;
            }
            for (int o = 16; o >= 1; o /= 2) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, o);
            }
            if (cfg.p_bits) {
                const float pmax = (float) ((1 << cfg.p_bits) - 1); // P >= 0: unsigned
                for (int b = 0; b < nvb; b++) {
                    float wmax = 0.0f;
                    for (int jj = lane; jj < TCKV_TILE; jj += 32) {
                        const int j = j0 + jj;
                        const float w = j < n_kv ? S[r][jj] * Vfold[((int64_t) j * n_head_kv + hk) * nvb + b] : 0.0f;
                        W[r][jj][b] = w;
                        wmax = fmaxf(wmax, w);
                    }
                    for (int o = 16; o >= 1; o /= 2) {
                        wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, o));
                    }
                    const float s = wmax / pmax;
                    for (int jj = lane; jj < TCKV_TILE; jj += 32) {
                        W[r][jj][b] = s > 0.0f ? rintf(W[r][jj][b] / s) * s : 0.0f;
                    }
                }
            } else if (!cfg.exact) {
                for (int jj = lane; jj < TCKV_TILE; jj += 32) {
                    S[r][jj] = tckv_h(S[r][jj]);
                }
            }
            if (lane == 0) {
                m_s[r]     = m_new;
                l_s[r]     = l_s[r] * alpha + sum;
                alpha_s[r] = alpha;
            }
        }
        __syncthreads();

        // PV
        for (int k = 0; k < NC; k++) {
            const int c = tid + k * TCKV_NT;
            const int b = c / cfg.vb;
            for (int r = 0; r < TCKV_QG; r++) {
                O[r][k] *= alpha_s[r];
                if (!cfg.p_bits && !cfg.exact) {
                    O[r][k] = tckv_h(O[r][k]);
                }
            }
            for (int jc = 0; jc < TCKV_TILE; jc += 16) {
                float part[TCKV_QG];
                for (int r = 0; r < TCKV_QG; r++) {
                    part[r] = 0.0f;
                }
                for (int jj = jc; jj < jc + 16; jj++) {
                    const int j = j0 + jj;
                    if (j >= n_kv) {
                        break;
                    }
                    const float v = Vq[((int64_t) j * n_head_kv + hk) * D + c];
                    for (int r = 0; r < TCKV_QG; r++) {
                        part[r] += (cfg.p_bits ? W[r][jj][b] : S[r][jj]) * v;
                    }
                }
                for (int r = 0; r < TCKV_QG; r++) {
                    O[r][k] = (!cfg.p_bits && !cfg.exact) ? tckv_h(O[r][k] + part[r]) : O[r][k] + part[r];
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < TCKV_QG; r++) {
        const int i = i0 + r;
        if (i >= n_q) {
            break;
        }
        const float inv_l = l_s[r] > 0.0f ? 1.0f / l_s[r] : 0.0f;
        float * drow = (float *) ((char *) dstp + seq * dnb3 + i * dnb2 + h * dnb1);
        for (int k = 0; k < NC; k++) {
            drow[tid + k * TCKV_NT] = O[r][k] * inv_l;
        }
    }
}

// True when the simulator can replace the attention kernel for this op.
static bool tckv_sim_applicable(const ggml_tensor * dst) {
    const tckv_sim_cfg & cfg = tckv_sim_get_cfg();
    if (!cfg.enabled) {
        return false;
    }
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    auto plain_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_TURBO8_0;
    };
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    return plain_turbo(K->type) && plain_turbo(V->type) && K->ne[0] == V->ne[0] &&
        (K->ne[0] == 128 || K->ne[0] == 256) && dst->src[4] == nullptr && max_bias == 0.0f && softcap == 0.0f;
}

// Runs after materialization: dst->src[0] is (rotated) f32 Q, src[1]/src[2] are dense f16 rotated-domain K/V.
static void tckv_sim_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const tckv_sim_cfg & cfg = tckv_sim_get_cfg();
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int D = (int) K->ne[0];
    GGML_ASSERT(K->ne[3] == 1 && D % cfg.qb == 0 && D % cfg.kb == 0 && D % cfg.vb == 0 && D / cfg.vb <= 8);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int64_t nrows = K->ne[1] * K->ne[2];
    ggml_cuda_pool_alloc<float> kq(ctx.pool(), nrows * D);
    ggml_cuda_pool_alloc<float> vq(ctx.pool(), nrows * D);
    ggml_cuda_pool_alloc<float> vfold(ctx.pool(), nrows * (D / cfg.vb));
    const size_t smem = 2 * D * sizeof(float);
    k_tckv_prep<<<nrows, TCKV_NT, smem, stream>>>((const half *) K->data, kq.get(), nullptr, D, cfg.k_bits, cfg.kb);
    k_tckv_prep<<<nrows, TCKV_NT, smem, stream>>>((const half *) V->data, vq.get(), cfg.p_bits ? vfold.get() : nullptr,
        D, cfg.p_bits ? 8 : 0, cfg.vb);

    const int n_q = (int) Q->ne[1];
    dim3 grid((n_q + TCKV_QG - 1) / TCKV_QG, Q->ne[2], Q->ne[3]);
    auto launch = [&](auto kern) {
        kern<<<grid, TCKV_NT, 0, stream>>>(
            (const char *) Q->data, kq.get(), vq.get(), vfold.get(), mask ? (const char *) mask->data : nullptr,
            (float *) dst->data, n_q, (int) Q->ne[2], (int) K->ne[1], (int) K->ne[2],
            Q->nb[1], Q->nb[2], Q->nb[3],
            mask ? (int) mask->ne[2] : 1, mask ? (int) mask->ne[3] : 1,
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
            dst->nb[1], dst->nb[2], dst->nb[3], scale, cfg);
    };
    if (D == 128) {
        launch(k_tckv_attend<128>);
    } else {
        launch(k_tckv_attend<256>);
    }
    CUDA_CHECK(cudaGetLastError());
}
