#pragma once
// Tensor-core int8 prefill attention for plain turbo KV (experimental): int8 QK (C2), int8 PV (C3).
#include "common.cuh"
#include "mma.cuh"

// Internal K scratch row: D signed bytes, then D/128 float scales, padded to 16 bytes.
// The private launch descriptor retains type F16 only to reuse the attention launch ABI;
// no generic F16 operation may consume this buffer.
static constexpr __host__ __device__ int tckv_int8_row_bytes(int D) { return D + 16; }

// C3 V scratch: one tile per (kv head, nbatch_fa keys), channel-major (V^T): D rows of nbatch_fa
// key bytes + 16 pad bytes. Keys in each 32-group are stored in P-fragment order (tckv_int8_pv),
// the pads hold the per-(key, 128-channel block) scales, slot b*nbatch_fa + key.
static constexpr __host__ __device__ int tckv_int8_v_tile_bytes(int D, int nbatch_fa) { return D*(nbatch_fa + 16); }

static bool tckv_int8_pv_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("TCKV_INT8_PV");
        return s && atoi(s) == 1;
    }();
    return enabled;
}

static bool tckv_int8_applicable(const ggml_tensor * dst, int cc) {
    static const bool enabled = [] {
        const char * s = getenv("TCKV_INT8_QK");
        return s && atoi(s) == 1;
    }();
    if (!enabled || !ampere_mma_available(cc) || dst->src[0]->ne[1] <= 4) {
        return false;
    }
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    auto plain_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 ||
               t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_TURBO8_0;
    };
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    return plain_turbo(K->type) && plain_turbo(V->type) && K->ne[0] == V->ne[0] &&
        (K->ne[0] == 128 || K->ne[0] == 256) && dst->src[4] == nullptr && max_bias == 0.0f && softcap == 0.0f;
}

// Logical byte L of a V^T tile row -> key within the tile. Inverse of the P A-fragment order:
// thread (lane%4 == q) holds keys 16t + 8h + 2q + e of a 32-group, packed as byte 16t + 4q + 2h + e.
static __device__ __forceinline__ int tckv_int8_v_key(int L) {
    const int j = L % 4;
    return (L/32)*32 + ((L%32)/16)*16 + (j/2)*8 + 2*((L%16)/4) + (j%2);
}

// Quantize unscaled rotated f32 Q. Fold attention scale into the stored row scale
// AFTER quantization, matching TCKV_Q=8 TCKV_QB=D (no f16 rounding of Q).
// half_off: also store 0.5*sum(codes) per (column, 128-block) at scales[ncols + jc*(D/128) + b],
// the +0.5 term of native turbo8 K codes (value = s*(code + 0.5)).
template<int D, int ncols1, int ncols2, int nwarps, int cols_per_warp, int np, bool half_off = false>
static __device__ __forceinline__ void tckv_int8_load_q(
        const float2 * Q, int * smem, ggml_cuda_mma::tile<16, 8, int> * qb, float * scales,
        float scale, int stride1, int stride2, int jt, int zt_gqa, int nq, int gqa) {
#ifdef TURING_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr int stride = D/4 + 4; // 16-byte padding: eight ldmatrix rows start in distinct bank groups.
    for (int jc = threadIdx.y; jc < ncols1*ncols2; jc += nwarps) {
        const int j = jt*ncols1 + jc/ncols2;
        const int c = jc % ncols2;
        const bool valid = j < nq && zt_gqa*ncols2 + c < gqa;
        const float * row = (const float *) (Q + j*stride1 + c*stride2);
        float v[D/32], amax = 0.0f;
#pragma unroll
        for (int k = 0; k < D/32; ++k) {
            v[k] = valid ? row[k*32 + threadIdx.x] : 0.0f;
            amax = fmaxf(amax, fabsf(v[k]));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        }
        const float s = amax / 127.0f;
        float bsum[D/128] = {};
#pragma unroll
        for (int k = 0; k < D/32; ++k) {
            const float code = s > 0.0f ? rintf(v[k]/s) : 0.0f;
            ((int8_t *) (smem + jc*stride))[k*32 + threadIdx.x] = (int8_t) code;
            bsum[k/4] += code;
        }
        if (threadIdx.x == 0) {
            scales[jc] = s*scale;
        }
        if constexpr (half_off) {
#pragma unroll
            for (int b = 0; b < D/128; ++b) {
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    bsum[b] += __shfl_xor_sync(0xffffffff, bsum[b], o);
                }
                if (threadIdx.x == 0) {
                    scales[ncols1*ncols2 + jc*(D/128) + b] = 0.5f*bsum[b];
                }
            }
        }
    }
    __syncthreads();
    const int j0 = (threadIdx.y / np)*cols_per_warp;
#pragma unroll
    for (int k = 0; k < D/32; ++k) {
        if constexpr (cols_per_warp == 8) {
            tile<8, 8, int> q;
            load_ldmatrix(q, smem + j0*stride + k*8, stride);
            qb[k].x[0] = q.x[0];
            qb[k].x[1] = q.x[1];
        } else {
            load_ldmatrix(qb[k], smem + j0*stride + k*8, stride);
        }
    }
    __syncthreads(); // Q scratch is reused for K/V.
#endif
}

// preloaded: the multi-stage pipeline already cp.async'd the K rows into smem.
// half_off: K codes are native turbo8 (value = sK*(code + 0.5)); q_offs holds 0.5*sum(Q codes).
template<int D, int nwarps, int nbatch, int cols_per_warp, int np, bool oob, bool preloaded, bool half_off, typename TC>
static __device__ __forceinline__ void tckv_int8_qk(
        const half2 * K, int stride_K, int * smem, const ggml_cuda_mma::tile<16, 8, int> * qb,
        const float * q_scales, const float * q_offs, TC * scores, int nkeys) {
#ifdef TURING_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    // smem rows are verbatim copies of the scratch rows: D codes, then D/128 float scales in the pad.
    constexpr int stride = tckv_int8_row_bytes(D)/4;
    constexpr int blocks = D/128;
    if constexpr (!preloaded) {
        const int tid = threadIdx.y*32 + threadIdx.x;
        for (int x = tid; x < nbatch*stride; x += nwarps*32) {
            const int i = x/stride, k = x%stride;
            smem[i*stride + k] = !oob || i < nkeys ? ((const int *) K)[i*stride_K + k] : 0;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i00 = 0; i00 < nbatch; i00 += np*16) {
        const int i0 = i00 + (threadIdx.y % np)*16;
#pragma unroll
        for (int b = 0; b < blocks; ++b) {
            tile<16, 8, int> acc[cols_per_warp == 8 ? 1 : 2];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                tile<16, 8, int> kt;
                load_ldmatrix(kt, smem + i0*stride + b*32 + k*8, stride);
                if constexpr (cols_per_warp == 8) {
                    tile<8, 8, int> q;
                    q.x[0] = qb[b*4 + k].x[0];
                    q.x[1] = qb[b*4 + k].x[1];
                    mma(acc[0], kt, q);
                } else {
                    tile<8, 8, int> lo, hi;
                    lo.x[0] = kt.x[0]; lo.x[1] = kt.x[2];
                    hi.x[0] = kt.x[1]; hi.x[1] = kt.x[3];
                    mma(acc[0], qb[b*4 + k], lo);
                    mma(acc[1], qb[b*4 + k], hi);
                }
            }
#pragma unroll
            for (int l = 0; l < TC::ne; ++l) {
                const int key = i0 + (cols_per_warp == 8 ? TC::get_i(l) : TC::get_j(l));
                float dot = float(acc[l/4].x[l%4]);
                if constexpr (half_off) {
                    const int q = (threadIdx.y / np)*cols_per_warp + (cols_per_warp == 8 ? TC::get_j(l) : TC::get_i(l));
                    dot += q_offs[q*blocks + b];
                }
                scores[i00/(np*16)].x[l] += dot*((const float *) (smem + key*stride + D/4))[b];
            }
        }
#pragma unroll
        for (int l = 0; l < TC::ne; ++l) {
            const int q = (threadIdx.y / np)*cols_per_warp + (cols_per_warp == 8 ? TC::get_j(l) : TC::get_i(l));
            scores[i00/(np*16)].x[l] *= q_scales[q];
        }
    }
    if constexpr (!preloaded) {
        __syncthreads(); // K scratch is reused by the unchanged f16 PV loader.
    }
#endif
}

static __device__ __forceinline__ void tckv_mma_u8s8(
        ggml_cuda_mma::tile<16, 8, int> & D, const ggml_cuda_mma::tile<16, 8, int> & A, const ggml_cuda_mma::tile<8, 8, int> & B) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif
}

// C3: VKQ += P V with int8 tensor cores (cols_per_warp == 16, np == 1). P is the f32 softmax
// numerator in tile<16,16,float> C layout. Per 128-channel block b, W = P*sV[key, b] is quantized
// to u8 per query row (absmax/255, the TCKV_P=8 sim), multiplied by the int8 V^T tile, and the
// int32 result is scaled into the f16 accumulators. half_off: V codes are native turbo8
// (value = sV*(code + 0.5)); the +0.5 term is 0.5*sum(W codes), folded into the epilogue bias.
template<int DV, int nbatch_fa, bool half_off, typename TC, typename TV>
static __device__ __forceinline__ void tckv_int8_pv(const int * tile_V, const TC * P, TV * VKQ_C) {
#ifdef TURING_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    static_assert(TC::I == 16 && TC::J == 16 && TV::I == 16 && TV::J == 8, "bad tiles");
    constexpr int stride  = (nbatch_fa + 16)/4;
    constexpr int ngroups = nbatch_fa/32;
    const int q = threadIdx.x % 4;
#pragma unroll
    for (int b = 0; b < DV/128; ++b) {
        float w[ngroups][2][8];
        float amax[2] = {0.0f, 0.0f};
#pragma unroll
        for (int g = 0; g < ngroups; ++g) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int l = 0; l < 8; l += 2) {
                    // Keys l and l+1 are adjacent pad slots (even slot, slot%4 in {0, 2}).
                    const int slot = b*nbatch_fa + g*32 + t*16 + (l/4)*8 + 2*q;
                    const float2 sv = *(const float2 *) (tile_V + (slot/4)*stride + nbatch_fa/4 + slot%4);
                    w[g][t][l + 0] = P[2*g + t].x[l + 0] * sv.x;
                    w[g][t][l + 1] = P[2*g + t].x[l + 1] * sv.y;
                    amax[(l/2)%2] = fmaxf(amax[(l/2)%2], fmaxf(w[g][t][l], w[g][t][l + 1]));
                }
            }
        }
        float s[2];
        float inv[2];
        float bias[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            amax[r] = fmaxf(amax[r], __shfl_xor_sync(0xffffffff, amax[r], 1));
            amax[r] = fmaxf(amax[r], __shfl_xor_sync(0xffffffff, amax[r], 2));
            s[r]    = amax[r] * (1.0f/255.0f);
            inv[r]  = amax[r] > 0.0f ? 255.0f/amax[r] : 0.0f;
            bias[r] = -12582912.0f * s[r];
        }
        tile<16, 8, int> pa[ngroups];
#pragma unroll
        for (int g = 0; g < ngroups; ++g) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    // Adding 1.5*2^23 rounds to nearest even and leaves the u8 code in the low byte.
                    int c[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        c[j] = __float_as_int(fmaf(w[g][t][(j/2)*4 + 2*r + (j%2)], inv[r], 12582912.0f));
                    }
                    pa[g].x[2*t + r] = __byte_perm(__byte_perm(c[0], c[1], 0x0040), __byte_perm(c[2], c[3], 0x0040), 0x5410);
                }
            }
        }
        if constexpr (half_off) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                unsigned int wsum = 0;
#pragma unroll
                for (int g = 0; g < ngroups; ++g) {
                    wsum = __dp4a((unsigned int) pa[g].x[r],     0x01010101u, wsum);
                    wsum = __dp4a((unsigned int) pa[g].x[2 + r], 0x01010101u, wsum);
                }
                wsum += __shfl_xor_sync(0xffffffff, wsum, 1);
                wsum += __shfl_xor_sync(0xffffffff, wsum, 2);
                bias[r] = (0.5f*float(wsum) - 12582912.0f) * s[r];
            }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            // Accumulators start at the bits of 1.5*2^23: |P V| < 2^22, so the int32 sum reinterpreted
            // as float is 1.5*2^23 + acc exactly and one FMA with bias converts and scales it.
            tile<16, 8, int> acc[2];
#pragma unroll
            for (int l = 0; l < acc[0].ne; ++l) {
                acc[0].x[l] = 0x4B400000;
                acc[1].x[l] = 0x4B400000;
            }
#pragma unroll
            for (int g = 0; g < ngroups; ++g) {
                tile<16, 8, int> vt;
                load_ldmatrix(vt, tile_V + (b*128 + n*16)*stride + g*8, stride);
                tile<8, 8, int> lo, hi;
                lo.x[0] = vt.x[0]; lo.x[1] = vt.x[2];
                hi.x[0] = vt.x[1]; hi.x[1] = vt.x[3];
                tckv_mma_u8s8(acc[0], pa[g], lo);
                tckv_mma_u8s8(acc[1], pa[g], hi);
            }
#pragma unroll
            for (int hc = 0; hc < 2; ++hc) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    VKQ_C[b*8 + n].x[2*hc + r] += make_half2(
                        fmaf(__int_as_float(acc[hc].x[2*r + 0]), s[r], bias[r]),
                        fmaf(__int_as_float(acc[hc].x[2*r + 1]), s[r], bias[r]));
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(tile_V, P, VKQ_C);
    NO_DEVICE_CODE;
#endif
}
