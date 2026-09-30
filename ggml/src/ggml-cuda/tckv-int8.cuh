#pragma once
// C2a experiment: materialized rotated K, int8 QK, unchanged f16 PV.
#include "common.cuh"
#include "mma.cuh"

// Internal K scratch row: D signed bytes, then D/128 float scales, padded to 16 bytes.
// The private launch descriptor retains type F16 only to reuse the attention launch ABI;
// no generic F16 operation may consume this buffer.
static constexpr __host__ __device__ int tckv_int8_row_bytes(int D) { return D + 16; }

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

// One warp per 128-channel block. Match k_tckv_prep: f16 input, absmax/127,
// round-to-nearest-even, zero block -> zero scale and codes.
template<int D>
static __global__ void tckv_int8_prep_k(const half * src, char * dst) {
    const int lane = threadIdx.x;
    const int b = threadIdx.y;
    const int64_t row = blockIdx.x;
    float v[4], amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = __half2float(src[row*D + b*128 + k*32 + lane]);
        amax = fmaxf(amax, fabsf(v[k]));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    const float s = amax / 127.0f;
    char * out = dst + row*tckv_int8_row_bytes(D);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        ((int8_t *) out)[b*128 + k*32 + lane] = s > 0.0f ? (int8_t) rintf(v[k]/s) : 0;
    }
    if (lane == 0) {
        ((float *) (out + D))[b] = s;
    }
}

// Quantize unscaled rotated f32 Q. Fold attention scale into the stored row scale
// AFTER quantization, matching TCKV_Q=8 TCKV_QB=D (no f16 rounding of Q).
template<int D, int ncols1, int ncols2, int nwarps, int cols_per_warp, int np>
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
#pragma unroll
        for (int k = 0; k < D/32; ++k) {
            ((int8_t *) (smem + jc*stride))[k*32 + threadIdx.x] = s > 0.0f ? (int8_t) rintf(v[k]/s) : 0;
        }
        if (threadIdx.x == 0) {
            scales[jc] = s*scale;
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
template<int D, int nwarps, int nbatch, int cols_per_warp, int np, bool oob, bool preloaded, typename TC>
static __device__ __forceinline__ void tckv_int8_qk(
        const half2 * K, int stride_K, int * smem, const ggml_cuda_mma::tile<16, 8, int> * qb,
        const float * q_scales, TC * scores, int nkeys) {
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
                scores[i00/(np*16)].x[l] += float(acc[l/4].x[l%4])*((const float *) (smem + key*stride + D/4))[b];
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
