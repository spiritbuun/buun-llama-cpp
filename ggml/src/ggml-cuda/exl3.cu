// EXL3 (exllamav3) trellis-coded weights: decode gemv, prefill reconstruct + cuBLAS,
// and the 128-block Hadamard input/output transforms.
// The tile decode and the gemv structure follow exllamav3 (MIT, Copyright (c) 2025 Turboderp):
// exllamav3_ext/quant/{exl3_gemv_kernel,hadamard_inner,reconstruct}.cu*.
#include "exl3.cuh"
#include "exl3-dq.cuh"

#if !defined(GGML_USE_HIP)

namespace {

using exl3::FragB;
using exl3::FragC_h;

constexpr int EXL3_CB = 2; // "mul1" codebook (the only one current exllamav3 checkpoints use)
constexpr float EXL3_HAD_SCALE = 0.088388347648f; // 1/sqrt(128)
constexpr int EXL3_GEMV_MAX_M = 8;

// tile element order (exllamav3 tensor_core_perm): lane t = idx/8, slot j = idx%8
__device__ __forceinline__ void exl3_tile_rc(int idx, int & r, int & c) {
    const int t = idx >> 3;
    const int j = idx & 7;
    r = (t & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);   // k within the tile
    c = (t >> 2) + ((j & 4) ? 8 : 0);                 // n within the tile
}

// n-tile-major tile stream: tile (nt, kt) of a [k, n] tensor
__device__ __forceinline__ const uint32_t * exl3_tile(const uint8_t * data, int bits, int nt, int kt, int k_tiles) {
    return reinterpret_cast<const uint32_t *>(data + (size_t(nt) * k_tiles + kt) * 32 * bits);
}

// ---- reconstruct: rows [n0, n1) of W[n][k] as F16 ------------------------------------------

template <int bits>
__global__ void exl3_reconstruct_kernel(const uint8_t * __restrict__ data, half * __restrict__ dst,
        int k, int n0, int k_tiles) {
    const int nt = blockIdx.x + n0 / 16;     // absolute n tile
    const int kt = blockIdx.y;
    const int idx = threadIdx.x;             // 0..255, one weight
    int r, c;
    exl3_tile_rc(idx, r, c);
    const half v = exl3::dq<bits, EXL3_CB>(exl3_tile(data, bits, nt, kt, k_tiles), idx);
    dst[size_t(blockIdx.x * 16 + c) * k + kt * 16 + r] = v;
}

// ---- Hadamard: 128 elements per warp, Sylvester order (had_hf_r_128_inner) ---------------

__device__ __forceinline__ void exl3_shuffle_had_f4x32(float & h0, float & h1, float & h2, float & h3, const int lane_id) {
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        const float ph0 = __shfl_xor_sync(0xffffffff, h0, i);
        const float ph1 = __shfl_xor_sync(0xffffffff, h1, i);
        const float ph2 = __shfl_xor_sync(0xffffffff, h2, i);
        const float ph3 = __shfl_xor_sync(0xffffffff, h3, i);
        const int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ __forceinline__ void exl3_had4(float & v0, float & v1, float & v2, float & v3) {
    const float s0 = v0 + v1;
    const float d0 = v0 - v1;
    const float s1 = v2 + v3;
    const float d1 = v2 - v3;
    v0 = s0 + s1;
    v1 = d0 + d1;
    v2 = s0 - s1;
    v3 = d0 - d1;
}

// xh[m][k] (F16) = had128(x[m][k] * suh) / sqrt(128); grid (k/128, m), block 32
__global__ void exl3_had_in_kernel(const float * __restrict__ x, const half * __restrict__ suh,
        half * __restrict__ xh, int k) {
    const int lane = threadIdx.x;
    const int col  = blockIdx.x * 128 + lane * 4;
    const size_t base = size_t(blockIdx.y) * k + col;
    const float4 xv = *reinterpret_cast<const float4 *>(x + base);
    const half2 s01 = *reinterpret_cast<const half2 *>(suh + col);
    const half2 s23 = *reinterpret_cast<const half2 *>(suh + col + 2);
    float v0 = xv.x * __low2float(s01);
    float v1 = xv.y * __high2float(s01);
    float v2 = xv.z * __low2float(s23);
    float v3 = xv.w * __high2float(s23);
    exl3_had4(v0, v1, v2, v3);
    exl3_shuffle_had_f4x32(v0, v1, v2, v3, lane);
    half2 * out = reinterpret_cast<half2 *>(xh + base);
    out[0] = __floats2half2_rn(v0 * EXL3_HAD_SCALE, v1 * EXL3_HAD_SCALE);
    out[1] = __floats2half2_rn(v2 * EXL3_HAD_SCALE, v3 * EXL3_HAD_SCALE);
}

// y[m][n] (F32, in place) = had128(y) / sqrt(128) * svh; grid (n/128, m), block 32
__global__ void exl3_had_out_kernel(float * __restrict__ y, const half * __restrict__ svh, int n) {
    const int lane = threadIdx.x;
    const int col  = blockIdx.x * 128 + lane * 4;
    const size_t base = size_t(blockIdx.y) * n + col;
    float4 v = *reinterpret_cast<const float4 *>(y + base);
    exl3_had4(v.x, v.y, v.z, v.w);
    exl3_shuffle_had_f4x32(v.x, v.y, v.z, v.w, lane);
    const half2 s01 = *reinterpret_cast<const half2 *>(svh + col);
    const half2 s23 = *reinterpret_cast<const half2 *>(svh + col + 2);
    v.x = v.x * EXL3_HAD_SCALE * __low2float(s01);
    v.y = v.y * EXL3_HAD_SCALE * __high2float(s01);
    v.z = v.z * EXL3_HAD_SCALE * __low2float(s23);
    v.w = v.w * EXL3_HAD_SCALE * __high2float(s23);
    *reinterpret_cast<float4 *>(y + base) = v;
}

// ---- decode gemv (m <= 8): warps split k, one m16n8k16 MMA pair per tile ------------------

__device__ __forceinline__ void exl3_mma_ab_h(const FragB & a01, const FragB & a23, const FragB & b, FragC_h & c) {
    const uint32_t * a0 = reinterpret_cast<const uint32_t *>(&a01);
    const uint32_t * a1 = reinterpret_cast<const uint32_t *>(&a23);
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    uint32_t * cc = reinterpret_cast<uint32_t *>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]), "r"(bb[0]), "r"(bb[1]));
}

// A: xh [m][k] F16; B: tile stream (n-tile-major); C: y_inner [m][n] F32.
// 256 threads = 8 warps splitting k; each warp covers 4 adjacent n tiles (64 columns).
template <int bits>
__global__ void __launch_bounds__(256) exl3_gemv_kernel(const half * __restrict__ A, const uint8_t * __restrict__ B,
        float * __restrict__ C, int size_m, int size_k, int size_n) {
    constexpr int WK     = 8;
    constexpr int WNT    = 4;
    constexpr int COLS   = WNT * 16;
    constexpr int ROWS   = EXL3_GEMV_MAX_M;
    constexpr int TWORDS = 8 * bits;   // uint32 per tile
    constexpr int FOLD   = 2;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t * B32 = reinterpret_cast<const uint32_t *>(B);
    const half2 * A2 = reinterpret_cast<const half2 *>(A);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = size_t(r0) * (size_k / 2);
    const bool r0_ok = r0 < size_m;

    __shared__ float    sh_red[WK][ROWS][COLS];
    __shared__ uint32_t sh_stage[WK][WNT * TWORDS];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
        FragC_h ch[WNT][2] = {};
        float2  acc[WNT][2] = {};
        for (int i = 0; i < myn; ++i) {
            const int kt = ks0 + i;
            // stage the four tiles of this k slice (tile (nt, kt) is contiguous)
            __syncwarp();
#pragma unroll
            for (int t = 0; t < WNT; ++t) {
                const uint32_t * tp = B32 + (size_t(group * WNT + t) * kslices + kt) * TWORDS;
                for (int w = lane; w < TWORDS; w += 32) {
                    sh_stage[warp][t * TWORDS + w] = __ldcs(tp + w);
                }
            }
            __syncwarp();
            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = size_t(kt) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
#pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                exl3::dq_dispatch<bits, EXL3_CB>(&sh_stage[warp][t * TWORDS], lane * 8, f0, f1);
                exl3_mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_mma_ab_h(a01, a23, f1, ch[t][1]);
            }
            if ((i + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
                for (int t = 0; t < WNT; ++t) {
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc[t][f].x += __low2float(ch[t][f][0]);
                        acc[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
                }
            }
        }
        // cross-warp reduction over the k splits; lane holds row r0, cols t*16 + f*8 + 2(lane%4) (+1)
        if (r0 < ROWS) {
            const int c0 = 2 * (lane & 3);
#pragma unroll
            for (int t = 0; t < WNT; ++t) {
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + c0;
                    sh_red[warp][r0][col + 0] = acc[t][f].x;
                    sh_red[warp][r0][col + 1] = acc[t][f].y;
                }
            }
        }
        __syncthreads();
        const int rows_out = min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += 256) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < WK; ++j) {
                sum += sh_red[j][r][c];
            }
            C[size_t(r) * size_n + group * COLS + c] = sum;
        }
        __syncthreads();
    }
}

template <int bits>
void exl3_gemv_launch(const half * A, const uint8_t * B, float * C, int m, int k, int n, int sms, cudaStream_t stream) {
    const int groups = n / 64;
    const int grid = std::min(groups, 2 * sms);
    exl3_gemv_kernel<bits><<<grid, 256, 0, stream>>>(A, B, C, m, k, n);
}

template <int bits>
void exl3_reconstruct_launch(const uint8_t * data, half * dst, int k, int n0, int n1, cudaStream_t stream) {
    const dim3 grid((n1 - n0) / 16, k / 16);
    exl3_reconstruct_kernel<bits><<<grid, 256, 0, stream>>>(data, dst, k, n0, k / 16);
}

#define EXL3_DISPATCH(fn, bits, ...)                          \
    switch (bits) {                                           \
        case 1: fn<1>(__VA_ARGS__); break;                    \
        case 2: fn<2>(__VA_ARGS__); break;                    \
        case 3: fn<3>(__VA_ARGS__); break;                    \
        case 4: fn<4>(__VA_ARGS__); break;                    \
        case 5: fn<5>(__VA_ARGS__); break;                    \
        case 6: fn<6>(__VA_ARGS__); break;                    \
        case 7: fn<7>(__VA_ARGS__); break;                    \
        case 8: fn<8>(__VA_ARGS__); break;                    \
        default: GGML_ABORT("invalid EXL3 bit width");        \
    }

} // namespace

bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * svh = dst->src[2];
    const ggml_tensor * suh = dst->src[3];
    // The loader probes buffer-type support with a bare MUL_MAT (no scale sources); the real
    // graph always carries svh/suh, which the executor asserts.
    return w != nullptr && x != nullptr &&
        ggml_cuda_is_exl3(w->type) && w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 &&
        w->ne[2] == 1 && w->ne[3] == 1 &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == w->ne[0] &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        (svh == nullptr || (svh->type == GGML_TYPE_F16 && ggml_is_contiguous(svh) && ggml_nelements(svh) == w->ne[1])) &&
        (suh == nullptr || (suh->type == GGML_TYPE_F16 && ggml_is_contiguous(suh) && ggml_nelements(suh) == w->ne[0]));
}

void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor * src0, int64_t n0, int64_t n1, half * dst, cudaStream_t stream) {
    const int bits = ggml_cuda_exl3_bits(src0->type);
    EXL3_DISPATCH(exl3_reconstruct_launch, bits, static_cast<const uint8_t *>(src0->data), dst,
        int(src0->ne[0]), int(n0), int(n1), stream);
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_exl3_supports_mul_mat(dst));
    if (dst->src[2] == nullptr || dst->src[3] == nullptr) {
        GGML_ABORT("EXL3 MUL_MAT on '%s' is missing its svh/suh sources", src0->name);
    }
    const int k = int(src0->ne[0]);
    const int n = int(src0->ne[1]);
    const int m = int(src1->ne[1]);
    const int bits = ggml_cuda_exl3_bits(src0->type);
    const half * suh = static_cast<const half *>(dst->src[3]->data);
    const half * svh = static_cast<const half *>(dst->src[2]->data);
    cudaStream_t stream = ctx.stream();

    // input transform: xh = had128(x * suh) / sqrt(128), F16 [m][k]
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(m) * k);
    exl3_had_in_kernel<<<dim3(k / 128, m), 32, 0, stream>>>(
        static_cast<const float *>(src1->data), suh, xh.get(), k);

    float * y = static_cast<float *>(dst->data);
    if (m <= EXL3_GEMV_MAX_M) {
        const int sms = ggml_cuda_info().devices[ctx.device].nsm;
        EXL3_DISPATCH(exl3_gemv_launch, bits, xh.get(), static_cast<const uint8_t *>(src0->data), y, m, k, n, sms, stream);
    } else {
        // prefill: reconstruct row chunks to F16 and multiply with cuBLAS (F32 accumulate)
        constexpr size_t chunk_bytes = size_t(256) << 20;
        const int rows_per_chunk = int(std::max<int64_t>(128, std::min<int64_t>(n, int64_t(chunk_bytes / (size_t(k) * sizeof(half))) / 128 * 128)));
        ggml_cuda_pool_alloc<half> w(ctx.pool(), size_t(rows_per_chunk) * k);
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        for (int row0 = 0; row0 < n; row0 += rows_per_chunk) {
            const int rows = std::min(rows_per_chunk, n - row0);
            ggml_cuda_exl3_reconstruct_rows(src0, row0, row0 + rows, w.get(), stream);
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, rows, m, k,
                &alpha, w.get(), CUDA_R_16F, k, xh.get(), CUDA_R_16F, k,
                &beta, y + row0, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    // output transform in place: y = had128(y) / sqrt(128) * svh
    exl3_had_out_kernel<<<dim3(n / 128, m), 32, 0, stream>>>(y, svh, n);
}

#else

bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor *) { return false; }
void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor *, int64_t, int64_t, half *, cudaStream_t) { GGML_ABORT("EXL3 is CUDA only"); }
void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { GGML_ABORT("EXL3 is CUDA only"); }

#endif
