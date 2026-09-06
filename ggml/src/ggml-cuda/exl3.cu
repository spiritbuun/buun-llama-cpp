// EXL3 (exllamav3) trellis-coded weights: decode gemv, prefill reconstruct + cuBLAS,
// and the 128-block Hadamard input/output transforms.
// The tile decode and the gemv structure follow exllamav3 (MIT, Copyright (c) 2025 Turboderp):
// exllamav3_ext/quant/{exl3_gemv_kernel,hadamard_inner,reconstruct}.cu*.
#include "exl3.cuh"
#include "exl3-dq.cuh"
#include "exl3-had.cuh"
#include "exl3-gemv.cuh"
#include "exl3-gemv-int8.cuh"

#if !defined(GGML_USE_HIP)

namespace {

using exl3::FragB;
using exl3::FragC_h;

constexpr int EXL3_CB = 2; // "mul1" codebook (the only one current exllamav3 checkpoints use)
constexpr float EXL3_HAD_SCALE = exl3_had::SCALE;
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
    exl3_had::had128(v0, v1, v2, v3, lane);
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
    exl3_had::had128(v.x, v.y, v.z, v.w, lane);
    const half2 s01 = *reinterpret_cast<const half2 *>(svh + col);
    const half2 s23 = *reinterpret_cast<const half2 *>(svh + col + 2);
    v.x = v.x * EXL3_HAD_SCALE * __low2float(s01);
    v.y = v.y * EXL3_HAD_SCALE * __high2float(s01);
    v.z = v.z * EXL3_HAD_SCALE * __low2float(s23);
    v.w = v.w * EXL3_HAD_SCALE * __high2float(s23);
    *reinterpret_cast<float4 *>(y + base) = v;
}

template <int bits>
void exl3_gemv_launch(const half * A, const uint8_t * B, float * C, int m, int k, int n, int sms, cudaStream_t stream) {
    // Micro-benchmarked on A100 (bench_gemv.cu): 16 k-splits x 2 tiles/warp with a 4-deep prefetch
    // ring and one block per 32-column group is the best single config across the Qwen3.8 shapes.
    constexpr int WK = 16, WNT = 2, PF = 4;
    const int grid = n / (WNT * 16);
    GGML_UNUSED(sms);
    exl3_gemv::exl3_gemv_kernel<bits, WK, WNT, PF, false><<<grid, WK * 32, 0, stream>>>(A, B, C, m, k, n);
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

// ---- int8 activation path (4 bpw, m <= 4) ---------------------------------------------------
// GGML_EXL3_INT8: 0 = off (fp16 tensor-core gemv), 1 = int8 + error-feedback residual, 2 = plain int8.

int exl3_int8_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_EXL3_INT8");
        return e ? atoi(e) : 2;
    }();
    return mode;
}

// Self-cleaning per-device counter block (one int per 256-column group), zero at rest.
constexpr size_t EXL3_INT8_MAX_N = 262144;

int * exl3_int8_counters(int device, cudaStream_t stream) {
    static int * ws[GGML_CUDA_MAX_DEVICES] = {};
    if (ws[device] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc(&ws[device], EXL3_INT8_MAX_N / exl3_int8::COLS * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ws[device], 0, EXL3_INT8_MAX_N / exl3_int8::COLS * sizeof(int), stream));
    }
    return ws[device];
}

template <int M, bool RESID>
void exl3_gemv_int8_launch(const uint8_t * B, const half * xh, const half * svh, float * y, float * partials, int * counters,
        int k, int n, int colblocks, int ksplit, int nrows, size_t smem, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(exl3_int8::gemv_int8_kernel<M, RESID>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        attr_set = true;
    }
    exl3_int8::gemv_int8_kernel<M, RESID><<<dim3(colblocks, ksplit), exl3_int8::THREADS, smem, stream>>>(
        B, xh, svh, y, partials, counters, k, n, nrows);
}

template <bool RESID>
void exl3_int8_run(ggml_backend_cuda_context & ctx, const half * xh, const uint8_t * B, const half * svh,
        float * y, int m, int k, int n, cudaStream_t stream) {
    const int kslices = k / 16;
    const int colblocks = n / exl3_int8::COLS;
    const int nacc = (RESID ? 2 : 1) * m;
    int ksplit = std::max(1, (640 + colblocks - 1) / colblocks);   // ~640 blocks keeps HBM busy
    int nrows  = std::max(4, (kslices + ksplit - 1) / ksplit);
    nrows  = std::min(nrows, (96 * 1024) / (nacc * 64));
    ksplit = (kslices + nrows - 1) / nrows;
    const size_t smem = size_t(nacc) * nrows * 64;
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), size_t(ksplit) * m * n);
    int * counters = exl3_int8_counters(ctx.device, stream);
    switch (m) {
        case 1: exl3_gemv_int8_launch<1, RESID>(B, xh, svh, y, partials.get(), counters, k, n, colblocks, ksplit, nrows, smem, stream); break;
        case 2: exl3_gemv_int8_launch<2, RESID>(B, xh, svh, y, partials.get(), counters, k, n, colblocks, ksplit, nrows, smem, stream); break;
        case 3: exl3_gemv_int8_launch<3, RESID>(B, xh, svh, y, partials.get(), counters, k, n, colblocks, ksplit, nrows, smem, stream); break;
        default: exl3_gemv_int8_launch<4, RESID>(B, xh, svh, y, partials.get(), counters, k, n, colblocks, ksplit, nrows, smem, stream); break;
    }
}

bool exl3_int8_applicable(int bits, int m, int k, int n) {
    return exl3_int8_mode() != 0 && bits == 4 && m >= 1 && m <= exl3_int8::MAX_M &&
        n % exl3_int8::COLS == 0 && k % 128 == 0 && size_t(n) <= EXL3_INT8_MAX_N;
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
    float * y = static_cast<float *>(dst->data);

    // input transform: xh = had128(x * suh) / sqrt(128), F16 [m][k]
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(m) * k);
    exl3_had_in_kernel<<<dim3(k / 128, m), 32, 0, stream>>>(
        static_cast<const float *>(src1->data), suh, xh.get(), k);

    if (exl3_int8_applicable(bits, m, k, n)) {
        // int8 activation path: per-slice quantization, fused output transform
        if (exl3_int8_mode() == 1) {
            exl3_int8_run<true>(ctx, xh.get(), static_cast<const uint8_t *>(src0->data), svh, y, m, k, n, stream);
        } else {
            exl3_int8_run<false>(ctx, xh.get(), static_cast<const uint8_t *>(src0->data), svh, y, m, k, n, stream);
        }
        return;
    }
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
