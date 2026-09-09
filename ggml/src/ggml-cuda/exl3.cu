// EXL3 (exllamav3) trellis-coded weights: decode gemv, prefill reconstruct + cuBLAS,
// and the 128-block Hadamard input/output transforms.
// The tile decode and the gemv structure follow exllamav3 (MIT, Copyright (c) 2025 Turboderp):
// exllamav3_ext/quant/{exl3_gemv_kernel,hadamard_inner,reconstruct}.cu*.
#include "exl3.cuh"
#include <cstring>
#include "exl3-dq.cuh"
#include "exl3-had.cuh"
#include "exl3-gemv.cuh"
#include "exl3-gemv-int8.cuh"

#if !defined(GGML_USE_HIP)

namespace {

using exl3::FragB;
using exl3::FragC_h;

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

// One warp per 16x16 tile: lane l decodes run t = 8l..8l+7 (rows (l%4)*2 + {0,1,8,9}, cols l/4 and
// l/4 + 8) and writes W[n][k] as half2 pairs.  grid = tiles / 8, 256 threads.
template <int bits, int cb>
__global__ void __launch_bounds__(256) exl3_reconstruct_kernel(const uint8_t * __restrict__ data, half * __restrict__ dst,
        int k, int nt0, int nt1) {
    const int lane = threadIdx.x & 31;
    const size_t tile = size_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int k_tiles = k / 16;
    const int nt = nt0 + int(tile / k_tiles);
    const int kt = int(tile % k_tiles);
    if (nt >= nt1) {
        return;
    }
    const uint32_t * tp = reinterpret_cast<const uint32_t *>(exl3_tile(data, bits, nt, kt, k_tiles));
    uint32_t w[8];
    exl3_int8::ext8w<bits>(tp, lane << 3, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    const half2 v01 = exl3_gemv::exl3_decode_pair<cb>(w[0], w[1]);   // rows r, r+1     col c
    const half2 v23 = exl3_gemv::exl3_decode_pair<cb>(w[2], w[3]);   // rows r+8, r+9   col c
    const half2 v45 = exl3_gemv::exl3_decode_pair<cb>(w[4], w[5]);   // rows r, r+1     col c+8
    const half2 v67 = exl3_gemv::exl3_decode_pair<cb>(w[6], w[7]);   // rows r+8, r+9   col c+8
    const int r = (lane & 3) * 2;
    const int c = lane >> 2;
    half * row0 = dst + (size_t(nt - nt0) * 16 + c) * k + size_t(kt) * 16 + r;
    half * row8 = row0 + size_t(8) * k;
    *reinterpret_cast<half2 *>(row0)     = v01;
    *reinterpret_cast<half2 *>(row0 + 8) = v23;
    *reinterpret_cast<half2 *>(row8)     = v45;
    *reinterpret_cast<half2 *>(row8 + 8) = v67;
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

template <int bits, int cb>
void exl3_gemv_launch(const half * A, const uint8_t * B, float * C, int m, int k, int n, int sms, cudaStream_t stream) {
    // Micro-benchmarked on A100 (bench_gemv.cu): 16 k-splits x 2 tiles/warp with a 4-deep prefetch
    // ring and one block per 32-column group is the best single config across the Qwen3.8 shapes.
    constexpr int WK = 16, WNT = 2, PF = 4;
    const int grid = n / (WNT * 16);
    GGML_UNUSED(sms);
    exl3_gemv::exl3_gemv_kernel<bits, cb, WK, WNT, PF, false><<<grid, WK * 32, 0, stream>>>(A, B, C, m, k, n);
}

template <int bits, int cb>
void exl3_reconstruct_launch(const uint8_t * data, half * dst, int k, int n0, int n1, cudaStream_t stream) {
    const size_t tiles = size_t(n1 - n0) / 16 * (k / 16);
    exl3_reconstruct_kernel<bits, cb><<<unsigned((tiles + 7) / 8), 256, 0, stream>>>(data, dst, k, n0 / 16, n1 / 16);
}

#define EXL3_DISPATCH_CB(fn, bits, cb, ...)                   \
    switch (bits) {                                           \
        case 1: fn<1, cb>(__VA_ARGS__); break;                \
        case 2: fn<2, cb>(__VA_ARGS__); break;                \
        case 3: fn<3, cb>(__VA_ARGS__); break;                \
        case 4: fn<4, cb>(__VA_ARGS__); break;                \
        case 5: fn<5, cb>(__VA_ARGS__); break;                \
        case 6: fn<6, cb>(__VA_ARGS__); break;                \
        case 7: fn<7, cb>(__VA_ARGS__); break;                \
        case 8: fn<8, cb>(__VA_ARGS__); break;                \
        default: GGML_ABORT("invalid EXL3 bit width");        \
    }

// bits x codebook (2 = mul1, 1 = mcg, 0 = 3inst)
#define EXL3_DISPATCH(fn, bits, cb, ...)                      \
    switch (cb) {                                             \
        case 2: EXL3_DISPATCH_CB(fn, bits, 2, __VA_ARGS__); break; \
        case 1: EXL3_DISPATCH_CB(fn, bits, 1, __VA_ARGS__); break; \
        case 0: EXL3_DISPATCH_CB(fn, bits, 0, __VA_ARGS__); break; \
        default: GGML_ABORT("invalid EXL3 codebook");         \
    }

// ---- int8 activation path (4 bpw, m <= 4) ---------------------------------------------------
// GGML_EXL3_INT8: 0 = off (fp16 tensor-core gemv), 1 = int8 + error-feedback residual everywhere,
// 2 = plain int8 (residual only for the head), unset = per-tensor rule: plain for K <= 6, residual for K >= 7.
// The activation rounding of plain int8 is invisible next to the weight error up to 6 bpw but becomes the
// dominant error at 7-8 bpw (worst-token KLD vs exllamav3 0.07 at K7, 0.56 at K8; residual restores 0.001/0.0005
// for 7% / 1% of decode speed on a 1.7B model).

int exl3_int8_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_EXL3_INT8");
        return e ? atoi(e) : -1;
    }();
    return mode;
}

// whether the int8 path takes the error-feedback residual pass for a weight of this bit width
static bool exl3_int8_resid(int bits, bool head) {
    const int mode = exl3_int8_mode();
    if (mode == 1) {
        return true;
    }
    // The head (output.weight) feeds the logits directly and is DRAM-bound anyway, so it always takes it
    return head || (mode != 2 && bits >= 7);
}

// Self-cleaning per-device counter block (one int per 256-column group), zero at rest.
constexpr size_t EXL3_INT8_MAX_N = 262144;
constexpr size_t EXL3_INT8_COUNTERS = 65536;   // column groups x (token, expert) pairs

int * exl3_int8_counters(int device, cudaStream_t stream) {
    static int * ws[GGML_CUDA_MAX_DEVICES] = {};
    if (ws[device] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc(&ws[device], EXL3_INT8_COUNTERS * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ws[device], 0, EXL3_INT8_COUNTERS * sizeof(int), stream));
    }
    return ws[device];
}

// k-split geometry: ~640 blocks in flight, 128-aligned slices, dynamic shared memory under `cap`
void exl3_int8_geometry(int bits, int nacc, int m, int k, int colblocks, int pairs, size_t cap, int & ksplit, int & nrows, size_t & smem) {
    const int kslices = k / 16;
    ksplit = std::max(1, (640 + colblocks * pairs - 1) / (colblocks * pairs));
    nrows  = std::max(8, ((kslices + ksplit - 1) / ksplit + 7) / 8 * 8);
    nrows  = std::min(nrows, int(cap - exl3_int8::stage_bytes(bits)) / (nacc * 64 + m * 32) / 8 * 8);
    GGML_ASSERT(nrows >= 8 && "EXL3 int8 gemv: device shared memory too small");
    ksplit = (kslices + nrows - 1) / nrows;
    smem   = size_t(nrows) * 16 * (size_t(nacc) * 4 + size_t(m) * 2) + exl3_int8::stage_bytes(bits);
}

// One launch = grid (n/256, ksplit, pairs); dense calls pass pairs = 1 with M tokens, the grouped
// MoE path M = 1 with one (token, expert) pair per block-z.
template <int bits, int cb, int M, bool RESID, bool GROUPED>
void exl3_gemv_int8_launch(ggml_backend_cuda_context & ctx, const uint8_t * B, const float * x, const half * suh, const half * svh,
        float * y, int k, int n, int pairs, exl3_int8::grouped_args ga, cudaStream_t stream) {
    const auto kernel = exl3_int8::gemv_int8_kernel<bits, cb, M, RESID, GROUPED>;
    // Dynamic shared memory budget: the 96 KB design cap, or what the device leaves after this
    // instantiation's static smem (sh_y grows with M; sm_86 opts in to 99 KB total). Function
    // attributes are per device, so set it once on each.
    static size_t cap[GGML_CUDA_MAX_DEVICES] = {};
    if (cap[ctx.device] == 0) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
        cap[ctx.device] = std::min<size_t>(96 * 1024, ggml_cuda_info().devices[ctx.device].smpbo - attr.sharedSizeBytes);
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(cap[ctx.device])));
    }
    const int colblocks = (n + exl3_int8::COLS - 1) / exl3_int8::COLS;
    const int nacc = GROUPED ? (cb == 2 ? 1 : 0) : (RESID ? 2 : 1) * M;
    int ksplit, nrows; size_t smem;
    exl3_int8_geometry(bits, nacc, M, k, colblocks, pairs, cap[ctx.device], ksplit, nrows, smem);
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), size_t(ksplit) * M * pairs * n);
    int * counters = exl3_int8_counters(ctx.device, stream);
    kernel<<<dim3(colblocks, ksplit, pairs), exl3_int8::THREADS, smem, stream>>>(
        B, x, suh, svh, y, partials.get(), counters, k, n, nrows, ga);
}

template <int bits, bool RESID>
void exl3_int8_run(ggml_backend_cuda_context & ctx, const float * x, const half * suh, const uint8_t * B, const half * svh,
        float * y, int m, int k, int n, cudaStream_t stream) {
    const exl3_int8::grouped_args ga {};
    switch (m) {
        case 1: exl3_gemv_int8_launch<bits, 2, 1, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 2: exl3_gemv_int8_launch<bits, 2, 2, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 3: exl3_gemv_int8_launch<bits, 2, 3, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        default: exl3_gemv_int8_launch<bits, 2, 4, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
    }
}

// grouped MoE launch: one block-z per (token, expert slot) pair, expert ids read on the device
template <int bits, int cb>
void exl3_moe_run(ggml_backend_cuda_context & ctx, const float * x, const half * suh, const uint8_t * B, const half * svh,
        float * y, int k, int n, int pairs, exl3_int8::grouped_args ga, cudaStream_t stream) {
    exl3_gemv_int8_launch<bits, cb, 1, false, true>(ctx, B, x, suh, svh, y, k, n, pairs, ga, stream);
}

// MoE decode shapes the grouped kernel takes: F32 in/out, contiguous dst, few pairs
bool exl3_mul_mat_id_fast_shape(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    const int64_t pairs = ids->ne[0] * ids->ne[1];
    return w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 && w->ne[1] <= int64_t(EXL3_INT8_MAX_N) &&
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        pairs >= 1 && pairs <= 64 && size_t(pairs) * ((w->ne[1] + exl3_int8::COLS - 1) / exl3_int8::COLS) <= EXL3_INT8_COUNTERS &&
        dst->src[3] != nullptr && dst->src[4] != nullptr;
}

bool exl3_int8_applicable(int bits, int m, int k, int n) {
    return exl3_int8_mode() != 0 && bits >= 1 && bits <= 8 && m >= 1 && m <= exl3_int8::MAX_M &&
        n % 128 == 0 && k % 128 == 0 && size_t(n) <= EXL3_INT8_MAX_N;
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
    const int cb   = ggml_cuda_exl3_codebook(src0->type);
    EXL3_DISPATCH(exl3_reconstruct_launch, bits, cb, static_cast<const uint8_t *>(src0->data), dst,
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
    const int cb   = ggml_cuda_exl3_codebook(src0->type);
    const half * suh = static_cast<const half *>(dst->src[3]->data);
    const half * svh = static_cast<const half *>(dst->src[2]->data);
    cudaStream_t stream = ctx.stream();
    float * y = static_cast<float *>(dst->data);

    // the int8 path relies on the mul1 codebook being affine in the byte sum
    if (cb == 2 && exl3_int8_applicable(bits, m, k, n)) {
        // int8 activation path: fused input transform, per-slice quantization, fused output transform
        const uint8_t * B = static_cast<const uint8_t *>(src0->data);
        const float * x = static_cast<const float *>(src1->data);
        const bool resid = exl3_int8_resid(bits, strcmp(src0->name, "output.weight") == 0);
        switch (bits) {
#define EXL3_INT8_CASE(K) case K: resid ? exl3_int8_run<K, true>(ctx, x, suh, B, svh, y, m, k, n, stream) \
                                       : exl3_int8_run<K, false>(ctx, x, suh, B, svh, y, m, k, n, stream); break;
            EXL3_INT8_CASE(1) EXL3_INT8_CASE(2) EXL3_INT8_CASE(3) EXL3_INT8_CASE(4)
            EXL3_INT8_CASE(5) EXL3_INT8_CASE(6) EXL3_INT8_CASE(7) EXL3_INT8_CASE(8)
#undef EXL3_INT8_CASE
            default: GGML_ABORT("EXL3 int8 path: unsupported bit width %d", bits);
        }
        return;
    }

    // input transform: xh = had128(x * suh) / sqrt(128), F16 [m][k]
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(m) * k);
    exl3_had_in_kernel<<<dim3(k / 128, m), 32, 0, stream>>>(
        static_cast<const float *>(src1->data), suh, xh.get(), k);

    if (m <= EXL3_GEMV_MAX_M) {
        const int sms = ggml_cuda_info().devices[ctx.device].nsm;
        EXL3_DISPATCH(exl3_gemv_launch, bits, cb, xh.get(), static_cast<const uint8_t *>(src0->data), y, m, k, n, sms, stream);
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

bool ggml_cuda_exl3_mul_mat_id_fast(const ggml_tensor * dst) {
    return exl3_int8_mode() != 0 && exl3_mul_mat_id_fast_shape(dst);
}

void ggml_cuda_mul_mat_id_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    const ggml_tensor * svh = dst->src[3], * suh = dst->src[4];
    GGML_ASSERT(exl3_mul_mat_id_fast_shape(dst));
    const int k = int(w->ne[0]), n = int(w->ne[1]);
    const int bits = ggml_cuda_exl3_bits(w->type);
    const int cb   = ggml_cuda_exl3_codebook(w->type);
    exl3_int8::grouped_args ga;
    ga.ids           = static_cast<const int32_t *>(ids->data);
    ga.ids_nb1       = int(ids->nb[1] / sizeof(int32_t));
    ga.n_expert_used = int(ids->ne[0]);
    ga.ne11          = int(x->ne[1]);
    ga.x_nb1         = x->nb[1] / sizeof(float);
    ga.x_nb2         = x->nb[2] / sizeof(float);
    ga.expert_stride = w->nb[2];
    const int pairs = int(ids->ne[0] * ids->ne[1]);
    cudaStream_t stream = ctx.stream();
    EXL3_DISPATCH(exl3_moe_run, bits, cb, ctx, static_cast<const float *>(x->data), static_cast<const half *>(suh->data),
        static_cast<const uint8_t *>(w->data), static_cast<const half *>(svh->data), static_cast<float *>(dst->data), k, n, pairs, ga, stream);
}

#else

bool ggml_cuda_exl3_mul_mat_id_fast(const ggml_tensor *) { return false; }
void ggml_cuda_mul_mat_id_exl3(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("EXL3 is CUDA only"); }
bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor *) { return false; }
void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor *, int64_t, int64_t, half *, cudaStream_t) { GGML_ABORT("EXL3 is CUDA only"); }
void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { GGML_ABORT("EXL3 is CUDA only"); }

#endif
