// EXL3 (exllamav3) trellis-coded weights: decode gemv, prefill reconstruct + cuBLAS,
// and the 128-block Hadamard input/output transforms.
// The tile decode and the gemv structure follow exllamav3 (MIT, Copyright (c) 2025 Turboderp):
// exllamav3_ext/quant/{exl3_gemv_kernel,hadamard_inner,reconstruct}.cu*.
#include "exl3.cuh"

#define EXL3_DISPATCH_CB(fn, bits, cb, ...)                    \
    switch (bits) {                                          \
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
#define EXL3_DISPATCH(fn, bits, cb, ...)                       \
    switch (cb) {                                            \
        case 2: EXL3_DISPATCH_CB(fn, bits, 2, __VA_ARGS__); break; \
        case 1: EXL3_DISPATCH_CB(fn, bits, 1, __VA_ARGS__); break; \
        case 0: EXL3_DISPATCH_CB(fn, bits, 0, __VA_ARGS__); break; \
        default: GGML_ABORT("invalid EXL3 codebook");         \
    }

#include <cstring>
#include "exl3-dq.cuh"
#include "exl3-had.cuh"
#include "exl3-gemv.cuh"
#include "exl3-gemv-int8.cuh"
#include "exl3-int8-warpk.cuh"
#include "exl3-moe-fused.cuh"
#include "moe-cache.cuh"
#include "moe-tier.cuh"
#if !defined(GGML_USE_HIP)
#include "exl3-gemm.cuh"
#include "exl3-head.cuh"
#endif

namespace {

using exl3::FragB;
using exl3::FragC_h;

constexpr float EXL3_HAD_SCALE = exl3_had::SCALE;

// Runtime CC alone is insufficient: a compatible older cubin/PTX can execute
// on SM86 without the SM86-only kernels or their compact shared-memory layout.
static bool exl3_sm86_available(int cc) {
    return cc == 860 && ggml_cuda_highest_compiled_arch(cc) == 860;
}
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
__device__ __forceinline__ void exl3_had_out(float * __restrict__ y, const half * __restrict__ svh, int n) {
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

__global__ void exl3_had_out_kernel(float * y, const half * svh, int n) {
    exl3_had_out(y, svh, n);
}

__global__ void exl3_had_out_pair(float * y0, const half * s0, float * y1, const half * s1, int n, int n1) {
    if (blockIdx.z) n = n1;
    if (blockIdx.x >= n / 128) return;
    exl3_had_out(blockIdx.z ? y1 : y0, blockIdx.z ? s1 : s0, n);
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

// ---- int8 activation path (up to MAX_M rows per launch) ---------------------------------
// GGML_EXL3_INT8: 0 = off (fp16 tensor-core gemv), 1 = int8 + error-feedback residual everywhere,
// 2 = plain int8 (residual only for the head), unset = per-tensor rule: plain for K <= 6, residual for K >= 7,
// with the SM86 six-bit vocabulary head using F16 activations and F32 accumulation at M=1..13.
// The residual limits additional activation error for high-bit weights. Plain activations at lower
// bit widths are a precision/performance policy, not exact agreement with the F16-activation executor.

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

// Self-cleaning context/stream counter block (one int per column group and pair), zero at rest.
constexpr size_t EXL3_INT8_MAX_N = 262144;
constexpr size_t EXL3_INT8_COUNTERS = 65536;   // column groups x (token, expert) pairs

int * exl3_int8_counters(ggml_backend_cuda_context & ctx) {
    int * & ws = ctx.exl3_int8_counter_storage[ctx.curr_stream_no];
    if (ws == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ws, EXL3_INT8_COUNTERS * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ws, 0, EXL3_INT8_COUNTERS * sizeof(int), ctx.stream()));
    }
    return ws;
}

// Dynamic shared-memory budget per device: the 96 KB design cap, or what the device leaves after the
// largest instantiation's static smem (sh_y[MAX_M][COLS] + reductions; sm_86 opts in to 99 KB total).
// One value for every instantiation so the k-split below does not depend on M.
size_t exl3_int8_smem_cap(int device) {
    constexpr size_t static_worst = size_t(exl3_int8::MAX_M) * exl3_int8::COLS * sizeof(float) + 4096;
    return std::min<size_t>(96 * 1024, ggml_cuda_info().devices[device].smpbo - static_worst);
}

// k-split geometry: ~640 blocks in flight, 128-aligned slices, rows bounded by the shared-memory cap
// at the worst-case per-row cost (M = MAX_M with residual). Dense cb2 calls pass pairs = 1, so fixed
// K/N and device give the same slices at M = 1..MAX_M. This keeps activation scales and partial-sum
// order stable within this path; it does not establish whole-model speculative trajectory equality.
void exl3_int8_geometry(int bits, int nacc, int m, int k, int colblocks, int pairs, size_t cap, int & ksplit, int & nrows, size_t & smem) {
    constexpr int worst_row_bytes = 2 * exl3_int8::MAX_M * 64 + exl3_int8::MAX_M * 32;
    const int kslices = k / 16;
    ksplit = std::max(1, (640 + colblocks * pairs - 1) / (colblocks * pairs));
    nrows  = std::max(8, ((kslices + ksplit - 1) / ksplit + 7) / 8 * 8);
    nrows  = std::min(nrows, int(cap - exl3_int8::stage_bytes(bits)) / worst_row_bytes / 8 * 8);
    GGML_ASSERT(nrows >= 8 && "EXL3 int8 gemv: device shared memory too small");
    ksplit = (kslices + nrows - 1) / nrows;
    smem   = size_t(nrows) * 16 * (size_t(nacc) * 4 + size_t(m) * 2) + exl3_int8::stage_bytes(bits);
}

// Small N cannot amortize preparation; large split counts exceed the ordered
// reduction's shared-memory budget. K6 qualification covers five/six K groups.
static bool exl3_warpk_shape(int bits, bool residual, int n, int ksplit) {
    return (bits == 4 && !residual && n >= 4096 && ksplit <= 40) ||
           (bits == 6 && residual && n >= 131072 && (ksplit == 5 || ksplit == 6));
}

template <int BITS, bool RESID, bool PAIR, int M = 8>
static void exl3_warpk_launch(ggml_backend_cuda_context & ctx, const uint8_t * weights, const float * x,
        const half * suh, const half * svh, float * output, int k, int n, int rows, int splits, int m,
        exl3_int8::warpk_pair pair = {}, const half * svh1 = nullptr) {
    constexpr int NACC = M * (RESID ? 2 : 1);
    const size_t bytes = size_t(splits) * (8 * NACC + NACC * rows * 32);
    const size_t pair_bytes = PAIR ? size_t(pair.splits) * (8 * NACC + NACC * pair.rows * 32) : 0;
    ggml_cuda_pool_alloc<uint8_t> prepared(ctx.pool(), bytes + pair_bytes);
    if constexpr (PAIR) pair.prepared = prepared.get() + bytes;
    const auto stream = ctx.stream();
    const int max_splits = PAIR ? std::max(splits, pair.splits) : splits;
    const int max_rows = PAIR ? std::max(rows, pair.rows) : rows;
    const int max_n = PAIR ? std::max(n, pair.n) : n;
    if constexpr (M == 16) {
        static bool attributes[GGML_CUDA_MAX_DEVICES] = {};
        if (!attributes[ctx.device]) {
            // The K6 consumer also owns a 4.5 KiB static weight stage.
            const int cap = int(ggml_cuda_info().devices[ctx.device].smpbo) - 8192;
            ggml_cuda_set_max_dynamic_smem(exl3_int8::prepare_warpk<RESID, PAIR, M>, cap);
            if constexpr (BITS == 4) {
                ggml_cuda_set_max_dynamic_smem(exl3_int8::gemv_warpk<4, BITS, RESID, PAIR, M>, cap);
                ggml_cuda_set_max_dynamic_smem(exl3_int8::gemv_warpk<5, BITS, RESID, PAIR, M>, cap);
                ggml_cuda_set_max_dynamic_smem(exl3_int8::gemv_warpk<16, BITS, RESID, PAIR, M>, cap);
            } else {
                ggml_cuda_set_max_dynamic_smem(exl3_int8::gemv_warpk<5, BITS, RESID, PAIR, M>, cap);
                ggml_cuda_set_max_dynamic_smem(exl3_int8::gemv_warpk<6, BITS, RESID, PAIR, M>, cap);
            }
            attributes[ctx.device] = true;
        }
    }
    exl3_int8::prepare_warpk<RESID, PAIR, M><<<dim3(1, max_splits, PAIR ? 2 : 1), M * 32, size_t(max_rows) * 16 * (NACC + 2 * M), stream>>>(
            x, suh, prepared.get(), k, rows, m, pair);
    const dim3 grid(max_n / 32, 1, PAIR ? 2 : 1);
    const size_t smem = size_t(max_splits) * M * 32 * sizeof(float);
    if constexpr (BITS == 4) {
        // Wide output projections favor fewer K warps; narrower projections
        // have more original K groups and benefit from the larger block.
        if (n >= 3 * k) {
            // Five warps divide ten original K groups evenly for wider verify
            // batches. Keep the independently measured eight-row schedule.
            if (M == 16 && splits == 10 && (!PAIR || pair.splits == 10)) {
                exl3_int8::gemv_warpk<5, BITS, RESID, PAIR, M><<<grid, 160, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
            } else {
                exl3_int8::gemv_warpk<4, BITS, RESID, PAIR, M><<<grid, 128, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
            }
        } else {
            if constexpr (M == 16 && !RESID) {
                // Two blocks fit by shared memory, but the unconstrained M16
                // kernel's 81-100 registers otherwise limit residency to one.
                // Drop unused partial rows when that also brings a larger
                // shared-memory projection below the two-block limit.
                const size_t live_smem = size_t(max_splits) * m * 32 * sizeof(float);
                if (smem > 48 * 1024 && live_smem <= 48 * 1024) {
                    exl3_int8::gemv_warpk_m16_compact<PAIR><<<grid, 512, live_smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
                } else if (smem <= 48 * 1024) {
                    exl3_int8::gemv_warpk_m16<PAIR><<<grid, 512, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
                } else {
                    exl3_int8::gemv_warpk<16, BITS, RESID, PAIR, M><<<grid, 512, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
                }
            } else {
                exl3_int8::gemv_warpk<16, BITS, RESID, PAIR, M><<<grid, 512, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
            }
        }
    } else {
        // One useful warp per original K group avoids idle warps in the head.
        if (splits == 5) {
            exl3_int8::gemv_warpk<5, BITS, RESID, PAIR, M><<<grid, 160, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
        } else if (M == 8 && m >= 7 && splits == 6 && n >= 200000) {
            // Large-vocabulary verification benefits from two K groups per
            // warp; smaller heads retain the independently measured schedule.
            exl3_int8::gemv_warpk<3, BITS, RESID, PAIR, M><<<grid, 96, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
        } else {
            exl3_int8::gemv_warpk<6, BITS, RESID, PAIR, M><<<grid, 192, smem, stream>>>(weights, prepared.get(), output, k, n, rows, splits, m, pair);
        }
    }
    if constexpr (PAIR) {
        exl3_had_out_pair<<<dim3(max_n / 128, m, 2), 32, 0, stream>>>(output, svh, pair.output, svh1, n, pair.n);
    } else {
        exl3_had_out_kernel<<<dim3(n / 128, m), 32, 0, stream>>>(output, svh, n);
    }
}

// One launch = grid (n/256, ksplit, pairs); dense calls pass pairs = 1 with M tokens, the grouped
// MoE path M = 1 with one (token, expert) pair per block-z.
template <int bits, int cb, int M, bool RESID, bool GROUPED, bool GLU = false>
void exl3_gemv_int8_launch(ggml_backend_cuda_context & ctx, const uint8_t * B, const float * x, const half * suh, const half * svh,
        float * y, int k, int n, int pairs, exl3_int8::grouped_args ga, cudaStream_t stream, exl3_int8::glu_args glu = {}) {
    const auto kernel = exl3_int8::gemv_int8_kernel<bits, cb, M, RESID, GROUPED, false, exl3_int8::input_mode::inline_quantize, GLU>;
    const size_t cap = M > exl3_int8::MAX_M
        ? std::min(exl3_int8_smem_cap(ctx.device),
                size_t(ggml_cuda_info().devices[ctx.device].smpbo) - size_t(M) * exl3_int8::COLS * sizeof(float) - 4096)
        : exl3_int8_smem_cap(ctx.device);
    // function attributes are per device: opt this instantiation in once on each
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {};
    if (!attr_set[ctx.device]) {
        ggml_cuda_set_max_dynamic_smem(kernel, int(cap));
        attr_set[ctx.device] = true;
    }
    const int colblocks = (n + exl3_int8::COLS - 1) / exl3_int8::COLS;
    // The kernel reserves the accumulator offset even for F16 codebooks.
    const int nacc = (RESID ? 2 : 1) * M;
    // Grouped launches size the split by pair count for occupancy; their activation scales can
    // therefore differ across batches. The fixed dense geometry does not apply to this grouped path.
    int ksplit, nrows; size_t smem;
    // Keep activation scales and partial-sum order identical to smaller batches.
    // Larger specializations are admitted only when this geometry fits their cap.
    exl3_int8_geometry(bits, nacc, M, k, colblocks, pairs,
            exl3_int8_smem_cap(ctx.device), ksplit, nrows, smem);
    if constexpr (M >= 3 && M <= 8 && exl3_int8::sm86_matrix_shape(bits, cb, M, RESID, GROUPED)) {
        if (exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc)) {
            // Byte activations, F16 transform scratch, and the head's weight ring.
            smem = size_t(nrows) * 16 * (nacc + M * 2) + exl3_int8::head_stage_bytes(bits);
        }
    }
    GGML_ASSERT(smem <= cap);
    if constexpr (cb == 2 && !GLU && !GROUPED && ((bits == 4 && !RESID && M >= 4 && M <= 8) || (bits == 6 && RESID && M >= 5 && M <= 8))) {
        if (exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc) && exl3_warpk_shape(bits, RESID, n, ksplit)) {
            exl3_warpk_launch<bits, RESID, false>(ctx, B, x, suh, svh, y, k, n, nrows, ksplit, M);
            return;
        }
    }
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), size_t(ksplit) * M * pairs * n);
    int * counters = exl3_int8_counters(ctx);
    if constexpr (cb == 2 && bits == 6 && M >= 3 && M <= 8 && RESID && !GROUPED && !GLU) {
        if (exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc)) {
            const auto prepare = exl3_int8::gemv_int8_kernel<bits, cb, M, RESID, false, false, exl3_int8::input_mode::prepare>;
            const auto consume = exl3_int8::gemv_int8_kernel<bits, cb, M, RESID, false, false, exl3_int8::input_mode::consume>;
            static bool prep_attr[GGML_CUDA_MAX_DEVICES] = {};
            if (!prep_attr[ctx.device]) {
                ggml_cuda_set_max_dynamic_smem(prepare, int(cap));
                ggml_cuda_set_max_dynamic_smem(consume, int(cap));
                prep_attr[ctx.device] = true;
            }
            ggml_cuda_pool_alloc<uint8_t> prepared(ctx.pool(), size_t(ksplit) * (2 * nacc * sizeof(uint32_t) + nacc * nrows * 16));
            prepare<<<dim3(1, ksplit), exl3_int8::THREADS, smem, stream>>>(
                B, x, suh, svh, y, partials.get(), counters, k, n, nrows, ga, {}, prepared.get(), {});
            const size_t consume_smem = size_t(nacc) * nrows * 16 + exl3_int8::head_stage_bytes(bits);
            consume<<<dim3(colblocks, ksplit), exl3_int8::THREADS, consume_smem, stream>>>(
                B, x, suh, svh, y, partials.get(), counters, k, n, nrows, ga, {}, prepared.get(), {});
            return;
        }
    }
    kernel<<<dim3(colblocks, ksplit, pairs), exl3_int8::THREADS, smem, stream>>>(
        B, x, suh, svh, y, partials.get(), counters, k, n, nrows, ga, {}, nullptr, glu);
}

template <int bits, bool RESID>
void exl3_int8_run(ggml_backend_cuda_context & ctx, const float * x, const half * suh, const uint8_t * B, const half * svh,
        float * y, int m, int k, int n, cudaStream_t stream) {
    const exl3_int8::grouped_args ga {};
    switch (m) {
        case 1: exl3_gemv_int8_launch<bits, 2, 1, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 2: exl3_gemv_int8_launch<bits, 2, 2, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 3: exl3_gemv_int8_launch<bits, 2, 3, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 4: exl3_gemv_int8_launch<bits, 2, 4, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 5: exl3_gemv_int8_launch<bits, 2, 5, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 6: exl3_gemv_int8_launch<bits, 2, 6, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 7: exl3_gemv_int8_launch<bits, 2, 7, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        case 8: exl3_gemv_int8_launch<bits, 2, 8, RESID, false>(ctx, B, x, suh, svh, y, k, n, 1, ga, stream); break;
        default: GGML_ABORT("EXL3 int8 gemv: m = %d exceeds MAX_M", m);
    }
}

// grouped MoE launch: one block-z per (token, expert slot) pair, expert ids read on the device
template <int bits, int cb>
void exl3_moe_run(ggml_backend_cuda_context & ctx, const float * x, const half * suh, const uint8_t * B, const half * svh,
        float * y, int k, int n, int pairs, exl3_int8::grouped_args ga, cudaStream_t stream) {
    // The mul1 codebook uses int8 activations in both executors. Preserve the
    // dense executor's residual policy when an expert takes the grouped path.
    if constexpr (cb == 2) {
        if (exl3_int8_resid(bits, false)) {
            exl3_gemv_int8_launch<bits, cb, 1, true, true>(ctx, B, x, suh, svh, y, k, n, pairs, ga, stream);
            return;
        }
    }
    exl3_gemv_int8_launch<bits, cb, 1, false, true>(ctx, B, x, suh, svh, y, k, n, pairs, ga, stream);
}

// (token, expert) pairs the grouped gemv takes before the per-expert fallback. The grouped kernel
// reads an expert once per pair, the fallback reads each active expert once but pays a launch per
// expert (Qwen3.8-Flash-Next: ~15k launches for a 32-token prompt). Measured crossover on 4x3090
// (top-10 of 512 experts): 32-token prompts 2.2x faster grouped, 64-token 1.8x, 128-token 1.3x,
// equal near 256 tokens (2560 pairs), fallback ahead at 512. Larger prompts want a sorted-token
// tensor-core grouped GEMM instead of either.
constexpr int EXL3_MOE_PAIRS_MAX = 2048;

// MoE shapes the grouped kernel takes: F32 in/out, contiguous dst, pair count under the crossover
bool exl3_mul_mat_id_fast_shape(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    const int64_t pairs = ids->ne[0] * ids->ne[1];
    return w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 && w->ne[1] <= int64_t(EXL3_INT8_MAX_N) &&
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        pairs >= 1 && pairs <= EXL3_MOE_PAIRS_MAX && size_t(pairs) * ((w->ne[1] + exl3_int8::COLS - 1) / exl3_int8::COLS) <= EXL3_INT8_COUNTERS &&
        dst->src[3] != nullptr && dst->src[4] != nullptr;
}

bool exl3_int8_applicable(int bits, int m, int k, int n, int cc) {
    // SM86: bounded speculative batches reuse the existing eight-row kernels
    // instead of reconstructing the entire weight matrix above eight rows.
    // Dense/head sweeps retain M<=16 for K2..4 and M<=13 for K5..8;
    // wider high-bit batches and K1 residual heads can lose to reconstruction.
    // Keep other architectures at their measured crossover.
    const int max_m = exl3_sm86_available(cc) && bits >= 2 ? (bits <= 4 ? 16 : 13) : exl3_int8::MAX_M;
    return exl3_int8_mode() != 0 && bits >= 1 && bits <= 8 && m >= 1 && m <= max_m &&
        n % 128 == 0 && k % 128 == 0 && size_t(n) <= EXL3_INT8_MAX_N;
}

} // namespace

template <int M>
static bool exl3_int8_bundle_launch(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    const auto kernel = exl3_int8::gemv_int8_kernel<4, 2, M, false, false, true>;
    const int k = a->src[0]->ne[0], n0 = a->ne[0], n1 = b->ne[0];
    const int c0 = (n0 + exl3_int8::COLS - 1) / exl3_int8::COLS;
    const int c1 = (n1 + exl3_int8::COLS - 1) / exl3_int8::COLS;
    const size_t cap = std::min(exl3_int8_smem_cap(ctx.device),
            size_t(ggml_cuda_info().devices[ctx.device].smpbo) - M * exl3_int8::COLS * sizeof(float) - 4096);
    int s0, r0, s1, r1;
    size_t sm0, sm1;
    exl3_int8_geometry(4, M, M, k, c0, 1, exl3_int8_smem_cap(ctx.device), s0, r0, sm0);
    exl3_int8_geometry(4, M, M, k, c1, 1, exl3_int8_smem_cap(ctx.device), s1, r1, sm1);
    if constexpr (M >= 4 && M <= 8) {
        sm0 = size_t(r0) * 16 * M * 3;
        sm1 = size_t(r1) * 16 * M * 3;
    }
    if (std::max(sm0, sm1) > cap) {
        return false;
    }
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {};
    if (!attr_set[ctx.device]) {
        ggml_cuda_set_max_dynamic_smem(kernel, int(cap));
        attr_set[ctx.device] = true;
    }
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), size_t(M) * (size_t(s0) * n0 + size_t(s1) * n1));
    int * counters = exl3_int8_counters(ctx);
    const exl3_int8::bundle_args args {
        static_cast<const uint8_t *>(b->src[0]->data), static_cast<const half *>(b->src[3]->data),
        static_cast<const half *>(b->src[2]->data), static_cast<float *>(b->data),
        partials.get() + size_t(M) * s0 * n0, counters + c0, n1, r1, s1, c0 * s0, c1 * s1,
    };
    kernel<<<c0 * s0 + c1 * s1, exl3_int8::THREADS, std::max(sm0, sm1), ctx.stream()>>>(
        static_cast<const uint8_t *>(a->src[0]->data), static_cast<const float *>(a->src[1]->data),
        static_cast<const half *>(a->src[3]->data), static_cast<const half *>(a->src[2]->data),
        static_cast<float *>(a->data), partials.get(), counters, k, n0, r0, {}, args, nullptr, {});
    return true;
}

bool ggml_cuda_exl3_bundle(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    // Measured SM86 mul1/plain-int8 shapes. Each source keeps its own signs;
    // concatenating these weights into an ordinary projection is not equivalent.
    if (!exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc) ||
            exl3_int8_mode() == 0 || exl3_int8_mode() == 1 ||
            a->src[1] != b->src[1] || a->src[0]->type != ggml_exl3_type(4, 2) ||
            b->src[0]->type != ggml_exl3_type(4, 2) ||
            !ggml_cuda_exl3_supports_mul_mat(a) || !ggml_cuda_exl3_supports_mul_mat(b) ||
            !a->src[2] || !a->src[3] || !b->src[2] || !b->src[3] ||
            a->ne[0] > 32768 || b->ne[0] > 32768 ||
            strcmp(a->src[0]->name, "output.weight") == 0 || strcmp(b->src[0]->name, "output.weight") == 0) {
        return false;
    }
    const int m = a->ne[1], k = a->src[0]->ne[0], n = a->ne[0], n1 = b->ne[0];
    if (((m >= 4 && m <= 8) || m == 13) && n >= 4096 && n1 >= 4096 &&
            (n >= 3 * k) == (n1 >= 3 * k)) {
        // Keep each projection's original quantization slices and input signs.
        int splits, rows, splits1, rows1; size_t smem;
        exl3_int8_geometry(4, 8, 8, k, (n + 255) / 256, 1, exl3_int8_smem_cap(ctx.device), splits, rows, smem);
        exl3_int8_geometry(4, 8, 8, k, (n1 + 255) / 256, 1, exl3_int8_smem_cap(ctx.device), splits1, rows1, smem);
        if (exl3_warpk_shape(4, false, n, splits) && exl3_warpk_shape(4, false, n1, splits1)) {
            exl3_int8::warpk_pair pair {
                static_cast<const uint8_t *>(b->src[0]->data), static_cast<const half *>(b->src[3]->data),
                nullptr, static_cast<float *>(b->data), n1, rows1, splits1,
            };
            if (m == 13) {
                exl3_warpk_launch<4, false, true, 16>(ctx, static_cast<const uint8_t *>(a->src[0]->data),
                        static_cast<const float *>(a->src[1]->data),
                        static_cast<const half *>(a->src[3]->data), static_cast<const half *>(a->src[2]->data),
                        static_cast<float *>(a->data), k, n, rows, splits, m, pair,
                        static_cast<const half *>(b->src[2]->data));
                return true;
            }
            // Smaller adaptive widths use the padded eight-row executor.
            for (int row = 0; row < m; row += 8) {
                pair.output = static_cast<float *>(b->data) + size_t(row) * n1;
                exl3_warpk_launch<4, false, true>(ctx, static_cast<const uint8_t *>(a->src[0]->data),
                        static_cast<const float *>(a->src[1]->data) + size_t(row) * k,
                        static_cast<const half *>(a->src[3]->data), static_cast<const half *>(a->src[2]->data),
                        static_cast<float *>(a->data) + size_t(row) * n, k, n, rows, splits,
                        std::min(8, m - row), pair, static_cast<const half *>(b->src[2]->data));
            }
            return true;
        }
    }
    // Preserve the measured scalar fallback outside the matrix executor's shapes.
    if (m == 13 && n == n1) return false;
    switch (a->ne[1]) {
        case 1:  return exl3_int8_bundle_launch<1>(ctx, a, b);
        case 4:  return exl3_int8_bundle_launch<4>(ctx, a, b);
        case 7:  return exl3_int8_bundle_launch<7>(ctx, a, b);
        case 13: return exl3_int8_bundle_launch<13>(ctx, a, b);
        default: return false;
    }
}

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

bool ggml_cuda_exl3_supports_mul_mat_batched(const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (w == nullptr || x == nullptr || w->ne[2] == 1) {
        return ggml_cuda_exl3_supports_mul_mat(dst);
    }
    const ggml_tensor * svh = dst->src[2];
    const ggml_tensor * suh = dst->src[3];
    const int64_t planes = w->ne[2];
    // weight planes matched to src1/dst planes; svh [n, planes], suh [k, planes]
    return ggml_cuda_is_exl3(w->type) && w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 && w->ne[3] == 1 &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == w->ne[0] && x->ne[2] == planes && x->ne[3] == 1 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[2] == planes &&
        (svh == nullptr || (svh->type == GGML_TYPE_F16 && ggml_is_contiguous(svh) && ggml_nelements(svh) == w->ne[1]*planes)) &&
        (suh == nullptr || (suh->type == GGML_TYPE_F16 && ggml_is_contiguous(suh) && ggml_nelements(suh) == w->ne[0]*planes));
}

static void exl3_mul_mat_plane(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->ne[2] == 1) {
        exl3_mul_mat_plane(ctx, src0, src1, dst);
        return;
    }
    GGML_ASSERT(ggml_cuda_exl3_supports_mul_mat_batched(dst));
    if (dst->src[2] == nullptr || dst->src[3] == nullptr) {
        GGML_ABORT("EXL3 MUL_MAT on '%s' is missing its svh/suh sources", src0->name);
    }
    // batched MUL_MAT: one 2-D product per weight plane, as the MUL_MAT_ID fallback slices experts
    for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
        ggml_tensor w = *src0, x = *src1, y = *dst, svh = *dst->src[2], suh = *dst->src[3];
        w.ne[2] = x.ne[2] = y.ne[2] = 1;
        w.data = (char *) src0->data + i2*src0->nb[2];
        x.data = (char *) src1->data + i2*src1->nb[2];
        y.data = (char *) dst->data  + i2*dst->nb[2];
        svh.ne[0] = src0->ne[1]; svh.ne[1] = svh.ne[2] = svh.ne[3] = 1;
        svh.nb[1] = svh.nb[2] = svh.nb[3] = svh.ne[0]*svh.nb[0];
        svh.data = (char *) dst->src[2]->data + i2*svh.nb[1];
        suh.ne[0] = src0->ne[0]; suh.ne[1] = suh.ne[2] = suh.ne[3] = 1;
        suh.nb[1] = suh.nb[2] = suh.nb[3] = suh.ne[0]*suh.nb[0];
        suh.data = (char *) dst->src[3]->data + i2*suh.nb[1];
        y.src[0] = &w; y.src[1] = &x; y.src[2] = &svh; y.src[3] = &suh;
        exl3_mul_mat_plane(ctx, &w, &x, &y);
    }
}

static void exl3_mul_mat_plane(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

#if !defined(GGML_USE_HIP)
    // Keep one head arithmetic policy across single-row decode and verification.
    // Explicit INT8 modes retain their original executors; this is the default
    // SM86 policy, not a drafting-only switch or an expanded-weight cache.
    if (exl3_int8_mode() == -1 && exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc) && bits == 6 && cb == 2 &&
            m >= 1 && m <= 13 && k % 128 == 0 && n >= 131072 && n % 128 == 0 &&
            strcmp(src0->name, "output.weight") == 0) {
        ggml_cuda_pool_alloc<half> transformed(ctx.pool(), size_t(m) * k);
        exl3_had_in_kernel<<<dim3(k / 128, m), 32, 0, stream>>>(
            static_cast<const float *>(src1->data), suh, transformed.get(), k);
        const auto * weights = static_cast<const uint8_t *>(src0->data);
        if (m <= 8) {
            exl3_head::project<1><<<n / 32, 512, 0, stream>>>(transformed.get(), weights, y, m, k, n);
        } else {
            exl3_head::project<2><<<n / 32, 512, 0, stream>>>(transformed.get(), weights, y, m, k, n);
        }
        exl3_had_out_kernel<<<dim3(n / 128, m), 32, 0, stream>>>(y, svh, n);
        return;
    }
#endif

    // the int8 path relies on the mul1 codebook being affine in the byte sum
    if (cb == 2 && exl3_int8_applicable(bits, m, k, n, ggml_cuda_info().devices[ctx.device].cc)) {
        // int8 activation path: fused input transform, per-slice quantization, fused output transform
        const uint8_t * B = static_cast<const uint8_t *>(src0->data);
        const float * x = static_cast<const float *>(src1->data);
        const bool resid = exl3_int8_resid(bits, strcmp(src0->name, "output.weight") == 0);
        if (exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc) && m == 13 && bits == 6 && resid) {
            int splits, rows; size_t smem;
            exl3_int8_geometry(bits, 16, 8, k, (n + 255) / 256, 1,
                    exl3_int8_smem_cap(ctx.device), splits, rows, smem);
            if (exl3_warpk_shape(bits, resid, n, splits)) {
                exl3_warpk_launch<6, true, false, 16>(ctx, B, x, suh, svh, y, k, n, rows, splits, m);
                return;
            }
        }
        if (exl3_sm86_available(ggml_cuda_info().devices[ctx.device].cc) && bits == 4 &&
                m == 13 && n <= 32768 && !resid) {
            const size_t cap = std::min(exl3_int8_smem_cap(ctx.device),
                    size_t(ggml_cuda_info().devices[ctx.device].smpbo) - 13 * exl3_int8::COLS * sizeof(float) - 4096);
            int ksplit, nrows; size_t smem;
            exl3_int8_geometry(bits, 13, 13, k, (n + exl3_int8::COLS - 1) / exl3_int8::COLS,
                    1, exl3_int8_smem_cap(ctx.device), ksplit, nrows, smem);
            if (exl3_warpk_shape(bits, resid, n, ksplit)) {
                exl3_warpk_launch<4, false, false, 16>(ctx, B, x, suh, svh, y, k, n, nrows, ksplit, m);
                return;
            }
            if (smem <= cap) {
                exl3_gemv_int8_launch<4, 2, 13, false, false>(
                        ctx, B, x, suh, svh, y, k, n, 1, {}, stream);
                return;
            }
        }
        for (int row = 0; row < m; row += exl3_int8::MAX_M) {
            const int rows = std::min(exl3_int8::MAX_M, m - row);
            const float * x_rows = x + size_t(row) * k;
            float * y_rows = y + size_t(row) * n;
            switch (bits) {
#define EXL3_INT8_CASE(K) case K: resid ? exl3_int8_run<K, true>(ctx, x_rows, suh, B, svh, y_rows, rows, k, n, stream) \
                                       : exl3_int8_run<K, false>(ctx, x_rows, suh, B, svh, y_rows, rows, k, n, stream); break;
                EXL3_INT8_CASE(1) EXL3_INT8_CASE(2) EXL3_INT8_CASE(3) EXL3_INT8_CASE(4)
                EXL3_INT8_CASE(5) EXL3_INT8_CASE(6) EXL3_INT8_CASE(7) EXL3_INT8_CASE(8)
#undef EXL3_INT8_CASE
                default: GGML_ABORT("EXL3 int8 path: unsupported bit width %d", bits);
            }
        }
        return;
    }

    // input transform: xh = had128(x * suh) / sqrt(128), F16 [m][k]
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(m) * k);
    exl3_had_in_kernel<<<dim3(k / 128, m), 32, 0, stream>>>(
        static_cast<const float *>(src1->data), suh, xh.get(), k);

#if !defined(GGML_USE_HIP)
    // Consume decoded shared tiles instead of reconstructing the entire
    // projection. SM75 wins at short batches, and up to 128 rows on wide
    // outputs; its narrower shapes favor cuBLAS sooner than SM86's do.
    const int cc_gemm = ggml_cuda_info().devices[ctx.device].cc;
    const bool turing_gemm = ggml_cuda_exl3_turing_gemm_supported(
        cc_gemm, ggml_cuda_highest_compiled_arch(cc_gemm), m, n, k);
    if (bits == 4 && cb == 2 &&
            ((exl3_sm86_available(cc_gemm) && m >= 17) || turing_gemm)) {
        const uint8_t * weights = static_cast<const uint8_t *>(src0->data);
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        // Short batches benefit from smaller row tiles. Keep enough blocks for
        // narrow outputs: BM128 halves their grid and can lose to BM32/64.
        if (m <= 32 || (m <= 64 && n / 64 < nsm)) {
            exl3_gemm_kernel<4, 2, 32, 3><<<dim3(n / 64, (m + 31) / 32), 256, 0, stream>>>(weights, xh.get(), y, k, n, m);
        } else if (m <= 64 || (m <= 128 && n / 64 < (3 * nsm + 1) / 2)) {
            exl3_gemm_kernel<4, 2, 64, 2><<<dim3(n / 64, (m + 63) / 64), 256, 0, stream>>>(weights, xh.get(), y, k, n, m);
        } else {
            const dim3 grid(n / 64, (m + 127) / 128);
            const int blocks = grid.x * grid.y;
            const int sms = ggml_cuda_info().devices[ctx.device].nsm;
            // A third resident CTA shortens some launch tails, but its tighter
            // register budget loses when two CTAs/SM finish in as few waves.
            const int waves2 = (blocks + 2 * sms - 1) / (2 * sms);
            const int waves3 = (blocks + 3 * sms - 1) / (3 * sms);
            if (waves3 < waves2) {
                exl3_gemm_kernel<4, 2, 128, 3><<<grid, 256, 0, stream>>>(weights, xh.get(), y, k, n, m);
            } else {
                exl3_gemm_kernel<4, 2, 128, 2><<<grid, 256, 0, stream>>>(weights, xh.get(), y, k, n, m);
            }
        }
        exl3_had_out_kernel<<<dim3(n / 128, m), 32, 0, stream>>>(y, svh, n);
        return;
    }
#endif

#if !defined(GGML_USE_HIP)
    if (m <= EXL3_GEMV_MAX_M && ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc)) {
        const int sms = ggml_cuda_info().devices[ctx.device].nsm;
        EXL3_DISPATCH(exl3_gemv_launch, bits, cb, xh.get(), static_cast<const uint8_t *>(src0->data), y, m, k, n, sms, stream);
    } else
#endif
    {
        // Prefill, or pre-Ampere FP16 decode: reconstruct row chunks and use cuBLAS (F32 accumulate).
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

bool ggml_cuda_mul_mat_exl3_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, ggml_tensor * dst,
        const ggml_tensor * residual, ggml_tensor * sum) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!ggml_cuda_exl3_supports_mul_mat(dst) || dst->src[1] != glu || dst->src[2] == nullptr || dst->src[3] == nullptr ||
            glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_CLAMP || up == nullptr ||
            ggml_get_op_params_i32(glu, 1) != 0 || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) ||
            !ggml_are_same_shape(gate, glu) || !ggml_are_same_shape(up, glu)) {
        return false;
    }
    const int k = int(src0->ne[0]);
    const int n = int(src0->ne[1]);
    const int bits = ggml_cuda_exl3_bits(src0->type);
    // Single-row plain int8 only: the one executor the int8 path reaches for these shapes.
    if (glu->ne[1] != 1 || ggml_cuda_exl3_codebook(src0->type) != 2 || strcmp(src0->name, "output.weight") == 0 ||
            !exl3_int8_applicable(bits, 1, k, n, ggml_cuda_info().devices[ctx.device].cc) || exl3_int8_resid(bits, false)) {
        return false;
    }
    const uint8_t * B = static_cast<const uint8_t *>(src0->data);
    const float * x = static_cast<const float *>(gate->data);
    const half * suh = static_cast<const half *>(dst->src[3]->data);
    const half * svh = static_cast<const half *>(dst->src[2]->data);
    float * y = static_cast<float *>(sum ? sum->data : dst->data);
    const exl3_int8::glu_args args { static_cast<const float *>(up->data), ggml_get_op_params_f32(glu, 3),
        sum ? static_cast<const float *>(residual->data) : nullptr };
    switch (bits) {
#define EXL3_GLU_CASE(K) case K: exl3_gemv_int8_launch<K, 2, 1, false, false, true>(ctx, B, x, suh, svh, y, k, n, 1, {}, ctx.stream(), args); return true;
        EXL3_GLU_CASE(2) EXL3_GLU_CASE(3) EXL3_GLU_CASE(4) EXL3_GLU_CASE(5) EXL3_GLU_CASE(6)
#undef EXL3_GLU_CASE
        default: return false;
    }
}

bool ggml_cuda_exl3_mul_mat_id_fast(const ggml_tensor * dst) {
    return exl3_int8_mode() != 0 && exl3_mul_mat_id_fast_shape(dst);
}

// The grouped kernel keeps few loads in flight: reading zero-copy host memory through an IOMMU
// it pays a translation miss per page and falls to ~4 GB/s, while a streaming copy runs near
// link rate. Routed launches therefore copy their host-resident experts into VRAM scratch first.
static int exl3_stage_slots(int64_t pairs, int64_t n_expert, size_t expert_size) {
    static const int slots = [] {
        int64_t value = 16;
        moe_cache_env_i64("GGML_CUDA_MOE_STAGE", 0, 256, value);
        return int(value);
    }();
    if (expert_size % sizeof(int4) != 0) {
        return 0;
    }
    return int(std::min<int64_t>({slots, pairs, n_expert}));
}

// One block: copies the table to out, then points up to slots distinct selected experts that live
// in host memory [lo, hi) at scratch and lists their sources after the table (null past the last).
static __global__ void exl3_stage_plan(const void * const * table, const int32_t * ids, int n_used, int n_rows,
        int64_t stride, int n_expert, const char * lo, const char * hi, char * scratch, size_t expert_size, int slots,
        const void ** out) {
    extern __shared__ uint32_t used[];
    __shared__ int n_staged;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        out[e] = table[e];
    }
    for (int i = threadIdx.x; i < (n_expert + 31) / 32; i += blockDim.x) {
        used[i] = 0;
    }
    if (threadIdx.x == 0) {
        n_staged = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_used * n_rows; i += blockDim.x) {
        const int32_t e = ids[(i / n_used) * stride + i % n_used];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        const char * p = (const char *) table[e];
        const uint32_t bit = 1u << (e % 32);
        if (p >= lo && p < hi && !(atomicOr(&used[e / 32], bit) & bit)) {
            const int k = atomicAdd(&n_staged, 1);
            if (k < slots) {
                out[n_expert + k] = p;
                out[e] = scratch + k * expert_size;
            }
        }
    }
    __syncthreads();
    for (int k = n_staged + threadIdx.x; k < slots; k += blockDim.x) {
        out[n_expert + k] = nullptr;
    }
}

// grid (blocks per expert, slots): four independent 16-byte loads in flight per thread
static __global__ void exl3_stage_copy(const void * const * src, char * scratch, size_t expert_size) {
    const int4 * s = (const int4 *) src[blockIdx.y];
    if (!s) {
        return;
    }
    int4 * d = (int4 *) (scratch + blockIdx.y * expert_size);
    const size_t n = expert_size / sizeof(int4);
    const size_t step = size_t(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 4 * step) {
        int4 v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (i + j * step < n) {
                v[j] = s[i + j * step];
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (i + j * step < n) {
                d[i + j * step] = v[j];
            }
        }
    }
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
    const void * svh_data = svh->data, * suh_data = suh->data;
    cudaStream_t stream = ctx.stream();
    const int pairs = int(ids->ne[0] * ids->ne[1]);
    ggml_moe_cache_route_table route;
    ggml_cuda_pool_alloc<const void *> stage_table(ctx.pool());
    ggml_cuda_pool_alloc<char> stage_scratch(ctx.pool());
    if (ggml_moe_cache_route_find(w->data, route)) {
        ga.table     = route.table;
        ga.route_log = route.log;
        if (route.scales[0]) {
            svh_data = route.scales[0];
            suh_data = route.scales[1];
        }
        ggml_moe_tier_resolve(route, nullptr, ids, x, stream);
        const int64_t n_expert = w->ne[2];
        const int slots = exl3_stage_slots(pairs, n_expert, w->nb[2]);
        if (slots > 0) {
            if (route.tier) {
                ggml_moe_tier_arena(route.tier, &route.host_lo, &route.host_hi);
            }
            stage_table.alloc(n_expert + slots);
            stage_scratch.alloc(size_t(slots) * w->nb[2]);
            exl3_stage_plan<<<1, 256, (n_expert + 31) / 32 * sizeof(uint32_t), stream>>>(route.table, ga.ids,
                    ga.n_expert_used, int(ids->ne[1]), ga.ids_nb1, int(n_expert), route.host_lo, route.host_hi,
                    stage_scratch.get(), w->nb[2], slots, stage_table.get());
            exl3_stage_copy<<<dim3(64, slots), 256, 0, stream>>>(stage_table.get() + n_expert, stage_scratch.get(), w->nb[2]);
            ga.table = stage_table.get();
        }
    }
    EXL3_DISPATCH(exl3_moe_run, bits, cb, ctx, static_cast<const float *>(x->data), static_cast<const half *>(suh_data),
        static_cast<const uint8_t *>(w->data), static_cast<const half *>(svh_data), static_cast<float *>(dst->data), k, n, pairs, ga, stream);
}

template <int BG, int BD, int BS>
static void exl3_moe_fused_launch(ggml_backend_cuda_context & ctx, const exl3_moe_fused::args & a) {
    const auto kernel = exl3_moe_fused::moe<BG, BD, BS>;
    static int occ[GGML_CUDA_MAX_DEVICES] = {};
    if (occ[ctx.device] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[ctx.device], kernel, exl3_moe_fused::THREADS, 0));
    }
    const int grid = occ[ctx.device] * ggml_cuda_info().devices[ctx.device].nsm;
    kernel<<<grid, exl3_moe_fused::THREADS, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_exl3_moe_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * down, float limit, const ggml_tensor * weights, const ggml_tensor * sh_up,
        const ggml_tensor * sh_gate, const ggml_tensor * sh_down, ggml_tensor * dst, bool dst_aliases_src) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_EXL3_MOE_FUSED");
        return e == nullptr || atoi(e) != 0;
    }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_AMPERE ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE || exl3_int8_mode() == 0) {
        return false;
    }
    const ggml_tensor * wu = up->src[0], * wg = gate->src[0], * wd = down->src[0];
    const ggml_tensor * x = up->src[1], * ids = up->src[2];
    const int bg = ggml_cuda_exl3_bits(wg->type), bd = ggml_cuda_exl3_bits(wd->type);
    const int k1 = int(wg->ne[0]), n1 = int(wg->ne[1]);
    const int n_used = int(ids->ne[0]), n_tok = int(ids->ne[1]);
    // shared expert: plain MUL_MATs on the same rows, n1 wide, one bit width
    const int bs = sh_gate ? ggml_cuda_exl3_bits(sh_gate->src[0]->type) : 0;
    if (sh_gate) {
        const ggml_tensor * sx = sh_gate->src[1];
        for (const ggml_tensor * mm : { sh_up, sh_gate, sh_down }) {
            if (!ggml_cuda_exl3_supports_mul_mat(mm) || mm->src[2] == nullptr || mm->src[3] == nullptr ||
                    ggml_cuda_exl3_codebook(mm->src[0]->type) != 2 || ggml_cuda_exl3_bits(mm->src[0]->type) != bs ||
                    exl3_int8_resid(bs, false)) {
                return false;
            }
        }
        // same rows of the same tensor: equal addresses alone could be a reused buffer
        const ggml_tensor * sroot = sx->view_src ? sx->view_src : sx;
        const ggml_tensor * xroot = x->view_src ? x->view_src : x;
        if ((bs != bg && bs != 4) || sh_up->src[1] != sx || sroot != xroot || sx->data != x->data ||sx->ne[1] != n_tok || sx->nb[1] != x->nb[2] ||
                sh_gate->src[0]->ne[0] != k1 || sh_gate->src[0]->ne[1] != n1 || !ggml_are_same_shape(sh_up->src[0], sh_gate->src[0]) ||
                sh_down->src[0]->ne[0] != n1 || sh_down->src[0]->ne[1] != k1) {
            return false;
        }
    }
    const int n_shared = sh_gate ? n_tok : 0, pairs = n_shared + n_used * n_tok;
    for (const ggml_tensor * mm : { up, gate, down }) {
        ggml_moe_cache_route_table route;
        if (!ggml_cuda_is_exl3(mm->src[0]->type) || ggml_cuda_exl3_codebook(mm->src[0]->type) != 2 ||
                !exl3_mul_mat_id_fast_shape(mm) || ggml_mmid_window_n_local(mm) != 0 ||
                ggml_moe_cache_route_find(mm->src[0]->data, route) ||
                exl3_int8_resid(ggml_cuda_exl3_bits(mm->src[0]->type), false)) {
            return false;
        }
    }
    // a few tokens, gate/up with identical layout, down mapping n1 back to k1
    if (wu->type != wg->type || bg != bd || bg < 2 || bg > 4 || gate->src[1] != x || gate->src[2] != ids || down->src[2] != ids ||
            !ggml_are_same_shape(wu, wg) || wu->nb[2] != wg->nb[2] || wd->ne[0] != n1 || wd->ne[1] != k1 ||
            k1 % (128 * exl3_moe_fused::NG1) != 0 || n1 % (128 * exl3_moe_fused::NG2) != 0 ||
            n_tok > exl3_moe_fused::MAX_TOKENS || x->ne[1] != 1 || x->ne[2] != n_tok || x->ne[3] != 1 ||
            weights->nb[1] != sizeof(float) || dst->ne[0] != k1 || ggml_nrows(dst) != n_tok || !ggml_is_contiguous(dst) ||
            size_t(exl3_moe_fused::sync_count(pairs, k1)) > EXL3_INT8_COUNTERS) {
        return false;
    }

    const int KW1 = k1 / 128 / exl3_moe_fused::NG1, KW2 = n1 / 128 / exl3_moe_fused::NG2;
    ggml_cuda_pool_alloc<float> part1(ctx.pool(), size_t(pairs) * 2 * KW1 * n1);
    ggml_cuda_pool_alloc<float> part2(ctx.pool(), size_t(pairs) * KW2 * k1);
    ggml_cuda_pool_alloc<uint32_t> xq2(ctx.pool(), size_t(pairs) * n1);
    ggml_cuda_pool_alloc<float2> xs2(ctx.pool(), size_t(pairs) * (n1 / 128));
    // items read ids/weights/x until late in the launch, so an aliased dst is written via scratch
    ggml_cuda_pool_alloc<float> out_tmp(ctx.pool());
    if (dst_aliases_src) {
        out_tmp.alloc(size_t(n_tok) * k1);
    }

    exl3_moe_fused::args a {};
    const ggml_tensor * mats[2][3] = { { gate, up, down }, { sh_gate, sh_up, sh_down } };
    for (int s = 0; s < (sh_gate ? 2 : 1); ++s) {
        for (int m = 0; m < 3; ++m) {
            // MUL_MAT_ID carries svh/suh in src[3]/src[4], MUL_MAT in src[2]/src[3]
            const ggml_tensor * mm = mats[s][m];
            a.m[s][m] = { static_cast<const uint8_t *>(mm->src[0]->data), s ? 0 : mm->src[0]->nb[2],
                static_cast<const half *>(mm->src[s ? 3 : 4]->data), static_cast<const half *>(mm->src[s ? 2 : 3]->data) };
        }
    }
    a.x = static_cast<const float *>(x->data);           a.x_nb1 = int(x->nb[2] / sizeof(float));
    a.ids = static_cast<const int32_t *>(ids->data);     a.ids_nb1 = int(ids->nb[1] / sizeof(int32_t));
    a.w = static_cast<const float *>(weights->data);     a.w_nb1 = int(weights->nb[2] / sizeof(float));
    a.n_used = n_used;
    a.n_tok = n_tok;
    a.pairs = pairs;
    a.limit = limit;
    a.out = dst_aliases_src ? out_tmp.get() : static_cast<float *>(dst->data);
    a.part1 = part1.get(); a.part2 = part2.get(); a.xq2 = xq2.get(); a.xs2 = xs2.get();
    a.sync = exl3_int8_counters(ctx);
    a.k1 = k1; a.n1 = n1;
    switch (bg * 10 + bs) {
        case 20: exl3_moe_fused_launch<2, 2, 0>(ctx, a); break;
        case 22: exl3_moe_fused_launch<2, 2, 2>(ctx, a); break;
        case 24: exl3_moe_fused_launch<2, 2, 4>(ctx, a); break;
        case 30: exl3_moe_fused_launch<3, 3, 0>(ctx, a); break;
        case 33: exl3_moe_fused_launch<3, 3, 3>(ctx, a); break;
        case 34: exl3_moe_fused_launch<3, 3, 4>(ctx, a); break;
        case 40: exl3_moe_fused_launch<4, 4, 0>(ctx, a); break;
        case 44: exl3_moe_fused_launch<4, 4, 4>(ctx, a); break;
        default: GGML_ABORT("unreachable");
    }
    if (dst_aliases_src) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, a.out, size_t(n_tok) * k1 * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    return true;
}

// The cache receives CPU-transformed activations and returns untransformed
// dot products. It needs no NVIDIA MMA instructions; keep its accumulation
// order and F16 codebook rounding identical to the CPU fallback on HIP too.
template<int cb>
__device__ __forceinline__ float exl3_cache_value(uint32_t state) {
#if defined(GGML_USE_HIP)
    if constexpr (cb == 2) {
        const uint32_t x = state * 0x83dcd12du;
        const uint16_t sum = 0x6400 + (x & 255) + ((x >> 8) & 255) +
                             ((x >> 16) & 255) + (x >> 24);
        return __half2float(__float2half_rn(fmaf(__half2float(__ushort_as_half(sum)),
            __half2float(__ushort_as_half(0x1eee)), __half2float(__ushort_as_half(0xc931)))));
    } else {
        uint32_t x = cb == 1 ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        return __half2float(__float2half_rn(__half2float(__ushort_as_half(uint16_t(x))) +
                                           __half2float(__ushort_as_half(uint16_t(x >> 16)))));
    }
#else
    return __half2float(exl3::decode_3inst<cb>(state));
#endif
}

template<int bits, int cb>
__global__ void exl3_cache_dot(const uint8_t * weights, const float * activations,
        const int32_t * slots, const int32_t * activation_ids, float * output,
        int k, int n, size_t expert_bytes, int activation_rows) {
    const int col = blockIdx.x*128 + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n) return;
    const int activation = activation_ids ? activation_ids[row] : row % activation_rows;
    const float * x = activations + size_t(activation)*k;
    const uint8_t * w = weights + size_t(slots[row])*expert_bytes;
    float sum = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    // Expose independent tile loads/decodes without changing the FMA order.
    // SM86 sweeps across expert shapes favor four tiles for mul1 2-4 bit,
    // two for 3inst 8-bit. Other combinations regress on some down shapes.
    constexpr int tile_unroll = cb == 2 && bits >= 2 && bits <= 4 ? 4 :
                               cb == 0 && bits == 8 ? 2 : 1;
#pragma unroll tile_unroll
#endif
    for (int kt = 0; kt < k/16; ++kt) {
        const uint32_t * tile = reinterpret_cast<const uint32_t *>(
            w + (size_t(col/16) * (k/16) + kt) * 32 * bits);
        const int c = col%16;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int element = ((c%8)*4+(r%8)/2)*8 + r%2 + (r/8)*2 + (c/8)*4;
            const int pos = ((element+1)*bits + 256*bits - 16) % (256*bits);
            const uint64_t words = (uint64_t(tile[pos/32]) << 32) | tile[(pos/32+1) % (8*bits)];
            const uint32_t state = (words >> (48-pos%32)) & 65535;
            sum = fmaf(exl3_cache_value<cb>(state), x[kt*16+r], sum);
        }
    }
    output[size_t(row)*n+col] = sum;
}

template<int bits, int cb>
void exl3_cache_launch(const void * weights, const float * activations,
        const int32_t * slots, const int32_t * activation_ids, float * output,
        int k, int n, size_t expert_bytes, int rows, int activation_rows, cudaStream_t stream) {
    exl3_cache_dot<bits, cb><<<dim3((n+127)/128, rows), 128, 0, stream>>>(
        static_cast<const uint8_t *>(weights), activations, slots, activation_ids,
        output, k, n, expert_bytes, activation_rows);
}

void ggml_cuda_exl3_cache_mmv(const void * weights, ggml_type type,
        const float * activations, const int32_t * slots, const int32_t * activation_ids,
        float * output, int k, int n, size_t expert_bytes, int rows, int activation_rows, cudaStream_t stream) {
    EXL3_DISPATCH(exl3_cache_launch, ggml_exl3_bits(type), ggml_exl3_codebook(type),
        weights, activations, slots, activation_ids, output, k, n, expert_bytes, rows, activation_rows, stream);
}
