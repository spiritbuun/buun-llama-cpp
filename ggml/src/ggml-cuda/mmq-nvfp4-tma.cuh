#pragma once

// The packed NVFP4 and MMQ activation layouts are unchanged. TMA brings two
// K256 stages into shared memory while the current stage feeds the same K64
// MMA/reduction sequence as the generic whole-K kernel.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080

namespace {

static_assert(QK_NVFP4 == 64 && sizeof(block_nvfp4) == 36 && sizeof(block_fp4_mmq) == 144);

struct nvfp4_tma_storage {
    alignas(128) int ids[128];
    alignas(128) int x[2][128 * 36];
    alignas(128) int y[2][128 * 36];
    alignas(16) uint64_t full[2];
};

#ifdef BLACKWELL_MMA_AVAILABLE
static __device__ __forceinline__ unsigned nvfp4_smem(const void * p) {
    return unsigned(__cvta_generic_to_shared(p));
}

static __device__ __forceinline__ void nvfp4_tma_wait(uint64_t * barrier, unsigned phase) {
    asm volatile(
        "{ .reg .pred p; AGAIN%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
        "@!p bra AGAIN%=; }" :: "r"(nvfp4_smem(barrier)), "r"(phase) : "memory");
}

static __device__ __forceinline__ void nvfp4_tma_copy(
        int * x, int * y, const CUtensorMap * map, const int * activation,
        int row, int stage, uint64_t * barrier) {
    constexpr int bytes = 128 * 36 * sizeof(int);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(nvfp4_smem(barrier)), "r"(2 * bytes) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(nvfp4_smem(x)), "l"(map), "r"(stage * 36), "r"(row),
           "r"(nvfp4_smem(barrier)) : "memory");
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(nvfp4_smem(y)), "l"(activation), "r"(bytes), "r"(nvfp4_smem(barrier)) : "memory");
}

static __device__ __forceinline__ void nvfp4_tma_dot(const int * x, const int * y, float * sum) {
    using TA = tile<16, 8, int>;
    using TB = tile<8, 8, int>;
    using TC = tile<16, 8, float>;
    constexpr int ntiles = 2, nfrags = 4;
    y += (threadIdx.y % ntiles) * 8 * 36;
    const int ia = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int ib = threadIdx.x / 4;
    const int i0 = (threadIdx.y / ntiles) * 32;
    TA A[ntiles][nfrags];
    uint32_t sa[ntiles][nfrags];
#pragma unroll
    for (int n = 0; n < ntiles; ++n) {
#pragma unroll
        for (int f = 0; f < nfrags; ++f) {
#pragma unroll
            for (int l = 0; l < TA::ne; ++l) {
                // Match load_ldmatrix(A)'s four m8n8 matrices, not the
                // accumulator-oriented tile<int>::get_i/get_j mapping.
                const int row = threadIdx.x / 4 + (l % 2) * 8;
                const int col = threadIdx.x % 4 + (l / 2) * 4;
                A[n][f].x[l] = x[(i0 + n * 16 + row) * 36 + f * 9 + 1 + col];
            }
            sa[n][f] = x[(i0 + n * 16 + ia) * 36 + f * 9];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < 128; j0 += 16) {
        TB B[nfrags];
        uint32_t sb[nfrags];
#pragma unroll
        for (int f = 0; f < nfrags; ++f) {
            load_generic(B[f], y + 4 + j0 * 36 + f * 8, 36);
            sb[f] = y[(j0 + ib) * 36 + f];
        }
#pragma unroll
        for (int n = 0; n < ntiles; ++n) {
#pragma unroll
            for (int f = 0; f < nfrags; ++f) {
                // A fresh MMA accumulator per K64 preserves generic MMQ's
                // rounding; accumulating directly into sum is not equivalent.
                TC C = {};
                mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C, A[n][f], B[f], sa[n][f], sb[f]);
#pragma unroll
                for (int l = 0; l < TC::ne; ++l) {
                    sum[(j0 / 8 + n) * TC::ne + l] += C.x[l];
                }
            }
        }
    }
}
#endif

static __global__ __launch_bounds__(256, 1) void mul_mat_nvfp4_tma(
#ifdef BLACKWELL_MMA_AVAILABLE
        const __grid_constant__ CUtensorMap map,
#else
        const CUtensorMap map,
#endif
        const int * activation,
        const float * scale, float * dst, int n, int m, int k) {
#ifdef BLACKWELL_MMA_AVAILABLE
    const int mt = (m + 127) / 128, nt = n / 128;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    int it, jt;
    if (m >= 1024) {
        const int group = blockIdx.x / (8 * mt), first = group * 8;
        const int width = min(8, nt - first), within = blockIdx.x - group * 8 * mt;
        it = first + within % width;
        jt = within / width;
    } else {
        it = blockIdx.x / mt;
        jt = blockIdx.x % mt;
    }
    extern __shared__ __align__(128) unsigned char shared[];
    auto & s = *reinterpret_cast<nvfp4_tma_storage *>(shared);
    if (tid < 128) {
        s.ids[tid] = tid;
    }
    if (tid == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                :: "r"(nvfp4_smem(s.full + i)) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int * y = activation + jt * 128 * 36;
    if (tid == 0) {
        nvfp4_tma_copy(s.x[0], s.y[0], &map, y, it * 128, 0, s.full);
    }
    float sum[64] = {};
    for (int h = 0; h < k / 256; ++h) {
        const int slot = h % 2;
        nvfp4_tma_wait(s.full + slot, (h / 2) & 1);
        if (tid == 0 && h + 1 < k / 256) {
            nvfp4_tma_copy(s.x[slot ^ 1], s.y[slot ^ 1], &map,
                y + size_t(m) * (h + 1) * 36, it * 128, h + 1, s.full + (slot ^ 1));
        }
        nvfp4_tma_dot(s.x[slot], s.y[slot], sum);
        // All consumers finish before this stage can be reused two steps later.
        __syncthreads();
    }
    ggml_cuda_mmq_write_back_mma<GGML_TYPE_NVFP4, 128, false>(sum, s.ids,
        dst + it * 128 + jt * 128 * n, scale + jt * 128, n, n - it * 128 - 1, m - jt * 128 - 1);
#else
    GGML_UNUSED_VARS(map, activation, scale, dst, n, m, k);
    NO_DEVICE_CODE;
#endif
}

} // namespace
#endif

bool ggml_cuda_mmq_nvfp4_tma(const mmq_args & a, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (device.cc != GGML_CUDA_CC_BLACKWELL || !blackwell_mma_available(device.cc) ||
        device.smpbo < sizeof(nvfp4_tma_storage) ||
        a.ids_dst || a.expert_bounds || !a.y_scale ||
        a.ncols_x % 512 || a.nrows_x % 128 || a.ncols_dst < 128 ||
        a.ncols_y != a.ncols_dst || a.ncols_max != a.ncols_dst || a.nrows_dst != a.nrows_x ||
        a.stride_row_x != a.ncols_x / QK_NVFP4 ||
        a.nchannels_x != 1 || a.nchannels_y != 1 || a.nsamples_x != 1 || a.nsamples_y != 1 ||
        uintptr_t(a.x) % 16 || uintptr_t(a.y) % 16) {
        return false;
    }
    // Dense MMQ allocates J_max extra activation blocks after the final K
    // block. J=128 is selected by the caller, so the unmasked final bulk load
    // stays inside that allocation; writeback masks token tails before scales.
    CUtensorMap map{};
    const uint64_t dims[] = {uint64_t(a.ncols_x / QK_NVFP4 * 9), uint64_t(a.nrows_x)};
    const uint64_t strides[] = {uint64_t(a.ncols_x / QK_NVFP4 * sizeof(block_nvfp4))};
    const uint32_t box[] = {36, 128}, element[] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
        const_cast<char *>(a.x), dims, strides, box, element,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    constexpr size_t bytes = sizeof(nvfp4_tma_storage);
    CUDA_SET_SHARED_MEMORY_LIMIT(mul_mat_nvfp4_tma, bytes);
    mul_mat_nvfp4_tma<<<(a.nrows_x / 128) * ((a.ncols_dst + 127) / 128), dim3(32, 8), bytes, stream>>>(
        map, a.y, a.y_scale, a.dst, int(a.nrows_x), int(a.ncols_dst), int(a.ncols_x));
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED_VARS(a, stream);
    return false;
#endif
}
