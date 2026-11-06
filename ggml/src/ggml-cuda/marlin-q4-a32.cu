#include "marlin-q4-a32.cuh"

#if !defined(GGML_USE_HIP)

#include "marlin-common.cuh"
#include "marlin-repack.cuh"
#include "marlin-vendor/kernel.h"
#include "marlin-vendor/marlin_template.h"

namespace {

using ggml_cuda_marlin::scale_source_row;

__device__ __forceinline__ uint8_t q4_a32_value(const block_q4_a32 & block, uint32_t index) {
    return (block.qs[index / 2] >> (4 * (index % 2))) & 0x0f;
}

__global__ void extract_q4_a32_marlin_inputs(
        const block_q4_a32 * canonical,
        uint32_t * raw_weight,
        nv_bfloat16 * scale,
        uint32_t * zero,
        uint32_t n,
        uint32_t k) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t weight_words = uint64_t(k / 8) * n;
    if (index < weight_words) {
        const uint32_t k_word = index / n;
        const uint32_t row = index % n;
        const uint32_t element = 8u * k_word;
        const block_q4_a32 & block = canonical[uint64_t(row) * (k / QK4_A32) + element / QK4_A32];
        const uint32_t lane = element % QK4_A32;
        uint32_t packed = 0;
#pragma unroll
        for (uint32_t sub = 0; sub < 8; ++sub) {
            packed |= uint32_t(q4_a32_value(block, lane + sub)) << (4u * sub);
        }
        raw_weight[index] = packed;
    }

    const uint64_t scales = uint64_t(k / QG4_A32) * n;
    if (index < scales) {
        const uint32_t group = index / n;
        const uint32_t src_row = scale_source_row(index % n);
        const block_q4_a32 & block = canonical[uint64_t(src_row) * (k / QK4_A32) + group / (QK4_A32 / QG4_A32)];
        scale[index] = *reinterpret_cast<const nv_bfloat16 *>(&block.d[group % (QK4_A32 / QG4_A32)]);
    }

    const uint64_t zero_words = uint64_t(k / QG4_A32) * (n / 8);
    if (index < zero_words) {
        constexpr uint32_t interleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        const uint32_t group = index / (n / 8);
        const uint32_t dst_base = 8u * (index % (n / 8));
        uint32_t packed = 0;
#pragma unroll
        for (uint32_t sub = 0; sub < 8; ++sub) {
            const uint32_t src_row = scale_source_row(dst_base + interleave[sub]);
            const block_q4_a32 & block = canonical[uint64_t(src_row) * (k / QK4_A32) + group / (QK4_A32 / QG4_A32)];
            const uint32_t group_in_block = group % (QK4_A32 / QG4_A32);
            const uint32_t zp = (block.z[group_in_block / 2] >> (4 * (group_in_block % 2))) & 0x0f;
            packed |= zp << (4u * sub);
        }
        zero[index] = packed;
    }
}

__global__ void gather_q4_a32_canonical(
        const uint32_t * weight, const uint16_t * scale, const uint32_t * zero,
        block_q4_a32 * canonical, uint32_t n, uint32_t k) {
    // Transpose a 128x64 tile through shared memory: contiguous Marlin word
    // loads, then contiguous stores within each canonical block. No arithmetic
    // on weights or scales, including negative scales and asymmetric zeros.
    __shared__ block_q4_a32 tile[64];
    const uint32_t row_tile = blockIdx.x % (n / 64);
    const uint32_t row0 = row_tile * 64;
    const uint32_t kb = blockIdx.x / (n / 64);
#pragma unroll
    for (uint32_t offset = threadIdx.x; offset < 1024; offset += 256) {
        const uint32_t kt = offset / 128;
        const uint32_t local = offset % 128;
        const uint32_t thread = local / 4;
        const uint32_t warp = local % 4;
        const uint32_t row = warp * 16 + thread / 4;
        const uint32_t pair = kt * 8 + thread % 4;
        const uint64_t src = (uint64_t(kb * 8 + kt) * (n / 64) + row_tile) * 128 + local;
        const uint32_t word = weight[src];
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
#pragma unroll
            for (uint32_t hi = 0; hi < 2; ++hi) {
                const uint32_t i = half * 2 + hi;
                tile[row + half * 8].qs[pair + hi * 4] = ((word >> (4 * i)) & 15) |
                    (((word >> (4 * (i + 4))) & 15) << 4);
            }
        }
    }
    const uint32_t row = threadIdx.x % 64;
    const uint32_t g = threadIdx.x / 64;
    const uint32_t group = kb * 4 + g;
    const uint32_t sr = scale_source_row(row0 + row);
    tile[row].d[g] = scale[uint64_t(group) * n + sr];
    if ((g & 1) == 0) {
        const uint32_t lane = sr & 7;
        const uint32_t sub = (lane & 1) ? 4 + lane / 2 : lane / 2;
        const uint32_t z0 = zero[uint64_t(group) * (n / 8) + sr / 8];
        const uint32_t z1 = zero[uint64_t(group + 1) * (n / 8) + sr / 8];
        tile[row].z[g / 2] = ((z0 >> (4 * sub)) & 15) | (((z1 >> (4 * sub)) & 15) << 4);
    }
    __syncthreads();
    constexpr uint32_t halves = sizeof(block_q4_a32) / sizeof(uint16_t);
    for (uint32_t i = threadIdx.x; i < 64 * halves; i += 256) {
        auto * dst = reinterpret_cast<uint16_t *>(&canonical[uint64_t(row0 + i / halves) * (k / 128) + kb]);
        dst[i % halves] = reinterpret_cast<const uint16_t *>(tile)[i];
    }
}

using marlin_fn = void (*)(MARLIN_KERNEL_PARAMS);

template<int M_BLOCKS, bool M_BLOCK_8, int N_BLOCKS, int K_BLOCKS, int THREADS, bool C_F32>
constexpr marlin_fn marlin_kernel() {
    return marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(),
        THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M_BLOCK_8, 4, 2, false, C_F32>;
}

template<bool C_F32>
marlin_fn select_marlin_kernel(int m_blocks, bool m_block_8, int64_t n, int64_t k, int cc) {
    if (m_blocks == 1) {
        // Ampere small-batch tuning: narrower outputs reuse a deeper K tile;
        // wide outputs benefit from more N work per block. Keep the existing
        // dispatch on other architectures. Tile changes alter split-K rounding.
        if (cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE) {
            if (n <= 8192 && k % 256 == 0) {
                return m_block_8 ? marlin_kernel<1, true, 4, 16, 256, C_F32>() :
                                   marlin_kernel<1, false, 4, 16, 256, C_F32>();
            }
            return m_block_8 ? marlin_kernel<1, true, 16, 4, 256, C_F32>() :
                               marlin_kernel<1, false, 16, 4, 256, C_F32>();
        }
        return m_block_8 ? marlin_kernel<1, true, 8, 8, 256, C_F32>() : marlin_kernel<1, false, 8, 8, 256, C_F32>();
    }
    if (m_blocks == 2) return marlin_kernel<2, false, 16, 4, 256, C_F32>();
    if (m_blocks == 3) return marlin_kernel<3, false, 16, 4, 256, C_F32>();
    if (m_blocks == 4) return marlin_kernel<4, false, 16, 4, 256, C_F32>();
    GGML_ABORT("invalid Marlin M block count");
}

} // namespace

bool ggml_cuda_marlin_q4_a32_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MARLIN_Q4_A32");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_marlin_q4_a32_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_BLACKWELL &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
        m >= 1 && n % 256 == 0 && k % 128 == 0;
}

void ggml_cuda_marlin_q4_a32_repack_upload(
        const void * canonical,
        void * storage,
        int64_t n,
        int64_t k,
        int max_shared,
        int sms,
        cudaStream_t stream) {
    const size_t canonical_size = size_t(n) * k / QK4_A32 * sizeof(block_q4_a32);
    const size_t weight_size = size_t(n) * k / 2;
    const size_t scale_size = size_t(n) * k / QG4_A32 * sizeof(nv_bfloat16);
    const size_t raw_size = weight_size;
    void * canonical_device = nullptr;
    void * raw_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&canonical_device, canonical_size));
    CUDA_CHECK(cudaMalloc(&raw_weight, raw_size));
    CUDA_CHECK(cudaMemcpyAsync(canonical_device, canonical, canonical_size, cudaMemcpyHostToDevice, stream));
    ggml_cuda_marlin_q4_a32_prepare(
        canonical_device,
        raw_weight,
        storage,
        static_cast<char *>(storage) + weight_size,
        static_cast<char *>(storage) + weight_size + scale_size,
        n, k, max_shared, sms, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(raw_weight));
    CUDA_CHECK(cudaFree(canonical_device));
}

void ggml_cuda_marlin_q4_a32_unrepack(
        const void * storage,
        void * canonical,
        int64_t n,
        int64_t k,
        cudaStream_t stream) {
    ggml_cuda_marlin_q4_a32_canonical_async(storage, canonical, n, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ggml_cuda_marlin_q4_a32_canonical_async(
        const void * storage, void * canonical, int64_t n, int64_t k, cudaStream_t stream) {
    const size_t weight_size = size_t(n) * k / 2;
    const size_t scale_size = size_t(n) * k / QG4_A32 * sizeof(nv_bfloat16);
    gather_q4_a32_canonical<<<(n / 64) * (k / 128), 256, 0, stream>>>(
        static_cast<const uint32_t *>(storage),
        reinterpret_cast<const uint16_t *>(static_cast<const char *>(storage) + weight_size),
        reinterpret_cast<const uint32_t *>(static_cast<const char *>(storage) + weight_size + scale_size),
        static_cast<block_q4_a32 *>(canonical), n, k);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_marlin_q4_a32_prepare(
        const void * canonical,
        void * raw_weight,
        void * marlin_weight,
        void * marlin_scale,
        void * marlin_zero,
        int64_t n,
        int64_t k,
        int max_shared,
        int sms,
        cudaStream_t stream) {
    GGML_ASSERT(n % 256 == 0 && k % 128 == 0);
    const uint64_t work = uint64_t(n) * k / 2;
    extract_q4_a32_marlin_inputs<<<(work + 255) / 256, 256, 0, stream>>>(
        static_cast<const block_q4_a32 *>(canonical),
        static_cast<uint32_t *>(raw_weight),
        static_cast<nv_bfloat16 *>(marlin_scale),
        static_cast<uint32_t *>(marlin_zero), n, k);

    constexpr auto repack = marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, false>;
    ggml_cuda_set_max_dynamic_smem(repack, max_shared);
    repack<<<sms, marlin::repack_threads, max_shared, stream>>>(
        static_cast<const uint32_t *>(raw_weight), nullptr,
        static_cast<uint32_t *>(marlin_weight), k, n);
}

void ggml_cuda_marlin_q4_a32_dequant_bf16(
        const void * storage,
        nv_bfloat16 * dst,
        int64_t n,
        int64_t k,
        int64_t row0,
        int64_t rows,
        cudaStream_t stream) {
    const char * scale = static_cast<const char *>(storage) + size_t(n) * k / 2;
    const char * zero  = scale + size_t(n) * k / QG4_A32 * sizeof(nv_bfloat16);
    ggml_cuda_marlin::dequant_bf16<true>(storage, scale, zero, dst, n, k, row0, rows, stream);
}

void ggml_cuda_marlin_q4_a32_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const void * zero,
        const void * weight_alt,
        const void * scale_alt,
        const void * zero_alt,
        void * output,
        bool out_f32,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int max_shared,
        int sms,
        cudaStream_t stream) {
    // With weight_alt the kernel computes [weight | weight_alt] as one 2n-wide GEMM.
    const int64_t out_n = weight_alt != nullptr ? 2 * n : n;
    const size_t out_elem = out_f32 ? sizeof(float) : sizeof(nv_bfloat16);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    int64_t remaining = m;
    int64_t offset = 0;
    while (remaining > 0) {
        const int64_t split = ggml_cuda_marlin::next_m_split(remaining);
        const int m_blocks = ggml_cuda_marlin::m_blocks_for(split);
        const bool m_block_8 = split <= 8;
        marlin_fn kernel = out_f32 ? select_marlin_kernel<true>(m_blocks, m_block_8, out_n, k, cc) :
                                     select_marlin_kernel<false>(m_blocks, m_block_8, out_n, k, cc);
        ggml_cuda_set_max_dynamic_smem(kernel, max_shared);
        kernel<<<sms, 256, max_shared, stream>>>(
            reinterpret_cast<const int4 *>(input + offset * k),
            static_cast<const int4 *>(weight),
            reinterpret_cast<int4 *>(static_cast<char *>(output) + offset * out_n * out_elem),
            nullptr, nullptr, nullptr,
            static_cast<const int4 *>(scale), nullptr,
            static_cast<const int4 *>(zero), nullptr,
            k / QG4_A32, split, out_n, k, k, locks,
            false, false, false, max_shared,
            static_cast<const int4 *>(weight_alt), static_cast<const int4 *>(scale_alt),
            static_cast<const int4 *>(zero_alt));
        offset += split;
        remaining -= split;
    }
}

#endif
