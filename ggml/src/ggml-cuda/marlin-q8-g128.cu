#include "marlin-q8-g128.cuh"

#if !defined(GGML_USE_HIP)

#include "marlin-common.cuh"
#include "marlin-repack.cuh"
#include "marlin-vendor/kernel.h"
#include "marlin-vendor/marlin_template.h"

namespace {

using ggml_cuda_marlin::scale_source_row;

__global__ void extract_q8_g128_marlin_inputs(
        const block_q8_0_g128 * canonical,
        uint32_t * raw_weight,
        nv_bfloat16 * scale,
        uint32_t n,
        uint32_t k) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t weight_words = uint64_t(k / 4) * n;
    if (index < weight_words) {
        const uint32_t k_word = index / n;
        const uint32_t row = index % n;
        const uint32_t element = 4u * k_word;
        const block_q8_0_g128 & block = canonical[uint64_t(row) * (k / QK8_0_G128) + element / QK8_0_G128];
        const uint32_t lane = element % QK8_0_G128;
        uint32_t packed = 0;
#pragma unroll
        for (uint32_t sub = 0; sub < 4; ++sub) {
            packed |= uint32_t(uint8_t(block.qs[lane + sub]) ^ 0x80u) << (8u * sub);
        }
        raw_weight[index] = packed;
    }

    const uint64_t scales = uint64_t(k / QK8_0_G128) * n;
    if (index < scales) {
        const uint32_t group = index / n;
        const uint32_t src_row = scale_source_row(index % n);
        const block_q8_0_g128 & block = canonical[uint64_t(src_row) * (k / QK8_0_G128) + group];
        scale[index] = *reinterpret_cast<const nv_bfloat16 *>(&block.d);
    }
}

__global__ void unrepack_q8_g128_words(
        const uint32_t * marlin_weight,
        uint32_t * raw_weight,
        uint32_t n,
        uint32_t k) {
    constexpr uint32_t tile_k = 16;
    constexpr uint32_t tile_n = 64;
    constexpr uint32_t tile_words = tile_k * tile_n / 4;
    constexpr uint32_t tc_offsets[4] = {0, 1, 8, 9};
    constexpr uint32_t pack_idx[4] = {0, 2, 1, 3};

    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t workers = uint64_t(n / tile_n) * (k / tile_k) * 128;
    if (index >= workers) {
        return;
    }

    const uint32_t local = index % 128;
    const uint32_t tile = index / 128;
    const uint32_t n_tiles = n / tile_n;
    const uint32_t k_tile = tile / n_tiles;
    const uint32_t n_tile = tile % n_tiles;
    const uint32_t th = local / 4;
    const uint32_t warp = local % 4;
    const uint32_t tc_col = th / 4;
    const uint32_t tc_row = (th % 4) * 2;
    const uint32_t cur_n = n_tile * tile_n + warp * 16 + tc_col;
    const uint32_t out_offset = tile * tile_words + th * 8 + warp * 2;
    const uint32_t packed0 = marlin_weight[out_offset];
    const uint32_t packed1 = marlin_weight[out_offset + 1];

    uint32_t values[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        values[pack_idx[i]] = (packed0 >> (8u * i)) & 0xffu;
        values[4 + pack_idx[i]] = (packed1 >> (8u * i)) & 0xffu;
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t element = k_tile * tile_k + tc_row + tc_offsets[i];
        atomicOr(&raw_weight[uint64_t(element / 4) * n + cur_n],
                 values[i] << (8u * (element % 4)));
        atomicOr(&raw_weight[uint64_t(element / 4) * n + cur_n + 8],
                 values[4 + i] << (8u * (element % 4)));
    }
}

__global__ void assemble_q8_g128_canonical(
        const uint32_t * raw_weight,
        const nv_bfloat16 * scale,
        block_q8_0_g128 * canonical,
        uint32_t n,
        uint32_t k) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks_per_row = k / QK8_0_G128;
    if (index >= uint64_t(n) * blocks_per_row) {
        return;
    }

    const uint32_t row = index / blocks_per_row;
    const uint32_t block = index % blocks_per_row;
    block_q8_0_g128 & dst = canonical[index];
    const uint32_t scale_row = scale_source_row(row);
    dst.d = *reinterpret_cast<const uint16_t *>(&scale[uint64_t(block) * n + scale_row]);

#pragma unroll
    for (uint32_t element = 0; element < QK8_0_G128; ++element) {
        const uint32_t k0 = block * QK8_0_G128 + element;
        const uint32_t word = raw_weight[uint64_t(k0 / 4) * n + row];
        dst.qs[element] = int8_t(((word >> (8u * (k0 % 4))) & 0xffu) ^ 0x80u);
    }
}

using marlin_fn = void (*)(MARLIN_KERNEL_PARAMS);

template<int M_BLOCKS, bool M_BLOCK_8, int N_BLOCKS, int K_BLOCKS, int THREADS>
constexpr marlin_fn marlin_kernel() {
    return marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kU8B128.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(),
        THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M_BLOCK_8, 4, 8, false>;
}

marlin_fn select_marlin_kernel(int m_blocks, bool m_block_8) {
    if (m_blocks == 1) {
        return m_block_8 ? marlin_kernel<1, true, 8, 8, 256>() : marlin_kernel<1, false, 8, 8, 256>();
    }
    if (m_blocks == 2) return marlin_kernel<2, false, 16, 4, 256>();
    if (m_blocks == 3) return marlin_kernel<3, false, 16, 4, 256>();
    if (m_blocks == 4) return marlin_kernel<4, false, 16, 4, 256>();
    GGML_ABORT("invalid Marlin M block count");
}

} // namespace

bool ggml_cuda_marlin_q8_g128_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MARLIN_Q8_G128");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_marlin_q8_g128_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_BLACKWELL &&
        m >= 1 && n % 256 == 0 && k % 128 == 0;
}

void ggml_cuda_marlin_q8_g128_repack_upload(
        const void * canonical,
        void * storage,
        int64_t n,
        int64_t k,
        int max_shared,
        int sms,
        cudaStream_t stream) {
    const size_t canonical_size = size_t(n) * k / QK8_0_G128 * sizeof(block_q8_0_g128);
    const size_t weight_size = size_t(n) * k;
    void * canonical_device = nullptr;
    void * raw_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&canonical_device, canonical_size));
    CUDA_CHECK(cudaMalloc(&raw_weight, weight_size));
    CUDA_CHECK(cudaMemcpyAsync(canonical_device, canonical, canonical_size, cudaMemcpyHostToDevice, stream));

    const uint64_t work = uint64_t(n) * k / 4;
    extract_q8_g128_marlin_inputs<<<(work + 255) / 256, 256, 0, stream>>>(
        static_cast<const block_q8_0_g128 *>(canonical_device),
        static_cast<uint32_t *>(raw_weight),
        reinterpret_cast<nv_bfloat16 *>(static_cast<char *>(storage) + weight_size), n, k);

    constexpr auto repack = marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 8, false, false>;
    CUDA_CHECK(cudaFuncSetAttribute(repack, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
    repack<<<sms, marlin::repack_threads, max_shared, stream>>>(
        static_cast<const uint32_t *>(raw_weight), nullptr,
        static_cast<uint32_t *>(storage), k, n);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(raw_weight));
    CUDA_CHECK(cudaFree(canonical_device));
}

void ggml_cuda_marlin_q8_g128_unrepack(
        const void * storage,
        void * canonical,
        int64_t n,
        int64_t k,
        cudaStream_t stream) {
    const size_t weight_size = size_t(n) * k;
    void * raw_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&raw_weight, weight_size));
    CUDA_CHECK(cudaMemsetAsync(raw_weight, 0, weight_size, stream));
    const uint64_t workers = uint64_t(n / 64) * (k / 16) * 128;
    unrepack_q8_g128_words<<<(workers + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint32_t *>(storage), static_cast<uint32_t *>(raw_weight), n, k);
    const uint64_t blocks = uint64_t(n) * (k / QK8_0_G128);
    assemble_q8_g128_canonical<<<(blocks + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint32_t *>(raw_weight),
        reinterpret_cast<const nv_bfloat16 *>(static_cast<const char *>(storage) + weight_size),
        static_cast<block_q8_0_g128 *>(canonical), n, k);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(raw_weight));
}

void ggml_cuda_marlin_q8_g128_dequant_bf16(
        const void * storage,
        nv_bfloat16 * dst,
        int64_t n,
        int64_t k,
        int64_t row0,
        int64_t rows,
        cudaStream_t stream) {
    const char * scale = static_cast<const char *>(storage) + size_t(n) * k;
    ggml_cuda_marlin::dequant_bf16<false>(storage, scale, nullptr, dst, n, k, row0, rows, stream);
}

void ggml_cuda_marlin_q8_g128_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const void * weight_alt,
        const void * scale_alt,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int max_shared,
        int sms,
        cudaStream_t stream) {
    // With weight_alt the kernel computes [weight | weight_alt] as one 2n-wide GEMM.
    const int64_t out_n = weight_alt != nullptr ? 2 * n : n;
    int64_t remaining = m;
    int64_t offset = 0;
    while (remaining > 0) {
        const int64_t split = ggml_cuda_marlin::next_m_split(remaining);
        const int m_blocks = ggml_cuda_marlin::m_blocks_for(split);
        const bool m_block_8 = split <= 8;
        marlin_fn kernel = select_marlin_kernel(m_blocks, m_block_8);
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
        kernel<<<sms, 256, max_shared, stream>>>(
            reinterpret_cast<const int4 *>(input + offset * k),
            static_cast<const int4 *>(weight),
            reinterpret_cast<int4 *>(output + offset * out_n),
            nullptr, nullptr, nullptr,
            static_cast<const int4 *>(scale), nullptr,
            nullptr, nullptr,
            k / QK8_0_G128, split, out_n, k, k, locks,
            false, false, false, max_shared,
            static_cast<const int4 *>(weight_alt), static_cast<const int4 *>(scale_alt));
        offset += split;
        remaining -= split;
    }
}

#endif
