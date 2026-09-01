#include "marlin-q4-a32.cuh"

#if !defined(GGML_USE_HIP)

#include "marlin-repack.cuh"
#include "marlin-vendor/kernel.h"
#include "marlin-vendor/marlin_template.h"

namespace {

__device__ __forceinline__ uint8_t q4_a32_value(const block_q4_a32 & block, uint32_t index) {
    return (block.qs[index / 2] >> (4 * (index % 2))) & 0x0f;
}

// Marlin consumes scales in the same 64-row lane order as its output tiles.
__device__ __forceinline__ uint32_t scale_source_row(uint32_t dst_row) {
    const uint32_t chunk = dst_row & ~63u;
    const uint32_t lane = dst_row & 63u;
    return chunk + (lane >> 3) + 8u * (lane & 7u);
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

__global__ void unrepack_q4_a32_words(
        const uint32_t * marlin_weight,
        uint32_t * raw_weight,
        uint32_t n,
        uint32_t k) {
    constexpr uint32_t tile_k = 16;
    constexpr uint32_t tile_n = 64;
    constexpr uint32_t tile_words = tile_k * tile_n / 8;
    constexpr uint32_t tc_offsets[4] = {0, 1, 8, 9};
    constexpr uint32_t pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t words = uint64_t(n) * k / 8;
    if (index >= words) {
        return;
    }

    const uint32_t n_tiles = n / tile_n;
    const uint32_t tile = index / tile_words;
    const uint32_t local = index % tile_words;
    const uint32_t k_tile = tile / n_tiles;
    const uint32_t n_tile = tile % n_tiles;
    const uint32_t thread = local / 4;
    const uint32_t warp = local % 4;
    const uint32_t tc_col = thread / 4;
    const uint32_t tc_row = (thread % 4) * 2;
    const uint32_t row = n_tile * tile_n + warp * 16 + tc_col;
    const uint32_t packed = marlin_weight[index];

    uint32_t values[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        values[pack_idx[i]] = (packed >> (4 * i)) & 0x0f;
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t element = k_tile * tile_k + tc_row + tc_offsets[i];
        atomicOr(&raw_weight[uint64_t(element / 8) * n + row],
                 values[i] << (4 * (element % 8)));
        atomicOr(&raw_weight[uint64_t(element / 8) * n + row + 8],
                 values[4 + i] << (4 * (element % 8)));
    }
}

__global__ void assemble_q4_a32_canonical(
        const uint32_t * raw_weight,
        const nv_bfloat16 * scale,
        const uint32_t * zero,
        block_q4_a32 * canonical,
        uint32_t n,
        uint32_t k) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks_per_row = k / QK4_A32;
    if (index >= uint64_t(n) * blocks_per_row) {
        return;
    }

    const uint32_t row = index / blocks_per_row;
    const uint32_t block = index % blocks_per_row;
    block_q4_a32 & dst = canonical[index];

#pragma unroll
    for (uint32_t group_in_block = 0; group_in_block < QK4_A32 / QG4_A32; ++group_in_block) {
        const uint32_t group = block * (QK4_A32 / QG4_A32) + group_in_block;
        const uint32_t scale_row = scale_source_row(row);
        dst.d[group_in_block] = *reinterpret_cast<const uint16_t *>(&scale[uint64_t(group) * n + scale_row]);

        const uint32_t dst_base = scale_row & ~7u;
        const uint32_t lane = scale_row & 7u;
        const uint32_t sub = (lane & 1u) ? 4u + lane / 2u : lane / 2u;
        const uint32_t packed_zero = zero[uint64_t(group) * (n / 8) + dst_base / 8];
        const uint8_t zp = (packed_zero >> (4 * sub)) & 0x0f;
        const uint32_t zero_byte = group_in_block / 2;
        const uint32_t zero_shift = 4 * (group_in_block % 2);
        if ((group_in_block & 1u) == 0) {
            dst.z[zero_byte] = zp;
        } else {
            dst.z[zero_byte] |= zp << zero_shift;
        }
    }

#pragma unroll
    for (uint32_t element = 0; element < QK4_A32; element += 2) {
        const uint32_t k0 = block * QK4_A32 + element;
        const uint32_t word0 = raw_weight[uint64_t(k0 / 8) * n + row];
        const uint32_t word1 = raw_weight[uint64_t((k0 + 1) / 8) * n + row];
        dst.qs[element / 2] = ((word0 >> (4 * (k0 % 8))) & 0x0f) |
                             (((word1 >> (4 * ((k0 + 1) % 8))) & 0x0f) << 4);
    }
}

using marlin_fn = void (*)(MARLIN_KERNEL_PARAMS);

template<int M_BLOCKS, bool M_BLOCK_8, int N_BLOCKS, int K_BLOCKS, int THREADS>
constexpr marlin_fn marlin_kernel() {
    return marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(),
        THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M_BLOCK_8, 4, 2, false>;
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

bool ggml_cuda_marlin_q4_a32_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MARLIN_Q4_A32");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_marlin_q4_a32_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_BLACKWELL &&
        m >= 1 && n % 256 == 0 && k % 128 == 0;
}

void ggml_cuda_marlin_q4_a32_repack_upload(
        const void * canonical,
        void * storage,
        int64_t n,
        int64_t k,
        int device,
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
        n, k, device, sms, stream);
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
    const size_t weight_size = size_t(n) * k / 2;
    const size_t scale_size = size_t(n) * k / QG4_A32 * sizeof(nv_bfloat16);
    void * raw_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&raw_weight, weight_size));
    CUDA_CHECK(cudaMemsetAsync(raw_weight, 0, weight_size, stream));
    const uint64_t words = uint64_t(n) * k / 8;
    unrepack_q4_a32_words<<<(words + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint32_t *>(storage), static_cast<uint32_t *>(raw_weight), n, k);
    const uint64_t blocks = uint64_t(n) * (k / QK4_A32);
    assemble_q4_a32_canonical<<<(blocks + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint32_t *>(raw_weight),
        reinterpret_cast<const nv_bfloat16 *>(static_cast<const char *>(storage) + weight_size),
        reinterpret_cast<const uint32_t *>(static_cast<const char *>(storage) + weight_size + scale_size),
        static_cast<block_q4_a32 *>(canonical), n, k);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(raw_weight));
}

void ggml_cuda_marlin_q4_a32_prepare(
        const void * canonical,
        void * raw_weight,
        void * marlin_weight,
        void * marlin_scale,
        void * marlin_zero,
        int64_t n,
        int64_t k,
        int device,
        int sms,
        cudaStream_t stream) {
    GGML_ASSERT(n % 256 == 0 && k % 128 == 0);
    const uint64_t work = uint64_t(n) * k / 2;
    extract_q4_a32_marlin_inputs<<<(work + 255) / 256, 256, 0, stream>>>(
        static_cast<const block_q4_a32 *>(canonical),
        static_cast<uint32_t *>(raw_weight),
        static_cast<nv_bfloat16 *>(marlin_scale),
        static_cast<uint32_t *>(marlin_zero), n, k);

    int max_shared = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    constexpr auto repack = marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, false>;
    CUDA_CHECK(cudaFuncSetAttribute(repack, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
    repack<<<sms, marlin::repack_threads, max_shared, stream>>>(
        static_cast<const uint32_t *>(raw_weight), nullptr,
        static_cast<uint32_t *>(marlin_weight), k, n);
}

void ggml_cuda_marlin_q4_a32_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const void * zero,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int device,
        int sms,
        cudaStream_t stream) {
    int max_shared = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    int64_t remaining = m;
    int64_t offset = 0;
    while (remaining > 0) {
        constexpr int max_m = 4 * 16 * 16;
        const int64_t split = std::min<int64_t>(remaining, max_m);
        const int m_blocks = std::min<int>(4, (split + 15) / 16);
        const bool m_block_8 = split <= 8;
        marlin_fn kernel = select_marlin_kernel(m_blocks, m_block_8);
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
        kernel<<<sms, 256, max_shared, stream>>>(
            reinterpret_cast<const int4 *>(input + offset * k),
            static_cast<const int4 *>(weight),
            reinterpret_cast<int4 *>(output + offset * n),
            nullptr, nullptr, nullptr,
            static_cast<const int4 *>(scale), nullptr,
            static_cast<const int4 *>(zero), nullptr,
            k / QG4_A32, split, n, k, k, locks,
            false, false, false, max_shared);
        offset += split;
        remaining -= split;
    }
}

#endif
