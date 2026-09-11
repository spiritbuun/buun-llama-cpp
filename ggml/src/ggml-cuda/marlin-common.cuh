#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "common.cuh"

// The vendored Marlin kernel handles one launch of at most MAX_M rows. Above its
// per-block row count (64 for the 4-block instantiation) it runs
// prob_m / m_block_size parallel slices and DROPS any remainder, so every launch
// must present either a whole multiple of 64 rows or a final tail of <= 64 rows
// (which the kernel masks). Both Marlin executors share this split policy.
namespace ggml_cuda_marlin {

constexpr int64_t rows_per_block = 4 * 16;                 // thread_m_blocks == 4
constexpr int64_t max_rows       = rows_per_block * 16;    // vendor max_par == 16

inline int64_t next_m_split(int64_t remaining) {
    if (remaining <= rows_per_block) {
        return remaining;
    }
    return std::min<int64_t>((remaining / rows_per_block) * rows_per_block, max_rows);
}

inline int m_blocks_for(int64_t split) {
    return std::min<int>(4, int((split + 15) / 16));
}

// Above this batch size the executors dequantize the private layout to BF16
// in row chunks and run cuBLAS instead of Marlin: Marlin's weight-only tiles
// reach about half the tensor-core rate of a dense BF16 GEMM at large m.
inline int64_t gemm_min_m() {
    static const int64_t value = [] {
        const char * env = std::getenv("GGML_CUDA_MARLIN_GEMM_MIN_M");
        return env != nullptr ? std::atoll(env) : int64_t(1024);
    }();
    return value;
}

// Marlin consumes scales in the same 64-row lane order as its output tiles.
__device__ __forceinline__ uint32_t scale_source_row(uint32_t dst_row) {
    const uint32_t chunk = dst_row & ~63u;
    const uint32_t lane = dst_row & 63u;
    return chunk + (lane >> 3) + 8u * (lane & 7u);
}

// Dequantize weight rows [row0, row0 + rows) from the Marlin layout into a
// BF16 [rows][k] matrix. A block covers one 64-row Marlin tile column by 128
// weight columns; each thread gathers the 8 packed words that hold its row's
// 32 consecutive columns straight into registers (each word is shared by two
// rows, so the re-read comes from L1) and stores them as 64 contiguous bytes.
// The word walk is the inverse of the executors' unrepack kernels.
template<bool q4>
__global__ void dequant_marlin_tile_bf16(
        const uint32_t * __restrict__ weight,
        const nv_bfloat16 * __restrict__ scale,
        const uint32_t * __restrict__ zero,   // Q4 only
        nv_bfloat16 * __restrict__ dst,
        uint32_t n,
        uint32_t k,
        uint32_t row0) {
    constexpr uint32_t tile_words = q4 ? 128 : 256;   // words per 16 x 64 Marlin tile
    constexpr uint32_t tc_offsets[4] = {0, 1, 8, 9};

    const uint32_t n_tiles = n / 64;
    const uint32_t n_tile  = row0 / 64 + blockIdx.y;
    const uint32_t k0      = blockIdx.x * 128;
    const uint32_t r_local = threadIdx.x / 4;
    const uint32_t seg     = threadIdx.x % 4;          // 32-column segment
    const uint32_t warp_q  = r_local / 16;
    const uint32_t tc_col  = (r_local % 16) % 8;
    const uint32_t high    = (r_local % 16) / 8;       // second row held by each word

    uint8_t codes[32];
#pragma unroll
    for (uint32_t kt = 0; kt < 2; ++kt) {
        const uint32_t * tile = weight + (uint64_t(k0 / 16 + seg * 2 + kt) * n_tiles + n_tile) * tile_words;
#pragma unroll
        for (uint32_t p = 0; p < 4; ++p) {
            if constexpr (q4) {
                // nibble i lands in slot pack_idx[i] = {0,2,4,6,1,3,5,7}: the low
                // row's slots 0..3 hold nibbles {0,4,1,5}, the high row's {2,6,3,7}
                const uint32_t word = tile[(tc_col * 4 + p) * 4 + warp_q];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t nibble = (j & 1u) * 4 + (j >> 1) + 2 * high;
                    codes[kt * 16 + 2 * p + tc_offsets[j]] = (word >> (4 * nibble)) & 0x0f;
                }
            } else {
                // byte i lands in slot pack_idx[i] = {0,2,1,3}
                const uint32_t word = tile[(tc_col * 4 + p) * 8 + warp_q * 2 + high];
                codes[kt * 16 + 2 * p + 0] = (word >>  0) & 0xff;
                codes[kt * 16 + 2 * p + 8] = (word >>  8) & 0xff;
                codes[kt * 16 + 2 * p + 1] = (word >> 16) & 0xff;
                codes[kt * 16 + 2 * p + 9] = (word >> 24) & 0xff;
            }
        }
    }

    const uint32_t row       = n_tile * 64 + r_local;
    const uint32_t scale_row = scale_source_row(row);
    const uint32_t group     = (k0 + seg * 32) / (q4 ? 32 : 128);
    const float s = __bfloat162float(scale[uint64_t(group) * n + scale_row]);
    float offset = 0.0f;
    if constexpr (q4) {
        const uint32_t lane = scale_row & 7u;
        const uint32_t sub  = (lane & 1u) ? 4u + lane / 2u : lane / 2u;
        offset = float((zero[uint64_t(group) * (n / 8) + (scale_row & ~7u) / 8] >> (4 * sub)) & 0x0f);
    }
    nv_bfloat16 * out = dst + uint64_t(row - row0) * k + k0 + seg * 32;
#pragma unroll
    for (uint32_t i = 0; i < 32; i += 8) {
        __align__(16) nv_bfloat16 v[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            const uint8_t c = codes[i + j];
            v[j] = __float2bfloat16(q4 ? (float(c) - offset) * s : float(int8_t(c ^ 0x80u)) * s);
        }
        *reinterpret_cast<uint4 *>(out + i) = *reinterpret_cast<const uint4 *>(v);
    }
}

template<bool q4>
inline void dequant_bf16(
        const void * weight, const void * scale, const void * zero, nv_bfloat16 * dst,
        int64_t n, int64_t k, int64_t row0, int64_t rows, cudaStream_t stream) {
    GGML_ASSERT(row0 % 64 == 0 && rows % 64 == 0 && k % 128 == 0);
    const dim3 grid(k / 128, rows / 64);
    dequant_marlin_tile_bf16<q4><<<grid, 256, 0, stream>>>(
        static_cast<const uint32_t *>(weight), static_cast<const nv_bfloat16 *>(scale),
        static_cast<const uint32_t *>(zero), dst, n, k, row0);
}

} // namespace ggml_cuda_marlin
