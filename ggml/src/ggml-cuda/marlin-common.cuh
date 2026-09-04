#pragma once

#include <algorithm>
#include <cstdint>

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

} // namespace ggml_cuda_marlin
