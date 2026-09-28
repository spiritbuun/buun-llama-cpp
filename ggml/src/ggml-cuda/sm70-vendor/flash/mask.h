/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit_left = (row_idx + max_seqlen_k - max_seqlen_q - window_size_left) > 0 ? (row_idx + max_seqlen_k - max_seqlen_q - window_size_left) : 0;  // std::max -> ternary (no __host__ calls in device code)
            const int col_idx_limit_right = (row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right) < max_seqlen_k ? (row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right) : max_seqlen_k;  // std::min -> ternary
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset)
{
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int col_idx_limit = (1 + row_idx_offset + get<0>(idx_rowcol(mi, 0))) < max_seqlen_k ? (1 + row_idx_offset + get<0>(idx_rowcol(mi, 0))) : max_seqlen_k;  // std::min -> ternary
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

template <int kWarpRows>
__forceinline__ __device__ int sm70_mask_row_idx(const int lane_row_base,
                                                 const int row_idx_offset,
                                                 const int i,
                                                 const int mi) {
    static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                  "SM70 mask only supports kWarpRows == 8, 16, 32, or 64");
    return row_idx_offset + lane_row_base + i * 2 + mi * 8;
}

template <int kBlockN>
__forceinline__ __device__ int sm70_mask_col_idx(const int lane_col_base,
                                                 const int col_idx_offset,
                                                 const int j,
                                                 const int nj,
                                                 const int n) {
    static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128 || kBlockN == 256,
                  "SM70 mask only supports kBlockN == 32, 64, 128, or 256");
    return col_idx_offset + lane_col_base + j + nj * 4 + n * 32;
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 8, "First dimension must be 8");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=8, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, 2, MMA_N)).
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            static constexpr int kWarpRows = decltype(size<0>(tensor))::value * 4;
            static constexpr int kBlockN = decltype(size<1>(tensor))::value * 8;
            static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                          "Unexpected SM70 row layout");
            static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128 || kBlockN == 256,
                          "Unexpected SM70 col layout");
            static_assert(decltype(size<0, 0>(tensor))::value == 2, "Unexpected SM70 row-inner layout");
            static_assert(decltype(size<0, 1>(tensor))::value == kWarpRows / 8, "Unexpected SM70 row-outer layout");
            static_assert(decltype(size<1, 0>(tensor))::value == 2, "Unexpected SM70 col-inner layout");
            static_assert(decltype(size<1, 1>(tensor))::value == 2, "Unexpected SM70 col-middle layout");
            static_assert(decltype(size<1, 2>(tensor))::value == kBlockN / 32, "Unexpected SM70 col-outer layout");

            const int lane_id = threadIdx.x % 32;
            const int lane_row_base = (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
            const int lane_col_base =
                (((lane_id >> 1) & 0x1) << 1) |
                (((lane_id >> 2) & 0x1) << 3) |
                (((lane_id >> 3) & 0x1) << 4);
            const int seqlen_delta = max_seqlen_k - max_seqlen_q;
            (void)warp_row_stride;

            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;

            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int n = 0; n < size<1, 2>(tensor); ++n) {
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = sm70_mask_col_idx<kBlockN>(lane_col_base, col_idx_offset_, j, nj, n);
                            #pragma unroll
                            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                                #pragma unroll
                                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj, n));
                                    if constexpr (Has_alibi) {
                                        tensor(coord) += alibi_slope * col_idx;
                                    }
                                    if constexpr (!Is_even_MN) {
                                        if (col_idx >= max_seqlen_k) {
                                            tensor(coord) = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = sm70_mask_row_idx<kWarpRows>(lane_row_base, row_idx_offset, i, mi);
                        const int col_idx_limit_left = (row_idx + seqlen_delta - window_size_left) > 0 ? (row_idx + seqlen_delta - window_size_left) : 0;  // std::max -> ternary
                        const int col_idx_limit_right = (row_idx + 1 + seqlen_delta + window_size_right) < max_seqlen_k ? (row_idx + 1 + seqlen_delta + window_size_right) : max_seqlen_k;  // std::min -> ternary

                        #pragma unroll
                        for (int n = 0; n < size<1, 2>(tensor); ++n) {
                            #pragma unroll
                            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                                #pragma unroll
                                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                    const int col_idx = sm70_mask_col_idx<kBlockN>(lane_col_base, col_idx_offset_, j, nj, n);
                                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj, n));

                                    if constexpr (Has_alibi) {
                                        if constexpr (Is_causal) {
                                            tensor(coord) += alibi_slope * col_idx;
                                        } else {
                                            tensor(coord) -= alibi_slope * abs(row_idx + seqlen_delta - col_idx);
                                        }
                                    }
                                    if constexpr (Causal_mask) {
                                        if (col_idx >= col_idx_limit_right) {
                                            tensor(coord) = -INFINITY;
                                        }
                                    }
                                    if constexpr (Is_local) {
                                        if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                            tensor(coord) = -INFINITY;
                                        }
                                    }
                                    if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                        // Causal and Local already handles MN masking
                                        if (col_idx >= max_seqlen_k) {
                                            tensor(coord) = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

} // namespace FLASH_NAMESPACE
