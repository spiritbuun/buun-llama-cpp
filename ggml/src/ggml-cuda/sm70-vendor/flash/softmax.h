/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "philox.cuh"
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max) {
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum) {
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
#ifdef UNFUSE_FMA
            tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
#else
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
#endif
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ int sm70_lane_row_base(const int lane_id) {
    return (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
}

template<int kWarpRows>
__forceinline__ __device__ int sm70_row_slot(const int slot, const int lane_id) {
    static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                  "SM70 softmax only supports kWarpRows == 8, 16, 32, or 64");
    const int lane_row_base = sm70_lane_row_base(lane_id);
    return lane_row_base + ((slot & 0x1) << 1) + ((slot >> 1) << 3);
}

template<typename Operator>
__forceinline__ __device__ float sm70_row_allreduce_8(float x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 2));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 4));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 8));
    return x;
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    MaxOp<float> max_op;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float row_max = zero_init ? float(tensor(mi, 0)) : max_op(max(mi), float(tensor(mi, 0)));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ++ni) {
            row_max = max_op(row_max, float(tensor(mi, ni)));
        }
        max(mi) = sm70_row_allreduce_8(row_max, max_op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(sum) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float row_sum = zero_init ? float(tensor(mi, 0)) : float(sum(mi)) + float(tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ++ni) {
            row_sum += float(tensor(mi, ni));
        }
        sum(mi) = row_sum;
    }
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    scale_apply_exp2<Scale_max>(tensor, max, scale);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_rescale_acc_o(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &row_scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(row_scale) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) *= row_scale(mi);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kWarpRows>
struct Softmax {
    static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                  "SM70 softmax only supports kWarpRows == 8, 16, 32, or 64");
    static constexpr int kRowsPerThread = kWarpRows / 4;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kRowsPerThread>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kRowsPerThread);
        if (Is_first) {
            FLASH_NAMESPACE::template sm70_reduce_max</*zero_init=*/true>(scores, row_max);
            Tensor scores_max = make_fragment_like(row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                scores_max(mi) = Check_inf && row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
            }
            FLASH_NAMESPACE::sm70_scale_apply_exp2(scores, scores_max, softmax_scale_log2);
            FLASH_NAMESPACE::sm70_reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template sm70_reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kRowsPerThread);
            Tensor scores_scale_row = make_fragment_like(row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                scores_scale_row(mi) = scores_scale;
                row_sum(mi) *= scores_scale;
            }
            FLASH_NAMESPACE::sm70_rescale_acc_o(acc_o_rowcol, scores_scale_row);
            Tensor scores_max = make_fragment_like(row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                scores_max(mi) = Check_inf && row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
            }
            FLASH_NAMESPACE::sm70_scale_apply_exp2(scores, scores_max, softmax_scale_log2);
            FLASH_NAMESPACE::sm70_reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0f) {
        SumOp<float> sum_op;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            row_sum(mi) = sm70_row_allreduce_8(row_sum(mi), sum_op);
        }
        TensorT lse = make_fragment_like(row_sum);
        TensorT row_scale = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kRowsPerThread);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            row_scale(mi) = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        }
        FLASH_NAMESPACE::sm70_rescale_acc_o(acc_o_rowcol, row_scale);
        return lse;
    };
};

}  // namespace FLASH_NAMESPACE
