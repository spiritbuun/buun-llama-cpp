#pragma once

#include <humming/utils/all.cuh>

template <class Ctx, class MMA>
class EpilogueSmemReducer {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementC = typename Ctx::ElementC;

  static constexpr bool kHasGroupScale = Ctx::kWeightScaleGroupSize > 0 || Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kForceFloatAccum = std::is_same<typename MmaOpClass::ValTypeC, int32_t>::value && kHasGroupScale && !Ctx::kUseIntWeightScale && !kUseFusedE8m0Scale;
  using MmaTypeC = std::conditional_t<kForceFloatAccum, float, typename MmaOpClass::ValTypeC>;
  using MmaShape = typename MmaOpClass::MmaShape;
  using OutputType32 = std::conditional_t<std::is_same<ElementC, Float16>::value, half2, nv_bfloat162>;
  using ValTypeC32 = std::conditional_t<sizeof(MmaTypeC) == 2, OutputType32, MmaTypeC>;

  using CRegistersArrayType = typename MMA::CRegistersArrayType;

public:
  Ctx &ctx;

  CUDA_INLINE
  EpilogueSmemReducer(Ctx &ctx) : ctx(ctx) {
  }

  CUDA_INLINE
  void reduce(uint32_t *regs_ptr) {
    constexpr bool kUseInt4 = sizeof(CRegistersArrayType) % sizeof(int4) == 0;
    using ReductionValue = std::conditional_t<kUseInt4, int4, uint32_t>;
    static_assert(sizeof(CRegistersArrayType) % sizeof(ReductionValue) == 0);
    static_assert(sizeof(ReductionValue) % sizeof(ValTypeC32) == 0);
    constexpr uint32_t num_values = sizeof(CRegistersArrayType) / sizeof(ReductionValue);
    constexpr uint32_t num_reduce_iters = MAX(CEIL_DIV(WarpShape::M, 16), 1);
    constexpr uint32_t num_values_per_time = CEIL_DIV(num_values, num_reduce_iters);
    constexpr uint32_t num_scalars_per_value = sizeof(ReductionValue) / sizeof(ValTypeC32);
    constexpr uint32_t group_num_warps = BlockShape::K / WarpShape::K;
    constexpr uint32_t num_groups = Ctx::kNumMathThreads / 32 / group_num_warps;
    uint32_t group_id = ctx.warp_id() % num_groups;
    uint32_t group_warp_id = ctx.warp_id() / num_groups;
    uint32_t laneid = ctx.lane_id();

    using ReductionSmemType = ReductionValue[group_num_warps / 2][num_groups][num_values_per_time][32];
    auto &smem_arr = *reinterpret_cast<ReductionSmemType *>(ctx.smem.reduce);

    auto write_to_smem = [&](uint32_t buffer_id, uint32_t m) {
      ReductionValue *regs_value_ptr =
          reinterpret_cast<ReductionValue *>(regs_ptr) + m * num_values_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_values_per_time; i++) {
        if (m * num_values_per_time + i < num_values) {
          smem_arr[buffer_id][group_id][i][laneid] = regs_value_ptr[i];
        }
      };
    };

    auto read_from_smem_and_reduce = [&](uint32_t buffer_id, uint32_t m) {
      ReductionValue *regs_value_ptr =
          reinterpret_cast<ReductionValue *>(regs_ptr) + m * num_values_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_values_per_time; i++) {
        if (m * num_values_per_time + i >= num_values) continue;
        ReductionValue val = smem_arr[buffer_id][group_id][i][laneid];

        ValTypeC32 *sval_ptr = reinterpret_cast<ValTypeC32 *>(&val);
        ValTypeC32 *reg_ptr = reinterpret_cast<ValTypeC32 *>(regs_value_ptr + i);
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < num_scalars_per_value; j++) {
          if constexpr (sizeof(MmaTypeC) == 2) {
            reg_ptr[j] = __hadd2(reg_ptr[j], sval_ptr[j]);
          } else {
            reg_ptr[j] += sval_ptr[j];
          }
        }
      };
    };

    PRAGMA_UNROLL
    for (uint32_t m = 0; m < num_reduce_iters; m++) {
      PRAGMA_UNROLL
      for (uint32_t i = 1; i < group_num_warps; i *= 2) {
        uint32_t buffer_id = group_warp_id % (group_num_warps / (2 * i));
        if (group_warp_id >= group_num_warps / i) {
          ctx.sync_math_threads();
        } else if (group_warp_id >= group_num_warps / (2 * i)) {
          write_to_smem(buffer_id, m);
          ctx.sync_math_threads();
        } else {
          ctx.sync_math_threads();
          read_from_smem_and_reduce(buffer_id, m);
        };

        ctx.sync_math_threads();
      };
    };
  };
};
