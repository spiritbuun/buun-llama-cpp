#pragma once

#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>
#include <humming/memory/g2s_loader/loader_a.cuh>
#include <humming/memory/g2s_loader/loader_as.cuh>
#include <humming/memory/g2s_loader/loader_b.cuh>
#include <humming/memory/g2s_loader/loader_bias.cuh>
#include <humming/memory/g2s_loader/loader_bs.cuh>
#include <humming/memory/g2s_loader/loader_bs2.cuh>
#include <humming/memory/g2s_loader/loader_bzp.cuh>
#include <humming/utils/all.cuh>


template <class Ctx>
class ProducerPipeline {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using ElementA = typename Ctx::ElementA;

  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;

  static constexpr bool kUseMBarrier = Ctx::kUseMBarrier;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseTma = Ctx::kUseTma;
  static constexpr bool kUseTmaA = Ctx::kUseTmaA;
  static constexpr bool kUseTmaAS = Ctx::kUseTmaAS && !Ctx::kIsIndexedGemm;
  static constexpr bool kUseTmaB = Ctx::kUseTmaB;
  static constexpr bool kUseTmaBS = Ctx::kUseTmaBS;
  static constexpr bool kUseTmaBS2 = Ctx::kUseTmaBS2;
  static constexpr bool kUseTmaBZP = Ctx::kUseTmaBZP;
  static constexpr bool kUseTmaBias = Ctx::kUseTmaBias;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = Ctx::kIsChannelWeightScale2;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kHasBias = Ctx::kHasBias;
  static constexpr bool kHasChannelData = kIsChannelInputScale || kIsChannelWeightScale || kIsChannelWeightScale2 || kHasBias;

  static constexpr uint32_t kNumStages = Ctx::kNumStages;
  static constexpr bool kUseTwoStageReduceBarrier = SharedStorage::kUseTwoStageReduceBarrier;

  template <bool kIsFirst = false>
  static constexpr uint2 get_stage_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kUseTmaA) tma_load_bytes += SharedStorage::kStageBytesA;
    else legacy_load_bytes += SharedStorage::kStageBytesA;

    if constexpr (kUseTmaB) tma_load_bytes += SharedStorage::kStageBytesB;
    else legacy_load_bytes += SharedStorage::kStageBytesB;

    if constexpr (kIsGroupInputScale) {
      if constexpr (kUseTmaAS) tma_load_bytes += SharedStorage::kStageBytesAS;
      else legacy_load_bytes += SharedStorage::kStageBytesAS;
    }

    if constexpr (kIsGroupWeightScale || kIsBlockWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += SharedStorage::kStageBytesBS;
      else legacy_load_bytes += SharedStorage::kStageBytesBS;
    }

    if constexpr (kHasZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
      constexpr uint32_t zero_point_bytes = kIsChannelWeightScale ? SharedStorage::kChannelBytesBZP : SharedStorage::kStageBytesBZP;
      if constexpr (kUseTmaBZP) tma_load_bytes += zero_point_bytes;
      else legacy_load_bytes += zero_point_bytes;
    }

    return {tma_load_bytes, legacy_load_bytes};
  }

  static constexpr uint2 get_channel_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kIsChannelInputScale) {
      if constexpr (kUseTmaAS) tma_load_bytes += SharedStorage::kChannelBytesAS;
      else legacy_load_bytes += SharedStorage::kChannelBytesAS;
    }

    if constexpr (kIsChannelWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += SharedStorage::kChannelBytesBS;
      else legacy_load_bytes += SharedStorage::kChannelBytesBS;
    }

    if constexpr (kIsChannelWeightScale2) {
      if constexpr (kUseTmaBS2) tma_load_bytes += SharedStorage::kChannelBytesBS2;
      else legacy_load_bytes += SharedStorage::kChannelBytesBS2;
    }

    if constexpr (kHasBias) {
      if constexpr (kUseTmaBias) tma_load_bytes += SharedStorage::kBiasBytes;
      else legacy_load_bytes += SharedStorage::kBiasBytes;
    }

    return {tma_load_bytes, legacy_load_bytes};
  }

public:
  static constexpr bool kHasFirstStageTmaMBarrier = get_stage_load_bytes<true>().x > 0;
  static constexpr bool kHasFirstStageCpAsyncMBarrier = get_stage_load_bytes<true>().y > 0;
  static constexpr bool kHasStageTmaMBarrier = get_stage_load_bytes().x > 0;
  static constexpr bool kHasStageCpAsyncMBarrier = get_stage_load_bytes().y > 0;
  static constexpr bool kHasChannelTmaMBarrier = get_channel_load_bytes().x > 0;
  static constexpr bool kHasChannelCpAsyncMBarrier = get_channel_load_bytes().y > 0;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  using LoaderA = G2SMemoryLoaderA<Ctx>;
  using LoaderB = G2SMemoryLoaderB<Ctx>;
  using LoaderAS = G2SMemoryLoaderAS<Ctx>;
  using LoaderBS = G2SMemoryLoaderBS<Ctx>;
  using LoaderBS2 = G2SMemoryLoaderBS2<Ctx>;
  using LoaderBZP = G2SMemoryLoaderBZP<Ctx>;
  using LoaderBias = G2SMemoryLoaderBias<Ctx>;

  Ctx &ctx;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBS2 loader_bs2;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;
  uint32_t phases[SharedStorage::kNumMathMbarriers] = {0};

  CUDA_INLINE
  ProducerPipeline(Ctx &ctx)
      : ctx(ctx),
        loader_a(ctx),
        loader_b(ctx),
        loader_as(ctx),
        loader_bs(ctx),
        loader_bs2(ctx),
        loader_bzp(ctx),
        loader_bias(ctx) {

    if (ctx.load_thread_id() == 0) {
      if constexpr (kUseTmaA) prefetch_tensor_map(ctx.params.a);
      if constexpr (kUseTmaAS) prefetch_tensor_map(ctx.params.as);
      if constexpr (kUseTmaB) prefetch_tensor_map(ctx.params.b);
      if constexpr (kUseTmaBS) prefetch_tensor_map(ctx.params.bs);
      if constexpr (kUseTmaBS2) prefetch_tensor_map(ctx.params.bs2);
      if constexpr (kUseTmaBZP) prefetch_tensor_map(ctx.params.bzp);
      if constexpr (kUseTmaBias) prefetch_tensor_map(ctx.params.bias);
    }
    __syncwarp();
  }

  CUDA_INLINE static void init_mbarrier(Ctx &ctx) {
    if constexpr (kUseMBarrier) {
      uint32_t thread_id = ctx.load_thread_id();
      uint32_t cluster_rank = ctx.cluster_rank();
      auto &smem = ctx.smem;
      uint32_t count;
      if (thread_id < kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasFirstStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasFirstStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages + 1) {
        constexpr uint32_t cp_async_thread_count = kHasChannelCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasChannelTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      }

      if (thread_id < kNumStages + 2) __mbarrier_init(&smem.load_mbar[thread_id], count);
      uint32_t factor = (kMultiCastSize > 1 && cluster_rank == 0 && thread_id < kNumStages) ? kMultiCastSize : 1;
      if constexpr (Ctx::kUseWarpSpec) {
        if (thread_id < SharedStorage::kNumMathMbarriers) {
          __mbarrier_init(&smem.math_mbar[thread_id], kNumMathThreads * factor);
        }
      }
    }
  }

  CUDA_INLINE void init_mbarrier() {
    init_mbarrier(ctx);
  }

  template <bool kShouldAdvance = true, bool kIsFirst = false>
  CUDA_INLINE void load_stage(uint32_t stage_id, bool pred = true) {
    stage_id = stage_id % kNumStages;
    auto &smem = ctx.smem;

    uint32_t mbar_index = kIsFirst ? kNumStages : stage_id;
    constexpr uint2 load_bytes = get_stage_load_bytes<kIsFirst>();
    constexpr bool kHasTmaMBarrier = kIsFirst ? kHasFirstStageTmaMBarrier : kHasStageTmaMBarrier;

    uint64_t *mbar_ptr = nullptr;
    if constexpr (kUseMBarrier) mbar_ptr = &smem.load_mbar[mbar_index];
    if (pred) {
      loader_a.template load<kShouldAdvance>(smem.stages[stage_id].a, mbar_ptr, stage_id);
      loader_b.template load<kShouldAdvance>(smem.stages[stage_id].b, mbar_ptr);
      if constexpr (kIsGroupInputScale) {
        loader_as.template load<kShouldAdvance>(smem.stages[stage_id].as, mbar_ptr);
      };
      if constexpr (kIsGroupWeightScale || kIsBlockWeightScale) {
        loader_bs.template load<kShouldAdvance>(smem.stages[stage_id].bs, mbar_ptr);
      };
      if constexpr (kHasZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
        if constexpr (kIsChannelWeightScale)
          loader_bzp.template load<kShouldAdvance>(smem.bzp_c, mbar_ptr);
        else
          loader_bzp.template load<kShouldAdvance>(smem.stages[stage_id].bzp, mbar_ptr);
      }
    }

    if constexpr (kIsFirst) {
      commit_cp_async_load<kHasFirstStageCpAsyncMBarrier>(mbar_index, pred);
    } else {
      commit_cp_async_load<kHasStageCpAsyncMBarrier>(mbar_index, pred);
    }
    if (pred) expect_tma_load<kHasTmaMBarrier>(mbar_ptr, load_bytes.x);
  }

  CUDA_INLINE void load_channel() {
    auto &smem = ctx.smem;
    uint64_t *channel_mbar_ptr = nullptr;
    constexpr uint2 load_bytes = get_channel_load_bytes();
    if constexpr (kUseMBarrier) channel_mbar_ptr = &smem.load_mbar[kNumStages + 1];
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, channel_mbar_ptr);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, channel_mbar_ptr);
    if constexpr (kIsChannelWeightScale2) loader_bs2.load(smem.bs2_c, channel_mbar_ptr);
    if constexpr (kHasBias) loader_bias.load(smem.bias, channel_mbar_ptr);

    if constexpr (load_bytes.x > 0 || load_bytes.y > 0) {
      commit_cp_async_load<kHasChannelCpAsyncMBarrier>(kNumStages + 1);
    }
    expect_tma_load<kHasChannelTmaMBarrier>(channel_mbar_ptr, load_bytes.x);
  }

  template <bool kHasTmaMBarrier>
  CUDA_INLINE void expect_tma_load(uint64_t *mbar_ptr, uint32_t bytes) {
    if constexpr (kUseMBarrier && kHasTmaMBarrier) {
      if (ctx.load_thread_id() == 0) {
        tma_expect_tx(mbar_ptr, bytes);
      }
    }
  }

  template <bool kHasCpAsyncMBarrier>
  CUDA_INLINE void commit_cp_async_load(uint32_t stage_id, bool pred = true) {
    auto &smem = ctx.smem;
    if constexpr (kUseMBarrier) {
      if (!pred) return;
      if constexpr (kHasCpAsyncMBarrier) {
        if constexpr (kUseCpAsync) cp_async_commit_mbarrier(&smem.load_mbar[stage_id]);
        else mbarrier_arrive(&smem.load_mbar[stage_id]);
      }
    } else if constexpr (kUseCpAsync) {
      cp_async_commit_group();
    }
  }

  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    mbarrier_wait(&ctx.smem.math_mbar[stage_id], phases[stage_id], "Humming producer waiting for math stage");
    if constexpr (Ctx::kUseWarpSpec) ctx.sync_load_threads();
    if constexpr (get_stage_load_bytes().x > 0) {
      tma_fence_async_shared();
    }
    phases[stage_id] ^= 1;
  }

  CUDA_INLINE void wait_channel() {
    if constexpr (kHasChannelData && kUseMBarrier) {
      mbarrier_wait(&ctx.smem.math_mbar[kNumStages], phases[kNumStages], "Humming producer waiting for channel consumer");
      if constexpr (Ctx::kUseWarpSpec) ctx.sync_load_threads();
      phases[kNumStages] ^= 1;
    }
  }

  CUDA_INLINE void wait_math_epilogue() {
    mbarrier_wait(&ctx.smem.math_mbar[kNumStages], phases[kNumStages], "Humming producer waiting for epilogue");
    if constexpr (Ctx::kUseWarpSpec) ctx.sync_load_threads();
    if constexpr (get_stage_load_bytes<true>().x > 0 || get_channel_load_bytes().x > 0) {
      tma_fence_async_shared();
    }
    phases[kNumStages] ^= 1;
  }

  CUDA_INLINE void wait_reduce_epilogue() {
    if constexpr (kUseTwoStageReduceBarrier) {
      constexpr uint32_t kBarrierId = kNumStages + 1;
      mbarrier_wait(&ctx.smem.math_mbar[kBarrierId], phases[kBarrierId], "Humming producer waiting to reuse reduce storage");
      if constexpr (Ctx::kUseWarpSpec) ctx.sync_load_threads();
      phases[kBarrierId] ^= 1;
    }
  }

  CUDA_INLINE void seek(
      uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t k_block_id,
      uint32_t current_shape_m, uint32_t m_offset) {
    loader_a.seek(m_block_id, k_block_id, current_shape_m, m_offset);
    loader_b.seek(expert_id, n_block_id, k_block_id);
    loader_as.seek(expert_id, m_block_id, k_block_id, current_shape_m, m_offset);
    loader_bs.seek(expert_id, n_block_id, k_block_id);
    loader_bs2.seek(expert_id, n_block_id);
    loader_bzp.seek(expert_id, n_block_id, k_block_id);
    loader_bias.seek(expert_id, n_block_id);
  }
};


template <class Ctx>
class ConsumerPipeline {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using ElementA = typename Ctx::ElementA;

  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;

  static constexpr bool kUseMBarrier = Ctx::kUseMBarrier;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = Ctx::kIsChannelWeightScale2;
  static constexpr bool kHasBias = Ctx::kHasBias;
  static constexpr bool kHasChannelData = kIsChannelInputScale || kIsChannelWeightScale || kIsChannelWeightScale2 || kHasBias;

  static constexpr uint32_t kNumStages = Ctx::kNumStages;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

public:
  Ctx &ctx;
  uint32_t phases[Ctx::kNumStages + 2] = {0};

  CUDA_INLINE
  ConsumerPipeline(Ctx &ctx) : ctx(ctx) {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    stage_id = kIsFirst ? kNumStages : (stage_id % kNumStages);
    if constexpr (kUseMBarrier) {
      mbarrier_wait(&ctx.smem.load_mbar[stage_id], phases[stage_id], "Humming consumer waiting for load stage");
      phases[stage_id] ^= 1;
    } else if constexpr (kUseCpAsync) {
      cp_async_wait_group<kNumStages - 2>();
      __syncthreads();
    } else {
      __syncthreads();
    }
  }

  CUDA_INLINE void wait_channel() {
    if constexpr (kHasChannelData) {
      if constexpr (kUseMBarrier) {
        mbarrier_wait(&ctx.smem.load_mbar[kNumStages + 1], phases[kNumStages + 1], "Humming consumer waiting for channel data");
        phases[kNumStages + 1] ^= 1;
      } else if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
        __syncthreads();
      } else {
        __syncthreads();
      }
    }
  }

  CUDA_INLINE void arrive(uint32_t stage_id) {
    auto &smem = ctx.smem;
    mbarrier_arrive(&smem.math_mbar[stage_id]);
    if constexpr (kMultiCastSize > 1) {
      if (ctx.cluster_rank() >= 1 && stage_id < kNumStages) {
        void *aa = __cluster_map_shared_rank(&smem.math_mbar[stage_id], 0);
        mbarrier_arrive<true>(aa);
      }
    }
  }
};
