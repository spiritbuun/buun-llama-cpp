#pragma once

#include <humming/utils/base.cuh>


CUDA_INLINE uint32_t warp_reduce_add(uint32_t local_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 16);
  local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 8);
  local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 4);
  local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 2);
  local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 1);
  return local_count;
#else
  return __reduce_add_sync(0xffffffff, local_count);
#endif
}
