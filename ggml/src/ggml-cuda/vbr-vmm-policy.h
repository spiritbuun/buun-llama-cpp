#pragma once

#include <cstddef>

// ROCm may report its 4 KiB minimum as the recommended VMM allocation granularity. Committing
// each such page separately makes prompt growth issue tens of thousands of synchronous
// hipMemCreate/hipMemMap/hipMemSetAccess calls. A 64 KiB commit remains small relative to a KV
// tensor while amortizing that host/runtime cost. CUDA retains the driver's recommendation.
static inline size_t ggml_cuda_vbr_vmm_commit_granularity(size_t driver_granularity, bool hip) {
    constexpr size_t hip_min_commit = 64 * 1024;
    if (!hip || driver_granularity == 0 || driver_granularity >= hip_min_commit) {
        return driver_granularity;
    }
    return ((hip_min_commit + driver_granularity - 1) / driver_granularity) * driver_granularity;
}
