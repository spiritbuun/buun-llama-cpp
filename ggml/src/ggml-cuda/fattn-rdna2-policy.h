#pragma once

#include <cstddef>

struct ggml_cuda_fattn_rdna2_occupancy_inputs {
    bool   hip;
    bool   rdna2;
    int    occupancy;
    int    dkq;
    int    dv;
    int    ncols1;
    int    ncols2;
    int    threads;
    int    max_threads_per_block;
    int    registers_per_thread;
    int    max_registers_per_block;
    size_t static_shared;
    size_t dynamic_shared;
    size_t max_shared_per_block;
};

// ROCm through 7.1 accounts LDS per CU when computing occupancy for a WGP-mode kernel. On
// RDNA2 that makes the D256, 32-column tile appear not to fit in a 32 KiB CU even though its
// 37 KiB allocation fits in the 64 KiB WGP and the kernel is launchable. Keep the correction
// restricted to the exact configuration validated on gfx1030; newer runtimes that report a
// positive occupancy never enter this path.
static inline int ggml_cuda_fattn_correct_rdna2_wgp_occupancy(
        const ggml_cuda_fattn_rdna2_occupancy_inputs & in) {
    if (!in.hip || !in.rdna2 || in.occupancy != 0 ||
            in.dkq != 256 || in.dv != 256 || in.ncols1 != 16 || in.ncols2 != 2 ||
            in.threads != 256 || in.max_threads_per_block < in.threads ||
            in.registers_per_thread <= 0 || in.max_registers_per_block <= 0 ||
            in.registers_per_thread > in.max_registers_per_block / in.threads ||
            in.max_shared_per_block == 0 || in.static_shared > in.max_shared_per_block ||
            in.dynamic_shared > in.max_shared_per_block - in.static_shared) {
        return in.occupancy;
    }
    return 1;
}
