#pragma once

#include <cstddef>
#include <cstdint>

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
    int    registers_per_cu;
    size_t static_shared;
    size_t dynamic_shared;
    size_t max_shared_per_block;
};

// ROCm through 7.1 reports CU-local LDS/register limits while computing occupancy for a
// WGP-mode kernel. On RDNA2 that makes the D256, 32-column tile appear not to fit in one CU,
// even though its workgroup is distributed across the WGP's two CUs and is launchable. Keep
// the correction restricted to the exact configuration validated on gfx1030; newer runtimes
// that report a positive occupancy never enter this path.
static inline int ggml_cuda_fattn_correct_rdna2_wgp_occupancy(
        const ggml_cuda_fattn_rdna2_occupancy_inputs & in) {
    if (!in.hip || !in.rdna2 || in.occupancy != 0 ||
            in.dkq != 256 || in.dv != 256 || in.ncols1 != 16 || in.ncols2 != 2 ||
            in.threads != 256 || in.max_threads_per_block < in.threads ||
            in.registers_per_thread <= 0 || in.registers_per_cu <= 0 ||
            in.max_shared_per_block == 0 || in.static_shared > in.max_shared_per_block ||
            in.dynamic_shared > in.max_shared_per_block - in.static_shared) {
        return in.occupancy;
    }

    constexpr uint64_t cus_per_wgp = 2;
    const uint64_t registers_required =
        uint64_t(in.registers_per_thread) * uint64_t(in.threads);
    const uint64_t registers_per_wgp =
        uint64_t(in.registers_per_cu) * cus_per_wgp;
    if (registers_required > registers_per_wgp) {
        return in.occupancy;
    }
    return 1;
}
