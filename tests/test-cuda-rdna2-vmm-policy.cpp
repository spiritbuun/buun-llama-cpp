#include "../ggml/src/ggml-cuda/fattn-rdna2-policy.h"
#include "../ggml/src/ggml-cuda/vbr-vmm-policy.h"

#include <cstdio>

static bool expect_occupancy(
        const char * label, ggml_cuda_fattn_rdna2_occupancy_inputs in, int expected) {
    const int actual = ggml_cuda_fattn_correct_rdna2_wgp_occupancy(in);
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s: got occupancy %d, expected %d\n", label, actual, expected);
    return false;
}

static bool expect_granularity(const char * label, size_t actual, size_t expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s: got granularity %zu, expected %zu\n", label, actual, expected);
    return false;
}

int main() {
    const ggml_cuda_fattn_rdna2_occupancy_inputs eligible = {
        true, true, 0, 256, 256, 16, 2, 256, 256, 209, 65536, 37888, 0, 65536,
    };
    bool ok = expect_occupancy("rdna2-wgp", eligible, 1);

    auto excluded = eligible;
    excluded.hip = false;
    ok &= expect_occupancy("non-hip", excluded, 0);
    excluded = eligible;
    excluded.rdna2 = false;
    ok &= expect_occupancy("non-rdna2", excluded, 0);
    excluded = eligible;
    excluded.occupancy = 2;
    ok &= expect_occupancy("runtime-authority", excluded, 2);
    excluded = eligible;
    excluded.dkq = 128;
    ok &= expect_occupancy("other-head", excluded, 0);
    excluded = eligible;
    excluded.ncols1 = 8;
    ok &= expect_occupancy("other-tile", excluded, 0);
    excluded = eligible;
    excluded.static_shared = 65537;
    ok &= expect_occupancy("shared-overflow", excluded, 0);
    excluded = eligible;
    excluded.dynamic_shared = 32 * 1024;
    ok &= expect_occupancy("combined-shared-overflow", excluded, 0);
    excluded = eligible;
    excluded.max_threads_per_block = 128;
    ok &= expect_occupancy("thread-overflow", excluded, 0);
    excluded = eligible;
    excluded.registers_per_thread = 257;
    ok &= expect_occupancy("register-overflow", excluded, 0);

    ok &= expect_granularity("cuda-driver-authority",
        ggml_cuda_vbr_vmm_commit_granularity(4096, false), 4096);
    ok &= expect_granularity("hip-4k-batch",
        ggml_cuda_vbr_vmm_commit_granularity(4096, true), 64 * 1024);
    ok &= expect_granularity("hip-16k-batch",
        ggml_cuda_vbr_vmm_commit_granularity(16 * 1024, true), 64 * 1024);
    ok &= expect_granularity("hip-nondivisor-roundup",
        ggml_cuda_vbr_vmm_commit_granularity(48 * 1024, true), 96 * 1024);
    ok &= expect_granularity("hip-larger-driver-authority",
        ggml_cuda_vbr_vmm_commit_granularity(2 * 1024 * 1024, true), 2 * 1024 * 1024);
    ok &= expect_granularity("unavailable",
        ggml_cuda_vbr_vmm_commit_granularity(0, true), 0);

    if (ok) {
        std::puts("PASS: RDNA2 FlashAttention occupancy and HIP VMM commit policies");
    }
    return ok ? 0 : 1;
}
