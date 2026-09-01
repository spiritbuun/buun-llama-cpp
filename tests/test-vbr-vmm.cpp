#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

static bool expect_eq(const char * label, size_t actual, size_t expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s: got %zu, expected %zu\n", label, actual, expected);
    return false;
}

static bool expect_epoch(const char * label, uint64_t actual, uint64_t expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s: got %" PRIu64 ", expected %" PRIu64 "\n", label, actual, expected);
    return false;
}

int main(int argc, char ** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const ggml_vbr_backend_iface * be = ggml_backend_cuda_vbr_iface();
    if (be == nullptr || device < 0 || device >= be->get_device_count() || !be->vmm_available(device)) {
        std::printf("SKIP: GPU device %d has no VMM support\n", device);
        return 0;
    }

    const size_t g = be->vmm_granularity(device);
    if (g == 0 || g % 2 != 0 || g > SIZE_MAX / 4) {
        std::fprintf(stderr, "invalid VMM granularity: %zu\n", g);
        return 1;
    }

    ggml_vbr_vmm_pool * pool = be->vmm_pool_init(device, 4 * g);
    if (pool == nullptr) {
        std::fprintf(stderr, "failed to reserve a four-page VMM test pool\n");
        return 1;
    }

    bool ok = true;
    auto range = [&](size_t first_page, size_t n_pages) {
        return be->vmm_pool_mapped_in_range(pool, first_page * g, n_pages * g);
    };

    ok = expect_eq("initial total", be->vmm_pool_mapped(pool), 0) && ok;
    ok = expect_eq("empty range", range(0, 0), 0) && ok;
    ok = expect_epoch("initial epoch", be->vmm_pool_residency_epoch(pool), 0) && ok;

    // This unaligned logical request intersects pages 1 and 2. Range accounting itself stays
    // page-aligned, matching the physical unit tracked by the pool.
    if (!be->vmm_pool_map(pool, g + g / 2, g)) {
        std::fprintf(stderr, "failed to map the two-page test span\n");
        be->vmm_pool_free(pool);
        return 1;
    }
    ok = expect_eq("mapped total", be->vmm_pool_mapped(pool), 2 * g) && ok;
    ok = expect_eq("page 0", range(0, 1), 0) && ok;
    ok = expect_eq("pages 1-2", range(1, 2), 2 * g) && ok;
    ok = expect_eq("page 1", range(1, 1), g) && ok;
    ok = expect_eq("page 2", range(2, 1), g) && ok;
    ok = expect_eq("page 3", range(3, 1), 0) && ok;
    ok = expect_epoch("mapped epoch", be->vmm_pool_residency_epoch(pool), 1) && ok;

    // An overlapping map is idempotent.
    if (!be->vmm_pool_map(pool, 2 * g, g)) {
        std::fprintf(stderr, "failed to remap an already resident page\n");
        be->vmm_pool_free(pool);
        return 1;
    }
    ok = expect_eq("idempotent total", be->vmm_pool_mapped(pool), 2 * g) && ok;
    ok = expect_epoch("idempotent epoch", be->vmm_pool_residency_epoch(pool), 1) && ok;

    // No page is fully contained in [1.5g, 2.5g), so both remain resident.
    be->vmm_pool_unmap(pool, g + g / 2, g);
    ok = expect_eq("partial unmap total", be->vmm_pool_mapped(pool), 2 * g) && ok;
    ok = expect_epoch("partial unmap epoch", be->vmm_pool_residency_epoch(pool), 1) && ok;

    // [1.5g, 3.5g) fully contains page 2 only.
    be->vmm_pool_unmap(pool, g + g / 2, 2 * g);
    ok = expect_eq("one-page unmap total", be->vmm_pool_mapped(pool), g) && ok;
    ok = expect_eq("surviving page 1", range(1, 1), g) && ok;
    ok = expect_eq("released page 2", range(2, 1), 0) && ok;
    ok = expect_epoch("one-page unmap epoch", be->vmm_pool_residency_epoch(pool), 2) && ok;

    // Move the one-page residency from page 1 to page 3. The pool total is identical before and
    // after, but the epoch must invalidate any cache whose result depends on per-range residency.
    const uint64_t redistribution_epoch = be->vmm_pool_residency_epoch(pool);
    if (!be->vmm_pool_map(pool, 3 * g, g)) {
        std::fprintf(stderr, "failed to map redistribution destination\n");
        be->vmm_pool_free(pool);
        return 1;
    }
    be->vmm_pool_unmap(pool, g, g);
    ok = expect_eq("redistributed total", be->vmm_pool_mapped(pool), g) && ok;
    ok = expect_eq("redistributed page 1", range(1, 1), 0) && ok;
    ok = expect_eq("redistributed page 3", range(3, 1), g) && ok;
    ok = expect_epoch("redistributed epoch", be->vmm_pool_residency_epoch(pool),
                      redistribution_epoch + 2) && ok;

    const uint64_t clear_epoch = be->vmm_pool_residency_epoch(pool);
    be->vmm_pool_clear(pool);
    ok = expect_eq("clear preserves residency", be->vmm_pool_mapped(pool), g) && ok;
    ok = expect_epoch("clear preserves epoch", be->vmm_pool_residency_epoch(pool), clear_epoch) && ok;

    be->vmm_pool_unmap(pool, 0, 4 * g);
    ok = expect_eq("final total", be->vmm_pool_mapped(pool), 0) && ok;
    ok = expect_eq("final range", range(0, 4), 0) && ok;
    ok = expect_epoch("final epoch", be->vmm_pool_residency_epoch(pool), clear_epoch + 1) && ok;
    be->vmm_pool_free(pool);

    if (!ok) {
        return 1;
    }
    std::printf("PASS: device %d VMM range accounting (%zu KiB pages)\n", device, g / 1024);
    return 0;
}
