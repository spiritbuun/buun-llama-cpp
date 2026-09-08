#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstring>
#include <vector>

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

static bool test_classic_transcode(const ggml_vbr_backend_iface * be, int device) {
    constexpr int64_t ne0 = 576; // Ling's physical coupled-MLA K width
    constexpr int64_t n   = 257; // one complete 256-row tile plus a tail
    constexpr std::array<ggml_type, 3> types = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
    };

    ggml_backend_t backend = be->backend_init(device);
    if (backend == nullptr) {
        std::fprintf(stderr, "classic transcode: backend init failed\n");
        return false;
    }
    ggml_backend_buffer_type_t buft = be->buffer_type(device);
    bool ok = true;

    std::vector<float> values((size_t) ne0 * n);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint32_t r = (uint32_t) i * 1103515245u + 12345u;
        values[i] = (float) (r & 0xffffu) / 32768.0f - 1.0f;
    }
    std::vector<int32_t> rows(n);
    for (int64_t i = 0; i < n; ++i) {
        rows[i] = (int32_t) i;
    }

    for (ggml_type type_a : types) {
        const size_t bytes_a = (size_t) n * ggml_row_size(type_a, ne0);
        const size_t work_bytes = (size_t) n * ggml_row_size(GGML_TYPE_F16, ne0);
        ggml_backend_buffer_t values_buf = ggml_backend_buft_alloc_buffer(buft, values.size() * sizeof(float));
        ggml_backend_buffer_t rows_buf   = ggml_backend_buft_alloc_buffer(buft, rows.size() * sizeof(int32_t));
        ggml_backend_buffer_t work_buf   = ggml_backend_buft_alloc_buffer(buft, work_bytes);
        if (values_buf == nullptr || rows_buf == nullptr || work_buf == nullptr) {
            std::fprintf(stderr, "classic transcode: source allocation failed\n");
            if (values_buf) ggml_backend_buffer_free(values_buf);
            if (rows_buf)   ggml_backend_buffer_free(rows_buf);
            if (work_buf)   ggml_backend_buffer_free(work_buf);
            ok = false;
            break;
        }

        ggml_init_params ip = { 32 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * values_dev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n);
        ggml_tensor * rows_dev   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_tensor * source     = ggml_new_tensor_2d(ctx, type_a, ne0, n);
        values_dev->data = ggml_backend_buffer_get_base(values_buf);
        values_dev->buffer = values_buf;
        rows_dev->data = ggml_backend_buffer_get_base(rows_buf);
        rows_dev->buffer = rows_buf;
        source->data = ggml_backend_buffer_get_base(work_buf);
        source->buffer = work_buf;
        ggml_set_name(source, "cache_k_l0_ms0");
        ggml_backend_tensor_set(values_dev, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(rows_dev, rows.data(), 0, rows.size() * sizeof(int32_t));

        ggml_tensor * source_encode = ggml_set_rows(ctx, source, values_dev, rows_dev);
        ggml_cgraph * source_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(source_graph, source_encode);
        ggml_backend_graph_compute(backend, source_graph);
        ggml_backend_synchronize(backend);
        std::vector<uint8_t> original(bytes_a);
        ggml_backend_tensor_get(source, original.data(), 0, original.size());

        for (ggml_type type_b : types) {
            if (type_b == type_a) {
                continue;
            }
            // The prior destination arm may have exercised an in-place conversion.
            ggml_backend_tensor_set(source, original.data(), 0, original.size());
            be->sync_device(device);
            const size_t bytes_b = (size_t) n * ggml_row_size(type_b, ne0);
            ggml_backend_buffer_t separate_buf = ggml_backend_buft_alloc_buffer(buft, bytes_b);
            ggml_backend_buffer_t expected_buf = ggml_backend_buft_alloc_buffer(buft, bytes_b);
            ggml_backend_buffer_t decoded_buf  = ggml_backend_buft_alloc_buffer(buft, values.size() * sizeof(float));
            if (separate_buf == nullptr || expected_buf == nullptr || decoded_buf == nullptr) {
                std::fprintf(stderr, "classic transcode: destination allocation failed\n");
                if (separate_buf) ggml_backend_buffer_free(separate_buf);
                if (expected_buf) ggml_backend_buffer_free(expected_buf);
                if (decoded_buf)  ggml_backend_buffer_free(decoded_buf);
                ok = false;
                break;
            }

            const ggml_vbr_transcode_params separate = {
                source, type_b, ggml_backend_buffer_get_base(separate_buf), separate_buf,
                n, false, nullptr, 0, 0,
            };
            be->kv_transcode(backend, &separate);
            ggml_backend_synchronize(backend);

            // Independent graph oracle: stock quantized copy decodes A to F32, then set_rows
            // encodes B. This is also the static-cache conversion path classic VBR must match.
            ggml_tensor * decoded  = ggml_cast(ctx, source, GGML_TYPE_F32);
            ggml_tensor * expected = ggml_new_tensor_2d(ctx, type_b, ne0, n);
            decoded->data = ggml_backend_buffer_get_base(decoded_buf);
            decoded->buffer = decoded_buf;
            expected->data = ggml_backend_buffer_get_base(expected_buf);
            expected->buffer = expected_buf;
            ggml_tensor * expected_encode = ggml_set_rows(ctx, expected, decoded, rows_dev);
            ggml_cgraph * expected_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(expected_graph, expected_encode);
            ggml_backend_graph_compute(backend, expected_graph);
            ggml_backend_synchronize(backend);

            std::vector<uint8_t> separate_host(bytes_b);
            std::vector<uint8_t> expected_host(bytes_b);
            ggml_backend_tensor_get(expected, expected_host.data(), 0, expected_host.size());
            ggml_tensor separate_tensor = *expected;
            separate_tensor.data = ggml_backend_buffer_get_base(separate_buf);
            separate_tensor.buffer = separate_buf;
            ggml_backend_tensor_get(&separate_tensor, separate_host.data(), 0, separate_host.size());
            if (separate_host != expected_host) {
                size_t first = 0;
                while (first < bytes_b && separate_host[first] == expected_host[first]) {
                    ++first;
                }
                size_t mismatches = 0;
                for (size_t i = first; i < bytes_b; ++i) {
                    mismatches += separate_host[i] != expected_host[i];
                }
                float max_abs = 0.0f;
                bool numerically_equal = type_b == GGML_TYPE_F16;
                if (type_b == GGML_TYPE_F16) {
                    for (size_t i = 0; i < bytes_b; i += sizeof(ggml_fp16_t)) {
                        ggml_fp16_t a;
                        ggml_fp16_t b;
                        std::memcpy(&a, separate_host.data() + i, sizeof(a));
                        std::memcpy(&b, expected_host.data() + i, sizeof(b));
                        const float av = ggml_fp16_to_fp32(a);
                        const float bv = ggml_fp16_to_fp32(b);
                        numerically_equal = numerically_equal && std::isfinite(av) && std::isfinite(bv) && av == bv;
                        max_abs = std::max(max_abs, std::abs(av - bv));
                    }
                }
                if (!numerically_equal) {
                    std::fprintf(stderr,
                            "classic transcode: %s -> %s differs from graph oracle "
                            "(%zu bytes, first offset %zu, max abs %.9g)\n",
                            ggml_type_name(type_a), ggml_type_name(type_b), mismatches, first, max_abs);
                    ok = false;
                }
            }

            // Restore A, then make the same conversion in place. Promotions walk tiles in
            // reverse; the 257-row shape makes overlap bugs observable at the tile boundary.
            ggml_backend_tensor_set(source, original.data(), 0, original.size());
            be->sync_device(device); // buffer upload and VBR side backend use different streams
            const ggml_vbr_transcode_params in_place = {
                source, type_b, source->data, work_buf, n, false, nullptr, 0, 0,
            };
            be->kv_transcode(backend, &in_place);
            ggml_backend_synchronize(backend);
            std::vector<uint8_t> in_place_host(bytes_b);
            ggml_tensor in_place_tensor = *expected;
            in_place_tensor.data = source->data;
            in_place_tensor.buffer = work_buf;
            ggml_backend_tensor_get(&in_place_tensor, in_place_host.data(), 0, in_place_host.size());
            if (in_place_host != separate_host) {
                size_t first = 0;
                while (first < bytes_b && in_place_host[first] == separate_host[first]) {
                    ++first;
                }
                size_t mismatches = 0;
                for (size_t i = first; i < bytes_b; ++i) {
                    mismatches += in_place_host[i] != separate_host[i];
                }
                std::fprintf(stderr,
                        "classic transcode: %s -> %s in-place differs from separate "
                        "(%zu bytes, first offset %zu)\n",
                        ggml_type_name(type_a), ggml_type_name(type_b), mismatches, first);
                ok = false;
            }

            ggml_backend_buffer_free(decoded_buf);
            ggml_backend_buffer_free(expected_buf);
            ggml_backend_buffer_free(separate_buf);
        }

        ggml_free(ctx);
        ggml_backend_buffer_free(work_buf);
        ggml_backend_buffer_free(rows_buf);
        ggml_backend_buffer_free(values_buf);
        if (!ok) {
            break;
        }
    }

    ggml_backend_free(backend);
    return ok;
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
    ok = test_classic_transcode(be, device) && ok;
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
    std::printf("PASS: device %d VMM range accounting and classic transcode (%zu KiB pages)\n",
            device, g / 1024);
    return 0;
}
