#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace {

constexpr int64_t n_in      = 256;
constexpr int64_t n_out     = 128;
constexpr int64_t n_expert  = 64;
constexpr int64_t n_used    = 2;
constexpr int64_t n_tokens  = 1;
constexpr int64_t multi_n_used   = 6;
constexpr int64_t multi_n_tokens = 10;
constexpr int     max_steps = 160;
static std::string test_executable;

struct log_capture {
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        text.clear();
    }

    std::string get() {
        std::lock_guard<std::mutex> lock(mutex);
        return text;
    }

    bool wait_for(
            const char * pattern,
            const std::atomic<bool> & stop,
            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&] {
            return text.find(pattern) != std::string::npos || stop.load();
        }) && text.find(pattern) != std::string::npos;
    }
};

static void log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    log_capture & capture = *static_cast<log_capture *>(user_data);
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        capture.text += text;
    }
    capture.cv.notify_all();
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct invalidation_record {
    const void * base;
    size_t size;
};

static std::vector<invalidation_record> * invalidation_records = nullptr;

static void record_invalidation(const void * base, size_t size) {
    if (invalidation_records) {
        invalidation_records->push_back({base, size});
    }
}

struct reset_buffer_context {
    uint8_t data[64];
    bool reset = false;
};

static void * reset_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((reset_buffer_context *)buffer->context)->data;
}

static void reset_buffer_reset(ggml_backend_buffer_t buffer) {
    ((reset_buffer_context *)buffer->context)->reset = true;
}

static bool run_invalidation_hook_coverage(ggml_backend_t cpu) {
    if (!ggml_moe_cache.invalidate) {
        fprintf(stderr, "cache-invalidation-hooks: cache API is not registered\n");
        return false;
    }

    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-invalidation-hooks: failed to create context\n");
        return false;
    }

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-invalidation-hooks: failed to allocate tensors\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> data(64, 0.25f);
    std::vector<invalidation_record> records;
    auto original_invalidate = ggml_moe_cache.invalidate;
    invalidation_records = &records;
    ggml_moe_cache.invalidate = record_invalidation;

    bool ok = true;
    auto check = [&](const char * operation,
                     std::initializer_list<invalidation_record> expected) {
        bool matches = records.size() == expected.size();
        size_t index = 0;
        for (const invalidation_record & item : expected) {
            if (index >= records.size() || records[index].base != item.base ||
                records[index].size != item.size) {
                matches = false;
            }
            index++;
        }
        if (!matches) {
            fprintf(stderr,
                    "cache-invalidation-hooks: %s produced %zu records, expected %zu\n",
                    operation, records.size(), expected.size());
        }
        records.clear();
        ok &= matches;
    };

    char * dst_base = (char *)dst->data;
    void * buffer_base = ggml_backend_buffer_get_base(buffer);
    const size_t buffer_size = ggml_backend_buffer_get_size(buffer);

    ggml_backend_tensor_set(dst, data.data(), 4, 16);
    check("tensor set", {{dst_base + 4, 16}});

    ggml_backend_tensor_set_async(cpu, dst, data.data(), 8, 20);
    check("async tensor set", {{dst_base + 8, 20}});

    ggml_backend_tensor_set_2d(
            dst, data.data(), 0, 8, 3, 16, 8);
    check("2D tensor set", {
            {dst_base, 8}, {dst_base + 16, 8}, {dst_base + 32, 8}});

    ggml_backend_tensor_set_2d_async(
            cpu, dst, data.data(), 4, 8, 3, 16, 8);
    check("async 2D tensor set", {
            {dst_base + 4, 8}, {dst_base + 20, 8}, {dst_base + 36, 8}});

    ggml_backend_tensor_memset(dst, 0, 12, 24);
    check("tensor memset", {{dst_base + 12, 24}});

    if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        fprintf(stderr, "cache-invalidation-hooks: direct tensor copy failed\n");
        ok = false;
    }
    check("direct tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy(src, dst);
    check("tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy_async(cpu, cpu, src, dst);
    check("async tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_buffer_clear(buffer, 0);
    check("buffer clear", {{buffer_base, buffer_size}});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    check("weight usage removal", {{buffer_base, buffer_size}});

    ggml_backend_tensor_set(dst, data.data(), 0, 16);
    check("non-weight tensor set", {});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    check("weight usage restoration", {});

    ggml_backend_buffer_reset(buffer);
    check("buffer reset without callback", {});

    reset_buffer_context reset_context;
    ggml_backend_buffer_i reset_iface = {};
    reset_iface.get_base = reset_buffer_get_base;
    reset_iface.reset = reset_buffer_reset;
    ggml_backend_buffer_t reset_buffer = ggml_backend_buffer_init(
            ggml_backend_cpu_buffer_type(), reset_iface,
            &reset_context, sizeof(reset_context.data));
    ggml_backend_buffer_set_usage(
            reset_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_reset(reset_buffer);
    check("buffer reset", {
            {reset_context.data, sizeof(reset_context.data)}});
    if (!reset_context.reset) {
        fprintf(stderr, "cache-invalidation-hooks: reset callback was not called\n");
        ok = false;
    }
    ggml_backend_buffer_free(reset_buffer);
    check("reset buffer free", {
            {reset_context.data, sizeof(reset_context.data)}});

    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
    check("buffer free", {{buffer_base, buffer_size}});

    ggml_moe_cache.invalidate = original_invalidate;
    invalidation_records = nullptr;
    ggml_free(ctx);
    printf("cache-invalidation-hooks: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool has_positive_field(const std::string & text, const char * field) {
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value > 0) {
            return true;
        }
    }
    return false;
}

static long long max_field_value(const std::string & text, const char * field) {
    long long result = -1;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position) {
            result = std::max(result, value);
        }
    }
    return result;
}

static bool has_partial_fraction(const std::string & text, const char * field) {
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * numerator_end = nullptr;
        const long long numerator =
            strtoll(text.c_str() + position, &numerator_end, 10);
        if (numerator_end == text.c_str() + position ||
            *numerator_end != '/') {
            position++;
            continue;
        }
        char * denominator_end = nullptr;
        const long long denominator =
            strtoll(numerator_end + 1, &denominator_end, 10);
        if (denominator_end != numerator_end + 1 &&
            numerator > 0 && numerator < denominator) {
            return true;
        }
        position = denominator_end - text.c_str();
    }
    return false;
}

static size_t count_field_at_least(
        const std::string & text, const char * field, long long minimum) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value >= minimum) {
            result++;
        }
    }
    return result;
}

static size_t count_occurrences(const std::string & text, const char * pattern) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(pattern, position)) != std::string::npos) {
        result++;
        position += strlen(pattern);
    }
    return result;
}

static bool is_cache_gpu(ggml_backend_dev_t device) {
    if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const char * name = ggml_backend_reg_name(
            ggml_backend_dev_backend_reg(device));
    return strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0;
}

static ggml_backend_dev_t find_cuda_device() {
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (is_cache_gpu(device)) {
            return device;
        }
    }
    return nullptr;
}

static ggml_backend_dev_t find_other_cuda_device(
        ggml_backend_dev_t excluded) {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device != excluded && is_cache_gpu(device)) {
            return device;
        }
    }
    return nullptr;
}

static long cuda_physical_device(ggml_backend_dev_t device) {
    const char * description = ggml_backend_dev_description(device);
    const char * marker = description ? strstr(description, "(physical device ") : nullptr;
    if (!marker) {
        return -1;
    }
    marker += strlen("(physical device ");
    char * end = nullptr;
    const long physical = strtol(marker, &end, 10);
    return end != marker ? physical : -1;
}

static ggml_backend_t init_cpu_backend() {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend =
            ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        ggml_backend_reg_t reg =
            ggml_backend_dev_backend_reg(device);
        auto set_n_threads =
            (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(backend, 4);
        }
        return backend;
    }
    return nullptr;
}

static bool compare_output(
        const std::vector<float> & reference,
        const std::vector<float> & actual,
        double max_nmse) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t index = 0; index < reference.size(); index++) {
        if (!std::isfinite(actual[index])) {
            return false;
        }
        const double difference = (double) actual[index] - reference[index];
        squared_error += difference * difference;
        squared_reference += (double) reference[index] * reference[index];
    }
    return squared_error / std::max(squared_reference, 1e-12) <= max_nmse;
}

static void configure_cache(
        const char * fail_stage,
        const char * max_batch = "1",
        const char * dedicated_mmv = "1",
        const char * budget_mb = "4",
        const char * dedicated_down_mmv = "0") {
    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "on");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", budget_mb);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "1");
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", max_batch);
    set_env("GGML_CUDA_MOE_CACHE_INSERTS", "4");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "1");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "1");
    set_env("GGML_CUDA_MOE_CACHE_QUEUE", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "1");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "1");
    set_env("GGML_CUDA_MOE_CACHE_EXPERT_PARALLEL", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", "0");
    set_env("GGML_CUDA_MOE_CACHE_SERIAL_FILL", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", dedicated_mmv);
    set_env("GGML_CUDA_MOE_CACHE_DOWN_DEDICATED_MMV", dedicated_down_mmv);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "0");
    set_env("GGML_CUDA_MOE_CACHE_FAIL", fail_stage);
    set_env("GGML_CUDA_MOE_CACHE_LFU", nullptr);
}

static bool run_capability_queries(
        ggml_backend_dev_t cuda_device, ggml_backend_t cpu) {
    if (!ggml_moe_cache.query_config || !ggml_moe_cache.query_device ||
        !ggml_moe_cache.query_shape) {
        fprintf(stderr, "cache-capabilities: query API is incomplete\n");
        return false;
    }

    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    bool ok = ggml_moe_cache.query_config(1, 0, &config) == 1;
    ok &= config.automatic == 1;
    ok &= config.min_devices == 1;
    ok &= config.budget_bytes == 4u * 1024 * 1024;
    ok &= config.reserve_bytes == 0;
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_expert_bytes == 1024;
    ok &= config.min_expert_explicit == 1;
    ok &= config.overlap_cpu_rows == 0;

    ggml_moe_cache_device_caps device = {};
    ok &= ggml_moe_cache.query_device(cuda_device, &config, &device) == 1;
    ok &= device.logical_device >= 0;
    ok &= device.physical_device >= 0;
    ok &= device.compute_capability >= config.min_compute_capability;
    ok &= device.min_expert_bytes == 1024;
    ok &= ggml_moe_cache.query_device(cpu->device, &config, &device) == 0;

    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", nullptr);
    ggml_moe_cache_config automatic_reserve = {};
    ggml_moe_cache_device_caps automatic_caps = {};
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t reserve_step = 128 * MiB;
    const int automatic_config_ok =
        ggml_moe_cache.query_config(0, 4, &automatic_reserve);
    ok &= automatic_config_ok == 1;
    ok &= automatic_reserve.reserve_explicit == 0;
    const int automatic_device_ok = ggml_moe_cache.query_device(
            cuda_device, &automatic_reserve, &automatic_caps);
    ok &= automatic_device_ok == 1;
    ok &= automatic_caps.recommended_reserve_bytes >= 512 * MiB;
    ok &= automatic_caps.recommended_reserve_bytes <= 3072 * MiB;
    ok &= automatic_caps.recommended_reserve_bytes % reserve_step == 0;
    if (std::strstr(ggml_backend_dev_name(cuda_device), "RTX 3090")) {
        ok &= automatic_caps.recommended_reserve_bytes == 640 * MiB;
    }
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "2048");
    ggml_moe_cache_config explicit_reserve = {};
    ggml_moe_cache_device_caps explicit_caps = {};
    ok &= ggml_moe_cache.query_config(0, 4, &explicit_reserve) == 1;
    ok &= explicit_reserve.reserve_explicit == 1;
    ok &= ggml_moe_cache.query_device(
            cuda_device, &explicit_reserve, &explicit_caps) == 1;
    ok &= explicit_caps.recommended_reserve_bytes == 2048 * MiB;
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", nullptr);
    if (!ok) {
        fprintf(stderr,
                "cache-capabilities debug: budget=%zu reserve=%zu explicit=%d "
                "min-devices=%d min-expert=%zu overlap=%d auto-config=%d "
                "auto-explicit=%d auto-device=%d auto-reserve=%zu dev=%s\n",
                config.budget_bytes, config.reserve_bytes, config.reserve_explicit,
                config.min_devices, config.min_expert_bytes, config.overlap_cpu_rows,
                automatic_config_ok, automatic_reserve.reserve_explicit,
                automatic_device_ok,
                automatic_caps.recommended_reserve_bytes,
                ggml_backend_dev_name(cuda_device));
    }
    configure_cache(nullptr);

    const size_t expert_size = ggml_row_size(GGML_TYPE_Q4_0, n_in) * n_out;
    ggml_moe_cache_shape_caps shape = {};
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= shape.minimum_bytes == shape.scratch_bytes + shape.pool_bytes;
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_F32, n_in, n_out, 64, expert_size, &shape) == 0;
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size - 1, &shape) == 0;
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 1, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 0, expert_size, &shape) == 0;

    const size_t q8_g128_expert_size =
        ggml_row_size(GGML_TYPE_Q8_0_G128, n_in) * n_out;
    ok &= ggml_moe_cache.query_shape(
            GGML_TYPE_Q8_0_G128, n_in, n_out, 64,
            q8_g128_expert_size, &shape) == 1;
    ok &= shape.pool_bytes == q8_g128_expert_size * 64;

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    ok &= ggml_moe_cache.query_config(-1, 0, &config) == 0;

    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "auto");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    ok &= ggml_moe_cache.query_config(1, 0, &config) == 1;
    ok &= config.automatic == 1;
    ok &= config.min_devices == 1;
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_compute_capability == 700;
    ok &= config.min_expert_bytes == 512u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 10;
    ok &= config.overlap_cpu_rows == -1;
    // Automatic and forced modes both support Volta; the byte thresholds for
    // ordinary quant types still depend on the actual device architecture.
    // Keep the physical capability from the successful forced probe above.
    const bool auto_admitted = device.compute_capability >= config.min_compute_capability;
    ok &= ggml_moe_cache.query_device(cuda_device, &config, &device) == int(auto_admitted);
    if (auto_admitted) {
        ok &= device.min_expert_bytes == (device.compute_capability >= 800
                ? 512u * 1024 : device.compute_capability >= 750 ? 256u * 1024 : 1024u * 1024);
    }

    ok &= ggml_moe_cache.query_config(0, 0, &config) == 1;
    ok &= config.automatic == 0;
    ok &= config.min_devices == 1;
    ok &= config.minimum_slab_bytes == 0;
    ok &= config.min_compute_capability == 700;
    ok &= config.min_expert_bytes == 1024u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 10;
    ok &= config.overlap_cpu_rows == -1;
    ok &= ggml_moe_cache.query_device(cuda_device, &config, &device) == 1;
    ok &= device.min_expert_bytes == (device.compute_capability >= 800
            ? 512u * 1024 : device.compute_capability >= 750 ? 256u * 1024 : 1024u * 1024);
    configure_cache(nullptr);

    printf("cache-capabilities: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static test_graph make_graph(
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        8 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    result.out = ggml_mul_mat_id(result.ctx, weights, activations, ids);
    ggml_set_name(result.out, "moe_cache_test_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        12 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    result.out = ggml_swiglu_split(result.ctx, gate, up);
    ggml_set_name(up, "moe_cache_fused_up");
    ggml_set_name(gate, "moe_cache_fused_gate");
    ggml_set_name(result.out, "moe_cache_fused_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

enum class ffn_clamp { none, separate, combined };

static test_graph make_full_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * down_weights,
        ggml_tensor * activations,
        ggml_tensor * ids,
        ffn_clamp clamp) {
    ggml_init_params params = {
        16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    ggml_tensor * glu = nullptr;
    if (clamp == ffn_clamp::combined) {
        glu = ggml_swiglu_clamp(result.ctx, gate, up, 0.20f);
    } else if (clamp == ffn_clamp::separate) {
        ggml_tensor * up_clamped =
            ggml_clamp(result.ctx, up, -0.25f, 0.25f);
        ggml_tensor * gate_clamped = ggml_clamp(
                result.ctx, gate,
                -std::numeric_limits<float>::infinity(), 0.20f);
        glu = ggml_swiglu_split(result.ctx, gate_clamped, up_clamped);
    } else {
        glu = ggml_swiglu_split(result.ctx, gate, up);
    }
    result.out = ggml_mul_mat_id(
            result.ctx, down_weights, glu, ids);
    ggml_set_name(up, "moe_cache_full_up");
    ggml_set_name(gate, "moe_cache_full_gate");
    ggml_set_name(glu, "moe_cache_full_glu");
    ggml_set_name(result.out, "moe_cache_full_down");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_clamped_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    ggml_tensor * up_clamped =
        ggml_clamp(result.ctx, up, -0.25f, 0.25f);
    ggml_tensor * gate_clamped = ggml_clamp(
            result.ctx, gate,
            -std::numeric_limits<float>::infinity(), 0.20f);
    result.out = ggml_swiglu_split(
            result.ctx, gate_clamped, up_clamped);
    ggml_set_name(up, "moe_cache_clamped_up");
    ggml_set_name(gate, "moe_cache_clamped_gate");
    ggml_set_name(up_clamped, "moe_cache_clamped_up_limit");
    ggml_set_name(gate_clamped, "moe_cache_clamped_gate_limit");
    ggml_set_name(result.out, "moe_cache_clamped_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static void free_graph(test_graph & graph) {
    if (graph.buffer) {
        ggml_backend_buffer_free(graph.buffer);
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
    }
    graph = {};
}

static void poison_graph(test_graph & graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph.graph, i);
        GGML_ASSERT(node->type == GGML_TYPE_F32);
        std::vector<float> poison(ggml_nelements(node), std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(node, poison.data(), 0, poison.size()*sizeof(float));
    }
}

// Exercise CPU misses and collection success/failure independently of GPU admission.
// Repeated experts exercise the IQ panel path as well as ordinary vec-dot work.
static bool run_fused_cpu_fallbacks(ggml_backend_t cpu) {
    const auto saved_api = ggml_moe_cache;
    static uint64_t forced_mask;
    static int fused_calls;
    static int full_calls;
    static bool fail_dispatch;
    static bool fail_collect;
    static bool pair_only;
    static const std::vector<float> * collect_reference;
    static int end_calls;
    static int collect_calls;
    bool ok = true;
    for (ggml_type type : { GGML_TYPE_Q2_0, GGML_TYPE_Q2_0_G128, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS }) {
        for (int tokens : { 1, 2, 3, 4, 8, 16 }) {
            ggml_context * ctx = ggml_init({ 8*ggml_tensor_overhead(), nullptr, true });
            GGML_ASSERT(ctx);
            constexpr int width = 256;
            ggml_tensor * up = ggml_new_tensor_3d(ctx, type, width, width, 4);
            ggml_tensor * gate = ggml_dup_tensor(ctx, up);
            ggml_tensor * down = ggml_dup_tensor(ctx, up);
            ggml_tensor * acts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, tokens);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, tokens);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            GGML_ASSERT(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> imatrix(width, 1.0f);
            int seed = 0;
            for (ggml_tensor * weight : { up, gate, down }) {
                std::vector<float> data(ggml_nelements(weight));
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = 0.02f*std::sin(float(i + 37*seed)*0.013f);
                }
                std::vector<uint8_t> quantized(ggml_nbytes(weight));
                GGML_ASSERT(ggml_quantize_chunk(type, data.data(), quantized.data(),
                            0, width*4, width, imatrix.data()) == quantized.size());
                ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
                ++seed;
            }
            std::vector<float> activation(ggml_nelements(acts));
            for (size_t i = 0; i < activation.size(); ++i) {
                activation[i] = std::sin(float(i)*0.037f);
            }
            ggml_backend_tensor_set(acts, activation.data(), 0, activation.size()*sizeof(float));
            std::vector<int32_t> routes(4*tokens);
            for (size_t i = 0; i < routes.size(); ++i) {
                routes[i] = i % 4;
            }
            ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*sizeof(int32_t));
            test_graph graph = make_full_fused_graph(cpu, up, gate, down, acts, ids, ffn_clamp::combined);
            GGML_ASSERT(graph.ctx && graph.buffer);
            ggml_moe_cache = {};
            ok &= ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
            std::vector<float> reference(ggml_nelements(graph.out));
            ggml_backend_tensor_get(graph.out, reference.data(), 0, reference.size()*sizeof(float));
            const ggml_tensor * glu = graph.out->src[1];
            std::vector<float> glu_reference(ggml_nelements(glu));
            ggml_backend_tensor_get(glu, glu_reference.data(), 0, glu_reference.size()*sizeof(float));
            ggml_moe_cache.fused_dispatch = [](void *) { return fail_dispatch ? 0 : 1; };
            ggml_moe_cache.fused_plan = [](
                    const ggml_moe_cache_tensor_desc *, const ggml_moe_cache_tensor_desc *,
                    const ggml_moe_cache_tensor_desc * down, int, float, float, float, float,
                    const int32_t *, int rows, int64_t, const float * const *, uint64_t * mask) -> void * {
                if (pair_only && down) {
                    return nullptr;
                }
                ++fused_calls;
                full_calls += down != nullptr;
                const uint64_t valid = rows == 64 ? UINT64_MAX : (UINT64_C(1) << rows) - 1;
                *mask = forced_mask & valid;
                return &forced_mask;
            };
            ggml_moe_cache.collect = [](void *, int hits, float * const * dst, int64_t width) {
                ++collect_calls;
                if (fail_collect) {
                    return hits == 0 ? 1 : 0;
                }
                int copied = 0;
                const size_t rows = collect_reference->size()/width;
                for (size_t row = 0; row < rows; ++row) {
                    if (forced_mask & (UINT64_C(1) << row)) {
                        GGML_ASSERT(copied < hits);
                        std::memcpy(dst[copied++], collect_reference->data() + row*width, width*sizeof(float));
                    }
                }
                return copied == hits ? 1 : 0;
            };
            ggml_moe_cache.end = [](void *) { ++end_calls; };
            for (bool partial : { false, true }) {
                pair_only = partial;
                collect_reference = partial ? &glu_reference : &reference;
                for (int failure : { 0, 1, 2 }) {
                    fail_dispatch = failure == 1;
                    fail_collect = failure == 2;
                    for (uint64_t mask : { UINT64_C(0), UINT64_MAX, UINT64_C(0xf0f0f0f0f0f0f0f0) }) {
                        forced_mask = mask;
                        fused_calls = full_calls = end_calls = collect_calls = 0;
                        // Reference execution populated the intermediates too. Poison
                        // every computed tensor so omitted up/gate/GLU work cannot pass.
                        poison_graph(graph);
                        std::vector<float> actual(reference.size());
                        ok &= ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
                        ggml_backend_tensor_get(graph.out, actual.data(), 0, actual.size()*sizeof(float));
                        // Mixed hits can move an expert below the IQ panel's batch cutoff.
                        const bool match = compare_output(reference, actual, 1e-10);
                        const bool cell_ok = match && fused_calls == 1 &&
                            (partial ? full_calls == 0 : full_calls > 0) && end_calls == fused_calls &&
                            (fail_dispatch ? collect_calls == 0 : collect_calls == fused_calls);
                        printf("cache-fused-cpu-%s-tokens%d-mask%llx-pair%d-failure%d: %s\n", ggml_type_name(type),
                                tokens, (unsigned long long)mask, partial, failure, cell_ok ? "OK" : "FAIL");
                        ok &= cell_ok;
                    }
                }
            }
            fail_dispatch = false;
            pair_only = false;
            const auto forced_api = ggml_moe_cache;
            for (bool window_pair : { false, true }) {
                ggml_tensor * glu = graph.out->src[1];
                ggml_mul_mat_id_set_expert_window(glu->src[0], window_pair ? 2 : 0, window_pair ? 4 : 0);
                ggml_mul_mat_id_set_expert_window(glu->src[1], window_pair ? 2 : 0, window_pair ? 4 : 0);
                ggml_mul_mat_id_set_expert_window(graph.out, 2, 4);
                // IDs 0/1 must zero when the window is active; 2/3 must
                // select local experts 0/1, not cache experts 2/3.
                ggml_moe_cache = {};
                ok &= ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
                ggml_backend_tensor_get(graph.out, reference.data(), 0, reference.size()*sizeof(float));
                ggml_moe_cache = forced_api;
                forced_mask = 0;
                fused_calls = full_calls = 0;
                poison_graph(graph);
                ok &= ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
                std::vector<float> actual(reference.size());
                ggml_backend_tensor_get(graph.out, actual.data(), 0, actual.size()*sizeof(float));
                const bool cell_ok = compare_output(reference, actual, 1e-10) &&
                    full_calls == 0 && (window_pair ? fused_calls == 0 : fused_calls > 0);
                printf("cache-fused-window-%s-tokens%d-pair%d: %s\n", ggml_type_name(type),
                        tokens, window_pair, cell_ok ? "OK" : "FAIL");
                ok &= cell_ok;
            }
            ggml_moe_cache = {};
            free_graph(graph);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_moe_cache = saved_api;
    return ok;
}

// A graph allocator can reuse an expired input/intermediate for a later output.
// Equal-width, separately allocated tensors do not exercise recovery from these
// aliases, especially after collect has already written part of its output.
static bool run_fused_alias_fallbacks(ggml_backend_t cpu) {
    const auto saved_api = ggml_moe_cache;
    static uint64_t mask;
    static int failure, plans, full_plans;
    static bool full;
    static std::vector<float> reference, glu_reference;
    constexpr int width = 256, inner = 64, experts = 4, tokens = 4;
    auto * ctx = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
    GGML_ASSERT(ctx);
    auto * up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, width, inner, experts);
    auto * gate = ggml_dup_tensor(ctx, up);
    auto * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, inner, width, experts);
    auto * acts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, experts, tokens);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 0;
    for (auto * weight : {up, gate, down}) {
        std::vector<float> data(ggml_nelements(weight));
        for (size_t i = 0; i < data.size(); ++i) data[i] = 0.02f*std::sin(float(i + 37*seed)*0.013f);
        std::vector<uint8_t> bytes(ggml_nbytes(weight));
        GGML_ASSERT(ggml_quantize_chunk(weight->type, data.data(), bytes.data(), 0,
                    weight->ne[1]*experts, weight->ne[0], nullptr) == bytes.size());
        ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
        ++seed;
    }
    std::vector<float> activation(ggml_nelements(acts));
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = std::sin(float(i)*0.037f);
    ggml_backend_tensor_set(acts, activation.data(), 0, ggml_nbytes(acts));
    std::vector<int32_t> routes(experts*tokens);
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = i % experts;
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    auto graph = make_full_fused_graph(cpu, up, gate, down, acts, ids, ffn_clamp::combined);
    auto * glu = graph.out->src[1];
    auto * up_node = glu->src[1];
    auto * gate_node = glu->src[0];
    ggml_moe_cache = {};
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS);
    reference.resize(ggml_nelements(graph.out));
    glu_reference.resize(ggml_nelements(glu));
    ggml_backend_tensor_get(graph.out, reference.data(), 0, ggml_nbytes(graph.out));
    ggml_backend_tensor_get(glu, glu_reference.data(), 0, ggml_nbytes(glu));
    void * up_data = up_node->data;
    void * gate_data = gate_node->data;
    void * acts_data = acts->data;
    auto acts_buffer = acts->buffer;
    ggml_moe_cache.fused_plan = [](const ggml_moe_cache_tensor_desc *, const ggml_moe_cache_tensor_desc *,
            const ggml_moe_cache_tensor_desc * down, int, float, float, float, float,
            const int32_t *, int, int64_t, const float * const *, uint64_t * hits) -> void * {
        ++plans;
        full = down != nullptr;
        full_plans += full;
        *hits = mask;
        return &mask;
    };
    ggml_moe_cache.fused_dispatch = [](void *) { return failure == 1 ? 0 : 1; };
    ggml_moe_cache.collect = [](void *, int hits, float * const * out, int64_t n) {
        const auto & expected = full ? reference : glu_reference;
        int copied = 0;
        for (size_t r = 0; r < expected.size()/n; ++r) {
            if (!(mask & (UINT64_C(1) << r))) continue;
            GGML_ASSERT(copied < hits);
            std::memcpy(out[copied++], expected.data() + r*n, n*sizeof(float));
            if (failure == 2) return 0;
        }
        return copied == hits ? 1 : 0;
    };
    ggml_moe_cache.end = [](void *) {};
    bool ok = true;
    for (int alias = 0; alias < 5; ++alias) {
        up_node->data = alias == 1 ? graph.out->data : up_data;
        gate_node->data = alias == 2 ? graph.out->data : gate_data;
        acts->data = alias == 3 ? graph.out->data : alias == 4 ? glu->data : acts_data;
        acts->buffer = alias >= 3 ? graph.buffer : acts_buffer;
        for (int fail : {0, 1, 2}) for (uint64_t hits : {UINT64_C(0), UINT64_C(0xffff), UINT64_C(0x5555)}) {
            failure = fail; mask = hits; plans = full_plans = 0;
            poison_graph(graph);
            ggml_backend_tensor_set(acts, activation.data(), 0, ggml_nbytes(acts));
            GGML_ASSERT(ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS);
            std::vector<float> actual(reference.size());
            ggml_backend_tensor_get(graph.out, actual.data(), 0, ggml_nbytes(graph.out));
            // Down over the activations stays fully fused: the hits are dispatched
            // before down is written and the activations restored for recovery.
            const bool cell_ok = std::memcmp(actual.data(), reference.data(), ggml_nbytes(graph.out)) == 0 &&
                plans == (alias == 4 ? 0 : 1) && full_plans == (alias < 4 ? 1 : 0);
            printf("cache-fused-alias%d-mask%llx-failure%d: %s\n", alias,
                    (unsigned long long)mask, fail, cell_ok ? "OK" : "FAIL");
            ok &= cell_ok;
        }
    }
    up_node->data = up_data; gate_node->data = gate_data;
    acts->data = acts_data; acts->buffer = acts_buffer;
    ggml_moe_cache = saved_api;
    free_graph(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

enum class down_mmv_expectation {
    unchecked,
    generic,
    dedicated,
};

struct scenario_options {
    const char * max_batch = "1";
    const char * required_field = nullptr;
    const char * dedicated_mmv = "1";
    const char * expert_parallel = nullptr;
    const char * n_devices = "1";
    const std::vector<ggml_backend_t> * supplied_backends = nullptr;
    bool use_expert_parallel_policy_defaults = false;
    const char * budget_mb = "4";
    const char * dedicated_down_mmv = "0";
    down_mmv_expectation down_mmv = down_mmv_expectation::unchecked;
};

static bool run_scenario(
        const char * name,
        const char * fail_stage,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture,
        const scenario_options & options = {}) {
    configure_cache(
            fail_stage, options.max_batch, options.dedicated_mmv,
            options.budget_mb, options.dedicated_down_mmv);
    if (options.use_expert_parallel_policy_defaults) {
        set_env("GGML_CUDA_MOE_CACHE_INSERTS", nullptr);
        set_env("GGML_CUDA_MOE_CACHE_THROTTLE", nullptr);
    }
    set_env("GGML_CUDA_MOE_CACHE_EXPERT_PARALLEL", options.expert_parallel);
    set_env("GGML_CUDA_MOE_CACHE_NDEV", options.n_devices);
    capture.clear();

    std::vector<ggml_backend_t> backends = options.supplied_backends
        ? *options.supplied_backends
        : std::vector<ggml_backend_t>{ cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends.data(), nullptr, (int)backends.size(),
            GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return false;
    }

    bool output_ok = true;
    bool live_hit = false;
    std::vector<float> actual(reference.size());
    for (int step = 0; step < max_steps; step++) {
        // Do not let a partial-cache bug hide behind correct rows retained from the
        // preceding iteration. Every fused execution must produce every output row.
        std::fill(actual.begin(), actual.end(),
                  std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                    name, step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(reference, actual, 5e-4)) {
            fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
            output_ok = false;
            break;
        }
        if (!fail_stage && has_positive_field(capture.get(), "hits=")) {
            live_hit = true;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    bool stage_ok = false;
    if (!fail_stage) {
        stage_ok = live_hit &&
            log.find("[moe-cache] enabled:") != std::string::npos &&
            max_field_value(log, "dispatch-fail=") == 0 &&
            max_field_value(log, "collect-fail=") == 0 &&
            (!options.required_field ||
             has_positive_field(log, options.required_field));
    } else if (strcmp(fail_stage, "dispatch") == 0 || strcmp(fail_stage, "dispatch-after-first") == 0) {
        stage_ok = has_positive_field(log, "dispatch-fail=");
    } else if (strcmp(fail_stage, "collect") == 0) {
        stage_ok = has_positive_field(log, "collect-fail=");
    } else if (strcmp(fail_stage, "insert") == 0) {
        stage_ok = has_positive_field(log, "fill-fail=");
    } else if (strcmp(fail_stage, "slab") == 0) {
        stage_ok = log.find("allocation failed") != std::string::npos;
    }
    if (options.required_field) {
        stage_ok &= has_positive_field(log, options.required_field);
    }
    if (options.down_mmv == down_mmv_expectation::generic) {
        stage_ok &= log.find("down-mmv=generic") != std::string::npos &&
            max_field_value(log, "down-mmv-dedicated=") == 0;
    } else if (options.down_mmv == down_mmv_expectation::dedicated) {
        stage_ok &= log.find("down-mmv=dedicated") != std::string::npos &&
            has_positive_field(log, "down-mmv-dedicated=");
    }
    if (!stage_ok) {
        fprintf(stderr, "%s: cache stage was not observed\n%s", name, log.c_str());
    }
    printf("%s: %s\n", name, output_ok && stage_ok ? "OK" : "FAIL");
    return output_ok && stage_ok;
}

static bool run_expert_parallel_partial_scenario(
        ggml_backend_dev_t first_device,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    std::vector<ggml_backend_t> backends = { cuda };
    std::vector<ggml_backend_t> owned;
    for (size_t index = 0;
         index < ggml_backend_dev_count() && backends.size() < 2;
         index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == first_device || !is_cache_gpu(device)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        backends.push_back(backend);
        owned.push_back(backend);
    }
    if (backends.size() < 2) {
        for (ggml_backend_t backend : owned) {
            ggml_backend_free(backend);
        }
        printf("cache-expert-parallel-partial: SKIP (fewer than two GPUs)\n");
        return true;
    }
    backends.push_back(cpu);

    scenario_options parallel_options;
    parallel_options.max_batch = "10";
    parallel_options.required_field = "full-fusion=";
    parallel_options.expert_parallel = "2";
    parallel_options.n_devices = "2";
    parallel_options.supplied_backends = &backends;
    parallel_options.use_expert_parallel_policy_defaults = true;
    const bool output_ok = run_scenario(
            "cache-expert-parallel-partial", nullptr, cuda, cpu,
            graph, reference, capture, parallel_options);
    const bool partial_seen = has_partial_fraction(capture.get(), "fusion=");
    const bool policy_defaults =
        capture.get().find("inserts=16") != std::string::npos &&
        capture.get().find("40-replace") != std::string::npos;
    if (output_ok && (!partial_seen || !policy_defaults)) {
        fprintf(stderr,
                "cache-expert-parallel-partial: partial dispatch or policy defaults were not observed\n%s",
                capture.get().c_str());
    }
    scenario_options fallback_options = parallel_options;
    fallback_options.required_field = nullptr;
    fallback_options.use_expert_parallel_policy_defaults = false;
    const bool dispatch_fallback = run_scenario(
            "cache-expert-parallel-dispatch-fallback", "dispatch",
            cuda, cpu, graph, reference, capture, fallback_options);
    const bool collect_fallback = run_scenario(
            "cache-expert-parallel-collect-fallback", "collect",
            cuda, cpu, graph, reference, capture, fallback_options);
    fallback_options.required_field = "full-fusion=";
    const bool partial_dispatch_fallback = run_scenario(
            "cache-expert-parallel-later-dispatch-fallback", "dispatch-after-first",
            cuda, cpu, graph, reference, capture, fallback_options);

    configure_cache(nullptr);
    for (ggml_backend_t backend : owned) {
        ggml_backend_free(backend);
    }
    return output_ok && partial_seen && policy_defaults &&
        dispatch_fallback && collect_fallback && partial_dispatch_fallback;
}

static bool run_multi_token_scenario(
        ggml_backend_dev_t cuda_device,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * gate_weights,
        ggml_tensor * down_weights,
        log_capture & capture) {
    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-multi-token: failed to create tensor context\n");
        return false;
    }

    ggml_tensor * ids = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, multi_n_used, multi_n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, n_in, 1, multi_n_tokens);
    ggml_set_name(ids, "moe_cache_multi_token_ids");
    ggml_set_name(activations, "moe_cache_multi_token_activations");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-multi-token: failed to allocate CPU tensors\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> ids_data(multi_n_used * multi_n_tokens);
    for (int64_t token = 0; token < multi_n_tokens; token++) {
        for (int64_t id = 0; id < multi_n_used; id++) {
            ids_data[token * multi_n_used + id] =
                (int32_t) ((token * 7 + id * 3) % n_expert);
        }
    }
    ggml_backend_tensor_set(
            ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.31f * std::sin((float) index * 0.053f) -
            0.12f * std::cos((float) index * 0.089f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    test_graph full_fused_graph = make_full_fused_graph(
            cpu, weights, gate_weights, down_weights,
            activations, ids, ffn_clamp::none);
    bool ok = graph.ctx && graph.buffer &&
        fused_graph.ctx && fused_graph.buffer &&
        clamped_fused_graph.ctx && clamped_fused_graph.buffer &&
        full_fused_graph.ctx && full_fused_graph.buffer;
    if (!ok) {
        fprintf(stderr, "cache-multi-token: failed to create graphs\n");
    } else {
        set_env("GGML_CUDA_MOE_CACHE", "0");
        auto make_reference = [&](const char * name, test_graph & test,
                                  std::vector<float> & reference) {
            if (ggml_backend_graph_compute(cpu, test.graph) !=
                    GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: CPU reference compute failed\n", name);
                return false;
            }
            reference.resize(ggml_nelements(test.out));
            ggml_backend_tensor_get(
                    test.out, reference.data(), 0,
                    reference.size() * sizeof(float));
            return true;
        };
        std::vector<float> reference;
        std::vector<float> fused_reference;
        std::vector<float> clamped_fused_reference;
        std::vector<float> full_fused_reference;
        ok = make_reference("cache-multi-token", graph, reference) &&
            make_reference(
                    "cache-fused-multi-token", fused_graph,
                    fused_reference) &&
            make_reference(
                    "cache-fused-clamped-multi-token",
                    clamped_fused_graph, clamped_fused_reference) &&
            make_reference(
                    "cache-fused-full-ffn-multi-token",
                    full_fused_graph, full_fused_reference);
        scenario_options multi_options;
        multi_options.max_batch = nullptr;
        multi_options.required_field = "act-dedup=";
        ok = ok && run_scenario(
                "cache-multi-token", nullptr, cuda, cpu,
                graph, reference, capture, multi_options);
        multi_options.required_field = "fusion-nodes=";
        ok = ok && run_scenario(
                "cache-fused-multi-token", nullptr, cuda, cpu,
                fused_graph, fused_reference, capture, multi_options);
        ok = ok && run_scenario(
                "cache-fused-clamped-multi-token", nullptr, cuda, cpu,
                clamped_fused_graph, clamped_fused_reference, capture,
                multi_options);
        multi_options.required_field = "full-fusion=";
        multi_options.budget_mb = "8";
        ok = ok && run_scenario(
                "cache-fused-full-ffn-multi-token", nullptr, cuda, cpu,
                full_fused_graph, full_fused_reference, capture,
                multi_options);
        ok = ok && run_expert_parallel_partial_scenario(
                cuda_device, cuda, cpu, full_fused_graph,
                full_fused_reference, capture);
    }

    free_graph(full_fused_graph);
    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

struct mxfp4_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_mxfp4_fixture(mxfp4_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_mxfp4_fixture(
        const char * name,
        const char * weight_name,
        int64_t mxfp4_n_in,
        int64_t mxfp4_n_out,
        ggml_backend_t cpu,
        mxfp4_fixture & fixture,
        ggml_type type = GGML_TYPE_MXFP4,
        int64_t tokens = n_tokens,
        int64_t used = n_used,
        int64_t activation_rows = 1) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        fprintf(stderr, "%s: failed to create tensor context\n", name);
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, type,
            mxfp4_n_in, mxfp4_n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, used, tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, mxfp4_n_in, activation_rows, tokens);
    ggml_set_name(fixture.weights, weight_name);
    ggml_set_name(ids, "moe_cache_mxfp4_ids");
    ggml_set_name(activations, "moe_cache_mxfp4_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        fprintf(stderr, "%s: failed to allocate CPU tensors\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.18f * std::sin((float) (index % 991) * 0.023f) +
            0.07f * std::cos((float) (index % 421) * 0.037f);
    }
    std::vector<uint8_t> weights_mxfp4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            type, weights_f32.data(), weights_mxfp4.data(),
            0, mxfp4_n_out * n_expert, mxfp4_n_in, nullptr);
    if (quantized != weights_mxfp4.size()) {
        fprintf(stderr, "%s: unexpected quantized size: %zu != %zu\n",
                name, quantized, weights_mxfp4.size());
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_mxfp4.data(), 0, weights_mxfp4.size());

    std::vector<int32_t> ids_data(used*tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t expert = 0; expert < used; ++expert) {
            ids_data[token*used + expert] = (expert + token) % n_expert;
        }
    }
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.45f * std::sin((float) index * 0.061f) -
            0.16f * std::cos((float) index * 0.097f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, activations, ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "%s: failed to create graph\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: CPU reference compute failed\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static bool run_mxfp4_shared_pool(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        log_capture & capture) {
    mxfp4_fixture up;
    mxfp4_fixture down;
    bool initialized = init_mxfp4_fixture(
            "cache-mxfp4-up", "blk.6.ffn_up_exps.weight",
            n_in, n_out, cpu, up);
    initialized = initialized && init_mxfp4_fixture(
            "cache-mxfp4-down", "blk.6.ffn_down_exps.weight",
            n_out, n_in, cpu, down);
    if (!initialized) {
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }
    if (ggml_nbytes(up.weights) != ggml_nbytes(down.weights)) {
        fprintf(stderr, "cache-mxfp4-shared-pool: expert sizes differ\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-mxfp4-shared-pool: failed to create scheduler\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    auto run_orientation = [&](const char * name, mxfp4_fixture & fixture,
                               long long hits_before) {
        ggml_backend_sched_reset(scheduler);
        ggml_backend_sched_set_tensor_backend(
                scheduler, fixture.graph.out, cpu);
        if (!ggml_backend_sched_alloc_graph(
                    scheduler, fixture.graph.graph) ||
            ggml_backend_sched_get_tensor_backend(
                    scheduler, fixture.graph.out) != cpu) {
            fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
            return false;
        }

        std::vector<float> actual(fixture.reference.size());
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(
                        scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                        name, step, ggml_status_to_string(status));
                return false;
            }
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            if (!compare_output(fixture.reference, actual, 5e-4)) {
                fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
                return false;
            }
            const std::string live_log = capture.get();
            if (max_field_value(live_log, "hits=") > hits_before) {
                return max_field_value(live_log, "dispatch-fail=") == 0 &&
                       max_field_value(live_log, "collect-fail=") == 0;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        fprintf(stderr, "%s: cache hit was not observed\n", name);
        return false;
    };

    const bool up_ok = run_orientation("cache-mxfp4-up", up, 0);
    const long long up_hits = max_field_value(capture.get(), "hits=");
    const bool down_ok = up_ok &&
        run_orientation("cache-mxfp4-down", down, up_hits);
    ggml_backend_sched_free(scheduler);

    const std::string log = capture.get();
    const bool one_pool = count_occurrences(log, " pool[") == 1;
    const bool no_failures =
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!one_pool) {
        fprintf(stderr,
                "cache-mxfp4-shared-pool: expected one shared pool\n%s",
                log.c_str());
    }
    const bool ok = up_ok && down_ok && one_pool && no_failures;
    printf("cache-mxfp4-shared-pool: %s\n", ok ? "OK" : "FAIL");
    free_mxfp4_fixture(up);
    free_mxfp4_fixture(down);
    return ok;
}

static bool run_pq2_cache(ggml_backend_t cuda, ggml_backend_t cpu, log_capture & capture) {
    bool ok = true;
    // Exercise actual cache-hit computation, not just admission or byte copies.
    // Both axes use Mooney's dimensions; dispatch/collect failures must retain
    // the same CPU reference through the fallback path.
    for (int tokens : {1, 3, 6})
    for (bool down : {false, true}) {
        const int used = tokens == 1 ? 2 : 3; // also exercise odd hit-pair tails
        printf("cache-pq2-shape: tokens=%d used=%d down=%d\n", tokens, used, down);
        mxfp4_fixture fixture;
        if (!init_mxfp4_fixture("cache-pq2", down ? "blk.6.ffn_down_exps.weight" : "blk.6.ffn_up_exps.weight",
                down ? 640 : 2560, down ? 2560 : 640, cpu, fixture, GGML_TYPE_Q2_0_G128,
                tokens, used, down ? used : 1)) {
            return false;
        }
        scenario_options options;
        options.max_batch = "6";
        options.budget_mb = "32"; // at least the minimum pool of full-sized experts
        options.dedicated_mmv = "0";
        ok &= run_scenario("cache-pq2-generic", nullptr,
                cuda, cpu, fixture.graph, fixture.reference, capture, options);
        std::vector<float> generic(fixture.reference.size());
        ggml_backend_tensor_get(fixture.graph.out, generic.data(), 0, generic.size() * sizeof(float));
        options.dedicated_mmv = "1";
        options.dedicated_down_mmv = "1";
        for (const char * failure : {static_cast<const char *>(nullptr), "dispatch", "collect"}) {
            ok &= run_scenario(down ? "cache-pq2-down" : "cache-pq2-up", failure,
                    cuda, cpu, fixture.graph, fixture.reference, capture, options);
            if (!failure) {
                std::vector<float> dedicated(generic.size());
                ggml_backend_tensor_get(fixture.graph.out, dedicated.data(), 0, dedicated.size() * sizeof(float));
                double delta = 0, magnitude = 0;
                float largest = 0;
                for (size_t i = 0; i < generic.size(); ++i) {
                    largest = std::max(largest, std::abs(dedicated[i] - generic[i]));
                    delta += double(dedicated[i] - generic[i]) * (dedicated[i] - generic[i]);
                    magnitude += double(generic[i]) * generic[i];
                }
                const double nmse = delta / (magnitude + 1e-20);
                ok &= nmse < 1e-10;
                printf("cache-pq2-kernel-pair: down=%d max_delta=%.9g nmse=%.9g\n", down, largest, nmse);
            }
        }
        // A separate rotated projection must also hand its resident rows to
        // the GPU consumer without publishing incomplete host output.
        for (int scenario = 0; scenario < 6; ++scenario) {
            const char * failure = scenario == 1 ? "dispatch" : scenario == 2 ? "collect" :
                                   scenario == 3 ? "collect-retire" : nullptr;
            configure_cache(failure, "6", "1", "32", "1");
            auto * ctx = ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead(), nullptr, true});
            auto * projection = ggml_mul_mat_id(ctx, fixture.weights,
                    fixture.graph.out->src[1], fixture.graph.out->src[2]);
            auto * result = ggml_scale(ctx, projection, .5f);
            if (scenario == 4) ggml_set_output(projection);
            ggml_set_output(result);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            ggml_backend_t backends[] = {cuda, cpu};
            auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE,
                                                scenario == 5, false);
            ggml_backend_sched_set_tensor_backend(sched, projection, cpu);
            ggml_backend_sched_set_tensor_backend(sched, result, cuda);
            bool pass = ggml_backend_sched_alloc_graph(sched, graph);
            const bool handoff_supported = ggml_moe_cache.output_supported &&
                ggml_moe_cache.output_supported(cuda, projection);
            std::vector<float> expected = fixture.reference, actual(expected.size());
            for (auto & value : expected) value *= .5f;
            int deferred_failures = 0;
            for (int step = 0; pass && step < 96; ++step) {
                // A missing device write must not pass using an earlier result.
                std::fill(actual.begin(), actual.end(), std::numeric_limits<float>::quiet_NaN());
                ggml_backend_tensor_set(result, actual.data(), 0, ggml_nbytes(result));
                const auto status = ggml_backend_sched_graph_compute(sched, graph);
                if (scenario == 3 && status == GGML_STATUS_FAILED) {
                    ++deferred_failures;
                    continue;
                }
                pass &= status == GGML_STATUS_SUCCESS;
                ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                pass &= compare_output(expected, actual, 5e-4);
                if (scenario == 4) {
                    ggml_backend_tensor_get(projection, actual.data(), 0, ggml_nbytes(projection));
                    pass &= compare_output(fixture.reference, actual, 5e-4);
                }
            }
            if (scenario == 3 && handoff_supported) pass &= deferred_failures > 0;
            printf("cache-pq2-device-output: down=%d scenario=%d deferred-failures=%d %s\n",
                    down, scenario, deferred_failures, pass ? "OK" : "FAIL");
            ok &= pass;
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        }
        free_mxfp4_fixture(fixture);
    }
    return ok;
}

static bool run_invalidation_scenario(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const std::vector<uint8_t> & replacement_row,
        const std::vector<float> & old_reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-invalidate: failed to create scheduler\n");
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "cache-invalidate: MUL_MAT_ID was not assigned to CPU\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::vector<float> actual(old_reference.size());
    long long hits_before = -1;
    bool output_ok = true;
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(old_reference, actual, 5e-4)) {
            fprintf(stderr, "cache-invalidate: warmup mismatch at step %d\n", step);
            output_ok = false;
            break;
        }
        hits_before = max_field_value(capture.get(), "hits=");
        if (hits_before > 0) {
            break;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (output_ok && hits_before <= 0) {
        fprintf(stderr, "cache-invalidate: no cache hit before mutation\n");
        output_ok = false;
    }

    std::vector<float> new_reference(old_reference.size());
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, replacement_row.data(), 0, replacement_row.size());
        if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: CPU reference compute failed\n");
            output_ok = false;
        } else {
            ggml_backend_tensor_get(
                    graph.out, new_reference.data(), 0,
                    new_reference.size() * sizeof(float));
            float max_change = 0.0f;
            for (size_t index = 0; index < new_reference.size(); index++) {
                max_change = std::max(
                        max_change, std::abs(new_reference[index] - old_reference[index]));
            }
            if (max_change < 0.01f) {
                fprintf(stderr, "cache-invalidate: mutation did not change the reference\n");
                output_ok = false;
            }
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "cache-invalidate: compute failed at step %d: %s\n",
                        step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
            ggml_backend_tensor_get(
                    graph.out, actual.data(), 0, actual.size() * sizeof(float));
            if (!compare_output(new_reference, actual, 5e-4)) {
                fprintf(stderr, "cache-invalidate: stale output at step %d\n", step);
                output_ok = false;
                break;
            }
            if (max_field_value(capture.get(), "hits=") > hits_before) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr, "cache-invalidate: mutated expert was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

constexpr int64_t stress_n_out  = 65;
constexpr int64_t stress_n_used = 64;

struct stress_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_stress_fixture(stress_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_stress_fixture(stress_fixture & fixture, ggml_backend_t cpu) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_Q4_0, n_in, stress_n_out, n_expert);
    fixture.ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, stress_n_used, n_tokens);
    fixture.activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, n_in, stress_n_used, n_tokens);
    ggml_set_name(fixture.weights, "blk.1.ffn_up_exps.weight");
    ggml_set_name(fixture.ids, "moe_cache_stress_ids");
    ggml_set_name(fixture.activations, "moe_cache_stress_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.13f * std::sin((float) (index % 983) * 0.019f) -
            0.04f * std::cos((float) (index % 419) * 0.029f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, stress_n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "stress: unexpected quantized size\n");
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<int32_t> ids_data(stress_n_used);
    for (int32_t index = 0; index < stress_n_used; index++) {
        ids_data[index] = index;
    }
    ggml_backend_tensor_set(
            fixture.ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(
            ggml_nelements(fixture.activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.35f * std::sin((float) index * 0.067f) +
            0.17f * std::cos((float) index * 0.103f);
    }
    ggml_backend_tensor_set(
            fixture.activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, fixture.activations, fixture.ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "stress: failed to create graph\n");
        free_stress_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "stress: CPU reference compute failed\n");
        free_stress_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static ggml_backend_sched_t make_scheduler(
        const char * name,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph) {
    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return nullptr;
    }
    return scheduler;
}

static bool compute_matches(
        const char * name,
        ggml_backend_sched_t scheduler,
        test_graph & graph,
        const std::vector<float> & reference,
        int step) {
    const enum ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, graph.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                name, step, ggml_status_to_string(status));
        return false;
    }
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(
            graph.out, actual.data(), 0, actual.size() * sizeof(float));
    if (!compare_output(reference, actual, 5e-4)) {
        fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
        return false;
    }
    return true;
}

static bool run_precensus_invalidation(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const uint8_t * unchanged_expert,
        size_t expert_size,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-precensus-invalidate", cuda, cpu, graph);
    if (!scheduler) {
        return false;
    }

    bool output_ok = compute_matches(
            "cache-precensus-invalidate", scheduler, graph, reference, 0);
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, unchanged_expert,
                (n_expert - 1) * expert_size, expert_size);
    }
    for (int step = 1; step < 80 && output_ok; step++) {
        output_ok = compute_matches(
                "cache-precensus-invalidate", scheduler, graph,
                reference, step);
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    const bool census_ok =
        log.find("slots=64 ") != std::string::npos;
    if (!census_ok) {
        fprintf(stderr,
                "cache-precensus-invalidate: tensor census was not stable\n%s",
                log.c_str());
    }
    printf("cache-precensus-invalidate: %s\n",
            output_ok && census_ok ? "OK" : "FAIL");
    return output_ok && census_ok;
}

static bool run_concurrent_sessions(
        ggml_backend_dev_t cuda_device,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t cuda_second =
        ggml_backend_dev_init(cuda_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!cuda_second || !cpu_second) {
        fprintf(stderr, "cache-concurrent: failed to create second backends\n");
        if (cuda_second) {
            ggml_backend_free(cuda_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }
    test_graph second_graph = make_graph(
            cpu_second, fixture.weights, fixture.activations, fixture.ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-concurrent-1", cuda, cpu, fixture.graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-concurrent-2", cuda_second, cpu_second, second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    constexpr int concurrent_steps = 112;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < concurrent_steps && output_ok.load(); step++) {
            if (!compute_matches(
                    name, scheduler, graph, fixture.reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-concurrent-1", first, std::ref(fixture.graph));
    std::thread second_thread(
            run, "cache-concurrent-2", second, std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= 2 &&
        count_field_at_least(log, "hits=", stress_n_used) >= 2 &&
        count_field_at_least(log, "used=", stress_n_used) >= 2;
    if (!cache_ok) {
        fprintf(stderr, "cache-concurrent: full cache use was not observed\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(cuda_second);
    ggml_backend_free(cpu_second);
    printf("cache-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_concurrent_sessions(
        ggml_backend_dev_t cuda_device,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids,
        test_graph & first_graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t cuda_second =
        ggml_backend_dev_init(cuda_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!cuda_second || !cpu_second) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second backends\n");
        if (cuda_second) {
            ggml_backend_free(cuda_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }

    test_graph second_graph = make_fused_graph(
            cpu_second, up_weights, gate_weights, activations, ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-fused-concurrent-1", cuda, cpu, first_graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-fused-concurrent-2", cuda_second, cpu_second,
            second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < max_steps && output_ok.load(); step++) {
            if (!compute_matches(name, scheduler, graph, reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-fused-concurrent-1", first,
            std::ref(first_graph));
    std::thread second_thread(
            run, "cache-fused-concurrent-2", second,
            std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_field_at_least(log, "fusion-nodes=", 1) >= 2 &&
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!cache_ok) {
        fprintf(stderr,
                "cache-fused-concurrent: fusion was not observed in both sessions\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(cuda_second);
    ggml_backend_free(cpu_second);
    configure_cache(nullptr);
    printf("cache-fused-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_repeated_lifecycle(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    constexpr int cycles = 8;
    bool ok = true;
    for (int cycle = 0; cycle < cycles && ok; cycle++) {
        capture.clear();
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-fused-lifecycle", cuda, cpu, graph);
        if (!scheduler) {
            ok = false;
            break;
        }

        bool fused = false;
        for (int step = 0; step < max_steps && ok && !fused; step++) {
            ok = compute_matches(
                    "cache-fused-lifecycle", scheduler, graph,
                    reference, step);
            fused = has_positive_field(
                    capture.get(), "fusion-nodes=");
            if (step >= 64 && !fused) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ok &= fused;
        ggml_backend_sched_free(scheduler);
    }

    printf("cache-fused-lifecycle: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_repeated_lifecycle(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    constexpr int cycles = 8;
    constexpr int census_steps = 65;
    bool output_ok = true;
    for (int cycle = 0; cycle < cycles && output_ok; cycle++) {
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-lifecycle", cuda, cpu, fixture.graph);
        if (!scheduler) {
            output_ok = false;
            break;
        }
        for (int step = 0; step < census_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "cache-lifecycle: compute failed in cycle %d step %d: %s\n",
                        cycle, step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
        }
        if (output_ok) {
            std::vector<float> actual(fixture.reference.size());
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            output_ok = compare_output(
                    fixture.reference, actual, 5e-4);
            if (!output_ok) {
                fprintf(stderr,
                        "cache-lifecycle: output mismatch in cycle %d\n", cycle);
            }
        }
        ggml_backend_sched_free(scheduler);
    }

    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= cycles &&
        has_positive_field(log, "enqueued=");
    if (!cache_ok) {
        fprintf(stderr, "cache-lifecycle: fill startup was not observed\n%s",
                log.c_str());
    }
    printf("cache-lifecycle: %s\n",
            output_ok && cache_ok ? "OK" : "FAIL");
    return output_ok && cache_ok;
}

static bool run_fill_invalidation(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-fill-invalidate", cuda, cpu, fixture.graph);
    if (!scheduler) {
        return false;
    }

    std::vector<float> replacement_f32(n_in * stress_n_out);
    for (size_t index = 0; index < replacement_f32.size(); index++) {
        replacement_f32[index] =
            1.1f + 0.2f * std::sin((float) index * 0.043f);
    }
    std::vector<uint8_t> replacement_q4(
            ggml_row_size(GGML_TYPE_Q4_0, n_in) * stress_n_out);
    const size_t replacement_size = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
            0, stress_n_out, n_in, nullptr);
    if (replacement_size != replacement_q4.size()) {
        fprintf(stderr,
                "cache-fill-invalidate: unexpected replacement size\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> mutation_started{false};
    std::atomic<bool> mutation_done{false};
    std::thread mutator([&] {
        if (capture.wait_for(
                " pool[", stop, std::chrono::seconds(5))) {
            mutation_started.store(true);
            ggml_backend_tensor_set(
                    fixture.weights, replacement_q4.data(), 0,
                    replacement_q4.size());
            mutation_done.store(true);
        }
    });

    bool output_ok = true;
    for (int step = 0; step < max_steps && !mutation_done.load(); step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "cache-fill-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        if (mutation_started.load()) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!mutation_done.load() &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            break;
        }
    }
    stop.store(true);
    capture.cv.notify_all();
    mutator.join();
    if (!mutation_started.load() || !mutation_done.load()) {
        fprintf(stderr,
                "cache-fill-invalidate: concurrent mutation did not complete\n");
        output_ok = false;
    }

    std::vector<float> new_reference(fixture.reference.size());
    if (output_ok &&
        ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr,
                "cache-fill-invalidate: CPU reference compute failed\n");
        output_ok = false;
    }
    if (output_ok) {
        ggml_backend_tensor_get(
                fixture.graph.out, new_reference.data(), 0,
                new_reference.size() * sizeof(float));
        float max_change = 0.0f;
        for (size_t index = 0; index < new_reference.size(); index++) {
            max_change = std::max(
                    max_change,
                    std::abs(new_reference[index] - fixture.reference[index]));
        }
        if (max_change < 0.01f) {
            fprintf(stderr,
                    "cache-fill-invalidate: mutation did not change output\n");
            output_ok = false;
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            if (!compute_matches(
                    "cache-fill-invalidate", scheduler, fixture.graph,
                    new_reference, step)) {
                output_ok = false;
                break;
            }
            if (has_positive_field(capture.get(), "hits=")) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr,
                "cache-fill-invalidate: cache was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-fill-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

// fused_plan + fused_dispatch; NULL (and no hits) when either fails.
static void * fused_begin(
        const ggml_moe_cache_tensor_desc * up,
        const ggml_moe_cache_tensor_desc * gate,
        const ggml_moe_cache_tensor_desc * down,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max,
        const int32_t * ids, int n_rows, int64_t n_tokens,
        const float * const * act_rows, uint64_t * hit_mask) {
    void * node = ggml_moe_cache.fused_plan(
            up, gate, down, glu_op, up_min, up_max, gate_min, gate_max,
            ids, n_rows, n_tokens, act_rows, hit_mask);
    if (node && !ggml_moe_cache.fused_dispatch(node)) {
        ggml_moe_cache.end(node);
        *hit_mask = 0;
        return nullptr;
    }
    return node;
}

static void * create_direct_session(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    if (!ggml_moe_cache.session_create) {
        return nullptr;
    }
    void * backends[] = { cuda, cpu };
    return ggml_moe_cache.session_create(backends, 2, nullptr);
}

static bool wait_for_direct_pool(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert);
static int direct_plan_one(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert, int32_t expert);
static bool wait_for_direct_resident(ggml_tensor * weights, int32_t expert);

static bool run_profile_writer_child(
        ggml_backend_t cuda, ggml_backend_t cpu, ggml_tensor * weights,
        const std::string & path, int32_t expert,
        const std::string & ready_path, const std::string & go_path) {
    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    if (!ggml_moe_cache.query_config ||
            !ggml_moe_cache.query_config(0, 4, &config)) {
        return false;
    }
    config.profile_path = path.c_str();
    void * backends[] = { cuda, cpu };
    void * session = ggml_moe_cache.session_create(backends, 2, &config);
    if (!session) {
        return false;
    }
    {
        std::ofstream ready(ready_path);
        ready << "ready\n";
    }
    bool released = false;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (std::filesystem::exists(go_path)) {
            released = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    bool observed = false;
    if (released) {
        const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
        ggml_moe_cache.session_enter(session);
        const bool ready = wait_for_direct_pool(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
        observed = ready && direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert) >= 0;
        ggml_moe_cache.session_leave(session);
    }
    ggml_moe_cache.session_destroy(session);
    return observed;
}

static bool run_cross_process_profile_merge(const std::filesystem::path & path) {
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
    if (test_executable.empty()) {
        return false;
    }
    const std::string profile = path.string();
    const std::string ready_a = profile + ".ready-a";
    const std::string ready_b = profile + ".ready-b";
    const std::string go = profile + ".go";
    auto spawn_writer = [&](const std::string & ready, pid_t & pid) {
        const std::string expert = "7";
        char * argv[] = {
            const_cast<char *>(test_executable.c_str()),
            const_cast<char *>("--profile-writer"),
            const_cast<char *>(profile.c_str()),
            const_cast<char *>(expert.c_str()),
            const_cast<char *>(ready.c_str()),
            const_cast<char *>(go.c_str()),
            nullptr,
        };
        return posix_spawn(&pid, test_executable.c_str(), nullptr, nullptr, argv, environ) == 0;
    };

    pid_t child_a = -1;
    pid_t child_b = -1;
    const bool spawned_a = spawn_writer(ready_a, child_a);
    const bool spawned_b = spawn_writer(ready_b, child_b);
    bool both_ready = false;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (std::filesystem::exists(ready_a) && std::filesystem::exists(ready_b)) {
            both_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (both_ready) {
        std::ofstream release(go);
        release << "go\n";
    }
    int status_a = -1;
    int status_b = -1;
    if (spawned_a) {
        (void) waitpid(child_a, &status_a, 0);
    }
    if (spawned_b) {
        (void) waitpid(child_b, &status_b, 0);
    }

    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> bytes(
            (std::istreambuf_iterator<char>(input)),
            std::istreambuf_iterator<char>());
    auto u32 = [](const unsigned char * p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
            uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    };
    auto u64 = [&](const unsigned char * p) {
        return uint64_t(u32(p)) | uint64_t(u32(p + 4)) << 32;
    };
    const bool merged = spawned_a && spawned_b && both_ready &&
        WIFEXITED(status_a) && WEXITSTATUS(status_a) == 0 &&
        WIFEXITED(status_b) && WEXITSTATUS(status_b) == 0 &&
        bytes.size() == 32 + 16 && u32(bytes.data() + 8) == 2 &&
        u32(bytes.data() + 12) == 1 && u64(bytes.data() + 16) == 2 &&
        u32(bytes.data() + 32 + 12) == 2;
    std::error_code ec;
    for (const std::string & cleanup : {
            profile, profile + ".lock", ready_a, ready_b, go}) {
        std::filesystem::remove(cleanup, ec);
    }
    return merged;
#else
    (void) path;
    return true;
#endif
}

static bool run_explicit_session_config(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    if (!ggml_moe_cache.query_config ||
        !ggml_moe_cache.query_config(0, 4, &config)) {
        fprintf(stderr, "cache-explicit-config: failed to query configuration\n");
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    void * backends[] = { cuda, cpu };
    void * session = ggml_moe_cache.session_create(backends, 2, &config);
    const bool ok = session != nullptr;
    if (session) {
        ggml_moe_cache.session_destroy(session);
    }
    bool invalid_rejected = true;
    for (int overlap_cpu_rows : { -2, 9 }) {
        config.overlap_cpu_rows = overlap_cpu_rows;
        void * invalid = ggml_moe_cache.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            ggml_moe_cache.session_destroy(invalid);
        }
    }
    config.overlap_cpu_rows = 0;
    for (int expert_parallel : { -2, 9 }) {
        config.expert_parallel = expert_parallel;
        void * invalid = ggml_moe_cache.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            ggml_moe_cache.session_destroy(invalid);
        }
    }
    config.expert_parallel = 0;
    for (int automatic : { -1, 2 }) {
        config.automatic = automatic;
        void * invalid = ggml_moe_cache.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            ggml_moe_cache.session_destroy(invalid);
        }
    }
    config.automatic = 0;
    for (int min_expert_explicit : { -1, 2 }) {
        config.min_expert_explicit = min_expert_explicit;
        void * invalid = ggml_moe_cache.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            ggml_moe_cache.session_destroy(invalid);
        }
    }
    configure_cache(nullptr);
    printf("cache-explicit-config: %s\n", ok && invalid_rejected ? "OK" : "FAIL");
    return ok && invalid_rejected;
}

static bool run_single_device_automatic_session(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    if (!ggml_moe_cache.query_config ||
        !ggml_moe_cache.query_config(1, 4, &config)) {
        fprintf(stderr, "cache-single-auto: failed to query automatic configuration\n");
        return false;
    }

    // Keep the production automatic policy bit while shrinking only the test
    // capacity floors enough for a model-free CI allocation.
    config.minimum_slab_bytes = 0;
    config.min_expert_bytes = 1;
    config.min_expert_explicit = 1;
    void * backends[] = { cuda, cpu };

    config.min_devices = 2;
    void * too_many = ggml_moe_cache.session_create(backends, 2, &config);
    const bool minimum_enforced = too_many == nullptr;
    if (too_many) {
        ggml_moe_cache.session_destroy(too_many);
    }

    config.min_devices = 1;
    void * session = ggml_moe_cache.session_create(backends, 2, &config);
    if (!session) {
        fprintf(stderr, "cache-single-auto: failed to create one-device automatic session\n");
        configure_cache(nullptr);
        return false;
    }

    constexpr int64_t direct_n_in = 1024;
    constexpr int64_t direct_n_out = 64;
    constexpr int64_t direct_n_expert = 64;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> weights(expert_size * direct_n_expert);
    ggml_moe_cache.session_enter(session);
    const bool pool_ready = wait_for_direct_pool(
            "blk.0.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    configure_cache(nullptr);

    const bool ok = config.automatic == 1 && minimum_enforced && pool_ready;
    printf("cache-single-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_expert_profile_roundtrip(
        ggml_backend_t cuda, ggml_backend_t cpu,
        ggml_tensor * weights, ggml_tensor * gate_weights,
        log_capture & capture) {
    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    if (!ggml_moe_cache.query_config ||
        !ggml_moe_cache.query_config(0, 4, &config)) {
        return false;
    }

    const auto nonce = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("llama-moe-profile-test-" + std::to_string(nonce) + ".v1");
    const std::string path_string = path.string();
    config.profile_path = path_string.c_str();
    void * backends[] = { cuda, cpu };
    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    const size_t gate_expert_size = ggml_nbytes(gate_weights) / gate_weights->ne[2];

    capture.clear();
    void * writer_a = ggml_moe_cache.session_create(backends, 2, &config);
    void * writer_b = ggml_moe_cache.session_create(backends, 2, &config);
    if (!writer_a || !writer_b) {
        if (writer_a) ggml_moe_cache.session_destroy(writer_a);
        if (writer_b) ggml_moe_cache.session_destroy(writer_b);
        return false;
    }
    ggml_moe_cache.session_enter(writer_a);
    const bool ready_a = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    const int planned_a = ready_a ? direct_plan_one(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2], 3) : -1;
    ggml_moe_cache.session_leave(writer_a);

    ggml_moe_cache.session_enter(writer_b);
    const bool ready_b = wait_for_direct_pool(
            gate_weights->name, gate_weights->data, gate_expert_size,
            gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
            gate_weights->ne[2]);
    const int planned_b = ready_b ? direct_plan_one(
            gate_weights->name, gate_weights->data, gate_expert_size,
            gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
            gate_weights->ne[2], 5) : -1;
    const bool ready_b_same = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    const int planned_b_same = ready_b_same ? direct_plan_one(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2], 3) : -1;
    ggml_moe_cache.session_leave(writer_b);

    // Both sessions loaded the same initially-empty file. Sequential teardown
    // must reload under the profile lock so the second writer cannot erase the first.
    ggml_moe_cache.session_destroy(writer_a);
    ggml_moe_cache.session_destroy(writer_b);

    std::ifstream profile(path, std::ios::binary);
    std::vector<unsigned char> profile_bytes(
            (std::istreambuf_iterator<char>(profile)),
            std::istreambuf_iterator<char>());
    profile.close();
    auto profile_u32 = [](const unsigned char * p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
            uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    };
    auto profile_u64 = [&](const unsigned char * p) {
        return uint64_t(profile_u32(p)) | uint64_t(profile_u32(p + 4)) << 32;
    };
    int heat_one = 0;
    int heat_two = 0;
    if (profile_bytes.size() == 32 + 2*16) {
        for (size_t offset = 32; offset < profile_bytes.size(); offset += 16) {
            const uint32_t heat = profile_u32(profile_bytes.data() + offset + 12);
            heat_one += heat == 1;
            heat_two += heat == 2;
        }
    }
    const bool written = planned_a >= 0 && planned_b >= 0 && planned_b_same >= 0 &&
        profile_bytes.size() == 32 + 2*16 &&
        std::memcmp(profile_bytes.data(), "GGMLMHC1", 8) == 0 &&
        profile_u32(profile_bytes.data() + 8) == 2 &&
        profile_u32(profile_bytes.data() + 12) == 2 &&
        profile_u64(profile_bytes.data() + 16) == 2 &&
        heat_one == 1 && heat_two == 1 &&
        capture.get().find("saved 2 expert heat entries") != std::string::npos;

    capture.clear();
    void * reader = ggml_moe_cache.session_create(backends, 2, &config);
    bool seeded = false;
    const bool loaded = reader &&
        capture.get().find("loaded 2 expert heat entries") != std::string::npos;
    if (reader) {
        ggml_moe_cache.session_enter(reader);
        const bool reader_ready_a = wait_for_direct_pool(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
        const bool reader_ready_b = wait_for_direct_pool(
                gate_weights->name, gate_weights->data, gate_expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2]);
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type, weights->ne[2], 0);
        (void) direct_plan_one(
                gate_weights->name, gate_weights->data, gate_expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2], 1);
        const bool unrelated_a = wait_for_direct_resident(weights, 0);
        const bool unrelated_b = wait_for_direct_resident(gate_weights, 1);
        const bool hot_a = direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type, weights->ne[2], 3) == 1;
        const bool hot_b = direct_plan_one(
                gate_weights->name, gate_weights->data, gate_expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2], 5) == 1;
        seeded = reader_ready_a && reader_ready_b && unrelated_a && unrelated_b && hot_a && hot_b;
        ggml_moe_cache.session_leave(reader);
        ggml_moe_cache.session_destroy(reader);
    }

    const std::filesystem::path cross_path = path_string + ".cross";
    const bool cross_process_merged = run_cross_process_profile_merge(cross_path);

    std::vector<unsigned char> corrupt_bytes = profile_bytes;
    if (corrupt_bytes.size() > 32) {
        corrupt_bytes[32] ^= 0x01;
    }
    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt.write(
                reinterpret_cast<const char *>(corrupt_bytes.data()),
                static_cast<std::streamsize>(corrupt_bytes.size()));
    }
    capture.clear();
    void * fallback = ggml_moe_cache.session_create(backends, 2, &config);
    const bool corrupt_ignored = fallback &&
        capture.get().find("ignoring corrupt or incompatible expert profile") != std::string::npos;
    if (fallback) {
        ggml_moe_cache.session_destroy(fallback);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    configure_cache(nullptr);

    const bool ok = ready_a && ready_b && ready_b_same &&
        written && loaded && seeded && cross_process_merged && corrupt_ignored;
    printf("cache-expert-profile: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool direct_begin_ready(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    void * node = ggml_moe_cache.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return false;
    }
    ggml_moe_cache.end(node);
    return true;
}

static bool wait_for_direct_pool(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    for (int step = 0; step < 80; step++) {
        if (direct_begin_ready(
                name, base, expert_size, direct_n_in, direct_n_out,
                direct_type, direct_n_expert)) {
            return true;
        }
    }
    return false;
}

static bool run_policy_diagnostics(
        ggml_backend_t cuda, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-policy-diagnostics: failed to create session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    ggml_moe_cache.session_enter(session);
    void * first = ggml_moe_cache.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_waited = first == nullptr;
    if (first) {
        ggml_moe_cache.end(first);
    }
    void * second = ggml_moe_cache.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_completed = second != nullptr;
    if (second) {
        ggml_moe_cache.end(second);
    }
    for (int64_t n_tokens : { 2, 3 }) {
        void * oversize = ggml_moe_cache.begin(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], n_tokens, n_tokens);
        if (oversize) {
            ggml_moe_cache.end(oversize);
        }
    }
    void * excess_rows = ggml_moe_cache.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 65);
    if (excess_rows) {
        ggml_moe_cache.end(excess_rows);
    }
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const std::string log = capture.get();
    const bool configured =
        log.find("configured: mode=on devices=1 budget=4 MiB cap") != std::string::npos &&
        log.find("max-batch=1") != std::string::npos &&
        log.find("down-mmv=generic") != std::string::npos;
    const bool capacity =
        log.find("capacity: cap=4 MiB granted=4 MiB") != std::string::npos &&
        log.find("free=") != std::string::npos &&
        log.find("reserve=0 MiB") != std::string::npos;
    const bool bypass = count_occurrences(log, "above max-batch=1") == 1;
    const bool row_bypass = count_occurrences(log, "above row limit=64") == 1;
    const bool pool = count_occurrences(log, " pool[") == 1;
    const bool ok = census_waited && census_completed && configured &&
        capacity && bypass && row_bypass && pool;
    if (!ok) {
        fprintf(stderr, "cache-policy-diagnostics: expected diagnostics were not observed\n%s", log.c_str());
    }
    printf("cache-policy-diagnostics: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static int direct_plan_one(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert, int32_t expert) {
    void * node = ggml_moe_cache.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return -1;
    }
    int32_t slot = -1;
    const int hits = ggml_moe_cache.plan(node, &expert, 1, &slot);
    ggml_moe_cache.end(node);
    return hits;
}

static int direct_plan_many(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert,
        const int32_t * experts, int n_experts,
        int64_t direct_n_tokens = 1) {
    void * node = ggml_moe_cache.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, direct_n_tokens, n_experts);
    if (!node) {
        return -1;
    }
    std::vector<int32_t> slots(n_experts, -1);
    const int hits = ggml_moe_cache.plan(
            node, experts, n_experts, slots.data());
    ggml_moe_cache.end(node);
    return hits;
}

static bool wait_for_direct_resident(
        ggml_tensor * weights, int32_t expert) {
    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    (void) direct_plan_one(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], expert);
    for (int attempt = 0; attempt < 200; attempt++) {
        if (direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert) == 1) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static bool run_activation_map_regression(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights) {
    if (!ggml_moe_cache.fused_plan || !ggml_moe_cache.fused_dispatch || !ggml_moe_cache.collect ||
        !ggml_moe_cache.end) {
        fprintf(stderr, "cache-activation-map: incomplete cache API\n");
        return false;
    }

    std::vector<float> activation_data(2 * n_in);
    for (int64_t index = 0; index < n_in; index++) {
        activation_data[index] =
            0.35f * std::sin((float)index * 0.043f) -
            0.18f * std::cos((float)index * 0.071f);
        activation_data[n_in + index] =
            1.25f + 0.27f * std::sin((float)index * 0.097f);
    }
    const float * activation_a = activation_data.data();
    const float * activation_b = activation_a + n_in;
    const float * act_rows[6] = {
        activation_a, activation_a, activation_a,
        activation_b, activation_b, activation_b,
    };
    const int32_t experts[6] = { 0, 1, 2, 3, 4, 5 };

    const size_t expert_size =
        ggml_nbytes(up_weights) / up_weights->ne[2];
    const ggml_moe_cache_tensor_desc up = {
        up_weights->name, up_weights->data, expert_size,
        up_weights->ne[0], up_weights->ne[1], up_weights->ne[2],
        (int32_t)up_weights->type,
    };
    const ggml_moe_cache_tensor_desc gate = {
        gate_weights->name, gate_weights->data, expert_size,
        gate_weights->ne[0], gate_weights->ne[1], gate_weights->ne[2],
        (int32_t)gate_weights->type,
    };

    auto execute = [&](const char * dedicated_mmv,
                       std::vector<float> & output) {
        configure_cache(nullptr, "2", dedicated_mmv);
        void * session = create_direct_session(cuda, cpu);
        if (!session) {
            return false;
        }

        ggml_moe_cache.session_enter(session);
        bool ok = wait_for_direct_pool(
                up_weights->name, up_weights->data, expert_size,
                up_weights->ne[0], up_weights->ne[1], up_weights->type,
                up_weights->ne[2]) &&
            wait_for_direct_pool(
                gate_weights->name, gate_weights->data, expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2]);
        for (int32_t expert = 0; expert < 4; expert++) {
            ok &= wait_for_direct_resident(up_weights, expert);
            ok &= wait_for_direct_resident(gate_weights, expert);
        }

        uint64_t hit_mask = 0;
        void * node = ok ? fused_begin(
                &up, &gate, nullptr, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                experts, 6, 2, act_rows, &hit_mask) : nullptr;
        ok &= node != nullptr && hit_mask == UINT64_C(0x0f);
        if (node) {
            output.assign(4 * n_out,
                          std::numeric_limits<float>::quiet_NaN());
            float * rows[4] = {
                output.data(), output.data() + n_out,
                output.data() + 2 * n_out, output.data() + 3 * n_out,
            };
            ok &= ggml_moe_cache.collect(node, 4, rows, n_out) == 1;
            ggml_moe_cache.end(node);
        }
        ggml_moe_cache.session_leave(session);
        ggml_moe_cache.session_destroy(session);
        return ok;
    };

    std::vector<float> default_output;
    std::vector<float> dedicated_output;
    const bool default_ok = execute("0", default_output);
    const bool dedicated_ok = execute("1", dedicated_output);
    const bool ok = default_ok && dedicated_ok &&
        compare_output(dedicated_output, default_output, 5e-4);
    printf("cache-activation-map: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_fused_partial_invalidation(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        const uint8_t * up_row,
        const uint8_t * gate_row,
        const std::vector<float> & reference,
        log_capture & capture) {
    if (!ggml_moe_cache.fused_plan || !ggml_moe_cache.fused_dispatch || !ggml_moe_cache.collect ||
        !ggml_moe_cache.end) {
        fprintf(stderr, "cache-fused-partial: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-fused-partial: failed to create session\n");
        return false;
    }

    const size_t expert_size =
        ggml_nbytes(up_weights) / up_weights->ne[2];
    const ggml_moe_cache_tensor_desc up = {
        up_weights->name, up_weights->data, expert_size,
        up_weights->ne[0], up_weights->ne[1], up_weights->ne[2],
        (int32_t)up_weights->type,
    };
    const ggml_moe_cache_tensor_desc gate = {
        gate_weights->name, gate_weights->data, expert_size,
        gate_weights->ne[0], gate_weights->ne[1], gate_weights->ne[2],
        (int32_t)gate_weights->type,
    };
    const float * act_rows[2] = {
        (const float *)activations->data,
        (const float *)activations->data,
    };

    ggml_moe_cache.session_enter(session);
    bool pool_ready = false;
    for (int step = 0; step < 80 && !pool_ready; step++) {
        const bool up_ready = direct_begin_ready(
                up_weights->name, up_weights->data, expert_size,
                up_weights->ne[0], up_weights->ne[1], up_weights->type,
                up_weights->ne[2]);
        const bool gate_ready = direct_begin_ready(
                gate_weights->name, gate_weights->data, expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2]);
        pool_ready = up_ready && gate_ready;
    }

    bool partial_ok = pool_ready &&
        wait_for_direct_resident(up_weights, 0) &&
        wait_for_direct_resident(gate_weights, 0);

    auto execute = [&](const int32_t * experts, int n_ids,
                       uint64_t expected_mask,
                       const int * reference_rows) {
        uint64_t hit_mask = 0;
        void * node = fused_begin(
                &up, &gate, nullptr, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                experts, n_ids, 1, act_rows, &hit_mask);
        if (!node || hit_mask != expected_mask) {
            if (node) {
                ggml_moe_cache.end(node);
            }
            return false;
        }

        int n_hits = 0;
        float output[2 * n_out];
        float * rows[2] = { nullptr, nullptr };
        for (int row = 0; row < n_ids; row++) {
            if (hit_mask & (UINT64_C(1) << row)) {
                rows[n_hits] = output + n_hits * n_out;
                n_hits++;
            }
        }
        const bool collected =
            ggml_moe_cache.collect(node, n_hits, rows, n_out) == 1;
        ggml_moe_cache.end(node);
        if (!collected) {
            return false;
        }

        for (int hit = 0; hit < n_hits; hit++) {
            const int reference_row = reference_rows[hit];
            std::vector<float> expected(
                    reference.begin() + reference_row * n_out,
                    reference.begin() + (reference_row + 1) * n_out);
            std::vector<float> actual(
                    output + hit * n_out,
                    output + (hit + 1) * n_out);
            if (!compare_output(expected, actual, 5e-4)) {
                return false;
            }
        }
        return true;
    };

    const int32_t partial_ids[2] = { 0, 1 };
    const int partial_reference[1] = { 0 };
    partial_ok &= execute(
            partial_ids, 2, UINT64_C(1), partial_reference);

    const int32_t duplicate_ids[2] = { 0, 0 };
    const int duplicate_reference[2] = { 0, 0 };
    partial_ok &= execute(
            duplicate_ids, 2, UINT64_C(3), duplicate_reference);

    auto invalidated = [&](ggml_tensor * weights,
                           const uint8_t * unchanged_row) {
        ggml_backend_tensor_set(
                weights, unchanged_row, 0, expert_size);
        const int32_t expert = 0;
        uint64_t hit_mask = 0;
        void * node = fused_begin(
                &up, &gate, nullptr, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &hit_mask);
        if (node) {
            ggml_moe_cache.end(node);
            return false;
        }
        if (hit_mask != 0 || !wait_for_direct_resident(weights, expert)) {
            return false;
        }
        const int reference_row[1] = { 0 };
        return execute(&expert, 1, UINT64_C(1), reference_row);
    };

    const bool up_invalidation_ok = partial_ok &&
        invalidated(up_weights, up_row);
    const bool gate_invalidation_ok = up_invalidation_ok &&
        invalidated(gate_weights, gate_row);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const bool stats_ok = has_positive_field(
            capture.get(), "fusion-nodes=");
    printf("cache-fused-partial-duplicate: %s\n",
            partial_ok ? "OK" : "FAIL");
    printf("cache-fused-up-invalidate: %s\n",
            up_invalidation_ok ? "OK" : "FAIL");
    printf("cache-fused-gate-invalidate: %s\n",
            gate_invalidation_ok && stats_ok ? "OK" : "FAIL");
    return gate_invalidation_ok && stats_ok;
}

static bool run_fused_full_ffn(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * down_weights,
        ggml_tensor * activations,
        const uint8_t * down_row) {
    configure_cache(nullptr);
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-fused-full-ffn: failed to create session\n");
        return false;
    }

    auto desc = [](ggml_tensor * weights) {
        return ggml_moe_cache_tensor_desc {
            weights->name, weights->data,
            ggml_nbytes(weights) / (size_t) weights->ne[2],
            weights->ne[0], weights->ne[1], weights->ne[2],
            (int32_t) weights->type,
        };
    };
    const ggml_moe_cache_tensor_desc up = desc(up_weights);
    const ggml_moe_cache_tensor_desc gate = desc(gate_weights);
    const ggml_moe_cache_tensor_desc down = desc(down_weights);
    const int32_t expert = 0;
    const float * act_rows[] = { (const float *) activations->data };
    bool ok = true;

    ggml_moe_cache.session_enter(session);
    for (ggml_tensor * weights : { up_weights, gate_weights, down_weights }) {
        ok &= wait_for_direct_pool(
                weights->name, weights->data,
                ggml_nbytes(weights) / (size_t) weights->ne[2],
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2]);
        ok &= wait_for_direct_resident(weights, expert);
    }

    float intermediate[n_out] = {};
    float reference[n_in] = {};
    if (ok) {
        uint64_t pair_mask = 0;
        void * pair = fused_begin(
                &up, &gate, nullptr, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &pair_mask);
        float * pair_rows[] = { intermediate };
        ok = pair && pair_mask == UINT64_C(1) &&
            ggml_moe_cache.collect(pair, 1, pair_rows, n_out) == 1;
        if (pair) {
            ggml_moe_cache.end(pair);
        }
    }
    if (ok) {
        void * down_node = ggml_moe_cache.begin(
                down.name, down.data, down.expert_size,
                down.n_in, down.n_out, down.type, down.n_expert, 1, 1);
        int32_t slot = -1;
        ok = down_node &&
            ggml_moe_cache.plan(down_node, &expert, 1, &slot) == 1;
        const float * down_acts[] = { intermediate };
        if (ok) {
            ok = ggml_moe_cache.dispatch(
                    down_node, down.type, down.n_in, down.n_out,
                    1, &slot, down_acts) == 1;
        }
        float * down_rows[] = { reference };
        if (ok) {
            ok = ggml_moe_cache.collect(
                    down_node, 1, down_rows, down.n_out) == 1;
        }
        if (down_node) {
            ggml_moe_cache.end(down_node);
        }
    }

    auto execute_full = [&](float * output) {
        uint64_t full_mask = 0;
        auto plan = [&]() {
            return ggml_moe_cache.fused_plan(
                &up, &gate, &down, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &full_mask);
        };
        // Cancelling a plan must release pins/locks without requiring a launch.
        void * cancelled = plan();
        if (!cancelled) return false;
        ggml_moe_cache.end(cancelled);
        void * full = plan();
        float * full_rows[] = { output };
        const bool result = full && full_mask == UINT64_C(1) &&
            ggml_moe_cache.collect(full, 1, full_rows, n_in) == 0 &&
            ggml_moe_cache.fused_dispatch(full) == 1 &&
            ggml_moe_cache.fused_dispatch(full) == 1 &&
            ggml_moe_cache.collect(full, 1, full_rows, n_in) == 1;
        if (full) {
            ggml_moe_cache.end(full);
        }
        return result;
    };

    float actual[n_in] = {};
    if (ok) {
        ok = execute_full(actual);
    }

    bool down_invalidation_ok = false;
    if (ok) {
        uint64_t full_mask = 0;
        void * full = fused_begin(
                &up, &gate, &down, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &full_mask);
        std::atomic<bool> mutation_started{false};
        std::atomic<bool> mutation_done{false};
        std::thread mutator;
        if (full && full_mask == UINT64_C(1)) {
            mutator = std::thread([&] {
                mutation_started.store(true);
                ggml_backend_tensor_set(
                        down_weights, down_row, 0, down.expert_size);
                mutation_done.store(true);
            });
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!mutation_started.load() &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const bool mutation_blocked =
            mutation_started.load() && !mutation_done.load();
        float pinned_output[n_in] = {};
        float * full_rows[] = { pinned_output };
        const bool collected = full &&
            ggml_moe_cache.collect(full, 1, full_rows, n_in) == 1;
        if (full) {
            ggml_moe_cache.end(full);
        }
        if (mutator.joinable()) {
            mutator.join();
        }

        uint64_t invalid_mask = 0;
        void * invalid = fused_begin(
                &up, &gate, &down, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &invalid_mask);
        if (invalid) {
            ggml_moe_cache.end(invalid);
        }
        const bool invalidated = !invalid && invalid_mask == 0;
        const bool repopulated = invalidated &&
            wait_for_direct_resident(down_weights, expert);
        float repopulated_output[n_in] = {};
        const bool recomputed = repopulated &&
            execute_full(repopulated_output);
        down_invalidation_ok = mutation_blocked && collected &&
            mutation_done.load() && recomputed &&
            memcmp(actual, pinned_output, sizeof(actual)) == 0 &&
            memcmp(actual, repopulated_output, sizeof(actual)) == 0;
    }

    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    if (ok && memcmp(reference, actual, sizeof(reference)) != 0) {
        float max_error = 0.0f;
        for (int index = 0; index < n_in; index++) {
            max_error = std::max(max_error,
                    std::abs(reference[index] - actual[index]));
        }
        fprintf(stderr,
                "cache-fused-full-ffn: output mismatch (max error %.9g)\n",
                max_error);
        ok = false;
    }
    printf("cache-fused-full-ffn: %s\n", ok ? "OK" : "FAIL");
    printf("cache-fused-full-ffn-down-invalidate: %s\n",
            down_invalidation_ok ? "OK" : "FAIL");
    return ok && down_invalidation_ok;
}

static bool run_cpu_overlap_policy(
        ggml_backend_t cuda, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "1");
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    ggml_moe_cache.session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    const int32_t experts[] = { 0, 1 };
    for (int index = 0; index < 2 && ok; index++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts[index]);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], experts[index]) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts, 2) == 1;
    }
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_adaptive_cpu_overlap_policy(
        ggml_backend_t cuda, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "10");
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap-auto: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    ggml_moe_cache.session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    for (int32_t expert = 0; expert < 8 && ok; expert++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], expert) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }

    const int32_t single_token_experts[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], single_token_experts, 8) == 6;
    }

    int32_t multi_token_experts[multi_n_used * multi_n_tokens];
    for (int token = 0; token < multi_n_tokens; token++) {
        for (int id = 0; id < multi_n_used; id++) {
            multi_token_experts[token * multi_n_used + id] = id;
        }
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], multi_token_experts,
                multi_n_used * multi_n_tokens, multi_n_tokens) == 50;
    }
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    ok &= capture.get().find("cpu-overlap=auto") != std::string::npos;
    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_scope_isolation(
        ggml_backend_t cuda, ggml_backend_t cpu, ggml_tensor * weights) {
    if (!ggml_moe_cache.session_enter || !ggml_moe_cache.session_leave ||
        !ggml_moe_cache.begin || !ggml_moe_cache.end ||
        !ggml_moe_cache.invalidate || !ggml_moe_cache.session_destroy) {
        fprintf(stderr, "cache-scope: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    void * outer = create_direct_session(cuda, cpu);
    if (!outer) {
        fprintf(stderr, "cache-scope: failed to create outer session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    ggml_moe_cache.session_enter(outer);
    const bool warmed = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(outer);

    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "1048576");
    void * dormant = create_direct_session(cuda, cpu);
    if (!dormant) {
        fprintf(stderr, "cache-scope: failed to create dormant session\n");
        ggml_moe_cache.session_destroy(outer);
        return false;
    }
    ggml_moe_cache.session_enter(dormant);
    const bool dormant_begin = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(dormant);

    ggml_moe_cache.session_enter(outer);
    const bool outer_before = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    ggml_moe_cache.session_enter(nullptr);
    const bool null_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(nullptr);
    const bool outer_after_null = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    ggml_moe_cache.session_enter(dormant);
    const bool dormant_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(dormant);
    const bool outer_after_dormant = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(outer);

    ggml_moe_cache.session_destroy(dormant);
    ggml_moe_cache.session_destroy(outer);
    const bool ok = warmed && !dormant_begin && outer_before &&
        !null_leaked && outer_after_null &&
        !dormant_leaked && outer_after_dormant;
    printf("cache-scope: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shape_liveness(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-liveness: failed to create session\n");
        return false;
    }

    constexpr int64_t shape_a_in = 256;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_out = 128;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    ggml_moe_cache.session_enter(session);
    (void)direct_begin_ready(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.invalidate(shape_a.data(), shape_a.size());
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.3.ffn_up_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const bool ok = shape_b_ready && shape_a_ready;
    printf("cache-shape-liveness: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_exact_shape_inventory(
        ggml_backend_t cuda, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_LFU", "0");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-inventory: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t first_n_expert = 16;
    constexpr int64_t second_n_expert = 49;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> first(expert_size * first_n_expert);
    std::vector<uint8_t> second(expert_size * second_n_expert);
    std::vector<uint8_t> late(expert_size);

    ggml_moe_cache.session_enter(session);
    (void)direct_begin_ready(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
    (void)direct_begin_ready(
            "blk.8.ffn_up_exps.weight", second.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    bool ready = false;
    for (int step = 0; step < 40 && !ready; step++) {
        ready |= direct_begin_ready(
                "blk.7.ffn_up_exps.weight", first.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
        ready |= direct_begin_ready(
                "blk.8.ffn_up_exps.weight", second.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    }
    const bool complete_first_miss = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool complete_second_hit = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 1;
    const bool late_first_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool late_second_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    bool late_hit = false;
    for (int attempt = 0; attempt < 100 && !late_hit; attempt++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        late_hit = direct_plan_one(
                "blk.9.ffn_up_exps.weight", late.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 1;
    }
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const std::string log = capture.get();
    const bool ok = ready &&
        max_field_value(log, "slots=") == first_n_expert + second_n_expert &&
        log.find("entries=65 coverage=complete") != std::string::npos &&
        log.find("admit=1-complete/2-partial/8-replace") != std::string::npos &&
        complete_first_miss && complete_second_hit &&
        late_first_miss && late_second_miss && late_hit &&
        log.find("fills=serial") != std::string::npos;
    if (!ok) {
        fprintf(stderr, "cache-shape-inventory: unexpected pool inventory\n%s",
                log.c_str());
    }
    printf("cache-shape-inventory: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_complete_pool_allocation(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-complete-pools: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 512;
    constexpr int64_t common_n_out = 64;
    constexpr int64_t rare_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t common_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * common_n_out;
    const size_t rare_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * rare_n_out;
    std::vector<std::vector<uint8_t>> common_tensors(
            8, std::vector<uint8_t>(common_expert * direct_n_expert));
    std::vector<uint8_t> rare_tensor(rare_expert * direct_n_expert);

    ggml_moe_cache.session_enter(session);
    for (const auto & tensor : common_tensors) {
        (void)direct_begin_ready(
                "blk.9.ffn_up_exps.weight", tensor.data(), common_expert,
                direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    }
    (void)direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);

    const bool common_ready = wait_for_direct_pool(
            "blk.9.ffn_up_exps.weight", common_tensors[0].data(), common_expert,
            direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    const bool rare_ready = direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const bool ok = common_ready && rare_ready;
    printf("cache-complete-pools: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shared_budget(
        ggml_backend_t cuda, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(cuda->device, &free_bytes, &total_bytes);
    const size_t free_mib = free_bytes >> 20;
    if (free_mib < 128) {
        printf("cache-shared-budget: SKIP (insufficient free VRAM)\n");
        return true;
    }

    // Leave enough headroom that a few MiB of driver/display allocation churn
    // does not look like a broken claim split on a display-attached GPU.
    const std::string reserve = std::to_string(free_mib - 64);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    capture.clear();

    void * first = create_direct_session(cuda, cpu);
    void * second = create_direct_session(cuda, cpu);
    if (!first || !second) {
        fprintf(stderr, "cache-shared-budget: failed to create sessions\n");
        if (first) {
            ggml_moe_cache.session_destroy(first);
        }
        if (second) {
            ggml_moe_cache.session_destroy(second);
        }
        configure_cache(nullptr);
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> weights(expert_size * direct_n_expert);

    ggml_moe_cache.session_enter(first);
    const bool first_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    ggml_moe_cache.session_leave(first);

    ggml_moe_cache.session_enter(second);
    const bool second_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    ggml_moe_cache.session_leave(second);

    ggml_moe_cache.session_destroy(second);
    ggml_moe_cache.session_destroy(first);

    const std::string shared_log = capture.get();
    const bool divided = count_field_at_least(shared_log, "granted=", 24) == 2 &&
        max_field_value(shared_log, "granted=") <= 40;

    capture.clear();
    void * replacement = create_direct_session(cuda, cpu);
    bool replacement_ready = false;
    if (replacement) {
        ggml_moe_cache.session_enter(replacement);
        replacement_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        ggml_moe_cache.session_leave(replacement);
        ggml_moe_cache.session_destroy(replacement);
    }
    const std::string replacement_log = capture.get();
    const bool released = max_field_value(replacement_log, "granted=") >= 56;

    capture.clear();
    const std::string high_reserve = std::to_string(free_mib - 16);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", high_reserve.c_str());
    void * high = create_direct_session(cuda, cpu);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    void * low = create_direct_session(cuda, cpu);
    if (high) {
        ggml_moe_cache.session_destroy(high);
    }
    void * after_high = create_direct_session(cuda, cpu);

    bool low_ready = false;
    if (low) {
        ggml_moe_cache.session_enter(low);
        low_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        ggml_moe_cache.session_leave(low);
    }
    bool after_high_ready = false;
    if (after_high) {
        ggml_moe_cache.session_enter(after_high);
        after_high_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        ggml_moe_cache.session_leave(after_high);
    }
    if (after_high) {
        ggml_moe_cache.session_destroy(after_high);
    }
    if (low) {
        ggml_moe_cache.session_destroy(low);
    }
    const std::string reserve_log = capture.get();
    const bool reserve_lowered = high && low && after_high &&
        low_ready && after_high_ready &&
        count_field_at_least(reserve_log, "granted=", 8) == 2;
    configure_cache(nullptr);

    const bool ok = first_ready && second_ready && divided &&
        replacement_ready && released && reserve_lowered;
    if (!ok) {
        fprintf(stderr, "cache-shared-budget: unexpected claim behavior\n%s%s%s",
                shared_log.c_str(), replacement_log.c_str(), reserve_log.c_str());
    }
    printf("cache-shared-budget: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_route_override(
        ggml_backend_dev_t first_device,
        ggml_backend_t cuda, ggml_backend_t cpu,
        log_capture & capture) {
    ggml_backend_dev_t second_device =
        find_other_cuda_device(first_device);
    if (!second_device) {
        printf("cache-route-override: SKIP (one CUDA device)\n");
        return true;
    }
    const long first_physical = cuda_physical_device(first_device);
    if (first_physical >= 0 &&
        first_physical == cuda_physical_device(second_device)) {
        printf("cache-route-override: SKIP (one physical CUDA device)\n");
        return true;
    }
    ggml_backend_t second_cuda =
        ggml_backend_dev_init(second_device, nullptr);
    if (!second_cuda) {
        fprintf(stderr,
                "cache-route-override: failed to initialize second device\n");
        return false;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "2");
    capture.clear();
    void * backends[] = { cuda, second_cuda, cpu };
    void * session = ggml_moe_cache.session_create(backends, 3, nullptr);
    if (!session) {
        fprintf(stderr,
                "cache-route-override: failed to create session\n");
        ggml_backend_free(second_cuda);
        return false;
    }

    constexpr int64_t shape_a_in = 1024;
    constexpr int64_t shape_a_out = 160;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_b_out = 384;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_a_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_b_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    ggml_moe_cache.session_enter(session);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.4.ffn_down_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_b_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.session_leave(session);

    ggml_moe_cache_config config = {};
    ggml_moe_cache_device_caps first_caps = {};
    ggml_moe_cache_device_caps second_caps = {};
    const bool queried = ggml_moe_cache.query_config(0, 8, &config) &&
        ggml_moe_cache.query_device(first_device, &config, &first_caps) &&
        ggml_moe_cache.query_device(second_device, &config, &second_caps);
    const bool expect_parallel = queried && first_caps.compute_capability >= 800 &&
        second_caps.compute_capability >= 800;
    ggml_moe_cache.session_destroy(session);
    const bool fill_policy = capture.get().find(
            expect_parallel ? "fills=parallel" : "fills=serial") != std::string::npos;

    ggml_moe_cache_config automatic = {};
    capture.clear();
    const bool automatic_queried =
        ggml_moe_cache.query_config(1, 4, &automatic);
    void * dormant = automatic_queried
        ? ggml_moe_cache.session_create(backends, 3, &automatic) : nullptr;
    const bool dormant_created = dormant != nullptr;
    bool dormant_begin = false;
    if (dormant) {
        ggml_moe_cache.session_enter(dormant);
        dormant_begin = direct_begin_ready(
                "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
                shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
        ggml_moe_cache.session_leave(dormant);
        ggml_moe_cache.session_destroy(dormant);
    }
    const bool slab_floor = dormant_created && !dormant_begin &&
        capture.get().find("automatic slab floor") != std::string::npos;
    ggml_backend_free(second_cuda);

    const bool ok = shape_a_ready && shape_b_ready && queried && fill_policy &&
        automatic_queried && slab_floor;
    printf("cache-route-override: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_admission_policy_once(
        ggml_backend_t cuda, ggml_backend_t cpu,
        int new_expert_misses, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "2");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_LFU", "0"); // pins the legacy readmit throttle
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        return false;
    }

    constexpr int64_t policy_n_in = 1024;
    constexpr int64_t policy_n_out = 206;
    constexpr int64_t policy_n_expert = 65;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, policy_n_in) * policy_n_out;
    std::vector<uint8_t> weights(expert_size * policy_n_expert);
    const char * name = "blk.5.ffn_up_exps.weight";

    ggml_moe_cache.session_enter(session);
    bool ok = wait_for_direct_pool(
            name, weights.data(), expert_size,
            policy_n_in, policy_n_out,
            GGML_TYPE_Q4_0, policy_n_expert);
    for (int32_t expert = 0; expert < 64 && ok; expert++) {
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }
        if (expert == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }

        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, expert) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    for (int miss = 0; miss < new_expert_misses && ok; miss++) {
        ok &= direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, 64) == 0;
    }
    if (new_expert_misses == 8 && ok) {
        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, 64) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    const std::string log = capture.get();
    const long long expected_enqueued =
        new_expert_misses == 8 ? 65 : 64;
    const long long expected_evictions =
        new_expert_misses == 8 ? 1 : 0;
    return ok &&
        max_field_value(log, "slots=") == 64 &&
        max_field_value(log, "enqueued=") == expected_enqueued &&
        max_field_value(log, "evictions=") == expected_evictions;
}

static bool run_admission_policy(
        ggml_backend_t cuda, ggml_backend_t cpu,
        log_capture & capture) {
    const bool seven = run_admission_policy_once(
            cuda, cpu, 7, capture);
    const bool eight = run_admission_policy_once(
            cuda, cpu, 8, capture);
    const bool ok = seven && eight;
    printf("cache-admission-policy: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

static bool run_scheduler_overlap_config(ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "3");
    static ggml_moe_cache_config captured;
    static bool supplied;
    ggml_backend_t backends[] = {cuda, cpu};
    auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
    // Backend initialization may refresh the provider's function table.
    // Observe explicit scheduler configuration only after construction.
    auto saved_create = ggml_moe_cache.session_create;
    ggml_moe_cache.session_create = [](void * const *, int, const ggml_moe_cache_config * config) -> void * {
        if (config) {
            captured = *config;
            supplied = true;
        }
        return nullptr;
    };
    bool ok = true;
    for (auto mode : {GGML_MOE_CACHE_MODE_UNSPECIFIED, GGML_MOE_CACHE_MODE_AUTO}) {
        for (int overlap : {-2, -1, 0, 2}) {
            supplied = false;
            ggml_backend_sched_set_moe_cache(scheduler, mode, 4, 0, 0, overlap, nullptr);
            const bool expected_call = mode != GGML_MOE_CACHE_MODE_UNSPECIFIED || overlap != -2;
            const bool matches = supplied == expected_call &&
                (!supplied || captured.overlap_cpu_rows == (overlap == -2 ? 3 : overlap));
            if (!matches) fprintf(stderr, "scheduler overlap mode=%d requested=%d supplied=%d actual=%d\n",
                int(mode), overlap, supplied, captured.overlap_cpu_rows);
            ok &= matches;
        }
    }
    // A device limit is an override even under the provider policy.
    supplied = false;
    ggml_backend_sched_set_moe_cache(scheduler, GGML_MOE_CACHE_MODE_UNSPECIFIED, 4, 1, 0, -2, nullptr);
    ok &= supplied && captured.max_devices == 1;
    supplied = false;
    ggml_backend_sched_set_moe_cache(scheduler, GGML_MOE_CACHE_MODE_OFF, 4, 0, 0, 2, nullptr);
    ok &= !supplied;
    ggml_backend_sched_free(scheduler);
    ggml_moe_cache.session_create = saved_create;
    configure_cache(nullptr);
    printf("cache-scheduler-overlap-config: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_pool_limits() {
    bool ok = true;
    const size_t ordinary_min = 1024u << 10;
    ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_EXL3_2, 0, ordinary_min) == (128u << 10);
    ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_EXL3_2, 1, ordinary_min) == ordinary_min;
    ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q4_0, 0, ordinary_min) == ordinary_min;
    for (size_t minimum : { 512u << 10, 1024u << 10 }) {
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q2_0, 0, minimum) == minimum / 2;
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q2_0, 1, minimum) == minimum;
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q2_0_G128, 0, minimum) == minimum * 34 / 72;
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q2_0_G128, 1, minimum) == minimum;
        // A same-shape IQ2 expert holds as many weights as the Q4_0 floor in fewer bytes.
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_IQ2_XXS, 0, minimum) == minimum * 66 / 144;
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_IQ2_XS, 0, minimum) == minimum * 74 / 144;
        ok &= ggml_moe_cache_effective_min_expert_bytes(GGML_TYPE_Q8_0, 0, minimum) == minimum;
    }
    const size_t expert_bytes = 400u << 10;
    const size_t exl3_limit = ggml_moe_cache_max_pool_slots(GGML_TYPE_EXL3_2, expert_bytes);
    ok &= exl3_limit == std::min(size_t(INT_MAX), SIZE_MAX / expert_bytes);
    // 20971 was the erroneous EXL3 limit: four-byte storage blocks, despite
    // the dedicated kernel addressing the complete pool with byte offsets.
    if (SIZE_MAX > UINT32_MAX) ok &= exl3_limit > 73728;
    const size_t q4_bytes = ggml_row_size(GGML_TYPE_Q4_0, 2560) * 640;
    ok &= ggml_moe_cache_max_pool_slots(GGML_TYPE_Q4_0, q4_bytes) ==
        std::min(SIZE_MAX / q4_bytes, size_t(INT_MAX) / (q4_bytes / ggml_type_size(GGML_TYPE_Q4_0)));
    ok &= ggml_moe_cache_max_pool_slots(GGML_TYPE_EXL3_2, 0) == 0;
    ok &= ggml_moe_cache_max_pool_slots(GGML_TYPE_EXL3_2, expert_bytes + 1) == 0;
    printf("cache-pool-limits: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Compare transient experts with fully resident GPU execution, including
// cancellation, source invalidation and failures before output publication.
static bool run_stream_staging_case(ggml_backend_dev_t dev,
                                    ggml_backend_t     gpu,
                                    ggml_backend_t     cpu,
                                    log_capture &      capture,
                                    int                T,
                                    const char *       dedicated_down,
                                    bool               boundary) {
    constexpr int K = 2560, F = 640;
    const int     E    = boundary ? 32 : 64;
    constexpr int topk = 10;
    const int     R    = T * topk;
    GGML_ASSERT(T >= 2 && T <= 4);
    uint64_t resident = 0;
    for (int t = 0; t < T; ++t) {
        resident |= 1ULL << (t * topk + 1);
    }
    const float inf = std::numeric_limits<float>::infinity();
    bool        ok  = true;
    for (bool pinned : { true, false }) {
        auto ctx   = ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true });
        auto up    = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, K, F, E);
        auto gate  = ggml_dup_tensor(ctx, up);
        auto down  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, F, K, E);
        auto dummy = boundary ? ggml_dup_tensor(ctx, up) : nullptr;
        ggml_set_name(up, "blk.0.ffn_up_exps.weight");
        ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
        ggml_set_name(down, "blk.0.ffn_down_exps.weight");
        if (dummy) {
            ggml_set_name(dummy, "blk.1.ffn_up_exps.weight");
        }
        auto buft = pinned ? ggml_backend_dev_host_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        GGML_ASSERT(buft);
        auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        int seed = 0;
        for (auto w : { up, gate, down }) {
            std::vector<float>   values(K * F);
            std::vector<uint8_t> bytes(ggml_nbytes(w) / E);
            for (int e = 0; e < E; ++e) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = .012f * std::sin(float(i + 31 * e + 71 * seed) * .017f);
                }
                GGML_ASSERT(ggml_quantize_chunk(w->type, values.data(), bytes.data(), 0, w->ne[1], w->ne[0], nullptr) ==
                            bytes.size());
                ggml_backend_tensor_set(w, bytes.data(), e * bytes.size(), bytes.size());
            }
            ++seed;
        }
        auto desc = [&](ggml_tensor * w) {
            return ggml_moe_cache_tensor_desc{ w->name,  w->data, ggml_nbytes(w) / E, w->ne[0],
                                               w->ne[1], E,       (int32_t) w->type };
        };
        auto                       ud = desc(up), gd = desc(gate), dd = desc(down);
        std::vector<float>         acts(K * T), ref(R * K), out(R * K);
        std::vector<const float *> rows(R);
        std::vector<int32_t>       ids(R);
        for (size_t i = 0; i < acts.size(); ++i) {
            acts[i] = std::sin(float(i) * .037f);
        }
        for (int i = 0; i < R; ++i) {
            rows[i] = acts.data() + (i / topk) * K;
        }
        auto routes = [&](int e1) {
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < topk; ++j) {
                    ids[t * topk + j] = j == 1 ? 0 : (j == 0 ? e1 : e1 + j - 1);
                }
            }
        };
        auto start = [&](const char * fail = nullptr, const char * profile = nullptr) {
            configure_cache(fail, "4", "1", boundary ? "46" : "40", dedicated_down);
            ggml_moe_cache_config config = {};
            GGML_ASSERT(ggml_moe_cache.query_config(0, 0, &config));
            config.profile_path = profile;
            void * backends[] = { gpu, cpu };
            void * s = ggml_moe_cache.session_create(backends, 2, &config);
            GGML_ASSERT(s);
            ggml_moe_cache.session_enter(s);
            if (dummy) {
                (void) direct_begin_ready(dummy->name, dummy->data, ggml_nbytes(dummy) / E, dummy->ne[0], dummy->ne[1],
                                          dummy->type, E);
            }
            for (auto w : { up, gate, down }) {
                (void) direct_begin_ready(w->name, w->data, ggml_nbytes(w) / E, w->ne[0], w->ne[1], w->type, E);
            }
            for (auto w : { up, gate, down }) {
                GGML_ASSERT(wait_for_direct_pool(w->name, w->data, ggml_nbytes(w) / E, w->ne[0], w->ne[1], w->type, E));
            }
            return s;
        };
        auto warm = [&](int e) {
            for (auto w : { up, gate, down }) {
                GGML_ASSERT(wait_for_direct_resident(w, e));
            }
        };
        auto plan = [&](uint64_t & mask) {
            return ggml_moe_cache.fused_plan(&ud, &gd, &dd, GGML_GLU_OP_SWIGLU, -inf, inf, -inf, inf, ids.data(), R, T,
                                             rows.data(), &mask);
        };
        auto collect = [&](void * node, uint64_t mask, std::vector<float> & result) {
            std::vector<float *> outs(R);
            int                  n = 0;
            for (int i = 0; i < R; ++i) {
                if (mask & (1ULL << i)) {
                    outs[n++] = result.data() + K * i;
                }
            }
            return node && ggml_moe_cache.fused_dispatch(node) && ggml_moe_cache.collect(node, n, outs.data(), K);
        };
        auto stop = [](void * s) {
            ggml_moe_cache.session_leave(s);
            ggml_moe_cache.session_destroy(s);
        };
        auto counter = [&](const char * field) {
            const auto text = capture.get();
            const auto pos  = text.rfind(field);
            GGML_ASSERT(pos != std::string::npos);
            return strtoll(text.c_str() + pos + strlen(field), nullptr, 10);
        };
        for (int round = 0; round < 3; ++round) {
            routes(1 + 10 * round);
            auto s = start();
            for (int j = 0; j < topk; ++j) {
                warm(ids[j]);
            }
            uint64_t mask = 0;
            auto     node = plan(mask);
            GGML_ASSERT(mask == ((1ULL << R) - 1) && collect(node, mask, ref));
            ggml_moe_cache.end(node);
            stop(s);
            if (round == 0 && ggml_moe_cache.output_supported) {
                auto hc = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
                auto dc = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
                auto source = ggml_new_tensor_2d(hc, GGML_TYPE_F32, K, R);
                source->op = GGML_OP_MUL_MAT_ID;
                auto destination = ggml_dup_tensor(dc, source);
                auto hb = ggml_backend_alloc_ctx_tensors_from_buft(hc, buft);
                auto db = ggml_backend_alloc_ctx_tensors(dc, gpu);
                GGML_ASSERT(hb && db);
                ggml_backend_buffer_set_usage(db, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (ggml_moe_cache.output_supported(gpu, source)) {
                    // Scatter a subset into nonconsecutive destination rows;
                    // untouched rows model completed CPU misses. Poison host
                    // hit rows so an accidental whole upload cannot pass.
                    for (int pattern : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
                        s = start(pattern == 4 ? "collect" : pattern == 5 ? "dispatch" :
                                pattern == 6 ? "collect-retire" : nullptr);
                        for (int j = 0; j < topk; ++j) warm(ids[j]);
                        ggml_backend_tensor_set(source, ref.data(), 0, ggml_nbytes(source));
                        std::vector<float> poison(ref.size(), -12345.f), actual(ref.size());
                        ggml_backend_tensor_set_async(gpu, destination, poison.data(), 0, ggml_nbytes(destination));
                        ggml_backend_synchronize(gpu);
                        const bool bound = ggml_moe_cache.output_bind(s, gpu, source, destination);
                        bool pass = bound;
                        std::vector<int32_t> selected_ids;
                        std::vector<const float *> selected_acts;
                        std::vector<float *> selected_outputs;
                        for (int r = 0; r < R; ++r) {
                            if (pattern == 0 || (pattern == 1 && r % 3 != 1) || (pattern == 2 && r != R-1)) continue;
                            selected_ids.push_back(ids[r]);
                            selected_acts.push_back(rows[r]);
                            selected_outputs.push_back((float *) source->data + r*K);
                            std::fill_n(selected_outputs.back(), K, -23456.f);
                        }
                        bool collected = false, dispatched = false;
                        uint64_t selected_mask = 0;
                        if (!selected_ids.empty()) {
                            auto job = ggml_moe_cache.fused_plan(&ud, &gd, &dd, GGML_GLU_OP_SWIGLU,
                                    -inf, inf, -inf, inf, selected_ids.data(), selected_ids.size(), 1,
                                    selected_acts.data(), &selected_mask);
                            pass &= job && selected_mask == ((UINT64_C(1) << selected_ids.size()) - 1);
                            dispatched = job && ggml_moe_cache.fused_dispatch(job);
                            collected = dispatched && ggml_moe_cache.collect(job, selected_ids.size(), selected_outputs.data(), K);
                            if (job) ggml_moe_cache.end(job);
                            pass &= collected == (pattern < 4 || pattern >= 6);
                        }
                        if (pattern == 4 || pattern == 5) {
                            // Existing CPU failure retry fills all host rows.
                            ggml_backend_tensor_set(source, ref.data(), 0, ggml_nbytes(source));
                        }
                        if (pattern == 7) {
                            // Retained source lease must drain before invalidation.
                            ggml_moe_cache.invalidate(up->data, ggml_nbytes(up));
                        }
                        if (pattern == 8) {
                            // A nested scope cannot strand a previous owner.
                            ggml_moe_cache.session_enter(nullptr);
                            ggml_moe_cache.session_leave(nullptr);
                        }
                        if (pattern == 9) {
                            // Abandon the scheduler scope before its copy door.
                            // Shutdown must still retire leases and slot readers.
                            ggml_moe_cache.session_leave(s);
                        }
                        std::atomic<bool> invalidating{false}, invalidated{false};
                        std::thread invalidator;
                        if (pattern == 10 && collected) {
                            // The parked composite must retain its source lease,
                            // then retire without acquiring the registry mutex
                            // held by a concurrent invalidator waiting on it.
                            invalidator = std::thread([&] {
                                invalidating.store(true);
                                ggml_moe_cache.invalidate(up->data, ggml_nbytes(up));
                                invalidated.store(true);
                            });
                            while (!invalidating.load()) std::this_thread::yield();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            pass &= !invalidated.load();
                        }
                        const int copied = ggml_moe_cache.output_copy(s, gpu, source, destination);
                        if (invalidator.joinable()) {
                            invalidator.join();
                            pass &= invalidated.load();
                        }
                        pass &= copied == (pattern == 6 ? -1 : pattern == 9 ? 0 : int(collected));
                        if (copied == 0 && pattern != 9) ggml_backend_tensor_copy(source, destination);
                        // Drain output_copy's consumer-stream uploads before the
                        // buffer read, which also orders after the fallback copy.
                        ggml_backend_synchronize(gpu);
                        ggml_backend_tensor_get(destination, actual.data(), 0, ggml_nbytes(destination));
                        pass &= std::memcmp(actual.data(), ref.data(), ggml_nbytes(source)) == 0;
                        pass &= !ggml_moe_cache.output_copy(s, gpu, source, destination);
                        if (!pass) {
                            size_t different = 0, first = actual.size();
                            for (size_t i = 0; i < actual.size(); ++i) {
                                if (std::memcmp(&actual[i], &ref[i], sizeof(float))) {
                                    first = std::min(first, i);
                                    ++different;
                                }
                            }
                            printf("cache-device-output-detail: bound=%d selected=%zu mask=%llx dispatched=%d collected=%d copied=%d different=%zu",
                                    bound, selected_ids.size(), (unsigned long long) selected_mask,
                                    dispatched, collected, copied, different);
                            if (different) printf(" first=%zu expected=%a actual=%a", first, double(ref[first]), double(actual[first]));
                            printf("\n");
                        }
                        printf("cache-device-output: pinned=%d tokens=%d down=%s pattern=%d %s\n",
                                pinned, T, dedicated_down ? dedicated_down : "auto", pattern, pass ? "OK" : "FAIL");
                        ok &= pass;
                        stop(s);
                    }
                    // No active owner, host destination, and views must refuse.
                    ok &= !ggml_moe_cache.output_bind(nullptr, gpu, source, destination);
                    s = start();
                    ok &= !ggml_moe_cache.output_bind(s, gpu, source, source);
                    auto view = *source;
                    view.view_src = source;
                    ok &= !ggml_moe_cache.output_bind(s, gpu, &view, destination);
                    stop(s);

                    // Actual scheduler boundary: CPU full-FFN producer followed
                    // by a GPU consumer. Four CPU workers must finish their
                    // misses before the complementary upload is submitted.
                    for (int run = 0; run < 16; ++run) {
                        // Recreate each graph/session twice rather than relying
                        // on a pointer-stable binding across evaluations.
                        const int scenario = run % 8;
                        const char * fail = scenario == 1 ? "dispatch" : scenario == 2 ? "collect" :
                                scenario == 7 ? "collect-retire" : nullptr;
                        configure_cache(fail, "4", "1", "40", dedicated_down);
                        auto ic = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
                        auto input = ggml_new_tensor_3d(ic, GGML_TYPE_F32, K, 1, T);
                        auto routes_tensor = ggml_new_tensor_2d(ic, GGML_TYPE_I32, topk, T);
                        auto ib = ggml_backend_alloc_ctx_tensors(ic, cpu);
                        GGML_ASSERT(ib);
                        ggml_set_input(input);
                        ggml_set_input(routes_tensor);
                        ggml_backend_tensor_set(input, acts.data(), 0, ggml_nbytes(input));
                        ggml_backend_tensor_set(routes_tensor, ids.data(), 0, ggml_nbytes(routes_tensor));
                        auto gc = ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead(), nullptr, true});
                        auto gu = ggml_mul_mat_id(gc, up, input, routes_tensor);
                        auto gg = ggml_mul_mat_id(gc, gate, input, routes_tensor);
                        auto glu = ggml_swiglu_split(gc, gg, gu);
                        auto gd = ggml_mul_mat_id(gc, down, glu, routes_tensor);
                        auto result = ggml_scale(gc, gd, .5f);
                        if (scenario == 3) ggml_set_input(gd);
                        if (scenario == 4) ggml_set_output(gd);
                        ggml_set_output(result);
                        auto graph = ggml_new_graph(gc);
                        ggml_build_forward_expand(graph, result);
                        ggml_backend_t backends[] = {gpu, cpu};
                        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, scenario == 5, false);
                        if (scenario == 6) {
                            ggml_backend_sched_set_eval_callback(sched,
                                    [](ggml_tensor *, bool, void *) { return true; }, nullptr);
                        }
                        ggml_backend_sched_set_tensor_backend(sched, gu, cpu);
                        ggml_backend_sched_set_tensor_backend(sched, gg, cpu);
                        ggml_backend_sched_set_tensor_backend(sched, glu, cpu);
                        ggml_backend_sched_set_tensor_backend(sched, gd, cpu);
                        ggml_backend_sched_set_tensor_backend(sched, result, gpu);
                        bool pass = ggml_backend_sched_alloc_graph(sched, graph);
                        std::vector<float> expected = ref, actual(ref.size());
                        for (auto & v : expected) v *= .5f;
                        int deferred_failures = 0;
                        for (int step = 0; pass && step < 96; ++step) {
                            const auto status = ggml_backend_sched_graph_compute(sched, graph);
                            if (scenario == 7 && status == GGML_STATUS_FAILED) {
                                ++deferred_failures;
                                continue;
                            }
                            pass &= status == GGML_STATUS_SUCCESS;
                            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                            pass &= compare_output(expected, actual, 5e-4);
                            if (scenario == 4) {
                                ggml_backend_tensor_get(gd, actual.data(), 0, ggml_nbytes(gd));
                                pass &= compare_output(ref, actual, 5e-4);
                            }
                        }
                        if (scenario == 7) pass &= deferred_failures > 0;
                        printf("cache-device-output-scheduler: pinned=%d tokens=%d scenario=%d failure=%s %s\n",
                                pinned, T, scenario, fail ? fail : "none", pass ? "OK" : "FAIL");
                        ok &= pass;
                        ggml_backend_sched_free(sched);
                        ggml_free(gc);
                        ggml_backend_buffer_free(ib);
                        ggml_free(ic);
                    }
                }
                ggml_backend_buffer_free(db);
                ggml_backend_buffer_free(hb);
                ggml_free(dc);
                ggml_free(hc);
            }
            {
                s = start();
                warm(0);
                long long before_boundary = 0;
                if (boundary && pinned) {
                    // Four 32-expert sources create a partial pool. Dropping the
                    // unrelated fourth leaves 96 entries: fewer than physical
                    // slots, but MORE than persistent slots after reservation.
                    const auto slots = counter("pool[0]: type=q2_0 expert=450 KiB slots=");
                    GGML_ASSERT(slots >= 96 && slots <= 98);
                    mask = 0;
                    node = plan(mask);
                    GGML_ASSERT(collect(node, mask, out));
                    ggml_moe_cache.end(node);
                    before_boundary = counter("stream-stage-experts=");
                    ggml_moe_cache.invalidate(dummy->data, ggml_nbytes(dummy));
                }
                mask              = 0;
                node              = plan(mask);
                uint64_t required = resident;
                if (pinned) {
                    for (int e = 0; e < 4; ++e) {
                        for (int t = 0; t < T; ++t) {
                            required |= 1ULL << (t * topk + (e == 0 ? 0 : e + 1));
                        }
                    }
                }
                // Normal async admission may add rows; staging must add its promised rows.
                bool           pass       = node && (mask & required) == required;
                const uint64_t first_mask = mask;
                if (!pinned) {
                    pass &= mask == resident;
                }
                uint64_t busy_mask = 0;
                auto     busy      = plan(busy_mask);
                pass &= !busy && !busy_mask;
                if (busy) {
                    ggml_moe_cache.end(busy);
                }
                pass &= collect(node, mask, out);
                if (boundary && pinned) {
                    const bool continued = counter("stream-stage-experts=") > before_boundary;
                    printf("cache-stream-stage-boundary: partial=%d\n", continued);
                    pass &= continued;
                }
                for (int i = 0; i < R; ++i) {
                    if (mask & (1ULL << i)) {
                        pass &= std::memcmp(ref.data() + i * K, out.data() + i * K, K * sizeof(float)) == 0;
                    }
                }
                if (node) {
                    ggml_moe_cache.end(node);
                }
                // Cancellation both before and after dispatch must leave scratch reusable.
                for (bool dispatch : { false, true }) {
                    uint64_t cancel_mask = 0;
                    auto     cancel      = plan(cancel_mask);
                    pass &= cancel != nullptr;
                    if (cancel && dispatch) {
                        pass &= ggml_moe_cache.fused_dispatch(cancel) != 0;
                    }
                    if (cancel) {
                        ggml_moe_cache.end(cancel);
                    }
                }
                mask = 0;
                node = plan(mask);
                pass &= collect(node, mask, out);
                for (int i = 0; i < R; ++i) {
                    if (mask & (1ULL << i)) {
                        pass &= std::memcmp(ref.data() + i * K, out.data() + i * K, K * sizeof(float)) == 0;
                    }
                }
                if (node) {
                    ggml_moe_cache.end(node);
                }
                // Drop all ordinary residents; swap expert order so the same reserved
                // scratch slots must receive different weight bytes on their next use.
                for (auto w : { up, gate, down }) {
                    ggml_moe_cache.invalidate(w->data, ggml_nbytes(w));
                }
                warm(0);
                for (int t = 0; t < T; ++t) {
                    std::swap(ids[topk * t], ids[topk * t + 2]);
                }
                mask = 0;
                node = plan(mask);
                pass &= node && (mask & required) == required;
                std::atomic<bool> started{ false }, done{ false };
                std::thread       invalidator;
                if (node) {
                    pass &= ggml_moe_cache.fused_dispatch(node) != 0;
                    invalidator = std::thread([&] {
                        started.store(true);
                        ggml_moe_cache.invalidate(down->data, ggml_nbytes(down));
                        done.store(true);
                    });
                    while (!started.load()) {
                        std::this_thread::yield();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    pass &= !done.load();
                }
                pass &= collect(node, mask, out);
                for (int i = 0; i < R; ++i) {
                    if (mask & (1ULL << i)) {
                        const int old = i % topk == 0 ? i + 2 : (i % topk == 2 ? i - 2 : i);
                        pass &= std::memcmp(ref.data() + old * K, out.data() + i * K, K * sizeof(float)) == 0;
                    }
                }
                if (node) {
                    ggml_moe_cache.end(node);
                }
                if (invalidator.joinable()) {
                    invalidator.join();
                }
                pass &= done.load();
                for (int t = 0; t < T; ++t) {
                    std::swap(ids[topk * t], ids[topk * t + 2]);
                }
                printf("cache-stream-stage: pinned=%d tokens=%d round=%d first=%llx final=%llx exact=%d\n", pinned, T,
                       round, (unsigned long long) first_mask, (unsigned long long) mask, pass);
                fflush(stdout);
                ok &= pass;
                stop(s);
            }
        }
        if (pinned && T == 2 && !dedicated_down && !boundary) {
            const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            const auto path = std::filesystem::temp_directory_path() /
                ("llama-moe-staging-profile-" + std::to_string(nonce));
            const auto profile = path.string();
            auto s = start(nullptr, profile.c_str());
            warm(E - 1);
            stop(s);

            s = start(nullptr, profile.c_str());
            routes(1); // The recorded hot expert is not requested in this batch.
            uint64_t mask = 0;
            auto node = plan(mask);
            bool pass = node && mask && collect(node, mask, out);
            if (node) ggml_moe_cache.end(node);

            // Probe residency without an ordinary plan: planning the hot expert
            // would itself seed the profile and hide a missing fused-path seed.
            auto copy_buffer = ggml_backend_alloc_buffer(gpu, ggml_nbytes(up));
            GGML_ASSERT(copy_buffer);
            ggml_backend_buffer_set_usage(copy_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            const uint32_t selected[2] = { 0, UINT32_C(1) << 31 };
            for (auto w : { up, gate, down }) {
                ggml_tensor dst = *w;
                dst.buffer = copy_buffer;
                dst.data = ggml_backend_buffer_get_base(copy_buffer);
                bool copied = false;
                for (int attempt = 0; attempt < 100 && !copied; ++attempt) {
                    copied = ggml_moe_cache.prefill_copy(s, gpu, w, &dst, selected, 2);
                    if (!copied) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                pass &= copied;
                if (copied) {
                    const size_t bytes = ggml_nbytes(w) / E;
                    std::vector<uint8_t> actual(bytes), expected(bytes);
                    ggml_backend_tensor_get(&dst, actual.data(), (E - 1) * bytes, bytes);
                    ggml_backend_tensor_get(w, expected.data(), (E - 1) * bytes, bytes);
                    pass &= actual == expected;
                }
            }
            ggml_backend_buffer_free(copy_buffer);
            stop(s);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            std::filesystem::remove(profile + ".lock", ec);
            printf("cache-stream-stage-profile: unrequested-hot-experts %s\n", pass ? "OK" : "FAIL");
            ok &= pass;
        }
        for (const char * fail : { "dispatch", "collect" }) {
            routes(1);
            auto s = start(fail);
            warm(0);
            uint64_t mask = 0;
            auto     node = plan(mask);
            std::fill(out.begin(), out.end(), -123.f);
            bool pass = node && !collect(node, mask, out);
            pass &= std::all_of(out.begin(), out.end(), [](float x) { return x == -123.f; });
            if (node) {
                ggml_moe_cache.end(node);
            }
            printf("cache-stream-stage-failure: pinned=%d tokens=%d failure=%s untouched=%d\n", pinned, T, fail, pass);
            fflush(stdout);
            ok &= pass;
            stop(s);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    return ok;
}

static bool run_stream_staging(ggml_backend_dev_t device,
                               ggml_backend_t     gpu,
                               ggml_backend_t     cpu,
                               log_capture &      capture) {
    configure_cache(nullptr);
    ggml_moe_cache_config      config = {};
    ggml_moe_cache_device_caps caps   = {};
    if (!ggml_moe_cache.query_config(0, 4, &config) || !ggml_moe_cache.query_device(device, &config, &caps)) {
        return false;
    }
    if (caps.compute_capability != 860 && caps.compute_capability != 1200) {
        printf("cache-stream-stage: SKIP (requires SM86 or SM120)\n");
        return true;
    }
    bool ok = true;
    for (int tokens : { 2, 3, 4 }) {
        for (const char * down : { "0", "1", static_cast<const char *>(nullptr) }) {
            ok &= run_stream_staging_case(device, gpu, cpu, capture, tokens, down, false);
        }
    }
    ok &= run_stream_staging_case(device, gpu, cpu, capture, 4, nullptr, true);
    return ok;
}

static bool run_prefill_copy_type(ggml_backend_dev_t dev, ggml_backend_t gpu, ggml_backend_t cpu, ggml_type type) {
    bool all_ok = true;
    for (bool pinned : {false, true}) {
        bool ok = true;
        const auto check = [&](bool passed, const char * label) {
            if (!passed) fprintf(stderr, "cache-prefill-copy: type=%s pinned=%d %s FAIL\n",
                ggml_type_name(type), int(pinned), label);
            ok &= passed;
        };
        auto host_ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        auto gpu_ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        // The provider requires at least 64 resident slots. Exercise all masks
        // of the first eight experts; the rest also verify untouched storage.
        auto source = ggml_new_tensor_3d(host_ctx, type, 256, 128, 64);
        auto destination = ggml_dup_tensor(gpu_ctx, source);
        ggml_set_name(source, "blk.0.ffn_up_exps.weight");
        auto host_buft = pinned ? ggml_backend_dev_host_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto host_buffer = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx, host_buft);
        auto gpu_buffer = ggml_backend_alloc_ctx_tensors(gpu_ctx, gpu);
        GGML_ASSERT(host_buffer && gpu_buffer);
        ggml_backend_buffer_set_usage(host_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(gpu_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        const size_t bytes = ggml_nbytes(source), expert = source->nb[2];
        std::vector<uint8_t> input(bytes), expected(bytes), actual(bytes), sentinel(bytes, 0xa5);
        // Copy-only test: cover every byte value, without interpreting weights.
        for (size_t i = 0; i < bytes; ++i) input[i] = uint8_t(i * 37 + (i / expert) * 83);
        ggml_backend_tensor_set(source, input.data(), 0, bytes);
        void * session = create_direct_session(gpu, cpu);
        GGML_ASSERT(session);
        ggml_moe_cache.session_enter(session);
        bool ready = true;
        // Physical resident order intentionally differs from destination order.
        for (int e : {2, 0, 3, 1}) ready &= wait_for_direct_resident(source, e);
        check(ready, "resident warmup");
        for (uint32_t mask = 0; ready && mask < 256; ++mask) {
            // Use the scheduler's stream. Small buffer_set_tensor uploads may
            // use a separate staging stream that only graph execution waits on.
            ggml_backend_tensor_set_async(gpu, destination, sentinel.data(), 0, bytes);
            ggml_backend_synchronize(gpu);
            const uint32_t selected[2] = {mask, 0};
            const bool copied = ggml_moe_cache.prefill_copy(session, gpu, source, destination, selected, 2);
            const bool wanted = (mask & 15) != 0;
            expected = sentinel;
            if (wanted) {
                for (int e = 0; e < 8; ++e) {
                    if (!(mask & (1u << e))) continue;
                    std::memcpy(expected.data() + e * expert, input.data() + e * expert, expert);
                    if (!(mask & (1u << (e + 1))))
                        std::memcpy(expected.data() + (e + 1) * expert, input.data() + (e + 1) * expert, 512);
                }
            }
            ggml_backend_tensor_get(destination, actual.data(), 0, bytes);
            if (copied != wanted || actual != expected) {
                fprintf(stderr, "cache-prefill-copy: type=%s pinned=%d mask=%u copied=%d expected=%d FAIL\n",
                    ggml_type_name(type), int(pinned), mask, int(copied), int(wanted));
                const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
                if (mismatch.first != actual.end()) {
                    fprintf(stderr, "cache-prefill-copy: first mismatch byte=%zu actual=%u expected=%u\n",
                        size_t(mismatch.first - actual.begin()), unsigned(*mismatch.first), unsigned(*mismatch.second));
                }
                ok = false;
                break;
            }
        }
        if (!pinned && ggml_moe_cache.prefetch_supported) {
            check(!ggml_moe_cache.prefetch_supported(gpu, source), "lookahead pageable allocation refusal");
        }
        if (ggml_moe_cache.prefetch_supported && ggml_moe_cache.prefetch_supported(gpu, source)) {
            for (int repeat = 0; repeat < 3; ++repeat) {
                ggml_backend_tensor_set_async(gpu, destination, sentinel.data(), 0, bytes);
                ggml_backend_synchronize(gpu);
                auto job = ggml_moe_cache.prefetch_begin(session, gpu, source, destination, nullptr, 0);
                check(bool(job) == pinned, "lookahead pinned admission");
                if (job) check(ggml_moe_cache.prefetch_end(job), "lookahead completion");
                ggml_backend_tensor_get(destination, actual.data(), 0, bytes);
                check(actual == (pinned ? input : sentinel), "lookahead full bytes");
            }
            check(!ggml_moe_cache.prefetch_begin(nullptr, gpu, source, destination, nullptr, 0), "lookahead null session");
            const uint32_t short_mask[] = {1};
            check(!ggml_moe_cache.prefetch_begin(session, gpu, source, destination, short_mask, 1), "lookahead short bitset");
            for (uint32_t pattern = 0; pinned && ready && pattern < 258; ++pattern) {
                const uint32_t selected[2] = {
                    pattern < 256 ? pattern : pattern == 257 ? UINT32_MAX : 0,
                    pattern == 256 ? uint32_t(1) << 31 : pattern == 257 ? UINT32_MAX : 0};
                const auto used = [&](int e) { return (selected[e/32] >> (e%32)) & 1u; };
                ggml_backend_tensor_set_async(gpu, destination, sentinel.data(), 0, bytes);
                ggml_backend_synchronize(gpu);
                auto job = ggml_moe_cache.prefetch_begin(session, gpu, source, destination, selected, 2);
                check(job != nullptr, "lookahead selected admission");
                if (job) check(ggml_moe_cache.prefetch_end(job), "lookahead selected completion");
                expected = sentinel;
                for (int e=0;e<64;++e) {
                    if (!used(e)) continue;
                    std::memcpy(expected.data()+e*expert,input.data()+e*expert,expert);
                    if (e+1<64 && !used(e+1)) {
                        std::memcpy(expected.data()+(e+1)*expert,input.data()+(e+1)*expert,512);
                    }
                }
                ggml_backend_tensor_get(destination,actual.data(),0,bytes);
                check(actual == expected, "lookahead selected bytes and padding");
                if (!ok) break;
            }
            check(!ggml_moe_cache.prefetch_supported(cpu, source), "lookahead CPU refusal");
            printf("cache-prefetch: pinned=%d repeated=3 %s\n", int(pinned), ok ? "OK" : "FAIL");
        }
        const uint32_t mask[2] = {15, 0};
        ggml_backend_tensor_set_async(gpu, destination, sentinel.data(), 0, bytes);
        ggml_backend_synchronize(gpu);
        check(!ggml_moe_cache.prefill_copy(nullptr, gpu, source, destination, mask, 2), "null session");
        check(!ggml_moe_cache.prefill_copy(session, gpu, source, destination, mask, 1), "short bitset");
        check(!ggml_moe_cache.prefill_copy(session, cpu, source, destination, mask, 2), "CPU backend");
        auto unaligned = *destination;
        unaligned.data = (char *) destination->data + 1;
        check(!ggml_moe_cache.prefill_copy(session, gpu, source, &unaligned, mask, 2), "unaligned destination");
        auto unsupported = *source;
        unsupported.type = GGML_TYPE_F32;
        check(!ggml_moe_cache.prefill_copy(session, gpu, &unsupported, destination, mask, 2), "unsupported type");
        ggml_backend_buffer_set_usage(gpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        check(!ggml_moe_cache.prefill_copy(session, gpu, source, destination, mask, 2), "non-compute destination");
        ggml_backend_buffer_set_usage(gpu_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_moe_cache.session_leave(session);
        check(!ggml_moe_cache.prefill_copy(session, gpu, source, destination, mask, 2), "outside scope");
        ggml_moe_cache.session_enter(session);
        input[0] ^= 0xff;
        ggml_backend_tensor_set(source, input.data(), 0, bytes);
        check(!ggml_moe_cache.prefill_copy(session, gpu, source, destination, mask, 2), "invalidated source");
        ggml_backend_tensor_get(destination, actual.data(), 0, bytes);
        check(actual == sentinel, "untouched refusals");
        if (pinned && ggml_moe_cache.prefetch_supported && ggml_moe_cache.prefetch_supported(gpu, source)) {
            auto job = ggml_moe_cache.prefetch_begin(session, gpu, source, destination, nullptr, 0);
            check(job != nullptr, "lookahead invalidated-source copy");
            if (job) {
                std::atomic<bool> entered{false}, completed{false};
                std::thread invalidate([&] {
                    entered.store(true, std::memory_order_release);
                    ggml_moe_cache.invalidate(source->data, bytes);
                    completed.store(true, std::memory_order_release);
                });
                while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                check(!completed.load(std::memory_order_acquire), "lookahead source lease");
                check(ggml_moe_cache.prefetch_end(job), "lookahead release under invalidation");
                invalidate.join();
                check(completed.load(std::memory_order_acquire), "lookahead invalidation completes");
                ggml_backend_tensor_get(destination, actual.data(), 0, bytes);
                check(actual == input, "lookahead post-invalidation bytes");
            }
        }
        ggml_moe_cache.session_leave(session);
        ggml_moe_cache.session_destroy(session);
        ggml_backend_buffer_free(gpu_buffer);
        ggml_backend_buffer_free(host_buffer);
        ggml_free(gpu_ctx);
        ggml_free(host_ctx);
        printf("cache-prefill-copy: type=%s pinned=%d masks=256 exact-padding untouched-refusals invalidation %s\n",
            ggml_type_name(type), int(pinned), ok ? "OK" : "FAIL");
        all_ok &= ok;
    }
    return all_ok;
}

static bool run_prefill_copy(ggml_backend_dev_t dev, ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    if (!ggml_moe_cache.prefill_copy) return false;
    bool all_ok = true;
    // Ask the provider for its supported canonical weight formats rather than
    // maintaining a second list. This includes every EXL3 codebook/bit width.
    int tested_types = 0;
    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = ggml_type(value);
        if (!ggml_blck_size(type) || 256 % ggml_blck_size(type)) continue;
        ggml_moe_cache_shape_caps shape = {};
        if (!ggml_moe_cache.query_shape(type, 256, 128, 64,
                ggml_row_size(type, 256) * 128, &shape)) continue;
        ++tested_types;
        all_ok &= run_prefill_copy_type(dev, gpu, cpu, type);
    }
    printf("cache-prefill-copy: formats=%d %s\n", tested_types, all_ok && tested_types > 0 ? "OK" : "FAIL");
    all_ok &= tested_types > 0;
    return all_ok;
}

static bool run_prefill_copy_multi(ggml_backend_dev_t dev, ggml_backend_t gpu, ggml_backend_t cpu) {
    auto other_dev = find_other_cuda_device(dev);
    if (!other_dev) {
        printf("cache-prefill-copy-multi: SKIP (requires two GPUs)\n");
        return true;
    }
    auto other = ggml_backend_dev_init(other_dev, nullptr);
    GGML_ASSERT(other);
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "2");
    set_env("GGML_CUDA_MOE_CACHE_EXPERT_PARALLEL", "0");
    ggml_backend_t devices[] = {gpu, other};
    void * backends[] = {gpu, other, cpu};
    auto session = ggml_moe_cache.session_create(backends, 3, nullptr);
    GGML_ASSERT(session);
    auto hc = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * source[2];
    for (int i = 0; i < 2; ++i) {
        source[i] = ggml_new_tensor_3d(hc, GGML_TYPE_MXFP4, 256, 128, 64);
        ggml_format_name(source[i], "blk.%d.ffn_up_exps.weight", i);
    }
    auto hb = ggml_backend_alloc_ctx_tensors(hc, cpu);
    GGML_ASSERT(hb);
    ggml_backend_buffer_set_usage(hb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t bytes = ggml_nbytes(source[0]), expert = source[0]->nb[2];
    std::vector<uint8_t> input[2], actual(bytes), expected(bytes), sentinel(bytes, 0xa5);
    ggml_context * dc[2];
    ggml_tensor * destination[2];
    ggml_backend_buffer_t db[2];
    for (int i = 0; i < 2; ++i) {
        input[i].resize(bytes);
        for (size_t j = 0; j < bytes; ++j) input[i][j] = uint8_t(j * 37 + i * 83);
        ggml_backend_tensor_set(source[i], input[i].data(), 0, bytes);
        dc[i] = ggml_init({ggml_tensor_overhead(), nullptr, true});
        destination[i] = ggml_dup_tensor(dc[i], source[i]);
        db[i] = ggml_backend_alloc_ctx_tensors(dc[i], devices[i]);
        GGML_ASSERT(db[i]);
        ggml_backend_buffer_set_usage(db[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    ggml_moe_cache.session_enter(session);
    bool ok = wait_for_direct_resident(source[0], 63) && wait_for_direct_resident(source[1], 63);
    const uint32_t selected[] = {0, UINT32_C(1) << 31};
    int owners[] = {-1, -1};
    for (int s = 0; s < 2; ++s) for (int d = 0; d < 2; ++d) {
        ggml_backend_tensor_set_async(devices[d], destination[d], sentinel.data(), 0, bytes);
        ggml_backend_synchronize(devices[d]);
        const bool copied = ggml_moe_cache.prefill_copy(session, devices[d], source[s], destination[d], selected, 2);
        expected = sentinel;
        if (copied) {
            ok &= owners[s] == -1;
            owners[s] = d;
            std::memcpy(expected.data() + 63 * expert, input[s].data() + 63 * expert, expert);
        }
        ggml_backend_tensor_get(destination[d], actual.data(), 0, bytes);
        ok &= actual == expected;
        // A backend must never write another device's destination buffer.
        ok &= !ggml_moe_cache.prefill_copy(session, devices[1-d], source[s], destination[d], selected, 2);
    }
    ok &= owners[0] >= 0 && owners[1] >= 0 && owners[0] != owners[1];
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    for (int i = 0; i < 2; ++i) {
        ggml_backend_buffer_free(db[i]);
        ggml_free(dc[i]);
    }
    ggml_backend_buffer_free(hb);
    ggml_free(hc);
    ggml_backend_free(other);
    configure_cache(nullptr);
    printf("cache-prefill-copy-multi: local-owners=%d,%d exact-last-expert no-peer-fallback %s\n",
        owners[0], owners[1], ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    const bool profile_writer = argc == 6 && std::strcmp(argv[1], "--profile-writer") == 0;
    const bool host_buffer = argc == 2 && std::strcmp(argv[1], "--host-buffer") == 0;
    const bool persistent_pool = argc == 2 && std::strcmp(argv[1], "--cpu-persistent-pool") == 0;
    const bool pq2_only = argc == 2 && std::strcmp(argv[1], "--pq2") == 0;
    if (!run_pool_limits()) return 1;
    if (argc > 0 && argv[0]) {
        std::error_code ec;
        test_executable = std::filesystem::absolute(argv[0], ec).string();
        if (ec) {
            test_executable = argv[0];
        }
    }
    log_capture capture;
    ggml_log_set(log_callback, &capture);

    if (persistent_pool) {
        ggml_backend_load_all();
    }
    ggml_backend_dev_t cuda_device = persistent_pool ? nullptr : find_cuda_device();
    ggml_backend_t cpu = init_cpu_backend();
    if (!cpu) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return 1;
    }
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");
    if (persistent_pool) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
        const auto create = (ggml_threadpool * (*)(ggml_threadpool_params *))
            ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        const auto destroy = (void (*)(ggml_threadpool *))
            ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        const auto attach = (void (*)(ggml_backend_t, ggml_threadpool *))
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        GGML_ASSERT(create && destroy && attach && set_n_threads);
        auto params = ggml_threadpool_params_default(6);
        auto * pool = create(&params);
        GGML_ASSERT(pool);
        attach(cpu, pool);
        bool ok = true;
        for (int threads : { 1, 2, 4, 6, 4, 2, 1 }) {
            set_n_threads(cpu, threads);
            printf("cache-fused-cpu-persistent-pool: threads=%d\n", threads);
            ok &= run_fused_cpu_fallbacks(cpu);
            ok &= run_fused_alias_fallbacks(cpu);
        }
        attach(cpu, nullptr);
        destroy(pool);
        printf("cache-fused-cpu-persistent-pool: %s\n", ok ? "PASS" : "FAIL");
        ggml_backend_free(cpu);
        return ok ? 0 : 1;
    }
    // Profile writer subprocesses must reach their rendezvous promptly. They
    // exercise profile I/O, not the parent's complete CPU fallback matrix.
    if (!profile_writer && !pq2_only) {
        for (int threads : { 1, 2, 4, 6 }) {
            if (set_n_threads) set_n_threads(cpu, threads);
            printf("cache-fused-cpu-fallbacks: threads=%d\n", threads);
            if (!run_fused_cpu_fallbacks(cpu) || !run_fused_alias_fallbacks(cpu)) {
                ggml_backend_free(cpu);
                return 1;
            }
        }
    }
    if (!cuda_device) {
        printf("SKIP: GPU cache scenarios (CUDA/HIP backend unavailable); CPU fallbacks passed\n");
        ggml_backend_free(cpu);
        return 0;
    }
    ggml_backend_reg_t cuda_reg =
        ggml_backend_dev_backend_reg(cuda_device);

    using flat_hits_reset_fn = void (*)();
    using flat_hits_get_fn = void (*)(uint64_t *, uint64_t *);
    const bool flat_hits_expected = std::getenv("GGML_TEST_MOE_CACHE_FLAT_HITS_EXPECT_BASELINE") != nullptr;
    flat_hits_reset_fn flat_hits_reset = nullptr;
    flat_hits_get_fn flat_hits_get = nullptr;
    if (flat_hits_expected) {
        flat_hits_reset = (flat_hits_reset_fn) ggml_backend_reg_get_proc_address(
            cuda_reg, "ggml_cuda_moe_cache_flat_hits_test_stats_reset");
        flat_hits_get = (flat_hits_get_fn) ggml_backend_reg_get_proc_address(
            cuda_reg, "ggml_cuda_moe_cache_flat_hits_test_stats_get");
        if (!flat_hits_reset || !flat_hits_get) {
            std::fprintf(stderr, "cache-flat-hits: instrumentation hooks unavailable\n");
            ggml_backend_free(cpu);
            return 1;
        }
        flat_hits_reset();
    }

    ggml_backend_t cuda = ggml_backend_dev_init(cuda_device, nullptr);
    if (!cuda) {
        fprintf(stderr, "failed to initialize GPU backend\n");
        ggml_backend_free(cpu);
        return 1;
    }
    if (pq2_only) {
        const bool ok = run_pq2_cache(cuda, cpu, capture);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return ok ? 0 : 1;
    }
    ggml_init_params static_params = {
        16 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * static_ctx = ggml_init(static_params);
    if (!static_ctx) {
        fprintf(stderr, "failed to create tensor context\n");
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * gate_weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * down_weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q5_0, n_out, n_in, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            static_ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, n_in, 1, n_tokens);
    ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");
    ggml_set_name(ids, "moe_cache_test_ids");
    ggml_set_name(activations, "moe_cache_test_activations");

    const auto static_buft = host_buffer ? ggml_backend_dev_host_buffer_type(cuda_device)
                                         : ggml_backend_get_default_buffer_type(cpu);
    ggml_backend_buffer_t static_buffer = static_buft
        ? ggml_backend_alloc_ctx_tensors_from_buft(static_ctx, static_buft) : nullptr;
    if (!static_buffer) {
        fprintf(stderr, "failed to allocate CPU tensors\n");
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_buffer_set_usage(
            static_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (host_buffer) {
        // Do not silently qualify the CPU fallback if pinned allocation failed.
        if (ggml_backend_buffer_get_type(static_buffer) != static_buft) {
            fprintf(stderr, "requested host buffer allocation fell back to another type\n");
            ggml_backend_buffer_free(static_buffer);
            ggml_free(static_ctx);
            ggml_backend_free(cuda);
            ggml_backend_free(cpu);
            return 1;
        }
        printf("cache-source-buffer: %s\n", ggml_backend_buft_name(static_buft));
    }

    std::vector<float> weights_f32(ggml_nelements(weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.15f * std::sin((float) (index % 997) * 0.017f) +
            0.05f * std::cos((float) (index % 431) * 0.031f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "unexpected quantized size: %zu != %zu\n",
                quantized, weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<float> gate_weights_f32(ggml_nelements(gate_weights));
    for (size_t index = 0; index < gate_weights_f32.size(); index++) {
        gate_weights_f32[index] =
            0.11f * std::sin((float) (index % 881) * 0.021f) -
            0.08f * std::cos((float) (index % 389) * 0.027f);
    }
    std::vector<uint8_t> gate_weights_q4(ggml_nbytes(gate_weights));
    const size_t gate_quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, gate_weights_f32.data(), gate_weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (gate_quantized != gate_weights_q4.size()) {
        fprintf(stderr, "unexpected gate quantized size: %zu != %zu\n",
                gate_quantized, gate_weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            gate_weights, gate_weights_q4.data(), 0,
            gate_weights_q4.size());

    std::vector<float> down_weights_f32(ggml_nelements(down_weights));
    for (size_t index = 0; index < down_weights_f32.size(); index++) {
        down_weights_f32[index] =
            0.09f * std::sin((float) (index % 787) * 0.019f) +
            0.06f * std::cos((float) (index % 353) * 0.023f);
    }
    std::vector<uint8_t> down_weights_q5(ggml_nbytes(down_weights));
    const size_t down_quantized = ggml_quantize_chunk(
            GGML_TYPE_Q5_0, down_weights_f32.data(), down_weights_q5.data(),
            0, n_in * n_expert, n_out, nullptr);
    if (down_quantized != down_weights_q5.size()) {
        fprintf(stderr, "unexpected down quantized size: %zu != %zu\n",
                down_quantized, down_weights_q5.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            down_weights, down_weights_q5.data(), 0,
            down_weights_q5.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.4f * std::sin((float) index * 0.07f) -
            0.2f * std::cos((float) index * 0.11f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    if (profile_writer) {
        const bool child_ok = run_profile_writer_child(
                cuda, cpu, weights, argv[2], (int32_t)std::strtol(argv[3], nullptr, 10),
                argv[4], argv[5]);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_log_set(nullptr, nullptr);
        return child_ok ? 0 : 1;
    }

    test_graph graph = make_graph(cpu, weights, activations, ids);
    if (!graph.ctx || !graph.buffer) {
        fprintf(stderr, "failed to create test graph\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPU reference compute failed\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> reference(ggml_nelements(graph.out));
    ggml_backend_tensor_get(
            graph.out, reference.data(), 0, reference.size() * sizeof(float));

    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!fused_graph.ctx || !fused_graph.buffer) {
        fprintf(stderr, "failed to create fused test graph\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fused CPU reference compute failed\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> fused_reference(ggml_nelements(fused_graph.out));
    ggml_backend_tensor_get(
            fused_graph.out, fused_reference.data(), 0,
            fused_reference.size() * sizeof(float));

    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!clamped_fused_graph.ctx || !clamped_fused_graph.buffer) {
        fprintf(stderr, "failed to create clamped fused test graph\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, clamped_fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "clamped fused CPU reference compute failed\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> clamped_fused_reference(
            ggml_nelements(clamped_fused_graph.out));
    ggml_backend_tensor_get(
            clamped_fused_graph.out, clamped_fused_reference.data(), 0,
            clamped_fused_reference.size() * sizeof(float));

    test_graph full_fused_graph = make_full_fused_graph(
            cpu, weights, gate_weights, down_weights, activations, ids, ffn_clamp::none);
    if (!full_fused_graph.ctx || !full_fused_graph.buffer) {
        fprintf(stderr, "failed to create full fused test graph\n");
        free_graph(full_fused_graph);
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, full_fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "full fused CPU reference compute failed\n");
        free_graph(full_fused_graph);
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> full_fused_reference(
            ggml_nelements(full_fused_graph.out));
    ggml_backend_tensor_get(
            full_fused_graph.out, full_fused_reference.data(), 0,
            full_fused_reference.size() * sizeof(float));

    test_graph clamped_full_fused_graph = make_full_fused_graph(
            cpu, weights, gate_weights, down_weights, activations, ids, ffn_clamp::separate);
    if (!clamped_full_fused_graph.ctx || !clamped_full_fused_graph.buffer) {
        fprintf(stderr, "failed to create clamped full fused test graph\n");
        free_graph(clamped_full_fused_graph);
        free_graph(full_fused_graph);
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, clamped_full_fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "clamped full fused CPU reference compute failed\n");
        free_graph(clamped_full_fused_graph);
        free_graph(full_fused_graph);
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> clamped_full_fused_reference(
            ggml_nelements(clamped_full_fused_graph.out));
    ggml_backend_tensor_get(
            clamped_full_fused_graph.out,
            clamped_full_fused_reference.data(), 0,
            clamped_full_fused_reference.size() * sizeof(float));

    bool ok = run_capability_queries(cuda_device, cpu);
    ok &= run_scheduler_overlap_config(cuda, cpu);
    ok &= run_invalidation_hook_coverage(cpu);
    ok &= run_scenario("cache-hit", nullptr, cuda, cpu, graph, reference, capture);
    scenario_options generic_mmv_options;
    generic_mmv_options.dedicated_mmv = "0";
    ok &= run_scenario(
            "cache-generic-mmv", nullptr, cuda, cpu, graph, reference,
            capture, generic_mmv_options);
    ok &= run_scenario("dispatch-fallback", "dispatch", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("collect-fallback", "collect", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("insert-fallback", "insert", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("slab-fallback", "slab", cuda, cpu, graph, reference, capture);
    scenario_options fused_options;
    fused_options.required_field = "fusion-nodes=";
    ok &= run_scenario(
            "cache-fused-swiglu", nullptr, cuda, cpu,
            fused_graph, fused_reference, capture, fused_options);
    scenario_options fusion_attempt_options = fused_options;
    fusion_attempt_options.required_field = "fusion-attempts=";
    ok &= run_scenario(
            "cache-fused-dispatch-fallback", "dispatch", cuda, cpu,
            fused_graph, fused_reference, capture, fusion_attempt_options);
    ok &= run_scenario(
            "cache-fused-collect-fallback", "collect", cuda, cpu,
            fused_graph, fused_reference, capture, fused_options);
    ok &= run_scenario(
            "cache-fused-clamped-swiglu", nullptr, cuda, cpu,
            clamped_fused_graph, clamped_fused_reference, capture,
            fused_options);
    ok &= run_scenario(
            "cache-fused-clamped-collect-fallback", "collect", cuda, cpu,
            clamped_fused_graph, clamped_fused_reference, capture,
            fused_options);
    scenario_options full_options;
    full_options.required_field = "full-fusion=";
    full_options.down_mmv = down_mmv_expectation::generic;
    ok &= run_scenario(
            "cache-fused-full-ffn-graph", nullptr, cuda, cpu,
            full_fused_graph, full_fused_reference, capture, full_options);
    scenario_options dedicated_down_options = full_options;
    dedicated_down_options.dedicated_down_mmv = "1";
    dedicated_down_options.down_mmv = down_mmv_expectation::dedicated;
    ok &= run_scenario(
            "cache-fused-full-ffn-dedicated-down", nullptr, cuda, cpu,
            full_fused_graph, full_fused_reference, capture,
            dedicated_down_options);
    ok &= run_scenario(
            "cache-fused-full-ffn-collect-fallback", "collect", cuda, cpu,
            full_fused_graph, full_fused_reference, capture, full_options);
    ok &= run_scenario(
            "cache-fused-clamped-full-ffn-graph", nullptr, cuda, cpu,
            clamped_full_fused_graph, clamped_full_fused_reference, capture,
            full_options);
    // DeepSeek's combined SWIGLU_CLAMP must use the same cache fusion as the
    // decomposed graph, including CPU misses and failed GPU collection.
    test_graph combined_graph = make_full_fused_graph(
            cpu, weights, gate_weights, down_weights, activations, ids, ffn_clamp::combined);
    if (!combined_graph.ctx || !combined_graph.buffer) {
        ok = false;
    } else {
        set_env("GGML_CUDA_MOE_CACHE", "0");
        if (ggml_backend_graph_compute(cpu, combined_graph.graph) != GGML_STATUS_SUCCESS) {
            ok = false;
        } else {
            std::vector<float> combined_reference(ggml_nelements(combined_graph.out));
            ggml_backend_tensor_get(combined_graph.out, combined_reference.data(), 0,
                    combined_reference.size()*sizeof(float));
            ok &= run_scenario("cache-combined-clamp-full-ffn", nullptr, cuda, cpu,
                    combined_graph, combined_reference, capture, full_options);
            ok &= run_scenario("cache-combined-clamp-collect-fallback", "collect", cuda, cpu,
                    combined_graph, combined_reference, capture, full_options);
        }
    }
    free_graph(combined_graph);
    ok &= run_fused_partial_invalidation(
            cuda, cpu, weights, gate_weights, activations,
            weights_q4.data(), gate_weights_q4.data(),
            fused_reference, capture);
    ok &= run_activation_map_regression(
            cuda, cpu, weights, gate_weights);
    ok &= run_fused_full_ffn(
            cuda, cpu, weights, gate_weights, down_weights, activations,
            down_weights_q5.data());
    ok &= run_fused_concurrent_sessions(
            cuda_device, cuda, cpu, weights, gate_weights,
            activations, ids, fused_graph, fused_reference, capture);
    ok &= run_fused_repeated_lifecycle(
            cuda, cpu, fused_graph, fused_reference, capture);
    ok &= run_multi_token_scenario(
            cuda_device, cuda, cpu, weights, gate_weights,
            down_weights, capture);
    ok &= run_mxfp4_shared_pool(cuda, cpu, capture);
    ok &= run_pq2_cache(cuda, cpu, capture);
    const size_t expert_size = ggml_nbytes(weights) / n_expert;
    ok &= run_precensus_invalidation(
            cuda, cpu, graph, weights,
            weights_q4.data() + (n_expert - 1) * expert_size,
            expert_size, reference, capture);

    const int32_t repeated_ids[n_used] = { 0, 0 };
    ggml_backend_tensor_set(ids, repeated_ids, 0, sizeof(repeated_ids));
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cache-invalidate: initial CPU reference compute failed\n");
        ok = false;
    } else {
        std::vector<float> old_reference(ggml_nelements(graph.out));
        ggml_backend_tensor_get(
                graph.out, old_reference.data(), 0,
                old_reference.size() * sizeof(float));

        std::vector<float> replacement_f32(n_in);
        for (size_t index = 0; index < replacement_f32.size(); index++) {
            replacement_f32[index] =
                1.25f + 0.3f * std::sin((float) index * 0.041f);
        }
        std::vector<uint8_t> replacement_q4(
                ggml_row_size(GGML_TYPE_Q4_0, n_in));
        const size_t replacement_size = ggml_quantize_chunk(
                GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
                0, 1, n_in, nullptr);
        if (replacement_size != replacement_q4.size()) {
            fprintf(stderr, "cache-invalidate: unexpected replacement size\n");
            ok = false;
        } else {
            ok &= run_invalidation_scenario(
                    cuda, cpu, graph, weights, replacement_q4,
                    old_reference, capture);
        }
    }

    stress_fixture stress;
    if (!init_stress_fixture(stress, cpu)) {
        fprintf(stderr, "failed to initialize cache stress fixture\n");
        ok = false;
    } else {
        ok &= run_concurrent_sessions(
                cuda_device, cuda, cpu, stress, capture);
        ok &= run_repeated_lifecycle(cuda, cpu, stress, capture);
        ok &= run_fill_invalidation(cuda, cpu, stress, capture);
    }
    free_stress_fixture(stress);
    ok &= run_scope_isolation(cuda, cpu, weights);
    ok &= run_explicit_session_config(cuda, cpu);
    ok &= run_single_device_automatic_session(cuda, cpu);
    ok &= run_expert_profile_roundtrip(cuda, cpu, weights, gate_weights, capture);
    ok &= run_policy_diagnostics(cuda, cpu, weights, capture);
    ok &= run_cpu_overlap_policy(cuda, cpu, weights, capture);
    ok &= run_adaptive_cpu_overlap_policy(cuda, cpu, weights, capture);
    ok &= run_shape_liveness(cuda, cpu);
    ok &= run_exact_shape_inventory(cuda, cpu, capture);
    ok &= run_complete_pool_allocation(cuda, cpu);
    ok &= run_shared_budget(cuda, cpu, capture);
    ok &= run_route_override(cuda_device, cuda, cpu, capture);
    ok &= run_admission_policy(cuda, cpu, capture);
    ok &= run_stream_staging(cuda_device, cuda, cpu, capture);
    ok &= run_prefill_copy(cuda_device, cuda, cpu);
    ok &= run_prefill_copy_multi(cuda_device, cuda, cpu);

    if (flat_hits_expected) {
        uint64_t factor_1 = 0;
        uint64_t factor_2 = 0;
        flat_hits_get(&factor_1, &factor_2);
        const bool receipt_ok = factor_1 > 0 && factor_2 == 0;
        std::printf("cache-flat-hits: expected=baseline f1=%llu f2=%llu %s\n",
            (unsigned long long) factor_1, (unsigned long long) factor_2,
            receipt_ok ? "OK" : "FAIL");
        ok &= receipt_ok;
    }

    free_graph(clamped_full_fused_graph);
    free_graph(full_fused_graph);
    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_free(cuda);
#ifdef GGML_BACKEND_DL
    ggml_backend_unload(cuda_reg);
    ggml_backend_buffer_free(static_buffer);
    static_buffer = nullptr;
    printf("cache-backend-unload: OK\n");

    ggml_backend_dev_t reloaded_device = find_cuda_device();
    ggml_backend_t reloaded_cuda = reloaded_device
        ? ggml_backend_dev_init(reloaded_device, nullptr) : nullptr;
    configure_cache(nullptr);
    void * reloaded_session = reloaded_cuda
        ? create_direct_session(reloaded_cuda, cpu) : nullptr;
    const bool reload_ok = reloaded_session != nullptr;
    if (reloaded_session) {
        ggml_moe_cache.session_destroy(reloaded_session);
    }
    if (reloaded_cuda) {
        ggml_backend_reg_t reloaded_reg =
            ggml_backend_dev_backend_reg(reloaded_device);
        ggml_backend_free(reloaded_cuda);
        ggml_backend_unload(reloaded_reg);
    }
    printf("cache-backend-reload: %s\n", reload_ok ? "OK" : "FAIL");
    ok &= reload_ok;
#else
    (void) cuda_reg;
    printf("cache-backend-unload: SKIP (static backend)\n");
#endif
    if (static_buffer) {
        ggml_backend_buffer_free(static_buffer);
    }
    ggml_free(static_ctx);
    ggml_quantize_free();
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    return ok ? 0 : 1;
}
