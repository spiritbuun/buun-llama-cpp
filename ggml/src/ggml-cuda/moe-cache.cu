// MoE expert cache by leloch (github.com/leloch/llama.cpp, branch moe-cache-v2-pr;
// RFC ggml-org/llama.cpp#24528); soft-mode partial-eviction fit by giveen
// (TheTom/llama-cpp-turboquant#284). MIT.

#include "moe-cache.cuh"

#if defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

void ggml_moe_cache_register(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <process.h>
#elif defined(__linux__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <sys/mman.h>
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)

static constexpr int    moe_cache_cc_forced_min               = 700;
static constexpr int    moe_cache_cc_ampere                   = 800;
static constexpr size_t moe_cache_expert_bytes_ampere_min     = 512u << 10;
static constexpr size_t moe_cache_expert_bytes_pre_ampere_min = 1u << 20;
static constexpr int    moe_cache_batch_max                   = 10;
static constexpr int    moe_cache_pool_slots_min              = 64;
static constexpr size_t moe_cache_slab_bytes_auto_min         = 1ull << 30;
static constexpr int    moe_cache_node_rows_max               = 64;
static constexpr int    moe_cache_expert_parallel_max         = 8;
static constexpr size_t moe_cache_overlap_bytes_per_token     = 8u << 20;

enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};

struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    moe_cache_slot_state state = moe_cache_slot_state::free;
};

struct moe_cache_pool {
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;
    bool covers_all_entries = false;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1;
    int lru_tail = -1;
};

struct moe_cache_shape {
    size_t expert_size = 0;
    int wtype = -1;
    uint64_t n_entries = 0;
    int64_t n_tensors = 0;
    int pool = -1;
    bool finished = false;
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
};

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_config {
    bool enabled = true;
    bool automatic = true;
    size_t budget_mb = 0;
    size_t reserve_mb = 3072;
    bool reserve_explicit = false;
    size_t minimum_slab_bytes = moe_cache_slab_bytes_auto_min;
    size_t min_expert_bytes = moe_cache_expert_bytes_pre_ampere_min;
    bool min_expert_explicit = false;
    int max_batch = moe_cache_batch_max;
    bool max_batch_explicit = false;
    int inserts_per_plan = 8;
    bool inserts_per_plan_explicit = false;
    int admit_after = 2;
    bool admit_after_explicit = false;
    int readmit_after = 8;
    bool readmit_after_explicit = false;
    int queue_max = 128;
    size_t queue_mb = 512;
    int stats_every = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = moe_cache_cc_forced_min;
    bool serial_fill = true;
    bool serial_fill_explicit = false;
    bool force_dedicated_mmv = false;
    bool force_dedicated_down_mmv = false;
    int overlap_cpu_rows = -1;
    bool overlap_cpu_rows_explicit = false;
    int expert_parallel = 0;
    std::string fail_stage;
    std::string profile_path;
};

struct moe_cache_profile_key {
    uint64_t tensor = 0;
    int32_t expert = -1;

    bool operator==(const moe_cache_profile_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_profile_key_hash {
    size_t operator()(const moe_cache_profile_key & key) const {
        uint64_t value = key.tensor ^
            ((uint64_t)(uint32_t)key.expert * UINT64_C(0x9e3779b97f4a7c15));
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        return (size_t)(value ^ (value >> 31));
    }
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t input = 0;
    size_t q8 = 0;
    size_t out = 0;
};

struct moe_cache_device {
    moe_cache_device(int logical, int physical) : logical(logical), physical(physical) {}

    int logical;
    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    size_t visits_since_new_tensor = 0;
    bool budget_ready = false;
    bool budget_registered = false;
    bool budget_claimed = false;
    size_t automatic_reserve_bytes = 0;
    size_t budget_reserve_bytes = 0;
    size_t budget_limit = 0;
    size_t coordinator_allocated_bytes = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    const void * inflight_source = nullptr;
    size_t inflight_bytes = 0;
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    char * h_input = nullptr;
    char * d_input = nullptr;
    size_t h_input_cap = 0;
    size_t d_input_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    // hits and misses count tensor-expert probes; fusion counters count row pairs.
    long long hits = 0;
    long long misses = 0;
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long activation_dedup = 0;
    long long overlap_rows = 0;
    long long fused_rows = 0;
    long long fused_candidates = 0;
    long long pair_both = 0;
    long long pair_up_only = 0;
    long long pair_gate_only = 0;
    long long pair_neither = 0;
    long long fused_attempts = 0;
    long long fused_nodes = 0;
    long long full_fused_rows = 0;
    long long full_fused_nodes = 0;
    long long dedicated_down_mmv_dispatches = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    // Dispatch mutex contention that turned a potential cache hit into a
    // complete CPU fallback (try_to_lock failed). Measured first so a bounded
    // wait or per-device FIFO can be justified with data.
    long long contention_bypasses = 0;
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;
    std::unordered_map<moe_cache_profile_key, uint32_t,
        moe_cache_profile_key_hash> profile_heat;
    uint64_t profile_generation = 0;
    std::unordered_map<uint64_t, std::vector<int32_t>> profile_hot;
    std::unordered_map<uint64_t, std::vector<uint32_t>> observed_heat;
    std::unordered_set<moe_cache_key, moe_cache_key_hash> profile_seeded;
    std::unordered_set<const void *> profile_advised;

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    std::atomic<bool> config_announced{false};
    std::atomic<bool> enabled_announced{false};
    std::atomic<bool> batch_bypass_announced{false};
    std::atomic<bool> row_bypass_announced{false};
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;
};

struct moe_cache_pin {
    moe_cache_pool * pool = nullptr;
    int slot = -1;
};

struct moe_cache_node {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    moe_cache_pool * pool = nullptr;
    moe_cache_pool * down_pool = nullptr;
    int pool_index = -1;
    const void * host_base = nullptr;
    const void * host_base2 = nullptr;
    const void * host_base3 = nullptr;
    size_t expert_size = 0;
    int64_t n_in = 0;
    int64_t n_mid = 0;
    int64_t n_out = 0;
    int64_t n_expert = 0;
    int64_t n_tokens = 0;
    int wtype = -1;
    uint64_t profile_keys[3] = {};
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[3 * moe_cache_node_rows_max];
    int n_pins = 0;
    bool planned = false;
    bool dispatched = false;
    bool owns_active = true;
    bool composite = false;
    int n_result_rows = 0;
    int row_indices[moe_cache_node_rows_max];
    std::vector<std::unique_ptr<moe_cache_node>> children;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};
struct moe_cache_physical_budget {
    int participants = 0;
    int prepared = 0;
    std::multiset<size_t> reserves;
    size_t outstanding_bytes = 0;
};
static std::mutex g_budget_mu;
static std::unordered_map<int, moe_cache_physical_budget> g_physical_budgets;
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;
static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);

static void moe_cache_budget_remove_participant(
        moe_cache_physical_budget & state, size_t reserve_bytes) {
    auto reserve = state.reserves.find(reserve_bytes);
    if (reserve != state.reserves.end()) {
        state.reserves.erase(reserve);
    }
    state.participants = std::max(state.participants - 1, 0);
}

static bool moe_cache_budget_register(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    std::vector<moe_cache_device *> registered;
    try {
        registered.reserve(session.devices.size());
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            moe_cache_physical_budget & state = g_physical_budgets[device.physical];
            const size_t reserve_bytes = ggml_moe_cache_effective_reserve_bytes(
                    session.config.reserve_explicit,
                    session.config.reserve_mb << 20,
                    device.automatic_reserve_bytes);
            state.reserves.insert(reserve_bytes);
            state.participants++;
            device.budget_registered = true;
            device.budget_reserve_bytes = reserve_bytes;
            registered.push_back(&device);
        }
        return true;
    } catch (...) {
        for (moe_cache_device * device : registered) {
            auto found = g_physical_budgets.find(device->physical);
            if (found != g_physical_budgets.end()) {
                moe_cache_budget_remove_participant(
                        found->second, device->budget_reserve_bytes);
                if (found->second.participants == 0) {
                    g_physical_budgets.erase(found);
                }
            }
            device->budget_registered = false;
            device->budget_reserve_bytes = 0;
        }
        for (auto found = g_physical_budgets.begin();
             found != g_physical_budgets.end();) {
            if (found->second.participants == 0) {
                found = g_physical_budgets.erase(found);
            } else {
                ++found;
            }
        }
        return false;
    }
}

static void moe_cache_budget_allocation(
        moe_cache_device & device, size_t bytes, bool allocated) {
    if (!device.budget_registered || !device.budget_claimed || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }
    moe_cache_physical_budget & state = found->second;
    if (allocated) {
        const size_t remaining = device.budget_limit > device.coordinator_allocated_bytes
            ? device.budget_limit - device.coordinator_allocated_bytes : 0;
        const size_t tracked = std::min(bytes, remaining);
        device.coordinator_allocated_bytes += tracked;
        state.outstanding_bytes = tracked <= state.outstanding_bytes
            ? state.outstanding_bytes - tracked : 0;
    } else {
        const size_t tracked = std::min(bytes, device.coordinator_allocated_bytes);
        device.coordinator_allocated_bytes -= tracked;
        if (tracked <= SIZE_MAX - state.outstanding_bytes) {
            state.outstanding_bytes += tracked;
        } else {
            state.outstanding_bytes = SIZE_MAX;
        }
    }
}

static void moe_cache_budget_reallocation(
        moe_cache_device & device, size_t old_bytes, size_t new_bytes) {
    if (!device.budget_registered || !device.budget_claimed || old_bytes == new_bytes) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }

    moe_cache_physical_budget & state = found->second;
    const size_t tracked_old = std::min(old_bytes, device.coordinator_allocated_bytes);
    const size_t base = device.coordinator_allocated_bytes - tracked_old;
    const size_t remaining = device.budget_limit > base
        ? device.budget_limit - base : 0;
    const size_t tracked_new = std::min(new_bytes, remaining);
    device.coordinator_allocated_bytes = base + tracked_new;

    if (tracked_new >= tracked_old) {
        const size_t consumed = tracked_new - tracked_old;
        state.outstanding_bytes = consumed <= state.outstanding_bytes
            ? state.outstanding_bytes - consumed : 0;
    } else {
        const size_t released = tracked_old - tracked_new;
        state.outstanding_bytes = released <= SIZE_MAX - state.outstanding_bytes
            ? state.outstanding_bytes + released : SIZE_MAX;
    }
}

static void moe_cache_budget_unregister(moe_cache_device & device) {
    if (!device.budget_registered) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found != g_physical_budgets.end()) {
        moe_cache_physical_budget & state = found->second;
        if (device.budget_claimed) {
            const size_t unallocated = device.budget_limit > device.coordinator_allocated_bytes
                ? device.budget_limit - device.coordinator_allocated_bytes : 0;
            state.outstanding_bytes = unallocated <= state.outstanding_bytes
                ? state.outstanding_bytes - unallocated : 0;
            state.prepared = std::max(state.prepared - 1, 0);
        }
        moe_cache_budget_remove_participant(
                state, device.budget_reserve_bytes);
        if (state.participants == 0) {
            g_physical_budgets.erase(found);
        }
    }
    device.budget_registered = false;
    device.budget_claimed = false;
    device.budget_ready = false;
    device.budget_reserve_bytes = 0;
    device.budget_limit = 0;
    device.coordinator_allocated_bytes = 0;
}

static size_t moe_cache_budget_claim(
        moe_cache_session & session, moe_cache_device & device,
        size_t free_memory, size_t & reserve_bytes) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    reserve_bytes = 0;
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end() || !device.budget_registered) {
        return 0;
    }
    moe_cache_physical_budget & state = found->second;
    reserve_bytes = state.reserves.empty()
        ? 0 : *state.reserves.rbegin();
    const size_t available_after_reserve = free_memory > reserve_bytes
        ? free_memory - reserve_bytes : 0;
    const size_t unclaimed = available_after_reserve > state.outstanding_bytes
        ? available_after_reserve - state.outstanding_bytes : 0;
    const int remaining_participants = std::max(
            state.participants - state.prepared, 1);
    size_t claim = unclaimed / (size_t)remaining_participants;
    if (session.config.budget_mb > 0) {
        claim = std::min(claim, session.config.budget_mb << 20);
    }
    state.prepared++;
    state.outstanding_bytes += claim;
    device.budget_claimed = true;
    return claim;
}

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static int moe_cache_min_compute_capability(bool automatic) {
    int result = automatic ? moe_cache_cc_ampere : moe_cache_cc_forced_min;
    int64_t value = 0;
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        result = (int)value;
    }
    return result;
}

static size_t moe_cache_default_min_expert_bytes(int compute_capability) {
    return compute_capability >= moe_cache_cc_ampere
        ? moe_cache_expert_bytes_ampere_min
        : moe_cache_expert_bytes_pre_ampere_min;
}

static void moe_cache_apply_mode_defaults(moe_cache_config & config) {
    config.minimum_slab_bytes = config.automatic
        ? moe_cache_slab_bytes_auto_min : 0;
    if (!config.min_expert_explicit) {
        config.min_expert_bytes = config.automatic
            ? moe_cache_expert_bytes_ampere_min
            : moe_cache_expert_bytes_pre_ampere_min;
    }
    if (!config.max_batch_explicit) {
        config.max_batch = moe_cache_batch_max;
    }
    if (!config.overlap_cpu_rows_explicit) {
        config.overlap_cpu_rows = -1;
    }
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
        config.reserve_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
        config.min_expert_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, moe_cache_batch_max, value)) {
        config.max_batch = (int)value;
        config.max_batch_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
        config.inserts_per_plan_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
        config.admit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
        config.readmit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
        config.serial_fill_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DOWN_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_down_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", 0, 8, value)) {
        config.overlap_cpu_rows = (int)value;
        config.overlap_cpu_rows_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_EXPERT_PARALLEL",
            -1, moe_cache_expert_parallel_max, value)) {
        config.expert_parallel = (int)value;
    }
    moe_cache_apply_mode_defaults(config);
    config.min_compute_capability = moe_cache_min_compute_capability(config.automatic);
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    return config;
}

static size_t moe_cache_recommended_reserve_bytes(size_t total_vram) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t step = 128 * MiB;
    size_t reserve = total_vram / 100 * 6;
    reserve = reserve / step * step;
    reserve = std::max<size_t>(1024 * MiB, reserve);
    reserve = std::min<size_t>(3072 * MiB, reserve);
    return std::min(reserve, total_vram / 4);
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static uint64_t moe_cache_profile_tensor_hash(
        const char * name, size_t expert_size, int wtype, int64_t n_expert) {
    uint64_t hash = moe_cache_name_hash(name);
    const uint64_t fields[] = {
        (uint64_t)expert_size, (uint64_t)(uint32_t)wtype, (uint64_t)n_expert,
    };
    for (uint64_t field : fields) {
        hash ^= field;
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static uint32_t moe_cache_profile_u32(const unsigned char * data) {
    return uint32_t(data[0]) | uint32_t(data[1]) << 8 |
        uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}

static uint64_t moe_cache_profile_u64(const unsigned char * data) {
    return uint64_t(moe_cache_profile_u32(data)) |
        uint64_t(moe_cache_profile_u32(data + 4)) << 32;
}

static void moe_cache_profile_put_u32(std::vector<unsigned char> & out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back((unsigned char)(value >> shift));
    }
}

static void moe_cache_profile_put_u64(std::vector<unsigned char> & out, uint64_t value) {
    moe_cache_profile_put_u32(out, (uint32_t)value);
    moe_cache_profile_put_u32(out, (uint32_t)(value >> 32));
}

static uint64_t moe_cache_profile_checksum(const unsigned char * data, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static void moe_cache_profile_load(moe_cache_session & session, bool announce = true) {
    if (session.config.profile_path.empty()) {
        return;
    }
    std::ifstream input(session.config.profile_path, std::ios::binary | std::ios::ate);
    if (!input) {
        return;
    }
    const std::streamoff end = input.tellg();
    constexpr size_t header_v1_size = 24;
    constexpr size_t header_v2_size = 32;
    constexpr size_t record_size = 16;
    constexpr uint32_t max_records = 1u << 20;
    if (end < (std::streamoff)header_v1_size || end > (std::streamoff)(header_v2_size +
            (size_t)max_records * record_size)) {
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid expert profile size: %s\n",
                    session.config.profile_path.c_str());
        }
        return;
    }
    std::vector<unsigned char> bytes((size_t)end);
    input.seekg(0);
    if (!input.read((char *)bytes.data(), bytes.size())) {
        return;
    }
    static constexpr unsigned char magic[8] = {'G','G','M','L','M','H','C','1'};
    const uint32_t version = moe_cache_profile_u32(bytes.data() + 8);
    const uint32_t count = moe_cache_profile_u32(bytes.data() + 12);
    const size_t header_size = version == 1 ? header_v1_size :
        version == 2 ? header_v2_size : 0;
    const uint64_t generation = version == 2 && bytes.size() >= header_v2_size
        ? moe_cache_profile_u64(bytes.data() + 16) : 0;
    const uint64_t checksum = header_size != 0 && bytes.size() >= header_size
        ? moe_cache_profile_u64(bytes.data() + (version == 2 ? 24 : 16)) : 0;
    if (memcmp(bytes.data(), magic, sizeof(magic)) != 0 || header_size == 0 ||
            count > max_records || bytes.size() != header_size + (size_t)count * record_size ||
            checksum != moe_cache_profile_checksum(bytes.data() + header_size,
                bytes.size() - header_size)) {
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] ignoring corrupt or incompatible expert profile: %s\n",
                    session.config.profile_path.c_str());
        }
        return;
    }
    try {
        session.profile_generation = generation;
        session.profile_heat.reserve(count);
        const unsigned char * record = bytes.data() + header_size;
        for (uint32_t i = 0; i < count; ++i, record += record_size) {
            const moe_cache_profile_key key{
                moe_cache_profile_u64(record),
                (int32_t)moe_cache_profile_u32(record + 8),
            };
            const uint32_t heat = moe_cache_profile_u32(record + 12);
            if (key.tensor != 0 && key.expert >= 0 && heat > 0) {
                session.profile_heat[key] = heat;
            }
        }
        std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, int32_t>>> ranked;
        for (const auto & item : session.profile_heat) {
            ranked[item.first.tensor].push_back({item.second, item.first.expert});
        }
        session.profile_hot.reserve(ranked.size());
        for (auto & tensor : ranked) {
            std::sort(tensor.second.begin(), tensor.second.end(), [](const auto & a, const auto & b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            if (tensor.second.size() > 32) {
                tensor.second.resize(32);
            }
            std::vector<int32_t> & hot = session.profile_hot[tensor.first];
            hot.reserve(tensor.second.size());
            for (const auto & item : tensor.second) {
                hot.push_back(item.second);
            }
        }
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] loaded %zu expert heat entries from %s\n",
                    session.profile_heat.size(), session.config.profile_path.c_str());
        }
    } catch (...) {
        session.profile_heat.clear();
        session.profile_hot.clear();
        session.profile_generation = 0;
    }
}

static std::mutex g_profile_io_mu;

struct moe_cache_profile_file_lock {
    bool locked = false;
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
    int fd = -1;
#endif

    explicit moe_cache_profile_file_lock(const std::string & path) {
        const std::string lock_path = path + ".lock";
#if defined(_WIN32)
        handle = CreateFileA(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            OVERLAPPED overlap = {};
            locked = LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0,
                    MAXDWORD, MAXDWORD, &overlap) != 0;
        }
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
        locked = fd >= 0 && flock(fd, LOCK_EX) == 0;
#else
        locked = true;
#endif
    }

    ~moe_cache_profile_file_lock() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) {
            if (locked) {
                OVERLAPPED overlap = {};
                UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlap);
            }
            CloseHandle(handle);
        }
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        if (fd >= 0) {
            if (locked) {
                flock(fd, LOCK_UN);
            }
            close(fd);
        }
#endif
    }
};

static void moe_cache_profile_save(moe_cache_session & session) {
    if (session.config.profile_path.empty() || session.observed_heat.empty()) {
        return;
    }
    try {
        std::lock_guard<std::mutex> process_lock(g_profile_io_mu);
        moe_cache_profile_file_lock file_lock(session.config.profile_path);
        if (!file_lock.locked) {
            return;
        }
        moe_cache_session latest;
        latest.config.profile_path = session.config.profile_path;
        moe_cache_profile_load(latest, false);
        auto & merged_heat = latest.profile_heat;
        if (latest.profile_generation == session.profile_generation) {
            for (auto it = merged_heat.begin(); it != merged_heat.end();) {
                it->second /= 2;
                if (it->second == 0) {
                    it = merged_heat.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const auto & tensor : session.observed_heat) {
            for (size_t expert = 0; expert < tensor.second.size(); ++expert) {
                const uint32_t observed = tensor.second[expert];
                if (observed == 0 || expert > INT32_MAX) {
                    continue;
                }
                uint32_t & heat = merged_heat[
                    {tensor.first, (int32_t)expert}];
                const uint64_t merged = uint64_t(heat) + observed;
                heat = (uint32_t)std::min<uint64_t>(merged, UINT32_MAX);
            }
        }
        std::vector<std::pair<moe_cache_profile_key, uint32_t>> records(
                merged_heat.begin(), merged_heat.end());
        std::sort(records.begin(), records.end(), [](const auto & a, const auto & b) {
            return a.first.tensor < b.first.tensor ||
                (a.first.tensor == b.first.tensor && a.first.expert < b.first.expert);
        });
        if (records.size() > (1u << 20)) {
            records.resize(1u << 20);
        }
        std::vector<unsigned char> payload;
        payload.reserve(records.size() * 16);
        for (const auto & record : records) {
            moe_cache_profile_put_u64(payload, record.first.tensor);
            moe_cache_profile_put_u32(payload, (uint32_t)record.first.expert);
            moe_cache_profile_put_u32(payload, record.second);
        }
        std::vector<unsigned char> bytes;
        bytes.insert(bytes.end(), {'G','G','M','L','M','H','C','1'});
        moe_cache_profile_put_u32(bytes, 2);
        moe_cache_profile_put_u32(bytes, (uint32_t)records.size());
        moe_cache_profile_put_u64(bytes, latest.profile_generation + 1);
        moe_cache_profile_put_u64(bytes,
                moe_cache_profile_checksum(payload.data(), payload.size()));
        bytes.insert(bytes.end(), payload.begin(), payload.end());

        char suffix[80];
#if defined(_WIN32)
        const long process_id = _getpid();
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        const long process_id = (long)getpid();
#else
        const long process_id = 0;
#endif
        snprintf(suffix, sizeof(suffix), ".tmp-%ld-%p", process_id, (void *)&session);
        const std::string temporary = session.config.profile_path + suffix;
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.write((const char *)bytes.data(), bytes.size())) {
            output.close();
            std::remove(temporary.c_str());
            return;
        }
        output.close();
#ifdef _WIN32
        const bool replaced = MoveFileExA(temporary.c_str(),
                session.config.profile_path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        const bool replaced = std::rename(
                temporary.c_str(), session.config.profile_path.c_str()) == 0;
#endif
        if (replaced) {
            MOE_CACHE_LOG("[moe-cache] saved %zu expert heat entries to %s\n",
                    records.size(), session.config.profile_path.c_str());
        } else {
            std::remove(temporary.c_str());
        }
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] could not update expert profile: %s\n",
                session.config.profile_path.c_str());
    }
}

static void moe_cache_profile_observe_locked(
        moe_cache_session & session, uint64_t tensor,
        const int32_t * experts, int n_experts, int64_t expert_count) {
    if (tensor == 0 || !experts || n_experts <= 0 || expert_count <= 0 ||
            expert_count > INT32_MAX || session.config.profile_path.empty()) {
        return;
    }
    try {
        std::vector<uint32_t> & heat = session.observed_heat[tensor];
        if (heat.empty()) {
            heat.resize((size_t)expert_count, 0);
        } else if (heat.size() != (size_t)expert_count) {
            return;
        }
        for (int i = 0; i < n_experts; ++i) {
            const int32_t expert = experts[i];
            if (expert >= 0 && expert < expert_count && heat[expert] < UINT32_MAX) {
                heat[expert]++;
            }
        }
    } catch (...) {
        // Profiling must never disable or perturb inference.
    }
}

static const std::vector<int32_t> & moe_cache_profile_hot_experts(
        const moe_cache_session & session, uint64_t tensor) {
    static const std::vector<int32_t> empty;
    const auto found = session.profile_hot.find(tensor);
    return found == session.profile_hot.end() ? empty : found->second;
}

static void moe_cache_profile_advise(
        const void * host_base, size_t expert_size,
        const std::vector<int32_t> & experts, int64_t n_expert) {
#if defined(__linux__)
    const long page_long = sysconf(_SC_PAGESIZE);
    if (!host_base || expert_size == 0 || page_long <= 0) {
        return;
    }
    const uintptr_t page = (uintptr_t)page_long;
    for (int32_t expert : experts) {
        if (expert < 0 || expert >= n_expert) {
            continue;
        }
        const uintptr_t first = (uintptr_t)host_base + (size_t)expert * expert_size;
        if (first < (uintptr_t)host_base || expert_size > UINTPTR_MAX - first) {
            continue;
        }
        const uintptr_t begin = first - first % page;
        const uintptr_t end_raw = first + expert_size;
        if (end_raw > UINTPTR_MAX - (page - 1)) {
            continue;
        }
        const uintptr_t end = (end_raw + page - 1) / page * page;
        if (end > begin) {
            (void)madvise((void *)begin, end - begin, MADV_WILLNEED);
        }
    }
#else
    (void)host_base;
    (void)expert_size;
    (void)experts;
    (void)n_expert;
#endif
}

static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static bool moe_cache_tensor_name_supported(const char * name) {
    return strstr(name, "_exps") || strstr(name, "_chexps");
}

static bool moe_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_0_G128:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static void moe_cache_lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) {
        pool.slots[slot.prev].next = slot.next;
    } else {
        pool.lru_head = slot.next;
    }
    if (slot.next >= 0) {
        pool.slots[slot.next].prev = slot.prev;
    } else {
        pool.lru_tail = slot.prev;
    }
    slot.prev = -1;
    slot.next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = pool.lru_tail;
    slot.next = -1;
    if (pool.lru_tail >= 0) {
        pool.slots[pool.lru_tail].next = index;
    } else {
        pool.lru_head = index;
    }
    pool.lru_tail = index;
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
    }
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.state == moe_cache_slot_state::valid) {
        moe_cache_lru_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    slot.key = {};
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::free;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
}

static bool moe_cache_grow_device(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    cudaError_t error = cudaMalloc(&fresh, requested);
    if (error == cudaErrorMemoryAllocation && *pointer) {
        (void)cudaGetLastError();
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            return false;
        }
        moe_cache_budget_allocation(device, capacity, false);
        *pointer = nullptr;
        capacity = 0;
        error = cudaMalloc(&fresh, requested);
    }
    if (!moe_cache_cuda_ok(device, error, operation, false)) {
        return false;
    }
    if (*pointer) {
        const size_t old_capacity = capacity;
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            cudaFree(fresh);
            return false;
        }
        moe_cache_budget_reallocation(device, old_capacity, requested);
    } else {
        moe_cache_budget_allocation(device, requested, true);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = moe_cache_node_rows_max;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t ids_bytes = 4 * max_rows * sizeof(int32_t);
    const size_t act_bytes = max_rows * (size_t)n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return false;
    }
    const size_t capacities[] = {
        moe_cache_growth_capacity(0, ids_bytes + act_bytes),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.input, additional->input) : current.input,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static int moe_cache_query_config(
        int automatic, size_t budget_mib, ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability(config.automatic);
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->reserve_explicit = config.reserve_explicit;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->min_devices = config.automatic ? 2 : 1;
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    result->expert_parallel = config.expert_parallel;
    result->profile_path = nullptr;
    return 1;
}

static int moe_cache_query_device(
        void * opaque, const ggml_moe_cache_config * config,
        ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !ggml_moe_cache.owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != ggml_moe_cache.owner) {
        return 0;
    }

    int logical = -1;
    const size_t count = ggml_backend_reg_dev_count(reg);
    for (size_t index = 0; index < count; index++) {
        if (ggml_backend_reg_dev_get(reg, index) == device) {
            logical = (int)index;
            break;
        }
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (logical < 0 || logical >= info.device_count) {
        return 0;
    }

    const auto & cuda_device = info.devices[logical];
    if (cuda_device.cc < config->min_compute_capability) {
        return 0;
    }
    result->logical_device = logical;
    result->physical_device = cuda_device.physical_device;
    result->compute_capability = cuda_device.cc;
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(cuda_device.cc);
    result->recommended_reserve_bytes = ggml_moe_cache_effective_reserve_bytes(
            config->reserve_explicit,
            config->reserve_bytes,
            moe_cache_recommended_reserve_bytes(cuda_device.total_vram));
    return 1;
}

static int moe_cache_query_shape(
        int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
        size_t expert_size,
        ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    moe_cache_scratch scratch;
    if (!moe_cache_scratch_requirements(n_in, n_out, scratch)) {
        return 0;
    }
    const size_t scratch_bytes = moe_cache_scratch_total(scratch);
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (scratch_bytes == SIZE_MAX || pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}

static bool moe_cache_grow_host(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMallocHost(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFreeHost(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty();
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            device->inflight_source = job.source;
            device->inflight_bytes = job.bytes;
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->logical);

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream) {
            int least_priority = 0;
            int greatest_priority = 0;
            error = cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority);
            if (error == cudaSuccess) {
                error = cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, least_priority);
            } else {
                (void)cudaGetLastError();
                error = cudaStreamCreateWithFlags(
                        &stream, cudaStreamNonBlocking);
            }
        }
        if (error == cudaSuccess && stage_capacity < job.bytes) {
            char * fresh = nullptr;
            cudaError_t alloc_error = cudaMallocHost((void **)&fresh, job.bytes);
            if (alloc_error == cudaSuccess) {
                if (stage) {
                    cudaFreeHost(stage);
                }
                stage = fresh;
                stage_capacity = job.bytes;
            } else {
                (void)cudaGetLastError();
            }
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                memcpy(stage, job.source, job.bytes);
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess) {
                error = cudaMemcpy(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice);
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->inflight = false;
            device->inflight_source = nullptr;
            device->inflight_bytes = 0;

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (error == cudaSuccess) {
                        slot.state = moe_cache_slot_state::valid;
                        moe_cache_lru_push_back(*pool, job.slot);
                        device->demand_count.erase(job.key);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        device->fill_failures++;
                    }
                }
            }
            session->idle_cv.notify_all();
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (stage) {
        cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device);
        device.worker_started = true;
        return true;
    } catch (...) {
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

static int moe_cache_find_pool(
        const moe_cache_device & device, size_t expert_size, int wtype) {
    for (int index = 0; index < (int)device.pools.size(); index++) {
        const moe_cache_pool & pool = *device.pools[index];
        if (pool.expert_size == expert_size && pool.wtype == wtype) {
            return index;
        }
    }
    return -1;
}

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.budget_ready) {
        return device.budget_limit > 0;
    }
    device.budget_ready = true;

    ggml_cuda_set_device(device.logical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

    size_t reserve_bytes = 0;
    const size_t available = moe_cache_budget_claim(
            session, device, free_memory, reserve_bytes);
    device.budget_limit = available;

    if (session.config.budget_mb > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: cap=%zu MiB granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, session.config.budget_mb, available >> 20,
                free_memory >> 20, reserve_bytes >> 20);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: policy=free-minus-reserve granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, available >> 20, free_memory >> 20,
                reserve_bytes >> 20);
    }
    return available > 0;
}

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_entries == 0) {
        return false;
    }
    shape.finished = true;

    size_t slots_by_budget = budget / shape.expert_size;
    size_t slot_count = std::min<size_t>(
            slots_by_budget,
            (size_t)std::min<uint64_t>(shape.n_entries, INT_MAX));
    const size_t type_size = ggml_type_size((ggml_type)shape.wtype);
    if (type_size == 0 || shape.expert_size % type_size != 0) {
        return false;
    }
    const size_t stride_blocks = shape.expert_size / type_size;
    if (stride_blocks == 0) {
        return false;
    }
    slot_count = std::min(slot_count, (size_t)INT_MAX / stride_blocks);
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < moe_cache_pool_slots_min) {
        return false;
    }

    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= moe_cache_pool_slots_min) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < moe_cache_pool_slots_min) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }
    const size_t allocated = slot_count * shape.expert_size;
    moe_cache_budget_allocation(device, allocated, true);

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }
    try {
        pool->expert_size = shape.expert_size;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        pool->covers_all_entries = (uint64_t)slot_count >= shape.n_entries;
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        cudaFree(device.pools.back()->slab);
        moe_cache_budget_allocation(device, allocated, false);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: type=%s expert=%zu KiB slots=%zu entries=%llu coverage=%s total=%zu MiB\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.wtype),
            shape.expert_size >> 10, slot_count,
            (unsigned long long)shape.n_entries,
            device.pools.back()->covers_all_entries ? "complete" : "partial",
            allocated >> 20);
    bool expected = false;
    if (session.enabled_announced.compare_exchange_strong(expected, true)) {
        MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on CUDA%d\n", device.physical);
    }

    return true;
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve);
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    long double total_weight = 0.0;
    size_t minimum_remaining = 0;
    for (moe_cache_shape & shape : device.shapes) {
        if (!shape.finished && shape.n_tensors > 0) {
            pending.push_back(&shape);
            total_weight +=
                (long double)shape.expert_size * (long double)shape.n_tensors;
            const size_t minimum = shape.expert_size * moe_cache_pool_slots_min;
            minimum_remaining = minimum <= SIZE_MAX - minimum_remaining
                ? minimum_remaining + minimum : SIZE_MAX;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const long double lhs_weight =
            (long double)lhs->expert_size * (long double)lhs->n_tensors;
        const long double rhs_weight =
            (long double)rhs->expert_size * (long double)rhs->n_tensors;
        return lhs_weight > rhs_weight;
    });

    auto weighted_share = [](size_t available, long double weight, long double total) {
        if (available == 0 || weight <= 0.0 || total <= 0.0) {
            return (size_t)0;
        }
        const long double value = (long double)available * weight / total;
        return value >= (long double)available ? available : (size_t)value;
    };
    const bool complete_pools = minimum_remaining <= remaining;
    for (moe_cache_shape * shape : pending) {
        const long double weight =
            (long double)shape->expert_size * (long double)shape->n_tensors;
        const size_t minimum = shape->expert_size * moe_cache_pool_slots_min;
        size_t share = weighted_share(remaining, weight, total_weight);
        if (complete_pools) {
            const size_t extra = remaining - minimum_remaining;
            const size_t bonus = weighted_share(extra, weight, total_weight);
            share = minimum + bonus;
        } else if (share < minimum && remaining >= minimum) {
            share = minimum;
        }
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
        if (complete_pools) {
            minimum_remaining -= minimum;
        }
    }
}

static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert) {
    int pool = moe_cache_find_pool(device, expert_size, wtype);
    if (pool >= 0 && !device.pools[pool]->covers_all_entries) {
        return pool;
    }
    if (pool >= 0 && device.seen_tensors.find(host_base) != device.seen_tensors.end()) {
        return pool;
    }

    moe_cache_shape * shape = nullptr;
    for (moe_cache_shape & candidate : device.shapes) {
        if (candidate.expert_size == expert_size && candidate.wtype == wtype) {
            shape = &candidate;
            break;
        }
    }
    if (!shape) {
        device.shapes.push_back({expert_size, wtype, 0, 0, -1, false});
        shape = &device.shapes.back();
        shape->pool = pool;
    }

    const bool first_visit = device.seen_tensors.emplace(
            host_base,
            moe_cache_seen_tensor{tensor_size, expert_size, wtype, n_expert}).second;
    if (first_visit) {
        if (shape->n_tensors == 0 && shape->pool < 0) {
            shape->finished = false;
        }
        const uint64_t entries = (uint64_t)n_expert;
        shape->n_entries = entries <= UINT64_MAX - shape->n_entries
            ? shape->n_entries + entries : UINT64_MAX;
        shape->n_tensors++;
        device.visits_since_new_tensor = 0;
    } else {
        if (device.visits_since_new_tensor < SIZE_MAX) {
            device.visits_since_new_tensor++;
        }
    }

    if (pool >= 0) {
        device.pools[pool]->covers_all_entries =
            (uint64_t)device.pools[pool]->n_slots >= shape->n_entries;
        return pool;
    }

    if (device.visits_since_new_tensor < device.seen_tensors.size()) {
        return -1;
    }

    moe_cache_build_pending(session, device);
    return moe_cache_find_pool(device, expert_size, wtype);
}

static void moe_cache_log_stats(moe_cache_device & device) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = device.hits + device.misses;
    MOE_CACHE_LOG("[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld act-dedup=%lld cpu-overlap=%lld fusion=%lld/%lld pairs=%lld/%lld/%lld/%lld fusion-attempts=%lld fusion-nodes=%lld full-fusion=%lld/%lld down-mmv-dedicated=%lld bypass=%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, device.inserts, device.fills, device.fill_failures,
            device.evictions, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures,
            device.activation_dedup, device.overlap_rows,
            device.fused_rows, device.fused_candidates,
            device.pair_both, device.pair_up_only,
            device.pair_gate_only, device.pair_neither,
            device.fused_attempts,
            device.fused_nodes,
            device.full_fused_rows, device.full_fused_nodes,
            device.dedicated_down_mmv_dispatches,
            device.contention_bypasses);
}

static void moe_cache_log_configuration(moe_cache_session & session) {
    bool expected = false;
    if (!session.config_announced.compare_exchange_strong(expected, true)) {
        return;
    }

    const std::string budget = session.config.budget_mb > 0
        ? std::to_string(session.config.budget_mb) + " MiB cap"
        : "free-minus-reserve";
    const std::string overlap_cpu_rows = session.config.overlap_cpu_rows < 0
        ? "auto" : std::to_string(session.config.overlap_cpu_rows);
    const std::string admission = session.config.admit_after_explicit
        ? std::to_string(session.config.admit_after) + "-fixed/" +
            std::to_string(session.config.readmit_after) + "-replace"
        : "1-complete/" + std::to_string(session.config.admit_after) +
            "-partial/" + std::to_string(session.config.readmit_after) + "-replace";
    MOE_CACHE_LOG("[moe-cache] configured: mode=%s devices=%zu budget=%s reserve=%zu MiB min-slab=%zu MiB min-expert=%zu KiB max-batch=%d inserts=%d admit=%s cpu-overlap=%s fills=%s expert-parallel=%d down-mmv=%s\n",
            session.config.automatic ? "auto" : "on",
            session.devices.size(), budget.c_str(),
            session.config.reserve_mb,
            session.config.minimum_slab_bytes >> 20,
            session.config.min_expert_bytes >> 10,
            session.config.max_batch,
            session.config.inserts_per_plan,
            admission.c_str(),
            overlap_cpu_rows.c_str(),
            session.config.serial_fill ? "serial" : "parallel",
            session.config.expert_parallel,
            session.config.force_dedicated_down_mmv ? "dedicated" : "generic");
}

static void * moe_cache_session_create(
        void * const * backends, int n_backends,
        const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->reserve_explicit < 0 ||
                supplied_config->reserve_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8 ||
                supplied_config->expert_parallel < -1 ||
                supplied_config->expert_parallel > moe_cache_expert_parallel_max) {
                return nullptr;
            }
            config.enabled = true;
            config.automatic = supplied_config->min_devices > 1;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.reserve_explicit = supplied_config->reserve_explicit;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
            config.expert_parallel = supplied_config->expert_parallel;
            config.profile_path = supplied_config->profile_path
                ? supplied_config->profile_path : "";
        }
        if (!config.enabled) {
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        size_t default_min_expert_bytes = moe_cache_expert_bytes_ampere_min;
        int minimum_capability = INT_MAX;
        const auto add_backend = [&](ggml_backend_t backend) {
            if (!backend || !ggml_backend_is_cuda(backend) ||
                (int)session->devices.size() >= session->config.max_devices) {
                return;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int logical = context->device;
            const ggml_cuda_device_info & info = ggml_cuda_info();
            if (logical < 0 || logical >= info.device_count) {
                return;
            }
            const int physical = info.devices[logical].physical_device;
            if (!seen_devices.insert(physical).second) {
                return;
            }

            ggml_cuda_set_device(logical);
            const int capability = info.devices[logical].cc;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d is below %d\n",
                        physical, capability,
                        session->config.min_compute_capability);
                return;
            }

            default_min_expert_bytes = std::max(
                    default_min_expert_bytes,
                    moe_cache_default_min_expert_bytes(capability));
            minimum_capability = std::min(minimum_capability, capability);
            session->devices.emplace_back(new moe_cache_device(logical, physical));
            session->devices.back()->automatic_reserve_bytes =
                moe_cache_recommended_reserve_bytes(info.devices[logical].total_vram);
        };
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (ggml_backend_is_meta(backend)) {
                const size_t n_simple = ggml_backend_meta_n_backends(backend);
                for (size_t i = 0; i < n_simple; ++i) {
                    add_backend(ggml_backend_meta_simple_backend(backend, i));
                }
            } else {
                add_backend(backend);
            }
        }

        if (session->devices.empty() ||
            (session->config.automatic && session->devices.size() < 2)) {
            return nullptr;
        }
        if (!session->config.reserve_explicit) {
            size_t reserve_bytes = 0;
            for (const auto & device_ptr : session->devices) {
                reserve_bytes = std::max(reserve_bytes,
                        device_ptr->automatic_reserve_bytes);
            }
            // Configuration logging reports the largest automatic reserve;
            // budget registration retains each device's own value.
            session->config.reserve_mb = reserve_bytes >> 20;
        }
        if (!session->config.min_expert_explicit) {
            session->config.min_expert_bytes = default_min_expert_bytes;
        }
        if (!session->config.serial_fill_explicit) {
            session->config.serial_fill =
                session->devices.size() < 2 || minimum_capability < moe_cache_cc_ampere;
        }
        if (session->config.expert_parallel < 0) {
            session->config.expert_parallel = std::min<int>(
                    3, session->devices.size());
        }
        if (session->config.expert_parallel > 0) {
            // Expert-parallel verification probes more routed rows per node and
            // consumes complete up/gate/down bundles. Admit enough rows to warm a
            // bundle promptly, then require sustained demand before displacing a
            // resident bundle. Serial cache policy keeps its established defaults.
            if (!session->config.inserts_per_plan_explicit) {
                session->config.inserts_per_plan = 16;
            }
            if (!session->config.readmit_after_explicit) {
                session->config.readmit_after = 40;
            }
        }
        if (!moe_cache_budget_register(*session)) {
            return nullptr;
        }
        moe_cache_profile_load(*session);

        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        } catch (...) {
            for (const auto & device_ptr : session->devices) {
                moe_cache_budget_unregister(*device_ptr);
            }
            throw;
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (auto it = device.queue.begin(); it != device.queue.end();) {
        if (all || moe_cache_ranges_overlap(it->source, it->bytes, base, size)) {
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = device.queue.erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.logical);
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            moe_cache_budget_allocation(
                    device, (size_t)pool_ptr->n_slots * pool_ptr->expert_size, false);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_input) {
        cudaFree(device.d_input);
        moe_cache_budget_allocation(device, device.d_input_cap, false);
        device.d_input = nullptr;
    }
    if (device.d_act_q8) {
        cudaFree(device.d_act_q8);
        moe_cache_budget_allocation(device, device.act_q8_cap, false);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        cudaFree(device.d_out);
        moe_cache_budget_allocation(device, device.d_out_cap, false);
        device.d_out = nullptr;
    }
    if (device.h_input) {
        cudaFreeHost(device.h_input);
        device.h_input = nullptr;
    }
    if (device.h_out) {
        cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    device.pools.clear();
    device.allocated_bytes = 0;
    device.d_input_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.h_input_cap = 0;
    device.h_out_cap = 0;
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
    }
    moe_cache_profile_save(*session);

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    delete session;
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    session->active_scopes++;
}

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}

static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
        int64_t n_tokens, int64_t n_rows) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    // Expert-parallel mode owns discovery and admission through the complete
    // up/gate/down path. Letting the single-device fallback see these nodes would
    // allocate a pool after the first tensor is revisited, before all layer shapes
    // have been counted, and permanently strand most of that device's capacity.
    if (session->config.expert_parallel && session->devices.size() >= 2) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (!name || !host_base ||
        !moe_cache_tensor_name_supported(name) || n_tokens < 1 ||
        expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }
    if (n_rows < n_tokens || n_rows % n_tokens != 0) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token node above max-batch=%d; prompt batches normally use the CPU path, while larger decode or speculative batches will not use the cache\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }
    if (n_rows > moe_cache_node_rows_max) {
        bool expected = false;
        if (session->row_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-row node above row limit=%d; the stock CPU path remains active\n",
                    (long long)n_rows, moe_cache_node_rows_max);
        }
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)n_expert * expert_size;
    moe_cache_scratch scratch_requirements;
    if (!moe_cache_scratch_requirements(
            n_in, n_out, scratch_requirements)) {
        return nullptr;
    }
    int layer = -1;
    const bool has_layer = moe_cache_layer_number(name, layer);
    const uint64_t profile_key = session->config.profile_path.empty() ? 0 :
        moe_cache_profile_tensor_hash(name, expert_size, wtype, n_expert);

    moe_cache_device * selected = nullptr;
    int pool_index = -1;
    moe_cache_pool * pool = nullptr;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (session->stopping) {
            return nullptr;
        }

        int budget_devices = 0;
        int slab_devices = 0;
        int eligible_devices = 0;
        const size_t minimum_pool = expert_size * moe_cache_pool_slots_min;
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & candidate = *device_ptr;
            if (candidate.dead.load() ||
                !moe_cache_prepare_budget(*session, candidate)) {
                continue;
            }
            budget_devices++;
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (slab_limit >= session->config.minimum_slab_bytes) {
                slab_devices++;
            }
            if (slab_limit >= minimum_pool &&
                slab_limit >= session->config.minimum_slab_bytes) {
                eligible_devices++;
            }
        }
        if (budget_devices == 0 ||
            (session->config.automatic &&
             (budget_devices < 2 || slab_devices < 2))) {
            if (session->config.automatic && slab_devices < 2) {
                MOE_CACHE_LOG("[moe-cache] session dormant: only %d devices satisfy the %zu MiB automatic slab floor\n",
                        slab_devices, session->config.minimum_slab_bytes >> 20);
            }
            session->dormant.store(true);
            lock.unlock();
            for (const auto & device_ptr : session->devices) {
                moe_cache_trim_session(*session, device_ptr->physical);
            }
            return nullptr;
        }
        if (session->config.automatic && eligible_devices < 2) {
            return nullptr;
        }

        auto route_weight = [&](moe_cache_device & candidate) -> size_t {
            if (candidate.dead.load() || !candidate.budget_ready) {
                return 0;
            }
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (candidate.allocated_bytes > slab_limit ||
                slab_limit < session->config.minimum_slab_bytes) {
                return 0;
            }
            if (moe_cache_find_pool(candidate, expert_size, wtype) >= 0) {
                return slab_limit;
            }
            const size_t available = slab_limit - candidate.allocated_bytes;
            return available >= minimum_pool ? available : 0;
        };

        bool tensor_override = !has_layer;
        auto find_routed_device = [&](int physical) -> moe_cache_device * {
            for (const auto & device_ptr : session->devices) {
                if (device_ptr->physical == physical) {
                    return device_ptr.get();
                }
            }
            return nullptr;
        };

        auto tensor_route = session->tensor_devices.find(host_base);
        if (tensor_route != session->tensor_devices.end()) {
            selected = find_routed_device(tensor_route->second);
            if (!selected || selected->dead.load() ||
                route_weight(*selected) == 0) {
                session->tensor_devices.erase(tensor_route);
                selected = nullptr;
            } else {
                tensor_override = true;
            }
        }

        if (!selected && has_layer) {
            auto layer_route = session->layer_devices.find(layer);
            if (layer_route != session->layer_devices.end()) {
                selected = find_routed_device(layer_route->second);
                if (!selected || selected->dead.load()) {
                    session->layer_devices.erase(layer);
                    selected = nullptr;
                } else if (route_weight(*selected) == 0) {
                    selected = nullptr;
                    tensor_override = true;
                }
            }
        }

        uint64_t total_weight = 0;
        if (!selected) {
            for (const auto & device_ptr : session->devices) {
                const size_t weight = route_weight(*device_ptr);
                if (weight > UINT64_MAX - total_weight) {
                    total_weight = UINT64_MAX;
                } else {
                    total_weight += weight;
                }
            }
        }
        if (!selected && total_weight == 0) {
            return nullptr;
        }

        if (!selected && has_layer && !tensor_override) {
            long double best_score = -1.0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                size_t assigned = 0;
                for (const auto & route : session->layer_devices) {
                    assigned += route.second == candidate.physical;
                }
                const long double score =
                    (long double)weight / (long double)(assigned + 1);
                if (!selected || score > best_score) {
                    selected = &candidate;
                    best_score = score;
                }
            }
            if (selected) {
                try {
                    session->layer_devices.emplace(layer, selected->physical);
                } catch (...) {
                    return nullptr;
                }
            }
        }

        const uint64_t hash = has_layer
            ? ((uint64_t)(uint32_t)layer + 1) * 0x9e3779b97f4a7c15ULL
            : moe_cache_name_hash(name);
        if (!selected) {
            const uint64_t target = hash % total_weight;
            uint64_t cumulative = 0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                cumulative = weight > UINT64_MAX - cumulative
                    ? UINT64_MAX : cumulative + weight;
                if (target < cumulative) {
                    selected = &candidate;
                    break;
                }
            }
            if (!selected) {
                return nullptr;
            }
            try {
                if (tensor_override) {
                    session->tensor_devices.emplace(host_base, selected->physical);
                } else {
                    session->layer_devices.emplace(layer, selected->physical);
                }
            } catch (...) {
                return nullptr;
            }
        }

        const size_t selected_scratch = moe_cache_scratch_total(
                selected->scratch_reserve, &scratch_requirements);
        if (selected_scratch >= selected->budget_limit ||
            minimum_pool > selected->budget_limit - selected_scratch) {
            return nullptr;
        }
        selected->scratch_reserve.input = std::max(
                selected->scratch_reserve.input, scratch_requirements.input);
        selected->scratch_reserve.q8 = std::max(
                selected->scratch_reserve.q8, scratch_requirements.q8);
        selected->scratch_reserve.out = std::max(
                selected->scratch_reserve.out, scratch_requirements.out);

        try {
            pool_index = moe_cache_discover_pool(
                    *session, *selected, host_base, tensor_size, expert_size, wtype, n_expert);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] disabled one node after host allocation failure\n");
            return nullptr;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size() ||
            !selected->pools[pool_index]->slab) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
    }

    moe_cache_device & device = *selected;
    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                device.dispatch_mu, std::try_to_lock);
    } catch (...) {
        device.contention_bypasses++;
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        device.contention_bypasses++;
    }
    if (!dispatch_lock.owns_lock() || device.dead.load()) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    node->session = session;
    node->device = &device;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_mid = n_out;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    node->profile_keys[0] = profile_key;
    node->dispatch_lock = std::move(dispatch_lock);
    return node.release();
}

static int moe_cache_overlap_rows(const moe_cache_node & node, int n_ids) {
    const int configured = node.session->config.overlap_cpu_rows;
    if (configured >= 0) {
        return std::min(configured, std::max(0, n_ids - 1));
    }
    if (n_ids <= 1 || node.n_tokens <= 0 || n_ids % node.n_tokens != 0) {
        return 0;
    }

    const int n_tokens = (int) node.n_tokens;
    const int top_k = n_ids / n_tokens;
    // Bound CPU work by expert bytes, token count, and one quarter of the node.
    const size_t work_budget = moe_cache_overlap_bytes_per_token * (size_t) n_tokens;
    int rows = (int) std::min<size_t>(work_budget / node.expert_size, INT_MAX);
    rows = std::max(rows, 1);
    if (top_k >= 8) {
        rows = std::max(rows, 2);
    }

    rows = std::min(rows, std::max(1, n_ids / 4));
    rows = std::min(rows, std::max(2, n_tokens));
    return std::min(rows, n_ids - 1);
}

static bool moe_cache_enqueue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker, bool allow_eviction) {
    const moe_cache_key key{host_base, expert};
    const size_t queue_limit = session.config.queue_mb << 20;
    if (inserts_left <= 0 || (int)device.queue.size() >= session.config.queue_max ||
        expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
        device.insert_skips++;
        return false;
    }

    int slot_index = -1;
    if (!pool.free_slots.empty()) {
        slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();
    } else if (allow_eviction) {
        int candidate = pool.lru_head;
        while (candidate >= 0 && pool.slots[candidate].readers > 0) {
            candidate = pool.slots[candidate].next;
        }
        if (candidate < 0) {
            device.insert_skips++;
            return false;
        }
        slot_index = candidate;
        moe_cache_slot_reset(pool, slot_index, false);
        device.evictions++;
    } else {
        return false;
    }

    moe_cache_slot & slot = pool.slots[slot_index];
    slot.key = key;
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::copying;
    try {
        const auto inserted = pool.map.emplace(key, slot_index);
        if (!inserted.second) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            return false;
        }

        const void * source =
            (const char *)host_base + (size_t)expert * expert_size;
        device.queue.push_back({
                pool_index, slot_index, slot.generation,
                key, source, expert_size});
        device.queued_bytes += expert_size;
    } catch (...) {
        moe_cache_slot_reset(pool, slot_index, true);
        device.insert_skips++;
        return false;
    }
    device.inserts++;
    inserts_left--;
    wake_worker = true;
    return true;
}

static int moe_cache_lookup_or_queue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker) {
    const moe_cache_key key{host_base, expert};
    auto found = pool.map.find(key);
    if (found != pool.map.end()) {
        return pool.slots[found->second].state == moe_cache_slot_state::valid
            ? found->second : -1;
    }

    moe_cache_demand * demand = nullptr;
    try {
        demand = &device.demand_count[key];
    } catch (...) {
        device.insert_skips++;
        return -1;
    }
    demand->expert_size = expert_size;
    if (demand->count < std::numeric_limits<uint16_t>::max()) {
        demand->count++;
    }
    const int initial_admit_after =
        pool.covers_all_entries && !session.config.admit_after_explicit
            ? 1 : session.config.admit_after;
    const int admit_after = pool.free_slots.empty()
        ? std::max(initial_admit_after, session.config.readmit_after)
        : initial_admit_after;
    if (demand->count < admit_after) {
        device.admission_skips++;
        return -1;
    }
    (void) moe_cache_enqueue_locked(
            session, device, pool, pool_index, host_base, expert,
            expert_size, inserts_left, wake_worker, true);
    return -1;
}

static void moe_cache_profile_seed_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        uint64_t tensor_profile, size_t expert_size, int64_t n_expert,
        bool & wake_worker) {
    if (tensor_profile == 0 || session.profile_heat.empty()) {
        return;
    }
    try {
        if (!session.profile_seeded.insert({host_base, device.physical}).second) {
            return;
        }
    } catch (...) {
        return;
    }
    const std::vector<int32_t> & hot = moe_cache_profile_hot_experts(
            session, tensor_profile);
    moe_cache_profile_advise(host_base, expert_size, hot, n_expert);
    int inserts_left = std::min(session.config.inserts_per_plan, 8);
    for (int32_t expert : hot) {
        if (expert < 0 || expert >= n_expert) {
            continue;
        }
        if (inserts_left <= 0) {
            break;
        }
        if (pool.map.find({host_base, expert}) == pool.map.end()) {
            (void) moe_cache_enqueue_locked(
                    session, device, pool, pool_index, host_base, expert,
                    expert_size, inserts_left, wake_worker, false);
        }
    }
}

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > moe_cache_node_rows_max || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;
    int inserts_left = session.config.inserts_per_plan;
    bool wake_worker = false;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }
    moe_cache_profile_seed_locked(
            session, device, pool, node->pool_index, node->host_base,
            node->profile_keys[0], node->expert_size, node->n_expert,
            wake_worker);
    moe_cache_profile_observe_locked(
            session, node->profile_keys[0], ids, n_ids, node->n_expert);
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }
        const int slot_index = moe_cache_lookup_or_queue_locked(
                session, device, pool, node->pool_index,
                node->host_base, expert, node->expert_size,
                inserts_left, wake_worker);
        if (slot_index < 0) {
            device.misses++;
            continue;
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.readers++;
        moe_cache_lru_remove(pool, slot_index);
        moe_cache_lru_push_back(pool, slot_index);
        node->pins[node->n_pins++] = {&pool, slot_index};
        slot_indices[index] = slot_index;
        device.hits++;
        hits++;
    }
    if (hits == n_ids) {
        int rows = moe_cache_overlap_rows(*node, n_ids);
        for (int index = n_ids - 1; index >= 0 && rows > 0; index--) {
            if (slot_indices[index] < 0 || node->n_pins <= 0) {
                continue;
            }
            const moe_cache_pin & pin = node->pins[node->n_pins - 1];
            if (pin.slot != slot_indices[index]) {
                continue;
            }
            moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.readers > 0) {
                slot.readers--;
            }
            node->n_pins--;
            slot_indices[index] = -1;
            hits--;
            rows--;
            device.overlap_rows++;
        }
    }
    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

static int moe_cache_dispatch_internal(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows,
        moe_cache_pool * gate_pool, const int32_t * gate_slot_indices,
        moe_cache_pool * down_pool, const int32_t * down_slot_indices,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    const bool fused = gate_pool != nullptr;
    const bool full = down_pool != nullptr;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        n_hits * (full ? 3 : fused ? 2 : 1) != node->n_pins ||
        !slot_indices || !act_rows ||
        (fused && (!gate_slot_indices ||
                   gate_pool->wtype != wtype ||
                   gate_pool->expert_size != node->expert_size ||
                   glu_op != GGML_GLU_OP_SWIGLU ||
                   std::isnan(up_min) || std::isnan(up_max) ||
                   std::isnan(gate_min) || std::isnan(gate_max) ||
                   up_min > up_max || gate_min > gate_max)) ||
        (full && (!down_slot_indices || node->down_pool != down_pool ||
                  !moe_cache_type_supported((ggml_type)down_pool->wtype) ||
                  down_pool->expert_size == 0)) ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_mid ||
        n_in > INT_MAX || n_out > INT_MAX || node->n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1) ||
        (full && n_out > INT64_MAX - (MATRIX_ROW_PADDING - 1))) {
        return 0;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    if (device.dead.load() || moe_cache_fail(session, "dispatch")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    ggml_cuda_set_device(device.logical);
    if (!device.compute_stream) {
        if (!moe_cache_cuda_ok(device,
                    cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking),
                    "compute stream creation", true)) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
    }

    const float * unique_act_rows[moe_cache_node_rows_max];
    int32_t activation_indices[moe_cache_node_rows_max];
    int activation_rows = 0;
    for (int index = 0; index < n_hits; index++) {
        if (slot_indices[index] < 0 || slot_indices[index] >= pool.n_slots ||
            (fused && (gate_slot_indices[index] < 0 ||
                       gate_slot_indices[index] >= gate_pool->n_slots)) ||
            (full && (down_slot_indices[index] < 0 ||
                      down_slot_indices[index] >= down_pool->n_slots)) ||
            !act_rows[index]) {
            return 0;
        }
        int activation = 0;
        while (activation < activation_rows &&
               unique_act_rows[activation] != act_rows[index]) {
            activation++;
        }
        if (activation == activation_rows) {
            unique_act_rows[activation_rows++] = act_rows[index];
        }
        activation_indices[index] = activation;
    }
    bool modulo_activation = true;
    for (int index = 0; index < n_hits; index++) {
        modulo_activation &= activation_indices[index] == index % activation_rows;
    }

    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const int64_t padded_n_mid = full
        ? ((n_out + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING
        : 0;
    const size_t type_size = ggml_type_size((ggml_type)wtype);
    const size_t down_type_size = full
        ? ggml_type_size((ggml_type)down_pool->wtype) : type_size;
    if (type_size == 0 || node->expert_size % type_size != 0 ||
        ggml_row_size((ggml_type)wtype, n_in) % type_size != 0 ||
        node->expert_size / type_size > INT_MAX ||
        ggml_row_size((ggml_type)wtype, n_in) / type_size > INT_MAX ||
        padded_n_in / QK8_1 > INT_MAX ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) > INT_MAX ||
        (uint64_t)n_out * n_hits > INT_MAX ||
        (uint64_t)(node->expert_size / type_size) * pool.n_slots > INT_MAX ||
        (full && (padded_n_mid / QK8_1 > INT_MAX ||
                  (uint64_t)n_hits * (padded_n_mid / QK8_1) > INT_MAX ||
                  (uint64_t)node->n_out * n_hits > INT_MAX ||
                  down_type_size == 0 ||
                  down_pool->expert_size % down_type_size != 0 ||
                  ggml_row_size((ggml_type)down_pool->wtype, n_out) !=
                      down_pool->expert_size / (size_t)node->n_out ||
                  (uint64_t)(down_pool->expert_size / down_type_size) *
                      down_pool->n_slots > INT_MAX))) {
        return 0;
    }

    if (n_in > INT64_MAX / activation_rows ||
        (uint64_t)n_in * activation_rows > SIZE_MAX / sizeof(float) ||
        n_out > INT64_MAX / n_hits ||
        (uint64_t)n_out * n_hits > SIZE_MAX / sizeof(float) ||
        (full && (node->n_out > INT64_MAX / n_hits ||
                  (uint64_t)node->n_out * n_hits > SIZE_MAX / sizeof(float) ||
                  (uint64_t)n_hits * (padded_n_mid / QK8_1) >
                      SIZE_MAX / sizeof(block_q8_1))) ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) >
            SIZE_MAX / sizeof(block_q8_1)) {
        return 0;
    }
    const bool use_activation_map =
        session.config.force_dedicated_mmv || !modulo_activation;
    const size_t id_arrays = 1 + (fused ? 1 : 0) + (full ? 1 : 0) +
        (use_activation_map ? 1 : 0);
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t) * id_arrays;
    const size_t act_bytes = (size_t)activation_rows * n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return 0;
    }
    const size_t input_bytes = ids_bytes + act_bytes;
    size_t q8_bytes =
        (size_t)activation_rows * (padded_n_in / QK8_1) * sizeof(block_q8_1);
    if (full) {
        q8_bytes = std::max(q8_bytes,
                (size_t)n_hits * (padded_n_mid / QK8_1) * sizeof(block_q8_1));
    }
    const size_t final_bytes = (size_t)n_hits * node->n_out * sizeof(float);
    const size_t out_bytes = std::max(
            (size_t)n_hits * n_out * sizeof(float), final_bytes);

    const size_t desired_caps[] = {
        moe_cache_growth_capacity(device.d_input_cap, input_bytes),
        moe_cache_growth_capacity(device.act_q8_cap, q8_bytes),
        moe_cache_growth_capacity(device.d_out_cap, out_bytes),
    };
    size_t scratch_bytes = 0;
    for (size_t capacity : desired_caps) {
        if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
        scratch_bytes += capacity;
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

    if (!moe_cache_grow_host(device, (void **)&device.h_input, device.h_input_cap,
                             input_bytes, "input host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_input, device.d_input_cap,
                               input_bytes, "input device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                               q8_bytes, "q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                               out_bytes, "output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_out, device.h_out_cap,
                             final_bytes, "output host allocation")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        device.dead.store(true);
        return 0;
    }

    int32_t * h_ids = (int32_t *)device.h_input;
    float * h_act = (float *)(device.h_input + ids_bytes);
    int32_t * d_ids = (int32_t *)device.d_input;
    float * d_act = (float *)(device.d_input + ids_bytes);
    for (int index = 0; index < n_hits; index++) {
        h_ids[index] = slot_indices[index];
        if (fused) {
            h_ids[n_hits + index] = gate_slot_indices[index];
        }
        if (full) {
            h_ids[2 * n_hits + index] = down_slot_indices[index];
        }
        if (use_activation_map) {
            h_ids[(full ? 3 : fused ? 2 : 1) * n_hits + index] =
                activation_indices[index];
        }
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(h_act + (size_t)index * n_in,
               unique_act_rows[index], n_in * sizeof(float));
    }

    (void)cudaGetLastError();
    bool ok =
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_input, device.h_input, input_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "input upload", true);
    if (ok) {
        quantize_row_q8_1_cuda(
                d_act, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_in, n_in, (int64_t)activation_rows * n_in,
                (int64_t)activation_rows * n_in, padded_n_in,
                activation_rows, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "activation quantization", true);
    }
    if (ok) {
        if (fused) {
            ggml_cuda_moe_cache_mmv_fused(
                    pool.slab, gate_pool->slab, (ggml_type)wtype,
                    (const char *)device.d_act_q8,
                    d_ids, d_ids + n_hits,
                    use_activation_map ? d_ids + (full ? 3 : 2) * n_hits : nullptr,
                    device.d_out, n_in, n_out,
                    (int64_t)pool.expert_size, n_hits, activation_rows,
                    up_min, up_max, gate_min, gate_max,
                    device.compute_stream);
        } else {
            (void)ggml_cuda_moe_cache_mmv(
                    pool.slab, (ggml_type)wtype,
                    (const char *)device.d_act_q8, d_ids,
                    use_activation_map ? d_ids + n_hits : nullptr,
                    device.d_out, n_in, n_out, pool.n_slots,
                    (int64_t)pool.expert_size, n_hits, activation_rows,
                    false, device.compute_stream);
        }
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
    }
    if (ok && full) {
        quantize_row_q8_1_cuda(
                device.d_out, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_out, n_out, (int64_t)n_hits * n_out,
                (int64_t)n_hits * n_out, padded_n_mid,
                n_hits, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "intermediate activation quantization", true);
    }
    ggml_cuda_moe_cache_mmv_path down_mmv_path =
        ggml_cuda_moe_cache_mmv_path::generic;
    if (ok && full) {
        down_mmv_path = ggml_cuda_moe_cache_mmv(
                down_pool->slab, (ggml_type)down_pool->wtype,
                (const char *)device.d_act_q8, d_ids + 2 * n_hits,
                nullptr, device.d_out, n_out, node->n_out,
                down_pool->n_slots, (int64_t)down_pool->expert_size,
                n_hits, n_hits, session.config.force_dedicated_down_mmv,
                device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "down expert matvec launch", true);
    }

    if (!ok) {
        cudaStreamSynchronize(device.compute_stream);
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    device.activation_dedup += n_hits - activation_rows;
    if (full && down_mmv_path == ggml_cuda_moe_cache_mmv_path::dedicated) {
        device.dedicated_down_mmv_dispatches++;
    }
    node->dispatched = true;
    return 1;
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    return moe_cache_dispatch_internal(
            opaque, wtype, n_in, n_out, n_hits,
            slot_indices, act_rows, nullptr, nullptr, nullptr, nullptr, -1,
            0.0f, 0.0f, 0.0f, 0.0f);
}

static bool moe_cache_collect_preflight(
        const moe_cache_node & node, int n_hits,
        float * const * dst_rows, int64_t n_out,
        const int * row_indices, int n_dst_rows) {
    const int pins_per_hit = node.host_base3 ? 3 : node.host_base2 ? 2 : 1;
    if (node.composite || !node.dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        n_hits * pins_per_hit != node.n_pins ||
        !dst_rows || n_out != node.n_out) {
        return false;
    }
    for (int row = 0; row < n_hits; row++) {
        const int dst = row_indices ? row_indices[row] : row;
        if (dst < 0 || (row_indices && dst >= n_dst_rows) || !dst_rows[dst]) {
            return false;
        }
    }
    return true;
}

static bool moe_cache_collect_enqueue(moe_cache_node & node, int n_hits) {
    moe_cache_session & session = *node.session;
    moe_cache_device & device = *node.device;

    bool ok = !device.dead.load() && !moe_cache_fail(session, "collect");
    if (ok) {
        // CUDA memory copies may use a stream owned by a non-current device.
        // finish() selects the device before synchronizing that stream.
        const size_t bytes =
            (size_t)n_hits * node.n_out * sizeof(float);
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.h_out, device.d_out, bytes,
                cudaMemcpyDeviceToHost, device.compute_stream),
                "output download", true);
    }
    return ok;
}

static bool moe_cache_collect_finish(moe_cache_node & node, bool ok) {
    moe_cache_session & session = *node.session;
    moe_cache_device & device = *node.device;
    ggml_cuda_set_device(device.logical);

    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        cudaStreamSynchronize(device.compute_stream);
    }
    node.dispatched = false;

    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            device.collect_failures++;
        }
        device.collect_calls++;
        if (session.config.stats_every > 0 &&
            device.collect_calls % session.config.stats_every == 0) {
            moe_cache_log_stats(device);
        }
    }
    return ok;
}

static void moe_cache_collect_scatter(
        const moe_cache_node & node, int n_hits, float * const * dst_rows,
        const int * row_indices) {
    for (int row = 0; row < n_hits; row++) {
        const int dst = row_indices ? row_indices[row] : row;
        memcpy(dst_rows[dst],
               node.device->h_out + (size_t)row * node.n_out,
               node.n_out * sizeof(float));
    }
}

static int moe_cache_collect(
        void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (node && node->composite) {
        if (!node->dispatched || n_hits != node->n_result_rows ||
            !dst_rows || n_out != node->n_out) {
            return 0;
        }

        // Queue every device's result download before waiting for any one of
        // them. The expert-parallel launches are concurrent, so serializing
        // copy+wait per child needlessly puts the later downloads behind the
        // first device's completion.
        if (node->children.size() > moe_cache_expert_parallel_max) {
            return 0;
        }
        bool child_ok[moe_cache_expert_parallel_max] = {};
        for (size_t index = 0; index < node->children.size(); index++) {
            moe_cache_node * child = node->children[index].get();
            if (!child || !moe_cache_collect_preflight(
                    *child, child->n_result_rows, dst_rows, n_out,
                    child->row_indices, n_hits)) {
                node->dispatched = false;
                return 0;
            }
        }

        for (size_t index = 0; index < node->children.size(); index++) {
            child_ok[index] = moe_cache_collect_enqueue(
                    *node->children[index],
                    node->children[index]->n_result_rows);
        }
        bool ok = true;
        for (size_t index = 0; index < node->children.size(); index++) {
            child_ok[index] = moe_cache_collect_finish(
                    *node->children[index], child_ok[index]);
            ok = ok && child_ok[index];
        }
        if (ok) {
            for (const auto & child : node->children) {
                moe_cache_collect_scatter(
                        *child, child->n_result_rows,
                        dst_rows, child->row_indices);
            }
        }
        node->dispatched = false;
        return ok ? 1 : 0;
    }
    if (!node || !moe_cache_collect_preflight(
            *node, n_hits, dst_rows, n_out, nullptr, n_hits)) {
        return 0;
    }
    bool ok = moe_cache_collect_enqueue(*node, n_hits);
    ok = moe_cache_collect_finish(*node, ok);
    if (ok) {
        moe_cache_collect_scatter(*node, n_hits, dst_rows, nullptr);
    }
    return ok ? 1 : 0;
}

static void moe_cache_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }

    if (node->composite) {
        for (auto & child : node->children) {
            moe_cache_end(child.release());
        }
        node->children.clear();
        node->dispatched = false;
    } else if (node->dispatched) {
        ggml_cuda_set_device(node->device->logical);
        moe_cache_cuda_ok(
                *node->device, cudaStreamSynchronize(node->device->compute_stream),
                "end synchronization", true);
        node->dispatched = false;
    }

    moe_cache_session & session = *node->session;
    const bool trim = node->device && node->device->dead.load();
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        if (node->owns_active) {
            for (const void * base : {
                    node->host_base, node->host_base2, node->host_base3}) {
                if (!base) {
                    continue;
                }
                auto source = session.active_sources.find(base);
                if (source != session.active_sources.end() &&
                    --source->second.references == 0) {
                    session.active_sources.erase(source);
                }
            }
            session.active_nodes--;
            session.idle_cv.notify_all();
        }
    }
    if (trim && node->dispatch_lock.owns_lock()) {
        node->dispatch_lock.unlock();
        moe_cache_trim_session(session, node->device->physical);
    }
}

static void * moe_cache_fused_begin_expert_parallel(
        moe_cache_session & session,
        const ggml_moe_cache_tensor_desc * up,
        const ggml_moe_cache_tensor_desc * gate,
        const ggml_moe_cache_tensor_desc * down,
        int layer, int glu_op, float up_min, float up_max,
        float gate_min, float gate_max,
        const int32_t * ids, int n_ids, int64_t n_tokens,
        const float * const * act_rows, uint64_t * hit_mask) {
    if (!down || session.devices.size() < 2) {
        return nullptr;
    }

    const size_t up_tensor_size = (size_t)up->n_expert * up->expert_size;
    const size_t down_tensor_size = (size_t)down->n_expert * down->expert_size;
    moe_cache_scratch up_scratch;
    moe_cache_scratch down_scratch;
    const bool profile_enabled = !session.config.profile_path.empty();
    const uint64_t profile_keys[3] = {
        profile_enabled ? moe_cache_profile_tensor_hash(
            up->name, up->expert_size, up->type, up->n_expert) : 0,
        profile_enabled ? moe_cache_profile_tensor_hash(
            gate->name, gate->expert_size, gate->type, gate->n_expert) : 0,
        profile_enabled ? moe_cache_profile_tensor_hash(
            down->name, down->expert_size, down->type, down->n_expert) : 0,
    };
    bool advise[3] = {};
    if (!session.profile_heat.empty()) {
        try {
            std::lock_guard<std::mutex> lock(session.mu);
            advise[0] = session.profile_advised.insert(up->data).second;
            advise[1] = session.profile_advised.insert(gate->data).second;
            advise[2] = session.profile_advised.insert(down->data).second;
        } catch (...) {
            advise[0] = advise[1] = advise[2] = false;
        }
    }
    if (advise[0]) {
        moe_cache_profile_advise(up->data, up->expert_size,
                moe_cache_profile_hot_experts(session, profile_keys[0]), up->n_expert);
    }
    if (advise[1]) {
        moe_cache_profile_advise(gate->data, gate->expert_size,
                moe_cache_profile_hot_experts(session, profile_keys[1]), gate->n_expert);
    }
    if (advise[2]) {
        moe_cache_profile_advise(down->data, down->expert_size,
                moe_cache_profile_hot_experts(session, profile_keys[2]), down->n_expert);
    }
    if (!moe_cache_scratch_requirements(up->n_in, up->n_out, up_scratch) ||
        !moe_cache_scratch_requirements(down->n_in, down->n_out, down_scratch)) {
        return nullptr;
    }

    struct route {
        moe_cache_device * device = nullptr;
        moe_cache_pool * pair_pool = nullptr;
        moe_cache_pool * down_pool = nullptr;
        int pair_pool_index = -1;
        int down_pool_index = -1;
        int rows[moe_cache_node_rows_max];
        int n_rows = 0;
        int n_hits = 0;
        int up_slots[moe_cache_node_rows_max];
        int gate_slots[moe_cache_node_rows_max];
        int down_slots[moe_cache_node_rows_max];
        const float * acts[moe_cache_node_rows_max];
        moe_cache_pin pins[3 * moe_cache_node_rows_max];
        int n_pins = 0;
        std::unique_lock<std::mutex> dispatch_lock;
        moe_cache_node * child = nullptr;
    };
    std::vector<route> routes;
    try {
        routes.reserve(session.devices.size());
    } catch (...) {
        return nullptr;
    }

    // Every participating device sees every tensor shape, but actual expert rows are
    // deterministically sharded below. This gives each device a correctly-sized slice
    // of every layer without replicating hot rows across all devices.
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            if (device.dead.load() || !moe_cache_prepare_budget(session, device)) {
                continue;
            }
            device.scratch_reserve.input = std::max(
                    device.scratch_reserve.input,
                    std::max(up_scratch.input, down_scratch.input));
            device.scratch_reserve.q8 = std::max(
                    device.scratch_reserve.q8,
                    std::max(up_scratch.q8, down_scratch.q8));
            device.scratch_reserve.out = std::max(
                    device.scratch_reserve.out,
                    std::max(up_scratch.out, down_scratch.out));

            int up_pool = -1;
            int down_pool = -1;
            try {
                up_pool = moe_cache_discover_pool(
                        session, device, up->data, up_tensor_size,
                        up->expert_size, up->type, up->n_expert);
                (void)moe_cache_discover_pool(
                        session, device, gate->data, up_tensor_size,
                        gate->expert_size, gate->type, gate->n_expert);
                down_pool = moe_cache_discover_pool(
                        session, device, down->data, down_tensor_size,
                        down->expert_size, down->type, down->n_expert);
            } catch (...) {
                device.insert_skips++;
                continue;
            }
            if (up_pool < 0 || down_pool < 0 ||
                up_pool >= (int)device.pools.size() ||
                down_pool >= (int)device.pools.size()) {
                continue;
            }
            routes.emplace_back();
            route & result = routes.back();
            result.device = &device;
            result.pair_pool = device.pools[up_pool].get();
            result.down_pool = device.pools[down_pool].get();
            result.pair_pool_index = up_pool;
            result.down_pool_index = down_pool;
        }
    }
    if (routes.size() < 2) {
        return nullptr;
    }

    const size_t fanout = std::min<size_t>(
            (size_t)std::max(session.config.expert_parallel, 1), routes.size());
    if (fanout < routes.size()) {
        std::vector<route> selected;
        try {
            selected.reserve(fanout);
            const size_t first = ((size_t)layer * fanout) % routes.size();
            for (size_t index = 0; index < fanout; index++) {
                selected.push_back(std::move(routes[(first + index) % routes.size()]));
            }
        } catch (...) {
            return nullptr;
        }
        routes = std::move(selected);
    }

    for (int row = 0; row < n_ids; row++) {
        if (ids[row] < 0 || ids[row] >= up->n_expert) {
            return nullptr;
        }
        const uint64_t key =
            ((uint64_t)(uint32_t)ids[row] + 1) * UINT64_C(0x9e3779b97f4a7c15) ^
            ((uint64_t)(uint32_t)layer + 1) * UINT64_C(0xbf58476d1ce4e5b9);
        route & selected = routes[key % routes.size()];
        selected.rows[selected.n_rows++] = row;
    }

    // Lock all selected devices in physical-device order. The launches remain
    // asynchronous; holding these locks only prevents another graph from reusing a
    // device's shared scratch buffers until collect completes.
    for (route & current : routes) {
        if (current.n_rows == 0) {
            continue;
        }
        try {
            current.dispatch_lock = std::unique_lock<std::mutex>(
                    current.device->dispatch_mu, std::try_to_lock);
        } catch (...) {
            current.device->contention_bypasses++;
            return nullptr;
        }
        if (!current.dispatch_lock.owns_lock()) {
            current.device->contention_bypasses++;
            return nullptr;
        }
    }

    const void * source_bases[] = {up->data, gate->data, down->data};
    const size_t source_bytes[] = {
        up_tensor_size, up_tensor_size, down_tensor_size};
    bool active_node = false;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        bool source_acquired[3] = {};
        auto acquire_source = [&](int index) {
            try {
                moe_cache_session::active_source & source =
                    session.active_sources[source_bases[index]];
                source.bytes = std::max(source.bytes, source_bytes[index]);
                source.references++;
                source_acquired[index] = true;
                return true;
            } catch (...) {
                return false;
            }
        };
        if (!acquire_source(0) || !acquire_source(1) || !acquire_source(2)) {
            for (int index = 0; index < 3; index++) {
                if (!source_acquired[index]) {
                    continue;
                }
                auto found = session.active_sources.find(source_bases[index]);
                if (found != session.active_sources.end() &&
                    --found->second.references == 0) {
                    session.active_sources.erase(found);
                }
            }
            return nullptr;
        }
        session.active_nodes++;
        active_node = true;
    }
    auto release_active = [&] {
        if (!active_node) {
            return;
        }
        std::lock_guard<std::mutex> lock(session.mu);
        for (const void * base : source_bases) {
            auto found = session.active_sources.find(base);
            if (found != session.active_sources.end() &&
                --found->second.references == 0) {
                session.active_sources.erase(found);
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
        active_node = false;
    };

    bool wake_worker = false;
    uint64_t mask = 0;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (uint64_t profile_key : profile_keys) {
            moe_cache_profile_observe_locked(
                    session, profile_key, ids, n_ids, up->n_expert);
        }
        for (route & current : routes) {
            int inserts_left = session.config.inserts_per_plan;
            for (int local = 0; local < current.n_rows; local++) {
                const int row = current.rows[local];
                const int32_t expert = ids[row];
                const int up_slot = moe_cache_lookup_or_queue_locked(
                        session, *current.device, *current.pair_pool,
                        current.pair_pool_index, up->data, expert,
                        up->expert_size, inserts_left, wake_worker);
                const int gate_slot = moe_cache_lookup_or_queue_locked(
                        session, *current.device, *current.pair_pool,
                        current.pair_pool_index, gate->data, expert,
                        gate->expert_size, inserts_left, wake_worker);
                const int down_slot = moe_cache_lookup_or_queue_locked(
                        session, *current.device, *current.down_pool,
                        current.down_pool_index, down->data, expert,
                        down->expert_size, inserts_left, wake_worker);
                current.device->misses += up_slot < 0;
                current.device->misses += gate_slot < 0;
                current.device->misses += down_slot < 0;
                if (up_slot < 0 || gate_slot < 0 || down_slot < 0) {
                    continue;
                }
                const int hit = current.n_hits++;
                current.rows[hit] = row;
                current.up_slots[hit] = up_slot;
                current.gate_slots[hit] = gate_slot;
                current.down_slots[hit] = down_slot;
                current.acts[hit] = act_rows[row];
                for (const moe_cache_pin pin : {
                        moe_cache_pin{current.pair_pool, up_slot},
                        moe_cache_pin{current.pair_pool, gate_slot},
                        moe_cache_pin{current.down_pool, down_slot}}) {
                    moe_cache_slot & slot = pin.pool->slots[pin.slot];
                    slot.readers++;
                    moe_cache_lru_remove(*pin.pool, pin.slot);
                    moe_cache_lru_push_back(*pin.pool, pin.slot);
                    current.pins[current.n_pins++] = pin;
                }
                mask |= UINT64_C(1) << row;
            }
        }
    }
    if (wake_worker) {
        session.cv.notify_all();
    }
    if (mask == 0) {
        release_active();
        return nullptr;
    }
    int total_hits = 0;
    for (const route & current : routes) {
        total_hits += current.n_hits;
    }
    auto release_route_pins = [&] {
        std::lock_guard<std::mutex> lock(session.mu);
        for (route & current : routes) {
            for (int index = 0; index < current.n_pins; index++) {
                const moe_cache_pin & pin = current.pins[index];
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                GGML_ASSERT(slot.readers > 0);
                slot.readers--;
            }
            current.n_pins = 0;
        }
    };

    std::unique_ptr<moe_cache_node> root(new (std::nothrow) moe_cache_node());
    if (!root) {
        release_route_pins();
        release_active();
        return nullptr;
    }
    root->session = &session;
    root->host_base = up->data;
    root->host_base2 = gate->data;
    root->host_base3 = down->data;
    root->n_out = down->n_out;
    root->n_result_rows = total_hits;
    root->composite = true;
    root->planned = true;
    std::copy(std::begin(profile_keys), std::end(profile_keys), root->profile_keys);
    active_node = false;

    size_t n_children = 0;
    for (const route & current : routes) {
        n_children += current.n_hits > 0;
    }
    try {
        root->children.reserve(n_children);
    } catch (...) {
        release_route_pins();
        moe_cache_end(root.release());
        return nullptr;
    }
    int hit_ranks[moe_cache_node_rows_max];
    int hit_rank = 0;
    for (int row = 0; row < n_ids; row++) {
        hit_ranks[row] = mask & (UINT64_C(1) << row) ? hit_rank++ : -1;
    }
    for (route & current : routes) {
        if (current.n_hits == 0) {
            continue;
        }
        std::unique_ptr<moe_cache_node> child(new (std::nothrow) moe_cache_node());
        if (!child) {
            release_route_pins();
            moe_cache_end(root.release());
            return nullptr;
        }
        child->session = &session;
        child->device = current.device;
        child->pool = current.pair_pool;
        child->down_pool = current.down_pool;
        child->pool_index = current.pair_pool_index;
        child->host_base = up->data;
        child->host_base2 = gate->data;
        child->host_base3 = down->data;
        child->expert_size = up->expert_size;
        child->n_in = up->n_in;
        child->n_mid = up->n_out;
        child->n_out = down->n_out;
        child->n_expert = up->n_expert;
        child->n_tokens = n_tokens;
        child->wtype = up->type;
        std::copy(std::begin(profile_keys), std::end(profile_keys), child->profile_keys);
        child->planned = true;
        child->owns_active = false;
        child->n_result_rows = current.n_hits;
        child->dispatch_lock = std::move(current.dispatch_lock);
        for (int local = 0; local < current.n_hits; local++) {
            child->row_indices[local] = hit_ranks[current.rows[local]];
        }
        current.child = child.get();
        root->children.push_back(std::move(child));
    }

    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (route & current : routes) {
            if (current.n_hits == 0) {
                continue;
            }
            moe_cache_node * child = current.child;
            GGML_ASSERT(current.n_pins == 3 * current.n_hits);
            for (int index = 0; index < current.n_pins; index++) {
                child->pins[child->n_pins++] = current.pins[index];
            }
            current.n_pins = 0;
            current.device->hits += 3 * current.n_hits;
            current.device->nodes++;
            current.device->fused_attempts++;
        }
    }

    for (route & current : routes) {
        if (current.n_hits == 0) {
            continue;
        }
        if (!moe_cache_dispatch_internal(
                current.child, up->type, up->n_in, up->n_out,
                current.child->n_result_rows,
                current.up_slots, current.acts,
                current.pair_pool, current.gate_slots,
                current.down_pool, current.down_slots,
                glu_op, up_min, up_max, gate_min, gate_max)) {
            moe_cache_end(root.release());
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(session.mu);
        current.device->fused_rows += current.child->n_result_rows;
        current.device->fused_candidates += current.n_rows;
        current.device->fused_nodes++;
        current.device->full_fused_rows += current.child->n_result_rows;
        current.device->full_fused_nodes++;
    }

    root->dispatched = true;
    *hit_mask = mask;
    return root.release();
}

static void * moe_cache_fused_begin(
        const ggml_moe_cache_tensor_desc * up,
        const ggml_moe_cache_tensor_desc * gate,
        const ggml_moe_cache_tensor_desc * down,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max,
        const int32_t * ids, int n_ids, int64_t n_tokens,
        const float * const * act_rows, uint64_t * hit_mask) {
    if (hit_mask) {
        *hit_mask = 0;
    }
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !up || !gate || !ids || !act_rows || !hit_mask ||
        !up->name || !gate->name || !up->data || !gate->data ||
        up->data == gate->data ||
        !moe_cache_tensor_name_supported(up->name) ||
        !moe_cache_tensor_name_supported(gate->name) ||
        glu_op != GGML_GLU_OP_SWIGLU ||
        std::isnan(up_min) || std::isnan(up_max) ||
        std::isnan(gate_min) || std::isnan(gate_max) ||
        up_min > up_max || gate_min > gate_max ||
        n_ids < 1 || n_ids > moe_cache_node_rows_max ||
        n_tokens < 1 || n_tokens > n_ids || n_ids % n_tokens != 0 ||
        up->expert_size != gate->expert_size ||
        up->n_in != gate->n_in || up->n_out != gate->n_out ||
        up->n_expert != gate->n_expert || up->type != gate->type ||
        up->n_in <= 0 || up->n_out <= 0 ||
        up->n_expert <= 0 ||
        up->type == GGML_TYPE_NVFP4 ||
        !moe_cache_type_supported((ggml_type)up->type)) {
        return nullptr;
    }
    if (down && (!down->name || !down->data ||
                 down->data == up->data || down->data == gate->data ||
                 !moe_cache_tensor_name_supported(down->name) ||
                 down->n_in != up->n_out || down->n_out != up->n_in ||
                 down->n_expert != up->n_expert ||
                 !moe_cache_type_supported((ggml_type)down->type) ||
                 down->expert_size == 0 ||
                 (uint64_t)down->n_expert > SIZE_MAX / down->expert_size ||
                 ggml_row_size((ggml_type)down->type, down->n_in) == 0 ||
                 (uint64_t)down->n_out >
                     SIZE_MAX / ggml_row_size((ggml_type)down->type, down->n_in) ||
                 down->expert_size != (size_t)down->n_out *
                     ggml_row_size((ggml_type)down->type, down->n_in))) {
        return nullptr;
    }
    for (int index = 0; index < n_ids; index++) {
        if (!act_rows[index]) {
            return nullptr;
        }
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    const bool expert_parallel = session->config.expert_parallel &&
        session->devices.size() >= 2;
    if (expert_parallel && !down) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (up->expert_size < session->config.min_expert_bytes) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token fused node above max-batch=%d; increase the limit only when decode or speculative batches benefit on this hardware\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)up->type, up->n_in);
    if (row_size == 0 || (uint64_t)up->n_out > SIZE_MAX / row_size ||
        up->expert_size != (size_t)up->n_out * row_size ||
        (uint64_t)up->n_expert > SIZE_MAX / up->expert_size) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)up->n_expert * up->expert_size;
    const size_t down_tensor_size = down
        ? (size_t)down->n_expert * down->expert_size : 0;

    int up_layer = -1;
    int gate_layer = -1;
    int down_layer = -1;
    if (!moe_cache_layer_number(up->name, up_layer) ||
        !moe_cache_layer_number(gate->name, gate_layer) ||
        up_layer != gate_layer ||
        (down && (!moe_cache_layer_number(down->name, down_layer) ||
                  down_layer != up_layer))) {
        return nullptr;
    }

    if (expert_parallel && down) {
        return moe_cache_fused_begin_expert_parallel(
                *session, up, gate, down, up_layer, glu_op,
                up_min, up_max, gate_min, gate_max,
                ids, n_ids, n_tokens, act_rows, hit_mask);
    }

    const bool profile_enabled = !session->config.profile_path.empty();
    const uint64_t profile_keys[3] = {
        profile_enabled ? moe_cache_profile_tensor_hash(
            up->name, up->expert_size, up->type, up->n_expert) : 0,
        profile_enabled ? moe_cache_profile_tensor_hash(
            gate->name, gate->expert_size, gate->type, gate->n_expert) : 0,
        profile_enabled && down ? moe_cache_profile_tensor_hash(
            down->name, down->expert_size, down->type, down->n_expert) : 0,
    };

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        return nullptr;
    }

    moe_cache_device * selected = nullptr;
    moe_cache_pool * pool = nullptr;
    moe_cache_pool * down_pool = nullptr;
    int pool_index = -1;
    int down_pool_index = -1;
    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->stopping || session->dormant) {
            return nullptr;
        }

        auto route_for = [&](const void * tensor) {
            const auto tensor_route = session->tensor_devices.find(tensor);
            if (tensor_route != session->tensor_devices.end()) {
                return tensor_route->second;
            }
            const auto layer_route = session->layer_devices.find(up_layer);
            return layer_route != session->layer_devices.end()
                ? layer_route->second : -1;
        };
        const int up_device = route_for(up->data);
        const int gate_device = route_for(gate->data);
        const int down_device = down ? route_for(down->data) : up_device;
        if (up_device < 0 || up_device != gate_device ||
            up_device != down_device) {
            return nullptr;
        }
        for (const auto & device_ptr : session->devices) {
            if (device_ptr->physical == up_device) {
                selected = device_ptr.get();
                break;
            }
        }
        if (!selected || selected->dead.load()) {
            return nullptr;
        }
        pool_index = moe_cache_find_pool(
                *selected, up->expert_size, up->type);
        if (pool_index < 0 || pool_index >= (int)selected->pools.size()) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        if (!pool || !pool->slab) {
            return nullptr;
        }
        if (down) {
            down_pool_index = moe_cache_find_pool(
                    *selected, down->expert_size, down->type);
            if (down_pool_index < 0 ||
                down_pool_index >= (int)selected->pools.size()) {
                return nullptr;
            }
            down_pool = selected->pools[down_pool_index].get();
            if (!down_pool || !down_pool->slab) {
                return nullptr;
            }
        }

        auto acquire_source = [&](const void * base, size_t bytes) {
            try {
                moe_cache_session::active_source & source =
                    session->active_sources[base];
                source.bytes = std::max(source.bytes, bytes);
                source.references++;
                return true;
            } catch (...) {
                return false;
            }
        };
        if (!acquire_source(up->data, tensor_size)) {
            return nullptr;
        }
        if (!acquire_source(gate->data, tensor_size)) {
            auto source = session->active_sources.find(up->data);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
            return nullptr;
        }
        if (down && !acquire_source(down->data, down_tensor_size)) {
            for (const void * base : {up->data, gate->data}) {
                auto source = session->active_sources.find(base);
                if (source != session->active_sources.end() &&
                    --source->second.references == 0) {
                    session->active_sources.erase(source);
                }
            }
            return nullptr;
        }
        session->active_nodes++;
    }

    auto release_active = [&] {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const void * base : {up->data, gate->data,
                                  down ? down->data : nullptr}) {
            if (!base) {
                continue;
            }
            auto source = session->active_sources.find(base);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
    };

    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                selected->dispatch_mu, std::try_to_lock);
    } catch (...) {
        selected->contention_bypasses++;
        release_active();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        selected->contention_bypasses++;
    }
    if (!dispatch_lock.owns_lock() || selected->dead.load()) {
        release_active();
        return nullptr;
    }

    node->session = session;
    node->device = selected;
    node->pool = pool;
    node->down_pool = down_pool;
    node->pool_index = pool_index;
    node->host_base = up->data;
    node->host_base2 = gate->data;
    node->host_base3 = down ? down->data : nullptr;
    node->expert_size = up->expert_size;
    node->n_in = up->n_in;
    node->n_mid = up->n_out;
    node->n_out = down ? down->n_out : up->n_out;
    node->n_expert = up->n_expert;
    node->n_tokens = n_tokens;
    node->wtype = up->type;
    std::copy(std::begin(profile_keys), std::end(profile_keys), node->profile_keys);
    node->dispatch_lock = std::move(dispatch_lock);
    node->planned = true;

    int32_t up_slots[moe_cache_node_rows_max];
    int32_t gate_slots[moe_cache_node_rows_max];
    int32_t down_slots[moe_cache_node_rows_max];
    int32_t resident_up[moe_cache_node_rows_max];
    int32_t resident_gate[moe_cache_node_rows_max];
    int32_t resident_down[moe_cache_node_rows_max];
    const float * hit_acts[moe_cache_node_rows_max];
    uint64_t mask = 0;
    int hits = 0;
    bool full_ready = true;
    int inserts_left = session->config.inserts_per_plan;
    bool wake_worker = false;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (!session->stopping && !selected->dead.load()) {
            moe_cache_profile_seed_locked(
                    *session, *selected, *pool, pool_index, up->data,
                    node->profile_keys[0], up->expert_size, up->n_expert,
                    wake_worker);
            moe_cache_profile_seed_locked(
                    *session, *selected, *pool, pool_index, gate->data,
                    node->profile_keys[1], gate->expert_size, gate->n_expert,
                    wake_worker);
            if (down) {
                moe_cache_profile_seed_locked(
                        *session, *selected, *down_pool, down_pool_index, down->data,
                        node->profile_keys[2], down->expert_size, down->n_expert,
                        wake_worker);
            }
            moe_cache_profile_observe_locked(
                    *session, node->profile_keys[0], ids, n_ids, up->n_expert);
            moe_cache_profile_observe_locked(
                    *session, node->profile_keys[1], ids, n_ids, up->n_expert);
            if (down) {
                moe_cache_profile_observe_locked(
                        *session, node->profile_keys[2], ids, n_ids, up->n_expert);
            }
            for (int index = 0; index < n_ids; index++) {
                resident_up[index] = -1;
                resident_gate[index] = -1;
                resident_down[index] = -1;
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    continue;
                }
                const auto up_found = pool->map.find({up->data, expert});
                const auto gate_found = pool->map.find({gate->data, expert});
                const auto down_found = down
                    ? down_pool->map.find({down->data, expert})
                    : pool->map.end();
                if (up_found != pool->map.end() &&
                    pool->slots[up_found->second].state == moe_cache_slot_state::valid) {
                    resident_up[index] = up_found->second;
                }
                if (gate_found != pool->map.end() &&
                    pool->slots[gate_found->second].state == moe_cache_slot_state::valid) {
                    resident_gate[index] = gate_found->second;
                }
                if (down && down_found != down_pool->map.end() &&
                    down_pool->slots[down_found->second].state == moe_cache_slot_state::valid) {
                    resident_down[index] = down_found->second;
                }
                if (resident_up[index] >= 0 && resident_gate[index] >= 0 &&
                    (!down || resident_down[index] >= 0)) {
                    if (!down) {
                        selected->pair_both++;
                    }
                } else if (resident_up[index] >= 0) {
                    if (!down) {
                        selected->pair_up_only++;
                    }
                } else if (resident_gate[index] >= 0) {
                    if (!down) {
                        selected->pair_gate_only++;
                    }
                } else {
                    if (!down) {
                        selected->pair_neither++;
                    }
                }
                if (down && (resident_up[index] < 0 ||
                             resident_gate[index] < 0 ||
                             resident_down[index] < 0)) {
                    full_ready = false;
                }
            }

            for (int index = 0; index < n_ids && (!down || full_ready); index++) {
                if (resident_up[index] < 0 || resident_gate[index] < 0 ||
                    (down && resident_down[index] < 0)) {
                    continue;
                }
                moe_cache_slot & up_slot = pool->slots[resident_up[index]];
                moe_cache_slot & gate_slot = pool->slots[resident_gate[index]];
                up_slot.readers++;
                gate_slot.readers++;
                moe_cache_lru_remove(*pool, resident_up[index]);
                moe_cache_lru_push_back(*pool, resident_up[index]);
                moe_cache_lru_remove(*pool, resident_gate[index]);
                moe_cache_lru_push_back(*pool, resident_gate[index]);
                node->pins[node->n_pins++] = {pool, resident_up[index]};
                node->pins[node->n_pins++] = {pool, resident_gate[index]};
                up_slots[hits] = resident_up[index];
                gate_slots[hits] = resident_gate[index];
                hit_acts[hits] = act_rows[index];
                mask |= UINT64_C(1) << index;
                selected->hits += 2;
                hits++;
            }

            // Preserve the unfused access order when all three tensors happen
            // to share one pool: pair rows first, then down rows.
            if (down && full_ready) {
                for (int index = 0; index < n_ids; index++) {
                    moe_cache_slot & down_slot =
                        down_pool->slots[resident_down[index]];
                    down_slot.readers++;
                    moe_cache_lru_remove(*down_pool, resident_down[index]);
                    moe_cache_lru_push_back(*down_pool, resident_down[index]);
                    node->pins[node->n_pins++] = {
                        down_pool, resident_down[index]};
                    down_slots[index] = resident_down[index];
                    selected->hits++;
                }
            }

            for (int index = 0; index < n_ids && hits > 0 && !down; index++) {
                if (mask & (UINT64_C(1) << index)) {
                    continue;
                }
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    selected->misses += 2;
                    continue;
                }

                const int up_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        up->data, expert, up->expert_size,
                        inserts_left, wake_worker);
                if (up_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[up_slot];
                    slot.readers++;
                    moe_cache_lru_remove(*pool, up_slot);
                    moe_cache_lru_push_back(*pool, up_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                const int gate_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        gate->data, expert, gate->expert_size,
                        inserts_left, wake_worker);
                if (gate_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[gate_slot];
                    slot.readers++;
                    moe_cache_lru_remove(*pool, gate_slot);
                    moe_cache_lru_push_back(*pool, gate_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                if (up_slot >= 0) {
                    pool->slots[up_slot].readers--;
                }
                if (gate_slot >= 0) {
                    pool->slots[gate_slot].readers--;
                }
            }
            if (hits > 0) {
                selected->nodes++;
                selected->fused_attempts++;
            }
        }
    }
    if (wake_worker) {
        session->cv.notify_all();
    }

    if (hits == 0 || !moe_cache_dispatch_internal(
            node.get(), up->type, up->n_in, up->n_out, hits,
            up_slots, hit_acts, pool, gate_slots,
            down_pool, down ? down_slots : nullptr, glu_op,
            up_min, up_max, gate_min, gate_max)) {
        moe_cache_end(node.release());
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(session->mu);
        selected->fused_rows += hits;
        selected->fused_candidates += n_ids;
        selected->fused_nodes++;
        if (down) {
            selected->full_fused_rows += hits;
            selected->full_fused_nodes++;
        }
    }

    *hit_mask = mask;
    return node.release();
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    session.tensor_devices.erase(base);
    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            if (device_ptr->inflight &&
                moe_cache_ranges_overlap(
                    device_ptr->inflight_source, device_ptr->inflight_bytes, base, size)) {
                return false;
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (int index = 0; index < pool.n_slots; index++) {
                moe_cache_slot & slot = pool.slots[index];
                const void * source = slot.key.expert >= 0
                    ? (const char *)slot.key.tensor +
                        (size_t)slot.key.expert * pool.expert_size
                    : nullptr;
                if (slot.state != moe_cache_slot_state::free &&
                    moe_cache_ranges_overlap(source, pool.expert_size, base, size)) {
                    moe_cache_slot_reset(pool, index, true);
                }
            }
        }
        for (auto it = device.seen_tensors.begin(); it != device.seen_tensors.end();) {
            if (moe_cache_ranges_overlap(it->first, it->second.bytes, base, size)) {
                session.tensor_devices.erase(it->first);
                for (moe_cache_shape & shape : device.shapes) {
                    if (shape.expert_size == it->second.expert_size &&
                        shape.wtype == it->second.wtype) {
                        const uint64_t entries = (uint64_t)it->second.n_expert;
                        shape.n_entries = entries <= shape.n_entries
                            ? shape.n_entries - entries : 0;
                        shape.n_tensors = std::max<int64_t>(shape.n_tensors - 1, 0);
                        if (shape.pool >= 0 && shape.pool < (int)device.pools.size()) {
                            device.pools[shape.pool]->covers_all_entries =
                                (uint64_t)device.pools[shape.pool]->n_slots >= shape.n_entries;
                        }
                        if (shape.n_tensors == 0 && shape.pool < 0) {
                            shape.finished = false;
                        }
                        break;
                    }
                }
                it = device.seen_tensors.erase(it);
                device.visits_since_new_tensor = 0;
            } else {
                ++it;
            }
        }
        for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
            const void * source = it->first.expert >= 0
                ? (const char *)it->first.tensor +
                    (size_t)it->first.expert * it->second.expert_size
                : nullptr;
            if (moe_cache_ranges_overlap(
                    source, it->second.expert_size, base, size)) {
                it = device.demand_count.erase(it);
            } else {
                ++it;
            }
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    std::unique_lock<std::mutex> lock(session.mu);
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    freed += selected->d_out_cap + selected->d_input_cap +
             selected->act_q8_cap;
    lock.unlock();
    moe_cache_free_device(*selected);
    moe_cache_budget_unregister(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count) {
        return 0;
    }
    const int physical_device = info.devices[device].physical_device;
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, physical_device);
    }
    return freed;
}

void ggml_moe_cache_register(const void * owner) {
    if (ggml_moe_cache.owner && ggml_moe_cache.owner != owner) {
        return;
    }
    ggml_moe_cache.owner = owner;
    ggml_moe_cache.query_config = moe_cache_query_config;
    ggml_moe_cache.query_device = moe_cache_query_device;
    ggml_moe_cache.query_shape = moe_cache_query_shape;
    ggml_moe_cache.session_create = moe_cache_session_create;
    ggml_moe_cache.session_destroy = moe_cache_session_destroy;
    ggml_moe_cache.session_enter = moe_cache_session_enter;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.plan = moe_cache_plan;
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect = moe_cache_collect;
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.fused_begin = moe_cache_fused_begin;
    ggml_moe_cache.invalidate = moe_cache_invalidate;
}

#endif
