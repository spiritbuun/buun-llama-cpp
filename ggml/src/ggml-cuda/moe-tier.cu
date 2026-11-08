#include "moe-tier.cuh"

#if defined(GGML_USE_MUSA) || defined(GGML_USE_HIP) || !defined(__linux__)

void ggml_moe_tier_resolve(const ggml_moe_cache_route_table &, const ggml_moe_cache_route_table *,
        const ggml_tensor *, const ggml_tensor *, cudaStream_t) {
}

void ggml_moe_tier_scan_graph(const ggml_cgraph *, int) {
}

void ggml_moe_tier_forget(const void *, size_t) {
}

bool ggml_moe_tier_enabled() {
    return false;
}

ggml_moe_tier * ggml_moe_tier_create(int, int, cudaStream_t) {
    return nullptr;
}

void ggml_moe_tier_free(ggml_moe_tier *) {
}

void ggml_moe_tier_log_stats(const ggml_moe_tier *, int) {
}

ggml_moe_tier_route * ggml_moe_tier_route_add(ggml_moe_tier *, const void *, int64_t, size_t, const void **) {
    return nullptr;
}

void ggml_moe_tier_route_link(ggml_moe_tier_route *, int) {
}

const void * ggml_moe_tier_pin(ggml_moe_tier_route *, int32_t) {
    return nullptr;
}

void ggml_moe_tier_unpin(ggml_moe_tier_route *, int32_t) {
}

void ggml_moe_tier_set_vram(ggml_moe_tier_route *, int32_t, bool) {
}

void ggml_moe_tier_arena(const ggml_moe_tier_route *, const char ** lo, const char ** hi) {
    *lo = nullptr;
    *hi = nullptr;
}

void ggml_moe_tier_prefill(ggml_moe_tier_route *, uint32_t *, bool, char *, cudaStream_t) {
}

#else

#include "ggml-moe-store.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

static constexpr int       moe_tier_box_max      = 2*moe_cache_route_log_max; // the x and gate routes
static constexpr int       moe_tier_routes_max   = 1 << 16; // box keys hold the id above a 16-bit expert
static constexpr int       moe_tier_slots_min    = moe_tier_box_max + 64;
static constexpr size_t    moe_tier_chunk_bytes  = 1ull << 30;
static constexpr size_t    moe_tier_piece_bytes  = 1u << 20;
static constexpr int       moe_tier_readers      = 16;
static constexpr long long moe_tier_wait_cycles  = 60000000000LL; // ~30 s: a read that never lands
static constexpr int       moe_tier_hint_max     = 64;   // experts a decode token routes per layer
static constexpr int       moe_tier_hint_experts = 1024; // routers wider than this are not hinted

// Posted by resolve kernels, read by the tier thread; seq is odd while keys change.
struct moe_tier_box {
    uint32_t seq;
    uint32_t n;
    uint32_t hint_from; // keys from here on are the next layer's predicted experts
    uint32_t keys[moe_tier_box_max]; // route << 16 | expert
};

struct moe_tier_slot {
    ggml_moe_tier * tier = nullptr;
    int32_t route = -1;
    int32_t expert = -1;
    int pins = 0;
    bool loading = false;
    bool ref = false;
    bool hinted = false; // read for a hint and not demanded since
    std::chrono::steady_clock::time_point queued; // for the demand latency stat
};

struct ggml_moe_tier_route {
    ggml_moe_tier * tier = nullptr;
    int id = 0;
    const void ** table = nullptr;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    int file = -1;
    uint64_t offset = 0;
    // mapped: written by the tier, read by resolve kernels through d_ram
    const void ** ram = nullptr;
    const void ** d_ram = nullptr;
    std::vector<int32_t> slot; // resident or loading slot per expert, else -1
    std::unique_ptr<std::atomic<bool>[]> vram; // an eviction hint, written by the cache
    int group = -1; // routes of one group read the same ids

    ~ggml_moe_tier_route() {
        if (ram) {
            cudaFreeHost(ram);
        }
    }
};

static uint32_t moe_tier_key(int route, int32_t expert) {
    return (uint32_t) route << 16 | (uint32_t) expert;
}

enum moe_tier_stage_state {
    MOE_TIER_STAGE_FREE,
    MOE_TIER_STAGE_READING,
    MOE_TIER_STAGE_READY,
    MOE_TIER_STAGE_FAILED,
    MOE_TIER_STAGE_COPYING, // its bytes stay valid until copied fires
};

// A prefill ring entry: an expert read from the store on its way to a launch.
struct moe_tier_stage {
    ggml_moe_tier * tier = nullptr;
    uint32_t key = 0; // route << 16 | expert
    moe_tier_stage_state state = MOE_TIER_STAGE_FREE;
    char * buffer = nullptr;
    const char * data = nullptr;
    cudaEvent_t copied = nullptr;
};

struct ggml_moe_tier {
    int device = -1;
    cudaStream_t stream = nullptr;

    size_t budget = 0;
    size_t claimed = 0; // of g_moe_tier_claimed
    size_t n_expert = 0; // of the routes: slots past one each would never fill
    size_t expert_max = 0; // sizes the slots when the arena is reserved

    // Reserved whole at the first resolve so table entries in it are recognizable;
    // registered in chunks.
    char * arena = nullptr;
    size_t reserved = 0;
    size_t slot_bytes = 0;
    std::vector<moe_tier_slot> slots;
    std::vector<std::pair<int, int>> chunks; // first slot, slot count
    int n_ready = 0;
    std::vector<int> free_slots;
    int hand = 0;
    int n_pinned = 0; // slots with pins, kept clear of what a resolve may need

    std::vector<std::unique_ptr<ggml_moe_tier_route>> routes;
    std::unordered_map<uint64_t, int> files; // store handle per mapped file
    ggml_moe_store * store = nullptr;

    std::unordered_map<int, std::vector<ggml_moe_tier_route *>> groups;
    int hinted = -1; // the last group resolved: only its first launch may hint

    moe_tier_box * box = nullptr;
    moe_tier_box * d_box = nullptr;
    uint32_t * d_seq = nullptr;
    float * d_scores = nullptr;   // [moe_tier_hint_experts]
    int32_t * d_hint = nullptr;   // [moe_tier_hint_max]

    // Prefill ring, taken in order: large launches read experts outside RAM from the
    // store through it rather than through the arena, which keeps the decode set.
    char * ring = nullptr;
    std::vector<moe_tier_stage> stages;
    std::unordered_map<uint32_t, int> staged; // key -> stage
    int stage_hand = 0;
    std::mutex ring_mu;
    std::condition_variable ring_cv; // a ring stage or an arena load landed

    std::mutex mu;
    std::atomic<bool> stopping{false};
    std::atomic<bool> retry{false};
    std::thread thread;
    std::thread grower;
    std::condition_variable grow_cv; // a route added, or stopping

    std::atomic<long long> loads{0};
    std::atomic<long long> load_failures{0};
    std::atomic<long long> evictions{0};
    std::atomic<long long> sets{0};
    std::atomic<long long> miss_sets{0};
    std::atomic<long long> load_us{0};
    std::atomic<long long> hint_loads{0};
    std::atomic<long long> hint_used{0}; // hinted slots a demand then found
    std::atomic<long long> prefill_ram{0};   // prefill experts copied from the arena
    std::atomic<long long> prefill_admits{0}; // prefill reads into free arena slots
    std::atomic<long long> prefill_reads{0}; // prefill experts read into the ring
    std::atomic<long long> prefill_ahead{0}; // of them, read before their launch asked
    std::atomic<long long> prefill_us{0};
};

struct moe_tier_resolve_route {
    const void ** table;
    const void * const * ram;
    int64_t n_expert;
    uint32_t id;
};

static constexpr int moe_tier_resolve_routes = 8;

struct moe_tier_resolve_args {
    moe_tier_resolve_route routes[moe_tier_resolve_routes];
    int n_routes;
    int n_wait; // the first n_wait routes are read by the launch; the rest are prefetched
    int n_hinted; // routes from here on take their ids from hint
    const int32_t * hint;
    int n_hint;
    const int32_t * ids;
    int n_used;
    int n_rows;
    int64_t stride;
    const char * arena_lo;
    const char * arena_hi;
    moe_tier_box * box;
    uint32_t * seq;
};

// Table entries that are null or in the arena are re-read from ram[], which the tier
// nulls before reusing a slot: a resolve posts its keys before reading ram[], and the
// tier re-reads the box after nulling, so one of the two always sees the other.
static __global__ void moe_tier_resolve_kernel(const moe_tier_resolve_args a) {
    __shared__ uint32_t keys[moe_tier_box_max];
    __shared__ uint8_t  which[moe_tier_box_max];
    __shared__ int n;
    __shared__ int hint_from;
    if (threadIdx.x == 0) {
        n = 0;
        hint_from = moe_tier_box_max;
    }
    __syncthreads();
    for (int r = 0; r < a.n_routes; r++) {
        const bool hinted = r >= a.n_hinted;
        if (r == a.n_hinted) {
            if (threadIdx.x == 0) {
                hint_from = n;
            }
            __syncthreads();
        }
        const int n_ids = hinted ? a.n_hint : a.n_used * a.n_rows;
        for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
            const int32_t expert = hinted ? a.hint[i] : a.ids[(i / a.n_used) * a.stride + i % a.n_used];
            if (expert < 0 || expert >= a.routes[r].n_expert) {
                continue;
            }
            const char * p = (const char *) a.routes[r].table[expert];
            if (p && (p < a.arena_lo || p >= a.arena_hi)) {
                continue;
            }
            const int k = atomicAdd(&n, 1);
            if (k < moe_tier_box_max) {
                keys[k] = a.routes[r].id << 16 | (uint32_t) expert;
                which[k] = (uint8_t) r;
            }
        }
        // read routes take the box first; prefetches fill what is left
        __syncthreads();
    }
    const int count = min(n, moe_tier_box_max);
    if (count == 0) {
        return;
    }
    if (threadIdx.x == 0) {
        volatile moe_tier_box * box = a.box;
        const uint32_t seq = ++*a.seq;
        box->seq = 2*seq - 1;
        __threadfence_system();
        for (int i = 0; i < count; i++) {
            box->keys[i] = keys[i];
        }
        box->n = count;
        box->hint_from = min(hint_from, count);
        __threadfence_system();
        box->seq = 2*seq;
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        if (which[i] >= a.n_wait) {
            continue;
        }
        const moe_tier_resolve_route & route = a.routes[which[i]];
        const int32_t expert = keys[i] & 0xffff;
        const void * const volatile * ram = route.ram;
        const void * p;
        const long long start = clock64();
        while (!(p = ram[expert])) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(2000);
#endif
            if (clock64() - start > moe_tier_wait_cycles) {
                printf("moe tier: expert %d of route %u was never read\n", expert, route.id);
                __trap();
            }
        }
        route.table[expert] = p;
    }
}

static void moe_tier_start(ggml_moe_tier * tier);

static bool moe_tier_env_on(const char * name) {
    int64_t value = 1;
    moe_cache_env_i64(name, 0, 1, value);
    return value != 0;
}

static bool moe_tier_siblings_enabled() {
    static const bool enabled = moe_tier_env_on("GGML_CUDA_MOE_TIER_SIBLINGS");
    return enabled;
}

static bool moe_tier_prefill_enabled() {
    static const bool enabled = moe_tier_env_on("GGML_CUDA_MOE_TIER_PREFILL");
    return enabled;
}

static bool moe_tier_hint_enabled() {
    static const bool enabled = moe_tier_env_on("GGML_CUDA_MOE_TIER_HINT");
    return enabled;
}

// A layer's router as the graph runs it.
struct moe_tier_router {
    const char * weight = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int n_embd = 0;
    int n_expert = 0;
    size_t nb1 = 0;
    const float * bias = nullptr; // selection bias over sigmoid scores, else softmax order
};

static std::mutex g_moe_tier_router_mu;
static std::unordered_map<uint64_t, moe_tier_router> g_moe_tier_routers; // device << 32 | layer
static std::unordered_map<int, std::array<const void *, 3>> g_moe_tier_scanned; // per device: the last graph

static uint64_t moe_tier_router_key(int device, int layer) {
    return (uint64_t) device << 32 | (uint32_t) layer;
}

// The layer whose tensor this is, when its name is exactly blk.<layer>.<suffix> and it lives
// in a plain CUDA buffer, whose free drops it again.
static bool moe_tier_layer_tensor(const ggml_tensor * t, const char * suffix, int & layer) {
    int end = 0;
    return t && t->buffer && ggml_backend_buffer_is_cuda(t->buffer) &&
        sscanf(t->name, "blk.%d.%n", &layer, &end) == 1 && end > 0 && strcmp(t->name + end, suffix) == 0;
}

// Routers only change with the graph, so a replayed graph is not rescanned.
void ggml_moe_tier_scan_graph(const ggml_cgraph * cgraph, int device) {
    if (!moe_tier_hint_enabled() || cgraph->n_nodes == 0) {
        return;
    }
    const std::array<const void *, 3> shape = {
        (const void *) (intptr_t) cgraph->n_nodes, cgraph->nodes[0], cgraph->nodes[cgraph->n_nodes - 1]};
    std::lock_guard<std::mutex> lock(g_moe_tier_router_mu);
    auto scanned = g_moe_tier_scanned.find(device);
    if (scanned != g_moe_tier_scanned.end() && scanned->second == shape) {
        return;
    }
    g_moe_tier_scanned[device] = shape;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        int layer;
        if (node->op == GGML_OP_MUL_MAT && moe_tier_layer_tensor(node->src[0], "ffn_gate_inp.weight", layer)) {
            const ggml_tensor * w = node->src[0];
            if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16) ||
                w->nb[0] != ggml_type_size(w->type)) {
                continue;
            }
            moe_tier_router & router = g_moe_tier_routers[moe_tier_router_key(device, layer)];
            router.weight = (const char *) w->data;
            router.type = w->type;
            router.n_embd = (int) w->ne[0];
            router.n_expert = (int) w->ne[1];
            router.nb1 = w->nb[1];
        } else if (node->op == GGML_OP_ADD && moe_tier_layer_tensor(node->src[1], "exp_probs_b.bias", layer) &&
            node->src[1]->type == GGML_TYPE_F32) {
            g_moe_tier_routers[moe_tier_router_key(device, layer)].bias = (const float *) node->src[1]->data;
        }
    }
}

void ggml_moe_tier_forget(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    auto inside = [&](const void * p) { return (const char *) p >= lo && (const char *) p < hi; };
    std::lock_guard<std::mutex> lock(g_moe_tier_router_mu);
    bool forgot = false;
    for (auto it = g_moe_tier_routers.begin(); it != g_moe_tier_routers.end();) {
        if (inside(it->second.weight) || inside(it->second.bias)) {
            it = g_moe_tier_routers.erase(it);
            forgot = true;
        } else {
            ++it;
        }
    }
    // a graph of the same shape may now run other routers
    if (forgot) {
        g_moe_tier_scanned.clear();
    }
}

struct moe_tier_hint_args {
    moe_tier_router router;
    const float * x;
    float * scores;
    int32_t * hint;
    int k;
};

static __device__ __forceinline__ float moe_tier_hint_load(const char * row, ggml_type type, int i) {
    switch (type) {
        case GGML_TYPE_F16:  return __half2float(((const half *) row)[i]);
        case GGML_TYPE_BF16: return __bfloat162float(((const nv_bfloat16 *) row)[i]);
        default:             return ((const float *) row)[i];
    }
}

// One warp per router row.
static __global__ void moe_tier_hint_scores(const moe_tier_hint_args a) {
    const moe_tier_router & router = a.router;
    const int e = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (e >= router.n_expert) {
        return;
    }
    const char * row = router.weight + (size_t) e * router.nb1;
    float sum = 0.0f;
    for (int i = threadIdx.x % WARP_SIZE; i < router.n_embd; i += WARP_SIZE) {
        sum += moe_tier_hint_load(row, router.type, i) * a.x[i];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x % WARP_SIZE == 0) {
        a.scores[e] = router.bias ? 1.0f / (1.0f + expf(-sum)) + router.bias[e] : sum;
    }
}

// One warp: the k best scores, by repeated argmax.
static __global__ void moe_tier_hint_top(const moe_tier_hint_args a) {
    __shared__ float scores[moe_tier_hint_experts];
    const int n_expert = a.router.n_expert;
    const int lane = threadIdx.x;
    for (int e = lane; e < n_expert; e += WARP_SIZE) {
        scores[e] = a.scores[e];
    }
    __syncwarp();
    for (int k = 0; k < a.k; k++) {
        float best = -INFINITY;
        int index = n_expert;
        for (int e = lane; e < n_expert; e += WARP_SIZE) {
            if (scores[e] > best) {
                best = scores[e];
                index = e;
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffff, best, offset);
            const int other_index = __shfl_xor_sync(0xffffffff, index, offset);
            if (other > best || (other == best && other_index < index)) {
                best = other;
                index = other_index;
            }
        }
        if (lane == 0) {
            const bool found = index < n_expert;
            a.hint[k] = found ? index : -1;
            if (found) {
                scores[index] = -INFINITY;
            }
        }
        __syncwarp();
    }
}

// At decode, the first resolve of a layer predicts the next layer's experts by running
// that layer's router on this layer's input. Returns the hinted group, else -1.
static int moe_tier_hint(ggml_moe_tier * tier, int group, const ggml_tensor * input, const ggml_tensor * ids,
        cudaStream_t stream) {
    if (!moe_tier_hint_enabled() || group < 0 || group == tier->hinted) {
        return -1;
    }
    // the layer's later launches (down) skip; the next layer, hinted or not, resets
    tier->hinted = group;
    if (!input || input->type != GGML_TYPE_F32 ||
        input->nb[0] != sizeof(float) || ids->ne[1] != 1 || ids->ne[0] > moe_tier_hint_max ||
        tier->groups.find(group + 1) == tier->groups.end()) {
        return -1;
    }
    moe_tier_router router;
    {
        std::lock_guard<std::mutex> lock(g_moe_tier_router_mu);
        auto it = g_moe_tier_routers.find(moe_tier_router_key(tier->device, group + 1));
        if (it == g_moe_tier_routers.end()) {
            return -1;
        }
        router = it->second;
    }
    if (!router.weight || router.n_embd != input->ne[0] || router.n_expert > moe_tier_hint_experts) {
        return -1;
    }
    moe_tier_hint_args a;
    a.router = router;
    a.x = (const float *) input->data;
    a.scores = tier->d_scores;
    a.hint = tier->d_hint;
    a.k = (int) ids->ne[0];
    const int rows_per_block = 8;
    moe_tier_hint_scores<<<(router.n_expert + rows_per_block - 1) / rows_per_block, rows_per_block*WARP_SIZE, 0, stream>>>(a);
    moe_tier_hint_top<<<1, WARP_SIZE, 0, stream>>>(a);
    return group + 1;
}

void ggml_moe_tier_resolve(const ggml_moe_cache_route_table & x, const ggml_moe_cache_route_table * gate,
        const ggml_tensor * ids, const ggml_tensor * input, cudaStream_t stream) {
    moe_tier_resolve_args a = {};
    ggml_moe_tier * tier = nullptr;
    auto add = [&](const ggml_moe_tier_route & route) {
        for (int r = 0; r < a.n_routes; r++) {
            if (a.routes[r].id == (uint32_t) route.id) {
                return;
            }
        }
        if (a.n_routes < moe_tier_resolve_routes) {
            a.routes[a.n_routes++] = {route.table, route.d_ram, route.n_expert, (uint32_t) route.id};
        }
    };
    for (const ggml_moe_cache_route_table * table : {&x, gate}) {
        if (!table || !table->tier) {
            continue;
        }
        const ggml_moe_tier_route & route = *table->tier;
        GGML_ASSERT(!tier || tier == route.tier);
        tier = route.tier;
        add(route);
    }
    if (!tier) {
        return;
    }
    a.n_wait = a.n_routes;
    // the other matrices of the layer: their misses read while this launch waits
    const int group = x.tier ? x.tier->group : gate->tier->group;
    if (moe_tier_siblings_enabled() && group >= 0) {
        for (const ggml_moe_tier_route * route : tier->groups[group]) {
            add(*route);
        }
    }
    a.n_hinted = a.n_routes;
    const int hinted = moe_tier_hint(tier, group, input, ids, stream);
    if (hinted >= 0) {
        a.hint = tier->d_hint;
        a.n_hint = (int) ids->ne[0];
        for (const ggml_moe_tier_route * route : tier->groups[hinted]) {
            add(*route);
        }
    }
    // the box orders resolves and their kernels on one stream
    GGML_ASSERT(stream == tier->stream && "moe tier: routed kernels must run on the routing stream");
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    if (!tier->arena) {
        moe_tier_start(tier);
    }
    a.ids = (const int32_t *) ids->data;
    a.n_used = (int) ids->ne[0];
    a.n_rows = (int) ids->ne[1];
    a.stride = ids->nb[1] / sizeof(int32_t);
    a.arena_lo = tier->arena;
    a.arena_hi = tier->arena + tier->reserved;
    a.box = tier->d_box;
    a.seq = tier->d_seq;
    moe_tier_resolve_kernel<<<1, 256, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_moe_tier_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_MOE_TIER");
        return value && *value && strcmp(value, "off") != 0 && strcmp(value, "0") != 0;
    }();
    return enabled;
}

// auto budgets claimed by live tiers, so tiers do not each spend the same headroom, out of
// the headroom when the first of them started
static std::mutex g_moe_tier_claim_mu;
static size_t g_moe_tier_claimed = 0;
static size_t g_moe_tier_pool = 0;

// Host memory free to pin now, less the margin left for everything else.
static size_t moe_tier_headroom() {
    static const size_t margin = [] {
        int64_t margin_mb = 4096;
        moe_cache_env_i64("GGML_CUDA_MOE_TIER_MARGIN_MB", 0, 1 << 30, margin_mb);
        return (size_t) margin_mb << 20;
    }();
    const size_t room = ggml_moe_store_host_room();
    return room != SIZE_MAX && room > margin ? room - margin : 0;
}

// An auto budget is claimed until ggml_moe_tier_free returns it.
static size_t moe_tier_budget(int n_share, size_t & claimed) {
    const char * value = getenv("GGML_CUDA_MOE_TIER");
    if (strcmp(value, "auto") != 0) {
        return (size_t) strtoull(value, nullptr, 10) << 20;
    }
    std::lock_guard<std::mutex> lock(g_moe_tier_claim_mu);
    if (g_moe_tier_claimed == 0) {
        g_moe_tier_pool = moe_tier_headroom();
    }
    const size_t left = g_moe_tier_pool > g_moe_tier_claimed ? g_moe_tier_pool - g_moe_tier_claimed : 0;
    claimed = std::min(left, g_moe_tier_pool / (size_t) n_share);
    g_moe_tier_claimed += claimed;
    return claimed;
}

static bool moe_tier_register(ggml_moe_tier * tier, int first, int count) {
    char * base = tier->arena + (size_t) first * tier->slot_bytes;
    const size_t bytes = (size_t) count * tier->slot_bytes;
    cudaError_t error = cudaHostRegister(base, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
    void * device = nullptr;
    if (error == cudaSuccess) {
        error = cudaHostGetDevicePointer(&device, base, 0);
        // resolve kernels test table entries against the host range
        if (error != cudaSuccess || device != base) {
            cudaHostUnregister(base);
            error = error != cudaSuccess ? error : cudaErrorNotSupported;
        }
    }
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("moe tier: device %d: RAM arena stops at %.1f GiB: %s\n", tier->device,
                (double) first * tier->slot_bytes / (1ull << 30), cudaGetErrorString(error));
        return false;
    }
    std::lock_guard<std::mutex> lock(tier->mu);
    tier->chunks.push_back({first, count});
    for (int i = first + count - 1; i >= first; i--) {
        tier->free_slots.push_back(i);
    }
    tier->n_ready = first + count;
    return true;
}

static int moe_tier_chunk_slots(const ggml_moe_tier * tier) {
    return (int) std::max<size_t>(1, moe_tier_chunk_bytes / tier->slot_bytes);
}

// Pins the rest of the arena in the background, a chunk at a time, while the
// host still has room: the budget is a snapshot from before the first prompt.
// Pins no more slots than the routes have experts, waiting for later routes.
static void moe_tier_grow(ggml_moe_tier * tier) {
    ggml_cuda_set_device(tier->device);
    int first = tier->n_ready;
    while (first < (int) tier->slots.size()) {
        int cap = 0;
        {
            std::unique_lock<std::mutex> lock(tier->mu);
            tier->grow_cv.wait(lock, [&] {
                cap = (int) std::min(tier->slots.size(), std::max(tier->n_expert, (size_t) moe_tier_slots_min));
                return tier->stopping.load() || first < cap;
            });
        }
        if (tier->stopping.load()) {
            break;
        }
        const int count = std::min(moe_tier_chunk_slots(tier), cap - first);
        if (moe_tier_headroom() < (size_t) count * tier->slot_bytes) {
            GGML_LOG_INFO("moe tier: device %d: RAM arena stops at %.1f GiB: host headroom\n", tier->device,
                    (double) first * tier->slot_bytes / (1ull << 30));
            break;
        }
        if (!moe_tier_register(tier, first, count)) {
            break;
        }
        first += count;
    }
}

static void moe_tier_sort_unique(std::vector<uint32_t> & keys) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

// set gets every posted key, hints those of them that are next-layer predictions.
static bool moe_tier_read_set(const ggml_moe_tier * tier, uint32_t & seq, std::vector<uint32_t> & set,
        std::vector<uint32_t> & hints) {
    const volatile moe_tier_box * box = tier->box;
    seq = box->seq;
    if (seq & 1) {
        return false;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint32_t n = std::min((uint32_t) box->n, (uint32_t) moe_tier_box_max);
    const uint32_t hint_from = std::min((uint32_t) box->hint_from, n);
    set.resize(n);
    for (size_t i = 0; i < set.size(); i++) {
        set[i] = box->keys[i];
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (box->seq != seq) {
        return false;
    }
    hints.assign(set.begin() + hint_from, set.end());
    moe_tier_sort_unique(set);
    moe_tier_sort_unique(hints);
    return true;
}

static void moe_tier_publish(ggml_moe_tier_route & route, int32_t expert, const void * value) {
    *(const void * volatile *) &route.ram[expert] = value;
}

static uint64_t moe_tier_expert_offset(const ggml_moe_tier_route & route, int32_t expert) {
    return route.offset + (uint64_t) expert * route.expert_size;
}

// Called under tier->mu. The slot's expert starts where its direct read places it.
static const void * moe_tier_slot_data(const ggml_moe_tier * tier, int index) {
    const moe_tier_slot & slot = tier->slots[index];
    const uint64_t offset = moe_tier_expert_offset(*tier->routes[slot.route], slot.expert);
    return tier->arena + (size_t) index * tier->slot_bytes + (offset & (GGML_MOE_STORE_ALIGN - 1));
}

static void moe_tier_slot_assign(ggml_moe_tier * tier, int index, int32_t route, int32_t expert, bool loading) {
    moe_tier_slot & slot = tier->slots[index];
    slot.route = route;
    slot.expert = expert;
    slot.loading = loading;
    tier->routes[route]->slot[expert] = index;
}

static void moe_tier_loaded(void * user, bool ok) {
    moe_tier_slot & slot = *(moe_tier_slot *) user;
    ggml_moe_tier * tier = slot.tier;
    const int index = (int) (&slot - tier->slots.data());
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(tier->mu);
        ggml_moe_tier_route & route = *tier->routes[slot.route];
        slot.loading = false;
        if (ok) {
            moe_tier_publish(route, slot.expert, moe_tier_slot_data(tier, index));
            tier->loads++;
            tier->load_us += std::chrono::duration_cast<std::chrono::microseconds>(now - slot.queued).count();
        } else {
            route.slot[slot.expert] = -1;
            slot.route = -1;
            tier->free_slots.push_back(index);
            tier->load_failures++;
            tier->retry = true;
        }
    }
    // a prefill may wait on it
    tier->ring_cv.notify_all();
}

struct moe_tier_victim {
    int slot;
    int32_t route;
    int32_t expert;
};

struct moe_tier_read {
    int file;
    uint64_t offset;
    size_t size;
    int slot;
    int priority;
};

// Starts reads into their slots; outside tier->mu, since a rejected read completes inside the call.
static void moe_tier_issue(ggml_moe_tier * tier, const std::vector<moe_tier_read> & reads) {
    const auto now = std::chrono::steady_clock::now();
    for (const moe_tier_read & read : reads) {
        tier->slots[read.slot].queued = now;
        ggml_moe_store_read(tier->store, read.file, read.offset, read.size,
                tier->arena + (size_t) read.slot * tier->slot_bytes,
                read.priority, moe_tier_loaded, &tier->slots[read.slot]);
    }
}

// Called under tier->mu.
static int moe_tier_free_slot(ggml_moe_tier * tier) {
    if (tier->free_slots.empty()) {
        return -1;
    }
    const int index = tier->free_slots.back();
    tier->free_slots.pop_back();
    return index;
}

// Called under tier->mu. A prefill read takes a free slot, so prompts fill the arena,
// but never evicts: it lands unreferenced, and CLOCK takes it before the decode set.
static bool moe_tier_admit(ggml_moe_tier * tier, ggml_moe_tier_route & route, int32_t expert,
        std::vector<moe_tier_read> & reads) {
    const int index = moe_tier_free_slot(tier);
    if (index < 0) {
        return false;
    }
    moe_tier_slot_assign(tier, index, route.id, expert, true);
    tier->slots[index].ref = false;
    tier->slots[index].hinted = false;
    reads.push_back({route.file, moe_tier_expert_offset(route, expert), route.expert_size, index, GGML_MOE_STORE_PREFILL});
    return true;
}

// Called under tier->mu. A free slot, or one taken by CLOCK from an expert outside
// set: VRAM-resident experts first, then any not referenced since the hand passed.
// A hint takes only what CLOCK would, without spending other slots' second chance.
static int moe_tier_take_slot(ggml_moe_tier * tier, const std::vector<uint32_t> & set, bool hint,
        std::vector<moe_tier_victim> & victims) {
    const int free_index = moe_tier_free_slot(tier);
    if (free_index >= 0) {
        return free_index;
    }
    for (int step = 0; step < 2*tier->n_ready; step++) {
        const int index = tier->hand;
        tier->hand = (tier->hand + 1) % tier->n_ready;
        moe_tier_slot & slot = tier->slots[index];
        if (slot.route < 0 || slot.loading || slot.pins > 0 ||
            std::binary_search(set.begin(), set.end(), moe_tier_key(slot.route, slot.expert))) {
            continue;
        }
        ggml_moe_tier_route & route = *tier->routes[slot.route];
        if (slot.ref && !route.vram[slot.expert].load(std::memory_order_relaxed)) {
            slot.ref = hint;
            continue;
        }
        moe_tier_publish(route, slot.expert, nullptr);
        route.slot[slot.expert] = -1;
        victims.push_back({index, slot.route, slot.expert});
        tier->evictions++;
        return index;
    }
    return -1;
}

// Gives every key of the posted set a resident or loading slot, hints last and only
// while slots last, then starts the reads.
static void moe_tier_serve(ggml_moe_tier * tier, uint32_t seq, std::vector<uint32_t> & set,
        std::vector<uint32_t> & hints) {
    std::vector<std::pair<int, bool>> loading; // slot, hinted
    std::vector<moe_tier_victim> victims;
    std::vector<moe_tier_read> reads;
    long long hint_reads = 0;
    {
        std::lock_guard<std::mutex> lock(tier->mu);
        // false only when no slot is left; a mispredicted hint gets no second chance from
        // CLOCK, a demand marks it
        auto place = [&](uint32_t key, bool hint) {
            const int32_t route_id = (int32_t) (key >> 16);
            const int32_t expert = (int32_t) (key & 0xffff);
            if (route_id >= (int) tier->routes.size() || expert >= tier->routes[route_id]->n_expert) {
                return true;
            }
            ggml_moe_tier_route & route = *tier->routes[route_id];
            if (route.slot[expert] >= 0) {
                moe_tier_slot & slot = tier->slots[route.slot[expert]];
                if (!hint) {
                    slot.ref = true;
                    tier->hint_used += slot.hinted ? 1 : 0;
                    slot.hinted = false;
                }
                return true;
            }
            const int index = moe_tier_take_slot(tier, set, hint, victims);
            if (index < 0) {
                return false;
            }
            moe_tier_slot_assign(tier, index, route_id, expert, true);
            tier->slots[index].ref = !hint;
            tier->slots[index].hinted = hint;
            loading.push_back({index, hint});
            return true;
        };
        for (;;) {
            victims.clear();
            bool short_of_slots = false;
            for (const uint32_t key : set) {
                if (!std::binary_search(hints.begin(), hints.end(), key) && !place(key, false)) {
                    short_of_slots = true;
                    break;
                }
            }
            if (!short_of_slots) {
                for (const uint32_t key : hints) {
                    if (!place(key, true)) {
                        break;
                    }
                }
            }
            tier->retry = short_of_slots;
            if (victims.empty()) {
                break;
            }
            // A resolve that posts after this fence reads the nulled entries; one that
            // posted before it is in the box now. Hand back victims the newer set names.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            uint32_t now;
            while (!moe_tier_read_set(tier, now, set, hints)) {
            }
            if (now == seq) {
                break;
            }
            seq = now;
            for (const moe_tier_victim & victim : victims) {
                if (!std::binary_search(set.begin(), set.end(),
                        moe_tier_key(victim.route, victim.expert))) {
                    continue;
                }
                const moe_tier_slot & slot = tier->slots[victim.slot];
                tier->routes[slot.route]->slot[slot.expert] = -1;
                loading.erase(std::find_if(loading.begin(), loading.end(),
                        [&](const std::pair<int, bool> & load) { return load.first == victim.slot; }));
                moe_tier_slot_assign(tier, victim.slot, victim.route, victim.expert, false);
                tier->slots[victim.slot].hinted = false;
                moe_tier_publish(*tier->routes[victim.route], victim.expert, moe_tier_slot_data(tier, victim.slot));
                tier->evictions--;
            }
        }
        // routes may grow once the lock drops
        for (const auto & [index, hint] : loading) {
            const moe_tier_slot & slot = tier->slots[index];
            const ggml_moe_tier_route & route = *tier->routes[slot.route];
            reads.push_back({route.file, moe_tier_expert_offset(route, slot.expert), route.expert_size, index,
                    hint ? GGML_MOE_STORE_HINT : GGML_MOE_STORE_DEMAND});
            hint_reads += hint;
        }
    }
    tier->sets++;
    tier->miss_sets += (long long) reads.size() > hint_reads;
    tier->hint_loads += hint_reads;
    moe_tier_issue(tier, reads);
}

// Polls the box: spins while sets arrive, then backs off to sleeps.
static void moe_tier_thread(ggml_moe_tier * tier) {
    using clock = std::chrono::steady_clock;
    std::vector<uint32_t> set;
    std::vector<uint32_t> hints;
    uint32_t seen = 0;
    clock::time_point active = clock::now();
    while (!tier->stopping.load()) {
        const uint32_t seq = ((const volatile moe_tier_box *) tier->box)->seq;
        if (seq != seen || tier->retry.load()) {
            const bool fresh = seq != seen;
            uint32_t consistent;
            if (moe_tier_read_set(tier, consistent, set, hints)) {
                seen = consistent;
                moe_tier_serve(tier, consistent, set, hints);
                active = clock::now();
            }
            if (!fresh) {
                // a retry: slots free only as reads land and pins drop
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            continue;
        }
        const auto idle = clock::now() - active;
        if (idle > std::chrono::seconds(1)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } else if (idle > std::chrono::milliseconds(100)) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        } else if (idle > std::chrono::microseconds(200)) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
}

void ggml_moe_tier_free(ggml_moe_tier * tier) {
    if (!tier) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(tier->mu);
        tier->stopping = true;
    }
    tier->grow_cv.notify_all();
    if (tier->thread.joinable()) {
        tier->thread.join();
    }
    if (tier->grower.joinable()) {
        tier->grower.join();
    }
    // reads still in flight land in the arena and the ring
    ggml_moe_store_free(tier->store);
    for (const moe_tier_stage & stage : tier->stages) {
        cudaEventDestroy(stage.copied);
    }
    if (tier->ring) {
        cudaFreeHost(tier->ring);
    }
    for (const auto & chunk : tier->chunks) {
        cudaHostUnregister(tier->arena + (size_t) chunk.first * tier->slot_bytes);
    }
    if (tier->arena) {
        munmap(tier->arena, tier->reserved);
    }
    tier->routes.clear();
    if (tier->box) {
        cudaFreeHost(tier->box);
    }
    if (tier->d_seq) {
        cudaFree(tier->d_seq);
    }
    if (tier->d_scores) {
        cudaFree(tier->d_scores);
    }
    if (tier->d_hint) {
        cudaFree(tier->d_hint);
    }
    (void) cudaGetLastError();
    {
        std::lock_guard<std::mutex> lock(g_moe_tier_claim_mu);
        g_moe_tier_claimed -= tier->claimed;
    }
    delete tier;
}

// Room for any expert of expert_size bytes wherever its direct read starts.
static size_t moe_tier_slot_bytes(size_t expert_size) {
    return ggml_moe_store_span(GGML_MOE_STORE_ALIGN - 1, expert_size);
}

ggml_moe_tier * ggml_moe_tier_create(int device, int n_share, cudaStream_t stream) {
    if (!ggml_moe_tier_enabled()) {
        return nullptr;
    }
    auto tier = new ggml_moe_tier;
    tier->device = device;
    tier->stream = stream;
    tier->budget = moe_tier_budget(n_share, tier->claimed);

    ggml_cuda_set_device(device);
    cudaError_t error = cudaHostAlloc((void **) &tier->box, sizeof(moe_tier_box), cudaHostAllocMapped | cudaHostAllocPortable);
    if (error == cudaSuccess) {
        memset(tier->box, 0, sizeof(moe_tier_box));
        error = cudaHostGetDevicePointer((void **) &tier->d_box, tier->box, 0);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc((void **) &tier->d_seq, sizeof(uint32_t));
    }
    if (error == cudaSuccess) {
        error = cudaMemset(tier->d_seq, 0, sizeof(uint32_t));
    }
    if (error == cudaSuccess) {
        error = cudaMalloc((void **) &tier->d_scores, moe_tier_hint_experts * sizeof(float));
    }
    if (error == cudaSuccess) {
        error = cudaMalloc((void **) &tier->d_hint, moe_tier_hint_max * sizeof(int32_t));
    }
    int64_t readers = moe_tier_readers;
    moe_cache_env_i64("GGML_CUDA_MOE_TIER_READERS", 1, 256, readers);
    tier->store = error == cudaSuccess ? ggml_moe_store_init((int) readers, moe_tier_piece_bytes) : nullptr;
    if (!tier->store) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("moe tier: device %d: setup failed; tier off\n", device);
        ggml_moe_tier_free(tier);
        return nullptr;
    }
    return tier;
}

// Pins the prefill ring out of the budget; without one, prefill copies stay the caller's.
static void moe_tier_ring_init(ggml_moe_tier * tier) {
    if (!moe_tier_prefill_enabled()) {
        return;
    }
    int64_t ring_mb = 1024;
    moe_cache_env_i64("GGML_CUDA_MOE_TIER_RING_MB", 0, 1 << 20, ring_mb);
    // the arena keeps at least three quarters of the budget, and its minimum slots
    const size_t arena_min = (size_t) moe_tier_slots_min * tier->slot_bytes;
    const size_t spare = tier->budget > arena_min ? tier->budget - arena_min : 0;
    const size_t n_stages = std::min({(size_t) ring_mb << 20, tier->budget / 4, spare}) / tier->slot_bytes;
    if (n_stages < 8) {
        return;
    }
    const size_t bytes = n_stages * tier->slot_bytes;
    if (cudaHostAlloc((void **) &tier->ring, bytes, cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        tier->ring = nullptr;
        GGML_LOG_WARN("moe tier: device %d: cannot pin the prefill ring\n", tier->device);
        return;
    }
    tier->budget -= bytes;
    tier->stages.resize(n_stages);
    for (size_t i = 0; i < n_stages; i++) {
        moe_tier_stage & stage = tier->stages[i];
        stage.tier = tier;
        stage.buffer = tier->ring + i * tier->slot_bytes;
        CUDA_CHECK(cudaEventCreateWithFlags(&stage.copied, cudaEventDisableTiming));
    }
}

// Reserves the arena for the routes added so far and pins its first chunk.
static void moe_tier_start(ggml_moe_tier * tier) {
    {
        std::lock_guard<std::mutex> lock(tier->mu);
        tier->slot_bytes = moe_tier_slot_bytes(tier->expert_max);
    }
    moe_tier_ring_init(tier);
    const size_t n_slots = tier->budget / tier->slot_bytes;
    void * arena = mmap(nullptr, n_slots * tier->slot_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (arena == MAP_FAILED) {
        GGML_ABORT("moe tier: device %d: cannot reserve the RAM arena; run with GGML_CUDA_MOE_TIER=off", tier->device);
    }
    tier->reserved = n_slots * tier->slot_bytes;
    tier->slots.resize(n_slots);
    for (moe_tier_slot & slot : tier->slots) {
        slot.tier = tier;
    }
    tier->arena = (char *) arena;
    // the minimum is pinned now: a resolve must always find slots outside its own set
    const int first = (int) std::min(n_slots, (size_t) std::max(moe_tier_slots_min, moe_tier_chunk_slots(tier)));
    if (!moe_tier_register(tier, 0, first)) {
        GGML_ABORT("moe tier: device %d: cannot pin the RAM arena; run with GGML_CUDA_MOE_TIER=off", tier->device);
    }
    tier->thread = std::thread(moe_tier_thread, tier);
    tier->grower = std::thread(moe_tier_grow, tier);
    GGML_LOG_INFO("moe tier: device %d: %.1f GiB RAM arena, %zu slots of %.2f MiB\n", tier->device,
            (double) tier->reserved / (1ull << 30), n_slots, (double) tier->slot_bytes / (1 << 20));
}

void ggml_moe_tier_log_stats(const ggml_moe_tier * tier, int device) {
    if (!tier) {
        return;
    }
    const long long loads = tier->loads.load();
    GGML_LOG_INFO("moe tier: device %d: %d slots pinned, %lld sets (%lld missed), %lld loads (%lld hinted, %lld used, "
            "%.1f GiB, %.2f ms avg), %lld evictions, %lld failed\n",
            device, tier->n_ready, tier->sets.load(), tier->miss_sets.load(), loads, tier->hint_loads.load(), tier->hint_used.load(),
            (double) loads * tier->slot_bytes / (1ull << 30), loads ? tier->load_us.load() / 1000.0 / loads : 0.0,
            tier->evictions.load(), tier->load_failures.load());
    if (tier->prefill_us.load()) {
        GGML_LOG_INFO("moe tier: device %d: prefill %lld from RAM (%lld read into it), %lld read through the ring (%lld ahead), %.1f s\n",
                device, tier->prefill_ram.load(), tier->prefill_admits.load(), tier->prefill_reads.load(), tier->prefill_ahead.load(),
                tier->prefill_us.load() / 1e6);
    }
}

ggml_moe_tier_route * ggml_moe_tier_route_add(ggml_moe_tier * tier, const void * host_base,
        int64_t n_expert, size_t expert_size, const void ** table) {
    char path[64];
    uint64_t file_id = 0;
    uint64_t offset = 0;
    if (n_expert > 0x10000 || tier->budget / moe_tier_slot_bytes(expert_size) < (size_t) moe_tier_slots_min ||
        !ggml_moe_store_find(host_base, (size_t) n_expert * expert_size, path, sizeof(path), &file_id, &offset)) {
        return nullptr;
    }
    auto route = std::make_unique<ggml_moe_tier_route>();
    route->tier = tier;
    route->table = table;
    route->n_expert = n_expert;
    route->expert_size = expert_size;
    route->offset = offset;
    route->slot.assign(n_expert, -1);
    route->vram.reset(new std::atomic<bool>[n_expert]());
    if (cudaHostAlloc((void **) &route->ram, n_expert * sizeof(void *), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        route->ram = nullptr;
        (void) cudaGetLastError();
        return nullptr;
    }
    memset(route->ram, 0, n_expert * sizeof(void *));
    if (cudaHostGetDevicePointer((void **) &route->d_ram, route->ram, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(tier->mu);
    // slots sized at the first resolve hold no larger expert
    if (tier->slot_bytes && moe_tier_slot_bytes(expert_size) > tier->slot_bytes) {
        return nullptr;
    }
    auto file = tier->files.find(file_id);
    if (file == tier->files.end()) {
        file = tier->files.emplace(file_id, ggml_moe_store_open(tier->store, path)).first;
    }
    if (file->second < 0 || (int) tier->routes.size() >= moe_tier_routes_max) {
        return nullptr;
    }
    route->file = file->second;
    route->id = (int) tier->routes.size();
    tier->expert_max = std::max(tier->expert_max, expert_size);
    tier->n_expert += (size_t) n_expert;
    tier->grow_cv.notify_all();
    tier->routes.push_back(std::move(route));
    return tier->routes.back().get();
}

void ggml_moe_tier_route_link(ggml_moe_tier_route * route, int group) {
    ggml_moe_tier * tier = route->tier;
    std::lock_guard<std::mutex> lock(tier->mu);
    route->group = group;
    tier->groups[group].push_back(route);
}

// Called under tier->mu.
static const void * moe_tier_pin_locked(ggml_moe_tier_route * route, int32_t expert) {
    ggml_moe_tier * tier = route->tier;
    const int index = route->slot[expert];
    if (index < 0 || tier->slots[index].loading) {
        return nullptr;
    }
    moe_tier_slot & slot = tier->slots[index];
    if (slot.pins == 0) {
        // fills may wait behind a resolve, which must still find a whole set of slots
        if (tier->n_pinned + moe_tier_box_max >= tier->n_ready) {
            return nullptr;
        }
        tier->n_pinned++;
    }
    slot.pins++;
    return moe_tier_slot_data(tier, index);
}

const void * ggml_moe_tier_pin(ggml_moe_tier_route * route, int32_t expert) {
    std::lock_guard<std::mutex> lock(route->tier->mu);
    return moe_tier_pin_locked(route, expert);
}

void ggml_moe_tier_unpin(ggml_moe_tier_route * route, int32_t expert) {
    ggml_moe_tier * tier = route->tier;
    std::lock_guard<std::mutex> lock(tier->mu);
    const int index = route->slot[expert];
    GGML_ASSERT(index >= 0 && tier->slots[index].pins > 0);
    if (--tier->slots[index].pins == 0) {
        tier->n_pinned--;
    }
}

void ggml_moe_tier_set_vram(ggml_moe_tier_route * route, int32_t expert, bool vram) {
    route->vram[expert].store(vram, std::memory_order_relaxed);
}

void ggml_moe_tier_arena(const ggml_moe_tier_route * route, const char ** lo, const char ** hi) {
    *lo = route->tier->arena;
    *hi = route->tier->arena + route->tier->reserved;
}

static void moe_tier_staged_done(void * user, bool ok) {
    moe_tier_stage & stage = *(moe_tier_stage *) user;
    {
        std::lock_guard<std::mutex> lock(stage.tier->ring_mu);
        stage.state = ok ? MOE_TIER_STAGE_READY : MOE_TIER_STAGE_FAILED;
    }
    stage.tier->ring_cv.notify_all();
}

// Called under ring_mu: the stage at the hand once it can take a new read, else -1.
// Copied and failed stages free; with keep, so do read stages outside it (reads ahead
// of a launch that turned out not to need them). keep: the sorted keys still wanted.
static int moe_tier_ring_take(ggml_moe_tier * tier, const std::vector<uint32_t> * keep) {
    moe_tier_stage & stage = tier->stages[tier->stage_hand];
    switch (stage.state) {
        case MOE_TIER_STAGE_READING:
            return -1;
        case MOE_TIER_STAGE_READY:
            if (!keep || std::binary_search(keep->begin(), keep->end(), stage.key)) {
                return -1;
            }
            break;
        case MOE_TIER_STAGE_COPYING: {
            const cudaError_t status = cudaEventQuery(stage.copied);
            if (status == cudaErrorNotReady) {
                return -1;
            }
            CUDA_CHECK(status);
            break;
        }
        default:
            break;
    }
    if (stage.state != MOE_TIER_STAGE_FREE) {
        tier->staged.erase(stage.key);
        stage.state = MOE_TIER_STAGE_FREE;
    }
    const int index = tier->stage_hand;
    tier->stage_hand = (tier->stage_hand + 1) % (int) tier->stages.size();
    return index;
}

void ggml_moe_tier_prefill(ggml_moe_tier_route * route, uint32_t * want, bool ahead, char * dst, cudaStream_t stream) {
    ggml_moe_tier * tier = route->tier;
    if (!moe_tier_prefill_enabled()) {
        return;
    }
    // a prompt may come before any resolve
    if (!tier->arena) {
        moe_tier_start(tier);
    }
    if (!tier->ring) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const size_t size = route->expert_size;
    const auto wanted = [&](int32_t e) { return (want[e / 32] >> (e % 32)) & 1u; };
    const auto copy = [&](int32_t e, const void * src) {
        CUDA_CHECK(cudaMemcpyAsync(dst + (size_t) e * size, src, size, cudaMemcpyHostToDevice, stream));
        want[e / 32] &= ~(1u << (e % 32));
    };

    std::vector<int32_t> pinned;
    std::vector<std::pair<int32_t, const void *>> ready; // pinned, copied once tier->mu is released
    // under tier->mu
    const auto take_ram = [&](int32_t e) {
        const void * ram = moe_tier_pin_locked(route, e);
        if (ram) {
            ready.push_back({e, ram});
        }
        return ram != nullptr;
    };
    const auto copy_ready = [&]() {
        for (const auto & [e, ram] : ready) {
            copy(e, ram);
            pinned.push_back(e);
        }
        ready.clear();
    };
    std::vector<int32_t> filling; // loading into the arena
    std::vector<uint32_t> misses; // for the ring, ascending
    std::vector<moe_tier_read> fills;
    // Read ahead: the experts outside VRAM and RAM of the layer's later routes, which
    // take the same ids, then any of the next layer's, in launch order.
    std::vector<std::pair<ggml_moe_tier_route *, uint32_t>> next;
    std::unique_lock<std::mutex> tier_lock(tier->mu);
    if (ahead && route->group >= 0) {
        for (int group = route->group; group <= route->group + 1; group++) {
            auto found = tier->groups.find(group);
            if (found == tier->groups.end()) {
                continue;
            }
            const auto & routes = found->second;
            auto it = routes.begin();
            if (group == route->group) {
                it = std::find(routes.begin(), routes.end(), route);
                it += it != routes.end();
            }
            for (; it != routes.end(); ++it) {
                ggml_moe_tier_route * r = *it;
                for (int32_t e = 0; e < r->n_expert; e++) {
                    if ((group != route->group || wanted(e)) && r->slot[e] < 0 && !r->vram[e].load(std::memory_order_relaxed)) {
                        next.push_back({r, moe_tier_key(r->id, e)});
                    }
                }
            }
        }
    }
    // Experts in RAM copy from the arena; the rest load into its free slots while it has
    // them, then read through the ring. Reads ahead take free slots after them.
    for (int32_t e = 0; e < route->n_expert; e++) {
        if (!wanted(e)) {
            continue;
        }
        const int index = route->slot[e];
        if (index >= 0 && tier->slots[index].loading) {
            filling.push_back(e);
        } else if (!take_ram(e)) {
            if (index < 0 && moe_tier_admit(tier, *route, e, fills)) {
                filling.push_back(e);
            } else {
                misses.push_back(moe_tier_key(route->id, e));
            }
        }
    }
    next.erase(std::remove_if(next.begin(), next.end(), [&](const auto & n) {
        return moe_tier_admit(tier, *n.first, (int32_t) (n.second & 0xffff), fills);
    }), next.end());
    tier_lock.unlock();
    moe_tier_issue(tier, fills);
    copy_ready();

    size_t ahead_next = 0;
    long long reads = 0;
    long long ahead_reads = 0;
    std::unique_lock<std::mutex> lock(tier->ring_mu);
    for (const uint32_t k : misses) {
        ahead_reads += tier->staged.count(k);
    }
    while (!misses.empty() || !filling.empty()) {
        if (!filling.empty()) {
            {
                // ring_mu then tier->mu: no path takes them the other way round
                std::lock_guard<std::mutex> lock_tier(tier->mu);
                filling.erase(std::remove_if(filling.begin(), filling.end(), [&](int32_t e) {
                    const int index = route->slot[e];
                    if (index >= 0 && tier->slots[index].loading) {
                        return false;
                    }
                    // a failed or evicted load stays wanted: the caller copies it from the mapping;
                    // one past the pin cap reads through the ring
                    if (!take_ram(e) && index >= 0) {
                        const uint32_t k = moe_tier_key(route->id, e);
                        misses.insert(std::lower_bound(misses.begin(), misses.end(), k), k);
                    }
                    return true;
                }), filling.end());
            }
            copy_ready();
        }
        std::vector<std::pair<moe_tier_stage *, const ggml_moe_tier_route *>> issue;
        // false once the ring is full this pass
        const auto stage_read = [&](uint32_t k, const ggml_moe_tier_route * r, const std::vector<uint32_t> * keep) {
            const int index = moe_tier_ring_take(tier, keep);
            if (index < 0) {
                return false;
            }
            moe_tier_stage & stage = tier->stages[index];
            stage.key = k;
            stage.state = MOE_TIER_STAGE_READING;
            tier->staged[k] = index;
            issue.push_back({&stage, r});
            return true;
        };
        bool issuing = true; // misses get stages in order
        for (size_t i = 0; i < misses.size();) {
            auto found = tier->staged.find(misses[i]);
            if (found == tier->staged.end()) {
                issuing = issuing && stage_read(misses[i], route, &misses);
                reads += issuing;
                i++;
                continue;
            }
            moe_tier_stage & stage = tier->stages[found->second];
            if (stage.state == MOE_TIER_STAGE_READY || stage.state == MOE_TIER_STAGE_COPYING) {
                copy(misses[i] & 0xffff, stage.data);
                CUDA_CHECK(cudaEventRecord(stage.copied, stream));
                stage.state = MOE_TIER_STAGE_COPYING;
            } else if (stage.state == MOE_TIER_STAGE_FAILED) {
                // left wanted: the caller copies it from the mapping
                tier->staged.erase(found);
                stage.state = MOE_TIER_STAGE_FREE;
            } else {
                i++;
                continue;
            }
            misses.erase(misses.begin() + i);
        }
        // reads ahead take only what the ring has spare
        for (; issuing && ahead_next < next.size(); ahead_next++) {
            const auto & [r, k] = next[ahead_next];
            if (!tier->staged.count(k) && !stage_read(k, r, nullptr)) {
                break;
            }
        }
        if (!issue.empty()) {
            // a rejected read completes inside the call, under ring_mu
            lock.unlock();
            for (const auto & [stage, r] : issue) {
                stage->data = (const char *) ggml_moe_store_read(tier->store, r->file, moe_tier_expert_offset(*r, stage->key & 0xffff),
                        r->expert_size, stage->buffer, GGML_MOE_STORE_PREFILL, moe_tier_staged_done, stage);
            }
            lock.lock();
        } else if (!misses.empty() || !filling.empty()) {
            // landed reads notify; finished copies free stages without notice
            tier->ring_cv.wait_for(lock, std::chrono::microseconds(200));
        }
    }
    lock.unlock();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const int32_t e : pinned) {
        ggml_moe_tier_unpin(route, e);
    }
    tier->prefill_ram += (long long) pinned.size();
    tier->prefill_admits += (long long) fills.size();
    tier->prefill_reads += reads + ahead_reads;
    tier->prefill_ahead += ahead_reads;
    tier->prefill_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

#endif
