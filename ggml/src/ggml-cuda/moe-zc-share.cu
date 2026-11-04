#include "common.cuh"
#include "moe-zc-share.cuh"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

struct moe_zc_state {
    bool init = false;
    size_t budget = 0;
    size_t registered = 0;
    std::map<uintptr_t, uintptr_t> ranges;
    std::mutex ranges_mu;
    std::mutex register_mu;
    std::atomic<bool> any_registered{false};
};

static moe_zc_state g_zc;

static void moe_zc_init() {
    if (g_zc.init) {
        return;
    }
    g_zc.init = true;
    const char * gb = getenv("GGML_MOE_ZC_SHARE_GB");
    g_zc.budget = (size_t) ((gb ? atof(gb) : 64.0) * (1ull << 30));
    if (!gb) {
        // registered pages stay locked in RAM: default to half of it, so fill registration covers big MoEs
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        size_t free = 0, total = 0;
        if (cpu) {
            ggml_backend_dev_memory(cpu, &free, &total);
        }
        g_zc.budget = std::max(g_zc.budget, total / 2);
    }
}

// Registers the whole pages inside the tensor, in place: no copy, the page cache stays the only one.
static void moe_zc_register(const void * data, size_t bytes) {
    const uintptr_t page = 4096;
    const uintptr_t begin = ((uintptr_t) data + page - 1) & ~(page - 1);
    const uintptr_t end = ((uintptr_t) data + bytes) & ~(page - 1);
    if (end <= begin || g_zc.registered + (end - begin) > g_zc.budget) {
        return;
    }
    cudaError_t err = cudaHostRegister((void *) begin, end - begin,
            cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void * device = nullptr;
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer(&device, (void *) begin, 0);
        if (err != cudaSuccess || device != (void *) begin) {
            cudaHostUnregister((void *) begin);
            err = err != cudaSuccess ? err : cudaErrorNotSupported;
        }
    }
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("moe zc share: registering %.1f MiB failed: %s\n", (end - begin) / 1048576.0, cudaGetErrorString(err));
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_zc.ranges_mu);
        g_zc.ranges[begin] = end;
    }
    g_zc.any_registered = true;
    g_zc.registered += end - begin;
}

static bool moe_zc_registered(const char * begin, const char * end) {
    std::lock_guard<std::mutex> lock(g_zc.ranges_mu);
    auto it = g_zc.ranges.upper_bound((uintptr_t) begin);
    if (it == g_zc.ranges.begin()) {
        return false;
    }
    --it;
    return (uintptr_t) end <= it->second;
}

// A host copy must not straddle a registration edge (cudaErrorInvalidValue), and registrations end
// inside tensors, so a copy from host weights is split at every edge it crosses.
cudaError_t moe_zc_memcpy_h2d_async(void * dst, const void * src, size_t bytes, cudaStream_t stream) {
    if (!g_zc.any_registered) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    }
    static thread_local std::vector<size_t> cuts;
    cuts.clear();
    {
        std::lock_guard<std::mutex> lock(g_zc.ranges_mu);
        const uintptr_t begin = (uintptr_t) src, end = begin + bytes;
        auto it = g_zc.ranges.upper_bound(begin);
        if (it != g_zc.ranges.begin()) {
            --it;
        }
        for (; it != g_zc.ranges.end() && it->first < end; ++it) {
            for (const uintptr_t edge : { it->first, it->second }) {
                if (edge > begin && edge < end) {
                    cuts.push_back(edge - begin);
                }
            }
        }
    }
    cuts.push_back(bytes);
    size_t off = 0;
    cudaError_t err = cudaSuccess;
    for (size_t i = 0; i < cuts.size() && err == cudaSuccess; i++) {
        err = cudaMemcpyAsync((char *) dst + off, (const char *) src + off, cuts[i] - off, cudaMemcpyHostToDevice, stream);
        off = cuts[i];
    }
    return err;
}

void moe_zc_register_fill_source(const void * src, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_zc.register_mu);
    moe_zc_init();
    const uintptr_t page = 4096;
    const char * begin = (const char *) (((uintptr_t) src + page - 1) & ~(page - 1));
    const char * end = (const char *) (((uintptr_t) src + bytes) & ~(page - 1));
    if (end <= begin || moe_zc_registered(begin, end)) {
        return;
    }
    moe_zc_register(src, bytes);
}
