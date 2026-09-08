#include "allreduce-oneshot.cuh"

#include <time.h>

#include <cstdlib>
#include <cstring>

// Layout of the shared pinned block:
//   data  [n_slots][n_ranks][max_bytes]      the slices
//   flags [n_slots][n_ranks][2] x 64 bytes   arrival token, done token (one cache line each)
static constexpr int    GGML_CUDA_AR1_MAX_RANKS = GGML_CUDA_MAX_DEVICES;
static constexpr int    GGML_CUDA_AR1_SLOTS     = 4;
static constexpr size_t GGML_CUDA_AR1_LINE      = 128;

struct ggml_cuda_ar_oneshot {
    int    n_ranks = 0;
    int    devices[GGML_CUDA_AR1_MAX_RANKS] = {};
    size_t max_bytes = 0;

    void * host_base = nullptr;   // cudaHostAlloc(Mapped | Portable)
    size_t data_bytes = 0;        // n_slots * n_ranks * max_bytes
    size_t flag_bytes = 0;        // + one 64-byte line per rank at the end: spin timeout diagnostics

    // device-visible pointers to the host block, per rank (mapped memory may map differently per device)
    char * dev_base[GGML_CUDA_AR1_MAX_RANKS] = {};
    // device-side per-rank state: [0] launch counter (token), [1] blocks finished writing, [2] blocks finished reading
    int  * dev_state[GGML_CUDA_AR1_MAX_RANKS] = {};

    // GGML_CUDA_AR1_TRACE=lo:hi — per-rank ring of kernel timelines (token, entry, arrived, peers seen, end) in
    // device globaltimer ns; clock_offset[i] = device time - host CLOCK_MONOTONIC at init, so ranks compare
    unsigned long long * host_trace = nullptr;
    unsigned long long * dev_trace[GGML_CUDA_AR1_MAX_RANKS] = {};
    long long clock_offset[GGML_CUDA_AR1_MAX_RANKS] = {};
    int trace_lo = -1, trace_hi = -1, trace_printed = 0;
};
static constexpr int AR1_TRACE_N = 256;
static constexpr int AR1_TRACE_W = 5;

static __global__ void k_ar1_clock(unsigned long long * out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
}

// flags live in mapped host memory: system-scope acquire loads and release stores, with a system fence after
// each store so the flag is pushed out while the same thread keeps polling
static __device__ __forceinline__ int ar1_load_flag(const int * p) {
    int v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void ar1_store_flag(int * p, int v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
    __threadfence_system();
}

// flag address of (slot, rank, which) inside the host block
static __device__ __forceinline__ int * ar1_flag(char * base, size_t data_bytes, int n_ranks, int slot, int rank, int which) {
    return (int *) (base + data_bytes + ((size_t) (slot * n_ranks + rank) * 2 + which) * GGML_CUDA_AR1_LINE);
}
// diagnostics: rank's timeout word lives after all slot flags
static __device__ __forceinline__ int * ar1_timeout(char * base, size_t data_bytes, int n_ranks, int rank) {
    return (int *) (base + data_bytes + ((size_t) GGML_CUDA_AR1_SLOTS * n_ranks * 2 + rank) * GGML_CUDA_AR1_LINE);
}
static __device__ __forceinline__ unsigned int ar1_now_us() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return (unsigned int) (t / 1000);
}

// Polls a peer flag until *f >= value. After ~0.4 s (400k host reads) it records a diagnostic entry (up to 4
// per rank: code, token, peer flag, n4, entry/spin/timeout times) and keeps waiting: a stuck collective must
// fail loudly rather than reduce garbage. Only in diagnostic mode (give_up, GGML_CUDA_AR1_DEBUG) does it
// return false so the run completes and the records get printed.
static __device__ __forceinline__ bool ar1_spin(const int * f, int value, int * timeout_word, int code, int token, int n4,
        unsigned int t_entry, int give_up) {
    const unsigned int t_spin = ar1_now_us();
    for (long it = 0; ar1_load_flag(f) < value; it++) {
        if (it > 400000L) {
            const int k = ar1_load_flag(timeout_word);
            if (k < 4) {
                int * e = timeout_word + 1 + 7 * k;
                ar1_store_flag(e + 0, code);
                ar1_store_flag(e + 1, token);
                ar1_store_flag(e + 2, ar1_load_flag(f));
                ar1_store_flag(e + 3, n4);
                ar1_store_flag(e + 4, (int) t_entry);
                ar1_store_flag(e + 5, (int) t_spin);
                ar1_store_flag(e + 6, (int) ar1_now_us());
                ar1_store_flag(timeout_word, k + 1);
            }
            if (give_up) {
                return false;
            }
            it = 0;
        }
        __nanosleep(100);
    }
    return true;
}
static __device__ __forceinline__ float4 * ar1_slice(char * base, size_t max_bytes, int n_ranks, int slot, int rank) {
    return (float4 *) (base + ((size_t) slot * n_ranks + rank) * max_bytes);
}

// One block per rank: the token comes from an in-kernel atomic on the rank's device counter, the block
// grid-strides over the slice (no per-thread arrays, so the kernel needs no local memory and records into a
// CUDA graph at any size), only thread 0 polls the peer flags, and the peer slices are read with
// cache-volatile vector loads.
static constexpr int AR1_THREADS = 1024;

static __global__ void __launch_bounds__(AR1_THREADS)
k_ar1_allreduce(float4 * __restrict__ dst, char * base, int * state,
        const size_t data_bytes, const size_t max_bytes,
        const int n_ranks, const int rank, const int n4, const int active, unsigned long long * trace, const int give_up) {
    __shared__ int s_token;
    const unsigned int t_entry = ar1_now_us();
    unsigned long long t_ns[4];
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[0]));
    if (threadIdx.x == 0) {
        s_token = atomicAdd(&state[0], 1) + 1;
    }
    __syncthreads();
    const int token = s_token;
    const int slot  = token % GGML_CUDA_AR1_SLOTS;

    // slot reuse: every peer must have finished reading the launch that used this slot last time
    if (threadIdx.x == 0 && token > GGML_CUDA_AR1_SLOTS) {
        const int need = token - GGML_CUDA_AR1_SLOTS;
        for (int r = 0; r < n_ranks; r++) {
            if (r == rank) continue;
            const int * f = ar1_flag(base, data_bytes, n_ranks, slot, r, 1);
            if (!ar1_spin(f, need, ar1_timeout(base, data_bytes, n_ranks, rank), 1000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
    }
    __syncthreads();

    // phase 1: publish own slice (zeros for an inactive shard)
    float4 * mine = ar1_slice(base, max_bytes, n_ranks, slot, rank);
    for (int i = threadIdx.x; i < n4; i += AR1_THREADS) {
        mine[i] = active ? dst[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 0), token);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[1]));
        // phase 2: wait for every peer's slice
        for (int r = 0; r < n_ranks; r++) {
            if (r == rank) continue;
            const int * f = ar1_flag(base, data_bytes, n_ranks, slot, r, 0);
            if (!ar1_spin(f, token, ar1_timeout(base, data_bytes, n_ranks, rank), 2000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[2]));
        __threadfence();
    }
    __syncthreads();

    // phase 3: sum all slices (own included, re-read from the published copy) into dst
    for (int i = threadIdx.x; i < n4; i += AR1_THREADS) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int r = 0; r < n_ranks; r++) {
            const float4 o = __ldcv(ar1_slice(base, max_bytes, n_ranks, slot, r) + i);
            acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
        }
        dst[i] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 1), token);
        if (trace != nullptr) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[3]));
            unsigned long long * e = trace + ((size_t) rank * AR1_TRACE_N + (token % AR1_TRACE_N)) * AR1_TRACE_W;
            e[0] = token; e[1] = t_ns[0]; e[2] = t_ns[1]; e[3] = t_ns[2]; e[4] = t_ns[3];
        }
    }
}

static ggml_cuda_ar_oneshot * g_ar1_last = nullptr;

// prints and clears the per-rank spin-timeout records (called from the backend synchronize under GGML_CUDA_AR1_DEBUG)
void ggml_cuda_ar_oneshot_report_all() {
    ggml_cuda_ar_oneshot * st = g_ar1_last;
    if (st == nullptr) {
        return;
    }
    if (st->host_trace != nullptr && !st->trace_printed) {
        bool ready = true;
        for (int i = 0; i < st->n_ranks && ready; i++) {
            volatile unsigned long long * e = st->host_trace + ((size_t) i * AR1_TRACE_N + (st->trace_hi % AR1_TRACE_N)) * AR1_TRACE_W;
            ready = e[0] == (unsigned long long) st->trace_hi;
        }
        if (ready) {
            st->trace_printed = 1;
            // host-aligned ms relative to the earliest entry of trace_lo
            long long t0 = 0;
            for (int i = 0; i < st->n_ranks; i++) {
                volatile unsigned long long * e = st->host_trace + ((size_t) i * AR1_TRACE_N + (st->trace_lo % AR1_TRACE_N)) * AR1_TRACE_W;
                const long long t = (long long) e[1] - st->clock_offset[i];
                if (i == 0 || t < t0) t0 = t;
            }
            GGML_LOG_ERROR("ar1 trace t0 = %.3f host ms\n", t0 / 1e6);
            for (int tok = st->trace_lo; tok <= st->trace_hi; tok++) {
                for (int i = 0; i < st->n_ranks; i++) {
                    volatile unsigned long long * e = st->host_trace + ((size_t) i * AR1_TRACE_N + (tok % AR1_TRACE_N)) * AR1_TRACE_W;
                    if (e[0] != (unsigned long long) tok) {
                        GGML_LOG_ERROR("ar1 trace tok %d rank %d: missing (has %llu)\n", tok, i, e[0]);
                        continue;
                    }
                    GGML_LOG_ERROR("ar1 trace tok %d rank %d: entry %.3f arrived %.3f peers %.3f end %.3f ms\n", tok, i,
                        ((long long) e[1] - st->clock_offset[i] - t0) / 1e6, ((long long) e[2] - st->clock_offset[i] - t0) / 1e6,
                        ((long long) e[3] - st->clock_offset[i] - t0) / 1e6, ((long long) e[4] - st->clock_offset[i] - t0) / 1e6);
                }
            }
        }
    }
    const char * host_flags = (const char *) st->host_base + st->data_bytes + (size_t) GGML_CUDA_AR1_SLOTS * st->n_ranks * 2 * GGML_CUDA_AR1_LINE;
    for (int i = 0; i < st->n_ranks; i++) {
        volatile int * w = (volatile int *) (host_flags + (size_t) i * GGML_CUDA_AR1_LINE);
        const int n = w[0];
        for (int k = 0; k < n && k < 4; k++) {
            const volatile int * e = w + 1 + 7 * k;
            GGML_LOG_ERROR("one-shot allreduce: rank %d timed out (phase %d slot %d peer %d) at token %d, peer flag held %d, %d bytes, entry %u us, spin %u us, out %u us\n",
                           i, e[0] / 1000000, (e[0] / 1000) % 1000, e[0] % 1000, e[1], e[2], e[3] * 16,
                           (unsigned) e[4], (unsigned) e[5], (unsigned) e[6]);
        }
        if (n > 0) {
            for (int k = 0; k < 29; k++) {
                w[k] = 0;
            }
        }
    }
}

ggml_cuda_ar_oneshot * ggml_cuda_ar_oneshot_init(const int * devices, size_t n_devices, size_t max_bytes) {
    if (n_devices < 2 || n_devices > (size_t) GGML_CUDA_AR1_MAX_RANKS) {
        return nullptr;
    }
    for (size_t i = 0; i < n_devices; i++) {
        if (ggml_cuda_info().devices[devices[i]].cc < GGML_CUDA_CC_VOLTA) {
            return nullptr; // __nanosleep
        }
    }
    auto * st = new ggml_cuda_ar_oneshot{};
    st->n_ranks   = (int) n_devices;
    st->max_bytes = (max_bytes + 15) / 16 * 16;
    for (size_t i = 0; i < n_devices; i++) {
        st->devices[i] = devices[i];
    }
    st->data_bytes = (size_t) GGML_CUDA_AR1_SLOTS * n_devices * st->max_bytes;
    st->flag_bytes = (size_t) GGML_CUDA_AR1_SLOTS * n_devices * 2 * GGML_CUDA_AR1_LINE + n_devices * GGML_CUDA_AR1_LINE;
    if (cudaHostAlloc(&st->host_base, st->data_bytes + st->flag_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        delete st;
        return nullptr;
    }
    memset(st->host_base, 0, st->data_bytes + st->flag_bytes);
    for (size_t i = 0; i < n_devices; i++) {
        ggml_cuda_set_device(devices[i]);
        void * dptr = nullptr;
        if (cudaHostGetDevicePointer(&dptr, st->host_base, 0) != cudaSuccess ||
                cudaMalloc(&st->dev_state[i], 3 * sizeof(int)) != cudaSuccess ||
                cudaMemset(st->dev_state[i], 0, 3 * sizeof(int)) != cudaSuccess) {
            (void) cudaGetLastError();
            ggml_cuda_ar_oneshot_free(st);
            return nullptr;
        }
        st->dev_base[i] = (char *) dptr;
    }
    if (const char * tr = getenv("GGML_CUDA_AR1_TRACE")) {
        if (sscanf(tr, "%d:%d", &st->trace_lo, &st->trace_hi) == 2 &&
                cudaHostAlloc((void **) &st->host_trace, (size_t) n_devices * AR1_TRACE_N * AR1_TRACE_W * sizeof(unsigned long long),
                              cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
            memset(st->host_trace, 0, (size_t) n_devices * AR1_TRACE_N * AR1_TRACE_W * sizeof(unsigned long long));
            for (size_t i = 0; i < n_devices; i++) {
                ggml_cuda_set_device(devices[i]);
                void * dptr = nullptr;
                CUDA_CHECK(cudaHostGetDevicePointer(&dptr, st->host_trace, 0));
                st->dev_trace[i] = (unsigned long long *) dptr;
                // align this device's globaltimer with the host clock
                unsigned long long * clk = st->host_trace; // scratch: entry 0 of rank 0 is rewritten by the trace later
                for (int rep = 0; rep < 3; rep++) {
                    k_ar1_clock<<<1, 1>>>(st->dev_trace[i]);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    struct timespec ts;
                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    st->clock_offset[i] = (long long) *clk - ((long long) ts.tv_sec * 1000000000LL + ts.tv_nsec);
                }
                *clk = 0;
            }
            GGML_LOG_INFO("%s: tracing one-shot kernels for tokens %d..%d\n", __func__, st->trace_lo, st->trace_hi);
        }
    }
    GGML_LOG_INFO("%s: one-shot host-memory AllReduce for %zu devices, up to %zu bytes per tensor\n", __func__, n_devices, st->max_bytes);
    g_ar1_last = st;
    return st;
}

void ggml_cuda_ar_oneshot_free(ggml_cuda_ar_oneshot * st) {
    if (st == nullptr) {
        return;
    }
    for (int i = 0; i < st->n_ranks; i++) {
        if (st->dev_state[i] != nullptr) {
            ggml_cuda_set_device(st->devices[i]);
            (void) cudaFree(st->dev_state[i]);
        }
    }
    if (st->host_base != nullptr) {
        (void) cudaFreeHost(st->host_base);
    }
    delete st;
}

bool ggml_cuda_ar_oneshot_eligible(const ggml_cuda_ar_oneshot * st, ggml_tensor ** tensors) {
    if (st == nullptr || tensors[0] == nullptr || tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(tensors[0]);
    if (nbytes == 0 || nbytes > st->max_bytes || nbytes % 16 != 0) {
        return false;
    }
    for (int i = 0; i < st->n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nbytes(tensors[i]) != nbytes ||
                !ggml_is_contiguously_allocated(tensors[i]) || ((uintptr_t) tensors[i]->data & 0xF) != 0) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_ar_oneshot_allreduce(ggml_cuda_ar_oneshot * st, ggml_backend_t * backends, ggml_tensor ** tensors) {
    const int n4 = (int) (ggml_nbytes(tensors[0]) / 16);
    static const int give_up = getenv("GGML_CUDA_AR1_DEBUG") != nullptr;
    ggml_cuda_ar_oneshot_report_all();
    for (int i = 0; i < st->n_ranks; i++) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == st->devices[i]);
        ggml_cuda_set_device(st->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();
        const int active = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        k_ar1_allreduce<<<1, AR1_THREADS, 0, stream>>>(
            (float4 *) tensors[i]->data, st->dev_base[i], st->dev_state[i],
            st->data_bytes, st->max_bytes, st->n_ranks, i, n4, active, st->dev_trace[i], give_up);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
