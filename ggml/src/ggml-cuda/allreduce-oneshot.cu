#include "allreduce-oneshot.cuh"

#include <cuda_bf16.h>

#include <time.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <thread>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Pinned host memory layout:
//   one slice block per rank, [n_slots][max_bytes], bound to the NUMA node of that rank's GPU (a rank writes
//   its slice locally; every rank reads all slices, half of them across the socket interconnect)
//   flag block: [n_slots][n_ranks][3] x 128 bytes (published / done / chunk reduced) + one line per rank of
//   spin-timeout diagnostics
static constexpr int    GGML_CUDA_AR1_MAX_RANKS = GGML_CUDA_MAX_DEVICES;
static constexpr int    GGML_CUDA_AR1_SLOTS     = 4;
static constexpr size_t GGML_CUDA_AR1_LINE      = 128;

struct ggml_cuda_ar_oneshot {
    int    n_ranks = 0;
    int    devices[GGML_CUDA_AR1_MAX_RANKS] = {};
    size_t max_bytes = 0;

    void * host_base = nullptr;   // flag block: cudaHostAlloc(Mapped | Portable)
    size_t data_bytes = 0;        // 0: the flags start at host_base (slices live in their own blocks)
    size_t flag_bytes = 0;

    // per-rank slice blocks and their device-visible pointers on every device: dev_slices[device][rank]
    void * host_slices[GGML_CUDA_AR1_MAX_RANKS] = {};
    char * dev_stage[GGML_CUDA_AR1_MAX_RANKS] = {};      // per-rank device staging for the BF16 large-message payload
    bool   slice_mmapped[GGML_CUDA_AR1_MAX_RANKS] = {};   // mmap+mbind+cudaHostRegister (else cudaHostAlloc)
    size_t slice_bytes = 0;
    char * dev_slices[GGML_CUDA_AR1_MAX_RANKS][GGML_CUDA_AR1_MAX_RANKS] = {};

    // device-visible pointers to the flag block, per rank (mapped memory may map differently per device)
    char * dev_base[GGML_CUDA_AR1_MAX_RANKS] = {};
    // device-side per-rank state: [0] launch counter (token x blocks), [1..3] large-message phase counters,
    // [4] current large token, [5] previous large token, [6 + slot] last token that used each small slot
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

// flag address of (slot, rank, which) inside the host block: 0 = slice published, 1 = done reading, 2 = own
// chunk reduced (scatter mode)
static constexpr int AR1_FLAGS_PER_RANK = 3;
static __device__ __forceinline__ int * ar1_flag(char * base, size_t data_bytes, int n_ranks, int slot, int rank, int which) {
    return (int *) (base + data_bytes + ((size_t) (slot * n_ranks + rank) * AR1_FLAGS_PER_RANK + which) * GGML_CUDA_AR1_LINE);
}
// diagnostics: rank's timeout word lives after all slot flags
static __device__ __forceinline__ int * ar1_timeout(char * base, size_t data_bytes, int n_ranks, int rank) {
    return (int *) (base + data_bytes + ((size_t) (GGML_CUDA_AR1_SLOTS + 1) * n_ranks * AR1_FLAGS_PER_RANK + rank) * GGML_CUDA_AR1_LINE);
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
// the slice blocks as this device sees them
struct ar1_bases {
    char * p[GGML_CUDA_AR1_MAX_RANKS];
};
static __device__ __forceinline__ float4 * ar1_slice(const ar1_bases & b, size_t max_bytes, int slot, int rank) {
    return (float4 *) (b.p[rank] + (size_t) slot * max_bytes);
}

// One block per rank: the token comes from an in-kernel atomic on the rank's device counter, the block
// grid-strides over the slice (no per-thread arrays, so the kernel needs no local memory and records into a
// CUDA graph at any size), only thread 0 polls the peer flags, and the peer slices are read with
// cache-volatile vector loads.
static constexpr int AR1_THREADS = 1024;
// blocks of the multi-block kernel; both kernels advance the launch counter by this much per launch so their
// tokens (and slots) stay consistent when small and large reduces alternate
static constexpr int AR1_MB_BLOCKS = 16;

static __global__ void __launch_bounds__(AR1_THREADS)
k_ar1_allreduce(float4 * __restrict__ dst, char * base, const ar1_bases bases, int * state,
        const size_t data_bytes, const size_t max_bytes,
        const int n_ranks, const int rank, const int n4, const int active, unsigned long long * trace, const int give_up,
        const int scatter) {
    __shared__ int s_token;
    const unsigned int t_entry = ar1_now_us();
    unsigned long long t_ns[4];
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[0]));
    if (threadIdx.x == 0) {
        s_token = atomicAdd(&state[0], AR1_MB_BLOCKS) / AR1_MB_BLOCKS + 1;
    }
    __syncthreads();
    const int token = s_token;
    const int slot  = token % GGML_CUDA_AR1_SLOTS;

    // slot reuse: every peer must have finished reading the launch that used this slot last time. That token
    // is tracked per slot (state[6 + slot]); it is not token - n_slots when large messages, which use their
    // own slot, were interleaved. Every rank runs the same sequence, so the peers' done flags reach it.
    if (threadIdx.x == 0) {
        const int need = state[6 + slot];
        state[6 + slot] = token;
        for (int r = 0; need > 0 && r < n_ranks; r++) {
            if (r == rank) continue;
            const int * f = ar1_flag(base, data_bytes, n_ranks, slot, r, 1);
            if (!ar1_spin(f, need, ar1_timeout(base, data_bytes, n_ranks, rank), 1000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
    }
    __syncthreads();

    // phase 1: publish own slice (zeros for an inactive shard)
    float4 * mine = ar1_slice(bases, max_bytes, slot, rank);
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

    if (scatter) {
        // reduce-scatter / all-gather through the same slices: rank r sums chunk r of every slice and writes
        // the result in place into its own slice (position r is read by nobody else), then everyone gathers
        // the reduced chunks. Host reads per rank: 2 x message instead of n_ranks x message.
        const int q  = (n4 + n_ranks - 1) / n_ranks;
        const int c0 = rank * q;
        const int c1 = min(c0 + q, n4);
        for (int i = c0 + threadIdx.x; i < c1; i += AR1_THREADS) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int r = 0; r < n_ranks; r++) {
                const float4 o = __ldcv(ar1_slice(bases, max_bytes, slot, r) + i);
                acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
            }
            mine[i] = acc;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 2), token);
            for (int r = 0; r < n_ranks; r++) {
                if (r == rank) continue;
                const int * f = ar1_flag(base, data_bytes, n_ranks, slot, r, 2);
                if (!ar1_spin(f, token, ar1_timeout(base, data_bytes, n_ranks, rank), 3000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                    break;
                }
            }
            __threadfence();
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n4; i += AR1_THREADS) {
            dst[i] = __ldcv(ar1_slice(bases, max_bytes, slot, i / q) + i);
        }
    } else {
        // phase 3: sum all slices (own included, re-read from the published copy) into dst
        for (int i = threadIdx.x; i < n4; i += AR1_THREADS) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int r = 0; r < n_ranks; r++) {
                const float4 o = __ldcv(ar1_slice(bases, max_bytes, slot, r) + i);
                acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
            }
            dst[i] = acc;
        }
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

// Multi-megabyte messages (prompt processing). GPU-initiated stores to mapped host memory run at ~3 GB/s
// while kernel loads reach ~18 GB/s and a copy-engine memcpy runs at PCIe rate, so the slice is published
// with a D2H memcpy issued by the host between two kernels (stream-ordered, capturable):
//   k_ar1_mb_begin  (1 block)  token; wait until every peer is done with the previous large message
//   cudaMemcpyAsync D2H        dst -> own slice in the dedicated large-message slot (zeros if inactive)
//   k_ar1_mb_reduce (16 blocks) publish, wait peers, reduce-scatter chunk `rank` in place, publish, wait,
//                              gather; the last block of a phase (per-launch device counter) sets the flag
// Large messages use their own slot (index GGML_CUDA_AR1_SLOTS) so they never share memory with the small
// pipelined ones, and the host cannot know a token under graph replay, so there is exactly one such slot:
// consecutive large reduces are serialized by the begin kernel's wait. All blocks must be co-resident.
static constexpr int AR1_MB_SLOT = GGML_CUDA_AR1_SLOTS;

static __global__ void k_ar1_mb_begin(char * base, int * state, const size_t data_bytes,
        const int n_ranks, const int rank, const int n4, const int give_up) {
    if (threadIdx.x != 0) {
        return;
    }
    const unsigned int t_entry = ar1_now_us();
    const int token = atomicAdd(&state[0], AR1_MB_BLOCKS) / AR1_MB_BLOCKS + 1;
    state[4] = token;
    const int prev = state[5]; // this rank's previous large-message token; every rank runs the same sequence
    state[5] = token;
    if (prev > 0) {
        for (int r = 0; r < n_ranks; r++) {
            if (r == rank) continue;
            if (!ar1_spin(ar1_flag(base, data_bytes, n_ranks, AR1_MB_SLOT, r, 1), prev, ar1_timeout(base, data_bytes, n_ranks, rank),
                    1000000 + AR1_MB_SLOT * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
    }
}

// BF16 payload variant (GGML_CUDA_AR1_BF16=1): the shard is converted to BF16 on the device, published as
// half the bytes, reduce-scattered with F32 accumulation into a BF16 chunk, and gathered back to F32.
// Halves the host reads and the memcpy of every multi-MB reduce at the cost of BF16 rounding of the
// summands (what the NCCL path did for large tensors).
static __device__ __forceinline__ float4 ar1_bf16x4_to_f32(uint2 v) {
    return make_float4(__bfloat162float(__ushort_as_bfloat16((unsigned short) (v.x & 0xFFFF))),
                       __bfloat162float(__ushort_as_bfloat16((unsigned short) (v.x >> 16))),
                       __bfloat162float(__ushort_as_bfloat16((unsigned short) (v.y & 0xFFFF))),
                       __bfloat162float(__ushort_as_bfloat16((unsigned short) (v.y >> 16))));
}
static __device__ __forceinline__ uint2 ar1_f32_to_bf16x4(float4 v) {
    uint2 r;
    r.x = (unsigned int) __bfloat16_as_ushort(__float2bfloat16_rn(v.x)) | ((unsigned int) __bfloat16_as_ushort(__float2bfloat16_rn(v.y)) << 16);
    r.y = (unsigned int) __bfloat16_as_ushort(__float2bfloat16_rn(v.z)) | ((unsigned int) __bfloat16_as_ushort(__float2bfloat16_rn(v.w)) << 16);
    return r;
}
static __global__ void k_ar1_f32_to_bf16(const float4 * __restrict__ src, uint2 * __restrict__ dst, const int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        dst[i] = ar1_f32_to_bf16x4(src[i]);
    }
}

static __global__ void __launch_bounds__(AR1_THREADS)
k_ar1_mb_reduce_bf16(float4 * __restrict__ dst, char * base, const ar1_bases bases, int * state,
        const size_t data_bytes, const size_t max_bytes,
        const int n_ranks, const int rank, const int n4, const int give_up) {
    const unsigned int t_entry = ar1_now_us();
    const int token = state[4];
    const int slot  = AR1_MB_SLOT;
    const int nb    = gridDim.x;
    const int b     = blockIdx.x;

    if (threadIdx.x == 0) {
        if (b == 0) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 0), token);
        }
        for (int r = 0; r < n_ranks; r++) {
            if (!ar1_spin(ar1_flag(base, data_bytes, n_ranks, slot, r, 0), token, ar1_timeout(base, data_bytes, n_ranks, rank),
                    2000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
        __threadfence();
    }
    __syncthreads();

    uint2 * mine = (uint2 *) ar1_slice(bases, max_bytes, slot, rank);
    const int q  = (n4 + n_ranks - 1) / n_ranks;
    const int c0 = rank * q;
    const int c1 = min(c0 + q, n4);
    for (int i = c0 + b * AR1_THREADS + threadIdx.x; i < c1; i += nb * AR1_THREADS) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int r = 0; r < n_ranks; r++) {
            const float4 o = ar1_bf16x4_to_f32(__ldcv((const uint2 *) ar1_slice(bases, max_bytes, slot, r) + i));
            acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
        }
        mine[i] = ar1_f32_to_bf16x4(acc);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(&state[2], 1) % nb == nb - 1) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 2), token);
        }
        for (int r = 0; r < n_ranks; r++) {
            if (!ar1_spin(ar1_flag(base, data_bytes, n_ranks, slot, r, 2), token, ar1_timeout(base, data_bytes, n_ranks, rank),
                    3000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
        __threadfence();
    }
    __syncthreads();

    for (int i = b * AR1_THREADS + threadIdx.x; i < n4; i += nb * AR1_THREADS) {
        dst[i] = ar1_bf16x4_to_f32(__ldcv((const uint2 *) ar1_slice(bases, max_bytes, slot, i / q) + i));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(&state[3], 1) % nb == nb - 1) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 1), token);
        }
    }
}

static __global__ void __launch_bounds__(AR1_THREADS)
k_ar1_mb_reduce(float4 * __restrict__ dst, char * base, const ar1_bases bases, int * state,
        const size_t data_bytes, const size_t max_bytes,
        const int n_ranks, const int rank, const int n4, const int give_up, unsigned long long * trace) {
    const unsigned int t_entry = ar1_now_us();
    unsigned long long t_ns[4];
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[0]));
    t_ns[1] = t_ns[2] = t_ns[0];
    const int token = state[4];
    const int slot  = AR1_MB_SLOT;
    const int nb    = gridDim.x;
    const int b     = blockIdx.x;

    // the memcpy that published this rank's slice completed before this kernel started (stream order)
    if (threadIdx.x == 0) {
        if (b == 0) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 0), token);
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[1]));
        for (int r = 0; r < n_ranks; r++) {
            if (!ar1_spin(ar1_flag(base, data_bytes, n_ranks, slot, r, 0), token, ar1_timeout(base, data_bytes, n_ranks, rank),
                    2000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[2]));
        __threadfence();
    }
    __syncthreads();

    // reduce this block's part of chunk `rank` in place (position `rank` of the own slice is read by nobody else)
    float4 * mine = ar1_slice(bases, max_bytes, slot, rank);
    const int q  = (n4 + n_ranks - 1) / n_ranks;
    const int c0 = rank * q;
    const int c1 = min(c0 + q, n4);
    for (int i = c0 + b * AR1_THREADS + threadIdx.x; i < c1; i += nb * AR1_THREADS) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int r = 0; r < n_ranks; r++) {
            const float4 o = __ldcv(ar1_slice(bases, max_bytes, slot, r) + i);
            acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
        }
        mine[i] = acc;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // per-launch counter: every launch adds exactly nb, so the block that brings the count to a multiple
        // of nb is this launch's last
        if (atomicAdd(&state[2], 1) % nb == nb - 1) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 2), token);
        }
        for (int r = 0; r < n_ranks; r++) {
            if (!ar1_spin(ar1_flag(base, data_bytes, n_ranks, slot, r, 2), token, ar1_timeout(base, data_bytes, n_ranks, rank),
                    3000000 + slot * 1000 + r, token, n4, t_entry, give_up)) {
                break;
            }
        }
        __threadfence();
    }
    __syncthreads();

    // gather the reduced chunks, this block's range
    for (int i = b * AR1_THREADS + threadIdx.x; i < n4; i += nb * AR1_THREADS) {
        dst[i] = __ldcv(ar1_slice(bases, max_bytes, slot, i / q) + i);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(&state[3], 1) % nb == nb - 1) {
            ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 1), token);
        }
        if (trace != nullptr && b == 0) {
            // block 0's view: entry, slice published, peers seen, end (the memcpy phase precedes the kernel)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_ns[3]));
            unsigned long long * e = trace + ((size_t) rank * AR1_TRACE_N + (token % AR1_TRACE_N)) * AR1_TRACE_W;
            e[0] = token; e[1] = t_ns[0]; e[2] = t_ns[1]; e[3] = t_ns[2]; e[4] = t_ns[3];
        }
    }
}

static ggml_cuda_ar_oneshot * g_ar1_last = nullptr;

// First-touch placement: fault the pages in from a thread pinned to the CPUs of `node` (sysfs cpulist), so the
// default local-allocation policy puts them on that node. Returns false if the node's CPUs cannot be used.
static bool ar1_first_touch_on_node(void * p, size_t bytes, int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    char list[1024] = {};
    const bool got = fgets(list, sizeof(list), f) != nullptr;
    fclose(f);
    if (!got) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    int n_cpus = 0;
    for (char * tok = strtok(list, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
        int a = 0, b = 0;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) {
            for (int c = a; c <= b && c < CPU_SETSIZE; c++) { CPU_SET(c, &set); n_cpus++; }
        } else if (sscanf(tok, "%d", &a) == 1 && a < CPU_SETSIZE) {
            CPU_SET(a, &set); n_cpus++;
        }
    }
    if (n_cpus == 0) {
        return false;
    }
    bool ok = false;
    std::thread([&]() {
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            return;
        }
        memset(p, 0, bytes);
        ok = true;
    }).join();
    return ok;
}

// NUMA node of a CUDA device via sysfs (-1 when unknown or NUMA-less)
static int ar1_gpu_numa_node(int device) {
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), ggml_cuda_info().devices[device].physical_device) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    for (char * c = bus_id; *c; c++) {
        *c = (char) tolower((unsigned char) *c);
    }
    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bus_id);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) {
        node = -1;
    }
    fclose(f);
    return node;
}

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
    const char * host_flags = (const char *) st->host_base + st->data_bytes + (size_t) (GGML_CUDA_AR1_SLOTS + 1) * st->n_ranks * AR1_FLAGS_PER_RANK * GGML_CUDA_AR1_LINE;
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
    st->data_bytes = 0;
    st->flag_bytes = (size_t) (GGML_CUDA_AR1_SLOTS + 1) * n_devices * AR1_FLAGS_PER_RANK * GGML_CUDA_AR1_LINE + n_devices * GGML_CUDA_AR1_LINE;
    if (cudaHostAlloc(&st->host_base, st->flag_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        delete st;
        return nullptr;
    }
    memset(st->host_base, 0, st->flag_bytes);
    // one slice block per rank on the NUMA node of its GPU (GGML_CUDA_AR1_NUMA=0 keeps plain pinned allocations)
    static const bool numa_slices = getenv("GGML_CUDA_AR1_NUMA") == nullptr || atoi(getenv("GGML_CUDA_AR1_NUMA")) != 0;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    st->slice_bytes = ((size_t) (GGML_CUDA_AR1_SLOTS + 1) * st->max_bytes + page - 1) / page * page;
    for (size_t r = 0; r < n_devices; r++) {
        const int node = numa_slices ? ar1_gpu_numa_node(devices[r]) : -1;
        void * p = nullptr;
        if (node >= 0) {
            p = mmap(nullptr, st->slice_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED) {
                unsigned long mask[16] = {};
                mask[node / 64] = 1UL << (node % 64);
                long ok = syscall(SYS_mbind, p, st->slice_bytes, /*MPOL_BIND*/ 2, mask, sizeof(mask) * 8, 0);
                const int mbind_errno = errno;
                if (ok == 0) {
                    memset(p, 0, st->slice_bytes); // first touch on the bound node
                } else {
                    // mbind can be denied (containers): fall back to first-touch from a thread pinned to the node's CPUs
                    ok = ar1_first_touch_on_node(p, st->slice_bytes, node) ? 0 : -1;
                }
                cudaError_t reg = cudaSuccess;
                if (ok == 0) {
                    reg = cudaHostRegister(p, st->slice_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
                }
                if (ok != 0 || reg != cudaSuccess) {
                    GGML_LOG_WARN("%s: rank %zu: NUMA placement unavailable (mbind: %s, first-touch: failed, cudaHostRegister: %s); using plain pinned memory\n",
                                  __func__, r, strerror(mbind_errno), reg != cudaSuccess ? cudaGetErrorString(reg) : "ok");
                    (void) cudaGetLastError();
                    munmap(p, st->slice_bytes);
                    p = nullptr;
                } else {
                    st->slice_mmapped[r] = true;
                }
            } else {
                p = nullptr;
            }
        }
        if (p == nullptr) {
            if (cudaHostAlloc(&p, st->slice_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                ggml_cuda_ar_oneshot_free(st);
                return nullptr;
            }
            memset(p, 0, st->slice_bytes);
        }
        st->host_slices[r] = p;
        if (numa_slices) {
            GGML_LOG_INFO("%s: rank %zu slice block on NUMA node %d (%s)\n", __func__, r, node, st->slice_mmapped[r] ? "bound" : "unbound");
        }
    }
    for (size_t i = 0; i < n_devices; i++) {
        ggml_cuda_set_device(devices[i]);
        void * dptr = nullptr;
        if (cudaHostGetDevicePointer(&dptr, st->host_base, 0) != cudaSuccess ||
                cudaMalloc(&st->dev_state[i], (6 + GGML_CUDA_AR1_SLOTS) * sizeof(int)) != cudaSuccess ||
                cudaMemset(st->dev_state[i], 0, (6 + GGML_CUDA_AR1_SLOTS) * sizeof(int)) != cudaSuccess) {
            (void) cudaGetLastError();
            ggml_cuda_ar_oneshot_free(st);
            return nullptr;
        }
        st->dev_base[i] = (char *) dptr;
        // BF16 staging for the large-message payload (optional: the F32 path is used without it)
        if (cudaMalloc((void **) &st->dev_stage[i], st->max_bytes / 2) != cudaSuccess) {
            (void) cudaGetLastError();
            st->dev_stage[i] = nullptr;
        }
        for (size_t r = 0; r < n_devices; r++) {
            void * sptr = nullptr;
            if (cudaHostGetDevicePointer(&sptr, st->host_slices[r], 0) != cudaSuccess) {
                (void) cudaGetLastError();
                ggml_cuda_ar_oneshot_free(st);
                return nullptr;
            }
            st->dev_slices[i][r] = (char *) sptr;
        }
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
        if (st->dev_stage[i] != nullptr) {
            ggml_cuda_set_device(st->devices[i]);
            (void) cudaFree(st->dev_stage[i]);
        }
    }
    for (int r = 0; r < st->n_ranks; r++) {
        if (st->host_slices[r] == nullptr) {
            continue;
        }
        if (st->slice_mmapped[r]) {
            (void) cudaHostUnregister(st->host_slices[r]);
            munmap(st->host_slices[r], st->slice_bytes);
        } else {
            (void) cudaFreeHost(st->host_slices[r]);
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
    // reduce-scatter mode from this message size on (GGML_CUDA_AR1_SCATTER=<bytes>, 0 = never)
    static const size_t scatter_from = getenv("GGML_CUDA_AR1_SCATTER") != nullptr ? (size_t) atoll(getenv("GGML_CUDA_AR1_SCATTER")) : (size_t) 32 * 1024;
    const int scatter = scatter_from != 0 && ggml_nbytes(tensors[0]) >= scatter_from && n4 >= st->n_ranks;
    // the single-block kernel and its token counter cover messages up to this size; beyond it the multi-block
    // kernel takes over (GGML_CUDA_AR1_MB_FROM=<bytes>; both kernels share the launch counter, which the
    // multi-block one advances by its block count, so the two must not interleave: the host serializes them)
    static const size_t mb_from = getenv("GGML_CUDA_AR1_MB_FROM") != nullptr ? (size_t) atoll(getenv("GGML_CUDA_AR1_MB_FROM")) : (size_t) 256 * 1024;
    const int multi_block = ggml_nbytes(tensors[0]) > mb_from && n4 >= st->n_ranks * AR1_MB_BLOCKS;
    ggml_cuda_ar_oneshot_report_all();
    for (int i = 0; i < st->n_ranks; i++) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == st->devices[i]);
        ggml_cuda_set_device(st->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();
        const int active = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        ar1_bases bases = {};
        for (int r = 0; r < st->n_ranks; r++) {
            bases.p[r] = st->dev_slices[i][r];
        }
        if (multi_block) {
            // BF16 payload by default (GGML_CUDA_AR1_BF16=0 keeps F32): halves the host traffic of every
            // multi-MB reduce; the NCCL path rounded these tensors to BF16 as well
            static const bool bf16 = getenv("GGML_CUDA_AR1_BF16") == nullptr || atoi(getenv("GGML_CUDA_AR1_BF16")) != 0;
            k_ar1_mb_begin<<<1, 32, 0, stream>>>(st->dev_base[i], st->dev_state[i], st->data_bytes, st->n_ranks, i, n4, give_up);
            // host address of the own slice (UVA resolves the direction); the memset uses the device alias
            char * slice_host = (char *) st->host_slices[i] + (size_t) AR1_MB_SLOT * st->max_bytes;
            char * slice_dev  = st->dev_slices[i][i] + (size_t) AR1_MB_SLOT * st->max_bytes;
            if (bf16 && st->dev_stage[i] != nullptr) {
                if (active) {
                    k_ar1_f32_to_bf16<<<(n4 + 255) / 256, 256, 0, stream>>>((const float4 *) tensors[i]->data, (uint2 *) st->dev_stage[i], n4);
                    CUDA_CHECK(cudaMemcpyAsync(slice_host, st->dev_stage[i], (size_t) n4 * 8, cudaMemcpyDefault, stream));
                } else {
                    CUDA_CHECK(cudaMemsetAsync(slice_dev, 0, (size_t) n4 * 8, stream));
                }
                k_ar1_mb_reduce_bf16<<<AR1_MB_BLOCKS, AR1_THREADS, 0, stream>>>(
                    (float4 *) tensors[i]->data, st->dev_base[i], bases, st->dev_state[i],
                    st->data_bytes, st->max_bytes, st->n_ranks, i, n4, give_up);
            } else {
                if (active) {
                    CUDA_CHECK(cudaMemcpyAsync(slice_host, tensors[i]->data, (size_t) n4 * 16, cudaMemcpyDefault, stream));
                } else {
                    CUDA_CHECK(cudaMemsetAsync(slice_dev, 0, (size_t) n4 * 16, stream));
                }
                k_ar1_mb_reduce<<<AR1_MB_BLOCKS, AR1_THREADS, 0, stream>>>(
                    (float4 *) tensors[i]->data, st->dev_base[i], bases, st->dev_state[i],
                    st->data_bytes, st->max_bytes, st->n_ranks, i, n4, give_up, st->dev_trace[i]);
            }
        } else {
            k_ar1_allreduce<<<1, AR1_THREADS, 0, stream>>>(
                (float4 *) tensors[i]->data, st->dev_base[i], bases, st->dev_state[i],
                st->data_bytes, st->max_bytes, st->n_ranks, i, n4, active, st->dev_trace[i], give_up, scatter);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
