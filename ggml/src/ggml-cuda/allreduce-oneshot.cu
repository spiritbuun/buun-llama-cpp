#include "allreduce-oneshot.cuh"

#include <cstdlib>
#include <cstring>

// Layout of the shared pinned block:
//   data  [n_slots][n_ranks][max_bytes]      the slices
//   flags [n_slots][n_ranks][2] x 64 bytes   arrival token, done token (one cache line each)
static constexpr int    GGML_CUDA_AR1_MAX_RANKS = GGML_CUDA_MAX_DEVICES;
static constexpr int    GGML_CUDA_AR1_SLOTS     = 4;
static constexpr size_t GGML_CUDA_AR1_LINE      = 64;

struct ggml_cuda_ar_oneshot {
    int    n_ranks = 0;
    int    devices[GGML_CUDA_AR1_MAX_RANKS] = {};
    size_t max_bytes = 0;

    void * host_base = nullptr;   // cudaHostAlloc(Mapped | Portable)
    size_t data_bytes = 0;        // n_slots * n_ranks * max_bytes
    size_t flag_bytes = 0;

    // device-visible pointers to the host block, per rank (mapped memory may map differently per device)
    char * dev_base[GGML_CUDA_AR1_MAX_RANKS] = {};
    // device-side per-rank state: [0] launch counter (token), [1] blocks finished writing, [2] blocks finished reading
    int  * dev_state[GGML_CUDA_AR1_MAX_RANKS] = {};
};

static __device__ __forceinline__ int ar1_load_flag(const int * p) {
    return *((volatile const int *) p);
}
static __device__ __forceinline__ void ar1_store_flag(int * p, int v) {
    *((volatile int *) p) = v;
}

// flag address of (slot, rank, which) inside the host block
static __device__ __forceinline__ int * ar1_flag(char * base, size_t data_bytes, int n_ranks, int slot, int rank, int which) {
    return (int *) (base + data_bytes + ((size_t) (slot * n_ranks + rank) * 2 + which) * GGML_CUDA_AR1_LINE);
}
static __device__ __forceinline__ float4 * ar1_slice(char * base, size_t max_bytes, int n_ranks, int slot, int rank) {
    return (float4 *) (base + ((size_t) slot * n_ranks + rank) * max_bytes);
}

// One block per rank: the token comes from an in-kernel atomic on the rank's device counter, every block
// thread owns up to AR1_PER_THREAD float4 elements, only thread 0 polls the peer flags, and the peer slices
// are read with cache-volatile vector loads issued back to back to overlap the PCIe latency.
static constexpr int AR1_THREADS    = 1024;
static constexpr int AR1_PER_THREAD = 4;

static __global__ void __launch_bounds__(AR1_THREADS)
k_ar1_allreduce(float4 * __restrict__ dst, char * base, int * state,
        const size_t data_bytes, const size_t max_bytes,
        const int n_ranks, const int rank, const int n4, const int active) {
    __shared__ int s_token;
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
            while (ar1_load_flag(f) < need) { __nanosleep(100); }
        }
    }
    __syncthreads();

    // phase 1: publish own slice (zeros for an inactive shard)
    float4 * mine = ar1_slice(base, max_bytes, n_ranks, slot, rank);
    float4 v[AR1_PER_THREAD];
#pragma unroll
    for (int k = 0; k < AR1_PER_THREAD; k++) {
        const int i = threadIdx.x + k * AR1_THREADS;
        v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < n4) {
            if (active) {
                v[k] = dst[i];
            }
            mine[i] = v[k];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 0), token);
        // phase 2: wait for every peer's slice
        for (int r = 0; r < n_ranks; r++) {
            if (r == rank) continue;
            const int * f = ar1_flag(base, data_bytes, n_ranks, slot, r, 0);
            while (ar1_load_flag(f) < token) { __nanosleep(100); }
        }
        __threadfence();
    }
    __syncthreads();

    // phase 3: sum the peer slices (loads for all peers issued before the adds)
#pragma unroll
    for (int k = 0; k < AR1_PER_THREAD; k++) {
        const int i = threadIdx.x + k * AR1_THREADS;
        if (i >= n4) {
            break;
        }
        float4 acc = v[k];
        float4 o[GGML_CUDA_AR1_MAX_RANKS];
#pragma unroll
        for (int r = 0; r < GGML_CUDA_AR1_MAX_RANKS; r++) {
            if (r < n_ranks && r != rank) {
                o[r] = __ldcv(ar1_slice(base, max_bytes, n_ranks, slot, r) + i);
            }
        }
#pragma unroll
        for (int r = 0; r < GGML_CUDA_AR1_MAX_RANKS; r++) {
            if (r < n_ranks && r != rank) {
                acc.x += o[r].x; acc.y += o[r].y; acc.z += o[r].z; acc.w += o[r].w;
            }
        }
        dst[i] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        ar1_store_flag(ar1_flag(base, data_bytes, n_ranks, slot, rank, 1), token);
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
    st->flag_bytes = (size_t) GGML_CUDA_AR1_SLOTS * n_devices * 2 * GGML_CUDA_AR1_LINE;
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
    GGML_LOG_INFO("%s: one-shot host-memory AllReduce for %zu devices, up to %zu bytes per tensor\n", __func__, n_devices, st->max_bytes);
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
    if (nbytes == 0 || nbytes > st->max_bytes || nbytes > (size_t) AR1_THREADS * AR1_PER_THREAD * 16 || nbytes % 16 != 0) {
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
    GGML_ASSERT(n4 <= AR1_THREADS * AR1_PER_THREAD);
    for (int i = 0; i < st->n_ranks; i++) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == st->devices[i]);
        ggml_cuda_set_device(st->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();
        const int active = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        k_ar1_allreduce<<<1, AR1_THREADS, 0, stream>>>(
            (float4 *) tensors[i]->data, st->dev_base[i], st->dev_state[i],
            st->data_bytes, st->max_bytes, st->n_ranks, i, n4, active);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
