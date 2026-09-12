#include "ggml.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

void ggml_cuda_exl3_cache_mmv(const void *, ggml_type, const float *, const int32_t *,
    const int32_t *, float *, int, int, size_t, int, int, cudaStream_t);

template <typename T> struct device_buffer {
    T * ptr = nullptr;
    explicit device_buffer(size_t count) { cudaMalloc((void **) &ptr, count * sizeof(T)); }
    ~device_buffer() { cudaFree(ptr); }
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

static bool cuda_ok(cudaError_t error) {
    if (error == cudaSuccess) return true;
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    return false;
}

int main() {
    constexpr int k = 2560, n = 640;
    constexpr int high = 22000;
    constexpr size_t expert_bytes = size_t(k) * n / 4;
    constexpr size_t pool_bytes = size_t(high + 2) * expert_bytes;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < pool_bytes + (size_t(1) << 30)) {
        printf("SKIP: large EXL3 pool test needs at least 9.4 GiB free GPU memory\n");
        return 77;
    }
    if (ggml_moe_cache_max_pool_slots(GGML_TYPE_EXL3_2, expert_bytes) <= high + 1) return 1;
    device_buffer<unsigned char> pool(pool_bytes);
    device_buffer<float> act(2 * k), out(4 * n);
    device_buffer<int32_t> ids(4), act_ids(4);
    if (!pool.ptr || !act.ptr || !out.ptr || !ids.ptr || !act_ids.ptr) return 1;

    // Identical packed experts at low and >8 GiB byte offsets. Only these
    // slots are initialized and read; the rest of the slab is unused.
    std::vector<unsigned char> packed(2 * expert_bytes);
    uint32_t random = 1234;
    for (auto & byte : packed) {
        random = random * 1664525u + 1013904223u;
        byte = random >> 24;
    }
    std::vector<float> input(2 * k), output(4 * n);
    for (int i = 0; i < 2 * k; ++i) input[i] = float(i % 17 - 8) / 16;
    const int32_t slots[] = {0, high, 1, high + 1};
    const int32_t rows[] = {0, 0, 1, 1};
    if (!cuda_ok(cudaMemcpy(pool.ptr, packed.data(), packed.size(), cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(pool.ptr + size_t(high) * expert_bytes, packed.data(), packed.size(), cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(act.ptr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(ids.ptr, slots, sizeof(slots), cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(act_ids.ptr, rows, sizeof(rows), cudaMemcpyHostToDevice))) return 1;
    ggml_cuda_exl3_cache_mmv(pool.ptr, GGML_TYPE_EXL3_2, act.ptr, ids.ptr, act_ids.ptr,
        out.ptr, k, n, expert_bytes, 4, 2, nullptr);
    if (!cuda_ok(cudaGetLastError()) ||
        !cuda_ok(cudaMemcpy(output.data(), out.ptr, output.size() * sizeof(float), cudaMemcpyDeviceToHost))) return 1;
    for (float value : output) if (!std::isfinite(value)) return 1;
    const bool equal = std::memcmp(output.data(), output.data() + n, n * sizeof(float)) == 0 &&
        std::memcmp(output.data() + 2 * n, output.data() + 3 * n, n * sizeof(float)) == 0;
    const bool distinct = std::memcmp(output.data(), output.data() + 2 * n, n * sizeof(float)) != 0;
    printf("EXL3 large-pool offsets: %s (high offset %zu bytes)\n",
        equal && distinct ? "OK" : "FAIL", size_t(high) * expert_bytes);
    return equal && distinct ? 0 : 1;
}
