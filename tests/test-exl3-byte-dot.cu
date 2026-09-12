#define EXL3_STANDALONE
#include <cuda_runtime.h>
#include "exl3-gemv-int8.cuh"

#include <cstdio>
#include <vector>

// Compile for SM60 to exercise the scalar fallbacks, and SM61+ for DP4A.
// Enumerate every mul1 codebook product, including unsigned bytes >= 128.
static __global__ void byte_dots(uint32_t * out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t a = i * 0x83dcd12du;
    const uint32_t b = i * 1664525u + 1013904223u;
    out[2*i]     = exl3::byte_sum(a, 0x6400u);
    out[2*i + 1] = uint32_t(exl3_int8::dp4a_us(a, b, int32_t(0x7ffffff0u)));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr size_t count = 65536;
    uint32_t * device = nullptr;
    if (cudaMalloc(&device, 2 * count * sizeof(uint32_t)) != cudaSuccess) return 1;
    byte_dots<<<count/256, 256>>>(device);
    std::vector<uint32_t> actual(2 * count);
    const cudaError_t copied = cudaMemcpy(actual.data(), device, actual.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    const cudaError_t freed = cudaFree(device);
    if (copied != cudaSuccess || freed != cudaSuccess) return 1;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t a = i * 0x83dcd12du;
        const uint32_t b = i * 1664525u + 1013904223u;
        uint32_t expected[] = {0x6400u, 0x7ffffff0u};
        for (int j = 0; j < 4; ++j) {
            const int au = (a >> (8*j)) & 255u;
            const int bu = (b >> (8*j)) & 255u;
            const int bs = bu >= 128 ? bu - 256 : bu;
            expected[0] += au;
            expected[1] += uint32_t(au * bs);
        }
        for (int j = 0; j < 2; ++j) {
            if (actual[2*i+j] != expected[j]) {
                std::fprintf(stderr, "byte dot mismatch state=%u kind=%d actual=%u expected=%u\n",
                             i, j, actual[2*i+j], expected[j]);
                return 1;
            }
        }
    }
    std::printf("PASS: all 65536 codebook products, unsigned byte sums, mixed byte dots and wrapping accumulation\n");
    return 0;
}
