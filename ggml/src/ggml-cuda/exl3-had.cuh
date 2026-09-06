#pragma once

// EXL3 Hadamard helpers: 128-element Sylvester transform, one warp = 128 values (4 per lane).
// Mirrors exllamav3's had_hf_r_128_inner ordering.

#include <cuda_fp16.h>
#include <cstdint>

namespace exl3_had {

constexpr float SCALE = 0.088388347648f; // 1/sqrt(128)

// ---- Hadamard: 128 elements per warp, Sylvester order (had_hf_r_128_inner) ---------------

__device__ __forceinline__ void shuffle_f4x32(float & h0, float & h1, float & h2, float & h3, const int lane_id) {
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        const float ph0 = __shfl_xor_sync(0xffffffff, h0, i);
        const float ph1 = __shfl_xor_sync(0xffffffff, h1, i);
        const float ph2 = __shfl_xor_sync(0xffffffff, h2, i);
        const float ph3 = __shfl_xor_sync(0xffffffff, h3, i);
        const int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ __forceinline__ void had4(float & v0, float & v1, float & v2, float & v3) {
    const float s0 = v0 + v1;
    const float d0 = v0 - v1;
    const float s1 = v2 + v3;
    const float d1 = v2 - v3;
    v0 = s0 + s1;
    v1 = d0 + d1;
    v2 = s0 - s1;
    v3 = d0 - d1;
}


// Full 128-point transform of the four values held by this lane (lane order = had_hf_r_128_inner)
__device__ __forceinline__ void had128(float & v0, float & v1, float & v2, float & v3, int lane) {
    had4(v0, v1, v2, v3);
    shuffle_f4x32(v0, v1, v2, v3, lane);
}

} // namespace exl3_had
