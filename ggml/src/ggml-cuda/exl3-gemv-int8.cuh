#pragma once

// EXL3 int8-activation decode gemv (4 bpw, m <= 4), after exllamav3's exl3_gemv_int8 (MIT).
//
// The mul1 codebook value is affine in the byte sum of (window * 0x83DCD12D):
//   v = k_inv * (1024 + bytesum) + bias
// so with activations quantized to int8, dp4a(window * M, splat(a), acc) accumulates a * bytesum
// exactly in int32 and
//   y[n] = q * (k_inv * acc[n] + (1024 * k_inv + bias) * sum(a))
// In residual mode the int8 rounding error (|r| <= q/2) is quantized with scale q2 = q / 254 and
// accumulated by a second dp4a sharing the decoded windows (~15-16 bit activation precision).
//
// Quantization is per k-slice: every block quantizes its own k range of the Hadamard-transformed
// activations (F16 xh from the fp16 path's input kernel) while staging the byte splats, so the
// scale tracks the local magnitude (this is what keeps plain int8 at fp16 parity).  Each block
// writes its float partial y_inner for its 256 columns; the last k-split block per column group
// sums the partials in slice order (bitwise deterministic), applies the output Hadamard and svh.
//
// gemv_int8_kernel grid (n/256, ksplit), 256 threads: warp = 2 adjacent n tiles, lane = uint2 of
// its tile word pair, two-row register prefetch, splats read from smem as uint4.

#include "exl3-dq.cuh"
#include "exl3-had.cuh"

namespace exl3_int8 {

constexpr int THREADS = 256;
constexpr int COLS    = 256;   // columns per block: 8 warps x 2 tiles
constexpr int MAX_M   = 4;

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// 4 bpw: 8 windows for run t0..t0+7 from words (a = previous, b = this)
__device__ __forceinline__ void extract8_4bits(uint32_t a, uint32_t b, uint32_t & w0, uint32_t & w1,
        uint32_t & w2, uint32_t & w3, uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    uint32_t s;
    EXL3_FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    EXL3_BFE16_IMM(w6, b, 4);
    EXL3_BFE16_IMM(w5, b, 8);
    EXL3_BFE16_IMM(w4, b, 12);
    EXL3_BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    EXL3_BFE16_IMM(w1, s, 4);
    EXL3_BFE16_IMM(w0, s, 8);
}

// Generic K: 8 windows for run t0..t0+7 straight from the tile words (pointer extraction, the
// index math of exl3_dq.cuh's dq8 paths).  Windows wrap around the 256*bits-bit tile stream.
template <int bits>
__device__ __forceinline__ int wrap_idx(int i) {
    constexpr int words = bits * 8;
    return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + 3 * bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = ptr[wrap_idx<bits>(i0)];
    const uint32_t b = ptr[wrap_idx<bits>(i2)];
    w3 = exl3::fshift(b, a, s2) & 0xffff;
    w2 = exl3::fshift(b, a, s2 + bits) & 0xffff;
    w1 = exl3::fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1) {
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = ptr[wrap_idx<bits>(i0)];
    const uint32_t b = ptr[wrap_idx<bits>(i2)];
    w1 = exl3::fshift(b, a, s2) & 0xffff;
    w0 = exl3::fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext8w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3,
        uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    if constexpr (bits == 1) {
        const uint32_t i1 = t0 >> 5;
        const uint32_t i0 = (i1 + 7) & 7;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = exl3::fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        EXL3_BFE16_IMM(w6, b, 1); EXL3_BFE16_IMM(w5, b, 2); EXL3_BFE16_IMM(w4, b, 3); EXL3_BFE16_IMM(w3, b, 4);
        EXL3_BFE16_IMM(w2, b, 5); EXL3_BFE16_IMM(w1, b, 6); EXL3_BFE16_IMM(w0, b, 7);
    } else if constexpr (bits == 2) {
        const uint32_t i1 = t0 >> 4;
        const uint32_t i0 = (i1 + 15) & 15;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = exl3::fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        EXL3_BFE16_IMM(w6, b, 2); EXL3_BFE16_IMM(w5, b, 4); EXL3_BFE16_IMM(w4, b, 6); EXL3_BFE16_IMM(w3, b, 8);
        EXL3_BFE16_IMM(w2, b, 10); EXL3_BFE16_IMM(w1, b, 12); EXL3_BFE16_IMM(w0, b, 14);
    } else if constexpr (bits == 3) {
        const int b1 = (t0 + 257) * bits;
        const int b0 = b1 - 16;
        const int b2 = b1 + bits * 7;
        const int i0 = b0 / 32;
        const int i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        const uint32_t a = ptr[wrap_idx<bits>(i0)];
        const uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = exl3::fshift(b, a, s2);
        w6 = w7 >> bits; w5 = w6 >> bits; w4 = w5 >> bits;
        w3 = exl3::fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits; w1 = w2 >> bits; w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    } else if constexpr (bits == 4) {
        const uint32_t i1 = t0 >> 3;
        const uint32_t i0 = (i1 + 31) & 31;
        extract8_4bits(ptr[i0], ptr[i1], w0, w1, w2, w3, w4, w5, w6, w7);
    } else if constexpr (bits == 7) {
        ext2w<bits>(ptr, t0,     w0, w1);
        ext2w<bits>(ptr, t0 + 2, w2, w3);
        ext2w<bits>(ptr, t0 + 4, w4, w5);
        ext2w<bits>(ptr, t0 + 6, w6, w7);
    } else {  // 5, 6, 8
        ext4w<bits>(ptr, t0,     w0, w1, w2, w3);
        ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
    }
}

// ---- gemv ------------------------------------------------------------------------------------

// B: n-tile-major tile stream; x: [M][k] F32 activations, suh: [k] F16 input signs; y: [M][n] F32;
// partials: [ksplit][M][n] F32 (fully overwritten); counters: n/256 ints, zero at rest.
template <int bits, int M, bool RESID>
__global__ void __launch_bounds__(THREADS) gemv_int8_kernel(const uint8_t * __restrict__ B,
        const float * __restrict__ x, const half * __restrict__ suh, const half * __restrict__ svh, float * __restrict__ y,
        float * __restrict__ partials, int * __restrict__ counters, int k, int n, int nrows_max) {
    constexpr int TWORDS = 8 * bits;
    constexpr bool WIDE = bits == 4;   // uint2-per-lane block pair; other K use pointer extraction
    constexpr int NACC = (RESID ? 2 : 1) * M;
    extern __shared__ uint32_t sh_as[];   // [NACC][nrows_max * 16] splats, then [M][nrows_max * 16] F16 xh
    half * sh_xh = reinterpret_cast<half *>(sh_as + size_t(NACC) * nrows_max * 16);
    __shared__ float sh_y[M][COLS];
    __shared__ float sh_redf[THREADS / 32][M];
    __shared__ int   sh_redi[THREADS / 32][NACC];
    __shared__ float sh_q[NACC];
    __shared__ int   sh_s[NACC];
    __shared__ int   sh_last;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, lq = lane & 15;
    const int kslices = k / 16;
    const int kb0   = blockIdx.y * nrows_max;
    const int nrows = min(nrows_max, kslices - kb0);
    const int kn    = nrows * 16;

    // input transform of this block's own k range (128-aligned: nrows % 8 == 0): xh = had128(x * suh) / sqrt(128),
    // F16 in smem, with the per-slice max |xh| per row
    {
        float amax[M];
#pragma unroll
        for (int r = 0; r < M; ++r) amax[r] = 0.0f;
        for (int b = warp; b < (kn / 128) * M; b += THREADS / 32) {
            const int r   = b / (kn / 128);
            const int col = kb0 * 16 + (b - r * (kn / 128)) * 128 + lane * 4;
            const float4 xv = *reinterpret_cast<const float4 *>(x + size_t(r) * k + col);
            const half2 s01 = *reinterpret_cast<const half2 *>(suh + col);
            const half2 s23 = *reinterpret_cast<const half2 *>(suh + col + 2);
            float v0 = xv.x * __low2float(s01);
            float v1 = xv.y * __high2float(s01);
            float v2 = xv.z * __low2float(s23);
            float v3 = xv.w * __high2float(s23);
            exl3_had::had128(v0, v1, v2, v3, lane);
            const half2 h01 = __floats2half2_rn(v0 * exl3_had::SCALE, v1 * exl3_had::SCALE);
            const half2 h23 = __floats2half2_rn(v2 * exl3_had::SCALE, v3 * exl3_had::SCALE);
            half * dst = sh_xh + size_t(r) * nrows_max * 16 + (col - kb0 * 16);
            *reinterpret_cast<half2 *>(dst)     = h01;
            *reinterpret_cast<half2 *>(dst + 2) = h23;
            amax[r] = fmaxf(amax[r], fmaxf(fmaxf(fabsf(__low2float(h01)), fabsf(__high2float(h01))),
                                           fmaxf(fabsf(__low2float(h23)), fabsf(__high2float(h23)))));
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax[r] = fmaxf(amax[r], __shfl_xor_sync(0xffffffffu, amax[r], o));
            if (lane == 0) sh_redf[warp][r] = amax[r];
        }
        __syncthreads();
        if (threadIdx.x < NACC) {
            const int r = RESID ? threadIdx.x >> 1 : threadIdx.x;
            float mx = 0.0f;
            for (int w = 0; w < THREADS / 32; ++w) mx = fmaxf(mx, sh_redf[w][r]);
            const float q = fmaxf(mx, 1e-30f) / 127.0f;
            sh_q[threadIdx.x] = (RESID && (threadIdx.x & 1)) ? q / 254.0f : q;
        }
        __syncthreads();
    }
    // quantize inline while staging the splats; exact int sums per plane
    {
        int sum[NACC];
#pragma unroll
        for (int p = 0; p < NACC; ++p) sum[p] = 0;
        for (int i = threadIdx.x; i < kn; i += THREADS) {
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                const float a  = __half2float(sh_xh[size_t(r) * nrows_max * 16 + i]);
                const float q  = sh_q[p0];
                int v = __float2int_rn(a / q);
                v = max(-127, min(127, v));
                sh_as[p0 * nrows_max * 16 + i] = uint32_t(uint8_t(int8_t(v))) * 0x01010101u;
                sum[p0] += v;
                if constexpr (RESID) {
                    const float rr = a - q * float(v);
                    int v2 = __float2int_rn(rr / sh_q[p0 + 1]);
                    v2 = max(-127, min(127, v2));
                    sh_as[(p0 + 1) * nrows_max * 16 + i] = uint32_t(uint8_t(int8_t(v2))) * 0x01010101u;
                    sum[p0 + 1] += v2;
                }
            }
        }
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum[p] += __shfl_xor_sync(0xffffffffu, sum[p], o);
            if (lane == 0) sh_redi[warp][p] = sum[p];
        }
        __syncthreads();
        if (threadIdx.x < NACC) {
            int t = 0;
            for (int w = 0; w < THREADS / 32; ++w) t += sh_redi[w][threadIdx.x];
            sh_s[threadIdx.x] = t;
        }
        __syncthreads();
    }

    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float bias  = __half2float(__ushort_as_half(0xc931));
    const float cbias = 1024.0f * k_inv + bias;
    const uint32_t * B32 = reinterpret_cast<const uint32_t *>(B);

    if constexpr (WIDE) {
        const int nt = blockIdx.x * 16 + warp * 2 + (lane >> 4);
        const uint32_t * bp = B32 + (size_t(nt) * kslices + kb0) * TWORDS + 2 * lq;
        const int c2 = (lane & 1) ? 4 : 0;
        const int shfl_src = (lane & 16) | ((lane + 15) & 15);

        int acc0[NACC], acc1[NACC];
#pragma unroll
        for (int p = 0; p < NACC; ++p) { acc0[p] = 0; acc1[p] = 0; }

        uint2 r0 = nrows > 0 ? __ldcs(reinterpret_cast<const uint2 *>(bp)) : make_uint2(0, 0);
        uint2 r1 = nrows > 1 ? __ldcs(reinterpret_cast<const uint2 *>(bp + TWORDS)) : make_uint2(0, 0);
        for (int kb = 0; kb < nrows; ++kb) {
            uint2 r2 = make_uint2(0, 0);
            if (kb + 2 < nrows) r2 = __ldcs(reinterpret_cast<const uint2 *>(bp + size_t(kb + 2) * TWORDS));
            const uint32_t prev = __shfl_sync(0xffffffffu, r0.y, shfl_src);
            uint32_t w0, w1, w2, w3, w4, w5, w6, w7, v0, v1, v2, v3, v4, v5, v6, v7;
            extract8_4bits(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);   // run t = 8*(2m)
            extract8_4bits(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);   // run t = 8*(2m+1)
            w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
            w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
            v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
            v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;
#pragma unroll
            for (int p = 0; p < NACC; ++p) {
                const uint32_t * as = sh_as + p * nrows_max * 16 + (kb << 4);
                const uint4 as0 = *reinterpret_cast<const uint4 *>(as + c2);
                const uint4 as8 = *reinterpret_cast<const uint4 *>(as + c2 + 8);
                int i0 = acc0[p], i1 = acc1[p];
                i0 = dp4a_us(w0, as0.x, i0); i0 = dp4a_us(w1, as0.y, i0); i0 = dp4a_us(w2, as8.x, i0); i0 = dp4a_us(w3, as8.y, i0);
                i1 = dp4a_us(w4, as0.x, i1); i1 = dp4a_us(w5, as0.y, i1); i1 = dp4a_us(w6, as8.x, i1); i1 = dp4a_us(w7, as8.y, i1);
                i0 = dp4a_us(v0, as0.z, i0); i0 = dp4a_us(v1, as0.w, i0); i0 = dp4a_us(v2, as8.z, i0); i0 = dp4a_us(v3, as8.w, i0);
                i1 = dp4a_us(v4, as0.z, i1); i1 = dp4a_us(v5, as0.w, i1); i1 = dp4a_us(v6, as8.z, i1); i1 = dp4a_us(v7, as8.w, i1);
                acc0[p] = i0; acc1[p] = i1;
            }
            r0 = r1;
            r1 = r2;
        }

        // lanes l, l^1 hold the same two columns; fold the affine codebook terms with this slice's scales
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
            acc0[p] += __shfl_xor_sync(0xffffffffu, acc0[p], 1);
            acc1[p] += __shfl_xor_sync(0xffffffffu, acc1[p], 1);
        }
        if (!(lane & 1)) {
            const int n0 = nt * 16 + (lq >> 1);
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                float o0 = sh_q[p0] * (k_inv * float(acc0[p0]) + cbias * float(sh_s[p0]));
                float o1 = sh_q[p0] * (k_inv * float(acc1[p0]) + cbias * float(sh_s[p0]));
                if constexpr (RESID) {
                    o0 += sh_q[p0 + 1] * (k_inv * float(acc0[p0 + 1]) + cbias * float(sh_s[p0 + 1]));
                    o1 += sh_q[p0 + 1] * (k_inv * float(acc1[p0 + 1]) + cbias * float(sh_s[p0 + 1]));
                }
                float * part = partials + (size_t(blockIdx.y) * M + r) * n;
                part[n0]     = o0;
                part[n0 + 8] = o1;
            }
        }
    } else {
        // narrow unit: lane = standard lane t0 = 8*lane for both tiles of the pair, windows read by
        // pointer straight from the tile words (L1); four lanes share each column
        const int ntA = blockIdx.x * 16 + warp * 2;
        const uint32_t * bpA = B32 + (size_t(ntA) * kslices + kb0) * TWORDS;
        const uint32_t * bpB = B32 + (size_t(ntA + 1) * kslices + kb0) * TWORDS;
        const int c2 = 2 * (lane & 3);
        const int t0 = lane << 3;
        int ia0[NACC], ia1[NACC], ib0[NACC], ib1[NACC];
#pragma unroll
        for (int p = 0; p < NACC; ++p) { ia0[p] = 0; ia1[p] = 0; ib0[p] = 0; ib1[p] = 0; }
        for (int kb = 0; kb < nrows; ++kb) {
            uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
            const uint32_t * as_kb = sh_as + (kb << 4);
            ext8w<bits>(bpA + size_t(kb) * TWORDS, t0, w0, w1, w2, w3, w4, w5, w6, w7);
            w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
            w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
            for (int p = 0; p < NACC; ++p) {
                const uint32_t * as = as_kb + p * nrows_max * 16;
                const uint2 as01 = *reinterpret_cast<const uint2 *>(as + c2);
                const uint2 as89 = *reinterpret_cast<const uint2 *>(as + c2 + 8);
                int i0 = ia0[p], i1 = ia1[p];
                i0 = dp4a_us(w0, as01.x, i0); i0 = dp4a_us(w1, as01.y, i0); i0 = dp4a_us(w2, as89.x, i0); i0 = dp4a_us(w3, as89.y, i0);
                i1 = dp4a_us(w4, as01.x, i1); i1 = dp4a_us(w5, as01.y, i1); i1 = dp4a_us(w6, as89.x, i1); i1 = dp4a_us(w7, as89.y, i1);
                ia0[p] = i0; ia1[p] = i1;
            }
            ext8w<bits>(bpB + size_t(kb) * TWORDS, t0, w0, w1, w2, w3, w4, w5, w6, w7);
            w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
            w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
            for (int p = 0; p < NACC; ++p) {
                const uint32_t * as = as_kb + p * nrows_max * 16;
                const uint2 as01 = *reinterpret_cast<const uint2 *>(as + c2);
                const uint2 as89 = *reinterpret_cast<const uint2 *>(as + c2 + 8);
                int i0 = ib0[p], i1 = ib1[p];
                i0 = dp4a_us(w0, as01.x, i0); i0 = dp4a_us(w1, as01.y, i0); i0 = dp4a_us(w2, as89.x, i0); i0 = dp4a_us(w3, as89.y, i0);
                i1 = dp4a_us(w4, as01.x, i1); i1 = dp4a_us(w5, as01.y, i1); i1 = dp4a_us(w6, as89.x, i1); i1 = dp4a_us(w7, as89.y, i1);
                ib0[p] = i0; ib1[p] = i1;
            }
        }
        // lanes with equal lane/4 share the same columns (col lane/4 and +8 of each tile)
#pragma unroll
        for (int p = 0; p < NACC; ++p) {
            ia0[p] += __shfl_xor_sync(0xffffffffu, ia0[p], 1); ia0[p] += __shfl_xor_sync(0xffffffffu, ia0[p], 2);
            ia1[p] += __shfl_xor_sync(0xffffffffu, ia1[p], 1); ia1[p] += __shfl_xor_sync(0xffffffffu, ia1[p], 2);
            ib0[p] += __shfl_xor_sync(0xffffffffu, ib0[p], 1); ib0[p] += __shfl_xor_sync(0xffffffffu, ib0[p], 2);
            ib1[p] += __shfl_xor_sync(0xffffffffu, ib1[p], 1); ib1[p] += __shfl_xor_sync(0xffffffffu, ib1[p], 2);
        }
        if (!(lane & 3)) {
            const int cA = ntA * 16 + (lane >> 2);
            const int cB = cA + 16;
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int p0 = RESID ? 2 * r : r;
                auto fold = [&](int a, int p) { return sh_q[p] * (k_inv * float(a) + cbias * float(sh_s[p])); };
                float oa0 = fold(ia0[p0], p0), oa1 = fold(ia1[p0], p0), ob0 = fold(ib0[p0], p0), ob1 = fold(ib1[p0], p0);
                if constexpr (RESID) {
                    oa0 += fold(ia0[p0 + 1], p0 + 1); oa1 += fold(ia1[p0 + 1], p0 + 1);
                    ob0 += fold(ib0[p0 + 1], p0 + 1); ob1 += fold(ib1[p0 + 1], p0 + 1);
                }
                float * part = partials + (size_t(blockIdx.y) * M + r) * n;
                part[cA] = oa0; part[cA + 8] = oa1;
                part[cB] = ob0; part[cB + 8] = ob1;
            }
        }
    }

    // last k-split block for these 256 columns reduces the partials in slice order
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        sh_last = atomicAdd(counters + blockIdx.x, 1) == int(gridDim.y) - 1;
    }
    __syncthreads();
    if (!sh_last) return;
    __threadfence();

    const int col = blockIdx.x * COLS + threadIdx.x;
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float v = 0.0f;
        for (int sl = 0; sl < int(gridDim.y); ++sl) {
            v += __ldcg(partials + (size_t(sl) * M + r) * n + col);
        }
        sh_y[r][threadIdx.x] = v;
    }
    if (threadIdx.x == 0) counters[blockIdx.x] = 0;
    __syncthreads();
    // output Hadamard: 2 x 128-blocks per row, one warp each
    for (int b = warp; b < 2 * M; b += THREADS / 32) {
        const int r = b >> 1;
        const int c = (b & 1) * 128 + lane * 4;
        float v0 = sh_y[r][c], v1 = sh_y[r][c + 1], v2 = sh_y[r][c + 2], v3 = sh_y[r][c + 3];
        exl3_had::had128(v0, v1, v2, v3, lane);
        const int gc = blockIdx.x * COLS + c;
        const half2 s01 = *reinterpret_cast<const half2 *>(svh + gc);
        const half2 s23 = *reinterpret_cast<const half2 *>(svh + gc + 2);
        float4 o;
        o.x = v0 * exl3_had::SCALE * __low2float(s01);
        o.y = v1 * exl3_had::SCALE * __high2float(s01);
        o.z = v2 * exl3_had::SCALE * __low2float(s23);
        o.w = v3 * exl3_had::SCALE * __high2float(s23);
        *reinterpret_cast<float4 *>(y + size_t(r) * n + gc) = o;
    }
}

} // namespace exl3_int8
