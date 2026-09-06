#pragma once

// EXL3 decode gemv (m <= 8), kernel only: shared by exl3.cu and the standalone micro-benchmark.
// Config: WK warps split k, WNT adjacent n tiles per warp, PF prefetch ring depth (k slices, or
// k-slice quads when VEC4).  Layouts (per n tile, k tiles ascending):
//   VEC4 = false: [kt][32*bits bytes]                        (tile words, lane = word)
//   VEC4 = true : [kt/4][lane 0..31][4 words]  (bits == 4)   (one uint4 per lane = 4 k slices)

#include "exl3-dq.cuh"
#include <cuda_fp16.h>

namespace exl3_gemv {

constexpr int EXL3_CB = 2;         // "mul1" codebook
constexpr int EXL3_GEMV_MAX_M = 8;

using exl3::FragB;
using exl3::FragC_h;

// ---- decode gemv (m <= 8): warps split k, one m16n8k16 MMA pair per tile ------------------

__device__ __forceinline__ void exl3_mma_ab_h(const FragB & a01, const FragB & a23, const FragB & b, FragC_h & c) {
    const uint32_t * a0 = reinterpret_cast<const uint32_t *>(&a01);
    const uint32_t * a1 = reinterpret_cast<const uint32_t *>(&a23);
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    uint32_t * cc = reinterpret_cast<uint32_t *>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]), "r"(bb[0]), "r"(bb[1]));
}

// mul1 codebook pair decode via dp4a byte sum (bit-identical to the vabsdiff4 form)
__device__ __forceinline__ half2 exl3_decode_pair_cb2(uint32_t x0, uint32_t x1) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    const half2 k_inv  = __half2half2(__ushort_as_half(0x1eee));
    const half2 k_bias = __half2half2(__ushort_as_half(0xc931));
    return __hfma2(__halves2half2(__ushort_as_half(uint16_t(sum0)), __ushort_as_half(uint16_t(sum1))), k_inv, k_bias);
}

__device__ __forceinline__ void exl3_decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB & f0, FragB & f1) {
    f0[0] = exl3_decode_pair_cb2(w0, w1);
    f0[1] = exl3_decode_pair_cb2(w2, w3);
    f1[0] = exl3_decode_pair_cb2(w4, w5);
    f1[1] = exl3_decode_pair_cb2(w6, w7);
}

// Register forms of the aligned window extraction (exl3_gemv_kernel.cuh): each lane resolves its
// eight 16-bit windows from two already-loaded tile words instead of a shared-memory stage.
__device__ __forceinline__ void exl3_dq8_regs_4bits(uint32_t a, uint32_t b, FragB & f0, FragB & f1) {
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    EXL3_FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    EXL3_BFE16_IMM(w6, b, 4);
    EXL3_BFE16_IMM(w5, b, 8);
    EXL3_BFE16_IMM(w4, b, 12);
    EXL3_BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    EXL3_BFE16_IMM(w1, s, 4);
    EXL3_BFE16_IMM(w0, s, 8);
    exl3_decode8(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

__device__ __forceinline__ void exl3_dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = exl3::fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    EXL3_BFE16_IMM(w6, b, 2);
    EXL3_BFE16_IMM(w5, b, 4);
    EXL3_BFE16_IMM(w4, b, 6);
    EXL3_BFE16_IMM(w3, b, 8);
    EXL3_BFE16_IMM(w2, b, 10);
    EXL3_BFE16_IMM(w1, b, 12);
    EXL3_BFE16_IMM(w0, b, 14);
    exl3_decode8(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

__device__ __forceinline__ void exl3_dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = exl3::fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = exl3::fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    exl3_decode8(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                 w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

// A: xh [m][k] F16; B: tile stream (n-tile-major); C: y_inner [m][n] F32.
// 256 threads = 8 warps splitting k; each warp covers 4 adjacent n tiles (64 columns).
// 2/3/4 bpw stream the tile words straight to registers behind a prefetch ring and resolve the
// windows with lane shuffles; other widths stage each tile through warp-private shared memory.
template <int bits, int WK, int WNT, int PF, bool VEC4>
__global__ void __launch_bounds__(WK * 32) exl3_gemv_kernel(const half * __restrict__ A, const uint8_t * __restrict__ B,
        float * __restrict__ C, int size_m, int size_k, int size_n) {
    static_assert(!VEC4 || bits == 4, "VEC4 layout is 4 bpw only");
    constexpr int COLS   = WNT * 16;
    constexpr int ROWS   = EXL3_GEMV_MAX_M;
    constexpr int TWORDS = 8 * bits;   // uint32 per tile
    constexpr int FOLD   = 2;
    constexpr bool REG   = bits == 2 || bits == 3 || bits == 4;
    constexpr int LOADS  = bits == 2 ? WNT / 2 : WNT;   // warp loads per k slice (register path)
    constexpr int THREADS = WK * 32;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;
    const int chunk = VEC4 ? ((kslices + 4 * WK - 1) / (4 * WK)) * 4 : (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t * B32 = reinterpret_cast<const uint32_t *>(B);
    const half2 * A2 = reinterpret_cast<const half2 *>(A);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = size_t(r0) * (size_k / 2);
    const bool r0_ok = r0 < size_m;

    // per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3-dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3) {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float    sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[REG ? 1 : WK][REG ? 1 : WNT * TWORDS];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
        // tile (nt, kt) is contiguous: TWORDS words at ((nt * kslices) + kt) * TWORDS
        const uint32_t * bg = B32 + size_t(group * WNT) * kslices * TWORDS;
        auto tile_ptr = [&](int t, int kt) { return bg + (size_t(t) * kslices + kt) * TWORDS; };
        auto ld_b = [&](int i, int l) -> uint32_t {
            const int kt = ks0 + i;
            if constexpr (bits == 2) {
                return __ldcs(tile_ptr(2 * l + (lane >> 4), kt) + (lane & 15));
            } else if constexpr (bits == 3) {
                return lane < 24 ? __ldcs(tile_ptr(l, kt) + lane) : 0u;
            } else {
                return __ldcs(tile_ptr(l, kt) + lane);
            }
        };

        // VEC4: lane owns its tile word for four consecutive k slices in one uint4
        const uint4 * bg4 = reinterpret_cast<const uint4 *>(bg);
        const int kquads = kslices / 4;
        auto ld_b4 = [&](int q, int t) -> uint4 {
            return __ldcs(bg4 + (size_t(t) * kquads + q) * 32 + lane);
        };
        const int q0 = ks0 / 4;
        const int myq = (myn + 3) / 4;

        [[maybe_unused]] uint32_t pf[PF][LOADS];
        [[maybe_unused]] uint4 pf4[PF][WNT];
        if constexpr (VEC4) {
#pragma unroll
            for (int d = 0; d < PF; ++d) {
                if (d < myq) {
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        pf4[d][t] = ld_b4(q0 + d, t);
                    }
                }
            }
        } else if constexpr (REG) {
#pragma unroll
            for (int d = 0; d < PF; ++d) {
                if (d < myn) {
#pragma unroll
                    for (int l = 0; l < LOADS; ++l) {
                        pf[d][l] = ld_b(d, l);
                    }
                }
            }
        }

        FragC_h ch[WNT][2] = {};
        float2  acc[WNT][2] = {};
        // VEC4 iterates k-slice quads; otherwise single slices.  STEP slices per ring entry.
        constexpr int STEP = VEC4 ? 4 : 1;
        const int nsteps = VEC4 ? myq : myn;
        for (int ib = 0; ib < nsteps; ib += PF) {
#pragma unroll
            for (int d = 0; d < PF; ++d) {
                const int istep = ib + d;
                if (istep >= nsteps) {
                    break;
                }
                [[maybe_unused]] uint4 bq[WNT];
                [[maybe_unused]] uint32_t bw[LOADS];
                if constexpr (VEC4) {
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        bq[t] = pf4[d][t];
                    }
                    if (istep + PF < nsteps) {
#pragma unroll
                        for (int t = 0; t < WNT; ++t) {
                            pf4[d][t] = ld_b4(q0 + istep + PF, t);
                        }
                    }
                } else if constexpr (REG) {
#pragma unroll
                    for (int l = 0; l < LOADS; ++l) {
                        bw[l] = pf[d][l];
                    }
                    if (istep + PF < nsteps) {
#pragma unroll
                        for (int l = 0; l < LOADS; ++l) {
                            pf[d][l] = ld_b(istep + PF, l);
                        }
                    }
                }
#pragma unroll
                for (int sub = 0; sub < STEP; ++sub) {
                const int i = istep * STEP + sub;
                if (VEC4 && i >= myn) {
                    break;
                }
                const int kt = ks0 + i;
                if constexpr (VEC4) {
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        bw[t] = sub == 0 ? bq[t].x : sub == 1 ? bq[t].y : sub == 2 ? bq[t].z : bq[t].w;
                    }
                } else if constexpr (!REG) {
                    __syncwarp();
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        const uint32_t * tp = tile_ptr(t, kt);
                        for (int w = lane; w < TWORDS; w += 32) {
                            sh_stage[warp][t * TWORDS + w] = __ldcs(tp + w);
                        }
                    }
                    __syncwarp();
                }

                // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
                const size_t a_col = size_t(kt) * 8 + (lane & 3);
                FragB a01, a23;
                a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
                a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
                a01[1] = hzero;
                a23[1] = hzero;
#pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    FragB f0, f1;
                    if constexpr (bits == 4) {
                        const uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                        exl3_dq8_regs_4bits(aw, bw[t], f0, f1);
                    } else if constexpr (bits == 2) {
                        // two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                        const uint32_t w = bw[t >> 1];
                        const int base = (t & 1) << 4;
                        const uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                        const uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                        exl3_dq8_regs_2bits(awv, bwv, lane << 3, f0, f1);
                    } else if constexpr (bits == 3) {
                        const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                        const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                        exl3_dq8_regs_3bits(awv, bwv, x_s2, f0, f1);
                    } else {
                        exl3::dq_dispatch<bits, EXL3_CB>(&sh_stage[warp][t * TWORDS], lane * 8, f0, f1);
                    }
                    exl3_mma_ab_h(a01, a23, f0, ch[t][0]);
                    exl3_mma_ab_h(a01, a23, f1, ch[t][1]);
                }
                if ((i + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
#pragma unroll
                        for (int f = 0; f < 2; ++f) {
                            acc[t][f].x += __low2float(ch[t][f][0]);
                            acc[t][f].y += __high2float(ch[t][f][0]);
                            ch[t][f][0] = hzero;
                        }
                    }
                }
                }  // sub
            }
        }
        // cross-warp reduction over the k splits; lane holds row r0, cols t*16 + f*8 + 2(lane%4) (+1)
        if (r0 < ROWS) {
            const int c0 = 2 * (lane & 3);
#pragma unroll
            for (int t = 0; t < WNT; ++t) {
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + c0;
                    sh_red[warp][r0][col + 0] = acc[t][f].x;
                    sh_red[warp][r0][col + 1] = acc[t][f].y;
                }
            }
        }
        __syncthreads();
        const int rows_out = min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < WK; ++j) {
                sum += sh_red[j][r][c];
            }
            C[size_t(r) * size_n + group * COLS + c] = sum;
        }
        __syncthreads();
    }
}


} // namespace exl3_gemv
