#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

// InnerQ: update the fattn-side inverse scale array from host
void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
}

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// === Turbo prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize turbo K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_turbo2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2;
    const int j_in_blk = j % QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_turbo_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3;
    const int j_in_blk = j % QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4;
    const int j_in_blk = j % QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_turbo_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

// turbo4 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo4 blocks.
static __global__ void k_turbo4_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        // Extract 4-bit index, lookup centroid, apply signs2
        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);
        smem[tid] = d_turbo_centroids_4bit_fattn[idx] * s2[tid];
        __syncthreads();

        // 7 butterfly passes (inverse FWHT)
        for (int h = 1; h < 128; h *= 2) {
            if (tid < 64) {
                int j = (tid / h) * (2 * h) + (tid % h);
                float a = smem[j], b = smem[j + h];
                smem[j] = a + b; smem[j + h] = a - b;
            }
            __syncthreads();
        }

        // Normalize, apply signs1, undo InnerQ scaling, apply norm, cast to fp16
        float val = smem[tid] * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        dst[dst_base + blk_idx * 128 + tid] = __float2half(val);
    }
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// Persistent TBQ decode dequant buffers per device (avoid cudaMallocAsync per token)
static half * tbq_k_dec_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t tbq_k_dec_size[GGML_CUDA_MAX_DEVICES] = {};
static half * tbq_v_dec_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t tbq_v_dec_size[GGML_CUDA_MAX_DEVICES] = {};

// === FWHT rotation kernels for pre-rotate-queries approach ===
// Forward rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block.
static __global__ void k_turbo_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;

    __shared__ float buf[128];

    if (threadIdx.x < 128) {
        // InnerQ: apply inverse channel scale to Q before rotation
        // This compensates for the channel scaling applied to K in SET_ROWS
        buf[threadIdx.x] = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x] * s1[threadIdx.x];
    }
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (threadIdx.x < 128) {
        dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s2[threadIdx.x];
    }
}

// === TBQ Rademacher constants + FWHT + dequant kernels ===
static __constant__ uint32_t d_tbq_rademacher_fattn[4] = {
    0xa3b1c6d9u, 0x7e4f2a85u, 0xd1936cf0u, 0x5b8e47a2u
};

static __global__ void k_tbq_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;
    __shared__ float buf[128];
    if (threadIdx.x < 128) {
        const int word = threadIdx.x / 32, bit = threadIdx.x % 32;
        const float sign = (d_tbq_rademacher_fattn[word] >> bit) & 1 ? -1.0f : 1.0f;
        buf[threadIdx.x] = src[offset + threadIdx.x] * sign;
    }
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (threadIdx.x < 128) dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128;
}

// TBQ2 dequant to f16 with full inverse SRHT (Hadamard + Rademacher signs)
static __global__ void k_tbq2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row = blockIdx.x, head = blockIdx.y, strm = blockIdx.z;
    const int tid = threadIdx.x;
    if (tid >= ne0) return;
    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx = tid / QK_TBQ2, j_in = tid % QK_TBQ2;
    const block_tbq2_0 * blk = (const block_tbq2_0 *)src_row + blk_idx;
    const float norm = __half2float(blk->norm);
    const int idx = (blk->qs[j_in / 4] >> ((j_in % 4) * 2)) & 0x3;
    extern __shared__ float smem_dq[];
    float * sm = smem_dq + blk_idx * QK_TBQ2;
    sm[j_in] = d_tbq2_centroids_fattn[idx];
    __syncthreads();
    for (int step = 1; step < QK_TBQ2; step <<= 1) {
        int partner = j_in ^ step;
        float a = sm[j_in], b = sm[partner];
        __syncthreads();
        if (j_in < partner) { sm[j_in] = a + b; sm[partner] = a - b; }
        __syncthreads();
    }
    sm[j_in] *= 0.08838834764831845f;
    __syncthreads();
    const int word = j_in / 32, bit = j_in % 32;
    const float sign = (d_tbq_rademacher_fattn[word] >> bit) & 1 ? -1.0f : 1.0f;
    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + tid] = __float2half(sm[j_in] * sign * norm);
}

// TBQ dequant to f16 with full inverse SRHT (Hadamard + Rademacher signs)
static __global__ void k_tbq3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row = blockIdx.x, head = blockIdx.y, strm = blockIdx.z;
    const int tid = threadIdx.x;
    if (tid >= ne0) return;
    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx = tid / QK_TBQ3, j_in = tid % QK_TBQ3;
    const block_tbq3_0 * blk = (const block_tbq3_0 *)src_row + blk_idx;
    const float norm = __half2float(blk->norm);
    const int bo = j_in * 3, by = bo / 8, bp = bo % 8;
    int idx = (blk->qs[by] >> bp);
    if (bp > 5) idx |= ((int)blk->qs[by + 1]) << (8 - bp);
    idx &= 0x7;
    extern __shared__ float smem_dq[];
    float * sm = smem_dq + blk_idx * QK_TBQ3;
    sm[j_in] = d_tbq3_centroids_fattn[idx];
    __syncthreads();
    for (int step = 1; step < QK_TBQ3; step <<= 1) {
        int partner = j_in ^ step;
        float a = sm[j_in], b = sm[partner];
        __syncthreads();
        if (j_in < partner) { sm[j_in] = a + b; sm[partner] = a - b; }
        __syncthreads();
    }
    sm[j_in] *= 0.08838834764831845f;
    __syncthreads();
    const int word = j_in / 32, bit = j_in % 32;
    const float sign = (d_tbq_rademacher_fattn[word] >> bit) & 1 ? -1.0f : 1.0f;
    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + tid] = __float2half(sm[j_in] * sign * norm);
}

static __global__ void k_tbq4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row = blockIdx.x, head = blockIdx.y, strm = blockIdx.z;
    const int tid = threadIdx.x;
    if (tid >= ne0) return;
    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx = tid / QK_TBQ4, j_in = tid % QK_TBQ4;
    const block_tbq4_0 * blk = (const block_tbq4_0 *)src_row + blk_idx;
    const float norm = __half2float(blk->norm);
    const uint8_t packed = blk->qs[j_in / 2];
    const int idx = (j_in & 1) ? ((packed >> 4) & 0xf) : (packed & 0xf);
    extern __shared__ float smem_dq[];
    float * sm = smem_dq + blk_idx * QK_TBQ4;
    sm[j_in] = d_tbq4_centroids_fattn[idx];
    __syncthreads();
    for (int step = 1; step < QK_TBQ4; step <<= 1) {
        int partner = j_in ^ step;
        float a = sm[j_in], b = sm[partner];
        __syncthreads();
        if (j_in < partner) { sm[j_in] = a + b; sm[partner] = a - b; }
        __syncthreads();
    }
    sm[j_in] *= 0.08838834764831845f;
    __syncthreads();
    const int word = j_in / 32, bit = j_in % 32;
    const float sign = (d_tbq_rademacher_fattn[word] >> bit) & 1 ? -1.0f : 1.0f;
    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + tid] = __float2half(sm[j_in] * sign * norm);
}

// === TBQ chunked prefill: FlashAttention online softmax with O(CHUNK) temp memory ===
// Enables 350K+ context on 27B models by processing KV in chunks of CHUNK tokens.
// Uses cuBLAS SGEMM for Q@K^T and P@V, custom kernels for online softmax updates.
//
// TBQ_CHUNK must be a power of 2. The last chunk may have fewer tokens (chunk_len < TBQ_CHUNK)
// — we pad shared memory to chunk_pad = next_pow2(chunk_len) for the binary-tree reduction.
// S is always allocated with stride TBQ_CHUNK so pointer arithmetic is consistent.
static constexpr int TBQ_CHUNK = 4096;

// Kernel 1: Initialize accumulators for online softmax.
// O_acc = 0, l_acc = 0, m_acc = -inf
// Grid: (nq_heads, 1, 1), blockDim.x = min(D, 1024).
// Thread 0 initializes l_acc and m_acc; all threads loop over D to zero O_acc.
static __global__ void k_chunked_attn_init(
        float * __restrict__ O_acc,
        float * __restrict__ l_acc,
        float * __restrict__ m_acc,
        const int64_t nq_heads,
        const int64_t D) {
    const int64_t hq  = (int64_t)blockIdx.x;
    const int     tid = (int)threadIdx.x;
    const int     bdx = (int)blockDim.x;
    if (hq >= nq_heads) return;
    if (tid == 0) {
        l_acc[hq] = 0.0f;
        m_acc[hq] = -INFINITY;
    }
    for (int64_t d = tid; d < D; d += bdx) {
        O_acc[hq * D + d] = 0.0f;
    }
}

// Kernel 2: Online softmax update.
// Processes one (head, query) pair per thread block.
// Uses dynamic shared memory of size (chunk_pad + 2) * sizeof(float):
//   sm[0..chunk_pad-1]: scores / exp values for reduction
//   sm[chunk_pad]:      broadcast alpha (exp(m_old - m_new))
//   sm[chunk_pad+1]:    broadcast beta  (exp(m_chunk - m_new))
//
// blockDim.x = min(chunk_pad, 1024). Each thread covers multiple sm slots in load/store passes.
// Binary tree reduction uses only threads 0..stride-1, which is ≤ blockDim.x for all strides
// once the initial load has populated sm[] with all chunk_pad values.
//
// Algorithm:
//   1. Load S[head,q,:] + mask into sm[0..chunk_len-1], pad sm[chunk_len..chunk_pad-1] = -inf
//   2. Tree-reduce sm → m_chunk = max
//   3. Compute exp(score - m_chunk), pad with 0
//   4. Tree-reduce sm → l_chunk = sum
//   5. Update m_acc, l_acc, broadcast alpha/beta via sm[chunk_pad..chunk_pad+1]
//   6. Rescale O_acc by alpha (parallel over D)
//   7. Write P[c] = beta * exp(S[c] - m_chunk) to S (for cuBLAS P@V)
static __global__ void k_chunked_softmax_update(
        float * __restrict__ S,          // [nh_q, nq, TBQ_CHUNK] scores → P after kernel
        float * __restrict__ O_acc,      // [nh_q, nq, D]
        float * __restrict__ l_acc,      // [nh_q, nq]
        float * __restrict__ m_acc,      // [nh_q, nq]
        const int chunk_len,             // actual tokens in this chunk (≤ TBQ_CHUNK)
        const int chunk_pad,             // next power-of-2 ≥ chunk_len
        const int64_t D,
        const int64_t nq,
        const int64_t nh_q,
        const half  * __restrict__ mask, // [nq, nkv_total] f16 mask, or nullptr
        const int64_t mask_stride,       // mask->nb[1]/sizeof(half)
        const int64_t kv_start) {        // absolute KV offset for mask column
    const int64_t hq_idx = (int64_t)blockIdx.x;
    const int64_t head   = hq_idx / nq;
    const int64_t q_pos  = hq_idx % nq;
    if (head >= nh_q) return;

    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;
    extern __shared__ float sm[];  // (chunk_pad + 2) floats
    // sm[chunk_pad]   = sh_alpha (broadcast)
    // sm[chunk_pad+1] = sh_beta  (broadcast)

    // S is strided by TBQ_CHUNK (constant), not chunk_len
    const int64_t s_base = head * (int64_t)nq * TBQ_CHUNK + q_pos * TBQ_CHUNK;

    // --- Step 1: Load scores + mask into sm[], pad with -inf ---
    for (int c = tid; c < chunk_pad; c += bdx) {
        if (c < chunk_len) {
            float val = S[s_base + c];
            if (mask != nullptr) {
                val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
            }
            sm[c] = val;
        } else {
            sm[c] = -INFINITY;
        }
    }
    __syncthreads();

    // --- Step 2: Binary tree max reduction ---
    for (int stride = chunk_pad >> 1; stride >= 1; stride >>= 1) {
        for (int c = tid; c < stride; c += bdx) {
            sm[c] = fmaxf(sm[c], sm[c + stride]);
        }
        __syncthreads();
    }
    const float m_chunk = sm[0];
    __syncthreads();

    // --- Step 3: Compute exp(score - m_chunk), pad with 0 ---
    for (int c = tid; c < chunk_pad; c += bdx) {
        if (c < chunk_len) {
            float val = S[s_base + c];
            if (mask != nullptr) {
                val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
            }
            sm[c] = __expf(val - m_chunk);
        } else {
            sm[c] = 0.0f;
        }
    }
    __syncthreads();

    // --- Step 4: Binary tree sum reduction ---
    for (int stride = chunk_pad >> 1; stride >= 1; stride >>= 1) {
        for (int c = tid; c < stride; c += bdx) {
            sm[c] += sm[c + stride];
        }
        __syncthreads();
    }
    const float l_chunk = sm[0];
    __syncthreads();

    // --- Step 5: Update m_acc, l_acc, compute and broadcast alpha/beta ---
    if (tid == 0) {
        const float m_old = m_acc[hq_idx];
        const float m_new = fmaxf(m_old, m_chunk);
        // When m_old == -inf (first chunk), alpha = 0 to avoid 0 * (-inf) = NaN
        const float alpha = (m_old > -INFINITY) ? __expf(m_old - m_new) : 0.0f;
        const float beta  = __expf(m_chunk - m_new);
        sm[chunk_pad]     = alpha;
        sm[chunk_pad + 1] = beta;
        l_acc[hq_idx] = alpha * l_acc[hq_idx] + beta * l_chunk;
        m_acc[hq_idx] = m_new;
    }
    __syncthreads();

    const float alpha = sm[chunk_pad];
    const float beta  = sm[chunk_pad + 1];

    // --- Step 6: Rescale O_acc in parallel over D ---
    for (int64_t d = tid; d < D; d += bdx) {
        O_acc[hq_idx * D + d] *= alpha;
    }
    __syncthreads();

    // --- Step 7: Write P = beta * exp(S - m_chunk) back to S ---
    for (int c = tid; c < chunk_len; c += bdx) {
        float val = S[s_base + c];
        if (mask != nullptr) {
            val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
        }
        S[s_base + c] = beta * __expf(val - m_chunk);
    }
}

// Kernel 3: Finalize attention output.
// Computes dst[head, q, d] = f16(O_acc[head, q, d] / l_acc[head, q])
static __global__ void k_chunked_attn_finalize(
        const float * __restrict__ O_acc, // [nh_q, nq, D]
        const float * __restrict__ l_acc, // [nh_q, nq]
        half        * __restrict__ dst,   // [nh_q, nq, D] f16 output
        const int64_t nq,
        const int64_t D) {
    const int64_t hq_idx = (int64_t)blockIdx.x;  // head * nq + q_pos
    const int64_t d      = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const float l = fmaxf(l_acc[hq_idx], 1e-30f);
    dst[hq_idx * D + d] = __float2half(O_acc[hq_idx * D + d] / l);
}

// Helper: next power of 2 >= n (host side, n >= 1)
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// TBQ chunked prefill: FlashAttention with O(CHUNK_SIZE) temp memory.
// Replaces ggml_cuda_tbq_prefill_attend for nkv > TBQ_CHUNKED_PREFILL_THRESHOLD.
static void ggml_cuda_tbq_chunked_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t D     = Q->ne[0];    // head_dim
    const int64_t nq    = Q->ne[1];    // query tokens
    const int64_t nkv   = K->ne[1];    // KV tokens
    const int64_t nh_q  = Q->ne[2];    // Q heads
    const int64_t nh_kv = K->ne[2];    // KV heads
    const int64_t gqa   = nh_q / nh_kv;

    // Extract attention scale from op_params
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // --- Allocate persistent chunk dequant buffers (TBQ_CHUNK * nh_kv * D each) ---
    half * k_tmp = nullptr;
    half * v_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&k_tmp, (size_t)D * TBQ_CHUNK * nh_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&v_tmp, (size_t)D * TBQ_CHUNK * nh_kv * sizeof(half)));

    // Check that total allocations fit in ~14GB available VRAM
    // S = nh_q * nq * TBQ_CHUNK * 4, O_acc+Q_f32 = 2 * nh_q * nq * D * 4
    const size_t s_size = (size_t)nh_q * nq * TBQ_CHUNK * sizeof(float);
    const size_t acc_size = 2 * (size_t)nh_q * nq * D * sizeof(float);
    if (s_size + acc_size > 14ULL * 1024 * 1024 * 1024) {
        // Too large for chunked cuBLAS — fall back to the standard MMA dequant path.
        // This re-enters the dispatch which will use MMA with full f16 temp buffer.
        half *k_fp16 = nullptr, *v_fp16 = nullptr;
        // ... just allocate and dequant the full KV like the original function
        // For now, ASSERT — caller should check before calling chunked path
        GGML_ASSERT(false && "chunked prefill: nq too large, reduce batch size or context");
        return;
    }

    // --- Allocate float accumulators ---
    float * O_acc = nullptr;
    float * l_acc = nullptr;
    float * m_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&O_acc, nh_q * nq * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l_acc, nh_q * nq     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_acc, nh_q * nq     * sizeof(float)));

    // --- Initialize accumulators ---
    // One block per (head, query) pair; D threads per block zero O_acc in parallel.
    {
        const int64_t nq_heads = nh_q * nq;
        // Cap D at 1024 (CUDA max threads/block); O_acc loop handles D > 1024
        const int threads_init = (int)std::min(D, (int64_t)1024);
        k_chunked_attn_init<<<(int)nq_heads, threads_init, 0, stream>>>(O_acc, l_acc, m_acc, nq_heads, D);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- cuBLAS handle, set to current stream ---
    cublasHandle_t cublas_handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    // Cast Q to float32 for cuBLAS (Q may be f32 already, but we copy to contiguous buffer)
    // Q layout: [D, nq, nh_q] with strides nb[0..3]
    float * Q_f32 = nullptr;
    CUDA_CHECK(cudaMalloc(&Q_f32, nh_q * nq * D * sizeof(float)));
    // Q->data is f32 (GGML_TYPE_F32), strides: nb[0]=4, nb[1]=D*4, nb[2]=nq*D*4, nb[3]=...
    // We need a contiguous [nh_q, nq, D] float buffer
    // Use cudaMemcpy2DAsync or a kernel if non-contiguous. For prefill Q is typically contiguous.
    if (Q->nb[0] == sizeof(float) &&
        Q->nb[1] == (size_t)D * sizeof(float) &&
        Q->nb[2] == (size_t)D * nq * sizeof(float)) {
        // Already contiguous
        CUDA_CHECK(cudaMemcpyAsync(Q_f32, Q->data, nh_q * nq * D * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        // Non-contiguous: copy element by element via a simple kernel
        // (Rare case — prefill Q is almost always contiguous)
        // For safety, use a 1D memcpy kernel
        // TODO: handle non-contiguous Q via a proper gather kernel if needed
        GGML_ASSERT(Q->nb[0] == sizeof(float)); // must be at least element-contiguous
        for (int64_t h = 0; h < nh_q; h++) {
            for (int64_t q = 0; q < nq; q++) {
                const char * src_ptr = (const char *)Q->data + h * Q->nb[2] + q * Q->nb[1];
                float * dst_ptr = Q_f32 + h * nq * D + q * D;
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, D * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    // Check K/V types
    const bool tbq_k = K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0;
    const bool tbq_v = V->type == GGML_TYPE_TBQ2_0 || V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0;

    // Mask layout: [nkv, nq] f16 — mask[q_pos * (mask->nb[1]/sizeof(half)) + kv_pos]
    const half * mask_data    = mask ? (const half *) mask->data : nullptr;
    const int64_t mask_stride = mask ? (int64_t)(mask->nb[1] / sizeof(half)) : 0;

    // S buffer: [nh_q, nq, TBQ_CHUNK] float32 — reused for each KV chunk.
    float * S = nullptr;
    CUDA_CHECK(cudaMalloc(&S, s_size));

    // Main loop over KV chunks
    for (int64_t kv_start = 0; kv_start < nkv; kv_start += TBQ_CHUNK) {
        const int64_t chunk_len = (kv_start + TBQ_CHUNK <= nkv) ? TBQ_CHUNK : (nkv - kv_start);

        // --- 1. Dequant K chunk to f16 ---
        // The existing dequant kernels use: row = blockIdx.x, head = blockIdx.y, strm = blockIdx.z
        // src offset: kv_start * nb[1] into K->data
        // Output: k_tmp[head * chunk_len * D + row * D + d] (contiguous, chunk_len rows per head)
        // We pass ne1=chunk_len and offset the src pointer by kv_start * K->nb[1]
        {
            const char * k_src = (const char *)K->data + kv_start * K->nb[1];
            dim3 grid_k((int)chunk_len, (int)nh_kv, 1);
            const size_t smem_k = D * sizeof(float);
            if (tbq_k) {
                if (K->type == GGML_TYPE_TBQ2_0) {
                    k_tbq2_dequant_f16<<<grid_k, (int)D, smem_k, stream>>>(
                        k_src, k_tmp, D, chunk_len, nh_kv, K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TBQ3_0) {
                    k_tbq3_dequant_f16<<<grid_k, (int)D, smem_k, stream>>>(
                        k_src, k_tmp, D, chunk_len, nh_kv, K->nb[1], K->nb[2], K->nb[3]);
                } else {
                    k_tbq4_dequant_f16<<<grid_k, (int)D, smem_k, stream>>>(
                        k_src, k_tmp, D, chunk_len, nh_kv, K->nb[1], K->nb[2], K->nb[3]);
                }
            } else {
                // K is already f16 — copy chunk to k_tmp
                // K->nb[1] = D * sizeof(half) for contiguous f16
                CUDA_CHECK(cudaMemcpy2DAsync(
                    k_tmp, D * sizeof(half),
                    (const char *)K->data + kv_start * K->nb[1], K->nb[1],
                    D * sizeof(half), (size_t)(chunk_len * nh_kv),
                    cudaMemcpyDeviceToDevice, stream));
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // --- 2. Dequant V chunk to f16 ---
        {
            const char * v_src = (const char *)V->data + kv_start * V->nb[1];
            dim3 grid_v((int)chunk_len, (int)nh_kv, 1);
            const size_t smem_v = D * sizeof(float);
            if (tbq_v) {
                if (V->type == GGML_TYPE_TBQ2_0) {
                    k_tbq2_dequant_f16<<<grid_v, (int)D, smem_v, stream>>>(
                        v_src, v_tmp, D, chunk_len, nh_kv, V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TBQ3_0) {
                    k_tbq3_dequant_f16<<<grid_v, (int)D, smem_v, stream>>>(
                        v_src, v_tmp, D, chunk_len, nh_kv, V->nb[1], V->nb[2], V->nb[3]);
                } else {
                    k_tbq4_dequant_f16<<<grid_v, (int)D, smem_v, stream>>>(
                        v_src, v_tmp, D, chunk_len, nh_kv, V->nb[1], V->nb[2], V->nb[3]);
                }
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    v_tmp, D * sizeof(half),
                    (const char *)V->data + kv_start * V->nb[1], V->nb[1],
                    D * sizeof(half), (size_t)(chunk_len * nh_kv),
                    cudaMemcpyDeviceToDevice, stream));
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // --- 3. S = Q_f32 @ k_tmp^T * scale via cuBLAS ---
        // Q_f32 layout: [nh_q, nq, D]  (row-major, each head is a batch)
        // k_tmp layout: [nh_kv, chunk_len, D] (after dequant kernel: contiguous)
        // S layout: [nh_q, nq, TBQ_CHUNK] float32 — stride is TBQ_CHUNK even for last chunk
        // cuBLAS is column-major. To compute S[b] = Q[b] @ K[b]^T (row-major):
        //   Treat as K[b]^T @ Q[b] in column-major:
        //   A = K chunk col-major: K[chunk_len × D] stored [D, chunk_len] → lda = D, OP_N
        //   B = Q col-major: Q[nq × D] stored [D, nq] → ldb = D, OP_T
        //   C = S: S[chunk_len × nq] col-major → ldc = chunk_len (but stride = TBQ_CHUNK)
        //   C[i,j] = sum_d K[i,d] * Q[j,d] = (Q @ K^T)[j,i] — correct row-major Q@K^T

        {
            const float alpha_v = scale;
            const float beta_v  = 0.0f;

            // k_tmp head stride: contiguous chunk (chunk_len rows × D cols)
            const long long stride_A = (long long)chunk_len * D;
            // Q head stride
            const long long stride_B = (long long)nq * D;
            // S head stride: TBQ_CHUNK (fixed for entire run)
            const long long stride_C = (long long)nq * TBQ_CHUNK;

            if (gqa == 1) {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,   // A=K no-T, B=Q transpose
                    (int)chunk_len, (int)nq, (int)D,
                    &alpha_v,
                    k_tmp,   CUDA_R_16F, (int)D,            stride_A,  // A: K chunk
                    Q_f32,   CUDA_R_32F, (int)D,            stride_B,  // B: Q
                    &beta_v,
                    S,       CUDA_R_32F, (int)TBQ_CHUNK,    stride_C,  // C: S (ldc = TBQ_CHUNK)
                    (int)nh_q,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                // GQA: share K head across gqa Q heads using stride=0 for A
                for (int64_t kv_h = 0; kv_h < nh_kv; kv_h++) {
                    const half  * k_head  = k_tmp + kv_h * chunk_len * D;
                    const float * q_start = Q_f32 + kv_h * gqa * nq * D;
                    float       * s_start = S     + kv_h * gqa * (long long)nq * TBQ_CHUNK;
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)chunk_len, (int)nq, (int)D,
                        &alpha_v,
                        k_head,   CUDA_R_16F, (int)D,            0LL,                   // A: same K for all gqa sub-batches
                        q_start,  CUDA_R_32F, (int)D,            (long long)nq * D,     // B: consecutive Q heads
                        &beta_v,
                        s_start,  CUDA_R_32F, (int)TBQ_CHUNK,    (long long)nq * TBQ_CHUNK,
                        (int)gqa,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        }

        // --- 4. Online softmax update (updates S in-place to P, updates O_acc/l_acc/m_acc) ---
        {
            const int64_t nq_heads   = nh_q * nq;
            const int chunk_len_int  = (int)chunk_len;
            // chunk_pad: next power-of-2 >= chunk_len, for binary tree reduction in shared memory
            const int chunk_pad      = next_pow2(chunk_len_int);
            // Use up to 1024 threads/block; each thread covers multiple sm slots via stride loops
            const int threads_sm     = (chunk_pad < 1024) ? chunk_pad : 1024;
            // smem: chunk_pad floats for scores/exp + 2 extra floats for alpha/beta broadcast
            const size_t smem        = ((size_t)chunk_pad + 2) * sizeof(float);
            k_chunked_softmax_update<<<(int)nq_heads, threads_sm, smem, stream>>>(
                S, O_acc, l_acc, m_acc,
                chunk_len_int, chunk_pad,
                D, nq, nh_q,
                mask_data, mask_stride, kv_start);
            CUDA_CHECK(cudaGetLastError());
        }

        // --- 5. O_acc += P @ v_tmp ---
        // P (now in S): [nh_q, nq, TBQ_CHUNK] float32
        // v_tmp: [nh_kv, chunk_len, D] float16
        // O_acc: [nh_q, nq, D] float32
        // cuBLAS col-major: O = P @ V → A=V, B=P^T
        //   A = V [chunk_len × D] col-major [D, chunk_len] → lda=D, OP_N
        //   B = P [nq × chunk_len] → transpose → col-major [chunk_len, nq] → ldb=TBQ_CHUNK, OP_T
        //   C = O [D × nq] col-major → ldc=D
        {
            const float alpha_v = 1.0f;
            const float beta_v  = 1.0f;  // accumulate into O_acc

            const long long stride_A = (long long)chunk_len * D;       // V head stride
            const long long stride_B = (long long)nq * TBQ_CHUNK;      // P head stride
            const long long stride_C = (long long)nq * D;              // O head stride

            if (gqa == 1) {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,   // A=V no-T, B=P transpose
                    (int)D, (int)nq, (int)chunk_len,
                    &alpha_v,
                    v_tmp, CUDA_R_16F, (int)D,            stride_A,  // A: V chunk
                    S,     CUDA_R_32F, (int)TBQ_CHUNK,    stride_B,  // B: P (stride = TBQ_CHUNK)
                    &beta_v,
                    O_acc, CUDA_R_32F, (int)D,            stride_C,  // C: O_acc
                    (int)nh_q,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                // GQA: each KV head is shared across gqa Q heads; V stride=0 for A
                for (int64_t kv_h = 0; kv_h < nh_kv; kv_h++) {
                    const half  * v_head  = v_tmp  + kv_h * chunk_len * D;
                    const float * p_start = S      + kv_h * gqa * (long long)nq * TBQ_CHUNK;
                    float       * o_start = O_acc  + kv_h * gqa * (long long)nq * D;
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)D, (int)nq, (int)chunk_len,
                        &alpha_v,
                        v_head,   CUDA_R_16F, (int)D,            0LL,                       // A: same V for all gqa sub-batches
                        p_start,  CUDA_R_32F, (int)TBQ_CHUNK,    (long long)nq * TBQ_CHUNK, // B: consecutive P heads
                        &beta_v,
                        o_start,  CUDA_R_32F, (int)D,            (long long)nq * D,
                        (int)gqa,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        }

    }  // end KV chunk loop

    // --- 6. Finalize: dst = f16(O_acc / l_acc) ---
    {
        // dst layout: [D, nq, nh_q] in ggml convention (ne[0]=D, ne[1]=nq, ne[2]=nh_q)
        // O_acc layout: [nh_q, nq, D] — same logical layout, just C-order vs ggml order
        // ggml stores as ne[0]=D fastest, so [nh_q, nq, D] matches dst->data layout
        const int64_t nq_heads = nh_q * nq;
        const int threads_fin = 128;
        // Grid: x = nq_heads, y = ceil(D / threads_fin)
        const dim3 grid_fin((int)nq_heads, (int)((D + threads_fin - 1) / threads_fin));
        k_chunked_attn_finalize<<<grid_fin, threads_fin, 0, stream>>>(
            O_acc, l_acc, (half *)dst->data, nq, D);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- 7. Sync and free ---
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(S));
    CUDA_CHECK(cudaFree(k_tmp));
    CUDA_CHECK(cudaFree(v_tmp));
    CUDA_CHECK(cudaFree(O_acc));
    CUDA_CHECK(cudaFree(l_acc));
    CUDA_CHECK(cudaFree(m_acc));
    CUDA_CHECK(cudaFree(Q_f32));
}

// Threshold for switching from bulk-dequant MMA to chunked FlashAttention.
// At 65536 KV tokens with D=128, bulk f16 = 128 * 65536 * 2 * 2 ≈ 32 MB (K+V, per head-group).
// Above this, chunked saves significant VRAM.
static constexpr int64_t TBQ_CHUNKED_PREFILL_THRESHOLD = 65536;

// TBQ prefill: dequant K/V to f16 via inverse SRHT, then dispatch MMA.
// Simpler than turbo prefill — no Q rotation needed (dequant produces original domain).
static void ggml_cuda_tbq_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool tbq_k = K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0;
    const bool tbq_v = V->type == GGML_TYPE_TBQ2_0 || V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0;

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    if (tbq_k) {
        const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
        CUDA_CHECK(cudaMalloc(&k_fp16, k_size));
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        const size_t smem = K->ne[0] * sizeof(float);
        if (K->type == GGML_TYPE_TBQ2_0) {
            k_tbq2_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TBQ3_0) {
            k_tbq3_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else {
            k_tbq4_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    if (tbq_v) {
        const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
        CUDA_CHECK(cudaMalloc(&v_fp16, v_size));
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        const size_t smem = V->ne[0] * sizeof(float);
        if (V->type == GGML_TYPE_TBQ2_0) {
            k_tbq2_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TBQ3_0) {
            k_tbq3_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else {
            k_tbq4_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
    }

    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * sizeof(half);
        K_f16.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * sizeof(half);
        V_f16.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;

    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // Sync + free immediately to release VRAM before decode starts.
    // cudaFreeAsync keeps memory in the pool, starving decode at long context.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (k_fp16) CUDA_CHECK(cudaFree(k_fp16));
    if (v_fp16) CUDA_CHECK(cudaFree(v_fp16));
}

static void ggml_cuda_turbo_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool turbo_k = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0;
    const bool turbo_v = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    // Allocate and dequant K to fp16 (turbo2, turbo3, or turbo4)
    if (turbo_k) {
        const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
        CUDA_CHECK(cudaMallocAsync(&k_fp16, k_size, stream));
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else {
            // turbo4 K: inverse FWHT dequant — produces K in original domain (no Q rotation needed)
            k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    // Allocate and dequant V to fp16 (turbo2, turbo3, or turbo4)
    if (turbo_v) {
        const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
        CUDA_CHECK(cudaMallocAsync(&v_fp16, v_size, stream));
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else {
            k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
    }

    // Create fp16 tensor copies on stack
    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * sizeof(half);
        K_f16.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * sizeof(half);
        V_f16.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    // Rotate Q for turbo pre-rotate-queries (only when K is in rotated space)
    // turbo4 K is dequanted via inverse FWHT → original domain, so Q stays unrotated
    // Uses persistent per-device buffer to avoid cudaMallocAsync issues with graph-level ops
    const ggml_tensor * Q = dst->src[0];
    float * q_rotated = nullptr;
    if (turbo_k && K->type != GGML_TYPE_TURBO4_0 && Q->ne[0] % 128 == 0) {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        if (q_size > q_rot_buf_size[device]) {
            if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
            CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
            q_rot_buf_size[device] = q_size;
        }
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // Free K/V temporary buffers (stream-ordered, Q uses persistent buffer)
    if (k_fp16) CUDA_CHECK(cudaFreeAsync(k_fp16, stream));
    if (v_fp16) CUDA_CHECK(cudaFreeAsync(v_fp16, stream));
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_TBQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_F16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_TBQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,    GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ2_0,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ3_0,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,  GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TBQ2_0:
        case GGML_TYPE_TBQ3_0:
        case GGML_TYPE_TBQ4_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:

    // TurboQuant/TBQ: only the vec kernel has dequant support.
    if (K->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO2_0 ||
        K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||
        K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0 ||
        K->type == GGML_TYPE_TBQ2_0   || V->type == GGML_TYPE_TBQ2_0   ||
        K->type == GGML_TYPE_TBQ3_0   || V->type == GGML_TYPE_TBQ3_0   ||
        K->type == GGML_TYPE_TBQ4_0   || V->type == GGML_TYPE_TBQ4_0) {
        if (Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Turbo prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // turbo4 K uses inverse FWHT during dequant — mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. turbo2/turbo3 use simple centroid×norm dequant.
    // Set TURBO_PREFILL_VEC=1 to force vec kernel for all turbo types (debug override).
    static const bool turbo_prefill_vec = [] {
        const char * e = getenv("TURBO_PREFILL_VEC");
        if (e) fprintf(stderr, "TURBO_PREFILL_VEC=%s: forcing vec prefill for turbo types\n", e);
        return e != nullptr;
    }();
    const bool turbo_kv = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 ||
                          V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;
    const bool tbq_kv = K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0 ||
                        V->type == GGML_TYPE_TBQ2_0 || V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0;
    // TBQ prefill dispatch:
    //   - Short context (nkv <= threshold): bulk-dequant MMA (fast tensor cores)
    //   - Long context (nkv > threshold): chunked FlashAttention (O(CHUNK) VRAM)
    if (tbq_kv && Q->ne[1] > 1 && turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        if (K->ne[1] > TBQ_CHUNKED_PREFILL_THRESHOLD) {
            ggml_cuda_tbq_chunked_prefill(ctx, dst);
        } else {
            ggml_cuda_tbq_prefill_attend(ctx, dst);
        }
    } else if (turbo_kv && !turbo_prefill_vec && Q->ne[1] > 1 && turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        // Prefill path: turbo4 K uses inverse FWHT dequant (original domain, no Q rotation),
        // turbo2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        ggml_cuda_turbo_prefill_attend(ctx, dst);
    } else {
        cudaStream_t stream = ctx.stream();

        // Dequant turbo3 K/V to fp16 for decode: trades extra memory bandwidth for
        // simpler inner loop (no bit extract + LUT). Eliminates context scaling on MoE,
        // zero cost on dense models. Set GGML_TURBO_DECODE_NATIVE=1 to disable.
        static const bool turbo_decode_native = (getenv("GGML_TURBO_DECODE_NATIVE") != nullptr);
        // TBQ: use native vec_dot (no dequant) to avoid O(context) temp f16 allocation
        // This requires Q pre-rotation via k_tbq_fwht_forward (same as TURBO's approach)
        const bool do_decode_dequant = !turbo_decode_native && turbo_kv;

        half * k_fp16_dec = nullptr;
        half * v_fp16_dec = nullptr;
        ggml_tensor K_f16_dec, V_f16_dec;
        ggml_tensor * orig_k_decode = nullptr;
        ggml_tensor * orig_v_decode = nullptr;

        if (do_decode_dequant) {
            if (K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0) {
                int device; CUDA_CHECK(cudaGetDevice(&device));
                const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
                if (k_size > tbq_k_dec_size[device]) {
                    if (tbq_k_dec_buf[device]) CUDA_CHECK(cudaFree(tbq_k_dec_buf[device]));
                    CUDA_CHECK(cudaMalloc(&tbq_k_dec_buf[device], k_size));
                    tbq_k_dec_size[device] = k_size;
                }
                k_fp16_dec = tbq_k_dec_buf[device];
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                const size_t smem = K->ne[0] * sizeof(float);
                if (K->type == GGML_TYPE_TBQ2_0) {
                    k_tbq2_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TBQ3_0) {
                    k_tbq3_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else {
                    k_tbq4_dequant_f16<<<grid_k, K->ne[0], smem, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * sizeof(half);
                K_f16_dec.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            } else if (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0) {
                const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
                CUDA_CHECK(cudaMallocAsync(&k_fp16_dec, k_size, stream));
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                if (K->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else {
                    k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * sizeof(half);
                K_f16_dec.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            }
            if (V->type == GGML_TYPE_TBQ2_0 || V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0) {
                int device; CUDA_CHECK(cudaGetDevice(&device));
                const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
                if (v_size > tbq_v_dec_size[device]) {
                    if (tbq_v_dec_buf[device]) CUDA_CHECK(cudaFree(tbq_v_dec_buf[device]));
                    CUDA_CHECK(cudaMalloc(&tbq_v_dec_buf[device], v_size));
                    tbq_v_dec_size[device] = v_size;
                }
                v_fp16_dec = tbq_v_dec_buf[device];
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                const size_t smem = V->ne[0] * sizeof(float);
                if (V->type == GGML_TYPE_TBQ2_0) {
                    k_tbq2_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TBQ3_0) {
                    k_tbq3_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else {
                    k_tbq4_dequant_f16<<<grid_v, V->ne[0], smem, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * sizeof(half);
                V_f16_dec.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
            } else if (V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0) {
                const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
                CUDA_CHECK(cudaMallocAsync(&v_fp16_dec, v_size, stream));
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                if (V->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else {
                    k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * sizeof(half);
                V_f16_dec.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
            }
        }

        // Pre-rotate Q for turbo (K/V stored in rotated space, whether turbo3 or dequanted fp16)
        ggml_tensor Q_rot_decode;
        ggml_tensor * orig_q_decode = nullptr;
        const bool turbo_k_any = (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0);
        const bool tbq_k_any = (K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0);
        if ((turbo_k_any || tbq_k_any) && Q->ne[0] % 128 == 0) {
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device]) {
                if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
                q_rot_buf_size[device] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            if (tbq_k_any) {
                // TBQ uses different Rademacher signs than TURBO
                k_tbq_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                    (const float *)Q->data, q_rot_buf[device], ggml_nelements(Q));
            } else {
                k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                    (const float *)Q->data, q_rot_buf[device], ggml_nelements(Q));
            }
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device];
            orig_q_decode = dst->src[0];
            dst->src[0] = &Q_rot_decode;
        }

        switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
            case BEST_FATTN_KERNEL_NONE:
                GGML_ABORT("fatal error");
            case BEST_FATTN_KERNEL_TILE:
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_VEC:
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_WMMA_F16:
                ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_MMA_F16:
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                break;
        }

        if (orig_q_decode) dst->src[0] = orig_q_decode;
        if (orig_k_decode) dst->src[1] = orig_k_decode;
        if (orig_v_decode) dst->src[2] = orig_v_decode;
        // Only free TURBO alloc-per-call buffers; TBQ uses persistent per-device buffers
        const bool tbq_k_used = (K->type == GGML_TYPE_TBQ2_0 || K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0);
        const bool tbq_v_used = (V->type == GGML_TYPE_TBQ2_0 || V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0);
        if (k_fp16_dec && !tbq_k_used) CUDA_CHECK(cudaFreeAsync(k_fp16_dec, stream));
        if (v_fp16_dec && !tbq_v_used) CUDA_CHECK(cudaFreeAsync(v_fp16_dec, stream));
    }

    // Output inverse rotation for turbo V types is handled at graph level
    // (ggml_turbo_wht op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
