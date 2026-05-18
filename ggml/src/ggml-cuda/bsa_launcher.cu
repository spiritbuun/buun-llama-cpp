// BSA (Block-Sparse-Attention) launcher for PFlash scorer.
// Adapts our FlashPrefill block selection → BSA blockmask format,
// then dispatches run_mha_fwd_block_<bf16, 128, false>.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "namespace_config.h"
#include "flash.h"

#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_block_(Flash_fwd_params &params, cudaStream_t stream);
}

namespace {

constexpr int kHeadDim   = 128;
constexpr int kBlockSize = 128;
constexpr float kLog2E   = 1.4426950408889634f;

struct BsaCache {
    int32_t* blockmask       = nullptr;
    size_t   blockmask_bytes = 0;
    int*     head_mask_type     = nullptr;
    int      head_mask_capacity = 0;
    int      head_mask_state    = -1;
    float*   softmax_lse       = nullptr;
    size_t   softmax_lse_bytes = 0;

    cudaError_t ensure_blockmask(size_t bytes) {
        if (bytes <= blockmask_bytes) return cudaSuccess;
        if (blockmask) cudaFree(blockmask);
        cudaError_t e = cudaMalloc(&blockmask, bytes);
        blockmask_bytes = (e == cudaSuccess) ? bytes : 0;
        return e;
    }
    cudaError_t ensure_head_mask(int n) {
        if (n <= head_mask_capacity) return cudaSuccess;
        if (head_mask_type) cudaFree(head_mask_type);
        cudaError_t e = cudaMalloc(&head_mask_type, n * sizeof(int));
        head_mask_capacity = (e == cudaSuccess) ? n : 0;
        head_mask_state = -1;
        return e;
    }
    cudaError_t ensure_lse(size_t bytes) {
        if (bytes <= softmax_lse_bytes) return cudaSuccess;
        if (softmax_lse) cudaFree(softmax_lse);
        cudaError_t e = cudaMalloc(&softmax_lse, bytes);
        softmax_lse_bytes = (e == cudaSuccess) ? bytes : 0;
        return e;
    }
    void release() {
        if (blockmask)      { cudaFree(blockmask);      blockmask = nullptr;      blockmask_bytes = 0; }
        if (head_mask_type) { cudaFree(head_mask_type); head_mask_type = nullptr; head_mask_capacity = 0; head_mask_state = -1; }
        if (softmax_lse)    { cudaFree(softmax_lse);    softmax_lse = nullptr;    softmax_lse_bytes = 0; }
    }
};

static BsaCache & cache() { static BsaCache c; return c; }

// Convert our indices[M, N, H_kv] + counts[M, H_kv] to BSA blockmask[H_kv, Mp, Np]
// BSA expects descending-sorted valid indices followed by -1 sentinels.
__global__ void k_convert_to_blockmask(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ blockmask_out,
    int M, int N, int H_kv, int Np)
{
    const int m = blockIdx.x;
    const int h = blockIdx.y;
    if (m >= M || h >= H_kv) return;

    // output row for this (h, m)
    int32_t* outp = blockmask_out + ((size_t)h * M + m) * Np;

    int cnt = counts[(size_t)m * H_kv + h];
    if (cnt >= Np) cnt = Np - 1;

    // indices layout: [M, N, H_kv] => idx[m][n][h] = indices[m*N*H_kv + n*H_kv + h]
    const int32_t* inp = indices + (size_t)m * N * H_kv + h;

    for (int n = threadIdx.x; n < Np; n += blockDim.x) {
        if (n < cnt) {
            // reverse: BSA wants descending
            int rev = cnt - 1 - n;
            outp[n] = inp[(size_t)rev * H_kv];
        } else {
            outp[n] = -1;
        }
    }
}

}  // anon namespace

extern "C" void pflash_bsa_free_persistent() {
    cache().release();
}

extern "C" int pflash_bsa_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int seq_len, int n_q_heads, int n_kv_heads,
    cudaStream_t stream)
{
    const int M  = (seq_len + kBlockSize - 1) / kBlockSize;
    const int Np = M;

    BsaCache & c = cache();

    // blockmask: [n_kv_heads, M, Np] (one mask per KV head, shared by its Q heads)
    const size_t bm_bytes  = (size_t)n_kv_heads * M * Np * sizeof(int32_t);
    const size_t lse_bytes = (size_t)n_q_heads * ((size_t)M * kBlockSize) * sizeof(float);

    cudaError_t err;
    if ((err = c.ensure_blockmask(bm_bytes)) != cudaSuccess) goto fail;
    if ((err = c.ensure_head_mask(n_q_heads)) != cudaSuccess) goto fail;
    if ((err = c.ensure_lse(lse_bytes)) != cudaSuccess) goto fail;

    // head_mask_type: map each Q head to its KV head mask (1-indexed)
    // Q heads [0,1] → KV head 0 → mask_type=1, Q heads [2,3] → KV head 1 → mask_type=2, etc.
    if (c.head_mask_state != n_q_heads) {
        int ratio = n_q_heads / n_kv_heads;
        int* h_hmt = (int*)malloc(n_q_heads * sizeof(int));
        for (int h = 0; h < n_q_heads; ++h) h_hmt[h] = (h / ratio) + 1;
        cudaMemcpyAsync(c.head_mask_type, h_hmt, n_q_heads * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        free(h_hmt);
        c.head_mask_state = n_q_heads;
    }

    // convert our indices → BSA blockmask
    {
        dim3 grid(M, n_kv_heads);
        dim3 block(64);
        k_convert_to_blockmask<<<grid, block, 0, stream>>>(
            indices, counts, c.blockmask, M, M, n_kv_heads, Np);
    }

    // fill Flash_fwd_params and dispatch
    {
        FLASH_NAMESPACE::Flash_fwd_params params{};
        params.q_ptr = const_cast<void*>(Q);
        params.k_ptr = const_cast<void*>(K);
        params.v_ptr = const_cast<void*>(V);
        params.o_ptr = O;

        // Layout: [seq_len, n_heads, head_dim] (interleaved heads)
        params.q_batch_stride = (int64_t)seq_len * n_q_heads * kHeadDim;
        params.q_row_stride   = (int64_t)n_q_heads * kHeadDim;
        params.q_head_stride  = kHeadDim;

        params.k_batch_stride = (int64_t)seq_len * n_kv_heads * kHeadDim;
        params.k_row_stride   = (int64_t)n_kv_heads * kHeadDim;
        params.k_head_stride  = kHeadDim;

        params.v_batch_stride = (int64_t)seq_len * n_kv_heads * kHeadDim;
        params.v_row_stride   = (int64_t)n_kv_heads * kHeadDim;
        params.v_head_stride  = kHeadDim;

        params.o_batch_stride = (int64_t)seq_len * n_q_heads * kHeadDim;
        params.o_row_stride   = (int64_t)n_q_heads * kHeadDim;
        params.o_head_stride  = kHeadDim;

        params.h           = n_q_heads;
        params.h_k         = n_kv_heads;
        params.h_h_k_ratio = n_q_heads / n_kv_heads;

        params.b               = 1;
        params.seqlen_q        = seq_len;
        params.seqlen_k        = seq_len;
        params.d               = kHeadDim;
        params.seqlen_q_rounded = M * kBlockSize;
        params.seqlen_k_rounded = M * kBlockSize;
        params.d_rounded       = kHeadDim;

        params.scale_softmax      = scale;
        params.scale_softmax_log2 = scale * kLog2E;

        params.cu_seqlens_q   = nullptr;
        params.cu_seqlens_k   = nullptr;
        params.seqused_k      = nullptr;
        params.softmax_lse_ptr = c.softmax_lse;
        params.p_ptr          = nullptr;

        params.blockmask             = c.blockmask;
        params.streaming_info        = nullptr;
        params.head_mask_type        = c.head_mask_type;
        params.m_block_dim           = kBlockSize;
        params.n_block_dim           = kBlockSize;
        params.num_blocksparse_heads = n_kv_heads;

        params.window_size_left      = -1;
        params.window_size_right     = -1;
        params.is_bf16               = true;
        params.is_causal             = false;
        params.is_exact_streaming    = false;
        params.is_seqlens_k_cumulative = false;
        params.is_rotary_interleaved = false;
        params.num_splits            = 1;
        params.alibi_slopes_ptr      = nullptr;
        params.alibi_slopes_batch_stride = 0;
        params.p_dropout             = 1.f;

        FLASH_NAMESPACE::run_mha_fwd_block_<cutlass::bfloat16_t,
                                            kHeadDim,
                                            /*Is_causal=*/false>(params, stream);
    }

    return 0;

fail:
    fprintf(stderr, "[bsa] cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return -1;
}
