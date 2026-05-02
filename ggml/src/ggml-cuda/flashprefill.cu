// FlashPrefill: block-sparse attention for speculative prefill scoring.
// SM80+ BF16 only. Based on Fan et al (2026), re-implemented for buun-llama.

#if !defined(GGML_USE_HIP)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>

#include "flashprefill.cuh"

using namespace nvcuda;

static constexpr int D_HEAD    = 128;
static constexpr int BLOCK_SZ  = 128;

__global__ static void k_compute_mean_K(
        const __nv_bfloat16 * __restrict__ K,
        __nv_bfloat16 * __restrict__ mean_K,
        const int seq_len,
        const int n_kv_heads) {

    const int block_idx = blockIdx.x;
    const int head      = blockIdx.y;
    const int dim       = threadIdx.x;

    const int start = block_idx * BLOCK_SZ;
    const int end   = min(start + BLOCK_SZ, seq_len);
    const int count = end - start;
    if (count <= 0) return;

    float acc = 0.0f;
    for (int t = start; t < end; t++) {
        const __nv_bfloat16 val = K[((size_t)t * n_kv_heads + head) * D_HEAD + dim];
        acc += __bfloat162float(val);
    }
    acc /= (float)count;

    mean_K[((size_t)block_idx * n_kv_heads + head) * D_HEAD + dim] = __float2bfloat16(acc);
}

__global__ static void k_compute_block_scores(
        const __nv_bfloat16 * __restrict__ Q,
        const __nv_bfloat16 * __restrict__ mean_K,
        float * __restrict__ scores,
        const int seq_len,
        const int n_heads,
        const int n_kv_heads,
        const int n_kv_blocks,
        const float scale) {

    const int q_block = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int q_row   = threadIdx.x;

    const int q_pos = q_block * BLOCK_SZ + q_row;
    const int heads_per_kv = n_heads / n_kv_heads;

    extern __shared__ float smem[];

    const int max_kv_block = q_block;

    for (int kv_block = 0; kv_block <= max_kv_block; kv_block++) {
        float max_dot = -FLT_MAX;

        for (int hoff = 0; hoff < heads_per_kv; hoff++) {
            const int q_head = kv_head * heads_per_kv + hoff;
            float dot = 0.0f;

            if (q_pos < seq_len) {
                const __nv_bfloat16 * q_ptr = Q + ((size_t)q_pos * n_heads + q_head) * D_HEAD;
                const __nv_bfloat16 * k_ptr = mean_K + ((size_t)kv_block * n_kv_heads + kv_head) * D_HEAD;

                for (int d = 0; d < D_HEAD; d++) {
                    dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
                }
                dot *= scale;
            }
            max_dot = fmaxf(max_dot, dot);
        }

        smem[q_row] = (q_pos < seq_len) ? max_dot : -FLT_MAX;
        __syncthreads();

        for (int s = BLOCK_SZ / 2; s > 0; s >>= 1) {
            if (q_row < s) {
                smem[q_row] = fmaxf(smem[q_row], smem[q_row + s]);
            }
            __syncthreads();
        }

        if (q_row == 0) {
            scores[((size_t)q_block * n_kv_blocks + kv_block) * n_kv_heads + kv_head] = smem[0];
        }
        __syncthreads();
    }
}

__global__ static void k_block_select(
        const float * __restrict__ scores,
        int * __restrict__ indices,
        int * __restrict__ counts,
        const int n_kv_blocks,
        const int n_q_blocks,
        const int n_kv_heads,
        const int attention_sink,
        const int local_window,
        const int last_n_full,
        const float alpha) {

    const int q_block = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane    = threadIdx.x;

    const int max_kv = q_block + 1;
    const bool is_last = (q_block >= n_q_blocks - last_n_full);

    float local_max = -FLT_MAX;
    for (int n = lane; n < max_kv; n += 32) {
        float s = scores[((size_t)q_block * n_kv_blocks + n) * n_kv_heads + kv_head];
        local_max = fmaxf(local_max, s);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    float thresh = __shfl_sync(0xFFFFFFFF, local_max, 0) * alpha;

    int out_count = 0;
    int * out_ptr = indices + ((size_t)q_block * n_kv_blocks + 0) * n_kv_heads + kv_head;
    const int out_stride = n_kv_heads;

    for (int base = 0; base < max_kv; base += 32) {
        int n = base + lane;
        bool keep = false;

        if (n < max_kv) {
            if (n < attention_sink) {
                keep = true;
            } else if (is_last) {
                keep = true;
            } else if (q_block - n < local_window) {
                keep = true;
            } else {
                float s = scores[((size_t)q_block * n_kv_blocks + n) * n_kv_heads + kv_head];
                keep = (s >= thresh);
            }
        }

        unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
        int n_active = __popc(mask);
        int prefix   = __popc(mask & ((1u << lane) - 1));

        if (keep) {
            out_ptr[(size_t)(out_count + prefix) * out_stride] = n;
        }
        out_count += n_active;
    }

    if (lane == 0) {
        counts[(size_t)q_block * n_kv_heads + kv_head] = out_count;
    }
}

static constexpr int Q_TILE  = 64;
static constexpr int K_TILE  = 64;
static constexpr int WARPS   = 4;
static constexpr int THREADS = WARPS * 32;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int QK_M = Q_TILE / WMMA_M;
static constexpr int QK_N = K_TILE / WMMA_N;
static constexpr int QK_K = D_HEAD / WMMA_K;
static constexpr int PV_K = K_TILE / WMMA_K;

__global__ static void k_sparse_flash_forward(
        const __nv_bfloat16 * __restrict__ Q,
        const __nv_bfloat16 * __restrict__ K,
        const __nv_bfloat16 * __restrict__ V,
        __nv_bfloat16 * __restrict__ O,
        const int * __restrict__ indices,
        const int * __restrict__ sel_counts,
        const int seq_len,
        const int n_heads,
        const int n_kv_heads,
        const int n_kv_blocks,
        const float scale) {

    const int cta_idx  = blockIdx.x;
    const int head     = blockIdx.y;
    const int kv_head  = head / (n_heads / n_kv_heads);
    const int warp_id  = threadIdx.x / 32;
    const int lane_id  = threadIdx.x % 32;

    const int q_start = cta_idx * Q_TILE;
    if (q_start >= seq_len) return;
    const int q_len = min(Q_TILE, seq_len - q_start);

    const int q_block = q_start / BLOCK_SZ;
    const int n_active = sel_counts[(size_t)q_block * n_kv_heads + kv_head];

    extern __shared__ char shared_raw[];
    __nv_bfloat16 * Q_sh = (__nv_bfloat16 *)shared_raw;
    __nv_bfloat16 * KV_sh = Q_sh + Q_TILE * D_HEAD;
    float * S_sh = (float *)(KV_sh + K_TILE * D_HEAD);
    float * row_m = S_sh + Q_TILE * K_TILE;
    float * row_l = row_m + Q_TILE;
    float * O_acc = row_l + Q_TILE;

    for (int i = threadIdx.x; i < Q_TILE; i += THREADS) {
        row_m[i] = -FLT_MAX;
        row_l[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < Q_TILE * D_HEAD; i += THREADS) {
        O_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < Q_TILE * D_HEAD; i += THREADS) {
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        int pos = q_start + row;
        if (pos < seq_len) {
            float val = __bfloat162float(Q[((size_t)pos * n_heads + head) * D_HEAD + col]);
            Q_sh[row * D_HEAD + col] = __float2bfloat16(val * scale);
        } else {
            Q_sh[row * D_HEAD + col] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    for (int ai = 0; ai < n_active; ai++) {
        int kv_block = indices[((size_t)q_block * n_kv_blocks + ai) * n_kv_heads + kv_head];
        int kv_start = kv_block * BLOCK_SZ;

        for (int kt_off = 0; kt_off < BLOCK_SZ; kt_off += K_TILE) {
            int k_start = kv_start + kt_off;
            int k_len = min(K_TILE, seq_len - k_start);
            if (k_len <= 0) continue;

            for (int i = threadIdx.x; i < K_TILE * D_HEAD; i += THREADS) {
                int row = i / D_HEAD;
                int col = i % D_HEAD;
                int pos = k_start + row;
                if (pos < seq_len && row < k_len) {
                    KV_sh[row * D_HEAD + col] = K[((size_t)pos * n_kv_heads + kv_head) * D_HEAD + col];
                } else {
                    KV_sh[row * D_HEAD + col] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();

            // S = Q_sh @ KV_sh^T via WMMA
            {
                const int n_tiles = QK_M * QK_N;
                const int tiles_per_warp = (n_tiles + WARPS - 1) / WARPS;

                for (int ti = 0; ti < tiles_per_warp; ti++) {
                    int tile_idx = warp_id * tiles_per_warp + ti;
                    if (tile_idx >= n_tiles) break;

                    int tm = tile_idx / QK_N;
                    int tn = tile_idx % QK_N;

                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
                    wmma::fill_fragment(acc, 0.0f);

                    for (int tk = 0; tk < QK_K; tk++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b;

                        wmma::load_matrix_sync(a, Q_sh + tm * WMMA_M * D_HEAD + tk * WMMA_K, D_HEAD);
                        wmma::load_matrix_sync(b, KV_sh + tn * WMMA_N * D_HEAD + tk * WMMA_K, D_HEAD);

                        wmma::mma_sync(acc, a, b, acc);
                    }

                    wmma::store_matrix_sync(S_sh + tm * WMMA_M * K_TILE + tn * WMMA_N, acc, K_TILE, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // causal mask
            for (int i = threadIdx.x; i < Q_TILE * K_TILE; i += THREADS) {
                int qi = i / K_TILE;
                int ki = i % K_TILE;
                int q_pos = q_start + qi;
                int k_pos = k_start + ki;
                if (k_pos > q_pos || ki >= k_len || qi >= q_len) {
                    S_sh[qi * K_TILE + ki] = -FLT_MAX;
                }
            }
            __syncthreads();

            // online softmax + rescale O_acc
            for (int qi = threadIdx.x; qi < q_len; qi += THREADS) {
                float m_old = row_m[qi];
                float l_old = row_l[qi];

                float m_new = m_old;
                for (int ki = 0; ki < k_len; ki++) {
                    m_new = fmaxf(m_new, S_sh[qi * K_TILE + ki]);
                }

                float l_new = l_old * expf(m_old - m_new);
                for (int ki = 0; ki < k_len; ki++) {
                    float p = expf(S_sh[qi * K_TILE + ki] - m_new);
                    S_sh[qi * K_TILE + ki] = p;
                    l_new += p;
                }

                float rescale = expf(m_old - m_new);
                for (int d = 0; d < D_HEAD; d++) {
                    O_acc[qi * D_HEAD + d] *= rescale;
                }

                row_m[qi] = m_new;
                row_l[qi] = l_new;
            }
            __syncthreads();

            // O += P @ V (scalar v1 — WMMA P@V deferred to optimization phase)
            for (int i = threadIdx.x; i < K_TILE * D_HEAD; i += THREADS) {
                int row = i / D_HEAD;
                int col = i % D_HEAD;
                int pos = k_start + row;
                if (pos < seq_len && row < k_len) {
                    KV_sh[row * D_HEAD + col] = V[((size_t)pos * n_kv_heads + kv_head) * D_HEAD + col];
                } else {
                    KV_sh[row * D_HEAD + col] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();

            for (int qi = threadIdx.x; qi < q_len; qi += THREADS) {
                for (int ki = 0; ki < k_len; ki++) {
                    float p = S_sh[qi * K_TILE + ki];
                    if (p > 0.0f) {
                        for (int d = 0; d < D_HEAD; d++) {
                            O_acc[qi * D_HEAD + d] += p * __bfloat162float(KV_sh[ki * D_HEAD + d]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < q_len * D_HEAD; i += THREADS) {
        int qi = i / D_HEAD;
        int d  = i % D_HEAD;
        float val = O_acc[qi * D_HEAD + d] / row_l[qi];
        int pos = q_start + qi;
        O[((size_t)pos * n_heads + head) * D_HEAD + d] = __float2bfloat16(val);
    }
}

FlashPrefillBuffers flash_prefill_alloc(int seq_len, int n_heads, int n_kv_heads, int block_size) {
    FlashPrefillBuffers bufs = {};
    const int n_blocks = (seq_len + block_size - 1) / block_size;

    cudaMalloc(&bufs.mean_K,   (size_t)n_blocks * n_kv_heads * D_HEAD * sizeof(__nv_bfloat16));
    cudaMalloc(&bufs.scores,   (size_t)n_blocks * n_blocks * n_kv_heads * sizeof(float));
    cudaMalloc(&bufs.indices,  (size_t)n_blocks * n_blocks * n_kv_heads * sizeof(int));
    cudaMalloc(&bufs.counts,   (size_t)n_blocks * n_kv_heads * sizeof(int));

    return bufs;
}

void flash_prefill_free(FlashPrefillBuffers * bufs) {
    if (bufs->mean_K)    { cudaFree(bufs->mean_K);    bufs->mean_K    = nullptr; }
    if (bufs->scores)    { cudaFree(bufs->scores);     bufs->scores    = nullptr; }
    if (bufs->indices)   { cudaFree(bufs->indices);    bufs->indices   = nullptr; }
    if (bufs->counts)    { cudaFree(bufs->counts);     bufs->counts    = nullptr; }
}

void flash_prefill_forward_bf16(
        const void * Q, const void * K, const void * V,
        void * O,
        int seq_len, int n_heads, int n_kv_heads, int d_head,
        float scale,
        const FlashPrefillConfig * cfg,
        FlashPrefillBuffers * bufs,
        void * stream_ptr) {

    (void)d_head;
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    const int n_blocks  = (seq_len + cfg->block_size - 1) / cfg->block_size;

    {
        dim3 grid(n_blocks, n_kv_heads);
        dim3 block(D_HEAD);
        k_compute_mean_K<<<grid, block, 0, stream>>>(
            (const __nv_bfloat16 *)K,
            (__nv_bfloat16 *)bufs->mean_K,
            seq_len, n_kv_heads);
    }

    {
        dim3 grid(n_blocks, n_kv_heads);
        dim3 block(BLOCK_SZ);
        size_t smem = BLOCK_SZ * sizeof(float);
        k_compute_block_scores<<<grid, block, smem, stream>>>(
            (const __nv_bfloat16 *)Q,
            (const __nv_bfloat16 *)bufs->mean_K,
            (float *)bufs->scores,
            seq_len, n_heads, n_kv_heads, n_blocks, scale);
    }

    {
        dim3 grid(n_blocks, n_kv_heads);
        dim3 block(32);
        k_block_select<<<grid, block, 0, stream>>>(
            (const float *)bufs->scores, bufs->indices, bufs->counts,
            n_blocks, n_blocks, n_kv_heads,
            cfg->attention_sink, cfg->local_window, cfg->last_n_full, cfg->alpha);
    }

    {
        const int n_q_tiles = (seq_len + Q_TILE - 1) / Q_TILE;
        dim3 grid(n_q_tiles, n_heads);
        dim3 block(THREADS);

        size_t smem = Q_TILE * D_HEAD * sizeof(__nv_bfloat16)
                    + K_TILE * D_HEAD * sizeof(__nv_bfloat16)
                    + Q_TILE * K_TILE * sizeof(float)
                    + Q_TILE * sizeof(float)
                    + Q_TILE * sizeof(float)
                    + Q_TILE * D_HEAD * sizeof(float);

        k_sparse_flash_forward<<<grid, block, smem, stream>>>(
            (const __nv_bfloat16 *)Q,
            (const __nv_bfloat16 *)K,
            (const __nv_bfloat16 *)V,
            (__nv_bfloat16 *)O,
            bufs->indices, bufs->counts,
            seq_len, n_heads, n_kv_heads, n_blocks, scale);
    }
}

#endif // !GGML_USE_HIP
