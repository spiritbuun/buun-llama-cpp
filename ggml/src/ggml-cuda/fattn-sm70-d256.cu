// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// sm70-attn plugin — commit B (path A): real SM70 D256 Split-D kernel.
//
// The device kernel is the 1CatAI-verified Split-D N32 flash attention
// (provenance in fattn-sm70-d256-kernel.cuh; 1Cat-vLLM v1.3.0). The
// verified core (smem layouts / HMMA.884 atoms / K-V pipeline / online
// softmax / causal mask) is byte-identical to upstream. Kernel edits:
// one extra `kv_offset` parameter (causal boundary for padded Q) + the
// Mask construction using it. All adaptation lives in this file.
//
// Design record: p1b-design-final.md. Key points:
//   * Scale: stock pre-multiplies Q by `scale` (natural-log domain); the
//     1Cat kernel keeps Q unscaled and folds the scale into exp2 via
//     softmax_scale_log2 = scale*log2(e). Mathematically identical.
//   * Q: staged f32->f16 into a 64-row-padded scratch (the kernel's tiled
//     Q copy is unguarded, so the last partial Q tile MUST be padded; pad
//     rows are zero and their outputs are never written back by the scatter).
//   * K/V: read directly from the stock f16 buffers — native f16 cache, or
//     the stock f16 dequant extra (this launcher runs the same to_fp16
//     dequant the stock launch_fattn does). No K/V staging: the kernel's
//     causal n_block_max bound + causal mask guarantee no read beyond
//     kv_len (only the causal diagonal block is read unguarded, and all
//     its columns are < kv_len), and the f16 buffers are physically
//     larger than kv_len*256.
//   * Causal: derived from positions (col > kv_offset + row); kv_offset =
//     kv_len - q_len passed explicitly (padded Q would break the kernel's
//     own derivation). No mask tensor consumed.
//   * GQA: kernel grid.z = hkv*gqa (head_q = j*gqa + c); the kernel maps
//     head_q -> head_kv. Q staging/scatter map head_q -> (b, j, c):
//     c = Q head within the KV group (Q data depends only on c),
//     j = KV head (K/V select), b = sequence.
//   * Scratch layout: [hkv][gqa][nb][rows][256] f16, slice(j,c) at
//     (j*gqa + c) * nb * rows * 256; seq b at b * rows * 256 inside.
//   * Output (8/23 review): the kernel writes f32 directly into Os
//     (ElementOut=float; was: f16 staging + f16->f32 scatter). The f16
//     staging was the largest sm70-side per-layer rounding source.
//   * Rollback: env LLAMA_SM70_D256=0 forces the stock path.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-sm70-d256-kernel.cuh"
#include <cstdio>
#include <vector>

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634f
#endif

// NOTE: no anonymous namespace in this file. The CuTe vendor headers
// (cute/atom/mma_traits_sm70.hpp) open their own anonymous namespaces;
// a second anonymous namespace in this TU makes cudafe's
// _GLOBAL__N__<hash> symbol mangling ambiguous ("reference to
// '_GLOBAL__N__...' is ambiguous"). Helpers below are therefore plain
// statics / file-scope constants.

constexpr int SM70_D256_BLOCK_M = 64;
constexpr int SM70_D256_D = 256;

// Q f32 -> f16 staging.
// grid = (q_pad, nb*hkv, gqa); block = 128 (float2 grain over the 256 row).
// dst (Qs) layout: [hkv][gqa][nb][q_pad][256] f16.
__global__ void sm70_d256_stage_q_kernel(
        const float2 * __restrict__ src, half2 * __restrict__ dst,
        const int q_len, const int hkv, const int gqa, const int nb,
        const int64_t src_row, const int64_t src_head, const int64_t src_seq) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;          // b*hkv + j
    const int c  = blockIdx.z;          // Q head within the KV group
    if (r >= q_len) {
        return;                         // pad rows: pre-zeroed scratch
    }
    const int j = bj % hkv;
    const int b = bj / hkv;
    const int head_q = j * gqa + c;     // GLOBAL Q head index (0..heads_q-1)
    const int heads_q = hkv * gqa;      // 24
    // Q source layout is (D, q_len, heads, batch): head index = head_q, NOT c.
    const float2 v = src[threadIdx.x
                   + (int64_t) r * src_row
                   + (int64_t) head_q * src_head
                   + (int64_t) b * src_seq];
    // Qs layout (kernel reads [b][head_q][q_pad][D]): batch-major.
    dst[threadIdx.x
      + (int64_t) b * heads_q * (gridDim.x * 128)
      + (int64_t) head_q * (gridDim.x * 128)
      + (int64_t) r * 128] = __float22half2_rn(v);
}

// Output scatter: staged [hkv][gqa][nb][rows][256] f32 -> stock f32 dst.
// grid = (q_len, nb*hkv, gqa); block = 128.
__global__ void sm70_d256_scatter_kernel(
        const float2 * __restrict__ src, float2 * __restrict__ dst,
        const int hkv, const int gqa, const int nb,
        const int q_pad,
        const int64_t dst_row, const int64_t dst_head, const int64_t dst_seq) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;          // b*hkv + j
    const int c  = blockIdx.z;          // Q head within the KV group
    const int j = bj % hkv;
    const int b = bj / hkv;
    const int head_q = j * gqa + c;     // GLOBAL Q head index (0..heads_q-1)
    const int heads_q = hkv * gqa;      // 24
    const int d2 = SM70_D256_D / 2;     // 128 half2 per head
    // Os layout (kernel-hardcoded): [batch][row][head_q][D]; batch stride uses q_pad.
    const float2 v = src[threadIdx.x
        + (int64_t) b * (int64_t) q_pad * heads_q * d2
        + (int64_t) r * (heads_q * d2)
        + (int64_t) head_q * d2];
    // dst (f32 Q layout (D, q_len, heads, batch)): row + global head_q + seq.
    dst[threadIdx.x
      + (int64_t) r * dst_row
      + (int64_t) head_q * dst_head
      + (int64_t) b * dst_seq] = v;
}

// dequant K/V (q4_0 / f32) into the stock f16 extra buffers — same code
// path (to_fp16 / to_fp16_nc) as the stock launch_fattn. Skipped per-tensor
// when q4-direct serves that tensor from the raw blocks in-kernel.
static void sm70_d256_dequant_kv(
        ggml_tensor * K, ggml_tensor * V,
        const ggml_cuda_flash_attn_ext_f16_extra_data & f16_extra,
        const bool V_is_K_view, cudaStream_t stream,
        const bool k_direct, const bool v_direct) {
    if (!k_direct && K->type != GGML_TYPE_F16) {
        const char * K_data = (const char *) K->data;
        half * K_f16 = (half *) f16_extra.K;
        GGML_ASSERT(f16_extra.K != 0);
        if (ggml_is_contiguously_allocated(K)) {
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), stream);
        } else {
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                    K->nb[1] / ts, K->nb[2] / ts, K->nb[3] / ts, stream);
        }
    }
    if (!v_direct && !V_is_K_view && V->type != GGML_TYPE_F16) {
        const char * V_data = (const char *) V->data;
        half * V_f16 = (half *) f16_extra.V;
        GGML_ASSERT(f16_extra.V != 0);
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V_data, V_f16, ggml_nelements(V), stream);
        } else {
            const size_t ts = ggml_type_size(V->type);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                    V->nb[1] / ts, V->nb[2] / ts, V->nb[3] / ts, stream);
        }
    }
}

static bool sm70_env_disabled() {
    static const bool disabled = [] {
        const char * e = getenv("LLAMA_SM70_D256");
        return e && e[0] == '0';
    }();
    return disabled;
}

// q4-direct (8/23, from the 1Cat XQA <KV_DTYPE> load architecture): q4_0 K/V
// blocks are dequantized IN-KERNEL instead of staging the whole cache to an
// f16 mirror per call. Kernel rounding is bit-identical to the staged path
// (exact f32 product + one RN; 23/23 harness, logit A/B verified).
// MEASURED (8/24, V100, Qwen3.8-27B q4_0 K + f16 V): prefill is 3.6% SLOWER
// at 176k (457 vs 474 tok/s) — the narrow u16 block loads cost more than the
// dequant pass they replace (which is only ~0.1s of a 770s prefill; the
// original "O(n^2) dequant disaster" estimate was a units error, see the
// 8/24 study erratum). The win is MEMORY: no f16 mirror (~470MB at -c 229k).
// Default OFF; set LLAMA_SM70_D256_Q4_DIRECT=1 to opt in (VRAM-tight setups).
static bool sm70_q4_direct() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SM70_D256_Q4_DIRECT");
        return e && e[0] == '1';
    }();
    return enabled;
}

// 8/23 cause hunt: capture the FIRST sm70 invocation's actual kernel inputs
// (dequantized K, raw V cache, staged Q) for offline minimal reproduction of
// the 0.69 real-input divergence. Env: SM70_DUMP_KV=<path>. Fires once per
// process (the first ACCEPT call = first full-attn layer of the first chunk,
// whose inputs are bit-identical across ON/OFF since only linear-attention
// layers precede it). File layout (little-endian):
//   u32 magic=0x51444B53, u32 version=1,
//   u32 kv_len, u32 q_len, u32 q_pad, u32 hkv, u32 heads_q,
//   i64 k_row_stride, i64 k_head_stride, i64 v_row_stride, i64 v_head_stride,
//   u64 k_count, u64 v_count, u64 q_count   (half element counts)
//   K blob (f16) | V blob (f16, raw cache memory incl. pos-major layout) | Q blob (f16)
static void sm70_dump_kernel_inputs(
        const half * K_h2, int64_t k_row_stride, int64_t k_head_stride,
        const half * V_h2, int64_t v_row_stride, int64_t v_head_stride,
        const half * Qs, int q_pad,
        int kv_len, int q_len, int hkv, int heads_q) {
    static const char * path = getenv("SM70_DUMP_KV");
    static bool done = false;
    if (!path || done) {
        return;
    }
    done = true;
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[sm70-dump-kv] cannot open %s for writing\n", path);
        return;
    }
    // dump spans: enough raw halves to cover every (head, ctx) pair under the
    // actual strides (pos-major V: kv_len*v_row_stride covers all heads).
    const int64_t k_span = (int64_t) hkv * k_head_stride;      // contiguous dequant
    const int64_t v_span = v_row_stride >= v_head_stride
        ? (int64_t) kv_len * v_row_stride
        : (int64_t) hkv * v_head_stride;
    const int64_t q_count = (int64_t) heads_q * q_pad * 256;
    const uint32_t hdr[7] = {
        0x51444B53u, 1u,
        (uint32_t) kv_len, (uint32_t) q_len, (uint32_t) q_pad,
        (uint32_t) hkv, (uint32_t) heads_q,
    };
    const int64_t strides[4] = { k_row_stride, k_head_stride, v_row_stride, v_head_stride };
    const uint64_t counts[3] = { (uint64_t) k_span, (uint64_t) v_span, (uint64_t) q_count };
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(strides, 1, sizeof(strides), f);
    fwrite(counts, 1, sizeof(counts), f);
    // K/Q live in this stream's allocations; sync before reading from host.
    // (called before the kernel launch, after dequant+staging kernels)
    std::vector<half> tmp;
    tmp.resize((size_t) (k_span > v_span ? k_span : v_span));
    cudaMemcpy(tmp.data(), K_h2, sizeof(half) * k_span, cudaMemcpyDeviceToHost);
    fwrite(tmp.data(), sizeof(half), k_span, f);
    tmp.resize((size_t) v_span);
    cudaMemcpy(tmp.data(), V_h2, sizeof(half) * v_span, cudaMemcpyDeviceToHost);
    fwrite(tmp.data(), sizeof(half), v_span, f);
    tmp.resize((size_t) q_count);
    cudaMemcpy(tmp.data(), Qs, sizeof(half) * q_count, cudaMemcpyDeviceToHost);
    fwrite(tmp.data(), sizeof(half), q_count, f);
    fclose(f);
    fprintf(stderr, "[sm70-dump-kv] wrote %s (k=%lld v=%lld q=%lld halves)\n",
            path, (long long) k_span, (long long) v_span, (long long) q_count);
}

// Routing probe: prints why a decision was made. Default = first decision
// only; set LLAMA_SM70_D256_DEBUG=1 for every call.
static void sm70_d256_probe(const char * reason, int cc,
                            const ggml_tensor * Q, const ggml_tensor * K,
                            const ggml_tensor * V, const ggml_tensor * mask) {
    static const bool verbose = getenv("LLAMA_SM70_D256_DEBUG") != nullptr;
    // Three classes of dispatch:
    //  * template: graph-build placeholders (any q_len incl. the 512-batch
    //    template; KV/mask views carry the FULL cache capacity)
    //  * real: actual prefill chunk (q_len >= 256, KV view sliced to the
    //    current kv_len < capacity)
    //  * ph: everything else (decode, MTP, small templates)
    // NB (8/19 46k post-mortem #2): K->ne[1] is the VIEW size, not the
    // capacity - real chunks are sliced to kv_len, so mask->ne[0] < K->ne[1]
    // is never true; and the build-time 512-batch template has q_len=512
    // with the full view, so q_len alone cannot separate it from real
    // chunks (it consumed real#1-5 with Mkv==capacity). Capacity = the
    // largest K->ne[1] ever observed: load templates (q=1/16/3/512) all
    // carry the full view and are built before any request arrives.
    // Edge: a chunk that actually fills the cache (kv_len == capacity)
    // is mislabelled ph - probe only, no functional impact.
    // NOTE (8/20 post-mortem): draft-model templates (head_dim != 256,
    // q_len = 512) were mislabelled REAL and consumed the 5 real# budget
    // slots at load, permanently silencing the probe for target-model
    // prefill afterwards. Draft dispatch is uninteresting for this
    // probe (the kernel only ever runs for head_dim == 256), so classify
    // those as ph.
    static int64_t kv_max_seen = 0;
    const bool real = Q->ne[0] == SM70_D256_D && Q->ne[1] >= 256
                      && kv_max_seen > 0 && K->ne[1] < kv_max_seen;
    if (K->ne[1] > kv_max_seen) { kv_max_seen = K->ne[1]; }
    static int printed_ph = 0;
    static int printed_real = 0;
    if (!verbose && (real ? printed_real >= 5 : printed_ph >= 20)) {
        return;
    }
    if (real) { printed_real++; } else { printed_ph++; }
    int n = (real ? printed_real : printed_ph);
    fprintf(stderr, "[sm70-d256] %s#%d %s | cc=%d Q=(%lld,%lld,%lld,%lld) Qtype=%d "
            "K=(%lld,%lld,%lld,%lld) Ktype=%d Knb0=%llu Knb1=%llu Knb2=%llu rowK=%llu "
            "Vtype=%d Vnb0=%llu Vnb1=%llu Vnb2=%llu rowV=%llu Mkv=%lld mask=%p\n",
            (real ? "real" : "ph"), n, reason, cc,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3], (int) Q->type,
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3], (int) K->type,
            (unsigned long long) K->nb[0], (unsigned long long) K->nb[1], (unsigned long long) K->nb[2], (unsigned long long) ggml_row_size(K->type, K->ne[0]),
            (int) V->type, (unsigned long long) V->nb[0], (unsigned long long) V->nb[1], (unsigned long long) V->nb[2], (unsigned long long) ggml_row_size(V->type, V->ne[0]),
            mask ? (long long) mask->ne[0] : -1LL, (const void *) mask);
}

// ------------------------------------------------------------------- public
bool ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA || sm70_env_disabled()) {
        // probe only when cc is volta (the interesting case for the probe)
        if (cc == GGML_CUDA_CC_VOLTA) {
            const ggml_tensor * Qp = dst->src[0];
            const ggml_tensor * Kp = dst->src[1];
            const ggml_tensor * Vp = dst->src[2];
            const ggml_tensor * Mp = dst->src[3];
            sm70_d256_probe(cc != GGML_CUDA_CC_VOLTA ? "REJECT: cc!=volta" : "REJECT: env disabled", cc, Qp, Kp, Vp, Mp);
        }
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q->ne[0] != SM70_D256_D || K->ne[0] != SM70_D256_D || V->ne[0] != SM70_D256_D) {
        sm70_d256_probe("REJECT: head_dim != 256", cc, Q, K, V, mask);
        return false;
    }
    if (!mask || mask->ne[0] < 256 || Q->ne[1] < 256) { // prefill only; decode/MTP/small batches -> stock
        sm70_d256_probe("REJECT: no mask or small batch", cc, Q, K, V, mask);
        return false;
    }
    if (Q->ne[1] > mask->ne[0]) { // kv_len (mask->ne[0]) must cover q_len
        sm70_d256_probe("REJECT: kv_len < q_len", cc, Q, K, V, mask);
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        sm70_d256_probe("REJECT: gqa ratio", cc, Q, K, V, mask);
        return false;
    }
    // NB (8/23): F32 K/V REJECTED — the launcher only implements F16 (direct)
    // and Q4_0 (dequant) branches; an F32 tensor would hit GGML_ABORT.
    const bool kv_ok = (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0)
                    && (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q4_0);
    if (!kv_ok) {
        sm70_d256_probe("REJECT: kv type", cc, Q, K, V, mask);
        return false;
    }
    // NB (route A post-mortem, 8/19): ggml nb[0] = bytes per ELEMENT (2 for F16);
    // the KV cache is laid out [ctx][head][dim] (dim fastest, nb[0] contiguous),
    // so rows are NOT contiguous - nb[1] (ctx stride) >> row size.
    // All the kernel needs for the f16-direct path is per-row contiguity
    // (nb[0] == elem size); head/ctx access goes through explicit strides.
    // The dequant path (to_fp16_nc) handles ANY source strides.
    if (K->type == GGML_TYPE_F16 && K->nb[0] != sizeof(half)) {
        sm70_d256_probe("REJECT: K rows not contiguous", cc, Q, K, V, mask);
        return false;
    }
    if (V->type == GGML_TYPE_F16 && V->nb[0] != sizeof(half)) {
        sm70_d256_probe("REJECT: V rows not contiguous", cc, Q, K, V, mask);
        return false;
    }
    // q4-direct: raw block reads require block-contiguous rows (nb[0] == 18).
    // Real caches always satisfy this; exotic strided views fall to stock.
    if (sm70_q4_direct()) {
        const size_t q4_blk = ggml_type_size(GGML_TYPE_Q4_0);
        if ((K->type == GGML_TYPE_Q4_0 && K->nb[0] != q4_blk)
         || (V->type == GGML_TYPE_Q4_0 && V->nb[0] != q4_blk)) {
            sm70_d256_probe("REJECT: q4 rows not block-contiguous", cc, Q, K, V, mask);
            return false;
        }
    }
    sm70_d256_probe(
        (K->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q4_0) && sm70_q4_direct()
            ? "ACCEPT: sm70 d256 + q4-direct"
            : "ACCEPT: sm70 d256 kernel selected",
        cc, Q, K, V, mask);
    return true;
}

// scratch (all carved from the get_alloc_size extra region after dst->data,
// the stock f16_extra model): [K dequant][V dequant] (f16_extra layout) +
// Qs (padded f16 Q) + Os (f16 output staging).
size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    // q4-direct: raw in-kernel reads need no f16 mirror for that tensor.
    // MUST stay consistent with the launcher's k_direct/v_direct predicate
    // (same env, same types) or the launcher would deref a null f16_extra.
    const bool k_direct = K->type == GGML_TYPE_Q4_0 && sm70_q4_direct();
    const bool v_direct = V->type == GGML_TYPE_Q4_0 && sm70_q4_direct();
    const bool need_f16_K = K->type != GGML_TYPE_F16 && !k_direct;
    // MUST match the launcher's need_f16_V exactly (edge case: V is a view of
    // a non-f16 K — the launcher still dequants V, so the alloc must cover it).
    const bool need_f16_V = !(V_is_K_view && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16 && !v_direct;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);
    // f16_extra size = bytes beyond the output data (base = dst->data + nbytes)
    const size_t f16_extra_size = (size_t) (f16_extra.end - ((uintptr_t) dst->data + ggml_nbytes(dst)));

    const int q_pad = (((int) Q->ne[1] + SM70_D256_BLOCK_M - 1) / SM70_D256_BLOCK_M) * SM70_D256_BLOCK_M;
    const int64_t nQ = (int64_t) Q->ne[2] * q_pad * SM70_D256_D * (int) Q->ne[3]; // elems
    // SplitKV3 partials (worst case): 3 x (q_pad*heads*nb) rows of f32 D
    // plus per-row max/sum stats, all f32.
    const int64_t rows3 = (int64_t) q_pad * Q->ne[2] * Q->ne[3];
    // total allocation = output + PAD(f16_extra, 128) + Qs (f16) + Os (f32)
    //                   + SplitKV3 partial_out/max/sum
    return ggml_nbytes(dst) + GGML_PAD(f16_extra_size, 128)
         + (size_t) nQ * sizeof(half) + (size_t) nQ * sizeof(float)
         + GGML_PAD((size_t) 3 * rows3 * SM70_D256_D * sizeof(float)
                    + 2 * (size_t) 3 * rows3 * sizeof(float), 128);
}

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    GGML_ASSERT(cc == GGML_CUDA_CC_VOLTA);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int hkv    = (int) K->ne[2];
    const int gqa    = (int) (Q->ne[2] / K->ne[2]);
    const int q_len  = (int) Q->ne[1];
    const ggml_tensor * mask = dst->src[3];
    // REAL KV length is mask->ne[0] (mask built as [n_kv, q_len]); K->ne[1] is the FULL cache size.
    const int kv_len = (int) mask->ne[0];
    const int nb     = (int) Q->ne[3];
    GGML_ASSERT(mask != nullptr);
    // NOTE (8/19 crash post-mortem): K and V types are legitimately
    // different (-ctk q4_0 -ctv f16): the dequant path handles each
    // independently. Do NOT assert K->type == V->type.
    GGML_ASSERT(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q4_0);

    const int q_pad = ((q_len + SM70_D256_BLOCK_M - 1) / SM70_D256_BLOCK_M) * SM70_D256_BLOCK_M;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(logit_softcap == 0.0f);
    const float softmax_scale_log2 = scale * M_LOG2E;
    const int kv_offset = kv_len - q_len;
    const int64_t nQ = (int64_t) hkv * gqa * nb * q_pad * SM70_D256_D; // Qs elems (f16); Os same count in f32

    // ------------------------------------- scratch layout (extra region)
    // base = start of the get_alloc_size extra region (right after dst out)
    const char * base = (const char *) dst->data + ggml_nbytes(dst);

    // q4-direct: K/V q4_0 served from the raw block cache in-kernel — no f16
    // mirror allocation, no dequant pass (alloc_size computed the same way).
    const bool k_direct = K->type == GGML_TYPE_Q4_0 && sm70_q4_direct();
    const bool v_direct = V->type == GGML_TYPE_Q4_0 && sm70_q4_direct();

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst,
            K->type != GGML_TYPE_F16 && !k_direct,
            !(V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs))
              && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16 && !v_direct);

    size_t dequant_bytes = (size_t) (f16_extra.end - (uintptr_t) base);  // f16_extra region only (NOT incl. output data)
    dequant_bytes = GGML_PAD(dequant_bytes, 128);
    char * Qs_bytes = (char *) base + dequant_bytes;
    half * Qs = (half *) Qs_bytes;
    float * Os = (float *) (Qs_bytes + (size_t) nQ * sizeof(half));

    // zero pad rows of Qs (their outputs are never scattered back) and all of Os
    CUDA_CHECK(cudaMemsetAsync((void *) Qs_bytes, 0,
        (size_t) nQ * sizeof(half) + (size_t) nQ * sizeof(float), ctx.stream()));

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    cudaStream_t stream = ctx.stream();
    sm70_d256_dequant_kv((ggml_tensor *) K, (ggml_tensor *) V, f16_extra,
                         V_is_K_view, stream, k_direct, v_direct);

    const half * K_h2;
    const half * V_h2;
    int64_t k_row_stride, k_head_stride;
    int64_t v_row_stride, v_head_stride;
    if (K->type == GGML_TYPE_F16) {
        K_h2 = (const half *) K->data;
        k_row_stride   = K->nb[1] / sizeof(half);
        k_head_stride  = K->nb[2] / sizeof(half);
    } else if (k_direct) {
        // q4-direct: raw block bytes; strides in BYTES ([ctx][head][block]
        // pos-major: nb[1] = ctx stride, nb[2] = head stride). The Kq4 kernel
        // branch reinterprets the pointer/strides accordingly.
        K_h2 = (const half *) K->data;
        k_row_stride   = K->nb[1];
        k_head_stride  = K->nb[2];
    } else {
        K_h2 = (const half *) f16_extra.K;
        // 8/23 fix (variant scan pinned): to_fp16_nc linearizes dst as
        // [ne1][ne2][ne0] = [ctx][hkv][D] (position-major), NOT [ne2][ne1][ne0].
        // The old head-major strides (row=D, head=ctx*D) scrambled K and were
        // the root cause of the 0.69 real-input divergence.
        k_row_stride   = (int64_t) K->ne[2] * K->ne[0]; // hkv * D
        k_head_stride  = K->ne[0];                       // D
    }
    if (V_is_K_view) {
        // Correct for BOTH paths: staged (dequantized f16 mirror of K) and
        // q4-direct (same raw bytes/strides as K — V is a view of K's buffer).
        V_h2 = K_h2;
        v_row_stride  = k_row_stride;
        v_head_stride = k_head_stride;
    } else if (V->type == GGML_TYPE_F16) {
        V_h2 = (const half *) V->data;
        v_row_stride   = V->nb[1] / sizeof(half);
        v_head_stride  = V->nb[2] / sizeof(half);
    } else if (v_direct) {
        // q4-direct: raw block bytes, strides in BYTES (see K above).
        V_h2 = (const half *) V->data;
        v_row_stride   = V->nb[1];
        v_head_stride  = V->nb[2];
    } else {
        V_h2 = (const half *) f16_extra.V;
        // 8/23 fix: same [ne1][ne2][ne0] position-major layout as K above.
        v_row_stride   = (int64_t) V->ne[2] * V->ne[0];
        v_head_stride  = V->ne[0];
    }

    // ------------------------------------------------------ stage Q (f32->f16)
    {
        const dim3 grid(q_pad, nb * hkv, gqa);
        sm70_d256_stage_q_kernel<<<grid, 128, 0, stream>>>(
            (const float2 *) Q->data, (half2 *) Qs, q_len, hkv, gqa, nb,
            Q->nb[1] / 8, Q->nb[2] / 8, Q->nb[3] / 8);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------- attention
    // 8/23 cause hunt: capture this (first) invocation's kernel inputs.
    // (q4-direct skips the dump: K_h2/V_h2 are raw block bytes, not the f16
    // tensors the dump/repro tooling consumes.)
    if (!k_direct && !v_direct) {
        sm70_dump_kernel_inputs(K_h2, k_row_stride, k_head_stride,
                                V_h2, v_row_stride, v_head_stride,
                                Qs, q_pad, kv_len, q_len, hkv, hkv * gqa);
    }
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    using El = cutlass::half_t;
    // ElOut=float: attention output stays f32 end-to-end (8/23 review — the f16
    // Os staging was the largest sm70-side per-layer rounding source).
    // 8 instantiations: {dense, SplitKV3} x {staged/f16, Kq4, Vq4, Kq4+Vq4}.
    auto kernel_00 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, false>;
    auto kernel_10 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, true,  false>;
    auto kernel_01 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, true>;
    auto kernel_11 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, true,  true>;
    // SplitKV3 (upstream sm70_flash_attn_d256_splitkv3 patch, 8/23 port):
    // 3-way KV split for long-prefix prefill — triples the CTA count so late
    // chunks of a long prefill stop serializing their KV sweep on a saturated
    // SM grid. Env-tunable threshold (default 2048; 0 disables).
    auto kernel_s3_00 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true, false, false>;
    auto kernel_s3_10 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true, true,  false>;
    auto kernel_s3_01 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true, false, true>;
    auto kernel_s3_11 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true, true,  true>;

    static bool smem_raised = false;
    if (!smem_raised) {
        for (const void * kfn : {(const void *) kernel_00, (const void *) kernel_10,
                                 (const void *) kernel_01, (const void *) kernel_11,
                                 (const void *) kernel_s3_00, (const void *) kernel_s3_10,
                                 (const void *) kernel_s3_01, (const void *) kernel_s3_11}) {
            CUDA_CHECK(cudaFuncSetAttribute(kfn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
        }
        smem_raised = true;
    }

    static const int splitkv3_min_kv = [] {
        const char * e = getenv("LLAMA_SM70_SPLITKV3_MIN_KV");
        return e ? atoi(e) : 2048;
    }();
    const bool use_splitkv3 = nb == 1 && splitkv3_min_kv > 0
        && kv_len >= splitkv3_min_kv && kv_len > q_len;

    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / SM70_D256_BLOCK_M,
                    use_splitkv3 ? (unsigned) (nb * 3) : (unsigned) nb,
                    hkv * gqa);

    // SplitKV3 partial buffers follow Os in the scratch allocation.
    const int64_t rows3 = (int64_t) nb * q_pad * hkv * gqa;
    float * partial_out = (float *) ((char *) Os + (size_t) nQ * sizeof(float));
    float * partial_max = partial_out + 3 * rows3 * SM70_D256_D;
    float * partial_sum = partial_max + 3 * rows3;

    if (use_splitkv3) {
        const auto kfn = k_direct ? (v_direct ? kernel_s3_11 : kernel_s3_10)
                                  : (v_direct ? kernel_s3_01 : kernel_s3_00);
        kfn<<<grid, block, Traits::kSmemBytes, stream>>>(
                (const El *) Qs,
                (const El *) K_h2,
                (const El *) V_h2,
                (float *) Os,
                /*q_batch_stride*/ (int64_t) (hkv * gqa) * q_pad * SM70_D256_D,
                /*q_row_stride  */ SM70_D256_D,
                /*q_head_stride */ (int64_t) q_pad * SM70_D256_D,
                /*k_outer_stride*/ 0,
                /*k_row_stride  */ (int) k_row_stride,
                /*k_head_stride */ (int) k_head_stride,
                /*v_outer_stride*/ 0,
                /*v_row_stride  */ (int) v_row_stride,
                /*v_head_stride */ (int) v_head_stride,
                q_pad,
                kv_len,
                hkv * gqa,
                hkv,
                kv_offset,
                softmax_scale_log2,
                nullptr, 0, 0,
                partial_out, partial_max, partial_sum);
    } else {
        const auto kfn = k_direct ? (v_direct ? kernel_11 : kernel_10)
                                  : (v_direct ? kernel_01 : kernel_00);
        kfn<<<grid, block, Traits::kSmemBytes, stream>>>(
                (const El *) Qs,
                (const El *) K_h2,
                (const El *) V_h2,
                (float *) Os,
                /*q_batch_stride*/ (int64_t) (hkv * gqa) * q_pad * SM70_D256_D,  // Qs: [b][head_q][row][d]
                /*q_row_stride  */ SM70_D256_D,
                /*q_head_stride */ (int64_t) q_pad * SM70_D256_D,
                /*k_outer_stride*/ 0,
                /*k_row_stride  */ (int) k_row_stride,
                /*k_head_stride */ (int) k_head_stride,
                /*v_outer_stride*/ 0,
                /*v_row_stride  */ (int) v_row_stride,
                /*v_head_stride */ (int) v_head_stride,
                q_pad,
                kv_len,
                hkv * gqa,   // heads_q
                hkv,         // heads_kv
                kv_offset,
                softmax_scale_log2,
                nullptr, 0, 0,
                nullptr, nullptr, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());

    if (use_splitkv3) {
        // Merge the three partial segments straight into the f32 Os staging
        // buffer (same [row][D] layout the dense path produces).
        FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel
            <<<dim3((unsigned) rows3), Traits::kHeadDim, 0, stream>>>(
                (const float *) partial_out,
                (const float *) partial_max,
                (const float *) partial_sum,
                (float *) Os,
                rows3,
                softmax_scale_log2);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------- scatter
    {
        const dim3 grid(q_len, nb * hkv, gqa);
        sm70_d256_scatter_kernel<<<grid, 128, 0, stream>>>(
            (const float2 *) Os, (float2 *) dst->data, hkv, gqa, nb, q_pad,
            Q->nb[1] / 8, Q->nb[2] / 8, Q->nb[3] / 8);
        CUDA_CHECK(cudaGetLastError());
    }
}
