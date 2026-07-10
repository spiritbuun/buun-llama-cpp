// Dynamic VBR transcode, Stage 2 (orchestration): turbo tier A -> lower tier B for n_cells cells.
// Reuses the fattn dequant (Stage 1, original-domain f32) + the validated set_rows encoder.
#include "common.cuh"
#include "convert.cuh"
#include "set-rows.cuh"
#include "vbr-transcode.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include <cstdlib>
#include <cmath>
#include <vector>

// Defined in set-rows.cu. Suppress the encode mean-sub tap during the transcode re-encode: the
// Stage-1 dequant already emits stored-domain (V - mu_V), so re-subtracting mu would double it.
extern bool g_turbo_meansub_suppress;

static __global__ void k_vbr_iota_i32(int32_t * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (int32_t) i;
    }
}

// f16<->f32 plane moves use the stock convert helpers (ggml_get_to_fp16_cuda / ggml_get_to_fp32_cuda)

// VBR_TRANSCODE_FIDELITY=1 debug: dequant every row of a tensor to host f32 (original domain).
static void vbr_fidelity_dequant_all(ggml_backend_cuda_context & ctx, const char * data, ggml_type type,
                                     size_t rowbytes, int64_t n_cells, int64_t ne0, bool is_v,
                                     cudaStream_t stream, std::vector<float> & out) {
    const int64_t TILE = 256;
    out.resize((size_t) n_cells * ne0);
    ggml_cuda_pool_alloc<half>  s16(ctx.pool(), (size_t) TILE * ne0);
    ggml_cuda_pool_alloc<float> s32(ctx.pool(), (size_t) TILE * ne0);
    for (int64_t c = 0; c < n_cells; c += TILE) {
        const int64_t Te = std::min<int64_t>(TILE, n_cells - c);
        vbr_dequant_turbo_to_f32(data + (size_t) c * rowbytes, type, type,
                                 s16.get(), s32.get(), Te, ne0, rowbytes, is_v, ctx.device, stream);
        CUDA_CHECK(cudaMemcpyAsync(out.data() + (size_t) c * ne0, s32.get(),
                                   (size_t) Te * ne0 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Compare dq(B, transcoded) against dq(A, original) row by row; a faithful transcode differs only
// by tier-B quantization noise (rms(B-A) ~ rms of static-B error, bias ~ 0, no row-band outliers).
static void vbr_fidelity_report(const char * name, ggml_type tA, ggml_type tB,
                                const std::vector<float> & A, const std::vector<float> & B,
                                int64_t n_cells, int64_t ne0) {
    double se = 0.0, bias = 0.0, ref = 0.0, dot = 0.0;
    double worst_rms = -1.0; int64_t worst_row = -1;
    const int64_t BAND = 1024;
    fprintf(stderr, "VBR FIDELITY %s %s->%s rows=%lld ne0=%lld\n", name, ggml_type_name(tA), ggml_type_name(tB),
            (long long) n_cells, (long long) ne0);
    for (int64_t b = 0; b < n_cells; b += BAND) {
        const int64_t be = std::min<int64_t>(b + BAND, n_cells);
        double bse = 0.0, bbias = 0.0, bref = 0.0, bdot = 0.0, bworst = -1.0; int64_t bworst_row = -1;
        for (int64_t r = b; r < be; ++r) {
            double rse = 0.0;
            for (int64_t i = 0; i < ne0; ++i) {
                const double a = A[r*ne0 + i], v = B[r*ne0 + i];
                const double d = v - a;
                rse   += d*d;
                bbias += d;
                bref  += a*a;
                bdot  += a*v;
            }
            bse += rse;
            const double rrms = sqrt(rse / ne0);
            if (rrms > bworst) { bworst = rrms; bworst_row = r; }
        }
        se += bse; bias += bbias; ref += bref; dot += bdot;
        if (bworst > worst_rms) { worst_rms = bworst; worst_row = bworst_row; }
        fprintf(stderr, "  band [%6lld,%6lld): rms(B-A)=%.5f rmsA=%.5f slope=%.5f bias=%+.6f worst row %lld rms=%.5f\n",
                (long long) b, (long long) be,
                sqrt(bse / ((double)(be - b) * ne0)), sqrt(bref / ((double)(be - b) * ne0)),
                bref > 0 ? bdot / bref : 0.0,
                bbias / ((double)(be - b) * ne0), (long long) bworst_row, bworst);
    }
    // slope = LS gain of B against A: 1.0 = energy-faithful; <1 = systematic norm shrink (attention
    // logits to these rows sink coherently — the repetition-collapse mechanism), >1 = inflation
    fprintf(stderr, "VBR FIDELITY TOTAL: rms(B-A)=%.5f rmsA=%.5f rel=%.4f slope=%.5f bias=%+.6f worst row %lld rms=%.5f\n",
            sqrt(se / ((double) n_cells * ne0)), sqrt(ref / ((double) n_cells * ne0)),
            sqrt(se / (ref > 0 ? ref : 1)), ref > 0 ? dot / ref : 0.0,
            bias / ((double) n_cells * ne0),
            (long long) worst_row, worst_rms);
}

// Transcode the first n_cells rows of p->src (turbo type A) into p->dst as turbo type B.
// p->dst points into the KV pool at the destination region. p->src->name MUST be the real cache
// tensor name (cache_k_l<L> / cache_v_l<L>) — the encoder keys its K/V codebook + kmean tap off it.
//
// STREAMING + IN-PLACE (VBR design decisions #2/#3): processed one TILE of cells at a time, so the
// f32/f16 scratch is bounded (~few MB, independent of n_cells) — NOT the whole-tensor f32 buffer.
// In-place safety is direction-dependent:
//   rB <= rA (degrade): ascending tiles — the write offset (c*rB) always trails the read (c*rA);
//   rB >  rA (container promotion): DESCENDING tiles — tile c's write [c*rB, ...) only overlaps
//     the sources of tiles > c, which a descending walk has already consumed. Either way a tile's
//     own read/write overlap is safe: Stage-1 fully dequants the tile into scratch before Stage-2
//     writes, and both are ordered on the same stream. Promotion requires the caller to have
//     MAPPED the grown extent first. (Promotion re-encodes the tier-A recon — it restores no
//     information; it exists so FUTURE rows encode at the higher tier.)
// ASYNC (S5): no end-of-call sync — everything is stream-ordered on ctx.stream(). The pool-alloc
// scratch returns to the per-context pool at scope exit; reuse by the NEXT transcode on the same
// stream is ordered behind this one's kernels, so that is safe by construction.
// NOTE: assumes decode-side InnerQ (d_innerq_channel_scale_inv_fattn) is already identity/calibrated
// from prior decode (true in the live decode-time path).
void ggml_cuda_vbr_kv_transcode(ggml_backend_cuda_context & ctx,
                                const ggml_vbr_transcode_params * p) {
    ggml_cuda_set_device(ctx.device); // multi-GPU waves interleave devices; never rely on the caller
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * src_A      = p->src;
    const ggml_type     type_B     = p->type_B;
    void *              dst_B_data = p->dst;
    const int64_t       n_cells    = p->n_cells;
    const bool          is_v       = p->is_v;
    const void *        stash_f16  = p->stash_f16;
    const int64_t       stash_rows = p->stash_rows;
    const char *        dst_name   = src_A->name;
    ggml_backend_buffer_t pool_buf = p->pool_buf;
    const int64_t ne0 = src_A->ne[0];
    const size_t  rA  = src_A->nb[1];                       // source bytes/cell
    const size_t  rB  = ggml_row_size(type_B, ne0);         // dest bytes/cell (contiguous)
    const bool reverse_tiles = dst_B_data == src_A->data && rB > rA; // in-place promotion

    const bool fidelity = getenv("VBR_TRANSCODE_FIDELITY") != nullptr;
    std::vector<float> fidA;
    if (fidelity) {
        vbr_fidelity_dequant_all(ctx, (const char *) src_A->data, src_A->type, rA, n_cells, ne0, is_v, stream, fidA);
    }
    // One tile of cells in flight. f32 scratch = TILE*ne0*4 (~6 MB at ne0=6144, TILE=256), reused.
    // VBR_TRANSCODE_NOTILE (debug): one tile = whole tensor = the old non-tiled behavior, to isolate
    // tiling bugs from source-state issues in the anchor.
    const int64_t TILE = (getenv("VBR_TRANSCODE_NOTILE") && n_cells > 0) ? n_cells : 256;
    ggml_cuda_pool_alloc<half>    scratch_f16(ctx.pool(), (size_t) TILE * ne0);
    ggml_cuda_pool_alloc<float>   scratch_f32(ctx.pool(), (size_t) TILE * ne0);
    ggml_cuda_pool_alloc<int32_t> idx_buf    (ctx.pool(), (size_t) TILE);
    {
        const int64_t threads = 256;
        const int64_t blocks  = (TILE + threads - 1) / threads;
        k_vbr_iota_i32<<<(unsigned) blocks, (unsigned) threads, 0, stream>>>(idx_buf.get(), TILE);
    }

    // Stage-2 scaffolding built ONCE (not per tile — a 32k-cell tensor has 128 tiles and per-tile
    // ggml_init/free serializes the host between GPU launches). Tensors describe a full TILE; full
    // tiles only retarget data pointers, and the (at most one) final partial tile shrinks ne/nb.
    // ->buffer is a valid handle merely to satisfy non-null checks; set_rows reads by ->data only.
    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * tctx = ggml_init(ip);
    ggml_tensor * src0 = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, ne0, TILE);
    src0->data = scratch_f32.get(); src0->buffer = pool_buf;
    ggml_tensor * src1 = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, TILE);
    src1->data = idx_buf.get();     src1->buffer = pool_buf;   // iota [0,Te) -> dst rows [0,Te) of the tile
    ggml_tensor * dstB = ggml_new_tensor_2d(tctx, type_B, ne0, TILE);
    dstB->buffer = pool_buf;
    ggml_set_name(dstB, dst_name);
    dstB->src[0] = src0;
    dstB->src[1] = src1;
    auto set_rows_count = [&](int64_t Te) {
        src0->ne[1] = Te; src0->nb[2] = src0->nb[1]*Te; src0->nb[3] = src0->nb[2];
        src1->ne[0] = Te;
        dstB->ne[1] = Te; dstB->nb[2] = dstB->nb[1]*Te; dstB->nb[3] = dstB->nb[2];
    };

    const int64_t n_tiles = (n_cells + TILE - 1) / TILE;
    int64_t cur_te = TILE;
    for (int64_t ti = 0; ti < n_tiles; ++ti) {
        const int64_t c  = (reverse_tiles ? n_tiles - 1 - ti : ti) * TILE;
        const int64_t Te = (n_cells - c < TILE) ? (n_cells - c) : TILE;

        // Stage 1: dequant cells [c, c+Te) of src_A -> original-domain f32 [Te, ne0]
        vbr_dequant_turbo_to_f32((const char *) src_A->data + (size_t) c * rA, src_A->type, type_B,
                                 scratch_f16.get(), scratch_f32.get(),
                                 Te, ne0, rA, is_v, ctx.device, stream);
        // f16 sink-stash: rows below stash_rows re-encode from the pristine stash captured at the
        // tensor's FIRST degrade, not from the tier-A recon — sink rows are permanently hot AND
        // permanently old (they survive every wave), so this caps their error at single-hop forever.
        // VBR_STASH_CAPTURE_ONLY=1 (debug): keep the stash as the fidelity reference but skip the
        // injection — measures the UNstashed accumulation against the same pristine yardstick.
        static const bool stash_capture_only = getenv("VBR_STASH_CAPTURE_ONLY") != nullptr;
        if (stash_f16 != nullptr && c < stash_rows && !stash_capture_only) {
            const int64_t overlap = std::min<int64_t>(Te, stash_rows - c);
            ggml_get_to_fp32_cuda(GGML_TYPE_F16)(
                (const half *) stash_f16 + (size_t) c * ne0, scratch_f32.get(), overlap * ne0, stream);
        }

        // Stage 2: re-encode f32 -> turbo B into dst_B_data + c*rB via the set_rows path
        if (Te != cur_te) {
            set_rows_count(Te); // partial tile (last in ascending order, FIRST in descending)
            cur_te = Te;
        }
        dstB->data = (char *) dst_B_data + (size_t) c * rB;

        // Encode-tap suppression is SOURCE-dependent, keyed on the source's STORED domain:
        //   tapped tiers (t4 and every TCQ tier) store mean-subtracted rows -> their dequant
        //     emits V - mu_V / K - mu_K, so the encode tap must NOT re-subtract (suppress);
        //   t8 (turbo_tap_mu gates it out of the tap; the graph V-restore has the matching
        //     exclusion) and F16 (the dynamic entry) store FULL-domain rows -> the tap must
        //     RUN when the destination tier expects mean-subtracted storage.
        // Read on the host during dispatch, so set/reset here is safe.
        g_turbo_meansub_suppress = ggml_is_turbo_kv_type(src_A->type) &&
                                   src_A->type != GGML_TYPE_TURBO8_0;
        ggml_cuda_op_set_rows(ctx, dstB);
        g_turbo_meansub_suppress = false;
    }
    ggml_free(tctx);

    // scrub stale tier-A bytes on kept pages past the new tier-B extent (stream-ordered after the
    // final tile's writes — the scrub region starts at the write high-water mark)
    if (p->scrub_bytes > 0) {
        CUDA_CHECK(cudaMemsetAsync((char *) dst_B_data + (size_t) n_cells * rB, 0, p->scrub_bytes, stream));
    }

    if (fidelity) {
        std::vector<float> fidB;
        vbr_fidelity_dequant_all(ctx, (const char *) dst_B_data, type_B, rB, n_cells, ne0, is_v, stream, fidB);
        vbr_fidelity_report(dst_name, src_A->type, type_B, fidA, fidB, n_cells, ne0);
        if (stash_f16 != nullptr && stash_rows > 0) {
            // sink rows judged against the STASH (≈pristine) — the honest reference for them
            std::vector<half>  hs((size_t) stash_rows * ne0);
            CUDA_CHECK(cudaMemcpy(hs.data(), stash_f16, hs.size()*sizeof(half), cudaMemcpyDeviceToHost));
            double se = 0.0, ref = 0.0;
            for (int64_t r = 0; r < std::min<int64_t>(stash_rows, n_cells); ++r) {
                for (int64_t i = 0; i < ne0; ++i) {
                    const double a = __half2float(hs[r*ne0 + i]);
                    const double d = (double) fidB[r*ne0 + i] - a;
                    se += d*d; ref += a*a;
                }
            }
            fprintf(stderr, "VBR FIDELITY SINK[0,%lld) vs stash: rms=%.5f rel=%.4f\n",
                    (long long) std::min<int64_t>(stash_rows, n_cells),
                    sqrt(se / ((double) std::min<int64_t>(stash_rows, n_cells) * ne0)),
                    sqrt(se / (ref > 0 ? ref : 1)));
        }
    }
}

// Capture the f16 sink stash: dequant the first n_rows of src (original/stored domain, same
// convention as the transcode's Stage 1) and pack to f16 at stash_f16. ASYNC (S5): stream-ordered
// ahead of the same-stream transcode that consumes/overwrites the source rows.
extern "C" void ggml_backend_cuda_kv_stash_capture(ggml_backend_t backend, const struct ggml_tensor * src,
                                                   void * stash_f16, int64_t n_rows, bool is_v) {
    ggml_backend_cuda_context & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    const int64_t ne0 = src->ne[0];
    ggml_cuda_pool_alloc<half>  s16(ctx.pool(), (size_t) n_rows * ne0);
    ggml_cuda_pool_alloc<float> s32(ctx.pool(), (size_t) n_rows * ne0);
    vbr_dequant_turbo_to_f32((const char *) src->data, src->type, src->type,
                             s16.get(), s32.get(), n_rows, ne0, src->nb[1], is_v, ctx.device, stream);
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(s32.get(), (half *) stash_f16, n_rows * ne0, stream);
}

// Host-facing wrapper (callable from llama-kv-cache under GGML_USE_CUDA). extern "C" to match the
// ggml-cuda.h declaration (C linkage).
extern "C" void ggml_backend_cuda_kv_transcode(ggml_backend_t backend,
                                               const struct ggml_vbr_transcode_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_vbr_kv_transcode(*cuda_ctx, params);
}

extern "C" void ggml_backend_cuda_sync_device(int device) {
    if (device >= 0) {
        ggml_cuda_set_device(device);
    }
    // this is the degrade wave's write-visibility barrier AND the pre-unmap barrier —
    // a silently failed sync would transcode stale bytes or rip pages still in use
    CUDA_CHECK(cudaDeviceSynchronize());
}

// S5 side-stream fence: one event per (device, arming stream). arm() records on the VBR side
// stream after a degrade wave's async work; the next graph_compute on the device consumes ALL
// armed fences for that device with GPU-side stream waits, so the decode graph runs after the
// transcodes without the host ever blocking. Keyed by stream because one device can host several
// VBR caches with their own side backends (iSWA base+SWA) — a single per-device event let the
// second cache's arm re-record onto a different stream and drop the first cache's fence.
// Re-arming the same stream's slot before a consume simply re-records its event — waiting on the
// newest record covers all earlier wave work on that stream. Single-threaded by design (all
// sites run on the llama_decode thread).
struct ggml_cuda_vbr_fence {
    int          device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t  ev     = nullptr;
    bool         armed  = false;
};
// 4 slots per device is generous: base + SWA side streams today, spares for future caches
static ggml_cuda_vbr_fence g_vbr_fences[GGML_CUDA_MAX_DEVICES * 4] = {};

extern "C" void ggml_backend_cuda_vbr_fence_arm(ggml_backend_t backend) {
    ggml_backend_cuda_context & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx.device);
    ggml_cuda_vbr_fence * slot = nullptr;
    for (auto & f : g_vbr_fences) {
        if (f.device == ctx.device && f.stream == ctx.stream()) {
            slot = &f;
            break;
        }
        if (slot == nullptr && f.device < 0) {
            slot = &f; // first free slot, claimed only if no exact match exists
        }
    }
    GGML_ASSERT(slot != nullptr && "VBR fence table full — more side streams than slots");
    slot->device = ctx.device;
    slot->stream = ctx.stream();
    if (slot->ev == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&slot->ev, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(slot->ev, ctx.stream()));
    slot->armed = true;
}

void ggml_cuda_vbr_fence_consume(int device, cudaStream_t stream) {
    // Fast-path early-out: skip CUDA runtime calls entirely when no fences are armed on this device.
    // At steady state (settled VBR, no degrades in flight) this eliminates ~4 driver transitions per token.
    bool any_armed = false;
    for (auto & f : g_vbr_fences) {
        if (f.armed && f.device == device) {
            any_armed = true;
            break;
        }
    }
    if (!any_armed) {
        return;
    }

    for (auto & f : g_vbr_fences) {
        if (!f.armed || f.device != device) {
            continue;
        }
        CUDA_CHECK(cudaStreamWaitEvent(stream, f.ev, 0));
        // disarm only once the wave has RETIRED: another graph_compute on this device (e.g. the
        // transcode oracle's throwaway backend) must not steal the decode graph's pending wait.
        // While in flight every graph waits (correct either way); after retirement the first
        // consumer clears the flag and the fast path is a per-slot branch again.
        if (cudaEventQuery(f.ev) == cudaSuccess) {
            f.armed = false;
        } else {
            (void) cudaGetLastError(); // absorb the benign cudaErrorNotReady from the query
        }
    }
}
