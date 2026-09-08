// Dynamic VBR transcode orchestration. Turbo sources reuse the fattn dequant;
// classic F16/Q8_0/Q4_0 sources use the stock converter. Both share the
// validated set_rows encoder.
#include "common.cuh"
#include "convert.cuh"
#include "set-rows.cuh"
#include "vbr-transcode.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "ggml-turbo-meansub.h"
#include <algorithm>
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

static __global__ void k_vbr_add_mean_f32(
        float * __restrict__ rows, const float * __restrict__ mean,
        int64_t n_rows, int64_t ne0) {
    const int64_t row = (int64_t) blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    float * current = rows + row*ne0;
    for (int64_t column = threadIdx.x; column < ne0; column += blockDim.x) {
        current[column] += mean[column];
    }
}

namespace {

constexpr size_t VBR_WORKSPACE_ALIGNMENT = 128;

struct vbr_workspace_layout {
    size_t f16_off = 0;
    size_t f32_off = 0;
    size_t idx_off = 0;
    size_t mean_off = 0;
    size_t bytes   = 0;
};

static bool vbr_workspace_add_plane(size_t count, size_t element_size, size_t & off, size_t & plane_off) {
    if (count > SIZE_MAX / element_size) {
        return false;
    }
    const size_t raw = count * element_size;
    if (raw > SIZE_MAX - (VBR_WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    const size_t padded = ((raw + VBR_WORKSPACE_ALIGNMENT - 1) / VBR_WORKSPACE_ALIGNMENT)
                        * VBR_WORKSPACE_ALIGNMENT;
    if (off > SIZE_MAX - padded) {
        return false;
    }
    plane_off = off;
    off += padded;
    return true;
}

static bool vbr_workspace_layout_for(int64_t rows, int64_t ne0, bool with_indices,
                                     bool with_mean,
                                     vbr_workspace_layout & layout) {
    layout = {};
    if (rows < 0 || ne0 <= 0) {
        return false;
    }
    if (rows == 0) {
        return true;
    }
    if ((uint64_t) rows > SIZE_MAX / (uint64_t) ne0) {
        return false;
    }
    const size_t elements = (size_t) rows * (size_t) ne0;
    size_t off = 0;
    if (!vbr_workspace_add_plane(elements, sizeof(half),  off, layout.f16_off) ||
        !vbr_workspace_add_plane(elements, sizeof(float), off, layout.f32_off)) {
        return false;
    }
    if (with_indices &&
        !vbr_workspace_add_plane((size_t) rows, sizeof(int32_t), off, layout.idx_off)) {
        return false;
    }
    if (with_mean &&
        !vbr_workspace_add_plane((size_t) ne0, sizeof(float), off, layout.mean_off)) {
        return false;
    }
    layout.bytes = off;
    return true;
}

static bool vbr_workspace_required(int64_t n_cells, int64_t ne0, int64_t stash_rows,
                                   bool with_mean,
                                   size_t & required) {
    if (n_cells < 0 || stash_rows < 0) {
        return false;
    }
    if (n_cells == 0 && stash_rows == 0) {
        required = 0;
        return true;
    }
    if (ne0 <= 0) {
        return false;
    }

    const int64_t transcode_rows = n_cells > 0
                                 ? (getenv("VBR_TRANSCODE_NOTILE") ? n_cells : 256)
                                 : 0;
    vbr_workspace_layout transcode;
    vbr_workspace_layout stash;
    if (!vbr_workspace_layout_for(transcode_rows, ne0, true, with_mean, transcode) ||
        !vbr_workspace_layout_for(stash_rows, ne0, false, false, stash)) {
        return false;
    }
    required = std::max(transcode.bytes, stash.bytes);

    // Fidelity mode dequantizes all rows in fixed 256-row tiles before and after the transcode.
    // The operations are stream-ordered, so only their maximum simultaneous layout is resident.
    if (getenv("VBR_TRANSCODE_FIDELITY") != nullptr) {
        vbr_workspace_layout fidelity;
        if (!vbr_workspace_layout_for(256, ne0, false, false, fidelity)) {
            return false;
        }
        required = std::max(required, fidelity.bytes);
    }
    return true;
}

static bool vbr_workspace_next_pow2(size_t bytes, size_t & result) {
    result = 1;
    while (result < bytes) {
        if (result > SIZE_MAX / 2) {
            return false;
        }
        result *= 2;
    }
    return true;
}

static bool vbr_workspace_round_up(size_t bytes, size_t granularity, size_t & result) {
    if (granularity == 0 || bytes > SIZE_MAX - (granularity - 1)) {
        return false;
    }
    result = ((bytes + granularity - 1) / granularity) * granularity;
    return true;
}

static size_t vbr_workspace_physical_now(const ggml_cuda_vbr_transcode_workspace & workspace) {
    const size_t vmm = workspace.vmm != nullptr
                     ? ggml_backend_cuda_vmm_pool_mapped(workspace.vmm) : 0;
    GGML_ASSERT(workspace.cuda_size <= SIZE_MAX - vmm);
    return workspace.cuda_size + vmm;
}

static uint8_t * vbr_workspace_try(ggml_backend_cuda_context & ctx, size_t need_bytes) {
    GGML_ASSERT(ctx.curr_stream_no == 0 && "VBR transcode workspace is owned by the side backend main stream");
    auto & workspace = ctx.vbr_transcode_workspace;
    if (need_bytes == 0) {
        return workspace.vmm != nullptr
             ? (uint8_t *) ggml_backend_cuda_vmm_pool_base(workspace.vmm)
             : workspace.cuda_buf;
    }
    if (workspace.vmm != nullptr && need_bytes <= workspace.vmm_hw) {
        return (uint8_t *) ggml_backend_cuda_vmm_pool_base(workspace.vmm);
    }

    ggml_cuda_set_device(ctx.device);
    // Once VA reservation falls back to cudaMalloc, keep that representation for the context's
    // lifetime. Migrating it later to a smaller VMM mapping would violate grow-only accounting.
    if (workspace.cuda_buf == nullptr && ggml_backend_cuda_vmm_available(ctx.device)) {
        if (workspace.vmm == nullptr || need_bytes > workspace.vmm_va) {
            if (workspace.vmm != nullptr) {
                ggml_backend_cuda_vmm_pool_free(workspace.vmm);
                workspace.vmm    = nullptr;
                workspace.vmm_va = 0;
                workspace.vmm_hw = 0;
            }
            size_t va = 0;
            if (!vbr_workspace_next_pow2(need_bytes, va)) {
                return nullptr;
            }
            workspace.vmm = ggml_backend_cuda_vmm_pool_init(ctx.device, va);
            if (workspace.vmm != nullptr) {
                workspace.vmm_va = va;
            }
        }
        if (workspace.vmm != nullptr) {
            const size_t mapped_before = ggml_backend_cuda_vmm_pool_mapped(workspace.vmm);
            if (!ggml_backend_cuda_vmm_pool_map(workspace.vmm, 0, need_bytes)) {
                // vmm_pool_map may have landed and zeroed some chunks before the failing chunk.
                // Its early return does not settle those legacy-stream memsets.
                if (ggml_backend_cuda_vmm_pool_mapped(workspace.vmm) > mapped_before) {
                    CUDA_CHECK(cudaStreamSynchronize(nullptr));
                }
                return nullptr;
            }
            workspace.vmm_hw = std::max(workspace.vmm_hw, need_bytes);
            return (uint8_t *) ggml_backend_cuda_vmm_pool_base(workspace.vmm);
        }
    }

    size_t alloc = 0;
    if (!vbr_workspace_next_pow2(need_bytes, alloc)) {
        return nullptr;
    }
    if (alloc > workspace.cuda_size) {
        if (workspace.cuda_buf != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            CUDA_CHECK(cudaFree(workspace.cuda_buf));
            workspace.cuda_buf  = nullptr;
            workspace.cuda_size = 0;
        }
        uint8_t * ptr = nullptr;
        if (cudaMalloc(&ptr, alloc) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        workspace.cuda_buf  = ptr;
        workspace.cuda_size = alloc;
    }
    return workspace.cuda_buf;
}

static uint8_t * vbr_workspace_get(ggml_backend_cuda_context & ctx, size_t need_bytes) {
    uint8_t * base = vbr_workspace_try(ctx, need_bytes);
    if (base == nullptr && need_bytes > 0) {
        GGML_ABORT("VBR transcode workspace: physical memory exhausted growing to %.1f MiB on "
                   "device %d (boundary reserve missed)", need_bytes / 1048576.0, ctx.device);
    }
    return base;
}

} // namespace

static bool ggml_backend_cuda_kv_transcode_workspace_memory_impl(
        ggml_backend_t backend_or_null, int device,
        int64_t n_cells, int64_t ne0, int64_t stash_rows, bool with_mean,
        size_t * physical_now, size_t * physical_if_reserved) {
    GGML_ASSERT(physical_now != nullptr);
    GGML_ASSERT(physical_if_reserved != nullptr);
    *physical_now = 0;
    *physical_if_reserved = 0;

    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        return false;
    }
    const ggml_cuda_vbr_transcode_workspace * workspace = nullptr;
    if (backend_or_null != nullptr) {
        GGML_ASSERT(ggml_backend_is_cuda(backend_or_null));
        const auto & ctx = *(const ggml_backend_cuda_context *) backend_or_null->context;
        GGML_ASSERT(ctx.device == device);
        workspace = &ctx.vbr_transcode_workspace;
        *physical_now = vbr_workspace_physical_now(*workspace);
    }

    size_t need_bytes = 0;
    if (!vbr_workspace_required(n_cells, ne0, stash_rows, with_mean, need_bytes)) {
        return false;
    }
    if (need_bytes == 0) {
        *physical_if_reserved = *physical_now;
        return true;
    }

    size_t fallback = 0;
    if (!vbr_workspace_next_pow2(need_bytes, fallback)) {
        return false;
    }
    if (!ggml_backend_cuda_vmm_available(device)) {
        *physical_if_reserved = std::max(*physical_now, fallback);
        return true;
    }

    size_t vmm = 0;
    if (!vbr_workspace_round_up(need_bytes, ggml_backend_cuda_vmm_granularity(device), vmm)) {
        return false;
    }
    if (workspace != nullptr && workspace->cuda_buf != nullptr) {
        // Fallback is sticky once selected, preserving grow-only physical accounting.
        *physical_if_reserved = std::max(*physical_now, fallback);
    } else if (workspace != nullptr && workspace->vmm != nullptr && need_bytes <= workspace->vmm_va) {
        *physical_if_reserved = std::max(*physical_now, vmm);
    } else {
        // The cold/re-reservation path can land on VMM or cudaMalloc fallback. Offers must not
        // rely on the physically smaller path succeeding or predict a grow-only shrink.
        *physical_if_reserved = std::max({ *physical_now, vmm, fallback });
    }
    return true;
}

extern "C" bool ggml_backend_cuda_kv_transcode_workspace_memory(
        ggml_backend_t backend_or_null, int device,
        int64_t n_cells, int64_t ne0, int64_t stash_rows,
        size_t * physical_now, size_t * physical_if_reserved) {
    return ggml_backend_cuda_kv_transcode_workspace_memory_impl(
        backend_or_null, device, n_cells, ne0, stash_rows, false,
        physical_now, physical_if_reserved);
}

extern "C" bool ggml_backend_cuda_kv_transcode_workspace_memory_v2(
        ggml_backend_t backend_or_null, int device,
        const ggml_vbr_transcode_workspace_params_v2 * params,
        size_t * physical_now, size_t * physical_if_reserved) {
    return params != nullptr && ggml_backend_cuda_kv_transcode_workspace_memory_impl(
        backend_or_null, device, params->n_cells, params->ne0,
        params->stash_rows, params->mean_addback,
        physical_now, physical_if_reserved);
}

extern "C" bool ggml_backend_cuda_kv_transcode_workspace_reserve(
        ggml_backend_t backend, int64_t n_cells, int64_t ne0, int64_t stash_rows) {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto & ctx = *(ggml_backend_cuda_context *) backend->context;
    size_t need_bytes = 0;
    if (!vbr_workspace_required(n_cells, ne0, stash_rows, false, need_bytes)) {
        return false;
    }
    return need_bytes == 0 || vbr_workspace_try(ctx, need_bytes) != nullptr;
}

extern "C" bool ggml_backend_cuda_kv_transcode_workspace_reserve_v2(
        ggml_backend_t backend,
        const ggml_vbr_transcode_workspace_params_v2 * params) {
    if (backend == nullptr || params == nullptr || !ggml_backend_is_cuda(backend)) {
        return false;
    }
    auto & ctx = *(ggml_backend_cuda_context *) backend->context;
    size_t need_bytes = 0;
    if (!vbr_workspace_required(
            params->n_cells, params->ne0, params->stash_rows,
            params->mean_addback, need_bytes)) {
        return false;
    }
    return need_bytes == 0 || vbr_workspace_try(ctx, need_bytes) != nullptr;
}

void ggml_cuda_vbr_transcode_workspace_free(ggml_backend_cuda_context & ctx) {
    auto & workspace = ctx.vbr_transcode_workspace;
    ggml_cuda_set_device(ctx.device);
    if (workspace.vmm != nullptr) {
        ggml_backend_cuda_vmm_pool_free(workspace.vmm);
        workspace.vmm    = nullptr;
        workspace.vmm_va = 0;
        workspace.vmm_hw = 0;
    }
    if (workspace.cuda_buf != nullptr) {
        CUDA_CHECK(cudaFree(workspace.cuda_buf));
        workspace.cuda_buf  = nullptr;
        workspace.cuda_size = 0;
    }
}

// f16<->f32 plane moves use the stock convert helpers (ggml_get_to_fp16_cuda / ggml_get_to_fp32_cuda)

// VBR_TRANSCODE_FIDELITY=1 debug: dequant every row of a tensor to host f32 (original domain).
static void vbr_fidelity_dequant_all(ggml_backend_cuda_context & ctx, const char * data, ggml_type type,
                                     size_t rowbytes, int64_t n_cells, int64_t ne0, bool is_v,
                                     cudaStream_t stream, std::vector<float> & out) {
    const int64_t TILE = 256;
    out.resize((size_t) n_cells * ne0);
    vbr_workspace_layout layout;
    GGML_ASSERT(vbr_workspace_layout_for(TILE, ne0, false, false, layout));
    uint8_t * workspace = vbr_workspace_get(ctx, layout.bytes);
    half *  s16 = (half *)  (workspace + layout.f16_off);
    float * s32 = (float *) (workspace + layout.f32_off);
    for (int64_t c = 0; c < n_cells; c += TILE) {
        const int64_t Te = std::min<int64_t>(TILE, n_cells - c);
        if (ggml_is_turbo_kv_type(type)) {
            vbr_dequant_turbo_to_f32(data + (size_t) c * rowbytes, type, type,
                                     s16, s32, Te, ne0, rowbytes, is_v, ctx.device, stream);
        } else {
            const to_fp32_cuda_t to_f32 = ggml_get_to_fp32_cuda(type);
            GGML_ASSERT(to_f32 != nullptr);
            to_f32(data + (size_t) c * rowbytes, s32, Te * ne0, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(out.data() + (size_t) c * ne0, s32,
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

// Transcode the first n_cells rows of p->src (type A) into p->dst as type B.
// p->dst points into the KV pool at the destination region. p->src->name MUST be the real cache
// tensor name (cache_k_l<L>_ms<M> / cache_v_l<L>_ms<M>) — the encoder keys its K/V codebook,
// affine-table identity, and kmean tap off it.
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
// ASYNC (S5): no end-of-call sync — everything is stream-ordered on ctx.stream(). The persistent
// context-owned workspace is reused by the NEXT transcode on the same stream, whose kernels are
// ordered behind this one's kernels by construction.
// NOTE: assumes decode-side InnerQ (d_innerq_channel_scale_inv_fattn) is already identity/calibrated
// from prior decode (true in the live decode-time path).
static void ggml_cuda_vbr_kv_transcode_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_vbr_transcode_params * p,
        const float * host_mean) {
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
    vbr_workspace_layout layout;
    GGML_ASSERT(vbr_workspace_layout_for(TILE, ne0, true, host_mean != nullptr, layout));
    uint8_t * workspace = vbr_workspace_get(ctx, layout.bytes);
    half *    scratch_f16 = (half *)    (workspace + layout.f16_off);
    float *   scratch_f32 = (float *)   (workspace + layout.f32_off);
    int32_t * idx_buf     = (int32_t *) (workspace + layout.idx_off);
    float *   mean_buf    = host_mean != nullptr
                          ? (float *) (workspace + layout.mean_off) : nullptr;
    if (host_mean != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(
            mean_buf, host_mean, (size_t) ne0*sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }
    {
        const int64_t threads = 256;
        const int64_t blocks  = (TILE + threads - 1) / threads;
        k_vbr_iota_i32<<<(unsigned) blocks, (unsigned) threads, 0, stream>>>(idx_buf, TILE);
    }

    // Stage-2 scaffolding built ONCE (not per tile — a 32k-cell tensor has 128 tiles and per-tile
    // ggml_init/free serializes the host between GPU launches). Tensors describe a full TILE; full
    // tiles only retarget data pointers, and the (at most one) final partial tile shrinks ne/nb.
    // ->buffer is a valid handle merely to satisfy non-null checks; set_rows reads by ->data only.
    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * tctx = ggml_init(ip);
    ggml_tensor * src0 = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, ne0, TILE);
    src0->data = scratch_f32; src0->buffer = pool_buf;
    ggml_tensor * src1 = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, TILE);
    src1->data = idx_buf;     src1->buffer = pool_buf;   // iota [0,Te) -> dst rows [0,Te) of the tile
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

    const auto is_classic_type = [](ggml_type type) {
        return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
    };
    const bool classic_edge = is_classic_type(src_A->type) && is_classic_type(type_B);
    const auto classic_to_f32 = classic_edge ? ggml_get_to_fp32_cuda(src_A->type) : nullptr;

    const int64_t n_tiles = (n_cells + TILE - 1) / TILE;
    int64_t cur_te = TILE;
    for (int64_t ti = 0; ti < n_tiles; ++ti) {
        const int64_t c  = (reverse_tiles ? n_tiles - 1 - ti : ti) * TILE;
        const int64_t Te = (n_cells - c < TILE) ? (n_cells - c) : TILE;

        // Stage 1: reconstruct cells [c, c+Te) in original-domain F32. Classic
        // VBR uses the stock quant conversion directly; Turbo retains its
        // FWHT/codebook/alpha path. Both feed the same generic set_rows encoder.
        if (classic_edge) {
            classic_to_f32(
                (const char *) src_A->data + (size_t) c * rA,
                scratch_f32, Te * ne0, stream);
        } else {
            vbr_dequant_turbo_to_f32((const char *) src_A->data + (size_t) c * rA, src_A->type, type_B,
                                     scratch_f16, scratch_f32,
                                     Te, ne0, rA, is_v, ctx.device, stream);
        }
        // f16 sink-stash: rows below stash_rows re-encode from the pristine stash captured at the
        // tensor's FIRST degrade, not from the tier-A recon — sink rows are permanently hot AND
        // permanently old (they survive every wave), so this caps their error at single-hop forever.
        // VBR_STASH_CAPTURE_ONLY=1 (debug): keep the stash as the fidelity reference but skip the
        // injection — measures the UNstashed accumulation against the same pristine yardstick.
        static const bool stash_capture_only = getenv("VBR_STASH_CAPTURE_ONLY") != nullptr;
        if (stash_f16 != nullptr && c < stash_rows && !stash_capture_only) {
            const int64_t overlap = std::min<int64_t>(Te, stash_rows - c);
            ggml_get_to_fp32_cuda(GGML_TYPE_F16)(
                (const half *) stash_f16 + (size_t) c * ne0, scratch_f32, overlap * ne0, stream);
        }
        if (mean_buf != nullptr) {
            ggml_vbr_mean_addback_launch_shape shape;
            GGML_ASSERT(ggml_vbr_mean_addback_launch_shape_for(Te, ne0, &shape));
            k_vbr_add_mean_f32<<<shape.blocks, shape.threads, 0, stream>>>(
                scratch_f32, mean_buf, Te, ne0);
        }

        // Stage 2: re-encode F32 into destination rung B through set_rows.
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

void ggml_cuda_vbr_kv_transcode(ggml_backend_cuda_context & ctx,
                                const ggml_vbr_transcode_params * p) {
    ggml_cuda_vbr_kv_transcode_impl(ctx, p, nullptr);
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
    vbr_workspace_layout layout;
    GGML_ASSERT(vbr_workspace_layout_for(n_rows, ne0, false, false, layout));
    uint8_t * workspace = vbr_workspace_get(ctx, layout.bytes);
    half *  s16 = (half *)  (workspace + layout.f16_off);
    float * s32 = (float *) (workspace + layout.f32_off);
    vbr_dequant_turbo_to_f32((const char *) src->data, src->type, src->type,
                             s16, s32, n_rows, ne0, src->nb[1], is_v, ctx.device, stream);
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(s32, (half *) stash_f16, n_rows * ne0, stream);
}

// Host-facing wrapper (callable from llama-kv-cache under GGML_USE_CUDA). extern "C" to match the
// ggml-cuda.h declaration (C linkage).
extern "C" void ggml_backend_cuda_kv_transcode(ggml_backend_t backend,
                                               const struct ggml_vbr_transcode_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_vbr_kv_transcode(*cuda_ctx, params);
}

extern "C" bool ggml_backend_cuda_kv_cross_domain_reconstruct(
        ggml_backend_t backend,
        const ggml_vbr_cross_domain_reconstruct_params * params) {
    if (backend == nullptr || params == nullptr || params->transcode.src == nullptr ||
        !ggml_backend_is_cuda(backend) || params->meansub_model_id <= 0 ||
        params->meansub_layer < 0 || params->transcode.src->ne[0] <= 0 ||
        (params->transcode.type_B != GGML_TYPE_TURBO8_0 &&
         params->transcode.type_B != GGML_TYPE_F16)) {
        return false;
    }
    switch (params->transcode.src->type) {
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            break;
        default:
            return false;
    }
    int max_l = 0;
    int max_c = 0;
    int live = 0;
    const float * table = ggml_turbo_meansub_table(
        params->meansub_model_id, params->transcode.is_v ? 1 : 0,
        &max_l, &max_c, &live);
    const uint64_t ne0 = uint64_t(params->transcode.src->ne[0]);
    if (table == nullptr || live <= 0 || params->meansub_layer >= max_l ||
        params->logical_offset > uint64_t(max_c) ||
        ne0 > uint64_t(max_c) - params->logical_offset) {
        return false;
    }
    const float * host_mean = table +
        size_t(params->meansub_layer)*size_t(max_c) + size_t(params->logical_offset);
    auto & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_vbr_kv_transcode_impl(ctx, &params->transcode, host_mean);
    return true;
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
