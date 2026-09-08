#pragma once

// Backend interface for dynamically retiered KV caches (TurboQuant and classic
// F16/Q8_0/Q4_0 representation ladders).
//
// libllama links NO backend symbols for this feature. At KV-cache init it resolves this
// vtable through the backend registry:
//
//     reg  = ggml_backend_dev_backend_reg(ggml_backend_buft_get_device(buft));
//     get  = (ggml_backend_vbr_iface_fn_t) ggml_backend_reg_get_proc_address(reg, GGML_VBR_BACKEND_IFACE_PROC);
//     face = get ? get() : NULL;
//
// A backend that can host turbo-typed KV (encode/decode kernels, VMM pool, async tier
// transcode) exports the proc and fills EVERY slot — no member may be NULL. Backends
// without support simply do not export the proc; the KV cache then refuses turbo KV
// types on their buffers with a clean init-time error instead of failing at decode.
//
// Today only the CUDA backend implements it (ggml_backend_cuda_vbr_iface).

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque VMM pool: one virtual-address reservation for the whole KV context, per-tensor
// fixed page-aligned offsets, physical pages mapped on demand as occupancy grows and
// unmapped after tier degrades. Nothing ever relocates.
struct ggml_vbr_vmm_pool;

// #88: which sides the flash-attention f16 dequant scratch serves for a (K,V) type pair —
// the SINGLE authoritative copy of the materialize condition, consumed by the fattn
// prefill/decode paths AND by the KV cache's boundary scratch reserve/estimator. Turbo tiers
// always materialize; q8_0/bf16 only next to a turbo partner (the mixed pair must become
// (F16,F16)); f16 never does. Drift between the kernels and the reserve would re-open the
// mid-decode abort this predicate exists to prevent — edit HERE only. (Decode additionally
// dequants q8_0/bf16 at head dims > 256; that term stays local to fattn.cu, ANDed on top.)
static inline void ggml_vbr_kv_dequant_sides(enum ggml_type tk, enum ggml_type tv,
                                             bool * need_k, bool * need_v) {
    const bool turbo_k = ggml_is_turbo_kv_type(tk);
    const bool turbo_v = ggml_is_turbo_kv_type(tv);
    *need_k = turbo_k || ((tk == GGML_TYPE_Q8_0 || tk == GGML_TYPE_BF16) && turbo_v);
    *need_v = turbo_v || ((tv == GGML_TYPE_Q8_0 || tv == GGML_TYPE_BF16) && turbo_k);
}

// Dynamic VBR: transcode the first n_cells rows of a managed KV tensor (src) to another rung
// (type_B), writing into dst (a region of the KV pool buffer; == src->data for the in-place
// degrade). src->name must be the cache tensor name (cache_k_l<L>_ms<M> / cache_v_l<L>_ms<M>) so the encoder
// picks the right K/V codebook. stash_f16/stash_rows (nullable/0): f16 sink-stash — rows
// [0, stash_rows) re-encode from this pristine snapshot instead of the tier-A recon, capping the
// permanently-hot sink rows at single-hop error across any number of degrades; capture it at the
// tensor's FIRST degrade. scrub_bytes: zero this many bytes after the tier-B extent (stale tier-A
// bytes on kept pages read as padding can carry NaN f16 scales).
// ASYNC: everything is enqueued on the backend's stream; the caller orders consumers with
// fence_arm or ggml_backend_synchronize.
struct ggml_vbr_transcode_params {
    const struct ggml_tensor * src;
    enum ggml_type             type_B;
    void *                     dst;
    ggml_backend_buffer_t      pool_buf;
    int64_t                    n_cells;
    bool                       is_v;
    const void *               stash_f16;
    int64_t                    stash_rows;
    size_t                     scrub_bytes;
};

// Cross-domain reconstruction reuses the ordinary tiled transcode, but restores the baked
// affine mean between source dequantization and full-domain re-encoding. logical_offset selects
// this tensor-parallel shard's columns within the canonical logical row.
struct ggml_vbr_cross_domain_reconstruct_params {
    struct ggml_vbr_transcode_params transcode;
    int32_t                          meansub_model_id;
    int32_t                          meansub_layer;
    uint64_t                         logical_offset;
};

struct ggml_vbr_transcode_workspace_params_v2 {
    int64_t n_cells;
    int64_t ne0;
    int64_t stash_rows;
    bool    mean_addback;
};

// Shared CPU-visible launch contract for the cross-domain mean add-back. One block owns one
// logical row; its threads stride columns, avoiding a runtime division/modulo for every value.
struct ggml_vbr_mean_addback_launch_shape {
    uint32_t blocks;
    uint32_t threads;
};

static inline bool ggml_vbr_mean_addback_launch_shape_for(
        int64_t rows, int64_t columns,
        struct ggml_vbr_mean_addback_launch_shape * output) {
    if (output == NULL || rows <= 0 || columns <= 0 || (uint64_t) rows > UINT32_MAX) {
        return false;
    }
    output->blocks = (uint32_t) rows;
    output->threads = 256;
    return true;
}

struct ggml_vbr_backend_iface {
    // -- device utilities ------------------------------------------------------------
    int                        (*get_device_count)(void);
    ggml_backend_buffer_type_t (*buffer_type)     (int device);       // default buffer type of `device`
    void                       (*get_device_memory)(int device, size_t * free, size_t * total);
    // dedicated side-stream backend instance on `device` for async transcodes
    ggml_backend_t             (*backend_init)(int device);
    // Block until ALL pending GPU work on `device` completes (device-wide, not one stream).
    // Serializes a VBR degrade wave (which reads live KV on the side backend) against the
    // model's in-flight KV writes; one call per wave, before the first transcode.
    void                       (*sync_device)(int device);

    // -- VMM pool --------------------------------------------------------------------
    bool   (*vmm_available)  (int device);
    size_t (*vmm_granularity)(int device);                            // page granularity, bytes
    struct ggml_vbr_vmm_pool * (*vmm_pool_init)(int device, size_t va_size); // NULL = reservation failed
    void   (*vmm_pool_free)  (struct ggml_vbr_vmm_pool * pool);
    void * (*vmm_pool_base)  (struct ggml_vbr_vmm_pool * pool);
    size_t (*vmm_pool_mapped)(struct ggml_vbr_vmm_pool * pool);       // mapped-physical bytes
    // Monotonic identity of the resident page set. Increments once per map/unmap call iff the
    // set changes, including a partially successful map that returns false.
    uint64_t (*vmm_pool_residency_epoch)(struct ggml_vbr_vmm_pool * pool);
    // Exact resident bytes in a page-aligned subrange. The range must be contained in the
    // reservation and aligned to vmm_granularity() at both ends. Read-only: never maps/unmaps.
    size_t (*vmm_pool_mapped_in_range)(struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
    // ensure [off, off+len) is backed by physical pages (rounded out to granularity; new pages
    // zeroed). false = physical memory exhausted (caller degrades or aborts); driver errors
    // beyond OOM are fatal inside the backend.
    bool   (*vmm_pool_map)   (struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
    // unmap chunks fully contained in [off, off+len); partially covered chunks stay mapped
    bool   (*vmm_pool_unmap) (struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
    // zero every mapped page (VMM-safe replacement for ggml_backend_buffer_clear on a
    // partially-mapped VA)
    void   (*vmm_pool_clear) (struct ggml_vbr_vmm_pool * pool);
    // wrap externally-managed device memory (a VMM VA range) as a backend buffer; the buffer
    // does NOT take ownership — freeing it never frees ptr
    ggml_backend_buffer_t (*buffer_from_ptr)(int device, void * ptr, size_t size);

    // -- KV transcode ----------------------------------------------------------------
    void (*kv_transcode)(ggml_backend_t backend, const struct ggml_vbr_transcode_params * params);
    // capture the f16 sink-stash: dequantize the first n_rows of src into stash_f16
    void (*kv_stash_capture)(ggml_backend_t backend, const struct ggml_tensor * src,
                             void * stash_f16, int64_t n_rows, bool is_v);
    // S5 side-stream overlap: record an event on the (side) backend's stream and arm a
    // per-device fence; the device's next graph_compute inserts a GPU-side wait on it, so
    // the decode graph waits for the degrade wave WITHOUT blocking the host.
    void (*fence_arm)(ggml_backend_t backend);
    // #88: grow the owning compute backend context's f16 dequant scratch (the buffer the
    // flash-attention prefill /
    // materialize paths grow implicitly to the attended width) to hold >= k_bytes / v_bytes,
    // OUTSIDE any graph. Called at the decode boundary so physical exhaustion fails HERE,
    // recoverably, instead of aborting mid-decode. false = physical memory exhausted (the
    // caller flushes its deferred unmaps and retries once before failing the batch).
    bool (*kv_dequant_scratch_reserve)(ggml_backend_t backend, size_t k_bytes, size_t v_bytes);
    // Read-only physical-memory projection for the same reserve. `physical_now` is the
    // backend context's currently resident K+V scratch. `physical_if_reserved` is a
    // conservative upper bound after a successful reserve to k_bytes/v_bytes, including
    // VMM allocation-granularity rounding and the cudaMalloc next-power-of-two fallback.
    // This never allocates, frees, maps, unmaps, migrates, or changes the scratch epoch.
    void (*kv_dequant_scratch_memory)(ggml_backend_t backend, size_t k_bytes, size_t v_bytes,
                                      size_t * physical_now, size_t * physical_if_reserved);
    // Persistent workspace used by both KV transcode and sink-stash capture. Projection accepts
    // a null backend before the lazy side backend exists; in that case `device` selects the
    // allocator policy and physical_now is zero. n_cells covers the transcode (including the
    // live VBR_TRANSCODE_NOTILE policy), while stash_rows covers a capture immediately before it.
    // false means invalid/overflowing dimensions. Neither callback mutates a tier.
    bool (*kv_transcode_workspace_memory)(ggml_backend_t backend_or_null, int device,
                                           int64_t n_cells, int64_t ne0, int64_t stash_rows,
                                           size_t * physical_now, size_t * physical_if_reserved);
    // Materialize the projected workspace on an existing side backend. Physical exhaustion is
    // recoverable and returns false, allowing the caller to reclaim/retry before tier mutation.
    bool (*kv_transcode_workspace_reserve)(ggml_backend_t backend,
                                            int64_t n_cells, int64_t ne0, int64_t stash_rows);
    // Clear a tensor subrange on the backend's side stream. Ordered with kv_transcode and async
    // tensor uploads submitted through the same backend. This is the final member of the legacy
    // interface: extending this unversioned object would make even a null-check read beyond an
    // older backend's static object.
    void (*tensor_memset_async)(ggml_backend_t backend, struct ggml_tensor * tensor,
                                size_t offset, size_t size);
};

// Cross-domain reconstruction is a separate, versioned proc-address capability. A backend that
// exports only GGML_VBR_BACKEND_IFACE_PROC remains safely usable for every legacy operation; the
// absence of this proc makes cross-domain reconstruction report-only without reading past the
// legacy object. Future versions use a new proc name rather than changing this layout.
#define GGML_VBR_CROSS_DOMAIN_IFACE_V1_VERSION 1u
struct ggml_vbr_cross_domain_iface_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    bool (*kv_cross_domain_reconstruct)(
        ggml_backend_t backend,
        const struct ggml_vbr_cross_domain_reconstruct_params * params);
    bool (*kv_transcode_workspace_memory_v2)(
        ggml_backend_t backend_or_null, int device,
        const struct ggml_vbr_transcode_workspace_params_v2 * params,
        size_t * physical_now, size_t * physical_if_reserved);
    bool (*kv_transcode_workspace_reserve_v2)(
        ggml_backend_t backend,
        const struct ggml_vbr_transcode_workspace_params_v2 * params);
};

// proc name resolved via ggml_backend_reg_get_proc_address
#define GGML_VBR_BACKEND_IFACE_PROC "ggml_backend_vbr_iface"
#define GGML_VBR_CROSS_DOMAIN_IFACE_V1_PROC "ggml_backend_vbr_cross_domain_iface_v1"

typedef const struct ggml_vbr_backend_iface * (*ggml_backend_vbr_iface_fn_t)(void);
typedef const struct ggml_vbr_cross_domain_iface_v1 *
    (*ggml_backend_vbr_cross_domain_iface_v1_fn_t)(void);

#ifdef __cplusplus
}
#endif
