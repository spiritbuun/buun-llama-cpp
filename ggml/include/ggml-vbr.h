#pragma once

// Backend interface for TurboQuant KV-cache support (turbo-typed KV tensors and the
// dynamic VBR degrade controller).
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

// Dynamic VBR: transcode the first n_cells rows of a turbo KV tensor (src) to a lower turbo tier
// (type_B), writing into dst (a region of the KV pool buffer; == src->data for the in-place
// degrade). src->name must be the cache tensor name (cache_k_l<L> / cache_v_l<L>) so the encoder
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
    // #88: grow the per-device f16 dequant scratch (the buffer the flash-attention prefill /
    // materialize paths grow implicitly to the attended width) to hold >= k_bytes / v_bytes,
    // OUTSIDE any graph. Called at the decode boundary so physical exhaustion fails HERE,
    // recoverably, instead of aborting mid-decode. false = physical memory exhausted (the
    // caller flushes its deferred unmaps and retries once before failing the batch).
    bool (*kv_dequant_scratch_reserve)(int device, size_t k_bytes, size_t v_bytes);
};

// proc name resolved via ggml_backend_reg_get_proc_address
#define GGML_VBR_BACKEND_IFACE_PROC "ggml_backend_vbr_iface"

typedef const struct ggml_vbr_backend_iface * (*ggml_backend_vbr_iface_fn_t)(void);

#ifdef __cplusplus
}
#endif
