#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include "ggml-vbr.h" // backend interface for turbo KV / dynamic VBR (resolved at init, never linked)

#include <unordered_map>
#include <utility>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

// Dynamic VBR (S3): one step of the measured decode-time degrade price order —
// knock (layer il, K/V side) down to `tier` (a vbr_tier index, see llama-kv-cache.cpp).
struct vbr_degrade_step {
    uint8_t il;
    uint8_t is_v;
    uint8_t tier;
};

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share = nullptr,
        const llama_memory_vbr_params & vbr = {});

    ~llama_kv_cache(); // frees the VBR VMM pool (if any); = default otherwise

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    // monotone counter of in-place VBR tier flips — graph reuse fences on it.
    // A share-linked cache (mem_other) views the owner's tensors, so its graphs must
    // fence on the OWNER's flips: delegate to the source cache (shared-KV drafters,
    // e.g. the gemma4 MTP assistant, follow the target's VBR tier changes this way).
    uint64_t vbr_tier_epoch() const { return other ? other->vbr_tier_epoch() : vbr_tier_epoch_; }

    // effective bits/value of this cache at the CURRENT tensor types (llama_memory_i)
    double kv_bpv() const override;

    llama_memory_vbr_state_data memory_vbr_state(llama_seq_id seq_id, uint32_t n_tokens_extra) const override;
    // totals for cross-cache aggregation (iSWA weights its children by stored values)
    void   kv_bpv_accum(double & bits, double & vals) const;

    double memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;
    double memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;

    // shared floor-walk core (runtime clamp + fit capacity math), see impl comment
    struct vbr_floor_sim_result {
        size_t clamp_step     = 0;     // steps applied before the clamp (== order size if unclamped)
        size_t n_pinned       = 0;
        double next_bpv       = 0.0;   // aggregate the clamping step would have produced
        double bits_per_token = 0.0;   // end-state KV bits per token (0 = no units)
        std::vector<ggml_type> end_types; // [layers*2] end-state tier, GGML_TYPE_COUNT = absent
    };
    vbr_floor_sim_result vbr_floor_sim(double floor_bpv, bool pooled_only,
            ggml_type entry_k = GGML_TYPE_COUNT, ggml_type entry_v = GGML_TYPE_COUNT) const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;


    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    // Dynamic VBR (M2): per-(layer,side) descriptor over the shared KV pool buffer. Tier is NOT
    // mirrored here — the cache tensor (layers[ikv].k/.v) is the single source of truth for the
    // TYPE; a degrade flips the tensor and this descriptor only tracks placement. Cell WIDTH is
    // per-pool: `t` is the tensor instance whose bytes live in this pool — the cache tensor
    // itself under -sm layer, or this device's shard of it under -sm tensor (same name, same
    // type, ne0 = this device's slice of the head*dim axis). All per-pool byte math (row sizes,
    // slots, stash sizing) derives from `t`, never from the canonical layers[] tensor.
    struct vbr_extent {
        ggml_tensor * t    = nullptr;           // pool-local tensor instance (canonical or shard)
        size_t    byte_off = 0;                 // offset of this tensor's data within the pool buffer
        ggml_type type0    = GGML_TYPE_COUNT;   // ENTRY tier (immutable; full-clear reset target)
        size_t    stash_off   = 0;              // offset into the f16 sink-stash buffer
        uint32_t  stash_valid = 0;              // rows captured (0 = not yet)
        // promote transcodes with live rows since the last full reset: each one re-encodes the
        // aged rows from their degraded recon, so error compounds per hop — cap bounds the damage
        uint8_t   promote_hops = 0;
    };
    // Multi-GPU: one vbr_pool per KV-hosting device buffer. Extent vectors stay indexed by ikv in
    // EVERY pool; only entries whose tensor (or tensor shard) lives in that pool's buffer are
    // populated (e.t == nullptr elsewhere). Under -sm layer the populated sets are DISJOINT (each
    // device owns whole layers); under -sm tensor every pool holds a per-device SHARD of every
    // (layer,side), so a tier flip transcodes in every pool that has a nonzero extent for it.
    // With a single GPU there is exactly one pool and all logic reduces to the previous
    // single-pool controller bit-for-bit.
    struct vbr_pool {
        ggml_backend_buffer_t buf     = nullptr; // non-owning (lives in ctxs_bufs); one KV buffer
        char *                base    = nullptr; // ggml_backend_buffer_get_base(buf)
        size_t                size    = 0;        // total pool bytes
        size_t                used    = 0;        // high-water of placed extents (log-only)
        size_t                budget  = 0;         // per-pool share of vbr_budget_bytes_ (VA-size proportional)
        size_t                budget_base = 0;      // init-armed/fallback share: the re-derivation floor
        // vbr_budget_eff memo: one live free-VRAM query per pool per boundary (the degrade loop
        // and promote hysteresis both consult it repeatedly within one boundary)
        mutable uint64_t      budget_eff_stamp = ~0ull;
        mutable size_t        budget_eff_cache = 0;
        std::vector<vbr_extent> k;                // indexed by kv-cache layer id (ikv)
        std::vector<vbr_extent> v;
        // backend VBR vtable that owns this pool's device (resolved from the buffer type's
        // registry at init; a pool only exists if the backend exports it)
        const ggml_vbr_backend_iface * be = nullptr;
        // S2 (option C): VMM-backed pool — per-tensor fixed VA slots, physical pages mapped on
        // demand. When set, `size` is the VA reservation (not physical); each extent's byte_off is
        // page-aligned so tensor-tail unmaps never straddle a neighbor's pages.
        struct ggml_vbr_vmm_pool * vmm = nullptr;
        uint32_t wm_cells    = 0;                 // cells already backed for every extent
        int      device      = -1;                // backend device ordinal backing the pool
        size_t   gran        = 0;                 // page granularity
        size_t   mapped_base = 0;                 // bytes mapped up front (rotation matrices)
        // #88 scratch-reserve memo: widest f16 row per dequant-active side, valid while no tier
        // flips (keyed on vbr_tier_epoch_; ~0 forces the first compute)
        uint64_t scratch_rows_epoch = ~0ull;
        size_t   scratch_k_row      = 0;
        size_t   scratch_v_row      = 0;
        // per-device transcode side stream (lazy) + S5 overlap state: transcodes run async on
        // backend's stream; the next decode graph GPU-waits via the armed per-device fence
        // (be->fence_arm). Tail pages a transcode may still READ (rA extent >
        // kept rB extent) can only be unmapped once it finishes — queue them and flush at the
        // next decode boundary, when the wave is long done.
        ggml_backend_t backend      = nullptr;
        bool           wave_pending = false;      // async GPU work enqueued, fence not yet armed
        std::vector<std::pair<size_t, size_t>> unmap_deferred; // {pool byte_off, len}
        // f16 sink-stash (VBR_STASH_ROWS env; 0 = off): pristine first-degrade snapshot of the
        // first N rows per tensor — permanently-hot sink rows re-encode from it at every hop
        ggml_backend_buffer_t stash_buf = nullptr;
    };
    void vbr_vmm_ensure_mapped(); // grow physical backing to the current cell watermark
    bool vbr_vmm_try_map(uint32_t wm); // same, recoverable: false on physical exhaustion

    // S3/S4: decode-time degrade controller (VMM mode only). The price order and its cursor stay
    // GLOBAL (layer-global price order); each step resolves the pool that owns its tensor.
    llama_memory_vbr_params vbr_params_;              // API/CLI inputs (ctor copy; env can override)
    // bumped on every in-place tier flip (degrade/promote/full reset). Graph reuse must be
    // fenced on it: a reused graph carries the OLD type/strides baked into its K/V views, and
    // a free-VRAM-clamp wave (or a promote map-retry) can flip tiers MID-band where the n_kv
    // shape check alone would still allow reuse.
    uint64_t vbr_tier_epoch_ = 0;
    std::vector<vbr_degrade_step> vbr_degrade_order_; // global price order, F16->t8 band first
    size_t         vbr_degrade_cursor_ = 0;
    size_t         vbr_budget_bytes_   = 0;           // global mapped-physical budget; 0 = no trigger
    uint32_t       vbr_stash_rows_     = 0;           // sink-stash rows per (layer,side); 0 = off
    // --vbr-floor (env VBR_MIN_BITS): first order step the aggregate bits/value floor forbids;
    // the cursor never advances past it (default = order size, i.e. unclamped)
    size_t vbr_degrade_limit_ = (size_t) -1;
    size_t vbr_floor_cost_bytes_ = 0;                 // page-exact cost of the floor layout at full
                                                      // kv_size (fallback budget in dynamic mode)
    bool   vbr_budget_warned_ = false;                // budget-unmeetable warning fired (terminal)
    // prepare() boundaries since the last applied degrade step — promote cooldown basis
    // (deterministic, unlike wall time); promotes wait for a quiet window after any degrade
    uint32_t vbr_quiet_boundaries_ = 0;
    // auto-budget runtime re-derivation (explicit budgets never move): boundary counter (the
    // FIRST boundary is skipped — lazy cuBLAS/CUDA-graph pools have not allocated yet and free
    // overstates reality) + resolved growth headroom
    uint64_t vbr_boundary_count_   = 0;
    size_t   vbr_growth_headroom_  = 0;
    bool     vbr_budget_explicit_  = false;
    // what this pool's device can give it right now: device_share x (mapped + free - headroom),
    // 64 MiB-quantized. Shared by the init-time auto-budget arm (fit-less modes, e.g.
    // SPLIT_MODE_TENSOR) and the periodic re-derivation.
    size_t   vbr_pool_reach(const vbr_pool & p) const;
    // Fast-path stability tracking: skip per-batch VBR bookkeeping when settled (avoids ~1ms/token)
    uint32_t vbr_last_used_        = 0;   // observed cell count last prepare() pass
    void     vbr_rederive_budget();
    // sink-stash staleness guard: set when any cell below stash_rows is freed (its content can be
    // rewritten by another request; injecting the old snapshot would corrupt the new rows)
    bool   vbr_stash_dirty_   = false;
    void     vbr_full_reset();                        // cache empty: undo every degrade (lossless)
    void     vbr_shrink_watermark();                  // occupancy dropped: release phantom tail pages
    bool     vbr_promote_next(uint32_t wm_next);      // occupancy dropped: re-promote one container
    void     vbr_floor_clamp_order();
    void     vbr_flush_deferred_unmaps();
    bool     vbr_scratch_reserve(uint32_t wm_cells);  // #88: boundary-time f16 dequant scratch grow
    char *   vbr_stash_ensure(vbr_pool & p);          // lazy per-pool sink-stash buffer; returns base
    void     vbr_load_degrade_order();                // baked table, VBR_DEGRADE_ORDER=<file>, or generic fallback
    void     vbr_synth_generic_order();               // cross-model curves for unsupported archs (VBR_FORCE_GENERIC=1 to force)
    size_t   vbr_vmm_projected_bytes(const vbr_pool & p, uint32_t wm_cells) const;
    size_t   vbr_budget_eff(const vbr_pool & p) const; // live-clamped per-pool budget (shared basis)
    bool     vbr_vmm_active() const;                  // any pool is VMM-backed
    bool     vbr_over_budget(uint32_t wm_cells) const; // any VMM pool projected past its budget
    vbr_pool *       vbr_pool_of(const ggml_tensor * t);       // pool owning the tensor (by buffer)
    const vbr_pool * vbr_pool_of(const ggml_tensor * t) const;
    // every VMM pool holding an extent for one (layer,side) unit: exactly one under -sm layer,
    // one per device under -sm tensor (each with that device's shard), empty for static units.
    // A tier step applies to the unit — i.e. to EVERY entry returned here.
    // every VMM pool holding unit (ikv, side): a const ref into vbr_units_tab_, precomputed
    // once after pool construction (membership is fixed for the cache's lifetime) — the
    // degrade/promote paths call this per decode boundary, so it must not allocate
    const std::vector<std::pair<vbr_pool *, vbr_extent *>> & vbr_units_of(size_t ikv, bool is_v) const;
    bool vbr_unit_pooled(size_t ikv, bool is_v) const;         // any VMM pool holds this unit
    // side pinned via mixed config (-ctk turbo8 -ctv vbr): ladder never touches it
    bool vbr_side_pinned(bool is_v) const { return is_v ? vbr_params_.pin_v : vbr_params_.pin_k; }
    // unified pin contract: a unit may be stepped only if its current type is a vbr tier AND
    // its side is not flag-pinned — every degrade/promote/sim walk must use this predicate
    bool vbr_unit_movable(ggml_type t, bool is_v) const;
    uint32_t vbr_watermark_cells(uint32_t extra_tokens) const; // shared by prepare() + ensure_mapped
    bool     vbr_degrade_next(uint32_t wm_next);      // one step down the order; false = exhausted
                                                      // wm_next = projected watermark incl. the
                                                      // incoming batch (bounds live pages/scrub)

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // Dynamic VBR shared KV pools (M2 bookkeeping; M3 transcode/relocate) — one per KV buffer
    // (per device under -sm layer; exactly one on a single GPU)
    std::vector<vbr_pool> vbr_pools_;
    // [ikv*2 + is_v] -> (pool, extent) units; built once at ctor end, immutable after
    std::vector<std::vector<std::pair<vbr_pool *, vbr_extent *>>> vbr_units_tab_;

    // Permanent transcode oracle (env VBR_TRANSCODE_TEST): synthetic turbo8 A->A byte round-trip +
    // turbo8->turbo4 in-place-vs-separate identity, on a scoped CUDA backend. See definition.
    void vbr_transcode_anchor_test();

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // VBR tier-flip epoch of the underlying cache (0 when VBR is off — the counter never moves)
    uint64_t get_vbr_epoch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;


    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
