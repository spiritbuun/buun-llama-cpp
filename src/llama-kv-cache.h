#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-vbr-policy.h"
#include "llama-vbr-transaction.h"

#include "ggml-vbr.h" // backend interface for turbo KV / dynamic VBR (resolved at init, never linked)
#include "llama-vram-ledger.h" // co-tenancy peer claim/marker types (P2)

#include <map>
#include <memory>
#include <set>
#include <tuple>
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

    void breathe() override;

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

    void vbr_cotenancy_accum(uint64_t & decrement, uint32_t & grants,
                             uint64_t & offer, uint64_t & pending) const override;

    bool vbr_ledger_tree_active() const override {
        return vbr_ledger_owner_ && vbr_vmm_active();
    }

    // Composite-cache ledger topology. A standalone cache is its own implicit root.
    // iSWA attaches both children after their pools exist so ownership follows the
    // actually active controller and the last child in parent execution order is root.
    bool vbr_controller_active() const { return vbr_vmm_active(); }
    void vbr_attach_ledger_tree(llama_kv_cache * root, llama_kv_cache * peer, double device_share);
    void vbr_finalize_ledger_tree();
    void vbr_finalize_failed_child(uint32_t n_tokens, bool root_ran);

    // A share-linked drafter consumes f16 dequant scratch on its own compute backend while
    // reading this cache's tensors. The owner-side donation planner needs the terminal
    // requirement for every DISTINCT registered backend, not merely its own backend.
    struct vbr_shared_scratch_plan {
        const ggml_vbr_backend_iface * be = nullptr;
        ggml_backend_t compute_backend = nullptr;
        int device = -1;
        size_t k_bytes = 0;
        size_t v_bytes = 0;
    };
    using vbr_shared_scratch_visitor = std::function<void(const vbr_shared_scratch_plan &)>;
    using vbr_device_watermarks = std::map<int, uint32_t>;

    // Visit one plan per distinct live consumer compute backend. `terminal_types` is indexed
    // like vbr_floor_sim_result::end_types; an empty vector (or GGML_TYPE_COUNT slot) uses the
    // tensor's live type. `terminal_watermarks` must name every registered consumer device;
    // device-local endpoints may differ after partial pool growth. The registry lock remains held
    // through each callback, so a drafter's context-teardown detach cannot return and free the
    // backend while it is being queried.
    void vbr_shared_scratch_visit(
            const std::vector<ggml_type> & terminal_types,
            const vbr_device_watermarks & terminal_watermarks,
            const vbr_shared_scratch_visitor & visitor) const;

    void vbr_shared_scratch_detach() override;

    double memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;
    double memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;

    // shared ladder-sim primitives: seed a per-(layer,side) type view + per-step
    // applicability under vbr_degrade_next's exact skip rules (see impl comment) — the
    // floor sim, the bpv-if-degraded walk and the co-tenancy offer all ride these
    void vbr_sim_seed(std::vector<ggml_type> & sim, bool pooled_only,
                      ggml_type entry_k, ggml_type entry_v,
                      double * sum_bits, int64_t * sum_vals, size_t * n_pinned) const;
    bool vbr_sim_step(const std::vector<ggml_type> & sim, size_t i,
                      size_t & slot, const ggml_tensor *& t, ggml_type & type_B) const;

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
    llama_turbo_meansub_ref get_turbo_meansub_ref(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

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

    // Side-effect-free admission preflight.  iSWA uses this to prove that both
    // child caches can place the whole batch before either child enters prepare().
    slot_info_vec_t plan_slots(const std::vector<llama_ubatch> & ubatches);

    // Complete preparation using a slot plan produced by plan_slots().
    slot_info_vec_t prepare_with_slots(
            const std::vector<llama_ubatch> & ubatches,
            slot_info_vec_t                   sinfos);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    // commit=false is used only by plan_slots() to suppress non-metadata side effects
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, bool commit = true);

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

        llama_turbo_meansub_ref turbo_meansub_ref;
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
        size_t                budget  = 0;         // current per-pool mapped-physical budget
        size_t                budget_base = 0;      // explicit arm or floor-layout share: re-derivation floor
        // vbr_budget_eff memo: one live free-VRAM query per pool per boundary (the degrade loop
        // and promote hysteresis both consult it repeatedly within one boundary)
        mutable uint64_t      budget_eff_stamp = ~0ull;
        mutable size_t        budget_eff_cache = 0;
        std::vector<vbr_extent> k;                // indexed by kv-cache layer id (ikv)
        std::vector<vbr_extent> v;
        // backend VBR vtable that owns this pool's device (resolved from the buffer type's
        // registry at init; a pool only exists if the backend exports it)
        const ggml_vbr_backend_iface * be = nullptr;
        // non-owning main compute backend whose context owns the fattn Q/K/V scratch. This is
        // intentionally distinct from `backend` below, which is a dedicated transcode stream.
        // Valid only while llama_context::backends is alive. llama_context declares `memory`
        // before `backends`, so backends are destroyed first: KV teardown must never dereference
        // this handle (runtime reserve calls happen while the complete context is alive).
        ggml_backend_t compute_backend = nullptr;
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
        // co-tenancy (P2): PCI bus id (resolved once from the backend device; empty = none)
        // and the summed unamortized grant decrement vbr_budget_eff subtracts
        std::string busid;
        mutable size_t grant_decrement = 0;
        // per-device transcode side stream (lazy) + S5 overlap state: transcodes run async on
        // backend's stream; the next decode graph GPU-waits via the armed per-device fence
        // (be->fence_arm). Tail pages a transcode may still READ (rA extent >
        // kept rB extent) can only be unmapped once it finishes — queue them and flush at the
        // next decode boundary, when the wave is long done.
        ggml_backend_t backend      = nullptr;
        bool           wave_pending = false;      // async GPU work enqueued, fence not yet armed
        std::vector<std::pair<size_t, size_t>> unmap_deferred; // {pool byte_off, len}
        // f16 sink-stash (VBR_STASH_ROWS env; 0 = off): one fixed-VA slab per KV pool. Extent
        // offsets are assigned once at construction; physical pages are mapped grow-only when an
        // extent first captures usable tapped-domain rows. This keeps allocation failure outside
        // tier mutation and makes current/projected physical occupancy exactly queryable.
        struct ggml_vbr_vmm_pool * stash_vmm = nullptr;
        size_t                     stash_size = 0;       // page-padded VA reservation size
    };
    // A share-linked cache aliases another context's K/V tensors but executes attention on
    // this context's compute backends. Since fattn scratch is backend-context-owned, each
    // device needs scratch-only metadata here even though the drafter intentionally owns no
    // VMM pool or VBR controller. Tensor pointers are non-owning aliases whose live types are
    // authoritative; row maxima are recomputed only when the delegated owner epoch changes.
    struct vbr_shared_scratch_binding {
        ggml_backend_buffer_t buf = nullptr;
        const ggml_vbr_backend_iface * be = nullptr;
        ggml_backend_t compute_backend = nullptr;
        int device = -1;
        std::vector<ggml_tensor *> k;
        std::vector<ggml_tensor *> v;
        uint64_t rows_epoch = ~0ull;
        size_t k_row = 0;
        size_t v_row = 0;
    };

    struct vbr_shared_scratch_registry;
    struct vbr_shared_scratch_registration {
        std::weak_ptr<vbr_shared_scratch_registry> registry;
        uint64_t id = 0;

        vbr_shared_scratch_registration() = default;
        vbr_shared_scratch_registration(
                const std::shared_ptr<vbr_shared_scratch_registry> & registry,
                uint64_t id);
        ~vbr_shared_scratch_registration();

        vbr_shared_scratch_registration(const vbr_shared_scratch_registration &) = delete;
        vbr_shared_scratch_registration & operator=(const vbr_shared_scratch_registration &) = delete;
        vbr_shared_scratch_registration(vbr_shared_scratch_registration && other) noexcept;
        vbr_shared_scratch_registration & operator=(vbr_shared_scratch_registration && other) noexcept;

        void reset();
    };
    vbr_shared_scratch_registration vbr_shared_scratch_register(
            const vbr_shared_scratch_binding & binding);
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
    // co-tenancy: end of the leading f16->t8 band of the order (demand sheds stop here);
    // 0 = no band (custom VBR_DEGRADE_ORDER carries no band guarantee -> demand shed off)
    size_t t8_band_end_ = 0;
    // peer-yield consent bound (Preston 2026-07-20, explicit-floor-as-consent): a TYPED
    // --vbr-floor (flag or VBR_MIN_BITS env) consents demand sheds down to the floor —
    // the ledger is per-uid, so the demander is the same human who typed it. A defaulted
    // floor keeps the conservative restorable band. 0 = demand shedding disabled.
    size_t vbr_demand_limit() const {
        if (t8_band_end_ == 0) {
            return 0;
        }
        return vbr_floor_typed_ ? vbr_degrade_limit_
                                : std::min(vbr_degrade_limit_, t8_band_end_);
    }
    bool vbr_floor_typed_ = false;
    // ---- co-tenancy donor state (P2) ----
    // grant rows: private in-memory liabilities recording a demand-shed's decrement,
    // keyed (pid, starttime, ver) with the demanded device's busid; one row per pool the
    // wave freed bytes in. Collateral rows (lockstep frees on non-demanded devices) carry
    // the full decrement until the lift event (delta_i = 0 — this also keeps the promote
    // cursor frozen so a promote cannot undo a lockstep shed).
    struct vbr_grant_row {
        std::string busid;      // device the demand named (claim file key)
        int32_t     pid;
        uint64_t    starttime;
        uint64_t    ver;
        size_t      pool_idx;
        uint64_t    bytes;
        uint64_t    bytes_now_at_grant; // staggered threshold: aggregate landed bytes consume rows once
        bool        collateral;
    };
    std::vector<vbr_grant_row> vbr_grants_;
    // reader-side heartbeat aging (shared llama_vram_hb_obs; claim keys "busid-pid",
    // marker keys "m-busid-pid")
    std::map<std::string, llama_vram_hb_obs> vbr_claim_obs_;
    // ledger scan pacing: dir-mtime pre-check baseline + last full-scan clock
    uint64_t vbr_ledger_mtime_  = 0;
    uint64_t vbr_last_scan_ns_  = 0;
    bool     vbr_ledger_force_  = false; // pre-check hit: run the full controller path
    // last published marker fields per busid (republish = rename only on change)
    std::map<std::string, std::pair<uint64_t, uint64_t>> vbr_marker_pub_; // {shed_avail, grant_pending}
    // Authoritative first-publish timestamps returned by successful marker publications,
    // plus per-device granted-but-not-yet-flushed bytes.
    std::map<std::string, uint64_t> vbr_marker_created_ts_;
    std::map<std::string, uint64_t> vbr_grant_pending_;

    // ---- P3 presence census ----
    // effective N_live per busid (self + live peer markers). Arrivals count immediately
    // (growing headroom is the safe direction); departures only after the raw count holds
    // for DEBOUNCE consecutive scan events (a GC'd marker of a crashed-and-restarting peer
    // must not flap the budgets). Promotes are presence-quiet gated on the change scan.
    struct vbr_presence { uint32_t cur = 0, raw = 0, stable = 0; };
    std::map<std::string, vbr_presence> vbr_presence_;
    uint32_t vbr_scan_events_          = 0;
    uint32_t vbr_nlive_change_scan_    = 0;
    uint32_t vbr_pool_n_live(const vbr_pool & p) const;
    bool     vbr_presence_quiet() const; // promote gate: no N_live change within DEBOUNCE scans

    // ---- P3 runtime-growth demand (idle-donor only) ----
    // a resident that spent its own consent window and is still over budget publishes a
    // phase=runtime claim; only donors idle >= IDLE honor it (active-vs-active residents
    // self-serve via their own ladders). CLEAR is demander-owned: the first boundary
    // where the recomputed shortage <= 0 unlinks (the donors' lift signal).
    uint64_t vbr_runtime_ver_      = 0;
    uint64_t vbr_last_prepare_ns_  = 0; // decode-based idleness input (ticks never update it)
    std::set<std::string> vbr_runtime_live_; // busids with a live runtime claim
    std::vector<size_t> vbr_pre_deficit_;    // per-pool pre-own-loop deficit (the honest ask)
    bool     vbr_runtime_was_over_ = false;  // current boundary's child-local pressure sample
    uint32_t vbr_runtime_wm_       = 0;
    void vbr_runtime_demand_update(uint32_t wm_next, bool was_over);

    void   vbr_ledger_precheck();                 // every boundary, outside the stable gate
    void   vbr_ledger_scan_service(uint32_t n_tokens); // composes the four phases below
    void   vbr_presence_census(const std::vector<llama_vram_peer_marker> & peers);
    bool   vbr_grants_upkeep(const std::vector<llama_vram_peer_claim> & claims, uint64_t now);
    bool   vbr_service_demands(const std::vector<llama_vram_peer_claim> & claims,
                               const std::vector<llama_vram_peer_marker> & peers,
                               const std::set<std::string> & announced,
                               uint64_t now, uint32_t n_tokens);
    void   vbr_grant_pending_clear();
    void   vbr_markers_publish(std::set<std::string> * changed = nullptr);
    void   vbr_maybe_promote(uint32_t wm_next); // gated promote step (boundary + tick)
    void   vbr_arm_wave_fences();               // arm fences for queued transcode waves
    vbr_pool * vbr_find_pool(const std::string & busid);
    void   vbr_apply_grant_decrements();          // recompute per-pool sums, bust memos
    size_t vbr_total_grant_decrement() const;     // promote freeze gate
    const std::string & vbr_pool_busid(vbr_pool & p) const;

    enum class vbr_tx_status {
        no_capacity,
        retryable_no_tier_mutation,
        committed,
    };
    struct vbr_tx_result {
        vbr_tx_status status = vbr_tx_status::no_capacity;
        uint64_t credited = 0;
    };
    bool vbr_ledger_owner_ = true;
    llama_kv_cache * vbr_ledger_root_ = nullptr;    // null means standalone/self
    llama_kv_cache * vbr_ledger_sibling_ = nullptr; // symmetric peer backlink in a composite
    double vbr_tree_device_share_ = 1.0;            // parent share before child normalization
    llama_kv_cache *       vbr_tree_root();
    const llama_kv_cache * vbr_tree_root() const;
    bool   vbr_tree_forced() const;
    void   vbr_tree_force();
    bool   vbr_tree_budget_explicit() const;
    size_t vbr_tree_total_grant_decrement() const;
    struct vbr_tree_pool_ref {
        llama_kv_cache * child = nullptr;
        vbr_pool * pool = nullptr;
    };
    vbr_tree_pool_ref vbr_tree_find_pool(const std::string & busid);
    size_t vbr_child_offer(const llama_kv_cache * child, const std::string & busid) const;
    size_t vbr_tree_offer(const std::string & busid) const;
    bool   vbr_tree_has_grant(const llama_vram_peer_claim & c) const;
    struct vbr_stash_request {
        const vbr_extent * extent = nullptr;
        uint32_t           rows   = 0;
    };

    struct vbr_tx_extent_key {
        size_t child_idx = 0;
        size_t pool_idx  = 0;
        size_t ikv       = 0;
        bool   is_v      = false;
        bool operator<(const vbr_tx_extent_key & other) const {
            return std::tie(child_idx, pool_idx, ikv, is_v) <
                   std::tie(other.child_idx, other.pool_idx, other.ikv, other.is_v);
        }
    };
    struct vbr_tx_pool_key {
        size_t child_idx = 0;
        size_t pool_idx  = 0;
        bool operator<(const vbr_tx_pool_key & other) const {
            return std::tie(child_idx, pool_idx) < std::tie(other.child_idx, other.pool_idx);
        }
    };
    struct vbr_tx_child {
        llama_kv_cache * cache = nullptr;
        uint32_t incoming_wm = 0;
        size_t start_cursor = 0;
        size_t final_cursor = 0;
        uint64_t start_epoch = 0;
        std::vector<ggml_type> types_before;
        std::vector<ggml_type> types_after;
    };
    struct vbr_tx_step {
        size_t child_idx = 0;
        size_t order_idx = 0;
        size_t slot = 0;
        size_t ikv = 0;
        bool is_v = false;
        ggml_type type_a = GGML_TYPE_COUNT;
        ggml_type type_b = GGML_TYPE_COUNT;
    };
    struct vbr_tx_endpoint {
        vbr_pool * pool = nullptr;
        vbr_extent * extent = nullptr;
        size_t final_span = 0;
        size_t slot_span = 0;
        uint64_t baseline_total = 0;
        uint64_t baseline_inside = 0;
        uint64_t gross_release = 0;
        bool touched = false;
        bool live = false;
    };
    struct vbr_tx_workspace_group {
        const ggml_vbr_backend_iface * be = nullptr;
        ggml_backend_t backend = nullptr;
        int device = -1;
        std::vector<llama_vbr_transaction::workspace_request> requests;
    };
    struct vbr_tx_stash_group {
        int device = -1;
        std::vector<vbr_stash_request> requests;
    };
    struct vbr_tx_scratch_group {
        const ggml_vbr_backend_iface * be = nullptr;
        ggml_backend_t backend = nullptr;
        int device = -1;
        size_t k_need = 0;
        size_t v_need = 0;
        uint64_t physical_projected = 0;
        bool owner_backend = false;
    };
    struct vbr_tx_grant_plan {
        size_t child_idx = 0;
        vbr_grant_row row;
    };
    struct vbr_shed_tx {
        int demanded_device = -1;
        uint64_t target = 0;
        std::vector<vbr_tx_child> children;
        std::vector<llama_vbr_policy::selection> policy_prefix;
        std::vector<vbr_tx_step> steps;
        std::map<vbr_tx_extent_key, vbr_tx_endpoint> endpoints;
        std::map<vbr_tx_pool_key, vbr_tx_workspace_group> workspaces;
        std::map<vbr_tx_pool_key, vbr_tx_stash_group> stashes;
        std::map<ggml_backend_t, vbr_tx_scratch_group> scratches;
        std::map<int, llama_vbr_transaction::device_cost> devices;
        std::map<vbr_tx_extent_key, uint64_t> endpoint_baseline_total;
        std::map<vbr_tx_extent_key, std::map<size_t, uint64_t>> endpoint_baseline_inside;
        std::map<vbr_tx_pool_key, uint64_t> workspace_baseline;
        std::map<vbr_tx_pool_key, uint64_t> stash_baseline;
        std::map<ggml_backend_t, uint64_t> scratch_baseline;
        std::map<vbr_tx_pool_key, uint64_t> gross_by_pool;
        std::map<vbr_tx_pool_key, uint64_t> deferred_by_pool;
        std::vector<vbr_tx_grant_plan> planned_grants;
        bool snapshot_open = true;
    };
    bool vbr_tx_settle_tree();
    bool vbr_tx_reprice(vbr_shed_tx & tx, bool actual) const;
    bool vbr_tx_preflight(vbr_shed_tx & tx);
    bool vbr_tx_map_endpoints(vbr_shed_tx & tx);
    bool vbr_tx_prepare_commit(vbr_shed_tx & tx, const llama_vram_peer_claim & c);
    bool vbr_tx_publish_zero_intent(const vbr_shed_tx & tx);
    void vbr_tx_apply(vbr_shed_tx & tx);
    void vbr_tx_suppress(const std::string & busid);
    vbr_tx_result vbr_execute_tree_shed(
            const llama_vram_peer_claim & c, uint64_t target, uint32_t n_tokens);

    // A retryable transaction failure temporarily withdraws this device's offer. The bound is
    // short enough to retry after physical capacity or marker state changes, but prevents every
    // scan from immediately selecting the same resident again.
    mutable std::map<std::string, uint64_t> vbr_offer_suppressed_until_;
    size_t vbr_floor_cost_bytes_ = 0;                 // page-exact cost of the floor layout at full
                                                      // kv_size (fallback budget in dynamic mode)
    bool   vbr_budget_warned_ = false;                // budget-unmeetable warning fired (terminal)
    // A recoverable ordinary boundary reserve failed during this boundary/tick. prepare() fails the
    // batch instead of executing over budget; idle breathe() retains the exact cursor and retries
    // on a later tick after physical capacity changes. Transaction retry suppression is separate.
    bool   vbr_reserve_failed_ = false;
    // prepare() boundaries since the last applied degrade step — promote cooldown basis
    // (deterministic, unlike wall time); promotes wait for a quiet window after any degrade
    uint32_t vbr_quiet_boundaries_ = 0;
    // auto-budget runtime re-derivation (explicit budgets never move): boundary counter (the
    // FIRST boundary is skipped — lazy cuBLAS/CUDA-graph pools have not allocated yet and free
    // overstates reality) + resolved growth headroom
    uint64_t vbr_boundary_count_   = 0;
    size_t   vbr_growth_headroom_  = 0;
    bool     vbr_budget_explicit_  = false;
    bool     vbr_budget_from_scalar_ = false;
    // what this pool's device can give it right now: device_share x (mapped + free - headroom),
    // 64 MiB-quantized. Shared by the init-time auto-budget arm (fit-less modes, e.g.
    // SPLIT_MODE_TENSOR) and the periodic re-derivation.
    size_t   vbr_pool_reach(const vbr_pool & p) const;
    // Fast-path stability tracking: skip per-batch VBR bookkeeping when settled (avoids ~1ms/token)
    uint32_t vbr_last_used_        = 0;   // observed cell count last prepare() pass
    uint32_t vbr_last_wm_          = 0;   // predicted padded watermark of last successful boundary
    void     vbr_rederive_budget();
    // sink-stash staleness guard: set when any cell below stash_rows is freed (its content can be
    // rewritten by another request; injecting the old snapshot would corrupt the new rows)
    bool   vbr_stash_dirty_   = false;
    void     vbr_full_reset();                        // cache empty: undo every degrade (lossless)
    void     vbr_shrink_watermark();                  // occupancy dropped: release phantom tail pages
    bool     vbr_promote_next(uint32_t wm_next);      // occupancy dropped: re-promote one container
    void     vbr_floor_clamp_order();
    bool     vbr_retire_pending_before_unmap(const std::string & busid);
    size_t   vbr_flush_deferred_unmaps(); // returns the number of entries flushed
    bool     vbr_scratch_reserve(size_t flat_cells);  // #88: boundary-time f16 dequant scratch grow
    // Pure, allocator-blind child stream used by the tree transaction. It derives real steps only
    // through vbr_sim_step(); physical pricing and preflight remain separate.
    llama_vbr_policy::child vbr_policy_child_stream(int demanded_device, uint32_t wm_next) const;
    // Pure physical projection for any set of captures in one pool. The first usable capture
    // requires the complete tightly-packed slab, matching the old allocation's fidelity lifetime;
    // current occupancy includes pages retained by earlier failed maps.
    bool     vbr_stash_memory(const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
                              size_t & physical_now, size_t & physical_if_reserved) const;
    // Idempotently reserve/map the same request set. false is recoverable and changes no tier or
    // stash-valid metadata; a partial VMM map may remain resident and is visible to the query.
    bool     vbr_stash_reserve(vbr_pool & p, const std::vector<vbr_stash_request> & requests);
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
    enum class vbr_degrade_result { applied, exhausted, reserve_failed };
    vbr_degrade_result vbr_degrade_next(uint32_t wm_next);
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
    // Scratch bindings for shared-KV aliases. Deliberately separate from vbr_pools_: a
    // controller-less drafter must not appear to own VMM, budget, ledger, or degrade state.
    std::vector<vbr_shared_scratch_binding> vbr_shared_scratch_bindings_;
    // Owner-side reverse registry plus drafter-side RAII registrations. Registration
    // records never own either context; weak registry links make target-first teardown safe.
    std::shared_ptr<vbr_shared_scratch_registry> vbr_shared_scratch_registry_;
    std::vector<vbr_shared_scratch_registration> vbr_shared_scratch_registrations_;
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
    llama_turbo_meansub_ref get_turbo_meansub_ref(int32_t il) const;


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
