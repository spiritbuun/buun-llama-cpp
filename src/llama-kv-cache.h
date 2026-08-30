#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-vbr-generation.h"
#include "llama-vbr-hard-seal.h"
#include "llama-vbr-downward.h"
#include "llama-vbr-upward.h"
#include "llama-vbr-policy.h"
#include "llama-vbr-transaction.h"

#include "ggml-vbr.h" // backend interface for turbo KV / dynamic VBR (resolved at init, never linked)
#include "llama-vram-ledger.h" // Co-tenancy peer claim and marker types.

#include <array>
#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;
class vbr_unit_build;
class vbr_pinned_chunk_ring;
class vbr_kv_import_session;
struct vbr_validated_child_plan;
struct vbr_target_unit_snapshot;
class vbr_import_receipt_group;
struct vbr_capture_stream_stats;
struct vbr_capture_projected_shard_source;
struct vbr_capture_unit_snapshot;
struct vbr_capture_unit_snapshot_provider;
enum class vbr_explicit_generation_failure : uint8_t;
enum class vbr_explicit_size_failure : uint8_t;
struct vbr_artifact_stream_placement;
struct vbr_artifact_unit_descriptor;

//
// llama_kv_cache
//

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
        const llama_memory_vbr_params & vbr = {},
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "");

    // Compatibility overload for fixed caches that only supply a name tag.
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
        const  layer_share_cb & share,
                 const char *   name_tag);

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
    bool can_seq_rm_partial() const override { return true; }

    llama_memory_vbr_representation_identity
    vbr_representation_identity() const override {
        return { vbr_tier_epoch(), 0 };
    }

    void breathe() override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    bool seq_rm_transient(llama_seq_id seq_id,                       llama_pos p0, llama_pos p1) override;
    bool seq_rm_attn_transient(llama_seq_id seq_id,                  llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool try_seq_cp_transient(
            llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
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

    // Checkpoint-facing semantic counter: unlike the graph tier fence, this also covers
    // clear/reset/import. It never resets, so cursor rewind cannot create an ABA.
    uint64_t vbr_representation_epoch() const {
        return other ? other->vbr_representation_epoch() : vbr_representation_epoch_;
    }

    // Checkpoint-facing attention-content lineage. The no-argument value is the cache-wide
    // mutation serial used for capacity/accounting checks; a concrete sequence observes only
    // mutations that can change that sequence's attention rows. In-place retiering preserves
    // both. Global clear/reset/import advances every sequence, so recurrent-only checkpoints
    // cannot survive an ABA while unrelated slot traffic no longer invalidates them.
    uint64_t vbr_checkpoint_epoch(llama_seq_id seq_id = -1) const {
        if (other) {
            return other->vbr_checkpoint_epoch(seq_id);
        }
        return seq_id >= 0 && seq_id < LLAMA_MAX_SEQ
            ? vbr_checkpoint_seq_epochs_[size_t(seq_id)]
            : vbr_checkpoint_epoch_;
    }

    // Adapter over the cache's canonical dependency index used by
    // explicit VBR artifact capture.
    bool vbr_generation_capture_live_guarded(
            uint32_t child_id,
            llama_seq_id seq_id,
            llama_pos computation_frontier,
            vbr_checkpoint_generation_controller & output,
            vbr_artifact_stream_placement * placement = nullptr,
            vbr_explicit_generation_failure * failure = nullptr) const;
    // effective bits/value of this cache at the CURRENT tensor types (llama_memory_i)
    double kv_bpv() const override;

    llama_memory_vbr_state_data_v2 memory_vbr_state_v2(
            llama_seq_id seq_id, uint32_t n_tokens_extra) const override;
    bool vbr_accumulate_exclusive_cells(
            uint32_t * counts, size_t size) const override;
    bool vbr_capture_readiness_cells(
            uint64_t logical_growth,
            uint64_t & committed,
            uint64_t & projected,
            uint64_t & capacity) const override;
    bool vbr_operation_armed() const override;
    // Boundary service: true while this cache's tracker is latched unavailable or its pool
    // has unresolved recovery-ring work. The update context reports an update NEEDED in this
    // state so the quarantine drain + monotone re-arm in update() actually run at quiet decode
    // boundaries (a NO_UPDATE short-circuit would starve recovery until an unrelated shift).
    bool vbr_recovery_service_pending() const;
    bool vbr_retier_freeze_begin(const char * owner, vbr_operation_id operation_id) override;
    void vbr_retier_freeze_end(const char * owner, vbr_operation_id operation_id) override;
    llama_memory_vbr_preflight_data vbr_retier_preflight(
        uint32_t n_tokens_extra,
        std::vector<llama_memory_vbr_physical_growth> * physical = nullptr) const override;
    bool vbr_retier_freeze_active() const {
        return other ? other->vbr_retier_freeze_active() : vbr_retier_freeze_depth_ > 0;
    }
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

    // Read-only policy classification. Enforcement remains an approved-guard
    // controller concern; this hook cannot change controller decisions.
    bool vbr_hard_seal_classify(
        vbr_hard_seal_classification & out) const noexcept;
    void vbr_hard_seal_guard_set(vbr_hard_seal_guard guard) override;
    bool vbr_hard_seal_blocked_take(bool decode_failed) override;
    void vbr_hard_seal_evidence_take(
            std::vector<vbr_hard_seal_subject> & out) override;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    llama_turbo_meansub_ref get_turbo_meansub_ref(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

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

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    friend class vbr_live_capture_adapter;
    friend class vbr_kv_import_session;
    friend struct llama_kv_cache_vbr_stash_batch_test;
    struct vbr_capture_unit_request {
        uint32_t child_id = 0;
        const void * bindings = nullptr;
    };
    struct vbr_capture_unit_plan {
        struct shard {
            void * pool = nullptr;
            void * extent = nullptr;
            uint32_t shard_index = 0;
            uint32_t topology_index = UINT32_MAX;
            uint16_t device_ordinal = UINT16_MAX;
            uint32_t lane = UINT32_MAX;
            uint64_t payload_bytes = 0;
            uint64_t row_bytes = 0;
            uint64_t columns = 0;
            uint64_t stash_bytes = 0;
        };
        uint32_t child_id = 0;
        uint32_t logical_unit = 0;
        uint32_t capture_index = UINT32_MAX;
        bool is_v = false;
        llama_turbo_meansub_ref meansub_ref;
        vbr_unit_generation generation;
        uint32_t n_stream = 0;
        bool unified = false;
        uint32_t wm_cells = 0;
        std::vector<shard> shards;
    };
    struct vbr_capture_stability_token {
        struct geometry {
            uint32_t logical_unit = 0;
            vbr_unit_generation generation;
            uint32_t wm_cells = 0;
            std::vector<const ggml_tensor *> tensors;
            std::vector<size_t> byte_offsets;
            std::vector<uint32_t> stash_valid;
            std::vector<size_t> stash_offsets;
        };
        vbr_lineage_uuid lineage_uuid;
        vbr_controller_instance_id instance_id;
        uint64_t controller_generation = 0;
        uint64_t mutation_serial = 0;
        std::array<uint8_t, 32> degrade_order_digest = {};
        std::array<uint8_t, 32> policy_digest = {};
        uint64_t degrade_cursor = 0;
        int32_t floor_type = -1;
        uint64_t pressure_independent_settings = 0;
        bool completed_wave = false;
        std::vector<geometry> units;
    };

    // One synchronous projected transfer owns this stack-scoped callback
    // context. The session borrows its immutable size-pass plan and owns the
    // canonical sources prepared before the unit reader lease. acquire() and
    // recheck() validate those sources in place without allocating while the
    // lease is held; release() is idempotent as a defensive terminal.
    struct vbr_capture_snapshot_session {
        const llama_kv_cache * cache = nullptr;
        const vbr_capture_unit_plan * plan = nullptr;
        std::vector<vbr_capture_projected_shard_source> sources;
        uint64_t source_namespace = 0;
        uint32_t shard_count = 0;
        std::array<uint8_t, 32> shard_topology_digest = {};
        bool active = false;

        vbr_capture_snapshot_session() = default;
        ~vbr_capture_snapshot_session();
        vbr_capture_snapshot_session(const vbr_capture_snapshot_session &) = delete;
        vbr_capture_snapshot_session & operator=(const vbr_capture_snapshot_session &) = delete;
        vbr_capture_unit_snapshot_provider provider() noexcept;
    };

    bool vbr_capture_settle() noexcept;
    bool vbr_capture_size_pass(
        const vbr_capture_unit_request & request,
        std::vector<vbr_capture_unit_plan> & output,
        vbr_capture_stability_token & stability,
        vbr_explicit_size_failure * failure = nullptr) const noexcept;
    bool vbr_capture_stream_unit(
        const vbr_capture_unit_plan & plan,
        vbr_unit_build & sink,
        vbr_pinned_chunk_ring & ring,
        vbr_capture_stream_stats & stats,
        void * continue_context = nullptr,
        bool (*continue_transfer)(void * context) noexcept = nullptr)
        const noexcept;
    // Converts one exact size-pass unit into the process-local backend
    // capabilities consumed by sequence-projected capture. This is a
    // preparation seam only: the later snapshot provider must still acquire
    // the unit-version lease before any source is read.
    bool vbr_capture_projected_sources(
        const vbr_capture_unit_plan & plan,
        std::vector<vbr_capture_projected_shard_source> & output) const noexcept;
    bool vbr_capture_projected_sources_leased(
        const vbr_capture_unit_plan & plan,
        const std::vector<vbr_capture_projected_shard_source> & expected,
        const vbr_capture_snapshot_session & session) const noexcept;
    bool vbr_capture_projected_sources_impl(
        const vbr_capture_unit_plan & plan,
        std::vector<vbr_capture_projected_shard_source> * output,
        const std::vector<vbr_capture_projected_shard_source> * expected,
        bool unit_leased) const noexcept;
    bool vbr_capture_snapshot_bind(
        const vbr_capture_unit_plan & plan,
        const std::vector<vbr_capture_projected_shard_source> & sources,
        uint64_t source_namespace,
        vbr_capture_snapshot_session & output) const noexcept;
    static bool vbr_capture_snapshot_acquire(
        void * context, uint64_t source_namespace, uint32_t child_id,
        uint32_t logical_unit_id, vbr_capture_unit_snapshot & output) noexcept;
    static bool vbr_capture_snapshot_recheck(
        void * context, const vbr_capture_unit_snapshot & expected) noexcept;
    static void vbr_capture_snapshot_release(
        void * context, const vbr_capture_unit_snapshot & snapshot) noexcept;
    bool vbr_capture_stability_matches(
        const vbr_capture_stability_token & token) const noexcept;
    bool vbr_capture_generation_record(
        uint32_t child_id,
        checkpoint_child_dependency_mode dependency_mode,
        llama_seq_id sequence,
        llama_pos frontier,
        vbr_checkpoint_generation_controller & output,
        vbr_artifact_stream_placement * placement = nullptr,
        vbr_explicit_generation_failure * failure = nullptr) const noexcept;
    bool vbr_capture_policy_snapshot(
        vbr_capture_stability_token & output) const noexcept;
    bool vbr_import_transform_reserve(
        const std::vector<const vbr_validated_child_plan *> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept;
    bool vbr_downward_policy_input(
        const std::vector<ggml_type> & source_types,
        uint64_t source_cursor,
        uint32_t projected_wm_cells,
        int demanded_device,
        vbr_downward_policy_child & output) const noexcept;
    bool vbr_import_destination_input(
        uint32_t projected_wm_cells,
        vbr_import_destination_child & output) const noexcept;
    struct vbr_import_destination_pricing {
        struct pool_row {
            const ggml_vbr_backend_iface * be = nullptr;
            int device = -1;
            uint64_t mapped = 0;
            uint64_t available = 0;
            uint64_t needed = 0;
        };
        struct device_row {
            const ggml_vbr_backend_iface * be = nullptr;
            int device = -1;
            uint64_t available = 0;
            uint64_t scratch_k_needed = 0;
            uint64_t scratch_v_needed = 0;
            uint64_t scratch_k_current = 0;
            uint64_t scratch_v_current = 0;
        };
        uint32_t watermark_cells = 0;
        bool active = false;
        bool overflow = false;
        std::vector<ggml_type> types;
        std::vector<pool_row> pools;
        std::vector<device_row> devices;
    };
    bool vbr_import_destination_pricing_begin(
        const std::vector<ggml_type> & types,
        uint32_t projected_wm_cells,
        vbr_import_destination_pricing & output) const noexcept;
    bool vbr_import_destination_pricing_apply(
        const llama_vbr_policy::step & step,
        vbr_import_destination_pricing & pricing) const noexcept;
    llama_memory_vbr_preflight_data vbr_import_destination_preflight(
        const vbr_import_destination_pricing & pricing,
        std::vector<llama_memory_vbr_physical_growth> * physical) const noexcept;
    llama_memory_vbr_preflight_data vbr_import_destination_preflight(
        const std::vector<ggml_type> & types,
        uint32_t projected_wm_cells,
        std::vector<llama_memory_vbr_physical_growth> * physical) const noexcept;
    bool vbr_policy_priced_steps(
        std::vector<ggml_type> & sim, size_t start_cursor,
        int demanded_device, uint32_t watermark, bool fixed_watermark,
        bool fail_closed, llama_vbr_policy::child & output,
        vbr_hard_seal_consult_session * seal_session = nullptr) const;
    bool vbr_import_bind_target_unit(
        const vbr_artifact_unit_descriptor & source,
        ggml_type target_type,
        const vbr_upward_representation_identity & selected_source_identity,
        const vbr_upward_representation_identity & selected_target_identity,
        const vbr_downward_policy_projection & projection,
        uint32_t projection_child,
        vbr_target_unit_snapshot & output) const noexcept;
    vbr_downward_transform_status vbr_downward_transform_import(
        const vbr_validated_child_plan & plan,
        bool stashless,
        uint32_t & stash_valid,
        uint32_t & edge_reached) noexcept;
    bool vbr_upward_transform_import(
        const vbr_validated_child_plan & plan) noexcept;
    bool vbr_import_source_alias(
        const ggml_tensor & destination,
        ggml_type source_type,
        ggml_tensor & output) const noexcept;
    void vbr_import_set_unit_type_noalloc(
        uint32_t logical_unit, ggml_type type) noexcept;

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

    // Per-(layer,side) descriptor over the shared dynamic-VBR KV pool buffer. Tier is not
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
        // Optional versioned capability. Kept separate from the legacy object so a backend built
        // before cross-domain reconstruction can still be queried without an out-of-bounds read.
        const ggml_vbr_cross_domain_iface_v1 * cross_be = nullptr;
        // non-owning main compute backend whose context owns the fattn Q/K/V scratch. This is
        // intentionally distinct from `backend` below, which is a dedicated transcode stream.
        // Valid only while llama_context::backends is alive. llama_context declares `memory`
        // before `backends`, so backends are destroyed first: KV teardown must never dereference
        // this handle (runtime reserve calls happen while the complete context is alive).
        ggml_backend_t compute_backend = nullptr;
        // VMM-backed pool: per-tensor fixed VA slots, physical pages mapped on
        // demand. When set, `size` is the VA reservation (not physical); each extent's byte_off is
        // page-aligned so tensor-tail unmaps never straddle a neighbor's pages.
        struct ggml_vbr_vmm_pool * vmm = nullptr;
        uint32_t wm_cells    = 0;                 // cells already backed for every extent
        int      device      = -1;                // backend device ordinal backing the pool
        size_t   gran        = 0;                 // page granularity
        size_t   mapped_base = 0;                 // bytes mapped up front (rotation matrices)
        // Scratch-reserve memo: widest f16 row per dequant-active side, valid while no tier
        // flips (keyed on vbr_tier_epoch_; ~0 forces the first compute)
        uint64_t scratch_rows_epoch = ~0ull;
        size_t   scratch_k_row      = 0;
        size_t   scratch_v_row      = 0;
        size_t   scratch_k_reserved = 0; // largest successful backend-global reserve requested here
        size_t   scratch_v_reserved = 0;
        // Co-tenancy PCI bus id, resolved once from the backend device; empty means none.
        // and the summed unamortized grant decrement vbr_budget_eff subtracts
        std::string busid;
        mutable size_t grant_decrement = 0;
        // Per-device transcode side stream (lazy): transcodes run asynchronously on the
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
        // Persistent transform workspace/stash receipts have exactly the owning
        // pool/side-backend lifetime.  The ledger remains the charge-once
        // authority; this holder owns only the committed C references.
        std::unique_ptr<vbr_downward_resource_receipts> transform_receipts;
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

    // Decode-time degrade controller (VMM mode only). The price order and its cursor stay
    // GLOBAL (layer-global price order); each step resolves the pool that owns its tensor.
    llama_memory_vbr_params vbr_params_;              // API/CLI inputs (ctor copy; env can override)
    // bumped on every in-place tier flip (degrade/promote/full reset). Graph reuse must be
    // fenced on it: a reused graph carries the OLD type/strides baked into its K/V views, and
    // a free-VRAM-clamp wave (or a promote map-retry) can flip tiers MID-band where the n_kv
    // shape check alone would still allow reuse.
    uint64_t vbr_tier_epoch_ = 0;
    // Bumped once per representation-changing operation (retier, attention sequence edit,
    // occupied-cell reuse, clear/full-reset, native state import). Never derive this from or
    // reset it with the cursor.
    uint64_t vbr_representation_epoch_ = 0;
    // Bumped only when the attention-content lineage changes. A tier transcode changes storage
    // representation but preserves every logical KV row, so it must not invalidate a
    // recurrent-only checkpoint paired with that live attention prefix.
    uint64_t vbr_checkpoint_epoch_ = 0;
    std::array<uint64_t, LLAMA_MAX_SEQ> vbr_checkpoint_seq_epochs_ = {};
    // VBR generation-tracker shadow generations. Allocated only for a construction-final armed VBR
    // controller; aliases delegate to their canonical owner and inert caches allocate nothing.
    // Checkpoint reads consult this store only through the generation authority.
    std::unique_ptr<vbr_generation_tracker> vbr_generation_;
    // Dual-view ownership index: updated in the same registrant transactions that stamp the
    // tracker; capture consumes rank_below for scan-free exact dependency cardinality.
    std::unique_ptr<vbr_ownership_index>    vbr_ownership_;
    // Import receipts keep every committed adoption/staging reference
    // accounted until reset/retirement. This is the fail-closed side of the
    // accounting handoff: no release-first window can exist before a live producer
    // gains an explicit serial-bound acknowledgement seam.
    // One receipt group is shared by every attention child in a composite
    // import. It releases only after the LAST child resets/retires, so a
    // sequential tree clear cannot expose an unaccounted still-live sibling.
    std::shared_ptr<vbr_import_receipt_group> vbr_import_receipt_;
    bool vbr_import_in_progress_ = false;
    vbr_operation_id vbr_import_operation_ = {};
    void vbr_import_receipts_release() noexcept;
    void vbr_import_receipts_release_if_empty() noexcept;
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
    // peer-yield consent bound (buun 2026-07-20, explicit-floor-as-consent): a TYPED
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
    // ---- co-tenancy donor state ----
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

    // ---- presence census ----
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

    // ---- runtime-growth demand (idle-donor only) ----
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
        std::vector<size_t> sealed_deferred;
        std::vector<size_t> capture_deferred;
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
    class vbr_unit_retier_guard {
    public:
        vbr_unit_retier_guard() = default;
        vbr_unit_retier_guard(llama_kv_cache * cache, uint32_t logical_unit) noexcept;
        ~vbr_unit_retier_guard();
        vbr_unit_retier_guard(const vbr_unit_retier_guard &) = delete;
        vbr_unit_retier_guard & operator=(const vbr_unit_retier_guard &) = delete;
        vbr_unit_retier_guard(vbr_unit_retier_guard && other) noexcept;
        vbr_unit_retier_guard & operator=(vbr_unit_retier_guard && other) noexcept;
        explicit operator bool() const noexcept { return active_; }
    private:
        llama_kv_cache * cache_ = nullptr;
        uint32_t logical_unit_ = UINT32_MAX;
        bool active_ = false;
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
        std::vector<vbr_unit_retier_guard> unit_guards;
        bool snapshot_open = true;
    };
    bool vbr_tx_settle_tree();
    bool vbr_tx_reprice(vbr_shed_tx & tx, bool actual) const;
    bool vbr_tx_hard_seal_allowed(vbr_shed_tx & tx);
    bool vbr_tx_capture_leases_allowed(vbr_shed_tx & tx);
    bool vbr_tx_preflight(vbr_shed_tx & tx);
    bool vbr_tx_map_endpoints(vbr_shed_tx & tx);
    bool vbr_tx_prepare_commit(vbr_shed_tx & tx, const llama_vram_peer_claim & c);
    bool vbr_tx_publish_zero_intent(const vbr_shed_tx & tx);
    void vbr_tx_apply(vbr_shed_tx & tx, vbr_operation_id operation_id);
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
    vbr_hard_seal_guard vbr_hard_seal_guard_;
    bool vbr_hard_seal_blocked_ = false;
    std::vector<vbr_hard_seal_subject> vbr_hard_seal_evidence_;
    std::vector<size_t> vbr_hard_seal_deferred_;
    std::vector<uint8_t> vbr_hard_seal_attempted_;
    std::vector<size_t> vbr_capture_retier_deferred_;
    std::vector<uint8_t> vbr_capture_retier_attempted_;
    uint64_t vbr_capture_retier_attempt_boundary_ = UINT64_MAX;
    std::vector<uint64_t> vbr_capture_unit_attempt_boundary_;
    bool vbr_hard_seal_step_blocked(
            size_t order_ordinal,
            vbr_hard_seal_consult_session & session) const;
    void vbr_hard_seal_evidence_record(size_t order_ordinal);
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
    // Deterministic freeze mode — env VBR_FREEZE, TEST/GATING ONLY. Neutralizes the two
    // live-VRAM / co-tenancy inputs that make the tier schedule irreproducible run-to-run: the
    // vbr_budget_eff clamp (live cudaMemGetInfo) and the ledger scan/precheck + wall-clock gates,
    // so the schedule becomes a pure function of the fixed budget + occupancy. Requires an explicit
    // VBR_BUDGET_MIB (else vbr_pool_reach re-derivation, which is !explicit-gated, is not frozen).
    // OFF => every gated branch runs verbatim: a freeze-off build is bit-identical to a pre-freeze
    // build. Never a production degrade-policy lever.
    bool     vbr_freeze_           = false;
    // Acceptance-only companion to VBR_FREEZE. The routed downward gate
    // needs an empty target that retains the naturally selected tier vector;
    // absent this exact env, empty boundaries retain their shipped full reset.
    bool     vbr_freeze_preserve_empty_tiers_ = false;
    // Production-scoped freeze of representation mutation. Orthogonal to the
    // deterministic-input freeze above: nesting never changes the ledger/presence machinery.
    struct vbr_retier_freeze_frame {
        vbr_operation_id operation_id = {};
        uint64_t started_ns = 0;
    };
    static constexpr size_t VBR_RETIER_FREEZE_MAX_DEPTH = 64;
    std::array<vbr_retier_freeze_frame, VBR_RETIER_FREEZE_MAX_DEPTH> vbr_retier_freeze_stack_ = {};
    uint32_t vbr_retier_freeze_depth_       = 0;
    uint64_t vbr_retier_freeze_enters_      = 0;
    uint64_t vbr_retier_freeze_exits_       = 0;
    uint64_t vbr_retier_deferred_decisions_ = 0;
    uint64_t vbr_retier_reconciles_         = 0;
    uint64_t vbr_retier_outer_deferred_base_ = 0;
    bool     vbr_retier_reconcile_pending_  = false;
    struct vbr_capture_unit_lease_state {
        uint32_t readers = 0;
        uint64_t mutation_serial = 0;
        bool writer = false;
        bool mutation_deferred = false;
    };
    mutable std::mutex vbr_capture_unit_leases_mutex_;
    mutable std::vector<vbr_capture_unit_lease_state> vbr_capture_unit_leases_;
    mutable bool vbr_capture_controller_writer_ = false;
    mutable std::atomic<bool> vbr_capture_reconcile_pending_ { false };
    bool vbr_capture_unit_read_begin(uint32_t logical_unit) const noexcept;
    bool vbr_capture_unit_read_serial(
        uint32_t logical_unit, uint64_t & output) const noexcept;
    void vbr_capture_unit_read_end(uint32_t logical_unit) const noexcept;
    bool vbr_capture_watermark_contains(
        const vbr_pool & pool, uint32_t planned) const noexcept;
    void vbr_capture_watermark_publish(
        vbr_pool & pool, uint32_t value) noexcept;
    bool vbr_capture_unit_write_begin(uint32_t logical_unit) noexcept;
    bool vbr_capture_unit_write_plan_available(uint32_t logical_unit) const noexcept;
    void vbr_capture_unit_write_end(uint32_t logical_unit) noexcept;
    bool vbr_capture_controller_write_begin() noexcept;
    void vbr_capture_controller_write_end() noexcept;
    bool     vbr_retier_defer(const char * decision);
    bool     vbr_retier_take_reconcile(const char * boundary);
    // Schedule-trace recorder — env VBR_TRACE=<path>, TEST/GATING ONLY. One line per
    // boundary: phase, boundary#, degrade cursor, tier-vector FNV digest, watermark, used cells,
    // mapped bytes. The disabled-controller control needs two runs proven schedule-identical
    // (not merely output-identical);
    // this makes the schedule diffable and localizes the first divergent boundary. null => no-op.
    // RAII ensures a throwing constructor after open still closes the handle during unwinding.
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> vbr_trace_fp_{nullptr, &std::fclose};
    void     vbr_trace_emit(const char * phase, uint32_t wm, uint32_t used);
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
    enum class seq_rm_mode : uint8_t {
        public_commit,
        nested_commit,
        dry_run,
    };
    void     vbr_full_reset();                        // cache empty: undo every degrade (lossless)
    void     vbr_representation_changed();             // monotone representation change detector
    void     vbr_attention_content_changed();          // global content mutation
    void     vbr_attention_content_changed(llama_seq_id seq_id); // one sequence
    void     vbr_attention_content_changed(
            const std::array<bool, LLAMA_MAX_SEQ> & affected); // one atomic multi-sequence edit
    bool     seq_rm_impl(llama_seq_id seq_id, llama_pos p0, llama_pos p1, seq_rm_mode mode);
    void     seq_cp_impl(
            llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
            llama_pos p0, llama_pos p1, bool publish_lineage);
    vbr_generation_tracker *       vbr_generation_tracker_mut();
    const vbr_generation_tracker * vbr_generation_tracker_get() const;
    static bool vbr_generation_cell_has_seq_cb(
            const void * context, uint32_t stream, uint32_t cell, llama_seq_id seq_id);
    static llama_pos vbr_generation_cell_pos_cb(
            const void * context, uint32_t stream, uint32_t cell);

    // Explicit mutation-operation binding: every mutation entry point opens one scope carrying
    // its authenticated multi-target manifest. The scope registers the operation and — for
    // provenance-bearing mutations — reserves the recovery record BEFORE any mutation; damage
    // Extents reserve lazily per selected target at the first destructive stamp.
    // Events minted while the scope is open cite its operation id. Close follows the
    // Per-family commit-boundary table: synchronous families commit at scope
    // end; deferred families transfer everything to the pending owner via detach_deferred().
    // Decode operations stay open past apply_ubatch and close only when the
    // decode outcome is known. One entry per in-flight committed ubatch.
  public:
    // Fixed parent-declared participant slots with a sealed-registration phase.
    // The parent declares every armed child before the first apply; each child claims its
    // slot in its scope constructor (before mutation), and the slot reports terminal EXACTLY
    // once — setup/decode/submit failure, or synchronize-time delivery. Detach transfers the
    // still-OPEN token to pending ownership (never terminal). seal() folds the wrapper
    // result, fails any never-claimed declared slot, and seals registration; only
    // `sealed && every declared slot terminal` closes the root, failure dominating. No
    // dynamic remaining++ anywhere — the v5 premature-close class is unrepresentable.
    // (Public: the iSWA wrapper constructs it; methods live in llama-kv-cache.cpp so the
    // registry close stays in that trust domain.)
    struct vbr_composite_outcome {
        vbr_operation_id operation_id = {};
        int32_t          declared     = 0;
        int32_t          claimed      = 0;
        int32_t          terminal     = 0;
        bool             sealed       = false;
        bool             failed       = false;
        bool             closed       = false;

        void claim();
        void report_terminal(bool ok);
        void seal(bool wrapper_ok);

      private:
        void try_close();
    };

  private:
    struct vbr_pending_decode_op {
        vbr_operation_id  operation_id   = {};
        // Per-target damage extents: submit, commit, fail, and recovery cover
        // every handle; each cell cited its SELECTED target's extent at stamp time.
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> extents = {};
        int32_t           recovery_index = -1;
        // Single-cache ops close directly (owns_close). Composite children instead report
        // their terminal result into the shared sealed aggregate, which closes the root.
        bool              owns_close     = true;
        std::shared_ptr<vbr_composite_outcome> composite;
    };

    class vbr_mutation_op {
      public:
        vbr_mutation_op(llama_kv_cache *    cache,
                        vbr_operation_kind  kind,
                        vbr_operation_class operation_class,
                        llama_seq_id        seq_id,
                        llama_pos           p0,
                        llama_pos           p1,
                        bool                provenance_bearing = false,
                        uint16_t            extent_stream      = 0);
        // Multi-target form (decode composites): the caller supplies the full manifest.
        vbr_mutation_op(llama_kv_cache *              cache,
                        const vbr_operation_binding & binding,
                        bool                          provenance_bearing);
        ~vbr_mutation_op();

        vbr_mutation_op(const vbr_mutation_op &)             = delete;
        vbr_mutation_op & operator=(const vbr_mutation_op &) = delete;
        vbr_mutation_op(vbr_mutation_op &&)                  = delete;
        vbr_mutation_op & operator=(vbr_mutation_op &&)      = delete;

        bool active() const { return static_cast<bool>(operation_id_); }
        std::optional<vbr_pending_decode_op> detach_deferred();
        // Per-target lazy extent, reserved at the first destructive stamp that
        // SELECTS manifest target `target_index` (the tracker calls through the trampoline).
        // Idempotent per target; empty on reservation failure (availability path taken).
        vbr_extent_handle ensure_extent_for(uint8_t target_index);
        static vbr_extent_handle extent_trampoline(void * ctx, uint8_t target_index);
        // A refused or unauthorized stamp poisons the whole logical operation: failure
        // Ownership follows the same root link as extent ownership, so a poison
        // in a joined child fails the root: it reports FAILED (into its aggregate for
        // composite children, at its close for owned scopes) and its recovery reservation
        // survives to quarantine through the failed close's autorecord.
        void poison() {
            poisoned_ = true;
            if (extent_owner_ != nullptr && extent_owner_ != this) {
                extent_owner_->poison();
            }
        }
        // v3.1 amendment 4: explicit success required — destruction without succeed() closes
        // the operation FAILED (exception unwind and forgotten paths fail by construction).
        // A poisoned scope can never succeed.
        void succeed() {
            if (!poisoned_) {
                succeeded_ = true;
            }
        }

        // For the always-succeed metadata-edit family: ONE opt-in per function. Succeeds at
        // scope exit UNLESS an exception entered flight — the default-fail pin holds on
        // unwind while every normal return path stops hand-spelling succeed().
        class success_on_return {
          public:
            explicit success_on_return(vbr_mutation_op & op)
                : op_(op), exceptions_at_entry_(std::uncaught_exceptions()) {}
            ~success_on_return() {
                if (std::uncaught_exceptions() == exceptions_at_entry_) {
                    op_.succeed();
                }
            }
          private:
            vbr_mutation_op & op_;
            int               exceptions_at_entry_;
        };

      private:
        void abort_to_shadow_unavailable();
        void fail_extents();
        // Owned scopes read their manifest from the RAII (which retains the identical
        // binding); only adopted scopes hold their own registry-fetched copy.
        const vbr_operation_binding & scope_manifest() const {
            return owned_op_ ? owned_op_->binding() : manifest_;
        }

        friend class llama_kv_cache;
        llama_kv_cache *      cache_          = nullptr;
        vbr_mutation_op *     outer_          = nullptr;
        // The root scope owning the per-target extents this chain stamps against; joined
        // scopes point at their root so the tracker's extent callback lands there.
        vbr_mutation_op *     extent_owner_   = nullptr;
        // Minting scopes own a registry operation; joining scopes (nested/adopted) borrow the
        // outer/adopted id and own nothing.
        std::optional<vbr_scoped_operation> owned_op_;
        vbr_operation_id      operation_id_   = {};
        vbr_operation_kind    kind_           = vbr_operation_kind::sequence_edit;
        // Adopted scopes' authenticated manifest copy (owned scopes read the
        // RAII's retained binding via scope_manifest()); one lazy extent per target.
        vbr_operation_binding manifest_       = {};
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> extents_ = {};
        // Participant aggregate in which this adopted child claimed a slot.
        std::shared_ptr<vbr_composite_outcome> composite_;
        int32_t               recovery_index_ = -1;
        bool                  succeeded_      = false;
        bool                  poisoned_       = false;
        bool                  detached_       = false;   // token transferred to pending owner
        bool                  joined_         = false;   // nested: borrows outer identity fully
        bool                  adopted_        = false;   // Shared id; owns reservations.
    };
    friend class vbr_mutation_op;
    // The innermost open mutation scope; vbr_generation_begin cites it.
    vbr_mutation_op * vbr_current_mutation_ = nullptr;
    // An operation adopted from a composite wrapper: child mutation scopes
    // join it instead of minting, so iSWA/hybrid children share ONE id per logical mutation.
    vbr_operation_id vbr_adopted_operation_ = {};
    // The composite root's mint was refused: child scopes open refused (fail
    // closed to shadow-unavailable) instead of minting independently (operation registry one-id in refusal).
    bool vbr_adopted_refused_ = false;
    // The composite aggregate that adopted children report their terminal
    // results into (set only for deferred/decode composites).
    std::shared_ptr<vbr_composite_outcome> vbr_adopted_composite_;
    // Decode operations stay open past apply_ubatch and close only when the
    // decode outcome is known. One entry per in-flight committed ubatch.
    std::vector<vbr_pending_decode_op> vbr_pending_decode_ops_;
    // Records whose extents are `submitted`, awaiting terminal delivery at the sync fence.
    // Their registry operations and recovery reservations remain OPEN until then.
    std::vector<vbr_pending_decode_op> vbr_awaiting_commit_;
    uint64_t vbr_pending_commit_failures_ = 0;  // sync-boundary commit failures, counted

  public:
    // Promote submitted extents to committed. Called from the context's existing synchronize
    // point — never introduces a new fence (Rev 5.1). No-op when nothing is pending.
    void vbr_commit_submitted();
    // Resolve in-flight decode operations at the decode boundary where the
    // outcome is known. ok=true: extents -> submitted, ops close committed. ok=false: extents
    // fail, ops close failed (autorecording their reserved recovery slots).
    void vbr_decode_ops_finish(bool ok);
    // Composite adoption: wrappers mint once and adopt into children.
    void vbr_adopt_operation(vbr_operation_id operation_id);
    void vbr_adopt_composite(std::shared_ptr<vbr_composite_outcome> composite);
    void vbr_adopt_refused();
    void vbr_release_adopted();
    // Decode manifest bound to the ubatch's actual sequences:
    // per touched seq an ordinary target over its exact position range, plus (when wrapping
    // is possible) a whole-range swa_wrap target per seq and ONE declared seq-wildcard
    // whole-range state_api target authorizing the nested §7.3 prefix purge: cross-sequence
    // masked reuse makes the destroyed position and the purged old owner unbounded by the
    // batch, and the owner is chosen by slot selection AFTER an adopted manifest has minted.
    // TRANSACTIONAL: any target-ceiling overflow zeroes the manifest and returns false — the
    // registry then refuses the mint (fail-closed shadow-unavailable), never partial.
    static bool vbr_decode_targets_from_ubatch(vbr_operation_binding & binding,
                                               vbr_controller_instance_id instance,
                                               bool wrap_possible, uint16_t stream,
                                               const llama_ubatch & ubatch);
    // Durable trajectory identity for checkpoint/artifact capture.
    vbr_lineage_uuid vbr_lineage_id() const {
        const auto * tracker = vbr_generation_tracker_get();
        return tracker != nullptr ? tracker->lineage_identity() : vbr_lineage_uuid{};
    }
    // Process-local routing identity for authenticated mutation/recovery manifests.
    vbr_controller_instance_id vbr_instance_id() const {
        const auto * tracker = vbr_generation_tracker_get();
        return tracker != nullptr ? tracker->runtime_instance() : vbr_controller_instance_id{};
    }

  private:

    vbr_generation_event vbr_generation_begin(
            vbr_mutation_registrant registrant,
            vbr_operation_class operation_class,
            uint32_t stream,
            vbr_generation_stamp_kind stamp_kind,
            bool destructive = false,
            bool imported = false);
    // The one spelling of "a refused stamp fails the owning operation", inert on
    // an empty (unarmed/latched) event, poisons the scope on refusal. Runtime fail-closed,
    // never an assert.
    void vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                   llama_seq_id membership_seq, llama_pos pre_mutation_pos = -1);
    void vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                   const llama_seq_id * seqs, int32_t n_seqs, llama_pos pre_mutation_pos);
    void vbr_generation_global(vbr_mutation_registrant registrant, vbr_operation_class operation_class);
    void vbr_ownership_rebuild();  // import/install-boundary ownership-index rebuild (sanctioned scan)
    void vbr_ownership_update_all_seqs(uint32_t stream, uint32_t cell, llama_pos pos,
                                       bool add, llama_seq_id exclude_seq = -1);
    void     vbr_shrink_watermark();                  // occupancy dropped: release phantom tail pages
    void     vbr_invalidate_dirty_stash();            // one dirty-stash settlement implementation
    bool     vbr_promote_next(uint32_t wm_next);      // occupancy dropped: re-promote one container
    void     vbr_floor_clamp_order();
    bool     vbr_retire_pending_before_unmap(const std::string & busid);
    size_t   vbr_flush_deferred_unmaps(); // returns the number of entries flushed
    bool     vbr_scratch_reserve(size_t flat_cells);  // boundary-time f16 dequant scratch growth
    // Pure, allocator-blind child stream used by the tree transaction. It derives real steps only
    // through vbr_sim_step(); physical pricing and preflight remain separate.
    llama_vbr_policy::child vbr_policy_child_stream(int demanded_device, uint32_t wm_next) const;
    // Pure physical projection for any set of captures in one pool. The first usable capture
    // requires the complete tightly-packed slab, matching the old allocation's fidelity lifetime;
    // current occupancy includes pages retained by earlier failed maps.
    bool     vbr_stash_memory(const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
                              size_t & physical_now, size_t & physical_if_reserved) const;
    bool     vbr_stash_memory_trusted(const vbr_pool & p,
                                      const std::vector<vbr_stash_request> & requests,
                                      size_t & physical_now, size_t & physical_if_reserved) const;
    bool     vbr_stash_memory_impl(const vbr_pool & p,
                                   const std::vector<vbr_stash_request> & requests,
                                   bool ownership_authenticated,
                                   size_t & physical_now, size_t & physical_if_reserved,
                                   uint64_t * request_checks = nullptr) const;
    static bool vbr_stash_requests_valid(
        const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        uint32_t stash_rows, bool ownership_authenticated,
        bool & needs_mapping, uint64_t * request_checks = nullptr);
    // Idempotently reserve/map the same request set. false is recoverable and changes no tier or
    // stash-valid metadata; a partial VMM map may remain resident and is visible to the query.
    bool     vbr_stash_reserve(vbr_pool & p, const std::vector<vbr_stash_request> & requests);
    bool     vbr_stash_reserve_trusted(
        vbr_pool & p, const std::vector<vbr_stash_request> & requests);
    bool     vbr_stash_reserve_impl(
        vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        bool ownership_authenticated);
    void     vbr_load_degrade_order();                // baked table, VBR_DEGRADE_ORDER=<file>, or generic fallback
    void     vbr_synth_generic_order();               // cross-model curves for unsupported archs (VBR_FORCE_GENERIC=1 to force)
    size_t   vbr_vmm_projected_bytes(const vbr_pool & p, uint32_t wm_cells) const;
    size_t   vbr_budget_eff(const vbr_pool & p) const; // live-clamped per-pool budget (shared basis)
    size_t   vbr_budget_eff_uncached(const vbr_pool & p) const; // restore preflight: fresh live capacity
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
    enum class vbr_degrade_result {
        applied,
        exhausted,
        reserve_failed,
        hard_lease_blocked,
        capture_lease_blocked,
    };
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

    // Dynamic VBR shared KV pools — one per KV buffer
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

    friend struct llama_kv_cache_vbr_epoch_test;

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

    // Canonical sequence-serializer inclusion predicate. The disabled generation oracle
    // reuses this cell-local rule while retaining its independent full scan (it must never
    // consume the ownership index or production mask builder).
    bool state_write_includes_cell(
            const llama_kv_cells & cells,
            uint32_t cell,
            llama_seq_id seq_id) const;

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
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

    // used by composite memories whose other child currently exposes fewer
    // logical graph sequences than the attention cache was configured with
    llama_kv_cache_context(
            llama_kv_cache * kv,
            uint32_t         max_graph_seqs_limit);

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
    uint32_t get_max_graph_seqs() const override;

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

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    uint32_t max_graph_seqs = std::numeric_limits<uint32_t>::max();

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
