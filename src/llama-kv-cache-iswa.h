#pragma once

#include "llama-kv-cache.h"

#include <array>
#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share = nullptr,
        const llama_memory_vbr_params & vbr = {});

    llama_kv_cache_iswa(
            const llama_model & model,
            const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        const llama_memory_vbr_params & vbr = {});

    ~llama_kv_cache_iswa() = default;

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
    // A bounded SWA cache does not retain enough history for arbitrary partial removal.
    bool can_seq_rm_partial() const override { return swa_full; }

    double kv_bpv() const override; // value-weighted combination of the base and SWA caches

    llama_memory_vbr_state_data memory_vbr_state(llama_seq_id seq_id, uint32_t n_tokens_extra) const override;
    bool vbr_operation_armed() const override;
    bool vbr_retier_freeze_begin(const char * owner, vbr_operation_id operation_id) override;
    void vbr_retier_freeze_end(const char * owner, vbr_operation_id operation_id) override;
    void vbr_commit_submitted() override;
    void vbr_decode_ops_finish(bool ok) override;
    void vbr_adopt_operation(vbr_operation_id operation_id) override;
    void vbr_release_adopted() override;
    llama_memory_vbr_preflight_data vbr_retier_preflight(uint32_t n_tokens_extra) const override;

    // summed across both children: each context token holds one row in each cache, so the
    // per-token floor cost is additive (SWA rows recycle, but the fit's measured KV bytes
    // count both caches the same way — the ratio consumer stays on one basis)
    double memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override {
        return kv_base->memory_vbr_floor_bits_per_token(entry_k, entry_v, floor_bpv) +
               kv_swa ->memory_vbr_floor_bits_per_token(entry_k, entry_v, floor_bpv);
    }

    // #88: NOT summed — both children share one per-device scratch sized by the widest attended
    // range, and the SWA cache's range is window-bound (n-invariant at depth). Only the base
    // (full-attention) cache's scratch scales with context.
    double memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override {
        return kv_base->memory_vbr_scratch_bytes_per_token(entry_k, entry_v, floor_bpv);
    }

    // per-child forwarding is for POOL maintenance only (both children can hold deferred
    // VBR unmaps); the P2 tick's ledger scan/demand servicing happens at the parent level,
    // never per child
    void breathe() override { kv_base->breathe(); kv_swa->breathe(); }

    void vbr_cotenancy_accum(uint64_t & d, uint32_t & g, uint64_t & o, uint64_t & p) const override {
        kv_base->vbr_cotenancy_accum(d, g, o, p);
        kv_swa ->vbr_cotenancy_accum(d, g, o, p);
    }

    bool vbr_ledger_tree_active() const override {
        return kv_vbr_root != nullptr && kv_vbr_root->vbr_ledger_tree_active();
    }
    void vbr_hard_seal_guard_set(vbr_hard_seal_guard guard) override {
        kv_base->vbr_hard_seal_guard_set(guard);
        kv_swa->vbr_hard_seal_guard_set(std::move(guard));
    }
    bool vbr_hard_seal_blocked_take(bool decode_failed) override {
        const bool base = kv_base->vbr_hard_seal_blocked_take(decode_failed);
        const bool swa = kv_swa->vbr_hard_seal_blocked_take(decode_failed);
        return base || swa;
    }
    void vbr_hard_seal_evidence_take(
            std::vector<vbr_hard_seal_subject> & out) override {
        kv_base->vbr_hard_seal_evidence_take(out);
        kv_swa->vbr_hard_seal_evidence_take(out);
    }

    void vbr_shared_scratch_detach() override {
        kv_base->vbr_shared_scratch_detach();
        kv_swa ->vbr_shared_scratch_detach();
    }

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
    // llama_kv_cache_iswa specific API
    //

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_swa () const;

private:
    struct vbr_retier_freeze_children {
        vbr_operation_id operation_id = {};
        bool froze_base = false;
        bool froze_swa  = false;
    };
    static constexpr size_t VBR_RETIER_FREEZE_MAX_DEPTH = 64;

    void vbr_finalize_prepare_failure(llama_kv_cache * child,
            const std::vector<llama_ubatch> & ubatches);

    const bool swa_full;
    const bool unified;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;

    // Exact begin/end pairing must not consult mutable armed policy at end. Runtime budget
    // renegotiation may change vbr_operation_armed() while a nested scope is live.
    std::array<vbr_retier_freeze_children, VBR_RETIER_FREEZE_MAX_DEPTH> vbr_retier_freeze_stack_ = {};
    uint32_t vbr_retier_freeze_depth_ = 0;
    llama_kv_cache * kv_vbr_root = nullptr;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv);

    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            uint32_t              max_graph_seqs_limit);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_iswa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    uint32_t get_max_graph_seqs() const override;

    // sum of both caches' tier epochs — a flip in either fences graph reuse
    uint64_t get_vbr_epoch() const override;

    //
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_swa()  const;

private:
    llama_kv_cache_iswa * kv = nullptr;  // C1: composite decode-op owner needs the caches

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;

    const llama_memory_status status;
};
