#include "llama-kv-cache-iswa.h"

#include <exception>
#include <limits>
#include <optional>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_iswa
//

llama_kv_cache_iswa::llama_kv_cache_iswa(
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
    const  layer_share_cb & share,
    const llama_memory_vbr_params & vbr) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_ubatch, n_pad, mem_other, filter, reuse, share, vbr) {
}

llama_kv_cache_iswa::llama_kv_cache_iswa(
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
    const llama_memory_vbr_params & vbr) : swa_full(swa_full), unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    // split the dynamic-VBR budget across the two caches proportional to their worst-case
    // (entry-tier) footprints — layers x cells; both instances arming with the FULL budget
    // would target ~2x the configured mapped-physical total before degrading
    llama_memory_vbr_params vbr_base = vbr;
    llama_memory_vbr_params vbr_swa  = vbr;
    // Distinct VBR_TRACE files per child record both schedules without the
    // second open truncating the first.
    vbr_base.trace_label = "base";
    vbr_swa.trace_label  = "swa";
    if (vbr.dynamic || vbr.budget_bytes > 0) {
        uint64_t n_base_l = 0;
        uint64_t n_swa_l  = 0;
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (filter && !filter(il)) {
                continue;
            }
            (hparams.is_swa(il) ? n_swa_l : n_base_l)++;
        }
        const double w_base = (double) n_base_l * size_base;
        const double w_swa  = (double) n_swa_l  * size_swa;
        if (w_base + w_swa > 0.0) {
            // the same footprint weights split BOTH the configured budget and the children's
            // claim on the device's spare VRAM (device_share): two independent controllers on
            // one device must never both re-derive against the full free amount
            vbr_base.device_share = vbr.device_share * (w_base / (w_base + w_swa));
            vbr_swa.device_share  = vbr.device_share - vbr_base.device_share;
            if (vbr.budget_bytes > 0) {
                vbr_base.budget_bytes = (uint64_t) ((double) vbr.budget_bytes * (w_base / (w_base + w_swa)));
                vbr_swa.budget_bytes  = vbr.budget_bytes - vbr_base.budget_bytes;
                if (vbr.dynamic) {
                    LLAMA_LOG_INFO("%s: VBR budget split: %.2f MiB base / %.2f MiB SWA (by entry-tier footprint)\n",
                            __func__, vbr_base.budget_bytes/1024.0/1024.0, vbr_swa.budget_bytes/1024.0/1024.0);
                }
            }
        }
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    llama_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llama_kv_cache_iswa *>(mem_other)->get_base();
    }

    llama_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llama_kv_cache_iswa *>(mem_other)->get_swa();
    }

    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share, vbr_base);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share, vbr_swa);

    // Run the process-external protocol once per composite. Choose the last active child
    // in the parent's fixed base->SWA execution order so the root can finalize both samples
    // and fence either child's service wave before graph launch.
    kv_vbr_root = kv_swa->vbr_controller_active() ? kv_swa.get()
                : kv_base->vbr_controller_active() ? kv_base.get()
                : nullptr;
    if (kv_vbr_root != nullptr) {
        llama_kv_cache * peer = kv_vbr_root == kv_swa.get() ? kv_base.get() : kv_swa.get();
        kv_vbr_root->vbr_attach_ledger_tree(kv_vbr_root, peer, vbr.device_share);
        peer       ->vbr_attach_ledger_tree(kv_vbr_root, kv_vbr_root, vbr.device_share);
        kv_vbr_root->vbr_finalize_ledger_tree();
    }
}

void llama_kv_cache_iswa::vbr_finalize_prepare_failure(
        llama_kv_cache * child, const std::vector<llama_ubatch> & ubatches) {
    if (kv_vbr_root == nullptr) {
        return;
    }
    uint32_t n_tokens = 0;
    for (const auto & ubatch : ubatches) {
        n_tokens += ubatch.n_tokens;
    }
    // Base runs before SWA. A failed SWA therefore follows a root run even when the only
    // active controller (and thus root) is base; a failed root recorded its own sample.
    const bool root_ran = child == kv_vbr_root || child == kv_swa.get();
    kv_vbr_root->vbr_finalize_failed_child(n_tokens, root_ran);
}

namespace {

// One operation per logical wrapper mutation. Minted here via the
// operation-TU RAII (registry_begin never called from this file), adopted into both children
// so their mutation scopes join the same id instead of minting divergent ones.
// A refused mint propagates: children open refused (fail closed to
// shadow-unavailable) instead of minting independently, preserving the operation registry one-id invariant
// even in refusal.
struct iswa_forwarded_op {
    // Synchronous families: armed iff the manifest carries at least one (armed-child) target.
    iswa_forwarded_op(llama_kv_cache * base, llama_kv_cache * swa,
                      const vbr_operation_binding & binding)
        : iswa_forwarded_op(base, swa, binding, binding.n_targets > 0) {}

    iswa_forwarded_op(llama_kv_cache * base, llama_kv_cache * swa,
                      const vbr_operation_binding & binding, bool armed)
        : base_(base), swa_(swa) {
        if (!armed) {
            return;
        }
        op_.emplace(binding);
        if (*op_) {
            base_->vbr_adopt_operation(op_->id());
            swa_ ->vbr_adopt_operation(op_->id());
            adopted_ = true;
        } else {
            base_->vbr_adopt_refused();
            swa_ ->vbr_adopt_refused();
            refused_ = true;
        }
    }
    // Parent-declared fixed participant slots, created before the first child
    // apply. Children claim their slot in their scope constructors; detach transfers the
    // still-open token; the root closes only at sealed && every declared slot terminal,
    // failure dominating.
    void adopt_composite(int32_t declared) {
        if (adopted_) {
            composite_ = std::make_shared<llama_kv_cache::vbr_composite_outcome>();
            composite_->operation_id = op_->id();
            composite_->declared     = declared;
            base_->vbr_adopt_composite(composite_);
            swa_ ->vbr_adopt_composite(composite_);
        }
    }
    // Called with the children's apply result. With a composite the aggregate owns the root
    // close from here: sealing folds the wrapper result and fails any never-claimed declared
    // slot; the close fires once every declared slot is terminal (possibly right now, or at
    // the sync fence for transferred tokens). Without one, close here, failure dominating.
    void finalize(bool ok) {
        if (!adopted_ && !refused_) {
            return;
        }
        base_->vbr_release_adopted();
        swa_ ->vbr_release_adopted();
        adopted_ = refused_ = false;
        // Refused roots fall through inert: composite_ was never built and *op_ is false.
        if (composite_) {
            op_->release();
            composite_->seal(ok);
        } else if (op_ && *op_) {
            op_->close(ok ? vbr_operation_outcome::committed : vbr_operation_outcome::failed);
        }
    }
    ~iswa_forwarded_op() {
        if (adopted_ || refused_) {
            // Exception/unwind before finalize: release the children and seal FAILED — the
            // sealed+closed guards make any later pending report inert, never a double close.
            base_->vbr_release_adopted();
            swa_ ->vbr_release_adopted();
            if (composite_) {
                op_->release();
                composite_->seal(false);
            }
        }
        // Synchronous wrapper families commit at scope end; unwind fails (root RAII default).
        if (op_ && *op_ && std::uncaught_exceptions() == exceptions_at_entry_) {
            op_->close(vbr_operation_outcome::committed);
        }
    }
    llama_kv_cache * base_;
    llama_kv_cache * swa_;
    std::optional<vbr_scoped_operation> op_;
    std::shared_ptr<llama_kv_cache::vbr_composite_outcome> composite_;
    bool adopted_ = false;
    bool refused_ = false;
    int  exceptions_at_entry_ = std::uncaught_exceptions();
};

// Synchronous wrapper families carry an exact-instance manifest: one target per
// ARMED child, never an instance wildcard through default arguments. Raw public-API ranges
// normalize here so the closed mint range rules see canonical values.
vbr_operation_binding iswa_edit_binding(const llama_kv_cache * base, const llama_kv_cache * swa,
                                        vbr_operation_kind kind, vbr_operation_class cls,
                                        llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    vbr_normalize_edit_range(p0, p1);  // Canonical range before the mint.
    vbr_operation_binding binding;
    binding.kind        = kind;
    binding.child_phase = vbr_operation_phase::mutate;
    for (const vbr_controller_instance_id instance :
            { base->vbr_instance_id(), swa->vbr_instance_id() }) {
        vbr_binding_add_instance_target(
            binding, kind, cls, instance, VBR_STREAM_ANY, seq_id, p0, p1);
    }
    return binding;
}

}  // namespace

void llama_kv_cache_iswa::clear(bool data) {
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api,
                              -1, 0, std::numeric_limits<llama_pos>::max()));
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llama_kv_cache_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id, p0, p1));
    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

bool llama_kv_cache_iswa::seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id, p0, p1));
    res = res & kv_base->seq_rm_transient(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm_transient(seq_id, p0, p1);
    return res;
}

bool llama_kv_cache_iswa::seq_rm_attn_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return seq_rm_transient(seq_id, p0, p1);
}

void llama_kv_cache_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id_dst, p0, p1));
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_kv_cache_iswa::try_seq_cp_transient(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id_dst, p0, p1));
    const bool base = kv_base->try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1);
    const bool swa  = kv_swa ->try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1);
    return base && swa;
}

void llama_kv_cache_iswa::seq_keep(llama_seq_id seq_id) {
    // The children's keep manifests declare the seq wildcard (membership leaves every OTHER
    // sequence); the wrapper's shared manifest matches.
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api,
                              -1, 0, std::numeric_limits<llama_pos>::max()));
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id, p0, p1));
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    const iswa_forwarded_op forwarded(kv_base.get(), kv_swa.get(),
            iswa_edit_binding(kv_base.get(), kv_swa.get(), vbr_operation_kind::sequence_edit,
                              vbr_operation_class::state_api, seq_id, p0, p1));
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown_vbr_managed() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown_vbr_managed();
    for (const auto & buft_size : kv_swa->memory_breakdown_vbr_managed()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->plan_slots(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->plan_slots(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        sinfos_base = kv_base->prepare_with_slots(ubatches, std::move(sinfos_base));
        if (sinfos_base.empty()) {
            vbr_finalize_prepare_failure(kv_base.get(), ubatches);
            break;
        }

        sinfos_swa = kv_swa->prepare_with_slots(ubatches, std::move(sinfos_swa));
        if (sinfos_swa.empty()) {
            vbr_finalize_prepare_failure(kv_swa.get(), ubatches);
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->plan_slots(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->plan_slots(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        sinfos_base = kv_base->prepare_with_slots(ubatches, std::move(sinfos_base));
        if (sinfos_base.empty()) {
            vbr_finalize_prepare_failure(kv_base.get(), ubatches);
            break;
        }

        sinfos_swa = kv_swa->prepare_with_slots(ubatches, std::move(sinfos_swa));
        if (sinfos_swa.empty()) {
            vbr_finalize_prepare_failure(kv_swa.get(), ubatches);
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_full() {
    return std::make_unique<llama_kv_cache_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_iswa_context>(this, lctx, optimize);
}

double llama_kv_cache_iswa::kv_bpv() const {
    double bits = 0.0;
    double vals = 0.0;
    kv_base->kv_bpv_accum(bits, vals);
    kv_swa ->kv_bpv_accum(bits, vals);
    return vals > 0.0 ? bits / vals : -1.0;
}

llama_memory_vbr_state_data_v2 llama_kv_cache_iswa::memory_vbr_state_v2(
        llama_seq_id seq_id, uint32_t n_tokens_extra) const {
    const llama_memory_vbr_state_data_v2 b = kv_base->memory_vbr_state_v2(seq_id, n_tokens_extra);
    const llama_memory_vbr_state_data_v2 s = kv_swa ->memory_vbr_state_v2(seq_id, n_tokens_extra);
    const auto & bs = b.state;
    const auto & ss = s.state;

    llama_memory_vbr_state_data_v2 r = {};
    auto & rs = r.state;
    // each child runs an independent controller with its own budget share: either one over
    // budget means degrades happen, so pressure combines as max, exactly like the trigger
    rs.deficit_raw      = std::max(bs.deficit_raw,     ss.deficit_raw);
    rs.deficit_clamped  = std::max(bs.deficit_clamped, ss.deficit_clamped);
    rs.cursor           = bs.cursor + ss.cursor;
    rs.used_cells_other = bs.used_cells_other + ss.used_cells_other;
    r.used_cells_exclusive =
        b.used_cells_exclusive + s.used_cells_exclusive;
    // Representation epochs are identities, not quantities: preserve the ordered child tuple.
    // Addition would make (base + 1, swa) collide with (base, swa + 1).
    rs.representation_epoch     = bs.representation_epoch;
    rs.representation_epoch_swa = ss.representation_epoch;
    rs.checkpoint_epoch         = bs.checkpoint_epoch;
    rs.checkpoint_epoch_swa     = ss.checkpoint_epoch;
    // Scoped entry/exit is parent-coordinated, so depth and scope counts agree whenever both
    // children are active. max also handles a model whose base or SWA side has no VBR layers.
    rs.retier_freeze_depth  = std::max(bs.retier_freeze_depth, ss.retier_freeze_depth);
    rs.retier_env_freeze    = std::max(bs.retier_env_freeze,   ss.retier_env_freeze);
    rs.retier_freeze_enters = std::max(bs.retier_freeze_enters, ss.retier_freeze_enters);
    rs.retier_freeze_exits  = std::max(bs.retier_freeze_exits,  ss.retier_freeze_exits);
    rs.retier_reconciles    = std::max(bs.retier_reconciles,    ss.retier_reconciles);
    // Decisions are per-controller (base and SWA may see different pressure), so preserve both.
    rs.retier_deferred_decisions =
        bs.retier_deferred_decisions + ss.retier_deferred_decisions;

    // value-weighted like kv_bpv: weight each child's landing bpv by its total KV values
    double bits_base = 0.0, vals_base = 0.0;
    double bits_swa  = 0.0, vals_swa  = 0.0;
    kv_base->kv_bpv_accum(bits_base, vals_base);
    kv_swa ->kv_bpv_accum(bits_swa,  vals_swa);
    const double vals_sum = vals_base + vals_swa;
    rs.bpv_if_degraded = vals_sum > 0.0
        ? (bs.bpv_if_degraded * vals_base + ss.bpv_if_degraded * vals_swa) / vals_sum
        : 0.0;
    return r;
}

bool llama_kv_cache_iswa::vbr_capture_readiness_cells(
        uint64_t logical_growth,
        uint64_t & committed,
        uint64_t & projected,
        uint64_t & capacity) const {
    uint64_t bc = 0, bp = 0, bz = 0;
    uint64_t sc = 0, sp = 0, sz = 0;
    if (!kv_base->vbr_capture_readiness_cells(
            logical_growth, bc, bp, bz) ||
        !kv_swa->vbr_capture_readiness_cells(
            logical_growth, sc, sp, sz) ||
        bc > UINT64_MAX-sc || bp > UINT64_MAX-sp || bz > UINT64_MAX-sz) {
        committed = projected = capacity = 0;
        return false;
    }
    committed = bc+sc;
    projected = bp+sp;
    capacity = bz+sz;
    return capacity != 0;
}

bool llama_kv_cache_iswa::vbr_operation_armed() const {
    return kv_base->vbr_operation_armed() || kv_swa->vbr_operation_armed();
}

bool llama_kv_cache_iswa::vbr_retier_freeze_begin(
        const char * owner, vbr_operation_id operation_id) {
    const bool base_armed = kv_base->vbr_operation_armed();
    const bool swa_armed  = kv_swa ->vbr_operation_armed();
    if (!base_armed && !swa_armed) {
        return false;
    }
    if (vbr_retier_freeze_depth_ >= VBR_RETIER_FREEZE_MAX_DEPTH) {
        return false;
    }
    const bool base_ok = !base_armed ||
        kv_base->vbr_retier_freeze_begin(owner, operation_id);
    if (!base_ok) {
        return false;
    }
    const bool swa_ok = !swa_armed ||
        kv_swa->vbr_retier_freeze_begin(owner, operation_id);
    if (!swa_ok) {
        if (base_armed) {
            kv_base->vbr_retier_freeze_end(owner, operation_id);
        }
        return false;
    }
    vbr_retier_freeze_stack_[vbr_retier_freeze_depth_++] = {
        operation_id,
        base_armed,
        swa_armed,
    };
    return true;
}

void llama_kv_cache_iswa::vbr_retier_freeze_end(
        const char * owner, vbr_operation_id operation_id) {
    GGML_ASSERT(vbr_retier_freeze_depth_ > 0);
    const vbr_retier_freeze_children children =
        vbr_retier_freeze_stack_[vbr_retier_freeze_depth_ - 1];
    GGML_ASSERT(children.operation_id == operation_id);

    // Pair against the immutable begin record, never the current budget/armed state.
    if (children.froze_base) {
        kv_base->vbr_retier_freeze_end(owner, operation_id);
    }
    if (children.froze_swa) {
        kv_swa->vbr_retier_freeze_end(owner, operation_id);
    }
    vbr_retier_freeze_depth_--;
    vbr_retier_freeze_stack_[vbr_retier_freeze_depth_] = {};
}

llama_memory_vbr_preflight_data llama_kv_cache_iswa::vbr_retier_preflight(
        uint32_t n_tokens_extra,
        std::vector<llama_memory_vbr_physical_growth> * physical) const {
    return llama_memory_vbr_preflight_children(
        *kv_base, *kv_swa, n_tokens_extra, physical);
}

bool llama_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_base->get_size() == kv_swa->get_size();
}

void llama_kv_cache_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache * llama_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_iswa_context
//

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv) : llama_kv_cache_iswa_context(kv, std::numeric_limits<uint32_t>::max()) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        uint32_t              max_graph_seqs_limit) :
    kv(kv),
    ctx_base(new llama_kv_cache_context(kv->get_base(), max_graph_seqs_limit)),
    ctx_swa (new llama_kv_cache_context(kv->get_swa (), max_graph_seqs_limit)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv(kv),
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    kv(kv),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llama_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llama_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context:: ~llama_kv_cache_iswa_context() = default;

bool llama_kv_cache_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    // KV-update / full contexts carry no decode ubatch — no composite decode operation
    // (the children's own update paths handle their recovery drains).
    if (ubatches.empty()) {
        res = res & ctx_base->apply();
        res = res & ctx_swa ->apply();
        return res;
    }

    // The composite owns one logical decode operation with
    // a FIXED parent-declared participant set. Both children apply under the adopted shared
    // id (their scopes claim their slot and own tracker-local extent/recovery reservations).
    // Manifest construction is ARMED-ONLY and TRANSACTIONAL: targets publish only when every
    // armed child's ubatch scan succeeded; any overflow leaves a zero-target manifest, the
    // registry refuses the mint, and the refusal PROPAGATES to both children — never
    // independent minters (operation registry one-id even in refusal).
    const vbr_controller_instance_id base_instance = kv->get_base()->vbr_instance_id();
    const vbr_controller_instance_id swa_instance  = kv->get_swa ()->vbr_instance_id();
    const bool base_armed = vbr_controller_instance_id_is_set(base_instance);
    const bool swa_armed  = vbr_controller_instance_id_is_set(swa_instance);
    if (!base_armed && !swa_armed) {
        res = res & ctx_base->apply();
        res = res & ctx_swa ->apply();
        return res;
    }
    const llama_ubatch & cur_ubatch = ubatches[i_next];
    vbr_operation_binding composite_binding;
    composite_binding.kind        = vbr_operation_kind::decode;
    composite_binding.child_phase = vbr_operation_phase::mutate;
    // A failed scan zeroes the whole manifest (transactional inside the builder).
    (void) ((!base_armed || llama_kv_cache::vbr_decode_targets_from_ubatch(
                    composite_binding, base_instance, false, VBR_STREAM_ANY, cur_ubatch)) &&
            (!swa_armed  || llama_kv_cache::vbr_decode_targets_from_ubatch(
                    composite_binding, swa_instance, true, VBR_STREAM_ANY, cur_ubatch)));
    iswa_forwarded_op composite(kv->get_base(), kv->get_swa(), composite_binding, /*armed=*/true);
    composite.adopt_composite((base_armed ? 1 : 0) + (swa_armed ? 1 : 0));

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    // Finalize folds the wrapper result and seals the aggregate; the root closes
    // at sealed && every declared slot terminal — right now for all-synchronous applies, at
    // the sync fence for transferred tokens — failure dominating.
    composite.finalize(res);
    return res;
}

llama_memory_status llama_kv_cache_iswa_context::get_status() const {
    return status;
}

uint32_t llama_kv_cache_iswa_context::get_max_graph_seqs() const {
    if (!ctx_base || !ctx_swa) {
        return 0;
    }
    return std::min(ctx_base->get_max_graph_seqs(), ctx_swa->get_max_graph_seqs());
}

const llama_ubatch & llama_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint64_t llama_kv_cache_iswa_context::get_vbr_epoch() const {
    return get_base()->get_vbr_epoch() + get_swa()->get_vbr_epoch();
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}

void llama_kv_cache_iswa::vbr_commit_submitted() {
    kv_base->vbr_commit_submitted();
    kv_swa ->vbr_commit_submitted();
}

void llama_kv_cache_iswa::vbr_decode_ops_finish(bool ok) {
    kv_base->vbr_decode_ops_finish(ok);
    kv_swa ->vbr_decode_ops_finish(ok);
}

void llama_kv_cache_iswa::vbr_adopt_operation(vbr_operation_id operation_id) {
    kv_base->vbr_adopt_operation(operation_id);
    kv_swa ->vbr_adopt_operation(operation_id);
}

void llama_kv_cache_iswa::vbr_release_adopted() {
    kv_base->vbr_release_adopted();
    kv_swa ->vbr_release_adopted();
}
