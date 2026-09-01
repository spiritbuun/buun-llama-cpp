#include "llama-vbr-artifact-adopt.h"

#include "llama-kv-cache.h"
#include "llama-memory-tree.h"
#include "llama-vbr-controller-id.h"
#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-operation.h"

#include "ggml.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

class vbr_import_receipt_group {
  public:
    vbr_import_receipt_group(
            llama_cache_acct_ledger & ledger,
            const std::vector<llama_cache_acct_op_id> & operations)
        : ledger_(&ledger) {
        operations_.reserve(operations.size());
        for (const auto op : operations) {
            if (op) {
                operations_.push_back(op);
            }
        }
    }

    ~vbr_import_receipt_group() {
        if (!active_ || ledger_ == nullptr) {
            return;
        }
        for (const auto op : operations_) {
            ledger_->release(op);
        }
    }

    vbr_import_receipt_group(const vbr_import_receipt_group &) = delete;
    vbr_import_receipt_group & operator=(
        const vbr_import_receipt_group &) = delete;

    void activate() noexcept { active_ = true; }
    bool valid() const noexcept {
        return ledger_ != nullptr && !operations_.empty();
    }

  private:
    llama_cache_acct_ledger * ledger_ = nullptr;
    std::vector<llama_cache_acct_op_id> operations_;
    bool active_ = false;
};

const char * vbr_adopt_phase_name(vbr_adopt_phase phase) noexcept {
    switch (phase) {
        case vbr_adopt_phase::consume_capabilities: return "consume_capabilities";
        case vbr_adopt_phase::operation_open: return "operation_open";
        case vbr_adopt_phase::target_recheck: return "target_recheck";
        case vbr_adopt_phase::journal_arm: return "journal_arm";
        case vbr_adopt_phase::private_backing: return "private_backing";
        case vbr_adopt_phase::unit_h2d: return "unit_h2d";
        case vbr_adopt_phase::unit_complete: return "unit_complete";
        case vbr_adopt_phase::stash_and_companions: return "stash_and_companions";
        case vbr_adopt_phase::live_image_prepare: return "live_image_prepare";
        case vbr_adopt_phase::complete_tree_barrier: return "complete_tree_barrier";
        case vbr_adopt_phase::tracker_prepare: return "tracker_prepare";
        case vbr_adopt_phase::composite_publish: return "composite_publish";
        case vbr_adopt_phase::close: return "close";
        case vbr_adopt_phase::rollback: return "rollback";
        case vbr_adopt_phase::_count: break;
    }
    return "invalid";
}

const char * vbr_adopt_status_name(vbr_adopt_status status) noexcept {
    switch (status) {
        case vbr_adopt_status::adopted: return "adopted";
        case vbr_adopt_status::invalid_capability_pair: return "invalid_capability_pair";
        case vbr_adopt_status::unsupported_decision: return "unsupported_decision";
        case vbr_adopt_status::downward_deferred: return "downward_deferred";
        case vbr_adopt_status::downward_recipe_invalid: return "downward_recipe_invalid";
        case vbr_adopt_status::downward_transform_failed: return "downward_transform_failed";
        case vbr_adopt_status::downward_stash_unavailable: return "downward_stash_unavailable";
        case vbr_adopt_status::upward_recipe_invalid: return "upward_recipe_invalid";
        case vbr_adopt_status::upward_transform_failed: return "upward_transform_failed";
        case vbr_adopt_status::target_drift: return "target_drift";
        case vbr_adopt_status::operation_unavailable: return "operation_unavailable";
        case vbr_adopt_status::recovery_unavailable: return "recovery_unavailable";
        case vbr_adopt_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_adopt_status::admission_refused: return "admission_refused";
        case vbr_adopt_status::private_backing_failed: return "private_backing_failed";
        case vbr_adopt_status::transfer_failed: return "transfer_failed";
        case vbr_adopt_status::event_failed: return "event_failed";
        case vbr_adopt_status::source_changed: return "source_changed";
        case vbr_adopt_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_adopt_status::companion_failed: return "companion_failed";
        case vbr_adopt_status::tracker_failed: return "tracker_failed";
        case vbr_adopt_status::barrier_failed: return "barrier_failed";
        case vbr_adopt_status::rollback_failed: return "rollback_failed";
        case vbr_adopt_status::quarantined: return "quarantined";
        case vbr_adopt_status::internal_error: return "internal_error";
        case vbr_adopt_status::_count: break;
    }
    return "invalid";
}

const char * vbr_adopt_recovery_outcome_name(
        vbr_adopt_recovery_outcome outcome) noexcept {
    switch (outcome) {
        case vbr_adopt_recovery_outcome::not_needed: return "not_needed";
        case vbr_adopt_recovery_outcome::replayed: return "replayed";
        case vbr_adopt_recovery_outcome::quarantined: return "quarantined";
        case vbr_adopt_recovery_outcome::_count: break;
    }
    return "invalid";
}

const char * vbr_downward_adopt_subphase_name(
        vbr_downward_adopt_subphase subphase) noexcept {
    switch (subphase) {
        case vbr_downward_adopt_subphase::none: return "none";
        case vbr_downward_adopt_subphase::source_h2d: return "source_h2d";
        case vbr_downward_adopt_subphase::edge_stash_capture:
            return "edge_stash_capture";
        case vbr_downward_adopt_subphase::edge_transcode:
            return "edge_transcode";
        case vbr_downward_adopt_subphase::edge_completion:
            return "edge_completion";
        case vbr_downward_adopt_subphase::_count: break;
    }
    return "_count";
}

vbr_prepared_companion_image::~vbr_prepared_companion_image() = default;

vbr_adopt_status vbr_adopt_check_complete_tree(
        const std::vector<vbr_adopt_expected_attention> & expected,
        const std::vector<llama_memory_tree_child> & live,
        const std::vector<vbr_companion_adoption_provider> & companions,
        bool occupied_replacement) noexcept {
    const size_t live_attention = std::count_if(
        live.begin(), live.end(),
        [](const llama_memory_tree_child & child) {
            return child.attention != nullptr;
        });
    if (expected.empty() || live_attention != expected.size()) {
        return vbr_adopt_status::target_drift;
    }
    for (const auto & item : expected) {
        const auto child = std::find_if(
            live.begin(), live.end(),
            [&](const llama_memory_tree_child & value) {
                return value.child_id == item.child_id &&
                       value.attention == item.cache;
            });
        if (child == live.end()) {
            return vbr_adopt_status::target_drift;
        }
    }
    for (const auto & child : live) {
        if (child.recurrent) {
            const auto prepared = std::find_if(
                companions.begin(), companions.end(),
                [&](const vbr_companion_adoption_provider & value) {
                    return value.kind == vbr_artifact_companion_kind::recurrent &&
                           value.target_cookie == child.recurrent;
                });
            if (prepared == companions.end() || !prepared->target_empty) {
                return vbr_adopt_status::required_companion_unavailable;
            }
            const auto duplicate = std::find_if(
                std::next(prepared), companions.end(),
                [&](const vbr_companion_adoption_provider & value) {
                    return value.kind == vbr_artifact_companion_kind::recurrent &&
                           value.target_cookie == child.recurrent;
                });
            if (duplicate != companions.end()) {
                return vbr_adopt_status::required_companion_unavailable;
            }
            if (!prepared->target_empty(prepared->context) &&
                (!occupied_replacement ||
                 prepared->prepare_replacement == nullptr)) {
                return vbr_adopt_status::target_drift;
            }
        }
        if (child.qsa_index_owner) {
            const auto qsa = std::find_if(
                companions.begin(), companions.end(),
                [&](const vbr_companion_adoption_provider & value) {
                    return value.kind == vbr_artifact_companion_kind::qsa_index &&
                           value.target_cookie == child.qsa_index_owner &&
                           value.attention_child_id == child.child_id;
                });
            if (qsa == companions.end() || !qsa->target_empty) {
                return vbr_adopt_status::required_companion_unavailable;
            }
            const auto duplicate = std::find_if(
                std::next(qsa), companions.end(),
                [&](const vbr_companion_adoption_provider & value) {
                    return value.kind == vbr_artifact_companion_kind::qsa_index &&
                           value.target_cookie == child.qsa_index_owner;
                });
            if (duplicate != companions.end()) {
                return vbr_adopt_status::required_companion_unavailable;
            }
            // Layout-aware QSA preparation installs reversible state directly
            // into the target.  The target was checked before preparation and
            // the prepared image is checked again below, so non-empty state is
            // expected at this barrier (unlike the off-side recurrent image).
        }
    }
    for (const auto & provider : companions) {
        if (provider.kind != vbr_artifact_companion_kind::recurrent &&
            provider.kind != vbr_artifact_companion_kind::qsa_index) {
            continue;
        }
        const auto child = std::find_if(
            live.begin(), live.end(),
            [&](const llama_memory_tree_child & value) {
                return provider.kind == vbr_artifact_companion_kind::recurrent
                    ? value.recurrent != nullptr &&
                      provider.target_cookie == value.recurrent
                    : value.qsa_index_owner != nullptr &&
                      provider.target_cookie == value.qsa_index_owner &&
                      provider.attention_child_id == value.child_id;
            });
        if (child == live.end()) {
            return vbr_adopt_status::target_drift;
        }
    }
    return vbr_adopt_status::adopted;
}

namespace {

bool fingerprint_equal(const vbr_target_empty_fingerprint & a,
                       const vbr_target_empty_fingerprint & b) noexcept {
    if (a.memory_instance_cookie != b.memory_instance_cookie ||
        a.target_state_serial != b.target_state_serial ||
        a.accounting_serial != b.accounting_serial ||
        a.tree_shape_digest != b.tree_shape_digest ||
        a.policy_epoch != b.policy_epoch ||
        a.previously_observed != b.previously_observed ||
        a.children.size() != b.children.size()) {
        return false;
    }
    for (size_t i = 0; i < a.children.size(); ++i) {
        if (a.children[i].child_id != b.children[i].child_id ||
            a.children[i].memory_cookie != b.children[i].memory_cookie ||
            a.children[i].state_serial != b.children[i].state_serial ||
            a.children[i].instance_id != b.children[i].instance_id) {
            return false;
        }
    }
    return true;
}

const vbr_checkpoint_generation_controller * source_controller(
        const vbr_validated_manifest & manifest,
        uint32_t child_id) noexcept {
    return manifest.source_controller(child_id);
}

const vbr_tracker_install_child * tracker_plan(
        const vbr_validated_manifest & manifest,
        uint32_t child_id) noexcept {
    const auto & children = manifest.tracker_install().children;
    const auto found = std::find_if(
        children.begin(), children.end(),
        [&](const vbr_tracker_install_child & value) {
            return value.child_id == child_id;
        });
    return found == children.end() ? nullptr : &*found;
}

} // namespace

bool vbr_artifact_epoch_capacity(
        uint64_t tier_epoch,
        uint64_t representation_epoch,
        uint64_t checkpoint_epoch,
        uint64_t tier_changes) noexcept {
    return representation_epoch != UINT64_MAX &&
           checkpoint_epoch != UINT64_MAX &&
           tier_changes <= UINT64_MAX-tier_epoch;
}

// This class is the sole friend of llama_kv_cache for artifact adoption. It owns
// every prepublication mutation and its inverse; callers cannot acquire a
// tensor/pool destination outside the open journal.
class vbr_kv_import_session {
  public:
    struct mapped_range {
        llama_kv_cache::vbr_pool * pool = nullptr;
        size_t offset = 0;
        size_t size = 0;
    };
    struct extent_prefix {
        llama_kv_cache::vbr_pool * pool = nullptr;
        llama_kv_cache::vbr_extent * extent = nullptr;
        size_t pool_index = SIZE_MAX;
        // A transform maps the larger of source and target before H2D.  The
        // publication invariant is the target-tier prefix left after the
        // fallible transform (and downward-only tail trim).
        uint64_t mapped_row_bytes = 0;
        uint64_t final_row_bytes = 0;
        uint32_t initial_watermark = 0;
    };
    struct final_unit_metadata {
        uint32_t logical_unit = UINT32_MAX;
        uint32_t stash_valid = 0;
        uint8_t promote_hops = 0;
        ggml_type target_type = GGML_TYPE_COUNT;
    };

    vbr_kv_import_session(llama_kv_cache & cache, uint32_t child_id,
                          llama_seq_id destination,
                          vbr_operation_id operation,
                          bool occupied_replacement,
                          vbr_adopt_test_seam * test_seam = nullptr) noexcept
        : cache_(&cache), child_id_(child_id), destination_(destination),
          operation_(operation), occupied_replacement_(occupied_replacement),
          test_seam_(test_seam) {}

    ~vbr_kv_import_session() {
        if (!published_) {
            rollback(false);
        }
    }

    bool recheck(const vbr_child_empty_fingerprint & expected,
                 bool journal_armed = false) const noexcept {
        if (test_seam_) {
            return test_seam_->session_recheck(
                child_id_, expected, journal_armed);
        }
        if (!cache_ || cache_->other != nullptr ||
            expected.child_id != child_id_ ||
            expected.memory_cookie != cache_ ||
            (journal_armed
                ? (!cache_->vbr_import_in_progress_ ||
                   cache_->vbr_import_operation_ != operation_)
                : cache_->vbr_import_in_progress_) ||
            !cache_->vbr_operation_armed()) {
            return false;
        }
        uint64_t tier_changes = 0;
        for (const auto & metadata : final_units_) {
            const size_t ikv = metadata.logical_unit/2;
            const bool is_v = (metadata.logical_unit & 1u) != 0;
            if (ikv >= cache_->layers.size()) {
                return false;
            }
            const auto * canonical = is_v
                ? cache_->layers[ikv].v : cache_->layers[ikv].k;
            if (canonical == nullptr) {
                return false;
            }
            tier_changes += canonical->type != metadata.target_type;
        }
        if (!vbr_artifact_epoch_capacity(
                cache_->vbr_tier_epoch_,
                cache_->vbr_representation_epoch_,
                cache_->vbr_checkpoint_epoch_,
                tier_changes)) {
            return false;
        }
        const auto * tracker = cache_->vbr_generation_tracker_get();
        if (!tracker || !tracker->active() || !tracker->stable() ||
            tracker->runtime_instance() != expected.instance_id ||
            !vbr_controller_instance_owned_by(expected.instance_id, tracker)) {
            return false;
        }
        if (!occupied_replacement_) {
            for (const auto & cells : cache_->v_cells) {
                if (cells.get_used() != 0) {
                    return false;
                }
            }
            for (const auto & pool : cache_->vbr_pools_) {
                if (pool.wm_cells != 0) {
                    return false;
                }
            }
        }
        return !vbr_recovery_pending_for_except(
            expected.instance_id, operation_);
    }

    bool arm() noexcept {
        if (test_seam_) {
            armed_ = test_seam_->session_arm(child_id_, operation_);
            return armed_;
        }
        if (!cache_ || cache_->vbr_import_in_progress_) {
            return false;
        }
        cache_->vbr_import_in_progress_ = true;
        cache_->vbr_import_operation_ = operation_;
        armed_ = true;
        return true;
    }

    bool prepare_backing(
            const std::vector<const vbr_validated_child_plan *> & plans,
            const std::vector<vbr_occupied_replacement_relocation_run> * relocation = nullptr) noexcept {
        if (test_seam_) {
            return armed_ && (relocation
                ? test_seam_->session_prepare_relocated_backing(
                    child_id_, plans, *relocation)
                : test_seam_->session_prepare_backing(child_id_, plans));
        }
        if (!armed_ || plans.empty()) {
            return false;
        }
        try {
            size_t shard_count = 0;
            for (const auto * plan : plans) {
                if (!plan || plan->shards.size() > SIZE_MAX-shard_count) {
                    return false;
                }
                shard_count += plan->shards.size();
            }
            unit_complete_.assign(cache_->vbr_generation_tracker_get()->unit_count(), false);
            final_watermarks_.assign(cache_->vbr_pools_.size(), 0);
            pool_indices_.clear();
            extent_prefixes_.clear();
            extent_prefixes_.reserve(shard_count);
            extent_prefix_indices_.clear();
            extent_prefix_indices_.reserve(shard_count);
            unit_by_pool_.clear();
            final_unit_indices_.clear();
            transform_backends_.clear();
            transfer_backends_.clear();
            for (size_t i = 0; i < cache_->vbr_pools_.size(); ++i) {
                pool_indices_[&cache_->vbr_pools_[i]] = i;
            }
            // derived per call; never read outside prepare_backing
            std::vector<uint32_t> backing_watermarks(cache_->vbr_pools_.size(), 0);
            for (const auto * plan : plans) {
                if (!plan || plan->child_id != child_id_ ||
                    plan->logical_unit_id >= unit_complete_.size()) {
                    return false;
                }
                final_units_.push_back({
                    plan->logical_unit_id,
                    plan->stash_action ==
                            vbr_validated_stash_action::restore_exact
                        ? uint32_t(plan->descriptor.clean_stash.valid_rows)
                        : 0,
                    uint8_t((plan->transform_kind ==
                            vbr_import_transform_kind::upward_same_domain ||
                             plan->transform_kind ==
                            vbr_import_transform_kind::upward_cross_domain)
                        ? plan->target_promote_hops
                        : plan->transform_kind ==
                                vbr_import_transform_kind::downward
                            ? 0
                            : plan->descriptor.promote_hops),
                    static_cast<ggml_type>(plan->selected_target_type),
                });
                final_unit_indices_[plan->logical_unit_id] =
                    final_units_.size()-1;
                const size_t ikv = plan->logical_unit_id/2;
                const bool is_v = (plan->logical_unit_id & 1u) != 0;
                if (ikv >= cache_->layers.size()) {
                    return false;
                }
                const auto * live_tensor = is_v
                    ? cache_->layers[ikv].v : cache_->layers[ikv].k;
                if (!live_tensor || plan->descriptor.current_type < 0 ||
                    plan->selected_target_type < 0) {
                    return false;
                }
                const auto source_type = static_cast<ggml_type>(
                    plan->descriptor.current_type);
                const auto target_type = static_cast<ggml_type>(
                    plan->selected_target_type);
                if (source_type != live_tensor->type) {
                    const auto inserted = source_types_.emplace(
                        plan->logical_unit_id, source_type);
                    if (!inserted.second &&
                        inserted.first->second != source_type) {
                        return false;
                    }
                }
                const auto & units = cache_->vbr_units_of(ikv, is_v);
                if (plan->shards.size() != units.size()) {
                    return false;
                }
                for (const auto & unit : units) {
                    unit_by_pool_[{ plan->logical_unit_id, unit.first }] =
                        unit.second;
                }
                for (const auto & shard : plan->shards) {
                    auto * pool = static_cast<llama_kv_cache::vbr_pool *>(
                        const_cast<void *>(shard.target_pool_cookie));
                    const auto pool_index = pool_indices_.find(pool);
                    if (pool_index == pool_indices_.end() ||
                        shard.shard_index >= units.size()) {
                        return false;
                    }
                    const auto & unit = units[shard.shard_index];
                    if (unit.first != pool || !unit.second ||
                        unit.second->t == nullptr || pool->vmm == nullptr ||
                        pool->be == nullptr || pool->backend == nullptr ||
                        uint64_t(ggml_row_size(
                            source_type,
                            unit.second->t->ne[0])) != shard.row_bytes ||
                        uint64_t(ggml_row_size(
                            target_type,
                            unit.second->t->ne[0])) !=
                            shard.target_row_bytes) {
                        return false;
                    }
                    if (plan->transform_kind !=
                            vbr_import_transform_kind::none &&
                        std::find(
                            transform_backends_.begin(),
                            transform_backends_.end(), pool->backend) ==
                                transform_backends_.end()) {
                        transform_backends_.push_back(pool->backend);
                    }
                    if (std::find(
                            transfer_backends_.begin(), transfer_backends_.end(),
                            pool->backend) == transfer_backends_.end()) {
                        transfer_backends_.push_back(pool->backend);
                    }
                    const auto prefix = extent_prefix_indices_.emplace(
                        unit.second, extent_prefixes_.size());
                    if (prefix.second) {
                        extent_prefixes_.push_back({
                            pool, unit.second, pool_index->second,
                            std::max(
                                shard.row_bytes,
                                shard.target_row_bytes),
                            shard.target_row_bytes, pool->wm_cells,
                        });
                    }
                    if (plan->descriptor.wm_cells > UINT32_MAX) {
                        return false;
                    }
                    // wm_cells is the controller's padded live watermark, not
                    // merely max(authorized physical cell)+1.  Preserve it
                    // across import even when the reference owns no row in
                    // the padding tail.  The backing is zero-filled before
                    // authorized H2D, so this does not widen row authority;
                    // it only prevents the first append from growing across
                    // a falsely-short published watermark.
                    const uint32_t descriptor_watermark =
                        uint32_t(plan->descriptor.wm_cells);
                    if (!relocation) {
                        backing_watermarks[pool_index->second] = std::max(
                            backing_watermarks[pool_index->second],
                            descriptor_watermark);
                        final_watermarks_[pool_index->second] = std::max(
                            final_watermarks_[pool_index->second],
                            descriptor_watermark);
                    }
                    const auto validate_run = [&](uint64_t source_first,
                                                  uint32_t destination_first,
                                                  uint32_t cell_count) {
                        if (cell_count == 0 ||
                            destination_first > UINT32_MAX-cell_count ||
                            source_first > UINT64_MAX/shard.row_bytes ||
                            uint64_t(cell_count) > UINT64_MAX/shard.row_bytes) {
                            return false;
                        }
                        const uint64_t relative =
                            uint64_t(source_first)*shard.row_bytes;
                        const uint64_t bytes = uint64_t(cell_count)*shard.row_bytes;
                        if (relative > shard.payload_bytes ||
                            bytes > shard.payload_bytes-relative ||
                            unit.second->byte_off > pool->size ||
                            uint64_t(destination_first)*shard.target_row_bytes >
                                pool->size-unit.second->byte_off ||
                            uint64_t(cell_count)*shard.target_row_bytes >
                                pool->size-unit.second->byte_off-
                                uint64_t(destination_first)*shard.target_row_bytes) {
                            return false;
                        }
                        final_watermarks_[pool_index->second] = std::max(
                            final_watermarks_[pool_index->second],
                            destination_first+cell_count);
                        return true;
                    };
                    if (relocation) {
                        for (const auto & run : *relocation) {
                            if (!validate_run(
                                    run.first_source_packed_row,
                                    run.first_destination_physical_cell,
                                    run.cell_count)) {
                                return false;
                            }
                        }
                    } else for (const auto & run : plan->authorized_runs) {
                        if (run.cell_count == 0 ||
                            run.first_physical_cell >
                                UINT32_MAX-run.cell_count ||
                            uint64_t(run.first_physical_cell) >
                                UINT64_MAX/shard.row_bytes ||
                            uint64_t(run.cell_count) > UINT64_MAX/shard.row_bytes) {
                            return false;
                        }
                        const uint64_t relative =
                            uint64_t(run.first_physical_cell)*shard.row_bytes;
                        const uint64_t bytes =
                            uint64_t(run.cell_count)*shard.row_bytes;
                        if (relative > shard.payload_bytes ||
                            bytes > shard.payload_bytes-relative ||
                            unit.second->byte_off > pool->size ||
                            relative > pool->size-unit.second->byte_off ||
                            bytes > pool->size-unit.second->byte_off-relative) {
                            return false;
                        }
                        final_watermarks_[pool_index->second] = std::max(
                            final_watermarks_[pool_index->second],
                            run.first_physical_cell+run.cell_count);
                    }
                }
            }
            for (size_t i = 0; i < final_watermarks_.size(); ++i) {
                if (relocation) {
                    final_watermarks_[i] = std::max(
                        final_watermarks_[i], cache_->vbr_pools_[i].wm_cells);
                }
                backing_watermarks[i] = std::max(
                    backing_watermarks[i], final_watermarks_[i]);
            }
            // Mapping is an extent-prefix invariant, not merely the union
            // of authorized copy runs.  Foreign rows remain untouched, but
            // every extent is backed over [0, final_watermark) before publish.
            for (const auto & prefix : extent_prefixes_) {
                if (prefix.pool_index >= cache_->vbr_pools_.size() ||
                    &cache_->vbr_pools_[prefix.pool_index] != prefix.pool ||
                    !prefix.extent || prefix.mapped_row_bytes == 0 ||
                    prefix.final_row_bytes == 0 ||
                    prefix.pool->gran == 0) {
                    return false;
                }
                const uint64_t watermark =
                    backing_watermarks[prefix.pool_index];
                if (watermark == 0 ||
                    watermark > UINT64_MAX/prefix.mapped_row_bytes) {
                    return false;
                }
                const uint64_t bytes64 = watermark*prefix.mapped_row_bytes;
                if (prefix.extent->byte_off > prefix.pool->size ||
                    bytes64 > prefix.pool->size-prefix.extent->byte_off ||
                    bytes64 > std::numeric_limits<size_t>::max()) {
                    return false;
                }
                const size_t absolute = prefix.extent->byte_off;
                const size_t bytes = size_t(bytes64);
                const size_t gran = prefix.pool->gran;
                const size_t first = (absolute/gran)*gran;
                if (absolute > std::numeric_limits<size_t>::max()-bytes ||
                    absolute+bytes > std::numeric_limits<size_t>::max()-(gran-1)) {
                    return false;
                }
                const size_t last = ((absolute+bytes+gran-1)/gran)*gran;
                if (last > prefix.pool->size) {
                    return false;
                }
                if (relocation) {
                    if (prefix.initial_watermark >
                            SIZE_MAX/prefix.mapped_row_bytes) {
                        return false;
                    }
                    if (prefix.initial_watermark >
                            SIZE_MAX/prefix.final_row_bytes) {
                        return false;
                    }
                    const size_t old_bytes = size_t(
                        prefix.initial_watermark*prefix.final_row_bytes);
                    if (absolute > SIZE_MAX-old_bytes ||
                        absolute+old_bytes > SIZE_MAX-(gran-1)) {
                        return false;
                    }
                    const size_t old_last =
                        ((absolute+old_bytes+gran-1)/gran)*gran;
                    if (last > old_last) {
                        mapped_.push_back({ prefix.pool, old_last, last-old_last });
                        if (!prefix.pool->be->vmm_pool_map(
                                prefix.pool->vmm, old_last, last-old_last)) {
                            return false;
                        }
                    }
                } else {
                    mapped_.push_back({ prefix.pool, first, last-first });
                    if (!prefix.pool->be->vmm_pool_map(
                            prefix.pool->vmm, absolute, bytes)) {
                        return false;
                    }
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    bool transfer(const vbr_staged_read_descriptor & read,
                  vbr_h2d_chunk_ring * ring,
                  const vbr_h2d_ring_operation * ring_operation,
                  uint64_t fail_completion,
                  vbr_h2d_stats & stats) noexcept {
        if (test_seam_) {
            return armed_ && test_seam_->session_transfer(
                child_id_, read, fail_completion, stats);
        }
        if (!armed_ || !ring || read.child_id != child_id_ ||
            read.kind == vbr_staged_read_kind::companion) {
            return false;
        }
        const size_t ikv = read.logical_unit_id/2;
        const bool is_v = (read.logical_unit_id & 1u) != 0;
        if (ikv >= cache_->layers.size()) {
            return false;
        }
        const auto & units = cache_->vbr_units_of(ikv, is_v);
        if (read.shard_index >= units.size()) {
            return false;
        }
        auto * pool = units[read.shard_index].first;
        auto * extent = units[read.shard_index].second;
        if (!pool || !extent || !extent->t || pool->backend == nullptr ||
            read.lane == UINT32_MAX) {
            return false;
        }
        ggml_tensor source_alias;
        ggml_tensor * destination = extent->t;
        const auto source_type = source_types_.find(read.logical_unit_id);
        const ggml_type destination_type = read.destination_type >= 0
            ? static_cast<ggml_type>(read.destination_type)
            : source_type != source_types_.end()
                ? source_type->second : extent->t->type;
        if (destination_type != extent->t->type) {
            if (!cache_->vbr_import_source_alias(
                    *extent->t, destination_type, source_alias)) {
                return false;
            }
            destination = &source_alias;
        }
        uint64_t destination_offset = read.projection_ranges.empty()
            ? read.source_offset : read.destination_offset;
        if (read.kind == vbr_staged_read_kind::clean_stash) {
            destination = stash_alias(*pool, *extent, read.size);
            destination_offset = 0;
            if (!destination) {
                return false;
            }
        }
        const vbr_artifact_byte_source source = {
            read.source ? read.source->size() : 0,
            read.source.get(),
            [](const void * context, uint64_t offset,
               uint8_t * out, size_t size) noexcept {
                return static_cast<const artifact_segment_chain *>(context)->read(
                    offset, out, size);
            } };
        if (!read.projection_ranges.empty()) {
            if (!ring_operation || !*ring_operation ||
                (read.kind != vbr_staged_read_kind::unit_payload &&
                 read.kind !=
                     vbr_staged_read_kind::recovery_unit_payload)) {
                return false;
            }
            vbr_h2d_packed_transfer transfer;
            transfer.lane = read.lane;
            transfer.source = source;
            transfer.ranges = read.projection_ranges.data();
            transfer.range_count = read.projection_ranges.size();
            transfer.size = read.size;
            transfer.backend = pool->backend;
            transfer.device = ggml_backend_get_device(pool->backend);
            transfer.destination = destination;
            transfer.destination_offset = destination_offset;
            transfer.fail_completion_at = fail_completion;
            return ring->stream_packed_reserved(
                *ring_operation, transfer, stats) == vbr_h2d_status::ok;
        }
        vbr_h2d_transfer transfer;
        transfer.lane = read.lane;
        transfer.source = source;
        transfer.source_offset = read.source_offset;
        transfer.size = read.size;
        transfer.backend = pool->backend;
        transfer.device = ggml_backend_get_device(pool->backend);
        transfer.destination = destination;
        transfer.destination_offset = destination_offset;
        transfer.fail_completion_at = fail_completion;
        return ring->stream(transfer, stats) == vbr_h2d_status::ok;
    }

    bool synchronize_recovery() noexcept {
        if (test_seam_) {
            return armed_ &&
                test_seam_->session_synchronize_recovery(child_id_);
        }
        if (!armed_) {
            return false;
        }
        for (auto * backend : transfer_backends_) {
            if (!backend) {
                return false;
            }
            ggml_backend_synchronize(backend);
        }
        return true;
    }

    bool mark_complete(uint32_t unit) noexcept {
        if (test_seam_) {
            return test_seam_->session_mark_complete(child_id_, unit);
        }
        if (unit >= unit_complete_.size() || unit_complete_[unit]) {
            return false;
        }
        unit_complete_[unit] = true;
        return true;
    }

    bool initialize_transform_backing(
            const vbr_validated_child_plan & plan) noexcept {
        if (test_seam_) {
            if (!armed_) {
                return false;
            }
            return plan.transform_kind ==
                    vbr_import_transform_kind::downward
                ? test_seam_->session_initialize_downward_backing(
                    child_id_, plan)
                : (plan.transform_kind ==
                        vbr_import_transform_kind::upward_same_domain ||
                   plan.transform_kind ==
                        vbr_import_transform_kind::upward_cross_domain) &&
                    test_seam_->session_initialize_upward_backing(
                        child_id_, plan);
        }
        if (!armed_ || plan.transform_kind ==
                vbr_import_transform_kind::none ||
            plan.logical_unit_id/2 >= cache_->layers.size()) {
            return false;
        }
        // The live CUDA edge kernels operate on a dense prefix. Initialize
        // that private backing to zero before overlaying only the authorized
        // reference rows; zeroed foreign positions are transformed but never
        // published, owned, or exposed by the imported reference.
        for (const auto & shard : plan.shards) {
            auto * pool = static_cast<llama_kv_cache::vbr_pool *>(
                const_cast<void *>(shard.target_pool_cookie));
            auto * extent = unit_for_pool(plan.logical_unit_id, pool);
            if (!pool || !extent || !extent->t ||
                !pool->backend || pool->device < 0 ||
                !pool->be || !pool->be->tensor_memset_async ||
                shard.row_bytes == 0 ||
                plan.descriptor.wm_cells > UINT64_MAX/shard.row_bytes) {
                return false;
            }
            const uint64_t bytes =
                plan.descriptor.wm_cells*shard.row_bytes;
            ggml_tensor source_alias;
            if (bytes == 0 || bytes > SIZE_MAX ||
                !cache_->vbr_import_source_alias(
                    *extent->t,
                    static_cast<ggml_type>(plan.descriptor.current_type),
                    source_alias)) {
                return false;
            }
            // Same side-backend stream as the following H2D and transcode
            // submissions: the clear is ordered before authorized row overlays.
            pool->be->tensor_memset_async(
                pool->backend, &source_alias, 0, size_t(bytes));
        }
        return true;
    }

    vbr_downward_transform_status transform_downward(
            const vbr_validated_child_plan & plan,
            bool stashless, uint32_t fail_edge, bool fail_stash,
            uint32_t & stash_valid, uint32_t & edge_reached) noexcept {
        edge_reached = UINT32_MAX;
        if (test_seam_) {
            return test_seam_->session_transform_downward(
                child_id_, plan, stashless, fail_edge, fail_stash,
                stash_valid, edge_reached);
        }
        if (!armed_ || plan.transform_kind !=
                vbr_import_transform_kind::downward || fail_stash) {
            return fail_stash
                ? vbr_downward_transform_status::stash_unavailable
                : vbr_downward_transform_status::invalid_recipe;
        }
        if (fail_edge < plan.transcode_recipe.n_edges) {
            edge_reached = fail_edge;
            return vbr_downward_transform_status::transform_failed;
        }
        return cache_->vbr_downward_transform_import(
            plan, stashless, stash_valid, edge_reached);
    }

    bool transform_upward(
            const vbr_validated_child_plan & plan) noexcept {
        if (test_seam_) {
            const uint32_t stash_rows =
                plan.source_domain == vbr_repr_domain::tapped &&
                (plan.stash_action ==
                     vbr_validated_stash_action::restore_exact ||
                 plan.stash_action ==
                     vbr_validated_stash_action::consume_exact_then_drop)
                    ? uint32_t(plan.descriptor.clean_stash.valid_rows)
                    : 0;
            return test_seam_->session_transform_upward(
                child_id_, plan, stash_rows);
        }
        const auto metadata = final_unit_indices_.find(plan.logical_unit_id);
        if (!armed_ ||
            (plan.transform_kind !=
                 vbr_import_transform_kind::upward_same_domain &&
             plan.transform_kind !=
                 vbr_import_transform_kind::upward_cross_domain) ||
            metadata == final_unit_indices_.end() ||
            !cache_->vbr_upward_transform_import(plan)) {
            return false;
        }
        // prepare_backing derived the final stash and hop metadata from the
        // authenticated validated plan.  The transform changes only the live
        // prefix bytes; preserve that exact terminal metadata for publication.
        return true;
    }

    bool synchronize_transform(
            const std::vector<const vbr_validated_child_plan *> & plans) noexcept {
        if (test_seam_) {
            const auto first = std::find_if(
                plans.begin(), plans.end(), [](const auto * plan) {
                    return plan && plan->transform_kind !=
                        vbr_import_transform_kind::none;
                });
            const auto kind = first == plans.end()
                ? vbr_import_transform_kind::none
                : (*first)->transform_kind;
            if (kind == vbr_import_transform_kind::downward) {
                return test_seam_->session_synchronize_downward(
                    child_id_, plans);
            }
            return (kind == vbr_import_transform_kind::upward_same_domain ||
                    kind == vbr_import_transform_kind::upward_cross_domain) &&
                test_seam_->session_synchronize_upward(child_id_, plans);
        }
        if (!armed_ || plans.empty() ||
            std::any_of(plans.begin(), plans.end(), [](const auto * plan) {
                return plan == nullptr;
            }) ||
            std::none_of(plans.begin(), plans.end(), [](const auto * plan) {
                return plan->transform_kind !=
                    vbr_import_transform_kind::none;
            })) {
            return false;
        }
        for (ggml_backend_t backend : transform_backends_) {
            ggml_backend_synchronize(backend);
        }
        return !transform_backends_.empty();
    }

    bool trim_source_tier_tail(
            const vbr_validated_child_plan & plan,
            uint32_t stash_valid) noexcept {
        if (test_seam_) {
            return test_seam_->session_trim_downward(
                child_id_, plan, stash_valid);
        }
        if (!armed_ || plan.transform_kind !=
                vbr_import_transform_kind::downward) {
            return false;
        }
        // Trim the source-tier tail after the side stream has completed. Keep
        // the rollback journal's physical-range image exact so rollback unmaps
        // only the surviving target prefix rather than relying on double-unmap
        // tolerance from a backend.
        for (const auto & shard : plan.shards) {
            auto * pool = static_cast<llama_kv_cache::vbr_pool *>(
                const_cast<void *>(shard.target_pool_cookie));
            auto * extent = unit_for_pool(plan.logical_unit_id, pool);
            const size_t pool_index = pool_index_of(pool);
            if (!extent || pool_index == SIZE_MAX ||
                pool->gran == 0) {
                return false;
            }
            const uint64_t watermark = final_watermarks_[
                pool_index];
            if (shard.target_row_bytes == 0 ||
                watermark > SIZE_MAX/shard.target_row_bytes) {
                return false;
            }
            const size_t bytes =
                size_t(watermark*shard.target_row_bytes);
            const size_t absolute = extent->byte_off;
            const size_t first = (absolute/pool->gran)*pool->gran;
            if (absolute > SIZE_MAX-bytes ||
                absolute+bytes > SIZE_MAX-(pool->gran-1)) {
                return false;
            }
            const size_t last =
                ((absolute+bytes+pool->gran-1)/pool->gran)*pool->gran;
            const auto range = std::find_if(
                mapped_.begin(), mapped_.end(),
                [&](const mapped_range & value) {
                    return value.pool == pool && value.offset == first;
                });
            if (range == mapped_.end() || last < first ||
                last-first > range->size) {
                return false;
            }
            const size_t final_size = last-first;
            if (range->size > final_size &&
                !pool->be->vmm_pool_unmap(
                    pool->vmm, range->offset+final_size,
                    range->size-final_size)) {
                return false;
            }
            range->size = final_size;
        }
        const auto metadata = final_unit_indices_.find(plan.logical_unit_id);
        if (metadata == final_unit_indices_.end()) {
            return false;
        }
        final_units_[metadata->second].stash_valid = stash_valid;
        final_units_[metadata->second].promote_hops = 0;
        return true;
    }

    bool build_live_image(
            const std::vector<const vbr_validated_child_plan *> & plans,
            const vbr_tracker_install_child & tracker_plan,
            const vbr_checkpoint_generation_controller & source,
            const vbr_occupied_replacement_guard * replacement = nullptr) noexcept {
        if (test_seam_) {
            return replacement
                ? test_seam_->session_build_relocated_live_image(
                    child_id_, plans, tracker_plan, source, *replacement)
                : test_seam_->session_build_live_image(
                    child_id_, plans, tracker_plan, source);
        }
        try {
            for (const auto * plan : plans) {
                if (!plan || plan->logical_unit_id >= unit_complete_.size() ||
                    (!replacement && !unit_complete_[plan->logical_unit_id])) {
                    return false;
                }
            }
            final_cells_.resize(cache_->n_stream);
            final_heads_.assign(cache_->n_stream, 0);
            for (uint32_t stream = 0; stream < cache_->n_stream; ++stream) {
                final_cells_[stream].resize(cache_->v_cells[stream].size());
            }
            final_ownership_ = std::make_unique<vbr_ownership_index>(
                cache_->n_stream, cache_->n_seq_max, cache_->get_size());
            std::vector<vbr_artifact_stream_placement> placements;
            if (replacement) {
                auto & cells = final_cells_.front();
                for (const auto & mapping : replacement->cell_mapping()) {
                    if (mapping.source_stream != 0 ||
                        mapping.destination_physical_cell >= cells.size() ||
                        mapping.logical_position < 0 ||
                        !cells.is_empty(mapping.destination_physical_cell)) {
                        return false;
                    }
                    cells.pos_set(mapping.destination_physical_cell,
                                  mapping.logical_position);
                    cells.ext_set(mapping.destination_physical_cell,
                                  { mapping.ext_x, mapping.ext_y });
                    cells.seq_add(mapping.destination_physical_cell, destination_);
                    if (!final_ownership_->add_cell(
                            0, destination_, mapping.destination_physical_cell,
                            mapping.logical_position)) {
                        return false;
                    }
                }
            } else {
                std::set<std::tuple<uint32_t, uint32_t, llama_pos>> installed;
                for (const auto * plan : plans) {
                    for (const auto & placement : plan->placements) {
                        if (placement.child_id != child_id_ ||
                            placement.stream_index >= final_cells_.size()) {
                            return false;
                        }
                        placements.push_back(placement);
                        auto & cells = final_cells_[placement.stream_index];
                        for (const auto & cell : placement.cells) {
                            if (cell.physical_cell >= cells.size() ||
                                cell.logical_position < 0) {
                                return false;
                            }
                            const auto key = std::make_tuple(
                                placement.stream_index, cell.physical_cell,
                                cell.logical_position);
                            if (installed.insert(key).second) {
                                if (!cells.is_empty(cell.physical_cell)) {
                                    return false;
                                }
                                cells.pos_set(cell.physical_cell,
                                              cell.logical_position);
                                cells.ext_set(cell.physical_cell,
                                    { cell.ext_x, cell.ext_y });
                                cells.seq_add(cell.physical_cell, destination_);
                                if (!final_ownership_->add_cell(
                                        placement.stream_index, destination_,
                                        cell.physical_cell,
                                        cell.logical_position)) {
                                    return false;
                                }
                            } else if (cells.is_empty(cell.physical_cell) ||
                                       cells.pos_get(cell.physical_cell) !=
                                           cell.logical_position ||
                                       !cells.seq_has(cell.physical_cell,
                                                      destination_)) {
                                return false;
                            }
                        }
                    }
                }
            }
            // The allocation cursor is live metadata, not a mere search
            // optimization: unified/SWA placement may legally recycle a
            // masked cell before a zero head reaches the imported frontier.
            // Resume immediately after the highest installed physical cell,
            // matching an ordinary sequentially-built prefix.
            for (size_t stream = 0; stream < final_cells_.size(); ++stream) {
                const uint32_t next = final_cells_[stream].used_max_p1();
                final_heads_[stream] = next < final_cells_[stream].size()
                    ? next : 0;
            }
            auto * tracker = cache_->vbr_generation_tracker_mut();
            if (!tracker || (replacement
                    ? !tracker->prepare_relocated_import_image(
                        tracker_plan, source, destination_, *replacement,
                        tracker_image_)
                    : !tracker->prepare_import_image(
                        tracker_plan, source, destination_, placements,
                        tracker_image_))) {
                return false;
            }
            // Validation writes the target cursor on every unit plan, even
            // when a coherent projection changes only a subset of units. The
            // cursor is tree-wide, so publication must never reconstruct it from
            // the source policy of an unchanged unit.
            final_cursor_ = plans.front()->target_controller_cursor;
            for (const auto * plan : plans) {
                if (plan->target_controller_cursor != final_cursor_) {
                    return false;
                }
            }
            image_ready_ = true;
            return true;
        } catch (...) {
            return false;
        }
    }

    bool barrier(uint64_t ledger_serial,
                 const vbr_validated_manifest & manifest) const noexcept {
        if (test_seam_) {
            return test_seam_->session_barrier(
                child_id_, ledger_serial, manifest);
        }
        const bool source_ready = manifest.is_prefix_projection()
            ? manifest.projection_transfer_ready()
            : manifest.is_occupied_replacement()
                ? manifest.occupied_replacement() != nullptr &&
                    manifest.occupied_replacement()->ready()
                : manifest.source_package().validate() ==
                    vbr_artifact_status::ok;
        if (!armed_ || !image_ready_ || !cache_->vbr_import_in_progress_ ||
            cache_->vbr_import_operation_ != operation_ || !source_ready ||
            !cache_->vbr_generation_tracker_get()->import_image_installable(
                tracker_image_, operation_)) {
            return false;
        }
        for (const auto complete : unit_complete_) {
            if (!complete) {
                return false;
            }
        }
        return ledger_serial != 0;
    }

    bool mapped_prefixes_complete() const noexcept {
        if (test_seam_) {
            return test_seam_->session_mapped_prefixes_complete(child_id_);
        }
        for (const auto & prefix : extent_prefixes_) {
            if (!prefix.pool || !prefix.extent || prefix.final_row_bytes == 0) {
                return false;
            }
            const auto pool_it = std::find_if(
                cache_->vbr_pools_.begin(), cache_->vbr_pools_.end(),
                [&](const llama_kv_cache::vbr_pool & value) {
                    return &value == prefix.pool;
                });
            if (pool_it == cache_->vbr_pools_.end()) {
                return false;
            }
            const uint64_t watermark = final_watermarks_[
                size_t(pool_it-cache_->vbr_pools_.begin())];
            if (watermark == 0 ||
                watermark > UINT64_MAX/prefix.final_row_bytes) {
                return false;
            }
            const uint64_t bytes64 = watermark*prefix.final_row_bytes;
            if (bytes64 > std::numeric_limits<size_t>::max()) {
                return false;
            }
            const size_t start = prefix.extent->byte_off;
            const size_t bytes = size_t(bytes64);
            if (occupied_replacement_ &&
                prefix.initial_watermark > SIZE_MAX/prefix.final_row_bytes) {
                return false;
            }
            const size_t initial_bytes = occupied_replacement_
                ? size_t(prefix.initial_watermark*prefix.final_row_bytes) : 0;
            if (start > SIZE_MAX-initial_bytes ||
                start+initial_bytes > SIZE_MAX-(prefix.pool->gran-1)) {
                return false;
            }
            const size_t old_last = occupied_replacement_
                ? ((start+initial_bytes+prefix.pool->gran-1)/prefix.pool->gran)*
                    prefix.pool->gran
                : start;
            if (occupied_replacement_) {
                if (start > SIZE_MAX-bytes ||
                    start+bytes > SIZE_MAX-(prefix.pool->gran-1)) {
                    return false;
                }
                const size_t new_last =
                    ((start+bytes+prefix.pool->gran-1)/prefix.pool->gran)*
                        prefix.pool->gran;
                if (new_last <= old_last) {
                    continue;
                }
            }
            const size_t required_start = old_last;
            const auto covered = std::find_if(
                mapped_.begin(), mapped_.end(),
                [&](const mapped_range & range) {
                    return range.pool == prefix.pool &&
                           range.offset <= required_start &&
                           range.size <= std::numeric_limits<size_t>::max()-range.offset &&
                           start <= std::numeric_limits<size_t>::max()-bytes &&
                           range.offset+range.size >= start+bytes;
                });
            if (covered == mapped_.end()) {
                return false;
            }
        }
        return !extent_prefixes_.empty();
    }

    // BEGIN VBR_IMPORT_KV_METADATA_SWAP
    void publish_metadata() noexcept {
        if (test_seam_) {
            test_seam_->session_publish_metadata(child_id_);
            published_ = true;
            armed_ = false;
            return;
        }
        GGML_ASSERT(armed_ && image_ready_ && !published_);
        GGML_ASSERT(mapped_prefixes_complete());
        for (size_t i = 0; i < final_cells_.size(); ++i) {
            std::swap(cache_->v_cells[i], final_cells_[i]);
        }
        cache_->v_heads.swap(final_heads_);
        cache_->vbr_ownership_.swap(final_ownership_);
        cache_->vbr_generation_tracker_mut()->install_import_image_swap(
            tracker_image_);
        for (size_t i = 0; i < final_watermarks_.size(); ++i) {
            cache_->vbr_pools_[i].wm_cells = final_watermarks_[i];
        }
        // Clean-stash bytes were written directly into each pool's fixed-VA stash slab at the
        // extents' construction-assigned offsets during staging; publishing stash_valid below
        // is what makes them visible.
        for (const auto & metadata : final_units_) {
            const size_t ikv = metadata.logical_unit/2;
            const bool is_v = (metadata.logical_unit & 1u) != 0;
            for (const auto & unit : cache_->vbr_units_of(ikv, is_v)) {
                unit.second->stash_valid = metadata.stash_valid;
                unit.second->promote_hops = metadata.promote_hops;
            }
            const auto * canonical = is_v
                ? cache_->layers[ikv].v : cache_->layers[ikv].k;
            GGML_ASSERT(canonical != nullptr);
            if (canonical->type != metadata.target_type) {
                // Graphs bake K/V types and strides. This is the import
                // analogue of every live retier publication: fence graph
                // reuse before changing the canonical tensor in place.
                GGML_ASSERT(cache_->vbr_tier_epoch_ != UINT64_MAX);
                ++cache_->vbr_tier_epoch_;
            }
            cache_->vbr_import_set_unit_type_noalloc(
                metadata.logical_unit, metadata.target_type);
        }
        cache_->vbr_degrade_cursor_ = size_t(final_cursor_);
        cache_->vbr_capture_retier_deferred_.clear();
        cache_->vbr_attention_content_changed();
        published_ = true;
        armed_ = false;
    }

    void finish_publish() noexcept {
        if (test_seam_) {
            test_seam_->session_finish_publish(child_id_);
            return;
        }
        GGML_ASSERT(cache_->vbr_import_in_progress_ && published_);
        cache_->vbr_import_operation_ = {};
        cache_->vbr_import_in_progress_ = false;
    }
    // END VBR_IMPORT_KV_METADATA_SWAP

    bool prepare_receipts(
            const std::shared_ptr<vbr_import_receipt_group> & receipt) noexcept {
        if (test_seam_) {
            return receipt && test_seam_->session_prepare_receipts(
                child_id_, receipt);
        }
        if (!receipt || receipt_ ||
            (!occupied_replacement_ && cache_->vbr_import_receipt_)) {
            return false;
        }
        receipt_ = receipt;
        return true;
    }

    void publish_receipts() noexcept {
        if (test_seam_) {
            test_seam_->session_publish_receipts(child_id_);
            return;
        }
        GGML_ASSERT(published_ && receipt_);
        if (occupied_replacement_) {
            cache_->vbr_import_receipt_.swap(receipt_);
        } else {
            GGML_ASSERT(!cache_->vbr_import_receipt_);
            cache_->vbr_import_receipt_ = receipt_;
            receipt_.reset();
        }
    }

    bool rollback(bool inject_failure) noexcept {
        if (!armed_ || published_) {
            return true;
        }
        if (test_seam_) {
            const bool ok = test_seam_->session_rollback(
                child_id_, inject_failure);
            armed_ = false;
            return ok;
        }
        bool ok = !inject_failure;
        std::set<std::tuple<llama_kv_cache::vbr_pool *, size_t, size_t>> done;
        for (auto it = mapped_.rbegin(); it != mapped_.rend(); ++it) {
            if (!it->pool || !it->pool->be || !it->pool->vmm) {
                ok = false;
                continue;
            }
            const auto key = std::make_tuple(it->pool, it->offset, it->size);
            if (done.insert(key).second &&
                !it->pool->be->vmm_pool_unmap(
                    it->pool->vmm, it->offset, it->size)) {
                ok = false;
            }
        }
        cache_->vbr_import_operation_ = {};
        cache_->vbr_import_in_progress_ = false;
        mapped_.clear();
        armed_ = false;
        return ok;
    }

    llama_kv_cache * cache() const noexcept { return cache_; }
    uint32_t child_id() const noexcept { return child_id_; }
    bool companion_layout(vbr_companion_attention_layout & output) const noexcept {
        output = {};
        output.child_id = child_id_;
        if (!image_ready_ || !cache_ || final_cells_.size() != cache_->n_stream) {
            return false;
        }
        try {
            for (uint32_t stream = 0; stream < final_cells_.size(); ++stream) {
                const auto & cells = final_cells_[stream];
                for (uint32_t physical = 0; physical < cells.size(); ++physical) {
                    if (!cells.seq_has(physical, destination_)) {
                        continue;
                    }
                    const auto ext = cells.ext_get(physical);
                    output.cells.push_back({ stream, physical,
                        cells.pos_get(physical), ext.x, ext.y, ext.tok });
                }
            }
            return !output.cells.empty();
        } catch (...) {
            output = {};
            return false;
        }
    }
    vbr_generation_teardown_state tracker_teardown_state() const noexcept {
        if (test_seam_) {
            return vbr_generation_teardown_state::clean;
        }
        const auto * tracker =
            cache_ ? cache_->vbr_generation_tracker_get() : nullptr;
        return tracker ? tracker->teardown_state()
                       : vbr_generation_teardown_state::instance_owner_mismatch;
    }

  private:
    // The pool's f16 sink stash is a fixed-VA VMM slab with construction-assigned extent
    // offsets (recoverable-stash rework). Import writes the clean stash DIRECTLY into the
    // destination extent's slot: the adopt target is construction-empty and readers ignore
    // stash content until composite publication sets stash_valid, so a failed import leaves only
    // ignored bytes behind. Mapping goes through the pool's idempotent grow-only reserve,
    // keeping allocation failure recoverable and outside the no-fail boundary.
    ggml_tensor * stash_alias(llama_kv_cache::vbr_pool & pool,
                              llama_kv_cache::vbr_extent & extent,
                              uint64_t bytes) noexcept {
        try {
            if (extent.t == nullptr || bytes == 0 ||
                bytes > std::numeric_limits<size_t>::max()) {
                return nullptr;
            }
            const size_t row_bytes =
                size_t(extent.t->ne[0]) * sizeof(uint16_t);
            if (row_bytes == 0 || bytes % row_bytes != 0 ||
                bytes / row_bytes > cache_->vbr_stash_rows_) {
                return nullptr;
            }
            const std::vector<llama_kv_cache::vbr_stash_request> requests = {
                { &extent, uint32_t(bytes / row_bytes) },
            };
            // unit_for_pool already proved this exact extent belongs to pool;
            // retain all size/offset checks without rescanning pool inventory
            // once per authenticated stash read.
            if (!cache_->vbr_stash_reserve_trusted(pool, requests) ||
                pool.stash_vmm == nullptr ||
                extent.stash_off > pool.stash_size ||
                bytes > pool.stash_size - extent.stash_off) {
                return nullptr;
            }
            ggml_init_params params = {
                2*ggml_tensor_overhead(), nullptr, true,
            };
            ggml_context_ptr context { ggml_init(params) };
            if (!context) {
                return nullptr;
            }
            ggml_tensor * alias = ggml_new_tensor_1d(
                context.get(), GGML_TYPE_I8, int64_t(bytes));
            // Borrow the extent's pool buffer for the backend same-device buft assert;
            // the H2D copy writes through alias->data directly.
            alias->buffer = extent.t->buffer;
            alias->data = static_cast<char *>(
                pool.be->vmm_pool_base(pool.stash_vmm)) + extent.stash_off;
            stash_contexts_.push_back(std::move(context));
            return alias;
        } catch (...) {
            return nullptr;
        }
    }

    llama_kv_cache::vbr_extent * unit_for_pool(
            uint32_t logical_unit, llama_kv_cache::vbr_pool * pool) const noexcept {
        const auto it = unit_by_pool_.find({ logical_unit, pool });
        return it == unit_by_pool_.end() ? nullptr : it->second;
    }

    size_t pool_index_of(
            llama_kv_cache::vbr_pool * pool) const noexcept {
        const auto it = pool_indices_.find(pool);
        return it == pool_indices_.end() ? SIZE_MAX : it->second;
    }

    llama_kv_cache * cache_ = nullptr;
    uint32_t child_id_ = UINT32_MAX;
    llama_seq_id destination_ = -1;
    vbr_operation_id operation_ = {};
    bool armed_ = false;
    bool image_ready_ = false;
    bool published_ = false;
    bool occupied_replacement_ = false;
    std::vector<mapped_range> mapped_;
    std::vector<extent_prefix> extent_prefixes_;
    std::unordered_map<llama_kv_cache::vbr_extent *, size_t>
        extent_prefix_indices_;
    std::vector<ggml_context_ptr> stash_contexts_;
    std::vector<bool> unit_complete_;
    std::vector<final_unit_metadata> final_units_;
    std::map<llama_kv_cache::vbr_pool *, size_t> pool_indices_;
    std::map<std::pair<uint32_t, llama_kv_cache::vbr_pool *>,
             llama_kv_cache::vbr_extent *> unit_by_pool_;
    std::map<uint32_t, size_t> final_unit_indices_;
    std::map<uint32_t, ggml_type> source_types_;
    std::vector<ggml_backend_t> transform_backends_;
    std::vector<ggml_backend_t> transfer_backends_;
    std::vector<uint32_t> final_watermarks_;
    std::vector<llama_kv_cells> final_cells_;
    std::vector<uint32_t> final_heads_;
    std::unique_ptr<vbr_ownership_index> final_ownership_;
    vbr_tracker_import_image tracker_image_;
    uint64_t final_cursor_ = 0;
    std::shared_ptr<vbr_import_receipt_group> receipt_;
    vbr_adopt_test_seam * test_seam_ = nullptr;
};

struct vbr_adopt_child_work {
    llama_kv_cache * cache = nullptr;
    std::vector<const vbr_validated_child_plan *> plans;
    std::unique_ptr<vbr_kv_import_session> session;
};

namespace {

class import_operation_scope {
  public:
    enum class open_status : uint8_t {
        ok,
        operation_unavailable,
        recovery_unavailable,
    };

    explicit import_operation_scope(
            vbr_adopt_test_seam * test_seam = nullptr) noexcept
        : test_seam_(test_seam) {}

    open_status open(const vbr_validated_manifest & manifest,
                     llama_seq_id destination,
                     const std::vector<llama_memory_tree_child> & tree) {
        if (test_seam_) {
            const auto status = test_seam_->operation_open(
                manifest, destination, tree, instances_, test_operation_);
            test_open_ = status == vbr_adopt_status::adopted;
            if (status == vbr_adopt_status::recovery_unavailable) {
                return open_status::recovery_unavailable;
            }
            return test_open_ ? open_status::ok :
                open_status::operation_unavailable;
        }
        vbr_operation_binding binding;
        binding.kind = vbr_operation_kind::state_import;
        binding.child_phase = vbr_operation_phase::mutate;
        std::vector<vbr_controller_instance_id> seen;
        for (const auto & child : manifest.tracker_install().children) {
            if (std::find(seen.begin(), seen.end(), child.target_instance) !=
                    seen.end()) {
                continue;
            }
            if (!vbr_binding_add_instance_target(
                    binding, vbr_operation_kind::state_import,
                    vbr_operation_class::state_api,
                    child.target_instance, VBR_STREAM_ANY,
                    destination, 0,
                    std::numeric_limits<llama_pos>::max())) {
                return open_status::operation_unavailable;
            }
            seen.push_back(child.target_instance);
        }
        if (seen.empty()) {
            return open_status::operation_unavailable;
        }
        operation_ = std::make_unique<vbr_scoped_operation>(binding);
        if (!*operation_) {
            operation_.reset();
            return open_status::operation_unavailable;
        }
        instances_ = std::move(seen);
        for (const auto instance : instances_) {
            const int32_t recovery =
                vbr_recovery_reserve(operation_->id(), instance);
            if (recovery < 0) {
                return open_status::recovery_unavailable;
            }
            recoveries_.push_back(recovery);
        }
        for (const auto & child : tree) {
            if (!child.attention) {
                continue;
            }
            const auto instance = child.attention->vbr_instance_id();
            if (std::find(instances_.begin(), instances_.end(), instance) ==
                    instances_.end()) {
                continue;
            }
            if (!child.attention->vbr_retier_freeze_begin(
                    "artifact_import", operation_->id())) {
                return open_status::operation_unavailable;
            }
            frozen_.push_back(child.attention);
        }
        return frozen_.size() == instances_.size()
            ? open_status::ok : open_status::operation_unavailable;
    }

    ~import_operation_scope() {
        finish(false, false);
    }

    vbr_operation_id id() const noexcept {
        if (test_seam_) {
            return test_open_ ? test_operation_ : vbr_operation_id{};
        }
        return operation_ ? operation_->id() : vbr_operation_id{};
    }
    const std::vector<vbr_controller_instance_id> & instances() const noexcept {
        return instances_;
    }

    bool finish(bool committed, bool quarantine) noexcept {
        if (test_seam_) {
            if (test_open_) {
                test_seam_->operation_finish(
                    test_operation_, committed, quarantine);
                test_open_ = false;
            }
            return true;
        }
        if (!operation_) {
            return true;
        }
        bool terminal_clean = true;
        if (quarantine) {
            for (const auto index : recoveries_) {
                if (vbr_recovery_record_failure(
                        index, operation_->id(),
                        vbr_operation_phase::cleanup,
                        vbr_recovery_failure_site::exception_unwind,
                        true)) {
                    auto capability = vbr_recovery_mint(index);
                    if (capability) {
                        capability.resolve_quarantined();
                    }
                }
            }
        } else {
            for (const auto index : recoveries_) {
                terminal_clean =
                    vbr_recovery_release_unused(index, operation_->id()) &&
                    terminal_clean;
            }
        }
        for (auto it = frozen_.rbegin(); it != frozen_.rend(); ++it) {
            (*it)->vbr_retier_freeze_end("artifact_import", operation_->id());
        }
        terminal_clean = operation_->close(committed
                ? vbr_operation_outcome::committed
                : vbr_operation_outcome::aborted) &&
            terminal_clean;
        operation_.reset();
        if (!quarantine) {
            for (const auto instance : instances_) {
                terminal_clean =
                    !vbr_recovery_owned_by(instance) &&
                    vbr_operation_registry_quiescent_for(&instance, 1) &&
                    terminal_clean;
            }
        }
        recoveries_.clear();
        frozen_.clear();
        return terminal_clean;
    }

  private:
    vbr_adopt_test_seam * test_seam_ = nullptr;
    vbr_operation_id test_operation_ = {};
    bool test_open_ = false;
    std::unique_ptr<vbr_scoped_operation> operation_;
    std::vector<vbr_controller_instance_id> instances_;
    std::vector<int32_t> recoveries_;
    std::vector<llama_kv_cache *> frozen_;
};

bool fault_before(const vbr_composite_publish_hooks & hooks,
                  vbr_adopt_phase phase) noexcept {
    if (hooks.test == nullptr) {
        return false;
    }
    return hooks.test->fault.fail_before == phase ||
           (hooks.test->target != nullptr &&
            hooks.test->target->phase_boundary(phase, false));
}

bool fault_after(const vbr_composite_publish_hooks & hooks,
                 vbr_adopt_phase phase) noexcept {
    if (hooks.test == nullptr) {
        return false;
    }
    return hooks.test->fault.fail_after == phase ||
           (hooks.test->target != nullptr &&
            hooks.test->target->phase_boundary(phase, true));
}

vbr_adopt_test_seam * test_target(
        const vbr_composite_publish_hooks & hooks) noexcept {
    return hooks.test ? hooks.test->target : nullptr;
}

bool collect_adopt_tree(
        llama_memory_i & target,
        const vbr_composite_publish_hooks & hooks,
        std::vector<llama_memory_tree_child> & output) noexcept {
    auto * seam = test_target(hooks);
    return seam
        ? seam->collect_tree(target, output)
        : llama_memory_tree_collect(&target, output);
}

bool operation_quiescent(
        const import_operation_scope & operation,
        const vbr_composite_publish_hooks & hooks) noexcept {
    auto * seam = test_target(hooks);
    return seam
        ? seam->operation_quiescent(
              operation.instances(), operation.id())
        : vbr_operation_registry_quiescent_for_except(
              operation.instances().data(), operation.instances().size(),
              operation.id());
}

vbr_adopt_status transaction_status(
        llama_cache_transaction_status status) noexcept {
    switch (status) {
        case llama_cache_transaction_status::committed:
            return vbr_adopt_status::adopted;
        case llama_cache_transaction_status::admission_refused:
            return vbr_adopt_status::admission_refused;
        case llama_cache_transaction_status::invalid_argument:
            return vbr_adopt_status::accounting_unavailable;
        case llama_cache_transaction_status::after_admit_failed:
        case llama_cache_transaction_status::stage_failed:
        case llama_cache_transaction_status::commit_failed:
        case llama_cache_transaction_status::post_commit_fault:
        case llama_cache_transaction_status::internal_fault:
        case llama_cache_transaction_status::_count:
            return vbr_adopt_status::accounting_unavailable;
    }
    return vbr_adopt_status::accounting_unavailable;
}

vbr_adopt_status transform_source_h2d(
        std::map<uint32_t, vbr_adopt_child_work> & children,
        const vbr_validated_manifest & manifest) noexcept {
    for (const auto & plan : manifest.children()) {
        if (plan.transform_kind == vbr_import_transform_kind::none) {
            continue;
        }
        const auto child = children.find(plan.child_id);
        if (child == children.end() ||
            !child->second.session->initialize_transform_backing(plan)) {
            return vbr_adopt_status::transfer_failed;
        }
    }
    return vbr_adopt_status::adopted;
}

vbr_adopt_status transform_all(
        std::map<uint32_t, vbr_adopt_child_work> & children,
        const vbr_validated_manifest & manifest,
        const vbr_staged_payloads & staged,
        const vbr_composite_publish_hooks & hooks,
        std::vector<uint32_t> & stash_valid,
        vbr_adopt_result & out) noexcept {
    const auto & stashless = staged.transform_stashless_units();
    if (stash_valid.size() != manifest.children().size()) {
        return vbr_adopt_status::internal_error;
    }
    for (size_t i = 0; i < manifest.children().size(); ++i) {
        const auto & plan = manifest.children()[i];
        if (plan.transform_kind == vbr_import_transform_kind::none) {
            continue;
        }
        const auto child = children.find(plan.child_id);
        if (child == children.end()) {
            return plan.transform_kind ==
                    vbr_import_transform_kind::downward
                ? vbr_adopt_status::downward_recipe_invalid
                : vbr_adopt_status::upward_recipe_invalid;
        }
        if (plan.transform_kind ==
                vbr_import_transform_kind::upward_same_domain ||
            plan.transform_kind ==
                vbr_import_transform_kind::upward_cross_domain) {
            if (!child->second.session->transform_upward(plan)) {
                return vbr_adopt_status::upward_transform_failed;
            }
            continue;
        }
        const uint64_t unit_key = vbr_downward_unit_key(
            plan.child_id, plan.logical_unit_id);
        const bool omit_stash = std::binary_search(
            stashless.begin(), stashless.end(), unit_key);
        uint32_t valid = 0;
        uint32_t edge = UINT32_MAX;
        out.downward_subphase = vbr_downward_adopt_subphase::edge_transcode;
        const auto status = child->second.session->transform_downward(
            plan, omit_stash,
            hooks.test ? hooks.test->fault.fail_downward_edge : UINT32_MAX,
            hooks.test && hooks.test->fault.fail_downward_stash,
            valid, edge);
        out.downward_edge = edge;
        switch (status) {
            case vbr_downward_transform_status::transformed:
                stash_valid[i] = valid;
                break;
            case vbr_downward_transform_status::invalid_recipe:
                return vbr_adopt_status::downward_recipe_invalid;
            case vbr_downward_transform_status::stash_unavailable:
                out.downward_subphase =
                    vbr_downward_adopt_subphase::edge_stash_capture;
                return vbr_adopt_status::downward_stash_unavailable;
            case vbr_downward_transform_status::transform_failed:
            case vbr_downward_transform_status::internal_error:
            case vbr_downward_transform_status::_count:
                return vbr_adopt_status::downward_transform_failed;
        }
    }
    // Every edge is now enqueued. Synchronize each distinct side backend once
    // before the completion fault seam or any destructive tail trim.
    for (auto & child : children) {
        const auto transformed = std::find_if(
            child.second.plans.begin(), child.second.plans.end(),
            [](const auto * plan) {
                return plan && plan->transform_kind !=
                    vbr_import_transform_kind::none;
            });
        if (transformed != child.second.plans.end() &&
            !child.second.session->synchronize_transform(
                child.second.plans)) {
            return (*transformed)->transform_kind ==
                    vbr_import_transform_kind::downward
                ? vbr_adopt_status::downward_transform_failed
                : vbr_adopt_status::upward_transform_failed;
        }
    }
    if (manifest.decision() == vbr_import_decision::downward_rebase) {
        out.downward_subphase =
            vbr_downward_adopt_subphase::edge_completion;
        if (hooks.test && hooks.test->fault.fail_downward_completion) {
            return vbr_adopt_status::downward_transform_failed;
        }
    }
    for (size_t i = 0; i < manifest.children().size(); ++i) {
        const auto & plan = manifest.children()[i];
        if (plan.transform_kind != vbr_import_transform_kind::downward) {
            continue;
        }
        const auto child = children.find(plan.child_id);
        if (child == children.end() ||
            !child->second.session->trim_source_tier_tail(
                plan, stash_valid[i])) {
            return vbr_adopt_status::downward_transform_failed;
        }
    }
    return vbr_adopt_status::adopted;
}

} // namespace

vbr_adopt_result vbr_adopt_empty_manifest(
        llama_memory_i & target,
        llama_seq_id destination,
        vbr_validated_manifest && manifest_value,
        vbr_staged_payloads && staged_value,
        llama_cache_acct_ledger & accounting,
        const vbr_composite_publish_hooks & server_hooks) noexcept {
    vbr_adopt_result out;
    std::unique_ptr<vbr_validated_manifest> manifest;
    std::unique_ptr<vbr_staged_payloads> staged;
    import_operation_scope operation(test_target(server_hooks));
    std::map<uint32_t, vbr_adopt_child_work> children;
    std::vector<std::pair<const vbr_companion_adoption_provider *,
                          std::unique_ptr<vbr_prepared_companion_image>>>
        companions;
    std::vector<uint32_t> transform_stash_valid;
    bool claims_committed = false;
    bool published = false;
    bool recycle = false;
    bool destructive_started = false;
    uint64_t post_commit_serial = 0;
    std::shared_ptr<vbr_import_receipt_group> receipt_group;
    vbr_h2d_ring_operation packed_operation;

    const auto fail = [&](vbr_adopt_status status) {
        out.status = status;
        if (!published) {
            bool rollback_ok = true;
            if (destructive_started) {
                bool replay_ok = recycle && staged &&
                    (test_target(server_hooks) || packed_operation);
                uint64_t recovery_reads = 0;
                if (replay_ok) {
                    for (const auto & read : staged->reads()) {
                        if (read.kind !=
                                vbr_staged_read_kind::recovery_unit_payload) {
                            continue;
                        }
                        ++recovery_reads;
                        const auto child = children.find(read.child_id);
                        vbr_h2d_stats stats;
                        if (child == children.end() || !child->second.session ||
                            !child->second.session->transfer(
                                read, staged->adoption_ring(),
                                test_target(server_hooks)
                                    ? nullptr : &packed_operation,
                                UINT64_MAX, stats)) {
                            replay_ok = false;
                            break;
                        }
                        out.recovery_h2d_bytes += stats.bytes;
                        out.recovery_h2d_chunks += stats.chunks;
                    }
                }
                replay_ok = replay_ok && recovery_reads != 0;
                if (replay_ok) {
                    for (auto & entry : children) {
                        replay_ok = entry.second.session &&
                            entry.second.session->synchronize_recovery() && replay_ok;
                    }
                }
                if (replay_ok) {
                    out.recovery = vbr_adopt_recovery_outcome::replayed;
                } else {
                    out.recovery = vbr_adopt_recovery_outcome::quarantined;
                    rollback_ok = false;
                }
                destructive_started = false;
            }
            for (auto it = children.rbegin(); it != children.rend(); ++it) {
                if (it->second.session) {
                    rollback_ok = it->second.session->rollback(
                                      server_hooks.test != nullptr &&
                                      server_hooks.test->fault.fail_rollback) &&
                                  rollback_ok;
                }
            }
            for (auto it = companions.rbegin(); it != companions.rend(); ++it) {
                if (it->second && it->first->rollback) {
                    rollback_ok = it->first->rollback(
                                      it->first->context, *it->second) &&
                                  rollback_ok;
                }
            }
            if (claims_committed && staged) {
                for (const auto op : staged->adoption_committed_ops()) {
                    if (op && accounting.release(op)) {
                        ++out.rollback_count;
                    }
                }
            }
            rollback_ok = operation.finish(false, !rollback_ok) && rollback_ok;
            if (!rollback_ok) {
                out.status = vbr_adopt_status::quarantined;
                if (recycle) {
                    out.recovery = vbr_adopt_recovery_outcome::quarantined;
                }
                out.phase = vbr_adopt_phase::rollback;
            }
        }
        return out;
    };

    try {
        out.phase = vbr_adopt_phase::consume_capabilities;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::internal_error);
        }
        manifest.reset(new vbr_validated_manifest(std::move(manifest_value)));
        staged.reset(new vbr_staged_payloads(std::move(staged_value)));
        if (!manifest || !staged ||
            (!manifest->is_prefix_projection() &&
             !manifest->source_package()) ||
            (manifest->is_prefix_projection() &&
             !manifest->projection_transfer_ready()) ||
            manifest->adoption_nonce() == 0 ||
            manifest->adoption_nonce() != staged->adoption_nonce() ||
            manifest->manifest_digest() != staged->manifest_digest() ||
            !fingerprint_equal(manifest->target(), staged->target_fingerprint()) ||
            manifest->decision() != staged->decision() ||
            destination < 0 ||
            server_hooks.validate_owner_token == nullptr ||
            server_hooks.owner_token == nullptr ||
            !server_hooks.validate_owner_token(
                server_hooks.context, server_hooks.owner_token, &target)) {
            return fail(vbr_adopt_status::invalid_capability_pair);
        }
        out.decision = manifest->decision();
        const bool occupied_replacement =
            manifest->is_occupied_replacement();
        recycle = occupied_replacement &&
            manifest->occupied_replacement()->strategy() ==
                vbr_occupied_replacement_strategy::recycle_incumbent_cells;
        if (out.decision == vbr_import_decision::downward_rebase ||
            out.decision == vbr_import_decision::upward_reconstruct) {
            transform_stash_valid.assign(manifest->children().size(), 0);
        }
        out.consistency = manifest->decision() == vbr_import_decision::native_import
            ? vbr_artifact_consistency_kind::capture_exact
            : vbr_artifact_consistency_kind::live_rebased;
        if (out.decision != vbr_import_decision::native_import &&
            out.decision != vbr_import_decision::live_rebased &&
            out.decision != vbr_import_decision::downward_rebase &&
            out.decision != vbr_import_decision::upward_reconstruct) {
            return fail(vbr_adopt_status::unsupported_decision);
        }
        if ((out.decision == vbr_import_decision::downward_rebase ||
             out.decision == vbr_import_decision::upward_reconstruct) &&
            !staged->transform_resources_ready()) {
            return fail(vbr_adopt_status::accounting_unavailable);
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::internal_error);
        }
        if (accounting.snapshot().serial !=
                staged->accounting_serial_after_prepare() ||
            staged->accounting_serial_after_prepare() == 0 ||
            !staged->claims_ready() ||
            fault_before(server_hooks, vbr_adopt_phase::operation_open)) {
            return fail(vbr_adopt_status::accounting_unavailable);
        }

        std::vector<llama_memory_tree_child> tree;
        if (!collect_adopt_tree(target, server_hooks, tree)) {
            return fail(vbr_adopt_status::target_drift);
        }
        out.phase = vbr_adopt_phase::operation_open;
        const auto open_status = operation.open(*manifest, destination, tree);
        if (open_status != import_operation_scope::open_status::ok) {
            return fail(open_status ==
                    import_operation_scope::open_status::recovery_unavailable
                ? vbr_adopt_status::recovery_unavailable
                : vbr_adopt_status::operation_unavailable);
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::operation_unavailable);
        }

        out.phase = vbr_adopt_phase::target_recheck;
        auto staged_fingerprint = manifest->target();
        staged_fingerprint.accounting_serial =
            staged->accounting_serial_after_prepare();
        const bool replacement_current = !occupied_replacement ||
            (test_target(server_hooks)
                ? test_target(server_hooks)->session_recheck_occupied(
                    0, *manifest->occupied_replacement(), false)
                : vbr_explicit_recheck_occupied_replacement_guard(
                    target, destination,
                    manifest->occupied_replacement()->accounting_serial(),
                    manifest->occupied_representation_context_,
                    manifest->occupied_representation_identity_,
                    *manifest->occupied_replacement_) ==
                        vbr_occupied_replacement_guard_status::ready);
        if (fault_before(server_hooks, out.phase) ||
            accounting.snapshot().serial != staged->accounting_serial_after_prepare() ||
            manifest->read_accounting_serial_ == nullptr ||
            manifest->read_policy_epoch_ == nullptr ||
            !replacement_current ||
            (!occupied_replacement &&
             (manifest->recheck_target_empty_ == nullptr ||
              !manifest->recheck_target_empty_(
                  manifest->recheck_context_, staged_fingerprint))) ||
            manifest->read_accounting_serial_(manifest->recheck_context_) !=
                staged->accounting_serial_after_prepare() ||
            manifest->read_policy_epoch_(manifest->recheck_context_) !=
                manifest->target().policy_epoch ||
            !operation_quiescent(operation, server_hooks)) {
            return fail(vbr_adopt_status::target_drift);
        }
        for (const auto & plan : manifest->children()) {
            auto & work = children[plan.child_id];
            work.plans.push_back(&plan);
        }
        if (children.empty()) {
            return fail(vbr_adopt_status::target_drift);
        }
        for (auto & entry : children) {
            const auto expected = std::find_if(
                manifest->target().children.begin(),
                manifest->target().children.end(),
                [&](const vbr_child_empty_fingerprint & value) {
                    return value.child_id == entry.first;
                });
            const auto live = std::find_if(
                tree.begin(), tree.end(),
                [&](const llama_memory_tree_child & value) {
                    return value.child_id == entry.first;
                });
            if (expected == manifest->target().children.end() ||
                live == tree.end() || !live->attention ||
                expected->memory_cookie != live->attention) {
                return fail(vbr_adopt_status::target_drift);
            }
            entry.second.cache = live->attention;
            entry.second.session = std::make_unique<vbr_kv_import_session>(
                *live->attention, entry.first, destination, operation.id(),
                occupied_replacement,
                test_target(server_hooks));
            if (!entry.second.session->recheck(*expected)) {
                return fail(vbr_adopt_status::target_drift);
            }
        }
        const size_t live_attention = std::count_if(
            tree.begin(), tree.end(),
            [](const llama_memory_tree_child & child) {
                return child.attention != nullptr;
            });
        if (live_attention != children.size()) {
            return fail(vbr_adopt_status::target_drift);
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::target_drift);
        }

        out.phase = vbr_adopt_phase::journal_arm;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::internal_error);
        }
        for (auto & entry : children) {
            if (!entry.second.session->arm()) {
                return fail(vbr_adopt_status::target_drift);
            }
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::internal_error);
        }

        out.phase = vbr_adopt_phase::private_backing;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::private_backing_failed);
        }
        out.accounting_status =
            staged->adoption_materialize_claims().status;
        if (out.accounting_status !=
                llama_cache_transaction_status::committed) {
            return fail(transaction_status(out.accounting_status));
        }
        claims_committed = true;
        post_commit_serial = accounting.snapshot().serial;
        if (post_commit_serial == 0) {
            return fail(vbr_adopt_status::accounting_unavailable);
        }
        for (auto & entry : children) {
            if (!entry.second.session->prepare_backing(
                    entry.second.plans,
                    occupied_replacement
                        ? &manifest->relocation_runs() : nullptr)) {
                return fail(vbr_adopt_status::private_backing_failed);
            }
        }
        if (occupied_replacement) {
            for (auto & entry : children) {
                const auto * install = tracker_plan(*manifest, entry.first);
                const auto * source = source_controller(*manifest, entry.first);
                if (!install || !source ||
                    !entry.second.session->build_live_image(
                        entry.second.plans, *install, *source,
                        manifest->occupied_replacement())) {
                    return fail(vbr_adopt_status::tracker_failed);
                }
            }
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::private_backing_failed);
        }

        out.phase = vbr_adopt_phase::unit_h2d;
        if (out.decision == vbr_import_decision::downward_rebase) {
            out.downward_subphase =
                vbr_downward_adopt_subphase::source_h2d;
        }
        if (fault_before(server_hooks, out.phase) ||
            (!test_target(server_hooks) && !staged->adoption_ring())) {
            return fail(vbr_adopt_status::transfer_failed);
        }
        const bool prefix_projection = manifest->is_prefix_projection();
        const bool packed_transfer =
            prefix_projection || occupied_replacement;
        if (prefix_projection && !manifest->projection_transfer_ready()) {
            return fail(vbr_adopt_status::source_changed);
        }
        if (packed_transfer && !test_target(server_hooks)) {
            packed_operation =
                staged->adoption_ring()->try_begin_operation();
            if (!packed_operation) {
                return fail(vbr_adopt_status::operation_unavailable);
            }
        }
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> transferred_units;
        uint64_t completion = 0;
        if (out.decision == vbr_import_decision::downward_rebase ||
            out.decision == vbr_import_decision::upward_reconstruct) {
            // A recycle transform may initialize/clear incumbent backing.
            // Arm byte-exact recovery before the first such mutation, not
            // merely before the first incoming H2D descriptor.
            destructive_started = destructive_started || recycle;
            const auto status = transform_source_h2d(children, *manifest);
            if (status != vbr_adopt_status::adopted) {
                return fail(status);
            }
        }
        for (const auto & read : staged->reads()) {
            if (read.kind != vbr_staged_read_kind::unit_payload) {
                continue;
            }
            const auto child = children.find(read.child_id);
            if (child == children.end() ||
                (server_hooks.test != nullptr &&
                 server_hooks.test->fault.fail_child == read.child_id &&
                 server_hooks.test->fault.fail_unit == read.logical_unit_id &&
                 server_hooks.test->fault.fail_shard == read.shard_index)) {
                return fail(vbr_adopt_status::transfer_failed);
            }
            vbr_h2d_stats stats;
            auto * ring = staged->adoption_ring();
            destructive_started = destructive_started || recycle;
            if (!child->second.session->transfer(
                    read, ring,
                    packed_transfer && !test_target(server_hooks)
                        ? &packed_operation : nullptr,
                    server_hooks.test != nullptr &&
                            completion == server_hooks.test->fault.fail_h2d_completion
                        ? 0 : UINT64_MAX, stats)) {
                return fail(vbr_adopt_status::transfer_failed);
            }
            ++completion;
            out.h2d_bytes += stats.bytes;
            out.h2d_chunks += stats.chunks;
            ++transferred_units[std::make_pair(
                read.child_id, read.logical_unit_id)];
        }
        // Tapped upward reconstruction consumes the authenticated clean
        // stash as an input to kv_transcode.  Populate the pre-reserved slab
        // after all compact source uploads and before enqueuing any transform;
        // the ordinary exact/downward phase order remains unchanged.
        if (out.decision == vbr_import_decision::upward_reconstruct) {
            for (const auto & read : staged->reads()) {
                if (read.kind != vbr_staged_read_kind::clean_stash) {
                    continue;
                }
                const auto child = children.find(read.child_id);
                vbr_h2d_stats stats;
                if (child == children.end() ||
                    !child->second.session->transfer(
                        read, staged->adoption_ring(), nullptr,
                        UINT64_MAX, stats)) {
                    return fail(vbr_adopt_status::transfer_failed);
                }
                out.h2d_bytes += stats.bytes;
                out.h2d_chunks += stats.chunks;
            }
        }
        // Projection owns the shared transport only across its selected H2D
        // bytes; all later validation and publication work is CPU-local.
        if (!recycle) {
            packed_operation = {};
        }
        if (out.decision == vbr_import_decision::downward_rebase ||
            out.decision == vbr_import_decision::upward_reconstruct) {
            const auto status = transform_all(
                children, *manifest, *staged, server_hooks,
                transform_stash_valid, out);
            if (status != vbr_adopt_status::adopted) {
                return fail(status);
            }
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::transfer_failed);
        }

        out.phase = vbr_adopt_phase::unit_complete;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::transfer_failed);
        }
        for (const auto & plan : manifest->children()) {
            const uint64_t expected = packed_transfer
                ? uint64_t(plan.shards.size())*
                    (occupied_replacement
                        ? manifest->relocation_runs().size() : 1)
                : uint64_t(plan.shards.size())*plan.authorized_runs.size();
            if (expected == 0 || expected > UINT32_MAX ||
                transferred_units[{ plan.child_id, plan.logical_unit_id }] !=
                    expected) {
                return fail(vbr_adopt_status::transfer_failed);
            }
            const auto unit = std::make_pair(
                plan.child_id, plan.logical_unit_id);
            const auto child = children.find(unit.first);
            if (child == children.end() ||
                !child->second.session->mark_complete(unit.second)) {
                return fail(vbr_adopt_status::transfer_failed);
            }
            ++out.units;
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::transfer_failed);
        }

        out.phase = vbr_adopt_phase::stash_and_companions;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::companion_failed);
        }
        for (const auto & read : staged->reads()) {
            if (read.kind != vbr_staged_read_kind::clean_stash) {
                continue;
            }
            if (out.decision == vbr_import_decision::upward_reconstruct) {
                continue;
            }
            const auto child = children.find(read.child_id);
            vbr_h2d_stats stats;
            if (child == children.end() ||
                !child->second.session->transfer(
                    read, staged->adoption_ring(), nullptr,
                    UINT64_MAX, stats)) {
                return fail(vbr_adopt_status::companion_failed);
            }
            out.h2d_bytes += stats.bytes;
            out.h2d_chunks += stats.chunks;
        }
        // The mirrored QSA companion needs the exact off-side attention cell
        // image. Build every non-replacement image before companion staging;
        // publication remains in the original no-fail composite terminal.
        if (!occupied_replacement) {
            for (auto & entry : children) {
                const auto * install = tracker_plan(*manifest, entry.first);
                const auto * source = source_controller(*manifest, entry.first);
                if (!install || !source ||
                    !entry.second.session->build_live_image(
                        entry.second.plans, *install, *source)) {
                    return fail(vbr_adopt_status::tracker_failed);
                }
            }
        }
        for (auto & plan : manifest->companions_) {
            const auto provider = std::find_if(
                server_hooks.companions.begin(), server_hooks.companions.end(),
                [&](const vbr_companion_adoption_provider & value) {
                    return value.kind == plan.descriptor.kind &&
                           value.target_cookie == plan.target_cookie;
                });
            const bool replacement = plan.recovery_parsed != nullptr;
            const bool layout_aware = provider != server_hooks.companions.end() &&
                provider->attention_child_id != UINT32_MAX;
            vbr_companion_attention_layout companion_layout;
            if (layout_aware) {
                const auto child = children.find(provider->attention_child_id);
                if (child == children.end() ||
                    !child->second.session->companion_layout(companion_layout)) {
                    return fail(vbr_adopt_status::companion_failed);
                }
            }
            if (provider == server_hooks.companions.end() ||
                (replacement
                    ? (layout_aware
                        ? provider->prepare_replacement_with_layout == nullptr
                        : provider->prepare_replacement == nullptr)
                    : (layout_aware
                        ? provider->prepare_with_layout == nullptr
                        : provider->prepare == nullptr)) ||
                provider->recheck == nullptr ||
                provider->publish_swap == nullptr ||
                provider->target_empty == nullptr ||
                !plan.parsed ||
                (!replacement &&
                 !provider->target_empty(provider->context))) {
                return fail(vbr_adopt_status::companion_failed);
            }
            std::unique_ptr<vbr_prepared_companion_image> image;
            const size_t prepared_before = companions.size();
            const bool prepared = replacement
                ? layout_aware
                    ? provider->prepare_replacement_with_layout(
                        provider->context, std::move(plan.parsed),
                        std::move(plan.recovery_parsed), destination,
                        companion_layout, image)
                    : provider->prepare_replacement(
                        provider->context, std::move(plan.parsed),
                        std::move(plan.recovery_parsed), destination, image)
                : layout_aware
                    ? provider->prepare_with_layout(
                        provider->context, std::move(plan.parsed), destination,
                        companion_layout, image)
                    : provider->prepare(
                        provider->context, std::move(plan.parsed),
                        destination, image);
            // Replacement providers publish their recovery owner before the
            // first mutation. Retain it even when preparation fails so the
            // common reverse-order rollback can restore the incumbent.
            if (image) {
                companions.push_back({ &*provider, std::move(image) });
            }
            if (!prepared || companions.size() != prepared_before+1 ||
                companions.back().first != &*provider) {
                return fail(vbr_adopt_status::companion_failed);
            }
            ++out.companions;
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::companion_failed);
        }

        out.phase = vbr_adopt_phase::live_image_prepare;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::tracker_failed);
        }
        if (staged->adoption_committed_ops().empty()) {
            return fail(vbr_adopt_status::accounting_unavailable);
        }
        receipt_group = std::make_shared<vbr_import_receipt_group>(
            accounting, staged->adoption_committed_ops());
        if (!receipt_group->valid()) {
            return fail(vbr_adopt_status::accounting_unavailable);
        }
        for (auto & entry : children) {
            if (!entry.second.session->prepare_receipts(receipt_group)) {
                return fail(vbr_adopt_status::accounting_unavailable);
            }
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::tracker_failed);
        }

        out.phase = vbr_adopt_phase::complete_tree_barrier;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::barrier_failed);
        }
        auto barrier_fingerprint = manifest->target();
        barrier_fingerprint.accounting_serial = post_commit_serial;
        std::array<uint8_t, 32> transform_tree_digest = {};
        const bool transform_projection_stable =
            (out.decision != vbr_import_decision::downward_rebase &&
             out.decision != vbr_import_decision::upward_reconstruct) ||
            (manifest->read_transform_tree_digest_ != nullptr &&
             manifest->read_transform_tree_digest_(
                manifest->recheck_context_, transform_tree_digest) &&
             std::all_of(
                manifest->children().begin(), manifest->children().end(),
                [&](const vbr_validated_child_plan & plan) {
                    return plan.transform_kind ==
                               vbr_import_transform_kind::none ||
                           plan.transcode_tree_digest ==
                               transform_tree_digest;
                }));
        const bool replacement_barrier_current = !occupied_replacement ||
            (test_target(server_hooks)
                ? test_target(server_hooks)->session_recheck_occupied(
                    0, *manifest->occupied_replacement(), true)
                : vbr_explicit_recheck_occupied_replacement_guard(
                    target, destination,
                    manifest->occupied_replacement()->accounting_serial(),
                    manifest->occupied_representation_context_,
                    manifest->occupied_representation_identity_,
                    operation.id(),
                    *manifest->occupied_replacement_) ==
                        vbr_occupied_replacement_guard_status::ready);
        if (accounting.snapshot().serial != post_commit_serial ||
            !replacement_barrier_current ||
            (!occupied_replacement &&
             !manifest->recheck_target_empty_(
                manifest->recheck_context_, barrier_fingerprint)) ||
            manifest->read_accounting_serial_(manifest->recheck_context_) !=
                post_commit_serial ||
            manifest->read_policy_epoch_(manifest->recheck_context_) !=
                manifest->target().policy_epoch ||
            !transform_projection_stable ||
            !operation_quiescent(operation, server_hooks) ||
            (manifest->is_prefix_projection()
                ? !manifest->projection_transfer_ready()
                : occupied_replacement
                    ? !manifest->occupied_replacement()->ready()
                    : manifest->source_package().validate() !=
                        vbr_artifact_status::ok)) {
            return fail(vbr_adopt_status::barrier_failed);
        }
        std::vector<llama_memory_tree_child> barrier_tree;
        if (!collect_adopt_tree(target, server_hooks, barrier_tree)) {
            return fail(vbr_adopt_status::target_drift);
        }
        std::vector<vbr_adopt_expected_attention> expected_attention;
        expected_attention.reserve(children.size());
        for (const auto & entry : children) {
            expected_attention.push_back({ entry.first, entry.second.cache });
        }
        std::vector<vbr_companion_adoption_provider> prepared_providers;
        prepared_providers.reserve(companions.size());
        for (const auto & companion : companions) {
            if (companion.first && companion.second) {
                prepared_providers.push_back(*companion.first);
            }
        }
        const auto tree_status = vbr_adopt_check_complete_tree(
            expected_attention, barrier_tree, prepared_providers,
            occupied_replacement);
        if (tree_status != vbr_adopt_status::adopted) {
            return fail(tree_status);
        }
        if (!std::all_of(
                companions.begin(), companions.end(),
                [](const auto & value) {
                    return value.first && value.second &&
                        value.first->recheck && value.first->recheck(
                            value.first->context, *value.second);
                })) {
            return fail(vbr_adopt_status::barrier_failed);
        }
        for (const auto & entry : children) {
            const auto expected = std::find_if(
                barrier_fingerprint.children.begin(),
                barrier_fingerprint.children.end(),
                [&](const vbr_child_empty_fingerprint & value) {
                    return value.child_id == entry.first;
                });
            const auto live = std::find_if(
                barrier_tree.begin(), barrier_tree.end(),
                [&](const llama_memory_tree_child & value) {
                    return value.child_id == entry.first;
                });
            if (expected == barrier_fingerprint.children.end() ||
                live == barrier_tree.end() ||
                live->attention != entry.second.cache) {
                return fail(vbr_adopt_status::target_drift);
            }
            if (!entry.second.session->recheck(*expected, true) ||
                !entry.second.session->mapped_prefixes_complete() ||
                !entry.second.session->barrier(post_commit_serial, *manifest)) {
                return fail(vbr_adopt_status::barrier_failed);
            }
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::barrier_failed);
        }

        out.phase = vbr_adopt_phase::tracker_prepare;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::tracker_failed);
        }
        if (fault_after(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::tracker_failed);
        }

        out.phase = vbr_adopt_phase::composite_publish;
        if (fault_before(server_hooks, out.phase)) {
            return fail(vbr_adopt_status::barrier_failed);
        }
        // This is the final fault-injection checkpoint. All substantive
        // installability revalidation is the complete-tree barrier.

        // BEGIN VBR_IMPORT_NOFAIL_PUBLISH: metadata-only; CI bans backend,
        // allocation, logging and legacy state-reader calls in this region.
        for (auto & entry : children) {
            entry.second.session->publish_metadata();
        }
        for (auto & companion : companions) {
            companion.first->publish_swap(
                companion.first->context, *companion.second);
        }
        if (server_hooks.publish) {
            server_hooks.publish(server_hooks.context);
        }
        for (auto & entry : children) {
            entry.second.session->finish_publish();
        }
        // END VBR_IMPORT_NOFAIL_PUBLISH
        published = true;
        destructive_started = false;
        packed_operation = {};

        out.phase = vbr_adopt_phase::close;
        // Retain every committed reference in the imported target until its
        // reset/retire. This conservative accounting handoff keeps staging
        // released before a live producer can account the installed bytes.
        for (auto & entry : children) {
            entry.second.session->publish_receipts();
        }
        receipt_group->activate();
        // Close is the committed no-fail terminal. Do not report adoption
        // unless the exact registry/recovery predicate enforced by the target
        // tracker's destructor has been proved here.
        GGML_ASSERT(operation.finish(true, false));
        for (const auto & entry : children) {
            GGML_ASSERT(entry.second.session->tracker_teardown_state() ==
                        vbr_generation_teardown_state::clean);
        }
        out.children = uint32_t(children.size());
        out.status = vbr_adopt_status::adopted;
        return out;
    } catch (...) {
        return fail(vbr_adopt_status::internal_error);
    }
}
