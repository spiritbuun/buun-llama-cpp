#pragma once

#include "llama-vbr-artifact-stage.h"

#include <cstdint>
#include <memory>
#include <vector>

class llama_memory_i;
class llama_memory_recurrent;
class llama_kv_cache;
struct llama_memory_tree_child;

enum class vbr_adopt_phase : uint8_t {
    consume_capabilities = 0,
    operation_open,
    target_recheck,
    journal_arm,
    private_backing,
    unit_h2d,
    unit_complete,
    stash_and_companions,
    live_image_prepare,
    complete_tree_barrier,
    tracker_prepare,
    composite_publish,
    close,
    rollback,
    _count,
};
static_assert(uint8_t(vbr_adopt_phase::_count) == 14);

enum class vbr_adopt_status : uint8_t {
    adopted = 0,
    invalid_capability_pair,
    unsupported_decision,
    // Retired: downward imports are now executed; retained for status ABI stability.
    downward_deferred,
    downward_recipe_invalid,
    downward_transform_failed,
    downward_stash_unavailable,
    upward_recipe_invalid,
    upward_transform_failed,
    target_drift,
    operation_unavailable,
    recovery_unavailable,
    accounting_unavailable,
    admission_refused,
    private_backing_failed,
    transfer_failed,
    event_failed,
    source_changed,
    required_companion_unavailable,
    companion_failed,
    tracker_failed,
    barrier_failed,
    rollback_failed,
    quarantined,
    internal_error,
    _count,
};
static_assert(uint8_t(vbr_adopt_status::_count) == 25);

enum class vbr_adopt_recovery_outcome : uint8_t {
    not_needed = 0,
    replayed,
    quarantined,
    _count,
};

const char * vbr_adopt_recovery_outcome_name(
    vbr_adopt_recovery_outcome outcome) noexcept;

enum class vbr_downward_adopt_subphase : uint8_t {
    none = 0,
    source_h2d,
    edge_stash_capture,
    edge_transcode,
    edge_completion,
    _count,
};

const char * vbr_adopt_phase_name(vbr_adopt_phase phase) noexcept;
const char * vbr_adopt_status_name(vbr_adopt_status status) noexcept;
const char * vbr_downward_adopt_subphase_name(
    vbr_downward_adopt_subphase subphase) noexcept;

class vbr_prepared_companion_image {
  public:
    virtual ~vbr_prepared_companion_image();
    vbr_prepared_companion_image(const vbr_prepared_companion_image &) = delete;
    vbr_prepared_companion_image & operator=(
        const vbr_prepared_companion_image &) = delete;

  protected:
    vbr_prepared_companion_image() = default;
};

// Logical-to-physical image prepared for one attention child but not yet
// published. Layout-aware companions use it to stage mirrored state into the
// exact cells that will become visible at the composite commit.
struct vbr_companion_cell_placement {
    uint32_t stream = 0;
    uint32_t physical_cell = 0;
    llama_pos logical_position = -1;
    int32_t ext_x = 0;
    int32_t ext_y = 0;
    llama_token token = LLAMA_TOKEN_NULL;
};

struct vbr_companion_attention_layout {
    uint32_t child_id = UINT32_MAX;
    std::vector<vbr_companion_cell_placement> cells;
};

// Provider callbacks are closed over one companion target. `prepare` runs
// after unit H2D and may either build an off-side image or install reversible
// state in an otherwise-empty target. `recheck` is the allocation-free late
// barrier for that prepared image. `publish_swap` is the no-fail composite
// terminal. `rollback` must restore the pre-prepare target and reports whether
// recovery was complete.
struct vbr_companion_adoption_provider {
    vbr_artifact_companion_kind kind =
        vbr_artifact_companion_kind::_count;
    const void * target_cookie = nullptr;
    const void * context = nullptr;
    bool (*prepare)(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> parsed,
        llama_seq_id destination,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept = nullptr;
    // Layout-aware form for a companion that mirrors one attention child.
    // Exactly one of prepare/prepare_with_layout is used.
    bool (*prepare_with_layout)(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> parsed,
        llama_seq_id destination,
        const vbr_companion_attention_layout & layout,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept = nullptr;
    // Occupied replacement authenticates the live target against `recovery`
    // before installing `incoming`. The callback must publish `output`
    // before its first destructive mutation so the common rollback loop can
    // replay recovery even when preparation itself fails.
    bool (*prepare_replacement)(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> incoming,
        std::unique_ptr<vbr_parsed_companion_image> recovery,
        llama_seq_id destination,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept = nullptr;
    bool (*prepare_replacement_with_layout)(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> incoming,
        std::unique_ptr<vbr_parsed_companion_image> recovery,
        llama_seq_id destination,
        const vbr_companion_attention_layout & layout,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept = nullptr;
    uint32_t attention_child_id = UINT32_MAX;
    // Late complete-tree drift gate. It must be allocation-free and describe the
    // same target object as target_cookie/prepare/publish_swap.
    bool (*target_empty)(const void * context) noexcept = nullptr;
    bool (*recheck)(
        const void * context,
        const vbr_prepared_companion_image & image) noexcept = nullptr;
    void (*publish_swap)(
        const void * context,
        vbr_prepared_companion_image & image) noexcept = nullptr;
    bool (*rollback)(
        const void * context,
        vbr_prepared_companion_image & image) noexcept = nullptr;
};

struct vbr_adopt_fault {
    // Fails immediately before the named phase. `_count` disables the seam.
    vbr_adopt_phase fail_before = vbr_adopt_phase::_count;
    // Composite publication and close deliberately have no after
    // seam: once publication begins the transaction is no-fail.
    vbr_adopt_phase fail_after = vbr_adopt_phase::_count;
    uint32_t fail_child = UINT32_MAX;
    uint32_t fail_unit = UINT32_MAX;
    uint32_t fail_shard = UINT32_MAX;
    uint64_t fail_h2d_completion = UINT64_MAX;
    uint32_t fail_downward_edge = UINT32_MAX;
    bool fail_downward_stash = false;
    bool fail_downward_completion = false;
    bool fail_rollback = false;
};

// TEST-ONLY target/session door for the model-free atomicity matrix. Production
// callers must leave vbr_composite_publish_hooks::test null. The real
// vbr_adopt_empty_manifest phase driver, barrier, accounting transaction and
// rollback loop remain authoritative; this interface substitutes only the
// concrete KV/backend operations that otherwise require a model and GPU.
class vbr_adopt_test_seam {
  public:
    virtual ~vbr_adopt_test_seam() = default;

    // Called at the real driver's existing phase boundaries. Returning true
    // injects the test failure. Production never has a seam object.
    virtual bool phase_boundary(
        vbr_adopt_phase phase, bool after) noexcept = 0;

    virtual bool collect_tree(
        llama_memory_i & target,
        std::vector<llama_memory_tree_child> & output) noexcept = 0;
    virtual vbr_adopt_status operation_open(
        const vbr_validated_manifest & manifest,
        llama_seq_id destination,
        const std::vector<llama_memory_tree_child> & tree,
        std::vector<vbr_controller_instance_id> & instances,
        vbr_operation_id & operation) noexcept = 0;
    virtual void operation_finish(
        vbr_operation_id operation, bool committed,
        bool quarantine) noexcept = 0;
    virtual bool operation_quiescent(
        const std::vector<vbr_controller_instance_id> & instances,
        vbr_operation_id operation) const noexcept = 0;

    virtual bool session_recheck(
        uint32_t child_id, const vbr_child_empty_fingerprint & expected,
        bool journal_armed) const noexcept = 0;
    virtual bool session_recheck_occupied(
        uint32_t, const vbr_occupied_replacement_guard &,
        bool) const noexcept {
        return false;
    }
    virtual bool session_arm(
        uint32_t child_id, vbr_operation_id operation) noexcept = 0;
    virtual bool session_prepare_backing(
        uint32_t child_id,
        const std::vector<const vbr_validated_child_plan *> & plans) noexcept = 0;
    virtual bool session_prepare_relocated_backing(
        uint32_t,
        const std::vector<const vbr_validated_child_plan *> &,
        const std::vector<vbr_occupied_replacement_relocation_run> &) noexcept {
        return false;
    }
    virtual bool session_transfer(
        uint32_t child_id, const vbr_staged_read_descriptor & read,
        uint64_t fail_completion, vbr_h2d_stats & stats) noexcept = 0;
    virtual bool session_synchronize_recovery(uint32_t) noexcept {
        return true;
    }
    virtual bool session_mark_complete(
        uint32_t child_id, uint32_t logical_unit) noexcept = 0;
    virtual bool session_initialize_downward_backing(
        uint32_t, const vbr_validated_child_plan &) noexcept {
        return true;
    }
    virtual bool session_initialize_upward_backing(
        uint32_t, const vbr_validated_child_plan &) noexcept {
        return true;
    }
    virtual vbr_downward_transform_status session_transform_downward(
        uint32_t, const vbr_validated_child_plan &, bool,
        uint32_t, bool, uint32_t & stash_valid,
        uint32_t & edge_reached) noexcept {
        stash_valid = 0;
        edge_reached = UINT32_MAX;
        return vbr_downward_transform_status::transformed;
    }
    virtual bool session_synchronize_downward(
        uint32_t, const std::vector<const vbr_validated_child_plan *> &) noexcept {
        return true;
    }
    virtual bool session_transform_upward(
        uint32_t, const vbr_validated_child_plan &,
        uint32_t) noexcept {
        return true;
    }
    virtual bool session_synchronize_upward(
        uint32_t, const std::vector<const vbr_validated_child_plan *> &) noexcept {
        return true;
    }
    virtual bool session_trim_downward(
        uint32_t, const vbr_validated_child_plan &, uint32_t) noexcept {
        return true;
    }
    virtual bool session_build_live_image(
        uint32_t child_id,
        const std::vector<const vbr_validated_child_plan *> & plans,
        const vbr_tracker_install_child & tracker_plan,
        const vbr_checkpoint_generation_controller & source) noexcept = 0;
    virtual bool session_build_relocated_live_image(
        uint32_t,
        const std::vector<const vbr_validated_child_plan *> &,
        const vbr_tracker_install_child &,
        const vbr_checkpoint_generation_controller &,
        const vbr_occupied_replacement_guard &) noexcept {
        return false;
    }
    // Test-only ownership mirror for the production shared receipt group. The
    // opaque shared owner lets the model-free real phase driver retain claims
    // across a successful adopt and release them at its simulated erase.
    virtual bool session_prepare_receipts(
        uint32_t child_id, std::shared_ptr<void> receipt_owner) noexcept = 0;
    virtual bool session_mapped_prefixes_complete(
        uint32_t child_id) const noexcept = 0;
    virtual bool session_barrier(
        uint32_t child_id, uint64_t ledger_serial,
        const vbr_validated_manifest & manifest) const noexcept = 0;
    virtual void session_publish_metadata(uint32_t child_id) noexcept = 0;
    virtual void session_publish_receipts(uint32_t child_id) noexcept = 0;
    virtual void session_finish_publish(uint32_t child_id) noexcept = 0;
    virtual bool session_rollback(
        uint32_t child_id, bool inject_failure) noexcept = 0;
};

// No-fail publication capacity check shared by the production recheck and the model-free
// adoption test. A publication advances representation/checkpoint once and tier identity once
// per unit whose type changes.
bool vbr_artifact_epoch_capacity(
    uint64_t tier_epoch,
    uint64_t representation_epoch,
    uint64_t checkpoint_epoch,
    uint64_t tier_changes) noexcept;

struct vbr_adopt_test_control {
    vbr_adopt_fault fault;
    vbr_adopt_test_seam * target = nullptr;
};

struct vbr_composite_publish_hooks {
    void * context = nullptr;
    // The context owner validates an opaque scheduler capability, not a
    // caller-supplied boolean. The token is never interpreted by the library.
    const void * owner_token = nullptr;
    bool (*validate_owner_token)(
        const void * context,
        const void * token,
        const llama_memory_i * target) noexcept = nullptr;
    // Optional complete-tree companions, in canonical target-tree order.
    std::vector<vbr_companion_adoption_provider> companions;
    // Optional already-prepared server metadata holder publication. It is
    // invoked only in the no-fail composite-publication region.
    void (*publish)(void * context) noexcept = nullptr;
    // TEST-ONLY and null by default. This is the single production-boundary
    // null test; fault policy and fake target both live behind it. CI rejects
    // any production caller that arms the door.
    const vbr_adopt_test_control * test = nullptr;
};

// Read-only complete-tree shape barrier, exposed only for the CPU fault matrix.
// Pointer members are identity cookies; this helper never dereferences them.
struct vbr_adopt_expected_attention {
    uint32_t child_id = UINT32_MAX;
    const llama_kv_cache * cache = nullptr;
};
vbr_adopt_status vbr_adopt_check_complete_tree(
        const std::vector<vbr_adopt_expected_attention> & expected,
        const std::vector<llama_memory_tree_child> & live,
        const std::vector<vbr_companion_adoption_provider> & companions,
        bool occupied_replacement = false) noexcept;

// Rollback quarantine is meaningful for VMM backends whose unmap door can
// return false. CUDA's current unmap door is fail-stop (CU_CHECK aborts on a
// driver error and otherwise returns true), so driver unmap faults never
// reach the typed quarantine terminal on that backend.

// Canonical recurrent-v1 companion doors. Validation parses the legacy-shaped
// bytes into an allocation-only CPU blueprint; adoption journals one target
// row and publishes it only through the no-fail metadata swap.
// The existing state_read path is deliberately not called or modified.
bool vbr_parse_recurrent_companion(
    const void * context,
    const vbr_artifact_companion_payload & descriptor,
    const artifact_segment_chain & source,
    const vbr_target_companion_snapshot & target,
    std::unique_ptr<vbr_parsed_companion_image> & output) noexcept;
vbr_companion_adoption_provider vbr_recurrent_companion_adoption_provider(
    llama_memory_recurrent & target) noexcept;

struct vbr_adopt_result {
    vbr_adopt_status status = vbr_adopt_status::internal_error;
    vbr_adopt_phase phase = vbr_adopt_phase::consume_capabilities;
    vbr_import_decision decision = vbr_import_decision::reject;
    vbr_artifact_consistency_kind consistency =
        vbr_artifact_consistency_kind::live_rebased;
    uint32_t children = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_chunks = 0;
    vbr_adopt_recovery_outcome recovery =
        vbr_adopt_recovery_outcome::not_needed;
    uint64_t recovery_h2d_bytes = 0;
    uint64_t recovery_h2d_chunks = 0;
    uint64_t rollback_count = 0;
    llama_cache_transaction_status accounting_status =
        llama_cache_transaction_status::internal_fault;
    vbr_downward_adopt_subphase downward_subphase =
        vbr_downward_adopt_subphase::none;
    uint32_t downward_edge = UINT32_MAX;
};

vbr_adopt_result vbr_adopt_empty_manifest(
    llama_memory_i & target,
    llama_seq_id destination,
    vbr_validated_manifest && manifest,
    vbr_staged_payloads && staged,
    llama_cache_acct_ledger & accounting,
    const vbr_composite_publish_hooks & server_hooks) noexcept;
