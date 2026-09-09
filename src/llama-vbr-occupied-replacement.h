#pragma once

#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-generation.h"
#include "llama-vbr-operation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class llama_kv_cache;
class llama_memory_i;
class vbr_live_capture_adapter;
struct vbr_explicit_representation_identity;
class vbr_import_schedule_quote;
struct vbr_target_validation_snapshot;
struct vbr_target_companion_snapshot;

using vbr_explicit_representation_identity_fn = bool (*)(
    const void * context,
    int32_t current_type,
    bool value_side,
    int32_t meansub_model_id,
    vbr_explicit_representation_identity & output) noexcept;

// Occupied-target preparation remains a separate capability from the
// construction-empty importer. These limits bound the one shared provisional
// cell map minted while the incumbent is still live.
static constexpr uint32_t VBR_OCCUPIED_REPLACEMENT_MAX_CELLS = 1u << 20;
static constexpr uint32_t VBR_OCCUPIED_REPLACEMENT_MAX_RUNS  = 4096;

enum class vbr_occupied_replacement_guard_status : uint8_t {
    ready = 0,
    invalid_argument,
    unsupported_tree,
    unsupported_layout,
    companion_unavailable,
    frontier_mismatch,
    representation_mismatch,
    generation_mismatch,
    ownership_mismatch,
    cell_limit_exceeded,
    run_limit_exceeded,
    capacity_unavailable,
    currency_changed,
    internal_error,
    _count,
};

enum class vbr_occupied_replacement_strategy : uint8_t {
    provisional_free_cells = 0,
    recycle_incumbent_cells,
    _count,
};

const char * vbr_occupied_replacement_guard_status_name(
    vbr_occupied_replacement_guard_status status) noexcept;

struct vbr_occupied_replacement_cell {
    uint32_t stream_index = UINT32_MAX;
    uint32_t physical_cell = UINT32_MAX;
    llama_pos logical_position = -1;
    llama_pos ext_x = 0;
    llama_pos ext_y = 0;
    llama_seq_id owner_sequence = -1;
    uint32_t reference_count = 0;
    bool owns_destination = false;
    llama_token token = LLAMA_TOKEN_NULL;
};

struct vbr_occupied_replacement_unit_currency {
    uint32_t child_id = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    vbr_unit_generation generation;
};

// CPU-testable view of the allocation-free live observation. Cells list every
// nonempty cell in strictly increasing (stream, physical) order.
struct vbr_occupied_replacement_observation {
    llama_seq_id destination = -1;
    uint64_t sequence_epoch = 0;
    uint64_t controller_generation = 0;
    uint64_t representation_epoch = 0;
    uint32_t cell_capacity = 0;
    const vbr_occupied_replacement_cell * cells = nullptr;
    size_t cell_count = 0;
    const vbr_occupied_replacement_unit_currency * units = nullptr;
    size_t unit_count = 0;
};

struct vbr_occupied_replacement_cell_mapping {
    uint32_t source_stream = UINT32_MAX;
    uint32_t source_physical_cell = UINT32_MAX;
    uint64_t source_packed_row = UINT64_MAX;
    uint32_t destination_physical_cell = UINT32_MAX;
    llama_pos logical_position = -1;
    llama_pos ext_x = 0;
    llama_pos ext_y = 0;
};

struct vbr_occupied_replacement_relocation_run {
    uint64_t first_source_packed_row = UINT64_MAX;
    uint32_t first_destination_physical_cell = UINT32_MAX;
    uint32_t cell_count = 0;
};

class vbr_occupied_replacement_guard {
public:
    struct map;

    vbr_occupied_replacement_guard() noexcept;
    ~vbr_occupied_replacement_guard();
    vbr_occupied_replacement_guard(
        const vbr_occupied_replacement_guard &) = delete;
    vbr_occupied_replacement_guard & operator=(
        const vbr_occupied_replacement_guard &) = delete;
    vbr_occupied_replacement_guard(
        vbr_occupied_replacement_guard &&) noexcept;
    vbr_occupied_replacement_guard & operator=(
        vbr_occupied_replacement_guard &&) noexcept;

    bool ready() const noexcept;
    llama_seq_id destination() const noexcept;
    uint64_t accounting_serial() const noexcept;
    llama_cache_acct_artifact_id incoming_artifact() const noexcept;
    llama_cache_acct_artifact_id recovery_artifact() const noexcept;
    vbr_occupied_replacement_strategy strategy() const noexcept;
    const std::vector<vbr_occupied_replacement_cell_mapping> &
        cell_mapping() const noexcept;
    const std::vector<vbr_occupied_replacement_relocation_run> &
        relocation_runs() const noexcept;
    const std::vector<vbr_occupied_replacement_relocation_run> &
        recovery_runs() const noexcept;
    // Live cells belonging exclusively to other logical sequences. Occupied
    // replacement republishes these unchanged alongside the new destination.
    const std::vector<vbr_occupied_replacement_cell> &
        preserved_cells() const noexcept;
    // Internal stage authority. The guard retains this immutable capability
    // through validation/adoption; callers must not resolve a second package.
    const vbr_artifact_package_view & recovery_package() const noexcept;
    uint64_t packed_rows_expanded() const noexcept;
    void reset() noexcept;

private:
    std::shared_ptr<const map> map_;
    vbr_artifact_package_view incoming_;
    vbr_artifact_package_view recovery_;
    std::array<uint8_t, 32> currency_digest_ = {};
    std::array<uint8_t, 32> direct_currency_digest_ = {};
    llama_memory_i * memory_ = nullptr;
    llama_kv_cache * cache_ = nullptr;
    llama_seq_id destination_ = -1;
    uint64_t accounting_serial_ = 0;

    friend class vbr_live_capture_adapter;
    friend vbr_occupied_replacement_guard_status
    vbr_prepare_occupied_replacement_guard(
        const vbr_target_validation_snapshot &,
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &,
        const vbr_artifact_package_view &,
        const vbr_occupied_replacement_observation &,
        vbr_occupied_replacement_guard &,
        const vbr_import_schedule_quote *,
        const vbr_import_schedule_quote *) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_prepare_occupied_replacement_guard(
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &,
        const vbr_artifact_package_view &,
        const vbr_occupied_replacement_observation &,
        vbr_occupied_replacement_guard &,
        const vbr_import_schedule_quote *,
        const vbr_import_schedule_quote *) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_prepare_occupied_prefix_replacement_guard(
        const vbr_target_validation_snapshot &,
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &, uint64_t,
        const std::vector<vbr_artifact_prefix_cell_run> &,
        const vbr_artifact_package_view &,
        const vbr_occupied_replacement_observation &,
        vbr_occupied_replacement_guard &,
        const vbr_import_schedule_quote &,
        const vbr_import_schedule_quote *) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_recheck_occupied_replacement_guard(
        vbr_occupied_replacement_guard &,
        const vbr_target_validation_snapshot &,
        const vbr_occupied_replacement_observation &) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_explicit_prepare_occupied_replacement_guard(
        llama_memory_i &, llama_seq_id,
        const vbr_artifact_package_view &,
        const vbr_artifact_package_view &,
        const std::vector<llama_vbr_artifact_domain_binding> &,
        uint64_t, const void *,
        vbr_explicit_representation_identity_fn,
        vbr_occupied_replacement_guard &,
        const vbr_import_schedule_quote *,
        const std::vector<vbr_target_companion_snapshot> *) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_explicit_prepare_occupied_prefix_replacement_guard(
        llama_memory_i &, llama_seq_id,
        const vbr_artifact_package_view &, uint64_t,
        const std::vector<vbr_artifact_prefix_cell_run> &,
        const vbr_artifact_package_view &,
        const std::vector<llama_vbr_artifact_domain_binding> &,
        uint64_t, const void *,
        vbr_explicit_representation_identity_fn,
        vbr_occupied_replacement_guard &,
        const vbr_import_schedule_quote &) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_explicit_recheck_occupied_replacement_guard(
        llama_memory_i &, llama_seq_id, uint64_t,
        const void *, vbr_explicit_representation_identity_fn,
        vbr_occupied_replacement_guard &) noexcept;
    friend vbr_occupied_replacement_guard_status
    vbr_explicit_recheck_occupied_replacement_guard(
        llama_memory_i &, llama_seq_id, uint64_t,
        const void *, vbr_explicit_representation_identity_fn,
        vbr_operation_id, vbr_occupied_replacement_guard &) noexcept;
};

// Pure canonical core used by the direct production factory and CPU mutants.
// Failure always resets output.
vbr_occupied_replacement_guard_status
vbr_prepare_occupied_replacement_guard(
    const vbr_target_validation_snapshot & live_target,
    const vbr_target_validation_snapshot & selected_target,
    const vbr_artifact_package_view & incoming,
    const vbr_artifact_package_view & recovery,
    const vbr_occupied_replacement_observation & observation,
    vbr_occupied_replacement_guard & output,
    const vbr_import_schedule_quote * authenticated_incoming = nullptr,
    const vbr_import_schedule_quote * authenticated_recovery = nullptr) noexcept;

vbr_occupied_replacement_guard_status
vbr_prepare_occupied_replacement_guard(
    const vbr_target_validation_snapshot & target,
    const vbr_artifact_package_view & incoming,
    const vbr_artifact_package_view & recovery,
    const vbr_occupied_replacement_observation & observation,
    vbr_occupied_replacement_guard & output,
    const vbr_import_schedule_quote * authenticated_incoming = nullptr,
    const vbr_import_schedule_quote * authenticated_recovery = nullptr) noexcept;

vbr_occupied_replacement_guard_status
vbr_prepare_occupied_prefix_replacement_guard(
    const vbr_target_validation_snapshot & live_target,
    const vbr_target_validation_snapshot & selected_target,
    const vbr_artifact_package_view & incoming_parent,
    uint64_t prefix_tokens,
    const std::vector<vbr_artifact_prefix_cell_run> & prefix_runs,
    const vbr_artifact_package_view & recovery,
    const vbr_occupied_replacement_observation & observation,
    vbr_occupied_replacement_guard & output,
    const vbr_import_schedule_quote & authenticated_incoming,
    const vbr_import_schedule_quote * authenticated_recovery = nullptr) noexcept;

// Allocation-free recheck against already-collected caller storage.
vbr_occupied_replacement_guard_status
vbr_recheck_occupied_replacement_guard(
    vbr_occupied_replacement_guard & guard,
    const vbr_target_validation_snapshot & target,
    const vbr_occupied_replacement_observation & observation) noexcept;
