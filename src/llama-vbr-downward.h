#pragma once

#include "llama-cache-authority.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-policy.h"
#include "llama-vbr-transaction.h"

#include "ggml-vbr.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t VBR_DOWNWARD_RECIPE_VERSION = 1;
// A package is bounded at 16,384 projected units. Each unit has at most the
// five adjacent downward edges represented by vbr_downward_recipe::edges.
constexpr size_t VBR_IMPORT_DESTINATION_MAX_STEPS = 16384u*5u;
// Adjacent-chain recipe family id, distinct from the serialization version
// above. Proofs carry both.
constexpr uint32_t VBR_DOWNWARD_RECIPE_ID = 1;

enum class vbr_downward_recipe_status : uint8_t {
    resolved = 0,
    equal_tier,
    upward_forbidden,
    unsupported_type,
    below_floor,
    nonmovable,
    invalid_argument,
    _count,
};

const char * vbr_downward_recipe_status_name(vbr_downward_recipe_status status) noexcept;

struct vbr_downward_edge {
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    vbr_repr_domain source_domain = vbr_repr_domain::full;
    vbr_repr_domain target_domain = vbr_repr_domain::full;
    bool capture_stash_before = false;

    bool operator==(const vbr_downward_edge & other) const noexcept {
        return source_type == other.source_type &&
               target_type == other.target_type &&
               source_domain == other.source_domain &&
               target_domain == other.target_domain &&
               capture_stash_before == other.capture_stash_before;
    }
};

struct vbr_downward_recipe {
    uint32_t version = VBR_DOWNWARD_RECIPE_VERSION;
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    std::array<vbr_downward_edge, 5> edges = {};
    size_t n_edges = 0;

    bool operator==(const vbr_downward_recipe & other) const noexcept {
        if (version != other.version || source_type != other.source_type ||
            target_type != other.target_type || n_edges != other.n_edges ||
            n_edges > edges.size()) {
            return false;
        }
        for (size_t i = 0; i < n_edges; ++i) {
            if (!(edges[i] == other.edges[i])) {
                return false;
            }
        }
        return true;
    }
};

// The stash-boundary rule is downward-owned; callers must not re-derive it.
inline bool vbr_downward_recipe_needs_stash(const vbr_downward_recipe & recipe) noexcept {
    for (size_t i = 0; i < recipe.n_edges && i < recipe.edges.size(); ++i) {
        if (recipe.edges[i].capture_stash_before) {
            return true;
        }
    }
    return false;
}

// Canonical tier-domain rule (F16/T8 = full, else tapped), proof-relevant in
// both the cache snapshot and the validator.
vbr_repr_domain vbr_downward_tier_domain(ggml_type type) noexcept;

vbr_downward_recipe_status vbr_downward_resolve_recipe(
        ggml_type source_type,
        ggml_type target_type,
        ggml_type floor_type,
        bool movable,
        vbr_downward_recipe & out) noexcept;

std::array<uint8_t, 32> vbr_downward_build_identity(
        const vbr_downward_recipe & recipe,
        int32_t meansub_model_id,
        const std::array<uint8_t, 32> & meansub_digest,
        const std::array<uint8_t, 32> & policy_digest,
        const std::array<uint8_t, 32> & tree_digest) noexcept;

enum class vbr_downward_policy_status : uint8_t {
    coherent = 0,
    incoherent,
    exhausted,
    invalid,
    overflow,
    _count,
};

const char * vbr_downward_policy_status_name(vbr_downward_policy_status status) noexcept;

// The canonical type-vector identity digest lives in llama-vbr-identity-digest.h
// (vbr_type_vector_digest) so capture does not depend on this import module.

struct vbr_downward_policy_child {
    llama_vbr_policy::child policy;
    std::vector<ggml_type> initial_types;
    std::vector<ggml_type> target_types;
    uint64_t initial_cursor = 0;
};

// Controller-owned input to empty-target destination negotiation.  The
// initial vector is the live tree schedule; policy contains only lawful
// remaining degradation steps in the controller's canonical price order.
// Import code may select a prefix, but it must never manufacture steps.
struct vbr_import_destination_child {
    llama_vbr_policy::child policy;
    std::vector<ggml_type> initial_types;
    uint64_t initial_cursor = 0;
    uint32_t watermark_cells = 0;
};

enum class vbr_import_destination_status : uint8_t {
    invalid = 0,
    feasible_current,
    feasible_degraded,
    exhausted,
    overflow,
    _count,
};

const char * vbr_import_destination_status_name(
    vbr_import_destination_status status) noexcept;

// Immutable result of the controller-owned empty-target schedule projection.
// The byte fields are observations, not reservations.  A later materializer
// must recheck this capability and reserve/apply the exact prefix atomically.
struct vbr_import_destination_projection {
    vbr_import_destination_status status =
        vbr_import_destination_status::invalid;
    std::vector<llama_vbr_policy::selection> prefix;
    std::vector<std::vector<ggml_type>> initial_types;
    std::vector<uint64_t> initial_cursors;
    std::vector<std::vector<ggml_type>> final_types;
    std::vector<uint64_t> final_cursors;
    std::vector<std::array<uint8_t, 32>> child_type_digests;
    std::array<uint8_t, 32> tree_digest = {};
    uint32_t pools = 0;
    uint64_t logical_bytes_needed = 0;
    uint64_t logical_bytes_available = 0;
    uint64_t physical_growth_needed = 0;
    uint64_t physical_growth_available = 0;
    int64_t max_deficit = 0;

    // Budgeted occupied import must reuse the destination's authenticated
    // incumbent rows instead of allocating provisional free cells.
    bool recycle_incumbent = false;

    bool feasible() const noexcept {
        return status == vbr_import_destination_status::feasible_current ||
               status == vbr_import_destination_status::feasible_degraded;
    }
};

struct vbr_import_destination_evidence {
    bool active = false;
    bool fits = false;
    uint32_t pools = 0;
    uint64_t logical_bytes_needed = 0;
    uint64_t logical_bytes_available = 0;
    uint64_t physical_growth_needed = 0;
    uint64_t physical_growth_available = 0;
    int64_t max_deficit = 0;
};

using vbr_import_destination_measure_fn = bool (*)(
    void * context,
    const std::vector<std::vector<ggml_type>> & types,
    const llama_vbr_policy::selection * selected,
    vbr_import_destination_evidence & evidence) noexcept;

// Select the shortest feasible prefix of the controller's canonical merged
// policy stream. The callback owns resource pricing; selection owns ordering.
vbr_import_destination_projection vbr_select_import_destination(
    const std::vector<vbr_import_destination_child> & children,
    void * context,
    vbr_import_destination_measure_fn measure) noexcept;

// Replays a bounded controller-minted prefix from the supplied live inputs
// and authenticates its final types, cursors, and digests. This performs no
// resource pricing and is shared by import binding and barrier rechecks.
bool vbr_import_destination_projection_coherent(
    const std::vector<vbr_import_destination_child> & children,
    const vbr_import_destination_projection & projection) noexcept;

struct vbr_downward_policy_projection {
    vbr_downward_policy_status status = vbr_downward_policy_status::invalid;
    std::vector<llama_vbr_policy::selection> prefix;
    std::vector<std::vector<ggml_type>> final_types;
    std::vector<uint64_t> final_cursors;
    std::vector<std::array<uint8_t, 32>> child_type_digests;
    std::array<uint8_t, 32> tree_digest = {};
};

// One simulator for both the ordinary single-child cursor and the merged tree
// policy stream. It accepts only the first coherent prefix of that stream.
vbr_downward_policy_projection vbr_downward_project_policy_prefix(
        const std::vector<vbr_downward_policy_child> & children) noexcept;

struct vbr_downward_workspace_endpoint {
    const void * owner = nullptr;
    const ggml_vbr_backend_iface * iface = nullptr;
    const ggml_vbr_cross_domain_iface_v1 * cross_iface = nullptr;
    ggml_backend_t backend = nullptr;
    int device = -1;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution attribution;
    std::vector<llama_vbr_transaction::workspace_request> requests;
};

// The KV cache owns vbr_pool, so its adapter is the only permitted way to
// expose vbr_stash_memory/vbr_stash_reserve without duplicating slab math.
struct vbr_downward_stash_endpoint {
    const void * owner = nullptr;
    // One pool projection covers the complete fixed slab. Every unit whose
    // requested stash shares that pool falls back independently if its one
    // grow-only reserve fails.
    std::vector<uint64_t> unit_ids;
    // Downward degradation may publish without this cache on allocation
    // failure.  A restored tapped-domain stash is authenticated source state
    // and therefore makes the physical endpoint mandatory before transfer.
    bool required = false;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution attribution;
    void * context = nullptr;
    bool (*memory)(void *, uint64_t &, uint64_t &) = nullptr;
    bool (*reserve)(void *) = nullptr;
};

enum class vbr_downward_reserve_status : uint8_t {
    not_attempted = 0,
    reserved,
    reserved_stashless,
    projection_unavailable,
    accounting_refused,
    workspace_reserve_failed,
    required_stash_reserve_failed,
    internal_error,
    _count,
};

const char * vbr_downward_reserve_status_name(vbr_downward_reserve_status status) noexcept;

struct vbr_downward_reserve_result {
    vbr_downward_reserve_status status = vbr_downward_reserve_status::internal_error;
    llama_cache_transaction_status transaction_status = llama_cache_transaction_status::internal_fault;
    llama_cache_admission_status admission_status = llama_cache_admission_status::internal_fault;
    uint64_t workspace_growth = 0;
    uint64_t stash_growth = 0;
    std::vector<uint64_t> stashless_units;
};

struct vbr_downward_stage_reservation {
    vbr_downward_reserve_status status =
        vbr_downward_reserve_status::not_attempted;
    std::vector<uint64_t> stashless_units;
};

inline uint64_t vbr_downward_unit_key(
        uint32_t child_id, uint32_t logical_unit_id) noexcept {
    // Zero is the closed endpoint API's invalid sentinel. The all-ones tuple
    // therefore fails closed to zero instead of aliasing a live unit.
    return ((uint64_t(child_id) << 32) | logical_unit_id) + 1;
}

// Owns the C references for persistent endpoints. The resource allocation and
// its accounting receipt intentionally have the same side-backend/pool
// lifetime. Existing bytes seen before this owner first projects are adopted
// as its uncharged baseline; only later endpoint growth is transacted.
class vbr_downward_resource_receipts {
public:
    explicit vbr_downward_resource_receipts(llama_cache_acct_ledger & ledger) noexcept;
    ~vbr_downward_resource_receipts();

    vbr_downward_resource_receipts(const vbr_downward_resource_receipts &) = delete;
    vbr_downward_resource_receipts & operator=(const vbr_downward_resource_receipts &) = delete;
    vbr_downward_resource_receipts(vbr_downward_resource_receipts &&) noexcept;
    vbr_downward_resource_receipts & operator=(vbr_downward_resource_receipts &&) noexcept;

    vbr_downward_reserve_result reserve_resources(
        const llama_cache_budget_config & budget,
        const std::vector<vbr_downward_workspace_endpoint> & workspaces,
        const std::vector<vbr_downward_stash_endpoint> & stashes) noexcept;

private:
    struct endpoint_key {
        const void * owner = nullptr;
        llama_cache_acct_category category = llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;

        bool operator==(const endpoint_key & other) const noexcept {
            return owner == other.owner && category == other.category && domain == other.domain;
        }
    };

    struct record {
        endpoint_key key;
        uint64_t endpoint = 0;
    };

    record * find(const endpoint_key & key) noexcept;
    void release_ops() noexcept;

    llama_cache_acct_ledger * ledger_ = nullptr;
    std::vector<record> records_;
    std::vector<llama_cache_acct_op_id> ops_;
};

enum class vbr_downward_transform_status : uint8_t {
    transformed = 0,
    invalid_recipe,
    stash_unavailable,
    transform_failed,
    internal_error,
    _count,
};

struct vbr_downward_transform_iface {
    void * context = nullptr;
    bool (*capture_stash)(void *, ggml_type, const std::vector<uint8_t> &, std::vector<uint8_t> &) = nullptr;
    bool (*transcode)(void *, const vbr_downward_edge &, const std::vector<uint8_t> &,
                      const std::vector<uint8_t> *, std::vector<uint8_t> &) = nullptr;
};

struct vbr_downward_transform_result {
    vbr_downward_transform_status status = vbr_downward_transform_status::internal_error;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> stash;
    bool stash_regenerated = false;
};

// The single canonical recipe walk.  CPU byte fixtures and the live KV
// importer adapt their representation-specific state to these two doors; the
// chain validation, stash boundary and edge order live here exactly once.
struct vbr_downward_edge_driver {
    void * context = nullptr;
    bool (*stash_available)(void *) noexcept = nullptr;
    bool (*capture_stash)(void *, const vbr_downward_edge &) noexcept = nullptr;
    bool (*transcode)(void *, const vbr_downward_edge &) noexcept = nullptr;
};

vbr_downward_transform_status vbr_downward_execute_edges(
        const vbr_downward_recipe & recipe,
        const vbr_downward_edge_driver & driver,
        bool & stash_regenerated,
        uint32_t * edge_reached = nullptr) noexcept;

// Injected CPU door used by the edge oracle now and by the live kernel adapter
// by transformed adoption. Intermediate recipe tiers are never published.
vbr_downward_transform_result vbr_downward_execute_recipe(
        const vbr_downward_recipe & recipe,
        const std::vector<uint8_t> & source,
        const std::vector<uint8_t> * authorized_stash,
        const vbr_downward_transform_iface & iface) noexcept;
