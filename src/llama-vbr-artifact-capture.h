#pragma once

#include "llama-cache-authority.h"
#include "llama-vbr-artifact.h"
#include "llama-vbr-generation.h"
#include "llama-vbr-pinned-ring.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Bounded streaming substrate. These types are internal to libllama:
// no live KV/cache or server policy enters this unit.
static constexpr uint64_t VBR_CAPTURE_PINNED_RING_MAX_BYTES =
    VBR_PINNED_RING_MAX_BYTES;

// Canonical authenticated range granularity. Projected transfers hash these
// chunks while copying from the backend, so later prefix projections verify
// only the chunks they cite plus a bounded proof.
static constexpr uint32_t VBR_CAPTURE_RANGE_CHUNK_BYTES = 64*1024;

// Canonical equality for the immutable controller-policy currency carried by
// capture, artifact, and occupied-replacement capabilities.
bool vbr_artifact_controller_policy_equal(
    const vbr_artifact_controller_policy & lhs,
    const vbr_artifact_controller_policy & rhs) noexcept;

enum class vbr_capture_stream_status : uint8_t {
    ok = 0,
    invalid_argument,
    ring_unavailable,
    cancelled,
    transfer_failed,
    short_read,
    duplicate_segment,
    missing_segment,
    late_segment,
    hash_mismatch,
    format_rejected,
    accounting_unavailable,
    accounting_refused,
    stage_failed,
    commit_failed,
    publication_failed,
    projection_invalid,
    snapshot_unavailable,
    snapshot_changed,
    internal_error,
    _count,
};

const char * vbr_capture_stream_status_name(
    vbr_capture_stream_status status) noexcept;

using vbr_capture_ring_create_failure =
    vbr_pinned_ring_create_failure;

const char * vbr_capture_ring_create_failure_name(
    vbr_capture_ring_create_failure failure) noexcept;
inline vbr_capture_stream_status vbr_capture_ring_failure_status(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::budget_exceeded:
            return vbr_capture_stream_status::accounting_refused;
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
        case vbr_capture_ring_create_failure::existing_ring_charge:
        case vbr_capture_ring_create_failure::accounting_update_failed:
        case vbr_capture_ring_create_failure::budget_reset_failed:
        case vbr_capture_ring_create_failure::budget_unavailable:
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return vbr_capture_stream_status::accounting_unavailable;
        case vbr_capture_ring_create_failure::internal_error:
            return vbr_capture_stream_status::internal_error;
        case vbr_capture_ring_create_failure::none:
        case vbr_capture_ring_create_failure::invalid_geometry:
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
        case vbr_capture_ring_create_failure::invalid_lane_binding:
        case vbr_capture_ring_create_failure::duplicate_device_lane:
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
        case vbr_capture_ring_create_failure::host_buffer_too_small:
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
        case vbr_capture_ring_create_failure::lane_underprovisioned:
        case vbr_capture_ring_create_failure::_count:
            return vbr_capture_stream_status::ring_unavailable;
    }
    return vbr_capture_stream_status::ring_unavailable;
}

struct artifact_segment {
    std::shared_ptr<const std::vector<uint8_t>> storage;
    uint64_t offset = 0;
    uint64_t length = 0;
};

using vbr_capture_continue_fn = bool (*)(void * context) noexcept;

// Immutable segmented pageable backing. Each append allocates at most one
// capture chunk; source() supports arbitrary reads across segment boundaries
// and never concatenates the complete artifact.
class artifact_segment_chain {
public:
    artifact_segment_chain();
    // Known-size construction incrementally authenticates the canonical
    // segment-stream digest during append(). A complete chain can therefore
    // be sealed without rereading its payload; incomplete/overlong chains
    // fail closed. Legacy construction retains scan-on-demand behavior.
    explicit artifact_segment_chain(uint64_t expected_stream_bytes);
    artifact_segment_chain(
        uint32_t authenticated_chunk_bytes,
        uint32_t max_authenticated_chunks);
    ~artifact_segment_chain();

    artifact_segment_chain(const artifact_segment_chain &) = delete;
    artifact_segment_chain & operator=(const artifact_segment_chain &) = delete;
    artifact_segment_chain(artifact_segment_chain &&) noexcept;
    artifact_segment_chain & operator=(artifact_segment_chain &&) noexcept;

    bool append(const uint8_t * data, size_t size) noexcept;
    // Transfers one already-filled bounded chunk without copying its payload.
    // The chain still computes every enabled authentication digest itself.
    bool append_owned(std::vector<uint8_t> data) noexcept;
    uint64_t size() const noexcept;
    size_t segment_count() const noexcept;
    size_t max_segment_size() const noexcept;
    bool authenticated() const noexcept;
    // A nonzero digest is available only after the immutable range tree has
    // sealed the complete chain. Restore staging can reuse this capability
    // instead of rereading a multi-GiB catalog payload.
    std::array<uint8_t, 32> sealed_digest() const noexcept;
    bool read(uint64_t offset, uint8_t * destination, size_t size) const noexcept;
    vbr_artifact_byte_source source() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    bool append_storage(
        std::shared_ptr<std::vector<uint8_t>> bytes) noexcept;
    friend bool vbr_capture_range_seal(
        artifact_segment_chain &,
        uint64_t,
        class vbr_capture_range_tree &,
        void *,
        vbr_capture_continue_fn,
        bool *) noexcept;
    friend std::array<uint8_t, 32> vbr_capture_stream_digest(
        const artifact_segment_chain &) noexcept;
};

struct vbr_capture_authenticated_range {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct vbr_capture_range_proof_limits {
    uint32_t max_ranges = 4096;
    uint32_t max_selected_chunks = 1048576;
    uint32_t max_proof_nodes = 1048576;
    uint64_t max_metadata_bytes = uint64_t(64)*1024*1024;
};

enum class vbr_capture_range_restrict_status : uint8_t {
    restricted = 0,
    invalid_argument,
    parent_invalid,
    range_unauthorized,
    limit_exceeded,
    internal_error,
    _count,
};

// Immutable Merkle owner produced by parallel hashing after the complete
// pageable segment chain has been captured. The root binds total bytes,
// canonical chunking, and the complete padded tree.
class vbr_capture_range_tree {
public:
    vbr_capture_range_tree() noexcept = default;
    explicit operator bool() const noexcept;
    uint64_t total_bytes() const noexcept;
    uint32_t chunk_bytes() const noexcept;
    uint32_t chunk_count() const noexcept;
    uint64_t metadata_bytes() const noexcept;
    const std::array<uint8_t, 32> & root() const noexcept;

private:
    struct data;
    explicit vbr_capture_range_tree(
        std::shared_ptr<const data> data) noexcept;
    std::shared_ptr<const data> data_;

    friend class artifact_segment_chain;
    friend bool vbr_capture_range_seal(
        artifact_segment_chain &,
        uint64_t,
        vbr_capture_range_tree &,
        void *,
        vbr_capture_continue_fn,
        bool *) noexcept;
    friend bool vbr_capture_range_prove(
        const vbr_capture_range_tree &,
        const std::vector<vbr_capture_authenticated_range> &,
        const vbr_capture_range_proof_limits &,
        class vbr_capture_range_proof &) noexcept;
};

// Immutable proof capability. Selected chunks are stored as indices rather
// than payload copies; verification reads them from the supplied byte source.
class vbr_capture_range_proof {
public:
    vbr_capture_range_proof() noexcept = default;
    explicit operator bool() const noexcept;
    const std::array<uint8_t, 32> & root() const noexcept;
    uint64_t total_bytes() const noexcept;
    uint32_t chunk_bytes() const noexcept;
    const std::vector<vbr_capture_authenticated_range> & ranges() const noexcept;
    uint32_t selected_chunk_count() const noexcept;
    uint32_t proof_node_count() const noexcept;
    uint64_t metadata_bytes() const noexcept;

private:
    struct data;
    explicit vbr_capture_range_proof(
        std::shared_ptr<const data> data) noexcept;
    std::shared_ptr<const data> data_;

    friend bool vbr_capture_range_prove(
        const vbr_capture_range_tree &,
        const std::vector<vbr_capture_authenticated_range> &,
        const vbr_capture_range_proof_limits &,
        vbr_capture_range_proof &) noexcept;
    friend vbr_capture_range_restrict_status vbr_capture_range_restrict(
        const vbr_capture_range_proof &,
        const std::vector<vbr_capture_authenticated_range> &,
        const vbr_capture_range_proof_limits &,
        vbr_capture_range_proof &) noexcept;
    friend bool vbr_capture_range_verify(
        const vbr_capture_range_proof &,
        const vbr_artifact_byte_source &,
        uint64_t *) noexcept;
};

// Sealing is valid only for a chain constructed with authenticated chunking,
// after all appends complete. All failures clear output and leave the chain
// readable but permanently unsealed for publication. Optional continuation
// polling bounds cancellation while large chains are hashed.
bool vbr_capture_range_seal(
    artifact_segment_chain & chain,
    uint64_t max_metadata_bytes,
    vbr_capture_range_tree & output,
    void * continue_context = nullptr,
    vbr_capture_continue_fn continue_work = nullptr,
    bool * was_cancelled = nullptr) noexcept;

// Ranges must be sorted, nonempty, nonoverlapping, and within the sealed byte
// extent. Proof construction and verification are transactional/fail-closed.
bool vbr_capture_range_prove(
    const vbr_capture_range_tree & tree,
    const std::vector<vbr_capture_authenticated_range> & ranges,
    const vbr_capture_range_proof_limits & limits,
    vbr_capture_range_proof & output) noexcept;

// Derives a manifest-local proof for an exact subset of a parent proof's
// authorized byte ranges. The parent retains the selected leaf digests needed
// to construct the child multiproof, so this operation never reads payload or
// widens authority to another manifest's rows. Ranges use the same canonical
// sorted/nonempty contract as vbr_capture_range_prove().
vbr_capture_range_restrict_status vbr_capture_range_restrict(
    const vbr_capture_range_proof & parent,
    const std::vector<vbr_capture_authenticated_range> & ranges,
    const vbr_capture_range_proof_limits & limits,
    vbr_capture_range_proof & output) noexcept;

// Verifies only selected canonical chunks. bytes_read is optional diagnostic
// evidence and is reset on every path.
bool vbr_capture_range_verify(
    const vbr_capture_range_proof & proof,
    const vbr_artifact_byte_source & source,
    uint64_t * bytes_read = nullptr) noexcept;

std::array<uint8_t, 32> vbr_capture_stream_digest(
    const artifact_segment_chain & chain) noexcept;

using vbr_capture_lane = vbr_pinned_ring_lane;

using vbr_capture_ring_accounting =
    vbr_pinned_ring_accounting;

struct vbr_capture_stream_source {
    using read_fn = bool (*)(
        const void * context,
        uint64_t offset,
        uint8_t * destination,
        size_t size) noexcept;

    uint32_t lane = 0;
    uint64_t size = 0;

    // Exactly one source shape is used. tensor != nullptr selects backend D2H;
    // otherwise read supplies deterministic CPU/synthetic bytes.
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    const ggml_tensor * tensor = nullptr;
    uint64_t tensor_offset = 0;
    const void * context = nullptr;
    read_fn read = nullptr;

    // Optional synchronous cancellation probe. It is called before each
    // bounded ring chunk is read/submitted, never while a backend event is
    // enqueued. False drains any already-pending events and aborts the pump;
    // higher-level capture keeps the
    // partial private chain unreachable and publishes no completed unit.
    void * continue_context = nullptr;
    bool (*continue_transfer)(void * context) noexcept = nullptr;

    // Deterministic synthetic completion-fault seam. Production callers keep
    // UINT64_MAX; tests prove a failed completion drains the ring and exposes
    // no verified segment.
    uint64_t fail_completion_at = UINT64_MAX;
};

struct vbr_capture_stream_range {
    uint64_t source_offset = 0;
    uint64_t size = 0;
};

struct vbr_capture_stream_stats {
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t submitted_bytes = 0;
    uint64_t submitted_chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    size_t max_segment_size = 0;
    std::array<uint8_t, 32> streaming_digest = {};
};

// Sequence-projected capture planning. Each logical manifest contributes
// exact live placement evidence; the planner lowers their physical-row union
// into deterministic runs whose dependency sets identify precisely which
// manifests must be cancelled if that run cannot be sealed. This is a
// process-local capture plan, not wire metadata.
struct vbr_capture_projection_manifest {
    uint64_t manifest_id = 0;
    // False records a preflight dependency refusal without inventing physical
    // placement. The planner retains the semantic result row but excludes it
    // from the transfer union; false therefore cannot broaden publication.
    bool dependencies_available = true;
    // Exact semantic frontier captured with the placement evidence. These
    // fields are retained by the sealed projection and later become the
    // catalog manifest authority; publication callers cannot substitute a
    // different identity, token block, or generation record after transfer.
    std::array<uint8_t, 32> identity_policy_order_digest = {};
    vbr_artifact_identity_block identity;
    vbr_artifact_token_block token_block;
    vbr_checkpoint_generation_record generation;
    std::vector<vbr_artifact_companion_payload> companions;
    std::vector<vbr_artifact_stream_placement> placements;
};

struct vbr_capture_projection_segment {
    uint32_t first_physical_cell = 0;
    uint32_t cell_count = 0;
    uint32_t first_dependency = 0;
    uint32_t dependency_count = 0;
    uint64_t packed_first_row = 0;
};

struct vbr_capture_projection_stream {
    uint32_t child_id = UINT32_MAX;
    uint32_t stream_index = UINT32_MAX;
    std::vector<vbr_capture_projection_segment> segments;
};

struct vbr_capture_projection_limits {
    uint32_t max_manifests = 4096;
    uint32_t max_placements = 4096;
    uint32_t max_input_cells = 1048576;
    uint32_t max_union_cells = 1048576;
    uint32_t max_segments = 1048576;
    uint32_t max_dependency_references = 1048576;
    // Semantic frontier evidence is retained by the sealed projection. Keep
    // its nested arenas under the same explicit bounded-input contract as the
    // physical placement plan; these counts are aggregate across the batch.
    uint32_t max_token_ids = 1048576;
    uint32_t max_string_bytes = 1048576;
    uint32_t max_generation_controllers = 4096;
    uint32_t max_generation_units = 1048576;
    uint32_t max_generation_streams = 4096;
    uint32_t max_generation_pages = 1048576;
    uint32_t max_companions = 16384;
    uint64_t max_companion_payload_bytes = uint64_t(16)*1024*1024*1024;
    uint64_t max_semantic_metadata_bytes = uint64_t(64)*1024*1024;
};

// One batch is structurally bound to one live memory-tree namespace. Child
// and stream IDs are meaningful only within that immutable source namespace;
// callers must start a separate batch for another live tree.
struct vbr_capture_projection_batch {
    uint64_t source_namespace = 0;
    std::vector<vbr_capture_projection_manifest> manifests;
};

struct vbr_capture_projection_plan {
    uint64_t source_namespace = 0;
    uint32_t manifest_count = 0;
    uint32_t placement_count = 0;
    uint64_t input_cell_references = 0;
    uint64_t union_cell_count = 0;
    uint64_t dependency_references = 0;
    std::vector<vbr_capture_projection_manifest> manifests;
    std::vector<vbr_capture_projection_stream> streams;
    std::vector<uint64_t> dependent_manifest_ids;
};

class vbr_pinned_chunk_ring;

// Process-local sealed projection capability. Construction is owned by the
// planner, so no mutable shared_ptr alias can outlive validation and race a
// projected transfer. Copying this handle shares immutable plan storage.
class vbr_capture_projection {
public:
    vbr_capture_projection() = default;
    const vbr_capture_projection_plan * operator->() const noexcept;
    const vbr_capture_projection_plan & operator*() const noexcept;
    explicit operator bool() const noexcept;
    bool operator==(const vbr_capture_projection & other) const noexcept;

private:
    explicit vbr_capture_projection(
        std::shared_ptr<const vbr_capture_projection_plan> plan) noexcept;
    std::shared_ptr<const vbr_capture_projection_plan> plan_;
    friend bool vbr_artifact_project_capture_union(
        const vbr_capture_projection_batch &,
        const vbr_capture_projection_limits &,
        vbr_capture_projection &) noexcept;
};

// Allocation failure, malformed placement evidence, duplicate identities, or
// any limit violation clears output and returns false.
bool vbr_artifact_project_capture_union(
    const vbr_capture_projection_batch & batch,
    const vbr_capture_projection_limits & limits,
    vbr_capture_projection & output) noexcept;

// One globally-bounded ring split across per-device lanes. A null device lane
// is the deterministic CPU test path. Real lanes allocate that device's host
// buffer type and use optional backend events; no event means a synchronized
// fallback, never an unbounded allocation.
class vbr_pinned_chunk_ring {
public:
    static std::unique_ptr<vbr_pinned_chunk_ring> create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting =
            nullptr,
        vbr_capture_ring_create_failure * failure =
            nullptr) noexcept;

    // Builds the D2H adapter over an already-accounted persistent core.
    static std::unique_ptr<vbr_pinned_chunk_ring> attach(
        std::shared_ptr<vbr_bounded_pinned_ring_core> core) noexcept;

    ~vbr_pinned_chunk_ring();
    vbr_pinned_chunk_ring(const vbr_pinned_chunk_ring &) = delete;
    vbr_pinned_chunk_ring & operator=(const vbr_pinned_chunk_ring &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;

    // Reserves the direction-neutral transport for a complete projected
    // batch. The move-only token releases on destruction.
    vbr_pinned_ring_operation try_begin_operation() noexcept;

    vbr_capture_stream_status stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;

    // Streams an ordered, non-overlapping set of subranges into one packed
    // immutable chain and one digest pass. Source offsets remain relative to
    // source.tensor_offset (tensor) or the read callback's logical source.
    vbr_capture_stream_status stream_ranges(
        const vbr_capture_stream_source & source,
        const std::vector<vbr_capture_stream_range> & ranges,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;

    vbr_capture_stream_status stream_ranges_reserved(
        const vbr_pinned_ring_operation & operation,
        const vbr_capture_stream_source & source,
        const std::vector<vbr_capture_stream_range> & ranges,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;

private:
    struct impl;
    explicit vbr_pinned_chunk_ring(std::unique_ptr<impl> state) noexcept;
    vbr_capture_stream_status stream_ranges_impl(
        const vbr_pinned_ring_operation * operation,
        const vbr_capture_stream_source & source,
        const vbr_capture_stream_range * ranges,
        size_t range_count,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;
    std::unique_ptr<impl> impl_;
};

struct vbr_capture_unit_snapshot {
    uint64_t source_namespace = 0;
    uint32_t child_id = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    vbr_lineage_uuid lineage_uuid = {};
    uint64_t controller_generation = 0;
    uint64_t mutation_serial = 0;
    vbr_unit_generation generation;
    uint32_t shard_count = 0;
    std::array<uint8_t, 32> shard_topology_digest = {};
};

// acquire() obtains the short unit-version lease and snapshots the exact
// representation tuple. recheck() runs after every shard has completed;
// release() is called exactly once on every post-acquire terminal.
struct vbr_capture_unit_snapshot_provider {
    using acquire_fn = bool (*)(
        void * context,
        uint64_t source_namespace,
        uint32_t child_id,
        uint32_t logical_unit_id,
        vbr_capture_unit_snapshot & output) noexcept;
    using recheck_fn = bool (*)(
        void * context,
        const vbr_capture_unit_snapshot & expected) noexcept;
    using release_fn = void (*)(
        void * context,
        const vbr_capture_unit_snapshot & snapshot) noexcept;

    void * context = nullptr;
    acquire_fn acquire = nullptr;
    recheck_fn recheck = nullptr;
    release_fn release = nullptr;
};

struct vbr_capture_projected_shard_source {
    uint32_t shard_index = UINT32_MAX;
    uint32_t row_count = 0;
    uint64_t row_bytes = 0;
    // Provider-owned stable identity for this physical shard source. The
    // snapshot authenticates the complete ordered identity/geometry set.
    uint64_t source_identity = 0;
    vbr_capture_stream_source source;
};

struct vbr_capture_projected_shard {
    uint32_t shard_index = UINT32_MAX;
    uint32_t source_row_count = 0;
    uint64_t row_bytes = 0;
    uint64_t source_identity = 0;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
    vbr_capture_range_tree authenticated_ranges;
};

struct vbr_capture_projected_transfer_limits {
    uint32_t max_shards = 128;
    uint64_t max_shard_segment_references = 1048576;
    // Exact aggregate source reads / async D2H enqueues while the unit-version
    // lease is held. More fragmented unions must use a bounded packed view.
    uint32_t max_source_operations = 4096;
    uint64_t max_total_packed_bytes = uint64_t(16)*1024*1024*1024;
    uint32_t max_authenticated_chunks = 262144;
    uint64_t max_authenticated_metadata_bytes = uint64_t(32)*1024*1024;
};

// Immutable unit capability minted only after the complete projected transfer
// and its terminal representation recheck succeed. Copying is shallow; no
// mutable segment-chain alias escapes the minting boundary.
class vbr_capture_projected_unit {
  public:
    vbr_capture_projected_unit() noexcept = default;

    explicit operator bool() const noexcept;
    const vbr_capture_projection & projection() const noexcept;
    const vbr_capture_unit_snapshot & snapshot() const noexcept;
    uint32_t child_id() const noexcept;
    uint32_t stream_index() const noexcept;
    uint32_t logical_unit_id() const noexcept;
    uint64_t packed_bytes() const noexcept;
    const vbr_capture_stream_stats & transfer() const noexcept;
    const std::vector<vbr_capture_projected_shard> & shards() const noexcept;

  private:
    struct data;
    explicit vbr_capture_projected_unit(
        std::shared_ptr<const data> data) noexcept;

    std::shared_ptr<const data> data_;

    friend vbr_capture_stream_status vbr_capture_projected_unit_transfer(
        vbr_capture_projection,
        uint32_t,
        uint32_t,
        uint32_t,
        const std::vector<vbr_capture_projected_shard_source> &,
        const vbr_capture_projected_transfer_limits &,
        const vbr_capture_unit_snapshot_provider &,
        vbr_pinned_chunk_ring &,
        vbr_capture_projected_unit &,
        vbr_capture_stream_stats *,
        const vbr_pinned_ring_operation *) noexcept;
};

// A controller-authenticated representation target for one projected child.
// The target is a value capability: assembly owns its copy, and the
// provider below rechecks that the canonical controller simulator still
// considers the complete tuple reachable before any manifest becomes ready.
struct vbr_capture_controller_target {
    uint64_t manifest_id = 0;
    uint64_t source_namespace = 0;
    uint32_t child_id = UINT32_MAX;
    vbr_lineage_uuid lineage_uuid = {};
    uint64_t controller_generation = 0;
    vbr_artifact_controller_policy policy;
    std::vector<vbr_unit_generation> units;
    // Complete immutable schema for each logical unit. The controller
    // provider authenticates this alongside the generation tuple, so catalog
    // publication cannot relabel one captured byte owner as another side,
    // layout, representation, or device geometry.
    std::vector<vbr_artifact_unit_descriptor> unit_descriptors;
};

// Exact payload-pointer-independent representation equality. Manifest ID and
// source namespace are reference-local and deliberately excluded; callers
// authenticate those separately before sharing or rechecking a controller
// tuple.
bool vbr_capture_controller_representation_equal(
    const vbr_capture_controller_target & lhs,
    const vbr_capture_controller_target & rhs) noexcept;

struct vbr_capture_controller_target_provider {
    using recheck_fn = bool (*)(
        void * context,
        uint64_t manifest_id,
        const vbr_capture_controller_target * targets,
        size_t target_count) noexcept;

    void * context = nullptr;
    recheck_fn recheck = nullptr;
};

enum class vbr_capture_manifest_state : uint8_t {
    ready = 0,
    dependency_unavailable,
};

struct vbr_capture_manifest_result {
    uint64_t manifest_id = 0;
    vbr_capture_manifest_state state =
        vbr_capture_manifest_state::dependency_unavailable;
    uint32_t first_controller = 0;
    uint32_t controller_count = 0;
    uint32_t first_unit = 0;
    uint32_t unit_count = 0;
    uint32_t first_range_proof = 0;
    uint32_t range_proof_count = 0;
};

struct vbr_capture_manifest_range_proof {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    vbr_capture_range_proof proof;
};

struct vbr_capture_manifest_assembly_limits {
    // A target is scoped to one manifest/child, so the projection's
    // 4,096-placement ceiling is also the reachable target/reference ceiling.
    uint32_t max_controller_targets = 4096;
    uint32_t max_projected_units = 16384;
    uint64_t max_unit_descriptor_shards = 1048576;
    uint64_t max_unit_descriptor_metadata_bytes =
        uint64_t(64)*1024*1024;
    uint32_t max_manifests = 4096;
    uint64_t max_controller_references = 4096;
    uint64_t max_unit_references = 1048576;
    // A packed fallback is required before one batch can retain more proof
    // owners than projected units. This prevents millions of tiny heap-owned
    // proofs even when their element arenas fit the byte cap.
    uint64_t max_range_proofs = 16384;
    uint64_t max_range_proof_metadata_bytes = uint64_t(64)*1024*1024;
    vbr_capture_range_proof_limits range_proof;
};

// One immutable transport batch can finish partially: a missing, stale, or
// no-longer-reachable unit invalidates only manifests whose projection names
// it. Ready rows reference assembly-owned controller targets and sealed unit
// capabilities through flat bounded arenas. Construction is private so no
// mutable alias can invalidate the certified controller tuple afterward.
class vbr_capture_manifest_assembly {
  public:
    vbr_capture_manifest_assembly() noexcept = default;

    explicit operator bool() const noexcept;
    const vbr_capture_projection & projection() const noexcept;
    const std::vector<vbr_capture_controller_target> &
        controller_targets() const noexcept;
    const std::vector<vbr_capture_projected_unit> &
        projected_units() const noexcept;
    const std::vector<uint32_t> & controller_references() const noexcept;
    const std::vector<uint32_t> & unit_references() const noexcept;
    const std::vector<vbr_capture_manifest_range_proof> &
        range_proofs() const noexcept;
    const std::vector<vbr_capture_manifest_result> & manifests() const noexcept;

  private:
    struct data;
    explicit vbr_capture_manifest_assembly(
        std::shared_ptr<const data> data) noexcept;

    std::shared_ptr<const data> data_;

    friend bool vbr_capture_assemble_manifests(
        vbr_capture_projection,
        std::vector<vbr_capture_controller_target> &&,
        std::vector<vbr_capture_projected_unit> &&,
        const vbr_capture_controller_target_provider &,
        const vbr_capture_manifest_assembly_limits &,
        vbr_capture_manifest_assembly &) noexcept;
};

// Structural corruption, duplicate identities, or a limit violation clears
// output and returns false. Dependency failure is a successful assembly with
// the affected manifest rows marked dependency_unavailable. The two rvalue
// inventories transfer ownership into the immutable result on success and
// prevent an implicit deep copy from occurring before limit validation.
bool vbr_capture_assemble_manifests(
    vbr_capture_projection projection,
    std::vector<vbr_capture_controller_target> && controller_targets,
    std::vector<vbr_capture_projected_unit> && projected_units,
    const vbr_capture_controller_target_provider & targets,
    const vbr_capture_manifest_assembly_limits & limits,
    vbr_capture_manifest_assembly & output) noexcept;

// Computes the canonical topology identity consumed by the snapshot owner.
// Sources may be supplied in any order, but must form exactly [0, count).
bool vbr_capture_projected_shard_topology(
    const std::vector<vbr_capture_projected_shard_source> & sources,
    uint32_t & shard_count,
    std::array<uint8_t, 32> & digest) noexcept;

// Captures one complete logical tensor unit across the exact shard topology
// authenticated by the snapshot owner. No output becomes visible unless every
// range transfers and the unit snapshot recheck succeeds. The sealed
// projection capability keeps each slice's dependency offsets immutable for
// the lifetime of the result.
vbr_capture_stream_status vbr_capture_projected_unit_transfer(
    vbr_capture_projection projection,
    uint32_t child_id,
    uint32_t stream_index,
    uint32_t logical_unit_id,
    const std::vector<vbr_capture_projected_shard_source> & sources,
    const vbr_capture_projected_transfer_limits & limits,
    const vbr_capture_unit_snapshot_provider & snapshots,
    vbr_pinned_chunk_ring & ring,
    vbr_capture_projected_unit & output,
    vbr_capture_stream_stats * attempted = nullptr,
    const vbr_pinned_ring_operation * operation = nullptr) noexcept;

struct vbr_verified_segment {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    bool clean_stash = false;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
};

struct vbr_verified_companion {
    uint32_t companion_index = UINT32_MAX;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
};

// Projected companions cross the catalog boundary as a consumptive immutable
// capability. The factory takes unique ownership of the completed chain and
// authenticates it before publication, so no mutable shared_ptr alias can
// rewrite bytes after the batch preflight.
class vbr_capture_sealed_companion {
  public:
    vbr_capture_sealed_companion() noexcept = default;
    vbr_capture_sealed_companion(
        vbr_capture_sealed_companion &&) noexcept = default;
    vbr_capture_sealed_companion & operator=(
        vbr_capture_sealed_companion &&) noexcept = default;
    vbr_capture_sealed_companion(
        const vbr_capture_sealed_companion &) = delete;
    vbr_capture_sealed_companion & operator=(
        const vbr_capture_sealed_companion &) = delete;
    explicit operator bool() const noexcept;
    uint32_t companion_index() const noexcept;
    uint64_t size() const noexcept;
    const std::array<uint8_t, 32> & streaming_digest() const noexcept;

  private:
    uint32_t companion_index_ = UINT32_MAX;
    std::shared_ptr<const artifact_segment_chain> bytes_;
    std::array<uint8_t, 32> streaming_digest_ = {};

    friend bool vbr_capture_seal_companion(
        uint32_t, std::unique_ptr<artifact_segment_chain>,
        vbr_capture_sealed_companion &) noexcept;
    friend class llama_vbr_artifact_catalog;
};

bool vbr_capture_seal_companion(
    uint32_t companion_index,
    std::unique_ptr<artifact_segment_chain> bytes,
    vbr_capture_sealed_companion & output) noexcept;

struct vbr_capture_sink_result {
    vbr_capture_stream_status status =
        vbr_capture_stream_status::internal_error;
    llama_cache_acct_artifact_id reference_artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id reference_lineage;
    bool adopted = false;
};

enum class vbr_capture_reservation_group : uint8_t {
    none = 0,
    transfer_staging,
    durable_artifact,
    _count,
};

const char * vbr_capture_reservation_group_name(
    vbr_capture_reservation_group group) noexcept;

struct vbr_capture_begin_diagnostics {
    vbr_capture_reservation_group reservation_group =
        vbr_capture_reservation_group::none;
    llama_cache_prepare_status prepare_status =
        llama_cache_prepare_status::prepared;
    llama_cache_admission_status admission_status =
        llama_cache_admission_status::admitted;
    size_t failed_leaf = SIZE_MAX;
};

class vbr_unit_build {
public:
    virtual ~vbr_unit_build() = default;
    virtual vbr_capture_stream_status accept_verified_segment(
        const vbr_verified_segment & segment) noexcept = 0;
    virtual vbr_capture_stream_status seal_unit() noexcept = 0;
};

class vbr_capture_build {
public:
    virtual ~vbr_capture_build() = default;
    virtual std::unique_ptr<vbr_unit_build> begin_unit(
        uint32_t unit_index,
        vbr_capture_stream_status & status) noexcept = 0;
    virtual vbr_capture_stream_status accept_verified_companion(
        const vbr_verified_companion & companion) noexcept = 0;
    virtual vbr_capture_sink_result publish_reference() noexcept = 0;
};

class vbr_unit_version_sink {
public:
    virtual ~vbr_unit_version_sink() = default;
    virtual std::unique_ptr<vbr_capture_build> begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics =
            nullptr) noexcept = 0;
};
