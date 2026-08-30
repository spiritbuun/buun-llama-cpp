#pragma once

#include "llama-cache-accounting.h"
#include "llama-vbr-generation-types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Internal VBR artifact format. This is intentionally not part of public llama.h:
// This header defines immutable value types and a fail-closed streaming codec.
constexpr uint32_t VBR_UNIT_ARTIFACT_FORMAT_VERSION_MIN = 1;
constexpr uint32_t VBR_UNIT_ARTIFACT_FORMAT_VERSION = 3;
constexpr uint32_t VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT = 2;
constexpr uint32_t VBR_UNIT_ARTIFACT_FORMAT_VERSION_MEANSUB_REFERENCE = 3;
constexpr uint32_t VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION = 1;

constexpr bool artifact_has_reference_placement(uint32_t version) {
    return version >= VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT;
}

constexpr bool artifact_has_meansub_reference(uint32_t version) {
    return version >= VBR_UNIT_ARTIFACT_FORMAT_VERSION_MEANSUB_REFERENCE;
}

struct vbr_unit_version_id_tag;
struct vbr_payload_digest_tag;
struct vbr_stash_payload_id_tag;
struct vbr_manifest_digest_tag;
struct vbr_capture_generation_id_tag;
struct vbr_transition_lineage_id_tag;
struct vbr_token_block_digest_tag;

using vbr_unit_version_id       = llama_cache_acct_digest<vbr_unit_version_id_tag>;
using vbr_payload_digest        = llama_cache_acct_digest<vbr_payload_digest_tag>;
using vbr_stash_payload_id      = llama_cache_acct_digest<vbr_stash_payload_id_tag>;
using vbr_manifest_digest       = llama_cache_acct_digest<vbr_manifest_digest_tag>;
using vbr_capture_generation_id = llama_cache_acct_digest<vbr_capture_generation_id_tag>;
using vbr_transition_lineage_id = llama_cache_acct_digest<vbr_transition_lineage_id_tag>;
using vbr_token_block_digest    = llama_cache_acct_digest<vbr_token_block_digest_tag>;

enum class vbr_artifact_status : uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_version,
    malformed,
    out_of_bounds,
    checksum_mismatch,
    content_id_mismatch,
    topology_mismatch,
    generation_mismatch,
    accounting_unavailable,
    internal_error,
    _count,
};

enum class vbr_artifact_layout : uint8_t {
    row_major = 0,
    _count,
};

enum class vbr_artifact_side : uint8_t {
    key = 0,
    value,
    _count,
};

enum class vbr_artifact_representation_kind : uint8_t {
    raw = 0,
    lossless,
    approximate,
    _count,
};

enum class vbr_artifact_recoverability : uint8_t {
    sealed_payload = 0,
    _count,
};

enum class vbr_artifact_clean_stash_state : uint8_t {
    absent_at_source = 0,
    present,
    omitted_source_present,
    _count,
};

enum class vbr_artifact_consistency_kind : uint8_t {
    capture_exact = 0,
    live_rebased,
    _count,
};

enum class vbr_artifact_section_kind : uint8_t {
    topology_table = 0,
    unit_blob,
    companion_payload,
    reference_manifest,
    _count,
};

enum class vbr_artifact_companion_kind : uint8_t {
    recurrent = 0,
    typed_accelerator,
    required_spec_payload,
    frontier_logits,
    qsa_index,
    _count,
};

enum class vbr_artifact_accounting_role : uint8_t {
    unit_payload = 0,
    clean_stash_payload,
    recurrent_payload,
    typed_accelerator_payload,
    descriptor_metadata,
    reference_metadata,
    _count,
};

static_assert(static_cast<uint8_t>(vbr_artifact_status::_count) == 11);
static_assert(static_cast<uint8_t>(vbr_artifact_layout::_count) == 1);
static_assert(static_cast<uint8_t>(vbr_artifact_side::_count) == 2);
static_assert(static_cast<uint8_t>(vbr_artifact_representation_kind::_count) == 3);
static_assert(static_cast<uint8_t>(vbr_artifact_recoverability::_count) == 1);
static_assert(static_cast<uint8_t>(vbr_artifact_clean_stash_state::_count) == 3);
static_assert(static_cast<uint8_t>(vbr_artifact_consistency_kind::_count) == 2);
static_assert(static_cast<uint8_t>(vbr_artifact_section_kind::_count) == 4);
static_assert(static_cast<uint8_t>(vbr_artifact_companion_kind::_count) == 5);
static_assert(static_cast<uint8_t>(vbr_artifact_accounting_role::_count) == 6);

// Repeatable byte source. The encoder's final write pass recomputes and verifies every embedded
// checksum/content ID, so a source mutation between preparation and publication fails closed.
struct vbr_artifact_byte_source {
    using read_fn = bool (*)(
        const void * context,
        uint64_t offset,
        uint8_t * destination,
        size_t size) noexcept;

    uint64_t size = 0;
    const void * context = nullptr;
    read_fn read = nullptr;

    bool valid() const noexcept {
        return size == 0 || read != nullptr;
    }
};

// Sequential envelope I/O. A failed encoder leaves a partial, unpublished sink by contract;
// callers publish only after an `ok` result.
struct vbr_artifact_stream_writer {
    using write_fn = bool (*)(
        void * context,
        const uint8_t * data,
        size_t size) noexcept;

    void * context = nullptr;
    write_fn write = nullptr;
};

struct vbr_artifact_stream_reader {
    using read_fn = bool (*)(
        void * context,
        uint8_t * data,
        size_t size) noexcept;

    void * context = nullptr;
    read_fn read = nullptr;
};

// Decode never needs to allocate one payload-sized vector. The consumer receives verified-order
// chunks while the decoder independently verifies section, payload, stash, unit, and package IDs.
struct vbr_artifact_payload_consumer {
    using consume_fn = bool (*)(
        void * context,
        vbr_artifact_section_kind section,
        uint32_t object_index,
        uint32_t shard_index,
        bool clean_stash,
        uint64_t offset,
        uint64_t total_size,
        const uint8_t * data,
        size_t size) noexcept;
    using finish_fn = void (*)(
        void * context,
        bool verified) noexcept;

    void * context = nullptr;
    consume_fn consume = nullptr;
    // For a valid consume/finish pair, called exactly once on every decode attempt: true only
    // after every package checksum/ID/manifest validation succeeds, false on every failure path
    // (including before the first consumed byte). A consumer stages privately until true.
    finish_fn finish = nullptr;
};

struct vbr_artifact_decode_limits {
    uint64_t max_total_bytes = 0; // required caller-preflighted bound
    uint32_t max_topologies = 16;
    uint32_t max_devices_per_topology = 128;
    uint32_t max_unit_blobs = 65536;
    uint32_t max_shards_per_unit = 128;
    uint32_t max_companions = 256;
    uint32_t max_controllers = 16;
    uint32_t max_units_per_controller = 65536;
    uint32_t max_streams_per_controller = 256;
    uint32_t max_pages_per_stream = 1048576;
    uint32_t max_accounting_rows = 1048576;
    uint32_t max_stream_placements = 4096;
    uint32_t max_placement_cells = 1048576;
    uint32_t max_token_ids = 1048576;
    uint32_t max_string_bytes = 1048576;
};

using vbr_artifact_portable_topology = llama_cache_acct_shard_topology;

// This is package-local rather than llama_cache_acct_resource_domain because topology_index
// addresses the envelope's portable topology table, not a process-local interned topology ID.
struct vbr_artifact_portable_domain {
    llama_cache_acct_residency residency = llama_cache_acct_residency::not_applicable;
    llama_cache_acct_domain_kind kind = llama_cache_acct_domain_kind::not_applicable;
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
};

inline bool operator==(const vbr_artifact_portable_domain & lhs,
                       const vbr_artifact_portable_domain & rhs) {
    return lhs.residency == rhs.residency &&
           lhs.kind == rhs.kind &&
           lhs.topology_index == rhs.topology_index &&
           lhs.device_ordinal == rhs.device_ordinal;
}
inline bool operator!=(const vbr_artifact_portable_domain & lhs,
                       const vbr_artifact_portable_domain & rhs) {
    return !(lhs == rhs);
}
inline bool vbr_artifact_portable_domain_less(
        const vbr_artifact_portable_domain & lhs,
        const vbr_artifact_portable_domain & rhs) {
    if (lhs.residency != rhs.residency) {
        return lhs.residency < rhs.residency;
    }
    if (lhs.kind != rhs.kind) {
        return lhs.kind < rhs.kind;
    }
    if (lhs.topology_index != rhs.topology_index) {
        return lhs.topology_index < rhs.topology_index;
    }
    return lhs.device_ordinal < rhs.device_ordinal;
}

struct vbr_artifact_portable_accounting_row {
    vbr_artifact_accounting_role role = vbr_artifact_accounting_role::unit_payload;
    vbr_artifact_portable_domain domain;
    uint64_t logical_bytes = 0;
    uint64_t resident_bytes = 0;
    llama_cache_acct_attr_kind attribution = llama_cache_acct_attr_kind::artifact;
};

struct vbr_artifact_shard_descriptor {
    uint32_t shard_index = 0;
    uint32_t topology_index = 0;
    uint16_t device_ordinal = 0;
    uint64_t logical_offset = 0;
    uint64_t row_count = 0;
    uint64_t column_count = 0;
    uint64_t row_bytes = 0;
    uint64_t payload_bytes = 0;
    std::array<uint8_t, 32> section_checksum = {};
    vbr_artifact_byte_source payload;
};

struct vbr_artifact_clean_stash {
    uint64_t valid_rows = 0;
    vbr_repr_domain domain = vbr_repr_domain::full;
    vbr_artifact_layout layout = vbr_artifact_layout::row_major;
    uint64_t row_count = 0;
    uint64_t column_count = 0;
    uint64_t row_bytes = 0;
    vbr_stash_payload_id payload_id;
    std::vector<vbr_artifact_shard_descriptor> shards;
};

struct vbr_artifact_representation {
    vbr_artifact_representation_kind kind = vbr_artifact_representation_kind::raw;
    uint32_t codec_id = 0;
    uint32_t codec_version = 0;
    std::array<uint8_t, 32> reference_digest = {};
    uint32_t source_loss_history = 0;
    uint32_t checkpoint_codec_hops = 0;
};

struct vbr_artifact_unit_descriptor {
    uint32_t child_id = 0;
    uint32_t logical_unit_id = 0;
    vbr_lineage_uuid lineage_uuid;
    uint64_t repr_gen = 0;
    int32_t current_type = -1;
    int32_t last_source_type = -1;
    uint8_t promote_hops = 0;
    vbr_repr_transition last_transition = vbr_repr_transition::initial;
    vbr_artifact_representation representation;
    vbr_artifact_recoverability recoverability =
        vbr_artifact_recoverability::sealed_payload;
    vbr_artifact_side side = vbr_artifact_side::key;
    vbr_artifact_layout layout = vbr_artifact_layout::row_major;
    uint32_t n_stream = 0;
    bool unified = false;
    uint64_t wm_cells = 0;
    uint32_t rank = 0;
    std::array<uint64_t, 4> dimensions = {};
    uint64_t row_alignment = 0;
    uint32_t row_codec_version = 0;
    std::array<uint8_t, 32> codebook_digest = {};
    std::array<uint8_t, 32> rotation_digest = {};
    std::array<uint8_t, 32> meansub_digest = {};
    // Exact calibrated row used when this unit was captured.  The full-table
    // digest above authenticates bytes; this pair prevents a shared-KV layer
    // mapping from silently selecting a different row during mean add-back.
    int32_t meansub_model_id = -1;
    int32_t meansub_layer = -1;
    // Cross-domain reconstruction is intentionally limited to immutable baked
    // tables. Override/off/inactive identities remain valid for exact and
    // same-domain use, but are never executable mean-add evidence.
    bool meansub_baked = false;
    std::vector<vbr_artifact_shard_descriptor> shards;
    vbr_artifact_clean_stash_state clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    vbr_artifact_clean_stash clean_stash;
};

struct vbr_artifact_unit_blob {
    vbr_unit_version_id unit_version_id;
    vbr_payload_digest payload_digest;
    vbr_artifact_unit_descriptor descriptor;
};

struct vbr_artifact_identity_block {
    std::string execution_identity;
    std::string adapter_config_identity;
    std::string media_content_identity;
    uint64_t sequence_epoch = 0;
    int64_t token_count = 0;
    llama_pos next_position = -1;
};

// Restore placement is reference-local: it authenticates which cells one
// reference may install without changing the dense unit blob's content
// address. `shift` is deliberately absent; capture accepts only zero shift.
struct vbr_artifact_cell_placement {
    uint32_t physical_cell = UINT32_MAX;
    llama_pos logical_position = -1;
    llama_pos ext_x = 0;
    llama_pos ext_y = 0;
};

struct vbr_artifact_stream_placement {
    uint32_t child_id = UINT32_MAX;
    uint32_t stream_index = UINT32_MAX;
    llama_seq_id source_sequence = -1;
    llama_pos computation_frontier = -1;
    std::vector<vbr_artifact_cell_placement> cells;
};

struct vbr_artifact_token_block {
    uint32_t codec_version = VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION;
    std::vector<llama_token> tokens;
    vbr_token_block_digest digest;
};

struct vbr_artifact_controller_policy {
    uint32_t child_id = 0;
    checkpoint_child_dependency_mode dependency_mode =
        checkpoint_child_dependency_mode::absent;
    std::array<uint8_t, 32> degrade_order_digest = {};
    std::array<uint8_t, 32> policy_digest = {};
    uint64_t cursor = 0;
    int32_t floor_type = -1;
    uint64_t pressure_independent_settings = 0;
    uint32_t n_stream = 0;
    bool unified = false;
    uint64_t wm_cells = 0;
    std::array<uint8_t, 32> current_type_vector_digest = {};
    bool completed_wave = false;
};

struct vbr_artifact_stash_reference {
    uint64_t valid_rows = 0;
    vbr_repr_domain domain = vbr_repr_domain::full;
    uint64_t row_count = 0;
    uint64_t column_count = 0;
    uint64_t row_bytes = 0;
    uint32_t captured_sink_count = 0;
    std::vector<vbr_generation_page_ref> covered_sink_pages;
    vbr_stash_payload_id payload_id;
};

struct vbr_artifact_unit_reference {
    vbr_lineage_uuid lineage_uuid;
    uint32_t logical_unit_id = 0;
    uint64_t repr_gen = 0;
    vbr_unit_version_id unit_version_id;
    vbr_payload_digest payload_digest;
    std::vector<uint32_t> authorized_stream_refs;
    bool has_stash_reference = false;
    vbr_artifact_stash_reference stash_reference;
};

struct vbr_artifact_companion_payload {
    vbr_artifact_companion_kind kind = vbr_artifact_companion_kind::recurrent;
    uint32_t format_version = 1;
    std::array<uint8_t, 32> build_identity_digest = {};
    vbr_artifact_portable_domain domain;
    vbr_payload_digest payload_digest;
    uint64_t payload_bytes = 0;
    std::array<uint8_t, 32> section_checksum = {};
    vbr_artifact_byte_source payload;
};

struct vbr_artifact_consistency {
    vbr_artifact_consistency_kind kind =
        vbr_artifact_consistency_kind::capture_exact;
    vbr_capture_generation_id source_capture_generation_id;
    vbr_capture_generation_id target_capture_generation_id;
    vbr_transition_lineage_id transition_lineage_id;
};

struct vbr_artifact_reference_manifest {
    uint32_t version = VBR_UNIT_ARTIFACT_FORMAT_VERSION;
    std::array<uint8_t, 32> identity_policy_order_digest = {};
    vbr_artifact_identity_block identity;
    vbr_artifact_token_block token_block;
    vbr_checkpoint_generation_record generation;
    vbr_capture_generation_id capture_generation_id;
    vbr_artifact_consistency consistency;
    std::vector<vbr_artifact_controller_policy> controller_policy;
    std::vector<vbr_artifact_stream_placement> stream_placements;
    std::vector<vbr_artifact_unit_reference> unit_references;
    std::vector<vbr_artifact_companion_payload> companions;
    std::vector<vbr_artifact_portable_accounting_row> accounting;
    vbr_manifest_digest manifest_digest;
};

struct vbr_artifact_package {
    uint32_t version = VBR_UNIT_ARTIFACT_FORMAT_VERSION;
    uint32_t flags = 0;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_artifact_unit_blob> unit_blobs;
    std::vector<vbr_artifact_companion_payload> companions;
    vbr_artifact_reference_manifest manifest;
};

vbr_artifact_status vbr_artifact_prepare(
    vbr_artifact_package & package) noexcept;

// Canonicalizes and validates the non-main-payload portion of a package whose
// unit bytes were authenticated by the projected-capture Merkle authority.
// Unit payload sources are deliberately not read. Companion sources, when
// present, are still hashed and bound here until they gain an equivalent
// opaque projected capability.
vbr_artifact_status vbr_artifact_prepare_projected_metadata(
    vbr_artifact_package & package) noexcept;

// Revalidate an already-prepared immutable package without changing it. This
// This is the import door: it reuses the codec's canonical metadata, placement,
// digest, and payload checks rather than growing a second wire validator.
vbr_artifact_status vbr_artifact_validate_prepared_package(
    const vbr_artifact_package & package) noexcept;

vbr_artifact_status vbr_artifact_encode(
    vbr_artifact_package & package,
    const vbr_artifact_stream_writer & output,
    uint64_t max_total_bytes,
    uint64_t * encoded_size = nullptr) noexcept;

vbr_artifact_status vbr_artifact_decode(
    const vbr_artifact_stream_reader & input,
    uint64_t encoded_size,
    const vbr_artifact_decode_limits & limits,
    const vbr_artifact_payload_consumer * payload_consumer,
    vbr_artifact_package & output) noexcept;

vbr_artifact_status vbr_artifact_encode_vector(
    vbr_artifact_package & package,
    std::vector<uint8_t> & output,
    uint64_t max_total_bytes) noexcept;

vbr_artifact_status vbr_artifact_decode_vector(
    const std::vector<uint8_t> & input,
    const vbr_artifact_decode_limits & limits,
    vbr_artifact_package & output) noexcept;

bool vbr_artifact_validate_portable_accounting(
    const std::vector<vbr_artifact_portable_topology> & topologies,
    const std::vector<vbr_artifact_portable_accounting_row> & rows) noexcept;

// Closed wire-role to C-leaf mapping shared by format validation and the
// transactional catalog. `_count` maps to the category sentinel.
llama_cache_acct_category vbr_artifact_accounting_category(
    vbr_artifact_accounting_role role) noexcept;

// Canonical process-local lineage key for one logical VBR unit. Keeping the
// generation-bearing tuple inside the reviewed artifact-format authority
// prevents catalog clients from becoming raw generation readers.
std::array<uint8_t, 32> vbr_artifact_logical_unit_digest(
    const vbr_artifact_unit_descriptor & descriptor) noexcept;

const char * vbr_artifact_status_name(vbr_artifact_status status) noexcept;
