#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// llama-cache-accounting.h — cache accounting contract, schema version 2.
//
// Policy-free, library-neutral accounting types shared by the server observer,
// lease/lifecycle owners, and artifact transactions. This header is the sole byte
// accounting interface: consumers must not grow private byte counters.
// Presentation (name strings, JSON) lives above, in common/ and server adapters — never here.
//
// The policy-free ledger supports both observation and admission authority.
// Plain `reserve` remains observational. The composed admission path uses
// `reserve_if_serial` before mutation and makes publication wait for stage/commit; without the
// lifecycle flag, existing producers retain the shadow behavior. Every ledger entry point remains
// genuinely non-throwing: allocation/transition/overflow failures latch counters and typed-
// unavailable cells, while the authority composer decides whether those failures refuse a mutation.

constexpr uint32_t LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2;
constexpr uint32_t LLAMA_CACHE_ACCT_TOPOLOGY_VERSION        = 1;
constexpr uint32_t LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR = 1000000;

// Mutually-exclusive semantic LEAF categories. host_cache / snapshot_blob are DERIVED
// provider/artifact groupings over these leaves, never additive leaves themselves: a
// host-cache entry's aggregate size already contains its state data and checkpoint payloads
// (server_prompt_cache_state::size()), so summing aggregate and parts would double-count.
enum class llama_cache_acct_category : uint8_t {
    live_attention_state = 0,
    live_recurrent_state,
    recurrent_rollback_planes,
    full_snapshot_payload,          // serialized main/draft state data of a host-cache entry
    checkpoint_state_payload,       // context-checkpoint target+draft payload bytes
    typed_accelerator_payload,
    checkpoint_generation_page_metadata,
    checkpoint_generation_unit_metadata,
    live_generation_metadata,
    ownership_index_metadata,
    unit_version_payload,
    clean_stash_payload,
    artifact_descriptor_metadata,
    artifact_reference_metadata,
    transfer_staging,
    codec_workspace,
    pinned_preimage_ring,
    // The parked rolling-window tape: present and typed so its zero is OBSERVED, not implied.
    // This names the parked mechanism specifically — the live fixed speculative tape is NOT
    // this category and is never asserted zero through it.
    rolling_window_tape,
    container_overhead,             // only where overhead cannot be attributed more precisely
    _count,
};

enum class llama_cache_acct_residency : uint8_t {
    device = 0,
    pinned_host,
    pageable_host,
    disk,
    remote,
    not_applicable,
    _count,
};

// Resource domains are closed accounting identities, not presentation labels. Device rows
// are qualified by one stable ordinal in one versioned, ordered shard-topology descriptor.
// Non-device rows carry an EXPLICIT not_applicable tag; they never fabricate ordinal zero
// or an empty topology that a v1 reader could mistake for a device.
enum class llama_cache_acct_domain_kind : uint8_t {
    not_applicable = 0,
    device_topology,
    _count,
};

struct llama_cache_acct_device_ordinal {
    uint16_t v = UINT16_MAX;
    explicit operator bool() const { return v != UINT16_MAX; }
};

struct llama_cache_acct_topology_id {
    uint32_t v = 0;
    explicit operator bool() const { return v != 0; }
};

struct llama_cache_acct_device_digest_tag;
struct llama_cache_acct_topology_digest_tag;

// A tag is part of the digest's TYPE, not merely its field name. The byte array is private
// so `topology_digest.bytes = device_digest.bytes` cannot bypass that distinction.
template <typename Tag>
class llama_cache_acct_digest {
public:
    llama_cache_acct_digest() = default;

    static llama_cache_acct_digest from_sha256(std::array<uint8_t, 32> bytes) {
        llama_cache_acct_digest out;
        out.bytes_ = std::move(bytes);
        return out;
    }

    const std::array<uint8_t, 32> & bytes() const { return bytes_; }

    bool valid() const noexcept {
        return std::any_of(bytes_.begin(), bytes_.end(), [](uint8_t value) {
            return value != 0;
        });
    }

    friend bool operator==(const llama_cache_acct_digest & a,
                           const llama_cache_acct_digest & b) {
        return a.bytes_ == b.bytes_;
    }
    friend bool operator!=(const llama_cache_acct_digest & a,
                           const llama_cache_acct_digest & b) {
        return !(a == b);
    }

private:
    std::array<uint8_t, 32> bytes_ = {};
};

using llama_cache_acct_device_digest =
    llama_cache_acct_digest<llama_cache_acct_device_digest_tag>;
using llama_cache_acct_topology_digest =
    llama_cache_acct_digest<llama_cache_acct_topology_digest_tag>;

inline bool operator==(llama_cache_acct_device_ordinal a, llama_cache_acct_device_ordinal b) {
    return a.v == b.v;
}
inline bool operator!=(llama_cache_acct_device_ordinal a, llama_cache_acct_device_ordinal b) {
    return !(a == b);
}
inline bool operator==(llama_cache_acct_topology_id a, llama_cache_acct_topology_id b) {
    return a.v == b.v;
}
inline bool operator!=(llama_cache_acct_topology_id a, llama_cache_acct_topology_id b) {
    return !(a == b);
}

struct llama_cache_acct_shard_topology {
    uint32_t version      = 0;
    uint16_t device_count = 0;
    int16_t  split_mode   = 0;
    llama_cache_acct_device_ordinal main_device;
    // The vectors have EXACTLY device_count entries. Their length is bounded by the
    // runtime's llama_max_devices(), not a second private constant in this contract.
    std::vector<llama_cache_acct_device_digest> device_identities;
    std::vector<uint32_t> shard_weights;
    llama_cache_acct_topology_digest digest;
};

// The ONE topology-construction door. It hashes the ordered device descriptions, normalizes
// explicit weights to exact millionths, and computes the canonical descriptor digest.
// Absent/all-zero weights are valid only for a one-device topology; multi-device auto-split
// must supply its resolved weights rather than fabricate equal placement.
bool llama_cache_acct_build_shard_topology(
        const std::vector<std::string> & ordered_device_identities,
        int16_t split_mode,
        int32_t main_device,
        const float * shard_weights,
        llama_cache_acct_shard_topology & out) noexcept;

// Canonical digest door for a normalized topology. Portable artifact readers use this
// to verify that an imported descriptor will intern to the same topology identity.
llama_cache_acct_topology_digest llama_cache_acct_compute_topology_digest(
        const llama_cache_acct_shard_topology & topology) noexcept;

inline bool operator==(const llama_cache_acct_shard_topology & a,
                       const llama_cache_acct_shard_topology & b) {
    return a.version           == b.version &&
           a.device_count      == b.device_count &&
           a.split_mode        == b.split_mode &&
           a.main_device       == b.main_device &&
           a.device_identities == b.device_identities &&
           a.shard_weights     == b.shard_weights &&
           a.digest            == b.digest;
}
inline bool operator!=(const llama_cache_acct_shard_topology & a,
                       const llama_cache_acct_shard_topology & b) {
    return !(a == b);
}

struct llama_cache_acct_resource_domain {
    llama_cache_acct_residency residency = llama_cache_acct_residency::not_applicable;
    llama_cache_acct_domain_kind kind = llama_cache_acct_domain_kind::not_applicable;
    llama_cache_acct_device_ordinal device_ordinal;
    // Ledger-local, immutable, non-reused citation into snapshot.topologies. Zero is valid
    // only for non-device domains.
    llama_cache_acct_topology_id topology;

    static llama_cache_acct_resource_domain non_device(llama_cache_acct_residency residency);
};

inline bool operator==(const llama_cache_acct_resource_domain & a,
                       const llama_cache_acct_resource_domain & b) {
    return a.residency      == b.residency &&
           a.kind           == b.kind &&
           a.device_ordinal == b.device_ordinal &&
           a.topology       == b.topology;
}
inline bool operator!=(const llama_cache_acct_resource_domain & a,
                       const llama_cache_acct_resource_domain & b) {
    return !(a == b);
}
inline bool llama_cache_acct_resource_domain_less(
        const llama_cache_acct_resource_domain & a,
        const llama_cache_acct_resource_domain & b) {
    if (a.residency != b.residency) {
        return a.residency < b.residency;
    }
    if (a.kind != b.kind) {
        return a.kind < b.kind;
    }
    if (a.topology != b.topology) {
        return a.topology.v < b.topology.v;
    }
    return a.device_ordinal.v < b.device_ordinal.v;
}
static_assert(sizeof(llama_cache_acct_resource_domain) <= 12,
              "resource domains must remain interned keys, not inline topology descriptors");

// Structural validation only. A ledger additionally verifies that topology is an interned
// registry key and that the ordinal is in that descriptor's range.
bool llama_cache_acct_resource_domain_valid(const llama_cache_acct_resource_domain & domain);

// Four measures per (category, resource-domain) cell. An observation that cannot be made stays a
// typed unknown/unavailable — zero always means a measured zero. `transient_peak` is the
// high-water mark of CONCURRENTLY staged bytes for the cell, not the largest single stage.
enum class llama_cache_acct_measure : uint8_t {
    logical_payload = 0,
    resident_allocated,
    reserved,
    transient_peak,
    _count,
};

enum class llama_cache_acct_known : uint8_t {
    known = 0,
    unknown,        // not yet observed / producer absent
    unavailable,    // observation attempted and failed (fault, overflow)
    _count,
};

// Closed producer identity for completeness certification. The configuration owner declares
// the required (domain, producer) manifest; producers may only transition a declared row.
// An absent producer therefore cannot make itself optional by omission.
enum class llama_cache_acct_producer : uint8_t {
    observer_init = 0,
    host_cache,
    // Reserved in the refrozen v2 vocabulary for the immediately following producers;
    // they are not required until their configuration paths are wired.
    live_memory,
    retention_sidecar,
    _count,
};

struct llama_cache_acct_value {
    uint64_t               value = 0;
    llama_cache_acct_known state = llama_cache_acct_known::unknown;

    static llama_cache_acct_value measured(uint64_t v) {
        return { v, llama_cache_acct_known::known };
    }
};

// Raw work quantities are never silently converted between units; a cost term carries its
// raw quantity for auditability plus an optional comparable time estimate.
enum class llama_cache_acct_unit : uint8_t {
    bytes = 0,
    tokens,
    operations,
    _count,
};

enum class llama_cache_acct_cost_kind : uint8_t {
    restore = 0,
    replay,
    transfer,
    eviction,
    workspace,
    _count,
};

// Canonical raw unit per cost kind: replay is counted in tokens, everything
// else in bytes. The unit is schema metadata, not a measurement — it is valid even while the
// term itself is unavailable.
constexpr llama_cache_acct_unit llama_cache_acct_cost_kind_unit(llama_cache_acct_cost_kind k) {
    return k == llama_cache_acct_cost_kind::replay ? llama_cache_acct_unit::tokens
                                                   : llama_cache_acct_unit::bytes;
}

// §7.5 cost-term shape consumed by the B planner. `estimated_us` is a versioned estimate;
// measured time is a separate actual-outcome field on the consumer's record and is never
// substituted into the estimate. `estimator_version` is meaningful only while `estimated_us`
// is known.
struct llama_cache_acct_cost_term {
    llama_cache_acct_cost_kind kind = llama_cache_acct_cost_kind::restore;
    llama_cache_acct_value     raw;
    llama_cache_acct_unit      raw_unit = llama_cache_acct_unit::bytes;
    llama_cache_acct_value     estimated_us;    // unknown until B lands an estimator
    uint32_t                   estimator_version = 0;
};

// Attribution axes: a closed kind tag; server-wide rows use the
// defaults. The tenant axis is deliberately ABSENT from this schema version, not an empty
// field: this schema has no tenant identity (adding one is a schema-version bump).
enum class llama_cache_acct_attr_kind : uint8_t {
    server = 0,
    slot,
    artifact,
    _count,
};

// Identity discipline (C/F freeze requirement 3): every DISTINCT identity must never be
// interchanged. The operation and allocation ids are process-local accounting identities;
// the artifact identity, content digest, and eligibility lineage identity are opaque
// contract fields carried and retained by the transaction (F populates and validates them).
// Distinct wrapper types make interchange a compile error (matrix asserted below). The
// zero id is the "none" sentinel (vbr_operation_id idiom): `explicit operator bool` tests
// it; only op/alloc ids get std::hash — they alone key the ledger maps.
struct llama_cache_acct_op_id {
    uint64_t v = 0;
    explicit operator bool() const { return v != 0; }
};
struct llama_cache_acct_alloc_id {
    uint64_t v = 0;
    explicit operator bool() const { return v != 0; }
};
struct llama_cache_acct_artifact_id    { uint64_t v = 0; };
struct llama_cache_acct_content_digest { uint64_t v = 0; };
struct llama_cache_acct_lineage_id     { uint64_t v = 0; };

inline bool operator==(llama_cache_acct_op_id          a, llama_cache_acct_op_id          b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_alloc_id       a, llama_cache_acct_alloc_id       b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_artifact_id    a, llama_cache_acct_artifact_id    b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_content_digest a, llama_cache_acct_content_digest b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_lineage_id     a, llama_cache_acct_lineage_id     b) { return a.v == b.v; }
inline bool operator< (llama_cache_acct_op_id          a, llama_cache_acct_op_id          b) { return a.v < b.v; }
inline bool operator< (llama_cache_acct_artifact_id    a, llama_cache_acct_artifact_id    b) { return a.v < b.v; }
inline bool operator!=(llama_cache_acct_op_id          a, llama_cache_acct_op_id          b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_alloc_id       a, llama_cache_acct_alloc_id       b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_artifact_id    a, llama_cache_acct_artifact_id    b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_content_digest a, llama_cache_acct_content_digest b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_lineage_id     a, llama_cache_acct_lineage_id     b) { return !(a == b); }

template <> struct std::hash<llama_cache_acct_op_id> {
    size_t operator()(const llama_cache_acct_op_id & id) const { return std::hash<uint64_t>{}(id.v); }
};
template <> struct std::hash<llama_cache_acct_alloc_id> {
    size_t operator()(const llama_cache_acct_alloc_id & id) const { return std::hash<uint64_t>{}(id.v); }
};

// The non-interchange proof, in the header so every consumer TU enforces it. Aggregate
// `{n}` init stays legal — mints construct ids on purpose; only IMPLICIT interchange is
// banned.
template <typename A, typename B>
constexpr bool llama_cache_acct_ids_distinct =
    !std::is_convertible_v<A, B> && !std::is_convertible_v<B, A>;

template <typename A, typename... Rest>
constexpr bool llama_cache_acct_distinct_from_all =
    (llama_cache_acct_ids_distinct<A, Rest> && ...);

template <typename... Ts>
struct llama_cache_acct_all_ids_distinct;

template <>
struct llama_cache_acct_all_ids_distinct<> : std::true_type {};

template <typename A, typename... Rest>
struct llama_cache_acct_all_ids_distinct<A, Rest...>
    : std::bool_constant<llama_cache_acct_distinct_from_all<A, Rest...> &&
                         llama_cache_acct_all_ids_distinct<Rest...>::value> {};

// Full compile-negative matrix: every strong identity is distinct from every other strong
// identity, and each is separately non-convertible to all three raw integer widths.
static_assert(llama_cache_acct_all_ids_distinct<
        llama_cache_acct_op_id,
        llama_cache_acct_alloc_id,
        llama_cache_acct_artifact_id,
        llama_cache_acct_content_digest,
        llama_cache_acct_lineage_id,
        llama_cache_acct_device_ordinal,
        llama_cache_acct_topology_id,
        llama_cache_acct_device_digest,
        llama_cache_acct_topology_digest>::value);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_op_id,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_alloc_id,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_artifact_id,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_content_digest,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_lineage_id,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_device_ordinal,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_topology_id,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_device_digest,
        uint16_t, uint32_t, uint64_t>);
static_assert(llama_cache_acct_distinct_from_all<llama_cache_acct_topology_digest,
        uint16_t, uint32_t, uint64_t>);

struct llama_cache_acct_attribution {
    llama_cache_acct_attr_kind   kind    = llama_cache_acct_attr_kind::server;
    int32_t                      slot_id = -1;   // meaningful when kind == slot
    llama_cache_acct_artifact_id artifact;       // meaningful when kind == artifact
};

enum class llama_cache_acct_txn_state : uint8_t {
    reserved = 0,
    staged,
    committed,
    aborted,
    released,
    _count,
};

// Point-in-time gauge cell (durable byte state). Counters below are monotone; every field is
// one or the other, never both.
struct llama_cache_acct_cell {
    std::array<llama_cache_acct_value, size_t(llama_cache_acct_measure::_count)> measures;
};

struct llama_cache_acct_cell_row {
    llama_cache_acct_category        category = llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    // Snapshot-boundary join of every required producer for this domain. Measures are
    // projected unavailable unless this is known.
    llama_cache_acct_known           certification = llama_cache_acct_known::unknown;
    llama_cache_acct_cell            cell;

private:
    // Canonical ledger-local concurrent staging state for this exact aggregate row. It is
    // deliberately not a public snapshot measure; only transient_peak is observable.
    uint64_t staged = 0;
    friend struct llama_cache_acct_ledger;
};

struct llama_cache_acct_topology_row {
    llama_cache_acct_topology_id    id;
    llama_cache_acct_shard_topology topology;
};

struct llama_cache_acct_completeness_requirement {
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_producer        producer = llama_cache_acct_producer::observer_init;
};

struct llama_cache_acct_completeness_row {
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_producer        producer = llama_cache_acct_producer::observer_init;
    llama_cache_acct_known           state    = llama_cache_acct_known::unknown;
};

// Normalized attributed row: one per live committed physical allocation. Server aggregates
// live in `cells`; slot/artifact attribution is read from these rows (an explicit normalized
// form — no private per-consumer counters).
struct llama_cache_acct_allocation_row {
    llama_cache_acct_alloc_id      alloc;
    llama_cache_acct_attribution   attribution;
    llama_cache_acct_category      category  = llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_known         certification = llama_cache_acct_known::unknown;
    uint64_t                       logical_bytes  = 0;
    uint64_t                       resident_bytes = 0;
    uint32_t                       committed_refs = 0;
    llama_cache_acct_artifact_id    artifact_identity;
    llama_cache_acct_content_digest content_digest;
    llama_cache_acct_lineage_id     lineage_identity;
};

struct llama_cache_acct_snapshot {
    uint32_t               schema_version = LLAMA_CACHE_ACCT_SCHEMA_VERSION;
    uint64_t               serial         = 0;    // bumped on EVERY observable change, faults included
    // Authoritative schema-v2 aggregates and certification. Completeness is NEVER a
    // server-wide scalar: each required (resource-domain, producer) pair has its own row.
    llama_cache_acct_known completeness_manifest = llama_cache_acct_known::unknown;
    // Topologies are interned once per ledger and emitted once per record. Every device
    // domain in cells/completeness/allocations cites one row by id.
    std::vector<llama_cache_acct_topology_row>     topologies;
    std::vector<llama_cache_acct_cell_row>         cells;
    std::vector<llama_cache_acct_completeness_row> completeness;
    std::vector<llama_cache_acct_allocation_row> allocations;
    // in-flight transaction count (reserved + staged + committed-unreleased): zero after a
    // producer's entries are fully destroyed — a leaked op is an accounting bug
    uint64_t live_ops = 0;
    // monotone fault counters
    uint64_t faults_invalid_transition = 0;
    uint64_t faults_overflow           = 0;
    uint64_t faults_unknown_id         = 0;
    uint64_t faults_allocation         = 0;   // internal ledger allocation failure (non-throwing contract)
};

// Explicit compatibility projection for schema-v1 readers. Device-topology rows cannot be
// represented and therefore make the projection fail closed (`false`, completeness
// unavailable). Callers must opt into this adapter; schema-v2 never silently flattens a
// device ordinal into the old residency-only cells.
struct llama_cache_acct_allocation_row_v1 {
    llama_cache_acct_alloc_id      alloc;
    llama_cache_acct_attribution   attribution;
    llama_cache_acct_category      category  = llama_cache_acct_category::container_overhead;
    llama_cache_acct_residency     residency = llama_cache_acct_residency::not_applicable;
    uint64_t                       logical_bytes  = 0;
    uint64_t                       resident_bytes = 0;
    uint32_t                       committed_refs = 0;
    llama_cache_acct_artifact_id    artifact_identity;
    llama_cache_acct_content_digest content_digest;
    llama_cache_acct_lineage_id     lineage_identity;
};

struct llama_cache_acct_snapshot_v1 {
    uint32_t schema_version = 1;
    uint64_t serial = 0;
    llama_cache_acct_known completeness = llama_cache_acct_known::unknown;
    std::array<std::array<llama_cache_acct_cell,
                          size_t(llama_cache_acct_residency::_count)>,
               size_t(llama_cache_acct_category::_count)> cells;
    std::vector<llama_cache_acct_allocation_row_v1> allocations;
    uint64_t live_ops = 0;
    uint64_t faults_invalid_transition = 0;
    uint64_t faults_overflow           = 0;
    uint64_t faults_unknown_id         = 0;
    uint64_t faults_allocation         = 0;
};

bool llama_cache_acct_snapshot_to_v1(
        const llama_cache_acct_snapshot & source,
        llama_cache_acct_snapshot_v1 & destination) noexcept;

// Non-mutating, last-reference-aware release preview used by shadow destruction quoting.
// Values are exact deltas the ledger's current release(op) would apply. A shared allocation
// therefore reports measured zero until its last committed reference. False means the op
// cannot be previewed as a live committed reference; no ledger state or fault counter changes.
struct llama_cache_acct_release_preview {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_value logical_payload;
    llama_cache_acct_value resident_allocated;
};

// Exact, non-mutating preview of releasing a SET of operation references together. Rows
// are aggregated by resource domain because the budget prices one release entry per domain.
// Unlike summing llama_cache_acct_release_preview values, this preserves last-reference
// semantics when several selected operations jointly own one shared allocation.
struct llama_cache_acct_release_set_row {
    llama_cache_acct_resource_domain domain;
    uint64_t logical_payload   = 0;
    uint64_t resident_allocated = 0;
};

struct llama_cache_acct_release_set_yield_row {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    uint64_t logical_payload = 0;
    uint64_t resident_allocated = 0;
};

struct llama_cache_acct_release_set_view {
    const llama_cache_acct_op_id * data = nullptr;
    size_t size = 0;
};

struct llama_cache_acct_release_set_preview {
    uint64_t accounting_serial = 0;
    std::vector<llama_cache_acct_release_set_row> rows;
    // Same exact last-reference union, additionally split by category for
    // bounded lifecycle evidence. Capacity planning consumes the canonical
    // domain-only rows above.
    std::vector<llama_cache_acct_release_set_yield_row> yield_rows;
};

// Conditional-reserve outcome. A bare op id cannot separate expected optimistic-concurrency
// drift (retry) from a hard ledger fault (refuse); the admission caller needs that distinction
// before the admission composer acts, so reserve_if_serial() returns it typed. serial_conflict mints no op, mutates
// nothing, and is NOT a ledger fault; ledger_fault collapses the reserve()-class hard failures.
enum class llama_cache_conditional_reserve_status : uint8_t {
    admitted,
    serial_conflict,
    ledger_fault,
    _count,
};

struct llama_cache_conditional_reserve_result {
    llama_cache_conditional_reserve_status status = llama_cache_conditional_reserve_status::ledger_fault;
    llama_cache_acct_op_id                  op     = {};
};

struct llama_cache_conditional_reserve_request {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution attribution;
    uint64_t expected_logical = 0;
    uint64_t expected_resident = 0;
};

struct llama_cache_conditional_reserve_set_result {
    llama_cache_conditional_reserve_status status =
        llama_cache_conditional_reserve_status::ledger_fault;
    size_t failed_request = SIZE_MAX;
};

// Mutation-boundary release outcome. serial_conflict is optimistic concurrency
// drift and mutates nothing; ledger_fault means at least one operation was no
// longer a live committed reference (also no partial release). `released`
// applies the complete set under one lock and advances the serial once.
enum class llama_cache_conditional_release_status : uint8_t {
    released,
    serial_conflict,
    ledger_fault,
    _count,
};

// Accounting ledger: reserve → stage → commit | abort → release. Charge-once for shared
// immutable allocations: the durable bytes of a
// physical allocation are charged when its FIRST reference commits and discharged when its
// LAST reference releases; per-reference metadata is reported by the referrer under its own
// leaf (artifact_reference_metadata), outside this refcount. Allocation ids must come from
// new_alloc() (zero and unminted ids are faults) and an allocation's (category,
// resource-domain, resident-size) tuple is immutable — a domain/topology mismatch is a
// fault, never a silent merge. NON-THROWING: no method throws; internal failure latches
// faults_allocation.
struct llama_cache_acct_ledger {
    llama_cache_acct_ledger();

    // Mint a fresh physical-allocation id (one owner for the whole accounting id space).
    llama_cache_acct_alloc_id new_alloc();

    // Canonical topology registration. The descriptor must come from
    // llama_cache_acct_build_shard_topology(); duplicate descriptors reuse the existing
    // immutable id. The returned domain is ledger-local and safe to place in citations.
    bool make_device_domain(
            const llama_cache_acct_shard_topology & topology,
            llama_cache_acct_device_ordinal ordinal,
            llama_cache_acct_resource_domain & out);

    // Observational reservation: records the expected resident bytes under `reserved`,
    // returns the op id (zero id on internal failure). Never blocks or admits anything.
    llama_cache_acct_op_id reserve(
            llama_cache_acct_category      category,
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_attribution   attribution,
            uint64_t                       expected_logical,
            uint64_t                       expected_resident);

    // Admission-gating reservation: reserve() only if the ledger is still at expected_serial
    // (the serial the caller's coordinator snapshot was priced against), closing the
    // snapshot→fits→reserve TOCTOU. Policy-free: it never consults capacity — that is the
    // coordinator's job before this call. On drift the ledger is untouched (no op, no cell change,
    // no serial bump, serial_conflicts++), so the caller re-snapshots and retries. This helper
    // mutates no cache or storage state; the caller owns those mutations after admission.
    llama_cache_conditional_reserve_result reserve_if_serial(
            uint64_t                       expected_serial,
            llama_cache_acct_category      category,
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_attribution   attribution,
            uint64_t                       expected_logical,
            uint64_t                       expected_resident);

    // Atomic multi-domain form used by the shared authority transaction. The
    // caller owns fixed-size request/output arenas; one ledger lock either
    // mints every reservation or leaves no reservation live.
    llama_cache_conditional_reserve_set_result reserve_set_if_serial(
            uint64_t expected_serial,
            const llama_cache_conditional_reserve_request * requests,
            size_t n_requests,
            llama_cache_acct_op_id * output_ops) noexcept;

    // Process-local count of reserve_if_serial() drift refusals. Deliberately OUTSIDE the
    // serialized snapshot (adding it there would bump accounting schema 2) — it is authority
    // telemetry, not versioned accounting state.
    uint64_t serial_conflicts() const noexcept;

    // Current optimistic-concurrency fence without materializing the
    // allocation/certification rows carried by snapshot().
    uint64_t serial() const noexcept;

    // Process-local registry size for lifecycle diagnostics/tests. This is deliberately absent
    // from the schema-v2 snapshot: allocation-map storage is an implementation detail, not a
    // serialized accounting measure.
    size_t allocation_registry_size() const noexcept;

    // Associate the op with a minted physical allocation and its actual resident size.
    // Validates the allocation tuple against any existing citation. Updates the concurrent
    // staged high-water mark. The three opaque identities are retained on the transaction
    // (artifact transactions populate them; empty is valid for observation). False on any fault.
    bool stage(llama_cache_acct_op_id op, llama_cache_acct_alloc_id alloc,
               uint64_t resident_bytes,
               llama_cache_acct_artifact_id    artifact = {},
               llama_cache_acct_content_digest digest   = {},
               llama_cache_acct_lineage_id     lineage  = {});

    // Record the shipped publication boundary. First committed reference of an allocation
    // charges its durable bytes; later references must cite the same logical size and only
    // join the refcount.
    bool commit(llama_cache_acct_op_id op, uint64_t logical_bytes);

    // Zero durable gauge delta; the observed transient peak is retained. The op is erased.
    bool abort(llama_cache_acct_op_id op);

    // Validate and abort a complete reservation/staging set under one lock.
    // No partial mutation is visible on an invalid member.
    bool abort_set(
            const llama_cache_acct_op_id * selected,
            size_t n_selected) noexcept;

    // Downward-only repricing for equal logical/resident staging claims. This
    // preserves the original serially admitted claims and cannot consume new
    // capacity. Every selected op must still be reserved.
    bool shrink_reservation_set(
            const llama_cache_acct_op_id * selected,
            const uint64_t * resident_bytes,
            size_t n_selected) noexcept;

    // Atomically replace one complete reserved set with a differently
    // partitioned reserved set. For every category/domain pair, replacement
    // resident bytes must be no greater than the selected aggregate. This is
    // the split-phase content-addressing seam: a conservative pre-materialize
    // fence can become exact fresh/deduplicated publication leaves without
    // releasing capacity or taking a second budget sample.
    bool repartition_reservation_set_downward(
            const llama_cache_acct_op_id * selected,
            size_t n_selected,
            const llama_cache_conditional_reserve_request * replacements,
            size_t n_replacements,
            llama_cache_acct_op_id * output_ops) noexcept;

    // Drop the op's reference; discharges durable bytes when the allocation loses its last
    // reference. Exactly-once per reference — a second release is a fault.
    bool release(llama_cache_acct_op_id op);

    bool preview_release(
            llama_cache_acct_op_id op,
            llama_cache_acct_release_preview & out) const noexcept;
    bool preview_release_set(
            const std::vector<llama_cache_acct_op_id> & ops,
            uint64_t expected_serial,
            llama_cache_acct_release_set_preview & out,
            bool include_category_yields = false) const noexcept;

    // Exact last-reference resident deltas for many independent candidate
    // sets under one ledger lock. The views need only remain valid for this
    // call. Output order matches input order; any malformed set or serial
    // drift fails the complete batch and clears `out`.
    bool preview_release_set_resident_batch(
            const std::vector<llama_cache_acct_release_set_view> & sets,
            uint64_t expected_serial,
            std::vector<uint64_t> & out) const noexcept;

    // Exact candidate marginals after one common baseline set has already
    // been removed. The baseline is resolved once; every candidate is then
    // evaluated in O(candidate ops) under the same ledger lock. Output is the
    // additional resident yield beyond the baseline, in candidate order.
    bool preview_release_set_resident_conditioned_batch(
            llama_cache_acct_release_set_view baseline,
            const std::vector<llama_cache_acct_release_set_view> & candidates,
            uint64_t expected_serial,
            std::vector<uint64_t> & out) const noexcept;

    // Commit a previously previewed canonical operation set atomically. The
    // caller must pass strictly increasing, nonzero operation ids; this keeps
    // the post-mutation terminal allocation-free and prevents duplicate refs.
    llama_cache_conditional_release_status release_set_if_serial(
            const std::vector<llama_cache_acct_op_id> & ops,
            uint64_t expected_serial) noexcept;
    // Exact unconditional cleanup terminal for an owner that has already
    // disappeared. The canonical op set must be strictly increasing. It
    // validates the complete union and releases it under one ledger lock.
    llama_cache_conditional_release_status release_set_current(
            const std::vector<llama_cache_acct_op_id> & ops) noexcept;

    // Direct gauge reporting for non-transactional producers (live state, metadata gauges).
    // Checked: overflow latches the cell unavailable and counts a fault.
    void gauge_set(llama_cache_acct_category category,
                   const llama_cache_acct_resource_domain & domain,
                   llama_cache_acct_measure measure,
                   uint64_t value);

    // Initialize an unobserved gauge to known zero without overwriting a
    // value already published by another authority sharing this ledger cell.
    // Returns false for an unavailable or invalid cell.
    bool gauge_initialize_zero(
            llama_cache_acct_category category,
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_measure measure);

    // A producer whose own observation failed (e.g. checked-sum overflow) latches the cell
    // unavailable instead of reporting a fabricated value.
    void mark_unavailable(llama_cache_acct_category category,
                          const llama_cache_acct_resource_domain & domain,
                          llama_cache_acct_measure measure);

    // Replace the REQUIRED completeness manifest atomically. This is configuration-owned:
    // producer code cannot add/remove its own requirement. Duplicate or malformed entries
    // fault and leave the previous manifest unchanged.
    bool configure_required_producers(
            const llama_cache_acct_completeness_requirement * requirements,
            size_t n_requirements);

    // Producer transitions for a manifest row. Known and unavailable are monotone for one
    // schema lifetime; an undeclared pair is an invalid-transition fault.
    bool certify_complete(
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_producer producer);
    void mark_producer_unavailable(
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_producer producer);

    // Coherent copy of the observable state under one serial (gauges + normalized
    // allocation rows + fault counters). On internal copy failure the returned snapshot has
    // every completeness row unavailable and no aggregate/allocation rows.
    llama_cache_acct_snapshot snapshot();

private:
    struct txn {
        llama_cache_acct_txn_state   state = llama_cache_acct_txn_state::reserved;
        llama_cache_acct_category    category  = llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;
        llama_cache_acct_attribution attribution;
        llama_cache_acct_alloc_id    alloc;
        uint64_t                     reserved_bytes = 0; // charged at reserve, unwound by commit/abort
        uint64_t                     resident_bytes = 0; // actual, set at stage
        llama_cache_acct_artifact_id    artifact;
        llama_cache_acct_content_digest digest;
        llama_cache_acct_lineage_id     lineage;
    };

    // Allocation lifecycle: MINTED (registry entry created by new_alloc) → LIVE (first stage
    // fixes the complete immutable citation tuple) → erased when the last staged/committed
    // reference leaves. Allocation ids are monotone and never reused, so a stale id remains
    // fail-closed as unknown without retaining an unbounded tombstone map.
    struct alloc_entry {
        bool     tuple_set      = false;
        uint32_t staged_refs    = 0;
        uint32_t committed_refs = 0;
        // immutable citation tuple, fixed by the first stage — ALL fields compared on every
        // shared citation (identity fields included)
        llama_cache_acct_category    category  = llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;
        uint64_t                     resident_bytes = 0;
        uint64_t                     charged_logical = 0; // set by the first commit
        llama_cache_acct_attribution attribution;         // first committer's
        llama_cache_acct_artifact_id    artifact;
        llama_cache_acct_content_digest digest;
        llama_cache_acct_lineage_id     lineage;
    };

    enum class release_resolution_status : uint8_t {
        ok = 0,
        unknown_op,
        invalid_state,
        unknown_allocation,
    };

    struct release_resolution {
        const txn * operation = nullptr;
        const alloc_entry * allocation = nullptr;
    };

    // Unlocked common policy for release() and preview_release(): resolve one live committed
    // reference and its allocation. The mutating caller applies fault/serial effects; preview
    // remains a neutral query.
    release_resolution_status resolve_release_locked(
            llama_cache_acct_op_id op,
            release_resolution & out) const noexcept;
    // Apply one release already validated by resolve_release_locked while the
    // caller retains the ledger lock. This has no failure arm: set release
    // validates the complete batch before invoking it, structurally excluding
    // a partial-release error return.
    void apply_release_locked(llama_cache_acct_op_id op) noexcept;
    llama_cache_conditional_release_status release_set_locked(
            const std::vector<llama_cache_acct_op_id> & ops) noexcept;

    // Unlocked helpers (callers hold the mutex). Aggregate rows are pre-created atomically
    // with the required-producer manifest, so the gauge and failure paths never allocate.
    llama_cache_acct_cell_row * find_cell(
            llama_cache_acct_category c,
            const llama_cache_acct_resource_domain & domain);
    llama_cache_acct_completeness_row * find_completeness(
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_producer producer);
    // unlocked internal latch (callers hold the mutex)
    void cell_latch_unavailable(llama_cache_acct_category c,
                                const llama_cache_acct_resource_domain & domain,
                                llama_cache_acct_measure m);
    // checked-add/sub on a cell measure; latches unavailable + fault on overflow/underflow
    void cell_add(llama_cache_acct_category c, const llama_cache_acct_resource_domain & domain,
                  llama_cache_acct_measure m, uint64_t v);
    void cell_sub(llama_cache_acct_category c, const llama_cache_acct_resource_domain & domain,
                  llama_cache_acct_measure m, uint64_t v);
    // concurrent-staged tracking: row.staged +=/-= v, peak = max(peak, row.staged)
    void staged_add(llama_cache_acct_category c, const llama_cache_acct_resource_domain & domain,
                    uint64_t v);
    void staged_sub(llama_cache_acct_category c, const llama_cache_acct_resource_domain & domain,
                    uint64_t v);
    void bump_serial();
    const llama_cache_acct_shard_topology * find_topology(
            llama_cache_acct_topology_id id) const;
    bool domain_registered(const llama_cache_acct_resource_domain & domain) const;
    bool domain_manifested(const llama_cache_acct_resource_domain & domain) const;
    bool domain_use_valid(llama_cache_acct_category category,
                          const llama_cache_acct_resource_domain & domain) const;
    llama_cache_acct_known domain_certification(
            const llama_cache_acct_resource_domain & domain) const;
    // Retirement accounts for BOTH claim kinds. Once the last staged and committed reference
    // leaves, erase its registry slot. Callers remove the terminal operation first, so no live
    // op can cite an erased allocation.
    void maybe_retire(llama_cache_acct_alloc_id alloc);

    // Unlocked reserve body shared by reserve() and reserve_if_serial() (callers hold the mutex).
    // Returns the minted op, or {} after latching the appropriate fault + bumping serial.
    llama_cache_acct_op_id reserve_locked(
            llama_cache_acct_category      category,
            const llama_cache_acct_resource_domain & domain,
            llama_cache_acct_attribution   attribution,
            uint64_t                       expected_logical,
            uint64_t                       expected_resident);

    mutable std::mutex mtx;
    llama_cache_acct_snapshot state;    // durable gauges + serial + faults live here (rows built on demand)
    uint64_t                  serial_conflicts_ = 0; // NOT in `state`: authority telemetry, unversioned
    llama_cache_acct_op_id    next_op       = {1};
    llama_cache_acct_alloc_id next_alloc_id = {1};
    std::unordered_map<llama_cache_acct_op_id, txn>            ops;
    std::unordered_map<llama_cache_acct_alloc_id, alloc_entry> allocs;
};
