#include "llama-cache-accounting.h"
#include "llama-sha256.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

// Shadow accounting ledger (C schema v2). Every entry point is fault-tolerant and
// observational: an invalid transition, tuple mismatch, overflow, or allocation failure
// increments a counter and cannot influence the shipped mutation it observes.

static bool acct_digest_nonzero(const std::array<uint8_t, 32> & bytes) {
    return std::any_of(bytes.begin(), bytes.end(), [](uint8_t v) { return v != 0; });
}

template <typename Digest>
static Digest acct_sha256(const void * data, size_t size) {
    llama_sha256_writer writer;
    writer.string(data, size);
    return Digest::from_sha256(writer.finish());
}

llama_cache_acct_topology_digest llama_cache_acct_compute_topology_digest(
        const llama_cache_acct_shard_topology & topology) noexcept {
    try {
        llama_sha256_writer writer;
        static constexpr char domain_separator[] = "llama-cache-acct/shard-topology";
        writer.string(domain_separator, sizeof(domain_separator) - 1);
        writer.u32(topology.version);
        writer.u32(topology.device_count);
        writer.u32(uint32_t(topology.split_mode));
        writer.u32(topology.main_device.v);
        const size_t n = std::min(topology.device_identities.size(),
                                  topology.shard_weights.size());
        writer.u64(n);
        for (size_t i = 0; i < n; ++i) {
            const auto & identity = topology.device_identities[i].bytes();
            writer.string(identity.data(), identity.size());
            writer.u32(topology.shard_weights[i]);
        }
        return llama_cache_acct_topology_digest::from_sha256(writer.finish());
    } catch (...) {
        return {};
    }
}

bool llama_cache_acct_build_shard_topology(
        const std::vector<std::string> & ordered_device_identities,
        int16_t split_mode,
        int32_t main_device,
        const float * shard_weights,
        llama_cache_acct_shard_topology & out) noexcept {
    out = {};
    const size_t n = ordered_device_identities.size();
    if (n == 0 || n > llama_max_devices() ||
        split_mode < 0 || split_mode > 3 ||
        main_device < 0 || size_t(main_device) >= n) {
        return false;
    }
    try {
        llama_cache_acct_shard_topology built;
        built.version      = LLAMA_CACHE_ACCT_TOPOLOGY_VERSION;
        built.device_count = uint16_t(n);
        built.split_mode   = split_mode;
        built.main_device  = { uint16_t(main_device) };
        built.device_identities.reserve(n);
        built.shard_weights.resize(n);

        for (const auto & identity : ordered_device_identities) {
            if (identity.empty()) {
                return false;
            }
            built.device_identities.push_back(
                acct_sha256<llama_cache_acct_device_digest>(identity.data(), identity.size()));
        }

        std::vector<double> weights(n, 0.0);
        if (shard_weights) {
            for (size_t i = 0; i < n; ++i) {
                const double w = shard_weights[i];
                if (!std::isfinite(w) || w < 0.0) {
                    return false;
                }
                weights[i] = w;
            }
        }
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (n == 1 && !(sum > 0.0)) {
            weights[0] = 1.0;
            sum = 1.0;
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            return false;
        }

        struct remainder {
            size_t index;
            double fraction;
        };
        std::vector<remainder> remainders;
        remainders.reserve(n);
        uint64_t assigned = 0;
        for (size_t i = 0; i < n; ++i) {
            const double exact =
                weights[i] * double(LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR) / sum;
            const uint32_t base = uint32_t(std::floor(exact));
            built.shard_weights[i] = base;
            assigned += base;
            remainders.push_back({ i, exact - double(base) });
        }
        std::stable_sort(remainders.begin(), remainders.end(),
            [](const remainder & a, const remainder & b) {
                if (a.fraction != b.fraction) {
                    return a.fraction > b.fraction;
                }
                return a.index < b.index;
            });
        if (assigned > LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR) {
            return false;
        }
        for (uint32_t i = uint32_t(assigned);
             i < LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR; ++i) {
            built.shard_weights[remainders[
                (i - uint32_t(assigned)) % remainders.size()].index]++;
        }
        built.digest = llama_cache_acct_compute_topology_digest(built);
        if (!acct_digest_nonzero(built.digest.bytes())) {
            return false;
        }
        out = std::move(built);
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

llama_cache_acct_resource_domain llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency residency) {
    llama_cache_acct_resource_domain out;
    out.residency = residency;
    out.kind      = llama_cache_acct_domain_kind::not_applicable;
    return out;
}

bool llama_cache_acct_resource_domain_valid(const llama_cache_acct_resource_domain & domain) {
    if (domain.residency >= llama_cache_acct_residency::_count ||
        domain.kind      >= llama_cache_acct_domain_kind::_count) {
        return false;
    }

    if (domain.kind == llama_cache_acct_domain_kind::not_applicable) {
        return domain.residency != llama_cache_acct_residency::device &&
               !domain.device_ordinal &&
               !domain.topology;
    }

    return domain.residency == llama_cache_acct_residency::device &&
           bool(domain.device_ordinal) && bool(domain.topology);
}

static bool acct_topology_valid(const llama_cache_acct_shard_topology & topology) {
    if (topology.version != LLAMA_CACHE_ACCT_TOPOLOGY_VERSION ||
        topology.device_count == 0 ||
        topology.device_count > llama_max_devices() ||
        topology.device_identities.size() != topology.device_count ||
        topology.shard_weights.size() != topology.device_count ||
        topology.split_mode < 0 || topology.split_mode > 3 ||
        !topology.main_device ||
        topology.main_device.v >= topology.device_count ||
        !acct_digest_nonzero(topology.digest.bytes()) ||
        topology.digest != llama_cache_acct_compute_topology_digest(topology)) {
        return false;
    }
    uint64_t weight_sum = 0;
    for (size_t i = 0; i < topology.device_count; ++i) {
        if (!acct_digest_nonzero(topology.device_identities[i].bytes())) {
            return false;
        }
        weight_sum += topology.shard_weights[i];
    }
    return weight_sum == LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR;
}

static const llama_cache_acct_shard_topology * acct_snapshot_topology(
        const llama_cache_acct_snapshot & snapshot,
        llama_cache_acct_topology_id id) {
    for (const auto & row : snapshot.topologies) {
        if (row.id == id) {
            return &row.topology;
        }
    }
    return nullptr;
}

static bool acct_snapshot_domain_valid(
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain) {
    if (!llama_cache_acct_resource_domain_valid(domain)) {
        return false;
    }
    if (domain.kind == llama_cache_acct_domain_kind::not_applicable) {
        return true;
    }
    const auto * topology = acct_snapshot_topology(snapshot, domain.topology);
    return topology && acct_topology_valid(*topology) &&
           domain.device_ordinal.v < topology->device_count;
}

bool llama_cache_acct_snapshot_to_v1(
        const llama_cache_acct_snapshot & source,
        llama_cache_acct_snapshot_v1 & destination) noexcept {
    destination = {};
    destination.serial                    = source.serial;
    destination.live_ops                  = source.live_ops;
    destination.faults_invalid_transition = source.faults_invalid_transition;
    destination.faults_overflow           = source.faults_overflow;
    destination.faults_unknown_id         = source.faults_unknown_id;
    destination.faults_allocation         = source.faults_allocation;
    for (auto & categories : destination.cells) {
        for (auto & cell : categories) {
            for (auto & measure : cell.measures) {
                measure = { 0, llama_cache_acct_known::unknown };
            }
        }
    }

    const auto reject = [&destination]() {
        destination.completeness = llama_cache_acct_known::unavailable;
        return false;
    };
    const auto v1_domain = [&source](const llama_cache_acct_resource_domain & domain) {
        return acct_snapshot_domain_valid(source, domain) &&
               domain.kind == llama_cache_acct_domain_kind::not_applicable;
    };
    if (source.schema_version != LLAMA_CACHE_ACCT_SCHEMA_VERSION ||
        source.completeness_manifest >= llama_cache_acct_known::_count) {
        return reject();
    }
    for (size_t i = 0; i < source.cells.size(); ++i) {
        const auto & row = source.cells[i];
        if (row.category >= llama_cache_acct_category::_count ||
            !v1_domain(row.domain) ||
            row.certification != llama_cache_acct_known::known) {
            return reject();
        }
        for (const auto & value : row.cell.measures) {
            if (value.state >= llama_cache_acct_known::_count) {
                return reject();
            }
        }
        for (size_t j = 0; j < i; ++j) {
            if (source.cells[j].category == row.category &&
                source.cells[j].domain == row.domain) {
                return reject();
            }
        }
        destination.cells[size_t(row.category)][size_t(row.domain.residency)] = row.cell;
    }
    try {
        destination.allocations.reserve(source.allocations.size());
        for (const auto & row : source.allocations) {
            if (row.category >= llama_cache_acct_category::_count ||
                row.attribution.kind >= llama_cache_acct_attr_kind::_count ||
                !v1_domain(row.domain) ||
                row.certification != llama_cache_acct_known::known) {
                destination.allocations.clear();
                return reject();
            }
            destination.allocations.push_back({
                row.alloc,
                row.attribution,
                row.category,
                row.domain.residency,
                row.logical_bytes,
                row.resident_bytes,
                row.committed_refs,
                row.artifact_identity,
                row.content_digest,
                row.lineage_identity,
            });
        }
    } catch (...) {
        destination.allocations.clear();
        return reject();
    }

    destination.completeness = source.completeness_manifest;
    for (size_t i = 0; i < source.completeness.size(); ++i) {
        const auto & row = source.completeness[i];
        if (row.producer >= llama_cache_acct_producer::_count ||
            row.state    >= llama_cache_acct_known::_count ||
            !v1_domain(row.domain)) {
            return reject();
        }
        for (size_t j = 0; j < i; ++j) {
            if (source.completeness[j].domain == row.domain &&
                source.completeness[j].producer == row.producer) {
                return reject();
            }
        }
        if (row.state == llama_cache_acct_known::unavailable) {
            destination.completeness = llama_cache_acct_known::unavailable;
        } else if (row.state == llama_cache_acct_known::unknown &&
                   destination.completeness == llama_cache_acct_known::known) {
            destination.completeness = llama_cache_acct_known::unknown;
        }
    }
    return true;
}

llama_cache_acct_ledger::llama_cache_acct_ledger() = default;

void llama_cache_acct_ledger::bump_serial() {
    if (state.serial == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        return;
    }
    state.serial++;
}

const llama_cache_acct_shard_topology * llama_cache_acct_ledger::find_topology(
        llama_cache_acct_topology_id id) const {
    for (const auto & row : state.topologies) {
        if (row.id == id) {
            return &row.topology;
        }
    }
    return nullptr;
}

bool llama_cache_acct_ledger::domain_registered(
        const llama_cache_acct_resource_domain & domain) const {
    if (!llama_cache_acct_resource_domain_valid(domain)) {
        return false;
    }
    if (domain.kind == llama_cache_acct_domain_kind::not_applicable) {
        return true;
    }
    const auto * topology = find_topology(domain.topology);
    return topology && domain.device_ordinal.v < topology->device_count;
}

bool llama_cache_acct_ledger::domain_manifested(
        const llama_cache_acct_resource_domain & domain) const {
    return std::any_of(state.completeness.begin(), state.completeness.end(),
        [&](const auto & row) { return row.domain == domain; });
}

bool llama_cache_acct_ledger::domain_use_valid(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain) const {
    return category < llama_cache_acct_category::_count &&
           domain_registered(domain) && domain_manifested(domain);
}

llama_cache_acct_known llama_cache_acct_ledger::domain_certification(
        const llama_cache_acct_resource_domain & domain) const {
    if (state.completeness_manifest != llama_cache_acct_known::known) {
        return llama_cache_acct_known::unavailable;
    }
    bool found = false;
    for (const auto & row : state.completeness) {
        if (row.domain != domain) {
            continue;
        }
        found = true;
        if (row.state != llama_cache_acct_known::known) {
            return llama_cache_acct_known::unavailable;
        }
    }
    return found ? llama_cache_acct_known::known
                 : llama_cache_acct_known::unavailable;
}

llama_cache_acct_cell_row * llama_cache_acct_ledger::find_cell(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain) {
    for (auto & row : state.cells) {
        if (row.category == c && row.domain == domain) {
            return &row;
        }
    }
    return nullptr;
}

llama_cache_acct_completeness_row * llama_cache_acct_ledger::find_completeness(
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) {
    for (auto & row : state.completeness) {
        if (row.domain == domain && row.producer == producer) {
            return &row;
        }
    }
    return nullptr;
}

void llama_cache_acct_ledger::cell_add(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure m,
        uint64_t v) {
    auto * row = find_cell(c, domain);
    if (!row) {
        state.faults_invalid_transition++;
        return;
    }
    auto & value = row->cell.measures[size_t(m)];
    if (value.state == llama_cache_acct_known::unavailable) {
        return;
    }
    if (value.value > std::numeric_limits<uint64_t>::max() - v) {
        value.state = llama_cache_acct_known::unavailable;
        state.faults_overflow++;
        return;
    }
    value.value += v;
    value.state  = llama_cache_acct_known::known;
}

void llama_cache_acct_ledger::cell_sub(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure m,
        uint64_t v) {
    auto * row = find_cell(c, domain);
    if (!row) {
        state.faults_invalid_transition++;
        return;
    }
    auto & value = row->cell.measures[size_t(m)];
    if (value.state == llama_cache_acct_known::unavailable) {
        return;
    }
    if (value.value < v) {
        value.state = llama_cache_acct_known::unavailable;
        state.faults_overflow++;
        return;
    }
    value.value -= v;
    value.state  = llama_cache_acct_known::known;
}

void llama_cache_acct_ledger::cell_latch_unavailable(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure m) {
    auto * row = find_cell(c, domain);
    if (row) {
        row->cell.measures[size_t(m)].state = llama_cache_acct_known::unavailable;
    } else {
        state.faults_invalid_transition++;
    }
}

void llama_cache_acct_ledger::staged_add(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain,
        uint64_t v) {
    auto * row = find_cell(c, domain);
    if (!row) {
        state.faults_invalid_transition++;
        return;
    }
    if (row->staged > std::numeric_limits<uint64_t>::max() - v) {
        state.faults_overflow++;
        row->cell.measures[size_t(llama_cache_acct_measure::transient_peak)].state =
            llama_cache_acct_known::unavailable;
        return;
    }
    row->staged += v;
    auto & peak = row->cell.measures[size_t(llama_cache_acct_measure::transient_peak)];
    if (peak.state != llama_cache_acct_known::unavailable && row->staged > peak.value) {
        peak.value = row->staged;
        peak.state = llama_cache_acct_known::known;
    }
}

void llama_cache_acct_ledger::staged_sub(
        llama_cache_acct_category c,
        const llama_cache_acct_resource_domain & domain,
        uint64_t v) {
    auto * row = find_cell(c, domain);
    if (!row) {
        state.faults_invalid_transition++;
        return;
    }
    if (row->staged < v) {
        state.faults_overflow++;
        cell_latch_unavailable(c, domain, llama_cache_acct_measure::transient_peak);
        return;
    }
    row->staged -= v;
}

void llama_cache_acct_ledger::maybe_retire(
        llama_cache_acct_alloc_id alloc) {
    const auto it = allocs.find(alloc);
    if (it == allocs.end()) {
        return;
    }
    auto & entry = it->second;
    if (entry.tuple_set &&
        entry.staged_refs == 0 &&
        entry.committed_refs == 0) {
        allocs.erase(it);
    }
}

bool llama_cache_acct_ledger::make_device_domain(
        const llama_cache_acct_shard_topology & topology,
        llama_cache_acct_device_ordinal ordinal,
        llama_cache_acct_resource_domain & out) {
    std::lock_guard<std::mutex> lock(mtx);
    out = {};
    if (!acct_topology_valid(topology) ||
        !ordinal || ordinal.v >= topology.device_count) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    try {
        llama_cache_acct_topology_id id;
        for (const auto & row : state.topologies) {
            if (row.topology == topology) {
                id = row.id;
                break;
            }
        }
        if (!id) {
            if (state.topologies.size() >= std::numeric_limits<uint32_t>::max()) {
                state.faults_overflow++;
                bump_serial();
                return false;
            }
            id = { uint32_t(state.topologies.size() + 1) };
            state.topologies.push_back({ id, topology });
            bump_serial();
        }
        out.residency      = llama_cache_acct_residency::device;
        out.kind           = llama_cache_acct_domain_kind::device_topology;
        out.device_ordinal = ordinal;
        out.topology       = id;
        return true;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return false;
    }
}

llama_cache_acct_alloc_id llama_cache_acct_ledger::new_alloc() {
    std::lock_guard<std::mutex> lock(mtx);
    if (next_alloc_id.v == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return {};
    }
    try {
        const llama_cache_acct_alloc_id id = next_alloc_id;
        allocs.emplace(id, alloc_entry{});
        next_alloc_id.v++;
        return id;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return {};
    }
}

llama_cache_acct_op_id llama_cache_acct_ledger::reserve(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_attribution attribution,
        uint64_t expected_logical,
        uint64_t expected_resident) {
    std::lock_guard<std::mutex> lock(mtx);
    return reserve_locked(category, domain, attribution, expected_logical, expected_resident);
}

llama_cache_conditional_reserve_result llama_cache_acct_ledger::reserve_if_serial(
        uint64_t expected_serial,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_attribution attribution,
        uint64_t expected_logical,
        uint64_t expected_resident) {
    std::lock_guard<std::mutex> lock(mtx);
    // Serial guard first: drift is expected optimistic-concurrency contention, not a fault. Leave
    // the ledger completely untouched (no op, no cell, no serial bump) so a re-snapshot sees the
    // real state; only the process-local conflict counter moves.
    if (state.serial != expected_serial) {
        serial_conflicts_++;
        return { llama_cache_conditional_reserve_status::serial_conflict, {} };
    }
    // reserve_locked returns {} (and latches a fault) on a hard failure, else the minted op.
    const llama_cache_acct_op_id op =
        reserve_locked(category, domain, attribution, expected_logical, expected_resident);
    return { op ? llama_cache_conditional_reserve_status::admitted
                : llama_cache_conditional_reserve_status::ledger_fault, op };
}

llama_cache_conditional_reserve_set_result
llama_cache_acct_ledger::reserve_set_if_serial(
        uint64_t expected_serial,
        const llama_cache_conditional_reserve_request * requests,
        size_t n_requests,
        llama_cache_acct_op_id * output_ops) noexcept {
    if (!requests || n_requests == 0 || !output_ops) {
        return {};
    }
    std::fill(output_ops, output_ops + n_requests,
              llama_cache_acct_op_id{});
    std::lock_guard<std::mutex> lock(mtx);
    if (state.serial != expected_serial) {
        serial_conflicts_++;
        return {
            llama_cache_conditional_reserve_status::serial_conflict,
            SIZE_MAX,
        };
    }
    if (n_requests >
            std::numeric_limits<uint64_t>::max() - next_op.v) {
        state.faults_overflow++;
        bump_serial();
        return {
            llama_cache_conditional_reserve_status::ledger_fault, 0,
        };
    }
    try {
        ops.reserve(ops.size() + n_requests);
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return {
            llama_cache_conditional_reserve_status::ledger_fault, 0,
        };
    }

    for (size_t i = 0; i < n_requests; ++i) {
        const auto & request = requests[i];
        const auto op = reserve_locked(
            request.category, request.domain, request.attribution,
            request.expected_logical, request.expected_resident);
        if (op) {
            output_ops[i] = op;
            continue;
        }
        for (size_t j = 0; j < i; ++j) {
            const auto found = ops.find(output_ops[j]);
            if (found != ops.end()) {
                cell_sub(found->second.category, found->second.domain,
                         llama_cache_acct_measure::reserved,
                         found->second.reserved_bytes);
                ops.erase(found);
            }
            output_ops[j] = {};
        }
        bump_serial();
        return {
            llama_cache_conditional_reserve_status::ledger_fault, i,
        };
    }
    return {
        llama_cache_conditional_reserve_status::admitted, SIZE_MAX,
    };
}

uint64_t llama_cache_acct_ledger::serial_conflicts() const noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    return serial_conflicts_;
}

uint64_t llama_cache_acct_ledger::serial() const noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    return state.serial;
}

size_t llama_cache_acct_ledger::allocation_registry_size() const noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    return allocs.size();
}

llama_cache_acct_op_id llama_cache_acct_ledger::reserve_locked(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_attribution attribution,
        uint64_t expected_logical,
        uint64_t expected_resident) {
    (void) expected_logical;
    if (!domain_use_valid(category, domain)) {
        state.faults_invalid_transition++;
        bump_serial();
        return {};
    }
    if (next_op.v == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return {};
    }

    try {
        // Aggregate/staging trackers were created atomically with the manifest.
        auto * row = find_cell(category, domain);
        if (!row) {
            state.faults_invalid_transition++;
            bump_serial();
            return {};
        }
        // Preflight the reserved aggregate: a reservation cell_add cannot record (target already
        // unavailable, or the checked add would overflow) must fail-closed BEFORE minting an op.
        // Otherwise reserve_if_serial would report `admitted` for a reservation the ledger never
        // actually took, since cell_add is void and latches silently.
        // No phantom op, no next_op consumed.
        auto & reserved = row->cell.measures[size_t(llama_cache_acct_measure::reserved)];
        if (reserved.state == llama_cache_acct_known::unavailable) {
            state.faults_invalid_transition++;
            bump_serial();
            return {};
        }
        if (reserved.value > std::numeric_limits<uint64_t>::max() - expected_resident) {
            state.faults_overflow++;
            bump_serial();
            return {};
        }

        const llama_cache_acct_op_id op = next_op;
        txn t;
        t.state          = llama_cache_acct_txn_state::reserved;
        t.category       = category;
        t.domain         = domain;
        t.attribution    = attribution;
        t.reserved_bytes = expected_resident;
        ops.emplace(op, t);
        next_op.v++;

        cell_add(category, domain, llama_cache_acct_measure::reserved, expected_resident);
        bump_serial();
        return op;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return {};
    }
}

bool llama_cache_acct_ledger::stage(
        llama_cache_acct_op_id op,
        llama_cache_acct_alloc_id alloc,
        uint64_t resident_bytes,
        llama_cache_acct_artifact_id artifact,
        llama_cache_acct_content_digest digest,
        llama_cache_acct_lineage_id lineage) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::reserved) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    const auto & domain = it->second.domain;
    auto ait = allocs.find(alloc);
    if (!alloc || ait == allocs.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (ait->second.tuple_set) {
        if (ait->second.category       != it->second.category ||
            ait->second.domain         != domain ||
            ait->second.resident_bytes != resident_bytes ||
            ait->second.artifact       != artifact ||
            ait->second.digest         != digest ||
            ait->second.lineage        != lineage) {
            state.faults_invalid_transition++;
            bump_serial();
            return false;
        }
    }
    if (ait->second.staged_refs == std::numeric_limits<uint32_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return false;
    }

    if (!ait->second.tuple_set) {
        ait->second.tuple_set      = true;
        ait->second.category       = it->second.category;
        ait->second.domain         = domain;
        ait->second.resident_bytes = resident_bytes;
        ait->second.artifact       = artifact;
        ait->second.digest         = digest;
        ait->second.lineage        = lineage;
    }
    ait->second.staged_refs++;

    it->second.state          = llama_cache_acct_txn_state::staged;
    it->second.alloc          = alloc;
    it->second.resident_bytes = resident_bytes;
    it->second.artifact       = artifact;
    it->second.digest         = digest;
    it->second.lineage        = lineage;
    staged_add(it->second.category, it->second.domain, resident_bytes);
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::commit(llama_cache_acct_op_id op, uint64_t logical_bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::staged) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    auto ait = allocs.find(it->second.alloc);
    if (ait == allocs.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }

    auto & entry = ait->second;
    if (entry.committed_refs > 0 &&
        entry.charged_logical != logical_bytes) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    if (entry.committed_refs == std::numeric_limits<uint32_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return false;
    }

    it->second.state = llama_cache_acct_txn_state::committed;
    staged_sub(it->second.category, it->second.domain, it->second.resident_bytes);
    if (entry.staged_refs > 0) {
        entry.staged_refs--;
    }
    cell_sub(it->second.category, it->second.domain,
             llama_cache_acct_measure::reserved, it->second.reserved_bytes);

    entry.committed_refs++;
    if (entry.committed_refs == 1) {
        entry.charged_logical = logical_bytes;
        entry.attribution     = it->second.attribution;
        cell_add(entry.category, entry.domain,
                 llama_cache_acct_measure::logical_payload, entry.charged_logical);
        cell_add(entry.category, entry.domain,
                 llama_cache_acct_measure::resident_allocated, entry.resident_bytes);
    }
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::abort(llama_cache_acct_op_id op) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::reserved &&
        it->second.state != llama_cache_acct_txn_state::staged) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }

    cell_sub(it->second.category, it->second.domain,
             llama_cache_acct_measure::reserved, it->second.reserved_bytes);
    llama_cache_acct_alloc_id staged_alloc;
    if (it->second.state == llama_cache_acct_txn_state::staged) {
        staged_sub(it->second.category, it->second.domain, it->second.resident_bytes);
        auto ait = allocs.find(it->second.alloc);
        if (ait != allocs.end() && ait->second.staged_refs > 0) {
            staged_alloc = it->second.alloc;
            ait->second.staged_refs--;
        }
    }
    ops.erase(it);
    if (staged_alloc) {
        maybe_retire(staged_alloc);
    }
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::abort_set(
        const llama_cache_acct_op_id * selected,
        size_t n_selected) noexcept {
    if (!selected || n_selected == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    llama_cache_acct_op_id previous;
    for (size_t i = 0; i < n_selected; ++i) {
        const auto op = selected[i];
        const auto found = ops.find(op);
        if (!op || (previous && op.v <= previous.v) ||
            found == ops.end() ||
            (found->second.state != llama_cache_acct_txn_state::reserved &&
             found->second.state != llama_cache_acct_txn_state::staged)) {
            state.faults_invalid_transition++;
            bump_serial();
            return false;
        }
        previous = op;
    }
    for (size_t i = 0; i < n_selected; ++i) {
        const auto found = ops.find(selected[i]);
        GGML_ASSERT(found != ops.end());
        cell_sub(found->second.category, found->second.domain,
                 llama_cache_acct_measure::reserved,
                 found->second.reserved_bytes);
        llama_cache_acct_alloc_id staged_alloc;
        if (found->second.state == llama_cache_acct_txn_state::staged) {
            staged_sub(found->second.category, found->second.domain,
                       found->second.resident_bytes);
            const auto allocation = allocs.find(found->second.alloc);
            if (allocation != allocs.end() &&
                allocation->second.staged_refs > 0) {
                staged_alloc = found->second.alloc;
                allocation->second.staged_refs--;
            }
        }
        ops.erase(found);
        if (staged_alloc) {
            maybe_retire(staged_alloc);
        }
    }
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::shrink_reservation_set(
        const llama_cache_acct_op_id * selected,
        const uint64_t * resident_bytes,
        size_t n_selected) noexcept {
    if (!selected || !resident_bytes || n_selected == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    llama_cache_acct_op_id previous;
    for (size_t i = 0; i < n_selected; ++i) {
        const auto op = selected[i];
        const auto found = ops.find(op);
        if (!op || (previous && op.v <= previous.v) ||
            found == ops.end() ||
            found->second.state != llama_cache_acct_txn_state::reserved ||
            resident_bytes[i] > found->second.reserved_bytes) {
            state.faults_invalid_transition++;
            bump_serial();
            return false;
        }
        previous = op;
    }
    for (size_t i = 0; i < n_selected; ++i) {
        auto & transaction = ops.find(selected[i])->second;
        const uint64_t released =
            transaction.reserved_bytes - resident_bytes[i];
        if (released != 0) {
            cell_sub(transaction.category, transaction.domain,
                     llama_cache_acct_measure::reserved, released);
            transaction.reserved_bytes = resident_bytes[i];
        }
    }
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::repartition_reservation_set_downward(
        const llama_cache_acct_op_id * selected,
        size_t n_selected,
        const llama_cache_conditional_reserve_request * replacements,
        size_t n_replacements,
        llama_cache_acct_op_id * output_ops) noexcept {
    if (!selected || n_selected == 0 || !replacements ||
        n_replacements == 0 || !output_ops) {
        return false;
    }
    std::fill(output_ops, output_ops + n_replacements,
              llama_cache_acct_op_id {});
    std::lock_guard<std::mutex> lock(mtx);
    try {
        struct aggregate {
            llama_cache_acct_category category;
            llama_cache_acct_resource_domain domain;
            uint64_t selected = 0;
            uint64_t replacement = 0;
            bool selected_present = false;
            bool replacement_present = false;
        };
        std::vector<aggregate> totals;
        totals.reserve(n_selected + n_replacements);

        llama_cache_acct_op_id previous;
        for (size_t i = 0; i < n_selected; ++i) {
            const auto op = selected[i];
            const auto found = ops.find(op);
            if (!op || (previous && op.v <= previous.v) ||
                found == ops.end() ||
                found->second.state != llama_cache_acct_txn_state::reserved) {
                state.faults_invalid_transition++;
                bump_serial();
                return false;
            }
            previous = op;
            totals.push_back({ found->second.category,
                               found->second.domain,
                               found->second.reserved_bytes, 0,
                               true, false });
        }
        for (size_t i = 0; i < n_replacements; ++i) {
            const auto & request = replacements[i];
            totals.push_back({ request.category, request.domain, 0,
                               request.expected_resident, false, true });
        }
        std::sort(totals.begin(), totals.end(), [&](const aggregate & lhs,
                                                    const aggregate & rhs) {
            if (lhs.category != rhs.category) {
                return lhs.category < rhs.category;
            }
            return llama_cache_acct_resource_domain_less(
                lhs.domain, rhs.domain);
        });
        size_t aggregate_count = 0;
        for (size_t begin = 0; begin < totals.size();) {
            size_t end = begin + 1;
            uint64_t selected_total = totals[begin].selected;
            uint64_t replacement_total = totals[begin].replacement;
            bool selected_present = totals[begin].selected_present;
            bool replacement_present = totals[begin].replacement_present;
            while (end < totals.size() &&
                   totals[end].category == totals[begin].category &&
                   totals[end].domain == totals[begin].domain) {
                if (totals[end].selected >
                        std::numeric_limits<uint64_t>::max() -
                            selected_total ||
                    totals[end].replacement >
                        std::numeric_limits<uint64_t>::max() -
                            replacement_total) {
                    state.faults_overflow++;
                    bump_serial();
                    return false;
                }
                selected_total += totals[end].selected;
                replacement_total += totals[end].replacement;
                selected_present = selected_present ||
                    totals[end].selected_present;
                replacement_present = replacement_present ||
                    totals[end].replacement_present;
                ++end;
            }
            if ((replacement_present && !selected_present) ||
                (replacement_present &&
                 (!domain_use_valid(totals[begin].category,
                                    totals[begin].domain) ||
                  !find_cell(totals[begin].category,
                             totals[begin].domain))) ||
                replacement_total > selected_total) {
                state.faults_invalid_transition++;
                bump_serial();
                return false;
            }
            totals[aggregate_count] = totals[begin];
            totals[aggregate_count].selected = selected_total;
            totals[aggregate_count].replacement = replacement_total;
            totals[aggregate_count].selected_present = selected_present;
            totals[aggregate_count].replacement_present = replacement_present;
            ++aggregate_count;
            begin = end;
        }
        totals.resize(aggregate_count);
        if (n_replacements >
                std::numeric_limits<uint64_t>::max() - next_op.v) {
            state.faults_overflow++;
            bump_serial();
            return false;
        }

        // Insert the replacement nodes before touching the selected set. If a
        // node allocation fails, erase the invisible replacements and leave
        // every old reservation and gauge intact.
        ops.reserve(ops.size() + n_replacements);
        size_t inserted = 0;
        try {
            for (; inserted < n_replacements; ++inserted) {
                const auto op = llama_cache_acct_op_id {
                    next_op.v + inserted,
                };
                txn transaction;
                transaction.state = llama_cache_acct_txn_state::reserved;
                transaction.category = replacements[inserted].category;
                transaction.domain = replacements[inserted].domain;
                transaction.attribution = replacements[inserted].attribution;
                transaction.reserved_bytes =
                    replacements[inserted].expected_resident;
                const auto added = ops.emplace(op, transaction);
                if (!added.second) {
                    throw std::bad_alloc();
                }
                output_ops[inserted] = op;
            }
        } catch (...) {
            for (size_t i = 0; i < inserted; ++i) {
                ops.erase(output_ops[i]);
                output_ops[i] = {};
            }
            state.faults_allocation++;
            bump_serial();
            return false;
        }

        for (size_t i = 0; i < n_selected; ++i) {
            const auto found = ops.find(selected[i]);
            GGML_ASSERT(found != ops.end());
            ops.erase(found);
        }
        for (const auto & total : totals) {
            cell_sub(total.category, total.domain,
                     llama_cache_acct_measure::reserved,
                     total.selected - total.replacement);
        }
        next_op.v += n_replacements;
        bump_serial();
        return true;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return false;
    }
}

llama_cache_acct_ledger::release_resolution_status
llama_cache_acct_ledger::resolve_release_locked(
        llama_cache_acct_op_id op,
        release_resolution & out) const noexcept {
    out = {};
    const auto it = ops.find(op);
    if (it == ops.end()) {
        return release_resolution_status::unknown_op;
    }
    if (it->second.state != llama_cache_acct_txn_state::committed) {
        return release_resolution_status::invalid_state;
    }
    const auto ait = allocs.find(it->second.alloc);
    if (ait == allocs.end() || ait->second.committed_refs == 0) {
        return release_resolution_status::unknown_allocation;
    }
    out.operation  = &it->second;
    out.allocation = &ait->second;
    return release_resolution_status::ok;
}

void llama_cache_acct_ledger::apply_release_locked(
        llama_cache_acct_op_id op) noexcept {
    const auto it = ops.find(op);
    GGML_ASSERT(it != ops.end());
    const llama_cache_acct_alloc_id alloc = it->second.alloc;
    const auto ait = allocs.find(alloc);
    GGML_ASSERT(ait != allocs.end());
    GGML_ASSERT(ait->second.committed_refs > 0);

    auto & entry = ait->second;
    entry.committed_refs--;
    if (entry.committed_refs == 0) {
        cell_sub(entry.category, entry.domain,
                 llama_cache_acct_measure::logical_payload,
                 entry.charged_logical);
        cell_sub(entry.category, entry.domain,
                 llama_cache_acct_measure::resident_allocated,
                 entry.resident_bytes);
    }
    ops.erase(it);
    maybe_retire(alloc);
}

bool llama_cache_acct_ledger::release(llama_cache_acct_op_id op) {
    std::lock_guard<std::mutex> lock(mtx);
    release_resolution resolved;
    const auto status = resolve_release_locked(op, resolved);
    if (status != release_resolution_status::ok) {
        if (status == release_resolution_status::invalid_state) {
            state.faults_invalid_transition++;
        } else {
            state.faults_unknown_id++;
        }
        bump_serial();
        return false;
    }

    apply_release_locked(op);
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::preview_release(
        llama_cache_acct_op_id op,
        llama_cache_acct_release_preview & out) const noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    release_resolution resolved;
    if (resolve_release_locked(op, resolved) !=
            release_resolution_status::ok) {
        return false;
    }

    out = {};
    out.category = resolved.allocation->category;
    out.domain   = resolved.allocation->domain;
    const bool last = resolved.allocation->committed_refs == 1;
    out.logical_payload = llama_cache_acct_value::measured(
        last ? resolved.allocation->charged_logical : 0);
    out.resident_allocated = llama_cache_acct_value::measured(
        last ? resolved.allocation->resident_bytes : 0);
    return true;
}

bool llama_cache_acct_ledger::preview_release_set(
        const std::vector<llama_cache_acct_op_id> & selected,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out,
        bool include_category_yields) const noexcept {
    out = {};
    std::lock_guard<std::mutex> lock(mtx);
    if (state.serial != expected_serial) {
        return false;
    }
    try {
        std::unordered_map<llama_cache_acct_alloc_id, uint32_t> selected_refs;
        selected_refs.reserve(selected.size());
        std::unordered_set<llama_cache_acct_op_id> unique_ops;
        unique_ops.reserve(selected.size());

        for (const auto op : selected) {
            if (!op || !unique_ops.insert(op).second) {
                return false;
            }
            release_resolution resolved;
            if (resolve_release_locked(op, resolved) !=
                    release_resolution_status::ok) {
                return false;
            }
            auto & count = selected_refs[resolved.operation->alloc];
            if (count == std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            count++;
        }

        llama_cache_acct_release_set_preview next;
        next.accounting_serial = expected_serial;
        next.rows.reserve(selected_refs.size());
        for (const auto & [alloc, count] : selected_refs) {
            const auto it = allocs.find(alloc);
            if (it == allocs.end() || count > it->second.committed_refs) {
                return false;
            }
            if (count != it->second.committed_refs) {
                continue;
            }
            auto row = std::find_if(
                next.rows.begin(), next.rows.end(),
                [&](const auto & candidate) {
                    return candidate.domain == it->second.domain;
                });
            if (row == next.rows.end()) {
                next.rows.push_back({ it->second.domain, 0, 0 });
                row = std::prev(next.rows.end());
            }
            if (it->second.charged_logical >
                    std::numeric_limits<uint64_t>::max() -
                        row->logical_payload ||
                it->second.resident_bytes >
                    std::numeric_limits<uint64_t>::max() -
                        row->resident_allocated) {
                return false;
            }
            row->logical_payload += it->second.charged_logical;
            row->resident_allocated += it->second.resident_bytes;
            if (include_category_yields) {
                auto yield = std::find_if(
                    next.yield_rows.begin(), next.yield_rows.end(),
                    [&](const auto & candidate) {
                        return candidate.category == it->second.category &&
                               candidate.domain == it->second.domain;
                    });
                if (yield == next.yield_rows.end()) {
                    next.yield_rows.push_back({
                        it->second.category, it->second.domain, 0, 0 });
                    yield = std::prev(next.yield_rows.end());
                }
                if (it->second.charged_logical >
                        std::numeric_limits<uint64_t>::max() -
                            yield->logical_payload ||
                    it->second.resident_bytes >
                        std::numeric_limits<uint64_t>::max() -
                            yield->resident_allocated) {
                    return false;
                }
                yield->logical_payload += it->second.charged_logical;
                yield->resident_allocated += it->second.resident_bytes;
            }
        }
        std::sort(next.rows.begin(), next.rows.end(),
            [](const auto & a, const auto & b) {
                if (a.domain.residency != b.domain.residency) {
                    return a.domain.residency < b.domain.residency;
                }
                if (a.domain.kind != b.domain.kind) {
                    return a.domain.kind < b.domain.kind;
                }
                if (a.domain.topology != b.domain.topology) {
                    return a.domain.topology.v < b.domain.topology.v;
                }
                return a.domain.device_ordinal.v <
                       b.domain.device_ordinal.v;
            });
        std::sort(next.yield_rows.begin(), next.yield_rows.end(),
            [](const auto & a, const auto & b) {
                if (a.category != b.category) {
                    return a.category < b.category;
                }
                if (a.domain.residency != b.domain.residency) {
                    return a.domain.residency < b.domain.residency;
                }
                if (a.domain.kind != b.domain.kind) {
                    return a.domain.kind < b.domain.kind;
                }
                if (a.domain.topology != b.domain.topology) {
                    return a.domain.topology.v < b.domain.topology.v;
                }
                return a.domain.device_ordinal.v <
                       b.domain.device_ordinal.v;
            });
        out = std::move(next);
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool llama_cache_acct_ledger::preview_release_set_resident_batch(
        const std::vector<llama_cache_acct_release_set_view> & sets,
        uint64_t expected_serial,
        std::vector<uint64_t> & out) const noexcept {
    out.clear();
    std::lock_guard<std::mutex> lock(mtx);
    if (state.serial != expected_serial) {
        return false;
    }
    try {
        size_t max_ops = 0;
        for (const auto & set : sets) {
            if ((set.size != 0 && set.data == nullptr) ||
                set.size > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            max_ops = std::max(max_ops, set.size);
        }
        out.assign(sets.size(), 0);
        std::unordered_map<llama_cache_acct_alloc_id, uint32_t> selected_refs;
        std::unordered_set<llama_cache_acct_op_id> unique_ops;
        selected_refs.reserve(max_ops);
        unique_ops.reserve(max_ops);
        for (size_t i = 0; i < sets.size(); ++i) {
            selected_refs.clear();
            unique_ops.clear();
            const auto & set = sets[i];
            for (size_t j = 0; j < set.size; ++j) {
                const auto op = set.data[j];
                if (!op || !unique_ops.insert(op).second) {
                    out.clear();
                    return false;
                }
                release_resolution resolved;
                if (resolve_release_locked(op, resolved) !=
                        release_resolution_status::ok) {
                    out.clear();
                    return false;
                }
                auto & count = selected_refs[resolved.operation->alloc];
                if (count == std::numeric_limits<uint32_t>::max()) {
                    out.clear();
                    return false;
                }
                count++;
            }
            uint64_t resident = 0;
            for (const auto & [alloc, count] : selected_refs) {
                const auto it = allocs.find(alloc);
                if (it == allocs.end() || count > it->second.committed_refs) {
                    out.clear();
                    return false;
                }
                if (count == it->second.committed_refs) {
                    if (it->second.resident_bytes > UINT64_MAX - resident) {
                        out.clear();
                        return false;
                    }
                    resident += it->second.resident_bytes;
                }
            }
            out[i] = resident;
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool llama_cache_acct_ledger::preview_release_set_resident_conditioned_batch(
        llama_cache_acct_release_set_view baseline,
        const std::vector<llama_cache_acct_release_set_view> & candidates,
        uint64_t expected_serial,
        std::vector<uint64_t> & out) const noexcept {
    out.clear();
    std::lock_guard<std::mutex> lock(mtx);
    if (state.serial != expected_serial || baseline.data == nullptr ||
        baseline.size == 0 ||
        baseline.size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    try {
        size_t max_candidate_ops = 0;
        for (const auto & candidate : candidates) {
            if ((candidate.size != 0 && candidate.data == nullptr) ||
                candidate.size > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            max_candidate_ops = std::max(max_candidate_ops, candidate.size);
        }

        std::unordered_map<llama_cache_acct_alloc_id, uint32_t> baseline_refs;
        std::unordered_set<llama_cache_acct_op_id> baseline_ops;
        baseline_refs.reserve(baseline.size);
        baseline_ops.reserve(baseline.size);
        for (size_t i = 0; i < baseline.size; ++i) {
            const auto op = baseline.data[i];
            if (!op || !baseline_ops.insert(op).second) {
                return false;
            }
            release_resolution resolved;
            if (resolve_release_locked(op, resolved) !=
                    release_resolution_status::ok) {
                return false;
            }
            auto & count = baseline_refs[resolved.operation->alloc];
            if (count == std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            count++;
        }
        for (const auto & [alloc, count] : baseline_refs) {
            const auto it = allocs.find(alloc);
            if (it == allocs.end() || count > it->second.committed_refs) {
                return false;
            }
        }

        out.assign(candidates.size(), 0);
        std::unordered_map<llama_cache_acct_alloc_id, uint32_t> candidate_refs;
        std::unordered_set<llama_cache_acct_op_id> unique_ops;
        candidate_refs.reserve(max_candidate_ops);
        unique_ops.reserve(max_candidate_ops);
        for (size_t i = 0; i < candidates.size(); ++i) {
            candidate_refs.clear();
            unique_ops.clear();
            const auto & candidate = candidates[i];
            for (size_t j = 0; j < candidate.size; ++j) {
                const auto op = candidate.data[j];
                if (!op || baseline_ops.count(op) != 0 ||
                    !unique_ops.insert(op).second) {
                    out.clear();
                    return false;
                }
                release_resolution resolved;
                if (resolve_release_locked(op, resolved) !=
                        release_resolution_status::ok) {
                    out.clear();
                    return false;
                }
                auto & count = candidate_refs[resolved.operation->alloc];
                if (count == std::numeric_limits<uint32_t>::max()) {
                    out.clear();
                    return false;
                }
                count++;
            }
            uint64_t resident = 0;
            for (const auto & [alloc, count] : candidate_refs) {
                const auto it = allocs.find(alloc);
                const auto base = baseline_refs.find(alloc);
                const uint32_t baseline_count = base == baseline_refs.end()
                    ? 0 : base->second;
                if (it == allocs.end() ||
                    baseline_count > it->second.committed_refs ||
                    count > it->second.committed_refs - baseline_count) {
                    out.clear();
                    return false;
                }
                if (baseline_count != it->second.committed_refs &&
                    baseline_count + count == it->second.committed_refs) {
                    if (it->second.resident_bytes > UINT64_MAX - resident) {
                        out.clear();
                        return false;
                    }
                    resident += it->second.resident_bytes;
                }
            }
            out[i] = resident;
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

llama_cache_conditional_release_status
llama_cache_acct_ledger::release_set_if_serial(
        const std::vector<llama_cache_acct_op_id> & selected,
        uint64_t expected_serial) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    if (state.serial != expected_serial) {
        return llama_cache_conditional_release_status::serial_conflict;
    }
    return release_set_locked(selected);
}

llama_cache_conditional_release_status
llama_cache_acct_ledger::release_set_current(
        const std::vector<llama_cache_acct_op_id> & selected) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    return release_set_locked(selected);
}

llama_cache_conditional_release_status
llama_cache_acct_ledger::release_set_locked(
        const std::vector<llama_cache_acct_op_id> & selected) noexcept {
    // An empty exact union is a valid known-zero release. Live-slot clearing
    // drops logical sequence ownership from fixed pooled KV allocations, so
    // there may be no transactional allocation to discharge. Preserve the
    // serial fence without fabricating an accounting mutation.
    if (selected.empty()) {
        return llama_cache_conditional_release_status::released;
    }

    // Full validation precedes the first mutation. prepare_release_set sorts
    // this vector, so adjacent comparison is a no-allocation uniqueness proof.
    llama_cache_acct_op_id previous;
    for (const auto op : selected) {
        if (!op || (previous && op.v <= previous.v)) {
            state.faults_invalid_transition++;
            bump_serial();
            return llama_cache_conditional_release_status::ledger_fault;
        }
        release_resolution resolved;
        const auto status = resolve_release_locked(op, resolved);
        if (status != release_resolution_status::ok) {
            if (status == release_resolution_status::invalid_state) {
                state.faults_invalid_transition++;
            } else {
                state.faults_unknown_id++;
            }
            bump_serial();
            return llama_cache_conditional_release_status::ledger_fault;
        }
        previous = op;
    }

    for (const auto op : selected) {
        apply_release_locked(op);
    }
    bump_serial();
    return llama_cache_conditional_release_status::released;
}

void llama_cache_acct_ledger::gauge_set(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure measure,
        uint64_t value) {
    std::lock_guard<std::mutex> lock(mtx);
    if (measure >= llama_cache_acct_measure::_count ||
        !domain_use_valid(category, domain)) {
        state.faults_invalid_transition++;
        bump_serial();
        return;
    }
    auto * row = find_cell(category, domain);
    if (!row) {
        state.faults_invalid_transition++;
        bump_serial();
        return;
    }
    auto & cell = row->cell.measures[size_t(measure)];
    if (cell.state != llama_cache_acct_known::unavailable) {
        // A gauge observation is accounting currency only when it changes
        // the observable ledger state.  Re-reporting the same known value is
        // common for live-memory sampling and must not invalidate optimistic
        // claims or mutation-driven retry witnesses.
        if (cell.state == llama_cache_acct_known::known &&
            cell.value == value) {
            return;
        }
        cell.value = value;
        cell.state = llama_cache_acct_known::known;
        bump_serial();
    }
}

bool llama_cache_acct_ledger::gauge_initialize_zero(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure measure) {
    std::lock_guard<std::mutex> lock(mtx);
    if (measure >= llama_cache_acct_measure::_count ||
        !domain_use_valid(category, domain)) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    auto * row = find_cell(category, domain);
    if (!row) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    auto & cell = row->cell.measures[size_t(measure)];
    if (cell.state == llama_cache_acct_known::unavailable) {
        return false;
    }
    if (cell.state == llama_cache_acct_known::unknown) {
        cell = { 0, llama_cache_acct_known::known };
        bump_serial();
    }
    return true;
}

void llama_cache_acct_ledger::mark_unavailable(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure measure) {
    std::lock_guard<std::mutex> lock(mtx);
    if (measure >= llama_cache_acct_measure::_count ||
        !domain_use_valid(category, domain)) {
        state.faults_invalid_transition++;
        bump_serial();
        return;
    }
    cell_latch_unavailable(category, domain, measure);
    bump_serial();
}

bool llama_cache_acct_ledger::configure_required_producers(
        const llama_cache_acct_completeness_requirement * requirements,
        size_t n_requirements) {
    std::lock_guard<std::mutex> lock(mtx);
    if ((n_requirements > 0 && !requirements) ||
        !ops.empty() || !state.cells.empty() || !state.completeness.empty()) {
        state.faults_invalid_transition++;
        if (state.completeness_manifest == llama_cache_acct_known::unknown) {
            state.completeness_manifest = llama_cache_acct_known::unavailable;
        }
        bump_serial();
        return false;
    }
    try {
        std::vector<llama_cache_acct_completeness_row> next;
        std::vector<llama_cache_acct_resource_domain> domains;
        std::vector<llama_cache_acct_cell_row> cells;
        next.reserve(n_requirements);
        domains.reserve(n_requirements);
        for (size_t i = 0; i < n_requirements; ++i) {
            const auto & req = requirements[i];
            if (req.producer >= llama_cache_acct_producer::_count ||
                !domain_registered(req.domain)) {
                state.faults_invalid_transition++;
                state.completeness_manifest = llama_cache_acct_known::unavailable;
                bump_serial();
                return false;
            }
            const bool duplicate = std::any_of(next.begin(), next.end(), [&](const auto & row) {
                return row.domain == req.domain && row.producer == req.producer;
            });
            if (duplicate) {
                state.faults_invalid_transition++;
                state.completeness_manifest = llama_cache_acct_known::unavailable;
                bump_serial();
                return false;
            }
            next.push_back({ req.domain, req.producer, llama_cache_acct_known::unknown });
            if (std::find(domains.begin(), domains.end(), req.domain) == domains.end()) {
                domains.push_back(req.domain);
            }
        }
        cells.reserve(domains.size() * size_t(llama_cache_acct_category::_count));
        for (const auto & domain : domains) {
            for (size_t c = 0; c < size_t(llama_cache_acct_category::_count); ++c) {
                llama_cache_acct_cell_row row;
                row.category = llama_cache_acct_category(c);
                row.domain   = domain;
                cells.push_back(std::move(row));
            }
        }
        state.completeness.swap(next);
        state.cells.swap(cells);
        state.completeness_manifest = llama_cache_acct_known::known;
        bump_serial();
        return true;
    } catch (...) {
        state.faults_allocation++;
        state.completeness_manifest = llama_cache_acct_known::unavailable;
        bump_serial();
        return false;
    }
}

bool llama_cache_acct_ledger::certify_complete(
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) {
    std::lock_guard<std::mutex> lock(mtx);
    auto * row = find_completeness(domain, producer);
    if (!row) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    if (row->state == llama_cache_acct_known::unavailable) {
        return false;
    }
    if (row->state == llama_cache_acct_known::unknown) {
        row->state = llama_cache_acct_known::known;
        bump_serial();
    }
    return true;
}

void llama_cache_acct_ledger::mark_producer_unavailable(
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) {
    std::lock_guard<std::mutex> lock(mtx);
    auto * row = find_completeness(domain, producer);
    if (!row) {
        state.faults_invalid_transition++;
        bump_serial();
        return;
    }
    if (row->state != llama_cache_acct_known::unavailable) {
        row->state = llama_cache_acct_known::unavailable;
        bump_serial();
    }
}

llama_cache_acct_snapshot llama_cache_acct_ledger::snapshot() {
    std::lock_guard<std::mutex> lock(mtx);
    llama_cache_acct_snapshot out;
    try {
        out = state;
        out.live_ops = (uint64_t) ops.size();
        std::vector<std::pair<llama_cache_acct_resource_domain, llama_cache_acct_known>>
            domain_certifications;
        domain_certifications.reserve(state.completeness.size());
        const auto certification_for =
            [this, &domain_certifications](const llama_cache_acct_resource_domain & domain) {
                for (const auto & cached : domain_certifications) {
                    if (cached.first == domain) {
                        return cached.second;
                    }
                }
                const auto certification = domain_certification(domain);
                domain_certifications.emplace_back(domain, certification);
                return certification;
            };
        for (auto & row : out.cells) {
            row.certification = certification_for(row.domain);
            if (row.certification != llama_cache_acct_known::known) {
                for (auto & measure : row.cell.measures) {
                    if (measure.state != llama_cache_acct_known::unknown) {
                        measure = { 0, llama_cache_acct_known::unavailable };
                    }
                }
            }
        }
        out.allocations.reserve(allocs.size());
        for (const auto & [alloc, entry] : allocs) {
            if (entry.committed_refs == 0) {
                continue;
            }
            llama_cache_acct_allocation_row row;
            row.alloc             = alloc;
            row.attribution       = entry.attribution;
            row.category          = entry.category;
            row.domain            = entry.domain;
            row.certification     = certification_for(entry.domain);
            row.logical_bytes     = entry.charged_logical;
            row.resident_bytes    = entry.resident_bytes;
            row.committed_refs    = entry.committed_refs;
            row.artifact_identity = entry.artifact;
            row.content_digest    = entry.digest;
            row.lineage_identity  = entry.lineage;
            out.allocations.push_back(row);
        }
        return out;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();

        llama_cache_acct_snapshot failed;
        failed.serial                    = state.serial;
        failed.completeness_manifest     = llama_cache_acct_known::unavailable;
        failed.live_ops                  = (uint64_t) ops.size();
        failed.faults_invalid_transition = state.faults_invalid_transition;
        failed.faults_overflow           = state.faults_overflow;
        failed.faults_unknown_id         = state.faults_unknown_id;
        failed.faults_allocation         = state.faults_allocation;
        // The failure object intentionally carries no partially-copied rows. Its manifest
        // state is the complete fail-closed signal; retrying allocation inside an allocation
        // failure handler would make the non-throwing contract recursive.
        return failed;
    }
}
