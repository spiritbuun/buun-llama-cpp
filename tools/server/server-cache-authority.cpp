#include "server-cache-authority.h"

#include "server-common.h"
#include "server-task.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

bool add_checked(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        out = 0;
        return false;
    }
    out = a + b;
    return true;
}

struct server_shared_payload_claim {
    const common_shared_byte_buffer * buffer = nullptr;
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    uint64_t bytes = 0;
    llama_cache_acct_artifact_id artifact;
    llama_cache_acct_alloc_id allocation;
    llama_cache_acct_op_id operation;
    bool existing_binding = false;
    bool newly_bound = false;
    bool handle_newly_owned = false;
};

bool add_shared_payload_claim(
        llama_cache_acct_ledger & ledger,
        const common_shared_byte_buffer & buffer,
        llama_cache_acct_category category,
        std::vector<server_shared_payload_claim> & claims) {
    if (buffer.empty() || buffer.storage_identity() == nullptr) {
        return true;
    }
    for (const auto & current : claims) {
        if (current.buffer && current.buffer->storage_identity() ==
                buffer.storage_identity()) {
            return current.category == category &&
                   current.bytes == buffer.size();
        }
    }
    server_shared_payload_claim claim;
    claim.buffer = &buffer;
    claim.category = category;
    claim.bytes = uint64_t(buffer.size());
    const void * owner = nullptr;
    uint64_t allocation = 0;
    if (buffer.accounting_binding(owner, allocation)) {
        if (owner != &ledger) {
            return false;
        }
        claim.allocation = { allocation };
        claim.existing_binding = true;
    }
    claims.push_back(std::move(claim));
    return true;
}

bool add_checkpoint_payload_claims(
        llama_cache_acct_ledger & ledger,
        const common_prompt_checkpoint & checkpoint,
        std::vector<server_shared_payload_claim> & claims) {
    return add_shared_payload_claim(
               ledger, checkpoint.data_tgt,
               llama_cache_acct_category::checkpoint_state_payload,
               claims) &&
           add_shared_payload_claim(
               ledger, checkpoint.data_dft,
               llama_cache_acct_category::checkpoint_state_payload,
               claims) &&
           add_shared_payload_claim(
               ledger, checkpoint.data_qsa,
               llama_cache_acct_category::typed_accelerator_payload,
               claims) &&
           add_shared_payload_claim(
               ledger, checkpoint.accel.ring,
               llama_cache_acct_category::typed_accelerator_payload,
               claims) &&
           add_shared_payload_claim(
               ledger, checkpoint.accel.spec,
               llama_cache_acct_category::typed_accelerator_payload,
               claims);
}

bool execute_shared_payload_claims(
        server_cache_authority & authority,
        std::vector<server_shared_payload_claim> & claims,
        bool fail_after_commit,
        std::vector<llama_cache_acct_op_id> & committed) noexcept {
    committed.clear();
    uint64_t pending = 0;
    try {
        if (claims.empty()) {
            return false;
        }
        for (const auto & claim : claims) {
            if (claim.bytes == 0 ||
                (!claim.allocation &&
                 !add_checked(pending, claim.bytes, pending))) {
                return false;
            }
        }
        llama_cache_budget_config config;
        if (!authority.sample_budget(config, pending)) {
            return false;
        }
        std::vector<llama_cache_transaction_leaf> leaves;
        leaves.reserve(claims.size());
        const auto domain =
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);
        for (auto & claim : claims) {
            llama_cache_transaction_leaf leaf;
            leaf.category = claim.category;
            leaf.domain = domain;
            leaf.attribution = {
                llama_cache_acct_attr_kind::artifact,
                -1,
                claim.artifact,
            };
            leaf.expected_logical = claim.bytes;
            leaf.reserve_resident = claim.allocation ? 0 : claim.bytes;
            leaf.stage_resident = claim.bytes;
            leaf.existing_allocation = claim.allocation;
            leaf.committed_op = &claim.operation;
            leaf.allocation_out = claim.allocation
                ? nullptr : &claim.allocation;
            leaves.push_back(leaf);
        }
        llama_cache_transaction_fault fault;
        fault.fail_after_commit = fail_after_commit;
        const auto transaction =
            llama_cache_execute_reservation_transaction(
                authority.ledger, config, leaves, fault);
        authority.admission_retries += transaction.serial_retries;
        authority.admission_rollbacks += transaction.rolled_back;
        if (transaction.status !=
                llama_cache_transaction_status::committed) {
            SRV_DBG(
                "CACHE_AUTHORITY shared payload transaction status=%s "
                "admission=%s failed_leaf=%zu pending_bytes=%" PRIu64
                " attempts=%u serial_retries=%" PRIu64 "\n",
                llama_cache_transaction_status_name(transaction.status),
                llama_cache_admission_status_name(
                    transaction.admission_status),
                transaction.failed_leaf, pending, transaction.attempts,
                transaction.serial_retries);
            return false;
        }
        size_t bound = 0;
        for (; bound < claims.size(); ++bound) {
            auto & claim = claims[bound];
            if (claim.buffer) {
                // Every independently retireable logical handle becomes
                // immutable after its operation commits. Existing physical
                // bindings still require this per-handle ownership mark.
                const bool already_owned =
                    claim.buffer->owns_accounting_binding(
                        &authority.ledger, claim.allocation.v);
                if (!claim.buffer->bind_accounting(
                        &authority.ledger, claim.allocation.v)) {
                    break;
                }
                claim.handle_newly_owned = !already_owned;
                claim.newly_bound = !claim.existing_binding;
            }
        }
        if (bound != claims.size()) {
            for (auto & claim : claims) {
                if (claim.handle_newly_owned) {
                    (void) claim.buffer->unbind_accounting(
                        &authority.ledger, claim.allocation.v,
                        claim.newly_bound);
                }
                if (claim.operation) {
                    (void) authority.ledger.release(claim.operation);
                }
            }
            authority.admission_rollbacks += claims.size();
            return false;
        }
        committed.reserve(claims.size());
        for (const auto & claim : claims) {
            committed.push_back(claim.operation);
        }
        return true;
    } catch (...) {
        for (auto & claim : claims) {
            if (claim.handle_newly_owned) {
                (void) claim.buffer->unbind_accounting(
                    &authority.ledger, claim.allocation.v,
                    claim.newly_bound);
            }
            if (claim.operation) {
                (void) authority.ledger.release(claim.operation);
                claim.operation = {};
            }
        }
        committed.clear();
        return false;
    }
}

} // namespace

bool server_cache_weighted_price_us(
        long double base_us,
        uint32_t weight_milli,
        uint64_t & out) noexcept {
    out = 0;
    const long double weighted = base_us * weight_milli /
        SERVER_CACHE_HOST_WEIGHT_SCALE;
    if (!std::isfinite((double) weighted) || weighted < 0.0L ||
        weighted > (long double) std::numeric_limits<long long>::max()) {
        return false;
    }
    out = uint64_t(std::llround(weighted));
    return true;
}

bool server_cache_retention_weight_milli(
        bool soft_leased,
        bool main_family,
        uint32_t additional_weight_milli,
        uint32_t & weight_milli) noexcept {
    weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    return (!soft_leased || server_cache_multiply_retention_weight(
                weight_milli, SERVER_CACHE_HOST_SOFT_LEASE_WEIGHT)) &&
           (!main_family || server_cache_multiply_retention_weight(
                weight_milli, SERVER_CACHE_HOST_MAIN_FAMILY_WEIGHT)) &&
           server_cache_multiply_retention_weight(
                weight_milli, additional_weight_milli);
}

bool server_cache_host_retention_price_us(
        const common_cache_plan_calib & calib,
        uint64_t bytes,
        bool soft_leased,
        bool main_family,
        uint32_t & weight_milli,
        uint64_t & price_us,
        uint32_t additional_weight_milli) noexcept {
    price_us = 0;
    if (!server_cache_retention_weight_milli(
            soft_leased, main_family, additional_weight_milli,
            weight_milli)) {
        return false;
    }
    double restore_us = 0.0;
    double workspace_us = 0.0;
    return common_cache_plan_restore_us(
               calib, bytes, restore_us, workspace_us) &&
           server_cache_weighted_price_us(
               (long double) restore_us + workspace_us,
               weight_milli, price_us);
}

server_cache_checkpoint_trade_plan server_cache_plan_checkpoint_thinning(
        const std::vector<server_cache_checkpoint_trade_input> & candidates,
        const common_cache_plan_calib * calib) noexcept {
    server_cache_checkpoint_trade_plan out;
    if (!calib || !std::isfinite(calib->replay_us_per_token) ||
        calib->replay_us_per_token < 0.0) {
        out.reason = common_cache_plan_destruction_reason::profile_unfitted;
        return out;
    }
    common_cache_plan_destruction_reason refusal =
        common_cache_plan_destruction_reason::recovery_unavailable;
    server_cache_checkpoint_protection protection =
        server_cache_checkpoint_protection::none;
    try {
        for (const auto & candidate : candidates) {
            if (candidate.artifact.v == 0 || !candidate.identity_known ||
                candidate.payload_bytes == 0 ||
                candidate.weight_milli == 0) {
                refusal =
                    common_cache_plan_destruction_reason::manifest_incomplete;
                continue;
            }
            if (candidate.seam_heuristic_protected ||
                candidate.mandatory_anchor) {
                refusal =
                    common_cache_plan_destruction_reason::mandatory_anchor;
                if (candidate.seam_heuristic_protected) {
                    protection =
                        server_cache_checkpoint_protection::seam_heuristic;
                } else if (protection !=
                        server_cache_checkpoint_protection::seam_heuristic) {
                    protection =
                        server_cache_checkpoint_protection::mandatory_anchor;
                }
                continue;
            }
            if (candidate.hard_leased) {
                refusal =
                    common_cache_plan_destruction_reason::hard_lease_blocked;
                if (protection ==
                        server_cache_checkpoint_protection::none) {
                    protection =
                        server_cache_checkpoint_protection::hard_lease;
                }
                continue;
            }
            if (!candidate.recovery_available ||
                candidate.recovery_ordinal == UINT32_MAX) {
                refusal =
                    common_cache_plan_destruction_reason::recovery_unavailable;
                continue;
            }
            double restore_us = 0.0;
            double workspace_us = 0.0;
            if (!common_cache_plan_restore_us(
                    *calib, candidate.payload_bytes,
                    restore_us, workspace_us)) {
                refusal =
                    common_cache_plan_destruction_reason::capacity_refused;
                continue;
            }
            const long double replay_us =
                (long double) candidate.replay_tokens *
                calib->replay_us_per_token;
            const long double base = replay_us + restore_us + workspace_us;
            uint64_t price = 0;
            if (!server_cache_weighted_price_us(
                    base, candidate.weight_milli, price)) {
                refusal =
                    common_cache_plan_destruction_reason::capacity_refused;
                continue;
            }
            const auto key = std::make_tuple(
                price, candidate.stable_id, candidate.ordinal);
            const auto best = std::make_tuple(
                out.price_us, out.stable_id, out.ordinal);
            if (!out.selected || key < best) {
                out.selected = true;
                out.ordinal = candidate.ordinal;
                out.recovery_ordinal = candidate.recovery_ordinal;
                out.price_us = price;
                out.stable_id = candidate.stable_id;
                out.weight_milli = candidate.weight_milli;
                out.protection =
                    server_cache_checkpoint_protection::none;
                out.reason = common_cache_plan_destruction_reason::none;
            }
        }
    } catch (...) {
        out = {};
        out.reason = common_cache_plan_destruction_reason::internal_fault;
    }
    if (!out.selected) {
        out.protection = protection;
        out.reason = refusal;
        switch (protection) {
            case server_cache_checkpoint_protection::seam_heuristic:
            case server_cache_checkpoint_protection::mandatory_anchor:
                out.reason =
                    common_cache_plan_destruction_reason::mandatory_anchor;
                break;
            case server_cache_checkpoint_protection::hard_lease:
                out.reason =
                    common_cache_plan_destruction_reason::hard_lease_blocked;
                break;
            case server_cache_checkpoint_protection::none:
            case server_cache_checkpoint_protection::_count:
                break;
        }
    }
    return out;
}

bool server_cache_checkpoint_bounded_replay(
        const common_prompt_checkpoint & recovery,
        const common_prompt_checkpoint & later,
        uint64_t max_replay_tokens) noexcept {
    return recovery.computation_frontier.valid() &&
           later.computation_frontier.valid() &&
           later.n_tokens >= recovery.n_tokens &&
           uint64_t(later.n_tokens - recovery.n_tokens) <= max_replay_tokens &&
           recovery.computation_frontier.sequence_epoch ==
               later.computation_frontier.sequence_epoch &&
           recovery.computation_frontier.execution_identity ==
               later.computation_frontier.execution_identity &&
           recovery.computation_frontier.adapter_config_identity ==
               later.computation_frontier.adapter_config_identity &&
           recovery.computation_frontier.media_content_identity ==
               later.computation_frontier.media_content_identity &&
           recovery.checkpoint_epoch == later.checkpoint_epoch &&
           recovery.checkpoint_epoch_swa == later.checkpoint_epoch_swa;
}

size_t server_cache_checkpoint_rebase_preserved_suffix(
        std::list<common_prompt_checkpoint> & checkpoints,
        const llama_memory_vbr_state_data & before,
        const llama_memory_vbr_state_data & after,
        llama_pos suffix_begin) noexcept {
    if (suffix_begin < 0) {
        return 0;
    }
    size_t rebased = 0;
    for (auto & checkpoint : checkpoints) {
        if (checkpoint.pos_max >= 0 && checkpoint.pos_max < suffix_begin &&
            common_prompt_checkpoint_lineage_matches(checkpoint, before)) {
            checkpoint.checkpoint_epoch     = after.checkpoint_epoch;
            checkpoint.checkpoint_epoch_swa = after.checkpoint_epoch_swa;
            rebased++;
        }
    }
    return rebased;
}

server_cache_checkpoint_floor_plan server_cache_plan_checkpoint_capacity_floor(
        const std::vector<server_cache_checkpoint_floor_input> & candidates) noexcept {
    server_cache_checkpoint_floor_plan out;
    uint32_t heuristic = UINT32_MAX;
    try {
        for (const auto & candidate : candidates) {
            if (candidate.recovery_pinned ||
                candidate.protection ==
                    server_cache_checkpoint_protection::mandatory_anchor ||
                candidate.protection ==
                    server_cache_checkpoint_protection::hard_lease) {
                if (candidate.protection ==
                        server_cache_checkpoint_protection::hard_lease) {
                    out.reason =
                        common_cache_plan_destruction_reason::hard_lease_blocked;
                }
                continue;
            }
            if (candidate.protection ==
                    server_cache_checkpoint_protection::seam_heuristic) {
                if (heuristic == UINT32_MAX) {
                    heuristic = candidate.ordinal;
                }
                continue;
            }
            out.selected = true;
            out.ordinal = candidate.ordinal;
            out.reason = common_cache_plan_destruction_reason::none;
            return out;
        }
        if (heuristic != UINT32_MAX) {
            out.selected = true;
            out.ordinal = heuristic;
            out.reason = common_cache_plan_destruction_reason::none;
        }
    } catch (...) {
        out = {};
        out.reason = common_cache_plan_destruction_reason::internal_fault;
    }
    return out;
}

bool server_cache_authority::sample_budget(
        llama_cache_budget_config & config,
        uint64_t pending_host_bytes) noexcept {
    try {
        config = {};
        config.devices = budget_devices;
        for (auto & input : config.devices) {
            size_t free = 0;
            size_t total = 0;
            auto * device = reinterpret_cast<ggml_backend_dev_t>(
                const_cast<void *>(input.backend_device));
            ggml_backend_dev_memory(device, &free, &total);
            input.physical_free  = uint64_t(free);
            input.physical_total = uint64_t(total);
            input.phys_state =
                total > 0 && free <= total
                    ? llama_cache_budget_capacity_state::known
                    : llama_cache_budget_capacity_state::unavailable;
        }

        config.host.pinned_cap = 0;
        config.host.pinned_state =
            llama_cache_budget_capacity_state::unbounded;
        config.host.total_state =
            llama_cache_budget_capacity_state::unavailable;
        config.global_cap_state =
            llama_cache_budget_capacity_state::unbounded;

        // PROPOSAL §9 requires pre-flip FIFO eviction/list order to remain legacy-identical.
        // Therefore the prompt cache's configured rotation limit is not an authority ceiling.
        // Price new host bytes against physical CPU-memory headroom instead. The detached entry
        // has already allocated pending_host_bytes, so add those bytes back before comparing the
        // canonical accounting `before` to the point-in-time headroom.
        ggml_backend_dev_t cpu =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        size_t free = 0;
        size_t total = 0;
        if (!cpu) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        ggml_backend_dev_memory(cpu, &free, &total);
        if (total == 0 || free > total) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }

        // Obtain the canonical host-domain `before` from the same closed category tables used by
        // fits(), rather than growing a second cache-byte classifier in the server.
        llama_cache_acct_snapshot snapshot = ledger.snapshot();
        llama_cache_budget_config probe = config;
        probe.host.pageable_state =
            llama_cache_budget_capacity_state::unbounded;
        llama_cache_budget_coordinator coordinator;
        const uint64_t serial = snapshot.serial;
        if (!coordinator.reset(std::move(snapshot), probe)) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        llama_cache_budget_plan baseline;
        baseline.accounting_serial = serial;
        const llama_cache_budget_result current = coordinator.fits(baseline);
        const auto host_domain =
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);
        const auto row = std::find_if(
            current.domains.begin(), current.domains.end(),
            [&](const llama_cache_budget_row & candidate) {
                return candidate.resource.domain == host_domain;
            });
        if (row == current.domains.end() ||
            row->before.state != llama_cache_acct_known::known) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }

        uint64_t host_before = 0;
        for (const auto & current_row : current.domains) {
            if (current_row.resource.residency !=
                    llama_cache_acct_residency::pageable_host &&
                current_row.resource.residency !=
                    llama_cache_acct_residency::pinned_host) {
                continue;
            }
            if (current_row.before.state !=
                    llama_cache_acct_known::known ||
                !add_checked(
                    host_before, current_row.before.value, host_before)) {
                config.host.pageable_state =
                    llama_cache_budget_capacity_state::unavailable;
                return true;
            }
        }

        uint64_t available = 0;
        uint64_t cap = 0;
        uint64_t total_cap = 0;
        if (!add_checked(uint64_t(free), pending_host_bytes, available) ||
            !add_checked(row->before.value, available, cap) ||
            !add_checked(host_before, available, total_cap)) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        config.host.pageable_cap = cap;
        config.host.pageable_state =
            llama_cache_budget_capacity_state::known;
        config.host.total_cap = total_cap;
        config.host.total_state =
            llama_cache_budget_capacity_state::known;
        return true;
    } catch (...) {
        config = {};
        return false;
    }
}

bool server_cache_authority::project_release(
        const llama_cache_acct_release_set_preview & release,
        std::vector<common_cache_plan_yield_domain> & out) noexcept {
    out.clear();
    try {
        llama_cache_budget_config config;
        if (!sample_budget(config)) {
            return false;
        }
        auto snapshot = ledger.snapshot();
        if (snapshot.serial != release.accounting_serial) {
            return false;
        }
        llama_cache_budget_coordinator coordinator;
        if (!coordinator.reset(std::move(snapshot), config)) {
            return false;
        }
        llama_cache_budget_plan plan;
        if (!server_cache_yield_release_plan(
                release, release.accounting_serial, plan)) {
            return false;
        }
        const auto fit = coordinator.fits(plan);
        if (fit.state != llama_cache_budget_fit_state::fits ||
            fit.accounting_serial != release.accounting_serial) {
            return false;
        }
        out.reserve(fit.domains.size());
        for (const auto & row : fit.domains) {
            if (std::none_of(
                    release.rows.begin(), release.rows.end(),
                    [&](const auto & released) {
                        return released.domain == row.resource.domain;
                    })) {
                continue;
            }
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(row, lowered)) {
                return false;
            }
            out.push_back(lowered);
        }
        return !out.empty();
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_authority::observe_release_domains(
        const std::vector<common_cache_plan_yield_domain> & projected,
        std::vector<common_cache_plan_yield_domain> & out) noexcept {
    out.clear();
    if (projected.empty()) {
        return true;
    }
    try {
        llama_cache_budget_config config;
        if (!sample_budget(config)) {
            return false;
        }
        auto snapshot = ledger.snapshot();
        const uint64_t serial = snapshot.serial;
        llama_cache_budget_coordinator coordinator;
        if (!coordinator.reset(std::move(snapshot), config)) {
            return false;
        }
        llama_cache_budget_plan plan;
        plan.accounting_serial = serial;
        const auto fit = coordinator.fits(plan);
        if (fit.accounting_serial != serial ||
            fit.state == llama_cache_budget_fit_state::unavailable) {
            return false;
        }
        out.reserve(projected.size());
        for (const auto & expected : projected) {
            const auto row = std::find_if(
                fit.domains.begin(), fit.domains.end(),
                [&](const auto & current) {
                    return current.resource.kind ==
                               llama_cache_budget_resource_kind::
                                   accounting_domain &&
                           current.resource.domain == expected.domain;
                });
            if (row == fit.domains.end()) {
                out.clear();
                return false;
            }
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(*row, lowered)) {
                out.clear();
                return false;
            }
            out.push_back(std::move(lowered));
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_authority::admit_host_entry(
        server_prompt_cache_state & entry) noexcept {
    if (!configured) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: substrate unavailable");
        return false;
    }
    std::vector<server_shared_payload_claim> claims;
    std::vector<llama_cache_acct_op_id> committed;
    try {
        const auto * fixed = entry.payload.fixed_state();
        if (!fixed || fixed->size() == 0 ||
            server_fault("acct_unavailable")) {
            throw std::runtime_error("payload unavailable");
        }
        claims.reserve(1 + entry.prompt.checkpoints.size() * 4);
        server_shared_payload_claim snapshot;
        snapshot.category =
            llama_cache_acct_category::full_snapshot_payload;
        snapshot.bytes = uint64_t(fixed->size());
        claims.push_back(snapshot);
        for (auto & checkpoint : entry.prompt.checkpoints) {
            if (!add_checkpoint_payload_claims(
                    ledger, checkpoint, claims)) {
                throw std::runtime_error("checkpoint binding unavailable");
            }
        }
    } catch (...) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: payload accounting unavailable");
        return false;
    }
    if (!execute_shared_payload_claims(
            *this, claims,
            server_fault("cache_lifecycle_after_commit"), committed)) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: shared payload transaction failed");
        return false;
    }
    entry.acct_ops = std::move(committed);
    entry.accounting_complete = true;
    admission_commits++;
    return true;
}

bool server_cache_authority::admit_live_checkpoints(
        std::vector<server_cache_live_checkpoint_admission> & batch) noexcept {
    const uint64_t refusal_count = std::max<size_t>(batch.size(), 1);
    const auto refuse = [&]() noexcept {
        for (auto & member : batch) {
            member.committed.clear();
        }
        admission_refusals += refusal_count;
        return false;
    };

    if (!configured || batch.empty()) {
        return refuse();
    }
    std::vector<server_shared_payload_claim> claims;
    std::vector<std::pair<size_t, size_t>> ranges;
    try {
        ranges.reserve(batch.size());
        for (auto & member : batch) {
            member.committed.clear();
            if (member.artifact.v == 0 || !member.checkpoint ||
                member.checkpoint->data_tgt.empty()) {
                return refuse();
            }
            std::vector<server_shared_payload_claim> member_claims;
            member_claims.reserve(4);
            if (!add_checkpoint_payload_claims(
                    ledger, *member.checkpoint, member_claims) ||
                member_claims.empty()) {
                return refuse();
            }
            const size_t first = claims.size();
            for (auto & claim : member_claims) {
                claim.artifact = member.artifact;
                claims.push_back(std::move(claim));
            }
            ranges.push_back({ first, claims.size() });
        }
    } catch (...) {
        return refuse();
    }
    std::vector<llama_cache_acct_op_id> committed;
    if (!execute_shared_payload_claims(
            *this, claims, false, committed)) {
        SRV_WRN(
            "%s\n",
            "CACHE_AUTHORITY checkpoint ownership refused: shared payload transaction failed");
        return refuse();
    }
    try {
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].committed.assign(
                committed.begin() + ranges[i].first,
                committed.begin() + ranges[i].second);
        }
    } catch (...) {
        for (auto & claim : claims) {
            if (claim.handle_newly_owned && claim.buffer) {
                (void) claim.buffer->unbind_accounting(
                    &ledger, claim.allocation.v,
                    claim.newly_bound);
            }
        }
        for (const auto op : committed) {
            (void) ledger.release(op);
        }
        admission_rollbacks += committed.size();
        return refuse();
    }
    // Preserve the established per-checkpoint admission counter semantics
    // even though the accounting terminal is now one batch transaction.
    admission_commits += batch.size();
    return true;
}

bool server_cache_authority::admit_live_checkpoint(
        llama_cache_acct_artifact_id artifact,
        const common_prompt_checkpoint & checkpoint,
        std::vector<llama_cache_acct_op_id> & committed) noexcept {
    committed.clear();
    try {
        std::vector<server_cache_live_checkpoint_admission> batch(1);
        batch[0].artifact = artifact;
        batch[0].checkpoint = &checkpoint;
        if (!admit_live_checkpoints(batch)) {
            return false;
        }
        committed = std::move(batch[0].committed);
        return !committed.empty();
    } catch (...) {
        admission_refusals++;
        return false;
    }
}

void server_cache_authority::observe_host_destruction(
        common_cache_plan_destruction_receipt receipt,
        bool observe_classification) noexcept {
    destruction_counters.observe(
        common_cache_plan_selection::none, receipt, observe_classification);
    destruction_counters.last_receipt = std::move(receipt);
    destruction_counters.has_receipt = true;
}
