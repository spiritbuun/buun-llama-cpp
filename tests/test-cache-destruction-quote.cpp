#include "server-cache-destruction-quote.h"
#include "server-cache-retention-proof.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

static common_cache_plan_record record_with_cold_candidates() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * live = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    live->accept();
    live->f_keep = 0.25;
    live->f_keep_known = true;
    auto * cold_a = rec.find_or_add(
        common_cache_plan_provider::cold_replay, 10,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    cold_a->accept();
    auto * cold_b = rec.find_or_add(
        common_cache_plan_provider::cold_replay, 11,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    cold_b->accept();
    return rec;
}

static server_cache_destruction_artifact artifact(uint64_t id, uint64_t op) {
    server_cache_destruction_artifact out;
    out.candidate.artifact_id = { id };
    out.kind = common_retention_artifact_kind::live_slot;
    out.candidate.record.kind = out.kind;
    auto turns = std::make_shared<common_retention_turn_table>();
    turns->source = common_retention_source_state::known;
    turns->token_count = 1;
    turns->boundaries = { { 0, 0, 1 } };
    out.candidate.record.turns = std::move(turns);
    out.candidate.record.stamp.state =
        common_retention_score_state::known;
    out.candidate.record.stamp.stable_id = id;
    out.candidate.record.stamp.lineage_id = id;
    out.candidate.record.stamp.recency_ordinal = id;
    out.candidate.record.stamp.coverage_tokens = 1;
    out.owner_slot = 0;
    out.pool = common_retention_pool::attention;
    out.candidate.identity_known = true;
    out.candidate.availability =
        server_retention_candidate_availability::available;
    out.candidate.lease.state = server_cache_lease_eval_state::known;
    out.candidate.lease.cls = server_cache_lease_class::none;
    out.candidate.lease.eligibility =
        server_cache_lease_eligibility::eligible;
    out.candidate.release_ops = { llama_cache_acct_op_id{op} };
    return out;
}

static server_cache_destruction_preview_callback preview(uint64_t serial) {
    return [serial](const auto & ops, uint64_t expected, auto & out) {
        out = {};
        if (expected != serial || ops.empty()) {
            return false;
        }
        out.accounting_serial = serial;
        out.rows.push_back({ HOST, 64, 64 });
        return true;
    };
}

static server_cache_destruction_projection_callback project() {
    return [](const auto & released, auto & out) {
        out.clear();
        for (const auto & row : released.rows) {
            llama_cache_budget_row fit;
            fit.resource.kind =
                llama_cache_budget_resource_kind::accounting_domain;
            fit.resource.domain = row.domain;
            fit.current_resident = llama_cache_acct_value::measured(128);
            fit.before = llama_cache_acct_value::measured(128);
            fit.released =
                llama_cache_acct_value::measured(row.resident_allocated);
            fit.reserved = llama_cache_acct_value::measured(0);
            fit.after = llama_cache_acct_value::measured(64);
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(fit, lowered)) {
                return false;
            }
            out.push_back(std::move(lowered));
        }
        return true;
    };
}

static void test_complete_memoized_and_permutation() {
    auto rec = record_with_cold_candidates();
    std::vector<server_cache_destruction_artifact> artifacts = {
        artifact(2, 22), artifact(1, 11),
    };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, artifacts, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() == 2);
    CHECK(rec.destruction_quotes[0].receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::none);
    CHECK(rec.destruction_quotes[0].receipt.selected_attention.size() == 2);
    CHECK(counters.quote_memo_misses == 1);
    CHECK(counters.quote_memo_hits == 1);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    const size_t cls = size_t(common_cache_plan_destruction_class::slot_drop);
    CHECK(counters.quoted[tier][cls] == 2);
    CHECK(counters.lease_verdict[tier][size_t(
              common_cache_plan_destruction_lease_verdict::unleased)] == 2);
    CHECK(!counters.has_receipt);
    const auto digest = rec.destruction_quotes[0].receipt.manifest_digest;
    const auto effect = rec.destruction_quotes[0].receipt.union_effect_digest;
    const auto domains = rec.destruction_quotes[0].projected_domains;
    CHECK(server_cache_destruction_effect_matches(
        rec.destruction_quotes[0].receipt, effect, domains, domains));

    std::reverse(artifacts.begin(), artifacts.end());
    common_cache_plan_destruction_counters permuted_counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, artifacts, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 2 }, permuted_counters));
    CHECK(rec.destruction_quotes[0].receipt.manifest_digest == digest);

    auto changed = common_cache_plan_destruction_effect_digest::from_sha256(
        std::array<uint8_t, 32>{ 1 });
    CHECK(!server_cache_destruction_effect_matches(
        rec.destruction_quotes[0].receipt, changed, domains, domains));
    CHECK(server_cache_destruction_effect_recheck(
        rec.destruction_quotes[0].receipt, changed, domains, domains) ==
        common_cache_plan_destruction_reason::effect_drift);
    auto later = rec.destruction_quotes[0].receipt;
    later.quote_accounting_serial = 99;
    CHECK(server_cache_destruction_effect_matches(
        later, effect, domains, domains));
    CHECK(server_cache_destruction_effect_recheck(
        later, effect, domains, domains) ==
        common_cache_plan_destruction_reason::none);
    auto gauge_drift = domains;
    gauge_drift[0].current_resident_bytes =
        llama_cache_acct_value::measured(999);
    gauge_drift[0].fit_before_bytes =
        llama_cache_acct_value::measured(999);
    gauge_drift[0].projected_after_bytes =
        llama_cache_acct_value::measured(935);
    CHECK(server_cache_destruction_effect_recheck(
        later, effect, domains, gauge_drift) ==
        common_cache_plan_destruction_reason::none);
    gauge_drift[0].projected_release_bytes =
        llama_cache_acct_value::measured(63);
    CHECK(server_cache_destruction_effect_recheck(
        later, effect, domains, gauge_drift) ==
        common_cache_plan_destruction_reason::effect_drift);
}

static void test_read_only_zero_sequence_quote() {
    common_cache_plan_destruction_counters production_counters;
    production_counters.quote_samples = 41;
    production_counters.quote_duration_us_total = 99;
    auto ordinary = record_with_cold_candidates();
    CHECK(!server_cache_destruction_quote_all(
        ordinary, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 0 },
        production_counters));
    CHECK(ordinary.destruction.state ==
          common_cache_plan_destruction_state::failed);
    CHECK(ordinary.destruction.reason ==
          common_cache_plan_destruction_reason::internal_fault);

    auto rec = record_with_cold_candidates();
    const auto production_before = production_counters;
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 0, 0,
          true },
        counters));
    CHECK(!rec.destruction_quotes.empty());
    CHECK(rec.destruction_quotes.front().receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(rec.destruction_quotes.front().receipt.admission_sequence == 0);
    CHECK(std::memcmp(
              &production_counters, &production_before,
              sizeof(production_counters)) == 0);

    llama_cache_acct_ledger ledger;
    auto prepared = server_cache_prepare_release_set(
        rec.destruction_quotes.front(), { artifact(1, 11) }, ledger,
        ledger.snapshot().serial, project(), {});
    CHECK(prepared.status ==
          server_cache_prepare_release_status::invalid_quote);
    CHECK(!prepared.capability.ready());
}

static void test_fail_closed_matrix() {
    struct cell {
        common_cache_plan_destruction_reason reason;
        void (*mutate)(server_cache_destruction_artifact &);
    };
    const cell cells[] = {
        { common_cache_plan_destruction_reason::identity_unavailable,
          [](auto & a) { a.candidate.identity_known = false; } },
        { common_cache_plan_destruction_reason::manifest_incomplete,
          [](auto & a) { a.candidate.availability = server_retention_candidate_availability::backing_missing_or_stale; } },
        { common_cache_plan_destruction_reason::manifest_incomplete,
          [](auto & a) { a.candidate.record.stamp.state = common_retention_score_state::unavailable; } },
        { common_cache_plan_destruction_reason::mandatory_anchor,
          [](auto & a) { a.mandatory_anchor = true; } },
        { common_cache_plan_destruction_reason::mandatory_anchor,
          [](auto & a) { a.candidate.record.stamp.mandatory_anchor = true; } },
        { common_cache_plan_destruction_reason::lease_unavailable,
          [](auto & a) { a.candidate.lease.state = server_cache_lease_eval_state::unavailable; } },
        { common_cache_plan_destruction_reason::hard_lease_blocked,
          [](auto & a) { a.candidate.lease.cls = server_cache_lease_class::hard; a.candidate.lease.eligibility = server_cache_lease_eligibility::hard_blocked; } },
        { common_cache_plan_destruction_reason::hard_lease_blocked,
          [](auto & a) { a.candidate.lease.eligibility = server_cache_lease_eligibility::hard_blocked; } },
        { common_cache_plan_destruction_reason::release_evidence_unavailable,
          [](auto & a) { a.candidate.release_ops.clear(); } },
    };
    for (const auto & cell : cells) {
        auto rec = record_with_cold_candidates();
        auto a = artifact(1, 11);
        cell.mutate(a);
        common_cache_plan_destruction_counters counters;
        CHECK(server_cache_destruction_quote_all(
            rec, 0, { a }, 17, preview(17), project(),
            { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
        CHECK(!rec.destruction_quotes.empty());
        CHECK(rec.destruction_quotes[0].receipt.state ==
              common_cache_plan_destruction_state::refused);
        CHECK(rec.destruction_quotes[0].receipt.reason == cell.reason);
    }

    auto rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17,
        [](const auto &, uint64_t, auto &) { return false; },
        project(), { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::accounting_unavailable);

    rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17),
        [](const auto &, auto &) { return false; },
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::capacity_refused);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    const size_t slot_drop = size_t(
        common_cache_plan_destruction_class::slot_drop);
    CHECK(counters.quoted[tier][slot_drop] == 0);
    CHECK(counters.refused[tier][size_t(
              common_cache_plan_destruction_reason::capacity_refused)] == 2);
    CHECK(rec.destruction_quotes[0].receipt.selected_attention.empty());
    CHECK(rec.destruction_quotes[0].receipt.selected_recurrent.empty());

    rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters whole_counters;
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, {}, 17, preview(17), project(),
        { false, common_cache_plan_recovery_citation::unavailable, 1 },
        whole_counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::lifecycle_disabled);
    CHECK(whole_counters.refused[size_t(common_cache_plan_selection::lru)]
          [size_t(common_cache_plan_destruction_reason::lifecycle_disabled)] == 1);

    std::vector<server_cache_destruction_artifact> overflow(
        SERVER_CACHE_YIELD_MAX_CANDIDATES + 1, artifact(1, 11));
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, overflow, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 },
        whole_counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
    CHECK(whole_counters.refused[size_t(common_cache_plan_selection::lru)]
          [size_t(common_cache_plan_destruction_reason::manifest_incomplete)] == 1);
    rec = record_with_cold_candidates();
    rec.derived_plans_incomplete = true;
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
}

static void test_refusal_mapping_and_selection() {
    auto rec = record_with_cold_candidates();
    const auto lifecycle_effects =
        server_cache_plan_nonconsuming_host_effects(true);
    CHECK(server_cache_destruction_has_effect(rec, 0));
    CHECK(server_cache_destruction_has_effect(
        rec, 0, lifecycle_effects));
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::same_target_cold_replacement));
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            rec, 1, 0, lifecycle_effects),
        common_cache_plan_destruction_effect::same_target_cold_replacement));
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::unavailable, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    auto prospective_same = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        prospective_same, 0, { artifact(1, 11) }, 17,
        preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 2 },
        counters));
    CHECK(prospective_same.destruction_quotes[0].receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 3 }, counters));
    rec.shadow_choice = 1;
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.plan_candidate == 1);
    CHECK(common_cache_plan_destruction_effect_has(
        rec.destruction.effects,
        common_cache_plan_destruction_effect::same_target_cold_replacement));

    server_cache_yield_result yield;
    yield.status = server_cache_yield_status::fits;
    yield.yield_policy_version = 1;
    yield.accounting_serial = 17;
    yield.selected[size_t(common_retention_pool::attention)] = { { 1 } };
    yield.projected_fit.state = llama_cache_budget_fit_state::fits;
    yield.projected_fit.accounting_serial = 17;
    llama_cache_budget_row projected;
    projected.resource.kind =
        llama_cache_budget_resource_kind::accounting_domain;
    projected.resource.domain = HOST;
    projected.current_resident = llama_cache_acct_value::measured(128);
    projected.before = llama_cache_acct_value::measured(128);
    projected.released = llama_cache_acct_value::measured(64);
    projected.reserved = llama_cache_acct_value::measured(0);
    projected.after = llama_cache_acct_value::measured(64);
    yield.projected_fit.domains.push_back(projected);
    rec.acct.serial = 23;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::matched);
    CHECK(rec.yield.actual_state ==
          common_cache_plan_yield_actual_state::not_observed);
    CHECK(rec.yield.projected_domains.size() == 1);
    CHECK(rec.yield.accounting_serial == rec.acct.serial);

    yield.selected[size_t(common_retention_pool::attention)] = { { 99 } };
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::differed);

    yield.selected[size_t(common_retention_pool::attention)] = { { 1 } };
    yield.projected_fit.domains[0].reserved =
        llama_cache_acct_value::measured(1);
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::differed);
    yield.projected_fit.domains[0].reserved =
        llama_cache_acct_value::measured(0);

    yield.status = server_cache_yield_status::insufficient_yield;
    rec.yield.unsupported = { { 77 } };
    rec.yield.projected_domains.clear();
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::ds6_insufficient_yield);
    CHECK(rec.yield.projected_domains.size() == 1);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 77 } }));
    CHECK(rec.yield.actual_state ==
          common_cache_plan_yield_actual_state::not_observed);

    yield.status = server_cache_yield_status::unsupported_required;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::
              ds6_unsupported_required);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 77 } }));
    yield.status = server_cache_yield_status::unavailable;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::ds6_unavailable);

    rec.selection = common_cache_plan_selection::similarity;
    rec.inventory[1].target_slot_id = 1;
    rec.inventory[1].provider = common_cache_plan_provider::live_slot;
    rec.inventory[1].origin_tier = common_cache_plan_selection::similarity;
    rec.inventory[1].f_keep_known = true;
    rec.inventory[1].f_keep = 0.5;
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::destructive_similarity_retarget));
    auto cross_artifact = artifact(3, 33);
    cross_artifact.owner_slot = 1;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { cross_artifact }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 3 },
        counters));
    CHECK(rec.destruction_quotes[0].receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(rec.destruction_quotes[0].receipt.recovery_citation ==
          common_cache_plan_recovery_citation::prospective);
    rec.selection = common_cache_plan_selection::route_home;
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::cross_target_displacement));

    common_cache_plan_record host_rec;
    host_rec.selection = common_cache_plan_selection::lru;
    auto * legacy_host = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 10,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    auto * planned_host = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 20,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(legacy_host != nullptr);
    CHECK(planned_host != nullptr);
    legacy_host->accept();
    planned_host->accept();
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(host_rec, 1, 0),
        common_cache_plan_destruction_effect::different_host_source_consumption));

    common_cache_plan_record nondestructive;
    nondestructive.selection = common_cache_plan_selection::lru;
    auto * only = nondestructive.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(only != nullptr);
    only->accept();
    CHECK(!server_cache_destruction_has_effect(nondestructive, 0));
}

static void test_plural_effect_union_and_counters() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * legacy = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    auto * planned = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 20,
        COMMON_CACHE_PLAN_PHASE_LRU, 1,
        common_cache_plan_selection::lru);
    CHECK(legacy != nullptr);
    CHECK(planned != nullptr);
    legacy->accept();
    planned->accept();

    const auto effects = server_cache_destruction_effects_for(rec, 1, 0);
    CHECK(common_cache_plan_destruction_effect_has(
        effects,
        common_cache_plan_destruction_effect::cross_target_displacement));
    CHECK(common_cache_plan_destruction_effect_has(
        effects,
        common_cache_plan_destruction_effect::different_host_source_consumption));

    auto live = artifact(1, 11);
    live.owner_slot = 1;
    auto host = artifact(2, 22);
    host.kind = common_retention_artifact_kind::host_entry;
    host.owner_slot = -1;
    host.host_source_id = 20;
    uint64_t preview_ops = 0;
    const auto preview_union = [&](const auto & ops, uint64_t serial, auto & out) {
        preview_ops = ops.size();
        return preview(17)(ops, serial, out);
    };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { live, host }, 17, preview_union, project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() == 1);
    const auto & receipt = rec.destruction_quotes[0].receipt;
    CHECK(receipt.state == common_cache_plan_destruction_state::quoted);
    CHECK(preview_ops == 2);
    CHECK(receipt.selected_attention.size() == 2);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    CHECK(counters.quoted[tier][size_t(
              common_cache_plan_destruction_class::slot_drop)] == 1);
    CHECK(counters.quoted[tier][size_t(
              common_cache_plan_destruction_class::host_artifact_drop)] == 1);

    // A displacement artifact cannot hide absent coverage for the independent
    // host-consumption effect: the exact union is incomplete, not merely the
    // available displacement subset.
    common_cache_plan_destruction_counters partial_counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { live }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 2 },
        partial_counters));
    CHECK(rec.destruction_quotes.size() == 1);
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
}

static void test_refused_projection_and_selection_failure() {
    auto rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17),
        [](const auto &, auto &) { return false; },
        { true, common_cache_plan_recovery_citation::resolved, 9 }, counters));
    rec.shadow_choice = 1;
    rec.destruction.quote_duration_us = 41;
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.state == common_cache_plan_destruction_state::refused);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::capacity_refused);
    CHECK(rec.destruction.admission_sequence == 9);
    CHECK(rec.destruction.quote_duration_us == 41);

    rec.yield.status = common_cache_plan_yield_status::insufficient_yield;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::unavailable;
    rec.yield.accounting_serial = 55;
    rec.yield.unsupported = { { 7 } };
    server_cache_yield_result yield;
    yield.status = server_cache_yield_status::fits;
    yield.projected_fit.state = llama_cache_budget_fit_state::fits;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::not_compared);
    CHECK(rec.yield.status ==
          common_cache_plan_yield_status::insufficient_yield);
    CHECK(rec.yield.plan_state ==
          common_cache_plan_yield_plan_state::unavailable);
    CHECK(rec.yield.accounting_serial == 55);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 7 } }));

    rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 10 }, counters));
    rec.shadow_choice = -1;
    rec.destruction.quote_duration_us = 42;
    const auto before = counters.refused[size_t(rec.selection)][size_t(
        common_cache_plan_destruction_reason::internal_fault)];
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.state == common_cache_plan_destruction_state::failed);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::internal_fault);
    CHECK(rec.destruction.admission_sequence == 10);
    CHECK(rec.destruction.quote_duration_us == 42);
    CHECK(counters.refused[size_t(rec.selection)][size_t(
              common_cache_plan_destruction_reason::internal_fault)] == before + 1);

    rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters selected_counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 11 },
        selected_counters));
    CHECK(!rec.destruction_quotes.empty());
    rec.shadow_choice = 0;
    rec.destruction.quote_duration_us = 43;
    server_cache_destruction_select_quote(rec, selected_counters);
    CHECK(rec.destruction.state ==
          common_cache_plan_destruction_state::not_required);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::none);
    CHECK(rec.destruction.quote_duration_us == 43);
    CHECK(rec.destruction.admission_sequence == 11);
    CHECK(selected_counters.refused[size_t(rec.selection)][size_t(
              common_cache_plan_destruction_reason::internal_fault)] == 0);
}

static void test_max_inventory_memoizes_one_manifest() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * legacy = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(legacy != nullptr);
    legacy->accept();
    for (size_t i = 1; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * candidate = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i),
            COMMON_CACHE_PLAN_PHASE_LRU, 0,
            common_cache_plan_selection::lru);
        CHECK(candidate != nullptr);
        candidate->accept();
    }
    CHECK(rec.n_inventory == COMMON_CACHE_PLAN_MAX_CANDIDATES);
    CHECK(!rec.inventory_saturated());

    uint64_t preview_calls = 0;
    uint64_t project_calls = 0;
    const server_cache_destruction_preview_callback preview_once =
        [&](const auto & ops, uint64_t serial, auto & out) {
            preview_calls++;
            out = {};
            out.accounting_serial = serial;
            CHECK(ops.size() == 1);
            out.rows.push_back({ HOST, 64, 64 });
            return true;
        };
    const server_cache_destruction_projection_callback project_once =
        [&](const auto & released, auto & out) {
            project_calls++;
            return project()(released, out);
        };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17,
        preview_once, project_once,
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() ==
          COMMON_CACHE_PLAN_MAX_CANDIDATES - 1);
    CHECK(preview_calls == 1);
    CHECK(project_calls == 1);
    CHECK(counters.quote_memo_misses == 1);
    CHECK(counters.quote_memo_hits ==
          COMMON_CACHE_PLAN_MAX_CANDIDATES - 2);
}

static llama_cache_acct_op_id committed_release_op(
        llama_cache_acct_ledger & ledger,
        uint64_t bytes) {
    const llama_cache_acct_completeness_requirement requirement = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, HOST, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(
        HOST, llama_cache_acct_producer::host_cache));
    const auto op = ledger.reserve(
        llama_cache_acct_category::full_snapshot_payload,
        HOST, {}, bytes, bytes);
    const auto alloc = ledger.new_alloc();
    CHECK(bool(op));
    CHECK(bool(alloc));
    CHECK(ledger.stage(op, alloc, bytes));
    CHECK(ledger.commit(op, bytes));
    return op;
}

static void release_pin(void * context) noexcept {
    ++*static_cast<uint64_t *>(context);
}

static void test_prepared_release_capability() {
    CHECK(server_cache_destruction_recovery_source_digest(
              { 7 }, { { 3 }, { 1 }, { 2 } }) ==
          server_cache_destruction_recovery_source_digest(
              { 7 }, { { 2 }, { 3 }, { 1 } }));
    llama_cache_acct_ledger ledger;
    const auto op = committed_release_op(ledger, 64);
    const auto quoted_snapshot = ledger.snapshot();
    llama_cache_acct_release_set_preview released;
    CHECK(ledger.preview_release_set(
        { op }, quoted_snapshot.serial, released));

    common_cache_plan_destruction_quote quote;
    quote.receipt.state = common_cache_plan_destruction_state::quoted;
    quote.receipt.admission_sequence = 7;
    quote.receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::cross_target_displacement);
    quote.receipt.selected_attention = { { 1 } };
    auto current = artifact(1, op.v);
    quote.receipt.manifest_digest = [](
            const server_cache_destruction_artifact & current) {
        common_cache_plan_record rec = record_with_cold_candidates();
        common_cache_plan_destruction_counters counters;
        CHECK(server_cache_destruction_quote_all(
            rec, 0, { current }, 17, preview(17), project(),
            { true, common_cache_plan_recovery_citation::resolved, 99 },
            counters));
        return rec.destruction_quotes[0].receipt.manifest_digest;
    }(current);
    quote.receipt.union_effect_digest =
        server_cache_destruction_union_effect_digest(
            { op }, released);
    CHECK(project()(released, quote.projected_domains));

    uint64_t releases = 0;
    auto binding_pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, { op });
    CHECK(binding_pin.binds_exact({ 99 }, { op }));
    CHECK(!binding_pin.binds_exact({ 98 }, { op }));
    CHECK(!binding_pin.binds_exact({ 99 }, {}));
    binding_pin = {};
    CHECK(releases == 1);
    releases = 0;

    auto missing_host_pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, {});
    auto missing_host = server_cache_prepare_release_set(
        quote, {}, ledger, quoted_snapshot.serial, project(),
        std::move(missing_host_pin));
    CHECK(missing_host.status ==
          server_cache_prepare_release_status::invalid_quote);
    CHECK(missing_host.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
    missing_host_pin = {};
    CHECK(releases == 1);
    releases = 0;

    auto pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, {});
    CHECK(pin.valid());

    // Unrelated serial drift is evidence, not invalidation: fresh reprepare
    // proves the operation union/effect remained identical.
    ledger.gauge_set(
        llama_cache_acct_category::checkpoint_state_payload,
        HOST, llama_cache_acct_measure::logical_payload, 0);
    const auto fresh = ledger.snapshot();
    auto result = server_cache_prepare_release_set(
        quote, { current }, ledger, fresh.serial, project(), std::move(pin));
    CHECK(result.status == server_cache_prepare_release_status::prepared);
    CHECK(result.reason == common_cache_plan_destruction_reason::none);
    CHECK(result.capability.ready());
    CHECK(result.capability.accounting_serial() == fresh.serial);

    server_cache_recovery_pin retained;
    CHECK(result.capability.commit(retained) ==
          common_cache_plan_destruction_reason::none);
    CHECK(retained.valid());
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(releases == 0);
    retained = {};
    CHECK(releases == 1);

    // A stale caller sample is retryable serial drift, not effect_drift.
    llama_cache_acct_ledger stale_ledger;
    const auto stale_op = committed_release_op(stale_ledger, 64);
    const auto stale_serial = stale_ledger.snapshot().serial;
    CHECK(stale_ledger.preview_release_set(
        { stale_op }, stale_serial, released));
    current.candidate.release_ops = { stale_op };
    quote.receipt.union_effect_digest =
        server_cache_destruction_union_effect_digest({ stale_op }, released);
    CHECK(project()(released, quote.projected_domains));
    stale_ledger.gauge_set(
        llama_cache_acct_category::checkpoint_state_payload,
        HOST, llama_cache_acct_measure::reserved, 1);
    CHECK(stale_ledger.snapshot().serial != stale_serial);
    auto stale_pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, {});
    auto serial_conflict = server_cache_prepare_release_set(
        quote, { current }, stale_ledger, stale_serial, project(),
        std::move(stale_pin));
    CHECK(serial_conflict.status ==
          server_cache_prepare_release_status::serial_conflict);
    CHECK(serial_conflict.reason ==
          common_cache_plan_destruction_reason::accounting_unavailable);
    CHECK(stale_ledger.snapshot().live_ops == 1);
    CHECK(stale_ledger.release(stale_op));

    // The recovery source is a fourth conjunct: overlap with either victims
    // or release ops fails before a capability is minted.
    llama_cache_acct_ledger second;
    const auto op2 = committed_release_op(second, 64);
    const auto serial2 = second.snapshot().serial;
    CHECK(second.preview_release_set({ op2 }, serial2, released));
    current.candidate.release_ops = { op2 };
    quote.receipt.union_effect_digest =
        server_cache_destruction_union_effect_digest({ op2 }, released);
    CHECK(project()(released, quote.projected_domains));
    auto overlapping = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 1 } }, {});
    // Duplicate-pair guard: selecting both the victim and its cited
    // survivor in one ladder union must fail the fourth conjunct before any
    // physical mutation. The release-set planner may otherwise select both members.
    CHECK(!overlapping.disjoint({ { 1 }, { 2 } }, { op2 }));
    auto refused = server_cache_prepare_release_set(
        quote, { current }, second, serial2, project(), std::move(overlapping));
    CHECK(refused.status ==
          server_cache_prepare_release_status::recovery_unavailable);
    CHECK(refused.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    CHECK(second.snapshot().live_ops == 1);

    auto drifted_quote = quote;
    drifted_quote.receipt.union_effect_digest =
        common_cache_plan_destruction_effect_digest::from_sha256(
            std::array<uint8_t, 32>{ 1 });
    auto clean_pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, {});
    auto effect_drift = server_cache_prepare_release_set(
        drifted_quote, { current }, second, serial2, project(),
        std::move(clean_pin));
    CHECK(effect_drift.status ==
          server_cache_prepare_release_status::effect_drift);
    CHECK(effect_drift.reason ==
          common_cache_plan_destruction_reason::effect_drift);
    auto hard = current;
    hard.candidate.lease.eligibility =
        server_cache_lease_eligibility::hard_blocked;
    auto hard_pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, {});
    auto hard_refused = server_cache_prepare_release_set(
        quote, { hard }, second, serial2, project(),
        std::move(hard_pin));
    CHECK(hard_refused.reason ==
          common_cache_plan_destruction_reason::hard_lease_blocked);
    CHECK(second.release(op2));
}

static void test_fixed_pool_known_zero_release() {
    llama_cache_acct_ledger ledger;
    const auto warmup = committed_release_op(ledger, 1);
    CHECK(ledger.release(warmup));
    const auto serial = ledger.snapshot().serial;
    auto live = artifact(1, 11);
    live.candidate.release_ops.clear();
    live.fixed_pool_logical_ownership = true;

    auto rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters counters;
    const server_cache_destruction_preview_callback preview_zero =
        [serial](const auto & ops, uint64_t expected, auto & out) {
            out = {};
            if (!ops.empty() || expected != serial) {
                return false;
            }
            out.accounting_serial = serial;
            return true;
        };
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { live }, serial, preview_zero, project(),
        { true, common_cache_plan_recovery_citation::prospective, 1 },
        counters));
    CHECK(rec.destruction_quotes.size() == 2);
    auto quote = rec.destruction_quotes.front();
    CHECK(quote.receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(quote.projected_domains.empty());
    CHECK(quote.receipt.union_effect_digest.valid());

    uint64_t releases = 0;
    auto pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 99 } }, { { 77 } });
    auto prepared = server_cache_prepare_release_set(
        quote, { live }, ledger, serial, project(), std::move(pin));
    CHECK(prepared.status ==
          server_cache_prepare_release_status::prepared);
    server_cache_recovery_pin retained;
    CHECK(prepared.capability.commit(retained) ==
          common_cache_plan_destruction_reason::none);
    CHECK(ledger.snapshot().serial == serial);
    retained = {};
    CHECK(releases == 1);

    // Missing release evidence remains fail-closed unless the artifact is
    // explicitly the fixed-pool logical owner.
    live.fixed_pool_logical_ownership = false;
    rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { live }, serial, preview_zero, project(),
        { true, common_cache_plan_recovery_citation::prospective, 2 },
        counters));
    CHECK(rec.destruction_quotes.front().receipt.reason ==
          common_cache_plan_destruction_reason::
              release_evidence_unavailable);
}

static void test_retention_fallback_proof_lifetime() {
    uint64_t releases = 0;
    auto pin = server_cache_recovery_pin::acquire(
        &releases, release_pin, { { 301 } }, { { 302 } });
    auto proof = server_cache_retention_fallback_proof_for_test(
        std::move(pin));
    CHECK(proof.available());
    CHECK(releases == 0);
    auto moved = std::move(proof);
    CHECK(!proof.available());
    CHECK(moved.available());
    moved = {};
    CHECK(releases == 1);

    auto invalid = server_cache_retention_fallback_proof_for_test({});
    CHECK(!invalid.available());
    CHECK(invalid.state() == server_cache_lease_fallback_state::invalid);
}

int main() {
    test_complete_memoized_and_permutation();
    test_read_only_zero_sequence_quote();
    test_fail_closed_matrix();
    test_refusal_mapping_and_selection();
    test_plural_effect_union_and_counters();
    test_refused_projection_and_selection_failure();
    test_max_inventory_memoizes_one_manifest();
    test_prepared_release_capability();
    test_fixed_pool_known_zero_release();
    test_retention_fallback_proof_lifetime();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("cache destruction quote tests passed");
    return 0;
}
